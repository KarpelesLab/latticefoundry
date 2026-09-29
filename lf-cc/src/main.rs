//! The `lf-cc` driver: compile C files and link them into a native x86-64
//! executable, with no external compiler, assembler or linker involved.
//!
//! `lf-cc [options] <inputs...>` where inputs are `.c` sources (compiled),
//! `.s` assembly (assembled by `rsasm`), and `.o`/`.a`/`.so` files or
//! `-l<lib>` libraries (handed to the linker in command-line order).
//!
//! * `-c` compiles each source to an ELF relocatable object (`foo.c` → `foo.o`).
//!   A source's file-scope `asm(...)` is assembled by `rsasm` and merged into
//!   that one object by a relocatable (`-r`) `qld` link.
//! * `-S` / `--emit-lf` dumps the lowered `.lf` IR instead.
//! * Otherwise the inputs are linked. By default that is a **hosted** link:
//!   the host C runtime is located with [`HostCrt::discover`] and our own
//!   GNU-ld-compatible linker `qld` produces a dynamically linked executable
//!   against the host libc (`link::gnu::host_c_link_args` + `link_gnu`).
//! * `-nostdlib` selects the **freestanding** link instead: no libc and no
//!   host startup files, a static executable whose `_start` (LatticeFoundry's
//!   own crt0) calls `main` and exits with its result. When every input is a
//!   C source (without file-scope asm) this is done entirely in memory by the
//!   framework's own linker core; otherwise `qld` links statically. This is also the
//!   fallback when no host C runtime is found.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use latticefoundry::ir::text;
use latticefoundry::link;
use latticefoundry::link::gnu::{HostCrt, host_c_link_args, link_gnu};
use latticefoundry::mc::asm::{AsmOptions, AsmSource, assemble, assemble_file};
use latticefoundry::mc::elf;
use latticefoundry::mc::object::ObjectModule;
use latticefoundry::support::diagnostics::{Diagnostic, Severity};
use latticefoundry::target::TargetArch;
use latticefoundry::transform::pipeline::OptLevel;

use lf_cc::{BuildError, CStd, MacroOp, PpOptions};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("lf-cc: {err}");
            ExitCode::FAILURE
        }
    }
}

/// One positional input or library request, kept in command-line order
/// (archive and `-l` order matters to the linker).
#[derive(Debug)]
enum Item {
    /// A C source file to compile.
    Source(String),
    /// An assembly file (`.s`) to assemble.
    Asm(String),
    /// A file handed to the linker as-is (`.o`, `.a`, `.so`, ...).
    File(String),
    /// A raw linker argument: `-l<lib>`, or one from `-Wl,`/`-Xlinker`.
    LinkerArg(String),
}

struct Options {
    items: Vec<Item>,
    output: Option<String>,
    opt: OptLevel,
    debug: bool,
    emit_lf: bool,
    emit_obj: bool,
    std: CStd,
    include_dirs: Vec<PathBuf>,
    lib_dirs: Vec<String>,
    cmdline: Vec<MacroOp>,
    nostdinc: bool,
    nostdlib: bool,
}

impl Options {
    fn pp_options(&self, input: &str) -> PpOptions {
        PpOptions {
            std: self.std,
            include_dirs: self.include_dirs.clone(),
            cmdline: self.cmdline.clone(),
            main_file_name: input.to_owned(),
            builtin_headers: !self.nostdinc,
        }
    }

    /// The translation units to compile (C and assembly sources), in order.
    fn sources(&self) -> impl Iterator<Item = &str> {
        self.items.iter().filter_map(|i| match i {
            Item::Source(p) | Item::Asm(p) => Some(p.as_str()),
            _ => None,
        })
    }
}

fn run(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") || args.is_empty() {
        print_usage();
        return Ok(());
    }
    let opts = parse_args(args)?;
    if opts.items.is_empty() {
        return Err("no input files (see `lf-cc --help`)".to_owned());
    }
    let n_sources = opts.sources().count();

    // `-S` / `--emit-lf`: lower each C source and dump its IR, then stop.
    if opts.emit_lf {
        if opts.output.is_some() && n_sources > 1 {
            return Err("cannot specify '-o' with '-S' and multiple source files".to_owned());
        }
        for item in &opts.items {
            let Item::Source(input) = item else { continue };
            let source = read_source(input)?;
            let module_name = Path::new(input)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("module")
                .to_owned();
            let (module, syms) =
                lf_cc::compile_to_ir_with(&source, &module_name, &opts.pp_options(input), opts.debug)
                    .map_err(|diags| render_diags(input, &source, &diags))?;
            let out = text::print_module(&module, &syms);
            match (&opts.output, n_sources) {
                (Some(path), _) => {
                    std::fs::write(path, out).map_err(|e| format!("cannot write {path}: {e}"))?
                }
                (None, 1) => print!("{out}"),
                (None, _) => {
                    let path = format!("{}.lf", stem(input));
                    std::fs::write(&path, out).map_err(|e| format!("cannot write {path}: {e}"))?
                }
            }
        }
        return Ok(());
    }

    // `-c`: compile each source to one relocatable ELF object (in the cwd).
    if opts.emit_obj {
        if opts.output.is_some() && n_sources > 1 {
            return Err("cannot specify '-o' with '-c' and multiple source files".to_owned());
        }
        if n_sources == 0 {
            return Err("no source files to compile with '-c'".to_owned());
        }
        for item in &opts.items {
            match item {
                Item::Source(input) | Item::Asm(input) => {
                    let output = opts.output.clone().unwrap_or_else(|| format!("{}.o", stem(input)));
                    let unit = compile_item(&opts, item)?;
                    write_unit_object(unit, Path::new(&output))?;
                }
                Item::File(path) => {
                    eprintln!("lf-cc: warning: {path}: linker input file unused because linking not done");
                }
                Item::LinkerArg(_) => {}
            }
        }
        return Ok(());
    }

    // Link. Freestanding when asked for (or when the host has no C runtime).
    let output = opts.output.clone().unwrap_or_else(|| match opts.items.as_slice() {
        [Item::Source(input)] => stem(input),
        _ => "a.out".to_owned(),
    });
    let crt = if opts.nostdlib { None } else { HostCrt::discover() };

    let mut units: Vec<Option<Unit>> = Vec::with_capacity(opts.items.len());
    for item in &opts.items {
        units.push(match item {
            Item::Source(_) | Item::Asm(_) => Some(compile_item(&opts, item)?),
            Item::File(_) | Item::LinkerArg(_) => None,
        });
    }

    // Pure path: only C sources (without file-scope asm) and no libc — the
    // framework's own in-memory linker core.
    let all_pure_c = units.iter().all(|u| matches!(u, Some(Unit::C { asm: None, .. })));
    if crt.is_none() && all_pure_c {
        let modules = units
            .into_iter()
            .map(|u| match u {
                Some(Unit::C { module, .. }) => module,
                _ => unreachable!("checked: all pure C units"),
            })
            .collect();
        let image = lf_cc::link_image(modules, opts.debug).map_err(|e| match e {
            BuildError::Backend(msg) => msg,
            BuildError::Frontend(_) => "link failed".to_owned(),
        })?;
        return link::write_executable(&output, &image);
    }

    // Otherwise write ELF objects to a temporary directory and let qld link
    // them with the other inputs, preserving command-line order.
    let tmp = TempDir::new()?;
    let mut link_items: Vec<String> = Vec::new();
    for (idx, (item, unit)) in opts.items.iter().zip(units).enumerate() {
        match (item, unit) {
            (Item::Source(input) | Item::Asm(input), Some(unit)) => {
                let base = tmp.path().join(format!("{idx}-{}", stem(input)));
                for obj in unit.write_objects(&base)? {
                    link_items.push(path_string(&obj)?);
                }
            }
            (Item::File(path) | Item::LinkerArg(path), _) => link_items.push(path.clone()),
            (_, None) => unreachable!("every source was compiled"),
        }
    }
    let mut extra: Vec<String> = opts.lib_dirs.iter().map(|d| format!("-L{d}")).collect();
    extra.extend(link_items);
    let argv: Vec<OsString> = match &crt {
        Some(crt) => host_c_link_args(crt, &[], &extra, Path::new(&output)),
        None => {
            // Freestanding static link: our own crt0 (a weak `_start`, so a
            // user-supplied one wins) followed by the inputs; no libraries.
            let crt0 = tmp.path().join("crt0.o");
            write_file(&crt0, &freestanding_crt0()?)?;
            let mut argv: Vec<OsString> =
                vec!["-static".into(), "-o".into(), output.clone().into(), crt0.into()];
            argv.extend(extra.into_iter().map(Into::into));
            argv
        }
    };
    link_gnu("lf-cc", &argv).map_err(|e| format!("link failed: {e}"))
}

/// One compiled translation unit.
enum Unit {
    /// A C source: its object module, plus the ELF object assembled from its
    /// file-scope `asm(...)` declarations when it has any.
    C { module: ObjectModule, asm: Option<Vec<u8>> },
    /// An assembled `.s` file (ELF bytes).
    Asm(Vec<u8>),
}

impl Unit {
    /// Write this unit as ELF objects named after `base` (`base.o`, plus
    /// `base.asm.o` for file-scope asm) and return their paths.
    fn write_objects(self, base: &Path) -> Result<Vec<PathBuf>, String> {
        let named = |suffix: &str| {
            let mut name = base.as_os_str().to_owned();
            name.push(suffix);
            PathBuf::from(name)
        };
        let main = named(".o");
        match self {
            Unit::C { module, asm } => {
                write_file(&main, &elf::write(&module))?;
                let mut paths = vec![main];
                if let Some(asm) = asm {
                    let extra = named(".asm.o");
                    write_file(&extra, &asm)?;
                    paths.push(extra);
                }
                Ok(paths)
            }
            Unit::Asm(bytes) => {
                write_file(&main, &bytes)?;
                Ok(vec![main])
            }
        }
    }
}

/// Compile (or assemble) one source item.
fn compile_item(opts: &Options, item: &Item) -> Result<Unit, String> {
    match item {
        Item::Source(input) => {
            let source = read_source(input)?;
            let compiled =
                lf_cc::compile_module_with(&source, input, &opts.pp_options(input), opts.opt, opts.debug)
                    .map_err(|e| build_error(input, &source, e))?;
            let asm = lf_cc::assemble_toplevel_asm(&compiled.toplevel_asm, input)
                .map_err(|e| build_error(input, &source, e))?;
            Ok(Unit::C { module: compiled.module, asm })
        }
        Item::Asm(input) => assemble_file(Path::new(input), &AsmOptions::new(TargetArch::X86_64))
            .map(Unit::Asm)
            .map_err(|e| format!("{input}: {e}")),
        Item::File(_) | Item::LinkerArg(_) => unreachable!("not a translation unit"),
    }
}

/// Write a unit as the single relocatable object `output` (the `-c` result).
/// A C source with file-scope asm yields two objects, which qld merges with a
/// relocatable (`-r`) link.
fn write_unit_object(unit: Unit, output: &Path) -> Result<(), String> {
    match unit {
        Unit::C { module, asm: None } => write_file(output, &elf::write(&module)),
        Unit::Asm(bytes) => write_file(output, &bytes),
        unit @ Unit::C { asm: Some(_), .. } => {
            let tmp = TempDir::new()?;
            let parts = unit.write_objects(&tmp.path().join("unit"))?;
            let mut argv: Vec<OsString> = vec!["-r".into(), "-o".into(), output.into()];
            argv.extend(parts.into_iter().map(Into::into));
            link_gnu("lf-cc", &argv).map_err(|e| format!("merging {}: {e}", output.display()))
        }
    }
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// The freestanding startup object: the same crt0 the in-memory linker core
/// synthesizes (call `main`, exit with its result via the raw syscall), but
/// with a *weak* `_start` so an input that defines its own takes precedence.
fn freestanding_crt0() -> Result<Vec<u8>, String> {
    let src = "        .text\n        .weak _start\n        .type _start, @function\n_start:\n\
               \x20       xorl %ebp, %ebp\n        call main\n        movl %eax, %edi\n\
               \x20       movl $60, %eax\n        syscall\n";
    assemble(&[AsmSource { name: "crt0.s", text: src }], &AsmOptions::new(TargetArch::X86_64))
}

fn read_source(input: &str) -> Result<String, String> {
    std::fs::read_to_string(input).map_err(|e| format!("cannot read {input}: {e}"))
}

fn build_error(input: &str, source: &str, e: BuildError) -> String {
    match e {
        BuildError::Frontend(diags) => render_diags(input, source, &diags),
        BuildError::Backend(msg) => format!("{input}: {msg}"),
    }
}

fn path_string(p: &Path) -> Result<String, String> {
    p.to_str().map(str::to_owned).ok_or_else(|| format!("non-UTF-8 path {}", p.display()))
}

/// A scratch directory for intermediate objects, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Result<TempDir, String> {
        let base = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        for attempt in 0..100u32 {
            let dir = base.join(format!("lf-cc-{}-{nanos}-{attempt}", std::process::id()));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(TempDir(dir)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(format!("cannot create {}: {e}", dir.display())),
            }
        }
        Err("cannot create a temporary directory".to_owned())
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Driver flags accepted and ignored because they cannot change the meaning of
/// the program lf-cc produces: warning controls, tuning, and code-generation
/// knobs whose effect is already lf-cc's behavior (`-fwrapv`: signed arithmetic
/// wraps; `-fno-strict-aliasing`: no type-based alias analysis; `-fsigned-char`:
/// `char` is signed on x86-64; `-fPIC`: executables only).
fn is_ignored_flag(arg: &str) -> bool {
    const EXACT: &[&str] = &[
        "-pipe", "-m64", "-w", "-pedantic", "-pedantic-errors", "-no-pie", "-nopie",
        "-fno-common", "-fcommon", "-fPIC", "-fpic", "-fPIE", "-fpie", "-fno-pic", "-fno-PIC",
        "-fno-pie", "-fno-PIE", "-fwrapv", "-fno-strict-aliasing", "-fstrict-aliasing",
        "-fomit-frame-pointer", "-fno-omit-frame-pointer", "-fsigned-char", "-fno-unsigned-char",
        "-fexceptions", "-fno-exceptions", "-fasynchronous-unwind-tables",
        "-fno-asynchronous-unwind-tables", "-funwind-tables", "-fno-unwind-tables",
        "-ffunction-sections", "-fdata-sections", "-fno-plt", "-fno-ident", "-fident",
        "-fstack-clash-protection", "-fno-stack-clash-protection", "-fno-semantic-interposition",
        "-fstack-protector", "-fstack-protector-strong", "-fstack-protector-all",
        "-fno-stack-protector", "-fno-builtin", "-fbuiltin", "-fno-inline", "-finline-functions",
        "-fno-strict-overflow", "-ffreestanding",
    ];
    const PREFIX: &[&str] = &[
        "-march=", "-mtune=", "-fdiagnostics-", "-fmessage-length=", "-fvisibility=",
        "-fno-builtin-", "-fcf-protection", "-fmax-errors=", "-fno-diagnostics-",
        "-fcolor-diagnostics", "-fno-color-diagnostics",
    ];
    if EXACT.contains(&arg) || PREFIX.iter().any(|p| arg.starts_with(p)) {
        return true;
    }
    // `-W<warning>` (but not the `-Wl,`/`-Wa,`/`-Wp,` pass-throughs).
    arg.starts_with("-W") && !arg.starts_with("-Wl,") && !arg.starts_with("-Wa,") && !arg.starts_with("-Wp,")
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut opts = Options {
        items: Vec::new(),
        output: None,
        opt: OptLevel::O0,
        debug: false,
        emit_lf: false,
        emit_obj: false,
        std: CStd::default(),
        include_dirs: Vec::new(),
        lib_dirs: Vec::new(),
        cmdline: Vec::new(),
        nostdinc: false,
        nostdlib: false,
    };

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let arg = arg.as_str();
        let mut value = |flag: &str| -> Result<String, String> {
            it.next().cloned().ok_or_else(|| format!("{flag} requires an argument"))
        };
        match arg {
            "-o" => opts.output = Some(value("-o")?),
            "-g" | "--debug" => opts.debug = true,
            "-g0" => opts.debug = false,
            "-S" | "--emit-lf" => opts.emit_lf = true,
            "-c" => opts.emit_obj = true,
            "-nostdinc" => opts.nostdinc = true,
            "-nostdlib" => opts.nostdlib = true,
            "-ansi" => opts.std = CStd::C89,
            "-pthread" => {
                opts.cmdline.push(MacroOp::Define("_REENTRANT".to_owned()));
                opts.items.push(Item::LinkerArg("-lpthread".to_owned()));
            }
            "-rdynamic" => opts.items.push(Item::LinkerArg("--export-dynamic".to_owned())),
            "-s" => opts.items.push(Item::LinkerArg("--strip-all".to_owned())),
            "-Os" | "-Oz" => opts.opt = OptLevel::O2,
            "-Og" => opts.opt = OptLevel::O1,
            "-I" | "-isystem" | "-iquote" | "-idirafter" => {
                opts.include_dirs.push(PathBuf::from(value(arg)?))
            }
            "-D" => opts.cmdline.push(MacroOp::Define(value("-D")?)),
            "-U" => opts.cmdline.push(MacroOp::Undef(value("-U")?)),
            "-L" => opts.lib_dirs.push(value("-L")?),
            "-l" => opts.items.push(Item::LinkerArg(format!("-l{}", value("-l")?))),
            "-Xlinker" => opts.items.push(Item::LinkerArg(value("-Xlinker")?)),
            _ if arg.starts_with("-Wl,") => opts
                .items
                .extend(arg[4..].split(',').filter(|a| !a.is_empty()).map(|a| Item::LinkerArg(a.to_owned()))),
            _ if is_ignored_flag(arg) => {}
            _ if arg.starts_with("-I") => opts.include_dirs.push(PathBuf::from(&arg[2..])),
            _ if arg.starts_with("-D") => opts.cmdline.push(MacroOp::Define(arg[2..].to_owned())),
            _ if arg.starts_with("-U") => opts.cmdline.push(MacroOp::Undef(arg[2..].to_owned())),
            _ if arg.starts_with("-L") => opts.lib_dirs.push(arg[2..].to_owned()),
            _ if arg.starts_with("-l") => opts.items.push(Item::LinkerArg(arg.to_owned())),
            _ if arg.starts_with("-o") => opts.output = Some(arg[2..].to_owned()),
            _ if arg.starts_with("-ggdb") || arg.starts_with("-gdwarf") || arg == "-g1"
                || arg == "-g2" || arg == "-g3" =>
            {
                opts.debug = true
            }
            _ if arg.starts_with("--std=") || arg.starts_with("-std=") => {
                let name = arg.split_once('=').map(|(_, v)| v).unwrap_or("");
                opts.std = CStd::parse(name)
                    .ok_or_else(|| format!("unknown -std value '{name}'"))?;
            }
            tok if OptLevel::parse_flag(tok).is_some() => {
                opts.opt = OptLevel::parse_flag(tok).expect("checked");
            }
            flag if flag.starts_with('-') && flag != "-" => {
                return Err(format!("unrecognized option '{flag}'"));
            }
            positional if positional.ends_with(".c") => {
                opts.items.push(Item::Source(positional.to_owned()))
            }
            positional if positional.ends_with(".s") => {
                opts.items.push(Item::Asm(positional.to_owned()))
            }
            positional if positional.ends_with(".S") || positional.ends_with(".h") => {
                return Err(format!("{positional}: unsupported input file type"));
            }
            positional => opts.items.push(Item::File(positional.to_owned())),
        }
    }
    Ok(opts)
}

fn print_usage() {
    println!("lf-cc — a C compiler driver built on LatticeFoundry\n");
    println!("usage:");
    println!("  lf-cc [options] <file.c|file.s|file.o|lib.a|-l<lib>>...\n");
    println!("  -c             compile each source to an object (foo.c -> foo.o)");
    println!("  -S / --emit-lf dump the lowered .lf IR instead of an executable");
    println!("  -o <out>       output path (default: the input stem for one .c, else a.out)");
    println!("  -O0..-O3       optimization level (default: -O0; -Os/-Oz = -O2, -Og = -O1)");
    println!("  -g / --debug   emit DWARF debug info (source lines)");
    println!("  --std=<std>    C standard: c89/c99/c11/c17/c23 or gnuNN (default: gnu17)");
    println!("  -I <dir>       add a directory to the #include search path (repeatable)");
    println!("  -nostdinc      do not consult the builtin freestanding standard headers");
    println!("  -D name[=val]  predefine a macro (repeatable)");
    println!("  -U name        undefine a macro (repeatable)");
    println!("  -L <dir>       add a library search directory");
    println!("  -l <lib>       link against lib<lib> (.so or .a)");
    println!("  -Wl,<a>,<b>    pass arguments to the linker (also -Xlinker <a>)");
    println!("  -nostdlib      freestanding link: static, no libc, LatticeFoundry's own crt0");
    println!("                 (_start calls main and exits with its result)\n");
    println!("Linking is hosted by default: the host C runtime (crt1.o, libc) is linked in");
    println!("with qld, LatticeFoundry's own linker. Without a host C runtime lf-cc falls");
    println!("back to the -nostdlib link. Warning flags (-W...), -pipe, -m64, -march=,");
    println!("and code-generation flags that do not change lf-cc's output are ignored.");
}

/// The file stem of `input` (`dir/foo.c` → `foo`).
fn stem(input: &str) -> String {
    Path::new(input)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| "a".to_owned())
}

/// Render a batch of front-end diagnostics against the C source for the terminal.
fn render_diags(path: &str, source: &str, diags: &[Diagnostic]) -> String {
    let mut out = String::new();
    for d in diags {
        let sev = match d.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        };
        match d.span {
            Some(span) => {
                let (line, col) = line_col(source, span.start);
                out.push_str(&format!("{path}:{line}:{col}: {sev}: {}\n", d.message));
            }
            None => out.push_str(&format!("{path}: {sev}: {}\n", d.message)),
        }
    }
    out.push_str(&format!("{} error(s)", diags.iter().filter(|d| d.is_error()).count()));
    out
}

fn line_col(src: &str, offset: u32) -> (u32, u32) {
    let mut line = 1u32;
    let mut col = 1u32;
    for (i, b) in src.bytes().enumerate() {
        if i as u32 >= offset {
            break;
        }
        if b == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}
