//! `lf` — the LatticeFoundry compiler driver.
//!
//! The umbrella front end that ties the other tools together. The `build`
//! subcommand is the Phase-8 end-to-end path: it parses an IR module (`.lf`
//! text or `.lfb` binary), verifies it, lowers it to x86-64 machine code, links
//! it into a **static native executable** with our own linker, and marks it
//! executable — no system linker or libc involved.
//!
//! Other outputs: `--shared` builds a **shared library** of position-
//! independent code (linked by `qld`, optional `-soname`), `--pie` a
//! position-independent executable against the host C library (its `main` is
//! called by the C runtime), and `-c` stops at the relocatable ELF object
//! (`--pic`/`--pie` pick its relocation model).

use std::path::Path;
use std::process::ExitCode;

use latticefoundry::codegen::{CodegenOptions, RelocModel, StackAssumptions, StackReport};
use latticefoundry::ir::{Module, binary, merge_modules, text};
use latticefoundry::link::{self, ImageOptions};
use latticefoundry::support::StrInterner;
use latticefoundry::support::diagnostics::{Diagnostic, FileId, Severity};
use latticefoundry::transform::pipeline::{self, OptLevel};
use latticefoundry::{target, verify};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("lf (LatticeFoundry) {}", latticefoundry::VERSION);
            ExitCode::SUCCESS
        }
        None | Some("--help" | "-h") => {
            print_usage();
            ExitCode::SUCCESS
        }
        Some("build") => match build(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("lf: {err}");
                ExitCode::FAILURE
            }
        },
        Some(other) => {
            eprintln!("lf: unrecognized subcommand '{other}' (try `lf --help`)");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("lf — LatticeFoundry compiler driver\n");
    println!("usage:");
    println!(
        "  lf build <inputs...> [-o <out>] [-O0|-O1|-O2|-O3] [--entry <name>] [-g] [--lto] [--no-verify]"
    );
    println!("           [--stack-usage] [--no-stack-probes]");
    println!("           [--shared [-soname <name>] | --pie | -c [--pic|--pie]] [-L<dir>] [-l<lib>]");
    println!("  lf --version | --help\n");
    println!("  -O0..-O3       optimization level (default: -O0)");
    println!("  -g / --debug   emit DWARF debug info (source lines, symbols)");
    println!("  --lto          link-time optimize across inputs (implied by 2+ inputs)");
    println!("  --stack-usage  print each function's stack frame and the worst-case depth");
    println!("  --no-stack-probes  omit stack probes (only with a proven stack bound)");
    println!("  --shared       build a shared library (position-independent; default lib<input>.so)");
    println!("  -soname <name> set the shared library's DT_SONAME");
    println!("  --pie          build a position-independent executable against the host C library");
    println!("  -c             emit the relocatable ELF object only (default <input>.o)");
    println!("  --pic          with -c: position-independent code for a shared library");
    println!("  -L<dir> -l<lib>  extra library search paths / libraries (--shared, --pie)");
    println!("`lf build` compiles one or more IR modules to a static native executable.");
    println!("With several inputs (or --lto), the modules are IR-linked into one, the");
    println!("-O pipeline runs over the whole program (cross-module inlining), then codegen.");
}

struct BuildOptions {
    inputs: Vec<String>,
    output: Option<String>,
    entry: Option<String>,
    verify: bool,
    debug: bool,
    opt: OptLevel,
    lto: bool,
    stack_usage: bool,
    stack_probes: bool,
    output_kind: OutputKind,
    /// `-c --pic`: shared-library (PIC) object code.
    pic: bool,
    /// `-c --pie`: PIE object code.
    pie: bool,
    soname: Option<String>,
    /// `-L<dir>` / `-l<lib>` arguments passed through to the linker.
    link_extra: Vec<String>,
}

/// What `lf build` produces.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputKind {
    /// A static executable linked by our own linker (the default).
    Static,
    /// A shared library (`--shared`), linked by qld.
    Shared,
    /// A PIE executable against the host C library (`--pie`), linked by qld.
    Pie,
    /// A relocatable ELF object (`-c`).
    Object,
}

impl BuildOptions {
    /// The relocation model the requested output needs.
    fn reloc_model(&self) -> RelocModel {
        match self.output_kind {
            OutputKind::Shared => RelocModel::Pic,
            OutputKind::Pie => RelocModel::Pie,
            OutputKind::Object if self.pic => RelocModel::Pic,
            OutputKind::Object if self.pie => RelocModel::Pie,
            OutputKind::Static | OutputKind::Object => RelocModel::Static,
        }
    }
}

fn build(args: &[String]) -> Result<(), String> {
    let opts = parse_build(args)?;

    // Parse every input in whichever encoding its extension names, threading one
    // interner so symbol names compare across modules (required for IR linking).
    let mut syms = StrInterner::new();
    let mut modules = Vec::with_capacity(opts.inputs.len());
    for input in &opts.inputs {
        modules.push(load(input, &mut syms)?);
    }

    // Combine into one module. Multiple inputs (or --lto) are IR-linked so the
    // optimizer sees the whole program; a single input needs no merge.
    let mut module = if modules.len() == 1 && !opts.lto {
        modules.into_iter().next().expect("one module")
    } else {
        merge_modules(modules, "lto").map_err(|e| {
            let name = syms.resolve(e.symbol());
            let kind = match e {
                latticefoundry::ir::MergeError::DuplicateFunction(_) => "function",
                latticefoundry::ir::MergeError::DuplicateGlobal(_) => "global",
            };
            format!("link (LTO) error: duplicate definition of {kind} '{name}'")
        })?
    };

    // Verify (Structural tier) unless suppressed.
    if opts.verify {
        verify_or_err(&module, "input")?;
    }

    // Run the optimization pipeline, then re-verify (a pass must preserve validity).
    pipeline::optimize(&mut module, opts.opt);
    if opts.verify && opts.opt != OptLevel::O0 {
        verify_or_err(&module, "optimized")?;
    }

    // Lower to a relocatable object, then link into a static executable. With
    // `-g`, also emit DWARF debug info and a debuggable image (section headers +
    // symbol table + `.debug_*`).
    let cg = CodegenOptions::default()
        .with_stack_probes(opts.stack_probes)
        .with_reloc_model(opts.reloc_model());
    let compiled = if opts.debug {
        let comp_dir = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(str::to_owned))
            .unwrap_or_default();
        let file_name = opts.inputs.first().cloned().unwrap_or_default();
        let source = target::x86_64::DebugSource { file_name, comp_dir };
        target::x86_64::compile_module_debug_with(&module, &syms, &source, &cg)
    } else {
        target::x86_64::compile_module_with(&module, &syms, &cg)
    };
    let entry = opts.entry.clone().unwrap_or_else(|| ImageOptions::default().entry);
    if opts.stack_usage {
        print_stack_usage(&compiled.stack, &entry);
    }
    let obj = compiled.object;
    if opts.output_kind != OutputKind::Static {
        return link_with_qld(&opts, &obj);
    }
    let image_opts = ImageOptions {
        debug: opts.debug,
        entry,
        ..ImageOptions::default()
    };
    let image =
        link::link_executable(vec![obj], &image_opts).map_err(|e| format!("link error: {e}"))?;

    let output = opts
        .output
        .clone()
        .unwrap_or_else(|| default_output(&opts.inputs[0]));
    link::write_executable(&output, &image)?;
    Ok(())
}

/// Write `obj` as an ELF object and, unless `-c`, link it with qld into a
/// shared library (`--shared`) or a PIE executable (`--pie`).
fn link_with_qld(opts: &BuildOptions, obj: &latticefoundry::mc::object::ObjectModule) -> Result<(), String> {
    use latticefoundry::link::gnu::{self, HostCrt};
    let stem = default_output(&opts.inputs[0]);
    let output = opts.output.clone().unwrap_or_else(|| match opts.output_kind {
        OutputKind::Shared => format!("lib{stem}.so"),
        OutputKind::Object => format!("{stem}.o"),
        OutputKind::Pie | OutputKind::Static => stem.clone(),
    });
    let elf = latticefoundry::mc::elf::write(obj);
    if opts.output_kind == OutputKind::Object {
        return std::fs::write(&output, elf).map_err(|e| format!("cannot write {output}: {e}"));
    }
    // qld reads its inputs from files: stage the object in the temp directory.
    let tmp = std::env::temp_dir().join(format!("lf-{}-{stem}.o", std::process::id()));
    std::fs::write(&tmp, elf).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    let out = Path::new(&output);
    let crt = HostCrt::discover();
    let args = match opts.output_kind {
        OutputKind::Shared => {
            gnu::shared_library_args(crt.as_ref(), &[&tmp], opts.soname.as_deref(), &opts.link_extra, out)
        }
        _ => {
            let Some(crt) = crt else {
                let _ = std::fs::remove_file(&tmp);
                return Err("--pie links against the host C library, but no crt1.o was found".to_owned());
            };
            gnu::host_c_pie_link_args(&crt, &[&tmp], &opts.link_extra, out)
        }
    };
    let result = gnu::link_gnu("lf", &args).map_err(|e| format!("link error: {e}"));
    let _ = std::fs::remove_file(&tmp);
    result
}

/// Print the `--stack-usage` table and the worst-case stack depth from `entry`
/// (counted from `_start`'s stack pointer just before it calls `entry`).
fn print_stack_usage(report: &StackReport, entry: &str) {
    print!("{report}");
    match report.worst_case_depth(entry, &StackAssumptions::new()) {
        Ok(bound) => println!(
            "worst-case stack from '{entry}': {} bytes ({})",
            bound.bytes,
            bound.path.join(" -> ")
        ),
        Err(why) => println!("worst-case stack from '{entry}': unbounded: {why}"),
    }
}

/// Verify `module`, rendering any error diagnostics and returning a driver error
/// naming the `stage` (`"input"` / `"optimized"`).
fn verify_or_err(module: &Module, stage: &str) -> Result<(), String> {
    if let Err(diags) = verify::verify_module(module) {
        for d in &diags {
            eprintln!("{}", render(d));
        }
        let errs = diags.iter().filter(|d| d.severity == Severity::Error).count();
        return Err(format!("{stage} verification failed ({errs} error(s))"));
    }
    Ok(())
}

fn parse_build(args: &[String]) -> Result<BuildOptions, String> {
    let mut inputs: Vec<String> = Vec::new();
    let mut output: Option<String> = None;
    let mut entry: Option<String> = None;
    let mut verify = true;
    let mut debug = false;
    let mut opt = OptLevel::O0;
    let mut lto = false;
    let mut stack_usage = false;
    let mut stack_probes = true;
    let mut output_kind = OutputKind::Static;
    let mut pic = false;
    let mut pie = false;
    let mut soname: Option<String> = None;
    let mut link_extra: Vec<String> = Vec::new();

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" => output = Some(it.next().ok_or("-o requires a path")?.clone()),
            "--entry" | "-e" => entry = Some(it.next().ok_or("--entry requires a name")?.clone()),
            "--no-verify" => verify = false,
            "-g" | "--debug" => debug = true,
            "--lto" => lto = true,
            "--stack-usage" => stack_usage = true,
            "--no-stack-probes" => stack_probes = false,
            "--shared" | "-shared" => output_kind = OutputKind::Shared,
            "--pie" | "-pie" => pie = true,
            "-c" => output_kind = OutputKind::Object,
            "--pic" | "-fPIC" | "-fpic" => pic = true,
            "-soname" | "--soname" => {
                soname = Some(it.next().ok_or("-soname requires a name")?.clone());
            }
            flag if flag.starts_with("--soname=") => soname = Some(flag["--soname=".len()..].to_owned()),
            flag if (flag.starts_with("-L") || flag.starts_with("-l")) && flag.len() > 2 => {
                link_extra.push(flag.to_owned());
            }
            tok if OptLevel::parse_flag(tok).is_some() => {
                opt = OptLevel::parse_flag(tok).expect("checked");
            }
            flag if flag.starts_with('-') && flag != "-" => {
                return Err(format!("unrecognized option '{flag}'"));
            }
            positional => inputs.push(positional.to_owned()),
        }
    }

    if inputs.is_empty() {
        return Err("no input file (see `lf --help`)".to_owned());
    }
    match output_kind {
        OutputKind::Shared if pie => return Err("--shared and --pie are exclusive".to_owned()),
        OutputKind::Static if pie => output_kind = OutputKind::Pie,
        OutputKind::Object if pie && pic => return Err("--pic and --pie are exclusive".to_owned()),
        _ => {}
    }
    if pic && output_kind != OutputKind::Object && output_kind != OutputKind::Shared {
        return Err("--pic only applies to -c (use --shared for a shared library)".to_owned());
    }
    if soname.is_some() && output_kind != OutputKind::Shared {
        return Err("-soname only applies to --shared".to_owned());
    }
    if !link_extra.is_empty() && !matches!(output_kind, OutputKind::Shared | OutputKind::Pie) {
        return Err("-L/-l only apply to --shared and --pie".to_owned());
    }
    if entry.is_some() && output_kind != OutputKind::Static {
        return Err("--entry only applies to static executables".to_owned());
    }
    Ok(BuildOptions {
        inputs,
        output,
        entry,
        verify,
        debug,
        opt,
        lto,
        stack_usage,
        stack_probes,
        output_kind,
        pic,
        pie,
        soname,
        link_extra,
    })
}

/// The default output path: the input with any extension stripped, or `a.out`.
fn default_output(input: &str) -> String {
    Path::new(input)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_owned())
        .unwrap_or_else(|| "a.out".to_owned())
}

fn load(path: &str, syms: &mut StrInterner) -> Result<Module, String> {
    let is_binary = Path::new(path).extension().and_then(|e| e.to_str()) == Some("lfb");
    if is_binary {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        binary::decode(&bytes, syms).map_err(|e| format!("decode error in {path}: {e}"))
    } else {
        let src = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        text::parse_module(&src, FileId::new(0), syms).map_err(|diags| {
            let rendered: Vec<String> = diags.iter().map(render).collect();
            format!("parse error in {path}:\n{}", rendered.join("\n"))
        })
    }
}

/// Render a diagnostic for the terminal.
fn render(d: &Diagnostic) -> String {
    let sev = match d.severity {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
    };
    match d.span {
        Some(span) => format!("{sev}[{}..{}]: {}", span.start, span.end, d.message),
        None => format!("{sev}: {}", d.message),
    }
}
