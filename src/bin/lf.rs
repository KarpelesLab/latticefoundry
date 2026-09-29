//! `lf` — the LatticeFoundry compiler driver.
//!
//! The umbrella front end that ties the other tools together. The `build`
//! subcommand is the Phase-8 end-to-end path: it parses an IR module (`.lf`
//! text or `.lfb` binary), verifies it, lowers it to x86-64 machine code, links
//! it into a **static native executable** with our own linker, and marks it
//! executable — no system linker or libc involved.
//!
//! `--target <triple>` picks the architecture and OS (and so the calling
//! convention and object format), `-c` stops at a relocatable object (ELF,
//! PE/COFF or Mach-O; `--format` overrides the triple's), and `--oformat
//! binary|ihex` writes a firmware image instead of an ELF executable. Windows
//! executables are linked by qld's PE driver. A Cortex-M target
//! (`thumbv7m-none-eabi`, `thumbv7em-none-eabi`) links through qld with a
//! generated vector table and reset handler, and `--oformat binary|ihex` turns
//! the result into a flashable image (`-L`/`-l` add the runtime library, e.g.
//! `-lgcc`, for the soft-float and 64-bit division helpers).
//!
//! For AVR (`--target avr-atmega328p`), `--oformat ihex|binary` links a
//! flashable firmware image (vector table, startup code, program, runtime)
//! and `-c` writes an ELF32 `EM_AVR` object.
//!
//! For x86-64 Linux, `--shared` builds a **shared library** of position-
//! independent code (linked by `qld`, optional `-soname`), `--pie` a
//! position-independent executable against the host C library (its `main` is
//! called by the C runtime), and `-c --pic`/`-c --pie` pick an object's
//! relocation model.
//!
//! `--target wasm32` (or `wasm32-unknown-unknown`) builds a self-contained
//! **WebAssembly module** (`.wasm`: memory, stack pointer, function table and
//! data included; undefined functions imported from `"env"`), and with `-c` a
//! relocatable wasm object for `wasm-ld`. A module that declares no data
//! layout gets the wasm32 one (ILP32).

use std::path::Path;
use std::process::ExitCode;

use latticefoundry::codegen::{CodegenOptions, RelocModel, StackAssumptions, StackReport};
use latticefoundry::ir::{Module, binary, merge_modules, text};
use latticefoundry::link::raw::{self, RawFormat};
use latticefoundry::link::{self, ImageOptions};
use latticefoundry::mc::object::ObjectModule;
use latticefoundry::target::{ObjectFormat, TargetArch, TargetOs, Triple};
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
    println!("           [--stack-usage] [--no-stack-probes] [--target <triple>] [-c [--format <fmt>]]");
    println!("           [--oformat elf|binary|ihex] [--base <addr>]");
    println!("           [--shared [-soname <name>] | --pie | -c [--pic|--pie]] [-L<dir>] [-l<lib>]");
    println!("  lf --version | --help\n");
    println!("  -O0..-O3       optimization level (default: -O0)");
    println!("  -g / --debug   emit DWARF debug info (source lines, symbols)");
    println!("  --lto          link-time optimize across inputs (implied by 2+ inputs)");
    println!("  --stack-usage  print each function's stack frame and the worst-case depth");
    println!("  --no-stack-probes  omit stack probes (only with a proven stack bound)");
    println!("  --target T     x86_64-linux (default), x86_64-windows, x86_64-apple-darwin,");
    println!("                 aarch64-windows, aarch64-apple-darwin, thumbv7m-none-eabi (Cortex-M),");
    println!("                 ... (ABI + object format), wasm32 (a WebAssembly module;");
    println!("                 with -c, a wasm-ld object), avr-atmega328p (AVR firmware:");
    println!("                 --oformat ihex|binary, or -c for an ELF object)");
    println!("  -c             emit a relocatable object instead of linking");
    println!("  --format F     object format for -c: elf, coff or macho (default: the target's)");
    println!("  --oformat F    executable format: elf (default), binary or ihex (firmware)");
    println!("  --base ADDR    image load address (default 0x400000); for binary/ihex, where");
    println!("                 the first byte of code goes; for Cortex-M, the flash origin (default 0)");
    println!("  --shared       build a shared library (position-independent; default lib<input>.so)");
    println!("  -soname <name> set the shared library's DT_SONAME");
    println!("  --pie          build a position-independent executable against the host C library");
    println!("  --pic          with -c: position-independent code for a shared library");
    println!("  -L<dir> -l<lib>  extra library search paths / libraries (--shared, --pie, Cortex-M)");
    println!("`lf build` compiles one or more IR modules to a static native executable");
    println!("(for a Windows target, a PE executable whose entry point is `main`).");
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
    target: Triple,
    /// The AVR device named by `--target` (for an AVR target).
    device: Option<target::avr::Device>,
    format: Option<ObjectFormat>,
    oformat: Option<RawFormat>,
    base: Option<u64>,
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
    /// An executable (or firmware image) for the target, the default.
    Executable,
    /// A shared library (`--shared`), linked by qld.
    Shared,
    /// A PIE executable against the host C library (`--pie`), linked by qld.
    Pie,
    /// A relocatable object (`-c`).
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
            OutputKind::Executable | OutputKind::Object => RelocModel::Static,
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
            let kind = match e {
                latticefoundry::ir::MergeError::DuplicateFunction(_) => "function",
                latticefoundry::ir::MergeError::DuplicateGlobal(_) => "global",
                latticefoundry::ir::MergeError::DataLayoutMismatch => {
                    return format!("link (LTO) error: {e}");
                }
            };
            let name = e.symbol().map_or("?", |s| syms.resolve(s));
            format!("link (LTO) error: duplicate definition of {kind} '{name}'")
        })?
    };

    // A module that did not declare a layout (so has the LP64 default) is
    // compiled with the wasm32 one: its pointers are 32 bits.
    if opts.target.arch == TargetArch::Wasm32 && *module.data_layout() == latticefoundry::ir::DataLayout::lp64() {
        module.set_data_layout(target::wasm32::data_layout());
    }
    // An AVR module that declares no layout gets AVR's (16-bit pointers, the
    // program-memory address space 1) before it is checked.
    if opts.target.arch == TargetArch::Avr && *module.data_layout() == latticefoundry::ir::DataLayout::lp64() {
        module.set_data_layout(target::avr::data_layout());
    }

    // Verify (Structural tier) unless suppressed.
    if opts.verify {
        verify_or_err(&module, "input")?;
    }

    // Run the optimization pipeline, then re-verify (a pass must preserve validity).
    pipeline::optimize(&mut module, opts.opt);
    if opts.verify && opts.opt != OptLevel::O0 {
        verify_or_err(&module, "optimized")?;
    }

    // Lower to a relocatable object for the target, then either write it
    // (`-c`) or link an executable. With `-g`, also emit DWARF debug info and a
    // debuggable image (section headers + symbol table + `.debug_*`).
    let triple = opts.target;
    let cg = CodegenOptions::default()
        .with_stack_probes(opts.stack_probes)
        .with_os(triple.os)
        .with_reloc_model(opts.reloc_model());
    if triple.arch == TargetArch::Wasm32 {
        return build_wasm(&opts, &module, &syms, &cg);
    }
    let compiled = match triple.arch {
        TargetArch::X86_64 if opts.debug => {
            let comp_dir = std::env::current_dir()
                .ok()
                .and_then(|p| p.to_str().map(str::to_owned))
                .unwrap_or_default();
            let file_name = opts.inputs.first().cloned().unwrap_or_default();
            let source = target::x86_64::DebugSource { file_name, comp_dir };
            target::x86_64::compile_module_debug_with(&module, &syms, &source, &cg)
        }
        _ if opts.debug => return Err(format!("-g is supported for x86-64 only, not {triple}")),
        TargetArch::Avr => {
            target::check_options(TargetArch::Avr, &cg).map_err(|e| e.to_string())?;
            let device = opts.device.unwrap_or(target::avr::Device::ATMEGA328P);
            target::avr::compile_module_for_device(&module, &syms, &cg, &device)
        }
        arch => target::compile_module_for(arch, &module, &syms, &cg).map_err(|e| e.to_string())?,
    };
    let entry = opts.entry.clone().unwrap_or_else(|| ImageOptions::default().entry);
    if opts.stack_usage {
        let mut report = compiled.stack.clone();
        if triple.arch == TargetArch::Avr {
            // The runtime helpers the program may call are part of its stack.
            let device = opts.device.unwrap_or(target::avr::Device::ATMEGA328P);
            for member in target::avr::runtime::compiled(&device) {
                report.extend(member.stack);
            }
        }
        print_stack_usage(&report, &entry);
    }
    let obj = compiled.object;
    let stem = default_output(&opts.inputs[0]);
    let output_or = |ext: &str| {
        opts.output.clone().unwrap_or_else(|| if ext.is_empty() { stem.clone() } else { format!("{stem}.{ext}") })
    };

    if matches!(opts.output_kind, OutputKind::Shared | OutputKind::Pie) {
        return link_with_qld(&opts, &obj);
    }
    if opts.output_kind == OutputKind::Object {
        let format = opts.format.unwrap_or(triple.object_format());
        let bytes = latticefoundry::mc::format::write_object_as(&obj, triple.arch, format)
            .map_err(|e| format!("cannot write a {} object for {triple}: {e}", format.name()))?;
        let output = output_or(if format == ObjectFormat::Coff { "obj" } else { "o" });
        return std::fs::write(&output, bytes).map_err(|e| format!("cannot write {output}: {e}"));
    }

    if triple.arch == TargetArch::Avr {
        let device = opts.device.unwrap_or(target::avr::Device::ATMEGA328P);
        let Some(format) = opts.oformat else {
            return Err("an AVR executable is a firmware image: add --oformat ihex (or binary), or -c".to_owned());
        };
        let fw = target::avr::link::build(vec![obj], &device, &entry).map_err(|e| format!("link error: {e}"))?;
        let (bytes, ext) = match format {
            RawFormat::Binary => (fw.flash.clone(), "bin"),
            RawFormat::Ihex => (fw.to_ihex().into_bytes(), "hex"),
        };
        let output = output_or(ext);
        return std::fs::write(&output, bytes).map_err(|e| format!("cannot write {output}: {e}"));
    }

    match (triple.arch, triple.object_format(), opts.oformat) {
        (TargetArch::X86_64, ObjectFormat::Elf, None) => {
            let image = link::link_executable(vec![obj], &image_options(&opts, entry))
                .map_err(|e| format!("link error: {e}"))?;
            link::write_executable(&output_or(""), &image)
        }
        (TargetArch::X86_64, ObjectFormat::Elf, Some(format)) => {
            let fw = raw::link_firmware(vec![obj], &image_options(&opts, entry))
                .map_err(|e| format!("link error: {e}"))?;
            let (bytes, ext) = match format {
                RawFormat::Binary => (raw::to_binary(&fw.segments, 0)?.1, "bin"),
                RawFormat::Ihex => (raw::to_ihex(&fw.segments, Some(fw.entry))?.into_bytes(), "hex"),
            };
            let output = output_or(ext);
            std::fs::write(&output, bytes).map_err(|e| format!("cannot write {output}: {e}"))
        }
        (TargetArch::Thumb, ObjectFormat::Elf, format) => link_cortex_m(&opts, obj, &entry, format, &output_or),
        (TargetArch::X86_64 | TargetArch::AArch64, ObjectFormat::Coff, None) => {
            let entry = opts.entry.as_deref().unwrap_or("main");
            link_pe(&obj, triple, entry, &output_or("exe"), opts.base)
        }
        (_, _, Some(_)) => Err(format!(
            "--oformat binary/ihex needs an x86-64 or Cortex-M ELF target, not {triple}"
        )),
        _ => Err(format!(
            "cannot link a {triple} executable yet: emit an object with -c and link it with the platform's linker"
        )),
    }
}

/// `lf build --target wasm32`: a self-contained wasm module, or with `-c` a
/// relocatable wasm object.
fn build_wasm(opts: &BuildOptions, module: &Module, syms: &StrInterner, cg: &CodegenOptions) -> Result<(), String> {
    if opts.debug {
        return Err("-g is supported for x86-64 only, not wasm32".to_owned());
    }
    let mut compiled = target::wasm32::compile(module, syms, cg).map_err(|e| e.to_string())?;
    if opts.stack_usage {
        print_stack_usage(&compiled.stack, opts.entry.as_deref().unwrap_or("main"));
    }
    let stem = default_output(&opts.inputs[0]);
    let (bytes, ext) = if opts.output_kind == OutputKind::Object {
        if opts.format.is_some_and(|f| f != ObjectFormat::Wasm) {
            return Err("a wasm32 object can only be written in the wasm format".to_owned());
        }
        (compiled.object.to_relocatable(), "o")
    } else {
        if let Some(entry) = &opts.entry {
            let f = compiled.object.funcs.iter_mut().find(|f| f.name == *entry && f.body.is_some());
            f.ok_or_else(|| format!("--entry: no function '{entry}' is defined"))?.export = true;
        }
        let linked = compiled.object.to_linked(&target::wasm32::LinkOptions::default()).map_err(|e| format!("link error: {e}"))?;
        (linked, "wasm")
    };
    let output = opts.output.clone().unwrap_or_else(|| format!("{stem}.{ext}"));
    std::fs::write(&output, bytes).map_err(|e| format!("cannot write {output}: {e}"))
}

/// Write `obj` as an ELF object and link it with qld into a shared library
/// (`--shared`) or a PIE executable (`--pie`).
fn link_with_qld(opts: &BuildOptions, obj: &latticefoundry::mc::object::ObjectModule) -> Result<(), String> {
    use latticefoundry::link::gnu::{self, HostCrt};
    let stem = default_output(&opts.inputs[0]);
    let output = opts.output.clone().unwrap_or_else(|| match opts.output_kind {
        OutputKind::Shared => format!("lib{stem}.so"),
        _ => stem.clone(),
    });
    let elf = latticefoundry::mc::elf::write(obj);
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

/// Link a Cortex-M program with qld: the compiled object plus a generated
/// vector table and reset handler calling `entry`, flash at `--base` (default
/// 0), into an ELF executable (`-o`, default `<input>.elf`) or, with
/// `--oformat binary|ihex`, a flashable image of its loadable contents.
fn link_cortex_m(
    opts: &BuildOptions,
    obj: ObjectModule,
    entry: &str,
    format: Option<RawFormat>,
    output_or: &dyn Fn(&str) -> String,
) -> Result<(), String> {
    use latticefoundry::target::thumb::firmware;
    let mut layout = firmware::MemoryLayout::default();
    if let Some(b) = opts.base {
        layout.flash_origin = b;
    }
    let startup = firmware::startup_object(entry, 32);
    let Some(format) = format else {
        let out = output_or("elf");
        return firmware::link_elf(&[obj, startup], &layout, &opts.link_extra, Path::new(&out))
            .map_err(|e| format!("link error: {e}"));
    };
    let (ext, fill) = match format {
        RawFormat::Binary => ("bin", 0xff),
        RawFormat::Ihex => ("hex", 0xff),
    };
    let output = output_or(ext);
    let elf_path = format!("{output}.lf-tmp.elf");
    let result = (|| {
        firmware::link_elf(&[obj, startup], &layout, &opts.link_extra, Path::new(&elf_path))
            .map_err(|e| format!("link error: {e}"))?;
        let elf = std::fs::read(&elf_path).map_err(|e| format!("cannot read {elf_path}: {e}"))?;
        let segments = raw::load_segments(&elf)?;
        let bytes = match format {
            RawFormat::Binary => raw::to_binary(&segments, fill)?.1,
            RawFormat::Ihex => raw::to_ihex(&segments, firmware::elf32_entry(&elf))?.into_bytes(),
        };
        std::fs::write(&output, bytes).map_err(|e| format!("cannot write {output}: {e}"))
    })();
    let _ = std::fs::remove_file(&elf_path);
    result
}

/// The static linker core's options from the command line.
fn image_options(opts: &BuildOptions, entry: String) -> ImageOptions {
    let mut image = ImageOptions { debug: opts.debug, entry, ..ImageOptions::default() };
    if let Some(base) = opts.base {
        image.base = base;
    }
    image
}

/// Link a Windows executable with qld's PE driver (MinGW flavor). The image
/// imports nothing: `entry` (by default `main`) is the process entry point,
/// and the value it returns becomes the process exit code.
fn link_pe(obj: &ObjectModule, triple: Triple, entry: &str, output: &str, base: Option<u64>) -> Result<(), String> {
    let bytes = latticefoundry::mc::write_object(obj, triple).map_err(|e| e.to_string())?;
    let emulation = if triple.arch == TargetArch::AArch64 { "arm64pe" } else { "i386pep" };
    let tmp = format!("{output}.lf-tmp.obj");
    std::fs::write(&tmp, bytes).map_err(|e| format!("cannot write {tmp}: {e}"))?;
    let mut args: Vec<String> =
        ["-m", emulation, "--entry", entry, "--subsystem", "console"].map(String::from).to_vec();
    if let Some(b) = base {
        args.push("--image-base".into());
        args.push(format!("{b:#x}"));
    }
    args.extend(["-o".to_owned(), output.to_owned(), tmp.clone()]);
    let result = link::gnu::link_gnu("lf", &args).map_err(|e| format!("link error: {e}"));
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
    let mut target = Triple::default_target();
    let mut device = None;
    let mut format = None;
    let mut oformat = None;
    let mut base = None;
    let mut output_kind = OutputKind::Executable;
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
            "--target" => {
                let t = it.next().ok_or("--target requires a triple")?;
                target = Triple::parse(t).ok_or_else(|| format!("unknown target '{t}'"))?;
                device = target::avr::Device::from_triple(t);
            }
            "-c" => output_kind = OutputKind::Object,
            "--shared" | "-shared" => output_kind = OutputKind::Shared,
            "--pie" | "-pie" => pie = true,
            "--pic" | "-fPIC" | "-fpic" => pic = true,
            "-soname" | "--soname" => {
                soname = Some(it.next().ok_or("-soname requires a name")?.clone());
            }
            flag if flag.starts_with("--soname=") => soname = Some(flag["--soname=".len()..].to_owned()),
            flag if (flag.starts_with("-L") || flag.starts_with("-l")) && flag.len() > 2 => {
                link_extra.push(flag.to_owned());
            }
            "--format" => {
                let f = it.next().ok_or("--format requires elf, coff or macho")?;
                format = Some(ObjectFormat::parse(f).ok_or_else(|| format!("unknown object format '{f}'"))?);
            }
            "--oformat" => {
                let f = it.next().ok_or("--oformat requires elf, binary or ihex")?;
                oformat = match f.as_str() {
                    "elf" => None,
                    other => Some(RawFormat::parse(other).ok_or_else(|| format!("unknown output format '{other}'"))?),
                };
            }
            "--base" => {
                let b = it.next().ok_or("--base requires an address")?;
                base = Some(parse_addr(b).ok_or_else(|| format!("bad address '{b}'"))?);
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
        OutputKind::Executable if pie => output_kind = OutputKind::Pie,
        OutputKind::Object if pie && pic => return Err("--pic and --pie are exclusive".to_owned()),
        _ => {}
    }
    if pic && output_kind != OutputKind::Object && output_kind != OutputKind::Shared {
        return Err("--pic only applies to -c (use --shared for a shared library)".to_owned());
    }
    if soname.is_some() && output_kind != OutputKind::Shared {
        return Err("-soname only applies to --shared".to_owned());
    }
    let cortex_m = target.arch == TargetArch::Thumb && output_kind == OutputKind::Executable;
    if !link_extra.is_empty() && !cortex_m && !matches!(output_kind, OutputKind::Shared | OutputKind::Pie) {
        return Err("-L/-l only apply to --shared, --pie and Cortex-M executables".to_owned());
    }
    if entry.is_some() && matches!(output_kind, OutputKind::Shared | OutputKind::Pie) {
        return Err("--entry only applies to executables linked by lf (not --shared or --pie)".to_owned());
    }
    if matches!(output_kind, OutputKind::Shared | OutputKind::Pie)
        && (target.arch != TargetArch::X86_64 || target.object_format() != ObjectFormat::Elf)
    {
        return Err(format!("--shared and --pie need an x86-64 ELF target, not {target}"));
    }
    if (oformat.is_some() || base.is_some()) && matches!(output_kind, OutputKind::Shared | OutputKind::Pie) {
        return Err("--oformat/--base do not apply to --shared or --pie".to_owned());
    }
    let object_only = output_kind == OutputKind::Object;
    if format.is_some() && !object_only {
        return Err("--format applies to objects: add -c".to_owned());
    }
    if object_only && oformat.is_some() {
        return Err("--oformat applies to executables, not to -c".to_owned());
    }
    if target.arch == TargetArch::Wasm32 && (oformat.is_some() || base.is_some()) {
        return Err("--oformat/--base do not apply to wasm32".to_owned());
    }
    if target.arch == TargetArch::Avr && base.is_some() {
        return Err("--base does not apply to AVR firmware (flash starts at 0)".to_owned());
    }
    if target.os == TargetOs::Darwin && output_kind == OutputKind::Executable {
        return Err(format!("cannot link a {target} executable yet: use -c"));
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
        target,
        device,
        format,
        oformat,
        base,
        output_kind,
        pic,
        pie,
        soname,
        link_extra,
    })
}

/// Parse a decimal or `0x`-prefixed hexadecimal address.
fn parse_addr(s: &str) -> Option<u64> {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
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
