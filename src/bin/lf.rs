//! `lf` — the LatticeFoundry compiler driver.
//!
//! The umbrella front end that ties the other tools together. The `build`
//! subcommand is the Phase-8 end-to-end path: it parses an IR module (`.lf`
//! text or `.lfb` binary), verifies it, lowers it to x86-64 machine code, links
//! it into a **static native executable** with our own linker, and marks it
//! executable — no system linker or libc involved.

use std::path::Path;
use std::process::ExitCode;

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
    println!("  lf --version | --help\n");
    println!("  -O0..-O3       optimization level (default: -O0)");
    println!("  -g / --debug   emit DWARF debug info (source lines, symbols)");
    println!("  --lto          link-time optimize across inputs (implied by 2+ inputs)");
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
    let obj = if opts.debug {
        let comp_dir = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(str::to_owned))
            .unwrap_or_default();
        let file_name = opts.inputs.first().cloned().unwrap_or_default();
        let source = target::x86_64::DebugSource { file_name, comp_dir };
        target::x86_64::compile_module_debug(&module, &syms, &source)
    } else {
        target::x86_64::compile_module(&module, &syms)
    };
    let image_opts = ImageOptions {
        debug: opts.debug,
        entry: opts.entry.clone().unwrap_or_else(|| ImageOptions::default().entry),
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

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" => output = Some(it.next().ok_or("-o requires a path")?.clone()),
            "--entry" | "-e" => entry = Some(it.next().ok_or("--entry requires a name")?.clone()),
            "--no-verify" => verify = false,
            "-g" | "--debug" => debug = true,
            "--lto" => lto = true,
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

    Ok(BuildOptions { inputs, output, entry, verify, debug, opt, lto })
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
