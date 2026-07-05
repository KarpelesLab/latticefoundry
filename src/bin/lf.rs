//! `lf` — the LatticeFoundry compiler driver.
//!
//! The umbrella front end that ties the other tools together. The `build`
//! subcommand is the Phase-8 end-to-end path: it parses an IR module (`.lf`
//! text or `.lfb` binary), verifies it, lowers it to x86-64 machine code, links
//! it into a **static native executable** with our own linker, and marks it
//! executable — no system linker or libc involved.

use std::path::Path;
use std::process::ExitCode;

use latticefoundry::ir::{Module, binary, text};
use latticefoundry::link::{self, ImageOptions};
use latticefoundry::support::StrInterner;
use latticefoundry::support::diagnostics::{Diagnostic, FileId, Severity};
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
    println!("  lf build <input.lf|input.lfb> [-o <out>] [--entry <name>] [--no-verify]");
    println!("  lf --version | --help\n");
    println!("`lf build` compiles an IR module to a static native executable.");
}

struct BuildOptions {
    input: String,
    output: Option<String>,
    entry: Option<String>,
    verify: bool,
}

fn build(args: &[String]) -> Result<(), String> {
    let opts = parse_build(args)?;

    // Parse the IR module in whichever encoding the extension names.
    let mut syms = StrInterner::new();
    let module = load(&opts.input, &mut syms)?;

    // Verify (Structural tier) unless suppressed.
    if opts.verify
        && let Err(diags) = verify::verify_module(&module)
    {
        for d in &diags {
            eprintln!("{}", render(d));
        }
        let errs = diags.iter().filter(|d| d.severity == Severity::Error).count();
        return Err(format!("verification failed ({errs} error(s))"));
    }

    // Lower to a relocatable object, then link into a static executable.
    let obj = target::x86_64::compile_module(&module, &syms);
    let mut image_opts = ImageOptions::default();
    if let Some(entry) = &opts.entry {
        image_opts.entry = entry.clone();
    }
    let image =
        link::link_executable(vec![obj], &image_opts).map_err(|e| format!("link error: {e}"))?;

    let output = opts.output.unwrap_or_else(|| default_output(&opts.input));
    link::write_executable(&output, &image)?;
    Ok(())
}

fn parse_build(args: &[String]) -> Result<BuildOptions, String> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut entry: Option<String> = None;
    let mut verify = true;

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" => output = Some(it.next().ok_or("-o requires a path")?.clone()),
            "--entry" | "-e" => entry = Some(it.next().ok_or("--entry requires a name")?.clone()),
            "--no-verify" => verify = false,
            flag if flag.starts_with('-') && flag != "-" => {
                return Err(format!("unrecognized option '{flag}'"));
            }
            positional => {
                if input.replace(positional.to_owned()).is_some() {
                    return Err("more than one input file given".to_owned());
                }
            }
        }
    }

    Ok(BuildOptions {
        input: input.ok_or("no input file (see `lf --help`)")?,
        output,
        entry,
        verify,
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
