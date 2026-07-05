//! `lf-opt` — the LatticeFoundry IR optimizer / tool driver.
//!
//! Loads an IR module in either the textual (`.lf`) or binary (`.lfb`) form,
//! runs the structural verifier over it (the `Structural` tier — see
//! `docs/design-tenets.md`), and writes it back out in the requested form.
//! The optimization pass pipeline and the `Refinement` verification tier land
//! in ROADMAP Phases 3–4/9; this is the load → verify → emit spine they plug
//! into.

use std::path::Path;
use std::process::ExitCode;

use latticefoundry::ir::{Module, binary, text};
use latticefoundry::support::StrInterner;
use latticefoundry::support::diagnostics::{Diagnostic, FileId, Severity};
use latticefoundry::transform::pipeline::{self, OptLevel};
use latticefoundry::verify;

/// Output (and, when not otherwise determined, input) IR encoding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Text,
    Binary,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Fast-path the informational flags.
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("lf-opt (LatticeFoundry) {}", latticefoundry::VERSION);
            return ExitCode::SUCCESS;
        }
        None | Some("--help" | "-h") => {
            print_usage();
            return if args.is_empty() { ExitCode::FAILURE } else { ExitCode::SUCCESS };
        }
        _ => {}
    }

    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("lf-opt: {err}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("lf-opt — LatticeFoundry IR tool\n");
    println!("usage: lf-opt [options] <input.lf|input.lfb>\n");
    println!("options:");
    println!("  -O0|-O1|-O2|-O3    run the optimization pipeline at this level");
    println!("  -p <p1,p2,...>     run an explicit comma-separated pass list");
    println!("  -o <path>          write output to <path> (default: stdout)");
    println!("  --emit <lf|lfb>    output encoding (default: inferred from -o, else lf)");
    println!("  --no-verify        skip structural verification");
    println!("  -V, --version      print version and exit");
    println!("  -h, --help         print this help and exit\n");
    println!("passes: mem2reg, sccp, simplify_cfg, dce, egraph, licm, inline");
    println!("With neither -O nor -p, lf-opt just verifies and re-emits the module.");
}

struct Options {
    input: String,
    output: Option<String>,
    emit: Option<Format>,
    verify: bool,
    /// The optimization level to run, if `-O<n>` was given.
    opt: Option<OptLevel>,
    /// An explicit pass list from `-p`, if given (overrides `-O`).
    passes: Option<Vec<String>>,
}

fn run(args: &[String]) -> Result<(), String> {
    let opts = parse_args(args)?;

    // Load the module, threading a single interner through parse/print/encode.
    let mut syms = StrInterner::new();
    let in_fmt = format_of(&opts.input);
    let module = load(&opts.input, in_fmt, &mut syms)?;

    // Verify (Structural tier) unless suppressed.
    if opts.verify {
        verify_or_err(&module, "input")?;
    }

    // Optimize: an explicit `-p` pass list takes precedence over `-O<n>`.
    let mut module = module;
    if let Some(names) = &opts.passes {
        let mut passes = Vec::with_capacity(names.len());
        for n in names {
            let p = pipeline::pass_by_name(n).ok_or_else(|| format!("unknown pass '{n}'"))?;
            passes.push(p);
        }
        pipeline::run_passes(&mut module, passes);
    } else if let Some(level) = opts.opt {
        pipeline::optimize(&mut module, level);
    }

    // Re-verify after transforming (a transform must preserve validity).
    if opts.verify && (opts.passes.is_some() || opts.opt.is_some()) {
        verify_or_err(&module, "optimized")?;
    }

    // Emit in the requested encoding.
    let out_fmt = opts
        .emit
        .or_else(|| opts.output.as_deref().map(format_of))
        .unwrap_or(Format::Text);
    emit(&module, &syms, out_fmt, opts.output.as_deref())
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut emit: Option<Format> = None;
    let mut verify = true;
    let mut opt: Option<OptLevel> = None;
    let mut passes: Option<Vec<String>> = None;

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" => {
                output = Some(it.next().ok_or("-o requires a path")?.clone());
            }
            "--emit" => {
                emit = Some(match it.next().map(String::as_str) {
                    Some("lf") => Format::Text,
                    Some("lfb") => Format::Binary,
                    other => return Err(format!("--emit expects lf|lfb, got {other:?}")),
                });
            }
            "-p" => {
                let list = it.next().ok_or("-p requires a comma-separated pass list")?;
                passes = Some(list.split(',').map(str::trim).map(str::to_owned).collect());
            }
            "--no-verify" => verify = false,
            tok if OptLevel::parse_flag(tok).is_some() => {
                opt = OptLevel::parse_flag(tok);
            }
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

    Ok(Options {
        input: input.ok_or("no input file (see --help)")?,
        output,
        emit,
        verify,
        opt,
        passes,
    })
}

/// Verify `module`, turning any error diagnostics into a driver-level error whose
/// message names the `stage` (e.g. `"input"` or `"optimized"`).
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

/// Infer the encoding from a path's extension; default to text.
fn format_of(path: &str) -> Format {
    match Path::new(path).extension().and_then(|e| e.to_str()) {
        Some("lfb") => Format::Binary,
        _ => Format::Text,
    }
}

fn load(path: &str, fmt: Format, syms: &mut StrInterner) -> Result<Module, String> {
    match fmt {
        Format::Text => {
            let src = read_to_string(path)?;
            let file = FileId::new(0);
            text::parse_module(&src, file, syms).map_err(|diags| {
                let rendered: Vec<String> = diags.iter().map(render).collect();
                format!("parse error in {path}:\n{}", rendered.join("\n"))
            })
        }
        Format::Binary => {
            let bytes = read_bytes(path)?;
            binary::decode(&bytes, syms).map_err(|e| format!("decode error in {path}: {e}"))
        }
    }
}

fn emit(module: &Module, syms: &StrInterner, fmt: Format, output: Option<&str>) -> Result<(), String> {
    match fmt {
        Format::Text => {
            let s = text::print_module(module, syms);
            match output {
                Some(path) => std::fs::write(path, s).map_err(|e| format!("cannot write {path}: {e}")),
                None => {
                    print!("{s}");
                    Ok(())
                }
            }
        }
        Format::Binary => {
            let bytes = binary::encode(module, syms);
            match output {
                Some(path) => {
                    std::fs::write(path, bytes).map_err(|e| format!("cannot write {path}: {e}"))
                }
                None => Err("refusing to write binary .lfb to a terminal; use -o <path>".to_owned()),
            }
        }
    }
}

fn read_to_string(path: &str) -> Result<String, String> {
    if path == "-" {
        use std::io::Read;
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .map_err(|e| format!("cannot read stdin: {e}"))?;
        Ok(s)
    } else {
        std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))
    }
}

fn read_bytes(path: &str) -> Result<Vec<u8>, String> {
    if path == "-" {
        use std::io::Read;
        let mut b = Vec::new();
        std::io::stdin()
            .read_to_end(&mut b)
            .map_err(|e| format!("cannot read stdin: {e}"))?;
        Ok(b)
    } else {
        std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))
    }
}

/// Render a diagnostic for the terminal. Presentation lives here, not in the
/// `support::diagnostics` types (which stay presentation-agnostic).
fn render(d: &Diagnostic) -> String {
    let sev = match d.severity {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
    };
    let mut s = match d.span {
        Some(span) => format!("{sev}[{}..{}]: {}", span.start, span.end, d.message),
        None => format!("{sev}: {}", d.message),
    };
    for note in &d.notes {
        s.push_str(&format!("\n  note: {}", note.message));
    }
    s
}
