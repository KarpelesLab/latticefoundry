//! The `lf-cc` driver: compile a C file to a native x86-64 executable.
//!
//! `lf-cc [-O0..-O3] [-g] [-o out] [-S|--emit-lf] foo.c`
//!
//! Mirrors `lf build`: lex/parse/sema → lower to IR → verify → optimize →
//! x86-64 compile → link → write an executable and `chmod +x` it. `-S` /
//! `--emit-lf` dumps the lowered `.lf` IR instead of compiling.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use latticefoundry::ir::text;
use latticefoundry::link;
use latticefoundry::support::diagnostics::{Diagnostic, Severity};
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

struct Options {
    input: String,
    output: Option<String>,
    opt: OptLevel,
    debug: bool,
    emit_lf: bool,
    emit_obj: bool,
    std: CStd,
    include_dirs: Vec<PathBuf>,
    cmdline: Vec<MacroOp>,
    nostdinc: bool,
}

impl Options {
    fn pp_options(&self) -> PpOptions {
        PpOptions {
            std: self.std,
            include_dirs: self.include_dirs.clone(),
            cmdline: self.cmdline.clone(),
            main_file_name: self.input.clone(),
            builtin_headers: !self.nostdinc,
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") || args.is_empty() {
        print_usage();
        return Ok(());
    }
    let opts = parse_args(args)?;

    let source = std::fs::read_to_string(&opts.input)
        .map_err(|e| format!("cannot read {}: {e}", opts.input))?;
    let module_name = Path::new(&opts.input)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("module")
        .to_owned();

    let pp = opts.pp_options();

    // `-S` / `--emit-lf`: lower and dump the IR, then stop.
    if opts.emit_lf {
        let (module, syms) = lf_cc::compile_to_ir_with(&source, &module_name, &pp, opts.debug)
            .map_err(|diags| render_diags(&opts.input, &source, &diags))?;
        let out = text::print_module(&module, &syms);
        match &opts.output {
            Some(path) => {
                std::fs::write(path, out).map_err(|e| format!("cannot write {path}: {e}"))?
            }
            None => print!("{out}"),
        }
        return Ok(());
    }

    // `-c`: compile to a relocatable ELF object (for linking against libc with cc).
    if opts.emit_obj {
        let obj = lf_cc::build_object_with(&source, &opts.input, &pp, opts.opt, opts.debug)
            .map_err(|e| match e {
                BuildError::Frontend(diags) => render_diags(&opts.input, &source, &diags),
                BuildError::Backend(msg) => msg,
            })?;
        let output = opts.output.clone().unwrap_or_else(|| {
            let stem = Path::new(&opts.input).file_stem().and_then(|s| s.to_str()).unwrap_or("a");
            format!("{stem}.o")
        });
        std::fs::write(&output, obj).map_err(|e| format!("cannot write {output}: {e}"))?;
        return Ok(());
    }

    // Full build: front end → IR → verify → optimize → codegen → link.
    let image = lf_cc::build_image_with(&source, &opts.input, &pp, opts.opt, opts.debug)
        .map_err(|e| match e {
            BuildError::Frontend(diags) => render_diags(&opts.input, &source, &diags),
            BuildError::Backend(msg) => msg,
        })?;

    let output = opts.output.clone().unwrap_or_else(|| default_output(&opts.input));
    link::write_executable(&output, &image)?;
    Ok(())
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut opt = OptLevel::O0;
    let mut debug = false;
    let mut emit_lf = false;
    let mut emit_obj = false;
    let mut std = CStd::default();
    let mut include_dirs: Vec<PathBuf> = Vec::new();
    let mut cmdline: Vec<MacroOp> = Vec::new();
    let mut nostdinc = false;

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let arg = arg.as_str();
        match arg {
            "-o" => output = Some(it.next().ok_or("-o requires a path")?.clone()),
            "-g" | "--debug" => debug = true,
            "-S" | "--emit-lf" => emit_lf = true,
            "-c" => emit_obj = true,
            "-nostdinc" => nostdinc = true,
            "-I" => include_dirs.push(PathBuf::from(it.next().ok_or("-I requires a directory")?)),
            "-D" => cmdline.push(MacroOp::Define(it.next().ok_or("-D requires a name")?.clone())),
            "-U" => cmdline.push(MacroOp::Undef(it.next().ok_or("-U requires a name")?.clone())),
            _ if arg.starts_with("-I") => include_dirs.push(PathBuf::from(&arg[2..])),
            _ if arg.starts_with("-D") => cmdline.push(MacroOp::Define(arg[2..].to_owned())),
            _ if arg.starts_with("-U") => cmdline.push(MacroOp::Undef(arg[2..].to_owned())),
            _ if arg.starts_with("--std=") || arg.starts_with("-std=") => {
                let name = arg.split_once('=').map(|(_, v)| v).unwrap_or("");
                std = CStd::parse(name)
                    .ok_or_else(|| format!("unknown -std value '{name}'"))?;
            }
            tok if OptLevel::parse_flag(tok).is_some() => {
                opt = OptLevel::parse_flag(tok).expect("checked");
            }
            flag if flag.starts_with('-') && flag != "-" => {
                return Err(format!("unrecognized option '{flag}'"));
            }
            positional => {
                if input.is_some() {
                    return Err("only one input file is supported".to_owned());
                }
                input = Some(positional.to_owned());
            }
        }
    }

    let input = input.ok_or("no input file (see `lf-cc --help`)")?;
    Ok(Options { input, output, opt, debug, emit_lf, emit_obj, std, include_dirs, cmdline, nostdinc })
}

fn print_usage() {
    println!("lf-cc — a C frontend for LatticeFoundry\n");
    println!("usage:");
    println!(
        "  lf-cc [-O0|-O1|-O2|-O3] [-g] [--std=<std>] [-I<dir>] [-D<m>] [-U<m>] \
         [-o <out>] [-S|--emit-lf] <file.c>\n"
    );
    println!("  -O0..-O3       optimization level (default: -O0)");
    println!("  -g / --debug   emit DWARF debug info (source lines)");
    println!("  --std=<std>    C standard: c89/c99/c11/c17/c23 or gnuNN (default: gnu17)");
    println!("  -I <dir>       add a directory to the #include search path (repeatable)");
    println!("  -nostdinc      do not consult the builtin freestanding standard headers");
    println!("  -D name[=val]  predefine a macro (repeatable)");
    println!("  -U name        undefine a macro (repeatable)");
    println!("  -S / --emit-lf dump the lowered .lf IR instead of an executable");
    println!("  -o <out>       output path (default: the input stem)");
}

fn default_output(input: &str) -> String {
    Path::new(input)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| "a.out".to_owned())
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
