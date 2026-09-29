//! `lf-as` — the LatticeFoundry assembler.
//!
//! Assembles target assembly (GNU syntax by default) into an ELF relocatable
//! object, using our own `rsasm` assembler through
//! [`latticefoundry::mc::asm`].

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use latticefoundry::mc::asm::{AsmOptions, AsmSource, assemble};
use latticefoundry::target::TargetArch;

fn usage() {
    println!("lf-as — LatticeFoundry assembler (rsasm)\n");
    println!("usage: lf-as [--arch <arch>] [-g] [-I <dir>] [-o <out.o>] <input.s...>\n");
    println!("options:");
    println!("  --arch <arch>  x86_64 (default), aarch64 or riscv64");
    println!("  -g             emit DWARF line info for the assembly source");
    println!("  -I <dir>       add a directory searched by `.include`");
    println!("  -o <path>      output object (default: a.out)");
    println!("Multiple inputs are assembled as one translation unit; `-` reads stdin.");
}

fn parse_arch(name: &str) -> Option<TargetArch> {
    match name {
        "x86_64" | "x86-64" | "amd64" => Some(TargetArch::X86_64),
        "aarch64" | "arm64" => Some(TargetArch::AArch64),
        "riscv64" | "rv64" => Some(TargetArch::Riscv64),
        _ => None,
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let mut options = AsmOptions::new(TargetArch::X86_64);
    let mut output = PathBuf::from("a.out");
    let mut inputs: Vec<String> = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" => output = PathBuf::from(it.next().ok_or("-o requires a path")?),
            "--arch" | "-arch" => {
                let name = it.next().ok_or("--arch requires a name")?;
                options.arch = parse_arch(name).ok_or_else(|| format!("unknown arch `{name}`"))?;
            }
            "-g" => options.debug = true,
            "-I" => options
                .include_paths
                .push(PathBuf::from(it.next().ok_or("-I requires a directory")?)),
            _ if arg.starts_with("--arch=") => {
                let name = &arg["--arch=".len()..];
                options.arch = parse_arch(name).ok_or_else(|| format!("unknown arch `{name}`"))?;
            }
            _ if arg.starts_with("-I") => options.include_paths.push(PathBuf::from(&arg[2..])),
            flag if flag.starts_with('-') && flag != "-" => {
                return Err(format!("unrecognized option `{flag}`"));
            }
            input => inputs.push(input.to_owned()),
        }
    }
    if inputs.is_empty() {
        return Err("no input files".to_owned());
    }
    let mut texts = Vec::new();
    for input in &inputs {
        let text = if input == "-" {
            std::io::read_to_string(std::io::stdin()).map_err(|e| format!("stdin: {e}"))?
        } else {
            std::fs::read_to_string(Path::new(input))
                .map_err(|e| format!("cannot read {input}: {e}"))?
        };
        texts.push(text);
    }
    let sources: Vec<AsmSource<'_>> = inputs
        .iter()
        .zip(&texts)
        .map(|(name, text)| AsmSource {
            name: if name == "-" { "<stdin>" } else { name },
            text,
        })
        .collect();
    let object = assemble(&sources, &options)?;
    std::fs::write(&output, object).map_err(|e| format!("cannot write {}: {e}", output.display()))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("lf-as (LatticeFoundry) {}", latticefoundry::VERSION);
            return ExitCode::SUCCESS;
        }
        None | Some("--help" | "-h") => {
            usage();
            return ExitCode::SUCCESS;
        }
        _ => {}
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprint!("lf-as: {err}");
            if !err.ends_with('\n') {
                eprintln!();
            }
            ExitCode::FAILURE
        }
    }
}
