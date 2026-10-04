//! `lf-dis` — the LatticeFoundry disassembler.
//!
//! Decodes the machine code of object files, executables and wasm modules
//! (ELF, `.lfo`, COFF/PE, Mach-O, wasm — the format and architecture come
//! from the header) or of a flat binary (`--raw --arch A`) back into
//! assembly, with symbol labels and inline relocation notes. See
//! [`latticefoundry::mc::disasm`] and ROADMAP Phase 6.

use std::process::ExitCode;

use latticefoundry::mc::disasm::listing::{ListingOptions, list};
use latticefoundry::mc::disasm::{Syntax, objfile, parse_arch};
use latticefoundry::target::TargetArch;

const USAGE: &str = "\
lf-dis — LatticeFoundry disassembler

usage: lf-dis [options] <file>...

  -d, --disassemble      disassemble the code sections (the default)
  -D, --disassemble-all  disassemble every section with contents
  --arch A               the architecture: x86_64, aarch64, riscv64, thumb,
                         avr, wasm32 (required with --raw; overrides the header)
  --syntax att|intel     x86 syntax (default att; also -M intel, -M att)
  --raw                  the input is a flat binary of --arch code
  --base ADDR            the load address of a --raw binary (default 0)
  --start ADDR           only instructions at or after ADDR
  --stop ADDR            only instructions before ADDR
  --no-show-raw-insn     omit the instruction bytes
  --no-relocs            omit the inline relocation notes
  -V, --version          print the version

Formats: ELF (32/64), .lfo, COFF and PE, Mach-O, WebAssembly. Addresses take
decimal or 0x-prefixed hex.";

struct Args {
    files: Vec<String>,
    arch: Option<TargetArch>,
    raw: bool,
    base: u64,
    opts: ListingOptions,
}

fn parse_addr(s: &str) -> Result<u64, String> {
    let r = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16),
        None => s.parse(),
    };
    r.map_err(|_| format!("invalid address `{s}`"))
}

fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut a = Args { files: Vec::new(), arch: None, raw: false, base: 0, opts: ListingOptions::new() };
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        // `--opt=value` and `--opt value` are both accepted.
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n, Some(v.to_owned())),
            _ => (arg.as_str(), None),
        };
        let mut value = |what: &str| -> Result<String, String> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => it.next().cloned().ok_or_else(|| format!("{what} needs a value")),
            }
        };
        match name {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("lf-dis (LatticeFoundry) {}", latticefoundry::VERSION);
                return Ok(None);
            }
            "-d" | "--disassemble" => {}
            "-D" | "--disassemble-all" => a.opts.all_sections = true,
            "--arch" | "--triple" | "-m" => {
                let v = value("--arch")?;
                let first = v.split('-').next().unwrap_or(&v);
                a.arch = Some(parse_arch(first).ok_or_else(|| format!("unknown architecture `{v}`"))?);
            }
            "--syntax" | "-M" | "--x86-asm-syntax" => {
                let v = value("--syntax")?;
                a.opts.disasm.syntax = Syntax::parse(&v).ok_or_else(|| format!("unknown syntax `{v}` (att or intel)"))?;
            }
            "--raw" | "-b" => {
                if name == "-b" {
                    let v = value("-b")?;
                    if v != "binary" {
                        return Err(format!("unsupported input format `{v}` (only `binary`)"));
                    }
                }
                a.raw = true;
            }
            "--base" | "--adjust-vma" => a.base = parse_addr(&value("--base")?)?,
            "--start" | "--start-address" => a.opts.start = Some(parse_addr(&value("--start")?)?),
            "--stop" | "--stop-address" => a.opts.stop = Some(parse_addr(&value("--stop")?)?),
            "--no-show-raw-insn" => a.opts.show_bytes = false,
            "--no-relocs" => a.opts.relocs = false,
            "-r" | "--reloc" => a.opts.relocs = true,
            s if s.starts_with('-') && s.len() > 1 => return Err(format!("unknown option `{s}` (see --help)")),
            _ => a.files.push(arg.clone()),
        }
    }
    if a.files.is_empty() {
        return Err("no input files (see --help)".to_owned());
    }
    if a.raw && a.arch.is_none() {
        return Err("--raw needs --arch".to_owned());
    }
    Ok(Some(a))
}

fn run(argv: &[String]) -> Result<(), String> {
    let Some(args) = parse_args(argv)? else { return Ok(()) };
    for path in &args.files {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let bin = match (args.raw, args.arch) {
            (true, Some(arch)) => objfile::raw(&bytes, arch, args.base),
            _ => objfile::read(&bytes).map_err(|e| format!("{path}: {e}"))?,
        };
        let arch = args.arch.or(bin.arch).ok_or_else(|| {
            format!("{path}: the file does not name a supported architecture; pass --arch")
        })?;
        print!("{}", list(&bin, arch, path, &args.opts));
    }
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match run(&argv) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lf-dis: {e}");
            ExitCode::FAILURE
        }
    }
}
