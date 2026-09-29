//! `lf-ld` — the LatticeFoundry linker.
//!
//! Two linkers behind one driver:
//!
//! - when every input is one of our own `.lfo` objects, the LatticeFoundry
//!   static linker core ([`latticefoundry::link`]) produces a self-contained
//!   static executable;
//! - otherwise — ELF `.o` objects, `.a` archives, `-l` libraries, shared
//!   objects, or any other GNU `ld` option — the whole command line goes to
//!   our GNU-ld-compatible linker `qld` ([`latticefoundry::link::gnu`]).

use std::process::ExitCode;

use latticefoundry::link::{self, LinkOptions};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("lf-ld (LatticeFoundry) {}", latticefoundry::VERSION);
            return ExitCode::SUCCESS;
        }
        None | Some("--help" | "-h") => {
            println!("lf-ld — LatticeFoundry linker\n");
            println!("usage: lf-ld [-o output] [-e entry] <inputs.lfo...>");
            println!("       lf-ld <GNU ld command line>\n");
            println!("With only `.lfo` inputs (and -o/-e), links a static ELF64 executable");
            println!("with the LatticeFoundry linker core:");
            println!("  -o <path>   output executable path (default: a.out)");
            println!("  -e <name>   entry symbol _start calls (default: main)");
            println!("Anything else (ELF objects, archives, -l, shared libraries, other");
            println!("options) is linked by qld, which accepts the GNU ld command line.");
            return ExitCode::SUCCESS;
        }
        _ => {}
    }

    let result = match parse(&args) {
        Some(options) => link::link(&options),
        None => link::gnu::link_gnu("lf-ld", &args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("lf-ld: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Parse a native (`.lfo`-only) link; `None` means "hand it to qld".
fn parse(args: &[String]) -> Option<LinkOptions> {
    let mut options =
        LinkOptions { output: "a.out".to_owned(), inputs: Vec::new(), entry: None };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" => {
                if let Some(out) = it.next() {
                    options.output = out.clone();
                }
            }
            "-e" => {
                if let Some(entry) = it.next() {
                    options.entry = Some(entry.clone());
                }
            }
            input if input.ends_with(".lfo") => options.inputs.push(input.to_owned()),
            _ => return None,
        }
    }
    (!options.inputs.is_empty()).then_some(options)
}
