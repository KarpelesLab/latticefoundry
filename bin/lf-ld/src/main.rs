//! `lf-ld` — the LatticeFoundry linker.
//!
//! Resolves symbols and relocations across relocatable objects and archives
//! and writes an executable or shared object. See ROADMAP Phase 8.

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
            println!("usage: lf-ld [-o output] <inputs...>");
            return ExitCode::SUCCESS;
        }
        _ => {}
    }

    let options = parse(&args);
    match link::link(&options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("lf-ld: {err}");
            ExitCode::FAILURE
        }
    }
}

fn parse(args: &[String]) -> LinkOptions {
    let mut options = LinkOptions { output: "a.out".to_owned(), inputs: Vec::new() };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" => {
                if let Some(out) = it.next() {
                    options.output = out.clone();
                }
            }
            input => options.inputs.push(input.to_owned()),
        }
    }
    options
}
