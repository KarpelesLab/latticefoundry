//! `lf-dis` — the LatticeFoundry disassembler.
//!
//! Decodes machine code from an object or raw byte stream back into target
//! assembly. See ROADMAP Phase 6.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("lf-dis (LatticeFoundry) {}", latticefoundry::VERSION);
            ExitCode::SUCCESS
        }
        None | Some("--help" | "-h") => {
            println!("lf-dis — LatticeFoundry disassembler\n");
            println!("usage: lf-dis <input.lfo>");
            ExitCode::SUCCESS
        }
        Some(_) => {
            eprintln!("lf-dis: disassembler not yet implemented — see ROADMAP Phase 6");
            ExitCode::FAILURE
        }
    }
}
