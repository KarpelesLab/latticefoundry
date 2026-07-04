//! `lf-as` — the LatticeFoundry assembler.
//!
//! Translates target assembly into relocatable objects via the machine-code
//! layer. See ROADMAP Phase 6.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("lf-as (LatticeFoundry) {}", latticefoundry::VERSION);
            ExitCode::SUCCESS
        }
        None | Some("--help" | "-h") => {
            println!("lf-as — LatticeFoundry assembler\n");
            println!("usage: lf-as [-o output.lfo] <input.s>");
            ExitCode::SUCCESS
        }
        Some(_) => {
            eprintln!("lf-as: assembler not yet implemented — see ROADMAP Phase 6");
            ExitCode::FAILURE
        }
    }
}
