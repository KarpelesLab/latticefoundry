//! `lf-opt` — the LatticeFoundry IR optimizer driver.
//!
//! Loads a `.lf` IR module, runs a configurable pass pipeline over it, and
//! writes the transformed module back out. See ROADMAP Phases 2–4.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("lf-opt (LatticeFoundry) {}", latticefoundry::VERSION);
            ExitCode::SUCCESS
        }
        None | Some("--help" | "-h") => {
            println!("lf-opt — LatticeFoundry IR optimizer\n");
            println!("usage: lf-opt [-p pass,pass,...] <input.lf>");
            ExitCode::SUCCESS
        }
        Some(_) => {
            eprintln!("lf-opt: IR parser/passes not yet implemented — see ROADMAP Phases 2-4");
            ExitCode::FAILURE
        }
    }
}
