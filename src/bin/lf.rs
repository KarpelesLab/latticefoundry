//! `lf` — the LatticeFoundry compiler driver.
//!
//! The umbrella front end that ties the other tools together. This is a
//! scaffold: subcommands are implemented as the ROADMAP phases land.

use std::process::ExitCode;

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
        Some(other) => {
            eprintln!(
                "lf: unrecognized subcommand '{other}' (not yet implemented — see ROADMAP)"
            );
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("lf — LatticeFoundry compiler driver\n");
    println!("usage: lf <subcommand> [options]\n");
    println!("options:");
    println!("  -V, --version    print version and exit");
    println!("  -h, --help       print this help and exit\n");
    println!("This is an early scaffold; subcommands land per ROADMAP phases.");
}
