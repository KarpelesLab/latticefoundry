//! Linking standard ELF objects, archives and shared libraries, via our own
//! [`qld`] linker.
//!
//! The [static linker core](super::image) links LatticeFoundry's own objects
//! into a self-contained executable. Everything beyond that — ELF `.o` inputs
//! from any compiler, `.a` archives, `libc.so` and other shared libraries,
//! dynamic executables, PIE, linker scripts — is `qld`'s job: it accepts a GNU
//! `ld` command line, so this module is a thin bridge onto it. `lf-ld` uses it
//! for non-`.lfo` inputs, and front ends use [`host_c_link_args`] to link
//! against the host's C library without a system compiler driver.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use qld::ParseOutcome;
use qld::diag::{Diagnostic, DiagnosticSink, Severity, Stderr};

/// Run a link described by a GNU `ld` command line (`args` excludes `argv[0]`).
///
/// Diagnostics are printed to standard error prefixed with `program`. `--help`
/// and `--version` print qld's version line and succeed.
///
/// # Errors
///
/// Returns a message when the command line is invalid or the link fails (the
/// details have already been reported to standard error).
pub fn link_gnu<S: AsRef<std::ffi::OsStr>>(program: &str, args: &[S]) -> Result<(), String> {
    let mut argv: Vec<OsString> = vec![OsString::from("ld")];
    argv.extend(args.iter().map(|a| a.as_ref().to_owned()));
    let sink = Stderr::new(program);
    match qld::parse_gnu(&argv).map_err(|e| e.to_string())? {
        ParseOutcome::Link(options) => {
            for warning in &options.warnings {
                sink.emit(Diagnostic::new(Severity::Warning, warning.clone()));
            }
            qld::link(&options, &sink).map_err(|e| e.to_string())
        }
        ParseOutcome::Help | ParseOutcome::Version => {
            println!("{}", qld::version_line());
            Ok(())
        }
    }
}

/// The host C runtime pieces a hosted executable links with.
#[derive(Clone, Debug, Default)]
pub struct HostCrt {
    /// Directory holding `crt1.o`, `crti.o`, `crtn.o` and `libc.so`.
    pub libdir: PathBuf,
    /// The compiler runtime directory with `crtbegin.o`/`crtend.o`, if found.
    pub gcc_libdir: Option<PathBuf>,
    /// The program interpreter (dynamic loader).
    pub dynamic_linker: PathBuf,
}

impl HostCrt {
    /// Find the host's C runtime (x86-64 Linux / glibc layouts).
    ///
    /// Returns `None` when no `crt1.o` is found in the usual library
    /// directories.
    #[must_use]
    pub fn discover() -> Option<Self> {
        let libdir = [
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib",
            "/lib64",
            "/lib/x86_64-linux-gnu",
        ]
        .into_iter()
        .map(PathBuf::from)
        .find(|d| d.join("crt1.o").is_file())?;
        let dynamic_linker = ["/lib64/ld-linux-x86-64.so.2", "/lib/ld-linux-x86-64.so.2"]
            .into_iter()
            .map(PathBuf::from)
            .find(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from("/lib64/ld-linux-x86-64.so.2"));
        Some(Self {
            libdir,
            gcc_libdir: find_gcc_libdir(),
            dynamic_linker,
        })
    }
}

/// The newest `…/gcc/<triple>/<version>/` directory holding `crtbegin.o`.
fn find_gcc_libdir() -> Option<PathBuf> {
    let mut best: Option<(Vec<u32>, PathBuf)> = None;
    for root in ["/usr/lib/gcc", "/usr/lib64/gcc"] {
        let Ok(triples) = std::fs::read_dir(root) else {
            continue;
        };
        for triple in triples.flatten() {
            if !triple.file_name().to_string_lossy().starts_with("x86_64") {
                continue;
            }
            let Ok(versions) = std::fs::read_dir(triple.path()) else {
                continue;
            };
            for v in versions.flatten() {
                let dir = v.path();
                if !dir.join("crtbegin.o").is_file() {
                    continue;
                }
                let key: Vec<u32> = v
                    .file_name()
                    .to_string_lossy()
                    .split('.')
                    .map(|p| p.parse().unwrap_or(0))
                    .collect();
                if best.as_ref().is_none_or(|(k, _)| key > *k) {
                    best = Some((key, dir));
                }
            }
        }
    }
    best.map(|(_, dir)| dir)
}

/// The GNU `ld` command line (without `argv[0]`) linking `objects` into a
/// dynamically linked executable `output` against the host C library, the way
/// a C compiler driver would. `extra` (e.g. `-lm`, `-L<dir>`) goes after the
/// objects and before `-lc`.
#[must_use]
pub fn host_c_link_args(
    crt: &HostCrt,
    objects: &[&Path],
    extra: &[String],
    output: &Path,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = Vec::new();
    let mut push = |a: &dyn AsRef<std::ffi::OsStr>| args.push(a.as_ref().to_owned());
    push(&"-o");
    push(&output);
    push(&"--dynamic-linker");
    push(&crt.dynamic_linker);
    push(&"--eh-frame-hdr");
    push(&crt.libdir.join("crt1.o"));
    push(&crt.libdir.join("crti.o"));
    if let Some(gcc) = &crt.gcc_libdir {
        push(&gcc.join("crtbegin.o"));
        push(&format!("-L{}", gcc.display()));
    }
    push(&format!("-L{}", crt.libdir.display()));
    for obj in objects {
        push(obj);
    }
    for e in extra {
        push(e);
    }
    push(&"-lc");
    if let Some(gcc) = &crt.gcc_libdir {
        push(&"-lgcc");
        push(&gcc.join("crtend.o"));
    }
    push(&crt.libdir.join("crtn.o"));
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::asm::{AsmOptions, AsmSource, assemble};
    use crate::target::TargetArch;

    /// A fresh scratch directory for one test.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lf-gnu-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn assemble_and_link_static_without_libc() {
        // rsasm assembles a raw-syscall `_start`; qld links it into a static
        // executable. No external tool is involved at any step.
        let src = "        .text\n        .globl _start\n_start:\n\
                   \x20       movq $60, %rax\n        movq $7, %rdi\n        syscall\n";
        let obj = assemble(&[AsmSource { name: "exit.s", text: src }], &AsmOptions::new(TargetArch::X86_64))
            .unwrap();
        let dir = scratch("static");
        let (o, exe) = (dir.join("exit.o"), dir.join("exit"));
        std::fs::write(&o, obj).unwrap();
        link_gnu("test", &[OsString::from("-static"), "-o".into(), exe.clone().into(), o.into()])
            .unwrap();
        let status = std::process::Command::new(&exe).status().unwrap();
        assert_eq!(status.code(), Some(7));
    }

    #[test]
    fn backend_object_links_against_host_libc() {
        // Our own x86-64 backend's ELF object calls libc's `abs`; qld links it
        // against the host C runtime. No system compiler driver or linker.
        let Some(crt) = HostCrt::discover() else {
            eprintln!("skipping: no host C runtime (crt1.o) found");
            return;
        };
        let src = "module \"t\"\n\
                   func @abs(i32) -> i32\n\
                   func @main() -> i32 {\n\
                   entry ^0:\n  %0 = sub i32 0, i32 40 : i32\n  %1 = call @abs(%0) : i32\n  \
                   %2 = add %1, i32 2 : i32\n  ret %2\n}\n";
        let mut syms = crate::support::StrInterner::new();
        let m = crate::ir::text::parse_module(src, crate::support::diagnostics::FileId::new(0), &mut syms)
            .unwrap();
        let obj = crate::target::x86_64::compile_module(&m, &syms);
        let dir = scratch("hosted");
        let (o, exe) = (dir.join("t.o"), dir.join("t"));
        std::fs::write(&o, crate::mc::elf::write(&obj)).unwrap();
        link_gnu("test", &host_c_link_args(&crt, &[&o], &[], &exe)).unwrap();
        let status = std::process::Command::new(&exe).status().unwrap();
        assert_eq!(status.code(), Some(42));
    }
}
