//! Linking Mach-O executables and dynamic libraries for macOS, via the ld64
//! flavor of our own [`qld`] linker.
//!
//! # The executable model
//!
//! A macOS process is started by `dyld`, and every program links
//! `libSystem`: `dyld` initializes it before calling the program, and when
//! the entry point named by `LC_MAIN` returns, `dyld` passes its result to
//! `libSystem`'s `exit`. There is no supported static executable (the kernel
//! system-call interface is private and changes between releases, and qld
//! rejects `-static`), so [`link_macho`] builds the ordinary kind:
//!
//! - `MH_EXECUTE`, position-independent, with `LC_LOAD_DYLINKER`
//!   (`/usr/lib/dyld`), `LC_MAIN` naming the entry function (by default
//!   `main`, called as `main(argc, argv, envp, apple)`; its return value is the
//!   exit status), `LC_LOAD_DYLIB` of `/usr/lib/libSystem.B.dylib`, and an
//!   ad-hoc code signature on arm64 (which the kernel requires);
//! - or with [`MachOutput::Dylib`], an `MH_DYLIB` whose `LC_ID_DYLIB` is the
//!   given install name.
//!
//! The link needs no SDK: unless the caller names `-lSystem` (with `-L` or
//! `-syslibroot` pointing at a real SDK), [`libsystem_stub`] writes a
//! text-based stub (`.tbd`, the format the SDK itself ships) for
//! `libSystem.B.dylib` that exports `_exit` and **every symbol the object
//! leaves undefined** — on macOS the C library, the math library and the
//! system calls all live in `libSystem`, so that is where an LF program's
//! external references go. A symbol the real `libSystem` lacks fails at load
//! time, as with any two-level-namespace binding.

use std::path::Path;

use crate::mc::object::ObjectModule;
use crate::target::{TargetArch, TargetOs, Triple};

/// What [`link_macho`] produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MachOutput {
    /// An executable whose `LC_MAIN` entry is this function (an IR name,
    /// without the Mach-O leading underscore).
    Executable {
        /// The entry function.
        entry: String,
    },
    /// A dynamic library with this install name (`LC_ID_DYLIB`).
    Dylib {
        /// The install name, e.g. `@rpath/libfoo.dylib`.
        install_name: String,
    },
}

/// The ld64 `-arch` name of `arch`.
fn arch_name(arch: TargetArch) -> Result<&'static str, String> {
    match arch {
        TargetArch::X86_64 => Ok("x86_64"),
        TargetArch::AArch64 => Ok("arm64"),
        other => Err(format!("no Mach-O linking for {}", other.name())),
    }
}

/// The macOS deployment target recorded by the objects and the link (11.0,
/// the first release with arm64).
pub const MIN_MACOS: &str = "11.0";

/// A text-based stub (`.tbd`, version 4) for `/usr/lib/libSystem.B.dylib`
/// on `arch`, exporting `_exit` and `symbols` (Mach-O names, with their
/// leading underscore).
pub fn libsystem_stub(arch: TargetArch, symbols: &[String]) -> Result<String, String> {
    let target = format!("{}-macos", arch_name(arch)?);
    let mut names: Vec<&str> = symbols.iter().map(String::as_str).collect();
    names.push("_exit");
    names.sort_unstable();
    names.dedup();
    let quoted: Vec<String> = names.iter().map(|n| format!("'{n}'")).collect();
    Ok(format!(
        "--- !tapi-tbd\n\
         tbd-version:     4\n\
         targets:         [ {target} ]\n\
         install-name:    '/usr/lib/libSystem.B.dylib'\n\
         current-version: 1311\n\
         compatibility-version: 1\n\
         exports:\n  \
         - targets:         [ {target} ]\n    \
         symbols:         [ {} ]\n\
         ...\n",
        quoted.join(", ")
    ))
}

/// Link `obj` (compiled for `arch` on Darwin) into a Mach-O executable or
/// dylib at `output` with qld's ld64 flavor (see the [module docs](self)).
/// `extra` holds `-L<dir>`/`-l<lib>` arguments passed through.
///
/// # Errors
///
/// A Mach-O writer error, an unwritable temporary file, or a failed link.
pub fn link_macho(
    obj: &ObjectModule,
    arch: TargetArch,
    kind: &MachOutput,
    extra: &[String],
    output: &Path,
) -> Result<(), String> {
    let arch_flag = arch_name(arch)?;
    let bytes = crate::mc::write_object(obj, Triple::new(arch, TargetOs::Darwin)).map_err(|e| e.to_string())?;
    let out = output.to_string_lossy().into_owned();
    let tmp_obj = format!("{out}.lf-tmp.o");
    let tmp_tbd = format!("{out}.lf-tmp.tbd");
    let mut args: Vec<String> = vec![
        "-arch".into(),
        arch_flag.into(),
        "-platform_version".into(),
        "macos".into(),
        MIN_MACOS.into(),
        MIN_MACOS.into(),
    ];
    match kind {
        MachOutput::Executable { entry } => args.extend(["-e".to_owned(), format!("_{entry}")]),
        MachOutput::Dylib { install_name } => {
            args.extend(["-dylib".to_owned(), "-install_name".to_owned(), install_name.clone()]);
        }
    }
    args.extend(["-o".to_owned(), out.clone(), tmp_obj.clone()]);
    let own_libsystem = !extra.iter().any(|a| a == "-lSystem");
    if own_libsystem {
        args.push(tmp_tbd.clone());
    }
    args.extend(extra.iter().cloned());

    let result = (|| {
        std::fs::write(&tmp_obj, &bytes).map_err(|e| format!("cannot write {tmp_obj}: {e}"))?;
        if own_libsystem {
            let undefined: Vec<String> =
                obj.symbols().iter().filter(|s| s.is_undefined()).map(|s| format!("_{}", s.name)).collect();
            std::fs::write(&tmp_tbd, libsystem_stub(arch, &undefined)?)
                .map_err(|e| format!("cannot write {tmp_tbd}: {e}"))?;
        }
        link_ld64("lf", &args)
    })();
    let _ = std::fs::remove_file(&tmp_obj);
    let _ = std::fs::remove_file(&tmp_tbd);
    result
}

/// Run an ld64-flavor qld link described by `args` (excluding `argv[0]`),
/// printing diagnostics prefixed with `program`.
///
/// # Errors
///
/// An invalid command line or a failed link.
pub fn link_ld64<S: AsRef<std::ffi::OsStr>>(program: &str, args: &[S]) -> Result<(), String> {
    use qld::diag::{Diagnostic, DiagnosticSink, Severity, Stderr};
    let mut argv: Vec<std::ffi::OsString> = vec!["ld64.qld".into()];
    argv.extend(args.iter().map(|a| a.as_ref().to_owned()));
    let sink = Stderr::new(program);
    match qld::args::parse_darwin(&argv).map_err(|e| e.to_string())? {
        qld::ParseOutcome::Link(options) => {
            for warning in &options.warnings {
                sink.emit(Diagnostic::new(Severity::Warning, warning.clone()));
            }
            qld::link(&options, &sink).map_err(|e| e.to_string())
        }
        qld::ParseOutcome::Help | qld::ParseOutcome::Version => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_lists_the_imports() {
        let tbd = libsystem_stub(TargetArch::AArch64, &["_puts".to_owned(), "_exit".to_owned()]).unwrap();
        assert!(tbd.starts_with("--- !tapi-tbd\n"), "{tbd}");
        assert!(tbd.contains("targets:         [ arm64-macos ]"), "{tbd}");
        assert!(tbd.contains("symbols:         [ '_exit', '_puts' ]"), "{tbd}");
        assert!(libsystem_stub(TargetArch::Riscv64, &[]).is_err());
    }
}
