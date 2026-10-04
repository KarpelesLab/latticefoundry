//! Linking AArch64 Linux programs and shared libraries with `qld`.
//!
//! The objects are written as ELF64 `EM_AARCH64` relocatable files
//! ([`ElfTarget::AARCH64`]) and linked by our GNU-ld-compatible linker
//! ([`crate::link::gnu`]) under its `aarch64linux` emulation:
//!
//! - [`link_executable`]: a static executable whose entry point is the
//!   `_start` of [`start_object`], which calls the entry function and passes
//!   its return value to the `exit` system call (no C library involved);
//! - [`link_shared`]: a shared library of position-independent code (compile
//!   with [`RelocModel::Pic`](crate::codegen::RelocModel::Pic)), linked with
//!   `-z text` so that a text relocation is an error.

use std::path::{Path, PathBuf};

use crate::mc::elf::ElfTarget;
use crate::mc::object::{ObjectModule, RelocKind, Relocation, Section, SectionKind, Symbol, SymbolBinding, SymbolType};

use super::encode::{bl, movz, svc};

/// The Linux AArch64 `exit` system call number.
const NR_EXIT: u32 = 93;

/// A relocatable object defining `_start`, the process entry point:
///
/// ```text
/// _start: movz x29, #0      // the outermost frame: no caller
///         movz x30, #0
///         bl   entry         // R_AARCH64_CALL26
///         movz x8, #93       // exit(entry's return value, in w0)
///         svc  #0
/// ```
///
/// The kernel starts a process with `sp` 16-byte aligned, as AAPCS64 asks
/// at a call.
pub fn start_object(entry: &str) -> ObjectModule {
    let mut obj = ObjectModule::new("lf-start");
    let text = obj.add_section(Section::new(".text", SectionKind::Text, 4));
    let words = [movz(1, 29, 0, 0), movz(1, 30, 0, 0), bl(0), movz(1, 8, NR_EXIT, 0), svc(0)];
    obj.section_mut(text).bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    obj.add_symbol(Symbol::defined("_start", SymbolBinding::Global, SymbolType::Func, text, 0, 4 * words.len() as u64));
    let callee = obj.reference_symbol(entry);
    obj.add_relocation(Relocation { section: text, offset: 8, symbol: callee, kind: RelocKind::Aarch64Call26, addend: 0 });
    obj
}

/// Write each object as an ELF file next to `output`, run `qld` with
/// `-m aarch64linux`, the `args` built from the staged paths, and remove the
/// staged files.
fn with_staged(
    objects: &[ObjectModule],
    output: &Path,
    args: impl FnOnce(&[PathBuf]) -> Vec<std::ffi::OsString>,
) -> Result<(), String> {
    let stem = output.file_name().map_or_else(|| "a".to_owned(), |s| s.to_string_lossy().into_owned());
    let dir = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let mut staged: Vec<PathBuf> = Vec::new();
    let result = (|| {
        for (k, obj) in objects.iter().enumerate() {
            let bytes = crate::mc::elf::write_with(obj, &ElfTarget::AARCH64).map_err(|e| e.to_string())?;
            let path = dir.join(format!("{stem}.lf-{}.{k}.o", std::process::id()));
            std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            staged.push(path);
        }
        let mut all: Vec<std::ffi::OsString> = vec!["-m".into(), "aarch64linux".into()];
        all.extend(args(&staged));
        crate::link::gnu::link_gnu("lf", &all)
    })();
    for p in staged {
        let _ = std::fs::remove_file(p);
    }
    result
}

/// Link `objects` (plus the [`start_object`] for `entry`) into the static
/// AArch64 Linux executable `output`. `extra` (`-L<dir>`, `-l<lib>`) goes
/// after the objects.
///
/// # Errors
///
/// A message when an object cannot be written or the link fails (qld has
/// printed the details).
pub fn link_executable(objects: Vec<ObjectModule>, entry: &str, extra: &[String], output: &Path) -> Result<(), String> {
    let mut objects = objects;
    objects.push(start_object(entry));
    with_staged(&objects, output, |paths| {
        let mut args: Vec<std::ffi::OsString> =
            ["-static", "-z", "noexecstack", "-e", "_start", "-o"].map(Into::into).to_vec();
        args.push(output.into());
        args.extend(paths.iter().map(Into::into));
        args.extend(extra.iter().map(Into::into));
        args
    })
}

/// Link the position-independent `objects` into the AArch64 Linux shared
/// library `output` (with `DT_SONAME` = `soname` when given; see
/// [`crate::link::gnu::shared_library_args`], whose `-z text` refuses any
/// text relocation).
///
/// # Errors
///
/// As for [`link_executable`].
pub fn link_shared(objects: &[ObjectModule], soname: Option<&str>, extra: &[String], output: &Path) -> Result<(), String> {
    with_staged(objects, output, |paths| {
        let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        crate::link::gnu::shared_library_args(None, &refs, soname, extra, output)
    })
}
