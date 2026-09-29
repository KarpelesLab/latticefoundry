//! Textual assembly → ELF relocatable objects, via our own [`rsasm`] assembler.
//!
//! LatticeFoundry's backends encode machine code directly (no assembly text in
//! the pipeline), so this is the path for assembly written by hand or emitted by
//! another tool: `lf-as` drives it, and front ends can use it for code they only
//! have as text (e.g. inline `asm`). The assembler itself — lexing, the GNU/
//! Intel dialects, encoding, relaxation and the ELF writer — lives in `rsasm`;
//! this module only maps LatticeFoundry's target names onto it and turns its
//! diagnostics into an error string.

use std::path::{Path, PathBuf};

use rsasm::assembler::{Assembler, Options};
use rsasm::output::Format;

use crate::target::TargetArch;

/// Options for [`assemble`].
#[derive(Clone, Debug)]
pub struct AsmOptions {
    /// Target architecture.
    pub arch: TargetArch,
    /// Directories searched by `.include`.
    pub include_paths: Vec<PathBuf>,
    /// Emit DWARF line info mapping the object back to the assembly source.
    pub debug: bool,
}

impl AsmOptions {
    /// Default options for `arch`.
    #[must_use]
    pub fn new(arch: TargetArch) -> Self {
        Self {
            arch,
            include_paths: Vec::new(),
            debug: false,
        }
    }
}

/// One assembly source: a display name (used in diagnostics) and its text.
#[derive(Clone, Copy, Debug)]
pub struct AsmSource<'a> {
    /// Name shown in diagnostics and debug info.
    pub name: &'a str,
    /// The assembly text.
    pub text: &'a str,
}

/// The `rsasm` architecture name for a LatticeFoundry target.
fn rsasm_arch(arch: TargetArch) -> &'static str {
    match arch {
        TargetArch::X86_64 => "x86-64",
        TargetArch::AArch64 => "aarch64",
        TargetArch::Riscv64 => "riscv64",
        // Needs rsasm's `arm` feature; without it `assemble` reports the
        // missing backend.
        TargetArch::Thumb => "thumb",
    }
}

/// Assemble `sources` (in order, as one translation unit) into an ELF
/// relocatable object for `options.arch`.
///
/// # Errors
///
/// Returns rsasm's rendered diagnostics when the source has errors, or a
/// message when the object cannot be written.
pub fn assemble(sources: &[AsmSource<'_>], options: &AsmOptions) -> Result<Vec<u8>, String> {
    let arch = rsasm::arch::lookup(rsasm_arch(options.arch))
        .ok_or_else(|| format!("rsasm has no `{}` backend", rsasm_arch(options.arch)))?;
    let mut opts = Options::new()
        .with_format(Format::Elf)
        .with_dialect(arch.default_dialect());
    for dir in &options.include_paths {
        opts = opts.with_include_path(dir.clone());
    }
    if options.debug {
        opts = opts.with_debug_source(true);
    }
    let mut asm = Assembler::new(arch, opts);
    for src in sources {
        asm.assemble_str(src.name, src.text);
    }
    let ok = asm.finish();
    if !ok || asm.diags().has_errors() {
        let rendered = asm.diags().render(asm.source_map(), false);
        return Err(if rendered.is_empty() {
            "assembly failed".to_owned()
        } else {
            rendered
        });
    }
    rsasm::output::elf::build(&asm).map_err(|e| e.to_string())
}

/// Assemble the file at `path` (see [`assemble`]).
///
/// # Errors
///
/// As [`assemble`], plus a failure to read `path`.
pub fn assemble_file(path: &Path, options: &AsmOptions) -> Result<Vec<u8>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let name = path.to_string_lossy();
    assemble(
        &[AsmSource {
            name: &name,
            text: &text,
        }],
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `.text` section bytes of an ELF64 little-endian relocatable object.
    fn text_bytes(elf: &[u8]) -> Vec<u8> {
        let u16_at = |o: usize| u16::from_le_bytes([elf[o], elf[o + 1]]) as usize;
        let u64_at = |o: usize| u64::from_le_bytes(elf[o..o + 8].try_into().unwrap()) as usize;
        let u32_at = |o: usize| u32::from_le_bytes(elf[o..o + 4].try_into().unwrap()) as usize;
        let (shoff, shentsize, shnum, shstrndx) =
            (u64_at(0x28), u16_at(0x3a), u16_at(0x3c), u16_at(0x3e));
        let sh = |i: usize| shoff + i * shentsize;
        let strtab = u64_at(sh(shstrndx) + 0x18);
        for i in 0..shnum {
            let name_off = strtab + u32_at(sh(i));
            if elf[name_off..].starts_with(b".text\0") {
                let (off, size) = (u64_at(sh(i) + 0x18), u64_at(sh(i) + 0x20));
                return elf[off..off + size].to_vec();
            }
        }
        panic!("no .text section");
    }

    #[test]
    fn assembles_x86_64_att() {
        let obj = assemble(
            &[AsmSource {
                name: "t.s",
                text: "movq %rbx, %rax\nret\n",
            }],
            &AsmOptions::new(TargetArch::X86_64),
        )
        .unwrap();
        assert_eq!(&obj[..4], b"\x7fELF");
        assert_eq!(obj[4], 2, "ELFCLASS64");
        assert_eq!(text_bytes(&obj), [0x48, 0x89, 0xd8, 0xc3]);
    }

    #[test]
    fn assembles_aarch64_and_riscv64() {
        let a64 = assemble(
            &[AsmSource {
                name: "t.s",
                text: "ret\n",
            }],
            &AsmOptions::new(TargetArch::AArch64),
        )
        .unwrap();
        assert_eq!(text_bytes(&a64), [0xc0, 0x03, 0x5f, 0xd6]);
        // rsasm assumes RV64GC; `norvc` keeps the 32-bit (RV64IM) encoding.
        let rv = assemble(
            &[AsmSource {
                name: "t.s",
                text: ".option norvc\naddi a0, a0, 5\n",
            }],
            &AsmOptions::new(TargetArch::Riscv64),
        )
        .unwrap();
        assert_eq!(text_bytes(&rv), [0x13, 0x05, 0x55, 0x00]);
    }

    #[test]
    fn reports_diagnostics() {
        let err = assemble(
            &[AsmSource {
                name: "bad.s",
                text: "notaninstruction %rax\n",
            }],
            &AsmOptions::new(TargetArch::X86_64),
        )
        .unwrap_err();
        assert!(
            err.contains("bad.s"),
            "diagnostic should name the file: {err}"
        );
    }
}
