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
        TargetArch::Wasm32 => "wasm32",
        // rsasm has no AVR backend: `assemble` reports that.
        TargetArch::Avr => "avr",
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

/// What a relocation of an [`AsmFragment`] refers to.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FragmentTarget {
    /// A symbol the fragment does not define (an external function or
    /// global), by name.
    External(String),
    /// A location inside the fragment's own code, by offset (a label the
    /// fragment defines, or its section symbol), to be added to the addend.
    Local(u64),
}

/// One relocation of an [`AsmFragment`]'s code.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FragmentReloc {
    /// The offset of the patched field within the fragment.
    pub offset: u64,
    /// The ELF relocation type (`R_X86_64_PC32`, ...) of the target.
    pub kind: u32,
    /// What it refers to.
    pub target: FragmentTarget,
    /// The RELA addend.
    pub addend: i64,
}

/// A piece of assembly assembled on its own, to be spliced into a function
/// being encoded (inline asm, `docs/ir-design.md` §6j): its `.text` bytes and
/// the relocations they need.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct AsmFragment {
    /// The code.
    pub bytes: Vec<u8>,
    /// Its relocations, by increasing offset.
    pub relocs: Vec<FragmentReloc>,
}

/// Assemble `text` (named `name` in diagnostics) for `arch` into a code
/// fragment: the `.text` bytes and their relocations, read back from the
/// relocatable ELF object rsasm writes. Labels the text defines are resolved
/// inside it; a reference to one that still needs a relocation (an absolute
/// address) comes back as a [`FragmentTarget::Local`] offset.
///
/// # Errors
///
/// rsasm's diagnostics; or the text puts anything in another allocated
/// section (`.data`, `.pushsection ...`), or defines a global symbol, neither
/// of which a fragment spliced into a function can carry.
pub fn assemble_fragment(arch: TargetArch, name: &str, text: &str) -> Result<AsmFragment, String> {
    let elf = assemble(&[AsmSource { name, text }], &AsmOptions::new(arch))?;
    read_fragment(&elf)
}

/// Read the `.text` bytes and relocations of an ELF64 little-endian
/// relocatable object (see [`assemble_fragment`]).
fn read_fragment(elf: &[u8]) -> Result<AsmFragment, String> {
    const SHT_SYMTAB: u32 = 2;
    const SHT_RELA: u32 = 4;
    const SHT_NOBITS: u32 = 8;
    const SHF_ALLOC: u64 = 2;
    const SHN_UNDEF: u16 = 0;
    const SHN_ABS: u16 = 0xfff1;
    const STB_GLOBAL: u8 = 1;
    const STB_WEAK: u8 = 2;
    let bad = || "the assembler produced a malformed object".to_owned();
    let get = |o: usize, n: usize| elf.get(o..o + n).ok_or_else(bad);
    let u16_at = |o: usize| get(o, 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u32_at = |o: usize| get(o, 4).map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")));
    let u64_at = |o: usize| get(o, 8).map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")));
    if elf.len() < 64 || &elf[..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return Err(bad());
    }
    let shoff = u64_at(0x28)? as usize;
    let shentsize = u16_at(0x3a)? as usize;
    let shnum = u16_at(0x3c)? as usize;
    let shstrndx = u16_at(0x3e)? as usize;
    struct Sh {
        name: u32,
        ty: u32,
        flags: u64,
        off: usize,
        size: usize,
        link: usize,
        info: usize,
    }
    let mut shs = Vec::with_capacity(shnum);
    for i in 0..shnum {
        let b = shoff + i * shentsize;
        shs.push(Sh {
            name: u32_at(b)?,
            ty: u32_at(b + 4)?,
            flags: u64_at(b + 8)?,
            off: u64_at(b + 0x18)? as usize,
            size: u64_at(b + 0x20)? as usize,
            link: u32_at(b + 0x28)? as usize,
            info: u32_at(b + 0x2c)? as usize,
        });
    }
    let strtab_str = |tab: &Sh, at: u32| -> Result<String, String> {
        let start = tab.off + at as usize;
        let bytes = elf.get(start..tab.off + tab.size).ok_or_else(bad)?;
        let end = bytes.iter().position(|&c| c == 0).ok_or_else(bad)?;
        Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
    };
    let shstr = shs.get(shstrndx).ok_or_else(bad)?;
    let mut text = None;
    for (i, sh) in shs.iter().enumerate() {
        let name = strtab_str(shstr, sh.name)?;
        if name == ".text" {
            text = Some(i);
        } else if sh.flags & SHF_ALLOC != 0 && sh.size > 0 {
            return Err(format!("inline asm may only emit code into the function, not into section `{name}`"));
        }
    }
    let Some(text) = text else { return Ok(AsmFragment::default()) };
    let tsh = &shs[text];
    let bytes = if tsh.ty == SHT_NOBITS { Vec::new() } else { get(tsh.off, tsh.size)?.to_vec() };

    // The symbol table: a defined global is refused; every referenced symbol
    // is resolved to an external name or an offset in `.text`.
    let mut relocs = Vec::new();
    for sh in &shs {
        if sh.ty == SHT_SYMTAB {
            let strtab = shs.get(sh.link).ok_or_else(bad)?;
            for k in 0..sh.size / 24 {
                let s = sh.off + k * 24;
                let bind = get(s + 4, 1)?[0] >> 4;
                let shndx = u16_at(s + 6)?;
                if matches!(bind, STB_GLOBAL | STB_WEAK) && shndx != SHN_UNDEF {
                    let name = strtab_str(strtab, u32_at(s)?)?;
                    return Err(format!("inline asm may not define the global symbol `{name}`"));
                }
            }
        }
    }
    for sh in &shs {
        if sh.ty != SHT_RELA || sh.info != text {
            continue;
        }
        let symtab = shs.get(sh.link).ok_or_else(bad)?;
        let strtab = shs.get(symtab.link).ok_or_else(bad)?;
        for k in 0..sh.size / 24 {
            let r = sh.off + k * 24;
            let offset = u64_at(r)?;
            let info = u64_at(r + 8)?;
            let addend = u64_at(r + 16)? as i64;
            let sym = (info >> 32) as usize;
            let s = symtab.off + sym * 24;
            let shndx = u16_at(s + 6)?;
            let value = u64_at(s + 8)?;
            let target = match shndx {
                SHN_UNDEF if sym != 0 => FragmentTarget::External(strtab_str(strtab, u32_at(s)?)?),
                _ if usize::from(shndx) == text => FragmentTarget::Local(value),
                SHN_UNDEF | SHN_ABS => {
                    return Err("inline asm: a relocation against an absolute address".to_owned());
                }
                _ => return Err("inline asm: a reference into another section".to_owned()),
            };
            relocs.push(FragmentReloc { offset, kind: info as u32, target, addend });
        }
    }
    relocs.sort_by_key(|r| r.offset);
    Ok(AsmFragment { bytes, relocs })
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
