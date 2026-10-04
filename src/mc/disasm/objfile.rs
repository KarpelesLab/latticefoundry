//! Reading the code of object files, executables and modules for
//! disassembly.
//!
//! [`read`] recognizes a file by its header and extracts what a listing
//! needs, in one neutral shape ([`Binary`]): the sections (with their load
//! addresses), the symbols defined in each (labels), and the relocations
//! that patch each (with the format's own relocation type names). Readers
//! are written from the format specifications:
//!
//! - **ELF** (System V gABI): ELF32 and ELF64, either byte order, `REL` and
//!   `RELA` relocations, relocatable objects and linked images (sections, or
//!   the executable `PT_LOAD` segments when the section table is stripped);
//!   machines x86-64, AArch64, RISC-V, Arm and AVR.
//! - **`.lfo`**, LatticeFoundry's own object format ([`crate::mc::lfo`]).
//! - **PE/COFF** (Microsoft PE format specification): COFF objects and PE
//!   images for AMD64, ARM64, Arm Thumb-2 and RISC-V.
//! - **Mach-O**: 64- and 32-bit `MH_OBJECT`s and images for x86-64, arm64 and
//!   Arm.
//! - **WebAssembly** (Core Specification §5; the tool-conventions
//!   `Linking.md`): modules and relocatable objects, with function names from
//!   the `name` section, the `linking` symbol table or the exports, and the
//!   `reloc.CODE` relocations.
//!
//! Every reader is bounds-checked: a truncated or corrupt file is an error
//! (or a partially read file), never a panic.

use crate::mc::object::{ObjectModule, RelocKind, SectionKind, SymbolType, SymbolValue};
use crate::target::TargetArch;
use crate::target::wasm32::leb::{read_i64 as sleb, read_u64 as uleb};

/// A file format [`read`] recognizes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FileFormat {
    /// ELF (32- or 64-bit).
    Elf,
    /// LatticeFoundry's `.lfo`.
    Lfo,
    /// A COFF object.
    Coff,
    /// A PE image (`MZ` … `PE\0\0`).
    Pe,
    /// Mach-O.
    MachO,
    /// A WebAssembly module or relocatable object.
    Wasm,
    /// A flat binary (see [`raw`]).
    Raw,
}

/// What a label marks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LabelKind {
    /// A function (`STT_FUNC`, a COFF function, a wasm function).
    Function,
    /// A data object.
    Object,
    /// Any other named location.
    Other,
    /// An Arm/AArch64 mapping symbol: what follows is code (`$x`, `$a`, `$t`)
    /// or data (`$d`).
    Mapping(MappingKind),
}

/// The kind of bytes an Arm/AArch64 mapping symbol introduces.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MappingKind {
    /// Instructions (`$x`, `$t`, `$a`).
    Code,
    /// Data (`$d`).
    Data,
}

/// A named address.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Label {
    /// The address (in the section's address space, see [`CodeSection::addr`]).
    pub addr: u64,
    /// The name.
    pub name: String,
    /// What it marks.
    pub kind: LabelKind,
}

/// A relocation patching a section.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Reloc {
    /// The address of the patched field.
    pub addr: u64,
    /// The format's name for the relocation type (`R_X86_64_PLT32`,
    /// `IMAGE_REL_ARM64_BRANCH26`, `ARM64_RELOC_PAGE21`, ...).
    pub kind: String,
    /// The target symbol (or section) name.
    pub symbol: String,
    /// The explicit addend, where the format records one.
    pub addend: Option<i64>,
}

impl Reloc {
    /// `KIND symbol±addend`, the form of an inline relocation note.
    pub fn note(&self) -> String {
        match self.addend {
            Some(a) if a > 0 => format!("{} {}+{:#x}", self.kind, self.symbol, a),
            Some(a) if a < 0 => format!("{} {}-{:#x}", self.kind, self.symbol, a.unsigned_abs()),
            _ => format!("{} {}", self.kind, self.symbol),
        }
    }
}

/// A sub-range of a section that holds instructions (a wasm function body's
/// expression), with an optional note printed before it (its locals).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Region {
    /// First address of the instructions.
    pub start: u64,
    /// One past the last address.
    pub end: u64,
    /// A note printed before the region (e.g. the function's locals).
    pub note: Option<String>,
}

/// One section of a file.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CodeSection {
    /// The section name (Mach-O: `segment,section`).
    pub name: String,
    /// The address of the first byte: 0 in a relocatable object, the load
    /// address in a linked image.
    pub addr: u64,
    /// The contents.
    pub bytes: Vec<u8>,
    /// Whether the section holds instructions.
    pub executable: bool,
    /// The labels defined in it, sorted by address.
    pub labels: Vec<Label>,
    /// The relocations patching it, sorted by address.
    pub relocs: Vec<Reloc>,
    /// When non-empty, only these ranges are instructions (wasm bodies).
    pub regions: Vec<Region>,
}

impl CodeSection {
    fn new(name: impl Into<String>, addr: u64, bytes: Vec<u8>, executable: bool) -> CodeSection {
        CodeSection {
            name: name.into(),
            addr,
            bytes,
            executable,
            labels: Vec::new(),
            relocs: Vec::new(),
            regions: Vec::new(),
        }
    }

    fn sort(&mut self) {
        self.labels.sort_by(|a, b| a.addr.cmp(&b.addr).then_with(|| label_rank(a).cmp(&label_rank(b))));
        self.labels.dedup();
        self.relocs.sort_by_key(|r| r.addr);
    }

    /// One past the section's last address.
    pub fn end(&self) -> u64 {
        self.addr.wrapping_add(self.bytes.len() as u64)
    }
}

/// Labels at one address print in this order: functions, objects, others,
/// then mapping symbols (which a listing does not print).
fn label_rank(l: &Label) -> u8 {
    match l.kind {
        LabelKind::Function => 0,
        LabelKind::Object => 1,
        LabelKind::Other => 2,
        LabelKind::Mapping(_) => 3,
    }
}

/// A file's code, as a listing needs it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Binary {
    /// The container format.
    pub format: FileFormat,
    /// A short description, e.g. `elf64-x86-64`, `mach-o arm64`, `wasm`.
    pub description: String,
    /// The architecture named by the header, if the format names one.
    pub arch: Option<TargetArch>,
    /// The sections (code and data) in file order.
    pub sections: Vec<CodeSection>,
}

/// Read `bytes`, recognizing the format from its header (ELF, `.lfo`,
/// COFF/PE, Mach-O, wasm).
///
/// # Errors
///
/// An unrecognized header, or a file too damaged to locate its sections.
pub fn read(bytes: &[u8]) -> Result<Binary, String> {
    if bytes.starts_with(b"\x7fELF") {
        return read_elf(bytes);
    }
    if bytes.starts_with(&crate::mc::lfo::MAGIC) {
        let obj = crate::mc::lfo::decode(bytes).map_err(|e| format!("cannot decode .lfo: {e}"))?;
        return Ok(from_object_module(&obj, None));
    }
    if bytes.starts_with(b"\0asm") {
        return read_wasm(bytes);
    }
    if bytes.starts_with(b"MZ") {
        return read_pe(bytes);
    }
    if bytes.len() >= 4 {
        let m = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if matches!(m, 0xfeed_facf | 0xfeed_face) {
            return read_macho(bytes);
        }
    }
    if bytes.len() >= 20 && coff_arch(u16::from_le_bytes([bytes[0], bytes[1]])).is_some() {
        return read_coff(bytes, 0, false);
    }
    Err("unrecognized file format (expected ELF, .lfo, COFF/PE, Mach-O or wasm; use --raw for a flat binary)"
        .to_owned())
}

/// A flat binary of `arch` code loaded at `base`: one executable section.
pub fn raw(bytes: &[u8], arch: TargetArch, base: u64) -> Binary {
    Binary {
        format: FileFormat::Raw,
        description: format!("binary ({})", arch.name()),
        arch: Some(arch),
        sections: vec![CodeSection::new(".data", base, bytes.to_vec(), true)],
    }
}

/// The code of an in-memory [`ObjectModule`] (what `.lfo` decodes to, and
/// what every backend produces). The architecture is `arch`, or else guessed
/// from the relocation kinds (AArch64, Thumb and AVR have their own), or
/// `None`.
pub fn from_object_module(obj: &ObjectModule, arch: Option<TargetArch>) -> Binary {
    let guessed = arch.or_else(|| {
        obj.relocations().iter().find_map(|r| match r.kind {
            k if k.is_avr() => Some(TargetArch::Avr),
            k if k.is_thumb() => Some(TargetArch::Thumb),
            RelocKind::Aarch64Call26
            | RelocKind::Aarch64AdrPrelPgHi21
            | RelocKind::Aarch64AddAbsLo12Nc
            | RelocKind::Aarch64AdrGotPage
            | RelocKind::Aarch64Ld64GotLo12Nc => {
                Some(TargetArch::AArch64)
            }
            _ => None,
        })
    });
    let mut sections: Vec<CodeSection> = obj
        .sections()
        .iter()
        .map(|s| CodeSection::new(s.name.clone(), 0, s.bytes.clone(), s.kind == SectionKind::Text))
        .collect();
    for sym in obj.symbols() {
        if let SymbolValue::Defined { section, offset } = sym.value
            && let Some(cs) = sections.get_mut(section.index())
        {
            let kind = match sym.kind {
                SymbolType::Func => LabelKind::Function,
                SymbolType::Object => LabelKind::Object,
                _ => mapping_kind(&sym.name).map_or(LabelKind::Other, LabelKind::Mapping),
            };
            let offset = if kind == LabelKind::Function && guessed == Some(TargetArch::Thumb) { offset & !1 } else { offset };
            cs.labels.push(Label { addr: offset, name: sym.name.clone(), kind });
        }
    }
    for r in obj.relocations() {
        if let Some(cs) = sections.get_mut(r.section.index()) {
            let symbol = obj.symbols().get(r.symbol.index()).map_or_else(|| "?".to_owned(), |s| s.name.clone());
            cs.relocs.push(Reloc { addr: r.offset, kind: format!("{:?}", r.kind), symbol, addend: Some(r.addend) });
        }
    }
    for s in &mut sections {
        s.sort();
    }
    Binary { format: FileFormat::Lfo, description: "lfo".to_owned(), arch: guessed, sections }
}

/// `$x`/`$a`/`$t` (code) or `$d` (data), optionally followed by `.anything`.
fn mapping_kind(name: &str) -> Option<MappingKind> {
    let base = name.split('.').next().unwrap_or(name);
    match base {
        "$x" | "$a" | "$t" => Some(MappingKind::Code),
        "$d" => Some(MappingKind::Data),
        _ => None,
    }
}

// ===========================================================================
// Bounds-checked reading
// ===========================================================================

/// A little- or big-endian view of a byte buffer whose reads fail instead of
/// panicking.
#[derive(Clone, Copy)]
struct Bytes<'a> {
    b: &'a [u8],
    big: bool,
}

fn short() -> String {
    "truncated or corrupt file".to_owned()
}

impl<'a> Bytes<'a> {
    fn slice(&self, at: u64, len: u64) -> Result<&'a [u8], String> {
        let at = usize::try_from(at).map_err(|_| short())?;
        let len = usize::try_from(len).map_err(|_| short())?;
        self.b.get(at..at.checked_add(len).ok_or_else(short)?).ok_or_else(short)
    }
    fn u8(&self, at: u64) -> Result<u8, String> {
        Ok(self.slice(at, 1)?[0])
    }
    fn u16(&self, at: u64) -> Result<u16, String> {
        let s = self.slice(at, 2)?;
        let a = [s[0], s[1]];
        Ok(if self.big { u16::from_be_bytes(a) } else { u16::from_le_bytes(a) })
    }
    fn u32(&self, at: u64) -> Result<u32, String> {
        let s = self.slice(at, 4)?;
        let a = [s[0], s[1], s[2], s[3]];
        Ok(if self.big { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) })
    }
    fn u64(&self, at: u64) -> Result<u64, String> {
        let s = self.slice(at, 8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(if self.big { u64::from_be_bytes(a) } else { u64::from_le_bytes(a) })
    }
    /// A word of the file's class: 4 bytes (32-bit) or 8 (64-bit).
    fn word(&self, at: u64, wide: bool) -> Result<u64, String> {
        if wide { self.u64(at) } else { self.u32(at).map(u64::from) }
    }
    /// A NUL-terminated string at `at` (up to the end of the buffer).
    fn cstr(&self, at: u64) -> String {
        let Ok(at) = usize::try_from(at) else { return String::new() };
        let Some(rest) = self.b.get(at..) else { return String::new() };
        let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
        String::from_utf8_lossy(&rest[..end]).into_owned()
    }
}

/// A fixed-size, NUL-padded name field.
fn fixed_name(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

// ===========================================================================
// ELF
// ===========================================================================

const EM_ARM: u16 = 40;
const EM_X86_64: u16 = 62;
const EM_AVR: u16 = 83;
const EM_AARCH64: u16 = 183;
const EM_RISCV: u16 = 243;

const SHT_SYMTAB: u32 = 2;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHT_REL: u32 = 9;
const SHT_DYNSYM: u32 = 11;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;

struct ElfSection {
    name: String,
    ty: u32,
    flags: u64,
    addr: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    entsize: u64,
}

fn read_elf(file: &[u8]) -> Result<Binary, String> {
    let class = *file.get(4).ok_or_else(short)?;
    let wide = match class {
        1 => false,
        2 => true,
        _ => return Err(format!("ELF: unknown class {class}")),
    };
    let big = match file.get(5) {
        Some(1) => false,
        Some(2) => true,
        _ => return Err("ELF: unknown byte order".to_owned()),
    };
    let b = Bytes { b: file, big };
    let e_type = b.u16(16)?;
    let machine = b.u16(18)?;
    let (phoff, shoff, ehsize_at) = if wide { (b.u64(32)?, b.u64(40)?, 52) } else { (u64::from(b.u32(28)?), u64::from(b.u32(32)?), 40) };
    let entry = if wide { b.u64(24)? } else { u64::from(b.u32(24)?) };
    let phentsize = u64::from(b.u16(ehsize_at + 2)?);
    let phnum = u64::from(b.u16(ehsize_at + 4)?);
    let shentsize = u64::from(b.u16(ehsize_at + 6)?);
    let mut shnum = u64::from(b.u16(ehsize_at + 8)?);
    let mut shstrndx = u64::from(b.u16(ehsize_at + 10)?);

    let arch = match machine {
        EM_X86_64 => Some(TargetArch::X86_64),
        EM_AARCH64 => Some(TargetArch::AArch64),
        EM_RISCV if wide => Some(TargetArch::Riscv64),
        EM_ARM => Some(TargetArch::Thumb),
        EM_AVR => Some(TargetArch::Avr),
        _ => None,
    };
    let mname = match machine {
        EM_X86_64 => "x86-64",
        EM_AARCH64 => "littleaarch64",
        EM_RISCV => "littleriscv",
        EM_ARM => "littlearm",
        EM_AVR => "avr",
        _ => "unknown",
    };
    let description = format!("elf{}-{mname}", if wide { 64 } else { 32 });

    // The section table (its count and string-table index may overflow into
    // section 0).
    let mut secs: Vec<ElfSection> = Vec::new();
    if shoff != 0 && shentsize != 0 {
        if shnum == 0 {
            shnum = b.word(shoff + if wide { 32 } else { 20 }, wide)?;
        }
        if shstrndx == 0xffff {
            shstrndx = u64::from(b.u32(shoff + if wide { 40 } else { 24 })?);
        }
        if shnum > 65_536 {
            return Err("ELF: implausible section count".to_owned());
        }
        for i in 0..shnum {
            let at = shoff + i * shentsize;
            let s = if wide {
                ElfSection {
                    name: String::new(),
                    ty: b.u32(at + 4)?,
                    flags: b.u64(at + 8)?,
                    addr: b.u64(at + 16)?,
                    offset: b.u64(at + 24)?,
                    size: b.u64(at + 32)?,
                    link: b.u32(at + 40)?,
                    info: b.u32(at + 44)?,
                    entsize: b.u64(at + 56)?,
                }
            } else {
                ElfSection {
                    name: String::new(),
                    ty: b.u32(at + 4)?,
                    flags: u64::from(b.u32(at + 8)?),
                    addr: u64::from(b.u32(at + 12)?),
                    offset: u64::from(b.u32(at + 16)?),
                    size: u64::from(b.u32(at + 20)?),
                    link: b.u32(at + 24)?,
                    info: b.u32(at + 28)?,
                    entsize: u64::from(b.u32(at + 36)?),
                }
            };
            secs.push(s);
        }
        if let Some(strtab) = secs.get(shstrndx as usize).map(|s| s.offset) {
            for (i, s) in secs.iter_mut().enumerate() {
                let name_off = b.u32(shoff + i as u64 * shentsize)?;
                s.name = b.cstr(strtab + u64::from(name_off));
            }
        }
    }

    // Allocated sections with contents become `CodeSection`s; `index_map`
    // maps an ELF section index to its position in `out`.
    let mut out: Vec<CodeSection> = Vec::new();
    let mut index_map: Vec<Option<usize>> = vec![None; secs.len()];
    for (i, s) in secs.iter().enumerate() {
        if s.flags & SHF_ALLOC == 0 || s.ty == SHT_NOBITS || s.ty == 0 {
            continue;
        }
        let Ok(data) = b.slice(s.offset, s.size) else { continue };
        index_map[i] = Some(out.len());
        out.push(CodeSection::new(s.name.clone(), s.addr, data.to_vec(), s.flags & SHF_EXECINSTR != 0));
    }

    // Symbols: the static table, else the dynamic one.
    let symtab = secs.iter().position(|s| s.ty == SHT_SYMTAB).or_else(|| secs.iter().position(|s| s.ty == SHT_DYNSYM));
    let mut sym_names: Vec<String> = Vec::new();
    if let Some(si) = symtab {
        let st = &secs[si];
        let strtab = secs.get(st.link as usize).map_or(0, |s| s.offset);
        let entsize = if st.entsize != 0 { st.entsize } else if wide { 24 } else { 16 };
        let count = (st.size / entsize).min(1 << 24);
        for k in 0..count {
            let at = st.offset + k * entsize;
            let (name, info, shndx, value) = if wide {
                (b.u32(at)?, b.u8(at + 4)?, b.u16(at + 6)?, b.u64(at + 8)?)
            } else {
                (b.u32(at)?, b.u8(at + 12)?, b.u16(at + 14)?, u64::from(b.u32(at + 4)?))
            };
            let mut name = b.cstr(strtab + u64::from(name));
            let ty = info & 0xf;
            if ty == 3 {
                // STT_SECTION: name it after its section.
                name = secs.get(usize::from(shndx)).map_or_else(String::new, |s| s.name.clone());
            }
            sym_names.push(name.clone());
            if k == 0 || ty == 4 || name.is_empty() || ty == 3 {
                continue; // the null symbol, STT_FILE, unnamed, sections
            }
            let Some(Some(ci)) = index_map.get(usize::from(shndx)) else { continue };
            let kind = match ty {
                2 => LabelKind::Function,
                1 => LabelKind::Object,
                _ => mapping_kind(&name).map_or(LabelKind::Other, LabelKind::Mapping),
            };
            let mut value = value;
            if machine == EM_ARM && kind == LabelKind::Function {
                value &= !1; // the Thumb bit
            }
            // In a relocatable object a symbol's value is an offset into its
            // section; in a linked image it is already an address.
            let addr = if e_type == 1 { out[*ci].addr.wrapping_add(value) } else { value };
            out[*ci].labels.push(Label { addr, name, kind });
        }
    }

    // Relocations.
    for s in &secs {
        let rela = match s.ty {
            SHT_RELA => true,
            SHT_REL => false,
            _ => continue,
        };
        let Some(Some(ci)) = index_map.get(s.info as usize) else { continue };
        let entsize = if s.entsize != 0 { s.entsize } else { (if wide { 16 } else { 8 }) + if rela { if wide { 8 } else { 4 } } else { 0 } };
        let count = (s.size / entsize).min(1 << 24);
        for k in 0..count {
            let at = s.offset + k * entsize;
            let off = b.word(at, wide)?;
            let info = b.word(at + if wide { 8 } else { 4 }, wide)?;
            let (sym, ty) = if wide { (info >> 32, (info & 0xffff_ffff) as u32) } else { (info >> 8, (info & 0xff) as u32) };
            let addend = if rela {
                let a = b.word(at + if wide { 16 } else { 8 }, wide)?;
                Some(if wide { a as i64 } else { i64::from(a as u32 as i32) })
            } else {
                None
            };
            let symbol = sym_names.get(sym as usize).cloned().unwrap_or_default();
            let addr = if e_type == 1 { out[*ci].addr.wrapping_add(off) } else { off };
            out[*ci].relocs.push(Reloc { addr, kind: elf_reloc_name(machine, ty), symbol, addend });
        }
    }

    // A linked image without a section table: its executable segments.
    if out.is_empty() && phoff != 0 && phentsize != 0 {
        for k in 0..phnum.min(4096) {
            let at = phoff + k * phentsize;
            let ty = b.u32(at)?;
            let (flags, offset, vaddr, filesz) = if wide {
                (b.u32(at + 4)?, b.u64(at + 8)?, b.u64(at + 16)?, b.u64(at + 32)?)
            } else {
                (b.u32(at + 24)?, u64::from(b.u32(at + 4)?), u64::from(b.u32(at + 8)?), u64::from(b.u32(at + 16)?))
            };
            // The first segment usually maps the file headers too: start after
            // the ELF header and the program header table, not at their bytes.
            let headers_end = (if wide { 64 } else { 52 }).max(phoff + phnum * phentsize);
            let skip = if offset < headers_end { (headers_end - offset).min(filesz) } else { 0 };
            if ty == 1
                && flags & 1 != 0
                && skip < filesz
                && let Ok(data) = b.slice(offset + skip, filesz - skip)
            {
                let mut sec = CodeSection::new(format!("LOAD{k}"), vaddr + skip, data.to_vec(), true);
                if (sec.addr..sec.addr + sec.bytes.len() as u64).contains(&entry) {
                    sec.labels.push(Label { addr: entry, name: "entry".to_owned(), kind: LabelKind::Function });
                }
                out.push(sec);
            }
        }
    }

    for s in &mut out {
        s.sort();
    }
    Ok(Binary { format: FileFormat::Elf, description, arch, sections: out })
}

/// The ELF relocation type name for `machine` (from each architecture's ELF
/// processor supplement), or `R_<n>` for an unnamed one.
pub fn elf_reloc_name(machine: u16, ty: u32) -> String {
    let name: Option<&str> = match machine {
        EM_X86_64 => X86_64_RELOCS.get(ty as usize).copied().filter(|s| !s.is_empty()),
        EM_AARCH64 => aarch64_reloc(ty),
        EM_RISCV => riscv_reloc(ty),
        EM_ARM => arm_reloc(ty),
        EM_AVR => AVR_RELOCS.get(ty as usize).copied().filter(|s| !s.is_empty()),
        _ => None,
    };
    let prefix = match machine {
        EM_X86_64 => "R_X86_64_",
        EM_AARCH64 => "R_AARCH64_",
        EM_RISCV => "R_RISCV_",
        EM_ARM => "R_ARM_",
        EM_AVR => "R_AVR_",
        _ => "R_",
    };
    match name {
        Some(n) => format!("{prefix}{n}"),
        None => format!("{prefix}{ty}"),
    }
}

const X86_64_RELOCS: [&str; 43] = [
    "NONE", "64", "PC32", "GOT32", "PLT32", "COPY", "GLOB_DAT", "JUMP_SLOT", "RELATIVE", "GOTPCREL", "32", "32S",
    "16", "PC16", "8", "PC8", "DTPMOD64", "DTPOFF64", "TPOFF64", "TLSGD", "TLSLD", "DTPOFF32", "GOTTPOFF", "TPOFF32",
    "PC64", "GOTOFF64", "GOTPC32", "GOT64", "GOTPCREL64", "GOTPC64", "GOTPLT64", "PLTOFF64", "SIZE32", "SIZE64",
    "GOTPC32_TLSDESC", "TLSDESC_CALL", "TLSDESC", "IRELATIVE", "RELATIVE64", "", "", "GOTPCRELX", "REX_GOTPCRELX",
];

fn aarch64_reloc(ty: u32) -> Option<&'static str> {
    Some(match ty {
        0 => "NONE",
        257 => "ABS64",
        258 => "ABS32",
        259 => "ABS16",
        260 => "PREL64",
        261 => "PREL32",
        262 => "PREL16",
        263 => "MOVW_UABS_G0",
        264 => "MOVW_UABS_G0_NC",
        265 => "MOVW_UABS_G1",
        266 => "MOVW_UABS_G1_NC",
        267 => "MOVW_UABS_G2",
        268 => "MOVW_UABS_G2_NC",
        269 => "MOVW_UABS_G3",
        273 => "LD_PREL_LO19",
        274 => "ADR_PREL_LO21",
        275 => "ADR_PREL_PG_HI21",
        276 => "ADR_PREL_PG_HI21_NC",
        277 => "ADD_ABS_LO12_NC",
        278 => "LDST8_ABS_LO12_NC",
        279 => "TSTBR14",
        280 => "CONDBR19",
        282 => "JUMP26",
        283 => "CALL26",
        284 => "LDST16_ABS_LO12_NC",
        285 => "LDST32_ABS_LO12_NC",
        286 => "LDST64_ABS_LO12_NC",
        299 => "LDST128_ABS_LO12_NC",
        311 => "ADR_GOT_PAGE",
        312 => "LD64_GOT_LO12_NC",
        512 => "TLSGD_ADR_PREL21",
        513 => "TLSGD_ADR_PAGE21",
        514 => "TLSGD_ADD_LO12_NC",
        541 => "TLSIE_ADR_GOTTPREL_PAGE21",
        542 => "TLSIE_LD64_GOTTPREL_LO12_NC",
        549 => "TLSLE_ADD_TPREL_HI12",
        550 => "TLSLE_ADD_TPREL_LO12",
        551 => "TLSLE_ADD_TPREL_LO12_NC",
        560 => "TLSDESC_ADR_PAGE21",
        561 => "TLSDESC_LD64_LO12",
        562 => "TLSDESC_ADD_LO12",
        569 => "TLSDESC_CALL",
        1024 => "COPY",
        1025 => "GLOB_DAT",
        1026 => "JUMP_SLOT",
        1027 => "RELATIVE",
        _ => return None,
    })
}

fn riscv_reloc(ty: u32) -> Option<&'static str> {
    Some(match ty {
        0 => "NONE",
        1 => "32",
        2 => "64",
        3 => "RELATIVE",
        4 => "COPY",
        5 => "JUMP_SLOT",
        16 => "BRANCH",
        17 => "JAL",
        18 => "CALL",
        19 => "CALL_PLT",
        20 => "GOT_HI20",
        21 => "TLS_GOT_HI20",
        22 => "TLS_GD_HI20",
        23 => "PCREL_HI20",
        24 => "PCREL_LO12_I",
        25 => "PCREL_LO12_S",
        26 => "HI20",
        27 => "LO12_I",
        28 => "LO12_S",
        29 => "TPREL_HI20",
        30 => "TPREL_LO12_I",
        31 => "TPREL_LO12_S",
        32 => "TPREL_ADD",
        33 => "ADD8",
        34 => "ADD16",
        35 => "ADD32",
        36 => "ADD64",
        37 => "SUB8",
        38 => "SUB16",
        39 => "SUB32",
        40 => "SUB64",
        43 => "ALIGN",
        44 => "RVC_BRANCH",
        45 => "RVC_JUMP",
        51 => "RELAX",
        52 => "SUB6",
        53 => "SET6",
        54 => "SET8",
        55 => "SET16",
        56 => "SET32",
        57 => "32_PCREL",
        _ => return None,
    })
}

fn arm_reloc(ty: u32) -> Option<&'static str> {
    Some(match ty {
        0 => "NONE",
        2 => "ABS32",
        3 => "REL32",
        5 => "ABS16",
        10 => "THM_CALL",
        28 => "CALL",
        29 => "JUMP24",
        30 => "THM_JUMP24",
        42 => "PREL31",
        43 => "MOVW_ABS_NC",
        44 => "MOVT_ABS",
        47 => "THM_MOVW_ABS_NC",
        48 => "THM_MOVT_ABS",
        51 => "THM_JUMP19",
        102 => "THM_JUMP11",
        103 => "THM_JUMP8",
        _ => return None,
    })
}

const AVR_RELOCS: [&str; 19] = [
    "NONE", "32", "7_PCREL", "13_PCREL", "16", "16_PM", "LO8_LDI", "HI8_LDI", "HH8_LDI", "LO8_LDI_NEG",
    "HI8_LDI_NEG", "HH8_LDI_NEG", "LO8_LDI_PM", "HI8_LDI_PM", "HH8_LDI_PM", "LO8_LDI_PM_NEG", "HI8_LDI_PM_NEG",
    "HH8_LDI_PM_NEG", "CALL",
];

// ===========================================================================
// PE/COFF
// ===========================================================================

const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
const IMAGE_FILE_MACHINE_ARM64: u16 = 0xaa64;
const IMAGE_FILE_MACHINE_ARMNT: u16 = 0x01c4;
const IMAGE_FILE_MACHINE_THUMB: u16 = 0x01c2;
const IMAGE_FILE_MACHINE_RISCV64: u16 = 0x5064;

fn coff_arch(machine: u16) -> Option<TargetArch> {
    match machine {
        IMAGE_FILE_MACHINE_AMD64 => Some(TargetArch::X86_64),
        IMAGE_FILE_MACHINE_ARM64 => Some(TargetArch::AArch64),
        IMAGE_FILE_MACHINE_ARMNT | IMAGE_FILE_MACHINE_THUMB => Some(TargetArch::Thumb),
        IMAGE_FILE_MACHINE_RISCV64 => Some(TargetArch::Riscv64),
        _ => None,
    }
}

fn read_pe(file: &[u8]) -> Result<Binary, String> {
    let b = Bytes { b: file, big: false };
    let pe = u64::from(b.u32(0x3c)?);
    if b.slice(pe, 4)? != b"PE\0\0" {
        return Err("an MZ executable without a PE header".to_owned());
    }
    read_coff(file, pe + 4, true)
}

/// A COFF file header at `at`: an object (`image` false) or the header of a
/// PE image.
fn read_coff(file: &[u8], at: u64, image: bool) -> Result<Binary, String> {
    let b = Bytes { b: file, big: false };
    let machine = b.u16(at)?;
    let nsects = u64::from(b.u16(at + 2)?);
    let symptr = u64::from(b.u32(at + 8)?);
    let nsyms = u64::from(b.u32(at + 12)?);
    let opt_size = u64::from(b.u16(at + 16)?);
    let arch = coff_arch(machine);
    let image_base = if image && opt_size >= 32 {
        match b.u16(at + 20)? {
            0x20b => b.u64(at + 20 + 24)?,
            _ => u64::from(b.u32(at + 20 + 28)?),
        }
    } else {
        0
    };
    let mname = match machine {
        IMAGE_FILE_MACHINE_AMD64 => "x86-64",
        IMAGE_FILE_MACHINE_ARM64 => "ARM64",
        IMAGE_FILE_MACHINE_ARMNT | IMAGE_FILE_MACHINE_THUMB => "ARM",
        IMAGE_FILE_MACHINE_RISCV64 => "RISCV64",
        _ => "unknown",
    };
    let description = if image { format!("pe-{mname}") } else { format!("coff-{mname}") };
    let strtab = symptr + nsyms * 18;
    let long_name = |name: &[u8]| -> String {
        if name[0] == b'/' {
            let digits = fixed_name(&name[1..]);
            if let Ok(off) = digits.parse::<u64>() {
                return b.cstr(strtab + off);
            }
        }
        fixed_name(name)
    };

    let shdrs = at + 20 + opt_size;
    let mut out = Vec::new();
    let mut rel_ptrs = Vec::new();
    for i in 0..nsects.min(65_536) {
        let h = shdrs + i * 40;
        let name = long_name(b.slice(h, 8)?);
        let vsize = u64::from(b.u32(h + 8)?);
        let vaddr = u64::from(b.u32(h + 12)?);
        let raw_size = u64::from(b.u32(h + 16)?);
        let raw_ptr = u64::from(b.u32(h + 20)?);
        let rel_ptr = u64::from(b.u32(h + 24)?);
        let nrel = u64::from(b.u16(h + 32)?);
        let chars = b.u32(h + 36)?;
        let size = if image && vsize != 0 { vsize.min(raw_size) } else { raw_size };
        let data = if raw_ptr == 0 { Vec::new() } else { b.slice(raw_ptr, size).map(<[u8]>::to_vec).unwrap_or_default() };
        let exec = chars & (0x20 | 0x2000_0000) != 0;
        let addr = if image { image_base + vaddr } else { 0 };
        out.push(CodeSection::new(name, addr, data, exec));
        rel_ptrs.push((rel_ptr, nrel));
    }

    // Symbols (with their aux records skipped).
    let mut sym_names: Vec<String> = Vec::new();
    let mut k = 0;
    while k < nsyms.min(1 << 24) {
        let e = symptr + k * 18;
        let Ok(raw) = b.slice(e, 18) else { break };
        let name = if raw[..4] == [0, 0, 0, 0] {
            b.cstr(strtab + u64::from(u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]])))
        } else {
            fixed_name(&raw[..8])
        };
        let value = u64::from(u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]));
        let section = i16::from_le_bytes([raw[12], raw[13]]);
        let ty = u16::from_le_bytes([raw[14], raw[15]]);
        let class = raw[16];
        let aux = u64::from(raw[17]);
        sym_names.push(name.clone());
        for _ in 0..aux {
            sym_names.push(name.clone());
        }
        let is_section_def = class == 3 && aux > 0 && value == 0;
        if section > 0
            && matches!(class, 2 | 3 | 6)
            && !is_section_def
            && !name.is_empty()
            && let Some(cs) = out.get_mut(section as usize - 1)
        {
            let kind = if ty & 0x30 == 0x20 { LabelKind::Function } else { mapping_kind(&name).map_or(LabelKind::Other, LabelKind::Mapping) };
            let addr = cs.addr + value;
            cs.labels.push(Label { addr, name, kind });
        }
        k += 1 + aux;
    }

    for (i, &(ptr, n)) in rel_ptrs.iter().enumerate() {
        if image || ptr == 0 {
            continue;
        }
        for r in 0..n {
            let e = ptr + r * 10;
            let (Ok(va), Ok(si), Ok(ty)) = (b.u32(e), b.u32(e + 4), b.u16(e + 8)) else { break };
            let symbol = sym_names.get(si as usize).cloned().unwrap_or_default();
            let addr = out[i].addr + u64::from(va);
            out[i].relocs.push(Reloc { addr, kind: coff_reloc_name(machine, ty), symbol, addend: None });
        }
    }
    for s in &mut out {
        s.sort();
    }
    Ok(Binary { format: if image { FileFormat::Pe } else { FileFormat::Coff }, description, arch, sections: out })
}

/// The PE/COFF relocation type name for `machine`.
pub fn coff_reloc_name(machine: u16, ty: u16) -> String {
    let (prefix, name) = match machine {
        IMAGE_FILE_MACHINE_AMD64 => (
            "IMAGE_REL_AMD64_",
            match ty {
                0 => "ABSOLUTE",
                1 => "ADDR64",
                2 => "ADDR32",
                3 => "ADDR32NB",
                4 => "REL32",
                5 => "REL32_1",
                6 => "REL32_2",
                7 => "REL32_3",
                8 => "REL32_4",
                9 => "REL32_5",
                10 => "SECTION",
                11 => "SECREL",
                12 => "SECREL7",
                _ => "",
            },
        ),
        IMAGE_FILE_MACHINE_ARM64 => (
            "IMAGE_REL_ARM64_",
            match ty {
                0 => "ABSOLUTE",
                1 => "ADDR32",
                2 => "ADDR32NB",
                3 => "BRANCH26",
                4 => "PAGEBASE_REL21",
                5 => "REL21",
                6 => "PAGEOFFSET_12A",
                7 => "PAGEOFFSET_12L",
                8 => "SECREL",
                9 => "SECREL_LOW12A",
                10 => "SECREL_HIGH12A",
                11 => "SECREL_LOW12L",
                12 => "TOKEN",
                13 => "SECTION",
                14 => "ADDR64",
                15 => "BRANCH19",
                16 => "BRANCH14",
                17 => "REL32",
                _ => "",
            },
        ),
        IMAGE_FILE_MACHINE_ARMNT | IMAGE_FILE_MACHINE_THUMB => (
            "IMAGE_REL_ARM_",
            match ty {
                0 => "ABSOLUTE",
                1 => "ADDR32",
                2 => "ADDR32NB",
                3 => "BRANCH24",
                4 => "BRANCH11",
                10 => "REL32",
                14 => "SECTION",
                15 => "SECREL",
                16 => "MOV32",
                17 => "MOV32T",
                20 => "BRANCH20T",
                21 => "BRANCH24T",
                22 => "BLX23T",
                _ => "",
            },
        ),
        _ => ("IMAGE_REL_", ""),
    };
    if name.is_empty() { format!("{prefix}{ty:#x}") } else { format!("{prefix}{name}") }
}

// ===========================================================================
// Mach-O
// ===========================================================================

const CPU_TYPE_X86_64: u32 = 0x0100_0007;
const CPU_TYPE_ARM64: u32 = 0x0100_000c;
const CPU_TYPE_ARM: u32 = 12;

fn read_macho(file: &[u8]) -> Result<Binary, String> {
    let b = Bytes { b: file, big: false };
    let wide = b.u32(0)? == 0xfeed_facf;
    let cpu = b.u32(4)?;
    let ncmds = u64::from(b.u32(16)?);
    let (arch, mname) = match cpu {
        CPU_TYPE_X86_64 => (Some(TargetArch::X86_64), "x86-64"),
        CPU_TYPE_ARM64 => (Some(TargetArch::AArch64), "arm64"),
        CPU_TYPE_ARM => (Some(TargetArch::Thumb), "arm"),
        _ => (None, "unknown"),
    };
    let mut at = if wide { 32 } else { 28 };
    let mut out: Vec<CodeSection> = Vec::new();
    let mut rels: Vec<(u64, u64)> = Vec::new();
    let mut symtab: Option<(u64, u64, u64)> = None;
    for _ in 0..ncmds.min(65_536) {
        let cmd = b.u32(at)?;
        let size = u64::from(b.u32(at + 4)?);
        if size < 8 {
            return Err("Mach-O: corrupt load command".to_owned());
        }
        match cmd {
            0x19 | 0x1 => {
                let seg64 = cmd == 0x19;
                let (hdr, sect) = if seg64 { (72, 80) } else { (56, 68) };
                let nsects = u64::from(b.u32(at + if seg64 { 64 } else { 48 })?);
                for k in 0..nsects.min(65_536) {
                    let s = at + hdr + k * sect;
                    let sectname = fixed_name(b.slice(s, 16)?);
                    let segname = fixed_name(b.slice(s + 16, 16)?);
                    let (addr, size) = if seg64 { (b.u64(s + 32)?, b.u64(s + 40)?) } else { (u64::from(b.u32(s + 32)?), u64::from(b.u32(s + 36)?)) };
                    let f = if seg64 { s + 48 } else { s + 40 };
                    let offset = u64::from(b.u32(f)?);
                    let reloff = u64::from(b.u32(f + 8)?);
                    let nreloc = u64::from(b.u32(f + 12)?);
                    let flags = b.u32(f + 16)?;
                    let zerofill = matches!(flags & 0xff, 1 | 12 | 18);
                    let data = if zerofill { Vec::new() } else { b.slice(offset, size).map(<[u8]>::to_vec).unwrap_or_default() };
                    let exec = flags & 0x8000_0400 != 0;
                    out.push(CodeSection::new(format!("{segname},{sectname}"), addr, data, exec));
                    rels.push((reloff, nreloc));
                }
            }
            0x2 => symtab = Some((u64::from(b.u32(at + 8)?), u64::from(b.u32(at + 12)?), u64::from(b.u32(at + 16)?))),
            _ => {}
        }
        at += size;
    }

    let mut sym_names: Vec<String> = Vec::new();
    if let Some((symoff, nsyms, stroff)) = symtab {
        let entsize = if wide { 16 } else { 12 };
        for k in 0..nsyms.min(1 << 24) {
            let e = symoff + k * entsize;
            let Ok(strx) = b.u32(e) else { break };
            let ty = b.u8(e + 4)?;
            let sect = b.u8(e + 5)?;
            let value = if wide { b.u64(e + 8)? } else { u64::from(b.u32(e + 8)?) };
            let name = b.cstr(stroff + u64::from(strx));
            sym_names.push(name.clone());
            // Defined in a section (N_SECT), not a debugging (stab) entry.
            if ty & 0xe0 != 0 || ty & 0x0e != 0x0e || sect == 0 || name.is_empty() {
                continue;
            }
            if let Some(cs) = out.get_mut(usize::from(sect) - 1) {
                let kind = if cs.executable && !name.starts_with("ltmp") && !name.starts_with('L') {
                    LabelKind::Function
                } else {
                    mapping_kind(&name).map_or(LabelKind::Other, LabelKind::Mapping)
                };
                let addr = if cpu == CPU_TYPE_ARM && kind == LabelKind::Function { value & !1 } else { value };
                cs.labels.push(Label { addr, name, kind });
            }
        }
    }

    for (i, &(reloff, n)) in rels.iter().enumerate() {
        let mut pending_addend: Option<i64> = None;
        for r in 0..n.min(1 << 20) {
            let e = reloff + r * 8;
            let (Ok(address), Ok(word)) = (b.u32(e), b.u32(e + 4)) else { break };
            if address & 0x8000_0000 != 0 {
                continue; // scattered
            }
            let symnum = word & 0x00ff_ffff;
            let is_extern = word >> 27 & 1 != 0;
            let ty = word >> 28;
            if cpu == CPU_TYPE_ARM64 && ty == 10 {
                pending_addend = Some(crate::mc::disasm::sext(u64::from(symnum), 24));
                continue;
            }
            let symbol = if is_extern {
                sym_names.get(symnum as usize).cloned().unwrap_or_default()
            } else {
                out.get((symnum as usize).wrapping_sub(1)).map_or_else(|| format!("section{symnum}"), |s| s.name.clone())
            };
            let addr = out[i].addr + u64::from(address);
            let kind = macho_reloc_name(cpu, ty);
            out[i].relocs.push(Reloc { addr, kind, symbol, addend: pending_addend.take() });
        }
    }
    for s in &mut out {
        s.sort();
    }
    Ok(Binary { format: FileFormat::MachO, description: format!("mach-o {mname}"), arch, sections: out })
}

/// The Mach-O relocation type name for `cpu`.
pub fn macho_reloc_name(cpu: u32, ty: u32) -> String {
    let (prefix, names): (&str, &[&str]) = match cpu {
        CPU_TYPE_X86_64 => (
            "X86_64_RELOC_",
            &["UNSIGNED", "SIGNED", "BRANCH", "GOT_LOAD", "GOT", "SUBTRACTOR", "SIGNED_1", "SIGNED_2", "SIGNED_4", "TLV"],
        ),
        CPU_TYPE_ARM64 => (
            "ARM64_RELOC_",
            &[
                "UNSIGNED",
                "SUBTRACTOR",
                "BRANCH26",
                "PAGE21",
                "PAGEOFF12",
                "GOT_LOAD_PAGE21",
                "GOT_LOAD_PAGEOFF12",
                "POINTER_TO_GOT",
                "TLVP_LOAD_PAGE21",
                "TLVP_LOAD_PAGEOFF12",
                "ADDEND",
            ],
        ),
        CPU_TYPE_ARM => (
            "ARM_RELOC_",
            &["VANILLA", "PAIR", "SECTDIFF", "LOCAL_SECTDIFF", "PB_LA_PTR", "BR24", "THUMB_RELOC_BR22", "THUMB_32BIT_BRANCH", "HALF", "HALF_SECTDIFF"],
        ),
        _ => ("RELOC_", &[]),
    };
    match names.get(ty as usize) {
        Some(n) => format!("{prefix}{n}"),
        None => format!("{prefix}{ty}"),
    }
}

// ===========================================================================
// WebAssembly
// ===========================================================================

fn wname(b: &[u8], at: &mut usize) -> Option<String> {
    let len = usize::try_from(uleb(b, at)?).ok()?;
    let s = b.get(*at..at.checked_add(len)?)?;
    *at += len;
    Some(String::from_utf8_lossy(s).into_owned())
}

/// The value-type name of a wasm type byte.
pub(crate) fn wasm_valtype(t: u8) -> &'static str {
    match t {
        0x7f => "i32",
        0x7e => "i64",
        0x7d => "f32",
        0x7c => "f64",
        0x7b => "v128",
        0x70 => "funcref",
        0x6f => "externref",
        _ => "?",
    }
}

fn read_wasm(file: &[u8]) -> Result<Binary, String> {
    if file.len() < 8 || file[4..8] != [1, 0, 0, 0] {
        return Err("wasm: unsupported version".to_owned());
    }
    // Pass 1: the sections.
    let mut secs: Vec<(u8, String, usize, usize)> = Vec::new(); // id, custom name, payload start, end
    let mut at = 8;
    while at < file.len() {
        let id = file[at];
        at += 1;
        let len = uleb(file, &mut at).ok_or_else(short)? as usize;
        let end = at.checked_add(len).filter(|&e| e <= file.len()).ok_or_else(short)?;
        let mut name = String::new();
        if id == 0 {
            let mut p = at;
            name = wname(file, &mut p).unwrap_or_default();
        }
        secs.push((id, name, at, end));
        at = end;
    }

    // Function names: imports first in the index space.
    let mut imported_funcs: u32 = 0;
    let mut func_names: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
    let mut global_names: Vec<String> = Vec::new();
    let mut syms: Vec<String> = Vec::new();
    let mut parse_imports = |p: &[u8]| -> Option<()> {
        let mut q = 0;
        let n = uleb(p, &mut q)?;
        for _ in 0..n {
            let _module = wname(p, &mut q)?;
            let field = wname(p, &mut q)?;
            let kind = *p.get(q)?;
            q += 1;
            match kind {
                0 => {
                    uleb(p, &mut q)?;
                    func_names.entry(imported_funcs).or_insert(field);
                    imported_funcs += 1;
                }
                1 => {
                    q += 1;
                    let flags = uleb(p, &mut q)?;
                    uleb(p, &mut q)?;
                    if flags & 1 != 0 {
                        uleb(p, &mut q)?;
                    }
                }
                2 => {
                    let flags = uleb(p, &mut q)?;
                    uleb(p, &mut q)?;
                    if flags & 1 != 0 {
                        uleb(p, &mut q)?;
                    }
                }
                3 => {
                    global_names.push(field);
                    q += 2;
                }
                4 => {
                    q += 1;
                    uleb(p, &mut q)?;
                }
                _ => return None,
            }
        }
        Some(())
    };
    if let Some(&(_, _, s, e)) = secs.iter().find(|s| s.0 == 2) {
        let _ = parse_imports(&file[s..e]);
    }
    // Exports (the weakest name source).
    if let Some(&(_, _, s, e)) = secs.iter().find(|s| s.0 == 7) {
        let p = &file[s..e];
        let mut q = 0;
        if let Some(n) = uleb(p, &mut q) {
            for _ in 0..n {
                let (Some(name), Some(&kind)) = (wname(p, &mut q), p.get(q)) else { break };
                q += 1;
                let Some(idx) = uleb(p, &mut q) else { break };
                if kind == 0 {
                    func_names.insert(idx as u32, name);
                }
            }
        }
    }
    // The linking section's symbol table (names of defined functions, and
    // what relocations refer to).
    if let Some(&(_, _, s, e)) = secs.iter().find(|s| s.0 == 0 && s.1 == "linking") {
        let p = &file[s..e];
        let mut q = 0;
        let _ = wname(p, &mut q);
        let _version = uleb(p, &mut q);
        while q < p.len() {
            let Some(&sub) = p.get(q) else { break };
            q += 1;
            let Some(len) = uleb(p, &mut q) else { break };
            let end = q.saturating_add(len as usize).min(p.len());
            if sub == 8 {
                let mut r = q;
                let n = uleb(p, &mut r).unwrap_or(0);
                for _ in 0..n {
                    let (Some(&kind), true) = (p.get(r), r < end) else { break };
                    r += 1;
                    let Some(flags) = uleb(p, &mut r) else { break };
                    let undefined = flags & 0x10 != 0;
                    let explicit = flags & 0x40 != 0;
                    match kind {
                        0 | 2 | 4 | 5 => {
                            let Some(idx) = uleb(p, &mut r) else { break };
                            let name = if !undefined || explicit {
                                wname(p, &mut r).unwrap_or_default()
                            } else if kind == 0 {
                                func_names.get(&(idx as u32)).cloned().unwrap_or_default()
                            } else if kind == 2 {
                                global_names.get(idx as usize).cloned().unwrap_or_default()
                            } else {
                                String::new()
                            };
                            if kind == 0 && !name.is_empty() && !undefined {
                                func_names.insert(idx as u32, name.clone());
                            }
                            syms.push(name);
                        }
                        1 => {
                            let name = wname(p, &mut r).unwrap_or_default();
                            if !undefined {
                                uleb(p, &mut r);
                                uleb(p, &mut r);
                                uleb(p, &mut r);
                            }
                            syms.push(name);
                        }
                        3 => {
                            uleb(p, &mut r);
                            syms.push("section".to_owned());
                        }
                        _ => break,
                    }
                }
            }
            q = end;
        }
    }
    // The name section (the strongest source).
    if let Some(&(_, _, s, e)) = secs.iter().find(|s| s.0 == 0 && s.1 == "name") {
        let p = &file[s..e];
        let mut q = 0;
        let _ = wname(p, &mut q);
        while q < p.len() {
            let Some(&sub) = p.get(q) else { break };
            q += 1;
            let Some(len) = uleb(p, &mut q) else { break };
            let end = q.saturating_add(len as usize).min(p.len());
            if sub == 1 {
                let mut r = q;
                let n = uleb(p, &mut r).unwrap_or(0);
                for _ in 0..n {
                    let (Some(idx), Some(name)) = (uleb(p, &mut r), wname(p, &mut r)) else { break };
                    func_names.insert(idx as u32, name);
                }
            }
            q = end;
        }
    }

    let mut out: Vec<CodeSection> = Vec::new();
    let mut code_index = None;
    for (k, &(id, ref name, s, e)) in secs.iter().enumerate() {
        let sname = match id {
            0 => name.clone(),
            1 => "TYPE".to_owned(),
            2 => "IMPORT".to_owned(),
            3 => "FUNCTION".to_owned(),
            4 => "TABLE".to_owned(),
            5 => "MEMORY".to_owned(),
            6 => "GLOBAL".to_owned(),
            7 => "EXPORT".to_owned(),
            8 => "START".to_owned(),
            9 => "ELEM".to_owned(),
            10 => "CODE".to_owned(),
            11 => "DATA".to_owned(),
            12 => "DATACOUNT".to_owned(),
            _ => format!("section{id}"),
        };
        if id != 10 {
            continue;
        }
        code_index = Some(k);
        // Addresses are offsets from the start of the section's payload,
        // the frame of reference of `reloc.CODE`.
        let mut cs = CodeSection::new(sname, 0, file[s..e].to_vec(), true);
        let p = &file[s..e];
        let mut q = 0;
        let n = uleb(p, &mut q).unwrap_or(0);
        for f in 0..n {
            let body_start = q;
            let Some(size) = uleb(p, &mut q) else { break };
            let body_end = q.saturating_add(size as usize).min(p.len());
            let mut r = q;
            let mut locals = Vec::new();
            if let Some(groups) = uleb(p, &mut r) {
                for _ in 0..groups {
                    let Some(count) = uleb(p, &mut r) else { break };
                    let Some(&ty) = p.get(r) else { break };
                    r += 1;
                    locals.push(format!("{count} x {}", wasm_valtype(ty)));
                }
            }
            let index = imported_funcs + f as u32;
            let name = func_names.get(&index).cloned().unwrap_or_else(|| format!("func{index}"));
            cs.labels.push(Label { addr: body_start as u64, name, kind: LabelKind::Function });
            let note = if locals.is_empty() { "locals: none".to_owned() } else { format!("locals: {}", locals.join(", ")) };
            cs.regions.push(Region { start: r.min(body_end) as u64, end: body_end as u64, note: Some(note) });
            q = body_end;
        }
        out.push(cs);
    }

    // reloc.CODE: offsets relative to the code section's payload.
    if let Some(ci) = code_index {
        for &(id, ref name, s, e) in &secs {
            if id != 0 || !name.starts_with("reloc.") {
                continue;
            }
            let p = &file[s..e];
            let mut q = 0;
            let _ = wname(p, &mut q);
            let Some(target) = uleb(p, &mut q) else { continue };
            if target as usize != ci {
                continue;
            }
            let n = uleb(p, &mut q).unwrap_or(0);
            let cs = out.last_mut().expect("the code section");
            for _ in 0..n {
                let Some(&ty) = p.get(q) else { break };
                q += 1;
                let (Some(off), Some(idx)) = (uleb(p, &mut q), uleb(p, &mut q)) else { break };
                let has_addend = matches!(ty, 3 | 4 | 5 | 8 | 9 | 11 | 14 | 15 | 16 | 17 | 21 | 22 | 23 | 25);
                let addend = if has_addend { sleb(p, &mut q) } else { None };
                let symbol = if ty == 6 { format!("type{idx}") } else { syms.get(idx as usize).cloned().unwrap_or_else(|| format!("sym{idx}")) };
                cs.relocs.push(Reloc { addr: off, kind: wasm_reloc_name(ty), symbol, addend });
            }
        }
    }
    for s in &mut out {
        s.sort();
    }
    let relocatable = secs.iter().any(|s| s.0 == 0 && s.1 == "linking");
    Ok(Binary {
        format: FileFormat::Wasm,
        description: if relocatable { "wasm (relocatable)".to_owned() } else { "wasm".to_owned() },
        arch: Some(TargetArch::Wasm32),
        sections: out,
    })
}

/// The tool-conventions name of a wasm relocation type.
pub fn wasm_reloc_name(ty: u8) -> String {
    const NAMES: [&str; 27] = [
        "FUNCTION_INDEX_LEB",
        "TABLE_INDEX_SLEB",
        "TABLE_INDEX_I32",
        "MEMORY_ADDR_LEB",
        "MEMORY_ADDR_SLEB",
        "MEMORY_ADDR_I32",
        "TYPE_INDEX_LEB",
        "GLOBAL_INDEX_LEB",
        "FUNCTION_OFFSET_I32",
        "SECTION_OFFSET_I32",
        "TAG_INDEX_LEB",
        "MEMORY_ADDR_REL_SLEB",
        "TABLE_INDEX_REL_SLEB",
        "GLOBAL_INDEX_I32",
        "MEMORY_ADDR_LEB64",
        "MEMORY_ADDR_SLEB64",
        "MEMORY_ADDR_I64",
        "MEMORY_ADDR_REL_SLEB64",
        "TABLE_INDEX_SLEB64",
        "TABLE_INDEX_I64",
        "TABLE_NUMBER_LEB",
        "MEMORY_ADDR_TLS_SLEB",
        "FUNCTION_OFFSET_I64",
        "MEMORY_ADDR_LOCREL_I32",
        "TABLE_INDEX_REL_SLEB64",
        "MEMORY_ADDR_TLS_SLEB64",
        "FUNCTION_INDEX_I32",
    ];
    match NAMES.get(usize::from(ty)) {
        Some(n) => format!("R_WASM_{n}"),
        None => format!("R_WASM_{ty}"),
    }
}
