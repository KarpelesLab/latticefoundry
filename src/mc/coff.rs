//! A PE/COFF relocatable-object writer for x86-64 (AMD64) and ARM64,
//! implemented from the Microsoft PE/COFF specification.
//!
//! This turns the framework's target-independent [`ObjectModule`] into a COFF
//! object (`.obj`) that a PE linker — MSVC `link`, `lld-link`, MinGW `ld`, or
//! our own `qld` — consumes. It is a clean-room implementation from the
//! published structure layouts and constants (tenet T1).
//!
//! # What it emits
//!
//! - the 20-byte COFF file header (`Machine` = `IMAGE_FILE_MACHINE_AMD64` or
//!   `IMAGE_FILE_MACHINE_ARM64`, no optional header, a zero time stamp so the
//!   output is deterministic);
//! - one 40-byte section header per [`Section`](crate::mc::object::Section),
//!   named `.text`, `.data`, `.rdata` (for `.rodata`), `.bss` or the section's
//!   own name (names longer than 8 bytes go to the string table as `/n`), with
//!   the `IMAGE_SCN_CNT_*`/`IMAGE_SCN_MEM_*` flags of its kind and its
//!   `IMAGE_SCN_ALIGN_*` alignment; `.debug_*` sections are
//!   `IMAGE_SCN_MEM_DISCARDABLE`;
//! - the raw data and the relocation table of each section (with the
//!   `IMAGE_SCN_LNK_NRELOC_OVFL` escape past 65535 relocations);
//! - the symbol table: a static section symbol (plus its section-definition
//!   auxiliary record) for every section, then the module's symbols — locals as
//!   `IMAGE_SYM_CLASS_STATIC`, globals and undefined references as
//!   `IMAGE_SYM_CLASS_EXTERNAL`, functions with type `0x20`. A weak symbol
//!   becomes an `IMAGE_SYM_CLASS_WEAK_EXTERNAL` whose auxiliary record names a
//!   default: for a weak definition, a `.weak.<name>.default.<tag>` external at
//!   the definition (searched as an alias); for a weak reference, an absolute
//!   zero of that name (no library search);
//! - the string table.
//!
//! # Relocations
//!
//! COFF keeps addends in place (REL form), so the writer stores each
//! relocation's addend into the field it patches:
//!
//! | [`RelocKind`] | AMD64 | ARM64 | in-place value |
//! |---|---|---|---|
//! | `Abs64` | `IMAGE_REL_AMD64_ADDR64` | `IMAGE_REL_ARM64_ADDR64` | `A` |
//! | `Abs32`, `Abs32S` | `IMAGE_REL_AMD64_ADDR32` | `IMAGE_REL_ARM64_ADDR32` | `A` |
//! | `Pc32`, `Plt32` | `IMAGE_REL_AMD64_REL32` | `IMAGE_REL_ARM64_REL32` (`Pc32`) | `A + 4` |
//! | `Aarch64Call26` | — | `IMAGE_REL_ARM64_BRANCH26` | `imm26 = A / 4` |
//! | `Aarch64AdrPrelPgHi21` | — | `IMAGE_REL_ARM64_PAGEBASE_REL21` | `immhi:immlo = A` |
//! | `Aarch64AddAbsLo12Nc` | — | `IMAGE_REL_ARM64_PAGEOFFSET_12A` | `imm12 = A & 0xfff` |
//!
//! `REL32` is relative to the end of its 4-byte field, so an ELF-style
//! `S + A - P` becomes an in-place `A + 4` (a `call rel32` with the usual
//! `A = -4` stores 0). `Pc64` and `GotPcRel` have no COFF equivalent (there is
//! no GOT: Windows reaches imports through `__imp_` pointers), and an
//! architecture's relocations are rejected in the other's object; each is a
//! clear [`ObjectWriteError`].
//!
//! Output is little-endian and deterministic.

use crate::mc::format::{ObjectWriteError, align_up, log2_align, read_u32, write_u32};
use crate::mc::object::{ObjectModule, RelocKind, SectionKind, SymbolBinding, SymbolType, SymbolValue};

// ===========================================================================
// Constants (from the PE/COFF specification)
// ===========================================================================

/// `IMAGE_FILE_MACHINE_AMD64`.
pub const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
/// `IMAGE_FILE_MACHINE_ARM64`.
pub const IMAGE_FILE_MACHINE_ARM64: u16 = 0xAA64;

const FILE_HEADER_SIZE: usize = 20;
const SECTION_HEADER_SIZE: usize = 40;
const SYMBOL_SIZE: usize = 18;
const RELOC_SIZE: usize = 10;

const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
const IMAGE_SCN_LNK_NRELOC_OVFL: u32 = 0x0100_0000;
const IMAGE_SCN_MEM_DISCARDABLE: u32 = 0x0200_0000;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;
/// `IMAGE_SCN_ALIGN_1BYTES`; `IMAGE_SCN_ALIGN_<2^k>BYTES` is `(k + 1) << 20`.
const IMAGE_SCN_ALIGN_SHIFT: u32 = 20;
/// The largest encodable alignment, `IMAGE_SCN_ALIGN_8192BYTES` (2^13).
const MAX_ALIGN_LOG2: u32 = 13;

const IMAGE_SYM_UNDEFINED: i16 = 0;
const IMAGE_SYM_ABSOLUTE: i16 = -1;
const IMAGE_SYM_DTYPE_FUNCTION: u16 = 0x20;
const IMAGE_SYM_CLASS_EXTERNAL: u8 = 2;
const IMAGE_SYM_CLASS_STATIC: u8 = 3;
const IMAGE_SYM_CLASS_WEAK_EXTERNAL: u8 = 105;
const IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY: u32 = 1;
const IMAGE_WEAK_EXTERN_SEARCH_ALIAS: u32 = 3;

const IMAGE_REL_AMD64_ADDR64: u16 = 0x0001;
const IMAGE_REL_AMD64_ADDR32: u16 = 0x0002;
const IMAGE_REL_AMD64_REL32: u16 = 0x0004;

const IMAGE_REL_ARM64_ADDR32: u16 = 0x0001;
const IMAGE_REL_ARM64_BRANCH26: u16 = 0x0003;
const IMAGE_REL_ARM64_PAGEBASE_REL21: u16 = 0x0004;
const IMAGE_REL_ARM64_PAGEOFFSET_12A: u16 = 0x0006;
const IMAGE_REL_ARM64_ADDR64: u16 = 0x000E;
const IMAGE_REL_ARM64_REL32: u16 = 0x0011;

/// The COFF machine an object is written for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CoffMachine {
    /// x86-64 (`IMAGE_FILE_MACHINE_AMD64`).
    Amd64,
    /// AArch64 (`IMAGE_FILE_MACHINE_ARM64`).
    Arm64,
}

impl CoffMachine {
    /// The `Machine` field value.
    pub fn code(self) -> u16 {
        match self {
            CoffMachine::Amd64 => IMAGE_FILE_MACHINE_AMD64,
            CoffMachine::Arm64 => IMAGE_FILE_MACHINE_ARM64,
        }
    }
}

// ===========================================================================
// Mapping the neutral model onto COFF
// ===========================================================================

/// The COFF section name for a module section: `.rodata*` becomes `.rdata*`
/// (the PE convention); every other name is kept.
fn coff_section_name(name: &str) -> String {
    match name.strip_prefix(".rodata") {
        Some(rest) => format!(".rdata{rest}"),
        None => name.to_owned(),
    }
}

/// The `Characteristics` of a section of `kind` aligned to `align`.
fn section_characteristics(kind: SectionKind, align: u64) -> u32 {
    let flags = match kind {
        SectionKind::Text => IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ,
        SectionKind::Data => IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE,
        SectionKind::Rodata => IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ,
        SectionKind::Bss => IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE,
        SectionKind::Debug => {
            IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_DISCARDABLE
        }
        // Rejected by `write` before this is reached (PE/COFF thread-local
        // storage goes through the TLS directory, not section flags).
        SectionKind::TData | SectionKind::TBss => {
            IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE
        }
    };
    let k = log2_align(align).min(MAX_ALIGN_LOG2);
    flags | ((k + 1) << IMAGE_SCN_ALIGN_SHIFT)
}

/// How one relocation is encoded: its COFF type and how its addend is stored.
#[derive(Clone, Copy, Debug)]
enum Patch {
    /// Store the addend as a little-endian 64-bit value.
    Word64(i64),
    /// Store the addend as a little-endian 32-bit value.
    Word32(u32),
    /// Insert into the `imm26` field of a `b`/`bl` (bits 25:0, scaled by 4).
    Branch26(u32),
    /// Insert into the `immhi:immlo` fields of an `adrp` (bits 23:5, 30:29).
    Adr21(u32),
    /// Insert into the `imm12` field of an `add` (bits 21:10).
    Imm12(u32),
}

fn unsupported(kind: RelocKind, machine: CoffMachine) -> ObjectWriteError {
    let why = match kind {
        RelocKind::GotPcRel => " (PE/COFF has no GOT; reference imports through `__imp_` pointers)",
        RelocKind::Pc64 => " (PE/COFF has no 64-bit PC-relative relocation)",
        _ => "",
    };
    ObjectWriteError::new(format!(
        "relocation {kind:?} cannot be expressed in a {machine:?} COFF object{why}"
    ))
}

fn fits_i32(v: i64) -> bool {
    i32::try_from(v).is_ok()
}

/// The COFF relocation type and in-place patch for `kind` with addend `addend`.
fn map_reloc(kind: RelocKind, addend: i64, machine: CoffMachine) -> Result<(u16, Patch), ObjectWriteError> {
    let too_big = || {
        ObjectWriteError::new(format!(
            "addend {addend} of a {kind:?} relocation does not fit its COFF field"
        ))
    };
    let abs32 = || -> Result<Patch, ObjectWriteError> {
        match kind {
            RelocKind::Abs32 if u32::try_from(addend).is_ok() => Ok(Patch::Word32(addend as u32)),
            RelocKind::Abs32S if fits_i32(addend) => Ok(Patch::Word32(addend as i32 as u32)),
            _ => Err(too_big()),
        }
    };
    let rel32 = || -> Result<Patch, ObjectWriteError> {
        let stored = addend.checked_add(4).filter(|&v| fits_i32(v)).ok_or_else(too_big)?;
        Ok(Patch::Word32(stored as i32 as u32))
    };
    match machine {
        CoffMachine::Amd64 => match kind {
            RelocKind::Abs64 => Ok((IMAGE_REL_AMD64_ADDR64, Patch::Word64(addend))),
            RelocKind::Abs32 | RelocKind::Abs32S => Ok((IMAGE_REL_AMD64_ADDR32, abs32()?)),
            RelocKind::Pc32 | RelocKind::Plt32 => Ok((IMAGE_REL_AMD64_REL32, rel32()?)),
            other => Err(unsupported(other, machine)),
        },
        CoffMachine::Arm64 => match kind {
            RelocKind::Abs64 => Ok((IMAGE_REL_ARM64_ADDR64, Patch::Word64(addend))),
            RelocKind::Abs32 | RelocKind::Abs32S => Ok((IMAGE_REL_ARM64_ADDR32, abs32()?)),
            RelocKind::Pc32 => Ok((IMAGE_REL_ARM64_REL32, rel32()?)),
            RelocKind::Aarch64Call26 => {
                if addend % 4 != 0 || !(-(1 << 27)..(1 << 27)).contains(&addend) {
                    return Err(too_big());
                }
                Ok((IMAGE_REL_ARM64_BRANCH26, Patch::Branch26(((addend >> 2) as u32) & 0x03ff_ffff)))
            }
            RelocKind::Aarch64AdrPrelPgHi21 => {
                if !(-(1 << 20)..(1 << 20)).contains(&addend) {
                    return Err(too_big());
                }
                Ok((IMAGE_REL_ARM64_PAGEBASE_REL21, Patch::Adr21((addend as u32) & 0x001f_ffff)))
            }
            RelocKind::Aarch64AddAbsLo12Nc => {
                Ok((IMAGE_REL_ARM64_PAGEOFFSET_12A, Patch::Imm12((addend as u32) & 0xfff)))
            }
            other => Err(unsupported(other, machine)),
        },
    }
}

/// Apply `patch` to the field at `at` in `bytes`.
fn apply_patch(bytes: &mut [u8], at: usize, patch: Patch) -> Result<(), ObjectWriteError> {
    let width = match patch {
        Patch::Word64(_) => 8,
        _ => 4,
    };
    if at + width > bytes.len() {
        return Err(ObjectWriteError::new(format!(
            "relocation at offset {at:#x} runs past the end of its section"
        )));
    }
    match patch {
        Patch::Word64(v) => bytes[at..at + 8].copy_from_slice(&v.to_le_bytes()),
        Patch::Word32(v) => write_u32(bytes, at, v),
        Patch::Branch26(imm) => {
            let insn = read_u32(bytes, at);
            write_u32(bytes, at, (insn & !0x03ff_ffff) | imm);
        }
        Patch::Adr21(v) => {
            let insn = read_u32(bytes, at);
            let immlo = v & 3;
            let immhi = (v >> 2) & 0x7ffff;
            let cleared = insn & !((3 << 29) | (0x7ffff << 5));
            write_u32(bytes, at, cleared | (immlo << 29) | (immhi << 5));
        }
        Patch::Imm12(v) => {
            let insn = read_u32(bytes, at);
            write_u32(bytes, at, (insn & !(0xfff << 10)) | (v << 10));
        }
    }
    Ok(())
}

// ===========================================================================
// Writer
// ===========================================================================

/// The COFF string table: a 4-byte total size, then NUL-terminated strings.
#[derive(Debug)]
struct StringTable {
    buf: Vec<u8>,
}

impl StringTable {
    fn new() -> StringTable {
        StringTable { buf: vec![0; 4] }
    }

    fn add(&mut self, s: &str) -> u32 {
        let off = self.buf.len() as u32;
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
        off
    }

    fn finish(mut self) -> Vec<u8> {
        let size = self.buf.len() as u32;
        self.buf[0..4].copy_from_slice(&size.to_le_bytes());
        self.buf
    }
}

/// One 18-byte symbol record (auxiliary records are pushed as raw bytes).
struct SymRec {
    name: String,
    value: u32,
    section: i16,
    ty: u16,
    class: u8,
    aux: Vec<[u8; SYMBOL_SIZE]>,
}

/// The symbol-table index of each module symbol, and the records to write.
struct SymbolTable {
    records: Vec<SymRec>,
    /// Module symbol index -> COFF symbol-table index (counting aux records).
    index_of: Vec<u32>,
}

impl SymbolTable {
    /// The index the next pushed record gets.
    fn next_index(&self) -> u32 {
        self.records.iter().map(|r| 1 + r.aux.len() as u32).sum()
    }
}

/// The section-definition auxiliary record of a section symbol.
fn section_aux(length: u32, nrelocs: u16, number: u16) -> [u8; SYMBOL_SIZE] {
    let mut aux = [0u8; SYMBOL_SIZE];
    aux[0..4].copy_from_slice(&length.to_le_bytes());
    aux[4..6].copy_from_slice(&nrelocs.to_le_bytes());
    // NumberOfLinenumbers, CheckSum: 0. Number (COMDAT only) and Selection: 0.
    aux[12..14].copy_from_slice(&number.to_le_bytes());
    aux
}

/// The weak-external auxiliary record: the default symbol and the search mode.
fn weak_aux(tag_index: u32, characteristics: u32) -> [u8; SYMBOL_SIZE] {
    let mut aux = [0u8; SYMBOL_SIZE];
    aux[0..4].copy_from_slice(&tag_index.to_le_bytes());
    aux[4..8].copy_from_slice(&characteristics.to_le_bytes());
    aux
}

fn build_symbols(obj: &ObjectModule, reloc_counts: &[usize]) -> SymbolTable {
    let sections = obj.sections();
    let mut records: Vec<SymRec> = Vec::new();

    // A static section symbol + section-definition aux record per section.
    let mut section_sym = Vec::with_capacity(sections.len());
    let mut next = 0u32;
    for (i, s) in sections.iter().enumerate() {
        section_sym.push(next);
        let length = u32::try_from(s.size()).unwrap_or(u32::MAX);
        let nrelocs = u16::try_from(reloc_counts[i]).unwrap_or(u16::MAX);
        records.push(SymRec {
            name: coff_section_name(&s.name),
            value: 0,
            section: (i + 1) as i16,
            ty: 0,
            class: IMAGE_SYM_CLASS_STATIC,
            aux: vec![section_aux(length, nrelocs, 0)],
        });
        next += 2;
    }

    // A tag that makes weak-default names unique across the objects of a
    // link: the first strong global this object defines (itself unique).
    let tag = obj
        .symbols()
        .iter()
        .find(|s| s.binding == SymbolBinding::Global && !s.is_undefined())
        .map(|s| s.name.clone())
        .unwrap_or_default();
    let default_name = |name: &str| {
        if tag.is_empty() { format!(".weak.{name}.default") } else { format!(".weak.{name}.default.{tag}") }
    };

    let mut table = SymbolTable { records, index_of: vec![0; obj.symbols().len()] };
    for (i, s) in obj.symbols().iter().enumerate() {
        let ty = if s.kind == SymbolType::Func { IMAGE_SYM_DTYPE_FUNCTION } else { 0 };
        match (s.value, s.binding) {
            // A section symbol at the section start is the section's own symbol.
            (SymbolValue::Defined { section, offset: 0 }, _) if s.kind == SymbolType::Section => {
                table.index_of[i] = section_sym[section.index()];
            }
            (SymbolValue::Defined { section, offset }, SymbolBinding::Weak) => {
                // The definition itself goes under a unique default name; the
                // weak external names it as its alias.
                let default_index = table.next_index();
                table.records.push(SymRec {
                    name: default_name(&s.name),
                    value: offset as u32,
                    section: (section.index() + 1) as i16,
                    ty,
                    class: IMAGE_SYM_CLASS_EXTERNAL,
                    aux: Vec::new(),
                });
                table.index_of[i] = table.next_index();
                table.records.push(SymRec {
                    name: s.name.clone(),
                    value: 0,
                    section: IMAGE_SYM_UNDEFINED,
                    ty,
                    class: IMAGE_SYM_CLASS_WEAK_EXTERNAL,
                    aux: vec![weak_aux(default_index, IMAGE_WEAK_EXTERN_SEARCH_ALIAS)],
                });
            }
            (SymbolValue::Defined { section, offset }, binding) => {
                table.index_of[i] = table.next_index();
                table.records.push(SymRec {
                    name: s.name.clone(),
                    value: offset as u32,
                    section: (section.index() + 1) as i16,
                    ty,
                    class: if binding == SymbolBinding::Local {
                        IMAGE_SYM_CLASS_STATIC
                    } else {
                        IMAGE_SYM_CLASS_EXTERNAL
                    },
                    aux: Vec::new(),
                });
            }
            (SymbolValue::Undefined, SymbolBinding::Weak) => {
                // An unresolved weak reference falls back to an absolute zero.
                let default_index = table.next_index();
                table.records.push(SymRec {
                    name: default_name(&s.name),
                    value: 0,
                    section: IMAGE_SYM_ABSOLUTE,
                    ty: 0,
                    class: IMAGE_SYM_CLASS_EXTERNAL,
                    aux: Vec::new(),
                });
                table.index_of[i] = table.next_index();
                table.records.push(SymRec {
                    name: s.name.clone(),
                    value: 0,
                    section: IMAGE_SYM_UNDEFINED,
                    ty,
                    class: IMAGE_SYM_CLASS_WEAK_EXTERNAL,
                    aux: vec![weak_aux(default_index, IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY)],
                });
            }
            (SymbolValue::Undefined, _) => {
                table.index_of[i] = table.next_index();
                table.records.push(SymRec {
                    name: s.name.clone(),
                    value: 0,
                    section: IMAGE_SYM_UNDEFINED,
                    ty,
                    class: IMAGE_SYM_CLASS_EXTERNAL,
                    aux: Vec::new(),
                });
            }
        }
    }
    table
}

/// Write an 8-byte short name, or a string-table reference for a longer one:
/// `/offset` for a section name, `0000` + offset for a symbol name.
fn short_or_long_name(name: &str, strtab: &mut StringTable, section: bool) -> [u8; 8] {
    let mut out = [0u8; 8];
    if name.len() <= 8 {
        out[..name.len()].copy_from_slice(name.as_bytes());
    } else if section {
        let r = format!("/{}", strtab.add(name));
        out[..r.len()].copy_from_slice(r.as_bytes());
    } else {
        let off = strtab.add(name);
        out[4..8].copy_from_slice(&off.to_le_bytes());
    }
    out
}

/// Serialize `obj` as a COFF relocatable object for `machine`.
///
/// # Errors
///
/// Returns an [`ObjectWriteError`] for a relocation the machine's COFF
/// relocation set cannot express (see the [module docs](self)), an addend
/// that does not fit its in-place field, or a section larger than 4 GiB.
pub fn write(obj: &ObjectModule, machine: CoffMachine) -> Result<Vec<u8>, ObjectWriteError> {
    let sections = obj.sections();
    let n = sections.len();
    if n > 0x7fff {
        return Err(ObjectWriteError::new("too many sections for a COFF object"));
    }
    for s in sections {
        if s.size() > u64::from(u32::MAX) {
            return Err(ObjectWriteError::new(format!("section {} is larger than 4 GiB", s.name)));
        }
        if s.kind.is_tls() {
            return Err(ObjectWriteError::new(format!(
                "section {} holds thread-local storage, which this COFF writer does not support",
                s.name
            )));
        }
    }

    // --- per-section relocation records, with addends patched in place ---
    let mut contents: Vec<Vec<u8>> = sections.iter().map(|s| s.bytes.clone()).collect();
    let mut relocs: Vec<Vec<(u32, u32, u16)>> = vec![Vec::new(); n]; // (offset, symbol, type)
    let mut reloc_counts = vec![0usize; n];
    for r in obj.relocations() {
        reloc_counts[r.section.index()] += 1;
    }
    let symtab = build_symbols(obj, &reloc_counts);
    for r in obj.relocations() {
        let si = r.section.index();
        if sections[si].is_nobits() {
            return Err(ObjectWriteError::new(format!(
                "relocation in the zero-fill section {}",
                sections[si].name
            )));
        }
        let (ty, patch) = map_reloc(r.kind, r.addend, machine)?;
        apply_patch(&mut contents[si], r.offset as usize, patch)?;
        relocs[si].push((r.offset as u32, symtab.index_of[r.symbol.index()], ty));
    }

    // --- layout: header, section headers, then data + relocations ---
    let mut strtab = StringTable::new();
    let mut offset = FILE_HEADER_SIZE + n * SECTION_HEADER_SIZE;
    let mut data_ptr = vec![0u32; n];
    let mut reloc_ptr = vec![0u32; n];
    for (i, s) in sections.iter().enumerate() {
        if !s.is_nobits() && !contents[i].is_empty() {
            offset = align_up(offset as u64, 4) as usize;
            data_ptr[i] = offset as u32;
            offset += contents[i].len();
        }
        if !relocs[i].is_empty() {
            offset = align_up(offset as u64, 4) as usize;
            reloc_ptr[i] = offset as u32;
            let overflow = relocs[i].len() >= 0xffff;
            offset += (relocs[i].len() + usize::from(overflow)) * RELOC_SIZE;
        }
    }
    offset = align_up(offset as u64, 4) as usize;
    let symtab_ptr = offset as u32;
    let nsyms = symtab.next_index();

    let mut buf = Vec::with_capacity(offset + nsyms as usize * SYMBOL_SIZE + 64);

    // File header.
    buf.extend_from_slice(&machine.code().to_le_bytes());
    buf.extend_from_slice(&(n as u16).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // TimeDateStamp (deterministic)
    buf.extend_from_slice(&symtab_ptr.to_le_bytes());
    buf.extend_from_slice(&nsyms.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // SizeOfOptionalHeader
    buf.extend_from_slice(&0u16.to_le_bytes()); // Characteristics

    // Section headers.
    for (i, s) in sections.iter().enumerate() {
        let name = short_or_long_name(&coff_section_name(&s.name), &mut strtab, true);
        buf.extend_from_slice(&name);
        buf.extend_from_slice(&0u32.to_le_bytes()); // VirtualSize (0 in objects)
        buf.extend_from_slice(&0u32.to_le_bytes()); // VirtualAddress
        buf.extend_from_slice(&(s.size() as u32).to_le_bytes()); // SizeOfRawData
        buf.extend_from_slice(&data_ptr[i].to_le_bytes());
        buf.extend_from_slice(&reloc_ptr[i].to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // PointerToLinenumbers
        let mut characteristics = section_characteristics(s.kind, s.align);
        let count = if relocs[i].len() >= 0xffff {
            characteristics |= IMAGE_SCN_LNK_NRELOC_OVFL;
            0xffff
        } else {
            relocs[i].len() as u16
        };
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // NumberOfLinenumbers
        buf.extend_from_slice(&characteristics.to_le_bytes());
    }

    // Section data and relocations.
    for i in 0..n {
        if data_ptr[i] != 0 {
            buf.resize(data_ptr[i] as usize, 0);
            buf.extend_from_slice(&contents[i]);
        }
        if reloc_ptr[i] != 0 {
            buf.resize(reloc_ptr[i] as usize, 0);
            if relocs[i].len() >= 0xffff {
                // NRELOC_OVFL: the first record's VirtualAddress is the real
                // count, including itself.
                let total = relocs[i].len() as u32 + 1;
                buf.extend_from_slice(&total.to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
                buf.extend_from_slice(&0u16.to_le_bytes());
            }
            for &(off, sym, ty) in &relocs[i] {
                buf.extend_from_slice(&off.to_le_bytes());
                buf.extend_from_slice(&sym.to_le_bytes());
                buf.extend_from_slice(&ty.to_le_bytes());
            }
        }
    }

    // Symbol table.
    buf.resize(symtab_ptr as usize, 0);
    for rec in &symtab.records {
        buf.extend_from_slice(&short_or_long_name(&rec.name, &mut strtab, false));
        buf.extend_from_slice(&rec.value.to_le_bytes());
        buf.extend_from_slice(&rec.section.to_le_bytes());
        buf.extend_from_slice(&rec.ty.to_le_bytes());
        buf.push(rec.class);
        buf.push(rec.aux.len() as u8);
        for aux in &rec.aux {
            buf.extend_from_slice(aux);
        }
    }

    // String table.
    buf.extend_from_slice(&strtab.finish());
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::object::{Relocation, Section, Symbol};

    // ---- a small structural reader for our own output ----

    fn u16_at(b: &[u8], o: usize) -> u16 {
        u16::from_le_bytes([b[o], b[o + 1]])
    }
    fn u32_at(b: &[u8], o: usize) -> u32 {
        read_u32(b, o)
    }

    struct Shdr {
        name: String,
        size: u32,
        data: u32,
        relocs: u32,
        nrelocs: u16,
        flags: u32,
    }

    struct Sym {
        name: String,
        value: u32,
        section: i16,
        ty: u16,
        class: u8,
        naux: u8,
        index: u32,
        aux: Vec<u8>,
    }

    fn strtab(b: &[u8]) -> usize {
        u32_at(b, 8) as usize + u32_at(b, 12) as usize * SYMBOL_SIZE
    }

    fn cstr(b: &[u8], at: usize) -> String {
        let end = b[at..].iter().position(|&c| c == 0).unwrap();
        String::from_utf8(b[at..at + end].to_vec()).unwrap()
    }

    fn name8(b: &[u8], at: usize, section: bool) -> String {
        let raw = &b[at..at + 8];
        if section && raw[0] == b'/' {
            let n: usize = std::str::from_utf8(&raw[1..])
                .unwrap()
                .trim_end_matches('\0')
                .parse()
                .unwrap();
            return cstr(b, strtab(b) + n);
        }
        if !section && raw[0..4] == [0; 4] {
            return cstr(b, strtab(b) + u32_at(raw, 4) as usize);
        }
        String::from_utf8(raw.iter().copied().take_while(|&c| c != 0).collect()).unwrap()
    }

    fn shdrs(b: &[u8]) -> Vec<Shdr> {
        (0..u16_at(b, 2) as usize)
            .map(|i| {
                let o = FILE_HEADER_SIZE + i * SECTION_HEADER_SIZE;
                Shdr {
                    name: name8(b, o, true),
                    size: u32_at(b, o + 16),
                    data: u32_at(b, o + 20),
                    relocs: u32_at(b, o + 24),
                    nrelocs: u16_at(b, o + 32),
                    flags: u32_at(b, o + 36),
                }
            })
            .collect()
    }

    fn syms(b: &[u8]) -> Vec<Sym> {
        let base = u32_at(b, 8) as usize;
        let n = u32_at(b, 12);
        let mut out = Vec::new();
        let mut i = 0u32;
        while i < n {
            let o = base + i as usize * SYMBOL_SIZE;
            let naux = b[o + 17];
            out.push(Sym {
                name: name8(b, o, false),
                value: u32_at(b, o + 8),
                section: u16_at(b, o + 12) as i16,
                ty: u16_at(b, o + 14),
                class: b[o + 16],
                naux,
                index: i,
                aux: b[o + SYMBOL_SIZE..o + SYMBOL_SIZE * (1 + naux as usize)].to_vec(),
            });
            i += 1 + u32::from(naux);
        }
        out
    }

    fn relocs(b: &[u8], s: &Shdr) -> Vec<(u32, u32, u16)> {
        (0..s.nrelocs as usize)
            .map(|k| {
                let o = s.relocs as usize + k * RELOC_SIZE;
                (u32_at(b, o), u32_at(b, o + 4), u16_at(b, o + 8))
            })
            .collect()
    }

    fn sym_named<'a>(syms: &'a [Sym], name: &str) -> &'a Sym {
        syms.iter().find(|s| s.name == name).unwrap_or_else(|| panic!("no symbol {name}"))
    }

    /// x86-64: `main` calls `puts` (PLT32) and loads a string (PC32), plus a
    /// data pointer (ABS64) in `.data`, a local label and a `.bss`.
    fn x86_object() -> ObjectModule {
        let mut m = ObjectModule::new("t.o");
        let text = m.add_section(Section::new(".text", SectionKind::Text, 16));
        // lea rcx,[rip+msg]; call puts; ret
        m.section_mut(text).bytes =
            vec![0x48, 0x8d, 0x0d, 0, 0, 0, 0, 0xe8, 0, 0, 0, 0, 0xc3];
        let rodata = m.add_section(Section::new(".rodata", SectionKind::Rodata, 1));
        m.section_mut(rodata).bytes = b"hi\0".to_vec();
        let data = m.add_section(Section::new(".data", SectionKind::Data, 8));
        m.section_mut(data).bytes = vec![0; 8];
        let bss = m.add_section(Section::bss(".bss", 16, 64));
        m.add_symbol(Symbol::defined("main", SymbolBinding::Global, SymbolType::Func, text, 0, 13));
        let msg = m.add_symbol(Symbol::defined("msg", SymbolBinding::Local, SymbolType::Object, rodata, 0, 3));
        m.add_symbol(Symbol::defined("ptr", SymbolBinding::Global, SymbolType::Object, data, 0, 8));
        m.add_symbol(Symbol::defined(
            "a_rather_long_buffer_name",
            SymbolBinding::Global,
            SymbolType::Object,
            bss,
            0,
            64,
        ));
        let puts = m.reference_symbol("puts");
        m.add_relocation(Relocation { section: text, offset: 3, symbol: msg, kind: RelocKind::Pc32, addend: -4 });
        m.add_relocation(Relocation { section: text, offset: 8, symbol: puts, kind: RelocKind::Plt32, addend: -4 });
        m.add_relocation(Relocation { section: data, offset: 0, symbol: msg, kind: RelocKind::Abs64, addend: 1 });
        m
    }

    #[test]
    fn file_header_and_sections() {
        let b = write(&x86_object(), CoffMachine::Amd64).unwrap();
        assert_eq!(u16_at(&b, 0), IMAGE_FILE_MACHINE_AMD64);
        assert_eq!(u16_at(&b, 2), 4, "four sections");
        assert_eq!(u32_at(&b, 4), 0, "deterministic time stamp");
        assert_eq!(u16_at(&b, 16), 0, "no optional header in an object");

        let sh = shdrs(&b);
        let names: Vec<&str> = sh.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, [".text", ".rdata", ".data", ".bss"]);
        assert_eq!(
            sh[0].flags,
            IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ | (5 << 20),
            ".text is code, 16-byte aligned"
        );
        assert_eq!(sh[1].flags, IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | (1 << 20));
        assert_eq!(
            sh[2].flags,
            IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE | (4 << 20)
        );
        assert_eq!(
            sh[3].flags,
            IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE | (5 << 20)
        );
        // .bss: a size but no file data.
        assert_eq!((sh[3].size, sh[3].data), (64, 0));
        // Raw data lies within the file.
        assert_eq!(&b[sh[1].data as usize..][..3], b"hi\0");
        assert_eq!(sh[0].size, 13);
    }

    #[test]
    fn symbols_and_section_symbols() {
        let b = write(&x86_object(), CoffMachine::Amd64).unwrap();
        let s = syms(&b);
        // Four section symbols, each with one aux record, come first.
        for (k, name) in [".text", ".rdata", ".data", ".bss"].iter().enumerate() {
            assert_eq!(s[k].name, *name);
            assert_eq!(s[k].class, IMAGE_SYM_CLASS_STATIC);
            assert_eq!(s[k].section, k as i16 + 1);
            assert_eq!(s[k].naux, 1);
            assert_eq!(s[k].index, 2 * k as u32);
        }
        // The .text aux record: length 13, two relocations.
        assert_eq!(u32_at(&s[0].aux, 0), 13);
        assert_eq!(u16_at(&s[0].aux, 4), 2);

        let main = sym_named(&s, "main");
        assert_eq!((main.class, main.section, main.ty), (IMAGE_SYM_CLASS_EXTERNAL, 1, 0x20));
        let msg = sym_named(&s, "msg");
        assert_eq!((msg.class, msg.section, msg.ty), (IMAGE_SYM_CLASS_STATIC, 2, 0));
        let puts = sym_named(&s, "puts");
        assert_eq!((puts.class, puts.section, puts.value), (IMAGE_SYM_CLASS_EXTERNAL, 0, 0));
        // A name longer than 8 bytes lives in the string table.
        let long = sym_named(&s, "a_rather_long_buffer_name");
        assert_eq!(long.section, 4);
    }

    #[test]
    fn relocations_store_addends_in_place() {
        let b = write(&x86_object(), CoffMachine::Amd64).unwrap();
        let sh = shdrs(&b);
        let s = syms(&b);
        let text = relocs(&b, &sh[0]);
        let msg = sym_named(&s, "msg").index;
        let puts = sym_named(&s, "puts").index;
        assert_eq!(text, [(3, msg, IMAGE_REL_AMD64_REL32), (8, puts, IMAGE_REL_AMD64_REL32)]);
        // `A = -4` on a field ending the instruction stores 0 (REL32 is
        // relative to the end of the field).
        let t = sh[0].data as usize;
        assert_eq!(&b[t + 3..t + 7], &[0; 4]);
        assert_eq!(&b[t + 8..t + 12], &[0; 4]);
        // ABS64 with A = 1 stores 1.
        let data = relocs(&b, &sh[2]);
        assert_eq!(data, [(0, msg, IMAGE_REL_AMD64_ADDR64)]);
        assert_eq!(&b[sh[2].data as usize..][..8], &1u64.to_le_bytes());
    }

    #[test]
    fn rel32_with_trailing_immediate_keeps_the_distance() {
        // `mov dword [rip+x], imm32`: the field is followed by 4 bytes, so
        // the ELF addend is -8 and the COFF in-place value is -4.
        let mut m = ObjectModule::new("t");
        let t = m.add_section(Section::new(".text", SectionKind::Text, 1));
        m.section_mut(t).bytes = vec![0xc7, 0x05, 0, 0, 0, 0, 1, 0, 0, 0];
        let x = m.reference_symbol("x");
        m.add_relocation(Relocation { section: t, offset: 2, symbol: x, kind: RelocKind::Pc32, addend: -8 });
        let b = write(&m, CoffMachine::Amd64).unwrap();
        let sh = shdrs(&b);
        let d = sh[0].data as usize;
        assert_eq!(&b[d + 2..d + 6], &(-4i32).to_le_bytes());
        assert_eq!(&b[d + 6..d + 10], &1u32.to_le_bytes(), "immediate untouched");
    }

    #[test]
    fn arm64_instruction_relocations() {
        let mut m = ObjectModule::new("a");
        let t = m.add_section(Section::new(".text", SectionKind::Text, 4));
        // adrp x0, 0; add x0, x0, #0; bl 0; ret
        let words: [u32; 4] = [0x9000_0000, 0x9100_0000, 0x9400_0000, 0xd65f_03c0];
        m.section_mut(t).bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        m.add_symbol(Symbol::defined("f", SymbolBinding::Global, SymbolType::Func, t, 0, 16));
        let g = m.reference_symbol("g");
        let h = m.reference_symbol("h");
        m.add_relocation(Relocation { section: t, offset: 0, symbol: g, kind: RelocKind::Aarch64AdrPrelPgHi21, addend: 0x1235 });
        m.add_relocation(Relocation { section: t, offset: 4, symbol: g, kind: RelocKind::Aarch64AddAbsLo12Nc, addend: 0x1235 });
        m.add_relocation(Relocation { section: t, offset: 8, symbol: h, kind: RelocKind::Aarch64Call26, addend: 8 });
        let b = write(&m, CoffMachine::Arm64).unwrap();
        assert_eq!(u16_at(&b, 0), IMAGE_FILE_MACHINE_ARM64);
        let sh = shdrs(&b);
        let types: Vec<u16> = relocs(&b, &sh[0]).iter().map(|r| r.2).collect();
        assert_eq!(
            types,
            [IMAGE_REL_ARM64_PAGEBASE_REL21, IMAGE_REL_ARM64_PAGEOFFSET_12A, IMAGE_REL_ARM64_BRANCH26]
        );
        let d = sh[0].data as usize;
        let adrp = read_u32(&b, d);
        // immlo = 0x1235 & 3 = 1, immhi = 0x1235 >> 2 = 0x48d.
        assert_eq!((adrp >> 29) & 3, 1);
        assert_eq!((adrp >> 5) & 0x7ffff, 0x48d);
        assert_eq!(adrp & 0x9f00_001f, 0x9000_0000, "opcode and Rd kept");
        let add = read_u32(&b, d + 4);
        assert_eq!((add >> 10) & 0xfff, 0x235);
        let bl = read_u32(&b, d + 8);
        assert_eq!(bl, 0x9400_0002, "imm26 = 8 / 4");
        assert_eq!(read_u32(&b, d + 12), 0xd65f_03c0);
    }

    #[test]
    fn weak_symbols_become_weak_externals() {
        let mut m = ObjectModule::new("w");
        let t = m.add_section(Section::new(".text", SectionKind::Text, 1));
        m.section_mut(t).bytes = vec![0xc3, 0xc3];
        m.add_symbol(Symbol::defined("strong", SymbolBinding::Global, SymbolType::Func, t, 0, 1));
        m.add_symbol(Symbol::defined("wdef", SymbolBinding::Weak, SymbolType::Func, t, 1, 1));
        let wref = m.add_symbol(Symbol::undefined("wref", SymbolBinding::Weak));
        m.add_relocation(Relocation { section: t, offset: 0, symbol: wref, kind: RelocKind::Abs32, addend: 0 });
        // Abs32 needs 4 bytes: grow the section.
        m.section_mut(t).bytes.extend_from_slice(&[0; 4]);
        let b = write(&m, CoffMachine::Amd64).unwrap();
        let s = syms(&b);

        let def = sym_named(&s, ".weak.wdef.default.strong");
        assert_eq!((def.class, def.section, def.value), (IMAGE_SYM_CLASS_EXTERNAL, 1, 1));
        let wdef = sym_named(&s, "wdef");
        assert_eq!((wdef.class, wdef.section, wdef.naux), (IMAGE_SYM_CLASS_WEAK_EXTERNAL, 0, 1));
        assert_eq!(u32_at(&wdef.aux, 0), def.index);
        assert_eq!(u32_at(&wdef.aux, 4), IMAGE_WEAK_EXTERN_SEARCH_ALIAS);

        let zero = sym_named(&s, ".weak.wref.default.strong");
        assert_eq!((zero.section, zero.value), (IMAGE_SYM_ABSOLUTE, 0));
        let wref_sym = sym_named(&s, "wref");
        assert_eq!(wref_sym.class, IMAGE_SYM_CLASS_WEAK_EXTERNAL);
        assert_eq!(u32_at(&wref_sym.aux, 0), zero.index);
        assert_eq!(u32_at(&wref_sym.aux, 4), IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY);
        // The relocation names the weak external itself.
        let sh = shdrs(&b);
        assert_eq!(relocs(&b, &sh[0])[0].1, wref_sym.index);
    }

    #[test]
    fn long_section_names_use_the_string_table() {
        let mut m = ObjectModule::new("d");
        let d = m.add_section(Section::new(".debug_abbrev", SectionKind::Debug, 1));
        m.section_mut(d).bytes = vec![1, 2, 3];
        let b = write(&m, CoffMachine::Amd64).unwrap();
        let sh = shdrs(&b);
        assert_eq!(sh[0].name, ".debug_abbrev");
        assert_eq!(&b[FILE_HEADER_SIZE..FILE_HEADER_SIZE + 1], b"/");
        assert_ne!(sh[0].flags & IMAGE_SCN_MEM_DISCARDABLE, 0);
    }

    #[test]
    fn relocation_overflow_escape() {
        let mut m = ObjectModule::new("big");
        let t = m.add_section(Section::new(".data", SectionKind::Data, 8));
        let n = 0x1_0000usize;
        m.section_mut(t).bytes = vec![0; n * 8];
        let x = m.reference_symbol("x");
        for k in 0..n {
            m.add_relocation(Relocation { section: t, offset: 8 * k as u64, symbol: x, kind: RelocKind::Abs64, addend: 0 });
        }
        let b = write(&m, CoffMachine::Amd64).unwrap();
        let sh = shdrs(&b);
        assert_eq!(sh[0].nrelocs, 0xffff);
        assert_ne!(sh[0].flags & IMAGE_SCN_LNK_NRELOC_OVFL, 0);
        assert_eq!(u32_at(&b, sh[0].relocs as usize), n as u32 + 1);
    }

    #[test]
    fn unsupported_relocations_are_errors() {
        for (kind, machine) in [
            (RelocKind::GotPcRel, CoffMachine::Amd64),
            (RelocKind::Pc64, CoffMachine::Amd64),
            (RelocKind::Aarch64Call26, CoffMachine::Amd64),
            (RelocKind::Plt32, CoffMachine::Arm64),
            (RelocKind::ThumbCall, CoffMachine::Arm64),
            (RelocKind::ThumbMovwAbsNc, CoffMachine::Amd64),
            (RelocKind::ThumbMovtAbs, CoffMachine::Arm64),
            (RelocKind::AvrCall, CoffMachine::Amd64),
            (RelocKind::Avr13Pcrel, CoffMachine::Arm64),
            (RelocKind::Avr16Pm, CoffMachine::Amd64),
            (RelocKind::AvrLo8Ldi, CoffMachine::Arm64),
            (RelocKind::AvrHi8Ldi, CoffMachine::Amd64),
            (RelocKind::AvrLo8LdiPm, CoffMachine::Arm64),
            (RelocKind::AvrHi8LdiPm, CoffMachine::Amd64),
        ] {
            let mut m = ObjectModule::new("e");
            let t = m.add_section(Section::new(".text", SectionKind::Text, 1));
            m.section_mut(t).bytes = vec![0; 8];
            let x = m.reference_symbol("x");
            m.add_relocation(Relocation { section: t, offset: 0, symbol: x, kind, addend: 0 });
            let e = write(&m, machine).unwrap_err();
            assert!(e.message().contains(&format!("{kind:?}")), "{e}");
        }
        // An addend that does not fit the in-place field.
        let mut m = ObjectModule::new("e");
        let t = m.add_section(Section::new(".text", SectionKind::Text, 1));
        m.section_mut(t).bytes = vec![0; 8];
        let x = m.reference_symbol("x");
        m.add_relocation(Relocation { section: t, offset: 0, symbol: x, kind: RelocKind::Pc32, addend: 1 << 40 });
        assert!(write(&m, CoffMachine::Amd64).is_err());
    }

    #[test]
    fn output_is_deterministic() {
        assert_eq!(write(&x86_object(), CoffMachine::Amd64), write(&x86_object(), CoffMachine::Amd64));
    }
}
