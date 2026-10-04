//! An ELF relocatable-object (`ET_REL`) writer — ELF64 for x86-64, and ELF32 /
//! ELF64 in either byte order for any machine an [`ElfTarget`] describes —
//! implemented from the ELF specification (ROADMAP Phase 6).
//!
//! This turns the framework's target-independent [`ObjectModule`] into a
//! standard SysV-ABI ELF64 relocatable object that a system linker (or our own
//! future linker) can consume. It is a clean-room implementation of the file
//! format from its public specification — the `Elf64_*` structure layouts,
//! constants, and x86-64 relocation numbers are those of the published standard,
//! not copied from any toolchain (tenet T1).
//!
//! # What it emits
//!
//! - an `Elf64_Ehdr` with class `ELFCLASS64`, data
//!   `ELFDATA2LSB`, type `ET_REL`, machine `EM_X86_64`;
//! - one section header per user [`Section`], plus `.symtab`, `.strtab`, a
//!   `.rela.<name>` for every section that has relocations, and `.shstrtab`;
//! - a symbol table with the null symbol first, then all local symbols, then
//!   global/weak ones (as ELF requires), and `.symtab`'s `sh_info` set to the
//!   first non-local index;
//! - `Elf64_Rela` entries mapping each generic [`RelocKind`] to its x86-64
//!   relocation number.
//!
//! Output is little-endian and fully deterministic: sections, symbols, and
//! relocations are emitted in the module's insertion order.
//!
//! # Other targets: ELF32, big-endian, `REL`
//!
//! [`write`](fn@write) is [`write_with`] for [`ElfTarget::X86_64`]. [`write_with`] takes
//! any [`ElfTarget`] — class, byte order, `e_machine`, `e_flags`, `REL` or
//! `RELA`, and the machine's relocation numbering — and lays out the same
//! object with the gABI's `Elf32_*` structures when the class is 32-bit
//! (52-byte header, 40-byte section headers, 16-byte symbols whose field order
//! differs from `Elf64_Sym`, `r_info = sym << 8 | type`, 4-byte table
//! alignment). A `REL` target gets `.rel<name>` sections and the addends
//! stored in the patched fields. A value that does not fit an ELF32 field is a
//! [`ElfError`], never a silent truncation.
//!
//! [`Section`]: crate::mc::object::Section

use crate::ir::Endian;
use crate::mc::object::{
    ObjectModule, RelocKind, SectionKind, SymbolBinding, SymbolType, SymbolValue, SymbolVisibility,
    write_field,
};

// ===========================================================================
// ELF constants (from the ELF-64 specification)
// ===========================================================================

const EI_NIDENT: usize = 16;

const ELFMAG: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS32: u8 = 1;
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const ELFDATA2MSB: u8 = 2;
const EV_CURRENT: u8 = 1;
const ELFOSABI_SYSV: u8 = 0;

const ET_REL: u16 = 1;
/// The `e_machine` value for x86-64.
pub const EM_X86_64: u16 = 62;

const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHT_REL: u32 = 9;

const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_TLS: u64 = 0x400;

const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STB_WEAK: u8 = 2;
const STV_DEFAULT: u8 = 0;
const STV_HIDDEN: u8 = 2;
const STV_PROTECTED: u8 = 3;

const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const STT_SECTION: u8 = 3;
const STT_TLS: u8 = 6;

const SHN_UNDEF: u16 = 0;

// x86-64 relocation type numbers.
const R_X86_64_64: u32 = 1;
const R_X86_64_PC32: u32 = 2;
const R_X86_64_PLT32: u32 = 4;
const R_X86_64_GOTPCREL: u32 = 9;
const R_X86_64_32: u32 = 10;
const R_X86_64_32S: u32 = 11;
const R_X86_64_PC64: u32 = 24;
const R_X86_64_16: u32 = 12;
const R_X86_64_TLSGD: u32 = 19;
const R_X86_64_GOTTPOFF: u32 = 22;
const R_X86_64_TPOFF32: u32 = 23;

/// The size in bytes of an `Elf64_Ehdr`.
const EHDR_SIZE: u64 = 64;
/// The size in bytes of an `Elf64_Shdr`.
const SHDR_SIZE: u64 = 64;
/// The size in bytes of an `Elf64_Sym`.
const SYM_SIZE: u64 = 24;
/// The size in bytes of an `Elf64_Rela`.
const RELA_SIZE: u64 = 24;

/// Map a generic [`RelocKind`] to its x86-64 ELF relocation number.
fn x86_64_reloc(kind: RelocKind) -> u32 {
    match kind {
        RelocKind::Abs64 => R_X86_64_64,
        RelocKind::Abs32 => R_X86_64_32,
        RelocKind::Abs16 => R_X86_64_16,
        RelocKind::Abs32S => R_X86_64_32S,
        RelocKind::Pc32 => R_X86_64_PC32,
        RelocKind::Pc64 => R_X86_64_PC64,
        RelocKind::Plt32 => R_X86_64_PLT32,
        RelocKind::GotPcRel => R_X86_64_GOTPCREL,
        RelocKind::TpOff32 => R_X86_64_TPOFF32,
        RelocKind::GotTpOff => R_X86_64_GOTTPOFF,
        RelocKind::TlsGd => R_X86_64_TLSGD,
        // AArch64, Thumb and AVR relocation kinds never appear in an x86-64 ELF
        // object (those backends do not emit through this mapping).
        RelocKind::Aarch64Call26
        | RelocKind::Aarch64AdrPrelPgHi21
        | RelocKind::Aarch64AddAbsLo12Nc
        | RelocKind::ThumbCall
        | RelocKind::ThumbMovwAbsNc
        | RelocKind::ThumbMovtAbs
        | RelocKind::AvrCall
        | RelocKind::Avr13Pcrel
        | RelocKind::Avr16Pm
        | RelocKind::AvrLo8Ldi
        | RelocKind::AvrHi8Ldi
        | RelocKind::AvrLo8LdiPm
        | RelocKind::AvrHi8LdiPm => {
            unreachable!("relocation kind {kind:?} in an x86-64 ELF object")
        }
    }
}

fn section_flags_type(kind: SectionKind) -> (u64, u32) {
    match kind {
        SectionKind::Text => (SHF_ALLOC | SHF_EXECINSTR, SHT_PROGBITS),
        SectionKind::Data => (SHF_ALLOC | SHF_WRITE, SHT_PROGBITS),
        SectionKind::Rodata => (SHF_ALLOC, SHT_PROGBITS),
        SectionKind::Bss => (SHF_ALLOC | SHF_WRITE, SHT_NOBITS),
        // Debug sections are present in the file but not allocated at run time.
        SectionKind::Debug => (0, SHT_PROGBITS),
        SectionKind::TData => (SHF_ALLOC | SHF_WRITE | SHF_TLS, SHT_PROGBITS),
        SectionKind::TBss => (SHF_ALLOC | SHF_WRITE | SHF_TLS, SHT_NOBITS),
    }
}

fn binding_code(b: SymbolBinding) -> u8 {
    match b {
        SymbolBinding::Local => STB_LOCAL,
        SymbolBinding::Global => STB_GLOBAL,
        SymbolBinding::Weak => STB_WEAK,
    }
}

fn visibility_code(v: SymbolVisibility) -> u8 {
    match v {
        SymbolVisibility::Default => STV_DEFAULT,
        SymbolVisibility::Hidden => STV_HIDDEN,
        SymbolVisibility::Protected => STV_PROTECTED,
    }
}

fn symtype_code(t: SymbolType) -> u8 {
    match t {
        SymbolType::NoType => STT_NOTYPE,
        SymbolType::Object => STT_OBJECT,
        SymbolType::Func => STT_FUNC,
        SymbolType::Section => STT_SECTION,
        SymbolType::Tls => STT_TLS,
    }
}

// ===========================================================================
// Small helpers
// ===========================================================================

/// A growable string table (`.strtab`/`.shstrtab`): a leading NUL byte, then
/// each added string NUL-terminated. Returns the offset of each string.
#[derive(Debug, Default)]
struct StringTable {
    buf: Vec<u8>,
}

impl StringTable {
    fn new() -> Self {
        StringTable { buf: vec![0] }
    }

    /// Add `s` and return its byte offset. The empty string maps to offset 0.
    fn add(&mut self, s: &str) -> u32 {
        if s.is_empty() {
            return 0;
        }
        let off = self.buf.len() as u32;
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
        off
    }
}

/// Append `n` zero bytes to `buf` until its length is a multiple of `align`.
fn pad_to(buf: &mut Vec<u8>, align: u64) {
    if align <= 1 {
        return;
    }
    let mask = align - 1;
    let rem = buf.len() as u64 & mask;
    if rem != 0 {
        let pad = (align - rem) as usize;
        buf.resize(buf.len() + pad, 0);
    }
}

// ===========================================================================
// Target description
// ===========================================================================

/// The ELF file class: 32-bit (`ELFCLASS32`) or 64-bit (`ELFCLASS64`)
/// structures and addresses.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ElfClass {
    /// `ELFCLASS32`: `Elf32_*` structures, 32-bit addresses and offsets (Arm
    /// Cortex-M, AVR, i386, wasm-adjacent toolchains).
    Elf32,
    /// `ELFCLASS64`: `Elf64_*` structures (x86-64, AArch64, RISC-V 64).
    Elf64,
}

/// How relocations are recorded.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RelocFormat {
    /// `SHT_RELA` (`.rela<name>`): each entry carries an explicit addend.
    Rela,
    /// `SHT_REL` (`.rel<name>`): the addend is stored in the patched field
    /// itself (the *implicit addend*). The writer stores it there for every
    /// kind that patches a whole field (the absolute and PC-relative kinds) and
    /// for the Thumb-2 instruction kinds (in the instruction's immediate
    /// fields); any other kind that patches a bitfield inside an instruction
    /// (the `Aarch64*` kinds) must carry a zero addend, its target having
    /// encoded any addend in the instruction already.
    Rel,
}

/// Everything the writer needs to know about the target of an object: the
/// file class and byte order, the `e_machine` / `e_flags` values, the
/// relocation format, and the mapping of each generic [`RelocKind`] onto the
/// machine's relocation numbers (`None` for a kind the machine cannot express).
///
/// [`ElfTarget::X86_64`] is the x86-64 System V target [`write`](fn@write) uses. A new
/// backend describes its own (e.g. `EM_ARM` with `R_ARM_ABS32` / `REL`, or
/// `EM_AVR` with `R_AVR_16` / `RELA`, both ELF32) and calls [`write_with`].
#[derive(Clone, Copy, Debug)]
pub struct ElfTarget {
    /// 32- or 64-bit ELF.
    pub class: ElfClass,
    /// The byte order of every multi-byte field (`ELFDATA2LSB` / `ELFDATA2MSB`).
    pub endian: Endian,
    /// The `e_machine` value.
    pub machine: u16,
    /// The `e_flags` value (processor-specific: the Arm EABI version, the AVR
    /// architecture, ...).
    pub flags: u32,
    /// `REL` or `RELA` relocation sections.
    pub reloc_format: RelocFormat,
    /// The machine's relocation number for a generic kind, or `None`.
    pub reloc_type: fn(RelocKind) -> Option<u32>,
}

impl ElfTarget {
    /// 32-bit Arm EABI version 5 (Cortex-M Thumb-2 code): ELF32,
    /// little-endian, `EM_ARM`, `e_flags` = EABI v5 with the soft-float
    /// procedure call standard, `REL` relocations with implicit addends (a
    /// Thumb instruction relocation keeps its addend in the instruction, see
    /// [`write_thumb_field`](crate::mc::object::write_thumb_field)).
    pub const ARM: ElfTarget = ElfTarget {
        class: ElfClass::Elf32,
        endian: Endian::Little,
        machine: EM_ARM,
        flags: EF_ARM_EABI_VER5 | EF_ARM_ABI_FLOAT_SOFT,
        reloc_format: RelocFormat::Rel,
        reloc_type: arm_reloc_type,
    };

    /// x86-64 System V: ELF64, little-endian, `EM_X86_64`, `RELA`.
    pub const X86_64: ElfTarget = ElfTarget {
        class: ElfClass::Elf64,
        endian: Endian::Little,
        machine: EM_X86_64,
        flags: 0,
        reloc_format: RelocFormat::Rela,
        reloc_type: x86_64_reloc_type,
    };
}

/// [`x86_64_reloc`] as a total mapping (`None` for the instruction kinds of
/// other machines).
fn x86_64_reloc_type(kind: RelocKind) -> Option<u32> {
    if kind.is_instruction_field() || kind.is_avr() {
        return None;
    }
    Some(x86_64_reloc(kind))
}

/// The `e_machine` value for 32-bit Arm.
pub const EM_ARM: u16 = 40;
/// Arm `e_flags`: EABI version 5.
pub const EF_ARM_EABI_VER5: u32 = 0x0500_0000;
/// Arm `e_flags`: the base procedure call standard (floating-point arguments
/// in core registers, the soft-float ABI).
pub const EF_ARM_ABI_FLOAT_SOFT: u32 = 0x200;

// Arm relocation type numbers (ELF for the Arm Architecture, "Relocation codes").
const R_ARM_ABS32: u32 = 2;
const R_ARM_REL32: u32 = 3;
const R_ARM_ABS16: u32 = 5;
const R_ARM_THM_CALL: u32 = 10;
const R_ARM_THM_MOVW_ABS_NC: u32 = 47;
const R_ARM_THM_MOVT_ABS: u32 = 48;

/// The 32-bit Arm relocation number of a generic kind: `R_ARM_ABS32`/`ABS16`
/// for data, `R_ARM_REL32` for a PC-relative word, and the Thumb-2
/// instruction relocations `R_ARM_THM_CALL`, `R_ARM_THM_MOVW_ABS_NC` and
/// `R_ARM_THM_MOVT_ABS`. `None` for anything else.
fn arm_reloc_type(kind: RelocKind) -> Option<u32> {
    Some(match kind {
        RelocKind::Abs32 => R_ARM_ABS32,
        RelocKind::Abs16 => R_ARM_ABS16,
        RelocKind::Pc32 => R_ARM_REL32,
        RelocKind::ThumbCall => R_ARM_THM_CALL,
        RelocKind::ThumbMovwAbsNc => R_ARM_THM_MOVW_ABS_NC,
        RelocKind::ThumbMovtAbs => R_ARM_THM_MOVT_ABS,
        _ => return None,
    })
}

/// Why an object could not be written for an [`ElfTarget`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ElfError {
    /// The target's relocation mapping has no number for this kind.
    UnsupportedReloc(RelocKind),
    /// A value does not fit its ELF32 field (an offset, size, symbol value or
    /// addend beyond 32 bits). `what` names the field.
    FieldOverflow {
        /// The field that overflowed.
        what: &'static str,
        /// The offending value.
        value: i128,
    },
    /// A `REL`-format relocation's addend cannot be stored in its field: the
    /// kind patches an instruction bitfield (the addend must be zero), or the
    /// addend does not fit the field's width.
    ImplicitAddend {
        /// The relocation kind.
        kind: RelocKind,
        /// The addend that could not be stored.
        addend: i64,
    },
}

impl std::fmt::Display for ElfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElfError::UnsupportedReloc(k) => write!(f, "relocation kind {k:?} is not supported by this ELF target"),
            ElfError::FieldOverflow { what, value } => write!(f, "{what} {value} does not fit an ELF32 field"),
            ElfError::ImplicitAddend { kind, addend } => {
                write!(f, "cannot store addend {addend} of a {kind:?} relocation in its field (REL format)")
            }
        }
    }
}

impl std::error::Error for ElfError {}

// ===========================================================================
// Class- and endian-aware field output
// ===========================================================================

/// The per-class structure sizes (`Ehdr`, `Shdr`, `Sym`, `Rel`, `Rela`) and the
/// natural alignment of the symbol and relocation tables.
struct Sizes {
    ehdr: u64,
    shdr: u64,
    sym: u64,
    rel: u64,
    rela: u64,
    table_align: u64,
}

impl ElfClass {
    fn sizes(self) -> Sizes {
        match self {
            ElfClass::Elf64 => Sizes { ehdr: EHDR_SIZE, shdr: SHDR_SIZE, sym: SYM_SIZE, rel: 16, rela: RELA_SIZE, table_align: 8 },
            ElfClass::Elf32 => Sizes { ehdr: 52, shdr: 40, sym: 16, rel: 8, rela: 12, table_align: 4 },
        }
    }
}

/// A byte sink writing ELF fields in the target's byte order, with the
/// class-dependent "address/offset" (`Elf*_Addr`/`Off`/`Xword`) width.
struct Out {
    buf: Vec<u8>,
    class: ElfClass,
    endian: Endian,
}

impl Out {
    fn int(&mut self, v: u64, width: usize) {
        let le = v.to_le_bytes();
        match self.endian {
            Endian::Little => self.buf.extend_from_slice(&le[..width]),
            Endian::Big => self.buf.extend(le[..width].iter().rev()),
        }
    }
    fn u16(&mut self, v: u16) {
        self.int(u64::from(v), 2);
    }
    fn u32(&mut self, v: u32) {
        self.int(u64::from(v), 4);
    }
    /// An `Elf*_Addr` / `Elf*_Off` / `Elf64_Xword`-or-`Elf32_Word` field. ELF32
    /// values were range-checked when the layout was computed.
    fn word(&mut self, v: u64) {
        match self.class {
            ElfClass::Elf64 => self.int(v, 8),
            ElfClass::Elf32 => self.int(v, 4),
        }
    }
    /// A signed addend (`Elf64_Sxword` / `Elf32_Sword`).
    fn sword(&mut self, v: i64) {
        self.word(v as u64);
    }
}

/// Check that `value` fits an unsigned ELF32 field (always true for ELF64).
fn fits(class: ElfClass, what: &'static str, value: u64) -> Result<u64, ElfError> {
    if class == ElfClass::Elf32 && value > u64::from(u32::MAX) {
        return Err(ElfError::FieldOverflow { what, value: i128::from(value) });
    }
    Ok(value)
}

/// Write one section header (`Elf64_Shdr` / `Elf32_Shdr`).
#[allow(clippy::too_many_arguments)]
fn write_shdr(
    o: &mut Out,
    name: u32,
    kind: u32,
    flags: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,
) {
    o.u32(name);
    o.u32(kind);
    o.word(flags);
    o.word(0); // sh_addr
    o.word(offset);
    o.word(size);
    o.u32(link);
    o.u32(info);
    o.word(addralign);
    o.word(entsize);
}

/// Write one symbol (`Elf64_Sym` / `Elf32_Sym`, whose field orders differ).
fn write_sym(o: &mut Out, name: u32, info: u8, other: u8, shndx: u16, value: u64, size: u64) {
    o.u32(name);
    match o.class {
        ElfClass::Elf64 => {
            o.buf.push(info);
            o.buf.push(other); // st_other: the visibility in its low two bits
            o.u16(shndx);
            o.word(value);
            o.word(size);
        }
        ElfClass::Elf32 => {
            o.word(value);
            o.word(size);
            o.buf.push(info);
            o.buf.push(other); // st_other: the visibility in its low two bits
            o.u16(shndx);
        }
    }
}

/// Write one relocation entry (`Elf*_Rel` or `Elf*_Rela`). `r_info` packs the
/// symbol index and type as `sym << 32 | type` (ELF64) or `sym << 8 | type`
/// (ELF32, type in the low byte).
fn write_reloc(o: &mut Out, rela: bool, offset: u64, sym_index: u32, ty: u32, addend: i64) {
    o.word(offset);
    match o.class {
        ElfClass::Elf64 => o.int((u64::from(sym_index) << 32) | u64::from(ty), 8),
        ElfClass::Elf32 => o.u32((sym_index << 8) | (ty & 0xff)),
    }
    if rela {
        o.sword(addend);
    }
}

// ===========================================================================
// Writer
// ===========================================================================

/// Serialize `obj` to a valid ELF64 x86-64 relocatable object image.
///
/// The returned bytes form a complete `ET_REL` file: an ELF header, the section
/// contents, a symbol and string table, `.rela.*` relocation sections, and the
/// section header table. Output is deterministic. This is
/// [`write_with`]`(obj, &`[`ElfTarget::X86_64`]`)`.
///
/// # Panics
///
/// If `obj` holds a relocation x86-64 cannot express (an AArch64 kind).
pub fn write(obj: &ObjectModule) -> Vec<u8> {
    write_with(obj, &ElfTarget::X86_64).unwrap_or_else(|e| panic!("x86-64 ELF object: {e}"))
}

/// Serialize `obj` to a relocatable (`ET_REL`) object for `target`: ELF32 or
/// ELF64, either byte order, `REL` or `RELA` relocations (see [`ElfTarget`]).
/// Output is deterministic. With [`ElfTarget::X86_64`] the bytes are exactly
/// those of [`write`](fn@write).
pub fn write_with(obj: &ObjectModule, target: &ElfTarget) -> Result<Vec<u8>, ElfError> {
    let class = target.class;
    let sz = class.sizes();
    let rela = target.reloc_format == RelocFormat::Rela;
    let sections = obj.sections();
    let n = sections.len();

    // --- assign ELF section indices ---
    // 0 = null; 1..=n = user sections; then symtab, strtab, rel(a).*, shstrtab.
    let symtab_index = (n + 1) as u32;
    let strtab_index = (n + 2) as u32;
    let mut rela_index_of: Vec<Option<u32>> = vec![None; n];
    let mut next = n as u32 + 3;
    let has_relocs = |i: usize| obj.relocations().iter().any(|r| r.section.index() == i);
    for (i, slot) in rela_index_of.iter_mut().enumerate() {
        if has_relocs(i) {
            *slot = Some(next);
            next += 1;
        }
    }
    let shstrtab_index = next;
    next += 1;
    let total_sections = next as usize;

    // The ELF section index a user section lives at.
    let user_elf_index = |i: usize| (i + 1) as u16;

    // --- order symbols: null, then locals, then globals/weaks ---
    let mut order: Vec<usize> = Vec::with_capacity(obj.symbols().len());
    for (i, s) in obj.symbols().iter().enumerate() {
        if matches!(s.binding, SymbolBinding::Local) {
            order.push(i);
        }
    }
    let nlocal = order.len();
    for (i, s) in obj.symbols().iter().enumerate() {
        if !matches!(s.binding, SymbolBinding::Local) {
            order.push(i);
        }
    }
    // obj symbol index -> symtab index (null occupies slot 0).
    let mut symtab_of = vec![0u32; obj.symbols().len()];
    for (pos, &obj_idx) in order.iter().enumerate() {
        symtab_of[obj_idx] = (pos + 1) as u32;
    }
    let first_global = (nlocal + 1) as u32; // sh_info for .symtab

    // --- build string tables ---
    let mut strtab = StringTable::new();
    let mut sym_name_off: Vec<u32> = vec![0; obj.symbols().len()];
    for (i, s) in obj.symbols().iter().enumerate() {
        sym_name_off[i] = strtab.add(&s.name);
    }

    let mut shstrtab = StringTable::new();
    let mut sec_name_off: Vec<u32> = Vec::with_capacity(n);
    for s in sections {
        sec_name_off.push(shstrtab.add(&s.name));
    }
    let symtab_name_off = shstrtab.add(".symtab");
    let strtab_name_off = shstrtab.add(".strtab");
    let mut rela_name_off: Vec<u32> = vec![0; n];
    let rel_prefix = if rela { ".rela" } else { ".rel" };
    for (i, s) in sections.iter().enumerate() {
        if rela_index_of[i].is_some() {
            let name = format!("{rel_prefix}{}", s.name);
            rela_name_off[i] = shstrtab.add(&name);
        }
    }
    let shstrtab_name_off = shstrtab.add(".shstrtab");

    let new_out = || Out { buf: Vec::new(), class, endian: target.endian };

    // --- build .symtab content ---
    let mut symtab = new_out();
    write_sym(&mut symtab, 0, 0, 0, SHN_UNDEF, 0, 0); // null symbol
    for &obj_idx in &order {
        let s = &obj.symbols()[obj_idx];
        let info = (binding_code(s.binding) << 4) | symtype_code(s.kind);
        let (shndx, value) = match s.value {
            SymbolValue::Defined { section, offset } => (user_elf_index(section.index()), offset),
            SymbolValue::Undefined => (SHN_UNDEF, 0),
        };
        let value = fits(class, "symbol value", value)?;
        let size = fits(class, "symbol size", s.size)?;
        let other = visibility_code(s.visibility);
        write_sym(&mut symtab, sym_name_off[obj_idx], info, other, shndx, value, size);
    }

    // --- build .rel(a).* content per user section ---
    // In REL form the addend moves into the patched field, so the affected
    // sections' bytes are copied and patched.
    let mut patched: Vec<Option<Vec<u8>>> = vec![None; n];
    let mut rela_data: Vec<Out> = (0..n).map(|_| new_out()).collect();
    for r in obj.relocations() {
        let i = r.section.index();
        let ty = (target.reloc_type)(r.kind).ok_or(ElfError::UnsupportedReloc(r.kind))?;
        let sym_index = symtab_of[r.symbol.index()];
        let offset = fits(class, "relocation offset", r.offset)?;
        if rela {
            if class == ElfClass::Elf32 && i32::try_from(r.addend).is_err() {
                return Err(ElfError::FieldOverflow { what: "relocation addend", value: i128::from(r.addend) });
            }
        } else {
            let stored = if r.kind.is_thumb() {
                // Thumb-2 instruction fields hold their implicit addend.
                let bytes = patched[i].get_or_insert_with(|| sections[i].bytes.clone());
                crate::mc::object::write_thumb_field(bytes, r.offset as usize, r.kind, r.addend)
            } else if r.kind.is_instruction_field() {
                r.addend == 0
            } else {
                let bytes = patched[i].get_or_insert_with(|| sections[i].bytes.clone());
                let (at, width) = (r.offset as usize, r.kind.field_width());
                at + width <= bytes.len() && write_field(bytes, at, width, r.addend, target.endian)
            };
            if !stored {
                return Err(ElfError::ImplicitAddend { kind: r.kind, addend: r.addend });
            }
        }
        write_reloc(&mut rela_data[i], rela, offset, sym_index, ty, r.addend);
    }

    // --- lay out the file, recording each section's file offset ---
    let mut buf = vec![0u8; sz.ehdr as usize];

    let mut user_offset = vec![0u64; n];
    for (i, s) in sections.iter().enumerate() {
        pad_to(&mut buf, s.align.max(1));
        user_offset[i] = buf.len() as u64;
        if !s.is_nobits() {
            buf.extend_from_slice(patched[i].as_deref().unwrap_or(&s.bytes));
        }
    }

    pad_to(&mut buf, sz.table_align);
    let symtab_offset = buf.len() as u64;
    buf.extend_from_slice(&symtab.buf);

    let strtab_offset = buf.len() as u64;
    buf.extend_from_slice(&strtab.buf);

    let mut rela_offset = vec![0u64; n];
    for (i, data) in rela_data.iter().enumerate() {
        if rela_index_of[i].is_some() {
            pad_to(&mut buf, sz.table_align);
            rela_offset[i] = buf.len() as u64;
            buf.extend_from_slice(&data.buf);
        }
    }

    let shstrtab_offset = buf.len() as u64;
    buf.extend_from_slice(&shstrtab.buf);

    // --- section header table ---
    pad_to(&mut buf, sz.table_align);
    let shoff = fits(class, "section header offset", buf.len() as u64)?;
    for s in sections {
        fits(class, "section size", s.size())?;
    }
    let mut sh = Out { buf, class, endian: target.endian };

    // 0: null section header.
    write_shdr(&mut sh, 0, SHT_NULL, 0, 0, 0, 0, 0, 0, 0);

    // 1..=n: user sections.
    for (i, s) in sections.iter().enumerate() {
        let (flags, kind) = section_flags_type(s.kind);
        write_shdr(
            &mut sh,
            sec_name_off[i],
            kind,
            flags,
            user_offset[i],
            s.size(),
            0,
            0,
            s.align.max(1),
            0,
        );
    }

    // .symtab
    write_shdr(
        &mut sh,
        symtab_name_off,
        SHT_SYMTAB,
        0,
        symtab_offset,
        symtab.buf.len() as u64,
        strtab_index,
        first_global,
        sz.table_align,
        sz.sym,
    );

    // .strtab
    write_shdr(
        &mut sh,
        strtab_name_off,
        SHT_STRTAB,
        0,
        strtab_offset,
        strtab.buf.len() as u64,
        0,
        0,
        1,
        0,
    );

    // .rel(a).* sections, in user-section order.
    for (i, data) in rela_data.iter().enumerate() {
        if rela_index_of[i].is_some() {
            write_shdr(
                &mut sh,
                rela_name_off[i],
                if rela { SHT_RELA } else { SHT_REL },
                0,
                rela_offset[i],
                data.buf.len() as u64,
                symtab_index,
                u32::from(user_elf_index(i)),
                sz.table_align,
                if rela { sz.rela } else { sz.rel },
            );
        }
    }

    // .shstrtab
    write_shdr(
        &mut sh,
        shstrtab_name_off,
        SHT_STRTAB,
        0,
        shstrtab_offset,
        shstrtab.buf.len() as u64,
        0,
        0,
        1,
        0,
    );
    let mut buf = sh.buf;

    // --- fill in the ELF header ---
    let mut ident = [0u8; EI_NIDENT];
    ident[0..4].copy_from_slice(&ELFMAG);
    ident[4] = if class == ElfClass::Elf64 { ELFCLASS64 } else { ELFCLASS32 };
    ident[5] = if target.endian == Endian::Little { ELFDATA2LSB } else { ELFDATA2MSB };
    ident[6] = EV_CURRENT;
    ident[7] = ELFOSABI_SYSV;
    let mut ehdr = Out { buf: Vec::with_capacity(sz.ehdr as usize), class, endian: target.endian };
    ehdr.buf.extend_from_slice(&ident);
    ehdr.u16(ET_REL);
    ehdr.u16(target.machine);
    ehdr.u32(1); // e_version
    ehdr.word(0); // e_entry
    ehdr.word(0); // e_phoff
    ehdr.word(shoff); // e_shoff
    ehdr.u32(target.flags); // e_flags
    ehdr.u16(sz.ehdr as u16); // e_ehsize
    ehdr.u16(0); // e_phentsize
    ehdr.u16(0); // e_phnum
    ehdr.u16(sz.shdr as u16); // e_shentsize
    ehdr.u16(total_sections as u16); // e_shnum
    ehdr.u16(shstrtab_index as u16); // e_shstrndx
    debug_assert_eq!(ehdr.buf.len(), sz.ehdr as usize);
    buf[0..sz.ehdr as usize].copy_from_slice(&ehdr.buf);

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::object::{Section, Symbol};

    // ---- a tiny structural re-parser, used only to validate our own output ----

    fn rd_u16(b: &[u8], o: usize) -> u16 {
        u16::from_le_bytes([b[o], b[o + 1]])
    }
    fn rd_u32(b: &[u8], o: usize) -> u32 {
        u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }
    fn rd_u64(b: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
    }
    fn rd_i64(b: &[u8], o: usize) -> i64 {
        i64::from_le_bytes(b[o..o + 8].try_into().unwrap())
    }

    struct Shdr {
        name: u32,
        kind: u32,
        flags: u64,
        offset: u64,
        size: u64,
        link: u32,
        info: u32,
        entsize: u64,
    }

    fn parse_shdrs(b: &[u8]) -> Vec<Shdr> {
        let shoff = rd_u64(b, 40) as usize;
        let shentsize = rd_u16(b, 58) as usize;
        let shnum = rd_u16(b, 60) as usize;
        let mut out = Vec::new();
        for i in 0..shnum {
            let o = shoff + i * shentsize;
            out.push(Shdr {
                name: rd_u32(b, o),
                kind: rd_u32(b, o + 4),
                flags: rd_u64(b, o + 8),
                offset: rd_u64(b, o + 24),
                size: rd_u64(b, o + 32),
                link: rd_u32(b, o + 40),
                info: rd_u32(b, o + 44),
                entsize: rd_u64(b, o + 56),
            });
        }
        out
    }

    fn cstr(b: &[u8], base: usize, name: u32) -> String {
        let start = base + name as usize;
        let mut end = start;
        while b[end] != 0 {
            end += 1;
        }
        String::from_utf8_lossy(&b[start..end]).into_owned()
    }

    /// Build a tiny object: a `.text` with a few bytes, a global function symbol
    /// named `main`, an undefined `puts`, and a PLT32 call relocation to it.
    fn tiny_object() -> ObjectModule {
        let mut m = ObjectModule::new("tiny.o");
        let mut text = Section::new(".text", SectionKind::Text, 16);
        // push rbp; mov rbp,rsp; call rel32(=0); pop rbp; ret
        text.bytes = vec![0x55, 0x48, 0x89, 0xe5, 0xe8, 0, 0, 0, 0, 0x5d, 0xc3];
        let text_id = m.add_section(text);
        m.add_symbol(Symbol::defined(
            "main",
            SymbolBinding::Global,
            SymbolType::Func,
            text_id,
            0,
            11,
        ));
        let puts = m.add_symbol(Symbol::undefined("puts", SymbolBinding::Global));
        m.add_relocation(crate::mc::object::Relocation {
            section: text_id,
            offset: 5,
            symbol: puts,
            kind: RelocKind::Plt32,
            addend: -4,
        });
        m
    }

    #[test]
    fn header_fields_are_correct() {
        let b = write(&tiny_object());
        // Magic, class, data, version, osabi.
        assert_eq!(&b[0..4], &[0x7f, b'E', b'L', b'F']);
        assert_eq!(b[4], ELFCLASS64);
        assert_eq!(b[5], ELFDATA2LSB);
        assert_eq!(b[6], EV_CURRENT);
        assert_eq!(b[7], ELFOSABI_SYSV);
        // Type ET_REL, machine EM_X86_64=62.
        assert_eq!(rd_u16(&b, 16), ET_REL);
        assert_eq!(rd_u16(&b, 18), EM_X86_64);
        assert_eq!(rd_u16(&b, 18), 62);
        // ehsize/shentsize.
        assert_eq!(rd_u16(&b, 52), 64);
        assert_eq!(rd_u16(&b, 58), 64);
        // e_phnum == 0 for a relocatable object.
        assert_eq!(rd_u16(&b, 56), 0);
    }

    #[test]
    fn section_headers_are_consistent() {
        let b = write(&tiny_object());
        let shdrs = parse_shdrs(&b);
        // null + .text + .symtab + .strtab + .rela.text + .shstrtab = 6.
        assert_eq!(shdrs.len(), 6);
        assert_eq!(shdrs[0].kind, SHT_NULL);

        let shstrndx = rd_u16(&b, 62) as usize;
        let shstr_base = shdrs[shstrndx].offset as usize;
        let names: Vec<String> =
            shdrs.iter().map(|s| cstr(&b, shstr_base, s.name)).collect();
        assert_eq!(names[0], "");
        assert_eq!(names[1], ".text");
        assert!(names.contains(&".symtab".to_owned()));
        assert!(names.contains(&".strtab".to_owned()));
        assert!(names.contains(&".rela.text".to_owned()));
        assert!(names.contains(&".shstrtab".to_owned()));

        // .text flags: ALLOC | EXECINSTR, PROGBITS.
        assert_eq!(shdrs[1].kind, SHT_PROGBITS);
        assert_eq!(shdrs[1].flags, SHF_ALLOC | SHF_EXECINSTR);
        // Every section's [offset, offset+size) lies within the file (except
        // NOBITS, of which this object has none).
        for s in &shdrs {
            if s.kind != SHT_NULL && s.kind != SHT_NOBITS {
                assert!(s.offset + s.size <= b.len() as u64, "section out of bounds");
            }
        }
    }

    #[test]
    fn symtab_and_strtab_are_well_formed() {
        let b = write(&tiny_object());
        let shdrs = parse_shdrs(&b);
        let symtab = shdrs.iter().find(|s| s.kind == SHT_SYMTAB).unwrap();
        assert_eq!(symtab.entsize, SYM_SIZE);
        // sh_link points to a STRTAB section.
        assert_eq!(shdrs[symtab.link as usize].kind, SHT_STRTAB);
        let strtab = &shdrs[symtab.link as usize];

        // Symbols: null + main(local? no, global) + puts(global). main and puts
        // are both global, so 0 locals -> sh_info == 1.
        let count = symtab.size / SYM_SIZE;
        assert_eq!(count, 3);
        assert_eq!(symtab.info, 1, "first global symbol index");

        // Resolve names via the linked strtab.
        let strbase = strtab.offset as usize;
        let symbase = symtab.offset as usize;
        let mut found_main = false;
        let mut found_puts_undef = false;
        for i in 0..count as usize {
            let o = symbase + i * SYM_SIZE as usize;
            let name_off = rd_u32(&b, o);
            let name = cstr(&b, strbase, name_off);
            let shndx = rd_u16(&b, o + 6);
            let info = b[o + 4];
            if name == "main" {
                found_main = true;
                assert_eq!(info >> 4, STB_GLOBAL);
                assert_eq!(info & 0xf, STT_FUNC);
                assert_eq!(shndx, 1); // defined in .text (elf index 1)
            }
            if name == "puts" {
                found_puts_undef = true;
                assert_eq!(shndx, SHN_UNDEF);
            }
        }
        assert!(found_main && found_puts_undef);
    }

    #[test]
    fn rela_section_is_well_formed() {
        let b = write(&tiny_object());
        let shdrs = parse_shdrs(&b);
        let rela = shdrs.iter().find(|s| s.kind == SHT_RELA).unwrap();
        assert_eq!(rela.entsize, RELA_SIZE);
        assert_eq!(shdrs[rela.link as usize].kind, SHT_SYMTAB);
        // sh_info identifies the section being relocated (.text at index 1).
        assert_eq!(rela.info, 1);

        let count = rela.size / RELA_SIZE;
        assert_eq!(count, 1);
        let o = rela.offset as usize;
        let r_offset = rd_u64(&b, o);
        let r_info = rd_u64(&b, o + 8);
        let r_addend = rd_i64(&b, o + 16);
        assert_eq!(r_offset, 5);
        assert_eq!((r_info & 0xffff_ffff) as u32, R_X86_64_PLT32);
        assert_eq!(r_addend, -4);
        // The referenced symbol index must be a valid symtab entry.
        let sym_index = (r_info >> 32) as usize;
        let symtab = shdrs.iter().find(|s| s.kind == SHT_SYMTAB).unwrap();
        assert!((sym_index as u64) < symtab.size / SYM_SIZE);
    }

    #[test]
    fn all_reloc_kinds_map() {
        // Exhaustive mapping sanity, so a new kind can't silently fall through.
        assert_eq!(x86_64_reloc(RelocKind::Abs64), R_X86_64_64);
        assert_eq!(x86_64_reloc(RelocKind::Abs32), R_X86_64_32);
        assert_eq!(x86_64_reloc(RelocKind::Abs32S), R_X86_64_32S);
        assert_eq!(x86_64_reloc(RelocKind::Pc32), R_X86_64_PC32);
        assert_eq!(x86_64_reloc(RelocKind::Pc64), R_X86_64_PC64);
        assert_eq!(x86_64_reloc(RelocKind::Plt32), R_X86_64_PLT32);
        assert_eq!(x86_64_reloc(RelocKind::GotPcRel), R_X86_64_GOTPCREL);
    }

    #[test]
    fn multiple_sections_and_local_ordering() {
        // An object with data/rodata/bss and a mix of local and global symbols:
        // check locals precede globals and .bss is NOBITS occupying no file space.
        let mut m = ObjectModule::new("multi.o");
        let text = m.add_section(Section::new(".text", SectionKind::Text, 16));
        m.section_mut(text).bytes = vec![0x90, 0xc3];
        let data = m.add_section(Section::new(".data", SectionKind::Data, 8));
        m.section_mut(data).bytes = vec![0; 16];
        let rodata = m.add_section(Section::new(".rodata", SectionKind::Rodata, 1));
        m.section_mut(rodata).bytes = b"msg\0".to_vec();
        let bss = m.add_section(Section::bss(".bss", 16, 256));

        m.add_symbol(Symbol::defined(
            "g_main",
            SymbolBinding::Global,
            SymbolType::Func,
            text,
            0,
            2,
        ));
        m.add_symbol(Symbol::defined(
            "l_tmp",
            SymbolBinding::Local,
            SymbolType::Object,
            data,
            0,
            16,
        ));
        m.add_symbol(Symbol::defined(
            "w_cache",
            SymbolBinding::Weak,
            SymbolType::Object,
            bss,
            0,
            256,
        ));

        let b = write(&m);
        let shdrs = parse_shdrs(&b);
        let bss_shdr = shdrs
            .iter()
            .find(|s| s.kind == SHT_NOBITS)
            .expect(".bss present");
        assert_eq!(bss_shdr.size, 256);

        let symtab = shdrs.iter().find(|s| s.kind == SHT_SYMTAB).unwrap();
        // 1 local user symbol -> first global at index 2 (null + local).
        assert_eq!(symtab.info, 2);
        // All local-binding symbols must appear before the first global.
        let symbase = symtab.offset as usize;
        for i in 1..symtab.info as usize {
            let info = b[symbase + i * SYM_SIZE as usize + 4];
            assert_eq!(info >> 4, STB_LOCAL, "symbol {i} should be local");
        }
    }

    // ---- ELF32 / big-endian / REL ----

    /// A test-only ELF32 target: Arm (`EM_ARM` = 40) with `REL` relocations,
    /// `R_ARM_ABS32` = 2 and `R_ARM_ABS16` = 5 (Arm ELF ABI numbering).
    const ARM_LIKE: ElfTarget = ElfTarget {
        class: ElfClass::Elf32,
        endian: Endian::Little,
        machine: 40,
        flags: 0x0500_0000, // EABI version 5
        reloc_format: RelocFormat::Rel,
        reloc_type: |k| match k {
            RelocKind::Abs32 => Some(2),
            RelocKind::Abs16 => Some(5),
            _ => None,
        },
    };

    /// A test-only ELF32 target: AVR (`EM_AVR` = 83) with `RELA` relocations,
    /// `R_AVR_32` = 1 and `R_AVR_16` = 4.
    const AVR_LIKE: ElfTarget = ElfTarget {
        class: ElfClass::Elf32,
        endian: Endian::Little,
        machine: 83,
        flags: 0,
        reloc_format: RelocFormat::Rela,
        reloc_type: |k| match k {
            RelocKind::Abs32 => Some(1),
            RelocKind::Abs16 => Some(4),
            _ => None,
        },
    };

    /// A data object: a local `.data` table holding two pointers, one 4-byte
    /// (`kind32`) to `ext + 8` and one 2-byte to `ext + 2`, plus a global
    /// symbol `tbl` over it and the undefined `ext`.
    fn data_object() -> ObjectModule {
        let mut m = ObjectModule::new("d.o");
        let mut data = Section::new(".data", SectionKind::Data, 4);
        data.bytes = vec![0; 8];
        let d = m.add_section(data);
        m.add_symbol(Symbol::defined("tbl", SymbolBinding::Global, SymbolType::Object, d, 0, 6));
        let ext = m.add_symbol(Symbol::undefined("ext", SymbolBinding::Global));
        for (offset, kind, addend) in [(0, RelocKind::Abs32, 8), (4, RelocKind::Abs16, 2)] {
            m.add_relocation(crate::mc::object::Relocation { section: d, offset, symbol: ext, kind, addend });
        }
        m
    }

    #[test]
    fn elf32_rela_layout() {
        let b = write_with(&data_object(), &AVR_LIKE).expect("writes");
        assert_eq!(b[4], ELFCLASS32);
        assert_eq!(b[5], ELFDATA2LSB);
        assert_eq!(rd_u16(&b, 18), 83);
        assert_eq!(rd_u16(&b, 40), 52, "e_ehsize");
        assert_eq!(rd_u16(&b, 46), 40, "e_shentsize");
        let shoff = rd_u32(&b, 32) as usize;
        let shnum = rd_u16(&b, 48) as usize;
        // null, .data, .symtab, .strtab, .rela.data, .shstrtab.
        assert_eq!(shnum, 6);
        let sh = |i: usize, field: usize| rd_u32(&b, shoff + i * 40 + field);
        assert_eq!(sh(4, 4), SHT_RELA);
        assert_eq!(sh(4, 36), 12, "Elf32_Rela entsize");
        assert_eq!(sh(2, 36), 16, "Elf32_Sym entsize");
        // The relocations: r_info = sym << 8 | type, explicit 32-bit addends.
        let rel = sh(4, 16) as usize;
        assert_eq!((rd_u32(&b, rel), rd_u32(&b, rel + 4) & 0xff, rd_u32(&b, rel + 8)), (0, 1, 8));
        assert_eq!((rd_u32(&b, rel + 12), rd_u32(&b, rel + 16) & 0xff, rd_u32(&b, rel + 20)), (4, 4, 2));
        // The data keeps its zero fields in RELA form.
        let data = sh(1, 16) as usize;
        assert_eq!(&b[data..data + 8], &[0; 8]);
    }

    #[test]
    fn elf32_rel_stores_addends_in_place() {
        let b = write_with(&data_object(), &ARM_LIKE).expect("writes");
        let shoff = rd_u32(&b, 32) as usize;
        let sh = |i: usize, field: usize| rd_u32(&b, shoff + i * 40 + field);
        assert_eq!(sh(4, 4), SHT_REL);
        assert_eq!(sh(4, 36), 8, "Elf32_Rel entsize");
        assert_eq!(rd_u32(&b, 36), 0x0500_0000, "e_flags");
        let data = sh(1, 16) as usize;
        assert_eq!(&b[data..data + 8], &[8, 0, 0, 0, 2, 0, 0, 0]);
        // A kind the target cannot express is an error, not a panic.
        let mut m = data_object();
        let s = m.symbol_id("ext").unwrap();
        m.add_relocation(crate::mc::object::Relocation {
            section: crate::mc::object::SectionId::from_index(0),
            offset: 0,
            symbol: s,
            kind: RelocKind::Pc32,
            addend: 0,
        });
        assert_eq!(write_with(&m, &ARM_LIKE), Err(ElfError::UnsupportedReloc(RelocKind::Pc32)));
    }

    #[test]
    fn elf32_big_endian_and_overflow() {
        let be = ElfTarget { endian: Endian::Big, machine: 20, ..AVR_LIKE };
        let b = write_with(&data_object(), &be).expect("writes");
        assert_eq!(b[5], ELFDATA2MSB);
        assert_eq!(u16::from_be_bytes([b[18], b[19]]), 20);
        assert_eq!(u16::from_be_bytes([b[40], b[41]]), 52);
        // A symbol past 4 GiB does not fit ELF32.
        let mut m = ObjectModule::new("big.o");
        let d = m.add_section(Section::bss(".bss", 1, 1 << 33));
        m.add_symbol(Symbol::defined("huge", SymbolBinding::Global, SymbolType::Object, d, 1 << 32, 1));
        assert!(matches!(write_with(&m, &AVR_LIKE), Err(ElfError::FieldOverflow { what: "symbol value", .. })));
        assert!(write_with(&m, &ElfTarget::X86_64).is_ok(), "fine in ELF64");
    }

    #[test]
    fn x86_64_target_matches_write() {
        assert_eq!(write_with(&tiny_object(), &ElfTarget::X86_64).unwrap(), write(&tiny_object()));
    }

    /// `readelf` accepts both ELF32 objects and decodes their relocations by
    /// machine (skipped when `readelf` is absent).
    #[test]
    fn readelf_parses_elf32_objects() {
        if !tool_available("readelf") {
            eprintln!("skipping: readelf is not available");
            return;
        }
        for (target, want) in [(&ARM_LIKE, ["ELF32", "R_ARM_ABS32", "R_ARM_ABS16"]), (&AVR_LIKE, ["ELF32", "R_AVR_32", "R_AVR_16"])] {
            let bytes = write_with(&data_object(), target).expect("writes");
            let path = std::env::temp_dir().join(format!("lf_elf32_test_{}_{}.o", std::process::id(), target.machine));
            std::fs::write(&path, &bytes).expect("write temp object");
            let output = std::process::Command::new("readelf").arg("-a").arg("-W").arg(&path).output().expect("run readelf");
            let _ = std::fs::remove_file(&path);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(output.status.success() && stderr.trim().is_empty(), "readelf complained: {stderr}");
            for w in want.iter().chain(&["tbl", "ext", "REL (Relocatable file)"]) {
                assert!(stdout.contains(w), "readelf output lacks `{w}`:\n{stdout}");
            }
        }
    }

    #[test]
    fn output_is_deterministic() {
        let a = write(&tiny_object());
        let b = write(&tiny_object());
        assert_eq!(a, b);
    }

    // ---- optional external cross-check with readelf / llvm-readobj ----

    fn tool_available(cmd: &str) -> bool {
        std::process::Command::new(cmd)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[test]
    fn readelf_parses_our_object() {
        let tool = if tool_available("readelf") {
            "readelf"
        } else if tool_available("llvm-readobj") {
            "llvm-readobj"
        } else {
            eprintln!("skipping: neither readelf nor llvm-readobj is available");
            return;
        };

        let bytes = write(&tiny_object());
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lf_elf_test_{}.o", std::process::id()));
        std::fs::write(&path, &bytes).expect("write temp object");

        let arg = if tool == "readelf" { "-a" } else { "--all" };
        let output = std::process::Command::new(tool)
            .arg(arg)
            .arg(&path)
            .output()
            .expect("run reader tool");
        let _ = std::fs::remove_file(&path);

        assert!(
            output.status.success(),
            "{tool} rejected our object: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        // The tool must recognize the type/machine and see our symbol + reloc.
        assert!(stdout.contains("REL") || stdout.contains("Relocatable"));
        assert!(stdout.contains("main"));
        assert!(stdout.contains("puts"));
    }
}
