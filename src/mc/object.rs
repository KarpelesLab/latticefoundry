//! The target-independent object model: sections, symbols, relocations, and the
//! [`ObjectModule`] that aggregates them (ROADMAP Phase 6).
//!
//! This is the framework's neutral representation of a *relocatable object*,
//! sitting between the machine-code [emitter](crate::mc::emit) that produces
//! section bytes and the concrete serializers — our own
//! [`.lfo`](crate::mc::lfo) format and the standard
//! [ELF64 writer](crate::mc::elf). Nothing here knows any real ISA's opcodes or
//! any file format's byte layout: a target's encoder fills sections with bytes
//! and records relocations against symbols, and a writer maps this model onto a
//! file format.
//!
//! # Model
//!
//! - A [`Section`] is a named blob of bytes (or, for `.bss`, a reserved zeroed
//!   size) with a [`SectionKind`] and an alignment.
//! - A [`Symbol`] names a location: either **defined** at an offset inside a
//!   section, or **undefined** (an external reference the linker resolves). It
//!   carries a [`SymbolBinding`] (local/global/weak), a [`SymbolType`], and a
//!   [`SymbolVisibility`] (default/protected/hidden).
//! - A [`Relocation`] records that a field at some offset inside a section must
//!   be patched with the address of a [`Symbol`], according to a
//!   [`RelocKind`], plus a RELA-style `addend`.
//!
//! Everything is index/`Copy`-handle based ([`SectionId`], [`SymbolId`]) and
//! stored in insertion order, so a module serializes deterministically (tenet
//! T5). A name→[`SymbolId`] map is kept purely for interning lookups and does
//! not influence output order.

use crate::support::hash::DetHashMap;

/// A `Copy` handle to a [`Section`] within an [`ObjectModule`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct SectionId(u32);

impl SectionId {
    /// The dense index this handle addresses.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Reconstruct a handle from its dense index (for deserialization).
    #[inline]
    pub fn from_index(i: usize) -> SectionId {
        SectionId(i as u32)
    }
}

/// A `Copy` handle to a [`Symbol`] within an [`ObjectModule`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct SymbolId(u32);

impl SymbolId {
    /// The dense index this handle addresses.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Reconstruct a handle from its dense index (for deserialization).
    #[inline]
    pub fn from_index(i: usize) -> SymbolId {
        SymbolId(i as u32)
    }
}

/// What kind of storage a [`Section`] describes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SectionKind {
    /// Executable machine code (`.text`).
    Text,
    /// Writable initialized data (`.data`).
    Data,
    /// Read-only initialized data (`.rodata`).
    Rodata,
    /// Zero-initialized data that occupies no file space (`.bss`).
    Bss,
    /// Non-allocated metadata that is present in the file but not loaded into
    /// memory at run time (e.g. the `.debug_*` DWARF sections). Occupies file
    /// space like a content section, but is never part of a `PT_LOAD` segment.
    Debug,
}

/// A named region of an object: code or data bytes (or, for [`SectionKind::Bss`],
/// a reserved zero-initialized size) with an alignment requirement.
///
/// For every kind except [`SectionKind::Bss`] the content lives in `bytes` and
/// the section's in-memory size is `bytes.len()`. For `.bss`, `bytes` is empty
/// and the reserved size is `bss_size`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Section {
    /// The section name (e.g. `.text`).
    pub name: String,
    /// The storage kind.
    pub kind: SectionKind,
    /// The required alignment in bytes (a power of two; `1` means unaligned).
    pub align: u64,
    /// The section content. Empty for [`SectionKind::Bss`].
    pub bytes: Vec<u8>,
    /// The reserved size for [`SectionKind::Bss`]; ignored for other kinds.
    pub bss_size: u64,
}

impl Section {
    /// Create an empty content section (`.text`/`.data`/`.rodata`) of the given
    /// kind and alignment.
    pub fn new(name: impl Into<String>, kind: SectionKind, align: u64) -> Section {
        Section { name: name.into(), kind, align, bytes: Vec::new(), bss_size: 0 }
    }

    /// Create a `.bss`-style section reserving `size` zero bytes.
    pub fn bss(name: impl Into<String>, align: u64, size: u64) -> Section {
        Section { name: name.into(), kind: SectionKind::Bss, align, bytes: Vec::new(), bss_size: size }
    }

    /// The in-memory size of the section in bytes.
    #[inline]
    pub fn size(&self) -> u64 {
        match self.kind {
            SectionKind::Bss => self.bss_size,
            _ => self.bytes.len() as u64,
        }
    }

    /// Whether this section occupies no space in a file image (`.bss`).
    #[inline]
    pub fn is_nobits(&self) -> bool {
        matches!(self.kind, SectionKind::Bss)
    }
}

/// A symbol's linkage: how the linker treats multiple definitions and
/// visibility across objects.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SymbolBinding {
    /// Not visible outside the object; may duplicate names in other objects.
    Local,
    /// Visible to other objects; a duplicate strong definition is an error.
    Global,
    /// Like [`SymbolBinding::Global`] but yields to a strong definition.
    Weak,
}

/// A symbol's visibility outside the linked component (the ELF `st_other`
/// field): who can see it and whether it can be preempted at run time. See
/// [`crate::ir::Visibility`] for the IR-level meaning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum SymbolVisibility {
    /// `STV_DEFAULT`: exported and preemptible.
    #[default]
    Default,
    /// `STV_PROTECTED`: exported, not preemptible.
    Protected,
    /// `STV_HIDDEN`: not exported from the linked component.
    Hidden,
}

impl From<crate::ir::Visibility> for SymbolVisibility {
    fn from(v: crate::ir::Visibility) -> SymbolVisibility {
        match v {
            crate::ir::Visibility::Default => SymbolVisibility::Default,
            crate::ir::Visibility::Protected => SymbolVisibility::Protected,
            crate::ir::Visibility::Hidden => SymbolVisibility::Hidden,
        }
    }
}

/// What a symbol denotes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SymbolType {
    /// Unspecified.
    NoType,
    /// A data object (variable).
    Object,
    /// A function / other executable code.
    Func,
    /// A section (used as an anchor for section-relative relocations).
    Section,
}

/// Where a [`Symbol`] lives.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SymbolValue {
    /// Defined at `offset` bytes into `section`.
    Defined {
        /// The section the symbol is defined in.
        section: SectionId,
        /// The byte offset of the symbol within its section.
        offset: u64,
    },
    /// Undefined here — an external reference the linker must resolve.
    Undefined,
}

/// A named location: a function, a datum, or an external reference.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Symbol {
    /// The symbol name.
    pub name: String,
    /// The linkage binding.
    pub binding: SymbolBinding,
    /// What the symbol denotes.
    pub kind: SymbolType,
    /// Where the symbol is defined, or that it is undefined.
    pub value: SymbolValue,
    /// The size in bytes of the entity (0 if unknown / not applicable).
    pub size: u64,
    /// The visibility (`STV_*`); [`SymbolVisibility::Default`] unless set.
    pub visibility: SymbolVisibility,
}

impl Symbol {
    /// A symbol defined at `offset` inside `section`.
    pub fn defined(
        name: impl Into<String>,
        binding: SymbolBinding,
        kind: SymbolType,
        section: SectionId,
        offset: u64,
        size: u64,
    ) -> Symbol {
        Symbol {
            name: name.into(),
            binding,
            kind,
            value: SymbolValue::Defined { section, offset },
            size,
            visibility: SymbolVisibility::Default,
        }
    }

    /// An undefined external reference with the given binding.
    pub fn undefined(name: impl Into<String>, binding: SymbolBinding) -> Symbol {
        Symbol {
            name: name.into(),
            binding,
            kind: SymbolType::NoType,
            value: SymbolValue::Undefined,
            size: 0,
            visibility: SymbolVisibility::Default,
        }
    }

    /// This symbol with the given visibility.
    #[must_use]
    pub fn with_visibility(mut self, visibility: SymbolVisibility) -> Symbol {
        self.visibility = visibility;
        self
    }

    /// Whether this symbol is undefined (an external reference).
    #[inline]
    pub fn is_undefined(&self) -> bool {
        matches!(self.value, SymbolValue::Undefined)
    }
}

/// The relocation kinds the framework understands.
///
/// These are *generic* — a target-independent description of how a field is
/// patched from a symbol's address. Each concrete object writer maps them onto
/// its format's numeric codes (see [`crate::mc::elf`] for the x86-64 mapping).
/// The set is deliberately extensible; it currently covers what x86-64,
/// AArch64 and Thumb-2 relocatable code need for calls, data references, and
/// PC-relative addressing, plus the absolute data kinds of every pointer width a
/// [`DataLayout`](crate::ir::DataLayout) allows a whole-byte relocation for
/// (`Abs64`, `Abs32`, `Abs16`; see [`RelocKind::abs_for_width`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RelocKind {
    /// Absolute 64-bit: field = S + A.
    Abs64,
    /// Absolute 32-bit, zero-extended: field = S + A.
    Abs32,
    /// Absolute 16-bit: field = S + A, which must fit 16 bits (signed or
    /// unsigned). The pointer relocation of 16-bit address spaces (AVR).
    Abs16,
    /// Absolute 32-bit, sign-extended: field = S + A.
    Abs32S,
    /// PC-relative 32-bit: field = S + A - P.
    Pc32,
    /// PC-relative 64-bit: field = S + A - P.
    Pc64,
    /// PC-relative 32-bit via the procedure linkage table (call/jump target):
    /// field = L + A - P.
    Plt32,
    /// PC-relative 32-bit reference to the symbol's global-offset-table entry.
    GotPcRel,
    /// AArch64 `R_AARCH64_CALL26`: a `bl`/`b` `imm26` branch field to `S + A`,
    /// scaled by 4. Patched into bits `[25:0]` of the 32-bit instruction word.
    Aarch64Call26,
    /// AArch64 `R_AARCH64_ADR_PREL_PG_HI21`: the page-relative high 21 bits of
    /// `S + A` for an `adrp`, split across the `immhi`/`immlo` fields.
    Aarch64AdrPrelPgHi21,
    /// AArch64 `R_AARCH64_ADD_ABS_LO12_NC`: the low 12 bits of `S + A` for the
    /// `add` that completes an `adrp`+`add` address materialization.
    Aarch64AddAbsLo12Nc,
    /// Arm `R_ARM_THM_CALL`: a Thumb-2 `bl` to `S + A`, field = `S + A - P`
    /// (halfword-scaled, ±16 MiB) split across the two halfwords of the
    /// instruction (see [`write_thumb_field`]). A `bl` at `P` lands at
    /// `P + 4 + imm`, so a call to `S` carries the addend `-4`.
    ThumbCall,
    /// Arm `R_ARM_THM_MOVW_ABS_NC`: the low 16 bits of `S + A` (with the Thumb
    /// bit of a Thumb function symbol) into a `movw`'s `imm16`.
    ThumbMovwAbsNc,
    /// Arm `R_ARM_THM_MOVT_ABS`: the high 16 bits of `S + A` into a `movt`'s
    /// `imm16`.
    ThumbMovtAbs,
    /// AVR `R_AVR_CALL`: the 22-bit word address `(S + A) / 2` in a 2-word
    /// `call`/`jmp` (the field is the whole 4-byte instruction).
    AvrCall,
    /// AVR `R_AVR_13_PCREL`: the 12-bit word displacement
    /// `(S + A - (P + 2)) / 2` of an `rcall`/`rjmp`.
    Avr13Pcrel,
    /// AVR `R_AVR_16_PM`: a program-memory word address `(S + A) / 2` in a
    /// 16-bit data field (a function pointer).
    Avr16Pm,
    /// AVR `R_AVR_LO8_LDI`: bits 0–7 of `S + A` in the `K` field of an
    /// `ldi`-form instruction (`ldi`, `cpi`, `subi`, ...).
    AvrLo8Ldi,
    /// AVR `R_AVR_HI8_LDI`: bits 8–15 of `S + A` in an `ldi`-form `K` field.
    AvrHi8Ldi,
    /// AVR `R_AVR_LO8_LDI_PM`: bits 0–7 of the word address `(S + A) / 2`.
    AvrLo8LdiPm,
    /// AVR `R_AVR_HI8_LDI_PM`: bits 8–15 of the word address `(S + A) / 2`.
    AvrHi8LdiPm,
}

impl RelocKind {
    /// The width in bytes of the field this relocation patches.
    #[inline]
    pub fn field_width(self) -> usize {
        match self {
            RelocKind::Abs64 | RelocKind::Pc64 => 8,
            RelocKind::Abs16
            | RelocKind::Avr13Pcrel
            | RelocKind::Avr16Pm
            | RelocKind::AvrLo8Ldi
            | RelocKind::AvrHi8Ldi
            | RelocKind::AvrLo8LdiPm
            | RelocKind::AvrHi8LdiPm => 2,
            RelocKind::AvrCall => 4,
            RelocKind::Abs32
            | RelocKind::Abs32S
            | RelocKind::Pc32
            | RelocKind::Plt32
            | RelocKind::GotPcRel
            // The AArch64 and Thumb kinds patch a bitfield inside a 4-byte
            // instruction (one word, or two halfwords).
            | RelocKind::Aarch64Call26
            | RelocKind::Aarch64AdrPrelPgHi21
            | RelocKind::Aarch64AddAbsLo12Nc
            | RelocKind::ThumbCall
            | RelocKind::ThumbMovwAbsNc
            | RelocKind::ThumbMovtAbs => 4,
        }
    }

    /// The generic absolute data relocation for a pointer field of `bytes`
    /// bytes: [`Abs64`](RelocKind::Abs64) for 8, [`Abs32`](RelocKind::Abs32)
    /// for 4, [`Abs16`](RelocKind::Abs16) for 2, `None` for any other width.
    pub fn abs_for_width(bytes: u64) -> Option<RelocKind> {
        match bytes {
            8 => Some(RelocKind::Abs64),
            4 => Some(RelocKind::Abs32),
            2 => Some(RelocKind::Abs16),
            _ => None,
        }
    }

    /// Whether the relocation is computed relative to the address of the field
    /// (PC-relative) rather than absolutely.
    #[inline]
    pub fn is_pcrel(self) -> bool {
        matches!(
            self,
            RelocKind::Pc32
                | RelocKind::Pc64
                | RelocKind::Plt32
                | RelocKind::GotPcRel
                | RelocKind::Aarch64Call26
                | RelocKind::Aarch64AdrPrelPgHi21
                | RelocKind::ThumbCall
                | RelocKind::Avr13Pcrel
        )
    }

    /// Whether this is one of the AVR-specific kinds (which only the AVR ELF
    /// writer and firmware linker understand).
    #[inline]
    pub fn is_avr(self) -> bool {
        matches!(
            self,
            RelocKind::AvrCall
                | RelocKind::Avr13Pcrel
                | RelocKind::Avr16Pm
                | RelocKind::AvrLo8Ldi
                | RelocKind::AvrHi8Ldi
                | RelocKind::AvrLo8LdiPm
                | RelocKind::AvrHi8LdiPm
        )
    }

    /// Whether the relocation patches a bitfield inside an instruction rather
    /// than a whole data field (the AArch64, Thumb and AVR instruction kinds).
    #[inline]
    pub fn is_instruction_field(self) -> bool {
        matches!(
            self,
            RelocKind::Aarch64Call26
                | RelocKind::Aarch64AdrPrelPgHi21
                | RelocKind::Aarch64AddAbsLo12Nc
                | RelocKind::ThumbCall
                | RelocKind::ThumbMovwAbsNc
                | RelocKind::ThumbMovtAbs
                | RelocKind::AvrCall
                | RelocKind::Avr13Pcrel
                | RelocKind::AvrLo8Ldi
                | RelocKind::AvrHi8Ldi
                | RelocKind::AvrLo8LdiPm
                | RelocKind::AvrHi8LdiPm
        )
    }

    /// Whether this is one of the Thumb-2 instruction kinds, whose field is
    /// read and written by [`write_thumb_field`].
    #[inline]
    pub fn is_thumb(self) -> bool {
        matches!(self, RelocKind::ThumbCall | RelocKind::ThumbMovwAbsNc | RelocKind::ThumbMovtAbs)
    }
}

/// Store `value` into the `width`-byte field at `buf[at..]` in `endian` byte
/// order, as a width-generic absolute (or PC-relative) relocation result.
/// Returns `false`, leaving `buf` untouched, if `value` fits the field neither
/// as a signed nor as an unsigned `width`-byte integer (an 8-byte field always
/// fits). The shared patching rule of the emitter, the JIT and the linker.
pub fn write_field(buf: &mut [u8], at: usize, width: usize, value: i64, endian: crate::ir::Endian) -> bool {
    debug_assert!((1..=8).contains(&width), "field width {width}");
    if width < 8 {
        let bits = 8 * width as u32;
        let (min, umax) = (-(1i64 << (bits - 1)), (1i64 << bits) - 1);
        if value < min || value > umax {
            return false;
        }
    }
    let le = value.to_le_bytes();
    let field = &mut buf[at..at + width];
    match endian {
        crate::ir::Endian::Little => field.copy_from_slice(&le[..width]),
        crate::ir::Endian::Big => {
            for (i, b) in field.iter_mut().enumerate() {
                *b = le[width - 1 - i];
            }
        }
    }
    true
}

/// Patch the Thumb-2 instruction at `buf[at..at + 4]` (two little-endian
/// halfwords) for a Thumb relocation `kind` (see [`RelocKind::is_thumb`]),
/// following *ELF for the Arm Architecture*:
///
/// - [`ThumbCall`](RelocKind::ThumbCall): `value` is the byte displacement
///   `S + A - P`; it must be even and within ±16 MiB, and goes into the `bl`'s
///   `S:I1:I2:imm10:imm11` fields (`J1 = !I1 ^ S`, `J2 = !I2 ^ S`).
/// - [`ThumbMovwAbsNc`](RelocKind::ThumbMovwAbsNc) /
///   [`ThumbMovtAbs`](RelocKind::ThumbMovtAbs): `value` is the 16-bit
///   immediate itself (the low or high half of `S + A`, which the caller
///   selects), stored in the `imm4:i:imm3:imm8` fields of `movw`/`movt`.
///
/// The same routine stores a `REL`-format *implicit addend*: the addend of a
/// call is its displacement field, and that of `movw`/`movt` the signed 16-bit
/// immediate. Returns `false`, leaving `buf` untouched, for a value that does
/// not fit or a non-Thumb kind.
pub fn write_thumb_field(buf: &mut [u8], at: usize, kind: RelocKind, value: i64) -> bool {
    if at + 4 > buf.len() {
        return false;
    }
    let hw = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let (mut h1, mut h2) = (hw(buf, at), hw(buf, at + 2));
    match kind {
        RelocKind::ThumbCall => {
            if value & 1 != 0 || !(-(1i64 << 24)..(1i64 << 24)).contains(&value) {
                return false;
            }
            let v = value as u32;
            let s = (v >> 24) & 1;
            let i1 = (v >> 23) & 1;
            let i2 = (v >> 22) & 1;
            let imm10 = (v >> 12) & 0x3ff;
            let imm11 = (v >> 1) & 0x7ff;
            let j1 = (i1 ^ 1) ^ s;
            let j2 = (i2 ^ 1) ^ s;
            h1 = (h1 & 0xf800) | (s << 10) as u16 | imm10 as u16;
            h2 = (h2 & 0xd000) | (j1 << 13) as u16 | (j2 << 11) as u16 | imm11 as u16;
        }
        RelocKind::ThumbMovwAbsNc | RelocKind::ThumbMovtAbs => {
            if !(-0x8000..=0xffff).contains(&value) {
                return false;
            }
            let v = (value as u32) & 0xffff;
            let (imm4, i, imm3, imm8) = (v >> 12, (v >> 11) & 1, (v >> 8) & 7, v & 0xff);
            h1 = (h1 & 0xfbf0) | (i << 10) as u16 | imm4 as u16;
            h2 = (h2 & 0x8f00) | (imm3 << 12) as u16 | imm8 as u16;
        }
        _ => return false,
    }
    buf[at..at + 2].copy_from_slice(&h1.to_le_bytes());
    buf[at + 2..at + 4].copy_from_slice(&h2.to_le_bytes());
    true
}

/// A patch to apply to a section's bytes once the target symbol's address is
/// known. Uses the explicit-addend (RELA) form.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Relocation {
    /// The section whose bytes are patched.
    pub section: SectionId,
    /// The byte offset within `section` of the field to patch.
    pub offset: u64,
    /// The symbol whose address drives the patch.
    pub symbol: SymbolId,
    /// How the field is computed.
    pub kind: RelocKind,
    /// The explicit addend `A`.
    pub addend: i64,
}

/// A whole relocatable object: its sections, symbols, and relocations.
///
/// Sections and symbols are stored in insertion order and addressed by their
/// `Copy` handles; the `by_name` map only accelerates symbol interning and does
/// not affect the serialized order.
#[derive(Clone, Debug)]
pub struct ObjectModule {
    /// A human-readable module name (informational).
    pub name: String,
    sections: Vec<Section>,
    symbols: Vec<Symbol>,
    relocations: Vec<Relocation>,
    by_name: DetHashMap<String, SymbolId>,
}

impl PartialEq for ObjectModule {
    fn eq(&self, other: &Self) -> bool {
        // `by_name` is a derived index of `symbols`; comparing the content
        // fields is sufficient and avoids depending on map internals.
        self.name == other.name
            && self.sections == other.sections
            && self.symbols == other.symbols
            && self.relocations == other.relocations
    }
}

impl Eq for ObjectModule {}

impl ObjectModule {
    /// Create an empty object module.
    pub fn new(name: impl Into<String>) -> ObjectModule {
        ObjectModule {
            name: name.into(),
            sections: Vec::new(),
            symbols: Vec::new(),
            relocations: Vec::new(),
            by_name: DetHashMap::default(),
        }
    }

    /// Append a section, returning its handle.
    pub fn add_section(&mut self, section: Section) -> SectionId {
        let id = SectionId::from_index(self.sections.len());
        self.sections.push(section);
        id
    }

    /// Borrow a section.
    #[inline]
    pub fn section(&self, id: SectionId) -> &Section {
        &self.sections[id.index()]
    }

    /// Mutably borrow a section (e.g. to append encoded bytes).
    #[inline]
    pub fn section_mut(&mut self, id: SectionId) -> &mut Section {
        &mut self.sections[id.index()]
    }

    /// All sections in insertion order.
    #[inline]
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// Add or update a symbol, interning by name.
    ///
    /// If a symbol with the same name already exists, its slot is *updated* to
    /// `symbol` (so a forward [`reference_symbol`](Self::reference_symbol) can
    /// later be turned into a definition) and its existing handle is returned.
    /// Otherwise the symbol is appended.
    pub fn add_symbol(&mut self, symbol: Symbol) -> SymbolId {
        if let Some(&id) = self.by_name.get(&symbol.name) {
            self.symbols[id.index()] = symbol;
            id
        } else {
            let id = SymbolId::from_index(self.symbols.len());
            self.by_name.insert(symbol.name.clone(), id);
            self.symbols.push(symbol);
            id
        }
    }

    /// Return the handle of the symbol named `name`, creating an undefined
    /// global reference if none exists yet. Never overwrites an existing symbol.
    pub fn reference_symbol(&mut self, name: &str) -> SymbolId {
        if let Some(&id) = self.by_name.get(name) {
            return id;
        }
        let id = SymbolId::from_index(self.symbols.len());
        self.by_name.insert(name.to_owned(), id);
        self.symbols.push(Symbol::undefined(name, SymbolBinding::Global));
        id
    }

    /// The handle of an existing symbol by name, if any.
    #[inline]
    pub fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.by_name.get(name).copied()
    }

    /// Borrow a symbol.
    #[inline]
    pub fn symbol(&self, id: SymbolId) -> &Symbol {
        &self.symbols[id.index()]
    }

    /// All symbols in insertion order.
    #[inline]
    pub fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    /// Record a relocation.
    pub fn add_relocation(&mut self, reloc: Relocation) {
        self.relocations.push(reloc);
    }

    /// All relocations in insertion order.
    #[inline]
    pub fn relocations(&self) -> &[Relocation] {
        &self.relocations
    }

    /// Convenience: add a section built by an [emitter](crate::mc::emit),
    /// translating each emitted external reference into a [`Relocation`] against
    /// an interned (undefined-if-new) symbol. Returns the new section handle.
    pub fn add_emitted_section(
        &mut self,
        name: impl Into<String>,
        kind: SectionKind,
        align: u64,
        emitted: crate::mc::emit::Emitted,
    ) -> SectionId {
        let mut section = Section::new(name, kind, align);
        section.bytes = emitted.bytes;
        let sid = self.add_section(section);
        for r in emitted.relocations {
            let sym = self.reference_symbol(&r.symbol);
            self.add_relocation(Relocation {
                section: sid,
                offset: r.offset,
                symbol: sym,
                kind: r.kind,
                addend: r.addend,
            });
        }
        sid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_size_and_nobits() {
        let mut t = Section::new(".text", SectionKind::Text, 16);
        t.bytes.extend_from_slice(&[1, 2, 3, 4]);
        assert_eq!(t.size(), 4);
        assert!(!t.is_nobits());

        let b = Section::bss(".bss", 8, 128);
        assert_eq!(b.size(), 128);
        assert!(b.is_nobits());
        assert!(b.bytes.is_empty());
    }

    #[test]
    fn symbol_interning_updates_in_place() {
        let mut m = ObjectModule::new("m");
        // Forward reference creates an undefined symbol.
        let a = m.reference_symbol("foo");
        assert!(m.symbol(a).is_undefined());
        // Referencing again returns the same handle.
        assert_eq!(m.reference_symbol("foo"), a);

        // Defining it later updates the same slot.
        let sec = m.add_section(Section::new(".text", SectionKind::Text, 1));
        let b = m.add_symbol(Symbol::defined(
            "foo",
            SymbolBinding::Global,
            SymbolType::Func,
            sec,
            0,
            0,
        ));
        assert_eq!(a, b);
        assert!(!m.symbol(a).is_undefined());
        assert_eq!(m.symbols().len(), 1);
    }

    #[test]
    fn reloc_kind_widths() {
        assert_eq!(RelocKind::Abs64.field_width(), 8);
        assert_eq!(RelocKind::Abs16.field_width(), 2);
        assert_eq!(RelocKind::abs_for_width(2), Some(RelocKind::Abs16));
        assert_eq!(RelocKind::abs_for_width(4), Some(RelocKind::Abs32));
        assert_eq!(RelocKind::abs_for_width(3), None);
        let mut buf = [0u8; 4];
        assert!(write_field(&mut buf, 1, 2, 0xbeef, crate::ir::Endian::Little));
        assert_eq!(buf, [0, 0xef, 0xbe, 0]);
        assert!(write_field(&mut buf, 0, 2, -2, crate::ir::Endian::Big));
        assert_eq!(buf[..2], [0xff, 0xfe]);
        assert!(!write_field(&mut buf, 0, 2, 0x1_0000, crate::ir::Endian::Little));
        assert!(!write_field(&mut buf, 0, 2, -0x8001, crate::ir::Endian::Little));
        assert_eq!(RelocKind::Pc32.field_width(), 4);
        assert!(RelocKind::Plt32.is_pcrel());
        assert!(!RelocKind::Abs64.is_pcrel());
    }
}
