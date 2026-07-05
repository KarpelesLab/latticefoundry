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
//!   carries a [`SymbolBinding`] (local/global/weak) and a [`SymbolType`].
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
        }
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
/// The set is deliberately extensible; it currently covers what x86-64 and
/// AArch64 relocatable code need for calls, data references, and PC-relative
/// addressing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RelocKind {
    /// Absolute 64-bit: field = S + A.
    Abs64,
    /// Absolute 32-bit, zero-extended: field = S + A.
    Abs32,
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
}

impl RelocKind {
    /// The width in bytes of the field this relocation patches.
    #[inline]
    pub fn field_width(self) -> usize {
        match self {
            RelocKind::Abs64 | RelocKind::Pc64 => 8,
            RelocKind::Abs32
            | RelocKind::Abs32S
            | RelocKind::Pc32
            | RelocKind::Plt32
            | RelocKind::GotPcRel
            // The AArch64 kinds patch a bitfield inside a 4-byte instruction word.
            | RelocKind::Aarch64Call26
            | RelocKind::Aarch64AdrPrelPgHi21
            | RelocKind::Aarch64AddAbsLo12Nc => 4,
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
        )
    }
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
        assert_eq!(RelocKind::Pc32.field_width(), 4);
        assert!(RelocKind::Plt32.is_pcrel());
        assert!(!RelocKind::Abs64.is_pcrel());
    }
}
