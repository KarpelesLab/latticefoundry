//! The `.lfo` object format: a compact, versioned, lossless serialization of an
//! [`ObjectModule`] (ROADMAP Phase 6).
//!
//! `.lfo` is LatticeFoundry's own relocatable-object container — the native
//! counterpart to the standard [ELF writer](crate::mc::elf). The byte layout
//! follows the same conventions as the `.lfb` IR format: a fixed [`MAGIC`] tag,
//! a [`VERSION`], then a stream of LEB128 unsigned varints (and zig-zag signed
//! varints for addends) with length-prefixed UTF-8 strings. An unknown magic or
//! version is refused with a [`DecodeError`] rather than misinterpreted, and
//! decoding is **panic-free**: every malformed, truncated, or out-of-range input
//! surfaces as an error.
//!
//! # Layout
//!
//! After magic + version:
//! 1. the module name (length-prefixed UTF-8);
//! 2. the **sections** — count, then each as name, kind byte, alignment,
//!    `bss_size`, and length-prefixed content bytes;
//! 3. the **symbols** — count, then each as name, binding byte, type byte,
//!    size, and a value (`0` undefined, or `1` + section index + offset);
//! 4. the **relocations** — count, then each as section index, offset, symbol
//!    index, kind byte, and a zig-zag signed addend.
//!
//! # Determinism / content-addressing
//!
//! Sections, symbols, and relocations are emitted in the module's insertion
//! order (they are `Vec`-backed), so the byte stream is a pure function of the
//! module's content — no hash-map iteration order leaks in (tenet T5). Two
//! encodes of the same module are byte-identical, and re-encoding a decoded
//! module reproduces the bytes.

use std::fmt;

use crate::mc::object::{
    ObjectModule, RelocKind, Relocation, Section, SectionId, SectionKind, Symbol, SymbolBinding,
    SymbolId, SymbolType, SymbolValue,
};

/// Four-byte file signature: "LFO" followed by a NUL, identifying a `.lfo`.
pub const MAGIC: [u8; 4] = *b"LFO\0";

/// Format version. Bumped on any incompatible change to the byte layout; a
/// decoder refuses a version it does not recognize.
pub const VERSION: u32 = 1;

/// Why decoding a `.lfo` byte stream failed. Decoding never panics: every
/// malformed, truncated, or unsupported input surfaces as one of these.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DecodeError {
    /// The leading four bytes were not [`MAGIC`] (not a `.lfo` stream).
    BadMagic,
    /// The stream declared a version this decoder does not support.
    UnsupportedVersion(u32),
    /// The stream ended in the middle of a value (truncated input).
    UnexpectedEof,
    /// A varint was longer than `u64` allows or overflowed.
    VarintOverflow,
    /// A discriminant/tag byte was not valid for its position.
    InvalidTag {
        /// The category being decoded (e.g. `"section-kind"`).
        what: &'static str,
        /// The unrecognized tag value.
        tag: u32,
    },
    /// A length-prefixed string was not valid UTF-8.
    InvalidUtf8,
    /// An index (into the section or symbol table) was out of range.
    IndexOutOfRange {
        /// The category the index addresses (e.g. `"section"`, `"symbol"`).
        what: &'static str,
        /// The out-of-range index.
        index: u64,
    },
    /// Decoding finished with unconsumed trailing bytes (corrupt stream).
    TrailingBytes,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::BadMagic => write!(f, "not a .lfo stream (bad magic)"),
            DecodeError::UnsupportedVersion(v) => {
                write!(f, "unsupported .lfo version {v} (this build reads {VERSION})")
            }
            DecodeError::UnexpectedEof => write!(f, "unexpected end of input (truncated .lfo)"),
            DecodeError::VarintOverflow => write!(f, "malformed varint (overflow)"),
            DecodeError::InvalidTag { what, tag } => write!(f, "invalid {what} tag {tag}"),
            DecodeError::InvalidUtf8 => write!(f, "string was not valid UTF-8"),
            DecodeError::IndexOutOfRange { what, index } => {
                write!(f, "{what} index {index} out of range")
            }
            DecodeError::TrailingBytes => write!(f, "trailing bytes after end of object"),
        }
    }
}

impl std::error::Error for DecodeError {}

// ===========================================================================
// Low-level Writer / Reader (LEB128 varints, no dependencies)
// ===========================================================================

/// A minimal append-only byte sink with LEB128 varint support.
#[derive(Debug, Default)]
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Writer { buf: Vec::new() }
    }

    #[inline]
    fn u8(&mut self, b: u8) {
        self.buf.push(b);
    }

    #[inline]
    fn raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Write an unsigned integer as LEB128.
    fn uvarint(&mut self, mut v: u64) {
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.u8(byte);
                break;
            }
            self.u8(byte | 0x80);
        }
    }

    /// Write a signed integer as a zig-zag LEB128.
    fn svarint(&mut self, v: i64) {
        // Zig-zag: map small-magnitude signed values to small unsigned ones.
        let zz = ((v << 1) ^ (v >> 63)) as u64;
        self.uvarint(zz);
    }

    /// Write a length-prefixed byte slice.
    fn bytes(&mut self, bytes: &[u8]) {
        self.uvarint(bytes.len() as u64);
        self.raw(bytes);
    }

    /// Write a length-prefixed UTF-8 string.
    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }

    fn finish(self) -> Vec<u8> {
        self.buf
    }
}

/// A minimal bounds-checked byte source with LEB128 varint support. Every read
/// returns [`DecodeError::UnexpectedEof`] rather than panicking at end of input.
#[derive(Debug)]
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn u8(&mut self) -> Result<u8, DecodeError> {
        let b = *self.buf.get(self.pos).ok_or(DecodeError::UnexpectedEof)?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::UnexpectedEof)?;
        let slice = self.buf.get(self.pos..end).ok_or(DecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }

    /// Read a LEB128 unsigned integer.
    fn uvarint(&mut self) -> Result<u64, DecodeError> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            if shift >= 64 {
                return Err(DecodeError::VarintOverflow);
            }
            let byte = self.u8()?;
            let low = u64::from(byte & 0x7f);
            if shift == 63 && low > 1 {
                return Err(DecodeError::VarintOverflow);
            }
            result |= low << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
    }

    /// Read a zig-zag LEB128 signed integer.
    fn svarint(&mut self) -> Result<i64, DecodeError> {
        let zz = self.uvarint()?;
        Ok(((zz >> 1) as i64) ^ -((zz & 1) as i64))
    }

    /// Read a `usize` index (rejecting values that do not fit).
    fn uindex(&mut self) -> Result<usize, DecodeError> {
        let v = self.uvarint()?;
        usize::try_from(v).map_err(|_| DecodeError::VarintOverflow)
    }

    /// Read a length-prefixed byte slice.
    fn bytes(&mut self) -> Result<&'a [u8], DecodeError> {
        let len = self.uindex()?;
        self.take(len)
    }

    /// Read a length-prefixed UTF-8 string.
    fn str(&mut self) -> Result<&'a str, DecodeError> {
        let bytes = self.bytes()?;
        std::str::from_utf8(bytes).map_err(|_| DecodeError::InvalidUtf8)
    }
}

/// Bounds-check `index` against `len`, tagging the failure with `what`.
fn checked(index: usize, len: usize, what: &'static str) -> Result<usize, DecodeError> {
    if index < len {
        Ok(index)
    } else {
        Err(DecodeError::IndexOutOfRange { what, index: index as u64 })
    }
}

// ===========================================================================
// Stable enum <-> byte code tables (hand-written so codes stay stable)
// ===========================================================================

fn section_kind_code(k: SectionKind) -> u8 {
    match k {
        SectionKind::Text => 0,
        SectionKind::Data => 1,
        SectionKind::Rodata => 2,
        SectionKind::Bss => 3,
    }
}

fn section_kind_from(c: u8) -> Result<SectionKind, DecodeError> {
    Ok(match c {
        0 => SectionKind::Text,
        1 => SectionKind::Data,
        2 => SectionKind::Rodata,
        3 => SectionKind::Bss,
        _ => return Err(DecodeError::InvalidTag { what: "section-kind", tag: u32::from(c) }),
    })
}

fn binding_code(b: SymbolBinding) -> u8 {
    match b {
        SymbolBinding::Local => 0,
        SymbolBinding::Global => 1,
        SymbolBinding::Weak => 2,
    }
}

fn binding_from(c: u8) -> Result<SymbolBinding, DecodeError> {
    Ok(match c {
        0 => SymbolBinding::Local,
        1 => SymbolBinding::Global,
        2 => SymbolBinding::Weak,
        _ => return Err(DecodeError::InvalidTag { what: "binding", tag: u32::from(c) }),
    })
}

fn symtype_code(t: SymbolType) -> u8 {
    match t {
        SymbolType::NoType => 0,
        SymbolType::Object => 1,
        SymbolType::Func => 2,
        SymbolType::Section => 3,
    }
}

fn symtype_from(c: u8) -> Result<SymbolType, DecodeError> {
    Ok(match c {
        0 => SymbolType::NoType,
        1 => SymbolType::Object,
        2 => SymbolType::Func,
        3 => SymbolType::Section,
        _ => return Err(DecodeError::InvalidTag { what: "symbol-type", tag: u32::from(c) }),
    })
}

fn reloc_kind_code(k: RelocKind) -> u8 {
    match k {
        RelocKind::Abs64 => 0,
        RelocKind::Abs32 => 1,
        RelocKind::Abs32S => 2,
        RelocKind::Pc32 => 3,
        RelocKind::Pc64 => 4,
        RelocKind::Plt32 => 5,
        RelocKind::GotPcRel => 6,
        RelocKind::Aarch64Call26 => 7,
        RelocKind::Aarch64AdrPrelPgHi21 => 8,
        RelocKind::Aarch64AddAbsLo12Nc => 9,
    }
}

fn reloc_kind_from(c: u8) -> Result<RelocKind, DecodeError> {
    Ok(match c {
        0 => RelocKind::Abs64,
        1 => RelocKind::Abs32,
        2 => RelocKind::Abs32S,
        3 => RelocKind::Pc32,
        4 => RelocKind::Pc64,
        5 => RelocKind::Plt32,
        6 => RelocKind::GotPcRel,
        7 => RelocKind::Aarch64Call26,
        8 => RelocKind::Aarch64AdrPrelPgHi21,
        9 => RelocKind::Aarch64AddAbsLo12Nc,
        _ => return Err(DecodeError::InvalidTag { what: "reloc-kind", tag: u32::from(c) }),
    })
}

// ===========================================================================
// Encoding
// ===========================================================================

/// Encode a whole [`ObjectModule`] to the compact, versioned `.lfo` byte form.
///
/// The output is deterministic: `encode(o) == encode(o)` for any `o`.
pub fn encode(obj: &ObjectModule) -> Vec<u8> {
    let mut w = Writer::new();
    w.raw(&MAGIC);
    w.uvarint(u64::from(VERSION));

    w.str(&obj.name);

    // --- sections ---
    w.uvarint(obj.sections().len() as u64);
    for s in obj.sections() {
        w.str(&s.name);
        w.u8(section_kind_code(s.kind));
        w.uvarint(s.align);
        w.uvarint(s.bss_size);
        w.bytes(&s.bytes);
    }

    // --- symbols ---
    w.uvarint(obj.symbols().len() as u64);
    for sym in obj.symbols() {
        w.str(&sym.name);
        w.u8(binding_code(sym.binding));
        w.u8(symtype_code(sym.kind));
        w.uvarint(sym.size);
        match sym.value {
            SymbolValue::Undefined => w.u8(0),
            SymbolValue::Defined { section, offset } => {
                w.u8(1);
                w.uvarint(section.index() as u64);
                w.uvarint(offset);
            }
        }
    }

    // --- relocations ---
    w.uvarint(obj.relocations().len() as u64);
    for r in obj.relocations() {
        w.uvarint(r.section.index() as u64);
        w.uvarint(r.offset);
        w.uvarint(r.symbol.index() as u64);
        w.u8(reloc_kind_code(r.kind));
        w.svarint(r.addend);
    }

    w.finish()
}

// ===========================================================================
// Decoding
// ===========================================================================

/// Decode an [`ObjectModule`] from the `.lfo` byte form produced by [`encode`].
///
/// Any malformed input — bad magic, unknown version, truncation, an
/// out-of-range index, invalid UTF-8 — yields an [`Err`] and never panics.
pub fn decode(bytes: &[u8]) -> Result<ObjectModule, DecodeError> {
    let mut r = Reader::new(bytes);

    let magic = r.take(MAGIC.len())?;
    if magic != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let version = r.uvarint()?;
    if version != u64::from(VERSION) {
        return Err(DecodeError::UnsupportedVersion(version.try_into().unwrap_or(u32::MAX)));
    }

    let mut obj = ObjectModule::new(r.str()?.to_owned());

    // --- sections ---
    let nsections = r.uindex()?;
    for _ in 0..nsections {
        let name = r.str()?.to_owned();
        let kind = section_kind_from(r.u8()?)?;
        let align = r.uvarint()?;
        let bss_size = r.uvarint()?;
        let content = r.bytes()?.to_vec();
        obj.add_section(Section { name, kind, align, bytes: content, bss_size });
    }
    let nsections = obj.sections().len();

    // --- symbols ---
    let nsymbols = r.uindex()?;
    for _ in 0..nsymbols {
        let name = r.str()?.to_owned();
        let binding = binding_from(r.u8()?)?;
        let kind = symtype_from(r.u8()?)?;
        let size = r.uvarint()?;
        let value = match r.u8()? {
            0 => SymbolValue::Undefined,
            1 => {
                let section = SectionId::from_index(checked(r.uindex()?, nsections, "section")?);
                let offset = r.uvarint()?;
                SymbolValue::Defined { section, offset }
            }
            t => return Err(DecodeError::InvalidTag { what: "symbol-value", tag: u32::from(t) }),
        };
        obj.add_symbol(Symbol { name, binding, kind, value, size });
    }
    let nsymbols = obj.symbols().len();

    // --- relocations ---
    let nrelocs = r.uindex()?;
    for _ in 0..nrelocs {
        let section = SectionId::from_index(checked(r.uindex()?, nsections, "section")?);
        let offset = r.uvarint()?;
        let symbol = SymbolId::from_index(checked(r.uindex()?, nsymbols, "symbol")?);
        let kind = reloc_kind_from(r.u8()?)?;
        let addend = r.svarint()?;
        obj.add_relocation(Relocation { section, offset, symbol, kind, addend });
    }

    if r.remaining() != 0 {
        return Err(DecodeError::TrailingBytes);
    }
    Ok(obj)
}

#[cfg(test)]
mod tests {
    use super::{DecodeError, MAGIC, VERSION, decode, encode};
    use crate::mc::object::{
        ObjectModule, RelocKind, Relocation, Section, SectionKind, Symbol, SymbolBinding,
        SymbolType,
    };

    /// Build an object exercising every construct the format must carry:
    /// multiple section kinds (text/data/rodata/bss), local + global + weak +
    /// undefined symbols, and several relocation kinds with positive, negative,
    /// and zero addends.
    fn build_sample() -> ObjectModule {
        let mut m = ObjectModule::new("sample.o");

        let mut text = Section::new(".text", SectionKind::Text, 16);
        text.bytes = vec![0x55, 0x48, 0x89, 0xe5, 0xe8, 0, 0, 0, 0, 0xc3];
        let text_id = m.add_section(text);

        let mut data = Section::new(".data", SectionKind::Data, 8);
        data.bytes = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let data_id = m.add_section(data);

        let mut rodata = Section::new(".rodata", SectionKind::Rodata, 1);
        rodata.bytes = b"hello\0".to_vec();
        m.add_section(rodata);

        let bss_id = m.add_section(Section::bss(".bss", 32, 4096));

        // A global function defined in .text.
        m.add_symbol(Symbol::defined(
            "main",
            SymbolBinding::Global,
            SymbolType::Func,
            text_id,
            0,
            10,
        ));
        // A local object in .data.
        m.add_symbol(Symbol::defined(
            "table",
            SymbolBinding::Local,
            SymbolType::Object,
            data_id,
            0,
            8,
        ));
        // A weak object in .bss.
        m.add_symbol(Symbol::defined(
            "cache",
            SymbolBinding::Weak,
            SymbolType::Object,
            bss_id,
            0,
            4096,
        ));
        // An undefined external.
        let printf = m.add_symbol(Symbol::undefined("printf", SymbolBinding::Global));

        // A call relocation into .text, and a few data references.
        m.add_relocation(Relocation {
            section: text_id,
            offset: 5,
            symbol: printf,
            kind: RelocKind::Plt32,
            addend: -4,
        });
        let table = m.symbol_id("table").unwrap();
        m.add_relocation(Relocation {
            section: data_id,
            offset: 0,
            symbol: table,
            kind: RelocKind::Abs64,
            addend: 0,
        });
        m.add_relocation(Relocation {
            section: text_id,
            offset: 6,
            symbol: table,
            kind: RelocKind::Pc32,
            addend: 0x1234,
        });
        m.add_relocation(Relocation {
            section: data_id,
            offset: 4,
            symbol: printf,
            kind: RelocKind::GotPcRel,
            addend: -0x7fff_ffff,
        });

        m
    }

    #[test]
    fn round_trips_losslessly() {
        let m = build_sample();
        let bytes = encode(&m);

        // Encoding is deterministic.
        assert_eq!(bytes, encode(&m), "encode must be deterministic");

        // decode(encode(m)) == m, structurally.
        let m2 = decode(&bytes).expect("decode should succeed");
        assert_eq!(m, m2, "decode(encode(m)) == m");

        // And re-encoding reproduces the exact bytes.
        assert_eq!(bytes, encode(&m2), "encode(decode(encode(m))) == encode(m)");

        // Spot-check a few round-tripped fields.
        assert_eq!(m2.name, "sample.o");
        assert_eq!(m2.sections().len(), 4);
        assert_eq!(m2.symbols().len(), 4);
        assert_eq!(m2.relocations().len(), 4);
        assert_eq!(m2.sections()[3].size(), 4096, ".bss reserved size survives");
        assert!(m2.symbol_id("printf").is_some());
    }

    #[test]
    fn empty_object_round_trips() {
        let m = ObjectModule::new("empty");
        let bytes = encode(&m);
        let m2 = decode(&bytes).expect("empty object decodes");
        assert_eq!(m, m2);
        assert_eq!(encode(&m2), bytes);
    }

    #[test]
    fn negative_and_wide_addends_survive() {
        let mut m = ObjectModule::new("addends");
        let s = m.add_section(Section::new(".text", SectionKind::Text, 1));
        let sym = m.add_symbol(Symbol::defined(
            "s",
            SymbolBinding::Local,
            SymbolType::NoType,
            s,
            0,
            0,
        ));
        for addend in [0i64, -1, 1, i64::MIN, i64::MAX, -0x1_0000_0000, 0x7fff_ffff] {
            m.add_relocation(Relocation {
                section: s,
                offset: 0,
                symbol: sym,
                kind: RelocKind::Abs64,
                addend,
            });
        }
        let bytes = encode(&m);
        let m2 = decode(&bytes).expect("decode");
        assert_eq!(m, m2);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let bytes = vec![b'X', b'X', b'X', b'X', 1];
        assert!(matches!(decode(&bytes), Err(DecodeError::BadMagic)));
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let mut bytes = MAGIC.to_vec();
        bytes.push(VERSION as u8 + 1);
        match decode(&bytes) {
            Err(DecodeError::UnsupportedVersion(v)) => assert_eq!(v, VERSION + 1),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn truncated_input_fails_gracefully() {
        let m = build_sample();
        let bytes = encode(&m);
        for k in 0..bytes.len() {
            let result = decode(&bytes[..k]);
            assert!(result.is_err(), "truncation at {k} should fail, got {result:?}");
        }
        assert!(decode(&bytes).is_ok());
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let m = ObjectModule::new("m");
        let mut bytes = encode(&m);
        bytes.push(0xff);
        assert!(matches!(decode(&bytes), Err(DecodeError::TrailingBytes)));
    }

    #[test]
    fn out_of_range_reference_is_rejected() {
        // Hand-build a stream with one section, no symbols, and a relocation
        // naming symbol index 5 — decoding must error, not panic.
        let mut b = MAGIC.to_vec();
        b.push(VERSION as u8); // version varint
        b.push(0); // module name: empty
        b.push(1); // section count: 1
        b.push(0); // section name: empty
        b.push(0); // kind: Text
        b.push(1); // align: 1
        b.push(0); // bss_size: 0
        b.push(0); // content bytes: empty
        b.push(0); // symbol count: 0
        b.push(1); // relocation count: 1
        b.push(0); // reloc section index: 0 (valid)
        b.push(0); // reloc offset: 0
        b.push(5); // reloc symbol index: 5 (out of range)
        b.push(0); // reloc kind: Abs64
        b.push(0); // addend zig-zag: 0
        match decode(&b) {
            Err(DecodeError::IndexOutOfRange { what: "symbol", index: 5 }) => {}
            other => panic!("expected symbol IndexOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn garbage_after_header_does_not_panic() {
        let mut junk = MAGIC.to_vec();
        junk.push(VERSION as u8);
        junk.extend_from_slice(&[0xff; 40]);
        let _ = decode(&junk); // must not panic
    }
}
