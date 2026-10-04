//! The `.lfb` binary form of the LatticeFoundry IR (`docs/ir-design.md` §8/§9).
//!
//! A compact, **versioned**, **content-addressed friendly** encoder/decoder for
//! a whole [`Module`] that round-trips losslessly with the in-memory form. The
//! file is self-describing: it opens with a [`MAGIC`] tag and a [`VERSION`]
//! number, and an unknown magic or version is refused with a clear
//! [`DecodeError`] rather than misinterpreted.
//!
//! # Layout
//!
//! Everything after the fixed 4-byte magic is a stream of **LEB128** unsigned
//! varints (implemented here, no dependencies) plus a handful of fixed
//! little-endian scalars (float bit patterns). The body is, in order:
//!
//! 1. the module name (a length-prefixed UTF-8 string), then (from version 3)
//!    the target name (a presence byte and a string) and the data-layout spec
//!    string (empty for the default LP64 layout);
//! 2. the **type table** — every type *reachable* from the module, emitted in a
//!    topological order (a composite type after its components) so each entry
//!    references only earlier ones;
//! 3. the **constant pool** — every reachable constant, likewise topologically
//!    ordered for aggregates;
//! 4. the **globals**, in module order;
//! 5. the **functions**, in module order — each as its flat value table,
//!    instruction arena, block list, and entry block.
//!
//! Types and constants are emitted as a self-contained table because a
//! [`Module`]'s interning pools cannot be enumerated from outside their module;
//! the encoder instead walks the module, collects the reachable handles, and
//! renumbers them densely. Globals and functions keep their module indices, so
//! `Global`/`Func` value references need no renumbering.
//!
//! # Determinism / content-addressing
//!
//! The encoder walks `Vec`-backed arenas in index order and emits the type and
//! constant tables sorted by their (topologically valid) interning index, so the
//! byte stream is a pure function of the module's content — no hash-map
//! iteration order leaks in (tenet T5). Two encodes of the same module are
//! byte-identical, and re-encoding a decoded module reproduces the bytes.
//!
//! Names ([`Function`]/[`Global`] `Sym`s) are stored as their **strings**, not
//! as raw interner handles: a handle's numeric value depends on interner
//! insertion order, which would make the same logical module encode to different
//! bytes and defeat content-addressing. Because a [`Module`] does not own the
//! [`StrInterner`] that backs its `Sym` names, both [`encode`] and [`decode`]
//! take it explicitly (there is no way to resolve or mint a `Sym` without it).

use std::collections::HashMap;
use std::fmt;

use crate::ir::inst::{
    AtomicOrdering, BinOp, CastOp, FastMath, Flags, FloatPred, InstData, InstId, InstKind, IntPred,
    ReduceOp, RmwOp, SwitchCase, SwitchData, UnaryOp, Use,
};
use crate::ir::types::{FloatKind, FuncType, Type, TypeId};
use crate::ir::value::{AddrTarget, Const, ConstId, FloatBits, Value, ValueDef, ValueId};
use crate::ir::{
    Block, BlockId, FuncAttrs, FuncId, Function, Global, GlobalAttrs, GlobalId, Linkage, Module,
    Visibility,
};
use crate::support::hash::{DetHashMap, DetHashSet};
use crate::support::StrInterner;

use puremp::{Int, Nat, Sign};

/// Four-byte file signature: "LFB" followed by a NUL, identifying an `.lfb`.
pub const MAGIC: [u8; 4] = *b"LFB\0";

/// Format version. Bumped on any incompatible change to the byte layout; a
/// decoder refuses a version it does not recognize.
///
/// - **1** — the original layout.
/// - **2** — adds a per-global attribute byte (linkage / constant / detached,
///   see [`GlobalAttrs`]) after each global's initializer, and the
///   address-constant tag ([`Const::Addr`]). Version-1 streams still decode:
///   their globals get [`GlobalAttrs::DEFAULT`] (external, mutable, emitted —
///   the meaning of a plain `.lf` `global` definition).
/// - **3** — adds symbol [`Visibility`] (bits 4–5 of the global attribute byte)
///   and a per-function attribute byte ([`FuncAttrs`]: linkage in bits 0–1,
///   visibility in bits 2–3) after each function's signature. Older streams
///   decode with default visibility and [`FuncAttrs::DEFAULT`].
/// - **4** — adds the module's target name and data layout after the module
///   name, and a per-global address space: bit 6 of the global attribute byte
///   says a varint address space follows the byte. The
///   `ptr addrspace(N)` type (tag 7) needs no bump, as older streams never
///   contain it. Version-1/2/3 streams still decode, with no target, the LP64
///   layout and every global in space 0.
/// - **5** — secrecy (`docs/ir-design.md` §6d). Bit 7 of both the global and
///   the function attribute byte now says an **extension varint** of further
///   flags follows (after the address space, for a global), so attribute
///   bytes never run out of bits again. Global extension bit 0 is `secret`;
///   function extension bit 0 is a secret return and bit 1 says a secret
///   parameter list follows (a count, then ascending parameter indices).
///   Unknown extension bits are rejected. Secret loads/stores and
///   `declassify` use new opcode tags 26/27/28. A module without secrets
///   encodes exactly as in version 4 apart from the version number;
///   version-1..4 streams still decode, with nothing secret.
///
///
/// Thread-local storage (`docs/ir-design.md` §4c) is global extension bit 1
/// (`thread_local`), carried by the version-5 extension varint: no bump, and a
/// module without thread-locals encodes exactly as before. A reader that
/// predates the bit rejects it as an unknown extension flag.
///
/// SIMD vectors (`docs/ir-design.md` §6e) only add tag *values* — type tag 16,
/// opcode tags 40–44 — so no stream without vectors changes and there was no
/// bump: a reader that predates them rejects such a stream with
/// [`DecodeError::InvalidTag`] rather than misreading it.
///
/// Inline assembly (`docs/ir-design.md` §6j) is the same: `inline_asm` is
/// opcode tag 45 (the template, a flag byte with bit 0 = `volatile`, then
/// counted outputs — constraint, optional name, optional type — inputs and
/// clobbers) and `asm_output` tag 46, with no bump.
pub const VERSION: u32 = 5;

/// Bit 7 of a (version ≥ 5) global or function attribute byte: an extension
/// varint of further attribute flags follows.
const ATTR_EXT_BIT: u8 = 0x80;

/// Global extension flag: the global is `secret`.
const GLOBAL_EXT_SECRET: u64 = 1;

/// Global extension flag: the global is `thread_local`.
const GLOBAL_EXT_THREAD_LOCAL: u64 = 2;

/// Function extension flag: the return value is `secret`.
const FUNC_EXT_SECRET_RET: u64 = 1;

/// Function extension flag: a secret-parameter list follows.
const FUNC_EXT_SECRET_PARAMS: u64 = 2;

/// The oldest format version [`decode`] still reads.
pub const MIN_VERSION: u32 = 1;

// ===========================================================================
// Errors
// ===========================================================================

/// Why decoding an `.lfb` byte stream failed. Decoding never panics: every
/// malformed, truncated, or unsupported input surfaces as one of these.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DecodeError {
    /// The leading four bytes were not [`MAGIC`] (not an `.lfb` stream).
    BadMagic,
    /// The stream declared a version this decoder does not support.
    UnsupportedVersion(u32),
    /// The stream ended in the middle of a value (truncated input).
    UnexpectedEof,
    /// A varint was longer than `u64` allows or overflowed.
    VarintOverflow,
    /// A discriminant/tag byte was not valid for its position. `what` names the
    /// category and `tag` is the offending value.
    InvalidTag {
        /// The category being decoded (e.g. `"type"`, `"opcode"`).
        what: &'static str,
        /// The unrecognized tag value.
        tag: u32,
    },
    /// A length-prefixed string was not valid UTF-8.
    InvalidUtf8,
    /// An index (into the type table, a value arena, ...) was out of range.
    IndexOutOfRange {
        /// The category the index addresses (e.g. `"type"`, `"value"`).
        what: &'static str,
        /// The out-of-range index.
        index: u64,
    },
    /// Decoding finished with unconsumed trailing bytes (corrupt stream).
    TrailingBytes,
    /// The module header's data-layout spec string was not valid (see
    /// [`DataLayout::parse`](crate::ir::DataLayout::parse)).
    InvalidDataLayout,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::BadMagic => write!(f, "not an .lfb stream (bad magic)"),
            DecodeError::UnsupportedVersion(v) => {
                write!(f, "unsupported .lfb version {v} (this build reads {VERSION})")
            }
            DecodeError::UnexpectedEof => write!(f, "unexpected end of input (truncated .lfb)"),
            DecodeError::VarintOverflow => write!(f, "malformed varint (overflow)"),
            DecodeError::InvalidTag { what, tag } => write!(f, "invalid {what} tag {tag}"),
            DecodeError::InvalidUtf8 => write!(f, "string was not valid UTF-8"),
            DecodeError::IndexOutOfRange { what, index } => {
                write!(f, "{what} index {index} out of range")
            }
            DecodeError::TrailingBytes => write!(f, "trailing bytes after end of module"),
            DecodeError::InvalidDataLayout => write!(f, "invalid data layout in module header"),
        }
    }
}

impl std::error::Error for DecodeError {}

// ===========================================================================
// Low-level Writer / Reader (LEB128 varints, no dependencies)
// ===========================================================================

/// A minimal append-only byte sink with LEB128 varint support.
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
            // Guard the final group so no set bit is shifted out of the u64.
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

    /// Read a `usize` index (rejecting values that do not fit).
    fn uindex(&mut self) -> Result<usize, DecodeError> {
        let v = self.uvarint()?;
        usize::try_from(v).map_err(|_| DecodeError::VarintOverflow)
    }

    /// Read a `u32` (rejecting values that do not fit).
    fn u32(&mut self) -> Result<u32, DecodeError> {
        u32::try_from(self.uvarint()?).map_err(|_| DecodeError::VarintOverflow)
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
// Small stable enum <-> byte code tables
// ===========================================================================
//
// These are written by hand rather than relying on `as u8` discriminants so the
// on-disk codes stay stable even if the in-memory enums are reordered.

fn ordering_code(o: AtomicOrdering) -> u8 {
    o.code()
}

fn ordering_from(c: u8) -> Result<AtomicOrdering, DecodeError> {
    AtomicOrdering::from_code(u64::from(c))
        .ok_or(DecodeError::InvalidTag { what: "atomic ordering", tag: u32::from(c) })
}

fn rmw_code(op: RmwOp) -> u8 {
    op.code()
}

fn rmw_from(c: u8) -> Result<RmwOp, DecodeError> {
    RmwOp::from_code(u64::from(c)).ok_or(DecodeError::InvalidTag { what: "atomic rmw op", tag: u32::from(c) })
}

fn binop_code(op: BinOp) -> u8 {
    use BinOp::*;
    match op {
        Add => 0, Sub => 1, Mul => 2, UDiv => 3, SDiv => 4, URem => 5, SRem => 6, And => 7,
        Or => 8, Xor => 9, Shl => 10, LShr => 11, AShr => 12, FAdd => 13, FSub => 14, FMul => 15,
        FDiv => 16, FRem => 17, SMin => 18, SMax => 19, UMin => 20, UMax => 21, SAddSat => 22,
        UAddSat => 23, SSubSat => 24, USubSat => 25,
    }
}

fn binop_from(c: u8) -> Result<BinOp, DecodeError> {
    use BinOp::*;
    Ok(match c {
        0 => Add, 1 => Sub, 2 => Mul, 3 => UDiv, 4 => SDiv, 5 => URem, 6 => SRem, 7 => And,
        8 => Or, 9 => Xor, 10 => Shl, 11 => LShr, 12 => AShr, 13 => FAdd, 14 => FSub, 15 => FMul,
        16 => FDiv, 17 => FRem, 18 => SMin, 19 => SMax, 20 => UMin, 21 => UMax, 22 => SAddSat,
        23 => UAddSat, 24 => SSubSat, 25 => USubSat,
        _ => return Err(DecodeError::InvalidTag { what: "binop", tag: u32::from(c) }),
    })
}

fn unop_code(op: UnaryOp) -> u8 {
    match op {
        UnaryOp::FNeg => 0,
    }
}

fn unop_from(c: u8) -> Result<UnaryOp, DecodeError> {
    match c {
        0 => Ok(UnaryOp::FNeg),
        _ => Err(DecodeError::InvalidTag { what: "unaryop", tag: u32::from(c) }),
    }
}

fn intpred_code(p: IntPred) -> u8 {
    use IntPred::*;
    match p {
        Eq => 0, Ne => 1, Ugt => 2, Uge => 3, Ult => 4, Ule => 5, Sgt => 6, Sge => 7, Slt => 8,
        Sle => 9,
    }
}

fn intpred_from(c: u8) -> Result<IntPred, DecodeError> {
    use IntPred::*;
    Ok(match c {
        0 => Eq, 1 => Ne, 2 => Ugt, 3 => Uge, 4 => Ult, 5 => Ule, 6 => Sgt, 7 => Sge, 8 => Slt,
        9 => Sle,
        _ => return Err(DecodeError::InvalidTag { what: "intpred", tag: u32::from(c) }),
    })
}

fn floatpred_code(p: FloatPred) -> u8 {
    use FloatPred::*;
    match p {
        False => 0, Oeq => 1, Ogt => 2, Oge => 3, Olt => 4, Ole => 5, One => 6, Ord => 7, Ueq => 8,
        Ugt => 9, Uge => 10, Ult => 11, Ule => 12, Une => 13, Uno => 14, True => 15,
    }
}

fn floatpred_from(c: u8) -> Result<FloatPred, DecodeError> {
    use FloatPred::*;
    Ok(match c {
        0 => False, 1 => Oeq, 2 => Ogt, 3 => Oge, 4 => Olt, 5 => Ole, 6 => One, 7 => Ord, 8 => Ueq,
        9 => Ugt, 10 => Uge, 11 => Ult, 12 => Ule, 13 => Une, 14 => Uno, 15 => True,
        _ => return Err(DecodeError::InvalidTag { what: "floatpred", tag: u32::from(c) }),
    })
}

fn cast_code(op: CastOp) -> u8 {
    use CastOp::*;
    match op {
        Trunc => 0, ZExt => 1, SExt => 2, FpTrunc => 3, FpExt => 4, FpToUi => 5, FpToSi => 6,
        UiToFp => 7, SiToFp => 8, PtrToInt => 9, IntToPtr => 10, Bitcast => 11,
    }
}

fn cast_from(c: u8) -> Result<CastOp, DecodeError> {
    use CastOp::*;
    Ok(match c {
        0 => Trunc, 1 => ZExt, 2 => SExt, 3 => FpTrunc, 4 => FpExt, 5 => FpToUi, 6 => FpToSi,
        7 => UiToFp, 8 => SiToFp, 9 => PtrToInt, 10 => IntToPtr, 11 => Bitcast,
        _ => return Err(DecodeError::InvalidTag { what: "cast", tag: u32::from(c) }),
    })
}

fn floatkind_code(k: FloatKind) -> u8 {
    match k {
        FloatKind::F16 => 0,
        FloatKind::F32 => 1,
        FloatKind::F64 => 2,
    }
}

fn floatkind_from(c: u8) -> Result<FloatKind, DecodeError> {
    Ok(match c {
        0 => FloatKind::F16,
        1 => FloatKind::F32,
        2 => FloatKind::F64,
        _ => return Err(DecodeError::InvalidTag { what: "floatkind", tag: u32::from(c) }),
    })
}

fn sign_code(s: Sign) -> u8 {
    match s {
        Sign::Zero => 0,
        Sign::Positive => 1,
        Sign::Negative => 2,
    }
}

fn sign_from(c: u8) -> Result<Sign, DecodeError> {
    Ok(match c {
        0 => Sign::Zero,
        1 => Sign::Positive,
        2 => Sign::Negative,
        _ => return Err(DecodeError::InvalidTag { what: "sign", tag: u32::from(c) }),
    })
}

/// Pack the nine flag bits into a single value: `nsw|nuw|exact` then the six
/// fast-math bits.
fn flags_bits(flags: Flags) -> u64 {
    let f = flags.fast;
    (u64::from(flags.nsw))
        | (u64::from(flags.nuw) << 1)
        | (u64::from(flags.exact) << 2)
        | (u64::from(f.nnan) << 3)
        | (u64::from(f.ninf) << 4)
        | (u64::from(f.nsz) << 5)
        | (u64::from(f.reassoc) << 6)
        | (u64::from(f.contract) << 7)
        | (u64::from(f.afn) << 8)
}

fn flags_from_bits(bits: u64) -> Flags {
    let bit = |i: u32| bits & (1 << i) != 0;
    Flags {
        nsw: bit(0),
        nuw: bit(1),
        exact: bit(2),
        fast: FastMath {
            nnan: bit(3),
            ninf: bit(4),
            nsz: bit(5),
            reassoc: bit(6),
            contract: bit(7),
            afn: bit(8),
        },
    }
}

// ===========================================================================
// puremp::Int encoding: sign byte + little-endian magnitude bytes
// ===========================================================================

fn write_int(w: &mut Writer, value: &Int) {
    w.u8(sign_code(value.sign()));
    w.bytes(&value.magnitude().to_bytes_le());
}

fn read_int(r: &mut Reader<'_>) -> Result<Int, DecodeError> {
    let sign = sign_from(r.u8()?)?;
    let mag = Nat::from_bytes_le(r.bytes()?);
    Ok(Int::from_sign_magnitude(sign, mag))
}

// ===========================================================================
// Reachable type / constant collection and dense renumbering
// ===========================================================================

/// The renumbering tables the encoder builds by walking the module: the
/// reachable types and constants, each in a topologically valid order, plus the
/// maps from their original handle to its dense serialized index.
struct Tables {
    types: Vec<TypeId>,
    type_index: DetHashMap<TypeId, u64>,
    consts: Vec<ConstId>,
    const_index: DetHashMap<ConstId, u64>,
}

impl Tables {
    #[inline]
    fn ty(&self, id: TypeId) -> u64 {
        self.type_index[&id]
    }

    #[inline]
    fn konst(&self, id: ConstId) -> u64 {
        self.const_index[&id]
    }
}

/// Walk `module`, collecting every reachable [`TypeId`] and [`ConstId`], and
/// build the dense renumbering. Ordering both tables by their original interning
/// index is a valid topological order: a composite type/aggregate constant is
/// always interned *after* its components, so components get smaller indices.
fn collect_tables(module: &Module) -> Tables {
    let consts_pool = module.consts();
    let types_ctx = module.types();

    // --- reachable constants (closure over aggregate elements) ---
    let mut const_set: DetHashSet<ConstId> = DetHashSet::default();
    let mut cstack: Vec<ConstId> = Vec::new();
    for f in module.functions() {
        for i in 0..f.value_count() {
            if let ValueDef::Const(c) = &f.value(ValueId::from_index(i)).def {
                cstack.push(*c);
            }
        }
    }
    for g in module.globals() {
        if let Some(c) = g.init {
            cstack.push(c);
        }
    }
    while let Some(c) = cstack.pop() {
        if const_set.insert(c)
            && let Const::Aggregate { elems, .. } = consts_pool.get(c)
        {
            cstack.extend(elems.iter().copied());
        }
    }

    // --- reachable types (closure over composite components) ---
    let mut type_set: DetHashSet<TypeId> = DetHashSet::default();
    let mut tstack: Vec<TypeId> = Vec::new();
    for g in module.globals() {
        tstack.push(g.ty);
    }
    for f in module.functions() {
        tstack.push(f.sig);
        for i in 0..f.value_count() {
            tstack.push(f.value(ValueId::from_index(i)).ty);
        }
        for i in 0..f.inst_count() {
            let inst = f.inst(InstId::from_index(i));
            tstack.push(inst.ty);
            match &inst.kind {
                InstKind::Alloca { elem_ty } => tstack.push(*elem_ty),
                InstKind::Load { ty, .. }
                | InstKind::Store { ty, .. }
                | InstKind::AtomicLoad { ty, .. }
                | InstKind::AtomicStore { ty, .. }
                | InstKind::AtomicRmw { ty, .. }
                | InstKind::CmpXchg { ty, .. } => tstack.push(*ty),
                InstKind::InlineAsm(asm) => tstack.extend(asm.outputs.iter().filter_map(|o| o.ty)),
                _ => {}
            }
        }
    }
    for &c in &const_set {
        tstack.push(consts_pool.get(c).type_id());
    }
    while let Some(t) = tstack.pop() {
        if type_set.insert(t) {
            match types_ctx.get(t) {
                Type::Array(elem, _) | Type::Vector(elem, _) => tstack.push(*elem),
                Type::Struct(fields) => tstack.extend(fields.iter().copied()),
                Type::Func(ft) => {
                    tstack.extend(ft.params.iter().copied());
                    tstack.push(ft.ret);
                }
                _ => {}
            }
        }
    }

    let mut types: Vec<TypeId> = type_set.into_iter().collect();
    types.sort_by_key(|t| t.index());
    let mut type_index: DetHashMap<TypeId, u64> = DetHashMap::default();
    for (pos, &t) in types.iter().enumerate() {
        type_index.insert(t, pos as u64);
    }

    let mut consts: Vec<ConstId> = const_set.into_iter().collect();
    consts.sort_by_key(|c| c.index());
    let mut const_index: DetHashMap<ConstId, u64> = DetHashMap::default();
    for (pos, &c) in consts.iter().enumerate() {
        const_index.insert(c, pos as u64);
    }

    Tables { types, type_index, consts, const_index }
}

// ===========================================================================
// Encoding
// ===========================================================================

/// Encode a whole [`Module`] to the compact, versioned `.lfb` byte form.
///
/// `names` is the [`StrInterner`] that backs the module's `Sym` names; it is
/// read (never mutated) to serialize function and global names as strings. See
/// the module docs for why the interner is required.
///
/// The output is deterministic: `encode(m, n) == encode(m, n)` for any `m`/`n`.
pub fn encode(module: &Module, names: &StrInterner) -> Vec<u8> {
    let tables = collect_tables(module);
    let mut w = Writer::new();
    w.raw(&MAGIC);
    w.uvarint(u64::from(VERSION));

    w.str(&module.name);
    match module.target() {
        Some(t) => {
            w.u8(1);
            w.str(t);
        }
        None => w.u8(0),
    }
    if *module.data_layout() == crate::ir::DataLayout::lp64() {
        w.str("");
    } else {
        w.str(&module.data_layout().to_spec());
    }

    // --- type table (topological order) ---
    w.uvarint(tables.types.len() as u64);
    for &tid in &tables.types {
        write_type(&mut w, module.types().get(tid), &tables);
    }

    // --- constant pool (topological order) ---
    w.uvarint(tables.consts.len() as u64);
    for &cid in &tables.consts {
        write_const(&mut w, module.consts().get(cid), &tables);
    }

    // --- globals (module order) ---
    let globals: Vec<&Global> = module.globals().collect();
    w.uvarint(globals.len() as u64);
    for (gi, g) in globals.into_iter().enumerate() {
        w.str(names.resolve(g.name));
        w.uvarint(tables.ty(g.ty));
        match g.init {
            Some(c) => {
                w.u8(1);
                w.uvarint(tables.konst(c));
            }
            None => w.u8(0),
        }
        let gid = GlobalId::from_index(gi);
        let space = module.global_addr_space(gid);
        let gattrs = module.global_attrs(gid);
        let mut byte = attrs_bits(gattrs);
        if space != 0 {
            byte |= ADDR_SPACE_BIT;
        }
        let ext = if gattrs.secret { GLOBAL_EXT_SECRET } else { 0 }
            | if gattrs.thread_local { GLOBAL_EXT_THREAD_LOCAL } else { 0 };
        if ext != 0 {
            byte |= ATTR_EXT_BIT;
        }
        w.u8(byte);
        if space != 0 {
            w.uvarint(u64::from(space));
        }
        if ext != 0 {
            w.uvarint(ext);
        }
    }

    // --- functions (module order) ---
    let functions: Vec<&Function> = module.functions().collect();
    w.uvarint(functions.len() as u64);
    for f in functions {
        write_function(&mut w, f, names, &tables);
    }

    w.finish()
}

fn write_type(w: &mut Writer, ty: &Type, t: &Tables) {
    match ty {
        Type::Void => w.u8(0),
        Type::Int(width) => {
            w.u8(1);
            w.uvarint(u64::from(*width));
        }
        Type::Float(kind) => {
            w.u8(2);
            w.u8(floatkind_code(*kind));
        }
        Type::Ptr => w.u8(3),
        Type::PtrIn(space) => {
            w.u8(7);
            w.uvarint(u64::from(*space));
        }
        Type::Array(elem, len) => {
            w.u8(4);
            w.uvarint(t.ty(*elem));
            w.uvarint(*len);
        }
        Type::Struct(fields) => {
            w.u8(5);
            w.uvarint(fields.len() as u64);
            for f in fields {
                w.uvarint(t.ty(*f));
            }
        }
        Type::Func(ft) => {
            w.u8(6);
            w.uvarint(ft.params.len() as u64);
            for p in &ft.params {
                w.uvarint(t.ty(*p));
            }
            w.uvarint(t.ty(ft.ret));
            w.u8(u8::from(ft.variadic));
        }
        // A distinct high tag, so other type additions can take the next small
        // ones without clashing; no version bump (older streams never used it).
        Type::Vector(elem, lanes) => {
            w.u8(16);
            w.uvarint(t.ty(*elem));
            w.uvarint(u64::from(*lanes));
        }
    }
}

fn write_const(w: &mut Writer, c: &Const, t: &Tables) {
    match c {
        Const::Int { ty, value } => {
            w.u8(0);
            w.uvarint(t.ty(*ty));
            write_int(w, value);
        }
        Const::Float { ty, bits } => {
            w.u8(1);
            w.uvarint(t.ty(*ty));
            match bits {
                FloatBits::F16(b) => {
                    w.u8(0);
                    w.raw(&b.to_le_bytes());
                }
                FloatBits::F32(b) => {
                    w.u8(1);
                    w.raw(&b.to_le_bytes());
                }
                FloatBits::F64(b) => {
                    w.u8(2);
                    w.raw(&b.to_le_bytes());
                }
            }
        }
        Const::Null(ty) => {
            w.u8(2);
            w.uvarint(t.ty(*ty));
        }
        Const::Poison(ty) => {
            w.u8(3);
            w.uvarint(t.ty(*ty));
        }
        Const::Aggregate { ty, elems } => {
            w.u8(4);
            w.uvarint(t.ty(*ty));
            w.uvarint(elems.len() as u64);
            for e in elems {
                w.uvarint(t.konst(*e));
            }
        }
        Const::Addr { ty, target, offset } => {
            w.u8(5);
            w.uvarint(t.ty(*ty));
            match target {
                AddrTarget::Global(g) => {
                    w.u8(0);
                    w.uvarint(g.index() as u64);
                }
                AddrTarget::Func(f) => {
                    w.u8(1);
                    w.uvarint(f.index() as u64);
                }
            }
            // Zigzag-encode the signed offset so small negatives stay short.
            w.uvarint(((*offset << 1) ^ (*offset >> 63)) as u64);
        }
    }
}

/// Bit 6 of a global's attribute byte (version 4): a varint address space
/// follows the byte.
const ADDR_SPACE_BIT: u8 = 1 << 6;

fn linkage_code(l: Linkage) -> u8 {
    match l {
        Linkage::External => 0,
        Linkage::Internal => 1,
        Linkage::Weak => 2,
    }
}

fn linkage_from(c: u8) -> Option<Linkage> {
    Some(match c {
        0 => Linkage::External,
        1 => Linkage::Internal,
        2 => Linkage::Weak,
        _ => return None,
    })
}

fn visibility_code(v: Visibility) -> u8 {
    match v {
        Visibility::Default => 0,
        Visibility::Hidden => 1,
        Visibility::Protected => 2,
    }
}

fn visibility_from(c: u8) -> Option<Visibility> {
    Some(match c {
        0 => Visibility::Default,
        1 => Visibility::Hidden,
        2 => Visibility::Protected,
        _ => return None,
    })
}

/// Pack a global's attributes into one byte: linkage in bits 0–1
/// (`0` external, `1` internal, `2` weak), `constant` in bit 2, `detached` in
/// bit 3, visibility in bits 4–5 (`0` default, `1` hidden, `2` protected).
/// (Bit 6, [`ADDR_SPACE_BIT`], is added by the encoder from version 4.)
fn attrs_bits(a: GlobalAttrs) -> u8 {
    linkage_code(a.linkage)
        | (u8::from(a.constant) << 2)
        | (u8::from(a.detached) << 3)
        | (visibility_code(a.visibility) << 4)
}

fn attrs_from_bits(b: u8) -> Result<GlobalAttrs, DecodeError> {
    let bad = || DecodeError::InvalidTag { what: "global-attrs", tag: u32::from(b) };
    let linkage = linkage_from(b & 3).ok_or_else(bad)?;
    let visibility = visibility_from((b >> 4) & 3).ok_or_else(bad)?;
    if b & !0b11_1111 != 0 {
        return Err(bad());
    }
    Ok(GlobalAttrs { linkage, visibility, constant: b & 4 != 0, detached: b & 8 != 0, ..GlobalAttrs::DEFAULT })
}

/// Pack a function's attributes into one byte: linkage in bits 0–1, visibility
/// in bits 2–3 (same codes as [`attrs_bits`]). (Bit 7, [`ATTR_EXT_BIT`], is
/// added by the encoder from version 5 when the function has secrets.)
fn func_attrs_bits(a: &FuncAttrs) -> u8 {
    linkage_code(a.linkage) | (visibility_code(a.visibility) << 2)
}

/// Write the version-5 function attributes: the attribute byte, and when
/// anything is secret the extension varint (plus the secret-parameter list).
fn write_func_attrs(w: &mut Writer, a: &FuncAttrs) {
    if !a.has_secrets() {
        w.u8(func_attrs_bits(a));
        return;
    }
    w.u8(func_attrs_bits(a) | ATTR_EXT_BIT);
    let params: Vec<usize> = a.secret_params().collect();
    let mut ext = 0;
    if a.secret_ret {
        ext |= FUNC_EXT_SECRET_RET;
    }
    if !params.is_empty() {
        ext |= FUNC_EXT_SECRET_PARAMS;
    }
    w.uvarint(ext);
    if !params.is_empty() {
        w.uvarint(params.len() as u64);
        for p in params {
            w.uvarint(p as u64);
        }
    }
}

/// Read a function's attributes as written for `version` (none before 3; the
/// extension from 5). `arity` bounds the secret-parameter indices.
fn read_func_attrs(r: &mut Reader<'_>, version: u64, arity: usize) -> Result<FuncAttrs, DecodeError> {
    if version < 3 {
        return Ok(FuncAttrs::DEFAULT);
    }
    let b = r.u8()?;
    let has_ext = version >= 5 && b & ATTR_EXT_BIT != 0;
    let mut attrs = func_attrs_from_bits(if version >= 5 { b & !ATTR_EXT_BIT } else { b })?;
    if has_ext {
        let ext = r.uvarint()?;
        if ext & !(FUNC_EXT_SECRET_RET | FUNC_EXT_SECRET_PARAMS) != 0 {
            return Err(DecodeError::InvalidTag {
                what: "func-attrs extension",
                tag: u32::try_from(ext).unwrap_or(u32::MAX),
            });
        }
        attrs.secret_ret = ext & FUNC_EXT_SECRET_RET != 0;
        if ext & FUNC_EXT_SECRET_PARAMS != 0 {
            let n = r.uindex()?;
            for _ in 0..n {
                let p = checked(r.uindex()?, arity, "parameter")?;
                attrs.set_param_secret(p, true);
            }
        }
    }
    Ok(attrs)
}

fn func_attrs_from_bits(b: u8) -> Result<FuncAttrs, DecodeError> {
    let bad = || DecodeError::InvalidTag { what: "func-attrs", tag: u32::from(b) };
    let linkage = linkage_from(b & 3).ok_or_else(bad)?;
    let visibility = visibility_from((b >> 2) & 3).ok_or_else(bad)?;
    if b & !0b1111 != 0 {
        return Err(bad());
    }
    Ok(FuncAttrs::new(linkage, visibility))
}

fn write_function(w: &mut Writer, f: &Function, names: &StrInterner, t: &Tables) {
    w.str(names.resolve(f.name));
    w.uvarint(t.ty(f.sig));
    write_func_attrs(w, &f.attrs);

    // Value table.
    w.uvarint(f.value_count() as u64);
    for i in 0..f.value_count() {
        let v = f.value(ValueId::from_index(i));
        w.uvarint(t.ty(v.ty));
        write_value_def(w, &v.def, t);
    }

    // Instruction arena.
    w.uvarint(f.inst_count() as u64);
    for i in 0..f.inst_count() {
        write_inst(w, f.inst(InstId::from_index(i)), t);
    }

    // Blocks.
    w.uvarint(f.block_count() as u64);
    for (_, b) in f.blocks() {
        w.uvarint(b.params().len() as u64);
        for p in b.params() {
            w.uvarint(p.index() as u64);
        }
        w.uvarint(b.insts().len() as u64);
        for inst in b.insts() {
            w.uvarint(inst.index() as u64);
        }
        match b.terminator() {
            Some(term) => {
                w.u8(1);
                w.uvarint(term.index() as u64);
            }
            None => w.u8(0),
        }
    }

    // Entry.
    match f.entry() {
        Some(e) => {
            w.u8(1);
            w.uvarint(e.index() as u64);
        }
        None => w.u8(0),
    }
}

fn write_value_def(w: &mut Writer, def: &ValueDef, t: &Tables) {
    match def {
        ValueDef::Inst(i) => {
            w.u8(0);
            w.uvarint(i.index() as u64);
        }
        ValueDef::Param(b, idx) => {
            w.u8(1);
            w.uvarint(b.index() as u64);
            w.uvarint(u64::from(*idx));
        }
        ValueDef::Const(c) => {
            w.u8(2);
            w.uvarint(t.konst(*c));
        }
        ValueDef::Global(g) => {
            w.u8(3);
            w.uvarint(g.index() as u64);
        }
        ValueDef::Func(fu) => {
            w.u8(4);
            w.uvarint(fu.index() as u64);
        }
    }
}

fn write_inst(w: &mut Writer, inst: &InstData, t: &Tables) {
    write_inst_kind(w, &inst.kind, t);
    w.uvarint(flags_bits(inst.flags));
    w.uvarint(t.ty(inst.ty));
    w.uvarint(inst.operands().len() as u64);
    for op in inst.operands() {
        w.uvarint(op.index() as u64);
    }
    match inst.result() {
        Some(v) => {
            w.u8(1);
            w.uvarint(v.index() as u64);
        }
        None => w.u8(0),
    }
}

fn write_inst_kind(w: &mut Writer, kind: &InstKind, t: &Tables) {
    match kind {
        InstKind::Bin(op) => {
            w.u8(0);
            w.u8(binop_code(*op));
        }
        InstKind::Unary(op) => {
            w.u8(1);
            w.u8(unop_code(*op));
        }
        InstKind::ICmp(p) => {
            w.u8(2);
            w.u8(intpred_code(*p));
        }
        InstKind::FCmp(p) => {
            w.u8(3);
            w.u8(floatpred_code(*p));
        }
        InstKind::Cast(op) => {
            w.u8(4);
            w.u8(cast_code(*op));
        }
        InstKind::Alloca { elem_ty } => {
            w.u8(5);
            w.uvarint(t.ty(*elem_ty));
        }
        InstKind::DynAlloca { align } => {
            w.u8(17);
            w.uvarint(u64::from(*align));
        }
        // A volatile access has its own tag (19/20) with the same payload, so
        // streams without one keep the original bytes (no version bump).
        InstKind::Load { ty, align, volatile, secret: false } => {
            w.u8(if *volatile { 19 } else { 6 });
            w.uvarint(t.ty(*ty));
            w.uvarint(u64::from(*align));
        }
        InstKind::Store { ty, align, volatile, secret: false } => {
            w.u8(if *volatile { 20 } else { 7 });
            w.uvarint(t.ty(*ty));
            w.uvarint(u64::from(*align));
        }
        // A secret access has its own tag (26/27) with a trailing volatile
        // byte, so streams without secrets keep their bytes (no version bump).
        InstKind::Load { ty, align, volatile, secret: true } => {
            w.u8(26);
            w.uvarint(t.ty(*ty));
            w.uvarint(u64::from(*align));
            w.u8(u8::from(*volatile));
        }
        InstKind::Store { ty, align, volatile, secret: true } => {
            w.u8(27);
            w.uvarint(t.ty(*ty));
            w.uvarint(u64::from(*align));
            w.u8(u8::from(*volatile));
        }
        InstKind::Declassify => w.u8(28),
        InstKind::AtomicLoad { ty, align, ordering } => {
            w.u8(21);
            w.uvarint(t.ty(*ty));
            w.uvarint(u64::from(*align));
            w.u8(ordering_code(*ordering));
        }
        InstKind::AtomicStore { ty, align, ordering } => {
            w.u8(22);
            w.uvarint(t.ty(*ty));
            w.uvarint(u64::from(*align));
            w.u8(ordering_code(*ordering));
        }
        InstKind::AtomicRmw { op, ty, align, ordering } => {
            w.u8(23);
            w.u8(rmw_code(*op));
            w.uvarint(t.ty(*ty));
            w.uvarint(u64::from(*align));
            w.u8(ordering_code(*ordering));
        }
        InstKind::CmpXchg { ty, align, success, failure } => {
            w.u8(24);
            w.uvarint(t.ty(*ty));
            w.uvarint(u64::from(*align));
            w.u8(ordering_code(*success));
            w.u8(ordering_code(*failure));
        }
        InstKind::Fence(ordering) => {
            w.u8(25);
            w.u8(ordering_code(*ordering));
        }
        InstKind::PtrAdd { inbounds } => {
            w.u8(8);
            w.u8(u8::from(*inbounds));
        }
        InstKind::Select => w.u8(9),
        InstKind::Freeze => w.u8(10),
        InstKind::Call => w.u8(11),
        InstKind::Syscall => w.u8(18),
        InstKind::Ret => w.u8(12),
        InstKind::Br(target) => {
            w.u8(13);
            w.uvarint(target.index() as u64);
        }
        InstKind::CondBr { if_true, if_false, true_args, false_args } => {
            w.u8(14);
            w.uvarint(if_true.index() as u64);
            w.uvarint(if_false.index() as u64);
            w.uvarint(u64::from(*true_args));
            w.uvarint(u64::from(*false_args));
        }
        InstKind::Switch(data) => {
            w.u8(15);
            w.uvarint(data.default.index() as u64);
            w.uvarint(u64::from(data.default_args));
            w.uvarint(data.cases.len() as u64);
            for case in &data.cases {
                write_int(w, &case.value);
                w.uvarint(case.target.index() as u64);
                w.uvarint(u64::from(case.args));
            }
        }
        InstKind::Unreachable => w.u8(16),
        // Vector ops use tags 40..=44 (`docs/ir-design.md` §6e); streams without
        // them keep their bytes, so no version bump was needed.
        InstKind::ExtractElement { lane } => {
            w.u8(40);
            w.uvarint(u64::from(*lane));
        }
        InstKind::InsertElement { lane } => {
            w.u8(41);
            w.uvarint(u64::from(*lane));
        }
        InstKind::ShuffleVector(mask) => {
            w.u8(42);
            w.uvarint(mask.len() as u64);
            for &m in mask.iter() {
                w.uvarint(u64::from(m));
            }
        }
        InstKind::Splat => w.u8(43),
        InstKind::Reduce(op) => {
            w.u8(44);
            w.u8(op.code());
        }
        // Inline asm uses tags 45/46 (`docs/ir-design.md` §6j); streams
        // without it keep their bytes, so no version bump was needed.
        InstKind::InlineAsm(asm) => {
            w.u8(45);
            w.str(&asm.template);
            w.u8(u8::from(asm.volatile));
            let name = |w: &mut Writer, n: &Option<String>| match n {
                Some(n) => {
                    w.u8(1);
                    w.str(n);
                }
                None => w.u8(0),
            };
            w.uvarint(asm.outputs.len() as u64);
            for o in &asm.outputs {
                w.str(&o.constraint);
                name(w, &o.name);
                match o.ty {
                    Some(ty) => {
                        w.u8(1);
                        w.uvarint(t.ty(ty));
                    }
                    None => w.u8(0),
                }
            }
            w.uvarint(asm.inputs.len() as u64);
            for i in &asm.inputs {
                w.str(&i.constraint);
                name(w, &i.name);
            }
            w.uvarint(asm.clobbers.len() as u64);
            for c in &asm.clobbers {
                w.str(c);
            }
        }
        InstKind::AsmOutput(n) => {
            w.u8(46);
            w.uvarint(u64::from(*n));
        }
    }
}

// ===========================================================================
// Decoding
// ===========================================================================

/// Decode a [`Module`] from the `.lfb` byte form produced by [`encode`].
///
/// `names` is the [`StrInterner`] into which function and global name strings
/// are (re-)interned to recover their `Sym` handles; passing the same interner
/// that encoded the module reproduces its exact handles. Any malformed input —
/// bad magic, unknown version, truncation, an out-of-range index, invalid UTF-8
/// — yields an [`Err`] and never panics.
pub fn decode(bytes: &[u8], names: &mut StrInterner) -> Result<Module, DecodeError> {
    let mut r = Reader::new(bytes);

    let magic = r.take(MAGIC.len())?;
    if magic != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let version = r.uvarint()?;
    if !(u64::from(MIN_VERSION)..=u64::from(VERSION)).contains(&version) {
        return Err(DecodeError::UnsupportedVersion(version.try_into().unwrap_or(u32::MAX)));
    }

    let module_name = r.str()?.to_owned();
    let mut module = Module::new(module_name);
    if version >= 4 {
        match r.u8()? {
            0 => {}
            1 => module.set_target(Some(r.str()?.to_owned())),
            t => return Err(DecodeError::InvalidTag { what: "target", tag: u32::from(t) }),
        }
        let spec = r.str()?;
        let layout = crate::ir::DataLayout::parse(spec).map_err(|_| DecodeError::InvalidDataLayout)?;
        module.set_data_layout(layout);
    }

    // --- type table: intern in order, mapping serialized index -> real TypeId ---
    let ntypes = r.uindex()?;
    let mut types: Vec<TypeId> = Vec::with_capacity(ntypes);
    for _ in 0..ntypes {
        let ty = read_type(&mut r, &types)?;
        let id = module.types_mut().intern(ty);
        types.push(id);
    }

    // --- constant pool ---
    let nconsts = r.uindex()?;
    let mut consts: Vec<ConstId> = Vec::with_capacity(nconsts);
    for _ in 0..nconsts {
        let c = read_const(&mut r, &types, &consts)?;
        let id = module.intern_const(c);
        consts.push(id);
    }

    // --- globals ---
    let nglobals = r.uindex()?;
    for _ in 0..nglobals {
        let name = names.intern(r.str()?);
        let ty = types[checked(r.uindex()?, types.len(), "type")?];
        let init = match r.u8()? {
            0 => None,
            1 => Some(consts[checked(r.uindex()?, consts.len(), "const")?]),
            t => return Err(DecodeError::InvalidTag { what: "global-init", tag: u32::from(t) }),
        };
        let (attrs, space) = if version >= 2 {
            let b = r.u8()?;
            let space = if version >= 4 && b & ADDR_SPACE_BIT != 0 { r.u32()? } else { 0 };
            let has_ext = version >= 5 && b & ATTR_EXT_BIT != 0;
            let low = if version >= 5 { b & !ATTR_EXT_BIT } else { b };
            let mut attrs = attrs_from_bits(if version >= 4 { low & !ADDR_SPACE_BIT } else { low })?;
            if has_ext {
                let ext = r.uvarint()?;
                if ext & !(GLOBAL_EXT_SECRET | GLOBAL_EXT_THREAD_LOCAL) != 0 {
                    return Err(DecodeError::InvalidTag {
                        what: "global-attrs extension",
                        tag: u32::try_from(ext).unwrap_or(u32::MAX),
                    });
                }
                attrs.secret = ext & GLOBAL_EXT_SECRET != 0;
                attrs.thread_local = ext & GLOBAL_EXT_THREAD_LOCAL != 0;
            }
            (attrs, space)
        } else {
            (GlobalAttrs::DEFAULT, 0)
        };
        let gid = module.define_global(Global { name, ty, init }, attrs);
        module.set_global_addr_space(gid, space);
    }

    // --- functions ---
    let nfuncs = r.uindex()?;
    for _ in 0..nfuncs {
        let f =
            read_function(&mut r, names, module.types(), &types, &consts, nglobals, nfuncs, version)?;
        module.functions.push(f);
    }

    // Address constants name globals/functions by module index; those counts are
    // only known now, so range-check them after the fact.
    for &c in &consts {
        if let Const::Addr { target, .. } = module.consts().get(c) {
            match *target {
                AddrTarget::Global(g) => {
                    checked(g.index(), nglobals, "global")?;
                }
                AddrTarget::Func(f) => {
                    checked(f.index(), nfuncs, "function")?;
                }
            }
        }
    }

    if r.remaining() != 0 {
        return Err(DecodeError::TrailingBytes);
    }
    Ok(module)
}

fn read_type(r: &mut Reader<'_>, types: &[TypeId]) -> Result<Type, DecodeError> {
    let resolve =
        |idx: usize| -> Result<TypeId, DecodeError> { Ok(types[checked(idx, types.len(), "type")?]) };
    Ok(match r.u8()? {
        0 => Type::Void,
        1 => Type::Int(r.u32()?),
        2 => Type::Float(floatkind_from(r.u8()?)?),
        3 => Type::Ptr,
        // `intern` normalizes a (malformed) `PtrIn(0)` back to `Ptr`.
        7 => Type::PtrIn(r.u32()?),
        4 => {
            let elem = resolve(r.uindex()?)?;
            let len = r.uvarint()?;
            Type::Array(elem, len)
        }
        5 => {
            let n = r.uindex()?;
            let mut fields = Vec::with_capacity(n);
            for _ in 0..n {
                fields.push(resolve(r.uindex()?)?);
            }
            Type::Struct(fields)
        }
        6 => {
            let n = r.uindex()?;
            let mut params = Vec::with_capacity(n);
            for _ in 0..n {
                params.push(resolve(r.uindex()?)?);
            }
            let ret = resolve(r.uindex()?)?;
            let variadic = r.u8()? != 0;
            Type::Func(FuncType { params, ret, variadic })
        }
        16 => {
            let elem = resolve(r.uindex()?)?;
            Type::Vector(elem, r.u32()?)
        }
        t => return Err(DecodeError::InvalidTag { what: "type", tag: u32::from(t) }),
    })
}

fn read_const(
    r: &mut Reader<'_>,
    types: &[TypeId],
    consts: &[ConstId],
) -> Result<Const, DecodeError> {
    let ty = |r: &mut Reader<'_>| -> Result<TypeId, DecodeError> {
        Ok(types[checked(r.uindex()?, types.len(), "type")?])
    };
    Ok(match r.u8()? {
        0 => {
            let ty = ty(r)?;
            let value = read_int(r)?;
            Const::Int { ty, value }
        }
        1 => {
            let ty = ty(r)?;
            let bits = match r.u8()? {
                0 => FloatBits::F16(u16::from_le_bytes(to_arr(r.take(2)?))),
                1 => FloatBits::F32(u32::from_le_bytes(to_arr(r.take(4)?))),
                2 => FloatBits::F64(u64::from_le_bytes(to_arr(r.take(8)?))),
                t => return Err(DecodeError::InvalidTag { what: "floatbits", tag: u32::from(t) }),
            };
            Const::Float { ty, bits }
        }
        2 => Const::Null(ty(r)?),
        3 => Const::Poison(ty(r)?),
        4 => {
            let ty = ty(r)?;
            let n = r.uindex()?;
            let mut elems = Vec::with_capacity(n);
            for _ in 0..n {
                elems.push(consts[checked(r.uindex()?, consts.len(), "const")?]);
            }
            Const::Aggregate { ty, elems }
        }
        5 => {
            let ty = ty(r)?;
            let kind = r.u8()?;
            // Ids are `u32`; the exact range is checked once the counts are known.
            let index = checked(r.uindex()?, u32::MAX as usize, "symbol")?;
            let target = match kind {
                0 => AddrTarget::Global(GlobalId::from_index(index)),
                1 => AddrTarget::Func(FuncId::from_index(index)),
                t => return Err(DecodeError::InvalidTag { what: "addr-target", tag: u32::from(t) }),
            };
            let z = r.uvarint()?;
            let offset = ((z >> 1) as i64) ^ -((z & 1) as i64);
            Const::Addr { ty, target, offset }
        }
        t => return Err(DecodeError::InvalidTag { what: "const", tag: u32::from(t) }),
    })
}

/// Convert a slice of exactly `N` bytes to an array (the length is guaranteed by
/// the caller's [`Reader::take`], so the conversion cannot fail).
fn to_arr<const N: usize>(slice: &[u8]) -> [u8; N] {
    let mut arr = [0u8; N];
    arr.copy_from_slice(slice);
    arr
}

#[allow(clippy::too_many_arguments)]
fn read_function(
    r: &mut Reader<'_>,
    names: &mut StrInterner,
    tcx: &crate::ir::TypeContext,
    types: &[TypeId],
    consts: &[ConstId],
    nglobals: usize,
    nfuncs: usize,
    version: u64,
) -> Result<Function, DecodeError> {
    let name = names.intern(r.str()?);
    let sig = types[checked(r.uindex()?, types.len(), "type")?];

    let mut f = Function::new(name, sig);
    // A secret parameter index is bounded by the signature's arity.
    let arity = match tcx.get(sig) {
        Type::Func(ft) => ft.params.len(),
        _ => 0,
    };
    f.attrs = read_func_attrs(r, version, arity)?;

    // Value table.
    let nvals = r.uindex()?;
    let mut values = Vec::with_capacity(nvals);
    for _ in 0..nvals {
        let ty = types[checked(r.uindex()?, types.len(), "type")?];
        let def = read_value_def(r, consts, nglobals, nfuncs)?;
        values.push(Value { def, ty });
    }

    // Instruction arena.
    let ninsts = r.uindex()?;
    let mut insts = Vec::with_capacity(ninsts);
    for _ in 0..ninsts {
        insts.push(read_inst(r, types, nvals)?);
    }

    // Blocks.
    let nblocks = r.uindex()?;
    let mut blocks = Vec::with_capacity(nblocks);
    for _ in 0..nblocks {
        let nparams = r.uindex()?;
        let mut params = Vec::with_capacity(nparams);
        for _ in 0..nparams {
            params.push(ValueId::from_index(checked(r.uindex()?, nvals, "value")?));
        }
        let nbi = r.uindex()?;
        let mut binsts = Vec::with_capacity(nbi);
        for _ in 0..nbi {
            binsts.push(InstId::from_index(checked(r.uindex()?, ninsts, "inst")?));
        }
        let terminator = match r.u8()? {
            0 => None,
            1 => Some(InstId::from_index(checked(r.uindex()?, ninsts, "inst")?)),
            t => return Err(DecodeError::InvalidTag { what: "terminator", tag: u32::from(t) }),
        };
        // `Block`'s fields are private but visible here (`binary` is a
        // descendant of `ir`), so populate them via a struct literal.
        blocks.push(Block { params, insts: binsts, terminator });
    }

    let entry = match r.u8()? {
        0 => None,
        1 => Some(BlockId::from_index(checked(r.uindex()?, nblocks, "block")?)),
        t => return Err(DecodeError::InvalidTag { what: "entry", tag: u32::from(t) }),
    };

    // Cross-check the value defs now that inst/block counts are known, so the
    // returned function cannot later panic on a dangling id.
    for v in &values {
        match &v.def {
            ValueDef::Inst(i) => {
                checked(i.index(), ninsts, "inst")?;
            }
            ValueDef::Param(b, _) => {
                checked(b.index(), nblocks, "block")?;
            }
            ValueDef::Const(_) | ValueDef::Global(_) | ValueDef::Func(_) => {}
        }
    }

    // Populate the flat arenas directly (`binary` is a descendant of `ir`).
    let uses = rebuild_uses(&values, &insts);
    let value_cache = rebuild_value_cache(&values);
    f.values = values;
    f.uses = uses;
    f.insts = insts;
    f.blocks = blocks;
    f.entry = entry;
    f.value_cache = value_cache;
    Ok(f)
}

fn read_value_def(
    r: &mut Reader<'_>,
    consts: &[ConstId],
    nglobals: usize,
    nfuncs: usize,
) -> Result<ValueDef, DecodeError> {
    Ok(match r.u8()? {
        // Inst/Param targets are cross-checked by the caller once the inst and
        // block counts are known.
        0 => ValueDef::Inst(InstId::from_index(r.uindex()?)),
        1 => {
            let b = BlockId::from_index(r.uindex()?);
            let idx = r.u32()?;
            ValueDef::Param(b, idx)
        }
        // `ConstId` has a private constructor, so it is resolved through the
        // decoder's serialized-index -> real-handle mapping vector.
        2 => ValueDef::Const(consts[checked(r.uindex()?, consts.len(), "const")?]),
        3 => ValueDef::Global(GlobalId::from_index(checked(r.uindex()?, nglobals, "global")?)),
        4 => ValueDef::Func(FuncId::from_index(checked(r.uindex()?, nfuncs, "function")?)),
        t => return Err(DecodeError::InvalidTag { what: "value-def", tag: u32::from(t) }),
    })
}

fn read_inst(r: &mut Reader<'_>, types: &[TypeId], nvals: usize) -> Result<InstData, DecodeError> {
    let kind = read_inst_kind(r, types)?;
    let flags = flags_from_bits(r.uvarint()?);
    let ty = types[checked(r.uindex()?, types.len(), "type")?];
    let nops = r.uindex()?;
    let mut operands = Vec::with_capacity(nops);
    for _ in 0..nops {
        operands.push(ValueId::from_index(checked(r.uindex()?, nvals, "value")?));
    }
    let result = match r.u8()? {
        0 => None,
        1 => Some(ValueId::from_index(checked(r.uindex()?, nvals, "value")?)),
        t => return Err(DecodeError::InvalidTag { what: "result", tag: u32::from(t) }),
    };
    Ok(InstData { kind, flags, ty, operands, result })
}

fn read_inst_kind(r: &mut Reader<'_>, types: &[TypeId]) -> Result<InstKind, DecodeError> {
    let ty = |r: &mut Reader<'_>| -> Result<TypeId, DecodeError> {
        Ok(types[checked(r.uindex()?, types.len(), "type")?])
    };
    Ok(match r.u8()? {
        0 => InstKind::Bin(binop_from(r.u8()?)?),
        1 => InstKind::Unary(unop_from(r.u8()?)?),
        2 => InstKind::ICmp(intpred_from(r.u8()?)?),
        3 => InstKind::FCmp(floatpred_from(r.u8()?)?),
        4 => InstKind::Cast(cast_from(r.u8()?)?),
        5 => InstKind::Alloca { elem_ty: ty(r)? },
        tag @ (6 | 19) => {
            let ty = ty(r)?;
            let align = r.u32()?;
            InstKind::Load { ty, align, volatile: tag == 19, secret: false }
        }
        tag @ (7 | 20) => {
            let ty = ty(r)?;
            let align = r.u32()?;
            InstKind::Store { ty, align, volatile: tag == 20, secret: false }
        }
        tag @ (26 | 27) => {
            let ty = ty(r)?;
            let align = r.u32()?;
            let volatile = match r.u8()? {
                0 => false,
                1 => true,
                t => return Err(DecodeError::InvalidTag { what: "volatile", tag: u32::from(t) }),
            };
            if tag == 26 {
                InstKind::Load { ty, align, volatile, secret: true }
            } else {
                InstKind::Store { ty, align, volatile, secret: true }
            }
        }
        28 => InstKind::Declassify,
        8 => InstKind::PtrAdd { inbounds: r.u8()? != 0 },
        9 => InstKind::Select,
        10 => InstKind::Freeze,
        11 => InstKind::Call,
        12 => InstKind::Ret,
        13 => InstKind::Br(BlockId::from_index(r.uindex()?)),
        14 => {
            let if_true = BlockId::from_index(r.uindex()?);
            let if_false = BlockId::from_index(r.uindex()?);
            let true_args = r.u32()?;
            let false_args = r.u32()?;
            InstKind::CondBr { if_true, if_false, true_args, false_args }
        }
        15 => {
            let default = BlockId::from_index(r.uindex()?);
            let default_args = r.u32()?;
            let ncases = r.uindex()?;
            let mut cases = Vec::with_capacity(ncases);
            for _ in 0..ncases {
                let value = read_int(r)?;
                let target = BlockId::from_index(r.uindex()?);
                let args = r.u32()?;
                cases.push(SwitchCase { value, target, args });
            }
            InstKind::Switch(Box::new(SwitchData { default, default_args, cases }))
        }
        16 => InstKind::Unreachable,
        17 => InstKind::DynAlloca { align: r.u32()? },
        18 => InstKind::Syscall,
        21 => {
            let ty = ty(r)?;
            let align = r.u32()?;
            InstKind::AtomicLoad { ty, align, ordering: ordering_from(r.u8()?)? }
        }
        22 => {
            let ty = ty(r)?;
            let align = r.u32()?;
            InstKind::AtomicStore { ty, align, ordering: ordering_from(r.u8()?)? }
        }
        23 => {
            let op = rmw_from(r.u8()?)?;
            let ty = ty(r)?;
            let align = r.u32()?;
            InstKind::AtomicRmw { op, ty, align, ordering: ordering_from(r.u8()?)? }
        }
        24 => {
            let ty = ty(r)?;
            let align = r.u32()?;
            let success = ordering_from(r.u8()?)?;
            let failure = ordering_from(r.u8()?)?;
            InstKind::CmpXchg { ty, align, success, failure }
        }
        25 => InstKind::Fence(ordering_from(r.u8()?)?),
        40 => InstKind::ExtractElement { lane: r.u32()? },
        41 => InstKind::InsertElement { lane: r.u32()? },
        42 => {
            let n = r.uindex()?;
            // Each index is at least one byte, so a count beyond the remaining
            // input is corrupt (and must not drive a huge allocation).
            if n > r.remaining() {
                return Err(DecodeError::UnexpectedEof);
            }
            let mut mask = Vec::with_capacity(n);
            for _ in 0..n {
                mask.push(r.u32()?);
            }
            InstKind::ShuffleVector(mask.into_boxed_slice())
        }
        43 => InstKind::Splat,
        44 => {
            let c = r.u8()?;
            let op = ReduceOp::from_code(u64::from(c))
                .ok_or(DecodeError::InvalidTag { what: "reduce op", tag: u32::from(c) })?;
            InstKind::Reduce(op)
        }
        45 => {
            let template = r.str()?.to_owned();
            let volatile = match r.u8()? {
                0 => false,
                1 => true,
                t => return Err(DecodeError::InvalidTag { what: "asm flags", tag: u32::from(t) }),
            };
            let name = |r: &mut Reader<'_>| -> Result<Option<String>, DecodeError> {
                match r.u8()? {
                    0 => Ok(None),
                    1 => Ok(Some(r.str()?.to_owned())),
                    t => Err(DecodeError::InvalidTag { what: "asm operand name", tag: u32::from(t) }),
                }
            };
            // Every entry is at least one byte, so a count beyond the remaining
            // input is corrupt (and must not drive a huge allocation).
            let count = |r: &mut Reader<'_>| -> Result<usize, DecodeError> {
                let n = r.uindex()?;
                if n > r.remaining() { Err(DecodeError::UnexpectedEof) } else { Ok(n) }
            };
            let nout = count(r)?;
            let mut outputs = Vec::with_capacity(nout);
            for _ in 0..nout {
                let constraint = r.str()?.to_owned();
                let name = name(r)?;
                let ty = match r.u8()? {
                    0 => None,
                    1 => Some(ty(r)?),
                    t => return Err(DecodeError::InvalidTag { what: "asm output type", tag: u32::from(t) }),
                };
                outputs.push(crate::ir::inst::AsmOutput { constraint, name, ty });
            }
            let nin = count(r)?;
            let mut inputs = Vec::with_capacity(nin);
            for _ in 0..nin {
                let constraint = r.str()?.to_owned();
                inputs.push(crate::ir::inst::AsmInput { constraint, name: name(r)? });
            }
            let nclob = count(r)?;
            let mut clobbers = Vec::with_capacity(nclob);
            for _ in 0..nclob {
                clobbers.push(r.str()?.to_owned());
            }
            InstKind::InlineAsm(Box::new(crate::ir::inst::InlineAsm { template, outputs, inputs, clobbers, volatile }))
        }
        46 => InstKind::AsmOutput(r.u32()?),
        t => return Err(DecodeError::InvalidTag { what: "opcode", tag: u32::from(t) }),
    })
}

/// Rebuild the def→use lists from the instruction arena, matching the order the
/// builder produces them (insts in index order, operands left to right).
fn rebuild_uses(values: &[Value], insts: &[InstData]) -> Vec<Vec<Use>> {
    let mut uses: Vec<Vec<Use>> = vec![Vec::new(); values.len()];
    for (i, inst) in insts.iter().enumerate() {
        let id = InstId::from_index(i);
        for (slot, op) in inst.operands().iter().enumerate() {
            uses[op.index()].push(Use { inst: id, operand: slot as u32 });
        }
    }
    uses
}

/// Rebuild the dedup cache for reference values (constants, global and function
/// refs), as the builder maintains it.
fn rebuild_value_cache(values: &[Value]) -> HashMap<ValueDef, ValueId> {
    let mut cache = HashMap::new();
    for (i, v) in values.iter().enumerate() {
        match &v.def {
            ValueDef::Const(_) | ValueDef::Global(_) | ValueDef::Func(_) => {
                cache.entry(v.def.clone()).or_insert_with(|| ValueId::from_index(i));
            }
            ValueDef::Inst(_) | ValueDef::Param(_, _) => {}
        }
    }
    cache
}

#[cfg(test)]
mod tests {
    use super::{
        DecodeError, MAGIC, Tables, VERSION, Writer, attrs_from_bits, decode, encode,
        func_attrs_from_bits, write_inst_kind,
    };
    use crate::support::hash::DetHashMap;
    use crate::ir::inst::{BinOp, CastOp, FastMath, Flags, FloatPred, IntPred};
    use crate::ir::types::FloatKind;
    use crate::ir::value::{AddrTarget, Const, FloatBits};
    use crate::ir::{FuncAttrs, FuncId, Global, GlobalAttrs, GlobalId, Linkage, Module, Visibility};
    use crate::support::StrInterner;
    use puremp::Int;

    /// Build a comprehensive module exercising every construct the format must
    /// carry: multiple functions (a definition and a body with a loop), a
    /// back-edge passing block arguments, `call`, `select`, `switch` with a wide
    /// case value, wide/negative integer constants, floats/fast-math, memory
    /// ops, poison/freeze, and struct/array/ptr types with a global initializer.
    fn build_sample(interner: &mut StrInterner) -> Module {
        let mut m = Module::new("sample");

        let i1 = m.types_mut().bool();
        let i8 = m.types_mut().int(8);
        let i32t = m.types_mut().int(32);
        let i64t = m.types_mut().int(64);
        let f32t = m.types_mut().float(FloatKind::F32);
        let f64t = m.types_mut().float(FloatKind::F64);
        let ptr = m.types_mut().ptr();
        let _arr = m.types_mut().array(i32t, 4);
        let strct = m.types_mut().struct_(vec![i8, i32t, ptr]);
        let _void = m.types_mut().void();
        let _ = i1;

        // A global of struct type with an aggregate initializer.
        let c_i8 = m.intern_const(Const::Int { ty: i8, value: Int::from_i64(7) });
        let c_i32 = m.intern_const(Const::Int { ty: i32t, value: Int::from_i64(-5) });
        let c_null = m.intern_const(Const::Null(ptr));
        let agg = m.intern_const(Const::Aggregate { ty: strct, elems: vec![c_i8, c_i32, c_null] });
        let gname = interner.intern("g");
        let g = m.add_global(Global { name: gname, ty: strct, init: Some(agg) });

        // A second function, given a real body.
        let helper_sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
        let helper_name = interner.intern("helper");
        let helper = m.declare_function(helper_name, helper_sig);

        let main_sig = m.types_mut().func(vec![i32t], i32t, false);
        let main_name = interner.intern("main");
        let main = m.declare_function(main_name, main_sig);

        // helper(a, b) = a * b
        {
            let mut hb = m.build(helper);
            let e = hb.create_entry_block();
            let a = hb.param(e, 0);
            let b = hb.param(e, 1);
            let c = hb.mul(a, b, Flags::NONE);
            hb.ret(Some(c));
        }

        // main(n): sum 0..n via a loop, then call helper, select, switch.
        {
            let mut b = m.build(main);
            let entry = b.create_entry_block();
            let n = b.param(entry, 0);
            let header = b.create_block(&[i32t, i32t]);
            let body = b.create_block(&[]);
            let exit = b.create_block(&[i32t]);

            let zero = b.const_i64(i32t, 0);
            b.br(header, &[zero, zero]);

            b.switch_to(header);
            let acc = b.param(header, 0);
            let iv = b.param(header, 1);
            let cond = b.icmp(IntPred::Slt, iv, n);
            b.cond_br(cond, body, &[], exit, &[acc]);

            b.switch_to(body);
            let acc1 = b.add(acc, iv, Flags::nsw());
            let one = b.const_i64(i32t, 1);
            let inext = b.add(iv, one, Flags::NONE);
            b.br(header, &[acc1, inext]); // back-edge, passes block arguments

            b.switch_to(exit);
            let res = b.param(exit, 0);
            let callee = b.func_ref(helper);
            let called = b.call(callee, &[res, n], i32t).unwrap();
            let z2 = b.const_i64(i32t, 0);
            let cond2 = b.icmp(IntPred::Sgt, res, z2);
            let sel = b.select(cond2, called, res);

            // Floats + fast-math + cast.
            let cf1 = b.const_float(f32t, FloatBits::F32(0x3f80_0000));
            let cf2 = b.const_float(f32t, FloatBits::F32(0x4000_0000));
            let fsum = b.bin(
                BinOp::FAdd,
                cf1,
                cf2,
                Flags::fast(FastMath { nnan: true, reassoc: true, ..FastMath::default() }),
            );
            let _fc = b.fcmp(FloatPred::Olt, cf1, cf2, Flags::NONE);
            let _ci = b.cast(CastOp::FpToSi, fsum, i32t);
            let cf64 = b.const_float(f64t, FloatBits::F64(0x4010_0000_0000_0000));
            let _n64 = b.fneg(cf64, Flags::fast(FastMath { nsz: true, ..FastMath::default() }));

            // Poison + freeze.
            let pois = b.poison(i32t);
            let _fr = b.freeze(pois);

            // Memory ops + pointer arithmetic + a global reference.
            let slot = b.alloca(i32t);
            b.store(i32t, slot, res, 4);
            let _ld = b.load(i32t, slot, 4);
            let off = b.const_i64(i64t, 8);
            let _pa = b.ptr_add(slot, off, true);
            let gref = b.global_ref(g);
            let _gp = b.ptr_add(gref, off, false);

            // Wide / negative integer constants.
            let wide_c = b.const_int(i64t, Int::from_i64(2).pow(100));
            let negwide_c = b.const_int(i64t, Int::from_i64(2).pow(90).neg());
            let _ws = b.add(wide_c, negwide_c, Flags::NONE);

            // Switch with a wide and a negative case value.
            let ca = b.create_block(&[]);
            let cb = b.create_block(&[]);
            let dflt = b.create_block(&[]);
            b.switch(
                sel,
                dflt,
                &[],
                vec![
                    (Int::from_i64(2).pow(80), ca, vec![]),
                    (Int::from_i64(-3), cb, vec![]),
                ],
            );
            b.switch_to(ca);
            b.ret(Some(res));
            b.switch_to(cb);
            b.ret(Some(called));
            b.switch_to(dflt);
            b.ret(Some(sel));
        }

        m
    }

    #[test]
    fn round_trips_losslessly() {
        let mut interner = StrInterner::new();
        let m = build_sample(&mut interner);

        let bytes = encode(&m, &interner);

        // Encoding is deterministic.
        assert_eq!(bytes, encode(&m, &interner), "encode must be deterministic");

        // decode(encode(m)) succeeds and re-encodes to identical bytes.
        let m2 = decode(&bytes, &mut interner).expect("decode should succeed");
        let bytes2 = encode(&m2, &interner);
        assert_eq!(bytes, bytes2, "encode(decode(encode(m))) == encode(m)");

        // And the fixed point is stable through another round.
        let m3 = decode(&bytes2, &mut interner).expect("second decode should succeed");
        assert_eq!(bytes, encode(&m3, &interner));

        // Spot-check structure via the public accessors.
        assert_eq!(m2.name, "sample");
        assert_eq!(m2.functions().count(), 2);
        assert_eq!(m2.globals().count(), 1);
        let names: Vec<&str> =
            m2.functions().map(|f| interner.resolve(f.name)).collect();
        assert_eq!(names, vec!["helper", "main"]);
        // `main` has entry + header + body + exit + three switch arms = 7 blocks.
        let main = m2.functions().nth(1).unwrap();
        assert_eq!(main.block_count(), 7);
        assert!(main.entry().is_some());
    }

    #[test]
    fn dyn_alloca_round_trips() {
        let mut interner = StrInterner::new();
        let mut m = Module::new("dyn");
        let i64t = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![i64t], i64t, false);
        let f = m.declare_function(interner.intern("d"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let n = b.param(e, 0);
            let p = b.dyn_alloca(n, 64);
            let v = b.load(i64t, p, 8);
            b.ret(Some(v));
        }
        let bytes = encode(&m, &interner);
        let mut back = StrInterner::new();
        let m2 = decode(&bytes, &mut back).expect("decode should succeed");
        assert_eq!(encode(&m2, &back), bytes, "binary form must be stable");
        let func = m2.function(crate::ir::FuncId::from_index(0));
        let has = (0..func.inst_count()).any(|i| {
            matches!(
                func.inst(crate::ir::InstId::from_index(i)).kind,
                crate::ir::InstKind::DynAlloca { align: 64 }
            )
        });
        assert!(has, "decoded module must contain dyn_alloca align 64");
    }

    #[test]
    fn syscall_round_trips() {
        let mut interner = StrInterner::new();
        let mut m = Module::new("sys");
        let i64t = m.types_mut().int(64);
        let ptr = m.types_mut().ptr();
        let sig = m.types_mut().func(vec![i64t, ptr], i64t, false);
        let f = m.declare_function(interner.intern("s"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let x = b.param(e, 0);
            let p = b.param(e, 1);
            let nr = b.const_i64(i64t, 9);
            b.syscall(nr, &[]);
            let r = b.syscall(nr, &[x, p, x, x, p, x]);
            b.ret(Some(r));
        }
        let bytes = encode(&m, &interner);
        let mut back = StrInterner::new();
        let m2 = decode(&bytes, &mut back).expect("decode should succeed");
        assert_eq!(encode(&m2, &back), bytes, "binary form must be stable");
        let func = m2.function(crate::ir::FuncId::from_index(0));
        let arities: Vec<usize> = (0..func.inst_count())
            .map(|i| func.inst(crate::ir::InstId::from_index(i)))
            .filter(|d| matches!(d.kind, crate::ir::InstKind::Syscall))
            .map(|d| d.operands().len())
            .collect();
        assert_eq!(arities, vec![1, 7], "decoded syscalls keep their operands");
        assert!(crate::verify::verify_module(&m2).is_ok());
    }

    #[test]
    fn volatile_and_atomics_round_trip() {
        let mut interner = StrInterner::new();
        let m = crate::ir::tests::atomics_module(&mut interner);
        let bytes = encode(&m, &interner);
        let mut back = StrInterner::new();
        let m2 = decode(&bytes, &mut back).expect("decode should succeed");
        assert_eq!(encode(&m2, &back), bytes, "binary form must be stable");
        assert_eq!(
            crate::ir::text::print_module(&m, &interner),
            crate::ir::text::print_module(&m2, &back),
            "the decoded module is the original"
        );
        assert!(crate::verify::verify_module(&m2).is_ok());
    }

    #[test]
    fn plain_memory_ops_keep_their_original_tags() {
        // Backward compatibility: a module with no volatile access encodes a
        // plain load/store with the original tags 6/7 and the original payload,
        // so version-2 streams written before volatile existed decode unchanged.
        let mut interner = StrInterner::new();
        let mut m = Module::new("plain");
        let i32t = m.types_mut().int(32);
        let ptr = m.types_mut().ptr();
        let sig = m.types_mut().func(vec![ptr], i32t, false);
        let f = m.declare_function(interner.intern("f"), sig);
        let (plain, vol) = {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let p = b.param(e, 0);
            let v = b.load(i32t, p, 4);
            b.store(i32t, p, v, 4);
            b.ret(Some(v));
            (
                crate::ir::InstKind::Load { ty: i32t, align: 4, volatile: false, secret: false },
                crate::ir::InstKind::Load { ty: i32t, align: 4, volatile: true, secret: false },
            )
        };
        let mut type_index = DetHashMap::default();
        type_index.insert(i32t, 0);
        let t = Tables { types: vec![i32t], type_index, consts: Vec::new(), const_index: DetHashMap::default() };
        let mut w = Writer::new();
        write_inst_kind(&mut w, &plain, &t);
        assert_eq!(w.buf, [6, 0, 4], "plain load keeps tag 6 and its payload");
        let mut w = Writer::new();
        write_inst_kind(&mut w, &vol, &t);
        assert_eq!(w.buf[0], 19, "volatile load uses tag 19");
        let bytes = encode(&m, &interner);
        assert_eq!(bytes[4], VERSION as u8);
        // Volatile/atomics needed no bump (version 2); version 4 is the
        // target/data-layout header and per-global address spaces, version 5
        // the attribute extensions (secrecy).
        assert_eq!(VERSION, 5);
        let m2 = decode(&bytes, &mut interner).expect("decode");
        let func = m2.function(crate::ir::FuncId::from_index(0));
        assert!((0..func.inst_count()).all(|i| !func.inst(crate::ir::InstId::from_index(i)).kind.is_volatile()));
    }

    #[test]
    fn bad_atomic_ordering_byte_is_rejected() {
        let mut interner = StrInterner::new();
        let mut m = Module::new("f");
        let void = m.types_mut().void();
        let sig = m.types_mut().func(vec![], void, false);
        let f = m.declare_function(interner.intern("f"), sig);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            b.fence(crate::ir::AtomicOrdering::SeqCst);
            b.ret(None);
        }
        let mut bytes = encode(&m, &interner);
        // The fence is `25 <ordering>`; corrupt the ordering byte (seq_cst = 4).
        let at = bytes.windows(2).rposition(|w| w == [25, 4]).expect("fence encoded as [25, 4]");
        bytes[at + 1] = 9;
        assert!(matches!(
            decode(&bytes, &mut interner),
            Err(DecodeError::InvalidTag { what: "atomic ordering", tag: 9 })
        ));
    }

    #[test]
    fn empty_module_round_trips() {
        let interner = StrInterner::new();
        let m = Module::new("empty");
        let bytes = encode(&m, &interner);
        let mut back = StrInterner::new();
        let m2 = decode(&bytes, &mut back).expect("empty module decodes");
        assert_eq!(m2.name, "empty");
        assert_eq!(m2.functions().count(), 0);
        assert_eq!(encode(&m2, &back), bytes);
    }

    #[test]
    fn wide_and_negative_ints_survive() {
        let interner = StrInterner::new();
        let mut m = Module::new("ints");
        let i128t = m.types_mut().int(128);
        let big = Int::from_i64(2).pow(127).sub(&Int::from_i64(1)); // 2^127 - 1
        let neg = Int::from_i64(-1).mul(&Int::from_i64(2).pow(96)); // -2^96
        let zero = Int::from_i64(0);
        let cb = m.intern_const(Const::Int { ty: i128t, value: big.clone() });
        let cn = m.intern_const(Const::Int { ty: i128t, value: neg.clone() });
        let cz = m.intern_const(Const::Int { ty: i128t, value: zero.clone() });
        let _ = (cb, cn, cz);

        let bytes = encode(&m, &interner);
        let mut back = StrInterner::new();
        let m2 = decode(&bytes, &mut back).expect("decode");
        assert_eq!(encode(&m2, &back), bytes, "wide/negative/zero ints round-trip");
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut interner = StrInterner::new();
        let mut bytes = vec![b'X', b'X', b'X', b'X'];
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        assert!(matches!(decode(&bytes, &mut interner), Err(DecodeError::BadMagic)));
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let mut interner = StrInterner::new();
        let mut bytes = MAGIC.to_vec();
        bytes.push(VERSION as u8 + 1); // a version this build does not read
        bytes.extend_from_slice(&[0, 0, 0]);
        match decode(&bytes, &mut interner) {
            Err(DecodeError::UnsupportedVersion(v)) => assert_eq!(v, VERSION + 1),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn truncated_input_fails_gracefully() {
        let mut interner = StrInterner::new();
        let m = build_sample(&mut interner);
        let bytes = encode(&m, &interner);

        // Every proper prefix must error (never panic, never succeed).
        for k in 0..bytes.len() {
            let mut fresh = StrInterner::new();
            let result = decode(&bytes[..k], &mut fresh);
            assert!(result.is_err(), "truncation at {k} should fail, got {result:?}");
        }
        // The full stream still decodes.
        assert!(decode(&bytes, &mut interner).is_ok());
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut interner = StrInterner::new();
        let m = Module::new("m");
        let mut bytes = encode(&m, &interner);
        bytes.push(0xff); // one extra byte past a complete module
        assert!(matches!(decode(&bytes, &mut interner), Err(DecodeError::TrailingBytes)));
    }

    #[test]
    fn garbage_after_header_does_not_panic() {
        let mut interner = StrInterner::new();
        // Valid header, then nonsense: must be a graceful Err.
        let mut bytes = MAGIC.to_vec();
        bytes.push(VERSION as u8);
        bytes.extend_from_slice(&[0xff; 32]);
        let _ = decode(&bytes, &mut interner); // must not panic
    }

    /// Global attributes and address constants (to a later global, a function,
    /// with positive and negative offsets) survive encode → decode.
    #[test]
    fn global_attrs_and_address_constants_round_trip() {
        let src = "module \"gd\"\n\
                   global internal constant @tab : [2 x ptr] = [2 x ptr] (ptr @x + 16, ptr @f)\n\
                   global weak @x : [4 x i64] = [4 x i64] poison\n\
                   global constant detached @d : ptr = ptr @tab - 8\n\
                   func @f() -> void {\nentry ^0:\n  ret\n}\n";
        let mut interner = StrInterner::new();
        let file = crate::support::diagnostics::FileId::new(0);
        let m = crate::ir::text::parse_module(src, file, &mut interner).expect("parse");
        let bytes = encode(&m, &interner);
        let m2 = decode(&bytes, &mut interner).expect("decode");
        assert_eq!(encode(&m2, &interner), bytes, "re-encode is byte-identical");
        assert_eq!(
            crate::ir::text::print_module(&m2, &interner),
            crate::ir::text::print_module(&m, &interner),
            "decoded module prints identically"
        );
        assert_eq!(
            m2.global_attrs(GlobalId::from_index(0)),
            GlobalAttrs { linkage: Linkage::Internal, constant: true, ..GlobalAttrs::DEFAULT }
        );
    }

    /// Visibility (globals and functions) and function linkage survive encode →
    /// decode, and a version-2 stream (no function attribute byte) still decodes
    /// with default function attributes.
    #[test]
    fn visibility_and_function_attrs_round_trip() {
        let src = "module \"v\"\n\
                   global hidden constant @h : i32 = i32 1\n\
                   global weak protected @p : i32 = i32 2\n\
                   func internal @i() -> void {\nentry ^0:\n  ret\n}\n\
                   func weak hidden @w() -> void {\nentry ^0:\n  ret\n}\n\
                   func protected @x() -> void\n";
        let mut interner = StrInterner::new();
        let file = crate::support::diagnostics::FileId::new(0);
        let m = crate::ir::text::parse_module(src, file, &mut interner).expect("parse");
        let bytes = encode(&m, &interner);
        let m2 = decode(&bytes, &mut interner).expect("decode");
        assert_eq!(crate::ir::text::print_module(&m2, &interner), crate::ir::text::print_module(&m, &interner));
        assert_eq!(m2.global_attrs(GlobalId::from_index(0)).visibility, Visibility::Hidden);
        assert_eq!(
            m2.global_attrs(GlobalId::from_index(1)),
            GlobalAttrs { linkage: Linkage::Weak, visibility: Visibility::Protected, ..GlobalAttrs::DEFAULT }
        );
        let attrs: Vec<FuncAttrs> = m2.functions().map(|f| f.attrs.clone()).collect();
        assert_eq!(
            attrs,
            [
                FuncAttrs::new(Linkage::Internal, Visibility::Default),
                FuncAttrs::new(Linkage::Weak, Visibility::Hidden),
                FuncAttrs::new(Linkage::External, Visibility::Protected),
            ]
        );
        // Bad visibility bits are rejected, not misread.
        assert!(attrs_from_bits(3 << 4).is_err());
        assert!(func_attrs_from_bits(3 << 2).is_err());

        // A version-2 stream: one bodiless function, no attribute byte.
        let mut m = Module::new("v2");
        let void = m.types_mut().void();
        let sig = m.types_mut().func(vec![], void, false);
        m.declare_function(interner.intern("decl"), sig);
        let mut bytes = encode(&m, &interner);
        // The stream ends `<name> <sig> <attrs = 0> <0 values> <0 insts> <0 blocks> <no entry>`.
        let n = bytes.len();
        assert_eq!(&bytes[n - 5..], &[0, 0, 0, 0, 0]);
        bytes.remove(n - 5);
        // Also drop the v4 header's `<no target> <empty layout>` after `"v2"`.
        assert_eq!(&bytes[5..10], &[2, b'v', b'2', 0, 0]);
        bytes.drain(8..10);
        bytes[4] = 2;
        let m2 = decode(&bytes, &mut interner).expect("v2 decodes");
        assert_eq!(m2.function(FuncId::from_index(0)).attrs, FuncAttrs::DEFAULT);
    }

    /// A version-1 stream (no per-global attribute byte) still decodes; its
    /// globals get the default attributes.
    #[test]
    fn version_1_stream_decodes_with_default_attrs() {
        let mut interner = StrInterner::new();
        let mut m = Module::new("v1");
        let i32t = m.types_mut().int(32);
        let c = m.intern_const(Const::Int { ty: i32t, value: Int::from_i64(9) });
        let g = Global { name: interner.intern("g"), ty: i32t, init: Some(c) };
        m.define_global(g, GlobalAttrs::DEFAULT);
        let mut bytes = encode(&m, &interner);
        // With no functions the stream ends `<attrs byte> <nfuncs = 0>`; drop the
        // attribute byte, and the v4 header's `<no target> <empty layout>` after
        // the name `"v1"`, and relabel the version to reconstruct the v1 layout.
        assert_eq!(bytes[4], VERSION as u8);
        let n = bytes.len();
        assert_eq!(&bytes[n - 2..], &[0, 0]);
        bytes.remove(n - 2);
        assert_eq!(&bytes[5..10], &[2, b'v', b'1', 0, 0]);
        bytes.drain(8..10);
        bytes[4] = 1;
        let m2 = decode(&bytes, &mut interner).expect("v1 decodes");
        assert_eq!(m2.global_count(), 1);
        let g2 = m2.global(GlobalId::from_index(0));
        assert_eq!(m2.global_attrs(GlobalId::from_index(0)), GlobalAttrs::DEFAULT);
        let init = m2.consts().get(g2.init.expect("initializer kept"));
        assert_eq!(init, &Const::Int { ty: g2.ty, value: Int::from_i64(9) });
    }

    /// The target name, the data layout, `ptr addrspace(N)` types and a
    /// global's address space survive an encode/decode round trip.
    #[test]
    fn target_datalayout_and_addrspaces_round_trip() {
        let mut interner = StrInterner::new();
        let mut m = Module::new("avr");
        m.set_target(Some("avr".to_owned()));
        let dl = crate::ir::DataLayout::parse("e-p:16:8-p1:16:8-i16:8-i32:8-i64:8-S8-n8-P1").unwrap();
        m.set_data_layout(dl.clone());
        let i8t = m.types_mut().int(8);
        let arr = m.types_mut().array(i8t, 2);
        let flash = m.types_mut().ptr_in(1);
        let c1 = m.intern_const(Const::Int { ty: i8t, value: Int::from_i64(7) });
        let init = m.intern_const(Const::Aggregate { ty: arr, elems: vec![c1, c1] });
        let g = m.define_global(
            Global { name: interner.intern("tbl"), ty: arr, init: Some(init) },
            GlobalAttrs { constant: true, ..GlobalAttrs::DEFAULT },
        );
        m.set_global_addr_space(g, 1);
        let sig = m.types_mut().func(vec![flash], i8t, false);
        let f = m.declare_function(interner.intern("rd"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let p = b.param(e, 0);
            let v = b.load(i8t, p, 1);
            b.ret(Some(v));
        }
        let bytes = encode(&m, &interner);
        let m2 = decode(&bytes, &mut interner).expect("decode");
        assert_eq!(m2.target(), Some("avr"));
        assert_eq!(m2.data_layout(), &dl);
        assert_eq!(m2.global_addr_space(GlobalId::from_index(0)), 1);
        let f2 = m2.function(crate::ir::FuncId::from_index(0));
        let crate::ir::Type::Func(ft) = m2.types().get(f2.sig) else { panic!("signature") };
        assert_eq!(m2.types().get(ft.params[0]), &crate::ir::Type::PtrIn(1));
        assert_eq!(encode(&m2, &interner), bytes, "re-encoding is byte-identical");
    }

    /// Version 5: secrecy rides in the attribute-byte extensions (bit 7), next
    /// to linkage, visibility and the address space; a secret-free module
    /// encodes exactly as version 4 did (bar the version), and a v4 stream
    /// still decodes; unknown extension bits and bit 7 in a v4 stream are
    /// rejected.
    #[test]
    fn version_5_attribute_extensions() {
        let mut interner = StrInterner::new();
        let mut m = Module::new("v5");
        let i64t = m.types_mut().int(64);
        let c = m.intern_const(Const::Int { ty: i64t, value: Int::from_i64(7) });
        let gattrs = GlobalAttrs {
            linkage: Linkage::Internal,
            visibility: Visibility::Hidden,
            constant: true,
            secret: true,
            ..GlobalAttrs::DEFAULT
        };
        let g = m.define_global(Global { name: interner.intern("k"), ty: i64t, init: Some(c) }, gattrs);
        m.set_global_addr_space(g, 3);
        let sig = m.types_mut().func(vec![i64t, i64t, i64t], i64t, false);
        let f = m.declare_function(interner.intern("f"), sig);
        let mut fattrs = FuncAttrs::new(Linkage::Weak, Visibility::Protected);
        fattrs.set_param_secret(0, true);
        fattrs.set_param_secret(2, true);
        fattrs.secret_ret = true;
        m.set_func_attrs(f, fattrs.clone());
        let bytes = encode(&m, &interner);
        let m2 = decode(&bytes, &mut interner).expect("decode");
        assert_eq!(m2.global_attrs(GlobalId::from_index(0)), gattrs);
        assert_eq!(m2.global_addr_space(GlobalId::from_index(0)), 3);
        assert_eq!(m2.function(FuncId::from_index(0)).attrs, fattrs);
        assert_eq!(encode(&m2, &interner), bytes);

        // The function attribute byte carries bit 7, then the extension
        // (secret return | parameter list), the count and the indices; the
        // stream ends with the (empty) function's body header.
        let fbyte = super::func_attrs_bits(&fattrs) | super::ATTR_EXT_BIT;
        let tail = [fbyte, 3, 2, 0, 2];
        let at = bytes.windows(tail.len()).rposition(|w| w == tail).expect("function attributes");
        let mut bad = bytes.clone();
        bad[at + 1] = 4; // an unknown function extension bit
        assert!(decode(&bad, &mut interner).is_err());
        let mut bad = bytes.clone();
        bad[at + 4] = 3; // a secret parameter index past the arity
        assert!(decode(&bad, &mut interner).is_err());
        // The global: attribute byte with bits 6 and 7, space 3, extension 1.
        let gbyte = super::attrs_bits(gattrs) | super::ADDR_SPACE_BIT | super::ATTR_EXT_BIT;
        let at = bytes.windows(3).position(|w| w == [gbyte, 3, 1]).expect("global attributes");
        let mut bad = bytes.clone();
        bad[at + 2] = 4; // an unknown global extension bit
        assert!(decode(&bad, &mut interner).is_err());

        // Without secrets, v5 is v4 with a new version number; v4 still decodes.
        m.set_func_attrs(f, FuncAttrs::new(Linkage::Weak, Visibility::Protected));
        m.set_global_attrs(g, GlobalAttrs { secret: false, ..gattrs });
        let mut v4 = encode(&m, &interner);
        assert!(!v4.contains(&fbyte));
        v4[4] = 4;
        let m4 = decode(&v4, &mut interner).expect("v4 decodes");
        assert!(!m4.has_secrets());
        assert_eq!(m4.function(FuncId::from_index(0)).attrs.linkage, Linkage::Weak);
        // In a v4 stream bit 7 is not an extension flag: it is rejected.
        let mut bad = v4.clone();
        let at = bad.iter().rposition(|&b| b == super::func_attrs_bits(&fattrs)).expect("byte");
        bad[at] |= super::ATTR_EXT_BIT;
        assert!(decode(&bad, &mut interner).is_err());
    }

    /// `thread_local` is global extension bit 1: it round-trips (alone and with
    /// `secret`), and a module without thread-locals keeps its bytes.
    #[test]
    fn thread_local_extension_bit() {
        let mut interner = StrInterner::new();
        let mut m = Module::new("tls");
        let i32t = m.types_mut().int(32);
        let c = m.intern_const(Const::Int { ty: i32t, value: Int::from_i64(5) });
        let plain = encode(&m, &interner);
        let tl = GlobalAttrs { thread_local: true, ..GlobalAttrs::DEFAULT };
        let g = m.define_global(Global { name: interner.intern("t"), ty: i32t, init: Some(c) }, tl);
        let bytes = encode(&m, &interner);
        assert_ne!(plain, bytes);
        let m2 = decode(&bytes, &mut interner).expect("decode");
        assert_eq!(m2.global_attrs(GlobalId::from_index(0)), tl);
        assert_eq!(encode(&m2, &interner), bytes);
        let gbyte = super::attrs_bits(tl) | super::ATTR_EXT_BIT;
        assert!(bytes.windows(2).any(|w| w == [gbyte, 2]), "attribute byte + extension 2");
        let both = GlobalAttrs { secret: true, ..tl };
        m.set_global_attrs(g, both);
        let m3 = decode(&encode(&m, &interner), &mut interner).expect("decode");
        assert_eq!(m3.global_attrs(GlobalId::from_index(0)), both);
        // Without the attribute the global's byte has no extension.
        m.set_global_attrs(g, GlobalAttrs::DEFAULT);
        let bytes = encode(&m, &interner);
        assert!(!bytes.contains(&(super::attrs_bits(GlobalAttrs::DEFAULT) | super::ATTR_EXT_BIT)));
    }

    /// Version-2 and version-3 streams (no target/layout header, no
    /// address-space bit) still decode, as LP64 modules with no target; a v3
    /// stream keeps its visibility and function attributes exactly.
    #[test]
    fn version_2_and_3_streams_decode_as_lp64() {
        let mut interner = StrInterner::new();
        let mut m = Module::new("v3");
        let i32t = m.types_mut().int(32);
        let c = m.intern_const(Const::Int { ty: i32t, value: Int::from_i64(1) });
        let hidden = GlobalAttrs { visibility: Visibility::Hidden, constant: true, ..GlobalAttrs::DEFAULT };
        m.define_global(Global { name: interner.intern("g"), ty: i32t, init: Some(c) }, hidden);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(interner.intern("f"), sig);
        let fattrs = FuncAttrs::new(Linkage::Weak, Visibility::Protected);
        m.set_func_attrs(f, fattrs.clone());
        let v4 = encode(&m, &interner);
        // A master-v3 stream is the v4 stream without `<no target> <empty layout>`.
        assert_eq!(&v4[5..10], &[2, b'v', b'3', 0, 0]);
        let mut v3 = v4.clone();
        v3.drain(8..10);
        v3[4] = 3;
        let m3 = decode(&v3, &mut interner).expect("v3 decodes");
        assert_eq!(m3.target(), None);
        assert_eq!(m3.data_layout(), &crate::ir::DataLayout::lp64());
        assert_eq!(m3.global_addr_space(GlobalId::from_index(0)), 0);
        assert_eq!(m3.global_attrs(GlobalId::from_index(0)), hidden);
        assert_eq!(m3.function(FuncId::from_index(0)).attrs, fattrs);
        // In a v3 stream bit 6 of the global attribute byte is not an address
        // space flag: it is rejected like any other unknown bit.
        let gattr = v3.iter().rposition(|&b| b == super::attrs_bits(hidden)).expect("attribute byte");
        let mut bad = v3.clone();
        bad[gattr] |= super::ADDR_SPACE_BIT;
        assert!(decode(&bad, &mut interner).is_err());
        // A v2 stream of a module without visibility or function attributes.
        let mut m = Module::new("v2");
        let i32t = m.types_mut().int(32);
        m.define_global(Global { name: interner.intern("h"), ty: i32t, init: None }, GlobalAttrs::DEFAULT);
        let mut v2 = encode(&m, &interner);
        v2.drain(8..10);
        v2[4] = 2;
        let m2 = decode(&v2, &mut interner).expect("v2 decodes");
        assert_eq!(m2.data_layout(), &crate::ir::DataLayout::lp64());
        assert_eq!(m2.global_addr_space(GlobalId::from_index(0)), 0);
    }

    /// A malformed data-layout spec in the header is a clean decode error.
    #[test]
    fn bad_datalayout_is_rejected() {
        let mut interner = StrInterner::new();
        let mut m = Module::new("x");
        m.set_data_layout(crate::ir::DataLayout::ilp32());
        let mut bytes = encode(&m, &interner);
        // Corrupt the first byte of the spec string (`e`) into an unknown item.
        let spec_at = 5 + 2 + 1 + 1; // version, name, no-target byte, spec length
        assert_eq!(bytes[spec_at], b'e');
        bytes[spec_at] = b'z';
        assert_eq!(decode(&bytes, &mut interner).unwrap_err(), DecodeError::InvalidDataLayout);
    }

    /// An address constant naming a nonexistent global is a decode error.
    #[test]
    fn address_constant_out_of_range_is_rejected() {
        let mut interner = StrInterner::new();
        let mut m = Module::new("bad");
        let ptr = m.types_mut().ptr();
        let target = AddrTarget::Global(GlobalId::from_index(5));
        let c = m.intern_const(Const::Addr { ty: ptr, target, offset: 0 });
        let g = Global { name: interner.intern("p"), ty: ptr, init: Some(c) };
        m.define_global(g, GlobalAttrs::DEFAULT);
        let bytes = encode(&m, &interner);
        assert!(matches!(
            decode(&bytes, &mut interner),
            Err(DecodeError::IndexOutOfRange { what: "global", index: 5 })
        ));
    }
}
