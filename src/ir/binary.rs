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
//! 1. the module name (a length-prefixed UTF-8 string);
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
    BinOp, CastOp, FastMath, Flags, FloatPred, InstData, InstId, InstKind, IntPred, SwitchCase,
    SwitchData, UnaryOp, Use,
};
use crate::ir::types::{FloatKind, FuncType, Type, TypeId};
use crate::ir::value::{Const, ConstId, FloatBits, Value, ValueDef, ValueId};
use crate::ir::{Block, BlockId, FuncId, Function, Global, GlobalId, Module};
use crate::support::hash::{DetHashMap, DetHashSet};
use crate::support::StrInterner;

use puremp::{Int, Nat, Sign};

/// Four-byte file signature: "LFB" followed by a NUL, identifying an `.lfb`.
pub const MAGIC: [u8; 4] = *b"LFB\0";

/// Format version. Bumped on any incompatible change to the byte layout; a
/// decoder refuses a version it does not recognize.
pub const VERSION: u32 = 1;

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

fn binop_code(op: BinOp) -> u8 {
    use BinOp::*;
    match op {
        Add => 0, Sub => 1, Mul => 2, UDiv => 3, SDiv => 4, URem => 5, SRem => 6, And => 7,
        Or => 8, Xor => 9, Shl => 10, LShr => 11, AShr => 12, FAdd => 13, FSub => 14, FMul => 15,
        FDiv => 16, FRem => 17,
    }
}

fn binop_from(c: u8) -> Result<BinOp, DecodeError> {
    use BinOp::*;
    Ok(match c {
        0 => Add, 1 => Sub, 2 => Mul, 3 => UDiv, 4 => SDiv, 5 => URem, 6 => SRem, 7 => And,
        8 => Or, 9 => Xor, 10 => Shl, 11 => LShr, 12 => AShr, 13 => FAdd, 14 => FSub, 15 => FMul,
        16 => FDiv, 17 => FRem,
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
                InstKind::Load { ty, .. } | InstKind::Store { ty, .. } => tstack.push(*ty),
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
                Type::Array(elem, _) => tstack.push(*elem),
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
    for g in globals {
        w.str(names.resolve(g.name));
        w.uvarint(tables.ty(g.ty));
        match g.init {
            Some(c) => {
                w.u8(1);
                w.uvarint(tables.konst(c));
            }
            None => w.u8(0),
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
    }
}

fn write_function(w: &mut Writer, f: &Function, names: &StrInterner, t: &Tables) {
    w.str(names.resolve(f.name));
    w.uvarint(t.ty(f.sig));

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
        InstKind::Load { ty, align } => {
            w.u8(6);
            w.uvarint(t.ty(*ty));
            w.uvarint(u64::from(*align));
        }
        InstKind::Store { ty, align } => {
            w.u8(7);
            w.uvarint(t.ty(*ty));
            w.uvarint(u64::from(*align));
        }
        InstKind::PtrAdd { inbounds } => {
            w.u8(8);
            w.u8(u8::from(*inbounds));
        }
        InstKind::Select => w.u8(9),
        InstKind::Freeze => w.u8(10),
        InstKind::Call => w.u8(11),
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
    if version != u64::from(VERSION) {
        return Err(DecodeError::UnsupportedVersion(version.try_into().unwrap_or(u32::MAX)));
    }

    let module_name = r.str()?.to_owned();
    let mut module = Module::new(module_name);

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
        module.add_global(Global { name, ty, init });
    }

    // --- functions ---
    let nfuncs = r.uindex()?;
    for _ in 0..nfuncs {
        let f = read_function(&mut r, names, &types, &consts, nglobals, nfuncs)?;
        module.functions.push(f);
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

fn read_function(
    r: &mut Reader<'_>,
    names: &mut StrInterner,
    types: &[TypeId],
    consts: &[ConstId],
    nglobals: usize,
    nfuncs: usize,
) -> Result<Function, DecodeError> {
    let name = names.intern(r.str()?);
    let sig = types[checked(r.uindex()?, types.len(), "type")?];

    let mut f = Function::new(name, sig);

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
        6 => {
            let ty = ty(r)?;
            let align = r.u32()?;
            InstKind::Load { ty, align }
        }
        7 => {
            let ty = ty(r)?;
            let align = r.u32()?;
            InstKind::Store { ty, align }
        }
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
    use super::{DecodeError, MAGIC, VERSION, decode, encode};
    use crate::ir::inst::{BinOp, CastOp, FastMath, Flags, FloatPred, IntPred};
    use crate::ir::types::FloatKind;
    use crate::ir::value::{Const, FloatBits};
    use crate::ir::{Global, Module};
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
}
