//! The WebAssembly decoder, from the Core Specification's binary format
//! (§5.4, "Instructions") and the proposals LatticeFoundry or common
//! toolchains emit.
//!
//! [`decode_inst`] turns the bytes of one instruction into a typed
//! [`WasmInst`] (its opcode, its immediates, and the byte width of each LEB
//! immediate, so it can be re-encoded exactly — `.o` files pad patchable
//! indices to five bytes); [`decode`] renders it in the text format's names,
//! spelled the way `llvm-objdump` prints wasm (a memory argument as
//! `offset` or `offset:p2align=A` when the alignment is not natural, a
//! `br_table` as `{l1, l2, default}`, floats in hexadecimal).
//!
//! Covered: the MVP (control, parametric, variable, table, memory and every
//! numeric instruction), sign extension, non-trapping float-to-int
//! conversion (`0xFC 0–7`), bulk memory and table instructions (`0xFC
//! 8–17`), reference types, typed `select`, tail calls, legacy exception
//! handling (`try`/`catch`/`throw`/`rethrow`/`delegate`/`catch_all`,
//! `throw_ref`), threads (`0xFE`: notify, waits, fence and every atomic
//! load, store, read-modify-write and compare-exchange) and fixed-width SIMD
//! (`0xFD 0–255`). Relaxed SIMD, GC, `try_table` and function references
//! decode as data (`.byte`).
//!
//! Wasm control flow is structured (branches name enclosing blocks, not
//! addresses), so no instruction has a [`target`](Inst::target).

use std::borrow::Cow;

use super::Inst;
use crate::mc::disasm::objfile::wasm_valtype;
use crate::target::wasm32::leb;

/// A block type (`block`, `loop`, `if`, `try`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BlockType {
    /// `0x40`: no parameters or results.
    Empty,
    /// One result of this value type (its type byte).
    Value(u8),
    /// A type index (a signed 33-bit LEB, non-negative).
    Index(u64),
}

/// An instruction's immediates.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Imm {
    /// None.
    None,
    /// A block type.
    Block(BlockType),
    /// One index (label, function, local, global, table, memory, data,
    /// element segment or tag).
    Index(u32),
    /// Two indices, in encoding order (`call_indirect` type and table;
    /// `memory.init` data and memory; `memory.copy` and `table.copy`
    /// destination and source; `table.init` element and table).
    Index2(u32, u32),
    /// `br_table`: the labels, then the default.
    BrTable {
        /// The label vector.
        targets: Vec<u32>,
        /// The default label.
        default: u32,
    },
    /// `i32.const`.
    I32(i32),
    /// `i64.const`.
    I64(i64),
    /// `f32.const`, as bits.
    F32(u32),
    /// `f64.const`, as bits.
    F64(u64),
    /// A memory argument: the alignment exponent and the offset.
    Mem {
        /// log2 of the alignment.
        align: u32,
        /// The static offset.
        offset: u64,
    },
    /// A memory argument and a lane (`v128.load8_lane`, ...).
    MemLane {
        /// log2 of the alignment.
        align: u32,
        /// The static offset.
        offset: u64,
        /// The lane index.
        lane: u8,
    },
    /// A lane index (`extract_lane`, `replace_lane`).
    Lane(u8),
    /// `v128.const`: sixteen bytes.
    V128([u8; 16]),
    /// `i8x16.shuffle`: sixteen lane indices.
    Shuffle([u8; 16]),
    /// `ref.null`: the reference type byte.
    RefType(u8),
    /// Typed `select`: the value types.
    Types(Vec<u8>),
    /// `atomic.fence`'s reserved zero byte.
    Fence(u8),
}

/// One decoded wasm instruction.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct WasmInst {
    /// The prefix byte (`0xFC`, `0xFD`, `0xFE`), if any.
    pub prefix: Option<u8>,
    /// The opcode byte, or the sub-opcode after a prefix.
    pub opcode: u32,
    /// The immediates.
    pub imm: Imm,
    /// The byte width of each LEB in the encoding, in order (a prefix's
    /// sub-opcode first), for an exact re-encoding.
    pub widths: Vec<u8>,
}

impl WasmInst {
    /// The instruction's text-format name.
    pub fn name(&self) -> Cow<'static, str> {
        info(self.prefix, self.opcode).map_or(Cow::Borrowed("?"), |(n, _)| n)
    }

    /// The opcode as one number: the byte, or `prefix << 8 | sub-opcode`.
    pub fn code(&self) -> u32 {
        match self.prefix {
            Some(p) => u32::from(p) << 8 | self.opcode,
            None => self.opcode,
        }
    }

    /// The operands, rendered as llvm-objdump does.
    pub fn operands(&self) -> Vec<String> {
        let natural = match info(self.prefix, self.opcode) {
            Some((_, Kind::Mem(n) | Kind::MemLane(n))) => n,
            _ => 0,
        };
        let memarg = |align: u32, offset: u64| {
            if align == natural { offset.to_string() } else { format!("{offset}:p2align={align}") }
        };
        match &self.imm {
            Imm::None | Imm::Fence(_) => Vec::new(),
            Imm::Block(BlockType::Empty) => Vec::new(),
            Imm::Block(BlockType::Value(t)) => vec![wasm_valtype(*t).to_owned()],
            Imm::Block(BlockType::Index(i)) => vec![format!("type[{i}]")],
            Imm::Index(i) => vec![i.to_string()],
            Imm::Index2(a, b) => {
                if self.prefix.is_none() && matches!(self.opcode, 0x11 | 0x13) && *b == 0 {
                    vec![a.to_string()] // `call_indirect` of table 0
                } else {
                    vec![a.to_string(), b.to_string()]
                }
            }
            Imm::BrTable { targets, default } => {
                let all: Vec<String> = targets.iter().chain(std::iter::once(default)).map(u32::to_string).collect();
                vec![format!("{{{}}}", all.join(", "))]
            }
            Imm::I32(v) => vec![v.to_string()],
            Imm::I64(v) => vec![v.to_string()],
            Imm::F32(b) => vec![float_text(f32_bits_as_f64(*b))],
            Imm::F64(b) => vec![float_text(*b)],
            Imm::Mem { align, offset } => vec![memarg(*align, *offset)],
            Imm::MemLane { align, offset, lane } => vec![memarg(*align, *offset), lane.to_string()],
            Imm::Lane(l) => vec![l.to_string()],
            Imm::V128(b) => b
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]).to_string())
                .collect(),
            Imm::Shuffle(l) => l.iter().map(u8::to_string).collect(),
            Imm::RefType(_) => Vec::new(),
            Imm::Types(ts) => ts.iter().map(|&t| wasm_valtype(t).to_owned()).collect(),
        }
    }

    /// The mnemonic: the [name](WasmInst::name), except that `ref.null`
    /// spells its type into it (`ref.null_func`).
    pub fn mnemonic(&self) -> String {
        match self.imm {
            Imm::RefType(t) => match t {
                0x70 => "ref.null_func".to_owned(),
                0x6f => "ref.null_extern".to_owned(),
                0x69 => "ref.null_exn".to_owned(),
                other => format!("ref.null_{other:#x}"),
            },
            _ => self.name().into_owned(),
        }
    }
}

/// How an opcode's immediates are encoded.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind {
    /// None.
    None,
    /// A block type.
    Block,
    /// One unsigned LEB index.
    Index,
    /// Two unsigned LEB indices.
    Index2,
    /// A `br_table` vector.
    BrTable,
    /// A signed LEB `i32`.
    I32,
    /// A signed LEB `i64`.
    I64,
    /// Four bytes.
    F32,
    /// Eight bytes.
    F64,
    /// A memory argument; the natural alignment exponent.
    Mem(u32),
    /// A memory argument and a lane byte; the natural alignment exponent.
    MemLane(u32),
    /// A lane byte.
    Lane,
    /// Sixteen bytes.
    V128,
    /// Sixteen lane bytes.
    Shuffle,
    /// A reference type byte.
    RefType,
    /// A vector of value types.
    Types,
    /// One reserved byte (`atomic.fence`).
    Fence,
}

/// The name and immediate kind of an opcode (`prefix` `None` for a
/// one-byte opcode), or `None` for an encoding this decoder does not know.
pub fn info(prefix: Option<u8>, op: u32) -> Option<(Cow<'static, str>, Kind)> {
    match prefix {
        None => {
            let op = u8::try_from(op).ok()?;
            plain(op).map(|(n, k)| (Cow::Borrowed(n), k))
        }
        Some(0xfc) => misc(op).map(|(n, k)| (Cow::Borrowed(n), k)),
        Some(0xfd) => simd(op),
        Some(0xfe) => atomic(op),
        Some(_) => None,
    }
}

/// The one-byte opcodes.
fn plain(op: u8) -> Option<(&'static str, Kind)> {
    use Kind::*;
    const I32_CMP: [&str; 11] =
        ["i32.eqz", "i32.eq", "i32.ne", "i32.lt_s", "i32.lt_u", "i32.gt_s", "i32.gt_u", "i32.le_s", "i32.le_u", "i32.ge_s", "i32.ge_u"];
    const I64_CMP: [&str; 11] =
        ["i64.eqz", "i64.eq", "i64.ne", "i64.lt_s", "i64.lt_u", "i64.gt_s", "i64.gt_u", "i64.le_s", "i64.le_u", "i64.ge_s", "i64.ge_u"];
    const F32_CMP: [&str; 6] = ["f32.eq", "f32.ne", "f32.lt", "f32.gt", "f32.le", "f32.ge"];
    const F64_CMP: [&str; 6] = ["f64.eq", "f64.ne", "f64.lt", "f64.gt", "f64.le", "f64.ge"];
    const I32_ARITH: [&str; 18] = [
        "i32.clz", "i32.ctz", "i32.popcnt", "i32.add", "i32.sub", "i32.mul", "i32.div_s", "i32.div_u", "i32.rem_s",
        "i32.rem_u", "i32.and", "i32.or", "i32.xor", "i32.shl", "i32.shr_s", "i32.shr_u", "i32.rotl", "i32.rotr",
    ];
    const I64_ARITH: [&str; 18] = [
        "i64.clz", "i64.ctz", "i64.popcnt", "i64.add", "i64.sub", "i64.mul", "i64.div_s", "i64.div_u", "i64.rem_s",
        "i64.rem_u", "i64.and", "i64.or", "i64.xor", "i64.shl", "i64.shr_s", "i64.shr_u", "i64.rotl", "i64.rotr",
    ];
    const F32_ARITH: [&str; 14] = [
        "f32.abs", "f32.neg", "f32.ceil", "f32.floor", "f32.trunc", "f32.nearest", "f32.sqrt", "f32.add", "f32.sub",
        "f32.mul", "f32.div", "f32.min", "f32.max", "f32.copysign",
    ];
    const F64_ARITH: [&str; 14] = [
        "f64.abs", "f64.neg", "f64.ceil", "f64.floor", "f64.trunc", "f64.nearest", "f64.sqrt", "f64.add", "f64.sub",
        "f64.mul", "f64.div", "f64.min", "f64.max", "f64.copysign",
    ];
    const CONV: [&str; 30] = [
        "i32.wrap_i64",
        "i32.trunc_f32_s",
        "i32.trunc_f32_u",
        "i32.trunc_f64_s",
        "i32.trunc_f64_u",
        "i64.extend_i32_s",
        "i64.extend_i32_u",
        "i64.trunc_f32_s",
        "i64.trunc_f32_u",
        "i64.trunc_f64_s",
        "i64.trunc_f64_u",
        "f32.convert_i32_s",
        "f32.convert_i32_u",
        "f32.convert_i64_s",
        "f32.convert_i64_u",
        "f32.demote_f64",
        "f64.convert_i32_s",
        "f64.convert_i32_u",
        "f64.convert_i64_s",
        "f64.convert_i64_u",
        "f64.promote_f32",
        "i32.reinterpret_f32",
        "i64.reinterpret_f64",
        "f32.reinterpret_i32",
        "f64.reinterpret_i64",
        "i32.extend8_s",
        "i32.extend16_s",
        "i64.extend8_s",
        "i64.extend16_s",
        "i64.extend32_s",
    ];
    Some(match op {
        0x00 => ("unreachable", None),
        0x01 => ("nop", None),
        0x02 => ("block", Block),
        0x03 => ("loop", Block),
        0x04 => ("if", Block),
        0x05 => ("else", None),
        0x06 => ("try", Block),
        0x07 => ("catch", Index),
        0x08 => ("throw", Index),
        0x09 => ("rethrow", Index),
        0x0a => ("throw_ref", None),
        0x0b => ("end", None),
        0x0c => ("br", Index),
        0x0d => ("br_if", Index),
        0x0e => ("br_table", BrTable),
        0x0f => ("return", None),
        0x10 => ("call", Index),
        0x11 => ("call_indirect", Index2),
        0x12 => ("return_call", Index),
        0x13 => ("return_call_indirect", Index2),
        0x18 => ("delegate", Index),
        0x19 => ("catch_all", None),
        0x1a => ("drop", None),
        0x1b => ("select", None),
        0x1c => ("select", Types),
        0x20 => ("local.get", Index),
        0x21 => ("local.set", Index),
        0x22 => ("local.tee", Index),
        0x23 => ("global.get", Index),
        0x24 => ("global.set", Index),
        0x25 => ("table.get", Index),
        0x26 => ("table.set", Index),
        0x28 => ("i32.load", Mem(2)),
        0x29 => ("i64.load", Mem(3)),
        0x2a => ("f32.load", Mem(2)),
        0x2b => ("f64.load", Mem(3)),
        0x2c => ("i32.load8_s", Mem(0)),
        0x2d => ("i32.load8_u", Mem(0)),
        0x2e => ("i32.load16_s", Mem(1)),
        0x2f => ("i32.load16_u", Mem(1)),
        0x30 => ("i64.load8_s", Mem(0)),
        0x31 => ("i64.load8_u", Mem(0)),
        0x32 => ("i64.load16_s", Mem(1)),
        0x33 => ("i64.load16_u", Mem(1)),
        0x34 => ("i64.load32_s", Mem(2)),
        0x35 => ("i64.load32_u", Mem(2)),
        0x36 => ("i32.store", Mem(2)),
        0x37 => ("i64.store", Mem(3)),
        0x38 => ("f32.store", Mem(2)),
        0x39 => ("f64.store", Mem(3)),
        0x3a => ("i32.store8", Mem(0)),
        0x3b => ("i32.store16", Mem(1)),
        0x3c => ("i64.store8", Mem(0)),
        0x3d => ("i64.store16", Mem(1)),
        0x3e => ("i64.store32", Mem(2)),
        0x3f => ("memory.size", Index),
        0x40 => ("memory.grow", Index),
        0x41 => ("i32.const", I32),
        0x42 => ("i64.const", I64),
        0x43 => ("f32.const", F32),
        0x44 => ("f64.const", F64),
        0x45..=0x4f => (I32_CMP[usize::from(op - 0x45)], None),
        0x50..=0x5a => (I64_CMP[usize::from(op - 0x50)], None),
        0x5b..=0x60 => (F32_CMP[usize::from(op - 0x5b)], None),
        0x61..=0x66 => (F64_CMP[usize::from(op - 0x61)], None),
        0x67..=0x78 => (I32_ARITH[usize::from(op - 0x67)], None),
        0x79..=0x8a => (I64_ARITH[usize::from(op - 0x79)], None),
        0x8b..=0x98 => (F32_ARITH[usize::from(op - 0x8b)], None),
        0x99..=0xa6 => (F64_ARITH[usize::from(op - 0x99)], None),
        0xa7..=0xc4 => (CONV[usize::from(op - 0xa7)], None),
        0xd0 => ("ref.null", RefType),
        0xd1 => ("ref.is_null", None),
        0xd2 => ("ref.func", Index),
        _ => return Option::None,
    })
}

/// The `0xFC` sub-opcodes.
fn misc(op: u32) -> Option<(&'static str, Kind)> {
    use Kind::*;
    Some(match op {
        0 => ("i32.trunc_sat_f32_s", None),
        1 => ("i32.trunc_sat_f32_u", None),
        2 => ("i32.trunc_sat_f64_s", None),
        3 => ("i32.trunc_sat_f64_u", None),
        4 => ("i64.trunc_sat_f32_s", None),
        5 => ("i64.trunc_sat_f32_u", None),
        6 => ("i64.trunc_sat_f64_s", None),
        7 => ("i64.trunc_sat_f64_u", None),
        8 => ("memory.init", Index2),
        9 => ("data.drop", Index),
        10 => ("memory.copy", Index2),
        11 => ("memory.fill", Index),
        12 => ("table.init", Index2),
        13 => ("elem.drop", Index),
        14 => ("table.copy", Index2),
        15 => ("table.grow", Index),
        16 => ("table.size", Index),
        17 => ("table.fill", Index),
        _ => return Option::None,
    })
}

/// The `0xFE` (threads) sub-opcodes.
fn atomic(op: u32) -> Option<(Cow<'static, str>, Kind)> {
    use Kind::*;
    let fixed: Option<(&'static str, Kind)> = match op {
        0x00 => Some(("memory.atomic.notify", Mem(2))),
        0x01 => Some(("memory.atomic.wait32", Mem(2))),
        0x02 => Some(("memory.atomic.wait64", Mem(3))),
        0x03 => Some(("atomic.fence", Fence)),
        0x10 => Some(("i32.atomic.load", Mem(2))),
        0x11 => Some(("i64.atomic.load", Mem(3))),
        0x12 => Some(("i32.atomic.load8_u", Mem(0))),
        0x13 => Some(("i32.atomic.load16_u", Mem(1))),
        0x14 => Some(("i64.atomic.load8_u", Mem(0))),
        0x15 => Some(("i64.atomic.load16_u", Mem(1))),
        0x16 => Some(("i64.atomic.load32_u", Mem(2))),
        0x17 => Some(("i32.atomic.store", Mem(2))),
        0x18 => Some(("i64.atomic.store", Mem(3))),
        0x19 => Some(("i32.atomic.store8", Mem(0))),
        0x1a => Some(("i32.atomic.store16", Mem(1))),
        0x1b => Some(("i64.atomic.store8", Mem(0))),
        0x1c => Some(("i64.atomic.store16", Mem(1))),
        0x1d => Some(("i64.atomic.store32", Mem(2))),
        _ => Option::None,
    };
    if let Some((n, k)) = fixed {
        return Some((Cow::Borrowed(n), k));
    }
    // The read-modify-write families: seven widths per operation.
    if !(0x1e..=0x4e).contains(&op) {
        return Option::None;
    }
    const OPS: [&str; 7] = ["add", "sub", "and", "or", "xor", "xchg", "cmpxchg"];
    let k = op - 0x1e;
    let rmw = OPS[(k / 7) as usize];
    let (name, natural) = match k % 7 {
        0 => (format!("i32.atomic.rmw.{rmw}"), 2),
        1 => (format!("i64.atomic.rmw.{rmw}"), 3),
        2 => (format!("i32.atomic.rmw8.{rmw}_u"), 0),
        3 => (format!("i32.atomic.rmw16.{rmw}_u"), 1),
        4 => (format!("i64.atomic.rmw8.{rmw}_u"), 0),
        5 => (format!("i64.atomic.rmw16.{rmw}_u"), 1),
        _ => (format!("i64.atomic.rmw32.{rmw}_u"), 2),
    };
    Some((Cow::Owned(name), Mem(natural)))
}

/// The fixed-width SIMD (`0xFD`) names, by sub-opcode; empty = reserved.
const SIMD: [&str; 256] = [
    // 0x00
    "v128.load", "v128.load8x8_s", "v128.load8x8_u", "v128.load16x4_s", "v128.load16x4_u", "v128.load32x2_s",
    "v128.load32x2_u", "v128.load8_splat", "v128.load16_splat", "v128.load32_splat", "v128.load64_splat",
    "v128.store", "v128.const", "i8x16.shuffle", "i8x16.swizzle", "i8x16.splat",
    // 0x10
    "i16x8.splat", "i32x4.splat", "i64x2.splat", "f32x4.splat", "f64x2.splat", "i8x16.extract_lane_s",
    "i8x16.extract_lane_u", "i8x16.replace_lane", "i16x8.extract_lane_s", "i16x8.extract_lane_u",
    "i16x8.replace_lane", "i32x4.extract_lane", "i32x4.replace_lane", "i64x2.extract_lane", "i64x2.replace_lane",
    "f32x4.extract_lane",
    // 0x20
    "f32x4.replace_lane", "f64x2.extract_lane", "f64x2.replace_lane", "i8x16.eq", "i8x16.ne", "i8x16.lt_s",
    "i8x16.lt_u", "i8x16.gt_s", "i8x16.gt_u", "i8x16.le_s", "i8x16.le_u", "i8x16.ge_s", "i8x16.ge_u", "i16x8.eq",
    "i16x8.ne", "i16x8.lt_s",
    // 0x30
    "i16x8.lt_u", "i16x8.gt_s", "i16x8.gt_u", "i16x8.le_s", "i16x8.le_u", "i16x8.ge_s", "i16x8.ge_u", "i32x4.eq",
    "i32x4.ne", "i32x4.lt_s", "i32x4.lt_u", "i32x4.gt_s", "i32x4.gt_u", "i32x4.le_s", "i32x4.le_u", "i32x4.ge_s",
    // 0x40
    "i32x4.ge_u", "f32x4.eq", "f32x4.ne", "f32x4.lt", "f32x4.gt", "f32x4.le", "f32x4.ge", "f64x2.eq", "f64x2.ne",
    "f64x2.lt", "f64x2.gt", "f64x2.le", "f64x2.ge", "v128.not", "v128.and", "v128.andnot",
    // 0x50
    "v128.or", "v128.xor", "v128.bitselect", "v128.any_true", "v128.load8_lane", "v128.load16_lane",
    "v128.load32_lane", "v128.load64_lane", "v128.store8_lane", "v128.store16_lane", "v128.store32_lane",
    "v128.store64_lane", "v128.load32_zero", "v128.load64_zero", "f32x4.demote_f64x2_zero",
    "f64x2.promote_low_f32x4",
    // 0x60
    "i8x16.abs", "i8x16.neg", "i8x16.popcnt", "i8x16.all_true", "i8x16.bitmask", "i8x16.narrow_i16x8_s",
    "i8x16.narrow_i16x8_u", "f32x4.ceil", "f32x4.floor", "f32x4.trunc", "f32x4.nearest", "i8x16.shl",
    "i8x16.shr_s", "i8x16.shr_u", "i8x16.add", "i8x16.add_sat_s",
    // 0x70
    "i8x16.add_sat_u", "i8x16.sub", "i8x16.sub_sat_s", "i8x16.sub_sat_u", "f64x2.ceil", "f64x2.floor",
    "i8x16.min_s", "i8x16.min_u", "i8x16.max_s", "i8x16.max_u", "f64x2.trunc", "i8x16.avgr_u",
    "i16x8.extadd_pairwise_i8x16_s", "i16x8.extadd_pairwise_i8x16_u", "i32x4.extadd_pairwise_i16x8_s",
    "i32x4.extadd_pairwise_i16x8_u",
    // 0x80
    "i16x8.abs", "i16x8.neg", "i16x8.q15mulr_sat_s", "i16x8.all_true", "i16x8.bitmask", "i16x8.narrow_i32x4_s",
    "i16x8.narrow_i32x4_u", "i16x8.extend_low_i8x16_s", "i16x8.extend_high_i8x16_s", "i16x8.extend_low_i8x16_u",
    "i16x8.extend_high_i8x16_u", "i16x8.shl", "i16x8.shr_s", "i16x8.shr_u", "i16x8.add", "i16x8.add_sat_s",
    // 0x90
    "i16x8.add_sat_u", "i16x8.sub", "i16x8.sub_sat_s", "i16x8.sub_sat_u", "f64x2.nearest", "i16x8.mul",
    "i16x8.min_s", "i16x8.min_u", "i16x8.max_s", "i16x8.max_u", "", "i16x8.avgr_u", "i16x8.extmul_low_i8x16_s",
    "i16x8.extmul_high_i8x16_s", "i16x8.extmul_low_i8x16_u", "i16x8.extmul_high_i8x16_u",
    // 0xa0
    "i32x4.abs", "i32x4.neg", "", "i32x4.all_true", "i32x4.bitmask", "", "", "i32x4.extend_low_i16x8_s",
    "i32x4.extend_high_i16x8_s", "i32x4.extend_low_i16x8_u", "i32x4.extend_high_i16x8_u", "i32x4.shl",
    "i32x4.shr_s", "i32x4.shr_u", "i32x4.add", "",
    // 0xb0
    "", "i32x4.sub", "", "", "", "i32x4.mul", "i32x4.min_s", "i32x4.min_u", "i32x4.max_s", "i32x4.max_u",
    "i32x4.dot_i16x8_s", "", "i32x4.extmul_low_i16x8_s", "i32x4.extmul_high_i16x8_s", "i32x4.extmul_low_i16x8_u",
    "i32x4.extmul_high_i16x8_u",
    // 0xc0
    "i64x2.abs", "i64x2.neg", "", "i64x2.all_true", "i64x2.bitmask", "", "", "i64x2.extend_low_i32x4_s",
    "i64x2.extend_high_i32x4_s", "i64x2.extend_low_i32x4_u", "i64x2.extend_high_i32x4_u", "i64x2.shl",
    "i64x2.shr_s", "i64x2.shr_u", "i64x2.add", "",
    // 0xd0
    "", "i64x2.sub", "", "", "", "i64x2.mul", "i64x2.eq", "i64x2.ne", "i64x2.lt_s", "i64x2.gt_s", "i64x2.le_s",
    "i64x2.ge_s", "i64x2.extmul_low_i32x4_s", "i64x2.extmul_high_i32x4_s", "i64x2.extmul_low_i32x4_u",
    "i64x2.extmul_high_i32x4_u",
    // 0xe0
    "f32x4.abs", "f32x4.neg", "", "f32x4.sqrt", "f32x4.add", "f32x4.sub", "f32x4.mul", "f32x4.div", "f32x4.min",
    "f32x4.max", "f32x4.pmin", "f32x4.pmax", "f64x2.abs", "f64x2.neg", "", "f64x2.sqrt",
    // 0xf0
    "f64x2.add", "f64x2.sub", "f64x2.mul", "f64x2.div", "f64x2.min", "f64x2.max", "f64x2.pmin", "f64x2.pmax",
    "i32x4.trunc_sat_f32x4_s", "i32x4.trunc_sat_f32x4_u", "f32x4.convert_i32x4_s", "f32x4.convert_i32x4_u",
    "i32x4.trunc_sat_f64x2_s_zero", "i32x4.trunc_sat_f64x2_u_zero", "f64x2.convert_low_i32x4_s",
    "f64x2.convert_low_i32x4_u",
];

/// The `0xFD` sub-opcodes.
fn simd(op: u32) -> Option<(Cow<'static, str>, Kind)> {
    use Kind::*;
    let name = *SIMD.get(op as usize)?;
    if name.is_empty() {
        return Option::None;
    }
    let kind = match op {
        0x00 | 0x0b => Mem(4),
        0x01..=0x06 => Mem(3),
        0x07 => Mem(0),
        0x08 => Mem(1),
        0x09 | 0x5c => Mem(2),
        0x0a | 0x5d => Mem(3),
        0x0c => V128,
        0x0d => Shuffle,
        0x15..=0x22 => Lane,
        0x54 | 0x58 => MemLane(0),
        0x55 | 0x59 => MemLane(1),
        0x56 | 0x5a => MemLane(2),
        0x57 | 0x5b => MemLane(3),
        _ => None,
    };
    Some((Cow::Borrowed(name), kind))
}

/// Decode the instruction at the start of `bytes`: the typed instruction and
/// its length, or `None` for an unknown opcode or truncated immediates.
pub fn decode_inst(bytes: &[u8]) -> Option<(WasmInst, usize)> {
    let first = *bytes.first()?;
    let mut at = 1;
    let mut widths = Vec::new();
    let uleb = |at: &mut usize, widths: &mut Vec<u8>| -> Option<u64> {
        let start = *at;
        let v = leb::read_u64(bytes, at)?;
        widths.push((*at - start) as u8);
        Some(v)
    };
    let (prefix, opcode) = match first {
        0xfc..=0xfe => {
            let sub = uleb(&mut at, &mut widths)?;
            (Some(first), u32::try_from(sub).ok()?)
        }
        _ => (None, u32::from(first)),
    };
    let (_, kind) = info(prefix, opcode)?;
    let u32_of = |v: u64| u32::try_from(v).ok();
    let byte = |at: &mut usize| -> Option<u8> {
        let b = *bytes.get(*at)?;
        *at += 1;
        Some(b)
    };
    let fixed = |at: &mut usize, n: usize| -> Option<&[u8]> {
        let s = bytes.get(*at..*at + n)?;
        *at += n;
        Some(s)
    };
    let imm = match kind {
        Kind::None => Imm::None,
        Kind::Block => {
            let b = *bytes.get(at)?;
            if b == 0x40 {
                at += 1;
                Imm::Block(BlockType::Empty)
            } else if matches!(b, 0x7f | 0x7e | 0x7d | 0x7c | 0x7b | 0x70 | 0x6f) {
                at += 1;
                Imm::Block(BlockType::Value(b))
            } else {
                let start = at;
                let v = leb::read_i64(bytes, &mut at)?;
                widths.push((at - start) as u8);
                if !(0..1 << 32).contains(&v) {
                    return None;
                }
                Imm::Block(BlockType::Index(v as u64))
            }
        }
        Kind::Index => Imm::Index(u32_of(uleb(&mut at, &mut widths)?)?),
        Kind::Index2 => {
            let a = u32_of(uleb(&mut at, &mut widths)?)?;
            let b = u32_of(uleb(&mut at, &mut widths)?)?;
            Imm::Index2(a, b)
        }
        Kind::BrTable => {
            let n = uleb(&mut at, &mut widths)?;
            // Each label takes at least a byte: a count beyond the input is
            // corrupt (and must not drive a huge allocation).
            if n > bytes.len() as u64 {
                return None;
            }
            let mut targets = Vec::with_capacity(n as usize);
            for _ in 0..n {
                targets.push(u32_of(uleb(&mut at, &mut widths)?)?);
            }
            let default = u32_of(uleb(&mut at, &mut widths)?)?;
            Imm::BrTable { targets, default }
        }
        Kind::I32 => {
            let start = at;
            let v = leb::read_i64(bytes, &mut at)?;
            widths.push((at - start) as u8);
            Imm::I32(i32::try_from(v).ok()?)
        }
        Kind::I64 => {
            let start = at;
            let v = leb::read_i64(bytes, &mut at)?;
            widths.push((at - start) as u8);
            Imm::I64(v)
        }
        Kind::F32 => {
            let s = fixed(&mut at, 4)?;
            Imm::F32(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        }
        Kind::F64 => {
            let s = fixed(&mut at, 8)?;
            let mut b = [0u8; 8];
            b.copy_from_slice(s);
            Imm::F64(u64::from_le_bytes(b))
        }
        Kind::Mem(_) => {
            let align = u32_of(uleb(&mut at, &mut widths)?)?;
            let offset = uleb(&mut at, &mut widths)?;
            Imm::Mem { align, offset }
        }
        Kind::MemLane(_) => {
            let align = u32_of(uleb(&mut at, &mut widths)?)?;
            let offset = uleb(&mut at, &mut widths)?;
            let lane = byte(&mut at)?;
            Imm::MemLane { align, offset, lane }
        }
        Kind::Lane => Imm::Lane(byte(&mut at)?),
        Kind::V128 | Kind::Shuffle => {
            let s = fixed(&mut at, 16)?;
            let mut b = [0u8; 16];
            b.copy_from_slice(s);
            if kind == Kind::V128 { Imm::V128(b) } else { Imm::Shuffle(b) }
        }
        Kind::RefType => Imm::RefType(byte(&mut at)?),
        Kind::Types => {
            let n = uleb(&mut at, &mut widths)?;
            if n > bytes.len() as u64 {
                return None;
            }
            let mut ts = Vec::new();
            for _ in 0..n {
                ts.push(byte(&mut at)?);
            }
            Imm::Types(ts)
        }
        Kind::Fence => Imm::Fence(byte(&mut at)?),
    };
    Some((WasmInst { prefix, opcode, imm, widths }, at))
}

/// Decode one WebAssembly instruction from the start of `bytes` (non-empty),
/// located at address `addr`.
pub fn decode(bytes: &[u8], _addr: u64) -> Inst {
    match decode_inst(bytes) {
        Some((w, len)) => Inst::new(len, w.mnemonic()).ops(w.operands()),
        None => Inst::data(bytes, 1, true),
    }
}

/// The `f64` bits of an `f32` given as bits, exact (a NaN keeps its payload
/// in the top of the wider significand), so both widths print alike.
fn f32_bits_as_f64(b: u32) -> u64 {
    let sign = u64::from(b >> 31) << 63;
    let exp = (b >> 23) & 0xff;
    let mant = u64::from(b & 0x7f_ffff);
    match exp {
        0xff => sign | 0x7ff << 52 | mant << 29,
        0 if mant == 0 => sign,
        0 => {
            // A subnormal f32 is a normal f64: normalize.
            let shift = mant.leading_zeros() - 40; // bring the top set bit to bit 23
            let m = (mant << shift) & 0x7f_ffff;
            let e = 1023 - 126 - u64::from(shift);
            sign | e << 52 | m << 29
        }
        _ => sign | (u64::from(exp) + 1023 - 127) << 52 | mant << 29,
    }
}

/// An `f64` (as bits) in the hexadecimal notation: `0x1.8p0`, `-0x0p0`,
/// `0x0.0000000000001p-1022`, `infinity`, `nan`, `nan:0x…` (a NaN whose
/// payload is not the canonical quiet one).
fn float_text(bits: u64) -> String {
    let sign = if bits >> 63 != 0 { "-" } else { "" };
    let exp = (bits >> 52) & 0x7ff;
    let mant = bits & 0x000f_ffff_ffff_ffff;
    if exp == 0x7ff {
        if mant == 0 {
            return format!("{sign}infinity");
        }
        if mant == 1 << 51 {
            return format!("{sign}nan");
        }
        return format!("{sign}nan:{mant:#x}");
    }
    if exp == 0 && mant == 0 {
        return format!("{sign}0x0p0");
    }
    let frac = format!("{mant:013x}");
    let frac = frac.trim_end_matches('0');
    let dot = if frac.is_empty() { String::new() } else { format!(".{frac}") };
    if exp == 0 {
        format!("{sign}0x0{dot}p-1022")
    } else {
        format!("{sign}0x1{dot}p{}", exp as i64 - 1023)
    }
}
