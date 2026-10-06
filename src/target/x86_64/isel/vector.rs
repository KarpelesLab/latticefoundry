//! SSE2 lowering of 128-bit vectors (`docs/ir-design.md` §6c), the x86-64
//! baseline — no SSE3/SSSE3/SSE4 instruction is used.
//!
//! ## Legal types and the mask convention
//!
//! [`Sse2Legality`] keeps these types whole in an xmm register:
//! `<16 x i8>`, `<8 x i16>`, `<4 x i32>`, `<2 x i64>`, `<4 x f32>`, `<2 x f64>`,
//! and the masks `<16 x i1>`, `<8 x i1>`, `<4 x i1>`, `<2 x i1>`. A mask
//! `<N x i1>` occupies the register as `N` lanes of `128 / N` bits each, every
//! lane all-ones (true) or all-zeros (false) — exactly what `pcmpeq`/`pcmpgt`/
//! `cmpps` produce for the compared lane width, and what `pand`/`pandn`/`por`
//! blend with. Every other vector type, and every op below that SSE2 cannot
//! do, is scalarized by the generic legalizer ([`crate::codegen::legalize`])
//! before instruction selection.
//!
//! ## Lowering table
//!
//! | IR | SSE2 |
//! |---|---|
//! | `add`/`sub` i8/i16/i32/i64 | `padd{b,w,d,q}` / `psub{b,w,d,q}` |
//! | `mul` i16 | `pmullw` |
//! | `mul` i32 | `pmuludq` ×2 + `pshufd` ×4 + `punpckldq` (no SSE4.1 `pmulld`) |
//! | `umin`/`umax` i8, `smin`/`smax` i16 | `pminub`/`pmaxub`, `pminsw`/`pmaxsw` |
//! | `sadd_sat`/`uadd_sat`/`ssub_sat`/`usub_sat` i8/i16 | `padds{b,w}`/`paddus{b,w}`/`psubs{b,w}`/`psubus{b,w}` |
//! | `and`/`or`/`xor` (any, incl. masks) | `pand` / `por` / `pxor` |
//! | `shl`/`lshr` i16/i32/i64, `ashr` i16/i32, by a uniform constant | `psll`/`psrl`/`psra` `imm8` |
//! | `fadd`/`fsub`/`fmul`/`fdiv` | `addps`/… / `addpd`/… |
//! | `fneg` | `xorps`/`xorpd` with a sign-bit constant |
//! | `icmp` i8/i16/i32 | `pcmpeq`/`pcmpgt` (swapped for `<`, `pxor` all-ones for the negations, a sign-bit `pxor` of both sides for unsigned) |
//! | `icmp eq`/`ne` i64 | `pcmpeqd` + `pshufd 0xB1` + `pand` |
//! | `fcmp` | `cmpps`/`cmppd` (swapped operands for `>`; `one` = `ord & neq`, `ueq` = `unord | eq`) |
//! | `select` (mask condition) | `pand` / `pandn` / `por` |
//! | `select` (`i1` condition) | the condition broadcast to a mask (`movd` + `pshufd`), then as above |
//! | `sitofp` i32→f32, `fptosi` f32→i32 | `cvtdq2ps`, `cvttps2dq` |
//! | `sext`/`zext` mask→int of the lane width, `trunc` back | a copy / `pand 1` / `pand 1` + `pcmpeq 1` (i64: `psllq 63` + `psrad 31` + `pshufd`) |
//! | `bitcast` between 128-bit types | a register copy |
//! | `extractelement` | `pshufd` + `movd`/`movq`, `pextrw` (+ `shr` for a byte) |
//! | `insertelement` | `pinsrw` (×2 for a dword, word merge for a byte), `movq` + `movsd`/`punpcklqdq`, `movsd`/`unpcklpd` for f64 |
//! | `shufflevector` 32-bit lanes | `pshufd` (one source) / `shufps` (lanes 0-1 from one, 2-3 from the other) |
//! | `shufflevector` 64-bit lanes | `pshufd` / `shufpd` |
//! | `splat` | `movd`/`movq` + `pshufd` (bytes/words replicated in a GPR first) |
//! | `load`/`store` | `movdqa` (align ≥ 16) / `movdqu` |
//! | constants | `pxor` (zero), `pcmpeqd` (all-ones), else `movq` ×2 + `punpcklqdq` |
//!
//! The other min/max/saturating ops are expanded by the legalizer into
//! compares and selects (vector ones stay vector code where those are legal).
//! Scalarized instead: integer division and remainder, `frem`, `mul` i8/i64,
//! variable or non-uniform shifts, byte shifts, `ashr` i64, ordered/unsigned
//! compares of i64, compares of masks, other casts, other shuffles (8/16-bit
//! lanes, lane-count changes, other two-source patterns), every `reduce`, and
//! loads/stores of masks (whose memory form is a byte per lane).
//!
//! Vectors cross calls in `xmm0..xmm7` and return in `xmm0` (the System V
//! `__m128` classes); a 16-byte stack argument takes a 16-aligned slot.

use crate::codegen::isel::Lower;
use crate::codegen::legalize::VectorLegality;
use crate::codegen::mir::{MachineInst, RegClass, VReg};
use crate::ir::inst::{BinOp, CastOp, FloatPred, InstData, InstKind, IntPred, UnaryOp};
use crate::ir::types::{TypeContext, TypeId};
use crate::ir::value::{ConstPool, ValueId};
use crate::ir::Function;

use crate::codegen::simd128::{Lanes, shape, uniform_const};

use super::{X86Op, X86_64Target, def_v, imm, use_v};

/// The SSE2 vector legality of the x86-64 backend (see the module docs).
#[derive(Clone, Copy, Debug, Default)]
pub struct Sse2Legality;

/// Whether a shuffle mask of `n`-lane operands, with `cw`-bit lanes, has a
/// direct SSE2 form: every 64-bit-lane mask, and for 32-bit lanes a
/// single-source mask or one taking result lanes 0-1 from one operand and 2-3
/// from the other.
fn shuffle_is_direct(mask: &[u32], n: u32, cw: u32) -> bool {
    if mask.len() != n as usize {
        return false;
    }
    match cw {
        64 => true,
        32 => {
            let src = |m: u32| m / 4;
            (src(mask[0]) == src(mask[1]) && src(mask[2]) == src(mask[3]))
                || mask.iter().all(|&m| src(m) == src(mask[0]))
        }
        _ => false,
    }
}

impl VectorLegality for Sse2Legality {
    /// Bulk memory (`docs/ir-design.md` §6k): up to four 16-byte (SSE) or
    /// smaller chunks inline, unaligned accesses being fine; anything longer
    /// or variable stays for `rep movsb` / `rep stosb`.
    fn bulk_memory(&self, _layout: &crate::ir::DataLayout) -> crate::codegen::legalize_mem::BulkMemoryLowering {
        crate::codegen::legalize_mem::BulkMemoryLowering {
            word: 8,
            vector16: true,
            unaligned: true,
            max_inline: 4,
            native: true,
            guard_zero: false,
        }
    }

    fn legal_type(&self, types: &TypeContext, ty: TypeId) -> bool {
        shape(types, ty).is_some()
    }

    fn legal_inst(&self, types: &TypeContext, consts: &ConstPool, func: &Function, inst: &InstData) -> bool {
        let ops = inst.operands();
        let rshape = shape(types, inst.ty);
        let oshape = |i: usize| ops.get(i).and_then(|&o| shape(types, func.value_type(o)));
        match &inst.kind {
            InstKind::Bin(op) => match rshape {
                Some((Lanes::Float(_), ..)) => matches!(op, BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv),
                Some((Lanes::Mask, ..)) => matches!(op, BinOp::And | BinOp::Or | BinOp::Xor),
                Some((Lanes::Int(w), ..)) => match op {
                    BinOp::Add | BinOp::Sub | BinOp::And | BinOp::Or | BinOp::Xor => true,
                    BinOp::Mul => w == 16 || w == 32,
                    BinOp::Shl | BinOp::LShr => w >= 16 && uniform_const(consts, func, ops[1], w).is_some(),
                    BinOp::AShr => (w == 16 || w == 32) && uniform_const(consts, func, ops[1], w).is_some(),
                    op if op.is_minmax_sat() => minmax_sat_opcode(*op, w).is_some(),
                    _ => false,
                },
                None => false,
            },
            InstKind::Unary(UnaryOp::FNeg) => true,
            InstKind::ICmp(pred) => match oshape(0) {
                Some((Lanes::Int(8 | 16 | 32), ..)) => true,
                Some((Lanes::Int(64), ..)) => matches!(pred, IntPred::Eq | IntPred::Ne),
                _ => false,
            },
            InstKind::FCmp(_) => true,
            InstKind::Cast(op) => {
                let (Some((fk, _, fcw)), Some((tk, _, tcw))) = (oshape(0), rshape) else {
                    return false;
                };
                match op {
                    CastOp::Bitcast => fk != Lanes::Mask && tk != Lanes::Mask,
                    CastOp::SiToFp => fk == Lanes::Int(32) && tk == Lanes::Float(32),
                    CastOp::FpToSi => fk == Lanes::Float(32) && tk == Lanes::Int(32),
                    CastOp::SExt | CastOp::ZExt => fk == Lanes::Mask && tk == Lanes::Int(fcw),
                    CastOp::Trunc => tk == Lanes::Mask && fk == Lanes::Int(tcw),
                    _ => false,
                }
            }
            InstKind::Select | InstKind::ExtractElement { .. } | InstKind::InsertElement { .. } | InstKind::Splat => {
                true
            }
            InstKind::ShuffleVector(mask) => match oshape(0) {
                Some((_, n, cw)) => shuffle_is_direct(mask, n, cw),
                None => false,
            },
            InstKind::Load { ty, .. } | InstKind::Store { ty, .. } => {
                !matches!(shape(types, *ty), Some((Lanes::Mask, ..)))
            }
            _ => false,
        }
    }
}

/// The packed immediate of [`X86Op::VOp`]/[`X86Op::VUnary`].
pub(crate) struct VEnc;

impl VEnc {
    /// `prefix 0F opcode`, optionally commutative.
    pub(crate) const fn op(prefix: u8, opcode: u8, commutative: bool) -> u64 {
        prefix as u64 | ((opcode as u64) << 8) | ((commutative as u64) << 16)
    }

    /// `prefix 0F opcode … imm8`.
    pub(crate) const fn op_imm(prefix: u8, opcode: u8, imm8: u8) -> u64 {
        prefix as u64 | ((opcode as u64) << 8) | (1 << 17) | ((imm8 as u64) << 24)
    }

    /// Decode `(prefix, opcode, commutative, imm8)`.
    pub(crate) fn decode(enc: u64) -> (u8, u8, bool, Option<u8>) {
        let imm = if enc & (1 << 17) != 0 { Some((enc >> 24) as u8) } else { None };
        (enc as u8, (enc >> 8) as u8, enc & (1 << 16) != 0, imm)
    }
}

// Opcodes (the byte after `0F`; all integer ones take the `66` prefix).
const PADD: [u8; 4] = [0xFC, 0xFD, 0xFE, 0xD4];
const PSUB: [u8; 4] = [0xF8, 0xF9, 0xFA, 0xFB];
const PCMPEQ: [u8; 3] = [0x74, 0x75, 0x76];
const PCMPGT: [u8; 3] = [0x64, 0x65, 0x66];
const PAND: u8 = 0xDB;
const PANDN: u8 = 0xDF;
const POR: u8 = 0xEB;
const PXOR: u8 = 0xEF;
const PMULLW: u8 = 0xD5;
const PMULUDQ: u8 = 0xF4;
const PUNPCKLDQ: u8 = 0x62;
const PUNPCKLQDQ: u8 = 0x6C;
const PSHUFD: u8 = 0x70;

/// The SSE2 opcode (and commutativity) of a min/max/saturating op on `w`-bit
/// lanes, where SSE2 has one: unsigned byte and signed word min/max, and
/// byte/word saturating add/subtract. The rest is expanded by the legalizer.
fn minmax_sat_opcode(op: BinOp, w: u32) -> Option<(u8, bool)> {
    Some(match (op, w) {
        (BinOp::UMin, 8) => (0xDA, true),      // pminub
        (BinOp::UMax, 8) => (0xDE, true),      // pmaxub
        (BinOp::SMin, 16) => (0xEA, true),     // pminsw
        (BinOp::SMax, 16) => (0xEE, true),     // pmaxsw
        (BinOp::SAddSat, 8) => (0xEC, true),   // paddsb
        (BinOp::SAddSat, 16) => (0xED, true),  // paddsw
        (BinOp::UAddSat, 8) => (0xDC, true),   // paddusb
        (BinOp::UAddSat, 16) => (0xDD, true),  // paddusw
        (BinOp::SSubSat, 8) => (0xE8, false),  // psubsb
        (BinOp::SSubSat, 16) => (0xE9, false), // psubsw
        (BinOp::USubSat, 8) => (0xD8, false),  // psubusb
        (BinOp::USubSat, 16) => (0xD9, false), // psubusw
        _ => return None,
    })
}

/// The index of an integer lane width in the `PADD`/`PSUB`/`PCMPEQ` tables.
fn widx(w: u32) -> usize {
    match w {
        8 => 0,
        16 => 1,
        32 => 2,
        _ => 3,
    }
}

/// `true` in every `w`-bit lane (`w` of 8/16/32/64), as two quadwords.
fn splat_bits(w: u32, x: u64) -> (u64, u64) {
    let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    let mut q = 0u64;
    let mut at = 0;
    while at < 64 {
        q |= (x & mask) << at;
        at += w;
    }
    (q, q)
}

impl X86_64Target {
    // --- small emitters (fresh vregs) ---------------------------------------

    fn vop(&self, lo: &mut Lower<'_, Self>, a: VReg, b: VReg, enc: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(X86Op::VOp.opcode(), vec![def_v(d), use_v(a), use_v(b), imm(enc)]));
        d
    }

    /// An integer-domain packed op (`66 0F op`).
    fn pop(&self, lo: &mut Lower<'_, Self>, a: VReg, b: VReg, opcode: u8, commutative: bool) -> VReg {
        self.vop(lo, a, b, VEnc::op(0x66, opcode, commutative))
    }

    fn vunary(&self, lo: &mut Lower<'_, Self>, s: VReg, enc: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(X86Op::VUnary.opcode(), vec![def_v(d), use_v(s), imm(enc)]));
        d
    }

    fn pshufd(&self, lo: &mut Lower<'_, Self>, s: VReg, imm8: u8) -> VReg {
        self.vunary(lo, s, VEnc::op_imm(0x66, PSHUFD, imm8))
    }

    fn vshift(&self, lo: &mut Lower<'_, Self>, a: VReg, opcode: u8, ext: u8, count: u8) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        let enc = u64::from(opcode) | (u64::from(ext) << 8) | (u64::from(count) << 16);
        lo.emit(MachineInst::new(X86Op::VShiftI.opcode(), vec![def_v(d), use_v(a), imm(enc)]));
        d
    }

    fn vconst(&self, lo: &mut Lower<'_, Self>, bits: (u64, u64)) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(X86Op::LoadVConst.opcode(), vec![def_v(d), imm(bits.0), imm(bits.1)]));
        d
    }

    fn vnot(&self, lo: &mut Lower<'_, Self>, x: VReg) -> VReg {
        let ones = self.vconst(lo, (u64::MAX, u64::MAX));
        self.pop(lo, x, ones, PXOR, true)
    }

    fn gpr_to_x(&self, lo: &mut Lower<'_, Self>, g: VReg, is64: bool) -> VReg {
        let x = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(X86Op::MovGprToX.opcode(), vec![def_v(x), use_v(g), imm(u64::from(is64))]));
        x
    }

    fn x_to_gpr(&self, lo: &mut Lower<'_, Self>, x: VReg, is64: bool) -> VReg {
        let g = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::MovXToGpr.opcode(), vec![def_v(g), use_v(x), imm(u64::from(is64))]));
        g
    }

    /// A GPR ALU op `d = a OP k` with an immediate `k` (materialized).
    fn gpr_imm(&self, lo: &mut Lower<'_, Self>, op: X86Op, a: VReg, k: u64, width: u32) -> VReg {
        let kr = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(kr), imm(k)]));
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(op.opcode(), vec![def_v(d), use_v(a), use_v(kr), imm(u64::from(width))]));
        d
    }

    fn gpr_bin(&self, lo: &mut Lower<'_, Self>, op: X86Op, a: VReg, b: VReg, width: u32) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(op.opcode(), vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))]));
        d
    }

    fn gpr_shift(&self, lo: &mut Lower<'_, Self>, op: X86Op, a: VReg, count: u64, width: u32) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(op.opcode(), vec![def_v(d), use_v(a), imm(count), imm(u64::from(width))]));
        d
    }

    /// `0` or `-1` (all bits) from an `i1` held in a GPR (upper bits garbage).
    fn bool_to_mask(&self, lo: &mut Lower<'_, Self>, g: VReg) -> VReg {
        let bit = self.gpr_imm(lo, X86Op::And, g, 1, 64);
        let zero = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(zero), imm(0)]));
        self.gpr_bin(lo, X86Op::Sub, zero, bit, 64)
    }

    /// Copy the computed value `r` into the instruction's result register.
    fn finish(&self, lo: &mut Lower<'_, Self>, inst: &InstData, r: VReg) {
        let d = lo.result_reg(inst);
        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(r)]));
    }

    // --- the dispatcher -----------------------------------------------------

    /// Lower a vector instruction; `false` if `inst` is not vector code (or is a
    /// whole-value op the scalar paths already handle: `call`, `freeze`).
    pub(super) fn lower_vector(&self, lo: &mut Lower<'_, Self>, inst: &InstData) -> bool {
        let types = lo.types();
        let func = lo.func();
        let touches = inst.result().is_some_and(|r| types.is_vector(func.value_type(r)))
            || inst.operands().iter().any(|&o| types.is_vector(func.value_type(o)))
            || matches!(&inst.kind, InstKind::Load { ty, .. } | InstKind::Store { ty, .. } if types.is_vector(*ty));
        if !touches || matches!(inst.kind, InstKind::Call | InstKind::Freeze) || inst.is_terminator() {
            return false;
        }
        let ops = inst.operands().to_vec();
        match &inst.kind {
            InstKind::Bin(op) => self.vec_bin(lo, inst, *op, &ops),
            InstKind::Unary(UnaryOp::FNeg) => {
                let (k, ..) = self.vshape(lo, inst.ty);
                let a = self.oper(lo, ops[0]);
                let (w, pfx) = if k == Lanes::Float(64) { (64, 0x66) } else { (32, 0x00) };
                let m = self.vconst(lo, splat_bits(w, 1u64 << (w - 1)));
                let r = self.vop(lo, a, m, VEnc::op(pfx, 0x57, true));
                self.finish(lo, inst, r);
            }
            InstKind::ICmp(pred) => self.vec_icmp(lo, inst, *pred, &ops),
            InstKind::FCmp(pred) => self.vec_fcmp(lo, inst, *pred, &ops),
            InstKind::Cast(op) => self.vec_cast(lo, inst, *op, &ops),
            InstKind::Select => self.vec_select(lo, inst, &ops),
            InstKind::ExtractElement { lane } => self.vec_extract(lo, inst, *lane, &ops),
            InstKind::InsertElement { lane } => self.vec_insert(lo, inst, *lane, &ops),
            InstKind::ShuffleVector(mask) => self.vec_shuffle(lo, inst, mask, &ops),
            InstKind::Splat => self.vec_splat(lo, inst, &ops),
            InstKind::Load { align, .. } => {
                let d = lo.result_reg(inst);
                let p = self.oper(lo, ops[0]);
                lo.emit(MachineInst::new(
                    X86Op::VLoad.opcode(),
                    vec![def_v(d), use_v(p), imm(u64::from(*align >= 16))],
                ));
            }
            InstKind::Store { align, .. } => {
                let p = self.oper(lo, ops[0]);
                let v = self.oper(lo, ops[1]);
                lo.emit(MachineInst::new(
                    X86Op::VStore.opcode(),
                    vec![use_v(p), use_v(v), imm(u64::from(*align >= 16))],
                ));
            }
            other => panic!("x86-64: vector {other:?} reached isel unlegalized (legalize with Sse2Legality first)"),
        }
        true
    }

    fn vshape(&self, lo: &Lower<'_, Self>, ty: TypeId) -> (Lanes, u32, u32) {
        shape(lo.types(), ty).expect("a legal SSE2 vector type (legalize with Sse2Legality first)")
    }

    fn vec_bin(&self, lo: &mut Lower<'_, Self>, inst: &InstData, op: BinOp, ops: &[ValueId]) {
        let (k, ..) = self.vshape(lo, inst.ty);
        let a = self.oper(lo, ops[0]);
        let r = match (k, op) {
            (Lanes::Float(w), _) => {
                let pfx = if w == 64 { 0x66 } else { 0x00 };
                let (opc, comm) = match op {
                    BinOp::FAdd => (0x58, true),
                    BinOp::FSub => (0x5C, false),
                    BinOp::FMul => (0x59, true),
                    BinOp::FDiv => (0x5E, false),
                    _ => unreachable!("unsupported float vector op {op:?}"),
                };
                let b = self.oper(lo, ops[1]);
                self.vop(lo, a, b, VEnc::op(pfx, opc, comm))
            }
            (Lanes::Int(w), BinOp::Shl | BinOp::LShr | BinOp::AShr) => {
                let c = uniform_const(lo.module().consts(), lo.func(), ops[1], w)
                    .expect("a uniform constant shift amount");
                let opcode = match w {
                    16 => 0x71,
                    32 => 0x72,
                    _ => 0x73,
                };
                let ext = match op {
                    BinOp::Shl => 6,
                    BinOp::LShr => 2,
                    _ => 4,
                };
                // A count ≥ the lane width is poison: any result will do.
                self.vshift(lo, a, opcode, ext, c.min(255) as u8)
            }
            (Lanes::Int(32), BinOp::Mul) => {
                let b = self.oper(lo, ops[1]);
                self.mul32(lo, a, b)
            }
            (Lanes::Int(w), op) if op.is_minmax_sat() => {
                let (opc, comm) = minmax_sat_opcode(op, w).expect("a direct SSE2 min/max/saturating form");
                let b = self.oper(lo, ops[1]);
                self.pop(lo, a, b, opc, comm)
            }
            (_, _) => {
                let b = self.oper(lo, ops[1]);
                let w = match k {
                    Lanes::Int(w) => w,
                    _ => 8,
                };
                let (opc, comm) = match op {
                    BinOp::Add => (PADD[widx(w)], true),
                    BinOp::Sub => (PSUB[widx(w)], false),
                    BinOp::And => (PAND, true),
                    BinOp::Or => (POR, true),
                    BinOp::Xor => (PXOR, true),
                    BinOp::Mul => (PMULLW, true),
                    _ => unreachable!("unsupported integer vector op {op:?}"),
                };
                self.pop(lo, a, b, opc, comm)
            }
        };
        self.finish(lo, inst, r);
    }

    /// `<4 x i32>` multiply without SSE4.1: two `pmuludq` on the even and odd
    /// dword lanes (the low 32 bits of each 64-bit product are the wrapped
    /// result), then gather the four low halves back in order.
    fn mul32(&self, lo: &mut Lower<'_, Self>, a: VReg, b: VReg) -> VReg {
        let even = self.pop(lo, a, b, PMULUDQ, true); // lanes 0, 2 (as qwords)
        let a_odd = self.pshufd(lo, a, 0xF5); // [1, 1, 3, 3]
        let b_odd = self.pshufd(lo, b, 0xF5);
        let odd = self.pop(lo, a_odd, b_odd, PMULUDQ, true); // lanes 1, 3
        let e = self.pshufd(lo, even, 0x08); // [p0, p2, _, _]
        let o = self.pshufd(lo, odd, 0x08); // [p1, p3, _, _]
        self.pop(lo, e, o, PUNPCKLDQ, false) // [p0, p1, p2, p3]
    }

    fn vec_icmp(&self, lo: &mut Lower<'_, Self>, inst: &InstData, pred: IntPred, ops: &[ValueId]) {
        let (k, ..) = self.vshape(lo, lo.func().value_type(ops[0]));
        let Lanes::Int(w) = k else { unreachable!("icmp on {k:?} lanes is scalarized") };
        let mut a = self.oper(lo, ops[0]);
        let mut b = self.oper(lo, ops[1]);
        if w == 64 {
            // eq/ne: both dword halves of each qword must match.
            let t = self.pop(lo, a, b, PCMPEQ[2], true);
            let s = self.pshufd(lo, t, 0xB1);
            let mut r = self.pop(lo, t, s, PAND, true);
            if pred == IntPred::Ne {
                r = self.vnot(lo, r);
            }
            return self.finish(lo, inst, r);
        }
        let i = widx(w);
        // Unsigned order = signed order after flipping both sign bits.
        if matches!(pred, IntPred::Ugt | IntPred::Uge | IntPred::Ult | IntPred::Ule) {
            let flip = self.vconst(lo, splat_bits(w, 1u64 << (w - 1)));
            a = self.pop(lo, a, flip, PXOR, true);
            b = self.pop(lo, b, flip, PXOR, true);
        }
        let r = match pred {
            IntPred::Eq => self.pop(lo, a, b, PCMPEQ[i], true),
            IntPred::Ne => {
                let e = self.pop(lo, a, b, PCMPEQ[i], true);
                self.vnot(lo, e)
            }
            IntPred::Sgt | IntPred::Ugt => self.pop(lo, a, b, PCMPGT[i], false),
            IntPred::Slt | IntPred::Ult => self.pop(lo, b, a, PCMPGT[i], false),
            IntPred::Sle | IntPred::Ule => {
                let g = self.pop(lo, a, b, PCMPGT[i], false);
                self.vnot(lo, g)
            }
            IntPred::Sge | IntPred::Uge => {
                let l = self.pop(lo, b, a, PCMPGT[i], false);
                self.vnot(lo, l)
            }
        };
        self.finish(lo, inst, r);
    }

    fn vec_fcmp(&self, lo: &mut Lower<'_, Self>, inst: &InstData, pred: FloatPred, ops: &[ValueId]) {
        let (k, ..) = self.vshape(lo, lo.func().value_type(ops[0]));
        let pfx = if k == Lanes::Float(64) { 0x66 } else { 0x00 };
        let cmp = |this: &Self, lo: &mut Lower<'_, Self>, x: VReg, y: VReg, p: u8| {
            this.vop(lo, x, y, VEnc::op_imm(pfx, 0xC2, p))
        };
        let r = match pred {
            FloatPred::False => self.vconst(lo, (0, 0)),
            FloatPred::True => self.vconst(lo, (u64::MAX, u64::MAX)),
            _ => {
                let a = self.oper(lo, ops[0]);
                let b = self.oper(lo, ops[1]);
                // cmpps predicates: 0 eq, 1 lt, 2 le, 3 unord, 4 neq, 5 nlt,
                // 6 nle, 7 ord (ordered ones false on NaN, "n" ones true).
                match pred {
                    FloatPred::Oeq => cmp(self, lo, a, b, 0),
                    FloatPred::Olt => cmp(self, lo, a, b, 1),
                    FloatPred::Ole => cmp(self, lo, a, b, 2),
                    FloatPred::Ogt => cmp(self, lo, b, a, 1),
                    FloatPred::Oge => cmp(self, lo, b, a, 2),
                    FloatPred::Uno => cmp(self, lo, a, b, 3),
                    FloatPred::Une => cmp(self, lo, a, b, 4),
                    FloatPred::Uge => cmp(self, lo, a, b, 5),
                    FloatPred::Ugt => cmp(self, lo, a, b, 6),
                    FloatPred::Ule => cmp(self, lo, b, a, 5),
                    FloatPred::Ult => cmp(self, lo, b, a, 6),
                    FloatPred::Ord => cmp(self, lo, a, b, 7),
                    FloatPred::One => {
                        let o = cmp(self, lo, a, b, 7);
                        let n = cmp(self, lo, a, b, 4);
                        self.pop(lo, o, n, PAND, true)
                    }
                    FloatPred::Ueq => {
                        let u = cmp(self, lo, a, b, 3);
                        let e = cmp(self, lo, a, b, 0);
                        self.pop(lo, u, e, POR, true)
                    }
                    FloatPred::False | FloatPred::True => unreachable!(),
                }
            }
        };
        self.finish(lo, inst, r);
    }

    fn vec_cast(&self, lo: &mut Lower<'_, Self>, inst: &InstData, op: CastOp, ops: &[ValueId]) {
        let s = self.oper(lo, ops[0]);
        let (tk, _, tcw) = self.vshape(lo, inst.ty);
        let r = match op {
            // Same 128 bits, reinterpreted; a mask is already the sign
            // extension of its lanes.
            CastOp::Bitcast | CastOp::SExt => s,
            CastOp::SiToFp => self.vunary(lo, s, VEnc::op(0x00, 0x5B, false)), // cvtdq2ps
            CastOp::FpToSi => self.vunary(lo, s, VEnc::op(0xF3, 0x5B, false)), // cvttps2dq
            CastOp::ZExt => {
                let one = self.vconst(lo, splat_bits(tcw, 1));
                self.pop(lo, s, one, PAND, true)
            }
            CastOp::Trunc => {
                let (fk, ..) = self.vshape(lo, lo.func().value_type(ops[0]));
                debug_assert_eq!(tk, Lanes::Mask);
                match fk {
                    Lanes::Int(64) => {
                        let t = self.vshift(lo, s, 0x73, 6, 63); // psllq 63
                        let t = self.vshift(lo, t, 0x72, 4, 31); // psrad 31
                        self.pshufd(lo, t, 0xF5) // copy each qword's high dword down
                    }
                    Lanes::Int(w) => {
                        let one = self.vconst(lo, splat_bits(w, 1));
                        let bit = self.pop(lo, s, one, PAND, true);
                        self.pop(lo, bit, one, PCMPEQ[widx(w)], true)
                    }
                    _ => unreachable!("trunc from {fk:?} is scalarized"),
                }
            }
            _ => unreachable!("vector cast {op:?} is scalarized"),
        };
        self.finish(lo, inst, r);
    }

    /// Also the lowering of a scalar float `select` (a blend of whole xmm
    /// registers; a GPR `cmov` cannot move xmm values).
    pub(super) fn vec_select(&self, lo: &mut Lower<'_, Self>, inst: &InstData, ops: &[ValueId]) {
        let mask = if lo.types().is_vector(lo.func().value_type(ops[0])) {
            self.oper(lo, ops[0])
        } else {
            // Broadcast an `i1` to an all-ones/all-zeros register: every dword
            // equal works for any lane width.
            let c = self.clean_cond(lo, ops[0]);
            let m = self.bool_to_mask(lo, c);
            let x = self.gpr_to_x(lo, m, false);
            self.pshufd(lo, x, 0)
        };
        let t = self.oper(lo, ops[1]);
        let f = self.oper(lo, ops[2]);
        let keep_t = self.pop(lo, mask, t, PAND, true);
        let keep_f = self.pop(lo, mask, f, PANDN, false); // !mask & f
        let r = self.pop(lo, keep_t, keep_f, POR, true);
        self.finish(lo, inst, r);
    }

    fn vec_extract(&self, lo: &mut Lower<'_, Self>, inst: &InstData, lane: u32, ops: &[ValueId]) {
        let (k, _, cw) = self.vshape(lo, lo.func().value_type(ops[0]));
        let v = self.oper(lo, ops[0]);
        let r = match (k, cw) {
            (Lanes::Float(_), 32) => {
                if lane == 0 { v } else { self.pshufd(lo, v, lane as u8) }
            }
            (Lanes::Float(_), _) => {
                if lane == 0 { v } else { self.pshufd(lo, v, 0xEE) }
            }
            (_, 32) => {
                let t = if lane == 0 { v } else { self.pshufd(lo, v, lane as u8) };
                self.x_to_gpr(lo, t, false)
            }
            (_, 64) => {
                let t = if lane == 0 { v } else { self.pshufd(lo, v, 0xEE) };
                self.x_to_gpr(lo, t, true)
            }
            (_, 16) => self.pextrw(lo, v, lane),
            (_, _) => {
                let w = self.pextrw(lo, v, lane / 2);
                if lane % 2 == 1 { self.gpr_shift(lo, X86Op::ShrI, w, 8, 32) } else { w }
            }
        };
        self.finish(lo, inst, r);
    }

    fn pextrw(&self, lo: &mut Lower<'_, Self>, v: VReg, idx: u32) -> VReg {
        let g = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::Pextrw.opcode(), vec![def_v(g), use_v(v), imm(u64::from(idx))]));
        g
    }

    fn pinsrw(&self, lo: &mut Lower<'_, Self>, v: VReg, g: VReg, idx: u32) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(
            X86Op::Pinsrw.opcode(),
            vec![def_v(d), use_v(v), use_v(g), imm(u64::from(idx))],
        ));
        d
    }

    fn vec_insert(&self, lo: &mut Lower<'_, Self>, inst: &InstData, lane: u32, ops: &[ValueId]) {
        let (k, _, cw) = self.vshape(lo, inst.ty);
        let v = self.oper(lo, ops[0]);
        let s = self.oper(lo, ops[1]);
        let r = match (k, cw) {
            (Lanes::Float(64), _) => {
                // movsd merges the low qword; unpcklpd appends it as lane 1.
                let enc = if lane == 0 { VEnc::op(0xF2, 0x10, false) } else { VEnc::op(0x66, 0x14, false) };
                self.vop(lo, v, s, enc)
            }
            _ => {
                // The lane's bits in a GPR.
                let g = match k {
                    Lanes::Float(_) => self.x_to_gpr(lo, s, false),
                    Lanes::Mask => self.bool_to_mask(lo, s),
                    Lanes::Int(_) => s,
                };
                match cw {
                    64 => {
                        let x = self.gpr_to_x(lo, g, true);
                        let enc = if lane == 0 { VEnc::op(0xF2, 0x10, false) } else { VEnc::op(0x66, PUNPCKLQDQ, false) };
                        self.vop(lo, v, x, enc)
                    }
                    32 => {
                        let t = self.pinsrw(lo, v, g, 2 * lane);
                        let hi = self.gpr_shift(lo, X86Op::ShrI, g, 16, 32);
                        self.pinsrw(lo, t, hi, 2 * lane + 1)
                    }
                    16 => self.pinsrw(lo, v, g, lane),
                    _ => {
                        // Merge the byte into its word.
                        let word = self.pextrw(lo, v, lane / 2);
                        let byte = self.gpr_imm(lo, X86Op::And, g, 0xFF, 32);
                        let merged = if lane.is_multiple_of(2) {
                            let keep = self.gpr_imm(lo, X86Op::And, word, 0xFF00, 32);
                            self.gpr_bin(lo, X86Op::Or, keep, byte, 32)
                        } else {
                            let keep = self.gpr_imm(lo, X86Op::And, word, 0x00FF, 32);
                            let hi = self.gpr_shift(lo, X86Op::ShlI, byte, 8, 32);
                            self.gpr_bin(lo, X86Op::Or, keep, hi, 32)
                        };
                        self.pinsrw(lo, v, merged, lane / 2)
                    }
                }
            }
        };
        self.finish(lo, inst, r);
    }

    fn vec_shuffle(&self, lo: &mut Lower<'_, Self>, inst: &InstData, mask: &[u32], ops: &[ValueId]) {
        let (_, n, cw) = self.vshape(lo, lo.func().value_type(ops[0]));
        let a = self.oper(lo, ops[0]);
        let b = self.oper(lo, ops[1]);
        let src = |m: u32| if m < n { a } else { b };
        let r = if cw == 32 {
            let l = |m: u32| (m % 4) as u8;
            let single = mask.iter().all(|&m| m < n) || mask.iter().all(|&m| m >= n);
            if single {
                let imm8 = l(mask[0]) | (l(mask[1]) << 2) | (l(mask[2]) << 4) | (l(mask[3]) << 6);
                self.pshufd(lo, src(mask[0]), imm8)
            } else {
                // shufps: result lanes 0-1 from the first operand, 2-3 from
                // the second.
                let imm8 = l(mask[0]) | (l(mask[1]) << 2) | (l(mask[2]) << 4) | (l(mask[3]) << 6);
                self.vop(lo, src(mask[0]), src(mask[2]), VEnc::op_imm(0x00, 0xC6, imm8))
            }
        } else {
            let l = |m: u32| (m % 2) as u8;
            if (mask[0] < n) == (mask[1] < n) {
                // One source: move each qword as its dword pair.
                let dw = |q: u8| (2 * q) | ((2 * q + 1) << 2);
                let imm8 = dw(l(mask[0])) | (dw(l(mask[1])) << 4);
                self.pshufd(lo, src(mask[0]), imm8)
            } else {
                // shufpd: lane 0 from the first operand, lane 1 from the second.
                let imm8 = l(mask[0]) | (l(mask[1]) << 1);
                self.vop(lo, src(mask[0]), src(mask[1]), VEnc::op_imm(0x66, 0xC6, imm8))
            }
        };
        self.finish(lo, inst, r);
    }

    fn vec_splat(&self, lo: &mut Lower<'_, Self>, inst: &InstData, ops: &[ValueId]) {
        let (k, _, cw) = self.vshape(lo, inst.ty);
        let s = self.oper(lo, ops[0]);
        let r = match (k, cw) {
            (Lanes::Float(_), 32) => self.pshufd(lo, s, 0x00),
            (Lanes::Float(_), _) => self.pshufd(lo, s, 0x44),
            (Lanes::Int(64), _) => {
                let x = self.gpr_to_x(lo, s, true);
                self.pshufd(lo, x, 0x44)
            }
            _ => {
                // Replicate the lane to a full dword in a GPR, then broadcast
                // the dword (a mask lane is all-ones/zeros at any width).
                let dword = match (k, cw) {
                    (Lanes::Mask, _) => self.bool_to_mask(lo, s),
                    (_, 8) => {
                        let b = self.gpr_imm(lo, X86Op::And, s, 0xFF, 32);
                        self.gpr_imm(lo, X86Op::Imul, b, 0x0101_0101, 32)
                    }
                    (_, 16) => {
                        let w = self.gpr_imm(lo, X86Op::And, s, 0xFFFF, 32);
                        self.gpr_imm(lo, X86Op::Imul, w, 0x0001_0001, 32)
                    }
                    _ => s,
                };
                let x = self.gpr_to_x(lo, dword, false);
                self.pshufd(lo, x, 0x00)
            }
        };
        self.finish(lo, inst, r);
    }
}
