//! AArch64 Advanced SIMD (NEON) lowering of 128-bit vectors
//! (`docs/ir-design.md` §6c).
//!
//! [`NeonLegality`] keeps the same types as the x86-64 SSE2 backend whole in a
//! `v` register — `<16 x i8>`, `<8 x i16>`, `<4 x i32>`, `<2 x i64>`,
//! `<4 x f32>`, `<2 x f64>` and the masks `<16/8/4/2 x i1>` (see
//! [`crate::codegen::simd128`] for the mask convention, which is exactly what
//! the NEON compares produce). NEON is much more regular than SSE2, so far
//! less is scalarized:
//!
//! | IR | NEON |
//! |---|---|
//! | `add`/`sub`, `and`/`or`/`xor` | `add`/`sub`, `and`/`orr`/`eor` |
//! | `mul` i8/i16/i32 | `mul` |
//! | `shl` / `lshr` / `ashr`, uniform constant | `shl` / `ushr` / `sshr` `#imm` |
//! | `shl` / `lshr` / `ashr`, per lane | `ushl` / `ushl` by `neg` / `sshl` by `neg` |
//! | `smin`/`smax`/`umin`/`umax` i8–i32 | `smin`/`smax`/`umin`/`umax` |
//! | saturating add/sub | `sqadd`/`uqadd`/`sqsub`/`uqsub` |
//! | `fadd`/`fsub`/`fmul`/`fdiv`, `fneg` | same names |
//! | `icmp` | `cmeq`/`cmgt`/`cmge`/`cmhi`/`cmhs` (swapped for `<`, `not` for `ne`) |
//! | `fcmp` | `fcmeq`/`fcmgt`/`fcmge` (swapped, `not`, `orr` for the unordered/`one` forms) |
//! | `select` | `and` + `bic` + `orr` (an `i1` condition broadcast with `dup`) |
//! | `sitofp`/`uitofp`/`fptosi`/`fptoui`, same lane width | `scvtf`/`ucvtf`/`fcvtzs`/`fcvtzu` |
//! | mask ↔ int of the lane width | a copy (`sext`), `ushr #w-1` (`zext`), `shl` + `sshr` (`trunc`) |
//! | `extractelement` | `umov` (integer), `dup` element (float) |
//! | `insertelement` | `ins` from a GPR / from an element |
//! | `shufflevector` (same lane count) | `tbl` with a constant byte index (two `tbl` + `orr` for two sources) |
//! | `splat` | `dup` (general / element) |
//! | `reduce add` / min / max, 8–32-bit lanes | `addv` / `sminv`… + `umov`; `reduce add` i64: `addp` |
//! | `load`/`store` | `ldr q` / `str q` |
//! | constants | `movz`/`movk` + `fmov` + `ins` through `x16` |
//!
//! Scalarized: division and remainder, `frem`, `mul` and min/max on i64
//! lanes, other casts and shuffles, the other reductions, and mask
//! loads/stores. Vectors pass in `v0..v7` and return in `v0` (AAPCS64 short
//! vectors); a function holding vectors treats `v8..v15` as clobbered by calls,
//! since AAPCS64 preserves only their low 64 bits.

use crate::codegen::isel::Lower;
use crate::codegen::legalize::VectorLegality;
use crate::codegen::mir::{MachineInst, RegClass, VReg};
use crate::codegen::simd128::{Lanes, shape, uniform_const};
use crate::ir::inst::{BinOp, CastOp, FloatPred, InstData, InstKind, IntPred, ReduceOp, UnaryOp};
use crate::ir::types::{TypeContext, TypeId};
use crate::ir::value::{ConstPool, ValueId};
use crate::ir::Function;

use super::{A64Op, AArch64Target, def_v, imm, use_v};

/// The NEON vector legality of the AArch64 backend (see the module docs).
#[derive(Clone, Copy, Debug, Default)]
pub struct NeonLegality;

/// The NEON operations the backend emits, as the `op` immediate of
/// [`A64Op::NeonOp3`]/[`A64Op::NeonOp2`]/[`A64Op::NeonShift`]. All run on the
/// full 128-bit register with the lane size given alongside.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum NeonOp {
    // --- three registers: `op Vd.T, Vn.T, Vm.T` ---------------------------
    Add,
    Sub,
    Mul,
    And,
    /// `bic`: `n & !m`.
    Bic,
    Orr,
    Eor,
    Cmeq,
    Cmgt,
    Cmge,
    Cmhi,
    Cmhs,
    Sshl,
    Ushl,
    Smax,
    Smin,
    Umax,
    Umin,
    Sqadd,
    Uqadd,
    Sqsub,
    Uqsub,
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Fcmeq,
    Fcmge,
    Fcmgt,
    /// `tbl Vd.16b, {Vn.16b}, Vm.16b`: byte `i` is `n[m[i]]`, or 0 when
    /// `m[i] ≥ 16`.
    Tbl,
    // --- two registers: `op Vd.T, Vn.T` -----------------------------------
    Neg,
    Not,
    Fneg,
    Scvtf,
    Ucvtf,
    Fcvtzs,
    Fcvtzu,
    /// `addv`: the wrapping sum of the lanes into lane 0 (the rest zero).
    Addv,
    Smaxv,
    Sminv,
    Umaxv,
    Uminv,
    /// `addp Dd, Vn.2d`: the sum of the two quadwords into the low one.
    Addp,
    // --- shift by immediate: `op Vd.T, Vn.T, #amt` ------------------------
    Shl,
    Ushr,
    Sshr,
}

impl NeonOp {
    /// Every operation, in code order.
    pub(crate) const ALL: [NeonOp; 46] = {
        use NeonOp::*;
        [
            Add, Sub, Mul, And, Bic, Orr, Eor, Cmeq, Cmgt, Cmge, Cmhi, Cmhs, Sshl, Ushl, Smax, Smin, Umax, Umin,
            Sqadd, Uqadd, Sqsub, Uqsub, Fadd, Fsub, Fmul, Fdiv, Fcmeq, Fcmge, Fcmgt, Tbl, Neg, Not, Fneg, Scvtf,
            Ucvtf, Fcvtzs, Fcvtzu, Addv, Smaxv, Sminv, Umaxv, Uminv, Addp, Shl, Ushr, Sshr,
        ]
    };

    /// This operation's immediate code.
    pub(crate) fn code(self) -> u64 {
        self as u64
    }

    /// The operation with code `c`.
    pub(crate) fn from_code(c: u64) -> NeonOp {
        NeonOp::ALL[c as usize]
    }
}

fn neon_mul_ok(w: u32) -> bool {
    w <= 32
}

impl VectorLegality for NeonLegality {
    /// Bulk memory (`docs/ir-design.md` §6k): up to eight 16-byte (`q`
    /// register) or smaller chunks inline, unaligned accesses being fine on
    /// normal memory; longer or variable lengths loop over 8-byte words.
    fn bulk_memory(&self, _layout: &crate::ir::DataLayout) -> crate::codegen::legalize_mem::BulkMemoryLowering {
        crate::codegen::legalize_mem::BulkMemoryLowering {
            word: 8,
            vector16: true,
            unaligned: true,
            max_inline: 8,
            native: false,
            guard_zero: false,
        }
    }

    fn legal_type(&self, types: &TypeContext, ty: TypeId) -> bool {
        shape(types, ty).is_some()
    }

    fn legal_inst(&self, types: &TypeContext, _consts: &ConstPool, func: &Function, inst: &InstData) -> bool {
        let ops = inst.operands();
        let rshape = shape(types, inst.ty);
        let oshape = |i: usize| ops.get(i).and_then(|&o| shape(types, func.value_type(o)));
        match &inst.kind {
            InstKind::Bin(op) => match rshape {
                Some((Lanes::Float(_), ..)) => matches!(op, BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv),
                Some((Lanes::Mask, ..)) => matches!(op, BinOp::And | BinOp::Or | BinOp::Xor),
                Some((Lanes::Int(w), ..)) => match op {
                    BinOp::Add
                    | BinOp::Sub
                    | BinOp::And
                    | BinOp::Or
                    | BinOp::Xor
                    | BinOp::Shl
                    | BinOp::LShr
                    | BinOp::AShr
                    | BinOp::SAddSat
                    | BinOp::UAddSat
                    | BinOp::SSubSat
                    | BinOp::USubSat => true,
                    BinOp::Mul | BinOp::SMin | BinOp::SMax | BinOp::UMin | BinOp::UMax => neon_mul_ok(w),
                    _ => false,
                },
                None => false,
            },
            InstKind::Unary(UnaryOp::FNeg) => true,
            InstKind::ICmp(_) => matches!(oshape(0), Some((Lanes::Int(_), ..))),
            InstKind::FCmp(_) => true,
            InstKind::Cast(op) => {
                let (Some((fk, _, fcw)), Some((tk, _, tcw))) = (oshape(0), rshape) else {
                    return false;
                };
                match op {
                    CastOp::Bitcast => fk != Lanes::Mask && tk != Lanes::Mask,
                    CastOp::SiToFp | CastOp::UiToFp => fk == Lanes::Int(tcw) && tk == Lanes::Float(tcw),
                    CastOp::FpToSi | CastOp::FpToUi => fk == Lanes::Float(fcw) && tk == Lanes::Int(fcw),
                    CastOp::SExt | CastOp::ZExt => fk == Lanes::Mask && tk == Lanes::Int(fcw),
                    CastOp::Trunc => tk == Lanes::Mask && fk == Lanes::Int(tcw),
                    _ => false,
                }
            }
            InstKind::Select | InstKind::ExtractElement { .. } | InstKind::InsertElement { .. } | InstKind::Splat => {
                true
            }
            InstKind::ShuffleVector(mask) => {
                matches!(oshape(0), Some((_, n, _)) if mask.len() == n as usize)
            }
            InstKind::Reduce(op) => match (op, oshape(0)) {
                (ReduceOp::Add, Some((Lanes::Int(_), ..))) => true,
                (ReduceOp::SMin | ReduceOp::SMax | ReduceOp::UMin | ReduceOp::UMax, Some((Lanes::Int(w), ..))) => {
                    w <= 32
                }
                _ => false,
            },
            InstKind::Load { ty, .. } | InstKind::Store { ty, .. } => {
                !matches!(shape(types, *ty), Some((Lanes::Mask, ..)))
            }
            _ => false,
        }
    }
}

/// The lane size an operation on a vector of shape `(k, _, cw)` runs at.
fn esize_of(k: Lanes, cw: u32) -> u32 {
    match k {
        Lanes::Int(w) | Lanes::Float(w) => w,
        Lanes::Mask => cw,
    }
}

impl AArch64Target {
    // --- emitters -------------------------------------------------------------

    fn n3(&self, lo: &mut Lower<'_, Self>, op: NeonOp, es: u32, a: VReg, b: VReg) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(
            A64Op::NeonOp3.opcode(),
            vec![def_v(d), use_v(a), use_v(b), imm(op.code()), imm(u64::from(es))],
        ));
        d
    }

    fn n2(&self, lo: &mut Lower<'_, Self>, op: NeonOp, es: u32, a: VReg) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(A64Op::NeonOp2.opcode(), vec![def_v(d), use_v(a), imm(op.code()), imm(u64::from(es))]));
        d
    }

    fn nshift(&self, lo: &mut Lower<'_, Self>, op: NeonOp, es: u32, a: VReg, amt: u32) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(
            A64Op::NeonShift.opcode(),
            vec![def_v(d), use_v(a), imm(op.code()), imm(u64::from(es)), imm(u64::from(amt))],
        ));
        d
    }

    fn nconst(&self, lo: &mut Lower<'_, Self>, bits: (u64, u64)) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(A64Op::NeonConst.opcode(), vec![def_v(d), imm(bits.0), imm(bits.1)]));
        d
    }

    fn ndup(&self, lo: &mut Lower<'_, Self>, g: VReg, es: u32) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(A64Op::NeonDup.opcode(), vec![def_v(d), use_v(g), imm(u64::from(es))]));
        d
    }

    fn ndup_lane(&self, lo: &mut Lower<'_, Self>, v: VReg, es: u32, lane: u32) -> VReg {
        let d = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(
            A64Op::NeonDupLane.opcode(),
            vec![def_v(d), use_v(v), imm(u64::from(es)), imm(u64::from(lane))],
        ));
        d
    }

    fn numov(&self, lo: &mut Lower<'_, Self>, v: VReg, es: u32, lane: u32) -> VReg {
        let g = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(
            A64Op::NeonUmov.opcode(),
            vec![def_v(g), use_v(v), imm(u64::from(es)), imm(u64::from(lane))],
        ));
        g
    }

    /// `0` or all-ones from an `i1` in a GPR (upper bits garbage).
    fn neon_bool_mask(&self, lo: &mut Lower<'_, Self>, g: VReg) -> VReg {
        let one = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(A64Op::MovRI.opcode(), vec![def_v(one), imm(1)]));
        let bit = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(A64Op::And.opcode(), vec![def_v(bit), use_v(g), use_v(one), imm(64)]));
        let zero = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(A64Op::MovRI.opcode(), vec![def_v(zero), imm(0)]));
        let m = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(A64Op::Sub.opcode(), vec![def_v(m), use_v(zero), use_v(bit), imm(64)]));
        m
    }

    fn nfinish(&self, lo: &mut Lower<'_, Self>, inst: &InstData, r: VReg) {
        let d = lo.result_reg(inst);
        lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_v(r)]));
    }

    fn nshape(&self, lo: &Lower<'_, Self>, ty: TypeId) -> (Lanes, u32, u32) {
        shape(lo.types(), ty).expect("a legal NEON vector type (legalize with NeonLegality first)")
    }

    // --- the dispatcher ---------------------------------------------------------

    /// Lower a vector instruction; `false` if `inst` is not vector code (or is a
    /// whole-value op the scalar paths already handle: `call`, `freeze`).
    pub(super) fn lower_neon(&self, lo: &mut Lower<'_, Self>, inst: &InstData) -> bool {
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
            InstKind::Bin(op) => self.neon_bin(lo, inst, *op, &ops),
            InstKind::Unary(UnaryOp::FNeg) => {
                let (k, _, cw) = self.nshape(lo, inst.ty);
                let a = lo.reg(ops[0]);
                let r = self.n2(lo, NeonOp::Fneg, esize_of(k, cw), a);
                self.nfinish(lo, inst, r);
            }
            InstKind::ICmp(pred) => self.neon_icmp(lo, inst, *pred, &ops),
            InstKind::FCmp(pred) => self.neon_fcmp(lo, inst, *pred, &ops),
            InstKind::Cast(op) => self.neon_cast(lo, inst, *op, &ops),
            InstKind::Select => self.neon_select(lo, inst, &ops),
            InstKind::ExtractElement { lane } => {
                let (k, _, cw) = self.nshape(lo, lo.func().value_type(ops[0]));
                let v = lo.reg(ops[0]);
                let es = esize_of(k, cw);
                let r = match k {
                    Lanes::Float(_) => self.ndup_lane(lo, v, es, *lane),
                    _ => self.numov(lo, v, es, *lane),
                };
                self.nfinish(lo, inst, r);
            }
            InstKind::InsertElement { lane } => {
                let (k, _, cw) = self.nshape(lo, inst.ty);
                let es = esize_of(k, cw);
                let v = lo.reg(ops[0]);
                let s = lo.reg(ops[1]);
                let d = lo.fresh_vreg(RegClass::Fp);
                if let Lanes::Float(_) = k {
                    lo.emit(MachineInst::new(
                        A64Op::NeonInsElem.opcode(),
                        vec![def_v(d), use_v(v), use_v(s), imm(u64::from(es)), imm(u64::from(*lane))],
                    ));
                } else {
                    let g = if k == Lanes::Mask { self.neon_bool_mask(lo, s) } else { s };
                    lo.emit(MachineInst::new(
                        A64Op::NeonInsGpr.opcode(),
                        vec![def_v(d), use_v(v), use_v(g), imm(u64::from(es)), imm(u64::from(*lane))],
                    ));
                }
                self.nfinish(lo, inst, d);
            }
            InstKind::ShuffleVector(mask) => self.neon_shuffle(lo, inst, mask, &ops),
            InstKind::Splat => {
                let (k, _, cw) = self.nshape(lo, inst.ty);
                let es = esize_of(k, cw);
                let s = lo.reg(ops[0]);
                let r = match k {
                    Lanes::Float(_) => self.ndup_lane(lo, s, es, 0),
                    Lanes::Mask => {
                        let m = self.neon_bool_mask(lo, s);
                        self.ndup(lo, m, 32)
                    }
                    Lanes::Int(_) => self.ndup(lo, s, es),
                };
                self.nfinish(lo, inst, r);
            }
            InstKind::Reduce(op) => {
                let (k, _, cw) = self.nshape(lo, lo.func().value_type(ops[0]));
                let es = esize_of(k, cw);
                let v = lo.reg(ops[0]);
                let across = match (op, es) {
                    (ReduceOp::Add, 64) => NeonOp::Addp,
                    (ReduceOp::Add, _) => NeonOp::Addv,
                    (ReduceOp::SMax, _) => NeonOp::Smaxv,
                    (ReduceOp::SMin, _) => NeonOp::Sminv,
                    (ReduceOp::UMax, _) => NeonOp::Umaxv,
                    (ReduceOp::UMin, _) => NeonOp::Uminv,
                    _ => unreachable!("reduce {op:?} is scalarized for NEON"),
                };
                let t = self.n2(lo, across, es, v);
                let r = self.numov(lo, t, es, 0);
                self.nfinish(lo, inst, r);
            }
            InstKind::Load { .. } => {
                let d = lo.result_reg(inst);
                let p = lo.reg(ops[0]);
                lo.emit(MachineInst::new(A64Op::NeonLoad.opcode(), vec![def_v(d), use_v(p)]));
            }
            InstKind::Store { .. } => {
                let p = lo.reg(ops[0]);
                let v = lo.reg(ops[1]);
                lo.emit(MachineInst::new(A64Op::NeonStore.opcode(), vec![use_v(p), use_v(v)]));
            }
            other => panic!("aarch64: vector {other:?} reached isel unlegalized (legalize with NeonLegality first)"),
        }
        true
    }

    /// `select` as a bitwise blend of whole `v` registers: a vector (mask or
    /// broadcast `i1` condition), or a scalar float (a GPR `csel` cannot move
    /// FP registers).
    pub(super) fn neon_select(&self, lo: &mut Lower<'_, Self>, inst: &InstData, ops: &[ValueId]) {
        let mask = if lo.types().is_vector(lo.func().value_type(ops[0])) {
            lo.reg(ops[0])
        } else {
            let c = self.clean_cond(lo, ops[0]);
            let m = self.neon_bool_mask(lo, c);
            self.ndup(lo, m, 32)
        };
        // The blend is bitwise, so the lane size is immaterial.
        let t = lo.reg(ops[1]);
        let f = lo.reg(ops[2]);
        let keep_t = self.n3(lo, NeonOp::And, 8, mask, t);
        let keep_f = self.n3(lo, NeonOp::Bic, 8, f, mask);
        let r = self.n3(lo, NeonOp::Orr, 8, keep_t, keep_f);
        self.nfinish(lo, inst, r);
    }

    fn neon_bin(&self, lo: &mut Lower<'_, Self>, inst: &InstData, op: BinOp, ops: &[ValueId]) {
        let (k, _, cw) = self.nshape(lo, inst.ty);
        let es = esize_of(k, cw);
        let a = lo.reg(ops[0]);
        let r = match (k, op) {
            (Lanes::Int(w), BinOp::Shl | BinOp::LShr | BinOp::AShr) => {
                match uniform_const(lo.module().consts(), lo.func(), ops[1], w) {
                    // A count ≥ the lane width is poison: any result will do.
                    Some(c) if c < u64::from(w) => {
                        let c = c as u32;
                        match op {
                            BinOp::Shl => self.nshift(lo, NeonOp::Shl, es, a, c),
                            _ if c == 0 => a,
                            BinOp::LShr => self.nshift(lo, NeonOp::Ushr, es, a, c),
                            _ => self.nshift(lo, NeonOp::Sshr, es, a, c),
                        }
                    }
                    Some(_) => a,
                    None => {
                        // Per-lane counts: `ushl`/`sshl` shift left by a signed
                        // byte count, right for a negative one.
                        let b = lo.reg(ops[1]);
                        match op {
                            BinOp::Shl => self.n3(lo, NeonOp::Ushl, es, a, b),
                            _ => {
                                let nb = self.n2(lo, NeonOp::Neg, es, b);
                                let shl = if op == BinOp::LShr { NeonOp::Ushl } else { NeonOp::Sshl };
                                self.n3(lo, shl, es, a, nb)
                            }
                        }
                    }
                }
            }
            _ => {
                let b = lo.reg(ops[1]);
                let nop = match op {
                    BinOp::Add => NeonOp::Add,
                    BinOp::Sub => NeonOp::Sub,
                    BinOp::Mul => NeonOp::Mul,
                    BinOp::And => NeonOp::And,
                    BinOp::Or => NeonOp::Orr,
                    BinOp::Xor => NeonOp::Eor,
                    BinOp::SMin => NeonOp::Smin,
                    BinOp::SMax => NeonOp::Smax,
                    BinOp::UMin => NeonOp::Umin,
                    BinOp::UMax => NeonOp::Umax,
                    BinOp::SAddSat => NeonOp::Sqadd,
                    BinOp::UAddSat => NeonOp::Uqadd,
                    BinOp::SSubSat => NeonOp::Sqsub,
                    BinOp::USubSat => NeonOp::Uqsub,
                    BinOp::FAdd => NeonOp::Fadd,
                    BinOp::FSub => NeonOp::Fsub,
                    BinOp::FMul => NeonOp::Fmul,
                    BinOp::FDiv => NeonOp::Fdiv,
                    _ => unreachable!("vector {op:?} is scalarized for NEON"),
                };
                self.n3(lo, nop, es, a, b)
            }
        };
        self.nfinish(lo, inst, r);
    }

    fn neon_icmp(&self, lo: &mut Lower<'_, Self>, inst: &InstData, pred: IntPred, ops: &[ValueId]) {
        let (k, _, cw) = self.nshape(lo, lo.func().value_type(ops[0]));
        let es = esize_of(k, cw);
        let a = lo.reg(ops[0]);
        let b = lo.reg(ops[1]);
        let r = match pred {
            IntPred::Eq => self.n3(lo, NeonOp::Cmeq, es, a, b),
            IntPred::Ne => {
                let e = self.n3(lo, NeonOp::Cmeq, es, a, b);
                self.n2(lo, NeonOp::Not, es, e)
            }
            IntPred::Sgt => self.n3(lo, NeonOp::Cmgt, es, a, b),
            IntPred::Sge => self.n3(lo, NeonOp::Cmge, es, a, b),
            IntPred::Slt => self.n3(lo, NeonOp::Cmgt, es, b, a),
            IntPred::Sle => self.n3(lo, NeonOp::Cmge, es, b, a),
            IntPred::Ugt => self.n3(lo, NeonOp::Cmhi, es, a, b),
            IntPred::Uge => self.n3(lo, NeonOp::Cmhs, es, a, b),
            IntPred::Ult => self.n3(lo, NeonOp::Cmhi, es, b, a),
            IntPred::Ule => self.n3(lo, NeonOp::Cmhs, es, b, a),
        };
        self.nfinish(lo, inst, r);
    }

    fn neon_fcmp(&self, lo: &mut Lower<'_, Self>, inst: &InstData, pred: FloatPred, ops: &[ValueId]) {
        let (k, _, cw) = self.nshape(lo, lo.func().value_type(ops[0]));
        let es = esize_of(k, cw);
        let r = match pred {
            FloatPred::False => self.nconst(lo, (0, 0)),
            FloatPred::True => self.nconst(lo, (u64::MAX, u64::MAX)),
            _ => {
                let a = lo.reg(ops[0]);
                let b = lo.reg(ops[1]);
                // The NEON compares are ordered (false on NaN).
                let gt = |s: &Self, lo: &mut Lower<'_, Self>, x, y| s.n3(lo, NeonOp::Fcmgt, es, x, y);
                let ge = |s: &Self, lo: &mut Lower<'_, Self>, x, y| s.n3(lo, NeonOp::Fcmge, es, x, y);
                let not = |s: &Self, lo: &mut Lower<'_, Self>, x| s.n2(lo, NeonOp::Not, es, x);
                let or = |s: &Self, lo: &mut Lower<'_, Self>, x, y| s.n3(lo, NeonOp::Orr, es, x, y);
                match pred {
                    FloatPred::Oeq => self.n3(lo, NeonOp::Fcmeq, es, a, b),
                    FloatPred::Ogt => gt(self, lo, a, b),
                    FloatPred::Oge => ge(self, lo, a, b),
                    FloatPred::Olt => gt(self, lo, b, a),
                    FloatPred::Ole => ge(self, lo, b, a),
                    FloatPred::One | FloatPred::Ueq => {
                        let x = gt(self, lo, a, b);
                        let y = gt(self, lo, b, a);
                        let one = or(self, lo, x, y);
                        if pred == FloatPred::One { one } else { not(self, lo, one) }
                    }
                    FloatPred::Ord | FloatPred::Uno => {
                        let x = ge(self, lo, a, b);
                        let y = gt(self, lo, b, a);
                        let ord = or(self, lo, x, y);
                        if pred == FloatPred::Ord { ord } else { not(self, lo, ord) }
                    }
                    FloatPred::Une => {
                        let e = self.n3(lo, NeonOp::Fcmeq, es, a, b);
                        not(self, lo, e)
                    }
                    FloatPred::Ugt => {
                        let x = ge(self, lo, b, a);
                        not(self, lo, x)
                    }
                    FloatPred::Uge => {
                        let x = gt(self, lo, b, a);
                        not(self, lo, x)
                    }
                    FloatPred::Ult => {
                        let x = ge(self, lo, a, b);
                        not(self, lo, x)
                    }
                    FloatPred::Ule => {
                        let x = gt(self, lo, a, b);
                        not(self, lo, x)
                    }
                    FloatPred::False | FloatPred::True => unreachable!(),
                }
            }
        };
        self.nfinish(lo, inst, r);
    }

    fn neon_cast(&self, lo: &mut Lower<'_, Self>, inst: &InstData, op: CastOp, ops: &[ValueId]) {
        let s = lo.reg(ops[0]);
        let (tk, _, tcw) = self.nshape(lo, inst.ty);
        let (fk, _, fcw) = self.nshape(lo, lo.func().value_type(ops[0]));
        let r = match op {
            CastOp::Bitcast | CastOp::SExt => s,
            CastOp::SiToFp => self.n2(lo, NeonOp::Scvtf, tcw, s),
            CastOp::UiToFp => self.n2(lo, NeonOp::Ucvtf, tcw, s),
            CastOp::FpToSi => self.n2(lo, NeonOp::Fcvtzs, fcw, s),
            CastOp::FpToUi => self.n2(lo, NeonOp::Fcvtzu, fcw, s),
            // An all-ones lane shifted down leaves 1.
            CastOp::ZExt => self.nshift(lo, NeonOp::Ushr, tcw, s, tcw - 1),
            CastOp::Trunc => {
                debug_assert!(tk == Lanes::Mask && matches!(fk, Lanes::Int(_)));
                let t = self.nshift(lo, NeonOp::Shl, fcw, s, fcw - 1);
                self.nshift(lo, NeonOp::Sshr, fcw, t, fcw - 1)
            }
            _ => unreachable!("vector cast {op:?} is scalarized for NEON"),
        };
        self.nfinish(lo, inst, r);
    }

    /// A shuffle through `tbl`: one table lookup per source with a constant
    /// byte-index vector (an out-of-range index yields zero), or'ed together.
    fn neon_shuffle(&self, lo: &mut Lower<'_, Self>, inst: &InstData, mask: &[u32], ops: &[ValueId]) {
        let (k, n, cw) = self.nshape(lo, lo.func().value_type(ops[0]));
        let lane_bytes = (esize_of(k, cw) / 8) as usize;
        let mut idx = [[0xFFu8; 16]; 2];
        for (i, &m) in mask.iter().enumerate() {
            let (src, lane) = ((m / n) as usize, (m % n) as usize);
            for j in 0..lane_bytes {
                idx[src][i * lane_bytes + j] = (lane * lane_bytes + j) as u8;
            }
        }
        let used: Vec<usize> = (0..2).filter(|&s| idx[s].iter().any(|&b| b != 0xFF)).collect();
        let mut parts = Vec::new();
        for s in used {
            let bytes = idx[s];
            let lo64 = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"));
            let hi64 = u64::from_le_bytes(bytes[8..].try_into().expect("8 bytes"));
            let table = lo.reg(ops[s]);
            let index = self.nconst(lo, (lo64, hi64));
            parts.push(self.n3(lo, NeonOp::Tbl, 8, table, index));
        }
        let r = match parts.as_slice() {
            [one] => *one,
            [x, y] => self.n3(lo, NeonOp::Orr, 8, *x, *y),
            _ => unreachable!("a shuffle picks at least one lane"),
        };
        self.nfinish(lo, inst, r);
    }
}
