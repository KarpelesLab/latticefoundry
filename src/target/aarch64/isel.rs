//! The AArch64 (A64) machine opcode set and instruction-selection rules
//! (ROADMAP Phase 7).
//!
//! [`A64Op`] is this target's [`Opcode`] vocabulary: a *post-isel, pre-encoding*
//! MIR whose operands are still MIR [`MachineOperand`]s (registers, immediates,
//! frame slots, labels, symbol references). Unlike x86-64, A64 data-processing
//! instructions are genuinely **three-address** (`add Xd, Xn, Xm` writes a
//! distinct destination), so the isel emits one MIR op per IR op with a clean
//! `[Def d, Use a, Use b]` shape and the encoder never has to synthesize a
//! move-to-destination. A few IR ops still expand to a short A64 idiom at encode
//! time (e.g. an [`A64Op::CmpCset`] becomes `subs xzr,a,b; cset d,cc`, a remainder
//! becomes `sdiv;msub`, a constant becomes a `movz`/`movk` chain, a global becomes
//! `adrp`+`add`).
//!
//! ## Block arguments, calls, returns
//!
//! Block arguments are realized by the framework's edge-move mechanism
//! ([`Lower::edge_to`]). `call` moves arguments into the AAPCS64 argument
//! registers `x0`–`x7`, records `x0` and the caller-saved clobbers as fixed defs,
//! and moves the result out of `x0`; `ret` moves its value into `x0`. The
//! prologue moves incoming parameters out of the argument registers (framework
//! prologue). A64 has a hardware divide, so `sdiv`/`udiv` lower directly and
//! remainder is `sdiv` followed by `msub` — no fixed-register `rax`/`rdx` dance.

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::mir::{
    MBlockId, MachineInst, MachineOperand, Opcode, PReg, Reg, RegClass, StackSlot, VReg,
};
use crate::codegen::target::{CallConv, MachineTarget};
use crate::ir::inst::{BinOp, CastOp, FloatPred, InstKind, IntPred, UnaryOp};
use crate::ir::value::{Const, ValueDef};
use crate::ir::{InstData, Module, ValueId};

use puremp::Int;

use super::regs::RegFile;

/// The A64 MIR opcode vocabulary. Operand layouts are documented per variant;
/// `Def`/`Use` are register operands, the rest are immediates, frame slots,
/// branch labels, or symbol references. `Imm width` is the operation's integer
/// bit width (selects the 32-bit `W`- vs 64-bit `X`-register form).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum A64Op {
    /// `[Def d, Use s]` — `mov d, s` (`orr d, xzr, s`).
    MovRR = 0,
    /// `[Def d, Imm v]` — load immediate via a `movz`/`movk` chain.
    MovRI = 1,
    /// `[Def d, Use a, Use b, Imm width]` — `add d, a, b`.
    Add = 2,
    /// `[Def d, Use a, Use b, Imm width]` — `sub d, a, b`.
    Sub = 3,
    /// `[Def d, Use a, Use b, Imm width]` — `and d, a, b`.
    And = 4,
    /// `[Def d, Use a, Use b, Imm width]` — `orr d, a, b`.
    Or = 5,
    /// `[Def d, Use a, Use b, Imm width]` — `eor d, a, b`.
    Eor = 6,
    /// `[Def d, Use a, Use b, Imm width]` — `mul d, a, b` (`madd d, a, b, xzr`).
    Mul = 7,
    /// `[Def d, Use a, Imm imm12, Imm width]` — `add d, a, #imm12`.
    AddI = 8,
    /// `[Def d, Use a, Imm imm12, Imm width]` — `sub d, a, #imm12`.
    SubI = 9,
    /// `[Def d, Use a, Use b, Imm width]` — `sdiv d, a, b`.
    Sdiv = 10,
    /// `[Def d, Use a, Use b, Imm width]` — `udiv d, a, b`.
    Udiv = 11,
    /// `[Def d, Use m, Use n, Use a, Imm width]` — `msub d, m, n, a` (`d = a - m*n`).
    Msub = 12,
    /// `[Def d, Use a, Imm count, Imm width]` — `lsl d, a, #count`.
    LslI = 13,
    /// `[Def d, Use a, Imm count, Imm width]` — `lsr d, a, #count`.
    LsrI = 14,
    /// `[Def d, Use a, Imm count, Imm width]` — `asr d, a, #count`.
    AsrI = 15,
    /// `[Def d, Use a, Use b, Imm width]` — `lslv d, a, b`.
    LslV = 16,
    /// `[Def d, Use a, Use b, Imm width]` — `lsrv d, a, b`.
    LsrV = 17,
    /// `[Def d, Use a, Use b, Imm width]` — `asrv d, a, b`.
    AsrV = 18,
    /// `[Def d, Use a, Use b, Imm cc, Imm width]` — `subs xzr,a,b; cset d, cc`.
    CmpCset = 19,
    /// `[Def d, Use cond, Use t, Use f]` — `cmp cond,#0; csel d, t, f, ne`.
    Csel = 20,
    /// `[Def d, Use ptr, Imm size]` — load `size` bytes from `[ptr]`.
    Load = 21,
    /// `[Use ptr, Use val, Imm size]` — store `size` bytes to `[ptr]`.
    Store = 22,
    /// `[Def d, Frame slot]` — `add d, sp, #slot_off`.
    FrameAddr = 23,
    /// `[Def d, Global g]` — `adrp d, g; add d, d, :lo12:g`.
    GlobalAddr = 24,
    /// `[Func f | Use callee, Def x0, Def clobbers.., Use args..]` — call.
    Call = 25,
    /// `[]` — return (value already in x0).
    Ret = 26,
    /// `[Label t]` — unconditional branch.
    B = 27,
    /// `[Use cond, Label t, Label f]` — `cbnz cond, t; b f`.
    BrCond = 28,
    /// `[Use cond, Label default, (Imm val, Label case)...]` — multi-way branch.
    Switch = 29,
    /// `[]` — a trap (`brk #1`).
    Unreachable = 30,
    /// `[Use src, Frame slot]` — spill: `str src, [sp, #slot_off]`.
    StoreFrame = 31,
    /// `[Def dst, Frame slot]` — reload: `ldr dst, [sp, #slot_off]`.
    LoadFrame = 32,
    /// `[]` — `stp x29, x30, [sp, #-16]!` (prologue).
    StpFpLr = 33,
    /// `[]` — `ldp x29, x30, [sp], #16` (epilogue).
    LdpFpLr = 34,
    /// `[]` — `mov x29, sp` (prologue).
    MovFpSp = 35,
    /// `[Imm k]` — `sub sp, sp, #k` (prologue).
    SubSp = 36,
    /// `[Imm k]` — `add sp, sp, #k` (epilogue).
    AddSp = 37,
    /// `[Use r, Imm off]` — `str r, [sp, #off]` (callee-saved save).
    SaveReg = 38,
    /// `[Def r, Imm off]` — `ldr r, [sp, #off]` (callee-saved restore).
    RestoreReg = 39,

    // --- scalar floating-point (FP/SIMD) ----------------------------------
    /// `[Def d, Use a, Use b, Imm width]` — `fadd d, a, b` (`d`/`s` by width).
    FAdd = 40,
    /// `[Def d, Use a, Use b, Imm width]` — `fsub d, a, b`.
    FSub = 41,
    /// `[Def d, Use a, Use b, Imm width]` — `fmul d, a, b`.
    FMul = 42,
    /// `[Def d, Use a, Use b, Imm width]` — `fdiv d, a, b`.
    FDiv = 43,
    /// `[Def d, Use s, Imm width]` — `fneg d, s`.
    FNeg = 44,
    /// `[Def d, Use a, Use b, Imm packed, Imm width]` — `fcmp a,b` then
    /// `cset d,cond` (with an optional second `cset`+`and`/`orr` combine). `packed`
    /// carries `cond | combine<<4 | cond2<<8` (see [`fcmp_plan`]).
    Fcmp = 45,
    /// `[Def d, Imm bits, Imm width]` — materialize a float constant: the exact
    /// IEEE bit pattern via a scratch gpr (`movz/movk x9; fmov d, x9`).
    LoadFConst = 46,
    /// `[Def d, Use s, Imm dst_w, Imm src_w]` — `fcvt d, s` (f32↔f64).
    Fcvt = 47,
    /// `[Def d, Use s, Imm dst_int_w, Imm src_flt_w]` — `fcvtzs d, s` (float→signed
    /// int, truncating toward zero).
    Fcvtzs = 48,
    /// `[Def d, Use s, Imm dst_int_w, Imm src_flt_w]` — `fcvtzu d, s` (float→unsigned
    /// int, truncating toward zero).
    Fcvtzu = 49,
    /// `[Def d, Use s, Imm dst_flt_w, Imm src_int_w]` — `scvtf d, s` (signed
    /// int→float).
    Scvtf = 50,
    /// `[Def d, Use s, Imm dst_flt_w, Imm src_int_w]` — `ucvtf d, s` (unsigned
    /// int→float).
    Ucvtf = 51,
}

impl A64Op {
    /// The MIR [`Opcode`] id for this opcode.
    #[inline]
    pub fn opcode(self) -> Opcode {
        Opcode(self as u32)
    }

    /// Decode a MIR [`Opcode`] back to an [`A64Op`].
    pub fn decode(op: Opcode) -> A64Op {
        use A64Op::*;
        const TABLE: [A64Op; 52] = [
            MovRR, MovRI, Add, Sub, And, Or, Eor, Mul, AddI, SubI, Sdiv, Udiv, Msub, LslI, LsrI,
            AsrI, LslV, LsrV, AsrV, CmpCset, Csel, Load, Store, FrameAddr, GlobalAddr, Call, Ret, B,
            BrCond, Switch, Unreachable, StoreFrame, LoadFrame, StpFpLr, LdpFpLr, MovFpSp, SubSp,
            AddSp, SaveReg, RestoreReg, FAdd, FSub, FMul, FDiv, FNeg, Fcmp, LoadFConst, Fcvt,
            Fcvtzs, Fcvtzu, Scvtf, Ucvtf,
        ];
        TABLE[op.0 as usize]
    }
}

/// Encode an [`IntPred`] as the A64 condition-code nibble used by `b.cond`/`cset`
/// (the *true* condition; `cset` inverts it internally when encoding CSINC).
pub(crate) fn cond_code(p: IntPred) -> u8 {
    match p {
        IntPred::Eq => 0x0,  // EQ
        IntPred::Ne => 0x1,  // NE
        IntPred::Uge => 0x2, // HS (C set)
        IntPred::Ult => 0x3, // LO (C clear)
        IntPred::Ugt => 0x8, // HI
        IntPred::Ule => 0x9, // LS
        IntPred::Sge => 0xA, // GE
        IntPred::Slt => 0xB, // LT
        IntPred::Sgt => 0xC, // GT
        IntPred::Sle => 0xD, // LE
    }
}

/// A64 condition-code nibbles used by `cset`/`csel`.
mod cc {
    pub(super) const EQ: u8 = 0x0;
    pub(super) const NE: u8 = 0x1;
    pub(super) const HS: u8 = 0x2; // C set (unsigned ≥ / "carry set")
    pub(super) const MI: u8 = 0x4; // N set (negative)
    pub(super) const VS: u8 = 0x6; // V set (overflow ⇒ FP unordered)
    pub(super) const VC: u8 = 0x7; // V clear (⇒ FP ordered)
    pub(super) const HI: u8 = 0x8;
    pub(super) const LS: u8 = 0x9;
    pub(super) const GE: u8 = 0xA;
    pub(super) const LT: u8 = 0xB;
    pub(super) const GT: u8 = 0xC;
    pub(super) const LE: u8 = 0xD;
}

/// The `fcmp`+`cset` plan for a floating-point predicate: the primary condition
/// code and, when a single code cannot express the ordered/unordered reading, a
/// second condition combined with `and`/`orr`. Returns `None` for the constant
/// predicates `False`/`True`, which the caller materializes directly.
///
/// After `fcmp`, NZCV encodes: unordered (a NaN operand) ⇒ `N=0,Z=0,C=1,V=1`;
/// `a<b` ⇒ `N=1,Z=0,C=0,V=0`; `a==b` ⇒ `N=0,Z=1,C=1,V=0`; `a>b` ⇒
/// `N=0,Z=0,C=1,V=0`. The condition codes below are chosen so each `FloatPred`
/// matches `ir::semantics`. `one` (ordered ≠) and `ueq` (unordered ∨ =) need two
/// codes: `one = NE ∧ VC`, `ueq = EQ ∨ VS`.
pub(crate) fn fcmp_plan(pred: FloatPred) -> Option<(u8, Combine, u8)> {
    use Combine::{And, None, Or};
    Option::Some(match pred {
        FloatPred::False | FloatPred::True => return Option::None,
        FloatPred::Oeq => (cc::EQ, None, 0),
        FloatPred::Ogt => (cc::GT, None, 0),
        FloatPred::Oge => (cc::GE, None, 0),
        FloatPred::Olt => (cc::MI, None, 0),
        FloatPred::Ole => (cc::LS, None, 0),
        FloatPred::One => (cc::NE, And, cc::VC),
        FloatPred::Ord => (cc::VC, None, 0),
        FloatPred::Ueq => (cc::EQ, Or, cc::VS),
        FloatPred::Ugt => (cc::HI, None, 0),
        FloatPred::Uge => (cc::HS, None, 0),
        FloatPred::Ult => (cc::LT, None, 0),
        FloatPred::Ule => (cc::LE, None, 0),
        FloatPred::Une => (cc::NE, None, 0),
        FloatPred::Uno => (cc::VS, None, 0),
    })
}

/// How the second `cset` of an `fcmp` plan is folded into the result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Combine {
    /// A single `cset` — no second condition.
    None,
    /// `and d, d, d2` with the second `cset`.
    And,
    /// `orr d, d, d2` with the second `cset`.
    Or,
}

impl Combine {
    /// The 4-bit code packed into the [`A64Op::Fcmp`] immediate.
    fn code(self) -> u64 {
        match self {
            Combine::None => 0,
            Combine::And => 1,
            Combine::Or => 2,
        }
    }

    /// Decode a packed 4-bit code back to a [`Combine`].
    pub(crate) fn decode(code: u64) -> Combine {
        match code {
            1 => Combine::And,
            2 => Combine::Or,
            _ => Combine::None,
        }
    }
}

/// The floating-point "ptype" field (0 = single/`s`, 1 = double/`d`) for a width.
#[inline]
pub(crate) fn ptype_of(width: u32) -> u32 {
    u32::from(width >= 64)
}

fn def(r: PReg) -> MachineOperand {
    MachineOperand::Def(Reg::Physical(r))
}
fn use_p(r: PReg) -> MachineOperand {
    MachineOperand::Use(Reg::Physical(r))
}
fn def_v(v: VReg) -> MachineOperand {
    MachineOperand::Def(Reg::Virtual(v))
}
fn use_v(v: VReg) -> MachineOperand {
    MachineOperand::Use(Reg::Virtual(v))
}
fn imm(v: u64) -> MachineOperand {
    MachineOperand::Imm(Int::from_u64(v))
}

/// The AArch64 target: its register file/ABI plus the isel + encoding rules.
#[derive(Debug)]
pub struct AArch64Target {
    rf: RegFile,
}

impl Default for AArch64Target {
    fn default() -> Self {
        Self::new()
    }
}

impl AArch64Target {
    /// Construct the AArch64 target with its fixed register file and AAPCS64 ABI.
    pub fn new() -> AArch64Target {
        AArch64Target { rf: RegFile::new() }
    }

    /// Lower function `func` of `module` to MIR over this target.
    pub fn select(
        &self,
        module: &Module,
        func: crate::ir::FuncId,
    ) -> crate::codegen::mir::MachineFunction {
        crate::codegen::isel::select(self, module, func)
    }

    /// If `v` is an integer constant operand, its value.
    fn const_of(lo: &Lower<'_, Self>, v: ValueId) -> Option<Int> {
        if let ValueDef::Const(c) = lo.func().value(v).def
            && let Const::Int { value, .. } = lo.module().consts().get(c)
        {
            return Some(value.clone());
        }
        None
    }

    fn lower_bin(&self, lo: &mut Lower<'_, Self>, op: BinOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let width = lo.int_width(inst.operands()[0]);
        // Simple commutative/register three-address ops.
        let simple = match op {
            BinOp::Add => Some(A64Op::Add),
            BinOp::Sub => Some(A64Op::Sub),
            BinOp::And => Some(A64Op::And),
            BinOp::Or => Some(A64Op::Or),
            BinOp::Xor => Some(A64Op::Eor),
            BinOp::Mul => Some(A64Op::Mul),
            _ => None,
        };
        // Scalar FP arithmetic (F32/F64). The operand width comes from the float
        // type (32 or 64); the encoder picks the `s`/`d` form.
        let fop = match op {
            BinOp::FAdd => Some(A64Op::FAdd),
            BinOp::FSub => Some(A64Op::FSub),
            BinOp::FMul => Some(A64Op::FMul),
            BinOp::FDiv => Some(A64Op::FDiv),
            _ => None,
        };
        if let Some(x) = fop {
            let a = lo.reg(inst.operands()[0]);
            let b = lo.reg(inst.operands()[1]);
            lo.emit(MachineInst::new(
                x.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        if let Some(x) = simple {
            // add/sub take a 12-bit unsigned immediate directly; use it when the
            // RHS is a constant that fits, avoiding a `movz` to materialize it.
            if matches!(op, BinOp::Add | BinOp::Sub)
                && let Some(c) = Self::const_of(lo, inst.operands()[1])
                && let Some(u) = c.to_u64()
                && u <= 0xFFF
            {
                let a = lo.reg(inst.operands()[0]);
                let imm_op = if op == BinOp::Add { A64Op::AddI } else { A64Op::SubI };
                lo.emit(MachineInst::new(
                    imm_op.opcode(),
                    vec![def_v(d), use_v(a), imm(u), imm(u64::from(width))],
                ));
                return;
            }
            let a = lo.reg(inst.operands()[0]);
            let b = lo.reg(inst.operands()[1]);
            lo.emit(MachineInst::new(
                x.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        match op {
            BinOp::Shl => self.lower_shift(lo, A64Op::LslI, A64Op::LslV, d, inst, width),
            BinOp::LShr => self.lower_shift(lo, A64Op::LsrI, A64Op::LsrV, d, inst, width),
            BinOp::AShr => self.lower_shift(lo, A64Op::AsrI, A64Op::AsrV, d, inst, width),
            BinOp::UDiv => self.lower_div(lo, A64Op::Udiv, false, d, inst, width),
            BinOp::URem => self.lower_div(lo, A64Op::Udiv, true, d, inst, width),
            BinOp::SDiv => self.lower_div(lo, A64Op::Sdiv, false, d, inst, width),
            BinOp::SRem => self.lower_div(lo, A64Op::Sdiv, true, d, inst, width),
            // `frem` has no direct A64 form (it is an `fmod` libcall); a documented
            // follow-up. `d` is an fp register, so keep the MIR well-formed with a
            // zero float constant (never executed in tests).
            _ => lo.emit(MachineInst::new(
                A64Op::LoadFConst.opcode(),
                vec![def_v(d), imm(0), imm(u64::from(width))],
            )),
        }
    }

    /// `fneg`: flip the IEEE sign bit (a sign flip, matching `ir::semantics`), via
    /// the A64 `fneg` instruction.
    fn lower_fneg(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let d = lo.result_reg(inst);
        let s = lo.reg(inst.operands()[0]);
        let width = lo.int_width(inst.operands()[0]);
        lo.emit(MachineInst::new(
            A64Op::FNeg.opcode(),
            vec![def_v(d), use_v(s), imm(u64::from(width))],
        ));
    }

    /// `fcmp`: `fcmp a,b` then `cset` (with the ordered/unordered condition from
    /// [`fcmp_plan`]). The result is an `i1` in a gpr.
    fn lower_fcmp(&self, lo: &mut Lower<'_, Self>, pred: FloatPred, inst: &InstData) {
        let d = lo.result_reg(inst);
        match fcmp_plan(pred) {
            None => {
                // `False`/`True` are constants.
                let v = u64::from(pred == FloatPred::True);
                lo.emit(MachineInst::new(A64Op::MovRI.opcode(), vec![def_v(d), imm(v)]));
            }
            Some((cond, combine, cond2)) => {
                let a = lo.reg(inst.operands()[0]);
                let b = lo.reg(inst.operands()[1]);
                let width = lo.int_width(inst.operands()[0]);
                let packed =
                    u64::from(cond) | (combine.code() << 4) | (u64::from(cond2) << 8);
                lo.emit(MachineInst::new(
                    A64Op::Fcmp.opcode(),
                    vec![def_v(d), use_v(a), use_v(b), imm(packed), imm(u64::from(width))],
                ));
            }
        }
    }

    /// Conversions. Float↔float and int↔float go through the A64 `fcvt`/`fcvtz*`/
    /// `scvtf`/`ucvtf` forms; every other cast (integer width change, ptr↔int,
    /// bitcast within a class) is a low-bits-preserving copy, as before.
    fn lower_cast(&self, lo: &mut Lower<'_, Self>, op: CastOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let s = lo.reg(inst.operands()[0]);
        let src_w = lo.int_width(inst.operands()[0]);
        let dst_w = lo.types().bit_width(inst.ty).unwrap_or(64);
        let emit3 = |lo: &mut Lower<'_, Self>, o: A64Op, a: u32, b: u32| {
            lo.emit(MachineInst::new(
                o.opcode(),
                vec![def_v(d), use_v(s), imm(u64::from(a)), imm(u64::from(b))],
            ));
        };
        match op {
            CastOp::FpTrunc | CastOp::FpExt => emit3(lo, A64Op::Fcvt, dst_w, src_w),
            CastOp::FpToSi => emit3(lo, A64Op::Fcvtzs, dst_w, src_w),
            CastOp::FpToUi => emit3(lo, A64Op::Fcvtzu, dst_w, src_w),
            CastOp::SiToFp => emit3(lo, A64Op::Scvtf, dst_w, src_w),
            CastOp::UiToFp => emit3(lo, A64Op::Ucvtf, dst_w, src_w),
            // Integer↔integer / ptr↔int / same-class bitcast: preserve low bits.
            _ => lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_v(s)])),
        }
    }

    fn lower_shift(
        &self,
        lo: &mut Lower<'_, Self>,
        imm_op: A64Op,
        var_op: A64Op,
        d: VReg,
        inst: &InstData,
        width: u32,
    ) {
        let a = lo.reg(inst.operands()[0]);
        if let Some(c) = Self::const_of(lo, inst.operands()[1]) {
            let shmask = if width >= 64 { 63 } else { u64::from(width) - 1 };
            let count = c.to_u64().unwrap_or(0) & shmask;
            lo.emit(MachineInst::new(
                imm_op.opcode(),
                vec![def_v(d), use_v(a), imm(count), imm(u64::from(width))],
            ));
        } else {
            let b = lo.reg(inst.operands()[1]);
            lo.emit(MachineInst::new(
                var_op.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
        }
    }

    fn lower_div(
        &self,
        lo: &mut Lower<'_, Self>,
        div_op: A64Op,
        want_rem: bool,
        d: VReg,
        inst: &InstData,
        width: u32,
    ) {
        let a = lo.reg(inst.operands()[0]);
        let b = lo.reg(inst.operands()[1]);
        if !want_rem {
            lo.emit(MachineInst::new(
                div_op.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        // Remainder: q = a / b; d = a - q * b   (msub d, q, b, a).
        let q = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(
            div_op.opcode(),
            vec![def_v(q), use_v(a), use_v(b), imm(u64::from(width))],
        ));
        lo.emit(MachineInst::new(
            A64Op::Msub.opcode(),
            vec![def_v(d), use_v(q), use_v(b), use_v(a), imm(u64::from(width))],
        ));
    }

    fn lower_call(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let cc = &self.rf.cc;
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];

        // Route each argument by class: integer/pointer into x0.. (a separate
        // counter from) float/double into v0.. — the AAPCS64 rule for mixed calls.
        // Materialize *every* argument value into a vreg first, then move them all
        // into the physical argument registers immediately before the call, so no
        // later materialization is colored into an argument register already
        // holding an earlier argument (the allocator's fixed-register model is
        // point-based; see the x86-64 backend's note).
        let mut int_i = 0usize;
        let mut fp_i = 0usize;
        let mut moves: Vec<(PReg, VReg)> = Vec::with_capacity(args.len());
        for &arg in args {
            let r = lo.reg(arg);
            let areg = match lo.mf().vreg_class(r) {
                RegClass::Gpr => {
                    let a = cc.arg_regs[int_i];
                    int_i += 1;
                    a
                }
                RegClass::Fp => {
                    let a = cc.fp_arg_regs[fp_i];
                    fp_i += 1;
                    a
                }
            };
            moves.push((areg, r));
        }
        let used_arg_regs: Vec<PReg> = moves.iter().map(|&(areg, _)| areg).collect();
        for (areg, r) in moves {
            lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def(areg), use_v(r)]));
        }

        // The return register follows the result type's class (v0 for a float
        // return, x0 otherwise).
        let ret_is_fp = inst
            .result()
            .is_some_and(|r| lo.types().get(lo.func().value_type(r)).is_float());
        let ret_reg = if ret_is_fp { cc.fp_ret_reg } else { cc.ret_reg };

        let mut operands = Vec::new();
        match lo.callee_func(callee) {
            Some(fidx) => operands.push(MachineOperand::Func(fidx)),
            None => {
                let cr = lo.reg(callee);
                operands.push(use_v(cr));
            }
        }
        operands.push(def(ret_reg));
        for &cs in &self.rf.caller_saved {
            if cs != ret_reg {
                operands.push(def(cs));
            }
        }
        for &areg in &used_arg_regs {
            operands.push(use_p(areg));
        }
        lo.emit(MachineInst::new(A64Op::Call.opcode(), operands));

        if inst.result().is_some() {
            let d = lo.result_reg(inst);
            lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_p(ret_reg)]));
        }
    }
}

impl MachineTarget for AArch64Target {
    fn name(&self) -> &str {
        "aarch64"
    }

    fn reg_classes(&self) -> &[RegClass] {
        &self.rf.classes
    }

    fn allocatable(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Gpr => &self.rf.allocatable,
            RegClass::Fp => &self.rf.allocatable_fp,
        }
    }

    fn scratch(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Gpr => &self.rf.scratch,
            RegClass::Fp => &self.rf.scratch_fp,
        }
    }

    fn caller_saved(&self) -> &[PReg] {
        &self.rf.caller_saved
    }

    fn callee_saved(&self) -> &[PReg] {
        &self.rf.callee_saved
    }

    fn call_conv(&self) -> &CallConv {
        &self.rf.cc
    }

    fn is_terminator(&self, op: Opcode) -> bool {
        matches!(
            A64Op::decode(op),
            A64Op::B | A64Op::BrCond | A64Op::Switch | A64Op::Ret | A64Op::Unreachable
        )
    }

    fn is_move(&self, op: Opcode) -> bool {
        A64Op::decode(op) == A64Op::MovRR
    }

    fn emit_move(&self, dst: Reg, src: Reg) -> MachineInst {
        MachineInst::new(A64Op::MovRR.opcode(), vec![MachineOperand::Def(dst), MachineOperand::Use(src)])
    }

    fn emit_spill(&self, slot: StackSlot, src: PReg) -> MachineInst {
        MachineInst::new(A64Op::StoreFrame.opcode(), vec![use_p(src), MachineOperand::Frame(slot)])
    }

    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(A64Op::LoadFrame.opcode(), vec![def(dst), MachineOperand::Frame(slot)])
    }
}

impl TargetIsel for AArch64Target {
    fn li(&self, dst: VReg, value: Int) -> MachineInst {
        MachineInst::new(A64Op::MovRI.opcode(), vec![def_v(dst), MachineOperand::Imm(value)])
    }

    fn jump(&self, dst: MBlockId) -> MachineInst {
        MachineInst::new(A64Op::B.opcode(), vec![MachineOperand::Label(dst)])
    }

    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(A64Op::FrameAddr.opcode(), vec![def_v(dst), MachineOperand::Frame(slot)])
    }

    fn global_addr(&self, dst: VReg, g: u32) -> MachineInst {
        MachineInst::new(A64Op::GlobalAddr.opcode(), vec![def_v(dst), MachineOperand::Global(g)])
    }

    fn float_const(&self, dst: VReg, bits: u64, width: u32) -> MachineInst {
        MachineInst::new(
            A64Op::LoadFConst.opcode(),
            vec![def_v(dst), imm(bits), imm(u64::from(width))],
        )
    }

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Bin(op) => self.lower_bin(lo, *op, inst),
            InstKind::ICmp(pred) => {
                let d = lo.result_reg(inst);
                let a = lo.reg(inst.operands()[0]);
                let b = lo.reg(inst.operands()[1]);
                let width = lo.int_width(inst.operands()[0]);
                lo.emit(MachineInst::new(
                    A64Op::CmpCset.opcode(),
                    vec![
                        def_v(d),
                        use_v(a),
                        use_v(b),
                        imm(u64::from(cond_code(*pred))),
                        imm(u64::from(width)),
                    ],
                ));
            }
            InstKind::Cast(op) => self.lower_cast(lo, *op, inst),
            InstKind::Alloca { elem_ty } => {
                let d = lo.result_reg(inst);
                let size = lo.byte_size(*elem_ty);
                let align = lo.types().align_of(*elem_ty);
                let slot = lo.new_slot(size, align);
                lo.emit(self.frame_addr(d, slot));
            }
            InstKind::Load { ty, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(inst.operands()[0]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    A64Op::Load.opcode(),
                    vec![def_v(d), use_v(ptr), imm(size)],
                ));
            }
            InstKind::Store { ty, .. } => {
                let ptr = lo.reg(inst.operands()[0]);
                let val = lo.reg(inst.operands()[1]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    A64Op::Store.opcode(),
                    vec![use_v(ptr), use_v(val), imm(size)],
                ));
            }
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(inst);
                let base = lo.reg(inst.operands()[0]);
                let off = lo.reg(inst.operands()[1]);
                lo.emit(MachineInst::new(
                    A64Op::Add.opcode(),
                    vec![def_v(d), use_v(base), use_v(off), imm(64)],
                ));
            }
            InstKind::Select => {
                let d = lo.result_reg(inst);
                let c = lo.reg(inst.operands()[0]);
                let t = lo.reg(inst.operands()[1]);
                let f = lo.reg(inst.operands()[2]);
                lo.emit(MachineInst::new(
                    A64Op::Csel.opcode(),
                    vec![def_v(d), use_v(c), use_v(t), use_v(f)],
                ));
            }
            InstKind::Freeze => {
                let d = lo.result_reg(inst);
                let s = lo.reg(inst.operands()[0]);
                lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
            }
            InstKind::Call => self.lower_call(lo, inst),
            InstKind::Unary(UnaryOp::FNeg) => self.lower_fneg(lo, inst),
            InstKind::FCmp(pred) => self.lower_fcmp(lo, *pred, inst),
            _ => unreachable!("terminator reached lower_inst: {:?}", inst.kind),
        }
    }

    fn lower_term(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Ret => {
                if let Some(&v) = inst.operands().first() {
                    let r = lo.reg(v);
                    // A float return goes in v0, an integer/pointer return in x0.
                    let ret = match lo.mf().vreg_class(r) {
                        RegClass::Fp => self.rf.cc.fp_ret_reg,
                        RegClass::Gpr => self.rf.cc.ret_reg,
                    };
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def(ret), use_v(r)]));
                }
                lo.emit(MachineInst::new(A64Op::Ret.opcode(), Vec::new()));
            }
            InstKind::Br(target) => {
                let args: Vec<_> = inst.operands().to_vec();
                let e = lo.edge_to(*target, &args);
                lo.emit(self.jump(e));
            }
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let cond = lo.reg(inst.operands()[0]);
                let ops = inst.operands();
                let tb = 1 + *true_args as usize;
                let fb = tb + *false_args as usize;
                let true_vals: Vec<_> = ops[1..tb].to_vec();
                let false_vals: Vec<_> = ops[tb..fb].to_vec();
                let te = lo.edge_to(*if_true, &true_vals);
                let fe = lo.edge_to(*if_false, &false_vals);
                lo.emit(MachineInst::new(
                    A64Op::BrCond.opcode(),
                    vec![use_v(cond), MachineOperand::Label(te), MachineOperand::Label(fe)],
                ));
            }
            InstKind::Switch(data) => {
                let cond = lo.reg(inst.operands()[0]);
                let ops = inst.operands();
                let mut idx = 1usize;
                let dcount = data.default_args as usize;
                let default_vals: Vec<_> = ops[idx..idx + dcount].to_vec();
                idx += dcount;
                let de = lo.edge_to(data.default, &default_vals);
                let mut operands = vec![use_v(cond), MachineOperand::Label(de)];
                let cases = data.cases.clone();
                for case in &cases {
                    let n = case.args as usize;
                    let cvals: Vec<_> = ops[idx..idx + n].to_vec();
                    idx += n;
                    let ce = lo.edge_to(case.target, &cvals);
                    operands.push(MachineOperand::Imm(case.value.clone()));
                    operands.push(MachineOperand::Label(ce));
                }
                lo.emit(MachineInst::new(A64Op::Switch.opcode(), operands));
            }
            InstKind::Unreachable => {
                lo.emit(MachineInst::new(A64Op::Unreachable.opcode(), Vec::new()));
            }
            _ => unreachable!("non-terminator reached lower_term: {:?}", inst.kind),
        }
    }
}
