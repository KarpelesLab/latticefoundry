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
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ValueDef};
use crate::ir::{InstData, Module, ValueId};

use puremp::Int;

use super::regs::{self, RegFile};

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

    // --- aggregate (by-value struct) ABI support --------------------------
    /// `[Def d, Imm off]` — `add d, sp, #off` (unsigned `off`). Materializes an
    /// address in the reserved outgoing-argument area at the bottom of the frame
    /// (`sp` is constant after the prologue), for stack-passed call arguments.
    LeaSpOff = 52,
    /// `[Def d, Imm off]` — `add d, x29, #off` (unsigned `off`). Materializes an
    /// address relative to the frame pointer: used to address an incoming
    /// stack-passed parameter's home (`[x29 + 16 + k]`, above the saved
    /// frame-pointer/link-register pair).
    LeaFpOff = 53,
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
        const TABLE: [A64Op; 54] = [
            MovRR, MovRI, Add, Sub, And, Or, Eor, Mul, AddI, SubI, Sdiv, Udiv, Msub, LslI, LsrI,
            AsrI, LslV, LsrV, AsrV, CmpCset, Csel, Load, Store, FrameAddr, GlobalAddr, Call, Ret, B,
            BrCond, Switch, Unreachable, StoreFrame, LoadFrame, StpFpLr, LdpFpLr, MovFpSp, SubSp,
            AddSp, SaveReg, RestoreReg, FAdd, FSub, FMul, FDiv, FNeg, Fcmp, LoadFConst, Fcvt,
            Fcvtzs, Fcvtzu, Scvtf, Ucvtf, LeaSpOff, LeaFpOff,
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

// ===========================================================================
// AAPCS64 aggregate classification
// ===========================================================================

/// How an aggregate (a struct/array passed or returned by value) crosses the
/// AAPCS64 ABI. The three cases are mutually exclusive and computed by
/// [`classify_aggregate`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum AbiClass {
    /// A Homogeneous Floating-point Aggregate: `1..=4` elements that are all the
    /// same floating-point type. Passed one element per consecutive SIMD/FP
    /// register (`v0..v7` for arguments; `v0..v3` for a return). `width` is the
    /// element bit width (32 or 64); `count` the flattened element count.
    Hfa { width: u32, count: u32 },
    /// A small aggregate (`≤ 16` bytes, not an HFA): passed as `len` consecutive
    /// general-register eightbytes (`x0..x7` for arguments; `x0`/`x1` for a
    /// return). `len` is 0, 1, or 2.
    Regs(usize),
    /// A large aggregate (`> 16` bytes, not an HFA): passed **by reference** to a
    /// caller-made copy (a pointer in the next general register); returned
    /// through an **indirect result** pointer supplied by the caller in `x8`.
    Reference,
}

/// Whether a type is an aggregate this backend represents, at the codegen level,
/// by a pointer to its in-memory storage.
fn is_aggregate(types: &TypeContext, ty: TypeId) -> bool {
    matches!(types.get(ty), Type::Struct(_) | Type::Array(_, _))
}

/// If every leaf of `ty` is a floating-point value of one identical width, that
/// width and the flattened leaf count; `None` if any leaf is not a float or the
/// float widths differ. (An HFA additionally requires `1..=4` leaves — the
/// caller enforces that bound.)
fn homogeneous_float(types: &TypeContext, ty: TypeId) -> Option<(u32, u32)> {
    fn walk(types: &TypeContext, ty: TypeId, width: &mut Option<u32>, count: &mut u32) -> bool {
        match types.get(ty) {
            Type::Float(k) => {
                let bw = k.bit_width();
                match *width {
                    None => *width = Some(bw),
                    Some(x) if x == bw => {}
                    Some(_) => return false,
                }
                *count += 1;
                true
            }
            Type::Struct(fields) => {
                let n = fields.len();
                (0..n).all(|i| {
                    let (_, fty) = types.field_offset(ty, i as u32);
                    walk(types, fty, width, count)
                })
            }
            Type::Array(elem, len) => {
                let (elem, len) = (*elem, *len);
                (0..len).all(|_| walk(types, elem, width, count))
            }
            _ => false,
        }
    }
    let (mut width, mut count) = (None, 0u32);
    if walk(types, ty, &mut width, &mut count) {
        width.map(|w| (w, count))
    } else {
        None
    }
}

/// Classify an aggregate `ty` under AAPCS64: an HFA (in FP registers), a small
/// aggregate (in general registers), or a large one (by reference / `x8`).
pub(crate) fn classify_aggregate(types: &TypeContext, ty: TypeId) -> AbiClass {
    if let Some((width, count)) = homogeneous_float(types, ty)
        && (1..=4).contains(&count)
    {
        return AbiClass::Hfa { width, count };
    }
    let size = types.size_of(ty);
    if size == 0 {
        return AbiClass::Regs(0);
    }
    if size > 16 {
        return AbiClass::Reference;
    }
    AbiClass::Regs(size.div_ceil(8) as usize)
}

/// Round `v` up to a multiple of `align` (a power of two ≥ 1).
fn align_up_u64(v: u64, align: u64) -> u64 {
    let a = align.max(1);
    v.div_ceil(a) * a
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

    /// Materialize `base + off` (a byte displacement) into a fresh GPR, or return
    /// `base` unchanged when `off == 0`. Uses the `add #imm12` form for a small
    /// offset, otherwise a `movz`-materialized register add.
    fn add_off(&self, lo: &mut Lower<'_, Self>, base: VReg, off: u64) -> VReg {
        if off == 0 {
            return base;
        }
        let d = lo.fresh_vreg(RegClass::Gpr);
        if off <= 0xFFF {
            lo.emit(MachineInst::new(
                A64Op::AddI.opcode(),
                vec![def_v(d), use_v(base), imm(off), imm(64)],
            ));
        } else {
            let k = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(MachineInst::new(A64Op::MovRI.opcode(), vec![def_v(k), imm(off)]));
            lo.emit(MachineInst::new(
                A64Op::Add.opcode(),
                vec![def_v(d), use_v(base), use_v(k), imm(64)],
            ));
        }
        d
    }

    /// Emit `add d, sp, #off` into a fresh GPR (addresses the outgoing
    /// stack-argument area at the bottom of the frame).
    fn lea_sp(&self, lo: &mut Lower<'_, Self>, off: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(A64Op::LeaSpOff.opcode(), vec![def_v(d), imm(off)]));
        d
    }

    /// Emit `add d, x29, #off` into a fresh GPR (addresses an incoming
    /// stack-passed parameter, above the saved fp/lr pair).
    fn lea_fp(&self, lo: &mut Lower<'_, Self>, off: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(A64Op::LeaFpOff.opcode(), vec![def_v(d), imm(off)]));
        d
    }

    /// Copy `size` bytes from `[src]` to `[dst]` (both GPR pointer vregs) in
    /// 8/4/2/1-byte chunks via a scratch GPR.
    fn emit_memcpy(&self, lo: &mut Lower<'_, Self>, dst: VReg, src: VReg, size: u64) {
        let mut o = 0u64;
        while o < size {
            let chunk = if size - o >= 8 {
                8
            } else if size - o >= 4 {
                4
            } else if size - o >= 2 {
                2
            } else {
                1
            };
            let sp = self.add_off(lo, src, o);
            let t = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(MachineInst::new(A64Op::Load.opcode(), vec![def_v(t), use_v(sp), imm(chunk)]));
            let dp = self.add_off(lo, dst, o);
            lo.emit(MachineInst::new(A64Op::Store.opcode(), vec![use_v(dp), use_v(t), imm(chunk)]));
            o += chunk;
        }
    }

    /// Copy an aggregate `arg` into the outgoing stack area at `stack_off` and
    /// return the new running stack offset (used when the argument registers of
    /// its bank are exhausted).
    fn arg_on_stack(&self, lo: &mut Lower<'_, Self>, arg: ValueId, ty: TypeId, stack_off: u64) -> u64 {
        let size = lo.byte_size(ty);
        let align = lo.types().align_of(ty).max(8);
        let at = align_up_u64(stack_off, align);
        let src = lo.reg(arg);
        let dst = self.lea_sp(lo, at);
        self.emit_memcpy(lo, dst, src, size);
        at + align_up_u64(size, 8)
    }

    /// Lower a `call`, implementing the AAPCS64 ABI for by-value struct arguments
    /// and returns on top of the existing scalar/float handling.
    ///
    /// A struct value is represented, at this codegen level, by a GPR vreg holding
    /// a pointer to the struct's in-memory storage. An HFA argument is loaded
    /// element-by-element into consecutive `v` registers; a small (`≤16`-byte)
    /// aggregate is loaded eightbyte-by-eightbyte into consecutive `x` registers;
    /// a large aggregate is copied into a fresh caller stack slot and passed by a
    /// pointer. An HFA result comes back in `v0..v3`, a small result in `x0`/`x1`,
    /// and a large result through the caller-allocated indirect-result slot whose
    /// address is passed in `x8`.
    fn lower_call(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let cc = &self.rf.cc;
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];

        // Return classification.
        let ret_ty = inst.result().map(|r| lo.func().value_type(r));
        let ret_agg = ret_ty.filter(|&t| is_aggregate(lo.types(), t));
        let ret_class = ret_agg.map(|t| classify_aggregate(lo.types(), t));
        let indirect_ret = matches!(ret_class, Some(AbiClass::Reference));

        // The final `arg-reg <- value-vreg` moves, emitted as one consecutive run
        // right before the `call` so no competing vreg definition sits in the gap
        // between an argument register's write and the call (the allocator's
        // fixed-register liveness reasons point-to-point).
        let mut reg_moves: Vec<(PReg, VReg)> = Vec::new();
        let mut int_i = 0usize;
        let mut fp_i = 0usize;
        let mut stack_off = 0u64;

        // A by-reference return: allocate the indirect-result slot and pass its
        // address in `x8` (a register outside the ordinary argument banks).
        let mut ret_slot = None;
        if indirect_ret {
            let t = ret_agg.unwrap();
            let size = align_up_u64(lo.byte_size(t).max(8), 8);
            let align = lo.types().align_of(t).max(8);
            let slot = lo.new_slot(size, align);
            ret_slot = Some(slot);
            let ptr = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(self.frame_addr(ptr, slot));
            reg_moves.push((regs::gpr(regs::X8), ptr));
        }

        for &arg in args {
            let ty = lo.func().value_type(arg);
            if is_aggregate(lo.types(), ty) {
                match classify_aggregate(lo.types(), ty) {
                    AbiClass::Hfa { width, count } => {
                        if fp_i + count as usize <= cc.fp_arg_regs.len() {
                            let ptr = lo.reg(arg);
                            let bytes = u64::from(width / 8);
                            for k in 0..count as usize {
                                let sp = self.add_off(lo, ptr, bytes * k as u64);
                                let d = lo.fresh_vreg(RegClass::Fp);
                                lo.emit(MachineInst::new(
                                    A64Op::Load.opcode(),
                                    vec![def_v(d), use_v(sp), imm(bytes)],
                                ));
                                let areg = cc.fp_arg_regs[fp_i];
                                fp_i += 1;
                                reg_moves.push((areg, d));
                            }
                            continue;
                        }
                        stack_off = self.arg_on_stack(lo, arg, ty, stack_off);
                    }
                    AbiClass::Regs(n) => {
                        if int_i + n <= cc.arg_regs.len() {
                            if n > 0 {
                                let ptr = lo.reg(arg);
                                for k in 0..n {
                                    let sp = self.add_off(lo, ptr, 8 * k as u64);
                                    let d = lo.fresh_vreg(RegClass::Gpr);
                                    lo.emit(MachineInst::new(
                                        A64Op::Load.opcode(),
                                        vec![def_v(d), use_v(sp), imm(8)],
                                    ));
                                    let areg = cc.arg_regs[int_i];
                                    int_i += 1;
                                    reg_moves.push((areg, d));
                                }
                            }
                            continue;
                        }
                        stack_off = self.arg_on_stack(lo, arg, ty, stack_off);
                    }
                    AbiClass::Reference => {
                        // Copy the struct into a fresh caller stack slot and pass a
                        // pointer to that copy (the callee only sees the pointer).
                        let size = lo.byte_size(ty);
                        let align = lo.types().align_of(ty).max(8);
                        let slot = lo.new_slot(align_up_u64(size.max(8), 8), align);
                        let dst = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(self.frame_addr(dst, slot));
                        let src = lo.reg(arg);
                        self.emit_memcpy(lo, dst, src, size);
                        let p = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(self.frame_addr(p, slot));
                        if int_i < cc.arg_regs.len() {
                            let areg = cc.arg_regs[int_i];
                            int_i += 1;
                            reg_moves.push((areg, p));
                        } else {
                            let dp = self.lea_sp(lo, stack_off);
                            lo.emit(MachineInst::new(
                                A64Op::Store.opcode(),
                                vec![use_v(dp), use_v(p), imm(8)],
                            ));
                            stack_off += 8;
                        }
                    }
                }
            } else {
                // Scalar / pointer / float argument.
                let v = lo.reg(arg);
                let is_fp = lo.mf().vreg_class(v) == RegClass::Fp;
                let has_reg =
                    if is_fp { fp_i < cc.fp_arg_regs.len() } else { int_i < cc.arg_regs.len() };
                if has_reg {
                    let areg = if is_fp {
                        let a = cc.fp_arg_regs[fp_i];
                        fp_i += 1;
                        a
                    } else {
                        let a = cc.arg_regs[int_i];
                        int_i += 1;
                        a
                    };
                    reg_moves.push((areg, v));
                } else {
                    let sz = lo.byte_size(ty);
                    let dp = self.lea_sp(lo, stack_off);
                    lo.emit(MachineInst::new(
                        A64Op::Store.opcode(),
                        vec![use_v(dp), use_v(v), imm(sz)],
                    ));
                    stack_off += 8;
                }
            }
        }
        if stack_off > 0 {
            lo.reserve_outgoing(align_up_u64(stack_off, 16));
        }

        let used_arg_regs: Vec<PReg> = reg_moves.iter().map(|&(areg, _)| areg).collect();
        for (areg, r) in reg_moves {
            lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def(areg), use_v(r)]));
        }

        // The primary return register (`x0`/`v0`); struct results reclaim their
        // registers (`x0`/`x1` or `v0..v3`), all covered by the clobber set.
        let ret_is_fp = ret_ty.is_some_and(|t| lo.types().get(t).is_float());
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

        match &ret_class {
            Some(AbiClass::Reference) => {
                // The result already sits in the caller-allocated indirect slot.
                let d = lo.result_reg(inst);
                lo.emit(self.frame_addr(d, ret_slot.unwrap()));
            }
            Some(AbiClass::Hfa { width, count }) => {
                let (width, count) = (*width, *count);
                let bytes = u64::from(width / 8);
                // Rescue each returned element from `v0..v3` (one consecutive run
                // right after the call), then store them into a fresh result slot.
                let mut saved: Vec<VReg> = Vec::with_capacity(count as usize);
                for k in 0..count as usize {
                    let r = regs::fp(k as u16);
                    let v = lo.fresh_vreg(RegClass::Fp);
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(v), use_p(r)]));
                    saved.push(v);
                }
                let t = ret_agg.unwrap();
                let size = align_up_u64(lo.byte_size(t).max(8), 8);
                let align = lo.types().align_of(t).max(8);
                let slot = lo.new_slot(size, align);
                let d = lo.result_reg(inst);
                lo.emit(self.frame_addr(d, slot));
                for (k, v) in saved.into_iter().enumerate() {
                    let dp = self.add_off(lo, d, bytes * k as u64);
                    lo.emit(MachineInst::new(
                        A64Op::Store.opcode(),
                        vec![use_v(dp), use_v(v), imm(bytes)],
                    ));
                }
            }
            Some(AbiClass::Regs(n)) => {
                let n = *n;
                let mut saved: Vec<VReg> = Vec::with_capacity(n);
                for k in 0..n {
                    let r = if k == 0 { cc.ret_reg } else { regs::gpr(regs::X1) };
                    let v = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(v), use_p(r)]));
                    saved.push(v);
                }
                let t = ret_agg.unwrap();
                let size = align_up_u64(lo.byte_size(t).max(8), 8);
                let align = lo.types().align_of(t).max(8);
                let slot = lo.new_slot(size, align);
                let d = lo.result_reg(inst);
                lo.emit(self.frame_addr(d, slot));
                for (k, v) in saved.into_iter().enumerate() {
                    let dp = self.add_off(lo, d, 8 * k as u64);
                    lo.emit(MachineInst::new(
                        A64Op::Store.opcode(),
                        vec![use_v(dp), use_v(v), imm(8)],
                    ));
                }
            }
            None => {
                if inst.result().is_some() {
                    let d = lo.result_reg(inst);
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_p(ret_reg)]));
                }
            }
        }
    }

    /// Lower the entry prologue with AAPCS64 aggregate / indirect-result /
    /// stack-parameter support. A register-passed struct parameter is stored into
    /// a private home slot (so the body sees it in memory) and its vreg is that
    /// slot's address; a by-reference parameter is already a pointer; a stack-
    /// passed parameter is addressed at `[x29 + 16 + off]`; a by-reference return
    /// stashes the incoming `x8` pointer into an aux slot for the return lowering.
    fn lower_prologue_aarch64(&self, lo: &mut Lower<'_, Self>) {
        let cc = &self.rf.cc;
        let entry = lo.mf().entry().expect("a function being lowered has an entry block");
        let param_vregs: Vec<VReg> = lo.mf().block(entry).params.clone();
        let (sig_params, ret_ty) = match lo.types().get(lo.func().sig) {
            Type::Func(ft) => (ft.params.clone(), ft.ret),
            _ => (Vec::new(), lo.func().sig),
        };
        let indirect_ret = is_aggregate(lo.types(), ret_ty)
            && matches!(classify_aggregate(lo.types(), ret_ty), AbiClass::Reference);

        let mut int_i = 0usize;
        let mut fp_i = 0usize;
        if indirect_ret {
            // The indirect-result pointer arrives in x8; stash it for `ret`.
            let slot = lo.new_slot(8, 8);
            lo.set_aux_slot(slot);
            lo.emit(MachineInst::new(
                A64Op::StoreFrame.opcode(),
                vec![use_p(regs::gpr(regs::X8)), MachineOperand::Frame(slot)],
            ));
        }

        let mut stack_in = 16u64; // first incoming stack arg, above the saved fp/lr
        for (i, &pv) in param_vregs.iter().enumerate() {
            let ty = sig_params[i];
            if is_aggregate(lo.types(), ty) {
                match classify_aggregate(lo.types(), ty) {
                    AbiClass::Hfa { width, count } => {
                        if fp_i + count as usize <= cc.fp_arg_regs.len() {
                            let size = align_up_u64(lo.byte_size(ty).max(8), 8);
                            let align = lo.types().align_of(ty).max(8);
                            let home = lo.new_slot(size, align);
                            lo.emit(self.frame_addr(pv, home));
                            let bytes = u64::from(width / 8);
                            for k in 0..count as usize {
                                let areg = cc.fp_arg_regs[fp_i];
                                fp_i += 1;
                                let v = lo.fresh_vreg(RegClass::Fp);
                                lo.emit(MachineInst::new(
                                    A64Op::MovRR.opcode(),
                                    vec![def_v(v), use_p(areg)],
                                ));
                                let dp = self.add_off(lo, pv, bytes * k as u64);
                                lo.emit(MachineInst::new(
                                    A64Op::Store.opcode(),
                                    vec![use_v(dp), use_v(v), imm(bytes)],
                                ));
                            }
                            continue;
                        }
                        stack_in = self.param_from_stack(lo, pv, ty, stack_in);
                    }
                    AbiClass::Regs(n) => {
                        if int_i + n <= cc.arg_regs.len() {
                            let size = align_up_u64(lo.byte_size(ty).max(8), 8);
                            let align = lo.types().align_of(ty).max(8);
                            let home = lo.new_slot(size, align);
                            lo.emit(self.frame_addr(pv, home));
                            for k in 0..n {
                                let areg = cc.arg_regs[int_i];
                                int_i += 1;
                                let v = lo.fresh_vreg(RegClass::Gpr);
                                lo.emit(MachineInst::new(
                                    A64Op::MovRR.opcode(),
                                    vec![def_v(v), use_p(areg)],
                                ));
                                let dp = self.add_off(lo, pv, 8 * k as u64);
                                lo.emit(MachineInst::new(
                                    A64Op::Store.opcode(),
                                    vec![use_v(dp), use_v(v), imm(8)],
                                ));
                            }
                            continue;
                        }
                        stack_in = self.param_from_stack(lo, pv, ty, stack_in);
                    }
                    AbiClass::Reference => {
                        // A by-reference parameter is just an incoming pointer.
                        if int_i < cc.arg_regs.len() {
                            let areg = cc.arg_regs[int_i];
                            int_i += 1;
                            lo.emit(MachineInst::new(
                                A64Op::MovRR.opcode(),
                                vec![def_v(pv), use_p(areg)],
                            ));
                        } else {
                            let p = self.lea_fp(lo, stack_in);
                            lo.emit(MachineInst::new(
                                A64Op::Load.opcode(),
                                vec![def_v(pv), use_v(p), imm(8)],
                            ));
                            stack_in += 8;
                        }
                    }
                }
            } else {
                let is_fp = lo.mf().vreg_class(pv) == RegClass::Fp;
                let has_reg =
                    if is_fp { fp_i < cc.fp_arg_regs.len() } else { int_i < cc.arg_regs.len() };
                if has_reg {
                    let areg = if is_fp {
                        let a = cc.fp_arg_regs[fp_i];
                        fp_i += 1;
                        a
                    } else {
                        let a = cc.arg_regs[int_i];
                        int_i += 1;
                        a
                    };
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(pv), use_p(areg)]));
                } else {
                    let sz = lo.byte_size(ty);
                    let p = self.lea_fp(lo, stack_in);
                    lo.emit(MachineInst::new(
                        A64Op::Load.opcode(),
                        vec![def_v(pv), use_v(p), imm(sz)],
                    ));
                    stack_in += 8;
                }
            }
        }
    }

    /// A stack-passed aggregate parameter: address the caller-placed copy in place
    /// at `[x29 + 16 + off]` and bind the parameter vreg to that address. Returns
    /// the new running incoming-stack offset.
    fn param_from_stack(&self, lo: &mut Lower<'_, Self>, pv: VReg, ty: TypeId, stack_in: u64) -> u64 {
        let size = lo.byte_size(ty);
        let align = lo.types().align_of(ty).max(8);
        let at = align_up_u64(stack_in, align);
        let d = self.lea_fp(lo, at);
        lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(pv), use_v(d)]));
        at + align_up_u64(size, 8)
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

    fn lower_prologue(&self, lo: &mut Lower<'_, Self>) {
        self.lower_prologue_aarch64(lo);
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
            // Dynamic (runtime-sized) stack allocation is implemented and
            // execution-tested only on x86-64 so far; the aarch64 sp-adjust
            // lowering is deferred (like other target-specific gaps here).
            InstKind::DynAlloca { .. } => {
                panic!("aarch64 backend: dynamic `dyn_alloca` is not yet supported")
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
                let cc = &self.rf.cc;
                let ret_ty = match lo.types().get(lo.func().sig) {
                    Type::Func(ft) => ft.ret,
                    _ => lo.func().sig,
                };
                if is_aggregate(lo.types(), ret_ty) {
                    // The return operand is a pointer to the struct's storage.
                    let src = lo.reg(inst.operands()[0]);
                    match classify_aggregate(lo.types(), ret_ty) {
                        AbiClass::Reference => {
                            // Copy the struct through the indirect-result pointer
                            // (stashed to the aux slot by the prologue).
                            let size = lo.byte_size(ret_ty);
                            let slot = lo
                                .aux_slot()
                                .expect("indirect result pointer saved by the prologue");
                            let dst = lo.fresh_vreg(RegClass::Gpr);
                            lo.emit(MachineInst::new(
                                A64Op::LoadFrame.opcode(),
                                vec![def_v(dst), MachineOperand::Frame(slot)],
                            ));
                            self.emit_memcpy(lo, dst, src, size);
                        }
                        AbiClass::Hfa { width, count } => {
                            // Place each element in `v0..v3`. Compute all source
                            // pointers first so the loads into the return
                            // registers are consecutive.
                            let bytes = u64::from(width / 8);
                            let ptrs: Vec<VReg> = (0..count as usize)
                                .map(|k| self.add_off(lo, src, bytes * k as u64))
                                .collect();
                            for (k, &p) in ptrs.iter().enumerate() {
                                let r = regs::fp(k as u16);
                                lo.emit(MachineInst::new(
                                    A64Op::Load.opcode(),
                                    vec![def(r), use_v(p), imm(bytes)],
                                ));
                            }
                        }
                        AbiClass::Regs(n) => {
                            // Place each eightbyte in `x0`/`x1`.
                            let ptrs: Vec<VReg> = (0..n)
                                .map(|k| self.add_off(lo, src, 8 * k as u64))
                                .collect();
                            for (k, &p) in ptrs.iter().enumerate() {
                                let r = if k == 0 { cc.ret_reg } else { regs::gpr(regs::X1) };
                                lo.emit(MachineInst::new(
                                    A64Op::Load.opcode(),
                                    vec![def(r), use_v(p), imm(8)],
                                ));
                            }
                        }
                    }
                } else if let Some(&v) = inst.operands().first() {
                    let r = lo.reg(v);
                    // A float return goes in v0, an integer/pointer return in x0.
                    let ret = match lo.mf().vreg_class(r) {
                        RegClass::Fp => cc.fp_ret_reg,
                        RegClass::Gpr => cc.ret_reg,
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
