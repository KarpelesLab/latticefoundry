//! The x86-64 machine opcode set and instruction-selection rules (ROADMAP
//! Phase 7).
//!
//! [`X86Op`] is this target's [`Opcode`] vocabulary. It is a *post-isel,
//! pre-encoding* MIR: operands are still MIR [`MachineOperand`]s (registers,
//! immediates, frame slots, labels, symbol references), and one MIR op may expand
//! to several machine instructions at encode time (e.g. an [`X86Op::Add`] becomes
//! `mov dst, a; add dst, b`). Keeping the two-address fixup and the
//! flags/setcc/movzx idioms as single MIR ops is what lets the register allocator
//! see clean three-address def/use information while the encoder still emits legal
//! two-address x86.
//!
//! ## Two-address handling
//!
//! x86 arithmetic is destructive (`add dst, src` computes `dst += src`). We model
//! the IR's three-address `d = a op b` as a single MIR op with operands
//! `[Def d, Use a, Use b]` and let the encoder materialize the copy:
//! since `d`, `a`, and `b` all interfere at the op (d is defined there, a and b
//! are read there), the allocator always gives them distinct physical registers,
//! so the encoder can emit `mov d, a; op d, b` unconditionally (with a `neg`
//! fixup for the one non-commutative case, `sub`, should `d` ever coincide with
//! `b`). Spilled operands are reloaded into scratch by the allocator *before* the
//! op, so the expansion still sees final physical registers.
//!
//! ## Block arguments, calls, returns
//!
//! Block arguments are realized by the framework's edge-move mechanism
//! ([`Lower::edge_to`]). `call` moves arguments into the SysV argument registers,
//! records the return register and the caller-saved clobbers as fixed defs, and
//! moves the result out of `rax`; `ret` moves its value into `rax`. The prologue
//! moves incoming parameters out of the argument registers (framework prologue).

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::mir::{
    MBlockId, MachineInst, MachineOperand, Opcode, PReg, Reg, RegClass, StackSlot, VReg,
};
use crate::codegen::target::{CallConv, MachineTarget};
use crate::ir::inst::{BinOp, CastOp, FloatPred, InstKind, IntPred, UnaryOp};
use crate::ir::value::{Const, ValueDef};
use crate::ir::{InstData, Module, ValueId};

use puremp::Int;

use super::regs::{self, RegFile};

/// The x86-64 MIR opcode vocabulary. Operand layouts are documented per variant;
/// `Def`/`Use` are register operands, the rest are immediates, frame slots,
/// branch labels, or symbol references.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum X86Op {
    /// `[Def d, Use s]` — `mov d, s` (64-bit copy).
    MovRR = 0,
    /// `[Def d, Imm v]` — load immediate `d = v`.
    MovRI = 1,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a + b`.
    Add = 2,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a - b`.
    Sub = 3,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a & b`.
    And = 4,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a | b`.
    Or = 5,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a ^ b`.
    Xor = 6,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a * b` (imul).
    Imul = 7,
    /// `[Def d, Use a, Imm count, Imm width]` — `d = a << count`.
    ShlI = 8,
    /// `[Def d, Use a, Imm count, Imm width]` — `d = a >>u count`.
    ShrI = 9,
    /// `[Def d, Use a, Imm count, Imm width]` — `d = a >>s count`.
    SarI = 10,
    /// `[Def d, Use a, Use rcx, Imm width]` — `d = a << (cl)`.
    ShlCl = 11,
    /// `[Def d, Use a, Use rcx, Imm width]` — `d = a >>u (cl)`.
    ShrCl = 12,
    /// `[Def d, Use a, Use rcx, Imm width]` — `d = a >>s (cl)`.
    SarCl = 13,
    /// `[Def rdx, Use rax, Imm width]` — sign-extend rax into rdx (cqo/cdq).
    Cqo = 14,
    /// `[Def rdx]` — zero rdx (`xor edx, edx`).
    ZeroRdx = 15,
    /// `[Def rax, Def rdx, Use rax, Use rdx, Use b, Imm width]` — signed divide.
    Idiv = 16,
    /// `[Def rax, Def rdx, Use rax, Use rdx, Use b, Imm width]` — unsigned divide.
    Div = 17,
    /// `[Def d, Use a, Use b, Imm cc, Imm width]` — `cmp a,b; setcc d; movzx d`.
    SetccCmp = 18,
    /// `[Use r]` — `test r, r` (sets flags for a following cmov).
    Test = 19,
    /// `[Def d, Use d, Use t]` — `cmovne d, t` (move if ZF=0).
    Cmovne = 20,
    /// `[Def d, Use ptr, Imm size]` — load `size` bytes from `[ptr]`.
    Load = 21,
    /// `[Use ptr, Use val, Imm size]` — store `size` bytes to `[ptr]`.
    Store = 22,
    /// `[Def d, Frame slot]` — `lea d, [rbp + slot]`.
    LeaFrame = 23,
    /// `[Def d, Global g]` — `lea d, [rip + global]` (RIP-relative).
    GlobalAddr = 24,
    /// `[Func f | Use callee, Def rax, Def clobbers.., Use args..]` — call.
    Call = 25,
    /// `[]` — return (value already in rax).
    Ret = 26,
    /// `[Label t]` — unconditional jump.
    Jmp = 27,
    /// `[Use cond, Label t, Label f]` — `test cond,cond; jne t; jmp f`.
    BrCond = 28,
    /// `[Use cond, Label default, (Imm val, Label case)...]` — multi-way branch.
    Switch = 29,
    /// `[]` — an unreachable trap (`ud2`).
    Unreachable = 30,
    /// `[Use r]` — `push r`.
    Push = 31,
    /// `[Def r]` — `pop r`.
    Pop = 32,
    /// `[]` — `mov rbp, rsp`.
    MovRbpRsp = 33,
    /// `[Imm k]` — `sub rsp, k`.
    SubRsp = 34,
    /// `[Imm k]` — `lea rsp, [rbp - k]`.
    LeaRspRbp = 35,
    /// `[Use src, Frame slot]` — spill: `mov`/`movsd` `[rbp+slot], src` (the
    /// mnemonic follows `src`'s register class).
    StoreFrame = 36,
    /// `[Def dst, Frame slot]` — reload: `mov`/`movsd` `dst, [rbp+slot]`.
    LoadFrame = 37,

    // --- SSE scalar floating-point ----------------------------------------
    /// `[Def d, Use a, Use b, Imm width]` — `d = a + b` (`addsd`/`addss`).
    FAdd = 38,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a - b` (`subsd`/`subss`).
    FSub = 39,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a * b` (`mulsd`/`mulss`).
    FMul = 40,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a / b` (`divsd`/`divss`).
    FDiv = 41,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a ^ b` (`xorpd`/`xorps`); used
    /// with a sign-bit mask to implement `fneg`.
    FXor = 42,
    /// `[Def d, Imm bits, Imm width]` — materialize a float constant: load the
    /// exact bit pattern via a scratch gpr (`mov r11, bits; movq/movd d, r11`).
    LoadFConst = 43,
    /// `[Def d, Use a, Use b, Imm packed, Imm width]` — `ucomis` + `setcc`
    /// (+ parity fixup) computing the `i1` result of an `fcmp` into gpr `d`.
    FCmpSet = 44,
    /// `[Def d, Use s]` — `cvtsd2ss d, s` (F64→F32, `fptrunc`).
    Cvtsd2ss = 45,
    /// `[Def d, Use s]` — `cvtss2sd d, s` (F32→F64, `fpext`).
    Cvtss2sd = 46,
    /// `[Def d, Use s, Imm srcfloatwidth, Imm flags]` — `cvttsd2si`/`cvttss2si`
    /// (float→int, truncating). `flags` bit0 = 64-bit gpr destination.
    CvtF2si = 47,
    /// `[Def d, Use s, Imm dstfloatwidth, Imm flags]` — `cvtsi2sd`/`cvtsi2ss`
    /// (int→float). `flags` bit0 = 64-bit gpr source, bit1 = zero-extend a
    /// 32-bit source first (unsigned ≤32), bit2 = full unsigned-64 fix-up (the
    /// `shr`/`and`/`or` halve-and-round sequence plus a doubling `addsd`).
    CvtSi2f = 48,
    /// `[Def d, Func f]` — `lea d, [rip + func]` (RIP-relative): materialize a
    /// function's runtime address into a GPR for use as a function pointer, with
    /// a `Pc32` relocation to the function symbol. A *direct* call still lowers
    /// through [`X86Op::Call`] with a `Func` operand and a `Plt32` relocation;
    /// only a function used as a plain *value* reaches here.
    FuncAddr = 49,
    /// `[Def d, Use s, Imm src_w, Imm dst_w]` — sign-extend `s` (`src_w` bits)
    /// into `d` (`movsx`/`movsxd`). Implements the IR `sext`.
    Movsx = 50,
    /// `[Def d, Use s, Imm src_w, Imm dst_w]` — zero-extend `s` (`src_w` bits)
    /// into `d` (`movzx`, or a 32-bit `mov`). Implements the IR `zext`.
    Movzx = 51,
}

impl X86Op {
    /// The MIR [`Opcode`] id for this opcode.
    #[inline]
    pub fn opcode(self) -> Opcode {
        Opcode(self as u32)
    }

    /// Decode a MIR [`Opcode`] back to an [`X86Op`].
    pub fn decode(op: Opcode) -> X86Op {
        use X86Op::*;
        const TABLE: [X86Op; 52] = [
            MovRR, MovRI, Add, Sub, And, Or, Xor, Imul, ShlI, ShrI, SarI, ShlCl, ShrCl, SarCl, Cqo,
            ZeroRdx, Idiv, Div, SetccCmp, Test, Cmovne, Load, Store, LeaFrame, GlobalAddr, Call,
            Ret, Jmp, BrCond, Switch, Unreachable, Push, Pop, MovRbpRsp, SubRsp, LeaRspRbp,
            StoreFrame, LoadFrame, FAdd, FSub, FMul, FDiv, FXor, LoadFConst, FCmpSet, Cvtsd2ss,
            Cvtss2sd, CvtF2si, CvtSi2f, FuncAddr, Movsx, Movzx,
        ];
        TABLE[op.0 as usize]
    }
}

/// Encode an [`IntPred`] as the x86 condition-code nibble used by `setcc`/`jcc`.
pub(crate) fn cc_code(p: IntPred) -> u8 {
    match p {
        IntPred::Eq => 0x4,  // E
        IntPred::Ne => 0x5,  // NE
        IntPred::Ugt => 0x7, // A  (above)
        IntPred::Uge => 0x3, // AE (not below)
        IntPred::Ult => 0x2, // B  (below)
        IntPred::Ule => 0x6, // BE
        IntPred::Sgt => 0xF, // G
        IntPred::Sge => 0xD, // GE
        IntPred::Slt => 0xC, // L
        IntPred::Sle => 0xE, // LE
    }
}

/// The `ucomis`+`setcc` plan for an `fcmp` predicate, packed into one immediate
/// for [`X86Op::FCmpSet`]. Returns `None` for the constant predicates
/// `False`/`True`, which the caller materializes directly.
///
/// After `ucomisd a, b` the flags are: `ZF=PF=CF=1` when unordered (a NaN
/// operand), else `CF` = "below" (a<b), `ZF` = "equal", `PF` = 0. Packing:
/// bits 0..8 = the primary `setcc` code; bit 8 = swap operands (`ucomis b, a`,
/// realizing the `<`/`<=` orderings from `>`/`>=`); bits 9..11 = the combine
/// step (`0` none, `1` AND `setnp`, `2` OR `setp`) that separates the ordered
/// and unordered readings of equality.
pub(crate) fn fcmp_pack(pred: FloatPred) -> Option<u64> {
    // (primary cc, swap, combine): combine 0 = none, 1 = AND setnp, 2 = OR setp.
    let (cc, swap, combine): (u8, bool, u8) = match pred {
        FloatPred::False | FloatPred::True => return None,
        FloatPred::Oeq => (0x4, false, 1), // sete AND setnp
        FloatPred::One => (0x5, false, 0), // setne
        FloatPred::Ogt => (0x7, false, 0), // seta
        FloatPred::Oge => (0x3, false, 0), // setae
        FloatPred::Olt => (0x7, true, 0),  // ucomis b,a; seta
        FloatPred::Ole => (0x3, true, 0),  // ucomis b,a; setae
        FloatPred::Ueq => (0x4, false, 0), // sete
        FloatPred::Une => (0x5, false, 2), // setne OR setp
        FloatPred::Ugt => (0x2, true, 0),  // ucomis b,a; setb
        FloatPred::Uge => (0x6, true, 0),  // ucomis b,a; setbe
        FloatPred::Ult => (0x2, false, 0), // setb
        FloatPred::Ule => (0x6, false, 0), // setbe
        FloatPred::Ord => (0xB, false, 0), // setnp
        FloatPred::Uno => (0xA, false, 0), // setp
    };
    Some(u64::from(cc) | (u64::from(swap) << 8) | (u64::from(combine) << 9))
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

/// The x86-64 target: its register file/ABI plus the isel + encoding rules.
#[derive(Debug)]
pub struct X86_64Target {
    rf: RegFile,
}

impl Default for X86_64Target {
    fn default() -> Self {
        Self::new()
    }
}

impl X86_64Target {
    /// Construct the x86-64 target with its fixed register file and SysV ABI.
    pub fn new() -> X86_64Target {
        X86_64Target { rf: RegFile::new() }
    }

    /// Lower function `func` of `module` to MIR over this target.
    pub fn select(&self, module: &Module, func: crate::ir::FuncId) -> crate::codegen::mir::MachineFunction {
        crate::codegen::isel::select(self, module, func)
    }

    /// Resolve a value operand to a register, but materialize a **function
    /// reference used as a value** (its address taken / stored / passed) into a
    /// GPR via a RIP-relative [`X86Op::FuncAddr`] `lea`, rather than the
    /// framework default (a zero placeholder). Every other value defers to
    /// [`Lower::reg`]. A direct call is unaffected: its callee is recognized by
    /// [`Lower::callee_func`] and never routed through here.
    fn oper(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        if let ValueDef::Func(f) = lo.func().value(v).def {
            let fidx = f.index() as u32;
            let d = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(MachineInst::new(
                X86Op::FuncAddr.opcode(),
                vec![def_v(d), MachineOperand::Func(fidx)],
            ));
            return d;
        }
        lo.reg(v)
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
        let simple = match op {
            BinOp::Add => Some(X86Op::Add),
            BinOp::Sub => Some(X86Op::Sub),
            BinOp::And => Some(X86Op::And),
            BinOp::Or => Some(X86Op::Or),
            BinOp::Xor => Some(X86Op::Xor),
            BinOp::Mul => Some(X86Op::Imul),
            _ => None,
        };
        if let Some(x) = simple {
            let a = self.oper(lo, inst.operands()[0]);
            let b = self.oper(lo, inst.operands()[1]);
            lo.emit(MachineInst::new(
                x.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        // Scalar SSE floating-point arithmetic (F32/F64). The operand width comes
        // from the float type (32 or 64); the encoder picks the ss/sd form.
        let fop = match op {
            BinOp::FAdd => Some(X86Op::FAdd),
            BinOp::FSub => Some(X86Op::FSub),
            BinOp::FMul => Some(X86Op::FMul),
            BinOp::FDiv => Some(X86Op::FDiv),
            _ => None,
        };
        if let Some(x) = fop {
            let a = self.oper(lo, inst.operands()[0]);
            let b = self.oper(lo, inst.operands()[1]);
            lo.emit(MachineInst::new(
                x.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        match op {
            BinOp::Shl => self.lower_shift(lo, X86Op::ShlI, X86Op::ShlCl, d, inst, width),
            BinOp::LShr => self.lower_shift(lo, X86Op::ShrI, X86Op::ShrCl, d, inst, width),
            BinOp::AShr => self.lower_shift(lo, X86Op::SarI, X86Op::SarCl, d, inst, width),
            BinOp::UDiv => self.lower_div(lo, X86Op::Div, false, false, d, inst, width),
            BinOp::URem => self.lower_div(lo, X86Op::Div, false, true, d, inst, width),
            BinOp::SDiv => self.lower_div(lo, X86Op::Idiv, true, false, d, inst, width),
            BinOp::SRem => self.lower_div(lo, X86Op::Idiv, true, true, d, inst, width),
            // `frem` has no scalar SSE form (it is the `fmod` libcall). Rather
            // than silently emit a wrong result, fail loudly: the frontend must
            // lower `frem` to a call, or this backend must grow the libcall. All
            // other binops are handled above.
            BinOp::FRem => {
                panic!("x86-64 backend: `frem` is unsupported (needs an fmod libcall)")
            }
            _ => unreachable!("binop already handled: {op:?}"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_shift(
        &self,
        lo: &mut Lower<'_, Self>,
        imm_op: X86Op,
        cl_op: X86Op,
        d: VReg,
        inst: &InstData,
        width: u32,
    ) {
        let a = self.oper(lo, inst.operands()[0]);
        if let Some(c) = Self::const_of(lo, inst.operands()[1]) {
            let count = c.to_i64().unwrap_or(0) as u64;
            lo.emit(MachineInst::new(
                imm_op.opcode(),
                vec![def_v(d), use_v(a), imm(count), imm(u64::from(width))],
            ));
        } else {
            let b = self.oper(lo, inst.operands()[1]);
            let rcx = regs::gpr(regs::RCX);
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(rcx), use_v(b)]));
            lo.emit(MachineInst::new(
                cl_op.opcode(),
                vec![def_v(d), use_v(a), use_p(rcx), imm(u64::from(width))],
            ));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_div(
        &self,
        lo: &mut Lower<'_, Self>,
        div_op: X86Op,
        signed: bool,
        want_rem: bool,
        d: VReg,
        inst: &InstData,
        width: u32,
    ) {
        let a = self.oper(lo, inst.operands()[0]);
        let b = self.oper(lo, inst.operands()[1]);
        let rax = regs::gpr(regs::RAX);
        let rdx = regs::gpr(regs::RDX);
        // dividend low half -> rax
        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(rax), use_v(a)]));
        // extend into rdx
        if signed {
            lo.emit(MachineInst::new(
                X86Op::Cqo.opcode(),
                vec![def(rdx), use_p(rax), imm(u64::from(width))],
            ));
        } else {
            lo.emit(MachineInst::new(X86Op::ZeroRdx.opcode(), vec![def(rdx)]));
        }
        lo.emit(MachineInst::new(
            div_op.opcode(),
            vec![def(rax), def(rdx), use_p(rax), use_p(rdx), use_v(b), imm(u64::from(width))],
        ));
        let src = if want_rem { rdx } else { rax };
        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_p(src)]));
    }

    /// `fneg`: flip the IEEE sign bit (matching the reference semantics, which is
    /// a sign flip, not `0 - x`). Materialize the sign mask
    /// (`0x8000_0000_0000_0000` / `0x8000_0000`) in an xmm and `xorpd`/`xorps`.
    fn lower_fneg(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let d = lo.result_reg(inst);
        let s = self.oper(lo, inst.operands()[0]);
        let width = lo.int_width(inst.operands()[0]);
        let mask_bits: u64 =
            if width == 64 { 0x8000_0000_0000_0000 } else { 0x8000_0000 };
        let mask = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(
            X86Op::LoadFConst.opcode(),
            vec![def_v(mask), imm(mask_bits), imm(u64::from(width))],
        ));
        lo.emit(MachineInst::new(
            X86Op::FXor.opcode(),
            vec![def_v(d), use_v(s), use_v(mask), imm(u64::from(width))],
        ));
    }

    /// `fcmp`: `ucomis` then `setcc` with the ordered/unordered parity fixup
    /// packed by [`fcmp_pack`]. The result is an `i1` in a gpr.
    fn lower_fcmp(&self, lo: &mut Lower<'_, Self>, pred: FloatPred, inst: &InstData) {
        let d = lo.result_reg(inst);
        match fcmp_pack(pred) {
            None => {
                // `False`/`True` are constants.
                let v = u64::from(pred == FloatPred::True);
                lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(d), imm(v)]));
            }
            Some(packed) => {
                let a = self.oper(lo, inst.operands()[0]);
                let b = self.oper(lo, inst.operands()[1]);
                let width = lo.int_width(inst.operands()[0]);
                lo.emit(MachineInst::new(
                    X86Op::FCmpSet.opcode(),
                    vec![def_v(d), use_v(a), use_v(b), imm(packed), imm(u64::from(width))],
                ));
            }
        }
    }

    /// Conversions. Float↔float and int↔float go through the SSE `cvt*` forms;
    /// every other cast (integer width change, ptr↔int, bitcast within a class)
    /// is a low-bits-preserving copy, matching the existing integer behavior.
    fn lower_cast(&self, lo: &mut Lower<'_, Self>, op: CastOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let s = self.oper(lo, inst.operands()[0]);
        let src_w = lo.int_width(inst.operands()[0]);
        let dst_w = lo.types().bit_width(inst.ty).unwrap_or(64);
        match op {
            CastOp::FpTrunc => lo.emit(MachineInst::new(
                X86Op::Cvtsd2ss.opcode(),
                vec![def_v(d), use_v(s)],
            )),
            CastOp::FpExt => lo.emit(MachineInst::new(
                X86Op::Cvtss2sd.opcode(),
                vec![def_v(d), use_v(s)],
            )),
            CastOp::FpToSi => {
                // bit0 of flags = 64-bit gpr destination.
                let flags = u64::from(dst_w > 32);
                lo.emit(MachineInst::new(
                    X86Op::CvtF2si.opcode(),
                    vec![def_v(d), use_v(s), imm(u64::from(src_w)), imm(flags)],
                ));
            }
            CastOp::FpToUi => {
                // A ≤32-bit unsigned result is exact through a 64-bit signed
                // `cvttsd2si` (it lands in `[0, 2^63)`). A full unsigned-64 result
                // needs the 2^63 fix-up (flags bit1): values ≥ 2^63 are converted
                // as `x - 2^63` and biased back. Truncation is toward zero.
                let flags: u64 = if dst_w > 32 { 0b11 } else { 1 };
                lo.emit(MachineInst::new(
                    X86Op::CvtF2si.opcode(),
                    vec![def_v(d), use_v(s), imm(u64::from(src_w)), imm(flags)],
                ));
            }
            CastOp::SiToFp => {
                // bit0 = 64-bit gpr source.
                let flags = u64::from(src_w > 32);
                lo.emit(MachineInst::new(
                    X86Op::CvtSi2f.opcode(),
                    vec![def_v(d), use_v(s), imm(u64::from(dst_w)), imm(flags)],
                ));
            }
            CastOp::UiToFp => {
                // Unsigned int→float: a ≤32-bit source zero-extends to 64 bits
                // (flags bit1) then a 64-bit signed conversion; a 64-bit source
                // uses the full unsigned-64 fix-up (flags bit2) — direct
                // `cvtsi2sd` when the sign bit is clear, else the halve-and-round
                // `(x>>1)|(x&1)` sequence followed by a doubling `addsd`, which
                // reproduces round-to-nearest for values ≥ 2^63.
                let flags: u64 = if src_w > 32 { 0b100 } else { 0b10 };
                lo.emit(MachineInst::new(
                    X86Op::CvtSi2f.opcode(),
                    vec![def_v(d), use_v(s), imm(u64::from(dst_w)), imm(flags)],
                ));
            }
            CastOp::SExt => lo.emit(MachineInst::new(
                X86Op::Movsx.opcode(),
                vec![def_v(d), use_v(s), imm(u64::from(src_w)), imm(u64::from(dst_w))],
            )),
            CastOp::ZExt => lo.emit(MachineInst::new(
                X86Op::Movzx.opcode(),
                vec![def_v(d), use_v(s), imm(u64::from(src_w)), imm(u64::from(dst_w))],
            )),
            // Truncation drops high bits, and ptr↔int / same-class bitcast preserve
            // the bit pattern: a plain register copy is correct (consumers operate
            // at the result's width).
            _ => lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(s)])),
        }
    }

    fn lower_call(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let cc = &self.rf.cc;
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];

        // Route each argument by class: integers into rdi.. (a separate counter
        // from) floats into xmm0.. — the SysV rule for mixed int/float calls.
        //
        // Materialize *every* argument value into a vreg FIRST, then move them all
        // into the physical argument registers immediately before the call. If we
        // instead materialized-and-moved one argument at a time, a later argument's
        // materialization (e.g. loading a float constant, which lands in a fresh
        // vreg) could be colored into an argument register already holding an
        // earlier argument: the register allocator's fixed-register model is
        // point-based, so it would not see that the argument register is live
        // across the gap between its arg-move and the call. Emitting all the moves
        // consecutively right before the call keeps each argument register's
        // occupied range free of any competing vreg definition.
        let mut int_i = 0usize;
        let mut fp_i = 0usize;
        let mut moves: Vec<(PReg, VReg)> = Vec::with_capacity(args.len());
        for &arg in args {
            let r = self.oper(lo, arg);
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
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(areg), use_v(r)]));
        }

        // The return register follows the result type's class (xmm0 for a float
        // return, rax otherwise).
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
        lo.emit(MachineInst::new(X86Op::Call.opcode(), operands));

        if inst.result().is_some() {
            let d = lo.result_reg(inst);
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_p(ret_reg)]));
        }
    }
}

impl MachineTarget for X86_64Target {
    fn name(&self) -> &str {
        "x86_64"
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
            X86Op::decode(op),
            X86Op::Jmp | X86Op::BrCond | X86Op::Switch | X86Op::Ret | X86Op::Unreachable
        )
    }

    fn is_move(&self, op: Opcode) -> bool {
        X86Op::decode(op) == X86Op::MovRR
    }

    fn emit_move(&self, dst: Reg, src: Reg) -> MachineInst {
        MachineInst::new(X86Op::MovRR.opcode(), vec![MachineOperand::Def(dst), MachineOperand::Use(src)])
    }

    fn emit_spill(&self, slot: StackSlot, src: PReg) -> MachineInst {
        MachineInst::new(X86Op::StoreFrame.opcode(), vec![use_p(src), MachineOperand::Frame(slot)])
    }

    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(X86Op::LoadFrame.opcode(), vec![def(dst), MachineOperand::Frame(slot)])
    }
}

impl TargetIsel for X86_64Target {
    fn li(&self, dst: VReg, value: Int) -> MachineInst {
        MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(dst), MachineOperand::Imm(value)])
    }

    fn jump(&self, dst: MBlockId) -> MachineInst {
        MachineInst::new(X86Op::Jmp.opcode(), vec![MachineOperand::Label(dst)])
    }

    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(X86Op::LeaFrame.opcode(), vec![def_v(dst), MachineOperand::Frame(slot)])
    }

    fn global_addr(&self, dst: VReg, g: u32) -> MachineInst {
        MachineInst::new(X86Op::GlobalAddr.opcode(), vec![def_v(dst), MachineOperand::Global(g)])
    }

    fn float_const(&self, dst: VReg, bits: u64, width: u32) -> MachineInst {
        MachineInst::new(
            X86Op::LoadFConst.opcode(),
            vec![def_v(dst), imm(bits), imm(u64::from(width))],
        )
    }

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Bin(op) => self.lower_bin(lo, *op, inst),
            InstKind::ICmp(pred) => {
                let d = lo.result_reg(inst);
                let a = self.oper(lo, inst.operands()[0]);
                let b = self.oper(lo, inst.operands()[1]);
                let width = lo.int_width(inst.operands()[0]);
                lo.emit(MachineInst::new(
                    X86Op::SetccCmp.opcode(),
                    vec![
                        def_v(d),
                        use_v(a),
                        use_v(b),
                        imm(u64::from(cc_code(*pred))),
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
                let ptr = self.oper(lo, inst.operands()[0]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    X86Op::Load.opcode(),
                    vec![def_v(d), use_v(ptr), imm(size)],
                ));
            }
            InstKind::Store { ty, .. } => {
                let ptr = self.oper(lo, inst.operands()[0]);
                let val = self.oper(lo, inst.operands()[1]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    X86Op::Store.opcode(),
                    vec![use_v(ptr), use_v(val), imm(size)],
                ));
            }
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(inst);
                let base = self.oper(lo, inst.operands()[0]);
                let off = self.oper(lo, inst.operands()[1]);
                lo.emit(MachineInst::new(
                    X86Op::Add.opcode(),
                    vec![def_v(d), use_v(base), use_v(off), imm(64)],
                ));
            }
            InstKind::Select => {
                let d = lo.result_reg(inst);
                let c = self.oper(lo, inst.operands()[0]);
                let t = self.oper(lo, inst.operands()[1]);
                let f = self.oper(lo, inst.operands()[2]);
                // d = f; test c,c; cmovne d, t   (cond != 0 -> t)
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(f)]));
                lo.emit(MachineInst::new(X86Op::Test.opcode(), vec![use_v(c)]));
                lo.emit(MachineInst::new(
                    X86Op::Cmovne.opcode(),
                    vec![def_v(d), use_v(d), use_v(t)],
                ));
            }
            InstKind::Freeze => {
                let d = lo.result_reg(inst);
                let s = self.oper(lo, inst.operands()[0]);
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
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
                    let r = self.oper(lo, v);
                    // A float return goes in xmm0, an integer/pointer return in rax.
                    let ret = match lo.mf().vreg_class(r) {
                        RegClass::Fp => self.rf.cc.fp_ret_reg,
                        RegClass::Gpr => self.rf.cc.ret_reg,
                    };
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(ret), use_v(r)]));
                }
                lo.emit(MachineInst::new(X86Op::Ret.opcode(), Vec::new()));
            }
            InstKind::Br(target) => {
                let args: Vec<_> = inst.operands().to_vec();
                let e = lo.edge_to(*target, &args);
                lo.emit(self.jump(e));
            }
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let cond = self.oper(lo, inst.operands()[0]);
                let ops = inst.operands();
                let tb = 1 + *true_args as usize;
                let fb = tb + *false_args as usize;
                let true_vals: Vec<_> = ops[1..tb].to_vec();
                let false_vals: Vec<_> = ops[tb..fb].to_vec();
                let te = lo.edge_to(*if_true, &true_vals);
                let fe = lo.edge_to(*if_false, &false_vals);
                lo.emit(MachineInst::new(
                    X86Op::BrCond.opcode(),
                    vec![use_v(cond), MachineOperand::Label(te), MachineOperand::Label(fe)],
                ));
            }
            InstKind::Switch(data) => {
                let cond = self.oper(lo, inst.operands()[0]);
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
                lo.emit(MachineInst::new(X86Op::Switch.opcode(), operands));
            }
            InstKind::Unreachable => {
                lo.emit(MachineInst::new(X86Op::Unreachable.opcode(), Vec::new()));
            }
            _ => unreachable!("non-terminator reached lower_term: {:?}", inst.kind),
        }
    }
}
