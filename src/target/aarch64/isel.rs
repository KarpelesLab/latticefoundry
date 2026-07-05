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
use crate::ir::inst::{BinOp, InstKind, IntPred};
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
        const TABLE: [A64Op; 40] = [
            MovRR, MovRI, Add, Sub, And, Or, Eor, Mul, AddI, SubI, Sdiv, Udiv, Msub, LslI, LsrI,
            AsrI, LslV, LsrV, AsrV, CmpCset, Csel, Load, Store, FrameAddr, GlobalAddr, Call, Ret, B,
            BrCond, Switch, Unreachable, StoreFrame, LoadFrame, StpFpLr, LdpFpLr, MovFpSp, SubSp,
            AddSp, SaveReg, RestoreReg,
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
            // Floating-point binops are outside the integer subset; keep the MIR
            // well-formed with a zero placeholder (never executed in tests).
            _ => lo.emit(MachineInst::new(A64Op::MovRI.opcode(), vec![def_v(d), imm(0)])),
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
        let n_arg = args.len().min(cc.arg_regs.len());
        debug_assert!(args.len() <= cc.arg_regs.len(), "stack-passed args are out of scope");

        for (areg, &arg) in cc.arg_regs.iter().zip(args) {
            let r = lo.reg(arg);
            lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def(*areg), use_v(r)]));
        }

        let mut operands = Vec::new();
        match lo.callee_func(callee) {
            Some(fidx) => operands.push(MachineOperand::Func(fidx)),
            None => {
                let cr = lo.reg(callee);
                operands.push(use_v(cr));
            }
        }
        operands.push(def(cc.ret_reg));
        for &cs in &self.rf.caller_saved {
            if cs != cc.ret_reg {
                operands.push(def(cs));
            }
        }
        for &areg in &cc.arg_regs[..n_arg] {
            operands.push(use_p(areg));
        }
        lo.emit(MachineInst::new(A64Op::Call.opcode(), operands));

        if inst.result().is_some() {
            let d = lo.result_reg(inst);
            lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_p(cc.ret_reg)]));
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
            RegClass::Fp => &self.rf.empty,
        }
    }

    fn scratch(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Gpr => &self.rf.scratch,
            RegClass::Fp => &self.rf.empty,
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
            InstKind::Cast(_) => {
                // Integer casts in the tested subset are width changes computed in
                // registers; a plain copy preserves the low bits (zero/sign
                // extension of narrow types is out of the executed subset).
                let d = lo.result_reg(inst);
                let s = lo.reg(inst.operands()[0]);
                lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
            }
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
            InstKind::Unary(_) | InstKind::FCmp(_) => {
                let d = lo.result_reg(inst);
                lo.emit(MachineInst::new(A64Op::MovRI.opcode(), vec![def_v(d), imm(0)]));
            }
            _ => unreachable!("terminator reached lower_inst: {:?}", inst.kind),
        }
    }

    fn lower_term(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Ret => {
                if let Some(&v) = inst.operands().first() {
                    let r = lo.reg(v);
                    let x0 = self.rf.cc.ret_reg;
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def(x0), use_v(r)]));
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
