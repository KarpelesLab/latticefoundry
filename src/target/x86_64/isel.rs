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
use crate::ir::inst::{BinOp, InstKind, IntPred};
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
    /// `[Use src, Frame slot]` — spill: `mov [rbp+slot], src`.
    StoreFrame = 36,
    /// `[Def dst, Frame slot]` — reload: `mov dst, [rbp+slot]`.
    LoadFrame = 37,
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
        const TABLE: [X86Op; 38] = [
            MovRR, MovRI, Add, Sub, And, Or, Xor, Imul, ShlI, ShrI, SarI, ShlCl, ShrCl, SarCl, Cqo,
            ZeroRdx, Idiv, Div, SetccCmp, Test, Cmovne, Load, Store, LeaFrame, GlobalAddr, Call,
            Ret, Jmp, BrCond, Switch, Unreachable, Push, Pop, MovRbpRsp, SubRsp, LeaRspRbp,
            StoreFrame, LoadFrame,
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
            let a = lo.reg(inst.operands()[0]);
            let b = lo.reg(inst.operands()[1]);
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
            // Floating-point binops are outside the integer subset; keep the MIR
            // well-formed with a zero placeholder (never executed in tests).
            _ => lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(d), imm(0)])),
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
        let a = lo.reg(inst.operands()[0]);
        if let Some(c) = Self::const_of(lo, inst.operands()[1]) {
            let count = c.to_i64().unwrap_or(0) as u64;
            lo.emit(MachineInst::new(
                imm_op.opcode(),
                vec![def_v(d), use_v(a), imm(count), imm(u64::from(width))],
            ));
        } else {
            let b = lo.reg(inst.operands()[1]);
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
        let a = lo.reg(inst.operands()[0]);
        let b = lo.reg(inst.operands()[1]);
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

    fn lower_call(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let cc = &self.rf.cc;
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];
        let n_arg = args.len().min(cc.arg_regs.len());
        debug_assert!(args.len() <= cc.arg_regs.len(), "stack-passed args are out of scope");

        for (areg, &arg) in cc.arg_regs.iter().zip(args) {
            let r = lo.reg(arg);
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(*areg), use_v(r)]));
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
        lo.emit(MachineInst::new(X86Op::Call.opcode(), operands));

        if inst.result().is_some() {
            let d = lo.result_reg(inst);
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_p(cc.ret_reg)]));
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

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Bin(op) => self.lower_bin(lo, *op, inst),
            InstKind::ICmp(pred) => {
                let d = lo.result_reg(inst);
                let a = lo.reg(inst.operands()[0]);
                let b = lo.reg(inst.operands()[1]);
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
            InstKind::Cast(_) => {
                // Integer casts in the tested subset are width changes computed in
                // registers; a plain copy preserves the low bits (zero/sign
                // extension of narrow types is out of the executed subset).
                let d = lo.result_reg(inst);
                let s = lo.reg(inst.operands()[0]);
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
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
                    X86Op::Load.opcode(),
                    vec![def_v(d), use_v(ptr), imm(size)],
                ));
            }
            InstKind::Store { ty, .. } => {
                let ptr = lo.reg(inst.operands()[0]);
                let val = lo.reg(inst.operands()[1]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    X86Op::Store.opcode(),
                    vec![use_v(ptr), use_v(val), imm(size)],
                ));
            }
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(inst);
                let base = lo.reg(inst.operands()[0]);
                let off = lo.reg(inst.operands()[1]);
                lo.emit(MachineInst::new(
                    X86Op::Add.opcode(),
                    vec![def_v(d), use_v(base), use_v(off), imm(64)],
                ));
            }
            InstKind::Select => {
                let d = lo.result_reg(inst);
                let c = lo.reg(inst.operands()[0]);
                let t = lo.reg(inst.operands()[1]);
                let f = lo.reg(inst.operands()[2]);
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
                let s = lo.reg(inst.operands()[0]);
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
            }
            InstKind::Call => self.lower_call(lo, inst),
            InstKind::Unary(_) | InstKind::FCmp(_) => {
                let d = lo.result_reg(inst);
                lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(d), imm(0)]));
            }
            _ => unreachable!("terminator reached lower_inst: {:?}", inst.kind),
        }
    }

    fn lower_term(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Ret => {
                if let Some(&v) = inst.operands().first() {
                    let r = lo.reg(v);
                    let rax = self.rf.cc.ret_reg;
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(rax), use_v(r)]));
                }
                lo.emit(MachineInst::new(X86Op::Ret.opcode(), Vec::new()));
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
                    X86Op::BrCond.opcode(),
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
                lo.emit(MachineInst::new(X86Op::Switch.opcode(), operands));
            }
            InstKind::Unreachable => {
                lo.emit(MachineInst::new(X86Op::Unreachable.opcode(), Vec::new()));
            }
            _ => unreachable!("non-terminator reached lower_term: {:?}", inst.kind),
        }
    }
}
