//! An abstract, RISC-like **virtual target** that implements the target
//! interface end to end, so the framework (isel + regalloc) can be exercised
//! and tested without committing to a real ISA (that is Phase 7).
//!
//! The machine is a simple load/store register machine with a 16-register
//! general-purpose file:
//!
//! - `r0..=r12` are allocatable;
//! - `r13..=r15` are reserved as scratch for spill/reload code;
//! - `r0..=r7` are caller-saved (a call may clobber them), `r8..=r12`
//!   callee-saved;
//! - the calling convention passes the leading integer/pointer arguments in
//!   `r0..=r3` and returns a value in `r0`; the stack grows down.
//!
//! Instructions are three-address and encoding-free — their operands are the
//! MIR [`MachineOperand`]s directly. Opcode operand layouts are documented on
//! [`VOp`]; the [`crate::codegen::interp`] interpreter is the executable
//! semantics they are validated against.

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::mir::{
    MBlockId, MachineInst, MachineOperand, Opcode, PReg, Reg, RegClass, StackSlot, VReg,
};
use crate::codegen::target::{CallConv, MachineTarget};
use crate::ir::inst::{BinOp, CastOp, InstKind, IntPred};
use crate::ir::{InstData, Module};

use puremp::Int;

/// The virtual machine's opcodes. Each variant documents its fixed operand
/// layout; `Def`/`Use` are register operands, `Imm`/`Frame`/`Label`/`Func`/
/// `Global` are the non-register operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum VOp {
    /// `[Def d, Imm v]` — `d = v`.
    Li = 0,
    /// `[Def d, Use s]` — `d = s`.
    Move = 1,
    /// `[Def d, Use a, Use b, Imm width]` — width-masked two's-complement add.
    Add = 2,
    /// `[Def d, Use a, Use b, Imm width]` — subtract.
    Sub = 3,
    /// `[Def d, Use a, Use b, Imm width]` — multiply.
    Mul = 4,
    /// `[Def d, Use a, Use b, Imm width]` — unsigned divide.
    UDiv = 5,
    /// `[Def d, Use a, Use b, Imm width]` — signed divide.
    SDiv = 6,
    /// `[Def d, Use a, Use b, Imm width]` — unsigned remainder.
    URem = 7,
    /// `[Def d, Use a, Use b, Imm width]` — signed remainder.
    SRem = 8,
    /// `[Def d, Use a, Use b, Imm width]` — bitwise and.
    And = 9,
    /// `[Def d, Use a, Use b, Imm width]` — bitwise or.
    Or = 10,
    /// `[Def d, Use a, Use b, Imm width]` — bitwise xor.
    Xor = 11,
    /// `[Def d, Use a, Use b, Imm width]` — left shift.
    Shl = 12,
    /// `[Def d, Use a, Use b, Imm width]` — logical right shift.
    LShr = 13,
    /// `[Def d, Use a, Use b, Imm width]` — arithmetic right shift.
    AShr = 14,
    /// `[Def d, Use a, Use b, Imm pred, Imm width]` — integer compare to `0`/`1`.
    ICmp = 15,
    /// `[Def d, Use c, Use t, Use f]` — `d = c != 0 ? t : f`.
    Select = 16,
    /// `[Def d, Use s, Imm castcode, Imm srcw, Imm dstw]` — integer conversion.
    Cast = 17,
    /// `[Def d, Use ptr, Imm size]` — load `size` bytes.
    Load = 18,
    /// `[Use ptr, Use val, Imm size]` — store `size` bytes.
    Store = 19,
    /// `[Def d, Frame slot]` — `d = address of slot`.
    FrameAddr = 20,
    /// `[Def d, Global g]` — `d = address of global g`.
    GlobalAddr = 21,
    /// `[Func callee | Use calleeReg, Def ret, Def clobbers..., Use args...]`.
    Call = 22,
    /// `[]` or `[Use ret]` — return.
    Ret = 23,
    /// `[Label t]` — unconditional jump.
    Jmp = 24,
    /// `[Use cond, Label t, Label f]` — branch to `t` if `cond != 0`, else `f`.
    BrCond = 25,
    /// `[Use cond, Label default, (Imm val, Label edge)...]` — multi-way branch.
    Switch = 26,
    /// `[]` — an unreachable point (executing it traps).
    Unreachable = 27,
    /// `[Use src, Frame slot]` — spill a register to a stack slot.
    StackStore = 28,
    /// `[Def dst, Frame slot]` — reload a register from a stack slot.
    StackLoad = 29,
    /// An operation outside the integer subset this target lowers (float ops):
    /// kept structurally well-formed but not executable.
    Unsupported = 30,
    /// `[Def dst, Use n, Imm align]` — dynamic (runtime-sized) stack allocation
    /// (`dyn_alloca`): bump-allocate `n` bytes of `align`-aligned frame memory and
    /// put the base address in `dst`. Modeled by the interpreter's flat memory.
    DynAlloca = 31,
}

impl VOp {
    /// The target [`Opcode`] id for this opcode.
    #[inline]
    pub fn opcode(self) -> Opcode {
        Opcode(self as u32)
    }

    /// Decode a target [`Opcode`] back to a [`VOp`].
    pub fn decode(op: Opcode) -> VOp {
        use VOp::*;
        const TABLE: [VOp; 32] = [
            Li, Move, Add, Sub, Mul, UDiv, SDiv, URem, SRem, And, Or, Xor, Shl, LShr, AShr, ICmp,
            Select, Cast, Load, Store, FrameAddr, GlobalAddr, Call, Ret, Jmp, BrCond, Switch,
            Unreachable, StackStore, StackLoad, Unsupported, DynAlloca,
        ];
        TABLE[op.0 as usize]
    }
}

/// Encode an [`IntPred`] as an immediate for [`VOp::ICmp`].
pub fn pred_code(p: IntPred) -> i64 {
    match p {
        IntPred::Eq => 0,
        IntPred::Ne => 1,
        IntPred::Ugt => 2,
        IntPred::Uge => 3,
        IntPred::Ult => 4,
        IntPred::Ule => 5,
        IntPred::Sgt => 6,
        IntPred::Sge => 7,
        IntPred::Slt => 8,
        IntPred::Sle => 9,
    }
}

/// Encode an integer [`CastOp`] as an immediate for [`VOp::Cast`].
pub fn cast_code(c: CastOp) -> i64 {
    match c {
        CastOp::Trunc => 0,
        CastOp::ZExt => 1,
        CastOp::SExt => 2,
        CastOp::PtrToInt => 3,
        CastOp::IntToPtr => 4,
        CastOp::Bitcast => 5,
        // Float casts are outside the integer subset.
        _ => -1,
    }
}

/// The abstract virtual target.
#[derive(Debug)]
pub struct VirtualTarget {
    classes: Vec<RegClass>,
    allocatable_gpr: Vec<PReg>,
    scratch_gpr: Vec<PReg>,
    caller_saved: Vec<PReg>,
    callee_saved: Vec<PReg>,
    empty: Vec<PReg>,
    cc: CallConv,
}

impl Default for VirtualTarget {
    fn default() -> Self {
        Self::new()
    }
}

fn gpr(n: u16) -> PReg {
    PReg::new(RegClass::Gpr, n)
}

impl VirtualTarget {
    /// Construct the virtual target with its fixed register file and ABI.
    pub fn new() -> VirtualTarget {
        let allocatable_gpr: Vec<PReg> = (0..=12).map(gpr).collect();
        let scratch_gpr: Vec<PReg> = (13..=15).map(gpr).collect();
        let caller_saved: Vec<PReg> = (0..=7).map(gpr).collect();
        let callee_saved: Vec<PReg> = (8..=12).map(gpr).collect();
        let cc = CallConv {
            arg_regs: (0..=3).map(gpr).collect(),
            fp_arg_regs: Vec::new(),
            ret_reg: gpr(0),
            fp_ret_reg: PReg::new(RegClass::Fp, 0),
            stack_grows_down: true,
        };
        VirtualTarget {
            classes: vec![RegClass::Gpr],
            allocatable_gpr,
            scratch_gpr,
            caller_saved,
            callee_saved,
            empty: Vec::new(),
            cc,
        }
    }

    /// Lower function `func` of `module` to MIR over this target.
    pub fn select(&self, module: &Module, func: crate::ir::FuncId) -> crate::codegen::mir::MachineFunction {
        crate::codegen::isel::select(self, module, func)
    }

    /// The `VOp` for an integer [`BinOp`], or `None` if it is a float op.
    fn bin_op(op: BinOp) -> Option<VOp> {
        Some(match op {
            BinOp::Add => VOp::Add,
            BinOp::Sub => VOp::Sub,
            BinOp::Mul => VOp::Mul,
            BinOp::UDiv => VOp::UDiv,
            BinOp::SDiv => VOp::SDiv,
            BinOp::URem => VOp::URem,
            BinOp::SRem => VOp::SRem,
            BinOp::And => VOp::And,
            BinOp::Or => VOp::Or,
            BinOp::Xor => VOp::Xor,
            BinOp::Shl => VOp::Shl,
            BinOp::LShr => VOp::LShr,
            BinOp::AShr => VOp::AShr,
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => return None,
        })
    }

    fn lower_call(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];
        let n_arg = args.len().min(self.cc.arg_regs.len());
        debug_assert!(args.len() <= self.cc.arg_regs.len(), "more call args than arg registers");

        // Move arguments into the physical argument registers.
        for (areg, &arg) in self.cc.arg_regs.iter().zip(args) {
            let r = lo.reg(arg);
            let mv = self.emit_move(Reg::Physical(*areg), Reg::Virtual(r));
            lo.emit(mv);
        }

        // Build the call: callee, defs (return + caller-saved clobbers), then
        // the physical argument uses.
        let mut operands = Vec::new();
        match lo.callee_func(callee) {
            Some(fidx) => operands.push(MachineOperand::Func(fidx)),
            None => {
                let cr = lo.reg(callee);
                operands.push(MachineOperand::Use(Reg::Virtual(cr)));
            }
        }
        operands.push(MachineOperand::Def(Reg::Physical(self.cc.ret_reg)));
        for &cs in &self.caller_saved {
            if cs != self.cc.ret_reg {
                operands.push(MachineOperand::Def(Reg::Physical(cs)));
            }
        }
        for &areg in &self.cc.arg_regs[..n_arg] {
            operands.push(MachineOperand::Use(Reg::Physical(areg)));
        }
        lo.emit(MachineInst::new(VOp::Call.opcode(), operands));

        // Move the return value into the result vreg.
        if inst.result().is_some() {
            let d = lo.result_reg(inst);
            let mv = self.emit_move(Reg::Virtual(d), Reg::Physical(self.cc.ret_reg));
            lo.emit(mv);
        }
    }
}

/// A three-register width-masked arithmetic instruction.
fn arith(op: VOp, d: VReg, a: VReg, b: VReg, width: u32) -> MachineInst {
    MachineInst::new(
        op.opcode(),
        vec![
            MachineOperand::Def(Reg::Virtual(d)),
            MachineOperand::Use(Reg::Virtual(a)),
            MachineOperand::Use(Reg::Virtual(b)),
            MachineOperand::Imm(Int::from_u64(u64::from(width))),
        ],
    )
}

impl MachineTarget for VirtualTarget {
    fn name(&self) -> &str {
        "vabstract"
    }

    fn reg_classes(&self) -> &[RegClass] {
        &self.classes
    }

    fn allocatable(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Gpr => &self.allocatable_gpr,
            RegClass::Fp => &self.empty,
        }
    }

    fn scratch(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Gpr => &self.scratch_gpr,
            RegClass::Fp => &self.empty,
        }
    }

    fn caller_saved(&self) -> &[PReg] {
        &self.caller_saved
    }

    fn callee_saved(&self) -> &[PReg] {
        &self.callee_saved
    }

    fn call_conv(&self) -> &CallConv {
        &self.cc
    }

    fn is_terminator(&self, op: Opcode) -> bool {
        matches!(
            VOp::decode(op),
            VOp::Jmp | VOp::BrCond | VOp::Switch | VOp::Ret | VOp::Unreachable
        )
    }

    fn is_move(&self, op: Opcode) -> bool {
        VOp::decode(op) == VOp::Move
    }

    fn emit_move(&self, dst: Reg, src: Reg) -> MachineInst {
        MachineInst::new(VOp::Move.opcode(), vec![MachineOperand::Def(dst), MachineOperand::Use(src)])
    }

    fn emit_spill(&self, slot: StackSlot, src: PReg) -> MachineInst {
        MachineInst::new(
            VOp::StackStore.opcode(),
            vec![MachineOperand::Use(Reg::Physical(src)), MachineOperand::Frame(slot)],
        )
    }

    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(
            VOp::StackLoad.opcode(),
            vec![MachineOperand::Def(Reg::Physical(dst)), MachineOperand::Frame(slot)],
        )
    }
}

impl TargetIsel for VirtualTarget {
    fn li(&self, dst: VReg, value: Int) -> MachineInst {
        MachineInst::new(
            VOp::Li.opcode(),
            vec![MachineOperand::Def(Reg::Virtual(dst)), MachineOperand::Imm(value)],
        )
    }

    fn jump(&self, dst: MBlockId) -> MachineInst {
        MachineInst::new(VOp::Jmp.opcode(), vec![MachineOperand::Label(dst)])
    }

    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(
            VOp::FrameAddr.opcode(),
            vec![MachineOperand::Def(Reg::Virtual(dst)), MachineOperand::Frame(slot)],
        )
    }

    fn global_addr(&self, dst: VReg, g: u32) -> MachineInst {
        MachineInst::new(
            VOp::GlobalAddr.opcode(),
            vec![MachineOperand::Def(Reg::Virtual(dst)), MachineOperand::Global(g)],
        )
    }

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Bin(op) => {
                let d = lo.result_reg(inst);
                let a = lo.reg(inst.operands()[0]);
                let b = lo.reg(inst.operands()[1]);
                let width = lo.int_width(inst.operands()[0]);
                match VirtualTarget::bin_op(*op) {
                    Some(vop) => lo.emit(arith(vop, d, a, b, width)),
                    None => lo.emit(unsupported(d, &[a, b])),
                }
            }
            InstKind::ICmp(pred) => {
                let d = lo.result_reg(inst);
                let a = lo.reg(inst.operands()[0]);
                let b = lo.reg(inst.operands()[1]);
                let width = lo.int_width(inst.operands()[0]);
                lo.emit(MachineInst::new(
                    VOp::ICmp.opcode(),
                    vec![
                        MachineOperand::Def(Reg::Virtual(d)),
                        MachineOperand::Use(Reg::Virtual(a)),
                        MachineOperand::Use(Reg::Virtual(b)),
                        MachineOperand::Imm(Int::from_i64(pred_code(*pred))),
                        MachineOperand::Imm(Int::from_u64(u64::from(width))),
                    ],
                ));
            }
            InstKind::Cast(op) if cast_code(*op) >= 0 => {
                let d = lo.result_reg(inst);
                let s = lo.reg(inst.operands()[0]);
                let srcw = lo.int_width(inst.operands()[0]);
                let dstw = lo.types().bit_width(inst.ty).unwrap_or(64);
                lo.emit(MachineInst::new(
                    VOp::Cast.opcode(),
                    vec![
                        MachineOperand::Def(Reg::Virtual(d)),
                        MachineOperand::Use(Reg::Virtual(s)),
                        MachineOperand::Imm(Int::from_i64(cast_code(*op))),
                        MachineOperand::Imm(Int::from_u64(u64::from(srcw))),
                        MachineOperand::Imm(Int::from_u64(u64::from(dstw))),
                    ],
                ));
            }
            InstKind::Alloca { elem_ty } => {
                let d = lo.result_reg(inst);
                let size = lo.byte_size(*elem_ty);
                let align = lo.types().align_of(*elem_ty);
                let slot = lo.new_slot(size, align);
                lo.emit(self.frame_addr(d, slot));
            }
            InstKind::DynAlloca { align } => {
                let d = lo.result_reg(inst);
                let n = lo.reg(inst.operands()[0]);
                lo.emit(MachineInst::new(
                    VOp::DynAlloca.opcode(),
                    vec![
                        MachineOperand::Def(Reg::Virtual(d)),
                        MachineOperand::Use(Reg::Virtual(n)),
                        MachineOperand::Imm(Int::from_u64(u64::from(*align))),
                    ],
                ));
            }
            InstKind::Load { ty, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(inst.operands()[0]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    VOp::Load.opcode(),
                    vec![
                        MachineOperand::Def(Reg::Virtual(d)),
                        MachineOperand::Use(Reg::Virtual(ptr)),
                        MachineOperand::Imm(Int::from_u64(size)),
                    ],
                ));
            }
            InstKind::Store { ty, .. } => {
                let ptr = lo.reg(inst.operands()[0]);
                let val = lo.reg(inst.operands()[1]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    VOp::Store.opcode(),
                    vec![
                        MachineOperand::Use(Reg::Virtual(ptr)),
                        MachineOperand::Use(Reg::Virtual(val)),
                        MachineOperand::Imm(Int::from_u64(size)),
                    ],
                ));
            }
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(inst);
                let base = lo.reg(inst.operands()[0]);
                let off = lo.reg(inst.operands()[1]);
                lo.emit(arith(VOp::Add, d, base, off, 64));
            }
            InstKind::Select => {
                let d = lo.result_reg(inst);
                let c = lo.reg(inst.operands()[0]);
                let t = lo.reg(inst.operands()[1]);
                let f = lo.reg(inst.operands()[2]);
                lo.emit(MachineInst::new(
                    VOp::Select.opcode(),
                    vec![
                        MachineOperand::Def(Reg::Virtual(d)),
                        MachineOperand::Use(Reg::Virtual(c)),
                        MachineOperand::Use(Reg::Virtual(t)),
                        MachineOperand::Use(Reg::Virtual(f)),
                    ],
                ));
            }
            InstKind::Freeze => {
                let d = lo.result_reg(inst);
                let s = lo.reg(inst.operands()[0]);
                lo.emit(self.emit_move(Reg::Virtual(d), Reg::Virtual(s)));
            }
            InstKind::Call => self.lower_call(lo, inst),
            // Float / unmodeled value ops: keep structurally well-formed.
            InstKind::Unary(_) | InstKind::FCmp(_) | InstKind::Cast(_) => {
                let d = lo.result_reg(inst);
                let uses: Vec<VReg> = inst.operands().iter().map(|&o| lo.reg(o)).collect();
                lo.emit(unsupported(d, &uses));
            }
            _ => unreachable!("terminator reached lower_inst: {:?}", inst.kind),
        }
    }

    fn lower_term(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Ret => {
                if let Some(&v) = inst.operands().first() {
                    let r = lo.reg(v);
                    let mv = self.emit_move(Reg::Physical(self.cc.ret_reg), Reg::Virtual(r));
                    lo.emit(mv);
                    lo.emit(MachineInst::new(
                        VOp::Ret.opcode(),
                        vec![MachineOperand::Use(Reg::Physical(self.cc.ret_reg))],
                    ));
                } else {
                    lo.emit(MachineInst::new(VOp::Ret.opcode(), Vec::new()));
                }
            }
            InstKind::Br(target) => {
                let args: Vec<_> = inst.operands().to_vec();
                let e = lo.edge_to(*target, &args);
                lo.emit(self.jump(e));
            }
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let cond = lo.reg(inst.operands()[0]);
                let ops = inst.operands();
                let ta = 1;
                let tb = 1 + *true_args as usize;
                let fb = tb + *false_args as usize;
                let true_vals: Vec<_> = ops[ta..tb].to_vec();
                let false_vals: Vec<_> = ops[tb..fb].to_vec();
                let te = lo.edge_to(*if_true, &true_vals);
                let fe = lo.edge_to(*if_false, &false_vals);
                lo.emit(MachineInst::new(
                    VOp::BrCond.opcode(),
                    vec![
                        MachineOperand::Use(Reg::Virtual(cond)),
                        MachineOperand::Label(te),
                        MachineOperand::Label(fe),
                    ],
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
                let mut operands = vec![
                    MachineOperand::Use(Reg::Virtual(cond)),
                    MachineOperand::Label(de),
                ];
                let cases = data.cases.clone();
                for case in &cases {
                    let n = case.args as usize;
                    let cvals: Vec<_> = ops[idx..idx + n].to_vec();
                    idx += n;
                    let ce = lo.edge_to(case.target, &cvals);
                    operands.push(MachineOperand::Imm(case.value.clone()));
                    operands.push(MachineOperand::Label(ce));
                }
                lo.emit(MachineInst::new(VOp::Switch.opcode(), operands));
            }
            InstKind::Unreachable => {
                lo.emit(MachineInst::new(VOp::Unreachable.opcode(), Vec::new()));
            }
            _ => unreachable!("non-terminator reached lower_term: {:?}", inst.kind),
        }
    }
}

/// An `Unsupported` placeholder defining `d` and reading `uses` (keeps the MIR
/// well-formed for register allocation; the interpreter refuses to run it).
fn unsupported(d: VReg, uses: &[VReg]) -> MachineInst {
    let mut operands = vec![MachineOperand::Def(Reg::Virtual(d))];
    operands.extend(uses.iter().map(|&u| MachineOperand::Use(Reg::Virtual(u))));
    MachineInst::new(VOp::Unsupported.opcode(), operands)
}
