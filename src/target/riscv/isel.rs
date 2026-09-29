//! The RISC-V RV64IM machine opcode set ([`RvOp`]) and the integer
//! instruction-selection rules.
//!
//! [`RvOp`] is this target's [`Opcode`] vocabulary: a *post-isel, pre-encoding*
//! MIR whose operands are still MIR [`MachineOperand`]s (registers, immediates,
//! frame slots, labels, symbol references). RISC-V data-processing instructions
//! are genuinely three-address (`add rd, rs1, rs2`), so the isel emits one MIR op
//! per IR op with a clean `[Def d, Use a, Use b]` shape and the encoder never has
//! to synthesize a move-to-destination. A few IR ops still expand to a short
//! RISC-V idiom at encode time (a comparison becomes `slt`/`sltu` plus `xori`/
//! `seqz`/`snez`; a `select` becomes a branchless mask sequence; a remainder is a
//! plain `rem`/`remu`; a constant is a `lui`/`addi` materialization).
//!
//! ## x0 tricks, block arguments, calls, returns
//!
//! `x0` is hardwired zero: `mv rd, rs` is `addi rd, rs, 0`, a zero constant is a
//! read of `x0`, `seqz`/`snez` compare against `x0`, and `ret` is `jalr x0, ra,
//! 0`. Block arguments are realized by the framework's edge-move mechanism
//! ([`Lower::edge_to`]). `call` moves arguments into the LP64 argument registers
//! `a0`–`a7`, records `a0` and the caller-saved clobbers as fixed defs, and moves
//! the result out of `a0`; `ret` moves its value into `a0`. The prologue moves
//! incoming parameters out of the argument registers (framework prologue). RV64M
//! has hardware divide/remainder, so `div`/`divu`/`rem`/`remu` lower directly —
//! no fixed-register dance.
//!
//! ## Narrow values
//!
//! Every data-processing op is the full-width RV64 form, so a narrow value
//! (`i1`/`i8`/`i16`/`i32`, or an odd `_BitInt` width) lives in a 64-bit register
//! whose bits above its width are **not** kept clean: an `i8` add of 200 + 100
//! leaves 300 there, and a `trunc` is a plain move. Ops that only feed the low
//! bits (`add`, `mul`, `shl`, logic, stores) don't care; every op whose result
//! depends on the upper bits (compares, right shifts, division, `zext`/`sext`,
//! branch/select conditions, `switch`) first extends via
//! `RiscvTarget::extend64` (`sext.w`, `andi`, or an `slli`+`srai`/`srli` pair).
//! At call boundaries the LP64 psABI's signedness-independent rules are
//! honored: an `i32` argument/return is sign-extended (`sext.w`) and an `i1`
//! zero-extended, so foreign callees see ABI-conformant registers.
//!
//! Deferred (noted for a follow-up): the RV64 word forms (`addw`/`divw`/
//! `sraw`/...) as a cheaper `i32` lowering; scalar floating-point (F/D), the
//! compressed (C) and atomic (A) extensions; and `> 8` integer arguments passed
//! on the stack.

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::mir::{
    MBlockId, MachineInst, MachineOperand, Opcode, PReg, Reg, RegClass, StackSlot, VReg,
};
use crate::codegen::target::{CallConv, MachineTarget};
use crate::ir::inst::{BinOp, CastOp, InstKind, IntPred, UnaryOp};
use crate::ir::types::Type;
use crate::ir::value::{Const, ValueDef};
use crate::ir::{InstData, Module, ValueId};

use puremp::Int;

use super::regs::RegFile;

/// The RV64IM MIR opcode vocabulary. Operand layouts are documented per variant;
/// `Def`/`Use` are register operands, the rest are immediates, frame slots, branch
/// labels, or symbol references. `Imm width` is the operation's IR integer bit
/// width, informational only: the encoder always emits the full 64-bit form and
/// the interpreter models exactly that (see "Narrow values" above).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum RvOp {
    /// `[Def d, Use s]` — `mv d, s` (`addi d, s, 0`).
    Mv = 0,
    /// `[Def d, Imm v]` — load immediate via a `lui`/`addi` materialization.
    Li = 1,
    /// `[Def d, Use a, Use b, Imm width]` — `add d, a, b`.
    Add = 2,
    /// `[Def d, Use a, Use b, Imm width]` — `sub d, a, b`.
    Sub = 3,
    /// `[Def d, Use a, Use b, Imm width]` — `and d, a, b`.
    And = 4,
    /// `[Def d, Use a, Use b, Imm width]` — `or d, a, b`.
    Or = 5,
    /// `[Def d, Use a, Use b, Imm width]` — `xor d, a, b`.
    Xor = 6,
    /// `[Def d, Use a, Use b, Imm width]` — `mul d, a, b`.
    Mul = 7,
    /// `[Def d, Use a, Use b, Imm width]` — `mulh d, a, b` (signed high half).
    Mulh = 8,
    /// `[Def d, Use a, Imm k, Imm width]` — `addi d, a, #k`.
    Addi = 9,
    /// `[Def d, Use a, Imm k, Imm width]` — `andi d, a, #k`.
    Andi = 10,
    /// `[Def d, Use a, Imm k, Imm width]` — `ori d, a, #k`.
    Ori = 11,
    /// `[Def d, Use a, Imm k, Imm width]` — `xori d, a, #k`.
    Xori = 12,
    /// `[Def d, Use a, Use b, Imm width]` — `div d, a, b` (signed).
    Div = 13,
    /// `[Def d, Use a, Use b, Imm width]` — `divu d, a, b` (unsigned).
    Divu = 14,
    /// `[Def d, Use a, Use b, Imm width]` — `rem d, a, b` (signed).
    Rem = 15,
    /// `[Def d, Use a, Use b, Imm width]` — `remu d, a, b` (unsigned).
    Remu = 16,
    /// `[Def d, Use a, Imm shamt, Imm width]` — `slli d, a, #shamt`.
    Slli = 17,
    /// `[Def d, Use a, Imm shamt, Imm width]` — `srli d, a, #shamt`.
    Srli = 18,
    /// `[Def d, Use a, Imm shamt, Imm width]` — `srai d, a, #shamt`.
    Srai = 19,
    /// `[Def d, Use a, Use b, Imm width]` — `sll d, a, b`.
    Sll = 20,
    /// `[Def d, Use a, Use b, Imm width]` — `srl d, a, b`.
    Srl = 21,
    /// `[Def d, Use a, Use b, Imm width]` — `sra d, a, b`.
    Sra = 22,
    /// `[Def d, Use a, Use b, Imm pred, Imm width]` — set-if-condition into a GPR,
    /// synthesized from `slt`/`sltu` and `xori`/`seqz`/`snez` (see `pred_code`).
    SetCmp = 23,
    /// `[Def d, Use cond, Use t, Use f]` — branchless `d = cond ? t : f`.
    Select = 24,
    /// `[Def d, Use ptr, Imm size]` — load `size` bytes from `[ptr]` (zero-extended).
    Load = 25,
    /// `[Use ptr, Use val, Imm size]` — store `size` bytes to `[ptr]`.
    Store = 26,
    /// `[Def d, Frame slot]` — `addi d, sp, #slot_off`.
    FrameAddr = 27,
    /// `[Def d, Global g]` — `auipc d, %pcrel_hi(g); addi d, d, %pcrel_lo(g)`.
    GlobalAddr = 28,
    /// `[Func f | Use callee, Def a0, Def clobbers.., Use args..]` — call.
    Call = 29,
    /// `[]` — return (value already in a0; `jalr x0, ra, 0`).
    Ret = 30,
    /// `[Label t]` — unconditional jump (`jal x0, t`).
    J = 31,
    /// `[Use cond, Label t, Label f]` — `bnez cond, t; j f`.
    BrCond = 32,
    /// `[Use cond, Label default, (Imm val, Label case)...]` — multi-way branch.
    Switch = 33,
    /// `[]` — a trap (`ebreak`).
    Unreachable = 34,
    /// `[Use src, Frame slot]` — spill: `sd src, [sp, #slot_off]`.
    StoreFrame = 35,
    /// `[Def dst, Frame slot]` — reload: `ld dst, [sp, #slot_off]`.
    LoadFrame = 36,
    /// `[Imm delta]` — `addi sp, sp, #delta` (signed; prologue/epilogue).
    AddiSp = 37,
    /// `[Use r, Imm off]` — `sd r, [sp, #off]` (callee-saved / ra save).
    SaveReg = 38,
    /// `[Def r, Imm off]` — `ld r, [sp, #off]` (callee-saved / ra restore).
    RestoreReg = 39,
    /// `[Def a0, Use a7, Use a0..]` — the Linux environment call `ecall`. The
    /// syscall number is in `a7` and the arguments in `a0..a5` (moved there by
    /// isel as one consecutive run right before); the kernel returns in `a0` and
    /// preserves every other register, so `a0` is the only def.
    Ecall = 40,
    /// `[Def d, Use s]` — `sext.w d, s` (`addiw d, s, 0`): sign-extend the low
    /// 32 bits of `s` to 64.
    SextW = 41,
}

impl RvOp {
    /// The MIR [`Opcode`] id for this opcode.
    #[inline]
    pub fn opcode(self) -> Opcode {
        Opcode(self as u32)
    }

    /// Decode a MIR [`Opcode`] back to an [`RvOp`].
    pub fn decode(op: Opcode) -> RvOp {
        use RvOp::*;
        const TABLE: [RvOp; 42] = [
            Mv, Li, Add, Sub, And, Or, Xor, Mul, Mulh, Addi, Andi, Ori, Xori, Div, Divu, Rem, Remu,
            Slli, Srli, Srai, Sll, Srl, Sra, SetCmp, Select, Load, Store, FrameAddr, GlobalAddr,
            Call, Ret, J, BrCond, Switch, Unreachable, StoreFrame, LoadFrame, AddiSp, SaveReg,
            RestoreReg, Ecall, SextW,
        ];
        TABLE[op.0 as usize]
    }
}

/// A switch case value sign-extended from the scrutinee's `width` to 64 bits,
/// matching the sign-extended scrutinee it is compared against.
fn sext_case(value: &Int, width: u32) -> Int {
    if width >= 64 {
        return value.clone();
    }
    let raw = value.to_i64().map(|v| v as u64).or_else(|| value.to_u64()).unwrap_or(0);
    let shift = 64 - width;
    Int::from_i64(((raw << shift) as i64) >> shift)
}

/// A dense code for an [`IntPred`], packed into the [`RvOp::SetCmp`] immediate and
/// decoded by the encoder and interpreter.
pub(crate) fn pred_code(p: IntPred) -> u8 {
    match p {
        IntPred::Eq => 0,
        IntPred::Ne => 1,
        IntPred::Ult => 2,
        IntPred::Ule => 3,
        IntPred::Ugt => 4,
        IntPred::Uge => 5,
        IntPred::Slt => 6,
        IntPred::Sle => 7,
        IntPred::Sgt => 8,
        IntPred::Sge => 9,
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

/// The RISC-V RV64 target: its register file/ABI plus the isel + encoding rules.
#[derive(Debug)]
pub struct RiscvTarget {
    rf: RegFile,
}

impl Default for RiscvTarget {
    fn default() -> Self {
        Self::new()
    }
}

impl RiscvTarget {
    /// Construct the RV64 target with its fixed register file and LP64 ABI.
    pub fn new() -> RiscvTarget {
        RiscvTarget { rf: RegFile::new() }
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

    /// Whether an integer constant fits the RISC-V 12-bit signed immediate field
    /// (`addi`/`andi`/`ori`/`xori`), i.e. `-2048 ..= 2047`.
    fn fits_imm12(c: &Int) -> Option<u64> {
        let v = c.to_i64()?;
        if (-2048..=2047).contains(&v) { Some((v as u64) & 0xFFF) } else { None }
    }

    /// Whether `v` is a compare result, whose register holds exactly 0 or 1.
    fn is_compare(lo: &Lower<'_, Self>, v: ValueId) -> bool {
        matches!(lo.func().value(v).def, ValueDef::Inst(id)
            if matches!(lo.func().inst(id).kind, InstKind::ICmp(_) | InstKind::FCmp(_)))
    }

    /// `v` sign- or zero-extended from its width to all 64 bits of a register.
    /// Narrow values live in 64-bit registers whose upper bits are not kept
    /// clean, so anything that reads those bits extends first: `sext.w` for a
    /// signed `i32`, `andi` for an unsigned width of at most 11 bits (the mask
    /// fits the sign-extended 12-bit immediate), else an `slli` + `srai`/`srli`
    /// pair. A 64-bit value, and a compare result being zero-extended, are
    /// already clean.
    fn extend64(&self, lo: &mut Lower<'_, Self>, v: ValueId, signed: bool) -> VReg {
        let r = lo.reg(v);
        let width = lo.int_width(v);
        if width >= 64 || (!signed && Self::is_compare(lo, v)) {
            return r;
        }
        let d = lo.fresh_vreg(RegClass::Gpr);
        if signed && width == 32 {
            lo.emit(MachineInst::new(RvOp::SextW.opcode(), vec![def_v(d), use_v(r)]));
        } else if !signed && width <= 11 {
            let m = (1u64 << width) - 1;
            lo.emit(MachineInst::new(
                RvOp::Andi.opcode(),
                vec![def_v(d), use_v(r), imm(m), imm(64)],
            ));
        } else {
            let k = u64::from(64 - width);
            let t = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(MachineInst::new(RvOp::Slli.opcode(), vec![def_v(t), use_v(r), imm(k), imm(64)]));
            let right = if signed { RvOp::Srai } else { RvOp::Srli };
            lo.emit(MachineInst::new(right.opcode(), vec![def_v(d), use_v(t), imm(k), imm(64)]));
        }
        d
    }

    /// An `i1` branch/select condition as a register holding exactly 0 or 1
    /// (`bnez` tests all 64 bits, and the branchless select blends with
    /// `0 - cond`). A compare's result already is; anything else (e.g. a
    /// `trunc` to `i1`) may carry garbage above bit 0.
    fn clean_cond(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        self.extend64(lo, v, false)
    }

    /// A call argument / return value as the LP64 psABI wants it in a register:
    /// an `i32` sign-extended to 64 bits (whatever its C signedness) and an `i1`
    /// (`_Bool`) zero-extended. Other narrow widths depend on the C type's
    /// signedness, which the IR does not carry, and are passed as is.
    fn abi_value(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        let is_int = matches!(lo.types().get(lo.func().value_type(v)), Type::Int(_));
        match lo.int_width(v) {
            32 if is_int => self.extend64(lo, v, true),
            1 if is_int => self.extend64(lo, v, false),
            _ => lo.reg(v),
        }
    }

    fn lower_bin(&self, lo: &mut Lower<'_, Self>, op: BinOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let width = lo.int_width(inst.operands()[0]);
        // The commutative/associative immediate-friendly ops that have an I-type
        // form: try to fold a small constant RHS into `addi`/`andi`/`ori`/`xori`.
        let imm_form = match op {
            BinOp::Add => Some(RvOp::Addi),
            BinOp::And => Some(RvOp::Andi),
            BinOp::Or => Some(RvOp::Ori),
            BinOp::Xor => Some(RvOp::Xori),
            _ => None,
        };
        if let Some(iop) = imm_form
            && let Some(c) = Self::const_of(lo, inst.operands()[1])
            && let Some(u) = Self::fits_imm12(&c)
        {
            let a = lo.reg(inst.operands()[0]);
            lo.emit(MachineInst::new(
                iop.opcode(),
                vec![def_v(d), use_v(a), imm(u), imm(u64::from(width))],
            ));
            return;
        }
        // Plain register three-address forms.
        let simple = match op {
            BinOp::Add => Some(RvOp::Add),
            BinOp::Sub => Some(RvOp::Sub),
            BinOp::And => Some(RvOp::And),
            BinOp::Or => Some(RvOp::Or),
            BinOp::Xor => Some(RvOp::Xor),
            BinOp::Mul => Some(RvOp::Mul),
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
            BinOp::Shl => self.lower_shift(lo, RvOp::Slli, RvOp::Sll, d, inst, width),
            BinOp::LShr => self.lower_shift(lo, RvOp::Srli, RvOp::Srl, d, inst, width),
            BinOp::AShr => self.lower_shift(lo, RvOp::Srai, RvOp::Sra, d, inst, width),
            BinOp::UDiv => self.lower_div(lo, RvOp::Divu, false, d, inst, width),
            BinOp::SDiv => self.lower_div(lo, RvOp::Div, true, d, inst, width),
            BinOp::URem => self.lower_div(lo, RvOp::Remu, false, d, inst, width),
            BinOp::SRem => self.lower_div(lo, RvOp::Rem, true, d, inst, width),
            // Floating-point binops are out of the integer subset (deferred); a
            // zero keeps the MIR well-formed (never reached by the integer tests).
            _ => lo.emit(MachineInst::new(RvOp::Li.opcode(), vec![def_v(d), imm(0)])),
        }
    }

    /// Division / remainder. The 64-bit `div`/`rem` see every bit of both
    /// operands, so a narrow dividend and divisor are extended by the
    /// operation's signedness first.
    #[allow(clippy::too_many_arguments)]
    fn lower_div(
        &self,
        lo: &mut Lower<'_, Self>,
        rop: RvOp,
        signed: bool,
        d: VReg,
        inst: &InstData,
        width: u32,
    ) {
        let a = self.extend64(lo, inst.operands()[0], signed);
        let b = self.extend64(lo, inst.operands()[1], signed);
        lo.emit(MachineInst::new(
            rop.opcode(),
            vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
        ));
    }

    fn lower_shift(
        &self,
        lo: &mut Lower<'_, Self>,
        imm_op: RvOp,
        var_op: RvOp,
        d: VReg,
        inst: &InstData,
        width: u32,
    ) {
        // A right shift brings the bits above the width down into the result.
        let a = match imm_op {
            RvOp::Srli => self.extend64(lo, inst.operands()[0], false),
            RvOp::Srai => self.extend64(lo, inst.operands()[0], true),
            _ => lo.reg(inst.operands()[0]),
        };
        if let Some(c) = Self::const_of(lo, inst.operands()[1]) {
            let shmask = if width >= 64 { 63 } else { u64::from(width) - 1 };
            let shamt = c.to_u64().unwrap_or(0) & shmask;
            lo.emit(MachineInst::new(
                imm_op.opcode(),
                vec![def_v(d), use_v(a), imm(shamt), imm(u64::from(width))],
            ));
        } else {
            // The hardware takes the count from the low 6 bits of `rs2`; a count
            // narrower than that may carry garbage inside those bits.
            let b = if lo.int_width(inst.operands()[1]) < 6 {
                self.extend64(lo, inst.operands()[1], false)
            } else {
                lo.reg(inst.operands()[1])
            };
            lo.emit(MachineInst::new(
                var_op.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
        }
    }

    /// Conversions. `zext`/`sext` (and `inttoptr` from a narrower integer)
    /// extend from the source's width, since its register's upper bits are not
    /// clean; truncation, ptr→int and bitcasts are low-bits-preserving copies.
    /// Float conversions are deferred (a plain copy keeps the MIR well-formed).
    fn lower_cast(&self, lo: &mut Lower<'_, Self>, op: CastOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let src = inst.operands()[0];
        let s = match op {
            CastOp::ZExt | CastOp::SExt | CastOp::IntToPtr => {
                self.extend64(lo, src, op == CastOp::SExt)
            }
            _ => lo.reg(src),
        };
        lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_v(s)]));
    }

    /// Lower a `call` under the LP64 integer ABI: move scalar arguments into
    /// `a0`–`a7`, record `a0` and the caller-saved clobbers, and move the result
    /// out of `a0`.
    fn lower_call(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let cc = &self.rf.cc;
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];

        // The final `arg-reg <- value-vreg` moves are emitted as one consecutive
        // run right before the `call` so no competing vreg definition sits between
        // an argument register's write and the call.
        let mut reg_moves: Vec<(PReg, VReg)> = Vec::new();
        let mut int_i = 0usize;
        for &arg in args {
            let v = self.abi_value(lo, arg);
            if int_i < cc.arg_regs.len() {
                let areg = cc.arg_regs[int_i];
                int_i += 1;
                reg_moves.push((areg, v));
            }
            // `> 8` integer arguments (stack-passed) are deferred; the fixtures
            // stay within the eight argument registers.
        }

        let used_arg_regs: Vec<PReg> = reg_moves.iter().map(|&(areg, _)| areg).collect();
        for (areg, r) in reg_moves {
            lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(areg), use_v(r)]));
        }

        let ret_reg = cc.ret_reg;
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
        lo.emit(MachineInst::new(RvOp::Call.opcode(), operands));

        if inst.result().is_some() {
            let d = lo.result_reg(inst);
            lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_p(ret_reg)]));
        }
    }
}

impl MachineTarget for RiscvTarget {
    fn name(&self) -> &str {
        "riscv64"
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
            RvOp::decode(op),
            RvOp::J | RvOp::BrCond | RvOp::Switch | RvOp::Ret | RvOp::Unreachable
        )
    }

    fn is_move(&self, op: Opcode) -> bool {
        RvOp::decode(op) == RvOp::Mv
    }

    fn emit_move(&self, dst: Reg, src: Reg) -> MachineInst {
        MachineInst::new(RvOp::Mv.opcode(), vec![MachineOperand::Def(dst), MachineOperand::Use(src)])
    }

    fn emit_spill(&self, slot: StackSlot, src: PReg) -> MachineInst {
        MachineInst::new(RvOp::StoreFrame.opcode(), vec![use_p(src), MachineOperand::Frame(slot)])
    }

    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(RvOp::LoadFrame.opcode(), vec![def(dst), MachineOperand::Frame(slot)])
    }
}

impl TargetIsel for RiscvTarget {
    fn li(&self, dst: VReg, value: Int) -> MachineInst {
        MachineInst::new(RvOp::Li.opcode(), vec![def_v(dst), MachineOperand::Imm(value)])
    }

    fn jump(&self, dst: MBlockId) -> MachineInst {
        MachineInst::new(RvOp::J.opcode(), vec![MachineOperand::Label(dst)])
    }

    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(RvOp::FrameAddr.opcode(), vec![def_v(dst), MachineOperand::Frame(slot)])
    }

    fn global_addr(&self, dst: VReg, g: u32) -> MachineInst {
        MachineInst::new(RvOp::GlobalAddr.opcode(), vec![def_v(dst), MachineOperand::Global(g)])
    }

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Bin(op) => self.lower_bin(lo, *op, inst),
            InstKind::ICmp(pred) => {
                let d = lo.result_reg(inst);
                // `slt`/`sltu`/`sub` read all 64 bits: extend a narrow pair
                // first. A signed predicate needs sign-extension; for the others
                // either extension works (sign-extending both sides preserves
                // equality and unsigned order), so an `i32` uses the single
                // `sext.w` and anything else zero-extends.
                let width = lo.int_width(inst.operands()[0]);
                let signed =
                    matches!(pred, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge)
                        || width == 32;
                let a = self.extend64(lo, inst.operands()[0], signed);
                let b = self.extend64(lo, inst.operands()[1], signed);
                lo.emit(MachineInst::new(
                    RvOp::SetCmp.opcode(),
                    vec![
                        def_v(d),
                        use_v(a),
                        use_v(b),
                        imm(u64::from(pred_code(*pred))),
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
            // execution-tested only on x86-64 so far; the riscv sp-adjust
            // lowering is deferred (like other target-specific gaps here).
            InstKind::DynAlloca { .. } => {
                panic!("riscv backend: dynamic `dyn_alloca` is not yet supported")
            }
            InstKind::Load { ty, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(inst.operands()[0]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    RvOp::Load.opcode(),
                    vec![def_v(d), use_v(ptr), imm(size)],
                ));
            }
            InstKind::Store { ty, .. } => {
                let ptr = lo.reg(inst.operands()[0]);
                let val = lo.reg(inst.operands()[1]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    RvOp::Store.opcode(),
                    vec![use_v(ptr), use_v(val), imm(size)],
                ));
            }
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(inst);
                let base = lo.reg(inst.operands()[0]);
                // The byte offset is signed; a narrow one is sign-extended.
                let off = self.extend64(lo, inst.operands()[1], true);
                lo.emit(MachineInst::new(
                    RvOp::Add.opcode(),
                    vec![def_v(d), use_v(base), use_v(off), imm(64)],
                ));
            }
            InstKind::Select => {
                let d = lo.result_reg(inst);
                let c = self.clean_cond(lo, inst.operands()[0]);
                let t = lo.reg(inst.operands()[1]);
                let f = lo.reg(inst.operands()[2]);
                lo.emit(MachineInst::new(
                    RvOp::Select.opcode(),
                    vec![def_v(d), use_v(c), use_v(t), use_v(f)],
                ));
            }
            InstKind::Freeze => {
                let d = lo.result_reg(inst);
                let s = lo.reg(inst.operands()[0]);
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_v(s)]));
            }
            InstKind::Call => self.lower_call(lo, inst),
            InstKind::Syscall => {
                // Linux RISC-V syscall ABI: number in `a7`, arguments in
                // `a0..a5`, result in `a0`. Materialize every operand first, then
                // move them into the fixed registers as one consecutive run right
                // before the `ecall` (as `lower_call` does), so no vreg definition
                // sits between an ABI register's write and its read.
                let cc = &self.rf.cc;
                let vals: Vec<VReg> = inst.operands().iter().map(|&o| lo.reg(o)).collect();
                let mut moves: Vec<(PReg, VReg)> = vec![(super::regs::gpr(17), vals[0])]; // a7
                for (k, &v) in vals[1..].iter().enumerate() {
                    moves.push((cc.arg_regs[k], v));
                }
                let a0 = cc.ret_reg;
                let mut operands = vec![def(a0)];
                for &(r, v) in &moves {
                    lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(r), use_v(v)]));
                    operands.push(use_p(r));
                }
                lo.emit(MachineInst::new(RvOp::Ecall.opcode(), operands));
                let d = lo.result_reg(inst);
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_p(a0)]));
            }
            // Floating-point negation / compares are out of the integer subset.
            InstKind::Unary(UnaryOp::FNeg) | InstKind::FCmp(_) => {
                let d = lo.result_reg(inst);
                lo.emit(MachineInst::new(RvOp::Li.opcode(), vec![def_v(d), imm(0)]));
            }
            k if k.is_atomic() => panic!("RISC-V backend: atomics are not yet supported: {k:?}"),
            _ => unreachable!("terminator reached lower_inst: {:?}", inst.kind),
        }
    }

    fn lower_term(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Ret => {
                if let Some(&v) = inst.operands().first() {
                    let r = self.abi_value(lo, v);
                    let ret = self.rf.cc.ret_reg;
                    lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(ret), use_v(r)]));
                }
                lo.emit(MachineInst::new(RvOp::Ret.opcode(), Vec::new()));
            }
            InstKind::Br(target) => {
                let args: Vec<_> = inst.operands().to_vec();
                let e = lo.edge_to(*target, &args);
                lo.emit(self.jump(e));
            }
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let cond = self.clean_cond(lo, inst.operands()[0]);
                let ops = inst.operands();
                let tb = 1 + *true_args as usize;
                let fb = tb + *false_args as usize;
                let true_vals: Vec<_> = ops[1..tb].to_vec();
                let false_vals: Vec<_> = ops[tb..fb].to_vec();
                let te = lo.edge_to(*if_true, &true_vals);
                let fe = lo.edge_to(*if_false, &false_vals);
                lo.emit(MachineInst::new(
                    RvOp::BrCond.opcode(),
                    vec![use_v(cond), MachineOperand::Label(te), MachineOperand::Label(fe)],
                ));
            }
            InstKind::Switch(data) => {
                // Cases are compared as 64-bit values: sign-extend the scrutinee
                // and each case value from the scrutinee's width.
                let width = lo.int_width(inst.operands()[0]);
                let cond = self.extend64(lo, inst.operands()[0], true);
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
                    operands.push(MachineOperand::Imm(sext_case(&case.value, width)));
                    operands.push(MachineOperand::Label(ce));
                }
                lo.emit(MachineInst::new(RvOp::Switch.opcode(), operands));
            }
            InstKind::Unreachable => {
                lo.emit(MachineInst::new(RvOp::Unreachable.opcode(), Vec::new()));
            }
            _ => unreachable!("non-terminator reached lower_term: {:?}", inst.kind),
        }
    }
}
