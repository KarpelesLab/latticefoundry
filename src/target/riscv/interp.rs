//! A small interpreter for the RISC-V backend's MIR, analogous to
//! [`crate::codegen::interp`] but over the [`RvOp`] opcode set.
//!
//! Since this host cannot execute RISC-V machine code, this is how the lowering
//! (isel) is validated *semantically*: it runs a lowered [`MachineFunction`] on
//! concrete integer inputs — before register allocation, so it exercises isel in
//! isolation — and returns the value the function computes, letting a test assert
//! `interp(select(f))(x) == expected(f)(x)`. Register operands (virtual or
//! physical) are just keys; ABI argument/return registers and call clobbers are
//! ordinary physical registers. It models one `RvOp` per step (e.g. `SetCmp` is
//! evaluated as compare-then-set, `Select` as the ternary choice, `Li` as a
//! load-immediate) rather than the encoder's multi-word idiom expansion.
//!
//! Like the AArch64 model it uses one shared flat address space so a pointer
//! handed across a call (a by-reference argument, an `alloca`'d slot) resolves in
//! the callee exactly as on hardware; a call's inputs are its `Use(physical)`
//! argument-register operands and its output the return register `a0`.

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, PReg, Reg, StackSlot};
use crate::codegen::target::MachineTarget;
use crate::support::DetHashMap;

use puremp::Int;

use super::isel::{RvOp, RiscvTarget};
use super::regs::gpr;

/// A cap on executed instructions, so a miscompiled loop fails fast.
const STEP_BUDGET: u64 = 5_000_000;

/// Run function `entry` of `funcs` with integer `args`, returning its return
/// value. `Err` on any modeled fault (division by zero, an unsupported opcode,
/// an out-of-budget loop, ...).
pub(super) fn run(
    target: &RiscvTarget,
    funcs: &[MachineFunction],
    entry: usize,
    args: &[Int],
) -> Result<Option<Int>, String> {
    let mut m = Machine { target, funcs, budget: STEP_BUDGET, mem: vec![0u8; 64], heap: 16 };

    let mf = m.funcs.get(entry).ok_or_else(|| format!("no function #{entry}"))?;
    let e = mf.entry().ok_or("call into a body-less function")?;
    let params: Vec<_> = mf.block(e).params.clone();
    let cc = target.call_conv();
    // Every entry parameter of the integer subset is a GPR, drawn from a0.. .
    let mut inputs: Vec<(PReg, Int)> = Vec::new();
    for (i, (_p, val)) in params.iter().zip(args).enumerate() {
        inputs.push((cc.arg_regs[i], val.clone()));
    }
    Ok(m.call(entry, &inputs)?.ret_val)
}

/// The whole-program interpreter state: a shared flat address space plus the
/// function table and a global step budget.
struct Machine<'a> {
    target: &'a RiscvTarget,
    funcs: &'a [MachineFunction],
    budget: u64,
    /// The single flat address space every activation's slots live in.
    mem: Vec<u8>,
    /// The bump cursor for the next activation's slot region.
    heap: u64,
}

/// One function activation's mutable state.
struct Frame {
    regs: DetHashMap<Reg, Int>,
    /// Absolute base address of each stack slot in the shared [`Machine::mem`].
    slot_base: Vec<u64>,
    /// Spill/aux slots addressed by handle rather than memory address.
    slot_val: DetHashMap<StackSlot, Int>,
    /// The most recent value moved into the physical return register (`a0`).
    ret_val: Option<Int>,
}

/// What a completed call hands back: the primary scalar return and a snapshot of
/// the return register (`a0`).
struct CallOut {
    ret_val: Option<Int>,
    regs: Vec<(PReg, Int)>,
}

fn mask(v: &Int, width: u32) -> Int {
    if width == 0 { Int::ZERO } else { v.mod_2k(width) }
}

fn signed(bits: &Int, width: u32) -> Int {
    if width > 0 && bits.bit(width - 1) {
        bits.sub(&Int::ONE.mul_2k(width))
    } else {
        bits.clone()
    }
}

/// Round `v` up to a multiple of `align` (≥ 1).
fn align_up(v: u64, align: u64) -> u64 {
    let a = align.max(1);
    v.div_ceil(a) * a
}

enum Flow {
    Next,
    Goto(crate::codegen::mir::MBlockId),
    Return,
}

impl Machine<'_> {
    /// Invoke function `fidx` with `inputs` pre-loaded into physical registers,
    /// running it to its `ret` and returning the primary scalar value plus the
    /// return-register snapshot.
    fn call(&mut self, fidx: usize, inputs: &[(PReg, Int)]) -> Result<CallOut, String> {
        let mf = self.funcs.get(fidx).ok_or_else(|| format!("no function #{fidx}"))?;
        let entry = mf.entry().ok_or("call into a body-less function")?;

        // Bump-allocate this activation's slot region from the shared address
        // space so slot addresses are globally unique (cross-call pointers work).
        let frame = mf.frame();
        let mut base = align_up(self.heap, 16);
        let mut slot_base = vec![0u64; frame.len()];
        for (i, b) in slot_base.iter_mut().enumerate() {
            let info = frame.slot(StackSlot::from_index(i));
            base = align_up(base, info.align.max(1));
            *b = base;
            base += info.size.max(1);
        }
        self.heap = align_up(base, 16);
        let need = self.heap as usize + 32;
        if need > self.mem.len() {
            self.mem.resize(need + 64, 0);
        }

        let mut fr = Frame {
            regs: DetHashMap::default(),
            slot_base,
            slot_val: DetHashMap::default(),
            ret_val: None,
        };
        for (p, v) in inputs {
            fr.regs.insert(Reg::Physical(*p), v.clone());
        }

        let mut block = entry;
        let mut ip = 0usize;
        loop {
            self.budget = self.budget.checked_sub(1).ok_or("step budget exhausted")?;
            let insts = &self.funcs[fidx].block(block).insts;
            let inst = insts.get(ip).ok_or("fell off the end of a block")?.clone();
            match self.step(&mut fr, &inst)? {
                Flow::Next => ip += 1,
                Flow::Goto(b) => {
                    block = b;
                    ip = 0;
                }
                Flow::Return => break,
            }
        }

        let a0 = gpr(super::regs::A0);
        let regs = fr
            .regs
            .get(&Reg::Physical(a0))
            .map(|v| vec![(a0, v.clone())])
            .unwrap_or_default();
        Ok(CallOut { ret_val: fr.ret_val, regs })
    }

    fn step(&mut self, fr: &mut Frame, inst: &MachineInst) -> Result<Flow, String> {
        let ops = &inst.operands;
        let op = RvOp::decode(inst.opcode);
        match op {
            RvOp::Li => {
                let d = def(ops, 0)?;
                fr.regs.insert(d, imm(ops, 1)?.clone());
            }
            RvOp::Mv => {
                let d = def(ops, 0)?;
                let s = self.rd(fr, use_reg(ops, 1)?);
                if d == Reg::Physical(self.target.call_conv().ret_reg) {
                    fr.ret_val = Some(s.clone());
                }
                fr.regs.insert(d, s);
            }
            RvOp::Add | RvOp::Sub | RvOp::And | RvOp::Or | RvOp::Xor | RvOp::Mul | RvOp::Mulh
            | RvOp::Sll | RvOp::Srl | RvOp::Sra => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let w = imm_u32(ops, 3)?;
                let res = match op {
                    RvOp::Add => a.add(&bb),
                    RvOp::Sub => a.sub(&bb),
                    RvOp::And => a.bitand(&bb),
                    RvOp::Or => a.bitor(&bb),
                    RvOp::Xor => a.bitxor(&bb),
                    RvOp::Mul => a.mul(&bb),
                    RvOp::Mulh => {
                        // Signed high half of the `w`-bit product.
                        let p = signed(&mask(&a, w), w).mul(&signed(&mask(&bb, w), w));
                        p.div_2k_trunc(w)
                    }
                    RvOp::Sll | RvOp::Srl | RvOp::Sra => {
                        let k = (bb.to_u64().unwrap_or(0) % u64::from(w.max(1))) as u32;
                        return self.set_shift(fr, d, op, &a, k, w);
                    }
                    _ => unreachable!(),
                };
                fr.regs.insert(d, mask(&res, w));
            }
            RvOp::Addi | RvOp::Andi | RvOp::Ori | RvOp::Xori => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let k = imm(ops, 2)?.clone();
                let w = imm_u32(ops, 3)?;
                // The 12-bit immediate is sign-extended before the operation.
                let k = signed(&k.mod_2k(12), 12);
                let res = match op {
                    RvOp::Addi => a.add(&k),
                    RvOp::Andi => a.bitand(&k),
                    RvOp::Ori => a.bitor(&k),
                    RvOp::Xori => a.bitxor(&k),
                    _ => unreachable!(),
                };
                fr.regs.insert(d, mask(&res, w));
            }
            RvOp::Slli | RvOp::Srli | RvOp::Srai => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let k = imm(ops, 2)?.to_u64().unwrap_or(0) as u32;
                let w = imm_u32(ops, 3)?;
                let sop = match op {
                    RvOp::Slli => RvOp::Sll,
                    RvOp::Srli => RvOp::Srl,
                    _ => RvOp::Sra,
                };
                return self.set_shift(fr, d, sop, &a, k, w);
            }
            RvOp::Div | RvOp::Divu | RvOp::Rem | RvOp::Remu => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let w = imm_u32(ops, 3)?;
                fr.regs.insert(d, self.divrem(op, &a, &bb, w)?);
            }
            RvOp::SetCmp => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let pred = imm(ops, 3)?.to_u64().unwrap_or(0) as u8;
                let w = imm_u32(ops, 4)?;
                let r = eval_pred(pred, &a, &bb, w);
                fr.regs.insert(d, if r { Int::ONE } else { Int::ZERO });
            }
            RvOp::Select => {
                let d = def(ops, 0)?;
                let c = self.rd(fr, use_reg(ops, 1)?);
                let t = self.rd(fr, use_reg(ops, 2)?);
                let f = self.rd(fr, use_reg(ops, 3)?);
                fr.regs.insert(d, if c.is_zero() { f } else { t });
            }
            RvOp::Load => {
                let d = def(ops, 0)?;
                let ptr = self.rd(fr, use_reg(ops, 1)?);
                let size = imm_u32(ops, 2)? as usize;
                let v = load_mem(&self.mem, addr(&ptr)?, size);
                fr.regs.insert(d, v);
            }
            RvOp::Store => {
                let ptr = self.rd(fr, use_reg(ops, 0)?);
                let val = self.rd(fr, use_reg(ops, 1)?);
                let size = imm_u32(ops, 2)? as usize;
                store_mem(&mut self.mem, addr(&ptr)?, size, &val);
            }
            RvOp::FrameAddr => {
                let d = def(ops, 0)?;
                let slot = frame_slot(ops, 1)?;
                fr.regs.insert(d, Int::from_u64(fr.slot_base[slot.index()]));
            }
            RvOp::StoreFrame => {
                let v = self.rd(fr, use_reg(ops, 0)?);
                let slot = frame_slot(ops, 1)?;
                fr.slot_val.insert(slot, v);
            }
            RvOp::LoadFrame => {
                let d = def(ops, 0)?;
                let slot = frame_slot(ops, 1)?;
                let v = fr.slot_val.get(&slot).cloned().unwrap_or(Int::ZERO);
                fr.regs.insert(d, v);
            }
            RvOp::Call => return self.exec_call(fr, inst),
            RvOp::Ret => return Ok(Flow::Return),
            RvOp::J => return Ok(Flow::Goto(label(ops, 0)?)),
            RvOp::BrCond => {
                let c = self.rd(fr, use_reg(ops, 0)?);
                let target = if c.is_zero() { label(ops, 2)? } else { label(ops, 1)? };
                return Ok(Flow::Goto(target));
            }
            RvOp::Switch => {
                let c = self.rd(fr, use_reg(ops, 0)?);
                let mut target = label(ops, 1)?;
                let mut i = 2;
                while i + 1 < ops.len() {
                    if let (MachineOperand::Imm(v), MachineOperand::Label(b)) = (&ops[i], &ops[i + 1])
                        && *v == c
                    {
                        target = *b;
                        break;
                    }
                    i += 2;
                }
                return Ok(Flow::Goto(target));
            }
            RvOp::GlobalAddr => return Err("global addressing is not modeled".into()),
            RvOp::Unreachable => return Err("reached an unreachable point (UB)".into()),
            // Prologue/epilogue pseudo-ops never appear in pre-regalloc MIR.
            RvOp::AddiSp | RvOp::SaveReg | RvOp::RestoreReg => {}
        }
        Ok(Flow::Next)
    }

    /// Model a shift (`Sll`/`Srl`/`Sra`) result and store it into `d`.
    fn set_shift(&self, fr: &mut Frame, d: Reg, op: RvOp, a: &Int, k: u32, w: u32) -> Result<Flow, String> {
        let a = mask(a, w);
        let res = if k >= w {
            Int::ZERO
        } else {
            match op {
                RvOp::Sll => mask(&a.mul_2k(k), w),
                RvOp::Srl => mask(&a.div_2k_trunc(k), w),
                RvOp::Sra => mask(&signed(&a, w).div_floor(&Int::ONE.mul_2k(k)), w),
                _ => unreachable!(),
            }
        };
        fr.regs.insert(d, res);
        Ok(Flow::Next)
    }

    fn exec_call(&mut self, fr: &mut Frame, inst: &MachineInst) -> Result<Flow, String> {
        let fidx = inst
            .operands
            .iter()
            .find_map(|o| match o {
                MachineOperand::Func(f) => Some(*f as usize),
                _ => None,
            })
            .ok_or("indirect calls are not modeled")?;
        // The call's inputs are exactly its `Use(physical)` operands: the argument
        // registers a0..a7.
        let inputs: Vec<(PReg, Int)> = inst
            .operands
            .iter()
            .filter_map(|o| match o {
                MachineOperand::Use(Reg::Physical(p)) => Some((*p, self.rd(fr, Reg::Physical(*p)))),
                _ => None,
            })
            .collect();
        let out = self.call(fidx, &inputs)?;
        for (p, v) in out.regs {
            fr.regs.insert(Reg::Physical(p), v);
        }
        Ok(Flow::Next)
    }

    fn divrem(&self, op: RvOp, a: &Int, b: &Int, w: u32) -> Result<Int, String> {
        match op {
            RvOp::Divu | RvOp::Remu => {
                let (ua, ub) = (mask(a, w), mask(b, w));
                if ub.is_zero() {
                    return Err("unsigned division by zero (UB)".into());
                }
                let (q, r) = ua.div_rem_trunc(&ub);
                Ok(mask(if op == RvOp::Divu { &q } else { &r }, w))
            }
            RvOp::Div | RvOp::Rem => {
                let (sa, sb) = (signed(&mask(a, w), w), signed(&mask(b, w), w));
                if sb.is_zero() {
                    return Err("signed division by zero (UB)".into());
                }
                let (q, r) = sa.div_rem_trunc(&sb);
                Ok(mask(if op == RvOp::Div { &q } else { &r }, w))
            }
            _ => unreachable!(),
        }
    }

    fn rd(&self, fr: &Frame, r: Reg) -> Int {
        // `x0` reads as zero regardless of any write.
        if r == Reg::Physical(gpr(super::regs::ZERO)) {
            return Int::ZERO;
        }
        fr.regs.get(&r).cloned().unwrap_or(Int::ZERO)
    }
}

/// Evaluate a packed [`super::isel::pred_code`] predicate on `w`-bit operands.
fn eval_pred(pred: u8, a: &Int, b: &Int, w: u32) -> bool {
    let (ua, ub) = (mask(a, w), mask(b, w));
    let (sa, sb) = (signed(&ua, w), signed(&ub, w));
    match pred {
        0 => ua == ub, // Eq
        1 => ua != ub, // Ne
        2 => ua < ub,  // Ult
        3 => ua <= ub, // Ule
        4 => ua > ub,  // Ugt
        5 => ua >= ub, // Uge
        6 => sa < sb,  // Slt
        7 => sa <= sb, // Sle
        8 => sa > sb,  // Sgt
        _ => sa >= sb, // Sge
    }
}

// --- operand decoding helpers ---------------------------------------------

fn def(ops: &[MachineOperand], i: usize) -> Result<Reg, String> {
    match ops.get(i) {
        Some(MachineOperand::Def(r)) => Ok(*r),
        _ => Err(format!("operand {i} is not a def")),
    }
}

fn use_reg(ops: &[MachineOperand], i: usize) -> Result<Reg, String> {
    match ops.get(i) {
        Some(MachineOperand::Use(r)) => Ok(*r),
        _ => Err(format!("operand {i} is not a use")),
    }
}

fn imm(ops: &[MachineOperand], i: usize) -> Result<&Int, String> {
    match ops.get(i) {
        Some(MachineOperand::Imm(v)) => Ok(v),
        _ => Err(format!("operand {i} is not an immediate")),
    }
}

fn imm_u32(ops: &[MachineOperand], i: usize) -> Result<u32, String> {
    Ok(imm(ops, i)?.to_u64().ok_or("immediate does not fit u64")? as u32)
}

fn label(ops: &[MachineOperand], i: usize) -> Result<crate::codegen::mir::MBlockId, String> {
    match ops.get(i) {
        Some(MachineOperand::Label(b)) => Ok(*b),
        _ => Err(format!("operand {i} is not a label")),
    }
}

fn frame_slot(ops: &[MachineOperand], i: usize) -> Result<StackSlot, String> {
    match ops.get(i) {
        Some(MachineOperand::Frame(s)) => Ok(*s),
        _ => Err(format!("operand {i} is not a frame slot")),
    }
}

fn addr(p: &Int) -> Result<usize, String> {
    p.to_u64().map(|a| a as usize).ok_or_else(|| "address does not fit u64".into())
}

fn load_mem(mem: &[u8], addr: usize, size: usize) -> Int {
    let mut acc = Int::ZERO;
    for i in (0..size).rev() {
        let byte = mem.get(addr + i).copied().unwrap_or(0);
        acc = acc.mul_2k(8).add(&Int::from_u64(u64::from(byte)));
    }
    acc
}

fn store_mem(mem: &mut Vec<u8>, addr: usize, size: usize, val: &Int) {
    if addr + size > mem.len() {
        mem.resize(addr + size + 16, 0);
    }
    let low = mask(val, 8 * size as u32);
    for i in 0..size {
        let byte = low.div_2k_trunc(8 * i as u32).mod_2k(8).to_u64().unwrap_or(0) as u8;
        mem[addr + i] = byte;
    }
}
