//! A small interpreter for the abstract virtual target's MIR.
//!
//! This is the executable semantics the lowering (isel) and register allocation
//! are validated against: it runs a [`MachineFunction`] on concrete integer
//! inputs and returns the value the function computes, so a test can assert that
//! `interp(lower(f))(x) == ir::semantics-expected(f)(x)`. It works on MIR
//! *before or after* register allocation — pre-allocation vregs and
//! post-allocation physical registers are both just register keys — which lets a
//! test check that allocation preserved behavior.
//!
//! It models only the integer subset the virtual target lowers (see
//! [`crate::codegen::vtarget::VOp`]); a float / `Unsupported` / unmodeled opcode
//! aborts with an error rather than guessing. Memory is a flat byte vector with
//! bump-allocated frame slots; spill slots hold whole register values in a
//! side table (width-agnostic), matching the `StackStore`/`StackLoad` model.

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, Reg, StackSlot};
use crate::codegen::target::MachineTarget;
use crate::codegen::vtarget::{VOp, VirtualTarget};
use crate::support::DetHashMap;

use puremp::Int;

/// A cap on executed instructions, so a miscompiled loop fails fast instead of
/// hanging the test suite.
const STEP_BUDGET: u64 = 5_000_000;

/// Run function `entry` of `funcs` with integer `args`, returning its return
/// value (`None` for a `void` function). `Err` on any modeled fault (division by
/// zero, an unsupported opcode, an out-of-budget loop, ...).
pub fn run(
    target: &VirtualTarget,
    funcs: &[MachineFunction],
    entry: usize,
    args: &[Int],
) -> Result<Option<Int>, String> {
    let mut budget = STEP_BUDGET;
    let mut interp = Interp { target, funcs, budget: &mut budget };
    interp.call(entry, args)
}

struct Interp<'a> {
    target: &'a VirtualTarget,
    funcs: &'a [MachineFunction],
    budget: &'a mut u64,
}

/// One function activation's mutable state.
struct Frame {
    regs: DetHashMap<Reg, Int>,
    mem: Vec<u8>,
    slot_base: Vec<u64>,
    slot_val: DetHashMap<StackSlot, Int>,
}

/// Mask `v` to the unsigned bit pattern of a `width`-bit type.
fn mask(v: &Int, width: u32) -> Int {
    if width == 0 { Int::ZERO } else { v.mod_2k(width) }
}

/// The signed value of a `width`-bit unsigned pattern.
fn signed(bits: &Int, width: u32) -> Int {
    if width > 0 && bits.bit(width - 1) {
        bits.sub(&Int::ONE.mul_2k(width))
    } else {
        bits.clone()
    }
}

impl Interp<'_> {
    fn call(&mut self, fidx: usize, args: &[Int]) -> Result<Option<Int>, String> {
        let mf = self.funcs.get(fidx).ok_or_else(|| format!("no function #{fidx}"))?;
        let entry = mf.entry().ok_or("call into a body-less function")?;

        // Lay out all frame slots by a downward-free bump (addresses start at 16
        // to keep null and low addresses clear).
        let frame = mf.frame();
        let mut slot_base = vec![0u64; frame.len()];
        let mut off = 16u64;
        for (i, base) in slot_base.iter_mut().enumerate() {
            let info = frame.slot(StackSlot::from_index(i));
            let align = info.align.max(1);
            off = off.div_ceil(align) * align;
            *base = off;
            off += info.size.max(1);
        }
        let mut fr = Frame {
            regs: DetHashMap::default(),
            mem: vec![0u8; (off + 32) as usize],
            slot_base,
            slot_val: DetHashMap::default(),
        };

        // Incoming arguments land in the physical argument registers.
        let cc = self.target.call_conv();
        for (areg, val) in cc.arg_regs.iter().zip(args) {
            fr.regs.insert(Reg::Physical(*areg), val.clone());
        }

        let mut block = entry;
        let mut ip = 0usize;
        loop {
            *self.budget = self.budget.checked_sub(1).ok_or("step budget exhausted")?;
            let insts = &mf.block(block).insts;
            let inst = insts.get(ip).ok_or("fell off the end of a block")?;
            match self.step(&mut fr, inst)? {
                Flow::Next => ip += 1,
                Flow::Goto(b) => {
                    block = b;
                    ip = 0;
                }
                Flow::Return(v) => return Ok(v),
            }
        }
    }

    fn step(&mut self, fr: &mut Frame, inst: &MachineInst) -> Result<Flow, String> {
        let ops = &inst.operands;
        match VOp::decode(inst.opcode) {
            VOp::Li => {
                let d = def(ops, 0)?;
                fr.regs.insert(d, imm(ops, 1)?.clone());
            }
            VOp::Move => {
                let d = def(ops, 0)?;
                let s = self.rd(fr, use_reg(ops, 1)?);
                fr.regs.insert(d, s);
            }
            VOp::Add | VOp::Sub | VOp::Mul | VOp::And | VOp::Or | VOp::Xor => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let b = self.rd(fr, use_reg(ops, 2)?);
                let w = imm_u32(ops, 3)?;
                let res = match VOp::decode(inst.opcode) {
                    VOp::Add => a.add(&b),
                    VOp::Sub => a.sub(&b),
                    VOp::Mul => a.mul(&b),
                    VOp::And => a.bitand(&b),
                    VOp::Or => a.bitor(&b),
                    VOp::Xor => a.bitxor(&b),
                    _ => unreachable!(),
                };
                fr.regs.insert(d, mask(&res, w));
            }
            VOp::UDiv | VOp::SDiv | VOp::URem | VOp::SRem => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let b = self.rd(fr, use_reg(ops, 2)?);
                let w = imm_u32(ops, 3)?;
                let res = self.div_rem(VOp::decode(inst.opcode), &a, &b, w)?;
                fr.regs.insert(d, res);
            }
            VOp::Shl | VOp::LShr | VOp::AShr => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let b = self.rd(fr, use_reg(ops, 2)?);
                let w = imm_u32(ops, 3)?;
                let res = shift(VOp::decode(inst.opcode), &a, &b, w);
                fr.regs.insert(d, res);
            }
            VOp::ICmp => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let b = self.rd(fr, use_reg(ops, 2)?);
                let pred = imm_i64(ops, 3)?;
                let w = imm_u32(ops, 4)?;
                let r = icmp(pred, &a, &b, w);
                fr.regs.insert(d, if r { Int::ONE } else { Int::ZERO });
            }
            VOp::Select => {
                let d = def(ops, 0)?;
                let c = self.rd(fr, use_reg(ops, 1)?);
                let t = self.rd(fr, use_reg(ops, 2)?);
                let f = self.rd(fr, use_reg(ops, 3)?);
                fr.regs.insert(d, if c.is_zero() { f } else { t });
            }
            VOp::Cast => {
                let d = def(ops, 0)?;
                let s = self.rd(fr, use_reg(ops, 1)?);
                let code = imm_i64(ops, 2)?;
                let srcw = imm_u32(ops, 3)?;
                let dstw = imm_u32(ops, 4)?;
                fr.regs.insert(d, cast(code, &s, srcw, dstw)?);
            }
            VOp::Load => {
                let d = def(ops, 0)?;
                let ptr = self.rd(fr, use_reg(ops, 1)?);
                let size = imm_u32(ops, 2)? as usize;
                let v = load_mem(&fr.mem, addr(&ptr)?, size);
                fr.regs.insert(d, v);
            }
            VOp::Store => {
                let ptr = self.rd(fr, use_reg(ops, 0)?);
                let val = self.rd(fr, use_reg(ops, 1)?);
                let size = imm_u32(ops, 2)? as usize;
                store_mem(&mut fr.mem, addr(&ptr)?, size, &val);
            }
            VOp::FrameAddr => {
                let d = def(ops, 0)?;
                let slot = frame_slot(ops, 1)?;
                fr.regs.insert(d, Int::from_u64(fr.slot_base[slot.index()]));
            }
            VOp::StackStore => {
                let v = self.rd(fr, use_reg(ops, 0)?);
                let slot = frame_slot(ops, 1)?;
                fr.slot_val.insert(slot, v);
            }
            VOp::StackLoad => {
                let d = def(ops, 0)?;
                let slot = frame_slot(ops, 1)?;
                let v = fr.slot_val.get(&slot).cloned().unwrap_or(Int::ZERO);
                fr.regs.insert(d, v);
            }
            VOp::Call => return self.exec_call(fr, inst),
            VOp::Ret => {
                let v = match ops.first() {
                    Some(MachineOperand::Use(r)) => Some(self.rd(fr, *r)),
                    _ => None,
                };
                return Ok(Flow::Return(v));
            }
            VOp::Jmp => return Ok(Flow::Goto(label(ops, 0)?)),
            VOp::BrCond => {
                let c = self.rd(fr, use_reg(ops, 0)?);
                let target = if c.is_zero() { label(ops, 2)? } else { label(ops, 1)? };
                return Ok(Flow::Goto(target));
            }
            VOp::Switch => {
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
            VOp::GlobalAddr => return Err("global addressing is not modeled".into()),
            VOp::Unreachable => return Err("reached an unreachable point (UB)".into()),
            VOp::Unsupported => return Err("executed an unsupported (non-integer) opcode".into()),
        }
        Ok(Flow::Next)
    }

    fn exec_call(&mut self, fr: &mut Frame, inst: &MachineInst) -> Result<Flow, String> {
        let cc = self.target.call_conv();
        let fidx = inst
            .operands
            .iter()
            .find_map(|o| match o {
                MachineOperand::Func(f) => Some(*f as usize),
                _ => None,
            })
            .ok_or("indirect calls are not modeled")?;
        // Arguments are in the physical argument registers named by the call's
        // `Use(Physical(arg_i))` operands, in order.
        let mut args = Vec::new();
        for &areg in &cc.arg_regs {
            let used = inst
                .operands
                .iter()
                .any(|o| matches!(o, MachineOperand::Use(Reg::Physical(p)) if *p == areg));
            if used {
                args.push(self.rd(fr, Reg::Physical(areg)));
            }
        }
        let ret = self.call(fidx, &args)?;
        fr.regs.insert(Reg::Physical(cc.ret_reg), ret.unwrap_or(Int::ZERO));
        Ok(Flow::Next)
    }

    fn div_rem(&self, op: VOp, a: &Int, b: &Int, w: u32) -> Result<Int, String> {
        match op {
            VOp::UDiv | VOp::URem => {
                if b.is_zero() {
                    return Err("unsigned division by zero (UB)".into());
                }
                let (q, r) = a.div_rem_trunc(b);
                Ok(mask(if op == VOp::UDiv { &q } else { &r }, w))
            }
            VOp::SDiv | VOp::SRem => {
                let (sa, sb) = (signed(a, w), signed(b, w));
                if sb.is_zero() {
                    return Err("signed division by zero (UB)".into());
                }
                let (q, r) = sa.div_rem_trunc(&sb);
                Ok(mask(if op == VOp::SDiv { &q } else { &r }, w))
            }
            _ => unreachable!(),
        }
    }

    fn rd(&self, fr: &Frame, r: Reg) -> Int {
        fr.regs.get(&r).cloned().unwrap_or(Int::ZERO)
    }
}

enum Flow {
    Next,
    Goto(crate::codegen::mir::MBlockId),
    Return(Option<Int>),
}

fn shift(op: VOp, a: &Int, b: &Int, w: u32) -> Int {
    // Read the operand as its `w`-bit pattern (registers may carry dirty high
    // bits from a wider input value).
    let a = mask(a, w);
    let k = match b.to_u64() {
        Some(k) if k < u64::from(w) => k as u32,
        // A shift amount out of range is poison; materialize zero.
        _ => return Int::ZERO,
    };
    match op {
        VOp::Shl => mask(&a.mul_2k(k), w),
        VOp::LShr => mask(&a.div_2k_trunc(k), w),
        VOp::AShr => mask(&signed(&a, w).div_floor(&Int::ONE.mul_2k(k)), w),
        _ => unreachable!(),
    }
}

fn icmp(pred: i64, a: &Int, b: &Int, w: u32) -> bool {
    // Compare the `w`-bit patterns (mask off any dirty high bits first).
    let a = mask(a, w);
    let b = mask(b, w);
    match pred {
        0 => a == b,
        1 => a != b,
        2 => a > b,
        3 => a >= b,
        4 => a < b,
        5 => a <= b,
        6 => signed(&a, w) > signed(&b, w),
        7 => signed(&a, w) >= signed(&b, w),
        8 => signed(&a, w) < signed(&b, w),
        9 => signed(&a, w) <= signed(&b, w),
        _ => false,
    }
}

fn cast(code: i64, s: &Int, srcw: u32, dstw: u32) -> Result<Int, String> {
    Ok(match code {
        0 => mask(s, dstw),                          // Trunc
        1 => mask(s, srcw),                          // ZExt
        2 => mask(&signed(&mask(s, srcw), srcw), dstw), // SExt
        3..=5 => mask(s, dstw),                      // PtrToInt / IntToPtr / Bitcast
        _ => return Err("unmodeled cast".into()),
    })
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

fn imm_i64(ops: &[MachineOperand], i: usize) -> Result<i64, String> {
    imm(ops, i)?.to_i64().ok_or_else(|| "immediate does not fit i64".into())
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
    for i in 0..size {
        let byte = val.div_2k_trunc(8 * i as u32).mod_2k(8).to_u64().unwrap_or(0) as u8;
        mem[addr + i] = byte;
    }
}
