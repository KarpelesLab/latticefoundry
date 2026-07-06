//! A small interpreter for the AArch64 backend's MIR, analogous to
//! [`crate::codegen::interp`] but over the [`A64Op`] opcode set.
//!
//! Since this host cannot execute AArch64 machine code, this is how the lowering
//! (isel) is validated *semantically*: it runs a lowered [`MachineFunction`] on
//! concrete integer inputs — before register allocation, so it exercises isel in
//! isolation — and returns the value the function computes, letting a test assert
//! `interp(select(f))(x) == expected(f)(x)`. Register operands (virtual or
//! physical) are just keys; ABI argument/return registers and call clobbers are
//! ordinary physical registers. It models one A64 op per step (e.g. `CmpCset` is
//! evaluated as compare-then-set, `MovRI` as a load-immediate) rather than the
//! encoder's multi-word expansion.

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, Reg, RegClass, StackSlot};
use crate::codegen::target::MachineTarget;
use crate::support::DetHashMap;

use puremp::Int;

use super::isel::{A64Op, AArch64Target};

/// A cap on executed instructions, so a miscompiled loop fails fast.
const STEP_BUDGET: u64 = 5_000_000;

/// Run function `entry` of `funcs` with integer `args`, returning its return
/// value. `Err` on any modeled fault (division by zero, an unsupported opcode,
/// an out-of-budget loop, ...).
pub(super) fn run(
    target: &AArch64Target,
    funcs: &[MachineFunction],
    entry: usize,
    args: &[Int],
) -> Result<Option<Int>, String> {
    let mut budget = STEP_BUDGET;
    let mut interp = Interp { target, funcs, budget: &mut budget };
    interp.call(entry, args)
}

struct Interp<'a> {
    target: &'a AArch64Target,
    funcs: &'a [MachineFunction],
    budget: &'a mut u64,
}

/// One function activation's mutable state.
struct Frame {
    regs: DetHashMap<Reg, Int>,
    mem: Vec<u8>,
    slot_base: Vec<u64>,
    slot_val: DetHashMap<StackSlot, Int>,
    /// The most recent value moved into a physical return register (`x0` or `v0`).
    /// The lowering writes the return value there immediately before `ret`, so
    /// this captures a float return (in `v0`) as well as an integer one (`x0`).
    ret_val: Option<Int>,
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

enum Flow {
    Next,
    Goto(crate::codegen::mir::MBlockId),
    Return(Option<Int>),
}

impl Interp<'_> {
    fn call(&mut self, fidx: usize, args: &[Int]) -> Result<Option<Int>, String> {
        let mf = self.funcs.get(fidx).ok_or_else(|| format!("no function #{fidx}"))?;
        let entry = mf.entry().ok_or("call into a body-less function")?;

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
            ret_val: None,
        };

        // Route each positional argument into its physical argument register by
        // the parameter's register class: integers into x0.. and floats into
        // v0.. (separate counters), matching the framework prologue.
        let cc = self.target.call_conv();
        let params: Vec<_> = mf.block(entry).params.clone();
        let mut int_i = 0usize;
        let mut fp_i = 0usize;
        for (&p, val) in params.iter().zip(args) {
            let areg = match mf.vreg_class(p) {
                RegClass::Gpr => {
                    let r = cc.arg_regs[int_i];
                    int_i += 1;
                    r
                }
                RegClass::Fp => {
                    let r = cc.fp_arg_regs[fp_i];
                    fp_i += 1;
                    r
                }
            };
            fr.regs.insert(Reg::Physical(areg), val.clone());
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
        let op = A64Op::decode(inst.opcode);
        match op {
            A64Op::MovRI => {
                let d = def(ops, 0)?;
                fr.regs.insert(d, imm(ops, 1)?.clone());
            }
            A64Op::MovRR => {
                let d = def(ops, 0)?;
                let s = self.rd(fr, use_reg(ops, 1)?);
                let cc = self.target.call_conv();
                if d == Reg::Physical(cc.ret_reg) || d == Reg::Physical(cc.fp_ret_reg) {
                    fr.ret_val = Some(s.clone());
                }
                fr.regs.insert(d, s);
            }
            A64Op::Add | A64Op::Sub | A64Op::And | A64Op::Or | A64Op::Eor | A64Op::Mul => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let w = imm_u32(ops, 3)?;
                let res = match op {
                    A64Op::Add => a.add(&bb),
                    A64Op::Sub => a.sub(&bb),
                    A64Op::And => a.bitand(&bb),
                    A64Op::Or => a.bitor(&bb),
                    A64Op::Eor => a.bitxor(&bb),
                    A64Op::Mul => a.mul(&bb),
                    _ => unreachable!(),
                };
                fr.regs.insert(d, mask(&res, w));
            }
            A64Op::AddI | A64Op::SubI => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let k = imm(ops, 2)?.clone();
                let w = imm_u32(ops, 3)?;
                let res = if op == A64Op::AddI { a.add(&k) } else { a.sub(&k) };
                fr.regs.insert(d, mask(&res, w));
            }
            A64Op::Sdiv | A64Op::Udiv => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let w = imm_u32(ops, 3)?;
                fr.regs.insert(d, self.div(op, &a, &bb, w)?);
            }
            A64Op::Msub => {
                let d = def(ops, 0)?;
                let m = self.rd(fr, use_reg(ops, 1)?);
                let n = self.rd(fr, use_reg(ops, 2)?);
                let a = self.rd(fr, use_reg(ops, 3)?);
                let w = imm_u32(ops, 4)?;
                fr.regs.insert(d, mask(&a.sub(&m.mul(&n)), w));
            }
            A64Op::LslI | A64Op::LsrI | A64Op::AsrI => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let k = imm(ops, 2)?.to_u64().unwrap_or(0) as u32;
                let w = imm_u32(ops, 3)?;
                fr.regs.insert(d, shift(op, &a, k, w));
            }
            A64Op::LslV | A64Op::LsrV | A64Op::AsrV => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let w = imm_u32(ops, 3)?;
                let k = (bb.to_u64().unwrap_or(0) % u64::from(w.max(1))) as u32;
                fr.regs.insert(d, shift(op, &a, k, w));
            }
            A64Op::CmpCset => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let cc = imm(ops, 3)?.to_u64().unwrap_or(0) as u8;
                let w = imm_u32(ops, 4)?;
                let r = eval_cond(cc, &a, &bb, w);
                fr.regs.insert(d, if r { Int::ONE } else { Int::ZERO });
            }
            A64Op::Csel => {
                let d = def(ops, 0)?;
                let c = self.rd(fr, use_reg(ops, 1)?);
                let t = self.rd(fr, use_reg(ops, 2)?);
                let f = self.rd(fr, use_reg(ops, 3)?);
                fr.regs.insert(d, if c.is_zero() { f } else { t });
            }
            A64Op::Load => {
                let d = def(ops, 0)?;
                let ptr = self.rd(fr, use_reg(ops, 1)?);
                let size = imm_u32(ops, 2)? as usize;
                let v = load_mem(&fr.mem, addr(&ptr)?, size);
                fr.regs.insert(d, v);
            }
            A64Op::Store => {
                let ptr = self.rd(fr, use_reg(ops, 0)?);
                let val = self.rd(fr, use_reg(ops, 1)?);
                let size = imm_u32(ops, 2)? as usize;
                store_mem(&mut fr.mem, addr(&ptr)?, size, &val);
            }
            A64Op::FrameAddr => {
                let d = def(ops, 0)?;
                let slot = frame_slot(ops, 1)?;
                fr.regs.insert(d, Int::from_u64(fr.slot_base[slot.index()]));
            }
            A64Op::StoreFrame => {
                let v = self.rd(fr, use_reg(ops, 0)?);
                let slot = frame_slot(ops, 1)?;
                fr.slot_val.insert(slot, v);
            }
            A64Op::LoadFrame => {
                let d = def(ops, 0)?;
                let slot = frame_slot(ops, 1)?;
                let v = fr.slot_val.get(&slot).cloned().unwrap_or(Int::ZERO);
                fr.regs.insert(d, v);
            }
            A64Op::Call => return self.exec_call(fr, inst),
            A64Op::Ret => {
                // The lowering moved the value into the class-appropriate return
                // register (`x0` or `v0`) immediately before this `ret`.
                return Ok(Flow::Return(fr.ret_val.clone()));
            }
            A64Op::B => return Ok(Flow::Goto(label(ops, 0)?)),
            A64Op::BrCond => {
                let c = self.rd(fr, use_reg(ops, 0)?);
                let target = if c.is_zero() { label(ops, 2)? } else { label(ops, 1)? };
                return Ok(Flow::Goto(target));
            }
            A64Op::Switch => {
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
            A64Op::GlobalAddr => return Err("global addressing is not modeled".into()),
            A64Op::Unreachable => return Err("reached an unreachable point (UB)".into()),

            // --- scalar floating-point ------------------------------------
            A64Op::FAdd | A64Op::FSub | A64Op::FMul | A64Op::FDiv => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let w = imm_u32(ops, 3)?;
                fr.regs.insert(d, fbin(op, &a, &bb, w));
            }
            A64Op::FNeg => {
                let d = def(ops, 0)?;
                let s = self.rd(fr, use_reg(ops, 1)?);
                let w = imm_u32(ops, 2)?;
                let raw = fbits(&s, w);
                let sign = if w >= 64 { 0x8000_0000_0000_0000 } else { 0x8000_0000 };
                fr.regs.insert(d, Int::from_u64(raw ^ sign));
            }
            A64Op::Fcmp => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let packed = imm(ops, 3)?.to_u64().unwrap_or(0);
                let w = imm_u32(ops, 4)?;
                let r = eval_fcmp(packed, &a, &bb, w);
                fr.regs.insert(d, if r { Int::ONE } else { Int::ZERO });
            }
            A64Op::LoadFConst => {
                let d = def(ops, 0)?;
                let bits = imm(ops, 1)?.to_u64().unwrap_or(0);
                let w = imm_u32(ops, 2)?;
                let raw = if w >= 64 { bits } else { bits & 0xFFFF_FFFF };
                fr.regs.insert(d, Int::from_u64(raw));
            }
            A64Op::Fcvt => {
                let d = def(ops, 0)?;
                let s = self.rd(fr, use_reg(ops, 1)?);
                let dst_w = imm_u32(ops, 2)?;
                let src_w = imm_u32(ops, 3)?;
                let x = fval(&s, src_w);
                fr.regs.insert(d, fenc(x, dst_w));
            }
            A64Op::Fcvtzs | A64Op::Fcvtzu => {
                let d = def(ops, 0)?;
                let s = self.rd(fr, use_reg(ops, 1)?);
                let dst_int_w = imm_u32(ops, 2)?;
                let src_flt_w = imm_u32(ops, 3)?;
                let x = fval(&s, src_flt_w).trunc();
                let v = if op == A64Op::Fcvtzs {
                    mask(&Int::from_i64(x as i64), dst_int_w)
                } else {
                    mask(&Int::from_u64(x as u64), dst_int_w)
                };
                fr.regs.insert(d, v);
            }
            A64Op::Scvtf | A64Op::Ucvtf => {
                let d = def(ops, 0)?;
                let s = self.rd(fr, use_reg(ops, 1)?);
                let dst_flt_w = imm_u32(ops, 2)?;
                let src_int_w = imm_u32(ops, 3)?;
                let bits = mask(&s, src_int_w);
                let x = if op == A64Op::Scvtf {
                    signed(&bits, src_int_w).to_i64().unwrap_or(0) as f64
                } else {
                    bits.to_u64().unwrap_or(0) as f64
                };
                fr.regs.insert(d, fenc(x, dst_flt_w));
            }
            // Prologue/epilogue pseudo-ops never appear in pre-regalloc MIR.
            A64Op::StpFpLr
            | A64Op::LdpFpLr
            | A64Op::MovFpSp
            | A64Op::SubSp
            | A64Op::AddSp
            | A64Op::SaveReg
            | A64Op::RestoreReg => {}
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
        // Reconstruct the positional argument list in *parameter order* by reading
        // each callee parameter's physical argument register (x-regs and v-regs
        // are counted separately, exactly as the argument-passing lowering did).
        let callee_mf = self.funcs.get(fidx).ok_or_else(|| format!("no function #{fidx}"))?;
        let params: Vec<_> =
            callee_mf.entry().map(|e| callee_mf.block(e).params.clone()).unwrap_or_default();
        let mut args = Vec::with_capacity(params.len());
        let mut int_i = 0usize;
        let mut fp_i = 0usize;
        for &p in &params {
            let areg = match callee_mf.vreg_class(p) {
                RegClass::Gpr => {
                    let r = cc.arg_regs[int_i];
                    int_i += 1;
                    r
                }
                RegClass::Fp => {
                    let r = cc.fp_arg_regs[fp_i];
                    fp_i += 1;
                    r
                }
            };
            args.push(self.rd(fr, Reg::Physical(areg)));
        }
        let ret = self.call(fidx, &args)?.unwrap_or(Int::ZERO);
        // Deposit the result in both return registers; the caller's lowering reads
        // exactly the class-appropriate one, and the value is identical.
        fr.regs.insert(Reg::Physical(cc.ret_reg), ret.clone());
        fr.regs.insert(Reg::Physical(cc.fp_ret_reg), ret);
        Ok(Flow::Next)
    }

    fn div(&self, op: A64Op, a: &Int, b: &Int, w: u32) -> Result<Int, String> {
        match op {
            A64Op::Udiv => {
                if b.is_zero() {
                    return Err("unsigned division by zero (UB)".into());
                }
                let (q, _) = a.div_rem_trunc(b);
                Ok(mask(&q, w))
            }
            A64Op::Sdiv => {
                let (sa, sb) = (signed(a, w), signed(b, w));
                if sb.is_zero() {
                    return Err("signed division by zero (UB)".into());
                }
                let (q, _) = sa.div_rem_trunc(&sb);
                Ok(mask(&q, w))
            }
            _ => unreachable!(),
        }
    }

    fn rd(&self, fr: &Frame, r: Reg) -> Int {
        fr.regs.get(&r).cloned().unwrap_or(Int::ZERO)
    }
}

fn shift(op: A64Op, a: &Int, k: u32, w: u32) -> Int {
    let a = mask(a, w);
    if k >= w {
        return Int::ZERO;
    }
    match op {
        A64Op::LslI | A64Op::LslV => mask(&a.mul_2k(k), w),
        A64Op::LsrI | A64Op::LsrV => mask(&a.div_2k_trunc(k), w),
        A64Op::AsrI | A64Op::AsrV => mask(&signed(&a, w).div_floor(&Int::ONE.mul_2k(k)), w),
        _ => unreachable!(),
    }
}

// --- floating-point helpers -----------------------------------------------

/// The raw IEEE bit pattern a register holds (masked to the float width).
fn fbits(v: &Int, width: u32) -> u64 {
    let raw = v.to_u64().unwrap_or(0);
    if width >= 64 { raw } else { raw & 0xFFFF_FFFF }
}

/// Decode a register's stored bit pattern to the `f64` it denotes at `width`.
fn fval(v: &Int, width: u32) -> f64 {
    if width >= 64 {
        f64::from_bits(fbits(v, 64))
    } else {
        f64::from(f32::from_bits(fbits(v, 32) as u32))
    }
}

/// Encode an `f64` value into the register bit pattern of a `width`-bit float.
fn fenc(x: f64, width: u32) -> Int {
    if width >= 64 {
        Int::from_u64(x.to_bits())
    } else {
        Int::from_u64(u64::from((x as f32).to_bits()))
    }
}

/// Execute a scalar FP binary op at the given width, in that width's precision.
fn fbin(op: A64Op, a: &Int, b: &Int, width: u32) -> Int {
    if width >= 64 {
        let (x, y) = (f64::from_bits(fbits(a, 64)), f64::from_bits(fbits(b, 64)));
        let r = match op {
            A64Op::FAdd => x + y,
            A64Op::FSub => x - y,
            A64Op::FMul => x * y,
            _ => x / y,
        };
        Int::from_u64(r.to_bits())
    } else {
        let (x, y) = (f32::from_bits(fbits(a, 32) as u32), f32::from_bits(fbits(b, 32) as u32));
        let r = match op {
            A64Op::FAdd => x + y,
            A64Op::FSub => x - y,
            A64Op::FMul => x * y,
            _ => x / y,
        };
        Int::from_u64(u64::from(r.to_bits()))
    }
}

/// Model the NZCV flags an A64 `fcmp` sets, then evaluate the packed condition
/// plan (`cond | combine<<4 | cond2<<8`) — validating the isel's condition
/// mapping the same way hardware would.
fn eval_fcmp(packed: u64, a: &Int, b: &Int, width: u32) -> bool {
    let (x, y) = (fval(a, width), fval(b, width));
    // fcmp NZCV: unordered ⇒ (0,0,1,1); a<b ⇒ (1,0,0,0); a==b ⇒ (0,1,1,0);
    // a>b ⇒ (0,0,1,0).
    let (n, z, c, v) = if x.is_nan() || y.is_nan() {
        (false, false, true, true)
    } else if x < y {
        (true, false, false, false)
    } else if x == y {
        (false, true, true, false)
    } else {
        (false, false, true, false)
    };
    let cond = (packed & 0xF) as u8;
    let combine = (packed >> 4) & 0xF;
    let cond2 = ((packed >> 8) & 0xF) as u8;
    let r1 = cond_holds(cond, n, z, c, v);
    match combine {
        1 => r1 && cond_holds(cond2, n, z, c, v), // And
        2 => r1 || cond_holds(cond2, n, z, c, v), // Or
        _ => r1,
    }
}

/// Whether an A64 condition code holds for the given NZCV flags.
fn cond_holds(cc: u8, n: bool, z: bool, c: bool, v: bool) -> bool {
    match cc {
        0x0 => z,                 // EQ
        0x1 => !z,                // NE
        0x2 => c,                 // CS/HS
        0x3 => !c,                // CC/LO
        0x4 => n,                 // MI
        0x5 => !n,                // PL
        0x6 => v,                 // VS
        0x7 => !v,                // VC
        0x8 => c && !z,           // HI
        0x9 => !c || z,           // LS
        0xA => n == v,            // GE
        0xB => n != v,            // LT
        0xC => !z && (n == v),    // GT
        0xD => z || (n != v),     // LE
        _ => true,                // AL
    }
}

/// Evaluate an A64 condition code against the compare `a - b` of `w`-bit values.
fn eval_cond(cc: u8, a: &Int, b: &Int, w: u32) -> bool {
    let (ua, ub) = (mask(a, w), mask(b, w));
    let (sa, sb) = (signed(&ua, w), signed(&ub, w));
    match cc {
        0x0 => ua == ub,  // EQ
        0x1 => ua != ub,  // NE
        0x2 => ua >= ub,  // HS (unsigned >=)
        0x3 => ua < ub,   // LO (unsigned <)
        0x8 => ua > ub,   // HI (unsigned >)
        0x9 => ua <= ub,  // LS (unsigned <=)
        0xA => sa >= sb,  // GE (signed >=)
        0xB => sa < sb,   // LT (signed <)
        0xC => sa > sb,   // GT (signed >)
        0xD => sa <= sb,  // LE (signed <=)
        _ => false,
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
    for i in 0..size {
        let byte = val.div_2k_trunc(8 * i as u32).mod_2k(8).to_u64().unwrap_or(0) as u8;
        mem[addr + i] = byte;
    }
}
