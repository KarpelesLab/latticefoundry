//! An interpreter for the Thumb backend's MIR ([`ThOp`]), analogous to the
//! AArch64 and RISC-V ones: it runs lowered [`MachineFunction`]s *before*
//! register allocation, so it checks instruction selection in isolation from
//! the allocator and the encoder.
//!
//! Every register holds 32 bits and every op is modeled as the instruction
//! the encoder emits for it: a narrow value keeps whatever its computation
//! left above its width (an `i8` add of 200 + 100 holds 300), so an isel that
//! forgets to extend before an op that reads those bits is caught. Memory is
//! the linked image of [`super::sim::link`] (so globals sit where the machine
//! code sees them); each activation's frame slots and outgoing argument area
//! are carved below a stack pointer, and a callee reads its incoming stack
//! arguments from its caller's outgoing area. A call to a declared function
//! runs the Rust implementation of that runtime helper
//! ([`super::sim::aeabi`]).

use std::collections::HashMap;

use crate::codegen::mir::{MachineFunction, MachineOperand, Reg};
use crate::support::DetHashMap;

use super::isel::ThOp;
use super::sim::{Memory, aeabi};

const STEP_BUDGET: u64 = 5_000_000;

/// The program the interpreter runs: machine functions by function index,
/// their names and the global names (to find symbol addresses), and the image
/// symbols.
pub(super) struct Program<'a> {
    pub(super) funcs: &'a [MachineFunction],
    pub(super) func_names: &'a [String],
    pub(super) global_names: &'a [String],
    pub(super) symbols: &'a HashMap<String, u32>,
}

struct Machine<'a> {
    p: &'a Program<'a>,
    mem: Memory,
    sp: u32,
    budget: u64,
}

/// Call function `entry` with word arguments in `r0`–`r3`, returning
/// `r0`–`r3` at its return.
pub(super) fn run(p: &Program<'_>, mem: Memory, entry: usize, args: &[u32]) -> Result<[u32; 4], String> {
    let mut m = Machine { p, mem, sp: super::sim::STACK_TOP, budget: STEP_BUDGET };
    let mut inputs = [0u32; 4];
    inputs[..args.len()].copy_from_slice(args);
    m.call(entry, inputs, 0)
}

fn reg(op: &MachineOperand) -> Result<Reg, String> {
    op.reg().ok_or_else(|| format!("expected a register operand, found {op:?}"))
}

fn imm(op: &MachineOperand) -> Result<u64, String> {
    match op {
        MachineOperand::Imm(v) => Ok(v.to_u64().or_else(|| v.to_i64().map(|i| i as u64)).unwrap_or(0)),
        other => Err(format!("expected an immediate, found {other:?}")),
    }
}

/// Whether condition `cc` holds after `cmp a, b`.
fn cond(cc: u64, a: u32, b: u32) -> bool {
    let (sa, sb) = (a as i32, b as i32);
    match cc {
        0x0 => a == b,
        0x1 => a != b,
        0x2 => a >= b,
        0x3 => a < b,
        0x8 => a > b,
        0x9 => a <= b,
        0xa => sa >= sb,
        0xb => sa < sb,
        0xc => sa > sb,
        0xd => sa <= sb,
        _ => true,
    }
}

fn shift(v: u32, ty: u32, amount: u32) -> u32 {
    let n = amount & 0xff;
    match ty {
        0 => v.checked_shl(n).unwrap_or(0),
        1 => v.checked_shr(n).unwrap_or(0),
        _ => ((v as i32) >> n.min(31)) as u32,
    }
}

impl Machine<'_> {
    fn call(&mut self, f: usize, inputs: [u32; 4], incoming: u32) -> Result<[u32; 4], String> {
        let mf = self.p.funcs.get(f).ok_or_else(|| format!("no function #{f}"))?;
        let Some(entry) = mf.entry() else {
            let mut r = inputs;
            aeabi(&self.p.func_names[f], &mut r)?;
            return Ok(r);
        };
        // The frame: slots, then the outgoing area at the bottom.
        let saved_sp = self.sp;
        let mut top = self.sp;
        let mut slot_addr = Vec::with_capacity(mf.frame().len());
        for i in 0..mf.frame().len() {
            let info = mf.frame().slot(crate::codegen::mir::StackSlot::from_index(i));
            top = (top - info.size.max(1) as u32) & !(info.align.clamp(4, 8) as u32 - 1);
            slot_addr.push(top);
        }
        let out_base = (top - mf.frame().outgoing() as u32) & !7;
        self.sp = out_base;

        let mut regs: DetHashMap<Reg, u32> = DetHashMap::default();
        for (k, &v) in inputs.iter().enumerate() {
            regs.insert(Reg::Physical(super::regs::gpr(k as u16)), v);
        }
        let rd = |regs: &DetHashMap<Reg, u32>, op: &MachineOperand| -> Result<u32, String> {
            let r = reg(op)?;
            regs.get(&r).copied().ok_or_else(|| format!("read of undefined register {r:?}"))
        };
        let mut block = entry;
        loop {
            let insts = &mf.block(block).insts;
            let mut next = None;
            for inst in insts {
                self.budget = self.budget.checked_sub(1).ok_or("step budget exhausted")?;
                let o = &inst.operands;
                let op = ThOp::decode(inst.opcode);
                let set = |regs: &mut DetHashMap<Reg, u32>, i: usize, v: u32| -> Result<(), String> {
                    regs.insert(reg(&o[i])?, v);
                    Ok(())
                };
                match op {
                    ThOp::Mov => {
                        let v = rd(&regs, &o[1])?;
                        set(&mut regs, 0, v)?;
                    }
                    ThOp::MovImm => set(&mut regs, 0, imm(&o[1])? as u32)?,
                    ThOp::Add | ThOp::Sub | ThOp::And | ThOp::Orr | ThOp::Eor | ThOp::Mul | ThOp::Sdiv
                    | ThOp::Udiv | ThOp::Srem | ThOp::Urem | ThOp::Lsl | ThOp::Lsr | ThOp::Asr => {
                        let (a, b) = (rd(&regs, &o[1])?, rd(&regs, &o[2])?);
                        let div0 = || Err::<u32, String>("division by zero".into());
                        let v = match op {
                            ThOp::Add => a.wrapping_add(b),
                            ThOp::Sub => a.wrapping_sub(b),
                            ThOp::And => a & b,
                            ThOp::Orr => a | b,
                            ThOp::Eor => a ^ b,
                            ThOp::Mul => a.wrapping_mul(b),
                            ThOp::Sdiv if b != 0 => (a as i32).wrapping_div(b as i32) as u32,
                            ThOp::Udiv if b != 0 => a / b,
                            ThOp::Srem if b != 0 => (a as i32).wrapping_rem(b as i32) as u32,
                            ThOp::Urem if b != 0 => a % b,
                            ThOp::Lsl => shift(a, 0, b),
                            ThOp::Lsr => shift(a, 1, b),
                            ThOp::Asr => shift(a, 2, b),
                            _ => div0()?,
                        };
                        set(&mut regs, 0, v)?;
                    }
                    ThOp::AddImm | ThOp::AndImm | ThOp::OrrImm | ThOp::EorImm | ThOp::RsbImm | ThOp::LslImm
                    | ThOp::LsrImm | ThOp::AsrImm => {
                        let a = rd(&regs, &o[1])?;
                        let k = imm(&o[2])? as u32;
                        let v = match op {
                            ThOp::AddImm => a.wrapping_add(k),
                            ThOp::AndImm => a & k,
                            ThOp::OrrImm => a | k,
                            ThOp::EorImm => a ^ k,
                            ThOp::RsbImm => k.wrapping_sub(a),
                            ThOp::LslImm => shift(a, 0, k),
                            ThOp::LsrImm => shift(a, 1, k),
                            _ => shift(a, 2, k),
                        };
                        set(&mut regs, 0, v)?;
                    }
                    ThOp::Mvn => {
                        let v = !rd(&regs, &o[1])?;
                        set(&mut regs, 0, v)?;
                    }
                    ThOp::Ext => {
                        let s = rd(&regs, &o[1])?;
                        let (w, signed) = (imm(&o[2])? as u32, imm(&o[3])? != 0);
                        let v = if signed {
                            (((s << (32 - w)) as i32) >> (32 - w)) as u32
                        } else {
                            s & ((1u64 << w) - 1) as u32
                        };
                        set(&mut regs, 0, v)?;
                    }
                    ThOp::SetCmp | ThOp::SetCmpImm => {
                        let a = rd(&regs, &o[1])?;
                        let b = if op == ThOp::SetCmp { rd(&regs, &o[2])? } else { imm(&o[2])? as u32 };
                        let v = u32::from(cond(imm(&o[3])?, a, b));
                        set(&mut regs, 0, v)?;
                    }
                    ThOp::Select => {
                        let c = rd(&regs, &o[1])?;
                        let v = if c & 1 != 0 { rd(&regs, &o[2])? } else { rd(&regs, &o[3])? };
                        set(&mut regs, 0, v)?;
                    }
                    ThOp::Load => {
                        let a = rd(&regs, &o[1])?.wrapping_add(imm(&o[2])? as u32);
                        let v = self.mem.read(a, imm(&o[3])? as u32);
                        set(&mut regs, 0, v)?;
                    }
                    ThOp::Store => {
                        let a = rd(&regs, &o[0])?.wrapping_add(imm(&o[2])? as u32);
                        let v = rd(&regs, &o[1])?;
                        self.mem.write(a, imm(&o[3])? as u32, v);
                    }
                    ThOp::LoadDual => {
                        let a = rd(&regs, &o[2])?;
                        let (x, y) = (self.mem.read(a, 4), self.mem.read(a + 4, 4));
                        set(&mut regs, 0, x)?;
                        set(&mut regs, 1, y)?;
                    }
                    ThOp::StoreDual => {
                        let a = rd(&regs, &o[0])?;
                        let (x, y) = (rd(&regs, &o[1])?, rd(&regs, &o[2])?);
                        self.mem.write(a, 4, x);
                        self.mem.write(a + 4, 4, y);
                    }
                    ThOp::FrameAddr | ThOp::LoadFrame | ThOp::StoreFrame => {
                        let MachineOperand::Frame(s) = &o[1] else { return Err("expected a frame slot".into()) };
                        let a = slot_addr[s.index()];
                        match op {
                            ThOp::FrameAddr => set(&mut regs, 0, a)?,
                            ThOp::LoadFrame => {
                                let v = self.mem.read(a, 4);
                                set(&mut regs, 0, v)?;
                            }
                            _ => {
                                let v = rd(&regs, &o[0])?;
                                self.mem.write(a, 4, v);
                            }
                        }
                    }
                    ThOp::SpAddr => set(&mut regs, 0, out_base + imm(&o[1])? as u32)?,
                    ThOp::StoreSp => {
                        let v = rd(&regs, &o[0])?;
                        self.mem.write(out_base + imm(&o[1])? as u32, imm(&o[2])? as u32, v);
                    }
                    ThOp::IncAddr => set(&mut regs, 0, incoming + imm(&o[1])? as u32)?,
                    ThOp::LoadInc => {
                        let v = self.mem.read(incoming + imm(&o[1])? as u32, imm(&o[2])? as u32);
                        set(&mut regs, 0, v)?;
                    }
                    ThOp::GlobalAddr | ThOp::FuncAddr => {
                        let name = match &o[1] {
                            MachineOperand::Global(g) => &self.p.global_names[*g as usize],
                            MachineOperand::Func(f) => &self.p.func_names[*f as usize],
                            other => return Err(format!("expected a symbol, found {other:?}")),
                        };
                        let a = *self.p.symbols.get(name).ok_or_else(|| format!("no symbol {name}"))?;
                        set(&mut regs, 0, a)?;
                    }
                    ThOp::Call => {
                        let target = match &o[0] {
                            MachineOperand::Func(f) => *f as usize,
                            callee => {
                                let a = rd(&regs, callee)?;
                                let name = self
                                    .p
                                    .symbols
                                    .iter()
                                    .find(|&(_, &v)| v == a)
                                    .map(|(n, _)| n.clone())
                                    .ok_or_else(|| format!("indirect call to {a:#x}"))?;
                                self.p.func_names.iter().position(|n| *n == name).ok_or("callee not a function")?
                            }
                        };
                        let mut args = [0u32; 4];
                        for u in inst.uses() {
                            if let Reg::Physical(p) = u
                                && p.num < 4
                            {
                                args[p.num as usize] = regs[&u];
                            }
                        }
                        let out = self.call(target, args, out_base)?;
                        for (k, &v) in out.iter().enumerate() {
                            regs.insert(Reg::Physical(super::regs::gpr(k as u16)), v);
                        }
                        // ip and lr are clobbered.
                        for r in [super::regs::IP, super::regs::LR] {
                            regs.insert(Reg::Physical(super::regs::gpr(r)), 0xdead_beef);
                        }
                    }
                    ThOp::Ret => {
                        let mut out = [0u32; 4];
                        for u in inst.uses() {
                            if let Reg::Physical(p) = u
                                && p.num < 4
                            {
                                out[p.num as usize] = regs[&u];
                            }
                        }
                        self.sp = saved_sp;
                        return Ok(out);
                    }
                    ThOp::B => next = Some(label(&o[0])?),
                    ThOp::BrCond => {
                        let c = rd(&regs, &o[0])?;
                        next = Some(if c & 1 != 0 { label(&o[1])? } else { label(&o[2])? });
                    }
                    ThOp::Switch => {
                        let c = rd(&regs, &o[0])?;
                        let mut t = label(&o[1])?;
                        for pair in o[2..].chunks(2) {
                            if imm(&pair[0])? as u32 == c {
                                t = label(&pair[1])?;
                                break;
                            }
                        }
                        next = Some(t);
                    }
                    ThOp::Switch64 => {
                        let c = u64::from(rd(&regs, &o[0])?) | u64::from(rd(&regs, &o[1])?) << 32;
                        let mut t = label(&o[2])?;
                        for pair in o[3..].chunks(2) {
                            if imm(&pair[0])? == c {
                                t = label(&pair[1])?;
                                break;
                            }
                        }
                        next = Some(t);
                    }
                    ThOp::Udf => return Err("reached `unreachable`".into()),
                    ThOp::Dmb => {}
                    ThOp::Svc | ThOp::SubSp | ThOp::AddSp | ThOp::Push | ThOp::Pop => {
                        return Err(format!("{op:?} is not modeled"));
                    }
                }
                if next.is_some() {
                    break;
                }
            }
            block = next.ok_or_else(|| format!("block {block:?} fell through"))?;
        }
    }
}

fn label(op: &MachineOperand) -> Result<crate::codegen::mir::MBlockId, String> {
    match op {
        MachineOperand::Label(b) => Ok(*b),
        other => Err(format!("expected a label, found {other:?}")),
    }
}
