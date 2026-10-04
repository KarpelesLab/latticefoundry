//! A small interpreter for the RISC-V backend's MIR, analogous to
//! [`crate::codegen::interp`] but over the [`RvOp`] opcode set.
//!
//! Since this host cannot execute RISC-V machine code, this is how the lowering
//! (isel) is validated *semantically*: it runs a lowered [`MachineFunction`] on
//! concrete inputs — before register allocation, so it exercises isel in
//! isolation — and returns the values the function computes, letting a test
//! assert `interp(select(f))(x) == expected(f)(x)`. Register operands (virtual
//! or physical) are just keys; ABI argument/return registers and call clobbers
//! are ordinary physical registers. It models one `RvOp` per step (e.g.
//! `SetCmp` is evaluated as compare-then-set, `Select` as the ternary choice,
//! `Li` as a load-immediate) rather than the encoder's multi-word idiom
//! expansion. (The machine code itself is run by the simulator, [`super::sim`].)
//!
//! It uses one shared flat address space ([`Memory`], the simulator's, so a
//! test can hand it a linked image) so a pointer handed across a call (a
//! by-reference argument, an `alloca`'d slot) resolves in the callee exactly as
//! on hardware. Each activation gets its slots, its outgoing stack-argument
//! area (`LeaSp`) and its `dyn_alloca` blocks from a bump region; a callee's
//! incoming stack arguments (`LeaInArg`) are its caller's outgoing area. A
//! call's inputs are its `Use(physical)` argument-register operands and its
//! outputs the result registers `a0`/`a1`/`fa0`/`fa1`.
//!
//! ## Register width
//!
//! Every register is modeled as the 64-bit pattern the hardware holds, and each
//! op as the RV64 instruction the encoder emits: the encoder always uses the
//! full-width forms (`add`, `sra`, `div`, `slt`, ...), so the MIR `width`
//! immediate is ignored here. A narrow value therefore keeps whatever bits its
//! computation left above its width (an `i8` add of 200 + 100 holds 300), and
//! an isel that forgets to extend before an op that reads those bits is caught.
//! The only 32-bit forms, `sext.w` (`addiw rd, rs, 0`), `fmv.x.w` and the
//! 32-bit float-to-integer conversions, sign-extend their low word.
//!
//! A floating-point value is its IEEE bit pattern (an `f32` in the low 32 bits;
//! the NaN-boxing of the hardware register file is invisible at this level).
//! Arithmetic rounds to nearest-even (the host's) and produces the canonical
//! NaN as the hardware does; conversions saturate as `fcvt` does.

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, PReg, Reg, StackSlot};
use crate::codegen::target::MachineTarget;
use crate::support::DetHashMap;

use puremp::Int;

use super::isel::{RvOp, RiscvTarget};
use super::regs::{fpr, gpr};
use super::sim::Memory;

/// A cap on executed instructions, so a miscompiled loop fails fast.
const STEP_BUDGET: u64 = 5_000_000;
/// Where activation frames (slots, outgoing areas, dynamic blocks) are
/// bump-allocated: away from a linked image and the simulator's stack.
const FRAMES_BASE: u64 = 0x0000_0030_0000_0000;

/// Run function `entry` of `funcs` with integer `args` (in `a0`..), returning
/// its return value (the last one moved into `a0` or `fa0`). `Err` on any
/// modeled fault (division by zero, an unsupported opcode, an out-of-budget
/// loop, ...).
pub(super) fn run(
    target: &RiscvTarget,
    funcs: &[MachineFunction],
    entry: usize,
    args: &[Int],
) -> Result<Option<Int>, String> {
    run_with_syscalls(target, funcs, entry, args, None)
}

/// The interpreter's (optional) operating-system environment: called for each
/// executed syscall with the number (`a7`) and the argument registers it
/// reads, in ABI order, and returning the raw 64-bit kernel result. There is no
/// real kernel here, so without a hook a syscall is a clean "unsupported side
/// effect" error rather than an invented result.
pub(super) type SyscallHook<'h> = &'h mut dyn FnMut(&Int, &[Int]) -> Result<Int, String>;

/// [`run`] with a [`SyscallHook`] servicing the program's syscalls (`None`
/// makes any executed syscall an error, as in [`run`]).
pub(super) fn run_with_syscalls<'a>(
    target: &'a RiscvTarget,
    funcs: &'a [MachineFunction],
    entry: usize,
    args: &[Int],
    syscalls: Option<SyscallHook<'a>>,
) -> Result<Option<Int>, String> {
    let prog = Program { target, funcs, globals: &[], func_addrs: &[], names: &[] };
    let cc = target.call_conv();
    let inputs: Vec<(PReg, Int)> = cc.arg_regs.iter().copied().zip(args.iter().cloned()).collect();
    let mut m = Machine::new(&prog, Memory::default(), syscalls);
    Ok(m.call(entry, &inputs, FRAMES_BASE)?.ret_val)
}

/// A whole program: its lowered functions (indexed by `FuncId`) and, for
/// symbol addressing, the address of each global and function (from a linked
/// image; empty when the program takes no addresses).
pub(super) struct Program<'a> {
    pub(super) target: &'a RiscvTarget,
    pub(super) funcs: &'a [MachineFunction],
    pub(super) globals: &'a [u64],
    pub(super) func_addrs: &'a [u64],
    /// Function names (by index), for the C library functions the code may
    /// call (`fmod`/`fmodf`), which run natively. May be empty.
    pub(super) names: &'a [String],
}

/// What a completed call hands back: the last value moved into a result
/// register, and the result registers `a0`, `a1`, `fa0`, `fa1` (absent ones
/// read as zero).
#[derive(Debug)]
pub(super) struct CallOut {
    pub(super) ret_val: Option<Int>,
    pub(super) regs: Vec<(PReg, Int)>,
}

impl CallOut {
    /// The 64-bit pattern of result register `r`.
    pub(super) fn reg(&self, r: PReg) -> u64 {
        self.regs.iter().find(|(p, _)| *p == r).and_then(|(_, v)| v.to_u64()).unwrap_or(0)
    }
}

/// Run `entry` of `prog` over `mem` with the given register inputs and the
/// bytes of its incoming stack arguments.
pub(super) fn run_program(
    prog: &Program<'_>,
    mem: Memory,
    entry: usize,
    inputs: &[(PReg, Int)],
    stack_args: &[u8],
) -> Result<CallOut, String> {
    let mut m = Machine::new(prog, mem, None);
    let incoming = FRAMES_BASE;
    m.mem.write_bytes(incoming, stack_args);
    m.heap = FRAMES_BASE + align_up(stack_args.len() as u64, 16) + 16;
    m.call(entry, inputs, incoming)
}

/// The whole-program interpreter state: a shared flat address space plus the
/// function table and a global step budget.
struct Machine<'a, 'p> {
    prog: &'p Program<'a>,
    budget: u64,
    /// The single flat address space every activation's slots live in.
    mem: Memory,
    /// The bump cursor for the next allocation.
    heap: u64,
    /// The syscall environment, if any (see [`SyscallHook`]).
    syscalls: Option<SyscallHook<'a>>,
}

/// One function activation's mutable state.
struct Frame {
    regs: DetHashMap<Reg, Int>,
    /// Absolute base address of each stack slot in the shared [`Machine::mem`].
    slot_base: Vec<u64>,
    /// Spill/aux slots addressed by handle rather than memory address.
    slot_val: DetHashMap<StackSlot, Int>,
    /// The base of this activation's outgoing stack-argument area.
    outgoing: u64,
    /// The base of the incoming stack arguments (the caller's outgoing area).
    incoming: u64,
    /// The most recent value moved into a result register (`a0` / `fa0`).
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

/// Round `v` up to a multiple of `align` (≥ 1).
fn align_up(v: u64, align: u64) -> u64 {
    let a = align.max(1);
    v.div_ceil(a) * a
}

fn u(v: &Int) -> u64 {
    v.to_u64().unwrap_or(0)
}

fn sext64(v: u64, bits: u32) -> u64 {
    (((v << (64 - bits)) as i64) >> (64 - bits)) as u64
}

const CANON_F32: u64 = 0x7fc0_0000;
const CANON_F64: u64 = 0x7ff8_0000_0000_0000;

/// A float result's bits, NaNs canonical (as the hardware produces them).
fn fbits(w: u32, v: f64) -> u64 {
    if w == 32 {
        let f = v as f32;
        if f.is_nan() { CANON_F32 } else { u64::from(f.to_bits()) }
    } else if v.is_nan() {
        CANON_F64
    } else {
        v.to_bits()
    }
}

/// A float operand's value (exact in `f64` for either width).
fn fval(w: u32, bits: u64) -> f64 {
    if w == 32 { f64::from(f32::from_bits(bits as u32)) } else { f64::from_bits(bits) }
}

enum Flow {
    Next,
    Goto(crate::codegen::mir::MBlockId),
    Return,
}

impl<'a, 'p> Machine<'a, 'p> {
    fn new(prog: &'p Program<'a>, mem: Memory, syscalls: Option<SyscallHook<'a>>) -> Machine<'a, 'p> {
        Machine { prog, budget: STEP_BUDGET, mem, heap: FRAMES_BASE + 4096, syscalls }
    }

    /// Bump-allocate `size` bytes at `align`.
    fn alloc(&mut self, size: u64, align: u64) -> u64 {
        let at = align_up(self.heap, align.max(16));
        self.heap = at + size.max(1);
        at
    }

    /// Invoke function `fidx` with `inputs` pre-loaded into physical registers
    /// and its incoming stack arguments at `incoming`, running it to its `ret`.
    fn call(&mut self, fidx: usize, inputs: &[(PReg, Int)], incoming: u64) -> Result<CallOut, String> {
        let funcs = self.prog.funcs;
        let mf = funcs.get(fidx).ok_or_else(|| format!("no function #{fidx}"))?;
        let Some(entry) = mf.entry() else {
            return self.libc(fidx, inputs);
        };

        // This activation's slots and outgoing area, globally unique (so
        // cross-call pointers work).
        let frame = mf.frame();
        let mut slot_base = vec![0u64; frame.len()];
        for (i, b) in slot_base.iter_mut().enumerate() {
            let info = frame.slot(StackSlot::from_index(i));
            *b = self.alloc(info.size, info.align);
        }
        let outgoing = self.alloc(frame.outgoing(), 16);

        let mut fr = Frame {
            regs: DetHashMap::default(),
            slot_base,
            slot_val: DetHashMap::default(),
            outgoing,
            incoming,
            ret_val: None,
        };
        for (p, v) in inputs {
            fr.regs.insert(Reg::Physical(*p), v.clone());
        }

        let mut block = entry;
        let mut ip = 0usize;
        loop {
            self.budget = self.budget.checked_sub(1).ok_or("step budget exhausted")?;
            let insts = &funcs[fidx].block(block).insts;
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

        let outs = [gpr(10), gpr(11), fpr(10), fpr(11)];
        let regs = outs
            .iter()
            .filter_map(|&r| fr.regs.get(&Reg::Physical(r)).map(|v| (r, mask(v, 64))))
            .collect();
        Ok(CallOut { ret_val: fr.ret_val, regs })
    }

    /// A call to a body-less function: one of the C library functions the
    /// code may call, run natively.
    fn libc(&self, fidx: usize, inputs: &[(PReg, Int)]) -> Result<CallOut, String> {
        let name = self.prog.names.get(fidx).map(String::as_str).unwrap_or("");
        let arg = |r: PReg| inputs.iter().find(|(p, _)| *p == r).map_or(0, |(_, v)| u(v));
        let (x, y) = (arg(fpr(10)), arg(fpr(11)));
        let r = match name {
            "fmod" => (f64::from_bits(x) % f64::from_bits(y)).to_bits(),
            "fmodf" => u64::from((f32::from_bits(x as u32) % f32::from_bits(y as u32)).to_bits()),
            _ => return Err(format!("call into the body-less function #{fidx} ({name})")),
        };
        Ok(CallOut { ret_val: Some(Int::from_u64(r)), regs: vec![(fpr(10), Int::from_u64(r))] })
    }

    fn step(&mut self, fr: &mut Frame, inst: &MachineInst) -> Result<Flow, String> {
        let ops = &inst.operands;
        let op = RvOp::decode(inst.opcode);
        match op {
            RvOp::Li => {
                let d = def(ops, 0)?;
                fr.regs.insert(d, mask(imm(ops, 1)?, 64));
            }
            RvOp::SextW => {
                let d = def(ops, 0)?;
                let s = self.rd(fr, use_reg(ops, 1)?);
                fr.regs.insert(d, mask(&signed(&mask(&s, 32), 32), 64));
            }
            RvOp::Mv => {
                let d = def(ops, 0)?;
                let s = self.rd(fr, use_reg(ops, 1)?);
                let cc = self.prog.target.call_conv();
                if d == Reg::Physical(cc.ret_reg) || d == Reg::Physical(cc.fp_ret_reg) {
                    fr.ret_val = Some(s.clone());
                }
                fr.regs.insert(d, s);
            }
            RvOp::Add | RvOp::Sub | RvOp::And | RvOp::Or | RvOp::Xor | RvOp::Mul | RvOp::Mulh
            | RvOp::Sll | RvOp::Srl | RvOp::Sra => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let w = 64;
                let res = match op {
                    RvOp::Add => a.add(&bb),
                    RvOp::Sub => a.sub(&bb),
                    RvOp::And => a.bitand(&bb),
                    RvOp::Or => a.bitor(&bb),
                    RvOp::Xor => a.bitxor(&bb),
                    RvOp::Mul => a.mul(&bb),
                    RvOp::Mulh => {
                        // Signed high half of the 128-bit product.
                        let p = signed(&a, w).mul(&signed(&bb, w));
                        p.div_floor(&Int::ONE.mul_2k(w))
                    }
                    RvOp::Sll | RvOp::Srl | RvOp::Sra => {
                        // The count is the low 6 bits of `rs2`.
                        let k = (bb.to_u64().unwrap_or(0) % 64) as u32;
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
                let w = 64;
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
                let w = 64;
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
                fr.regs.insert(d, self.divrem(op, &a, &bb, 64)?);
            }
            RvOp::SetCmp => {
                let d = def(ops, 0)?;
                let a = self.rd(fr, use_reg(ops, 1)?);
                let bb = self.rd(fr, use_reg(ops, 2)?);
                let pred = imm(ops, 3)?.to_u64().unwrap_or(0) as u8;
                let r = eval_pred(pred, &a, &bb, 64);
                fr.regs.insert(d, if r { Int::ONE } else { Int::ZERO });
            }
            RvOp::Select => {
                let d = def(ops, 0)?;
                let c = self.rd(fr, use_reg(ops, 1)?);
                let t = self.rd(fr, use_reg(ops, 2)?);
                let f = self.rd(fr, use_reg(ops, 3)?);
                // The encoder's branchless blend: `mask = 0 - c`, then
                // `(t & mask) | (f & !mask)`. Only a condition of exactly 0 or 1
                // selects cleanly; anything else mixes the two operands' bits.
                let all = Int::ONE.mul_2k(64).sub(&Int::ONE);
                let m = mask(&Int::ZERO.sub(&c), 64);
                let res = t.bitand(&m).bitor(&f.bitand(&all.bitxor(&m)));
                fr.regs.insert(d, res);
            }
            RvOp::Load => {
                let d = def(ops, 0)?;
                let ptr = self.rd(fr, use_reg(ops, 1)?);
                let size = imm_u64(ops, 2)?;
                fr.regs.insert(d, Int::from_u64(self.mem.read(u(&ptr), size)));
            }
            RvOp::Store => {
                let ptr = self.rd(fr, use_reg(ops, 0)?);
                let val = self.rd(fr, use_reg(ops, 1)?);
                let size = imm_u64(ops, 2)?;
                self.mem.write(u(&ptr), size, u(&val));
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
            RvOp::LeaSp => {
                let d = def(ops, 0)?;
                fr.regs.insert(d, Int::from_u64(fr.outgoing + imm_u64(ops, 1)?));
            }
            RvOp::LeaInArg => {
                let d = def(ops, 0)?;
                fr.regs.insert(d, Int::from_u64(fr.incoming + imm_u64(ops, 1)?));
            }
            RvOp::DynAlloca => {
                let d = def(ops, 0)?;
                let n = u(&self.rd(fr, use_reg(ops, 1)?));
                let align = imm_u64(ops, 2)?;
                let at = self.alloc(n, align);
                fr.regs.insert(d, Int::from_u64(at));
            }
            RvOp::GlobalAddr | RvOp::FuncAddr => {
                let d = def(ops, 0)?;
                let a = match &ops[1] {
                    MachineOperand::Global(g) => self.prog.globals.get(*g as usize),
                    MachineOperand::Func(f) => self.prog.func_addrs.get(*f as usize),
                    _ => None,
                }
                .ok_or("symbol addressing is not modeled (no linked image)")?;
                fr.regs.insert(d, Int::from_u64(*a));
            }
            // --- the F and D extensions ----------------------------------------
            RvOp::FAdd | RvOp::FSub | RvOp::FMul | RvOp::FDiv => {
                let d = def(ops, 0)?;
                let w = imm_u64(ops, 3)? as u32;
                let a = fval(w, u(&self.rd(fr, use_reg(ops, 1)?)));
                let b = fval(w, u(&self.rd(fr, use_reg(ops, 2)?)));
                // Each operation is exact in f64 then rounded once to f32 for
                // a single: f64 carries more than 2·24 + 2 bits, so that
                // double rounding is innocuous (the result equals the single
                // rounding the hardware does).
                let r = match op {
                    RvOp::FAdd => a + b,
                    RvOp::FSub => a - b,
                    RvOp::FMul => a * b,
                    _ => a / b,
                };
                fr.regs.insert(d, Int::from_u64(fbits(w, r)));
            }
            RvOp::FMadd => {
                let d = def(ops, 0)?;
                let w = imm_u64(ops, 4)? as u32;
                let kind = imm_u64(ops, 5)?;
                let a = fval(w, u(&self.rd(fr, use_reg(ops, 1)?)));
                let b = fval(w, u(&self.rd(fr, use_reg(ops, 2)?)));
                let c = fval(w, u(&self.rd(fr, use_reg(ops, 3)?)));
                let a = if kind >= 2 { -a } else { a };
                let c = if kind % 2 == 1 { -c } else { c };
                let r = if w == 32 {
                    let x = (a as f32).mul_add(b as f32, c as f32);
                    if x.is_nan() { CANON_F32 } else { u64::from(x.to_bits()) }
                } else {
                    fbits(64, a.mul_add(b, c))
                };
                fr.regs.insert(d, Int::from_u64(r));
            }
            RvOp::FSgnj => {
                let d = def(ops, 0)?;
                let a = u(&self.rd(fr, use_reg(ops, 1)?));
                let b = u(&self.rd(fr, use_reg(ops, 2)?));
                let w = imm_u64(ops, 3)? as u32;
                let sign = 1u64 << (w - 1);
                let s = match imm_u64(ops, 4)? {
                    0 => b & sign,
                    1 => !b & sign,
                    _ => (a ^ b) & sign,
                };
                let lowmask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
                fr.regs.insert(d, Int::from_u64(((a & !sign) | s) & lowmask));
            }
            RvOp::FCmp => {
                let d = def(ops, 0)?;
                let w = imm_u64(ops, 4)? as u32;
                let x = fval(w, u(&self.rd(fr, use_reg(ops, 1)?)));
                let y = fval(w, u(&self.rd(fr, use_reg(ops, 2)?)));
                let code = imm_u64(ops, 3)?;
                let uno = x.is_nan() || y.is_nan();
                let r = match code & 7 {
                    0 => x == y,
                    1 => x < y,
                    2 => x <= y,
                    3 => x > y,
                    4 => x >= y,
                    5 => !uno,
                    _ => !uno && x != y,
                };
                fr.regs.insert(d, Int::from_u64(u64::from(r != (code & 8 != 0))));
            }
            RvOp::FLi => {
                let d = def(ops, 0)?;
                fr.regs.insert(d, mask(imm(ops, 1)?, 64));
            }
            RvOp::FCvtFF => {
                let d = def(ops, 0)?;
                let (dw, sw) = (imm_u64(ops, 2)? as u32, imm_u64(ops, 3)? as u32);
                let x = fval(sw, u(&self.rd(fr, use_reg(ops, 1)?)));
                fr.regs.insert(d, Int::from_u64(fbits(dw, x)));
            }
            RvOp::FCvtFI => {
                let d = def(ops, 0)?;
                let (sgn, iw, fw) = (imm_u64(ops, 2)? != 0, imm_u64(ops, 3)? as u32, imm_u64(ops, 4)? as u32);
                let x = fval(fw, u(&self.rd(fr, use_reg(ops, 1)?)));
                fr.regs.insert(d, Int::from_u64(fcvt_to_int(x, sgn, iw)));
            }
            RvOp::FCvtIF => {
                let d = def(ops, 0)?;
                let (sgn, iw, fw) = (imm_u64(ops, 2)? != 0, imm_u64(ops, 3)? as u32, imm_u64(ops, 4)? as u32);
                let a = u(&self.rd(fr, use_reg(ops, 1)?));
                let n: i128 = match (iw, sgn) {
                    (32, true) => i128::from(a as i32),
                    (32, false) => i128::from(a as u32),
                    (_, true) => i128::from(a as i64),
                    _ => i128::from(a),
                };
                // `as` rounds to nearest, ties to even, directly to the width.
                let r = if fw == 32 { u64::from((n as f32).to_bits()) } else { (n as f64).to_bits() };
                fr.regs.insert(d, Int::from_u64(r));
            }
            RvOp::FMvXF => {
                let d = def(ops, 0)?;
                let w = imm_u64(ops, 2)? as u32;
                let s = u(&self.rd(fr, use_reg(ops, 1)?));
                fr.regs.insert(d, Int::from_u64(if w == 32 { sext64(s & 0xffff_ffff, 32) } else { s }));
            }
            RvOp::FMvFX => {
                let d = def(ops, 0)?;
                let w = imm_u64(ops, 2)? as u32;
                let s = u(&self.rd(fr, use_reg(ops, 1)?));
                fr.regs.insert(d, Int::from_u64(if w == 32 { s & 0xffff_ffff } else { s }));
            }
            // --- atomics: the machine is single-threaded, so each op runs its
            // sequential meaning and a fence does nothing ----------------------
            RvOp::Fence => {}
            RvOp::AtomicRmw => {
                let d = def(ops, 0)?;
                let at = u(&self.rd(fr, use_reg(ops, 1)?));
                let val = self.rd(fr, use_reg(ops, 2)?);
                let size = imm_u64(ops, 3)?;
                let rmw = crate::ir::RmwOp::from_code(imm_u64(ops, 4)?)
                    .ok_or("atomic rmw: bad operation code")?;
                let w = (8 * size) as u32;
                let old = self.mem.read(at, size);
                let new = rmw.apply(old, u(&mask(&val, w)), w);
                self.mem.write(at, size, new);
                fr.regs.insert(d, Int::from_u64(old));
            }
            RvOp::CmpXchg => {
                let d = def(ops, 0)?;
                let at = u(&self.rd(fr, use_reg(ops, 1)?));
                let expected = self.rd(fr, use_reg(ops, 2)?);
                let new = self.rd(fr, use_reg(ops, 3)?);
                let size = imm_u64(ops, 4)?;
                let w = (8 * size) as u32;
                let old = self.mem.read(at, size);
                if old == u(&mask(&expected, w)) {
                    self.mem.write(at, size, u(&new));
                }
                fr.regs.insert(d, Int::from_u64(old));
            }
            RvOp::Call => return self.exec_call(fr, inst),
            RvOp::Ecall => {
                // Inputs: the `Use(physical)` operands, number first then the
                // arguments in ABI order; output: the result register (the def).
                let d = def(ops, 0)?;
                let uses: Vec<Int> = ops
                    .iter()
                    .filter_map(|o| match o {
                        MachineOperand::Use(r) => Some(self.rd(fr, *r)),
                        _ => None,
                    })
                    .collect();
                let hook = self
                    .syscalls
                    .as_mut()
                    .ok_or("unsupported side effect: syscall (no syscall hook installed)")?;
                let (nr, args) = uses.split_first().ok_or("syscall without a number operand")?;
                let r = hook(nr, args)?;
                fr.regs.insert(d, mask(&r, 64));
            }
            RvOp::Ret => return Ok(Flow::Return),
            RvOp::J => return Ok(Flow::Goto(label(ops, 0)?)),
            RvOp::BrCond => {
                let c = self.rd(fr, use_reg(ops, 0)?);
                let target = if c.is_zero() { label(ops, 2)? } else { label(ops, 1)? };
                return Ok(Flow::Goto(target));
            }
            RvOp::Switch => {
                // Each case value is materialized (`li t2`) as a 64-bit pattern
                // and compared with all 64 bits of the scrutinee (`beq`).
                let c = self.rd(fr, use_reg(ops, 0)?);
                let mut target = label(ops, 1)?;
                let mut i = 2;
                while i + 1 < ops.len() {
                    if let (MachineOperand::Imm(v), MachineOperand::Label(b)) = (&ops[i], &ops[i + 1])
                        && mask(v, 64) == c
                    {
                        target = *b;
                        break;
                    }
                    i += 2;
                }
                return Ok(Flow::Goto(target));
            }
            RvOp::Unreachable => return Err("reached an unreachable point (UB)".into()),
            // Prologue/epilogue pseudo-ops never appear in pre-regalloc MIR.
            RvOp::AddiSp | RvOp::SaveReg | RvOp::RestoreReg | RvOp::FpSetup | RvOp::FpRestore
            | RvOp::TouchSp => {}
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
        let fidx = match &inst.operands[0] {
            MachineOperand::Func(f) => *f as usize,
            MachineOperand::Use(r) => {
                let a = u(&self.rd(fr, *r));
                self.prog
                    .func_addrs
                    .iter()
                    .position(|&x| x == a)
                    .ok_or_else(|| format!("indirect call to an unknown address {a:#x}"))?
            }
            other => return Err(format!("a call to {other:?}")),
        };
        // The call's inputs are exactly its `Use(physical)` operands: the
        // argument registers.
        let inputs: Vec<(PReg, Int)> = inst
            .operands
            .iter()
            .skip(1)
            .filter_map(|o| match o {
                MachineOperand::Use(Reg::Physical(p)) => Some((*p, self.rd(fr, Reg::Physical(*p)))),
                _ => None,
            })
            .collect();
        let out = self.call(fidx, &inputs, fr.outgoing)?;
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
        // Inputs handed in as negative `Int`s are normalized to their 64-bit
        // two's-complement pattern.
        fr.regs.get(&r).map(|v| mask(v, 64)).unwrap_or(Int::ZERO)
    }
}

/// `fcvt.{w,wu,l,lu}` with `rtz`: truncate, saturating to the destination's
/// range (a NaN converts to the maximum); a 32-bit result is sign-extended.
pub(super) fn fcvt_to_int(x: f64, signed: bool, iw: u32) -> u64 {
    let (lo, hi): (i128, i128) = match (iw, signed) {
        (32, true) => (-(1 << 31), (1 << 31) - 1),
        (32, false) => (0, (1 << 32) - 1),
        (_, true) => (-(1 << 63), (1 << 63) - 1),
        _ => (0, (1 << 64) - 1),
    };
    let v = if x.is_nan() {
        hi
    } else {
        let t = x.trunc();
        if t <= lo as f64 {
            lo
        } else if t >= hi as f64 {
            hi
        } else {
            t as i128
        }
    };
    if iw == 32 { sext64(v as u64 & 0xffff_ffff, 32) } else { v as u64 }
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

fn imm_u64(ops: &[MachineOperand], i: usize) -> Result<u64, String> {
    imm(ops, i)?.to_u64().ok_or_else(|| "immediate does not fit u64".into())
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
