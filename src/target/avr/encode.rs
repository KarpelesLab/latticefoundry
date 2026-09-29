//! The AVR machine-code encoder: instruction words from the AVR Instruction
//! Set Manual, frame layout and prologue/epilogue, the expansion of each
//! [`AvrOp`] into instruction words, and branch relaxation.
//!
//! # Frame
//!
//! The stack grows down and `SP` points at the next free byte (`push` stores
//! then decrements). After the prologue:
//!
//! ```text
//!   Y + N + P + 3 + k   incoming stack argument byte k
//!   Y + N + P + 1..2    return address (2 bytes: PC ≤ 128 KiB devices)
//!   Y + N + 1 ..        P bytes of pushed call-saved registers (then r28, r29)
//!   Y + 1 .. Y + N      N bytes of locals: alloca and spill slots
//!   Y = SP
//! ```
//!
//! `Y` (`r29:r28`) is the frame pointer, set up only when the function has
//! locals, incoming stack arguments or a `dyn_alloca`; a slot is `ldd`/`std
//! Y+q` (`q ≤ 63`) or, farther away, addressed through `Z`. The stack pointer
//! is the I/O register pair `SPH:SPL`; it is written high byte first with
//! interrupts masked and `SREG` restored between the two writes (the
//! instruction after an `out` to `SREG` completes before a pending interrupt
//! is taken):
//!
//! ```text
//! in r0, SREG ; cli ; out SPH, r29 ; out SREG, r0 ; out SPL, r28
//! ```
//!
//! Outgoing stack arguments are pushed right before a call and popped after
//! it, so they are not part of the frame layout; the stack-usage report counts
//! the largest such push area in [`StackUsage::outgoing_args`].
//!
//! **Stack probes do not apply.** An AVR has no MMU and no guard page: the
//! stack simply runs into `.bss`/`.data` below it. The
//! [`CodegenOptions::stack_probes`] option is ignored; bound the stack with
//! [`StackReport::worst_case_depth`](crate::codegen::StackReport::worst_case_depth)
//! instead (every AVR function's [`StackUsage::probed`] is `false`).
//!
//! # Branches
//!
//! Intra-function branches are resolved here, with **relaxation**: a
//! conditional branch starts as `brXX` (±64 words); if its target is out of
//! reach it becomes the inverted branch over an `rjmp` (±2 K words), and
//! beyond that over a `jmp` (absolute, relocated with `R_AVR_CALL` against the
//! function's own symbol). An unconditional jump is nothing when it targets the
//! next block, else `rjmp`, else `jmp`. Sizes only grow, so the iteration
//! reaches a fixed point.

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, Reg, StackSlot};
use crate::codegen::options::{CodegenOptions, CompiledModule};
use crate::codegen::regalloc;
use crate::codegen::stack::{StackReport, StackUsage, scan_calls};
use crate::ir::Module;
use crate::ir::inst::RmwOp;
use crate::mc::emit::{Emitted, EmittedReloc};
use crate::mc::object::{ObjectModule, RelocKind, Section, SectionKind, Symbol, SymbolBinding, SymbolType};
use crate::support::StrInterner;

use super::isel::{AvrOp, AvrTarget};
use super::regs::{TMP, Y, Z, ZERO};

// ===========================================================================
// Instruction words (AVR Instruction Set Manual)
// ===========================================================================

/// The I/O address of `SPL`.
pub(crate) const SPL: u8 = 0x3d;
/// The I/O address of `SPH`.
pub(crate) const SPH: u8 = 0x3e;
/// The I/O address of `SREG`.
pub(crate) const SREG: u8 = 0x3f;

/// `SREG` bit numbers for `brbs`/`brbc`.
pub(crate) const FLAG_C: u8 = 0;
pub(crate) const FLAG_Z: u8 = 1;
pub(crate) const FLAG_N: u8 = 2;
pub(crate) const FLAG_S: u8 = 4;

/// A two-register instruction builder.
type Enc2 = fn(u8, u8) -> u16;

/// A two-register ALU op `oooo oord dddd rrrr` (`op6` = the top six bits).
#[inline]
fn rr(op6: u16, d: u8, r: u8) -> u16 {
    let (d, r) = (u16::from(d), u16::from(r));
    (op6 << 10) | ((r & 0x10) << 5) | (d << 4) | (r & 0xf)
}
pub(crate) fn add(d: u8, r: u8) -> u16 {
    rr(0b000011, d, r)
}
pub(crate) fn adc(d: u8, r: u8) -> u16 {
    rr(0b000111, d, r)
}
pub(crate) fn sub(d: u8, r: u8) -> u16 {
    rr(0b000110, d, r)
}
pub(crate) fn sbc(d: u8, r: u8) -> u16 {
    rr(0b000010, d, r)
}
pub(crate) fn and(d: u8, r: u8) -> u16 {
    rr(0b001000, d, r)
}
pub(crate) fn or(d: u8, r: u8) -> u16 {
    rr(0b001010, d, r)
}
pub(crate) fn eor(d: u8, r: u8) -> u16 {
    rr(0b001001, d, r)
}
pub(crate) fn mov(d: u8, r: u8) -> u16 {
    rr(0b001011, d, r)
}
pub(crate) fn cp(d: u8, r: u8) -> u16 {
    rr(0b000101, d, r)
}
pub(crate) fn cpc(d: u8, r: u8) -> u16 {
    rr(0b000001, d, r)
}
pub(crate) fn mul(d: u8, r: u8) -> u16 {
    rr(0b100111, d, r)
}
/// `lsl d` = `add d, d`.
pub(crate) fn lsl(d: u8) -> u16 {
    add(d, d)
}
/// `rol d` = `adc d, d`.
pub(crate) fn rol(d: u8) -> u16 {
    adc(d, d)
}
/// `tst d` = `and d, d`.
pub(crate) fn tst(d: u8) -> u16 {
    and(d, d)
}

/// An immediate op on `r16`–`r31`: `oooo KKKK dddd KKKK`.
#[inline]
fn ri(op4: u16, d: u8, k: u8) -> u16 {
    debug_assert!(d >= 16, "immediate op on r{d}");
    let k = u16::from(k);
    (op4 << 12) | ((k & 0xf0) << 4) | (u16::from(d - 16) << 4) | (k & 0xf)
}
pub(crate) fn ldi(d: u8, k: u8) -> u16 {
    ri(0xe, d, k)
}
pub(crate) fn cpi(d: u8, k: u8) -> u16 {
    ri(0x3, d, k)
}
pub(crate) fn subi(d: u8, k: u8) -> u16 {
    ri(0x5, d, k)
}
pub(crate) fn sbci(d: u8, k: u8) -> u16 {
    ri(0x4, d, k)
}
pub(crate) fn andi(d: u8, k: u8) -> u16 {
    ri(0x7, d, k)
}
pub(crate) fn ori(d: u8, k: u8) -> u16 {
    ri(0x6, d, k)
}

/// A one-register op `1001 010d dddd oooo`.
#[inline]
fn one(d: u8, op: u16) -> u16 {
    0x9400 | (u16::from(d) << 4) | op
}
pub(crate) fn com(d: u8) -> u16 {
    one(d, 0x0)
}
pub(crate) fn dec(d: u8) -> u16 {
    one(d, 0xa)
}
pub(crate) fn neg(d: u8) -> u16 {
    one(d, 0x1)
}
pub(crate) fn swap(d: u8) -> u16 {
    one(d, 0x2)
}
pub(crate) fn asr(d: u8) -> u16 {
    one(d, 0x5)
}
pub(crate) fn lsr(d: u8) -> u16 {
    one(d, 0x6)
}
pub(crate) fn ror(d: u8) -> u16 {
    one(d, 0x7)
}

/// `movw d, r` (both even).
pub(crate) fn movw(d: u8, r: u8) -> u16 {
    0x0100 | (u16::from(d / 2) << 4) | u16::from(r / 2)
}
/// `adiw d, k` (`d` ∈ {24, 26, 28, 30}, `k` ≤ 63).
pub(crate) fn adiw(d: u8, k: u8) -> u16 {
    let k = u16::from(k);
    0x9600 | ((k & 0x30) << 2) | (u16::from((d - 24) / 2) << 4) | (k & 0xf)
}
/// `sbiw d, k`.
pub(crate) fn sbiw(d: u8, k: u8) -> u16 {
    adiw(d, k) | 0x0100
}

/// `ldd d, Y+q` / `ldd d, Z+q` (`q` ≤ 63); `ld d, Z` is `ldd d, Z+0`.
pub(crate) fn ldd(d: u8, y: bool, q: u8) -> u16 {
    let q = u16::from(q);
    0x8000 | ((q & 0x20) << 8) | ((q & 0x18) << 7) | (u16::from(d) << 4) | (u16::from(y) << 3) | (q & 7)
}
/// `std Y+q, r` / `std Z+q, r`.
pub(crate) fn std(y: bool, q: u8, r: u8) -> u16 {
    ldd(r, y, q) | 0x0200
}
/// `st X+, r`.
pub(crate) fn st_x_inc(r: u8) -> u16 {
    0x920d | (u16::from(r) << 4)
}
/// `lpm d, Z`.
pub(crate) fn lpm(d: u8) -> u16 {
    0x9004 | (u16::from(d) << 4)
}
/// `lpm d, Z+`.
pub(crate) fn lpm_inc(d: u8) -> u16 {
    0x9005 | (u16::from(d) << 4)
}
/// `push r`.
pub(crate) fn push(r: u8) -> u16 {
    0x920f | (u16::from(r) << 4)
}
/// `pop d`.
pub(crate) fn pop(d: u8) -> u16 {
    0x900f | (u16::from(d) << 4)
}
/// `in d, A` (I/O address `A` ≤ 63).
pub(crate) fn in_(d: u8, a: u8) -> u16 {
    let a = u16::from(a);
    0xb000 | ((a & 0x30) << 5) | (u16::from(d) << 4) | (a & 0xf)
}
/// `out A, r`.
pub(crate) fn out(a: u8, r: u8) -> u16 {
    in_(r, a) | 0x0800
}
/// `brbs s, k` (`k` in words, −64..=63).
pub(crate) fn brbs(s: u8, k: i32) -> u16 {
    0xf000 | (((k as u16) & 0x7f) << 3) | u16::from(s)
}
/// `brbc s, k`.
pub(crate) fn brbc(s: u8, k: i32) -> u16 {
    brbs(s, k) | 0x0400
}
/// `brbs`/`brbc` by `set`.
pub(crate) fn br(s: u8, set: bool, k: i32) -> u16 {
    if set { brbs(s, k) } else { brbc(s, k) }
}
/// `rjmp k` (−2048..=2047 words).
pub(crate) fn rjmp(k: i32) -> u16 {
    0xc000 | ((k as u16) & 0xfff)
}
/// `jmp k` (a 22-bit word address): two words.
pub(crate) fn jmp(k: u32) -> [u16; 2] {
    [0x940c | (((k >> 17) & 0x1f) as u16) << 4 | ((k >> 16) & 1) as u16, k as u16]
}
/// `call k`: two words.
pub(crate) fn call(k: u32) -> [u16; 2] {
    let [a, b] = jmp(k);
    [a | 0x0002, b]
}
/// `sbrc r, b` (skip if bit `b` of `r` is clear).
pub(crate) fn sbrc(r: u8, b: u8) -> u16 {
    0xfc00 | (u16::from(r) << 4) | u16::from(b)
}
/// `bst r, b` (the T flag = bit `b` of `r`).
pub(crate) fn bst(r: u8, b: u8) -> u16 {
    0xfa00 | (u16::from(r) << 4) | u16::from(b)
}
/// `bld d, b` (bit `b` of `d` = the T flag).
pub(crate) fn bld(d: u8, b: u8) -> u16 {
    0xf800 | (u16::from(d) << 4) | u16::from(b)
}
pub(crate) const ICALL: u16 = 0x9509;
pub(crate) const RET: u16 = 0x9508;
pub(crate) const CLI: u16 = 0x94f8;
pub(crate) const BREAK: u16 = 0x9598;

/// The `(swap operands, SREG flag, branch-if-set)` of a predicate code (see
/// [`super::isel::pred_code`]) after `cp a, b`.
pub(crate) fn pred_branch(code: u64) -> (bool, u8, bool) {
    match code {
        0 => (false, FLAG_Z, true),  // eq: breq
        1 => (false, FLAG_Z, false), // ne: brne
        2 => (false, FLAG_C, true),  // ult: brlo
        3 => (true, FLAG_C, false),  // ule: b >= a, brsh
        4 => (true, FLAG_C, true),   // ugt: b < a, brlo
        5 => (false, FLAG_C, false), // uge: brsh
        6 => (false, FLAG_S, true),  // slt: brlt
        7 => (true, FLAG_S, false),  // sle: b >= a, brge
        8 => (true, FLAG_S, true),   // sgt: b < a, brlt
        _ => (false, FLAG_S, false), // sge: brge
    }
}

// ===========================================================================
// Frame layout + prologue/epilogue
// ===========================================================================

/// The frame of one allocated function.
#[derive(Clone, Debug)]
pub struct FrameLayout {
    /// Offset of each slot from `Y + 1`.
    slot_off: Vec<u64>,
    /// `N`: the bytes of locals below the saved registers.
    locals: u64,
    /// The call-saved byte registers pushed, in push order (not `Y`).
    saved: Vec<u8>,
    /// Whether `Y` is the frame pointer (pushed, then set to `SP`).
    fp: bool,
    /// Whether the function has a `dyn_alloca`.
    dynamic: bool,
    /// The largest stack-argument area pushed for a call.
    max_push: u64,
}

impl FrameLayout {
    /// `P`: bytes pushed by the prologue.
    pub fn pushed(&self) -> u64 {
        self.saved.len() as u64 + if self.fp { 2 } else { 0 }
    }

    /// The stack usage this layout gives `mf` (`func_name` resolves function
    /// indices to symbol names).
    pub fn stack_usage(&self, mf: &MachineFunction, func_name: &dyn Fn(u32) -> String) -> StackUsage {
        let scan = scan_calls(mf, AvrOp::Call.opcode(), crate::codegen::mir::Opcode(u32::MAX), Some(AvrOp::DynAlloca.opcode()));
        StackUsage {
            name: func_name(mf.info().source),
            frame_size: 2 + self.pushed() + self.locals + self.max_push,
            return_address: 2,
            saved_registers: self.pushed(),
            sp_adjust: self.locals,
            outgoing_args: self.max_push,
            dynamic_alloca: scan.dynamic_alloca,
            direct_callees: scan.direct.iter().map(|&f| func_name(f)).collect(),
            indirect_calls: scan.indirect,
            syscalls: false,
            probed: false,
        }
    }
}

fn imm_of(op: &MachineOperand) -> u64 {
    match op {
        MachineOperand::Imm(v) => v.to_u64().or_else(|| v.to_i64().map(|x| x as u64)).unwrap_or(0),
        other => panic!("expected an immediate operand, found {other:?}"),
    }
}

fn reg_of(op: &MachineOperand) -> u8 {
    match op {
        MachineOperand::Def(Reg::Physical(p)) | MachineOperand::Use(Reg::Physical(p)) => p.num as u8,
        other => panic!("expected a physical register operand, found {other:?}"),
    }
}

fn slot_of(op: &MachineOperand) -> StackSlot {
    match op {
        MachineOperand::Frame(s) => *s,
        other => panic!("expected a frame operand, found {other:?}"),
    }
}

fn label_of(op: &MachineOperand) -> usize {
    match op {
        MachineOperand::Label(b) => b.index(),
        other => panic!("expected a label operand, found {other:?}"),
    }
}

/// Lay out the frame of an allocated function. Slots with index below
/// `isel_slots` were created by isel (allocas: their own size); the rest are
/// the allocator's spill slots, which hold one register pair.
pub fn layout_frame(mf: &MachineFunction, isel_slots: usize) -> FrameLayout {
    let mut used = [false; 32];
    let mut dynamic = false;
    let mut stack_args = false;
    let mut max_push = 0u64;
    for bid in mf.block_ids() {
        for i in &mf.block(bid).insts {
            match AvrOp::decode(i.opcode) {
                AvrOp::DynAlloca => dynamic = true,
                AvrOp::LoadArg => stack_args = true,
                AvrOp::PopArgs => max_push = max_push.max(imm_of(&i.operands[0])),
                _ => {}
            }
            for d in i.defs() {
                if let Reg::Physical(p) = d {
                    used[p.num as usize] = true;
                }
            }
        }
    }
    let mut saved = Vec::new();
    for n in (2u8..=16).step_by(2) {
        if used[n as usize] {
            saved.push(n);
            saved.push(n + 1);
        }
    }
    let mut slot_off = Vec::with_capacity(mf.frame().len());
    let mut off = 0u64;
    for i in 0..mf.frame().len() {
        slot_off.push(off);
        let size = if i < isel_slots { mf.frame().slot(StackSlot::from_index(i)).size } else { 2 };
        off += size;
    }
    let fp = off > 0 || dynamic || stack_args;
    FrameLayout { slot_off, locals: off, saved, fp, dynamic, max_push }
}

fn imm_op(v: u64) -> MachineOperand {
    MachineOperand::Imm(puremp::Int::from_u64(v))
}

/// Splice the prologue into the entry block and an epilogue before every `ret`.
pub fn insert_prologue_epilogue(mf: &mut MachineFunction, layout: &FrameLayout) {
    let entry = mf.entry().expect("a compiled function has an entry block");
    let mut pro: Vec<MachineInst> = Vec::new();
    for &r in &layout.saved {
        pro.push(MachineInst::new(AvrOp::Push.opcode(), vec![imm_op(u64::from(r))]));
    }
    if layout.fp {
        pro.push(MachineInst::new(AvrOp::Push.opcode(), vec![imm_op(u64::from(Y))]));
        pro.push(MachineInst::new(AvrOp::Push.opcode(), vec![imm_op(u64::from(Y + 1))]));
        pro.push(MachineInst::new(AvrOp::FrameEnter.opcode(), vec![imm_op(layout.locals)]));
    }
    let old = std::mem::take(&mut mf.block_mut(entry).insts);
    pro.extend(old);
    mf.block_mut(entry).insts = pro;

    let ids: Vec<_> = mf.block_ids().collect();
    for bid in ids {
        let old = std::mem::take(&mut mf.block_mut(bid).insts);
        let mut out = Vec::with_capacity(old.len());
        for i in old {
            if AvrOp::decode(i.opcode) == AvrOp::Ret {
                if layout.fp {
                    if layout.locals > 0 || layout.dynamic {
                        out.push(MachineInst::new(AvrOp::FrameLeave.opcode(), vec![imm_op(layout.locals)]));
                    }
                    out.push(MachineInst::new(AvrOp::Pop.opcode(), vec![imm_op(u64::from(Y + 1))]));
                    out.push(MachineInst::new(AvrOp::Pop.opcode(), vec![imm_op(u64::from(Y))]));
                }
                for &r in layout.saved.iter().rev() {
                    out.push(MachineInst::new(AvrOp::Pop.opcode(), vec![imm_op(u64::from(r))]));
                }
            }
            out.push(i);
        }
        mf.block_mut(bid).insts = out;
    }
}

// ===========================================================================
// Expansion with relaxable branches
// ===========================================================================

/// One element of a function's code before branch resolution.
#[derive(Clone, Debug)]
enum Item {
    /// A fixed instruction word.
    W(u16),
    /// An instruction word patched by a relocation against a symbol.
    R(u16, RelocKind, String, i64),
    /// `call symbol` (two words, `R_AVR_CALL`).
    Call(String),
    /// A conditional branch on `SREG` bit `flag` being `set`, to a block.
    Br { flag: u8, set: bool, target: usize },
    /// An unconditional jump to a block.
    J(usize),
    /// The start of a block.
    Label(usize),
}

/// What the expansion needs to know besides the instruction.
struct Ctx<'a> {
    layout: &'a FrameLayout,
    func_name: &'a dyn Fn(u32) -> String,
    global_name: &'a dyn Fn(u32) -> String,
}

/// The code of one function under construction.
struct Code<'a> {
    items: Vec<Item>,
    ctx: Ctx<'a>,
}

impl Code<'_> {
    fn w(&mut self, w: u16) {
        self.items.push(Item::W(w));
    }

    fn ws(&mut self, ws: &[u16]) {
        for &w in ws {
            self.w(w);
        }
    }

    /// `Z += k`.
    fn z_add(&mut self, k: u64) {
        if k == 0 {
        } else if k <= 63 {
            self.w(adiw(Z, k as u8));
        } else {
            let n = (k as u16).wrapping_neg();
            self.w(subi(Z, n as u8));
            self.w(sbci(Z + 1, (n >> 8) as u8));
        }
    }

    /// `SP = r+1:r` with interrupts masked (see the module docs).
    fn set_sp(&mut self, r: u8) {
        self.ws(&[in_(TMP, SREG), CLI, out(SPH, r + 1), out(SREG, TMP), out(SPL, r)]);
    }

    /// Load (`load`) or store `size` bytes of the pair `reg` at `Y + q`.
    fn y_access(&mut self, load: bool, reg: u8, q: u64, size: u64) {
        if q + size - 1 <= 63 {
            for k in 0..size {
                let qq = (q + k) as u8;
                self.w(if load { ldd(reg + k as u8, true, qq) } else { std(true, qq, reg + k as u8) });
            }
        } else {
            self.w(movw(Z, Y));
            self.z_add(q);
            for k in 0..size {
                self.w(if load { ldd(reg + k as u8, false, k as u8) } else { std(false, k as u8, reg + k as u8) });
            }
        }
    }

    /// `cp`/`cpc` of two containers.
    fn compare(&mut self, a: u8, b: u8, swap: bool, cw: u64) {
        let (x, y) = if swap { (b, a) } else { (a, b) };
        self.w(cp(x, y));
        if cw == 16 {
            self.w(cpc(x + 1, y + 1));
        }
    }

    /// A two-source ALU op `d = a op b` over a container.
    fn alu(&mut self, op: AvrOp, d: u8, a: u8, b: u8, cw: u64) {
        let (lo, hi): (Enc2, Enc2) = match op {
            AvrOp::Add => (add, adc),
            AvrOp::Sub => (sub, sbc),
            AvrOp::And => (and, and),
            AvrOp::Or => (or, or),
            _ => (eor, eor),
        };
        let commutative = op != AvrOp::Sub;
        let (dst, src) = if d == a {
            (d, b)
        } else if d == b && commutative {
            (d, a)
        } else if d == b {
            // `d` aliases the subtrahend: compute in Z.
            self.w(movw(Z, a));
            self.w(lo(Z, b));
            if cw == 16 {
                self.w(hi(Z + 1, b + 1));
            }
            self.w(movw(d, Z));
            return;
        } else {
            self.w(movw(d, a));
            (d, b)
        };
        self.w(lo(dst, src));
        if cw == 16 {
            self.w(hi(dst + 1, src + 1));
        }
    }

    fn expand(&mut self, i: &MachineInst) {
        let ops = &i.operands;
        let r = |k: usize| reg_of(&ops[k]);
        let n = |k: usize| imm_of(&ops[k]);
        match AvrOp::decode(i.opcode) {
            AvrOp::Mov => {
                if r(0) != r(1) {
                    self.w(movw(r(0), r(1)));
                }
            }
            AvrOp::Li => {
                let (d, v) = (r(0), n(1) as u16);
                let (lo, hi) = (v as u8, (v >> 8) as u8);
                if d >= 16 {
                    self.ws(&[ldi(d, lo), ldi(d + 1, hi)]);
                } else if v == 0 {
                    self.ws(&[mov(d, ZERO), mov(d + 1, ZERO)]);
                } else {
                    self.ws(&[ldi(Z, lo), ldi(Z + 1, hi), movw(d, Z)]);
                }
            }
            op @ (AvrOp::Add | AvrOp::Sub | AvrOp::And | AvrOp::Or | AvrOp::Xor) => {
                self.alu(op, r(0), r(1), r(2), n(3));
            }
            AvrOp::Mul => {
                let (d, a, b) = (r(0), r(1), r(2));
                if n(3) == 8 {
                    self.ws(&[mul(a, b), mov(d, TMP), eor(ZERO, ZERO)]);
                } else {
                    self.ws(&[
                        mul(a, b),
                        movw(Z, TMP),
                        mul(a, b + 1),
                        add(Z + 1, TMP),
                        mul(a + 1, b),
                        add(Z + 1, TMP),
                        eor(ZERO, ZERO),
                        movw(d, Z),
                    ]);
                }
            }
            op @ (AvrOp::ShlC | AvrOp::LshrC | AvrOp::AshrC) => self.shift_const(op, r(0), r(1), n(2), n(3)),
            op @ (AvrOp::ShlV | AvrOp::LshrV | AvrOp::AshrV) => {
                // A secret operand: a branch-free barrel shifter (constant time
                // in the count):
                // stage k shifts a copy by 2^k and keeps it when bit k of the
                // count is set, blending with a mask made from that bit.
                // Scratch: Z (the value), r0 (the count), r1 (the mask,
                // cleared again at the end) and the fixed pair `t` the isel
                // reserved.
                let (d, a, cnt, cw) = (r(0), r(1), r(2), n(3));
                if n(4) == 0 {
                    // Public operands: a compact counted loop (it branches on
                    // the count).
                    self.w(if cw == 16 { movw(Z, a) } else { mov(Z, a) });
                    self.w(mov(TMP, cnt));
                    let body: Vec<u16> = match (op, cw) {
                        (AvrOp::ShlV, 16) => vec![lsl(Z), rol(Z + 1)],
                        (AvrOp::LshrV, 16) => vec![lsr(Z + 1), ror(Z)],
                        (AvrOp::AshrV, 16) => vec![asr(Z + 1), ror(Z)],
                        (AvrOp::ShlV, _) => vec![lsl(Z)],
                        (AvrOp::LshrV, _) => vec![lsr(Z)],
                        _ => vec![asr(Z)],
                    };
                    let bl = body.len() as i32;
                    self.w(rjmp(bl));
                    self.ws(&body);
                    self.w(dec(TMP));
                    self.w(brbc(FLAG_N, -(bl + 2))); // brpl
                    self.w(if cw == 16 { movw(d, Z) } else { mov(d, Z) });
                    return;
                }
                let t = r(5);
                let c_op = match op {
                    AvrOp::ShlV => AvrOp::ShlC,
                    AvrOp::LshrV => AvrOp::LshrC,
                    _ => AvrOp::AshrC,
                };
                self.w(if cw == 16 { movw(Z, a) } else { mov(Z, a) });
                self.w(mov(TMP, cnt));
                let stages = if cw == 16 { 4 } else { 3 };
                for k in 0..stages {
                    self.ws(&[eor(ZERO, ZERO), movw(t, Z)]);
                    self.shift_const(c_op, t, t, 1 << k, cw);
                    self.ws(&[bst(TMP, k as u8), bld(ZERO, 0), neg(ZERO)]);
                    self.ws(&[eor(t, Z), and(t, ZERO), eor(Z, t)]);
                    if cw == 16 {
                        self.ws(&[eor(t + 1, Z + 1), and(t + 1, ZERO), eor(Z + 1, t + 1)]);
                    }
                }
                self.w(eor(ZERO, ZERO));
                self.w(if cw == 16 { movw(d, Z) } else { mov(d, Z) });
            }
            AvrOp::Ext => {
                let (d, s, from, signed, cw, ct) = (r(0), r(1), n(2) as u8, n(3) != 0, n(4), n(5) != 0);
                self.w(mov(Z, s));
                if from > 8 {
                    self.w(mov(Z + 1, s + 1));
                }
                let mask = |bits: u8| -> u8 { ((1u16 << bits) - 1) as u8 };
                if !signed {
                    if from < 8 {
                        self.w(andi(Z, mask(from)));
                    }
                    if from > 8 && from < 16 {
                        self.w(andi(Z + 1, mask(from - 8)));
                    }
                    if cw == 16 && from <= 8 {
                        self.w(ldi(Z + 1, 0));
                    }
                } else if !ct {
                    // Public: mask, then fill the sign with a skip.
                    if from < 8 {
                        let m = mask(from);
                        self.ws(&[andi(Z, m), sbrc(Z, from - 1), ori(Z, !m)]);
                    }
                    if cw == 16 && from <= 8 {
                        self.ws(&[mov(Z + 1, Z), lsl(Z + 1), sbc(Z + 1, Z + 1)]);
                    }
                    if from > 8 && from < 16 {
                        let m = mask(from - 8);
                        self.ws(&[andi(Z + 1, m), sbrc(Z + 1, from - 9), ori(Z + 1, !m)]);
                    }
                } else {
                    // Secret: branch-free — move the sign bit to the top of
                    // its byte, then shift it back arithmetically.
                    if from < 8 {
                        for _ in from..8 {
                            self.w(lsl(Z));
                        }
                        for _ in from..8 {
                            self.w(asr(Z));
                        }
                    }
                    if cw == 16 && from <= 8 {
                        self.ws(&[mov(Z + 1, Z), lsl(Z + 1), sbc(Z + 1, Z + 1)]);
                    }
                    if from > 8 && from < 16 {
                        for _ in from..16 {
                            self.w(lsl(Z + 1));
                        }
                        for _ in from..16 {
                            self.w(asr(Z + 1));
                        }
                    }
                }
                self.w(if cw == 16 { movw(d, Z) } else { mov(d, Z) });
            }
            AvrOp::Mask => {
                let (d, c) = (r(0), r(1));
                self.ws(&[mov(Z, ZERO), sub(Z, c), mov(d, Z), mov(d + 1, Z)]);
            }
            AvrOp::SetCmp => {
                let (d, a, b, pred, cw) = (r(0), r(1), r(2), n(3), n(4));
                let (swapped, flag, set) = pred_branch(pred);
                self.compare(a, b, swapped, cw);
                if n(5) == 0 {
                    // Public operands: skip over the `ldi` of 0.
                    self.ws(&[ldi(Z, 1), br(flag, set, 1), ldi(Z, 0), mov(d, Z), mov(d + 1, ZERO)]);
                    return;
                }
                // Secret operands: branch-free — read the flag out of SREG.
                self.w(in_(Z, SREG));
                match flag {
                    FLAG_Z => self.w(lsr(Z)),
                    FLAG_S => self.w(swap(Z)),
                    _ => {}
                }
                if !set {
                    self.w(com(Z));
                }
                self.ws(&[andi(Z, 1), mov(d, Z), mov(d + 1, ZERO)]);
            }
            AvrOp::Load => {
                let (d, p, size, space, atomic) = (r(0), r(1), n(2), n(3), n(4) != 0);
                self.w(movw(Z, p));
                if space == 1 {
                    if size == 1 {
                        self.w(lpm(d));
                    } else {
                        self.ws(&[lpm_inc(d), lpm(d + 1)]);
                    }
                } else {
                    let guard = atomic && size > 1;
                    if guard {
                        self.ws(&[in_(TMP, SREG), CLI]);
                    }
                    for k in 0..size as u8 {
                        self.w(ldd(d + k, false, k));
                    }
                    if guard {
                        self.w(out(SREG, TMP));
                    }
                }
            }
            AvrOp::Store => {
                let (p, v, size, atomic) = (r(0), r(1), n(2), n(3) != 0);
                self.w(movw(Z, p));
                let guard = atomic && size > 1;
                if guard {
                    self.ws(&[in_(TMP, SREG), CLI]);
                }
                for k in 0..size as u8 {
                    self.w(std(false, k, v + k));
                }
                if guard {
                    self.w(out(SREG, TMP));
                }
            }
            AvrOp::LoadSlot | AvrOp::StoreSlot => {
                let load = AvrOp::decode(i.opcode) == AvrOp::LoadSlot;
                let q = 1 + self.ctx.layout.slot_off[slot_of(&ops[1]).index()] + n(2);
                if n(3) > 0 {
                    self.y_access(load, r(0), q, n(3));
                }
            }
            AvrOp::StoreFrame | AvrOp::LoadFrame => {
                let load = AvrOp::decode(i.opcode) == AvrOp::LoadFrame;
                let q = 1 + self.ctx.layout.slot_off[slot_of(&ops[1]).index()];
                self.y_access(load, r(0), q, 2);
            }
            AvrOp::FrameAddr => {
                let q = 1 + self.ctx.layout.slot_off[slot_of(&ops[1]).index()];
                self.w(movw(Z, Y));
                self.z_add(q);
                self.w(movw(r(0), Z));
            }
            op @ (AvrOp::GlobalAddr | AvrOp::FuncAddr) => {
                let d = r(0);
                let (sym, lo_k, hi_k) = match (op, &ops[1]) {
                    (AvrOp::GlobalAddr, MachineOperand::Global(g)) => {
                        ((self.ctx.global_name)(*g), RelocKind::AvrLo8Ldi, RelocKind::AvrHi8Ldi)
                    }
                    (_, MachineOperand::Func(f)) => {
                        ((self.ctx.func_name)(*f), RelocKind::AvrLo8LdiPm, RelocKind::AvrHi8LdiPm)
                    }
                    (_, other) => panic!("address of {other:?}"),
                };
                let t = if d >= 16 { d } else { Z };
                self.items.push(Item::R(ldi(t, 0), lo_k, sym.clone(), 0));
                self.items.push(Item::R(ldi(t + 1, 0), hi_k, sym, 0));
                if t != d {
                    self.w(movw(d, Z));
                }
            }
            AvrOp::LoadArg => {
                let q = self.ctx.layout.locals + self.ctx.layout.pushed() + 3 + n(1);
                self.y_access(true, r(0), q, n(2));
            }
            AvrOp::PushArg => {
                let (v, size) = (r(0), n(1));
                if size == 2 {
                    self.w(push(v + 1));
                }
                self.w(push(v));
            }
            AvrOp::PopArgs => {
                let k = n(0);
                if k <= 6 {
                    for _ in 0..k {
                        self.w(pop(TMP));
                    }
                } else {
                    self.ws(&[in_(Z, SPL), in_(Z + 1, SPH)]);
                    self.z_add(k);
                    self.set_sp(Z);
                }
            }
            AvrOp::Call => match &ops[0] {
                MachineOperand::Func(f) => {
                    let name = (self.ctx.func_name)(*f);
                    self.items.push(Item::Call(name));
                }
                MachineOperand::Use(Reg::Physical(p)) => {
                    self.ws(&[movw(Z, p.num as u8), ICALL]);
                }
                other => panic!("Call target {other:?}"),
            },
            AvrOp::Ret => self.w(RET),
            AvrOp::Jmp => self.items.push(Item::J(label_of(&ops[0]))),
            AvrOp::BrCond => {
                self.w(tst(r(0)));
                self.items.push(Item::Br { flag: FLAG_Z, set: false, target: label_of(&ops[1]) });
                self.items.push(Item::J(label_of(&ops[2])));
            }
            AvrOp::CmpBr => {
                let (swap, flag, set) = pred_branch(n(2));
                self.compare(r(0), r(1), swap, n(3));
                self.items.push(Item::Br { flag, set, target: label_of(&ops[4]) });
                self.items.push(Item::J(label_of(&ops[5])));
            }
            AvrOp::Switch => {
                let (c, cw) = (r(0), n(1));
                let default = label_of(&ops[2]);
                let mut k = 3;
                while k + 1 < ops.len() {
                    let v = n(k) as u16;
                    self.ws(&[ldi(Z, v as u8), cp(c, Z)]);
                    if cw == 16 {
                        self.ws(&[ldi(Z + 1, (v >> 8) as u8), cpc(c + 1, Z + 1)]);
                    }
                    self.items.push(Item::Br { flag: FLAG_Z, set: true, target: label_of(&ops[k + 1]) });
                    k += 2;
                }
                self.items.push(Item::J(default));
            }
            AvrOp::Unreachable => self.ws(&[BREAK, rjmp(-1)]),
            AvrOp::Push => self.w(push(n(0) as u8)),
            AvrOp::Pop => self.w(pop(n(0) as u8)),
            AvrOp::FrameEnter => {
                let k = n(0);
                self.ws(&[in_(Y, SPL), in_(Y + 1, SPH)]);
                if k > 0 {
                    if k <= 63 {
                        self.w(sbiw(Y, k as u8));
                    } else {
                        self.ws(&[subi(Y, k as u8), sbci(Y + 1, (k >> 8) as u8)]);
                    }
                    self.set_sp(Y);
                }
            }
            AvrOp::FrameLeave => {
                let k = n(0);
                if k > 0 {
                    if k <= 63 {
                        self.w(adiw(Y, k as u8));
                    } else {
                        let m = (k as u16).wrapping_neg();
                        self.ws(&[subi(Y, m as u8), sbci(Y + 1, (m >> 8) as u8)]);
                    }
                }
                self.set_sp(Y);
            }
            AvrOp::DynAlloca => {
                let (d, cnt) = (r(0), r(1));
                self.ws(&[in_(Z, SPL), in_(Z + 1, SPH), sub(Z, cnt), sbc(Z + 1, cnt + 1)]);
                self.set_sp(Z);
                self.ws(&[adiw(Z, 1), movw(d, Z)]);
            }
            AvrOp::AtomicRmw => {
                let (d, p, v, size, op) = (r(0), r(1), r(2), n(3) as u8, n(4));
                let op = RmwOp::from_code(op).expect("a valid rmw code");
                self.ws(&[movw(Z, p), in_(TMP, SREG), CLI]);
                for k in 0..size {
                    self.w(ldd(d + k, false, k));
                }
                match op {
                    RmwOp::Xchg => {
                        for k in 0..size {
                            self.w(std(false, k, v + k));
                        }
                    }
                    RmwOp::Max | RmwOp::Min | RmwOp::UMax | RmwOp::UMin => {
                        self.compare(d, v, false, 8 * u64::from(size));
                        let (flag, set) = match op {
                            RmwOp::Max => (FLAG_S, false), // keep old if old >= v: brge
                            RmwOp::Min => (FLAG_S, true),  // keep old if old < v: brlt
                            RmwOp::UMax => (FLAG_C, false),
                            _ => (FLAG_C, true),
                        };
                        self.w(br(flag, set, i32::from(size)));
                        for k in 0..size {
                            self.w(std(false, k, v + k));
                        }
                    }
                    _ => {
                        for k in 0..size {
                            self.w(mov(ZERO, d + k));
                            let w = match (op, k) {
                                (RmwOp::Add, 0) => add(ZERO, v + k),
                                (RmwOp::Add, _) => adc(ZERO, v + k),
                                (RmwOp::Sub, 0) => sub(ZERO, v + k),
                                (RmwOp::Sub, _) => sbc(ZERO, v + k),
                                (RmwOp::And | RmwOp::Nand, _) => and(ZERO, v + k),
                                (RmwOp::Or, _) => or(ZERO, v + k),
                                _ => eor(ZERO, v + k),
                            };
                            self.w(w);
                            if op == RmwOp::Nand {
                                self.w(com(ZERO));
                            }
                            self.w(std(false, k, ZERO));
                        }
                        self.w(eor(ZERO, ZERO));
                    }
                }
                self.w(out(SREG, TMP));
            }
            AvrOp::CmpXchg => {
                let (d, p, e, nw, size) = (r(0), r(1), r(2), r(3), n(4) as u8);
                self.ws(&[movw(Z, p), in_(TMP, SREG), CLI]);
                for k in 0..size {
                    self.w(ldd(d + k, false, k));
                }
                self.compare(d, e, false, 8 * u64::from(size));
                self.w(brbc(FLAG_Z, i32::from(size))); // brne over the stores
                for k in 0..size {
                    self.w(std(false, k, nw + k));
                }
                self.w(out(SREG, TMP));
            }
        }
    }

    fn shift_const(&mut self, op: AvrOp, d: u8, a: u8, k: u64, cw: u64) {
        if d != a {
            self.w(movw(d, a));
        }
        let mut k = k;
        if cw == 8 {
            for _ in 0..k {
                self.w(match op {
                    AvrOp::ShlC => lsl(d),
                    AvrOp::LshrC => lsr(d),
                    _ => asr(d),
                });
            }
            return;
        }
        let (lo, hi) = (d, d + 1);
        match op {
            AvrOp::ShlC => {
                if k >= 8 {
                    self.ws(&[mov(hi, lo), mov(lo, ZERO)]);
                    k -= 8;
                }
                for _ in 0..k {
                    self.ws(&[lsl(lo), rol(hi)]);
                }
            }
            AvrOp::LshrC => {
                if k >= 8 {
                    self.ws(&[mov(lo, hi), mov(hi, ZERO)]);
                    k -= 8;
                }
                for _ in 0..k {
                    self.ws(&[lsr(hi), ror(lo)]);
                }
            }
            _ => {
                if k == 15 {
                    self.ws(&[lsl(hi), sbc(hi, hi), mov(lo, hi)]);
                    return;
                }
                if k >= 8 {
                    self.ws(&[mov(lo, hi), lsl(hi), sbc(hi, hi)]);
                    k -= 8;
                }
                for _ in 0..k {
                    self.ws(&[asr(hi), ror(lo)]);
                }
            }
        }
    }
}

/// Resolve the relaxable branches of `items` and produce the function's bytes
/// and relocations. `self_sym` names the function (for relocated `jmp`s).
fn assemble(items: &[Item], self_sym: &str) -> Emitted {
    // Size in words of each relaxable item (0 = not yet decided / fixed).
    let mut size: Vec<u32> = items
        .iter()
        .map(|it| match it {
            Item::W(_) | Item::R(..) => 1,
            Item::Call(_) => 2,
            Item::Br { .. } => 1,
            Item::J(_) => 1,
            Item::Label(_) => 0,
        })
        .collect();
    // A jump to the label(s) right after it is a fallthrough.
    let falls = |i: usize, t: usize| -> bool {
        items[i + 1..].iter().take_while(|it| matches!(it, Item::Label(_))).any(|it| matches!(it, Item::Label(l) if *l == t))
    };
    for (i, it) in items.iter().enumerate() {
        if let Item::J(t) = it
            && falls(i, *t)
        {
            size[i] = 0;
        }
    }
    let max_label = items.iter().filter_map(|it| if let Item::Label(l) = it { Some(*l) } else { None }).max().unwrap_or(0);
    let mut label_at = vec![0u32; max_label + 1];
    let mut at = vec![0u32; items.len()];
    loop {
        let mut pc = 0u32;
        for (i, it) in items.iter().enumerate() {
            at[i] = pc;
            if let Item::Label(l) = it {
                label_at[*l] = pc;
            }
            pc += size[i];
        }
        let mut changed = false;
        for (i, it) in items.iter().enumerate() {
            let need = match it {
                Item::Br { target, .. } => {
                    let k = label_at[*target] as i64 - (at[i] as i64 + 1);
                    let k2 = label_at[*target] as i64 - (at[i] as i64 + 2);
                    if (-64..=63).contains(&k) {
                        1
                    } else if (-2048..=2047).contains(&k2) {
                        2
                    } else {
                        3
                    }
                }
                Item::J(target) if size[i] != 0 => {
                    let k = label_at[*target] as i64 - (at[i] as i64 + 1);
                    if (-2048..=2047).contains(&k) { 1 } else { 2 }
                }
                _ => size[i],
            };
            if need > size[i] {
                size[i] = need;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut bytes = Vec::new();
    let mut relocations = Vec::new();
    let put = |bytes: &mut Vec<u8>, w: u16| bytes.extend_from_slice(&w.to_le_bytes());
    for (i, it) in items.iter().enumerate() {
        let a = at[i] as i64;
        match it {
            Item::W(w) => put(&mut bytes, *w),
            Item::R(w, kind, sym, addend) => {
                relocations.push(EmittedReloc { offset: bytes.len() as u64, symbol: sym.clone(), kind: *kind, addend: *addend });
                put(&mut bytes, *w);
            }
            Item::Call(sym) => {
                relocations.push(EmittedReloc { offset: bytes.len() as u64, symbol: sym.clone(), kind: RelocKind::AvrCall, addend: 0 });
                for w in call(0) {
                    put(&mut bytes, w);
                }
            }
            Item::Br { flag, set, target } => {
                let t = i64::from(label_at[*target]);
                match size[i] {
                    1 => put(&mut bytes, br(*flag, *set, (t - (a + 1)) as i32)),
                    2 => {
                        put(&mut bytes, br(*flag, !*set, 1));
                        put(&mut bytes, rjmp((t - (a + 2)) as i32));
                    }
                    _ => {
                        put(&mut bytes, br(*flag, !*set, 2));
                        relocations.push(EmittedReloc {
                            offset: bytes.len() as u64,
                            symbol: self_sym.to_owned(),
                            kind: RelocKind::AvrCall,
                            addend: 2 * t,
                        });
                        for w in jmp(0) {
                            put(&mut bytes, w);
                        }
                    }
                }
            }
            Item::J(target) => {
                let t = i64::from(label_at[*target]);
                match size[i] {
                    0 => {}
                    1 => put(&mut bytes, rjmp((t - (a + 1)) as i32)),
                    _ => {
                        relocations.push(EmittedReloc {
                            offset: bytes.len() as u64,
                            symbol: self_sym.to_owned(),
                            kind: RelocKind::AvrCall,
                            addend: 2 * t,
                        });
                        for w in jmp(0) {
                            put(&mut bytes, w);
                        }
                    }
                }
            }
            Item::Label(_) => {}
        }
    }
    Emitted { bytes, relocations }
}

/// Encode an allocated, prologue-inserted machine function. The entry block
/// comes first; the rest follow in arena order.
pub fn encode_function(
    mf: &MachineFunction,
    layout: &FrameLayout,
    self_sym: &str,
    func_name: &dyn Fn(u32) -> String,
    global_name: &dyn Fn(u32) -> String,
) -> Emitted {
    let entry = mf.entry().expect("a compiled function has an entry block");
    let mut order = vec![entry];
    order.extend(mf.block_ids().filter(|&b| b != entry));
    let mut code = Code { items: Vec::new(), ctx: Ctx { layout, func_name, global_name } };
    for b in order {
        code.items.push(Item::Label(b.index()));
        for i in &mf.block(b).insts {
            code.expand(i);
        }
    }
    assemble(&code.items, self_sym)
}

// ===========================================================================
// Function + module drivers
// ===========================================================================

/// Compile one function of a **prepared** module (see [`super::prepare`]):
/// isel → register allocation → frame layout → prologue/epilogue → encoding.
fn compile_function_full(
    module: &Module,
    func: crate::ir::FuncId,
    target: &AvrTarget,
    names: &dyn Fn(u32) -> String,
    globals: &dyn Fn(u32) -> String,
) -> (Emitted, StackUsage) {
    let mut mf = target.select(module, func);
    let isel_slots = mf.frame().len();
    regalloc::allocate(&mut mf, target);
    let layout = layout_frame(&mf, isel_slots);
    insert_prologue_epilogue(&mut mf, &layout);
    let usage = layout.stack_usage(&mf, names);
    let me = names(func.index() as u32);
    (encode_function(&mf, &layout, &me, names, globals), usage)
}

/// Compile every defined function of `module` into an AVR relocatable object
/// with the default [`CodegenOptions`] for an ATmega328P. See
/// [`compile_module_with`].
pub fn compile_module(module: &Module, syms: &StrInterner) -> ObjectModule {
    compile_module_with(module, syms, &CodegenOptions::default()).object
}

/// Compile `module` for the ATmega328P (AVR5, with `mul`) under `opts`.
///
/// # Panics
///
/// If `opts` asks for position-independent code, or the module cannot be
/// lowered (see [`super::prepare`]).
pub fn compile_module_with(module: &Module, syms: &StrInterner, opts: &CodegenOptions) -> CompiledModule {
    compile_module_for_device(module, syms, opts, &super::Device::ATMEGA328P)
}

/// Compile `module` for `device` under `opts`: prepare a copy (soft float,
/// integer legalization, runtime calls), then compile each function and emit
/// the globals (address-space-1 globals to `.progmem.data`, in flash).
///
/// # Panics
///
/// As [`compile_module_with`].
pub fn compile_module_for_device(
    module: &Module,
    syms: &StrInterner,
    opts: &CodegenOptions,
    device: &super::Device,
) -> CompiledModule {
    if let Err(e) = crate::target::check_options(crate::target::TargetArch::Avr, opts) {
        panic!("{e}");
    }
    let (m, s, helpers) = super::prepare::prepare(module, syms, device).unwrap_or_else(|e| panic!("avr backend: {e}"));
    let target = AvrTarget::new(device.has_mul).with_helpers(helpers);
    let mut obj = ObjectModule::new(m.name.clone());
    let text = obj.add_section(Section::new(".text", SectionKind::Text, 2));
    let mut stack = StackReport::new();
    let names = |idx: u32| -> String { s.resolve(m.function(crate::ir::FuncId::from_index(idx as usize)).name).to_owned() };
    let globals = |idx: u32| -> String { s.resolve(m.global(crate::ir::GlobalId::from_index(idx as usize)).name).to_owned() };
    for (i, f) in m.functions().enumerate() {
        if f.is_declaration() {
            continue;
        }
        let fid = crate::ir::FuncId::from_index(i);
        let (emitted, usage) = compile_function_full(&m, fid, &target, &names, &globals);
        stack.push(usage);
        let off = obj.section(text).bytes.len() as u64;
        let len = emitted.bytes.len() as u64;
        obj.section_mut(text).bytes.extend_from_slice(&emitted.bytes);
        let name = s.resolve(f.name).to_owned();
        obj.add_symbol(Symbol::defined(name, SymbolBinding::Global, SymbolType::Func, text, off, len));
        for r in &emitted.relocations {
            let sym = obj.reference_symbol(&r.symbol);
            obj.add_relocation(crate::mc::object::Relocation {
                section: text,
                offset: off + r.offset,
                symbol: sym,
                kind: r.kind,
                addend: r.addend,
            });
        }
    }
    super::data::emit_globals(&m, &s, &mut obj);
    crate::codegen::linkage::apply_symbol_attrs(&m, &s, &mut obj);
    CompiledModule { object: obj, stack }
}
