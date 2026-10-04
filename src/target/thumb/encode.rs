//! The Thumb-2 (ARMv7-M) machine-code encoder, the frame layout and the
//! compile entry points.
//!
//! After instruction selection ([`super::isel`]) and register allocation
//! ([`crate::codegen::regalloc`]) a [`MachineFunction`] holds only physical
//! registers and [`ThOp`] opcodes. This module:
//!
//! 1. lays out the stack frame ([`layout_frame`]): the callee-saved registers
//!    the allocation used plus `lr`, pushed together; below them the
//!    spill/`alloca` slots and the outgoing argument area; and one `sub sp`
//!    that keeps the frame a multiple of 8 bytes (the AAPCS stack alignment
//!    at public interfaces);
//! 2. splices in the prologue/epilogue ([`insert_prologue_epilogue`]):
//!    `push {r4.., lr}` / `sub sp, #n`, and before every return
//!    `add sp, #n` / `pop {r4.., pc}` (the return itself);
//! 3. encodes every instruction to one or two **halfwords**, little-endian,
//!    choosing the 16-bit encoding whenever the registers (`r0`–`r7`) and the
//!    immediate fit it, and the 32-bit Thumb-2 encoding otherwise
//!    ([`encode_function`]). Branches are **relaxed**: each starts as a 16-bit
//!    `b`/`b<cond>` and grows to `b.w`/`b<cond>.w` only when its target is out
//!    of range, until the layout is stable; a jump to the next block is
//!    dropped. Calls, global and function addresses become relocations
//!    (`R_ARM_THM_CALL`, `R_ARM_THM_MOVW_ABS_NC` / `R_ARM_THM_MOVT_ABS`);
//! 4. assembles the functions of a prepared module into an [`ObjectModule`]
//!    ([`compile_module_thumb`]), each function symbol carrying the Thumb bit
//!    (bit 0 of its value, *ELF for the Arm Architecture* §5.5.3) and a `$t`
//!    mapping symbol marking the section's code as Thumb.
//!
//! **Conditional execution.** Comparisons and selects are branch-free `IT`
//! blocks: `cmp a, b; ite <cond>; mov<cond> d, #1; mov<!cond> d, #0`, and
//! `tst c, #1; ite ne; movne d, t; moveq d, f`. Inside an IT block the 16-bit
//! `mov` immediate does not set the flags.
//!
//! **Constants and addresses.** A constant is `movs` (8 bits), `mov.w`/`mvn`
//! (a Thumb-2 *modified immediate*: an 8-bit value rotated, or replicated
//! across the halfwords or bytes), `movw` (16 bits) or `movw`+`movt`. An
//! address is always `movw`+`movt` against the symbol: no literal pools, so
//! `.text` is pure code.
//!
//! **Large frames and stack probes.** A frame offset beyond the immediate
//! forms goes through `ip`. With probes on (the default, see
//! [`crate::codegen::stack`]) a `sub sp` of at least [`STACK_PROBE_INTERVAL`]
//! bytes moves `sp` one interval at a time and stores to each new top:
//!
//! ```text
//! movw ip, #pages
//! L: sub.w sp, sp, #4096 ; str.w ip, [sp] ; subs.w ip, ip, #1 ; bne L
//! sub sp, #remainder
//! ```
//!
//! A Cortex-M has no guard page unless an MPU region provides one, so probing
//! only helps where the stack is bounded by one; it is harmless otherwise.
//!
//! The encodings are implemented from the ARMv7-M Architecture Reference
//! Manual (chapter A7, "Instruction Details"), not copied from an assembler,
//! and are checked against `llvm-mc --triple=thumbv7m` in the tests.

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, Reg, StackSlot};
use crate::codegen::options::{CodegenOptions, CompiledModule};
use crate::codegen::regalloc;
use crate::codegen::stack::{STACK_PROBE_INTERVAL, StackReport, StackUsage, scan_calls};
use crate::ir::Module;
use crate::mc::emit::{Emitted, EmittedReloc};
use crate::mc::object::{ObjectModule, RelocKind, Section, SectionKind, Symbol, SymbolBinding, SymbolType};
use crate::support::StrInterner;

use super::isel::{Helpers, ThOp, ThumbTarget};
use super::regs::{IP, LR, PC, SP};

// ===========================================================================
// Instruction builders (from the ARMv7-M ARM, A7.7)
// ===========================================================================

/// One encoded instruction: a 16-bit halfword or a 32-bit pair (first
/// halfword, second halfword), each stored little-endian.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum T {
    /// A 16-bit instruction.
    N(u16),
    /// A 32-bit instruction: its two halfwords in order.
    W(u16, u16),
}

impl T {
    /// The instruction's bytes.
    pub fn bytes(self) -> Vec<u8> {
        match self {
            T::N(h) => h.to_le_bytes().to_vec(),
            T::W(a, b) => {
                let mut v = a.to_le_bytes().to_vec();
                v.extend_from_slice(&b.to_le_bytes());
                v
            }
        }
    }

    /// The size in bytes (2 or 4).
    pub fn len(self) -> usize {
        match self {
            T::N(_) => 2,
            T::W(..) => 4,
        }
    }

    /// Always `false`: an instruction has at least one halfword.
    pub fn is_empty(self) -> bool {
        false
    }
}

#[inline]
fn low(r: u32) -> bool {
    r < 8
}

/// Encode `v` as a Thumb-2 *modified immediate* (`i:imm3:imm8`), if it is one:
/// an 8-bit value, `0x00XY00XY`, `0xXY00XY00`, `0xXYXYXYXY`, or an 8-bit value
/// with its top bit set rotated right by 8..=31.
pub fn mod_imm(v: u32) -> Option<u32> {
    if v < 0x100 {
        return Some(v);
    }
    let b0 = v & 0xff;
    let b1 = (v >> 8) & 0xff;
    if b0 != 0 && v == (b0 << 16) | b0 {
        return Some(0x100 | b0);
    }
    if b1 != 0 && v == (b1 << 24) | (b1 << 8) {
        return Some(0x200 | b1);
    }
    if b0 != 0 && v == b0.wrapping_mul(0x0101_0101) {
        return Some(0x300 | b0);
    }
    for rot in 8..32 {
        let x = v.rotate_left(rot);
        if x < 0x100 && x & 0x80 != 0 {
            return Some((rot << 7) | (x & 0x7f));
        }
    }
    None
}

// --- 16-bit encodings -------------------------------------------------------

/// `lsls/lsrs/asrs d, m, #imm5` (`kind` 0/1/2).
pub fn shift_imm16(kind: u32, d: u32, m: u32, imm5: u32) -> T {
    T::N((kind << 11 | imm5 << 6 | m << 3 | d) as u16)
}
/// `adds d, n, m` / `subs d, n, m`.
pub fn addsub_reg16(sub: bool, d: u32, n: u32, m: u32) -> T {
    T::N((0x1800 | u32::from(sub) << 9 | m << 6 | n << 3 | d) as u16)
}
/// `adds d, n, #imm3` / `subs d, n, #imm3`.
pub fn addsub_imm3(sub: bool, d: u32, n: u32, imm3: u32) -> T {
    T::N((0x1c00 | u32::from(sub) << 9 | imm3 << 6 | n << 3 | d) as u16)
}
/// `movs d, #imm8` (`mov` inside an IT block).
pub fn movs_imm8(d: u32, imm8: u32) -> T {
    T::N((0x2000 | d << 8 | imm8) as u16)
}
/// `cmp n, #imm8`.
pub fn cmp_imm8(n: u32, imm8: u32) -> T {
    T::N((0x2800 | n << 8 | imm8) as u16)
}
/// `adds dn, #imm8` / `subs dn, #imm8`.
pub fn addsub_imm8(sub: bool, dn: u32, imm8: u32) -> T {
    T::N((0x3000 | u32::from(sub) << 11 | dn << 8 | imm8) as u16)
}
/// The 16-bit data-processing (register) group: `op` 0 `ands`, 1 `eors`, 2
/// `lsls`, 3 `lsrs`, 4 `asrs`, 8 `tst`, 9 `rsbs #0`, 10 `cmp`, 12 `orrs`, 13
/// `muls`, 14 `bics`, 15 `mvns`; `dn` the first operand (and destination),
/// `m` the second.
pub fn dp16(op: u32, dn: u32, m: u32) -> T {
    T::N((0x4000 | op << 6 | m << 3 | dn) as u16)
}
/// `add dn, m` (any registers, flags untouched).
pub fn add_hi(dn: u32, m: u32) -> T {
    T::N((0x4400 | (dn >> 3) << 7 | m << 3 | (dn & 7)) as u16)
}
/// `cmp n, m` with at least one high register.
pub fn cmp_hi(n: u32, m: u32) -> T {
    T::N((0x4500 | (n >> 3) << 7 | m << 3 | (n & 7)) as u16)
}
/// `mov d, m` (any registers, flags untouched).
pub fn mov_reg16(d: u32, m: u32) -> T {
    T::N((0x4600 | (d >> 3) << 7 | m << 3 | (d & 7)) as u16)
}
/// `blx m`.
pub fn blx(m: u32) -> T {
    T::N((0x4780 | m << 3) as u16)
}
/// `bx m`.
pub fn bx(m: u32) -> T {
    T::N((0x4700 | m << 3) as u16)
}
/// The 16-bit load/store (immediate) forms: `size` 4/2/1, `off` a multiple of
/// the size below `32 * size`.
pub fn ldst_imm16(load: bool, size: u32, t: u32, n: u32, off: u32) -> T {
    let (base, scaled) = match size {
        4 => (0x6000, off / 4),
        1 => (0x7000, off),
        _ => (0x8000, off / 2),
    };
    T::N((base | u32::from(load) << 11 | scaled << 6 | n << 3 | t) as u16)
}
/// `ldr/str t, [sp, #off]` (`off` a multiple of 4 up to 1020).
pub fn ldst_sp16(load: bool, t: u32, off: u32) -> T {
    T::N((0x9000 | u32::from(load) << 11 | t << 8 | (off / 4)) as u16)
}
/// `add d, sp, #off` (`off` a multiple of 4 up to 1020).
pub fn add_rd_sp16(d: u32, off: u32) -> T {
    T::N((0xa800 | d << 8 | (off / 4)) as u16)
}
/// `add sp, #off` / `sub sp, #off` (`off` a multiple of 4 up to 508).
pub fn addsub_sp16(sub: bool, off: u32) -> T {
    T::N((0xb000 | u32::from(sub) << 7 | (off / 4)) as u16)
}
/// `sxth`/`sxtb`/`uxth`/`uxtb d, m` (`kind` 0/1/2/3).
pub fn ext16(kind: u32, d: u32, m: u32) -> T {
    T::N((0xb200 | kind << 6 | m << 3 | d) as u16)
}
/// `push {list}` (`list` the r0-r7 bits, plus `lr`).
pub fn push16(list: u32, lr: bool) -> T {
    T::N((0xb400 | u32::from(lr) << 8 | list) as u16)
}
/// `pop {list}` (`list` the r0-r7 bits, plus `pc`).
pub fn pop16(list: u32, pc: bool) -> T {
    T::N((0xbc00 | u32::from(pc) << 8 | list) as u16)
}
/// `it<mask> firstcond` (`mask` the raw 4-bit field).
pub fn it(firstcond: u32, mask: u32) -> T {
    T::N((0xbf00 | firstcond << 4 | mask) as u16)
}
/// The `IT` mask of an `ITE <cond>` block (then, else).
pub fn ite_mask(cond: u32) -> u32 {
    ((!cond & 1) << 3) | 0b100
}
/// `udf #imm8`.
pub fn udf(imm8: u32) -> T {
    T::N((0xde00 | imm8) as u16)
}
/// `svc #imm8`.
pub fn svc(imm8: u32) -> T {
    T::N((0xdf00 | imm8) as u16)
}
/// `b<cond> <off>` (16-bit, `off` relative to the instruction + 4, even, in
/// -256..=254).
pub fn bcond16(cond: u32, off: i32) -> T {
    T::N((0xd000 | cond << 8 | ((off >> 1) as u32 & 0xff)) as u16)
}
/// `b <off>` (16-bit, in -2048..=2046).
pub fn b16(off: i32) -> T {
    T::N((0xe000 | ((off >> 1) as u32 & 0x7ff)) as u16)
}

// --- 32-bit encodings ---------------------------------------------------------

/// Data processing with a modified immediate: `op` 0 `and`, 1 `bic`, 2 `orr`
/// (`mov` with `n` = 15), 3 `orn` (`mvn` with `n` = 15), 4 `eor`, 8 `add`,
/// 13 `sub`, 14 `rsb`; `s` sets the flags; `d` = 15 with `s` is
/// `tst`/`cmn`/`cmp`.
pub fn dp_modimm(op: u32, s: bool, d: u32, n: u32, imm12: u32) -> T {
    T::W(
        (0xf000 | (imm12 >> 11) << 10 | op << 5 | u32::from(s) << 4 | n) as u16,
        (((imm12 >> 8) & 7) << 12 | d << 8 | (imm12 & 0xff)) as u16,
    )
}
/// Data processing with a plain 12- or 16-bit immediate: `op` 0 `addw`,
/// 10 `subw` (`imm` 12 bits), 4 `movw`, 12 `movt` (`imm` 16 bits).
pub fn dp_plainimm(op: u32, d: u32, n: u32, imm: u32) -> T {
    let (n, imm) = if op == 4 || op == 12 { (imm >> 12, imm & 0xfff) } else { (n, imm) };
    T::W(
        (0xf200 | ((imm >> 11) & 1) << 10 | op << 4 | n) as u16,
        (((imm >> 8) & 7) << 12 | d << 8 | (imm & 0xff)) as u16,
    )
}
/// `movw d, #imm16`.
pub fn movw(d: u32, imm16: u32) -> T {
    dp_plainimm(4, d, 0, imm16)
}
/// `movt d, #imm16`.
pub fn movt(d: u32, imm16: u32) -> T {
    dp_plainimm(12, d, 0, imm16)
}
/// `ubfx`/`sbfx d, n, #lsb, #width`.
pub fn bfx(signed: bool, d: u32, n: u32, lsb: u32, width: u32) -> T {
    T::W(
        (if signed { 0xf340 } else { 0xf3c0 } | n) as u16,
        ((lsb >> 2) << 12 | d << 8 | (lsb & 3) << 6 | (width - 1)) as u16,
    )
}
/// Data processing with a (shifted) register: `op` as for [`dp_modimm`]
/// (`orr` with `n` = 15 is `mov`/`lsl`/`lsr`/`asr` by an immediate, `orn`
/// with `n` = 15 is `mvn`); `ty` 0 `lsl`, 1 `lsr`, 2 `asr`.
#[allow(clippy::too_many_arguments)]
pub fn dp_reg(op: u32, s: bool, d: u32, n: u32, m: u32, ty: u32, imm5: u32) -> T {
    T::W(
        (0xea00 | op << 5 | u32::from(s) << 4 | n) as u16,
        ((imm5 >> 2) << 12 | d << 8 | (imm5 & 3) << 6 | ty << 4 | m) as u16,
    )
}
/// `lsl.w/lsr.w/asr.w d, n, m` (`ty` 0/1/2): shift by a register.
pub fn shift_reg32(ty: u32, d: u32, n: u32, m: u32) -> T {
    T::W((0xfa00 | ty << 5 | n) as u16, (0xf000 | d << 8 | m) as u16)
}
/// `sxth.w`/`uxth.w`/`sxtb.w`/`uxtb.w d, m` (`kind` 0/1/4/5).
pub fn ext32(kind: u32, d: u32, m: u32) -> T {
    T::W((0xfa0f | kind << 4) as u16, (0xf080 | d << 8 | m) as u16)
}
/// `mul.w d, n, m`.
pub fn mul32(d: u32, n: u32, m: u32) -> T {
    T::W((0xfb00 | n) as u16, (0xf000 | d << 8 | m) as u16)
}
/// `mls d, n, m, a` (`d = a - n * m`).
pub fn mls(d: u32, n: u32, m: u32, a: u32) -> T {
    T::W((0xfb00 | n) as u16, (a << 12 | d << 8 | 0x10 | m) as u16)
}
/// `sdiv`/`udiv d, n, m`.
pub fn div(signed: bool, d: u32, n: u32, m: u32) -> T {
    T::W((if signed { 0xfb90 } else { 0xfbb0 } | n) as u16, (0xf0f0 | d << 8 | m) as u16)
}
/// The 32-bit load/store opcode bits for a size.
fn ldst_base(load: bool, size: u32) -> u32 {
    let sz = match size {
        1 => 0,
        2 => 1,
        _ => 2,
    };
    0xf800 | sz << 5 | u32::from(load) << 4
}
/// `ldr.w`/`str.w` (and `b`/`h`) `t, [n, #imm12]`.
pub fn ldst_imm12(load: bool, size: u32, t: u32, n: u32, imm12: u32) -> T {
    T::W((ldst_base(load, size) | 0x80 | n) as u16, (t << 12 | imm12) as u16)
}
/// `ldr`/`str` (and `b`/`h`) `t, [n, #-imm8]`.
pub fn ldst_neg8(load: bool, size: u32, t: u32, n: u32, imm8: u32) -> T {
    T::W((ldst_base(load, size) | n) as u16, (t << 12 | 0xc00 | imm8) as u16)
}
/// `ldrd`/`strd t, t2, [n, #off]` (`off` a multiple of 4 up to 1020).
pub fn ldst_dual(load: bool, t: u32, t2: u32, n: u32, off: u32) -> T {
    T::W((0xe9c0 | u32::from(load) << 4 | n) as u16, (t << 12 | t2 << 8 | (off / 4)) as u16)
}
/// `push.w {list}` (`stmdb sp!`, `list` a 16-bit register mask).
pub fn push32(list: u32) -> T {
    T::W(0xe92d, list as u16)
}
/// `pop.w {list}` (`ldmia sp!`).
pub fn pop32(list: u32) -> T {
    T::W(0xe8bd, list as u16)
}
/// `b.w <off>` (±16 MiB) or, with `link`, `bl <off>`.
pub fn b32(link: bool, off: i32) -> T {
    let v = off as u32;
    let s = (v >> 24) & 1;
    let j1 = (((v >> 23) & 1) ^ 1) ^ s;
    let j2 = (((v >> 22) & 1) ^ 1) ^ s;
    T::W(
        (0xf000 | s << 10 | (v >> 12) & 0x3ff) as u16,
        (0x9000 | u32::from(link) << 14 | j1 << 13 | j2 << 11 | (v >> 1) & 0x7ff) as u16,
    )
}
/// `b<cond>.w <off>` (±1 MiB).
pub fn bcond32(cond: u32, off: i32) -> T {
    let v = off as u32;
    let s = (v >> 20) & 1;
    let j2 = (v >> 19) & 1;
    let j1 = (v >> 18) & 1;
    T::W(
        (0xf000 | s << 10 | cond << 6 | (v >> 12) & 0x3f) as u16,
        (0x8000 | j1 << 13 | j2 << 11 | (v >> 1) & 0x7ff) as u16,
    )
}
/// `dmb sy`.
pub fn dmb_sy() -> T {
    T::W(0xf3bf, 0x8f5f)
}


// ===========================================================================
// The assembler: pieces, labels and branch relaxation
// ===========================================================================

/// A relocation inside a code piece: byte offset in the piece, symbol, kind,
/// addend.
type PieceReloc = (usize, String, RelocKind, i64);

/// One unit of a function's code: fixed bytes (with the relocations inside
/// them), a label definition, or a relaxable branch to a label.
#[derive(Clone, Debug)]
enum Piece {
    Code(Vec<u8>, Vec<PieceReloc>),
    Label(usize),
    Branch { cond: u32, label: usize, long: bool },
}

/// The "always" condition: an unconditional branch.
pub(super) const AL: u32 = 0xe;

/// A function's code as [`Piece`]s, relaxed and flattened by [`Asm::finish`].
#[derive(Default)]
pub(super) struct Asm {
    pieces: Vec<Piece>,
    cur: Vec<u8>,
    cur_relocs: Vec<PieceReloc>,
    labels: usize,
}

impl Asm {
    fn flush(&mut self) {
        if !self.cur.is_empty() {
            self.pieces.push(Piece::Code(std::mem::take(&mut self.cur), std::mem::take(&mut self.cur_relocs)));
        }
    }

    /// Append one instruction.
    pub(super) fn i(&mut self, t: T) {
        self.cur.extend_from_slice(&t.bytes());
    }

    /// Append one instruction with a relocation against `symbol` at its start.
    pub(super) fn reloc(&mut self, t: T, symbol: String, kind: RelocKind, addend: i64) {
        self.cur_relocs.push((self.cur.len(), symbol, kind, addend));
        self.i(t);
    }

    pub(super) fn new_label(&mut self) -> usize {
        self.labels += 1;
        self.labels - 1
    }

    pub(super) fn bind(&mut self, label: usize) {
        self.flush();
        self.pieces.push(Piece::Label(label));
    }

    /// A branch to `label` (`cond` [`AL`] for an unconditional one).
    pub(super) fn branch(&mut self, cond: u32, label: usize) {
        self.flush();
        self.pieces.push(Piece::Branch { cond, label, long: false });
    }

    /// Lay the pieces out, growing every branch whose 16-bit form cannot
    /// reach its target, until nothing changes (sizes only grow, so this
    /// terminates); then encode the branches.
    pub(super) fn finish(mut self) -> Emitted {
        self.flush();
        let mut label_off = vec![0i64; self.labels];
        loop {
            let mut off = 0i64;
            for p in &self.pieces {
                match p {
                    Piece::Code(b, _) => off += b.len() as i64,
                    Piece::Label(l) => label_off[*l] = off,
                    Piece::Branch { long, .. } => off += if *long { 4 } else { 2 },
                }
            }
            let mut changed = false;
            let mut off = 0i64;
            for p in &mut self.pieces {
                match p {
                    Piece::Code(b, _) => off += b.len() as i64,
                    Piece::Label(_) => {}
                    Piece::Branch { cond, label, long } => {
                        if !*long {
                            let disp = label_off[*label] - (off + 4);
                            let range = if *cond == AL { -2048..=2046 } else { -256..=254 };
                            if !range.contains(&disp) {
                                *long = true;
                                changed = true;
                            }
                        }
                        off += if *long { 4 } else { 2 };
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut bytes = Vec::new();
        let mut relocations = Vec::new();
        for p in self.pieces {
            match p {
                Piece::Code(b, r) => {
                    for (at, symbol, kind, addend) in r {
                        relocations.push(EmittedReloc { offset: (bytes.len() + at) as u64, symbol, kind, addend });
                    }
                    bytes.extend_from_slice(&b);
                }
                Piece::Label(_) => {}
                Piece::Branch { cond, label, long } => {
                    let disp = (label_off[label] - (bytes.len() as i64 + 4)) as i32;
                    let t = match (cond == AL, long) {
                        (true, false) => b16(disp),
                        (false, false) => bcond16(cond, disp),
                        (true, true) => {
                            assert!((-(1 << 24)..(1 << 24)).contains(&disp), "b.w out of range");
                            b32(false, disp)
                        }
                        (false, true) => {
                            assert!((-(1 << 20)..(1 << 20)).contains(&disp), "b<cond>.w out of range");
                            bcond32(cond, disp)
                        }
                    };
                    bytes.extend_from_slice(&t.bytes());
                }
            }
        }
        Emitted { bytes, relocations }
    }

    // --- idioms -------------------------------------------------------------

    /// `mov d, s` (nothing when equal).
    fn mov(&mut self, d: u32, s: u32) {
        if d != s {
            self.i(mov_reg16(d, s));
        }
    }

    /// Materialize the constant `v` into `d` (may set the flags).
    pub(super) fn mov_imm(&mut self, d: u32, v: u32) {
        if low(d) && v < 0x100 {
            self.i(movs_imm8(d, v));
        } else if let Some(m) = mod_imm(v) {
            self.i(dp_modimm(2, false, d, 15, m));
        } else if let Some(m) = mod_imm(!v) {
            self.i(dp_modimm(3, false, d, 15, m));
        } else {
            self.i(movw(d, v & 0xffff));
            if v >> 16 != 0 {
                self.i(movt(d, v >> 16));
            }
        }
    }

    /// `d = n + k` for a signed `k`, in the shortest form (`n` may be `sp`).
    /// Goes through `ip` for a constant no immediate form holds.
    pub(super) fn add_imm(&mut self, d: u32, n: u32, k: i64) {
        let sp = u32::from(SP);
        let (sub, a) = if k < 0 { (true, k.unsigned_abs()) } else { (false, k as u64) };
        if a == 0 {
            self.mov(d, n);
            return;
        }
        if a <= 0xffff_ffff {
            let a32 = a as u32;
            if n == sp {
                if d == sp && a32.is_multiple_of(4) && a32 <= 508 {
                    return self.i(addsub_sp16(sub, a32));
                }
                if !sub && low(d) && a32.is_multiple_of(4) && a32 <= 1020 {
                    return self.i(add_rd_sp16(d, a32));
                }
            } else if low(d) && low(n) && a32 <= 7 {
                return self.i(addsub_imm3(sub, d, n, a32));
            } else if d == n && low(d) && a32 <= 255 {
                return self.i(addsub_imm8(sub, d, a32));
            }
            if let Some(m) = mod_imm(a32) {
                return self.i(dp_modimm(if sub { 13 } else { 8 }, false, d, n, m));
            }
            if a32 < 4096 {
                return self.i(dp_plainimm(if sub { 10 } else { 0 }, d, n, a32));
            }
        }
        // Through ip: d = n + ip (n may be sp).
        let ip = u32::from(IP);
        assert!(n != ip, "add through ip onto itself");
        self.mov_imm(ip, k as u32);
        self.i(dp_reg(8, false, d, n, ip, 0, 0));
    }

    /// A load (`load`) or store of `size` bytes between `t` and `[n + off]`.
    /// Goes through `ip` for an offset beyond the immediate forms.
    pub(super) fn ldst(&mut self, load: bool, size: u32, t: u32, n: u32, off: i64) {
        let sp = u32::from(SP);
        if off >= 0 {
            let o = off as u32;
            if n == sp && size == 4 && low(t) && o.is_multiple_of(4) && o <= 1020 {
                return self.i(ldst_sp16(load, t, o));
            }
            if n != sp && low(t) && low(n) && o.is_multiple_of(size) && o < 32 * size {
                return self.i(ldst_imm16(load, size, t, n, o));
            }
            if o < 4096 {
                return self.i(ldst_imm12(load, size, t, n, o));
            }
        } else if off >= -255 {
            return self.i(ldst_neg8(load, size, t, n, off.unsigned_abs() as u32));
        }
        let ip = u32::from(IP);
        assert!(load || t != ip, "store of ip at an offset beyond the immediate forms");
        self.mov_imm(ip, off as u32);
        self.i(add_hi(ip, n));
        self.i(ldst_imm12(load, size, t, ip, 0));
    }

    /// `cmp n, #k`, with `cmn` for a negated immediate and `ip` otherwise.
    fn cmp_imm(&mut self, n: u32, k: u32) {
        if low(n) && k < 0x100 {
            self.i(cmp_imm8(n, k));
        } else if let Some(m) = mod_imm(k) {
            self.i(dp_modimm(13, true, 15, n, m));
        } else if let Some(m) = mod_imm(k.wrapping_neg()) {
            self.i(dp_modimm(8, true, 15, n, m));
        } else {
            let ip = u32::from(IP);
            self.mov_imm(ip, k);
            self.i(cmp_hi(n, ip));
        }
    }

    /// `cmp n, m`.
    pub(super) fn cmp_reg(&mut self, n: u32, m: u32) {
        if low(n) && low(m) {
            self.i(dp16(10, n, m));
        } else {
            self.i(cmp_hi(n, m));
        }
    }

    /// `mov d, #v` (`v` 0 or 1) inside an IT block (no flag update).
    fn mov_imm_in_it(&mut self, d: u32, v: u32) {
        if low(d) {
            self.i(movs_imm8(d, v));
        } else {
            self.i(dp_modimm(2, false, d, 15, v));
        }
    }

    /// `tst c, #1`.
    fn tst1(&mut self, c: u32) {
        self.i(dp_modimm(0, true, 15, c, 1));
    }
}

// ===========================================================================
// Frame layout, prologue and epilogue
// ===========================================================================

/// The stack-frame layout of one function, computed after allocation.
///
/// ```text
///   incoming stack arguments      <- sp at entry
///   push {r4.., lr}               push_bytes
///   (padding to 8)
///   spill / alloca slots
///   outgoing argument area        <- sp in the body
/// ```
///
/// Slot offsets are `sp`-relative; `sp` is fixed for the whole body.
#[derive(Clone, Debug)]
pub struct FrameLayout {
    slot_off: Vec<u32>,
    /// The pushed registers (r4-r11 as used, and lr), as a 16-bit mask.
    push_mask: u32,
    /// The `sub sp` amount below the pushed registers.
    sub: u64,
    /// The outgoing stack-argument area at the bottom of the frame.
    outgoing: u64,
    /// Whether the prologue's `sub sp` is probed.
    probes: bool,
}

impl FrameLayout {
    fn push_bytes(&self) -> u64 {
        4 * u64::from(self.push_mask.count_ones())
    }

    /// The whole frame: pushed registers plus the `sub sp` amount.
    pub fn frame_size(&self) -> u64 {
        self.push_bytes() + self.sub
    }

    /// The stack usage this layout gives `mf`: the `push` (callee-saved
    /// registers and `lr`) plus the `sub sp` amount. A `bl` pushes nothing.
    pub fn stack_usage(&self, mf: &MachineFunction, func_name: &dyn Fn(u32) -> String) -> StackUsage {
        let scan = scan_calls(mf, ThOp::Call.opcode(), ThOp::Svc.opcode(), None);
        StackUsage {
            name: func_name(mf.info().source),
            frame_size: self.frame_size(),
            return_address: 0,
            saved_registers: self.push_bytes(),
            sp_adjust: self.sub,
            outgoing_args: self.outgoing,
            dynamic_alloca: scan.dynamic_alloca,
            direct_callees: scan.direct.iter().map(|&f| func_name(f)).collect(),
            indirect_calls: scan.indirect,
            syscalls: scan.syscalls,
            probed: self.probes,
        }
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a.max(1)) * a.max(1)
}

/// Compute the frame layout of an allocated machine function under `opts`.
pub fn layout_frame(mf: &MachineFunction, target: &ThumbTarget, opts: &CodegenOptions) -> FrameLayout {
    use crate::codegen::target::MachineTarget;
    let mut used = [false; 16];
    for bid in mf.block_ids() {
        for inst in &mf.block(bid).insts {
            for d in inst.defs() {
                if let Reg::Physical(p) = d {
                    used[p.num as usize] = true;
                }
            }
        }
    }
    let mut push_mask = 1u32 << LR;
    for p in target.callee_saved() {
        if used[p.num as usize] {
            push_mask |= 1 << p.num;
        }
    }
    let outgoing = align_up(mf.frame().outgoing(), 8);
    let mut off = outgoing;
    let mut slot_off = vec![0u32; mf.frame().len()];
    for (i, so) in slot_off.iter_mut().enumerate() {
        let info = mf.frame().slot(StackSlot::from_index(i));
        off = align_up(off, info.align.clamp(4, 8));
        *so = off as u32;
        off += align_up(info.size.max(1), 4);
    }
    let push_bytes = 4 * u64::from(push_mask.count_ones());
    let sub = align_up(off + push_bytes, 8) - push_bytes;
    FrameLayout { slot_off, push_mask, sub, outgoing, probes: opts.stack_probes }
}

fn imm_op(v: u64) -> MachineOperand {
    MachineOperand::Imm(puremp::Int::from_u64(v))
}

/// Splice the prologue into the entry block and an epilogue before every
/// return.
pub fn insert_prologue_epilogue(mf: &mut MachineFunction, layout: &FrameLayout) {
    let entry = mf.entry().expect("a function being compiled has an entry block");
    let mut prologue = vec![MachineInst::new(ThOp::Push.opcode(), vec![imm_op(u64::from(layout.push_mask))])];
    if layout.sub > 0 {
        prologue.push(MachineInst::new(
            ThOp::SubSp.opcode(),
            vec![imm_op(layout.sub), imm_op(u64::from(layout.probes))],
        ));
    }
    let old = std::mem::take(&mut mf.block_mut(entry).insts);
    prologue.extend(old);
    mf.block_mut(entry).insts = prologue;

    let pop_mask = (layout.push_mask & !(1 << LR)) | (1 << PC);
    let ids: Vec<_> = mf.block_ids().collect();
    for bid in ids {
        let old = std::mem::take(&mut mf.block_mut(bid).insts);
        let mut new = Vec::with_capacity(old.len());
        for inst in old {
            if ThOp::decode(inst.opcode) == ThOp::Ret {
                if layout.sub > 0 {
                    new.push(MachineInst::new(ThOp::AddSp.opcode(), vec![imm_op(layout.sub)]));
                }
                new.push(MachineInst::new(ThOp::Pop.opcode(), vec![imm_op(u64::from(pop_mask))]));
            }
            new.push(inst);
        }
        mf.block_mut(bid).insts = new;
    }
}

// ===========================================================================
// Instruction encoding
// ===========================================================================

fn rn(op: &MachineOperand) -> u32 {
    match op {
        MachineOperand::Def(Reg::Physical(p)) | MachineOperand::Use(Reg::Physical(p)) => u32::from(p.num),
        other => panic!("expected a physical register operand, found {other:?}"),
    }
}

fn uimm(op: &MachineOperand) -> u64 {
    match op {
        MachineOperand::Imm(v) => v.to_u64().or_else(|| v.to_i64().map(|i| i as u64)).unwrap_or(0),
        other => panic!("expected an immediate operand, found {other:?}"),
    }
}

fn simm(op: &MachineOperand) -> i64 {
    match op {
        MachineOperand::Imm(v) => v.to_i64().or_else(|| v.to_u64().map(|u| u as i64)).unwrap_or(0),
        other => panic!("expected an immediate operand, found {other:?}"),
    }
}

fn label_of(op: &MachineOperand) -> usize {
    match op {
        MachineOperand::Label(b) => b.index(),
        other => panic!("expected a label operand, found {other:?}"),
    }
}

fn slot_of(op: &MachineOperand) -> usize {
    match op {
        MachineOperand::Frame(s) => s.index(),
        other => panic!("expected a frame operand, found {other:?}"),
    }
}

/// What the encoder needs to resolve non-local references while emitting.
struct EncodeCtx<'a> {
    layout: &'a FrameLayout,
    func_name: &'a dyn Fn(u32) -> String,
    global_name: &'a dyn Fn(u32) -> String,
    /// The block laid out right after the current one (a jump to it is
    /// dropped).
    next: Option<usize>,
}

/// The 32-bit (shifted-register / modified-immediate) opcode of an op.
fn dp32_op(op: ThOp) -> u32 {
    match op {
        ThOp::And | ThOp::AndImm => 0,
        ThOp::Orr | ThOp::OrrImm => 2,
        ThOp::Eor | ThOp::EorImm => 4,
        ThOp::Add => 8,
        _ => 13, // Sub
    }
}

/// Encode one machine instruction into `a`.
fn encode_inst(a: &mut Asm, inst: &MachineInst, ctx: &EncodeCtx<'_>) {
    let ops = &inst.operands;
    let ip = u32::from(IP);
    let sp = u32::from(SP);
    let op = ThOp::decode(inst.opcode);
    match op {
        ThOp::Mov => a.mov(rn(&ops[0]), rn(&ops[1])),
        ThOp::MovImm => a.mov_imm(rn(&ops[0]), uimm(&ops[1]) as u32),
        ThOp::Add | ThOp::Sub => {
            let (d, n, m) = (rn(&ops[0]), rn(&ops[1]), rn(&ops[2]));
            if low(d) && low(n) && low(m) {
                a.i(addsub_reg16(op == ThOp::Sub, d, n, m));
            } else {
                a.i(dp_reg(dp32_op(op), false, d, n, m, 0, 0));
            }
        }
        ThOp::And | ThOp::Orr | ThOp::Eor => {
            let (d, n, m) = (rn(&ops[0]), rn(&ops[1]), rn(&ops[2]));
            let o16 = match op {
                ThOp::And => 0,
                ThOp::Eor => 1,
                _ => 12,
            };
            if low(d) && low(n) && low(m) && (d == n || d == m) {
                a.i(dp16(o16, d, if d == n { m } else { n }));
            } else {
                a.i(dp_reg(dp32_op(op), false, d, n, m, 0, 0));
            }
        }
        ThOp::Mul => {
            let (d, n, m) = (rn(&ops[0]), rn(&ops[1]), rn(&ops[2]));
            if low(d) && low(n) && low(m) && (d == n || d == m) {
                a.i(dp16(13, d, if d == m { n } else { m }));
            } else {
                a.i(mul32(d, n, m));
            }
        }
        ThOp::Sdiv | ThOp::Udiv => a.i(div(op == ThOp::Sdiv, rn(&ops[0]), rn(&ops[1]), rn(&ops[2]))),
        ThOp::Srem | ThOp::Urem => {
            let (d, n, m) = (rn(&ops[0]), rn(&ops[1]), rn(&ops[2]));
            a.i(div(op == ThOp::Srem, ip, n, m));
            a.i(mls(d, ip, m, n));
        }
        ThOp::Lsl | ThOp::Lsr | ThOp::Asr => {
            let (d, n, m) = (rn(&ops[0]), rn(&ops[1]), rn(&ops[2]));
            let ty = match op {
                ThOp::Lsl => 0,
                ThOp::Lsr => 1,
                _ => 2,
            };
            if low(d) && low(m) && d == n {
                a.i(dp16(2 + ty, d, m));
            } else {
                a.i(shift_reg32(ty, d, n, m));
            }
        }
        ThOp::AddImm => a.add_imm(rn(&ops[0]), rn(&ops[1]), simm(&ops[2])),
        ThOp::AndImm => {
            let (d, n, k) = (rn(&ops[0]), rn(&ops[1]), uimm(&ops[2]) as u32);
            if let Some(m) = mod_imm(k) {
                a.i(dp_modimm(0, false, d, n, m));
            } else if let Some(m) = mod_imm(!k) {
                a.i(dp_modimm(1, false, d, n, m));
            } else if k.wrapping_add(1).is_power_of_two() {
                // A low mask 2^w - 1: an unsigned bitfield extract.
                a.i(bfx(false, d, n, 0, k.count_ones()));
            } else {
                a.mov_imm(ip, k);
                a.i(dp_reg(0, false, d, n, ip, 0, 0));
            }
        }
        ThOp::OrrImm => {
            let (d, n, k) = (rn(&ops[0]), rn(&ops[1]), uimm(&ops[2]) as u32);
            if let Some(m) = mod_imm(k) {
                a.i(dp_modimm(2, false, d, n, m));
            } else if let Some(m) = mod_imm(!k) {
                a.i(dp_modimm(3, false, d, n, m));
            } else {
                a.mov_imm(ip, k);
                a.i(dp_reg(2, false, d, n, ip, 0, 0));
            }
        }
        ThOp::EorImm => {
            let (d, n, k) = (rn(&ops[0]), rn(&ops[1]), uimm(&ops[2]) as u32);
            if k == u32::MAX {
                encode_mvn(a, d, n);
            } else if let Some(m) = mod_imm(k) {
                a.i(dp_modimm(4, false, d, n, m));
            } else {
                a.mov_imm(ip, k);
                a.i(dp_reg(4, false, d, n, ip, 0, 0));
            }
        }
        ThOp::RsbImm => {
            let (d, n, k) = (rn(&ops[0]), rn(&ops[1]), uimm(&ops[2]) as u32);
            if k == 0 && low(d) && low(n) {
                a.i(dp16(9, d, n));
            } else if let Some(m) = mod_imm(k) {
                a.i(dp_modimm(14, false, d, n, m));
            } else {
                a.mov_imm(ip, k);
                a.i(dp_reg(13, false, d, ip, n, 0, 0));
            }
        }
        ThOp::LslImm | ThOp::LsrImm | ThOp::AsrImm => {
            let (d, m, k) = (rn(&ops[0]), rn(&ops[1]), uimm(&ops[2]) as u32);
            let ty = match op {
                ThOp::LslImm => 0,
                ThOp::LsrImm => 1,
                _ => 2,
            };
            if k == 0 {
                a.mov(d, m);
            } else if low(d) && low(m) {
                a.i(shift_imm16(ty, d, m, k & 31));
            } else {
                a.i(dp_reg(2, false, d, 15, m, ty, k & 31));
            }
        }
        ThOp::Mvn => encode_mvn(a, rn(&ops[0]), rn(&ops[1])),
        ThOp::Ext => {
            let (d, m, w, signed) = (rn(&ops[0]), rn(&ops[1]), uimm(&ops[2]) as u32, uimm(&ops[3]) != 0);
            match (w, signed) {
                (8 | 16, _) => {
                    // 16-bit kinds: 0 sxth, 1 sxtb, 2 uxth, 3 uxtb; 32-bit: 0
                    // sxth, 1 uxth, 4 sxtb, 5 uxtb.
                    let byte = w == 8;
                    if low(d) && low(m) {
                        a.i(ext16(u32::from(!signed) << 1 | u32::from(byte), d, m));
                    } else {
                        a.i(ext32(u32::from(byte) << 2 | u32::from(!signed), d, m));
                    }
                }
                (1, false) => a.i(dp_modimm(0, false, d, m, 1)),
                _ => a.i(bfx(signed, d, m, 0, w)),
            }
        }
        ThOp::SetCmp | ThOp::SetCmpImm => {
            let d = rn(&ops[0]);
            let n = rn(&ops[1]);
            if op == ThOp::SetCmp {
                a.cmp_reg(n, rn(&ops[2]));
            } else {
                a.cmp_imm(n, uimm(&ops[2]) as u32);
            }
            let cond = uimm(&ops[3]) as u32;
            a.i(it(cond, ite_mask(cond)));
            a.mov_imm_in_it(d, 1);
            a.mov_imm_in_it(d, 0);
        }
        ThOp::Select => {
            let (d, c, t, f) = (rn(&ops[0]), rn(&ops[1]), rn(&ops[2]), rn(&ops[3]));
            a.tst1(c);
            // NE: bit 0 set, select `t`; EQ: select `f`.
            if d == t && d == f {
                // Both arms already in place.
            } else if d == t {
                a.i(it(0x0, 0b1000));
                a.i(mov_reg16(d, f));
            } else if d == f {
                a.i(it(0x1, 0b1000));
                a.i(mov_reg16(d, t));
            } else {
                a.i(it(0x1, ite_mask(0x1)));
                a.i(mov_reg16(d, t));
                a.i(mov_reg16(d, f));
            }
        }
        ThOp::Load | ThOp::Store => {
            let load = op == ThOp::Load;
            let (t, n) = if load { (rn(&ops[0]), rn(&ops[1])) } else { (rn(&ops[1]), rn(&ops[0])) };
            a.ldst(load, uimm(&ops[3]) as u32, t, n, simm(&ops[2]));
        }
        ThOp::LoadDual => a.i(ldst_dual(true, rn(&ops[0]), rn(&ops[1]), rn(&ops[2]), 0)),
        ThOp::StoreDual => a.i(ldst_dual(false, rn(&ops[1]), rn(&ops[2]), rn(&ops[0]), 0)),
        ThOp::FrameAddr => {
            let off = ctx.layout.slot_off[slot_of(&ops[1])];
            a.add_imm(rn(&ops[0]), sp, i64::from(off));
        }
        ThOp::LoadFrame => {
            let off = ctx.layout.slot_off[slot_of(&ops[1])];
            a.ldst(true, 4, rn(&ops[0]), sp, i64::from(off));
        }
        ThOp::StoreFrame => {
            let off = ctx.layout.slot_off[slot_of(&ops[1])];
            a.ldst(false, 4, rn(&ops[0]), sp, i64::from(off));
        }
        ThOp::SpAddr => a.add_imm(rn(&ops[0]), sp, simm(&ops[1])),
        ThOp::StoreSp => a.ldst(false, uimm(&ops[2]) as u32, rn(&ops[0]), sp, simm(&ops[1])),
        ThOp::IncAddr => {
            let off = ctx.layout.frame_size() as i64 + simm(&ops[1]);
            a.add_imm(rn(&ops[0]), sp, off);
        }
        ThOp::LoadInc => {
            let off = ctx.layout.frame_size() as i64 + simm(&ops[1]);
            a.ldst(true, uimm(&ops[2]) as u32, rn(&ops[0]), sp, off);
        }
        ThOp::GlobalAddr | ThOp::FuncAddr => {
            let d = rn(&ops[0]);
            let sym = match &ops[1] {
                MachineOperand::Global(g) => (ctx.global_name)(*g),
                MachineOperand::Func(f) => (ctx.func_name)(*f),
                other => panic!("expected a symbol operand, found {other:?}"),
            };
            a.reloc(movw(d, 0), sym.clone(), RelocKind::ThumbMovwAbsNc, 0);
            a.reloc(movt(d, 0), sym, RelocKind::ThumbMovtAbs, 0);
        }
        ThOp::Call => match &ops[0] {
            MachineOperand::Func(f) => {
                // `bl` lands at P + 4 + imm: the call's addend is -4, which is
                // also its implicit (REL) addend in the instruction.
                a.reloc(b32(true, -4), (ctx.func_name)(*f), RelocKind::ThumbCall, -4);
            }
            callee => a.i(blx(rn(callee))),
        },
        // The epilogue's `pop {.., pc}` returned already.
        ThOp::Ret => {}
        ThOp::B => {
            let t = label_of(&ops[0]);
            if ctx.next != Some(t) {
                a.branch(AL, t);
            }
        }
        ThOp::BrCond => {
            let (c, t, f) = (rn(&ops[0]), label_of(&ops[1]), label_of(&ops[2]));
            a.tst1(c);
            if ctx.next == Some(t) {
                a.branch(0x0, f); // beq f
            } else {
                a.branch(0x1, t); // bne t
                if ctx.next != Some(f) {
                    a.branch(AL, f);
                }
            }
        }
        ThOp::Switch => {
            let c = rn(&ops[0]);
            let default = label_of(&ops[1]);
            for pair in ops[2..].chunks(2) {
                a.cmp_imm(c, uimm(&pair[0]) as u32);
                a.branch(0x0, label_of(&pair[1]));
            }
            if ctx.next != Some(default) {
                a.branch(AL, default);
            }
        }
        ThOp::Switch64 => {
            let (lo, hi) = (rn(&ops[0]), rn(&ops[1]));
            let default = label_of(&ops[2]);
            for pair in ops[3..].chunks(2) {
                let v = uimm(&pair[0]);
                let skip = a.new_label();
                a.cmp_imm(lo, v as u32);
                a.branch(0x1, skip);
                a.cmp_imm(hi, (v >> 32) as u32);
                a.branch(0x0, label_of(&pair[1]));
                a.bind(skip);
            }
            if ctx.next != Some(default) {
                a.branch(AL, default);
            }
        }
        ThOp::Udf => a.i(udf(0)),
        ThOp::Svc => a.i(svc(0)),
        ThOp::Dmb => a.i(dmb_sy()),
        ThOp::SubSp => sub_sp(a, uimm(&ops[0]), uimm(&ops[1]) != 0),
        ThOp::AddSp => {
            let k = uimm(&ops[0]);
            if k <= 508 {
                a.i(addsub_sp16(false, k as u32));
            } else {
                a.add_imm(sp, sp, k as i64);
            }
        }
        ThOp::Push => {
            let mask = uimm(&ops[0]) as u32;
            if mask & !0x40ff == 0 {
                a.i(push16(mask & 0xff, mask & (1 << LR) != 0));
            } else {
                a.i(push32(mask));
            }
        }
        ThOp::Pop => {
            let mask = uimm(&ops[0]) as u32;
            if mask & !0x80ff == 0 {
                a.i(pop16(mask & 0xff, mask & (1 << PC) != 0));
            } else {
                a.i(pop32(mask));
            }
        }
    }
}

/// `mvn d, m`.
fn encode_mvn(a: &mut Asm, d: u32, m: u32) {
    if low(d) && low(m) {
        a.i(dp16(15, d, m));
    } else {
        a.i(dp_reg(3, false, d, 15, m, 0, 0));
    }
}

/// Pages up to which a probed `sub sp` is unrolled rather than looped.
const PROBE_UNROLL: u64 = 4;

/// The prologue's `sub sp, sp, #amount`, probed (see the module docs) when
/// `probe` and `amount` is at least [`STACK_PROBE_INTERVAL`].
fn sub_sp(a: &mut Asm, amount: u64, probe: bool) {
    let sp = u32::from(SP);
    let ip = u32::from(IP);
    if !probe || amount < STACK_PROBE_INTERVAL {
        if amount <= 508 {
            a.i(addsub_sp16(true, amount as u32));
        } else {
            a.add_imm(sp, sp, -(amount as i64));
        }
        return;
    }
    let pages = amount / STACK_PROBE_INTERVAL;
    let rem = amount % STACK_PROBE_INTERVAL;
    let page = mod_imm(STACK_PROBE_INTERVAL as u32).expect("4096 is a modified immediate");
    let step = dp_modimm(13, false, sp, sp, page); // sub.w sp, sp, #4096
    let touch = ldst_imm12(false, 4, ip, sp, 0); // str.w ip, [sp]
    if pages <= PROBE_UNROLL {
        for _ in 0..pages {
            a.i(step);
            a.i(touch);
        }
    } else {
        a.mov_imm(ip, pages as u32);
        let top = a.new_label();
        a.bind(top);
        a.i(step);
        a.i(touch);
        a.i(dp_modimm(13, true, ip, ip, 1)); // subs.w ip, ip, #1
        a.branch(0x1, top); // bne
    }
    if rem > 0 {
        a.add_imm(sp, sp, -(rem as i64));
    }
}

// ===========================================================================
// Function + module drivers
// ===========================================================================

/// Encode an allocated, prologue-inserted machine function into bytes and the
/// relocations its external references produced.
pub fn encode_function(
    mf: &MachineFunction,
    layout: &FrameLayout,
    func_name: &dyn Fn(u32) -> String,
    global_name: &dyn Fn(u32) -> String,
) -> Emitted {
    let mut a = Asm { labels: mf.num_blocks(), ..Asm::default() };
    // The entry block first (the function symbol is its start), then the
    // other blocks in arena order.
    let entry = mf.entry().expect("a function being compiled has an entry block");
    let mut order = vec![entry];
    order.extend(mf.block_ids().filter(|&b| b != entry));
    for (k, &bid) in order.iter().enumerate() {
        a.bind(bid.index());
        let ctx = EncodeCtx { layout, func_name, global_name, next: order.get(k + 1).map(|b| b.index()) };
        for inst in &mf.block(bid).insts {
            encode_inst(&mut a, inst, &ctx);
        }
    }
    a.finish()
}

/// Code-generation choices specific to the Thumb backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ThumbOptions {
    /// Use the ARMv7-M `sdiv`/`udiv` instructions (default `true`); without
    /// them 32-bit division calls `__aeabi_idiv` and friends.
    pub hw_div: bool,
}

impl Default for ThumbOptions {
    fn default() -> ThumbOptions {
        ThumbOptions { hw_div: true }
    }
}

impl ThumbOptions {
    /// Use (or avoid) the hardware divide instructions.
    pub fn with_hw_div(mut self, on: bool) -> ThumbOptions {
        self.hw_div = on;
        self
    }
}

/// Run isel → register allocation → frame layout → prologue/epilogue →
/// encoding for one function of a **prepared** module (see
/// [`super::prepare_module`]), returning the code and its stack usage.
pub fn compile_prepared_function(
    module: &Module,
    func: crate::ir::FuncId,
    syms: &StrInterner,
    opts: &CodegenOptions,
    topts: &ThumbOptions,
) -> (Emitted, StackUsage) {
    let target = ThumbTarget::new().with_hw_div(topts.hw_div).with_helpers(Helpers::resolve(module, syms));
    let mut mf = target.select(module, func, syms);
    regalloc::allocate(&mut mf, &target);
    let layout = layout_frame(&mf, &target, opts);
    insert_prologue_epilogue(&mut mf, &layout);
    let func_name = |idx: u32| -> String {
        syms.resolve(module.function(crate::ir::FuncId::from_index(idx as usize)).name).to_owned()
    };
    let global_name = |idx: u32| -> String {
        syms.resolve(module.global(crate::ir::GlobalId::from_index(idx as usize)).name).to_owned()
    };
    let stack = layout.stack_usage(&mf, &func_name);
    (encode_function(&mf, &layout, &func_name, &global_name), stack)
}

/// Compile one function of `module` (not yet prepared) to its encoded bytes
/// and relocations, with the default options.
///
/// # Panics
///
/// If the module cannot be prepared ([`super::prepare_module`]).
pub fn compile_function(module: &Module, func: crate::ir::FuncId, syms: &StrInterner) -> Emitted {
    let (m, s) = super::prepare_module(module, syms, &ThumbOptions::default())
        .unwrap_or_else(|e| panic!("thumb backend: {e}"));
    compile_prepared_function(&m, func, &s, &CodegenOptions::default(), &ThumbOptions::default()).0
}

/// Compile every defined function of `module` into a relocatable
/// [`ObjectModule`] with the default [`CodegenOptions`] (see
/// [`compile_module_with`]).
///
/// # Panics
///
/// As [`compile_module_with`].
pub fn compile_module(module: &Module, syms: &StrInterner) -> ObjectModule {
    compile_module_with(module, syms, &CodegenOptions::default()).object
}

/// Like [`compile_module`], under `opts`, also returning every defined
/// function's [`StackUsage`] (see [`compile_module_thumb`]).
///
/// # Panics
///
/// If `opts` asks for position-independent code (not implemented for this
/// target; [`crate::target::compile_module_for`] reports it as an error), or
/// if the module cannot be prepared ([`super::prepare_module`]).
pub fn compile_module_with(module: &Module, syms: &StrInterner, opts: &CodegenOptions) -> CompiledModule {
    compile_module_thumb(module, syms, opts, &ThumbOptions::default())
}

/// Compile `module` for Thumb-2: prepare it ([`super::prepare_module`]: the
/// ILP32 layout, soft-float lowering and 64-bit legalization, on a copy), then
/// compile every defined function into one `.text` section with one global
/// function symbol per definition (Thumb bit set), a `$t` mapping symbol at
/// the start of `.text`, and the module's data (`R_ARM_ABS32` for 32-bit
/// address fields).
///
/// # Panics
///
/// As [`compile_module_with`].
pub fn compile_module_thumb(
    module: &Module,
    syms: &StrInterner,
    opts: &CodegenOptions,
    topts: &ThumbOptions,
) -> CompiledModule {
    if let Err(e) = crate::target::check_options(crate::target::TargetArch::Thumb, opts) {
        panic!("{e}");
    }
    let (m, s) = super::prepare_module(module, syms, topts).unwrap_or_else(|e| panic!("thumb backend: {e}"));
    let mut obj = ObjectModule::new(module.name.clone());
    let align = opts.function_alignment_for(4, 4) as usize;
    let text = obj.add_section(Section::new(".text", SectionKind::Text, align as u64));
    let mut stack = StackReport::new();
    let mut any = false;
    for (i, f) in m.functions().enumerate() {
        if f.is_declaration() {
            continue;
        }
        if !any {
            // The mapping symbol: the section's code is Thumb from offset 0.
            obj.add_symbol(Symbol::defined("$t", SymbolBinding::Local, SymbolType::NoType, text, 0, 0));
            any = true;
        }
        let fid = crate::ir::FuncId::from_index(i);
        let (emitted, usage) = compile_prepared_function(&m, fid, &s, opts, topts);
        stack.push(usage);
        {
            // Functions start `align`-aligned, at least 4 (padded with a
            // 16-bit `nop`).
            let sec = obj.section_mut(text);
            while !sec.bytes.len().is_multiple_of(align) {
                sec.bytes.extend_from_slice(&0xbf00u16.to_le_bytes());
            }
        }
        let off = obj.section(text).bytes.len() as u64;
        let len = emitted.bytes.len() as u64;
        obj.section_mut(text).bytes.extend_from_slice(&emitted.bytes);
        let name = s.resolve(f.name).to_owned();
        obj.add_symbol(Symbol::defined(name, SymbolBinding::Global, SymbolType::Func, text, off | 1, len));
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
    crate::codegen::data::emit_globals(&m, &s, &mut obj, RelocKind::Abs32);
    crate::codegen::linkage::apply_symbol_attrs(&m, &s, &mut obj);
    CompiledModule { object: obj, stack }
}
