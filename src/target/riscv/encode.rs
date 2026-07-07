//! The RISC-V RV64IM machine-code encoder and the compile entry points.
//!
//! After instruction selection ([`super::isel`]) and register allocation
//! ([`crate::codegen::regalloc`]) a [`MachineFunction`] holds only physical
//! registers and [`RvOp`] opcodes. This module:
//!
//! 1. lays out the stack frame ([`layout_frame`]) — which callee-saved registers
//!    (plus `ra`, when the function calls) the allocation used, the `sp`-relative
//!    offset of every spill/`alloca` slot, and the single frame size that keeps
//!    the stack 16-byte aligned (the RISC-V psABI requires 16-byte `sp`
//!    alignment);
//! 2. splices in the prologue/epilogue as ordinary [`RvOp`] instructions
//!    ([`insert_prologue_epilogue`]) — `addi sp, sp, -frame` + `sd ra`/callee-saved
//!    stores, and the mirror-image epilogue + `ret`;
//! 3. encodes each instruction to a 32-bit little-endian word
//!    ([`encode_function`]) — building each R/I/S/B/U/J bitfield by hand from the
//!    RISC-V ISA manual, resolving intra-function branches through a local
//!    label/fixup table (the B/J immediates are *bit-scrambled* inside the
//!    instruction word, which the generic emitter's whole-field patcher cannot
//!    express, so branch resolution is done here);
//! 4. assembles the functions of a module into an [`ObjectModule`]
//!    ([`compile_module`]).
//!
//! The encoding tables are implemented from the published RISC-V ISA (tenet T1),
//! not copied from any assembler.
//!
//! **Deferred (relocations).** RISC-V direct calls and global addresses want the
//! `R_RISCV_CALL` / `R_RISCV_PCREL_HI20`+`LO12` relocations, whose [`RelocKind`]s
//! are not modeled by the machine-code layer this backend is allowed to touch. A
//! `call` therefore emits a self-relative `auipc`+`jalr` placeholder and a global
//! address an `auipc`+`addi` placeholder, without a relocation; this suffices for
//! the self-contained functions and the MIR interpreter that gate correctness.
//! Wiring the relocations is a documented follow-up (it needs new `RelocKind`s).

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, PReg, Reg, RegClass, StackSlot};
use crate::codegen::regalloc;
use crate::ir::Module;
use crate::mc::emit::Emitted;
use crate::mc::object::{ObjectModule, Section, SectionKind, Symbol, SymbolBinding, SymbolType};
use crate::support::StrInterner;

use super::isel::{RvOp, RiscvTarget};
use super::regs::{RA, SP, T0, T1, T2, T6, ZERO};

// ===========================================================================
// 32-bit instruction-word builders (bitfields from the RISC-V ISA formats)
// ===========================================================================

/// An R-type word `funct7 | rs2 | rs1 | funct3 | rd | opcode`.
#[inline]
pub(crate) fn r_type(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

/// An I-type word `imm[11:0] | rs1 | funct3 | rd | opcode` (`imm` masked to 12 bits).
#[inline]
pub(crate) fn i_type(imm: i32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (((imm as u32) & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

/// An S-type word `imm[11:5] | rs2 | rs1 | funct3 | imm[4:0] | opcode`.
#[inline]
pub(crate) fn s_type(imm: i32, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    let imm = imm as u32;
    (((imm >> 5) & 0x7F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm & 0x1F) << 7)
        | opcode
}

/// The scrambled B-type immediate bits for a (signed, even) branch displacement.
#[inline]
pub(crate) fn b_imm_bits(imm: i32) -> u32 {
    let u = imm as u32;
    (((u >> 12) & 1) << 31)
        | (((u >> 5) & 0x3F) << 25)
        | (((u >> 1) & 0xF) << 8)
        | (((u >> 11) & 1) << 7)
}

/// A B-type word `imm-bits | rs2 | rs1 | funct3 | opcode`.
#[inline]
pub(crate) fn b_type(imm: i32, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    b_imm_bits(imm) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | opcode
}

/// A U-type word `imm[31:12] | rd | opcode` (`imm20` is the raw 20-bit field).
#[inline]
pub(crate) fn u_type(imm20: u32, rd: u32, opcode: u32) -> u32 {
    ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode
}

/// The scrambled J-type immediate bits for a (signed, even) jump displacement.
#[inline]
pub(crate) fn j_imm_bits(imm: i32) -> u32 {
    let u = imm as u32;
    (((u >> 20) & 1) << 31)
        | (((u >> 1) & 0x3FF) << 21)
        | (((u >> 11) & 1) << 20)
        | (((u >> 12) & 0xFF) << 12)
}

/// A J-type word `imm-bits | rd | opcode`.
#[inline]
pub(crate) fn j_type(imm: i32, rd: u32, opcode: u32) -> u32 {
    j_imm_bits(imm) | (rd << 7) | opcode
}

// --- named R-type integer + M-extension ops --------------------------------
pub(crate) fn add(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x00, rs2, rs1, 0x0, rd, 0x33)
}
pub(crate) fn sub(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x20, rs2, rs1, 0x0, rd, 0x33)
}
pub(crate) fn sll(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x00, rs2, rs1, 0x1, rd, 0x33)
}
pub(crate) fn slt(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x00, rs2, rs1, 0x2, rd, 0x33)
}
pub(crate) fn sltu(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x00, rs2, rs1, 0x3, rd, 0x33)
}
pub(crate) fn xor(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x00, rs2, rs1, 0x4, rd, 0x33)
}
pub(crate) fn srl(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x00, rs2, rs1, 0x5, rd, 0x33)
}
pub(crate) fn sra(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x20, rs2, rs1, 0x5, rd, 0x33)
}
pub(crate) fn or(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x00, rs2, rs1, 0x6, rd, 0x33)
}
pub(crate) fn and(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x00, rs2, rs1, 0x7, rd, 0x33)
}
pub(crate) fn mul(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x01, rs2, rs1, 0x0, rd, 0x33)
}
pub(crate) fn mulh(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x01, rs2, rs1, 0x1, rd, 0x33)
}
pub(crate) fn div(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x01, rs2, rs1, 0x4, rd, 0x33)
}
pub(crate) fn divu(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x01, rs2, rs1, 0x5, rd, 0x33)
}
pub(crate) fn rem(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x01, rs2, rs1, 0x6, rd, 0x33)
}
pub(crate) fn remu(rd: u32, rs1: u32, rs2: u32) -> u32 {
    r_type(0x01, rs2, rs1, 0x7, rd, 0x33)
}

// --- named I-type ops ------------------------------------------------------
pub(crate) fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    i_type(imm, rs1, 0x0, rd, 0x13)
}
pub(crate) fn addiw(rd: u32, rs1: u32, imm: i32) -> u32 {
    i_type(imm, rs1, 0x0, rd, 0x1B)
}
pub(crate) fn andi(rd: u32, rs1: u32, imm: i32) -> u32 {
    i_type(imm, rs1, 0x7, rd, 0x13)
}
pub(crate) fn ori(rd: u32, rs1: u32, imm: i32) -> u32 {
    i_type(imm, rs1, 0x6, rd, 0x13)
}
pub(crate) fn xori(rd: u32, rs1: u32, imm: i32) -> u32 {
    i_type(imm, rs1, 0x4, rd, 0x13)
}
pub(crate) fn sltiu(rd: u32, rs1: u32, imm: i32) -> u32 {
    i_type(imm, rs1, 0x3, rd, 0x13)
}
/// `mv rd, rs` is `addi rd, rs, 0`.
pub(crate) fn mv(rd: u32, rs: u32) -> u32 {
    addi(rd, rs, 0)
}
/// `slli rd, rs1, shamt` — RV64 shift-immediate (6-bit `shamt`, funct7 `0000000`).
pub(crate) fn slli(rd: u32, rs1: u32, shamt: u32) -> u32 {
    i_type((shamt & 0x3F) as i32, rs1, 0x1, rd, 0x13)
}
/// `srli rd, rs1, shamt` (logical, funct7 `0000000`).
pub(crate) fn srli(rd: u32, rs1: u32, shamt: u32) -> u32 {
    i_type((shamt & 0x3F) as i32, rs1, 0x5, rd, 0x13)
}
/// `srai rd, rs1, shamt` (arithmetic, funct7 `0100000` ⇒ imm bit 10 set).
pub(crate) fn srai(rd: u32, rs1: u32, shamt: u32) -> u32 {
    i_type(((shamt & 0x3F) | 0x400) as i32, rs1, 0x5, rd, 0x13)
}
/// `jalr rd, rs1, imm`.
pub(crate) fn jalr(rd: u32, rs1: u32, imm: i32) -> u32 {
    i_type(imm, rs1, 0x0, rd, 0x67)
}
/// `ret` is `jalr x0, ra, 0`.
pub(crate) fn ret() -> u32 {
    jalr(ZERO.into(), RA.into(), 0)
}
/// `ebreak` (a trap for `unreachable`).
pub(crate) fn ebreak() -> u32 {
    0x0010_0073
}

/// A load `l{b,h,w,d}{,u} rd, imm(rs1)` for a byte `size` (unsigned for sub-word,
/// matching the interpreter's zero-extending load model). `funct3`: `ld`=011,
/// `lwu`=110, `lhu`=101, `lbu`=100.
pub(crate) fn load(size: u64, rd: u32, rs1: u32, imm: i32) -> u32 {
    let funct3 = match size {
        1 => 0x4, // lbu
        2 => 0x5, // lhu
        4 => 0x6, // lwu
        _ => 0x3, // ld
    };
    i_type(imm, rs1, funct3, rd, 0x03)
}

/// A store `s{b,h,w,d} rs2, imm(rs1)` for a byte `size`.
pub(crate) fn store(size: u64, rs2: u32, rs1: u32, imm: i32) -> u32 {
    let funct3 = match size {
        1 => 0x0, // sb
        2 => 0x1, // sh
        4 => 0x2, // sw
        _ => 0x3, // sd
    };
    s_type(imm, rs2, rs1, funct3, 0x23)
}

// --- named U/J/B ops -------------------------------------------------------
pub(crate) fn lui(rd: u32, imm20: u32) -> u32 {
    u_type(imm20, rd, 0x37)
}
pub(crate) fn auipc(rd: u32, imm20: u32) -> u32 {
    u_type(imm20, rd, 0x17)
}
pub(crate) fn jal(rd: u32, imm: i32) -> u32 {
    j_type(imm, rd, 0x6F)
}
pub(crate) fn beq(rs1: u32, rs2: u32, imm: i32) -> u32 {
    b_type(imm, rs2, rs1, 0x0, 0x63)
}
pub(crate) fn bne(rs1: u32, rs2: u32, imm: i32) -> u32 {
    b_type(imm, rs2, rs1, 0x1, 0x63)
}

// ===========================================================================
// Constant materialization (li)
// ===========================================================================

/// One step of a `li` (load-immediate) materialization sequence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LiStep {
    /// `lui rd, imm` (establishes `rd`).
    Lui(u32),
    /// `addi rd, x0, imm` (establishes `rd` from zero) — only ever the first step.
    AddiZero(i32),
    /// `addiw rd, rd, imm`.
    Addiw(i32),
    /// `addi rd, rd, imm`.
    Addi(i32),
    /// `slli rd, rd, shamt`.
    Slli(u32),
}

/// Sign-extend the low 12 bits of `val`.
#[inline]
fn sext12(val: i64) -> i32 {
    ((val << 52) >> 52) as i32
}

/// The materialization steps for a 64-bit constant into a register.
///
/// A 12-bit constant is a single `addi rd, x0`. A 32-bit constant is
/// `lui`+`addiw` (matching the assembler's `li`). A wider constant is built by
/// recursion — establish the high part, `slli` it up by 12, and add the low 12
/// bits — which is correct (though not always minimal) for the full 64-bit range.
fn li_steps(val: i64) -> Vec<LiStep> {
    // 32-bit fast paths.
    if (i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&val) {
        let lo12 = sext12(val);
        let hi20 = (((val - i64::from(lo12)) >> 12) & 0xFFFFF) as u32;
        let mut steps = Vec::new();
        if hi20 != 0 {
            steps.push(LiStep::Lui(hi20));
            if lo12 != 0 {
                steps.push(LiStep::Addiw(lo12));
            }
        } else {
            steps.push(LiStep::AddiZero(lo12));
        }
        return steps;
    }
    let lo12 = sext12(val);
    let hi = (val - i64::from(lo12)) >> 12;
    let mut steps = li_steps(hi);
    steps.push(LiStep::Slli(12));
    if lo12 != 0 {
        steps.push(LiStep::Addi(lo12));
    }
    steps
}

/// The encoded bytes of a `li rd, val` materialization (for tests).
#[cfg(test)]
pub(crate) fn emit_li_bytes(rd: u32, val: i64) -> Vec<u8> {
    let mut b = RvBuf::new();
    emit_li(&mut b, rd, val);
    b.bytes
}

/// Emit a `li rd, val` materialization into `b`.
fn emit_li(b: &mut RvBuf, rd: u32, val: i64) {
    for step in li_steps(val) {
        match step {
            LiStep::Lui(imm) => b.word(lui(rd, imm)),
            LiStep::AddiZero(imm) => b.word(addi(rd, ZERO.into(), imm)),
            LiStep::Addiw(imm) => b.word(addiw(rd, rd, imm)),
            LiStep::Addi(imm) => b.word(addi(rd, rd, imm)),
            LiStep::Slli(sh) => b.word(slli(rd, rd, sh)),
        }
    }
}

// ===========================================================================
// Frame layout + prologue/epilogue
// ===========================================================================

/// The stack-frame layout of one function, computed after allocation. All slot
/// offsets are `sp`-relative and non-negative: `sp` is fixed for the whole body
/// (no dynamic stack growth), so `off(sp)` addressing is stable.
#[derive(Clone, Debug)]
pub struct FrameLayout {
    /// `sp`-relative byte offset of each stack slot (by slot index).
    slot_off: Vec<u64>,
    /// The registers saved across the body: `ra` (when the function calls) then
    /// the callee-saved registers the allocation used.
    saved: Vec<PReg>,
    /// `sp`-relative byte offset each saved register is stored at.
    saved_off: Vec<u64>,
    /// The total frame size subtracted from `sp` (16-byte aligned).
    size: u64,
}

/// Round `value` up to a multiple of `align` (a power of two ≥ 1).
fn align_up(value: u64, align: u64) -> u64 {
    let a = align.max(1);
    value.div_ceil(a) * a
}

/// Compute the frame layout of an allocated machine function.
pub fn layout_frame(mf: &MachineFunction, target: &RiscvTarget) -> FrameLayout {
    use crate::codegen::target::MachineTarget;
    let callee: Vec<PReg> = target.callee_saved().to_vec();

    // Which callee-saved registers does the allocation actually define, and does
    // the function contain a call (so `ra` must be preserved)?
    let mut used = [false; 32];
    let mut has_call = false;
    for bid in mf.block_ids() {
        for inst in &mf.block(bid).insts {
            if RvOp::decode(inst.opcode) == RvOp::Call {
                has_call = true;
            }
            for d in inst.defs() {
                if let Reg::Physical(p) = d
                    && p.class == RegClass::Gpr
                {
                    used[p.num as usize] = true;
                }
            }
        }
    }

    let mut saved: Vec<PReg> = Vec::new();
    if has_call {
        saved.push(super::regs::gpr(RA));
    }
    saved.extend(callee.into_iter().filter(|p| used[p.num as usize]));

    // Saved registers live at the bottom of the frame, 8 bytes each.
    let saved_off: Vec<u64> = (0..saved.len()).map(|i| (i * 8) as u64).collect();
    let mut off = (saved.len() * 8) as u64;

    // Local slots (spills/allocas) sit above the saved region, each 8-aligned.
    let mut slot_off = vec![0u64; mf.frame().len()];
    for (i, off_slot) in slot_off.iter_mut().enumerate() {
        let info = mf.frame().slot(StackSlot::from_index(i));
        let align = info.align.max(8);
        off = align_up(off, align);
        *off_slot = off;
        off += align_up(info.size.max(1), 8);
    }
    let size = align_up(off, 16);

    FrameLayout { slot_off, saved, saved_off, size }
}

fn def_preg(r: PReg) -> MachineOperand {
    MachineOperand::Def(Reg::Physical(r))
}
fn use_preg(r: PReg) -> MachineOperand {
    MachineOperand::Use(Reg::Physical(r))
}
fn imm_op(v: i64) -> MachineOperand {
    MachineOperand::Imm(puremp::Int::from_i64(v))
}

/// Splice the prologue into the entry block and an epilogue before every `ret`.
pub fn insert_prologue_epilogue(mf: &mut MachineFunction, layout: &FrameLayout) {
    let entry = mf.entry().expect("a function being compiled has an entry block");

    // --- prologue: addi sp, sp, -size; sd ra/cs, off(sp) ---
    let mut prologue = Vec::new();
    if layout.size > 0 {
        prologue.push(MachineInst::new(RvOp::AddiSp.opcode(), vec![imm_op(-(layout.size as i64))]));
    }
    for (&r, &off) in layout.saved.iter().zip(&layout.saved_off) {
        prologue.push(MachineInst::new(RvOp::SaveReg.opcode(), vec![use_preg(r), imm_op(off as i64)]));
    }
    let old = std::mem::take(&mut mf.block_mut(entry).insts);
    prologue.extend(old);
    mf.block_mut(entry).insts = prologue;

    // --- epilogue before each Ret: ld ra/cs; addi sp, sp, +size ---
    let block_ids: Vec<_> = mf.block_ids().collect();
    for bid in block_ids {
        let old = std::mem::take(&mut mf.block_mut(bid).insts);
        let mut new_insts = Vec::with_capacity(old.len());
        for inst in old {
            if RvOp::decode(inst.opcode) == RvOp::Ret {
                for (&r, &off) in layout.saved.iter().zip(&layout.saved_off) {
                    new_insts.push(MachineInst::new(
                        RvOp::RestoreReg.opcode(),
                        vec![def_preg(r), imm_op(off as i64)],
                    ));
                }
                if layout.size > 0 {
                    new_insts.push(MachineInst::new(
                        RvOp::AddiSp.opcode(),
                        vec![imm_op(layout.size as i64)],
                    ));
                }
            }
            new_insts.push(inst);
        }
        mf.block_mut(bid).insts = new_insts;
    }
}

// ===========================================================================
// Instruction encoding
// ===========================================================================

fn rnum(op: &MachineOperand) -> u32 {
    match op {
        MachineOperand::Def(Reg::Physical(p)) | MachineOperand::Use(Reg::Physical(p)) => {
            u32::from(p.num)
        }
        other => panic!("expected a physical register operand, found {other:?}"),
    }
}

fn simm(op: &MachineOperand) -> i64 {
    match op {
        MachineOperand::Imm(v) => v.to_i64().or_else(|| v.to_u64().map(|u| u as i64)).unwrap_or(0),
        other => panic!("expected an immediate operand, found {other:?}"),
    }
}

fn slot_index(op: &MachineOperand) -> usize {
    match op {
        MachineOperand::Frame(s) => s.index(),
        other => panic!("expected a frame operand, found {other:?}"),
    }
}

fn label_index(op: &MachineOperand) -> usize {
    match op {
        MachineOperand::Label(b) => b.index(),
        other => panic!("expected a label operand, found {other:?}"),
    }
}

/// A pending intra-function branch fixup: the byte offset of the instruction
/// word, the target block index, and which immediate field to patch.
#[derive(Clone, Copy, Debug)]
struct Fixup {
    at: u64,
    block: usize,
    kind: FixupKind,
}

#[derive(Clone, Copy, Debug)]
enum FixupKind {
    /// A B-type conditional-branch immediate (13-bit signed, scrambled).
    BType,
    /// A J-type jump immediate (21-bit signed, scrambled).
    JType,
}

/// The little-endian 32-bit-word buffer with a branch-fixup table.
struct RvBuf {
    bytes: Vec<u8>,
    fixups: Vec<Fixup>,
}

impl RvBuf {
    fn new() -> RvBuf {
        RvBuf { bytes: Vec::new(), fixups: Vec::new() }
    }

    #[inline]
    fn offset(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Append one 32-bit instruction word.
    #[inline]
    fn word(&mut self, w: u32) {
        self.bytes.extend_from_slice(&w.to_le_bytes());
    }

    /// Append a branch word and record a fixup to `block`.
    fn branch(&mut self, w: u32, block: usize, kind: FixupKind) {
        self.fixups.push(Fixup { at: self.offset(), block, kind });
        self.word(w);
    }

    /// Resolve every branch fixup against the final block offsets.
    fn resolve(&mut self, block_off: &[u64]) {
        for fx in &self.fixups {
            let target = block_off[fx.block] as i64;
            let disp = (target - fx.at as i64) as i32;
            let at = fx.at as usize;
            let mut w = u32::from_le_bytes(self.bytes[at..at + 4].try_into().unwrap());
            match fx.kind {
                FixupKind::BType => {
                    debug_assert!((-(1 << 12)..(1 << 12)).contains(&disp), "B-branch out of range");
                    w = (w & !b_imm_mask()) | b_imm_bits(disp);
                }
                FixupKind::JType => {
                    debug_assert!((-(1 << 20)..(1 << 20)).contains(&disp), "J-jump out of range");
                    w = (w & !j_imm_mask()) | j_imm_bits(disp);
                }
            }
            self.bytes[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
}

/// The set of bits a B-type immediate occupies.
#[inline]
fn b_imm_mask() -> u32 {
    (1 << 31) | (0x3F << 25) | (0xF << 8) | (1 << 7)
}
/// The set of bits a J-type immediate occupies.
#[inline]
fn j_imm_mask() -> u32 {
    (1 << 31) | (0x3FF << 21) | (1 << 20) | (0xFF << 12)
}

/// What the encoder needs to resolve frame offsets while emitting.
struct EncodeCtx<'a> {
    layout: &'a FrameLayout,
}

/// Whether a signed byte offset fits the 12-bit immediate of a load/store/`addi`.
#[inline]
fn fits12(off: i64) -> bool {
    (-2048..=2047).contains(&off)
}

/// Emit `sd`/`ld reg, off(sp)`, materializing a large `off` through `t6`.
fn frame_mem(b: &mut RvBuf, is_load: bool, reg: u32, off: i64) {
    if fits12(off) {
        if is_load {
            b.word(load(8, reg, SP.into(), off as i32));
        } else {
            b.word(store(8, reg, SP.into(), off as i32));
        }
    } else {
        emit_li(b, T6.into(), off);
        b.word(add(T6.into(), SP.into(), T6.into()));
        if is_load {
            b.word(load(8, reg, T6.into(), 0));
        } else {
            b.word(store(8, reg, T6.into(), 0));
        }
    }
}

/// Encode one machine instruction into `b`.
fn encode_inst(b: &mut RvBuf, inst: &MachineInst, ctx: &EncodeCtx<'_>) {
    let ops = &inst.operands;
    let rr = |b: &mut RvBuf, f: fn(u32, u32, u32) -> u32| {
        b.word(f(rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
    };
    match RvOp::decode(inst.opcode) {
        RvOp::Mv => {
            let d = rnum(&ops[0]);
            let s = rnum(&ops[1]);
            if d != s {
                b.word(mv(d, s));
            }
        }
        RvOp::Li => emit_li(b, rnum(&ops[0]), simm(&ops[1])),
        RvOp::Add => rr(b, add),
        RvOp::Sub => rr(b, sub),
        RvOp::And => rr(b, and),
        RvOp::Or => rr(b, or),
        RvOp::Xor => rr(b, xor),
        RvOp::Mul => rr(b, mul),
        RvOp::Mulh => rr(b, mulh),
        RvOp::Div => rr(b, div),
        RvOp::Divu => rr(b, divu),
        RvOp::Rem => rr(b, rem),
        RvOp::Remu => rr(b, remu),
        RvOp::Sll => rr(b, sll),
        RvOp::Srl => rr(b, srl),
        RvOp::Sra => rr(b, sra),
        RvOp::Addi => b.word(addi(rnum(&ops[0]), rnum(&ops[1]), simm(&ops[2]) as i32)),
        RvOp::Andi => b.word(andi(rnum(&ops[0]), rnum(&ops[1]), simm(&ops[2]) as i32)),
        RvOp::Ori => b.word(ori(rnum(&ops[0]), rnum(&ops[1]), simm(&ops[2]) as i32)),
        RvOp::Xori => b.word(xori(rnum(&ops[0]), rnum(&ops[1]), simm(&ops[2]) as i32)),
        RvOp::Slli => b.word(slli(rnum(&ops[0]), rnum(&ops[1]), simm(&ops[2]) as u32)),
        RvOp::Srli => b.word(srli(rnum(&ops[0]), rnum(&ops[1]), simm(&ops[2]) as u32)),
        RvOp::Srai => b.word(srai(rnum(&ops[0]), rnum(&ops[1]), simm(&ops[2]) as u32)),
        RvOp::SetCmp => encode_setcmp(b, ops),
        RvOp::Select => encode_select(b, ops),
        RvOp::Load => {
            let d = rnum(&ops[0]);
            let ptr = rnum(&ops[1]);
            let size = simm(&ops[2]) as u64;
            b.word(load(size, d, ptr, 0));
        }
        RvOp::Store => {
            let ptr = rnum(&ops[0]);
            let val = rnum(&ops[1]);
            let size = simm(&ops[2]) as u64;
            b.word(store(size, val, ptr, 0));
        }
        RvOp::FrameAddr => {
            let d = rnum(&ops[0]);
            let off = ctx.layout.slot_off[slot_index(&ops[1])] as i64;
            if fits12(off) {
                b.word(addi(d, SP.into(), off as i32));
            } else {
                emit_li(b, T6.into(), off);
                b.word(add(d, SP.into(), T6.into()));
            }
        }
        RvOp::StoreFrame => {
            let off = ctx.layout.slot_off[slot_index(&ops[1])] as i64;
            frame_mem(b, false, rnum(&ops[0]), off);
        }
        RvOp::LoadFrame => {
            let off = ctx.layout.slot_off[slot_index(&ops[1])] as i64;
            frame_mem(b, true, rnum(&ops[0]), off);
        }
        RvOp::SaveReg => frame_mem(b, false, rnum(&ops[0]), simm(&ops[1])),
        RvOp::RestoreReg => frame_mem(b, true, rnum(&ops[0]), simm(&ops[1])),
        RvOp::AddiSp => {
            let delta = simm(&ops[0]);
            if fits12(delta) {
                b.word(addi(SP.into(), SP.into(), delta as i32));
            } else {
                emit_li(b, T6.into(), delta);
                b.word(add(SP.into(), SP.into(), T6.into()));
            }
        }
        RvOp::GlobalAddr => {
            // Placeholder (relocations deferred): auipc d, 0; addi d, d, 0.
            let d = rnum(&ops[0]);
            b.word(auipc(d, 0));
            b.word(addi(d, d, 0));
        }
        RvOp::Call => match &ops[0] {
            // Placeholder self-relative call (relocations deferred).
            MachineOperand::Func(_) => {
                b.word(auipc(RA.into(), 0));
                b.word(jalr(RA.into(), RA.into(), 0));
            }
            MachineOperand::Use(Reg::Physical(p)) => {
                b.word(jalr(RA.into(), u32::from(p.num), 0));
            }
            other => panic!("Call expects a Func or register operand, found {other:?}"),
        },
        RvOp::Ret => b.word(ret()),
        RvOp::J => {
            let t = label_index(&ops[0]);
            b.branch(jal(ZERO.into(), 0), t, FixupKind::JType);
        }
        RvOp::BrCond => {
            let cond = rnum(&ops[0]);
            let t = label_index(&ops[1]);
            let f = label_index(&ops[2]);
            b.branch(bne(cond, ZERO.into(), 0), t, FixupKind::BType); // bnez cond, t
            b.branch(jal(ZERO.into(), 0), f, FixupKind::JType); // j f
        }
        RvOp::Switch => {
            let cond = rnum(&ops[0]);
            let default = label_index(&ops[1]);
            let mut i = 2;
            while i + 1 < ops.len() {
                let value = simm(&ops[i]);
                let case = label_index(&ops[i + 1]);
                emit_li(b, T2.into(), value); // materialize case value into t2
                b.branch(beq(cond, T2.into(), 0), case, FixupKind::BType); // beq cond, t2, case
                i += 2;
            }
            b.branch(jal(ZERO.into(), 0), default, FixupKind::JType);
        }
        RvOp::Unreachable => b.word(ebreak()),
    }
}

/// Encode a `SetCmp` (`[Def d, Use a, Use b, Imm pred, Imm width]`) into a
/// `slt`/`sltu` idiom, using `t0` as scratch for the `eq`/`ne` subtract.
fn encode_setcmp(b: &mut RvBuf, ops: &[MachineOperand]) {
    let d = rnum(&ops[0]);
    let a = rnum(&ops[1]);
    let bb = rnum(&ops[2]);
    let pred = simm(&ops[3]) as u8;
    let t0 = u32::from(T0);
    match pred {
        0 => {
            // Eq: sub t0, a, b; sltiu d, t0, 1  (seqz)
            b.word(sub(t0, a, bb));
            b.word(sltiu(d, t0, 1));
        }
        1 => {
            // Ne: sub t0, a, b; sltu d, x0, t0  (snez)
            b.word(sub(t0, a, bb));
            b.word(sltu(d, ZERO.into(), t0));
        }
        2 => b.word(sltu(d, a, bb)),                       // Ult
        3 => {
            b.word(sltu(d, bb, a)); // Ule = !(b<a)
            b.word(xori(d, d, 1));
        }
        4 => b.word(sltu(d, bb, a)),                       // Ugt
        5 => {
            b.word(sltu(d, a, bb)); // Uge = !(a<b)
            b.word(xori(d, d, 1));
        }
        6 => b.word(slt(d, a, bb)),                        // Slt
        7 => {
            b.word(slt(d, bb, a)); // Sle = !(b<a)
            b.word(xori(d, d, 1));
        }
        8 => b.word(slt(d, bb, a)),                        // Sgt
        _ => {
            b.word(slt(d, a, bb)); // Sge = !(a<b)
            b.word(xori(d, d, 1));
        }
    }
}

/// Encode a `Select` (`[Def d, Use c, Use t, Use f]`) as a branchless mask blend
/// `d = (c ? -1 : 0)`-masked, using `t0`/`t1` as scratch so `d` is written last
/// (safe even when `d` aliases a source register after allocation).
fn encode_select(b: &mut RvBuf, ops: &[MachineOperand]) {
    let d = rnum(&ops[0]);
    let c = rnum(&ops[1]);
    let t = rnum(&ops[2]);
    let f = rnum(&ops[3]);
    let (t0, t1) = (u32::from(T0), u32::from(T1));
    b.word(sub(t0, ZERO.into(), c)); // t0 = -c  (0 or all-ones; c is 0/1)
    b.word(xori(t1, t0, -1)); // t1 = ~mask
    b.word(and(t0, t, t0)); // t0 = t & mask
    b.word(and(t1, f, t1)); // t1 = f & ~mask
    b.word(or(d, t0, t1)); // d = result
}

// ===========================================================================
// Function + module drivers
// ===========================================================================

/// Encode an allocated, prologue-inserted machine function into bytes.
pub fn encode_function(mf: &MachineFunction, layout: &FrameLayout) -> Emitted {
    let mut b = RvBuf::new();
    let ctx = EncodeCtx { layout };

    // Emit the entry block first (so the function symbol at offset 0 is the
    // entry), then the remaining blocks in arena order.
    let entry = mf.entry().expect("a function being compiled has an entry block");
    let mut order = vec![entry];
    for bid in mf.block_ids() {
        if bid != entry {
            order.push(bid);
        }
    }
    let mut block_off = vec![0u64; mf.num_blocks()];
    for &bid in &order {
        block_off[bid.index()] = b.offset();
        for inst in &mf.block(bid).insts {
            encode_inst(&mut b, inst, &ctx);
        }
    }
    b.resolve(&block_off);
    Emitted { bytes: b.bytes, relocations: Vec::new() }
}

/// Compile one function of `module` to its encoded bytes. Runs isel → register
/// allocation → frame layout → prologue/epilogue → encoding.
pub fn compile_function(module: &Module, func: crate::ir::FuncId) -> Emitted {
    let target = RiscvTarget::new();
    let mut mf = target.select(module, func);
    regalloc::allocate(&mut mf, &target);
    let layout = layout_frame(&mf, &target);
    insert_prologue_epilogue(&mut mf, &layout);
    encode_function(&mf, &layout)
}

/// Compile every defined function of `module` into a relocatable
/// [`ObjectModule`]: a single `.text` section with one global function symbol per
/// definition. Call/global relocations are deferred (see the module docs), so a
/// module whose functions are not self-contained links only after that follow-up.
/// `syms` resolves the interned function names.
pub fn compile_module(module: &Module, syms: &StrInterner) -> ObjectModule {
    let mut obj = ObjectModule::new(module.name.clone());
    let text = obj.add_section(Section::new(".text", SectionKind::Text, 4));

    for (i, f) in module.functions().enumerate() {
        if f.is_declaration() {
            continue;
        }
        let fid = crate::ir::FuncId::from_index(i);
        let emitted = compile_function(module, fid);
        // 4-align this function's start within .text (RV instructions are words).
        {
            let sec = obj.section_mut(text);
            while !sec.bytes.len().is_multiple_of(4) {
                sec.bytes.push(0);
            }
        }
        let off = obj.section(text).bytes.len() as u64;
        let len = emitted.bytes.len() as u64;
        obj.section_mut(text).bytes.extend_from_slice(&emitted.bytes);

        let name = syms.resolve(f.name).to_owned();
        obj.add_symbol(Symbol::defined(
            name,
            SymbolBinding::Global,
            SymbolType::Func,
            text,
            off,
            len,
        ));
    }
    obj
}
