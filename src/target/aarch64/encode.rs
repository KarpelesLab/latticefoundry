//! The AArch64 (A64) machine-code encoder and the compile entry points (ROADMAP
//! Phase 7).
//!
//! After instruction selection ([`super::isel`]) and register allocation
//! ([`crate::codegen::regalloc`]) a [`MachineFunction`] holds only physical
//! registers and [`A64Op`] opcodes. This module:
//!
//! 1. lays out the stack frame ([`layout_frame`]) — which callee-saved registers
//!    the allocation used, the `sp`-relative offset of every spill/`alloca` slot,
//!    and the single `sub sp` amount that keeps the stack 16-byte aligned
//!    (AAPCS64 requires 16-byte `sp` alignment);
//! 2. splices in the prologue/epilogue as ordinary [`A64Op`] instructions
//!    ([`insert_prologue_epilogue`]) — `stp x29,x30,[sp,#-16]!` / `mov x29,sp` /
//!    `sub sp` / callee-saved stores, and the mirror-image epilogue + `ret`;
//! 3. encodes each instruction to a fixed **32-bit little-endian word**
//!    ([`encode_function`]) — building each bitfield by hand from the ARM A64
//!    encoding rules, resolving intra-function branches through a local
//!    label/fixup table (A64 branch immediates are bitfields *inside* the
//!    instruction word, which the generic [`crate::mc::emit::Emitter`]'s
//!    whole-field patcher cannot express, so branch resolution is done here) and
//!    turning `bl`/global references into relocations;
//! 4. assembles the functions of a module into an [`ObjectModule`]
//!    ([`compile_module`]); [`compile_module_with`] also takes
//!    [`CodegenOptions`] and returns each function's [`StackUsage`], read off the
//!    same [`FrameLayout`] the prologue is built from.
//!
//! **Large frames and stack probes.** Stack-pointer adjustments and `sp`-relative
//! slot offsets of any size are encoded (`#imm12, lsl #12` + `#imm12` pairs up
//! to 16 MiB, beyond that through `x16`). With probes on (the default, see
//! [`crate::codegen::stack`]), a `sub sp` of at least [`STACK_PROBE_INTERVAL`]
//! bytes is emitted as
//!
//! ```text
//! sub sp, sp, #1, lsl #12 ; str xzr, [sp]      // × pages, when pages <= 4
//!
//! mov x16, #pages                               // otherwise, a counted loop
//! L: sub sp, sp, #1, lsl #12 ; str xzr, [sp] ; subs x16, x16, #1 ; b.ne L
//! sub sp, sp, #remainder                        // < 4096, if nonzero
//! ```
//!
//! `x16` (IP0) is volatile, never allocated, and holds nothing live inside a
//! function body or at its entry, so it serves as this module's
//! large-offset/loop scratch.
//!
//! **Dynamic stack allocation.** A `dyn_alloca` ([`A64Op::DynAlloca`]) moves
//! `sp` at run time: probed, one page at a time, like the prologue (see
//! `encode_dyn_alloca`). Its function addresses every frame slot from `x29`
//! instead (`x16 = x29 - (extra - off)`, as `x29 - extra` is where `sp` stood
//! after the prologue), keeps the outgoing-argument area at the new bottom of
//! the stack, and restores `sp` from `x29` before its epilogue.
//!
//! **Position-independent code.** Under [`RelocModel::Pic`] / `Pie`
//! ([`CodegenOptions::reloc_model`]), the address of a global or function that
//! may be preempted is loaded from its GOT entry (`adrp`+`ldr`,
//! `R_AARCH64_ADR_GOT_PAGE` + `R_AARCH64_LD64_GOT_LO12_NC`); a locally bound
//! one (internal, hidden, or any definition in a PIE) is formed directly
//! (`adrp`+`add`). Calls stay `bl` (`R_AARCH64_CALL26`: the linker adds a PLT
//! entry for a preemptible callee). With [`compile_module_debug_with`] the
//! object also carries DWARF line tables.
//!
//! [`RelocModel::Pic`]: crate::codegen::RelocModel::Pic
//!
//! The encoding tables are implemented from the published ARM A64 instruction
//! encodings (tenet T1), not copied from any assembler.

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, PReg, Reg, RegClass, StackSlot};
use crate::codegen::options::{CodegenOptions, CompiledModule};
use crate::codegen::stack::{STACK_PROBE_INTERVAL, StackReport, StackUsage, scan_calls};
use crate::codegen::regalloc;
use crate::ir::Module;
use crate::mc::emit::{Emitted, EmittedReloc};
use crate::mc::object::{
    ObjectModule, RelocKind, Section, SectionKind, Symbol, SymbolBinding, SymbolType,
};
use crate::support::StrInterner;

use super::isel::neon::NeonOp;
use super::isel::{A64Op, AArch64Target, NeonLegality};
use super::regs::{FP, LR, SP, XZR};

// ===========================================================================
// 32-bit instruction-word builders (bitfields from the ARM A64 encodings)
// ===========================================================================

/// The `sf` bit (bit 31) selects the 64-bit (`X`) form: a `width` of 32 or less
/// uses the 32-bit (`W`) form, anything wider (including odd widths such as
/// `i48`, which need the upper word) the 64-bit form.
#[inline]
pub(crate) fn sf_of(width: u32) -> u32 {
    u32::from(width > 32)
}

/// A data-processing (shifted register) form `op Rd, Rn, Rm` (shift=LSL #0):
/// `add`/`sub`/`and`/`orr`/`eor`/`subs` share this shape and differ in `base`.
#[inline]
pub(crate) fn dp_reg(base: u32, sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    base | (sf << 31) | (rm << 16) | (rn << 5) | rd
}

/// `add`/`sub` with the shifted-register bases.
pub(crate) fn add_reg(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_reg(0x0B00_0000, sf, rd, rn, rm)
}
pub(crate) fn sub_reg(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_reg(0x4B00_0000, sf, rd, rn, rm)
}
pub(crate) fn and_reg(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_reg(0x0A00_0000, sf, rd, rn, rm)
}
pub(crate) fn orr_reg(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_reg(0x2A00_0000, sf, rd, rn, rm)
}
pub(crate) fn eor_reg(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_reg(0x4A00_0000, sf, rd, rn, rm)
}
/// `subs Rd, Rn, Rm` (sets flags); `cmp` is `subs xzr, Rn, Rm`.
pub(crate) fn subs_reg(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_reg(0x6B00_0000, sf, rd, rn, rm)
}
/// `mov Rd, Rm` (`orr Rd, xzr, Rm`).
pub(crate) fn mov_reg(sf: u32, rd: u32, rm: u32) -> u32 {
    orr_reg(sf, rd, XZR.into(), rm)
}

/// A data-processing (immediate) add/sub form `op Rd, Rn, #imm12` (no shift).
#[inline]
pub(crate) fn addsub_imm(base: u32, sf: u32, rd: u32, rn: u32, imm12: u32) -> u32 {
    base | (sf << 31) | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd
}
pub(crate) fn add_imm(sf: u32, rd: u32, rn: u32, imm12: u32) -> u32 {
    addsub_imm(0x1100_0000, sf, rd, rn, imm12)
}
pub(crate) fn sub_imm(sf: u32, rd: u32, rn: u32, imm12: u32) -> u32 {
    addsub_imm(0x5100_0000, sf, rd, rn, imm12)
}

/// A move-wide-immediate form `movz`/`movk`/`movn Rd, #imm16, lsl #(16*hw)`.
#[inline]
pub(crate) fn mov_wide(base: u32, sf: u32, rd: u32, imm16: u32, hw: u32) -> u32 {
    base | (sf << 31) | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd
}
pub(crate) fn movz(sf: u32, rd: u32, imm16: u32, hw: u32) -> u32 {
    mov_wide(0x5280_0000, sf, rd, imm16, hw)
}
pub(crate) fn movk(sf: u32, rd: u32, imm16: u32, hw: u32) -> u32 {
    mov_wide(0x7280_0000, sf, rd, imm16, hw)
}
pub(crate) fn movn(sf: u32, rd: u32, imm16: u32, hw: u32) -> u32 {
    mov_wide(0x1280_0000, sf, rd, imm16, hw)
}

/// A data-processing (2-source) form; `base` carries the `op2` selector.
#[inline]
fn dp_2src(base: u32, sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    base | (sf << 31) | (rm << 16) | (rn << 5) | rd
}
pub(crate) fn udiv(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_2src(0x1AC0_0800, sf, rd, rn, rm)
}
pub(crate) fn sdiv(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_2src(0x1AC0_0C00, sf, rd, rn, rm)
}
pub(crate) fn lslv(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_2src(0x1AC0_2000, sf, rd, rn, rm)
}
pub(crate) fn lsrv(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_2src(0x1AC0_2400, sf, rd, rn, rm)
}
pub(crate) fn asrv(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_2src(0x1AC0_2800, sf, rd, rn, rm)
}

/// A data-processing (3-source) form `madd`/`msub Rd, Rn, Rm, Ra` (`o0` in base).
#[inline]
fn dp_3src(base: u32, sf: u32, rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
    base | (sf << 31) | (rm << 16) | (ra << 10) | (rn << 5) | rd
}
/// `madd Rd, Rn, Rm, Ra` = `Ra + Rn*Rm`; `mul` is `madd Rd, Rn, Rm, xzr`.
pub(crate) fn madd(sf: u32, rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
    dp_3src(0x1B00_0000, sf, rd, rn, rm, ra)
}
/// `msub Rd, Rn, Rm, Ra` = `Ra - Rn*Rm`.
pub(crate) fn msub(sf: u32, rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
    dp_3src(0x1B00_8000, sf, rd, rn, rm, ra)
}

/// A bitfield-move form (`UBFM`/`SBFM`), the basis of the shift-immediate aliases.
#[inline]
fn bfm(base: u32, sf: u32, rd: u32, rn: u32, immr: u32, imms: u32) -> u32 {
    // The `N` bit (bit 22) always equals `sf` for the 32-/64-bit forms.
    base | (sf << 31) | (sf << 22) | ((immr & 0x3F) << 16) | ((imms & 0x3F) << 10) | (rn << 5) | rd
}
/// `sbfx Rd, Rn, #0, #width` (`SBFM Rd, Rn, #0, #(width-1)`): sign-extend the
/// low `width` bits. With `sf = 1` and a `width` of 8/16/32 this is `sxtb`/
/// `sxth`/`sxtw Xd, Wn`.
pub(crate) fn sbfx0(sf: u32, rd: u32, rn: u32, width: u32) -> u32 {
    bfm(0x1300_0000, sf, rd, rn, 0, width - 1)
}
/// `ubfx Rd, Rn, #0, #width` (`UBFM Rd, Rn, #0, #(width-1)`): zero-extend the
/// low `width` bits.
pub(crate) fn ubfx0(sf: u32, rd: u32, rn: u32, width: u32) -> u32 {
    bfm(0x5300_0000, sf, rd, rn, 0, width - 1)
}
/// `lsl Rd, Rn, #shift` (`UBFM Rd, Rn, #(-shift MOD w), #(w-1-shift)`).
pub(crate) fn lsl_imm(sf: u32, rd: u32, rn: u32, shift: u32) -> u32 {
    let w = if sf == 1 { 64 } else { 32 };
    let immr = (w - shift % w) % w;
    let imms = w - 1 - shift;
    bfm(0x5300_0000, sf, rd, rn, immr, imms)
}
/// `lsr Rd, Rn, #shift` (`UBFM Rd, Rn, #shift, #(w-1)`).
pub(crate) fn lsr_imm(sf: u32, rd: u32, rn: u32, shift: u32) -> u32 {
    let w = if sf == 1 { 64 } else { 32 };
    bfm(0x5300_0000, sf, rd, rn, shift, w - 1)
}
/// `asr Rd, Rn, #shift` (`SBFM Rd, Rn, #shift, #(w-1)`).
pub(crate) fn asr_imm(sf: u32, rd: u32, rn: u32, shift: u32) -> u32 {
    let w = if sf == 1 { 64 } else { 32 };
    bfm(0x1300_0000, sf, rd, rn, shift, w - 1)
}

/// An unsigned-offset load/store `ldr`/`str Rt, [Rn, #(imm12*scale)]`. `size` is
/// the log2 of the access width (0=byte, 1=half, 2=word, 3=dword); `load` picks
/// the load vs store opcode.
#[inline]
pub(crate) fn ldst_uimm(load: bool, size: u32, rt: u32, rn: u32, imm12: u32) -> u32 {
    let base = if load { 0x3940_0000 } else { 0x3900_0000 };
    base | (size << 30) | ((imm12 & 0xFFF) << 10) | (rn << 5) | rt
}

// --- load/store exclusive, acquire/release, barriers (atomics) -----------

/// `ldxr`/`ldaxr Rt, [Rn]` (`size` = log2 bytes; the `b`/`h` forms for 0/1,
/// `W`/`X` for 2/3). `acquire` selects the load-acquire form.
pub(crate) fn ldxr(size: u32, acquire: bool, rt: u32, rn: u32) -> u32 {
    0x085F_7C00 | (size << 30) | (u32::from(acquire) << 15) | (rn << 5) | rt
}
/// `stxr`/`stlxr Ws, Rt, [Rn]`: store-exclusive, `Ws` = 0 on success.
/// `release` selects the store-release form.
pub(crate) fn stxr(size: u32, release: bool, rs: u32, rt: u32, rn: u32) -> u32 {
    0x0800_7C00 | (size << 30) | (u32::from(release) << 15) | (rs << 16) | (rn << 5) | rt
}
/// `ldar Rt, [Rn]` (load-acquire).
pub(crate) fn ldar(size: u32, rt: u32, rn: u32) -> u32 {
    0x08DF_FC00 | (size << 30) | (rn << 5) | rt
}
/// `stlr Rt, [Rn]` (store-release).
pub(crate) fn stlr(size: u32, rt: u32, rn: u32) -> u32 {
    0x089F_FC00 | (size << 30) | (rn << 5) | rt
}
/// `dmb <option>` with the 4-bit `CRm` option (`0b1011` ish, `0b1001` ishld).
pub(crate) fn dmb(crm: u32) -> u32 {
    0xD503_30BF | ((crm & 0xF) << 8)
}
/// `subs Rd, Rn, Rm, <extend>` (extended-register form; `option` 0 `uxtb`,
/// 1 `uxth`, 4 `sxtb`, 5 `sxth`): compares `Rn` with the extended low bits of
/// `Rm`.
pub(crate) fn subs_ext(sf: u32, rd: u32, rn: u32, rm: u32, option: u32) -> u32 {
    0x6B20_0000 | (sf << 31) | (rm << 16) | (option << 13) | (rn << 5) | rd
}
/// `orn Rd, Rn, Rm` (`mvn Rd, Rm` when `Rn` is the zero register).
pub(crate) fn orn_reg(sf: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    dp_reg(0x2A20_0000, sf, rd, rn, rm)
}

/// The intra-procedure scratch registers, never allocated: the atomic loops use
/// `x16`/IP0 for the new value and `w17`/IP1 for the store-exclusive status, and
/// large-offset addressing and the stack-probe loop use IP0. None of these
/// sequences overlap, so nothing is live in them across another (the linker
/// only uses them in call veneers).
const IP0: u32 = 16;
const IP1: u32 = 17;

/// Expand [`A64Op::AtomicRmw`] into an exclusive-monitor retry loop (see the
/// opcode docs). Arithmetic runs in the `W`/`X` form of the access width; the
/// exclusive store writes only the low `size` bytes, so garbage above a narrow
/// width is harmless, except for the signed/unsigned compares of `max`/`min`,
/// which extend at the width (`sbfx`, or the `sxt`/`uxt` compare forms).
fn encode_atomic_rmw(b: &mut A64Buf, ops: &[MachineOperand]) {
    use crate::ir::inst::RmwOp;
    let (d, ptr, val) = (rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]));
    let bytes = uimm(&ops[3]);
    let op = RmwOp::from_code(uimm(&ops[4])).expect("AtomicRmw carries a valid rmw code");
    let acqrel = uimm(&ops[5]);
    let (size, sf) = (ldst_size(bytes), u32::from(bytes == 8));
    let zr = u32::from(XZR);
    let top = b.offset();
    b.word(ldxr(size, acqrel & 1 != 0, d, ptr));
    let data = match op {
        RmwOp::Xchg => val,
        RmwOp::Add => {
            b.word(add_reg(sf, IP0, d, val));
            IP0
        }
        RmwOp::Sub => {
            b.word(sub_reg(sf, IP0, d, val));
            IP0
        }
        RmwOp::And => {
            b.word(and_reg(sf, IP0, d, val));
            IP0
        }
        RmwOp::Nand => {
            b.word(and_reg(sf, IP0, d, val));
            b.word(orn_reg(sf, IP0, zr, IP0));
            IP0
        }
        RmwOp::Or => {
            b.word(orr_reg(sf, IP0, d, val));
            IP0
        }
        RmwOp::Xor => {
            b.word(eor_reg(sf, IP0, d, val));
            IP0
        }
        RmwOp::Max | RmwOp::Min | RmwOp::UMax | RmwOp::UMin => {
            let signed = matches!(op, RmwOp::Max | RmwOp::Min);
            // Compare old with val at the access width.
            if bytes < 4 {
                let (lhs, option) = if signed {
                    // The exclusive load zero-extended `old`; sign-extend a copy.
                    b.word(sbfx0(0, IP0, d, 8 * bytes as u32));
                    (IP0, if bytes == 1 { 4 } else { 5 })
                } else {
                    (d, if bytes == 1 { 0 } else { 1 })
                };
                b.word(subs_ext(0, zr, lhs, val, option));
            } else {
                b.word(subs_reg(sf, zr, d, val));
            }
            // Keep `old` when it already wins, else take `val`.
            let cond = match op {
                RmwOp::Max => 0xC,  // GT
                RmwOp::Min => 0xB,  // LT
                RmwOp::UMax => 0x8, // HI
                _ => 0x3,           // LO
            };
            b.word(csel(sf, IP0, d, val, cond));
            IP0
        }
    };
    b.word(stxr(size, acqrel & 2 != 0, IP1, data, ptr));
    let back = (top as i64 - b.offset() as i64) / 4;
    b.word(cbz(0, IP1, back as i32, true));
}

/// Expand [`A64Op::CmpXchg`] into a strong compare-and-exchange loop (see the
/// opcode docs). The exclusive load zero-extends `old`, and the compare
/// zero-extends `expected` at the access width (`uxtb`/`uxth`), so garbage
/// above a narrow `expected` does not cause a spurious failure.
fn encode_cmpxchg(b: &mut A64Buf, ops: &[MachineOperand]) {
    let (d, ptr, expected, new) = (rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), rnum(&ops[3]));
    let bytes = uimm(&ops[4]);
    let acqrel = uimm(&ops[5]);
    let (size, sf) = (ldst_size(bytes), u32::from(bytes == 8));
    let zr = u32::from(XZR);
    let top = b.offset();
    b.word(ldxr(size, acqrel & 1 != 0, d, ptr));
    match bytes {
        1 => b.word(subs_ext(0, zr, d, expected, 0)),
        2 => b.word(subs_ext(0, zr, d, expected, 1)),
        _ => b.word(subs_reg(sf, zr, d, expected)),
    }
    // b.ne done: skip this branch, the store and the retry (3 words).
    b.word(b_cond(0x1, 3));
    b.word(stxr(size, acqrel & 2 != 0, IP1, new, ptr));
    let back = (top as i64 - b.offset() as i64) / 4;
    b.word(cbz(0, IP1, back as i32, true));
}

/// Test hook: the encoding of one allocated [`A64Op::AtomicRmw`] or
/// [`A64Op::CmpXchg`] (its whole loop).
#[cfg(test)]
pub(crate) fn encode_atomic_loop_for_test(inst: &MachineInst) -> Vec<u8> {
    let mut b = A64Buf::new();
    match A64Op::decode(inst.opcode) {
        A64Op::AtomicRmw => encode_atomic_rmw(&mut b, &inst.operands),
        A64Op::CmpXchg => encode_cmpxchg(&mut b, &inst.operands),
        other => panic!("not an atomic loop: {other:?}"),
    }
    b.bytes
}

/// `csel Rd, Rn, Rm, cond`.
pub(crate) fn csel(sf: u32, rd: u32, rn: u32, rm: u32, cond: u32) -> u32 {
    0x1A80_0000 | (sf << 31) | (rm << 16) | (cond << 12) | (rn << 5) | rd
}
/// `cset Rd, cond` (`csinc Rd, xzr, xzr, invert(cond)`).
pub(crate) fn cset(sf: u32, rd: u32, cond: u32) -> u32 {
    0x1A80_0400 | (sf << 31) | (u32::from(XZR) << 16) | ((cond ^ 1) << 12) | (u32::from(XZR) << 5) | rd
}

/// `b`/`bl` with a 26-bit immediate (word-scaled displacement `>>2`).
pub(crate) fn b_uncond(imm26: i32) -> u32 {
    0x1400_0000 | ((imm26 as u32) & 0x03FF_FFFF)
}
pub(crate) fn bl(imm26: i32) -> u32 {
    0x9400_0000 | ((imm26 as u32) & 0x03FF_FFFF)
}
/// `b.cond` with a 19-bit immediate.
pub(crate) fn b_cond(cond: u32, imm19: i32) -> u32 {
    0x5400_0000 | (((imm19 as u32) & 0x7FFFF) << 5) | cond
}
/// `cbz`/`cbnz Rt, #imm19`.
pub(crate) fn cbz(sf: u32, rt: u32, imm19: i32, nonzero: bool) -> u32 {
    let base = if nonzero { 0x3500_0000 } else { 0x3400_0000 };
    base | (sf << 31) | (((imm19 as u32) & 0x7FFFF) << 5) | rt
}
/// `blr Rn` / `br Rn` / `ret Rn`.
pub(crate) fn blr(rn: u32) -> u32 {
    0xD63F_0000 | (rn << 5)
}
pub(crate) fn ret(rn: u32) -> u32 {
    0xD65F_0000 | (rn << 5)
}
/// `svc #imm16` (supervisor call; Linux uses `svc #0`).
pub(crate) fn svc(imm16: u32) -> u32 {
    0xD400_0001 | ((imm16 & 0xFFFF) << 5)
}
/// `brk #imm16`.
pub(crate) fn brk(imm16: u32) -> u32 {
    0xD420_0000 | ((imm16 & 0xFFFF) << 5)
}
/// `adrp Rd, <page>` with the immediate zeroed (a relocation fills it).
pub(crate) fn adrp(rd: u32) -> u32 {
    0x9000_0000 | rd
}

/// A `stp`/`ldp Rt, Rt2, [Rn, ...]` (64-bit) with signed `imm7` (offset `>>3`);
/// `base` carries the pre-/post-/signed-offset selector and the load bit.
#[inline]
fn ldstp(base: u32, rt: u32, rt2: u32, rn: u32, imm7: i32) -> u32 {
    base | (((imm7 as u32) & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt
}
/// `stp Rt, Rt2, [sp, #imm]!` (pre-index).
pub(crate) fn stp_pre(rt: u32, rt2: u32, rn: u32, imm7: i32) -> u32 {
    ldstp(0xA980_0000, rt, rt2, rn, imm7)
}
/// `ldp Rt, Rt2, [sp], #imm` (post-index).
pub(crate) fn ldp_post(rt: u32, rt2: u32, rn: u32, imm7: i32) -> u32 {
    ldstp(0xA8C0_0000, rt, rt2, rn, imm7)
}

// ---------------------------------------------------------------------------
// Scalar floating-point (FP/SIMD) instruction words (from the ARM A64 encodings)
// ---------------------------------------------------------------------------

/// The FP "ptype" field: `0` = single (`s`/f32), `1` = double (`d`/f64). It sits
/// in bits `[23:22]` of most FP instructions.
#[inline]
fn ptype_bits(ptype: u32) -> u32 {
    ptype << 22
}

/// Floating-point data-processing (2 source) `op Vd, Vn, Vm`. `opcode` (bits
/// `[15:12]`): 0=`fmul`, 1=`fdiv`, 2=`fadd`, 3=`fsub`.
#[inline]
pub(crate) fn fp_dp2(ptype: u32, opcode: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    0x1E20_0800 | ptype_bits(ptype) | (rm << 16) | (opcode << 12) | (rn << 5) | rd
}
pub(crate) fn fadd(ptype: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    fp_dp2(ptype, 0b0010, rd, rn, rm)
}
pub(crate) fn fsub(ptype: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    fp_dp2(ptype, 0b0011, rd, rn, rm)
}
pub(crate) fn fmul(ptype: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    fp_dp2(ptype, 0b0000, rd, rn, rm)
}
pub(crate) fn fdiv(ptype: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    fp_dp2(ptype, 0b0001, rd, rn, rm)
}

/// Floating-point data-processing (1 source) `op Vd, Vn`. `opcode` (bits
/// `[20:15]`): 0=`fmov`, 2=`fneg`, `fcvt`→single=4/double=5/half=7.
#[inline]
pub(crate) fn fp_dp1(ptype: u32, opcode: u32, rd: u32, rn: u32) -> u32 {
    0x1E20_4000 | ptype_bits(ptype) | (opcode << 15) | (rn << 5) | rd
}
/// `fmov Vd, Vn` (register move within the FP file). Register copies now use
/// the full-width `mov Vd.16b` ([`simd_mov`]); kept for the encoding tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn fmov_reg(ptype: u32, rd: u32, rn: u32) -> u32 {
    fp_dp1(ptype, 0b000000, rd, rn)
}
/// `fneg Vd, Vn`.
pub(crate) fn fneg(ptype: u32, rd: u32, rn: u32) -> u32 {
    fp_dp1(ptype, 0b000010, rd, rn)
}
/// `fcvt Vd, Vn` between precisions; `src`/`dst` are ptypes (0=s,1=d).
pub(crate) fn fcvt(src: u32, dst: u32, rd: u32, rn: u32) -> u32 {
    // The FCVT opcode is `0001` concatenated with the destination type opc
    // (single=00, double=01, half=11).
    let opc = match dst {
        1 => 0b01, // to double
        3 => 0b11, // to half (unused here)
        _ => 0b00, // to single
    };
    fp_dp1(src, 0b000100 | opc, rd, rn)
}

/// Floating-point compare `fcmp Vn, Vm` (sets NZCV; opcode2 = 00000).
pub(crate) fn fcmp(ptype: u32, rn: u32, rm: u32) -> u32 {
    0x1E20_2000 | ptype_bits(ptype) | (rm << 16) | (rn << 5)
}

/// Conversion between floating-point and integer `op Rd, Rn`. `sf` selects the
/// 64-bit gpr; `rmode`/`opcode` select the operation (see the ARM ARM):
/// `fcvtzs`=(11,000), `fcvtzu`=(11,001), `scvtf`=(00,010), `ucvtf`=(00,011),
/// `fmov` gpr↔fp = (00,110)/(00,111).
#[inline]
pub(crate) fn fp_int_cvt(sf: u32, ptype: u32, rmode: u32, opcode: u32, rd: u32, rn: u32) -> u32 {
    0x1E20_0000 | (sf << 31) | ptype_bits(ptype) | (rmode << 19) | (opcode << 16) | (rn << 5) | rd
}
/// `fcvtzs Rd(gpr), Vn` — float→signed int, round toward zero.
pub(crate) fn fcvtzs(sf: u32, ptype: u32, rd: u32, rn: u32) -> u32 {
    fp_int_cvt(sf, ptype, 0b11, 0b000, rd, rn)
}
/// `fcvtzu Rd(gpr), Vn` — float→unsigned int, round toward zero.
pub(crate) fn fcvtzu(sf: u32, ptype: u32, rd: u32, rn: u32) -> u32 {
    fp_int_cvt(sf, ptype, 0b11, 0b001, rd, rn)
}
/// `scvtf Vd, Rn(gpr)` — signed int→float.
pub(crate) fn scvtf(sf: u32, ptype: u32, rd: u32, rn: u32) -> u32 {
    fp_int_cvt(sf, ptype, 0b00, 0b010, rd, rn)
}
/// `ucvtf Vd, Rn(gpr)` — unsigned int→float.
pub(crate) fn ucvtf(sf: u32, ptype: u32, rd: u32, rn: u32) -> u32 {
    fp_int_cvt(sf, ptype, 0b00, 0b011, rd, rn)
}
/// `fmov Vd, Rn(gpr)` — move gpr bit pattern into the low FP lane.
pub(crate) fn fmov_from_gpr(sf: u32, ptype: u32, rd: u32, rn: u32) -> u32 {
    fp_int_cvt(sf, ptype, 0b00, 0b111, rd, rn)
}

/// An FP unsigned-offset load/store `ldr`/`str Vt, [Rn, #(imm12*scale)]`. `size`
/// is the log2 access width (2=`s`/word, 3=`d`/dword).
#[inline]
pub(crate) fn fp_ldst_uimm(load: bool, size: u32, rt: u32, rn: u32, imm12: u32) -> u32 {
    let base = if load { 0x3D40_0000 } else { 0x3D00_0000 };
    base | (size << 30) | ((imm12 & 0xFFF) << 10) | (rn << 5) | rt
}

// ===========================================================================
// Frame layout + prologue/epilogue
// ===========================================================================

/// The stack-frame layout of one function, computed after allocation. All slot
/// offsets are `sp`-relative and non-negative: `sp` is fixed for the whole body
/// of a function without `dyn_alloca`, so `[sp, #off]` addressing is stable
/// (a function with one reaches the same bytes from `x29`; see `dynamic`).
#[derive(Clone, Debug)]
pub struct FrameLayout {
    /// `sp`-relative byte offset of each stack slot (by slot index).
    slot_off: Vec<u32>,
    /// The callee-saved registers the allocation used (class-tagged, so the
    /// prologue saves GPRs and FP registers with the right `str`/`ldr` form).
    cs_regs: Vec<PReg>,
    /// `sp`-relative byte offset each callee-saved register is stored at.
    cs_off: Vec<u32>,
    /// The `sub sp` amount below the fp/lr save (16-byte aligned).
    extra: u32,
    /// The outgoing stack-argument area at the bottom of the frame (bytes).
    outgoing: u64,
    /// Whether the prologue's `sub sp` (and every `DynAlloca`) emits stack
    /// probes.
    probes: bool,
    /// Whether the function moves `sp` at run time (it has a `DynAlloca`): its
    /// slots are then addressed from `x29` (`x29 - extra + off`), and its
    /// epilogue first puts `sp` back at `x29 - extra`.
    dynamic: bool,
}

impl FrameLayout {
    /// The stack usage this layout gives `mf` (whose MIR supplies the call
    /// information; `func_name` resolves a function index — the callees and `mf`
    /// itself — to its symbol name): the 16-byte `x29`/`x30` pair pushed by
    /// `stp ..., [sp, #-16]!` plus the `sub sp` amount — exactly what the
    /// prologue built from this layout moves `sp` by. `bl` pushes nothing.
    pub fn stack_usage(
        &self,
        mf: &MachineFunction,
        func_name: &dyn Fn(u32) -> String,
    ) -> StackUsage {
        let scan = scan_calls(
            mf,
            A64Op::Call.opcode(),
            A64Op::Svc.opcode(),
            Some(A64Op::DynAlloca.opcode()),
        );
        let extra = u64::from(self.extra);
        StackUsage {
            name: func_name(mf.info().source),
            frame_size: 16 + extra,
            return_address: 0,
            saved_registers: 16 + 8 * self.cs_regs.len() as u64,
            sp_adjust: extra,
            outgoing_args: self.outgoing,
            dynamic_alloca: scan.dynamic_alloca,
            direct_callees: scan.direct.iter().map(|&f| func_name(f)).collect(),
            indirect_calls: scan.indirect,
            syscalls: scan.syscalls,
            probed: self.probes,
        }
    }
}

/// Round `value` up to a multiple of `align` (a power of two ≥ 1).
fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

/// Compute the frame layout of an allocated machine function, with the default
/// [`CodegenOptions`] (stack probes on).
pub fn layout_frame(mf: &MachineFunction, target: &AArch64Target) -> FrameLayout {
    layout_frame_with(mf, target, &CodegenOptions::default())
}

/// Compute the frame layout of an allocated machine function under `opts`.
pub fn layout_frame_with(
    mf: &MachineFunction,
    target: &AArch64Target,
    opts: &CodegenOptions,
) -> FrameLayout {
    use crate::codegen::target::MachineTarget;
    let callee: Vec<PReg> = target.callee_saved().to_vec();

    // Which callee-saved registers does the allocation actually define? Track by
    // (class, number): the GPR `x8` and the FP `v8` share the number 8, so a
    // per-number-only set would confuse the two files.
    let mut used_gpr = [false; 32];
    let mut used_fp = [false; 32];
    for bid in mf.block_ids() {
        for inst in &mf.block(bid).insts {
            for d in inst.defs() {
                if let Reg::Physical(p) = d {
                    let set = match p.class {
                        RegClass::Gpr => &mut used_gpr,
                        RegClass::Fp => &mut used_fp,
                    };
                    set[p.num as usize] = true;
                }
            }
        }
    }
    let is_used = |p: &PReg| match p.class {
        RegClass::Gpr => used_gpr[p.num as usize],
        RegClass::Fp => used_fp[p.num as usize],
    };
    let cs_regs: Vec<PReg> = callee.into_iter().filter(is_used).collect();
    // The outgoing stack-argument area sits at the very bottom of the frame
    // (`[sp .. sp + outgoing)`), addressed `add d, sp, #off` (`A64Op::LeaSpOff`).
    // `sp` is constant after the prologue. Zero unless a call passed arguments on
    // the stack, so scalar/FP functions are unaffected.
    let outgoing = align_up(mf.frame().outgoing(), 16);
    // Callee-saved live just above the outgoing area, 8 bytes each.
    let cs_off: Vec<u32> = (0..cs_regs.len()).map(|i| (outgoing + (i * 8) as u64) as u32).collect();
    let cs_bytes = (cs_regs.len() * 8) as u64;

    // Local slots (spills/allocas) sit above the callee-saved region. Over-align
    // every slot to 8 bytes so scaled `ldr`/`str [sp,#off]` addressing is valid.
    let mut off = outgoing + cs_bytes;
    let mut slot_off = vec![0u32; mf.frame().len()];
    for (i, off_slot) in slot_off.iter_mut().enumerate() {
        let info = mf.frame().slot(StackSlot::from_index(i));
        let align = info.align.max(8);
        off = align_up(off, align);
        *off_slot = off as u32;
        off += align_up(info.size.max(1), 8);
    }
    let extra = align_up(off, 16) as u32;
    let dynamic = mf
        .block_ids()
        .any(|b| mf.block(b).insts.iter().any(|i| A64Op::decode(i.opcode) == A64Op::DynAlloca));

    FrameLayout { slot_off, cs_regs, cs_off, extra, outgoing, probes: opts.stack_probes, dynamic }
}

fn def_preg(r: PReg) -> MachineOperand {
    MachineOperand::Def(Reg::Physical(r))
}
fn use_preg(r: PReg) -> MachineOperand {
    MachineOperand::Use(Reg::Physical(r))
}
fn imm_op(v: u64) -> MachineOperand {
    MachineOperand::Imm(puremp::Int::from_u64(v))
}

/// Splice the prologue into the entry block and an epilogue before every `ret`.
pub fn insert_prologue_epilogue(mf: &mut MachineFunction, layout: &FrameLayout) {
    let entry = mf.entry().expect("a function being compiled has an entry block");

    // --- prologue: stp fp,lr,[sp,#-16]!; mov fp,sp; sub sp,#extra; save cs ---
    let mut prologue = vec![
        MachineInst::new(A64Op::StpFpLr.opcode(), Vec::new()),
        MachineInst::new(A64Op::MovFpSp.opcode(), Vec::new()),
    ];
    if layout.extra > 0 {
        // The second operand requests the probed form.
        prologue.push(MachineInst::new(
            A64Op::SubSp.opcode(),
            vec![imm_op(u64::from(layout.extra)), imm_op(u64::from(layout.probes))],
        ));
    }
    for (&cs, &off) in layout.cs_regs.iter().zip(&layout.cs_off) {
        prologue.push(MachineInst::new(
            A64Op::SaveReg.opcode(),
            vec![use_preg(cs), imm_op(u64::from(off))],
        ));
    }
    let old = std::mem::take(&mut mf.block_mut(entry).insts);
    prologue.extend(old);
    mf.block_mut(entry).insts = prologue;

    // --- epilogue before each Ret: restore cs; add sp,#extra; ldp fp,lr ---
    let block_ids: Vec<_> = mf.block_ids().collect();
    for bid in block_ids {
        let old = std::mem::take(&mut mf.block_mut(bid).insts);
        let mut new_insts = Vec::with_capacity(old.len());
        for inst in old {
            if A64Op::decode(inst.opcode) == A64Op::Ret {
                if layout.dynamic {
                    // `sp` moved below the fixed frame: put it back first.
                    new_insts.push(MachineInst::new(
                        A64Op::SpFromFp.opcode(),
                        vec![imm_op(u64::from(layout.extra))],
                    ));
                }
                for (&cs, &off) in layout.cs_regs.iter().zip(&layout.cs_off) {
                    new_insts.push(MachineInst::new(
                        A64Op::RestoreReg.opcode(),
                        vec![def_preg(cs), imm_op(u64::from(off))],
                    ));
                }
                if layout.extra > 0 {
                    new_insts.push(MachineInst::new(
                        A64Op::AddSp.opcode(),
                        vec![imm_op(u64::from(layout.extra))],
                    ));
                }
                new_insts.push(MachineInst::new(A64Op::LdpFpLr.opcode(), Vec::new()));
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

/// The register class of a physical register operand.
fn rclass(op: &MachineOperand) -> RegClass {
    match op {
        MachineOperand::Def(Reg::Physical(p)) | MachineOperand::Use(Reg::Physical(p)) => p.class,
        other => panic!("expected a physical register operand, found {other:?}"),
    }
}

fn uimm(op: &MachineOperand) -> u64 {
    match op {
        MachineOperand::Imm(v) => v.to_u64().or_else(|| v.to_i64().map(|i| i as u64)).unwrap_or(0),
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

/// The width of a spill/reload access as an `ldst` `size` field (always dword).
const SIZE_DWORD: u32 = 3;

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
    /// `b`/`bl` `imm26` in bits `[25:0]`.
    Imm26,
    /// `b.cond`/`cbz`/`cbnz` `imm19` in bits `[23:5]`.
    Imm19,
}

/// The little-endian 32-bit-word buffer with a branch/relocation fixup table.
struct A64Buf {
    bytes: Vec<u8>,
    fixups: Vec<Fixup>,
    relocs: Vec<EmittedReloc>,
}

impl A64Buf {
    fn new() -> A64Buf {
        A64Buf { bytes: Vec::new(), fixups: Vec::new(), relocs: Vec::new() }
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

    /// Append a word and record a relocation against `symbol` at its offset.
    fn reloc(&mut self, w: u32, symbol: String, kind: RelocKind) {
        self.relocs.push(EmittedReloc { offset: self.offset(), symbol, kind, addend: 0 });
        self.word(w);
    }

    /// Resolve every branch fixup against the final block offsets.
    fn resolve(&mut self, block_off: &[u64]) {
        for fx in &self.fixups {
            let target = block_off[fx.block] as i64;
            let disp = target - fx.at as i64;
            debug_assert_eq!(disp % 4, 0, "A64 branch displacement must be word-aligned");
            let imm = (disp / 4) as i32;
            let at = fx.at as usize;
            let mut w = u32::from_le_bytes(self.bytes[at..at + 4].try_into().unwrap());
            match fx.kind {
                FixupKind::Imm26 => {
                    debug_assert!((-(1 << 25)..(1 << 25)).contains(&imm), "b/bl out of range");
                    w = (w & !0x03FF_FFFF) | ((imm as u32) & 0x03FF_FFFF);
                }
                FixupKind::Imm19 => {
                    debug_assert!((-(1 << 18)..(1 << 18)).contains(&imm), "cond branch out of range");
                    w = (w & !(0x7FFFF << 5)) | (((imm as u32) & 0x7FFFF) << 5);
                }
            }
            self.bytes[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
}

/// What the encoder needs to resolve non-local references while emitting.
struct EncodeCtx<'a> {
    layout: &'a FrameLayout,
    func_name: &'a dyn Fn(u32) -> String,
    global_name: &'a dyn Fn(u32) -> String,
    got: GotQuery<'a>,
}

/// Which symbols position-independent code reaches through the GOT: those
/// that may be preempted (see [`crate::codegen::linkage`]), by function and
/// global index.
#[derive(Clone, Copy)]
struct GotQuery<'a> {
    func: &'a dyn Fn(u32) -> bool,
    global: &'a dyn Fn(u32) -> bool,
}

impl GotQuery<'static> {
    /// Position-dependent code: no symbol goes through the GOT.
    const NONE: GotQuery<'static> = GotQuery { func: &|_| false, global: &|_| false };
}

/// The address of symbol `sym` into `d`: `adrp d, sym; add d, d, :lo12:sym`
/// (`R_AARCH64_ADR_PREL_PG_HI21` + `R_AARCH64_ADD_ABS_LO12_NC`), or with
/// `via_got` the load of its GOT entry, `adrp d, :got:sym; ldr d, [d,
/// :got_lo12:sym]` (`R_AARCH64_ADR_GOT_PAGE` + `R_AARCH64_LD64_GOT_LO12_NC`).
fn symbol_addr(b: &mut A64Buf, d: u32, sym: String, via_got: bool) {
    if via_got {
        b.reloc(adrp(d), sym.clone(), RelocKind::Aarch64AdrGotPage);
        b.reloc(ldst_uimm(true, SIZE_DWORD, d, d, 0), sym, RelocKind::Aarch64Ld64GotLo12Nc);
    } else {
        b.reloc(adrp(d), sym.clone(), RelocKind::Aarch64AdrPrelPgHi21);
        b.reloc(add_imm(1, d, d, 0), sym, RelocKind::Aarch64AddAbsLo12Nc);
    }
}

/// The `ldst` `size` field (log2 access width) for a byte count.
fn ldst_size(bytes: u64) -> u32 {
    match bytes {
        1 => 0,
        2 => 1,
        4 => 2,
        _ => 3,
    }
}


// ===========================================================================
// Advanced SIMD (NEON), 128-bit (`Q = 1`) forms
// ===========================================================================

/// The `size` field for `esize`-bit lanes (8→0, 16→1, 32→2, 64→3).
fn simd_size(esize: u32) -> u32 {
    match esize {
        8 => 0,
        16 => 1,
        32 => 2,
        _ => 3,
    }
}

/// AdvSIMD three-same: `0 1 U 01110 size 1 Rm opcode 1 Rn Rd`.
fn simd3(u: u32, size: u32, opcode: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    0x4E20_0400 | (u << 29) | (size << 22) | (rm << 16) | (opcode << 11) | (rn << 5) | rd
}

/// AdvSIMD two-register miscellaneous: `0 1 U 01110 size 10000 opcode 10 Rn Rd`.
fn simd2(u: u32, size: u32, opcode: u32, rd: u32, rn: u32) -> u32 {
    0x4E20_0800 | (u << 29) | (size << 22) | (opcode << 12) | (rn << 5) | rd
}

/// AdvSIMD across lanes: `0 1 U 01110 size 11000 opcode 10 Rn Rd`.
fn simd_across(u: u32, size: u32, opcode: u32, rd: u32, rn: u32) -> u32 {
    0x4E30_0800 | (u << 29) | (size << 22) | (opcode << 12) | (rn << 5) | rd
}

/// The `imm5` of the copy instructions (`dup`/`umov`/`ins`) for lane `lane`
/// of `esize`-bit lanes: the lowest set bit marks the size, the index above it.
fn simd_imm5(esize: u32, lane: u32) -> u32 {
    match esize {
        8 => (lane << 1) | 1,
        16 => (lane << 2) | 2,
        32 => (lane << 3) | 4,
        _ => (lane << 4) | 8,
    }
}

/// `mov Vd.16b, Vn.16b` (`orr Vd.16b, Vn.16b, Vn.16b`): a full 128-bit copy.
pub(crate) fn simd_mov(rd: u32, rn: u32) -> u32 {
    0x4EA0_1C00 | (rn << 16) | (rn << 5) | rd
}

/// The word of a three-register NEON op on `esize`-bit lanes.
pub(crate) fn neon3(op: NeonOp, esize: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    let sz = simd_size(esize);
    let fsz = u32::from(esize == 64); // the float `sz` bit
    match op {
        NeonOp::Add => simd3(0, sz, 0b10000, rd, rn, rm),
        NeonOp::Sub => simd3(1, sz, 0b10000, rd, rn, rm),
        NeonOp::Mul => simd3(0, sz, 0b10011, rd, rn, rm),
        NeonOp::And => simd3(0, 0b00, 0b00011, rd, rn, rm),
        NeonOp::Bic => simd3(0, 0b01, 0b00011, rd, rn, rm),
        NeonOp::Orr => simd3(0, 0b10, 0b00011, rd, rn, rm),
        NeonOp::Eor => simd3(1, 0b00, 0b00011, rd, rn, rm),
        NeonOp::Cmeq => simd3(1, sz, 0b10001, rd, rn, rm),
        NeonOp::Cmgt => simd3(0, sz, 0b00110, rd, rn, rm),
        NeonOp::Cmge => simd3(0, sz, 0b00111, rd, rn, rm),
        NeonOp::Cmhi => simd3(1, sz, 0b00110, rd, rn, rm),
        NeonOp::Cmhs => simd3(1, sz, 0b00111, rd, rn, rm),
        NeonOp::Sshl => simd3(0, sz, 0b01000, rd, rn, rm),
        NeonOp::Ushl => simd3(1, sz, 0b01000, rd, rn, rm),
        NeonOp::Smax => simd3(0, sz, 0b01100, rd, rn, rm),
        NeonOp::Smin => simd3(0, sz, 0b01101, rd, rn, rm),
        NeonOp::Umax => simd3(1, sz, 0b01100, rd, rn, rm),
        NeonOp::Umin => simd3(1, sz, 0b01101, rd, rn, rm),
        NeonOp::Sqadd => simd3(0, sz, 0b00001, rd, rn, rm),
        NeonOp::Uqadd => simd3(1, sz, 0b00001, rd, rn, rm),
        NeonOp::Sqsub => simd3(0, sz, 0b00101, rd, rn, rm),
        NeonOp::Uqsub => simd3(1, sz, 0b00101, rd, rn, rm),
        NeonOp::Fadd => simd3(0, fsz, 0b11010, rd, rn, rm),
        NeonOp::Fsub => simd3(0, 0b10 | fsz, 0b11010, rd, rn, rm),
        NeonOp::Fmul => simd3(1, fsz, 0b11011, rd, rn, rm),
        NeonOp::Fdiv => simd3(1, fsz, 0b11111, rd, rn, rm),
        NeonOp::Fcmeq => simd3(0, fsz, 0b11100, rd, rn, rm),
        NeonOp::Fcmge => simd3(1, fsz, 0b11100, rd, rn, rm),
        NeonOp::Fcmgt => simd3(1, 0b10 | fsz, 0b11100, rd, rn, rm),
        // tbl Vd.16b, {Vn.16b}, Vm.16b: 0 1 001110 000 Rm 0 00 0 00 Rn Rd.
        NeonOp::Tbl => 0x4E00_0000 | (rm << 16) | (rn << 5) | rd,
        other => panic!("{other:?} is not a three-register NEON op"),
    }
}

/// The word of a two-register (or across-lane) NEON op on `esize`-bit lanes.
pub(crate) fn neon2(op: NeonOp, esize: u32, rd: u32, rn: u32) -> u32 {
    let sz = simd_size(esize);
    let fsz = u32::from(esize == 64);
    match op {
        NeonOp::Neg => simd2(1, sz, 0b01011, rd, rn),
        NeonOp::Not => simd2(1, 0b00, 0b00101, rd, rn),
        NeonOp::Fneg => simd2(1, 0b10 | fsz, 0b01111, rd, rn),
        NeonOp::Scvtf => simd2(0, fsz, 0b11101, rd, rn),
        NeonOp::Ucvtf => simd2(1, fsz, 0b11101, rd, rn),
        NeonOp::Fcvtzs => simd2(0, 0b10 | fsz, 0b11011, rd, rn),
        NeonOp::Fcvtzu => simd2(1, 0b10 | fsz, 0b11011, rd, rn),
        NeonOp::Addv => simd_across(0, sz, 0b11011, rd, rn),
        NeonOp::Smaxv => simd_across(0, sz, 0b01010, rd, rn),
        NeonOp::Sminv => simd_across(0, sz, 0b11010, rd, rn),
        NeonOp::Umaxv => simd_across(1, sz, 0b01010, rd, rn),
        NeonOp::Uminv => simd_across(1, sz, 0b11010, rd, rn),
        // addp Dd, Vn.2d.
        NeonOp::Addp => 0x5EF1_B800 | (rn << 5) | rd,
        other => panic!("{other:?} is not a two-register NEON op"),
    }
}

/// The word of a NEON shift by immediate: `shl` (`immh:immb = esize + amt`),
/// `ushr`/`sshr` (`immh:immb = 2*esize - amt`, `amt` in `1..=esize`).
pub(crate) fn neon_shift(op: NeonOp, esize: u32, rd: u32, rn: u32, amt: u32) -> u32 {
    let (u, opcode, immhb) = match op {
        NeonOp::Shl => (0, 0b01010, esize + amt),
        NeonOp::Ushr => (1, 0b00000, 2 * esize - amt),
        NeonOp::Sshr => (0, 0b00000, 2 * esize - amt),
        other => panic!("{other:?} is not a NEON immediate shift"),
    };
    0x4F00_0400 | (u << 29) | (immhb << 16) | (opcode << 11) | (rn << 5) | rd
}

/// `dup Vd.T, Rn` (general).
pub(crate) fn neon_dup(esize: u32, rd: u32, rn: u32) -> u32 {
    0x4E00_0C00 | (simd_imm5(esize, 0) << 16) | (rn << 5) | rd
}

/// `dup Vd.T, Vn.T[lane]` (element).
pub(crate) fn neon_dup_lane(esize: u32, lane: u32, rd: u32, rn: u32) -> u32 {
    0x4E00_0400 | (simd_imm5(esize, lane) << 16) | (rn << 5) | rd
}

/// `umov Wd/Xd, Vn.T[lane]` (the `X` form for 64-bit lanes).
pub(crate) fn neon_umov(esize: u32, lane: u32, rd: u32, rn: u32) -> u32 {
    0x0E00_3C00 | (u32::from(esize == 64) << 30) | (simd_imm5(esize, lane) << 16) | (rn << 5) | rd
}

/// `ins Vd.T[lane], Rn` (general).
pub(crate) fn neon_ins_gpr(esize: u32, lane: u32, rd: u32, rn: u32) -> u32 {
    0x4E00_1C00 | (simd_imm5(esize, lane) << 16) | (rn << 5) | rd
}

/// `ins Vd.T[lane], Vn.T[0]` (element).
pub(crate) fn neon_ins_elem(esize: u32, lane: u32, rd: u32, rn: u32) -> u32 {
    0x6E00_0400 | (simd_imm5(esize, lane) << 16) | (rn << 5) | rd
}

/// `ldr Qt, [Rn, #imm12*16]` / `str Qt, [Rn, #imm12*16]`.
pub(crate) fn q_ldst_uimm(load: bool, rt: u32, rn: u32, imm12: u32) -> u32 {
    let base = if load { 0x3DC0_0000 } else { 0x3D80_0000 };
    base | ((imm12 & 0xFFF) << 10) | (rn << 5) | rt
}

/// A spill/reload of a whole `q` register at `[base, #off]` (`off` 16-aligned).
fn frame_q_ldst(b: &mut A64Buf, load: bool, rt: u32, base: u32, off: u32) {
    if off / 16 < 4096 {
        b.word(q_ldst_uimm(load, rt, base, off / 16));
        return;
    }
    addsub_any(b, false, IP0, base, u64::from(off & !0xFFF));
    b.word(q_ldst_uimm(load, rt, IP0, (off & 0xFFF) / 16));
}

/// The base register and offset that reach the frame slot at (static,
/// `sp`-relative) offset `off`: `[sp, #off]` in a fixed frame; in a frame
/// whose `sp` moves (a `DynAlloca`), `x16 = x29 - (extra - off)` then
/// `[x16, #0]`.
fn slot_base(b: &mut A64Buf, layout: &FrameLayout, off: u32) -> (u32, u32) {
    if !layout.dynamic {
        return (SP.into(), off);
    }
    addsub_any(b, true, IP0, FP.into(), u64::from(layout.extra - off));
    (IP0, 0)
}

/// A free FP scratch (`v29..v31`, never allocated) not named in `avoid`.
fn free_fp_scratch(avoid: &[u32]) -> u32 {
    [31u32, 30, 29].into_iter().find(|r| !avoid.contains(r)).expect("three FP scratches")
}

/// Encode the NEON MIR ops.
fn encode_neon(b: &mut A64Buf, op: A64Op, ops: &[MachineOperand]) {
    match op {
        A64Op::NeonOp3 => {
            let nop = NeonOp::from_code(uimm(&ops[3]));
            b.word(neon3(nop, uimm(&ops[4]) as u32, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::NeonOp2 => {
            let nop = NeonOp::from_code(uimm(&ops[2]));
            b.word(neon2(nop, uimm(&ops[3]) as u32, rnum(&ops[0]), rnum(&ops[1])));
        }
        A64Op::NeonShift => {
            let nop = NeonOp::from_code(uimm(&ops[2]));
            let (d, n) = (rnum(&ops[0]), rnum(&ops[1]));
            b.word(neon_shift(nop, uimm(&ops[3]) as u32, d, n, uimm(&ops[4]) as u32));
        }
        A64Op::NeonDup => b.word(neon_dup(uimm(&ops[2]) as u32, rnum(&ops[0]), rnum(&ops[1]))),
        A64Op::NeonDupLane => {
            b.word(neon_dup_lane(uimm(&ops[2]) as u32, uimm(&ops[3]) as u32, rnum(&ops[0]), rnum(&ops[1])));
        }
        A64Op::NeonUmov => {
            b.word(neon_umov(uimm(&ops[2]) as u32, uimm(&ops[3]) as u32, rnum(&ops[0]), rnum(&ops[1])));
        }
        A64Op::NeonInsGpr => {
            let (d, v, g) = (rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]));
            if d != v {
                b.word(simd_mov(d, v));
            }
            b.word(neon_ins_gpr(uimm(&ops[3]) as u32, uimm(&ops[4]) as u32, d, g));
        }
        A64Op::NeonInsElem => {
            let (d, v, mut s) = (rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]));
            if d != v {
                if s == d {
                    // Save the element before the copy overwrites it.
                    let t = free_fp_scratch(&[d, v, s]);
                    b.word(simd_mov(t, s));
                    s = t;
                }
                b.word(simd_mov(d, v));
            }
            b.word(neon_ins_elem(uimm(&ops[3]) as u32, uimm(&ops[4]) as u32, d, s));
        }
        A64Op::NeonLoad => b.word(q_ldst_uimm(true, rnum(&ops[0]), rnum(&ops[1]), 0)),
        A64Op::NeonStore => b.word(q_ldst_uimm(false, rnum(&ops[1]), rnum(&ops[0]), 0)),
        A64Op::NeonConst => {
            let d = rnum(&ops[0]);
            let (lo, hi) = (uimm(&ops[1]), uimm(&ops[2]));
            // fmov Dd, x16 zeroes the upper half; ins fills it when needed.
            encode_movri(b, IP0, lo);
            b.word(fmov_from_gpr(1, 1, d, IP0));
            if hi != 0 {
                encode_movri(b, IP0, hi);
                b.word(neon_ins_gpr(64, 1, d, IP0));
            }
        }
        other => unreachable!("not a NEON op: {other:?}"),
    }
}

/// Encode one machine instruction into `b`.
fn encode_inst(b: &mut A64Buf, inst: &MachineInst, ctx: &EncodeCtx<'_>) {
    let ops = &inst.operands;
    match A64Op::decode(inst.opcode) {
        A64Op::MovRR => {
            let d = rnum(&ops[0]);
            let s = rnum(&ops[1]);
            if d != s {
                match rclass(&ops[0]) {
                    // A GPR copy is `orr d, xzr, s`; an FP/SIMD copy is `mov
                    // Vd.16b, Vs.16b`, all 128 bits (a register may hold a
                    // vector; a scalar float lives in the low lane).
                    RegClass::Gpr => b.word(mov_reg(1, d, s)),
                    RegClass::Fp => b.word(simd_mov(d, s)),
                }
            }
        }
        op @ (A64Op::NeonOp3
        | A64Op::NeonOp2
        | A64Op::NeonShift
        | A64Op::NeonDup
        | A64Op::NeonDupLane
        | A64Op::NeonUmov
        | A64Op::NeonInsGpr
        | A64Op::NeonInsElem
        | A64Op::NeonLoad
        | A64Op::NeonStore
        | A64Op::NeonConst) => encode_neon(b, op, ops),
        A64Op::MovRI => encode_movri(b, rnum(&ops[0]), uimm(&ops[1])),
        A64Op::Add => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(add_reg(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::Sub => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(sub_reg(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::And => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(and_reg(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::Or => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(orr_reg(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::Eor => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(eor_reg(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::Mul => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(madd(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), XZR.into()));
        }
        A64Op::AddI => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(add_imm(sf, rnum(&ops[0]), rnum(&ops[1]), uimm(&ops[2]) as u32));
        }
        A64Op::SubI => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(sub_imm(sf, rnum(&ops[0]), rnum(&ops[1]), uimm(&ops[2]) as u32));
        }
        A64Op::Sdiv => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(sdiv(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::Udiv => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(udiv(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::Msub => {
            let sf = sf_of(uimm(&ops[4]) as u32);
            // [d, m, n, a] => msub d, m, n, a  (Rd, Rn, Rm, Ra) = d = a - m*n.
            b.word(msub(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), rnum(&ops[3])));
        }
        A64Op::LslI => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(lsl_imm(sf, rnum(&ops[0]), rnum(&ops[1]), uimm(&ops[2]) as u32));
        }
        A64Op::LsrI => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(lsr_imm(sf, rnum(&ops[0]), rnum(&ops[1]), uimm(&ops[2]) as u32));
        }
        A64Op::AsrI => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(asr_imm(sf, rnum(&ops[0]), rnum(&ops[1]), uimm(&ops[2]) as u32));
        }
        A64Op::LslV => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(lslv(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::LsrV => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(lsrv(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::AsrV => {
            let sf = sf_of(uimm(&ops[3]) as u32);
            b.word(asrv(sf, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2])));
        }
        A64Op::CmpCset => {
            let d = rnum(&ops[0]);
            let a = rnum(&ops[1]);
            let bb = rnum(&ops[2]);
            let cc = uimm(&ops[3]) as u32;
            let sf = sf_of(uimm(&ops[4]) as u32);
            b.word(subs_reg(sf, XZR.into(), a, bb)); // cmp a, b
            b.word(cset(sf, d, cc)); // cset d, cond  (result is a 32/64-bit 0/1)
        }
        A64Op::CmpZero => b.word(subs_reg(1, XZR.into(), rnum(&ops[0]), XZR.into())), // cmp cond, xzr
        A64Op::CselNe => b.word(csel(1, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), 0x1)), // csel ..., NE
        A64Op::Csel => {
            let d = rnum(&ops[0]);
            let c = rnum(&ops[1]);
            let t = rnum(&ops[2]);
            let f = rnum(&ops[3]);
            b.word(subs_reg(1, XZR.into(), c, XZR.into())); // cmp cond, xzr
            b.word(csel(1, d, t, f, 0x1)); // csel d, t, f, NE  (cond != 0 -> t)
        }
        A64Op::Load => {
            let d = rnum(&ops[0]);
            let ptr = rnum(&ops[1]);
            let size = uimm(&ops[2]);
            let word = match rclass(&ops[0]) {
                RegClass::Gpr => ldst_uimm(true, ldst_size(size), d, ptr, 0),
                RegClass::Fp => fp_ldst_uimm(true, ldst_size(size), d, ptr, 0),
            };
            b.word(word);
        }
        A64Op::Store => {
            let ptr = rnum(&ops[0]);
            let val = rnum(&ops[1]);
            let size = uimm(&ops[2]);
            let word = match rclass(&ops[1]) {
                RegClass::Gpr => ldst_uimm(false, ldst_size(size), val, ptr, 0),
                RegClass::Fp => fp_ldst_uimm(false, ldst_size(size), val, ptr, 0),
            };
            b.word(word);
        }
        A64Op::LoadAcq => {
            b.word(ldar(ldst_size(uimm(&ops[2])), rnum(&ops[0]), rnum(&ops[1])));
        }
        A64Op::StoreRel => {
            b.word(stlr(ldst_size(uimm(&ops[2])), rnum(&ops[1]), rnum(&ops[0])));
        }
        A64Op::Dmb => b.word(dmb(uimm(&ops[0]) as u32)),
        A64Op::AtomicRmw => encode_atomic_rmw(b, ops),
        A64Op::CmpXchg => encode_cmpxchg(b, ops),
        A64Op::FrameAddr => {
            let d = rnum(&ops[0]);
            let off = ctx.layout.slot_off[slot_index(&ops[1])];
            if ctx.layout.dynamic {
                addsub_any(b, true, d, FP.into(), u64::from(ctx.layout.extra - off));
            } else {
                addsub_any(b, false, d, SP.into(), u64::from(off));
            }
        }
        // A spilled FP/SIMD register may hold a vector: the whole `q` register
        // goes to its 16-byte, 16-aligned slot.
        A64Op::StoreFrame | A64Op::LoadFrame => {
            let load = A64Op::decode(inst.opcode) == A64Op::LoadFrame;
            let r = rnum(&ops[0]);
            let (base, off) = slot_base(b, ctx.layout, ctx.layout.slot_off[slot_index(&ops[1])]);
            match rclass(&ops[0]) {
                RegClass::Fp => frame_q_ldst(b, load, r, base, off),
                RegClass::Gpr => frame_ldst_any(b, RegClass::Gpr, load, r, base, off),
            }
        }
        A64Op::GlobalAddr => {
            let d = rnum(&ops[0]);
            let g = match ops[1] {
                MachineOperand::Global(g) => g,
                _ => panic!("GlobalAddr expects a global operand"),
            };
            symbol_addr(b, d, (ctx.global_name)(g), (ctx.got.global)(g));
        }
        A64Op::FuncAddr => {
            let d = rnum(&ops[0]);
            let f = match ops[1] {
                MachineOperand::Func(f) => f,
                _ => panic!("FuncAddr expects a function operand"),
            };
            symbol_addr(b, d, (ctx.func_name)(f), (ctx.got.func)(f));
        }
        A64Op::DynAlloca => encode_dyn_alloca(b, ops, ctx.layout),
        A64Op::SpFromFp => addsub_any(b, true, SP.into(), FP.into(), uimm(&ops[0])),
        A64Op::Call => match &ops[0] {
            MachineOperand::Func(idx) => {
                b.reloc(bl(0), (ctx.func_name)(*idx), RelocKind::Aarch64Call26);
            }
            MachineOperand::Use(Reg::Physical(p)) => {
                b.word(blr(u32::from(p.num)));
            }
            other => panic!("Call expects a Func or register operand, found {other:?}"),
        },
        A64Op::Ret => b.word(ret(LR.into())),
        A64Op::B => {
            let t = label_index(&ops[0]);
            b.branch(b_uncond(0), t, FixupKind::Imm26);
        }
        A64Op::BrCond => {
            let cond = rnum(&ops[0]);
            let t = label_index(&ops[1]);
            let f = label_index(&ops[2]);
            b.branch(cbz(1, cond, 0, true), t, FixupKind::Imm19); // cbnz cond, t
            b.branch(b_uncond(0), f, FixupKind::Imm26); // b f
        }
        A64Op::Switch => {
            let cond = rnum(&ops[0]);
            let default = label_index(&ops[1]);
            let mut i = 2;
            while i + 1 < ops.len() {
                let value = uimm(&ops[i]);
                let case = label_index(&ops[i + 1]);
                // Materialize the case value into scratch x9, compare, branch equal.
                encode_movri(b, u32::from(super::regs::X9), value);
                b.word(subs_reg(1, XZR.into(), cond, u32::from(super::regs::X9)));
                b.branch(b_cond(0x0, 0), case, FixupKind::Imm19); // b.eq case
                i += 2;
            }
            b.branch(b_uncond(0), default, FixupKind::Imm26);
        }
        A64Op::Unreachable => b.word(brk(1)),
        A64Op::Svc => b.word(svc(0)),
        A64Op::Sbfx => b.word(sbfx0(1, rnum(&ops[0]), rnum(&ops[1]), uimm(&ops[2]) as u32)),
        A64Op::Ubfx => b.word(ubfx0(1, rnum(&ops[0]), rnum(&ops[1]), uimm(&ops[2]) as u32)),
        A64Op::StpFpLr => b.word(stp_pre(FP.into(), LR.into(), SP.into(), -2)),
        A64Op::LdpFpLr => b.word(ldp_post(FP.into(), LR.into(), SP.into(), 2)),
        A64Op::MovFpSp => b.word(add_imm(1, FP.into(), SP.into(), 0)),
        A64Op::SubSp => {
            let probe = ops.get(1).is_some_and(|o| uimm(o) != 0);
            sub_sp(b, uimm(&ops[0]), probe);
        }
        A64Op::AddSp => addsub_any(b, false, SP.into(), SP.into(), uimm(&ops[0])),
        A64Op::SaveReg => {
            let r = rnum(&ops[0]);
            let off = uimm(&ops[1]) as u32;
            frame_ldst_any(b, rclass(&ops[0]), false, r, SP.into(), off);
        }
        A64Op::RestoreReg => {
            let r = rnum(&ops[0]);
            let off = uimm(&ops[1]) as u32;
            frame_ldst_any(b, rclass(&ops[0]), true, r, SP.into(), off);
        }

        // --- scalar floating-point ----------------------------------------
        A64Op::FAdd | A64Op::FSub | A64Op::FMul | A64Op::FDiv => {
            let ptype = super::isel::ptype_of(uimm(&ops[3]) as u32);
            let (d, a, m) = (rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]));
            let word = match A64Op::decode(inst.opcode) {
                A64Op::FAdd => fadd(ptype, d, a, m),
                A64Op::FSub => fsub(ptype, d, a, m),
                A64Op::FMul => fmul(ptype, d, a, m),
                _ => fdiv(ptype, d, a, m),
            };
            b.word(word);
        }
        A64Op::FNeg => {
            let ptype = super::isel::ptype_of(uimm(&ops[2]) as u32);
            b.word(fneg(ptype, rnum(&ops[0]), rnum(&ops[1])));
        }
        A64Op::Fcmp => {
            let d = rnum(&ops[0]);
            let a = rnum(&ops[1]);
            let m = rnum(&ops[2]);
            let packed = uimm(&ops[3]);
            let ptype = super::isel::ptype_of(uimm(&ops[4]) as u32);
            let cond = (packed & 0xF) as u32;
            let combine = super::isel::Combine::decode((packed >> 4) & 0xF);
            let cond2 = ((packed >> 8) & 0xF) as u32;
            b.word(fcmp(ptype, a, m)); // fcmp Da, Db (sets NZCV)
            // The i1 result is a 32-bit gpr value; `cset w` zeroes the top 32 bits.
            b.word(cset(0, d, cond));
            match combine {
                super::isel::Combine::None => {}
                super::isel::Combine::And | super::isel::Combine::Or => {
                    let tmp = u32::from(super::regs::X9);
                    b.word(cset(0, tmp, cond2));
                    if combine == super::isel::Combine::And {
                        b.word(and_reg(0, d, d, tmp));
                    } else {
                        b.word(orr_reg(0, d, d, tmp));
                    }
                }
            }
        }
        A64Op::LoadFConst => {
            let d = rnum(&ops[0]);
            let bits = uimm(&ops[1]);
            let width = uimm(&ops[2]) as u32;
            let tmp = u32::from(super::regs::X9);
            let ptype = super::isel::ptype_of(width);
            if width >= 64 {
                encode_movri(b, tmp, bits);
                b.word(fmov_from_gpr(1, ptype, d, tmp)); // fmov Dd, x9
            } else {
                // Materialize the 32-bit pattern (a `movz`/`movk` chain), then move
                // the low word into the single-precision lane (`fmov Sd, w9`).
                encode_movri(b, tmp, bits & 0xFFFF_FFFF);
                b.word(fmov_from_gpr(0, ptype, d, tmp)); // fmov Sd, w9
            }
        }
        A64Op::Fcvt => {
            let dst_w = uimm(&ops[2]) as u32;
            let src_w = uimm(&ops[3]) as u32;
            b.word(fcvt(super::isel::ptype_of(src_w), fcvt_dst_ptype(dst_w), rnum(&ops[0]), rnum(&ops[1])));
        }
        A64Op::Fcvtzs | A64Op::Fcvtzu => {
            let dst_int_w = uimm(&ops[2]) as u32;
            let src_flt_w = uimm(&ops[3]) as u32;
            let sf = u32::from(dst_int_w > 32);
            let ptype = super::isel::ptype_of(src_flt_w);
            let word = if A64Op::decode(inst.opcode) == A64Op::Fcvtzs {
                fcvtzs(sf, ptype, rnum(&ops[0]), rnum(&ops[1]))
            } else {
                fcvtzu(sf, ptype, rnum(&ops[0]), rnum(&ops[1]))
            };
            b.word(word);
        }
        A64Op::Scvtf | A64Op::Ucvtf => {
            let dst_flt_w = uimm(&ops[2]) as u32;
            let src_int_w = uimm(&ops[3]) as u32;
            let sf = u32::from(src_int_w > 32);
            let ptype = super::isel::ptype_of(dst_flt_w);
            let word = if A64Op::decode(inst.opcode) == A64Op::Scvtf {
                scvtf(sf, ptype, rnum(&ops[0]), rnum(&ops[1]))
            } else {
                ucvtf(sf, ptype, rnum(&ops[0]), rnum(&ops[1]))
            };
            b.word(word);
        }

        // --- aggregate ABI stack addressing -------------------------------
        A64Op::LeaSpOff => {
            // add d, sp, #off — address the outgoing stack-argument area.
            let d = rnum(&ops[0]);
            let off = uimm(&ops[1]) as u32;
            b.word(add_imm(1, d, SP.into(), off));
        }
        A64Op::LeaFpOff => {
            // add d, x29, #off — address an incoming stack-passed parameter.
            let d = rnum(&ops[0]);
            let off = uimm(&ops[1]) as u32;
            b.word(add_imm(1, d, FP.into(), off));
        }
    }
}

/// The `fcvt` destination ptype for an integer bit width: `1` for f64, `0` for
/// f32 (the FP data-processing "ptype" convention).
#[inline]
fn fcvt_dst_ptype(dst_w: u32) -> u32 {
    u32::from(dst_w >= 64)
}

/// A frame (spill/reload/callee-save) `ldr`/`str` of a whole 64-bit lane at
/// `[base, #off]`, using the GPR (`x`) or FP (`d`) form per the register
/// class. `off` is a byte offset; the encoded unsigned immediate is `off / 8`.
fn frame_ldst(class: RegClass, load: bool, rt: u32, base: u32, off: u32) -> u32 {
    match class {
        RegClass::Gpr => ldst_uimm(load, SIZE_DWORD, rt, base, off / 8),
        RegClass::Fp => fp_ldst_uimm(load, SIZE_DWORD, rt, base, off / 8),
    }
}


/// `rd = rn + amount` (or `- amount` when `sub`) for any `amount`; `rd`/`rn` may
/// be `sp`. Up to 16 MiB this is `#hi, lsl #12` then `#lo` (one word when either
/// half is zero); beyond, `amount` goes through `x16` and the extended-register
/// form (`uxtx`, which accepts `sp`).
fn addsub_any(b: &mut A64Buf, sub: bool, rd: u32, rn: u32, amount: u64) {
    let base = if sub { 0x5100_0000 } else { 0x1100_0000 };
    if amount < 1 << 24 {
        let hi = (amount >> 12) as u32;
        let lo = (amount & 0xFFF) as u32;
        let mut src = rn;
        if hi != 0 {
            b.word(addsub_imm(base, 1, rd, src, hi) | (1 << 22)); // #hi, lsl #12
            src = rd;
        }
        if lo != 0 || hi == 0 {
            b.word(addsub_imm(base, 1, rd, src, lo));
        }
    } else {
        encode_movri(b, IP0, amount);
        let op = if sub { 0xCB20_6000 } else { 0x8B20_6000 }; // add/sub Xd|SP, Xn|SP, x16, uxtx
        b.word(op | (IP0 << 16) | (rn << 5) | rd);
    }
}

/// A 64-bit spill/reload `[base, #off]` for any `off` (8-aligned): the scaled
/// `imm12` form when it reaches, else `x16 = base + (off & !0xFFF)` and the
/// remainder as the immediate.
fn frame_ldst_any(b: &mut A64Buf, class: RegClass, load: bool, rt: u32, base: u32, off: u32) {
    if off / 8 < 4096 {
        b.word(frame_ldst(class, load, rt, base, off));
        return;
    }
    addsub_any(b, false, IP0, base, u64::from(off & !0xFFF));
    let imm12 = (off & 0xFFF) / 8;
    b.word(match class {
        RegClass::Gpr => ldst_uimm(load, SIZE_DWORD, rt, IP0, imm12),
        RegClass::Fp => fp_ldst_uimm(load, SIZE_DWORD, rt, IP0, imm12),
    });
}

/// Pages up to which a probed `sub sp` is unrolled rather than looped.
const PROBE_UNROLL: u64 = 4;

/// The prologue's `sub sp, sp, #amount`, probed (see the module docs) when
/// `probe` and `amount` is at least [`STACK_PROBE_INTERVAL`].
fn sub_sp(b: &mut A64Buf, amount: u64, probe: bool) {
    if !probe || amount < STACK_PROBE_INTERVAL {
        addsub_any(b, true, SP.into(), SP.into(), amount);
        return;
    }
    let pages = amount / STACK_PROBE_INTERVAL;
    let rem = amount % STACK_PROBE_INTERVAL;
    let step = STACK_PROBE_INTERVAL as u32 >> 12;
    let sub_page = addsub_imm(0x5100_0000, 1, SP.into(), SP.into(), step) | (1 << 22);
    let probe_word = ldst_uimm(false, SIZE_DWORD, XZR.into(), SP.into(), 0); // str xzr, [sp]
    if pages <= PROBE_UNROLL {
        for _ in 0..pages {
            b.word(sub_page);
            b.word(probe_word);
        }
    } else {
        encode_movri(b, IP0, pages);
        b.word(sub_page);
        b.word(probe_word);
        b.word(addsub_imm(0x7100_0000, 1, IP0, IP0, 1)); // subs x16, x16, #1
        b.word(b_cond(0x1, -3)); // b.ne (back to the sub)
    }
    if rem > 0 {
        b.word(sub_imm(1, SP.into(), SP.into(), rem as u32));
    }
}

/// `sub sp, sp, Xm, uxtx` (extended-register form, which accepts `sp`).
pub(crate) fn sub_sp_reg(rm: u32) -> u32 {
    0xCB20_6000 | (rm << 16) | (u32::from(SP) << 5) | u32::from(SP)
}

/// `cmp Xn, #1, lsl #12` (`subs xzr, Xn, #4096`).
pub(crate) fn cmp_page(rn: u32) -> u32 {
    addsub_imm(0x7100_0000, 1, XZR.into(), rn, 1) | (1 << 22)
}

/// Expand [`A64Op::DynAlloca`] `[d, n, align]`. `d` is the size scratch, then
/// the result; `C = outgoing + slack` (`slack = align` when `align > 16`, so
/// the pointer can round up inside the block), a multiple of 16:
///
/// ```text
/// add  d, n, #(15 + C) ; lsr d, d, #4 ; lsl d, d, #4   // round16(n) + C
/// ldr  xzr, [sp]                                     // probes: touch the top
/// L: cmp d, #1, lsl #12 ; b.lo done                  //   while d >= 4096:
///    sub sp, sp, #1, lsl #12 ; str xzr, [sp]         //     one probed page
///    sub d, d, #1, lsl #12 ; b L
/// done:
/// sub  sp, sp, d, uxtx                               // the remainder
/// str  xzr, [sp]                                     // probes: touch it
/// add  d, sp, #outgoing                              // above the new outgoing area
/// add  d, d, #(align - 1) ; lsr ; lsl                // align > 16 only
/// ```
///
/// The outgoing-argument area stays at the bottom of the frame
/// (`[sp, sp + outgoing)`): the block handed back starts above it, so
/// `LeaSpOff` addressing of stack arguments is unchanged. The current top is
/// probed with a load (it may hold a callee-saved register's save slot or a
/// local), the fresh pages with stores; so with probes `sp` never moves more
/// than one interval below the deepest access (see [`crate::codegen::stack`]).
fn encode_dyn_alloca(b: &mut A64Buf, ops: &[MachineOperand], layout: &FrameLayout) {
    let (d, n) = (rnum(&ops[0]), rnum(&ops[1]));
    let align = uimm(&ops[2]).max(1);
    let slack = if align > 16 { align } else { 0 };
    let c = layout.outgoing + slack;
    addsub_any(b, false, d, n, 15 + c);
    b.word(lsr_imm(1, d, d, 4));
    b.word(lsl_imm(1, d, d, 4));
    if layout.probes {
        b.word(ldst_uimm(true, SIZE_DWORD, XZR.into(), SP.into(), 0)); // ldr xzr, [sp]
        b.word(cmp_page(d));
        b.word(b_cond(0x3, 5)); // b.lo done
        b.word(addsub_imm(0x5100_0000, 1, SP.into(), SP.into(), 1) | (1 << 22)); // sub sp, #4096
        b.word(ldst_uimm(false, SIZE_DWORD, XZR.into(), SP.into(), 0)); // str xzr, [sp]
        b.word(addsub_imm(0x5100_0000, 1, d, d, 1) | (1 << 22)); // sub d, d, #4096
        b.word(b_uncond(-5)); // b L
    }
    b.word(sub_sp_reg(d));
    if layout.probes {
        b.word(ldst_uimm(false, SIZE_DWORD, XZR.into(), SP.into(), 0)); // str xzr, [sp]
    }
    addsub_any(b, false, d, SP.into(), layout.outgoing);
    if align > 16 {
        let k = align.trailing_zeros();
        addsub_any(b, false, d, d, align - 1);
        b.word(lsr_imm(1, d, d, k));
        b.word(lsl_imm(1, d, d, k));
    }
}

/// Test hook: the expansion of an allocated `DynAlloca` `[x<d>, x<n>,
/// align]` in a frame with `outgoing` bytes of outgoing-argument area.
#[cfg(test)]
pub(crate) fn dyn_alloca_for_test(d: u16, n: u16, align: u64, outgoing: u64, probes: bool) -> Vec<u8> {
    let layout = FrameLayout {
        slot_off: Vec::new(),
        cs_regs: Vec::new(),
        cs_off: Vec::new(),
        extra: 0,
        outgoing,
        probes,
        dynamic: true,
    };
    let ops = [
        MachineOperand::Def(Reg::Physical(super::regs::gpr(d))),
        MachineOperand::Use(Reg::Physical(super::regs::gpr(n))),
        imm_op(align),
    ];
    let mut b = A64Buf::new();
    encode_dyn_alloca(&mut b, &ops, &layout);
    b.bytes
}

/// Materialize a 64-bit constant into `rd` with a minimal `movz`/`movn`/`movk`
/// chain: seed with `movz` (or `movn`, when more lanes are all-ones) and patch
/// the remaining differing lanes with `movk`.
fn encode_movri(b: &mut A64Buf, rd: u32, value: u64) {
    let lanes = [
        (value & 0xFFFF) as u32,
        ((value >> 16) & 0xFFFF) as u32,
        ((value >> 32) & 0xFFFF) as u32,
        ((value >> 48) & 0xFFFF) as u32,
    ];
    let ones = lanes.iter().filter(|&&l| l == 0xFFFF).count();
    let zeros = lanes.iter().filter(|&&l| l == 0).count();

    if ones > zeros {
        // Seed with `movn` (inverted): the filler lanes become all-ones for free.
        let first = lanes.iter().position(|&l| l != 0xFFFF).unwrap_or(0);
        b.word(movn(1, rd, (!lanes[first]) & 0xFFFF, first as u32));
        for (hw, &lane) in lanes.iter().enumerate().skip(first + 1) {
            if lane != 0xFFFF {
                b.word(movk(1, rd, lane, hw as u32));
            }
        }
    } else {
        // Seed with `movz`: filler lanes become zero for free. `movz #0` covers
        // the all-zero case.
        let first = lanes.iter().position(|&l| l != 0).unwrap_or(0);
        b.word(movz(1, rd, lanes[first], first as u32));
        for (hw, &lane) in lanes.iter().enumerate().skip(first + 1) {
            if lane != 0 {
                b.word(movk(1, rd, lane, hw as u32));
            }
        }
    }
}

// ===========================================================================
// Function + module drivers
// ===========================================================================

/// Encode an allocated, prologue-inserted machine function into bytes and the
/// relocations its external references produced (position-dependent: every
/// symbol is addressed directly).
pub fn encode_function(
    mf: &MachineFunction,
    layout: &FrameLayout,
    func_name: &dyn Fn(u32) -> String,
    global_name: &dyn Fn(u32) -> String,
) -> Emitted {
    encode_function_inner(mf, layout, func_name, global_name, GotQuery::NONE, None)
}

/// [`encode_function`] with the GOT choice per symbol, collecting the
/// `(offset, source line)` statement rows of a `.debug_line` program when
/// `lines` is given (a row wherever the line changes).
fn encode_function_inner(
    mf: &MachineFunction,
    layout: &FrameLayout,
    func_name: &dyn Fn(u32) -> String,
    global_name: &dyn Fn(u32) -> String,
    got: GotQuery<'_>,
    mut lines: Option<&mut Vec<(u64, u32)>>,
) -> Emitted {
    let mut b = A64Buf::new();
    let ctx = EncodeCtx { layout, func_name, global_name, got };

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
            if let Some(rows) = lines.as_deref_mut()
                && inst.line != 0
                && rows.last().map(|&(_, l)| l) != Some(inst.line)
            {
                rows.push((b.offset(), inst.line));
            }
            encode_inst(&mut b, inst, &ctx);
        }
    }
    b.resolve(&block_off);
    Emitted { bytes: b.bytes, relocations: b.relocs }
}

/// One function's compile output: bytes + relocations, the `.debug_line`
/// statement rows (when requested), and its stack usage.
struct FunctionOutput {
    emitted: Emitted,
    rows: Vec<(u64, u32)>,
    stack: StackUsage,
    /// The frame saves no callee-saved register (only `x29`/`x30`).
    cs_free: bool,
}

/// Run isel → register allocation → frame layout → prologue/epilogue →
/// encoding for one function under `opts` (its OS selects the variadic
/// convention, its relocation model the GOT use), returning the code, the
/// line rows if `lines`, and its stack usage (`syms` names the callees).
fn compile_function_full(
    module: &Module,
    func: crate::ir::FuncId,
    syms: &StrInterner,
    opts: &CodegenOptions,
    lines: bool,
) -> FunctionOutput {
    let target = AArch64Target::for_os(opts.os);
    let mut mf = target.select_with_syms(module, func, syms);
    regalloc::allocate(&mut mf, &target);
    let layout = layout_frame_with(&mf, &target, opts);
    insert_prologue_epilogue(&mut mf, &layout);
    let func_name = |idx: u32| -> String {
        syms.resolve(module.function(crate::ir::FuncId::from_index(idx as usize)).name).to_owned()
    };
    let global_name = |idx: u32| -> String {
        syms.resolve(module.global(crate::ir::GlobalId::from_index(idx as usize)).name).to_owned()
    };
    let stack = layout.stack_usage(&mf, &func_name);
    let model = opts.reloc_model;
    let got_func = |idx: u32| {
        !crate::codegen::linkage::func_binds_locally(module, crate::ir::FuncId::from_index(idx as usize), model)
    };
    let got_global = |idx: u32| {
        !crate::codegen::linkage::global_binds_locally(module, crate::ir::GlobalId::from_index(idx as usize), model)
    };
    let mut rows = Vec::new();
    let emitted = encode_function_inner(
        &mf,
        &layout,
        &func_name,
        &global_name,
        GotQuery { func: &got_func, global: &got_global },
        if lines { Some(&mut rows) } else { None },
    );
    FunctionOutput { emitted, rows, stack, cs_free: layout.cs_regs.is_empty() }
}

/// Compile one function of `module` to its encoded bytes and relocations. Runs
/// isel → register allocation → frame layout → prologue/epilogue → encoding.
pub fn compile_function(module: &Module, func: crate::ir::FuncId, syms: &StrInterner) -> Emitted {
    let legal = crate::codegen::legalize::legalized(module, &NeonLegality);
    compile_function_full(&legal, func, syms, &CodegenOptions::default(), false).emitted
}

/// Compile every defined function of `module` into a relocatable
/// [`ObjectModule`]: a single `.text` section with one global function symbol
/// per definition, and the call/global relocations wired to (undefined-if-new)
/// symbols. `syms` resolves the interned function/global names. Uses the
/// default [`CodegenOptions`] (stack probes on); see [`compile_module_with`].
pub fn compile_module(module: &Module, syms: &StrInterner) -> ObjectModule {
    compile_module_with(module, syms, &CodegenOptions::default()).object
}

/// Like [`compile_module`], under `opts`, and also returning every defined
/// function's [`StackUsage`] (in definition order) in the [`CompiledModule`].
///
/// Under a position-independent [`RelocModel`](crate::codegen::RelocModel)
/// (`Pic`/`Pie`), the address of a symbol that may be preempted is loaded
/// from its GOT entry (`adrp`+`ldr`, `R_AARCH64_ADR_GOT_PAGE` +
/// `R_AARCH64_LD64_GOT_LO12_NC`), a locally bound one is formed directly
/// (`adrp`+`add`), calls stay `R_AARCH64_CALL26` (the linker adds a PLT entry
/// for a preemptible callee), and constants holding an address move to
/// `.data.rel.ro`: the object needs no text relocation.
pub fn compile_module_with(
    module: &Module,
    syms: &StrInterner,
    opts: &CodegenOptions,
) -> CompiledModule {
    build_module(module, syms, opts, None)
}

/// The source file a debug build describes (`DW_AT_name` / `DW_AT_comp_dir`).
pub use crate::target::x86_64::DebugSource;

/// [`compile_module_with`] plus DWARF: `.debug_abbrev`/`.debug_info`/
/// `.debug_str`/`.debug_line` describing every defined function (name,
/// address range, source-line table), their address fields
/// `R_AARCH64_ABS64` relocations against the function symbols.
pub fn compile_module_debug_with(
    module: &Module,
    syms: &StrInterner,
    source: &DebugSource,
    opts: &CodegenOptions,
) -> CompiledModule {
    build_module(module, syms, opts, Some(source))
}

/// The shared module driver behind [`compile_module_with`] and
/// [`compile_module_debug_with`] (DWARF when `debug` is given).
fn build_module(
    module: &Module,
    syms: &StrInterner,
    opts: &CodegenOptions,
    debug: Option<&DebugSource>,
) -> CompiledModule {
    use crate::mc::dwarf::{DebugUnit, FuncDebug};

    // Vector code NEON cannot hold or select is scalarized first.
    let legal = crate::codegen::legalize::legalized(module, &NeonLegality);
    let module: &Module = &legal;
    let mut obj = ObjectModule::new(module.name.clone());
    let text = obj.add_section(Section::new(".text", SectionKind::Text, 4));
    let mut stack = StackReport::new();
    let mut funcs: Vec<FuncDebug> = Vec::new();
    // Mach-O compact unwind records (function symbol, size, encoding).
    let mut compact = Vec::new();

    for (i, f) in module.functions().enumerate() {
        if f.is_declaration() {
            continue;
        }
        let fid = crate::ir::FuncId::from_index(i);
        let out = compile_function_full(module, fid, syms, opts, debug.is_some());
        let emitted = out.emitted;
        stack.push(out.stack);
        // 4-align this function's start within .text (A64 instructions are words).
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
        let fsym = obj.add_symbol(Symbol::defined(
            name.clone(),
            SymbolBinding::Global,
            SymbolType::Func,
            text,
            off,
            len,
        ));
        // `UNWIND_ARM64_MODE_FRAME` is `stp x29, x30, [sp, #-16]!; mov x29,
        // sp` with any callee-saved registers in pairs right below the
        // fp/lr pair; this layout keeps them at the bottom of the frame
        // instead, so only frames without them have a compact encoding.
        if out.cs_free {
            compact.push((fsym, len, crate::codegen::unwind::UNWIND_ARM64_MODE_FRAME));
        }
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

        if debug.is_some() {
            // A function-entry row at the declaration line, then the
            // statement rows (dropping runs of the same line).
            let decl_line = f.decl_line.unwrap_or(1);
            let mut rows = vec![(0u64, decl_line)];
            for (roff, line) in out.rows {
                if rows.last().map(|&(_, l)| l) != Some(line) {
                    rows.push((roff, line));
                }
            }
            funcs.push(FuncDebug { name, decl_line, size: len, rows });
        }
    }
    // Every defined global's storage, as on x86-64 (the `adrp`+`add` above
    // address these symbols). Under PIC, pointer-holding constants go to
    // `.data.rel.ro`, and the object says it needs no executable stack.
    let pic = opts.reloc_model.is_pic();
    crate::codegen::data::emit_globals_with(module, syms, &mut obj, RelocKind::Abs64, pic);
    crate::codegen::linkage::apply_symbol_attrs(module, syms, &mut obj);
    if pic {
        obj.add_section(Section::new(".note.GNU-stack", SectionKind::Debug, 1));
    }
    if opts.unwind_tables() == crate::codegen::UnwindTables::CompactUnwind {
        crate::codegen::unwind::emit_compact_unwind(&mut obj, &compact);
    }

    if let Some(source) = debug {
        let text_size = obj.section(text).bytes.len() as u64;
        let unit = DebugUnit {
            file_name: source.file_name.clone(),
            comp_dir: source.comp_dir.clone(),
            producer: "LatticeFoundry".to_owned(),
            text_size,
            funcs,
        };
        let dw = crate::mc::dwarf::build_with_address_size(&unit, 8);
        for (name, bytes) in [(".debug_abbrev", dw.abbrev), (".debug_str", dw.str)] {
            let mut s = Section::new(name, SectionKind::Debug, 1);
            s.bytes = bytes;
            obj.add_section(s);
        }
        obj.add_emitted_section(".debug_info", SectionKind::Debug, 1, dw.info);
        obj.add_emitted_section(".debug_line", SectionKind::Debug, 1, dw.line);
    }
    CompiledModule { object: obj, stack }
}
