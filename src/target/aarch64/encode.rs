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
//!    ([`compile_module`]).
//!
//! The encoding tables are implemented from the published ARM A64 instruction
//! encodings (tenet T1), not copied from any assembler.

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, PReg, Reg, RegClass, StackSlot};
use crate::codegen::regalloc;
use crate::ir::Module;
use crate::mc::emit::{Emitted, EmittedReloc};
use crate::mc::object::{
    ObjectModule, RelocKind, Section, SectionKind, Symbol, SymbolBinding, SymbolType,
};
use crate::support::StrInterner;

use super::isel::{A64Op, AArch64Target};
use super::regs::{FP, LR, SP, XZR};

// ===========================================================================
// 32-bit instruction-word builders (bitfields from the ARM A64 encodings)
// ===========================================================================

/// The `sf` bit (bit 31) selects the 64-bit (`X`) form; a `width` of 64 is
/// 64-bit, anything narrower uses the 32-bit (`W`) form.
#[inline]
fn sf_of(width: u32) -> u32 {
    u32::from(width >= 64)
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
/// `fmov Vd, Vn` (register move within the FP file).
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
/// (no dynamic stack growth), so `[sp, #off]` addressing is stable.
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
}

/// Round `value` up to a multiple of `align` (a power of two ≥ 1).
fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

/// Compute the frame layout of an allocated machine function.
pub fn layout_frame(mf: &MachineFunction, target: &AArch64Target) -> FrameLayout {
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

    FrameLayout { slot_off, cs_regs, cs_off, extra }
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
        prologue.push(MachineInst::new(A64Op::SubSp.opcode(), vec![imm_op(u64::from(layout.extra))]));
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

/// Encode one machine instruction into `b`.
fn encode_inst(b: &mut A64Buf, inst: &MachineInst, ctx: &EncodeCtx<'_>) {
    let ops = &inst.operands;
    match A64Op::decode(inst.opcode) {
        A64Op::MovRR => {
            let d = rnum(&ops[0]);
            let s = rnum(&ops[1]);
            if d != s {
                match rclass(&ops[0]) {
                    // A GPR copy is `orr d, xzr, s`; an FP copy is `fmov d, s`
                    // (the double form copies the whole 64-bit lane, which holds an
                    // f32 too).
                    RegClass::Gpr => b.word(mov_reg(1, d, s)),
                    RegClass::Fp => b.word(fmov_reg(1, d, s)),
                }
            }
        }
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
        A64Op::FrameAddr => {
            let d = rnum(&ops[0]);
            let off = ctx.layout.slot_off[slot_index(&ops[1])];
            b.word(add_imm(1, d, SP.into(), off));
        }
        A64Op::StoreFrame => {
            let src = rnum(&ops[0]);
            let off = ctx.layout.slot_off[slot_index(&ops[1])];
            b.word(frame_ldst(rclass(&ops[0]), false, src, off));
        }
        A64Op::LoadFrame => {
            let dst = rnum(&ops[0]);
            let off = ctx.layout.slot_off[slot_index(&ops[1])];
            b.word(frame_ldst(rclass(&ops[0]), true, dst, off));
        }
        A64Op::GlobalAddr => {
            let d = rnum(&ops[0]);
            let g = match ops[1] {
                MachineOperand::Global(g) => g,
                _ => panic!("GlobalAddr expects a global operand"),
            };
            let name = (ctx.global_name)(g);
            b.reloc(adrp(d), name.clone(), RelocKind::Aarch64AdrPrelPgHi21);
            b.reloc(add_imm(1, d, d, 0), name, RelocKind::Aarch64AddAbsLo12Nc);
        }
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
        A64Op::StpFpLr => b.word(stp_pre(FP.into(), LR.into(), SP.into(), -2)),
        A64Op::LdpFpLr => b.word(ldp_post(FP.into(), LR.into(), SP.into(), 2)),
        A64Op::MovFpSp => b.word(add_imm(1, FP.into(), SP.into(), 0)),
        A64Op::SubSp => b.word(sub_imm(1, SP.into(), SP.into(), uimm(&ops[0]) as u32)),
        A64Op::AddSp => b.word(add_imm(1, SP.into(), SP.into(), uimm(&ops[0]) as u32)),
        A64Op::SaveReg => {
            let r = rnum(&ops[0]);
            let off = uimm(&ops[1]) as u32;
            b.word(frame_ldst(rclass(&ops[0]), false, r, off));
        }
        A64Op::RestoreReg => {
            let r = rnum(&ops[0]);
            let off = uimm(&ops[1]) as u32;
            b.word(frame_ldst(rclass(&ops[0]), true, r, off));
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

/// A frame (spill/reload/callee-save) `ldr`/`str` of a whole 64-bit lane, using
/// the GPR (`x`) or FP (`d`) form per the register class. `off` is a byte offset;
/// the encoded unsigned immediate is `off / 8`.
fn frame_ldst(class: RegClass, load: bool, rt: u32, off: u32) -> u32 {
    match class {
        RegClass::Gpr => ldst_uimm(load, SIZE_DWORD, rt, SP.into(), off / 8),
        RegClass::Fp => fp_ldst_uimm(load, SIZE_DWORD, rt, SP.into(), off / 8),
    }
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
/// relocations its external references produced.
pub fn encode_function(
    mf: &MachineFunction,
    layout: &FrameLayout,
    func_name: &dyn Fn(u32) -> String,
    global_name: &dyn Fn(u32) -> String,
) -> Emitted {
    let mut b = A64Buf::new();
    let ctx = EncodeCtx { layout, func_name, global_name };

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
    Emitted { bytes: b.bytes, relocations: b.relocs }
}

/// Compile one function of `module` to its encoded bytes and relocations. Runs
/// isel → register allocation → frame layout → prologue/epilogue → encoding.
pub fn compile_function(module: &Module, func: crate::ir::FuncId, syms: &StrInterner) -> Emitted {
    let target = AArch64Target::new();
    let mut mf = target.select(module, func);
    regalloc::allocate(&mut mf, &target);
    let layout = layout_frame(&mf, &target);
    insert_prologue_epilogue(&mut mf, &layout);
    let func_name = |idx: u32| -> String {
        syms.resolve(module.function(crate::ir::FuncId::from_index(idx as usize)).name).to_owned()
    };
    let global_name = |idx: u32| -> String {
        syms.resolve(module.global(crate::ir::GlobalId::from_index(idx as usize)).name).to_owned()
    };
    encode_function(&mf, &layout, &func_name, &global_name)
}

/// Compile every defined function of `module` into a relocatable
/// [`ObjectModule`]: a single `.text` section with one global function symbol
/// per definition, and the call/global relocations wired to (undefined-if-new)
/// symbols. `syms` resolves the interned function/global names.
pub fn compile_module(module: &Module, syms: &StrInterner) -> ObjectModule {
    let mut obj = ObjectModule::new(module.name.clone());
    let text = obj.add_section(Section::new(".text", SectionKind::Text, 4));

    for (i, f) in module.functions().enumerate() {
        if f.is_declaration() {
            continue;
        }
        let fid = crate::ir::FuncId::from_index(i);
        let emitted = compile_function(module, fid, syms);
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
        obj.add_symbol(Symbol::defined(
            name,
            SymbolBinding::Global,
            SymbolType::Func,
            text,
            off,
            len,
        ));
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
    obj
}
