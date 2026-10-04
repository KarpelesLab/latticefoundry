//! The x86-64 machine-code encoder and the compile entry points (ROADMAP
//! Phase 7).
//!
//! After instruction selection ([`super::isel`]) and register allocation
//! ([`crate::codegen::regalloc`]) a [`MachineFunction`] holds only physical
//! registers and [`X86Op`] opcodes. This module:
//!
//! 1. lays out the stack frame ([`layout_frame`]) — which callee-saved registers
//!    the allocation used, the rbp-relative offset of every spill/`alloca` slot,
//!    and the `sub rsp` amount that keeps the stack 16-byte aligned at `call`s;
//! 2. splices in the prologue/epilogue as ordinary [`X86Op`] instructions
//!    ([`insert_prologue_epilogue`]);
//! 3. encodes each instruction to bytes ([`encode_function`]) — building the
//!    `REX` prefix, `ModRM`, `SIB`, displacements, and immediates by hand from
//!    the Intel/AMD encoding rules, resolving intra-function branches through the
//!    [`Emitter`]'s label mechanism and turning `call`/global references into
//!    relocations;
//! 4. assembles the functions of a module into an [`ObjectModule`]
//!    ([`compile_module`]) and, via [`crate::mc::elf`], an ELF64 object;
//!    [`compile_module_with`] also takes [`CodegenOptions`] and returns each
//!    function's [`StackUsage`] (read off the same [`FrameLayout`] the prologue
//!    is built from).
//!
//! **Stack probes** (on by default, see [`crate::codegen::stack`]): a frame whose
//! `sub rsp` amount is at least [`STACK_PROBE_INTERVAL`] is allocated one
//! interval at a time, each step followed by `or qword [rsp], 0`:
//!
//! ```text
//! sub rsp, 4096 ; or qword [rsp], 0      ; × pages, when pages <= 4
//!
//! mov r11d, pages                        ; otherwise, a counted loop
//! L: sub rsp, 4096 ; or qword [rsp], 0 ; dec r11 ; jnz L
//! sub rsp, remainder                     ; < 4096, if nonzero
//! ```
//!
//! and a `dyn_alloca` probes at run time: `or qword [rsp], 0`, then while the
//! (rounded) size is at least 4096, `sub rsp, 4096; or qword [rsp], 0` and
//! subtract 4096 from it, then `sub rsp` by the remainder.
//!
//! The encoding tables are implemented from the published x86-64 instruction-set
//! reference (tenet T1), not copied from any assembler.

use crate::codegen::mir::{MBlockId, MachineFunction, MachineInst, MachineOperand, Reg, RegClass, StackSlot};
use crate::codegen::options::{CodegenOptions, CompiledModule};
use crate::codegen::stack::{STACK_PROBE_INTERVAL, StackReport, StackUsage, scan_calls};
use crate::codegen::legalize::legalized;
use crate::codegen::regalloc;
use crate::codegen::unwind::{self, FrameOp, FrameStep, FunctionFrame, UnwindTables};
use crate::ir::Module;
use crate::mc::emit::{Emitted, Emitter, Label, Ref};
use crate::mc::object::{
    ObjectModule, RelocKind, Section, SectionKind, Symbol, SymbolBinding, SymbolType,
};
use crate::support::StrInterner;

use super::isel::{Sse2Legality, X86Op, X86_64Target};
use super::regs::{self, RBP, RSP};

// ===========================================================================
// Low-level byte builders (REX / ModRM / SIB and the instruction forms)
// ===========================================================================

/// Build a `REX` prefix byte from its four bits.
#[inline]
pub(crate) fn rex(w: bool, r: bool, x: bool, b: bool) -> u8 {
    0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8)
}

/// Build a `ModRM` byte.
#[inline]
pub(crate) fn modrm(md: u8, reg: u8, rm: u8) -> u8 {
    (md << 6) | ((reg & 7) << 3) | (rm & 7)
}

/// Build a `SIB` byte.
#[inline]
pub(crate) fn sib(scale: u8, index: u8, base: u8) -> u8 {
    (scale << 6) | ((index & 7) << 3) | (base & 7)
}

/// Emit a register-to-register ALU form `op r/m, r` (destination is the r/m
/// operand, source the reg operand): `add`, `sub`, `and`, `or`, `xor`, `mov`,
/// `cmp`, `test` all share this shape and differ only in the opcode byte.
pub(crate) fn alu_rr(e: &mut Emitter, opcode: u8, dst: u8, src: u8, w: bool) {
    if w || src >= 8 || dst >= 8 {
        e.u8(rex(w, src >= 8, false, dst >= 8));
    }
    e.u8(opcode);
    e.u8(modrm(3, src, dst));
}

/// Emit `mov dst, src` (64-bit register copy).
pub(crate) fn mov_rr(e: &mut Emitter, dst: u8, src: u8, w: bool) {
    alu_rr(e, 0x89, dst, src, w);
}

/// Emit `imul dst, src` (`0F AF /r`; destination is the reg operand).
pub(crate) fn imul_rr(e: &mut Emitter, dst: u8, src: u8, w: bool) {
    if w || dst >= 8 || src >= 8 {
        e.u8(rex(w, dst >= 8, false, src >= 8));
    }
    e.u8(0x0F);
    e.u8(0xAF);
    e.u8(modrm(3, dst, src));
}

/// Emit `neg r` (`F7 /3`).
pub(crate) fn neg_r(e: &mut Emitter, r: u8, w: bool) {
    if w || r >= 8 {
        e.u8(rex(w, false, false, r >= 8));
    }
    e.u8(0xF7);
    e.u8(modrm(3, 3, r));
}

/// Emit a `mov r, imm` — `B8+r id` (zero-extending) when the value fits in 32
/// bits, `REX.W C7 /0 id` (sign-extending) when it is a negative 32-bit
/// value, else `REX.W B8+r io` (`movabs`). Never touches the flags.
pub(crate) fn mov_ri(e: &mut Emitter, dst: u8, value: u64) {
    if value <= u64::from(u32::MAX) {
        if dst >= 8 {
            e.u8(rex(false, false, false, true));
        }
        e.u8(0xB8 + (dst & 7));
        e.u32(value as u32);
    } else if let Ok(v) = i32::try_from(value as i64) {
        e.u8(rex(true, false, false, dst >= 8));
        e.u8(0xC7);
        e.u8(modrm(3, 0, dst));
        e.u32(v as u32);
    } else {
        e.u8(rex(true, false, false, dst >= 8));
        e.u8(0xB8 + (dst & 7));
        e.u64(value);
    }
}

/// Emit a memory-operand instruction `opcode reg_field, [base + disp]`, choosing
/// the `ModRM.mod`/displacement size and inserting a `SIB` for `rsp`/`r12` and a
/// forced `disp8` for `rbp`/`r13`.
pub(crate) fn mem(
    e: &mut Emitter,
    opcode: &[u8],
    reg_field: u8,
    base: u8,
    disp: i32,
    w: bool,
    force_rex: bool,
) {
    let rexr = reg_field >= 8;
    let rexb = base >= 8;
    if w || rexr || rexb || force_rex {
        e.u8(rex(w, rexr, false, rexb));
    }
    e.bytes(opcode);
    let base3 = base & 7;
    let is_bp = base3 == 5; // rbp / r13: mod=00 would mean rip-relative
    let is_sp = base3 == 4; // rsp / r12: needs a SIB byte
    let (md, dsz) = if disp == 0 && !is_bp {
        (0u8, 0)
    } else if (-128..=127).contains(&disp) {
        (1u8, 1)
    } else {
        (2u8, 4)
    };
    e.u8(modrm(md, reg_field, base3));
    if is_sp {
        e.u8(sib(0, 4, base3));
    }
    match dsz {
        1 => e.u8(disp as u8),
        4 => e.u32(disp as u32),
        _ => {}
    }
}

/// Emit a `shift r/m, imm8` (`C1 /ext ib`); `ext` selects shl(4)/shr(5)/sar(7).
pub(crate) fn shift_imm(e: &mut Emitter, ext: u8, dst: u8, count: u8, w: bool) {
    if w || dst >= 8 {
        e.u8(rex(w, false, false, dst >= 8));
    }
    e.u8(0xC1);
    e.u8(modrm(3, ext, dst));
    e.u8(count);
}

/// Emit a `shift r/m, cl` (`D3 /ext`).
pub(crate) fn shift_cl(e: &mut Emitter, ext: u8, dst: u8, w: bool) {
    if w || dst >= 8 {
        e.u8(rex(w, false, false, dst >= 8));
    }
    e.u8(0xD3);
    e.u8(modrm(3, ext, dst));
}

/// Emit `setcc r/m8` (`0F 90+cc /0`), forcing a `REX` so `spl`/`bpl`/`sil`/`dil`
/// and `r8b..r15b` are addressable.
pub(crate) fn setcc(e: &mut Emitter, cc: u8, reg: u8) {
    if reg >= 4 {
        e.u8(rex(false, false, false, reg >= 8));
    }
    e.u8(0x0F);
    e.u8(0x90 + cc);
    e.u8(modrm(3, 0, reg));
}

/// Emit `movsx dst, src`: **sign**-extend a `src_w`-bit source into a `dst_w`-bit
/// destination. `0F BE` (byte) / `0F BF` (word) / `movsxd` `63` (dword→qword).
/// `REX.W` is set for a 64-bit destination.
pub(crate) fn movsx_rr(e: &mut Emitter, dst: u8, src: u8, src_w: u32, dst_w: u32) {
    let w = dst_w == 64;
    match src_w {
        8 => {
            // A byte source `spl/bpl/sil/dil` (src>=4) needs any REX to be addressable.
            if w || dst >= 8 || src >= 4 {
                e.u8(rex(w, dst >= 8, false, src >= 8));
            }
            e.u8(0x0F);
            e.u8(0xBE);
            e.u8(modrm(3, dst, src));
        }
        16 => {
            if w || dst >= 8 || src >= 8 {
                e.u8(rex(w, dst >= 8, false, src >= 8));
            }
            e.u8(0x0F);
            e.u8(0xBF);
            e.u8(modrm(3, dst, src));
        }
        32 => {
            // 32 → 64: `movsxd r64, r/m32` = REX.W 63 /r.
            e.u8(rex(true, dst >= 8, false, src >= 8));
            e.u8(0x63);
            e.u8(modrm(3, dst, src));
        }
        _ => extend_by_shifts(e, dst, src, src_w, 7),
    }
}

/// Extend an odd-width (`i1`, `i24`, `i48`, …) source to 64 bits by moving it to
/// the top of the register and shifting back down: `ext` 7 = `sar` (sign), 5 =
/// `shr` (zero). A 64-bit (or wider) source is a plain copy.
fn extend_by_shifts(e: &mut Emitter, dst: u8, src: u8, src_w: u32, ext: u8) {
    if dst != src {
        mov_rr(e, dst, src, true);
    }
    if src_w < 64 {
        let n = (64 - src_w) as u8;
        shift_imm(e, 4, dst, n, true); // shl dst, n
        shift_imm(e, ext, dst, n, true); // sar/shr dst, n
    }
}

/// Emit `movzx dst, src`: **zero**-extend a `src_w`-bit source. `0F B6` (byte) /
/// `0F B7` (word) zero-extend into the full register; a 32-bit source uses a
/// plain 32-bit `mov`, which zero-extends bits 32..63 automatically. Any other
/// width goes through [`extend_by_shifts`].
pub(crate) fn movzx_rr(e: &mut Emitter, dst: u8, src: u8, src_w: u32) {
    match src_w {
        8 => {
            if dst >= 8 || src >= 4 {
                e.u8(rex(false, dst >= 8, false, src >= 8));
            }
            e.u8(0x0F);
            e.u8(0xB6);
            e.u8(modrm(3, dst, src));
        }
        16 => {
            if dst >= 8 || src >= 8 {
                e.u8(rex(false, dst >= 8, false, src >= 8));
            }
            e.u8(0x0F);
            e.u8(0xB7);
            e.u8(modrm(3, dst, src));
        }
        32 => mov_rr(e, dst, src, false),
        _ => extend_by_shifts(e, dst, src, src_w, 5),
    }
}

/// Emit `movzx r32, r8` on the same register (`0F B6 /r`).
pub(crate) fn movzx_byte(e: &mut Emitter, reg: u8) {
    if reg >= 8 {
        e.u8(rex(false, true, false, true));
    } else if reg >= 4 {
        e.u8(0x40);
    }
    e.u8(0x0F);
    e.u8(0xB6);
    e.u8(modrm(3, reg, reg));
}

/// Emit `cmovcc dst, src` (`0F 40+cc /r`; destination is the reg operand).
pub(crate) fn cmov_rr(e: &mut Emitter, cc: u8, dst: u8, src: u8, w: bool) {
    if w || dst >= 8 || src >= 8 {
        e.u8(rex(w, dst >= 8, false, src >= 8));
    }
    e.u8(0x0F);
    e.u8(0x40 + cc);
    e.u8(modrm(3, dst, src));
}

/// Emit `idiv`/`div r/m` (`F7 /ext`); `ext` is 7 for idiv, 6 for div.
pub(crate) fn divide(e: &mut Emitter, ext: u8, r: u8, w: bool) {
    if w || r >= 8 {
        e.u8(rex(w, false, false, r >= 8));
    }
    e.u8(0xF7);
    e.u8(modrm(3, ext, r));
}

/// Emit `push r` (`50+r`, with `REX.B` for the extended registers).
pub(crate) fn push_r(e: &mut Emitter, r: u8) {
    if r >= 8 {
        e.u8(0x41);
    }
    e.u8(0x50 + (r & 7));
}

/// Emit `pop r` (`58+r`).
pub(crate) fn pop_r(e: &mut Emitter, r: u8) {
    if r >= 8 {
        e.u8(0x41);
    }
    e.u8(0x58 + (r & 7));
}

/// Emit `cmp a, b` at the operands' integer `width`. Values narrower than 32
/// bits live in wider host registers whose upper bits are not kept clean (an
/// `i8` add of 200 + 100 leaves 300 in the register), so an 8- or 16-bit
/// comparison must use the 8- or 16-bit form of `cmp` (`38 /r`, `66 39 /r`)
/// rather than comparing the whole 32-bit register.
pub(crate) fn cmp_rr_width(e: &mut Emitter, a: u8, b: u8, width: u32) {
    match width {
        0..=8 => {
            // Any REX prefix makes registers 4..7 name spl/bpl/sil/dil instead of
            // ah/ch/dh/bh, so emit one whenever such a register is involved.
            if a >= 4 || b >= 4 {
                e.u8(rex(false, b >= 8, false, a >= 8));
            }
            e.u8(0x38);
            e.u8(modrm(3, b, a));
        }
        9..=16 => {
            e.u8(0x66);
            alu_rr(e, 0x39, a, b, false);
        }
        _ => alu_rr(e, 0x39, a, b, width > 32),
    }
}

/// Emit `cmp r, imm32` (`REX.W 81 /7 id`).
fn cmp_ri(e: &mut Emitter, reg: u8, value: i32, w: bool) {
    if w || reg >= 8 {
        e.u8(rex(w, false, false, reg >= 8));
    }
    e.u8(0x81);
    e.u8(modrm(3, 7, reg));
    e.u32(value as u32);
}

/// Emit an ALU group-1 `op r, imm` (`83 /ext ib` when the value fits a
/// sign-extended byte, else `81 /ext id`); `ext` selects add(0)/or(1)/and(4)/
/// sub(5)/xor(6)/cmp(7).
pub(crate) fn alu_ri(e: &mut Emitter, ext: u8, reg: u8, value: i32, w: bool) {
    if w || reg >= 8 {
        e.u8(rex(w, false, false, reg >= 8));
    }
    match i8::try_from(value) {
        Ok(b) => {
            e.u8(0x83);
            e.u8(modrm(3, ext, reg));
            e.u8(b as u8);
        }
        Err(_) => {
            e.u8(0x81);
            e.u8(modrm(3, ext, reg));
            e.u32(value as u32);
        }
    }
}

/// Emit `cmp r, value` at the integer `width` (`value` sign-extended from
/// it): `test r, r` against 0 (the same flags), else the shortest immediate
/// form — `80 /7 ib` for a byte, `66 83 /7 ib` / `66 81 /7 iw` for a word,
/// `83 /7 ib` / `81 /7 id` otherwise.
pub(crate) fn cmp_ri_width(e: &mut Emitter, r: u8, value: i64, width: u32) {
    match width {
        0..=8 => {
            if value == 0 {
                if r >= 4 {
                    e.u8(rex(false, r >= 8, false, r >= 8));
                }
                e.u8(0x84);
                e.u8(modrm(3, r, r));
            } else {
                if r >= 4 {
                    e.u8(rex(false, false, false, r >= 8));
                }
                e.u8(0x80);
                e.u8(modrm(3, 7, r));
                e.u8(value as u8);
            }
        }
        9..=16 => {
            e.u8(0x66);
            if value == 0 {
                alu_rr(e, 0x85, r, r, false);
            } else if let Ok(b) = i8::try_from(value) {
                alu_ri(e, 7, r, i32::from(b), false);
            } else {
                if r >= 8 {
                    e.u8(rex(false, false, false, true));
                }
                e.u8(0x81);
                e.u8(modrm(3, 7, r));
                e.u16(value as u16);
            }
        }
        _ => {
            if value == 0 {
                alu_rr(e, 0x85, r, r, width > 32);
            } else {
                alu_ri(e, 7, r, value as i32, width > 32);
            }
        }
    }
}

/// A conditional jump `jcc` to `label` (`70+cc rel8`, or `0F 80+cc rel32`
/// when out of reach: the emitter relaxes it).
fn jcc(e: &mut Emitter, cc: u8, label: Label) {
    e.branch(&[0x70 + cc], &[0x0F, 0x80 + cc], label);
}

/// An unconditional `jmp` to `label` (`EB rel8` or `E9 rel32`).
fn jmp(e: &mut Emitter, label: Label) {
    e.branch(&[0xEB], &[0xE9], label);
}

// --- SSE (scalar floating-point) forms -------------------------------------

/// Emit an SSE register-to-register instruction: an optional mandatory prefix
/// (`0xF2`/`0xF3`/`0x66`; `0` means none), an optional `REX` (`.W` when `w`,
/// `.R`/`.B` for `xmm8..15` and `r8..15`), the `0F` escape, the opcode, and a
/// `ModRM` pairing the `reg` and `rm` register fields.
pub(crate) fn sse_rr(e: &mut Emitter, prefix: u8, w: bool, opcode: u8, reg: u8, rm: u8) {
    if prefix != 0 {
        e.u8(prefix);
    }
    if w || reg >= 8 || rm >= 8 {
        e.u8(rex(w, reg >= 8, false, rm >= 8));
    }
    e.u8(0x0F);
    e.u8(opcode);
    e.u8(modrm(3, reg, rm));
}

/// Emit an SSE memory-form instruction `prefix 0F opcode reg, [base + disp]`
/// (used by `movss`/`movsd` load/store). The mandatory prefix precedes the `REX`
/// that [`mem`] emits; SSE scalar moves never set `REX.W`.
pub(crate) fn sse_mem(e: &mut Emitter, prefix: u8, opcode: u8, reg: u8, base: u8, disp: i32) {
    if prefix != 0 {
        e.u8(prefix);
    }
    mem(e, &[0x0F, opcode], reg, base, disp, false, false);
}

/// The mandatory SSE prefix for a scalar op of the given width: `F2` (double) or
/// `F3` (single). Widths other than 64 use the single form.
#[inline]
fn scalar_prefix(is_f64: bool) -> u8 {
    if is_f64 { 0xF2 } else { 0xF3 }
}

/// The two-address expansion of an SSE binary op `d = a OP b`. Like the integer
/// ALU, the allocator gives `d`, `a`, `b` distinct registers, so a `movsd`/`movss`
/// copy of `a` into `d` precedes the op; commutativity lets `d == b` reuse `a`.
fn fbin(e: &mut Emitter, is_f64: bool, opcode: u8, d: u8, a: u8, b: u8, commutative: bool) {
    let pfx = scalar_prefix(is_f64);
    if d == a {
        sse_rr(e, pfx, false, opcode, d, b);
    } else if commutative && d == b {
        sse_rr(e, pfx, false, opcode, d, a);
    } else {
        debug_assert!(d != b, "non-commutative SSE op needs a distinct destination");
        sse_rr(e, pfx, false, 0x10, d, a); // movsd/movss d, a
        sse_rr(e, pfx, false, opcode, d, b);
    }
}

/// The two-address expansion of `d = a ^ b` (`xorpd`/`xorps`), used for `fneg`.
fn fxor(e: &mut Emitter, is_f64: bool, d: u8, a: u8, b: u8) {
    debug_assert!(d != b, "fneg mask must be distinct from the destination");
    if d != a {
        sse_rr(e, scalar_prefix(is_f64), false, 0x10, d, a); // movsd/movss d, a
    }
    let xor_pfx = if is_f64 { 0x66 } else { 0x00 };
    sse_rr(e, xor_pfx, false, 0x57, d, b); // xorpd/xorps d, b
}

/// `movaps d, s` (`0F 28 /r`): a full 128-bit xmm copy.
pub(crate) fn movaps(e: &mut Emitter, d: u8, s: u8) {
    sse_rr(e, 0x00, false, 0x28, d, s);
}

/// A scratch xmm (`xmm13..15`, never allocated) distinct from every register
/// in `avoid`. Scratch registers only carry a reload into the one instruction
/// that uses it, so one this instruction does not name is free here.
fn free_xmm_scratch(avoid: &[u8]) -> u8 {
    [15u8, 14, 13].into_iter().find(|r| !avoid.contains(r)).expect("three scratch xmms")
}

/// Expand [`X86Op::VOp`] `d = a OP b` into two-address SSE form (see
/// `vector::VEnc` for the packed immediate).
fn encode_vop(e: &mut Emitter, ops: &[MachineOperand]) {
    let (d, a, b) = (rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]));
    let (prefix, opcode, comm, imm8) = super::isel::vector::VEnc::decode(uimm(&ops[3]));
    let op = |e: &mut Emitter, reg: u8, rm: u8| {
        sse_rr(e, prefix, false, opcode, reg, rm);
        if let Some(i) = imm8 {
            e.u8(i);
        }
    };
    if d == a {
        op(e, d, b);
    } else if comm && d == b {
        op(e, d, a);
    } else if d == b {
        // Non-commutative with the destination aliasing the second source:
        // save it first.
        let t = free_xmm_scratch(&[d, a]);
        movaps(e, t, b);
        movaps(e, d, a);
        op(e, d, t);
    } else {
        movaps(e, d, a);
        op(e, d, b);
    }
}

/// Emit `and r64/r32, imm8` (sign-extended immediate) — `83 /4 ib`.
fn and_ri8(e: &mut Emitter, reg: u8, imm8: i8, w: bool) {
    if w || reg >= 8 {
        e.u8(rex(w, false, false, reg >= 8));
    }
    e.u8(0x83);
    e.u8(modrm(3, 4, reg)); // /4 selects AND
    e.u8(imm8 as u8);
}

/// Emit an ALU `op r/m, imm32` (`81 /ext id`, the immediate sign-extended);
/// `ext` selects add(0)/or(1)/and(4)/sub(5)/cmp(7).
fn alu_ri32(e: &mut Emitter, ext: u8, reg: u8, imm32: i32, w: bool) {
    if w || reg >= 8 {
        e.u8(rex(w, false, false, reg >= 8));
    }
    e.u8(0x81);
    e.u8(modrm(3, ext, reg));
    e.u32(imm32 as u32);
}

/// Emit the unsigned `u64 → f64`/`f32` conversion (`uitofp` from a 64-bit
/// source). x86 has no unsigned int→float, so: if the source's sign bit is clear
/// a direct `cvtsi2sd` is exact; otherwise convert `(s>>1)|(s&1)` — a halving
/// that keeps a sticky low bit so round-to-nearest matches gcc/clang — and
/// double the result. `s` is the 64-bit source GPR (preserved), `d` the xmm dest.
///
/// The two scratch GPRs are chosen to never collide with `s` or with a spilled
/// operand's reload register: this op has a single GPR operand (`s`), so the
/// allocator uses at most scratch index 0/1 (`r10`/`r11`) for it; `rbx` (index 2)
/// is always free, and the other temp is whichever of `r10`/`r11` is not `s`.
fn u64tof(e: &mut Emitter, d: u8, s: u8, is_f64: bool) {
    let pfx = scalar_prefix(is_f64);
    let t1 = regs::RBX as u8;
    let t2 = if s == regs::R10 as u8 { regs::R11 as u8 } else { regs::R10 as u8 };
    let neg = e.create_label();
    let done = e.create_label();
    alu_rr(e, 0x85, s, s, true); // test s, s  (64-bit: SF = bit 63)
    jcc(e, 0x8, neg); // js neg
    sse_rr(e, pfx, true, 0x2A, d, s); // cvtsi2sd/ss d, s  (in range ⇒ exact)
    jmp(e, done);
    e.bind_label(neg);
    mov_rr(e, t1, s, true); // t1 = s
    shift_imm(e, 5, t1, 1, true); // t1 >>= 1  (shr)
    mov_rr(e, t2, s, true); // t2 = s
    and_ri8(e, t2, 1, true); // t2 &= 1  (sticky low bit)
    alu_rr(e, 0x09, t1, t2, true); // t1 |= t2
    sse_rr(e, pfx, true, 0x2A, d, t1); // cvtsi2sd/ss d, t1
    sse_rr(e, pfx, false, 0x58, d, d); // addsd/ss d, d  (× 2)
    e.bind_label(done);
}

/// Emit the unsigned `f64`/`f32 → u64` conversion (`fptoui` to a 64-bit result),
/// truncating toward zero. `cvttsd2si` is signed, so inputs ≥ 2^63 are converted
/// as `x − 2^63` with the bias added back (bit 63 set via `xor`); inputs below
/// 2^63 convert directly. `s` is the source xmm (preserved), `d` the dest GPR.
///
/// Scratch choice mirrors `u64tof`: `r11` is free (the single GPR operand, `d`,
/// uses `r10` if spilled), `xmm15` is free (single xmm operand uses xmm13/xmm14),
/// and the value temp is whichever of `xmm13`/`xmm14` is not `s`.
fn fptou64(e: &mut Emitter, d: u8, s: u8, is_f64: bool) {
    let pfx = scalar_prefix(is_f64);
    let ucomi_pfx = if is_f64 { 0x66 } else { 0x00 };
    let thresh: u64 = if is_f64 { 0x43E0_0000_0000_0000 } else { 0x5F00_0000 }; // 2^63
    let t_thresh = 15u8;
    let t_val = if s == 13 { 14u8 } else { 13u8 };
    let tmp = regs::R11 as u8;
    let big = e.create_label();
    let done = e.create_label();
    if is_f64 {
        mov_ri(e, tmp, thresh); // movabs r11, 2^63
        sse_rr(e, 0x66, true, 0x6E, t_thresh, tmp); // movq t_thresh, r11
    } else {
        mov_ri(e, tmp, thresh & 0xFFFF_FFFF); // mov r11d, 2^63f
        sse_rr(e, 0x66, false, 0x6E, t_thresh, tmp); // movd t_thresh, r11d
    }
    sse_rr(e, ucomi_pfx, false, 0x2E, s, t_thresh); // ucomis s, 2^63
    jcc(e, 0x3, big); // jae big  (CF=0 ⇒ s ≥ 2^63)
    sse_rr(e, pfx, true, 0x2C, d, s); // cvttsd2si d, s  (in range)
    jmp(e, done);
    e.bind_label(big);
    sse_rr(e, pfx, false, 0x10, t_val, s); // movsd/ss t_val, s
    sse_rr(e, pfx, false, 0x5C, t_val, t_thresh); // subsd/ss t_val, 2^63
    sse_rr(e, pfx, true, 0x2C, d, t_val); // cvttsd2si d, (x − 2^63)
    mov_ri(e, tmp, 0x8000_0000_0000_0000); // movabs r11, 2^63
    alu_rr(e, 0x31, d, tmp, true); // xor d, r11  (add the bias back)
    e.bind_label(done);
}

/// Emit an 8-bit ALU `op r/m8, r8` (`opcode /r`), forcing a `REX` so `spl`-style
/// and `r8b..r15b` low bytes are addressable.
fn alu_byte(e: &mut Emitter, opcode: u8, rm: u8, reg: u8) {
    if rm >= 4 || reg >= 4 {
        e.u8(rex(false, reg >= 8, false, rm >= 8));
    }
    e.u8(opcode);
    e.u8(modrm(3, reg, rm));
}

/// The register class of a physical register operand.
fn rclass(op: &MachineOperand) -> RegClass {
    match op {
        MachineOperand::Def(Reg::Physical(p)) | MachineOperand::Use(Reg::Physical(p)) => p.class,
        other => panic!("expected a physical register operand, found {other:?}"),
    }
}

// ===========================================================================
// Frame layout + prologue/epilogue
// ===========================================================================

/// The stack-frame layout of one function, computed after allocation.
#[derive(Clone, Debug)]
pub struct FrameLayout {
    /// rbp-relative displacement of each stack slot (by slot index).
    slot_off: Vec<i32>,
    /// The callee-saved registers the allocation used, in push order.
    cs_regs: Vec<u8>,
    /// The callee-saved `xmm` registers the function writes (Win64's
    /// `xmm6..xmm15`), each with the rbp-relative offset of its 16-byte save
    /// slot just below the pushed GPRs. Always empty under System V.
    xmm_saves: Vec<(u8, i32)>,
    /// Bytes occupied by the pushed callee-saved registers (`8 * cs_regs.len()`).
    cs_bytes: i32,
    /// The `sub rsp` amount that follows the callee-saved pushes.
    sub_size: i32,
    /// The reserved outgoing-argument area size (bytes, a multiple of 16). The
    /// area sits at `[rsp, rsp + outgoing)`; a `DynAlloca` relocates it below the
    /// carved block so `[rsp + k]` stack-argument addressing survives a moving
    /// `rsp`.
    outgoing: i64,
    /// Whether the prologue's `sub rsp` and every `DynAlloca` emit stack probes.
    probes: bool,
    /// Whether the frame takes the Windows x64 shape (see
    /// [`FrameLayout::prologue_plan`]): `rbp` set after the pushes and a small
    /// fixed allocation, so the `.pdata`/`.xdata` unwind codes can describe
    /// it. Set for [`TargetOs::Windows`](crate::target::TargetOs::Windows).
    windows: bool,
    /// Windows: the fixed allocation made before `rbp` is set (the `xmm` save
    /// area plus the 8-byte pad that makes `cs_bytes + fixed` a multiple of
    /// 16); the rest of `sub_size` follows. Always 0 under System V.
    fixed: i32,
    /// Whether the frame omits the frame pointer (see [`layout_frame_with`]):
    /// no `push rbp`, and `slot_off` is relative to `rsp` (which stays put
    /// after the prologue) instead of `rbp`.
    fpo: bool,
    /// Whether the function has a `dyn_alloca` (which moves `rsp`, so the
    /// epilogue must restore it from `rbp`).
    dynamic: bool,
}

impl FrameLayout {
    /// The stack usage this layout gives `mf` (whose MIR supplies the call
    /// information; `func_name` resolves a function index — the callees and `mf`
    /// itself — to its symbol name): the return address, `push rbp`, the
    /// callee-saved pushes, and the `sub rsp` amount — exactly what the prologue
    /// built from this layout moves `rsp` by.
    pub fn stack_usage(
        &self,
        mf: &MachineFunction,
        func_name: &dyn Fn(u32) -> String,
    ) -> StackUsage {
        let scan = scan_calls(
            mf,
            X86Op::Call.opcode(),
            X86Op::Syscall.opcode(),
            Some(X86Op::DynAlloca.opcode()),
        );
        // rbp (unless omitted) + callee-saved pushes
        let saved = if self.fpo { 0 } else { 8 } + self.cs_bytes as u64;
        let sub = self.sub_size as u64;
        StackUsage {
            name: func_name(mf.info().source),
            frame_size: 8 + saved + sub,
            return_address: 8,
            saved_registers: saved,
            sp_adjust: sub,
            outgoing_args: self.outgoing as u64,
            dynamic_alloca: scan.dynamic_alloca,
            direct_callees: scan.direct.iter().map(|&f| func_name(f)).collect(),
            indirect_calls: scan.indirect,
            syscalls: scan.syscalls,
            probed: self.probes,
        }
    }
}

impl FrameLayout {
    /// The register stack slots are addressed from: `rbp`, or `rsp` in a
    /// frame without a frame pointer.
    pub(crate) fn base(&self) -> u8 {
        if self.fpo { RSP as u8 } else { RBP as u8 }
    }

    /// Whether the frame omits the frame pointer.
    pub fn omits_frame_pointer(&self) -> bool {
        self.fpo
    }

    /// The epilogue that undoes this layout's prologue, before each `ret`.
    ///
    /// System V restores `rsp` from `rbp` only when something moved it below
    /// the callee-saved pushes, and a frame without callee-saved registers
    /// ends in `leave`; one without a frame pointer pops its allocation with
    /// `add rsp`. Windows keeps the `lea rsp, [rbp - cs]; pop ...; pop rbp`
    /// form its unwinder recognizes.
    fn epilogue(&self) -> Vec<MachineInst> {
        let mut out = Vec::new();
        for &(x, off) in &self.xmm_saves {
            out.push(MachineInst::new(
                X86Op::RestoreXmm.opcode(),
                vec![
                    MachineOperand::Def(Reg::Physical(regs::xmm(u16::from(x)))),
                    MachineOperand::Imm(puremp::Int::from_i64(i64::from(off))),
                ],
            ));
        }
        let pops = |out: &mut Vec<MachineInst>| {
            for &cs in self.cs_regs.iter().rev() {
                out.push(MachineInst::new(X86Op::Pop.opcode(), vec![phys(u16::from(cs))]));
            }
        };
        if self.fpo {
            if self.sub_size > 0 {
                out.push(MachineInst::new(X86Op::AddRsp.opcode(), vec![imm_op(self.sub_size as u64)]));
            }
            pops(&mut out);
            return out;
        }
        if !self.windows && self.cs_regs.is_empty() {
            out.push(MachineInst::new(X86Op::Leave.opcode(), Vec::new()));
            return out;
        }
        if self.windows || self.sub_size > 0 || self.dynamic {
            out.push(MachineInst::new(X86Op::LeaRspRbp.opcode(), vec![imm_op(self.cs_bytes as u64)]));
        }
        pops(&mut out);
        out.push(MachineInst::new(X86Op::Pop.opcode(), vec![phys(RBP)]));
        out
    }

    /// The prologue built from this layout: each instruction, with what it
    /// does to the frame (for the unwind tables, see
    /// [`crate::codegen::unwind`]).
    ///
    /// System V:
    ///
    /// ```text
    /// push rbp ; mov rbp, rsp ; push <callee-saved>... ; sub rsp, sub_size
    /// ```
    ///
    /// Without a frame pointer (a leaf, see [`layout_frame_with`]):
    ///
    /// ```text
    /// push <callee-saved>... ; sub rsp, sub_size
    /// ```
    ///
    /// Windows x64, ordered as its unwind codes require (pushes, then the
    /// fixed allocation, then the frame register, then the `xmm` saves), with
    /// `rbp` landing on the same saved-`rbp` slot so every `rbp`-relative
    /// offset is the same as System V's:
    ///
    /// ```text
    /// push rbp ; push <callee-saved>... ; sub rsp, fixed
    /// lea rbp, [rsp + cs_bytes + fixed]        ; UWOP_SET_FPREG
    /// movups [rbp - k], xmm6..15               ; UWOP_SAVE_XMM128
    /// sub rsp, sub_size - fixed                ; probed; unwound through rbp
    /// ```
    pub fn prologue_plan(&self) -> Vec<(MachineInst, FrameOp)> {
        let push = |r: u8| (MachineInst::new(X86Op::Push.opcode(), vec![phys_use(u16::from(r))]), FrameOp::Push(r));
        let sub = |n: i32, probe: bool, touch: bool| {
            MachineInst::new(
                X86Op::SubRsp.opcode(),
                vec![imm_op(n as u64), imm_op(u64::from(probe)), imm_op(u64::from(touch))],
            )
        };
        let rbp = RBP as u8;
        if self.fpo {
            let mut plan: Vec<_> = self.cs_regs.iter().map(|&r| push(r)).collect();
            if self.sub_size > 0 {
                plan.push((sub(self.sub_size, self.probes, false), FrameOp::Alloc(self.sub_size as u32)));
            }
            return plan;
        }
        let mut plan = vec![push(rbp)];
        let set_frame = |offset: i32| {
            let inst = if offset == 0 {
                MachineInst::new(X86Op::MovRbpRsp.opcode(), Vec::new())
            } else {
                MachineInst::new(
                    X86Op::LeaRspOff.opcode(),
                    vec![phys(RBP), MachineOperand::Imm(puremp::Int::from_i64(i64::from(offset)))],
                )
            };
            (inst, FrameOp::SetFrame { reg: rbp, offset: offset as u32 })
        };
        if !self.windows {
            plan.push(set_frame(0));
            plan.extend(self.cs_regs.iter().map(|&r| push(r)));
            if self.sub_size > 0 {
                plan.push((sub(self.sub_size, self.probes, false), FrameOp::Alloc(self.sub_size as u32)));
            }
            return plan;
        }
        plan.extend(self.cs_regs.iter().map(|&r| push(r)));
        if self.fixed > 0 {
            plan.push((sub(self.fixed, false, false), FrameOp::Alloc(self.fixed as u32)));
        }
        plan.push(set_frame(self.cs_bytes + self.fixed));
        for &(x, off) in &self.xmm_saves {
            plan.push((
                MachineInst::new(
                    X86Op::SaveXmm.opcode(),
                    vec![
                        MachineOperand::Use(Reg::Physical(regs::xmm(u16::from(x)))),
                        MachineOperand::Imm(puremp::Int::from_i64(i64::from(off))),
                    ],
                ),
                FrameOp::SaveXmm { reg: x, fp_offset: off },
            ));
        }
        let rest = self.sub_size - self.fixed;
        if rest > 0 {
            // The fixed part moved rsp below the last push without touching
            // the stack: with probes, touch the new top before stepping down
            // (the probing invariant of `codegen::stack`).
            let touch = self.probes && self.fixed > 0 && self.sub_size as u64 >= STACK_PROBE_INTERVAL;
            plan.push((sub(rest, self.probes, touch), FrameOp::Alloc(rest as u32)));
        }
        plan
    }
}

/// Round `value` up to a multiple of `align` (a power of two ≥ 1).
fn align_up(value: i64, align: i64) -> i64 {
    (value + align - 1) / align * align
}

/// Compute the frame layout of an allocated machine function, with the default
/// [`CodegenOptions`] (stack probes on).
pub fn layout_frame(mf: &MachineFunction, target: &X86_64Target) -> FrameLayout {
    layout_frame_with(mf, target, &CodegenOptions::default())
}

/// Compute the frame layout of an allocated machine function under `opts`.
pub fn layout_frame_with(
    mf: &MachineFunction,
    target: &X86_64Target,
    opts: &CodegenOptions,
) -> FrameLayout {
    use crate::codegen::target::MachineTarget;

    // Which registers does the function write? Every physical def after
    // allocation, plus the temporaries an encoder expansion uses internally.
    let mut used_gpr = [false; 16];
    let mut used_xmm = [false; 16];
    let mut mark = |p: crate::codegen::mir::PReg| match p.class {
        RegClass::Gpr => used_gpr[p.num as usize] = true,
        RegClass::Fp => used_xmm[p.num as usize] = true,
    };
    for bid in mf.block_ids() {
        for inst in &mf.block(bid).insts {
            for d in inst.defs() {
                if let Reg::Physical(p) = d {
                    mark(p);
                }
            }
            for p in hidden_clobbers(inst) {
                mark(p);
            }
        }
    }
    let callee = target.callee_saved();
    let cs_regs: Vec<u8> = callee
        .iter()
        .filter(|p| p.class == RegClass::Gpr && used_gpr[p.num as usize])
        .map(|p| p.num as u8)
        .collect();
    let cs_bytes = (cs_regs.len() * 8) as i32;
    let xmm_cs: Vec<u8> = callee
        .iter()
        .filter(|p| p.class == RegClass::Fp && used_xmm[p.num as usize])
        .map(|p| p.num as u8)
        .collect();
    // Windows: an 8-byte pad below an odd number of pushes keeps the xmm save
    // slots 16-byte aligned and makes `cs_bytes + fixed` (rbp's distance from
    // rsp when the prologue sets it) a multiple of 16, as UWOP_SET_FPREG needs.
    let windows = opts.os == crate::target::TargetOs::Windows;
    let pad = if windows { cs_bytes % 16 } else { 0 };
    let xmm_saves: Vec<(u8, i32)> = xmm_cs
        .iter()
        .enumerate()
        .map(|(k, &x)| (x, -(cs_bytes + pad + 16 * (k as i32 + 1))))
        .collect();
    let fixed = if windows { pad + 16 * xmm_saves.len() as i32 } else { 0 };

    // A leaf function (no call, no `dyn_alloca`, no inline asm, which could
    // name `rbp`) needs no frame pointer: `rsp` is constant after the
    // prologue, so slots are addressed from it. The unwind tables describe
    // a frame on `rsp` only while nothing is saved or allocated (it is the
    // entry state), so a frame that needs either keeps `rbp` when they are
    // requested.
    let mut leaf = !windows;
    let mut dynamic = false;
    for bid in mf.block_ids() {
        for inst in &mf.block(bid).insts {
            match X86Op::decode(inst.opcode) {
                X86Op::Call | X86Op::TlsGd | X86Op::InlineAsm | X86Op::SaveXmm => leaf = false,
                X86Op::DynAlloca => {
                    leaf = false;
                    dynamic = true;
                }
                _ => {}
            }
        }
    }
    if leaf {
        // Slots grow upward from the final `rsp`, which is 16-byte aligned
        // once anything is allocated (the return address and the pushes
        // above it are accounted for).
        let mut top = 0i64;
        let mut slot_off = vec![0i32; mf.frame().len()];
        for (i, off_slot) in slot_off.iter_mut().enumerate() {
            let info = mf.frame().slot(StackSlot::from_index(i));
            let at = align_up(top, info.align.max(1) as i64);
            *off_slot = at as i32;
            top = at + info.size as i64;
        }
        let sub_size = if top == 0 { 0 } else { (align_up(8 + cs_bytes as i64 + top, 16) - 8 - cs_bytes as i64) as i32 };
        if opts.unwind_tables() == crate::codegen::unwind::UnwindTables::None || (cs_regs.is_empty() && sub_size == 0) {
            return FrameLayout {
                slot_off,
                cs_regs,
                xmm_saves,
                cs_bytes,
                sub_size,
                outgoing: 0,
                probes: opts.stack_probes,
                windows,
                fixed,
                fpo: true,
                dynamic,
            };
        }
    }

    // Slot offsets grow downward from just below the callee-saved region (the
    // pushed GPRs, then any xmm save slots). The alignment is taken relative
    // to `rbp`, which the prologue leaves 16-byte aligned (`push rbp` right
    // after the call's return address), so a slot of alignment up to 16 (a
    // vector) is really aligned whatever the number of callee-saved pushes.
    let mut off = i64::from(pad) + 16 * xmm_saves.len() as i64;
    let mut slot_off = vec![0i32; mf.frame().len()];
    for (i, off_slot) in slot_off.iter_mut().enumerate() {
        let info = mf.frame().slot(StackSlot::from_index(i));
        let below_rbp = align_up(cs_bytes as i64 + off + info.size as i64, (info.align.max(1)) as i64);
        off = below_rbp - cs_bytes as i64;
        *off_slot = -below_rbp as i32;
    }
    let locals = off;
    // The outgoing stack-argument area sits at the very bottom of the frame
    // (`[rsp .. rsp + outgoing)`), below the spill/alloca locals. rsp is constant
    // after the prologue, so isel addresses stack arguments as `[rsp + k]`.
    let outgoing = mf.frame().outgoing() as i64;
    let total = cs_bytes as i64 + locals + outgoing;
    let padded = align_up(total, 16);
    let sub_size = (padded - cs_bytes as i64) as i32;

    FrameLayout {
        slot_off,
        cs_regs,
        xmm_saves,
        cs_bytes,
        sub_size,
        outgoing,
        probes: opts.stack_probes,
        windows,
        fixed,
        fpo: false,
        dynamic,
    }
}

/// The registers an instruction's encoder expansion writes without naming
/// them as operands (so the frame layout still sees a callee-saved one used):
/// the unsigned 64-bit float conversions borrow `rbx`/`r10`/`r11` and
/// `xmm13..xmm15` (see `u64tof`/`fptou64`).
fn hidden_clobbers(inst: &MachineInst) -> Vec<crate::codegen::mir::PReg> {
    let flag = |i: usize| inst.operands.get(i).map_or(0, uimm);
    match X86Op::decode(inst.opcode) {
        // A vector op may borrow a free scratch xmm (the first of xmm15, xmm14,
        // xmm13 it does not name): a non-commutative op whose destination is
        // its second source, and a constant with a nonzero high half. (These
        // are callee-saved on Win64.)
        X86Op::VOp | X86Op::LoadVConst => vec![regs::xmm(13), regs::xmm(14), regs::xmm(15)],
        X86Op::CvtSi2f if flag(3) & 0b100 != 0 => {
            vec![regs::gpr(regs::RBX), regs::gpr(regs::R10), regs::gpr(regs::R11)]
        }
        X86Op::CvtF2si if flag(3) & 0b10 != 0 => vec![
            regs::gpr(regs::R11),
            regs::xmm(13),
            regs::xmm(14),
            regs::xmm(15),
        ],
        _ => Vec::new(),
    }
}

fn phys(r: u16) -> MachineOperand {
    MachineOperand::Def(Reg::Physical(regs::gpr(r)))
}
fn phys_use(r: u16) -> MachineOperand {
    MachineOperand::Use(Reg::Physical(regs::gpr(r)))
}
fn imm_op(v: u64) -> MachineOperand {
    MachineOperand::Imm(puremp::Int::from_u64(v))
}

/// Splice the prologue into the entry block and an epilogue before every `ret`.
///
/// When several `ret`s would each repeat an epilogue longer than a short
/// jump, they jump to one shared epilogue block instead.
pub fn insert_prologue_epilogue(mf: &mut MachineFunction, layout: &FrameLayout) {
    let entry = mf.entry().expect("a function being compiled has an entry block");

    // --- prologue: see `FrameLayout::prologue_plan` ---
    let mut prologue: Vec<MachineInst> = layout.prologue_plan().into_iter().map(|(inst, _)| inst).collect();
    let old = std::mem::take(&mut mf.block_mut(entry).insts);
    prologue.extend(old);
    mf.block_mut(entry).insts = prologue;

    // --- the epilogue before each Ret (see `FrameLayout::epilogue`) ---
    let epilogue = layout.epilogue();
    let rets = mf
        .block_ids()
        .flat_map(|b| mf.block(b).insts.iter())
        .filter(|i| X86Op::decode(i.opcode) == X86Op::Ret)
        .count();
    // Its size in bytes, roughly: whether it beats a 2-byte `jmp`.
    let size: usize = epilogue
        .iter()
        .map(|i| match X86Op::decode(i.opcode) {
            X86Op::Leave => 1,
            X86Op::Pop => 1 + usize::from(rnum(&i.operands[0]) >= 8),
            _ => 4,
        })
        .sum();
    let shared = (rets > 1 && size > 2).then(|| {
        let b = mf.add_block();
        let mut insts = epilogue.clone();
        insts.push(MachineInst::new(X86Op::Ret.opcode(), Vec::new()));
        mf.block_mut(b).insts = insts;
        b
    });
    let block_ids: Vec<_> = mf.block_ids().collect();
    for bid in block_ids {
        if Some(bid) == shared {
            continue;
        }
        let old = std::mem::take(&mut mf.block_mut(bid).insts);
        let mut new_insts = Vec::with_capacity(old.len());
        for inst in old {
            if X86Op::decode(inst.opcode) == X86Op::Ret {
                if let Some(epi) = shared {
                    new_insts.push(
                        MachineInst::new(X86Op::Jmp.opcode(), vec![MachineOperand::Label(epi)]).with_line(inst.line),
                    );
                    continue;
                }
                new_insts.extend(epilogue.iter().cloned());
            }
            new_insts.push(inst);
        }
        mf.block_mut(bid).insts = new_insts;
    }
}

// ===========================================================================
// Block layout
// ===========================================================================

/// Drop the instructions (before allocation) that only compute values
/// nothing reads: copies and constants left on edges into block parameters
/// that are never used, and the arithmetic feeding only them. Only
/// side-effect-free opcodes whose every definition is a dead vreg go;
/// repeated until nothing changes.
pub fn remove_dead_defs(mf: &mut MachineFunction) {
    use X86Op::*;
    let pure = |op: X86Op| {
        matches!(
            op,
            MovRR | MovRI | Add | Sub | And | Or | Xor | Imul | ShlI | ShrI | SarI | AluRI | ImulRI | SetccCmp
                | SetccCmpI | Movsx | Movzx | LeaFrame | GlobalAddr | FuncAddr | LeaRbpOff | LoadFConst | FAdd
                | FSub | FMul | FDiv | FXor | Cvtsd2ss | Cvtss2sd
        )
    };
    loop {
        let live = regalloc::compute_liveness(mf);
        let mut changed = false;
        let block_ids: Vec<_> = mf.block_ids().collect();
        for bid in block_ids {
            let mut alive = live.live_out[bid.index()].clone();
            let insts = std::mem::take(&mut mf.block_mut(bid).insts);
            let mut kept: Vec<MachineInst> = Vec::with_capacity(insts.len());
            for inst in insts.into_iter().rev() {
                let dead = pure(X86Op::decode(inst.opcode))
                    && inst.defs().next().is_some()
                    && inst.defs().all(|d| matches!(d, Reg::Virtual(v) if !alive.contains(&v)));
                if dead {
                    changed = true;
                    continue;
                }
                for d in inst.defs() {
                    if let Reg::Virtual(v) = d {
                        alive.remove(&v);
                    }
                }
                for u in inst.uses() {
                    if let Reg::Virtual(v) = u {
                        alive.insert(v);
                    }
                }
                kept.push(inst);
            }
            kept.reverse();
            mf.block_mut(bid).insts = kept;
        }
        if !changed {
            return;
        }
    }
}

/// Drop the register copies allocation made into no-ops (`mov r, r`).
pub fn remove_nop_moves(mf: &mut MachineFunction) {
    let block_ids: Vec<_> = mf.block_ids().collect();
    for bid in block_ids {
        mf.block_mut(bid).insts.retain(|i| {
            !(X86Op::decode(i.opcode) == X86Op::MovRR
                && matches!(
                    (&i.operands[0], &i.operands[1]),
                    (MachineOperand::Def(Reg::Physical(d)), MachineOperand::Use(Reg::Physical(s))) if d == s
                ))
        });
    }
}

/// Thread jumps: every branch to a block that is nothing but a `jmp` goes
/// straight to its final target, and a conditional branch whose targets
/// coincide becomes a `jmp`. (Copies that allocation coalesced away leave
/// such blocks behind on the edges they were split into.)
pub fn thread_jumps(mf: &mut MachineFunction) {
    let n = mf.num_blocks();
    let entry = mf.entry();
    let jump_of = |b: MBlockId| -> Option<MBlockId> {
        match mf.block(b).insts.as_slice() {
            [j] if X86Op::decode(j.opcode) == X86Op::Jmp && Some(b) != entry => match j.operands[0] {
                MachineOperand::Label(t) => Some(t),
                _ => None,
            },
            _ => None,
        }
    };
    let fin: Vec<MBlockId> = (0..n)
        .map(|i| {
            let mut b = MBlockId::from_index(i);
            // A cycle of empty jumps (an infinite loop) stays as it is.
            for _ in 0..n {
                match jump_of(b) {
                    Some(t) if t != b => b = t,
                    _ => break,
                }
            }
            b
        })
        .collect();
    let block_ids: Vec<_> = mf.block_ids().collect();
    for bid in block_ids {
        for inst in &mut mf.block_mut(bid).insts {
            for op in &mut inst.operands {
                if let MachineOperand::Label(t) = op {
                    *t = fin[t.index()];
                }
            }
            let labels: Vec<MBlockId> = inst.labels().collect();
            let two_way = matches!(X86Op::decode(inst.opcode), X86Op::BrCond | X86Op::CmpBr | X86Op::CmpBrI);
            if two_way && labels[0] == labels[1] {
                *inst = MachineInst::new(X86Op::Jmp.opcode(), vec![MachineOperand::Label(labels[0])])
                    .with_line(inst.line);
            }
        }
    }
}

/// The order blocks are laid out in: the entry first, then each block's
/// preferred successor (a jump's target, a conditional branch's false then
/// true target, a switch's default) right after it when still unplaced, so
/// that branch can fall through; otherwise the next unplaced block in arena
/// order. Blocks unreachable from the entry are left out.
pub fn block_order(mf: &MachineFunction) -> Vec<MBlockId> {
    let n = mf.num_blocks();
    let Some(entry) = mf.entry() else { return Vec::new() };
    let mut reachable = vec![false; n];
    let mut work = vec![entry];
    while let Some(b) = work.pop() {
        if !reachable[b.index()] {
            reachable[b.index()] = true;
            work.extend(mf.block(b).successors());
        }
    }
    let mut placed = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut scan = 0usize;
    let mut cur = Some(entry);
    while let Some(b) = cur {
        placed[b.index()] = true;
        order.push(b);
        let prefer: Vec<MBlockId> = match mf.block(b).insts.last() {
            Some(t) => {
                let labels: Vec<MBlockId> = t.labels().collect();
                match X86Op::decode(t.opcode) {
                    X86Op::Jmp => labels,
                    X86Op::BrCond | X86Op::CmpBr | X86Op::CmpBrI => labels.into_iter().rev().collect(),
                    X86Op::Switch | X86Op::Switch128 => labels.into_iter().take(1).collect(),
                    _ => Vec::new(),
                }
            }
            None => Vec::new(),
        };
        cur = prefer.into_iter().find(|s| !placed[s.index()]);
        if cur.is_none() {
            while scan < n && (placed[scan] || !reachable[scan]) {
                scan += 1;
            }
            cur = (scan < n).then(|| MBlockId::from_index(scan));
        }
    }
    order
}

// ===========================================================================
// Instruction encoding
// ===========================================================================

fn rnum(op: &MachineOperand) -> u8 {
    match op {
        MachineOperand::Def(Reg::Physical(p)) | MachineOperand::Use(Reg::Physical(p)) => {
            p.num as u8
        }
        other => panic!("expected a physical register operand, found {other:?}"),
    }
}

fn iimm(op: &MachineOperand) -> i64 {
    match op {
        MachineOperand::Imm(v) => v.to_i64().unwrap_or(0),
        other => panic!("expected an immediate operand, found {other:?}"),
    }
}

fn uimm(op: &MachineOperand) -> u64 {
    match op {
        MachineOperand::Imm(v) => v.to_u64().or_else(|| v.to_i64().map(|i| i as u64)).unwrap_or(0),
        other => panic!("expected an immediate operand, found {other:?}"),
    }
}

/// What the encoder needs to resolve non-local references while emitting.
struct EncodeCtx<'a> {
    labels: &'a [crate::mc::emit::Label],
    layout: &'a FrameLayout,
    func_name: &'a dyn Fn(u32) -> String,
    global_name: &'a dyn Fn(u32) -> String,
    /// Which symbols' addresses must be loaded from the GOT (position-
    /// independent code; see [`crate::codegen::linkage`]).
    got: GotQuery<'a>,
    /// The function's inline asm statements ([`X86Op::InlineAsm`]).
    asm: &'a [crate::codegen::mir::MachineAsm],
    /// The function's own symbol name and IR index (for inline asm).
    self_name: String,
    source: u32,
    /// The block laid out right after the one being encoded (by index): a
    /// branch to it falls through.
    next: Option<usize>,
}

/// A branch to block `t`, unless it is laid out next.
fn goto(e: &mut Emitter, ctx: &EncodeCtx<'_>, t: usize) {
    if ctx.next != Some(t) {
        jmp(e, ctx.labels[t]);
    }
}

/// Branch to block `t` on condition `cc` (after the flags are set), else to
/// `f`, letting whichever of them is laid out next fall through: `jcc t`,
/// `jncc f` (the inverse condition is `cc ^ 1`) or `jcc t; jmp f`.
fn branch_cc(e: &mut Emitter, ctx: &EncodeCtx<'_>, cc: u8, t: usize, f: usize) {
    if t == f {
        goto(e, ctx, t);
    } else if ctx.next == Some(t) {
        jcc(e, cc ^ 1, ctx.labels[f]);
    } else {
        jcc(e, cc, ctx.labels[t]);
        goto(e, ctx, f);
    }
}

/// Per-function and per-global "address through the GOT?" predicates, by IR
/// index. Both are constant `false` for position-dependent code.
#[derive(Clone, Copy)]
struct GotQuery<'a> {
    func: &'a dyn Fn(u32) -> bool,
    global: &'a dyn Fn(u32) -> bool,
}

impl GotQuery<'static> {
    /// Position-dependent code: no symbol goes through the GOT.
    const NONE: GotQuery<'static> = GotQuery { func: &|_| false, global: &|_| false };
}

/// `lea d, [rip + sym]` (`R_X86_64_PC32`), or with `via_got`
/// `mov d, [rip + sym@GOTPCREL]` (`R_X86_64_GOTPCREL`): the address of a
/// symbol, directly or from its GOT slot.
fn symbol_addr(e: &mut Emitter, d: u8, sym: String, via_got: bool) {
    e.u8(rex(true, d >= 8, false, false));
    e.u8(if via_got { 0x8B } else { 0x8D });
    e.u8(modrm(0, d, 5));
    let kind = if via_got { RelocKind::GotPcRel } else { RelocKind::Pc32 };
    e.reference(kind, Ref::Symbol(sym), 0);
}

/// `cmp r, value` for any 64-bit `value`: `cmp r, imm32` when it
/// sign-extends from 32 bits, else through the `r11` scratch.
fn cmp_r_u64(e: &mut Emitter, r: u8, value: u64) {
    match i32::try_from(value as i64) {
        Ok(v) => cmp_ri(e, r, v, true),
        Err(_) => {
            let tmp = regs::R11 as u8;
            mov_ri(e, tmp, value);
            alu_rr(e, 0x39, r, tmp, true); // cmp r, r11
        }
    }
}

/// `mov d, fs:[0]` (`64 REX.W 8B /r` with an absolute `disp32` of 0): the
/// thread pointer, read from the self pointer at the start of the thread
/// control block (x86-64 TLS ABI).
fn tls_thread_pointer(e: &mut Emitter, d: u8) {
    e.u8(0x64);
    e.u8(rex(true, d >= 8, false, false));
    e.u8(0x8B);
    e.u8(modrm(0, d, 4));
    e.u8(sib(0, 4, 5));
    e.u32(0);
}

/// The two-address expansion of a commutative ALU op `d = a OP b`.
fn bin_commutative(e: &mut Emitter, opcode: u8, d: u8, a: u8, b: u8, w: bool) {
    if d == a {
        alu_rr(e, opcode, d, b, w);
    } else if d == b {
        alu_rr(e, opcode, d, a, w);
    } else {
        mov_rr(e, d, a, w);
        alu_rr(e, opcode, d, b, w);
    }
}

/// The two-address expansion of `d = a * b` (imul, commutative).
fn bin_imul(e: &mut Emitter, d: u8, a: u8, b: u8, w: bool) {
    if d == a {
        imul_rr(e, d, b, w);
    } else if d == b {
        imul_rr(e, d, a, w);
    } else {
        mov_rr(e, d, a, w);
        imul_rr(e, d, b, w);
    }
}

/// The two-address expansion of `d = a - b` (sub, non-commutative).
fn bin_sub(e: &mut Emitter, d: u8, a: u8, b: u8, w: bool) {
    if d == a {
        alu_rr(e, 0x29, d, b, w);
    } else if d == b {
        // d holds b: `sub d, a` gives b - a, `neg d` flips to a - b.
        alu_rr(e, 0x29, d, a, w);
        neg_r(e, d, w);
    } else {
        mov_rr(e, d, a, w);
        alu_rr(e, 0x29, d, b, w);
    }
}

fn shift(e: &mut Emitter, ext: u8, ops: &[MachineOperand], cl: bool) {
    let d = rnum(&ops[0]);
    let a = rnum(&ops[1]);
    let w = iimm(ops.last().unwrap()) == 64;
    if d != a {
        mov_rr(e, d, a, w);
    }
    if cl {
        shift_cl(e, ext, d, w);
    } else {
        let count = uimm(&ops[2]) as u8;
        shift_imm(e, ext, d, count, w);
    }
}

fn encode_load(e: &mut Emitter, ops: &[MachineOperand]) {
    let d = rnum(&ops[0]);
    let ptr = rnum(&ops[1]);
    let size = uimm(&ops[2]);
    if rclass(&ops[0]) == RegClass::Fp {
        // movss (4-byte) / movsd (8-byte) / movdqu (16-byte vector) load.
        if size == 16 {
            sse_mem(e, 0xF3, 0x6F, d, ptr, 0);
        } else {
            sse_mem(e, scalar_prefix(size != 4), 0x10, d, ptr, 0);
        }
        return;
    }
    match size {
        1 => mem(e, &[0x0F, 0xB6], d, ptr, 0, false, false),
        2 => mem(e, &[0x0F, 0xB7], d, ptr, 0, false, false),
        4 => mem(e, &[0x8B], d, ptr, 0, false, false),
        _ => mem(e, &[0x8B], d, ptr, 0, true, false),
    }
}

fn encode_store(e: &mut Emitter, ops: &[MachineOperand]) {
    let ptr = rnum(&ops[0]);
    let val = rnum(&ops[1]);
    let size = uimm(&ops[2]);
    if rclass(&ops[1]) == RegClass::Fp {
        // movss / movsd store from an xmm register (store opcode 0x11), or a
        // 16-byte vector with movdqu.
        if size == 16 {
            sse_mem(e, 0xF3, 0x7F, val, ptr, 0);
        } else {
            sse_mem(e, scalar_prefix(size != 4), 0x11, val, ptr, 0);
        }
        return;
    }
    match size {
        1 => mem(e, &[0x88], val, ptr, 0, false, val >= 4),
        2 => {
            e.u8(0x66);
            mem(e, &[0x89], val, ptr, 0, false, false);
        }
        4 => mem(e, &[0x89], val, ptr, 0, false, false),
        _ => mem(e, &[0x89], val, ptr, 0, true, false),
    }
}

/// Emit a (optionally `lock`-prefixed) `op [base], reg` with the operand size
/// `size` bytes: the byte form uses `op8`, the others `op`, with a `66` prefix
/// for 16 bits and `REX.W` for 64. The byte form forces a `REX` when `reg` is
/// 4..7 so it names `spl`/`bpl`/`sil`/`dil` rather than `ah`..`bh`. The legacy
/// prefixes (`66`, `F0`; any order is legal, we use the assemblers' `66 F0`)
/// precede the `REX` byte, as the encoding requires.
fn atomic_mem_rr(e: &mut Emitter, lock: bool, op8: &[u8], op: &[u8], reg: u8, base: u8, size: u64) {
    if size == 2 {
        e.u8(0x66);
    }
    if lock {
        e.u8(0xF0);
    }
    match size {
        1 => mem(e, op8, reg, base, 0, false, reg >= 4),
        2 => mem(e, op, reg, base, 0, false, false),
        4 => mem(e, op, reg, base, 0, false, false),
        _ => mem(e, op, reg, base, 0, true, false),
    }
}

/// Test hook: [`atomic_mem_rr`] for the three atomic opcode families, named by
/// the byte of their full-width form (`0xB1` cmpxchg, `0xC1` xadd, `0x87`
/// xchg).
#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn atomic_mem_rr_for_test(e: &mut Emitter, lock: bool, op: u8, reg: u8, base: u8, size: u64) {
    let (op8, opw): (&[u8], &[u8]) = match op {
        0xB1 => (&[0x0F, 0xB0], &[0x0F, 0xB1]),
        0xC1 => (&[0x0F, 0xC0], &[0x0F, 0xC1]),
        _ => (&[0x86], &[0x87]),
    };
    atomic_mem_rr(e, lock, op8, opw, reg, base, size);
}

/// Expand [`X86Op::RmwLoop`] — `[Def rax, Def tmp, Use ptr, Use val, Imm size,
/// Imm op]` — into a `lock cmpxchg` retry loop:
///
/// ```text
///     mov{zx} eax/rax, [ptr]        ; old
/// L:  mov tmp, rax
///     <tmp = op(tmp, val)>          ; and/or/xor, and+not, or cmp+cmov
///     lock cmpxchg [ptr], tmp       ; if [ptr] == old: [ptr] = tmp, else rax = [ptr]
///     jne L
/// ```
///
/// On exit `rax` holds the value the successful exchange replaced (the old
/// value). Registers narrower than 64 bits carry garbage above the width; every
/// step only consumes the low `8 * size` bits (the sized `cmp` and `cmpxchg`),
/// so that is harmless.
fn encode_rmw_loop(e: &mut Emitter, ops: &[MachineOperand]) {
    use crate::ir::inst::RmwOp;
    let rax = regs::RAX as u8;
    let tmp = rnum(&ops[1]);
    let ptr = rnum(&ops[2]);
    let val = rnum(&ops[3]);
    let size = uimm(&ops[4]);
    let op = RmwOp::from_code(uimm(&ops[5])).expect("RmwLoop carries a valid rmw code");
    let width = (8 * size) as u32;
    encode_load(e, &[ops[0].clone(), ops[2].clone(), MachineOperand::Imm(puremp::Int::from_u64(size))]);
    let top = e.create_label();
    e.bind_label(top);
    mov_rr(e, tmp, rax, true);
    match op {
        RmwOp::And => alu_rr(e, 0x21, tmp, val, true),
        RmwOp::Or => alu_rr(e, 0x09, tmp, val, true),
        RmwOp::Xor => alu_rr(e, 0x31, tmp, val, true),
        RmwOp::Nand => {
            alu_rr(e, 0x21, tmp, val, true);
            // not tmp (F7 /2)
            e.u8(rex(true, false, false, tmp >= 8));
            e.u8(0xF7);
            e.u8(modrm(3, 2, tmp));
        }
        RmwOp::Max | RmwOp::Min | RmwOp::UMax | RmwOp::UMin => {
            // cmp old, val at the access width; take `val` when it wins.
            cmp_rr_width(e, rax, val, width);
            let cc = match op {
                RmwOp::Max => 0xC,  // old <  val (signed)   -> val
                RmwOp::Min => 0xF,  // old >  val (signed)   -> val
                RmwOp::UMax => 0x2, // old <  val (unsigned) -> val
                _ => 0x7,           // old >  val (unsigned) -> val
            };
            cmov_rr(e, cc, tmp, val, true);
        }
        RmwOp::Xchg | RmwOp::Add | RmwOp::Sub => {
            unreachable!("xchg/add/sub lower to xchg / lock xadd, not a loop")
        }
    }
    atomic_mem_rr(e, true, &[0x0F, 0xB0], &[0x0F, 0xB1], tmp, ptr, size);
    jcc(e, 0x5, top); // jne top
}

/// Encode one machine instruction into `e`.
fn encode_inst(e: &mut Emitter, inst: &MachineInst, ctx: &EncodeCtx<'_>) {
    let ops = &inst.operands;
    match X86Op::decode(inst.opcode) {
        X86Op::InlineAsm => {
            let id = uimm(&ops[0]) as u32;
            super::isel::encode_inline_asm(
                e,
                inst,
                &ctx.asm[id as usize],
                &ctx.self_name,
                (u64::from(ctx.source) << 16) | u64::from(id),
                ctx.global_name,
                ctx.func_name,
            );
        }
        X86Op::MovRR => {
            let d = rnum(&ops[0]);
            let s = rnum(&ops[1]);
            let (dc, sc) = (rclass(&ops[0]), rclass(&ops[1]));
            if dc != sc {
                // A cross-class copy of the raw 64 bits: `movq gpr, xmm`
                // (`66 REX.W 0F 7E /r`) or `movq xmm, gpr` (`66 REX.W 0F 6E /r`).
                // Win64 variadic calls pass a float in both registers.
                if dc == RegClass::Gpr {
                    sse_rr(e, 0x66, true, 0x7E, s, d);
                } else {
                    sse_rr(e, 0x66, true, 0x6E, d, s);
                }
            } else if d == s {
                // A self-move is a no-op regardless of class.
            } else if dc == RegClass::Fp {
                // xmm↔xmm copy of all 128 bits via `movaps` (an xmm may hold a
                // vector; a scalar float lives in the low lane).
                movaps(e, d, s);
            } else {
                mov_rr(e, d, s, true);
            }
        }
        X86Op::MovRI => {
            let d = rnum(&ops[0]);
            match uimm(&ops[1]) {
                // `xor r32, r32` (it clobbers the flags, which no MIR
                // instruction carries across a constant's materialization).
                0 => alu_rr(e, 0x31, d, d, false),
                v => mov_ri(e, d, v),
            }
        }
        X86Op::AluRI => {
            let (d, a, v) = (rnum(&ops[0]), rnum(&ops[1]), iimm(&ops[2]) as i32);
            let ext = uimm(&ops[3]) as u8;
            let w = iimm(&ops[4]) > 32;
            // An add/sub into another register is one `lea d, [a + v]`.
            let disp = match ext {
                0 => Some(v),
                5 => v.checked_neg(),
                _ => None,
            };
            match disp {
                Some(disp) if d != a => mem(e, &[0x8D], d, a, disp, w, false),
                _ => {
                    if d != a {
                        mov_rr(e, d, a, w);
                    }
                    alu_ri(e, ext, d, v, w);
                }
            }
        }
        X86Op::ImulRI => {
            let (d, a, v) = (rnum(&ops[0]), rnum(&ops[1]), iimm(&ops[2]) as i32);
            let w = iimm(&ops[3]) > 32;
            if w || d >= 8 || a >= 8 {
                e.u8(rex(w, d >= 8, false, a >= 8));
            }
            match i8::try_from(v) {
                Ok(b) => {
                    e.u8(0x6B);
                    e.u8(modrm(3, d, a));
                    e.u8(b as u8);
                }
                Err(_) => {
                    e.u8(0x69);
                    e.u8(modrm(3, d, a));
                    e.u32(v as u32);
                }
            }
        }
        X86Op::Add => {
            let w = iimm(&ops[3]) == 64;
            bin_commutative(e, 0x01, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), w);
        }
        X86Op::Sub => {
            let w = iimm(&ops[3]) == 64;
            bin_sub(e, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), w);
        }
        X86Op::And => {
            let w = iimm(&ops[3]) == 64;
            bin_commutative(e, 0x21, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), w);
        }
        X86Op::Or => {
            let w = iimm(&ops[3]) == 64;
            bin_commutative(e, 0x09, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), w);
        }
        X86Op::Xor => {
            let w = iimm(&ops[3]) == 64;
            bin_commutative(e, 0x31, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), w);
        }
        X86Op::Imul => {
            let w = iimm(&ops[3]) == 64;
            bin_imul(e, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), w);
        }
        X86Op::ShlI => shift(e, 4, ops, false),
        X86Op::ShrI => shift(e, 5, ops, false),
        X86Op::SarI => shift(e, 7, ops, false),
        X86Op::ShlCl => shift(e, 4, ops, true),
        X86Op::ShrCl => shift(e, 5, ops, true),
        X86Op::SarCl => shift(e, 7, ops, true),
        X86Op::Cqo => {
            if iimm(&ops[2]) == 64 {
                e.u8(0x48);
            }
            e.u8(0x99);
        }
        X86Op::ZeroRdx => alu_rr(e, 0x31, regs::RDX as u8, regs::RDX as u8, false),
        X86Op::Idiv => divide(e, 7, rnum(&ops[4]), iimm(&ops[5]) == 64),
        X86Op::Div => divide(e, 6, rnum(&ops[4]), iimm(&ops[5]) == 64),
        X86Op::SetccCmp | X86Op::SetccCmpI => {
            let d = rnum(&ops[0]);
            let a = rnum(&ops[1]);
            let cc = uimm(&ops[3]) as u8;
            let width = iimm(&ops[4]) as u32;
            let imm = X86Op::decode(inst.opcode) == X86Op::SetccCmpI;
            // `xor d, d` first when d is no source (the compare's flags must
            // survive), else `movzx` the byte after.
            let fresh = d != a && (imm || d != rnum(&ops[2]));
            if fresh {
                alu_rr(e, 0x31, d, d, false);
            }
            if imm {
                cmp_ri_width(e, a, iimm(&ops[2]), width);
            } else {
                cmp_rr_width(e, a, rnum(&ops[2]), width); // cmp a, b
            }
            setcc(e, cc, d);
            if !fresh {
                movzx_byte(e, d);
            }
        }
        X86Op::CmpBr => {
            cmp_rr_width(e, rnum(&ops[0]), rnum(&ops[1]), iimm(&ops[3]) as u32);
            branch_cc(e, ctx, uimm(&ops[2]) as u8, label_index(&ops[4]), label_index(&ops[5]));
        }
        X86Op::CmpFlags => match &ops[1] {
            MachineOperand::Imm(_) => cmp_ri_width(e, rnum(&ops[0]), iimm(&ops[1]), iimm(&ops[3]) as u32),
            _ => cmp_rr_width(e, rnum(&ops[0]), rnum(&ops[1]), iimm(&ops[3]) as u32),
        },
        X86Op::Cmov => {
            let (d, d2, t) = (rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]));
            if d != d2 {
                mov_rr(e, d, d2, true);
            }
            cmov_rr(e, uimm(&ops[3]) as u8, d, t, true);
        }
        X86Op::CmpBrI => {
            cmp_ri_width(e, rnum(&ops[0]), iimm(&ops[1]), iimm(&ops[3]) as u32);
            branch_cc(e, ctx, uimm(&ops[2]) as u8, label_index(&ops[4]), label_index(&ops[5]));
        }
        X86Op::Test => {
            let r = rnum(&ops[0]);
            alu_rr(e, 0x85, r, r, false);
        }
        X86Op::Cmovne => {
            let d = rnum(&ops[0]);
            let d2 = rnum(&ops[1]);
            let t = rnum(&ops[2]);
            if d != d2 {
                mov_rr(e, d, d2, true);
            }
            cmov_rr(e, 0x5, d, t, true);
        }
        X86Op::Load => encode_load(e, ops),
        X86Op::Store => encode_store(e, ops),
        X86Op::LeaFrame => {
            let d = rnum(&ops[0]);
            let slot = slot_index(&ops[1]);
            mem(e, &[0x8D], d, ctx.layout.base(), ctx.layout.slot_off[slot], true, false);
        }
        X86Op::StoreFrame => {
            let src = rnum(&ops[0]);
            let slot = slot_index(&ops[1]);
            let (base, off) = (ctx.layout.base(), ctx.layout.slot_off[slot]);
            if rclass(&ops[0]) == RegClass::Fp {
                // The whole register (a vector or a scalar float).
                sse_mem(e, 0xF3, 0x7F, src, base, off); // movdqu [base+off], xmm
            } else {
                mem(e, &[0x89], src, base, off, true, false);
            }
        }
        X86Op::LoadFrame => {
            let dst = rnum(&ops[0]);
            let slot = slot_index(&ops[1]);
            let (base, off) = (ctx.layout.base(), ctx.layout.slot_off[slot]);
            if rclass(&ops[0]) == RegClass::Fp {
                sse_mem(e, 0xF3, 0x6F, dst, base, off); // movdqu xmm, [base+off]
            } else {
                mem(e, &[0x8B], dst, base, off, true, false);
            }
        }
        X86Op::GlobalAddr => {
            let d = rnum(&ops[0]);
            let g = match ops[1] {
                MachineOperand::Global(g) => g,
                _ => panic!("GlobalAddr expects a global operand"),
            };
            // lea d, [rip + disp32]  with a PC32 relocation to the global, or
            // under PIC a GOT load for a preemptible/external one.
            symbol_addr(e, d, (ctx.global_name)(g), (ctx.got.global)(g));
        }
        X86Op::TlsAddr => {
            let d = rnum(&ops[0]);
            let MachineOperand::Global(g) = ops[1] else { panic!("TlsAddr expects a global operand") };
            let sym = (ctx.global_name)(g);
            tls_thread_pointer(e, d);
            if uimm(&ops[2]) != 0 {
                // add d, [rip + g@gottpoff]  (initial-exec)
                e.u8(rex(true, d >= 8, false, false));
                e.u8(0x03);
                e.u8(modrm(0, d, 5));
                e.reference(RelocKind::GotTpOff, Ref::Symbol(sym), 0);
            } else {
                // lea d, [d + g@tpoff]  (local-exec)
                e.u8(rex(true, d >= 8, false, d >= 8));
                e.u8(0x8D);
                e.u8(modrm(2, d, d));
                if d & 7 == 4 {
                    e.u8(sib(0, 4, 4)); // r12 as a base needs a SIB
                }
                e.reference(RelocKind::TpOff32, Ref::Symbol(sym), 0);
            }
        }
        X86Op::MulWide => {
            // mul b  (rdx:rax = rax * b, unsigned)
            let b = rnum(&ops[3]);
            e.u8(rex(true, false, false, b >= 8));
            e.u8(0xF7);
            e.u8(modrm(3, 4, b));
        }
        X86Op::Switch128 => {
            let (lo_r, hi_r) = (rnum(&ops[0]), rnum(&ops[1]));
            let default = label_index(&ops[2]);
            let mut i = 3;
            while i + 2 < ops.len() {
                let (vlo, vhi) = (uimm(&ops[i]), uimm(&ops[i + 1]));
                let case = label_index(&ops[i + 2]);
                let next = e.create_label();
                cmp_r_u64(e, lo_r, vlo);
                jcc(e, 0x5, next); // jne next
                cmp_r_u64(e, hi_r, vhi);
                jcc(e, 0x4, ctx.labels[case]); // je case
                e.bind_label(next);
                i += 3;
            }
            goto(e, ctx, default);
        }
        X86Op::TlsGd => {
            let MachineOperand::Global(g) = ops[0] else { panic!("TlsGd expects a global operand") };
            // The canonical general-dynamic sequence (padded with prefixes to
            // 16 bytes so a linker can rewrite it into another model):
            // data16 lea rdi, [rip + g@tlsgd]; data16 data16 rex.w call
            // __tls_get_addr@plt.
            e.u8(0x66);
            e.u8(0x48);
            e.u8(0x8D);
            e.u8(0x3D);
            e.reference(RelocKind::TlsGd, Ref::Symbol((ctx.global_name)(g)), 0);
            e.u8(0x66);
            e.u8(0x66);
            e.u8(0x48);
            e.u8(0xE8);
            e.plt32(Ref::Symbol("__tls_get_addr".to_owned()), 0);
        }
        X86Op::FuncAddr => {
            let d = rnum(&ops[0]);
            let f = match ops[1] {
                MachineOperand::Func(f) => f,
                _ => panic!("FuncAddr expects a Func operand"),
            };
            // lea d, [rip + disp32] with a PC32 relocation to the function
            // symbol — the same materialization as GlobalAddr, but naming a
            // function. Taking a function's address for a function pointer; a
            // *direct* call still uses `E8` + PLT32 (the `Call`/`Func` arm).
            // Under PIC a preemptible/external function's address comes from
            // the GOT, so every component agrees on its canonical address.
            symbol_addr(e, d, (ctx.func_name)(f), (ctx.got.func)(f));
        }
        X86Op::Movsx => {
            let d = rnum(&ops[0]);
            let s = rnum(&ops[1]);
            let src_w = uimm(&ops[2]) as u32;
            let dst_w = uimm(&ops[3]) as u32;
            movsx_rr(e, d, s, src_w, dst_w);
        }
        X86Op::Movzx => {
            let d = rnum(&ops[0]);
            let s = rnum(&ops[1]);
            let src_w = uimm(&ops[2]) as u32;
            movzx_rr(e, d, s, src_w);
        }
        X86Op::Call => match &ops[0] {
            MachineOperand::Func(idx) => {
                e.u8(0xE8);
                e.plt32(Ref::Symbol((ctx.func_name)(*idx)), 0);
            }
            MachineOperand::Use(Reg::Physical(p)) => {
                let r = p.num as u8;
                if r >= 8 {
                    e.u8(0x41);
                }
                e.u8(0xFF);
                e.u8(modrm(3, 2, r));
            }
            other => panic!("Call expects a Func or register operand, found {other:?}"),
        },
        X86Op::Ret => e.u8(0xC3),
        X86Op::SaveXmm => sse_mem(e, 0, 0x11, rnum(&ops[0]), RBP as u8, iimm(&ops[1]) as i32),
        X86Op::RestoreXmm => sse_mem(e, 0, 0x10, rnum(&ops[0]), RBP as u8, iimm(&ops[1]) as i32),
        X86Op::Jmp => goto(e, ctx, label_index(&ops[0])),
        X86Op::BrCond => {
            let cond = rnum(&ops[0]);
            let t = label_index(&ops[1]);
            let f = label_index(&ops[2]);
            alu_rr(e, 0x85, cond, cond, false); // test cond, cond
            branch_cc(e, ctx, 0x5, t, f); // jne t; jmp f
        }
        X86Op::Switch => {
            let cond = rnum(&ops[0]);
            let default = label_index(&ops[1]);
            let mut i = 2;
            while i + 1 < ops.len() {
                let value = iimm(&ops[i]);
                let case = label_index(&ops[i + 1]);
                match i32::try_from(value) {
                    Ok(v) => cmp_ri(e, cond, v, true),
                    Err(_) => {
                        // No `cmp r64, imm64`: materialize the case value in the
                        // r11 scratch first.
                        let tmp = regs::R11 as u8;
                        mov_ri(e, tmp, value as u64);
                        alu_rr(e, 0x39, cond, tmp, true); // cmp cond, r11
                    }
                }
                jcc(e, 0x4, ctx.labels[case]); // je case
                i += 2;
            }
            goto(e, ctx, default);
        }
        X86Op::Unreachable => {
            e.u8(0x0F);
            e.u8(0x0B); // ud2
        }
        X86Op::Syscall => {
            // The operands only document the fixed-register contract for the
            // allocator; the instruction itself is operand-free.
            e.u8(0x0F);
            e.u8(0x05); // syscall
        }
        X86Op::Mfence => e.bytes(&[0x0F, 0xAE, 0xF0]),
        X86Op::Xchg => {
            // mov d, val; xchg [ptr], d  (xchg with memory is implicitly locked)
            let (d, ptr, val, size) = (rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), uimm(&ops[3]));
            if d != val {
                mov_rr(e, d, val, true);
            }
            atomic_mem_rr(e, false, &[0x86], &[0x87], d, ptr, size);
        }
        X86Op::LockXadd => {
            // mov d, val; [neg d;] lock xadd [ptr], d
            let (d, ptr, val, size) = (rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), uimm(&ops[3]));
            if d != val {
                mov_rr(e, d, val, true);
            }
            if uimm(&ops[4]) != 0 {
                neg_r(e, d, true);
            }
            atomic_mem_rr(e, true, &[0x0F, 0xC0], &[0x0F, 0xC1], d, ptr, size);
        }
        X86Op::LockCmpxchg => {
            // lock cmpxchg [ptr], new   (expected/old in rax)
            let (ptr, new, size) = (rnum(&ops[2]), rnum(&ops[3]), uimm(&ops[4]));
            atomic_mem_rr(e, true, &[0x0F, 0xB0], &[0x0F, 0xB1], new, ptr, size);
        }
        X86Op::RmwLoop => encode_rmw_loop(e, ops),
        X86Op::Leave => e.u8(0xC9),
        X86Op::AddRsp => alu_ri(e, 0, RSP as u8, uimm(&ops[0]) as i32, true),
        X86Op::Push => push_r(e, rnum(&ops[0])),
        X86Op::Pop => pop_r(e, rnum(&ops[0])),
        X86Op::MovRbpRsp => mov_rr(e, RBP as u8, RSP as u8, true),
        X86Op::SubRsp => {
            let probe = ops.get(1).is_some_and(|o| uimm(o) != 0);
            if ops.get(2).is_some_and(|o| uimm(o) != 0) {
                probe_rsp(e); // touch the current top first
            }
            sub_rsp(e, uimm(&ops[0]), probe);
        }
        X86Op::LeaRspRbp => {
            let k = uimm(&ops[0]) as i64;
            mem(e, &[0x8D], RSP as u8, RBP as u8, (-k) as i32, true, false);
        }
        X86Op::LeaRbpOff => {
            let d = rnum(&ops[0]);
            let off = iimm(&ops[1]) as i32;
            if ctx.layout.fpo {
                // Where rbp would point: `rsp` plus the frame below it.
                let k = ctx.layout.cs_bytes + ctx.layout.sub_size - 8;
                mem(e, &[0x8D], d, RSP as u8, off + k, true, false); // lea d, [rsp + ..]
            } else {
                mem(e, &[0x8D], d, RBP as u8, off, true, false); // lea d, [rbp + off]
            }
        }
        X86Op::LeaRspOff => {
            let d = rnum(&ops[0]);
            let off = iimm(&ops[1]) as i32;
            mem(e, &[0x8D], d, RSP as u8, off, true, false); // lea d, [rsp + off]
        }
        X86Op::DynAlloca => {
            // Dynamic (runtime-sized) stack allocation. `d` doubles as the size
            // scratch and then receives the result pointer; `n` is the byte count.
            //
            // The moving `rsp` must coexist with the fixed rsp-relative
            // outgoing-argument area, which the framework reserves at the bottom
            // of the frame and addresses as `[rsp + k]`. We keep that area at the
            // bottom of the *current* rsp by relocating it below the carved block:
            // we subtract `outgoing` extra bytes and hand back `rsp + outgoing`, so
            // after the allocation `[rsp, rsp + outgoing)` is again free outgoing
            // space and every prior allocation sits above it, untouched. The
            // rbp-relative epilogue (`lea rsp, [rbp - cs]`) reclaims the whole
            // dynamic region on return.
            let d = rnum(&ops[0]);
            let n = rnum(&ops[1]);
            let align = uimm(&ops[2]);
            let outgoing = ctx.layout.outgoing;
            // Slack so the pointer can be rounded *up* to a large alignment while
            // staying inside the carved region. For align ≤ 16 no slack and no
            // rounding are needed: rsp stays 16-aligned (we subtract a multiple of
            // 16) and `outgoing` is itself a multiple of 16.
            let extra = if align > 16 { align as i64 } else { 0 };
            let c = outgoing + extra; // a multiple of 16
            if d != n {
                mov_rr(e, d, n, true); // d = n
            }
            // d = ((n + 15 + C) & ~15): round the size up to 16 and add the
            // relocated outgoing area + alignment slack (both multiples of 16, so
            // the mask rounds only the `n + 15` part).
            alu_ri32(e, 0, d, (15 + c) as i32, true); // add d, 15 + C
            and_ri8(e, d, -16, true); // and d, ~15
            if ctx.layout.probes {
                // Touch the current top, then move down one probe interval at a
                // time touching each step, leaving a remainder < the interval.
                let step = STACK_PROBE_INTERVAL as i32;
                probe_rsp(e); // or qword [rsp], 0
                alu_ri32(e, 7, d, step, true); // cmp d, 4096
                let done = e.create_label();
                jcc(e, 0x2, done); // jb done
                let top = e.create_label();
                e.bind_label(top);
                sub_rsp_imm(e, STACK_PROBE_INTERVAL as u32); // sub rsp, 4096
                probe_rsp(e); // or qword [rsp], 0
                alu_ri32(e, 5, d, step, true); // sub d, 4096
                alu_ri32(e, 7, d, step, true); // cmp d, 4096
                jcc(e, 0x3, top); // jae top
                e.bind_label(done);
            }
            alu_rr(e, 0x29, RSP as u8, d, true); // sub rsp, d
            // result = rsp + outgoing (its own alignment), rounded up to `align`.
            mem(e, &[0x8D], d, RSP as u8, outgoing as i32, true, false); // lea d,[rsp+outgoing]
            if align > 16 {
                alu_ri32(e, 0, d, (align - 1) as i32, true); // add d, align-1
                alu_ri32(e, 4, d, -(align as i64) as i32, true); // and d, -align
            }
        }

        // --- SSE scalar floating-point ------------------------------------
        X86Op::FAdd => {
            let w = iimm(&ops[3]) == 64;
            fbin(e, w, 0x58, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), true);
        }
        X86Op::FSub => {
            let w = iimm(&ops[3]) == 64;
            fbin(e, w, 0x5C, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), false);
        }
        X86Op::FMul => {
            let w = iimm(&ops[3]) == 64;
            fbin(e, w, 0x59, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), true);
        }
        X86Op::FDiv => {
            let w = iimm(&ops[3]) == 64;
            fbin(e, w, 0x5E, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]), false);
        }
        X86Op::FXor => {
            let w = iimm(&ops[3]) == 64;
            fxor(e, w, rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]));
        }
        X86Op::LoadFConst => {
            let d = rnum(&ops[0]);
            let bits = uimm(&ops[1]);
            let width = iimm(&ops[2]);
            let tmp = regs::R11 as u8;
            if width == 64 {
                mov_ri(e, tmp, bits); // movabs r11, bits
                sse_rr(e, 0x66, true, 0x6E, d, tmp); // movq xmm, r11
            } else {
                mov_ri(e, tmp, bits & 0xFFFF_FFFF); // mov r11d, bits
                sse_rr(e, 0x66, false, 0x6E, d, tmp); // movd xmm, r11d
            }
        }
        X86Op::FCmpSet => {
            let d = rnum(&ops[0]);
            let a = rnum(&ops[1]);
            let b = rnum(&ops[2]);
            let packed = uimm(&ops[3]);
            let width = iimm(&ops[4]);
            let cc = (packed & 0xFF) as u8;
            let swap = (packed >> 8) & 1 != 0;
            let combine = (packed >> 9) & 0x3;
            // ucomisd (prefix 66) / ucomiss (no prefix): opcode 0F 2E, reg,rm.
            let pfx = if width == 64 { 0x66 } else { 0x00 };
            let (reg, rm) = if swap { (b, a) } else { (a, b) };
            sse_rr(e, pfx, false, 0x2E, reg, rm);
            setcc(e, cc, d);
            if combine != 0 {
                let tmp = regs::R11 as u8;
                debug_assert!(d != tmp, "fcmp parity temp must be distinct from the result");
                // combine 1 = AND setnp (0x0B); combine 2 = OR setp (0x0A).
                let (second_cc, alu) = if combine == 1 { (0x0B, 0x20) } else { (0x0A, 0x08) };
                setcc(e, second_cc, tmp);
                alu_byte(e, alu, d, tmp); // and/or d8, r11b
            }
            movzx_byte(e, d);
        }
        // --- SSE2 vectors ---------------------------------------------------
        X86Op::VOp => encode_vop(e, ops),
        X86Op::VUnary => {
            let (prefix, opcode, _, imm8) = super::isel::vector::VEnc::decode(uimm(&ops[2]));
            sse_rr(e, prefix, false, opcode, rnum(&ops[0]), rnum(&ops[1]));
            if let Some(i) = imm8 {
                e.u8(i);
            }
        }
        X86Op::VShiftI => {
            let (d, a) = (rnum(&ops[0]), rnum(&ops[1]));
            let enc = uimm(&ops[2]);
            if d != a {
                movaps(e, d, a);
            }
            // 66 0F 71/72/73 /ext ib: the extension in ModRM.reg, d in rm.
            sse_rr(e, 0x66, false, enc as u8, (enc >> 8) as u8, d);
            e.u8((enc >> 16) as u8);
        }
        X86Op::VLoad => {
            let pfx = if uimm(&ops[2]) != 0 { 0x66 } else { 0xF3 }; // movdqa / movdqu
            sse_mem(e, pfx, 0x6F, rnum(&ops[0]), rnum(&ops[1]), 0);
        }
        X86Op::VStore => {
            let pfx = if uimm(&ops[2]) != 0 { 0x66 } else { 0xF3 };
            sse_mem(e, pfx, 0x7F, rnum(&ops[1]), rnum(&ops[0]), 0);
        }
        X86Op::LoadVConst => {
            let d = rnum(&ops[0]);
            let (lo, hi) = (uimm(&ops[1]), uimm(&ops[2]));
            if lo == 0 && hi == 0 {
                sse_rr(e, 0x66, false, 0xEF, d, d); // pxor d, d
            } else if lo == u64::MAX && hi == u64::MAX {
                sse_rr(e, 0x66, false, 0x76, d, d); // pcmpeqd d, d
            } else {
                let tmp = regs::R11 as u8;
                mov_ri(e, tmp, lo);
                sse_rr(e, 0x66, true, 0x6E, d, tmp); // movq d, r11 (zeroes the top)
                if hi != 0 {
                    let t = free_xmm_scratch(&[d]);
                    mov_ri(e, tmp, hi);
                    sse_rr(e, 0x66, true, 0x6E, t, tmp); // movq t, r11
                    sse_rr(e, 0x66, false, 0x6C, d, t); // punpcklqdq d, t
                }
            }
        }
        X86Op::MovGprToX => sse_rr(e, 0x66, uimm(&ops[2]) != 0, 0x6E, rnum(&ops[0]), rnum(&ops[1])),
        X86Op::MovXToGpr => sse_rr(e, 0x66, uimm(&ops[2]) != 0, 0x7E, rnum(&ops[1]), rnum(&ops[0])),
        X86Op::Pinsrw => {
            let (d, v, g) = (rnum(&ops[0]), rnum(&ops[1]), rnum(&ops[2]));
            if d != v {
                movaps(e, d, v);
            }
            sse_rr(e, 0x66, false, 0xC4, d, g); // pinsrw d, g32, idx
            e.u8(uimm(&ops[3]) as u8);
        }
        X86Op::Pextrw => {
            sse_rr(e, 0x66, false, 0xC5, rnum(&ops[0]), rnum(&ops[1])); // pextrw g32, v, idx
            e.u8(uimm(&ops[2]) as u8);
        }
        X86Op::Cvtsd2ss => sse_rr(e, 0xF2, false, 0x5A, rnum(&ops[0]), rnum(&ops[1])),
        X86Op::Cvtss2sd => sse_rr(e, 0xF3, false, 0x5A, rnum(&ops[0]), rnum(&ops[1])),
        X86Op::CvtF2si => {
            let d = rnum(&ops[0]); // gpr
            let s = rnum(&ops[1]); // xmm
            let src_w = iimm(&ops[2]);
            let flags = uimm(&ops[3]);
            if flags & 0b10 != 0 {
                // Full unsigned float→u64 with the 2^63 fix-up.
                fptou64(e, d, s, src_w == 64);
            } else {
                let pfx = scalar_prefix(src_w == 64);
                let w = flags & 1 != 0;
                sse_rr(e, pfx, w, 0x2C, d, s); // cvttsd2si/cvttss2si d(gpr), s(xmm)
            }
        }
        X86Op::CvtSi2f => {
            let d = rnum(&ops[0]); // xmm
            let s = rnum(&ops[1]); // gpr
            let dst_w = iimm(&ops[2]);
            let flags = uimm(&ops[3]);
            let pfx = scalar_prefix(dst_w == 64);
            if flags & 0b100 != 0 {
                // Full unsigned u64→float with the halve-and-round fix-up.
                u64tof(e, d, s, dst_w == 64);
            } else if flags & 0b10 != 0 {
                // Unsigned ≤32: zero-extend the source into r11, then a 64-bit
                // signed conversion (the value fits in [0, 2^32) ⊂ i64).
                let tmp = regs::R11 as u8;
                debug_assert!(s != tmp, "uitofp zero-extend temp must differ from the source");
                alu_rr(e, 0x89, tmp, s, false); // mov r11d, s  (zero-extends)
                sse_rr(e, pfx, true, 0x2A, d, tmp); // cvtsi2sd xmm, r11
            } else {
                let w = flags & 1 != 0;
                sse_rr(e, pfx, w, 0x2A, d, s); // cvtsi2sd/ss xmm, gpr
            }
        }
    }
}

/// `sub rsp, imm8` / `sub rsp, imm32`.
fn sub_rsp_imm(e: &mut Emitter, n: u32) {
    alu_ri(e, 5, RSP as u8, n as i32, true);
}

/// `or qword [rsp], 0` — a stack probe: a read-modify-write of the new top that
/// faults if it lies in the guard region, and changes nothing otherwise.
fn probe_rsp(e: &mut Emitter) {
    e.bytes(&[0x48, 0x83, modrm(0, 1, 4), sib(0, 4, RSP as u8), 0x00]);
}

/// Pages up to which a probed `sub rsp` is unrolled rather than looped.
const PROBE_UNROLL: u64 = 4;

/// The prologue's stack allocation of `size` bytes: one `sub rsp`, or with
/// `probe` and a size of at least [`STACK_PROBE_INTERVAL`], the probed sequence
/// (see the module docs). `r11` (encoder scratch, never allocated, and free at
/// function entry) counts the loop.
fn sub_rsp(e: &mut Emitter, size: u64, probe: bool) {
    if !probe || size < STACK_PROBE_INTERVAL {
        sub_rsp_imm(e, size as u32);
        return;
    }
    let pages = size / STACK_PROBE_INTERVAL;
    let rem = size % STACK_PROBE_INTERVAL;
    if pages <= PROBE_UNROLL {
        for _ in 0..pages {
            sub_rsp_imm(e, STACK_PROBE_INTERVAL as u32);
            probe_rsp(e);
        }
    } else {
        mov_ri(e, regs::R11 as u8, pages); // mov r11d, pages
        let top = e.offset();
        sub_rsp_imm(e, STACK_PROBE_INTERVAL as u32); // sub rsp, 4096
        probe_rsp(e); // or qword [rsp], 0
        e.bytes(&[rex(true, false, false, true), 0xFF, modrm(3, 1, regs::R11 as u8)]); // dec r11
        let back = top as i64 - (e.offset() as i64 + 2);
        e.bytes(&[0x75, back as i8 as u8]); // jnz top
    }
    if rem > 0 {
        sub_rsp_imm(e, rem as u32);
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
    encode_function_inner(mf, layout, func_name, global_name, GotQuery::NONE, None, None)
}

/// Like [`encode_function`], but also collects the `(function-relative offset,
/// source line)` statement rows for a `.debug_line` program. A row is recorded
/// at the start of each machine instruction whose source line differs from the
/// previous row's; instructions with no line (synthesized prologue/moves) are
/// skipped.
pub fn encode_function_lines(
    mf: &MachineFunction,
    layout: &FrameLayout,
    func_name: &dyn Fn(u32) -> String,
    global_name: &dyn Fn(u32) -> String,
) -> (Emitted, Vec<(u64, u32)>) {
    let mut rows = Vec::new();
    let emitted =
        encode_function_inner(mf, layout, func_name, global_name, GotQuery::NONE, Some(&mut rows), None);
    (emitted, rows)
}

/// Where the frame-changing instructions of an encoded function ended up,
/// for its unwind tables.
#[derive(Debug, Default)]
struct UnwindMarks {
    /// The end offset of each of the entry block's first instructions — the
    /// prologue, one per [`FrameLayout::prologue_plan`] step.
    prologue_ends: Vec<u32>,
    /// Each epilogue's `(end of pop rbp, end of ret)`.
    epilogues: Vec<(u32, u32)>,
}

fn encode_function_inner(
    mf: &MachineFunction,
    layout: &FrameLayout,
    func_name: &dyn Fn(u32) -> String,
    global_name: &dyn Fn(u32) -> String,
    got: GotQuery<'_>,
    lines: Option<&mut Vec<(u64, u32)>>,
    mut marks: Option<(usize, &mut UnwindMarks)>,
) -> Emitted {
    let mut e = Emitter::new();
    let labels: Vec<_> = (0..mf.num_blocks()).map(|_| e.create_label()).collect();
    let mut ctx = EncodeCtx {
        labels: &labels,
        layout,
        func_name,
        global_name,
        got,
        asm: mf.inline_asms(),
        self_name: func_name(mf.info().source),
        source: mf.info().source,
        next: None,
    };

    // The entry block comes first (so the function symbol at offset 0 is the
    // entry), then the others as `block_order` lays them out. Branches are
    // relaxed when the emitter finishes, which moves code: every offset
    // recorded here is a label, read back afterwards.
    let entry = mf.entry().expect("a function being compiled has an entry block");
    let order = block_order(mf);
    let mut rows: Vec<(Label, u32)> = Vec::new();
    let mut prologue_ends: Vec<Label> = Vec::new();
    let mut epilogues: Vec<(Label, Label)> = Vec::new();
    let here = |e: &mut Emitter| {
        let l = e.create_label();
        e.bind_label(l);
        l
    };
    let mut pending_pop = None;
    for (i, &bid) in order.iter().enumerate() {
        ctx.next = order.get(i + 1).map(|b| b.index());
        e.bind_label(labels[bid.index()]);
        for (k, inst) in mf.block(bid).insts.iter().enumerate() {
            if lines.is_some() && inst.line != 0 && rows.last().map(|&(_, l)| l) != Some(inst.line) {
                rows.push((here(&mut e), inst.line));
            }
            encode_inst(&mut e, inst, &ctx);
            if marks.is_some() {
                let op = X86Op::decode(inst.opcode);
                let prologue_len = marks.as_ref().map_or(0, |m| m.0);
                if bid == entry && k < prologue_len {
                    prologue_ends.push(here(&mut e));
                } else if op == X86Op::Leave || (op == X86Op::Pop && rnum(&inst.operands[0]) == RBP as u8) {
                    pending_pop = Some(here(&mut e));
                } else if op == X86Op::Ret {
                    if let Some(pop) = pending_pop.take() {
                        epilogues.push((pop, here(&mut e)));
                    }
                } else {
                    pending_pop = None;
                }
            }
        }
    }
    let (emitted, at) = e.finish_with_labels().expect("intra-function branch resolution never overflows");
    let at = |l: Label| at[l.index()].expect("a bound label");
    if let Some(out) = lines {
        out.extend(rows.iter().map(|&(l, line)| (at(l), line)));
    }
    if let Some((_, m)) = marks.as_mut() {
        m.prologue_ends.extend(prologue_ends.iter().map(|&l| at(l) as u32));
        m.epilogues.extend(epilogues.iter().map(|&(p, r)| (at(p) as u32, at(r) as u32)));
    }
    emitted
}

/// One function's compile output: bytes + relocations, the `.debug_line`
/// statement rows (when requested), and its stack usage.
struct FunctionOutput {
    emitted: Emitted,
    rows: Vec<(u64, u32)>,
    stack: StackUsage,
    /// The unwind description (its `offset` and `size` filled in by the
    /// module driver).
    frame: FunctionFrame,
}

/// Run isel → register allocation → frame layout → prologue/epilogue →
/// encoding for one function under `opts`, collecting line rows if `lines`.
fn compile_function_full(
    module: &Module,
    func: crate::ir::FuncId,
    syms: &StrInterner,
    opts: &CodegenOptions,
    lines: bool,
) -> FunctionOutput {
    let target = X86_64Target::for_os(opts.os).with_reloc_model(opts.reloc_model);
    let mut mf = target.select_with_syms(module, func, syms);
    // Program points follow the block layout, so a value's live range does
    // not span blocks laid out elsewhere.
    remove_dead_defs(&mut mf);
    let order = block_order(&mf);
    regalloc::allocate_with(&mut mf, &target, &regalloc::AllocOptions { order: Some(order), precise: true });
    remove_nop_moves(&mut mf);
    let layout = layout_frame_with(&mf, &target, opts);
    insert_prologue_epilogue(&mut mf, &layout);
    thread_jumps(&mut mf);
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
    let plan = layout.prologue_plan();
    let mut marks = UnwindMarks::default();
    let emitted = encode_function_inner(
        &mf,
        &layout,
        &func_name,
        &global_name,
        GotQuery { func: &got_func, global: &got_global },
        if lines { Some(&mut rows) } else { None },
        Some((plan.len(), &mut marks)),
    );
    // The prologue's steps: the planned frame operations at the offsets their
    // instructions were encoded to.
    debug_assert_eq!(marks.prologue_ends.len(), plan.len(), "the prologue opens the entry block");
    let steps = plan
        .iter()
        .zip(&marks.prologue_ends)
        .map(|((_, op), &end)| FrameStep { end, op: *op })
        .collect();
    let frame = FunctionFrame { offset: 0, size: emitted.bytes.len() as u64, steps, epilogues: marks.epilogues };
    FunctionOutput { emitted, rows, stack, frame }
}

/// Compile one function of `module` to its encoded bytes and relocations. Runs
/// isel → register allocation → frame layout → prologue/epilogue → encoding.
pub fn compile_function(module: &Module, func: crate::ir::FuncId, syms: &StrInterner) -> Emitted {
    let legal = legalized(module, &Sse2Legality);
    let wide = super::prepared_if_wide(&legal, syms);
    let (module, syms) = wide.as_ref().map_or((&*legal, syms), |(m, s)| (m, s));
    compile_function_full(module, func, syms, &CodegenOptions::default(), false).emitted
}

/// Compile every defined function of `module` into a relocatable
/// [`ObjectModule`]: a single `.text` section with one global function symbol
/// per definition, and the call/global relocations wired to (undefined-if-new)
/// symbols; then every defined global's storage into `.rodata`/`.data`/`.bss`
/// with `R_X86_64_64` data relocations for address-valued initializers (see
/// [`crate::codegen::data`]). `syms` resolves the interned function/global
/// names. Uses the default [`CodegenOptions`] (stack probes on); see
/// [`compile_module_with`] for options and the stack-usage report.
pub fn compile_module(module: &Module, syms: &StrInterner) -> ObjectModule {
    build_module(module, syms, &CodegenOptions::default(), None).object
}

/// Like [`compile_module`], under `opts`, and also returning every defined
/// function's [`StackUsage`] (in definition order) in the [`CompiledModule`].
pub fn compile_module_with(
    module: &Module,
    syms: &StrInterner,
    opts: &CodegenOptions,
) -> CompiledModule {
    build_module(module, syms, opts, None)
}

/// Like [`compile_function`], but also returns the `(offset, line)` statement
/// rows for the function's `.debug_line` program.
pub fn compile_function_lines(
    module: &Module,
    func: crate::ir::FuncId,
    syms: &StrInterner,
) -> (Emitted, Vec<(u64, u32)>) {
    let legal = legalized(module, &Sse2Legality);
    let wide = super::prepared_if_wide(&legal, syms);
    let (module, syms) = wide.as_ref().map_or((&*legal, syms), |(m, s)| (m, s));
    let out = compile_function_full(module, func, syms, &CodegenOptions::default(), true);
    (out.emitted, out.rows)
}

/// Metadata identifying the `.lf` source a debug build was compiled from.
#[derive(Clone, Debug)]
pub struct DebugSource {
    /// The source file name (`DW_AT_name`), relative to `comp_dir`.
    pub file_name: String,
    /// The compilation directory (`DW_AT_comp_dir`).
    pub comp_dir: String,
}

/// Compile `module` to a relocatable [`ObjectModule`] like [`compile_module`],
/// and additionally emit the DWARF `.debug_abbrev`/`.debug_info`/`.debug_str`/
/// `.debug_line` sections describing every defined function (name, address
/// range, and source-line table). Address fields in the debug data become
/// [`Abs64`](crate::mc::object::RelocKind::Abs64) relocations against the
/// function symbols, so the linker fills real addresses.
pub fn compile_module_debug(
    module: &Module,
    syms: &StrInterner,
    source: &DebugSource,
) -> ObjectModule {
    build_module(module, syms, &CodegenOptions::default(), Some(source)).object
}

/// Like [`compile_module_debug`], under `opts`, and also returning the
/// stack-usage report (see [`compile_module_with`]).
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

    // Vector code the SSE2 baseline cannot hold or select is scalarized first,
    // then integers wider than 64 bits are split (all but their ABI boundary).
    let legal = legalized(module, &Sse2Legality);
    let wide = super::prepared_if_wide(&legal, syms);
    let (module, syms): (&Module, &StrInterner) = wide.as_ref().map_or((&*legal, syms), |(m, s)| (m, s));

    let mut obj = ObjectModule::new(module.name.clone());
    // Functions start on `align`-byte boundaries (16 unless the options
    // say otherwise), and `.text` asks for that much so a linker keeps it.
    let align = opts.function_alignment_for(16, 1) as usize;
    let text = obj.add_section(Section::new(".text", SectionKind::Text, align as u64));
    let mut funcs: Vec<FuncDebug> = Vec::new();
    let mut stack = StackReport::new();
    // Each function's unwind description and symbol.
    let mut frames: Vec<(FunctionFrame, crate::mc::object::SymbolId)> = Vec::new();

    for (i, f) in module.functions().enumerate() {
        if f.is_declaration() {
            continue;
        }
        let fid = crate::ir::FuncId::from_index(i);
        let out = compile_function_full(module, fid, syms, opts, debug.is_some());
        let emitted = out.emitted;
        stack.push(out.stack);
        // Align this function's start within .text.
        {
            let sec = obj.section_mut(text);
            while !sec.bytes.len().is_multiple_of(align) {
                sec.bytes.push(0x90); // nop padding
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
        frames.push((FunctionFrame { offset: off, size: len, ..out.frame }, fsym));
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
            // Build the function's line rows: a function-entry row at the decl
            // line, then the statement rows (dropping runs of the same line).
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

    // Under PIC, pointer-holding constants go to `.data.rel.ro` (see
    // `codegen::data`), and the object says it needs no executable stack.
    let pic = opts.reloc_model.is_pic();
    crate::codegen::data::emit_globals_with(module, syms, &mut obj, RelocKind::Abs64, pic);
    crate::codegen::linkage::apply_symbol_attrs(module, syms, &mut obj);
    if pic {
        obj.add_section(Section::new(".note.GNU-stack", SectionKind::Debug, 1));
    }
    emit_unwind_tables(&mut obj, text, &frames, opts.unwind_tables());

    if let Some(source) = debug {
        let text_size = obj.section(text).bytes.len() as u64;
        let unit = DebugUnit {
            file_name: source.file_name.clone(),
            comp_dir: source.comp_dir.clone(),
            producer: "LatticeFoundry".to_owned(),
            text_size,
            funcs,
        };
        let dw = crate::mc::dwarf::build(&unit);

        // Plain (relocation-free) sections.
        obj.add_section(debug_section(".debug_abbrev", dw.abbrev));
        obj.add_section(debug_section(".debug_str", dw.str));
        // Sections carrying address relocations against the function symbols.
        obj.add_emitted_section(".debug_info", SectionKind::Debug, 1, dw.info);
        obj.add_emitted_section(".debug_line", SectionKind::Debug, 1, dw.line);
    }

    CompiledModule { object: obj, stack }
}

/// Add the unwind tables `tables` describing the module's functions (each
/// frame with its function symbol) to `obj`. See [`crate::codegen::unwind`].
fn emit_unwind_tables(
    obj: &mut ObjectModule,
    text: crate::mc::object::SectionId,
    frames: &[(FunctionFrame, crate::mc::object::SymbolId)],
    tables: UnwindTables,
) {
    let funcs: Vec<FunctionFrame> = frames.iter().map(|(f, _)| f.clone()).collect();
    match tables {
        UnwindTables::None => {}
        // The Windows layout is built for these codes (see
        // `FrameLayout::prologue_plan`); only System V frames (a non-Windows
        // OS asking for `.pdata`) cannot be described, and then the object
        // gets no table (`emit_win64` adds nothing on error).
        UnwindTables::Win64 => {
            let _ = unwind::emit_win64(obj, text, &funcs);
        }
        UnwindTables::EhFrame => unwind::emit_eh_frame(obj, text, &funcs),
        UnwindTables::CompactUnwind => {
            // A frame the compact encoding cannot express (Windows-only
            // callee-saved registers, xmm saves) gets no record.
            let records: Vec<_> = frames
                .iter()
                .filter_map(|(f, sym)| unwind::compact_unwind_x86_64(f).map(|enc| (*sym, f.size, enc)))
                .collect();
            unwind::emit_compact_unwind(obj, &records);
        }
    }
}

/// A non-allocated debug [`Section`] holding `bytes`.
fn debug_section(name: &str, bytes: Vec<u8>) -> Section {
    let mut s = Section::new(name, SectionKind::Debug, 1);
    s.bytes = bytes;
    s
}

/// Compile `module` to a complete ELF64 relocatable object image.
pub fn compile_to_elf(module: &Module, syms: &StrInterner) -> Vec<u8> {
    crate::mc::elf::write(&compile_module(module, syms))
}
