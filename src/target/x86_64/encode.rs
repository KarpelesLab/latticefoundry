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
//!    ([`compile_module`]) and, via [`crate::mc::elf`], an ELF64 object.
//!
//! The encoding tables are implemented from the published x86-64 instruction-set
//! reference (tenet T1), not copied from any assembler.

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, Reg, RegClass, StackSlot};
use crate::codegen::regalloc;
use crate::ir::Module;
use crate::mc::emit::{Emitted, Emitter, Ref};
use crate::mc::object::{
    ObjectModule, Section, SectionKind, Symbol, SymbolBinding, SymbolType,
};
use crate::support::StrInterner;

use super::isel::{X86Op, X86_64Target};
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
/// bits, else `REX.W B8+r io` (`movabs`).
pub(crate) fn mov_ri(e: &mut Emitter, dst: u8, value: u64) {
    if value <= u64::from(u32::MAX) {
        if dst >= 8 {
            e.u8(rex(false, false, false, true));
        }
        e.u8(0xB8 + (dst & 7));
        e.u32(value as u32);
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
        _ => {
            // 32 → 64: `movsxd r64, r/m32` = REX.W 63 /r.
            e.u8(rex(true, dst >= 8, false, src >= 8));
            e.u8(0x63);
            e.u8(modrm(3, dst, src));
        }
    }
}

/// Emit `movzx dst, src`: **zero**-extend a `src_w`-bit source. `0F B6` (byte) /
/// `0F B7` (word) zero-extend into the full register; a 32-bit source uses a
/// plain 32-bit `mov`, which zero-extends bits 32..63 automatically.
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
        _ => mov_rr(e, dst, src, false),
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

/// Emit `cmp r, imm32` (`REX.W 81 /7 id`).
fn cmp_ri(e: &mut Emitter, reg: u8, value: i32, w: bool) {
    if w || reg >= 8 {
        e.u8(rex(w, false, false, reg >= 8));
    }
    e.u8(0x81);
    e.u8(modrm(3, 7, reg));
    e.u32(value as u32);
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

/// Emit `and r64/r32, imm8` (sign-extended immediate) — `83 /4 ib`.
fn and_ri8(e: &mut Emitter, reg: u8, imm8: i8, w: bool) {
    if w || reg >= 8 {
        e.u8(rex(w, false, false, reg >= 8));
    }
    e.u8(0x83);
    e.u8(modrm(3, 4, reg)); // /4 selects AND
    e.u8(imm8 as u8);
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
    e.u8(0x0F);
    e.u8(0x88); // js neg
    e.pcrel32(Ref::Label(neg), 0);
    sse_rr(e, pfx, true, 0x2A, d, s); // cvtsi2sd/ss d, s  (in range ⇒ exact)
    e.u8(0xE9); // jmp done
    e.pcrel32(Ref::Label(done), 0);
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
    e.u8(0x0F);
    e.u8(0x83); // jae big  (CF=0 ⇒ s ≥ 2^63)
    e.pcrel32(Ref::Label(big), 0);
    sse_rr(e, pfx, true, 0x2C, d, s); // cvttsd2si d, s  (in range)
    e.u8(0xE9); // jmp done
    e.pcrel32(Ref::Label(done), 0);
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
    /// Bytes occupied by the pushed callee-saved registers (`8 * cs_regs.len()`).
    cs_bytes: i32,
    /// The `sub rsp` amount that follows the callee-saved pushes.
    sub_size: i32,
}

/// Round `value` up to a multiple of `align` (a power of two ≥ 1).
fn align_up(value: i64, align: i64) -> i64 {
    (value + align - 1) / align * align
}

/// Compute the frame layout of an allocated machine function.
pub fn layout_frame(mf: &MachineFunction, target: &X86_64Target) -> FrameLayout {
    use crate::codegen::target::MachineTarget;
    let callee: Vec<u8> = target.callee_saved().iter().map(|p| p.num as u8).collect();

    // Which callee-saved registers does the allocation actually define?
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
    let cs_regs: Vec<u8> = callee.into_iter().filter(|&r| used[r as usize]).collect();
    let cs_bytes = (cs_regs.len() * 8) as i32;

    // Slot offsets grow downward from just below the callee-saved region.
    let mut off = 0i64;
    let mut slot_off = vec![0i32; mf.frame().len()];
    for (i, off_slot) in slot_off.iter_mut().enumerate() {
        let info = mf.frame().slot(StackSlot::from_index(i));
        off += info.size as i64;
        off = align_up(off, (info.align.max(1)) as i64);
        *off_slot = -((cs_bytes as i64) + off) as i32;
    }
    let locals = off;
    // The outgoing stack-argument area sits at the very bottom of the frame
    // (`[rsp .. rsp + outgoing)`), below the spill/alloca locals. rsp is constant
    // after the prologue, so isel addresses stack arguments as `[rsp + k]`.
    let outgoing = mf.frame().outgoing() as i64;
    let total = cs_bytes as i64 + locals + outgoing;
    let padded = align_up(total, 16);
    let sub_size = (padded - cs_bytes as i64) as i32;

    FrameLayout { slot_off, cs_regs, cs_bytes, sub_size }
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
pub fn insert_prologue_epilogue(mf: &mut MachineFunction, layout: &FrameLayout) {
    let entry = mf.entry().expect("a function being compiled has an entry block");

    // --- prologue: push rbp; mov rbp,rsp; push callee-saved; sub rsp,frame ---
    let mut prologue = vec![
        MachineInst::new(X86Op::Push.opcode(), vec![phys_use(RBP)]),
        MachineInst::new(X86Op::MovRbpRsp.opcode(), Vec::new()),
    ];
    for &cs in &layout.cs_regs {
        prologue.push(MachineInst::new(X86Op::Push.opcode(), vec![phys_use(u16::from(cs))]));
    }
    if layout.sub_size > 0 {
        prologue
            .push(MachineInst::new(X86Op::SubRsp.opcode(), vec![imm_op(layout.sub_size as u64)]));
    }
    let old = std::mem::take(&mut mf.block_mut(entry).insts);
    prologue.extend(old);
    mf.block_mut(entry).insts = prologue;

    // --- epilogue before each Ret: lea rsp,[rbp-cs]; pop callee-saved; pop rbp ---
    let block_ids: Vec<_> = mf.block_ids().collect();
    for bid in block_ids {
        let old = std::mem::take(&mut mf.block_mut(bid).insts);
        let mut new_insts = Vec::with_capacity(old.len());
        for inst in old {
            if X86Op::decode(inst.opcode) == X86Op::Ret {
                new_insts.push(MachineInst::new(
                    X86Op::LeaRspRbp.opcode(),
                    vec![imm_op(layout.cs_bytes as u64)],
                ));
                for &cs in layout.cs_regs.iter().rev() {
                    new_insts.push(MachineInst::new(X86Op::Pop.opcode(), vec![phys(u16::from(cs))]));
                }
                new_insts.push(MachineInst::new(X86Op::Pop.opcode(), vec![phys(RBP)]));
            }
            new_insts.push(inst);
        }
        mf.block_mut(bid).insts = new_insts;
    }
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
        // movss (4-byte) / movsd (8-byte) load into an xmm register.
        sse_mem(e, scalar_prefix(size != 4), 0x10, d, ptr, 0);
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
        // movss / movsd store from an xmm register (store opcode 0x11).
        sse_mem(e, scalar_prefix(size != 4), 0x11, val, ptr, 0);
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

/// Encode one machine instruction into `e`.
fn encode_inst(e: &mut Emitter, inst: &MachineInst, ctx: &EncodeCtx<'_>) {
    let ops = &inst.operands;
    match X86Op::decode(inst.opcode) {
        X86Op::MovRR => {
            let d = rnum(&ops[0]);
            let s = rnum(&ops[1]);
            if d == s {
                // A self-move is a no-op regardless of class.
            } else if rclass(&ops[0]) == RegClass::Fp {
                // xmm↔xmm copy via `movsd` (copies the low 64 bits, which holds
                // both f32 and f64 values exactly).
                sse_rr(e, 0xF2, false, 0x10, d, s);
            } else {
                mov_rr(e, d, s, true);
            }
        }
        X86Op::MovRI => mov_ri(e, rnum(&ops[0]), uimm(&ops[1])),
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
        X86Op::SetccCmp => {
            let d = rnum(&ops[0]);
            let a = rnum(&ops[1]);
            let b = rnum(&ops[2]);
            let cc = uimm(&ops[3]) as u8;
            let w = iimm(&ops[4]) == 64;
            alu_rr(e, 0x39, a, b, w); // cmp a, b
            setcc(e, cc, d);
            movzx_byte(e, d);
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
            mem(e, &[0x8D], d, RBP as u8, ctx.layout.slot_off[slot], true, false);
        }
        X86Op::StoreFrame => {
            let src = rnum(&ops[0]);
            let slot = slot_index(&ops[1]);
            let off = ctx.layout.slot_off[slot];
            if rclass(&ops[0]) == RegClass::Fp {
                sse_mem(e, 0xF2, 0x11, src, RBP as u8, off); // movsd [rbp+off], xmm
            } else {
                mem(e, &[0x89], src, RBP as u8, off, true, false);
            }
        }
        X86Op::LoadFrame => {
            let dst = rnum(&ops[0]);
            let slot = slot_index(&ops[1]);
            let off = ctx.layout.slot_off[slot];
            if rclass(&ops[0]) == RegClass::Fp {
                sse_mem(e, 0xF2, 0x10, dst, RBP as u8, off); // movsd xmm, [rbp+off]
            } else {
                mem(e, &[0x8B], dst, RBP as u8, off, true, false);
            }
        }
        X86Op::GlobalAddr => {
            let d = rnum(&ops[0]);
            let g = match ops[1] {
                MachineOperand::Global(g) => g,
                _ => panic!("GlobalAddr expects a global operand"),
            };
            // lea d, [rip + disp32]  with a PC32 relocation to the global.
            e.u8(rex(true, d >= 8, false, false));
            e.u8(0x8D);
            e.u8(modrm(0, d, 5));
            e.pcrel32(Ref::Symbol((ctx.global_name)(g)), 0);
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
            e.u8(rex(true, d >= 8, false, false));
            e.u8(0x8D);
            e.u8(modrm(0, d, 5));
            e.pcrel32(Ref::Symbol((ctx.func_name)(f)), 0);
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
        X86Op::Jmp => {
            let t = label_index(&ops[0]);
            e.u8(0xE9);
            e.pcrel32(Ref::Label(ctx.labels[t]), 0);
        }
        X86Op::BrCond => {
            let cond = rnum(&ops[0]);
            let t = label_index(&ops[1]);
            let f = label_index(&ops[2]);
            alu_rr(e, 0x85, cond, cond, false); // test cond, cond
            e.u8(0x0F);
            e.u8(0x85); // jne t
            e.pcrel32(Ref::Label(ctx.labels[t]), 0);
            e.u8(0xE9); // jmp f
            e.pcrel32(Ref::Label(ctx.labels[f]), 0);
        }
        X86Op::Switch => {
            let cond = rnum(&ops[0]);
            let default = label_index(&ops[1]);
            let mut i = 2;
            while i + 1 < ops.len() {
                let value = iimm(&ops[i]) as i32;
                let case = label_index(&ops[i + 1]);
                cmp_ri(e, cond, value, true);
                e.u8(0x0F);
                e.u8(0x84); // je case
                e.pcrel32(Ref::Label(ctx.labels[case]), 0);
                i += 2;
            }
            e.u8(0xE9);
            e.pcrel32(Ref::Label(ctx.labels[default]), 0);
        }
        X86Op::Unreachable => {
            e.u8(0x0F);
            e.u8(0x0B); // ud2
        }
        X86Op::Push => push_r(e, rnum(&ops[0])),
        X86Op::Pop => pop_r(e, rnum(&ops[0])),
        X86Op::MovRbpRsp => mov_rr(e, RBP as u8, RSP as u8, true),
        X86Op::SubRsp => {
            e.u8(0x48);
            e.u8(0x81);
            e.u8(modrm(3, 5, RSP as u8));
            e.u32(uimm(&ops[0]) as u32);
        }
        X86Op::LeaRspRbp => {
            let k = uimm(&ops[0]) as i64;
            mem(e, &[0x8D], RSP as u8, RBP as u8, (-k) as i32, true, false);
        }
        X86Op::LeaRbpOff => {
            let d = rnum(&ops[0]);
            let off = iimm(&ops[1]) as i32;
            mem(e, &[0x8D], d, RBP as u8, off, true, false); // lea d, [rbp + off]
        }
        X86Op::LeaRspOff => {
            let d = rnum(&ops[0]);
            let off = iimm(&ops[1]) as i32;
            mem(e, &[0x8D], d, RSP as u8, off, true, false); // lea d, [rsp + off]
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
    encode_function_inner(mf, layout, func_name, global_name, None)
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
    let emitted = encode_function_inner(mf, layout, func_name, global_name, Some(&mut rows));
    (emitted, rows)
}

fn encode_function_inner(
    mf: &MachineFunction,
    layout: &FrameLayout,
    func_name: &dyn Fn(u32) -> String,
    global_name: &dyn Fn(u32) -> String,
    mut lines: Option<&mut Vec<(u64, u32)>>,
) -> Emitted {
    let mut e = Emitter::new();
    let labels: Vec<_> = (0..mf.num_blocks()).map(|_| e.create_label()).collect();
    let ctx = EncodeCtx { labels: &labels, layout, func_name, global_name };

    // Emit the entry block first (so the function symbol at offset 0 is the
    // entry), then the remaining blocks in arena order.
    let entry = mf.entry().expect("a function being compiled has an entry block");
    let mut order = vec![entry];
    for bid in mf.block_ids() {
        if bid != entry {
            order.push(bid);
        }
    }
    for bid in order {
        e.bind_label(labels[bid.index()]);
        for inst in &mf.block(bid).insts {
            if let Some(rows) = lines.as_deref_mut()
                && inst.line != 0
                && rows.last().map(|&(_, l)| l) != Some(inst.line)
            {
                rows.push((e.offset(), inst.line));
            }
            encode_inst(&mut e, inst, &ctx);
        }
    }
    e.finish().expect("intra-function branch resolution never overflows")
}

/// Compile one function of `module` to its encoded bytes and relocations. Runs
/// isel → register allocation → frame layout → prologue/epilogue → encoding.
pub fn compile_function(module: &Module, func: crate::ir::FuncId, syms: &StrInterner) -> Emitted {
    let target = X86_64Target::new();
    let mut mf = target.select_with_syms(module, func, syms);
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
    let text = obj.add_section(Section::new(".text", SectionKind::Text, 16));

    for (i, f) in module.functions().enumerate() {
        if f.is_declaration() {
            continue;
        }
        let fid = crate::ir::FuncId::from_index(i);
        let emitted = compile_function(module, fid, syms);
        // 16-align this function's start within .text.
        {
            let sec = obj.section_mut(text);
            while !sec.bytes.len().is_multiple_of(16) {
                sec.bytes.push(0x90); // nop padding
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

/// Like [`compile_function`], but also returns the `(offset, line)` statement
/// rows for the function's `.debug_line` program.
pub fn compile_function_lines(
    module: &Module,
    func: crate::ir::FuncId,
    syms: &StrInterner,
) -> (Emitted, Vec<(u64, u32)>) {
    let target = X86_64Target::new();
    let mut mf = target.select_with_syms(module, func, syms);
    regalloc::allocate(&mut mf, &target);
    let layout = layout_frame(&mf, &target);
    insert_prologue_epilogue(&mut mf, &layout);
    let func_name = |idx: u32| -> String {
        syms.resolve(module.function(crate::ir::FuncId::from_index(idx as usize)).name).to_owned()
    };
    let global_name = |idx: u32| -> String {
        syms.resolve(module.global(crate::ir::GlobalId::from_index(idx as usize)).name).to_owned()
    };
    encode_function_lines(&mf, &layout, &func_name, &global_name)
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
    use crate::mc::dwarf::{DebugUnit, FuncDebug};

    let mut obj = ObjectModule::new(module.name.clone());
    let text = obj.add_section(Section::new(".text", SectionKind::Text, 16));
    let mut funcs: Vec<FuncDebug> = Vec::new();

    for (i, f) in module.functions().enumerate() {
        if f.is_declaration() {
            continue;
        }
        let fid = crate::ir::FuncId::from_index(i);
        let (emitted, stmt_rows) = compile_function_lines(module, fid, syms);
        // 16-align this function's start within .text.
        {
            let sec = obj.section_mut(text);
            while !sec.bytes.len().is_multiple_of(16) {
                sec.bytes.push(0x90); // nop padding
            }
        }
        let off = obj.section(text).bytes.len() as u64;
        let len = emitted.bytes.len() as u64;
        obj.section_mut(text).bytes.extend_from_slice(&emitted.bytes);

        let name = syms.resolve(f.name).to_owned();
        obj.add_symbol(Symbol::defined(
            name.clone(),
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

        // Build the function's line rows: a function-entry row at the decl line,
        // then the statement rows (dropping runs of the same line).
        let decl_line = f.decl_line.unwrap_or(1);
        let mut rows = vec![(0u64, decl_line)];
        for (roff, line) in stmt_rows {
            if rows.last().map(|&(_, l)| l) != Some(line) {
                rows.push((roff, line));
            }
        }
        funcs.push(FuncDebug { name, decl_line, size: len, rows });
    }

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

    obj
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
