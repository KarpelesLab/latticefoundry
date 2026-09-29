//! Green-thread **context-switching runtime** for RV64 (`G` = `IMAFD`), emitted
//! as machine code by LatticeFoundry (the RISC-V counterpart of
//! [`crate::target::x86_64::runtime`]; see there for the design).
//!
//! # The context layout (version [`layout::VERSION`] = 1)
//!
//! [`layout::SIZE`] = 528 bytes, 16-byte aligned:
//!
//! | offset          | size | field                                              |
//! |-----------------|------|----------------------------------------------------|
//! | `0x000`         | 8    | `pc` — where execution resumes (the `x0` slot)     |
//! | `0x000 + 8*n`   | 8    | `x<n>` for `n` in `1..=31`                          |
//! | `0x100`         | 4    | `fcsr`                                             |
//! | `0x108`         | 4    | layout version (`1`)                               |
//! | `0x10C`         | 4    | kind: `0` = cooperative, `1` = full                |
//! | `0x110 + 8*n`   | 8    | `f<n>` (64-bit `D` registers)                      |
//!
//! A cooperative context holds the LP64D callee-saved state: `ra`, `sp`,
//! `s0..s11`, `fs0..fs11` and `fcsr`. A full context also holds every other
//! integer and FP register. `gp` and `tp` are saved by a full save but never
//! restored: they belong to the process / OS thread, not to a green thread.
//! Vector (`V`) state is not part of version 1 (the LF RISC-V target does not
//! use `V`).
//!
//! # Routines (LP64D)
//!
//! `lf_ctx_save`, `lf_ctx_save_full`, `lf_ctx_restore`, `lf_ctx_switch`,
//! `lf_ctx_switch_full` and `lf_ctx_init`, with the same signatures and
//! semantics as on x86-64.
//!
//! **Full restore clobbers `t6`** (`x31`), which holds the resume address for
//! the final `jr` — the same trade-off as AArch64's `x17`: exact for contexts
//! captured at a call (where `t6` is dead); asynchronous resumption without
//! losing `t6` has to go through the kernel.
//!
//! **Not provided here (yet):** the signal-`ucontext` mapping. On riscv64
//! Linux, `uc_mcontext` is a `struct sigcontext` holding `user_regs_struct`
//! (`pc`, `x1..x31`) followed by the `__riscv_d_ext_state` (`f[32]`, `fcsr`),
//! then any extension (`V`) records.

use super::super::rt_words::{WLabel, WordAsm};
use crate::mc::object::ObjectModule;

/// The `LfCtx` layout for RV64.
pub mod layout {
    /// The layout version.
    pub const VERSION: u32 = 1;
    /// Total size in bytes.
    pub const SIZE: usize = 0x210;
    /// Required alignment in bytes.
    pub const ALIGN: usize = 16;
    /// Offset of the saved `pc`.
    pub const PC: usize = 0;
    /// Offset of `x<n>` (`n` in `1..=31`).
    pub const fn x(n: u32) -> usize {
        8 * n as usize
    }
    /// Offset of `fcsr` (`u32`).
    pub const FCSR: usize = 0x100;
    /// Offset of the `u32` layout version.
    pub const VERSION_OFF: usize = 0x108;
    /// Offset of the `u32` kind.
    pub const KIND: usize = 0x10C;
    /// Offset of `f<n>`.
    pub const fn f(n: u32) -> usize {
        0x110 + 8 * n as usize
    }
    /// Kind: only callee-saved state is valid.
    pub const KIND_COOP: u32 = 0;
    /// Kind: every field is valid.
    pub const KIND_FULL: u32 = 1;
}

const ZERO: u32 = 0;
const RA: u32 = 1;
const SP: u32 = 2;
const GP: u32 = 3;
const TP: u32 = 4;
const T0: u32 = 5;
const T1: u32 = 6;
const S1: u32 = 9;
const A0: u32 = 10;
const A1: u32 = 11;
const S2: u32 = 18;
const T6: u32 = 31;

/// The callee-saved integer registers besides `ra`/`sp`: `s0`, `s1`, `s2..s11`.
const S_REGS: [u32; 12] = [8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];
/// The callee-saved FP registers: `fs0`, `fs1`, `fs2..fs11`.
const FS_REGS: [u32; 12] = [8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];

fn i_type(imm: i32, rs1: u32, f3: u32, rd: u32, op: u32) -> u32 {
    ((imm as u32) & 0xFFF) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
}
fn s_type(imm: i32, rs2: u32, rs1: u32, f3: u32, op: u32) -> u32 {
    let i = imm as u32;
    ((i >> 5) & 0x7F) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (i & 0x1F) << 7 | op
}

fn sd(a: &mut WordAsm, rs2: u32, off: usize, rs1: u32) {
    a.emit(s_type(off as i32, rs2, rs1, 3, 0x23), format!("sd x{rs2}, {off}(x{rs1})"));
}
fn ld(a: &mut WordAsm, rd: u32, off: usize, rs1: u32) {
    a.emit(i_type(off as i32, rs1, 3, rd, 0x03), format!("ld x{rd}, {off}(x{rs1})"));
}
fn sw(a: &mut WordAsm, rs2: u32, off: usize, rs1: u32) {
    a.emit(s_type(off as i32, rs2, rs1, 2, 0x23), format!("sw x{rs2}, {off}(x{rs1})"));
}
fn lw(a: &mut WordAsm, rd: u32, off: usize, rs1: u32) {
    a.emit(i_type(off as i32, rs1, 2, rd, 0x03), format!("lw x{rd}, {off}(x{rs1})"));
}
fn fsd(a: &mut WordAsm, rs2: u32, off: usize, rs1: u32) {
    a.emit(s_type(off as i32, rs2, rs1, 3, 0x27), format!("fsd f{rs2}, {off}(x{rs1})"));
}
fn fld(a: &mut WordAsm, rd: u32, off: usize, rs1: u32) {
    a.emit(i_type(off as i32, rs1, 3, rd, 0x07), format!("fld f{rd}, {off}(x{rs1})"));
}
fn addi(a: &mut WordAsm, rd: u32, rs1: u32, imm: i32) {
    a.emit(i_type(imm, rs1, 0, rd, 0x13), format!("addi x{rd}, x{rs1}, {imm}"));
}
fn andi(a: &mut WordAsm, rd: u32, rs1: u32, imm: i32) {
    a.emit(i_type(imm, rs1, 7, rd, 0x13), format!("andi x{rd}, x{rs1}, {imm}"));
}
fn jalr(a: &mut WordAsm, rd: u32, rs1: u32) {
    a.emit(i_type(0, rs1, 0, rd, 0x67), format!("jalr x{rd}, 0(x{rs1})"));
}
/// `csrr rd, fcsr` (`csrrs rd, fcsr, x0`).
fn frcsr(a: &mut WordAsm, rd: u32) {
    a.emit(0x0030_2073 | rd << 7, format!("csrr x{rd}, fcsr"));
}
/// `csrw fcsr, rs` (`csrrw x0, fcsr, rs`).
fn fscsr(a: &mut WordAsm, rs: u32) {
    a.emit(0x0030_1073 | rs << 15, format!("csrw fcsr, x{rs}"));
}
fn patch_b(w: u32, d: i64) -> u32 {
    let i = d as u32;
    w | ((i >> 12) & 1) << 31 | ((i >> 5) & 0x3F) << 25 | ((i >> 1) & 0xF) << 8 | ((i >> 11) & 1) << 7
}
fn patch_j(w: u32, d: i64) -> u32 {
    let i = d as u32;
    w | ((i >> 20) & 1) << 31 | ((i >> 1) & 0x3FF) << 21 | ((i >> 11) & 1) << 20 | ((i >> 12) & 0xFF) << 12
}
fn bne(a: &mut WordAsm, rs1: u32, rs2: u32, l: WLabel) {
    a.emit_ref(rs2 << 20 | rs1 << 15 | 1 << 12 | 0x63, format!("bne x{rs1}, x{rs2}, {}", WordAsm::name(l)), l, patch_b);
}
fn j(a: &mut WordAsm, l: WLabel) {
    a.emit_ref(0x6F, format!("jal x0, {}", WordAsm::name(l)), l, patch_j);
}

fn save_coop(a: &mut WordAsm) {
    sd(a, RA, layout::x(RA), A0);
    sd(a, SP, layout::x(SP), A0);
    for s in S_REGS {
        sd(a, s, layout::x(s), A0);
    }
    sd(a, RA, layout::PC, A0);
    for f in FS_REGS {
        fsd(a, f, layout::f(f), A0);
    }
    frcsr(a, T0);
    sw(a, T0, layout::FCSR, A0);
    addi(a, T0, ZERO, layout::VERSION as i32);
    sw(a, T0, layout::VERSION_OFF, A0);
    sw(a, ZERO, layout::KIND, A0);
}

fn save_full(a: &mut WordAsm, a0_is_one: bool) {
    for r in 1..=31 {
        sd(a, r, layout::x(r), A0);
    }
    if a0_is_one {
        addi(a, T0, ZERO, 1);
        sd(a, T0, layout::x(A0), A0);
    }
    sd(a, RA, layout::PC, A0);
    for f in 0..=31 {
        fsd(a, f, layout::f(f), A0);
    }
    frcsr(a, T0);
    sw(a, T0, layout::FCSR, A0);
    addi(a, T0, ZERO, 1);
    sw(a, T0, layout::VERSION_OFF, A0);
    sw(a, T0, layout::KIND, A0);
}

fn restore(a: &mut WordAsm) {
    let full = a.label();
    lw(a, T0, layout::KIND, A0);
    bne(a, T0, ZERO, full);
    ld(a, RA, layout::x(RA), A0);
    ld(a, SP, layout::x(SP), A0);
    for s in S_REGS {
        ld(a, s, layout::x(s), A0);
    }
    for f in FS_REGS {
        fld(a, f, layout::f(f), A0);
    }
    lw(a, T0, layout::FCSR, A0);
    fscsr(a, T0);
    ld(a, T1, layout::PC, A0);
    addi(a, A0, ZERO, 1);
    jalr(a, ZERO, T1);

    a.bind(full);
    for f in 0..=31 {
        fld(a, f, layout::f(f), A0);
    }
    lw(a, T0, layout::FCSR, A0);
    fscsr(a, T0);
    for r in 1..=30 {
        if r == A0 || r == GP || r == TP {
            continue;
        }
        ld(a, r, layout::x(r), A0);
    }
    ld(a, T6, layout::PC, A0);
    ld(a, A0, layout::x(A0), A0);
    jalr(a, ZERO, T6);
}

/// Build the whole runtime.
pub(crate) fn assemble() -> WordAsm {
    use crate::target::x86_64::runtime as names;
    let mut a = WordAsm::new();
    let restore_l = a.label();
    let stub = a.label();

    a.begin(names::SYM_SAVE, true);
    save_coop(&mut a);
    addi(&mut a, A0, ZERO, 0);
    jalr(&mut a, ZERO, RA);

    a.begin(names::SYM_SAVE_FULL, true);
    save_full(&mut a, true);
    addi(&mut a, A0, ZERO, 0);
    jalr(&mut a, ZERO, RA);

    a.begin(names::SYM_RESTORE, true);
    a.bind(restore_l);
    restore(&mut a);

    a.begin(names::SYM_SWITCH, true);
    save_coop(&mut a);
    addi(&mut a, A0, A1, 0);
    j(&mut a, restore_l);

    a.begin(names::SYM_SWITCH_FULL, true);
    save_full(&mut a, false);
    addi(&mut a, A0, A1, 0);
    j(&mut a, restore_l);

    // The fresh-thread trampoline: entry(arg), then exit_group(result). Placed
    // before lf_ctx_init so its offset is known for the `auipc`/`addi` there.
    a.begin("lf_ctx_thread_start", false);
    a.bind(stub);
    let stub_at = a.words.len();
    addi(&mut a, A0, S2, 0);
    jalr(&mut a, RA, S1);
    addi(&mut a, 17, ZERO, 94); // a7 = exit_group
    a.emit(0x0000_0073, "ecall".into());
    a.emit(0x0010_0073, "ebreak".into());

    // lf_ctx_init(ctx, stack_top, entry, arg)
    a.begin(names::SYM_INIT, true);
    let zero = a.label();
    addi(&mut a, T0, A0, 0);
    addi(&mut a, T1, A0, layout::SIZE as i32);
    a.bind(zero);
    sd(&mut a, ZERO, 0, T0);
    sd(&mut a, ZERO, 8, T0);
    addi(&mut a, T0, T0, 16);
    bne(&mut a, T0, T1, zero);
    andi(&mut a, A1, A1, -16);
    sd(&mut a, A1, layout::x(SP), A0);
    // t0 = &stub (auipc t0, 0; addi t0, t0, stub - here)
    let here = a.words.len();
    a.emit(0x17 | T0 << 7, "auipc x5, 0".into());
    addi(&mut a, T0, T0, (stub_at as i32 - here as i32) * 4);
    sd(&mut a, T0, layout::PC, A0);
    sd(&mut a, 12, layout::x(S1), A0); // entry (a2) in s1
    sd(&mut a, 13, layout::x(S2), A0); // arg (a3) in s2
    frcsr(&mut a, T0);
    sw(&mut a, T0, layout::FCSR, A0);
    addi(&mut a, T0, ZERO, layout::VERSION as i32);
    sw(&mut a, T0, layout::VERSION_OFF, A0);
    jalr(&mut a, ZERO, RA);
    a
}

/// Append the RISC-V context runtime to `obj` (section `.text.lf_rt`).
pub fn emit_context_runtime(obj: &mut ObjectModule) {
    assemble().into_object(obj);
}

/// The RISC-V context runtime as a stand-alone object.
pub fn context_runtime_object() -> ObjectModule {
    let mut obj = ObjectModule::new("lf_rt");
    emit_context_runtime(&mut obj);
    obj
}
