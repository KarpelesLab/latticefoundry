//! Green-thread **context-switching runtime** for AArch64, emitted as A64
//! machine code by LatticeFoundry (the AArch64 counterpart of
//! [`crate::target::x86_64::runtime`]; see there for the design).
//!
//! # The context layout (version [`layout::VERSION`] = 1)
//!
//! [`layout::SIZE`] = 800 bytes, 16-byte aligned:
//!
//! | offset         | size | field                                         |
//! |----------------|------|-----------------------------------------------|
//! | `0x000 + 8*n`  | 8    | `x0..x30`                                     |
//! | `0x0F8`        | 8    | `sp`                                          |
//! | `0x100`        | 8    | `pc` — where execution resumes                |
//! | `0x108`        | 8    | `NZCV` (as read by `mrs nzcv`)                |
//! | `0x110`        | 4    | layout version (`1`)                          |
//! | `0x114`        | 4    | kind: `0` = cooperative, `1` = full           |
//! | `0x118`        | 4    | `FPSR`                                        |
//! | `0x11C`        | 4    | `FPCR`                                        |
//! | `0x120 + 16*n` | 16   | `v0..v31` (full 128-bit `q` registers)        |
//!
//! A cooperative context holds the AAPCS64 callee-saved state: `x19..x28`,
//! `x29` (fp), `x30` (lr), `sp`, `pc`, `q8..q15` (whole registers, a superset of
//! the callee-saved `d8..d15`) and `FPCR`. A full context holds everything
//! listed above. SVE/SME state is not part of version 1.
//!
//! # Routines (AAPCS64)
//!
//! `lf_ctx_save`, `lf_ctx_save_full`, `lf_ctx_restore`, `lf_ctx_switch`,
//! `lf_ctx_switch_full` and `lf_ctx_init`, with the same signatures and
//! semantics as on x86-64. A fresh thread starts at a trampoline that calls
//! `entry(arg)` and then `exit_group` with its result.
//!
//! **Full restore clobbers `x17`.** A64 has no way to branch to an address
//! without holding it in a register, so resuming a full context loads every
//! register but `x17` (IP1, which the AAPCS64 lets veneers clobber at any call)
//! with its saved value and branches through `x17`. Full contexts captured by
//! `lf_ctx_save_full`/`lf_ctx_switch_full` are taken at a call, where `x17` is
//! dead, so they resume exactly; resuming an *asynchronously* interrupted
//! thread without losing `x17` has to go through the kernel (`rt_sigreturn`).
//!
//! **Not provided here (yet):** the signal-`ucontext` mapping and the
//! `rt_sigaction` helpers. On arm64 Linux the interrupted registers are in
//! `uc_mcontext` (`struct sigcontext`: `regs[31]`, `sp`, `pc`, `pstate`) and the
//! vector state is a `fpsimd_context` record (magic `0x46508001`) inside its
//! `__reserved` area, possibly followed by SVE/ZA records that would also need
//! handling; the kernel supplies a vDSO `rt_sigreturn` trampoline when
//! `SA_RESTORER` is not set, so no restorer stub is needed.

use super::super::rt_words::{WLabel, WordAsm};
use crate::mc::object::ObjectModule;

/// The `LfCtx` layout for AArch64.
pub mod layout {
    /// The layout version.
    pub const VERSION: u32 = 1;
    /// Total size in bytes.
    pub const SIZE: usize = 0x320;
    /// Required alignment in bytes.
    pub const ALIGN: usize = 16;
    /// Offset of `x<n>` (`n` in `0..=30`).
    pub const fn x(n: u32) -> usize {
        8 * n as usize
    }
    /// Offset of the saved `sp`.
    pub const SP: usize = 0xF8;
    /// Offset of the saved `pc`.
    pub const PC: usize = 0x100;
    /// Offset of the saved `NZCV`.
    pub const NZCV: usize = 0x108;
    /// Offset of the `u32` layout version.
    pub const VERSION_OFF: usize = 0x110;
    /// Offset of the `u32` kind.
    pub const KIND: usize = 0x114;
    /// Offset of `FPSR` (`u32`).
    pub const FPSR: usize = 0x118;
    /// Offset of `FPCR` (`u32`).
    pub const FPCR: usize = 0x11C;
    /// Offset of `v<n>` (`n` in `0..=31`).
    pub const fn v(n: u32) -> usize {
        0x120 + 16 * n as usize
    }
    /// Kind: only callee-saved state is valid.
    pub const KIND_COOP: u32 = 0;
    /// Kind: every field is valid.
    pub const KIND_FULL: u32 = 1;
}

const X9: u32 = 9;
const X16: u32 = 16;
const X17: u32 = 17;
const LR: u32 = 30;
/// `sp` or `xzr`, by context.
const R31: u32 = 31;

fn xn(r: u32) -> String {
    if r == R31 { "xzr".into() } else { format!("x{r}") }
}
fn base(r: u32) -> String {
    if r == R31 { "sp".into() } else { format!("x{r}") }
}

/// `stp/ldp Xt, Xt2, [Xn, #off]` (signed offset, `off` a multiple of 8).
fn pair_x(a: &mut WordAsm, load: bool, t: u32, t2: u32, n: u32, off: usize) {
    let imm7 = (off / 8) as u32 & 0x7F;
    let op = if load { 0xA940_0000 } else { 0xA900_0000 };
    let m = if load { "ldp" } else { "stp" };
    a.emit(op | imm7 << 15 | t2 << 10 | n << 5 | t, format!("{m} {}, {}, [{}, #{off}]", xn(t), xn(t2), base(n)));
}
/// `stp/ldp Qt, Qt2, [Xn, #off]` (signed offset, `off` a multiple of 16).
fn pair_q(a: &mut WordAsm, load: bool, t: u32, t2: u32, n: u32, off: usize) {
    let imm7 = (off / 16) as u32 & 0x7F;
    let op = if load { 0xAD40_0000 } else { 0xAD00_0000 };
    let m = if load { "ldp" } else { "stp" };
    a.emit(op | imm7 << 15 | t2 << 10 | n << 5 | t, format!("{m} q{t}, q{t2}, [{}, #{off}]", base(n)));
}
/// `str/ldr Xt, [Xn, #off]` (unsigned offset).
fn one_x(a: &mut WordAsm, load: bool, t: u32, n: u32, off: usize) {
    let op = if load { 0xF940_0000 } else { 0xF900_0000 };
    let m = if load { "ldr" } else { "str" };
    a.emit(op | ((off / 8) as u32) << 10 | n << 5 | t, format!("{m} {}, [{}, #{off}]", xn(t), base(n)));
}
/// `str/ldr Wt, [Xn, #off]` (unsigned offset).
fn one_w(a: &mut WordAsm, load: bool, t: u32, n: u32, off: usize) {
    let op = if load { 0xB940_0000 } else { 0xB900_0000 };
    let m = if load { "ldr" } else { "str" };
    let w = if t == R31 { "wzr".to_string() } else { format!("w{t}") };
    a.emit(op | ((off / 4) as u32) << 10 | n << 5 | t, format!("{m} {w}, [{}, #{off}]", base(n)));
}
/// `add Xd|sp, Xn|sp, #imm`.
fn add_imm(a: &mut WordAsm, d: u32, n: u32, imm: u32) {
    a.emit(0x9100_0000 | imm << 10 | n << 5 | d, format!("add {}, {}, #{imm}", base(d), base(n)));
}
/// `movz Xd|Wd, #imm16`.
fn movz(a: &mut WordAsm, wide: bool, d: u32, imm: u32) {
    let (op, r) = if wide { (0xD280_0000, format!("x{d}")) } else { (0x5280_0000, format!("w{d}")) };
    a.emit(op | imm << 5 | d, format!("movz {r}, #{imm}"));
}
/// `mov Xd, Xm` (`orr Xd, xzr, Xm`).
fn mov(a: &mut WordAsm, d: u32, m: u32) {
    a.emit(0xAA00_03E0 | m << 16 | d, format!("mov x{d}, x{m}"));
}
/// A system register accessed by `mrs`/`msr`: (encoding bits, name).
#[derive(Clone, Copy)]
enum SysReg {
    Nzcv,
    Fpcr,
    Fpsr,
}
impl SysReg {
    fn bits(self) -> (u32, &'static str) {
        match self {
            SysReg::Nzcv => (0x3_4200, "nzcv"),
            SysReg::Fpcr => (0x3_4400, "fpcr"),
            SysReg::Fpsr => (0x3_4420, "fpsr"),
        }
    }
}
fn mrs(a: &mut WordAsm, t: u32, r: SysReg) {
    let (b, n) = r.bits();
    a.emit(0xD538_0000 | b | t, format!("mrs x{t}, {n}"));
}
fn msr(a: &mut WordAsm, r: SysReg, t: u32) {
    let (b, n) = r.bits();
    a.emit(0xD518_0000 | b | t, format!("msr {n}, x{t}"));
}
fn br(a: &mut WordAsm, n: u32) {
    a.emit(0xD61F_0000 | n << 5, format!("br x{n}"));
}
fn blr(a: &mut WordAsm, n: u32) {
    a.emit(0xD63F_0000 | n << 5, format!("blr x{n}"));
}
fn ret(a: &mut WordAsm) {
    a.emit(0xD65F_03C0, "ret".into());
}
fn patch_imm26(w: u32, d: i64) -> u32 {
    w | ((d >> 2) as u32 & 0x03FF_FFFF)
}
fn patch_imm19(w: u32, d: i64) -> u32 {
    w | (((d >> 2) as u32 & 0x7_FFFF) << 5)
}
fn patch_adr(w: u32, d: i64) -> u32 {
    let d = d as u32;
    w | (d & 3) << 29 | ((d >> 2) & 0x7_FFFF) << 5
}
fn b(a: &mut WordAsm, l: WLabel) {
    a.emit_ref(0x1400_0000, format!("b {}", WordAsm::name(l)), l, patch_imm26);
}
fn b_ne(a: &mut WordAsm, l: WLabel) {
    a.emit_ref(0x5400_0001, format!("b.ne {}", WordAsm::name(l)), l, patch_imm19);
}
fn cbnz_w(a: &mut WordAsm, t: u32, l: WLabel) {
    a.emit_ref(0x3500_0000 | t, format!("cbnz w{t}, {}", WordAsm::name(l)), l, patch_imm19);
}
fn adr(a: &mut WordAsm, d: u32, l: WLabel) {
    a.emit_ref(0x1000_0000 | d, format!("adr x{d}, {}", WordAsm::name(l)), l, patch_adr);
}

/// Save the cooperative state into `[x0]` (clobbers `x9`).
fn save_coop(a: &mut WordAsm) {
    for r in (19..=29).step_by(2) {
        pair_x(a, false, r, r + 1, 0, layout::x(r));
    }
    add_imm(a, X9, R31, 0);
    one_x(a, false, X9, 0, layout::SP);
    one_x(a, false, LR, 0, layout::PC);
    for q in (8..=14).step_by(2) {
        pair_q(a, false, q, q + 1, 0, layout::v(q));
    }
    mrs(a, X9, SysReg::Fpcr);
    one_w(a, false, X9, 0, layout::FPCR);
    movz(a, false, X9, layout::VERSION);
    one_w(a, false, X9, 0, layout::VERSION_OFF);
    one_w(a, false, R31, 0, layout::KIND);
}

/// Save the full state into `[x0]` (`x0`'s slot gets `x0_is_one ? 1 : x0`;
/// clobbers `x9` after saving it).
fn save_full(a: &mut WordAsm, x0_is_one: bool) {
    for r in (0..=28).step_by(2) {
        pair_x(a, false, r, r + 1, 0, layout::x(r));
    }
    one_x(a, false, LR, 0, layout::x(LR));
    if x0_is_one {
        movz(a, true, X9, 1);
        one_x(a, false, X9, 0, layout::x(0));
    }
    mrs(a, X9, SysReg::Nzcv);
    one_x(a, false, X9, 0, layout::NZCV);
    add_imm(a, X9, R31, 0);
    one_x(a, false, X9, 0, layout::SP);
    one_x(a, false, LR, 0, layout::PC);
    mrs(a, X9, SysReg::Fpsr);
    one_w(a, false, X9, 0, layout::FPSR);
    mrs(a, X9, SysReg::Fpcr);
    one_w(a, false, X9, 0, layout::FPCR);
    for q in (0..=30).step_by(2) {
        pair_q(a, false, q, q + 1, 0, layout::v(q));
    }
    movz(a, false, X9, 1);
    one_w(a, false, X9, 0, layout::VERSION_OFF);
    one_w(a, false, X9, 0, layout::KIND);
}

/// Resume the context at `x0` (never returns).
fn restore(a: &mut WordAsm) {
    let full = a.label();
    one_w(a, true, X9, 0, layout::KIND);
    cbnz_w(a, X9, full);
    for r in (19..=29).step_by(2) {
        pair_x(a, true, r, r + 1, 0, layout::x(r));
    }
    one_x(a, true, X9, 0, layout::SP);
    add_imm(a, R31, X9, 0);
    for q in (8..=14).step_by(2) {
        pair_q(a, true, q, q + 1, 0, layout::v(q));
    }
    one_w(a, true, X9, 0, layout::FPCR);
    msr(a, SysReg::Fpcr, X9);
    one_x(a, true, X16, 0, layout::PC);
    movz(a, true, 0, 1);
    br(a, X16);

    a.bind(full);
    for q in (0..=30).step_by(2) {
        pair_q(a, true, q, q + 1, 0, layout::v(q));
    }
    one_w(a, true, X9, 0, layout::FPSR);
    msr(a, SysReg::Fpsr, X9);
    one_w(a, true, X9, 0, layout::FPCR);
    msr(a, SysReg::Fpcr, X9);
    one_x(a, true, X9, 0, layout::NZCV);
    msr(a, SysReg::Nzcv, X9);
    one_x(a, true, X9, 0, layout::SP);
    add_imm(a, R31, X9, 0);
    one_x(a, true, X17, 0, layout::PC);
    for r in (2..=14).step_by(2) {
        pair_x(a, true, r, r + 1, 0, layout::x(r));
    }
    one_x(a, true, X16, 0, layout::x(X16));
    for r in (18..=28).step_by(2) {
        pair_x(a, true, r, r + 1, 0, layout::x(r));
    }
    one_x(a, true, LR, 0, layout::x(LR));
    pair_x(a, true, 0, 1, 0, layout::x(0));
    br(a, X17);
}

/// Build the whole runtime.
pub(crate) fn assemble() -> WordAsm {
    use crate::target::x86_64::runtime as names;
    let mut a = WordAsm::new();
    let restore_l = a.label();
    let stub = a.label();

    a.begin(names::SYM_SAVE, true);
    save_coop(&mut a);
    movz(&mut a, true, 0, 0);
    ret(&mut a);

    a.begin(names::SYM_SAVE_FULL, true);
    save_full(&mut a, true);
    movz(&mut a, true, 0, 0);
    ret(&mut a);

    a.begin(names::SYM_RESTORE, true);
    a.bind(restore_l);
    restore(&mut a);

    a.begin(names::SYM_SWITCH, true);
    save_coop(&mut a);
    mov(&mut a, 0, 1);
    b(&mut a, restore_l);

    a.begin(names::SYM_SWITCH_FULL, true);
    save_full(&mut a, false);
    mov(&mut a, 0, 1);
    b(&mut a, restore_l);

    // lf_ctx_init(ctx, stack_top, entry, arg)
    a.begin(names::SYM_INIT, true);
    let zero = a.label();
    add_imm(&mut a, X9, 0, 0);
    add_imm(&mut a, 10, 0, layout::SIZE as u32);
    a.bind(zero);
    // stp xzr, xzr, [x9], #16
    a.emit(0xA880_0000 | 2 << 15 | 31 << 10 | X9 << 5 | 31, "stp xzr, xzr, [x9], #16".into());
    a.emit(0xEB00_001F | 10 << 16 | X9 << 5, "cmp x9, x10".into());
    b_ne(&mut a, zero);
    // and x1, x1, #~15 (N=1, immr=60, imms=59)
    a.emit(0x9240_0000 | 60 << 16 | 59 << 10 | 1 << 5 | 1, "and x1, x1, #0xfffffffffffffff0".into());
    one_x(&mut a, false, 1, 0, layout::SP);
    adr(&mut a, X9, stub);
    one_x(&mut a, false, X9, 0, layout::PC);
    one_x(&mut a, false, 2, 0, layout::x(19));
    one_x(&mut a, false, 3, 0, layout::x(20));
    mrs(&mut a, X9, SysReg::Fpcr);
    one_w(&mut a, false, X9, 0, layout::FPCR);
    movz(&mut a, false, X9, layout::VERSION);
    one_w(&mut a, false, X9, 0, layout::VERSION_OFF);
    ret(&mut a);

    // The fresh-thread trampoline: entry(arg), then exit_group(result).
    a.begin("lf_ctx_thread_start", false);
    a.bind(stub);
    mov(&mut a, 0, 20);
    blr(&mut a, 19);
    movz(&mut a, true, 8, 94);
    a.emit(0xD400_0001, "svc #0".into());
    a.emit(0xD420_0000, "brk #0".into());
    a
}

/// Append the AArch64 context runtime to `obj` (section `.text.lf_rt`).
pub fn emit_context_runtime(obj: &mut ObjectModule) {
    assemble().into_object(obj);
}

/// The AArch64 context runtime as a stand-alone object.
pub fn context_runtime_object() -> ObjectModule {
    let mut obj = ObjectModule::new("lf_rt");
    emit_context_runtime(&mut obj);
    obj
}
