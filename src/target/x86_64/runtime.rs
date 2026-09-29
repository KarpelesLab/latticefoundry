//! Green-thread **context-switching runtime** for x86-64 Linux, emitted as
//! machine code by LatticeFoundry itself (no C runtime, no libc).
//!
//! A front end whose language has green threads (Lode) links this object next to
//! its own compiled module and calls the routines below as ordinary external
//! functions. Because they are *external* functions to the IR — declared, never
//! defined in the module — every optimizer pass already treats a call to one as
//! an opaque call to an unknown function: a full memory clobber and a barrier
//! nothing is moved across, and nothing is inlined into or out of. No IR opcode
//! or pass had to learn about context switching.
//!
//! [`emit_context_runtime`] appends the routines to an existing
//! [`ObjectModule`]; [`context_runtime_object`] returns them as a stand-alone
//! object ready for [`crate::link::link_executable`].
//!
//! # The context layout (version [`layout::VERSION`] = 1)
//!
//! An `LfCtx` is [`layout::SIZE`] = 672 bytes and **must be 16-byte aligned**
//! (`fxsave64`/`fxrstor64` fault otherwise). In IR, declare it as
//! `[42 x i128]` (16-aligned) or place it in `mmap`ed memory.
//!
//! | offset          | size | field                                                      |
//! |-----------------|------|------------------------------------------------------------|
//! | `0x000 + 8*n`   | 8    | GPR number `n` (x86 encoding order: `rax rcx rdx rbx rsp rbp rsi rdi r8..r15`) |
//! | `0x080`         | 8    | `rip` — where execution resumes                            |
//! | `0x088`         | 8    | `rflags`                                                   |
//! | `0x090`         | 4    | layout version (`1`)                                       |
//! | `0x094`         | 4    | kind: `0` = cooperative, `1` = full                        |
//! | `0x098`         | 8    | reserved (zero)                                            |
//! | `0x0A0`         | 512  | `fxsave64` image: x87, MXCSR, `xmm0..xmm15`                |
//!
//! The FP/vector area is exactly the 64-bit `FXSAVE` format, which is also the
//! first 512 bytes of the kernel's signal-frame `fpstate` (`struct
//! _fpstate_64`), so the signal path copies it verbatim.
//!
//! **Kinds.** A *cooperative* context (saved by [`SYM_SAVE`] / [`SYM_SWITCH`] /
//! [`SYM_INIT`]) is only valid in what the System V ABI makes callee-saved
//! across a call: `rbx rbp r12..r15`, `rsp`, `rip`, the x87 control word and
//! MXCSR (written at their `fxsave` positions, offsets `0x0A0` and `0x0B8`). A
//! *full* context (saved by [`SYM_SAVE_FULL`] / [`SYM_SWITCH_FULL`] or taken
//! from a signal's `ucontext`) is valid in every field. Every restore dispatches
//! on the kind, so any mix works: a cooperatively-switched thread can resume a
//! preempted one and vice versa.
//!
//! **AVX.** Version 1 keeps x87/SSE state only (`FXSAVE`, the baseline every
//! x86-64 CPU has); the upper halves of `ymm`/`zmm` and the AVX-512 mask
//! registers are *not* part of the context. LatticeFoundry emits no AVX code
//! today, so no LF-compiled thread has live state there. When a context is
//! written into a signal frame, the frame's `XSTATE_BV` is edited so the kernel
//! resets the AVX components to their initial (zero) state rather than leaking
//! the interrupted thread's. A later layout version can append an `XSAVE` area
//! sized from CPUID leaf `0xD` (the reserved word at `0x098` then flags it).
//!
//! # The routines (all System V calling convention)
//!
//! | symbol                     | signature                                              |
//! |----------------------------|--------------------------------------------------------|
//! | [`SYM_SAVE`]               | `(ctx: ptr) -> i64` — cooperative save; 0 now, 1 on resume (like `setjmp`) |
//! | [`SYM_SAVE_FULL`]          | `(ctx: ptr) -> i64` — full save; 0 now, 1 on resume    |
//! | [`SYM_RESTORE`]            | `(ctx: ptr) -> !` — resume a context of either kind    |
//! | [`SYM_SWITCH`]             | `(from: ptr, to: ptr)` — cooperative save into `from`, resume `to` |
//! | [`SYM_SWITCH_FULL`]        | `(from: ptr, to: ptr)` — full save (every register, as at the call) |
//! | [`SYM_INIT`]               | `(ctx: ptr, stack_top: ptr, entry: ptr, arg: i64)` — a fresh thread |
//! | [`SYM_FROM_UCONTEXT`]      | `(ctx: ptr, uc: ptr)` — capture a signal's interrupted state |
//! | [`SYM_TO_UCONTEXT`]        | `(uc: ptr, ctx: ptr)` — make `rt_sigreturn` resume `ctx` |
//! | [`SYM_PREEMPT`]            | `(uc: ptr, from: ptr, to: ptr)` — both of the above    |
//! | [`SYM_UC_IN_RUNTIME`]      | `(uc: ptr) -> i64` — 1 if the signal hit inside these routines |
//! | [`SYM_SIG_INSTALL`]        | `(signo: i64, handler: ptr, flags: i64) -> i64` — `rt_sigaction` |
//! | [`SYM_SIG_RESTORER`]       | the `SA_RESTORER` trampoline: `rt_sigreturn`           |
//!
//! `lf_ctx_save`/`lf_ctx_save_full` return twice, with `setjmp`'s caveat: after
//! the second return, the saving function may only rely on values it had
//! before the save and did not change afterwards (its spill slots can have been
//! reused in between). [`SYM_SWITCH`] has no such caveat — it is the
//! green-thread switch.
//!
//! **A fresh thread** ([`SYM_INIT`]): `ctx` is zeroed, then set to start at an
//! internal trampoline on `stack_top` (rounded down to 16) that calls
//! `entry(arg)` with a correctly aligned stack; if `entry` returns, the
//! trampoline calls `exit_group` with its return value.
//!
//! **Full restore without a free register.** Resuming a full context must load
//! all sixteen GPRs, `rflags` and `rip`. The restore writes `rflags` and `rip`
//! *below the target's 128-byte red zone* (at `rsp - 144` / `rsp - 136`), points
//! `rsp` there, loads every GPR, then `popfq` and `ret 128` — which pops `rip`
//! and lands `rsp` exactly on the target's value. Nothing live in the target's
//! red zone is disturbed, so a thread preempted by a signal in the middle of a
//! leaf function resumes intact.
//!
//! # Signal preemption, end to end
//!
//! 1. Install a handler with [`SYM_SIG_INSTALL`]: it issues `rt_sigaction` with
//!    `SA_SIGINFO | SA_RESTORER | flags` and [`SYM_SIG_RESTORER`] as the
//!    restorer (without libc nobody else provides one); arm a timer with
//!    `setitimer` through the IR `syscall` instruction.
//! 2. The kernel delivers the signal on the interrupted thread's stack and calls
//!    `handler(signo, info, uc)`, where `uc` is the `ucontext_t` holding the
//!    interrupted registers (`uc_mcontext`, a `struct sigcontext`) and a pointer
//!    to the FP/vector state (`fpstate`).
//! 3. The handler calls [`SYM_PREEMPT`]`(uc, current, next)`: the interrupted
//!    state is copied into `current` (a *full* context) and `next`'s state is
//!    written into the `ucontext` and its `fpstate`.
//! 4. The handler returns into the restorer; `rt_sigreturn` loads the edited
//!    frame, so the kernel itself resumes `next`, with the signal mask saved in
//!    the frame (normally: nothing blocked).
//! 5. Later, some thread switches back to `current` (with [`SYM_SWITCH`] or
//!    through another signal); its full context resumes it exactly where the
//!    signal hit, every register and `xmm` intact.
//!
//! The runtime is not reentrant with respect to the green-thread scheduler's own
//! data: a handler must not preempt a thread that is in the middle of a switch.
//! [`SYM_UC_IN_RUNTIME`] reports a signal that landed inside these routines;
//! scheduler code of its own needs a "preemption disabled" flag the handler
//! checks.
//!
//! The kernel structures are used per the Linux x86-64 UAPI ABI
//! (`uc_mcontext` at offset 40 of `ucontext_t`; the `sigcontext` register order
//! `r8..r15 rdi rsi rbp rbx rdx rax rcx rsp rip eflags`, `fpstate` at `+184`; the
//! `_fpx_sw_bytes` magic `0x46505853` at `fpstate + 464` announcing the XSAVE
//! header whose `XSTATE_BV` is at `fpstate + 512`), and the instruction
//! encodings per the Intel SDM (tenet T1).

use crate::mc::emit::{Emitter, Label};
use crate::mc::object::{ObjectModule, RelocKind, SectionKind, Symbol, SymbolBinding, SymbolType};

use super::encode::{mem, mov_ri, mov_rr, rex};
use super::regs::{R8, R9, R12, R13, R14, R15, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP};

/// The `LfCtx` layout for x86-64 (see the [module docs](self)).
pub mod layout {
    /// The layout version stored at [`VERSION_OFF`].
    pub const VERSION: u32 = 1;
    /// Total size in bytes.
    pub const SIZE: usize = 0x2A0;
    /// Required alignment in bytes.
    pub const ALIGN: usize = 16;
    /// Offset of GPR number `n` (x86 encoding order, `rax` = 0 … `r15` = 15).
    pub const fn gpr(n: u16) -> usize {
        8 * n as usize
    }
    /// Offset of the saved `rsp` (GPR 4).
    pub const RSP: usize = 0x20;
    /// Offset of the saved `rip`.
    pub const RIP: usize = 0x80;
    /// Offset of the saved `rflags`.
    pub const RFLAGS: usize = 0x88;
    /// Offset of the `u32` layout version.
    pub const VERSION_OFF: usize = 0x90;
    /// Offset of the `u32` context kind ([`KIND_COOP`] / [`KIND_FULL`]).
    pub const KIND: usize = 0x94;
    /// Offset of the 512-byte `fxsave64` area (16-byte aligned).
    pub const FXSAVE: usize = 0xA0;
    /// Size of the `fxsave64` area.
    pub const FXSAVE_SIZE: usize = 512;
    /// Offset of the x87 control word (inside the `fxsave` area).
    pub const FCW: usize = FXSAVE;
    /// Offset of MXCSR (inside the `fxsave` area).
    pub const MXCSR: usize = FXSAVE + 24;
    /// Offset of `xmm0` (inside the `fxsave` area); `xmm n` is at `XMM + 16*n`.
    pub const XMM: usize = FXSAVE + 160;
    /// Kind: only callee-saved state is valid.
    pub const KIND_COOP: u32 = 0;
    /// Kind: every field is valid.
    pub const KIND_FULL: u32 = 1;
}

/// `lf_ctx_save(ctx) -> i64`: cooperative save; returns 0, then 1 on resume.
pub const SYM_SAVE: &str = "lf_ctx_save";
/// `lf_ctx_save_full(ctx) -> i64`: full save; returns 0, then 1 on resume.
pub const SYM_SAVE_FULL: &str = "lf_ctx_save_full";
/// `lf_ctx_restore(ctx) -> !`: resume a context of either kind.
pub const SYM_RESTORE: &str = "lf_ctx_restore";
/// `lf_ctx_switch(from, to)`: cooperative save into `from`, resume `to`.
pub const SYM_SWITCH: &str = "lf_ctx_switch";
/// `lf_ctx_switch_full(from, to)`: full save into `from`, resume `to`.
pub const SYM_SWITCH_FULL: &str = "lf_ctx_switch_full";
/// `lf_ctx_init(ctx, stack_top, entry, arg)`: a context that runs `entry(arg)`.
pub const SYM_INIT: &str = "lf_ctx_init";
/// `lf_ctx_from_ucontext(ctx, uc)`: copy a signal's interrupted state into `ctx`.
pub const SYM_FROM_UCONTEXT: &str = "lf_ctx_from_ucontext";
/// `lf_ctx_to_ucontext(uc, ctx)`: make the signal return resume `ctx`.
pub const SYM_TO_UCONTEXT: &str = "lf_ctx_to_ucontext";
/// `lf_ctx_preempt(uc, from, to)`: `from_ucontext(from, uc)` + `to_ucontext(uc, to)`.
pub const SYM_PREEMPT: &str = "lf_ctx_preempt";
/// `lf_ctx_uc_in_runtime(uc) -> i64`: whether the interrupted `rip` is in this runtime.
pub const SYM_UC_IN_RUNTIME: &str = "lf_ctx_uc_in_runtime";
/// `lf_sig_install(signo, handler, flags) -> i64`: `rt_sigaction` with our restorer.
pub const SYM_SIG_INSTALL: &str = "lf_sig_install";
/// The `SA_RESTORER` trampoline (`rt_sigreturn`).
pub const SYM_SIG_RESTORER: &str = "lf_sig_restorer";

/// `SA_SIGINFO`: the handler takes `(signo, siginfo*, ucontext*)`.
pub const SA_SIGINFO: u64 = 0x4;
/// `SA_RESTORER`: `sa_restorer` is valid (required on x86-64 without libc).
pub const SA_RESTORER: u64 = 0x0400_0000;
/// `SA_RESTART`: restart interrupted system calls.
pub const SA_RESTART: u64 = 0x1000_0000;

// --- Linux x86-64 UAPI offsets ----------------------------------------------

/// Offset of `uc_mcontext` (`struct sigcontext`) in `ucontext_t`.
const UC_MCONTEXT: i32 = 40;
/// `struct sigcontext`: offset of each register, as (x86 register number, offset).
const SC_GPRS: [(u16, i32); 16] = [
    (R8, 0),
    (R9, 8),
    (10, 16),
    (11, 24),
    (R12, 32),
    (R13, 40),
    (R14, 48),
    (R15, 56),
    (RDI, 64),
    (RSI, 72),
    (RBP, 80),
    (RBX, 88),
    (RDX, 96),
    (RAX, 104),
    (RCX, 112),
    (RSP, 120),
];
/// `struct sigcontext`: `rip`.
const SC_RIP: i32 = 128;
/// `struct sigcontext`: `eflags`.
const SC_EFLAGS: i32 = 136;
/// `struct sigcontext`: the `fpstate` pointer.
const SC_FPSTATE: i32 = 184;
/// `fpstate`: `sw_reserved.magic1`, present when an XSAVE header follows.
const FP_SW_MAGIC1_OFF: i32 = 464;
/// `FP_XSTATE_MAGIC1`.
const FP_XSTATE_MAGIC1: u32 = 0x4650_5853;
/// `fpstate`: the XSAVE header's `xfeatures` (`XSTATE_BV`).
const FP_XSTATE_BV: i32 = 512;
/// `XSTATE_BV` bits of the AVX family (YMM_Hi128, opmask, ZMM_Hi256, Hi16_ZMM),
/// cleared when a version-1 context (which has no AVX state) is written.
const XSTATE_AVX_BITS: u32 = 0x4 | 0x20 | 0x40 | 0x80;

const NR_RT_SIGACTION: u64 = 13;
const NR_RT_SIGRETURN: u64 = 15;
const NR_EXIT_GROUP: u64 = 231;

// ---------------------------------------------------------------------------
// A tiny assembler over the Emitter for the forms these routines use
// ---------------------------------------------------------------------------

/// Instruction builders for hand-written x86-64 runtime code (and the test
/// harnesses that exercise it). Each method emits one instruction.
#[derive(Debug, Default)]
pub(crate) struct X86Asm {
    pub(crate) e: Emitter,
}

impl X86Asm {
    pub(crate) fn new() -> X86Asm {
        X86Asm { e: Emitter::new() }
    }
    /// `mov r64, [base + disp]`.
    pub(crate) fn load(&mut self, dst: u16, base: u16, disp: i32) {
        mem(&mut self.e, &[0x8B], dst as u8, base as u8, disp, true, false);
    }
    /// `mov [base + disp], r64`.
    pub(crate) fn store(&mut self, base: u16, disp: i32, src: u16) {
        mem(&mut self.e, &[0x89], src as u8, base as u8, disp, true, false);
    }
    /// `mov qword [base + disp], simm32`.
    pub(crate) fn store_imm64(&mut self, base: u16, disp: i32, imm: i32) {
        mem(&mut self.e, &[0xC7], 0, base as u8, disp, true, false);
        self.e.u32(imm as u32);
    }
    /// `mov dword [base + disp], imm32`.
    pub(crate) fn store_imm32(&mut self, base: u16, disp: i32, imm: u32) {
        mem(&mut self.e, &[0xC7], 0, base as u8, disp, false, false);
        self.e.u32(imm);
    }
    /// `mov word [base + disp], imm16`.
    pub(crate) fn store_imm16(&mut self, base: u16, disp: i32, imm: u16) {
        self.e.u8(0x66);
        mem(&mut self.e, &[0xC7], 0, base as u8, disp, false, false);
        self.e.u16(imm);
    }
    /// `lea r64, [base + disp]`.
    pub(crate) fn lea(&mut self, dst: u16, base: u16, disp: i32) {
        mem(&mut self.e, &[0x8D], dst as u8, base as u8, disp, true, false);
    }
    /// `mov dst, src` (64-bit).
    pub(crate) fn mov(&mut self, dst: u16, src: u16) {
        mov_rr(&mut self.e, dst as u8, src as u8, true);
    }
    /// `mov r, imm` (shortest form).
    pub(crate) fn mov_imm(&mut self, dst: u16, imm: u64) {
        mov_ri(&mut self.e, dst as u8, imm);
    }
    /// `xor r32, r32` (zero a register).
    pub(crate) fn zero(&mut self, r: u16) {
        super::encode::alu_rr(&mut self.e, 0x31, r as u8, r as u8, false);
    }
    /// A group-1 ALU op `op r64, simm32` (`REX.W 81 /ext id`): ext 0 = add,
    /// 1 = or, 4 = and, 5 = sub, 7 = cmp.
    pub(crate) fn alu_imm(&mut self, ext: u8, r: u16, imm: i32) {
        self.e.u8(rex(true, false, false, r >= 8));
        self.e.u8(0x81);
        self.e.u8(0xC0 | (ext << 3) | (r as u8 & 7));
        self.e.u32(imm as u32);
    }
    /// `cmp dword [base + disp], imm32`.
    pub(crate) fn cmp_mem32_imm(&mut self, base: u16, disp: i32, imm: u32) {
        mem(&mut self.e, &[0x81], 7, base as u8, disp, false, false);
        self.e.u32(imm);
    }
    /// `test r64, r64`.
    pub(crate) fn test(&mut self, a: u16, b: u16) {
        super::encode::alu_rr(&mut self.e, 0x85, a as u8, b as u8, true);
    }
    /// `cmp a, b` (64-bit; flags from `a - b`).
    pub(crate) fn cmp(&mut self, a: u16, b: u16) {
        super::encode::alu_rr(&mut self.e, 0x39, a as u8, b as u8, true);
    }
    /// `test al, imm8`.
    pub(crate) fn test_al(&mut self, imm: u8) {
        self.e.u8(0xA8);
        self.e.u8(imm);
    }
    /// `jcc label` (`0F 80+cc rel32`): cc 2 = b, 3 = ae, 4 = e, 5 = ne.
    pub(crate) fn jcc(&mut self, cc: u8, target: Label) {
        self.e.u8(0x0F);
        self.e.u8(0x80 + cc);
        self.e.reference_label(RelocKind::Pc32, target, 0);
    }
    /// `jmp label` (`E9 rel32`).
    pub(crate) fn jmp(&mut self, target: Label) {
        self.e.u8(0xE9);
        self.e.reference_label(RelocKind::Pc32, target, 0);
    }
    /// `call label` (`E8 rel32`).
    pub(crate) fn call_label(&mut self, target: Label) {
        self.e.u8(0xE8);
        self.e.reference_label(RelocKind::Pc32, target, 0);
    }
    /// `call r64` (`FF /2`).
    pub(crate) fn call_reg(&mut self, r: u16) {
        if r >= 8 {
            self.e.u8(0x41);
        }
        self.e.u8(0xFF);
        self.e.u8(0xD0 | (r as u8 & 7));
    }
    /// `jmp qword [base + disp]` (`FF /4`).
    pub(crate) fn jmp_mem(&mut self, base: u16, disp: i32) {
        mem(&mut self.e, &[0xFF], 4, base as u8, disp, false, false);
    }
    /// `lea r64, [rip + label]`.
    pub(crate) fn lea_label(&mut self, dst: u16, target: Label) {
        self.e.u8(rex(true, dst >= 8, false, false));
        self.e.u8(0x8D);
        self.e.u8(((dst as u8) & 7) << 3 | 0b101);
        self.e.reference_label(RelocKind::Pc32, target, 0);
    }
    /// `pushfq`.
    pub(crate) fn pushfq(&mut self) {
        self.e.u8(0x9C);
    }
    /// `popfq`.
    pub(crate) fn popfq(&mut self) {
        self.e.u8(0x9D);
    }
    /// `push r64`.
    pub(crate) fn push(&mut self, r: u16) {
        super::encode::push_r(&mut self.e, r as u8);
    }
    /// `pop r64`.
    pub(crate) fn pop(&mut self, r: u16) {
        super::encode::pop_r(&mut self.e, r as u8);
    }
    /// `pop qword [base + disp]` (`8F /0`).
    pub(crate) fn pop_mem(&mut self, base: u16, disp: i32) {
        mem(&mut self.e, &[0x8F], 0, base as u8, disp, false, false);
    }
    /// `ret`.
    pub(crate) fn ret(&mut self) {
        self.e.u8(0xC3);
    }
    /// `ret imm16` (pop `rip`, then add `imm16` to `rsp`).
    pub(crate) fn ret_imm(&mut self, n: u16) {
        self.e.u8(0xC2);
        self.e.u16(n);
    }
    /// `syscall`.
    pub(crate) fn syscall(&mut self) {
        self.e.bytes(&[0x0F, 0x05]);
    }
    /// `ud2`.
    pub(crate) fn ud2(&mut self) {
        self.e.bytes(&[0x0F, 0x0B]);
    }
    /// `rep stosq` (`[rdi..] = rax`, `rcx` times).
    pub(crate) fn rep_stosq(&mut self) {
        self.e.bytes(&[0xF3, 0x48, 0xAB]);
    }
    /// `rep movsq` (`[rdi..] = [rsi..]`, `rcx` qwords).
    pub(crate) fn rep_movsq(&mut self) {
        self.e.bytes(&[0xF3, 0x48, 0xA5]);
    }
    /// `fxsave64 [base + disp]` (`REX.W 0F AE /0`).
    pub(crate) fn fxsave64(&mut self, base: u16, disp: i32) {
        mem(&mut self.e, &[0x0F, 0xAE], 0, base as u8, disp, true, false);
    }
    /// `fxrstor64 [base + disp]` (`REX.W 0F AE /1`).
    pub(crate) fn fxrstor64(&mut self, base: u16, disp: i32) {
        mem(&mut self.e, &[0x0F, 0xAE], 1, base as u8, disp, true, false);
    }
    /// `fnstcw [base + disp]` (`D9 /7`).
    pub(crate) fn fnstcw(&mut self, base: u16, disp: i32) {
        mem(&mut self.e, &[0xD9], 7, base as u8, disp, false, false);
    }
    /// `fldcw [base + disp]` (`D9 /5`).
    pub(crate) fn fldcw(&mut self, base: u16, disp: i32) {
        mem(&mut self.e, &[0xD9], 5, base as u8, disp, false, false);
    }
    /// `stmxcsr [base + disp]` (`0F AE /3`).
    pub(crate) fn stmxcsr(&mut self, base: u16, disp: i32) {
        mem(&mut self.e, &[0x0F, 0xAE], 3, base as u8, disp, false, false);
    }
    /// `ldmxcsr [base + disp]` (`0F AE /2`).
    pub(crate) fn ldmxcsr(&mut self, base: u16, disp: i32) {
        mem(&mut self.e, &[0x0F, 0xAE], 2, base as u8, disp, false, false);
    }
}

/// `disp` as the `i32` displacement of a layout offset.
fn d(off: usize) -> i32 {
    off as i32
}

/// The callee-saved GPRs of the System V ABI (besides `rsp`).
const CALLEE_SAVED: [u16; 6] = [RBX, RBP, R12, R13, R14, R15];

/// Store the cooperative state of the caller into `[rdi]`: callee-saved GPRs,
/// the resume point (return address) and the post-return `rsp`, the x87 control
/// word and MXCSR, the version and kind. Clobbers `rax`.
fn emit_save_coop(a: &mut X86Asm) {
    a.load(RAX, RSP, 0);
    a.store(RDI, d(layout::RIP), RAX);
    a.lea(RAX, RSP, 8);
    a.store(RDI, d(layout::RSP), RAX);
    for r in CALLEE_SAVED {
        a.store(RDI, d(layout::gpr(r)), r);
    }
    a.fnstcw(RDI, d(layout::FCW));
    a.stmxcsr(RDI, d(layout::MXCSR));
    a.store_imm32(RDI, d(layout::VERSION_OFF), layout::VERSION);
    a.store_imm32(RDI, d(layout::KIND), layout::KIND_COOP);
}

/// Store the full state of the caller into `[rdi]` — every GPR as it was at the
/// call (`rax` replaced by `rax_value` when given), `rflags`, the resume point,
/// the post-return `rsp` and the `fxsave64` image. Clobbers `rax` after saving it.
fn emit_save_full(a: &mut X86Asm, rax_value: Option<i32>) {
    a.pushfq();
    a.pop_mem(RDI, d(layout::RFLAGS));
    for r in 0..16u16 {
        if r == RSP || r == RAX {
            continue;
        }
        a.store(RDI, d(layout::gpr(r)), r);
    }
    match rax_value {
        Some(v) => a.store_imm64(RDI, d(layout::gpr(RAX)), v),
        None => a.store(RDI, d(layout::gpr(RAX)), RAX),
    }
    a.load(RAX, RSP, 0);
    a.store(RDI, d(layout::RIP), RAX);
    a.lea(RAX, RSP, 8);
    a.store(RDI, d(layout::RSP), RAX);
    a.fxsave64(RDI, d(layout::FXSAVE));
    a.store_imm32(RDI, d(layout::VERSION_OFF), layout::VERSION);
    a.store_imm32(RDI, d(layout::KIND), layout::KIND_FULL);
}

/// `lf_ctx_restore` body: resume the context at `rdi` (never returns).
fn emit_restore(a: &mut X86Asm) {
    let full = a.e.create_label();
    a.cmp_mem32_imm(RDI, d(layout::KIND), layout::KIND_COOP);
    a.jcc(5, full); // jne full

    // Cooperative: callee-saved state, rsp, then jump to rip with rax = 1.
    for r in CALLEE_SAVED {
        a.load(r, RDI, d(layout::gpr(r)));
    }
    a.fldcw(RDI, d(layout::FCW));
    a.ldmxcsr(RDI, d(layout::MXCSR));
    a.load(RSP, RDI, d(layout::RSP));
    a.mov_imm(RAX, 1);
    a.jmp_mem(RDI, d(layout::RIP));

    // Full: stage rflags + rip below the target's red zone, load every GPR,
    // `popfq`, `ret 128`.
    a.e.bind_label(full);
    a.fxrstor64(RDI, d(layout::FXSAVE));
    a.load(RAX, RDI, d(layout::RSP));
    a.alu_imm(5, RAX, 144); // sub rax, 144
    a.load(RCX, RDI, d(layout::RIP));
    a.store(RAX, 8, RCX);
    a.load(RCX, RDI, d(layout::RFLAGS));
    a.store(RAX, 0, RCX);
    a.mov(RSP, RAX);
    for r in 0..16u16 {
        if r == RSP || r == RDI {
            continue;
        }
        a.load(r, RDI, d(layout::gpr(r)));
    }
    a.load(RDI, RDI, d(layout::gpr(RDI)));
    a.popfq();
    a.ret_imm(128);
}

/// The byte range and name of one emitted routine.
struct Routine {
    name: &'static str,
    start: u64,
    end: u64,
    global: bool,
}

/// Append the context-switching runtime to `obj` as a new `.text.lf_rt`
/// section, defining the global function symbols listed in the
/// [module docs](self).
pub fn emit_context_runtime(obj: &mut ObjectModule) {
    let mut a = X86Asm::new();
    let mut routines: Vec<Routine> = Vec::new();
    let rt_start = a.e.create_label();
    let rt_end = a.e.create_label();
    let restore = a.e.create_label();
    let start_stub = a.e.create_label();
    let from_uc = a.e.create_label();
    let to_uc = a.e.create_label();
    let restorer = a.e.create_label();
    a.e.bind_label(rt_start);

    let begin = |a: &mut X86Asm, routines: &mut Vec<Routine>, name: &'static str, global: bool| {
        while !a.e.offset().is_multiple_of(16) {
            a.e.u8(0xCC); // int3 padding between routines
        }
        routines.push(Routine { name, start: a.e.offset(), end: 0, global });
    };
    let end = |a: &X86Asm, routines: &mut Vec<Routine>| {
        routines.last_mut().expect("a routine is open").end = a.e.offset();
    };

    // lf_ctx_save(ctx) -> i64
    begin(&mut a, &mut routines, SYM_SAVE, true);
    emit_save_coop(&mut a);
    a.zero(RAX);
    a.ret();
    end(&a, &mut routines);

    // lf_ctx_save_full(ctx) -> i64 (the saved rax is 1: the resumed return value)
    begin(&mut a, &mut routines, SYM_SAVE_FULL, true);
    emit_save_full(&mut a, Some(1));
    a.zero(RAX);
    a.ret();
    end(&a, &mut routines);

    // lf_ctx_restore(ctx) -> !
    begin(&mut a, &mut routines, SYM_RESTORE, true);
    a.e.bind_label(restore);
    emit_restore(&mut a);
    end(&a, &mut routines);

    // lf_ctx_switch(from, to)
    begin(&mut a, &mut routines, SYM_SWITCH, true);
    emit_save_coop(&mut a);
    a.mov(RDI, RSI);
    a.jmp(restore);
    end(&a, &mut routines);

    // lf_ctx_switch_full(from, to)
    begin(&mut a, &mut routines, SYM_SWITCH_FULL, true);
    emit_save_full(&mut a, None);
    a.mov(RDI, RSI);
    a.jmp(restore);
    end(&a, &mut routines);

    // lf_ctx_init(ctx, stack_top, entry, arg)
    begin(&mut a, &mut routines, SYM_INIT, true);
    a.mov(R8, RDI);
    a.mov(R9, RCX);
    a.zero(RAX);
    a.mov_imm(RCX, (layout::SIZE / 8) as u64);
    a.rep_stosq();
    a.mov(RDI, R8);
    a.alu_imm(4, RSI, -16); // and rsi, -16
    a.store(RDI, d(layout::RSP), RSI);
    a.lea_label(RAX, start_stub);
    a.store(RDI, d(layout::RIP), RAX);
    a.store(RDI, d(layout::gpr(R12)), RDX);
    a.store(RDI, d(layout::gpr(R13)), R9);
    a.store_imm64(RDI, d(layout::RFLAGS), 0x202);
    a.fnstcw(RDI, d(layout::FCW));
    a.stmxcsr(RDI, d(layout::MXCSR));
    a.store_imm32(RDI, d(layout::VERSION_OFF), layout::VERSION);
    a.ret();
    end(&a, &mut routines);

    // The fresh-thread trampoline: entry(arg), then exit_group(result).
    begin(&mut a, &mut routines, "lf_ctx_thread_start", false);
    a.e.bind_label(start_stub);
    a.mov(RDI, R13);
    a.call_reg(R12);
    a.mov(RDI, RAX);
    a.mov_imm(RAX, NR_EXIT_GROUP);
    a.syscall();
    a.ud2();
    end(&a, &mut routines);

    // lf_ctx_from_ucontext(ctx, uc)
    begin(&mut a, &mut routines, SYM_FROM_UCONTEXT, true);
    a.e.bind_label(from_uc);
    for (r, off) in SC_GPRS {
        a.load(RAX, RSI, UC_MCONTEXT + off);
        a.store(RDI, d(layout::gpr(r)), RAX);
    }
    a.load(RAX, RSI, UC_MCONTEXT + SC_RIP);
    a.store(RDI, d(layout::RIP), RAX);
    a.load(RAX, RSI, UC_MCONTEXT + SC_EFLAGS);
    a.store(RDI, d(layout::RFLAGS), RAX);
    a.store_imm32(RDI, d(layout::VERSION_OFF), layout::VERSION);
    a.store_imm32(RDI, d(layout::KIND), layout::KIND_FULL);
    {
        let nofp = a.e.create_label();
        let done = a.e.create_label();
        let sse_ok = a.e.create_label();
        a.load(RSI, RSI, UC_MCONTEXT + SC_FPSTATE);
        a.test(RSI, RSI);
        a.jcc(4, nofp); // jz
        a.mov(RDX, RDI);
        a.lea(RDI, RDX, d(layout::FXSAVE));
        a.mov_imm(RCX, (layout::FXSAVE_SIZE / 8) as u64);
        a.rep_movsq();
        a.alu_imm(5, RSI, layout::FXSAVE_SIZE as i32); // rsi back to fpstate
        a.mov(RDI, RDX);
        // An XSAVE frame whose XSTATE_BV marks x87 / SSE as in their initial
        // state may hold stale bytes there: normalize to the initial values.
        a.cmp_mem32_imm(RSI, FP_SW_MAGIC1_OFF, FP_XSTATE_MAGIC1);
        a.jcc(5, done);
        a.load(RAX, RSI, FP_XSTATE_BV);
        a.test_al(2);
        a.jcc(5, sse_ok);
        a.lea(RDI, RDX, d(layout::XMM));
        a.push(RAX);
        a.zero(RAX);
        a.mov_imm(RCX, 32);
        a.rep_stosq();
        a.pop(RAX);
        a.mov(RDI, RDX);
        a.e.bind_label(sse_ok);
        a.test_al(1);
        a.jcc(5, done);
        a.store_imm32(RDX, d(layout::FCW), 0x037F); // fcw = 0x37f, fsw = 0
        a.store_imm32(RDX, d(layout::FCW + 4), 0); // ftw, fop
        a.store_imm64(RDX, d(layout::FXSAVE + 8), 0); // fip
        a.store_imm64(RDX, d(layout::FXSAVE + 16), 0); // fdp
        a.lea(RDI, RDX, d(layout::FXSAVE + 32));
        a.zero(RAX);
        a.mov_imm(RCX, 16);
        a.rep_stosq();
        a.mov(RDI, RDX);
        a.e.bind_label(done);
        a.ret();
        // No FP frame: the initial x87/SSE state.
        a.e.bind_label(nofp);
        a.mov(RDX, RDI);
        a.lea(RDI, RDX, d(layout::FXSAVE));
        a.zero(RAX);
        a.mov_imm(RCX, (layout::FXSAVE_SIZE / 8) as u64);
        a.rep_stosq();
        a.mov(RDI, RDX);
        a.store_imm32(RDI, d(layout::FCW), 0x037F);
        a.store_imm32(RDI, d(layout::MXCSR), 0x1F80);
        a.ret();
    }
    end(&a, &mut routines);

    // lf_ctx_to_ucontext(uc, ctx)
    begin(&mut a, &mut routines, SYM_TO_UCONTEXT, true);
    a.e.bind_label(to_uc);
    for (r, off) in SC_GPRS {
        a.load(RAX, RSI, d(layout::gpr(r)));
        a.store(RDI, UC_MCONTEXT + off, RAX);
    }
    a.load(RAX, RSI, d(layout::RIP));
    a.store(RDI, UC_MCONTEXT + SC_RIP, RAX);
    a.load(RAX, RSI, d(layout::RFLAGS));
    a.store(RDI, UC_MCONTEXT + SC_EFLAGS, RAX);
    {
        let done = a.e.create_label();
        let full = a.e.create_label();
        a.load(RDX, RDI, UC_MCONTEXT + SC_FPSTATE);
        a.test(RDX, RDX);
        a.jcc(4, done);
        a.mov(R9, RSI);
        // Copy the legacy region up to (not including) the kernel's
        // `sw_reserved` bytes, which describe the frame and must survive.
        a.mov(RDI, RDX);
        a.lea(RSI, R9, d(layout::FXSAVE));
        a.mov_imm(RCX, (FP_SW_MAGIC1_OFF / 8) as u64);
        a.rep_movsq();
        // A cooperative context's x87 status/tag words were never saved: an
        // empty x87 stack.
        a.cmp_mem32_imm(R9, d(layout::KIND), layout::KIND_COOP);
        a.jcc(5, full);
        a.store_imm32(RDX, 2, 0); // fsw, ftw
        a.store_imm16(RDX, 6, 0); // fop
        a.e.bind_label(full);
        // XSAVE frame: x87 + SSE come from our image; AVX state resets.
        a.cmp_mem32_imm(RDX, FP_SW_MAGIC1_OFF, FP_XSTATE_MAGIC1);
        a.jcc(5, done);
        a.load(RAX, RDX, FP_XSTATE_BV);
        a.alu_imm(1, RAX, 3); // or rax, 3
        a.alu_imm(4, RAX, !(XSTATE_AVX_BITS as i32)); // and rax, ~avx
        a.store(RDX, FP_XSTATE_BV, RAX);
        a.e.bind_label(done);
        a.ret();
    }
    end(&a, &mut routines);

    // lf_ctx_preempt(uc, from, to)
    begin(&mut a, &mut routines, SYM_PREEMPT, true);
    a.push(RDX);
    a.push(RDI);
    a.mov(RAX, RDI);
    a.mov(RDI, RSI);
    a.mov(RSI, RAX);
    a.call_label(from_uc);
    a.pop(RDI);
    a.pop(RSI);
    a.jmp(to_uc);
    end(&a, &mut routines);

    // lf_ctx_uc_in_runtime(uc) -> i64
    begin(&mut a, &mut routines, SYM_UC_IN_RUNTIME, true);
    {
        let out = a.e.create_label();
        a.load(RDX, RDI, UC_MCONTEXT + SC_RIP);
        a.zero(RAX);
        a.lea_label(RCX, rt_start);
        a.cmp(RDX, RCX);
        a.jcc(2, out); // jb
        a.lea_label(RCX, rt_end);
        a.cmp(RDX, RCX);
        a.jcc(3, out); // jae
        a.mov_imm(RAX, 1);
        a.e.bind_label(out);
        a.ret();
    }
    end(&a, &mut routines);

    // lf_sig_install(signo, handler, flags) -> i64
    begin(&mut a, &mut routines, SYM_SIG_INSTALL, true);
    // struct kernel_sigaction { handler, flags, restorer, mask } on the stack.
    a.alu_imm(5, RSP, 40);
    a.store(RSP, 0, RSI);
    a.alu_imm(1, RDX, (SA_SIGINFO | SA_RESTORER) as i32);
    a.store(RSP, 8, RDX);
    a.lea_label(RAX, restorer);
    a.store(RSP, 16, RAX);
    a.store_imm64(RSP, 24, 0);
    a.mov(RSI, RSP);
    a.zero(RDX);
    a.mov_imm(10, 8); // r10 = sizeof(sigset_t)
    a.mov_imm(RAX, NR_RT_SIGACTION);
    a.syscall();
    a.alu_imm(0, RSP, 40);
    a.ret();
    end(&a, &mut routines);

    // lf_sig_restorer: rt_sigreturn (the kernel reads the frame at rsp).
    begin(&mut a, &mut routines, SYM_SIG_RESTORER, true);
    a.e.bind_label(restorer);
    a.mov_imm(RAX, NR_RT_SIGRETURN);
    a.syscall();
    a.ud2();
    end(&a, &mut routines);

    a.e.bind_label(rt_end);
    let emitted = a.e.finish().expect("runtime labels are all bound and in range");
    let sec = obj.add_emitted_section(".text.lf_rt", SectionKind::Text, 16, emitted);
    for r in routines {
        let binding = if r.global { SymbolBinding::Global } else { SymbolBinding::Local };
        obj.add_symbol(Symbol::defined(r.name, binding, SymbolType::Func, sec, r.start, r.end - r.start));
    }
}

/// The context-switching runtime as a stand-alone object (`lf_rt`).
pub fn context_runtime_object() -> ObjectModule {
    let mut obj = ObjectModule::new("lf_rt");
    emit_context_runtime(&mut obj);
    obj
}
