//! The ARMv7-M core register file and the AAPCS (base standard) calling
//! convention.
//!
//! Thumb code sees sixteen 32-bit core registers. Their AAPCS roles, and what
//! this backend does with each:
//!
//! | reg | AAPCS role | here |
//! |---|---|---|
//! | `r0`–`r3` | arguments / results, caller-saved | allocatable |
//! | `r4`–`r8` | callee-saved (`v1`–`v5`) | allocatable |
//! | `r9`–`r11` | callee-saved (`v6`, `v7`, `v8`; `r9` is the platform register) | spill/reload scratch |
//! | `r12` (`ip`) | intra-procedure scratch, caller-saved | the encoder's expansion temporary; fourth spill scratch |
//! | `r13` (`sp`) | stack pointer | reserved |
//! | `r14` (`lr`) | link register | allocatable (saved in every prologue; a `bl` clobbers it) |
//! | `r15` (`pc`) | program counter | reserved |
//!
//! A [`PReg`] is numbered with its **hardware register number** (the value in
//! the `Rd`/`Rn`/`Rm` fields), so the encoder never translates.
//!
//! **Scratch.** The register allocator reloads spilled operands into the
//! scratch registers, one per spilled operand of an instruction. Every MIR op
//! of this backend has at most three register operands except `select`
//! (four); the first three scratch registers are `r9`–`r11` (callee-saved, so a
//! function that spills saves them in its prologue like any other callee-saved
//! register it defines) and the fourth is `r12`, which only a four-operand op
//! can receive and whose encoding needs no temporary. Every other op may use
//! `r12` freely inside its encoding (large immediates, frame offsets beyond the
//! immediate forms, the remainder's quotient, stack probes).
//!
//! **`lr`.** Every prologue pushes `lr` and every epilogue pops the return
//! address straight into `pc`, so between them `lr` is an ordinary register;
//! it is caller-saved because a `bl` overwrites it.
//!
//! Implemented from the *Procedure Call Standard for the Arm Architecture*
//! (AAPCS32) and the ARMv7-M Architecture Reference Manual, not from any
//! toolchain's tables.

use crate::codegen::mir::{PReg, RegClass};
use crate::codegen::target::CallConv;

/// The first argument / result register, `r0`.
pub(crate) const R0: u16 = 0;
/// The intra-procedure scratch register `ip` (`r12`): the encoder's temporary.
pub(crate) const IP: u16 = 12;
/// The stack pointer.
pub(crate) const SP: u16 = 13;
/// The link register.
pub(crate) const LR: u16 = 14;
/// The program counter.
pub(crate) const PC: u16 = 15;

/// Construct a core-register [`PReg`] from its hardware number.
#[inline]
pub(crate) fn gpr(n: u16) -> PReg {
    PReg::new(RegClass::Gpr, n)
}

/// The register-file and ABI sets, computed once and borrowed by the target.
///
/// There is no floating-point register file: floating-point values are
/// integers by the time they reach instruction selection (the soft-float
/// lowering, [`super::softfloat`]), so the `Fp` lists are empty.
#[derive(Debug)]
pub(crate) struct RegFile {
    pub(crate) classes: Vec<RegClass>,
    pub(crate) allocatable: Vec<PReg>,
    pub(crate) scratch: Vec<PReg>,
    pub(crate) caller_saved: Vec<PReg>,
    pub(crate) callee_saved: Vec<PReg>,
    pub(crate) cc: CallConv,
    pub(crate) none: Vec<PReg>,
}

impl RegFile {
    pub(crate) fn new() -> RegFile {
        // Caller-saved first (no save cost in a leaf), low registers before
        // high ones (the 16-bit encodings reach r0-r7 only), `lr` last among
        // the caller-saved.
        let allocatable = [0u16, 1, 2, 3, 4, 5, 6, 7, 8, LR].into_iter().map(gpr).collect();
        let scratch = [9u16, 10, 11, IP].into_iter().map(gpr).collect();
        // A call may clobber r0-r3, ip and lr.
        let caller_saved = [0u16, 1, 2, 3, IP, LR].into_iter().map(gpr).collect();
        let callee_saved = (4u16..=11).map(gpr).collect();
        let cc = CallConv {
            arg_regs: (0u16..=3).map(gpr).collect(),
            fp_arg_regs: Vec::new(),
            ret_reg: gpr(R0),
            // The soft-float ABI returns floating-point values in r0 (r0:r1).
            fp_ret_reg: gpr(R0),
            stack_grows_down: true,
        };
        RegFile {
            classes: vec![RegClass::Gpr],
            allocatable,
            scratch,
            caller_saved,
            callee_saved,
            cc,
            none: Vec::new(),
        }
    }
}
