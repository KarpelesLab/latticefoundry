//! The AArch64 general-purpose register file and the AAPCS64 ABI description
//! (ROADMAP Phase 7).
//!
//! There are 31 general-purpose 64-bit registers `x0`–`x30` (their 32-bit views
//! are `w0`–`w30`), plus the stack pointer `sp` and the zero register `xzr`.
//! `sp` and `xzr` share the register-field encoding `31`; which one an encoding
//! means depends on the instruction (base register ⇒ `sp`, data operand ⇒
//! `xzr`), so the encoder decides per opcode. We number each [`PReg`] with its
//! **hardware register number** — the value that appears in the `Rd`/`Rn`/`Rm`
//! fields — so the encoder never translates between an allocator id and a
//! machine number.
//!
//! | num | reg | role                 | num | reg | role            |
//! |-----|-----|----------------------|-----|-----|-----------------|
//! | 0–7 | x0–x7  | arg / result (vol) | 18  | x18 | platform (vol)  |
//! | 8   | x8  | indirect result (vol) | 19–28 | x19–x28 | callee-saved |
//! | 9–11| x9–x11 | our spill scratch  | 29  | x29 | frame pointer   |
//! | 12–15| x12–x15 | temporaries (vol) | 30  | x30 | link register   |
//! | 16–17| x16–x17 | IP0/IP1 (vol)     | 31  | sp/xzr | stack / zero  |
//!
//! `x29` (FP), `x30` (LR), and `sp` are reserved for the frame. `x9`, `x10`,
//! and `x11` are reserved as spill/reload **scratch** (never handed to the
//! allocator): a single three-operand instruction whose destination and both
//! sources all spill needs three free registers, and these three collide with
//! no fixed ABI operand. The remaining registers are allocatable. AAPCS64 passes
//! the leading integer arguments in `x0`–`x7`, returns in `x0`, and requires
//! `x19`–`x28` to be preserved by the callee.
//!
//! This is implemented from the published AAPCS64 procedure-call standard and
//! the ARM A64 register model (tenet T1), not from any toolchain's tables.

use crate::codegen::mir::{PReg, RegClass};
use crate::codegen::target::CallConv;

/// Hardware register numbers, named for readability in the isel/encoder.
pub(crate) const X0: u16 = 0;
pub(crate) const X9: u16 = 9;
pub(crate) const X10: u16 = 10;
pub(crate) const X11: u16 = 11;
/// The frame pointer, `x29`.
pub(crate) const FP: u16 = 29;
/// The link register, `x30`.
pub(crate) const LR: u16 = 30;
/// The `sp`/`xzr` register-field encoding (`31`); meaning is instruction-defined.
pub(crate) const SP: u16 = 31;
/// The zero register, sharing encoding `31` with `sp`.
pub(crate) const XZR: u16 = 31;

/// Construct a GPR [`PReg`] from its hardware number.
#[inline]
pub(crate) fn gpr(n: u16) -> PReg {
    PReg::new(RegClass::Gpr, n)
}

/// Construct an FP/SIMD [`PReg`] from its hardware number. `n` is the hardware
/// `v`-register number 0..=31 (the value the encoder puts into the `Rd`/`Rn`/`Rm`
/// fields of an FP instruction). Scalar `d`/`s` views share the same number.
#[inline]
pub(crate) fn fp(n: u16) -> PReg {
    PReg::new(RegClass::Fp, n)
}

/// The register-file and ABI sets, computed once and borrowed by the target.
///
/// The FP/SIMD file is `v0..v31` (scalar `d0..d31` / `s0..s31`). Under AAPCS64
/// float/double arguments go in `v0..v7` (a counter separate from the integer
/// `x0..x7`), a float/double return is in `v0`, `v8..v15` are callee-saved (only
/// their low 64 bits, but treated as fully callee-saved here for simplicity), and
/// `v16..v31` are caller-saved. `v29..v31` are reserved as spill/reload scratch —
/// three of them, so an FP three-operand instruction whose destination and both
/// sources all spill still has a distinct scratch per operand, mirroring the GPR
/// `x9/x10/x11` reservation.
#[derive(Debug)]
pub(crate) struct RegFile {
    pub(crate) classes: Vec<RegClass>,
    pub(crate) allocatable: Vec<PReg>,
    pub(crate) allocatable_fp: Vec<PReg>,
    pub(crate) scratch: Vec<PReg>,
    pub(crate) scratch_fp: Vec<PReg>,
    pub(crate) caller_saved: Vec<PReg>,
    pub(crate) callee_saved: Vec<PReg>,
    pub(crate) cc: CallConv,
}

impl RegFile {
    pub(crate) fn new() -> RegFile {
        // Allocatable: caller-saved first (preferred, no save cost), then the
        // callee-saved x19..x28. x9/x10/x11 are scratch; x29/x30/sp are the
        // frame; x16/x17/x18 are reserved (IP0/IP1/platform).
        let allocatable = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
            .into_iter()
            .map(gpr)
            .collect();
        // Three scratch registers: enough for one instruction whose dest and both
        // source operands are all spilled. None is an ABI-fixed operand.
        let scratch = [X9, X10, X11].into_iter().map(gpr).collect();

        // FP/SIMD file: v0..v7 (arg/caller-saved) and v16..v28 (caller-saved)
        // preferred first, then the callee-saved v8..v15; v29..v31 are scratch.
        let allocatable_fp =
            (0u16..=7).chain(16u16..=28).chain(8u16..=15).map(fp).collect();
        let scratch_fp = [29u16, 30, 31].into_iter().map(fp).collect();

        // The AAPCS64 caller-saved (volatile) set: x0..x18 plus the volatile FP
        // registers v0..v7 and v16..v31. Listing the reserved non-allocatable
        // ones is harmless — it only tells the allocator a call may clobber them,
        // which is true.
        let mut caller_saved: Vec<PReg> = (0u16..=18).map(gpr).collect();
        caller_saved.extend((0u16..=7).chain(16u16..=31).map(fp));
        let mut callee_saved: Vec<PReg> = (19u16..=28).map(gpr).collect();
        callee_saved.extend((8u16..=15).map(fp));

        let cc = CallConv {
            arg_regs: (0u16..=7).map(gpr).collect(),
            fp_arg_regs: (0u16..=7).map(fp).collect(),
            ret_reg: gpr(X0),
            fp_ret_reg: fp(0),
            stack_grows_down: true,
        };
        RegFile {
            classes: vec![RegClass::Gpr, RegClass::Fp],
            allocatable,
            allocatable_fp,
            scratch,
            scratch_fp,
            caller_saved,
            callee_saved,
            cc,
        }
    }
}
