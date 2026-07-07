//! The RISC-V RV64 general-purpose register file and the LP64 (System V) calling
//! convention (a third CPU target after x86-64 and AArch64).
//!
//! RV64I has 32 general-purpose 64-bit registers `x0`–`x31`. `x0` is hardwired to
//! zero (writes are discarded, reads yield 0). The ABI names and roles are:
//!
//! | reg | abi | role                          | reg | abi | role              |
//! |-----|-----|-------------------------------|-----|-----|-------------------|
//! | x0  | zero| hardwired zero                | x16 | a6  | arg (vol)         |
//! | x1  | ra  | return address (vol)          | x17 | a7  | arg (vol)         |
//! | x2  | sp  | stack pointer (frame)         | x18 | s2  | callee-saved      |
//! | x3  | gp  | global pointer (reserved)     | ... | ... | callee-saved      |
//! | x4  | tp  | thread pointer (reserved)     | x27 | s11 | callee-saved      |
//! | x5  | t0  | our expand scratch (vol)      | x28 | t3  | spill scratch     |
//! | x6  | t1  | our expand scratch (vol)      | x29 | t4  | spill scratch     |
//! | x7  | t2  | our expand scratch (vol)      | x30 | t5  | spill scratch     |
//! | x8  | s0  | callee-saved                  | x31 | t6  | addr scratch      |
//! | x9  | s1  | callee-saved                  |     |     |                   |
//! | x10 | a0  | arg / result (vol)            |     |     |                   |
//! | ... | ..  | args a1..a5 (vol)             |     |     |                   |
//!
//! We number each [`PReg`] with its **hardware register number** (the value that
//! appears in the `rd`/`rs1`/`rs2` fields), so the encoder never translates
//! between an allocator id and a machine number.
//!
//! `sp` (x2), `gp` (x3), `tp` (x4), and `x0` are reserved; they are never handed
//! to the allocator. `t3`/`t4`/`t5` (x28–x30) are reserved as spill/reload
//! **scratch** (three, enough that a single three-operand instruction whose
//! destination and both sources all spill still has a distinct scratch per
//! operand). `t0`/`t1`/`t2` (x5–x7) and `t6` (x31) are reserved for the encoder's
//! multi-instruction idiom expansions (comparison synthesis, `select`, large
//! frame offsets, switch materialization) — they are never allocated, so they are
//! always free to clobber at encode time. Everything else is allocatable: LP64
//! passes the leading integer arguments in `a0`–`a7` (x10–x17), returns in `a0`,
//! and requires `s0`–`s11` to be preserved by the callee.
//!
//! This is implemented from the published RISC-V ISA manual and the RISC-V
//! psABI (LP64) — not from any toolchain's tables.

use crate::codegen::mir::{PReg, RegClass};
use crate::codegen::target::CallConv;

/// The hardwired zero register, `x0`.
pub(crate) const ZERO: u16 = 0;
/// The return-address register, `ra` (x1).
pub(crate) const RA: u16 = 1;
/// The stack pointer, `sp` (x2).
pub(crate) const SP: u16 = 2;
/// The first integer argument/result register, `a0` (x10).
pub(crate) const A0: u16 = 10;
/// Encoder expansion scratch `t0` (x5) — never allocated.
pub(crate) const T0: u16 = 5;
/// Encoder expansion scratch `t1` (x6) — never allocated.
pub(crate) const T1: u16 = 6;
/// Encoder expansion scratch `t2` (x7) — never allocated (switch materialization).
pub(crate) const T2: u16 = 7;
/// Encoder address scratch `t6` (x31) — never allocated (large frame offsets).
pub(crate) const T6: u16 = 31;

/// Construct a GPR [`PReg`] from its hardware number.
#[inline]
pub(crate) fn gpr(n: u16) -> PReg {
    PReg::new(RegClass::Gpr, n)
}

/// The register-file and ABI sets, computed once and borrowed by the target.
///
/// RV64IM models no floating-point file in this phase, so the `Fp` class lists are
/// empty; a float-typed value never reaches this backend in the integer subset.
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
        // Allocatable: caller-saved argument registers first (no save cost), then
        // the callee-saved s-registers. The `t` registers are reserved as scratch
        // (spill/reload) or encoder expansion temporaries and are never allocated.
        let allocatable = [10, 11, 12, 13, 14, 15, 16, 17, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
            .into_iter()
            .map(gpr)
            .collect();
        // Three spill/reload scratch registers: enough for one instruction whose
        // destination and both source operands are all spilled. None is an
        // ABI-fixed operand, and none collides with the encoder expansion temps.
        let scratch = [28u16, 29, 30].into_iter().map(gpr).collect();

        // The LP64 caller-saved (volatile) set a `call` may clobber: ra, all the
        // temporaries t0..t6, and the argument registers a0..a7. Listing the
        // reserved non-allocatable ones (ra, t*) is harmless — it only tells the
        // allocator a call clobbers them, which is true.
        let mut caller_saved: Vec<PReg> = vec![gpr(RA)];
        caller_saved.extend([5u16, 6, 7, 28, 29, 30, 31].into_iter().map(gpr));
        caller_saved.extend((10u16..=17).map(gpr));
        // Callee-saved: s0..s11 (x8, x9, x18..x27).
        let mut callee_saved: Vec<PReg> = vec![gpr(8), gpr(9)];
        callee_saved.extend((18u16..=27).map(gpr));

        let cc = CallConv {
            arg_regs: (10u16..=17).map(gpr).collect(),
            fp_arg_regs: Vec::new(),
            ret_reg: gpr(A0),
            // No floating-point file in the integer subset; a dummy that is never
            // consulted (float-typed values do not reach this backend).
            fp_ret_reg: gpr(A0),
            stack_grows_down: true,
        };
        RegFile {
            classes: vec![RegClass::Gpr, RegClass::Fp],
            allocatable,
            allocatable_fp: Vec::new(),
            scratch,
            scratch_fp: Vec::new(),
            caller_saved,
            callee_saved,
            cc,
        }
    }
}
