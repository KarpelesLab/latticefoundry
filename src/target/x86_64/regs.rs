//! The x86-64 general-purpose register file and the System V AMD64 ABI
//! description (ROADMAP Phase 7).
//!
//! There are sixteen 64-bit integer registers. We number each [`PReg`] with its
//! **hardware encoding number** — the value that appears in the `ModRM.reg` /
//! `ModRM.rm` fields and (extended) in the `REX` prefix — so the encoder never
//! has to translate between an allocator id and a machine number:
//!
//! | num | reg | num | reg |
//! |-----|-----|-----|-----|
//! | 0   | rax | 8   | r8  |
//! | 1   | rcx | 9   | r9  |
//! | 2   | rdx | 10  | r10 |
//! | 3   | rbx | 11  | r11 |
//! | 4   | rsp | 12  | r12 |
//! | 5   | rbp | 13  | r13 |
//! | 6   | rsi | 14  | r14 |
//! | 7   | rdi | 15  | r15 |
//!
//! `rsp`/`rbp` are reserved for the frame. `r10`, `r11`, and `rbx` are reserved
//! as spill/reload **scratch** (never handed to the allocator): a single
//! three-operand instruction whose destination and both sources all spill needs
//! three free registers, and these three collide with no fixed ABI operand. The
//! remaining eleven registers are allocatable. The System V ABI passes the
//! leading integer arguments in `rdi, rsi, rdx, rcx, r8, r9`, returns in `rax`,
//! and requires `rbx, rbp, rsp, r12..r15` to be preserved by the callee.
//!
//! This is implemented from the published SysV AMD64 psABI (tenet T1), not from
//! any toolchain's tables.

use crate::codegen::mir::{PReg, RegClass};
use crate::codegen::target::CallConv;

/// Hardware register numbers, named for readability in the isel/encoder.
pub(crate) const RAX: u16 = 0;
pub(crate) const RCX: u16 = 1;
pub(crate) const RDX: u16 = 2;
pub(crate) const RBX: u16 = 3;
pub(crate) const RSP: u16 = 4;
pub(crate) const RBP: u16 = 5;
pub(crate) const RSI: u16 = 6;
pub(crate) const RDI: u16 = 7;
pub(crate) const R8: u16 = 8;
pub(crate) const R9: u16 = 9;
pub(crate) const R10: u16 = 10;
pub(crate) const R11: u16 = 11;
pub(crate) const R12: u16 = 12;
pub(crate) const R13: u16 = 13;
pub(crate) const R14: u16 = 14;
pub(crate) const R15: u16 = 15;

/// Construct a GPR [`PReg`] from its hardware number.
#[inline]
pub(crate) fn gpr(n: u16) -> PReg {
    PReg::new(RegClass::Gpr, n)
}

/// Construct an XMM (SSE floating-point) [`PReg`] from its hardware number. The
/// `num` is the hardware xmm number 0..=15 (the value the encoder puts into
/// `ModRM.reg`/`ModRM.rm` and, extended, into `REX.R`/`REX.B`).
#[inline]
pub(crate) fn xmm(n: u16) -> PReg {
    PReg::new(RegClass::Fp, n)
}

/// The register-file and ABI sets, computed once and borrowed by the target.
///
/// The floating-point file is `xmm0..xmm15`. On System V *every* xmm is
/// caller-saved (a `call` may clobber all of them). `xmm13..xmm15` are reserved
/// as spill/reload scratch — three of them, so an SSE three-operand instruction
/// whose destination and both sources all spill still has a distinct scratch per
/// operand, mirroring the GPR `r10/r11/rbx` reservation. `xmm0..xmm12` are
/// allocatable. Float/double arguments go in `xmm0..xmm7` (a counter separate
/// from the integer `rdi..r9`), and a float/double return is in `xmm0`.
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
        // callee-saved r12..r15. rbx/r10/r11 are scratch; rsp/rbp are the frame.
        let allocatable = [RAX, RCX, RDX, RSI, RDI, R8, R9, R12, R13, R14, R15]
            .into_iter()
            .map(gpr)
            .collect();
        // Three scratch registers: enough for one instruction whose dest and both
        // source operands are all spilled. None is an ABI-fixed operand.
        let scratch = [R10, R11, RBX].into_iter().map(gpr).collect();

        // Floating-point file: xmm0..xmm12 allocatable, xmm13..xmm15 scratch.
        let allocatable_fp = (0u16..=12).map(xmm).collect();
        let scratch_fp = [13u16, 14, 15].into_iter().map(xmm).collect();

        // Caller-saved: the volatile GPRs plus *all* xmm registers (SysV).
        let mut caller_saved: Vec<PReg> =
            [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11].into_iter().map(gpr).collect();
        caller_saved.extend((0u16..=15).map(xmm));

        let callee_saved = [RBX, R12, R13, R14, R15].into_iter().map(gpr).collect();
        let cc = CallConv {
            arg_regs: [RDI, RSI, RDX, RCX, R8, R9].into_iter().map(gpr).collect(),
            fp_arg_regs: (0u16..=7).map(xmm).collect(),
            ret_reg: gpr(RAX),
            fp_ret_reg: xmm(0),
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
