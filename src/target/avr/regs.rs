//! The AVR register file as the allocator sees it, and the avr-gcc calling
//! convention.
//!
//! AVR has 32 8-bit registers `r0`–`r31`. The allocator works on **register
//! pairs**: a [`PReg`] numbered `n` (always even) is the pair `r(n+1):r(n)`,
//! low byte in `r(n)`. An `i8` (or narrower) value lives in the low register of
//! its pair and the high register is don't-care; an `i16` or a pointer fills
//! the pair; wider integers are split into 16-bit parts by
//! [`legalize_ints`](crate::codegen::legalize_int::legalize_ints) and live in
//! several pairs. Pairing everything keeps one vreg = one register (what the
//! linear-scan allocator models), makes `movw` the universal move, and matches
//! the ABI, which aligns every argument to an even register.
//!
//! | pair | role |
//! |---|---|
//! | `r1:r0` | reserved: `r0` is the temporary (`mul` result, `SREG` save), `r1` the zero register (always 0 between instructions) |
//! | `r3:r2`, `r5:r4` | spill/reload scratch (callee-saved, saved by the prologue when used) |
//! | `r7:r6` … `r17:r16` | allocatable, callee-saved |
//! | `r19:r18` … `r25:r24` | allocatable, caller-saved; arguments and results |
//! | `r27:r26` (X) | spill/reload scratch (caller-saved) |
//! | `r29:r28` (Y) | the frame pointer (callee-saved) |
//! | `r31:r30` (Z) | the encoder's temporary: addresses for `ld`/`st`/`lpm`, `icall` targets, immediates for low registers |
//!
//! The spill scratch pairs are neither argument nor return registers, so a
//! reload in the middle of an argument sequence never clobbers one.
//!
//! # Calling convention (avr-gcc)
//!
//! Implemented from the published avr-gcc ABI description:
//!
//! - Arguments are assigned left to right starting from `r26`: each argument's
//!   size is rounded up to an even number of bytes and subtracted from the
//!   running register number; if the result is at least `r8`, the argument is
//!   passed there (least significant byte in the lowest register), otherwise it
//!   and **every following argument** go on the stack.
//! - Stack arguments are laid out in order from the stack pointer upward: the
//!   first one starts at `SP + 1` at the moment of the call, each at its exact
//!   size (no padding). The caller pushes them and pops them after the call.
//! - A return value of `s` bytes (at most 8) is returned in the registers from
//!   `r(26 - round_up_even(s))` upward: `r24` for 1 byte, `r25:r24` for 2,
//!   `r25..r22` for 4, `r25..r18` for 8.
//! - `r18`–`r27`, `r30`, `r31` and `r0` are call-clobbered; `r2`–`r17`, `r28`,
//!   `r29` are call-saved; `r1` must be zero on entry and on return.
//! - An `i1` (`_Bool`) argument or return value is passed as 0 or 1. Other
//!   narrow values leave the bits above their width unspecified (the IR does
//!   not say whether a C `char` was signed).

use crate::codegen::mir::{PReg, RegClass};
use crate::codegen::target::CallConv;

/// The temporary register `r0`.
pub(crate) const TMP: u8 = 0;
/// The zero register `r1`.
pub(crate) const ZERO: u8 = 1;
/// The Y pointer pair `r29:r28` (frame pointer).
pub(crate) const Y: u8 = 28;
/// The Z pointer pair `r31:r30` (encoder temporary).
pub(crate) const Z: u8 = 30;

/// The lowest register an argument may be passed in.
pub(crate) const ARG_FLOOR: u8 = 8;
/// The register the argument assignment counts down from.
pub(crate) const ARG_TOP: u8 = 26;

/// The pair [`PReg`] whose low register is `r(n)` (`n` even).
#[inline]
pub(crate) fn pair(n: u8) -> PReg {
    debug_assert!(n.is_multiple_of(2) && n < 32, "pair r{n}");
    PReg::new(RegClass::Gpr, u16::from(n))
}

/// The call-clobbered pairs (besides `r1:r0`, which is never allocated).
pub(crate) const CALLER_SAVED: [u8; 6] = [18, 20, 22, 24, 26, 30];
/// The call-saved pairs.
pub(crate) const CALLEE_SAVED: [u8; 9] = [2, 4, 6, 8, 10, 12, 14, 16, 28];

/// Where one argument travels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ArgLoc {
    /// In registers, least significant byte in `r(n)` (`n` even).
    Regs(u8),
    /// On the stack, `off` bytes above the first stack argument.
    Stack(u64),
}

/// Round a byte size up to an even number of bytes.
#[inline]
pub(crate) fn even(bytes: u64) -> u64 {
    bytes.div_ceil(2) * 2
}

/// The avr-gcc argument assignment for arguments of the given byte sizes.
/// Returns each argument's location and the total size of the stack arguments.
pub(crate) fn assign_args(sizes: &[u64]) -> (Vec<ArgLoc>, u64) {
    let mut out = Vec::with_capacity(sizes.len());
    let mut next = i64::from(ARG_TOP);
    let mut on_stack = false;
    let mut stack = 0u64;
    for &s in sizes {
        if !on_stack {
            let n = next - even(s) as i64;
            if n >= i64::from(ARG_FLOOR) && s > 0 {
                next = n;
                out.push(ArgLoc::Regs(n as u8));
                continue;
            }
            on_stack = true;
        }
        out.push(ArgLoc::Stack(stack));
        stack += s;
    }
    (out, stack)
}

/// The first (lowest) register of a return value of `bytes` bytes (1..=8).
#[inline]
pub(crate) fn ret_base(bytes: u64) -> u8 {
    ARG_TOP - even(bytes.max(1)) as u8
}

/// The register-file and ABI tables, computed once and borrowed by the target.
#[derive(Debug)]
pub(crate) struct RegFile {
    pub(crate) classes: Vec<RegClass>,
    pub(crate) allocatable: Vec<PReg>,
    pub(crate) scratch: Vec<PReg>,
    pub(crate) empty: Vec<PReg>,
    pub(crate) caller_saved: Vec<PReg>,
    pub(crate) callee_saved: Vec<PReg>,
    pub(crate) cc: CallConv,
}

impl RegFile {
    pub(crate) fn new() -> RegFile {
        // Caller-saved pairs first (no save cost), then the callee-saved ones.
        let allocatable = [18u8, 20, 22, 24, 16, 14, 12, 10, 8, 6].into_iter().map(pair).collect();
        // Three scratch pairs: one per register operand of the widest MIR
        // instruction (a destination and two sources).
        let scratch = [26u8, 2, 4].into_iter().map(pair).collect();
        let cc = CallConv {
            // The register pairs of the leading 2-byte arguments; the real
            // assignment depends on every argument's size (`assign_args`).
            arg_regs: [24u8, 22, 20, 18, 16, 14, 12, 10, 8].into_iter().map(pair).collect(),
            fp_arg_regs: Vec::new(),
            ret_reg: pair(24),
            fp_ret_reg: pair(24),
            stack_grows_down: true,
        };
        RegFile {
            classes: vec![RegClass::Gpr],
            allocatable,
            scratch,
            empty: Vec::new(),
            caller_saved: CALLER_SAVED.into_iter().map(pair).collect(),
            callee_saved: CALLEE_SAVED.into_iter().map(pair).collect(),
            cc,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argument_assignment_follows_the_avr_gcc_rules() {
        // char, int, long: r24, r22, r18..r21.
        assert_eq!(assign_args(&[1, 2, 4]).0, vec![ArgLoc::Regs(24), ArgLoc::Regs(22), ArgLoc::Regs(18)]);
        // Two long longs: r18..r25, r10..r17; a third goes on the stack, and
        // so does everything after it (even a char that would still fit).
        let (locs, stack) = assign_args(&[8, 8, 8, 1, 2]);
        assert_eq!(
            locs,
            vec![ArgLoc::Regs(18), ArgLoc::Regs(10), ArgLoc::Stack(0), ArgLoc::Stack(8), ArgLoc::Stack(9)]
        );
        assert_eq!(stack, 11);
        // Nine ints fill r24 down to r8; the tenth is on the stack.
        let (locs, _) = assign_args(&[2; 10]);
        assert_eq!(locs[8], ArgLoc::Regs(8));
        assert_eq!(locs[9], ArgLoc::Stack(0));
        assert_eq!((ret_base(1), ret_base(2), ret_base(4), ret_base(8)), (24, 24, 22, 18));
    }
}
