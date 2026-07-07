//! The RISC-V RV64IM backend: register file + LP64 ABI, the integer
//! instruction-selection rules, the stack-frame/prologue construction, and a
//! from-spec RV64 machine-code encoder.
//!
//! This is the framework's **third** CPU target, after x86-64 and AArch64,
//! demonstrating that the code-generation framework is genuinely retargetable:
//! the same reusable lowering driver ([`crate::codegen::isel`]) and linear-scan
//! allocator ([`crate::codegen::regalloc`]) drive all three, and only the ISA
//! details differ. The backend implements both
//! [`crate::codegen::target::MachineTarget`] (register file, ABI, move/spill
//! builders) and [`crate::codegen::isel::TargetIsel`] (per-opcode lowering).
//!
//! Scope covers the RV64IM integer subset — arithmetic/bitwise/shift/multiply/
//! divide/remainder, comparisons and branches, loads/stores and `alloca`, and
//! calls under LP64. Deferred (and noted at their sites): scalar floating-point
//! (F/D), the compressed (C) and atomic (A) extensions, the RV32-word forms for
//! sub-64-bit widths, `> 8` stack-passed arguments, and the `R_RISCV_CALL` /
//! `R_RISCV_PCREL_*` relocations (which need machine-code-layer `RelocKind`s this
//! backend does not add).
//!
//! Submodules:
//!
//! - [`regs`] — the 32 GPRs (`x0`–`x31`, `x0` hardwired zero), the
//!   allocatable/scratch split, and the LP64 calling convention;
//! - [`isel`] — the [`RvOp`] opcode set and the lowering rules;
//! - [`encode`] — the fixed-width bitfield encoder (R/I/S/B/U/J formats), frame
//!   layout + prologue/epilogue, and the `compile_function`/`compile_module`
//!   drivers.

pub mod encode;
pub mod isel;
pub(crate) mod regs;

#[cfg(test)]
mod interp;
#[cfg(test)]
mod tests;

pub use encode::{compile_function, compile_module};
pub use isel::{RiscvTarget, RvOp};
