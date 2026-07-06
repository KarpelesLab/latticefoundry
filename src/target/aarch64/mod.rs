//! The AArch64 (ARM64) backend (ROADMAP Phase 7): register file + AAPCS64 ABI,
//! the integer instruction-selection rules, the stack-frame/prologue
//! construction, and a from-spec A64 machine-code encoder.
//!
//! This is the framework's **second** CPU target, mirroring the x86-64 backend
//! trait-for-trait to demonstrate that the Phase-5 code-generation framework is
//! genuinely retargetable: the same reusable lowering driver
//! ([`crate::codegen::isel`]) and linear-scan allocator
//! ([`crate::codegen::regalloc`]) drive both, and only the ISA details differ.
//! The backend implements both [`crate::codegen::target::MachineTarget`]
//! (register file, ABI, move/spill builders) and
//! [`crate::codegen::isel::TargetIsel`] (per-opcode lowering).
//!
//! Scope covers the integer subset — arithmetic/bitwise/shift/divide, comparisons
//! and branches, loads/stores and `alloca`, and calls under AAPCS64 — plus scalar
//! floating-point (FP/SIMD): `fadd`/`fsub`/`fmul`/`fdiv`/`fneg`, `fcmp` with the
//! ordered/unordered condition mapping, the `fcvt`/`fcvtz*`/`scvtf`/`ucvtf`
//! conversions, gpr-materialized float constants, and the AAPCS64 float ABI
//! (`v0`–`v7` args, `v0` return). Packed SIMD/vector ops are deferred.
//!
//! Submodules:
//!
//! - [`regs`] — the 31 GPRs (`x0`–`x30`) plus `sp`/`xzr`, the allocatable/scratch
//!   split, and the AAPCS64 calling convention;
//! - [`isel`] — the [`A64Op`] opcode set and the lowering rules;
//! - [`encode`] — the fixed-width bitfield encoder, frame layout +
//!   prologue/epilogue, and the `compile_function`/`compile_module` drivers.

pub mod encode;
pub mod isel;
pub(crate) mod regs;

#[cfg(test)]
mod interp;
#[cfg(test)]
mod tests;

pub use encode::{compile_function, compile_module};
pub use isel::{A64Op, AArch64Target};
