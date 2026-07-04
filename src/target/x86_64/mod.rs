//! The x86-64 backend (ROADMAP Phase 7): register file + System V ABI, the
//! integer instruction-selection rules, the stack-frame/prologue construction,
//! and a from-spec machine-code encoder that produces relocatable ELF64 objects.
//!
//! The backend plugs into the Phase-5 code-generation framework — it implements
//! both [`crate::codegen::target::MachineTarget`] (register file, ABI,
//! move/spill builders) and [`crate::codegen::isel::TargetIsel`] (per-opcode
//! lowering) — and emits into the Phase-6 machine-code layer
//! ([`crate::mc::emit`] / [`crate::mc::object`] / [`crate::mc::elf`]).
//!
//! Scope is the integer subset sufficient to compile and *run* real functions:
//! arithmetic/bitwise/shift/divide, comparisons and branches, loads/stores and
//! `alloca`, and calls under the SysV ABI. Floating-point and SIMD are deferred.
//!
//! Submodules:
//!
//! - [`regs`] — the 16 GPRs as physical registers, the allocatable/scratch split,
//!   and the SysV calling convention;
//! - [`isel`] — the [`X86Op`] opcode set and the lowering rules;
//! - [`encode`] — the REX/ModRM/SIB encoder, frame layout + prologue/epilogue,
//!   and the `compile_function`/`compile_module` drivers.

pub mod encode;
pub mod isel;
pub(crate) mod regs;

#[cfg(test)]
mod tests;

pub use encode::{compile_function, compile_module, compile_to_elf};
pub use isel::{X86Op, X86_64Target};
