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
//! Scope covers the integer subset sufficient to compile and *run* real
//! functions — arithmetic/bitwise/shift/divide, comparisons and branches,
//! loads/stores and `alloca`, and calls under the SysV ABI — plus **scalar SSE
//! floating-point** (F32/F64): `addsd`/…/`divsd` and the `ss` forms, `fneg` via
//! a sign-bit `xorpd`, `fcmp` via `ucomis`+`setcc` (with the ordered/unordered
//! parity fixup), the `cvt*` conversions, `movsd`/`movss` loads/stores/spills,
//! and float argument/return passing in `xmm0..xmm7`/`xmm0`. F16 is widened or
//! deferred (it is not an x86 scalar type); wider SIMD and AArch64 FP are
//! follow-ups.
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

pub use encode::{
    DebugSource, compile_function, compile_module, compile_module_debug, compile_to_elf,
};
pub use isel::{X86Op, X86_64Target};
