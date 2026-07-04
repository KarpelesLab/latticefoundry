//! Target-independent code generation.
//!
//! Lowers the SSA [`ir`](crate::ir) to a machine-level IR (MIR), then runs
//! instruction selection and register allocation to produce target machine
//! instructions ready for the [`mc`](crate::mc) layer (ROADMAP Phase 5). The
//! layer is split into:
//!
//! - [`mir`] — the target-abstract MIR data model (registers, operands, blocks,
//!   functions, stack frame);
//! - [`target`] — the [`MachineTarget`](target::MachineTarget) interface a
//!   backend implements to describe its register file, calling convention, and
//!   move/spill builders, *without* committing to encodings;
//! - [`isel`] — the reusable instruction-selection framework (block-argument
//!   lowering, the ABI seam, constant materialization) parameterized by a
//!   target's per-opcode rules ([`isel::TargetIsel`]);
//! - [`vtarget`] — an abstract RISC-like virtual target that implements both
//!   traits, so the framework can be exercised end to end without a real ISA;
//! - [`regalloc`] — a correct linear-scan register allocator with spilling;
//! - [`interp`] — a small MIR interpreter over the virtual target, the
//!   executable semantics isel + regalloc are validated against.
//!
//! Real ISAs (x86-64, AArch64, RISC-V) and instruction *encoding* are Phases
//! 6–7; this phase produces MIR, not bytes.

pub mod interp;
pub mod isel;
pub mod mir;
pub mod regalloc;
pub mod target;
pub mod vtarget;

pub use mir::MachineFunction;
pub use target::MachineTarget;
pub use vtarget::VirtualTarget;

#[cfg(test)]
mod tests;
