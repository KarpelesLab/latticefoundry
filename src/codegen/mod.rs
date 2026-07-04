//! Target-independent code generation.
//!
//! Lowers the SSA IR to a machine-level IR (MIR), then runs instruction
//! selection, register allocation, and scheduling to produce target machine
//! instructions ready for the [`mc`](crate::mc) layer. See ROADMAP Phase 5.

/// Entry point for lowering IR to machine code (placeholder).
#[derive(Debug, Default)]
pub struct CodeGen;
