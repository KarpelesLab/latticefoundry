//! The IR verifier.
//!
//! Before a [`Module`] is handed to later stages it is checked for structural
//! and type invariants (every block ends in exactly one terminator, operands
//! are dominated by their definitions, types agree, ...). See ROADMAP Phase 2.

use crate::ir::Module;

/// A single verifier diagnostic.
#[derive(Debug)]
pub struct VerifyError {
    /// Human-readable description of the broken invariant.
    pub message: String,
}

/// Verify the structural invariants of a module.
///
/// Currently a placeholder that accepts every module; the invariant checks
/// land alongside the IR builder in Phase 2.
pub fn verify_module(_module: &Module) -> Result<(), Vec<VerifyError>> {
    Ok(())
}

/// Bridge to the external [`z3rs`] SMT solver, used to discharge verification
/// conditions (bounds, overflow, and refinement checks). `z3rs` is a separate
/// clean-room crate; it is re-exported here so verifier code has a single
/// canonical path to it. See ROADMAP Phase 9.
pub mod smt {
    #[doc(inline)]
    pub use z3rs;
}
