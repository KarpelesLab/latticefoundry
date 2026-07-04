//! The IR verifier.
//!
//! Before a [`Module`] is handed to later stages it is checked for structural
//! and semantic well-formedness. This module implements the **`Structural`**
//! verification tier of `docs/design-tenets.md` §2: the cheap, solver-free
//! invariants that every subsequent stage — and the `Refinement`/`z3rs` tier
//! layered on later — is entitled to assume.
//!
//! What is checked (all violations are reported, never just the first):
//!
//! - every block ends in exactly one terminator, with terminators only in the
//!   terminator slot;
//! - control-flow integrity: successor blocks exist, and nothing branches into
//!   the entry block (whose parameters are the function parameters);
//! - block-argument arity and typing on every edge (our replacement for
//!   φ-node operand checks);
//! - SSA dominance: every operand's definition dominates its use, via a
//!   Cooper–Harvey–Kennedy dominator tree ([`cfg`]);
//! - type agreement per opcode: `call` arity/signature, `cond_br` on `i1`,
//!   `switch` on an integer, `select` arms, cast compatibility, `load`/`store`
//!   sanity, and result/operand consistency;
//! - well-typed constants and existence of referenced functions and globals.
//!
//! The entry points are [`verify_module`] (whole module) and [`verify_function`]
//! (one function). [`structural_verify`] is the driver-facing pass: it runs the
//! same checks but returns a [`Diagnostics`] sink for callers that thread one
//! through their phase.
//!
//! The **`Refinement` tier** (per-opcode poison/UB refinement obligations
//! discharged by `z3rs`) lives in [`refinement`]: it encodes a single-block,
//! pure-integer rewrite `src ⇒ tgt` into QF_BV and proves `tgt ⊑ src`. It builds
//! on the invariants established here, and the `z3rs` re-export in [`smt`] is the
//! seam it plugs into. This module only *reads* the IR — it never mutates a
//! module.

mod cfg;
pub mod refinement;
mod structural;

#[cfg(test)]
mod tests;

use crate::ir::{FuncId, Module};
use crate::support::diagnostics::{Diagnostic, Diagnostics};

pub use refinement::{RefinementResult, RefinementTier, check_refinement};
pub use structural::verify_function;

/// Verify the structural + semantic invariants of every function in `module`.
///
/// Returns `Ok(())` if the module is well-formed at the `Structural` tier, or
/// `Err` with every error [`Diagnostic`] found across all functions.
pub fn verify_module(module: &Module) -> Result<(), Vec<Diagnostic>> {
    let mut diags = Vec::new();
    for i in 0..module.functions().count() {
        diags.extend(verify_function(module, FuncId::from_index(i)));
    }
    if diags.iter().any(Diagnostic::is_error) { Err(diags) } else { Ok(()) }
}

/// The driver-facing structural verify pass: run the verifier over `module` and
/// collect its diagnostics into a [`Diagnostics`] sink (whose [`has_errors`]
/// reports whether the module is well-formed).
///
/// [`has_errors`]: Diagnostics::has_errors
pub fn structural_verify(module: &Module) -> Diagnostics {
    let mut sink = Diagnostics::new();
    for i in 0..module.functions().count() {
        for d in verify_function(module, FuncId::from_index(i)) {
            sink.emit(d);
        }
    }
    sink
}

/// Bridge to the external [`z3rs`] SMT solver, used to discharge verification
/// conditions (bounds, overflow, and refinement checks). `z3rs` is a separate
/// clean-room crate; it is re-exported here so verifier code has a single
/// canonical path to it. See ROADMAP Phase 9.
pub mod smt {
    #[doc(inline)]
    pub use z3rs;
}
