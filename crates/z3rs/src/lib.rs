//! # z3rs
//!
//! A clean-room SMT (Satisfiability Modulo Theories) solver written from
//! scratch in pure Rust. It is developed as part of the LatticeFoundry
//! project and used by the framework's verifier and by optimization passes
//! that need to discharge logical side conditions (bounds, overflow, range
//! and refinement checks, and eventually superoptimization).
//!
//! The long-term shape is a DPLL(T) core with theory solvers for fixed-width
//! bit-vectors, linear integer arithmetic, and arrays. This scaffold pins the
//! public surface; the engine is built out in ROADMAP Phase 9.

/// The result of a satisfiability query.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sat {
    /// A satisfying assignment exists.
    Sat,
    /// The formula is unsatisfiable.
    Unsat,
    /// The solver could not decide within its limits.
    Unknown,
}

/// A logical sort (type) of a term.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Sort {
    /// The boolean sort.
    Bool,
    /// A fixed-width bit-vector of the given width.
    BitVec(u32),
}

/// An incremental SMT solving context.
///
/// The assertion stack and decision engine are placeholders; see ROADMAP
/// Phase 9. The API intentionally mirrors an incremental (push/pop) solver so
/// callers written against the scaffold keep working once the engine lands.
#[derive(Debug, Default)]
pub struct Solver {
    assertions: usize,
}

impl Solver {
    /// Create a fresh solver with an empty assertion stack.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an assertion (placeholder: terms are not yet modeled).
    pub fn assert(&mut self) {
        self.assertions += 1;
    }

    /// Check satisfiability of the current assertion stack.
    ///
    /// The trivial base case — no assertions — is genuinely satisfiable; any
    /// non-empty problem is reported as [`Sat::Unknown`] until the engine is
    /// implemented.
    pub fn check(&self) -> Sat {
        if self.assertions == 0 { Sat::Sat } else { Sat::Unknown }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_context_is_trivially_sat() {
        let mut s = Solver::new();
        assert_eq!(s.check(), Sat::Sat);
        s.assert();
        assert_eq!(s.check(), Sat::Unknown);
    }
}
