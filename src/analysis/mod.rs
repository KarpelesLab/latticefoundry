//! The analysis layer **is** one sound abstract-interpretation engine, not a
//! drawer of bespoke analyses (tenet **T4**, bet **B8**; ROADMAP Phase 3).
//!
//! Everything here hangs off two ideas:
//!
//! - [`AbstractDomain`] — a lattice with a concretization γ, transfer functions,
//!   and a control-flow model. "A new analysis" is a new domain, never a new
//!   dataflow pass.
//! - [`solver::solve`] — a single generic, sparse, SSA-based monotone fixpoint
//!   engine, parameterized by any domain, with widening at loop headers and
//!   sparse conditional constant propagation over block-argument edges.
//!
//! Around them:
//!
//! - [`domains`] holds the concrete domains ([`ConstLattice`], the first);
//! - [`soundness`] is the harness that checks a domain's transfer is sound
//!   against the reference semantics (the teeth of bet B8);
//! - [`manager`] runs analyses, caches results, and invalidates them when a pass
//!   mutates the IR — the seam into [`crate::pass`].
//!
//! Determinism (tenet T5) is maintained throughout: dense-id-indexed `Vec`s and
//! [`DetHashMap`](crate::support::DetHashMap), never `std` `HashMap` iteration.

pub mod cfg;
pub mod domain;
pub mod domfrontier;
pub mod domains;
pub mod manager;
pub mod soundness;
pub mod solver;

pub use domain::{AbstractDomain, DomainCtx, EdgeGuard};
pub use domfrontier::DominanceFrontiers;
pub use domains::ConstLattice;
pub use manager::{AnalysisCache, ConstantPropagation, FunctionAnalysis};
pub use solver::{FixpointResult, solve};

#[cfg(test)]
mod tests;
