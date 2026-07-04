//! Concrete abstract domains implemented against the one lattice engine.
//!
//! Each domain here is "a lattice + a transfer function + a γ" (tenet **T4**),
//! run by [`crate::analysis::solver`]. The first is constant propagation;
//! integer ranges, known-bits, nullness and points-to follow the same recipe
//! (ROADMAP Phase 3), each validated by [`crate::analysis::soundness`].

pub mod constants;
pub mod known_bits;
pub mod nullness;
pub mod ranges;

pub use constants::ConstLattice;
