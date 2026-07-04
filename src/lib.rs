//! # LatticeFoundry
//!
//! A clean-room compiler construction framework in pure Rust.
//!
//! LatticeFoundry provides the reusable machinery a compiler back end needs:
//! a typed SSA intermediate representation, a verifier, a pass and analysis
//! pipeline, target-independent code generation, a machine-code / object-file
//! layer, pluggable targets, and a linker core.
//!
//! It is delivered as a single package: one library ([`latticefoundry`](crate))
//! plus a family of driver binaries (`lf`, `lf-ld`, `lf-as`, `lf-opt`,
//! `lf-dis`) under `src/bin/`. This is **not** a Cargo workspace.
//!
//! ## Principles
//!
//! - **Clean room.** Every line is written from first principles. No source,
//!   text format, or table is copied or transliterated from any third-party
//!   compiler or toolchain.
//! - **Only our own crates.** The dependencies are only our own focused,
//!   clean-room library crates — `z3rs` (SMT solver) and `puremp`
//!   (arbitrary-precision numerics). Nothing third-party, no C.
//! - **Pure, safe Rust.** `unsafe` is a `warn`-level lint and used only where
//!   an invariant genuinely cannot be expressed in the type system.
//!
//! The current state is an early scaffold; see `ROADMAP.md` for the phased
//! build-out plan. Each module documents the roadmap phase that fills it in.

pub mod analysis;
pub mod codegen;
pub mod ir;
pub mod link;
pub mod mc;
pub mod pass;
pub mod support;
pub mod target;
pub mod transform;
pub mod verify;

pub use ir::Module;

/// The framework version, taken from the crate manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
