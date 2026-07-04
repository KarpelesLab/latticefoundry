//! # LatticeFoundry
//!
//! A clean-room compiler construction framework in pure Rust.
//!
//! LatticeFoundry provides the reusable machinery a compiler back end needs:
//! a typed SSA intermediate representation, a verifier, a pass and analysis
//! pipeline, target-independent code generation, a machine-code / object-file
//! layer, pluggable targets, and a linker core.
//!
//! It is delivered as a single library ([`latticefoundry`](crate)) plus a
//! family of driver binaries (`lf`, `lf-ld`, `lf-as`, `lf-opt`, `lf-dis`).
//!
//! ## Principles
//!
//! - **Clean room.** Every line is written from first principles. No source,
//!   text format, or table is copied or transliterated from any existing
//!   compiler or toolchain.
//! - **No third-party code.** The only dependencies are other crates in this
//!   workspace (for example [`z3rs`], our own SMT solver).
//! - **Pure, safe Rust.** `unsafe` is a `warn`-level lint and used only where
//!   an invariant genuinely cannot be expressed in the type system.
//!
//! The current state is an early scaffold; see `ROADMAP.md` for the phased
//! build-out plan. Each module documents the roadmap phase that fills it in.

pub mod codegen;
pub mod ir;
pub mod link;
pub mod mc;
pub mod pass;
pub mod support;
pub mod target;
pub mod verify;

pub use ir::Module;

/// The framework version, taken from the crate manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
