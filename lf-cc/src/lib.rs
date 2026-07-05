//! lf-cc — a C frontend for LatticeFoundry (separate crate; see Cargo.toml).
//!
//! Lexes/parses/type-checks a freestanding subset of C and lowers it to
//! `latticefoundry::ir`, reusing the framework's optimize → codegen → link
//! pipeline. Stub — implemented next.

/// Placeholder so the crate builds until the frontend lands.
pub const NAME: &str = "lf-cc";
