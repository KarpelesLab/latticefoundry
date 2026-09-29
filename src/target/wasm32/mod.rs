//! The **wasm32** backend (in progress): LatticeFoundry IR to WebAssembly
//! modules, clean-room from the WebAssembly Core Specification and the
//! WebAssembly tool-conventions linking format.
//!
//! WebAssembly is a structured stack machine, so this backend will not use the
//! register-machine pipeline of the other targets. Its first pieces:
//!
//! - [`leb`] — the LEB128 integer encoding of the binary format, including the
//!   padded forms relocatable objects use;
//! - [`structure`] — control-flow structuring: a CFG to nested
//!   `block` / `loop` / `if` / `br_table` constructs, placed from the dominator
//!   tree, with a dispatch loop for irreducible CFGs.

pub mod leb;
pub mod structure;
