//! The machine-code layer.
//!
//! Encodes target instructions into bytes and emits relocatable object files
//! in the LatticeFoundry object format (`.lfo`) as well as the standard
//! platform formats. Also backs the `lf-as` assembler and `lf-dis`
//! disassembler. See ROADMAP Phase 6.

/// A relocatable object being assembled (placeholder).
#[derive(Debug, Default)]
pub struct Object;
