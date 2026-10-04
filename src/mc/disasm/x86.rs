//! The x86-64 decoder (not yet implemented: every encoding decodes as data).

use super::{Inst, Options};

/// Decode one x86-64 instruction from the start of `bytes` (non-empty),
/// located at address `addr`.
pub fn decode(bytes: &[u8], _addr: u64, _opts: &Options) -> Inst {
    Inst::data(bytes, 1, true)
}
