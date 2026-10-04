//! The AVR decoder (not yet implemented: every encoding decodes as data).

use super::Inst;

/// Decode one AVR instruction from the start of `bytes` (non-empty),
/// located at address `addr`.
pub fn decode(bytes: &[u8], _addr: u64) -> Inst {
    Inst::data(bytes, 2, true)
}
