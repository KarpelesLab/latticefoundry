//! The Thumb-2 decoder (not yet implemented: every encoding decodes as data).

use super::{Inst, State};

/// Decode one Thumb-2 instruction from the start of `bytes` (non-empty),
/// located at address `addr`, outside any IT block.
pub fn decode(bytes: &[u8], addr: u64) -> Inst {
    decode_in(bytes, addr, &mut State::default())
}

/// Decode one Thumb-2 instruction, reading and advancing the IT-block
/// `state`.
pub fn decode_in(bytes: &[u8], _addr: u64, state: &mut State) -> Inst {
    *state = State::default();
    Inst::data(bytes, 2, true)
}
