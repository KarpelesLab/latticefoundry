//! The machine-code layer.
//!
//! This is the target-*independent* framework for turning selected instructions
//! into bytes and packaging them as relocatable objects (ROADMAP Phase 6). It
//! knows nothing about any real ISA's opcodes: a concrete backend (Phase 7)
//! plugs its instruction encoder into the [emitter](emit) and records
//! relocations against [symbols](object), and the resulting
//! [`ObjectModule`](object::ObjectModule) is serialized either to our own
//! [`.lfo`](lfo) format or to a standard [ELF64](elf) relocatable object.
//!
//! The pieces:
//!
//! - [`emit`] — a growable little-endian byte buffer with a **label + fixup**
//!   mechanism: define labels, reference them before they are bound (forward
//!   references), resolve PC-relative and absolute fixups at finalize time, and
//!   surface unresolved external references as relocations.
//! - [`object`] — the neutral model: [`Section`](object::Section),
//!   [`Symbol`](object::Symbol), [`Relocation`](object::Relocation), and the
//!   [`ObjectModule`](object::ObjectModule) that aggregates them, plus the
//!   extensible generic [`RelocKind`](object::RelocKind) set.
//! - [`lfo`] — our versioned, compact, lossless `.lfo` object serialization.
//! - [`elf`] — an ELF64 `ET_REL` writer for x86-64, implemented from the ELF
//!   specification.
//!
//! Everything here is deterministic (tenet T5): the same inputs always produce
//! byte-identical output.

pub mod elf;
pub mod emit;
pub mod lfo;
pub mod object;

#[doc(inline)]
pub use emit::{Emitted, Emitter, Label};
#[doc(inline)]
pub use object::{
    ObjectModule, RelocKind, Relocation, Section, SectionKind, Symbol, SymbolBinding, SymbolType,
    SymbolValue,
};
