//! The code emitter: a growable byte buffer with a label + fixup mechanism
//! (ROADMAP Phase 6).
//!
//! A target's instruction encoder appends bytes with the little-endian helpers
//! ([`Emitter::u8`], [`u16`](Emitter::u16), [`u32`](Emitter::u32),
//! [`u64`](Emitter::u64), [`bytes`](Emitter::bytes)) and refers to *locations*
//! it may not have emitted yet through two kinds of reference:
//!
//! - a **[`Label`]** — a location *inside this same buffer*. A reference to a
//!   label may appear before the label is bound (a **forward reference**); at
//!   [`finish`](Emitter::finish) time every label reference is resolved and the
//!   field is patched in place. Absolute references patch the label's
//!   section-relative offset; PC-relative references patch the displacement from
//!   the end of the field to the label.
//! - an **external symbol**, named by string. These cannot be resolved within
//!   the buffer, so each becomes a [`EmittedReloc`] (a pending
//!   [`Relocation`](crate::mc::object::Relocation)) in the
//!   [`Emitted`] result.
//!
//! A third kind of reference is the **relaxable branch** ([`Emitter::branch`]):
//! a PC-relative jump to a label with a short (`rel8`) and a long (`rel32`)
//! encoding. Every such branch starts short; [`finish`](Emitter::finish) lays
//! the buffer out, grows each branch whose target is out of the short form's
//! reach, and repeats until nothing changes (sizes only grow, so this
//! terminates). The bytes between branches are moved, never re-encoded, so
//! every other label, fixup and offset is remapped to where it ends up; a
//! caller that needs final offsets of its own binds labels and reads them back
//! with [`Emitter::finish_with_labels`]. Code that computes a PC-relative
//! displacement by hand must not span a relaxable branch.
//!
//! The emitter is entirely target-independent and deterministic: relocations
//! come out in emission order and the same sequence of calls always yields the
//! same bytes.

use crate::mc::object::RelocKind;

/// A location within an [`Emitter`]'s buffer, resolved at
/// [`finish`](Emitter::finish) time.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Label(u32);

impl Label {
    /// The dense index backing this label.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// The target of a fixup: an internal [`Label`] or an external symbol name.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Ref {
    /// A location inside the same buffer.
    Label(Label),
    /// An external symbol, resolved by the linker via a relocation.
    Symbol(String),
}

/// A pending relocation surfaced by [`Emitter::finish`]: an external reference
/// that could not be resolved inside the buffer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EmittedReloc {
    /// Byte offset within the finished buffer of the field to patch.
    pub offset: u64,
    /// The external symbol referenced.
    pub symbol: String,
    /// How the field is to be patched.
    pub kind: RelocKind,
    /// The RELA addend, already adjusted for the PC-relative field width.
    pub addend: i64,
}

/// The output of [`Emitter::finish`]: the finished bytes plus the relocations
/// its unresolved external references produced.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Emitted {
    /// The assembled bytes, with every internal label reference patched in.
    pub bytes: Vec<u8>,
    /// One entry per unresolved external reference, in emission order.
    pub relocations: Vec<EmittedReloc>,
}

/// Why finishing an [`Emitter`] failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum EmitError {
    /// A label was referenced but never bound to an offset.
    UndefinedLabel(Label),
    /// A resolved fixup value did not fit in its field (e.g. a PC-relative
    /// displacement wider than 32 bits). `offset` is the field position and
    /// `value` the out-of-range value.
    FieldOverflow {
        /// The byte offset of the field that overflowed.
        offset: u64,
        /// The value that did not fit.
        value: i64,
    },
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::UndefinedLabel(l) => write!(f, "label {} was never bound", l.0),
            EmitError::FieldOverflow { offset, value } => {
                write!(f, "fixup value {value} at offset {offset} does not fit its field")
            }
        }
    }
}

impl std::error::Error for EmitError {}

/// One recorded reference to patch when finishing.
#[derive(Clone, Debug)]
struct Fixup {
    at: u64,
    kind: RelocKind,
    target: Ref,
    addend: i64,
}

/// One relaxable branch: its long form sits in the buffer at `at` until
/// [`Emitter::finish`] decides which form it takes.
#[derive(Clone, Debug)]
struct Relax {
    at: u64,
    short: Vec<u8>,
    long: Vec<u8>,
    label: Label,
}

impl Relax {
    /// The length of the long form (opcode + `rel32`).
    fn long_len(&self) -> u64 {
        self.long.len() as u64 + 4
    }

    /// The length of the short form (opcode + `rel8`).
    fn short_len(&self) -> u64 {
        self.short.len() as u64 + 1
    }
}

/// A growable little-endian byte buffer with a label + fixup resolver.
#[derive(Clone, Debug, Default)]
pub struct Emitter {
    buf: Vec<u8>,
    labels: Vec<Option<u64>>,
    fixups: Vec<Fixup>,
    relaxes: Vec<Relax>,
}

impl Emitter {
    /// Create an empty emitter.
    pub fn new() -> Emitter {
        Emitter::default()
    }

    /// The current write position (number of bytes emitted so far).
    #[inline]
    pub fn offset(&self) -> u64 {
        self.buf.len() as u64
    }

    /// Borrow the bytes emitted so far (before fixups are applied).
    #[inline]
    pub fn bytes_so_far(&self) -> &[u8] {
        &self.buf
    }

    // --- raw little-endian appends ---

    /// Append one byte.
    #[inline]
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Append a little-endian `u16`.
    #[inline]
    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a little-endian `u32`.
    #[inline]
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a little-endian `u64`.
    #[inline]
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append raw bytes verbatim.
    #[inline]
    pub fn bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Append `n` zero bytes (e.g. to reserve padding).
    #[inline]
    pub fn zeros(&mut self, n: usize) {
        self.buf.resize(self.buf.len() + n, 0);
    }

    /// Pad with zero bytes until the write position is a multiple of `align`
    /// (which must be a power of two, or `1` for no padding).
    pub fn align_to(&mut self, align: u64) {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
        let mask = align - 1;
        let pad = (align - (self.buf.len() as u64 & mask)) & mask;
        self.zeros(pad as usize);
    }

    // --- labels ---

    /// Create a fresh, initially-unbound label.
    pub fn create_label(&mut self) -> Label {
        let id = Label(self.labels.len() as u32);
        self.labels.push(None);
        id
    }

    /// Bind `label` to the current write position.
    ///
    /// Re-binding an already-bound label overwrites its offset (the last
    /// binding wins); this keeps the operation total for simple callers.
    pub fn bind_label(&mut self, label: Label) {
        let here = self.offset();
        self.labels[label.index()] = Some(here);
    }

    /// Bind `label` to an explicit offset.
    pub fn bind_label_at(&mut self, label: Label, offset: u64) {
        self.labels[label.index()] = Some(offset);
    }

    // --- references (append a placeholder field + record a fixup) ---

    /// Emit a zeroed field of `kind`'s width at the current position and record
    /// a fixup that patches it from `target`. For PC-relative kinds the addend
    /// is adjusted by the field width so an external relocation and an internal
    /// resolution agree.
    pub fn reference(&mut self, kind: RelocKind, target: Ref, addend: i64) {
        let at = self.offset();
        self.zeros(kind.field_width());
        self.fixups.push(Fixup { at, kind, target, addend });
    }

    /// Reference a [`Label`] with the given relocation kind and addend.
    #[inline]
    pub fn reference_label(&mut self, kind: RelocKind, label: Label, addend: i64) {
        self.reference(kind, Ref::Label(label), addend);
    }

    /// Reference an external symbol by name with the given relocation kind and
    /// addend.
    #[inline]
    pub fn reference_symbol(&mut self, kind: RelocKind, name: impl Into<String>, addend: i64) {
        self.reference(kind, Ref::Symbol(name.into()), addend);
    }

    // Ergonomic shorthands for the common kinds.

    /// Emit an absolute 64-bit reference ([`RelocKind::Abs64`]).
    #[inline]
    pub fn abs64(&mut self, target: Ref, addend: i64) {
        self.reference(RelocKind::Abs64, target, addend);
    }

    /// Emit an absolute 32-bit reference ([`RelocKind::Abs32`]).
    #[inline]
    pub fn abs32(&mut self, target: Ref, addend: i64) {
        self.reference(RelocKind::Abs32, target, addend);
    }

    /// Emit a PC-relative 32-bit reference ([`RelocKind::Pc32`]).
    #[inline]
    pub fn pcrel32(&mut self, target: Ref, addend: i64) {
        self.reference(RelocKind::Pc32, target, addend);
    }

    /// Emit a PC-relative 32-bit PLT reference ([`RelocKind::Plt32`]), the usual
    /// form for a `call`/`jmp` to an external function.
    #[inline]
    pub fn plt32(&mut self, target: Ref, addend: i64) {
        self.reference(RelocKind::Plt32, target, addend);
    }

    /// Emit a relaxable PC-relative branch to `label`: the opcode bytes
    /// `short` followed by a `rel8` displacement when the target is within
    /// reach, else `long` followed by a `rel32` (both measured from the end
    /// of the instruction). The form is chosen by [`Emitter::finish`]; until
    /// then the long form occupies the buffer.
    pub fn branch(&mut self, short: &[u8], long: &[u8], label: Label) {
        let at = self.offset();
        self.bytes(long);
        self.zeros(4);
        self.relaxes.push(Relax { at, short: short.to_vec(), long: long.to_vec(), label });
    }

    /// Resolve every fixup and return the finished bytes plus the relocations
    /// that unresolved external references produced.
    ///
    /// Relaxable branches are sized and encoded first (see the
    /// [module docs](self)). Internal [`Label`] references are patched in
    /// place; each external symbol reference becomes an [`EmittedReloc`].
    /// Fails if a referenced label was never bound, or if a resolved value
    /// does not fit its field.
    pub fn finish(self) -> Result<Emitted, EmitError> {
        self.finish_with_labels().map(|(out, _)| out)
    }

    /// Like [`Emitter::finish`], but also returns every label's final offset
    /// (by [`Label::index`]; `None` for one never bound), after relaxation
    /// moved the code.
    pub fn finish_with_labels(mut self) -> Result<(Emitted, Vec<Option<u64>>), EmitError> {
        self.relax()?;
        let mut relocations = Vec::new();
        for fx in &self.fixups {
            match &fx.target {
                Ref::Label(label) => {
                    let target_off = self.labels[label.index()]
                        .ok_or(EmitError::UndefinedLabel(*label))?;
                    let width = fx.kind.field_width();
                    let value = if fx.kind.is_pcrel() {
                        // Displacement from the end of the field to the target.
                        (target_off as i64) - (fx.at as i64 + width as i64) + fx.addend
                    } else {
                        // Section-relative absolute value (base 0).
                        target_off as i64 + fx.addend
                    };
                    patch(&mut self.buf, fx.at, width, value)?;
                }
                Ref::Symbol(name) => {
                    // Unresolved: emit a relocation. For a PC-relative field the
                    // reloc is applied at the field address P, so fold the field
                    // width into the addend (S + A - P with A = addend - width
                    // reproduces target - (P + width) + addend).
                    let addend = if fx.kind.is_pcrel() {
                        fx.addend - fx.kind.field_width() as i64
                    } else {
                        fx.addend
                    };
                    relocations.push(EmittedReloc {
                        offset: fx.at,
                        symbol: name.clone(),
                        kind: fx.kind,
                        addend,
                    });
                }
            }
        }
        Ok((Emitted { bytes: self.buf, relocations }, self.labels))
    }

    /// Size the relaxable branches, rebuild the buffer with each in its
    /// chosen form, and move every label and fixup to its new offset.
    fn relax(&mut self) -> Result<(), EmitError> {
        if self.relaxes.is_empty() {
            return Ok(());
        }
        let n = self.relaxes.len();
        let mut long = vec![false; n];
        // `shrink[i]`: the bytes saved by the branches before branch `i`.
        let mut shrink = vec![0u64; n + 1];
        let saved_before = |shrink: &[u64], relaxes: &[Relax], p: u64| -> u64 {
            // Branches entirely before `p` (`at < p`); a label at a branch's
            // own start stays in front of it.
            let k = relaxes.partition_point(|r| r.at < p);
            shrink[k]
        };
        loop {
            for i in 0..n {
                let r = &self.relaxes[i];
                let saving = if long[i] { 0 } else { r.long_len() - r.short_len() };
                shrink[i + 1] = shrink[i] + saving;
            }
            let mut changed = false;
            for i in 0..n {
                if long[i] {
                    continue;
                }
                let r = &self.relaxes[i];
                let target = self.labels[r.label.index()].ok_or(EmitError::UndefinedLabel(r.label))?;
                let target = target - saved_before(&shrink, &self.relaxes, target);
                let end = r.at - shrink[i] + r.short_len();
                let disp = target as i64 - end as i64;
                if i8::try_from(disp).is_err() {
                    long[i] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Rebuild the buffer: the bytes between branches move down, each
        // branch takes its chosen form.
        let mut out = Vec::with_capacity(self.buf.len());
        let mut from = 0usize;
        for (i, r) in self.relaxes.iter().enumerate() {
            out.extend_from_slice(&self.buf[from..r.at as usize]);
            from = (r.at + r.long_len()) as usize;
            let target = self.labels[r.label.index()].ok_or(EmitError::UndefinedLabel(r.label))?;
            let target = (target - saved_before(&shrink, &self.relaxes, target)) as i64;
            if long[i] {
                out.extend_from_slice(&r.long);
                let disp = target - (out.len() as i64 + 4);
                let disp = i32::try_from(disp).map_err(|_| EmitError::FieldOverflow { offset: out.len() as u64, value: disp })?;
                out.extend_from_slice(&disp.to_le_bytes());
            } else {
                out.extend_from_slice(&r.short);
                let disp = target - (out.len() as i64 + 1);
                out.push(disp as i8 as u8);
            }
        }
        out.extend_from_slice(&self.buf[from..]);
        for l in self.labels.iter_mut().flatten() {
            *l -= saved_before(&shrink, &self.relaxes, *l);
        }
        for fx in &mut self.fixups {
            fx.at -= saved_before(&shrink, &self.relaxes, fx.at);
        }
        self.buf = out;
        self.relaxes.clear();
        Ok(())
    }
}

/// Patch a little-endian field of `width` bytes at `at` with `value`, checking
/// that it fits.
fn patch(buf: &mut [u8], at: u64, width: usize, value: i64) -> Result<(), EmitError> {
    // Accept anything representable in either the signed or the unsigned
    // field (it is reinterpreted per the relocation's signedness downstream).
    if crate::mc::object::write_field(buf, at as usize, width, value, crate::ir::Endian::Little) {
        Ok(())
    } else {
        Err(EmitError::FieldOverflow { offset: at, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn little_endian_appends() {
        let mut e = Emitter::new();
        e.u8(0x11);
        e.u16(0x2233);
        e.u32(0x4455_6677);
        e.u64(0x8899_aabb_ccdd_eeff);
        e.bytes(&[0xde, 0xad]);
        let out = e.finish().unwrap();
        assert_eq!(
            out.bytes,
            vec![
                0x11, 0x33, 0x22, 0x77, 0x66, 0x55, 0x44, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99,
                0x88, 0xde, 0xad,
            ]
        );
        assert!(out.relocations.is_empty());
    }

    #[test]
    fn backward_pcrel_reference_resolves() {
        // Bind a label, emit some bytes, then a pcrel32 pointing back to it.
        let mut e = Emitter::new();
        let target = e.create_label();
        e.bind_label(target); // at offset 0
        e.u8(0x90);
        e.u8(0x90); // now at offset 2
        e.pcrel32(Ref::Label(target), 0); // field at offset 2, ends at 6
        let out = e.finish().unwrap();
        // disp = target(0) - (2 + 4) = -6
        let disp = i32::from_le_bytes([out.bytes[2], out.bytes[3], out.bytes[4], out.bytes[5]]);
        assert_eq!(disp, -6);
        assert!(out.relocations.is_empty());
    }

    #[test]
    fn forward_pcrel_reference_resolves() {
        let mut e = Emitter::new();
        let target = e.create_label();
        e.u8(0xe9); // a one-byte opcode
        e.pcrel32(Ref::Label(target), 0); // field at 1, ends at 5
        e.u8(0xcc);
        e.u8(0xcc);
        e.bind_label(target); // bound at offset 7
        let out = e.finish().unwrap();
        // disp = 7 - (1 + 4) = 2
        let disp = i32::from_le_bytes([out.bytes[1], out.bytes[2], out.bytes[3], out.bytes[4]]);
        assert_eq!(disp, 2);
    }

    #[test]
    fn absolute_label_reference_uses_offset() {
        let mut e = Emitter::new();
        let l = e.create_label();
        e.abs64(Ref::Label(l), 0x10); // field at offset 0
        e.zeros(3);
        e.bind_label(l); // bound at offset 11
        let out = e.finish().unwrap();
        let v = u64::from_le_bytes(out.bytes[0..8].try_into().unwrap());
        assert_eq!(v, 11 + 0x10);
    }

    #[test]
    fn external_reference_becomes_relocation() {
        let mut e = Emitter::new();
        e.u8(0xe8); // call rel32
        e.plt32(Ref::Symbol("printf".to_owned()), 0); // field at 1
        let out = e.finish().unwrap();
        assert_eq!(out.relocations.len(), 1);
        let r = &out.relocations[0];
        assert_eq!(r.offset, 1);
        assert_eq!(r.symbol, "printf");
        assert_eq!(r.kind, RelocKind::Plt32);
        // PC-relative: width folded into the addend.
        assert_eq!(r.addend, -4);
        // The field is left zeroed for the linker to fill.
        assert_eq!(&out.bytes[1..5], &[0, 0, 0, 0]);
    }

    #[test]
    fn abs64_external_reference_keeps_addend() {
        let mut e = Emitter::new();
        e.abs64(Ref::Symbol("global".to_owned()), 8);
        let out = e.finish().unwrap();
        assert_eq!(out.relocations.len(), 1);
        assert_eq!(out.relocations[0].addend, 8);
        assert_eq!(out.relocations[0].kind, RelocKind::Abs64);
    }

    #[test]
    fn undefined_label_is_an_error() {
        let mut e = Emitter::new();
        let l = e.create_label();
        e.pcrel32(Ref::Label(l), 0);
        assert_eq!(e.finish(), Err(EmitError::UndefinedLabel(l)));
    }

    #[test]
    fn pcrel_overflow_is_reported() {
        let mut e = Emitter::new();
        let l = e.create_label();
        e.bind_label_at(l, u32::MAX as u64 + 100); // far away
        e.pcrel32(Ref::Label(l), 0);
        match e.finish() {
            Err(EmitError::FieldOverflow { .. }) => {}
            other => panic!("expected overflow, got {other:?}"),
        }
    }

    #[test]
    fn align_to_pads_with_zeros() {
        let mut e = Emitter::new();
        e.u8(1);
        e.align_to(4);
        assert_eq!(e.offset(), 4);
        assert_eq!(e.bytes_so_far(), &[1, 0, 0, 0]);
    }

    #[test]
    fn emission_is_deterministic() {
        let build = || {
            let mut e = Emitter::new();
            let l = e.create_label();
            e.u8(0xe9);
            e.pcrel32(Ref::Label(l), 0);
            e.plt32(Ref::Symbol("f".to_owned()), 0);
            e.bind_label(l);
            e.u32(0xdead_beef);
            e.finish().unwrap()
        };
        assert_eq!(build(), build());
    }

    /// `jmp` to a label `gap` bytes of filler past the branch (forward) and
    /// the bytes produced.
    fn forward_jmp(gap: usize) -> Vec<u8> {
        let mut e = Emitter::new();
        let l = e.create_label();
        e.branch(&[0xeb], &[0xe9], l);
        e.zeros(gap);
        e.bind_label(l);
        e.u8(0xc3);
        e.finish().unwrap().bytes
    }

    #[test]
    fn relaxable_branch_takes_the_short_form_up_to_the_rel8_boundary() {
        // +127 is the farthest a rel8 reaches.
        let b = forward_jmp(127);
        assert_eq!(&b[..2], &[0xeb, 127]);
        assert_eq!(b.len(), 2 + 127 + 1);
        // One byte more needs the rel32 form.
        let b = forward_jmp(128);
        assert_eq!(&b[..5], &[0xe9, 128, 0, 0, 0]);
        assert_eq!(b.len(), 5 + 128 + 1);
    }

    #[test]
    fn backward_relaxable_branch_boundary() {
        // A backward branch reaches -128 from its end: the label 126 bytes
        // before a 2-byte branch.
        for (gap, short) in [(126usize, true), (127, false)] {
            let mut e = Emitter::new();
            let l = e.create_label();
            e.bind_label(l);
            e.zeros(gap);
            e.branch(&[0x75], &[0x0f, 0x85], l);
            let b = e.finish().unwrap().bytes;
            if short {
                assert_eq!(&b[gap..], &[0x75, 0x80]);
            } else {
                let disp = -(gap as i32) - 6;
                let mut want = vec![0x0f, 0x85];
                want.extend_from_slice(&disp.to_le_bytes());
                assert_eq!(&b[gap..], &want[..]);
            }
        }
    }

    #[test]
    fn relaxation_moves_labels_fixups_and_cascades() {
        // Two forward branches over each other: the outer one's target is in
        // reach only if the inner one is short. With 124 bytes between, both
        // fit; a pcrel32 fixup and a label after them move down too.
        let mut e = Emitter::new();
        let outer = e.create_label();
        let inner = e.create_label();
        let after = e.create_label();
        e.branch(&[0xeb], &[0xe9], outer);
        e.branch(&[0x74], &[0x0f, 0x84], inner);
        e.zeros(122);
        e.bind_label(inner);
        e.bind_label(outer);
        e.u8(0xe8);
        e.pcrel32(Ref::Label(after), 0);
        e.bind_label(after);
        let (out, labels) = e.finish_with_labels().unwrap();
        assert_eq!(&out.bytes[..4], &[0xeb, 124, 0x74, 122]);
        assert_eq!(labels[after.index()], Some(4 + 122 + 5));
        assert_eq!(&out.bytes[127..131], &[0, 0, 0, 0], "call to the next instruction");
        // With a longer body the inner branch grows, which pushes the outer
        // one out of reach as well.
        let mut e = Emitter::new();
        let outer = e.create_label();
        let inner = e.create_label();
        e.branch(&[0xeb], &[0xe9], outer);
        e.zeros(124);
        e.branch(&[0x74], &[0x0f, 0x84], inner);
        e.zeros(128);
        e.bind_label(inner);
        e.bind_label(outer);
        let b = e.finish().unwrap().bytes;
        assert_eq!(b[0], 0xe9);
        assert_eq!(&b[5 + 124..5 + 124 + 2], &[0x0f, 0x84]);
        assert_eq!(b.len(), 5 + 124 + 6 + 128);
    }

    #[test]
    fn unbound_relaxable_target_is_an_error() {
        let mut e = Emitter::new();
        let l = e.create_label();
        e.branch(&[0xeb], &[0xe9], l);
        assert_eq!(e.finish(), Err(EmitError::UndefinedLabel(l)));
    }
}
