//! The LP64D calling convention's argument classification (RISC-V ELF psABI,
//! "Integer Calling Convention" and "Hardware Floating-point Calling
//! Convention", with `XLEN` = `FLEN` = 64).
//!
//! Every argument, in order, is assigned [`Loc`]ations by an [`Assigner`]
//! that tracks the next free integer register (`a0`–`a7`), floating-point
//! register (`fa0`–`fa7`) and stack offset:
//!
//! - an integer or pointer scalar takes the next integer register, else an
//!   8-byte stack slot;
//! - a named `f32`/`f64` takes the next floating-point register; when those
//!   are exhausted — or for a variadic argument — it is passed like an
//!   integer of its size (its bits in an integer register, else on the
//!   stack);
//! - an aggregate (struct or array, **flattened**: nested structs and array
//!   elements count as separate fields) whose fields are one float, two
//!   floats, or one float and one integer of at most 8 bytes (in either
//!   order) is passed in a floating-point register per float field and an
//!   integer register per integer field, *if* enough of both remain;
//!   otherwise, and for every other aggregate, the integer convention
//!   applies: up to 8 bytes in one integer register, up to 16 in two (with
//!   the second half on the stack when only one register is left), anything
//!   larger **by reference** (a pointer to a caller-made copy, in the next
//!   integer register or on the stack);
//! - an empty aggregate is not passed at all;
//! - every stack slot is 8 bytes, in argument order from the outgoing
//!   argument area at the caller's `sp`.
//!
//! A value is returned as a first named argument of its type would be passed
//! with only `a0`/`a1` and `fa0`/`fa1` available; one that would not fit
//! (anything passed by reference, or partly on the stack) is returned
//! through memory the caller provides, its address passed as a hidden first
//! argument in `a0`.

use crate::ir::types::{Type, TypeContext, TypeId};

/// A flattened scalar field of an aggregate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Leaf {
    /// An integer or pointer of `size` bytes at byte offset `off`.
    Int { off: u64, size: u64 },
    /// A float of `width` bits at byte offset `off`.
    Float { off: u64, width: u32 },
}

/// How a value's type crosses the ABI.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Class {
    /// An integer or pointer scalar.
    Int,
    /// An `f32` (32) or `f64` (64).
    Float(u32),
    /// An integer of two `XLEN` words (`i128`).
    Wide,
    /// An aggregate eligible for the floating-point convention (one float,
    /// two floats, or a float and an integer), with its total size.
    FpAgg(Vec<Leaf>, u64),
    /// An aggregate of at most 16 bytes under the integer convention.
    IntAgg(u64),
    /// An aggregate larger than 16 bytes: passed by reference.
    Ref,
    /// A zero-sized aggregate: not passed.
    Empty,
}

/// Whether `ty` is an aggregate, which the backend represents by a pointer to
/// its storage.
pub(crate) fn is_aggregate(types: &TypeContext, ty: TypeId) -> bool {
    matches!(types.get(ty), Type::Struct(_) | Type::Array(..))
}

/// Flatten the scalar leaves of `ty` at byte offset `base` into `out`, giving
/// up (`false`) past `limit` leaves or at a leaf that is neither an integer,
/// a pointer nor an `f32`/`f64`.
fn flatten(types: &TypeContext, ty: TypeId, base: u64, out: &mut Vec<Leaf>, limit: usize) -> bool {
    match types.get(ty) {
        Type::Struct(fields) => {
            let n = fields.len();
            (0..n).all(|i| {
                let (off, fty) = types.field_offset(ty, i as u32);
                flatten(types, fty, base + off, out, limit)
            })
        }
        Type::Array(elem, len) => {
            let (elem, len) = (*elem, *len);
            let stride = types.size_of(elem);
            (0..len).all(|k| flatten(types, elem, base + k * stride, out, limit))
        }
        Type::Float(k) if matches!(k.bit_width(), 32 | 64) => {
            out.push(Leaf::Float { off: base, width: k.bit_width() });
            out.len() <= limit
        }
        Type::Int(_) | Type::Ptr | Type::PtrIn(_) => {
            let size = types.size_of(ty);
            out.push(Leaf::Int { off: base, size });
            size <= 8 && out.len() <= limit
        }
        _ => false,
    }
}

/// Classify a type (see the [module docs](self)).
pub(crate) fn classify(types: &TypeContext, ty: TypeId) -> Class {
    match types.get(ty) {
        Type::Float(k) => return Class::Float(k.bit_width()),
        Type::Int(b) if *b > 64 => return Class::Wide,
        Type::Struct(_) | Type::Array(..) => {}
        _ => return Class::Int,
    }
    let size = types.size_of(ty);
    if size == 0 {
        return Class::Empty;
    }
    let mut leaves = Vec::new();
    if size <= 16 && flatten(types, ty, 0, &mut leaves, 2) {
        let floats = leaves.iter().filter(|l| matches!(l, Leaf::Float { .. })).count();
        if floats >= 1 {
            return Class::FpAgg(leaves, size);
        }
    }
    if size <= 16 { Class::IntAgg(size) } else { Class::Ref }
}

/// Where one part of an argument travels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Loc {
    /// An integer register (hardware number).
    Gpr(u16),
    /// A floating-point register (hardware number).
    Fpr(u16),
    /// An 8-byte slot at this offset of the stack-argument area.
    Stack(u64),
}

/// Which part of an argument a [`Loc`] carries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Part {
    /// The whole scalar.
    Whole,
    /// 64-bit word `k` of an `i128` (0: the low half).
    Half(u8),
    /// `size` bytes of the aggregate at byte offset `off`: an integer chunk,
    /// or (with `float`) a float field of that width.
    Chunk { off: u64, size: u64, float: Option<u32> },
    /// The address of a copy of the aggregate (passed by reference).
    Ref,
}

/// The register/stack cursor of one argument list.
#[derive(Clone, Debug)]
pub(crate) struct Assigner {
    gpr: u16,
    fpr: u16,
    max_gpr: u16,
    max_fpr: u16,
    /// Bytes of stack arguments assigned so far.
    pub(crate) stack: u64,
}

impl Assigner {
    /// An argument list: `a0`–`a7`, `fa0`–`fa7`, then the stack. With `sret`,
    /// `a0` carries the hidden return-memory pointer.
    pub(crate) fn args(sret: bool) -> Assigner {
        Assigner { gpr: u16::from(sret), fpr: 0, max_gpr: 8, max_fpr: 8, stack: 0 }
    }

    /// A return value: `a0`/`a1` and `fa0`/`fa1`.
    fn ret() -> Assigner {
        Assigner { gpr: 0, fpr: 0, max_gpr: 2, max_fpr: 2, stack: 0 }
    }

    fn take_gpr(&mut self) -> Option<Loc> {
        (self.gpr < self.max_gpr).then(|| {
            self.gpr += 1;
            Loc::Gpr(10 + self.gpr - 1)
        })
    }

    fn take_fpr(&mut self) -> Option<Loc> {
        (self.fpr < self.max_fpr).then(|| {
            self.fpr += 1;
            Loc::Fpr(10 + self.fpr - 1)
        })
    }

    fn take_stack(&mut self) -> Loc {
        self.stack += 8;
        Loc::Stack(self.stack - 8)
    }

    /// An integer register, else a stack slot.
    fn int_loc(&mut self) -> Loc {
        self.take_gpr().unwrap_or_else(|| self.take_stack())
    }

    /// Assign the next argument, of type `ty` (`named`: not a variadic
    /// argument), returning its parts and their locations.
    pub(crate) fn assign(&mut self, types: &TypeContext, ty: TypeId, named: bool) -> Vec<(Part, Loc)> {
        match classify(types, ty) {
            Class::Int => vec![(Part::Whole, self.int_loc())],
            Class::Float(_) => {
                let fpr = if named { self.take_fpr() } else { None };
                vec![(Part::Whole, fpr.unwrap_or_else(|| self.int_loc()))]
            }
            Class::FpAgg(leaves, size) => {
                let nf = leaves.iter().filter(|l| matches!(l, Leaf::Float { .. })).count() as u16;
                let ni = leaves.len() as u16 - nf;
                if named && self.fpr + nf <= self.max_fpr && self.gpr + ni <= self.max_gpr {
                    leaves
                        .iter()
                        .map(|l| match *l {
                            Leaf::Float { off, width } => {
                                let loc = self.take_fpr().expect("checked above");
                                (Part::Chunk { off, size: u64::from(width / 8), float: Some(width) }, loc)
                            }
                            Leaf::Int { off, size } => {
                                let loc = self.take_gpr().expect("checked above");
                                (Part::Chunk { off, size, float: None }, loc)
                            }
                        })
                        .collect()
                } else {
                    self.int_agg(size)
                }
            }
            Class::IntAgg(size) => self.int_agg(size),
            Class::Wide => self.wide(named),
            Class::Ref => vec![(Part::Ref, self.int_loc())],
            Class::Empty => Vec::new(),
        }
    }

    /// A scalar of two `XLEN` words: a register pair (low word first), the
    /// low word in the last register and the high one on the stack, or a
    /// 16-aligned stack slot. A variadic one takes an aligned (even) pair or
    /// goes on the stack.
    fn wide(&mut self, named: bool) -> Vec<(Part, Loc)> {
        if !named && self.gpr % 2 == 1 {
            self.gpr = (self.gpr + 1).min(self.max_gpr);
        }
        if self.gpr < self.max_gpr {
            let lo = self.take_gpr().expect("a register is left");
            let hi = self.int_loc();
            return vec![(Part::Half(0), lo), (Part::Half(1), hi)];
        }
        self.stack = self.stack.next_multiple_of(16);
        let lo = self.take_stack();
        let hi = self.take_stack();
        vec![(Part::Half(0), lo), (Part::Half(1), hi)]
    }

    /// An aggregate of at most 16 bytes under the integer convention: one or
    /// two 8-byte chunks, each in the next integer register or on the stack.
    fn int_agg(&mut self, size: u64) -> Vec<(Part, Loc)> {
        let mut out = vec![(Part::Chunk { off: 0, size: size.min(8), float: None }, self.int_loc())];
        if size > 8 {
            out.push((Part::Chunk { off: 8, size: size - 8, float: None }, self.int_loc()));
        }
        out
    }
}

/// How a function returns a value of type `ty`: in registers (its parts and
/// their `a0`/`a1`/`fa0`/`fa1` locations), or `None` for memory the caller
/// provides (a hidden first argument).
pub(crate) fn ret_locs(types: &TypeContext, ty: TypeId) -> Option<Vec<(Part, Loc)>> {
    let parts = Assigner::ret().assign(types, ty, true);
    let in_regs = parts.iter().all(|(p, l)| *p != Part::Ref && !matches!(l, Loc::Stack(_)));
    in_regs.then_some(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::types::FloatKind;

    /// The psABI's examples of the floating-point convention, by structure.
    #[test]
    fn classification_follows_the_psabi() {
        let mut t = TypeContext::new();
        let (f32t, f64t) = (t.float(FloatKind::F32), t.float(FloatKind::F64));
        let (i8t, i32t, i64t, i128t) = (t.int(8), t.int(32), t.int(64), t.int(128));
        let ptr = t.ptr();
        let s = |t: &mut TypeContext, f: Vec<TypeId>| t.struct_(f);
        // { double } is a lone double; { float, float } two floats.
        let d1 = s(&mut t, vec![f64t]);
        assert_eq!(classify(&t, d1), Class::FpAgg(vec![Leaf::Float { off: 0, width: 64 }], 8));
        let ff = s(&mut t, vec![f32t, f32t]);
        assert_eq!(
            classify(&t, ff),
            Class::FpAgg(vec![Leaf::Float { off: 0, width: 32 }, Leaf::Float { off: 4, width: 32 }], 8)
        );
        // { int8, double } (integer first) and { double, ptr }.
        let id = s(&mut t, vec![i8t, f64t]);
        assert_eq!(
            classify(&t, id),
            Class::FpAgg(vec![Leaf::Int { off: 0, size: 1 }, Leaf::Float { off: 8, width: 64 }], 16)
        );
        let dp = s(&mut t, vec![f64t, ptr]);
        assert!(matches!(classify(&t, dp), Class::FpAgg(..)));
        // Arrays and nesting flatten: { [1 x float], { float } } is two floats.
        let a1 = t.array(f32t, 1);
        let inner = s(&mut t, vec![f32t]);
        let nested = s(&mut t, vec![a1, inner]);
        assert!(matches!(classify(&t, nested), Class::FpAgg(ref l, 8) if l.len() == 2));
        // Three fields, two integers, or an integer wider than XLEN: integer
        // convention; past 16 bytes, by reference.
        let fff = s(&mut t, vec![f32t, f32t, f32t]);
        assert_eq!(classify(&t, fff), Class::IntAgg(12));
        let ii = s(&mut t, vec![i32t, i32t]);
        assert_eq!(classify(&t, ii), Class::IntAgg(8));
        let fw = s(&mut t, vec![f32t, i128t]);
        assert_eq!(classify(&t, fw), Class::Ref);
        let big = s(&mut t, vec![f64t, f64t, f64t]);
        assert_eq!(classify(&t, big), Class::Ref);
        let empty = s(&mut t, vec![]);
        assert_eq!(classify(&t, empty), Class::Empty);
        assert_eq!(classify(&t, i64t), Class::Int);
        assert_eq!(classify(&t, i128t), Class::Wide);
        assert_eq!(classify(&t, f32t), Class::Float(32));
    }

    #[test]
    fn assignment_spills_floats_to_integer_registers_then_the_stack() {
        let mut t = TypeContext::new();
        let f64t = t.float(FloatKind::F64);
        let mut a = Assigner::args(false);
        let locs: Vec<Loc> = (0..18).map(|_| a.assign(&t, f64t, true)[0].1).collect();
        assert_eq!(&locs[..8], &(10..18).map(Loc::Fpr).collect::<Vec<_>>()[..]);
        assert_eq!(&locs[8..16], &(10..18).map(Loc::Gpr).collect::<Vec<_>>()[..]);
        assert_eq!(&locs[16..], &[Loc::Stack(0), Loc::Stack(8)]);
        // A variadic double goes to an integer register.
        let mut a = Assigner::args(false);
        assert_eq!(a.assign(&t, f64t, false), vec![(Part::Whole, Loc::Gpr(10))]);
    }

    #[test]
    fn fp_aggregates_fall_back_to_the_integer_convention() {
        let mut t = TypeContext::new();
        let (f64t, i64t) = (t.float(FloatKind::F64), t.int(64));
        let fi = t.struct_(vec![f64t, i64t]);
        let mut a = Assigner::args(false);
        for _ in 0..7 {
            a.assign(&t, f64t, true);
        }
        // One FPR left: { double, long } fits (fa7 + a0).
        assert_eq!(
            a.assign(&t, fi, true),
            vec![
                (Part::Chunk { off: 0, size: 8, float: Some(64) }, Loc::Fpr(17)),
                (Part::Chunk { off: 8, size: 8, float: None }, Loc::Gpr(10)),
            ]
        );
        // None left: two integer chunks.
        assert_eq!(
            a.assign(&t, fi, true),
            vec![
                (Part::Chunk { off: 0, size: 8, float: None }, Loc::Gpr(11)),
                (Part::Chunk { off: 8, size: 8, float: None }, Loc::Gpr(12)),
            ]
        );
        // A 16-byte aggregate with one integer register left splits.
        let mut a = Assigner::args(false);
        for _ in 0..7 {
            a.assign(&t, i64t, true);
        }
        assert_eq!(
            a.assign(&t, fi, false),
            vec![
                (Part::Chunk { off: 0, size: 8, float: None }, Loc::Gpr(17)),
                (Part::Chunk { off: 8, size: 8, float: None }, Loc::Stack(0)),
            ]
        );
    }

    #[test]
    fn returns() {
        let mut t = TypeContext::new();
        let (f32t, f64t, i32t) = (t.float(FloatKind::F32), t.float(FloatKind::F64), t.int(32));
        let fi = t.struct_(vec![f32t, i32t]);
        assert_eq!(
            ret_locs(&t, fi),
            Some(vec![
                (Part::Chunk { off: 0, size: 4, float: Some(32) }, Loc::Fpr(10)),
                (Part::Chunk { off: 4, size: 4, float: None }, Loc::Gpr(10)),
            ])
        );
        let big = t.struct_(vec![f64t, f64t, f64t]);
        assert_eq!(ret_locs(&t, big), None);
        assert_eq!(ret_locs(&t, f64t), Some(vec![(Part::Whole, Loc::Fpr(10))]));
    }
}
