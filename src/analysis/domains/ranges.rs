//! The **integer interval (range)** domain: the classic *tall* lattice that
//! exercises the engine's **widening** (tenet **T4**, bet **B8**).
//!
//! # Representation & signedness convention
//!
//! An abstract value is one of:
//!
//! ```text
//!   Bottom                     γ = ∅            (empty / unreachable)
//!   Interval { width, lo, hi } γ = { v : lo ≤ signed(v) ≤ hi }   (a bounded range)
//!   Top                        γ = every value  (no information, any width)
//! ```
//!
//! Intervals are **signed and non-wrapping**: `lo`/`hi` are the *signed*
//! mathematical bounds of a `width`-bit two's-complement integer, so they live
//! in `[INT_MIN(width), INT_MAX(width)] = [-2^(w-1), 2^(w-1)-1]`. A concrete
//! [`SemValue::Int`] (held as an unsigned bit pattern in `[0, 2^w)`) is in γ iff
//! its **signed** reading lies in `[lo, hi]`. This one convention is used
//! everywhere; the only place it looks surprising is `i1`, where the two boolean
//! bit patterns `0`/`1` read as signed `0`/`-1`, so the full `i1` range is
//! `[-1, 0]`, a definite `true` is `[-1, -1]` and a definite `false` is `[0, 0]`.
//!
//! `Top` carries no width (it is the value the solver seeds parameters and
//! globals with, and the "unknown operand" the soundness harness uses), so it
//! stands for "any concrete value of whatever this SSA value's type is",
//! **including poison** — that is what lets a transfer that might produce poison
//! (a violated `nsw`/`nuw`) stay sound by returning `Top`.
//!
//! # Widening
//!
//! [`Range::widen`] is the standard interval widening: at a loop header, a bound
//! that moved *outward* since the previous iterate is thrown to the width's
//! extreme (`INT_MIN`/`INT_MAX`). Each bound can therefore change at most twice
//! before it pins to an extreme, so every ascending chain stabilizes and a loop
//! with an induction variable reaches a fixpoint (see the tests).
//!
//! # Soundness around overflow / flags
//!
//! `add`/`sub`/`mul` are computed in **exact, unbounded** `puremp::Int`
//! arithmetic. A result is emitted as a tight interval only when the exact signed
//! result provably stays in range (so no wrap and no flag violation can occur);
//! otherwise, if a set `nsw`/`nuw` flag *could* be violated the result is `Top`
//! (which γ-contains the poison the concrete semantics would produce), and if it
//! merely wraps (no flag) the result is the full range for the width. Bitwise
//! `and`/`or`/`xor` return the full range; shifts and division/remainder return
//! `Top` (their poison cases make a precise interval not worth the risk). `icmp`,
//! `trunc`/`zext`/`sext`, and constants are precise.

use crate::analysis::domain::{AbstractDomain, DomainCtx, EdgeGuard};
use crate::ir::inst::{BinOp, CastOp, Flags, InstData, InstKind, IntPred};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::Const;
use crate::ir::SemValue;

use puremp::Int;

/// An element of the signed integer interval lattice.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Range {
    /// ⊥: the empty set — undetermined or unreachable. γ = ∅.
    Bottom,
    /// A bounded signed interval `[lo, hi]` of a `width`-bit integer type.
    ///
    /// Invariant (maintained by [`interval`]): `width ≥ 1`,
    /// `INT_MIN(width) ≤ lo ≤ hi ≤ INT_MAX(width)`. An empty interval is never
    /// represented here — it canonicalizes to [`Range::Bottom`].
    Interval {
        /// The bit width of the integer type this range describes.
        width: u32,
        /// The inclusive signed lower bound.
        lo: Int,
        /// The inclusive signed upper bound.
        hi: Int,
    },
    /// ⊤: no information — any concrete value of the type (poison included).
    Top,
}

impl Range {
    /// Whether this is the ⊥ element.
    pub fn is_bottom(&self) -> bool {
        matches!(self, Range::Bottom)
    }

    /// Whether this is the ⊤ element.
    pub fn is_top(&self) -> bool {
        matches!(self, Range::Top)
    }

    /// The `(width, lo, hi)` of a bounded interval, or `None` for ⊥/⊤.
    pub fn bounds(&self) -> Option<(u32, &Int, &Int)> {
        match self {
            Range::Interval { width, lo, hi } => Some((*width, lo, hi)),
            _ => None,
        }
    }

    /// The `width`-bit interval `[lo, hi]` (canonicalized: clamped to the width's
    /// signed range, and collapsed to ⊥ if empty).
    pub fn from_bounds(width: u32, lo: Int, hi: Int) -> Range {
        interval(width, lo, hi)
    }

    /// The singleton `{value}` as a `width`-bit interval (the abstraction α of a
    /// single concrete value; `value` is interpreted modulo `2^width`).
    pub fn singleton(width: u32, value: &Int) -> Range {
        let bits = mask(value, width);
        let s = signed(&bits, width);
        interval(width, s.clone(), s)
    }

    /// The full range of a `width`-bit signed integer type
    /// (`[INT_MIN, INT_MAX]`), kept as an interval so the width is retained.
    pub fn full(width: u32) -> Range {
        interval(width, int_min(width), int_max(width))
    }
}

impl AbstractDomain for Range {
    fn bottom() -> Self {
        Range::Bottom
    }

    fn top() -> Self {
        Range::Top
    }

    fn join(&self, other: &Self) -> Self {
        use Range::{Bottom, Interval, Top};
        match (self, other) {
            (Bottom, x) | (x, Bottom) => x.clone(),
            (Top, _) | (_, Top) => Top,
            (
                Interval { width: w1, lo: l1, hi: h1 },
                Interval { width: w2, lo: l2, hi: h2 },
            ) => {
                if w1 == w2 {
                    Interval { width: *w1, lo: min_int(l1, l2), hi: max_int(h1, h2) }
                } else {
                    // Two ranges over differently-typed values have no common
                    // interval: the sound join is ⊤ (this never arises for one
                    // SSA value, whose width is fixed).
                    Top
                }
            }
        }
    }

    fn le(&self, other: &Self) -> bool {
        use Range::{Bottom, Interval, Top};
        match (self, other) {
            (Bottom, _) => true,
            (_, Top) => true,
            (Top, _) => false,
            (Interval { .. }, Bottom) => false,
            (
                Interval { width: w1, lo: l1, hi: h1 },
                Interval { width: w2, lo: l2, hi: h2 },
            ) => w1 == w2 && l2 <= l1 && h1 <= h2,
        }
    }

    fn widen(&self, next: &Self) -> Self {
        use Range::{Bottom, Interval, Top};
        match (self, next) {
            // First rise out of ⊥: adopt `next` (no bound has moved yet).
            (Bottom, x) => x.clone(),
            // `next` is the post-join iterate, so it is never ⊥ below a
            // non-⊥ `self`; keep the match total and monotone regardless.
            (_, Bottom) => self.clone(),
            (Top, _) | (_, Top) => Top,
            (
                Interval { width: w1, lo: sl, hi: sh },
                Interval { width: w2, lo: nl, hi: nh },
            ) => {
                if w1 != w2 {
                    return Top;
                }
                let w = *w1;
                // A bound that moved outward jumps to the extreme; a stable
                // bound is kept. This bounds the number of changes per bound.
                let lo = if nl < sl { int_min(w) } else { sl.clone() };
                let hi = if nh > sh { int_max(w) } else { sh.clone() };
                interval(w, lo, hi)
            }
        }
    }

    fn contains(&self, v: &SemValue) -> bool {
        match self {
            Range::Bottom => false,
            // γ(⊤) is every value of the type — poison included.
            Range::Top => true,
            Range::Interval { lo, hi, .. } => match v {
                SemValue::Int { width, bits } => {
                    let s = signed(bits, *width);
                    lo <= &s && &s <= hi
                }
                // A bounded integer interval contains no non-integer value (and
                // no poison); such a concrete result never occurs where this
                // domain returns a bounded interval, so this is sound.
                _ => false,
            },
        }
    }

    fn abstract_const(ctx: DomainCtx<'_>, c: &Const) -> Self {
        match c {
            Const::Int { ty, value } => match int_width(ctx.types, *ty) {
                Some(w) => Range::singleton(w, value),
                None => Range::Top,
            },
            // Non-integer constants are not modeled by this scalar domain.
            _ => Range::Top,
        }
    }

    fn transfer(ctx: DomainCtx<'_>, inst: &InstData, operands: &[Self]) -> Self {
        // Stateful / opaque opcodes: the produced value is unknown.
        if matches!(inst.kind, InstKind::Alloca { .. } | InstKind::Load { .. } | InstKind::Call) {
            return Range::Top;
        }
        // SCCP optimism: an undetermined operand keeps the result undetermined.
        if operands.iter().any(Range::is_bottom) {
            return Range::Bottom;
        }
        match &inst.kind {
            InstKind::Bin(op) => transfer_bin(ctx, *op, &inst.flags, inst.ty, operands),
            InstKind::ICmp(pred) => transfer_icmp(*pred, operands),
            InstKind::Cast(op) => transfer_cast(ctx, *op, inst.ty, operands),
            // Everything else (unary/fcmp/select/freeze/ptr_add/…) is not
            // modeled precisely: ⊤ is always sound.
            _ => Range::Top,
        }
    }

    fn edge_feasible(&self, guard: &EdgeGuard<'_>) -> bool {
        match guard {
            EdgeGuard::CondIs(want) => match self {
                // Unreachable / undetermined condition (SCCP optimism).
                Range::Bottom => false,
                // Unknown condition: either edge could be taken.
                Range::Top => true,
                Range::Interval { width, lo, hi } if *width == 1 => {
                    // An `i1`: `true` is bit pattern 1 (signed −1), `false` is 0.
                    let has_true = lo <= &Int::MINUS_ONE && &Int::MINUS_ONE <= hi;
                    let has_false = lo <= &Int::ZERO && &Int::ZERO <= hi;
                    if *want { has_true } else { has_false }
                }
                // A non-`i1` condition is unexpected; do not prune.
                Range::Interval { .. } => true,
            },
            // Switch guards are not refined here (default: feasible).
            _ => true,
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer helpers.
// ---------------------------------------------------------------------------

/// Transfer for a binary op. `ty` is the result type (its width also being the
/// operand width for the arithmetic/bitwise integer ops).
fn transfer_bin(
    ctx: DomainCtx<'_>,
    op: BinOp,
    flags: &Flags,
    ty: TypeId,
    operands: &[Range],
) -> Range {
    let Some(w) = int_width(ctx.types, ty) else {
        return Range::Top; // float ops (or a non-integer result): unmodeled.
    };
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul => transfer_arith(op, flags, w, operands),
        // Always defined; a precise result is not worth the effort here.
        BinOp::And | BinOp::Or | BinOp::Xor => Range::full(w),
        // Shifts and div/rem have poison paths (over-wide shift, `exact`,
        // `nsw`/`nuw`): ⊤ is the safe, simple choice.
        _ => Range::Top,
    }
}

/// Sound interval arithmetic for `add`/`sub`/`mul`, honoring `nsw`/`nuw`.
fn transfer_arith(op: BinOp, flags: &Flags, w: u32, operands: &[Range]) -> Range {
    let (Some((wa, al, ah)), Some((wb, bl, bh))) =
        (operand_interval(operands.first()), operand_interval(operands.get(1)))
    else {
        return Range::Top; // an operand is ⊤ (unknown width): result unknown.
    };
    if wa != w || wb != w {
        return Range::Top;
    }

    // Exact signed result interval (unbounded precision).
    let (slo, shi) = arith_signed(op, &al, &ah, &bl, &bh);
    // Exact unsigned result interval, from each operand's unsigned image.
    let (aul, auh) = to_unsigned(w, &al, &ah);
    let (bul, buh) = to_unsigned(w, &bl, &bh);
    let (ulo, uhi) = arith_unsigned(op, &aul, &auh, &bul, &buh);

    let s_fits = slo >= int_min(w) && shi <= int_max(w);
    let u_fits = !ulo.is_negative() && uhi <= uint_max(w);

    // A set flag that *could* be violated makes the result poison for some
    // inputs; ⊤ (which contains poison) is the sound abstraction.
    if (flags.nsw && !s_fits) || (flags.nuw && !u_fits) {
        return Range::Top;
    }

    if s_fits {
        // No signed wrap for any input, so the concrete signed result equals the
        // exact signed result: the interval is tight.
        interval(w, slo, shi)
    } else {
        // Wraps (no flag forbids it): a defined but unpredictable width-`w`
        // integer.
        Range::full(w)
    }
}

/// The exact signed result interval of `op` over signed operand bounds.
fn arith_signed(op: BinOp, al: &Int, ah: &Int, bl: &Int, bh: &Int) -> (Int, Int) {
    match op {
        BinOp::Add => (al.add(bl), ah.add(bh)),
        BinOp::Sub => (al.sub(bh), ah.sub(bl)),
        BinOp::Mul => corner_products(al, ah, bl, bh),
        _ => unreachable!("arith_signed on a non-arithmetic op"),
    }
}

/// The exact result interval of `op` over *unsigned* (non-negative) bounds.
fn arith_unsigned(op: BinOp, al: &Int, ah: &Int, bl: &Int, bh: &Int) -> (Int, Int) {
    match op {
        BinOp::Add => (al.add(bl), ah.add(bh)),
        BinOp::Sub => (al.sub(bh), ah.sub(bl)),
        // Unsigned bounds are non-negative, so the product is monotone.
        BinOp::Mul => (al.mul(bl), ah.mul(bh)),
        _ => unreachable!("arith_unsigned on a non-arithmetic op"),
    }
}

/// The `[min, max]` of the four corner products `{al,ah} × {bl,bh}`.
fn corner_products(al: &Int, ah: &Int, bl: &Int, bh: &Int) -> (Int, Int) {
    let p = [al.mul(bl), al.mul(bh), ah.mul(bl), ah.mul(bh)];
    let mut lo = p[0].clone();
    let mut hi = p[0].clone();
    for x in &p[1..] {
        lo = min_int(&lo, x);
        hi = max_int(&hi, x);
    }
    (lo, hi)
}

/// Transfer for `icmp`: an `i1` range, folded to a constant `0`/`1` when the
/// comparison is decided on the operand ranges.
fn transfer_icmp(pred: IntPred, operands: &[Range]) -> Range {
    let (Some((wa, al, ah)), Some((wb, bl, bh))) =
        (operand_interval(operands.first()), operand_interval(operands.get(1)))
    else {
        return boolean_range(None); // an operand is ⊤: result is a full `i1`.
    };
    if wa != wb {
        return boolean_range(None);
    }
    boolean_range(decide_icmp(pred, wa, &al, &ah, &bl, &bh))
}

/// Decide `pred` over signed operand bounds, if the ranges pin it down.
fn decide_icmp(pred: IntPred, w: u32, al: &Int, ah: &Int, bl: &Int, bh: &Int) -> Option<bool> {
    match pred {
        IntPred::Eq => decide_eq(al, ah, bl, bh),
        IntPred::Ne => decide_eq(al, ah, bl, bh).map(|x| !x),
        IntPred::Slt => decide_lt(al, ah, bl, bh),
        IntPred::Sle => decide_le(al, ah, bl, bh),
        IntPred::Sgt => decide_lt(bl, bh, al, ah),
        IntPred::Sge => decide_le(bl, bh, al, ah),
        IntPred::Ult | IntPred::Ule | IntPred::Ugt | IntPred::Uge => {
            let (aul, auh) = to_unsigned(w, al, ah);
            let (bul, buh) = to_unsigned(w, bl, bh);
            match pred {
                IntPred::Ult => decide_lt(&aul, &auh, &bul, &buh),
                IntPred::Ule => decide_le(&aul, &auh, &bul, &buh),
                IntPred::Ugt => decide_lt(&bul, &buh, &aul, &auh),
                IntPred::Uge => decide_le(&bul, &buh, &aul, &auh),
                _ => unreachable!("non-unsigned predicate in unsigned arm"),
            }
        }
    }
}

/// `a < b` for `a ∈ [al,ah]`, `b ∈ [bl,bh]` (over a total order).
fn decide_lt(al: &Int, ah: &Int, bl: &Int, bh: &Int) -> Option<bool> {
    if ah < bl {
        Some(true)
    } else if al >= bh {
        Some(false)
    } else {
        None
    }
}

/// `a ≤ b` for `a ∈ [al,ah]`, `b ∈ [bl,bh]`.
fn decide_le(al: &Int, ah: &Int, bl: &Int, bh: &Int) -> Option<bool> {
    if ah <= bl {
        Some(true)
    } else if al > bh {
        Some(false)
    } else {
        None
    }
}

/// `a == b`: decided true only for equal singletons, false for disjoint ranges.
fn decide_eq(al: &Int, ah: &Int, bl: &Int, bh: &Int) -> Option<bool> {
    if al == ah && bl == bh && al == bl {
        Some(true)
    } else if ah < bl || bh < al {
        Some(false)
    } else {
        None
    }
}

/// Materialize a comparison outcome as an `i1` range (see the signedness note in
/// the module docs: `true` is bit `1` = signed `-1`, `false` is `0`).
fn boolean_range(decision: Option<bool>) -> Range {
    match decision {
        Some(true) => interval(1, Int::MINUS_ONE, Int::MINUS_ONE),
        Some(false) => interval(1, Int::ZERO, Int::ZERO),
        None => interval(1, Int::MINUS_ONE, Int::ZERO),
    }
}

/// Transfer for the integer casts `trunc`/`zext`/`sext` (precise); other casts
/// are unmodeled (⊤).
fn transfer_cast(ctx: DomainCtx<'_>, op: CastOp, ty: TypeId, operands: &[Range]) -> Range {
    let Some((sw, al, ah)) = operand_interval(operands.first()) else {
        return Range::Top;
    };
    let Some(tw) = int_width(ctx.types, ty) else {
        return Range::Top;
    };
    match op {
        CastOp::Trunc => {
            if al == ah {
                // A singleton truncates exactly.
                Range::singleton(tw, &al)
            } else {
                // The low `tw` bits of a multi-value range can be anything.
                Range::full(tw)
            }
        }
        CastOp::ZExt => {
            // The zero-extended value equals the source's *unsigned* image.
            let (ul, uh) = to_unsigned(sw, &al, &ah);
            interval(tw, ul, uh)
        }
        // Sign-extension preserves the signed value.
        CastOp::SExt => interval(tw, al, ah),
        _ => Range::Top,
    }
}

/// The `(width, lo, hi)` of an operand that is a bounded interval, else `None`.
fn operand_interval(op: Option<&Range>) -> Option<(u32, Int, Int)> {
    match op {
        Some(Range::Interval { width, lo, hi }) => Some((*width, lo.clone(), hi.clone())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Numeric helpers (all exact, all width-parameterized).
// ---------------------------------------------------------------------------

/// The `width`-bit interval `[lo, hi]`, clamped to the signed range and collapsed
/// to ⊥ when empty. `width == 0` (no real value) is treated as ⊤.
fn interval(width: u32, lo: Int, hi: Int) -> Range {
    if width == 0 {
        return Range::Top;
    }
    let (imin, imax) = (int_min(width), int_max(width));
    let lo = max_int(&lo, &imin);
    let hi = min_int(&hi, &imax);
    if lo > hi {
        Range::Bottom
    } else {
        Range::Interval { width, lo, hi }
    }
}

/// The bit width of an integer type, or `None` if `ty` is not an integer type.
fn int_width(types: &TypeContext, ty: TypeId) -> Option<u32> {
    match types.get(ty) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}

/// `2^n` as a non-negative [`Int`].
fn two_pow(n: u32) -> Int {
    Int::ONE.mul_2k(n)
}

/// The most negative signed `width`-bit value, `-2^(width-1)`.
fn int_min(width: u32) -> Int {
    two_pow(width - 1).neg()
}

/// The most positive signed `width`-bit value, `2^(width-1) - 1`.
fn int_max(width: u32) -> Int {
    two_pow(width - 1).sub(&Int::ONE)
}

/// The largest unsigned `width`-bit value, `2^width - 1`.
fn uint_max(width: u32) -> Int {
    two_pow(width).sub(&Int::ONE)
}

/// The unsigned bit pattern of `v` in a `width`-bit type: the Euclidean low
/// `width` bits, in `[0, 2^width)` (exactly the two's-complement encoding).
fn mask(v: &Int, width: u32) -> Int {
    if width == 0 { Int::ZERO } else { v.mod_2k(width) }
}

/// The signed value of a `width`-bit pattern `bits` (assumed in `[0, 2^width)`).
fn signed(bits: &Int, width: u32) -> Int {
    if width > 0 && bits.bit(width - 1) { bits.sub(&two_pow(width)) } else { bits.clone() }
}

/// The unsigned image `[ulo, uhi]` of the signed interval `[lo, hi]`. When the
/// interval straddles zero the image wraps, so it is over-approximated to the
/// full unsigned range `[0, 2^width - 1]`.
fn to_unsigned(width: u32, lo: &Int, hi: &Int) -> (Int, Int) {
    if !lo.is_negative() {
        // Entirely non-negative: unsigned == signed.
        (lo.clone(), hi.clone())
    } else if hi.is_negative() {
        // Entirely negative: shift by 2^width to the high half.
        let m = two_pow(width);
        (lo.add(&m), hi.add(&m))
    } else {
        // Straddles zero: the unsigned image is not contiguous.
        (Int::ZERO, uint_max(width))
    }
}

/// The smaller of two integers.
fn min_int(a: &Int, b: &Int) -> Int {
    if a <= b { a.clone() } else { b.clone() }
}

/// The larger of two integers.
fn max_int(a: &Int, b: &Int) -> Int {
    if a >= b { a.clone() } else { b.clone() }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::Range;

    use crate::analysis::domain::{AbstractDomain, DomainCtx, EdgeGuard};
    use crate::analysis::soundness::check_integer_transfer_sound;
    use crate::analysis::solver::solve;
    use crate::ir::inst::{BinOp, CastOp, Flags, InstData, InstKind, IntPred};
    use crate::ir::types::TypeContext;
    use crate::ir::value::ValueId;
    use crate::ir::{Module, SemValue};
    use crate::support::StrInterner;

    use puremp::Int;

    // -- small helpers ------------------------------------------------------

    fn i(v: i64) -> Int {
        Int::from_i64(v)
    }

    /// A width-32 interval `[lo, hi]`.
    fn r32(lo: i64, hi: i64) -> Range {
        Range::from_bounds(32, i(lo), i(hi))
    }

    /// Build a bare value-producing instruction (operands are supplied to
    /// `transfer` separately, as the solver does).
    fn inst(kind: InstKind, flags: Flags, ty: crate::ir::TypeId) -> InstData {
        InstData { kind, flags, ty, operands: Vec::new(), result: None }
    }

    /// A fixed sample of ranges (mixed widths, plus ⊥/⊤) for the lattice laws.
    fn sample() -> Vec<Range> {
        vec![
            Range::Bottom,
            Range::Top,
            r32(0, 0),
            r32(5, 5),
            r32(-3, 7),
            r32(0, 100),
            r32(-50, -10),
            Range::full(32),
            Range::from_bounds(8, i(-1), i(1)),
            Range::from_bounds(8, i(0), i(127)),
        ]
    }

    // -- lattice laws -------------------------------------------------------

    #[test]
    fn lattice_laws() {
        let elems = sample();
        for a in &elems {
            assert_eq!(a.join(a), *a, "join idempotent");
            assert_eq!(a.join(&Range::Bottom), *a, "bottom is join identity");
            assert_eq!(a.join(&Range::Top), Range::Top, "top absorbs");
            assert!(a.le(a), "le reflexive");
            assert!(Range::Bottom.le(a), "bottom is least");
            assert!(a.le(&Range::Top), "top is greatest");
            for b in &elems {
                assert_eq!(a.join(b), b.join(a), "join commutative");
                assert_eq!(a.le(b), &a.join(b) == b, "le consistent with join");
                // The widened value is an upper bound of both.
                let w = a.widen(b);
                assert!(a.le(&w), "widen ⊒ self");
                assert!(b.le(&w), "widen ⊒ next");
                for c in &elems {
                    assert_eq!(a.join(b).join(c), a.join(&b.join(c)), "join associative");
                }
            }
        }
    }

    #[test]
    fn gamma_contains() {
        let r = r32(-3, 7);
        assert!(r.contains(&SemValue::int(32, i(-3))));
        assert!(r.contains(&SemValue::int(32, i(0))));
        assert!(r.contains(&SemValue::int(32, i(7))));
        assert!(!r.contains(&SemValue::int(32, i(8))));
        assert!(!r.contains(&SemValue::int(32, i(-4))));
        // −1 is the bit pattern 0xFFFF_FFFF; its signed reading is −3..7's −1.
        assert!(r.contains(&SemValue::int(32, Int::MINUS_ONE)));
        assert!(Range::Top.contains(&SemValue::int(32, i(123))));
        assert!(Range::Top.contains(&SemValue::Poison), "top contains poison");
        assert!(!Range::Bottom.contains(&SemValue::int(32, i(0))));
        assert!(!r.contains(&SemValue::Poison), "an interval excludes poison");
    }

    // -- soundness harness (the key gate) ----------------------------------

    #[test]
    fn transfer_is_sound_on_random_inputs() {
        let report = check_integer_transfer_sound::<Range>(20_000, 0x0BAD_C0DE);
        assert!(report.is_sound(), "soundness violations: {:?}", report.violations);
        assert!(report.checked > 1000, "harness must exercise many cases: {report:?}");

        // Several more seeds, thousands of cases each, must all be sound.
        for seed in [0x1u64, 0xDEAD_BEEF, 0x1234_5678, 0xFACE_FEED, 0xA5A5_A5A5] {
            let rep = check_integer_transfer_sound::<Range>(5_000, seed);
            assert!(rep.is_sound(), "seed {seed:#x} violations: {:?}", rep.violations);
        }
    }

    // -- precision spot-checks ---------------------------------------------

    #[test]
    fn add_of_two_singletons_is_the_singleton_sum() {
        let mut types = TypeContext::new();
        let i32t = types.int(32);
        let ctx = DomainCtx::new(&types);
        let add = inst(InstKind::Bin(BinOp::Add), Flags::NONE, i32t);
        let r = Range::transfer(ctx, &add, &[Range::singleton(32, &i(5)), Range::singleton(32, &i(7))]);
        assert_eq!(r, Range::singleton(32, &i(12)));

        // sub too.
        let sub = inst(InstKind::Bin(BinOp::Sub), Flags::NONE, i32t);
        let rs = Range::transfer(ctx, &sub, &[Range::singleton(32, &i(5)), Range::singleton(32, &i(7))]);
        assert_eq!(rs, Range::singleton(32, &i(-2)));
    }

    #[test]
    fn add_of_ranges_is_the_hull_sum() {
        let mut types = TypeContext::new();
        let i32t = types.int(32);
        let ctx = DomainCtx::new(&types);
        let add = inst(InstKind::Bin(BinOp::Add), Flags::NONE, i32t);
        let r = Range::transfer(ctx, &add, &[r32(0, 10), r32(100, 200)]);
        assert_eq!(r, r32(100, 210));
    }

    #[test]
    fn nsw_add_that_can_overflow_is_top() {
        let mut types = TypeContext::new();
        let i8t = types.int(8);
        let ctx = DomainCtx::new(&types);
        let add = inst(InstKind::Bin(BinOp::Add), Flags::nsw(), i8t);
        // 100 + 100 overflows i8's signed range ⇒ poison possible ⇒ ⊤.
        let a = Range::singleton(8, &i(100));
        let r = Range::transfer(ctx, &add, &[a.clone(), a]);
        assert!(r.is_top(), "nsw overflow must widen to ⊤, got {r:?}");
    }

    #[test]
    fn plain_add_that_wraps_is_full_not_top() {
        let mut types = TypeContext::new();
        let i8t = types.int(8);
        let ctx = DomainCtx::new(&types);
        let add = inst(InstKind::Bin(BinOp::Add), Flags::NONE, i8t);
        let a = Range::singleton(8, &i(100));
        let r = Range::transfer(ctx, &add, &[a.clone(), a]);
        // No flag: the sum wraps but is defined ⇒ full i8 range, not ⊤.
        assert_eq!(r, Range::full(8));
    }

    #[test]
    fn icmp_on_disjoint_ranges_folds_to_a_constant() {
        let mut types = TypeContext::new();
        let i1t = types.int(1);
        let ctx = DomainCtx::new(&types);

        let slt = inst(InstKind::ICmp(IntPred::Slt), Flags::NONE, i1t);
        let t = Range::transfer(ctx, &slt, &[r32(0, 5), r32(10, 20)]);
        // Definitely true: the i1 constant 1 (signed −1).
        assert_eq!(t, Range::from_bounds(1, Int::MINUS_ONE, Int::MINUS_ONE));
        assert!(t.contains(&SemValue::boolean(true)));
        assert!(!t.contains(&SemValue::boolean(false)));

        // Definitely false the other way.
        let f = Range::transfer(ctx, &slt, &[r32(10, 20), r32(0, 5)]);
        assert_eq!(f, Range::from_bounds(1, Int::ZERO, Int::ZERO));
        assert!(f.contains(&SemValue::boolean(false)));

        // Overlapping ⇒ full i1.
        let u = Range::transfer(ctx, &slt, &[r32(0, 15), r32(10, 20)]);
        assert_eq!(u, Range::from_bounds(1, Int::MINUS_ONE, Int::ZERO));
    }

    #[test]
    fn casts_are_precise() {
        let mut types = TypeContext::new();
        let i16t = types.int(16);
        let i64t = types.int(64);
        let ctx = DomainCtx::new(&types);

        // zext of a non-negative i32 range keeps the bounds.
        let zext = inst(InstKind::Cast(CastOp::ZExt), Flags::NONE, i64t);
        let z = Range::transfer(ctx, &zext, &[r32(0, 1000)]);
        assert_eq!(z, Range::from_bounds(64, i(0), i(1000)));

        // sext preserves the signed range.
        let sext = inst(InstKind::Cast(CastOp::SExt), Flags::NONE, i64t);
        let s = Range::transfer(ctx, &sext, &[r32(-5, 5)]);
        assert_eq!(s, Range::from_bounds(64, i(-5), i(5)));

        // trunc of a singleton is exact.
        let trunc = inst(InstKind::Cast(CastOp::Trunc), Flags::NONE, i16t);
        let tr = Range::transfer(ctx, &trunc, &[Range::singleton(32, &i(0x1_2345))]);
        assert_eq!(tr, Range::singleton(16, &i(0x2345)));
    }

    #[test]
    fn edge_feasible_refines_a_known_i1() {
        let t = Range::from_bounds(1, Int::MINUS_ONE, Int::MINUS_ONE); // definitely true
        assert!(t.edge_feasible(&EdgeGuard::CondIs(true)));
        assert!(!t.edge_feasible(&EdgeGuard::CondIs(false)));

        let f = Range::from_bounds(1, Int::ZERO, Int::ZERO); // definitely false
        assert!(!f.edge_feasible(&EdgeGuard::CondIs(true)));
        assert!(f.edge_feasible(&EdgeGuard::CondIs(false)));

        let unknown = Range::from_bounds(1, Int::MINUS_ONE, Int::ZERO);
        assert!(unknown.edge_feasible(&EdgeGuard::CondIs(true)));
        assert!(unknown.edge_feasible(&EdgeGuard::CondIs(false)));
    }

    // -- solver integration: a parameter is ⊤ ------------------------------

    #[test]
    fn parameter_is_top() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("param");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);

        let x;
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            x = b.param(entry, 0);
            let one = b.const_i64(i32t, 1);
            let r = b.add(x, one, Flags::NONE);
            b.ret(Some(r));
        }
        let res = solve::<Range>(m.function(f), m.types(), m.consts());
        assert!(res.value(x).is_top(), "a function parameter must be ⊤");
    }

    // -- widening termination on a counting loop ---------------------------

    #[test]
    fn widening_terminates_a_counting_loop() {
        // f() -> i32:
        //   entry: br header(0)
        //   header(i): cond = i <s 1_000_000; cond_br cond, body(i), exit(i)
        //   body(i):   i' = i + 1; br header(i')      [back edge]
        //   exit(i):   ret i
        //
        // Under a plain join the header's `i` would climb 0,1,2,… forever;
        // widening lifts it to the full i32 range so the fixpoint terminates.
        let mut syms = StrInterner::new();
        let mut m = Module::new("loop");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("count"), sig);

        let header;
        let header_i: ValueId;
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            header = b.create_block(&[i32t]);
            let body = b.create_block(&[i32t]);
            let exit = b.create_block(&[i32t]);

            b.switch_to(entry);
            let zero = b.const_i64(i32t, 0);
            b.br(header, &[zero]);

            b.switch_to(header);
            header_i = b.param(header, 0);
            let bound = b.const_i64(i32t, 1_000_000);
            let cond = b.icmp(IntPred::Slt, header_i, bound);
            b.cond_br(cond, body, &[header_i], exit, &[header_i]);

            b.switch_to(body);
            let bi = b.param(body, 0);
            let one = b.const_i64(i32t, 1);
            let next = b.add(bi, one, Flags::NONE);
            b.br(header, &[next]);

            b.switch_to(exit);
            let ev = b.param(exit, 0);
            b.ret(Some(ev));
        }

        // The essential property: `solve` returns at all (it terminates).
        let r = solve::<Range>(m.function(f), m.types(), m.consts());
        assert!(r.is_reachable(header), "header must be reachable");

        // The inferred induction-variable range must be sound: it contains every
        // value the loop actually takes (0..=1_000_000).
        let iv = r.value(header_i);
        for v in [0_i64, 1, 500_000, 999_999, 1_000_000] {
            assert!(
                iv.contains(&SemValue::int(32, i(v))),
                "IV range {iv:?} must contain reachable value {v}",
            );
        }
        // Widening drove it to the full i32 range.
        assert_eq!(*iv, Range::full(32), "widening should lift the IV to the full range");
    }
}

