//! The **known-bits** abstract domain: for an `N`-bit integer, track which bits
//! are known to be `0` and which are known to be `1` (à la LLVM's `KnownBits`,
//! but our own clean-room construction on the one lattice engine — tenet **T4**,
//! bet **B8**).
//!
//! # Representation and invariant
//!
//! A non-trivial element is [`KnownBits::Bits`], carrying the bit width and two
//! width-`N` masks held as non-negative [`puremp::Int`] patterns in `[0, 2ⁿ)`:
//!
//! - `zeros` — the bits known to be `0`;
//! - `ones` — the bits known to be `1`.
//!
//! The **invariant** is `zeros & ones == 0`: a bit is never simultaneously known
//! `0` and known `1`. Every constructor and transfer here preserves it (a would-be
//! contradiction is widened to [`KnownBits::Top`], never smuggled through). Two
//! extra points of the lattice sit outside the mask representation:
//!
//! - [`KnownBits::Bottom`] — the empty/unreachable element (γ = ∅). It is distinct
//!   because a `zeros & ones != 0` pattern would violate the invariant, so
//!   "contradiction" gets its own tag.
//! - [`KnownBits::Top`] — no information (every bit unknown), width-agnostic. It is
//!   what [`AbstractDomain::top`] returns (a fresh function parameter, an opaque
//!   pointer, …) where no width is yet known. A *width-known* "all unknown" value
//!   is representable as `Bits { width, zeros: 0, ones: 0 }` and is used internally
//!   so casts/shifts can still place known bits around an unknown core.
//!
//! # The lattice: `join` / `le`
//!
//! `join` is the **meet toward less-known**: the result knows a bit only if *both*
//! inputs agree on it, so `zeros = a.zeros & b.zeros` and `ones = a.ones & b.ones`;
//! `Bottom` is the identity (joining with it yields the other side) and `Top`
//! absorbs. The order is `a ⊑ b` iff `a` knows at least the bits `b` knows, with
//! the same values — i.e. `b.zeros ⊆ a.zeros && b.ones ⊆ a.ones`. This is exactly
//! consistent with `join`: `a ⊑ b ⇔ a.join(b) == b`.
//!
//! # Widening
//!
//! The lattice has **finite height** `N` (each join can only *remove* known bits,
//! and there are at most `N` of them), so any ascending chain stabilizes in at
//! most `N` steps. No special widening is needed: `widen` is left as its default
//! (`join`), which already terminates.
//!
//! # Precision
//!
//! Transfers are **bit-precise** on the tractable operations:
//!
//! - `and` / `or` / `xor` (and hence `not`, which the IR spells `xor x, -1`):
//!   exact bit logic.
//! - `shl` / `lshr` / `ashr` by a *fully known* amount `< width`: the masks are
//!   shifted (`lshr` fills known-zero high bits; `ashr` replicates a known sign
//!   bit; `shl` fills known-zero low bits).
//! - `add` / `sub` (no `nsw`/`nuw`): a carry-aware bit-serial transfer — a low run
//!   of known bits propagates through the carry, and a bit whose carry cannot be
//!   pinned down is left unknown.
//! - `mul` (no `nsw`/`nuw`): exact when both operands are fully known; otherwise
//!   the product's trailing known-zero count (`tz(a) + tz(b)`).
//! - `zext` / `trunc` / `sext`: extend/truncate the masks (`zext` zero-fills the
//!   high bits; `sext` replicates a known sign bit).
//! - `icmp`: exact when both operands are fully known; `eq`/`ne` are also decided
//!   whenever the operands' known bits *conflict* (differ on a known bit).
//!
//! Everything else — `nsw`/`nuw`/`exact`-flagged ops (a flag violation is *poison*
//! concretely, and poison is γ-contained only by `Top`), `div`/`rem`, float ops,
//! `select`, `freeze`, `ptr_add`, pointer/float casts, and every stateful/opaque
//! opcode (`alloca`/`load`/`call`) — conservatively yields [`KnownBits::Top`].
//!
//! ## Non-relational limitation
//!
//! This is a **non-relational** domain: it abstracts each SSA value independently
//! and does *not* track correlations between values. In particular `xor x, x` of
//! two *independent* `Top` operands is `Top`, **not** the constant `0` — the domain
//! has no way to know the two operand values are the same SSA value. (Constant
//! folding / value numbering, not this domain, is what recovers `xor x, x == 0`.)

use crate::analysis::domain::{AbstractDomain, DomainCtx};
use crate::ir::SemValue;
use crate::ir::inst::{BinOp, CastOp, InstData, InstKind, IntPred};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::Const;

use puremp::Int;

/// An element of the known-bits lattice (see the module docs for the
/// representation, invariant, and lattice direction).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum KnownBits {
    /// ⊥: unreachable / undetermined. γ = ∅.
    Bottom,
    /// ⊤: no information (every bit unknown), width-agnostic. γ = every value.
    Top,
    /// Known bits of a `width`-bit integer. Invariant: `zeros & ones == 0`, and
    /// both masks lie in `[0, 2^width)`.
    Bits {
        /// The bit width `N`.
        width: u32,
        /// The bits known to be `0`.
        zeros: Int,
        /// The bits known to be `1`.
        ones: Int,
    },
}

impl KnownBits {
    /// Whether this is the ⊥ element.
    pub fn is_bottom(&self) -> bool {
        matches!(self, KnownBits::Bottom)
    }

    /// Whether this is the ⊤ element.
    pub fn is_top(&self) -> bool {
        matches!(self, KnownBits::Top)
    }
}

// ---------------------------------------------------------------------------
// Small mask helpers (all width-parameterized, all producing non-negative
// patterns in `[0, 2^width)`).
// ---------------------------------------------------------------------------

/// The non-negative low `w` bits of `v` (its unsigned `w`-bit pattern).
fn mask(v: &Int, w: u32) -> Int {
    v.mod_2k(w)
}

/// The all-ones `w`-bit pattern `2^w - 1`.
fn full(w: u32) -> Int {
    mask(&Int::MINUS_ONE, w)
}

/// The low-`k`-bits mask `2^k - 1`.
fn low_ones(k: u32) -> Int {
    mask(&Int::MINUS_ONE, k)
}

/// The high-bits mask covering positions `[w - k, w)` (the top `k` bits of a
/// `w`-bit value); empty when `k == 0`.
fn high_bits(w: u32, k: u32) -> Int {
    full(w).bitxor(&low_ones(w - k))
}

/// The bits set in positions `[src, dst)` — the region a `src → dst` widening
/// cast fills (requires `src <= dst`).
fn widen_region(src: u32, dst: u32) -> Int {
    full(dst).bitxor(&low_ones(src))
}

/// Set bit `i` of `acc`.
fn set_bit(acc: &mut Int, i: u32) {
    *acc = acc.bitor(&Int::ONE.mul_2k(i));
}

/// The integer type width of `ty`, or `None` if it is not an integer type.
fn int_width(types: &TypeContext, ty: TypeId) -> Option<u32> {
    match types.get(ty) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}

/// Construct a [`KnownBits::Bits`], normalizing both masks to `width` and
/// upholding the `zeros & ones == 0` invariant.
///
/// A contradiction (`zeros & ones != 0`) can only arise from a bug in a transfer,
/// never from a sound derivation; should one ever appear it is widened to
/// [`KnownBits::Top`] (which γ-contains everything, so the result stays sound)
/// rather than to an unsound `Bottom`.
fn make(width: u32, zeros: Int, ones: Int) -> KnownBits {
    let z = mask(&zeros, width);
    let o = mask(&ones, width);
    if !z.bitand(&o).is_zero() {
        return KnownBits::Top;
    }
    KnownBits::Bits { width, zeros: z, ones: o }
}

/// The fully-known element denoting exactly the single `w`-bit value `val`.
fn known_const(w: u32, val: &Int) -> KnownBits {
    let ones = mask(val, w);
    let zeros = full(w).bitxor(&ones);
    make(w, zeros, ones)
}

/// The known-bits element for an `i1` boolean.
fn known_bool(b: bool) -> KnownBits {
    known_const(1, if b { &Int::ONE } else { &Int::ZERO })
}

/// The `(zeros, ones)` masks of `op` coerced to width `w`: a `Bits` of matching
/// width passes through; a `Top` (or a width mismatch) contributes no known bits.
/// `Bottom` never reaches here (the caller filters it first).
fn operand_masks(op: &KnownBits, w: u32) -> (Int, Int) {
    match op {
        KnownBits::Bits { width, zeros, ones } if *width == w => (zeros.clone(), ones.clone()),
        _ => (Int::ZERO, Int::ZERO),
    }
}

/// Whether the masks pin down *every* bit of a `w`-bit value.
fn is_fully_known(w: u32, zeros: &Int, ones: &Int) -> bool {
    zeros.bitor(ones) == full(w)
}

/// The count of trailing (low) bits known to be `0`.
fn trailing_known_zeros(w: u32, zeros: &Int) -> u32 {
    let mut t = 0;
    while t < w && zeros.bit(t) {
        t += 1;
    }
    t
}

/// The signed value of a `w`-bit pattern `bits` (which lies in `[0, 2^w)`).
fn signed(bits: &Int, w: u32) -> Int {
    if w > 0 && bits.bit(w - 1) { bits.sub(&Int::ONE.mul_2k(w)) } else { bits.clone() }
}

impl AbstractDomain for KnownBits {
    fn bottom() -> Self {
        KnownBits::Bottom
    }

    fn top() -> Self {
        KnownBits::Top
    }

    fn join(&self, other: &Self) -> Self {
        use KnownBits::{Bits, Bottom, Top};
        match (self, other) {
            (Bottom, x) | (x, Bottom) => x.clone(),
            (Top, _) | (_, Top) => Top,
            (
                Bits { width: w1, zeros: z1, ones: o1 },
                Bits { width: w2, zeros: z2, ones: o2 },
            ) => {
                if w1 == w2 {
                    // Keep only bits both sides agree on: the meet toward less-known.
                    make(*w1, z1.bitand(z2), o1.bitand(o2))
                } else {
                    // Different widths never describe the same value; ⊤ is the safe
                    // upper bound.
                    Top
                }
            }
        }
    }

    fn le(&self, other: &Self) -> bool {
        use KnownBits::{Bits, Bottom, Top};
        match (self, other) {
            // ⊥ is below everything; everything is below ⊤.
            (Bottom, _) => true,
            (_, Top) => true,
            // ⊤ knows nothing, so it is below only ⊤ (handled above).
            (Top, _) => false,
            // Only ⊥ is below ⊥ (handled above).
            (_, Bottom) => false,
            (
                Bits { width: w1, zeros: z1, ones: o1 },
                Bits { width: w2, zeros: z2, ones: o2 },
            ) => {
                // a ⊑ b iff b's known bits are a subset of a's (same values):
                // z2 ⊆ z1 and o2 ⊆ o1.
                w1 == w2 && &z2.bitand(z1) == z2 && &o2.bitand(o1) == o2
            }
        }
    }

    // `widen` is the default (`join`): the lattice has finite height `N`, so every
    // ascending chain stabilizes without a widening step (see the module docs).

    fn contains(&self, v: &SemValue) -> bool {
        match self {
            KnownBits::Bottom => false,
            KnownBits::Top => true,
            KnownBits::Bits { zeros, ones, .. } => match v {
                SemValue::Int { bits, .. } => {
                    // Every known-zero bit is clear in `bits`, and every known-one
                    // bit is set in `bits`.
                    bits.bitand(zeros).is_zero() && &ones.bitand(bits) == ones
                }
                // Poison / non-integer values are γ-contained only by ⊤.
                _ => false,
            },
        }
    }

    fn abstract_const(ctx: DomainCtx<'_>, c: &Const) -> Self {
        if let Const::Int { ty, value } = c
            && let Some(w) = int_width(ctx.types, *ty)
        {
            known_const(w, value)
        } else {
            // Poison / float / null / aggregate: not modeled by the integer
            // known-bits lattice, so ⊤ (sound over-approximation).
            KnownBits::Top
        }
    }

    fn transfer(ctx: DomainCtx<'_>, inst: &InstData, operands: &[Self]) -> Self {
        // Stateful / opaque opcodes produce a value the domain cannot know.
        if matches!(
            inst.kind,
            InstKind::Alloca { .. }
                | InstKind::DynAlloca { .. }
                | InstKind::Load { .. }
                | InstKind::Call
        ) {
            return KnownBits::Top;
        }
        // An undetermined operand keeps the result undetermined (SCCP optimism).
        if operands.iter().any(KnownBits::is_bottom) {
            return KnownBits::Bottom;
        }
        match &inst.kind {
            InstKind::Bin(op) => transfer_bin(ctx, inst, *op, operands),
            InstKind::ICmp(pred) => transfer_icmp(*pred, operands),
            InstKind::Cast(op) => transfer_cast(ctx, inst, *op, operands),
            // select / freeze / fcmp / unary(float) / ptr_add and anything else:
            // conservatively ⊤.
            _ => KnownBits::Top,
        }
    }

    // `edge_feasible` keeps its default (every edge feasible), which is always
    // sound; refining branches on a known-bits condition is a future precision
    // improvement.
}

// ---------------------------------------------------------------------------
// Transfer: binary ops.
// ---------------------------------------------------------------------------

fn transfer_bin(
    ctx: DomainCtx<'_>,
    inst: &InstData,
    op: BinOp,
    operands: &[KnownBits],
) -> KnownBits {
    let Some(w) = int_width(ctx.types, inst.ty) else {
        // Float result (FAdd/FSub/...) or otherwise non-integer: not modeled.
        return KnownBits::Top;
    };
    let (Some(a), Some(b)) = (operands.first(), operands.get(1)) else {
        return KnownBits::Top;
    };
    let (az, ao) = operand_masks(a, w);
    let (bz, bo) = operand_masks(b, w);
    let f = &inst.flags;

    match op {
        // --- exact bit logic (never poison; flags are irrelevant) ---
        BinOp::And => make(w, az.bitor(&bz), ao.bitand(&bo)),
        BinOp::Or => make(w, az.bitand(&bz), ao.bitor(&bo)),
        BinOp::Xor => make(
            w,
            // xor == 0 where both bits are known and equal.
            ao.bitand(&bo).bitor(&az.bitand(&bz)),
            // xor == 1 where both bits are known and differ.
            ao.bitand(&bz).bitor(&az.bitand(&bo)),
        ),

        // --- carry-aware add / sub (only without nsw/nuw: a wrap under a set
        //     flag is poison, which only ⊤ γ-contains) ---
        BinOp::Add if !f.nsw && !f.nuw => add_with_carry(w, &az, &ao, &bz, &bo, false),
        // a - b == a + ~b + 1: flip b's masks (zeros<->ones) and carry in a 1.
        BinOp::Sub if !f.nsw && !f.nuw => add_with_carry(w, &az, &ao, &bo, &bz, true),

        // --- multiply (only without nsw/nuw) ---
        BinOp::Mul if !f.nsw && !f.nuw => mul_known(w, &az, &ao, &bz, &bo),

        // --- shifts by a fully known amount (< width) ---
        BinOp::Shl if !f.nsw && !f.nuw => shl_known(w, &az, &ao, &bz, &bo),
        // `exact` on a right shift is poison if a set bit is shifted out: ⊤.
        BinOp::LShr if !f.exact => shr_known(w, &az, &ao, &bz, &bo, false),
        BinOp::AShr if !f.exact => shr_known(w, &az, &ao, &bz, &bo, true),

        // Flagged arithmetic, div/rem, and float ops: conservatively ⊤.
        _ => KnownBits::Top,
    }
}

/// Carry-aware known-bits addition `a + b + carry_in`. Iterates LSB→MSB tracking
/// whether the running carry is known, propagating known sum bits and pinning the
/// carry-out whenever a majority of `{aᵢ, bᵢ, carry}` is known.
fn add_with_carry(w: u32, az: &Int, ao: &Int, bz: &Int, bo: &Int, carry_in: bool) -> KnownBits {
    let mut zeros = Int::ZERO;
    let mut ones = Int::ZERO;
    let mut carry_known = true;
    let mut carry_val = carry_in;

    for i in 0..w {
        let a1 = ao.bit(i); // a known 1
        let a0 = az.bit(i); // a known 0
        let b1 = bo.bit(i);
        let b0 = bz.bit(i);
        let c1 = carry_known && carry_val; // carry known 1
        let c0 = carry_known && !carry_val; // carry known 0

        // The sum bit is known iff both operand bits and the carry are known.
        if (a0 || a1) && (b0 || b1) && carry_known {
            let s = (a1 ^ b1) ^ carry_val;
            if s {
                set_bit(&mut ones, i);
            } else {
                set_bit(&mut zeros, i);
            }
        }

        // carry_out = majority(aᵢ, bᵢ, carry): known 1 if ≥2 inputs are known 1,
        // known 0 if ≥2 are known 0, otherwise unknown.
        let ones_ct = u32::from(a1) + u32::from(b1) + u32::from(c1);
        let zeros_ct = u32::from(a0) + u32::from(b0) + u32::from(c0);
        if zeros_ct >= 2 {
            carry_known = true;
            carry_val = false;
        } else if ones_ct >= 2 {
            carry_known = true;
            carry_val = true;
        } else {
            carry_known = false;
            carry_val = false;
        }
    }

    make(w, zeros, ones)
}

/// Known bits of `a * b`: exact when both operands are fully known, otherwise the
/// product's guaranteed trailing-zero run `tz(a) + tz(b)`.
fn mul_known(w: u32, az: &Int, ao: &Int, bz: &Int, bo: &Int) -> KnownBits {
    if is_fully_known(w, az, ao) && is_fully_known(w, bz, bo) {
        // `ao` / `bo` are the concrete unsigned values (all bits known).
        return known_const(w, &ao.mul(bo));
    }
    let tz = (trailing_known_zeros(w, az) + trailing_known_zeros(w, bz)).min(w);
    // The low `tz` bits of the product are known 0; nothing else is guaranteed.
    make(w, low_ones(tz), Int::ZERO)
}

/// Known bits of `a << b` when `b` is a fully known amount `< width`; otherwise ⊤
/// (an unknown or `≥ width` amount could be poison, which only ⊤ γ-contains).
fn shl_known(w: u32, az: &Int, ao: &Int, bz: &Int, bo: &Int) -> KnownBits {
    let Some(k) = known_shift_amount(w, bz, bo) else {
        return KnownBits::Top;
    };
    // Shift both masks left; the vacated low `k` bits are known 0.
    let zeros = az.mul_2k(k).bitor(&low_ones(k));
    let ones = ao.mul_2k(k);
    make(w, zeros, ones)
}

/// Known bits of a right shift of `a` by a fully known amount `b < width`
/// (`arithmetic` selects `ashr` vs. `lshr`); otherwise ⊤.
fn shr_known(w: u32, az: &Int, ao: &Int, bz: &Int, bo: &Int, arithmetic: bool) -> KnownBits {
    let Some(k) = known_shift_amount(w, bz, bo) else {
        return KnownBits::Top;
    };
    let mut zeros = az.div_2k_trunc(k);
    let mut ones = ao.div_2k_trunc(k);
    if arithmetic {
        // Arithmetic shift replicates the sign bit (bit w-1) into the top `k` bits
        // — but only if that sign bit is itself known.
        if w > 0 && ao.bit(w - 1) {
            ones = ones.bitor(&high_bits(w, k));
        } else if w > 0 && az.bit(w - 1) {
            zeros = zeros.bitor(&high_bits(w, k));
        }
    } else {
        // Logical shift zero-fills: the top `k` bits are known 0.
        zeros = zeros.bitor(&high_bits(w, k));
    }
    make(w, zeros, ones)
}

/// A fully known shift amount strictly `< w`, or `None` when the amount is not
/// fully known or is `≥ w` (a `≥ width` shift is poison — the caller returns ⊤).
fn known_shift_amount(w: u32, bz: &Int, bo: &Int) -> Option<u32> {
    if !is_fully_known(w, bz, bo) {
        return None;
    }
    let k = bo.to_u64()?;
    (k < u64::from(w)).then_some(k as u32)
}

// ---------------------------------------------------------------------------
// Transfer: integer comparison.
// ---------------------------------------------------------------------------

fn transfer_icmp(pred: IntPred, operands: &[KnownBits]) -> KnownBits {
    let (Some(a), Some(b)) = (operands.first(), operands.get(1)) else {
        return KnownBits::Top;
    };
    let (
        KnownBits::Bits { width: wa, zeros: za, ones: oa },
        KnownBits::Bits { width: wb, zeros: zb, ones: ob },
    ) = (a, b)
    else {
        // A ⊤ (unknown-width) operand: cannot decide.
        return KnownBits::Top;
    };
    if wa != wb {
        return KnownBits::Top;
    }
    let w = *wa;

    // Fully known on both sides: compute the comparison exactly.
    if is_fully_known(w, za, oa) && is_fully_known(w, zb, ob) {
        return known_bool(eval_icmp_known(pred, w, oa, ob));
    }

    // Otherwise, only equality is decidable — and only when the known bits
    // *conflict* (one side knows 0 where the other knows 1), forcing inequality.
    let conflict = !oa.bitand(zb).is_zero() || !za.bitand(ob).is_zero();
    match pred {
        IntPred::Eq if conflict => known_bool(false),
        IntPred::Ne if conflict => known_bool(true),
        _ => KnownBits::Top,
    }
}

/// The exact `icmp` result on two fully known `w`-bit unsigned patterns.
fn eval_icmp_known(pred: IntPred, w: u32, a: &Int, b: &Int) -> bool {
    match pred {
        IntPred::Eq => a == b,
        IntPred::Ne => a != b,
        IntPred::Ugt => a > b,
        IntPred::Uge => a >= b,
        IntPred::Ult => a < b,
        IntPred::Ule => a <= b,
        IntPred::Sgt => signed(a, w) > signed(b, w),
        IntPred::Sge => signed(a, w) >= signed(b, w),
        IntPred::Slt => signed(a, w) < signed(b, w),
        IntPred::Sle => signed(a, w) <= signed(b, w),
    }
}

// ---------------------------------------------------------------------------
// Transfer: integer casts.
// ---------------------------------------------------------------------------

fn transfer_cast(
    ctx: DomainCtx<'_>,
    inst: &InstData,
    op: CastOp,
    operands: &[KnownBits],
) -> KnownBits {
    // Only the integer↔integer width casts are known-bits tractable; a pointer or
    // float result is never γ-contained by a `Bits`, so everything else is ⊤.
    if !matches!(op, CastOp::Trunc | CastOp::ZExt | CastOp::SExt) {
        return KnownBits::Top;
    }
    let Some(dst) = int_width(ctx.types, inst.ty) else {
        return KnownBits::Top;
    };
    let Some(KnownBits::Bits { width: src, zeros, ones }) = operands.first() else {
        // A ⊥ is filtered earlier; a ⊤ source has no width to extend from.
        return KnownBits::Top;
    };
    let src = *src;

    match op {
        // Drop the high bits: the low `dst` bits' knownness is preserved.
        CastOp::Trunc => make(dst, mask(zeros, dst), mask(ones, dst)),
        // Zero-extend: keep the low `src` bits; the new high bits are known 0.
        CastOp::ZExt => {
            let high = widen_region(src, dst);
            make(dst, mask(zeros, src).bitor(&high), mask(ones, src))
        }
        // Sign-extend: keep the low `src` bits; replicate a *known* sign bit into
        // the new high bits (otherwise leave them unknown).
        CastOp::SExt => {
            let high = widen_region(src, dst);
            let mut z = mask(zeros, src);
            let mut o = mask(ones, src);
            if src > 0 && ones.bit(src - 1) {
                o = o.bitor(&high);
            } else if src > 0 && zeros.bit(src - 1) {
                z = z.bitor(&high);
            }
            make(dst, z, o)
        }
        _ => KnownBits::Top,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::soundness::check_integer_transfer_sound;
    use crate::analysis::solver::solve;
    use crate::ir::inst::{Flags, InstData, InstKind};
    use crate::ir::types::TypeContext;
    use crate::ir::{Module, SemValue};
    use crate::support::StrInterner;

    // -- construction helpers ------------------------------------------------

    /// A `w`-bit value with the given known-zero / known-one bit patterns.
    fn kb(width: u32, zeros: u64, ones: u64) -> KnownBits {
        make(width, Int::from_u64(zeros), Int::from_u64(ones))
    }

    /// A width-known "all unknown" value (every bit unknown, but the width is
    /// pinned) — distinct from the width-agnostic [`KnownBits::Top`].
    fn unknown(width: u32) -> KnownBits {
        KnownBits::Bits { width, zeros: Int::ZERO, ones: Int::ZERO }
    }

    /// Assert the `zeros & ones == 0` invariant holds (trivially for ⊥/⊤).
    fn assert_invariant(kb: &KnownBits) {
        if let KnownBits::Bits { width, zeros, ones } = kb {
            assert!(zeros.bitand(ones).is_zero(), "zeros & ones must be disjoint: {kb:?}");
            assert_eq!(zeros, &mask(zeros, *width), "zeros out of width range: {kb:?}");
            assert_eq!(ones, &mask(ones, *width), "ones out of width range: {kb:?}");
        }
    }

    /// A representative spread of lattice elements for the algebraic laws.
    fn sample() -> Vec<KnownBits> {
        vec![
            KnownBits::Bottom,
            KnownBits::Top,
            unknown(8),
            kb(8, 0xF0, 0x00),      // high nibble known 0
            kb(8, 0x00, 0x0F),      // low nibble known 1
            kb(8, 0xF0, 0x0F),      // both nibbles pinned
            kb(8, 0xFF, 0x00),      // the constant 0
            kb(8, 0x00, 0xFF),      // the constant 0xFF
            kb(8, 0xAA, 0x55),      // alternating
            kb(4, 0x0, 0xF),        // a different width
            kb(32, 0xFFFF_FF00, 0x0000_0001),
        ]
    }

    // -- lattice laws --------------------------------------------------------

    #[test]
    fn lattice_laws() {
        let elems = sample();
        for a in &elems {
            assert_eq!(&a.join(a), a, "join idempotent");
            assert_eq!(&a.join(&KnownBits::Bottom), a, "bottom is join identity");
            assert_eq!(a.join(&KnownBits::Top), KnownBits::Top, "top absorbs");
            assert!(a.le(&KnownBits::Top), "everything ⊑ ⊤");
            assert!(KnownBits::Bottom.le(a), "⊥ ⊑ everything");
            assert_invariant(&a.join(a));
            for b in &elems {
                let ab = a.join(b);
                assert_eq!(ab, b.join(a), "join commutative");
                assert_invariant(&ab);
                // le is exactly consistent with join.
                assert_eq!(a.le(b), &a.join(b) == b, "le consistent with join: {a:?} {b:?}");
                for c in &elems {
                    assert_eq!(
                        a.join(b).join(c),
                        a.join(&b.join(c)),
                        "join associative: {a:?} {b:?} {c:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn gamma_contains() {
        // low nibble known 1, high nibble known 0 -> exactly 0x0F.
        let v = kb(8, 0xF0, 0x0F);
        assert!(v.contains(&SemValue::int(8, Int::from_u64(0x0F))));
        assert!(!v.contains(&SemValue::int(8, Int::from_u64(0x1F))), "high bit set violates zeros");
        assert!(!v.contains(&SemValue::int(8, Int::from_u64(0x0E))), "bit 0 clear violates ones");
        assert!(!v.contains(&SemValue::Poison), "a Bits never contains poison");
        assert!(KnownBits::Top.contains(&SemValue::Poison), "⊤ contains poison");
        assert!(KnownBits::Top.contains(&SemValue::int(8, Int::from_u64(0x99))));
        assert!(!KnownBits::Bottom.contains(&SemValue::int(8, Int::ZERO)));
    }

    // -- soundness (the key gate for bet B8) ---------------------------------

    #[test]
    fn transfer_is_sound_on_random_inputs() {
        for seed in [0x1234_5678u64, 0xDEAD_BEEF, 0x0BAD_F00D, 0x5EED_1234] {
            let report = check_integer_transfer_sound::<KnownBits>(8000, seed);
            assert!(report.is_sound(), "soundness violations (seed {seed:#x}): {:?}", report.violations);
            assert!(report.checked > 0, "the harness must check some cases");
        }
    }

    // -- precision spot-checks ----------------------------------------------

    /// Build an `InstData` for a binary op with the given flags and result type.
    fn bin_inst(op: BinOp, flags: Flags, ty: TypeId) -> InstData {
        InstData { kind: InstKind::Bin(op), flags, ty, operands: Vec::new(), result: None }
    }

    fn cast_inst(op: CastOp, ty: TypeId) -> InstData {
        InstData { kind: InstKind::Cast(op), flags: Flags::NONE, ty, operands: Vec::new(), result: None }
    }

    #[test]
    fn precision_and_mask_clears_high_bits() {
        let mut types = TypeContext::new();
        let i8t = types.int(8);
        let ctx = DomainCtx::new(&types);
        // (unknown i8) & 0x0F  ->  high nibble known 0.
        let inst = bin_inst(BinOp::And, Flags::NONE, i8t);
        let r = KnownBits::transfer(ctx, &inst, &[unknown(8), kb(8, 0xF0, 0x0F)]);
        assert_eq!(r, kb(8, 0xF0, 0x00), "x & 0x0F must know the high nibble is 0");
        assert_invariant(&r);
    }

    #[test]
    fn precision_or_sets_low_bit() {
        let mut types = TypeContext::new();
        let i8t = types.int(8);
        let ctx = DomainCtx::new(&types);
        // (unknown i8) | 0x01  ->  bit 0 known 1.
        let inst = bin_inst(BinOp::Or, Flags::NONE, i8t);
        let r = KnownBits::transfer(ctx, &inst, &[unknown(8), kb(8, 0xFE, 0x01)]);
        assert_eq!(r, kb(8, 0x00, 0x01), "x | 0x01 must know bit 0 is 1");
    }

    #[test]
    fn precision_shl_zeros_low_bits() {
        let mut types = TypeContext::new();
        let i8t = types.int(8);
        let ctx = DomainCtx::new(&types);
        // (unknown i8) << 4  ->  low 4 bits known 0.
        let inst = bin_inst(BinOp::Shl, Flags::NONE, i8t);
        let amt = known_const(8, &Int::from_u64(4));
        let r = KnownBits::transfer(ctx, &inst, &[unknown(8), amt]);
        assert_eq!(r, kb(8, 0x0F, 0x00), "x << 4 must know the low 4 bits are 0");
    }

    #[test]
    fn precision_lshr_zeros_high_bits() {
        let mut types = TypeContext::new();
        let i8t = types.int(8);
        let ctx = DomainCtx::new(&types);
        // (unknown i8) >>u 4  ->  high 4 bits known 0.
        let inst = bin_inst(BinOp::LShr, Flags::NONE, i8t);
        let amt = known_const(8, &Int::from_u64(4));
        let r = KnownBits::transfer(ctx, &inst, &[unknown(8), amt]);
        assert_eq!(r, kb(8, 0xF0, 0x00), "x >>u 4 must know the high 4 bits are 0");
    }

    #[test]
    fn precision_ashr_replicates_known_sign() {
        let mut types = TypeContext::new();
        let i8t = types.int(8);
        let ctx = DomainCtx::new(&types);
        let inst = bin_inst(BinOp::AShr, Flags::NONE, i8t);
        let amt = known_const(8, &Int::from_u64(4));
        // sign bit (bit 7) known 1 -> high 4 bits known 1 after ashr 4.
        let neg = kb(8, 0x00, 0x80);
        let r = KnownBits::transfer(ctx, &inst, &[neg, amt.clone()]);
        assert!(
            matches!(&r, KnownBits::Bits { ones, .. } if ones.bitand(&Int::from_u64(0xF0)) == Int::from_u64(0xF0)),
            "ashr must sign-extend a known 1 sign bit: {r:?}"
        );
        // sign bit known 0 -> high 4 bits known 0.
        let pos = kb(8, 0x80, 0x00);
        let r0 = KnownBits::transfer(ctx, &inst, &[pos, amt]);
        assert!(
            matches!(&r0, KnownBits::Bits { zeros, .. } if zeros.bitand(&Int::from_u64(0xF0)) == Int::from_u64(0xF0)),
            "ashr must sign-extend a known 0 sign bit: {r0:?}"
        );
    }

    #[test]
    fn precision_zext_zeros_high_bits() {
        let mut types = TypeContext::new();
        let i32t = types.int(32);
        let ctx = DomainCtx::new(&types);
        // zext (unknown i8) -> i32 : high 24 bits known 0, low 8 unknown.
        let inst = cast_inst(CastOp::ZExt, i32t);
        let r = KnownBits::transfer(ctx, &inst, &[unknown(8)]);
        match &r {
            KnownBits::Bits { width, zeros, ones } => {
                assert_eq!(*width, 32);
                assert_eq!(*zeros, Int::from_u64(0xFFFF_FF00), "high 24 bits must be known 0");
                assert!(ones.is_zero(), "no bit is known 1");
            }
            other => panic!("expected Bits, got {other:?}"),
        }
    }

    #[test]
    fn xor_of_two_independent_tops_is_top() {
        // The non-relational limitation: xor of two *independent* unknown i8s is
        // fully unknown, NOT the constant 0 (the domain cannot know they alias).
        let mut types = TypeContext::new();
        let i8t = types.int(8);
        let ctx = DomainCtx::new(&types);
        let inst = bin_inst(BinOp::Xor, Flags::NONE, i8t);
        let r = KnownBits::transfer(ctx, &inst, &[unknown(8), unknown(8)]);
        assert_eq!(r, unknown(8), "xor of two independent unknowns must stay unknown");
    }

    #[test]
    fn flagged_add_is_top() {
        // An nsw/nuw wrap is poison concretely, so a flagged add abstracts to ⊤.
        let mut types = TypeContext::new();
        let i8t = types.int(8);
        let ctx = DomainCtx::new(&types);
        let a = known_const(8, &Int::from_u64(0x7F));
        let inst_nsw = bin_inst(BinOp::Add, Flags::nsw(), i8t);
        assert_eq!(KnownBits::transfer(ctx, &inst_nsw, &[a.clone(), a.clone()]), KnownBits::Top);
        let inst_nuw = bin_inst(BinOp::Add, Flags::nuw(), i8t);
        assert_eq!(KnownBits::transfer(ctx, &inst_nuw, &[a.clone(), a]), KnownBits::Top);
    }

    #[test]
    fn add_carry_propagates_known_low_bits() {
        // 0b..00 (bit0 known 0) + 0b..01 (bit0 known 1) -> bit0 known 1, no carry.
        let mut types = TypeContext::new();
        let i8t = types.int(8);
        let ctx = DomainCtx::new(&types);
        let inst = bin_inst(BinOp::Add, Flags::NONE, i8t);
        // a: bit0 known 0, rest unknown; b: bit0 known 1, rest unknown.
        let a = kb(8, 0x01, 0x00);
        let b = kb(8, 0x00, 0x01);
        let r = KnownBits::transfer(ctx, &inst, &[a, b]);
        match &r {
            KnownBits::Bits { ones, .. } => assert!(ones.bit(0), "bit 0 of the sum must be known 1"),
            other => panic!("expected Bits, got {other:?}"),
        }
    }

    // -- end-to-end on the fixpoint engine -----------------------------------

    #[test]
    fn engine_infers_masked_bits() {
        // f(x: i32) -> i32 {
        //   t = x & 0x0F ;  r = t | 0x01 ;  ret r
        // }
        // The solver must infer: r has bits [4,32) known 0 and bit 0 known 1.
        let mut syms = StrInterner::new();
        let mut m = Module::new("kb");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);

        let r;
        {
            let mut bld = m.build(f);
            let entry = bld.create_entry_block();
            let x = bld.param(entry, 0);
            let m0f = bld.const_i64(i32t, 0x0F);
            let one = bld.const_i64(i32t, 0x01);
            let t = bld.bin(BinOp::And, x, m0f, Flags::NONE);
            r = bld.bin(BinOp::Or, t, one, Flags::NONE);
            bld.ret(Some(r));
        }

        let res = solve::<KnownBits>(m.function(f), m.types(), m.consts());
        assert_value_bits(res.value(r), 0xFFFF_FFF0, 0x0000_0001);
    }

    /// Assert the SSA value's known-bits masks match `zeros` / `ones` exactly.
    fn assert_value_bits(v: &KnownBits, zeros: u64, ones: u64) {
        match v {
            KnownBits::Bits { zeros: z, ones: o, .. } => {
                assert_eq!(*z, Int::from_u64(zeros), "known-zero mask mismatch: {v:?}");
                assert_eq!(*o, Int::from_u64(ones), "known-one mask mismatch: {v:?}");
            }
            other => panic!("expected inferred Bits, got {other:?}"),
        }
    }
}

