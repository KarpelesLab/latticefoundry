//! Tests for the reference semantics ([`super`]).
//!
//! These pin down the ground-truth behavior: poison propagation, the exact
//! poison-vs-UB split for the flagged/faulting ops, two's-complement widths,
//! IEEE comparisons, and the constant-folding oracle.

use super::{EvalOutcome, FoldResult, SemValue, eval, fold};
use crate::ir::inst::{BinOp, CastOp, FastMath, Flags, FloatPred, InstKind, IntPred, UnaryOp};
use crate::ir::types::{FloatKind, TypeContext};
use crate::ir::value::{Const, FloatBits};

use puremp::Int;

// --- small helpers ---------------------------------------------------------

/// An `n`-bit integer semantic value from a signed `i64`.
fn iv(w: u32, v: i64) -> SemValue {
    SemValue::int(w, Int::from_i64(v))
}

/// An `n`-bit integer semantic value from an unsigned `u64`.
fn uv(w: u32, v: u64) -> SemValue {
    SemValue::int(w, Int::from_u64(v))
}

fn f32v(x: f32) -> SemValue {
    SemValue::Float(FloatBits::F32(x.to_bits()))
}

fn f64v(x: f64) -> SemValue {
    SemValue::Float(FloatBits::F64(x.to_bits()))
}

/// Assert an outcome is a defined integer with the given unsigned bit pattern.
#[track_caller]
fn assert_int(out: EvalOutcome, width: u32, expected: u64) {
    match out {
        EvalOutcome::Value(SemValue::Int { width: w, bits }) => {
            assert_eq!(w, width, "width mismatch");
            assert_eq!(bits, Int::from_u64(expected), "value mismatch");
        }
        other => panic!("expected int {expected}, got {other:?}"),
    }
}

#[track_caller]
fn assert_poison(out: EvalOutcome) {
    assert_eq!(out, EvalOutcome::Value(SemValue::Poison), "expected poison");
}

#[track_caller]
fn assert_ub(out: EvalOutcome) {
    assert_eq!(out, EvalOutcome::UndefinedBehavior, "expected UB");
}

/// Decode the `f64` a float outcome denotes.
#[track_caller]
fn as_f64(out: EvalOutcome) -> f64 {
    match out {
        EvalOutcome::Value(SemValue::Float(b)) => super::decode(b),
        other => panic!("expected float, got {other:?}"),
    }
}

fn bin(op: BinOp) -> InstKind {
    InstKind::Bin(op)
}

// A fresh context with a few interned int/float types is convenient.
struct Ctx {
    cx: TypeContext,
}
impl Ctx {
    fn new() -> Self {
        Ctx { cx: TypeContext::new() }
    }
    fn int(&mut self, w: u32) -> crate::ir::types::TypeId {
        self.cx.int(w)
    }
    fn f32(&mut self) -> crate::ir::types::TypeId {
        self.cx.float(FloatKind::F32)
    }
    fn eval(&self, ty: crate::ir::types::TypeId, k: &InstKind, f: &Flags, ops: &[SemValue]) -> EvalOutcome {
        eval(&self.cx, ty, k, f, ops)
    }
}

// --- poison propagation ----------------------------------------------------

#[test]
fn poison_propagates_through_each_op_class() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    let i1 = c.int(1);
    let p = SemValue::Poison;

    // Binary, unary-ish, compare, cast, ptr_add all propagate a poison operand.
    assert_poison(c.eval(i8, &bin(BinOp::Add), &Flags::NONE, &[p.clone(), iv(8, 1)]));
    assert_poison(c.eval(i8, &bin(BinOp::And), &Flags::NONE, &[iv(8, 1), p.clone()]));
    assert_poison(c.eval(i1, &InstKind::ICmp(IntPred::Eq), &Flags::NONE, &[p.clone(), iv(8, 1)]));
    assert_poison(c.eval(i8, &InstKind::Cast(CastOp::ZExt), &Flags::NONE, std::slice::from_ref(&p)));
    let f32 = c.f32();
    assert_poison(c.eval(f32, &bin(BinOp::FAdd), &Flags::NONE, &[p.clone(), f32v(1.0)]));
    assert_poison(c.eval(f32, &InstKind::Unary(UnaryOp::FNeg), &Flags::NONE, std::slice::from_ref(&p)));
}

// --- add nsw / nuw ---------------------------------------------------------

#[test]
fn add_wraps_without_flags() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    // 127 + 1 = 128 (0x80) as an unsigned pattern.
    assert_int(c.eval(i8, &bin(BinOp::Add), &Flags::NONE, &[iv(8, 127), iv(8, 1)]), 8, 128);
    // 255 + 1 = 0 (wrap).
    assert_int(c.eval(i8, &bin(BinOp::Add), &Flags::NONE, &[uv(8, 255), iv(8, 1)]), 8, 0);
}

#[test]
fn add_nsw_overflow_is_poison_but_non_overflow_is_fine() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    // 127 + 1 signed-overflows (128 not representable in i8) ⇒ poison.
    assert_poison(c.eval(i8, &bin(BinOp::Add), &Flags::nsw(), &[iv(8, 127), iv(8, 1)]));
    // 100 + 27 = 127, in range ⇒ defined.
    assert_int(c.eval(i8, &bin(BinOp::Add), &Flags::nsw(), &[iv(8, 100), iv(8, 27)]), 8, 127);
}

#[test]
fn add_nuw_overflow_is_poison() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    assert_poison(c.eval(i8, &bin(BinOp::Add), &Flags::nuw(), &[uv(8, 255), iv(8, 1)]));
    assert_int(c.eval(i8, &bin(BinOp::Add), &Flags::nuw(), &[uv(8, 200), iv(8, 55)]), 8, 255);
}

#[test]
fn sub_and_mul_flags() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    // sub nuw: 0 - 1 unsigned-underflows ⇒ poison.
    assert_poison(c.eval(i8, &bin(BinOp::Sub), &Flags::nuw(), &[iv(8, 0), iv(8, 1)]));
    // mul nsw: 64 * 2 = 128 ⇒ signed overflow ⇒ poison.
    assert_poison(c.eval(i8, &bin(BinOp::Mul), &Flags::nsw(), &[iv(8, 64), iv(8, 2)]));
    // mul plain wraps: 64 * 4 = 256 ≡ 0.
    assert_int(c.eval(i8, &bin(BinOp::Mul), &Flags::NONE, &[iv(8, 64), iv(8, 4)]), 8, 0);
}

// --- div / rem: poison (exact) vs UB (fault) -------------------------------

#[test]
fn udiv_sdiv_urem_srem_basics() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    assert_int(c.eval(i8, &bin(BinOp::UDiv), &Flags::NONE, &[uv(8, 200), uv(8, 3)]), 8, 66);
    assert_int(c.eval(i8, &bin(BinOp::URem), &Flags::NONE, &[uv(8, 200), uv(8, 3)]), 8, 2);
    // sdiv -7 / 2 = -3 (trunc toward zero) ⇒ 0xFD = 253.
    assert_int(c.eval(i8, &bin(BinOp::SDiv), &Flags::NONE, &[iv(8, -7), iv(8, 2)]), 8, 253);
    // srem -7 % 2 = -1 (sign of dividend) ⇒ 0xFF = 255.
    assert_int(c.eval(i8, &bin(BinOp::SRem), &Flags::NONE, &[iv(8, -7), iv(8, 2)]), 8, 255);
    // Signed vs unsigned differ: 0xF8 (=-8 signed / 248 unsigned) / 2.
    assert_int(c.eval(i8, &bin(BinOp::SDiv), &Flags::NONE, &[uv(8, 248), iv(8, 2)]), 8, 252); // -8/2=-4
    assert_int(c.eval(i8, &bin(BinOp::UDiv), &Flags::NONE, &[uv(8, 248), iv(8, 2)]), 8, 124); // 248/2
}

#[test]
fn division_by_zero_is_ub() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    assert_ub(c.eval(i8, &bin(BinOp::UDiv), &Flags::NONE, &[iv(8, 5), iv(8, 0)]));
    assert_ub(c.eval(i8, &bin(BinOp::SDiv), &Flags::NONE, &[iv(8, 5), iv(8, 0)]));
    assert_ub(c.eval(i8, &bin(BinOp::URem), &Flags::NONE, &[iv(8, 5), iv(8, 0)]));
    assert_ub(c.eval(i8, &bin(BinOp::SRem), &Flags::NONE, &[iv(8, 5), iv(8, 0)]));
}

#[test]
fn sdiv_srem_int_min_by_minus_one_is_ub() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    // INT_MIN = -128 = 0x80.
    assert_ub(c.eval(i8, &bin(BinOp::SDiv), &Flags::NONE, &[iv(8, -128), iv(8, -1)]));
    assert_ub(c.eval(i8, &bin(BinOp::SRem), &Flags::NONE, &[iv(8, -128), iv(8, -1)]));
    // But sdiv INT_MIN / 1 is fine, and udiv 0x80 / 0xFF is fine (unsigned).
    assert_int(c.eval(i8, &bin(BinOp::SDiv), &Flags::NONE, &[iv(8, -128), iv(8, 1)]), 8, 128);
    assert_int(c.eval(i8, &bin(BinOp::UDiv), &Flags::NONE, &[uv(8, 128), uv(8, 255)]), 8, 0);
}

#[test]
fn exact_div_violation_is_poison() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    // 7 / 2 has remainder 1 ⇒ exact violated ⇒ poison.
    assert_poison(c.eval(i8, &bin(BinOp::UDiv), &Flags::exact(), &[iv(8, 7), iv(8, 2)]));
    // 6 / 2 is exact ⇒ 3.
    assert_int(c.eval(i8, &bin(BinOp::UDiv), &Flags::exact(), &[iv(8, 6), iv(8, 2)]), 8, 3);
}

// --- shifts ----------------------------------------------------------------

#[test]
fn over_wide_shift_is_poison() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    assert_poison(c.eval(i8, &bin(BinOp::Shl), &Flags::NONE, &[iv(8, 1), iv(8, 8)]));
    assert_poison(c.eval(i8, &bin(BinOp::LShr), &Flags::NONE, &[iv(8, 1), iv(8, 100)]));
    assert_poison(c.eval(i8, &bin(BinOp::AShr), &Flags::NONE, &[iv(8, 1), iv(8, 8)]));
}

#[test]
fn ashr_vs_lshr() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    // 0x80 (=-128) >> 1: ashr ⇒ 0xC0 (=-64=192), lshr ⇒ 0x40 (64).
    assert_int(c.eval(i8, &bin(BinOp::AShr), &Flags::NONE, &[uv(8, 0x80), iv(8, 1)]), 8, 0xC0);
    assert_int(c.eval(i8, &bin(BinOp::LShr), &Flags::NONE, &[uv(8, 0x80), iv(8, 1)]), 8, 0x40);
    // -1 >> 3 arithmetic stays -1.
    assert_int(c.eval(i8, &bin(BinOp::AShr), &Flags::NONE, &[iv(8, -1), iv(8, 3)]), 8, 0xFF);
}

#[test]
fn shl_flags_and_exact_shift() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    // shl 1 << 7 = 0x80; nsw: signed 1*128=128 out of range ⇒ poison.
    assert_poison(c.eval(i8, &bin(BinOp::Shl), &Flags::nsw(), &[iv(8, 1), iv(8, 7)]));
    // nuw: 3 << 7 loses bits ⇒ poison; plain wraps to 0x80.
    assert_poison(c.eval(i8, &bin(BinOp::Shl), &Flags::nuw(), &[iv(8, 3), iv(8, 7)]));
    assert_int(c.eval(i8, &bin(BinOp::Shl), &Flags::NONE, &[iv(8, 3), iv(8, 7)]), 8, 0x80);
    // exact lshr: 0b11 >> 1 shifts out a set bit ⇒ poison; 0b10 >> 1 is exact.
    assert_poison(c.eval(i8, &bin(BinOp::LShr), &Flags::exact(), &[iv(8, 0b11), iv(8, 1)]));
    assert_int(c.eval(i8, &bin(BinOp::LShr), &Flags::exact(), &[iv(8, 0b10), iv(8, 1)]), 8, 1);
}

// --- icmp ------------------------------------------------------------------

#[test]
fn icmp_all_predicates() {
    let mut c = Ctx::new();
    let i1 = c.int(1);
    // a = 0xFF (=-1 signed / 255 unsigned), b = 1.
    let a = uv(8, 0xFF);
    let b = iv(8, 1);
    let check = |c: &Ctx, pred: IntPred, expect: bool| {
        assert_int(
            c.eval(i1, &InstKind::ICmp(pred), &Flags::NONE, &[a.clone(), b.clone()]),
            1,
            u64::from(expect),
        );
    };
    check(&c, IntPred::Eq, false);
    check(&c, IntPred::Ne, true);
    check(&c, IntPred::Ugt, true); // 255 > 1
    check(&c, IntPred::Uge, true);
    check(&c, IntPred::Ult, false);
    check(&c, IntPred::Ule, false);
    check(&c, IntPred::Sgt, false); // -1 > 1 is false
    check(&c, IntPred::Sge, false);
    check(&c, IntPred::Slt, true); // -1 < 1
    check(&c, IntPred::Sle, true);
}

// --- casts -----------------------------------------------------------------

#[test]
fn trunc_zext_sext_round_trips() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    let i32 = c.int(32);
    // trunc i32 0x1FF -> i8 0xFF.
    assert_int(c.eval(i8, &InstKind::Cast(CastOp::Trunc), &Flags::NONE, &[uv(32, 0x1FF)]), 8, 0xFF);
    // zext i8 0xFF -> i32 255.
    assert_int(c.eval(i32, &InstKind::Cast(CastOp::ZExt), &Flags::NONE, &[uv(8, 0xFF)]), 32, 255);
    // sext i8 0xFF (=-1) -> i32 0xFFFFFFFF.
    assert_int(
        c.eval(i32, &InstKind::Cast(CastOp::SExt), &Flags::NONE, &[uv(8, 0xFF)]),
        32,
        0xFFFF_FFFF,
    );
    // sext of a positive value keeps it.
    assert_int(c.eval(i32, &InstKind::Cast(CastOp::SExt), &Flags::NONE, &[iv(8, 5)]), 32, 5);
}

#[test]
fn fp_to_int_range_and_conversions() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    let f32 = c.f32();
    // fptosi 100.9 -> 100 (trunc toward zero).
    assert_int(c.eval(i8, &InstKind::Cast(CastOp::FpToSi), &Flags::NONE, &[f32v(100.9)]), 8, 100);
    // fptosi -1.5 -> -1 = 0xFF.
    assert_int(c.eval(i8, &InstKind::Cast(CastOp::FpToSi), &Flags::NONE, &[f32v(-1.5)]), 8, 0xFF);
    // Out of range ⇒ poison.
    assert_poison(c.eval(i8, &InstKind::Cast(CastOp::FpToSi), &Flags::NONE, &[f32v(300.0)]));
    // NaN ⇒ poison.
    assert_poison(c.eval(i8, &InstKind::Cast(CastOp::FpToSi), &Flags::NONE, &[f32v(f32::NAN)]));
    // fptoui of a negative ⇒ poison.
    assert_poison(c.eval(i8, &InstKind::Cast(CastOp::FpToUi), &Flags::NONE, &[f32v(-1.0)]));
    // sitofp i8 -1 -> -1.0f.
    assert_eq!(as_f64(c.eval(f32, &InstKind::Cast(CastOp::SiToFp), &Flags::NONE, &[iv(8, -1)])), -1.0);
    // uitofp of 0xFF (=255) -> 255.0.
    assert_eq!(as_f64(c.eval(f32, &InstKind::Cast(CastOp::UiToFp), &Flags::NONE, &[uv(8, 0xFF)])), 255.0);
}

#[test]
fn fptrunc_fpext_and_bitcast() {
    let mut c = Ctx::new();
    let f32 = c.f32();
    let f64_ = c.cx.float(FloatKind::F64);
    let i32 = c.int(32);
    // fpext f32 1.5 -> f64 1.5.
    assert_eq!(as_f64(c.eval(f64_, &InstKind::Cast(CastOp::FpExt), &Flags::NONE, &[f32v(1.5)])), 1.5);
    // fptrunc f64 1.5 -> f32 1.5.
    assert_eq!(as_f64(c.eval(f32, &InstKind::Cast(CastOp::FpTrunc), &Flags::NONE, &[f64v(1.5)])), 1.5);
    // bitcast f32 <-> i32 preserves bits.
    let bits = 1.5f32.to_bits();
    assert_int(
        c.eval(i32, &InstKind::Cast(CastOp::Bitcast), &Flags::NONE, &[f32v(1.5)]),
        32,
        u64::from(bits),
    );
    assert_eq!(
        as_f64(c.eval(f32, &InstKind::Cast(CastOp::Bitcast), &Flags::NONE, &[uv(32, u64::from(bits))])),
        1.5
    );
}

// --- float arithmetic & comparisons ---------------------------------------

#[test]
fn float_arithmetic() {
    let mut c = Ctx::new();
    let f32 = c.f32();
    assert_eq!(as_f64(c.eval(f32, &bin(BinOp::FAdd), &Flags::NONE, &[f32v(1.0), f32v(2.0)])), 3.0);
    assert_eq!(as_f64(c.eval(f32, &bin(BinOp::FSub), &Flags::NONE, &[f32v(1.0), f32v(2.0)])), -1.0);
    assert_eq!(as_f64(c.eval(f32, &bin(BinOp::FMul), &Flags::NONE, &[f32v(3.0), f32v(4.0)])), 12.0);
    assert_eq!(as_f64(c.eval(f32, &bin(BinOp::FDiv), &Flags::NONE, &[f32v(7.0), f32v(2.0)])), 3.5);
    assert_eq!(as_f64(c.eval(f32, &bin(BinOp::FRem), &Flags::NONE, &[f32v(7.0), f32v(3.0)])), 1.0);
}

#[test]
fn fneg_flips_sign() {
    let mut c = Ctx::new();
    let f32 = c.f32();
    assert_eq!(as_f64(c.eval(f32, &InstKind::Unary(UnaryOp::FNeg), &Flags::NONE, &[f32v(1.0)])), -1.0);
    // Sign of zero flips (deterministic), no poison.
    let out = c.eval(f32, &InstKind::Unary(UnaryOp::FNeg), &Flags::NONE, &[f32v(0.0)]);
    assert!(as_f64(out).is_sign_negative());
}

#[test]
fn fcmp_nan_handling() {
    let mut c = Ctx::new();
    let i1 = c.int(1);
    let nan = f32v(f32::NAN);
    let one = f32v(1.0);
    let check = |c: &Ctx, pred: FloatPred, a: &SemValue, b: &SemValue, expect: bool| {
        assert_int(
            c.eval(i1, &InstKind::FCmp(pred), &Flags::NONE, &[a.clone(), b.clone()]),
            1,
            u64::from(expect),
        );
    };
    // Ordered predicates are false when an operand is NaN.
    check(&c, FloatPred::Oeq, &nan, &one, false);
    check(&c, FloatPred::Olt, &nan, &one, false);
    check(&c, FloatPred::Ord, &nan, &one, false);
    // Unordered predicates are true when an operand is NaN.
    check(&c, FloatPred::Ueq, &nan, &one, true);
    check(&c, FloatPred::Une, &nan, &one, true);
    check(&c, FloatPred::Uno, &nan, &one, true);
    // Without NaN, ordered/unordered agree on the relation.
    check(&c, FloatPred::Oeq, &one, &one, true);
    check(&c, FloatPred::Olt, &f32v(1.0), &f32v(2.0), true);
    check(&c, FloatPred::One, &one, &one, false);
    // Constants: True/False.
    check(&c, FloatPred::True, &nan, &one, true);
    check(&c, FloatPred::False, &one, &one, false);
}

#[test]
fn fast_math_nnan_ninf_are_poison_on_violation() {
    let mut c = Ctx::new();
    let f32 = c.f32();
    let nnan = Flags::fast(FastMath { nnan: true, ..FastMath::default() });
    // nnan but a NaN operand ⇒ poison.
    assert_poison(c.eval(f32, &bin(BinOp::FAdd), &nnan, &[f32v(f32::NAN), f32v(1.0)]));
    // ninf but the result overflows to +inf ⇒ poison.
    let ninf = Flags::fast(FastMath { ninf: true, ..FastMath::default() });
    assert_poison(c.eval(f32, &bin(BinOp::FMul), &ninf, &[f32v(f32::MAX), f32v(f32::MAX)]));
    // nsz alone never poisons.
    let nsz = Flags::fast(FastMath { nsz: true, ..FastMath::default() });
    assert_eq!(as_f64(c.eval(f32, &bin(BinOp::FAdd), &nsz, &[f32v(1.0), f32v(1.0)])), 2.0);
}

// --- select ----------------------------------------------------------------

#[test]
fn select_true_false_and_poison() {
    let mut c = Ctx::new();
    let i8 = c.int(8);
    let t = iv(8, 10);
    let f = iv(8, 20);
    // cond true selects the true arm; false selects the false arm.
    assert_int(
        c.eval(i8, &InstKind::Select, &Flags::NONE, &[SemValue::boolean(true), t.clone(), f.clone()]),
        8,
        10,
    );
    assert_int(
        c.eval(i8, &InstKind::Select, &Flags::NONE, &[SemValue::boolean(false), t.clone(), f.clone()]),
        8,
        20,
    );
    // Poison condition ⇒ poison.
    assert_poison(c.eval(i8, &InstKind::Select, &Flags::NONE, &[SemValue::Poison, t.clone(), f.clone()]));
    // Poison in the *non-selected* arm does not taint the result.
    assert_int(
        c.eval(i8, &InstKind::Select, &Flags::NONE, &[SemValue::boolean(true), t.clone(), SemValue::Poison]),
        8,
        10,
    );
}

// --- freeze ----------------------------------------------------------------

#[test]
fn freeze_poison_is_fixed_value_and_freeze_identity() {
    let mut c = Ctx::new();
    let i32 = c.int(32);
    let f32 = c.f32();
    // freeze(poison) ⇒ the fixed zero value.
    assert_int(c.eval(i32, &InstKind::Freeze, &Flags::NONE, &[SemValue::Poison]), 32, 0);
    let fz = c.eval(f32, &InstKind::Freeze, &Flags::NONE, &[SemValue::Poison]);
    assert_eq!(as_f64(fz), 0.0);
    // Determinism: two freezes of poison agree.
    let a = c.eval(i32, &InstKind::Freeze, &Flags::NONE, &[SemValue::Poison]);
    let b = c.eval(i32, &InstKind::Freeze, &Flags::NONE, &[SemValue::Poison]);
    assert_eq!(a, b);
    // freeze(v) = v for a defined value.
    assert_int(c.eval(i32, &InstKind::Freeze, &Flags::NONE, &[iv(32, 42)]), 32, 42);
}

// --- ptr_add / ptrtoint / inttoptr ----------------------------------------

#[test]
fn pointer_arithmetic() {
    let mut c = Ctx::new();
    let ptr = c.cx.ptr();
    let i64_ = c.int(64);
    // ptr_add null + 16 = 16.
    let out = c.eval(ptr, &InstKind::PtrAdd { inbounds: false }, &Flags::NONE, &[SemValue::ptr(Int::ZERO), iv(64, 16)]);
    match out {
        EvalOutcome::Value(SemValue::Ptr(a)) => assert_eq!(a, Int::from_u64(16)),
        other => panic!("expected ptr, got {other:?}"),
    }
    // ptr_add base + (-1) wraps within 64 bits.
    let out = c.eval(ptr, &InstKind::PtrAdd { inbounds: false }, &Flags::NONE, &[SemValue::ptr(Int::from_u64(0)), iv(64, -1)]);
    match out {
        EvalOutcome::Value(SemValue::Ptr(a)) => assert_eq!(a, Int::from_u64(u64::MAX)),
        other => panic!("expected ptr, got {other:?}"),
    }
    // ptrtoint of address 16 -> i64 16.
    assert_int(
        c.eval(i64_, &InstKind::Cast(CastOp::PtrToInt), &Flags::NONE, &[SemValue::ptr(Int::from_u64(16))]),
        64,
        16,
    );
    // inttoptr 16 -> ptr 16.
    let out = c.eval(ptr, &InstKind::Cast(CastOp::IntToPtr), &Flags::NONE, &[iv(64, 16)]);
    match out {
        EvalOutcome::Value(SemValue::Ptr(a)) => assert_eq!(a, Int::from_u64(16)),
        other => panic!("expected ptr, got {other:?}"),
    }
}

// --- constant folding ------------------------------------------------------

#[test]
fn fold_produces_constants() {
    let mut cx = TypeContext::new();
    let i8 = cx.int(8);
    let a = Const::Int { ty: i8, value: Int::from_i64(100) };
    let b = Const::Int { ty: i8, value: Int::from_i64(100) };
    let r = fold(&cx, i8, &bin(BinOp::Add), &Flags::NONE, &[a, b]);
    // 100 + 100 = 200 (unsigned representative of the i8 pattern).
    assert_eq!(r, Some(FoldResult::Folded(Const::Int { ty: i8, value: Int::from_u64(200) })));
}

#[test]
fn fold_refuses_ub() {
    let mut cx = TypeContext::new();
    let i8 = cx.int(8);
    let a = Const::Int { ty: i8, value: Int::from_i64(5) };
    let zero = Const::Int { ty: i8, value: Int::from_i64(0) };
    assert_eq!(fold(&cx, i8, &bin(BinOp::SDiv), &Flags::NONE, &[a, zero]), Some(FoldResult::WouldBeUb));
}

#[test]
fn fold_poison_and_flag_violation() {
    let mut cx = TypeContext::new();
    let i8 = cx.int(8);
    // A poison operand folds to a poison constant.
    let p = Const::Poison(i8);
    let one = Const::Int { ty: i8, value: Int::from_i64(1) };
    assert_eq!(
        fold(&cx, i8, &bin(BinOp::Add), &Flags::NONE, &[p, one.clone()]),
        Some(FoldResult::Folded(Const::Poison(i8)))
    );
    // add nsw overflow folds to poison, not UB.
    let big = Const::Int { ty: i8, value: Int::from_i64(127) };
    assert_eq!(
        fold(&cx, i8, &bin(BinOp::Add), &Flags::nsw(), &[big, one]),
        Some(FoldResult::Folded(Const::Poison(i8)))
    );
}

#[test]
fn fold_aggregate_operand_is_not_foldable() {
    let mut cx = TypeContext::new();
    let i8 = cx.int(8);
    let arr = cx.array(i8, 2);
    let agg = Const::Aggregate { ty: arr, elems: vec![] };
    // Not a scalar the evaluator models ⇒ None.
    assert_eq!(fold(&cx, i8, &InstKind::Cast(CastOp::ZExt), &Flags::NONE, &[agg]), None);
}

// --- f16 codec -------------------------------------------------------------

#[test]
fn f16_codec_round_trips() {
    // A few exact binary16 values, decoded to f64 and re-encoded.
    let cases: &[(u16, f64)] = &[
        (0x0000, 0.0),
        (0x3C00, 1.0),   // 1.0
        (0x4000, 2.0),   // 2.0
        (0xC000, -2.0),  // -2.0
        (0x3800, 0.5),   // 0.5
        (0x7C00, f64::INFINITY),
        (0xFC00, f64::NEG_INFINITY),
    ];
    for &(bits, val) in cases {
        assert_eq!(super::f16_to_f64(bits), val, "decode 0x{bits:04X}");
        if val.is_finite() {
            assert_eq!(super::f16_from_f64(val), bits, "encode {val}");
        }
    }
    // Overflow to inf, and NaN round-trips to a NaN.
    assert_eq!(super::f16_from_f64(70000.0), 0x7C00); // overflow ⇒ +inf
    assert!(super::f16_to_f64(super::f16_from_f64(f64::NAN)).is_nan());
    // A value below the smallest subnormal rounds to zero.
    assert_eq!(super::f16_from_f64(1e-30), 0x0000);
}

#[test]
fn f16_arithmetic_via_evaluator() {
    let mut cx = TypeContext::new();
    let f16 = cx.float(FloatKind::F16);
    // 1.0 + 0.5 = 1.5 in binary16.
    let a = SemValue::Float(FloatBits::F16(0x3C00));
    let b = SemValue::Float(FloatBits::F16(0x3800));
    let out = eval(&cx, f16, &bin(BinOp::FAdd), &Flags::NONE, &[a, b]);
    assert_eq!(as_f64(out), 1.5);
}
