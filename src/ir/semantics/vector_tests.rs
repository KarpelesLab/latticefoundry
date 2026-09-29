//! Tests for the **vector** reference semantics (`docs/ir-design.md` §6e):
//! lane-wise evaluation with per-lane poison, the whole-instruction UB rule,
//! the lane moves (`extractelement`, `insertelement`, `shufflevector`,
//! `splat`), the ordered reductions, and bit-level `bitcast`.
//!
//! Besides targeted cases, a randomized differential checks every lane-wise op
//! on random vectors (with random poison lanes) against the op evaluated one
//! lane at a time on scalars, and against an independent native-Rust model of
//! the integer and float arithmetic.

use super::{EvalOutcome, SemValue, eval};
use crate::ir::inst::{BinOp, CastOp, FastMath, Flags, FloatPred, InstKind, IntPred, ReduceOp, UnaryOp};
use crate::ir::types::{FloatKind, TypeContext, TypeId};
use crate::ir::value::FloatBits;

use puremp::Int;

fn iv(w: u32, v: i64) -> SemValue {
    SemValue::int(w, Int::from_i64(v))
}

fn vecv(lanes: Vec<SemValue>) -> SemValue {
    SemValue::Vector(lanes)
}

fn ivec(w: u32, vals: &[i64]) -> SemValue {
    vecv(vals.iter().map(|&v| iv(w, v)).collect())
}

fn f32v(x: f32) -> SemValue {
    SemValue::Float(FloatBits::F32(x.to_bits()))
}

fn val(out: EvalOutcome) -> SemValue {
    match out {
        EvalOutcome::Value(v) => v,
        EvalOutcome::UndefinedBehavior => panic!("unexpected UB"),
    }
}

/// A tiny deterministic generator (SplitMix64), so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn lanewise_add_keeps_poison_in_its_lane() {
    let mut cx = TypeContext::new();
    let i32t = cx.int(32);
    let v4 = cx.vector(i32t, 4);
    let a = vecv(vec![iv(32, 1), SemValue::Poison, iv(32, 3), iv(32, -1)]);
    let b = ivec(32, &[10, 20, 30, 1]);
    let r = val(eval(&cx, v4, &InstKind::Bin(BinOp::Add), &Flags::NONE, &[a, b]));
    assert_eq!(r, vecv(vec![iv(32, 11), SemValue::Poison, iv(32, 33), iv(32, 0)]));
}

#[test]
fn nsw_overflow_poisons_only_the_overflowing_lane() {
    let mut cx = TypeContext::new();
    let i8t = cx.int(8);
    let v2 = cx.vector(i8t, 2);
    let a = ivec(8, &[127, 1]);
    let b = ivec(8, &[1, 1]);
    let r = val(eval(&cx, v2, &InstKind::Bin(BinOp::Add), &Flags::nsw(), &[a, b]));
    assert_eq!(r, vecv(vec![SemValue::Poison, iv(8, 2)]));
}

#[test]
fn over_wide_shift_poisons_only_its_lane() {
    let mut cx = TypeContext::new();
    let i16t = cx.int(16);
    let v4 = cx.vector(i16t, 4);
    let a = ivec(16, &[1, 1, 1, -1]);
    let amt = ivec(16, &[3, 16, 15, 40]);
    let r = val(eval(&cx, v4, &InstKind::Bin(BinOp::Shl), &Flags::NONE, &[a.clone(), amt.clone()]));
    assert_eq!(r, vecv(vec![iv(16, 8), SemValue::Poison, iv(16, -32768), SemValue::Poison]));
    let r = val(eval(&cx, v4, &InstKind::Bin(BinOp::AShr), &Flags::NONE, &[a, amt]));
    assert_eq!(r, vecv(vec![iv(16, 0), SemValue::Poison, iv(16, 0), SemValue::Poison]));
}

#[test]
fn a_zero_divisor_in_any_lane_is_ub_of_the_whole_op() {
    let mut cx = TypeContext::new();
    let i32t = cx.int(32);
    let v4 = cx.vector(i32t, 4);
    let a = ivec(32, &[8, 8, 8, 8]);
    let b = ivec(32, &[2, 4, 0, 1]);
    assert_eq!(
        eval(&cx, v4, &InstKind::Bin(BinOp::UDiv), &Flags::NONE, &[a.clone(), b]),
        EvalOutcome::UndefinedBehavior
    );
    // ... even when another lane is poison.
    let b = vecv(vec![SemValue::Poison, iv(32, 0), iv(32, 1), iv(32, 1)]);
    assert_eq!(eval(&cx, v4, &InstKind::Bin(BinOp::SDiv), &Flags::NONE, &[a, b]), EvalOutcome::UndefinedBehavior);
}

#[test]
fn vector_compares_give_i1_lanes() {
    let mut cx = TypeContext::new();
    let i32t = cx.int(32);
    let b1 = cx.bool();
    let m4 = cx.vector(b1, 4);
    let a = ivec(32, &[1, -1, 5, 7]);
    let b = ivec(32, &[2, 1, 5, 7]);
    let r = val(eval(&cx, m4, &InstKind::ICmp(IntPred::Slt), &Flags::NONE, &[a.clone(), b.clone()]));
    assert_eq!(r, vecv(vec![iv(1, 1), iv(1, 1), iv(1, 0), iv(1, 0)]));
    let r = val(eval(&cx, m4, &InstKind::ICmp(IntPred::Ult), &Flags::NONE, &[a, b]));
    assert_eq!(r, vecv(vec![iv(1, 1), iv(1, 0), iv(1, 0), iv(1, 0)]));
    let _ = i32t;

    let x = vecv(vec![f32v(1.0), f32v(f32::NAN), f32v(2.0), SemValue::Poison]);
    let y = vecv(vec![f32v(1.0), f32v(0.0), f32v(3.0), f32v(0.0)]);
    let r = val(eval(&cx, m4, &InstKind::FCmp(FloatPred::Ueq), &Flags::NONE, &[x, y]));
    assert_eq!(r, vecv(vec![iv(1, 1), iv(1, 1), iv(1, 0), SemValue::Poison]));
}

#[test]
fn select_with_a_vector_condition_is_per_lane() {
    let mut cx = TypeContext::new();
    let i32t = cx.int(32);
    let v4 = cx.vector(i32t, 4);
    let c = vecv(vec![iv(1, 1), iv(1, 0), SemValue::Poison, iv(1, 0)]);
    let t = vecv(vec![iv(32, 1), iv(32, 2), iv(32, 3), SemValue::Poison]);
    let f = ivec(32, &[10, 20, 30, 40]);
    let r = val(eval(&cx, v4, &InstKind::Select, &Flags::NONE, &[c, t.clone(), f.clone()]));
    // Lane 2's poison condition poisons it; lane 3 picks the defined false arm
    // (the non-selected poison lane of `t` does not matter).
    assert_eq!(r, vecv(vec![iv(32, 1), iv(32, 20), SemValue::Poison, iv(32, 40)]));
    // A scalar condition picks a whole arm.
    let r = val(eval(&cx, v4, &InstKind::Select, &Flags::NONE, &[iv(1, 0), t, f.clone()]));
    assert_eq!(r, f);
}

#[test]
fn freeze_is_per_lane() {
    let mut cx = TypeContext::new();
    let i32t = cx.int(32);
    let v3 = cx.vector(i32t, 3);
    let r = val(eval(&cx, v3, &InstKind::Freeze, &Flags::NONE, &[vecv(vec![iv(32, 4), SemValue::Poison, iv(32, 6)])]));
    assert_eq!(r, ivec(32, &[4, 0, 6]));
    // A whole-poison vector freezes to all zeros.
    let r = val(eval(&cx, v3, &InstKind::Freeze, &Flags::NONE, &[SemValue::Poison]));
    assert_eq!(r, ivec(32, &[0, 0, 0]));
}

#[test]
fn lane_moves() {
    let mut cx = TypeContext::new();
    let i32t = cx.int(32);
    let v4 = cx.vector(i32t, 4);
    let a = vecv(vec![iv(32, 0), iv(32, 1), SemValue::Poison, iv(32, 3)]);
    let b = ivec(32, &[4, 5, 6, 7]);

    // extractelement
    assert_eq!(val(eval(&cx, i32t, &InstKind::ExtractElement { lane: 1 }, &Flags::NONE, std::slice::from_ref(&a))), iv(32, 1));
    assert_eq!(val(eval(&cx, i32t, &InstKind::ExtractElement { lane: 2 }, &Flags::NONE, std::slice::from_ref(&a))), SemValue::Poison);
    assert_eq!(val(eval(&cx, i32t, &InstKind::ExtractElement { lane: 0 }, &Flags::NONE, &[SemValue::Poison])), SemValue::Poison);

    // insertelement: into a whole-poison vector only the written lane is defined.
    let r = val(eval(&cx, v4, &InstKind::InsertElement { lane: 2 }, &Flags::NONE, &[SemValue::Poison, iv(32, 9)]));
    assert_eq!(r, vecv(vec![SemValue::Poison, SemValue::Poison, iv(32, 9), SemValue::Poison]));
    let r = val(eval(&cx, v4, &InstKind::InsertElement { lane: 2 }, &Flags::NONE, &[a.clone(), iv(32, 2)]));
    assert_eq!(r, ivec(32, &[0, 1, 2, 3]));

    // shufflevector: picks from a ++ b; the result may change the lane count.
    let v2 = cx.vector(i32t, 2);
    let mask: Box<[u32]> = vec![7, 0, 5, 3].into();
    let r = val(eval(&cx, v4, &InstKind::ShuffleVector(mask), &Flags::NONE, &[a.clone(), b.clone()]));
    assert_eq!(r, ivec(32, &[7, 0, 5, 3]));
    let mask: Box<[u32]> = vec![2, 4].into();
    let r = val(eval(&cx, v2, &InstKind::ShuffleVector(mask), &Flags::NONE, &[a, SemValue::Poison]));
    assert_eq!(r, vecv(vec![SemValue::Poison, SemValue::Poison]));

    // splat
    let r = val(eval(&cx, v4, &InstKind::Splat, &Flags::NONE, &[iv(32, -2)]));
    assert_eq!(r, ivec(32, &[-2, -2, -2, -2]));
}

#[test]
fn reductions() {
    let mut cx = TypeContext::new();
    let i8t = cx.int(8);
    let f32t = cx.float(FloatKind::F32);
    let v = ivec(8, &[100, 100, -3, 7]); // the two 100s cancel in the xor
    let red = |op: ReduceOp, v: &SemValue| val(eval(&cx, i8t, &InstKind::Reduce(op), &Flags::NONE, std::slice::from_ref(v)));
    assert_eq!(red(ReduceOp::Add, &v), iv(8, (100i8.wrapping_add(100).wrapping_add(-3).wrapping_add(7)) as i64));
    assert_eq!(red(ReduceOp::SMin, &v), iv(8, -3));
    assert_eq!(red(ReduceOp::SMax, &v), iv(8, 100));
    assert_eq!(red(ReduceOp::UMax, &v), iv(8, -3)); // 0xFD is the unsigned max
    assert_eq!(red(ReduceOp::UMin, &v), iv(8, 7));
    assert_eq!(red(ReduceOp::Xor, &v), iv(8, (-3i64 ^ 7) & 0xFF));
    let pv = vecv(vec![iv(8, 1), SemValue::Poison]);
    assert_eq!(red(ReduceOp::Or, &pv), SemValue::Poison);

    // The float sum is ordered: ((1e8 + 1) + -1e8) + 1 == 1 in f32 (the +1 is
    // absorbed by rounding), while a reassociated order would give 2.
    let fv = vecv(vec![f32v(1.0e8), f32v(1.0), f32v(-1.0e8), f32v(1.0)]);
    let r = val(eval(&cx, f32t, &InstKind::Reduce(ReduceOp::FAdd), &Flags::NONE, &[fv]));
    assert_eq!(r, f32v(((1.0e8f32 + 1.0) + -1.0e8) + 1.0));
    assert_eq!(r, f32v(1.0));
    // A fast-math violation at any step poisons the result.
    let fv = vecv(vec![f32v(1.0), f32v(f32::INFINITY)]);
    let ninf = Flags::fast(FastMath { ninf: true, ..FastMath::default() });
    assert_eq!(val(eval(&cx, f32t, &InstKind::Reduce(ReduceOp::FMul), &ninf, &[fv])), SemValue::Poison);
}

#[test]
fn bitcast_packs_lanes_low_first_and_tracks_poison_by_overlap() {
    let mut cx = TypeContext::new();
    let i8t = cx.int(8);
    let i16t = cx.int(16);
    let i32t = cx.int(32);
    let v4i8 = cx.vector(i8t, 4);
    let v2i16 = cx.vector(i16t, 2);
    let bc = InstKind::Cast(CastOp::Bitcast);

    let src = ivec(8, &[0x11, 0x22, 0x33, 0x44]);
    assert_eq!(val(eval(&cx, i32t, &bc, &Flags::NONE, std::slice::from_ref(&src))), iv(32, 0x4433_2211));
    assert_eq!(val(eval(&cx, v2i16, &bc, &Flags::NONE, &[src])), ivec(16, &[0x2211, 0x4433]));
    // Scalar -> vector.
    assert_eq!(val(eval(&cx, v4i8, &bc, &Flags::NONE, &[iv(32, 0x0403_0201)])), ivec(8, &[1, 2, 3, 4]));
    // A poison byte poisons only the halfword it lands in.
    let src = vecv(vec![iv(8, 1), iv(8, 2), SemValue::Poison, iv(8, 4)]);
    assert_eq!(
        val(eval(&cx, v2i16, &bc, &Flags::NONE, std::slice::from_ref(&src))),
        vecv(vec![iv(16, 0x0201), SemValue::Poison])
    );
    assert_eq!(val(eval(&cx, i32t, &bc, &Flags::NONE, &[src])), SemValue::Poison);
    // Floats reinterpret their IEEE bits.
    let f32t = cx.float(FloatKind::F32);
    let v2f = cx.vector(f32t, 2);
    let i64t = cx.int(64);
    let r = val(eval(&cx, i64t, &bc, &Flags::NONE, &[vecv(vec![f32v(1.0), f32v(-2.0)])]));
    assert_eq!(r, SemValue::int(64, Int::from_u64((u64::from((-2.0f32).to_bits()) << 32) | u64::from(1.0f32.to_bits()))));
    let back = val(eval(&cx, v2f, &bc, &Flags::NONE, &[r]));
    assert_eq!(back, vecv(vec![f32v(1.0), f32v(-2.0)]));
    // <8 x i1> <-> i8: one bit per lane.
    let b1 = cx.bool();
    let m8 = cx.vector(b1, 8);
    let r = val(eval(&cx, m8, &bc, &Flags::NONE, &[iv(8, 0b1000_0101)]));
    assert_eq!(r, vecv([1, 0, 1, 0, 0, 0, 0, 1].iter().map(|&b| iv(1, b)).collect()));
}

#[test]
fn lanewise_casts() {
    let mut cx = TypeContext::new();
    let i8t = cx.int(8);
    let i32t = cx.int(32);
    let f32t = cx.float(FloatKind::F32);
    let v4i32 = cx.vector(i32t, 4);
    let v4i8 = cx.vector(i8t, 4);
    let v4f = cx.vector(f32t, 4);
    let src = ivec(8, &[-1, 2, -128, 127]);
    let r = val(eval(&cx, v4i32, &InstKind::Cast(CastOp::SExt), &Flags::NONE, std::slice::from_ref(&src)));
    assert_eq!(r, ivec(32, &[-1, 2, -128, 127]));
    let r = val(eval(&cx, v4i32, &InstKind::Cast(CastOp::ZExt), &Flags::NONE, &[src]));
    assert_eq!(r, ivec(32, &[255, 2, 128, 127]));
    let r = val(eval(&cx, v4i8, &InstKind::Cast(CastOp::Trunc), &Flags::NONE, &[ivec(32, &[0x1FF, 2, 256, -1])]));
    assert_eq!(r, ivec(8, &[-1, 2, 0, -1]));
    // An out-of-range fptosi lane is poison, the others convert.
    let fv = vecv(vec![f32v(1.5), f32v(-2.5), f32v(3.0e9), f32v(f32::NAN)]);
    let r = val(eval(&cx, v4i32, &InstKind::Cast(CastOp::FpToSi), &Flags::NONE, &[fv]));
    assert_eq!(r, vecv(vec![iv(32, 1), iv(32, -2), SemValue::Poison, SemValue::Poison]));
    let r = val(eval(&cx, v4f, &InstKind::Cast(CastOp::SiToFp), &Flags::NONE, &[ivec(32, &[1, -2, 3, 0])]));
    assert_eq!(r, vecv(vec![f32v(1.0), f32v(-2.0), f32v(3.0), f32v(0.0)]));
    let r = val(eval(&cx, v4f, &InstKind::Unary(UnaryOp::FNeg), &Flags::NONE, &[vecv(vec![f32v(1.0), f32v(-0.0), SemValue::Poison, f32v(2.0)])]));
    assert_eq!(r, vecv(vec![f32v(-1.0), f32v(0.0), SemValue::Poison, f32v(-2.0)]));
}

#[test]
fn refinement_is_lanewise() {
    let src = vecv(vec![iv(32, 1), SemValue::Poison]);
    assert!(ivec(32, &[1, 99]).refines(&src));
    assert!(!ivec(32, &[2, 99]).refines(&src));
    assert!(ivec(32, &[5, 5]).refines(&SemValue::Poison));
}

// ---------------------------------------------------------------------------
// Randomized differential: vector op vs. lane-by-lane scalar op vs. a native
// model.
// ---------------------------------------------------------------------------

const INT_BINOPS: [BinOp; 21] = [
    BinOp::SMin,
    BinOp::SMax,
    BinOp::UMin,
    BinOp::UMax,
    BinOp::SAddSat,
    BinOp::UAddSat,
    BinOp::SSubSat,
    BinOp::USubSat,
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::UDiv,
    BinOp::SDiv,
    BinOp::URem,
    BinOp::SRem,
    BinOp::And,
    BinOp::Or,
    BinOp::Xor,
    BinOp::Shl,
    BinOp::LShr,
    BinOp::AShr,
];

const INT_PREDS: [IntPred; 10] = [
    IntPred::Eq,
    IntPred::Ne,
    IntPred::Ugt,
    IntPred::Uge,
    IntPred::Ult,
    IntPred::Ule,
    IntPred::Sgt,
    IntPred::Sge,
    IntPred::Slt,
    IntPred::Sle,
];

/// A random lane value of `width` bits biased toward edge cases, or poison
/// with probability 1/8.
fn rand_lane(rng: &mut Rng, width: u32) -> SemValue {
    if rng.below(8) == 0 {
        return SemValue::Poison;
    }
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    let raw = match rng.below(6) {
        0 => 0,
        1 => mask,                     // -1
        2 => 1u64 << (width - 1),      // INT_MIN
        3 => rng.below(u64::from(width) + 2), // small (good shift amounts)
        _ => rng.next(),
    } & mask;
    SemValue::int(width, Int::from_u64(raw))
}

/// A native-Rust model of a scalar integer binop on `width`-bit patterns:
/// `Some(Some(bits))` a value, `Some(None)` poison, `None` UB.
fn native_bin(op: BinOp, width: u32, a: u64, b: u64) -> Option<Option<u64>> {
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    let sext = |x: u64| -> i64 { ((x << (64 - width)) as i64) >> (64 - width) };
    let (sa, sb) = (sext(a), sext(b));
    let min = if width == 64 { i64::MIN } else { -(1i64 << (width - 1)) };
    let r = match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::UDiv | BinOp::URem if b == 0 => return None,
        BinOp::UDiv => a / b,
        BinOp::URem => a % b,
        BinOp::SDiv | BinOp::SRem if sb == 0 || (sa == min && sb == -1) => return None,
        BinOp::SDiv => sa.wrapping_div(sb) as u64,
        BinOp::SRem => sa.wrapping_rem(sb) as u64,
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl | BinOp::LShr | BinOp::AShr if b >= u64::from(width) => return Some(None),
        BinOp::Shl => a << b,
        BinOp::LShr => a >> b,
        BinOp::AShr => (sa >> b) as u64,
        BinOp::SMin => if sa <= sb { a } else { b },
        BinOp::SMax => if sa >= sb { a } else { b },
        BinOp::UMin => a.min(b),
        BinOp::UMax => a.max(b),
        BinOp::SAddSat | BinOp::SSubSat => {
            let max = if width == 64 { i64::MAX } else { (1i64 << (width - 1)) - 1 };
            let exact = if op == BinOp::SAddSat { i128::from(sa) + i128::from(sb) } else { i128::from(sa) - i128::from(sb) };
            exact.clamp(i128::from(min), i128::from(max)) as i64 as u64
        }
        BinOp::UAddSat => a.checked_add(b).filter(|&s| s <= mask).unwrap_or(mask),
        BinOp::USubSat => a.saturating_sub(b),
        _ => unreachable!(),
    };
    Some(Some(r & mask))
}

fn bits_of(v: &SemValue) -> Option<u64> {
    match v {
        SemValue::Int { bits, .. } => bits.to_u64(),
        _ => None,
    }
}

#[test]
fn randomized_vector_int_ops_match_lane_by_lane_scalar_ops() {
    let mut rng = Rng(0x5eed_1234);
    let mut cx = TypeContext::new();
    let b1 = cx.bool();
    for &width in &[8u32, 16, 32, 64] {
        let elem = cx.int(width);
        for &lanes in &[1u32, 2, 3, 4, 8, 16] {
            let vty = cx.vector(elem, lanes);
            let mty = cx.vector(b1, lanes);
            for _ in 0..40 {
                let a: Vec<SemValue> = (0..lanes).map(|_| rand_lane(&mut rng, width)).collect();
                let b: Vec<SemValue> = (0..lanes).map(|_| rand_lane(&mut rng, width)).collect();
                let (va, vb) = (vecv(a.clone()), vecv(b.clone()));
                for op in INT_BINOPS {
                    let vec_out = eval(&cx, vty, &InstKind::Bin(op), &Flags::NONE, &[va.clone(), vb.clone()]);
                    // Lane by lane through the scalar evaluator.
                    let mut ub = false;
                    let mut want = Vec::new();
                    for i in 0..lanes as usize {
                        match eval(&cx, elem, &InstKind::Bin(op), &Flags::NONE, &[a[i].clone(), b[i].clone()]) {
                            EvalOutcome::Value(v) => want.push(v),
                            EvalOutcome::UndefinedBehavior => ub = true,
                        }
                        // ...and the scalar evaluator against the native model
                        // (on defined lanes).
                        if let (Some(x), Some(y)) = (bits_of(&a[i]), bits_of(&b[i])) {
                            let got = eval(&cx, elem, &InstKind::Bin(op), &Flags::NONE, &[a[i].clone(), b[i].clone()]);
                            let model = native_bin(op, width, x, y);
                            match (got, model) {
                                (EvalOutcome::UndefinedBehavior, None) => {}
                                (EvalOutcome::Value(SemValue::Poison), Some(None)) => {}
                                (EvalOutcome::Value(v), Some(Some(m))) => {
                                    assert_eq!(bits_of(&v), Some(m), "{op:?} i{width} {x:#x},{y:#x}")
                                }
                                (g, m) => panic!("{op:?} i{width} {x:#x},{y:#x}: evaluator {g:?} vs model {m:?}"),
                            }
                        }
                    }
                    let expect = if ub { EvalOutcome::UndefinedBehavior } else { EvalOutcome::Value(vecv(want)) };
                    assert_eq!(vec_out, expect, "{op:?} on <{lanes} x i{width}>");
                }
                for pred in INT_PREDS {
                    let got = val(eval(&cx, mty, &InstKind::ICmp(pred), &Flags::NONE, &[va.clone(), vb.clone()]));
                    let want: Vec<SemValue> = (0..lanes as usize)
                        .map(|i| val(eval(&cx, b1, &InstKind::ICmp(pred), &Flags::NONE, &[a[i].clone(), b[i].clone()])))
                        .collect();
                    assert_eq!(got, vecv(want), "icmp {pred:?} on <{lanes} x i{width}>");
                }
                // Reductions against a native fold (skipped when a lane is poison,
                // where the result is poison).
                let defined: Option<Vec<u64>> = a.iter().map(bits_of).collect();
                for op in [ReduceOp::Add, ReduceOp::Mul, ReduceOp::And, ReduceOp::Or, ReduceOp::Xor, ReduceOp::SMin, ReduceOp::SMax, ReduceOp::UMin, ReduceOp::UMax] {
                    let got = val(eval(&cx, elem, &InstKind::Reduce(op), &Flags::NONE, std::slice::from_ref(&va)));
                    match &defined {
                        None => assert_eq!(got, SemValue::Poison),
                        Some(xs) => {
                            let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
                            let sext = |x: u64| -> i64 { ((x << (64 - width)) as i64) >> (64 - width) };
                            let r = xs[1..].iter().fold(xs[0], |acc, &x| match op {
                                ReduceOp::Add => acc.wrapping_add(x) & mask,
                                ReduceOp::Mul => acc.wrapping_mul(x) & mask,
                                ReduceOp::And => acc & x,
                                ReduceOp::Or => acc | x,
                                ReduceOp::Xor => acc ^ x,
                                ReduceOp::SMin => if sext(acc) <= sext(x) { acc } else { x },
                                ReduceOp::SMax => if sext(acc) >= sext(x) { acc } else { x },
                                ReduceOp::UMin => acc.min(x),
                                ReduceOp::UMax => acc.max(x),
                                _ => unreachable!(),
                            });
                            assert_eq!(bits_of(&got), Some(r), "reduce {op:?} on <{lanes} x i{width}>");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn randomized_vector_float_ops_match_lane_by_lane_native_floats() {
    let mut rng = Rng(0xf10a7);
    let mut cx = TypeContext::new();
    let f32t = cx.float(FloatKind::F32);
    let f64t = cx.float(FloatKind::F64);
    let b1 = cx.bool();
    let specials = [0.0f64, -0.0, 1.0, -1.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e30, 3.25];
    let rand_f = |rng: &mut Rng| -> f64 {
        if rng.below(3) == 0 {
            specials[rng.below(specials.len() as u64) as usize]
        } else {
            (rng.next() as i64 as f64) / 1e9
        }
    };
    for (elem, is64) in [(f32t, false), (f64t, true)] {
        for &lanes in &[2u32, 4] {
            let vty = cx.vector(elem, lanes);
            let mty = cx.vector(b1, lanes);
            for _ in 0..200 {
                let xs: Vec<f64> = (0..lanes).map(|_| rand_f(&mut rng)).collect();
                let ys: Vec<f64> = (0..lanes).map(|_| rand_f(&mut rng)).collect();
                let enc = |x: f64| {
                    if is64 { SemValue::Float(FloatBits::F64(x.to_bits())) } else { f32v(x as f32) }
                };
                let va = vecv(xs.iter().map(|&x| enc(x)).collect());
                let vb = vecv(ys.iter().map(|&y| enc(y)).collect());
                for op in [BinOp::FAdd, BinOp::FSub, BinOp::FMul, BinOp::FDiv] {
                    let got = val(eval(&cx, vty, &InstKind::Bin(op), &Flags::NONE, &[va.clone(), vb.clone()]));
                    let want: Vec<SemValue> = xs
                        .iter()
                        .zip(&ys)
                        .map(|(&x, &y)| {
                            if is64 {
                                let r = match op {
                                    BinOp::FAdd => x + y,
                                    BinOp::FSub => x - y,
                                    BinOp::FMul => x * y,
                                    _ => x / y,
                                };
                                SemValue::Float(FloatBits::F64(r.to_bits()))
                            } else {
                                let (x, y) = (x as f32, y as f32);
                                f32v(match op {
                                    BinOp::FAdd => x + y,
                                    BinOp::FSub => x - y,
                                    BinOp::FMul => x * y,
                                    _ => x / y,
                                })
                            }
                        })
                        .collect();
                    // NaN payloads may differ; compare NaN-ness then bits.
                    let (SemValue::Vector(g), w) = (got, want) else { panic!("not a vector") };
                    for (gl, wl) in g.iter().zip(&w) {
                        match (gl, wl) {
                            (SemValue::Float(p), SemValue::Float(q)) if super::decode(*p).is_nan() => {
                                assert!(super::decode(*q).is_nan());
                            }
                            _ => assert_eq!(gl, wl, "{op:?}"),
                        }
                    }
                }
                for pred in [FloatPred::Oeq, FloatPred::Olt, FloatPred::Ule, FloatPred::Une, FloatPred::Uno, FloatPred::One] {
                    let got = val(eval(&cx, mty, &InstKind::FCmp(pred), &Flags::NONE, &[va.clone(), vb.clone()]));
                    let want: Vec<SemValue> = (0..lanes as usize)
                        .map(|i| val(eval(&cx, b1, &InstKind::FCmp(pred), &Flags::NONE, &[va.lane(i), vb.lane(i)])))
                        .collect();
                    assert_eq!(got, vecv(want));
                }
                // Ordered sum == a native left fold.
                let got = val(eval(&cx, elem, &InstKind::Reduce(ReduceOp::FAdd), &Flags::NONE, std::slice::from_ref(&va)));
                let want = if is64 {
                    SemValue::Float(FloatBits::F64(xs[1..].iter().fold(xs[0], |a, &x| a + x).to_bits()))
                } else {
                    let v: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
                    f32v(v[1..].iter().fold(v[0], |a, &x| a + x))
                };
                match (&got, &want) {
                    (SemValue::Float(p), SemValue::Float(q)) if super::decode(*q).is_nan() => {
                        assert!(super::decode(*p).is_nan())
                    }
                    _ => assert_eq!(got, want),
                }
            }
        }
    }
}

#[test]
fn randomized_shuffles_match_a_lane_picking_model() {
    let mut rng = Rng(77);
    let mut cx = TypeContext::new();
    let i16t = cx.int(16);
    for &n in &[1u32, 2, 4, 8] {
        for _ in 0..50 {
            let a: Vec<SemValue> = (0..n).map(|_| rand_lane(&mut rng, 16)).collect();
            let b: Vec<SemValue> = (0..n).map(|_| rand_lane(&mut rng, 16)).collect();
            let m = 1 + rng.below(8) as u32;
            let mask: Vec<u32> = (0..m).map(|_| rng.below(u64::from(2 * n)) as u32).collect();
            let rty: TypeId = cx.vector(i16t, m);
            let got = val(eval(&cx, rty, &InstKind::ShuffleVector(mask.clone().into()), &Flags::NONE, &[vecv(a.clone()), vecv(b.clone())]));
            let cat: Vec<SemValue> = a.iter().chain(&b).cloned().collect();
            let want: Vec<SemValue> = mask.iter().map(|&k| cat[k as usize].clone()).collect();
            assert_eq!(got, vecv(want));
        }
    }
}
