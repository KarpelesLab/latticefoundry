//! Executable reference semantics for the IR's value-producing operations
//! (bet **B1**, tenet **T2**: the opcode table *is* a formal semantics).
//!
//! This module is the ground-truth **concrete interpreter** for every *pure*,
//! value-producing opcode: it maps operand [`SemValue`]s to a result, modeling
//! **poison**, **freeze**, and **undefined behavior (UB)** exactly, with
//! width-correct two's-complement integer arithmetic done in [`puremp::Int`] at
//! full precision. It has two jobs:
//!
//! 1. It is the specification the `///` prose on each opcode in
//!    [`crate::ir::inst`] is checked against, so spec and implementation cannot
//!    drift (B1).
//! 2. It is the oracle for constant folding ([`fold`], Phase 4) and the
//!    reference the Phase 2 symbolic `z3rs` translation is validated against.
//!
//! ## Poison vs. UB (the two failure modes)
//!
//! The refinement contract (`docs/ir-design.md` §5) is *conditional on the
//! source triggering no UB*, so poison and UB are genuinely different outcomes:
//!
//! - **Poison** is an ordinary [`SemValue`] — a deferred error that taints any
//!   dependent op. Producing poison is *not* a failure of evaluation; it is a
//!   value. Any poison operand yields a poison result (except `freeze`, and
//!   `select` when only a non-selected arm is poison).
//! - **Undefined behavior** ([`EvalOutcome::UndefinedBehavior`]) is a *distinct*
//!   outcome: the operation has no defined meaning at all. `fold` must refuse to
//!   fold a UB instruction (folding it away would be unsound).
//!
//! ### The UB set (resolving `docs/ir-design.md` §10)
//!
//! We keep UB as small as possible. Among the pure ops evaluated here, the
//! **only** UB cases are integer division/remainder that has no defined result:
//!
//! - `udiv`/`sdiv`/`urem`/`srem` **by zero**;
//! - `sdiv`/`srem` of `INT_MIN` by `-1` (the quotient `-INT_MIN` is not
//!   representable — signed-division overflow).
//!
//! Everything else that could "go wrong" is **poison, not UB** (tenet §7):
//! `nsw`/`nuw` overflow, `exact` violations, over-wide shifts, and
//! out-of-range float→int conversions all produce poison. This is the smallest
//! UB surface that still lets `sdiv`/`srem` avoid defining a nonexistent
//! result, and it is what gives [`EvalOutcome`] its two-case shape.
//!
//! > Note: the `///` prose on the four div/rem opcodes in [`crate::ir::inst`]
//! > has been reconciled with this resolution (div/rem faults are UB, per the
//! > LLVM convention and the design of [`EvalOutcome`]).
//!
//! ## What is *not* here
//!
//! Stateful and positional ops — `Alloca`, `Load`, `Store`, `Call`, and the
//! terminators `Ret`/`Br`/`CondBr`/`Switch`/`Unreachable` — are about memory
//! and control-flow *state*, not pure value production. They belong to the
//! interpreter / verifier layer built in a later phase, and [`eval`] panics if
//! asked to evaluate one. `PtrAdd` *is* handled: it is pure address arithmetic.
//!
//! ## Overflow / exactness detection
//!
//! All integer values are carried as their **unsigned bit pattern** — a
//! [`puremp::Int`] normalized to `[0, 2ⁿ)` for an `n`-bit type. To decide a flag
//! violation we compute the *exact, full-precision* result and test it against
//! the `n`-bit range:
//!
//! - `nsw` (no signed wrap): the exact signed result `sₐ OP s_b` must lie in
//!   `[-2ⁿ⁻¹, 2ⁿ⁻¹)`; otherwise poison.
//! - `nuw` (no unsigned wrap): the exact unsigned result `uₐ OP u_b` must lie in
//!   `[0, 2ⁿ)`; otherwise poison.
//! - `exact` (div/shift): the operation must lose no information — a zero
//!   remainder for `udiv`/`sdiv`, and no set bit shifted out for `lshr`/`ashr`.
//!
//! Because the check is done in unbounded precision and only *then* reduced
//! modulo `2ⁿ` (via [`puremp::Int::mod_2k`], which yields the Euclidean low `n`
//! bits — exactly the two's-complement pattern), the semantics are exact and
//! host-independent for any width.

use crate::ir::inst::{BinOp, CastOp, Flags, FloatPred, InstKind, IntPred, UnaryOp};
use crate::ir::types::{FloatKind, Type, TypeContext, TypeId};
use crate::ir::value::{Const, FloatBits};

use puremp::{Float, Int, RoundingMode};

/// The width, in bits, of the abstract pointer address model.
///
/// Pointers are modeled as an opaque byte address held in `[0, 2⁶⁴)`, matching
/// the provisional 64-bit data layout in [`crate::ir::types`]. This is enough
/// for the pure address arithmetic (`ptr_add`, `ptrtoint`, `inttoptr`) the
/// evaluator must model; a richer provenance model is bet B10, deferred.
const POINTER_BITS: u32 = 64;

/// A concrete semantic value: the runtime denotation of an SSA value.
///
/// Integers are stored as their **unsigned two's-complement bit pattern** — a
/// [`puremp::Int`] normalized to `[0, 2^width)` — so the same value serves both
/// the signed and unsigned readings (an operation picks the interpretation it
/// wants). Floats are the exact IEEE bit pattern ([`FloatBits`]). Pointers are
/// an abstract byte address. [`SemValue::Poison`] is the (typeless) deferred
/// error of `docs/ir-design.md` §5.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SemValue {
    /// An `n`-bit integer, held as its unsigned bit pattern in `[0, 2ⁿ)`.
    Int {
        /// The bit width `n`.
        width: u32,
        /// The unsigned representative of the bit pattern, in `[0, 2ⁿ)`.
        bits: Int,
    },
    /// A floating-point value, as its exact IEEE-754 bit pattern.
    Float(FloatBits),
    /// An abstract pointer: an opaque byte address in `[0, 2⁶⁴)`.
    Ptr(Int),
    /// Poison — a deferred error that taints any dependent operation.
    Poison,
}

impl SemValue {
    /// An `n`-bit integer from an arbitrary [`puremp::Int`], normalized to its
    /// unsigned bit pattern in `[0, 2ⁿ)`.
    pub fn int(width: u32, raw: Int) -> SemValue {
        SemValue::Int { width, bits: mask(&raw, width) }
    }

    /// The one-bit boolean value `0`/`1` (an `i1`).
    pub fn boolean(b: bool) -> SemValue {
        SemValue::Int { width: 1, bits: if b { Int::ONE } else { Int::ZERO } }
    }

    /// An abstract pointer at the given byte address (normalized to 64 bits).
    pub fn ptr(addr: Int) -> SemValue {
        SemValue::Ptr(mask(&addr, POINTER_BITS))
    }

    /// Whether this value is poison.
    pub fn is_poison(&self) -> bool {
        matches!(self, SemValue::Poison)
    }

    /// The `(width, bits)` of an integer value, or `None` for other kinds.
    fn as_int(&self) -> Option<(u32, &Int)> {
        match self {
            SemValue::Int { width, bits } => Some((*width, bits)),
            _ => None,
        }
    }

    /// The IEEE bit pattern of a float value, or `None` for other kinds.
    fn as_float(&self) -> Option<FloatBits> {
        match self {
            SemValue::Float(b) => Some(*b),
            _ => None,
        }
    }

    /// This value re-read as an integer operand: pointers become a 64-bit
    /// unsigned address, integers pass through. Used by `icmp`, which the IR
    /// permits on pointers as well as integers.
    fn to_int_operand(&self) -> Option<(u32, Int)> {
        match self {
            SemValue::Int { width, bits } => Some((*width, bits.clone())),
            SemValue::Ptr(addr) => Some((POINTER_BITS, addr.clone())),
            _ => None,
        }
    }
}

/// The outcome of evaluating one instruction.
///
/// [`EvalOutcome::Value`] carries a produced [`SemValue`] — which may itself be
/// [`SemValue::Poison`]. [`EvalOutcome::UndefinedBehavior`] is the *distinct*
/// outcome that the operation has no defined meaning (see the module docs for
/// the exact UB set). Keeping UB separate from poison is what makes the
/// refinement contract (precondition: "source triggers no UB") expressible.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum EvalOutcome {
    /// A produced value (possibly poison).
    Value(SemValue),
    /// The operation is undefined behavior on these operands.
    UndefinedBehavior,
}

/// The result of constant-folding a pure instruction ([`fold`]).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FoldResult {
    /// The instruction folds to this constant (possibly [`Const::Poison`]).
    Folded(Const),
    /// The instruction would be undefined behavior on these constant operands;
    /// the caller **must not** fold it (doing so would be unsound).
    WouldBeUb,
}

// ---------------------------------------------------------------------------
// Small numeric helpers (all exact, all width-parameterized).
// ---------------------------------------------------------------------------

/// `2^n` as a non-negative [`puremp::Int`].
fn two_pow(n: u32) -> Int {
    Int::ONE.mul_2k(n)
}

/// The unsigned bit pattern of `v` in an `n`-bit type: the Euclidean low `n`
/// bits, i.e. `v mod 2ⁿ` in `[0, 2ⁿ)` — exactly the two's-complement encoding.
fn mask(v: &Int, n: u32) -> Int {
    if n == 0 { Int::ZERO } else { v.mod_2k(n) }
}

/// The signed value of an `n`-bit pattern `bits` (which must be in `[0, 2ⁿ)`).
fn signed(bits: &Int, n: u32) -> Int {
    if n > 0 && bits.bit(n - 1) { bits.sub(&two_pow(n)) } else { bits.clone() }
}

/// The most negative signed `n`-bit value, `-2ⁿ⁻¹`.
fn int_min(n: u32) -> Int {
    two_pow(n - 1).neg()
}

/// The integer type width of `ty`, or `None` if it is not an integer type.
fn int_width(types: &TypeContext, ty: TypeId) -> Option<u32> {
    match types.get(ty) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}

/// The float format of `ty`, or `None` if it is not a float type.
fn float_kind(types: &TypeContext, ty: TypeId) -> Option<FloatKind> {
    match types.get(ty) {
        Type::Float(k) => Some(*k),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Top-level evaluation.
// ---------------------------------------------------------------------------

/// Evaluate a **pure, value-producing** instruction on concrete operands.
///
/// `result_ty` is the instruction's result type; it supplies the target width /
/// float format that casts and `freeze` need (operand types come from the
/// [`SemValue`]s themselves). `types` resolves `result_ty`.
///
/// # Panics
///
/// Panics if `kind` is a stateful or terminator opcode (`Alloca`, `Load`,
/// `Store`, `Call`, `Ret`, `Br`, `CondBr`, `Switch`, `Unreachable`); those are
/// not pure value production and are handled by the interpreter/verifier layer,
/// not here. Callers must only pass value-producing opcodes.
pub fn eval(
    types: &TypeContext,
    result_ty: TypeId,
    kind: &InstKind,
    flags: &Flags,
    operands: &[SemValue],
) -> EvalOutcome {
    match kind {
        // `select` and `freeze` have bespoke poison rules; everything else
        // propagates poison from any operand.
        InstKind::Select => eval_select(operands),
        InstKind::Freeze => eval_freeze(types, result_ty, operands),

        _ if operands.iter().any(SemValue::is_poison) => EvalOutcome::Value(SemValue::Poison),

        InstKind::Bin(op) => eval_bin(*op, flags, operands),
        InstKind::Unary(op) => eval_unary(*op, flags, operands),
        InstKind::ICmp(pred) => eval_icmp(*pred, operands),
        InstKind::FCmp(pred) => eval_fcmp(*pred, flags, operands),
        InstKind::Cast(op) => eval_cast(types, result_ty, *op, operands),
        InstKind::PtrAdd { .. } => eval_ptr_add(operands),

        InstKind::Alloca { .. }
        | InstKind::Load { .. }
        | InstKind::Store { .. }
        | InstKind::Call
        | InstKind::Ret
        | InstKind::Br(_)
        | InstKind::CondBr { .. }
        | InstKind::Switch(_)
        | InstKind::Unreachable => {
            panic!("semantics::eval called on a non-value-producing opcode: {kind:?}")
        }
    }
}

/// Wrap a produced value.
#[inline]
fn ok(v: SemValue) -> EvalOutcome {
    EvalOutcome::Value(v)
}

/// A poison result.
#[inline]
fn poison() -> EvalOutcome {
    EvalOutcome::Value(SemValue::Poison)
}

// ---------------------------------------------------------------------------
// Binary integer & float ops.
// ---------------------------------------------------------------------------

fn eval_bin(op: BinOp, flags: &Flags, operands: &[SemValue]) -> EvalOutcome {
    if op.is_float() {
        return eval_fbin(op, flags, operands);
    }
    let (Some((w, a)), Some((wb, b))) =
        (operands.first().and_then(SemValue::as_int), operands.get(1).and_then(SemValue::as_int))
    else {
        return poison();
    };
    if w != wb {
        return poison();
    }
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul => eval_addsubmul(op, flags, w, a, b),
        BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => eval_divrem(op, flags, w, a, b),
        BinOp::And => ok(SemValue::int(w, a.bitand(b))),
        BinOp::Or => ok(SemValue::int(w, a.bitor(b))),
        BinOp::Xor => ok(SemValue::int(w, a.bitxor(b))),
        BinOp::Shl | BinOp::LShr | BinOp::AShr => eval_shift(op, flags, w, a, b),
        // Float variants handled above.
        BinOp::FAdd
        | BinOp::FSub
        | BinOp::FMul
        | BinOp::FDiv
        | BinOp::FRem => unreachable!("float op routed to integer path"),
    }
}

/// `add`/`sub`/`mul`, with exact `nsw`/`nuw` overflow detection.
fn eval_addsubmul(op: BinOp, flags: &Flags, w: u32, a: &Int, b: &Int) -> EvalOutcome {
    let (ua, ub) = (a, b);
    let (sa, sb) = (signed(a, w), signed(b, w));
    let (u_full, s_full) = match op {
        BinOp::Add => (ua.add(ub), sa.add(&sb)),
        BinOp::Sub => (ua.sub(ub), sa.sub(&sb)),
        BinOp::Mul => (ua.mul(ub), sa.mul(&sb)),
        _ => unreachable!(),
    };
    if flags.nsw && (s_full < int_min(w) || s_full >= two_pow(w - 1)) {
        return poison();
    }
    if flags.nuw && (u_full.is_negative() || u_full >= two_pow(w)) {
        return poison();
    }
    ok(SemValue::int(w, s_full))
}

/// `udiv`/`sdiv`/`urem`/`srem`. Division/remainder by zero and the
/// `INT_MIN / -1` signed overflow are **UB**; a violated `exact` flag is poison.
fn eval_divrem(op: BinOp, flags: &Flags, w: u32, a: &Int, b: &Int) -> EvalOutcome {
    match op {
        BinOp::UDiv | BinOp::URem => {
            if b.is_zero() {
                return EvalOutcome::UndefinedBehavior;
            }
            let (q, r) = a.div_rem_trunc(b); // a, b are non-negative bit patterns
            if op == BinOp::UDiv {
                if flags.exact && !r.is_zero() {
                    return poison();
                }
                ok(SemValue::int(w, q))
            } else {
                ok(SemValue::int(w, r))
            }
        }
        BinOp::SDiv | BinOp::SRem => {
            let (sa, sb) = (signed(a, w), signed(b, w));
            if sb.is_zero() {
                return EvalOutcome::UndefinedBehavior;
            }
            // INT_MIN / -1 overflows (quotient not representable): UB for both
            // sdiv and srem (the paired division has no result).
            if sa == int_min(w) && sb == Int::MINUS_ONE {
                return EvalOutcome::UndefinedBehavior;
            }
            let (q, r) = sa.div_rem_trunc(&sb);
            if op == BinOp::SDiv {
                if flags.exact && !r.is_zero() {
                    return poison();
                }
                ok(SemValue::int(w, q))
            } else {
                ok(SemValue::int(w, r))
            }
        }
        _ => unreachable!(),
    }
}

/// `shl`/`lshr`/`ashr`. A shift amount `≥ width` is poison; `nsw`/`nuw` on
/// `shl` and `exact` on the right shifts follow the flag model.
fn eval_shift(op: BinOp, flags: &Flags, w: u32, a: &Int, b: &Int) -> EvalOutcome {
    // The shift amount is read unsigned from `b`.
    if b >= &width_as_int(w) {
        return poison();
    }
    // Safe: b < w ≤ u32::MAX, so it fits a u32.
    let k = b.to_u64().expect("shift amount < width fits u64") as u32;
    match op {
        BinOp::Shl => {
            let u_full = a.mul_2k(k); // unsigned a · 2^k, exact
            if flags.nuw && u_full >= two_pow(w) {
                return poison();
            }
            if flags.nsw {
                let s_full = signed(a, w).mul_2k(k);
                if s_full < int_min(w) || s_full >= two_pow(w - 1) {
                    return poison();
                }
            }
            ok(SemValue::int(w, u_full))
        }
        BinOp::LShr => {
            if flags.exact && !a.mod_2k(k).is_zero() {
                return poison();
            }
            ok(SemValue::int(w, a.div_2k_trunc(k))) // a ≥ 0, so trunc == floor
        }
        BinOp::AShr => {
            if flags.exact && !a.mod_2k(k).is_zero() {
                return poison();
            }
            // Arithmetic right shift = floor(signed(a) / 2^k).
            let res = signed(a, w).div_floor(&two_pow(k));
            ok(SemValue::int(w, res))
        }
        _ => unreachable!(),
    }
}

/// The type width as an `Int` (the bound a shift amount must stay below).
fn width_as_int(w: u32) -> Int {
    Int::from_u64(u64::from(w))
}

/// Floating-point `fadd`/`fsub`/`fmul`/`fdiv`/`frem`.
fn eval_fbin(op: BinOp, flags: &Flags, operands: &[SemValue]) -> EvalOutcome {
    let (Some(a), Some(b)) =
        (operands.first().and_then(SemValue::as_float), operands.get(1).and_then(SemValue::as_float))
    else {
        return poison();
    };
    let kind = float_kind_of(a);
    let (av, bv) = (decode(a), decode(b));
    let rv = match op {
        BinOp::FAdd => native_binop(kind, a, b, |x, y| x + y),
        BinOp::FSub => native_binop(kind, a, b, |x, y| x - y),
        BinOp::FMul => native_binop(kind, a, b, |x, y| x * y),
        BinOp::FDiv => native_binop(kind, a, b, |x, y| x / y),
        BinOp::FRem => native_binop(kind, a, b, |x, y| x % y), // C `fmod`, LLVM `frem`
        _ => unreachable!(),
    };
    if let Some(p) = fast_math_violation(flags, &[av, bv, decode(rv)]) {
        return p;
    }
    ok(SemValue::Float(rv))
}

// ---------------------------------------------------------------------------
// Unary float op.
// ---------------------------------------------------------------------------

fn eval_unary(op: UnaryOp, flags: &Flags, operands: &[SemValue]) -> EvalOutcome {
    match op {
        UnaryOp::FNeg => {
            let Some(v) = operands.first().and_then(SemValue::as_float) else {
                return poison();
            };
            if let Some(p) = fast_math_violation(flags, &[decode(v)]) {
                return p;
            }
            ok(SemValue::Float(fneg_bits(v)))
        }
    }
}

/// Flip the IEEE sign bit (fast-math `nsz` makes a zero's sign insignificant but
/// does not make the operation poison; the result is deterministic).
fn fneg_bits(v: FloatBits) -> FloatBits {
    match v {
        FloatBits::F16(b) => FloatBits::F16(b ^ 0x8000),
        FloatBits::F32(b) => FloatBits::F32(b ^ 0x8000_0000),
        FloatBits::F64(b) => FloatBits::F64(b ^ 0x8000_0000_0000_0000),
    }
}

// ---------------------------------------------------------------------------
// Comparisons.
// ---------------------------------------------------------------------------

fn eval_icmp(pred: IntPred, operands: &[SemValue]) -> EvalOutcome {
    let (Some((wa, a)), Some((wb, b))) = (
        operands.first().and_then(SemValue::to_int_operand),
        operands.get(1).and_then(SemValue::to_int_operand),
    ) else {
        return poison();
    };
    if wa != wb {
        return poison();
    }
    let w = wa;
    let result = match pred {
        IntPred::Eq => a == b,
        IntPred::Ne => a != b,
        IntPred::Ugt => a > b,
        IntPred::Uge => a >= b,
        IntPred::Ult => a < b,
        IntPred::Ule => a <= b,
        IntPred::Sgt => signed(&a, w) > signed(&b, w),
        IntPred::Sge => signed(&a, w) >= signed(&b, w),
        IntPred::Slt => signed(&a, w) < signed(&b, w),
        IntPred::Sle => signed(&a, w) <= signed(&b, w),
    };
    ok(SemValue::boolean(result))
}

fn eval_fcmp(pred: FloatPred, flags: &Flags, operands: &[SemValue]) -> EvalOutcome {
    let (Some(a), Some(b)) =
        (operands.first().and_then(SemValue::as_float), operands.get(1).and_then(SemValue::as_float))
    else {
        return poison();
    };
    let (x, y) = (decode(a), decode(b));
    if let Some(p) = fast_math_violation(flags, &[x, y]) {
        return p;
    }
    let uno = x.is_nan() || y.is_nan();
    let result = match pred {
        FloatPred::False => false,
        FloatPred::True => true,
        FloatPred::Ord => !uno,
        FloatPred::Uno => uno,
        FloatPred::Oeq => !uno && x == y,
        FloatPred::Ogt => !uno && x > y,
        FloatPred::Oge => !uno && x >= y,
        FloatPred::Olt => !uno && x < y,
        FloatPred::Ole => !uno && x <= y,
        FloatPred::One => !uno && x != y,
        FloatPred::Ueq => uno || x == y,
        FloatPred::Ugt => uno || x > y,
        FloatPred::Uge => uno || x >= y,
        FloatPred::Ult => uno || x < y,
        FloatPred::Ule => uno || x <= y,
        FloatPred::Une => uno || x != y,
    };
    ok(SemValue::boolean(result))
}

// ---------------------------------------------------------------------------
// Casts.
// ---------------------------------------------------------------------------

fn eval_cast(
    types: &TypeContext,
    result_ty: TypeId,
    op: CastOp,
    operands: &[SemValue],
) -> EvalOutcome {
    let Some(src) = operands.first() else {
        return poison();
    };
    match op {
        CastOp::Trunc | CastOp::ZExt | CastOp::SExt => {
            let (Some((sw, bits)), Some(tw)) =
                (src.as_int(), int_width(types, result_ty))
            else {
                return poison();
            };
            let out = match op {
                CastOp::Trunc => mask(bits, tw),
                CastOp::ZExt => bits.clone(), // already the unsigned value
                CastOp::SExt => mask(&signed(bits, sw), tw),
                _ => unreachable!(),
            };
            ok(SemValue::int(tw, out))
        }
        CastOp::FpTrunc | CastOp::FpExt => {
            let (Some(fb), Some(tk)) = (src.as_float(), float_kind(types, result_ty)) else {
                return poison();
            };
            ok(SemValue::Float(encode(decode(fb), tk)))
        }
        CastOp::FpToUi | CastOp::FpToSi => {
            let (Some(fb), Some(tw)) = (src.as_float(), int_width(types, result_ty)) else {
                return poison();
            };
            eval_fp_to_int(fb, tw, op == CastOp::FpToSi)
        }
        CastOp::UiToFp | CastOp::SiToFp => {
            let (Some((sw, bits)), Some(tk)) = (src.as_int(), float_kind(types, result_ty)) else {
                return poison();
            };
            let n = if op == CastOp::SiToFp { signed(bits, sw) } else { bits.clone() };
            ok(SemValue::Float(int_to_float(&n, tk)))
        }
        CastOp::PtrToInt => {
            let (SemValue::Ptr(addr), Some(tw)) = (src, int_width(types, result_ty)) else {
                return poison();
            };
            ok(SemValue::int(tw, addr.clone()))
        }
        CastOp::IntToPtr => {
            let Some((_, bits)) = src.as_int() else {
                return poison();
            };
            ok(SemValue::ptr(bits.clone()))
        }
        CastOp::Bitcast => eval_bitcast(types, result_ty, src),
    }
}

/// `fptoui`/`fptosi`: round toward zero to an integer; an out-of-range or
/// non-finite value is **poison**.
fn eval_fp_to_int(fb: FloatBits, tw: u32, signed_dst: bool) -> EvalOutcome {
    let x = decode(fb);
    if !x.is_finite() {
        return poison(); // NaN or ±∞ is out of every integer range
    }
    // Exact: an f64 has ≤ 53 significant bits, so precision 53 is lossless.
    let f = Float::from_f64(x, 53, RoundingMode::TowardZero);
    let Some(n) = f.trunc() else {
        return poison();
    };
    let in_range = if signed_dst {
        n >= int_min(tw) && n < two_pow(tw - 1)
    } else {
        !n.is_negative() && n < two_pow(tw)
    };
    if !in_range {
        return poison();
    }
    ok(SemValue::int(tw, n))
}

fn eval_bitcast(types: &TypeContext, result_ty: TypeId, src: &SemValue) -> EvalOutcome {
    match (src, types.get(result_ty)) {
        // integer ⇄ same-width float
        (SemValue::Int { width, bits }, Type::Float(k)) if *width == k.bit_width() => {
            let raw = bits.to_u64().unwrap_or(0);
            ok(SemValue::Float(bits_to_float(raw, *k)))
        }
        (SemValue::Float(fb), Type::Int(w)) if float_kind_of(*fb).bit_width() == *w => {
            ok(SemValue::int(*w, Int::from_u64(float_raw(*fb))))
        }
        // same-shape identities
        (SemValue::Int { width, .. }, Type::Int(w)) if width == w => ok(src.clone()),
        (SemValue::Float(fb), Type::Float(k)) if float_kind_of(*fb) == *k => {
            ok(SemValue::Float(*fb))
        }
        (SemValue::Ptr(_), Type::Ptr) => ok(src.clone()),
        // Any other bitcast is ill-typed for this evaluator.
        _ => poison(),
    }
}

// ---------------------------------------------------------------------------
// Select, freeze, ptr_add.
// ---------------------------------------------------------------------------

/// `select cond, t, f`. A poison **condition** yields poison; a poison value in
/// the *non-selected* arm does not taint the result (`docs/ir-design.md`,
/// [`InstKind::Select`]).
fn eval_select(operands: &[SemValue]) -> EvalOutcome {
    let (Some(cond), Some(t), Some(f)) =
        (operands.first(), operands.get(1), operands.get(2))
    else {
        return poison();
    };
    match cond {
        SemValue::Poison => poison(),
        SemValue::Int { bits, .. } => {
            if bits.is_zero() { ok(f.clone()) } else { ok(t.clone()) }
        }
        _ => poison(),
    }
}

/// `freeze`. On a defined operand it is the identity; on poison it produces a
/// **fixed, deterministic** concrete value of the type — we choose the all-zero
/// value (zero / positive-zero float / null pointer) so results are stable.
fn eval_freeze(types: &TypeContext, result_ty: TypeId, operands: &[SemValue]) -> EvalOutcome {
    let Some(v) = operands.first() else {
        return poison();
    };
    if !v.is_poison() {
        return ok(v.clone());
    }
    match types.get(result_ty) {
        Type::Int(w) => ok(SemValue::Int { width: *w, bits: Int::ZERO }),
        Type::Float(k) => ok(SemValue::Float(bits_to_float(0, *k))),
        Type::Ptr => ok(SemValue::ptr(Int::ZERO)),
        // Aggregates/void/func are out of scope for the scalar evaluator; a
        // poison of such a type is left as poison.
        _ => poison(),
    }
}

/// `ptr_add base, off`: byte-address arithmetic, `base + signed(off)` wrapped to
/// the pointer width. `inbounds` cannot be checked without a memory/allocation
/// model, so it is not enforced here (that check belongs to the memory-model
/// layer); poison only propagates from the operands.
fn eval_ptr_add(operands: &[SemValue]) -> EvalOutcome {
    let (Some(SemValue::Ptr(base)), Some((ow, off))) =
        (operands.first(), operands.get(1).and_then(SemValue::as_int))
    else {
        return poison();
    };
    let addr = base.add(&signed(off, ow));
    ok(SemValue::ptr(addr))
}

// ---------------------------------------------------------------------------
// Fast-math poison checks.
// ---------------------------------------------------------------------------

/// If a set fast-math assumption is violated by any of `vals` (operands and
/// result, as decoded `f64`s), return the poison outcome.
///
/// Only `nnan` ("never NaN") and `ninf` ("never ±∞") are *runtime* assumptions
/// that a concrete value can violate. `nsz`/`reassoc`/`contract`/`afn` license
/// transformations rather than constrain runtime values, so they never make the
/// reference result poison.
fn fast_math_violation(flags: &Flags, vals: &[f64]) -> Option<EvalOutcome> {
    let fm = flags.fast;
    if fm.nnan && vals.iter().any(|v| v.is_nan()) {
        return Some(poison());
    }
    if fm.ninf && vals.iter().any(|v| v.is_infinite()) {
        return Some(poison());
    }
    None
}

// ---------------------------------------------------------------------------
// Float codec: IEEE bit patterns ⇄ f64, with a self-contained binary16 codec.
// ---------------------------------------------------------------------------

fn float_kind_of(fb: FloatBits) -> FloatKind {
    match fb {
        FloatBits::F16(_) => FloatKind::F16,
        FloatBits::F32(_) => FloatKind::F32,
        FloatBits::F64(_) => FloatKind::F64,
    }
}

/// The raw IEEE bits of a [`FloatBits`], zero-extended to `u64`.
fn float_raw(fb: FloatBits) -> u64 {
    match fb {
        FloatBits::F16(b) => u64::from(b),
        FloatBits::F32(b) => u64::from(b),
        FloatBits::F64(b) => b,
    }
}

/// Build a [`FloatBits`] of `kind` from a raw `u64` bit pattern.
fn bits_to_float(raw: u64, kind: FloatKind) -> FloatBits {
    match kind {
        FloatKind::F16 => FloatBits::F16(raw as u16),
        FloatKind::F32 => FloatBits::F32(raw as u32),
        FloatKind::F64 => FloatBits::F64(raw),
    }
}

/// Decode an IEEE bit pattern to the exact `f64` it denotes. Every one of our
/// formats (binary16/32/64) embeds in binary64 exactly, so this is lossless.
fn decode(fb: FloatBits) -> f64 {
    match fb {
        FloatBits::F16(b) => f16_to_f64(b),
        FloatBits::F32(b) => f64::from(f32::from_bits(b)),
        FloatBits::F64(b) => f64::from_bits(b),
    }
}

/// Encode an `f64` value to the bit pattern of `kind`, rounding once to the
/// target format (round-to-nearest, ties to even).
fn encode(x: f64, kind: FloatKind) -> FloatBits {
    match kind {
        FloatKind::F16 => FloatBits::F16(f16_from_f64(x)),
        FloatKind::F32 => FloatBits::F32((x as f32).to_bits()),
        FloatKind::F64 => FloatBits::F64(x.to_bits()),
    }
}

/// Compute a binary float op natively in the *target* format (so there is no
/// double rounding), returning the result bits.
///
/// - `F64`: native `f64` arithmetic (IEEE-754 binary64, guaranteed by Rust).
/// - `F32`: native `f32` arithmetic.
/// - `F16`: computed in `f64` then rounded once to binary16 — harmless double
///   rounding, since binary64's 53-bit significand exceeds `2·11 + 2`.
fn native_binop(kind: FloatKind, a: FloatBits, b: FloatBits, f: fn(f64, f64) -> f64) -> FloatBits {
    match kind {
        FloatKind::F64 => FloatBits::F64(f(decode(a), decode(b)).to_bits()),
        FloatKind::F32 => {
            // `f` is defined on f64; apply it to the f32 values promoted losslessly,
            // then round once to f32. For +,−,×,÷,fmod this equals native f32.
            let (x, y) = (decode(a), decode(b));
            FloatBits::F32((f(x, y) as f32).to_bits())
        }
        FloatKind::F16 => FloatBits::F16(f16_from_f64(f(decode(a), decode(b)))),
    }
}

/// The IEEE binary-precision (significand bits) of a format: 11/24/53.
fn float_precision(kind: FloatKind) -> u64 {
    match kind {
        FloatKind::F16 => 11,
        FloatKind::F32 => 24,
        FloatKind::F64 => 53,
    }
}

/// Convert an integer to a float of `kind`, correctly rounded to nearest.
fn int_to_float(n: &Int, kind: FloatKind) -> FloatBits {
    let f = Float::from_int(n, float_precision(kind), RoundingMode::Nearest);
    match kind {
        // The `Float` already carries the target precision, so these are exact.
        FloatKind::F64 => FloatBits::F64(f.to_f64().to_bits()),
        FloatKind::F32 => FloatBits::F32(f.to_f32().to_bits()),
        // Integer inputs are never subnormal in f16, so encoding the exact value
        // (overflowing to ±∞ past f16's range) matches direct correct rounding.
        FloatKind::F16 => FloatBits::F16(f16_from_f64(f.to_f64())),
    }
}

/// Decode an IEEE binary16 bit pattern to the exact `f64` it denotes.
fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let mag = if exp == 0 {
        // Zero or subnormal: value = mant · 2⁻²⁴.
        f64::from(mant) * f64::exp2(-24.0)
    } else if exp == 0x1f {
        if mant == 0 { f64::INFINITY } else { f64::NAN }
    } else {
        // Normal: (1 + mant/1024) · 2^(exp − 15).
        let frac = 1.0 + f64::from(mant) / 1024.0;
        frac * f64::exp2(f64::from(exp) - 15.0)
    };
    sign * mag
}

/// Round an `f64` to an IEEE binary16 bit pattern (round-to-nearest, ties to
/// even). Handles NaN, ±∞, ±0, subnormals, and overflow-to-∞.
fn f16_from_f64(x: f64) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    if x.is_nan() {
        return 0x7e00; // canonical quiet NaN
    }
    if x.is_infinite() {
        return sign | 0x7c00;
    }
    if x == 0.0 {
        return sign; // ±0
    }
    let bits = x.to_bits();
    let f_exp = ((bits >> 52) & 0x7ff) as i64; // biased f64 exponent
    let f_mant = bits & 0x000f_ffff_ffff_ffff; // 52-bit fraction
    // Full 53-bit significand of a normal f64; every value here is normal
    // (subnormal f64s are far below f16's subnormal range and round to 0).
    if f_exp == 0 {
        return sign; // f64 subnormal ⇒ far below f16 min ⇒ ±0
    }
    let full = (1u64 << 52) | f_mant; // in [2⁵², 2⁵³)
    let e = f_exp - 1023; // unbiased power of two

    if e > 15 {
        return sign | 0x7c00; // overflow ⇒ ±∞
    }
    if e >= -14 {
        // Normal f16: round the 53-bit significand to 11 bits (implicit + 10).
        let r = round_shift(full, 42); // 53 − 11
        if r == (1u64 << 11) {
            // Rounding carried into a new binade.
            let e16 = e + 1 + 15;
            if e16 >= 0x1f {
                return sign | 0x7c00;
            }
            return sign | ((e16 as u16) << 10);
        }
        let e16 = (e + 15) as u16;
        sign | (e16 << 10) | ((r as u16) & 0x3ff)
    } else {
        // Subnormal f16 (or underflow to 0). value = full · 2^(e−52); round to
        // the nearest multiple of 2⁻²⁴, i.e. round(full · 2^(e−28)).
        let shift = (28 - e) as u32; // ≥ 43
        let m = round_shift(full, shift);
        // `m == 0x400` means rounding reached the smallest normal (exp field 1),
        // which the bit pattern `sign | 0x400` already encodes correctly.
        sign | (m as u16)
    }
}

/// `round(value / 2^shift)` with round-half-to-even. For `shift ≥ 64` the shifted
/// magnitudes here are always < ½ ulp, so the result is 0.
fn round_shift(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        return value;
    }
    if shift >= 64 {
        return 0;
    }
    let lo = value & ((1u64 << shift) - 1);
    let hi = value >> shift;
    let half = 1u64 << (shift - 1);
    if lo > half || (lo == half && (hi & 1) == 1) { hi + 1 } else { hi }
}

// ---------------------------------------------------------------------------
// Constant folding.
// ---------------------------------------------------------------------------

/// Constant-fold a pure instruction whose operands are all constants.
///
/// Returns `None` when folding is not applicable — an operand is not a scalar
/// constant this evaluator models (e.g. an aggregate), or the result cannot be
/// represented as a [`Const`] (e.g. a non-null abstract pointer). Otherwise it
/// returns [`FoldResult::Folded`] with the resulting constant (poison folds to
/// [`Const::Poison`]) or [`FoldResult::WouldBeUb`] if evaluation is UB — in
/// which case the caller must **not** fold the instruction away.
///
/// Folded integer constants use the **unsigned representative** of the bit
/// pattern (the value in `[0, 2ⁿ)`); e.g. an `i8` result of `−1` is emitted as
/// `255`. This is deterministic and matches `i1` booleans folding to `0`/`1`.
pub fn fold(
    types: &TypeContext,
    result_ty: TypeId,
    kind: &InstKind,
    flags: &Flags,
    operands: &[Const],
) -> Option<FoldResult> {
    let mut sem = Vec::with_capacity(operands.len());
    for c in operands {
        sem.push(const_to_sem(types, c)?);
    }
    match eval(types, result_ty, kind, flags, &sem) {
        EvalOutcome::UndefinedBehavior => Some(FoldResult::WouldBeUb),
        EvalOutcome::Value(v) => sem_to_const(v, result_ty).map(FoldResult::Folded),
    }
}

/// Convert an interned constant to a [`SemValue`], or `None` if it is a kind the
/// scalar evaluator does not model (aggregates).
fn const_to_sem(types: &TypeContext, c: &Const) -> Option<SemValue> {
    match c {
        Const::Int { ty, value } => {
            let w = int_width(types, *ty)?;
            Some(SemValue::int(w, value.clone()))
        }
        Const::Float { bits, .. } => Some(SemValue::Float(*bits)),
        Const::Null(_) => Some(SemValue::ptr(Int::ZERO)),
        Const::Poison(_) => Some(SemValue::Poison),
        Const::Aggregate { .. } => None,
    }
}

/// Convert a produced [`SemValue`] back to a [`Const`] of `result_ty`, or `None`
/// if it has no constant representation (a non-null abstract pointer).
fn sem_to_const(v: SemValue, result_ty: TypeId) -> Option<Const> {
    match v {
        SemValue::Int { bits, .. } => Some(Const::Int { ty: result_ty, value: bits }),
        SemValue::Float(bits) => Some(Const::Float { ty: result_ty, bits }),
        SemValue::Ptr(addr) => addr.is_zero().then_some(Const::Null(result_ty)),
        SemValue::Poison => Some(Const::Poison(result_ty)),
    }
}

#[cfg(test)]
mod tests;
