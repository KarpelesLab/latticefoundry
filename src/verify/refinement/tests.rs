//! End-to-end refinement tests, each driving a real `src`/`tgt` function pair
//! through the SMT encoder and `z3rs`. Sound rewrites must prove `Refines`,
//! unsound ones must return a `Counterexample`, and out-of-scope constructs must
//! come back cleanly as `Unknown`. A couple of cases are cross-checked against
//! the concrete reference evaluator (`ir::semantics::eval`) to show the symbolic
//! and concrete semantics agree.

use super::{RefinementResult, check_refinement};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, Flags};
use crate::ir::value::ValueId;
use crate::ir::{
    BlockId, CastOp, EvalOutcome, FloatKind, FloatBits, Module, SemValue, TypeId, eval,
};
use crate::support::StrInterner;

use puremp::Int;

/// A tiny harness: build a `src` and a `tgt` function with the given parameter
/// types and return type, then run the refinement check between them. All types
/// are interned in the harness's own module (so [`TypeId`]s stay valid).
struct Harness {
    module: Module,
    syms: StrInterner,
    params: Vec<TypeId>,
    ret: TypeId,
    src: Option<crate::ir::FuncId>,
    tgt: Option<crate::ir::FuncId>,
}

impl Harness {
    /// A harness whose signature is fixed by `params`/`ret`, which must be
    /// [`TypeId`]s interned via the returned harness's own module. Use
    /// [`Harness::signature`] to intern them first.
    fn signature(build: impl FnOnce(&mut Module) -> (Vec<TypeId>, TypeId)) -> Harness {
        let mut module = Module::new("refine_test");
        let (params, ret) = build(&mut module);
        Harness { module, syms: StrInterner::new(), params, ret, src: None, tgt: None }
    }

    /// Build a function body; `f` receives the builder and the parameter values
    /// and returns the value to `ret`.
    fn build(
        &mut self,
        name: &str,
        f: impl FnOnce(&mut FunctionBuilder, &[ValueId]) -> ValueId,
    ) -> crate::ir::FuncId {
        let sig = self.module.types_mut().func(self.params.clone(), self.ret, false);
        let sym = self.syms.intern(name);
        let id = self.module.declare_function(sym, sig);
        {
            let mut b = self.module.build(id);
            let entry = b.create_entry_block();
            let params: Vec<ValueId> = b.block_params(entry).to_vec();
            let r = f(&mut b, &params);
            b.ret(Some(r));
        }
        id
    }

    /// Intern an integer type in the harness module (for intermediate widths).
    fn int(&mut self, width: u32) -> TypeId {
        self.module.types_mut().int(width)
    }

    fn set_src(&mut self, f: impl FnOnce(&mut FunctionBuilder, &[ValueId]) -> ValueId) {
        self.src = Some(self.build("src", f));
    }

    fn set_tgt(&mut self, f: impl FnOnce(&mut FunctionBuilder, &[ValueId]) -> ValueId) {
        self.tgt = Some(self.build("tgt", f));
    }

    /// Build a **multi-block** body: `f` receives the builder, the entry block id,
    /// and the entry (function) parameters, and is responsible for creating any
    /// further blocks and terminating every one of them (including its `ret`s).
    fn build_raw(
        &mut self,
        name: &str,
        f: impl FnOnce(&mut FunctionBuilder, BlockId, &[ValueId]),
    ) -> crate::ir::FuncId {
        let sig = self.module.types_mut().func(self.params.clone(), self.ret, false);
        let sym = self.syms.intern(name);
        let id = self.module.declare_function(sym, sig);
        {
            let mut b = self.module.build(id);
            let entry = b.create_entry_block();
            let params: Vec<ValueId> = b.block_params(entry).to_vec();
            f(&mut b, entry, &params);
        }
        id
    }

    fn set_src_raw(&mut self, f: impl FnOnce(&mut FunctionBuilder, BlockId, &[ValueId])) {
        self.src = Some(self.build_raw("src", f));
    }

    fn set_tgt_raw(&mut self, f: impl FnOnce(&mut FunctionBuilder, BlockId, &[ValueId])) {
        self.tgt = Some(self.build_raw("tgt", f));
    }

    fn check(&self) -> RefinementResult {
        check_refinement(
            self.module.types(),
            self.module.consts(),
            self.module.function(self.src.expect("src")),
            self.module.function(self.tgt.expect("tgt")),
        )
    }
}

/// A one-parameter `iN -> iN` harness (the common shape).
fn unary_int(width: u32) -> Harness {
    Harness::signature(|m| {
        let t = m.types_mut().int(width);
        (vec![t], t)
    })
}

fn assert_refines(r: RefinementResult) {
    assert!(matches!(r, RefinementResult::Refines), "expected Refines, got {r:?}");
}

fn assert_counterexample(r: RefinementResult) {
    assert!(matches!(r, RefinementResult::Counterexample(_)), "expected Counterexample, got {r:?}");
}

// ---------------------------------------------------------------------------
// Sound rewrites: proven `Refines`.
// ---------------------------------------------------------------------------

#[test]
fn mul_two_to_shl_one() {
    // x * 2  ⇒  x << 1
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(move |b, p| {
        let two = b.const_i64(ity, 2);
        b.mul(p[0], two, Flags::NONE)
    });
    h.set_tgt(move |b, p| {
        let one = b.const_i64(ity, 1);
        b.bin(BinOp::Shl, p[0], one, Flags::NONE)
    });
    assert_refines(h.check());
}

#[test]
fn add_zero_to_identity() {
    // x + 0  ⇒  x
    let mut h = unary_int(16);
    let ity = h.params[0];
    h.set_src(move |b, p| {
        let zero = b.const_i64(ity, 0);
        b.add(p[0], zero, Flags::NONE)
    });
    h.set_tgt(|_, p| p[0]);
    assert_refines(h.check());
}

#[test]
fn and_self_to_identity() {
    // x & x  ⇒  x
    let mut h = unary_int(8);
    h.set_src(|b, p| b.bin(BinOp::And, p[0], p[0], Flags::NONE));
    h.set_tgt(|_, p| p[0]);
    assert_refines(h.check());
}

#[test]
fn sub_self_to_zero() {
    // x - x  ⇒  0
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(|b, p| b.sub(p[0], p[0], Flags::NONE));
    h.set_tgt(move |b, _| b.const_i64(ity, 0));
    assert_refines(h.check());
}

#[test]
fn add_one_sub_one_to_identity_wrapping() {
    // (x + 1) - 1  ⇒  x   (wrapping, no flags)
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(move |b, p| {
        let one = b.const_i64(ity, 1);
        let t = b.add(p[0], one, Flags::NONE);
        b.sub(t, one, Flags::NONE)
    });
    h.set_tgt(|_, p| p[0]);
    assert_refines(h.check());
}

#[test]
fn select_true_to_first_arm() {
    // select true, a, b  ⇒  a
    let mut h = Harness::signature(|m| {
        let ity = m.types_mut().int(8);
        (vec![ity, ity], ity)
    });
    h.set_src(|b, p| {
        let t = b.const_bool(true);
        b.select(t, p[0], p[1])
    });
    h.set_tgt(|_, p| p[0]);
    assert_refines(h.check());
}

#[test]
fn xor_self_to_zero() {
    // x ^ x  ⇒  0  (extra sound case for good measure)
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(|b, p| b.bin(BinOp::Xor, p[0], p[0], Flags::NONE));
    h.set_tgt(move |b, _| b.const_i64(ity, 0));
    assert_refines(h.check());
}

// ---------------------------------------------------------------------------
// Unsound rewrites: caught as `Counterexample`.
// ---------------------------------------------------------------------------

#[test]
fn add_one_to_identity_is_unsound() {
    // x + 1  ⇒  x   (wrong: differs for every x)
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(move |b, p| {
        let one = b.const_i64(ity, 1);
        b.add(p[0], one, Flags::NONE)
    });
    h.set_tgt(|_, p| p[0]);
    assert_counterexample(h.check());
}

#[test]
fn shl_one_to_mul_three_is_unsound() {
    // x << 1  ⇒  x * 3   (wrong: 2x != 3x in general)
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(move |b, p| {
        let one = b.const_i64(ity, 1);
        b.bin(BinOp::Shl, p[0], one, Flags::NONE)
    });
    h.set_tgt(move |b, p| {
        let three = b.const_i64(ity, 3);
        b.mul(p[0], three, Flags::NONE)
    });
    assert_counterexample(h.check());
}

#[test]
fn adding_nsw_flag_breaks_poison_refinement() {
    // x + x  ⇒  x << 1 (nsw)
    //
    // The value is the same (2x wrapping), but the `nsw` on the target makes it
    // *poison* on signed overflow, whereas the source `add` is always defined.
    // A defined source pins the target, so the target's new poison behavior
    // breaks refinement — a genuinely flag-induced counterexample.
    let mut h = unary_int(8);
    let ity = h.params[0];
    // src: x + x (wrapping, always defined).
    h.set_src(|b, p| b.add(p[0], p[0], Flags::NONE));
    // tgt: x << 1 with nsw (poison when the signed shift overflows i8).
    h.set_tgt(move |b, p| {
        let one = b.const_i64(ity, 1);
        b.bin(BinOp::Shl, p[0], one, Flags::nsw())
    });
    assert_counterexample(h.check());
}

// ---------------------------------------------------------------------------
// Poison direction.
// ---------------------------------------------------------------------------

#[test]
fn possibly_poison_to_defined_refines() {
    // src: add nsw x 100 (possibly poison)  ⇒  tgt: wrapping add x 100 (defined).
    // Replacing a possibly-poison value with a more-defined one is a valid
    // refinement: where src is poison, any target is acceptable; where src is
    // defined, the wrapping add equals it.
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(move |b, p| {
        let c = b.const_i64(ity, 100);
        b.add(p[0], c, Flags::nsw())
    });
    h.set_tgt(move |b, p| {
        let c = b.const_i64(ity, 100);
        b.add(p[0], c, Flags::NONE)
    });
    assert_refines(h.check());
}

#[test]
fn defined_to_possibly_poison_is_unsound() {
    // The reverse of the above: defined src ⇒ possibly-poison tgt is NOT a
    // refinement.
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(move |b, p| {
        let c = b.const_i64(ity, 100);
        b.add(p[0], c, Flags::NONE)
    });
    h.set_tgt(move |b, p| {
        let c = b.const_i64(ity, 100);
        b.add(p[0], c, Flags::nsw())
    });
    assert_counterexample(h.check());
}

#[test]
fn freeze_of_poison_replacing_defined_is_unsound() {
    // src: x  (defined)     ⇒     tgt: freeze(poison)  (any value).
    // freeze(poison) can produce any value, so it does not refine a defined x.
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(|_, p| p[0]);
    h.set_tgt(move |b, _| {
        let p = b.poison(ity);
        b.freeze(p)
    });
    assert_counterexample(h.check());
}

#[test]
fn freeze_of_defined_is_identity() {
    // freeze(x) ⇒ x when x is a (possibly-poison) parameter: freeze never adds
    // poison and equals its operand when the operand is defined, so replacing
    // freeze(x) by x is UNSOUND (x may be poison, freeze(x) never is)...
    // instead test the sound direction: x ⇒ freeze(x) is a valid refinement
    // (target is more-defined).
    let mut h = unary_int(8);
    h.set_src(|_, p| p[0]);
    h.set_tgt(|b, p| b.freeze(p[0]));
    assert_refines(h.check());
}

// ---------------------------------------------------------------------------
// Out of scope: clean `Unknown`.
// ---------------------------------------------------------------------------

#[test]
fn float_op_is_unknown() {
    let mut h = Harness::signature(|m| {
        let fty = m.types_mut().float(FloatKind::F32);
        (vec![fty], fty)
    });
    h.set_src(|b, p| b.bin(BinOp::FAdd, p[0], p[0], Flags::NONE));
    h.set_tgt(|_, p| p[0]);
    match h.check() {
        RefinementResult::Unknown(reason) => {
            assert!(reason.contains("unsupported"), "reason was {reason:?}");
        }
        other => panic!("expected Unknown, got {other:?}"),
    }
}

#[test]
fn float_constant_input_is_unknown() {
    // A float-typed parameter is itself out of scope even before any op.
    let bits = FloatBits::F64(0);
    let mut h = Harness::signature(|m| {
        let fty = m.types_mut().float(FloatKind::F64);
        (vec![fty], fty)
    });
    let fty = h.params[0];
    h.set_src(|_, p| p[0]);
    h.set_tgt(move |b, _| b.const_float(fty, bits));
    assert!(matches!(h.check(), RefinementResult::Unknown(_)));
}

// ---------------------------------------------------------------------------
// Cross-checks against the concrete reference evaluator.
// ---------------------------------------------------------------------------

/// Concretely evaluate `x * 2` and `x << 1` on an i8 and confirm they agree —
/// the same fact the symbolic `mul_two_to_shl_one` proves for all inputs.
#[test]
fn concrete_agrees_mul2_eq_shl1() {
    let mut m = Module::new("tmp");
    let i8 = m.types_mut().int(8);
    for x in [0i64, 1, 5, 63, 100, -1, -50, 127] {
        let xv = SemValue::int(8, Int::from_i64(x));
        let two = SemValue::int(8, Int::from_i64(2));
        let one = SemValue::int(8, Int::from_i64(1));
        let mul = eval(
            m.types(),
            i8,
            &crate::ir::InstKind::Bin(BinOp::Mul),
            &Flags::NONE,
            &[xv.clone(), two],
        );
        let shl = eval(
            m.types(),
            i8,
            &crate::ir::InstKind::Bin(BinOp::Shl),
            &Flags::NONE,
            &[xv, one],
        );
        assert_eq!(mul, shl, "x*2 vs x<<1 disagree at x={x}");
        assert!(matches!(mul, EvalOutcome::Value(SemValue::Int { .. })));
    }
}

/// Concretely show `x + 1 != x` for a sample input, matching the symbolic
/// counterexample in `add_one_to_identity_is_unsound`.
#[test]
fn concrete_agrees_add1_differs() {
    let mut m = Module::new("tmp");
    let i8 = m.types_mut().int(8);
    let x = SemValue::int(8, Int::from_i64(41));
    let one = SemValue::int(8, Int::from_i64(1));
    let sum = eval(
        m.types(),
        i8,
        &crate::ir::InstKind::Bin(BinOp::Add),
        &Flags::NONE,
        &[x.clone(), one],
    );
    assert_ne!(EvalOutcome::Value(x), sum, "x+1 must differ from x");
}

/// A cast round-trip refinement: sext(trunc(x)) is NOT x in general (loses the
/// high bits), so the rewrite trunc-then-sext ⇒ identity must be caught.
#[test]
fn sext_trunc_not_identity_is_unsound() {
    let mut h = unary_int(8);
    let i8 = h.params[0];
    let i4 = h.int(4);
    h.set_src(move |b, p| {
        let narrow = b.cast(CastOp::Trunc, p[0], i4);
        b.cast(CastOp::SExt, narrow, i8)
    });
    h.set_tgt(|_, p| p[0]);
    assert_counterexample(h.check());
}

/// zext(trunc(x)) with a mask: (x & 0x0F) equals zext(trunc x to i4) — a sound
/// cast/bitwise refinement exercising trunc + zext.
#[test]
fn zext_trunc_equals_mask() {
    let mut h = unary_int(8);
    let i8 = h.params[0];
    let i4 = h.int(4);
    h.set_src(move |b, p| {
        let narrow = b.cast(CastOp::Trunc, p[0], i4);
        b.cast(CastOp::ZExt, narrow, i8)
    });
    h.set_tgt(move |b, p| {
        let mask = b.const_i64(i8, 0x0F);
        b.bin(BinOp::And, p[0], mask, Flags::NONE)
    });
    assert_refines(h.check());
}

/// A UB precondition case: `sdiv`. `(x sdiv x)` — when the source is UB-free
/// (x != 0 and not INT_MIN/-1, but x sdiv x avoids the latter) — equals 1.
/// Rewriting `x sdiv x ⇒ 1` is sound *because* division by zero in the source is
/// assumed not to occur.
#[test]
fn sdiv_self_to_one_under_ub_precondition() {
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(|b, p| b.bin(BinOp::SDiv, p[0], p[0], Flags::NONE));
    h.set_tgt(move |b, _| b.const_i64(ity, 1));
    assert_refines(h.check());
}

/// Introducing a division in the target that can be UB (divisor not ruled out
/// by the source) must break refinement: `x ⇒ (100 sdiv x)` is unsound because
/// the target is UB when x == 0, which the source does not rule out.
#[test]
fn target_introducing_ub_is_unsound() {
    let mut h = unary_int(8);
    let ity = h.params[0];
    h.set_src(|_, p| p[0]);
    h.set_tgt(move |b, p| {
        let c = b.const_i64(ity, 100);
        b.bin(BinOp::SDiv, c, p[0], Flags::NONE)
    });
    assert_counterexample(h.check());
}

// ---------------------------------------------------------------------------
// Multi-block (acyclic) control flow.
// ---------------------------------------------------------------------------

/// An `(i8, i8, i1) -> i8` harness: two integer values and a branch condition.
fn diamond_harness() -> Harness {
    Harness::signature(|m| {
        let i8 = m.types_mut().int(8);
        let i1 = m.types_mut().bool();
        (vec![i8, i8, i1], i8)
    })
}

/// THE simplify_cfg-class miscompile, caught automatically. `src` is a diamond
/// that merges `a`/`b` through a block parameter and returns it; `tgt`
/// (incorrectly) drops the merge to `poison`. A defined source pins the target,
/// so the checker must produce a counterexample.
#[test]
fn merge_dropped_to_poison_is_caught() {
    let mut h = diamond_harness();
    // src: cond_br(c) -> L(a) / R(b); M(x) = phi; ret x.
    h.set_src_raw(|b, entry, p| {
        let (a, bb, c) = (p[0], p[1], p[2]);
        let i8 = b.value_type(a);
        let l = b.create_block(&[]);
        let r = b.create_block(&[]);
        let m = b.create_block(&[i8]);
        b.switch_to(entry);
        b.cond_br(c, l, &[], r, &[]);
        b.switch_to(l);
        b.br(m, &[a]);
        b.switch_to(r);
        b.br(m, &[bb]);
        b.switch_to(m);
        let x = b.param(m, 0);
        b.ret(Some(x));
    });
    // tgt: same CFG, but both edges feed poison into M (the merge is dropped).
    h.set_tgt_raw(|b, entry, p| {
        let c = p[2];
        let i8 = b.value_type(p[0]);
        let poison = b.poison(i8);
        let l = b.create_block(&[]);
        let r = b.create_block(&[]);
        let m = b.create_block(&[i8]);
        b.switch_to(entry);
        b.cond_br(c, l, &[], r, &[]);
        b.switch_to(l);
        b.br(m, &[poison]);
        b.switch_to(r);
        b.br(m, &[poison]);
        b.switch_to(m);
        let x = b.param(m, 0);
        b.ret(Some(x));
    });
    assert_counterexample(h.check());
}

/// Sound: constant-condition branch folding. `src` branches on a constant `true`
/// into one of two `ret` blocks; `tgt` returns the taken edge's value directly.
#[test]
fn const_condition_branch_folded_refines() {
    let mut h = Harness::signature(|m| {
        let i8 = m.types_mut().int(8);
        (vec![i8, i8], i8)
    });
    h.set_src_raw(|b, entry, p| {
        let (a, bb) = (p[0], p[1]);
        let t = b.create_block(&[]);
        let f = b.create_block(&[]);
        b.switch_to(entry);
        let tru = b.const_bool(true);
        b.cond_br(tru, t, &[], f, &[]);
        b.switch_to(t);
        b.ret(Some(a));
        b.switch_to(f);
        b.ret(Some(bb));
    });
    h.set_tgt_raw(|b, entry, p| {
        b.switch_to(entry);
        b.ret(Some(p[0]));
    });
    assert_refines(h.check());
}

/// Sound: removing an unreachable block. `src` carries a dangling block with no
/// predecessor (a different `ret`); `tgt` drops it. Both return the same value.
#[test]
fn removing_unreachable_block_refines() {
    let mut h = Harness::signature(|m| {
        let i8 = m.types_mut().int(8);
        (vec![i8, i8], i8)
    });
    h.set_src_raw(|b, entry, p| {
        let (a, bb) = (p[0], p[1]);
        let i8 = b.value_type(a);
        let m = b.create_block(&[i8]);
        let dead = b.create_block(&[]); // no predecessor: unreachable
        b.switch_to(entry);
        b.br(m, &[a]);
        b.switch_to(m);
        let x = b.param(m, 0);
        b.ret(Some(x));
        b.switch_to(dead);
        b.ret(Some(bb));
    });
    h.set_tgt_raw(|b, entry, p| {
        let a = p[0];
        let i8 = b.value_type(a);
        let m = b.create_block(&[i8]);
        b.switch_to(entry);
        b.br(m, &[a]);
        b.switch_to(m);
        let x = b.param(m, 0);
        b.ret(Some(x));
    });
    assert_refines(h.check());
}

/// Sound: a diamond that merges `a`/`b` on `c` equals `select(c, a, b)`.
#[test]
fn diamond_equals_select_refines() {
    let mut h = diamond_harness();
    h.set_src_raw(|b, entry, p| {
        let (a, bb, c) = (p[0], p[1], p[2]);
        let i8 = b.value_type(a);
        let l = b.create_block(&[]);
        let r = b.create_block(&[]);
        let m = b.create_block(&[i8]);
        b.switch_to(entry);
        b.cond_br(c, l, &[], r, &[]);
        b.switch_to(l);
        b.br(m, &[a]);
        b.switch_to(r);
        b.br(m, &[bb]);
        b.switch_to(m);
        let x = b.param(m, 0);
        b.ret(Some(x));
    });
    h.set_tgt_raw(|b, entry, p| {
        let (a, bb, c) = (p[0], p[1], p[2]);
        b.switch_to(entry);
        let s = b.select(c, a, bb);
        b.ret(Some(s));
    });
    assert_refines(h.check());
}

/// Sound: straightening a single-predecessor block. `src` jumps to `M(a)` and
/// computes `x + 1` there; `tgt` computes it inline. Equivalent.
#[test]
fn straighten_single_predecessor_refines() {
    let mut h = unary_int(8);
    let i8 = h.params[0];
    h.set_src_raw(move |b, entry, p| {
        let a = p[0];
        let m = b.create_block(&[i8]);
        b.switch_to(entry);
        b.br(m, &[a]);
        b.switch_to(m);
        let x = b.param(m, 0);
        let one = b.const_i64(i8, 1);
        let y = b.add(x, one, Flags::NONE);
        b.ret(Some(y));
    });
    h.set_tgt_raw(move |b, entry, p| {
        let a = p[0];
        b.switch_to(entry);
        let one = b.const_i64(i8, 1);
        let y = b.add(a, one, Flags::NONE);
        b.ret(Some(y));
    });
    assert_refines(h.check());
}

/// Sound: two `ret` blocks. `src` returns `a` on the true edge and `b` on the
/// false edge (a multi-`ret` function); `tgt` folds them to `select(c, a, b)`.
#[test]
fn two_ret_blocks_refine_select() {
    let mut h = diamond_harness();
    h.set_src_raw(|b, entry, p| {
        let (a, bb, c) = (p[0], p[1], p[2]);
        let t = b.create_block(&[]);
        let f = b.create_block(&[]);
        b.switch_to(entry);
        b.cond_br(c, t, &[], f, &[]);
        b.switch_to(t);
        b.ret(Some(a));
        b.switch_to(f);
        b.ret(Some(bb));
    });
    h.set_tgt_raw(|b, entry, p| {
        let (a, bb, c) = (p[0], p[1], p[2]);
        b.switch_to(entry);
        let s = b.select(c, a, bb);
        b.ret(Some(s));
    });
    assert_refines(h.check());
}

/// Unsound: swapping the arms of a `cond_br`. `tgt` feeds `b` on the true edge
/// and `a` on the false edge — the opposite of `src`.
#[test]
fn swapped_cond_br_arms_is_caught() {
    let mut h = diamond_harness();
    let build = |swap: bool| {
        move |b: &mut FunctionBuilder, entry: BlockId, p: &[ValueId]| {
            let (a, bb, c) = (p[0], p[1], p[2]);
            let i8 = b.value_type(a);
            let l = b.create_block(&[]);
            let r = b.create_block(&[]);
            let m = b.create_block(&[i8]);
            let (larg, rarg) = if swap { (bb, a) } else { (a, bb) };
            b.switch_to(entry);
            b.cond_br(c, l, &[], r, &[]);
            b.switch_to(l);
            b.br(m, &[larg]);
            b.switch_to(r);
            b.br(m, &[rarg]);
            b.switch_to(m);
            let x = b.param(m, 0);
            b.ret(Some(x));
        }
    };
    h.set_src_raw(build(false));
    h.set_tgt_raw(build(true));
    assert_counterexample(h.check());
}

/// Unsound: a merge that picks the wrong incoming value. `src` returns `a`/`b`
/// on `c`; `tgt` folds to `select(c, b, a)` (arms reversed).
#[test]
fn wrong_merge_value_is_caught() {
    let mut h = diamond_harness();
    h.set_src_raw(|b, entry, p| {
        let (a, bb, c) = (p[0], p[1], p[2]);
        let i8 = b.value_type(a);
        let l = b.create_block(&[]);
        let r = b.create_block(&[]);
        let m = b.create_block(&[i8]);
        b.switch_to(entry);
        b.cond_br(c, l, &[], r, &[]);
        b.switch_to(l);
        b.br(m, &[a]);
        b.switch_to(r);
        b.br(m, &[bb]);
        b.switch_to(m);
        let x = b.param(m, 0);
        b.ret(Some(x));
    });
    h.set_tgt_raw(|b, entry, p| {
        let (a, bb, c) = (p[0], p[1], p[2]);
        b.switch_to(entry);
        let s = b.select(c, bb, a); // reversed arms
        b.ret(Some(s));
    });
    assert_counterexample(h.check());
}

/// Sound: a `switch` folded to its matching case. `src` switches on `a`; `tgt`
/// straight-lines the block a concrete constant selects.
#[test]
fn switch_constant_folds_refines() {
    let mut h = Harness::signature(|m| {
        let i8 = m.types_mut().int(8);
        (vec![i8], i8)
    });
    let i8 = h.params[0];
    // src: switch a { 7 => C7 (ret 70), default => D (ret 99) }, but a is the
    // constant 7, so it always lands in C7.
    h.set_src_raw(move |b, entry, _p| {
        let c7 = b.create_block(&[]);
        let d = b.create_block(&[]);
        b.switch_to(entry);
        let seven = b.const_i64(i8, 7);
        b.switch(seven, d, &[], vec![(puremp::Int::from_i64(7), c7, vec![])]);
        b.switch_to(c7);
        let r70 = b.const_i64(i8, 70);
        b.ret(Some(r70));
        b.switch_to(d);
        let r99 = b.const_i64(i8, 99);
        b.ret(Some(r99));
    });
    h.set_tgt_raw(move |b, entry, _p| {
        b.switch_to(entry);
        let r70 = b.const_i64(i8, 70);
        b.ret(Some(r70));
    });
    assert_refines(h.check());
}

/// A function with a loop (a back-edge) is reported cleanly as `Unknown`,
/// never encoded. `src` loops on `L`; the checker must bail before the target.
#[test]
fn loop_is_unknown() {
    let mut h = Harness::signature(|m| {
        let i8 = m.types_mut().int(8);
        let i1 = m.types_mut().bool();
        (vec![i8, i1], i8)
    });
    h.set_src_raw(|b, entry, p| {
        let (a, c) = (p[0], p[1]);
        let i8 = b.value_type(a);
        let l = b.create_block(&[i8]);
        let m = b.create_block(&[i8]);
        b.switch_to(entry);
        b.br(l, &[a]);
        b.switch_to(l);
        let x = b.param(l, 0);
        // true edge is a back-edge to L; false edge exits to M.
        b.cond_br(c, l, &[x], m, &[x]);
        b.switch_to(m);
        let y = b.param(m, 0);
        b.ret(Some(y));
    });
    h.set_tgt_raw(|b, entry, p| {
        b.switch_to(entry);
        b.ret(Some(p[0]));
    });
    match h.check() {
        RefinementResult::Unknown(s) => assert_eq!(s, "loops unsupported"),
        other => panic!("expected Unknown(\"loops unsupported\"), got {other:?}"),
    }
}
