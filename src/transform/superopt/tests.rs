//! Tests for the bounded superoptimizer: rediscovery of known-optimal rewrites,
//! soundness (the solver is the gate, not the samples), termination under budget,
//! and determinism.

use super::{
    Budget, Expr, Spec, concretely_refines, cost_of, discover_rules, eval_expr, prove_refines,
    superoptimize, synthesize, TypeMap,
};

use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, Flags, InstKind};
use crate::ir::value::ValueId;
use crate::ir::{EvalOutcome, Function, Module, SemValue};
use crate::support::StrInterner;
use crate::verify::{RefinementResult, check_refinement};

use puremp::Int;

// --- small builders --------------------------------------------------------

/// Add a single-parameter `iwidth -> iwidth` function to `m` and return its id.
fn add_unary(
    m: &mut Module,
    syms: &mut StrInterner,
    name: &str,
    width: u32,
    body: impl FnOnce(&mut FunctionBuilder, ValueId) -> ValueId,
) -> crate::ir::FuncId {
    let ity = m.types_mut().int(width);
    let sig = m.types_mut().func(vec![ity], ity, false);
    let sym = syms.intern(name);
    let id = m.declare_function(sym, sig);
    {
        let mut b = m.build(id);
        let entry = b.create_entry_block();
        let x = b.block_params(entry)[0];
        let r = body(&mut b, x);
        b.ret(Some(r));
    }
    id
}

/// Add a two-parameter `(iwidth, iwidth) -> iwidth` function to `m`.
fn add_binary(
    m: &mut Module,
    syms: &mut StrInterner,
    name: &str,
    width: u32,
    body: impl FnOnce(&mut FunctionBuilder, ValueId, ValueId) -> ValueId,
) -> crate::ir::FuncId {
    let ity = m.types_mut().int(width);
    let sig = m.types_mut().func(vec![ity, ity], ity, false);
    let sym = syms.intern(name);
    let id = m.declare_function(sym, sig);
    {
        let mut b = m.build(id);
        let entry = b.create_entry_block();
        let p = b.block_params(entry).to_vec();
        let r = body(&mut b, p[0], p[1]);
        b.ret(Some(r));
    }
    id
}

fn count_bin(f: &Function) -> usize {
    let mut c = 0;
    for (_bid, blk) in f.blocks() {
        for &i in blk.insts() {
            if matches!(f.inst(i).kind, InstKind::Bin(_)) {
                c += 1;
            }
        }
    }
    c
}

fn has_op(f: &Function, op: BinOp) -> bool {
    for (_bid, blk) in f.blocks() {
        for &i in blk.insts() {
            if matches!(f.inst(i).kind, InstKind::Bin(o) if o == op) {
                return true;
            }
        }
    }
    false
}

fn func_ops(f: &Function) -> Vec<String> {
    let mut v = Vec::new();
    for (_bid, blk) in f.blocks() {
        for &i in blk.insts() {
            v.push(format!("{:?}", f.inst(i).kind));
        }
    }
    v
}

// --- rediscovery of known-optimal rewrites ---------------------------------

#[test]
fn rediscovers_mul_two_as_shift() {
    let mut m = Module::new("t");
    let mut syms = StrInterner::new();
    let f = add_unary(&mut m, &mut syms, "f", 8, |b, x| {
        let ty = b.value_type(x);
        let two = b.const_int(ty, Int::from_u64(2));
        b.mul(x, two, Flags::NONE)
    });

    let out = superoptimize(&mut m, f).expect("x*2 has a cheaper equivalent");
    // Independently re-verify: the replacement refines the original.
    let r = check_refinement(m.types(), m.consts(), m.function(f), &out);
    assert!(matches!(r, RefinementResult::Refines), "not a proven refinement: {r:?}");
    // It found a shift and eliminated the multiply.
    assert!(has_op(&out, BinOp::Shl), "expected a shift in {:?}", func_ops(&out));
    assert!(!has_op(&out, BinOp::Mul), "multiply should be gone: {:?}", func_ops(&out));
}

#[test]
fn shift_rewrite_directly_proves() {
    // The specific x*2 ⇒ x<<1 rewrite proves `Refines` on its own.
    let spec = Spec {
        param_widths: vec![8],
        ret_width: 8,
        target: Expr::Bin(
            BinOp::Mul,
            Flags::NONE,
            Box::new(Expr::Param(0)),
            Box::new(Expr::Const { width: 8, value: Int::from_u64(2) }),
        ),
    };
    let cand = Expr::Bin(
        BinOp::Shl,
        Flags::NONE,
        Box::new(Expr::Param(0)),
        Box::new(Expr::Const { width: 8, value: Int::ONE }),
    );
    assert!(prove_refines(&spec, &cand).is_refines());
}

#[test]
fn rediscovers_mul_zero_as_zero() {
    let mut m = Module::new("t");
    let mut syms = StrInterner::new();
    let f = add_unary(&mut m, &mut syms, "f", 8, |b, x| {
        let ty = b.value_type(x);
        let zero = b.const_int(ty, Int::ZERO);
        b.mul(x, zero, Flags::NONE)
    });

    let out = superoptimize(&mut m, f).expect("x*0 folds to 0");
    let r = check_refinement(m.types(), m.consts(), m.function(f), &out);
    assert!(matches!(r, RefinementResult::Refines), "not a refinement: {r:?}");
    // The result is a bare constant: no arithmetic left.
    assert_eq!(count_bin(&out), 0, "x*0 should reduce to a constant: {:?}", func_ops(&out));
}

#[test]
fn rediscovers_and_x_x_as_x() {
    let mut m = Module::new("t");
    let mut syms = StrInterner::new();
    let f = add_unary(&mut m, &mut syms, "f", 8, |b, x| b.bin(BinOp::And, x, x, Flags::NONE));

    let out = superoptimize(&mut m, f).expect("x & x = x");
    let r = check_refinement(m.types(), m.consts(), m.function(f), &out);
    assert!(matches!(r, RefinementResult::Refines), "not a refinement: {r:?}");
    assert_eq!(count_bin(&out), 0, "x & x should reduce to x: {:?}", func_ops(&out));
}

#[test]
fn add_x_x_finds_equivalent_of_cost_at_most_input() {
    // (x + x): the synthesizer finds x<<1 (or, at worst, keeps an equal-cost
    // equivalent) — a proven refinement of cost ≤ the input.
    let spec = Spec {
        param_widths: vec![8],
        ret_width: 8,
        target: Expr::Bin(
            BinOp::Add,
            Flags::NONE,
            Box::new(Expr::Param(0)),
            Box::new(Expr::Param(0)),
        ),
    };
    let found = synthesize(&spec, &Budget::default(), false).expect("x+x has an equivalent");
    assert!(found.proof.is_refines(), "equivalent must be proven");
    assert!(found.cost <= cost_of(&spec.target), "must not be more expensive than the input");

    // And the strict, module-level API strictly improves it (shift < add).
    let mut m = Module::new("t");
    let mut syms = StrInterner::new();
    let f = add_unary(&mut m, &mut syms, "f", 8, |b, x| b.add(x, x, Flags::NONE));
    if let Some(out) = superoptimize(&mut m, f) {
        let r = check_refinement(m.types(), m.consts(), m.function(f), &out);
        assert!(matches!(r, RefinementResult::Refines));
    }
}

// --- soundness: the solver is the gate, not the samples --------------------

/// Test-only convenience: does `cand` agree with `spec.target` on `samples`?
fn concrete_agrees(spec: &Spec, cand: &Expr, samples: &[Vec<SemValue>]) -> bool {
    let mut scratch = Module::new("s");
    let tm = TypeMap::build(&mut scratch, &spec.param_widths, spec.ret_width, &spec.target);
    let types = scratch.types();
    let target_out: Vec<EvalOutcome> =
        samples.iter().map(|inp| eval_expr(types, &tm, spec, &spec.target, inp)).collect();
    concretely_refines(types, &tm, spec, cand, samples, &target_out)
}

#[test]
fn solver_is_the_gate_not_samples() {
    // Target: the identity `x` (i8). Near-miss candidate: `x & 0xFE`, which
    // equals x on every *even* input but differs whenever bit 0 is set.
    let spec = Spec { param_widths: vec![8], ret_width: 8, target: Expr::Param(0) };
    let cand = Expr::Bin(
        BinOp::And,
        Flags::NONE,
        Box::new(Expr::Param(0)),
        Box::new(Expr::Const { width: 8, value: Int::from_u64(0xFE) }),
    );

    // On a sample set of only even values, the concrete filter *accepts* it —
    // demonstrating that sample-agreement alone would admit an unsound rewrite.
    let even_samples: Vec<Vec<SemValue>> =
        (0u64..8).map(|k| vec![SemValue::int(8, Int::from_u64(k * 2))]).collect();
    assert!(concrete_agrees(&spec, &cand, &even_samples), "candidate agrees on even samples");

    // But z3rs refutes it: it is not a refinement, so the synthesizer rejects it.
    let verdict = prove_refines(&spec, &cand);
    assert!(
        matches!(verdict, RefinementResult::Counterexample(_)),
        "solver must refute the near-miss, got {verdict:?}",
    );
    assert!(!verdict.is_refines());
}

#[test]
fn every_discovered_rule_is_independently_proven() {
    let mut m = Module::new("seeds");
    let mut syms = StrInterner::new();
    let f_mul2 = add_unary(&mut m, &mut syms, "mul2", 8, |b, x| {
        let ty = b.value_type(x);
        let two = b.const_int(ty, Int::from_u64(2));
        b.mul(x, two, Flags::NONE)
    });
    let f_mul0 = add_unary(&mut m, &mut syms, "mul0", 8, |b, x| {
        let ty = b.value_type(x);
        let zero = b.const_int(ty, Int::ZERO);
        b.mul(x, zero, Flags::NONE)
    });
    let f_andxx = add_unary(&mut m, &mut syms, "andxx", 8, |b, x| b.bin(BinOp::And, x, x, Flags::NONE));

    let db = discover_rules(&m, &[f_mul2, f_mul0, f_andxx], &Budget::default());
    assert!(!db.is_empty(), "should discover at least one rule");
    for rule in db.rules() {
        // Re-verify each rule independently against the database's own tables.
        let r = check_refinement(db.module().types(), db.module().consts(), db.lhs(rule), db.rhs(rule));
        assert!(matches!(r, RefinementResult::Refines), "unproven rule admitted: {r:?}");
        assert!(rule.proof().is_refines());
        assert!(rule.rhs_cost() < rule.lhs_cost(), "rule must be strictly cheaper");
    }
}

// --- termination under budget ----------------------------------------------

#[test]
fn identity_is_already_optimal() {
    let mut m = Module::new("t");
    let mut syms = StrInterner::new();
    let f = add_unary(&mut m, &mut syms, "id", 8, |_b, x| x);
    assert!(superoptimize(&mut m, f).is_none(), "identity has no cheaper equivalent");
}

#[test]
fn no_cheaper_equivalent_terminates_with_none() {
    // `x + y` over two distinct parameters is minimal; the search must terminate
    // without finding a cheaper proven equivalent.
    let mut m = Module::new("t");
    let mut syms = StrInterner::new();
    let f = add_binary(&mut m, &mut syms, "addxy", 8, |b, x, y| b.add(x, y, Flags::NONE));
    assert!(superoptimize(&mut m, f).is_none());
}

#[test]
fn out_of_subset_returns_none() {
    // A float target is outside the integer subset — lifted to `None`, never
    // mis-optimized.
    let mut m = Module::new("t");
    let mut syms = StrInterner::new();
    let fty = m.types_mut().float(crate::ir::FloatKind::F32);
    let sig = m.types_mut().func(vec![fty], fty, false);
    let sym = syms.intern("fadd");
    let f = m.declare_function(sym, sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.block_params(entry)[0];
        let r = b.bin(BinOp::FAdd, x, x, Flags::NONE);
        b.ret(Some(r));
    }
    assert!(superoptimize(&mut m, f).is_none());
}

// --- determinism -----------------------------------------------------------

#[test]
fn deterministic_across_runs() {
    let run = || {
        let mut m = Module::new("t");
        let mut syms = StrInterner::new();
        let f = add_unary(&mut m, &mut syms, "f", 8, |b, x| {
            let ty = b.value_type(x);
            let two = b.const_int(ty, Int::from_u64(2));
            b.mul(x, two, Flags::NONE)
        });
        superoptimize(&mut m, f).map(|out| func_ops(&out))
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "superoptimization must be reproducible");
    assert!(a.is_some());
}
