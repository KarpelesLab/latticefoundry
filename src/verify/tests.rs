//! Tests for the structural verifier: valid modules verify clean, a battery of
//! malformed modules is each rejected with a precise diagnostic, and the
//! dominator tree is checked against hand-built diamond and loop CFGs.

use super::cfg::DomTree;
use super::{verify_function, verify_module};
use crate::ir::inst::{CastOp, Flags, IntPred};
use crate::ir::types::FloatKind;
use crate::ir::value::FloatBits;
use crate::ir::{BlockId, FuncId, Function, Module};
use crate::support::StrInterner;

/// Assert the diagnostics are all errors and that exactly one of them contains
/// `needle`, returning nothing. Prints the messages on failure.
fn assert_one_error(diags: &[crate::support::diagnostics::Diagnostic], needle: &str) {
    let msgs: Vec<&str> = diags.iter().map(|d| d.message.as_str()).collect();
    assert!(diags.iter().all(|d| d.is_error()), "all diagnostics must be errors: {msgs:?}");
    let hits = msgs.iter().filter(|m| m.contains(needle)).count();
    assert_eq!(hits, 1, "expected exactly one diagnostic containing {needle:?}, got {msgs:?}");
}

fn assert_clean(module: &Module) {
    match verify_module(module) {
        Ok(()) => {}
        Err(diags) => {
            let msgs: Vec<&str> = diags.iter().map(|d| d.message.as_str()).collect();
            panic!("expected a clean module, got: {msgs:?}");
        }
    }
}

// --- valid modules ----------------------------------------------------------

#[test]
fn valid_trivial_void_function() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let void = module.types_mut().void();
    let sig = module.types_mut().func(vec![], void, false);
    let f = module.declare_function(syms.intern("main"), sig);
    {
        let mut b = module.build(f);
        b.create_entry_block();
        b.ret(None);
    }
    assert_clean(&module);
}

#[test]
fn valid_diamond_with_block_args() {
    let module = build_diamond();
    assert_clean(&module);
}

#[test]
fn valid_loop_with_back_edge() {
    let module = build_loop();
    assert_clean(&module);
}

#[test]
fn valid_memory_and_calls() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("mem");
    let i64_ = module.types_mut().int(64);
    let unary = module.types_mut().func(vec![i64_], i64_, false);
    let g = module.declare_function(syms.intern("g"), unary);
    let f = module.declare_function(syms.intern("f"), unary);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let v = b.param(entry, 0);
        let p = b.alloca(i64_);
        b.store(i64_, p, v, 8);
        let r = b.load(i64_, p, 8);
        let gref = b.func_ref(g);
        let c = b.call(gref, &[r], i64_).expect("call returns i64");
        b.ret(Some(c));
    }
    assert_clean(&module);
}

// --- malformed modules ------------------------------------------------------

#[test]
fn missing_terminator_is_rejected() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let void = module.types_mut().void();
    let sig = module.types_mut().func(vec![], void, false);
    let f = module.declare_function(syms.intern("f"), sig);
    {
        let mut b = module.build(f);
        b.create_entry_block();
        // No terminator emitted.
    }
    let diags = verify_function(&module, f);
    assert_one_error(&diags, "has no terminator");
}

#[test]
fn dangling_successor_is_rejected() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let void = module.types_mut().void();
    let sig = module.types_mut().func(vec![], void, false);
    let f = module.declare_function(syms.intern("f"), sig);
    {
        let mut b = module.build(f);
        b.create_entry_block();
        b.br(BlockId::from_index(99), &[]);
    }
    let diags = verify_function(&module, f);
    assert_one_error(&diags, "nonexistent block #99");
}

#[test]
fn wrong_block_arg_arity_is_rejected() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let void = module.types_mut().void();
    let i32_ = module.types_mut().int(32);
    let sig = module.types_mut().func(vec![i32_], void, false);
    let f = module.declare_function(syms.intern("f"), sig);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let target = b.create_block(&[i32_]); // one parameter
        b.switch_to(entry);
        b.br(target, &[]); // ...but zero arguments
        b.switch_to(target);
        b.ret(None);
    }
    let diags = verify_function(&module, f);
    assert_one_error(&diags, "passes 0 argument(s) but the block has 1");
}

#[test]
fn type_mismatched_block_arg_is_rejected() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let void = module.types_mut().void();
    let i32_ = module.types_mut().int(32);
    let i64_ = module.types_mut().int(64);
    let sig = module.types_mut().func(vec![i32_], void, false);
    let f = module.declare_function(syms.intern("f"), sig);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let target = b.create_block(&[i64_]); // expects i64
        b.switch_to(entry);
        b.br(target, &[x]); // passes i32
        b.switch_to(target);
        b.ret(None);
    }
    let diags = verify_function(&module, f);
    assert_one_error(&diags, "has type i32 but the parameter is i64");
}

#[test]
fn use_not_dominated_is_rejected() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let i1 = module.types_mut().bool();
    let i32_ = module.types_mut().int(32);
    let sig = module.types_mut().func(vec![i1, i32_], i32_, false);
    let f = module.declare_function(syms.intern("f"), sig);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let c = b.param(entry, 0);
        let x = b.param(entry, 1);
        let b1 = b.create_block(&[]);
        let b2 = b.create_block(&[]);
        let exit = b.create_block(&[i32_]);

        b.switch_to(entry);
        b.cond_br(c, b1, &[], b2, &[]);

        // v is defined only on the b1 path.
        b.switch_to(b1);
        let v = b.add(x, x, Flags::NONE);
        b.br(exit, &[v]);

        // b2 uses v, which b1 does not dominate.
        b.switch_to(b2);
        let w = b.add(v, x, Flags::NONE);
        b.br(exit, &[w]);

        b.switch_to(exit);
        let r = b.param(exit, 0);
        b.ret(Some(r));
    }
    let diags = verify_function(&module, f);
    assert_one_error(&diags, "does not dominate the use");
}

#[test]
fn cond_br_on_non_i1_is_rejected() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let void = module.types_mut().void();
    let i32_ = module.types_mut().int(32);
    let sig = module.types_mut().func(vec![i32_], void, false);
    let f = module.declare_function(syms.intern("f"), sig);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0); // i32, not i1
        let b1 = b.create_block(&[]);
        let b2 = b.create_block(&[]);
        b.switch_to(entry);
        b.cond_br(x, b1, &[], b2, &[]);
        b.switch_to(b1);
        b.ret(None);
        b.switch_to(b2);
        b.ret(None);
    }
    let diags = verify_function(&module, f);
    assert_one_error(&diags, "cond_br condition must be i1");
}

#[test]
fn call_arity_mismatch_is_rejected() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let i64_ = module.types_mut().int(64);
    let unary = module.types_mut().func(vec![i64_], i64_, false);
    let g = module.declare_function(syms.intern("g"), unary);
    let f = module.declare_function(syms.intern("f"), unary);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let gref = b.func_ref(g);
        let r = b.call(gref, &[a, a], i64_).expect("call has result"); // two args, expects one
        b.ret(Some(r));
    }
    let diags = verify_function(&module, f);
    assert_one_error(&diags, "passes 2 argument(s) but callee");
}

#[test]
fn widening_trunc_cast_is_rejected() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let i32_ = module.types_mut().int(32);
    let i64_ = module.types_mut().int(64);
    let sig = module.types_mut().func(vec![i32_], i64_, false);
    let f = module.declare_function(syms.intern("f"), sig);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let y = b.cast(CastOp::Trunc, x, i64_); // trunc that widens: invalid
        b.ret(Some(y));
    }
    let diags = verify_function(&module, f);
    assert_one_error(&diags, "trunc from i32 to i64 is not a valid conversion");
}

#[test]
fn branch_into_entry_is_rejected() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let void = module.types_mut().void();
    let sig = module.types_mut().func(vec![], void, false);
    let f = module.declare_function(syms.intern("f"), sig);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let other = b.create_block(&[]);
        b.switch_to(entry);
        b.br(other, &[]);
        b.switch_to(other);
        b.br(entry, &[]); // illegal back-branch into the entry block
    }
    let diags = verify_function(&module, f);
    assert_one_error(&diags, "branches to the entry block");
}

#[test]
fn float_constant_typing_is_checked() {
    // A well-typed float constant module verifies clean (exercises the constant
    // and fcmp paths).
    let mut syms = StrInterner::new();
    let mut module = Module::new("m");
    let i1 = module.types_mut().bool();
    let f32 = module.types_mut().float(FloatKind::F32);
    let sig = module.types_mut().func(vec![f32], i1, false);
    let f = module.declare_function(syms.intern("f"), sig);
    {
        use crate::ir::inst::FloatPred;
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let one = b.const_float(f32, FloatBits::F32(1.0f32.to_bits()));
        let cmp = b.fcmp(FloatPred::Oeq, x, one, Flags::NONE);
        b.ret(Some(cmp));
    }
    assert_clean(&module);
}

// --- dominator tree unit tests ----------------------------------------------

/// Convenience: dominance over block indices in `func`.
fn dom(func: &Function) -> DomTree {
    DomTree::build(func)
}

#[test]
fn dominator_tree_on_diamond() {
    let module = build_diamond();
    let func = module.function(FuncId::from_index(0));
    let d = dom(func);
    // Blocks: 0 = entry, 1 = t, 2 = f, 3 = merge (creation order below).
    for b in 0..4 {
        assert!(d.is_reachable(b), "block {b} should be reachable");
        assert!(d.dominates(0, b), "entry dominates block {b}");
        assert!(d.dominates(b, b), "dominance is reflexive for {b}");
    }
    // Neither arm dominates the merge, and they do not dominate each other.
    assert!(!d.dominates(1, 3));
    assert!(!d.dominates(2, 3));
    assert!(!d.dominates(1, 2));
    assert!(!d.dominates(2, 1));
    // The merge dominates only itself.
    assert!(!d.dominates(3, 0));
    assert!(!d.dominates(3, 1));
}

#[test]
fn dominator_tree_on_loop() {
    let module = build_loop();
    let func = module.function(FuncId::from_index(0));
    let d = dom(func);
    // Blocks: 0 = entry, 1 = header, 2 = body, 3 = exit.
    assert!(d.dominates(0, 1) && d.dominates(0, 2) && d.dominates(0, 3));
    assert!(d.dominates(1, 2), "header dominates body");
    assert!(d.dominates(1, 3), "header dominates exit");
    // The back edge must not make the body dominate the header.
    assert!(!d.dominates(2, 1));
    assert!(!d.dominates(2, 3));
}

// --- shared CFG builders ----------------------------------------------------

/// entry(c: i1, x: i32) -> i32 with a diamond into a `merge(m: i32)` block.
fn build_diamond() -> Module {
    let mut syms = StrInterner::new();
    let mut module = Module::new("diamond");
    let i1 = module.types_mut().bool();
    let i32_ = module.types_mut().int(32);
    let sig = module.types_mut().func(vec![i1, i32_], i32_, false);
    let f = module.declare_function(syms.intern("d"), sig);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let c = b.param(entry, 0);
        let x = b.param(entry, 1);
        let t = b.create_block(&[]);
        let fl = b.create_block(&[]);
        let merge = b.create_block(&[i32_]);

        b.switch_to(entry);
        b.cond_br(c, t, &[], fl, &[]);

        b.switch_to(t);
        let a = b.add(x, x, Flags::NONE);
        b.br(merge, &[a]);

        b.switch_to(fl);
        let s = b.sub(x, x, Flags::NONE);
        b.br(merge, &[s]);

        b.switch_to(merge);
        let m = b.param(merge, 0);
        b.ret(Some(m));
    }
    module
}

/// sum(n: i64) -> i64 loop with a back edge (mirrors the IR builder test).
fn build_loop() -> Module {
    let mut syms = StrInterner::new();
    let mut module = Module::new("loop");
    let i64_ = module.types_mut().int(64);
    let sig = module.types_mut().func(vec![i64_], i64_, false);
    let f = module.declare_function(syms.intern("sum"), sig);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let n = b.param(entry, 0);
        let header = b.create_block(&[i64_, i64_]);
        let body = b.create_block(&[i64_, i64_]);
        let exit = b.create_block(&[i64_]);

        b.switch_to(entry);
        let zero = b.const_i64(i64_, 0);
        b.br(header, &[zero, zero]);

        b.switch_to(header);
        let acc = b.param(header, 0);
        let i = b.param(header, 1);
        let cond = b.icmp(IntPred::Slt, i, n);
        b.cond_br(cond, body, &[acc, i], exit, &[acc]);

        b.switch_to(body);
        let bacc = b.param(body, 0);
        let bi = b.param(body, 1);
        let new_acc = b.add(bacc, bi, Flags::nsw());
        let one = b.const_i64(i64_, 1);
        let new_i = b.add(bi, one, Flags::nsw());
        b.br(header, &[new_acc, new_i]);

        b.switch_to(exit);
        let result = b.param(exit, 0);
        b.ret(Some(result));
    }
    module
}
