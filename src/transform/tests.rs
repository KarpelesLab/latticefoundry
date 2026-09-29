//! Tests for the transform layer: the mem2reg SSA-construction pass and DCE,
//! their end-to-end value correctness (validated with the constant-propagation
//! analysis), verifier validity of every output, and determinism.

use std::fmt::Write as _;

use super::{Dce, FunctionTransform, FunctionTransformPass, Mem2Reg};

use crate::analysis::domains::ConstLattice;
use crate::analysis::solver::solve;
use crate::ir::inst::{Flags, InstKind, IntPred};
use crate::ir::value::Const;
use crate::ir::{FuncId, Function, InstId, Module, ValueId};
use crate::pass::{Changed, ModulePass, PassManager};
use crate::support::StrInterner;
use crate::verify::verify_module;

use puremp::Int;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_mem2reg(m: &mut Module, f: FuncId) -> Changed {
    let mut t = Mem2Reg;
    let (fresh, c) = m.map_function(f, |old, b| t.run(old, b));
    if c == Changed::Yes {
        m.replace_function(f, fresh);
    }
    c
}

fn run_dce(m: &mut Module, f: FuncId) -> Changed {
    let mut t = Dce;
    let (fresh, c) = m.map_function(f, |old, b| t.run(old, b));
    if c == Changed::Yes {
        m.replace_function(f, fresh);
    }
    c
}

/// Count the non-terminator instructions of a function matching `pred`.
fn count_kind(f: &Function, pred: impl Fn(&InstKind) -> bool) -> usize {
    let mut c = 0;
    for (_bid, blk) in f.blocks() {
        for &i in blk.insts() {
            if pred(&f.inst(i).kind) {
                c += 1;
            }
        }
    }
    c
}

fn n_alloca(f: &Function) -> usize {
    count_kind(f, |k| matches!(k, InstKind::Alloca { .. }))
}
fn n_load(f: &Function) -> usize {
    count_kind(f, |k| matches!(k, InstKind::Load { .. }))
}
fn n_store(f: &Function) -> usize {
    count_kind(f, |k| matches!(k, InstKind::Store { .. }))
}

/// The abstract constant of the operand of whichever block returns a value.
fn ret_value_const(m: &Module, f: FuncId) -> ConstLattice {
    let func = m.function(f);
    let r = solve::<ConstLattice>(func, m.types(), m.consts());
    for (_bid, blk) in func.blocks() {
        if let Some(t) = blk.terminator()
            && matches!(func.inst(t).kind, InstKind::Ret)
            && let Some(&v) = func.inst(t).operands().first()
        {
            return r.value(v).clone();
        }
    }
    panic!("no value-returning ret found");
}

fn assert_ret_int(m: &Module, f: FuncId, width: u32, expected: i64) {
    match ret_value_const(m, f) {
        ConstLattice::Const(Const::Int { value, .. }) => {
            assert_eq!(
                value.mod_2k(width),
                Int::from_i64(expected).mod_2k(width),
                "returned constant mismatch"
            );
        }
        other => panic!("expected constant int {expected}, got {other:?}"),
    }
}

/// A structural fingerprint of a function, for determinism checks.
fn canon(f: &Function) -> String {
    let mut s = String::new();
    for i in 0..f.inst_count() {
        let _ = writeln!(s, "I{i}: {:?}", f.inst(InstId::from_index(i)));
    }
    for (bid, b) in f.blocks() {
        let _ = writeln!(
            s,
            "B{}: params={:?} insts={:?} term={:?}",
            bid.index(),
            b.params(),
            b.insts(),
            b.terminator()
        );
    }
    for v in 0..f.value_count() {
        let val = f.value(ValueId::from_index(v));
        let _ = writeln!(s, "V{v}: {:?} : {:?}", val.def, val.ty);
    }
    s
}

// ---------------------------------------------------------------------------
// mem2reg
// ---------------------------------------------------------------------------

/// f() -> i32 { a = alloca; store a, 7; x = load a; ret x }  ==> ret 7
fn build_straight_line() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("m2r-straight");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        b.create_entry_block();
        let a = b.alloca(i32t);
        let seven = b.const_i64(i32t, 7);
        b.store(i32t, a, seven, 4);
        let x = b.load(i32t, a, 4);
        b.ret(Some(x));
    }
    (m, f)
}

#[test]
fn mem2reg_straight_line_promotes_to_constant() {
    let (mut m, f) = build_straight_line();
    assert!(verify_module(&m).is_ok());
    let c = run_mem2reg(&mut m, f);
    assert_eq!(c, Changed::Yes);

    let func = m.function(f);
    assert_eq!(n_alloca(func), 0, "alloca must be gone");
    assert_eq!(n_load(func), 0, "load must be gone");
    assert_eq!(n_store(func), 0, "store must be gone");
    assert!(verify_module(&m).is_ok(), "mem2reg output must verify");
    assert_ret_int(&m, f, 32, 7);
}

/// A diamond storing distinct constants on each arm, merged by a load. With an
/// unknown condition the join is genuinely non-constant (Top), but a block
/// parameter must appear at the merge.
fn build_diamond_unknown() -> (Module, FuncId, crate::ir::BlockId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("m2r-diamond");
    let i1 = m.types_mut().bool();
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i1], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    let merge;
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let then_b = b.create_block(&[]);
        let els_b = b.create_block(&[]);
        merge = b.create_block(&[]);
        let c = b.param(entry, 0);
        b.switch_to(entry);
        let a = b.alloca(i32t);
        b.cond_br(c, then_b, &[], els_b, &[]);
        b.switch_to(then_b);
        let ten = b.const_i64(i32t, 10);
        b.store(i32t, a, ten, 4);
        b.br(merge, &[]);
        b.switch_to(els_b);
        let twenty = b.const_i64(i32t, 20);
        b.store(i32t, a, twenty, 4);
        b.br(merge, &[]);
        b.switch_to(merge);
        let x = b.load(i32t, a, 4);
        b.ret(Some(x));
    }
    (m, f, merge)
}

#[test]
fn mem2reg_diamond_places_block_parameter() {
    let (mut m, f, merge) = build_diamond_unknown();
    assert_eq!(m.function(f).block(merge).params().len(), 0);
    let c = run_mem2reg(&mut m, f);
    assert_eq!(c, Changed::Yes);

    let func = m.function(f);
    assert_eq!(n_alloca(func), 0);
    assert_eq!(n_load(func), 0);
    assert_eq!(n_store(func), 0);
    assert_eq!(
        func.block(merge).params().len(),
        1,
        "the merge block must gain a parameter for the promoted slot"
    );
    assert!(verify_module(&m).is_ok());
    // Two distinct incoming values under an unknown condition ⇒ not a constant.
    assert!(ret_value_const(&m, f).is_top(), "merged value must be Top");
}

/// The same diamond but with a *constant* (true) condition: SCCP prunes the
/// false arm, so the promoted merge parameter resolves to the taken constant.
#[test]
fn mem2reg_diamond_constant_condition_folds_end_to_end() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("m2r-diamond-c");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    let merge;
    {
        let mut b = m.build(f);
        let _entry = b.create_entry_block();
        let then_b = b.create_block(&[]);
        let els_b = b.create_block(&[]);
        merge = b.create_block(&[]);
        let a = b.alloca(i32t);
        let t = b.const_bool(true);
        b.cond_br(t, then_b, &[], els_b, &[]);
        b.switch_to(then_b);
        let ten = b.const_i64(i32t, 10);
        b.store(i32t, a, ten, 4);
        b.br(merge, &[]);
        b.switch_to(els_b);
        let twenty = b.const_i64(i32t, 20);
        b.store(i32t, a, twenty, 4);
        b.br(merge, &[]);
        b.switch_to(merge);
        let x = b.load(i32t, a, 4);
        b.ret(Some(x));
    }
    run_mem2reg(&mut m, f);
    assert_eq!(m.function(f).block(merge).params().len(), 1);
    assert!(verify_module(&m).is_ok());
    assert_ret_int(&m, f, 32, 10);
}

/// A counting loop carrying its induction variable in an alloca. mem2reg must
/// introduce a header parameter and pass the incremented value on the back edge.
#[test]
fn mem2reg_loop_introduces_header_parameter_and_back_edge_arg() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("m2r-loop");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    let (header, body);
    {
        let mut b = m.build(f);
        let _entry = b.create_entry_block();
        header = b.create_block(&[]);
        body = b.create_block(&[]);
        let exit = b.create_block(&[]);
        let a = b.alloca(i32t);
        let zero = b.const_i64(i32t, 0);
        b.store(i32t, a, zero, 4);
        b.br(header, &[]);
        b.switch_to(header);
        let i = b.load(i32t, a, 4);
        let three = b.const_i64(i32t, 3);
        let cond = b.icmp(IntPred::Slt, i, three);
        b.cond_br(cond, body, &[], exit, &[]);
        b.switch_to(body);
        let i2 = b.load(i32t, a, 4);
        let one = b.const_i64(i32t, 1);
        let i3 = b.add(i2, one, Flags::NONE);
        b.store(i32t, a, i3, 4);
        b.br(header, &[]);
        b.switch_to(exit);
        let r = b.load(i32t, a, 4);
        b.ret(Some(r));
    }
    let c = run_mem2reg(&mut m, f);
    assert_eq!(c, Changed::Yes);

    let func = m.function(f);
    assert_eq!(n_alloca(func), 0);
    assert_eq!(n_load(func), 0);
    assert_eq!(n_store(func), 0);
    assert_eq!(func.block(header).params().len(), 1, "loop header needs a parameter");

    // The back edge (body → header) must now carry exactly one argument.
    let term = func.block(body).terminator().expect("body terminator");
    assert_eq!(
        func.inst(term).operands().len(),
        1,
        "back edge must pass the incremented induction value"
    );
    assert!(verify_module(&m).is_ok(), "loop mem2reg output must verify");
}

#[test]
fn mem2reg_leaves_address_taken_alloca_via_ptr_add() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("m2r-escape-ptradd");
    let i32t = m.types_mut().int(32);
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        b.create_entry_block();
        let a = b.alloca(i32t);
        let five = b.const_i64(i32t, 5);
        b.store(i32t, a, five, 4);
        let zero = b.const_i64(i64t, 0);
        let p = b.ptr_add(a, zero, true);
        let x = b.load(i32t, p, 4);
        b.ret(Some(x));
    }
    let c = run_mem2reg(&mut m, f);
    assert_eq!(c, Changed::No, "address-taken alloca is not promotable");
    assert_eq!(n_alloca(m.function(f)), 1, "alloca must be left untouched");
    assert!(verify_module(&m).is_ok());
}

#[test]
fn mem2reg_leaves_alloca_passed_to_call() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("m2r-escape-call");
    let i32t = m.types_mut().int(32);
    let ptr = m.types_mut().ptr();
    let void = m.types_mut().void();
    let g_sig = m.types_mut().func(vec![ptr], void, false);
    let g = m.declare_function(syms.intern("g"), g_sig);
    let f_sig = m.types_mut().func(vec![], void, false);
    let f = m.declare_function(syms.intern("f"), f_sig);
    {
        let mut b = m.build(f);
        b.create_entry_block();
        let a = b.alloca(i32t);
        let five = b.const_i64(i32t, 5);
        b.store(i32t, a, five, 4);
        let gref = b.func_ref(g);
        b.call(gref, &[a], void);
        b.ret(None);
    }
    let c = run_mem2reg(&mut m, f);
    assert_eq!(c, Changed::No, "alloca whose address escapes to a call is not promotable");
    assert_eq!(n_alloca(m.function(f)), 1);
    assert!(verify_module(&m).is_ok());
}

#[test]
fn mem2reg_is_deterministic() {
    let (mut m1, f1) = build_straight_line();
    let (mut m2, f2) = build_straight_line();
    run_mem2reg(&mut m1, f1);
    run_mem2reg(&mut m2, f2);
    assert_eq!(canon(m1.function(f1)), canon(m2.function(f2)));

    // And on a non-trivial (loop) shape via the pass driver.
    let (a, fa) = loop_module();
    let (b, fb) = loop_module();
    let (mut a, mut b) = (a, b);
    FunctionTransformPass::new(Mem2Reg).run(&mut a);
    FunctionTransformPass::new(Mem2Reg).run(&mut b);
    assert_eq!(canon(a.function(fa)), canon(b.function(fb)));
}

fn loop_module() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("m2r-loopmod");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let _entry = b.create_entry_block();
        let header = b.create_block(&[]);
        let body = b.create_block(&[]);
        let exit = b.create_block(&[]);
        let a = b.alloca(i32t);
        let zero = b.const_i64(i32t, 0);
        b.store(i32t, a, zero, 4);
        b.br(header, &[]);
        b.switch_to(header);
        let i = b.load(i32t, a, 4);
        let three = b.const_i64(i32t, 3);
        let cond = b.icmp(IntPred::Slt, i, three);
        b.cond_br(cond, body, &[], exit, &[]);
        b.switch_to(body);
        let i2 = b.load(i32t, a, 4);
        let one = b.const_i64(i32t, 1);
        let i3 = b.add(i2, one, Flags::NONE);
        b.store(i32t, a, i3, 4);
        b.br(header, &[]);
        b.switch_to(exit);
        let r = b.load(i32t, a, 4);
        b.ret(Some(r));
    }
    (m, f)
}

#[test]
fn mem2reg_via_pass_manager_invalidates_and_verifies() {
    let (mut m, f) = build_straight_line();
    let mut pm = PassManager::new();
    // Seed a cached analysis, then run the transform pass over the module.
    {
        let (types, consts, func) = (m.types(), m.consts(), m.function(f));
        let _ = pm.analyses_mut().get_or_compute(
            &crate::analysis::ConstantPropagation,
            f,
            func,
            types,
            consts,
        );
    }
    assert!(!pm.analyses().is_empty());
    pm.add(Box::new(FunctionTransformPass::new(Mem2Reg)));
    pm.run(&mut m);
    assert!(pm.analyses().is_empty(), "a changing pass must invalidate the cache");
    assert_eq!(n_alloca(m.function(f)), 0);
    assert!(verify_module(&m).is_ok());
    assert_ret_int(&m, f, 32, 7);
}

// ---------------------------------------------------------------------------
// DCE
// ---------------------------------------------------------------------------

/// f(x) -> i32 { d1 = x+1; d2 = d1*2; r = x+5; ret r }
fn build_dead_chain() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("dce-chain");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let one = b.const_i64(i32t, 1);
        let two = b.const_i64(i32t, 2);
        let five = b.const_i64(i32t, 5);
        let d1 = b.add(x, one, Flags::NONE);
        let _d2 = b.mul(d1, two, Flags::NONE);
        let r = b.add(x, five, Flags::NONE);
        b.ret(Some(r));
    }
    (m, f)
}

#[test]
fn dce_removes_dead_pure_chain() {
    let (mut m, f) = build_dead_chain();
    let before = count_kind(m.function(f), |_| true);
    assert_eq!(before, 3);
    let c = run_dce(&mut m, f);
    assert_eq!(c, Changed::Yes);
    let func = m.function(f);
    assert_eq!(count_kind(func, |_| true), 1, "only the returned add survives");
    assert_eq!(count_kind(func, |k| matches!(k, InstKind::Bin(crate::ir::BinOp::Mul))), 0);
    assert!(verify_module(&m).is_ok());
    // r == x + 5 is still non-constant (x unknown) but present and correct.
    assert!(ret_value_const(&m, f).is_top());
}

#[test]
fn dce_keeps_side_effecting_store_but_drops_dead_value() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("dce-store");
    let i32t = m.types_mut().int(32);
    let ptr = m.types_mut().ptr();
    let void = m.types_mut().void();
    let sig = m.types_mut().func(vec![ptr, i32t], void, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let p = b.param(entry, 0);
        let x = b.param(entry, 1);
        b.store(i32t, p, x, 4);
        let one = b.const_i64(i32t, 1);
        let _dead = b.add(x, one, Flags::NONE);
        b.ret(None);
    }
    let c = run_dce(&mut m, f);
    assert_eq!(c, Changed::Yes);
    let func = m.function(f);
    assert_eq!(n_store(func), 1, "the store has a side effect and must survive");
    assert_eq!(count_kind(func, |k| matches!(k, InstKind::Bin(_))), 0, "the dead add is gone");
    assert!(verify_module(&m).is_ok());
}

#[test]
fn dce_iterates_to_fixpoint() {
    // A three-deep dead chain: removing d3 exposes d2, then d1.
    let mut syms = StrInterner::new();
    let mut m = Module::new("dce-fixpoint");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let one = b.const_i64(i32t, 1);
        let two = b.const_i64(i32t, 2);
        let three = b.const_i64(i32t, 3);
        let d1 = b.add(x, one, Flags::NONE);
        let d2 = b.add(d1, two, Flags::NONE);
        let _d3 = b.add(d2, three, Flags::NONE);
        b.ret(Some(x));
    }
    let c = run_dce(&mut m, f);
    assert_eq!(c, Changed::Yes);
    assert_eq!(count_kind(m.function(f), |_| true), 0, "the whole dead chain is removed");
    assert!(verify_module(&m).is_ok());
}

#[test]
fn dce_removes_dead_load() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("dce-load");
    let i32t = m.types_mut().int(32);
    let ptr = m.types_mut().ptr();
    let sig = m.types_mut().func(vec![ptr], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let p = b.param(entry, 0);
        let _l = b.load(i32t, p, 4);
        let zero = b.const_i64(i32t, 0);
        b.ret(Some(zero));
    }
    let c = run_dce(&mut m, f);
    assert_eq!(c, Changed::Yes);
    assert_eq!(n_load(m.function(f)), 0, "an unused load is pure and removable");
    assert!(verify_module(&m).is_ok());
    assert_ret_int(&m, f, 32, 0);
}

#[test]
fn dce_keeps_everything_live() {
    // No dead code ⇒ Changed::No, body untouched.
    let mut syms = StrInterner::new();
    let mut m = Module::new("dce-nolive");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let one = b.const_i64(i32t, 1);
        let r = b.add(x, one, Flags::NONE);
        b.ret(Some(r));
    }
    let c = run_dce(&mut m, f);
    assert_eq!(c, Changed::No);
    assert_eq!(count_kind(m.function(f), |_| true), 1);
    assert!(verify_module(&m).is_ok());
}

#[test]
fn dce_is_deterministic() {
    let (mut m1, f1) = build_dead_chain();
    let (mut m2, f2) = build_dead_chain();
    run_dce(&mut m1, f1);
    run_dce(&mut m2, f2);
    assert_eq!(canon(m1.function(f1)), canon(m2.function(f2)));
}

// ---------------------------------------------------------------------------
// `syscall`: an opaque effect every pass must keep, in place and in order.
// ---------------------------------------------------------------------------

/// `f(n) -> i64`: a stack slot initialized to 5 whose address escapes into a
/// syscall, then a loop running two syscalls per iteration with loop-invariant
/// operands (one result used, one unused), then a reload of the slot:
///
/// ```text
/// entry:  slot = alloca i64; store 5 -> slot; s0 = syscall 0, 0, slot, 8
///         br header(0, 0)
/// header(i, acc): cond_br (i < n), body, exit
/// body:   r = syscall 1, 1, slot, 8; syscall 39 (dead); br header(i+1, acc+r)
/// exit:   v = load slot; ret v + acc
/// ```
fn build_syscall_loop() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("sys");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let n = b.param(entry, 0);
        let header = b.create_block(&[i64t, i64t]);
        let body = b.create_block(&[]);
        let exit = b.create_block(&[]);

        b.switch_to(entry);
        let slot = b.alloca(i64t);
        let five = b.const_i64(i64t, 5);
        b.store(i64t, slot, five, 8);
        let zero = b.const_i64(i64t, 0);
        let eight = b.const_i64(i64t, 8);
        // read(0, slot, 8): the kernel may overwrite the slot.
        b.syscall(zero, &[zero, slot, eight]);
        b.br(header, &[zero, zero]);

        b.switch_to(header);
        let i = b.param(header, 0);
        let acc = b.param(header, 1);
        let c = b.icmp(IntPred::Slt, i, n);
        b.cond_br(c, body, &[], exit, &[]);

        b.switch_to(body);
        let one = b.const_i64(i64t, 1);
        let r = b.syscall(one, &[one, slot, eight]);
        let getpid = b.const_i64(i64t, 39);
        b.syscall(getpid, &[]);
        let acc2 = b.add(acc, r, Flags::NONE);
        let i2 = b.add(i, one, Flags::NONE);
        b.br(header, &[i2, acc2]);

        b.switch_to(exit);
        let v = b.load(i64t, slot, 8);
        let s = b.add(v, acc, Flags::NONE);
        b.ret(Some(s));
    }
    (m, f)
}

fn n_syscall(f: &Function) -> usize {
    count_kind(f, |k| matches!(k, InstKind::Syscall))
}

/// The blocks holding a syscall, as the number of syscalls per such block.
fn syscalls_per_block(f: &Function) -> Vec<usize> {
    f.blocks()
        .map(|(_, blk)| blk.insts().iter().filter(|&&i| matches!(f.inst(i).kind, InstKind::Syscall)).count())
        .filter(|&c| c > 0)
        .collect()
}

#[test]
fn every_pass_keeps_syscalls_in_place() {
    for name in ["dce", "sccp", "licm", "egraph", "simplify_cfg", "mem2reg", "inline"] {
        let (mut m, f) = build_syscall_loop();
        let before = syscalls_per_block(m.function(f));
        crate::transform::pipeline::run_passes(
            &mut m,
            vec![crate::transform::pipeline::pass_by_name(name).unwrap()],
        );
        assert!(verify_module(&m).is_ok(), "{name} keeps the module valid");
        let func = m.function(f);
        assert_eq!(n_syscall(func), 3, "{name} must keep all three syscalls");
        // One in the entry, two together in the loop body: nothing hoisted
        // (LICM), sunk, split or merged across blocks.
        assert_eq!(syscalls_per_block(func), before, "{name} must not move a syscall");
    }
}

#[test]
fn syscall_escapes_its_pointer_and_clobbers_memory() {
    // Even the full -O3 pipeline must keep the slot in memory (its address
    // escaped into a syscall that may write it) and must not forward the stored
    // 5 into the final load, nor fold the syscall-derived result.
    let (mut m, f) = build_syscall_loop();
    crate::transform::pipeline::optimize(&mut m, crate::transform::pipeline::OptLevel::O3);
    assert!(verify_module(&m).is_ok());
    let func = m.function(f);
    assert_eq!(n_syscall(func), 3, "no syscall dropped at -O3");
    assert_eq!(n_alloca(func), 1, "the escaped slot is not promoted");
    assert_eq!(n_load(func), 1, "the reload after the syscalls survives");
    assert!(ret_value_const(&m, f).is_top(), "a syscall-dependent result is unknown");
}

#[test]
fn syscall_result_is_never_a_constant() {
    // `ret syscall(39)`: the constant domain (SCCP's lattice) knows nothing.
    let mut syms = StrInterner::new();
    let mut m = Module::new("sys-top");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![], i64t, false);
    let f = m.declare_function(syms.intern("g"), sig);
    {
        let mut b = m.build(f);
        b.create_entry_block();
        let nr = b.const_i64(i64t, 39);
        let r = b.syscall(nr, &[]);
        b.ret(Some(r));
    }
    assert!(ret_value_const(&m, f).is_top());
    crate::transform::pipeline::optimize(&mut m, crate::transform::pipeline::OptLevel::O2);
    assert_eq!(n_syscall(m.function(f)), 1);
    assert!(ret_value_const(&m, f).is_top());
}
