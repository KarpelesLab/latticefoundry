//! In-process execution tests for the JIT.
//!
//! Each test JIT-compiles an IR function and **calls it in-process**, asserting
//! the value the generated machine code returns — the in-process analogue of the
//! M5 native-executable execution tests. These run on x86-64 Linux (this
//! environment).

use super::Jit;
use crate::ir::inst::{Flags, IntPred};
use crate::ir::Module;
use crate::support::StrInterner;

/// `lfadd(a, b) = a + b` over `i32`.
fn build_add() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
    let f = m.declare_function(syms.intern("lfadd"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let s = b.add(a, bb, Flags::NONE);
        b.ret(Some(s));
    }
    (m, syms)
}

/// `lfmax(a, b)` over `i32`, via a branch diamond.
fn build_max() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
    let f = m.declare_function(syms.intern("lfmax"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let then_b = b.create_block(&[]);
        let else_b = b.create_block(&[]);
        let join = b.create_block(&[i32t]);
        let cond = b.icmp(IntPred::Sgt, a, bb);
        b.cond_br(cond, then_b, &[], else_b, &[]);
        b.switch_to(then_b);
        b.br(join, &[a]);
        b.switch_to(else_b);
        b.br(join, &[bb]);
        b.switch_to(join);
        let r = b.param(join, 0);
        b.ret(Some(r));
    }
    (m, syms)
}

/// `lfsum(n) = 0 + 1 + ... + (n-1)` over `i64` — a loop with back-edge args.
fn build_loop_sum() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("lfsum"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let n = b.param(entry, 0);
        let header = b.create_block(&[i64t, i64t]);
        let body = b.create_block(&[i64t, i64t]);
        let exit = b.create_block(&[i64t]);
        b.switch_to(entry);
        let zero = b.const_i64(i64t, 0);
        b.br(header, &[zero, zero]);
        b.switch_to(header);
        let acc = b.param(header, 0);
        let i = b.param(header, 1);
        let cond = b.icmp(IntPred::Slt, i, n);
        b.cond_br(cond, body, &[acc, i], exit, &[acc]);
        b.switch_to(body);
        let bacc = b.param(body, 0);
        let bi = b.param(body, 1);
        let new_acc = b.add(bacc, bi, Flags::NONE);
        let one = b.const_i64(i64t, 1);
        let new_i = b.add(bi, one, Flags::NONE);
        b.br(header, &[new_acc, new_i]);
        b.switch_to(exit);
        let result = b.param(exit, 0);
        b.ret(Some(result));
    }
    (m, syms)
}

/// A caller `lfcaller(x) = lfcallee(x) + lfcallee(x)` and callee `lfcallee(y) = y*3`.
fn build_call() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let callee = m.declare_function(syms.intern("lfcallee"), sig);
    let caller = m.declare_function(syms.intern("lfcaller"), sig);
    {
        let mut b = m.build(callee);
        let entry = b.create_entry_block();
        let y = b.param(entry, 0);
        let three = b.const_i64(i64t, 3);
        let r = b.mul(y, three, Flags::NONE);
        b.ret(Some(r));
    }
    {
        let mut b = m.build(caller);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let cref1 = b.func_ref(callee);
        let c1 = b.call(cref1, &[x], i64t).unwrap();
        let cref2 = b.func_ref(callee);
        let c2 = b.call(cref2, &[x], i64t).unwrap();
        let s = b.add(c1, c2, Flags::NONE);
        b.ret(Some(s));
    }
    (m, syms)
}

#[test]
fn jit_add_including_negatives() {
    let (m, syms) = build_add();
    let cm = Jit::new().compile(&m, &syms).unwrap();
    let add = cm.get_fn_i32_i32_i32("lfadd").expect("lfadd is compiled");
    assert_eq!(add(3, 4), 7);
    assert_eq!(add(-2, 10), 8);
    assert_eq!(add(0, 0), 0);
    assert_eq!(add(-5, -6), -11);
    assert_eq!(add(i32::MAX, 0), i32::MAX);
}

#[test]
fn jit_max_branch() {
    let (m, syms) = build_max();
    let cm = Jit::new().compile(&m, &syms).unwrap();
    let max = cm.get_fn_i32_i32_i32("lfmax").expect("lfmax is compiled");
    assert_eq!(max(3, 4), 4);
    assert_eq!(max(9, 2), 9);
    assert_eq!(max(-1, -5), -1);
    assert_eq!(max(7, 7), 7);
}

#[test]
fn jit_loop_sum() {
    let (m, syms) = build_loop_sum();
    let cm = Jit::new().compile(&m, &syms).unwrap();
    let sum = cm.get_fn_i64_i64("lfsum").expect("lfsum is compiled");
    assert_eq!(sum(0), 0);
    assert_eq!(sum(1), 0);
    assert_eq!(sum(5), 10);
    assert_eq!(sum(10), 45);
    assert_eq!(sum(100), 4950);
}

#[test]
fn jit_intra_module_call() {
    // The caller calls the callee through a Plt32 relocation that the JIT
    // resolves within the same mapping.
    let (m, syms) = build_call();
    let cm = Jit::new().compile(&m, &syms).unwrap();
    let callee = cm.get_fn_i64_i64("lfcallee").expect("lfcallee is compiled");
    let caller = cm.get_fn_i64_i64("lfcaller").expect("lfcaller is compiled");
    assert_eq!(callee(4), 12);
    assert_eq!(caller(2), 12); // 2*3 + 2*3
    assert_eq!(caller(7), 42); // 7*3 + 7*3
}

#[test]
fn unknown_function_is_none() {
    let (m, syms) = build_add();
    let cm = Jit::new().compile(&m, &syms).unwrap();
    assert!(cm.get_fn_i32_i32_i32("does_not_exist").is_none());
    assert!(cm.func_addr("lfadd").is_some());
}

#[test]
fn drop_frees_executable_memory() {
    // Compile, call, then drop; a botched unmap would fault. Repeating exercises
    // that each mapping is independently freed (no leak/double-free on drop).
    for _ in 0..32 {
        let (m, syms) = build_add();
        let cm = Jit::new().compile(&m, &syms).unwrap();
        let add = cm.get_fn_i32_i32_i32("lfadd").unwrap();
        assert_eq!(add(21, 21), 42);
        drop(add);
        drop(cm);
    }
}

#[test]
fn generated_code_is_deterministic() {
    // The same IR must lay down identical machine-code bytes on every compile.
    let (m, syms) = build_loop_sum();
    let obj_a = crate::target::x86_64::compile_module(&m, &syms);
    let obj_b = crate::target::x86_64::compile_module(&m, &syms);
    assert_eq!(obj_a, obj_b, "identical IR must yield identical objects");

    // And two live mappings of it must compute the same results.
    let cm_a = Jit::new().compile(&m, &syms).unwrap();
    let cm_b = Jit::new().compile(&m, &syms).unwrap();
    let sum_a = cm_a.get_fn_i64_i64("lfsum").unwrap();
    let sum_b = cm_b.get_fn_i64_i64("lfsum").unwrap();
    for n in [0i64, 1, 5, 10, 100] {
        assert_eq!(sum_a(n), sum_b(n));
    }
}
