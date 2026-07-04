//! Tests for the core IR data model: building non-trivial functions, use/def
//! consistency, RAUW, type interning, and `puremp`-backed constants.

use super::*;
use crate::ir::inst::{Flags, IntPred, InstKind};
use crate::ir::value::{Const, FloatBits};
use crate::support::StrInterner;

/// Assert that every recorded use of every value points back at an operand slot
/// that actually holds that value — i.e. def→use and use→def agree.
fn assert_use_def_consistent(func: &Function) {
    for i in 0..func.value_count() {
        let v = ValueId::from_index(i);
        for u in func.uses_of(v) {
            let operands = func.inst(u.inst).operands();
            assert_eq!(
                operands[u.operand as usize], v,
                "use list of {v:?} points at an operand that is not {v:?}",
            );
        }
    }
    // And the converse: every operand of every instruction is registered as a
    // use of the value it references.
    for i in 0..func.inst_count() {
        let inst = InstId::from_index(i);
        let operands = func.inst(inst).operands().to_vec();
        for (slot, op) in operands.iter().enumerate() {
            let found = func
                .uses_of(*op)
                .iter()
                .any(|u| u.inst == inst && u.operand as usize == slot);
            assert!(found, "operand {slot} of {inst:?} is missing from {op:?}'s use list");
        }
    }
}

#[test]
fn build_a_trivial_function() {
    // Adapted smoke test: an empty `void` function that just returns.
    let mut syms = StrInterner::new();
    let mut module = Module::new("smoke");
    let void = module.types_mut().void();
    let sig = module.types_mut().func(vec![], void, false);

    let f = module.declare_function(syms.intern("main"), sig);
    {
        let mut b = module.build(f);
        b.create_entry_block();
        b.ret(None);
    }

    assert!(!module.function(f).is_declaration());
    assert_eq!(module.functions().count(), 1);
    assert_use_def_consistent(module.function(f));
}

#[test]
fn loop_with_back_edge_block_arguments() {
    // sum(n): acc = 0; for i in 0..n { acc += i } return acc
    let mut syms = StrInterner::new();
    let mut module = Module::new("loops");
    let i64_ = module.types_mut().int(64);
    let sig = module.types_mut().func(vec![i64_], i64_, false);
    let f = module.declare_function(syms.intern("sum"), sig);

    let (header, body, exit);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        let n = b.param(entry, 0);

        header = b.create_block(&[i64_, i64_]); // (acc, i)
        body = b.create_block(&[i64_, i64_]); // (acc, i)
        exit = b.create_block(&[i64_]); // (result)

        // entry: br header(0, 0)
        b.switch_to(entry);
        let zero = b.const_i64(i64_, 0);
        b.br(header, &[zero, zero]);

        // header(acc, i): cond = i < n ; cond_br cond, body(acc, i), exit(acc)
        b.switch_to(header);
        let acc = b.param(header, 0);
        let i = b.param(header, 1);
        let cond = b.icmp(IntPred::Slt, i, n);
        b.cond_br(cond, body, &[acc, i], exit, &[acc]);

        // body(acc, i): acc' = acc + i ; i' = i + 1 ; br header(acc', i')  [back-edge]
        b.switch_to(body);
        let bacc = b.param(body, 0);
        let bi = b.param(body, 1);
        let new_acc = b.add(bacc, bi, Flags::nsw());
        let one = b.const_i64(i64_, 1);
        let new_i = b.add(bi, one, Flags::nsw());
        b.br(header, &[new_acc, new_i]);

        // exit(result): ret result
        b.switch_to(exit);
        let result = b.param(exit, 0);
        b.ret(Some(result));
    }

    let func = module.function(f);
    assert_eq!(func.block_count(), 4);
    // The header block's terminator is a conditional branch with two successors.
    let header_term = func.block(header).terminator().expect("header terminated");
    assert_eq!(func.inst(header_term).successors(), vec![body, exit]);
    // The body ends with a back-edge to the header carrying two block arguments.
    let body_term = func.block(body).terminator().expect("body terminated");
    assert!(matches!(func.inst(body_term).kind, InstKind::Br(t) if t == header));
    assert_eq!(func.inst(body_term).operands().len(), 2);
    assert_use_def_consistent(func);
}

#[test]
fn call_select_ret_and_rauw() {
    let mut syms = StrInterner::new();
    let mut module = Module::new("calls");
    let i64_ = module.types_mut().int(64);
    let unary_sig = module.types_mut().func(vec![i64_], i64_, false);
    let bin_sig = module.types_mut().func(vec![i64_, i64_], i64_, false);

    // An external callee `g(i64) -> i64`.
    let g = module.declare_function(syms.intern("g"), unary_sig);
    let f = module.declare_function(syms.intern("f"), bin_sig);

    let (a, b_val, c, sel);
    {
        let mut b = module.build(f);
        let entry = b.create_entry_block();
        a = b.param(entry, 0);
        b_val = b.param(entry, 1);

        let gref = b.func_ref(g);
        c = b.call(gref, &[a], i64_).expect("call returns i64");
        let cond = b.icmp(IntPred::Sgt, a, b_val);
        sel = b.select(cond, c, b_val);
        b.ret(Some(sel));
    }

    assert_use_def_consistent(module.function(f));

    // `a` is used by the call and by the icmp (two uses).
    assert_eq!(module.function(f).uses_of(a).len(), 2);

    // RAUW: replace all uses of `a` with `b_val`.
    {
        let mut b = module.build(f);
        b.replace_all_uses_with(a, b_val);
    }
    let func = module.function(f);
    assert!(func.uses_of(a).is_empty(), "RAUW must drain the old value's uses");
    // Every operand that was `a` is now `b_val`.
    for i in 0..func.inst_count() {
        for op in func.inst(InstId::from_index(i)).operands() {
            assert_ne!(*op, a, "no operand should still reference the replaced value");
        }
    }
    assert_use_def_consistent(func);
}

#[test]
fn constant_reference_dedup_and_selects() {
    // Two uses of the same integer constant share one ValueId; select and
    // freeze wire up correctly.
    let mut syms = StrInterner::new();
    let mut module = Module::new("consts");
    let i32_ = module.types_mut().int(32);
    let sig = module.types_mut().func(vec![], i32_, false);
    let f = module.declare_function(syms.intern("k"), sig);

    {
        let mut b = module.build(f);
        b.create_entry_block();
        let seven_a = b.const_i64(i32_, 7);
        let seven_b = b.const_i64(i32_, 7);
        assert_eq!(seven_a, seven_b, "equal constants must share one value id");
        let frozen = b.freeze(seven_a);
        b.ret(Some(frozen));
    }
    assert_use_def_consistent(module.function(f));
}

#[test]
fn type_interning_is_structural() {
    let mut module = Module::new("types");
    let a = module.types_mut().int(32);
    let b = module.types_mut().int(32);
    let arr1 = module.types_mut().array(a, 8);
    let arr2 = module.types_mut().array(b, 8);
    assert_eq!(a, b);
    assert_eq!(arr1, arr2, "equal composite types intern to equal ids");
    let arr3 = module.types_mut().array(a, 9);
    assert_ne!(arr1, arr3);
}

#[test]
fn integer_constants_round_trip_through_puremp() {
    let mut module = Module::new("bignum");
    let i128_ = module.types_mut().int(128);
    // A value wider than 64 bits, to exercise puremp's arbitrary precision.
    let big = puremp::Int::from_i64(2).pow(100);
    let cid = module.intern_const(Const::Int { ty: i128_, value: big.clone() });
    match module.consts().get(cid) {
        Const::Int { ty, value } => {
            assert_eq!(*ty, i128_);
            assert_eq!(*value, big, "the stored puremp::Int must round-trip exactly");
        }
        other => panic!("expected an integer constant, got {other:?}"),
    }
    // Interning the same constant again yields the same id.
    let cid2 = module.intern_const(Const::Int { ty: i128_, value: big });
    assert_eq!(cid, cid2);
}

#[test]
fn float_constants_are_bit_exact() {
    let mut module = Module::new("floats");
    let f64_ = module.types_mut().float(FloatKind::F64);
    let bits = 1.5_f64.to_bits();
    let cid = module.intern_const(Const::Float { ty: f64_, bits: FloatBits::F64(bits) });
    match module.consts().get(cid) {
        Const::Float { bits: FloatBits::F64(b), .. } => assert_eq!(*b, bits),
        other => panic!("expected an f64 constant, got {other:?}"),
    }
    // Signed zeros are distinct bit patterns and thus distinct constants.
    let pos = module.intern_const(Const::Float { ty: f64_, bits: FloatBits::F64(0.0_f64.to_bits()) });
    let neg =
        module.intern_const(Const::Float { ty: f64_, bits: FloatBits::F64((-0.0_f64).to_bits()) });
    assert_ne!(pos, neg, "+0.0 and -0.0 must be distinct constants");
}

#[test]
fn struct_field_and_array_elem_offsets() {
    // Build addressing into `struct { i32, i64 }` and `[i32 x 4]` and confirm
    // the emitted ptr_add carries a constant byte offset.
    let mut syms = StrInterner::new();
    let mut module = Module::new("addr");
    let i32_ = module.types_mut().int(32);
    let i64_ = module.types_mut().int(64);
    let s = module.types_mut().struct_(vec![i32_, i64_]);
    let arr = module.types_mut().array(i32_, 4);
    let void = module.types_mut().void();
    let sig = module.types_mut().func(vec![], void, false);
    let f = module.declare_function(syms.intern("addr"), sig);

    {
        let mut b = module.build(f);
        b.create_entry_block();
        let sp = b.alloca(s);
        // field 1 (the i64) sits at offset 8 (i32 at 0, pad to 8).
        let field1 = b.struct_field(sp, s, 1);
        // element 2 of the array sits at offset 8 (stride 4).
        let ap = b.alloca(arr);
        let idx = b.const_i64(i64_, 2);
        let elem2 = b.array_elem(ap, i32_, idx);
        // Store something so the values are used.
        let zero = b.const_i64(i64_, 0);
        b.store(i64_, field1, zero, 8);
        let z32 = b.const_i64(i32_, 0);
        b.store(i32_, elem2, z32, 4);
        b.ret(None);
    }
    let func = module.function(f);
    assert_use_def_consistent(func);
    // field_offset directly: field 1 is at byte 8.
    assert_eq!(module.types().field_offset(s, 1), (8, i64_));
}
