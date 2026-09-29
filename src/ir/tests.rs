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

// ---------------------------------------------------------------------------
// Volatile accesses, atomics and fences (docs/ir-design.md §6b)
// ---------------------------------------------------------------------------

/// A module exercising every volatile / atomic form: volatile load and store,
/// `atomic_load`/`atomic_store` at every legal ordering, every `atomic_rmw`
/// operation, `cmpxchg` (with its success flag), and every legal fence, over
/// `i8`/`i16`/`i32`/`i64`/`ptr`, on a global and on a parameter pointer. Shared
/// by the text, binary and verifier tests.
pub(crate) const ATOMICS_LF: &str = r#"
module "atomics"
global @g : i32 = i32 0
global @gp : ptr = ptr null

func @f(ptr, i64) -> i64 {
entry ^0(%p: ptr, %n: i64):
  %a = load volatile @g align 4 : i32
  store volatile %a, @g align 4 : i32
  %b = atomic_load relaxed %p align 8 : i64
  %c = atomic_load acquire %p align 8 : i64
  %d = atomic_load seq_cst @gp align 8 : ptr
  atomic_store relaxed %b, %p align 8 : i64
  atomic_store release %c, %p align 8 : i64
  atomic_store seq_cst %d, @gp align 8 : ptr
  %x0 = atomic_rmw xchg seq_cst %p, %n align 8 : i64
  %x1 = atomic_rmw add relaxed %p, i64 1 align 8 : i64
  %x2 = atomic_rmw sub acquire %p, i64 2 align 8 : i64
  %x3 = atomic_rmw and release %p, i64 3 align 8 : i64
  %x4 = atomic_rmw nand acq_rel %p, i64 4 align 8 : i64
  %x5 = atomic_rmw or seq_cst %p, i64 5 align 8 : i64
  %x6 = atomic_rmw xor seq_cst %p, i64 6 align 8 : i64
  %x7 = atomic_rmw max seq_cst @g, i32 7 align 4 : i32
  %x8 = atomic_rmw min seq_cst @g, i32 8 align 4 : i32
  %x9 = atomic_rmw umax seq_cst @g, i32 9 align 4 : i32
  %x10 = atomic_rmw umin seq_cst @g, i32 10 align 4 : i32
  %x11 = atomic_rmw xchg seq_cst @gp, %p align 8 : ptr
  %x12 = atomic_rmw add seq_cst %p, i8 1 align 1 : i8
  %x13 = atomic_rmw add seq_cst %p, i16 1 align 2 : i16
  %old = cmpxchg seq_cst relaxed %p, %n, i64 42 align 8 : i64
  %ok = icmp eq %old, %n : i1
  %pold = cmpxchg acq_rel acquire @gp, ptr null, %p align 16 : ptr
  fence acquire
  fence release
  fence acq_rel
  fence seq_cst
  %r = select %ok, %old, %x1 : i64
  ret %r
}
"#;

/// Parse [`ATOMICS_LF`] (panicking on a parse error).
pub(crate) fn atomics_module(syms: &mut StrInterner) -> Module {
    crate::ir::text::parse_module(ATOMICS_LF, crate::support::diagnostics::FileId::new(0), syms)
        .unwrap_or_else(|e| panic!("parse ATOMICS_LF: {e:?}"))
}

/// A module exercising every secrecy spelling (`docs/ir-design.md` §6d):
/// secret globals, secret parameters and returns on a definition and on a
/// declaration, secret (and volatile secret) loads and stores, and
/// `declassify`.
pub(crate) const SECRET_LF: &str = r#"
module "secret"

global secret @key : [4 x i64] = [4 x i64] (i64 1, i64 2, i64 3, i64 4)

global internal constant secret @k2 : i64 = i64 7

func @ext(secret i64, ptr) -> secret i64

func @f(secret i64, i64, ptr) -> secret i64 {
entry ^0(%s: i64, %p: i64, %q: ptr):
  %a = load secret %q align 8 : i64
  %b = load volatile secret %q align 8 : i64
  store secret %a, %q align 8 : i64
  store volatile secret %b, %q align 8 : i64
  %x = add %s, %a : i64
  %c = icmp eq %p, i64 0 : i1
  %y = select %c, %x, %b : i64
  %d = declassify %y : i64
  %e = add %d, i64 1 : i64
  %z = call @ext(%y, %q) : i64
  ret %z
}
"#;

/// Parse [`SECRET_LF`] (panicking on a parse error).
pub(crate) fn secret_module(syms: &mut StrInterner) -> Module {
    crate::ir::text::parse_module(SECRET_LF, crate::support::diagnostics::FileId::new(0), syms)
        .unwrap_or_else(|e| panic!("parse SECRET_LF: {e:?}"))
}

#[test]
fn secrecy_attributes_parse_and_round_trip() {
    use crate::ir::{FuncId, GlobalId};
    let mut syms = StrInterner::new();
    let m = secret_module(&mut syms);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    assert!(m.has_secrets());
    assert!(m.global_attrs(GlobalId::from_index(0)).secret);
    let k2 = m.global_attrs(GlobalId::from_index(1));
    assert!(k2.secret && k2.constant && k2.linkage == crate::ir::Linkage::Internal);
    let ext = m.func_attrs(FuncId::from_index(0));
    assert!(ext.is_param_secret(0) && !ext.is_param_secret(1) && ext.secret_ret);
    let f = m.func_attrs(FuncId::from_index(1));
    assert_eq!(f.secret_params().collect::<Vec<_>>(), [0]);
    assert!(f.secret_ret);

    // Text: canonical and idempotent.
    let text = crate::ir::text::print_module(&m, &syms);
    for needle in [
        "global secret @key",
        "global internal constant secret @k2",
        "func @ext(secret i64, ptr) -> secret i64\n",
        "func @f(secret i64, i64, ptr) -> secret i64 {",
        "= load secret %2 align 8 : i64",
        "= load volatile secret %2 align 8 : i64",
        "store secret %3, %2 align 8 : i64",
        "store volatile secret %4, %2 align 8 : i64",
        "= declassify %7 : i64",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
    let file = crate::support::diagnostics::FileId::new(0);
    let again = crate::ir::text::parse_module(&text, file, &mut syms).expect("reparse");
    assert_eq!(crate::ir::text::print_module(&again, &syms), text);

    // Binary: stable bytes, and the decoded module is the original.
    let bytes = crate::ir::binary::encode(&m, &syms);
    let mut back = StrInterner::new();
    let m2 = crate::ir::binary::decode(&bytes, &mut back).expect("decode");
    assert_eq!(crate::ir::binary::encode(&m2, &back), bytes);
    assert_eq!(crate::ir::text::print_module(&m2, &back), text);

    // A module without secrets reports none.
    let plain = atomics_module(&mut syms);
    assert!(!plain.has_secrets());
}

#[test]
fn secrecy_combines_with_linkage_visibility_and_address_spaces() {
    use crate::ir::{FuncId, GlobalId, Linkage, Visibility};
    let src = r#"module "combo"

global internal hidden constant secret addrspace(1) @a : i64 = i64 1

global weak protected secret @b : i64 = i64 2

global secret addrspace(2) @c : i64

global hidden detached secret @d : i64 = i64 4

func weak protected @f(secret i64, ptr, secret i64) -> secret i64

func internal hidden @g(i64) -> secret i64 {
entry ^0(%x: i64):
  ret %x
}

func protected @h(secret i64) -> i64

func internal @k() -> void {
entry ^0:
  ret
}
"#;
    let mut syms = StrInterner::new();
    let file = crate::support::diagnostics::FileId::new(0);
    let m = crate::ir::text::parse_module(src, file, &mut syms).expect("parse");
    let a = m.global_attrs(GlobalId::from_index(0));
    assert!(a.secret && a.constant && a.linkage == Linkage::Internal);
    assert_eq!(a.visibility, Visibility::Hidden);
    assert_eq!(m.global_addr_space(GlobalId::from_index(0)), 1);
    assert!(m.global_attrs(GlobalId::from_index(1)).secret);
    assert_eq!(m.global_addr_space(GlobalId::from_index(2)), 2);
    assert!(m.global_attrs(GlobalId::from_index(3)).detached);
    let f = m.func_attrs(FuncId::from_index(0));
    assert_eq!((f.linkage, f.visibility), (Linkage::Weak, Visibility::Protected));
    assert_eq!(f.secret_params().collect::<Vec<_>>(), [0, 2]);
    assert!(f.secret_ret);
    let g = m.func_attrs(FuncId::from_index(1));
    assert_eq!((g.linkage, g.visibility, g.secret_ret), (Linkage::Internal, Visibility::Hidden, true));
    assert!(!m.func_attrs(FuncId::from_index(3)).has_secrets());

    // The printer writes every combination back in the canonical order.
    let text = crate::ir::text::print_module(&m, &syms);
    for needle in [
        "global internal hidden constant secret addrspace(1) @a : i64 = i64 1",
        "global weak protected secret @b",
        "global secret addrspace(2) @c : i64\n",
        "global hidden detached secret @d",
        "func weak protected @f(secret i64, ptr, secret i64) -> secret i64\n",
        "func internal hidden @g(i64) -> secret i64 {",
        "func protected @h(secret i64) -> i64\n",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
    let again = crate::ir::text::parse_module(&text, file, &mut syms).expect("reparse");
    assert_eq!(crate::ir::text::print_module(&again, &syms), text);

    // And through the binary form.
    let bytes = crate::ir::binary::encode(&m, &syms);
    let mut back = StrInterner::new();
    let m2 = crate::ir::binary::decode(&bytes, &mut back).expect("decode");
    assert_eq!(crate::ir::text::print_module(&m2, &back), text);

    // A functional rebuild keeps every attribute (linkage and secrecy alike).
    let mut m = m;
    let (fresh, ()) = m.map_function(FuncId::from_index(1), |old, b| {
        let e = b.create_entry_block();
        let _ = old;
        let x = b.param(e, 0);
        b.ret(Some(x));
    });
    assert_eq!(fresh.attrs, *m.func_attrs(FuncId::from_index(1)));
}

#[test]
fn func_attrs_normalize() {
    let mut a = crate::ir::FuncAttrs::default();
    assert!(!a.has_secrets());
    a.set_param_secret(3, true);
    assert!(a.is_param_secret(3) && !a.is_param_secret(2) && !a.is_param_secret(9));
    a.set_param_secret(3, false);
    // Clearing the last secret parameter restores the default value.
    assert_eq!(a, crate::ir::FuncAttrs::default());
    a.set_param_secret(7, false);
    assert_eq!(a, crate::ir::FuncAttrs::default());
}

#[test]
fn secret_merge_keeps_the_union_of_secrecy() {
    use crate::ir::FuncId;
    let mut syms = StrInterner::new();
    let file = crate::support::diagnostics::FileId::new(0);
    // A declares `g` with a secret parameter; B defines it with a secret return.
    let a = crate::ir::text::parse_module(
        "module \"a\"\nfunc @g(secret i64, i64) -> i64\n",
        file,
        &mut syms,
    )
    .expect("parse a");
    let b = crate::ir::text::parse_module(
        "module \"b\"\nfunc @g(i64, i64) -> secret i64 {\nentry ^0(%x: i64, %y: i64):\n  ret %x\n}\n",
        file,
        &mut syms,
    )
    .expect("parse b");
    let m = crate::ir::merge_modules(vec![a, b], "ab").expect("merge");
    let g = m.func_attrs(FuncId::from_index(0));
    assert!(g.is_param_secret(0) && !g.is_param_secret(1) && g.secret_ret);
}

#[test]
fn atomics_fixture_verifies_and_carries_every_form() {
    let mut syms = StrInterner::new();
    let m = atomics_module(&mut syms);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let f = m.function(FuncId::from_index(0));
    let kinds: Vec<&InstKind> =
        f.blocks().flat_map(|(_, b)| b.insts().iter().map(|&i| &f.inst(i).kind)).collect();
    assert_eq!(kinds.iter().filter(|k| k.is_volatile()).count(), 2);
    assert_eq!(kinds.iter().filter(|k| matches!(k, InstKind::AtomicRmw { .. })).count(), 14);
    assert_eq!(kinds.iter().filter(|k| matches!(k, InstKind::CmpXchg { .. })).count(), 2);
    assert_eq!(kinds.iter().filter(|k| matches!(k, InstKind::Fence(_))).count(), 4);
    assert!(kinds.iter().all(|k| !k.is_atomic() || k.has_side_effect()), "every atomic is kept by DCE");
    assert!(kinds.iter().all(|k| !k.is_volatile() || k.has_side_effect()), "volatile is kept by DCE");
}

#[test]
fn builder_atomics_use_natural_alignment() {
    use crate::ir::inst::{AtomicOrdering, RmwOp};
    let mut syms = StrInterner::new();
    let mut m = Module::new("b");
    let i16t = m.types_mut().int(16);
    let i64t = m.types_mut().int(64);
    let ptr = m.types_mut().ptr();
    let sig = m.types_mut().func(vec![ptr, i16t], i64t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let e = b.create_entry_block();
        let p = b.param(e, 0);
        let h = b.param(e, 1);
        let v = b.load_volatile(i16t, p, 2);
        b.store_volatile(i16t, p, v, 2);
        b.atomic_store(i16t, p, h, AtomicOrdering::Release);
        let l = b.atomic_load(ptr, p, AtomicOrdering::Acquire);
        b.atomic_rmw(RmwOp::Or, p, h, AtomicOrdering::Relaxed);
        let old = b.cmpxchg(p, l, l, AtomicOrdering::SeqCst, AtomicOrdering::Acquire);
        let ok = b.cmpxchg_success(old, l);
        b.fence(AtomicOrdering::SeqCst);
        let r = b.cast(crate::ir::CastOp::ZExt, ok, i64t);
        b.ret(Some(r));
    }
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let func = m.function(f);
    let aligns: Vec<u32> = func
        .blocks()
        .flat_map(|(_, b)| b.insts().to_vec())
        .filter_map(|i| match func.inst(i).kind {
            InstKind::AtomicLoad { align, .. }
            | InstKind::AtomicStore { align, .. }
            | InstKind::AtomicRmw { align, .. }
            | InstKind::CmpXchg { align, .. } => Some(align),
            _ => None,
        })
        .collect();
    assert_eq!(aligns, vec![2, 8, 2, 8], "atomics default to natural alignment");
}

#[test]
fn rmw_apply_matches_std_atomics() {
    use crate::ir::inst::RmwOp;
    use std::sync::atomic::{AtomicI8, AtomicI32, AtomicU8, AtomicU64, Ordering::SeqCst};
    // An independent oracle: the standard library's own fetch_* operations.
    for &(old, v) in &[(0x7Fu8, 0x81u8), (0x80, 0x7F), (0xFF, 0x01), (5, 250), (0, 0)] {
        let s = |op: RmwOp| op.apply(u64::from(old), u64::from(v), 8);
        let after_i = |f: &dyn Fn(&AtomicI8)| {
            let a = AtomicI8::new(old as i8);
            f(&a);
            u64::from(a.load(SeqCst) as u8)
        };
        let after_u = |f: &dyn Fn(&AtomicU8)| {
            let a = AtomicU8::new(old);
            f(&a);
            u64::from(a.load(SeqCst))
        };
        assert_eq!(s(RmwOp::Xchg), u64::from(v));
        assert_eq!(s(RmwOp::Add), after_u(&|a| { a.fetch_add(v, SeqCst); }));
        assert_eq!(s(RmwOp::Sub), after_u(&|a| { a.fetch_sub(v, SeqCst); }));
        assert_eq!(s(RmwOp::And), after_u(&|a| { a.fetch_and(v, SeqCst); }));
        assert_eq!(s(RmwOp::Nand), after_u(&|a| { a.fetch_nand(v, SeqCst); }));
        assert_eq!(s(RmwOp::Or), after_u(&|a| { a.fetch_or(v, SeqCst); }));
        assert_eq!(s(RmwOp::Xor), after_u(&|a| { a.fetch_xor(v, SeqCst); }));
        assert_eq!(s(RmwOp::Max), after_i(&|a| { a.fetch_max(v as i8, SeqCst); }));
        assert_eq!(s(RmwOp::Min), after_i(&|a| { a.fetch_min(v as i8, SeqCst); }));
        assert_eq!(s(RmwOp::UMax), after_u(&|a| { a.fetch_max(v, SeqCst); }));
        assert_eq!(s(RmwOp::UMin), after_u(&|a| { a.fetch_min(v, SeqCst); }));
    }
    // 32- and 64-bit spot checks of the signed/unsigned split and wrapping.
    let a = AtomicI32::new(-5);
    a.fetch_max(3, SeqCst);
    assert_eq!(RmwOp::Max.apply((-5i32) as u32 as u64, 3, 32), a.load(SeqCst) as u32 as u64);
    let u = AtomicU64::new(u64::MAX);
    u.fetch_add(2, SeqCst);
    assert_eq!(RmwOp::Add.apply(u64::MAX, 2, 64), u.load(SeqCst));
    assert_eq!(RmwOp::UMin.apply((-5i32) as u32 as u64, 3, 32), 3);
}

#[test]
fn ordering_and_rmw_codes_round_trip() {
    use crate::ir::inst::{AtomicOrdering, RmwOp};
    for o in AtomicOrdering::ALL {
        assert_eq!(AtomicOrdering::from_code(u64::from(o.code())), Some(o));
        assert_eq!(AtomicOrdering::from_name(o.name()), Some(o));
    }
    for op in RmwOp::ALL {
        assert_eq!(RmwOp::from_code(u64::from(op.code())), Some(op));
        assert_eq!(RmwOp::from_name(op.name()), Some(op));
    }
    assert_eq!(AtomicOrdering::from_code(5), None);
    assert_eq!(RmwOp::from_code(11), None);
    // Which orderings each op admits.
    use AtomicOrdering::*;
    let names = |f: fn(AtomicOrdering) -> bool| -> Vec<&str> {
        AtomicOrdering::ALL.into_iter().filter(|&o| f(o)).map(AtomicOrdering::name).collect()
    };
    assert_eq!(names(AtomicOrdering::valid_for_load), ["relaxed", "acquire", "seq_cst"]);
    assert_eq!(names(AtomicOrdering::valid_for_store), ["relaxed", "release", "seq_cst"]);
    assert_eq!(names(AtomicOrdering::valid_for_fence), ["acquire", "release", "acq_rel", "seq_cst"]);
    assert!(SeqCst.is_acquire() && SeqCst.is_release() && AcqRel.is_acquire() && AcqRel.is_release());
    assert!(!Relaxed.is_acquire() && !Relaxed.is_release() && !Acquire.is_release() && !Release.is_acquire());
}

/// Under a 16-bit layout the builder's `struct_field` offsets are `i16`
/// constants at the layout's offsets, a global in address space 1 is referenced
/// as a `ptr addrspace(1)`, and `ptr_add` keeps its base's space.
#[test]
fn builder_follows_the_data_layout_and_address_spaces() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("avr");
    m.set_data_layout(DataLayout::parse("p:16:8-p1:16:8-i16:8-i32:8-n8").unwrap());
    let i8t = m.types_mut().int(8);
    let i32t = m.types_mut().int(32);
    let st = m.types_mut().struct_(vec![i8t, i32t]);
    let ptr = m.types_mut().ptr();
    let sig = m.types_mut().func(vec![ptr], ptr, false);
    let g = m.define_global(Global { name: syms.intern("tbl"), ty: i8t, init: None }, GlobalAttrs::DEFAULT);
    m.set_global_addr_space(g, 1);
    let f = m.declare_function(syms.intern("f"), sig);
    let (field, gref, moved) = {
        let mut b = m.build(f);
        let e = b.create_entry_block();
        let p = b.param(e, 0);
        let field = b.struct_field(p, st, 1);
        let gref = b.global_ref(g);
        let one = b.const_i64(i8t, 1);
        let moved = b.ptr_add(gref, one, false);
        b.ret(Some(field));
        (field, gref, moved)
    };
    let func = m.function(f);
    let ValueDef::Inst(add) = func.value(field).def else { panic!("ptr_add result") };
    let off = func.inst(add).operands()[1];
    let ValueDef::Const(c) = func.value(off).def else { panic!("constant offset") };
    let Const::Int { value, .. } = m.consts().get(c) else { panic!("integer offset") };
    assert_eq!(*value, puremp::Int::from_i64(1), "i32 is byte-aligned, so field 1 is at offset 1");
    assert_eq!(m.types().get(func.value_type(off)), &Type::Int(16), "offsets are pointer-width");
    assert_eq!(m.types().get(func.value_type(gref)), &Type::PtrIn(1));
    assert_eq!(m.types().get(func.value_type(moved)), &Type::PtrIn(1));
    assert_eq!(m.global_ref_type(g), m.types_mut().ptr_in(1));
}

/// Linking modules with different data layouts is refused; an empty module
/// adopts the first input's target and layout.
#[test]
fn merging_checks_data_layouts() {
    let mut a = Module::new("a");
    a.set_target(Some("thumbv7m".to_owned()));
    a.set_data_layout(DataLayout::ilp32());
    let mut syms = StrInterner::new();
    let i32t = a.types_mut().int(32);
    let sig = a.types_mut().func(vec![], i32t, false);
    a.declare_function(syms.intern("f"), sig);
    let merged = merge_modules([a], "lto").expect("one module merges");
    assert_eq!(merged.target(), Some("thumbv7m"));
    assert_eq!(merged.data_layout(), &DataLayout::ilp32());

    let mut b = Module::new("b");
    let i32t = b.types_mut().int(32);
    let sig = b.types_mut().func(vec![], i32t, false);
    b.declare_function(syms.intern("g"), sig);
    let mut merged = merged;
    assert_eq!(merged.link_module(b), Err(MergeError::DataLayoutMismatch));
}
