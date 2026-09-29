//! Tests for wide-integer legalization ([`super`]).
//!
//! Each test builds a function on `i64` (or `i128`) values, legalizes it for a
//! part width of 32, 16 and 8 bits, checks that the verifier accepts the result
//! and that no wide arithmetic is left ([`illegal_int_ops`]), then runs it on
//! the virtual target's MIR interpreter and compares every result with the
//! reference evaluator ([`crate::ir::eval`]) applied to the original
//! operation. The mul/div/rem libcalls are defined in the module as plain IR
//! (a native wide op), which the pass leaves untouched.

use super::{LegalizeError, LegalizeOptions, illegal_int_ops, legalize_ints, libgcc_libcall};
use crate::codegen::interp;
use crate::codegen::vtarget::VirtualTarget;
use crate::ir::inst::{BinOp, CastOp, Flags, InstKind, IntPred};
use crate::ir::{EvalOutcome, FuncId, Module, SemValue, TypeId};
use crate::support::StrInterner;

use puremp::Int;

/// Interesting 64-bit operands: edges of every part width, plus a few
/// pseudo-random values from a fixed LCG (deterministic).
fn samples() -> Vec<u64> {
    let mut v = vec![
        0,
        1,
        2,
        0x7f,
        0x80,
        0xff,
        0x100,
        0x7fff,
        0x8000,
        0xffff,
        0x1_0000,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
        0x1_0000_0000,
        0x7fff_ffff_ffff_ffff,
        0x8000_0000_0000_0000,
        u64::MAX,
        u64::MAX - 1,
        0x0123_4567_89ab_cdef,
    ];
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..6 {
        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        v.push(x);
    }
    v
}

/// Shift amounts to try (all in range; out-of-range shifts are poison).
const SHIFTS: [u64; 12] = [0, 1, 3, 7, 8, 9, 15, 16, 31, 32, 33, 63];

/// A module with `f(a: T, b: T) -> R = body(a, b)` as function 0, plus native
/// definitions of the libgcc helpers for `T`'s width.
fn module_with(
    bits: u32,
    ret_bits: u32,
    body: impl FnOnce(&mut crate::ir::builder::FunctionBuilder<'_>, crate::ir::ValueId, crate::ir::ValueId, TypeId) -> crate::ir::ValueId,
) -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("wide");
    let t = m.types_mut().int(bits);
    let rt = m.types_mut().int(ret_bits);
    let sig = m.types_mut().func(vec![t, t], rt, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let e = b.create_entry_block();
        let (x, y) = (b.param(e, 0), b.param(e, 1));
        let r = body(&mut b, x, y, t);
        b.ret(Some(r));
    }
    // The runtime: each helper is the native operation (never legalized).
    let helper_sig = m.types_mut().func(vec![t, t], t, false);
    for op in [BinOp::Mul, BinOp::UDiv, BinOp::SDiv, BinOp::URem, BinOp::SRem] {
        let h = m.declare_function(syms.intern(&libgcc_libcall(op, bits)), helper_sig);
        let mut b = m.build(h);
        let e = b.create_entry_block();
        let (x, y) = (b.param(e, 0), b.param(e, 1));
        let r = b.bin(op, x, y, Flags::NONE);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// Legalize `m` for `part_bits`, verify it, check no wide arithmetic is left
/// in `f`, and lower every function over the virtual target.
fn legalize_and_lower(
    mut m: Module,
    mut syms: StrInterner,
    f: FuncId,
    part_bits: u32,
) -> (VirtualTarget, Vec<crate::codegen::MachineFunction>) {
    let report = legalize_ints(&mut m, &mut syms, &LegalizeOptions::new(part_bits)).expect("legalizes");
    assert!(report.functions.contains(&f));
    if let Err(d) = crate::verify::verify_module(&m) {
        panic!("legalized module does not verify: {d:?}\n{}", crate::ir::text::print_module(&m, &syms));
    }
    let bad = illegal_int_ops(&m, f, part_bits);
    assert!(bad.is_empty(), "wide ops left at W={part_bits}: {bad:?}\n{}", crate::ir::text::print_module(&m, &syms));
    let target = VirtualTarget::new();
    let funcs = (0..m.function_count()).map(|i| target.select(&m, FuncId::from_index(i))).collect();
    (target, funcs)
}

/// The reference result of `kind` on `bits`-bit operands (unsigned patterns),
/// or `None` when it is poison or UB (nothing to compare).
fn reference(kind: &InstKind, bits: u32, ret_bits: u32, a: u64, b: u64) -> Option<Int> {
    let mut tc = crate::ir::TypeContext::new();
    let rt = tc.int(ret_bits);
    let ops = [SemValue::int(bits, Int::from_u64(a)), SemValue::int(bits, Int::from_u64(b))];
    match crate::ir::eval(&tc, rt, kind, &Flags::NONE, &ops) {
        EvalOutcome::Value(SemValue::Int { bits, .. }) => Some(bits),
        _ => None,
    }
}

/// Check `f(a, b) = kind(a, b)` over the samples at every part width.
fn check_binary(kind: InstKind, ret_bits: u32, pairs: &[(u64, u64)]) {
    let k = kind.clone();
    for part_bits in [32, 16, 8] {
        let k2 = k.clone();
        let (m, syms, f) = module_with(64, ret_bits, move |b, x, y, _| {
            let rt = b.types_mut().int(ret_bits);
            b.append_inst(k2, vec![x, y], Flags::NONE, Some(rt)).expect("a result")
        });
        let (target, funcs) = legalize_and_lower(m, syms, f, part_bits);
        for &(a, b) in pairs {
            let Some(want) = reference(&kind, 64, ret_bits, a, b) else { continue };
            let got = interp::run(&target, &funcs, f.index(), &[Int::from_u64(a), Int::from_u64(b)])
                .unwrap_or_else(|e| panic!("{kind:?}({a:#x}, {b:#x}) at W={part_bits}: {e}"))
                .expect("a return value")
                .mod_2k(ret_bits);
            assert_eq!(got, want, "{kind:?}({a:#x}, {b:#x}) at W={part_bits}");
        }
    }
}

fn all_pairs() -> Vec<(u64, u64)> {
    let s = samples();
    s.iter().flat_map(|&a| s.iter().map(move |&b| (a, b))).collect()
}

#[test]
fn add_sub_and_bitwise_match_the_reference() {
    let pairs = all_pairs();
    for op in [BinOp::Add, BinOp::Sub, BinOp::And, BinOp::Or, BinOp::Xor] {
        check_binary(InstKind::Bin(op), 64, &pairs);
    }
}

#[test]
fn variable_shifts_match_the_reference() {
    let pairs: Vec<(u64, u64)> = samples().iter().flat_map(|&a| SHIFTS.iter().map(move |&s| (a, s))).collect();
    for op in [BinOp::Shl, BinOp::LShr, BinOp::AShr] {
        check_binary(InstKind::Bin(op), 64, &pairs);
    }
}

#[test]
fn constant_shifts_match_the_reference() {
    for op in [BinOp::Shl, BinOp::LShr, BinOp::AShr] {
        for s in SHIFTS {
            for part_bits in [32, 16, 8] {
                let (m, syms, f) = module_with(64, 64, move |b, x, _, t| {
                    let c = b.const_int(t, Int::from_u64(s));
                    b.bin(op, x, c, Flags::NONE)
                });
                let (target, funcs) = legalize_and_lower(m, syms, f, part_bits);
                for a in samples() {
                    let want = reference(&InstKind::Bin(op), 64, 64, a, s).expect("in range");
                    let got = interp::run(&target, &funcs, 0, &[Int::from_u64(a), Int::ZERO]).unwrap().unwrap();
                    assert_eq!(got.mod_2k(64), want, "{op:?} {a:#x} by {s} at W={part_bits}");
                }
            }
        }
    }
}

#[test]
fn compares_match_the_reference() {
    let pairs = all_pairs();
    for pred in [
        IntPred::Eq,
        IntPred::Ne,
        IntPred::Ult,
        IntPred::Ule,
        IntPred::Ugt,
        IntPred::Uge,
        IntPred::Slt,
        IntPred::Sle,
        IntPred::Sgt,
        IntPred::Sge,
    ] {
        check_binary(InstKind::ICmp(pred), 1, &pairs);
    }
}

#[test]
fn mul_div_rem_go_through_libcalls() {
    let pairs = all_pairs();
    for op in [BinOp::Mul, BinOp::UDiv, BinOp::SDiv, BinOp::URem, BinOp::SRem] {
        // Division by zero and INT_MIN / -1 are UB: `reference` skips them.
        check_binary(InstKind::Bin(op), 64, &pairs);
    }
    // The helper is declared when the module lacks it.
    let mut syms = StrInterner::new();
    let mut m = Module::new("m");
    let t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![t, t], t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let e = b.create_entry_block();
        let (x, y) = (b.param(e, 0), b.param(e, 1));
        let r = b.mul(x, y, Flags::nsw());
        b.ret(Some(r));
    }
    let report = legalize_ints(&mut m, &mut syms, &LegalizeOptions::new(32)).unwrap();
    assert_eq!(report.libcalls.len(), 1);
    assert_eq!(report.libcalls[0].0, "__muldi3");
    assert!(m.function(report.libcalls[0].1).is_declaration());
    crate::verify::verify_module(&m).expect("verifies");
}

#[test]
fn casts_select_and_freeze_match() {
    // f(a, b) = select(a <u b, sext(trunc a to i16), zext(trunc b to i8)) + freeze(a)
    for part_bits in [32, 16, 8] {
        let (m, syms, f) = module_with(64, 64, |b, x, y, t| {
            let i16t = b.types_mut().int(16);
            let i8t = b.types_mut().int(8);
            let xs = b.cast(CastOp::Trunc, x, i16t);
            let xe = b.cast(CastOp::SExt, xs, t);
            let yb = b.cast(CastOp::Trunc, y, i8t);
            let ye = b.cast(CastOp::ZExt, yb, t);
            let c = b.icmp(IntPred::Ult, x, y);
            let s = b.select(c, xe, ye);
            let fr = b.freeze(x);
            b.add(s, fr, Flags::NONE)
        });
        let (target, funcs) = legalize_and_lower(m, syms, f, part_bits);
        for (a, b) in all_pairs() {
            let xe = a as u16 as i16 as i64 as u64;
            let ye = u64::from(b as u8);
            let want = (if a < b { xe } else { ye }).wrapping_add(a);
            let got = interp::run(&target, &funcs, 0, &[Int::from_u64(a), Int::from_u64(b)]).unwrap().unwrap();
            assert_eq!(got.mod_2k(64), Int::from_u64(want), "({a:#x}, {b:#x}) at W={part_bits}");
        }
    }
}

/// A loop carrying an `i64` accumulator in a block parameter and memory
/// traffic through an `i64` stack slot: `f(a, n)` sums `a << i` for `i` in
/// `0..(n & 15)`, storing and reloading the sum each iteration.
#[test]
fn loops_and_memory_split_across_blocks() {
    for part_bits in [32, 16, 8] {
        let mut syms = StrInterner::new();
        let mut m = Module::new("loop");
        let t = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![t, t], t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let head = b.create_block(&[t, t]); // (i, acc)
            let body = b.create_block(&[t, t]);
            let exit = b.create_block(&[t]);
            let (a, n) = (b.param(e, 0), b.param(e, 1));
            let slot = b.alloca(t);
            let fifteen = b.const_i64(t, 15);
            let lim = b.bin(BinOp::And, n, fifteen, Flags::NONE);
            let zero = b.const_i64(t, 0);
            b.store(t, slot, zero, 8);
            b.br(head, &[zero, zero]);
            b.switch_to(head);
            let (i, acc) = (b.param(head, 0), b.param(head, 1));
            let c = b.icmp(IntPred::Slt, i, lim);
            b.cond_br(c, body, &[i, acc], exit, &[acc]);
            b.switch_to(body);
            let (i2, acc2) = (b.param(body, 0), b.param(body, 1));
            let sh = b.bin(BinOp::Shl, a, i2, Flags::NONE);
            let sum = b.add(acc2, sh, Flags::NONE);
            b.store(t, slot, sum, 8);
            let back = b.load(t, slot, 8);
            let one = b.const_i64(t, 1);
            let inc = b.add(i2, one, Flags::nsw());
            b.br(head, &[inc, back]);
            b.switch_to(exit);
            let r = b.param(exit, 0);
            b.ret(Some(r));
        }
        let (target, funcs) = legalize_and_lower(m, syms, f, part_bits);
        for a in samples() {
            for n in [0u64, 1, 5, 15, 0xffff_ffff_ffff_fff3] {
                let want = (0..(n & 15)).fold(0u64, |acc, i| acc.wrapping_add(a << i));
                let got = interp::run(&target, &funcs, 0, &[Int::from_u64(a), Int::from_u64(n)]).unwrap().unwrap();
                assert_eq!(got.mod_2k(64), Int::from_u64(want), "f({a:#x}, {n}) at W={part_bits}");
            }
        }
    }
}

/// `i128` splits into four `i32` parts (and eight `i16`s); an `i64` target
/// legalizes it too.
#[test]
fn i128_splits_at_every_width() {
    let big = |x: u64, y: u64| (u128::from(x) << 64) | u128::from(y);
    for part_bits in [64, 32, 16] {
        let (m, syms, f) = module_with(128, 128, |b, x, y, _| {
            let s = b.add(x, y, Flags::NONE);
            b.bin(BinOp::Xor, s, x, Flags::NONE)
        });
        let (target, funcs) = legalize_and_lower(m, syms, f, part_bits);
        for (a, b) in [(big(1, u64::MAX), big(0, 1)), (big(u64::MAX, u64::MAX), big(0, 1)), (big(7, 9), big(3, 5))] {
            let want = a.wrapping_add(b) ^ a;
            let got = interp::run(&target, &funcs, 0, &[Int::from_u128(a), Int::from_u128(b)]).unwrap().unwrap();
            assert_eq!(got.mod_2k(128), Int::from_u128(want), "at W={part_bits}");
        }
    }
}

#[test]
fn nothing_to_do_and_errors() {
    // No integer above the part width: nothing is rewritten.
    let (mut m, mut syms, _) = module_with(32, 32, |b, x, y, _| b.add(x, y, Flags::NONE));
    let report = legalize_ints(&mut m, &mut syms, &LegalizeOptions::new(32)).unwrap();
    assert!(report.functions.is_empty() && report.libcalls.is_empty());
    // The options follow a layout's widest native integer.
    assert_eq!(LegalizeOptions::for_layout(&crate::ir::DataLayout::ilp32()).part_bits, 32);
    // Bad part width, and a wide width that is not a multiple of it.
    assert_eq!(legalize_ints(&mut m, &mut syms, &LegalizeOptions::new(12)).unwrap_err(), LegalizeError::BadPartWidth(12));
    let (mut m, mut syms, _) = module_with(40, 40, |b, x, y, _| b.add(x, y, Flags::NONE));
    assert_eq!(legalize_ints(&mut m, &mut syms, &LegalizeOptions::new(32)).unwrap_err(), LegalizeError::UnsupportedWidth(40));
    assert_eq!(libgcc_libcall(BinOp::SDiv, 64), "__divdi3");
    assert_eq!(libgcc_libcall(BinOp::URem, 128), "__umodti3");
    assert_eq!(libgcc_libcall(BinOp::Mul, 16), "__lf_mul_i16");
}

/// A rebuilt function keeps its linkage, visibility, secrecy and declaration
/// line (an `internal` helper must not become a global symbol after
/// legalization).
#[test]
fn rebuilt_functions_keep_their_attributes() {
    let src = "module \"a\"\n\
        func internal hidden @f(secret i64, i64) -> secret i64 {\n\
        entry ^0(%a: i64, %b: i64):\n  %s = add %a, %b : i64\n  ret %s\n}\n";
    let mut syms = StrInterner::new();
    let mut m = crate::ir::text::parse_module(src, crate::support::diagnostics::FileId::new(0), &mut syms).unwrap();
    let f = FuncId::from_index(0);
    let mut b = m.build(f);
    b.set_decl_line(7);
    let before = m.function(f).attrs.clone();
    assert!(before.is_param_secret(0) && before.secret_ret);
    let report = legalize_ints(&mut m, &mut syms, &LegalizeOptions::new(32)).unwrap();
    assert!(report.functions.contains(&f), "the function was rebuilt");
    assert_eq!(m.function(f).attrs, before);
    assert_eq!(m.function(f).decl_line, Some(7));
    let obj = crate::target::x86_64::compile_module(&m, &syms);
    let sym = obj.symbol(obj.symbol_id("f").unwrap());
    assert_eq!(sym.binding, crate::mc::object::SymbolBinding::Local);
}
