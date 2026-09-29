//! Vector IR plumbing tests (`docs/ir-design.md` §6e): the `<N x T>` text
//! syntax and the vector ops round-trip through the printer/parser and the
//! `.lfb` binary form, the builder helpers type their results, and the
//! structural verifier rejects ill-formed vector code.

use crate::ir::inst::{Flags, InstKind, ReduceOp};
use crate::ir::text::{parse_module, print_module};
use crate::ir::types::Type;
use crate::ir::value::Const;
use crate::ir::{FuncId, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

/// Every vector construct in one module.
const ALL_OPS: &str = r#"
module "vec"
global @g : <4 x i32> = <4 x i32> (i32 1, i32 2, i32 3, i32 4)

func @f(<4 x i32>, <4 x f32>, i32) -> <4 x i32> {
entry ^0(%a: <4 x i32>, %x: <4 x f32>, %s: i32):
  %c = add nsw %a, <4 x i32> (i32 1, i32 poison, i32 -3, i32 2147483647) : <4 x i32>
  %m = icmp slt %c, %a : <4 x i1>
  %y = fadd nnan %x, %x : <4 x f32>
  %fm = fcmp olt %y, %x : <4 x i1>
  %both = and %m, %fm : <4 x i1>
  %sel = select %both, %a, %c : <4 x i32>
  %e = extractelement %sel, 3 : i32
  %i = insertelement %sel, %s, 0 : <4 x i32>
  %sh = shufflevector %i, %a, [7, 0, 5, 2] : <4 x i32>
  %lo = shufflevector %sh, %sh, [0, 1] : <2 x i32>
  %sp = splat %e : <4 x i32>
  %r = reduce umax %sp : i32
  %rf = reduce fadd reassoc %y : f32
  %w = sext %m : <4 x i32>
  %z = zext %lo : <2 x i64>
  %bc = bitcast %z : <8 x i16>
  %bc2 = bitcast %bc : i128
  %fz = freeze %sh : <4 x i32>
  %cv = fptosi %y : <4 x i32>
  %ld = load @g align 16 : <4 x i32>
  store %fz, @g align 16 : <4 x i32>
  %sum = add %ld, %w : <4 x i32>
  %t = select i1 1, %sum, %sp : <4 x i32>
  ret %t
}
"#;

fn parse(src: &str, syms: &mut StrInterner) -> Module {
    parse_module(src, FileId::new(0), syms).unwrap_or_else(|e| panic!("parse: {e:?}"))
}

#[test]
fn vector_text_round_trips_and_verifies() {
    let mut syms = StrInterner::new();
    let m = parse(ALL_OPS, &mut syms);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let printed = print_module(&m, &syms);
    let m2 = parse(&printed, &mut syms);
    assert_eq!(print_module(&m2, &syms), printed, "print(parse(print(m))) == print(m)");
    // A few spot checks of the canonical spelling.
    assert!(printed.contains("<4 x i32> (i32 1, i32 poison, i32 -3, i32 2147483647)"), "{printed}");
    assert!(printed.contains("shufflevector %"), "{printed}");
    assert!(printed.contains(", [7, 0, 5, 2] : <4 x i32>"), "{printed}");
    assert!(printed.contains("reduce fadd reassoc %"), "{printed}");
    assert!(printed.contains("icmp slt %0, %1 : <4 x i1>") || printed.contains(": <4 x i1>"));
}

#[test]
fn vector_binary_round_trips() {
    let mut syms = StrInterner::new();
    let m = parse(ALL_OPS, &mut syms);
    let bytes = crate::ir::binary::encode(&m, &syms);
    let mut syms2 = StrInterner::new();
    let back = crate::ir::binary::decode(&bytes, &mut syms2).expect("decode");
    assert_eq!(print_module(&back, &syms2), print_module(&m, &syms));
    assert_eq!(crate::ir::binary::encode(&back, &syms2), bytes, "re-encoding is byte-identical");
}

#[test]
fn builder_types_vector_results() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("b");
    let i32t = m.types_mut().int(32);
    let v4 = m.types_mut().vector(i32t, 4);
    let sig = m.types_mut().func(vec![v4], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let e = b.create_entry_block();
        let a = b.param(e, 0);
        let cmp = b.icmp(crate::ir::IntPred::Eq, a, a);
        let lo = b.shuffle_vector(a, a, &[0, 1]);
        let s = b.extract_element(lo, 1);
        let sp = b.splat(s, 8);
        let r = b.reduce(ReduceOp::Add, sp, Flags::NONE);
        let _ = b.insert_element(a, r, 2);
        assert!(matches!(b.types().get(b.value_type(cmp)), Type::Vector(_, 4)));
        assert_eq!(b.types().vector_parts(b.value_type(lo)).map(|p| p.1), Some(2));
        assert_eq!(b.types().vector_parts(b.value_type(sp)).map(|p| p.1), Some(8));
        assert_eq!(b.value_type(r), i32t);
        let lanes: Vec<_> = (0..4)
            .map(|k| b.intern_const(Const::Int { ty: i32t, value: puremp::Int::from_i64(k) }))
            .collect();
        let c = b.const_vector(v4, lanes);
        let x = b.extract_element(c, 3);
        let y = b.add(x, r, Flags::NONE);
        b.ret(Some(y));
    }
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let f0 = m.function(FuncId::from_index(0));
    assert!(f0.blocks().any(|(_, blk)| blk.insts().iter().any(|&i| matches!(f0.inst(i).kind, InstKind::Splat))));
}

/// Parse `body` as the body of `func @f(<4 x i32>, i32) -> i32` and return the
/// verifier's messages (empty if it verifies).
fn verify_errors(body: &str) -> Vec<String> {
    let src = format!(
        "module \"t\"\nglobal @g : <4 x i32> = <4 x i32> poison\nfunc @f(<4 x i32>, i32) -> i32 {{\nentry ^0(%a: <4 x i32>, %s: i32):\n{body}\n}}\n"
    );
    let mut syms = StrInterner::new();
    let m = parse(&src, &mut syms);
    match crate::verify::verify_module(&m) {
        Ok(()) => Vec::new(),
        Err(ds) => ds.iter().map(|d| format!("{d:?}")).collect(),
    }
}

#[track_caller]
fn rejects(body: &str, needle: &str) {
    let errs = verify_errors(body);
    assert!(
        errs.iter().any(|e| e.contains(needle)),
        "expected an error containing {needle:?} for:\n{body}\ngot: {errs:#?}"
    );
}

#[test]
fn verifier_accepts_the_baseline() {
    assert!(verify_errors("  %e = extractelement %a, 3 : i32\n  ret %e").is_empty());
}

#[test]
fn verifier_rejects_ill_formed_vector_code() {
    rejects("  %e = extractelement %a, 4 : i32\n  ret %e", "lane 4 is out of range");
    rejects("  %v = insertelement %a, %s, 9 : <4 x i32>\n  ret %s", "lane 9 is out of range");
    rejects(
        "  %v = shufflevector %a, %a, [0, 8] : <2 x i32>\n  ret %s",
        "mask index 8 is out of range",
    );
    rejects(
        "  %v = shufflevector %a, %a, [0, 1] : <4 x i32>\n  ret %s",
        "shufflevector result must be <2 x i32>",
    );
    rejects("  %v = splat %s : <4 x i64>\n  ret %s", "splat operand vs. element type");
    rejects("  %v = splat %s : i32\n  ret %s", "splat result must be a vector");
    rejects("  %r = reduce fadd %a : i32\n  ret %r", "must be floating-point");
    rejects(
        "  %c = icmp eq %a, %a : <4 x i1>\n  %v = select %c, %s, %s : i32\n  ret %v",
        "select condition must be i1",
    );
    rejects(
        "  %c = icmp eq %a, %a : <4 x i1>\n  %l = shufflevector %c, %c, [0, 1] : <2 x i1>\n  %v = select %l, %a, %a : <4 x i32>\n  ret %s",
        "select condition must be i1",
    );
    rejects("  %v = load volatile @g align 16 : <4 x i32>\n  ret %s", "vector load cannot be volatile");
    rejects("  store volatile %a, @g align 16 : <4 x i32>\n  ret %s", "vector store cannot be volatile");
    rejects("  %b = bitcast %a : i64\n  ret %s", "is not a valid conversion");
    rejects("  %b = zext %a : <2 x i64>\n  ret %s", "is not a valid conversion");
    rejects("  %b = trunc %a : i16\n  ret %s", "is not a valid conversion");
    rejects("  %b = fadd %a, %a : <4 x i32>\n  ret %s", "must be floating-point");
    rejects("  %x = add %a, <4 x i7> poison : <4 x i32>\n  ret %s", "invalid vector type <4 x i7>");
    rejects("  %x = freeze <0 x i32> poison : <0 x i32>\n  ret %s", "invalid vector type <0 x i32>");
    rejects("  %x = freeze <2 x ptr> poison : <2 x ptr>\n  ret %s", "invalid vector type <2 x ptr>");
    rejects("  %x = extractelement %s, 0 : i32\n  ret %x", "extractelement operand must be a vector");
}

#[test]
fn vector_constant_parse_errors() {
    let mut syms = StrInterner::new();
    let bad_count = "module \"t\"\nfunc @f() -> <2 x i32> {\nentry ^0:\n  ret <2 x i32> (i32 1)\n}\n";
    let e = parse_module(bad_count, FileId::new(0), &mut syms).expect_err("lane count");
    assert!(format!("{e:?}").contains("has 1 lane(s) but its type has 2"), "{e:?}");
    let bad_lane = "module \"t\"\nfunc @f() -> <2 x i32> {\nentry ^0:\n  ret <2 x i32> (i32 1, i64 2)\n}\n";
    let e = parse_module(bad_lane, FileId::new(0), &mut syms).expect_err("lane type");
    assert!(format!("{e:?}").contains("wrong type"), "{e:?}");
    // Arrays stay initializer-only.
    let arr = "module \"t\"\nfunc @f() -> i32 {\nentry ^0:\n  %x = extractelement [2 x i32] (i32 1, i32 2), 0 : i32\n  ret %x\n}\n";
    assert!(parse_module(arr, FileId::new(0), &mut syms).is_err());
}
