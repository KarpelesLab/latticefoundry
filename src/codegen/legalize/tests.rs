//! Tests for the vector legalizer: every program is executed by the reference
//! executor before and after legalization (for a scalar-only target, and for a
//! partial "only `<4 x i32>` with `add`" target that forces the mixed
//! extract/insert paths), and the results must agree lane by lane.

use super::{ScalarOnly, VectorLegality, legalize_vectors, uses_vectors};
use crate::ir::inst::{BinOp, InstData, InstKind};
use crate::ir::refexec::run_named;
use crate::ir::semantics::SemValue;
use crate::ir::text::{parse_module, print_module};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{ConstPool, FloatBits};
use crate::ir::{Function, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

use puremp::Int;

/// A partial target: `<4 x i32>` (and its `<4 x i1>` mask) are legal, but the
/// only op it selects on them is `add` (plus extract/insert, needed by the
/// scalarizer, and whole-vector loads/stores).
struct OnlyAdd4;

impl VectorLegality for OnlyAdd4 {
    fn legal_type(&self, types: &TypeContext, ty: TypeId) -> bool {
        matches!(types.vector_parts(ty), Some((e, 4)) if matches!(types.get(e), Type::Int(32 | 1)))
    }

    fn legal_inst(&self, _: &TypeContext, _: &ConstPool, _: &Function, inst: &InstData) -> bool {
        matches!(
            inst.kind,
            InstKind::Bin(BinOp::Add)
                | InstKind::ExtractElement { .. }
                | InstKind::InsertElement { .. }
                | InstKind::Load { .. }
                | InstKind::Store { .. }
        )
    }
}

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    (m, syms)
}

fn int(w: u32, v: i64) -> SemValue {
    SemValue::int(w, Int::from_i64(v))
}

/// Run `name(args)` before and after legalizing `src` for `legality`, and
/// check the result refines the original.
fn check(src: &str, legality: &dyn VectorLegality, name: &str, args: &[SemValue]) {
    let (orig, syms) = parse(src);
    let (mut m, s2) = parse(src);
    legalize_vectors(&mut m, legality);
    if let Err(e) = crate::verify::verify_module(&m) {
        panic!("legalized module fails to verify: {e:?}\n{}", print_module(&m, &s2));
    }
    let want = run_named(&orig, &syms, name, args).unwrap_or_else(|e| panic!("source: {e:?}"));
    let got = run_named(&m, &s2, name, args)
        .unwrap_or_else(|e| panic!("legalized: {e:?}\n{}", print_module(&m, &s2)));
    assert!(
        !matches!(want, Some(SemValue::Poison)),
        "@{name}{args:?}: the source result is poison, so the check would be vacuous"
    );
    match (want, got) {
        (Some(w), Some(g)) => assert!(g.refines(&w), "@{name}{args:?}: {g:?} does not refine {w:?}\n{}", print_module(&m, &s2)),
        (None, None) => {}
        (w, g) => panic!("result shape differs: {w:?} vs {g:?}"),
    }
}

const PROGRAM: &str = r#"
module "lg"
global @g : <8 x i16> = <8 x i16> (i16 1, i16 -2, i16 3, i16 -4, i16 5, i16 -6, i16 7, i16 -8)

func @mix(<4 x i32>, <4 x i32>) -> <4 x i32> {
entry ^0(%a: <4 x i32>, %b: <4 x i32>):
  %s = add %a, %b : <4 x i32>
  %m = mul %s, <4 x i32> (i32 3, i32 -1, i32 5, i32 7) : <4 x i32>
  %c = icmp ult %m, %b : <4 x i1>
  %r = select %c, %m, %s : <4 x i32>
  %sh = shl %r, <4 x i32> (i32 1, i32 2, i32 3, i32 31) : <4 x i32>
  ret %sh
}

func @wide(<8 x i32>) -> <8 x i32> {
entry ^0(%v: <8 x i32>):
  %x = xor %v, <8 x i32> (i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8) : <8 x i32>
  %lo = shufflevector %x, %v, [0, 9, 2, 11] : <4 x i32>
  %hi = shufflevector %x, %x, [4, 5, 6, 7, 0, 1, 2, 3] : <8 x i32>
  %l2 = shufflevector %lo, %lo, [3, 2, 1, 0, 0, 1, 2, 3] : <8 x i32>
  %y = sub %hi, %l2 : <8 x i32>
  ret %y
}

func @main(i32) -> i32 {
entry ^0(%n: i32):
  %sp = splat %n : <8 x i32>
  %w = call @wide(%sp) : <8 x i32>
  %a = shufflevector %w, %w, [0, 1, 2, 3] : <4 x i32>
  %b = shufflevector %w, %w, [7, 6, 5, 4] : <4 x i32>
  %m = call @mix(%a, %b) : <4 x i32>
  %t = trunc %m : <4 x i16>
  %z = sext %t : <4 x i64>
  %r64 = reduce add %z : i64
  %mx = reduce smax %m : i32
  %mn = reduce umin %w : i32
  %g = load @g align 16 : <8 x i16>
  %g2 = add %g, <8 x i16> (i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1) : <8 x i16>
  store %g2, @g align 16 : <8 x i16>
  %g3 = load @g align 16 : <8 x i16>
  %gi = bitcast %g3 : <2 x i64>
  %e = extractelement %gi, 1 : i64
  %t32 = trunc %e : i32
  %r32 = trunc %r64 : i32
  %acc = add %r32, %mx : i32
  %acc2 = add %acc, %mn : i32
  %acc3 = xor %acc2, %t32 : i32
  br ^1(i32 0, %a, %acc3)
^1(%i: i32, %v: <4 x i32>, %s: i32):
  %v2 = add %v, %b : <4 x i32>
  %e2 = extractelement %v2, 2 : i32
  %s2 = add %s, %e2 : i32
  %i2 = add %i, i32 1 : i32
  %k = icmp slt %i2, i32 3 : i1
  cond_br %k, ^1(%i2, %v2, %s2), ^2(%s2)
^2(%out: i32):
  switch %n, ^3(%out) [7: ^4(%v)]
^3(%o: i32):
  ret %o
^4(%vv: <4 x i32>):
  %q = reduce xor %vv : i32
  ret %q
}

func @masks(i8) -> i32 {
entry ^0(%x: i8):
  %m = bitcast %x : <8 x i1>
  %z = zext %m : <8 x i16>
  %n = xor %m, <8 x i1> (i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0) : <8 x i1>
  %back = bitcast %n : i8
  %w = zext %back : i32
  %r = reduce add %z : i16
  %rw = zext %r : i32
  %s = shl %rw, i32 8 : i32
  %o = or %w, %s : i32
  ret %o
}

func @floats(f32) -> f64 {
entry ^0(%x: f32):
  %v = splat %x : <4 x f32>
  %w = fmul %v, <4 x f32> (f32 0x3f800000, f32 0xc0000000, f32 0x40400000, f32 0x80000000) : <4 x f32>
  %n = fneg %w : <4 x f32>
  %c = fcmp ole %w, %n : <4 x i1>
  %s = select %c, %w, %n : <4 x f32>
  %i = fptosi %s : <4 x i32>
  %f = sitofp %i : <4 x f32>
  %e = fpext %f : <4 x f64>
  %lo = shufflevector %e, %e, [0, 1] : <2 x f64>
  %bits = bitcast %lo : <4 x i32>
  %back = bitcast %bits : <2 x f64>
  %r = reduce fadd %back : f64
  %fr = freeze %r : f64
  ret %fr
}
"#;

#[test]
fn scalar_only_legalization_removes_all_vectors_and_preserves_results() {
    let (mut m, _) = parse(PROGRAM);
    legalize_vectors(&mut m, &ScalarOnly);
    assert!(!uses_vectors(&m), "a scalar-only target keeps no vector code");
    for n in [0i64, 1, 7, -5, 1 << 20, i64::from(i32::MIN)] {
        check(PROGRAM, &ScalarOnly, "main", &[int(32, n)]);
    }
    for x in [0i64, 1, 0x5a, 0xff, 0x80] {
        check(PROGRAM, &ScalarOnly, "masks", &[int(8, x)]);
    }
    for x in [0.0f32, 1.5, -3.25, 1e20, f32::NAN] {
        check(PROGRAM, &ScalarOnly, "floats", &[SemValue::Float(FloatBits::F32(x.to_bits()))]);
    }
}

#[test]
fn partial_legality_mixes_whole_and_split_vectors() {
    for n in [0i64, 3, 7, -9, 123_456] {
        check(PROGRAM, &OnlyAdd4, "main", &[int(32, n)]);
    }
    for x in [0i64, 0x5a, 0xff] {
        check(PROGRAM, &OnlyAdd4, "masks", &[int(8, x)]);
    }
    for x in [2.5f32, -7.0] {
        check(PROGRAM, &OnlyAdd4, "floats", &[SemValue::Float(FloatBits::F32(x.to_bits()))]);
    }
    // The legal type survives where it is legal: @mix keeps its <4 x i32>
    // signature and whole-vector add, while its mul/icmp/select/shl are
    // scalarized through extract/insert.
    let (mut m, syms) = parse(PROGRAM);
    legalize_vectors(&mut m, &OnlyAdd4);
    let text = print_module(&m, &syms);
    assert!(text.contains("func @mix(<4 x i32>, <4 x i32>) -> <4 x i32>"), "{text}");
    assert!(text.contains("func @wide(ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> void"), "{text}");
}

#[test]
fn illegal_signatures_use_lanes_and_a_hidden_result_pointer() {
    let (mut m, syms) = parse(PROGRAM);
    legalize_vectors(&mut m, &ScalarOnly);
    let text = print_module(&m, &syms);
    assert!(text.contains("func @mix(ptr, i32, i32, i32, i32, i32, i32, i32, i32) -> void"), "{text}");
    assert!(text.contains("func @masks(i8) -> i32"), "{text}");
}

#[test]
fn ub_lanes_stay_ub_and_poison_lanes_stay_poison() {
    let src = r#"
module "u"
func @d(i32) -> i32 {
entry ^0(%x: i32):
  %v = splat %x : <3 x i32>
  %q = udiv <3 x i32> (i32 12, i32 9, i32 6), %v : <3 x i32>
  %e = extractelement %q, 1 : i32
  ret %e
}
func @p(i32) -> i32 {
entry ^0(%x: i32):
  %v = insertelement <2 x i32> poison, %x, 0 : <2 x i32>
  %s = shl %v, <2 x i32> (i32 40, i32 1) : <2 x i32>
  %a = extractelement %s, 0 : i32
  %b = extractelement %s, 1 : i32
  %fb = freeze %b : i32
  %r = add %a, %fb : i32
  ret %r
}
"#;
    check(src, &ScalarOnly, "d", &[int(32, 3)]);
    let (mut m, syms) = parse(src);
    legalize_vectors(&mut m, &ScalarOnly);
    // A zero lane divisor is still UB after scalarization.
    assert!(run_named(&m, &syms, "d", &[int(32, 0)]).is_err());
    // An over-wide shift in lane 0 poisons the sum (poison + anything).
    assert_eq!(run_named(&m, &syms, "p", &[int(32, 1)]).unwrap(), Some(SemValue::Poison));
}

#[test]
fn modules_without_vectors_are_borrowed_unchanged() {
    let src = "module \"s\"\nfunc @f(i32) -> i32 {\nentry ^0(%x: i32):\n  ret %x\n}\n";
    let (m, _) = parse(src);
    assert!(matches!(super::legalized(&m, &ScalarOnly), std::borrow::Cow::Borrowed(_)));
}
