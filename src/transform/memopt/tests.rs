//! Tests for `memopt`: scalar replacement of slots that bulk ops touch,
//! forwarding through stores, fills and copies, and dead-write removal, each
//! checked on the rewritten shape and, by running the program before and
//! after with the reference executor, as a refinement.

use super::MemOpt;
use crate::ir::refexec::run_named;
use crate::ir::semantics::SemValue;
use crate::ir::text::{parse_module, print_module};
use crate::ir::{FuncId, Function, InstKind, Module};
use crate::pass::{Changed, ModulePass};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize};

use puremp::Int;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    (m, syms)
}

fn func<'a>(m: &'a Module, syms: &StrInterner, name: &str) -> &'a Function {
    let i = m.functions().position(|f| syms.resolve(f.name) == name).unwrap_or_else(|| panic!("no @{name}"));
    m.function(FuncId::from_index(i))
}

fn count(f: &Function, pred: impl Fn(&InstKind) -> bool) -> usize {
    f.blocks().map(|(_, b)| b.insts().iter().filter(|&&i| pred(&f.inst(i).kind)).count()).sum()
}

fn int(v: i64) -> SemValue {
    SemValue::int(64, Int::from_i64(v))
}

/// Run `memopt` alone, then the -O2 pipeline, on `src`; each result must
/// verify and agree with the original on every `(function, args)` case (a
/// poison original result accepts anything). Returns the two rewritten
/// modules.
fn check(src: &str, cases: &[(&str, Vec<i64>)]) -> ((Module, StrInterner), (Module, StrInterner)) {
    let (orig, syms) = parse(src);
    let (mut one, s1) = parse(src);
    MemOpt.run(&mut one);
    crate::verify::verify_module(&one).unwrap_or_else(|e| panic!("memopt output: {e:?}\n{}", print_module(&one, &s1)));
    let (mut o2, s2) = parse(src);
    optimize(&mut o2, OptLevel::O2);
    crate::verify::verify_module(&o2).unwrap_or_else(|e| panic!("-O2 output: {e:?}"));
    for (name, args) in cases {
        let a: Vec<SemValue> = args.iter().map(|&x| int(x)).collect();
        let want = run_named(&orig, &syms, name, &a).unwrap_or_else(|e| panic!("source @{name}{args:?}: {e:?}"));
        for (what, m, s) in [("memopt", &one, &s1), ("-O2", &o2, &s2)] {
            let got = run_named(m, s, name, &a)
                .unwrap_or_else(|e| panic!("{what} @{name}{args:?}: {e:?}\n{}", print_module(m, s)));
            match (&want, &got) {
                (Some(w), Some(g)) => assert!(g.refines(w), "{what} @{name}{args:?}: {g:?} vs {w:?}\n{}", print_module(m, s)),
                (None, None) => {}
                _ => panic!("{what} @{name}{args:?}: {got:?} vs {want:?}"),
            }
        }
    }
    ((one, s1), (o2, s2))
}

/// Struct copies between locals and through pointers.
const STRUCTS: &str = r#"module "structs"

func @local_copy(i64, i64) -> i64 {
entry ^0(%x: i64, %y: i64):
  %a = alloca { i64, i64 } : ptr
  %b = alloca { i64, i64 } : ptr
  store %x, %a align 8 : i64
  %a1 = ptr_add %a, i64 8 : ptr
  store %y, %a1 align 8 : i64
  memcpy %b, %a, i64 16 align 8
  %b0 = load %b align 8 : i64
  %b1p = ptr_add %b, i64 8 : ptr
  %b1 = load %b1p align 8 : i64
  %r = sub %b0, %b1 : i64
  ret %r
}

func @chain(i64, i64) -> i64 {
entry ^0(%x: i64, %yw: i64):
  %y = trunc %yw : i32
  %a = alloca { i64, i32, i32 } : ptr
  %t = alloca { i64, i32, i32 } : ptr
  %c = alloca { i64, i32, i32 } : ptr
  store %x, %a align 8 : i64
  %a1 = ptr_add %a, i64 8 : ptr
  store %y, %a1 align 4 : i32
  %a2 = ptr_add %a, i64 12 : ptr
  store i32 77, %a2 align 4 : i32
  memcpy %t, %a, i64 16 align 8
  memcpy %c, %t, i64 16 align 8
  %c0 = load %c align 8 : i64
  %c2p = ptr_add %c, i64 12 : ptr
  %c2 = load %c2p align 4 : i32
  %c1p = ptr_add %c, i64 8 : ptr
  %c1 = load %c1p align 4 : i32
  %w2 = zext %c2 : i64
  %w1 = sext %c1 : i64
  %s = add %c0, %w2 : i64
  %r = add %s, %w1 : i64
  ret %r
}

func @get_y(ptr) -> i64 {
entry ^0(%p: ptr):
  %tmp = alloca { i64, i64, i64 } : ptr
  memcpy %tmp, %p, i64 24 align 8
  %y = ptr_add %tmp, i64 8 : ptr
  %v = load %y align 8 : i64
  ret %v
}

func @put_pair(ptr, i64) -> void {
entry ^0(%out: ptr, %x: i64):
  %s = alloca { i64, i64 } : ptr
  store %x, %s align 8 : i64
  %s1 = ptr_add %s, i64 8 : ptr
  %x2 = mul %x, i64 3 : i64
  store %x2, %s1 align 8 : i64
  memcpy %out, %s, i64 16 align 8
  ret
}

func @zeroed(i64) -> i64 {
entry ^0(%x: i64):
  %s = alloca { i64, i32, i16, i8, i8 } : ptr
  memset %s, i8 0, i64 16 align 8
  %f = ptr_add %s, i64 12 : ptr
  store i16 5, %f align 4 : i16
  %v0 = load %s align 8 : i64
  %v1p = ptr_add %s, i64 8 : ptr
  %v1 = load %v1p align 4 : i32
  %v2 = load %f align 2 : i16
  %v3p = ptr_add %s, i64 15 : ptr
  %v3 = load %v3p align 1 : i8
  %w1 = zext %v1 : i64
  %w2 = zext %v2 : i64
  %w3 = zext %v3 : i64
  %a = add %v0, %w1 : i64
  %b = add %a, %w2 : i64
  %c = add %b, %w3 : i64
  %r = add %c, %x : i64
  ret %r
}

func @drive(i64, i64) -> i64 {
entry ^0(%x: i64, %y: i64):
  %buf = alloca [3 x i64] : ptr
  store %x, %buf align 8 : i64
  %b1 = ptr_add %buf, i64 8 : ptr
  store %y, %b1 align 8 : i64
  %b2 = ptr_add %buf, i64 16 : ptr
  store i64 9, %b2 align 8 : i64
  %g = call @get_y(%buf) : i64
  %out = alloca [2 x i64] : ptr
  call @put_pair(%out, %g) : void
  %o1p = ptr_add %out, i64 8 : ptr
  %o1 = load %o1p align 8 : i64
  %o0 = load %out align 8 : i64
  %r = add %o0, %o1 : i64
  ret %r
}
"#;

#[test]
fn struct_copies_become_scalars() {
    let cases: Vec<(&str, Vec<i64>)> = [(3i64, 5i64), (-1, 1 << 40), (0, 0)]
        .iter()
        .flat_map(|&(x, y)| {
            vec![
                ("local_copy", vec![x, y]),
                ("chain", vec![x, y]),
                ("zeroed", vec![x]),
                ("drive", vec![x, y]),
            ]
        })
        .collect();
    let ((one, s1), (o2, s2)) = check(STRUCTS, &cases);
    // memopt alone: the local copies are split, no bulk op left on a slot.
    for name in ["local_copy", "chain", "zeroed"] {
        let f = func(&one, &s1, name);
        assert_eq!(count(f, InstKind::is_bulk_memory), 0, "@{name}:\n{}", print_module(&one, &s1));
    }
    // After -O2 (mem2reg promotes the slices): no memory traffic at all.
    for name in ["local_copy", "chain", "zeroed"] {
        let f = func(&o2, &s2, name);
        let mem = count(f, |k| matches!(k, InstKind::Load { .. } | InstKind::Store { .. } | InstKind::Alloca { .. }));
        assert_eq!(mem, 0, "@{name} still touches memory:\n{}", print_module(&o2, &s2));
    }
    // A copy in from a pointer and a field read: one load from the pointer.
    let f = func(&o2, &s2, "get_y");
    assert_eq!(count(f, |k| matches!(k, InstKind::Load { .. })), 1, "{}", print_module(&o2, &s2));
    assert_eq!(count(f, |k| matches!(k, InstKind::Alloca { .. } | InstKind::MemCopy { .. })), 0);
    // A struct built locally and copied out: two stores to the pointer.
    let f = func(&o2, &s2, "put_pair");
    assert_eq!(count(f, |k| matches!(k, InstKind::Store { .. })), 2, "{}", print_module(&o2, &s2));
    assert_eq!(count(f, |k| matches!(k, InstKind::Alloca { .. } | InstKind::MemCopy { .. })), 0);
}

/// Forwarding through stores, fills and copies, and dead writes.
const FORWARD: &str = r#"module "forward"

global @g : [4 x i64] = [4 x i64] (i64 1, i64 2, i64 3, i64 4)

func @store_then_copy(ptr, i64) -> void {
entry ^0(%dst: ptr, %v: i64):
  %t = alloca i64 : ptr
  store %v, %t align 8 : i64
  memcpy %dst, %t, i64 8 align 8
  ret
}

func @fill_then_load(ptr) -> i64 {
entry ^0(%p: ptr):
  memset %p, i8 0, i64 32 align 8
  %q = ptr_add %p, i64 8 : ptr
  %v = load %q align 8 : i64
  %w = ptr_add %p, i64 3 : ptr
  %b = load %w align 1 : i8
  %bw = zext %b : i64
  %r = add %v, %bw : i64
  ret %r
}

func @copy_of_copy(ptr, ptr) -> void {
entry ^0(%dst: ptr, %src: ptr):
  %t = alloca [4 x i64] : ptr
  memcpy %t, %src, i64 32 align 8
  memcpy %dst, %t, i64 32 align 8
  ret
}

func @copy_then_load(ptr, ptr) -> i64 {
entry ^0(%dst: ptr, %src: ptr):
  memcpy %dst, %src, i64 32 align 8
  %q = ptr_add %dst, i64 16 : ptr
  %v = load %q align 8 : i64
  ret %v
}

func @dead_fill(ptr, i64) -> void {
entry ^0(%p: ptr, %x: i64):
  memset %p, i8 0, i64 16 align 8
  store %x, %p align 8 : i64
  %q = ptr_add %p, i64 8 : ptr
  store %x, %q align 8 : i64
  ret
}

func @live_fill(ptr, i64) -> void {
entry ^0(%p: ptr, %x: i64):
  memset %p, i8 0, i64 16 align 8
  %v = load %p align 8 : i64
  %v2 = add %v, %x : i64
  store %v2, %p align 8 : i64
  %q = ptr_add %p, i64 8 : ptr
  store %x, %q align 8 : i64
  ret
}

func @clobber(ptr, ptr) -> i64 {
entry ^0(%dst: ptr, %src: ptr):
  %t = alloca [2 x i64] : ptr
  memcpy %t, %src, i64 16 align 8
  call @scribble(%src) : void
  %q = ptr_add %t, i64 8 : ptr
  %v = load %q align 8 : i64
  ret %v
}

func @scribble(ptr) -> void {
entry ^0(%p: ptr):
  %q = ptr_add %p, i64 8 : ptr
  store i64 -1, %q align 8 : i64
  ret
}

func @drive(i64, i64) -> i64 {
entry ^0(%x: i64, %y: i64):
  %a = alloca [4 x i64] : ptr
  %b = alloca [4 x i64] : ptr
  memcpy %a, @g, i64 32 align 8
  call @store_then_copy(%b, %x) : void
  %b0 = load %b align 8 : i64
  %f = call @fill_then_load(%b) : i64
  %a2p = ptr_add %a, i64 16 : ptr
  store %y, %a2p align 8 : i64
  call @copy_of_copy(%b, %a) : void
  %c = call @copy_then_load(%a, %b) : i64
  call @dead_fill(%b, %x) : void
  %d0 = load %b align 8 : i64
  call @live_fill(%a, %y) : void
  %l0 = load %a align 8 : i64
  %l1p = ptr_add %a, i64 8 : ptr
  %l1 = load %l1p align 8 : i64
  %k = call @clobber(%b, %a) : i64
  %s1 = add %b0, %f : i64
  %s2 = add %s1, %c : i64
  %s3 = add %s2, %d0 : i64
  %s4 = add %s3, %l0 : i64
  %s5 = add %s4, %l1 : i64
  %s6 = mul %s5, i64 31 : i64
  %r = add %s6, %k : i64
  ret %r
}
"#;

#[test]
fn forwarding_and_dead_writes() {
    let cases: Vec<(&str, Vec<i64>)> = vec![("drive", vec![5, 11]), ("drive", vec![-3, 0]), ("drive", vec![1 << 50, -7])];
    let ((one, s1), _) = check(FORWARD, &cases);
    let text = print_module(&one, &s1);
    let f = |n: &str| func(&one, &s1, n);
    // A copy of a just-stored value is that store.
    assert_eq!(count(f("store_then_copy"), |k| matches!(k, InstKind::MemCopy { .. })), 0, "{text}");
    // Loads from a constant fill are constants.
    assert_eq!(count(f("fill_then_load"), |k| matches!(k, InstKind::Load { .. })), 0, "{text}");
    // A copy of a copy reads the original (a memmove: the pointers may alias).
    let cc = f("copy_of_copy");
    assert_eq!(count(cc, |k| matches!(k, InstKind::MemCopy { overlapping: true, .. })), 1, "{text}");
    // A load from a copied range reads the source.
    let ctl = f("copy_then_load");
    assert_eq!(count(ctl, |k| matches!(k, InstKind::MemCopy { .. })), 1);
    // The fill fully overwritten before any read is gone; the read one stays.
    assert_eq!(count(f("dead_fill"), InstKind::is_bulk_memory), 0, "{text}");
    assert_eq!(count(f("live_fill"), InstKind::is_bulk_memory), 1, "{text}");
}

#[test]
fn copy_of_copy_ends_up_one_copy_at_o2() {
    let (_, (o2, s2)) = check(FORWARD, &[("drive", vec![2, 3])]);
    let f = func(&o2, &s2, "copy_of_copy");
    assert_eq!(count(f, InstKind::is_bulk_memory), 1, "{}", print_module(&o2, &s2));
    assert_eq!(count(f, |k| matches!(k, InstKind::Alloca { .. })), 0, "{}", print_module(&o2, &s2));
}

#[test]
fn volatile_and_escaping_are_left_alone() {
    let src = r#"module "keep"
func @vol(ptr) -> i64 {
entry ^0(%p: ptr):
  memset volatile %p, i8 0, i64 8 align 8
  store i64 1, %p align 8 : i64
  %t = alloca i64 : ptr
  store i64 4, %t align 8 : i64
  memcpy volatile %p, %t, i64 8 align 8
  %v = load %p align 8 : i64
  ret %v
}
func @esc(ptr) -> i64 {
entry ^0(%p: ptr):
  %t = alloca { i64, i64 } : ptr
  memcpy %t, %p, i64 16 align 8
  call @sink(%t) : void
  %v = load %t align 8 : i64
  ret %v
}
func @sink(ptr) -> void
"#;
    let (mut m, syms) = parse(src);
    MemOpt.run(&mut m);
    crate::verify::verify_module(&m).unwrap();
    let f = func(&m, &syms, "vol");
    assert_eq!(count(f, |k| matches!(k, InstKind::MemSet { volatile: true, .. })), 1);
    assert_eq!(count(f, |k| matches!(k, InstKind::MemCopy { volatile: true, .. })), 1);
    let f = func(&m, &syms, "esc");
    assert_eq!(count(f, InstKind::is_bulk_memory), 1, "an escaping slot keeps its copy");
    assert_eq!(count(f, |k| matches!(k, InstKind::Load { .. })), 1, "and its load after the call");
}

#[test]
fn pass_through_slots_keep_per_byte_poison() {
    // `t` is filled by a copy from partially uninitialized memory and copied
    // out again: splitting it into an i64 would turn the defined low byte
    // into poison. The low byte must stay defined.
    let src = r#"module "poison"
func @f() -> i8 {
entry ^0:
  %src = alloca i64 : ptr
  store i8 42, %src align 8 : i8
  %t = alloca i64 : ptr
  %dst = alloca i64 : ptr
  memcpy %t, %src, i64 8 align 8
  memcpy %dst, %t, i64 8 align 8
  %b = load %dst align 8 : i8
  ret %b
}
"#;
    let ((one, s1), (o2, s2)) = check(src, &[("f", vec![])]);
    for (m, s) in [(&one, &s1), (&o2, &s2)] {
        assert_eq!(run_named(m, s, "f", &[]), Ok(Some(SemValue::int(8, Int::from_u64(42)))), "{}", print_module(m, s));
    }
}

#[test]
fn unchanged_without_memory_work() {
    let src = "module \"n\"\nfunc @f(i64) -> i64 {\nentry ^0(%x: i64):\n  %r = add %x, i64 1 : i64\n  ret %r\n}\n";
    let (mut m, _) = parse(src);
    assert_eq!(MemOpt.run(&mut m), Changed::No);
}
