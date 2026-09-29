//! Unit tests for the secret-taint analysis ([`super::SecretTaint`]).

use super::{MemRoot, Origin, SecretTaint, Taint};
use crate::analysis::domain::AbstractDomain;
use crate::analysis::soundness::check_integer_transfer_sound;
use crate::ir::inst::InstKind;
use crate::ir::value::ValueId;
use crate::ir::{BlockId, FuncId, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

/// Parse `src` and return the module.
fn parse(src: &str) -> Module {
    let mut syms = StrInterner::new();
    crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"))
}

/// The function named by position `idx`.
fn fid(idx: usize) -> FuncId {
    FuncId::from_index(idx)
}

/// The results of the non-terminator instructions of block `b`, in order.
fn results(m: &Module, f: FuncId, b: usize) -> Vec<ValueId> {
    let func = m.function(f);
    func.block(BlockId::from_index(b)).insts().iter().filter_map(|&i| func.inst(i).result()).collect()
}

/// The operand of the `ret` terminator of block `b`.
fn ret_operand(m: &Module, f: FuncId, b: usize) -> ValueId {
    let func = m.function(f);
    let t = func.block(BlockId::from_index(b)).terminator().expect("terminated");
    assert!(matches!(func.inst(t).kind, InstKind::Ret));
    func.inst(t).operands()[0]
}

/// Taint of each result of block `b`, as a string of `S`/`P` for compact asserts.
fn pattern(m: &Module, f: FuncId, b: usize) -> String {
    let t = SecretTaint::compute(m, f);
    results(m, f, b).iter().map(|&v| if t.is_secret(v) { 'S' } else { 'P' }).collect()
}

#[test]
fn lattice_laws_and_vacuous_soundness() {
    let all = [Taint::Bottom, Taint::Public, Taint::Secret];
    for a in all {
        assert!(Taint::bottom().le(&a) && a.le(&Taint::top()));
        for b in all {
            let j = a.join(&b);
            assert!(a.le(&j) && b.le(&j));
            assert_eq!(a.le(&b), a.join(&b) == b);
            assert_eq!(j, b.join(&a));
        }
    }
    // Taint's γ is every value (for a reached element), so its transfer is
    // trivially sound against the reference semantics.
    assert!(check_integer_transfer_sound::<Taint>(200, 7).is_sound());
}

#[test]
fn secret_params_propagate_through_arithmetic_and_casts() {
    let m = parse(
        r#"module "t"
func @f(secret i64, i64) -> i64 {
entry ^0(%s: i64, %p: i64):
  %a = add %s, %p : i64
  %b = mul %p, %p : i64
  %c = trunc %a : i32
  %d = zext %c : i64
  %e = icmp ult %s, %p : i1
  %g = select %e, %p, %b : i64
  %h = freeze %b : i64
  %k = xor %h, i64 3 : i64
  ret %k
}
"#,
    );
    // a S, b P, c S, d S, e S, g S (secret condition), h P, k P
    assert_eq!(pattern(&m, fid(0), 0), "SPSSSSPP");
}

#[test]
fn declassify_ends_secrecy() {
    let m = parse(
        r#"module "t"
func @f(secret i64) -> i64 {
entry ^0(%s: i64):
  %a = add %s, i64 1 : i64
  %d = declassify %a : i64
  %e = mul %d, i64 3 : i64
  ret %e
}
"#,
    );
    assert_eq!(pattern(&m, fid(0), 0), "SPP");
}

#[test]
fn block_arguments_carry_taint_around_loops() {
    // acc starts public and becomes secret through the back edge.
    let m = parse(
        r#"module "t"
func @f(secret i64, i64) -> i64 {
entry ^0(%s: i64, %n: i64):
  br ^1(i64 0, i64 0)
^1(%i: i64, %acc: i64):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %acc2 = xor %acc, %s : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %acc2)
^3:
  ret %acc
}
"#,
    );
    let t = SecretTaint::compute(&m, fid(0));
    let func = m.function(fid(0));
    let hdr = func.block(BlockId::from_index(1)).params();
    assert!(!t.is_secret(hdr[0]), "the counter stays public");
    assert!(t.is_secret(hdr[1]), "the accumulator is secret via the back edge");
    assert_eq!(pattern(&m, fid(0), 1), "P", "the loop condition is public");
    assert_eq!(pattern(&m, fid(0), 2), "SP");
    // The chain explains itself: acc <- acc2 (block arg) <- %s (operand).
    let Some(Origin::BlockArg(_, from)) = t.origin(&m, fid(0), hdr[1]) else { panic!() };
    let Some(Origin::Operand(_, src)) = t.origin(&m, fid(0), from) else { panic!() };
    assert_eq!(t.origin(&m, fid(0), src), Some(Origin::SecretParam(0)));
    let chain = t.explain(&m, fid(0), hdr[1]);
    assert_eq!(chain.len(), 3);
    assert_eq!(chain.last(), Some(&Origin::SecretParam(0)));
    assert!(t.explain(&m, fid(0), hdr[0]).is_empty());
}

#[test]
fn stack_slots_are_tracked_individually() {
    let m = parse(
        r#"module "t"
func @f(secret i64, i64) -> i64 {
entry ^0(%s: i64, %p: i64):
  %x = alloca i64 : ptr
  %y = alloca [2 x i64] : ptr
  store %s, %x align 8 : i64
  %y1 = ptr_add %y, i64 8 : ptr
  store %p, %y1 align 8 : i64
  %a = load %x align 8 : i64
  %b = load %y1 align 8 : i64
  %c = load %y align 8 : i64
  ret %b
}
"#,
    );
    // x P(ptr), y P, y1 P, a S (slot x holds a secret), b P, c P
    assert_eq!(pattern(&m, fid(0), 0), "PPPSPP");
    let t = SecretTaint::compute(&m, fid(0));
    assert!(matches!(t.root_of(results(&m, fid(0), 0)[2]), MemRoot::Stack(_)));
}

#[test]
fn escaping_slots_and_calls_are_unknown_memory() {
    // `x` escapes into a call; the call receives a secret, so unknown memory
    // (including what `%q` points to) may hold a secret afterwards.
    let m = parse(
        r#"module "t"
func @sink(ptr, secret i64) -> void

func @f(secret i64, ptr) -> i64 {
entry ^0(%s: i64, %q: ptr):
  %x = alloca i64 : ptr
  call @sink(%x, %s) : void
  %a = load %x align 8 : i64
  %b = load %q align 8 : i64
  ret %b
}
"#,
    );
    let t = SecretTaint::compute(&m, fid(1));
    let r = results(&m, fid(1), 0);
    assert_eq!(t.root_of(r[0]), MemRoot::Unknown, "an escaping slot is unknown memory");
    assert!(t.is_secret(r[1]) && t.is_secret(r[2]));
    assert!(t.memory_secret(MemRoot::Unknown));
}

#[test]
fn public_calls_leave_memory_public() {
    let m = parse(
        r#"module "t"
func @use(ptr) -> i64

func @f(secret i64, ptr) -> i64 {
entry ^0(%s: i64, %q: ptr):
  %r = call @use(%q) : i64
  %b = load %q align 8 : i64
  %c = add %b, %r : i64
  ret %c
}
"#,
    );
    assert_eq!(pattern(&m, fid(1), 0), "PPP");
}

#[test]
fn secret_globals_secret_loads_and_secret_returns() {
    let m = parse(
        r#"module "t"
global secret @key : [4 x i64] = [4 x i64] (i64 1, i64 2, i64 3, i64 4)

global @pub : i64 = i64 9

func @k() -> secret i64

func @p() -> i64

func @f(ptr) -> i64 {
entry ^0(%q: ptr):
  %e = ptr_add @key, i64 8 : ptr
  %a = load %e align 8 : i64
  %b = load @pub align 8 : i64
  %c = load secret %q align 8 : i64
  %d = load %q align 8 : i64
  %x = call @k() : i64
  %y = call @p() : i64
  ret %y
}
"#,
    );
    // e P (an address), a S (secret global), b P, c S (flagged), d P, x S, y P
    assert_eq!(pattern(&m, fid(2), 0), "PSPSPSP");
    let t = SecretTaint::compute(&m, fid(2));
    let r = results(&m, fid(2), 0);
    assert!(matches!(t.origin(&m, fid(2), r[1]), Some(Origin::SecretMemory(_, MemRoot::Global(_)))));
    assert!(matches!(t.origin(&m, fid(2), r[3]), Some(Origin::SecretLoad(_))));
    assert!(matches!(t.origin(&m, fid(2), r[5]), Some(Origin::SecretReturn(_, _))));
}

#[test]
fn storing_a_secret_to_a_global_taints_aliasing_loads() {
    let m = parse(
        r#"module "t"
global @g : i64 = i64 0

global @h : i64 = i64 0

func @f(secret i64, ptr) -> i64 {
entry ^0(%s: i64, %q: ptr):
  %a = load %q align 8 : i64
  %b = load @h align 8 : i64
  store secret %s, @g align 8 : i64
  %c = load @g align 8 : i64
  ret %b
}
"#,
    );
    // a S (an unknown pointer may alias @g), b P (a distinct global), c S.
    // Flow-insensitive: `a` is secret although it is read before the store.
    assert_eq!(pattern(&m, fid(0), 0), "SPS");
}

#[test]
fn writes_through_unknown_pointers_leave_named_globals_public() {
    // A preemption flag read by name stays public although the function
    // writes secrets through a pointer and passes one to a callee.
    let m = parse(
        r#"module "t"
global @flag : i32 = i32 0

func @sink(secret i64) -> void

func @f(secret i64, ptr) -> i64 {
entry ^0(%s: i64, %q: ptr):
  store secret %s, %q align 8 : i64
  call @sink(%s) : void
  %v = load volatile @flag align 4 : i32
  %a = load %q align 8 : i64
  ret i64 0
}
"#,
    );
    // v P, a S (unknown memory holds a secret)
    assert_eq!(pattern(&m, fid(1), 0), "PS");
}

#[test]
fn memory_and_values_reach_a_joint_fixpoint() {
    // The secret reaches slot x only through a load of slot y, which itself
    // only holds a secret after the loop's store: two rounds of memory.
    let m = parse(
        r#"module "t"
func @f(secret i64) -> i64 {
entry ^0(%s: i64):
  %x = alloca i64 : ptr
  %y = alloca i64 : ptr
  store i64 0, %x align 8 : i64
  store i64 0, %y align 8 : i64
  %vy = load %y align 8 : i64
  store %vy, %x align 8 : i64
  store %s, %y align 8 : i64
  %vx = load %x align 8 : i64
  ret %vx
}
"#,
    );
    assert_eq!(pattern(&m, fid(0), 0), "PPSS");
}

#[test]
fn a_module_without_secrets_has_no_taint() {
    let mut syms = StrInterner::new();
    let m = crate::ir::tests::atomics_module(&mut syms);
    for i in 0..m.function_count() {
        if !m.function(fid(i)).is_declaration() {
            assert!(!SecretTaint::compute(&m, fid(i)).any_secret());
        }
    }
}

#[test]
fn ret_of_fixture_is_secret() {
    let mut syms = StrInterner::new();
    let m = crate::ir::tests::secret_module(&mut syms);
    let t = SecretTaint::compute(&m, fid(1));
    assert!(t.is_secret(ret_operand(&m, fid(1), 0)));
}
