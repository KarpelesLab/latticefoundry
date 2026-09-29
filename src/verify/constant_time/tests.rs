//! Accept/reject tests for the constant-time verifier.

use super::{CtPolicy, CtRole, ct_violations, verify_function_ct, verify_module_ct};
use crate::ir::{FuncId, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

/// Parse `src`; it must be structurally valid.
fn parse(src: &str) -> Module {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    for i in 0..m.function_count() {
        let d = crate::verify::verify_function(&m, FuncId::from_index(i));
        assert!(d.is_empty(), "structural: {d:?}");
    }
    m
}

/// The roles of every violation in the *last* function of `src`.
fn roles(src: &str, policy: CtPolicy) -> Vec<CtRole> {
    let m = parse(src);
    let f = FuncId::from_index(m.function_count() - 1);
    ct_violations(&m, f, policy).into_iter().map(|v| v.role).collect()
}

fn roles_default(src: &str) -> Vec<CtRole> {
    roles(src, CtPolicy::DEFAULT)
}

/// Wrap a body in `func @f(secret i64 %s, i64 %p, ptr %q) -> i64`.
fn body(insts: &str) -> String {
    format!(
        "module \"t\"\nfunc @f(secret i64, i64, ptr) -> i64 {{\nentry ^0(%s: i64, %p: i64, %q: ptr):\n{insts}}}\n"
    )
}

#[test]
fn branch_on_a_secret_is_rejected() {
    let src = r#"module "t"
func @f(secret i64, i64) -> i64 {
entry ^0(%s: i64, %p: i64):
  %x = xor %s, %p : i64
  %c = icmp eq %x, i64 0 : i1
  cond_br %c, ^1, ^2
^1:
  ret i64 1
^2:
  ret i64 0
}
"#;
    assert_eq!(roles_default(src), [CtRole::BranchCondition]);
    let m = parse(src);
    let d = verify_function_ct(&m, FuncId::from_index(0), CtPolicy::DEFAULT);
    assert_eq!(d.len(), 1);
    let msg = &d[0].message;
    // Which value, which use, and where it comes from.
    assert!(msg.contains("secret-derived %3 is the condition of a `cond_br`"), "{msg}");
    assert!(msg.contains("(`cond_br` in block ^0)"), "{msg}");
    assert!(msg.contains("%3 <- %2 <- secret parameter 0") || msg.contains("<- secret parameter 0"), "{msg}");
    assert!(verify_module_ct(&m).is_err());
    // verify_module includes the check.
    let err = crate::verify::verify_module(&m).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("constant-time violation")));
}

#[test]
fn switch_on_a_secret_is_rejected() {
    let src = body("  switch %s, ^1 [0: ^1, 1: ^1]\n^1:\n  ret i64 0\n");
    assert_eq!(roles_default(&src), [CtRole::SwitchCondition]);
}

#[test]
fn declassify_then_branch_is_accepted() {
    let src = r#"module "t"
func @f(secret i64, i64) -> i64 {
entry ^0(%s: i64, %p: i64):
  %x = xor %s, %p : i64
  %d = declassify %x : i64
  %c = icmp eq %d, i64 0 : i1
  cond_br %c, ^1, ^2
^1:
  ret i64 1
^2:
  %q = udiv %p, %d : i64
  ret %q
}
"#;
    assert!(roles_default(src).is_empty());
    assert!(crate::verify::verify_module(&parse(src)).is_ok());
}

#[test]
fn select_on_a_secret_is_accepted() {
    let src = r#"module "t"
func @f(secret i64, i64, i64) -> secret i64 {
entry ^0(%s: i64, %a: i64, %b: i64):
  %c = icmp ult %s, %a : i1
  %r = select %c, %a, %b : i64
  %t = trunc %s : i1
  %r2 = select %t, %r, %s : i64
  ret %r2
}
"#;
    assert!(roles_default(src).is_empty());
}

#[test]
fn indexing_by_a_secret_is_rejected() {
    let src = body(
        "  %e = ptr_add %q, %s : ptr\n  %v = load %e align 8 : i64\n  %p2 = inttoptr %s : ptr\n  store %p, %p2 align 8 : i64\n  ret %v\n",
    );
    // The ptr_add offset, the load address (derived from it), the store
    // address; `%v` itself is public (memory through an unknown pointer).
    assert_eq!(roles_default(&src), [CtRole::Address, CtRole::Address, CtRole::Address]);
}

#[test]
fn dividing_by_or_dividing_a_secret_is_rejected() {
    let src = body(
        "  %a = udiv %p, %s : i64\n  %b = srem %s, %p : i64\n  %c = udiv %p, i64 3 : i64\n  %d = add %a, %b : i64\n  ret %c\n",
    );
    assert_eq!(roles_default(&src), [CtRole::Division, CtRole::Division]);
}

#[test]
fn shifts_and_multiplies_follow_the_policy() {
    let src = body(
        "  %a = shl %p, %s : i64\n  %b = lshr %s, %p : i64\n  %c = mul %s, %p : i64\n  %d = ashr %p, i64 3 : i64\n  ret %d\n",
    );
    assert!(roles(&src, CtPolicy::DEFAULT).is_empty());
    // Only the shift *amount* matters for shifts; both multiply operands do.
    assert_eq!(roles(&src, CtPolicy::STRICT), [CtRole::ShiftAmount, CtRole::Multiply]);
}

#[test]
fn floating_point_on_secrets_is_rejected() {
    let src = r#"module "t"
func @f(secret f64, secret i64) -> secret f64 {
entry ^0(%x: f64, %s: i64):
  %n = fneg %x : f64
  %b = bitcast %x : i64
  %c = bitcast %b : f64
  %a = fadd %x, f64 0x3FF0000000000000 : f64
  %k = fcmp olt %x, f64 0x0 : i1
  %u = uitofp %s : f64
  %t = fptosi %x : i32
  ret %n
}
"#;
    assert_eq!(
        roles_default(src),
        [CtRole::FloatOperation, CtRole::FloatOperation, CtRole::FloatOperation, CtRole::FloatOperation]
    );
}

#[test]
fn calls_respect_parameter_secrecy() {
    let src = r#"module "t"
func @takes_secret(secret i64, i64) -> i64

func @takes_public(i64) -> i64

func @va(i64, ...) -> i64

func @f(secret i64, i64, ptr) -> i64 {
entry ^0(%s: i64, %p: i64, %fp: ptr):
  %a = call @takes_secret(%s, %p) : i64
  %b = call @takes_public(%s) : i64
  %c = call @takes_secret(%p, %s) : i64
  %d = call @va(%p, %s) : i64
  %e = call %fp(%s) : i64
  %g = inttoptr %s : ptr
  %h = call %g(%p) : i64
  ret %a
}
"#;
    assert_eq!(
        roles_default(src),
        [
            CtRole::PublicParameter(0),
            CtRole::PublicParameter(1),
            CtRole::VariadicArgument,
            CtRole::IndirectArgument,
            CtRole::CallTarget,
        ]
    );
}

#[test]
fn secret_returns_need_a_secret_signature() {
    let public = body("  %a = add %s, i64 1 : i64\n  ret %a\n");
    assert_eq!(roles_default(&public), [CtRole::PublicReturn]);
    let secret = r#"module "t"
func @f(secret i64) -> secret i64 {
entry ^0(%s: i64):
  %a = add %s, i64 1 : i64
  ret %a
}
"#;
    assert!(roles_default(secret).is_empty());
    // A call to a secret-returning function yields a secret.
    let caller = r#"module "t"
func @k() -> secret i64

func @f() -> i64 {
entry ^0:
  %a = call @k() : i64
  ret %a
}
"#;
    assert_eq!(roles_default(caller), [CtRole::PublicReturn]);
}

#[test]
fn stores_of_secrets_must_target_declared_or_local_memory() {
    let src = r#"module "t"
global @pub : i64 = i64 0

global secret @key : i64 = i64 0

func @f(secret i64, ptr) -> i64 {
entry ^0(%s: i64, %q: ptr):
  %x = alloca i64 : ptr
  store %s, %x align 8 : i64
  store %s, @key align 8 : i64
  store secret %s, %q align 8 : i64
  store %s, %q align 8 : i64
  store %s, @pub align 8 : i64
  store i64 5, %q align 8 : i64
  ret i64 0
}
"#;
    assert_eq!(roles_default(src), [CtRole::PublicStore, CtRole::PublicStore]);
}

#[test]
fn secret_operands_of_variable_time_effects_are_rejected() {
    let src = r#"module "t"
global secret @key : i64 = i64 0

global @ctr : i64 = i64 0

func @f(secret i64, ptr) -> i64 {
entry ^0(%s: i64, %q: ptr):
  %m = dyn_alloca %s align 16 : ptr
  %r = syscall i64 1, %s : i64
  atomic_store seq_cst %s, @ctr align 8 : i64
  %o = atomic_rmw add seq_cst @key, i64 1 align 8 : i64
  %v = atomic_load seq_cst @key align 8 : i64
  ret i64 0
}
"#;
    assert_eq!(
        roles_default(src),
        [
            CtRole::AllocaSize,
            CtRole::SyscallOperand,
            CtRole::AtomicOperand,
            CtRole::AtomicOnSecretMemory,
        ]
    );
}

#[test]
fn a_constant_time_loop_is_accepted() {
    // A public counter drives the loop; the secret only flows through data.
    let src = r#"module "t"
func @f(ptr, ptr, i64) -> secret i64 {
entry ^0(%a: ptr, %b: ptr, %n: i64):
  br ^1(i64 0, i64 0)
^1(%i: i64, %acc: i64):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %pa = ptr_add %a, %i : ptr
  %pb = ptr_add %b, %i : ptr
  %x = load secret %pa align 1 : i8
  %y = load secret %pb align 1 : i8
  %d = xor %x, %y : i8
  %dz = zext %d : i64
  %acc2 = or %acc, %dz : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %acc2)
^3:
  ret %acc
}
"#;
    assert!(roles_default(src).is_empty());
}

#[test]
fn modules_without_secrets_are_trivially_constant_time() {
    let mut syms = StrInterner::new();
    let m = crate::ir::tests::atomics_module(&mut syms);
    assert!(verify_module_ct(&m).is_ok());
    // The secrecy fixture is constant-time as written.
    let s = crate::ir::tests::secret_module(&mut syms);
    assert!(verify_module_ct(&s).is_ok());
}

#[test]
fn vector_code_is_checked_lane_wise() {
    // Taint flows through splats, lane-wise ops and lane moves. A vector
    // select on a secret mask is fine (it lowers to a bitwise blend); a
    // vector division by a secret-derived value, and a branch on an extracted
    // secret lane, are rejected.
    let ok = body(
        "  %v = splat %s : <2 x i64>
  %w = splat %p : <2 x i64>
  %m = icmp ult %v, %w : <2 x i1>
  %b = select %m, %v, %w : <2 x i64>
  %r = reduce add %b : i64
  %d = declassify %r : i64
  ret %d
",
    );
    assert_eq!(roles_default(&ok), []);
    let bad = body(
        "  %v = splat %s : <2 x i64>
  %w = splat %p : <2 x i64>
  %d = udiv %w, %v : <2 x i64>
  %e = extractelement %d, 0 : i64
  %c = icmp eq %e, i64 0 : i1
  cond_br %c, ^1, ^2
^1:
  ret i64 1
^2:
  ret i64 0
",
    );
    let r = roles_default(&bad);
    assert!(r.contains(&CtRole::Division) && r.contains(&CtRole::BranchCondition), "{r:?}");
}
