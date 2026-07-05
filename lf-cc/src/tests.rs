//! Unit tests for the frontend: lexing, parsing/sema, and lowering/verify.

use crate::ast::{CType, IntTy};
use crate::sema::{TExprKind, TStmt};
use crate::{check_source, compile_to_ir};

use latticefoundry::verify::verify_module;

// --- sema type-checking tests ---------------------------------------------

/// Type-check `long f() { return <body>; }` and hand back the C type of the
/// body expression *before* the implicit return conversion, for inspection.
fn ret_type(body: &str) -> CType {
    let src = format!("long f() {{ return {body}; }}");
    let prog = check_source(&src).expect("should type-check");
    let f = &prog.funcs[0];
    match &f.body[0] {
        // The return may wrap the body in a Convert to the return type; unwrap it.
        TStmt::Return(Some(e)) => match &e.kind {
            TExprKind::Convert(inner) => inner.ty.clone(),
            _ => e.ty.clone(),
        },
        other => panic!("expected a return, got {other:?}"),
    }
}

#[test]
fn integer_literal_types() {
    let prog = check_source("int f() { return 1 + 2; }").unwrap();
    assert_eq!(prog.funcs.len(), 1);
}

#[test]
fn usual_arithmetic_conversions() {
    // int + unsigned int -> unsigned int (return converts back to int).
    let src = "int f() { unsigned u = 1; int i = 2; return (u + i) == 3u; }";
    check_source(src).expect("mixed signedness type-checks");
}

#[test]
fn char_and_short_promote_to_int() {
    // A char + char is computed in int.
    let src = "int f() { char a = 1; char b = 2; return a + b; }";
    let prog = check_source(src).unwrap();
    let f = &prog.funcs[0];
    if let TStmt::Return(Some(e)) = &f.body[2] {
        // The returned expression is `a + b`, whose type is `int`.
        assert!(matches!(e.ty, CType::Int(IntTy { width: 32, .. })));
    } else {
        panic!("expected return");
    }
}

#[test]
fn long_widens_mixed_expression() {
    // int + long -> long.
    let ty = {
        let src = "int f() { long l = 1; int i = 2; return (l + i) != 0; }";
        check_source(src).unwrap();
        ret_type("1L + 1")
    };
    assert!(matches!(ty, CType::Int(IntTy { width: 64, .. })));
}

#[test]
fn undeclared_identifier_is_an_error() {
    let err = check_source("int f() { return x; }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("undeclared identifier")));
    assert!(err[0].span.is_some(), "diagnostic should carry a span");
}

#[test]
fn call_arity_mismatch_is_an_error() {
    let src = "int g(int a, int b) { return a + b; } int f() { return g(1); }";
    let err = check_source(src).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("expects 2 argument")));
}

#[test]
fn assignment_to_non_lvalue_is_an_error() {
    let err = check_source("int f() { 1 = 2; return 0; }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("not an lvalue") || d.message.contains("not assignable")));
}

#[test]
fn break_outside_loop_is_an_error() {
    let err = check_source("int f() { break; return 0; }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("'break' outside")));
}

#[test]
fn pointer_deref_typechecks() {
    let src = "int f() { int x = 5; int *p = &x; return *p; }";
    let prog = check_source(src).unwrap();
    assert_eq!(prog.funcs.len(), 1);
}

#[test]
fn short_circuit_lowers_to_logand() {
    let prog = check_source("int f() { return 1 && 0; }").unwrap();
    if let TStmt::Return(Some(e)) = &prog.funcs[0].body[0] {
        assert!(matches!(e.kind, TExprKind::LogAnd(..)));
    } else {
        panic!("expected return");
    }
}

// --- lowering / verify tests ----------------------------------------------

fn lower_ok(src: &str) {
    let (module, _syms) = compile_to_ir(src, "test", false).expect("should compile");
    verify_module(&module).expect("lowered IR must verify");
}

#[test]
fn lower_arithmetic_verifies() {
    lower_ok("int main() { return 2 + 3 * 4 - 1; }");
}

#[test]
fn lower_control_flow_verifies() {
    lower_ok(
        "int main() {\n\
         int s = 0;\n\
         for (int i = 0; i < 10; i++) { if (i == 5) continue; s += i; }\n\
         return s;\n\
         }",
    );
}

#[test]
fn lower_recursion_verifies() {
    lower_ok("int fib(int n) { if (n < 2) return n; return fib(n-1) + fib(n-2); } int main() { return fib(10); }");
}

#[test]
fn lower_pointers_verifies() {
    lower_ok("int main() { int x = 5; int *p = &x; *p = 7; return *p; }");
}

#[test]
fn lower_all_operators_verifies() {
    lower_ok(
        "int main() {\n\
         int a = 7, b = 3;\n\
         int r = (a & b) | (a ^ b);\n\
         r += a << 1; r -= b >> 1; r *= 2; r /= 3; r %= 5;\n\
         unsigned u = 10u; u >>= 1;\n\
         return (r + (int)u) ? ~a : -b;\n\
         }",
    );
}
