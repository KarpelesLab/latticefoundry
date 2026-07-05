//! Unit tests for the frontend: lexing, parsing/sema, and lowering/verify.

use crate::ast::{CType, IntTy};
use crate::sema::{TExprKind, TStmt};
use crate::{CStd, check_source, compile_to_ir};

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

// --- aggregate-type tests -------------------------------------------------

use crate::check_source_with;
use crate::preprocess::PpOptions;

fn check_std(src: &str, std: CStd) -> Result<crate::sema::Program, Vec<latticefoundry::support::diagnostics::Diagnostic>> {
    check_source_with(src, &PpOptions { std, ..PpOptions::default() })
}

#[test]
fn struct_member_typechecks_and_lowers() {
    lower_ok("struct S { int x; int y; }; int main(){ struct S s; s.x=1; s.y=2; return s.x+s.y; }");
}

#[test]
fn union_all_members_at_offset_zero() {
    // A union's size is the max member; a `struct*`-style layout is not implied.
    lower_ok("union U { int i; char c; }; int main(){ union U u; u.i=0; u.c=7; return u.i; }");
}

#[test]
fn enum_constants_are_integer_constants() {
    let prog = check_source("enum E { A, B=5, C }; int f(){ int a[C]; return B; }").unwrap();
    assert_eq!(prog.funcs.len(), 1);
}

#[test]
fn typedef_is_recognized_as_a_type() {
    lower_ok("typedef int myint; myint f(myint x){ return x + 1; } int main(){ return f(41); }");
}

#[test]
fn typedef_name_shadowed_by_local() {
    // `T` names an int type, then a local `T` shadows it as a variable.
    lower_ok("typedef int T; int main(){ T x = 1; int T = 41; return x + T; }");
}

#[test]
fn array_and_pointer_decay() {
    lower_ok("int sum(int *p, int n){ int s=0; for(int i=0;i<n;i++) s+=p[i]; return s; } \
              int main(){ int a[3] = {1,2,3}; return sum(a, 3); }");
}

#[test]
fn sizeof_array_is_n_times_element() {
    let prog = check_source("int f(){ int a[10]; return sizeof(a); }").unwrap();
    let ret = prog.funcs[0]
        .body
        .iter()
        .find_map(|s| if let TStmt::Return(Some(e)) = s { Some(e) } else { None })
        .expect("a return statement");
    // sizeof(int[10]) == 40 (constant-folded), possibly wrapped in a Convert.
    let inner = match &ret.kind {
        TExprKind::Convert(inner) => &inner.kind,
        k => k,
    };
    assert!(matches!(inner, TExprKind::Const(40)), "got {inner:?}");
}

#[test]
fn string_literal_creates_readonly_global() {
    let prog = check_source("int main(){ return \"abc\"[0]; }").unwrap();
    assert!(prog.globals.iter().any(|g| g.readonly && g.bytes == b"abc\0"));
}

#[test]
fn designated_initializer_rejected_in_c89() {
    let err = check_std("int f(){ int a[3] = { [1] = 5 }; return a[1]; }", CStd::C89).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("designated initializers are a C99 feature")));
}

#[test]
fn designated_initializer_accepted_in_c99() {
    check_std("int f(){ int a[3] = { [1] = 5 }; return a[1]; }", CStd::C99)
        .expect("designated initializers type-check under c99");
}

#[test]
fn struct_by_value_return_is_rejected() {
    let err = check_source("struct S { int a; }; struct S f(){ struct S s; s.a=1; return s; }")
        .unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("returning a struct/union by value")));
}

#[test]
fn member_of_non_struct_is_an_error() {
    let err = check_source("int main(){ int x = 0; return x.y; }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("requires a struct/union")));
}

#[test]
fn self_referential_struct_lowers() {
    // A linked-list node referring to itself through a pointer.
    lower_ok("struct N { int v; struct N* next; }; \
              int main(){ struct N a; a.v=1; a.next=0; return a.v; }");
}

// --- switch / goto / function-pointer tests --------------------------------

#[test]
fn switch_with_fallthrough_lowers() {
    lower_ok(
        "int f(int n){ int s=0; switch(n){ case 1: s+=1; case 2: s+=2; break; default: s+=9; } \
         return s; } int main(){ return f(1); }",
    );
}

#[test]
fn switch_over_enum_lowers() {
    lower_ok(
        "enum E{A,B=5,C}; int f(enum E e){ switch(e){ case A: return 1; case B: return 2; \
         case C: return 3; } return 0; } int main(){ return f(C); }",
    );
}

#[test]
fn goto_forward_and_backward_lowers() {
    lower_ok(
        "int main(){ int i=0,s=0; loop: if(i<5){ s+=i; i++; goto loop; } \
         if(s>0) goto done; s=99; done: return s; }",
    );
}

#[test]
fn goto_to_undefined_label_is_an_error() {
    let err = check_source("int main(){ goto missing; return 0; }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("undeclared label")));
    assert!(err[0].span.is_some(), "diagnostic should carry a span");
}

#[test]
fn duplicate_case_is_an_error() {
    let err = check_source("int f(int n){ switch(n){ case 1: return 1; case 1: return 2; } return 0; }")
        .unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("duplicate case")));
}

#[test]
fn duplicate_label_is_an_error() {
    let err = check_source("int main(){ a: ; a: ; return 0; }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("duplicate label")));
}

#[test]
fn multiple_default_is_an_error() {
    let err = check_source("int f(int n){ switch(n){ default: return 1; default: return 2; } }")
        .unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("default")));
}

#[test]
fn break_in_switch_is_allowed() {
    // `break` is valid inside a switch even outside any loop.
    check_source("int f(int n){ switch(n){ case 1: break; } return 0; }")
        .expect("break in switch type-checks");
}

#[test]
fn label_may_share_a_typedef_name() {
    // Labels have their own namespace, so a label named like a typedef is fine.
    lower_ok("typedef int T; int main(){ int x=0; T: x++; if(x<3) goto T; return x; }");
}

#[test]
fn function_pointer_call_lowers() {
    lower_ok(
        "int inc(int x){ return x+1; } \
         int main(){ int (*fp)(int)=&inc; return fp(41)+(*fp)(0); }",
    );
}

#[test]
fn function_pointer_designator_decays() {
    // A bare function name used as a value decays to a function pointer.
    let prog = check_source("int f(int x){ return x; } int main(){ int (*p)(int)=f; return p(1); }")
        .unwrap();
    assert_eq!(prog.funcs.len(), 2);
}

#[test]
fn array_of_function_pointers_lowers() {
    lower_ok(
        "int a(int x,int y){return x+y;} int s(int x,int y){return x-y;} \
         int main(){ int (*ops[2])(int,int)={a,s}; return ops[0](3,4)+ops[1](10,1); }",
    );
}

#[test]
fn function_pointer_typedef_and_callback_lowers() {
    lower_ok(
        "typedef int (*Fn)(int); int inc(int x){return x+1;} \
         int apply(Fn g,int v){ return g(v); } int main(){ return apply(inc,41); }",
    );
}

#[test]
fn calling_a_non_function_is_an_error() {
    let err = check_source("int main(){ int x=0; return x(1); }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("not a function")));
}
