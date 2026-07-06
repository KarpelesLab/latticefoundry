//! Unit tests for the frontend: lexing, parsing/sema, and lowering/verify.

use crate::ast::{CType, FloatTy, IntTy};
use crate::sema::{TExprKind, TStmt};
use crate::{CStd, check_source, compile_to_ir};

use latticefoundry::verify::verify_module;

// --- floating-constant lexing ---------------------------------------------

/// Lex a single expression's worth of source and return the first token's kind.
fn first_token(src: &str) -> crate::lex::TokenKind {
    let toks = crate::lex::lex(src, latticefoundry::support::diagnostics::FileId::new(0))
        .expect("should lex");
    toks.into_iter().next().expect("a token").kind
}

#[test]
fn lex_floating_constant_forms() {
    use crate::lex::TokenKind::FloatLit;
    // A `.`, a leading/trailing dot, an exponent, and a signed exponent all lex
    // as double floating constants with the exact value.
    for (src, want) in [("1.5", 1.5), (".5", 0.5), ("3.", 3.0), ("1e10", 1e10), ("2.5e-3", 2.5e-3)]
    {
        match first_token(src) {
            FloatLit(v, ty) => {
                assert_eq!(v, want, "value of {src}");
                assert_eq!(ty, CType::double(), "type of {src}");
            }
            other => panic!("{src} should lex as a float, got {other:?}"),
        }
    }
}

#[test]
fn lex_float_suffix_selects_float_type() {
    use crate::lex::TokenKind::FloatLit;
    // `f`/`F` -> float (rounded to binary32); `l`/`L` -> long double (== double).
    match first_token("0.1f") {
        FloatLit(v, ty) => {
            assert_eq!(ty, CType::Float(FloatTy::F32));
            // 0.1f rounds to binary32, stored back as the exact f64 of that value.
            assert_eq!(v, f64::from(0.1f32));
        }
        other => panic!("got {other:?}"),
    }
    assert!(matches!(first_token("2.0L"), FloatLit(_, ty) if ty == CType::double()));
}

#[test]
fn lex_hex_floating_constant() {
    use crate::lex::TokenKind::FloatLit;
    // 0x1.8p3 = 1.5 * 2^3 = 12.0.
    match first_token("0x1.8p3") {
        FloatLit(v, ty) => {
            assert_eq!(v, 12.0);
            assert_eq!(ty, CType::double());
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn lex_integer_is_not_a_float() {
    use crate::lex::TokenKind::{FloatLit, IntLit};
    // A bare integer, and a hex integer with no `p`, stay integers.
    assert!(matches!(first_token("42"), IntLit(42, _)));
    assert!(matches!(first_token("0xff"), IntLit(255, _)));
    assert!(!matches!(first_token("100"), FloatLit(..)));
}

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

// --- floating-point sema / conversions ------------------------------------

#[test]
fn float_usual_arithmetic_conversions() {
    // int + double -> double; float + int -> float; float + double -> double.
    assert_eq!(ret_type("1 + 2.0"), CType::double());
    assert_eq!(ret_type("1 + 2.0f"), CType::float());
    assert_eq!(ret_type("2.0f + 3.0"), CType::double());
    assert_eq!(ret_type("3.5 * 2"), CType::double());
}

#[test]
fn float_comparison_is_int() {
    // A floating comparison yields `int` 0/1.
    assert_eq!(ret_type("1.5 < 2.5"), CType::int());
    assert_eq!(ret_type("1.5 == 1.5"), CType::int());
}

#[test]
fn modulo_on_double_is_rejected() {
    let err = check_source("int f(){ double a=1.0, b=2.0; return (int)(a % b); }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("floating-point operands are not allowed")));
}

#[test]
fn double_array_size_is_rejected() {
    let err = check_source("int f(){ double a[1.5]; return (int)a[0]; }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("constant integer expression")));
}

#[test]
fn pointer_to_float_cast_is_rejected() {
    let err = check_source("int f(){ int x=0; int*p=&x; return (int)(double)p; }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("pointer and a floating-point type")));
}

#[test]
fn float_conversions_lower_and_verify() {
    // int->double, double->int (truncation), float<->double, and a float in a
    // boolean context all lower to well-typed IR.
    lower_ok(
        "int main(){ int n=7; double r = n/2.0; float f=0.1f; double d=f; \
         if(d) return (int)(r*10) + (int)3.9; return 0; }",
    );
}

#[test]
fn float_function_args_and_return_lower() {
    lower_ok(
        "double add3(double a,double b,double c){ return a+b+c; } \
         double mix(int a,double b){ return a+b; } \
         int main(){ double x=1.0,y=2.0,z=3.0; return (int)add3(x,y,z) + (int)mix(4,5.0); }",
    );
}

#[test]
fn float_global_initializer_bytes() {
    // A double global's initializer is materialized to its IEEE little-endian
    // image (2.5 == 0x4004000000000000).
    let prog = check_source("double g = 2.5; int main(){ return (int)g; }").unwrap();
    let g = prog.globals.iter().find(|g| g.name == "g").expect("global g");
    assert_eq!(g.bytes, 2.5f64.to_le_bytes());
    assert_eq!(g.ty, CType::double());
}

#[test]
fn float_sizeof() {
    assert!(matches!(
        &check_source("int f(){ return sizeof(float); }").unwrap().funcs[0].body[0],
        TStmt::Return(Some(e)) if matches!(unwrap_convert(&e.kind), TExprKind::Const(4)),
    ));
    assert!(matches!(
        &check_source("int f(){ return sizeof(double); }").unwrap().funcs[0].body[0],
        TStmt::Return(Some(e)) if matches!(unwrap_convert(&e.kind), TExprKind::Const(8)),
    ));
}

/// Unwrap a leading implicit `Convert` (e.g. `sizeof` → return type) node.
fn unwrap_convert(k: &TExprKind) -> &TExprKind {
    match k {
        TExprKind::Convert(inner) => &inner.kind,
        other => other,
    }
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
fn struct_by_value_return_and_pass_lowers() {
    // Returning and passing a struct by value is supported (a by-reference +
    // `sret` ABI at lowering); both directions type-check and lower.
    lower_ok(
        "struct P { int x, y; }; \
         struct P mk(int a, int b){ struct P p; p.x=a; p.y=b; return p; } \
         int sum(struct P p){ return p.x + p.y; } \
         int main(){ struct P q = mk(3, 4); return sum(q); }",
    );
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

// --- C11/C23 language features --------------------------------------------

/// The (implicit-`Convert`-unwrapped) kind of a function's first `return` value.
fn return_kind(prog: &crate::sema::Program) -> TExprKind {
    let ret = prog.funcs[0]
        .body
        .iter()
        .find_map(|s| if let TStmt::Return(Some(e)) = s { Some(e) } else { None })
        .expect("a return statement");
    unwrap_convert(&ret.kind).clone()
}

#[test]
fn static_assert_true_compiles_false_diagnoses() {
    check_source("_Static_assert(sizeof(int)==4, \"int is 4\"); int main(){ return 0; }")
        .expect("a true _Static_assert compiles");
    let err = check_source("_Static_assert(sizeof(int)==8, \"int must be 8\"); int main(){ return 0; }")
        .unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("static assertion failed: int must be 8")));
}

#[test]
fn static_assert_c23_allows_no_message() {
    check_std("int main(){ static_assert(1); return 0; }", CStd::C23)
        .expect("C23 static_assert may omit the message");
    // A block-scope false assertion is still diagnosed.
    let err = check_std("int main(){ static_assert(0); return 0; }", CStd::C23).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("static assertion failed")));
}

#[test]
fn generic_selects_expression_by_controlling_type() {
    // A controlling `int` selects the `int` association's value (7).
    let prog = check_source("int f(){ int x=0; return _Generic((x), int: 7, double: 9, default: 0); }")
        .unwrap();
    assert!(matches!(return_kind(&prog), TExprKind::Const(7)));
    // A controlling `double` selects the `double` association (9).
    let prog = check_source("int f(){ double x=0; return _Generic((x), int: 7, double: 9, default: 0); }")
        .unwrap();
    assert!(matches!(return_kind(&prog), TExprKind::Const(9)));
    // No exact match falls to `default`.
    let prog = check_source("int f(){ char*p=0; return _Generic((p), int: 7, double: 9, default: 5); }")
        .unwrap();
    assert!(matches!(return_kind(&prog), TExprKind::Const(5)));
}

#[test]
fn generic_without_match_or_default_is_an_error() {
    let err = check_source("int f(){ int x=0; return _Generic((x), double: 1, char*: 2); }")
        .unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("no _Generic association matches")));
}

#[test]
fn generic_duplicate_association_is_an_error() {
    let err = check_source("int f(){ int x=0; return _Generic((x), int: 1, int: 2); }").unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("two associations for type")));
}

#[test]
fn generic_is_rejected_before_c11() {
    let err = check_std("int f(){ int x=0; return _Generic((x), int: 1, default: 0); }", CStd::C99)
        .unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("`_Generic` is a C11 feature")));
}

#[test]
fn alignof_yields_type_alignment() {
    let prog = check_source("int f(){ return _Alignof(double); }").unwrap();
    assert!(matches!(return_kind(&prog), TExprKind::Const(8)));
    // A struct's alignment is its widest member's alignment (int -> 4).
    let prog = check_source("struct S{ char c; int x; }; int f(){ return _Alignof(struct S); }").unwrap();
    assert!(matches!(return_kind(&prog), TExprKind::Const(4)));
}

#[test]
fn alignof_alignas_keywords_are_c23() {
    check_std("int f(){ return (int)alignof(int); }", CStd::C23).expect("alignof keyword under C23");
    assert!(check_std("int f(){ return (int)alignof(int); }", CStd::C11).is_err());
    check_std("int f(){ alignas(16) int x = 0; return x; }", CStd::C23).expect("alignas keyword under C23");
}

#[test]
fn alignas_sets_local_alignment() {
    // `_Alignas` is available under C11; it over-aligns the object's storage.
    let prog = check_source("int f(){ _Alignas(16) int x = 3; return x; }").unwrap();
    assert!(prog.funcs[0].locals.iter().any(|l| l.name == "x" && l.align == Some(16)));
    // `_Alignas(type)` uses that type's alignment.
    let prog = check_source("int f(){ _Alignas(double) int x = 3; return x; }").unwrap();
    assert!(prog.funcs[0].locals.iter().any(|l| l.name == "x" && l.align == Some(8)));
}

#[test]
fn typeof_declares_a_variable_of_the_operand_type() {
    // `typeof(x)` (x a `long`) declares `y` as `long`.
    let prog = check_source("int f(){ long x=5; typeof(x) y=10; return (int)(x+y); }").unwrap();
    assert!(prog.funcs[0].locals.iter().any(|l| l.name == "y"
        && matches!(l.ty, CType::Int(IntTy { width: 64, signed: true }))));
    // `sizeof(typeof(expr))` is the operand type's size.
    let prog = check_source("int f(){ double d=0; return sizeof(typeof(d)); }").unwrap();
    assert!(matches!(return_kind(&prog), TExprKind::Const(8)));
}

#[test]
fn typeof_keyword_is_rejected_in_strict_c11() {
    // Under strict ISO C11 `typeof` is an ordinary identifier (not a keyword).
    assert!(check_std("int f(){ int x=5; typeof(x) y=x; return y; }", CStd::C11).is_err());
}

#[test]
fn compound_literal_lvalue_index_and_argument_lower() {
    // As an argument (address taken), indexed, and used as an lvalue.
    lower_ok(
        "struct P{int x,y;}; int use(struct P*p){ return p->x*p->y; } \
         int main(){ int i=2; return use(&(struct P){6,7}) + (int[]){1,2,3,4,5}[i]; }",
    );
    // Modifying a compound literal through its lvalue.
    lower_ok("int main(){ struct Q{int a,b;} *q = &(struct Q){2,3}; q->a += 5; return q->a; }");
}

#[test]
fn compound_literal_is_rejected_before_c99() {
    let err = check_std("int f(){ return (int[]){1,2,3}[0]; }", CStd::C89).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("compound literals are a C99 feature")));
}

#[test]
fn anonymous_member_access_resolves_and_lowers() {
    // A member of an anonymous union is accessed as a member of the enclosing.
    lower_ok(
        "struct S{ int a; union { int u; unsigned char b[4]; }; }; \
         int main(){ struct S s; s.a=1; s.u=0; s.b[0]=2; return s.a+s.u; }",
    );
    // Offsets compose through nested anonymous members.
    let prog = check_source(
        "struct S{ union { long all; struct { int lo, hi; }; }; }; \
         int main(){ struct S s; s.hi = 1; return s.hi; }",
    )
    .unwrap();
    assert_eq!(prog.funcs.len(), 1);
}

#[test]
fn anonymous_members_are_rejected_before_c11() {
    let err = check_std(
        "struct S{ int a; struct { int b; }; }; int f(){ struct S s; s.b=1; return s.b; }",
        CStd::C99,
    )
    .unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("anonymous struct/union members are a C11 feature")));
}

#[test]
fn attributes_are_ignored_under_c23() {
    check_std(
        "[[nodiscard]] int f(void){ return 1; } \
         int main(){ [[maybe_unused]] int x = 0; int y = 1; \
         switch(y){ case 1: y = f(); [[fallthrough]]; case 2: break; } return x + y; }",
        CStd::C23,
    )
    .expect("standard attributes are accepted and ignored under C23");
}

#[test]
fn attributes_are_rejected_before_c23() {
    let err = check_std("int main(){ [[maybe_unused]] int x = 0; return x; }", CStd::C11).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("C23")));
}

#[test]
fn noreturn_specifier_is_accepted() {
    // `_Noreturn` (C11) and the `noreturn` keyword (C23) are accepted on a
    // function declaration/definition.
    check_source("_Noreturn void die(void); void die(void){ for(;;){} } int main(){ return 0; }")
        .expect("_Noreturn is accepted");
    check_std(
        "noreturn void die(void); void die(void){ for(;;){} } int main(){ return 0; }",
        CStd::C23,
    )
    .expect("the noreturn keyword is accepted under C23");
}
