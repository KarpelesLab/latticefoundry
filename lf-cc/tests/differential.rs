//! End-to-end differential tests: compile a suite of C programs with `lf-cc`
//! (at -O0 and -O2), run each native binary, and assert its process exit code
//! matches the same program compiled with `gcc -O0`.
//!
//! Every program keeps its result in `0..256` so the process exit code is an
//! unambiguous observation. If `gcc` is not installed the gcc comparison is
//! skipped, but the -O0-vs-O2 self-consistency check still runs.

use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::link::write_executable;
use latticefoundry::transform::pipeline::OptLevel;
use lf_cc::{CStd, PpOptions};

/// The corpus: `(name, source)`. Each `main` returns a value in `0..256`.
fn programs() -> Vec<(&'static str, &'static str)> {
    vec![
        // Arithmetic + precedence.
        ("precedence", "int main(){ return 2 + 3 * 4 - 10 / 2 + 7 % 4; }"),
        ("paren", "int main(){ return (2 + 3) * (4 - 1) + 1; }"),
        ("bitwise", "int main(){ int a=0xF0, b=0x3C; return (a & b) | (a ^ b); }"),
        ("shifts", "int main(){ return (1 << 6) + (255 >> 2); }"),
        ("unary", "int main(){ int a=5; return -(-a) + ~(-3) + !0 + !5; }"),
        // Signed vs unsigned division / comparison / shift.
        ("sdiv", "int main(){ int a=-100, b=7; return (a / b) + 100; }"),
        ("udiv", "int main(){ unsigned a=200u, b=7u; return a / b; }"),
        ("srem", "int main(){ int a=-17, b=5; return a % b + 100; }"),
        ("ucmp", "int main(){ unsigned x=1; int y=-1; return (x < (unsigned)y) + 40; }"),
        ("scmp", "int main(){ int x=-1, y=1; return (x < y) + 40; }"),
        ("ashr", "int main(){ int x=-64; return (x >> 2) + 100; }"),
        ("lshr", "int main(){ unsigned x=250u; return x >> 1; }"),
        // Control flow.
        ("if_else", "int main(){ int x=7; if (x > 5) return 20; else return 10; }"),
        ("while_sum", "int main(){ int i=0,s=0; while (i<10){ s+=i; i++; } return s; }"),
        ("do_while", "int main(){ int i=0,s=0; do { s+=i; i++; } while(i<5); return s; }"),
        ("for_sum", "int main(){ int s=0; for (int i=1;i<=10;i++) s+=i; return s; }"),
        ("nested_loops", "int main(){ int c=0; for(int i=0;i<5;i++) for(int j=0;j<5;j++) c++; return c; }"),
        ("break_continue", "int main(){ int s=0; for(int i=0;i<20;i++){ if(i==10) break; if(i%2) continue; s+=i; } return s; }"),
        // Recursion + mutual recursion.
        ("factorial", "int fact(int n){ return n<=1?1:n*fact(n-1); } int main(){ return fact(5); }"),
        ("fib", "int fib(int n){ if(n<2) return n; return fib(n-1)+fib(n-2); } int main(){ return fib(11); }"),
        (
            "even_odd",
            "int is_even(int n); int is_odd(int n){ if(n==0) return 0; return is_even(n-1); } \
             int is_even(int n){ if(n==0) return 1; return is_odd(n-1); } \
             int main(){ return is_even(17) + is_odd(17)*2; }",
        ),
        // --- old-style (K&R) function definitions ---------------------------
        // A K&R definition whose parameters are typed by a following
        // declaration-list (`int a; int b;`), called through a prototype.
        (
            "knr_add",
            "int add(a, b) int a; int b; { return a + b; } \
             int main(){ return add(19, 23); }",
        ),
        // `register` on a K&R parameter declaration is accepted and ignored.
        // (The default-to-`int` rule for an unlisted parameter is covered by a
        // separate lf-cc unit test; modern gcc rejects implicit int.)
        (
            "knr_register_param",
            "int f(a, b) register int a; register int b; { return a + b; } \
             int main(){ return f(40, 2); }",
        ),
        // A K&R definition with a pointer parameter and a multi-name declaration.
        (
            "knr_ptr_param",
            "int sum3(p, n) int *p; int n; \
             { int s = 0, i; for (i = 0; i < n; i++) s += p[i]; return s; } \
             int main(){ int a[3]; a[0]=10; a[1]=14; a[2]=18; return sum3(a, 3); }",
        ),
        // A K&R definition with two names declared in one declaration.
        (
            "knr_two_names_one_decl",
            "int g(x, y) int x, y; { return x - y; } \
             int main(){ return g(50, 8); }",
        ),
        // A K&R definition returning a non-int type, with a `char` parameter
        // (default-argument-promoted at the call, truncated by the callee).
        (
            "knr_char_param",
            "long scale(c, k) char c; long k; { return (long)c * k; } \
             int main(){ return (int)scale('\\7', 6); }",
        ),
        // Pointer to a local.
        ("ptr_local", "int main(){ int x=5, *p=&x; *p=7; return *p; }"),
        ("ptr_swap", "void swap(int*a,int*b){ int t=*a; *a=*b; *b=t; } int main(){ int x=3,y=100; swap(&x,&y); return y; }"),
        ("ptr_arith", "int main(){ int x=10; int*p=&x; *(p+0)+=90; return x; }"),
        // Casts + promotions.
        ("cast_trunc", "int main(){ long l=300; char c=(char)l; return (int)(unsigned char)c; }"),
        ("char_promote", "int main(){ char a=100, b=100; return (a+b) > 127 ? 42 : 0; }"),
        ("mixed_width", "int main(){ long a=1000000; int b=7; return (int)(a % b) + 40; }"),
        // Short-circuit with observable RHS effect.
        (
            "sc_and",
            "int g; int side(){ g=1; return 1; } int main(){ g=0; int r = (0 && side()); return r + g; }",
        ),
        (
            "sc_or",
            "int g; int side(){ g=5; return 1; } int main(){ g=0; int r = (1 || side()); return r + g + 40; }",
        ),
        (
            "sc_eval",
            "int g; int inc(){ g++; return 1; } int main(){ g=0; if (inc() && inc() && inc()) {} return g; }",
        ),
        // Ternary.
        ("ternary", "int main(){ int a=5; return a>3 ? (a<10?42:1) : 7; }"),
        // Compound assignment.
        ("compound", "int main(){ int x=100; x-=10; x/=3; x%=7; x<<=2; x|=1; return x; }"),
        // ++ / --.
        ("incdec", "int main(){ int i=5; int a=i++ + ++i; return a + i; }"),
        ("post_pre", "int main(){ int i=10; int j = i-- - --i; return j + i; }"),
        // sizeof.
        ("sizeof", "int main(){ return sizeof(int) + sizeof(char)*4 + sizeof(long)*2; }"),
        // Globals.
        ("global_counter", "int counter = 40; int bump(){ counter += 1; return counter; } int main(){ bump(); bump(); return counter; }"),
        // A `static` block-scope object keeps its value across calls (static
        // storage duration), initialized once.
        (
            "static_local",
            "int next(void){ static int n = 40; n++; return n; } \
             int main(){ next(); next(); return next(); }",
        ),
        // A tentative definition followed by an initialized definition of the same
        // object: they denote one object with the initialized value.
        (
            "tentative_then_def",
            "int g; int g = 41; int main(){ return g + 1; }",
        ),
        // A `static` file-scope object initialized once, plus a same-name `static`
        // local in another function (distinct objects — no linkage clash).
        (
            "static_file_and_local",
            "static int s = 20; int addfile(int x){ return s + x; } \
             int tick(void){ static int s = 1; s++; return s; } \
             int main(){ tick(); return addfile(tick()) + 18; }",
        ),
        // A global pointer array initialized with string-literal addresses and a
        // function-pointer global initialized with a function's address
        // (data relocations); an octal escape carries a byte value.
        (
            "global_relocations",
            "char *names[] = { \"ab\", \"cde\", 0 }; \
             int sq(int x){ return x*x; } int (*fp)(int) = sq; \
             char oct = '\\52'; \
             int main(){ int n=0; char **p=names; while(*p){ n += (int)(*p)[0]; p++; } \
             return n + fp(3) - 187 + (int)oct; }",
        ),
        // A string literal with a high-byte octal escape is byte-exact (not
        // UTF-8-encoded): the two bytes 0x1f and 0x8b, read back individually.
        (
            "string_high_byte_escape",
            "char magic[] = \"\\037\\213\"; \
             int main(){ return (unsigned char)magic[0] + (unsigned char)magic[1] - 100; }",
        ),
        // A larger mixed program.
        (
            "mixed",
            "int gcd(int a,int b){ while(b){ int t=b; b=a%b; a=t; } return a; } \
             int main(){ int acc=0; for(int i=1;i<=12;i++) acc += gcd(i,12); return acc; }",
        ),
        // Preprocessor: object- and function-like macros, `##`, conditionals, and
        // a predefined macro. These use the default dialect (gnu17), which also
        // matches gcc's default, so the same corpus compares cleanly.
        ("pp_object_macro", "#define N 21\nint main(){ return N + N; }"),
        ("pp_func_macro", "#define MAX(a,b) ((a)>(b)?(a):(b))\nint main(){ return MAX(40,42); }"),
        ("pp_paste", "#define CAT(a,b) a##b\nint CAT(v,al)=34;\nint main(){ return val + 8; }"),
        (
            "pp_conditional",
            "#define MODE 2\n#if MODE==1\nint main(){return 1;}\n\
             #elif MODE==2\nint main(){return 42;}\n#else\nint main(){return 0;}\n#endif",
        ),
        (
            "pp_ifdef",
            "#define ON\n#ifdef ON\nint main(){return 42;}\n#else\nint main(){return 0;}\n#endif",
        ),
        (
            "pp_nested_macro",
            "#define SQ(x) ((x)*(x))\n#define SUMSQ(a,b) (SQ(a)+SQ(b))\n\
             int main(){ return SUMSQ(3,4) - 3; }",
        ),
        // --- aggregate types (enum / struct / union / typedef / arrays) -----
        // enum with auto and explicit values, used in arithmetic and as an
        // array size, plus sizeof of the array.
        (
            "enum_vals",
            "enum E{A,B=10,C,D=C+5}; \
             int main(){ int t[D]; return A+B+C+D+(int)(sizeof(t)/sizeof(t[0])); }",
        ),
        // A struct read/written through a local and through a `struct*`.
        (
            "struct_ptr",
            "struct P{int x,y;}; int viaptr(struct P*p){return p->x*p->y;} \
             int main(){ struct P s; s.x=6; s.y=7; struct P*p=&s; p->x+=0; return viaptr(&s); }",
        ),
        // The classic linked list: nodes as stack locals, traversed to sum.
        (
            "linked_list",
            "struct N{int v; struct N* next;}; \
             int main(){ struct N c={3,0}; struct N b={2,&c}; struct N a={1,&b}; \
             int s=0; for(struct N*p=&a;p;p=p->next) s+=p->v; return s*7; }",
        ),
        // Union aliasing: write bytes, read the (little-endian) integer.
        (
            "union_alias",
            "union U{int i; unsigned char b[4];}; \
             int main(){ union U u; u.i=0; u.b[0]=200; u.b[1]=1; return u.i; }",
        ),
        // typedef of a struct and of a base type, with a typedef-name shadowed
        // by a local variable in an inner scope.
        (
            "typedef_shadow",
            "typedef struct{int a,b;} Pair; typedef unsigned long U; \
             int main(){ Pair p={20,22}; U T=100; { int Pair=5; return p.a+p.b+(int)T-Pair-95; } }",
        ),
        // An array decaying to a pointer passed to a summing function.
        (
            "array_decay",
            "int sum(int*a,int n){int s=0;for(int i=0;i<n;i++)s+=a[i];return s;} \
             int main(){ int a[6]={1,2,3,4,5,6}; return sum(a,6)+(int)(sizeof(a)/sizeof(a[0])); }",
        ),
        // A 2-D array indexed in a nested loop.
        (
            "array_2d",
            "int main(){ int m[3][3]; \
             for(int i=0;i<3;i++)for(int j=0;j<3;j++)m[i][j]=(i+1)*(j+1); \
             int s=0; for(int i=0;i<3;i++)for(int j=0;j<3;j++)s+=m[i][j]; return s; }",
        ),
        // Aggregate initializer with omitted (zero-filled) trailing elements.
        (
            "agg_init",
            "struct S{int a,b,c;}; \
             int main(){ struct S s={1,2}; int a[5]={9,8}; \
             return s.a+s.b+s.c+a[0]+a[1]+a[2]+a[3]+a[4]+22; }",
        ),
        // Designated initializers (a C99 feature; the default gnu17 enables it).
        (
            "desig_init",
            "struct S{int x,y,z;}; \
             int main(){ struct S s={.z=3,.x=1}; int a[6]={[5]=6,[0]=10}; \
             return s.x+s.y+s.z+a[0]+a[5]+22; }",
        ),
        // A char[] initialized from a string literal, plus indexing a literal.
        (
            "string_literal",
            "int main(){ char s[]=\"hello\"; int n=0; for(int i=0;s[i];i++)n++; \
             return n + (\"abc\"[1]=='b') + 36; }",
        ),
        // Taking the address of a struct member and modifying through it.
        (
            "addr_field",
            "struct S{int x,y;}; \
             int main(){ struct S s; s.x=1; s.y=2; int*p=&s.y; *p=40; return s.x+s.y; }",
        ),
        // Whole-struct assignment (memberwise copy).
        (
            "struct_copy",
            "struct S{int a,b,c;}; \
             int main(){ struct S x={1,2,3}; struct S y; y=x; y.b=40; return y.a+y.b+y.c-x.b; }",
        ),
        // --- switch / case / default / fall-through -------------------------
        // A Duff-ish accumulation relying on fall-through between cases, with a
        // `break` before the `default` arm.
        (
            "switch_fallthrough",
            "int f(int n){ int s=0; switch(n){ \
             case 5: s+=5; case 4: s+=4; case 3: s+=3; case 2: s+=2; case 1: s+=1; break; \
             default: s+=100; } \
             return s; } \
             int main(){ return f(3)+f(1)+f(5)+f(0)+f(2); }",
        ),
        // A switch where every arm breaks (no fall-through), plus default.
        (
            "switch_break",
            "int classify(int x){ switch(x){ \
             case 0: return 100; case 1: case 2: return 20; case 3: return 30; default: return 7; } } \
             int main(){ return classify(0)+classify(1)+classify(2)+classify(3)+classify(9); }",
        ),
        // A switch over an enum (enumerators as case constants).
        (
            "switch_enum",
            "enum Op{ADD,SUB=10,MUL,DIV=20}; \
             int ev(enum Op o,int a,int b){ switch(o){ \
             case ADD: return a+b; case SUB: return a-b; case MUL: return a*b; case DIV: return a/b; } \
             return 0; } \
             int main(){ return ev(ADD,3,4)+ev(SUB,10,3)+ev(MUL,5,6)+ev(DIV,40,8); }",
        ),
        // --- goto -----------------------------------------------------------
        // A backward goto forming a loop.
        (
            "goto_loop",
            "int main(){ int i=0,s=0; \
             again: if(i<=10){ s+=i; i++; goto again; } return s; }",
        ),
        // A forward `goto cleanup` error-handling idiom skipping code.
        (
            "goto_cleanup",
            "int main(){ int acc=0; int fail=1; \
             acc+=10; if(fail) goto cleanup; acc+=1000; \
             cleanup: acc+=5; return acc; }",
        ),
        // A loop nested inside a switch: `continue` targets the loop, `break`
        // targets the switch (each hit exactly once here).
        (
            "switch_loop_break_continue",
            "int f(int sel){ int s=0; switch(sel){ \
             case 1: for(int i=0;i<10;i++){ if(i==3) continue; if(i==7) break; s+=i; } break; \
             default: s=999; } return s; } \
             int main(){ return f(1); }",
        ),
        // --- function pointers ----------------------------------------------
        // Assign `&f`, then call via `fp(x)` and `(*fp)(x)`: same result.
        (
            "fnptr_basic",
            "int dbl(int x){ return x*2; } \
             int main(){ int (*fp)(int)=&dbl; return fp(20)+(*fp)(1); }",
        ),
        // An array of function pointers used as a dispatch table.
        (
            "fnptr_table",
            "int add(int a,int b){return a+b;} int sub(int a,int b){return a-b;} \
             int mul(int a,int b){return a*b;} int dvd(int a,int b){return a/b;} \
             int main(){ int (*ops[4])(int,int)={add,sub,mul,dvd}; \
             int r=0; for(int i=0;i<4;i++) r+=ops[i](12,3); return r; }",
        ),
        // A function-pointer typedef and a callback parameter.
        (
            "fnptr_callback",
            "typedef int (*IntFn)(int); \
             int inc(int x){return x+1;} int neg(int x){return -x;} \
             int apply(IntFn g,int v){ return g(v); } \
             int main(){ IntFn f=inc; return apply(f,41)+apply(neg,-1)+(apply(inc,0)); }",
        ),
        // --- floating point (float / double) --------------------------------
        // The IEEE operations below are exact, so lf-cc and gcc must agree
        // bit-for-bit; each `main` returns an integral value in 0..256. Function
        // arguments are passed through named variables (the natural C form).
        //
        // `double` arithmetic honoring precedence: 3.5*4 + 10/2.5 = 14 + 4 = 18.
        ("dbl_arith", "int main(){ return (int)(3.5*4.0 + 10.0/2.5); }"),
        // `float` arithmetic: (1.5f + 2.5f) * 10 = 40.
        ("flt_arith", "int main(){ float a=1.5f, b=2.5f; return (int)((a+b)*10.0f); }"),
        // Mixed int/double usual arithmetic conversions: 7/2.0 = 3.5, *10 = 35.
        ("mix_int_double", "int main(){ int n=7; double r = n/2.0; return (int)(r*10); }"),
        // Mixed float/int: (float)5 + 3 computed in float, *4 = 32.
        ("mix_float_int", "int main(){ float f=5.0f; int n=3; return (int)((f+n)*4.0f); }"),
        // A double `max` driving a branch (an ordered, NaN-free comparison).
        (
            "dbl_max_branch",
            "double dmax(double a,double b){ if(a>b) return a; else return b; } \
             int main(){ double x=3.25, y=9.5; return (int)dmax(x,y); }",
        ),
        // A chain of ordered comparisons yielding int 0/1 results.
        (
            "dbl_compare",
            "int main(){ double a=2.5, b=7.5; \
             return (a<b) + (b>a)*2 + (a<=a)*4 + (b>=b)*8 + (a!=b)*16 + (a==a)*32; }",
        ),
        // Conversions both ways: (int)3.9 == 3 truncates toward zero; (double)7 in
        // arithmetic; total = 3 + 35 = 38.
        (
            "dbl_conversions",
            "int main(){ int a=(int)3.9; double d=(double)7; return a + (int)(d/2.0*10); }",
        ),
        // float→double widening: 0.1f promoted to double keeps the binary32
        // rounding, so (int)(d*1000) == 100 on both compilers.
        ("float_to_double", "int main(){ float f=0.1f; double d=f; return (int)(d*1000); }"),
        // A function taking several doubles (exercises xmm0..xmm5); sum = 21.
        (
            "many_doubles",
            "double s6(double a,double b,double c,double d,double e,double f){ \
             return a+b+c+d+e+f; } \
             int main(){ double a=1.0,b=2.0,c=3.0,d=4.0,e=5.0,f=6.0; \
             return (int)s6(a,b,c,d,e,f); }",
        ),
        // Mixed int and double parameters (the split integer/xmm ABI): the ints go
        // in rdi/rdx, the doubles in xmm0/xmm1; 1 + 2.5 + 3 + 4.0 = 10.5 -> 10.
        (
            "mixed_abi",
            "double mix(int a,double b,int c,double d){ return a+b+c+d; } \
             int main(){ int p=1,q=3; double b=2.5,d=4.0; return (int)mix(p,b,q,d); }",
        ),
        // A float in a boolean/controlling context: 0.0 is false.
        ("dbl_bool_if", "int main(){ double x=0.0; if(x) return 1; else return 2; }"),
        // `!x`, `x && y`, `x || y` on doubles.
        (
            "dbl_bool_ops",
            "int main(){ double x=0.0, y=3.0; \
             return (!x)*1 + (!y)*10 + (x&&y ? 4 : 0) + (x||y ? 8 : 0) + (y && y ? 16 : 0); }",
        ),
        // A float used as a `while` condition, counting it down to zero.
        (
            "dbl_while",
            "int main(){ double x=5.0; int n=0; while(x){ x-=1.0; n++; } return n*8; }",
        ),
        // A double global with an initializer, read and used: 2.5 * 16 = 40.
        ("dbl_global", "double g = 2.5; int main(){ return (int)(g*16.0); }"),
        // A float global with an initializer and a constant-expression init.
        (
            "flt_global",
            "float g = 0.25f + 0.25f; int main(){ return (int)(g*84.0f); }",
        ),
        // sizeof(float)==4 && sizeof(double)==8.
        (
            "flt_sizeof",
            "int main(){ return (sizeof(float)==4 && sizeof(double)==8) ? 42 : 0; }",
        ),
        // Unary minus / plus on doubles: -(-5.5) + 94.5 = 100.
        ("dbl_unary", "int main(){ double x=5.5; return (int)(-(-x) + +94.5); }"),
        // The conditional operator with double arms (usual arithmetic conversions).
        (
            "dbl_ternary",
            "int main(){ int c=1; double a=12.5, b=3.0; return (int)((c ? a : b) * 4.0); }",
        ),
    ]
}

/// C11/C23/C99 language-feature programs, each with the `--std` it must compile
/// under. Both `lf-cc` and `gcc` are invoked with that same `-std`, so the two
/// sides agree on which features are in scope. Each `main` returns `0..256`.
fn std_programs() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        // _Generic (C11): a `TY(x)` macro selecting by controlling type, checked
        // across several types (int/double/char*/long, and a non-matching float).
        (
            "generic_macro",
            "c11",
            "#define TY(x) _Generic((x), int:1, double:2, char*:3, long:4, default:0)\n\
             int main(){ int i=0; double d=0; char*p=0; long l=0; float f=0; \
             return TY(i)+TY(d)*3+TY(p)*7+TY(l)*11+TY(f)*13+3; }",
        ),
        // _Generic selecting different *expressions* (not just constants) by type.
        (
            "generic_select_expr",
            "c11",
            "int gi(void){return 10;} double gd(void){return 2.0;} \
             int pick(int which){ double d=0; int i=0; \
             return which ? _Generic((d), double: (int)gd(), default: 0) \
                          : _Generic((i), int: gi(), default: 0); } \
             int main(){ return pick(1)*4 + pick(0) + 2; }",
        ),
        // _Alignof (C11): alignment of scalar types.
        (
            "alignof_c11",
            "c11",
            "int main(){ return (int)_Alignof(double)+(int)_Alignof(int)*4 \
             +(int)_Alignof(char)*2 + 18; }",
        ),
        // alignof keyword (C23): alignment of a struct (max member alignment).
        (
            "alignof_kw",
            "c23",
            "struct S{ char c; int x; double d; }; \
             int main(){ return (int)alignof(struct S)*5 + (int)alignof(short) + 20; }",
        ),
        // _Alignas (C11): an over-aligned local; verify the address is aligned.
        (
            "alignas_c11",
            "c11",
            "int main(){ _Alignas(32) int x = 7; \
             return (((unsigned long)&x) % 32 == 0) ? x + 35 : 0; }",
        ),
        // alignas keyword (C23) with a type operand: align a local like a double.
        (
            "alignas_kw",
            "c23",
            "int main(){ alignas(16) char buf[8]; \
             return (((unsigned long)&buf[0]) % 16 == 0) ? 42 : 1; }",
        ),
        // typeof (C23): declare a variable of the operand's type; sizeof(typeof).
        (
            "typeof_decl",
            "c23",
            "int main(){ int a[4]={1,2,3,4}; typeof(a[0]) s=0; \
             for(typeof(s) i=0;i<4;i++) s+=a[i]; \
             return s*3 + (int)sizeof(typeof(s)) + 20; }",
        ),
        // typeof_unqual (C23): behaves as typeof here (qualifiers are unmodelled).
        (
            "typeof_unqual",
            "c23",
            "int main(){ double d=3.5; typeof_unqual(d) e=d*2.0; \
             return (int)e + (int)sizeof(typeof_unqual(d)) + 27; }",
        ),
        // Compound literals (C99): as an lvalue (&), indexed, and passed to a fn.
        (
            "compound_literal",
            "c99",
            "struct P{int x,y;}; int use(struct P*p){ return p->x*p->y; } \
             int main(){ int i=2; \
             return use(&(struct P){6,7}) + (int[]){1,2,3,4,5}[i] - 3; }",
        ),
        // A compound literal modified through its lvalue then re-read.
        (
            "compound_literal_lvalue",
            "c11",
            "int main(){ int i=1; int r = ((int[]){10,20,30})[i]; \
             struct Q{int a,b;} *q = &(struct Q){2,3}; q->a += 37; return r + q->a; }",
        ),
        // Anonymous union member (C11): accessed as if a member of the enclosing.
        (
            "anon_union",
            "c11",
            "struct S{ int a; union { int u; unsigned char b[4]; }; }; \
             int main(){ struct S s; s.a=40; s.u=0; s.b[0]=2; return s.a+s.u; }",
        ),
        // Anonymous struct nested in an anonymous union (offsets compose).
        (
            "anon_struct_nested",
            "c11",
            "struct S{ union { long all; struct { int lo, hi; }; }; }; \
             int main(){ struct S s; s.all=0; s.lo=42; s.hi=0; return (int)s.all; }",
        ),
        // C23 attributes: ignored, program still compiles and runs.
        (
            "attributes",
            "c23",
            "[[nodiscard]] int f(void){ return 30; } \
             [[deprecated]] int g(void){ return 5; } \
             int main(){ [[maybe_unused]] int unused = 99; int x = 1; \
             switch(x){ case 1: x = f()+g(); [[fallthrough]]; case 2: break; default: x=0; } \
             return x + 7; }",
        ),
        // _Noreturn (C11): accepted on a declaration/definition (here never called).
        (
            "noreturn_fn",
            "c11",
            "_Noreturn void fail(void); void fail(void){ for(;;){} } \
             int dbl(int x){ return x*2; } int main(){ return dbl(20)+2; }",
        ),
    ]
}

struct Harness {
    dir: PathBuf,
    gcc: Option<String>,
}

impl Harness {
    fn new() -> Harness {
        // A per-instance unique directory: the two differential tests run in
        // parallel within one process, so keying only on the process id would let
        // them share (and race on) the same scratch directory.
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("lf-cc-difftest-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let gcc = which("gcc").or_else(|| which("cc"));
        Harness { dir, gcc }
    }

    /// Compile with lf-cc at `opt`, run, and return the exit code.
    fn lf_run(&self, name: &str, src: &str, opt: OptLevel) -> i32 {
        let image = lf_cc::build_image(src, &format!("{name}.c"), opt, false)
            .unwrap_or_else(|e| panic!("lf-cc failed to build '{name}': {e:?}"));
        let bin = self.dir.join(format!("{name}.lf.{}", opt.name()));
        write_executable(bin.to_str().unwrap(), &image).expect("write executable");
        run_exit(&bin)
    }

    /// Compile with gcc -O0 and run, if gcc is available.
    fn gcc_run(&self, name: &str, src: &str) -> Option<i32> {
        let gcc = self.gcc.as_ref()?;
        let c = self.dir.join(format!("{name}.c"));
        std::fs::write(&c, src).expect("write source");
        let bin = self.dir.join(format!("{name}.gcc"));
        let status = Command::new(gcc)
            .args(["-O0", "-w", "-o"])
            .arg(&bin)
            .arg(&c)
            .status()
            .expect("run gcc");
        assert!(status.success(), "gcc failed to compile '{name}'");
        Some(run_exit(&bin))
    }

    /// Compile with lf-cc under an explicit `--std`, run, return the exit code.
    fn lf_run_std(&self, name: &str, src: &str, opt: OptLevel, std: &str) -> i32 {
        let input = format!("{name}.c");
        let pp = PpOptions {
            std: CStd::parse(std).expect("known --std"),
            main_file_name: input.clone(),
            ..PpOptions::default()
        };
        let image = lf_cc::build_image_with(src, &input, &pp, opt, false)
            .unwrap_or_else(|e| panic!("lf-cc failed to build '{name}' (--std={std}): {e:?}"));
        let bin = self.dir.join(format!("{name}.lf.{}", opt.name()));
        write_executable(bin.to_str().unwrap(), &image).expect("write executable");
        run_exit(&bin)
    }

    /// Compile with `gcc -std=<std> -O0` and run, if gcc is available.
    fn gcc_run_std(&self, name: &str, src: &str, std: &str) -> Option<i32> {
        let gcc = self.gcc.as_ref()?;
        let c = self.dir.join(format!("{name}.c"));
        std::fs::write(&c, src).expect("write source");
        let bin = self.dir.join(format!("{name}.gcc"));
        let status = Command::new(gcc)
            .args(["-O0", "-w", &format!("-std={std}"), "-o"])
            .arg(&bin)
            .arg(&c)
            .status()
            .expect("run gcc");
        assert!(status.success(), "gcc failed to compile '{name}' (-std={std})");
        Some(run_exit(&bin))
    }
}

fn which(prog: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(prog);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    None
}

fn run_exit(bin: &Path) -> i32 {
    let status = Command::new(bin).status().expect("run binary");
    if let Some(code) = status.code() {
        return code;
    }
    use std::os::unix::process::ExitStatusExt;
    panic!("{} killed by signal {:?}", bin.display(), status.signal());
}

#[test]
fn differential_against_gcc() {
    let h = Harness::new();
    let mut matched_gcc = 0usize;
    let mut ran = 0usize;
    let total = programs().len();
    let mut failures: Vec<String> = Vec::new();

    for (name, src) in programs() {
        let o0 = h.lf_run(name, src, OptLevel::O0);
        let o2 = h.lf_run(name, src, OptLevel::O2);
        ran += 1;

        if o0 != o2 {
            failures.push(format!("{name}: lf-cc -O0={o0} != -O2={o2}"));
            continue;
        }
        if let Some(g) = h.gcc_run(name, src) {
            if o0 != g {
                failures.push(format!("{name}: lf-cc={o0} != gcc={g}"));
                continue;
            }
            matched_gcc += 1;
        }
    }

    // The C11/C23/C99 language-feature corpus, each under its required `--std`
    // (see `std_programs`). These run in the same test — rather than a second
    // parallel one — so two full compiler pipelines never contend for memory at
    // once (which could OOM-kill a child), keeping the suite deterministic.
    let std_total = std_programs().len();
    for (name, std, src) in std_programs() {
        let o0 = h.lf_run_std(name, src, OptLevel::O0, std);
        let o2 = h.lf_run_std(name, src, OptLevel::O2, std);
        ran += 1;
        if o0 != o2 {
            failures.push(format!("{name} (--std={std}): lf-cc -O0={o0} != -O2={o2}"));
            continue;
        }
        if let Some(g) = h.gcc_run_std(name, src, std) {
            if o0 != g {
                failures.push(format!("{name} (--std={std}): lf-cc={o0} != gcc={g}"));
                continue;
            }
            matched_gcc += 1;
        }
    }

    eprintln!(
        "differential: {ran}/{} programs ran; {matched_gcc} matched gcc's exit code \
         (gcc {})",
        total + std_total,
        if h.gcc.is_some() { "present" } else { "absent — comparison skipped" }
    );
    assert!(failures.is_empty(), "differential mismatches:\n{}", failures.join("\n"));
    assert_eq!(ran, total + std_total, "every program should compile and run");
}
