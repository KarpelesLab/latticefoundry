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
    ]
}

struct Harness {
    dir: PathBuf,
    gcc: Option<String>,
}

impl Harness {
    fn new() -> Harness {
        let dir = std::env::temp_dir().join(format!("lf-cc-difftest-{}", std::process::id()));
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
    status.code().expect("process returned an exit code")
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

    eprintln!(
        "differential: {ran}/{total} programs ran; {matched_gcc} matched gcc's exit code \
         (gcc {})",
        if h.gcc.is_some() { "present" } else { "absent — comparison skipped" }
    );
    assert!(failures.is_empty(), "differential mismatches:\n{}", failures.join("\n"));
    assert_eq!(ran, total, "every program should compile and run");
}
