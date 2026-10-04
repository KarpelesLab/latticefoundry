//! GCC vector extensions: `__attribute__((vector_size(N)))` types lower to IR
//! `<N x T>` vectors with element-wise arithmetic, comparisons yielding `-1`/`0`
//! lanes, subscripting, scalar broadcast, initialization, casts and the
//! shuffle/convert builtins — compared with gcc, including vectors passed to
//! and returned from gcc-compiled functions (the System V XMM convention).

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use latticefoundry::link::gnu::HostCrt;

const LF_CC: &str = env!("CARGO_BIN_EXE_lf-cc");

/// A per-test scratch directory, removed when the test finishes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lf-cc-vectors-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn file(&self, name: &str, text: &str) {
        std::fs::write(self.0.join(name), text).expect("write source");
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn which(prog: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(prog)).find(|p| p.is_file())
}

fn run(exe: &Path) -> Output {
    for _ in 0..50 {
        match Command::new(exe).output() {
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(20)),
            other => return other.expect("run executable"),
        }
    }
    panic!("{} stayed busy", exe.display());
}

fn lf_cc(dir: &Path, args: &[&str]) {
    let out = Command::new(LF_CC).args(args).current_dir(dir).output().expect("run lf-cc");
    assert!(out.status.success(), "lf-cc {args:?}: {}", String::from_utf8_lossy(&out.stderr));
}

/// Run `gcc args` in `dir`; `false` (with a note) when it fails, e.g. a gcc too
/// old for a builtin.
fn gcc(gcc: &Path, dir: &Path, args: &[&str]) -> bool {
    let out = Command::new(gcc).args(args).current_dir(dir).output().expect("run gcc");
    if !out.status.success() {
        eprintln!("skipping the gcc comparison: gcc {args:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    out.status.success()
}

/// Every operation on several vector types; the output is gcc's.
const VECTOR_C: &str = r#"#include <stdio.h>

typedef int v4si __attribute__((vector_size(16)));
typedef unsigned int v4su __attribute__((vector_size(16)));
typedef float v4sf __attribute__((vector_size(16)));
typedef double v2df __attribute__((vector_size(16)));
typedef long long v2di __attribute__((vector_size(16)));
typedef short v8hi __attribute__((vector_size(16)));
typedef unsigned char v16qu __attribute__((vector_size(16)));
typedef int v2si __attribute__((vector_size(8)));
typedef long long __m128i __attribute__((__vector_size__(16), __may_alias__));

v4si gv = {1, 2, 3, 4};
static v4sf gf = {0.5f, 1.5f};
struct S { int tag; v4si v; } gs = {7, {9, 8, 7, 6}};

static void pi(const char *n, v4si v) { printf("%s %d %d %d %d\n", n, v[0], v[1], v[2], v[3]); }
static void pf(const char *n, v4sf v) { printf("%s %g %g %g %g\n", n, v[0], v[1], v[2], v[3]); }

v4si add(v4si a, v4si b) { return a + b; }
v4sf fmadd(v4sf a, v4sf b, v4sf c) { return a * b + c; }
v2df dsub(v2df a, v2df b) { return a - b; }

int main(void) {
    v4si a = {1, -2, 3, -4};
    v4si b = {10, 20, 30, 40};
    pi("add", add(a, b));
    pi("sub", a - b);
    pi("mul", a * b);
    pi("div", b / a);
    pi("rem", b % a);
    pi("and", a & b);
    pi("or", a | b);
    pi("xor", a ^ b);
    pi("shl", a << 2);
    pi("shr", a >> 1);
    pi("shv", b >> (v4si){0, 1, 2, 3});
    pi("neg", -a);
    pi("not", ~a);
    pi("scal", a + 5);
    pi("scal2", 100 - a);
    pi("lt", a < b);
    pi("eq", a == (v4si){1, 2, 3, -4});
    v4su ua = (v4su)a;
    pi("ushr", (v4si)(ua >> 28));
    pi("ult", (v4si)(ua < (v4su){5, 5, 5, 5}));
    v4sf f = {1.0f, 2.5f, -3.0f, 4.25f};
    pf("fadd", f + gf);
    pf("fma", fmadd(f, f, gf));
    pf("fneg", -f);
    pf("fdiv", f / 2);
    pi("fcmp", f > (v4sf){0, 0, 0, 0});
    v2df d = {1.5, -2.25};
    v2df e = dsub(d, (v2df){0.5, 0.25});
    printf("dsub %g %g\n", e[0], e[1]);
    v2di dl = (v2di)(d < e);
    printf("dcmp %lld %lld\n", dl[0], dl[1]);
    a[1] = 77;
    a[3] += 3;
    pi("idx", a);
    int s = 0;
    for (int i = 0; i < 4; i++) s += b[i];
    printf("sum %d\n", s);
    printf("rv %d\n", add(a, b)[2]);
    a += b;
    a *= 2;
    a -= 1;
    a <<= 1;
    pi("cmpd", a);
    pi("glob", gv);
    gv[2] = 33;
    pi("glob2", gv + gs.v);
    v8hi h = {1, 2, 3, 4, 5, 6, 7, 32767};
    h = h + 1;
    printf("v8hi %d %d %d\n", h[0], h[6], h[7]);
    v16qu q = {250, 1, 2};
    q += 10;
    printf("v16qu %d %d %d\n", q[0], q[1], q[15]);
    __m128i m = (__m128i)b;
    printf("m128 %llx %llx\n", (unsigned long long)m[0], (unsigned long long)m[1]);
    v2si small = {3, 4};
    small = small * small;
    printf("v2si %d %d %zu\n", small[0], small[1], sizeof small);
    pi("shuf", __builtin_shufflevector(b, a, 3, 2, 5, 4));
    pi("shuf1", __builtin_shuffle(b, (v4si){3, 3, 0, 1}));
    pi("shuf2", __builtin_shuffle(b, a, (v4si){7, 0, 6, 1}));
    pf("conv", __builtin_convertvector(b, v4sf));
    pi("conv2", __builtin_convertvector(f, v4si));
    v2df cd = __builtin_convertvector((v2si){-1, 7}, v2df);
    printf("conv3 %g %g\n", cd[0], cd[1]);
    v4si z = {5};
    pi("zero", z);
    v4si cond = 1 ? a : b;
    pi("cond", cond);
    printf("sz %zu %zu %zu\n", sizeof(v4si), _Alignof(v4si), _Alignof(v2si));
    return 0;
}
"#;

const VECTOR_OUT: &str = "add 11 18 33 36\n\
sub -9 -22 -27 -44\n\
mul 10 -40 90 -160\n\
div 10 -10 10 -10\n\
rem 0 0 0 0\n\
and 0 20 2 40\n\
or 11 -2 31 -4\n\
xor 11 -22 29 -44\n\
shl 4 -8 12 -16\n\
shr 0 -1 1 -2\n\
shv 10 10 7 5\n\
neg -1 2 -3 4\n\
not -2 1 -4 3\n\
scal 6 3 8 1\n\
scal2 99 102 97 104\n\
lt -1 -1 -1 -1\n\
eq -1 0 -1 -1\n\
ushr 0 15 0 15\n\
ult -1 0 -1 0\n\
fadd 1.5 4 -3 4.25\n\
fma 1.5 7.75 9 18.0625\n\
fneg -1 -2.5 3 -4.25\n\
fdiv 0.5 1.25 -1.5 2.125\n\
fcmp -1 -1 0 -1\n\
dsub 1 -2.5\n\
dcmp 0 0\n\
idx 1 77 3 -1\n\
sum 100\n\
rv 33\n\
cmpd 42 386 130 154\n\
glob 1 2 3 4\n\
glob2 10 10 40 10\n\
v8hi 2 8 -32768\n\
v16qu 4 11 10\n\
m128 140000000a 280000001e\n\
v2si 9 16 8\n\
shuf 40 30 386 42\n\
shuf1 40 40 10 20\n\
shuf2 154 10 130 20\n\
conv 10 20 30 40\n\
conv2 1 2 -3 4\n\
conv3 -1 7\n\
zero 5 0 0 0\n\
cond 42 386 130 154\n\
sz 16 16 8\n\
";

#[test]
fn vector_programs_match_gcc() {
    if HostCrt::discover().is_none() {
        eprintln!("skipping: no host C runtime (crt1.o) found");
        return;
    }
    let s = Scratch::new("ops");
    s.file("vec.c", VECTOR_C);
    for opt in ["-O0", "-O2"] {
        let exe = format!("vec{opt}");
        lf_cc(&s.0, &[opt, "vec.c", "-o", &exe]);
        assert_eq!(String::from_utf8_lossy(&run(&s.0.join(&exe)).stdout), VECTOR_OUT, "lf-cc {opt}");
    }
    if let Some(g) = which("gcc") {
        let std = common::gcc_std_flag(&g, "gnu17");
        if gcc(&g, &s.0, &[&std, "-w", "vec.c", "-o", "vec-gcc"]) {
            assert_eq!(String::from_utf8_lossy(&run(&s.0.join("vec-gcc")).stdout), VECTOR_OUT, "gcc");
        }
    }
}

/// Functions taking and returning vectors (in XMM registers, also past the
/// eighth vector argument on the stack), mixed with float arguments, a
/// memory-class struct holding a vector, and vectors through pointers.
const ABI_LIB_C: &str = r#"typedef long long m128i __attribute__((vector_size(16)));
typedef float v4sf __attribute__((vector_size(16)));
typedef double v2df __attribute__((vector_size(16)));
typedef int v4si __attribute__((vector_size(16)));
struct WV { v4sf v; };
struct Mix { int tag; v4si v; };

m128i g_add(m128i a, m128i b) { return a + b; }
v4sf g_mix(float s, v4sf a, double t, v2df b, v4si c) {
    return a * s + (v4sf){(float)b[0], (float)b[1], (float)t, (float)c[3]};
}
v4si g_many(v4si a, v4si b, v4si c, v4si d, v4si e, v4si f, v4si g, v4si h, v4si i, v4si j) {
    return a + b + c + d + e + f + g + h + i * 2 + j * 3;
}
int g_mixsum(struct Mix m) { return m.tag + m.v[0] + m.v[1] + m.v[2] + m.v[3]; }
v4si g_ptr(const v4si *p, v4si *out) { *out = *p * 3; return *p + 1; }
"#;

const ABI_MAIN_C: &str = r#"#include <stdio.h>
typedef long long m128i __attribute__((vector_size(16)));
typedef float v4sf __attribute__((vector_size(16)));
typedef double v2df __attribute__((vector_size(16)));
typedef int v4si __attribute__((vector_size(16)));
struct WV { v4sf v; };
struct Mix { int tag; v4si v; };
m128i g_add(m128i a, m128i b);
v4sf g_mix(float s, v4sf a, double t, v2df b, v4si c);
v4si g_many(v4si a, v4si b, v4si c, v4si d, v4si e, v4si f, v4si g, v4si h, v4si i, v4si j);
int g_mixsum(struct Mix m);
v4si g_ptr(const v4si *p, v4si *out);

int main(void) {
    m128i r = g_add((m128i){1, 2}, (m128i){40, 50});
    printf("%lld %lld\n", r[0], r[1]);
    v4sf m = g_mix(2.0f, (v4sf){1, 2, 3, 4}, 9.5, (v2df){0.25, 0.5}, (v4si){0, 0, 0, 7});
    printf("%g %g %g %g\n", m[0], m[1], m[2], m[3]);
    v4si one = {1, 1, 1, 1};
    v4si k = g_many(one, one, one, one, one, one, one, one, (v4si){1, 2, 3, 4}, (v4si){10, 20, 30, 40});
    printf("%d %d %d %d\n", k[0], k[1], k[2], k[3]);
    printf("%d\n", g_mixsum((struct Mix){100, {1, 2, 3, 4}}));
    v4si in = {5, 6, 7, 8}, out;
    v4si p = g_ptr(&in, &out);
    printf("%d %d %d\n", p[0], out[1], out[3]);
    return 0;
}
"#;

const ABI_OUT: &str = "41 52\n2.25 4.5 15.5 15\n40 72 104 136\n110\n6 18 24\n";

#[test]
fn vectors_cross_the_abi_with_gcc_objects() {
    if HostCrt::discover().is_none() {
        eprintln!("skipping: no host C runtime (crt1.o) found");
        return;
    }
    let s = Scratch::new("abi");
    s.file("lib.c", ABI_LIB_C);
    s.file("main.c", ABI_MAIN_C);
    lf_cc(&s.0, &["-c", "lib.c", "-o", "lib_lf.o"]);
    lf_cc(&s.0, &["-O2", "-c", "main.c", "-o", "main_lf.o"]);
    lf_cc(&s.0, &["main_lf.o", "lib_lf.o", "-o", "lf"]);
    assert_eq!(String::from_utf8_lossy(&run(&s.0.join("lf")).stdout), ABI_OUT);
    let Some(g) = which("gcc") else {
        eprintln!("skipping the gcc interoperation: gcc is not installed");
        return;
    };
    assert!(gcc(&g, &s.0, &["-c", "lib.c", "-o", "lib_gcc.o"]));
    assert!(gcc(&g, &s.0, &["-c", "main.c", "-o", "main_gcc.o"]));
    // lf-cc calls gcc's functions, and gcc calls lf-cc's.
    lf_cc(&s.0, &["main_lf.o", "lib_gcc.o", "-o", "lf_main"]);
    assert_eq!(String::from_utf8_lossy(&run(&s.0.join("lf_main")).stdout), ABI_OUT, "lf-cc caller");
    lf_cc(&s.0, &["main_gcc.o", "lib_lf.o", "-o", "gcc_main"]);
    assert_eq!(String::from_utf8_lossy(&run(&s.0.join("gcc_main")).stdout), ABI_OUT, "gcc caller");
}

#[test]
fn vector_operations_are_ir_vectors() {
    let src = "typedef int v4si __attribute__((vector_size(16)));\n\
               typedef float v4sf __attribute__((vector_size(16)));\n\
               v4si add(v4si a, v4si b) { return a + b; }\n\
               v4si lt(v4sf a, v4sf b) { return a < b; }\n\
               v4si twice(v4si a) { return a * 2; }\n\
               v4si rev(v4si a) { return __builtin_shufflevector(a, a, 3, 2, 1, 0); }\n";
    let (module, syms) = lf_cc::compile_to_ir(src, "t.c", false).expect("compiles");
    let ir = latticefoundry::ir::text::print_module(&module, &syms);
    assert!(ir.contains("func @add(<4 x i32>, <4 x i32>) -> <4 x i32>"), "{ir}");
    assert!(ir.contains("fcmp olt"), "{ir}");
    assert!(ir.contains("splat"), "{ir}");
    assert!(ir.contains("shufflevector"), "{ir}");
}

#[test]
fn invalid_vector_code_is_diagnosed() {
    for (src, needle) in [
        ("typedef _Bool vb __attribute__((vector_size(16))); int main(void) { return 0; }", "element type"),
        ("typedef int v3 __attribute__((vector_size(12))); int main(void) { return 0; }", "power-of-two"),
        (
            "typedef int v4si __attribute__((vector_size(16))); int main(void) { v4si v; v = 1; return 0; }",
            "cannot assign",
        ),
        (
            "typedef int v4si __attribute__((vector_size(16))); int main(void) { v4si v = {0}; return (int)v; }",
            "equal sizes",
        ),
        (
            "typedef float v4sf __attribute__((vector_size(16))); int main(void) { v4sf v = {0}; v = v % 2; return 0; }",
            "integer vector elements",
        ),
        (
            "typedef int v4si __attribute__((vector_size(16))); int main(void) { v4si v = {0}, m = {0}; v = __builtin_shuffle(v, m); return 0; }",
            "constant shuffle mask",
        ),
        (
            "typedef int v4si __attribute__((vector_size(16))); int main(void) { v4si v = {0}; if (v) return 1; return 0; }",
            "scalar",
        ),
    ] {
        let errs = match lf_cc::check_source(src) {
            Ok(_) => String::new(),
            Err(d) => d.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"),
        };
        assert!(errs.contains(needle), "expected '{needle}' for {src}: {errs:?}");
    }
}
