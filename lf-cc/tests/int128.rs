//! `__int128` and `unsigned __int128` values: arithmetic, comparisons, shifts,
//! conversions to and from every integer and floating type, constants built
//! from shifts and casts (folded by sema without overflow), globals, struct
//! members and arrays laid out as gcc lays them out (16-byte aligned),
//! `va_arg`, `switch`, and calls to and from gcc-compiled functions in both
//! directions (two registers or a 16-aligned stack slot, `rax:rdx` results).
//! Division and the float conversions call libgcc.
//!
//! Every program is differential: its output must equal what the same source
//! compiled by gcc prints, at -O0 and -O2. The tests skip without the host C
//! runtime or gcc.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use latticefoundry::link::gnu::HostCrt;

const LF_CC: &str = env!("CARGO_BIN_EXE_lf-cc");

/// A per-test scratch directory, removed when the test finishes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lf-cc-int128-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn file(&self, name: &str, text: &str) {
        std::fs::write(self.0.join(name), text).expect("write source");
    }

    fn lf_cc(&self, args: &[&str]) {
        let out = Command::new(LF_CC).args(args).current_dir(&self.0).output().expect("run lf-cc");
        assert!(out.status.success(), "lf-cc {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }

    fn gcc(&self, gcc: &Path, args: &[&str]) {
        let out = Command::new(gcc).args(args).current_dir(&self.0).output().expect("run gcc");
        assert!(out.status.success(), "gcc {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }

    /// Run executable `exe` (in this directory); its stdout.
    fn run(&self, exe: &str) -> String {
        let path = self.0.join(exe);
        for _ in 0..50 {
            match Command::new(&path).output() {
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(20))
                }
                other => {
                    let out: Output = other.expect("run executable");
                    assert!(out.status.success(), "{exe} failed: {}", String::from_utf8_lossy(&out.stderr));
                    return String::from_utf8_lossy(&out.stdout).into_owned();
                }
            }
        }
        panic!("{} stayed busy", path.display());
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

/// The host gcc and C runtime, or `None` (with a note) to skip.
fn host_gcc() -> Option<PathBuf> {
    if HostCrt::discover().is_none() {
        eprintln!("skipping: no host C runtime (crt1.o) found");
        return None;
    }
    let gcc = which("gcc");
    if gcc.is_none() {
        eprintln!("skipping: gcc is not installed");
    }
    gcc
}

/// Compile `src` with gcc (the reference output) and with lf-cc at -O0 and
/// -O2 (and `extra` flags), and compare the outputs.
fn differential(tag: &str, src: &str, extra: &[&str]) {
    let Some(gcc) = host_gcc() else { return };
    let s = Scratch::new(tag);
    s.file("t.c", src);
    let std = common::gcc_std_flag(&gcc, "gnu17");
    let mut args = vec![std.as_str(), "-w", "t.c", "-o", "ref"];
    args.extend_from_slice(extra);
    s.gcc(&gcc, &args);
    let expected = s.run("ref");
    assert!(!expected.is_empty());
    for opt in ["-O0", "-O2"] {
        let exe = format!("lf{opt}");
        let mut args = vec![opt, "t.c", "-o", exe.as_str()];
        args.extend_from_slice(extra);
        s.lf_cc(&args);
        let got = s.run(&exe);
        if got != expected {
            let diff: Vec<String> = expected
                .lines()
                .zip(got.lines())
                .filter(|(e, g)| e != g)
                .take(10)
                .map(|(e, g)| format!("gcc: {e}\nlf:  {g}"))
                .collect();
            panic!("lf-cc {opt} differs from gcc:\n{}", diff.join("\n"));
        }
    }
}

/// A randomized battery: values of every magnitude and sign through every
/// operator and conversion, each operation's results folded into a hash.
const OPS_C: &str = r#"#include <stdio.h>
typedef __int128 i128;
typedef unsigned __int128 u128;

static unsigned long long rng = 0x9e3779b97f4a7c15ULL;
static unsigned long long next(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}
/* A value of a random width (1..128 bits), sometimes negated. */
static u128 value(void) {
    u128 v = ((u128)next() << 64) | next();
    int bits = (int)(next() % 128) + 1;
    if (bits < 128) v &= ((u128)1 << bits) - 1;
    if (next() & 1) v = -v;
    return v;
}

#define NOPS 40
static u128 hash[NOPS];
static const char *names[NOPS];
static void mix(int op, const char *name, u128 v) {
    names[op] = name;
    hash[op] = (hash[op] ^ v) * (((u128)0x100000001b3ULL << 64) | 0x13ULL) + (v >> 67);
}
static void mixd(int op, const char *name, double d) {
    union { double d; unsigned long long u; } b = { d };
    mix(op, name, b.u);
}

int main(void) {
    for (int i = 0; i < 2000; i++) {
        u128 a = value(), b = value();
        i128 sa = (i128)a, sb = (i128)b;
        u128 nz = b ? b : 1;
        i128 snz = sb ? sb : 1;
        if (sa == -sa && snz == -1) snz = 3; /* INT128_MIN / -1 */
        int sh = (int)(b & 127);
        mix(0, "add", a + b);
        mix(1, "sub", a - b);
        mix(2, "mul", a * b);
        mix(3, "udiv", a / nz);
        mix(4, "urem", a % nz);
        mix(5, "sdiv", (u128)(sa / snz));
        mix(6, "srem", (u128)(sa % snz));
        mix(7, "and", a & b);
        mix(8, "or", a | b);
        mix(9, "xor", a ^ b);
        mix(10, "shl", a << sh);
        mix(11, "lshr", a >> sh);
        mix(12, "ashr", (u128)(sa >> sh));
        mix(13, "neg", -a);
        mix(14, "not", ~a);
        mix(15, "ucmp", (a < b) | (a <= b) << 1 | (a > b) << 2 | (a >= b) << 3 | (a == b) << 4 | (a != b) << 5);
        mix(16, "scmp", (sa < sb) | (sa <= sb) << 1 | (sa > sb) << 2 | (sa >= sb) << 3);
        mix(17, "narrow", (u128)(signed char)a ^ (u128)(unsigned short)a << 8 ^ (u128)(int)a << 24 ^ (u128)(unsigned)a << 60);
        mix(18, "narrow64", (u128)(long)a ^ (u128)(unsigned long)b << 3);
        mix(19, "widen", (u128)(i128)(signed char)b ^ (u128)(i128)(short)b ^ (u128)(i128)(int)b ^ (u128)(i128)(long)b);
        mix(20, "uwiden", (u128)(unsigned char)b + (u128)(unsigned short)b + (u128)(unsigned)b + (u128)(unsigned long)b);
        mix(21, "bool", (_Bool)a + (_Bool)(a >> 100) * 2 + !b * 4);
        mixd(22, "u2d", (double)a);
        mixd(23, "s2d", (double)sa);
        mixd(24, "u2f", (float)a);
        mixd(25, "s2f", (float)sa);
        double d = (double)(long)next() * (double)(1ULL << (next() % 60));
        float f = (float)d;
        mix(26, "d2s", (u128)(i128)d);
        mix(27, "d2u", (u128)(d < 0 ? -d : d));
        mix(28, "f2s", (u128)(i128)f);
        mix(29, "f2u", (u128)(f < 0 ? -f : f));
        u128 c = a;
        c += b; c *= 3; c -= a; c ^= b; c <<= sh & 7; c >>= 1; c |= 5; c &= ~(u128)2;
        if (b) c /= b;
        if (b) c %= b | 1;
        mix(30, "compound", c);
        i128 t = sa;
        t++; t--; ++t;
        mix(31, "incdec", (u128)t);
        mix(32, "ternary", sa < 0 ? a : b);
        mix(33, "mixed", (u128)(sa + (int)b) ^ (u128)((long)a * sb) ^ (u128)(a + (unsigned)b));
        mix(34, "longmix", (u128)((unsigned long)b + sa) ^ (u128)(-1 < sa));
    }
    for (int k = 0; k < 35; k++)
        printf("%-8s %016llx%016llx\n", names[k], (unsigned long long)(hash[k] >> 64), (unsigned long long)hash[k]);
    return 0;
}
"#;

#[test]
fn int128_operations_match_gcc() {
    differential("ops", OPS_C, &[]);
}

/// Constants from shifts and casts (folded by sema in static initializers
/// and array bounds), globals, struct members and arrays (gcc's layout),
/// unions, `va_arg`, `switch` and the libgcc conversions.
const PROGRAM_C: &str = r#"#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
typedef __int128 i128;
typedef unsigned __int128 u128;

static void show(const char *tag, u128 v) {
    printf("%s %016llx%016llx\n", tag, (unsigned long long)(v >> 64), (unsigned long long)v);
}

struct S { char c; i128 a; long b; u128 d[2]; };
struct P { char c; struct { i128 v; } in; short s; };
union U { i128 x; long y; char z[24]; };
static const u128 BIG = ~(u128)0 / 3;
static const u128 TOP = (u128)1 << 127;
static const i128 NEG = -((i128)1 << 100) + 5;
static const u128 LOW = ~(u128)0 >> 64;
static const u128 PROD = ((u128)0xffffffffffffffffULL * 0xffffffffffffffffULL) * 7;
u128 gtab[3] = { 1, (u128)0x123456789abcdefULL << 64, ~(u128)0 >> 1 };
i128 gneg = -42;
struct S gs = { 'g', -((i128)1 << 90), 3, { 1, (u128)-1 } };
char bound[(int)(~(u128)0 >> 123)];
enum { EBIG = (int)((u128)1 << 127 >> 120) };
_Static_assert(sizeof(i128) == 16 && _Alignof(u128) == 16, "size");
_Static_assert((u128)-1 > 0 && (i128)-1 < 0, "signedness");
_Static_assert(((u128)1 << 127) / 2 == (u128)1 << 126, "unsigned division");

static u128 sum_va(int n, ...) {
    va_list ap;
    va_start(ap, n);
    u128 s = 0;
    for (int i = 0; i < n; i++) s = s * 7 + va_arg(ap, u128) + (u128)va_arg(ap, int);
    va_end(ap);
    return s;
}

static int classify(i128 v) {
    switch (v) {
    case -1: return 1;
    case 0: return 2;
    case (i128)1 << 64: return 3;
    case ((i128)1 << 100) - 1: return 4;
    default: return 5;
    }
}

int main(void) {
    volatile int k = 3;
    show("BIG", BIG);
    show("TOP", TOP);
    show("NEG", (u128)NEG);
    show("LOW", LOW);
    show("PROD", PROD);
    for (int i = 0; i < 3; i++) show("gtab", gtab[i]);
    show("gneg", (u128)gneg);
    show("gs.a", (u128)gs.a);
    show("gs.d1", gs.d[1]);
    printf("bound %zu ebig %d\n", sizeof bound, EBIG);
    printf("S %zu %zu %zu %zu %zu\n", sizeof(struct S), _Alignof(struct S), offsetof(struct S, a),
           offsetof(struct S, b), offsetof(struct S, d));
    printf("P %zu %zu %zu %zu\n", sizeof(struct P), _Alignof(struct P), offsetof(struct P, in), offsetof(struct P, s));
    printf("U %zu %zu\n", sizeof(union U), _Alignof(union U));
    struct S s = { 'x', -7, 9, { 3, 4 } };
    s.d[1] += s.a * k;
    show("s.d1", s.d[1]);
    struct S arr[3];
    for (int i = 0; i < 3; i++) arr[i] = s, arr[i].a = (i128)i << 70;
    show("arr", (u128)(arr[2].a - arr[1].a));
    printf("arr stride %d\n", (int)((char *)&arr[1] - (char *)&arr[0]));
    struct P p = { 'p', { -5 }, 6 };
    show("p", (u128)(p.in.v * p.s));
    union U u;
    u.x = -1;
    u.y = 0;
    show("u", (u128)u.x);
    i128 local[4] = { 1, -2, (i128)1 << 80, -((i128)1 << 80) };
    i128 acc = 0;
    for (int i = 0; i < 4; i++) acc = acc * 3 + local[i];
    show("local", (u128)acc);
    printf("aligned %d\n", (int)((unsigned long)&local[1] % 16));
    show("va", sum_va(4, (u128)1, 10, ~(u128)0, 20, (u128)gneg, 30, TOP, 40));
    printf("switch %d %d %d %d %d\n", classify(-1), classify(0), classify((i128)1 << 64),
           classify(((i128)1 << 100) - 1), classify(k));
    double d = 1e30;
    show("d2u", (u128)d);
    show("d2s", (u128)(i128)-d);
    printf("u2d %.17g %.17g\n", (double)TOP, (double)NEG);
    printf("u2f %.9g %.9g\n", (float)BIG, (float)(i128)-3);
    return 0;
}
"#;

#[test]
fn int128_programs_match_gcc() {
    differential("program", PROGRAM_C, &[]);
}

/// Functions of `__int128` parameters and results, compiled by gcc and by
/// lf-cc and called across: register pairs, the stack once the registers run
/// out, a struct wrapping an `__int128` (two INTEGER eightbytes), a memory-
/// class struct holding one, variadic calls, and a callback.
const ABI_LIB_C: &str = r#"#include <stdarg.h>
typedef __int128 i128;
typedef unsigned __int128 u128;
struct W { i128 v; };
struct M { long tag; i128 v; char c; };

i128 lib_mul(i128 a, i128 b) { return a * b; }
u128 lib_many(int a, u128 b, long c, u128 d, int e, u128 f, i128 g, int h) {
    return (u128)a + b * 3 + (u128)c + d * 5 + (u128)e + f * 7 + (u128)g * 11 + (u128)h;
}
struct W lib_w(struct W x, int k) { x.v = x.v * k + 1; return x; }
struct M lib_m(int pad, struct M m) { m.v -= m.tag; m.c++; m.tag = pad; return m; }
u128 lib_va(int n, ...) {
    va_list ap;
    va_start(ap, n);
    u128 s = 0;
    for (int i = 0; i < n; i++) s = s * 31 + va_arg(ap, u128);
    va_end(ap);
    return s;
}
i128 lib_cb(i128 (*f)(i128, int), i128 x) { return f(x, 3) + f(-x, 5); }
double lib_tof(i128 x, u128 y) { return (double)x + (double)y; }
"#;

const ABI_MAIN_C: &str = r#"#include <stdio.h>
typedef __int128 i128;
typedef unsigned __int128 u128;
struct W { i128 v; };
struct M { long tag; i128 v; char c; };
i128 lib_mul(i128 a, i128 b);
u128 lib_many(int a, u128 b, long c, u128 d, int e, u128 f, i128 g, int h);
struct W lib_w(struct W x, int k);
struct M lib_m(int pad, struct M m);
u128 lib_va(int n, ...);
i128 lib_cb(i128 (*f)(i128, int), i128 x);
double lib_tof(i128 x, u128 y);

static void show(const char *tag, u128 v) {
    printf("%s %016llx%016llx\n", tag, (unsigned long long)(v >> 64), (unsigned long long)v);
}
static i128 cb(i128 x, int k) { return x * k - 1; }

int main(void) {
    u128 a = ((u128)0x0fedcba987654321ULL << 64) | 0xffeeddccbbaa9988ULL;
    i128 b = -(i128)(a >> 3);
    show("mul", (u128)lib_mul(b, (i128)a));
    show("many", lib_many(1, a, -2, (u128)b, 3, a + 1, b - 1, 4));
    struct W w = { b };
    struct W w2 = lib_w(w, 9);
    show("w", (u128)w2.v);
    struct M m = { 77, b, 'a' };
    struct M r = lib_m(5, m);
    show("m", (u128)r.v);
    printf("m %ld %c\n", r.tag, r.c);
    show("va", lib_va(4, a, (u128)b, (u128)1, a * a));
    show("cb", (u128)lib_cb(cb, b));
    printf("tof %.17g\n", lib_tof(b, a));
    return 0;
}
"#;

#[test]
fn int128_crosses_the_abi_with_gcc_objects() {
    let Some(gcc) = host_gcc() else { return };
    let s = Scratch::new("abi");
    s.file("lib.c", ABI_LIB_C);
    s.file("main.c", ABI_MAIN_C);
    s.gcc(&gcc, &["-O2", "lib.c", "main.c", "-o", "ref"]);
    let expected = s.run("ref");
    s.gcc(&gcc, &["-O2", "-c", "lib.c", "-o", "lib_gcc.o"]);
    s.gcc(&gcc, &["-O2", "-c", "main.c", "-o", "main_gcc.o"]);
    for opt in ["-O0", "-O2"] {
        s.lf_cc(&[opt, "-c", "lib.c", "-o", "lib_lf.o"]);
        s.lf_cc(&[opt, "-c", "main.c", "-o", "main_lf.o"]);
        // lf-cc calls gcc's functions, and gcc calls lf-cc's.
        s.lf_cc(&["main_lf.o", "lib_gcc.o", "-o", "lf_main"]);
        assert_eq!(s.run("lf_main"), expected, "lf-cc {opt} caller, gcc callee");
        s.lf_cc(&["main_gcc.o", "lib_lf.o", "-o", "lf_lib"]);
        assert_eq!(s.run("lf_lib"), expected, "gcc caller, lf-cc {opt} callee");
    }
}

/// A shared library using the libgcc helpers (`__udivti3`, `__floattidf`)
/// links them in, so a program `dlopen`s it.
#[test]
fn shared_library_links_the_libgcc_helpers() {
    let Some(gcc) = host_gcc() else { return };
    let s = Scratch::new("shared");
    s.file(
        "lib.c",
        "unsigned long lib_div(unsigned long hi, unsigned long lo, unsigned long d) {\n\
             unsigned __int128 v = ((unsigned __int128)hi << 64) | lo;\n\
             return (unsigned long)(v / d) ^ (unsigned long)(v % d);\n\
         }\n\
         double lib_conv(long hi) { return (double)((__int128)hi << 64); }\n",
    );
    s.file(
        "host.c",
        "#include <dlfcn.h>\n#include <stdio.h>\n\
         int main(int argc, char **argv) {\n\
             void *h = dlopen(argv[1], RTLD_NOW);\n\
             if (!h) { printf(\"dlopen: %s\\n\", dlerror()); return 0; }\n\
             unsigned long (*div)(unsigned long, unsigned long, unsigned long) =\n\
                 (unsigned long (*)(unsigned long, unsigned long, unsigned long))dlsym(h, \"lib_div\");\n\
             double (*conv)(long) = (double (*)(long))dlsym(h, \"lib_conv\");\n\
             printf(\"%lu %.17g\\n\", div(12345, 678910, 1000003), conv(-3));\n\
             return 0;\n\
         }\n",
    );
    s.lf_cc(&["-O2", "-shared", "lib.c", "-o", "libwide.so"]);
    s.gcc(&gcc, &["host.c", "-o", "host", "-ldl"]);
    let lib = s.0.join("libwide.so");
    let out = Command::new(s.0.join("host")).arg(&lib).output().expect("run host");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "227724372416602294 -5.5340232221128655e+19\n");
}

/// Compile `src` through sema and return the diagnostics' text (empty on success).
fn check_errors(src: &str) -> String {
    match lf_cc::check_source(src) {
        Ok(_) => String::new(),
        Err(diags) => diags.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>().join("\n"),
    }
}

#[test]
fn unsupported_wide_forms_are_rejected_clearly() {
    for (src, needle) in [
        ("_Atomic __int128 q; int f(void) { return (int)q; }", "wider than 8 bytes"),
        ("int f(_Atomic unsigned __int128 *p) { return (int)(*p += 1); }", "wider than 8 bytes"),
        ("struct B { __int128 x : 70; };", "bit-fields of type '__int128'"),
    ] {
        let errs = check_errors(src);
        assert!(errs.contains(needle), "expected '{needle}' for:\n{src}\ngot: {errs}");
    }
}

/// The module declares gcc's `__int128` alignment, so an IR struct of an
/// `i128` member lays out like the C struct.
#[test]
fn the_data_layout_aligns_i128_to_16() {
    let (module, _) = lf_cc::compile_to_ir(
        "struct S { char c; __int128 v; }; long f(struct S *s) { return (long)s->v; }",
        "t.c",
        false,
    )
    .expect("compiles");
    assert_eq!(module.data_layout().int_align(128), 16);
    assert_eq!(module.data_layout(), &lf_cc::lower::data_layout());
}
