//! The GNU C dialect the glibc headers are written in (M9), exercised with
//! focused, self-contained programs — no system headers and no gcc needed.
//!
//! Each run-program is compiled by lf-cc at -O0 and -O2, run (a freestanding,
//! self-contained image), and its exit code checked against the expected value
//! (and against gcc's when gcc is installed). Further tests check that the
//! constructs lf-cc can declare but not compute with are rejected clearly, and
//! that a `__builtin_va_list` crosses into the host libc's `vsnprintf`.

use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::link::gnu::HostCrt;
use latticefoundry::link::write_executable;
use latticefoundry::transform::pipeline::OptLevel;

/// `(name, source, expected exit code)`.
fn programs() -> Vec<(&'static str, &'static str, i32)> {
    vec![
        (
            "attributes_everywhere",
            r#"
typedef int myint __attribute__((__aligned__(4)));
struct __attribute__((__packed__)) P { char c; int i; } __attribute__((unused));
extern int f1(int a __attribute__((unused)), char *__restrict __attribute__((nonnull)) p)
    __attribute__((__nothrow__, __leaf__)) __attribute__((pure));
enum __attribute__((packed)) E { EA __attribute__((deprecated)) = 3, EB };
__attribute__((noinline)) static int g(int x) { return x + 1; }
extern __inline __attribute__((__always_inline__)) int h(void);
int main(void) {
    int __attribute__((unused)) local = 5;
    char * __attribute__((aligned(8))) q = 0;
    myint m = 1;
    struct P p = { 'a', 7 };
    (void)q;
lbl: __attribute__((unused));
    switch (local) { case 5: local++; __attribute__((fallthrough)); default: break; }
    return g(p.i) + EB + (int)sizeof(struct P) + m;
}
"#,
            8 + 4 + 5 + 1,
        ),
        (
            "mode_attribute",
            r#"
typedef int i8 __attribute__((mode(QI)));
typedef unsigned int u16 __attribute__((__mode__(__HI__)));
typedef int w __attribute__((mode(__word__)));
typedef unsigned uw __attribute__((mode(pointer)));
typedef int __attribute__((__mode__(__SI__))) si;
_Static_assert(sizeof(i8) == 1 && sizeof(u16) == 2 && sizeof(w) == 8 && sizeof(uw) == 8
               && sizeof(si) == 4, "modes");
int main(void) {
    i8 a = 127; a++;
    u16 b = 65535; b++;
    w c = 1; c <<= 40;
    uw d = 0; d--;
    return (a == -128) + (b == 0) * 2 + (c == (1L << 40)) * 4 + (d > 0xffffffffu) * 8;
}
"#,
            15,
        ),
        (
            "packed_and_aligned_layout",
            r#"
struct __attribute__((packed)) A { char c; int i; short s; };
struct B { char c; int i __attribute__((aligned(16))); };
struct C { char c; } __attribute__((aligned(8)));
union __attribute__((aligned(16))) U { int x; };
typedef struct { char c; double d; } __attribute__((packed)) D;
_Static_assert(sizeof(struct A) == 7, "packed");
_Static_assert(__builtin_offsetof(struct A, i) == 1, "packed offset");
_Static_assert(sizeof(struct B) == 32 && __builtin_offsetof(struct B, i) == 16, "aligned member");
_Static_assert(sizeof(struct C) == 8 && _Alignof(struct C) == 8, "aligned struct");
_Static_assert(sizeof(union U) == 16 && __alignof__(union U) == 16, "aligned union");
_Static_assert(sizeof(D) == 9, "packed typedef");
int main(void) {
    struct A a;
    a.c = 1; a.i = 0x12345678; a.s = 3;
    unsigned char *p = (unsigned char *)&a;
    return p[1] + p[5] + a.s + (a.i == 0x12345678);
}
"#,
            0x78 + 3 + 3 + 1,
        ),
        (
            "statement_expressions",
            r#"
#define MAX(a, b) ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })
int calls;
static int next(void) { return ++calls; }
struct pt { int x, y; };
int main(void) {
    int x = MAX(3, 9);
    int y = MAX(next(), 0);
    ({ calls += 10; });
    int z = ({ int t = 4; for (int i = 0; i < 3; i++) t += i; t; });
    struct pt p = ({ struct pt q = { 5, 6 }; q; });
    __typeof(p.y) py = p.y;
    return x + y + calls + z + p.x + py;
}
"#,
            9 + 1 + 11 + 7 + 5 + 6,
        ),
        (
            "compile_time_builtins",
            r#"
struct S { int a; char b[10]; struct { short x, y; } in; };
int main(void) {
    int r = 0;
    r += __builtin_offsetof(struct S, in.y) == 16;
    r += (__builtin_offsetof(struct S, b[3]) == 7) * 2;
    r += __builtin_types_compatible_p(int, int) * 4;
    r += __builtin_types_compatible_p(int, long) * 8;
    r += __builtin_choose_expr(1, 16, (void)0);
    r += (__builtin_bswap32(0x11223344u) == 0x44332211u) * 32;
    r += (__builtin_bswap16(0xabcd) == 0xcdab) * 64;
    r += __builtin_bswap64(0x0102030405060708ull) == 0x0807060504030201ull;
    r += __builtin_expect(r, 0) > 0;
    return r;
}
"#,
            1 + 2 + 4 + 16 + 32 + 64 + 1 + 1,
        ),
        (
            "float_classification",
            r#"
int main(void) {
    double z = 0.0, nan = z / z, inf = 1.0 / z, negz = -z;
    float nf = (float)nan;
    int r = 0;
    r += __builtin_isnan(nan) != 0;
    r += (__builtin_isnan(1.0) == 0) * 2;
    r += (__builtin_isinf_sign(-inf) == -1) * 4;
    r += (__builtin_isfinite(inf) == 0) * 8;
    r += (__builtin_isnormal(1e-310) == 0) * 16;
    r += (__builtin_signbit(negz) != 0) * 32;
    r += (__builtin_fpclassify(0, 1, 2, 3, 4, 1e-310) == 3) * 64;
    r += __builtin_isnan(nf) != 0;
    r += (__builtin_isunordered(nan, 1.0) != 0) + __builtin_isgreater(2.0, 1.0);
    r += (__builtin_huge_val() == inf) + (__builtin_inff() > 1e38f);
    r += __builtin_isinf(-inf) != 0 && __builtin_isnormal(1.0) && !__builtin_signbit(1.0f);
    return r;
}
"#,
            1 + 2 + 4 + 8 + 16 + 32 + 64 + 1 + 2 + 2 + 1,
        ),
        (
            "transparent_union_parameter",
            r#"
typedef union { int *ip; const long *lp; } arg_t __attribute__((__transparent_union__));
int take(arg_t a);
int take(arg_t a) { return *a.ip; }
int main(void) { int v = 42; long w = 7; return take(&v) + take(&w); }
"#,
            49,
        ),
        (
            "function_name_identifiers",
            r#"
static int len(const char *s) { int n = 0; while (s[n]) n++; return n; }
int some_function(void) { return len(__func__) * 10 + len(__FUNCTION__); }
int main(void) { return some_function() - 100 + (__PRETTY_FUNCTION__[0] == 'm'); }
"#,
            44,
        ),
        (
            "declared_only_wide_types",
            r#"
typedef __int128 i128;
typedef unsigned __int128 u128;
typedef int ti __attribute__((mode(TI)));
_Static_assert(sizeof(i128) == 16 && _Alignof(u128) == 16 && sizeof(ti) == 16, "int128");
_Static_assert(sizeof(__uint128_t) == 16, "uint128_t");
extern i128 big_op(i128 a);
_Float128 f128_op(_Float128);
extern __float128 fq;
struct holder { char c; _Float128 q; };
_Static_assert(sizeof(struct holder) == 32, "_Float128 is 16-aligned");
typedef _Float64 f64; typedef _Float32 f32; typedef _Float32x f32x; typedef _Float64x f64x;
int main(void) {
    f64 a = 1.5; f32 b = 2.5f; f32x c = 3; f64x d = 4;
    return (int)(a + b + c + d) + sizeof(f32) + sizeof(f64) + (int)sizeof(fq);
}
"#,
            11 + 4 + 8 + 16,
        ),
        (
            "gnu_inline_extern_inline",
            r#"
extern __inline __attribute__((__gnu_inline__)) int twice(int x) { return x * 2; }
int twice(int x) { return x * 3; }
static inline int unused_helper(void) { __asm__("nop" ::: "memory"); return 0; }
static inline int used_helper(int x) { return x + 1; }
static __inline int chained(int x) { return used_helper(x) * 2; }
int main(void) { return twice(5) + chained(1); }
"#,
            15 + 4,
        ),
        (
            "parameter_array_forms",
            r#"
static int sum(int n, const int a[__restrict n]) { int s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }
static int first(int a[static 2]) { return a[0] + a[1]; }
int star(int n, int a[*]);
int main(void) { int v[] = {1, 2, 3, 4}; return sum(4, v) + first(v); }
"#,
            13,
        ),
        (
            "struct_member_extras",
            r#"
struct T { int a; _Static_assert(sizeof(int) == 4, "int"); ; union { int u; float f; }; };
struct F { int n; int data[]; };
struct Z { int n; char pad[0]; };
int main(void) {
    struct T t; t.u = 5;
    return t.u + (int)sizeof(struct T) + (int)sizeof(struct F) + (int)sizeof(struct Z);
}
"#,
            5 + 8 + 4 + 4,
        ),
        (
            "alignof_forms",
            r#"
int main(void) { int x = 0; double d = 0; return __alignof__(x) + __alignof__ (double) + _Alignof(char) + __alignof(d) + x; }
"#,
            4 + 8 + 1 + 8,
        ),
        (
            "computed_goto",
            r#"
static int run(const unsigned char *code) {
    static const void *const ops[] = { &&op_halt, &&op_inc, &&op_dbl, &&op_add10 };
    int acc = 0;
    const void *entry = &&next;
    goto *entry;
next:
    goto *ops[*code++];
op_inc: acc += 1; goto next;
op_dbl: acc *= 2; goto next;
op_add10: acc += 10; goto *ops[*code++];
op_halt:
    return acc + (&&op_inc != &&op_dbl);
}
int main(void) {
    const unsigned char prog[] = { 1, 1, 2, 3, 2, 1, 0 };
    return run(prog);
}
"#,
            ((1 + 1) * 2 + 10) * 2 + 1 + 1,
        ),
        (
            "builtin_va_list_in_program",
            r#"
typedef __builtin_va_list my_va;
_Static_assert(sizeof(my_va) == 24, "the SysV va_list is 24 bytes");
static int sum_v(int n, my_va ap) { int s = 0; while (n--) s += __builtin_va_arg(ap, int); return s; }
static int sum(int n, ...) {
    my_va ap, aq;
    __builtin_va_start(ap, n);
    __builtin_va_copy(aq, ap);
    int a = sum_v(n, ap);
    int b = sum_v(n, aq);
    __builtin_va_end(aq);
    __builtin_va_end(ap);
    return a + b;
}
int main(void) { return sum(4, 1, 2, 3, 4); }
"#,
            20,
        ),
    ]
}

fn which(prog: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(prog)).find(|c| c.is_file())
}

fn run_exit(bin: &Path) -> i32 {
    for _ in 0..50 {
        match Command::new(bin).status() {
            // ETXTBSY: another test thread's fork may briefly hold the file open.
            Err(e) if e.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            Ok(status) => {
                use std::os::unix::process::ExitStatusExt;
                return status
                    .code()
                    .unwrap_or_else(|| panic!("{} killed by {:?}", bin.display(), status.signal()));
            }
            Err(e) => panic!("run {}: {e}", bin.display()),
        }
    }
    panic!("{} stayed busy", bin.display());
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-cc-gnuc-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

#[test]
fn gnu_c_programs_run_correctly() {
    let dir = scratch("run");
    let gcc = which("gcc");
    let mut failures = Vec::new();
    for (name, src, expected) in programs() {
        for opt in [OptLevel::O0, OptLevel::O2] {
            let image = match lf_cc::build_image(src, &format!("{name}.c"), opt, false) {
                Ok(i) => i,
                Err(e) => {
                    failures.push(format!("{name} ({}): lf-cc failed: {e:?}", opt.name()));
                    continue;
                }
            };
            let bin = dir.join(format!("{name}.{}", opt.name()));
            write_executable(bin.to_str().unwrap(), &image).expect("write executable");
            let got = run_exit(&bin);
            if got != expected {
                failures.push(format!("{name} ({}): exit {got}, expected {expected}", opt.name()));
            }
        }
        if let Some(gcc) = &gcc {
            let c = dir.join(format!("{name}.c"));
            std::fs::write(&c, src).expect("write source");
            let bin = dir.join(format!("{name}.gcc"));
            let ok = Command::new(gcc)
                .args(["-std=gnu17", "-O0", "-w", "-o"])
                .arg(&bin)
                .arg(&c)
                .status()
                .expect("run gcc")
                .success();
            assert!(ok, "gcc failed to compile '{name}'");
            let g = run_exit(&bin);
            if g != expected {
                failures.push(format!("{name}: gcc gives {g}, the test expects {expected}"));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "GNU C program failures:\n{}", failures.join("\n"));
}

/// Compile `src` through sema and return the diagnostics' text (empty on success).
fn check_errors(src: &str) -> String {
    match lf_cc::check_source(src) {
        Ok(_) => String::new(),
        Err(diags) => diags.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>().join("\n"),
    }
}

#[test]
fn unsupported_constructs_are_rejected_clearly() {
    let cases: &[(&str, &str)] = &[
        ("int main(void) { __int128 x = 1; return (int)x; }", "__int128"),
        ("__int128 g; int main(void) { return 0; }", "__int128"),
        ("_Float128 f(_Float128); int main(void) { f(1.0); return 0; }", "_Float128"),
        ("_Float128 q(void); int main(void) { return (int)q(); }", "_Float128"),
        ("typedef float v4 __attribute__((vector_size(16))); int main(void){return 0;}", "vector"),
        ("__thread int t; int main(void) { return 0; }", "thread-local"),
        ("int main(void) { static _Thread_local int t; return t; }", "thread-local"),
        ("_Complex double z; int main(void) { return 0; }", "complex"),
        ("typedef int bad __attribute__((mode(V4SI))); int main(void){return 0;}", "machine mode"),
        (
            "static inline int u(void) { __asm__(\"nop\"); return 0; } int main(void) { return u(); }",
            "inline assembly",
        ),
    ];
    for (src, needle) in cases {
        let errs = check_errors(src);
        assert!(
            errs.contains(needle),
            "expected an error mentioning '{needle}' for:\n{src}\ngot: {errs:?}"
        );
    }
}

#[test]
fn declaration_only_uses_are_accepted() {
    // Naming the types in declarations and `sizeof` is fine; only values are not.
    for src in [
        "extern __int128 v; int main(void) { return (int)sizeof v; }",
        "typedef _Float128 q; q *p; int main(void) { return p == 0; }",
        "static inline int u(void) { __asm__(\"nop\"); return 0; } int main(void) { return 0; }",
    ] {
        let errs = check_errors(src);
        assert!(errs.is_empty(), "unexpected errors for:\n{src}\n{errs}");
    }
}

#[test]
fn gnu_inline_definition_is_not_emitted() {
    let prog = lf_cc::check_source(
        "extern __inline __attribute__ ((__gnu_inline__)) int getc_like(int x) { return x; }\n\
         int main(void) { return getc_like(3); }",
    )
    .expect("checks");
    assert!(
        prog.funcs.iter().all(|f| f.name != "getc_like"),
        "an extern gnu_inline definition must not be emitted"
    );
    let sig = prog.sigs.iter().find(|s| s.name == "getc_like").expect("declared");
    assert!(!sig.defined, "calls bind to the external symbol");
}

/// A `va_list` built by an lf-cc variadic function is handed to the host
/// libc's `vsnprintf` (the gcc-compatible SysV representation).
#[test]
fn builtin_va_list_crosses_into_host_libc() {
    if HostCrt::discover().is_none() {
        eprintln!("skipping: no host C runtime (crt1.o) found");
        return;
    }
    let dir = scratch("valist");
    let src = r#"
typedef __builtin_va_list va_list;
typedef unsigned long size_t;
int vsnprintf(char *, size_t, const char *, va_list);
int printf(const char *, ...);
static int fmt(char *buf, size_t n, const char *f, ...) {
    va_list ap;
    __builtin_va_start(ap, f);
    int r = vsnprintf(buf, n, f, ap);
    __builtin_va_end(ap);
    return r;
}
static int twice(char *buf, size_t n, const char *f, ...) {
    va_list ap, aq;
    __builtin_va_start(ap, f);
    __builtin_va_copy(aq, ap);
    int r = vsnprintf(buf, n, f, ap);
    r += vsnprintf(buf + r, n - r, f, aq);
    __builtin_va_end(aq);
    __builtin_va_end(ap);
    return r;
}
int main(void) {
    char b[128];
    int r = fmt(b, sizeof b, "%d-%s-%.2f-%ld-%c", 42, "xy", 2.5, 123456789012L, 'q');
    printf("%s|%d\n", b, r);
    r = twice(b, sizeof b, "[%u %x %g]", 7u, 255, 0.5);
    printf("%s|%d\n", b, r);
    return 0;
}
"#;
    let c = dir.join("valist.c");
    std::fs::write(&c, src).expect("write source");
    let exe = dir.join("valist");
    let out = Command::new(env!("CARGO_BIN_EXE_lf-cc"))
        .arg(&c)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("run lf-cc");
    assert!(out.status.success(), "lf-cc failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&exe).output().expect("run program");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "42-xy-2.50-123456789012-q|25\n[7 ff 0.5][7 ff 0.5]|20\n"
    );
}
