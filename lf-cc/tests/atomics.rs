//! `volatile` and C11 atomics: volatile accesses lower to the IR's `load
//! volatile`/`store volatile` and survive optimization; `_Atomic` objects are
//! accessed with `seq_cst` atomics (compound assignment as `atomic_rmw`); the
//! builtin `<stdatomic.h>`, the GNU `__atomic_*` builtins and the legacy
//! `__sync_*` ones lower to `atomic_load`/`atomic_store`/`atomic_rmw`/`cmpxchg`/
//! `fence`. The programs run against gcc's results (when gcc is installed),
//! including a two-thread program that must lose no increment.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use latticefoundry::link::gnu::HostCrt;
use latticefoundry::transform::pipeline::{self, OptLevel};

const LF_CC: &str = env!("CARGO_BIN_EXE_lf-cc");

/// The optimized (`-O2`) IR of `src`, as text.
fn optimized_ir(src: &str) -> String {
    let (mut module, syms) = lf_cc::compile_to_ir(src, "t.c", false).expect("compiles");
    pipeline::optimize(&mut module, OptLevel::O2);
    latticefoundry::ir::text::print_module(&module, &syms)
}

/// The text of function `name` in printed IR.
fn function_text<'a>(ir: &'a str, name: &str) -> &'a str {
    let start = ir.find(&format!("@{name}(")).unwrap_or_else(|| panic!("no function {name} in:\n{ir}"));
    let rest = &ir[start..];
    let end = rest[1..].find("\nfunc ").map_or(rest.len(), |e| e + 1);
    &rest[..end]
}

#[test]
fn volatile_accesses_survive_optimization() {
    let src = "int three(volatile int *p) { return *p + *p + *p; }\n\
               int three_plain(int *p) { return *p + *p + *p; }\n\
               void two_stores(volatile int *p) { *p = 1; *p = 2; }\n\
               int unused_plain(void) { int local = 5; local; local = 6; return local; }\n\
               int poll(volatile int *p, int n) { int s = 0; for (int i = 0; i < n; i++) s += *p; return s; }\n\
               int unused(void) { volatile int local = 5; local; local = 6; return 0; }\n\
               struct S { volatile unsigned bits : 3; volatile char c; };\n\
               int bf(struct S *s) { s->bits = 5; return s->bits + s->c + s->c; }\n";
    let ir = optimized_ir(src);
    let count = |f: &str, what: &str| function_text(&ir, f).matches(what).count();
    assert_eq!(count("three", "load volatile"), 3, "{ir}");
    assert_eq!(count("two_stores", "store volatile"), 2, "{ir}");
    // The same local without `volatile` is promoted to a register (no memory
    // access survives), so the counts below are the qualifier's doing.
    assert_eq!(count("unused_plain", "load"), 0, "{ir}");
    assert_eq!(count("unused_plain", "store"), 0, "{ir}");
    // The load in the loop is not hoisted out of it: the entry block holds none.
    let poll = function_text(&ir, "poll");
    assert_eq!(poll.matches("load volatile").count(), 1, "{poll}");
    let entry = poll.split("\n^").next().unwrap();
    assert!(!entry.contains("load volatile"), "{poll}");
    // A volatile local stays in memory, every access performed.
    assert_eq!(count("unused", "load volatile"), 1, "{ir}");
    assert_eq!(count("unused", "store volatile"), 2, "{ir}");
    // Bit-field and member accesses of a volatile member are volatile too.
    assert_eq!(count("bf", "load volatile"), 4, "{ir}");
    assert_eq!(count("bf", "store volatile"), 1, "{ir}");
}

#[test]
fn atomic_objects_use_seq_cst_atomics() {
    let src = "_Atomic int a;\n\
               _Atomic(long) b;\n\
               int *_Atomic p;\n\
               _Atomic double d;\n\
               int load(void) { return a; }\n\
               void store(int v) { a = v; }\n\
               int add(int v) { return a += v; }\n\
               long inc(void) { return b++; }\n\
               int mul(int v) { return a *= v; }\n\
               int *step(void) { return p += 2; }\n\
               double fadd(double v) { return d += v; }\n";
    let ir = optimized_ir(src);
    let f = |name: &str| function_text(&ir, name).to_owned();
    assert!(f("load").contains("atomic_load seq_cst"), "{ir}");
    assert!(f("store").contains("atomic_store seq_cst"), "{ir}");
    assert!(f("add").contains("atomic_rmw add seq_cst"), "{ir}");
    assert!(f("inc").contains("atomic_rmw add seq_cst"), "{ir}");
    // No single instruction multiplies atomically: a compare-exchange loop.
    assert!(f("mul").contains("cmpxchg seq_cst seq_cst"), "{ir}");
    assert!(f("step").contains("atomic_rmw add seq_cst"), "{ir}");
    assert!(f("fadd").contains("cmpxchg seq_cst seq_cst"), "{ir}");
}

#[test]
fn atomic_builtins_lower_to_the_ir_atomics() {
    let src = "int x; long y; int *q;\n\
               int ld(void) { return __atomic_load_n(&x, __ATOMIC_ACQUIRE); }\n\
               void st(int v) { __atomic_store_n(&x, v, __ATOMIC_RELEASE); }\n\
               int fa(int v) { return __atomic_fetch_add(&x, v, __ATOMIC_RELAXED); }\n\
               long nf(long v) { return __atomic_nand_fetch(&y, v, __ATOMIC_ACQ_REL); }\n\
               int cas(int *e, int d) { return __atomic_compare_exchange_n(&x, e, d, 1, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE); }\n\
               int sv(int o, int n) { return __sync_val_compare_and_swap(&x, o, n); }\n\
               void fence(void) { __atomic_thread_fence(__ATOMIC_ACQUIRE); __sync_synchronize(); }\n\
               int *xq(int *n) { return __atomic_exchange_n(&q, n, __ATOMIC_SEQ_CST); }\n";
    let ir = optimized_ir(src);
    let f = |name: &str| function_text(&ir, name).to_owned();
    assert!(f("ld").contains("atomic_load acquire"), "{ir}");
    assert!(f("st").contains("atomic_store release"), "{ir}");
    assert!(f("fa").contains("atomic_rmw add relaxed"), "{ir}");
    assert!(f("nf").contains("atomic_rmw nand acq_rel"), "{ir}");
    // A weak compare-exchange is implemented by the strong one.
    assert!(f("cas").contains("cmpxchg seq_cst acquire"), "{ir}");
    assert!(f("sv").contains("cmpxchg seq_cst seq_cst"), "{ir}");
    assert!(f("fence").contains("fence acquire") && f("fence").contains("fence seq_cst"), "{ir}");
    assert!(f("xq").contains("atomic_rmw xchg seq_cst"), "{ir}");
}

#[test]
fn atomic_misuse_is_diagnosed() {
    for (src, needle) in [
        ("struct S { int a, b; }; _Atomic struct S s; int main(void) { return 0; }", "_Atomic"),
        ("int main(void) { struct { char c[3]; } v; return __atomic_load_n(&v, 5).c[0]; }", "1, 2, 4 or 8 bytes"),
        ("int main(void) { double d; return (int)__atomic_fetch_add(&d, 1, 5); }", "integer or pointer"),
        ("int main(void) { int x; return __atomic_frobnicate(&x); }", "unknown atomic builtin"),
    ] {
        let errs = match lf_cc::check_source(src) {
            Ok(_) => String::new(),
            Err(d) => d.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"),
        };
        assert!(errs.contains(needle), "expected '{needle}' for {src}: {errs:?}");
    }
}

/// A per-test scratch directory, removed when the test finishes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lf-cc-atomics-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
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

/// Build `src` with lf-cc (`-std=<std>`, at -O0 and -O2) and, when installed,
/// gcc; every build must print `expected`.
fn check_program(name: &str, std: &str, src: &str, expected: &str, extra: &[&str]) {
    if HostCrt::discover().is_none() {
        eprintln!("skipping: no host C runtime (crt1.o) found");
        return;
    }
    let s = Scratch::new(name);
    let c = s.0.join(format!("{name}.c"));
    std::fs::write(&c, src).unwrap();
    let mut failures = Vec::new();
    for opt in ["-O0", "-O2"] {
        let exe = s.0.join(format!("{name}{opt}"));
        let std_flag = format!("-std={std}");
        let out = Command::new(LF_CC)
            .arg(&std_flag)
            .arg(opt)
            .arg(&c)
            .arg("-o")
            .arg(&exe)
            .args(extra)
            .output()
            .expect("run lf-cc");
        if !out.status.success() {
            failures.push(format!("lf-cc {opt}: {}", String::from_utf8_lossy(&out.stderr)));
            continue;
        }
        let got = String::from_utf8_lossy(&run(&exe).stdout).into_owned();
        if got != expected {
            failures.push(format!("lf-cc {opt} printed:\n{got}"));
        }
    }
    if let Some(gcc) = which("gcc") {
        let exe = s.0.join(format!("{name}-gcc"));
        let out = Command::new(&gcc)
            .arg(common::gcc_std_flag(&gcc, std))
            .arg("-w")
            .arg(&c)
            .arg("-o")
            .arg(&exe)
            .args(extra)
            .output()
            .expect("run gcc");
        assert!(out.status.success(), "gcc: {}", String::from_utf8_lossy(&out.stderr));
        let got = String::from_utf8_lossy(&run(&exe).stdout).into_owned();
        if got != expected {
            failures.push(format!("gcc printed:\n{got}"));
        }
    }
    assert!(failures.is_empty(), "{name}: expected\n{expected}\n{}", failures.join("\n"));
}

/// Two threads hammer shared counters through every flavour of atomic access
/// (C11 generic functions with explicit orders, `_Atomic` compound assignment
/// and `++`, `__atomic_*`, `__sync_*`, an `atomic_flag` spin lock guarding a
/// plain counter, and a compare-exchange max loop). No update may be lost.
const THREADS_C: &str = r#"
#include <stdatomic.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>

#define N 200000
static atomic_int counter;
static atomic_long total = ATOMIC_VAR_INIT(0);
static _Atomic unsigned short wrap16 = 65530;
static int plain;
static atomic_flag lock = ATOMIC_FLAG_INIT;
static int guarded;
static atomic_uintptr_t pmax;
static long sync_counter;

static void *worker(void *arg) {
    long id = (long)arg;
    for (int i = 0; i < N; i++) {
        atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
        total += id;
        wrap16++;
        __atomic_fetch_add(&plain, 1, __ATOMIC_SEQ_CST);
        __sync_fetch_and_add(&sync_counter, 2);
        while (atomic_flag_test_and_set_explicit(&lock, memory_order_acquire)) { }
        guarded++;
        atomic_flag_clear_explicit(&lock, memory_order_release);
        uintptr_t cur = atomic_load(&pmax);
        while (cur < (uintptr_t)i && !atomic_compare_exchange_weak(&pmax, &cur, (uintptr_t)i)) { }
    }
    return NULL;
}

int main(void) {
    pthread_t t[2];
    for (long i = 0; i < 2; i++) pthread_create(&t[i], NULL, worker, (void *)(i + 1));
    for (int i = 0; i < 2; i++) pthread_join(t[i], NULL);
    printf("counter=%d total=%ld wrap16=%u plain=%d sync=%ld guarded=%d pmax=%lu\n",
           atomic_load(&counter), atomic_load(&total), (unsigned)wrap16, plain, sync_counter, guarded,
           (unsigned long)atomic_load(&pmax));
    return 0;
}
"#;

#[test]
fn two_threads_lose_no_atomic_update() {
    check_program(
        "threads",
        "c11",
        THREADS_C,
        "counter=400000 total=600000 wrap16=6778 plain=400000 sync=800000 guarded=400000 pmax=199999\n",
        &["-pthread"],
    );
}

/// The single-threaded results of the builtins: values returned, values left
/// behind, and the expected-value write-back of a failed compare-exchange.
const BUILTINS_C: &str = r#"
#include <stdatomic.h>
#include <stdio.h>

#define P(fmt, v) printf(fmt " ", v)

int main(void) {
    int x = 5, e = 5;
    P("%d", __atomic_exchange_n(&x, 9, __ATOMIC_ACQ_REL));
    P("%d", x);
    P("%d", __atomic_compare_exchange_n(&x, &e, 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED));
    P("%d", e);
    P("%d", __atomic_compare_exchange_n(&x, &e, 1, 1, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE));
    P("%d", x);
    P("%d", __sync_val_compare_and_swap(&x, 1, 3));
    P("%d", __sync_bool_compare_and_swap(&x, 3, 4));
    P("%d", __sync_bool_compare_and_swap(&x, 3, 5));
    P("%d", __atomic_add_fetch(&x, 10, 0));
    P("%d", __atomic_nand_fetch(&x, 6, 5));
    P("%d", __sync_fetch_and_xor(&x, 0xff));
    P("%d", __sync_or_and_fetch(&x, 0x100));
    P("%d", __atomic_fetch_and(&x, 0x1f0, __ATOMIC_CONSUME));
    P("%d", __atomic_sub_fetch(&x, 1000, __ATOMIC_RELAXED));
    printf("%d\n", x);
    unsigned char ub = 250;
    P("%d", __atomic_fetch_add(&ub, 10, 5));
    printf("%d\n", ub);
    long arr[4] = {1, 2, 3, 4};
    long *p = arr;
    long *old = __atomic_fetch_add(&p, sizeof(long), __ATOMIC_SEQ_CST);
    printf("%ld %ld\n", *old, *p);
    _Atomic(long *) ap = arr;
    ap += 2;
    P("%ld", *ap);
    ap--;
    printf("%ld\n", *ap);
    double d = 1.25, d2;
    __atomic_store(&d2, &d, __ATOMIC_RELEASE);
    double d3;
    __atomic_load(&d2, &d3, __ATOMIC_ACQUIRE);
    double nd = 7.5, od;
    __atomic_exchange(&d2, &nd, &od, __ATOMIC_SEQ_CST);
    printf("%g %g %g %d\n", d3, od, d2, __atomic_always_lock_free(sizeof(long), 0));
    atomic_bool fl = 0;
    P("%d", atomic_exchange(&fl, 1));
    P("%d", atomic_load(&fl));
    printf("%d\n", atomic_is_lock_free(&fl));
    atomic_thread_fence(memory_order_seq_cst);
    atomic_signal_fence(memory_order_acquire);
    __sync_synchronize();
    int lk = 0;
    P("%d", __sync_lock_test_and_set(&lk, 1));
    P("%d", lk);
    __sync_lock_release(&lk);
    printf("%d\n", lk);
    atomic_int ai;
    atomic_init(&ai, 41);
    int expect = 41;
    P("%d", atomic_compare_exchange_strong_explicit(&ai, &expect, 42, memory_order_acq_rel, memory_order_relaxed));
    P("%d", atomic_fetch_sub(&ai, 2));
    P("%d", atomic_fetch_or(&ai, 1));
    P("%d", atomic_fetch_xor_explicit(&ai, 3, memory_order_release));
    P("%d", atomic_fetch_and(&ai, ~1));
    printf("%d\n", atomic_load_explicit(&ai, memory_order_acquire));
    _Atomic short s = 7;
    s *= 3; s <<= 2; s -= 100; s %= 7; s |= 0x40; s >>= 1;
    _Atomic unsigned char uc = 200;
    uc += 100;
    uc++;
    _Atomic _Bool b = 0;
    b++;
    b++;
    printf("%d %d %d\n", s, uc, b);
    return 0;
}
"#;

#[test]
fn atomic_builtins_match_gcc() {
    let expected = "5 9 0 9 1 1 1 1 0 14 -7 -7 -250 -250 -744 -744\n\
                    250 4\n\
                    1 2\n\
                    3 2\n\
                    1.25 1.25 7.5 1\n\
                    0 1 1\n\
                    0 1 0\n\
                    1 42 40 41 42 42\n\
                    -1 45 1\n";
    for std in ["c11", "gnu17", "c23"] {
        check_program(&format!("builtins-{std}"), std, BUILTINS_C, expected, &[]);
    }
}

/// `volatile` objects of every kind, read and written (results compared with
/// gcc); `_Atomic` floating-point compound assignment (lf-cc only: gcc calls
/// libatomic for it).
#[test]
fn volatile_and_atomic_objects_compute_like_gcc() {
    let src = r#"
#include <stdio.h>
volatile int vg = 3;
struct S { volatile int a; int b : 4; volatile unsigned c : 3; };
volatile struct S gs = {1, 2, 3};
int poll(volatile int *p, int n) { int s = 0; for (int i = 0; i < n; i++) s += *p; return s; }
int main(void) {
    volatile int local = 5;
    volatile char buf[4] = {1, 2, 3, 4};
    struct S s = {1, 2, 3};
    volatile struct S vs = {4, 5, 6};
    int *volatile vp = (int *)&local;
    s.c++;
    vs.a += 2;
    vs.c = vs.c + 9;
    local *= 3;
    buf[2] += buf[1];
    gs.b = -3;
    printf("%d %d %d %d %d %d %d %d %d\n", poll(&vg, 4), local, s.c, vs.a, vs.c, buf[2], gs.b, gs.c, *vp);
    return 0;
}
"#;
    check_program("volatile", "gnu17", src, "12 15 4 6 7 5 -3 3 15\n", &[]);
    let atomic_float = r#"
#include <stdio.h>
_Atomic double ad = 1.5;
_Atomic float af;
int main(void) {
    ad += 2.25;
    ad *= 2;
    af = 0.5f;
    af++;
    af /= 4;
    printf("%g %g\n", ad, af);
    return 0;
}
"#;
    if HostCrt::discover().is_some() {
        let s = Scratch::new("afloat");
        let c = s.0.join("af.c");
        std::fs::write(&c, atomic_float).unwrap();
        for opt in ["-O0", "-O2"] {
            let exe = s.0.join(format!("af{opt}"));
            let st = Command::new(LF_CC).arg(opt).arg(&c).arg("-o").arg(&exe).status().unwrap();
            assert!(st.success());
            assert_eq!(String::from_utf8_lossy(&run(&exe).stdout), "7.5 0.375\n");
        }
    }
}

/// `<stdatomic.h>` compiles under every C11-and-later standard, also with
/// `-ffreestanding`, and the macros it relies on are predefined.
#[test]
fn stdatomic_header_compiles() {
    let src = "#include <stdatomic.h>\n\
               #if !defined __ATOMIC_SEQ_CST || ATOMIC_INT_LOCK_FREE != 2 || !__has_builtin(__atomic_load_n)\n\
               #error atomics model\n\
               #endif\n\
               atomic_int a = ATOMIC_VAR_INIT(1); atomic_flag f = ATOMIC_FLAG_INIT; memory_order mo = memory_order_acquire;\n\
               atomic_ullong u; atomic_size_t z; atomic_intptr_t ip; atomic_char16_t c16; atomic_uint_fast32_t uf;\n\
               int main(void) { atomic_store_explicit(&a, 2, memory_order_release);\n\
                 return atomic_load_explicit(&a, mo) + atomic_flag_test_and_set(&f) + (int)kill_dependency(u) - 2; }\n";
    for std in ["c11", "c17", "c23", "gnu11", "gnu17"] {
        for hosted in [true, false] {
            let opts = lf_cc::PpOptions {
                std: lf_cc::CStd::parse(std).unwrap(),
                hosted,
                ..lf_cc::PpOptions::default()
            };
            if let Err(d) = lf_cc::check_source_with(src, &opts) {
                panic!("<stdatomic.h> under {std} (hosted={hosted}): {d:?}");
            }
        }
    }
}
