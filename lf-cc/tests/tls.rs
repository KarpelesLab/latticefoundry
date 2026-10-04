//! Thread-local storage: `__thread` (GNU), `_Thread_local` (C11) and C23
//! `thread_local` lower to IR `thread_local` globals, emitted in `.tdata` /
//! `.tbss` with `STT_TLS` symbols and addressed through the thread pointer by
//! the access model the relocation model selects (local-exec, initial-exec,
//! or general-dynamic in a shared library).
//!
//! The hosted programs are differential against gcc: a multi-threaded program
//! where every thread mutates its own copies, `static` and `extern`
//! thread-locals across two translation units (also mixing lf-cc and gcc
//! objects both ways), position-independent executables, and an lf-cc shared
//! library with thread-locals that a program `dlopen`s. They skip without the
//! host C runtime or gcc.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use latticefoundry::link::gnu::HostCrt;
use latticefoundry::link::write_executable;
use latticefoundry::mc::object::{SectionKind, SymbolType, SymbolValue};
use latticefoundry::transform::pipeline::OptLevel;
use lf_cc::PpOptions;

const LF_CC: &str = env!("CARGO_BIN_EXE_lf-cc");

/// A per-test scratch directory, removed when the test finishes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lf-cc-tls-{tag}-{}", std::process::id()));
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

    /// Run executable `exe` (in this directory) with `args`; its stdout.
    fn run(&self, exe: &str, args: &[&str]) -> String {
        let path = self.0.join(exe);
        for _ in 0..50 {
            match Command::new(&path).args(args).output() {
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

/// Four threads each mutate their own copies of initialized, zeroed, `static`,
/// block-scope `static`, array, struct and pointer thread-locals (one holding
/// the address of an ordinary global), and one defined in the other unit.
const MAIN_C: &str = r#"#include <pthread.h>
#include <stdio.h>

__thread long counter = 100;
static __thread int hits;
_Thread_local char tag[8] = "main";
__thread int *self_ptr;
extern __thread int shared_tls;
int bump_shared(int by);
int *shared_addr(void);
struct pt { int x, y; };
__thread struct pt pts[2] = { { 1, 2 }, { 3, 4 } };
static int plain = 5;
__thread int *to_plain = &plain;

static long work(long id) {
    static __thread int calls;
    for (int i = 0; i < 1000; i++) {
        counter += id;
        hits++;
        calls++;
    }
    tag[0] = (char)('a' + id);
    self_ptr = &hits;
    pts[1].y += (int)id;
    shared_tls += (int)id;
    return counter + hits + calls + bump_shared((int)id) + (self_ptr == &hits) + pts[1].y + *to_plain
        + (shared_addr() == &shared_tls) * 1000000 + tag[0];
}

static void *thread_main(void *arg) {
    return (void *)work((long)arg);
}

int main(void) {
    pthread_t t[4];
    for (long i = 0; i < 4; i++) pthread_create(&t[i], 0, thread_main, (void *)(i + 1));
    for (int i = 0; i < 4; i++) {
        void *r;
        pthread_join(t[i], &r);
        printf("thread %d -> %ld\n", i + 1, (long)r);
    }
    printf("main counter=%ld hits=%d tag=%s shared=%d pts=%d\n", counter, hits, tag, bump_shared(0), pts[1].y);
    return 0;
}
"#;

const OTHER_C: &str = r#"__thread int shared_tls = 7;
int bump_shared(int by) {
    extern __thread int shared_tls;
    shared_tls += by * 10;
    return shared_tls;
}
int *shared_addr(void) { return &shared_tls; }
"#;

const THREADS_OUT: &str = "thread 1 -> 1003227\n\
thread 2 -> 1004240\n\
thread 3 -> 1005253\n\
thread 4 -> 1006266\n\
main counter=100 hits=0 tag=main shared=7 pts=4\n";

#[test]
fn threads_have_their_own_copies() {
    let Some(gcc) = host_gcc() else { return };
    let s = Scratch::new("threads");
    s.file("main.c", MAIN_C);
    s.file("other.c", OTHER_C);
    let std = common::gcc_std_flag(&gcc, "gnu17");
    s.gcc(&gcc, &[&std, "-O2", "main.c", "other.c", "-o", "ref", "-pthread"]);
    assert_eq!(s.run("ref", &[]), THREADS_OUT, "gcc");
    for flags in [&["-O0"][..], &["-O2"], &["-O2", "-fPIE", "-pie"], &["-O0", "-fPIE", "-pie"]] {
        let pie = flags.contains(&"-pie");
        let mut args = flags.to_vec();
        args.extend(["main.c", "other.c", "-o", "lf", "-pthread"]);
        s.lf_cc(&args);
        assert_eq!(s.run("lf", &[]), THREADS_OUT, "lf-cc {flags:?}");
        // Mixed objects, both ways: lf-cc's thread-locals are gcc's.
        let cflags = &flags[..if pie { 2 } else { 1 }];
        let link = if pie { "-pie" } else { "-no-pie" };
        let mut args = cflags.to_vec();
        args.extend(["-c", "main.c", "-o", "main_lf.o"]);
        s.lf_cc(&args);
        let mut args = cflags.to_vec();
        args.extend(["-c", "other.c", "-o", "other_lf.o"]);
        s.lf_cc(&args);
        let mut args = cflags.to_vec();
        args.extend(["-c", "main.c", "-o", "main_gcc.o"]);
        s.gcc(&gcc, &args);
        let mut args = cflags.to_vec();
        args.extend(["-c", "other.c", "-o", "other_gcc.o"]);
        s.gcc(&gcc, &args);
        s.gcc(&gcc, &[link, "main_lf.o", "other_gcc.o", "-o", "mix1", "-pthread"]);
        assert_eq!(s.run("mix1", &[]), THREADS_OUT, "lf-cc main, gcc other {flags:?}");
        let mut args = vec!["main_gcc.o", "other_lf.o", "-o", "mix2", "-pthread"];
        if pie {
            args.push("-pie");
        }
        s.lf_cc(&args);
        assert_eq!(s.run("mix2", &[]), THREADS_OUT, "gcc main, lf-cc other {flags:?}");
    }
}

/// A shared library's thread-locals (general-dynamic: `__tls_get_addr`),
/// per thread in a program that `dlopen`s it.
const LIB_C: &str = r#"__thread long lib_counter = 40;
static __thread int lib_hidden;
long lib_bump(long by) {
    lib_counter += by;
    lib_hidden++;
    return lib_counter * 100 + lib_hidden;
}
long *lib_addr(void) { return &lib_counter; }
"#;

const HOST_C: &str = r#"#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>

static long (*bump)(long);
static long *(*addr)(void);

static void *run(void *arg) {
    long id = (long)arg, r = 0;
    for (int i = 0; i < 100; i++) r = bump(id);
    return (void *)(r + (addr() != 0));
}

int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 0; }
    bump = (long (*)(long))dlsym(h, "lib_bump");
    addr = (long *(*)(void))dlsym(h, "lib_addr");
    pthread_t t[3];
    for (long i = 0; i < 3; i++) pthread_create(&t[i], 0, run, (void *)(i + 1));
    for (int i = 0; i < 3; i++) {
        void *r;
        pthread_join(t[i], &r);
        printf("t%d %ld\n", i, (long)r);
    }
    printf("main %ld %ld\n", bump(0), *addr());
    return 0;
}
"#;

const HOST_OUT: &str = "t0 14101\nt1 24101\nt2 34101\nmain 4001 40\n";

#[test]
fn a_shared_library_with_thread_locals_is_dlopened() {
    let Some(gcc) = host_gcc() else { return };
    let s = Scratch::new("shared");
    s.file("lib.c", LIB_C);
    s.file("host.c", HOST_C);
    s.gcc(&gcc, &["-O2", "host.c", "-o", "host_gcc", "-pthread", "-ldl"]);
    s.lf_cc(&["host.c", "-o", "host_lf", "-pthread", "-ldl"]);
    for opt in ["-O0", "-O2"] {
        s.lf_cc(&[opt, "-shared", "lib.c", "-o", "libtls.so"]);
        let so = s.0.join("libtls.so");
        let so = so.to_str().unwrap();
        assert_eq!(s.run("host_gcc", &[so]), HOST_OUT, "gcc host, lf-cc {opt} library");
        assert_eq!(s.run("host_lf", &[so]), HOST_OUT, "lf-cc host, lf-cc {opt} library");
    }
    // The same through a `-fPIC` object.
    s.lf_cc(&["-fPIC", "-c", "lib.c", "-o", "lib.o"]);
    s.lf_cc(&["-shared", "lib.o", "-o", "libobj.so"]);
    let so = s.0.join("libobj.so");
    assert_eq!(s.run("host_gcc", &[so.to_str().unwrap()]), HOST_OUT);
}

/// The three spellings: `__thread`, `_Thread_local`, and C23 `thread_local`
/// (a keyword only from C23).
#[test]
fn every_spelling_declares_a_thread_local() {
    let ir = |src: &str, std: &str| {
        let opts = PpOptions { std: lf_cc::CStd::parse(std).unwrap(), ..PpOptions::default() };
        let (module, syms) = lf_cc::compile_to_ir_with(src, "t.c", &opts, false).expect("compiles");
        latticefoundry::ir::text::print_module(&module, &syms)
    };
    let text = ir(
        "__thread int a; _Thread_local int b = 1; static __thread int c;\n\
         extern _Thread_local int d;\n\
         int f(void) { static _Thread_local int e; extern __thread int g; return a + b + c + d + e++ + g; }\n",
        "gnu17",
    );
    for name in ["a", "b", "c", "d", "e.static", "g"] {
        let line = text.lines().find(|l| l.starts_with("global") && l.contains(&format!("@{name}")));
        let line = line.unwrap_or_else(|| panic!("no global @{name} in:\n{text}"));
        assert!(line.contains("thread_local"), "{line}");
    }
    let text = ir("thread_local int z = 1; int g(void) { static thread_local int w; return z + w++; }", "c23");
    assert_eq!(text.lines().filter(|l| l.starts_with("global") && l.contains("thread_local")).count(), 2, "{text}");
}

/// The object: initialized thread-locals in `.tdata`, zeroed ones in
/// `.tbss`, every thread-local symbol (definitions and references) `STT_TLS`.
#[test]
fn thread_locals_go_to_the_tls_sections() {
    let src = "__thread long a = 5; __thread int z; static _Thread_local char buf[64];\n\
               extern __thread int ext; int plain = 1;\n\
               long f(void) { buf[1]++; return a + z + ext + buf[1] + plain; }\n";
    let unit = lf_cc::compile_module_with(src, "t.c", &PpOptions::default(), OptLevel::O0, false).expect("compiles");
    let obj = unit.module;
    let kind_of = |name: &str| {
        let id = obj.symbol_id(name).unwrap_or_else(|| panic!("no symbol {name}"));
        let sym = obj.symbol(id);
        let section = match sym.value {
            SymbolValue::Defined { section, .. } => Some(obj.section(section).kind),
            _ => None,
        };
        (sym.kind, section)
    };
    assert_eq!(kind_of("a"), (SymbolType::Tls, Some(SectionKind::TData)));
    assert_eq!(kind_of("z"), (SymbolType::Tls, Some(SectionKind::TBss)));
    assert_eq!(kind_of("ext"), (SymbolType::Tls, None));
    assert_eq!(kind_of("plain").0, SymbolType::Object);
    let tbss = obj.sections().iter().find(|s| s.kind == SectionKind::TBss).expect("a .tbss");
    assert!(tbss.size() >= 68, "z and buf: {}", tbss.size());
}

/// A static image (our own `_start`, which builds the TLS block itself) runs
/// with thread-locals too.
#[test]
fn static_images_set_up_the_tls_block() {
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return;
    }
    let src = "__thread int x = 5; __thread long big[4] = { 1, 2, 3, 4 }; static __thread int z;\n\
               int main(void) { x += 2; z += big[3]; return x * 10 + z; }\n";
    let dir = std::env::temp_dir().join(format!("lf-cc-tls-static-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create dir");
    for opt in [OptLevel::O0, OptLevel::O2] {
        let image = lf_cc::build_image(src, "t.c", opt, false).expect("builds");
        let bin = dir.join(format!("t.{}", opt.name()));
        write_executable(bin.to_str().unwrap(), &image).expect("write executable");
        let status = Command::new(&bin).status().expect("run");
        assert_eq!(status.code(), Some(74), "{}", opt.name());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Compile `src` through sema and return the diagnostics' text (empty on success).
fn check_errors(src: &str) -> String {
    match lf_cc::check_source(src) {
        Ok(_) => String::new(),
        Err(diags) => diags.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>().join("\n"),
    }
}

#[test]
fn thread_local_constraints_are_enforced() {
    for (src, needle) in [
        ("int f(void) { __thread int x; return x; }", "must be 'static' or 'extern'"),
        ("int f(void) { _Thread_local int x = 1; return x; }", "must be 'static' or 'extern'"),
        ("__thread int x; int *p = &x;", "is not a constant"),
        ("__thread int a[4]; int *p = a;", "is not a constant"),
        ("struct S { int m; }; __thread struct S s; int *p = &s.m;", "is not a constant"),
        ("int x; __thread int x;", "thread-local declaration of 'x' follows a non-thread-local"),
        ("__thread int x; extern int x;", "non-thread-local declaration of 'x' follows a thread-local"),
        ("__thread int x; int f(void) { extern int x; return x; }", "follows a thread-local"),
        ("__thread int f(void);", "function 'f' declared thread-local"),
        ("int g(void) { extern __thread int f(void); return 0; }", "function 'f' declared thread-local"),
        ("typedef __thread int T;", "a typedef cannot be thread-local"),
    ] {
        let errs = check_errors(src);
        assert!(errs.contains(needle), "expected '{needle}' for:\n{src}\ngot: {errs}");
    }
    // Taking the address at run time is fine.
    assert_eq!(check_errors("__thread int x; int *f(void) { static int y; int *p = &x; return p ? p : &y; }"), "");
}
