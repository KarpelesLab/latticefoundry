//! End-to-end tests of the `lf-cc` compiler driver binary: multiple inputs,
//! per-file `-c`, hosted linking against the host C library through `qld`
//! (no gcc or system linker anywhere), `-l` libraries, and the freestanding
//! `-nostdlib` link.
//!
//! The hosted tests need the host C runtime (`crt1.o`, `libc.so`); they skip
//! with a message when [`HostCrt::discover`] finds none.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use latticefoundry::link::gnu::HostCrt;

const LF_CC: &str = env!("CARGO_BIN_EXE_lf-cc");

/// A per-test scratch directory, removed when the test finishes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lf-cc-drvtest-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn write(&self, name: &str, text: &str) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, text).expect("write source");
        p
    }

    /// Run `lf-cc` with `args` in this directory; panic with its stderr on failure.
    fn lf_cc(&self, args: &[&str]) {
        let out = Command::new(LF_CC).args(args).current_dir(&self.0).output().expect("run lf-cc");
        assert!(
            out.status.success(),
            "lf-cc {args:?} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// Run `lf-cc` with `args`, expecting failure; return its stderr.
    fn lf_cc_fails(&self, args: &[&str]) -> String {
        let out = Command::new(LF_CC).args(args).current_dir(&self.0).output().expect("run lf-cc");
        assert!(!out.status.success(), "lf-cc {args:?} unexpectedly succeeded");
        String::from_utf8_lossy(&out.stderr).into_owned()
    }

    fn run(&self, exe: &str) -> Output {
        run_retrying(&self.0.join(exe))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Execute a freshly written binary, retrying briefly on `ETXTBSY` (another
/// test thread's fork may transiently hold a writable fd to it).
fn run_retrying(exe: &Path) -> Output {
    for _ in 0..50 {
        match Command::new(exe).output() {
            Err(e) if e.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            other => return other.expect("run executable"),
        }
    }
    panic!("{} stayed busy (ETXTBSY)", exe.display());
}

fn host_crt() -> Option<HostCrt> {
    let crt = HostCrt::discover();
    if crt.is_none() {
        eprintln!("skipping: no host C runtime (crt1.o) found");
    }
    crt
}

const MAIN_C: &str = r#"
int printf(const char *fmt, ...);
unsigned long strlen(const char *s);
void *malloc(unsigned long n);
void free(void *p);
int twice(int x);
extern int counter;

int main(void) {
    char *p = malloc(32);
    const char *msg = "hello, libc";
    unsigned long i;
    for (i = 0; msg[i]; i++) p[i] = msg[i];
    p[i] = 0;
    printf("%s len=%lu twice=%d counter=%d\n", p, strlen(p), twice(21), counter);
    free(p);
    return twice(3) + counter;
}
"#;

const UTIL_C: &str = r#"
int counter = 5;
static int helper(int x) { return x + x; }
int twice(int x) { counter++; return helper(x); }
"#;

#[test]
fn hosted_multi_file_program_links_against_libc() {
    if host_crt().is_none() {
        return;
    }
    let s = Scratch::new("multi");
    s.write("main.c", MAIN_C);
    s.write("util.c", UTIL_C);
    s.lf_cc(&["-Wall", "-O2", "-pipe", "main.c", "util.c", "-o", "prog"]);
    let out = s.run("prog");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hello, libc len=11 twice=42 counter=6\n");
    // twice(3) = 6, counter is then 7.
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn compiled_object_mixes_with_c_source() {
    if host_crt().is_none() {
        return;
    }
    let s = Scratch::new("mix");
    s.write("main.c", MAIN_C);
    s.write("util.c", UTIL_C);
    // `-c` names the object after the source by default.
    s.lf_cc(&["-c", "util.c"]);
    assert!(s.0.join("util.o").is_file());
    s.lf_cc(&["main.c", "util.o", "-o", "prog"]);
    let out = s.run("prog");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hello, libc len=11 twice=42 counter=6\n");
    assert_eq!(out.status.code(), Some(13));
}

#[test]
fn multi_input_compile_writes_one_object_each() {
    let s = Scratch::new("multic");
    s.write("main.c", MAIN_C);
    s.write("util.c", UTIL_C);
    s.lf_cc(&["-c", "main.c", "util.c"]);
    assert!(s.0.join("main.o").is_file() && s.0.join("util.o").is_file());
    let err = s.lf_cc_fails(&["-c", "main.c", "util.c", "-o", "x.o"]);
    assert!(err.contains("multiple source files"), "{err}");
    assert!(!s.0.join("x.o").exists());
    let err = s.lf_cc_fails(&["-c", "-funsigned-char", "util.c"]);
    assert!(err.contains("unrecognized option"), "{err}");
}

#[test]
fn links_libm_with_dash_l() {
    if host_crt().is_none() {
        return;
    }
    let s = Scratch::new("libm");
    s.write(
        "m.c",
        "double sqrt(double x);\ndouble cbrt(double x);\n\
         int main(void) { return (int)sqrt(1764.0) + (int)cbrt(27.0); }\n",
    );
    // Both the joined and the separated spellings of `-l`.
    s.lf_cc(&["m.c", "-lm", "-o", "m1"]);
    assert_eq!(s.run("m1").status.code(), Some(45));
    s.lf_cc(&["m.c", "-l", "m", "-o", "m2"]);
    assert_eq!(s.run("m2").status.code(), Some(45));
}

#[test]
fn nostdlib_produces_a_static_executable() {
    let s = Scratch::new("nostdlib");
    s.write("a.c", "int add(int a, int b);\nint main(void) { return add(40, 2); }\n");
    s.write("b.c", "int add(int a, int b) { return a + b; }\n");
    // All-C inputs: the framework's in-memory linker core.
    s.lf_cc(&["-nostdlib", "a.c", "b.c", "-o", "pure"]);
    assert_eq!(s.run("pure").status.code(), Some(42));
    // With an object input: qld links statically with lf-cc's own crt0.
    s.lf_cc(&["-c", "b.c"]);
    s.lf_cc(&["-nostdlib", "a.c", "b.o", "-o", "mixed"]);
    assert_eq!(s.run("mixed").status.code(), Some(42));
    for exe in ["pure", "mixed"] {
        let bytes = std::fs::read(s.0.join(exe)).unwrap();
        assert!(!contains(&bytes, b"ld-linux"), "{exe} must not have a program interpreter");
    }
}

/// A C file whose `add_asm` is written in file-scope asm (and calls back into C).
const TOPLEVEL_ASM_C: &str = r#"
int c_helper(int v) { return v + 2; }
__asm__(".text\n"
        ".globl add_asm\n"
        ".type add_asm, @function\n"
        "add_asm:\n"
        "    leal (%rdi,%rsi), %edi\n"
        "    subq $8, %rsp\n"
        "    call c_helper\n"
        "    addq $8, %rsp\n"
        "    ret\n");
"#;

const USES_ASM_C: &str = "int add_asm(int a, int b);\nint main(void) { return add_asm(30, 10); }\n";

#[test]
fn file_scope_asm_is_linked() {
    let s = Scratch::new("tlasm-link");
    s.write("t.c", TOPLEVEL_ASM_C);
    s.write("u.c", USES_ASM_C);
    // The freestanding link can no longer be in-memory: qld links the
    // assembled object next to the C objects.
    s.lf_cc(&["-nostdlib", "u.c", "t.c", "-o", "free"]);
    assert_eq!(s.run("free").status.code(), Some(42));
    if host_crt().is_some() {
        s.lf_cc(&["u.c", "t.c", "-o", "hosted"]);
        assert_eq!(s.run("hosted").status.code(), Some(42));
    }
}

#[test]
fn file_scope_asm_is_merged_into_the_dash_c_object() {
    let s = Scratch::new("tlasm-c");
    s.write("t.c", TOPLEVEL_ASM_C);
    s.write("u.c", USES_ASM_C);
    // One `t.o` carrying both the C code and the assembled file-scope asm.
    s.lf_cc(&["-c", "t.c"]);
    let obj = std::fs::read(s.0.join("t.o")).unwrap();
    assert!(contains(&obj, b"add_asm") && contains(&obj, b"c_helper"));
    s.lf_cc(&["-nostdlib", "u.c", "t.o", "-o", "free"]);
    assert_eq!(s.run("free").status.code(), Some(42));
    if host_crt().is_some() {
        s.lf_cc(&["u.c", "t.o", "-o", "hosted"]);
        assert_eq!(s.run("hosted").status.code(), Some(42));
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}
