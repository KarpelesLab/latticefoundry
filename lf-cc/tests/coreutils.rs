//! The constructs building GNU coreutils (and the gnulib it carries) needed,
//! each as a focused test: autoconf's probes through the driver (`-E`, the
//! `-M` dependency family, `-x c`, standard input), declarations that must be
//! rejected the way gcc rejects them (configure detects a function's signature
//! by redeclaring it), and run-programs for the language features gnulib uses.
//!
//! The run-programs need no headers or C library: each is compiled at -O0 and
//! -O2 into a freestanding image and its exit code checked (and gcc's too, when
//! gcc is installed). See `docs/coreutils.md` for the whole build.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use latticefoundry::link::write_executable;
use latticefoundry::transform::pipeline::OptLevel;

const LF_CC: &str = env!("CARGO_BIN_EXE_lf-cc");

/// `(name, source, expected exit code)`.
fn programs() -> Vec<(&'static str, &'static str, i32)> {
    vec![
        (
            // Block-scope variable-length arrays (autoconf's C99 probe uses
            // one): run-time length, `sizeof`, an array of fixed-size rows, and
            // one declared in a loop, whose storage is reused per iteration.
            "vla",
            r#"
static int sum(int n) {
    int a[n];
    for (int i = 0; i < n; i++) a[i] = i * i;
    int s = 0;
    for (int i = 0; i < n; i++) s += a[i];
    return s + (int)(sizeof a / sizeof a[0]);
}
int main(void) {
    volatile int n = 5;
    char buf[n * 2];
    int rows[n][3];
    rows[n - 1][2] = 9;
    long total = 0;
    for (int k = 0; k < 200000; k++) {
        int t[k % 50 + 1];
        t[k % 50] = k;
        total += t[k % 50] == k;
    }
    return (sum(n) == 30 + 5) + 2 * (sizeof buf == 10) + 4 * (sizeof rows == 60)
         + 8 * (rows[4][2] == 9) + 16 * (total == 200000);
}
"#,
            31,
        ),
        (
            // Members of record rvalues: a call's result and a conditional
            // choosing between struct-returning calls (`get_stat_mtime`).
            "struct_rvalues",
            r#"
struct ts { long sec, nsec; };
struct big { int a[6]; };
static struct ts mk(long s) { struct ts t = { s, s * 2 }; return t; }
static struct big mkbig(int x) { struct big b = { { x, x + 1, x + 2, x + 3, x + 4, x + 5 } }; return b; }
int main(void) {
    volatile int c = 1;
    struct ts t = c ? mk(3) : mk(4);
    long n = (c ? mk(5) : c > 1 ? mk(6) : mk(7)).nsec;
    return (mk(2).nsec == 4) + 2 * (mkbig(7).a[5] == 12) + 4 * (t.sec == 3) + 8 * (n == 10);
}
"#,
            15,
        ),
        (
            // Initializers: a struct member from a struct expression, nested
            // designators (`[0].sec`, `.in.y`), a designator naming a member of
            // an anonymous struct, and designators into an array of structs.
            "initializers",
            r#"
struct ts { long sec, nsec; };
struct its { struct ts interval, value; };
struct outer { int a; struct { int x, y; } in; struct { int p, q; }; };
static struct ts g[2] = { [0].nsec = 5, [1] = { 6, 7 } };
static struct outer go = { .in.y = 3, .q = 4, .a = 1 };
int main(void) {
    struct ts t = { 8, 9 };
    struct its it = { .interval = { 0 }, .value = t };
    struct ts l[2] = { [1].sec = 2, [0].nsec = 1 };
    struct outer o = { .in.x = 5, .p = 6 };
    return (it.value.nsec == 9) + 2 * (it.interval.sec == 0) + 4 * (l[1].sec == 2 && l[0].nsec == 1)
         + 8 * (o.in.x == 5 && o.p == 6 && o.in.y == 0) + 16 * (g[0].nsec == 5 && g[1].nsec == 7)
         + 32 * (go.in.y == 3 && go.q == 4 && go.a == 1 && go.in.x == 0);
}
"#,
            63,
        ),
        (
            // Static initializers: address constants with arithmetic, `&f`
            // through a qualified pointer, `sizeof` of a member through a null
            // pointer and of a block-scope static array.
            "address_constants",
            r#"
struct lconv { char *decimal_point; int n; };
static char line_buf[20] = "abcdefghijklmnopqrs";
static char *p1 = line_buf + 20 - 8;
static char const zs[] = "0KkM";
static char const *vs = 1 + zs;
static int arr[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
static int *pa = &arr[2] + 3;
static int *pb = arr - -4;
int y = sizeof (((struct lconv *) 0)->decimal_point);
static int add1(int x) { return x + 1; }
int main(void) {
    static int (*const volatile fp)(int) = &add1;
    static char const prefix[] = "posix-";
    static const unsigned long prefix_len = sizeof prefix - 1;
    return (*p1 == 'm') + 2 * (*vs == 'K') + 4 * (*pa == 5) + 8 * (*pb == 4) + 16 * (y == 8)
         + 32 * (fp(1) == 2) + 64 * (prefix_len == 6);
}
"#,
            127,
        ),
        (
            // Integer constant expressions gnulib writes: a floating constant
            // cast to an integer type (`TYPE_IS_INTEGER`), and a static
            // assertion whose message is several adjacent literals (`verify`).
            "constant_expressions",
            r#"
typedef long time_t;
_Static_assert (((time_t) 1.5 == 1), "verify (" "TYPE_IS_INTEGER" ")");
_Static_assert ((! ((time_t) 0 < (time_t) -1)), "signed");
enum { A = (int) 2.9, B = (unsigned char) 200.7 };
int main(void) {
    _Static_assert ((_Bool) 0.5, "bool" " cast");
    return A + B;
}
"#,
            2 + 200,
        ),
        (
            // The preprocessor: a conditional directive inside a macro's
            // arguments, a block comment spanning lines in a `#define`, and
            // characters that start no token in a skipped group.
            "preprocessor",
            r#"
#define ADD(a, b) ((a) + (b))
#define TWO 1 /* a comment
                 spanning lines */ + 1
#if 0
# error mail bug-gnulib@gnu.org, it's broken
#endif
static const char msg[] = "ab\
cd";
int main(void) {
    int r = ADD(1,
#ifdef TWO
                TWO
#else
                100
#endif
    );
    switch (r) {
    case 3:
        r++;
        [[fallthrough]];
    default:
        break;
    }
    return r + 8 * (sizeof msg == 5 && msg[2] == 'c') + 16 * (__LINE__ == 25);
}
"#,
            4 + 8 + 16,
        ),
        (
            // `#pragma weak`: an undefined weak function's address is null.
            "pragma_weak",
            r#"
extern int no_such_function(void);
#pragma weak no_such_function
int main(void) {
    return no_such_function ? 1 : 7;
}
"#,
            7,
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
                return status
                    .code()
                    .unwrap_or_else(|| panic!("{} terminated abnormally: {status}", bin.display()));
            }
            Err(e) => panic!("run {}: {e}", bin.display()),
        }
    }
    panic!("{} stayed busy", bin.display());
}

/// A per-test scratch directory, removed when the test finishes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lf-cc-coreutils-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, name: &str, text: &str) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, text).expect("write source");
        p
    }

    /// Run `lf-cc` with `args` (and `stdin`) in this directory.
    fn lf_cc(&self, args: &[&str], stdin: Option<&str>) -> Output {
        let mut child = Command::new(LF_CC)
            .args(args)
            .current_dir(&self.0)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("run lf-cc");
        {
            use std::io::Write;
            let mut pipe = child.stdin.take().expect("stdin");
            if let Some(text) = stdin {
                pipe.write_all(text.as_bytes()).expect("write stdin");
            }
        }
        child.wait_with_output().expect("wait for lf-cc")
    }

    /// Like [`lf_cc`](Self::lf_cc), asserting success and returning stdout.
    fn lf_cc_ok(&self, args: &[&str], stdin: Option<&str>) -> String {
        let out = self.lf_cc(args, stdin);
        assert!(out.status.success(), "lf-cc {args:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(self.0.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn gnulib_constructs_run_correctly() {
    let dir = Scratch::new("run");
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
            let bin = dir.path().join(format!("{name}.{}", opt.name()));
            write_executable(bin.to_str().unwrap(), &image).expect("write executable");
            let got = run_exit(&bin);
            if got != expected {
                failures.push(format!("{name} ({}): exit {got}, expected {expected}", opt.name()));
            }
        }
        if let Some(gcc) = &gcc {
            let c = dir.write(&format!("{name}.c"), src);
            let bin = dir.path().join(format!("{name}.gcc"));
            let ok = Command::new(gcc)
                .arg(common::gcc_std_flag(gcc, "gnu17"))
                .args(["-O0", "-w", "-o"])
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
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
}

fn errors_of(src: &str) -> String {
    match lf_cc::check_source(src) {
        Ok(_) => String::new(),
        Err(diags) => diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("\n"),
    }
}

/// configure learns a function's signature by redeclaring it: the
/// redeclaration must fail when the types conflict, as it does with gcc.
#[test]
fn conflicting_redeclarations_are_rejected() {
    for src in [
        // glibc's GNU `char *strerror_r` against the POSIX `int` one.
        "char *strerror_r(int, char *, unsigned long); int strerror_r(int, char *, unsigned long);",
        // `ioctl (int, unsigned long, ...)` against `(int, int, ...)`.
        "int ioctl(int, unsigned long, ...); int ioctl(int, int, ...);",
        // `gid_t *` (unsigned) against `int *`.
        "int getgroups(int, unsigned *); int getgroups(int, int *);",
        // a prototype's arity, and its variadic-ness
        "int f(int); int f(int, int);",
        "int g(int, ...); int g(int);",
        // objects
        "extern int x; extern long x;",
        "extern char *names[]; extern int names[];",
    ] {
        let errs = errors_of(src);
        assert!(errs.contains("conflicting types"), "`{src}` should conflict, got: {errs:?}");
    }
    // A bound that wraps to a huge size (`sizeof (long double) - sizeof
    // (double) - 1` where they are equal) is an error, not a crash.
    let errs = errors_of("int foo[sizeof (int) - sizeof (unsigned) - 1];");
    assert!(errs.contains("too large"), "{errs:?}");
    for src in [
        // no prototype, and the parameter's promotion (an old-style definition)
        "int f(); int f(int a, char *b);",
        "int g(int); int g(c) char c; { return c; }",
        "extern int a[]; int a[4];",
        "typedef unsigned long size_t; unsigned long h(size_t); size_t h(unsigned long);",
    ] {
        let errs = errors_of(src);
        assert!(errs.is_empty(), "`{src}` is a compatible redeclaration, got: {errs:?}");
    }
}

/// `-E` prints the preprocessed text: macros expanded, the line structure kept
/// with line markers, `#pragma` lines passed through, and `#error` failing.
#[test]
fn dash_e_prints_preprocessed_text() {
    let dir = Scratch::new("dashe");
    dir.write("inc.h", "#define GREETING \"hi\"\nint from_header;\n");
    dir.write(
        "e.c",
        "#include \"inc.h\"\n#define F(x) x+1\n#define S(x) #x\n#pragma weak w\n\
         int a = F(2)+ +3; const char *s = S(a  b \"c\");\n\n\n\n\n\n\n\n\n\n\n\nint last = __LINE__;\nchar *g = GREETING;\n",
    );
    let out = dir.lf_cc_ok(&["-E", "e.c"], None);
    assert!(out.starts_with("# 1 \"e.c\"\n"), "{out}");
    assert!(out.contains("inc.h\"\nint from_header;\n"), "{out}");
    assert!(out.contains("#pragma weak w\n"), "{out}");
    assert!(out.contains("int a = 2+1+ +3; const char *s = \"a b \\\"c\\\"\";"), "{out}");
    assert!(out.contains("# 17 \"e.c\"\nint last = 17;\nchar *g = \"hi\";"), "{out}");
    // Tokens that would paste stay apart.
    let out = dir.lf_cc_ok(&["-E", "-x", "c", "-"], Some("#define P +\n#define E\nint a = 1 P+ 2; int b = -E-1;\n"));
    assert!(out.contains("int a = 1 + + 2; int b = - -1;"), "{out}");
    // `-o` writes the file; `#error` fails.
    dir.lf_cc_ok(&["-E", "e.c", "-o", "e.i"], None);
    assert!(dir.read("e.i").contains("from_header"));
    let out = dir.lf_cc(&["-E", "-"], Some("#if 1\n#error \"stop here\"\n#endif\n"));
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("stop here"));
}

/// The `-M` family writes make rules (automake's `gcc3` dependency mode:
/// `-MT obj -MD -MP -MF file`).
#[test]
fn dependency_rules() {
    let dir = Scratch::new("deps");
    dir.write("a.h", "#include \"b.h\"\n");
    dir.write("b.h", "#define B 1\n");
    dir.write("m.c", "#include \"a.h\"\n#include \"a.h\"\nint main(void) { return B - 1; }\n");
    let out = dir.lf_cc_ok(&["-M", "-nostdinc", "m.c"], None);
    assert_eq!(out, "m.o: m.c a.h b.h\n");
    let out = dir.lf_cc_ok(&["-MM", "-MT", "obj/m.o", "-MP", "m.c"], None);
    assert_eq!(out, "obj/m.o: m.c a.h b.h\n\na.h:\n\nb.h:\n");
    std::fs::create_dir_all(dir.path().join("deps")).expect("create deps dir");
    dir.lf_cc_ok(&["-MT", "m.o", "-MD", "-MP", "-MF", "deps/m.Tpo", "-c", "-o", "m.o", "m.c"], None);
    let rule = dir.read("deps/m.Tpo");
    assert!(rule.starts_with("m.o: m.c "), "{rule}");
    assert!(rule.contains("a.h") && rule.contains("\nb.h:\n"), "{rule}");
    assert!(dir.path().join("m.o").is_file());
    // `-MMD` leaves out system headers; without `-MF` the rule goes next to the object.
    dir.write("s.c", "#include <stddef.h>\n#include <stdio.h>\n#include \"b.h\"\nint main(void) { return 0; }\n");
    dir.lf_cc_ok(&["-MMD", "-c", "s.c", "-o", "s.o"], None);
    assert_eq!(dir.read("s.d"), "s.o: s.c b.h\n");
}

/// `__attribute__((constructor))` / `((destructor))` functions run around
/// `main`, static ones included (coreutils' `libstdbuf.so` is set up so).
#[test]
fn constructors_and_destructors_run() {
    if latticefoundry::link::gnu::HostCrt::discover().is_none() {
        eprintln!("skipping: no host C runtime (crt1.o) found");
        return;
    }
    let dir = Scratch::new("ctor");
    dir.write(
        "ctor.c",
        "#include <stdio.h>\nstatic int v;\n\
         static void __attribute ((constructor)) init_it (void) { v = 42; }\n\
         void __attribute__((constructor(200))) init2 (void) { v += 1; }\n\
         static void bye (void) __attribute__((destructor));\n\
         static void bye (void) { printf (\"bye %d\\n\", v); }\n\
         int main (void) { printf (\"main %d\\n\", v); return 0; }\n",
    );
    dir.lf_cc_ok(&["-O2", "ctor.c", "-o", "ctor"], None);
    let mut out = None;
    for _ in 0..50 {
        match Command::new(dir.path().join("ctor")).output() {
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(20)),
            other => {
                out = Some(other.expect("run ctor"));
                break;
            }
        }
    }
    let out = out.expect("ctor stayed busy");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "main 43\nbye 43\n");
}
