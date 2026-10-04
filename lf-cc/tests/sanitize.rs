//! `lf-cc -fsanitize=…`: C programs with undefined behavior report the same
//! set of issues — the same kinds at the same source lines — as the host
//! `gcc -fsanitize=…` (the message text differs), correct programs report
//! nothing, and `-fsanitize-trap` dies with `SIGILL`.
//!
//! Each program marks its faulting lines `/* UB:<kind> */`; the marks are the
//! expected set for both compilers. Without a gcc that links the UBSan runtime
//! only the lf-cc half runs. (`-fsanitize=unreachable` has nothing to check in
//! lf-cc's C: it lowers `__builtin_unreachable()` to no code at all.)

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::collections::BTreeSet;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const LF_CC: &str = env!("CARGO_BIN_EXE_lf-cc");

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lf-cc-sanitize-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(exe: &Path) -> Output {
    loop {
        match Command::new(exe).output() {
            Ok(o) => break o,
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
            Err(e) => panic!("exec {}: {e}", exe.display()),
        }
    }
}

fn compile(compiler: &str, src: &Path, exe: &Path, flags: &[&str]) -> Result<(), String> {
    let out = Command::new(compiler)
        .args(flags)
        .arg(src)
        .arg("-o")
        .arg(exe)
        .output()
        .map_err(|e| format!("{compiler}: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!("{compiler} {flags:?} failed:\n{}", String::from_utf8_lossy(&out.stderr)))
    }
}

/// Whether a host gcc can build and run a UBSan program.
fn gcc_ubsan(dir: &Scratch) -> bool {
    let src = dir.path("probe.c");
    std::fs::write(&src, "int main(void){ return 0; }\n").unwrap();
    let exe = dir.path("probe");
    compile("gcc", &src, &exe, &["-fsanitize=undefined"]).is_ok() && run(&exe).status.success()
}

/// The kind of one `runtime error:` message, from either compiler.
fn category(msg: &str) -> &'static str {
    let rules: &[(&str, &str)] = &[
        ("signed integer overflow", "overflow"),
        ("negation of", "overflow"),
        ("by -1 cannot be represented", "div-overflow"),
        ("shift exponent", "shift"),
        ("left shift of", "shift"),
        ("division by zero", "div-zero"),
        ("outside the range of representable values", "float-cast"),
        ("out of bounds", "bounds"),
        ("insufficient space", "bounds"),
        ("null pointer", "null"),
        ("misaligned address", "alignment"),
        ("unreachable program point", "unreachable"),
    ];
    rules.iter().find(|(pat, _)| msg.contains(pat)).map_or("other", |(_, c)| c)
}

/// The `(line, kind)` set of a run's reports (`file:line[:col]: runtime
/// error: msg` lines).
fn reports(stderr: &[u8]) -> BTreeSet<(usize, &'static str)> {
    let text = String::from_utf8_lossy(stderr);
    let mut set = BTreeSet::new();
    for l in text.lines() {
        let Some((head, msg)) = l.split_once(": runtime error: ") else { continue };
        let line = head.split(':').nth(1).and_then(|n| n.parse().ok()).unwrap_or(0);
        set.insert((line, category(msg)));
    }
    set
}

/// The `/* UB:<kind> */` marks of a source.
fn marks(src: &str) -> BTreeSet<(usize, &'static str)> {
    let kinds = ["overflow", "div-overflow", "shift", "div-zero", "float-cast", "bounds", "null", "alignment", "unreachable"];
    let mut set = BTreeSet::new();
    for (i, l) in src.lines().enumerate() {
        for k in kinds {
            if l.contains(&format!("/* UB:{k} */")) {
                set.insert((i + 1, k));
            }
        }
    }
    set
}

struct Program {
    name: &'static str,
    /// The `-fsanitize=` list given to both compilers.
    checks: &'static str,
    src: &'static str,
}

const PROGRAMS: &[Program] = &[
    Program {
        name: "arith",
        checks: "undefined",
        src: r#"
volatile int big = 2147483647, small = -2147483647 - 1, one = 1;
volatile long lbig = 9223372036854775807L;
volatile unsigned u = 4294967295u;
volatile signed char c = 127;
int main(void) {
    int a = big + one;                /* UB:overflow */
    int b = small - one;              /* UB:overflow */
    int m = big * 2;                  /* UB:overflow */
    int n = -small;                   /* UB:overflow */
    long l = lbig + one;              /* UB:overflow */
    int x = big;
    x++;                              /* UB:overflow */
    int y = small;
    y += small;                       /* UB:overflow */
    unsigned w = u + 1u;
    signed char d = c + 1;
    int fine = big - one * 2 + one;
    return (a ^ b ^ m ^ n ^ (int)l ^ x ^ y ^ (int)w ^ d ^ fine) & 1;
}
"#,
    },
    Program {
        name: "shifts",
        checks: "undefined",
        src: r#"
volatile int s33 = 33, s31 = 31, sneg = -1, s3 = 3, two = 2;
volatile long l = 1;
int main(void) {
    int a = 1 << s33;                 /* UB:shift */
    int b = 8 >> sneg;                /* UB:shift */
    int c = two << s31;               /* UB:shift */
    long d = l << 64;                 /* UB:shift */
    unsigned e = 1u << s31;
    int f = 1 << s3;
    return (a ^ b ^ c ^ (int)d ^ (int)e ^ f) & 1;
}
"#,
    },
    Program {
        name: "div_zero",
        checks: "undefined",
        src: r#"
volatile int z = 0, seven = 7;
int main(void) {
    int ok = seven / 2 + seven % 3;
    return ok + seven / z;            /* UB:div-zero */
}
"#,
    },
    Program {
        name: "div_overflow",
        checks: "undefined",
        src: r#"
volatile int m = -2147483647 - 1, neg = -1;
int main(void) {
    int ok = m / 2;
    return ok + m / neg;              /* UB:div-overflow */
}
"#,
    },
    Program {
        name: "float_cast",
        checks: "float-cast-overflow",
        src: r#"
volatile double big = 1e20, neg = -1.0, half = -0.5, fine = 123.75;
volatile float f = 3e10f;
int main(void) {
    int a = (int)big;                 /* UB:float-cast */
    unsigned b = (unsigned)neg;       /* UB:float-cast */
    unsigned c = (unsigned)half;
    long d = (long)f;
    int e = (int)fine;
    short g = (short)f;               /* UB:float-cast */
    return (a ^ (int)b ^ (int)c ^ (int)d ^ e ^ g) & 1;
}
"#,
    },
    Program {
        name: "bounds",
        checks: "undefined",
        src: r#"
int table[10];
volatile int ten = 10, four = 4, three = 3;
int read(int i) { return table[i]; } /* UB:bounds */
int main(void) {
    int local[4] = {1, 2, 3, 4};
    int a = read(ten);
    local[four] = 5;                  /* UB:bounds */
    int b = local[three] + read(three);
    return (a ^ b) & 1;
}
"#,
    },
    Program {
        name: "null",
        checks: "undefined",
        src: r#"
int * volatile p = 0;
int main(void) {
    return *p;                        /* UB:null */
}
"#,
    },
    Program {
        name: "alignment",
        checks: "undefined",
        src: r#"
long buf[4];
volatile int off = 2;
int main(void) {
    int *p = (int *)((char *)buf + off);
    int v = *p;                       /* UB:alignment */
    int *q = (int *)((char *)buf + off + 2);
    return (v ^ *q) & 1;
}
"#,
    },
    Program {
        name: "clean",
        checks: "undefined,float-cast-overflow",
        src: r#"
int prime[1000];
volatile int n = 1000;
int main(void) {
    for (int i = 2; i < n; i++) prime[i] = 1;
    for (int i = 2; i * i < n; i++)
        if (prime[i])
            for (int j = i * i; j < n; j += i) prime[j] = 0;
    int count = 0;
    for (int i = 0; i < n; i++) count += prime[i];
    long h = 0;
    for (int i = 0; i < 100; i++) h = (h * 31 + (i << 3) - (i >> 1) + i / 7 + (int)(i * 0.5)) % 1000003;
    unsigned u = 0;
    for (int i = 0; i < 100; i++) u = u * 2654435761u + (unsigned)i;
    return (count == 168 && h != 0 && u != 1) ? 42 : 1;
}
"#,
    },
];

#[test]
fn lf_cc_reports_what_gcc_reports() {
    let dir = Scratch::new("vsgcc");
    let have_gcc = gcc_ubsan(&dir);
    if !have_gcc {
        eprintln!("note: no gcc with UBSan; checking lf-cc against the marks only");
    }
    for p in PROGRAMS {
        let src = dir.path(&format!("{}.c", p.name));
        std::fs::write(&src, p.src).unwrap();
        let want = marks(p.src);
        let flag = format!("-fsanitize={}", p.checks);
        for opt in ["-O0", "-O2"] {
            let exe = dir.path(&format!("{}{opt}", p.name));
            compile(LF_CC, &src, &exe, &[opt, &flag]).unwrap();
            let out = run(&exe);
            let got = reports(&out.stderr);
            assert_eq!(got, want, "lf-cc {opt} {}: reports differ from the marks:\n{}", p.name, String::from_utf8_lossy(&out.stderr));
            if want.is_empty() {
                assert_eq!(out.status.code(), Some(42), "lf-cc {opt} {}: the clean program's result", p.name);
            }
        }
        if have_gcc {
            let exe = dir.path(&format!("{}-gcc", p.name));
            compile("gcc", &src, &exe, &["-O0", "-w", &flag]).unwrap();
            let out = run(&exe);
            let got = reports(&out.stderr);
            assert_eq!(got, want, "gcc {}: reports differ from the marks:\n{}", p.name, String::from_utf8_lossy(&out.stderr));
        }
    }
}

#[test]
fn trap_mode_and_recovery() {
    let dir = Scratch::new("trap");
    let arith = &PROGRAMS[0];
    let src = dir.path("arith.c");
    std::fs::write(&src, arith.src).unwrap();
    let exe = dir.path("arith");
    for flags in [&["-fsanitize=undefined", "-fsanitize-trap"][..], &["-O2", "-fsanitize=signed-integer-overflow", "-fsanitize-trap=undefined"]] {
        compile(LF_CC, &src, &exe, flags).unwrap();
        let out = run(&exe);
        assert_eq!(out.status.signal(), Some(4), "{flags:?}: SIGILL expected, got {:?}", out.status);
        assert!(out.stderr.is_empty(), "{flags:?}: a trap prints nothing");
    }
    // Without recovery the first report ends the program.
    compile(LF_CC, &src, &exe, &["-fsanitize=undefined", "-fno-sanitize-recover=all"]).unwrap();
    let out = run(&exe);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(reports(&out.stderr).len(), 1, "{}", String::from_utf8_lossy(&out.stderr));
    // A check left out is not made: only shifts are checked here.
    compile(LF_CC, &src, &exe, &["-fsanitize=shift"]).unwrap();
    let out = run(&exe);
    assert!(out.stderr.is_empty(), "{}", String::from_utf8_lossy(&out.stderr));
    // Unsupported sanitizers are refused.
    assert!(compile(LF_CC, &src, &exe, &["-fsanitize=address"]).is_err());
}

#[test]
fn signed_arithmetic_carries_nsw_only_when_checked() {
    let dir = Scratch::new("nsw");
    let src = dir.path("f.c");
    std::fs::write(&src, "int f(int a, int b) { return a + b; }\nunsigned g(unsigned a) { return a * 3u; }\n").unwrap();
    let emit = |flags: &[&str]| {
        let out = Command::new(LF_CC).args(flags).arg("-S").arg(&src).output().expect("run lf-cc");
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let plain = emit(&[]);
    assert!(plain.contains("add %") && !plain.contains("nsw"), "{plain}");
    let checked = emit(&["-fsanitize=signed-integer-overflow"]);
    assert!(checked.contains("add nsw"), "{checked}");
    assert!(!checked.contains("mul nsw"), "unsigned arithmetic stays unflagged:\n{checked}");
}

/// A benchmark-ish program: an LCG-filled array sorted by quicksort, a matrix
/// product and a byte histogram, folded into one checksum.
const BENCH: &str = r#"
#define N 60000
#define M 48
static int data[N];
static long a[M][M], b[M][M], c[M][M];
static unsigned char bytes[N];
static int hist[256];

static void sort(int *v, int lo, int hi) {
    while (lo < hi) {
        int p = v[(lo + hi) / 2], i = lo, j = hi;
        while (i <= j) {
            while (v[i] < p) i++;
            while (v[j] > p) j--;
            if (i <= j) { int t = v[i]; v[i] = v[j]; v[j] = t; i++; j--; }
        }
        if (j - lo < hi - i) { sort(v, lo, j); lo = i; } else { sort(v, i, hi); hi = j; }
    }
}

int main(void) {
    unsigned s = 12345;
    for (int i = 0; i < N; i++) {
        s = s * 1103515245u + 12345u;
        data[i] = (int)(s >> 1) % 1000000 - 500000;
        bytes[i] = (unsigned char)(s >> 16);
    }
    sort(data, 0, N - 1);
    for (int i = 1; i < N; i++) if (data[i - 1] > data[i]) return 1;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) { a[i][j] = i * 3 - j; b[i][j] = (i ^ j) % 17 - 8; }
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) {
            long t = 0;
            for (int k = 0; k < M; k++) t += a[i][k] * b[k][j];
            c[i][j] = t;
        }
    for (int i = 0; i < N; i++) hist[bytes[i]]++;
    long sum = data[N / 3] + data[N / 2];
    for (int i = 0; i < M; i++) sum += c[i][(i * 7) % M];
    for (int i = 0; i < 256; i++) sum += hist[i] * (i % 5);
    return (int)(((sum % 251) + 251) % 251);
}
"#;

#[test]
fn a_checked_benchmark_computes_the_same_result() {
    let dir = Scratch::new("bench");
    let src = dir.path("bench.c");
    std::fs::write(&src, BENCH).unwrap();
    let mut results = Vec::new();
    for flags in [&["-O2"][..], &["-O0", "-fsanitize=undefined"], &["-O2", "-fsanitize=undefined"], &["-O2", "-fsanitize=undefined", "-fsanitize-trap"]] {
        let exe = dir.path("bench");
        compile(LF_CC, &src, &exe, flags).unwrap();
        let start = std::time::Instant::now();
        let out = run(&exe);
        let took = start.elapsed();
        assert!(out.stderr.is_empty(), "{flags:?}: {}", String::from_utf8_lossy(&out.stderr));
        eprintln!("bench {flags:?}: exit {:?} in {took:?}", out.status.code());
        results.push(out.status.code());
    }
    assert!(results.iter().all(|r| *r == results[0] && r.is_some_and(|c| c != 1)), "{results:?}");
    let exe = dir.path("bench-gcc");
    if compile("gcc", &src, &exe, &["-O2"]).is_ok() {
        assert_eq!(run(&exe).status.code(), results[0], "gcc agrees");
    }
}
