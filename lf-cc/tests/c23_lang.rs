//! Differential and unit tests for the C23 language features `_BitInt(N)` and
//! `constexpr`.
//!
//! The differential half mirrors `tests/differential.rs`: each program is
//! compiled with `lf-cc` (at -O0 and -O2) and, when a C23-capable `gcc` is
//! available, with `gcc -std=c23 -O0`; the process exit codes must agree. Every
//! program keeps its result in `0..256` so the exit code is an unambiguous
//! observation, and every program carries an explicit expected value so the
//! test is meaningful even when `gcc` is absent. If `gcc` is missing (or too old
//! to accept `_BitInt`/`constexpr`), the gcc comparison is skipped but the
//! -O0-vs-O2 self-consistency and the absolute-value checks still run.
//!
//! The unit half checks the diagnostics: the pre-C23 rejection of both features,
//! a non-constant `constexpr` initializer, `constexpr`'s `const` implication
//! (assignment rejected), and the `_BitInt` width constraints.

use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::link::write_executable;
use latticefoundry::transform::pipeline::OptLevel;
use lf_cc::{CStd, PpOptions};

/// The C23 corpus: `(name, source, expected exit code)`.
fn programs() -> Vec<(&'static str, &'static str, i32)> {
    vec![
        // --- _BitInt(N): wrapping arithmetic -------------------------------
        // unsigned _BitInt(4) 15 + 1 wraps to 0.
        ("ubi4_wrap", "int main(){ unsigned _BitInt(4) x=15; x+=1; return (int)x + 100; }", 100),
        // signed _BitInt(4) 7 + 1 wraps to -8.
        ("sbi4_wrap", "int main(){ signed _BitInt(4) y=7; y+=1; return (int)y + 108; }", 100),
        // unsigned _BitInt(12) 4095 + 2 wraps to 1.
        ("ubi12_wrap", "int main(){ unsigned _BitInt(12) b=4095; b+=2; return (int)b + 99; }", 100),
        // signed _BitInt(12) 2047 + 1 wraps to -2048.
        ("sbi12_wrap", "int main(){ _BitInt(12) a=2047; a+=1; return (int)a + 2148; }", 100),
        // unsigned _BitInt(8) 200 * 2 wraps to 144.
        ("ubi8_mul", "int main(){ unsigned _BitInt(8) x=200; x*=2; return (int)x + 56; }", 200),
        // A 33-bit shift: 1 << 32 overflows signed _BitInt(33) to a negative value.
        ("bi33_shift", "int main(){ _BitInt(33) c=1; c<<=32; return (int)(c<0) + 99; }", 100),
        // Unary negate of an unsigned _BitInt(4) 1 wraps to 15.
        ("bi_neg", "int main(){ unsigned _BitInt(4) x=1; x = -x; return (int)x + 85; }", 100),
        // --- _BitInt(N): sizeof / alignof ----------------------------------
        (
            "bi_sizeof",
            "int main(){ return sizeof(_BitInt(2))+sizeof(_BitInt(9))+sizeof(_BitInt(17))\
             +sizeof(_BitInt(33))+sizeof(_BitInt(64)); }",
            23,
        ),
        (
            "bi_alignof",
            "int main(){ return _Alignof(_BitInt(4))+_Alignof(_BitInt(12))\
             +_Alignof(_BitInt(20))+_Alignof(_BitInt(40)); }",
            15,
        ),
        // --- _BitInt(N): mixed operands and conversions --------------------
        // unsigned _BitInt(4) + int: the int (higher rank) wins, so no wrap: 16.
        ("bi_mixed", "int main(){ unsigned _BitInt(4) x=15; int r = x + 1; return r + 84; }", 100),
        // _BitInt(M) -> _BitInt(N) narrowing conversion (300 -> 12 in 4 bits -> -4).
        (
            "bi_conv",
            "int main(){ _BitInt(16) a=300; _BitInt(4) b=(_BitInt(4))a; return (int)b + 104; }",
            100,
        ),
        // --- constexpr -----------------------------------------------------
        // constexpr as an array dimension, then read back via sizeof.
        (
            "cx_array",
            "int main(){ constexpr int N=3+4; int a[N]; return (int)(sizeof(a)/sizeof(a[0])) + 93; }",
            100,
        ),
        // Two constexpr dimensions of a 2-D array.
        (
            "cx_dim2",
            "int main(){ constexpr int R=3; constexpr int C=4; int m[R][C]; \
             return (int)(sizeof(m)/sizeof(int)) + 88; }",
            100,
        ),
        // constexpr as a `case` label.
        (
            "cx_case",
            "int main(){ constexpr int K=2; int x=2; switch(x){ case K: return 100; default: return 0; } }",
            100,
        ),
        // constexpr in a `_Static_assert`.
        ("cx_sassert", "int main(){ constexpr int Z=5; _Static_assert(Z==5, \"z\"); return 100; }", 100),
        // A constexpr initialized from another constexpr.
        ("cx_nested", "constexpr int A=10; constexpr int B=A*2+5; int main(){ return B + 75; }", 100),
        // A constexpr read at run time by value.
        ("cx_read", "constexpr int V=42; int main(){ return V + 58; }", 100),
        // A constexpr of _BitInt type.
        ("cx_bitint", "int main(){ constexpr unsigned _BitInt(4) C=15; return (int)C + 85; }", 100),
    ]
}

struct Harness {
    dir: PathBuf,
    /// A C23-capable C compiler, if one is available.
    gcc: Option<String>,
}

impl Harness {
    fn new() -> Harness {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("lf-cc-c23test-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let gcc = which("gcc").or_else(|| which("cc")).filter(|g| gcc_supports_c23(g, &dir));
        Harness { dir, gcc }
    }

    /// Compile with lf-cc under `-std=c23` at `opt`, run, and return the exit code.
    fn lf_run(&self, name: &str, src: &str, opt: OptLevel) -> i32 {
        let input = format!("{name}.c");
        let pp = PpOptions {
            std: CStd::parse("c23").expect("known --std"),
            main_file_name: input.clone(),
            ..PpOptions::default()
        };
        let image = lf_cc::build_image_with(src, &input, &pp, opt, false)
            .unwrap_or_else(|e| panic!("lf-cc failed to build '{name}': {e:?}"));
        let bin = self.dir.join(format!("{name}.lf.{}", opt.name()));
        write_executable(bin.to_str().unwrap(), &image).expect("write executable");
        run_exit(&bin)
    }

    /// Compile with `gcc -std=c23 -O0` and run, if a C23-capable gcc is present.
    fn gcc_run(&self, name: &str, src: &str) -> Option<i32> {
        let gcc = self.gcc.as_ref()?;
        let c = self.dir.join(format!("{name}.c"));
        std::fs::write(&c, src).expect("write source");
        let bin = self.dir.join(format!("{name}.gcc"));
        let status = Command::new(gcc)
            .args(["-O0", "-w", "-std=c23", "-o"])
            .arg(&bin)
            .arg(&c)
            .status()
            .expect("run gcc");
        assert!(status.success(), "gcc failed to compile '{name}'");
        Some(run_exit(&bin))
    }
}

/// Whether `gcc` accepts C23 `_BitInt` and `constexpr` (older toolchains do not).
fn gcc_supports_c23(gcc: &str, dir: &Path) -> bool {
    let c = dir.join("c23probe.c");
    if std::fs::write(&c, "int main(){ constexpr unsigned _BitInt(4) x=15; return (int)x; }").is_err()
    {
        return false;
    }
    let bin = dir.join("c23probe");
    Command::new(gcc)
        .args(["-O0", "-w", "-std=c23", "-o"])
        .arg(&bin)
        .arg(&c)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
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
fn c23_differential() {
    let h = Harness::new();
    let mut matched_gcc = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let progs = programs();

    for (name, src, expected) in &progs {
        let o0 = h.lf_run(name, src, OptLevel::O0);
        let o2 = h.lf_run(name, src, OptLevel::O2);
        if o0 != o2 {
            failures.push(format!("{name}: lf-cc -O0={o0} != -O2={o2}"));
            continue;
        }
        if o0 != *expected {
            failures.push(format!("{name}: lf-cc={o0} != expected={expected}"));
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

    assert!(failures.is_empty(), "c23 differential failures:\n{}", failures.join("\n"));
    eprintln!(
        "c23 differential: {}/{} programs; {matched_gcc} matched gcc ({})",
        progs.len(),
        progs.len(),
        if h.gcc.is_some() { "C23 gcc present" } else { "C23 gcc absent — comparison skipped" },
    );
}

// --- unit tests: feature gating & diagnostics ------------------------------

/// Type-check `src` under `std`, returning whether the front end rejected it.
fn rejected(src: &str, std: &str) -> bool {
    let pp = PpOptions {
        std: CStd::parse(std).expect("known --std"),
        main_file_name: "t.c".to_owned(),
        ..PpOptions::default()
    };
    lf_cc::check_source_with(src, &pp).is_err()
}

/// Type-check `src` under `std`, returning whether the front end accepted it.
fn accepted(src: &str, std: &str) -> bool {
    let pp = PpOptions {
        std: CStd::parse(std).expect("known --std"),
        main_file_name: "t.c".to_owned(),
        ..PpOptions::default()
    };
    lf_cc::check_source_with(src, &pp).is_ok()
}

#[test]
fn bitint_requires_c23() {
    let src = "int main(){ _BitInt(8) x = 0; return x; }";
    assert!(rejected(src, "c11"), "_BitInt must be rejected before C23");
    assert!(accepted(src, "c23"), "_BitInt must be accepted under C23");
}

#[test]
fn constexpr_requires_c23() {
    let src = "int main(){ constexpr int n = 5; return n; }";
    assert!(rejected(src, "c11"), "constexpr must be rejected before C23");
    assert!(accepted(src, "c23"), "constexpr must be accepted under C23");
}

#[test]
fn constexpr_initializer_must_be_constant() {
    // A call is not a constant expression.
    let src = "int f(void){ return 1; } int main(){ constexpr int n = f(); return n; }";
    assert!(rejected(src, "c23"), "a non-constant constexpr initializer must be rejected");
}

#[test]
fn constexpr_requires_an_initializer() {
    let src = "int main(){ constexpr int n; return n; }";
    assert!(rejected(src, "c23"), "a constexpr object without an initializer must be rejected");
}

#[test]
fn constexpr_implies_const() {
    // constexpr objects are compile-time constants, so assignment is a constraint
    // violation (they are not modifiable lvalues).
    let src = "int main(){ constexpr int n = 5; n = 6; return n; }";
    assert!(rejected(src, "c23"), "assigning to a constexpr object must be rejected");
}

#[test]
fn constexpr_single_declarator() {
    // C23 allows only a single declarator with constexpr (as gcc enforces).
    let src = "int main(){ constexpr int a = 1, b = 2; return a + b; }";
    assert!(rejected(src, "c23"), "constexpr with multiple declarators must be rejected");
}

#[test]
fn bitint_width_constraints() {
    // A signed _BitInt needs at least 2 bits (one is the sign); unsigned needs 1.
    assert!(rejected("int main(){ signed _BitInt(1) x = 0; return 0; }", "c23"));
    assert!(accepted("int main(){ unsigned _BitInt(1) x = 1; return (int)x; }", "c23"));
    // Width 0 is invalid.
    assert!(rejected("int main(){ unsigned _BitInt(0) x = 0; return 0; }", "c23"));
    // Over 64 bits is unsupported (multi-word is future work).
    assert!(rejected("int main(){ _BitInt(65) x = 0; return 0; }", "c23"));
    // A 64-bit _BitInt is supported.
    assert!(accepted("int main(){ _BitInt(64) x = 5; return (int)x; }", "c23"));
}

#[test]
fn constexpr_non_integer_rejected() {
    // Only integer-category constexpr objects are supported; a pointer constexpr
    // is diagnosed rather than silently mishandled.
    let src = "int main(){ constexpr int *p = 0; return p != 0; }";
    assert!(rejected(src, "c23"), "a non-integer constexpr must be diagnosed");
}
