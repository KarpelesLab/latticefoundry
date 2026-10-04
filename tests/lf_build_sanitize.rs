//! End-to-end `lf build --sanitize=…` through the real driver binary: for each
//! kind of undefined behavior, a program that triggers it reports the right
//! kind at the right source line (and the same program without the UB runs
//! cleanly with every check on); trap mode dies with `SIGILL`; a small
//! benchmark computes the same result with and without checks.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-sanitize-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn lf_build(src: &Path, exe: &Path, extra: &[&str]) {
    let out = Command::new(env!("CARGO_BIN_EXE_lf"))
        .arg("build")
        .arg(src)
        .arg("-o")
        .arg(exe)
        .args(extra)
        .output()
        .expect("run lf");
    assert!(out.status.success(), "lf build {extra:?} failed: {}", String::from_utf8_lossy(&out.stderr));
}

fn run(exe: &Path) -> Output {
    // Retry a transient ETXTBSY (errno 26) from a concurrent fork.
    loop {
        match Command::new(exe).output() {
            Ok(o) => break o,
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
            Err(e) => panic!("exec {}: {e}", exe.display()),
        }
    }
}

/// One undefined-behavior case: the faulting code (marked `; UB` on the
/// faulting line), the `@v` value that triggers it and one that does not, the
/// expected message, and whether the kind always exits.
struct Case {
    name: &'static str,
    body: &'static str,
    bad: i64,
    good: i64,
    message: &'static str,
    fatal: bool,
}

const CASES: &[Case] = &[
    Case {
        name: "signed_overflow",
        body: "  %a = trunc %v : i32\n  %r = add nsw %a, i32 2147483647 : i32 ; UB\n  %rr = sext %r : i64\n",
        bad: 1,
        good: 0,
        message: "signed integer overflow: 1 + 2147483647 cannot be represented in type i32",
        fatal: false,
    },
    Case {
        name: "unsigned_overflow",
        body: "  %rr = sub nuw %v, i64 5 : i64 ; UB\n",
        bad: 3,
        good: 7,
        message: "unsigned integer overflow: 3 - 5 cannot be represented in type i64",
        fatal: false,
    },
    Case {
        name: "shift_exponent",
        body: "  %rr = shl i64 1, %v : i64 ; UB\n",
        bad: 64,
        good: 3,
        message: "shift exponent 64 is too large for 64-bit type",
        fatal: false,
    },
    Case {
        name: "shift_base",
        body: "  %a = trunc %v : i32\n  %r = shl nsw %a, i32 30 : i32 ; UB\n  %rr = sext %r : i64\n",
        bad: 2,
        good: 1,
        message: "left shift of 2 by 30 places cannot be represented in type i32",
        fatal: false,
    },
    Case {
        name: "div_by_zero",
        body: "  %rr = sdiv i64 10, %v : i64 ; UB\n",
        bad: 0,
        good: 2,
        message: "division by zero",
        fatal: true,
    },
    Case {
        name: "div_overflow",
        body: "  %rr = srem i64 -9223372036854775808, %v : i64 ; UB\n",
        bad: -1,
        good: 3,
        message: "division of -9223372036854775808 by -1 cannot be represented in type i64",
        fatal: true,
    },
    Case {
        name: "exact",
        body: "  %rr = udiv exact %v, i64 4 : i64 ; UB\n",
        bad: 6,
        good: 8,
        message: "exact operation lost information: 6 / 4",
        fatal: false,
    },
    Case {
        name: "float_cast",
        body: "  %f = sitofp %v : f64\n  %r = fptosi %f : i8 ; UB\n  %rr = sext %r : i64\n",
        bad: 300,
        good: -100,
        message: "floating-point value is outside the range of representable values of type i8",
        fatal: false,
    },
    Case {
        name: "pointer_bounds",
        body: "  %s = alloca [8 x i8] : ptr\n  %p = ptr_add inbounds %s, %v : ptr ; UB\n  %rr = ptrtoint %p : i64\n",
        bad: 9,
        good: 8,
        message: "pointer offset 9 is out of bounds for an object of 8 bytes",
        fatal: false,
    },
    Case {
        name: "object_size",
        body: "  %s = alloca [8 x i8] : ptr\n  %p = ptr_add %s, %v : ptr\n  store i32 7, %p align 1 : i32 ; UB\n  %rr = load %s align 1 : i64\n",
        bad: 6,
        good: 4,
        message: "store at offset 6 is out of bounds for an object of 8 bytes",
        fatal: false,
    },
    Case {
        name: "null",
        body: "  %z = icmp eq %v, i64 0 : i1\n  %p = select %z, ptr null, @buf : ptr\n  %r = load %p align 1 : i8 ; UB\n  %rr = zext %r : i64\n",
        bad: 0,
        good: 1,
        message: "load of null pointer",
        fatal: true,
    },
    Case {
        name: "alignment",
        body: "  %p = ptr_add @buf, %v : ptr\n  %r = load %p align 4 : i32 ; UB\n  %rr = zext %r : i64\n",
        bad: 2,
        good: 4,
        message: "load of misaligned address 0x",
        fatal: false,
    },
    Case {
        name: "unreachable",
        body: "  %c = icmp eq %v, i64 1 : i1\n  cond_br %c, ^1, ^2\n^1:\n  unreachable ; UB\n^2:\n  %rr = add %v, i64 0 : i64\n",
        bad: 1,
        good: 0,
        message: "execution reached an unreachable program point",
        fatal: true,
    },
];

fn source(case: &Case, v: i64) -> String {
    format!(
        "module \"{name}\"\nglobal @v : i64 = i64 {v}\nglobal @buf : [4 x i32] = [4 x i32] (i32 1, i32 2, i32 3, i32 4)\n\
         global @sink : i64 = i64 0\n\nfunc @main() -> i32 {{\nentry ^0:\n  %v = load @v align 8 : i64\n{body}  \
         store volatile %rr, @sink align 8 : i64\n  ret i32 0\n}}\n",
        name = case.name,
        body = case.body,
    )
}

/// The 1-based line of the `; UB` marker.
fn ub_line(src: &str) -> usize {
    src.lines().position(|l| l.contains("; UB")).expect("a marked line") + 1
}

#[test]
fn each_kind_reports_its_location_and_correct_programs_run_cleanly() {
    let dir = scratch("kinds");
    for case in CASES {
        for opt in ["-O0", "-O2"] {
            // The faulting program reports, at the marked line.
            let src = source(case, case.bad);
            let path = dir.join(format!("{}.lf", case.name));
            std::fs::write(&path, &src).unwrap();
            let exe = dir.join(case.name);
            lf_build(&path, &exe, &[opt, "--sanitize=undefined"]);
            let out = run(&exe);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let want = format!("{}:{}: runtime error: {}", path.display(), ub_line(&src), case.message);
            assert!(stderr.contains(&want), "{} {opt}: expected `{want}`, got:\n{stderr}", case.name);
            assert_eq!(stderr.lines().count(), 1, "{} {opt}: one report:\n{stderr}", case.name);
            let code = if case.fatal { 1 } else { 0 };
            assert_eq!(out.status.code(), Some(code), "{} {opt}: exit status", case.name);

            // Without recovery, every kind exits after its report.
            lf_build(&path, &exe, &[opt, "--sanitize=undefined", "--sanitize-halt"]);
            assert_eq!(run(&exe).status.code(), Some(1), "{} {opt} --sanitize-halt", case.name);

            // The correct program runs cleanly with every check on.
            let src = source(case, case.good);
            std::fs::write(&path, &src).unwrap();
            lf_build(&path, &exe, &[opt, "--sanitize=undefined"]);
            let out = run(&exe);
            assert!(out.stderr.is_empty(), "{} {opt}: clean run reported:\n{}", case.name, String::from_utf8_lossy(&out.stderr));
            assert_eq!(out.status.code(), Some(0), "{} {opt}: clean exit", case.name);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trap_mode_dies_with_sigill_and_a_disabled_kind_is_not_checked() {
    let dir = scratch("trap");
    for case in CASES.iter().filter(|c| ["signed_overflow", "div_by_zero", "object_size", "unreachable"].contains(&c.name)) {
        let src = source(case, case.bad);
        let path = dir.join(format!("{}.lf", case.name));
        std::fs::write(&path, &src).unwrap();
        let exe = dir.join(case.name);
        lf_build(&path, &exe, &["-O2", "--sanitize=undefined", "--sanitize-trap"]);
        let out = run(&exe);
        assert_eq!(out.status.signal(), Some(4), "{}: SIGILL expected, got {:?}", case.name, out.status);
        assert!(out.stderr.is_empty(), "{}: trap mode prints nothing", case.name);
    }
    // Only the selected kinds are checked: an overflow under `--sanitize=null`
    // runs on silently.
    let case = &CASES[0];
    let path = dir.join("only_null.lf");
    std::fs::write(&path, source(case, case.bad)).unwrap();
    let exe = dir.join("only_null");
    lf_build(&path, &exe, &["--sanitize=null,shift"]);
    let out = run(&exe);
    assert!(out.stderr.is_empty() && out.status.success(), "{out:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A sieve of Eratosthenes over a stack array plus a checked arithmetic
/// series: every access and operation is checked, none fails.
const BENCH: &str = r#"
module "bench"

func @main() -> i32 {
entry ^0:
  %s = alloca [20000 x i8] : ptr
  br ^1(i64 0)
^1(%i: i64):
  %p = ptr_add inbounds %s, %i : ptr
  store i8 1, %p align 1 : i8
  %i2 = add nsw %i, i64 1 : i64
  %d = icmp slt %i2, i64 20000 : i1
  cond_br %d, ^1(%i2), ^2(i64 2, i64 0)
^2(%n: i64, %count: i64):
  %np = ptr_add inbounds %s, %n : ptr
  %flag = load %np align 1 : i8
  %prime = icmp ne %flag, i8 0 : i1
  %inc = zext %prime : i64
  %c2 = add nsw %count, %inc : i64
  %sq = mul nsw %n, %n : i64
  %small = icmp slt %sq, i64 20000 : i1
  %mark = and %prime, %small : i1
  cond_br %mark, ^3(%sq), ^4
^3(%m: i64):
  %mp = ptr_add inbounds %s, %m : ptr
  store i8 0, %mp align 1 : i8
  %m2 = add nsw %m, %n : i64
  %more = icmp slt %m2, i64 20000 : i1
  cond_br %more, ^3(%m2), ^4
^4:
  %n2 = add nsw %n, i64 1 : i64
  %go = icmp slt %n2, i64 20000 : i1
  cond_br %go, ^2(%n2, %c2), ^5(%c2)
^5(%total: i64):
  %q = sdiv %total, i64 10 : i64
  %r = srem %q, i64 256 : i64
  %t = trunc %r : i32
  ret %t
}
"#;

#[test]
fn a_checked_benchmark_computes_the_same_result() {
    // 2262 primes below 20000: exit status (2262 / 10) % 256 = 226.
    let dir = scratch("bench");
    let path = dir.join("bench.lf");
    std::fs::write(&path, BENCH).unwrap();
    let exe = dir.join("bench");
    for flags in [&["-O0"][..], &["-O0", "--sanitize=undefined"], &["-O2", "--sanitize=undefined"], &["-O2", "--sanitize=undefined", "--sanitize-trap"]] {
        lf_build(&path, &exe, flags);
        let out = run(&exe);
        assert!(out.stderr.is_empty(), "{flags:?}: {}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(out.status.code(), Some(226), "{flags:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn other_linux_targets_link_a_sanitized_program() {
    let dir = scratch("targets");
    let case = &CASES[0];
    let path = dir.join("t.lf");
    std::fs::write(&path, source(case, case.bad)).unwrap();
    for target in ["aarch64-linux", "riscv64-linux"] {
        let exe = dir.join(target);
        lf_build(&path, &exe, &["--target", target, "--sanitize=undefined"]);
        assert!(exe.exists(), "{target}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
