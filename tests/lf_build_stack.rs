//! `lf build --stack-usage [--no-stack-probes]` through the real driver binary:
//! the table lists every function with its frame, the worst-case depth from the
//! entry follows the deepest path, and the program runs with and without
//! probes.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
module "su"
func @leaf(i64) -> i64 {
entry ^0(%x: i64):
  %a = alloca [10000 x i8] : ptr
  store i8 1, %a align 1 : i8
  %r = add %x, i64 1 : i64
  ret %r
}
func @mid(i64) -> i64 {
entry ^0(%x: i64):
  %r = call @leaf(%x) : i64
  %e = syscall i64 39 : i64
  ret %r
}
func @rec(i64) -> i64 {
entry ^0(%x: i64):
  %r = call @rec(%x) : i64
  ret %r
}
func @main() -> i64 {
entry ^0:
  %r = call @mid(i64 41) : i64
  %s = sub %r, i64 42 : i64
  ret %s
}
"#;

fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-build-stack-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn lf_build_reports_stack_usage() {
    let dir = scratch_dir();
    let (src, exe) = (dir.join("su.lf"), dir.join("su"));
    std::fs::write(&src, SRC).unwrap();
    for extra in [&[][..], &["--no-stack-probes"][..], &["--entry", "rec"][..]] {
        let out = Command::new(env!("CARGO_BIN_EXE_lf"))
            .arg("build")
            .arg(&src)
            .arg("-o")
            .arg(&exe)
            .arg("--stack-usage")
            .args(extra)
            .output()
            .expect("run lf");
        assert!(out.status.success(), "lf build {extra:?}: {}", String::from_utf8_lossy(&out.stderr));
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        if extra.first() == Some(&"--entry") {
            // Only `rec` is reachable from `rec`.
            assert!(text.lines().any(|l| l.starts_with("rec ")), "{text}");
            assert!(!text.lines().any(|l| l.starts_with("main ")), "{text}");
            assert!(text.contains("(3 function(s) not reachable from 'rec' not shown"), "{text}");
            assert!(
                text.contains("worst-case stack from 'rec': no bound (1 reason):\n  rec: 'rec' calls itself\n"),
                "{text}"
            );
            continue;
        }
        for name in ["leaf", "mid", "main", "<syscall>"] {
            assert!(text.contains(name), "{extra:?}: table lacks {name}:\n{text}");
        }
        // `rec` is dead code from `main`: not listed, and no obstacle.
        assert!(!text.lines().any(|l| l.starts_with("rec ")), "{text}");
        assert!(text.contains("(1 function(s) not reachable from 'main' not shown"), "{text}");
        let line = text.lines().find(|l| l.starts_with("worst-case stack from 'main'")).expect(&text);
        assert!(line.ends_with("(main -> mid -> leaf)"), "{line}");
        let run = loop {
            match Command::new(&exe).output() {
                Ok(o) => break o,
                // ETXTBSY from a concurrent fork.
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec built program: {e}"),
            }
        };
        assert_eq!(run.status.code(), Some(0), "{extra:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every obstacle to a bound at once (issue #6): two recursions, a
/// `dyn_alloca`, an indirect call and an unknown callee, each with its call
/// path from the entry; the dead `std.io.eprint` neither blocks nor shows.
const BLOCKED: &str = r#"
module "blocked"
func @ext(i64) -> i64
func @main.fib(i64) -> i64 {
entry ^0(%x: i64):
  %c = icmp ult %x, i64 2 : i1
  cond_br %c, ^1, ^2
^1:
  ret %x
^2:
  %y = sub %x, i64 1 : i64
  %r = call @main.fib(%y) : i64
  ret %r
}
func @main.even(i64) -> i64 {
entry ^0(%x: i64):
  %r = call @main.odd(%x) : i64
  ret %r
}
func @main.odd(i64) -> i64 {
entry ^0(%x: i64):
  %r = call @main.even(%x) : i64
  ret %r
}
func @main.dyn(i64) -> i64 {
entry ^0(%x: i64):
  %n = and %x, i64 255 : i64
  %p = dyn_alloca %n align 16 : ptr
  store i64 1, %p align 8 : i64
  ret %x
}
func @main.ind(i64) -> i64 {
entry ^0(%x: i64):
  %fp = select i1 1, @main.dyn, @ext : ptr
  %r = call %fp(%x) : i64
  ret %r
}
func @main.main() -> i64 {
entry ^0:
  %a = call @main.fib(i64 10) : i64
  %b = call @main.even(%a) : i64
  %c = call @main.dyn(%b) : i64
  %d = call @main.ind(%c) : i64
  %e = call @ext(%d) : i64
  ret %e
}
func @main() -> i64 {
entry ^0:
  %r = call @main.main() : i64
  ret %r
}
func @std.io.eprint(i64) -> i64 {
entry ^0(%x: i64):
  %r = call @std.io.eprint(%x) : i64
  ret %r
}
"#;

#[test]
fn lf_build_reports_every_obstacle_with_its_path() {
    let dir = scratch_dir().join("blocked");
    std::fs::create_dir_all(&dir).unwrap();
    let (src, obj) = (dir.join("blocked.lf"), dir.join("blocked.o"));
    std::fs::write(&src, BLOCKED).unwrap();
    let run = |extra: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_lf"))
            .arg("build")
            .arg(&src)
            .args(["-c", "-o"])
            .arg(&obj)
            .args(extra)
            .output()
            .expect("run lf");
        assert!(out.status.success(), "lf build {extra:?}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let text = run(&["--stack-usage"]);
    let expected = "worst-case stack from 'main': no bound (5 reasons):\n\
        \x20 main -> main.main -> main.fib: 'main.fib' calls itself\n\
        \x20 main -> main.main -> main.even: recursion main.even -> main.odd -> main.even\n\
        \x20 main -> main.main -> main.dyn: 'main.dyn' uses dyn_alloca with no assumed bound\n\
        \x20 main -> main.main -> main.ind: 'main.ind' makes an indirect call with no assumed bound\n\
        \x20 main -> main.main -> ext: 'main.main' calls 'ext', whose stack usage is unknown\n";
    assert!(text.ends_with(expected), "{text}");
    assert!(!text.contains("std.io.eprint"), "{text}");
    assert!(text.contains("(1 function(s) not reachable from 'main' not shown; --stack-usage=all"), "{text}");
    // `--stack-usage=all` lists the dead function too, with the same reasons.
    let all = run(&["--stack-usage=all"]);
    assert!(all.lines().any(|l| l.starts_with("std.io.eprint ")), "{all}");
    assert!(all.ends_with(expected) && !all.contains("not shown"), "{all}");
    let _ = std::fs::remove_dir_all(&dir);
}
