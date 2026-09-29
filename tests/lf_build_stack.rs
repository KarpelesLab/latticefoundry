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
        for name in ["leaf", "mid", "rec", "main", "<syscall>"] {
            assert!(text.contains(name), "{extra:?}: table lacks {name}:\n{text}");
        }
        if extra.first() == Some(&"--entry") {
            assert!(text.contains("unbounded: recursion: rec -> rec"), "{text}");
            continue;
        }
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
