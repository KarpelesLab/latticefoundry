//! End-to-end `lf build foo.lf -o foo` through the real driver binary, for a
//! freestanding program whose data lives in `.rodata`/`.data`/`.bss`: it
//! prints a string constant with the `write` syscall, through a pointer stored
//! in a constant table, and exits with a value computed from mutable globals.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::path::PathBuf;
use std::process::Command;

const SRC: &str = r#"
module "hello_data"
global constant @msg : [12 x i8] = [12 x i8] "hi from lf!\n"
global constant @ptrs : [1 x ptr] = [1 x ptr] (ptr @msg)
global @count : i64 = i64 40
global @scratch : [8 x i64] = [8 x i64] poison

func @main() -> i64 {
entry ^0:
  %p = load @ptrs align 8 : ptr
  %n = syscall i64 1, i64 1, %p, i64 12 : i64
  %s = ptr_add @scratch, i64 56 : ptr
  store i64 2, %s align 8 : i64
  %c = load @count align 8 : i64
  %t = load %s align 8 : i64
  %r = add %c, %t : i64
  ret %r
}
"#;

fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-build-data-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn lf_build_links_and_runs_a_program_with_global_data() {
    let dir = scratch_dir();
    let (src, exe) = (dir.join("hello_data.lf"), dir.join("hello_data"));
    std::fs::write(&src, SRC).unwrap();
    for opt in ["-O0", "-O2"] {
        let out = Command::new(env!("CARGO_BIN_EXE_lf"))
            .arg("build")
            .arg(&src)
            .arg("-o")
            .arg(&exe)
            .arg(opt)
            .output()
            .expect("run lf");
        assert!(out.status.success(), "lf build {opt} failed: {}", String::from_utf8_lossy(&out.stderr));
        // Retry a transient ETXTBSY (errno 26) from a concurrent fork.
        let run = loop {
            match Command::new(&exe).output() {
                Ok(o) => break o,
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec built program: {e}"),
            }
        };
        assert_eq!(run.stdout, b"hi from lf!\n", "stdout at {opt}");
        assert_eq!(run.status.code(), Some(42), "exit status at {opt}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
