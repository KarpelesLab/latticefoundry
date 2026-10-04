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

/// `(p_flags of each PT_LOAD, e_phnum)` of an ELF64 image.
fn load_flags(image: &[u8]) -> (Vec<u32>, usize) {
    let u16_at = |o: usize| u16::from_le_bytes([image[o], image[o + 1]]) as usize;
    let u32_at = |o: usize| u32::from_le_bytes(image[o..o + 4].try_into().unwrap());
    let phnum = u16_at(56);
    let flags = (0..phnum).map(|i| 64 + i * 56).filter(|&o| u32_at(o) == 1).map(|o| u32_at(o + 4)).collect();
    (flags, phnum)
}

/// The size-layout options: `--merge-rodata[=]`, `--function-alignment=` and
/// the `-Os` preset (which an explicit option overrides) pick the segments,
/// with debug info and unwind tables too, and the program runs the same.
#[test]
fn lf_build_size_layout_options() {
    const RX: u32 = 5;
    const R: u32 = 4;
    const RW: u32 = 6;
    // Not under `scratch_dir()`, which the other test removes when it is done.
    let dir = std::env::temp_dir().join(format!("lf-build-size-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, exe) = (dir.join("hello_data.lf"), dir.join("hello_data"));
    std::fs::write(&src, SRC).unwrap();
    let cases: &[(&[&str], &[u32])] = &[
        (&["--merge-rodata=never"], &[RX, R, RW]),
        (&[], &[RX, R, RW]),
        (&["--merge-rodata=auto"], &[RX, RW]),
        (&["--merge-rodata"], &[RX, RW]),
        (&["--function-alignment=1", "--merge-rodata=never"], &[RX, R, RW]),
        (&["-Os"], &[RX, RW]),
        (&["-Os", "--merge-rodata=never"], &[RX, R, RW]),
        (&["--merge-rodata=never", "-Os", "--function-alignment=64"], &[RX, R, RW]),
        (&["-Os", "-g"], &[RX, RW]),
        (&["-Os", "--unwind-tables"], &[RX, RW]),
        (&["--unwind-tables", "--merge-rodata=never"], &[RX, R, RW]),
    ];
    let mut sizes = Vec::new();
    for (args, want) in cases {
        let out = Command::new(env!("CARGO_BIN_EXE_lf"))
            .arg("build")
            .arg(&src)
            .args(*args)
            .arg("-o")
            .arg(&exe)
            .output()
            .expect("run lf");
        assert!(out.status.success(), "lf build {args:?} failed: {}", String::from_utf8_lossy(&out.stderr));
        let image = std::fs::read(&exe).unwrap();
        let (flags, phnum) = load_flags(&image);
        assert_eq!(flags, *want, "{args:?}");
        assert_eq!(phnum, want.len(), "{args:?}: only PT_LOADs");
        sizes.push(image.len());
        let run = loop {
            match Command::new(&exe).output() {
                Ok(o) => break o,
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec built program: {e}"),
            }
        };
        assert_eq!(run.stdout, b"hi from lf!\n", "stdout with {args:?}");
        assert_eq!(run.status.code(), Some(42), "exit status with {args:?}");
    }
    // The default is not merged; merging saves a program header; -Os packs
    // the functions too.
    assert_eq!(sizes[1], sizes[0], "{sizes:?}");
    assert!(sizes[3] < sizes[1] && sizes[5] <= sizes[3], "{sizes:?}");
    for (bad, msg) in [
        ("--function-alignment=3", "power of two"),
        ("--function-alignment=0", "power of two"),
        ("--function-alignment=x", "power of two"),
        ("--merge-rodata=sometimes", "auto, always or never"),
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_lf")).arg("build").arg(&src).arg(bad).output().expect("run lf");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success() && err.contains(msg), "{bad}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
