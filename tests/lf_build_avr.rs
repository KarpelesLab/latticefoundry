//! `lf build --target avr-atmega328p` through the real driver: Intel HEX and
//! raw-binary firmware, ELF32 `EM_AVR` objects, the stack report, and the
//! errors for what an AVR build cannot do. (Execution of AVR code is tested
//! in the library, on the AVR interpreter.)

use std::path::{Path, PathBuf};
use std::process::Command;

const SRC: &str = r#"
module "blink"
global constant addrspace(1) @pattern : [4 x i8] = [4 x i8] (i8 1, i8 3, i8 7, i8 15)
global @ticks : i16 = i16 0

func @step(i16) -> i8 {
entry ^0(%i: i16):
  %k = and %i, i16 3 : i16
  %p = ptr_add @pattern, %k : ptr addrspace(1)
  %v = load %p align 1 : i8
  ret %v
}

func @main() -> i16 {
entry ^0:
  %v = call @step(i16 6) : i8
  %w = zext %v : i16
  %t = udiv %w, i16 3 : i16
  store %t, @ticks align 1 : i16
  ret %t
}
"#;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-build-avr-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("blink.lf"), SRC).unwrap();
    dir
}

/// Run `lf build <dir>/blink.lf args..`, returning (success, stdout, stderr).
fn lf(dir: &Path, args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lf")).arg("build").arg(dir.join("blink.lf")).args(args).output().expect("run lf");
    (out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn ihex_firmware() {
    let dir = scratch("ihex");
    let out = dir.join("blink.hex");
    let (ok, _, err) = lf(&dir, &["--target", "avr-atmega328p", "--oformat", "ihex", "-o", out.to_str().unwrap()]);
    assert!(ok, "{err}");
    let hex = std::fs::read_to_string(&out).unwrap();
    let lines: Vec<&str> = hex.lines().collect();
    assert!(lines.iter().all(|l| l.starts_with(':')));
    // Data from address 0, the reset vector a `jmp` (0x940c, little-endian).
    assert!(lines[0].starts_with(":10000000") && lines[0][9..13].eq_ignore_ascii_case("0C94"), "{}", lines[0]);
    assert_eq!(*lines.last().unwrap(), ":00000001FF");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn binary_firmware_and_objects() {
    let dir = scratch("bin");
    let bin = dir.join("blink.bin");
    let (ok, _, err) = lf(&dir, &["--target", "avr", "--oformat", "binary", "-o", bin.to_str().unwrap()]);
    assert!(ok, "{err}");
    let image = std::fs::read(&bin).unwrap();
    assert_eq!(&image[..2], &[0x0c, 0x94]);
    assert!(image.len() < 32 * 1024);
    let obj = dir.join("blink.o");
    let (ok, _, err) = lf(&dir, &["--target", "avr-atmega328p", "-c", "-o", obj.to_str().unwrap()]);
    assert!(ok, "{err}");
    let elf = std::fs::read(&obj).unwrap();
    assert_eq!(&elf[..5], b"\x7fELF\x01", "ELF32");
    assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), 83, "EM_AVR");
    assert_eq!(u32::from_le_bytes(elf[36..40].try_into().unwrap()), 5, "e_flags: avr5");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stack_report_includes_the_runtime() {
    let dir = scratch("stack");
    let out = dir.join("s.hex");
    let (ok, stdout, err) =
        lf(&dir, &["--target", "avr-atmega328p", "--oformat", "ihex", "--stack-usage", "-o", out.to_str().unwrap()]);
    assert!(ok, "{err}");
    assert!(stdout.contains("__lf_udiv_i16"), "{stdout}");
    assert!(stdout.contains("worst-case stack from 'main':") && !stdout.contains("unbounded"), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn avr_build_errors() {
    let dir = scratch("errors");
    let (ok, _, err) = lf(&dir, &["--target", "avr-atmega328p"]);
    assert!(!ok && err.contains("--oformat ihex"), "{err}");
    let (ok, _, err) = lf(&dir, &["--target", "avr-atmega328p", "--oformat", "ihex", "--base", "0x100"]);
    assert!(!ok && err.contains("--base"), "{err}");
    let (ok, _, err) = lf(&dir, &["--target", "avr-atmega9999", "-c"]);
    assert!(!ok && err.contains("unknown target"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
