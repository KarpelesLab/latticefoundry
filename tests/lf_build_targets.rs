//! `lf build --target <triple>`, `-c [--format]` and `--oformat binary|ihex`
//! through the real driver binary: the object formats, a Windows PE link, the
//! firmware images, and the errors for combinations that cannot be built.

use std::path::{Path, PathBuf};
use std::process::Command;

const SRC: &str = r#"
module "tg"
global @k : i64 = i64 2
func @helper(i64) -> i64 {
entry ^0(%a: i64):
  %b = load @k align 8 : i64
  %s = add %a, %b : i64
  ret %s
}
func @main() -> i64 {
entry ^0:
  %r = call @helper(i64 40) : i64
  ret %r
}
"#;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-build-targets-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("tg.lf");
    std::fs::write(&src, SRC).unwrap();
    dir
}

/// Run `lf build <dir>/tg.lf args..`, returning (success, stderr).
fn lf(dir: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lf"))
        .arg("build")
        .arg(dir.join("tg.lf"))
        .args(args)
        .output()
        .expect("run lf");
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

fn build_ok(dir: &Path, args: &[&str], out: &str) -> Vec<u8> {
    let path = dir.join(out);
    let path_s = path.to_str().unwrap().to_owned();
    let mut all: Vec<&str> = args.to_vec();
    all.extend(["-o", &path_s]);
    let (ok, err) = lf(dir, &all);
    assert!(ok, "lf build {args:?}: {err}");
    std::fs::read(&path).unwrap()
}

#[test]
fn objects_in_each_format() {
    let dir = scratch("objects");
    let elf = build_ok(&dir, &["-c"], "t.o");
    assert_eq!(&elf[..4], b"\x7fELF");
    let coff = build_ok(&dir, &["-c", "--target", "x86_64-pc-windows-msvc"], "t.obj");
    assert_eq!(u16::from_le_bytes([coff[0], coff[1]]), 0x8664);
    let coff = build_ok(&dir, &["-c", "--target", "aarch64-windows"], "a.obj");
    assert_eq!(u16::from_le_bytes([coff[0], coff[1]]), 0xaa64);
    let macho = build_ok(&dir, &["-c", "--target", "x86_64-apple-darwin"], "t.mo");
    assert_eq!(&macho[..8], &[0xcf, 0xfa, 0xed, 0xfe, 7, 0, 0, 1]);
    let macho = build_ok(&dir, &["-c", "--target", "arm64-apple-darwin"], "a.mo");
    assert_eq!(&macho[..8], &[0xcf, 0xfa, 0xed, 0xfe, 12, 0, 0, 1]);
    // --format overrides the triple's format (a Mach-O object of Linux code).
    let macho = build_ok(&dir, &["-c", "--format", "macho"], "f.mo");
    assert_eq!(&macho[..4], &[0xcf, 0xfa, 0xed, 0xfe]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn windows_executables_link_through_qld() {
    let dir = scratch("pe");
    for (triple, machine) in [("x86_64-windows", 0x8664u16), ("aarch64-windows", 0xaa64)] {
        let pe = build_ok(&dir, &["--target", triple], "t.exe");
        assert_eq!(&pe[..2], b"MZ");
        let off = u32::from_le_bytes(pe[0x3c..0x40].try_into().unwrap()) as usize;
        assert_eq!(&pe[off..off + 4], b"PE\0\0");
        assert_eq!(u16::from_le_bytes([pe[off + 4], pe[off + 5]]), machine);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Decode Intel HEX, checking each record's checksum: `(address, byte)`s and
/// the start linear address.
fn decode_ihex(text: &str) -> (Vec<(u32, u8)>, Option<u32>) {
    let (mut bytes, mut upper, mut start) = (Vec::new(), 0u32, None);
    for line in text.lines().map(str::trim_end).filter(|l| !l.is_empty()) {
        let raw: Vec<u8> =
            (1..line.len()).step_by(2).map(|i| u8::from_str_radix(&line[i..i + 2], 16).unwrap()).collect();
        assert_eq!(raw.iter().map(|&b| u32::from(b)).sum::<u32>() & 0xff, 0, "checksum of {line}");
        let off = u32::from(u16::from_be_bytes([raw[1], raw[2]]));
        let data = &raw[4..raw.len() - 1];
        match raw[3] {
            0 => bytes.extend(data.iter().enumerate().map(|(k, &b)| ((upper << 16) + off + k as u32, b))),
            4 => upper = u32::from(u16::from_be_bytes([data[0], data[1]])),
            5 => start = Some(u32::from_be_bytes(data.try_into().unwrap())),
            1 => break,
            t => panic!("record type {t}"),
        }
    }
    (bytes, start)
}

#[test]
fn firmware_images() {
    let dir = scratch("fw");
    let hex = build_ok(&dir, &["--oformat", "ihex", "--base", "0x100000"], "t.hex");
    let (bytes, start) = decode_ihex(std::str::from_utf8(&hex).unwrap());
    assert_eq!(start, Some(0x10_0000), "entry is the first byte of code");
    assert_eq!(bytes[0].0, 0x10_0000);
    let bin = build_ok(&dir, &["--oformat", "binary", "--base", "0x100000"], "t.bin");
    for (addr, b) in bytes {
        assert_eq!(bin[(addr - 0x10_0000) as usize], b, "binary and ihex agree at {addr:#x}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unsupported_combinations_are_reported() {
    let dir = scratch("errors");
    let (ok, err) = lf(&dir, &["--target", "x86_64-apple-darwin", "-o", "/dev/null"]);
    assert!(!ok && err.contains("-c"), "{err}");
    let (ok, err) = lf(&dir, &["--target", "x86_64-windows", "--oformat", "ihex"]);
    assert!(!ok && err.contains("oformat"), "{err}");
    let (ok, err) = lf(&dir, &["--target", "sparc-sun-solaris"]);
    assert!(!ok && err.contains("unknown target"), "{err}");
    let (ok, err) = lf(&dir, &["-c", "--target", "riscv64-linux"]);
    assert!(!ok && err.contains("ELF"), "{err}");
    let (ok, err) = lf(&dir, &["--format", "coff"]);
    assert!(!ok && err.contains("-c"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
