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
    // A RISC-V ELF64 object: EM_RISCV, the LP64D double-float ABI flag.
    let rv = build_ok(&dir, &["-c", "--target", "riscv64-linux"], "r.o");
    assert_eq!((rv[4], u16::from_le_bytes([rv[18], rv[19]])), (2, 243), "ELF64 EM_RISCV");
    assert_eq!(u32::from_le_bytes(rv[48..52].try_into().unwrap()), 0x4, "EF_RISCV_FLOAT_ABI_DOUBLE");
    // --format overrides the triple's format (a Mach-O object of Linux code).
    let macho = build_ok(&dir, &["-c", "--format", "macho"], "f.mo");
    assert_eq!(&macho[..4], &[0xcf, 0xfa, 0xed, 0xfe]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn riscv64_executables_link_through_qld() {
    let dir = scratch("riscv64");
    let exe = build_ok(&dir, &["--target", "riscv64-linux"], "t");
    assert_eq!(&exe[..4], b"\x7fELF");
    assert_eq!(u16::from_le_bytes([exe[16], exe[17]]), 2, "ET_EXEC");
    assert_eq!(u16::from_le_bytes([exe[18], exe[19]]), 243, "EM_RISCV");
    assert_ne!(u64::from_le_bytes(exe[24..32].try_into().unwrap()), 0, "an entry point");
    let pic = build_ok(&dir, &["--target", "riscv64-linux", "-c", "--pic"], "p.o");
    assert_eq!(u16::from_le_bytes([pic[18], pic[19]]), 243);
    let so = build_ok(&dir, &["--target", "riscv64-linux", "--shared", "-soname", "libtg.so.1"], "libtg.so");
    assert_eq!(u16::from_le_bytes([so[16], so[17]]), 3, "ET_DYN");
    assert_eq!(u16::from_le_bytes([so[18], so[19]]), 243, "EM_RISCV");
    assert!(so.windows(10).any(|w| w == b"libtg.so.1"), "the DT_SONAME string");
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
    let (ok, err) = lf(&dir, &["-c", "--target", "riscv64-windows"]);
    assert!(!ok && err.contains("COFF"), "{err}");
    let (ok, err) = lf(&dir, &["--format", "coff"]);
    assert!(!ok && err.contains("-c"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cortex_m_firmware_through_qld() {
    let dir = scratch("cortex-m");
    let t = ["--target", "thumbv7m-none-eabi"];
    // A relocatable Arm ELF object.
    let obj = build_ok(&dir, &[&t[..], &["-c"]].concat(), "t.o");
    assert_eq!((obj[4], u16::from_le_bytes([obj[18], obj[19]])), (1, 40), "ELF32 EM_ARM");
    // A linked ELF executable.
    let elf = build_ok(&dir, &t, "t.elf");
    assert_eq!(u16::from_le_bytes([elf[16], elf[17]]), 2, "ET_EXEC");
    // A flashable Intel HEX image at the requested flash origin: the vector
    // table (initial sp at the end of RAM, a Thumb reset vector) comes first.
    let args = [&t[..], &["--oformat", "ihex", "--base", "0x08000000"]].concat();
    let hex = build_ok(&dir, &args, "t.hex");
    let (bytes, start) = decode_ihex(std::str::from_utf8(&hex).unwrap());
    assert_eq!(bytes[0].0, 0x0800_0000);
    let word = |k: usize| u32::from_le_bytes([bytes[k].1, bytes[k + 1].1, bytes[k + 2].1, bytes[k + 3].1]);
    assert_eq!(word(0), 0x2001_0000);
    assert_eq!(word(4) & 1, 1);
    assert_eq!(start, Some(word(4)), "the start address is the reset handler");
    let args = [&t[..], &["--oformat", "binary", "--base", "0x08000000"]].concat();
    let bin = build_ok(&dir, &args, "t.bin");
    for (addr, b) in bytes {
        assert_eq!(bin[(addr - 0x0800_0000) as usize], b, "binary and ihex agree at {addr:#x}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
