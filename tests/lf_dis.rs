//! `lf-dis` through the real binaries: objects `lf build -c` writes for each
//! target and format, flat binaries, address ranges, and the errors.

use std::path::{Path, PathBuf};
use std::process::Command;

const SRC: &str = r#"
module "dis"
global @k : i64 = i64 2
func @ext(i64) -> i64
func @helper(i64) -> i64 {
entry ^0(%a: i64):
  %b = load @k align 8 : i64
  %s = add %a, %b : i64
  %t = call @ext(%s) : i64
  ret %t
}
func @main() -> i64 {
entry ^0:
  %r = call @helper(i64 40) : i64
  ret %r
}
"#;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-dis-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("dis.lf"), SRC).unwrap();
    dir
}

/// `lf build dis.lf -c <args> -o <out>`.
fn build(dir: &Path, args: &[&str], out: &str) -> PathBuf {
    let path = dir.join(out);
    let o = Command::new(env!("CARGO_BIN_EXE_lf"))
        .arg("build")
        .arg(dir.join("dis.lf"))
        .arg("-c")
        .args(args)
        .arg("-o")
        .arg(&path)
        .output()
        .expect("run lf");
    assert!(o.status.success(), "lf build {args:?}: {}", String::from_utf8_lossy(&o.stderr));
    path
}

/// Run `lf-dis args..`: (success, stdout, stderr).
fn dis(args: &[&str]) -> (bool, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_lf-dis")).args(args).output().expect("run lf-dis");
    (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

fn dis_ok(args: &[&str]) -> String {
    let (ok, out, err) = dis(args);
    assert!(ok, "lf-dis {args:?}: {err}");
    out
}

#[test]
fn objects_of_every_target_and_format() {
    let dir = scratch("objects");
    for (args, file, format, reloc) in [
        (&[][..], "x.o", "elf64-x86-64", "R_X86_64_PLT32 ext-0x4"),
        (&["--target", "x86_64-windows"][..], "x.obj", "coff-x86-64", "IMAGE_REL_AMD64_REL32 ext"),
        (&["--target", "x86_64-apple-darwin"][..], "x.mo", "mach-o x86-64", "X86_64_RELOC_BRANCH _ext"),
        (&["--target", "aarch64-windows"][..], "a.obj", "coff-ARM64", "IMAGE_REL_ARM64_BRANCH26 ext"),
        (&["--target", "arm64-apple-darwin"][..], "a.mo", "mach-o arm64", "ARM64_RELOC_BRANCH26 _ext"),
        (&["--target", "thumbv7m-none-eabi"][..], "t.o", "elf32-littlearm", "R_ARM_THM_CALL ext"),
        (&["--target", "avr-atmega328p"][..], "avr.o", "elf32-avr", "R_AVR_CALL ext"),
        (&["--target", "wasm32"][..], "w.o", "wasm (relocatable)", "R_WASM_FUNCTION_INDEX_LEB ext"),
    ] {
        let path = build(&dir, args, file);
        let p = path.to_str().unwrap();
        let out = dis_ok(&[p]);
        assert!(out.contains(&format!("file format {format}")), "{format}:\n{out}");
        assert!(out.contains("Disassembly of section"), "{format}:\n{out}");
        assert!(out.contains("helper>:") && out.contains("main>:"), "{format}: labels\n{out}");
        assert!(out.contains(reloc), "{format}: relocation note `{reloc}`\n{out}");
        // -d is the default; --no-relocs drops the notes.
        assert_eq!(dis_ok(&["-d", p]), out);
        assert!(!dis_ok(&["--no-relocs", p]).contains(reloc));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn raw_binaries_and_ranges() {
    let dir = scratch("raw");
    let bin = dir.join("code.bin");
    // x86-64: push %rbp; mov %rsp,%rbp; nop; pop %rbp; ret
    std::fs::write(&bin, [0x55, 0x48, 0x89, 0xe5, 0x90, 0x5d, 0xc3]).unwrap();
    let p = bin.to_str().unwrap();
    let out = dis_ok(&["--raw", "--arch", "x86_64", "--base", "0x1000", p]);
    assert!(out.contains("    1000: 55"), "{out}");
    assert!(out.contains("    1006: c3"), "{out}");
    let insts = |s: &str| s.lines().filter(|l| l.starts_with(' ') && l.contains(": ")).count();
    let ranged = dis_ok(&["--raw", "--arch=x86_64", "--base=0x1000", "--start", "0x1004", "--stop=0x1006", p]);
    assert!(ranged.contains("    1004: 90") && !ranged.contains("1006:") && !ranged.contains("1000:"), "{ranged}");
    assert!(insts(&ranged) >= 1);
    let bare = dis_ok(&["--raw", "--arch", "x86_64", "--no-show-raw-insn", p]);
    assert!(!bare.contains(" 55 "), "{bare}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn errors_and_help() {
    let (ok, out, _) = dis(&["--help"]);
    assert!(ok && out.contains("usage: lf-dis"));
    let (ok, out, _) = dis(&["--version"]);
    assert!(ok && out.starts_with("lf-dis"));
    let dir = scratch("errors");
    let junk = dir.join("junk.bin");
    std::fs::write(&junk, b"not an object").unwrap();
    let j = junk.to_str().unwrap();
    let (ok, _, err) = dis(&[j]);
    assert!(!ok && err.contains("unrecognized file format"), "{err}");
    let (ok, _, err) = dis(&["--raw", j]);
    assert!(!ok && err.contains("--raw needs --arch"), "{err}");
    let (ok, _, err) = dis(&["--arch", "pdp11", j]);
    assert!(!ok && err.contains("unknown architecture"), "{err}");
    let (ok, _, err) = dis(&["--syntax", "weird", j]);
    assert!(!ok && err.contains("unknown syntax"), "{err}");
    let (ok, _, err) = dis(&[dir.join("missing.o").to_str().unwrap()]);
    assert!(!ok && err.contains("cannot read"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
