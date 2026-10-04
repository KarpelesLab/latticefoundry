//! `lf build --target aarch64-linux` through the real driver binary: an ELF64
//! `EM_AARCH64` object (`-c`, `-c --pic`), a static executable and a shared
//! library linked by qld, and `-g` line tables. (The programs themselves run
//! on the A64 emulator in the library's unit tests; this host cannot execute
//! them.)

use std::path::{Path, PathBuf};
use std::process::Command;

const SRC: &str = r#"
module "a64"
global @k : i64 = i64 2
global constant @ptrs : [1 x ptr] = [1 x ptr] (ptr @k)
func @helper(i64) -> i64 {
entry ^0(%a: i64):
  %b = load @k align 8 : i64
  %s = add %a, %b : i64
  ret %s
}
func @main() -> i32 {
entry ^0:
  %r = call @helper(i64 40) : i64
  %t = trunc %r : i32
  ret %t
}
"#;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-build-aarch64-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("a64.lf"), SRC).unwrap();
    dir
}

/// `lf build <dir>/a64.lf --target aarch64-linux args.. -o <dir>/<out>`: the
/// output bytes.
fn build(dir: &Path, args: &[&str], out: &str) -> Result<Vec<u8>, String> {
    let path = dir.join(out);
    let o = Command::new(env!("CARGO_BIN_EXE_lf"))
        .arg("build")
        .arg(dir.join("a64.lf"))
        .args(["--target", "aarch64-linux"])
        .args(args)
        .arg("-o")
        .arg(&path)
        .output()
        .expect("run lf");
    if !o.status.success() {
        return Err(String::from_utf8_lossy(&o.stderr).into_owned());
    }
    Ok(std::fs::read(&path).unwrap())
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

/// `llvm-readobj args file`, when it is installed.
fn readobj(args: &[&str], file: &Path) -> Option<String> {
    let o = Command::new("llvm-readobj").args(args).arg(file).output().ok()?;
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    Some(String::from_utf8_lossy(&o.stdout).into_owned())
}

#[test]
fn object_executable_shared_library_and_debug_info() {
    let dir = scratch("all");
    // A relocatable object: ELF64, EM_AARCH64.
    let obj = build(&dir, &["-c"], "a.o").unwrap();
    assert_eq!((obj[4], u16_at(&obj, 16), u16_at(&obj, 18)), (2, 1, 183));
    if let Some(s) = readobj(&["-r"], &dir.join("a.o")) {
        for want in ["R_AARCH64_CALL26 helper", "R_AARCH64_ADR_PREL_PG_HI21 k", "R_AARCH64_ADD_ABS_LO12_NC k", "R_AARCH64_ABS64 k"] {
            assert!(s.contains(want), "{want}:\n{s}");
        }
    }
    // Position-independent: `k` (default visibility) through the GOT.
    build(&dir, &["-c", "--pic"], "p.o").unwrap();
    if let Some(s) = readobj(&["-r"], &dir.join("p.o")) {
        assert!(s.contains("R_AARCH64_ADR_GOT_PAGE k") && s.contains("R_AARCH64_LD64_GOT_LO12_NC k"), "{s}");
    }
    // A static executable entered at `_start`.
    let exe = build(&dir, &[], "a.out").unwrap();
    assert_eq!((u16_at(&exe, 16), u16_at(&exe, 18)), (2, 183), "ET_EXEC, EM_AARCH64");
    if let Some(s) = readobj(&["--symbols", "-l"], &dir.join("a.out")) {
        assert!(s.contains("_start") && s.contains("main") && s.contains("PT_LOAD"), "{s}");
    }
    // A shared library: no text relocation.
    let so = build(&dir, &["--shared", "-soname", "liba64.so"], "liba64.so").unwrap();
    assert_eq!((u16_at(&so, 16), u16_at(&so, 18)), (3, 183), "ET_DYN, EM_AARCH64");
    if let Some(s) = readobj(&["--dynamic-table", "--dyn-relocations"], &dir.join("liba64.so")) {
        assert!(s.contains("liba64.so") && !s.contains("TEXTREL"), "{s}");
        assert!(s.contains("R_AARCH64_GLOB_DAT k"), "{s}");
    }
    // `-g`: DWARF sections, carried into the linked executable.
    build(&dir, &["-g"], "g.out").unwrap();
    if let Some(s) = readobj(&["-S"], &dir.join("g.out")) {
        assert!(s.contains(".debug_line") && s.contains(".debug_info"), "{s}");
    }
    // PIE needs a C runtime: x86-64 only.
    let err = build(&dir, &["--pie"], "pie").unwrap_err();
    assert!(err.contains("--pie"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
