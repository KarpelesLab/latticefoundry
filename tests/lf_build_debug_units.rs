//! Debug info of a program built from several modules: separate `-g`
//! objects linked by `lf-ld` (qld) keep one compile unit each — names,
//! strings and line tables — because every DWARF section offset in an object
//! carries a relocation against its target section; a multi-module LTO
//! `-g` build stays valid. A missing tool skips its check.

use std::path::{Path, PathBuf};
use std::process::Command;

const MOD_A: &str = r#"module "a"
func @helper(i64) -> i64
func @main() -> i64 {
entry ^0:
  %h = call @helper(i64 40) : i64
  ret %h
}
"#;

/// `helper`'s `add` is on line 5.
const MOD_B: &str = r#"module "b"

func @helper(i64) -> i64 {
entry ^0(%x: i64):
  %a = add %x, i64 2 : i64
  ret %a
}
"#;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-units-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("ma.lf"), MOD_A).unwrap();
    std::fs::write(dir.join("mb.lf"), MOD_B).unwrap();
    dir
}

/// Find an LLVM tool on `PATH` or in `/usr/lib/llvm/*/bin`.
fn llvm_tool(name: &str) -> Option<PathBuf> {
    if Command::new(name).arg("--version").output().is_ok_and(|o| o.status.success()) {
        return Some(PathBuf::from(name));
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir("/usr/lib/llvm")
        .ok()?
        .flatten()
        .map(|e| e.path().join("bin").join(name))
        .filter(|p| p.is_file())
        .collect();
    found.sort();
    found.pop()
}

/// Run `bin` with `args` in `dir`, panicking with its stderr on failure.
fn run_ok(bin: &str, args: &[&str], dir: &Path) {
    let out = Command::new(bin).args(args).current_dir(dir).output().unwrap();
    assert!(out.status.success(), "{bin} {args:?}:\n{}", String::from_utf8_lossy(&out.stderr));
}

/// The `DW_AT_name`s of the compile units of `exe`, and whether
/// `llvm-dwarfdump --verify` accepts it; `None` without llvm-dwarfdump.
fn compile_units(exe: &Path) -> Option<(Vec<String>, bool)> {
    let Some(dwarfdump) = llvm_tool("llvm-dwarfdump") else {
        eprintln!("skipping the DWARF check: llvm-dwarfdump is not installed");
        return None;
    };
    let out = Command::new(&dwarfdump).arg("--debug-info").arg(exe).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let mut names = Vec::new();
    let mut in_cu = false;
    for l in text.lines().map(str::trim) {
        if l.contains("DW_TAG_") {
            in_cu = l.contains("DW_TAG_compile_unit");
        } else if in_cu && l.starts_with("DW_AT_name") {
            names.push(l.split('"').nth(1).unwrap_or_default().to_owned());
            in_cu = false;
        }
    }
    let verified = Command::new(&dwarfdump).arg("--verify").arg(exe).output().unwrap().status.success();
    Some((names, verified))
}

/// `-c -g` objects of two modules linked by `lf-ld` (qld): one compile unit
/// per module, and gdb resolves a line of the second module.
#[test]
fn separate_objects_keep_their_compile_units() {
    let dir = scratch("sep");
    let lf = env!("CARGO_BIN_EXE_lf");
    for (target, tag) in [("x86_64-linux", "x86"), ("aarch64-linux", "a64")] {
        for m in ["ma", "mb"] {
            run_ok(lf, &["build", "--target", target, "-g", "-c", &format!("{m}.lf"), "-o", &format!("{m}-{tag}.o")], &dir);
        }
        let exe = format!("sep-{tag}");
        let mut args = vec!["-e", "main", "-o", &exe];
        if tag == "a64" {
            args.extend(["-m", "aarch64linux"]);
        }
        let (a, b) = (format!("ma-{tag}.o"), format!("mb-{tag}.o"));
        args.extend([a.as_str(), b.as_str()]);
        run_ok(env!("CARGO_BIN_EXE_lf-ld"), &args, &dir);
        if let Some((names, verified)) = compile_units(&dir.join(&exe)) {
            assert_eq!(names, ["ma.lf", "mb.lf"], "{target}");
            assert!(verified, "{target}: llvm-dwarfdump --verify failed");
        }
    }
    if Command::new("gdb").arg("--version").output().is_ok_and(|o| o.status.success()) {
        let out = Command::new("gdb")
            .args(["-batch", "-nx", "-ex", "break mb.lf:5"])
            .arg(dir.join("sep-x86"))
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("file mb.lf, line 5."), "gdb did not resolve mb.lf:5:\n{text}");
    } else {
        eprintln!("skipping the gdb check: gdb is not installed");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two modules IR-linked (LTO) into one `-g` executable: it runs, and its
/// debug info is well-formed.
#[test]
fn lto_debug_build_is_valid() {
    let dir = scratch("lto");
    run_ok(env!("CARGO_BIN_EXE_lf"), &["build", "-g", "-O2", "ma.lf", "mb.lf", "-o", "lto"], &dir);
    let exe = dir.join("lto");
    let status = loop {
        match Command::new(&exe).status() {
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(20)),
            other => break other.unwrap(),
        }
    };
    assert_eq!(status.code(), Some(42));
    if let Some((names, verified)) = compile_units(&exe) {
        assert_eq!(names, ["ma.lf"]);
        assert!(verified, "llvm-dwarfdump --verify failed");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
