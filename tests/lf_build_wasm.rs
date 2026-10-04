//! `lf build --target wasm32` through the real driver binary: a self-contained
//! `.wasm` module run under node, a relocatable object linked by `wasm-ld`,
//! and the options that do not apply to wasm32. The node and `wasm-ld` steps
//! are skipped when those tools are not installed.

use std::path::{Path, PathBuf};
use std::process::Command;

/// No `datalayout` line: the driver gives the module the wasm32 layout.
const SRC: &str = r#"
module "wasmdemo"
global @k : i64 = i64 2
global @squares : [4 x i32] = [4 x i32] (i32 0, i32 1, i32 4, i32 9)
func @helper(i64) -> i64 {
entry ^0(%a: i64):
  %b = load @k align 8 : i64
  %s = add %a, %b : i64
  ret %s
}
func @main() -> i64 {
entry ^0:
  %p = ptr_add @squares, i32 12 : ptr
  %v = load %p align 4 : i32
  %w = zext %v : i64
  %r = call @helper(%w) : i64
  %f = mul %r, i64 3 : i64
  ret %f
}
"#;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-build-wasm-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("demo.lf"), SRC).unwrap();
    dir
}

/// Run `lf build <dir>/demo.lf args..`, returning (success, stdout, stderr).
fn lf(dir: &Path, args: &[&str]) -> (bool, String, String) {
    // Run in the scratch directory: outputs without `-o` land there.
    let out = Command::new(env!("CARGO_BIN_EXE_lf"))
        .current_dir(dir)
        .arg("build")
        .arg(dir.join("demo.lf"))
        .args(args)
        .output()
        .expect("run lf");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// Instantiate `wasm` under node and print `main()`'s result.
fn run_main(dir: &Path, wasm: &Path) -> String {
    let js = dir.join("run.js");
    std::fs::write(
        &js,
        "const b = require('fs').readFileSync(process.argv[2]);\n\
         WebAssembly.instantiate(b, {}).then(({instance}) => console.log(String(instance.exports.main())));\n",
    )
    .unwrap();
    let out = Command::new("node").arg(&js).arg(wasm).output().expect("node");
    assert!(out.status.success(), "node: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

#[test]
fn builds_a_runnable_wasm_module() {
    let dir = scratch("module");
    let out = dir.join("demo.wasm");
    let (ok, stdout, err) = lf(&dir, &["--target", "wasm32", "--stack-usage", "-o", out.to_str().unwrap()]);
    assert!(ok, "lf build: {err}");
    assert!(stdout.contains("main"), "stack usage report: {stdout}");
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(&bytes[..8], b"\0asm\x01\0\0\0");
    if have("node") {
        // (9 + 2) * 3
        assert_eq!(run_main(&dir, &out), "33");
    } else {
        eprintln!("skipping the node run: node is not installed");
    }
    // Without -o, the output is named after the input.
    let (ok, _, err) = lf(&dir, &["--target", "wasm32-unknown-unknown"]);
    assert!(ok, "{err}");
    assert!(dir.join("demo.wasm").is_file(), "default output name");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn builds_a_relocatable_object_for_wasm_ld() {
    let dir = scratch("object");
    let obj = dir.join("demo.o");
    let (ok, _, err) = lf(&dir, &["--target", "wasm32", "-c", "-o", obj.to_str().unwrap()]);
    assert!(ok, "lf build -c: {err}");
    let bytes = std::fs::read(&obj).unwrap();
    assert_eq!(&bytes[..4], b"\0asm");
    assert!(bytes.windows(7).any(|w| w == b"linking"), "a `linking` section");
    let (ok, _, err) = lf(&dir, &["--target", "wasm32", "-c", "--format", "wasm", "-o", obj.to_str().unwrap()]);
    assert!(ok, "{err}");
    if !have("wasm-ld") || !have("node") {
        eprintln!("skipping the wasm-ld link: wasm-ld or node is not installed");
        return;
    }
    let linked = dir.join("linked.wasm");
    let out = Command::new("wasm-ld")
        .args(["--no-entry", "--stack-first", "-o"])
        .arg(&linked)
        .arg(&obj)
        .output()
        .unwrap();
    assert!(out.status.success(), "wasm-ld: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(run_main(&dir, &linked), "33");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn options_that_do_not_apply_to_wasm32() {
    let dir = scratch("errors");
    for (args, want) in [
        (&["--target", "wasm32", "--oformat", "binary"][..], "do not apply to wasm32"),
        (&["--target", "wasm32", "-g"][..], "aarch64 only"),
        (&["--target", "wasm32", "--shared"][..], "x86-64 ELF"),
        (&["--target", "wasm32", "-c", "--format", "elf"][..], "wasm format"),
        (&["--target", "wasm32-linux"][..], "unknown target"),
    ] {
        let (ok, _, err) = lf(&dir, args);
        assert!(!ok, "{args:?} should fail");
        assert!(err.contains(want), "{args:?}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
