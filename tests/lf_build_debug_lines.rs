//! Source lines survive the optimizer (GitHub issue #19): an `-O2 -g` x86-64
//! executable has DWARF line rows for its optimized code — including the
//! lines of an inlined callee — and gdb can break on a source line and stop
//! there. A missing tool (or host C runtime) skips its check.

use std::path::PathBuf;
use std::process::Command;

/// One instruction per line. At `-O2`, `@step` is inlined into the loop and
/// deleted; the loop bound (4..=7) comes from `getpid`, so nothing folds away
/// and the loop body always runs.
const SRC: &str = r#"module "dbg"
func @getpid() -> i32
func internal @step(i64, i64) -> i64 {
entry ^0(%acc: i64, %i: i64):
  %m = mul %i, i64 3 : i64
  %s = add %acc, %m : i64
  ret %s
}
func @main() -> i32 {
entry ^0:
  %p = call @getpid() : i32
  %n = zext %p : i64
  %k0 = and %n, i64 7 : i64
  %k = or %k0, i64 4 : i64
  br ^1(i64 0, i64 0)
^1(%i: i64, %acc: i64):
  %done = icmp uge %i, %k : i1
  cond_br %done, ^3, ^2
^2:
  %acc2 = call @step(%acc, %i) : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %acc2)
^3:
  %t = trunc %acc : i32
  %r = and %t, i32 1 : i32
  ret %r
}
"#;

/// The 1-based line of `SRC` containing `needle`.
fn line_of(needle: &str) -> u32 {
    SRC.lines().position(|l| l.contains(needle)).expect("line in SRC") as u32 + 1
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-lines-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
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

/// Build `SRC` as an `-O2 -g` PIE in `dir`, or `None` without a host C runtime.
fn build_o2(dir: &std::path::Path) -> Option<PathBuf> {
    if latticefoundry::link::gnu::HostCrt::discover().is_none() {
        eprintln!("skipping: the host C runtime is missing");
        return None;
    }
    let src = dir.join("dbg.lf");
    std::fs::write(&src, SRC).unwrap();
    let exe = dir.join("dbg");
    let st = Command::new(env!("CARGO_BIN_EXE_lf"))
        .args(["build", "--pie", "-O2", "-g", "-o"])
        .args([&exe, &src])
        .status()
        .unwrap();
    assert!(st.success());
    Some(exe)
}

#[test]
fn o2_line_table_has_rows_for_optimized_code() {
    let Some(dwarfdump) = llvm_tool("llvm-dwarfdump") else {
        eprintln!("skipping: llvm-dwarfdump is not installed");
        return;
    };
    let dir = scratch("dwarf");
    let Some(exe) = build_o2(&dir) else { return };
    let out = Command::new(dwarfdump).arg("--debug-line").arg(&exe).output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    // The rows' line column (`0x<address> <line> <column> ...`).
    let lines: Vec<u32> = text
        .lines()
        .filter(|l| l.starts_with("0x"))
        .filter_map(|l| l.split_whitespace().nth(1)?.parse().ok())
        .collect();
    for needle in ["%p = call @getpid", "cond_br %done", "%i2 = add", "%m = mul", "%s = add %acc", "ret %r"] {
        let want = line_of(needle);
        assert!(lines.contains(&want), "no row for line {want} (`{needle}`) at -O2:\n{text}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gdb_breaks_on_a_source_line_at_o2() {
    if !Command::new("gdb").arg("--version").output().is_ok_and(|o| o.status.success()) {
        eprintln!("skipping: gdb is not installed");
        return;
    }
    let dir = scratch("gdb");
    let Some(exe) = build_o2(&dir) else { return };
    // A line of the inlined callee and one of the caller's loop.
    for needle in ["%m = mul", "%i2 = add"] {
        let line = line_of(needle);
        let out = Command::new("gdb")
            .args(["-batch", "-nx", "-ex", &format!("break dbg.lf:{line}"), "-ex", "run", "-ex", "bt 1"])
            .arg(&exe)
            .output()
            .unwrap();
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(
            s.contains("Breakpoint 1, main () at ") && s.contains(&format!("dbg.lf:{line}\n")),
            "gdb did not stop at dbg.lf:{line}:\n{s}\n{}",
            String::from_utf8_lossy(&out.stderr),
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
