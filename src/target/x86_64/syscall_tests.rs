//! Execution tests for the native `syscall` IR instruction on x86-64.
//!
//! Each program is written as `.lf` text, optionally run through the `-O`
//! pipeline, compiled with our own x86-64 backend, linked by our own static
//! linker (no gcc/ld/libc — `lf build`'s freestanding path), executed on the
//! bare Linux kernel, and checked on its stdout and exit status. The syscalls go
//! straight to the kernel: `write`(1), `exit`(60), `getpid`(39), `mmap`(9),
//! `munmap`(11).

use crate::ir::Module;
use crate::link::{ImageOptions, link_executable, write_executable};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize};

/// Parse `src`, verify, optimize at `level`, verify again, and return it.
fn prepare(src: &str, level: OptLevel) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse .lf: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    optimize(&mut m, level);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify after {level:?}: {e:?}"));
    (m, syms)
}

/// Compile + link `m` to a static executable, run it, and return
/// `(stdout, exit code, child pid)`.
fn build_and_run(m: &Module, syms: &StrInterner, tag: &str) -> (Vec<u8>, i32, u32) {
    let obj = super::compile_module(m, syms);
    let image = link_executable(vec![obj], &ImageOptions::default()).expect("link should succeed");

    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("lf_sys_{tag}_{}_{uniq}", std::process::id()));
    let path_str = path.to_str().unwrap().to_owned();
    write_executable(&path_str, &image).expect("write executable");

    // Retry a transient ETXTBSY (errno 26): another test thread's fork may briefly
    // hold a writable fd to the file just written.
    let child = loop {
        match std::process::Command::new(&path)
            .stdout(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => break c,
            Err(e) if e.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(e) => panic!("exec our native binary: {e}"),
        }
    };
    let pid = child.id();
    let out = child.wait_with_output().expect("wait for child");
    let _ = std::fs::remove_file(&path);
    let code = out.status.code().expect("child exited via signal, not code");
    (out.stdout, code, pid)
}

fn run_lf(src: &str, level: OptLevel, tag: &str) -> (Vec<u8>, i32, u32) {
    let (m, syms) = prepare(src, level);
    build_and_run(&m, &syms, tag)
}

/// `write(1, buf, 14)` from a stack buffer (filled with two `i64` stores of
/// "Hello, kernel\n"), checks the byte count, then `exit(42)` via the `exit`
/// syscall (main never returns).
const HELLO: &str = r#"
module "hello"
func @main() -> i64 {
entry ^0:
  %buf = alloca [16 x i8] : ptr
  store i64 7719218618385065288, %buf align 8 : i64
  %hi = ptr_add %buf, i64 8 : ptr
  store i64 11460674482789, %hi align 8 : i64
  %n = syscall i64 1, i64 1, %buf, i64 14 : i64
  %ok = icmp eq %n, i64 14 : i1
  cond_br %ok, ^1, ^2
^1:
  %a = syscall i64 60, i64 42 : i64
  unreachable
^2:
  %b = syscall i64 60, i64 1 : i64
  unreachable
}
"#;

#[test]
fn write_and_exit_via_syscalls() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let (out, code, _) = run_lf(HELLO, level, "hello");
        assert_eq!(out, b"Hello, kernel\n", "stdout at {level:?}");
        assert_eq!(code, 42, "exit status at {level:?}");
    }
}

/// `getpid()`, stored into a stack slot and written out as 8 raw bytes; the
/// harness compares them with the child's pid. The slot's address escapes into a
/// syscall, so `mem2reg` must leave it in memory even at `-O2`.
const GETPID: &str = r#"
module "pid"
func @main() -> i64 {
entry ^0:
  %slot = alloca i64 : ptr
  %pid = syscall i64 39 : i64
  store %pid, %slot align 8 : i64
  %n = syscall i64 1, i64 1, %slot, i64 8 : i64
  %rc = sub %n, i64 8 : i64
  ret %rc
}
"#;

#[test]
fn getpid_matches_the_process() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let (out, code, pid) = run_lf(GETPID, level, "pid");
        assert_eq!(code, 0, "write must report 8 bytes at {level:?}");
        assert_eq!(out.len(), 8, "8 raw pid bytes at {level:?}");
        let got = u64::from_le_bytes(out[..8].try_into().unwrap());
        assert_eq!(got, u64::from(pid), "getpid() == child pid at {level:?}");
    }
}

/// A syscall in a loop: five one-byte writes of the same (loop-invariant)
/// buffer, summing their results; exits with that sum. LICM must not hoist the
/// invariant syscall and DCE/SCCP must keep all five executions.
const LOOP: &str = r#"
module "loop"
func @main() -> i64 {
entry ^0:
  %dot = alloca i8 : ptr
  store i8 46, %dot align 1 : i8
  br ^1(i64 0, i64 0)
^1(%i: i64, %acc: i64):
  %c = icmp slt %i, i64 5 : i1
  cond_br %c, ^2, ^3
^2:
  %r = syscall i64 1, i64 1, %dot, i64 1 : i64
  %acc2 = add %acc, %r : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %acc2)
^3:
  ret %acc
}
"#;

#[test]
fn syscall_in_a_loop() {
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let (out, code, _) = run_lf(LOOP, level, "loop");
        assert_eq!(out, b".....", "five writes at {level:?}");
        assert_eq!(code, 5, "summed write results at {level:?}");
    }
}

/// Register pressure around a 6-argument syscall: `mmap(NULL, 4096,
/// PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)` with every argument
/// a *computed* value (derived from `getpid`, so nothing folds) that is *also*
/// used after the call, while ten more computed values stay live across both the
/// `mmap` and the `munmap` — more live values than registers, so operands of the
/// argument-move run are spilled and reloaded through the scratch registers
/// (one of which, `r10`, is itself the 4th syscall argument register). The mapping
/// is written, echoed with `write`, then unmapped. The exit status folds the
/// live-across values, the re-used arguments and the `munmap` result together:
/// 45 iff everything survived and `munmap` returned 0.
const MMAP: &str = r#"
module "mmap"
func @main() -> i64 {
entry ^0:
  %pid = syscall i64 39 : i64
  %z = sub %pid, %pid : i64
  %v0 = add %pid, i64 0 : i64
  %v1 = add %pid, i64 1 : i64
  %v2 = add %pid, i64 2 : i64
  %v3 = add %pid, i64 3 : i64
  %v4 = add %pid, i64 4 : i64
  %v5 = add %pid, i64 5 : i64
  %v6 = add %pid, i64 6 : i64
  %v7 = add %pid, i64 7 : i64
  %v8 = add %pid, i64 8 : i64
  %v9 = add %pid, i64 9 : i64
  %addr = add %z, i64 0 : i64
  %len = add %z, i64 4096 : i64
  %prot = add %z, i64 3 : i64
  %flags = add %z, i64 34 : i64
  %fd = sub %z, i64 1 : i64
  %off = add %z, i64 0 : i64
  %nr = add %z, i64 9 : i64
  %p = syscall %nr, %addr, %len, %prot, %flags, %fd, %off : i64
  %bad = icmp ugt %p, i64 -4096 : i1
  cond_br %bad, ^2, ^1
^1:
  %ptr = inttoptr %p : ptr
  store i64 750815948002389357, %ptr align 8 : i64
  %w = syscall i64 1, i64 1, %ptr, i64 8 : i64
  %u = syscall i64 11, %ptr, %len : i64
  %s1 = add %v0, %v1 : i64
  %s2 = add %s1, %v2 : i64
  %s3 = add %s2, %v3 : i64
  %s4 = add %s3, %v4 : i64
  %s5 = add %s4, %v5 : i64
  %s6 = add %s5, %v6 : i64
  %s7 = add %s6, %v7 : i64
  %s8 = add %s7, %v8 : i64
  %s9 = add %s8, %v9 : i64
  %ten = mul %pid, i64 10 : i64
  %d = sub %s9, %ten : i64
  %w8 = sub %w, i64 8 : i64
  %e1 = add %d, %u : i64
  %e2 = add %e1, %w8 : i64
  %t1 = add %addr, %prot : i64
  %t2 = add %t1, %flags : i64
  %t3 = add %t2, %fd : i64
  %t4 = add %t3, %off : i64
  %t5 = add %t4, %nr : i64
  %t6 = sub %t5, i64 45 : i64
  %e3 = add %e2, %t6 : i64
  ret %e3
^2:
  ret i64 99
}
"#;

#[test]
fn six_argument_mmap_under_register_pressure() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let (out, code, _) = run_lf(MMAP, level, "mmap");
        // The stored constant is "mmap ok\n" as a little-endian i64.
        assert_eq!(out, b"mmap ok\n", "mapping echoed at {level:?}");
        assert_eq!(code, 45, "live-across values + munmap==0 at {level:?}");
    }
}

/// `-O2` must keep effect order and effects with unused results. One stack byte
/// is rewritten between writes ("A", "B", "C"), so the syscalls are ordered with
/// respect to the stores as well as each other (a syscall reads memory through
/// its escaped pointer: store-to-load forwarding or dead-store elimination across
/// it would be wrong). The first and third writes' results are unused, and a
/// helper that only performs a syscall is inlined without losing it.
const ORDER: &str = r#"
module "order"
func @emit_c() -> i64 {
entry ^0:
  %c = alloca i8 : ptr
  store i8 67, %c align 1 : i8
  %r = syscall i64 1, i64 1, %c, i64 1 : i64
  ret i64 0
}
func @main() -> i64 {
entry ^0:
  %buf = alloca i8 : ptr
  store i8 65, %buf align 1 : i8
  %x = syscall i64 1, i64 1, %buf, i64 1 : i64
  store i8 66, %buf align 1 : i8
  %y = syscall i64 1, i64 1, %buf, i64 1 : i64
  store i8 67, %buf align 1 : i8
  %z = syscall i64 1, i64 1, %buf, i64 1 : i64
  %h = call @emit_c() : i64
  %r = add %y, i64 6 : i64
  ret %r
}
"#;

#[test]
fn optimizer_keeps_syscall_order_and_unused_results() {
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let (out, code, _) = run_lf(ORDER, level, "order");
        assert_eq!(out, b"ABCC", "all writes kept, in order, at {level:?}");
        assert_eq!(code, 7, "the used write result at {level:?}");
    }
}

/// The `-O` pipelines never drop a syscall: every level keeps at least the
/// four the program executes (inlining may copy the helper's into `main`).
#[test]
fn optimizer_preserves_syscall_count() {
    use crate::ir::InstKind;
    let count = |m: &Module| -> usize {
        m.functions()
            .map(|func| {
                func.blocks()
                    .flat_map(|(_, b)| b.insts().iter().copied())
                    .filter(|&i| matches!(func.inst(i).kind, InstKind::Syscall))
                    .count()
            })
            .sum()
    };
    let (m0, _) = prepare(ORDER, OptLevel::O0);
    assert_eq!(count(&m0), 4);
    for level in [OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let (m, _) = prepare(ORDER, level);
        assert!(count(&m) >= 4, "no syscall may be dropped at {level:?}");
    }
}

/// The x86-64 lowering follows the Linux convention — number in `rax`,
/// arguments in `rdi, rsi, rdx, r10, r8, r9`, result in `rax`, `rcx`/`r11`
/// clobbered — with every fixed-register move in one run right before the
/// instruction and `r10` (a spill/reload scratch register) written last.
#[test]
fn syscall_mir_register_convention() {
    use super::isel::{X86_64Target, X86Op};
    use super::regs::gpr;
    use crate::codegen::mir::{MachineOperand, Reg};
    let src = "module \"m\"\nfunc @f(i64) -> i64 {\nentry ^0(%x: i64):\n  \
               %r = syscall i64 9, %x, %x, %x, %x, %x, %x : i64\n  ret %r\n}\n";
    let (m, syms) = prepare(src, OptLevel::O0);
    let mf = X86_64Target::new().select_with_syms(&m, crate::ir::FuncId::from_index(0), &syms);
    let insts: Vec<_> = mf.block_ids().flat_map(|b| mf.block(b).insts.clone()).collect();
    let at = insts
        .iter()
        .position(|i| X86Op::decode(i.opcode) == X86Op::Syscall)
        .expect("a Syscall is emitted");
    let sys = &insts[at];
    // rax rdi rsi rdx r8 r9, then r10 last.
    let order: Vec<Reg> = [0u16, 7, 6, 2, 8, 9, 10].iter().map(|&n| Reg::Physical(gpr(n))).collect();
    assert_eq!(sys.uses().collect::<Vec<_>>(), order);
    let defs: Vec<Reg> = [0u16, 1, 11].iter().map(|&n| Reg::Physical(gpr(n))).collect();
    assert_eq!(sys.defs().collect::<Vec<_>>(), defs, "rax result, rcx/r11 clobbered");
    for (k, want) in order.iter().enumerate() {
        let mv = &insts[at - 7 + k];
        assert_eq!(X86Op::decode(mv.opcode), X86Op::MovRR);
        assert_eq!(mv.operands[0], MachineOperand::Def(*want));
    }
}

#[test]
fn syscall_encoding_matches_llvm_mc() {
    let src = "module \"m\"\nfunc @f() -> i64 {\nentry ^0:\n  %r = syscall i64 39 : i64\n  ret %r\n}\n";
    let (m, syms) = prepare(src, OptLevel::O0);
    let code = super::compile_function(&m, crate::ir::FuncId::from_index(0), &syms);
    assert!(code.bytes.windows(2).any(|w| w == [0x0F, 0x05]), "`syscall` (0F 05) is emitted");

    use std::io::Write;
    let Ok(mut child) = std::process::Command::new("llvm-mc")
        .args(["--triple=x86_64", "--show-encoding"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        eprintln!("skipping llvm-mc cross-check of syscall: no llvm-mc");
        return;
    };
    child.stdin.as_mut().unwrap().write_all(b"syscall\n").unwrap();
    let out = child.wait_with_output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("encoding: [0x0f,0x05]"), "llvm-mc says: {text}");
}
