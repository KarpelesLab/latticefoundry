//! Execution tests for **volatile accesses, atomics and fences** on x86-64
//! (`docs/ir-design.md` §6b).
//!
//! Each program is `.lf` text run through the optimization pipeline at several
//! levels, compiled by our x86-64 backend, linked by our own static linker, and
//! run on the bare kernel. The expected results come from an independent
//! oracle: the standard library's own atomics (`fetch_add`, `fetch_nand`,
//! `fetch_max`, `compare_exchange`, ...) run on the same inputs. Machine-code
//! shape checks (the MIR after isel) pin the TSO lowering: `mov` for loads and
//! non-`seq_cst` stores, `xchg` for `seq_cst` stores, `lock xadd`, the `lock
//! cmpxchg` loop, `mfence` only for `fence seq_cst`, and one `mov` per volatile
//! access at exactly its width. A real two-thread test (a raw `clone` with a
//! stack in `.bss`) checks that the `lock`ed increments are atomic.

use std::path::PathBuf;

use crate::codegen::mir::{MachineFunction, MachineOperand};
use crate::ir::{FuncId, Module};
use crate::link::{ImageOptions, link_executable, write_executable};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize};

use super::isel::{X86Op, X86_64Target};
use crate::target::atomic_fixtures::{RmwCase, rmw_cases, ty_name};

const LEVELS: [OptLevel; 4] = [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3];

/// Parse `src`, verify, optimize at `level`, verify again, and return it.
fn prepare(src: &str, level: OptLevel) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse .lf: {e:?}\n{src}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    optimize(&mut m, level);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify after {level:?}: {e:?}"));
    (m, syms)
}

/// A unique temp path for one test artifact.
fn temp_path(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("lf_atomic_{tag}_{}_{uniq}", std::process::id()))
}

/// Compile + link `src` at `level`, run it, and return its exit code.
fn exit_code(src: &str, level: OptLevel, tag: &str) -> i32 {
    let (m, syms) = prepare(src, level);
    let obj = super::compile_module(&m, &syms);
    let image = link_executable(vec![obj], &ImageOptions::default()).expect("link should succeed");
    let path = temp_path(tag);
    write_executable(path.to_str().unwrap(), &image).expect("write executable");
    // Retry a transient ETXTBSY (errno 26): another test thread's fork may
    // briefly hold a writable fd to the file just written.
    let status = loop {
        match std::process::Command::new(&path).status() {
            Ok(s) => break s,
            Err(e) if e.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(e) => panic!("exec our native binary: {e}"),
        }
    };
    let _ = std::fs::remove_file(&path);
    status.code().unwrap_or_else(|| panic!("{tag} at {level:?} died: {status:?}"))
}

/// The instruction-selected MIR of function `name` (before allocation).
fn mir_of(m: &Module, syms: &StrInterner, name: &str) -> MachineFunction {
    let idx = m.functions().position(|f| syms.resolve(f.name) == name).expect("function exists");
    X86_64Target::new().select_with_syms(m, FuncId::from_index(idx), syms)
}

/// Every MIR instruction of `mf` with its decoded opcode, in block order.
fn ops_of(mf: &MachineFunction) -> Vec<(X86Op, Vec<MachineOperand>)> {
    mf.block_ids()
        .flat_map(|b| mf.block(b).insts.iter().map(|i| (X86Op::decode(i.opcode), i.operands.clone())))
        .collect()
}

fn imm_at(ops: &[MachineOperand], i: usize) -> u64 {
    match &ops[i] {
        MachineOperand::Imm(v) => v.to_u64().expect("small immediate"),
        other => panic!("expected an immediate, found {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Counters, fetch-ops and compare-exchange on globals
// ---------------------------------------------------------------------------

/// 1000 iterations of `+3` (seq_cst) and `-1` (relaxed) on a `.data` counter;
/// unused results. Exit code 0 iff the counter ends at 2000 + 7.
const COUNTER: &str = r#"
module "counter"
global @counter : i64 = i64 7

func @main() -> i64 {
entry ^0:
  br ^1(i64 0)
^1(%i: i64):
  %c = icmp slt %i, i64 1000 : i1
  cond_br %c, ^2, ^3
^2:
  %o = atomic_rmw add seq_cst @counter, i64 3 align 8 : i64
  %o2 = atomic_rmw sub relaxed @counter, i64 1 align 8 : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2)
^3:
  %v = atomic_load seq_cst @counter align 8 : i64
  %r = sub %v, i64 2007 : i64
  ret %r
}
"#;

#[test]
fn atomic_counter_on_a_global() {
    for level in LEVELS {
        assert_eq!(exit_code(COUNTER, level, "counter"), 0, "at {level:?}");
    }
}

/// Build a program that runs every `(width, op, init, v)` case on its own
/// global and checks the returned old value and the final memory against the
/// expected pair, returning the (1-based) index of the last failing check, or
/// 0. Optionally keeps `pressure` volatile loads live across every atomic so
/// the allocator must spill around the fixed-register sequences.
fn rmw_program(cases: &[RmwCase], pressure: usize) -> String {
    let mut s = String::from("module \"rmw\"\n");
    for (k, &(bytes, _, init, ..)) in cases.iter().enumerate() {
        s += &format!("global @g{k} : {t} = {t} {init}\n", t = ty_name(bytes));
    }
    s += "global @dev : i64 = i64 1\n";
    s += "func @main() -> i64 {\nentry ^0:\n";
    for j in 0..pressure {
        s += &format!("  %live{j} = load volatile @dev align 8 : i64\n");
    }
    s += "  %f0 = add i64 0, i64 0 : i64\n";
    for (k, &(bytes, op, _, v, old, new)) in cases.iter().enumerate() {
        let t = ty_name(bytes);
        let a = bytes;
        s += &format!("  %old{k} = atomic_rmw {op} seq_cst @g{k}, {t} {v} align {a} : {t}\n");
        s += &format!("  %new{k} = load @g{k} align {a} : {t}\n");
        s += &format!("  %eo{k} = icmp ne %old{k}, {t} {old} : i1\n");
        s += &format!("  %en{k} = icmp ne %new{k}, {t} {new} : i1\n");
        s += &format!("  %bad{k} = or %eo{k}, %en{k} : i1\n");
        s += &format!("  %f{} = select %bad{k}, i64 {}, %f{k} : i64\n", k + 1, k + 1);
    }
    let n = cases.len();
    if pressure == 0 {
        s += &format!("  ret %f{n}\n}}\n");
    } else {
        // Every pressure value is 1; they must all have survived the atomics.
        s += "  %sum0 = add i64 0, i64 0 : i64\n";
        for j in 0..pressure {
            s += &format!("  %sum{} = add %sum{j}, %live{j} : i64\n", j + 1);
        }
        s += &format!("  %okp = icmp eq %sum{pressure}, i64 {pressure} : i1\n");
        s += &format!("  %r = select %okp, %f{n}, i64 255 : i64\n  ret %r\n}}\n");
    }
    s
}

#[test]
fn every_rmw_op_returns_the_old_value_at_every_width() {
    for bytes in [1, 2, 4, 8] {
        let cases = rmw_cases(bytes);
        let src = rmw_program(&cases, 0);
        for level in [OptLevel::O0, OptLevel::O2] {
            let code = exit_code(&src, level, "rmw");
            assert_eq!(code, 0, "i{} at {level:?}: case {:?} failed", 8 * bytes, cases.get((code as usize).wrapping_sub(1)));
        }
    }
}

#[test]
fn shared_slot_fixtures_run_natively() {
    // The same stack-slot programs the AArch64/RISC-V interpreters run.
    use crate::target::atomic_fixtures::{CMPXCHG_SLOTS, rmw_slot_program};
    for level in [OptLevel::O0, OptLevel::O2] {
        for bytes in [1, 2, 4, 8] {
            let cases = rmw_cases(bytes);
            let code = exit_code(&rmw_slot_program(&cases), level, "rmwslots");
            assert_eq!(code, 0, "i{} at {level:?}: case {:?}", 8 * bytes, cases.get((code as usize).wrapping_sub(1)));
        }
        assert_eq!(exit_code(CMPXCHG_SLOTS, level, "casslots"), 0, "at {level:?}");
    }
}

#[test]
fn rmw_under_register_pressure_spills_around_fixed_registers() {
    // Fourteen live values across every atomic: more than the eleven
    // allocatable GPRs, so pointers/operands are spilled and reloaded through
    // the scratch registers around `lock cmpxchg` (fixed rax) and friends.
    let mut cases = rmw_cases(4);
    cases.truncate(11);
    let src = rmw_program(&cases, 14);
    for level in [OptLevel::O0, OptLevel::O2] {
        let code = exit_code(&src, level, "rmwpressure");
        assert_eq!(code, 0, "at {level:?}: {code}");
    }
}

/// `cmpxchg` success and failure on each width, plus the `icmp eq` success
/// flag. Returns a bitmask of failed checks.
fn cmpxchg_program(t: &str, align: u32, init: &str, other: &str, new: &str) -> String {
    format!(
        r#"
module "cas"
global @x : {t} = {init}

func @main() -> i64 {{
entry ^0:
  ; success: x == init, so x := new; returns init
  %o1 = cmpxchg seq_cst seq_cst @x, {init}, {new} align {align} : {t}
  %s1 = icmp eq %o1, {init} : i1
  %v1 = load @x align {align} : {t}
  ; failure: x (== new) != other, so x is unchanged; returns new
  %o2 = cmpxchg acquire relaxed @x, {other}, {init} align {align} : {t}
  %s2 = icmp eq %o2, {other} : i1
  %v2 = load @x align {align} : {t}
  %c1 = icmp eq %o1, {init} : i1
  %c2 = icmp eq %v1, {new} : i1
  %c3 = icmp eq %o2, {new} : i1
  %c4 = icmp eq %v2, {new} : i1
  %m1 = select %s1, i64 0, i64 1 : i64
  %m2 = select %s2, i64 2, i64 0 : i64
  %m3 = select %c1, i64 0, i64 4 : i64
  %m4 = select %c2, i64 0, i64 8 : i64
  %m5 = select %c3, i64 0, i64 16 : i64
  %m6 = select %c4, i64 0, i64 32 : i64
  %a1 = or %m1, %m2 : i64
  %a2 = or %a1, %m3 : i64
  %a3 = or %a2, %m4 : i64
  %a4 = or %a3, %m5 : i64
  %a5 = or %a4, %m6 : i64
  ret %a5
}}
"#
    )
}

#[test]
fn cmpxchg_success_and_failure_paths() {
    // (type, align, init, other, new) as full operands: `other` differs from
    // `new`, so the second exchange fails.
    for (t, align, init, other, new) in [
        ("i8", 1, "i8 -3", "i8 4", "i8 100"),
        ("i16", 2, "i16 -300", "i16 7", "i16 3000"),
        ("i32", 4, "i32 10", "i32 11", "i32 -20"),
        ("i64", 8, "i64 -1", "i64 1", "i64 123456789012"),
        ("ptr", 8, "ptr null", "ptr null", "@x"),
    ] {
        let src = cmpxchg_program(t, align, init, other, new);
        for level in LEVELS {
            assert_eq!(exit_code(&src, level, "cas"), 0, "{t} at {level:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Real concurrency: two threads, one counter
// ---------------------------------------------------------------------------

/// `main` clones a thread (`CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND |
/// CLONE_THREAD | CLONE_SYSVSEM`) whose stack is the top of a `.bss` array; both
/// threads call `@work`, which does 200000 `atomic_rmw add` on a shared
/// counter. The child then publishes `done` (release) and exits its thread; the
/// parent spins on `done` (acquire) and exits the group with 0 iff the counter
/// is exactly 400000. The child's path only calls a function (whose frame lives
/// on the new stack) and makes syscalls with constant operands, so it never
/// touches the parent's frame. (The threads really overlap: the same program
/// with a plain `load`/`add`/`store` increment loses updates.)
const TWO_THREADS: &str = r#"
module "threads"
global @counter : i64 = i64 0
global @done : i64 = i64 0
global @stack : [8192 x i64] = [8192 x i64] poison

func @work() -> void {
entry ^0:
  br ^1(i64 0)
^1(%i: i64):
  %o = atomic_rmw add seq_cst @counter, i64 1 align 8 : i64
  %i2 = add %i, i64 1 : i64
  %c = icmp slt %i2, i64 200000 : i1
  cond_br %c, ^1(%i2), ^2
^2:
  ret
}

func @main() -> i64 {
entry ^0:
  %top = ptr_add @stack, i64 65536 : ptr
  %topi = ptrtoint %top : i64
  %sp = and %topi, i64 -16 : i64
  %t = syscall i64 56, i64 331520, %sp, i64 0, i64 0, i64 0 : i64
  %is_child = icmp eq %t, i64 0 : i1
  cond_br %is_child, ^1, ^2
^1:
  call @work() : void
  atomic_store release i64 1, @done align 8 : i64
  %e = syscall i64 60, i64 0 : i64
  unreachable
^2:
  %failed = icmp slt %t, i64 0 : i1
  cond_br %failed, ^5, ^6
^6:
  call @work() : void
  br ^3
^3:
  %d = atomic_load acquire @done align 8 : i64
  %wait = icmp eq %d, i64 0 : i1
  cond_br %wait, ^3, ^4
^4:
  %v = atomic_load seq_cst @counter align 8 : i64
  %ok = icmp eq %v, i64 400000 : i1
  %r = select %ok, i64 0, i64 1 : i64
  %x = syscall i64 231, %r : i64
  unreachable
^5:
  %x2 = syscall i64 231, i64 2 : i64
  unreachable
}
"#;

#[test]
fn two_threads_increment_one_counter_atomically() {
    for level in [OptLevel::O0, OptLevel::O2] {
        for _ in 0..3 {
            assert_eq!(exit_code(TWO_THREADS, level, "threads"), 0, "lost an update at {level:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Machine-code shape
// ---------------------------------------------------------------------------

/// Four volatile loads of one device register (two identical and unused), two
/// identical volatile stores, and narrow volatile accesses; plus a loop polling
/// a volatile register.
const VOLATILE: &str = r#"
module "vol"
global @dev : i32 = i32 7
global @b8 : i8 = i8 1
global @h16 : i16 = i16 2

func @four() -> i32 {
entry ^0:
  %a = load volatile @dev align 4 : i32
  %b = load volatile @dev align 4 : i32
  %c = load volatile @dev align 4 : i32
  %d = load volatile @dev align 4 : i32
  store volatile i32 1, @dev align 4 : i32
  store volatile i32 1, @dev align 4 : i32
  %x = load volatile @b8 align 1 : i8
  store volatile %x, @b8 align 1 : i8
  %y = load volatile @h16 align 2 : i16
  store volatile %y, @h16 align 2 : i16
  %s = add %a, %b : i32
  ret %s
}

func @poll(i64) -> i32 {
entry ^0(%n: i64):
  br ^1(i64 0, i32 0)
^1(%i: i64, %acc: i32):
  %c = icmp slt %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %v = load volatile @dev align 4 : i32
  %acc2 = add %acc, %v : i32
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %acc2)
^3:
  ret %acc
}

func @main() -> i64 {
entry ^0:
  %f = call @four() : i32
  %p = call @poll(i64 5) : i32
  %f64 = sext %f : i64
  %p64 = sext %p : i64
  ; four() = 7 + 7 = 14 (it then stores 1); poll(5) = 5 * 1 = 5
  %t = add %f64, %p64 : i64
  %r = sub %t, i64 19 : i64
  ret %r
}
"#;

#[test]
fn volatile_accesses_survive_o2_at_their_exact_width() {
    for level in LEVELS {
        assert_eq!(exit_code(VOLATILE, level, "volatile"), 0, "at {level:?}");
    }
    let (m, syms) = prepare(VOLATILE, OptLevel::O2);
    let insts = ops_of(&mir_of(&m, &syms, "four"));
    let sizes = |op: X86Op| -> Vec<u64> {
        insts.iter().filter(|(o, _)| *o == op).map(|(_, ops)| imm_at(ops, 2)).collect()
    };
    // Exactly one machine access per volatile access, at exactly its width.
    assert_eq!(sizes(X86Op::Load), vec![4, 4, 4, 4, 1, 2], "volatile loads");
    assert_eq!(sizes(X86Op::Store), vec![4, 4, 1, 2], "volatile stores");

    // The polling load stays inside the loop (not hoisted by LICM): its block
    // lies on a CFG cycle.
    let poll = m.functions().position(|f| syms.resolve(f.name) == "poll").unwrap();
    let func = m.function(FuncId::from_index(poll));
    let load_block = func
        .blocks()
        .find(|(_, b)| b.insts().iter().any(|&i| func.inst(i).kind.is_volatile()))
        .map(|(id, _)| id.index())
        .expect("the volatile load survives");
    let succs = |b: usize| -> Vec<usize> {
        let blk = func.block(crate::ir::BlockId::from_index(b));
        blk.terminator().map(|t| func.inst(t).successors().iter().map(|s| s.index()).collect()).unwrap_or_default()
    };
    let mut seen = vec![false; func.block_count()];
    let mut stack = succs(load_block);
    let mut cyclic = false;
    while let Some(b) = stack.pop() {
        if b == load_block {
            cyclic = true;
            break;
        }
        if !std::mem::replace(&mut seen[b], true) {
            stack.extend(succs(b));
        }
    }
    assert!(cyclic, "the volatile load was hoisted out of the loop");
}

#[test]
fn atomic_lowering_follows_the_tso_mapping() {
    let src = r#"
module "shape"
global @x : i64 = i64 0
func @f(i64) -> i64 {
entry ^0(%v: i64):
  %a = atomic_load seq_cst @x align 8 : i64
  atomic_store relaxed %v, @x align 8 : i64
  atomic_store release %v, @x align 8 : i64
  atomic_store seq_cst %v, @x align 8 : i64
  fence acquire
  fence release
  fence acq_rel
  fence seq_cst
  %b = atomic_rmw xchg relaxed @x, %v align 8 : i64
  %c = atomic_rmw add acquire @x, %v align 8 : i64
  %d = atomic_rmw sub release @x, %v align 8 : i64
  %e = atomic_rmw xor seq_cst @x, %v align 8 : i64
  %g = cmpxchg seq_cst relaxed @x, %a, %v align 8 : i64
  %s1 = add %a, %b : i64
  %s2 = add %s1, %c : i64
  %s3 = add %s2, %d : i64
  %s4 = add %s3, %e : i64
  %s5 = add %s4, %g : i64
  ret %s5
}
"#;
    let (m, syms) = prepare(src, OptLevel::O2);
    let insts = ops_of(&mir_of(&m, &syms, "f"));
    let count = |op: X86Op| insts.iter().filter(|(o, _)| *o == op).count();
    assert_eq!(count(X86Op::Load), 1, "atomic_load is a plain mov");
    assert_eq!(count(X86Op::Store), 2, "relaxed/release stores are plain movs");
    assert_eq!(count(X86Op::Xchg), 2, "seq_cst store and rmw xchg use xchg");
    assert_eq!(count(X86Op::Mfence), 1, "only fence seq_cst emits mfence");
    assert_eq!(count(X86Op::LockXadd), 2, "add/sub use lock xadd");
    let negs: Vec<u64> =
        insts.iter().filter(|(o, _)| *o == X86Op::LockXadd).map(|(_, ops)| imm_at(ops, 4)).collect();
    assert_eq!(negs, vec![0, 1], "sub negates first");
    assert_eq!(count(X86Op::RmwLoop), 1, "xor uses the cmpxchg loop");
    assert_eq!(count(X86Op::LockCmpxchg), 1);
    // cmpxchg: expected moved into rax immediately before, old read out of rax
    // immediately after (the whole fixed-register window).
    let at = insts.iter().position(|(o, _)| *o == X86Op::LockCmpxchg).unwrap();
    let rax = MachineOperand::Def(crate::codegen::mir::Reg::Physical(super::regs::gpr(super::regs::RAX)));
    assert_eq!(insts[at - 1].0, X86Op::MovRR);
    assert_eq!(insts[at - 1].1[0], rax, "expected goes to rax");
    assert_eq!(insts[at].1[0], rax, "cmpxchg defines rax");
    assert_eq!(insts[at + 1].0, X86Op::MovRR);

    // And the bytes: mfence (0F AE F0), lock cmpxchg (F0 48 0F B1), lock xadd
    // (F0 48 0F C1), xchg r64 (48 87 / 4C 87 / 49 87 / 4D 87).
    let code = super::compile_function(&m, FuncId::from_index(0), &syms).bytes;
    let has = |needle: &[u8]| code.windows(needle.len()).any(|w| w == needle);
    assert!(has(&[0x0F, 0xAE, 0xF0]), "mfence emitted");
    assert!(has(&[0xF0, 0x48, 0x0F, 0xB1]) || has(&[0xF0, 0x4C, 0x0F, 0xB1]) || has(&[0xF0, 0x49, 0x0F, 0xB1]) || has(&[0xF0, 0x4D, 0x0F, 0xB1]), "lock cmpxchg emitted");
    assert!(code.windows(4).any(|w| w[0] == 0xF0 && (w[1] & 0xF8) == 0x48 && w[2] == 0x0F && w[3] == 0xC1), "lock xadd emitted");
    assert_eq!(code.windows(3).filter(|w| *w == [0x0F, 0xAE, 0xF0]).count(), 1, "exactly one mfence");
}

/// Our encodings, byte for byte, against `llvm-mc` (skipped when absent).
#[test]
fn atomic_encodings_match_llvm_mc() {
    use crate::mc::emit::Emitter;
    fn llvm_mc(asm: &str) -> Option<Vec<u8>> {
        use std::io::Write;
        let mut child = std::process::Command::new("llvm-mc")
            .args(["--triple=x86_64", "--show-encoding", "--x86-asm-syntax=intel", "-output-asm-variant=1"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok()?;
        child.stdin.as_mut()?.write_all(asm.as_bytes()).ok()?;
        let out = child.wait_with_output().ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let mut bytes = Vec::new();
        let mut rest = &text[..];
        while let Some(pos) = rest.find("encoding: [") {
            let start = pos + "encoding: [".len();
            let end = rest[start..].find(']')? + start;
            for tok in rest[start..end].split(',') {
                bytes.push(u8::from_str_radix(tok.trim().trim_start_matches("0x"), 16).ok()?);
            }
            rest = &rest[end..];
        }
        Some(bytes)
    }
    let enc = |f: &dyn Fn(&mut Emitter)| {
        let mut e = Emitter::new();
        f(&mut e);
        e.finish().expect("no labels").bytes
    };
    use super::encode::atomic_mem_rr_for_test as amr;
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (enc(&|e| amr(e, true, 0xB1, 1, 3, 8)), "lock cmpxchg qword ptr [rbx], rcx"),
        (enc(&|e| amr(e, true, 0xB1, 9, 12, 4)), "lock cmpxchg dword ptr [r12], r9d"),
        (enc(&|e| amr(e, true, 0xB1, 6, 13, 2)), "lock cmpxchg word ptr [r13], si"),
        (enc(&|e| amr(e, true, 0xB1, 7, 0, 1)), "lock cmpxchg byte ptr [rax], dil"),
        (enc(&|e| amr(e, true, 0xB1, 2, 5, 1)), "lock cmpxchg byte ptr [rbp], dl"),
        (enc(&|e| amr(e, true, 0xC1, 10, 4, 8)), "lock xadd qword ptr [rsp], r10"),
        (enc(&|e| amr(e, true, 0xC1, 1, 7, 1)), "lock xadd byte ptr [rdi], cl"),
        (enc(&|e| amr(e, false, 0x87, 11, 15, 8)), "xchg qword ptr [r15], r11"),
        (enc(&|e| amr(e, false, 0x87, 6, 1, 1)), "xchg byte ptr [rcx], sil"),
        (enc(&|e| amr(e, false, 0x87, 3, 2, 2)), "xchg word ptr [rdx], bx"),
        (enc(&|e| e.bytes(&[0x0F, 0xAE, 0xF0])), "mfence"),
    ];
    let mut checked = 0;
    for (ours, asm) in cases {
        match llvm_mc(asm) {
            Some(want) => {
                assert_eq!(ours, want, "`{asm}`");
                checked += 1;
            }
            None => eprintln!("skipping llvm-mc cross-check of `{asm}`"),
        }
    }
    eprintln!("checked {checked} x86-64 atomic encodings against llvm-mc");
}
