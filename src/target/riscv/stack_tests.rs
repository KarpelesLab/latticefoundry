//! Stack-usage reports, large frames, and stack probes on RISC-V.
//!
//! As on AArch64, compiled leaf functions run in a small RV64IM emulator whose
//! stack has a guard page (an access there is the defined `SIGSEGV`) and nothing
//! mapped below it (an access there is the silent corruption probes prevent).
//! The tests check that the deepest `sp` equals the reported frame size, that
//! results and `sp` are right for frames far beyond the 12-bit immediates, that
//! with probes on no stack write lands more than one probe interval below the
//! deepest earlier one (so overflows hit the guard, while unprobed frames skip
//! it), and that the probe words match `llvm-mc`.

use super::encode::compile_module_with;
use crate::codegen::CodegenOptions;
use crate::codegen::stack::{STACK_PROBE_INTERVAL, StackAssumptions};
use crate::ir::Module;
use crate::mc::object::{ObjectModule, SymbolValue};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

fn prepare(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse .lf: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    (m, syms)
}

fn func_bytes<'a>(obj: &'a ObjectModule, name: &str) -> &'a [u8] {
    let sym = obj.symbols().iter().find(|s| s.name == name).expect("function symbol");
    let SymbolValue::Defined { section, offset } = sym.value else { panic!("{name} undefined") };
    let bytes = &obj.section(section).bytes;
    &bytes[offset as usize..(offset + sym.size) as usize]
}

// ---------------------------------------------------------------------------
// A tiny RV64IM emulator
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// `ret` with this `a0`.
    Returned(u64),
    /// An access inside the guard page: the defined `SIGSEGV`.
    Guard,
    /// An access below the guard page: a frame jumped over it.
    Skipped,
}

/// The emulated stack: `[guard_lo, top)` is mapped, the guard page is
/// `[guard_lo, guard_lo + 4096)`, and nothing is mapped below.
struct Stack {
    top: u64,
    guard_lo: u64,
}

struct Run {
    outcome: Outcome,
    /// The deepest `sp` reached, as a depth below the entry `sp`.
    max_depth: u64,
    /// Every stack write's depth below the entry `sp`, in order.
    writes: Vec<i64>,
    sp_restored: bool,
}

fn sext(v: u64, bits: u32) -> i64 {
    ((v << (64 - bits)) as i64) >> (64 - bits)
}

/// Run the leaf function `code` with `a0 = arg` on `stack`.
fn emulate(code: &[u8], arg: u64, stack: &Stack) -> Run {
    let mut x = [0u64; 32];
    let entry_sp = stack.top - 256;
    x[2] = entry_sp;
    x[1] = 0xDEAD_0000; // ra
    x[10] = arg;
    let mut mem: std::collections::HashMap<u64, u8> = Default::default();
    let mut writes = Vec::new();
    let mut max_depth = 0u64;
    let mut pc = 0usize;
    let guard_hi = stack.guard_lo + STACK_PROBE_INTERVAL;
    for _ in 0..50_000_000u64 {
        x[0] = 0;
        max_depth = max_depth.max(entry_sp.wrapping_sub(x[2]));
        let w = u32::from_le_bytes(code[pc..pc + 4].try_into().unwrap());
        let at = pc;
        pc += 4;
        let op = w & 0x7F;
        let rd = ((w >> 7) & 31) as usize;
        let f3 = (w >> 12) & 7;
        let rs1 = ((w >> 15) & 31) as usize;
        let rs2 = ((w >> 20) & 31) as usize;
        let f7 = w >> 25;
        let imm_i = sext(u64::from(w >> 20), 12);
        let (a, b) = (x[rs1], x[rs2]);
        let mut set = |r: usize, v: u64| {
            if r != 0 {
                x[r] = v;
            }
        };
        match op {
            0x37 => set(rd, sext(u64::from(w & 0xFFFF_F000), 32) as u64), // lui
            0x13 => {
                let sh = (w >> 20) & 0x3F;
                let v = match f3 {
                    0 => a.wrapping_add(imm_i as u64),
                    1 => a << sh,
                    3 => u64::from(a < imm_i as u64),
                    4 => a ^ imm_i as u64,
                    5 if w >> 30 & 1 == 1 => ((a as i64) >> sh) as u64,
                    5 => a >> sh,
                    6 => a | imm_i as u64,
                    7 => a & imm_i as u64,
                    _ => panic!("op-imm {w:#010x}"),
                };
                set(rd, v);
            }
            0x1B => {
                assert_eq!(f3, 0, "op-imm-32 {w:#010x}");
                set(rd, sext(a.wrapping_add(imm_i as u64) & 0xFFFF_FFFF, 32) as u64); // addiw
            }
            0x33 => {
                let v = match (f7, f3) {
                    (0, 0) => a.wrapping_add(b),
                    (0x20, 0) => a.wrapping_sub(b),
                    (0, 7) => a & b,
                    (0, 6) => a | b,
                    (0, 4) => a ^ b,
                    (0, 1) => a << (b & 63),
                    (0, 5) => a >> (b & 63),
                    (0x20, 5) => ((a as i64) >> (b & 63)) as u64,
                    (0, 3) => u64::from(a < b),
                    (0, 2) => u64::from((a as i64) < (b as i64)),
                    (1, 0) => a.wrapping_mul(b),
                    _ => panic!("op {w:#010x}"),
                };
                set(rd, v);
            }
            0x03 | 0x23 => {
                let load = op == 0x03;
                let imm = if load {
                    imm_i
                } else {
                    sext(u64::from(((w >> 25) << 5) | ((w >> 7) & 31)), 12)
                };
                let addr = a.wrapping_add(imm as u64);
                let size = 1u64 << (f3 & 3);
                let fault = if addr < stack.guard_lo {
                    Some(Outcome::Skipped)
                } else if addr < guard_hi {
                    Some(Outcome::Guard)
                } else {
                    None
                };
                if let Some(o) = fault {
                    return Run { outcome: o, max_depth, writes, sp_restored: false };
                }
                if load {
                    let mut v = 0u64;
                    for i in 0..size {
                        v |= u64::from(*mem.get(&(addr + i)).unwrap_or(&0)) << (8 * i);
                    }
                    if f3 < 4 && size < 8 {
                        v = sext(v, 8 * size as u32) as u64; // signed lb/lh/lw
                    }
                    set(rd, v);
                } else {
                    if addr < stack.top {
                        writes.push(entry_sp as i64 - addr as i64);
                    }
                    for i in 0..size {
                        mem.insert(addr + i, (b >> (8 * i)) as u8);
                    }
                }
            }
            0x63 => {
                let imm = ((w >> 31) << 12) | (((w >> 7) & 1) << 11) | (((w >> 25) & 0x3F) << 5) | (((w >> 8) & 0xF) << 1);
                let off = sext(u64::from(imm), 13);
                let take = match f3 {
                    0 => a == b,
                    1 => a != b,
                    4 => (a as i64) < (b as i64),
                    5 => (a as i64) >= (b as i64),
                    6 => a < b,
                    7 => a >= b,
                    _ => panic!("branch {w:#010x}"),
                };
                if take {
                    pc = (at as i64 + off) as usize;
                }
            }
            0x67 if rd == 0 && rs1 == 1 => {
                return Run {
                    outcome: Outcome::Returned(x[10]),
                    max_depth,
                    writes,
                    sp_restored: x[2] == entry_sp,
                };
            }
            _ => panic!("emulator: unsupported word {w:#010x} at {at:#x}"),
        }
    }
    panic!("emulation does not terminate");
}

fn roomy() -> Stack {
    Stack { top: 0x7000_0000_0000, guard_lo: 0x7000_0000_0000 - (64 << 20) }
}

/// Assert the probe invariant over a leaf's writes: the caller's deepest write
/// is at its `sp` (every calling RISC-V frame saves `ra` at `0(sp)`), no write
/// lands more than one interval below the deepest earlier one, and the frame
/// bottom is within one interval of the deepest write.
fn assert_probed(r: &Run, frame: u64, what: &str) {
    let interval = STACK_PROBE_INTERVAL as i64;
    let mut deepest = 0i64;
    for &t in &r.writes {
        assert!(t - deepest <= interval, "{what}: write at depth {t} skips from {deepest}");
        deepest = deepest.max(t);
    }
    assert!(frame as i64 - deepest <= interval, "{what}: frame bottom {frame} vs deepest {deepest}");
}

/// A leaf with an `n`-byte array touched at both ends, returning `x + 7`.
fn frame_fn(name: &str, n: u64) -> String {
    format!(
        "func @{name}(i64) -> i64 {{\nentry ^0(%x: i64):\n  %a = alloca [{n} x i8] : ptr\n  \
         %hi = ptr_add %a, i64 {} : ptr\n  store i8 3, %a align 1 : i8\n  \
         store i8 4, %hi align 1 : i8\n  %p = load %a align 1 : i8\n  \
         %q = load %hi align 1 : i8\n  %s = add %p, %q : i8\n  %w = zext %s : i64\n  \
         %r = add %w, %x : i64\n  ret %r\n}}\n",
        n - 1
    )
}

fn frames_src() -> String {
    let mut s = String::from(
        "module \"frames\"\nfunc @tiny(i64) -> i64 {\nentry ^0(%x: i64):\n  %r = add %x, i64 7 : i64\n  ret %r\n}\n",
    );
    for (name, n) in [("mid", 3000), ("page", 4064), ("big", 70000), ("huge", 1 << 20)] {
        s.push_str(&frame_fn(name, n));
    }
    s
}

#[test]
fn frames_run_and_match_the_report() {
    let (m, syms) = prepare(&frames_src());
    for probes in [true, false] {
        let out = compile_module_with(&m, &syms, &CodegenOptions::default().with_stack_probes(probes));
        assert_eq!(out.stack.functions().len(), 5);
        for u in out.stack.functions() {
            let what = format!("{} (probes {probes})", u.name);
            let run = emulate(func_bytes(&out.object, &u.name), 100, &roomy());
            assert_eq!(run.outcome, Outcome::Returned(107), "{what}");
            assert!(run.sp_restored, "{what}: sp restored");
            assert_eq!(run.max_depth, u.frame_size, "{what}: deepest sp vs reported frame");
            assert_eq!(u.frame_size, u.sp_adjust, "{what}");
            assert_eq!(u.return_address, 0);
            assert_eq!(u.frame_size % 16, 0);
            if probes {
                assert_probed(&run, u.frame_size, &what);
            }
            let b = out.stack.worst_case_depth(&u.name, &StackAssumptions::new()).unwrap();
            assert_eq!(b.bytes, u.frame_size);
        }
    }
}

#[test]
fn probes_hit_the_guard_instead_of_skipping_it() {
    let (m, syms) = prepare(&frames_src());
    let top = 0x7000_0000_0000u64;
    let tight = Stack { top, guard_lo: top - 32 * 1024 };
    let on = compile_module_with(&m, &syms, &CodegenOptions::default());
    let off = compile_module_with(&m, &syms, &CodegenOptions::default().with_stack_probes(false));
    for name in ["big", "huge"] {
        assert_eq!(emulate(func_bytes(&on.object, name), 1, &tight).outcome, Outcome::Guard, "{name}");
        assert_eq!(emulate(func_bytes(&off.object, name), 1, &tight).outcome, Outcome::Skipped, "{name}");
    }
}

#[test]
fn spills_beyond_a_large_frame() {
    // 40 values live at once after a 1 MiB alloca: their spill slots sit above
    // the array, far beyond the 12-bit offsets.
    let n = 40usize;
    let mut src = String::from(
        "module \"spill\"\nfunc @spill(i64) -> i64 {\nentry ^0(%x: i64):\n  \
         %a = alloca [1048576 x i8] : ptr\n  store i8 9, %a align 1 : i8\n",
    );
    for i in 0..n {
        src.push_str(&format!("  %v{i} = add %x, i64 {i} : i64\n"));
    }
    src.push_str("  %b = load %a align 1 : i8\n  %acc0 = zext %b : i64\n");
    for i in 0..n {
        src.push_str(&format!("  %acc{} = add %acc{i}, %v{i} : i64\n", i + 1));
    }
    src.push_str(&format!("  ret %acc{n}\n}}\n"));
    let (m, syms) = prepare(&src);
    for probes in [true, false] {
        let out = compile_module_with(&m, &syms, &CodegenOptions::default().with_stack_probes(probes));
        let u = out.stack.get("spill").unwrap();
        let run = emulate(func_bytes(&out.object, "spill"), 1000, &roomy());
        let n = n as u64;
        assert_eq!(run.outcome, Outcome::Returned(n * 1000 + n * (n - 1) / 2 + 9), "probes {probes}");
        assert!(run.sp_restored);
        assert_eq!(run.max_depth, u.frame_size);
    }
}

/// Assemble `asm` with `llvm-mc` (RV64IM), concatenating the encodings.
fn llvm_mc(asm: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let mut child = std::process::Command::new("llvm-mc")
        .args(["--triple=riscv64", "--mattr=+m", "--riscv-no-aliases", "--show-encoding"])
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
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut bytes = Vec::new();
    for part in text.split("encoding: [").skip(1) {
        let end = part.find(']')?;
        for tok in part[..end].split(',') {
            bytes.push(u8::from_str_radix(tok.trim().trim_start_matches("0x"), 16).ok()?);
        }
    }
    Some(bytes)
}

#[test]
fn probe_sequences_match_llvm_mc() {
    let src = format!("module \"p\"\n{}{}", frame_fn("two", 9000), frame_fn("big", 70000));
    let (m, syms) = prepare(&src);
    let out = compile_module_with(&m, &syms, &CodegenOptions::default());
    // Unrolled (2 pages) and looped (17 pages) forms, each with its remainder.
    let two = out.stack.get("two").unwrap().sp_adjust;
    let big = out.stack.get("big").unwrap().sp_adjust;
    assert_eq!((two / 4096, big / 4096), (2, 17));
    let rem = |adj: u64| -> String {
        let r = adj % 4096;
        if r <= 2048 {
            format!("addi sp, sp, -{r}\n")
        } else {
            // li t6, -r (lui + addiw) then add sp, sp, t6
            let v = -(r as i64);
            let lo = sext((v as u64) & 0xFFF, 12);
            let hi = ((v - lo) >> 12) & 0xFFFFF;
            format!("lui t6, {hi}\naddiw t6, t6, {lo}\nadd sp, sp, t6\n")
        }
    };
    let unrolled = format!(
        "lui t1, 1\nsub sp, sp, t1\nsd zero, 0(sp)\nsub sp, sp, t1\nsd zero, 0(sp)\n{}",
        rem(two)
    );
    let looped = format!(
        "lui t1, 1\naddi t0, zero, 17\nsub sp, sp, t1\nsd zero, 0(sp)\naddi t0, t0, -1\n\
         bne t0, zero, -12\n{}",
        rem(big)
    );
    for (name, asm) in [("two", unrolled), ("big", looped)] {
        let Some(want) = llvm_mc(&asm) else {
            eprintln!("skipping probe_sequences_match_llvm_mc: no llvm-mc");
            return;
        };
        assert_eq!(&func_bytes(&out.object, name)[..want.len()], &want[..], "{name} prologue");
    }
}

#[test]
fn report_lists_calls_and_syscalls() {
    let src = r#"
module "calls"
func @ext(i64) -> i64
func @leaf(i64) -> i64 {
entry ^0(%x: i64):
  ret %x
}
func @top(i64) -> i64 {
entry ^0(%x: i64):
  %a = call @leaf(%x) : i64
  %b = call @ext(%a) : i64
  %e = syscall i64 172 : i64
  %s = add %b, %e : i64
  ret %s
}
"#;
    let (m, syms) = prepare(src);
    let out = compile_module_with(&m, &syms, &CodegenOptions::default());
    let t = out.stack.get("top").unwrap();
    assert_eq!(t.direct_callees, ["leaf", "ext"]);
    assert!(!t.indirect_calls && t.syscalls && !t.dynamic_alloca);
    assert!(t.saved_registers >= 8, "ra is saved in a calling function");
    let a = StackAssumptions::new().external("ext", 100);
    let b = out.stack.worst_case_depth("top", &a).unwrap();
    let leaf = out.stack.get("leaf").unwrap().frame_size;
    assert_eq!(b.bytes, t.frame_size + leaf.max(100));
}
