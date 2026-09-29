//! Stack-usage reports, large frames, and stack probes on AArch64.
//!
//! This host cannot run A64 code, so compiled leaf functions are executed by a
//! small instruction-level emulator covering the forms these tests' functions
//! use. The emulated stack has a guard page (an access there is the `SIGSEGV`
//! a real guard gives) and nothing mapped below it (an access there is the
//! silent corruption probes exist to prevent). The tests check that:
//!
//! - the deepest `sp` reached equals the reported frame size, the function
//!   computes the right value (large `sp` offsets encode correctly), and `sp` is
//!   restored;
//! - with probes on, no stack write lands more than one probe interval below
//!   the deepest earlier write, so an overflowing frame always hits the guard;
//!   with probes off, the same frame skips it;
//! - the probe words are what `llvm-mc` assembles from the intended sequence.

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
// A tiny A64 emulator
// ---------------------------------------------------------------------------

/// How an emulated run ended.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// `ret` with this `x0`.
    Returned(u64),
    /// An access inside the guard page: the defined `SIGSEGV`.
    Guard,
    /// An access below the guard page: a frame jumped over it.
    Skipped,
}

/// The emulated stack: the region `[guard_lo, top)` is mapped, with the guard
/// page at `[guard_lo, guard_lo + 4096)`; below it nothing is mapped.
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
    /// `sp` when the function returned equals the entry `sp`.
    sp_restored: bool,
}

fn sext(v: u64, bits: u32) -> i64 {
    ((v << (64 - bits)) as i64) >> (64 - bits)
}

/// Run the leaf function `code` with `x0 = arg` on `stack`.
fn emulate(code: &[u8], arg: u64, stack: &Stack) -> Run {
    let mut x = [0u64; 32]; // x31 is unused: sp is separate, xzr reads as 0
    let entry_sp = stack.top - 256;
    let mut sp = entry_sp;
    x[0] = arg;
    x[30] = 0xDEAD_0000;
    let mut z = false;
    let mut mem: std::collections::HashMap<u64, u8> = Default::default();
    let mut writes = Vec::new();
    let mut max_depth = 0u64;
    let mut pc = 0usize;
    let guard_hi = stack.guard_lo + STACK_PROBE_INTERVAL;

    // Register read: 31 is sp or xzr depending on the operand.
    macro_rules! rd_sp { ($r:expr) => { if $r == 31 { sp } else { x[$r] } } }
    macro_rules! rd_zr { ($r:expr) => { if $r == 31 { 0 } else { x[$r] } } }

    for _ in 0..50_000_000u64 {
        max_depth = max_depth.max(entry_sp - sp);
        let w = u32::from_le_bytes(code[pc..pc + 4].try_into().unwrap());
        pc += 4;
        let rd = (w & 31) as usize;
        let rn = ((w >> 5) & 31) as usize;
        let rm = ((w >> 16) & 31) as usize;
        let sf = w >> 31 == 1;
        let mask = |v: u64| if sf { v } else { v & 0xFFFF_FFFF };
        macro_rules! mem_op {
            ($addr:expr, $size:expr, $load:expr, $rt:expr) => {{
                let a: u64 = $addr;
                let fault = if a < stack.guard_lo {
                    Some(Outcome::Skipped)
                } else if a < guard_hi {
                    Some(Outcome::Guard)
                } else {
                    None
                };
                if let Some(o) = fault {
                    return Run { outcome: o, max_depth, writes, sp_restored: false };
                }
                if !$load && a < stack.top {
                    writes.push(entry_sp as i64 - a as i64);
                }
                if $load {
                    let mut v = 0u64;
                    for i in 0..$size {
                        v |= u64::from(*mem.get(&(a + i)).unwrap_or(&0)) << (8 * i);
                    }
                    if $rt != 31 {
                        x[$rt] = v;
                    }
                } else {
                    let v = rd_zr!($rt);
                    for i in 0..$size {
                        mem.insert(a + i, (v >> (8 * i)) as u8);
                    }
                }
            }};
        }
        if w == 0xD65F_03C0 {
            // ret
            return Run {
                outcome: Outcome::Returned(x[0]),
                max_depth,
                writes,
                sp_restored: sp == entry_sp,
            };
        } else if w & 0xFFC0_0000 == 0xA980_0000 || w & 0xFFC0_0000 == 0xA8C0_0000 {
            // stp (pre-index) / ldp (post-index), 64-bit
            let load = w & 0xFFC0_0000 == 0xA8C0_0000;
            let imm = sext(u64::from((w >> 15) & 0x7F), 7) * 8;
            let rt2 = ((w >> 10) & 31) as usize;
            let base = rd_sp!(rn);
            let a = if load { base } else { base.wrapping_add(imm as u64) };
            mem_op!(a, 8, load, rd);
            mem_op!(a + 8, 8, load, rt2);
            let nb = base.wrapping_add(imm as u64);
            if rn == 31 { sp = nb } else { x[rn] = nb }
        } else if w & 0x1F00_0000 == 0x1100_0000 {
            // add/sub (immediate), optional lsl #12, optional flags
            let sub = w & (1 << 30) != 0;
            let s = w & (1 << 29) != 0;
            let mut imm = u64::from((w >> 10) & 0xFFF);
            if w & (1 << 22) != 0 {
                imm <<= 12;
            }
            let a = rd_sp!(rn);
            let r = mask(if sub { a.wrapping_sub(imm) } else { a.wrapping_add(imm) });
            if s {
                z = r == 0;
                if rd != 31 { x[rd] = r }
            } else if rd == 31 {
                sp = r;
            } else {
                x[rd] = r;
            }
        } else if w & 0x7FE0_0000 == 0x0B20_0000 || w & 0x7FE0_0000 == 0x4B20_0000 {
            // add/sub (extended register, uxtx): Xd|SP = Xn|SP ± Xm
            let sub = w & (1 << 30) != 0;
            let a = rd_sp!(rn);
            let r = if sub { a.wrapping_sub(x[rm]) } else { a.wrapping_add(x[rm]) };
            if rd == 31 { sp = r } else { x[rd] = r }
        } else if w & 0x1F80_0000 == 0x1280_0000 {
            // movn / movz / movk
            let opc = (w >> 29) & 3;
            let hw = (w >> 21) & 3;
            let imm = u64::from((w >> 5) & 0xFFFF) << (16 * hw);
            let v = match opc {
                0 => !imm,
                2 => imm,
                3 => (x[rd] & !(0xFFFFu64 << (16 * hw))) | imm,
                _ => panic!("bad move-wide {w:#010x}"),
            };
            x[rd] = mask(v);
        } else if w & 0x1F20_0000 == 0x0A00_0000 || w & 0x1F20_0000 == 0x0B00_0000 {
            // logical / add-sub (shifted register, shift 0 only)
            assert_eq!((w >> 10) & 0x3F, 0, "shifted register form {w:#010x}");
            let a = rd_zr!(rn);
            let b = rd_zr!(rm);
            let r = match (w >> 24) & 0x1F {
                0x0A => match (w >> 29) & 3 {
                    0 => a & b,
                    1 => a | b,
                    2 => a ^ b,
                    _ => panic!("ands {w:#010x}"),
                },
                _ => {
                    let sub = w & (1 << 30) != 0;
                    let r = if sub { a.wrapping_sub(b) } else { a.wrapping_add(b) };
                    if w & (1 << 29) != 0 {
                        z = mask(r) == 0;
                    }
                    r
                }
            };
            if rd != 31 { x[rd] = mask(r) }
        } else if w & 0x3B00_0000 == 0x3900_0000 {
            // ldr/str (unsigned immediate), sizes 1/2/4/8
            let size = 1u64 << (w >> 30);
            let load = w & (1 << 22) != 0;
            let a = rd_sp!(rn) + u64::from((w >> 10) & 0xFFF) * size;
            mem_op!(a, size, load, rd);
        } else if w & 0x7F80_0000 == 0x5300_0000 {
            // ubfm (ubfx / lsr / uxt*)
            let immr = (w >> 16) & 0x3F;
            let imms = (w >> 10) & 0x3F;
            assert!(imms >= immr, "ubfm as lsl {w:#010x}");
            let width = imms - immr + 1;
            let v = rd_zr!(rn) >> immr;
            x[rd] = if width == 64 { v } else { v & ((1u64 << width) - 1) };
        } else if w & 0xFF00_0010 == 0x5400_0000 {
            // b.cond (eq / ne)
            let off = sext(u64::from((w >> 5) & 0x7FFFF), 19) * 4;
            let take = match w & 0xF {
                0 => z,
                1 => !z,
                c => panic!("b.cond {c}"),
            };
            if take {
                pc = (pc as i64 - 4 + off) as usize;
            }
        } else {
            panic!("emulator: unsupported word {w:#010x} at {:#x}", pc - 4);
        }
    }
    panic!("emulation does not terminate");
}

/// A roomy stack whose guard is far below any test frame.
fn roomy() -> Stack {
    Stack { top: 0x7000_0000_0000, guard_lo: 0x7000_0000_0000 - (64 << 20) }
}

/// Assert the probe invariant over a run's writes, from a caller whose deepest
/// write may be up to 4080 bytes above `sp` (the invariant AArch64 frames keep),
/// and that the function leaves `sp` at most 4080 below its deepest write.
fn assert_probed(r: &Run, frame: u64, what: &str) {
    let interval = STACK_PROBE_INTERVAL as i64;
    let mut deepest = -4080i64;
    for &t in &r.writes {
        assert!(t - deepest <= interval, "{what}: write at depth {t} skips from {deepest}");
        deepest = deepest.max(t);
    }
    assert!(frame as i64 - deepest <= 4080, "{what}: frame bottom {frame} vs deepest {deepest}");
}

/// Leaf functions from tiny to 1 MiB frames (every one returns `x + 7`), with
/// the large frames reaching their far ends through big `sp` offsets.
const FRAMES: &str = r#"
module "frames"
func @tiny(i64) -> i64 {
entry ^0(%x: i64):
  %r = add %x, i64 7 : i64
  ret %r
}
func @mid(i64) -> i64 {
entry ^0(%x: i64):
  %a = alloca [3000 x i8] : ptr
  %hi = ptr_add %a, i64 2999 : ptr
  store i8 3, %a align 1 : i8
  store i8 4, %hi align 1 : i8
  %p = load %a align 1 : i8
  %q = load %hi align 1 : i8
  %s = add %p, %q : i8
  %w = zext %s : i64
  %r = add %w, %x : i64
  ret %r
}
func @page(i64) -> i64 {
entry ^0(%x: i64):
  %a = alloca [4064 x i8] : ptr
  %hi = ptr_add %a, i64 4063 : ptr
  store i8 3, %a align 1 : i8
  store i8 4, %hi align 1 : i8
  %p = load %a align 1 : i8
  %q = load %hi align 1 : i8
  %s = add %p, %q : i8
  %w = zext %s : i64
  %r = add %w, %x : i64
  ret %r
}
func @big(i64) -> i64 {
entry ^0(%x: i64):
  %a = alloca [70000 x i8] : ptr
  %hi = ptr_add %a, i64 69999 : ptr
  store i8 3, %a align 1 : i8
  store i8 4, %hi align 1 : i8
  %p = load %a align 1 : i8
  %q = load %hi align 1 : i8
  %s = add %p, %q : i8
  %w = zext %s : i64
  %r = add %w, %x : i64
  ret %r
}
func @huge(i64) -> i64 {
entry ^0(%x: i64):
  %a = alloca [1048576 x i8] : ptr
  %hi = ptr_add %a, i64 1048575 : ptr
  store i8 3, %a align 1 : i8
  store i8 4, %hi align 1 : i8
  %p = load %a align 1 : i8
  %q = load %hi align 1 : i8
  %s = add %p, %q : i8
  %w = zext %s : i64
  %r = add %w, %x : i64
  ret %r
}
"#;

#[test]
fn frames_run_and_match_the_report() {
    let (m, syms) = prepare(FRAMES);
    for probes in [true, false] {
        let out = compile_module_with(&m, &syms, &CodegenOptions::default().with_stack_probes(probes));
        assert_eq!(out.stack.functions().len(), 5);
        for u in out.stack.functions() {
            let what = format!("{} (probes {probes})", u.name);
            let run = emulate(func_bytes(&out.object, &u.name), 100, &roomy());
            assert_eq!(run.outcome, Outcome::Returned(107), "{what}");
            assert!(run.sp_restored, "{what}: sp restored");
            assert_eq!(run.max_depth, u.frame_size, "{what}: deepest sp vs reported frame");
            assert_eq!(u.frame_size, 16 + u.sp_adjust, "{what}: fp/lr pair + sub sp");
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
    let (m, syms) = prepare(FRAMES);
    // A guard page 32 KiB below the stack top: `big` (70 KiB) and `huge`
    // (1 MiB) overflow into it.
    let top = 0x7000_0000_0000u64;
    let tight = Stack { top, guard_lo: top - 32 * 1024 };
    for name in ["big", "huge"] {
        let on = compile_module_with(&m, &syms, &CodegenOptions::default());
        let off = compile_module_with(&m, &syms, &CodegenOptions::default().with_stack_probes(false));
        assert_eq!(emulate(func_bytes(&on.object, name), 1, &tight).outcome, Outcome::Guard, "{name}");
        assert_eq!(emulate(func_bytes(&off.object, name), 1, &tight).outcome, Outcome::Skipped, "{name}");
    }
}

/// Assemble `asm` with `llvm-mc`, returning the concatenated encodings (`None`
/// when `llvm-mc` is unavailable).
fn llvm_mc(asm: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let mut child = std::process::Command::new("llvm-mc")
        .args(["--triple=aarch64", "--show-encoding"])
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
    let (m, syms) = prepare(FRAMES);
    let out = compile_module_with(&m, &syms, &CodegenOptions::default());
    // `big`: 70016 - 16 = 70000 bytes below the fp/lr pair, 17 pages + 368.
    let big = out.stack.get("big").unwrap();
    assert_eq!(big.sp_adjust, 17 * 4096 + 368);
    let looped = "stp x29, x30, [sp, #-16]!\nmov x29, sp\nmovz x16, #17\n\
                  sub sp, sp, #1, lsl #12\nstr xzr, [sp]\nsubs x16, x16, #1\nb.ne #-12\n\
                  sub sp, sp, #368\n";
    // `page`: under one interval, a single sub.
    let page = out.stack.get("page").unwrap();
    assert!(page.sp_adjust < 4096);
    let single = format!("stp x29, x30, [sp, #-16]!\nmov x29, sp\nsub sp, sp, #{}\n", page.sp_adjust);
    for (name, asm) in [("big", looped.to_owned()), ("page", single)] {
        let Some(want) = llvm_mc(&asm) else {
            eprintln!("skipping probe_sequences_match_llvm_mc: no llvm-mc");
            return;
        };
        let got = func_bytes(&out.object, name);
        assert_eq!(&got[..want.len()], &want[..], "{name} prologue");
    }
    // Unrolled: 2 pages + remainder, from a frame of 2*4096 + 16 + r.
    let src = "module \"u\"\nfunc @two(i64) -> i64 {\nentry ^0(%x: i64):\n  %a = alloca [9000 x i8] : ptr\n  store i8 1, %a align 1 : i8\n  ret %x\n}\n";
    let (m2, syms2) = prepare(src);
    let out2 = compile_module_with(&m2, &syms2, &CodegenOptions::default());
    let adj = out2.stack.get("two").unwrap().sp_adjust;
    assert_eq!(adj / 4096, 2);
    let asm = format!(
        "stp x29, x30, [sp, #-16]!\nmov x29, sp\nsub sp, sp, #1, lsl #12\nstr xzr, [sp]\n\
         sub sp, sp, #1, lsl #12\nstr xzr, [sp]\nsub sp, sp, #{}\n",
        adj % 4096
    );
    let want = llvm_mc(&asm).expect("llvm-mc ran above");
    assert_eq!(&func_bytes(&out2.object, "two")[..want.len()], &want[..]);
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
  %fp = select i1 1, @leaf, @ext : ptr
  %c = call %fp(%b) : i64
  %e = syscall i64 172 : i64
  %s = add %c, %e : i64
  ret %s
}
"#;
    let (m, syms) = prepare(src);
    let out = compile_module_with(&m, &syms, &CodegenOptions::default());
    let t = out.stack.get("top").unwrap();
    assert_eq!(t.direct_callees, ["leaf", "ext"]);
    assert!(t.indirect_calls && t.syscalls && !t.dynamic_alloca);
    assert!(t.saved_registers >= 16);
    let a = StackAssumptions::new().external("ext", 100).indirect("top", 200);
    let b = out.stack.worst_case_depth("top", &a).unwrap();
    assert_eq!(b.bytes, t.frame_size + 200);
}

/// `n` values derived from `x`, all live at once (so most spill), summed after
/// a 1 MiB `alloca`: the spill slots sit above the array, beyond every
/// immediate `sp` offset. Returns `n*x + n*(n-1)/2 + 9`.
fn spill_heavy(n: usize) -> String {
    let mut s = String::from(
        "module \"spill\"\nfunc @spill(i64) -> i64 {\nentry ^0(%x: i64):\n  \
         %a = alloca [1048576 x i8] : ptr\n  store i8 9, %a align 1 : i8\n",
    );
    for i in 0..n {
        s.push_str(&format!("  %v{i} = add %x, i64 {i} : i64\n"));
    }
    s.push_str("  %b = load %a align 1 : i8\n  %acc0 = zext %b : i64\n");
    for i in 0..n {
        s.push_str(&format!("  %acc{} = add %acc{i}, %v{i} : i64\n", i + 1));
    }
    s.push_str(&format!("  ret %acc{n}\n}}\n"));
    s
}

#[test]
fn spills_beyond_a_large_frame() {
    let n = 40;
    let (m, syms) = prepare(&spill_heavy(n));
    for probes in [true, false] {
        let out = compile_module_with(&m, &syms, &CodegenOptions::default().with_stack_probes(probes));
        let u = out.stack.get("spill").unwrap();
        assert!(u.frame_size > (1 << 20) + 8 * 16, "spill slots above the array: {}", u.frame_size);
        let run = emulate(func_bytes(&out.object, "spill"), 1000, &roomy());
        let n = n as u64;
        assert_eq!(run.outcome, Outcome::Returned(n * 1000 + n * (n - 1) / 2 + 9), "probes {probes}");
        assert!(run.sp_restored);
        assert_eq!(run.max_depth, u.frame_size);
    }
}
