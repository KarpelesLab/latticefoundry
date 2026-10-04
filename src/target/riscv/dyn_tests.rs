//! `dyn_alloca` on RISC-V: a function that moves `sp` at run time addresses
//! its frame through the frame pointer `s0`, keeps its outgoing-argument area
//! at the bottom of the stack, and (with stack probes) touches every page it
//! allocates. Checked differentially against the reference semantics and,
//! on the simulator, for the stack-probe invariant, guard-page behavior and
//! the measured stack depth.

use crate::codegen::CodegenOptions;
use crate::codegen::stack::{STACK_PROBE_INTERVAL, StackAssumptions};

use super::diff_tests::{Harness, parse};
use super::sim::{Cpu, Fault, STACK_TOP, StackGuard};

const DYN: &str = r#"
module "dyn"
func @fill(i64) -> i64 {
entry ^0(%n: i64):
  %p = dyn_alloca %n align 16 : ptr
  br ^1(i64 0)
^1(%i: i64):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^3(i64 0, i64 0)
^2:
  %q = ptr_add %p, %i : ptr
  %b = trunc %i : i8
  store %b, %q align 1 : i8
  %j = add %i, i64 1 : i64
  br ^1(%j)
^3(%k: i64, %s: i64):
  %d = icmp ult %k, %n : i1
  cond_br %d, ^4, ^5
^4:
  %r = ptr_add %p, %k : ptr
  %v = load %r align 1 : i8
  %w = zext %v : i64
  %s2 = add %s, %w : i64
  %k2 = add %k, i64 1 : i64
  br ^3(%k2, %s2)
^5:
  ret %s
}

func @nine(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64, %e: i64, %f: i64, %g: i64, %h: i64, %i: i64, %j: i64):
  %x = mul %i, i64 1000 : i64
  %y = add %x, %j : i64
  %z = add %y, %a : i64
  ret %z
}

func @calls(i64) -> i64 {
entry ^0(%n: i64):
  %keep = alloca i64 : ptr
  store i64 77, %keep align 8 : i64
  %p = dyn_alloca %n align 16 : ptr
  store %n, %p align 8 : i64
  %r = call @nine(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, %n, i64 9) : i64
  %q = dyn_alloca i64 64 align 16 : ptr
  store %r, %q align 8 : i64
  %r2 = call @nine(%r, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 3, i64 4) : i64
  %v = load %p align 8 : i64
  %w = load %q align 8 : i64
  %k = load %keep align 8 : i64
  %s = add %v, %w : i64
  %s2 = add %s, %r2 : i64
  %s3 = add %s2, %k : i64
  ret %s3
}

func @looped(i64) -> i64 {
entry ^0(%n: i64):
  br ^1(i64 0, i64 0, ptr null)
^1(%i: i64, %s: i64, %prev: ptr):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %p = dyn_alloca i64 16 align 16 : ptr
  store %i, %p align 8 : i64
  %q = ptr_add %p, i64 8 : ptr
  store %prev, %q align 8 : ptr
  %j = add %i, i64 1 : i64
  br ^1(%j, %s, %p)
^3:
  br ^4(%prev, i64 0)
^4(%at: ptr, %acc: i64):
  %z = icmp eq %at, ptr null : i1
  cond_br %z, ^6, ^5
^5:
  %v = load %at align 8 : i64
  %acc2 = add %acc, %v : i64
  %nx = ptr_add %at, i64 8 : ptr
  %next = load %nx align 8 : ptr
  br ^4(%next, %acc2)
^6:
  ret %acc
}

func @rec(i64) -> i64 {
entry ^0(%n: i64):
  %sz = mul %n, i64 8 : i64
  %sz1 = add %sz, i64 8 : i64
  %p = dyn_alloca %sz1 align 16 : ptr
  store %n, %p align 8 : i64
  %z = icmp eq %n, i64 0 : i1
  cond_br %z, ^1, ^2
^1:
  ret i64 0
^2:
  %m = sub %n, i64 1 : i64
  %r = call @rec(%m) : i64
  %v = load %p align 8 : i64
  %s = add %r, %v : i64
  ret %s
}

func @aligned(i64) -> i64 {
entry ^0(%n: i64):
  %p = dyn_alloca %n align 256 : ptr
  store %n, %p align 8 : i64
  %i = ptrtoint %p : i64
  %m = and %i, i64 255 : i64
  %v = load %p align 8 : i64
  %r = add %m, %v : i64
  ret %r
}

func @fp_spill(f64, i64) -> f64 {
entry ^0(%x: f64, %n: i64):
  %p = dyn_alloca %n align 16 : ptr
  store %x, %p align 8 : f64
  %y = call @fid(%x) : f64
  %z = load %p align 8 : f64
  %r = fadd %y, %z : f64
  ret %r
}

func @fid(f64) -> f64 {
entry ^0(%x: f64):
  %r = fmul %x, f64 0x4000000000000000 : f64
  ret %r
}
"#;

#[test]
fn dyn_alloca_programs_match_the_reference() {
    for probes in [true, false] {
        let h = Harness::with_options(DYN, &CodegenOptions::default().with_stack_probes(probes));
        let mut n = 0;
        for k in [0u64, 1, 7, 16, 100, 4095, 4096, 9000] {
            n += h.check("fill", &[vec![k]]);
            n += h.check("calls", &[vec![k.max(8)]]);
            n += h.check("aligned", &[vec![k.max(8)]]);
            n += h.check("fp_spill", &[vec![1.5f64.to_bits(), k.max(8)]]);
        }
        for k in [0u64, 1, 5, 30] {
            n += h.check("looped", &[vec![k]]);
            n += h.check("rec", &[vec![k]]);
        }
        assert_eq!(n, 40, "probes {probes}");
    }
}

/// The frame of a `dyn_alloca` function: `s0` is saved and set, the report
/// flags the dynamic allocation and counts the outgoing arguments.
#[test]
fn dyn_alloca_functions_use_a_frame_pointer() {
    let (m, syms) = parse(DYN);
    let out = super::compile_module_with(&m, &syms, &CodegenOptions::default());
    let c = out.stack.get("calls").unwrap();
    assert!(c.dynamic_alloca && !out.stack.get("nine").unwrap().dynamic_alloca);
    assert_eq!(c.outgoing_args, 16, "two stack-passed arguments, a 16-byte area");
    // `mv s0, sp` after the saves, `mv sp, s0` before the restores.
    let sym = out.object.symbols().iter().find(|s| s.name == "calls").unwrap();
    let crate::mc::object::SymbolValue::Defined { section, offset } = sym.value else { unreachable!() };
    let code = &out.object.section(section).bytes[offset as usize..(offset + sym.size) as usize];
    let words: Vec<u32> = code.chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
    assert!(words.contains(&super::encode::mv(8, 2)), "mv s0, sp");
    assert!(words.contains(&super::encode::mv(2, 8)), "mv sp, s0");
    let b = out
        .stack
        .worst_case_depth("calls", &StackAssumptions::new().dynamic("calls", 4096))
        .unwrap();
    assert_eq!(b.bytes, c.frame_size + 4096 + out.stack.get("nine").unwrap().frame_size);
}

/// A 1 MiB `dyn_alloca` written at both ends.
const BIG: &str = r#"
module "big"
func @dynamic(i64) -> i64 {
entry ^0(%n: i64):
  %p = dyn_alloca %n align 16 : ptr
  store i8 5, %p align 1 : i8
  %last = sub %n, i64 1 : i64
  %hi = ptr_add %p, %last : ptr
  store i8 6, %hi align 1 : i8
  %x = load %p align 1 : i8
  %y = load %hi align 1 : i8
  %s = add %x, %y : i8
  %r = zext %s : i64
  ret %r
}
"#;

/// Run `name(arg)` of `src` on the simulator with a stack whose guard page
/// starts `room` bytes below the entry `sp` (`None`: no guard), recording
/// every memory access.
fn run_guarded(src: &str, probes: bool, name: &str, arg: u64, room: Option<u64>) -> (Result<u64, Fault>, Vec<u64>, u64) {
    let h = Harness::with_options(src, &CodegenOptions::default().with_stack_probes(probes));
    let mut cpu = Cpu::new(&h.image);
    let sp = STACK_TOP - 4096;
    cpu.guard = room.map(|r| StackGuard { guard_lo: sp - r - STACK_PROBE_INTERVAL });
    cpu.touches = Some(Vec::new());
    let r = cpu.call(h.image.symbols[name], &[arg], &[], sp).map(|()| cpu.x[10]);
    (r, cpu.touches.take().unwrap(), sp - cpu.min_sp)
}

#[test]
fn dyn_alloca_probes_hit_the_guard_instead_of_skipping_it() {
    let mib = 1u64 << 20;
    for probes in [true, false] {
        let (r, touches, depth) = run_guarded(BIG, probes, "dynamic", mib, None);
        assert_eq!(r, Ok(11), "probes {probes}");
        assert!(depth >= mib && depth < mib + 4096, "depth {depth}");
        if probes {
            // No access lands more than one probe interval below the deepest
            // earlier one.
            let sp = STACK_TOP - 4096;
            let mut deepest = 0u64;
            for &a in touches.iter().filter(|&&a| a < sp) {
                let d = sp - a;
                assert!(d <= deepest + STACK_PROBE_INTERVAL, "an access at depth {d} skips from {deepest}");
                deepest = deepest.max(d);
            }
            assert!(depth <= deepest + STACK_PROBE_INTERVAL);
        }
    }
    // A 32 KiB stack: probed code faults on the guard page, unprobed code
    // writes below it.
    let (on, _, _) = run_guarded(BIG, true, "dynamic", mib, Some(32 * 1024));
    assert_eq!(on, Err(Fault::Guard));
    let (off, _, _) = run_guarded(BIG, false, "dynamic", mib, Some(32 * 1024));
    assert_eq!(off, Err(Fault::Skipped));
}

/// With probes on, a `dyn_alloca` of `n` bytes probes about `n / 4096`
/// pages, and the probe loop's words decode as RV64I.
#[test]
fn dyn_alloca_probe_loop_decodes() {
    let (m, syms) = parse(BIG);
    let on = super::compile_module_with(&m, &syms, &CodegenOptions::default());
    let off = super::compile_module_with(&m, &syms, &CodegenOptions::default().with_stack_probes(false));
    let len = |o: &crate::codegen::CompiledModule| o.object.sections()[0].bytes.len();
    assert_eq!(len(&on), len(&off) + 4 * 7, "touch, lui, and the five-word loop");
    let Some(dis) = super::fd_tests::objdump(&on.object.sections()[0].bytes, false) else { return };
    for needle in ["bltu", "ld\tzero, 0x0(sp)", "sd\tzero, 0x0(sp)", "lui\tt1, 0x1", "sub\tsp, sp, t1", "sub\tsp, sp, t0", "mv\ts0, sp", "mv\tsp, s0"] {
        assert!(dis.contains(needle), "`{needle}` missing:\n{dis}");
    }
}
