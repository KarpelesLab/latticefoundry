//! Stack-usage reports and stack probes on x86-64.
//!
//! Two kinds of evidence:
//!
//! - **Decoded prologues.** Each compiled function's entry bytes are decoded and
//!   simulated (`push`, `sub rsp`, the probe `or qword [rsp], 0`, and the probe
//!   loop's `mov r11d`/`dec r11`/`jnz`) to recover how far `rsp` moves and which
//!   stack addresses are touched on the way. The reported frame size must equal
//!   the decoded movement, and with probes on no gap between touches (including
//!   the caller's return-address push and the next frame's) may exceed one probe
//!   interval.
//! - **Execution** (Linux x86-64): the real stack depth of a call chain,
//!   measured with a two-instruction `rsp` reader, must equal the report's
//!   worst-case depth; large frames and a 1 MiB `dyn_alloca` must run with probes
//!   on; and an overflow must die with `SIGSEGV` at the guard, even where an
//!   unprobed frame would silently write into a mapping below the stack.

use super::encode::{compile_module, compile_module_with};
use crate::codegen::stack::{STACK_PROBE_INTERVAL, StackAssumptions, StackBound, StackReport};
use crate::codegen::{CodegenOptions, CompiledModule};
use crate::ir::Module;
use crate::mc::object::{ObjectModule, SymbolValue};
use crate::support::StrInterner;
use crate::target::TargetOs;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize};

/// Parse and verify `src`, then optimize at `level`.
fn prepare(src: &str, level: OptLevel) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse .lf: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    optimize(&mut m, level);
    (m, syms)
}

/// The bytes of the defined function `name` in `obj`'s `.text`.
fn func_bytes<'a>(obj: &'a ObjectModule, name: &str) -> &'a [u8] {
    let sym = obj.symbols().iter().find(|s| s.name == name).expect("function symbol");
    let SymbolValue::Defined { section, offset } = sym.value else { panic!("{name} undefined") };
    let bytes = &obj.section(section).bytes;
    &bytes[offset as usize..(offset + sym.size) as usize]
}

/// What the simulated prologue did, in *depths*: bytes below the caller's `rsp`
/// just before its `call` (so the return address is written at depth 8).
#[derive(Debug)]
struct Prologue {
    /// The depth of `rsp` when the prologue ends.
    depth: i64,
    /// Every stack write, in execution order.
    touches: Vec<i64>,
    /// How many `sub rsp` instructions executed.
    subs: usize,
}

/// Simulate the prologue at the start of `code` until the first instruction
/// that is not part of it.
fn simulate_prologue(code: &[u8]) -> Prologue {
    let mut depth = 8i64;
    let mut touches = vec![8i64]; // the call's return-address push
    let mut subs = 0;
    let mut r11 = 0u64;
    let mut zf = false;
    let mut pc = 0usize;
    let imm32 = |at: usize| u32::from_le_bytes(code[at..at + 4].try_into().unwrap());
    let mut steps = 0u64;
    loop {
        steps += 1;
        assert!(steps < 10_000_000, "prologue simulation does not terminate");
        let c = &code[pc..];
        if c[0] == 0x55 || (0x50..=0x57).contains(&c[0]) {
            depth += 8; // push r64
            touches.push(depth);
            pc += 1;
        } else if c[0] == 0x41 && (0x50..=0x57).contains(&c[1]) {
            depth += 8; // push r8..r15
            touches.push(depth);
            pc += 2;
        } else if c.starts_with(&[0x48, 0x89, 0xE5]) {
            pc += 3; // mov rbp, rsp
        } else if c.starts_with(&[0x48, 0x8D, 0x6C, 0x24]) {
            pc += 5; // lea rbp, [rsp + disp8] (the Windows frame)
        } else if c.starts_with(&[0x48, 0x8D, 0xAC, 0x24]) {
            pc += 8; // lea rbp, [rsp + disp32]
        } else if let Some(at) = [0usize, 1].into_iter().find(|&p| {
            (p == 0 || c[0] == 0x44) && c[p..].starts_with(&[0x0F, 0x11]) && c[p + 2] & 0xC7 != 0x05 && c[p + 2] & 7 == 5
        }) {
            // movups [rbp + disp], xmm (the Windows xmm saves): rbp is the
            // saved-rbp slot, at depth 16.
            let modrm = c[at + 2];
            let (disp, len) = if modrm >> 6 == 1 {
                (i64::from(c[at + 3] as i8), 1)
            } else {
                (i64::from(i32::from_le_bytes(c[at + 3..at + 7].try_into().unwrap())), 4)
            };
            touches.push(16 - disp - 15);
            pc += at + 3 + len;
        } else if c.starts_with(&[0x48, 0x81, 0xEC]) {
            depth += i64::from(imm32(pc + 3)); // sub rsp, imm32
            subs += 1;
            pc += 7;
        } else if c.starts_with(&[0x48, 0x83, 0xEC]) {
            depth += i64::from(c[3] as i8); // sub rsp, imm8
            subs += 1;
            pc += 4;
        } else if c.starts_with(&[0x48, 0x83, 0x0C, 0x24, 0x00]) {
            touches.push(depth); // or qword [rsp], 0
            pc += 5;
        } else if c.starts_with(&[0x41, 0xBB]) {
            r11 = u64::from(imm32(pc + 2)); // mov r11d, imm32
            pc += 6;
        } else if c.starts_with(&[0x49, 0xFF, 0xCB]) {
            r11 = r11.wrapping_sub(1); // dec r11
            zf = r11 == 0;
            pc += 3;
        } else if c[0] == 0x75 {
            let rel = i64::from(c[1] as i8); // jnz rel8
            pc += 2;
            if !zf {
                pc = (pc as i64 + rel) as usize;
            }
        } else {
            return Prologue { depth, touches, subs };
        }
    }
}

/// Assert the probe invariant over a prologue: starting from a caller whose
/// deepest touch may be up to `slack` bytes above its `rsp`, no touch lands more
/// than one interval below the deepest touch so far, and the prologue leaves
/// `rsp` at most `slack` below the deepest touch (so the next frame's first
/// push, 8 bytes further down, is again within one interval).
fn assert_probed(p: &Prologue, slack: i64, what: &str) {
    let interval = STACK_PROBE_INTERVAL as i64;
    let mut deepest = -slack;
    for &t in &p.touches {
        assert!(t - deepest <= interval, "{what}: touch at depth {t} skips from {deepest}");
        deepest = deepest.max(t);
    }
    assert!(p.depth - deepest <= slack, "{what}: rsp ends {} below the last touch", p.depth - deepest);
}

/// Functions with frames around and far beyond one page.
const FRAMES: &str = r#"
module "frames"
func @tiny() -> i64 {
entry ^0:
  ret i64 1
}
func @mid() -> i64 {
entry ^0:
  %a = alloca [3000 x i8] : ptr
  store i8 5, %a align 1 : i8
  %v = load %a align 1 : i8
  %r = zext %v : i64
  ret %r
}
func @page() -> i64 {
entry ^0:
  %a = alloca [4072 x i8] : ptr
  store i8 5, %a align 1 : i8
  %v = load %a align 1 : i8
  %r = zext %v : i64
  ret %r
}
func @three() -> i64 {
entry ^0:
  %a = alloca [13000 x i8] : ptr
  store i8 5, %a align 1 : i8
  %v = load %a align 1 : i8
  %r = zext %v : i64
  ret %r
}
func @huge() -> i64 {
entry ^0:
  %a = alloca [1048576 x i8] : ptr
  store i8 5, %a align 1 : i8
  %v = load %a align 1 : i8
  %c = call @mid() : i64
  %w = zext %v : i64
  %r = add %w, %c : i64
  ret %r
}
"#;

#[test]
fn reported_frame_equals_decoded_prologue() {
    let (m, syms) = prepare(FRAMES, OptLevel::O0);
    for (probes, os) in [true, false].into_iter().flat_map(|p| [(p, TargetOs::Linux), (p, TargetOs::Windows)]) {
        let opts = CodegenOptions::default().with_stack_probes(probes).with_os(os);
        let out: CompiledModule = compile_module_with(&m, &syms, &opts);
        assert_eq!(out.stack.functions().len(), 5);
        for u in out.stack.functions() {
            let p = simulate_prologue(func_bytes(&out.object, &u.name));
            let what = format!("{} (probes {probes}, {os:?})", u.name);
            assert_eq!(p.depth as u64, u.frame_size, "{what}: frame size vs decoded prologue");
            assert_eq!(u.frame_size, u.return_address + u.saved_registers + u.sp_adjust, "{what}");
            assert_eq!(u.return_address, 8);
            assert_eq!(u.probed, probes);
            assert!(!u.dynamic_alloca && !u.indirect_calls && !u.syscalls, "{what}");
            if probes {
                assert_probed(&p, 4088, &what);
            } else {
                // Windows splits off the fixed part allocated before rbp is set.
                let most = if os == TargetOs::Windows { 2 } else { 1 };
                assert!(p.subs <= most, "{what}: {most} `sub rsp` at most without probes");
            }
            // Every static frame keeps rsp 16-aligned at calls; a leaf
            // without locals (and without a frame pointer on System V)
            // leaves it where the call put it.
            if u.direct_callees.is_empty() && u.sp_adjust == 0 && os != TargetOs::Windows {
                assert_eq!(u.frame_size, 8 + u.saved_registers, "{what}");
            } else {
                assert_eq!(u.frame_size % 16, 0, "{what}");
            }
        }
        let huge = out.stack.get("huge").unwrap();
        assert!(huge.frame_size >= 1 << 20);
        assert_eq!(huge.direct_callees, ["mid"]);
        let b = out.stack.worst_case_depth("huge", &StackAssumptions::new()).unwrap();
        assert_eq!(b.bytes, huge.frame_size + out.stack.get("mid").unwrap().frame_size);
        assert_eq!(b.path, ["huge", "mid"]);
    }
    // The default entry point probes.
    let obj = compile_module(&m, &syms);
    let p = simulate_prologue(func_bytes(&obj, "huge"));
    assert_probed(&p, 4088, "huge (compile_module)");
    assert!(p.touches.len() > 256, "a 1 MiB frame touches every page");
}

/// Calls of every kind, for the report's call information.
const CALLS: &str = r#"
module "calls"
func @ext(i64) -> i64
func @leaf(i64) -> i64 {
entry ^0(%x: i64):
  ret %x
}
func @kinds(i64) -> i64 {
entry ^0(%x: i64):
  %a = call @leaf(%x) : i64
  %b = call @ext(%a) : i64
  %c = call @leaf(%b) : i64
  %fp = select i1 1, @leaf, @ext : ptr
  %d = call %fp(%c) : i64
  %e = syscall i64 39 : i64
  %n = and %x, i64 255 : i64
  %p = dyn_alloca %n align 16 : ptr
  store i64 1, %p align 8 : i64
  %s = add %d, %e : i64
  ret %s
}
func @rec(i64) -> i64 {
entry ^0(%x: i64):
  %c = icmp eq %x, i64 0 : i1
  cond_br %c, ^1, ^2
^1:
  ret i64 0
^2:
  %y = sub %x, i64 1 : i64
  %r = call @rec(%y) : i64
  ret %r
}
"#;

#[test]
fn report_lists_calls_syscalls_and_dynamic_allocation() {
    let (m, syms) = prepare(CALLS, OptLevel::O0);
    let out = compile_module_with(&m, &syms, &CodegenOptions::default());
    let r: &StackReport = &out.stack;
    assert!(r.get("ext").is_none(), "declarations are not reported");
    let k = r.get("kinds").unwrap();
    assert_eq!(k.direct_callees, ["leaf", "ext"]);
    assert!(k.indirect_calls && k.syscalls && k.dynamic_alloca);
    let leaf = r.get("leaf").unwrap();
    assert!(leaf.direct_callees.is_empty() && !leaf.indirect_calls && !leaf.syscalls);

    use crate::codegen::stack::StackBoundError as E;
    let none = StackAssumptions::new();
    assert_eq!(
        r.worst_case_depth("kinds", &none),
        Err(E::DynamicAlloca { function: "kinds".into() })
    );
    let a = StackAssumptions::new().dynamic("kinds", 272);
    assert_eq!(r.worst_case_depth("kinds", &a), Err(E::IndirectCall { function: "kinds".into() }));
    let a = a.indirect("kinds", 64);
    assert_eq!(
        r.worst_case_depth("kinds", &a),
        Err(E::UnknownCallee { caller: "kinds".into(), callee: "ext".into() })
    );
    let a = a.external("ext", 1000);
    let b = r.worst_case_depth("kinds", &a).unwrap();
    assert_eq!(b.bytes, k.frame_size + 272 + 1000);
    assert_eq!(b.path, ["kinds", "ext"]);
    assert_eq!(
        r.worst_case_depth("rec", &a),
        Err(E::Recursion { cycle: vec!["rec".into(), "rec".into()] })
    );
    // The table names every function.
    let table = r.to_string();
    for name in ["leaf", "kinds", "rec", StackBound::INDIRECT] {
        assert!(table.contains(name), "{table}");
    }
}

/// `dyn_alloca` with probes: the emitted code touches the current top first,
/// then loops a page at a time (the `jae` back-edge), and without probes it is
/// a bare `sub rsp, d`.
#[test]
fn dyn_alloca_probe_sequence() {
    let (m, syms) = prepare(CALLS, OptLevel::O0);
    let find = |bytes: &[u8], pat: &[u8]| bytes.windows(pat.len()).filter(|w| *w == pat).count();
    let probe = [0x48, 0x83, 0x0C, 0x24, 0x00];
    let page = [0x48, 0x81, 0xEC, 0x00, 0x10, 0x00, 0x00];
    let on = compile_module_with(&m, &syms, &CodegenOptions::default());
    let off = compile_module_with(&m, &syms, &CodegenOptions::default().with_stack_probes(false));
    let (b_on, b_off) = (func_bytes(&on.object, "kinds"), func_bytes(&off.object, "kinds"));
    assert_eq!(find(b_on, &probe), 2, "top probe + loop probe");
    assert_eq!(find(b_on, &page), 1, "one page step inside the loop");
    assert_eq!(find(b_off, &probe), 0);
    assert_eq!(find(b_off, &page), 0);
    // probe, cmp, jb (rel8), sub, probe, sub, cmp, jae (rel8)
    assert_eq!(b_on.len(), b_off.len() + 5 + 7 + 2 + 7 + 5 + 7 + 7 + 2);
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod exec {
    use super::*;
    use crate::link::{ImageOptions, link_executable, write_executable};
    use crate::mc::object::{Section, SectionKind, Symbol, SymbolBinding, SymbolType};
    use std::os::unix::process::ExitStatusExt;

    /// A hand-assembled object defining `get_sp`: `lea rax, [rsp + 8]; ret` —
    /// the caller's `rsp` just before its `call`.
    fn get_sp_object() -> ObjectModule {
        let mut obj = ObjectModule::new("get_sp");
        let mut text = Section::new(".text", SectionKind::Text, 16);
        text.bytes = vec![0x48, 0x8D, 0x44, 0x24, 0x08, 0xC3];
        let sec = obj.add_section(text);
        obj.add_symbol(Symbol::defined(
            "get_sp",
            SymbolBinding::Global,
            SymbolType::Func,
            sec,
            0,
            6,
        ));
        obj
    }

    /// Link `objs`, run the executable, and return `(stdout, status)`.
    fn run(objs: Vec<ObjectModule>, tag: &str) -> (Vec<u8>, std::process::ExitStatus) {
        let image = link_executable(objs, &ImageOptions::default()).expect("link");
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("lf_stack_{tag}_{}_{uniq}", std::process::id()));
        write_executable(path.to_str().unwrap(), &image).expect("write executable");
        let child = loop {
            match std::process::Command::new(&path).stdout(std::process::Stdio::piped()).spawn() {
                Ok(c) => break c,
                // ETXTBSY: another test's fork briefly holds the fresh file open.
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5))
                }
                Err(e) => panic!("exec: {e}"),
            }
        };
        let out = child.wait_with_output().expect("wait");
        let _ = std::fs::remove_file(&path);
        (out.stdout, out.status)
    }

    /// `main` reads its own `rsp`, calls `f → g → leaf` (f has a 5000-byte frame,
    /// so its prologue probes; g `dyn_alloca`s 100 bytes and passes 8 arguments,
    /// two on the stack; leaf reads `rsp` again), and writes the difference.
    const DEPTH: &str = r#"
module "depth"
func @get_sp() -> i64
func @leaf(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 {
entry ^0(%x: i64, %a: i64, %b: i64, %c: i64, %d: i64, %e: i64, %f: i64, %g: i64):
  %sp = call @get_sp() : i64
  %r = sub %x, %sp : i64
  ret %r
}
func @g(i64) -> i64 {
entry ^0(%x: i64):
  %n = add %x, i64 0 : i64
  %m = sub %n, %x : i64
  %sz = add %m, i64 100 : i64
  %p = dyn_alloca %sz align 16 : ptr
  store i64 7, %p align 8 : i64
  %r = call @leaf(%x, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7) : i64
  ret %r
}
func @f(i64) -> i64 {
entry ^0(%x: i64):
  %buf = alloca [5000 x i8] : ptr
  store i8 1, %buf align 1 : i8
  %r = call @g(%x) : i64
  ret %r
}
func @main() -> i64 {
entry ^0:
  %slot = alloca i64 : ptr
  %sp = call @get_sp() : i64
  %d = call @f(%sp) : i64
  store %d, %slot align 8 : i64
  %w = syscall i64 1, i64 1, %slot, i64 8 : i64
  ret i64 0
}
"#;

    #[test]
    fn measured_depth_equals_reported_bound() {
        let (m, syms) = prepare(DEPTH, OptLevel::O0);
        for probes in [true, false] {
            let out = compile_module_with(&m, &syms, &CodegenOptions::default().with_stack_probes(probes));
            let r = &out.stack;
            assert!(r.get("g").unwrap().dynamic_alloca);
            assert_eq!(r.get("g").unwrap().outgoing_args, 16, "two stack-passed arguments");
            // g carves roundup16(100 + 15 + outgoing) = 128 bytes for its dyn_alloca.
            let a = StackAssumptions::new().external("get_sp", 8).dynamic("g", 128);
            let bound = r.worst_case_depth("f", &a).unwrap();
            assert_eq!(bound.path, ["f", "g", "leaf", "get_sp"]);

            let (stdout, status) = run(vec![out.object, get_sp_object()], "depth");
            assert_eq!(status.code(), Some(0), "probes {probes}");
            let measured = u64::from_le_bytes(stdout[..8].try_into().unwrap());
            // measured = main's rsp - leaf's rsp; the bound adds get_sp's return address.
            assert_eq!(measured + 8, bound.bytes, "probes {probes}: measured vs reported");
            let frames: u64 = ["f", "g", "leaf"].iter().map(|n| r.get(n).unwrap().frame_size).sum();
            assert_eq!(measured, frames + 128);
        }
    }

    /// A 1 MiB frame touched at both ends, and a 1 MiB `dyn_alloca` written at
    /// both ends; the exit status sums what was read back (3 + 4 + 5 + 6).
    const BIG: &str = r#"
module "big"
func @frame() -> i64 {
entry ^0:
  %a = alloca [1048576 x i8] : ptr
  store i8 3, %a align 1 : i8
  %hi = ptr_add %a, i64 1048575 : ptr
  store i8 4, %hi align 1 : i8
  %x = load %a align 1 : i8
  %y = load %hi align 1 : i8
  %s = add %x, %y : i8
  %r = zext %s : i64
  ret %r
}
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
func @main() -> i64 {
entry ^0:
  %a = call @frame() : i64
  %b = call @dynamic(i64 1048576) : i64
  %r = add %a, %b : i64
  ret %r
}
"#;

    #[test]
    fn large_frames_run_with_probes() {
        for level in [OptLevel::O0, OptLevel::O2] {
            let (m, syms) = prepare(BIG, level);
            for probes in [true, false] {
                let opts = CodegenOptions::default().with_stack_probes(probes);
                let out = compile_module_with(&m, &syms, &opts);
                let (_, status) = run(vec![out.object], "big");
                assert_eq!(status.code(), Some(18), "{level:?}, probes {probes}");
            }
        }
    }

    /// Lower `RLIMIT_STACK`'s soft limit to 256 KiB (`getrlimit`/`setrlimit`),
    /// then call `@victim`, which needs 4 MiB of stack. Exit 0 would mean the
    /// overflow went unnoticed; a probed program must die with `SIGSEGV`.
    fn overflow_program(victim: &str) -> String {
        format!(
            r#"
module "overflow"
{victim}
func @main() -> i64 {{
entry ^0:
  %rl = alloca [2 x i64] : ptr
  %g = syscall i64 97, i64 3, %rl : i64
  store i64 262144, %rl align 8 : i64
  %s = syscall i64 160, i64 3, %rl : i64
  %ok = icmp eq %s, i64 0 : i1
  cond_br %ok, ^1, ^2
^1:
  %r = call @victim(i64 4194304) : i64
  ret i64 0
^2:
  ret i64 97
}}
"#
        )
    }

    #[test]
    fn overflow_is_a_deterministic_sigsegv() {
        let frame = r#"
func @victim(i64) -> i64 {
entry ^0(%n: i64):
  %a = alloca [4194304 x i8] : ptr
  store i8 1, %a align 1 : i8
  %v = load %a align 1 : i8
  %r = zext %v : i64
  ret %r
}"#;
        let dynamic = r#"
func @victim(i64) -> i64 {
entry ^0(%n: i64):
  %p = dyn_alloca %n align 16 : ptr
  store i8 1, %p align 1 : i8
  %v = load %p align 1 : i8
  %r = zext %v : i64
  ret %r
}"#;
        for (tag, victim) in [("frame", frame), ("dyn", dynamic)] {
            let (m, syms) = prepare(&overflow_program(victim), OptLevel::O0);
            let obj = compile_module(&m, &syms); // probes on by default
            let (_, status) = run(vec![obj], tag);
            assert_eq!(status.signal(), Some(11), "{tag}: expected SIGSEGV, got {status:?}");
        }
    }

    /// The hazard probes exist for: `main` maps 2 MiB at 2–4 MiB below its stack
    /// (`MAP_FIXED_NOREPLACE`), then calls `@big`, whose 3 MiB frame's low end
    /// lands inside that mapping. Unprobed, `big` writes straight into the
    /// mapping and returns normally (exit 0: memory below the stack silently
    /// modified). Probed, `big` walks down page by page and faults in the
    /// kernel's stack guard gap above the mapping. Exit 98/99: the mapping could
    /// not be placed (skipped).
    const SKIP: &str = r#"
module "skip"
func @big() -> i64 {
entry ^0:
  %a = alloca [3145728 x i8] : ptr
  store i8 77, %a align 1 : i8
  %r = ptrtoint %a : i64
  ret %r
}
func @main() -> i64 {
entry ^0:
  %loc = alloca i64 : ptr
  %top = ptrtoint %loc : i64
  %low = sub %top, i64 4194304 : i64
  %want = and %low, i64 -4096 : i64
  %m = syscall i64 9, %want, i64 2097152, i64 3, i64 1048610, i64 -1, i64 0 : i64
  %placed = icmp eq %m, %want : i1
  cond_br %placed, ^1, ^3
^1:
  %p = call @big() : i64
  %lo = icmp uge %p, %m : i1
  %end = add %m, i64 2097152 : i64
  %hi = icmp ult %p, %end : i1
  %in = and %lo, %hi : i1
  cond_br %in, ^2, ^4
^2:
  %ptr = inttoptr %p : ptr
  %v = load %ptr align 1 : i8
  %c = icmp eq %v, i8 77 : i1
  %r = select %c, i64 0, i64 1 : i64
  ret %r
^3:
  ret i64 99
^4:
  ret i64 98
}
"#;

    #[test]
    fn probes_stop_a_frame_from_jumping_the_guard() {
        let (m, syms) = prepare(SKIP, OptLevel::O0);
        let off = compile_module_with(&m, &syms, &CodegenOptions::default().with_stack_probes(false));
        let (_, status) = run(vec![off.object], "skip_off");
        if matches!(status.code(), Some(98 | 99)) {
            eprintln!("note: could not place a mapping below the stack; skipping");
            return;
        }
        assert_eq!(status.code(), Some(0), "unprobed: the frame silently lands in the mapping");
        let on = compile_module_with(&m, &syms, &CodegenOptions::default());
        let (_, status) = run(vec![on.object], "skip_on");
        assert_eq!(status.signal(), Some(11), "probed: SIGSEGV at the guard, got {status:?}");
    }
}
