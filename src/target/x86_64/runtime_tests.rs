//! Execution tests for the x86-64 green-thread runtime ([`super::runtime`]).
//!
//! Each program is `.lf` text compiled by our backend, linked by our own static
//! linker together with the runtime object (and, where a test must control
//! every register, a small hand-assembled harness object), and run on the bare
//! kernel — no libc anywhere.

use std::os::unix::process::ExitStatusExt;

use crate::ir::Module;
use crate::link::{ImageOptions, link_executable, write_executable};
use crate::mc::object::{ObjectModule, RelocKind, SectionKind, Symbol, SymbolBinding, SymbolType};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize};

use super::regs::{RAX, RDI, RSI, RSP};
use super::runtime::{self, X86Asm, layout};

/// The runtime entry points, as `.lf` declarations.
const DECLS: &str = r#"
func @lf_ctx_save(ptr) -> i64
func @lf_ctx_save_full(ptr) -> i64
func @lf_ctx_restore(ptr) -> void
func @lf_ctx_switch(ptr, ptr) -> void
func @lf_ctx_switch_full(ptr, ptr) -> void
func @lf_ctx_init(ptr, ptr, ptr, i64) -> void
func @lf_ctx_preempt(ptr, ptr, ptr) -> void
func @lf_ctx_uc_in_runtime(ptr) -> i64
func @lf_sig_install(i64, ptr, i64) -> i64
"#;

fn prepare(src: &str, level: OptLevel) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let src = format!("{src}\n{DECLS}");
    let mut m = crate::ir::text::parse_module(&src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse .lf: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    optimize(&mut m, level);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify after {level:?}: {e:?}"));
    (m, syms)
}

/// Link `objects` + the runtime, run the executable, return `(stdout, status)`.
fn link_and_run(mut objects: Vec<ObjectModule>, tag: &str) -> (Vec<u8>, std::process::ExitStatus) {
    objects.push(runtime::context_runtime_object());
    let image = link_executable(objects, &ImageOptions::default()).expect("link should succeed");
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("lf_rt_{tag}_{}_{uniq}", std::process::id()));
    write_executable(path.to_str().unwrap(), &image).expect("write executable");
    let child = loop {
        match std::process::Command::new(&path).stdout(std::process::Stdio::piped()).spawn() {
            Ok(c) => break c,
            Err(e) if e.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(e) => panic!("exec our native binary: {e}"),
        }
    };
    let out = child.wait_with_output().expect("wait for child");
    let _ = std::fs::remove_file(&path);
    (out.stdout, out.status)
}

fn run_lf(src: &str, level: OptLevel, extra: Vec<ObjectModule>, tag: &str) -> (Vec<u8>, std::process::ExitStatus) {
    let (m, syms) = prepare(src, level);
    let mut objs = vec![super::compile_module(&m, &syms)];
    objs.extend(extra);
    link_and_run(objs, tag)
}

// ---------------------------------------------------------------------------
// Harness assembly helpers (RIP-relative references to IR globals)
// ---------------------------------------------------------------------------

/// A hand-assembled object of named functions.
struct Harness {
    a: X86Asm,
    funcs: Vec<(&'static str, u64)>,
}

impl Harness {
    fn new() -> Harness {
        Harness { a: X86Asm::new(), funcs: Vec::new() }
    }
    fn func(&mut self, name: &'static str) {
        while !self.a.e.offset().is_multiple_of(16) {
            self.a.e.u8(0xCC);
        }
        self.funcs.push((name, self.a.e.offset()));
    }
    fn finish(self) -> ObjectModule {
        let mut obj = ObjectModule::new("harness");
        let total = self.a.e.offset();
        let emitted = self.a.e.finish().expect("harness assembles");
        let sec = obj.add_emitted_section(".text", SectionKind::Text, 16, emitted);
        for (i, &(name, start)) in self.funcs.iter().enumerate() {
            let end = self.funcs.get(i + 1).map_or(total, |f| f.1);
            obj.add_symbol(Symbol::defined(name, SymbolBinding::Global, SymbolType::Func, sec, start, end - start));
        }
        obj
    }
    #[allow(clippy::too_many_arguments)]
    /// `REX(W, R) opcode.. modrm(00, reg, rip)` + a PC32 field against `sym+off`;
    /// `tail` immediate bytes follow the displacement.
    fn rip(&mut self, prefix: Option<u8>, w: bool, opcode: &[u8], reg: u16, sym: &str, off: i64, tail: usize) {
        let e = &mut self.a.e;
        if let Some(p) = prefix {
            e.u8(p);
        }
        if w || reg >= 8 {
            e.u8(0x40 | ((w as u8) << 3) | (((reg >= 8) as u8) << 2));
        }
        e.bytes(opcode);
        e.u8(((reg as u8 & 7) << 3) | 0b101);
        e.reference_symbol(RelocKind::Pc32, sym, off - tail as i64);
    }
    /// `lea r64, [rip + sym + off]`.
    fn lea_sym(&mut self, dst: u16, sym: &str, off: i64) {
        self.rip(None, true, &[0x8D], dst, sym, off, 0);
    }
    /// `call sym` (`E8 rel32`, PLT32 relocation).
    fn call_sym(&mut self, sym: &str) {
        self.a.e.u8(0xE8);
        self.a.e.reference_symbol(RelocKind::Plt32, sym, 0);
    }
    /// `mov [rip + sym + off], r64`.
    fn store_sym(&mut self, sym: &str, off: i64, r: u16) {
        self.rip(None, true, &[0x89], r, sym, off, 0);
    }
    /// `mov qword [rip + sym + off], simm32`.
    fn store_sym_imm(&mut self, sym: &str, off: i64, imm: i32) {
        self.rip(None, true, &[0xC7], 0, sym, off, 4);
        self.a.e.u32(imm as u32);
    }
    /// `cmp qword [rip + sym + off], simm8`.
    fn cmp_sym_imm8(&mut self, sym: &str, off: i64, imm: i8) {
        self.rip(None, true, &[0x83], 7, sym, off, 1);
        self.a.e.u8(imm as u8);
    }
    /// `pop qword [rip + sym + off]`.
    fn pop_sym(&mut self, sym: &str, off: i64) {
        self.rip(None, false, &[0x8F], 0, sym, off, 0);
    }
    /// `movdqu xmm, [rip + sym + off]`.
    fn movdqu_load(&mut self, x: u16, sym: &str, off: i64) {
        self.rip(Some(0xF3), false, &[0x0F, 0x6F], x, sym, off, 0);
    }
    /// `movdqu [rip + sym + off], xmm`.
    fn movdqu_store(&mut self, sym: &str, off: i64, x: u16) {
        self.rip(Some(0xF3), false, &[0x0F, 0x7F], x, sym, off, 0);
    }
    /// `pcmpeqd xmm, xmm` (all ones).
    fn ones(&mut self, x: u16) {
        let e = &mut self.a.e;
        e.u8(0x66);
        if x >= 8 {
            e.u8(0x45);
        }
        e.bytes(&[0x0F, 0x76]);
        e.u8(0xC0 | ((x as u8 & 7) << 3) | (x as u8 & 7));
    }
    /// `push simm32; popfq`.
    fn set_flags(&mut self, v: u32) {
        self.a.e.u8(0x68);
        self.a.e.u32(v);
        self.a.popfq();
    }
    /// Dump every GPR (slot n = register n), rflags (slot 16) and xmm0..15
    /// (slots 17.. as lo/hi pairs) into `@dump`. Leaves every register intact.
    fn dump_all(&mut self) {
        self.a.pushfq();
        self.pop_sym("dump", 16 * 8);
        for r in 0..16u16 {
            self.store_sym("dump", 8 * i64::from(r), r);
        }
        for x in 0..16u16 {
            self.movdqu_store("dump", 17 * 8 + 16 * i64::from(x), x);
        }
    }
    /// Load every GPR except `rsp` and `skip` with its pattern and every xmm from
    /// `@xpat`.
    fn load_patterns(&mut self, skip: &[u16]) {
        for x in 0..16u16 {
            self.movdqu_load(x, "xpat", 16 * i64::from(x));
        }
        for r in 0..16u16 {
            if r == RSP || skip.contains(&r) {
                continue;
            }
            self.a.mov_imm(r, gpr_pattern(r));
        }
    }
    /// Overwrite every GPR but `rsp` and every xmm with junk, and the flags.
    fn clobber_all(&mut self) {
        for r in 0..16u16 {
            if r != RSP {
                self.a.mov_imm(r, 0xDEAD_BEEF_0000_0000 | u64::from(r));
            }
        }
        for x in 0..16u16 {
            self.ones(x);
        }
        self.set_flags(0x202);
    }
    fn push_callee_saved(&mut self) {
        for r in [3u16, 5, 12, 13, 14, 15] {
            self.a.push(r);
        }
        self.a.alu_imm(5, RSP, 8);
    }
    fn pop_callee_saved(&mut self) {
        self.a.alu_imm(0, RSP, 8);
        for r in [15u16, 14, 13, 12, 5, 3] {
            self.a.pop(r);
        }
    }
}

/// The distinctive value register `r` is loaded with.
fn gpr_pattern(r: u16) -> u64 {
    0x0123_4567_89AB_CDEFu64.rotate_left(4 * u32::from(r)) ^ (u64::from(r) << 56) ^ 0x5A
}

/// [`gpr_pattern`] of every register, indexed by number.
fn expected_gprs() -> Vec<u64> {
    (0..16u16).map(gpr_pattern).collect()
}

/// The distinctive values of `xmm0..xmm15` as 32 little-endian qwords.
fn xmm_patterns() -> Vec<u64> {
    (0..32u64).map(|i| 0xF00D_0000_0000_0000 ^ (i * 0x0001_0203_0405_0607) ^ (i << 40)).collect()
}

/// `@xpat` as a `.lf` global holding [`xmm_patterns`].
fn xpat_global() -> String {
    let vals: Vec<String> = xmm_patterns().iter().map(|v| format!("i64 {}", *v as i64)).collect();
    format!("global constant @xpat : [32 x i64] = [32 x i64] ({})\n", vals.join(", "))
}

/// Parse `@dump` (49 qwords) from the program's stdout.
fn parse_dump(out: &[u8]) -> Vec<u64> {
    assert!(out.len() >= 49 * 8, "dump too short: {} bytes", out.len());
    out[..49 * 8].chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// The checksum the programs fold over the dump (every GPR but `rsp` and those
/// in `skip`, then every xmm qword): `h = h * 31 + v`, wrapping.
fn checksum(regs: &[u64], xmm: &[u64], skip: &[u16]) -> u64 {
    let mut h = 0u64;
    for r in 0..16u16 {
        if r == RSP || skip.contains(&r) {
            continue;
        }
        h = h.wrapping_mul(31).wrapping_add(regs[r as usize]);
    }
    for &v in xmm {
        h = h.wrapping_mul(31).wrapping_add(v);
    }
    h
}

/// `.lf` for the same checksum over `@dump`, skipping slots listed in `skip`;
/// writes the dump then the checksum to stdout.
fn checksum_lf(skip: &[u16]) -> String {
    let mut s = String::from("func @emit_dump() -> void {\nentry ^0:\n  %h0 = add i64 0, i64 0 : i64\n");
    let mut h = 0;
    let mut slots: Vec<i64> = (0..16).filter(|&r| r != i64::from(RSP) && !skip.contains(&(r as u16))).collect();
    slots.extend(17..49);
    for slot in slots {
        s.push_str(&format!(
            "  %p{slot} = ptr_add @dump, i64 {} : ptr\n  %v{slot} = load %p{slot} align 8 : i64\n  %m{slot} = mul %h{h}, i64 31 : i64\n  %h{} = add %m{slot}, %v{slot} : i64\n",
            slot * 8,
            h + 1
        ));
        h += 1;
    }
    s.push_str(&format!(
        "  store %h{h}, @sum align 8 : i64\n  %w = syscall i64 1, i64 1, @dump, i64 400 : i64\n  ret\n}}\n"
    ));
    s
}

// ---------------------------------------------------------------------------
// The object itself
// ---------------------------------------------------------------------------

#[test]
fn runtime_object_defines_every_entry_point() {
    let obj = runtime::context_runtime_object();
    for name in [
        runtime::SYM_SAVE,
        runtime::SYM_SAVE_FULL,
        runtime::SYM_RESTORE,
        runtime::SYM_SWITCH,
        runtime::SYM_SWITCH_FULL,
        runtime::SYM_INIT,
        runtime::SYM_FROM_UCONTEXT,
        runtime::SYM_TO_UCONTEXT,
        runtime::SYM_PREEMPT,
        runtime::SYM_UC_IN_RUNTIME,
        runtime::SYM_SIG_INSTALL,
        runtime::SYM_SIG_RESTORER,
    ] {
        let id = obj.symbol_id(name).unwrap_or_else(|| panic!("{name} defined"));
        let s = obj.symbol(id);
        assert!(!s.is_undefined(), "{name} defined");
        assert_eq!(s.binding, SymbolBinding::Global);
        assert!(s.size > 0, "{name} has a size");
    }
    // Self-contained: no relocations against anything outside the runtime.
    assert!(obj.relocations().is_empty(), "{:?}", obj.relocations());
    assert_eq!(layout::SIZE, 672);
    assert_eq!(layout::FXSAVE % 16, 0);
    assert_eq!(layout::gpr(RSP), layout::RSP);
}

/// The whole runtime disassembles cleanly with `objdump` and uses the
/// instructions the design relies on.
#[test]
fn runtime_disassembles_with_objdump() {
    let elf = crate::mc::elf::write(&runtime::context_runtime_object());
    let path = std::env::temp_dir().join(format!("lf_rt_obj_{}.o", std::process::id()));
    std::fs::write(&path, &elf).unwrap();
    let out = std::process::Command::new("objdump").arg("-d").arg(&path).output();
    let _ = std::fs::remove_file(&path);
    let Ok(out) = out else {
        return; // objdump not installed
    };
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "objdump failed");
    assert!(!text.contains("(bad)"), "undecodable bytes:\n{text}");
    for needle in ["fxsave64", "fxrstor64", "popf", "ret    $0x80", "stmxcsr", "ldmxcsr", "fnstcw", "fldcw", "syscall", "rep stos", "rep movs"] {
        assert!(text.contains(needle), "missing {needle}:\n{text}");
    }
}

// ---------------------------------------------------------------------------
// Cooperative switching
// ---------------------------------------------------------------------------

/// Two green threads — `main` and `B` on an `mmap`ed stack — take turns
/// appending to a buffer, switching with `lf_ctx_switch`. Each keeps its own
/// loop counter live across the switches.
const PING_PONG: &str = r#"
module "pingpong"
global @ctx_main : [42 x i128] = [42 x i128] poison
global @ctx_b : [42 x i128] = [42 x i128] poison
global @out : [64 x i8] = [64 x i8] poison
global @pos : i64 = i64 0

func @put(i64) -> void {
entry ^0(%v: i64):
  %c = trunc %v : i8
  %p = load @pos align 8 : i64
  %a = ptr_add @out, %p : ptr
  store %c, %a align 1 : i8
  %p1 = add %p, i64 1 : i64
  store %p1, @pos align 8 : i64
  ret
}

func @thread_b(i64) -> i64 {
entry ^0(%arg: i64):
  br ^1(i64 0)
^1(%i: i64):
  %v = add %arg, %i : i64
  call @put(%v) : void
  call @lf_ctx_switch(@ctx_b, @ctx_main) : void
  %i1 = add %i, i64 1 : i64
  br ^1(%i1)
}

func @main() -> i64 {
entry ^0:
  %stk = syscall i64 9, i64 0, i64 65536, i64 3, i64 34, i64 -1, i64 0 : i64
  %sp = inttoptr %stk : ptr
  %top = ptr_add %sp, i64 65536 : ptr
  call @lf_ctx_init(@ctx_b, %top, @thread_b, i64 66) : void
  br ^1(i64 0)
^1(%i: i64):
  %v = add %i, i64 97 : i64
  call @put(%v) : void
  call @lf_ctx_switch(@ctx_main, @ctx_b) : void
  %i1 = add %i, i64 1 : i64
  %d = icmp eq %i1, i64 5 : i1
  cond_br %d, ^2, ^1(%i1)
^2:
  %n = load @pos align 8 : i64
  %w = syscall i64 1, i64 1, @out, %n : i64
  ret i64 0
}
"#;

#[test]
fn cooperative_ping_pong_between_two_green_threads() {
    for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let (out, status) = run_lf(PING_PONG, level, vec![], "pingpong");
        assert_eq!(String::from_utf8_lossy(&out), "aBbCcDdEeF", "at {level:?}");
        assert_eq!(status.code(), Some(0), "at {level:?}: {status:?}");
    }
}

/// A fresh thread whose entry returns ends the process with its return value.
#[test]
fn returning_thread_entry_exits_with_its_value() {
    let src = r#"
module "ret"
global @ctx_main : [42 x i128] = [42 x i128] poison
global @ctx_b : [42 x i128] = [42 x i128] poison
func @thread_b(i64) -> i64 {
entry ^0(%arg: i64):
  %r = add %arg, i64 1 : i64
  ret %r
}
func @main() -> i64 {
entry ^0:
  %stk = syscall i64 9, i64 0, i64 16384, i64 3, i64 34, i64 -1, i64 0 : i64
  %sp = inttoptr %stk : ptr
  %top = ptr_add %sp, i64 16384 : ptr
  call @lf_ctx_init(@ctx_b, %top, @thread_b, i64 41) : void
  call @lf_ctx_switch(@ctx_main, @ctx_b) : void
  ret i64 1
}
"#;
    let (_, status) = run_lf(src, OptLevel::O2, vec![], "ret");
    assert_eq!(status.code(), Some(42), "{status:?}");
}

/// `lf_ctx_save` / `lf_ctx_save_full` return twice: 0, then 1 after each
/// `lf_ctx_restore`. The program resumes the saved context twice; the exit
/// status is `10 * saves + sum(results)` = 32.
fn save_restore_src(save: &str) -> String {
    format!(
        r#"
module "setjmp"
global @ctx : [42 x i128] = [42 x i128] poison
global @count : i64 = i64 0
global @sum : i64 = i64 0
func @main() -> i64 {{
entry ^0:
  %r = call @{save}(@ctx) : i64
  %s = load volatile @sum align 8 : i64
  %s1 = add %s, %r : i64
  store volatile %s1, @sum align 8 : i64
  %c = load volatile @count align 8 : i64
  %c1 = add %c, i64 1 : i64
  store volatile %c1, @count align 8 : i64
  %again = icmp slt %c1, i64 3 : i1
  cond_br %again, ^1, ^2
^1:
  call @lf_ctx_restore(@ctx) : void
  unreachable
^2:
  %t = mul %c1, i64 10 : i64
  %s2 = load volatile @sum align 8 : i64
  %e = add %t, %s2 : i64
  ret %e
}}
"#
    )
}

#[test]
fn save_returns_twice_and_restore_resumes() {
    for save in ["lf_ctx_save", "lf_ctx_save_full"] {
        for level in [OptLevel::O0, OptLevel::O2] {
            let (_, status) = run_lf(&save_restore_src(save), level, vec![], "setjmp");
            assert_eq!(status.code(), Some(32), "{save} at {level:?}: {status:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Full register preservation across a switch
// ---------------------------------------------------------------------------

/// `reg_test_a` loads every GPR, the flags and every xmm with a distinctive
/// value, `lf_ctx_switch_full`es to thread B (`reg_test_b`, which clobbers
/// everything and switches back cooperatively), then dumps every register.
fn full_switch_harness() -> ObjectModule {
    let mut h = Harness::new();
    h.func("reg_test_a");
    h.push_callee_saved();
    h.load_patterns(&[RDI, RSI]);
    h.lea_sym(RDI, "ctx_a", 0);
    h.lea_sym(RSI, "ctx_b", 0);
    h.set_flags(0xAD7); // CF PF AF ZF SF OF + IF + bit 1
    h.call_sym(runtime::SYM_SWITCH_FULL);
    h.dump_all();
    h.pop_callee_saved();
    h.a.ret();

    h.func("reg_test_b");
    h.clobber_all();
    h.lea_sym(RDI, "ctx_b", 0);
    h.lea_sym(RSI, "ctx_a", 0);
    h.call_sym(runtime::SYM_SWITCH);
    h.a.ud2();
    h.finish()
}

fn full_switch_src() -> String {
    format!(
        r#"
module "fullswitch"
global @ctx_a : [42 x i128] = [42 x i128] poison
global @ctx_b : [42 x i128] = [42 x i128] poison
global @dump : [50 x i64] = [50 x i64] poison
global @sum : i64 = i64 0
{xpat}
func @reg_test_a() -> void
func @reg_test_b(i64) -> i64
{sum_fn}
func @main() -> i64 {{
entry ^0:
  %stk = syscall i64 9, i64 0, i64 65536, i64 3, i64 34, i64 -1, i64 0 : i64
  %sp = inttoptr %stk : ptr
  %top = ptr_add %sp, i64 65536 : ptr
  call @lf_ctx_init(@ctx_b, %top, @reg_test_b, i64 0) : void
  call @reg_test_a() : void
  call @emit_dump() : void
  %s = load @sum align 8 : i64
  %p = ptr_add @dump, i64 392 : ptr
  store %s, %p align 8 : i64
  %w = syscall i64 1, i64 1, %p, i64 8 : i64
  ret i64 0
}}
"#,
        xpat = xpat_global(),
        sum_fn = checksum_lf(&[RDI, RSI]),
    )
}

#[test]
fn full_switch_preserves_every_register_and_xmm() {
    let (out, status) = run_lf(&full_switch_src(), OptLevel::O2, vec![full_switch_harness()], "fullsw");
    assert_eq!(status.code(), Some(0), "{status:?}");
    assert_eq!(out.len(), 50 * 8 + 8, "dump + checksum");
    let dump = parse_dump(&out);
    for r in 0..16u16 {
        if r == RSP || r == RDI || r == RSI {
            continue;
        }
        assert_eq!(dump[r as usize], gpr_pattern(r), "gpr {r}");
    }
    // rdi/rsi held the context addresses at the call and must too after it.
    assert_ne!(dump[RDI as usize], dump[RSI as usize]);
    assert_eq!(dump[RSI as usize] - dump[RDI as usize], layout::SIZE as u64, "ctx_b follows ctx_a");
    assert_eq!(dump[16] & 0xCD5, 0xAD7 & 0xCD5, "arithmetic flags restored: {:#x}", dump[16]);
    assert_eq!(&dump[17..49], &xmm_patterns()[..], "xmm0..15");
    let sum = u64::from_le_bytes(out[400..408].try_into().unwrap());
    assert_eq!(sum, checksum(&expected_gprs(), &xmm_patterns(), &[RDI, RSI]));
}

// ---------------------------------------------------------------------------
// Preemption from a timer signal
// ---------------------------------------------------------------------------

/// `busy_a` loads every register with its pattern, arms the handler, and spins
/// (counting in `rax`) until thread B has run; then dumps every register. B
/// can only run if the `SIGALRM` handler preempts A by rewriting the
/// `ucontext`; B clobbers everything and switches back with the cooperative
/// `lf_ctx_switch`, which must resume A's *full* context mid-loop.
fn preempt_harness() -> ObjectModule {
    let mut h = Harness::new();
    h.func("busy_a");
    h.push_callee_saved();
    h.load_patterns(&[RAX]);
    h.a.zero(RAX);
    h.store_sym_imm("armed", 0, 1);
    let top = h.a.e.create_label();
    let out = h.a.e.create_label();
    h.a.e.bind_label(top);
    h.a.alu_imm(0, RAX, 1); // add rax, 1
    h.a.alu_imm(7, RAX, 0x7FFF_FFFF); // cmp rax, LIMIT
    h.a.jcc(3, out); // jae: give up (the test then fails on `b_done`)
    h.cmp_sym_imm8("b_done", 0, 0);
    h.a.jcc(4, top); // je
    h.a.e.bind_label(out);
    h.dump_all();
    h.pop_callee_saved();
    h.a.ret();

    h.func("clobber_and_switch");
    h.clobber_all();
    h.lea_sym(RDI, "ctx_b", 0);
    h.lea_sym(RSI, "ctx_a", 0);
    h.call_sym(runtime::SYM_SWITCH);
    h.a.ud2();
    h.finish()
}

fn preempt_src() -> String {
    format!(
        r#"
module "preempt"
global @ctx_a : [42 x i128] = [42 x i128] poison
global @ctx_b : [42 x i128] = [42 x i128] poison
global @dump : [50 x i64] = [50 x i64] poison
global @sum : i64 = i64 0
global @armed : i64 = i64 0
global @switched : i64 = i64 0
global @b_done : i64 = i64 0
global @ticks : i64 = i64 0
{xpat}
func @busy_a() -> void
func @clobber_and_switch() -> void
{sum_fn}

func @on_alarm(i32, ptr, ptr) -> void {{
entry ^0(%sig: i32, %info: ptr, %uc: ptr):
  %t = load volatile @ticks align 8 : i64
  %t1 = add %t, i64 1 : i64
  store volatile %t1, @ticks align 8 : i64
  %armed = load volatile @armed align 8 : i64
  %sw = load volatile @switched align 8 : i64
  %a = icmp eq %armed, i64 1 : i1
  %s = icmp eq %sw, i64 0 : i1
  %go = and %a, %s : i1
  cond_br %go, ^1, ^3
^1:
  %inrt = call @lf_ctx_uc_in_runtime(%uc) : i64
  %ok = icmp eq %inrt, i64 0 : i1
  cond_br %ok, ^2, ^3
^2:
  store volatile i64 1, @switched align 8 : i64
  call @lf_ctx_preempt(%uc, @ctx_a, @ctx_b) : void
  ret
^3:
  ret
}}

func @set_timer(i64) -> i64 {{
entry ^0(%usec: i64):
  %tv = alloca [4 x i64] : ptr
  store i64 0, %tv align 8 : i64
  %p1 = ptr_add %tv, i64 8 : ptr
  store %usec, %p1 align 8 : i64
  %p2 = ptr_add %tv, i64 16 : ptr
  store i64 0, %p2 align 8 : i64
  %p3 = ptr_add %tv, i64 24 : ptr
  store %usec, %p3 align 8 : i64
  %r = syscall i64 38, i64 0, %tv, i64 0 : i64
  ret %r
}}

func @thread_b(i64) -> i64 {{
entry ^0(%arg: i64):
  %r = call @set_timer(i64 0) : i64
  store volatile i64 1, @b_done align 8 : i64
  call @clobber_and_switch() : void
  unreachable
}}

func @main() -> i64 {{
entry ^0:
  %stk = syscall i64 9, i64 0, i64 65536, i64 3, i64 34, i64 -1, i64 0 : i64
  %sp = inttoptr %stk : ptr
  %top = ptr_add %sp, i64 65536 : ptr
  call @lf_ctx_init(@ctx_b, %top, @thread_b, i64 0) : void
  %i = call @lf_sig_install(i64 14, @on_alarm, i64 268435456) : i64
  %iok = icmp eq %i, i64 0 : i1
  cond_br %iok, ^1, ^9
^1:
  %t = call @set_timer(i64 1000) : i64
  call @busy_a() : void
  call @emit_dump() : void
  %s = load @sum align 8 : i64
  %p = ptr_add @dump, i64 392 : ptr
  store %s, %p align 8 : i64
  %w = syscall i64 1, i64 1, %p, i64 8 : i64
  %d = load volatile @b_done align 8 : i64
  %sw = load volatile @switched align 8 : i64
  %both = and %d, %sw : i64
  %rc = xor %both, i64 1 : i64
  ret %rc
^9:
  ret i64 9
}}
"#,
        xpat = xpat_global(),
        sum_fn = checksum_lf(&[RAX]),
    )
}

#[test]
fn timer_signal_preempts_busy_loop_and_resumes_it_intact() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let (out, status) = run_lf(&preempt_src(), level, vec![preempt_harness()], "preempt");
        assert_eq!(status.code(), Some(0), "at {level:?}: B ran and the handler switched: {status:?} (signal {:?})", status.signal());
        assert_eq!(out.len(), 50 * 8 + 8, "dump + checksum at {level:?}");
        let dump = parse_dump(&out);
        assert!(dump[RAX as usize] > 0 && dump[RAX as usize] < 0x7FFF_FFFF, "loop counter {:#x}", dump[RAX as usize]);
        for r in 0..16u16 {
            if r == RSP || r == RAX {
                continue;
            }
            assert_eq!(dump[r as usize], gpr_pattern(r), "gpr {r} at {level:?}");
        }
        assert_eq!(&dump[17..49], &xmm_patterns()[..], "xmm0..15 at {level:?}");
        let sum = u64::from_le_bytes(out[400..408].try_into().unwrap());
        assert_eq!(sum, checksum(&expected_gprs(), &xmm_patterns(), &[RAX]), "checksum at {level:?}");
    }
}
