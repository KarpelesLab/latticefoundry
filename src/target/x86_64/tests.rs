//! Tests for the x86-64 backend.
//!
//! Two tiers, matching how encoding correctness is proven:
//!
//! - **Golden byte encodings** — a handful of instructions hand-encoded from the
//!   ISA manual, asserted exactly. These need no toolchain, so encoding is
//!   validated even in a bare CI.
//! - **Execution tests** — compile an IR function to an ELF `.o`, link it with a
//!   tiny C driver using the system C compiler, run it, and check the result.
//!   These are guarded: if no C compiler is present they are skipped.

use super::encode::*;
use crate::ir::inst::{Flags, IntPred};
use crate::ir::{FuncId, Module};
use crate::mc::emit::Emitter;
use crate::support::StrInterner;

// ===========================================================================
// Golden byte encodings (validated by hand from the x86-64 manual)
// ===========================================================================

fn enc(f: impl FnOnce(&mut Emitter)) -> Vec<u8> {
    let mut e = Emitter::new();
    f(&mut e);
    e.finish().unwrap().bytes
}

#[test]
fn golden_mov_rax_rdi() {
    // mov rax, rdi = 48 89 f8
    assert_eq!(enc(|e| mov_rr(e, 0, 7, true)), vec![0x48, 0x89, 0xf8]);
}

#[test]
fn golden_add_rax_rbx() {
    // add rax, rbx = 48 01 d8
    assert_eq!(enc(|e| alu_rr(e, 0x01, 0, 3, true)), vec![0x48, 0x01, 0xd8]);
}

#[test]
fn golden_sub_rcx_rdx() {
    // sub rcx, rdx = 48 29 d1
    assert_eq!(enc(|e| alu_rr(e, 0x29, 1, 2, true)), vec![0x48, 0x29, 0xd1]);
}

#[test]
fn golden_imul_rax_rsi() {
    // imul rax, rsi = 48 0f af c6
    assert_eq!(enc(|e| imul_rr(e, 0, 6, true)), vec![0x48, 0x0f, 0xaf, 0xc6]);
}

#[test]
fn golden_mov_imm32() {
    // mov eax, 5 = b8 05 00 00 00  (zero-extends into rax)
    assert_eq!(enc(|e| mov_ri(e, 0, 5)), vec![0xb8, 0x05, 0x00, 0x00, 0x00]);
    // mov r8d, 1 = 41 b8 01 00 00 00
    assert_eq!(enc(|e| mov_ri(e, 8, 1)), vec![0x41, 0xb8, 0x01, 0x00, 0x00, 0x00]);
}

#[test]
fn golden_movabs() {
    // movabs rax, 0x1_0000_0000 = 48 b8 00 00 00 00 01 00 00 00
    assert_eq!(
        enc(|e| mov_ri(e, 0, 0x1_0000_0000)),
        vec![0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]
    );
}

#[test]
fn golden_push_pop() {
    // push rbp = 55 ; pop rbp = 5d ; push r12 = 41 54 ; pop r13 = 41 5d
    assert_eq!(enc(|e| push_r(e, 5)), vec![0x55]);
    assert_eq!(enc(|e| pop_r(e, 5)), vec![0x5d]);
    assert_eq!(enc(|e| push_r(e, 12)), vec![0x41, 0x54]);
    assert_eq!(enc(|e| pop_r(e, 13)), vec![0x41, 0x5d]);
}

#[test]
fn golden_mov_rbp_rsp() {
    // mov rbp, rsp = 48 89 e5
    assert_eq!(enc(|e| mov_rr(e, 5, 4, true)), vec![0x48, 0x89, 0xe5]);
}

#[test]
fn golden_load_store_frame() {
    // mov rax, [rbp - 8]  = 48 8b 45 f8
    assert_eq!(enc(|e| mem(e, &[0x8B], 0, 5, -8, true, false)), vec![0x48, 0x8b, 0x45, 0xf8]);
    // mov [rbp - 8], rax  = 48 89 45 f8
    assert_eq!(enc(|e| mem(e, &[0x89], 0, 5, -8, true, false)), vec![0x48, 0x89, 0x45, 0xf8]);
    // mov rax, [rsp]      = 48 8b 04 24   (SIB required for rsp base)
    assert_eq!(enc(|e| mem(e, &[0x8B], 0, 4, 0, true, false)), vec![0x48, 0x8b, 0x04, 0x24]);
}

#[test]
fn golden_setcc_and_cmp() {
    // cmp rax, rbx = 48 39 d8 ; sete al = 0f 94 c0 ; movzx eax, al = 0f b6 c0
    assert_eq!(enc(|e| alu_rr(e, 0x39, 0, 3, true)), vec![0x48, 0x39, 0xd8]);
    assert_eq!(enc(|e| setcc(e, 0x4, 0)), vec![0x0f, 0x94, 0xc0]);
    assert_eq!(enc(|e| movzx_byte(e, 0)), vec![0x0f, 0xb6, 0xc0]);
    // setl sil needs a REX prefix to name the low byte: 40 0f 9c c6
    assert_eq!(enc(|e| setcc(e, 0xc, 6)), vec![0x40, 0x0f, 0x9c, 0xc6]);
}

#[test]
fn golden_ret_via_emitter() {
    // ret = c3
    assert_eq!(enc(|e| e.u8(0xC3)), vec![0xc3]);
}

// ===========================================================================
// Execution tests
// ===========================================================================

/// The first available system C compiler, if any.
fn find_cc() -> Option<&'static str> {
    for cc in ["cc", "gcc", "clang"] {
        let ok = std::process::Command::new(cc)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return Some(cc);
        }
    }
    None
}

/// Compile `module` to an ELF object, link it with `c_main`, run the result, and
/// return the process exit code. Returns `None` when no C compiler is present
/// (so the test is skipped rather than failed).
fn compile_link_run(module: &Module, syms: &StrInterner, tag: &str, c_main: &str) -> Option<i32> {
    let cc = find_cc()?;
    let elf = compile_to_elf(module, syms);

    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let obj = dir.join(format!("lf_x86_{tag}_{pid}.o"));
    let src = dir.join(format!("lf_x86_{tag}_{pid}.c"));
    let exe = dir.join(format!("lf_x86_{tag}_{pid}.bin"));

    std::fs::write(&obj, &elf).expect("write object");
    std::fs::write(&src, c_main).expect("write C driver");

    let status = std::process::Command::new(cc)
        .arg(&src)
        .arg(&obj)
        .arg("-o")
        .arg(&exe)
        // Our relocatable object has no `.note.GNU-stack` (that section is the
        // ELF writer's concern, in `mc::elf`); ask the linker for a non-exec
        // stack explicitly so it does not warn about the missing note.
        .arg("-Wl,-z,noexecstack")
        .status()
        .expect("run cc");
    assert!(status.success(), "linking failed with {cc}");

    let out = std::process::Command::new(&exe).status().expect("run linked binary");

    let _ = std::fs::remove_file(&obj);
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&exe);

    Some(out.code().expect("child exited via signal"))
}

// --- IR fixtures -----------------------------------------------------------

/// `lfadd(a, b) = a + b` over `i32`.
fn build_add() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
    let f = m.declare_function(syms.intern("lfadd"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let s = b.add(a, bb, Flags::NONE);
        b.ret(Some(s));
    }
    (m, syms, f)
}

/// `lfmax(a, b)` over `i32`, via a branch diamond passing the larger value as a
/// block argument (exercises block-argument lowering and conditional branches).
fn build_max() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
    let f = m.declare_function(syms.intern("lfmax"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let then_b = b.create_block(&[]);
        let else_b = b.create_block(&[]);
        let join = b.create_block(&[i32t]);
        let cond = b.icmp(IntPred::Sgt, a, bb);
        b.cond_br(cond, then_b, &[], else_b, &[]);
        b.switch_to(then_b);
        b.br(join, &[a]);
        b.switch_to(else_b);
        b.br(join, &[bb]);
        b.switch_to(join);
        let r = b.param(join, 0);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// `lfsum(n) = 0 + 1 + ... + (n-1)` over `i64` — a loop with back-edge args.
fn build_loop_sum() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("lfsum"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let n = b.param(entry, 0);
        let header = b.create_block(&[i64t, i64t]);
        let body = b.create_block(&[i64t, i64t]);
        let exit = b.create_block(&[i64t]);
        b.switch_to(entry);
        let zero = b.const_i64(i64t, 0);
        b.br(header, &[zero, zero]);
        b.switch_to(header);
        let acc = b.param(header, 0);
        let i = b.param(header, 1);
        let cond = b.icmp(IntPred::Slt, i, n);
        b.cond_br(cond, body, &[acc, i], exit, &[acc]);
        b.switch_to(body);
        let bacc = b.param(body, 0);
        let bi = b.param(body, 1);
        let new_acc = b.add(bacc, bi, Flags::NONE);
        let one = b.const_i64(i64t, 1);
        let new_i = b.add(bi, one, Flags::NONE);
        b.br(header, &[new_acc, new_i]);
        b.switch_to(exit);
        let result = b.param(exit, 0);
        b.ret(Some(result));
    }
    (m, syms, f)
}

/// A caller `lfcaller(x) = lfcallee(x) + lfcallee(x)` and callee `lfcallee(y) = y*3`.
fn build_call() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let callee = m.declare_function(syms.intern("lfcallee"), sig);
    let caller = m.declare_function(syms.intern("lfcaller"), sig);
    {
        let mut b = m.build(callee);
        let entry = b.create_entry_block();
        let y = b.param(entry, 0);
        let three = b.const_i64(i64t, 3);
        let r = b.mul(y, three, Flags::NONE);
        b.ret(Some(r));
    }
    {
        let mut b = m.build(caller);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let cref1 = b.func_ref(callee);
        let c1 = b.call(cref1, &[x], i64t).unwrap();
        let cref2 = b.func_ref(callee);
        let c2 = b.call(cref2, &[x], i64t).unwrap();
        let s = b.add(c1, c2, Flags::NONE);
        b.ret(Some(s));
    }
    (m, syms)
}

/// `lfmem(x)`: alloca an i64, store x, load it back, return it.
fn build_mem() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("lfmem"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let slot = b.alloca(i64t);
        b.store(i64t, slot, x, 8);
        let loaded = b.load(i64t, slot, 8);
        b.ret(Some(loaded));
    }
    (m, syms, f)
}

// --- the execution tests ---------------------------------------------------

#[test]
fn run_add() {
    let (m, syms, _) = build_add();
    let c = r#"
        int lfadd(int, int);
        int main(void) {
            if (lfadd(3, 4) != 7) return 1;
            if (lfadd(-2, 10) != 8) return 2;
            if (lfadd(0, 0) != 0) return 3;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "add", c) {
        Some(code) => assert_eq!(code, 0, "lfadd produced wrong result (exit {code})"),
        None => eprintln!("skipping run_add: no C compiler"),
    }
}

#[test]
fn run_max() {
    let (m, syms, _) = build_max();
    let c = r#"
        int lfmax(int, int);
        int main(void) {
            if (lfmax(3, 4) != 4) return 1;
            if (lfmax(9, 2) != 9) return 2;
            if (lfmax(-1, -5) != -1) return 3;
            if (lfmax(7, 7) != 7) return 4;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "max", c) {
        Some(code) => assert_eq!(code, 0, "lfmax produced wrong result (exit {code})"),
        None => eprintln!("skipping run_max: no C compiler"),
    }
}

#[test]
fn run_loop_sum() {
    let (m, syms, _) = build_loop_sum();
    let c = r#"
        long long lfsum(long long);
        int main(void) {
            if (lfsum(0) != 0) return 1;
            if (lfsum(1) != 0) return 2;
            if (lfsum(5) != 10) return 3;
            if (lfsum(10) != 45) return 4;
            if (lfsum(100) != 4950) return 5;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "sum", c) {
        Some(code) => assert_eq!(code, 0, "lfsum produced wrong result (exit {code})"),
        None => eprintln!("skipping run_loop_sum: no C compiler"),
    }
}

#[test]
fn run_call() {
    let (m, syms) = build_call();
    let c = r#"
        long long lfcaller(long long);
        long long lfcallee(long long);
        int main(void) {
            if (lfcallee(4) != 12) return 1;
            if (lfcaller(2) != 12) return 2;   /* 2*3 + 2*3 */
            if (lfcaller(7) != 42) return 3;   /* 7*3 + 7*3 */
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "call", c) {
        Some(code) => assert_eq!(code, 0, "call chain produced wrong result (exit {code})"),
        None => eprintln!("skipping run_call: no C compiler"),
    }
}

#[test]
fn run_mem() {
    let (m, syms, _) = build_mem();
    let c = r#"
        long long lfmem(long long);
        int main(void) {
            if (lfmem(0) != 0) return 1;
            if (lfmem(42) != 42) return 2;
            if (lfmem(1234567) != 1234567) return 3;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "mem", c) {
        Some(code) => assert_eq!(code, 0, "lfmem produced wrong result (exit {code})"),
        None => eprintln!("skipping run_mem: no C compiler"),
    }
}

// --- cross-checks: objdump + determinism -----------------------------------

#[test]
fn objdump_decodes_add() {
    // If objdump is available, disassemble the compiled `lfadd` and confirm the
    // decoder recognizes our bytes (an external cross-check of the encoding).
    let objdump_ok = std::process::Command::new("objdump")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !objdump_ok {
        eprintln!("skipping objdump_decodes_add: no objdump");
        return;
    }
    let (m, syms, _) = build_add();
    let elf = compile_to_elf(&m, &syms);
    let dir = std::env::temp_dir();
    let obj = dir.join(format!("lf_x86_objdump_{}.o", std::process::id()));
    std::fs::write(&obj, &elf).unwrap();
    let out = std::process::Command::new("objdump").arg("-d").arg(&obj).output().unwrap();
    let _ = std::fs::remove_file(&obj);
    assert!(out.status.success(), "objdump rejected our object");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("<lfadd>:"), "objdump did not find the lfadd symbol");
    // The prologue and a ret should decode.
    assert!(text.contains("push") && text.contains("ret"), "expected push/ret in disassembly");
}

#[test]
fn encoding_is_deterministic() {
    let (m, syms, _) = build_loop_sum();
    let a = compile_to_elf(&m, &syms);
    let b = compile_to_elf(&m, &syms);
    assert_eq!(a, b, "identical input must yield identical bytes");
}
