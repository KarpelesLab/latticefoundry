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
use crate::ir::inst::{BinOp, CastOp, Flags, FloatPred, IntPred};
use crate::ir::types::FloatKind;
use crate::ir::value::FloatBits;
use crate::ir::{FuncId, Module};
use crate::mc::emit::Emitter;
use crate::mc::object::RelocKind;
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
fn golden_movsx_movzx() {
    // movsxd rax, ecx = 48 63 c1  (32->64 sign-extend)
    assert_eq!(enc(|e| movsx_rr(e, 0, 1, 32, 64)), vec![0x48, 0x63, 0xc1]);
    // movsx eax, cl = 0f be c1  (8->32)
    assert_eq!(enc(|e| movsx_rr(e, 0, 1, 8, 32)), vec![0x0f, 0xbe, 0xc1]);
    // movsx rax, cx = 48 0f bf c1  (16->64)
    assert_eq!(enc(|e| movsx_rr(e, 0, 1, 16, 64)), vec![0x48, 0x0f, 0xbf, 0xc1]);
    // movzx eax, cl = 0f b6 c1  (8->zero-extend)
    assert_eq!(enc(|e| movzx_rr(e, 0, 1, 8)), vec![0x0f, 0xb6, 0xc1]);
    // movzx eax, cx = 0f b7 c1  (16->)
    assert_eq!(enc(|e| movzx_rr(e, 0, 1, 16)), vec![0x0f, 0xb7, 0xc1]);
    // zext of a 32-bit source is a plain 32-bit mov (zero-extends 32->64):
    // mov eax, ecx = 89 c8
    assert_eq!(enc(|e| movzx_rr(e, 0, 1, 32)), vec![0x89, 0xc8]);
}

#[test]
fn golden_ret_via_emitter() {
    // ret = c3
    assert_eq!(enc(|e| e.u8(0xC3)), vec![0xc3]);
}

// --- SSE golden byte encodings (hand-verified from the manual) --------------

#[test]
fn golden_sse_arith_reg_reg() {
    // addsd xmm0, xmm1 = f2 0f 58 c1     (prefix, 0F, opcode, modrm reg=0 rm=1)
    assert_eq!(enc(|e| sse_rr(e, 0xF2, false, 0x58, 0, 1)), vec![0xf2, 0x0f, 0x58, 0xc1]);
    // addss xmm0, xmm1 = f3 0f 58 c1
    assert_eq!(enc(|e| sse_rr(e, 0xF3, false, 0x58, 0, 1)), vec![0xf3, 0x0f, 0x58, 0xc1]);
    // subsd xmm0, xmm1 = f2 0f 5c c1
    assert_eq!(enc(|e| sse_rr(e, 0xF2, false, 0x5C, 0, 1)), vec![0xf2, 0x0f, 0x5c, 0xc1]);
    // mulsd xmm0, xmm1 = f2 0f 59 c1
    assert_eq!(enc(|e| sse_rr(e, 0xF2, false, 0x59, 0, 1)), vec![0xf2, 0x0f, 0x59, 0xc1]);
    // divsd xmm0, xmm1 = f2 0f 5e c1
    assert_eq!(enc(|e| sse_rr(e, 0xF2, false, 0x5E, 0, 1)), vec![0xf2, 0x0f, 0x5e, 0xc1]);
    // movsd xmm0, xmm1 = f2 0f 10 c1
    assert_eq!(enc(|e| sse_rr(e, 0xF2, false, 0x10, 0, 1)), vec![0xf2, 0x0f, 0x10, 0xc1]);
    // movss xmm0, xmm1 = f3 0f 10 c1
    assert_eq!(enc(|e| sse_rr(e, 0xF3, false, 0x10, 0, 1)), vec![0xf3, 0x0f, 0x10, 0xc1]);
    // ucomisd xmm0, xmm1 = 66 0f 2e c1
    assert_eq!(enc(|e| sse_rr(e, 0x66, false, 0x2E, 0, 1)), vec![0x66, 0x0f, 0x2e, 0xc1]);
    // xorpd xmm0, xmm1 = 66 0f 57 c1
    assert_eq!(enc(|e| sse_rr(e, 0x66, false, 0x57, 0, 1)), vec![0x66, 0x0f, 0x57, 0xc1]);
    // addsd xmm8, xmm1 = f2 44 0f 58 c1  (REX.R for the xmm8 destination)
    assert_eq!(enc(|e| sse_rr(e, 0xF2, false, 0x58, 8, 1)), vec![0xf2, 0x44, 0x0f, 0x58, 0xc1]);
}

#[test]
fn golden_sse_convert_and_movq() {
    // cvtsi2sd xmm0, rax = f2 48 0f 2a c0   (REX.W, xmm=reg, rax=rm)
    assert_eq!(enc(|e| sse_rr(e, 0xF2, true, 0x2A, 0, 0)), vec![0xf2, 0x48, 0x0f, 0x2a, 0xc0]);
    // cvttsd2si rax, xmm0 = f2 48 0f 2c c0  (REX.W, rax=reg, xmm=rm)
    assert_eq!(enc(|e| sse_rr(e, 0xF2, true, 0x2C, 0, 0)), vec![0xf2, 0x48, 0x0f, 0x2c, 0xc0]);
    // movq xmm0, rax = 66 48 0f 6e c0
    assert_eq!(enc(|e| sse_rr(e, 0x66, true, 0x6E, 0, 0)), vec![0x66, 0x48, 0x0f, 0x6e, 0xc0]);
}

#[test]
fn golden_sse_mem() {
    // movsd xmm0, [rbp-8] = f2 0f 10 45 f8
    assert_eq!(enc(|e| sse_mem(e, 0xF2, 0x10, 0, 5, -8)), vec![0xf2, 0x0f, 0x10, 0x45, 0xf8]);
    // movsd [rbp-8], xmm0 = f2 0f 11 45 f8
    assert_eq!(enc(|e| sse_mem(e, 0xF2, 0x11, 0, 5, -8)), vec![0xf2, 0x0f, 0x11, 0x45, 0xf8]);
    // movss xmm0, [rbp-8] = f3 0f 10 45 f8
    assert_eq!(enc(|e| sse_mem(e, 0xF3, 0x10, 0, 5, -8)), vec![0xf3, 0x0f, 0x10, 0x45, 0xf8]);
    // movsd xmm0, [rsp] = f2 0f 10 04 24  (SIB required for the rsp base)
    assert_eq!(enc(|e| sse_mem(e, 0xF2, 0x10, 0, 4, 0)), vec![0xf2, 0x0f, 0x10, 0x04, 0x24]);
}

/// Assemble one AT&T-syntax instruction with `llvm-mc --show-encoding` and
/// return its bytes, or `None` if `llvm-mc` is unavailable.
fn llvm_mc_encode(att: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let mut child = std::process::Command::new("llvm-mc")
        .arg("--triple=x86_64")
        .arg("--show-encoding")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(format!("{att}\n").as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // Find the `# encoding: [0x..,0x..]` comment and parse its bytes.
    let marker = "encoding: [";
    let start = text.find(marker)? + marker.len();
    let end = text[start..].find(']')? + start;
    let mut bytes = Vec::new();
    for tok in text[start..end].split(',') {
        let t = tok.trim().trim_start_matches("0x");
        if !t.is_empty() {
            bytes.push(u8::from_str_radix(t, 16).ok()?);
        }
    }
    Some(bytes)
}

#[test]
fn differential_sse_vs_llvm_mc() {
    // (AT&T `op src, dst`, our-encoder closure). AT&T reverses operand order.
    type Case = (&'static str, Box<dyn Fn(&mut Emitter)>);
    let cases: Vec<Case> = vec![
        ("addsd %xmm1, %xmm0", Box::new(|e| sse_rr(e, 0xF2, false, 0x58, 0, 1))),
        ("addss %xmm3, %xmm2", Box::new(|e| sse_rr(e, 0xF3, false, 0x58, 2, 3))),
        ("subsd %xmm1, %xmm0", Box::new(|e| sse_rr(e, 0xF2, false, 0x5C, 0, 1))),
        ("mulsd %xmm5, %xmm4", Box::new(|e| sse_rr(e, 0xF2, false, 0x59, 4, 5))),
        ("divsd %xmm1, %xmm0", Box::new(|e| sse_rr(e, 0xF2, false, 0x5E, 0, 1))),
        ("movsd %xmm7, %xmm6", Box::new(|e| sse_rr(e, 0xF2, false, 0x10, 6, 7))),
        ("movss %xmm1, %xmm0", Box::new(|e| sse_rr(e, 0xF3, false, 0x10, 0, 1))),
        ("ucomisd %xmm1, %xmm0", Box::new(|e| sse_rr(e, 0x66, false, 0x2E, 0, 1))),
        ("ucomiss %xmm1, %xmm0", Box::new(|e| sse_rr(e, 0x00, false, 0x2E, 0, 1))),
        ("xorpd %xmm1, %xmm0", Box::new(|e| sse_rr(e, 0x66, false, 0x57, 0, 1))),
        ("xorps %xmm1, %xmm0", Box::new(|e| sse_rr(e, 0x00, false, 0x57, 0, 1))),
        ("cvtsd2ss %xmm1, %xmm0", Box::new(|e| sse_rr(e, 0xF2, false, 0x5A, 0, 1))),
        ("cvtss2sd %xmm1, %xmm0", Box::new(|e| sse_rr(e, 0xF3, false, 0x5A, 0, 1))),
        ("cvtsi2sd %rax, %xmm0", Box::new(|e| sse_rr(e, 0xF2, true, 0x2A, 0, 0))),
        ("cvtsi2sd %eax, %xmm0", Box::new(|e| sse_rr(e, 0xF2, false, 0x2A, 0, 0))),
        ("cvtsi2ss %rcx, %xmm2", Box::new(|e| sse_rr(e, 0xF3, true, 0x2A, 2, 1))),
        ("cvttsd2si %xmm0, %rax", Box::new(|e| sse_rr(e, 0xF2, true, 0x2C, 0, 0))),
        ("cvttss2si %xmm0, %eax", Box::new(|e| sse_rr(e, 0xF3, false, 0x2C, 0, 0))),
        ("cvttsd2si %xmm2, %edx", Box::new(|e| sse_rr(e, 0xF2, false, 0x2C, 2, 2))),
        ("movq %rax, %xmm0", Box::new(|e| sse_rr(e, 0x66, true, 0x6E, 0, 0))),
        ("movd %eax, %xmm0", Box::new(|e| sse_rr(e, 0x66, false, 0x6E, 0, 0))),
        ("addsd %xmm1, %xmm8", Box::new(|e| sse_rr(e, 0xF2, false, 0x58, 8, 1))),
        ("addsd %xmm9, %xmm0", Box::new(|e| sse_rr(e, 0xF2, false, 0x58, 0, 9))),
        ("movsd -8(%rbp), %xmm0", Box::new(|e| sse_mem(e, 0xF2, 0x10, 0, 5, -8))),
        ("movsd %xmm0, -8(%rbp)", Box::new(|e| sse_mem(e, 0xF2, 0x11, 0, 5, -8))),
        ("movss -8(%rbp), %xmm0", Box::new(|e| sse_mem(e, 0xF3, 0x10, 0, 5, -8))),
        ("movsd (%rsp), %xmm0", Box::new(|e| sse_mem(e, 0xF2, 0x10, 0, 4, 0))),
    ];

    if llvm_mc_encode("addsd %xmm1, %xmm0").is_none() {
        eprintln!("skipping differential_sse_vs_llvm_mc: no llvm-mc");
        return;
    }
    let mut matched = 0usize;
    for (att, build) in &cases {
        let mine = enc(build);
        let theirs = llvm_mc_encode(att).unwrap_or_else(|| panic!("llvm-mc failed on {att}"));
        assert_eq!(mine, theirs, "encoding mismatch for `{att}`: ours={mine:x?} llvm={theirs:x?}");
        matched += 1;
    }
    assert_eq!(matched, cases.len(), "all SSE instructions matched llvm-mc");
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

/// `double lffd(double a, double b) = a*b + a/b - (a-b)` — F64 arithmetic.
fn build_fdouble() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let f64t = m.types_mut().float(FloatKind::F64);
    let sig = m.types_mut().func(vec![f64t, f64t], f64t, false);
    let f = m.declare_function(syms.intern("lffd"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let mul = b.bin(BinOp::FMul, a, bb, Flags::NONE);
        let div = b.bin(BinOp::FDiv, a, bb, Flags::NONE);
        let s1 = b.bin(BinOp::FAdd, mul, div, Flags::NONE);
        let sub = b.bin(BinOp::FSub, a, bb, Flags::NONE);
        let r = b.bin(BinOp::FSub, s1, sub, Flags::NONE);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// `float lffs(float a, float b) = a*b + a/b - (a-b)` — F32 arithmetic.
fn build_ffloat() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let f32t = m.types_mut().float(FloatKind::F32);
    let sig = m.types_mut().func(vec![f32t, f32t], f32t, false);
    let f = m.declare_function(syms.intern("lffs"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let mul = b.bin(BinOp::FMul, a, bb, Flags::NONE);
        let div = b.bin(BinOp::FDiv, a, bb, Flags::NONE);
        let s1 = b.bin(BinOp::FAdd, mul, div, Flags::NONE);
        let sub = b.bin(BinOp::FSub, a, bb, Flags::NONE);
        let r = b.bin(BinOp::FSub, s1, sub, Flags::NONE);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// `widecmp(n)` sign-extends an `i32` to `i64` and returns whether it is negative
/// (1/0). Regression for integer `sext`, which used to lower to a plain 64-bit
/// `mov` — so a negative `i32` widened to a *positive* `i64` and `< 0` was false.
fn build_sext_cmp() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i32t], i32t, false);
    let f = m.declare_function(syms.intern("widecmp"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let n = b.param(entry, 0);
        let w = b.cast(CastOp::SExt, n, i64t);
        let zero = b.const_i64(i64t, 0);
        let c = b.icmp(IntPred::Slt, w, zero);
        let r = b.cast(CastOp::ZExt, c, i32t);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// `double call_g() = g(1.5, 2.5)` where `double g(double,double)=a+b`. Passes
/// two float *constants* directly as call arguments — a regression fixture for
/// the arg-register clobber (the 2nd constant's materialization must not land in
/// `xmm0`, which already holds arg0).
fn build_fconst_args() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let f64t = m.types_mut().float(FloatKind::F64);
    let sig_g = m.types_mut().func(vec![f64t, f64t], f64t, false);
    let g = m.declare_function(syms.intern("g"), sig_g);
    {
        let mut b = m.build(g);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let s = b.bin(BinOp::FAdd, a, bb, Flags::NONE);
        b.ret(Some(s));
    }
    let sig_c = m.types_mut().func(vec![], f64t, false);
    let caller = m.declare_function(syms.intern("call_g"), sig_c);
    {
        let mut b = m.build(caller);
        let _entry = b.create_entry_block();
        let gref = b.func_ref(g);
        let c1 = b.const_float(f64t, FloatBits::F64(1.5f64.to_bits()));
        let c2 = b.const_float(f64t, FloatBits::F64(2.5f64.to_bits()));
        let r = b.call(gref, &[c1, c2], f64t).expect("call has a result");
        b.ret(Some(r));
    }
    (m, syms)
}

/// `int lff2i(double x) = (int)(x*x)` — `fptosi` after an `fmul`.
fn build_fptosi() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let f64t = m.types_mut().float(FloatKind::F64);
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![f64t], i32t, false);
    let f = m.declare_function(syms.intern("lff2i"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let sq = b.bin(BinOp::FMul, x, x, Flags::NONE);
        let r = b.cast(CastOp::FpToSi, sq, i32t);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// `double lfi2f(int n) = (double)n / 2.0` — `sitofp` then an `fdiv` by a
/// materialized float constant.
fn build_sitofp() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let f64t = m.types_mut().float(FloatKind::F64);
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t], f64t, false);
    let f = m.declare_function(syms.intern("lfi2f"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let n = b.param(entry, 0);
        let nf = b.cast(CastOp::SiToFp, n, f64t);
        let two = b.const_float(f64t, FloatBits::F64(2.0f64.to_bits()));
        let r = b.bin(BinOp::FDiv, nf, two, Flags::NONE);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// `double lffmax(double a, double b) = a > b ? a : b` — an `fcmp ogt` driving a
/// branch, with the winner passed as an F64 block argument.
fn build_fmax() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let f64t = m.types_mut().float(FloatKind::F64);
    let sig = m.types_mut().func(vec![f64t, f64t], f64t, false);
    let f = m.declare_function(syms.intern("lffmax"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let then_b = b.create_block(&[]);
        let else_b = b.create_block(&[]);
        let join = b.create_block(&[f64t]);
        let cond = b.fcmp(FloatPred::Ogt, a, bb, Flags::NONE);
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

/// `double lfmix(int a, double x, int b, double y) = (double)(a - b) + x*y` — a
/// function mixing integer (rdi/rsi) and float (xmm0/xmm1) parameters and
/// returning a double (xmm0).
fn build_fmix() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let f64t = m.types_mut().float(FloatKind::F64);
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t, f64t, i32t, f64t], f64t, false);
    let f = m.declare_function(syms.intern("lfmix"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let x = b.param(entry, 1);
        let bb = b.param(entry, 2);
        let y = b.param(entry, 3);
        let diff = b.bin(BinOp::Sub, a, bb, Flags::NONE);
        let difff = b.cast(CastOp::SiToFp, diff, f64t);
        let prod = b.bin(BinOp::FMul, x, y, Flags::NONE);
        let r = b.bin(BinOp::FAdd, difff, prod, Flags::NONE);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// `double lfneg(double x) = -x` — `fneg` via the sign-bit `xorpd`.
fn build_fneg() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let f64t = m.types_mut().float(FloatKind::F64);
    let sig = m.types_mut().func(vec![f64t], f64t, false);
    let f = m.declare_function(syms.intern("lfneg"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let r = b.fneg(x, Flags::NONE);
        b.ret(Some(r));
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
    if let Some(code) = compile_link_run(&m, &syms, "add", c) {
        assert_eq!(code, 0, "lfadd checks failed (exit {code})");
    }
}

#[test]
fn run_sext_negative() {
    let (m, syms, _f) = build_sext_cmp();
    // widecmp(-3) must be 1 (negative), widecmp(5) must be 0.
    let c = "extern int widecmp(int); int main(void){ return (widecmp(-3)==1 && widecmp(5)==0) ? 0 : 1; }";
    if let Some(code) = compile_link_run(&m, &syms, "sext", c) {
        assert_eq!(code, 0, "sext of a negative i32 to i64 was not sign-extended");
    }
}

#[test]
fn run_float_constant_args() {
    // Regression: `g(1.5, 2.5)` with both arguments passed as float constants.
    // The second constant used to be materialized into xmm0, clobbering arg0, so
    // g received (2.5, 2.5) and returned 5.0 instead of 4.0.
    let (m, syms) = build_fconst_args();
    let c = "extern double call_g(void); int main(void){ return call_g() == 4.0 ? 0 : 1; }";
    if let Some(code) = compile_link_run(&m, &syms, "fconst", c) {
        assert_eq!(code, 0, "call_g() != 4.0 — an argument register was clobbered");
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

#[test]
fn run_fdouble() {
    let (m, syms, _) = build_fdouble();
    // Exact IEEE-754 bit equality: the C driver computes the same f64 ops.
    let c = r#"
        double lffd(double, double);
        static int eq(double x, double y) { return x == y; }
        int main(void) {
            double cases[][2] = {{3.0,4.0},{1.5,-2.25},{100.0,7.0},{-8.0,0.5}};
            for (int i = 0; i < 4; i++) {
                double a = cases[i][0], b = cases[i][1];
                double want = a*b + a/b - (a-b);
                if (!eq(lffd(a, b), want)) return i + 1;
            }
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "fdouble", c) {
        Some(code) => assert_eq!(code, 0, "lffd produced wrong result (exit {code})"),
        None => eprintln!("skipping run_fdouble: no C compiler"),
    }
}

#[test]
fn run_ffloat() {
    let (m, syms, _) = build_ffloat();
    let c = r#"
        float lffs(float, float);
        static int eq(float x, float y) { return x == y; }
        int main(void) {
            float cases[][2] = {{3.0f,4.0f},{1.5f,-2.25f},{100.0f,7.0f},{-8.0f,0.5f}};
            for (int i = 0; i < 4; i++) {
                float a = cases[i][0], b = cases[i][1];
                float want = a*b + a/b - (a-b);
                if (!eq(lffs(a, b), want)) return i + 1;
            }
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "ffloat", c) {
        Some(code) => assert_eq!(code, 0, "lffs produced wrong result (exit {code})"),
        None => eprintln!("skipping run_ffloat: no C compiler"),
    }
}

#[test]
fn run_fptosi() {
    let (m, syms, _) = build_fptosi();
    let c = r#"
        int lff2i(double);
        int main(void) {
            if (lff2i(3.0) != 9) return 1;
            if (lff2i(2.5) != 6) return 2;   /* (int)6.25 */
            if (lff2i(-4.0) != 16) return 3;
            if (lff2i(10.9) != 118) return 4; /* (int)118.81 */
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "fptosi", c) {
        Some(code) => assert_eq!(code, 0, "lff2i produced wrong result (exit {code})"),
        None => eprintln!("skipping run_fptosi: no C compiler"),
    }
}

#[test]
fn run_sitofp() {
    let (m, syms, _) = build_sitofp();
    let c = r#"
        double lfi2f(int);
        static int eq(double x, double y) { return x == y; }
        int main(void) {
            if (!eq(lfi2f(10), 5.0)) return 1;
            if (!eq(lfi2f(7), 3.5)) return 2;
            if (!eq(lfi2f(-3), -1.5)) return 3;
            if (!eq(lfi2f(0), 0.0)) return 4;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "sitofp", c) {
        Some(code) => assert_eq!(code, 0, "lfi2f produced wrong result (exit {code})"),
        None => eprintln!("skipping run_sitofp: no C compiler"),
    }
}

#[test]
fn run_fmax() {
    let (m, syms, _) = build_fmax();
    let c = r#"
        double lffmax(double, double);
        static int eq(double x, double y) { return x == y; }
        int main(void) {
            if (!eq(lffmax(3.0, 4.0), 4.0)) return 1;
            if (!eq(lffmax(9.5, 2.0), 9.5)) return 2;
            if (!eq(lffmax(-1.0, -5.0), -1.0)) return 3;
            if (!eq(lffmax(7.0, 7.0), 7.0)) return 4;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "fmax", c) {
        Some(code) => assert_eq!(code, 0, "lffmax produced wrong result (exit {code})"),
        None => eprintln!("skipping run_fmax: no C compiler"),
    }
}

#[test]
fn run_fmix() {
    let (m, syms, _) = build_fmix();
    let c = r#"
        double lfmix(int, double, int, double);
        static int eq(double x, double y) { return x == y; }
        int main(void) {
            /* (double)(a-b) + x*y */
            if (!eq(lfmix(10, 1.5, 4, 2.0), 9.0)) return 1;   /* 6 + 3.0 */
            if (!eq(lfmix(0, -2.5, 5, 4.0), -15.0)) return 2; /* -5 + -10.0 */
            if (!eq(lfmix(3, 0.5, 3, 8.0), 4.0)) return 3;    /* 0 + 4.0 */
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "fmix", c) {
        Some(code) => assert_eq!(code, 0, "lfmix produced wrong result (exit {code})"),
        None => eprintln!("skipping run_fmix: no C compiler"),
    }
}

#[test]
fn run_fneg() {
    let (m, syms, _) = build_fneg();
    let c = r#"
        double lfneg(double);
        static int eq(double x, double y) { return x == y; }
        int main(void) {
            if (!eq(lfneg(3.5), -3.5)) return 1;
            if (!eq(lfneg(-2.0), 2.0)) return 2;
            /* fneg is a sign flip, not 0-x: -0.0 has the sign bit set, so
               1.0 / -0.0 is -inf (< 0). */
            if (1.0 / lfneg(0.0) >= 0.0) return 3;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "fneg", c) {
        Some(code) => assert_eq!(code, 0, "lfneg produced wrong result (exit {code})"),
        None => eprintln!("skipping run_fneg: no C compiler"),
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

// ===========================================================================
// Function-address materialization + unsigned-64 conversions
// ===========================================================================

/// `long long fp_add7(long long)` = `callee(x)` reached *indirectly*: the address
/// of `fp_callee` is taken (a `func_ref` used as a value), stored into a stack
/// slot, loaded back as a plain pointer, and called through the register. The
/// stored value forces the function's address to be materialized, and the loaded
/// pointer forces an indirect `call` (not a direct `call @callee`).
fn build_func_pointer() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let ptrt = m.types_mut().ptr();
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let callee = m.declare_function(syms.intern("fp_callee"), sig);
    let caller = m.declare_function(syms.intern("fp_add7"), sig);
    {
        let mut b = m.build(callee);
        let entry = b.create_entry_block();
        let y = b.param(entry, 0);
        let seven = b.const_i64(i64t, 7);
        let r = b.mul(y, seven, Flags::NONE);
        b.ret(Some(r));
    }
    {
        let mut b = m.build(caller);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let slot = b.alloca(ptrt);
        // `func_ref` used as a VALUE: its address is stored to memory.
        let fref = b.func_ref(callee);
        b.store(ptrt, slot, fref, 8);
        // Load the pointer back and call *through it* (indirect call).
        let fp = b.load(ptrt, slot, 8);
        let r = b.call(fp, &[x], i64t).expect("call has a result");
        b.ret(Some(r));
    }
    (m, syms)
}

#[test]
fn func_address_is_riprel_lea_with_reloc() {
    // A function used as a value must materialize as a RIP-relative `lea` with a
    // relocation to the function symbol — never `mov reg, 0`. `mov reg, 0` emits
    // no relocation, so the presence of a PC32 reloc against `fp_callee` proves
    // the address is real.
    let (m, syms) = build_func_pointer();
    let obj = compile_module(&m, &syms);

    let mut found = None;
    for r in obj.relocations() {
        if obj.symbol(r.symbol).name == "fp_callee" && r.kind == RelocKind::Pc32 {
            found = Some(*r);
        }
    }
    let r = found.expect("expected a PC32 relocation to fp_callee (the taken address)");

    // The reloc patches the disp32 of a `lea d, [rip + disp32]`: the three bytes
    // just before it are `REX.W(0x48/0x4C) 8D modrm` with ModRM.mod=00, rm=101.
    let bytes = &obj.section(r.section).bytes;
    let at = r.offset as usize;
    assert!(at >= 3, "relocation cannot sit at the very start of .text");
    let rex = bytes[at - 3];
    let opcode = bytes[at - 2];
    let modrm = bytes[at - 1];
    assert!(rex == 0x48 || rex == 0x4C, "expected a REX.W prefix, got {rex:#04x}");
    assert_eq!(opcode, 0x8D, "expected the `lea` opcode 0x8D");
    assert_eq!(modrm & 0b1100_0111, 0b0000_0101, "expected ModRM rip-relative (mod=00, rm=101)");
}

#[test]
fn run_func_pointer() {
    // Execution: the indirect call through the materialized address must reach
    // `fp_callee` and return the right value (x * 7).
    let (m, syms) = build_func_pointer();
    let c = r#"
        long long fp_add7(long long);
        long long fp_callee(long long);
        int main(void) {
            if (fp_callee(6) != 42) return 1;
            if (fp_add7(6) != 42) return 2;   /* 6 * 7, reached indirectly */
            if (fp_add7(0) != 0) return 3;
            if (fp_add7(-3) != -21) return 4;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "funcptr", c) {
        Some(code) => assert_eq!(code, 0, "indirect call via taken address failed (exit {code})"),
        None => eprintln!("skipping run_func_pointer: no C compiler"),
    }
}

/// `f64 u2d(u64)` = `(double)(unsigned long)x` — `uitofp` from a 64-bit source.
fn build_u64_to_f64() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let f64t = m.types_mut().float(FloatKind::F64);
    let sig = m.types_mut().func(vec![i64t], f64t, false);
    let f = m.declare_function(syms.intern("u2d"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let r = b.cast(CastOp::UiToFp, x, f64t);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// `f32 u2f(u64)` = `(float)(unsigned long)x` — `uitofp` from 64 bits to `f32`.
fn build_u64_to_f32() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let f32t = m.types_mut().float(FloatKind::F32);
    let sig = m.types_mut().func(vec![i64t], f32t, false);
    let f = m.declare_function(syms.intern("u2f"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let r = b.cast(CastOp::UiToFp, x, f32t);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// `u64 d2u(f64)` = `(unsigned long)x` — `fptoui` to a 64-bit result (truncating).
fn build_f64_to_u64() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let f64t = m.types_mut().float(FloatKind::F64);
    let sig = m.types_mut().func(vec![f64t], i64t, false);
    let f = m.declare_function(syms.intern("d2u"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let r = b.cast(CastOp::FpToUi, x, i64t);
        b.ret(Some(r));
    }
    (m, syms, f)
}

/// `u64 f2u(f32)` = `(unsigned long)x` — `fptoui` from `f32` to a 64-bit result.
fn build_f32_to_u64() -> (Module, StrInterner, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let f32t = m.types_mut().float(FloatKind::F32);
    let sig = m.types_mut().func(vec![f32t], i64t, false);
    let f = m.declare_function(syms.intern("f2u"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let r = b.cast(CastOp::FpToUi, x, i64t);
        b.ret(Some(r));
    }
    (m, syms, f)
}

#[test]
fn run_uitofp_u64() {
    // uitofp of unsigned 64-bit values, including the > 2^63 range, must match
    // the C compiler's `(double)(unsigned long)` / `(float)(unsigned long)`
    // (round-to-nearest), across boundary values.
    let (m, syms, _) = build_u64_to_f64();
    let c = r#"
        double u2d(unsigned long);
        static int eqd(double a, double b) { return a == b; }
        int main(void) {
            unsigned long xs[] = {
                0UL, 1UL, 1024UL,
                0x7FFFFFFFFFFFFFFFUL,           /* 2^63 - 1 (sign bit clear) */
                0x8000000000000000UL,           /* 2^63     (sign bit set)   */
                0x8000000000000001UL,           /* 2^63 + 1 (rounding)       */
                0xFFFFFFFFFFFFFFFFUL,           /* 2^64 - 1                  */
                1234567890123456789UL
            };
            for (int i = 0; i < 8; i++)
                if (!eqd(u2d(xs[i]), (double)xs[i])) return i + 1;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "u2d", c) {
        Some(code) => assert_eq!(code, 0, "u2d mismatch vs C (double)(unsigned long) (exit {code})"),
        None => eprintln!("skipping run_uitofp_u64: no C compiler"),
    }
}

#[test]
fn run_uitofp_u64_to_f32() {
    let (m, syms, _) = build_u64_to_f32();
    let c = r#"
        float u2f(unsigned long);
        static int eqf(float a, float b) { return a == b; }
        int main(void) {
            unsigned long xs[] = {
                0UL, 1UL, 0x7FFFFFFFFFFFFFFFUL, 0x8000000000000000UL,
                0x8000000000000001UL, 0xFFFFFFFFFFFFFFFFUL, 1234567890123456789UL
            };
            for (int i = 0; i < 7; i++)
                if (!eqf(u2f(xs[i]), (float)xs[i])) return i + 1;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "u2f", c) {
        Some(code) => assert_eq!(code, 0, "u2f mismatch vs C (float)(unsigned long) (exit {code})"),
        None => eprintln!("skipping run_uitofp_u64_to_f32: no C compiler"),
    }
}

#[test]
fn run_fptoui_u64() {
    // fptoui to unsigned 64-bit, including values ≥ 2^63, must match the C
    // compiler's `(unsigned long)` truncation. Values stay below 2^64 (≥ 2^64 is
    // C UB).
    let (m, syms, _) = build_f64_to_u64();
    let c = r#"
        unsigned long d2u(double);
        int main(void) {
            double xs[] = {
                0.0, 1.5, 42.9,
                4611686018427387904.0,      /* 2^62 */
                9223372036854775808.0,      /* 2^63 (boundary) */
                9223372036854775809.0,      /* just above 2^63 */
                1.0e19,
                1.8e19,                     /* < 2^64 ≈ 1.8446744e19 */
                12345678901234.5
            };
            for (int i = 0; i < 9; i++)
                if (d2u(xs[i]) != (unsigned long)xs[i]) return i + 1;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "d2u", c) {
        Some(code) => assert_eq!(code, 0, "d2u mismatch vs C (unsigned long)double (exit {code})"),
        None => eprintln!("skipping run_fptoui_u64: no C compiler"),
    }
}

#[test]
fn run_fptoui_u64_from_f32() {
    let (m, syms, _) = build_f32_to_u64();
    let c = r#"
        unsigned long f2u(float);
        int main(void) {
            float xs[] = {
                0.0f, 1.5f, 42.9f,
                4611686018427387904.0f,     /* 2^62 */
                9223372036854775808.0f,     /* 2^63 */
                1.0e19f,
                1.8e19f
            };
            for (int i = 0; i < 7; i++)
                if (f2u(xs[i]) != (unsigned long)xs[i]) return i + 1;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "f2u", c) {
        Some(code) => assert_eq!(code, 0, "f2u mismatch vs C (unsigned long)float (exit {code})"),
        None => eprintln!("skipping run_fptoui_u64_from_f32: no C compiler"),
    }
}

// ===========================================================================
// By-value struct passing / returning (System V AMD64 aggregate ABI)
//
// A struct value is represented at the codegen level by a pointer to its
// in-memory storage, so the IR reads/writes its fields with `ptr_add`+`load`/
// `store` off the struct value used as a pointer, and returns an `alloca`'d
// struct by its address. The backend implements the eightbyte classification,
// register/stack argument placement, and register/`sret` return. Each program
// is linked against a gcc-compiled C driver using the *same* struct type, so a
// correct ABI must interoperate exactly.
// ===========================================================================

/// `struct P { int x, y; }` (8 bytes → one INTEGER eightbyte): `addP` returns
/// `{a.x+b.x, a.y+b.y}`. Args in `rdi`/`rsi`, result in `rax`.
fn build_struct_int() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let p = m.types_mut().struct_(vec![i32t, i32t]);
    let sig = m.types_mut().func(vec![p, p], p, false);
    let f = m.declare_function(syms.intern("addP"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let ax_p = b.struct_field(a, p, 0);
        let ax = b.load(i32t, ax_p, 4);
        let ay_p = b.struct_field(a, p, 1);
        let ay = b.load(i32t, ay_p, 4);
        let bx_p = b.struct_field(bb, p, 0);
        let bx = b.load(i32t, bx_p, 4);
        let by_p = b.struct_field(bb, p, 1);
        let by = b.load(i32t, by_p, 4);
        let sx = b.add(ax, bx, Flags::NONE);
        let sy = b.add(ay, by, Flags::NONE);
        let r = b.alloca(p);
        let rx = b.struct_field(r, p, 0);
        b.store(i32t, rx, sx, 4);
        let ry = b.struct_field(r, p, 1);
        b.store(i32t, ry, sy, 4);
        b.ret(Some(r));
    }
    (m, syms)
}

#[test]
fn run_struct_int() {
    let (m, syms) = build_struct_int();
    let c = r#"
        struct P { int x, y; };
        struct P addP(struct P, struct P);
        int main(void) {
            struct P r = addP((struct P){3, 4}, (struct P){10, 20});
            if (r.x != 13 || r.y != 24) return 1;
            struct P r2 = addP((struct P){-1, 100}, (struct P){1, -100});
            if (r2.x != 0 || r2.y != 0) return 2;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "struct_int", c) {
        Some(code) => assert_eq!(code, 0, "addP struct-by-value mismatch vs gcc (exit {code})"),
        None => eprintln!("skipping run_struct_int: no C compiler"),
    }
}

/// `struct V { double x, y; }` (16 bytes → two SSE eightbytes): `addV` returns
/// `{a.x+b.x, a.y+b.y}`. Args in `xmm0..3`, result in `xmm0`/`xmm1`.
fn build_struct_double() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let f64t = m.types_mut().float(FloatKind::F64);
    let v = m.types_mut().struct_(vec![f64t, f64t]);
    let sig = m.types_mut().func(vec![v, v], v, false);
    let f = m.declare_function(syms.intern("addV"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let ax_p = b.struct_field(a, v, 0);
        let ax = b.load(f64t, ax_p, 8);
        let ay_p = b.struct_field(a, v, 1);
        let ay = b.load(f64t, ay_p, 8);
        let bx_p = b.struct_field(bb, v, 0);
        let bx = b.load(f64t, bx_p, 8);
        let by_p = b.struct_field(bb, v, 1);
        let by = b.load(f64t, by_p, 8);
        let sx = b.bin(BinOp::FAdd, ax, bx, Flags::NONE);
        let sy = b.bin(BinOp::FAdd, ay, by, Flags::NONE);
        let r = b.alloca(v);
        let rx = b.struct_field(r, v, 0);
        b.store(f64t, rx, sx, 8);
        let ry = b.struct_field(r, v, 1);
        b.store(f64t, ry, sy, 8);
        b.ret(Some(r));
    }
    (m, syms)
}

#[test]
fn run_struct_double() {
    let (m, syms) = build_struct_double();
    let c = r#"
        struct V { double x, y; };
        struct V addV(struct V, struct V);
        int main(void) {
            struct V r = addV((struct V){1.5, 2.25}, (struct V){0.5, 0.75});
            if (r.x != 2.0 || r.y != 3.0) return 1;
            struct V r2 = addV((struct V){-8.0, 0.5}, (struct V){8.0, -0.5});
            if (r2.x != 0.0 || r2.y != 0.0) return 2;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "struct_double", c) {
        Some(code) => assert_eq!(code, 0, "addV struct-by-value mismatch vs gcc (exit {code})"),
        None => eprintln!("skipping run_struct_double: no C compiler"),
    }
}

/// `struct M { int i; double d; }` (16 bytes → INTEGER + SSE): `addM` returns
/// `{a.i+b.i, a.d+b.d}`. First arg `{rdi, xmm0}`, second `{rsi, xmm1}`, result
/// `{rax, xmm0}` — exercises the independent integer/SSE eightbyte counters.
fn build_struct_mixed() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let f64t = m.types_mut().float(FloatKind::F64);
    let mm = m.types_mut().struct_(vec![i32t, f64t]);
    let sig = m.types_mut().func(vec![mm, mm], mm, false);
    let f = m.declare_function(syms.intern("addM"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let ai_p = b.struct_field(a, mm, 0);
        let ai = b.load(i32t, ai_p, 4);
        let ad_p = b.struct_field(a, mm, 1);
        let ad = b.load(f64t, ad_p, 8);
        let bi_p = b.struct_field(bb, mm, 0);
        let bi = b.load(i32t, bi_p, 4);
        let bd_p = b.struct_field(bb, mm, 1);
        let bd = b.load(f64t, bd_p, 8);
        let si = b.add(ai, bi, Flags::NONE);
        let sd = b.bin(BinOp::FAdd, ad, bd, Flags::NONE);
        let r = b.alloca(mm);
        let ri = b.struct_field(r, mm, 0);
        b.store(i32t, ri, si, 4);
        let rd = b.struct_field(r, mm, 1);
        b.store(f64t, rd, sd, 8);
        b.ret(Some(r));
    }
    (m, syms)
}

#[test]
fn run_struct_mixed() {
    let (m, syms) = build_struct_mixed();
    let c = r#"
        struct M { int i; double d; };
        struct M addM(struct M, struct M);
        int main(void) {
            struct M r = addM((struct M){3, 1.5}, (struct M){4, 2.25});
            if (r.i != 7 || r.d != 3.75) return 1;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "struct_mixed", c) {
        Some(code) => assert_eq!(code, 0, "addM mixed struct mismatch vs gcc (exit {code})"),
        None => eprintln!("skipping run_struct_mixed: no C compiler"),
    }
}

/// `struct Big { long a, b, c; }` (24 bytes > 16 → MEMORY): `addBig` returns the
/// field-wise sum. Both arguments are passed on the stack and the result uses a
/// hidden `sret` pointer (in `rdi`, returned in `rax`).
fn build_struct_big() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let big = m.types_mut().struct_(vec![i64t, i64t, i64t]);
    let sig = m.types_mut().func(vec![big, big], big, false);
    let f = m.declare_function(syms.intern("addBig"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let r = b.alloca(big);
        for k in 0..3u32 {
            let ap = b.struct_field(a, big, k);
            let av = b.load(i64t, ap, 8);
            let bp = b.struct_field(bb, big, k);
            let bv = b.load(i64t, bp, 8);
            let s = b.add(av, bv, Flags::NONE);
            let rp = b.struct_field(r, big, k);
            b.store(i64t, rp, s, 8);
        }
        b.ret(Some(r));
    }
    (m, syms)
}

#[test]
fn run_struct_big() {
    let (m, syms) = build_struct_big();
    let c = r#"
        struct Big { long a, b, c; };
        struct Big addBig(struct Big, struct Big);
        int main(void) {
            struct Big r = addBig((struct Big){1, 2, 3}, (struct Big){10, 20, 30});
            if (r.a != 11 || r.b != 22 || r.c != 33) return 1;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "struct_big", c) {
        Some(code) => assert_eq!(code, 0, "addBig large-struct (sret) mismatch vs gcc (exit {code})"),
        None => eprintln!("skipping run_struct_big: no C compiler"),
    }
}

/// Caller side, small struct: LF `double_it(struct P p)` computes `cadd(p, p)`
/// (a gcc-compiled callee that returns `struct P`) and returns it. Exercises
/// LF passing a struct argument in registers and reclaiming a register-returned
/// struct result, plus returning it in turn.
fn build_struct_caller_small() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let p = m.types_mut().struct_(vec![i32t, i32t]);
    let sig = m.types_mut().func(vec![p, p], p, false);
    let cadd = m.declare_function(syms.intern("cadd"), sig); // defined in C
    let sig1 = m.types_mut().func(vec![p], p, false);
    let f = m.declare_function(syms.intern("double_it"), sig1);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let pv = b.param(entry, 0);
        let cref = b.func_ref(cadd);
        let r = b.call(cref, &[pv, pv], p).expect("struct-returning call");
        b.ret(Some(r));
    }
    (m, syms)
}

#[test]
fn run_struct_caller_small() {
    let (m, syms) = build_struct_caller_small();
    let c = r#"
        struct P { int x, y; };
        struct P cadd(struct P a, struct P b) { return (struct P){a.x + b.x, a.y + b.y}; }
        struct P double_it(struct P);
        int main(void) {
            struct P r = double_it((struct P){3, 4});
            if (r.x != 6 || r.y != 8) return 1;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "struct_caller_small", c) {
        Some(code) => assert_eq!(code, 0, "LF struct caller/return mismatch vs gcc (exit {code})"),
        None => eprintln!("skipping run_struct_caller_small: no C compiler"),
    }
}

/// Caller side, large struct: LF `double_big(struct Big b)` computes
/// `cadd_big(b, b)` (gcc callee returning `struct Big`) and returns it.
/// Exercises the caller allocating an `sret` return slot, passing two MEMORY
/// arguments on the outgoing stack, and forwarding the sret result.
fn build_struct_caller_big() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let big = m.types_mut().struct_(vec![i64t, i64t, i64t]);
    let sig = m.types_mut().func(vec![big, big], big, false);
    let cadd = m.declare_function(syms.intern("cadd_big"), sig); // defined in C
    let sig1 = m.types_mut().func(vec![big], big, false);
    let f = m.declare_function(syms.intern("double_big"), sig1);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let pv = b.param(entry, 0);
        let cref = b.func_ref(cadd);
        let r = b.call(cref, &[pv, pv], big).expect("struct-returning call");
        b.ret(Some(r));
    }
    (m, syms)
}

#[test]
fn run_struct_caller_big() {
    let (m, syms) = build_struct_caller_big();
    let c = r#"
        struct Big { long a, b, c; };
        struct Big cadd_big(struct Big a, struct Big b) {
            return (struct Big){a.a + b.a, a.b + b.b, a.c + b.c};
        }
        struct Big double_big(struct Big);
        int main(void) {
            struct Big r = double_big((struct Big){1, 2, 3});
            if (r.a != 2 || r.b != 4 || r.c != 6) return 1;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "struct_caller_big", c) {
        Some(code) => assert_eq!(code, 0, "LF sret caller/forward mismatch vs gcc (exit {code})"),
        None => eprintln!("skipping run_struct_caller_big: no C compiler"),
    }
}

/// Struct argument mixed with scalar arguments that exhaust the integer
/// registers: `mixfn(int a,b,c,d,e, struct P p, int f)` — `a..e` fill
/// `rdi..r8`, `p` (one INTEGER eightbyte) takes `r9`, and `f` overflows onto the
/// stack. Returns `a+b+c+d+e + p.x + p.y + f`.
fn build_struct_scalar_mix() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let p = m.types_mut().struct_(vec![i32t, i32t]);
    let sig = m.types_mut().func(vec![i32t, i32t, i32t, i32t, i32t, p, i32t], i32t, false);
    let f = m.declare_function(syms.intern("mixfn"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let c1 = b.param(entry, 1);
        let c2 = b.param(entry, 2);
        let c3 = b.param(entry, 3);
        let e = b.param(entry, 4);
        let pv = b.param(entry, 5);
        let fv = b.param(entry, 6);
        let mut acc = b.add(a, c1, Flags::NONE);
        acc = b.add(acc, c2, Flags::NONE);
        acc = b.add(acc, c3, Flags::NONE);
        acc = b.add(acc, e, Flags::NONE);
        let px_p = b.struct_field(pv, p, 0);
        let px = b.load(i32t, px_p, 4);
        let py_p = b.struct_field(pv, p, 1);
        let py = b.load(i32t, py_p, 4);
        acc = b.add(acc, px, Flags::NONE);
        acc = b.add(acc, py, Flags::NONE);
        acc = b.add(acc, fv, Flags::NONE);
        b.ret(Some(acc));
    }
    (m, syms)
}

#[test]
fn run_struct_scalar_mix() {
    let (m, syms) = build_struct_scalar_mix();
    let c = r#"
        struct P { int x, y; };
        int mixfn(int, int, int, int, int, struct P, int);
        int main(void) {
            int r = mixfn(1, 2, 3, 4, 5, (struct P){6, 7}, 8);
            if (r != 36) return 1;               /* 1+2+3+4+5 + 6+7 + 8 */
            if (mixfn(0, 0, 0, 0, 0, (struct P){0, 0}, 100) != 100) return 2;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "struct_scalar_mix", c) {
        Some(code) => assert_eq!(code, 0, "struct+scalar register-exhaustion mismatch vs gcc (exit {code})"),
        None => eprintln!("skipping run_struct_scalar_mix: no C compiler"),
    }
}

// ===========================================================================
// Variadic functions (System V AMD64)
// ===========================================================================

/// Emit an integer `va_arg` (reading one `i32`) against a `va_list` at `vl`,
/// branching on `gp_offset < 48` between the register save area and the overflow
/// area, and jumping to `cont` with `[acc, i, value]`. Helper for the loop body.
///
/// This mirrors exactly the sequence the C frontend's `va_arg` expands to,
/// exercising the backend's register-save-area / overflow-area addresses.
fn build_va_sum() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let i64t = m.types_mut().int(64);
    let ptrt = m.types_mut().ptr();
    let vlty = m.types_mut().struct_(vec![i32t, i32t, ptrt, ptrt]); // 24 bytes

    // Frontend hooks: external `ptr @name()`.
    let hook_sig = m.types_mut().func(vec![], ptrt, false);
    let reg_save = m.declare_function(syms.intern("__lf_va_reg_save_area"), hook_sig);
    let overflow = m.declare_function(syms.intern("__lf_va_overflow_area"), hook_sig);

    // int sum(int n, ...)
    let sig = m.types_mut().func(vec![i32t], i32t, true);
    let sum = m.declare_function(syms.intern("sum"), sig);
    {
        let mut b = m.build(sum);
        let entry = b.create_entry_block();
        let n = b.param(entry, 0);
        let vl = b.alloca(vlty);
        let o0 = b.const_i64(i64t, 0);
        let o4 = b.const_i64(i64t, 4);
        let o8 = b.const_i64(i64t, 8);
        let o16 = b.const_i64(i64t, 16);

        // va_start: gp_offset = 8 (one named int arg), fp_offset = 48 (no fp).
        let gp_ptr = b.ptr_add(vl, o0, true);
        let v8 = b.const_i64(i32t, 8);
        b.store(i32t, gp_ptr, v8, 4);
        let fp_ptr = b.ptr_add(vl, o4, true);
        let v48 = b.const_i64(i32t, 48);
        b.store(i32t, fp_ptr, v48, 4);
        let rsa_ref = b.func_ref(reg_save);
        let rsa = b.call(rsa_ref, &[], ptrt).unwrap();
        let rsa_slot = b.ptr_add(vl, o16, true);
        b.store(ptrt, rsa_slot, rsa, 8);
        let ov_ref = b.func_ref(overflow);
        let ov = b.call(ov_ref, &[], ptrt).unwrap();
        let ov_slot = b.ptr_add(vl, o8, true);
        b.store(ptrt, ov_slot, ov, 8);

        let header = b.create_block(&[i32t, i32t]);
        let body = b.create_block(&[i32t, i32t]);
        let from_reg = b.create_block(&[i32t, i32t, i32t]);
        let from_ov = b.create_block(&[i32t, i32t]);
        let cont = b.create_block(&[i32t, i32t, i32t]);
        let exit = b.create_block(&[i32t]);

        let zero = b.const_i64(i32t, 0);
        b.br(header, &[zero, zero]);

        b.switch_to(header);
        let acc_h = b.param(header, 0);
        let i_h = b.param(header, 1);
        let cond = b.icmp(IntPred::Slt, i_h, n);
        b.cond_br(cond, body, &[acc_h, i_h], exit, &[acc_h]);

        b.switch_to(body);
        let acc_b = b.param(body, 0);
        let i_b = b.param(body, 1);
        let gp_ptr2 = b.ptr_add(vl, o0, true);
        let gp = b.load(i32t, gp_ptr2, 4);
        let c48 = b.const_i64(i32t, 48);
        let is_reg = b.icmp(IntPred::Ult, gp, c48);
        b.cond_br(is_reg, from_reg, &[acc_b, i_b, gp], from_ov, &[acc_b, i_b]);

        // register path: addr = reg_save_area + gp_offset; gp_offset += 8.
        b.switch_to(from_reg);
        let acc_r = b.param(from_reg, 0);
        let i_r = b.param(from_reg, 1);
        let gp_r = b.param(from_reg, 2);
        let rsa_slot2 = b.ptr_add(vl, o16, true);
        let rsa2 = b.load(ptrt, rsa_slot2, 8);
        let gp64 = b.cast(CastOp::ZExt, gp_r, i64t);
        let addr_r = b.ptr_add(rsa2, gp64, true);
        let eight = b.const_i64(i32t, 8);
        let new_gp = b.add(gp_r, eight, Flags::NONE);
        let gp_ptr3 = b.ptr_add(vl, o0, true);
        b.store(i32t, gp_ptr3, new_gp, 4);
        let v_r = b.load(i32t, addr_r, 4);
        b.br(cont, &[acc_r, i_r, v_r]);

        // overflow path: addr = overflow_arg_area; overflow_arg_area += 8.
        b.switch_to(from_ov);
        let acc_o = b.param(from_ov, 0);
        let i_o = b.param(from_ov, 1);
        let ov_slot2 = b.ptr_add(vl, o8, true);
        let ov2 = b.load(ptrt, ov_slot2, 8);
        let new_ov = b.ptr_add(ov2, o8, true);
        let ov_slot3 = b.ptr_add(vl, o8, true);
        b.store(ptrt, ov_slot3, new_ov, 8);
        let v_o = b.load(i32t, ov2, 4);
        b.br(cont, &[acc_o, i_o, v_o]);

        b.switch_to(cont);
        let acc_c = b.param(cont, 0);
        let i_c = b.param(cont, 1);
        let v_c = b.param(cont, 2);
        let new_acc = b.add(acc_c, v_c, Flags::NONE);
        let one = b.const_i64(i32t, 1);
        let new_i = b.add(i_c, one, Flags::NONE);
        b.br(header, &[new_acc, new_i]);

        b.switch_to(exit);
        let acc_e = b.param(exit, 0);
        b.ret(Some(acc_e));
    }
    (m, syms)
}

/// LF-compiled `int sum(int n, ...)` (register save area + overflow area,
/// `va_arg` hand-lowered via the frame-address hooks) called from a gcc driver,
/// which passes the varargs per SysV and sets `al`. Covers both the ≤6-register
/// case and a >6-vararg case that spills to the overflow/stack area.
#[test]
fn run_va_sum() {
    let (m, syms) = build_va_sum();
    let c = r#"
        extern int sum(int, ...);
        int main(void) {
            if (sum(4, 10, 20, 30, 40) != 100) return 1;   /* all in registers */
            if (sum(9, 1,2,3,4,5,6,7,8,9) != 45) return 2; /* spills to overflow */
            if (sum(0) != 0) return 3;                      /* no varargs */
            if (sum(1, -7) != -7) return 4;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "va_sum", c) {
        Some(code) => assert_eq!(code, 0, "LF variadic int-sum mismatch vs gcc (exit {code})"),
        None => eprintln!("skipping run_va_sum: no C compiler"),
    }
}

/// LF-compiled `double dsum(int n, ...)` summing `n` `double` varargs — exercises
/// the SSE half of the register save area (`xmm0..7` at `fp_offset`) and the `al`
/// vector count gcc sets at the call.
fn build_va_dsum() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let i64t = m.types_mut().int(64);
    let f64t = m.types_mut().float(FloatKind::F64);
    let ptrt = m.types_mut().ptr();
    let vlty = m.types_mut().struct_(vec![i32t, i32t, ptrt, ptrt]);

    let hook_sig = m.types_mut().func(vec![], ptrt, false);
    let reg_save = m.declare_function(syms.intern("__lf_va_reg_save_area"), hook_sig);
    let overflow = m.declare_function(syms.intern("__lf_va_overflow_area"), hook_sig);

    let sig = m.types_mut().func(vec![i32t], f64t, true);
    let dsum = m.declare_function(syms.intern("dsum"), sig);
    {
        let mut b = m.build(dsum);
        let entry = b.create_entry_block();
        let n = b.param(entry, 0);
        let vl = b.alloca(vlty);
        let o0 = b.const_i64(i64t, 0);
        let o4 = b.const_i64(i64t, 4);
        let o8 = b.const_i64(i64t, 8);
        let o16 = b.const_i64(i64t, 16);

        // va_start: gp_offset = 8 (named int `n`), fp_offset = 48 (no named fp).
        let gp_ptr = b.ptr_add(vl, o0, true);
        let v8 = b.const_i64(i32t, 8);
        b.store(i32t, gp_ptr, v8, 4);
        let fp_ptr = b.ptr_add(vl, o4, true);
        let v48 = b.const_i64(i32t, 48);
        b.store(i32t, fp_ptr, v48, 4);
        let rsa_ref = b.func_ref(reg_save);
        let rsa = b.call(rsa_ref, &[], ptrt).unwrap();
        let rsa_slot = b.ptr_add(vl, o16, true);
        b.store(ptrt, rsa_slot, rsa, 8);
        let ov_ref = b.func_ref(overflow);
        let ov = b.call(ov_ref, &[], ptrt).unwrap();
        let ov_slot = b.ptr_add(vl, o8, true);
        b.store(ptrt, ov_slot, ov, 8);

        let header = b.create_block(&[f64t, i32t]);
        let body = b.create_block(&[f64t, i32t]);
        let from_reg = b.create_block(&[f64t, i32t, i32t]);
        let from_ov = b.create_block(&[f64t, i32t]);
        let cont = b.create_block(&[f64t, i32t, f64t]);
        let exit = b.create_block(&[f64t]);

        let z = b.const_float(f64t, FloatBits::F64(0.0f64.to_bits()));
        let zero_i = b.const_i64(i32t, 0);
        b.br(header, &[z, zero_i]);

        b.switch_to(header);
        let acc_h = b.param(header, 0);
        let i_h = b.param(header, 1);
        let cond = b.icmp(IntPred::Slt, i_h, n);
        b.cond_br(cond, body, &[acc_h, i_h], exit, &[acc_h]);

        b.switch_to(body);
        let acc_b = b.param(body, 0);
        let i_b = b.param(body, 1);
        let fp_ptr2 = b.ptr_add(vl, o4, true);
        let fp = b.load(i32t, fp_ptr2, 4);
        let c176 = b.const_i64(i32t, 176);
        let is_reg = b.icmp(IntPred::Ult, fp, c176);
        b.cond_br(is_reg, from_reg, &[acc_b, i_b, fp], from_ov, &[acc_b, i_b]);

        // register path: addr = reg_save_area + fp_offset; fp_offset += 16.
        b.switch_to(from_reg);
        let acc_r = b.param(from_reg, 0);
        let i_r = b.param(from_reg, 1);
        let fp_r = b.param(from_reg, 2);
        let rsa_slot2 = b.ptr_add(vl, o16, true);
        let rsa2 = b.load(ptrt, rsa_slot2, 8);
        let fp64 = b.cast(CastOp::ZExt, fp_r, i64t);
        let addr_r = b.ptr_add(rsa2, fp64, true);
        let sixteen = b.const_i64(i32t, 16);
        let new_fp = b.add(fp_r, sixteen, Flags::NONE);
        let fp_ptr3 = b.ptr_add(vl, o4, true);
        b.store(i32t, fp_ptr3, new_fp, 4);
        let v_r = b.load(f64t, addr_r, 8);
        b.br(cont, &[acc_r, i_r, v_r]);

        // overflow path: addr = overflow_arg_area; overflow_arg_area += 8.
        b.switch_to(from_ov);
        let acc_o = b.param(from_ov, 0);
        let i_o = b.param(from_ov, 1);
        let ov_slot2 = b.ptr_add(vl, o8, true);
        let ov2 = b.load(ptrt, ov_slot2, 8);
        let new_ov = b.ptr_add(ov2, o8, true);
        let ov_slot3 = b.ptr_add(vl, o8, true);
        b.store(ptrt, ov_slot3, new_ov, 8);
        let v_o = b.load(f64t, ov2, 8);
        b.br(cont, &[acc_o, i_o, v_o]);

        b.switch_to(cont);
        let acc_c = b.param(cont, 0);
        let i_c = b.param(cont, 1);
        let v_c = b.param(cont, 2);
        let new_acc = b.bin(BinOp::FAdd, acc_c, v_c, Flags::NONE);
        let one = b.const_i64(i32t, 1);
        let new_i = b.add(i_c, one, Flags::NONE);
        b.br(header, &[new_acc, new_i]);

        b.switch_to(exit);
        let acc_e = b.param(exit, 0);
        b.ret(Some(acc_e));
    }
    (m, syms)
}

/// LF-compiled `double dsum(int n, ...)` (SSE register save area + `al`) called
/// from a gcc driver passing `double` varargs.
#[test]
fn run_va_dsum() {
    let (m, syms) = build_va_dsum();
    let c = r#"
        extern double dsum(int, ...);
        int main(void) {
            double r = dsum(3, 1.5, 2.5, 3.0);
            if (r != 7.0) return 1;
            if (dsum(0) != 0.0) return 2;
            if (dsum(5, 1.0, 2.0, 4.0, 8.0, 16.0) != 31.0) return 3;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "va_dsum", c) {
        Some(code) => assert_eq!(code, 0, "LF variadic double-sum mismatch vs gcc (exit {code})"),
        None => eprintln!("skipping run_va_dsum: no C compiler"),
    }
}

/// LF as the *caller* of a variadic function: `long lf_call_ivsum()` calls the
/// gcc-provided `long ivsum(int n, ...)` with integer varargs (`al` must be 0),
/// and `double lf_call_dvsum()` calls `double dvsum(int n, ...)` with `double`
/// varargs (`al` must be 3). Verifies the backend sets the SSE vector count `al`
/// at a variadic call site.
fn build_va_caller() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let i64t = m.types_mut().int(64);
    let f64t = m.types_mut().float(FloatKind::F64);

    // Variadic callees provided by the C driver.
    let ivsum_sig = m.types_mut().func(vec![i32t], i64t, true);
    let ivsum = m.declare_function(syms.intern("ivsum"), ivsum_sig);
    let dvsum_sig = m.types_mut().func(vec![i32t], f64t, true);
    let dvsum = m.declare_function(syms.intern("dvsum"), dvsum_sig);

    // long lf_call_ivsum() = ivsum(3, 100, 200, 300)
    let ret_i_sig = m.types_mut().func(vec![], i64t, false);
    let call_i = m.declare_function(syms.intern("lf_call_ivsum"), ret_i_sig);
    {
        let mut b = m.build(call_i);
        let _e = b.create_entry_block();
        let n = b.const_i64(i32t, 3);
        let a = b.const_i64(i64t, 100);
        let c = b.const_i64(i64t, 200);
        let d = b.const_i64(i64t, 300);
        let r = b.func_ref(ivsum);
        let res = b.call(r, &[n, a, c, d], i64t).unwrap();
        b.ret(Some(res));
    }

    // double lf_call_dvsum() = dvsum(3, 1.5, 2.5, 3.0)
    let ret_d_sig = m.types_mut().func(vec![], f64t, false);
    let call_d = m.declare_function(syms.intern("lf_call_dvsum"), ret_d_sig);
    {
        let mut b = m.build(call_d);
        let _e = b.create_entry_block();
        let n = b.const_i64(i32t, 3);
        let a = b.const_float(f64t, FloatBits::F64(1.5f64.to_bits()));
        let c = b.const_float(f64t, FloatBits::F64(2.5f64.to_bits()));
        let d = b.const_float(f64t, FloatBits::F64(3.0f64.to_bits()));
        let r = b.func_ref(dvsum);
        let res = b.call(r, &[n, a, c, d], f64t).unwrap();
        b.ret(Some(res));
    }
    (m, syms)
}

#[test]
fn run_va_caller() {
    let (m, syms) = build_va_caller();
    let c = r#"
        #include <stdarg.h>
        long ivsum(int n, ...) {
            va_list ap; va_start(ap, n);
            long s = 0;
            for (int i = 0; i < n; i++) s += va_arg(ap, long);
            va_end(ap);
            return s;
        }
        double dvsum(int n, ...) {
            va_list ap; va_start(ap, n);
            double s = 0;
            for (int i = 0; i < n; i++) s += va_arg(ap, double);
            va_end(ap);
            return s;
        }
        extern long lf_call_ivsum(void);
        extern double lf_call_dvsum(void);
        int main(void) {
            if (lf_call_ivsum() != 600) return 1;
            if (lf_call_dvsum() != 7.0) return 2;
            return 0;
        }
    "#;
    match compile_link_run(&m, &syms, "va_caller", c) {
        Some(code) => assert_eq!(code, 0, "LF-as-caller variadic mismatch vs gcc (exit {code})"),
        None => eprintln!("skipping run_va_caller: no C compiler"),
    }
}
