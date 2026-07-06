//! Tests for the AArch64 backend.
//!
//! Because this host cannot *execute* AArch64 code, correctness rests on three
//! tiers:
//!
//! - **Golden byte encodings** — instructions hand-verified from the ARM A64
//!   encodings, asserted exactly. These need no toolchain.
//! - **Differential encoding vs `llvm-mc`** — the primary encoder gate: for a
//!   broad corpus, assemble the equivalent A64 asm with
//!   `llvm-mc --triple=aarch64 --show-encoding` and assert our bytes match.
//!   Skipped if `llvm-mc` is absent.
//! - **A64-MIR interpretation** — lower real IR functions and run them on the
//!   [`super::interp`] MIR interpreter, asserting the computed values. This
//!   proves instruction *selection* is semantically right even though the host
//!   cannot run ARM.

use super::encode::*;
use super::interp;
use super::isel::AArch64Target;
use crate::ir::inst::{BinOp, CastOp, Flags, FloatPred, IntPred};
use crate::ir::types::FloatKind;
use crate::ir::value::FloatBits;
use crate::ir::{FuncId, Module};
use crate::support::StrInterner;

use puremp::Int;

// ===========================================================================
// Golden byte encodings (verified from the ARM A64 encodings)
// ===========================================================================

#[test]
fn golden_add_sub_reg() {
    // add x0, x1, x2 = 20 00 02 8b ; sub x3, x4, x5 = 83 00 05 cb
    assert_eq!(add_reg(1, 0, 1, 2).to_le_bytes(), [0x20, 0x00, 0x02, 0x8b]);
    assert_eq!(sub_reg(1, 3, 4, 5).to_le_bytes(), [0x83, 0x00, 0x05, 0xcb]);
    // 32-bit add w0, w1, w2 = 20 00 02 0b
    assert_eq!(add_reg(0, 0, 1, 2).to_le_bytes(), [0x20, 0x00, 0x02, 0x0b]);
}

#[test]
fn golden_logical_and_mov() {
    assert_eq!(and_reg(1, 0, 1, 2).to_le_bytes(), [0x20, 0x00, 0x02, 0x8a]);
    assert_eq!(orr_reg(1, 0, 1, 2).to_le_bytes(), [0x20, 0x00, 0x02, 0xaa]);
    assert_eq!(eor_reg(1, 0, 1, 2).to_le_bytes(), [0x20, 0x00, 0x02, 0xca]);
    // mov x0, xzr = e0 03 1f aa  (orr x0, xzr, xzr)
    assert_eq!(mov_reg(1, 0, 31).to_le_bytes(), [0xe0, 0x03, 0x1f, 0xaa]);
    // mov x0, x1 = e0 03 01 aa
    assert_eq!(mov_reg(1, 0, 1).to_le_bytes(), [0xe0, 0x03, 0x01, 0xaa]);
}

#[test]
fn golden_addsub_imm_and_mul() {
    assert_eq!(add_imm(1, 0, 1, 10).to_le_bytes(), [0x20, 0x28, 0x00, 0x91]);
    assert_eq!(sub_imm(1, 0, 1, 10).to_le_bytes(), [0x20, 0x28, 0x00, 0xd1]);
    // mul x0, x1, x2 = madd x0,x1,x2,xzr = 20 7c 02 9b
    assert_eq!(madd(1, 0, 1, 2, 31).to_le_bytes(), [0x20, 0x7c, 0x02, 0x9b]);
    // madd x0,x1,x2,x3 = 20 0c 02 9b ; msub = 20 8c 02 9b
    assert_eq!(madd(1, 0, 1, 2, 3).to_le_bytes(), [0x20, 0x0c, 0x02, 0x9b]);
    assert_eq!(msub(1, 0, 1, 2, 3).to_le_bytes(), [0x20, 0x8c, 0x02, 0x9b]);
}

#[test]
fn golden_div_and_shift() {
    assert_eq!(sdiv(1, 0, 1, 2).to_le_bytes(), [0x20, 0x0c, 0xc2, 0x9a]);
    assert_eq!(udiv(1, 0, 1, 2).to_le_bytes(), [0x20, 0x08, 0xc2, 0x9a]);
    assert_eq!(lsl_imm(1, 0, 1, 3).to_le_bytes(), [0x20, 0xf0, 0x7d, 0xd3]);
    assert_eq!(lsr_imm(1, 0, 1, 3).to_le_bytes(), [0x20, 0xfc, 0x43, 0xd3]);
    assert_eq!(asr_imm(1, 0, 1, 3).to_le_bytes(), [0x20, 0xfc, 0x43, 0x93]);
    assert_eq!(lslv(1, 0, 1, 2).to_le_bytes(), [0x20, 0x20, 0xc2, 0x9a]);
}

#[test]
fn golden_cmp_cset_csel() {
    // cmp x1, x2 = subs xzr, x1, x2 = 3f 00 02 eb
    assert_eq!(subs_reg(1, 31, 1, 2).to_le_bytes(), [0x3f, 0x00, 0x02, 0xeb]);
    // cset x0, eq = e0 17 9f 9a ; cset x0, lt = e0 a7 9f 9a
    assert_eq!(cset(1, 0, 0x0).to_le_bytes(), [0xe0, 0x17, 0x9f, 0x9a]);
    assert_eq!(cset(1, 0, 0xB).to_le_bytes(), [0xe0, 0xa7, 0x9f, 0x9a]);
    // csel x0, x1, x2, eq = 20 00 82 9a
    assert_eq!(csel(1, 0, 1, 2, 0x0).to_le_bytes(), [0x20, 0x00, 0x82, 0x9a]);
}

#[test]
fn golden_movwide_and_memory() {
    assert_eq!(movz(1, 0, 0x1234, 0).to_le_bytes(), [0x80, 0x46, 0x82, 0xd2]);
    assert_eq!(movz(1, 0, 0x1234, 1).to_le_bytes(), [0x80, 0x46, 0xa2, 0xd2]);
    assert_eq!(movk(1, 0, 0x5678, 2).to_le_bytes(), [0x00, 0xcf, 0xca, 0xf2]);
    assert_eq!(movn(1, 0, 0, 0).to_le_bytes(), [0x00, 0x00, 0x80, 0x92]);
    // ldr x0,[x1] = 20 00 40 f9 ; ldr x0,[x1,#16] = 20 08 40 f9
    assert_eq!(ldst_uimm(true, 3, 0, 1, 0).to_le_bytes(), [0x20, 0x00, 0x40, 0xf9]);
    assert_eq!(ldst_uimm(true, 3, 0, 1, 2).to_le_bytes(), [0x20, 0x08, 0x40, 0xf9]);
    // str x0,[x1] = 20 00 00 f9 ; ldrb w0,[x1] = 20 00 40 39
    assert_eq!(ldst_uimm(false, 3, 0, 1, 0).to_le_bytes(), [0x20, 0x00, 0x00, 0xf9]);
    assert_eq!(ldst_uimm(true, 0, 0, 1, 0).to_le_bytes(), [0x20, 0x00, 0x40, 0x39]);
}

#[test]
fn golden_branches_and_frame() {
    // ret = c0 03 5f d6 ; blr x0 = 00 00 3f d6 ; brk #1 = 20 00 20 d4
    assert_eq!(ret(30).to_le_bytes(), [0xc0, 0x03, 0x5f, 0xd6]);
    assert_eq!(blr(0).to_le_bytes(), [0x00, 0x00, 0x3f, 0xd6]);
    assert_eq!(brk(1).to_le_bytes(), [0x20, 0x00, 0x20, 0xd4]);
    // b #0 = 00 00 00 14 ; bl #0 = 00 00 00 94 ; adrp x0,#0 = 00 00 00 90
    assert_eq!(b_uncond(0).to_le_bytes(), [0x00, 0x00, 0x00, 0x14]);
    assert_eq!(bl(0).to_le_bytes(), [0x00, 0x00, 0x00, 0x94]);
    assert_eq!(adrp(0).to_le_bytes(), [0x00, 0x00, 0x00, 0x90]);
    // stp x29,x30,[sp,#-16]! = fd 7b bf a9 ; ldp x29,x30,[sp],#16 = fd 7b c1 a8
    assert_eq!(stp_pre(29, 30, 31, -2).to_le_bytes(), [0xfd, 0x7b, 0xbf, 0xa9]);
    assert_eq!(ldp_post(29, 30, 31, 2).to_le_bytes(), [0xfd, 0x7b, 0xc1, 0xa8]);
}

// ===========================================================================
// Differential encoding vs llvm-mc (the primary encoder gate)
// ===========================================================================

/// Assemble one A64 instruction with `llvm-mc --show-encoding`, returning its
/// bytes. `None` when `llvm-mc` is unavailable.
fn llvm_mc(asm: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let mut child = std::process::Command::new("llvm-mc")
        .arg("--triple=aarch64")
        .arg("--show-encoding")
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
    // Find `encoding: [0x..,0x..,..]` and parse the bytes.
    let start = text.find("encoding: [")? + "encoding: [".len();
    let end = text[start..].find(']')? + start;
    let mut bytes = Vec::new();
    for tok in text[start..end].split(',') {
        let tok = tok.trim().trim_start_matches("0x");
        bytes.push(u8::from_str_radix(tok, 16).ok()?);
    }
    Some(bytes)
}

#[test]
fn differential_encoding_matches_llvm_mc() {
    // A corpus covering every A64 instruction form the isel emits. Each pair is
    // (our 32-bit word, the equivalent A64 assembly).
    let corpus: Vec<(u32, &str)> = vec![
        // data-processing register
        (add_reg(1, 0, 1, 2), "add x0, x1, x2"),
        (add_reg(0, 0, 1, 2), "add w0, w1, w2"),
        (sub_reg(1, 3, 4, 5), "sub x3, x4, x5"),
        (and_reg(1, 7, 8, 9), "and x7, x8, x9"),
        (orr_reg(1, 10, 11, 12), "orr x10, x11, x12"),
        (eor_reg(1, 0, 1, 2), "eor x0, x1, x2"),
        (mov_reg(1, 0, 1), "mov x0, x1"),
        (mov_reg(1, 5, 31), "mov x5, xzr"),
        // data-processing immediate
        (add_imm(1, 0, 1, 10), "add x0, x1, #10"),
        (add_imm(1, 2, 3, 4095), "add x2, x3, #4095"),
        (sub_imm(1, 0, 1, 10), "sub x0, x1, #10"),
        (add_imm(0, 0, 1, 7), "add w0, w1, #7"),
        // multiply / divide
        (madd(1, 0, 1, 2, 31), "mul x0, x1, x2"),
        (madd(1, 0, 1, 2, 3), "madd x0, x1, x2, x3"),
        (msub(1, 4, 5, 6, 7), "msub x4, x5, x6, x7"),
        (sdiv(1, 0, 1, 2), "sdiv x0, x1, x2"),
        (udiv(1, 0, 1, 2), "udiv x0, x1, x2"),
        (sdiv(0, 0, 1, 2), "sdiv w0, w1, w2"),
        // shifts (immediate + variable)
        (lsl_imm(1, 0, 1, 3), "lsl x0, x1, #3"),
        (lsr_imm(1, 0, 1, 3), "lsr x0, x1, #3"),
        (asr_imm(1, 0, 1, 3), "asr x0, x1, #3"),
        (lsl_imm(0, 0, 1, 5), "lsl w0, w1, #5"),
        (lslv(1, 0, 1, 2), "lslv x0, x1, x2"),
        (lsrv(1, 0, 1, 2), "lsrv x0, x1, x2"),
        (asrv(1, 0, 1, 2), "asrv x0, x1, x2"),
        // compare / conditional set / select
        (subs_reg(1, 31, 1, 2), "cmp x1, x2"),
        (cset(1, 0, 0x0), "cset x0, eq"),
        (cset(1, 0, 0x1), "cset x0, ne"),
        (cset(1, 0, 0xB), "cset x0, lt"),
        (cset(1, 3, 0xC), "cset x3, gt"),
        (csel(1, 0, 1, 2, 0x0), "csel x0, x1, x2, eq"),
        (csel(1, 0, 1, 2, 0x1), "csel x0, x1, x2, ne"),
        // move-wide immediate
        (movz(1, 0, 0x1234, 0), "movz x0, #0x1234"),
        (movz(1, 0, 0x1234, 1), "movz x0, #0x1234, lsl #16"),
        (movk(1, 0, 0x5678, 2), "movk x0, #0x5678, lsl #32"),
        (movn(1, 0, 0, 0), "movn x0, #0"),
        (movz(0, 3, 5, 0), "movz w3, #5"),
        // loads / stores (unsigned offset)
        (ldst_uimm(true, 3, 0, 1, 0), "ldr x0, [x1]"),
        (ldst_uimm(true, 3, 0, 1, 2), "ldr x0, [x1, #16]"),
        (ldst_uimm(false, 3, 0, 1, 0), "str x0, [x1]"),
        (ldst_uimm(true, 2, 0, 1, 0), "ldr w0, [x1]"),
        (ldst_uimm(false, 2, 0, 1, 0), "str w0, [x1]"),
        (ldst_uimm(true, 0, 0, 1, 0), "ldrb w0, [x1]"),
        (ldst_uimm(false, 0, 0, 1, 0), "strb w0, [x1]"),
        (ldst_uimm(true, 1, 0, 1, 0), "ldrh w0, [x1]"),
        // branches / calls / traps
        (ret(30), "ret"),
        (blr(0), "blr x0"),
        (brk(1), "brk #1"),
        (b_uncond(0), "b #0"),
        (bl(0), "bl #0"),
        (b_cond(0x0, 0), "b.eq #0"),
        (b_cond(0x1, 0), "b.ne #0"),
        (cbz(1, 0, 0, true), "cbnz x0, #0"),
        (cbz(1, 0, 0, false), "cbz x0, #0"),
        (adrp(0), "adrp x0, #0"),
        // stack-pointer frame idioms
        (stp_pre(29, 30, 31, -2), "stp x29, x30, [sp, #-16]!"),
        (ldp_post(29, 30, 31, 2), "ldp x29, x30, [sp], #16"),
        (sub_imm(1, 31, 31, 16), "sub sp, sp, #16"),
        (add_imm(1, 31, 31, 16), "add sp, sp, #16"),
        (add_imm(1, 29, 31, 0), "mov x29, sp"),
    ];

    if llvm_mc("ret").is_none() {
        eprintln!("skipping differential_encoding_matches_llvm_mc: no llvm-mc");
        return;
    }

    let mut checked = 0usize;
    for (word, asm) in &corpus {
        let expected = llvm_mc(asm).unwrap_or_else(|| panic!("llvm-mc failed on `{asm}`"));
        assert_eq!(
            word.to_le_bytes().to_vec(),
            expected,
            "encoding mismatch for `{asm}`: ours={:02x?} llvm={:02x?}",
            word.to_le_bytes(),
            expected
        );
        checked += 1;
    }
    assert_eq!(checked, corpus.len(), "every corpus instruction is differentially checked");
    assert!(checked >= 55, "the corpus stays broad ({checked} instructions)");
    eprintln!("differential encoder gate: {checked} instructions matched llvm-mc");
}

// ===========================================================================
// IR fixtures + A64-MIR interpretation (isel correctness without execution)
// ===========================================================================

/// Lower every function of `m` to MIR (indexed by `FuncId`), for the interpreter.
fn lower_all(m: &Module) -> (AArch64Target, Vec<crate::codegen::mir::MachineFunction>) {
    let target = AArch64Target::new();
    let funcs: Vec<_> =
        (0..m.functions().count()).map(|i| target.select(m, FuncId::from_index(i))).collect();
    (target, funcs)
}

fn i(v: i64) -> Int {
    Int::from_i64(v)
}

/// `lfadd(a, b) = a + b` over `i64`.
fn build_add() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t, i64t], i64t, false);
    let f = m.declare_function(syms.intern("lfadd"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let s = b.add(a, bb, Flags::NONE);
        b.ret(Some(s));
    }
    (m, f)
}

/// `lfmax(a, b)` via a branch diamond passing the larger value as a block arg.
fn build_max() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t, i64t], i64t, false);
    let f = m.declare_function(syms.intern("lfmax"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let then_b = b.create_block(&[]);
        let else_b = b.create_block(&[]);
        let join = b.create_block(&[i64t]);
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
    (m, f)
}

/// `lfsum(n) = 0 + 1 + ... + (n-1)` — a loop with back-edge args.
fn build_loop_sum() -> (Module, FuncId) {
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
        let idx = b.param(header, 1);
        let cond = b.icmp(IntPred::Slt, idx, n);
        b.cond_br(cond, body, &[acc, idx], exit, &[acc]);
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
    (m, f)
}

/// A caller `lfcaller(x) = lfcallee(x) + lfcallee(x)` and `lfcallee(y) = y*3`.
fn build_call() -> (Module, FuncId) {
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
    (m, caller)
}

/// `lfmem(x)`: alloca an i64, store x, load it back, return it.
fn build_mem() -> (Module, FuncId) {
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
    (m, f)
}

/// `lfdivmod(a, b) = (a / b) + (a % b)` — exercises `sdiv` and `sdiv;msub`.
fn build_divmod() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t, i64t], i64t, false);
    let f = m.declare_function(syms.intern("lfdivmod"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let q = b.bin(BinOp::SDiv, a, bb, Flags::NONE);
        let r = b.bin(BinOp::SRem, a, bb, Flags::NONE);
        let s = b.add(q, r, Flags::NONE);
        b.ret(Some(s));
    }
    (m, f)
}

/// `lfbits(x) = ((x << 4) ^ (x >> 1)) & 0xff` — shifts + bitwise (imm materialized).
fn build_bits() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("lfbits"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let four = b.const_i64(i64t, 4);
        let one = b.const_i64(i64t, 1);
        let hi = b.bin(BinOp::Shl, x, four, Flags::NONE);
        let lo = b.bin(BinOp::LShr, x, one, Flags::NONE);
        let xored = b.bin(BinOp::Xor, hi, lo, Flags::NONE);
        let mask = b.const_i64(i64t, 0xff);
        let r = b.bin(BinOp::And, xored, mask, Flags::NONE);
        b.ret(Some(r));
    }
    (m, f)
}

fn eval1(m: &Module, f: FuncId, x: i64) -> Int {
    let (target, funcs) = lower_all(m);
    interp::run(&target, &funcs, f.index(), &[i(x)])
        .expect("interpretation succeeds")
        .expect("function returns a value")
}

fn eval2(m: &Module, f: FuncId, x: i64, y: i64) -> Int {
    let (target, funcs) = lower_all(m);
    interp::run(&target, &funcs, f.index(), &[i(x), i(y)])
        .expect("interpretation succeeds")
        .expect("function returns a value")
}

#[test]
fn interp_add() {
    let (m, f) = build_add();
    assert_eq!(eval2(&m, f, 3, 4), i(7));
    assert_eq!(eval2(&m, f, -2, 10), i(8));
    assert_eq!(eval2(&m, f, 0, 0), i(0));
}

#[test]
fn interp_max() {
    let (m, f) = build_max();
    assert_eq!(eval2(&m, f, 3, 4), i(4));
    assert_eq!(eval2(&m, f, 9, 2), i(9));
    assert_eq!(eval2(&m, f, -1, -5), i(-1));
    assert_eq!(eval2(&m, f, 7, 7), i(7));
}

#[test]
fn interp_loop_sum() {
    let (m, f) = build_loop_sum();
    assert_eq!(eval1(&m, f, 0), i(0));
    assert_eq!(eval1(&m, f, 1), i(0));
    assert_eq!(eval1(&m, f, 5), i(10));
    assert_eq!(eval1(&m, f, 10), i(45));
    assert_eq!(eval1(&m, f, 100), i(4950));
}

#[test]
fn interp_call() {
    let (m, f) = build_call();
    assert_eq!(eval1(&m, f, 2), i(12)); // 2*3 + 2*3
    assert_eq!(eval1(&m, f, 7), i(42)); // 7*3 + 7*3
}

#[test]
fn interp_mem() {
    let (m, f) = build_mem();
    assert_eq!(eval1(&m, f, 0), i(0));
    assert_eq!(eval1(&m, f, 42), i(42));
    assert_eq!(eval1(&m, f, 1234567), i(1234567));
}

#[test]
fn interp_divmod() {
    // Results come back as the 64-bit-masked (unsigned) bit pattern, so compare
    // against the same masking of the mathematical result.
    let m64 = |v: i64| i(v).mod_2k(64);
    let (m, f) = build_divmod();
    assert_eq!(eval2(&m, f, 17, 5), m64(3 + 2)); // 17/5=3, 17%5=2
    assert_eq!(eval2(&m, f, 100, 9), m64(11 + 1)); // 100/9=11, 100%9=1
    assert_eq!(eval2(&m, f, -17, 5), m64(-3 + -2)); // trunc toward zero
}

#[test]
fn interp_bits() {
    let (m, f) = build_bits();
    for x in [0i64, 1, 5, 0xab, 255, 4096] {
        let expected = ((x << 4) ^ (x >> 1)) & 0xff;
        assert_eq!(eval1(&m, f, x), i(expected), "lfbits({x})");
    }
}

// ===========================================================================
// llvm-mc disassembly round-trip + determinism
// ===========================================================================

/// Disassemble comma-separated hex bytes with `llvm-mc --disassemble`.
fn llvm_disasm(bytes: &[u8]) -> Option<String> {
    use std::io::Write;
    let hex: Vec<String> = bytes.iter().map(|b| format!("0x{b:02x}")).collect();
    let input = hex.join(",");
    let mut child = std::process::Command::new("llvm-mc")
        .arg("--triple=aarch64")
        .arg("--disassemble")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(input.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn disasm_round_trip_lfadd() {
    let (m, f) = build_add();
    let syms = {
        let mut s = StrInterner::new();
        s.intern("lfadd");
        s
    };
    let emitted = compile_function(&m, f, &syms);
    let Some(text) = llvm_disasm(&emitted.bytes) else {
        eprintln!("skipping disasm_round_trip_lfadd: no llvm-mc");
        return;
    };
    // The prologue/body/epilogue must decode to the expected A64 idioms.
    for needle in ["stp", "mov", "add", "ldp", "ret"] {
        assert!(text.contains(needle), "expected `{needle}` in disassembly:\n{text}");
    }
}

#[test]
fn compile_module_has_text_and_call_reloc() {
    let (m, _) = build_call();
    let mut syms = StrInterner::new();
    // Rebuild the interner so names resolve (the fixture drops its own).
    syms.intern("lfcallee");
    syms.intern("lfcaller");
    let obj = compile_module(&m, &syms);
    let text = obj.section(crate::mc::object::SectionId::from_index(0));
    assert!(!text.bytes.is_empty(), "text section is non-empty");
    assert!(text.bytes.len().is_multiple_of(4), "A64 code is word-sized");
    // A `bl` to `lfcallee` must have produced a CALL26 relocation.
    let has_call26 = obj
        .relocations()
        .iter()
        .any(|r| r.kind == crate::mc::object::RelocKind::Aarch64Call26);
    assert!(has_call26, "expected an AArch64 CALL26 relocation for the call");
}

#[test]
fn encoding_is_deterministic() {
    let (m, f) = build_loop_sum();
    let syms = {
        let mut s = StrInterner::new();
        s.intern("lfsum");
        s
    };
    let a = compile_function(&m, f, &syms);
    let b = compile_function(&m, f, &syms);
    assert_eq!(a, b, "identical input must yield identical bytes");
}

// ===========================================================================
// Scalar floating-point: golden bytes, differential encoding, interpretation
// ===========================================================================

#[test]
fn golden_fp_encodings() {
    // Verified against the ARM A64 encodings (and cross-checked with llvm-mc).
    // fadd d0, d1, d2 = 20 28 62 1e ; fmul s0, s1, s2 = 20 08 22 1e
    assert_eq!(fadd(1, 0, 1, 2).to_le_bytes(), [0x20, 0x28, 0x62, 0x1e]);
    assert_eq!(fmul(0, 0, 1, 2).to_le_bytes(), [0x20, 0x08, 0x22, 0x1e]);
    // fneg d0, d1 = 20 40 61 1e ; fcmp d0, d1 = 00 20 61 1e
    assert_eq!(fneg(1, 0, 1).to_le_bytes(), [0x20, 0x40, 0x61, 0x1e]);
    assert_eq!(fcmp(1, 0, 1).to_le_bytes(), [0x00, 0x20, 0x61, 0x1e]);
    // fcvt d0, s0 = 00 c0 22 1e ; fcvtzs w0, d0 = 00 00 78 1e
    assert_eq!(fcvt(0, 1, 0, 0).to_le_bytes(), [0x00, 0xc0, 0x22, 0x1e]);
    assert_eq!(fcvtzs(0, 1, 0, 0).to_le_bytes(), [0x00, 0x00, 0x78, 0x1e]);
    // scvtf d0, x0 = 00 00 62 9e ; fmov d0, x1 = 20 00 67 9e
    assert_eq!(scvtf(1, 1, 0, 0).to_le_bytes(), [0x00, 0x00, 0x62, 0x9e]);
    assert_eq!(fmov_from_gpr(1, 1, 0, 1).to_le_bytes(), [0x20, 0x00, 0x67, 0x9e]);
    // ldr d0, [x1] = 20 00 40 fd ; fmov d0, d1 = 20 40 60 1e
    assert_eq!(fp_ldst_uimm(true, 3, 0, 1, 0).to_le_bytes(), [0x20, 0x00, 0x40, 0xfd]);
    assert_eq!(fmov_reg(1, 0, 1).to_le_bytes(), [0x20, 0x40, 0x60, 0x1e]);
    // A v16+ form: fadd d16, d17, d18 = 30 2a 72 1e (exercises the high registers).
    assert_eq!(fadd(1, 16, 17, 18).to_le_bytes(), [0x30, 0x2a, 0x72, 0x1e]);
}

#[test]
fn differential_fp_encoding_matches_llvm_mc() {
    // Every FP instruction form the isel emits, checked against llvm-mc.
    let corpus: Vec<(u32, &str)> = vec![
        // data-processing (2 source), double and single
        (fadd(1, 0, 1, 2), "fadd d0, d1, d2"),
        (fadd(0, 0, 1, 2), "fadd s0, s1, s2"),
        (fsub(1, 3, 4, 5), "fsub d3, d4, d5"),
        (fsub(0, 3, 4, 5), "fsub s3, s4, s5"),
        (fmul(1, 0, 1, 2), "fmul d0, d1, d2"),
        (fmul(0, 0, 1, 2), "fmul s0, s1, s2"),
        (fdiv(1, 0, 1, 2), "fdiv d0, d1, d2"),
        (fdiv(0, 0, 1, 2), "fdiv s0, s1, s2"),
        (fadd(1, 16, 17, 18), "fadd d16, d17, d18"),
        (fmul(0, 20, 21, 22), "fmul s20, s21, s22"),
        // data-processing (1 source)
        (fneg(1, 0, 1), "fneg d0, d1"),
        (fneg(0, 0, 1), "fneg s0, s1"),
        (fmov_reg(1, 0, 1), "fmov d0, d1"),
        (fmov_reg(0, 5, 6), "fmov s5, s6"),
        (fcvt(0, 1, 0, 0), "fcvt d0, s0"),
        (fcvt(1, 0, 0, 0), "fcvt s0, d0"),
        // compare
        (fcmp(1, 0, 1), "fcmp d0, d1"),
        (fcmp(0, 3, 4), "fcmp s3, s4"),
        // float↔int conversions
        (fcvtzs(0, 1, 0, 0), "fcvtzs w0, d0"),
        (fcvtzs(1, 1, 0, 0), "fcvtzs x0, d0"),
        (fcvtzs(0, 0, 0, 0), "fcvtzs w0, s0"),
        (fcvtzu(0, 1, 0, 0), "fcvtzu w0, d0"),
        (fcvtzu(1, 1, 0, 0), "fcvtzu x0, d0"),
        (scvtf(1, 1, 0, 0), "scvtf d0, x0"),
        (scvtf(0, 1, 0, 0), "scvtf d0, w0"),
        (scvtf(0, 0, 0, 0), "scvtf s0, w0"),
        (ucvtf(1, 1, 0, 0), "ucvtf d0, x0"),
        (ucvtf(0, 1, 0, 0), "ucvtf d0, w0"),
        (fmov_from_gpr(1, 1, 0, 1), "fmov d0, x1"),
        (fmov_from_gpr(0, 0, 0, 1), "fmov s0, w1"),
        // FP loads / stores (unsigned offset)
        (fp_ldst_uimm(true, 3, 0, 1, 0), "ldr d0, [x1]"),
        (fp_ldst_uimm(false, 3, 0, 1, 0), "str d0, [x1]"),
        (fp_ldst_uimm(true, 2, 0, 1, 0), "ldr s0, [x1]"),
        (fp_ldst_uimm(false, 2, 0, 1, 0), "str s0, [x1]"),
        (fp_ldst_uimm(true, 3, 0, 1, 2), "ldr d0, [x1, #16]"),
    ];

    if llvm_mc("ret").is_none() {
        eprintln!("skipping differential_fp_encoding_matches_llvm_mc: no llvm-mc");
        return;
    }
    let mut checked = 0usize;
    for (word, asm) in &corpus {
        let expected = llvm_mc(asm).unwrap_or_else(|| panic!("llvm-mc failed on `{asm}`"));
        assert_eq!(
            word.to_le_bytes().to_vec(),
            expected,
            "FP encoding mismatch for `{asm}`: ours={:02x?} llvm={:02x?}",
            word.to_le_bytes(),
            expected
        );
        checked += 1;
    }
    assert_eq!(checked, corpus.len());
    assert!(checked >= 30, "the FP corpus stays broad ({checked} instructions)");
    eprintln!("differential FP encoder gate: {checked} instructions matched llvm-mc");
}

// --- FP IR fixtures --------------------------------------------------------

/// `double lffd(double a, double b) = a*b + a/b` — F64 arithmetic.
fn build_fdouble() -> (Module, FuncId) {
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
        let r = b.bin(BinOp::FAdd, mul, div, Flags::NONE);
        b.ret(Some(r));
    }
    (m, f)
}

/// `float lffs(float a, float b) = a*b + a/b` — F32 arithmetic.
fn build_ffloat() -> (Module, FuncId) {
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
        let r = b.bin(BinOp::FAdd, mul, div, Flags::NONE);
        b.ret(Some(r));
    }
    (m, f)
}

/// `int lff2i(double x) = (int)(x*x)` — `fptosi` after an `fmul` (`fcvtzs`).
fn build_fptosi() -> (Module, FuncId) {
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
    (m, f)
}

/// `double lfi2f(int n) = (double)n / 2.0` — `sitofp` (`scvtf`) then `fdiv` by a
/// materialized float constant.
fn build_sitofp() -> (Module, FuncId) {
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
    (m, f)
}

/// `double lffmax(double a, double b) = a > b ? a : b` — an `fcmp ogt` driving a
/// branch, the winner passed as an F64 block argument.
fn build_fmax() -> (Module, FuncId) {
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
    (m, f)
}

/// `double lfmix(int a, double b, int c, double d) = (double)(a + c) + (b - d)` —
/// a mixed integer/float parameter list exercising the split ABI (`a`→x0, `b`→v0,
/// `c`→x1, `d`→v1).
fn build_fmixed() -> (Module, FuncId) {
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
        let bb = b.param(entry, 1);
        let c = b.param(entry, 2);
        let d = b.param(entry, 3);
        let sum = b.add(a, c, Flags::NONE);
        let sumf = b.cast(CastOp::SiToFp, sum, f64t);
        let diff = b.bin(BinOp::FSub, bb, d, Flags::NONE);
        let r = b.bin(BinOp::FAdd, sumf, diff, Flags::NONE);
        b.ret(Some(r));
    }
    (m, f)
}

// --- FP interpretation helpers + tests -------------------------------------

fn fd(x: f64) -> Int {
    Int::from_u64(x.to_bits())
}
fn ff(x: f32) -> Int {
    Int::from_u64(u64::from(x.to_bits()))
}
fn as_f64(v: &Int) -> f64 {
    f64::from_bits(v.to_u64().expect("f64 result fits u64"))
}
fn as_f32(v: &Int) -> f32 {
    f32::from_bits(v.to_u64().expect("f32 result fits u64") as u32)
}

fn run_fp(m: &Module, f: FuncId, args: &[Int]) -> Int {
    let (target, funcs) = lower_all(m);
    interp::run(&target, &funcs, f.index(), args)
        .expect("interpretation succeeds")
        .expect("function returns a value")
}

#[test]
fn interp_fdouble() {
    let (m, f) = build_fdouble();
    for (a, b) in [(3.0, 4.0), (1.5, -2.25), (100.0, 7.0), (-8.0, 0.5)] {
        let want = a * b + a / b;
        assert_eq!(as_f64(&run_fp(&m, f, &[fd(a), fd(b)])), want, "lffd({a},{b})");
    }
}

#[test]
fn interp_ffloat() {
    let (m, f) = build_ffloat();
    for (a, b) in [(3.0f32, 4.0f32), (1.5, -2.25), (100.0, 7.0), (-8.0, 0.5)] {
        let want = a * b + a / b;
        assert_eq!(as_f32(&run_fp(&m, f, &[ff(a), ff(b)])), want, "lffs({a},{b})");
    }
}

#[test]
fn interp_fptosi() {
    let (m, f) = build_fptosi();
    for x in [0.0f64, 2.0, 3.5, 7.9, -4.2] {
        let want = Int::from_i64((x * x) as i64).mod_2k(32);
        assert_eq!(run_fp(&m, f, &[fd(x)]), want, "lff2i({x})");
    }
}

#[test]
fn interp_sitofp() {
    let (m, f) = build_sitofp();
    for n in [0i64, 1, 5, 42, -7, 100] {
        let want = n as f64 / 2.0;
        assert_eq!(as_f64(&run_fp(&m, f, &[i(n)])), want, "lfi2f({n})");
    }
}

#[test]
fn interp_fmax() {
    let (m, f) = build_fmax();
    for (a, b) in [(3.0, 4.0), (9.0, 2.0), (-1.0, -5.0), (7.0, 7.0)] {
        let want = if a > b { a } else { b };
        assert_eq!(as_f64(&run_fp(&m, f, &[fd(a), fd(b)])), want, "lffmax({a},{b})");
    }
}

#[test]
fn interp_fmixed() {
    let (m, f) = build_fmixed();
    let cases = [(3i64, 1.5f64, 4i64, 0.25f64), (-2, 10.0, 5, -3.5), (0, 0.0, 0, 0.0)];
    for (a, b, c, d) in cases {
        let want = (a + c) as f64 + (b - d);
        let got = run_fp(&m, f, &[i(a), fd(b), i(c), fd(d)]);
        assert_eq!(as_f64(&got), want, "lfmix({a},{b},{c},{d})");
    }
}

/// Every `fcmp` predicate lowers to a condition plan that matches `ir::semantics`
/// across ordered, unordered (NaN), and equal operands.
#[test]
fn interp_fcmp_all_predicates() {
    use crate::ir::semantics::{self, SemValue};
    let preds = [
        FloatPred::Oeq, FloatPred::Ogt, FloatPred::Oge, FloatPred::Olt, FloatPred::Ole,
        FloatPred::One, FloatPred::Ord, FloatPred::Ueq, FloatPred::Ugt, FloatPred::Uge,
        FloatPred::Ult, FloatPred::Ule, FloatPred::Une, FloatPred::Uno,
    ];
    let nan = f64::NAN;
    let operands = [(1.0, 2.0), (2.0, 1.0), (3.0, 3.0), (nan, 1.0), (1.0, nan), (nan, nan)];
    for pred in preds {
        // Build `int cmp(double a, double b) = (a <pred> b) ? 1 : 0`.
        let mut syms = StrInterner::new();
        let mut m = Module::new("t");
        let f64t = m.types_mut().float(FloatKind::F64);
        let i1t = m.types_mut().int(1);
        let sig = m.types_mut().func(vec![f64t, f64t], i1t, false);
        let f = m.declare_function(syms.intern("cmp"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let a = b.param(entry, 0);
            let bb = b.param(entry, 1);
            let r = b.fcmp(pred, a, bb, Flags::NONE);
            b.ret(Some(r));
        }
        for (x, y) in operands {
            let got = run_fp(&m, f, &[fd(x), fd(y)]);
            let want = match semantics::eval(
                m.types(),
                i1t,
                &crate::ir::inst::InstKind::FCmp(pred),
                &Flags::NONE,
                &[SemValue::Float(FloatBits::F64(x.to_bits())), SemValue::Float(FloatBits::F64(y.to_bits()))],
            ) {
                semantics::EvalOutcome::Value(SemValue::Int { bits, .. }) => bits.to_u64().unwrap(),
                other => panic!("unexpected fcmp semantics outcome: {other:?}"),
            };
            assert_eq!(
                got.to_u64().unwrap(),
                want,
                "fcmp {pred:?} ({x}, {y}): interp={got:?} semantics={want}"
            );
        }
    }
}

#[test]
fn fp_encoding_is_deterministic() {
    let (m, f) = build_fdouble();
    let syms = {
        let mut s = StrInterner::new();
        s.intern("lffd");
        s
    };
    let a = compile_function(&m, f, &syms);
    let b = compile_function(&m, f, &syms);
    assert_eq!(a, b, "identical FP input must yield identical bytes");
}

/// Full FP pipeline (isel → regalloc → frame → encode) round-trips through
/// `llvm-mc` disassembly to the expected A64 FP idioms.
#[test]
fn fp_pipeline_disassembles() {
    let (m, f) = build_fdouble();
    let mut syms = StrInterner::new();
    syms.intern("lffd");
    let emitted = compile_function(&m, f, &syms);
    let Some(text) = llvm_disasm(&emitted.bytes) else {
        eprintln!("skipping fp_pipeline_disassembles: no llvm-mc");
        return;
    };
    for needle in ["fmul", "fdiv", "fadd", "ret"] {
        assert!(text.contains(needle), "expected `{needle}` in disassembly:\n{text}");
    }
}
