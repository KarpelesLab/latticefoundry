//! Tests for the RISC-V RV64IM backend.
//!
//! Because this host cannot *execute* RISC-V code, correctness rests on three
//! tiers:
//!
//! - **Golden byte encodings** — instructions hand-verified from the RISC-V ISA
//!   manual, asserted exactly. These need no toolchain.
//! - **Differential encoding vs `llvm-mc`** — the primary encoder gate: for a
//!   broad corpus, assemble the equivalent RV64 asm with
//!   `llvm-mc --triple=riscv64 -mattr=+m --show-encoding` and assert our bytes
//!   match. Skipped if `llvm-mc` is absent.
//! - **RV64-MIR interpretation** — lower real IR functions and run them on the
//!   [`super::interp`] MIR interpreter, asserting the computed values. This
//!   proves instruction *selection* is semantically right even though the host
//!   cannot run RISC-V.

use super::encode::*;
use super::interp;
use super::isel::RiscvTarget;
use crate::ir::inst::{BinOp, Flags, IntPred};
use crate::ir::{FuncId, Module};
use crate::support::StrInterner;

use puremp::Int;

// ===========================================================================
// Golden byte encodings (verified from the RISC-V ISA manual / llvm-mc)
// ===========================================================================

#[test]
fn golden_r_type() {
    // add a0,a1,a2 ; sub a0,a1,a2 ; mul a0,a1,a2 ; xor a0,a1,a2 ; slt a0,a1,a2
    assert_eq!(add(10, 11, 12).to_le_bytes(), [0x33, 0x85, 0xc5, 0x00]);
    assert_eq!(sub(10, 11, 12).to_le_bytes(), [0x33, 0x85, 0xc5, 0x40]);
    assert_eq!(mul(10, 11, 12).to_le_bytes(), [0x33, 0x85, 0xc5, 0x02]);
    assert_eq!(xor(10, 11, 12).to_le_bytes(), [0x33, 0xc5, 0xc5, 0x00]);
    assert_eq!(slt(10, 11, 12).to_le_bytes(), [0x33, 0xa5, 0xc5, 0x00]);
    // High-register form: add t3,t4,t5 (x28,x29,x30) and sub a6,a7,s11 (x16+).
    assert_eq!(add(28, 29, 30).to_le_bytes(), [0x33, 0x8e, 0xee, 0x01]);
    assert_eq!(sub(16, 17, 27).to_le_bytes(), [0x33, 0x88, 0xb8, 0x41]);
}

#[test]
fn golden_i_type_and_shift() {
    // addi a0,a0,5 ; slli a0,a1,3 ; mv a0,a1 (addi a0,a1,0)
    assert_eq!(addi(10, 10, 5).to_le_bytes(), [0x13, 0x05, 0x55, 0x00]);
    assert_eq!(slli(10, 11, 3).to_le_bytes(), [0x13, 0x95, 0x35, 0x00]);
    assert_eq!(srai(10, 11, 3).to_le_bytes(), [0x13, 0xd5, 0x35, 0x40]);
    assert_eq!(mv(10, 11).to_le_bytes(), [0x13, 0x85, 0x05, 0x00]);
    assert_eq!(sltiu(10, 11, 1).to_le_bytes(), [0x13, 0xb5, 0x15, 0x00]); // seqz
}

#[test]
fn golden_mem_branch_ret() {
    // ld a0,0(sp) ; sd a0,0(sp)
    assert_eq!(load(8, 10, 2, 0).to_le_bytes(), [0x03, 0x35, 0x01, 0x00]);
    assert_eq!(store(8, 10, 2, 0).to_le_bytes(), [0x23, 0x30, 0xa1, 0x00]);
    // beq a0,a1,0 ; jal ra,0 ; ret (jalr x0,ra,0)
    assert_eq!(beq(10, 11, 0).to_le_bytes(), [0x63, 0x00, 0xb5, 0x00]);
    assert_eq!(jal(1, 0).to_le_bytes(), [0xef, 0x00, 0x00, 0x00]);
    assert_eq!(ret().to_le_bytes(), [0x67, 0x80, 0x00, 0x00]);
    // lui a0,1 ; auipc a0,0
    assert_eq!(lui(10, 1).to_le_bytes(), [0x37, 0x15, 0x00, 0x00]);
    assert_eq!(auipc(10, 0).to_le_bytes(), [0x17, 0x05, 0x00, 0x00]);
}

// ===========================================================================
// Differential encoding vs llvm-mc (the primary encoder gate)
// ===========================================================================

/// Assemble one RV64 instruction with `llvm-mc --show-encoding`, returning its
/// bytes. `None` when `llvm-mc` is unavailable. The `+m` feature enables the
/// multiply/divide (M) extension mnemonics.
fn llvm_mc(asm: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let mut child = std::process::Command::new("llvm-mc")
        .arg("--triple=riscv64")
        .arg("-mattr=+m")
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
    // Collect every `encoding: [..]` group and concatenate, so a multi-instruction
    // pseudo-op (`li`, `mv`, ...) yields all its instruction bytes in order.
    let mut bytes = Vec::new();
    let mut rest = &text[..];
    let mut any = false;
    while let Some(pos) = rest.find("encoding: [") {
        any = true;
        let start = pos + "encoding: [".len();
        let end = rest[start..].find(']')? + start;
        for tok in rest[start..end].split(',') {
            let tok = tok.trim().trim_start_matches("0x");
            bytes.push(u8::from_str_radix(tok, 16).ok()?);
        }
        rest = &rest[end..];
    }
    if !any {
        return None;
    }
    Some(bytes)
}

#[test]
fn differential_encoding_matches_llvm_mc() {
    // A corpus covering every RV64IM instruction form the isel/encoder emits.
    let corpus: Vec<(u32, &str)> = vec![
        // R-type integer
        (add(10, 11, 12), "add a0, a1, a2"),
        (sub(10, 11, 12), "sub a0, a1, a2"),
        (and(7, 8, 9), "and t2, s0, s1"),
        (or(10, 11, 12), "or a0, a1, a2"),
        (xor(10, 11, 12), "xor a0, a1, a2"),
        (sll(10, 11, 12), "sll a0, a1, a2"),
        (srl(10, 11, 12), "srl a0, a1, a2"),
        (sra(10, 11, 12), "sra a0, a1, a2"),
        (slt(10, 11, 12), "slt a0, a1, a2"),
        (sltu(10, 11, 12), "sltu a0, a1, a2"),
        (add(28, 29, 30), "add t3, t4, t5"),
        // M-extension
        (mul(10, 11, 12), "mul a0, a1, a2"),
        (mulh(10, 11, 12), "mulh a0, a1, a2"),
        (mulhu(10, 11, 12), "mulhu a0, a1, a2"),
        (div(10, 11, 12), "div a0, a1, a2"),
        (divu(10, 11, 12), "divu a0, a1, a2"),
        (rem(10, 11, 12), "rem a0, a1, a2"),
        (remu(10, 11, 12), "remu a0, a1, a2"),
        // I-type
        (addi(10, 10, 5), "addi a0, a0, 5"),
        (addi(10, 11, -5), "addi a0, a1, -5"),
        (addiw(10, 10, 5), "addiw a0, a0, 5"),
        (addiw(10, 11, 0), "sext.w a0, a1"), // RvOp::SextW
        (andi(10, 11, 255), "andi a0, a1, 255"),
        (slli(10, 11, 56), "slli a0, a1, 56"),
        (srai(10, 10, 56), "srai a0, a0, 56"),
        (srli(10, 10, 40), "srli a0, a0, 40"),
        (andi(10, 11, 15), "andi a0, a1, 15"),
        (ori(10, 11, 15), "ori a0, a1, 15"),
        (xori(10, 11, -1), "not a0, a1"),
        (sltiu(10, 11, 1), "seqz a0, a1"),
        (slli(10, 11, 3), "slli a0, a1, 3"),
        (srli(10, 11, 3), "srli a0, a1, 3"),
        (srai(10, 11, 3), "srai a0, a1, 3"),
        (mv(10, 11), "mv a0, a1"),
        // loads (unsigned sub-word) / stores
        (load(8, 10, 2, 0), "ld a0, 0(sp)"),
        (load(4, 10, 2, 0), "lwu a0, 0(sp)"),
        (load(2, 10, 11, 4), "lhu a0, 4(a1)"),
        (load(1, 10, 11, 0), "lbu a0, 0(a1)"),
        (store(8, 10, 2, 0), "sd a0, 0(sp)"),
        (store(4, 10, 2, 8), "sw a0, 8(sp)"),
        (store(2, 10, 11, 2), "sh a0, 2(a1)"),
        (store(1, 10, 11, 1), "sb a0, 1(a1)"),
        // U-type
        (lui(10, 1), "lui a0, 1"),
        (auipc(10, 0), "auipc a0, 0"),
        // branches / jumps / calls (zero displacement)
        (beq(10, 11, 0), "beq a0, a1, 0"),
        (bne(10, 11, 0), "bne a0, a1, 0"),
        (jal(1, 0), "jal ra, 0"),
        (jal(0, 0), "jal zero, 0"),
        (jalr(1, 1, 0), "jalr ra"),
        (jalr(0, 1, 0), "ret"),
        (ret(), "ret"),
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
    assert!(checked >= 40, "the corpus stays broad ({checked} instructions)");
    eprintln!("differential encoder gate: {checked} instructions matched llvm-mc");
}

#[test]
fn differential_li_materialization() {
    // The `li` materialization (12-bit `addi`, 32-bit `lui`+`addi`/`addiw`) must match the
    // assembler's `li` expansion byte-for-byte.
    if llvm_mc("ret").is_none() {
        eprintln!("skipping differential_li_materialization: no llvm-mc");
        return;
    }
    for val in [0i64, 5, -5, 2047, -2048, 0x12345, -0x12345, 0x7FFF_FFFF, -0x8000_0000, 0x7FFF_F800] {
        let ours = emit_li_bytes(10, val);
        let asm = format!("li a0, {val}");
        let expected = llvm_mc(&asm).unwrap_or_else(|| panic!("llvm-mc failed on `{asm}`"));
        assert_eq!(ours, expected, "li a0, {val}: ours={ours:02x?} llvm={expected:02x?}");
    }
}

/// Wide `li` materializations evaluate to their value, including near
/// `i64::MAX`, where the high part's computation must wrap (it used to
/// overflow). Evaluated by a tiny model of the five instruction forms `li`
/// emits (`lui`, `addi` from `x0` or `rd`, `addiw`, `slli`).
#[test]
fn wide_li_sequences_evaluate_to_their_value() {
    fn eval(bytes: &[u8]) -> i64 {
        let mut r: i64 = 0;
        for w in bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())) {
            let imm12 = i64::from((w as i32) >> 20);
            let rs1_is_zero = (w >> 15) & 0x1F == 0;
            r = match (w & 0x7F, (w >> 12) & 7) {
                (0x37, _) => i64::from((w & 0xFFFF_F000) as i32),
                (0x13, 0) => if rs1_is_zero { imm12 } else { r.wrapping_add(imm12) },
                (0x13, 1) => r << ((w >> 20) & 0x3F),
                (0x1B, 0) => i64::from(r.wrapping_add(imm12) as i32),
                other => panic!("unexpected li word {w:#010x} {other:?}"),
            };
        }
        r
    }
    for val in [i64::MAX, i64::MAX - 1, i64::MIN, i64::MIN + 1, 0x7FFF_FFFF_FFFF_F800, 0x1234_5678_9ABC_DEF0, -2, 0x8000_0000] {
        assert_eq!(eval(&emit_li_bytes(10, val)), val, "li {val:#x}");
    }
}

// ===========================================================================
// IR fixtures + RV64-MIR interpretation (isel correctness without execution)
// ===========================================================================

/// Lower every function of `m` to MIR (indexed by `FuncId`), for the interpreter.
fn lower_all(m: &Module) -> (RiscvTarget, Vec<crate::codegen::mir::MachineFunction>) {
    let target = RiscvTarget::new();
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

/// `lfdivmod(a, b) = (a / b) + (a % b)` — exercises `div` and `rem` (signed).
fn build_divmod() -> (Module, FuncId) {
    build_divmod_ops("lfdivmod", BinOp::SDiv, BinOp::SRem)
}

/// `lfudivmod(a, b) = (a /u b) + (a %u b)` — exercises `divu` and `remu`.
fn build_udivmod() -> (Module, FuncId) {
    build_divmod_ops("lfudivmod", BinOp::UDiv, BinOp::URem)
}

fn build_divmod_ops(name: &str, dop: BinOp, rop: BinOp) -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t, i64t], i64t, false);
    let f = m.declare_function(syms.intern(name), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let q = b.bin(dop, a, bb, Flags::NONE);
        let r = b.bin(rop, a, bb, Flags::NONE);
        let s = b.add(q, r, Flags::NONE);
        b.ret(Some(s));
    }
    (m, f)
}

/// `lfbits(x) = ((x << 4) ^ (x >> 1)) & 0xff` — shifts + bitwise (imm folded).
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
    // Registers hold 64-bit patterns: -1 comes back as all-ones.
    assert_eq!(eval2(&m, f, -1, -5), i(-1).mod_2k(64));
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
fn interp_divmod_signed() {
    let m64 = |v: i64| i(v).mod_2k(64);
    let (m, f) = build_divmod();
    assert_eq!(eval2(&m, f, 17, 5), m64(3 + 2)); // 17/5=3, 17%5=2
    assert_eq!(eval2(&m, f, 100, 9), m64(11 + 1)); // 100/9=11, 100%9=1
    assert_eq!(eval2(&m, f, -17, 5), m64(-3 + -2)); // trunc toward zero
}

#[test]
fn interp_divmod_unsigned() {
    let m64 = |v: i64| i(v).mod_2k(64);
    let (m, f) = build_udivmod();
    assert_eq!(eval2(&m, f, 17, 5), m64(3 + 2));
    assert_eq!(eval2(&m, f, 100, 9), m64(11 + 1));
    assert_eq!(eval2(&m, f, 255, 16), m64(15 + 15)); // 255/16=15, 255%16=15
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
// llvm-objdump round-trip + determinism
// ===========================================================================

/// Disassemble a `.text` byte blob with `llvm-objdump`.
fn llvm_objdump(bytes: &[u8]) -> Option<String> {
    use std::io::Write;
    // Write the bytes as a raw binary and disassemble as RV64.
    let dir = std::env::temp_dir();
    let path = dir.join(format!("lf_rv_{}.bin", std::process::id()));
    std::fs::File::create(&path).ok()?.write_all(bytes).ok()?;
    let out = std::process::Command::new("llvm-objdump")
        .arg("-D")
        .arg("--triple=riscv64")
        .arg("-b")
        .arg("binary")
        .arg("-m")
        .arg("riscv")
        .arg(&path)
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&path);
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn disasm_round_trip_lfadd() {
    let (m, f) = build_add();
    let emitted = compile_function(&m, f);
    let Some(text) = llvm_objdump(&emitted.bytes) else {
        eprintln!("skipping disasm_round_trip_lfadd: no llvm-objdump");
        return;
    };
    // The body must decode to the expected RV64 idioms.
    for needle in ["add", "ret"] {
        assert!(text.contains(needle), "expected `{needle}` in disassembly:\n{text}");
    }
}

#[test]
fn disasm_round_trip_lfmax_branches() {
    let (m, f) = build_max();
    let emitted = compile_function(&m, f);
    let Some(text) = llvm_objdump(&emitted.bytes) else {
        eprintln!("skipping disasm_round_trip_lfmax_branches: no llvm-objdump");
        return;
    };
    // The comparison + branch idiom must round-trip (a conditional branch and a
    // jump appear, and the compare lowers to `slt`).
    for needle in ["slt", "bne", "jal", "ret"] {
        assert!(text.contains(needle), "expected `{needle}` in disassembly:\n{text}");
    }
}

#[test]
fn compile_module_has_text() {
    let (m, _) = build_call();
    let mut syms = StrInterner::new();
    syms.intern("lfcallee");
    syms.intern("lfcaller");
    let obj = compile_module(&m, &syms);
    let text = obj.section(crate::mc::object::SectionId::from_index(0));
    assert!(!text.bytes.is_empty(), "text section is non-empty");
    assert!(text.bytes.len().is_multiple_of(4), "RV code is word-sized");
}

#[test]
fn encoding_is_deterministic() {
    let (m, f) = build_loop_sum();
    let a = compile_function(&m, f);
    let b = compile_function(&m, f);
    assert_eq!(a, b, "identical input must yield identical bytes");
}

#[test]
fn frame_and_spill_round_trip() {
    // A caller/callee pair forces a call frame (ra save) and callee-saved usage,
    // exercising the prologue/epilogue + frame layout end to end.
    let (m, f) = build_call();
    let emitted = compile_function(&m, f);
    assert!(!emitted.bytes.is_empty());
    assert!(emitted.bytes.len().is_multiple_of(4));
    // The caller saves ra (a store) and restores it (a load) around the frame.
    let Some(text) = llvm_objdump(&emitted.bytes) else {
        eprintln!("skipping frame_and_spill_round_trip disasm checks: no llvm-objdump");
        return;
    };
    for needle in ["addi", "sd", "ld", "ret"] {
        assert!(text.contains(needle), "expected `{needle}` in caller disassembly:\n{text}");
    }
}

// ===========================================================================
// `syscall` — Linux RISC-V ABI (number a7, args a0..a5, `ecall`, result a0)
// ===========================================================================

/// `lfsys(x) = syscall(64, x, x+1, .., x+5) + 1` — a 6-argument syscall whose
/// arguments are computed values.
fn build_syscall6() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("lfsys"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let mut args = vec![x];
        for k in 1..6 {
            let c = b.const_i64(i64t, k);
            args.push(b.add(x, c, Flags::NONE));
        }
        let nr = b.const_i64(i64t, 64);
        let r = b.syscall(nr, &args);
        let one = b.const_i64(i64t, 1);
        let s = b.add(r, one, Flags::NONE);
        b.ret(Some(s));
    }
    (m, f)
}

#[test]
fn syscall_lowers_to_the_linux_register_convention() {
    use super::isel::RvOp;
    use super::regs::gpr;
    use crate::codegen::mir::{MachineOperand, Reg};
    let (m, _) = build_syscall6();
    let (_, funcs) = lower_all(&m);
    let insts: Vec<_> = funcs[0].block_ids().flat_map(|b| funcs[0].block(b).insts.clone()).collect();
    let at = insts
        .iter()
        .position(|i| RvOp::decode(i.opcode) == RvOp::Ecall)
        .expect("an Ecall is emitted");
    let ecall_inst = &insts[at];
    let uses: Vec<Reg> = ecall_inst.uses().collect();
    let expect: Vec<Reg> = [17u16, 10, 11, 12, 13, 14, 15].iter().map(|&n| Reg::Physical(gpr(n))).collect();
    assert_eq!(uses, expect, "number in a7, arguments in a0..a5");
    assert_eq!(ecall_inst.defs().collect::<Vec<_>>(), vec![Reg::Physical(gpr(10))], "result in a0 only");
    // The seven fixed-register moves form one consecutive run right before it.
    for (k, want) in expect.iter().enumerate() {
        let mv = &insts[at - 7 + k];
        assert_eq!(RvOp::decode(mv.opcode), RvOp::Mv);
        assert_eq!(mv.operands[0], MachineOperand::Def(*want));
    }
    // And the result is read back out of a0 immediately after.
    assert_eq!(insts[at + 1].uses().collect::<Vec<_>>(), vec![Reg::Physical(gpr(10))]);
}

#[test]
fn syscall_interpreter_hook_and_clean_error() {
    let (m, _) = build_syscall6();
    let (target, funcs) = lower_all(&m);
    // No kernel: a clean "unsupported side effect" error, not an invented value.
    let err = interp::run(&target, &funcs, 0, &[i(10)]).unwrap_err();
    assert!(err.contains("unsupported side effect: syscall"), "{err}");

    // With a hook: it sees exactly (a7, a0..a5) and its raw -errno comes back.
    let mut seen: Vec<(Int, Vec<Int>)> = Vec::new();
    let mut hook = |nr: &Int, args: &[Int]| -> Result<Int, String> {
        seen.push((nr.clone(), args.to_vec()));
        Ok(i(-9))
    };
    let got = interp::run_with_syscalls(&target, &funcs, 0, &[i(10)], Some(&mut hook)).unwrap();
    // -9 + 1 as a 64-bit pattern.
    assert_eq!(got, Some(i(-8).mod_2k(64)));
    assert_eq!(seen, vec![(i(64), (10..16).map(i).collect())]);
}

#[test]
fn syscall_encodes_ecall() {
    assert_eq!(ecall(), 0x0000_0073);
    if let Some(bytes) = llvm_mc("ecall") {
        assert_eq!(ecall().to_le_bytes().to_vec(), bytes, "ecall matches llvm-mc");
    } else {
        eprintln!("skipping llvm-mc cross-check of ecall: no llvm-mc");
    }
    // The compiled function contains the instruction word.
    let (m, f) = build_syscall6();
    let code = compile_function(&m, f);
    assert!(
        code.bytes.chunks(4).any(|w| w == 0x0000_0073u32.to_le_bytes()),
        "the encoded body contains `ecall`"
    );
}


// ===========================================================================
// Narrow values with dirty upper register bits
// ===========================================================================
//
// i1/i8/i16 and odd-width (`_BitInt`) values live in wider registers whose bits
// above the value's width are not kept clean: an `i8` add of 200 + 100 leaves
// 300 in the register, a `trunc` to `i1` keeps the source's other bits, and a
// negative constant is materialized sign-extended. Every op whose result
// depends on those bits must extend first. These are the x86-64 execution
// probes (`src/link/mod.rs`) run on the MIR interpreter, which models the
// register width faithfully. Each checking function returns 0 when correct;
// `main` ORs a distinct bit per failing check.

/// The `sitofp`/`uitofp` checks' bits in the two probes. RV64IM has no
/// floating-point lowering here (the F/D extensions are deferred), so those
/// checks exercise nothing and are masked out.
const RV_NO_FP_OPS: u64 = 0b11 << 8;
const RV_NO_FP_DIRTY: u64 = 1 << 2;

/// Parse `src`, lower every function, and run its `main` on the interpreter.
/// Every function also goes through the whole encoding pipeline (register
/// allocation, frame layout, encoding) to keep the new extension ops covered,
/// except the int→float probes: there is no FP register file to allocate yet.
fn run_lf_main(src: &str) -> u64 {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, crate::support::diagnostics::FileId::new(0), &mut syms)
        .expect("parse .lf");
    for (k, f) in m.functions().enumerate() {
        if !syms.resolve(f.name).ends_with("tofp") {
            assert!(!compile_function(&m, FuncId::from_index(k)).bytes.is_empty());
        }
    }
    let main = m
        .functions()
        .position(|f| syms.resolve(f.name) == "main")
        .expect("a `main` function");
    let (target, funcs) = lower_all(&m);
    let v = interp::run(&target, &funcs, main, &[])
        .expect("interpretation succeeds")
        .expect("main returns a value");
    v.to_u64().expect("a 64-bit pattern")
}

/// The names of the checks whose bit is set in `code`.
fn failing_checks<'a>(code: u64, names: &[&'a str]) -> Vec<&'a str> {
    names.iter().enumerate().filter(|(i, _)| code & (1 << i) != 0).map(|(_, n)| *n).collect()
}

#[test]
fn narrow_icmp_ignores_upper_register_bits() {
    // Each function returns 1 when the comparison sees the wrapped value.
    let src = "\
module \"k\"
func @ult8(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %c = icmp ult %s, %a : i1
  %r = zext %c : i64
  ret %r
}
func @slt8(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %c = icmp slt %s, i8 0 : i1
  %r = zext %c : i64
  ret %r
}
func @eq16(i16, i16) -> i64 {
entry ^0(%a: i16, %b: i16):
  %s = add %a, %b : i16
  %c = icmp eq %s, i16 4 : i1
  %r = zext %c : i64
  ret %r
}
func @main() -> i64 {
entry ^0:
  %x = call @ult8(i8 -56, i8 100) : i64
  %y = call @slt8(i8 100, i8 100) : i64
  %z = call @eq16(i16 -2, i16 6) : i64
  %xy = shl %y, i64 1 : i64
  %xz = shl %z, i64 2 : i64
  %t = or %x, %xy : i64
  %u = or %t, %xz : i64
  ret %u
}
";
    let code = run_lf_main(src);
    let failing = failing_checks(!code & 0b111, &["ult i8", "slt i8", "eq i16"]);
    assert!(failing.is_empty(), "narrow icmp saw dirty upper bits: {failing:?} (got {code:#b})");
}

#[test]
fn narrow_ops_ignore_upper_register_bits() {
    // 200 + 100 is 44 as an i8 (300 in the register), and 100 + 100 is -56 as
    // an i8 (200 in the register).
    let src = "\
module \"k\"
func @lshr(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = lshr %s, i8 1 : i8
  %c = icmp ne %r, i8 22 : i1
  %z = zext %c : i64
  ret %z
}
func @ashr(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = ashr %s, i8 1 : i8
  %c = icmp ne %r, i8 -28 : i1
  %z = zext %c : i64
  ret %z
}
func @udiv(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = udiv %s, i8 2 : i8
  %c = icmp ne %r, i8 22 : i1
  %z = zext %c : i64
  ret %z
}
func @urem(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = urem %s, i8 3 : i8
  %c = icmp ne %r, i8 2 : i1
  %z = zext %c : i64
  ret %z
}
func @sdiv(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = sdiv %s, i8 2 : i8
  %c = icmp ne %r, i8 -28 : i1
  %z = zext %c : i64
  ret %z
}
func @srem(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = srem %s, i8 3 : i8
  %c = icmp ne %r, i8 -2 : i1
  %z = zext %c : i64
  ret %z
}
func @switch(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  switch %s, ^1 [44: ^2]
^1:
  ret i64 1
^2:
  ret i64 0
}
func @condbr(i32) -> i64 {
entry ^0(%a: i32):
  %t = trunc %a : i1
  cond_br %t, ^1, ^2
^1:
  ret i64 1
^2:
  ret i64 0
}
func @sitofp(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %f = sitofp %s : f64
  %i = fptosi %f : i64
  %c = icmp ne %i, i64 -56 : i1
  %z = zext %c : i64
  ret %z
}
func @uitofp(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %f = uitofp %s : f64
  %i = fptosi %f : i64
  %c = icmp ne %i, i64 44 : i1
  %z = zext %c : i64
  ret %z
}
func @main() -> i64 {
entry ^0:
  %v0 = call @lshr(i8 -56, i8 100) : i64
  %v1 = call @ashr(i8 100, i8 100) : i64
  %v2 = call @udiv(i8 -56, i8 100) : i64
  %v3 = call @urem(i8 -56, i8 100) : i64
  %v4 = call @sdiv(i8 100, i8 100) : i64
  %v5 = call @srem(i8 100, i8 100) : i64
  %v6 = call @switch(i8 -56, i8 100) : i64
  %v7 = call @condbr(i32 2) : i64
  %v8 = call @sitofp(i8 100, i8 100) : i64
  %v9 = call @uitofp(i8 -56, i8 100) : i64
  %s1 = shl %v1, i64 1 : i64
  %s2 = shl %v2, i64 2 : i64
  %s3 = shl %v3, i64 3 : i64
  %s4 = shl %v4, i64 4 : i64
  %s5 = shl %v5, i64 5 : i64
  %s6 = shl %v6, i64 6 : i64
  %s7 = shl %v7, i64 7 : i64
  %s8 = shl %v8, i64 8 : i64
  %s9 = shl %v9, i64 9 : i64
  %o1 = or %v0, %s1 : i64
  %o2 = or %o1, %s2 : i64
  %o3 = or %o2, %s3 : i64
  %o4 = or %o3, %s4 : i64
  %o5 = or %o4, %s5 : i64
  %o6 = or %o5, %s6 : i64
  %o7 = or %o6, %s7 : i64
  %o8 = or %o7, %s8 : i64
  %o9 = or %o8, %s9 : i64
  ret %o9
}
";
    let names = ["lshr", "ashr", "udiv", "urem", "sdiv", "srem", "switch", "cond_br", "sitofp", "uitofp"];
    let code = run_lf_main(src) & !RV_NO_FP_OPS;
    let failing = failing_checks(code, &names);
    assert!(failing.is_empty(), "narrow ops saw dirty upper bits: {failing:?} ({code:#b})");
}

#[test]
fn narrow_values_dirty_above_width() {
    // `0 - 1` as an i8 is 255 unsigned but all-ones in the register; a `trunc`
    // to i1 keeps the source's other bits; odd widths (i24) have no compare of
    // their own; a switch compares full registers, and case values may not fit
    // an immediate.
    let src = "\
module \"k\"
func @ushr(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = sub %a, %b : i8
  %r = lshr %s, i8 1 : i8
  %c = icmp ne %r, i8 127 : i1
  %z = zext %c : i64
  ret %z
}
func @udiv(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = sub %a, %b : i8
  %r = udiv %s, i8 2 : i8
  %c = icmp ne %r, i8 127 : i1
  %z = zext %c : i64
  ret %z
}
func @uitofp(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = sub %a, %b : i8
  %f = uitofp %s : f64
  %i = fptosi %f : i64
  %c = icmp ne %i, i64 255 : i1
  %z = zext %c : i64
  ret %z
}
func @zext1(i32) -> i64 {
entry ^0(%a: i32):
  %t = trunc %a : i1
  %z = zext %t : i64
  ret %z
}
func @select1(i32) -> i64 {
entry ^0(%a: i32):
  %t = trunc %a : i1
  %r = select %t, i64 1, i64 0 : i64
  ret %r
}
func @cmp24(i32) -> i64 {
entry ^0(%a: i32):
  %t = trunc %a : i24
  %c = icmp ne %t, i24 5 : i1
  %z = zext %c : i64
  ret %z
}
func @switch32(i32, i32) -> i64 {
entry ^0(%a: i32, %b: i32):
  %s = sub %a, %b : i32
  switch %s, ^1 [-1: ^2]
^1:
  ret i64 1
^2:
  ret i64 0
}
func @switch64(i64) -> i64 {
entry ^0(%a: i64):
  switch %a, ^1 [4294967296: ^2]
^1:
  ret i64 1
^2:
  ret i64 0
}
func @main() -> i64 {
entry ^0:
  %v0 = call @ushr(i8 0, i8 1) : i64
  %v1 = call @udiv(i8 0, i8 1) : i64
  %v2 = call @uitofp(i8 0, i8 1) : i64
  %v3 = call @zext1(i32 2) : i64
  %v4 = call @select1(i32 2) : i64
  %v5 = call @cmp24(i32 16777221) : i64
  %v6 = call @switch32(i32 0, i32 1) : i64
  %v7 = call @switch64(i64 4294967296) : i64
  %s1 = shl %v1, i64 1 : i64
  %s2 = shl %v2, i64 2 : i64
  %s3 = shl %v3, i64 3 : i64
  %s4 = shl %v4, i64 4 : i64
  %s5 = shl %v5, i64 5 : i64
  %s6 = shl %v6, i64 6 : i64
  %s7 = shl %v7, i64 7 : i64
  %o1 = or %v0, %s1 : i64
  %o2 = or %o1, %s2 : i64
  %o3 = or %o2, %s3 : i64
  %o4 = or %o3, %s4 : i64
  %o5 = or %o4, %s5 : i64
  %o6 = or %o5, %s6 : i64
  %o7 = or %o6, %s7 : i64
  ret %o7
}
";
    let names = ["lshr", "udiv", "uitofp", "zext i1", "select i1", "icmp i24", "switch i32", "switch imm64"];
    let code = run_lf_main(src) & !RV_NO_FP_DIRTY;
    let failing = failing_checks(code, &names);
    assert!(failing.is_empty(), "narrow values mishandled: {failing:?} ({code:#b})");
}

#[test]
fn narrow_casts_and_wide_odd_widths() {
    // Integer casts and odd widths above 32 bits: `sext`/`zext` must extend
    // from the source's own width (its register may hold anything above it),
    // an i48 op must run on the full 64-bit register, and a signed compare of
    // i1s sees `true` as -1.
    let src = "\
module \"k\"
func @sext8(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %x = sext %s : i64
  %c = icmp ne %x, i64 -56 : i1
  %z = zext %c : i64
  ret %z
}
func @zext8(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %x = zext %s : i64
  %c = icmp ne %x, i64 44 : i1
  %z = zext %c : i64
  ret %z
}
func @zext32(i64) -> i64 {
entry ^0(%a: i64):
  %t = trunc %a : i32
  %x = zext %t : i64
  %c = icmp ne %x, i64 5 : i1
  %z = zext %c : i64
  ret %z
}
func @sext32(i64) -> i64 {
entry ^0(%a: i64):
  %t = trunc %a : i32
  %x = sext %t : i64
  %c = icmp ne %x, i64 -1 : i1
  %z = zext %c : i64
  ret %z
}
func @sext1(i64) -> i64 {
entry ^0(%a: i64):
  %t = icmp eq %a, %a : i1
  %x = sext %t : i64
  %c = icmp ne %x, i64 -1 : i1
  %z = zext %c : i64
  ret %z
}
func @add48(i64) -> i64 {
entry ^0(%a: i64):
  %t = trunc %a : i48
  %s = add %t, i48 1 : i48
  %x = zext %s : i64
  %c = icmp ne %x, i64 140737488355328 : i1
  %z = zext %c : i64
  ret %z
}
func @ashr48(i64) -> i64 {
entry ^0(%a: i64):
  %t = trunc %a : i48
  %r = ashr %t, i48 4 : i48
  %x = sext %r : i64
  %c = icmp ne %x, i64 -1 : i1
  %z = zext %c : i64
  ret %z
}
func @sge1(i64) -> i64 {
entry ^0(%a: i64):
  %t = icmp eq %a, %a : i1
  %f = icmp ne %a, %a : i1
  %c = icmp sge %t, %f : i1
  %z = zext %c : i64
  ret %z
}
func @main() -> i64 {
entry ^0:
  %v0 = call @sext8(i8 100, i8 100) : i64
  %v1 = call @zext8(i8 -56, i8 100) : i64
  %v2 = call @zext32(i64 4294967301) : i64
  %v3 = call @sext32(i64 4294967295) : i64
  %v4 = call @sext1(i64 7) : i64
  %v5 = call @add48(i64 140737488355327) : i64
  %v6 = call @ashr48(i64 -1) : i64
  %v7 = call @sge1(i64 7) : i64
  %s1 = shl %v1, i64 1 : i64
  %s2 = shl %v2, i64 2 : i64
  %s3 = shl %v3, i64 3 : i64
  %s4 = shl %v4, i64 4 : i64
  %s5 = shl %v5, i64 5 : i64
  %s6 = shl %v6, i64 6 : i64
  %s7 = shl %v7, i64 7 : i64
  %o1 = or %v0, %s1 : i64
  %o2 = or %o1, %s2 : i64
  %o3 = or %o2, %s3 : i64
  %o4 = or %o3, %s4 : i64
  %o5 = or %o4, %s5 : i64
  %o6 = or %o5, %s6 : i64
  %o7 = or %o6, %s7 : i64
  ret %o7
}
";
    let names = ["sext i8", "zext i8", "zext i32", "sext i32", "sext i1", "add i48", "ashr i48", "sge i1"];
    let code = run_lf_main(src);
    let failing = failing_checks(code, &names);
    assert!(failing.is_empty(), "narrow casts / odd widths mishandled: {failing:?} ({code:#b})");
}

#[test]
fn narrow_shift_count_and_ptr_offset() {
    // A variable shift takes its count from the low 5/6 bits of the count
    // register, which an `i4` count doesn't own; and a `ptr_add` offset is a
    // signed value of its own width (an i32 -8 held as 2^32 - 8 is still -8).
    let src = "\
module \"k\"
func @shl4(i8) -> i64 {
entry ^0(%a: i8):
  %c = trunc %a : i4
  %r = shl i4 3, %c : i4
  %z = zext %r : i64
  %k = icmp ne %z, i64 6 : i1
  %o = zext %k : i64
  ret %o
}
func @poff(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %p = alloca [4 x i64] : ptr
  %e = ptr_add %p, i64 8 : ptr
  store i64 7, %e align 8 : i64
  %q = ptr_add %p, i64 16 : ptr
  %ta = trunc %a : i32
  %tb = trunc %b : i32
  %off = sub %ta, %tb : i32
  %r = ptr_add %q, %off : ptr
  %v = load %r align 8 : i64
  %k = icmp ne %v, i64 7 : i1
  %o = zext %k : i64
  ret %o
}
func @main() -> i64 {
entry ^0:
  %v0 = call @shl4(i8 17) : i64
  %v1 = call @poff(i64 4294967296, i64 8) : i64
  %s1 = shl %v1, i64 1 : i64
  %o1 = or %v0, %s1 : i64
  ret %o1
}
";
    let code = run_lf_main(src);
    let failing = failing_checks(code, &["shl i4 count", "ptr_add i32 offset"]);
    assert!(failing.is_empty(), "narrow shift count / offset mishandled: {failing:?} ({code:#b})");
}

#[test]
fn lp64_i32_and_i1_cross_calls_extended() {
    // The LP64 psABI passes and returns an `i32` sign-extended to 64 bits
    // (whatever its C signedness) and a `_Bool` zero-extended, and foreign
    // (e.g. GCC-compiled) code relies on it. A `trunc` leaves the source's upper
    // bits in the register, so both the argument and the return must extend.
    // `@id32`/`@id1` hand their parameter straight back, so `a0` shows exactly
    // what crossed each boundary.
    let src = "\
module \"k\"
func @id32(i32) -> i32 {
entry ^0(%a: i32):
  ret %a
}
func @pass32(i64) -> i32 {
entry ^0(%a: i64):
  %t = trunc %a : i32
  %r = call @id32(%t) : i32
  ret %r
}
func @id1(i1) -> i1 {
entry ^0(%a: i1):
  ret %a
}
func @pass1(i64) -> i1 {
entry ^0(%a: i64):
  %t = trunc %a : i1
  %r = call @id1(%t) : i1
  ret %r
}
";
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, crate::support::diagnostics::FileId::new(0), &mut syms)
        .expect("parse .lf");
    let (target, funcs) = lower_all(&m);
    let run = |f: usize, x: i64| {
        interp::run(&target, &funcs, f, &[i(x)]).expect("runs").expect("returns").to_u64().unwrap()
    };
    // 0x1_FFFF_FFFF truncates to the i32 -1: all-ones once sign-extended.
    assert_eq!(run(1, 0x1_FFFF_FFFF), u64::MAX, "i32 argument/return sign-extended");
    // 0x1_7FFF_FFFF truncates to i32::MAX: upper word cleared.
    assert_eq!(run(1, 0x1_7FFF_FFFF), 0x7FFF_FFFF, "i32 argument/return sign-extended");
    // 6 truncates to the i1 0, 7 to 1.
    assert_eq!(run(3, 6), 0, "i1 argument/return zero-extended");
    assert_eq!(run(3, 7), 1, "i1 argument/return zero-extended");
    // The callee also extends its own return, so the i32 case holds even for a
    // direct entry into `@id32` with a dirty register.
    assert_eq!(run(0, 0x1_FFFF_FFFF), u64::MAX);
}

// ===========================================================================
// Atomics: the A extension (AMOs, LR/SC loops, fences)
// ===========================================================================

/// Assemble RV64IMA code with `llvm-mc` (labels `1:`/`2:` with references
/// `1b`/`2f`, resolved here to numeric branch offsets since `--show-encoding`
/// leaves label fixups unresolved), returning all its bytes; `None` when
/// `llvm-mc` is unavailable or rejects the input.
fn llvm_mc_rva(asm: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let mut labels: Vec<(String, i64)> = Vec::new();
    let mut insns: Vec<&str> = Vec::new();
    for line in asm.lines().map(str::trim).filter(|l| !l.is_empty()) {
        match line.strip_suffix(':') {
            Some(l) => labels.push((l.to_string(), insns.len() as i64)),
            None => insns.push(line),
        }
    }
    let at = |name: &str| labels.iter().find(|(l, _)| l == name).map(|&(_, i)| i).unwrap_or(0);
    let resolved: Vec<String> = insns
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let i = i as i64;
            l.replace("1b", &(4 * (at("1") - i)).to_string()).replace("2f", &(4 * (at("2") - i)).to_string())
        })
        .collect();
    let mut child = std::process::Command::new("llvm-mc")
        .args(["--triple=riscv64", "-mattr=+m,+a", "--show-encoding"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(resolved.join("\n").as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut bytes = Vec::new();
    for line in text.lines() {
        let Some(pos) = line.find("encoding: [") else { continue };
        let start = pos + "encoding: [".len();
        let end = line[start..].find(']')? + start;
        for tok in line[start..end].split(',') {
            bytes.push(u8::from_str_radix(tok.trim().trim_start_matches("0x"), 16).ok()?);
        }
    }
    Some(bytes)
}

#[test]
fn atomics_run_their_sequential_meaning_in_the_interpreter() {
    use crate::target::atomic_fixtures::{CMPXCHG_SLOTS, rmw_cases, rmw_slot_program};
    for bytes in [1, 2, 4, 8] {
        let cases = rmw_cases(bytes);
        let code = run_lf_main(&rmw_slot_program(&cases));
        assert_eq!(code, 0, "i{}: case {:?}", 8 * bytes, cases.get((code as usize).wrapping_sub(1)));
    }
    assert_eq!(run_lf_main(CMPXCHG_SLOTS), 0);
}

#[test]
fn atomics_select_fences_amos_and_loops() {
    use super::isel::RvOp;
    use crate::codegen::mir::MachineOperand;
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(
        crate::target::atomic_fixtures::ALL_FORMS,
        crate::support::diagnostics::FileId::new(0),
        &mut syms,
    )
    .expect("parse");
    crate::verify::verify_module(&m).expect("verify");
    let (_, funcs) = lower_all(&m);
    let insts: Vec<_> = funcs[0].block_ids().flat_map(|b| funcs[0].block(b).insts.clone()).collect();
    let imm = |o: &MachineOperand| match o {
        MachineOperand::Imm(v) => v.to_u64().unwrap(),
        other => panic!("{other:?}"),
    };
    // The memory-ordering skeleton: every fence (fm, pred, succ) and access in
    // order, per the RVWMO mapping.
    let skeleton: Vec<String> = insts
        .iter()
        .filter_map(|i| match RvOp::decode(i.opcode) {
            RvOp::Fence => Some(format!(
                "fence{}.{:04b}.{:04b}",
                if imm(&i.operands[0]) == 8 { ".tso" } else { "" },
                imm(&i.operands[1]),
                imm(&i.operands[2])
            )),
            RvOp::Load => Some(format!("l{}", imm(&i.operands[2]))),
            RvOp::Store => Some(format!("s{}", imm(&i.operands[2]))),
            RvOp::AtomicRmw => Some(format!("rmw{}.{:03b}", imm(&i.operands[3]), imm(&i.operands[5]))),
            RvOp::CmpXchg => Some(format!("cas{}.{:03b}", imm(&i.operands[4]), imm(&i.operands[5]))),
            _ => None,
        })
        .collect();
    let want = [
        "l8",                                          // relaxed load
        "l8", "fence.0010.0011",                       // acquire: l; fence r,rw
        "fence.0011.0011", "l8", "fence.0010.0011",    // seq_cst: fence rw,rw; l; fence r,rw
        "s8",                                          // relaxed store
        "fence.0011.0001", "s8",                       // release: fence rw,w; s
        "fence.0011.0001", "s8",                       // seq_cst: fence rw,w; s
        "fence.0010.0011",                             // fence acquire
        "fence.0011.0001",                             // fence release
        "fence.tso.0011.0011",                         // fence acq_rel
        "fence.0011.0011",                             // fence seq_cst
        "rmw8.000", "rmw8.001", "rmw8.100", "rmw8.101", "rmw8.111", "rmw1.111",
        "cas8.111", "cas2.001",
        "l1", "s1",                                    // the volatile byte access
    ];
    assert_eq!(skeleton, want);
    assert!(!compile_function(&m, FuncId::from_index(0)).bytes.is_empty());
}

#[test]
fn a_extension_encodings_match_llvm_mc() {
    let cases: Vec<(u32, &str)> = vec![
        (lr(4, false, false, 5, 31), "lr.w t0, (t6)"),
        (lr(8, true, true, 10, 11), "lr.d.aqrl a0, (a1)"),
        (sc(4, false, true, 10, 6, 31), "sc.w.rl a0, t1, (t6)"),
        (sc(8, false, false, 5, 12, 13), "sc.d t0, a2, (a3)"),
        (amo(0b00001, true, true, 10, 11, 12, 8), "amoswap.d.aqrl a0, a1, (a2)"),
        (amo(0b00000, true, false, 10, 11, 12, 4), "amoadd.w.aq a0, a1, (a2)"),
        (amo(0b00100, false, true, 13, 14, 15, 8), "amoxor.d.rl a3, a4, (a5)"),
        (amo(0b01100, false, false, 8, 9, 18, 4), "amoand.w s0, s1, (s2)"),
        (amo(0b01000, false, false, 8, 9, 18, 8), "amoor.d s0, s1, (s2)"),
        (amo(0b10000, false, false, 8, 9, 18, 4), "amomin.w s0, s1, (s2)"),
        (amo(0b10100, false, false, 8, 9, 18, 8), "amomax.d s0, s1, (s2)"),
        (amo(0b11000, false, false, 8, 9, 18, 4), "amominu.w s0, s1, (s2)"),
        (amo(0b11100, true, true, 8, 9, 18, 8), "amomaxu.d.aqrl s0, s1, (s2)"),
        (fence(0, 0b0011, 0b0011), "fence rw, rw"),
        (fence(0, 0b0010, 0b0011), "fence r, rw"),
        (fence(0, 0b0011, 0b0001), "fence rw, w"),
        (fence(0b1000, 0b0011, 0b0011), "fence.tso"),
        (bcmp(5, 6, 10, 8), "bge t1, a0, 8"),
        (bcmp(7, 10, 6, 8), "bgeu a0, t1, 8"),
    ];
    let mut checked = 0;
    for (word, asm) in cases {
        match llvm_mc_rva(asm) {
            Some(bytes) => {
                assert_eq!(word.to_le_bytes().to_vec(), bytes, "`{asm}`");
                checked += 1;
            }
            None => eprintln!("skipping llvm-mc cross-check of `{asm}`"),
        }
    }
    eprintln!("checked {checked} RISC-V A-extension encodings against llvm-mc");
}

#[test]
fn atomic_sequences_match_llvm_mc() {
    use super::isel::RvOp;
    use super::regs::gpr;
    use crate::codegen::mir::{MachineInst, MachineOperand, Reg};
    use crate::ir::RmwOp;
    let d = |n: u16| MachineOperand::Def(Reg::Physical(gpr(n)));
    let u = |n: u16| MachineOperand::Use(Reg::Physical(gpr(n)));
    let k = |v: u64| MachineOperand::Imm(Int::from_u64(v));
    // d = a0 (x10), ptr = a1 (x11), val / expected = a2 (x12), new = a3 (x13).
    let rmw = |size: u64, op: RmwOp, aqrl: u64| {
        MachineInst::new(
            RvOp::AtomicRmw.opcode(),
            vec![d(10), u(11), u(12), k(size), k(u64::from(op.code())), k(aqrl)],
        )
    };
    let cas = |size: u64, aqrl: u64| {
        MachineInst::new(RvOp::CmpXchg.opcode(), vec![d(10), u(11), u(12), u(13), k(size), k(aqrl)])
    };
    let lane = "andi t6, a1, -4\nandi t2, a1, 3\nslli t2, t2, 3\n";
    let merge = |top: u32| {
        format!("sll t1, t1, t2\nxor t1, t1, t0\nsrl t1, t1, t2\nslli t1, t1, {top}\nsrli t1, t1, {top}\nsll t1, t1, t2\nxor t1, t1, t0\n")
    };
    let cases: Vec<(MachineInst, String)> = vec![
        (rmw(8, RmwOp::Add, 0b101), "amoadd.d.aqrl a0, a2, (a1)".into()),
        (rmw(4, RmwOp::Sub, 0b001), "neg t0, a2\namoadd.w.aq a0, t0, (a1)".into()),
        (rmw(4, RmwOp::UMax, 0b100), "amomaxu.w.rl a0, a2, (a1)".into()),
        (
            rmw(8, RmwOp::Nand, 0b111),
            "1:\nlr.d.aqrl a0, (a1)\nand t0, a0, a2\nnot t0, t0\nsc.d.rl t1, t0, (a1)\nbnez t1, 1b".into(),
        ),
        (
            rmw(1, RmwOp::Add, 0b111),
            format!(
                "{lane}1:\nlr.w.aqrl t0, (t6)\nsrl t1, t0, t2\nadd t1, t1, a2\n{}sc.w.rl a0, t1, (t6)\nbnez a0, 1b\nsrl a0, t0, t2",
                merge(56)
            ),
        ),
        (
            rmw(2, RmwOp::Max, 0b001),
            format!(
                "{lane}1:\nlr.w.aq t0, (t6)\nsrl t1, t0, t2\nslli t1, t1, 48\nslli a0, a2, 48\nbge t1, a0, 8\nmv t1, a0\nsrli t1, t1, 48\n{}sc.w a0, t1, (t6)\nbnez a0, 1b\nsrl a0, t0, t2",
                merge(48)
            ),
        ),
        (
            rmw(1, RmwOp::UMin, 0b000),
            format!(
                "{lane}1:\nlr.w t0, (t6)\nsrl t1, t0, t2\nslli t1, t1, 56\nslli a0, a2, 56\nbgeu a0, t1, 8\nmv t1, a0\nsrli t1, t1, 56\n{}sc.w a0, t1, (t6)\nbnez a0, 1b\nsrl a0, t0, t2",
                merge(56)
            ),
        ),
        (
            cas(8, 0b111),
            "1:\nlr.d.aqrl a0, (a1)\nbne a0, a2, 2f\nsc.d.rl t0, a3, (a1)\nbnez t0, 1b\n2:".into(),
        ),
        (
            cas(4, 0b001),
            "sext.w t1, a2\n1:\nlr.w.aq a0, (a1)\nbne a0, t1, 2f\nsc.w t0, a3, (a1)\nbnez t0, 1b\n2:".into(),
        ),
        (
            cas(2, 0b100),
            format!(
                "{lane}1:\nlr.w t0, (t6)\nsrl t1, t0, t2\nslli t1, t1, 48\nslli a0, a2, 48\nbne t1, a0, 2f\nmv t1, a3\n{}sc.w.rl a0, t1, (t6)\nbnez a0, 1b\n2:\nsrl a0, t0, t2",
                merge(48)
            ),
        ),
    ];
    let mut checked = 0;
    for (inst, asm) in &cases {
        let ours = encode_atomic_for_test(inst);
        match llvm_mc_rva(asm) {
            Some(want) => {
                assert_eq!(ours, want, "\n{asm}");
                checked += 1;
            }
            None => eprintln!("skipping llvm-mc cross-check of\n{asm}"),
        }
    }
    eprintln!("checked {checked} RISC-V atomic sequences against llvm-mc");
}

#[test]
fn every_lr_sc_loop_is_a_constrained_loop() {
    // The ISA guarantees forward progress only for a constrained LR/SC loop: at
    // most 16 instructions from the `lr` through the retry branch, and no
    // loads, stores or backward branches in between. Check every loop shape.
    use super::isel::RvOp;
    use super::regs::gpr;
    use crate::codegen::mir::{MachineInst, MachineOperand, Reg};
    use crate::ir::RmwOp;
    let d = |n: u16| MachineOperand::Def(Reg::Physical(gpr(n)));
    let u = |n: u16| MachineOperand::Use(Reg::Physical(gpr(n)));
    let k = |v: u64| MachineOperand::Imm(Int::from_u64(v));
    let mut insts = Vec::new();
    for size in [1u64, 2, 4, 8] {
        for op in RmwOp::ALL {
            insts.push(MachineInst::new(
                RvOp::AtomicRmw.opcode(),
                vec![d(10), u(11), u(12), k(size), k(u64::from(op.code())), k(0b111)],
            ));
        }
        insts.push(MachineInst::new(RvOp::CmpXchg.opcode(), vec![d(10), u(11), u(12), u(13), k(size), k(0b111)]));
    }
    let mut loops = 0;
    for inst in &insts {
        let words: Vec<u32> = encode_atomic_for_test(inst)
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let is_lr = |w: u32| w & 0x7F == 0x2F && (w >> 27) == 0b00010;
        let Some(start) = words.iter().position(|&w| is_lr(w)) else { continue };
        // The retry branch is the (only) backward branch.
        let is_branch = |w: u32| w & 0x7F == 0x63;
        let back = |w: u32| (w >> 31) == 1; // negative B-immediate
        let end = words.iter().position(|&w| is_branch(w) && back(w)).expect("a retry branch");
        let body = &words[start..=end];
        assert!(body.len() <= 16, "{} instructions in {inst:?}", body.len());
        for &w in &body[1..body.len() - 1] {
            let opc = w & 0x7F;
            assert!(opc != 0x03 && opc != 0x23, "no load/store inside the loop: {inst:?}");
            assert!(!(is_branch(w) && back(w)), "no other backward branch: {inst:?}");
            assert!(opc != 0x2F || (w >> 27) == 0b00011, "only the closing sc: {inst:?}");
        }
        loops += 1;
    }
    // nand at 4/8 bytes, every op at 1/2 bytes, cmpxchg at every size.
    assert_eq!(loops, 2 + 2 * 11 + 4);
}
