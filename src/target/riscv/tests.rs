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
        (div(10, 11, 12), "div a0, a1, a2"),
        (divu(10, 11, 12), "divu a0, a1, a2"),
        (rem(10, 11, 12), "rem a0, a1, a2"),
        (remu(10, 11, 12), "remu a0, a1, a2"),
        // I-type
        (addi(10, 10, 5), "addi a0, a0, 5"),
        (addi(10, 11, -5), "addi a0, a1, -5"),
        (addiw(10, 10, 5), "addiw a0, a0, 5"),
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
    // The `li` materialization (12-bit `addi`, 32-bit `lui`+`addiw`) must match the
    // assembler's `li` expansion byte-for-byte.
    if llvm_mc("ret").is_none() {
        eprintln!("skipping differential_li_materialization: no llvm-mc");
        return;
    }
    for val in [0i64, 5, -5, 2047, -2048, 0x12345, -0x12345, 0x7FFF_FFFF, -0x8000_0000] {
        let ours = emit_li_bytes(10, val);
        let asm = format!("li a0, {val}");
        let expected = llvm_mc(&asm).unwrap_or_else(|| panic!("llvm-mc failed on `{asm}`"));
        assert_eq!(ours, expected, "li a0, {val}: ours={ours:02x?} llvm={expected:02x?}");
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
