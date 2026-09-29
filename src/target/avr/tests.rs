//! Tests for the AVR backend.
//!
//! - **Encodings vs `llvm-mc`** (`--triple=avr -mcpu=atmega328p`): every
//!   instruction form the encoder emits (skipped without `llvm-mc`).
//! - **Execution** on the instruction-level interpreter ([`super::interp`]):
//!   IR programs are compiled, linked into a firmware image with the startup
//!   code and the runtime, and run; results are compared with the reference
//!   evaluator ([`crate::ir::eval`]) or host arithmetic.
//! - **Objects and firmware**: ELF output through `llvm-readobj` /
//!   `llvm-objdump` (when present), and whole images run from reset.

use super::interp::Avr;
use super::link::Firmware;
use super::regs::{self, ArgLoc};
use super::{Device, startup};
use crate::codegen::CodegenOptions;
use crate::ir::Module;
use crate::mc::object::ObjectModule;
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

const DEV: Device = Device::ATMEGA328P;
/// An AVR5 with 64 KiB of flash (the ATmega644 class), for scalarized vector
/// code, which outgrows 32 KiB.
const BIG: Device = Device { flash: 64 * 1024, ..DEV };
const BUDGET: u64 = 50_000_000;

/// Parse and verify `.lf` text.
fn parse(src: &str) -> (Module, StrInterner) {
    // Programs without a layout get AVR's (after the `module` line).
    let src = if src.contains("datalayout") {
        src.to_owned()
    } else {
        let src = src.trim_start();
        let nl = src.find('\n').unwrap_or(src.len());
        format!("{}\ndatalayout \"{}\"{}", &src[..nl], super::data_layout().to_spec(), &src[nl..])
    };
    let src = src.as_str();
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    (m, syms)
}

/// Compile `.lf` text for `dev`.
fn compile_for(src: &str, dev: &Device) -> ObjectModule {
    let (m, syms) = parse(src);
    super::compile_module_for_device(&m, &syms, &CodegenOptions::default(), dev).object
}

/// Compile and link `.lf` text into firmware (adding a `main` returning 0 if
/// the program has none).
fn firmware_for(src: &str, dev: &Device) -> Firmware {
    let mut objs = vec![compile_for(src, dev)];
    if !src.contains("@main(") {
        objs.push(compile_for("module \"m\"\nfunc @main() -> i16 {\nentry ^0:\n  ret i16 0\n}\n", dev));
    }
    super::link::build(objs, dev, "main").unwrap_or_else(|e| panic!("link: {e}"))
}

fn firmware(src: &str) -> Firmware {
    firmware_for(src, &DEV)
}

/// Boot `fw` (startup code and `main`), then leave the machine at the stop loop.
fn boot(fw: &Firmware, dev: &Device) -> (Avr, u32) {
    let stop = fw.symbol(startup::STOP).expect("the stop symbol") / 2;
    let mut m = Avr::new(&fw.flash, dev);
    m.run_until(stop, BUDGET).unwrap_or_else(|e| panic!("boot: {e} {m:?}"));
    (m, stop)
}

/// Run firmware from reset; return main's `r25:r24` and the machine.
fn run_main(fw: &Firmware) -> (u16, Avr) {
    let (m, _) = boot(fw, &DEV);
    (u16::from(m.reg(24)) | (u16::from(m.reg(25)) << 8), m)
}

/// Call `name` in a booted machine with `args` (value, byte size) per the
/// avr-gcc convention, returning the `ret` low bytes of the result.
fn call_in(m: &mut Avr, fw: &Firmware, dev: &Device, stop: u32, name: &str, args: &[(u64, u64)], ret: u64) -> u64 {
    let addr = fw.symbol(name).unwrap_or_else(|| panic!("no symbol {name}"));
    let sizes: Vec<u64> = args.iter().map(|a| a.1).collect();
    let (locs, stack) = regs::assign_args(&sizes);
    m.set_sp(dev.ram_end);
    m.set_reg(1, 0);
    let sp = dev.ram_end - stack as u16;
    for ((&(v, size), loc), _) in args.iter().zip(&locs).zip(0..) {
        match *loc {
            ArgLoc::Regs(base) => {
                for k in 0..size {
                    m.set_reg(usize::from(base) + k as usize, (v >> (8 * k)) as u8);
                }
            }
            ArgLoc::Stack(off) => {
                for k in 0..size {
                    m.data[usize::from(sp) + 1 + (off + k) as usize] = (v >> (8 * k)) as u8;
                }
            }
        }
    }
    m.set_sp(sp);
    m.push_ret(stop).expect("push the return address");
    m.pc = addr / 2;
    m.run_until(stop, BUDGET).unwrap_or_else(|e| panic!("call {name}{args:?}: {e} {m:?}"));
    assert_eq!(m.sp(), sp, "{name}: the stack pointer is restored");
    assert_eq!(m.reg(1), 0, "{name}: r1 is zero on return");
    let base = usize::from(regs::ret_base(ret));
    (0..ret).fold(0u64, |acc, k| acc | (u64::from(m.reg(base + k as usize)) << (8 * k)))
}

/// Call `name` in a fresh boot of `fw`.
fn call(fw: &Firmware, name: &str, args: &[(u64, u64)], ret: u64) -> u64 {
    let (mut m, stop) = boot(fw, &DEV);
    call_in(&mut m, fw, &DEV, stop, name, args, ret)
}

#[test]
fn smoke_add_and_main() {
    let fw = firmware(
        r#"
module "t"
func @add(i16, i16) -> i16 {
entry ^0(%a: i16, %b: i16):
  %s = add %a, %b : i16
  ret %s
}
func @main() -> i16 {
entry ^0:
  %r = call @add(i16 40, i16 2) : i16
  ret %r
}
"#,
    );
    assert_eq!(run_main(&fw).0, 42);
    assert_eq!(call(&fw, "add", &[(1000, 2), (234, 2)], 2), 1234);
}

// ===========================================================================
// Encodings vs llvm-mc
// ===========================================================================

use super::encode::*;

/// The scratch directory for tool round trips.
fn scratch_file(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("lf_avr_{tag}_{}_{n}", std::process::id()))
}

fn have(tool: &str) -> bool {
    std::process::Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// Assemble `asm` (one or more instructions) with `llvm-mc`, returning the
/// bytes of every `encoding: [..]` group, or `None` without `llvm-mc`.
fn llvm_mc(asm: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let mut child = std::process::Command::new("llvm-mc")
        .args(["--triple=avr", "-mcpu=atmega328p", "--show-encoding"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(asm.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    assert!(out.status.success(), "llvm-mc rejected the input: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let mut bytes = Vec::new();
    let mut rest = &text[..];
    while let Some(pos) = rest.find("encoding: [") {
        let start = pos + "encoding: [".len();
        let end = rest[start..].find(']')? + start;
        for tok in rest[start..end].split(',') {
            bytes.push(u8::from_str_radix(tok.trim().trim_start_matches("0x"), 16).ok()?);
        }
        rest = &rest[end..];
    }
    Some(bytes)
}

fn le(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Every non-branch instruction form the encoder, prologue/epilogue and
/// startup code emit, with registers chosen to exercise every field bit.
fn corpus() -> Vec<(Vec<u16>, String)> {
    let mut c: Vec<(Vec<u16>, String)> = Vec::new();
    let mut add1 = |w: u16, s: String| c.push((vec![w], s));
    for (d, r) in [(0u8, 1u8), (24, 22), (17, 31), (30, 16), (1, 0), (15, 2)] {
        add1(add(d, r), format!("add r{d}, r{r}"));
        add1(adc(d, r), format!("adc r{d}, r{r}"));
        add1(sub(d, r), format!("sub r{d}, r{r}"));
        add1(sbc(d, r), format!("sbc r{d}, r{r}"));
        add1(and(d, r), format!("and r{d}, r{r}"));
        add1(or(d, r), format!("or r{d}, r{r}"));
        add1(eor(d, r), format!("eor r{d}, r{r}"));
        add1(mov(d, r), format!("mov r{d}, r{r}"));
        add1(cp(d, r), format!("cp r{d}, r{r}"));
        add1(cpc(d, r), format!("cpc r{d}, r{r}"));
        add1(mul(d, r), format!("mul r{d}, r{r}"));
    }
    for d in [0u8, 5, 16, 24, 31] {
        add1(lsl(d), format!("lsl r{d}"));
        add1(rol(d), format!("rol r{d}"));
        add1(tst(d), format!("tst r{d}"));
        add1(com(d), format!("com r{d}"));
        add1(asr(d), format!("asr r{d}"));
        add1(lsr(d), format!("lsr r{d}"));
        add1(ror(d), format!("ror r{d}"));
        add1(neg(d), format!("neg r{d}"));
        add1(swap(d), format!("swap r{d}"));
        add1(push(d), format!("push r{d}"));
        add1(pop(d), format!("pop r{d}"));
        add1(lpm(d), format!("lpm r{d}, Z"));
        add1(lpm_inc(d), format!("lpm r{d}, Z+"));
        add1(st_x_inc(d), format!("st X+, r{d}"));
        add1(in_(d, 0x3d), format!("in r{d}, 0x3d"));
        add1(in_(d, 0x3e), format!("in r{d}, 0x3e"));
        add1(out(0x3f, d), format!("out 0x3f, r{d}"));
        add1(out(0x3e, d), format!("out 0x3e, r{d}"));
        add1(out(0x3d, d), format!("out 0x3d, r{d}"));
        for b in [0u8, 3, 7] {
            add1(bst(d, b), format!("bst r{d}, {b}"));
            add1(bld(d, b), format!("bld r{d}, {b}"));
        }
    }
    for (d, k) in [(16u8, 0u8), (17, 0xff), (30, 0x5a), (31, 0xa5), (28, 1), (26, 0x80)] {
        add1(ldi(d, k), format!("ldi r{d}, {k}"));
        add1(cpi(d, k), format!("cpi r{d}, {k}"));
        add1(subi(d, k), format!("subi r{d}, {k}"));
        add1(sbci(d, k), format!("sbci r{d}, {k}"));
        add1(andi(d, k), format!("andi r{d}, {k}"));
    }
    for (d, r) in [(0u8, 30u8), (24, 22), (30, 28), (2, 0), (18, 26)] {
        add1(movw(d, r), format!("movw r{d}, r{r}"));
    }
    for (d, k) in [(24u8, 1u8), (26, 17), (28, 63), (30, 32)] {
        add1(adiw(d, k), format!("adiw r{d}, {k}"));
        add1(sbiw(d, k), format!("sbiw r{d}, {k}"));
    }
    for (d, q) in [(0u8, 0u8), (24, 1), (31, 63), (7, 8), (16, 33), (2, 62)] {
        add1(ldd(d, true, q), format!("ldd r{d}, Y+{q}"));
        add1(ldd(d, false, q), format!("ldd r{d}, Z+{q}"));
        add1(std(true, q, d), format!("std Y+{q}, r{d}"));
        add1(std(false, q, d), format!("std Z+{q}, r{d}"));
    }
    for k in [0u32, 0x1234 / 2, 0x7ffe / 2, 0x3f_ffff] {
        c.push((jmp(k).to_vec(), format!("jmp {}", 2 * k)));
        c.push((super::encode::call(k).to_vec(), format!("call {}", 2 * k)));
    }
    for (w, s) in [(ICALL, "icall"), (RET, "ret"), (CLI, "cli"), (BREAK, "break")] {
        c.push((vec![w], s.to_owned()));
    }
    c
}

#[test]
fn encodings_match_llvm_mc() {
    if !have("llvm-mc") {
        eprintln!("skipping: no llvm-mc");
        return;
    }
    let corpus = corpus();
    // One llvm-mc run for the whole corpus: the encodings come back in order.
    let asm: String = corpus.iter().map(|(_, s)| format!("{s}\n")).collect();
    let got = llvm_mc(&asm).expect("llvm-mc output");
    let ours: Vec<u8> = corpus.iter().flat_map(|(w, _)| le(w)).collect();
    assert_eq!(ours.len(), got.len(), "llvm-mc encoded a different number of bytes");
    let mut at = 0;
    for (w, s) in &corpus {
        let n = 2 * w.len();
        assert_eq!(&ours[at..at + n], &got[at..at + n], "encoding mismatch for `{s}`");
        at += n;
    }
    eprintln!("avr encodings: {} instructions match llvm-mc", corpus.len());
    assert!(corpus.len() > 250);
}

/// Wrap `words` in an ELF object with our writer and disassemble it with
/// `llvm-objdump`, returning one `mnemonic operands` string per instruction.
fn objdump_words(words: &[u16]) -> Option<Vec<String>> {
    use crate::mc::object::{Section, SectionKind};
    let mut obj = ObjectModule::new("w");
    let mut sec = Section::new(".text", SectionKind::Text, 2);
    sec.bytes = le(words);
    obj.add_section(sec);
    objdump_object(&obj)
}

fn objdump_object(obj: &ObjectModule) -> Option<Vec<String>> {
    if !have("llvm-objdump") {
        return None;
    }
    let path = scratch_file("dis.o");
    std::fs::write(&path, super::write_elf(obj).expect("write ELF")).unwrap();
    let out = std::process::Command::new("llvm-objdump")
        .args(["-d", "--triple=avr", "--mcpu=atmega328p", "--no-show-raw-insn", "--no-leading-addr"])
        .arg(&path)
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&path);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    Some(
        text.lines()
            .filter(|l| l.starts_with(char::is_whitespace) && !l.trim().is_empty())
            .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
            .collect(),
    )
}

#[test]
fn relative_branches_match_llvm_objdump() {
    let mut words = Vec::new();
    let mut want = Vec::new();
    let rel = |k: i32| if k >= 0 { format!(".+{}", 2 * k) } else { format!(".-{}", -2 * k) };
    for k in [-64, -1, 0, 1, 5, 63] {
        for (flag, set, name) in [
            (FLAG_Z, true, "breq"),
            (FLAG_Z, false, "brne"),
            (FLAG_C, true, "brlo"),
            (FLAG_C, false, "brsh"),
            (FLAG_S, true, "brlt"),
            (FLAG_S, false, "brge"),
        ] {
            words.push(br(flag, set, k));
            want.push(format!("{name} {}", rel(k)));
        }
    }
    for k in [-2048, -1, 0, 2, 2047] {
        words.push(rjmp(k));
        want.push(format!("rjmp {}", rel(k)));
    }
    let Some(got) = objdump_words(&words) else {
        eprintln!("skipping: no llvm-objdump");
        return;
    };
    assert_eq!(got, want);
}

/// Every word of every function the backend compiles for a varied program
/// disassembles (no `<unknown>`): isel + encoding produce only real
/// instructions.
#[test]
fn compiled_code_disassembles() {
    let obj = compile_for(PROGRAM, &DEV);
    let Some(lines) = objdump_object(&obj) else {
        eprintln!("skipping: no llvm-objdump");
        return;
    };
    assert!(lines.len() > 100, "{lines:?}");
    for l in &lines {
        assert!(!l.contains("unknown"), "undecodable instruction: {l}");
    }
}

/// A program touching most of the backend (calls, globals in both memories,
/// loops, wide arithmetic, switches, selects).
const PROGRAM: &str = r#"
module "prog"
global @counter : i16 = i16 7
global @big : i32 = i32 100000
global constant addrspace(1) @table : [4 x i16] = [4 x i16] (i16 10, i16 20, i16 30, i16 40)
global @zeros : [8 x i8] = [8 x i8] (i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0)

func @lookup(i16) -> i16 {
entry ^0(%i: i16):
  %off = shl %i, i16 1 : i16
  %p = ptr_add @table, %off : ptr addrspace(1)
  %v = load %p align 1 : i16
  ret %v
}

func @sum32(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %s = add %a, %b : i32
  %m = mul %s, i32 3 : i32
  %d = udiv %m, i32 7 : i32
  ret %d
}

func @classify(i8) -> i8 {
entry ^0(%x: i8):
  switch %x, ^3 [1: ^1, 2: ^2]
^1:
  ret i8 10
^2:
  ret i8 20
^3:
  %lt = icmp slt %x, i8 0 : i1
  %r = select %lt, i8 -1, i8 0 : i8
  ret %r
}

func @main() -> i16 {
entry ^0:
  br ^1(i16 0, i16 0)
^1(%i: i16, %acc: i16):
  %v = call @lookup(%i) : i16
  %acc2 = add %acc, %v : i16
  %i2 = add %i, i16 1 : i16
  %done = icmp eq %i2, i16 4 : i1
  cond_br %done, ^2, ^1(%i2, %acc2)
^2:
  %c = load @counter align 1 : i16
  %t = add %acc2, %c : i16
  store %t, @counter align 1 : i16
  %w = call @sum32(i32 1, i32 6) : i32
  %w16 = trunc %w : i16
  %r = add %t, %w16 : i16
  ret %r
}
"#;

#[test]
fn whole_program_runs_from_reset() {
    let fw = firmware(PROGRAM);
    // 10+20+30+40 = 100, + 7 = 107, + (1+6)*3/7 = 3 → 110.
    let (r, m) = run_main(&fw);
    assert_eq!(r, 110);
    // The global in SRAM was updated (107) and the 32-bit one initialized.
    let c = fw.symbol("counter").unwrap() as usize;
    assert_eq!(u16::from_le_bytes([m.data[c], m.data[c + 1]]), 107);
    let b = fw.symbol("big").unwrap() as usize;
    assert_eq!(u32::from_le_bytes(m.data[b..b + 4].try_into().unwrap()), 100_000);
    // .bss is zeroed (the SRAM starts with garbage in the interpreter).
    let z = fw.symbol("zeros").unwrap() as usize;
    assert_eq!(&m.data[z..z + 8], &[0; 8]);
    // The flash table is in flash, not SRAM.
    assert!(fw.symbol("table").unwrap() < fw.symbol("__data_load_start").unwrap());
    assert_eq!(call(&fw, "classify", &[(1, 1)], 1), 10);
    assert_eq!(call(&fw, "classify", &[(2, 1)], 1), 20);
    assert_eq!(call(&fw, "classify", &[(0xfb, 1)], 1), 0xff);
    assert_eq!(call(&fw, "classify", &[(9, 1)], 1), 0);
}

// ===========================================================================
// Differential execution vs the reference evaluator
// ===========================================================================

use crate::ir::inst::{BinOp, Flags, InstKind, IntPred};
use crate::ir::{EvalOutcome, SemValue};
use puremp::Int;

/// Interesting operands of a `w`-bit type: the edges, and a few values from a
/// fixed LCG.
fn samples(w: u32, extra: usize) -> Vec<u64> {
    let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    let mut v = vec![0, 1, 2, 3, mask, mask - 1, mask >> 1, (mask >> 1) + 1, 0x7f, 0x80, 0xff, 0x100, 0x8000, 0xffff, 0x1_0000];
    let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ u64::from(w);
    for _ in 0..extra {
        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        v.push(x >> (x % 61));
    }
    let mut out: Vec<u64> = v.into_iter().map(|a| a & mask).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The reference result of `kind` on two `w`-bit operands, or `None` for
/// poison / undefined behavior.
fn reference(kind: &InstKind, w: u32, ret: u32, a: u64, b: u64) -> Option<u64> {
    let mut tc = crate::ir::TypeContext::new();
    let rt = tc.int(ret);
    let ops = [SemValue::int(w, Int::from_u64(a)), SemValue::int(w, Int::from_u64(b))];
    match crate::ir::eval(&tc, rt, kind, &Flags::NONE, &ops) {
        EvalOutcome::Value(SemValue::Int { bits, .. }) => bits.to_u64(),
        _ => None,
    }
}

const BIN_OPS: [(&str, BinOp); 13] = [
    ("add", BinOp::Add),
    ("sub", BinOp::Sub),
    ("mul", BinOp::Mul),
    ("udiv", BinOp::UDiv),
    ("sdiv", BinOp::SDiv),
    ("urem", BinOp::URem),
    ("srem", BinOp::SRem),
    ("and", BinOp::And),
    ("or", BinOp::Or),
    ("xor", BinOp::Xor),
    ("shl", BinOp::Shl),
    ("lshr", BinOp::LShr),
    ("ashr", BinOp::AShr),
];

const PREDS: [(&str, IntPred); 10] = [
    ("eq", IntPred::Eq),
    ("ne", IntPred::Ne),
    ("ult", IntPred::Ult),
    ("ule", IntPred::Ule),
    ("ugt", IntPred::Ugt),
    ("uge", IntPred::Uge),
    ("slt", IntPred::Slt),
    ("sle", IntPred::Sle),
    ("sgt", IntPred::Sgt),
    ("sge", IntPred::Sge),
];

/// A module with one function per binary op and per compare on `iW`.
fn arith_module(w: u32) -> String {
    let t = format!("i{w}");
    let mut s = String::from("module \"arith\"\n");
    for (name, _) in BIN_OPS {
        s.push_str(&format!(
            "func @op_{name}({t}, {t}) -> {t} {{\nentry ^0(%a: {t}, %b: {t}):\n  %r = {name} %a, %b : {t}\n  ret %r\n}}\n"
        ));
    }
    for (name, _) in PREDS {
        s.push_str(&format!(
            "func @cmp_{name}({t}, {t}) -> i1 {{\nentry ^0(%a: {t}, %b: {t}):\n  %r = icmp {name} %a, %b : i1\n  ret %r\n}}\n"
        ));
    }
    s
}

/// `v` with garbage in the bits of its ABI container above `w` (narrow values
/// do not keep those bits clean, and the backend must not rely on them). An
/// `i1` is passed clean (0/1), as the ABI requires.
fn dirty(v: u64, w: u32) -> u64 {
    if w == 1 || w.is_multiple_of(8) {
        return v;
    }
    let bytes = w.div_ceil(8);
    let garbage = 0xa5a5_a5a5_a5a5_a5a5u64 & ((1u64 << (8 * bytes)) - 1);
    v | (garbage & !((1u64 << w) - 1))
}

fn check_arith(w: u32, dev: &Device, extra: usize) -> usize {
    let fw = firmware_for(&arith_module(w), dev);
    let (mut m, stop) = boot(&fw, dev);
    let size = u64::from(w.div_ceil(8));
    let mask = |bits: u32, v: u64| if bits == 64 { v } else { v & ((1u64 << bits) - 1) };
    let vals = samples(w, extra);
    let mut checked = 0;
    for &a in &vals {
        for &b in &vals {
            let args = [(dirty(a, w), size), (dirty(b, w), size)];
            for (name, op) in BIN_OPS {
                let Some(want) = reference(&InstKind::Bin(op), w, w, a, b) else { continue };
                let got = mask(w, call_in(&mut m, &fw, dev, stop, &format!("op_{name}"), &args, size));
                assert_eq!(got, want, "i{w} {name} {a:#x}, {b:#x}");
                checked += 1;
            }
            for (name, pred) in PREDS {
                let want = reference(&InstKind::ICmp(pred), w, 1, a, b).expect("a compare is defined");
                let got = call_in(&mut m, &fw, dev, stop, &format!("cmp_{name}"), &args, 1);
                assert_eq!(got, want, "i{w} icmp {name} {a:#x}, {b:#x} (the result must be a clean 0/1)");
                checked += 1;
            }
        }
    }
    checked
}

#[test]
fn arithmetic_8_and_16_bits_matches_the_reference() {
    let n = check_arith(8, &DEV, 6) + check_arith(16, &DEV, 6);
    eprintln!("avr i8/i16 arithmetic: {n} results checked");
}

#[test]
fn arithmetic_32_bits_matches_the_reference() {
    let n = check_arith(32, &DEV, 4);
    eprintln!("avr i32 arithmetic: {n} results checked");
}

#[test]
fn arithmetic_64_bits_matches_the_reference() {
    let n = check_arith(64, &DEV, 2);
    eprintln!("avr i64 arithmetic: {n} results checked");
}

#[test]
fn odd_narrow_widths_extend_before_use() {
    let n: usize = [1, 4, 12].into_iter().map(|w| check_arith(w, &DEV, 3)).sum();
    eprintln!("avr i1/i4/i12 arithmetic: {n} results checked");
}

/// A core without the multiplier: every multiply is a runtime call.
#[test]
fn arithmetic_without_hardware_multiply() {
    let dev = Device::AVR3;
    let n = check_arith(8, &dev, 4) + check_arith(16, &dev, 4);
    // No `mul` instruction anywhere in the code.
    let obj = compile_for(&arith_module(16), &dev);
    for w in obj.sections()[0].bytes.chunks(2) {
        let w = u16::from_le_bytes([w[0], w[1]]);
        assert_ne!(w & 0xfc00, 0x9c00, "a `mul` in code for a core without one");
    }
    eprintln!("avr no-mul arithmetic: {n} results checked");
}

#[test]
fn casts_match_the_reference() {
    use crate::ir::inst::CastOp;
    let widths = [1u32, 4, 8, 12, 16, 32, 64];
    let mut src = String::from("module \"casts\"\n");
    let mut cases = Vec::new();
    for &f in &widths {
        for &t in &widths {
            let ops: &[(&str, CastOp)] = match f.cmp(&t) {
                std::cmp::Ordering::Less => &[("zext", CastOp::ZExt), ("sext", CastOp::SExt)],
                std::cmp::Ordering::Greater => &[("trunc", CastOp::Trunc)],
                std::cmp::Ordering::Equal => &[],
            };
            for &(name, op) in ops {
                src.push_str(&format!(
                    "func @{name}_{f}_{t}(i{f}) -> i{t} {{\nentry ^0(%a: i{f}):\n  %r = {name} %a : i{t}\n  ret %r\n}}\n"
                ));
                cases.push((name, op, f, t));
            }
        }
    }
    let fw = firmware(&src);
    let (mut m, stop) = boot(&fw, &DEV);
    let mut n = 0;
    for (name, op, f, t) in cases {
        for a in samples(f, 3) {
            let mut tc = crate::ir::TypeContext::new();
            let rt = tc.int(t);
            let want = match crate::ir::eval(&tc, rt, &InstKind::Cast(op), &Flags::NONE, &[SemValue::int(f, Int::from_u64(a))]) {
                EvalOutcome::Value(SemValue::Int { bits, .. }) => bits.to_u64().unwrap(),
                other => panic!("{other:?}"),
            };
            let got = call_in(&mut m, &fw, &DEV, stop, &format!("{name}_{f}_{t}"), &[(dirty(a, f), u64::from(f.div_ceil(8)))], u64::from(t.div_ceil(8)));
            let got = if t == 64 { got } else { got & ((1u64 << t) - 1) };
            assert_eq!(got, want, "{name} i{f} {a:#x} to i{t}");
            n += 1;
        }
    }
    eprintln!("avr casts: {n} results checked");
}

const CONTROL: &str = r#"
module "control"
func @fib(i16) -> i16 {
entry ^0(%n: i16):
  %small = icmp ult %n, i16 2 : i1
  cond_br %small, ^1, ^2
^1:
  ret %n
^2:
  %a = sub %n, i16 1 : i16
  %b = sub %n, i16 2 : i16
  %fa = call @fib(%a) : i16
  %fb = call @fib(%b) : i16
  %r = add %fa, %fb : i16
  ret %r
}

func @fact(i32) -> i64 {
entry ^0(%n: i32):
  %z = icmp eq %n, i32 0 : i1
  cond_br %z, ^1, ^2
^1:
  ret i64 1
^2:
  %m = sub %n, i32 1 : i32
  %f = call @fact(%m) : i64
  %w = zext %n : i64
  %r = mul %f, %w : i64
  ret %r
}

func @sum_to(i16) -> i64 {
entry ^0(%n: i16):
  br ^1(i16 0, i64 0)
^1(%i: i16, %acc: i64):
  %done = icmp sgt %i, %n : i1
  cond_br %done, ^3(%acc), ^2
^2:
  %w = sext %i : i64
  %sq = mul %w, %w : i64
  %acc2 = add %acc, %sq : i64
  %i2 = add %i, i16 1 : i16
  br ^1(%i2, %acc2)
^3(%r: i64):
  ret %r
}

func @many(i8, i16, i32, i64, i16, i16, i8, i32, i16, i64, i8) -> i64 {
entry ^0(%a: i8, %b: i16, %c: i32, %d: i64, %e: i16, %f: i16, %g: i8, %h: i32, %i: i16, %j: i64, %k: i8):
  %a1 = sext %a : i64
  %b1 = sext %b : i64
  %c1 = sext %c : i64
  %e1 = zext %e : i64
  %f1 = zext %f : i64
  %g1 = zext %g : i64
  %h1 = zext %h : i64
  %i1 = sext %i : i64
  %k1 = sext %k : i64
  %s1 = mul %a1, i64 3 : i64
  %s2 = mul %b1, i64 5 : i64
  %s3 = mul %c1, i64 7 : i64
  %s4 = mul %d, i64 11 : i64
  %s5 = mul %e1, i64 13 : i64
  %s6 = mul %f1, i64 17 : i64
  %s7 = mul %g1, i64 19 : i64
  %s8 = mul %h1, i64 23 : i64
  %s9 = mul %i1, i64 29 : i64
  %s10 = mul %j, i64 31 : i64
  %s11 = mul %k1, i64 37 : i64
  %t1 = add %s1, %s2 : i64
  %t2 = add %t1, %s3 : i64
  %t3 = add %t2, %s4 : i64
  %t4 = add %t3, %s5 : i64
  %t5 = add %t4, %s6 : i64
  %t6 = add %t5, %s7 : i64
  %t7 = add %t6, %s8 : i64
  %t8 = add %t7, %s9 : i64
  %t9 = add %t8, %s10 : i64
  %t10 = add %t9, %s11 : i64
  ret %t10
}

func @call_many() -> i64 {
entry ^0:
  %r = call @many(i8 -3, i16 1000, i32 -70000, i64 123456789012, i16 65535, i16 7, i8 200, i32 3000000000, i16 -2, i64 -5, i8 9) : i64
  ret %r
}

func @wide_switch(i32) -> i16 {
entry ^0(%x: i32):
  switch %x, ^4 [0: ^1, 70000: ^2, -1: ^3]
^1:
  ret i16 1
^2:
  ret i16 2
^3:
  ret i16 3
^4:
  ret i16 4
}

func @sel64(i1, i64, i64) -> i64 {
entry ^0(%c: i1, %a: i64, %b: i64):
  %r = select %c, %a, %b : i64
  ret %r
}

func @narrow_switch(i4) -> i8 {
entry ^0(%x: i4):
  switch %x, ^3 [3: ^1, -1: ^2]
^1:
  ret i8 30
^2:
  ret i8 15
^3:
  ret i8 0
}
"#;

#[allow(clippy::too_many_arguments)]
fn many_host(a: i8, b: i16, c: i32, d: i64, e: u16, f: u16, g: u8, h: u32, i: i16, j: i64, k: i8) -> i64 {
    let terms = [
        i64::from(a) * 3,
        i64::from(b) * 5,
        i64::from(c) * 7,
        d.wrapping_mul(11),
        i64::from(e) * 13,
        i64::from(f) * 17,
        i64::from(g) * 19,
        i64::from(h) * 23,
        i64::from(i) * 29,
        j.wrapping_mul(31),
        i64::from(k) * 37,
    ];
    terms.iter().fold(0i64, |s, t| s.wrapping_add(*t))
}

#[test]
fn control_flow_recursion_and_calls() {
    let fw = firmware(CONTROL);
    let (mut m, stop) = boot(&fw, &DEV);
    let mut run = |name: &str, args: &[(u64, u64)], ret: u64| call_in(&mut m, &fw, &DEV, stop, name, args, ret);
    let fibs = [0u64, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610];
    for (n, &f) in fibs.iter().enumerate() {
        assert_eq!(run("fib", &[(n as u64, 2)], 2), f, "fib({n})");
    }
    let mut f = 1u64;
    for n in 0..=20u64 {
        if n > 0 {
            f *= n;
        }
        assert_eq!(run("fact", &[(n, 4)], 8), f, "fact({n})");
    }
    for n in [0i64, 1, 10, 100, 1000] {
        let want: i64 = (0..=n).map(|i| i * i).sum();
        assert_eq!(run("sum_to", &[(n as u64, 2)], 8), want as u64, "sum_to({n})");
    }
    // Eleven arguments: r24, r22, r18..r21, r10..r17, r8:r9, then the stack
    // (i16, i8, i32, i16, i64, i8 do not fit in the registers left).
    let want = many_host(-3, 1000, -70000, 123_456_789_012, 65535, 7, 200, 3_000_000_000, -2, -5, 9);
    let args = [
        (0xfd, 1),
        (1000, 2),
        ((-70000i32) as u32 as u64, 4),
        (123_456_789_012, 8),
        (65535, 2),
        (7, 2),
        (200, 1),
        (3_000_000_000, 4),
        ((-2i16) as u16 as u64, 2),
        ((-5i64) as u64, 8),
        (9, 1),
    ];
    assert_eq!(run("many", &args, 8), want as u64, "callee-side stack arguments");
    assert_eq!(run("call_many", &[], 8), want as u64, "caller-side stack arguments");
    for (x, want) in [(0u64, 1), (70000, 2), (0xffff_ffff, 3), (5, 4), (70000 + 65536, 4), (0x1_0000, 4)] {
        assert_eq!(run("wide_switch", &[(x, 4)], 2), want, "wide_switch({x:#x})");
    }
    assert_eq!(run("sel64", &[(1, 1), (0x1122_3344_5566_7788, 8), (9, 8)], 8), 0x1122_3344_5566_7788);
    assert_eq!(run("sel64", &[(0, 1), (0x1122_3344_5566_7788, 8), (9, 8)], 8), 9);
    for (x, want) in [(3u64, 30), (0xf3, 30), (0xf, 15), (0x5f, 15), (4, 0)] {
        assert_eq!(run("narrow_switch", &[(x, 1)], 1), want, "narrow_switch({x:#x}) ignores bits above i4");
    }
}

const MEMORY: &str = r#"
module "memory"
global @arr : [6 x i32] = [6 x i32] (i32 1, i32 -2, i32 300000, i32 4, i32 5, i32 6)
global constant addrspace(1) @farr : [4 x i32] = [4 x i32] (i32 10, i32 20, i32 -30, i32 1000000)
global constant addrspace(1) @fns : [2 x ptr addrspace(1)] = [2 x ptr addrspace(1)] (ptr addrspace(1) @twice, ptr addrspace(1) @thrice)
global @bytes : [4 x i8] = [4 x i8] (i8 1, i8 2, i8 3, i8 4)
global @flag : i16 = i16 0

func @twice(i16) -> i16 {
entry ^0(%x: i16):
  %r = shl %x, i16 1 : i16
  ret %r
}

func @thrice(i16) -> i16 {
entry ^0(%x: i16):
  %r = mul %x, i16 3 : i16
  ret %r
}

func @sum_arr() -> i32 {
entry ^0:
  br ^1(i16 0, i32 0)
^1(%i: i16, %acc: i32):
  %off = mul %i, i16 4 : i16
  %p = ptr_add @arr, %off : ptr
  %v = load %p align 1 : i32
  %acc2 = add %acc, %v : i32
  %i2 = add %i, i16 1 : i16
  %done = icmp eq %i2, i16 6 : i1
  cond_br %done, ^2(%acc2), ^1(%i2, %acc2)
^2(%r: i32):
  ret %r
}

func @flash_at(i16) -> i32 {
entry ^0(%i: i16):
  %off = shl %i, i16 2 : i16
  %p = ptr_add @farr, %off : ptr addrspace(1)
  %v = load %p align 1 : i32
  ret %v
}

func @dispatch(i16, i16) -> i16 {
entry ^0(%k: i16, %x: i16):
  %off = shl %k, i16 1 : i16
  %p = ptr_add @fns, %off : ptr addrspace(1)
  %f = load %p align 1 : ptr addrspace(1)
  %r = call %f(%x) : i16
  ret %r
}

func @pick(i1, i16) -> i16 {
entry ^0(%c: i1, %x: i16):
  %f = select %c, @twice, @thrice : ptr addrspace(1)
  %r = call %f(%x) : i16
  ret %r
}

func @locals(i16) -> i16 {
entry ^0(%n: i16):
  %buf = alloca [8 x i16] : ptr
  br ^1(i16 0)
^1(%i: i16):
  %off = shl %i, i16 1 : i16
  %p = ptr_add %buf, %off : ptr
  %v = mul %i, %n : i16
  store %v, %p align 1 : i16
  %i2 = add %i, i16 1 : i16
  %d = icmp eq %i2, i16 8 : i1
  cond_br %d, ^2(i16 0, i16 0), ^1(%i2)
^2(%j: i16, %acc: i16):
  %off2 = shl %j, i16 1 : i16
  %q = ptr_add %buf, %off2 : ptr
  %w = load %q align 1 : i16
  %acc2 = add %acc, %w : i16
  %j2 = add %j, i16 1 : i16
  %e = icmp eq %j2, i16 8 : i1
  cond_br %e, ^3(%acc2), ^2(%j2, %acc2)
^3(%r: i16):
  ret %r
}

func @dyn(i16) -> i16 {
entry ^0(%n: i16):
  %p = dyn_alloca %n align 1 : ptr
  store i8 7, %p align 1 : i8
  %last = sub %n, i16 1 : i16
  %q = ptr_add %p, %last : ptr
  store i8 9, %q align 1 : i8
  %a = load %p align 1 : i8
  %b = load %q align 1 : i8
  %s = add %a, %b : i8
  %r = zext %s : i16
  ret %r
}

func @atomics() -> i16 {
entry ^0:
  %o1 = atomic_rmw add seq_cst @flag, i16 300 align 2 : i16
  %o2 = atomic_rmw max seq_cst @flag, i16 -5 align 2 : i16
  %o3 = atomic_rmw umax seq_cst @flag, i16 -5 align 2 : i16
  %o4 = cmpxchg seq_cst seq_cst @flag, i16 -5, i16 42 align 2 : i16
  %o5 = cmpxchg seq_cst seq_cst @flag, i16 -5, i16 99 align 2 : i16
  %b0 = atomic_rmw sub seq_cst @bytes, i8 5 align 1 : i8
  %b1 = atomic_load seq_cst @flag align 2 : i16
  fence seq_cst
  %x1 = add %o1, %o2 : i16
  %x2 = add %x1, %o3 : i16
  %x3 = add %x2, %o4 : i16
  %x4 = add %x3, %o5 : i16
  %bw = zext %b0 : i16
  %x5 = add %x4, %bw : i16
  %x6 = add %x5, %b1 : i16
  ret %x6
}
"#;

#[test]
fn memory_globals_function_pointers_and_atomics() {
    let fw = firmware(MEMORY);
    let (mut m, stop) = boot(&fw, &DEV);
    let run = |m: &mut Avr, name: &str, args: &[(u64, u64)], ret: u64| call_in(m, &fw, &DEV, stop, name, args, ret);
    assert_eq!(run(&mut m, "sum_arr", &[], 4), 300_014);
    for (i, v) in [10i32, 20, -30, 1_000_000].into_iter().enumerate() {
        assert_eq!(run(&mut m, "flash_at", &[(i as u64, 2)], 4), u64::from(v as u32), "flash_at({i})");
    }
    assert_eq!(run(&mut m, "dispatch", &[(0, 2), (21, 2)], 2), 42);
    assert_eq!(run(&mut m, "dispatch", &[(1, 2), (21, 2)], 2), 63);
    assert_eq!(run(&mut m, "pick", &[(1, 1), (5, 2)], 2), 10);
    assert_eq!(run(&mut m, "pick", &[(0, 1), (5, 2)], 2), 15);
    assert_eq!(run(&mut m, "locals", &[(3, 2)], 2), 3 * 28);
    assert_eq!(run(&mut m, "dyn", &[(5, 2)], 2), 16);
    assert_eq!(run(&mut m, "dyn", &[(1, 2)], 2), 18);
    // flag: 0 -add 300-> 300 -max(-5)-> 300 -umax(0xfffb)-> 0xfffb
    //       -cmpxchg(-5 → 42) succeeds-> 42 -cmpxchg(-5 → 99) fails-> 42.
    // Old values: 0, 300, 300, 0xfffb, 42; bytes[0] = 1 (then 1-5); load 42.
    let want = (300u16 + 300).wrapping_add(0xfffb).wrapping_add(42).wrapping_add(1).wrapping_add(42);
    assert_eq!(run(&mut m, "atomics", &[], 2), u64::from(want));
    let flag = fw.symbol("flag").unwrap() as usize;
    assert_eq!(u16::from_le_bytes([m.data[flag], m.data[flag + 1]]), 42);
    let bytes = fw.symbol("bytes").unwrap() as usize;
    assert_eq!(m.data[bytes], 1u8.wrapping_sub(5));
}




// ===========================================================================
// Soft float (f32) vs host IEEE arithmetic
// ===========================================================================

const FLOATS: &str = r#"
module "floats"
func @fadd(f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32):
  %r = fadd %a, %b : f32
  ret %r
}
func @fsub(f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32):
  %r = fsub %a, %b : f32
  ret %r
}
func @fmul(f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32):
  %r = fmul %a, %b : f32
  ret %r
}
func @fdiv(f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32):
  %r = fdiv %a, %b : f32
  ret %r
}
func @fneg(f32) -> f32 {
entry ^0(%a: f32):
  %r = fneg %a : f32
  ret %r
}
func @to_i32(f32) -> i32 {
entry ^0(%a: f32):
  %r = fptosi %a : i32
  ret %r
}
func @to_u32(f32) -> i32 {
entry ^0(%a: f32):
  %r = fptoui %a : i32
  ret %r
}
func @to_i64(f32) -> i64 {
entry ^0(%a: f32):
  %r = fptosi %a : i64
  ret %r
}
func @to_i16(f32) -> i16 {
entry ^0(%a: f32):
  %r = fptosi %a : i16
  ret %r
}
func @from_i32(i32) -> f32 {
entry ^0(%a: i32):
  %r = sitofp %a : f32
  ret %r
}
func @from_u32(i32) -> f32 {
entry ^0(%a: i32):
  %r = uitofp %a : f32
  ret %r
}
func @from_i64(i64) -> f32 {
entry ^0(%a: i64):
  %r = sitofp %a : f32
  ret %r
}
func @from_u64(i64) -> f32 {
entry ^0(%a: i64):
  %r = uitofp %a : f32
  ret %r
}
func @from_i8(i8) -> f32 {
entry ^0(%a: i8):
  %r = sitofp %a : f32
  ret %r
}
func @poly(f32) -> f32 {
entry ^0(%x: f32):
  %x2 = fmul %x, %x : f32
  %a = fmul %x2, f32 0x40400000 : f32
  %b = fsub %a, %x : f32
  %c = fadd %b, f32 0x3f000000 : f32
  %lt = fcmp olt %c, f32 0x00000000 : i1
  %n = fneg %c : f32
  %r = select %lt, %n, %c : f32
  ret %r
}
"#;

const FPREDS: [&str; 14] = ["oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno"];

fn fpred_host(p: &str, a: f32, b: f32) -> bool {
    let uno = a.is_nan() || b.is_nan();
    match p {
        "oeq" => a == b,
        "ogt" => a > b,
        "oge" => a >= b,
        "olt" => a < b,
        "ole" => a <= b,
        "one" => !uno && a != b,
        "ord" => !uno,
        "ueq" => uno || a == b,
        "ugt" => uno || a > b,
        "uge" => uno || a >= b,
        "ult" => uno || a < b,
        "ule" => uno || a <= b,
        "une" => a != b,
        _ => uno,
    }
}

fn float_samples() -> Vec<u32> {
    let mut v: Vec<u32> = [
        0.0f32, -0.0, 1.0, -1.0, 0.5, 1.5, 3.0, -7.25, std::f32::consts::PI, 1e30, -1e30, f32::MAX, f32::MIN_POSITIVE,
        1e-40, -1e-42, 1e-45, 16_777_216.0, 16_777_217.0, 2_147_483_520.0, -2_147_483_648.0, 3e9, 65536.5, 0.1,
        f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
    ]
    .iter()
    .map(|f| f.to_bits())
    .collect();
    let mut x = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..14 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        v.push(x as u32);
    }
    v
}

fn same_float(got: u32, want: f32) -> bool {
    let g = f32::from_bits(got);
    (g.is_nan() && want.is_nan()) || got == want.to_bits()
}

#[test]
fn soft_float_matches_ieee() {
    let mut src = FLOATS.to_owned();
    for p in FPREDS {
        src.push_str(&format!(
            "func @cmp_{p}(f32, f32) -> i1 {{\nentry ^0(%a: f32, %b: f32):\n  %r = fcmp {p} %a, %b : i1\n  ret %r\n}}\n"
        ));
    }
    let fw = firmware(&src);
    let (mut m, stop) = boot(&fw, &DEV);
    let mut run = |name: &str, args: &[(u64, u64)], ret: u64| call_in(&mut m, &fw, &DEV, stop, name, args, ret);
    let vals = float_samples();
    let mut n = 0;
    for &a in &vals {
        let fa = f32::from_bits(a);
        for &b in &vals {
            let fb = f32::from_bits(b);
            let args = [(u64::from(a), 4), (u64::from(b), 4)];
            for (name, want) in [("fadd", fa + fb), ("fsub", fa - fb), ("fmul", fa * fb), ("fdiv", fa / fb)] {
                let got = run(name, &args, 4) as u32;
                assert!(same_float(got, want), "{name}({fa:e} [{a:#x}], {fb:e} [{b:#x}]) = {:e} [{got:#x}], want {want:e} [{:#x}]", f32::from_bits(got), want.to_bits());
                n += 1;
            }
            for p in FPREDS {
                let got = run(&format!("cmp_{p}"), &args, 1);
                assert_eq!(got, u64::from(fpred_host(p, fa, fb)), "fcmp {p} {fa:e}, {fb:e}");
                n += 1;
            }
        }
        let got = run("fneg", &[(u64::from(a), 4)], 4) as u32;
        assert_eq!(got, a ^ 0x8000_0000);
        let t = fa.trunc();
        if (-2_147_483_648.0..2_147_483_648.0).contains(&t) {
            assert_eq!(run("to_i32", &[(u64::from(a), 4)], 4) as u32, (fa as i32) as u32, "fptosi {fa:e}");
        }
        if (0.0..4_294_967_296.0).contains(&t) {
            assert_eq!(run("to_u32", &[(u64::from(a), 4)], 4) as u32, fa as u32, "fptoui {fa:e}");
        }
        if (-9.223_372e18..9.223_372e18).contains(&t) {
            assert_eq!(run("to_i64", &[(u64::from(a), 4)], 8), (fa as i64) as u64, "fptosi i64 {fa:e}");
        }
        if (-32768.0..32768.0).contains(&t) {
            assert_eq!(run("to_i16", &[(u64::from(a), 4)], 2) as u16, (fa as i16) as u16, "fptosi i16 {fa:e}");
        }
        n += 4;
    }
    let mut ints: Vec<u64> = vec![0, 1, 2, 3, 7, 255, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 16_777_217, 0x1234_5678, 0xffff_ff80];
    ints.extend(samples(64, 10));
    for &i in &ints {
        let i32v = i as u32;
        for (name, want, size) in [
            ("from_i32", (i32v as i32) as f32, 4),
            ("from_u32", i32v as f32, 4),
            ("from_i64", (i as i64) as f32, 8),
            ("from_u64", i as f32, 8),
            ("from_i8", f32::from(i as u8 as i8), 1),
        ] {
            let arg = if size == 8 { i } else if size == 4 { u64::from(i32v) } else { i & 0xff };
            let got = run(name, &[(arg, size)], 4) as u32;
            assert_eq!(got, want.to_bits(), "{name}({arg:#x}) = {:e}, want {want:e}", f32::from_bits(got));
            n += 1;
        }
    }
    for x in [0.0f32, 1.0, -2.5, 0.25, 100.0, 1e-3] {
        let want = {
            let c = x * x * 3.0 - x + 0.5;
            if c < 0.0 { -c } else { c }
        };
        let got = run("poly", &[(u64::from(x.to_bits()), 4)], 4) as u32;
        assert!(same_float(got, want), "poly({x})");
        n += 1;
    }
    eprintln!("avr soft float: {n} results checked");
}

// ===========================================================================
// Branch relaxation, stack usage, objects, firmware
// ===========================================================================

/// A function whose conditional branch skips a body of `n` operations of
/// type `t`: past 64 words the branch becomes `brXX` over `rjmp`, past 2 K
/// words over a relocated `jmp`.
fn long_body(t: &str, n: usize) -> String {
    let mut s = format!("func @long_{t}({t}, i1) -> {t} {{\nentry ^0(%x: {t}, %c: i1):\n  cond_br %c, ^1, ^2(%x)\n^1:\n  %v0 = add %x, {t} 1 : {t}\n");
    for k in 1..n {
        s.push_str(&format!("  %v{k} = xor %v{}, {t} {} : {t}\n", k - 1, (k * 37) % 251));
    }
    s.push_str(&format!("  br ^2(%v{})\n^2(%r: {t}):\n  ret %r\n}}\n", n - 1));
    s
}

#[test]
fn long_branches_are_relaxed() {
    let n16 = 120; // ~120 × 4 words: beyond `brXX`, within `rjmp`
    let n64 = 450; // ~450 × 8+ words: beyond `rjmp`
    let src = format!("module \"long\"\n{}{}", long_body("i16", n16), long_body("i64", n64));
    let fw = firmware(&src);
    let (mut m, stop) = boot(&fw, &DEV);
    let mut run = |name: &str, args: &[(u64, u64)], ret: u64| call_in(&mut m, &fw, &DEV, stop, name, args, ret);
    let host = |x: u64, n: usize, mask: u64| {
        let mut v = x.wrapping_add(1) & mask;
        for k in 1..n {
            v ^= ((k * 37) % 251) as u64;
        }
        v
    };
    for x in [0u64, 5, 0xffff] {
        assert_eq!(run("long_i16", &[(x, 2), (1, 1)], 2), host(x, n16, 0xffff));
        assert_eq!(run("long_i16", &[(x, 2), (0, 1)], 2), x);
    }
    for x in [0u64, 0x1234_5678_9abc_def0] {
        assert_eq!(run("long_i64", &[(x, 8), (1, 1)], 8), host(x, n64, u64::MAX));
        assert_eq!(run("long_i64", &[(x, 8), (0, 1)], 8), x);
    }
    // The i64 function is long enough to need a `jmp` relocated against
    // itself.
    let obj = compile_for(&src, &DEV);
    let me = obj.symbol_id("long_i64").unwrap();
    assert!(
        obj.relocations().iter().any(|r| r.symbol == me && r.kind == crate::mc::object::RelocKind::AvrCall),
        "a long branch uses a relocated jmp"
    );
    let size = obj.symbols().iter().find(|s| s.name == "long_i64").unwrap().size;
    assert!(size > 4096 + 256, "the body really is longer than rjmp's reach ({size} bytes)");
}

#[test]
fn stack_usage_bounds_the_real_stack() {
    // The report of the program and of the runtime it links, and the depth
    // actually reached running it from reset.
    let (m, syms) = parse(PROGRAM);
    let compiled = super::compile_module_for_device(&m, &syms, &CodegenOptions::default(), &DEV);
    let mut report = compiled.stack.clone();
    for member in super::runtime::compiled(&DEV) {
        report.extend(member.stack);
    }
    let bound = report.worst_case_depth("main", &crate::codegen::StackAssumptions::new()).expect("a static bound");
    let fw = super::link::build(vec![compiled.object], &DEV, "main").unwrap();
    let (_, mach) = run_main(&fw);
    let used = u64::from(DEV.ram_end - mach.min_sp);
    assert!(used <= bound.bytes, "used {used} bytes, bound {} ({:?})", bound.bytes, bound.path);
    assert!(used + 16 >= bound.bytes / 2, "the bound is not wildly loose: used {used}, bound {}", bound.bytes);
    // Every AVR frame counts the 2-byte return address, and nothing is probed.
    for u in report.functions() {
        assert_eq!(u.return_address, 2);
        assert!(!u.probed);
        assert_eq!(u.frame_size, 2 + u.saved_registers + u.sp_adjust + u.outgoing_args, "{}", u.name);
    }
    // Locals and stack arguments show up where expected.
    let (m, syms) = parse(CONTROL);
    let r = super::compile_module_for_device(&m, &syms, &CodegenOptions::default(), &DEV).stack;
    assert_eq!(r.get("call_many").unwrap().outgoing_args, 18);
    let (m, syms) = parse(MEMORY);
    let r = super::compile_module_for_device(&m, &syms, &CodegenOptions::default(), &DEV).stack;
    assert!(r.get("locals").unwrap().sp_adjust >= 16, "the 16-byte alloca is in the frame");
    assert!(r.get("dyn").unwrap().dynamic_alloca);
}

#[test]
fn elf_object_has_every_avr_relocation() {
    use crate::mc::object::RelocKind;
    let obj = compile_for(MEMORY, &DEV);
    let crt = startup::object(&DEV, "main");
    let kinds: Vec<RelocKind> = obj.relocations().iter().chain(crt.relocations()).map(|r| r.kind).collect();
    for k in [
        RelocKind::AvrCall,
        RelocKind::Avr13Pcrel,
        RelocKind::Avr16Pm,
        RelocKind::AvrLo8Ldi,
        RelocKind::AvrHi8Ldi,
        RelocKind::AvrLo8LdiPm,
        RelocKind::AvrHi8LdiPm,
    ] {
        assert!(kinds.contains(&k), "{k:?} is emitted");
    }
    // The flash table of function pointers is in .progmem.data with _PM
    // relocations; the SRAM data in .data.
    let pm = obj.sections().iter().position(|s| s.name == super::data::PROGMEM).expect(".progmem.data");
    assert!(obj.relocations().iter().any(|r| r.section.index() == pm && r.kind == RelocKind::Avr16Pm));
    if !have("llvm-readobj") {
        eprintln!("skipping the llvm-readobj check: no llvm-readobj");
        return;
    }
    let mut names = String::new();
    for o in [&obj, &crt] {
        let path = scratch_file("rel.o");
        std::fs::write(&path, super::write_elf(o).unwrap()).unwrap();
        let out = std::process::Command::new("llvm-readobj").args(["-h", "-r", "-s"]).arg(&path).output().unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        names.push_str(&String::from_utf8_lossy(&out.stdout));
    }
    assert!(names.contains("EM_AVR"), "{names}");
    assert!(names.contains("Class: 32-bit"));
    for n in ["R_AVR_CALL", "R_AVR_13_PCREL", "R_AVR_16_PM", "R_AVR_LO8_LDI ", "R_AVR_HI8_LDI ", "R_AVR_LO8_LDI_PM", "R_AVR_HI8_LDI_PM"] {
        assert!(names.contains(n), "llvm-readobj sees {n}:\n{names}");
    }
    assert!(names.contains(".progmem.data"));
    // And llvm-objdump decodes the relocated code, naming the callee.
    if let Some(lines) = objdump_object(&obj) {
        assert!(lines.iter().any(|l| l.contains("icall")), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("lpm")), "{lines:?}");
    }
}

/// Decode Intel HEX (checking every record's checksum) into a flash image.
fn decode_ihex(hex: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut end = false;
    for line in hex.lines() {
        let b: Vec<u8> = (1..line.len()).step_by(2).map(|i| u8::from_str_radix(&line[i..i + 2], 16).unwrap()).collect();
        assert!(line.starts_with(':'));
        assert_eq!(b.iter().fold(0u8, |s, x| s.wrapping_add(*x)), 0, "checksum of {line}");
        let (n, addr, ty) = (usize::from(b[0]), usize::from(u16::from_be_bytes([b[1], b[2]])), b[3]);
        match ty {
            0 => {
                if out.len() < addr + n {
                    out.resize(addr + n, 0xff);
                }
                out[addr..addr + n].copy_from_slice(&b[4..4 + n]);
            }
            1 => end = true,
            other => panic!("unexpected record type {other}"),
        }
    }
    assert!(end, "an end-of-file record");
    out
}

#[test]
fn intel_hex_image_boots() {
    let fw = firmware(PROGRAM);
    let hex = fw.to_ihex();
    let flash = decode_ihex(&hex);
    assert_eq!(flash, fw.flash);
    // The reset vector jumps to the startup code, and the image runs from
    // the decoded HEX alone.
    assert_eq!(u16::from_le_bytes([flash[0], flash[1]]) & 0xfe0e, 0x940c, "jmp at the reset vector");
    let stop = fw.symbol(startup::STOP).unwrap() / 2;
    let mut m = Avr::new(&flash, &DEV);
    m.run_until(stop, BUDGET).unwrap();
    assert_eq!(u16::from(m.reg(24)) | (u16::from(m.reg(25)) << 8), 110);
    // Unused vectors go to __bad_interrupt; a strong __vector_N takes over.
    let src = format!("{PROGRAM}\nfunc @__vector_3() -> void {{\nentry ^0:\n  ret\n}}\n");
    let fw = firmware(&src);
    let target = |v: usize| {
        let w = |i: usize| u16::from_le_bytes([fw.flash[4 * v + i], fw.flash[4 * v + i + 1]]);
        2 * (u32::from(w(2)) | (u32::from(w(0) & 1) << 16))
    };
    assert_eq!(target(3), fw.symbol("__vector_3").unwrap());
    assert_eq!(target(4), fw.symbol("__bad_interrupt").unwrap());
}

#[test]
fn link_errors_are_reported() {
    let obj = compile_for("module \"u\"\nfunc @missing(i16) -> i16\nfunc @main() -> i16 {\nentry ^0:\n  %r = call @missing(i16 1) : i16\n  ret %r\n}\n", &DEV);
    let err = super::link::build(vec![obj], &DEV, "main").unwrap_err();
    assert!(err.contains("undefined reference to `missing`"), "{err}");
    let big = format!("module \"big\"\nglobal @huge : [3000 x i8] = [3000 x i8] ({})\n", vec!["i8 1"; 3000].join(", "));
    let err = super::link::build(vec![compile_for(&big, &DEV), compile_for("module \"m\"\nfunc @main() -> i16 {\nentry ^0:\n  ret i16 0\n}\n", &DEV)], &DEV, "main").unwrap_err();
    assert!(err.contains("SRAM"), "{err}");
}

/// A module written without a data layout (LP64 by default, function
/// references in space 0, `i64` pointer offsets) still compiles for AVR.
#[test]
fn module_without_a_layout() {
    let src = r#"
module "plain"
global @vals : [3 x i16] = [3 x i16] (i16 4, i16 5, i16 6)
global @fp : ptr = ptr @square
func @square(i16) -> i16 {
entry ^0(%x: i16):
  %r = mul %x, %x : i16
  ret %r
}
func @main() -> i16 {
entry ^0:
  %p = ptr_add @vals, i64 4 : ptr
  %v = load %p align 2 : i16
  %f = load @fp align 8 : ptr
  %r = call %f(%v) : i16
  ret %r
}
"#;
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms).unwrap();
    assert_eq!(*m.data_layout(), crate::ir::DataLayout::lp64());
    let obj = super::compile_module_for_device(&m, &syms, &CodegenOptions::default(), &DEV).object;
    let fw = super::link::build(vec![obj], &DEV, "main").unwrap();
    assert_eq!(run_main(&fw).0, 36);
}

// ===========================================================================
// Vectors: scalarized, then soft float and integer legalization
// ===========================================================================

/// An `f32` vector program (the LF runtime has no `f64` helpers, so the
/// shared float fixture, which uses `<2 x f64>`, cannot link on AVR).
const F32_VECTORS: &str = r#"
module "vf32"
func @vf32(i64, i64, i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64):
  %p0 = insertelement <2 x i64> poison, %a, 0 : <2 x i64>
  %x0 = insertelement %p0, %b, 1 : <2 x i64>
  %p1 = insertelement <2 x i64> poison, %c, 0 : <2 x i64>
  %y0 = insertelement %p1, %d, 1 : <2 x i64>
  %i = bitcast %x0 : <4 x i32>
  %j = bitcast %y0 : <4 x i32>
  %small = and %i, <4 x i32> (i32 65535, i32 65535, i32 65535, i32 65535) : <4 x i32>
  %f = sitofp %small : <4 x f32>
  %g = sitofp %j : <4 x f32>
  %s = fadd %f, %g : <4 x f32>
  %u = fmul %s, <4 x f32> (f32 0x3fc00000, f32 0xc0000000, f32 0x3f000000, f32 0x41200000) : <4 x f32>
  %v = fdiv %u, <4 x f32> (f32 0x40400000, f32 0x3f800000, f32 0xc0800000, f32 0x3e800000) : <4 x f32>
  %lt = fcmp olt %v, %f : <4 x i1>
  %n = fneg %v : <4 x f32>
  %w = select %lt, %n, %v : <4 x f32>
  %back = fptosi %f : <4 x i32>
  %wi = bitcast %w : <2 x i64>
  %bi = bitcast %back : <2 x i64>
  %o = xor %wi, %bi : <2 x i64>
  %l = extractelement %o, 0 : i64
  %h = extractelement %o, 1 : i64
  %m = mul %h, i64 1000003 : i64
  %r = xor %l, %m : i64
  ret %r
}
"#;

/// Run vector test functions `(i64, i64, i64, i64) -> i64` on the AVR
/// interpreter (two of the four arguments travel on the stack) and compare
/// with the reference executor on the original vector IR.
fn check_vectors(what: &str, src: &str, cases: &[crate::target::vector_fixtures::Case]) -> usize {
    // Scalarized vector code is large: each function gets its own image, on
    // an AVR5 with 64 KiB of flash (the ATmega644 class).
    let dev = BIG;
    let want = crate::target::vector_fixtures::reference(src, cases);
    let (m, syms) = parse(src);
    let (pm, _, _) = super::prepare::prepare(&m, &syms, &dev).unwrap();
    assert!(!crate::codegen::legalize::uses_vectors(&pm), "{what}: every vector is scalarized");
    let head = &src[..src.find("func @").expect("a function")];
    let mut n = 0;
    let mut images: Vec<(String, Option<Firmware>)> = Vec::new();
    for ((name, args), w) in cases.iter().zip(&want) {
        let Some(w) = w else { continue };
        if !images.iter().any(|(k, _)| k == name) {
            let start = src.find(&format!("func @{name}(")).expect("the function");
            let end = src[start + 1..].find("\nfunc @").map_or(src.len(), |e| start + 2 + e);
            let body = &src[start..end];
            // The LF runtime has no f64 helpers.
            let fw = (!body.contains("f64")).then(|| firmware_for(&format!("{head}{body}"), &dev));
            images.push((name.clone(), fw));
        }
        let Some(fw) = &images.iter().find(|(k, _)| k == name).expect("built").1 else { continue };
        let (mut mach, stop) = boot(fw, &dev);
        let a: Vec<(u64, u64)> = args.iter().map(|&x| (x as u64, 8)).collect();
        let got = call_in(&mut mach, fw, &dev, stop, name, &a, 8);
        assert_eq!(got, *w, "{what}: @{name}{args:?}");
        n += 1;
    }
    n
}

#[test]
fn vector_programs_are_scalarized_and_run() {
    use crate::target::vector_fixtures as vf;
    let mut n = 0;
    for (what, src) in [
        ("int arith", vf::int_arith_src()),
        ("compares", vf::compare_src()),
        ("lanes", vf::lanes_src()),
        ("masks", vf::masks_src()),
        ("f32", F32_VECTORS.to_owned()),
    ] {
        let names: Vec<&str> =
            src.split("func @").skip(1).filter_map(|rest| rest.split_once('(').map(|(n, _)| n)).collect();
        n += check_vectors(what, &src, &vf::cases(&names, &vf::INPUTS));
    }
    let mut rng = vf::Rng(0xa7e);
    for p in 0..2u64 {
        // Integer vectors only: float lanes may be f64, which has no LF runtime.
        let (src, names) = vf::random_program(0xa000 + p, 3, 5, false);
        let mut cs = Vec::new();
        for name in &names {
            for _ in 0..2 {
                cs.push((name.clone(), vf::random_inputs(&mut rng)));
            }
        }
        n += check_vectors(&format!("random{p}"), &src, &cs);
    }
    eprintln!("avr vector programs: {n} results compared");
    assert!(n >= 100, "{n}");
}

/// A function with no secrets gets the compact lowerings: the skip-based
/// compare (`ldi; br<cond>; ldi`), the counted-loop shifter (`dec; brpl`)
/// and the `sbrc` sign fill — none of the branch-free forms (no `in r30,
/// SREG`, no `bst`/`bld`).
#[test]
fn public_code_gets_the_compact_forms() {
    let src = r#"
module "public"
func @f(i16, i16, i4) -> i16 {
entry ^0(%a: i16, %b: i16, %c: i4):
  %lt = icmp ult %a, %b : i1
  %k = and %b, i16 15 : i16
  %s = shl %a, %k : i16
  %x = sext %c : i16
  %z = zext %lt : i16
  %t = add %s, %z : i16
  %r = add %t, %x : i16
  ret %r
}
"#;
    let obj = compile_for(src, &DEV);
    let words: Vec<u16> = obj.sections()[0].bytes.chunks(2).map(|w| u16::from_le_bytes([w[0], w[1]])).collect();
    let has = |w: u16, mask: u16| words.iter().any(|&x| x & mask == w);
    assert!(has(0xf400 | u16::from(FLAG_N), 0xfc07), "a brpl closes the shift loop");
    assert!(has(0x940a, 0xfe0f), "a dec counts the shift loop");
    assert!(has(0xfc00, 0xfe08), "an sbrc fills the sign");
    assert!(!words.contains(&in_(30, SREG)), "no SREG read");
    assert!(!has(0xf800, 0xfc00) && !has(0xfa00, 0xfe00), "no bst/bld");
    // And it computes the same as the branch-free forms.
    let fw = firmware(src);
    for (a, b, c) in [(1u64, 2u64, 0xfu64), (0x8000, 3, 7), (5, 0xffff, 8)] {
        let s = (a << (b & 15)) & 0xffff;
        let z = u64::from(a < b);
        let x = if c & 8 != 0 { c | 0xfff0 } else { c };
        let want = (s + z + x) & 0xffff;
        assert_eq!(call(&fw, "f", &[(a, 2), (b, 2), (c, 1)], 2), want);
    }
}

