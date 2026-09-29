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
        add1(dec(d), format!("dec r{d}"));
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
            add1(sbrc(d, b), format!("sbrc r{d}, {b}"));
        }
    }
    for (d, k) in [(16u8, 0u8), (17, 0xff), (30, 0x5a), (31, 0xa5), (28, 1), (26, 0x80)] {
        add1(ldi(d, k), format!("ldi r{d}, {k}"));
        add1(cpi(d, k), format!("cpi r{d}, {k}"));
        add1(subi(d, k), format!("subi r{d}, {k}"));
        add1(sbci(d, k), format!("sbci r{d}, {k}"));
        add1(andi(d, k), format!("andi r{d}, {k}"));
        add1(ori(d, k), format!("ori r{d}, {k}"));
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
            (FLAG_N, false, "brpl"),
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
