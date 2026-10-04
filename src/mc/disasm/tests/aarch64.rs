//! AArch64 decoder tests: round trips against the encoder, and llvm-objdump
//! differential tests.
//!
//! - **Golden texts**: hand-checked words and their exact disassembly.
//! - **Encoder round trip**: a fuzzed corpus built with LF's own A64
//!   instruction-word builders ([`crate::target::aarch64::encode`]) is
//!   decoded (every word must be known), then the decoded text is assembled
//!   again by our own assembler (`rsasm` through [`crate::mc::asm`]) and the
//!   words compared bit for bit. Where `rsasm` picks another encoding for the
//!   same text, the text fixpoint `decode(assemble(text)) == text` is
//!   required instead (counted separately).
//! - **llvm-objdump differential**: objects LF compiles (COFF, Mach-O and
//!   ELF) and a broad hand-written corpus assembled by `llvm-mc` are
//!   disassembled by both and compared instruction by instruction.

use std::process::Command;

use super::corpus::{FLOATS, INTS};
use super::{Rng, assert_clean, compile, differential, llvm_tool, object_file, scratch};
use crate::mc::disasm::aarch64::{decode_bit_masks, decode_word, fp_imm};
use crate::mc::disasm::{Options, decode, disassemble};
use crate::target::aarch64::encode as enc;
use crate::target::{ObjectFormat, TargetArch};

fn text(word: u32, addr: u64) -> String {
    decode(TargetArch::AArch64, &word.to_le_bytes(), addr, &Options::default()).text().replace('\t', " ")
}

#[test]
fn golden() {
    let cases: &[(u32, &str)] = &[
        (0xa9bf7bfd, "stp x29, x30, [sp, #-0x10]!"),
        (0x910003fd, "mov x29, sp"),
        (0xa8c17bfd, "ldp x29, x30, [sp], #0x10"),
        (0xd65f03c0, "ret"),
        (0xaa0003e3, "mov x3, x0"),
        (0x9b007c62, "mul x2, x3, x0"),
        (0x9343fc64, "asr x4, x3, #3"),
        (0xd3407c22, "ubfx x2, x1, #0, #32"),
        (0x1e600841, "fmul d1, d2, d0"),
        (0x9e780040, "fcvtzs x0, d2"),
        (0x9e620001, "scvtf d1, x0"),
        (0x4ea01c02, "mov v2.16b, v0.16b"),
        (0xd2824680, "mov x0, #0x1234"),
        (0xf2a24680, "movk x0, #0x1234, lsl #16"),
        (0x92800020, "mov x0, #-0x2"),
        (0x1a9fa7e0, "cset w0, lt"),
        (0xd53bd040, "mrs x0, TPIDR_EL0"),
        (0xd5033bbf, "dmb ish"),
        (0xd503201f, "nop"),
        (0xf9400420, "ldr x0, [x1, #0x8]"),
        (0xb8616820, "ldr w0, [x1, x1]"),
        (0x885f7c20, "ldxr w0, [x1]"),
        (0xc8027c20, "stxr w2, x0, [x1]"),
        (0x1e6e1000, "fmov d0, #1.00000000"),
        (0x6f00e400, "movi v0.2d, #0000000000000000"),
        (0x00000000, "udf #0x0"),
    ];
    for &(w, want) in cases {
        assert_eq!(text(w, 0), want, "{w:#010x}");
    }
    // Branch targets are absolute and recorded.
    let b = decode(TargetArch::AArch64, &0x9400_0010u32.to_le_bytes(), 0x1000, &Options::default());
    assert_eq!((b.text(), b.target), ("bl\t0x1040".to_owned(), Some(0x1040)));
    let adrp = decode(TargetArch::AArch64, &0x9000_0020u32.to_le_bytes(), 0x1234, &Options::default());
    // immhi = 1, immlo = 0: four pages past the instruction's page.
    assert_eq!(adrp.target, Some(0x5000));
    // An unknown word is data; a short tail is bytes.
    assert_eq!(text(0xffff_ffff, 0), ".word 0xffffffff");
    assert_eq!(text(0, 0), "udf #0x0");
    let short = decode(TargetArch::AArch64, &[1, 2], 0, &Options::default());
    assert!(!short.known && short.len == 2);
}

#[test]
fn bit_masks_and_fp_immediates() {
    // DecodeBitMasks: 0xff, alternating bits, the 32-bit rotate.
    assert_eq!(decode_bit_masks(1, 0b000111, 0, 64), Some(0xff));
    assert_eq!(decode_bit_masks(0, 0b111100, 0, 64), Some(0x5555_5555_5555_5555));
    assert_eq!(decode_bit_masks(0, 0b011110, 1, 32), Some(0xbfff_ffff));
    assert_eq!(decode_bit_masks(0, 0b111111, 0, 64), None);
    assert_eq!(fp_imm(0x70), 1.0);
    assert_eq!(fp_imm(0x84), -2.5);
    assert_eq!(fp_imm(0x00), 2.0);
    assert_eq!(fp_imm(0x40), 0.125);
    assert_eq!(fp_imm(0xc0), -0.125);
}

/// Assemble `src` with `llvm-mc` into an AArch64 ELF object (`None` when it
/// is not installed).
fn llvm_mc(src: &str) -> Option<Vec<u8>> {
    let mc = llvm_tool("llvm-mc")?;
    let dir = scratch("a64-mc");
    let (s, o) = (dir.join("in.s"), dir.join("out.o"));
    std::fs::write(&s, src).unwrap();
    let out = Command::new(mc)
        .args(["--triple=aarch64", "-mattr=+v8.5a,+lse,+crc,+fullfp16,+rcpc,+rcpc-immo", "-filetype=obj", "-o"])
        .arg(&o)
        .arg(&s)
        .output()
        .expect("run llvm-mc");
    assert!(out.status.success(), "llvm-mc: {}", String::from_utf8_lossy(&out.stderr));
    let bytes = std::fs::read(&o).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(bytes)
}

/// llvm-objdump's Apple (Mach-O) spellings mapped onto the generic ones,
/// after normalization: its `;` comment marker (`mov x0, #0xff ; =255`) is
/// dropped, and a vector instruction's arrangement suffix (`mov.16b v2,
/// v0`) moves onto its register operands (`mov v2.16b, v0.16b`). Nothing
/// else is mapped.
fn apple_vector_syntax(s: String) -> String {
    let s = match s.find(';') {
        Some(i) => s[..i].trim_end().to_owned(),
        None => s,
    };
    let Some((mnem, ops)) = s.split_once(' ') else { return s };
    let Some((base, arr)) = mnem.split_once('.') else { return s };
    let letter = arr.chars().last().unwrap_or(' ');
    if base == "b" || !"bhsd".contains(letter) || !arr[..arr.len() - 1].bytes().all(|c| c.is_ascii_digit()) {
        return s;
    }
    let is_vreg = |o: &str| o.len() > 1 && o.starts_with('v') && o[1..].bytes().all(|c| c.is_ascii_digit());
    let ops: Vec<String> = ops
        .split(',')
        .map(|o| match o.split_once('[') {
            // `v1[2]` → `v1.s[2]`.
            Some((reg, idx)) if is_vreg(reg) => format!("{reg}.{letter}[{idx}"),
            // `v2` → `v2.16b`.
            None if is_vreg(o) && arr.len() > 1 => format!("{o}.{arr}"),
            _ => o.to_owned(),
        })
        .collect();
    format!("{base} {}", ops.join(","))
}

/// A broad hand-written corpus (every class the decoder covers, with the
/// aliases and edge cases of each), assembled by llvm-mc.
#[test]
fn llvm_mc_corpus_matches_llvm_objdump() {
    let src = include_str!("aarch64/corpus.s");
    let Some(obj) = llvm_mc(src) else {
        eprintln!("skipping: no llvm-mc");
        return;
    };
    let Some(report) = differential(TargetArch::AArch64, &obj, &[], &Options::default(), &|s| s) else {
        eprintln!("skipping: no llvm-objdump");
        return;
    };
    assert_clean("aarch64 llvm-mc corpus", &report);
    assert!(report.compared >= src.lines().filter(|l| !l.trim().is_empty()).count() - 2);
}

/// Every word LF's AArch64 encoder emits for the corpus programs, in COFF,
/// Mach-O and ELF objects, disassembles as llvm-objdump does.
#[test]
fn compiled_objects_match_llvm_objdump() {
    for (name, src) in [("ints", INTS), ("floats", FLOATS), ("vectors", VECTORS), ("atomics", ATOMICS)] {
        let obj = compile(TargetArch::AArch64, src);
        for format in [ObjectFormat::Coff, ObjectFormat::MachO, ObjectFormat::Elf] {
            let file = object_file(TargetArch::AArch64, &obj, format);
            let Some(report) = differential(TargetArch::AArch64, &file, &[], &Options::default(), &apple_vector_syntax) else {
                eprintln!("skipping: no llvm-objdump");
                return;
            };
            assert_clean(&format!("aarch64 {name} {format:?}"), &report);
        }
    }
}

/// Vector code (NEON).
const VECTORS: &str = r#"
module "vectors"

func @vadd(<4 x i32>, <4 x i32>) -> <4 x i32> {
entry ^0(%a: <4 x i32>, %b: <4 x i32>):
  %s = add %a, %b : <4 x i32>
  %m = mul %s, %b : <4 x i32>
  %x = xor %m, %a : <4 x i32>
  ret %x
}

func @vfloat(<2 x f64>, <2 x f64>) -> <2 x f64> {
entry ^0(%a: <2 x f64>, %b: <2 x f64>):
  %s = fadd %a, %b : <2 x f64>
  %m = fmul %s, %b : <2 x f64>
  %d = fdiv %m, %a : <2 x f64>
  ret %d
}

func @vlane(<4 x i32>, i32) -> i32 {
entry ^0(%a: <4 x i32>, %x: i32):
  %v = insertelement %a, %x, 2 : <4 x i32>
  %e = extractelement %v, 1 : i32
  %f = extractelement %v, 2 : i32
  %r = add %e, %f : i32
  ret %r
}
"#;

/// Atomics (exclusive loops, acquire/release, barriers).
const ATOMICS: &str = r#"
module "atomics"

global @w32 : i32 = i32 100
global @w8 : i8 = i8 -3
global @w64 : i64 = i64 -5

func @rmw(i32, i64, i8) -> i64 {
entry ^0(%v: i32, %d: i64, %b: i8):
  %a = atomic_rmw add seq_cst @w32, %v align 4 : i32
  %c = atomic_rmw max acq_rel @w8, %b align 1 : i8
  %e = atomic_rmw xchg relaxed @w64, %d align 8 : i64
  %f = atomic_rmw umin seq_cst @w64, %d align 8 : i64
  %g = atomic_rmw nand seq_cst @w32, %v align 4 : i32
  %ax = zext %a : i64
  %r = add %ax, %e : i64
  %r2 = add %r, %f : i64
  ret %r2
}
"#;

// ===========================================================================
// Encoder round trip
// ===========================================================================

/// The `.text` words of an ELF object.
fn text_words(elf: &[u8]) -> Result<Vec<u32>, String> {
    let bin = crate::mc::disasm::objfile::read(elf)?;
    let text = bin.sections.iter().find(|s| s.name == ".text").ok_or("no .text")?;
    Ok(text.bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Assemble one line with our own assembler (rsasm): its word, or `None`
/// when rsasm does not take that instruction.
fn rsasm_word(line: &str) -> Option<u32> {
    use crate::mc::asm::{AsmOptions, AsmSource, assemble};
    let src = format!(".text\n{line}\n");
    let elf = assemble(&[AsmSource { name: "rt.s", text: &src }], &AsmOptions::new(TargetArch::AArch64)).ok()?;
    match text_words(&elf).ok()?.as_slice() {
        [w] => Some(*w),
        _ => None,
    }
}

/// A fuzzed corpus of words from LF's encoder helpers.
fn encoder_corpus(rng: &mut Rng) -> Vec<u32> {
    let mut out = Vec::new();
    let reg = |rng: &mut Rng| rng.below(31) as u32; // never 31 (sp/zr ambiguity is covered by golden tests)
    for _ in 0..40 {
        let (d, n, m, a) = (reg(rng), reg(rng), reg(rng), reg(rng));
        let sf = rng.below(2) as u32;
        let width = if sf == 1 { 64 } else { 32 };
        let cond = rng.below(14) as u32;
        let ptype = rng.below(2) as u32;
        let size = rng.below(4) as u32;
        out.extend([
            enc::add_reg(sf, d, n, m),
            enc::sub_reg(sf, d, n, m),
            enc::and_reg(sf, d, n, m),
            enc::orr_reg(sf, d, n, m),
            enc::eor_reg(sf, d, n, m),
            enc::subs_reg(sf, d, n, m),
            enc::subs_reg(sf, 31, n, m),
            enc::mov_reg(sf, d, m),
            enc::add_imm(sf, d, n, rng.below(4096) as u32),
            enc::sub_imm(sf, d, n, rng.below(4096) as u32),
            enc::add_imm(1, 31, 31, rng.below(4096) as u32),
            enc::movz(sf, d, rng.below(65536) as u32, rng.below(if sf == 1 { 4 } else { 2 }) as u32),
            enc::movk(sf, d, rng.below(65536) as u32, rng.below(if sf == 1 { 4 } else { 2 }) as u32),
            enc::movn(sf, d, rng.below(65536) as u32, rng.below(if sf == 1 { 4 } else { 2 }) as u32),
            enc::udiv(sf, d, n, m),
            enc::sdiv(sf, d, n, m),
            enc::lslv(sf, d, n, m),
            enc::lsrv(sf, d, n, m),
            enc::asrv(sf, d, n, m),
            enc::madd(sf, d, n, m, a),
            enc::msub(sf, d, n, m, a),
            enc::madd(sf, d, n, m, 31),
            enc::sbfx0(sf, d, n, [8, 16, 32][rng.below(if sf == 1 { 3 } else { 2 }) as usize]),
            enc::ubfx0(sf, d, n, 1 + rng.below(width as u64 - 1) as u32),
            enc::lsl_imm(sf, d, n, 1 + rng.below(width as u64 - 1) as u32),
            enc::lsr_imm(sf, d, n, 1 + rng.below(width as u64 - 1) as u32),
            enc::asr_imm(sf, d, n, 1 + rng.below(width as u64 - 1) as u32),
            enc::ldst_uimm(rng.below(2) == 1, size, d, n, rng.below(4096) as u32),
            enc::ldxr(size, rng.below(2) == 1, d, n),
            // The status register must differ from the data and base (else
            // the encoding is CONSTRAINED UNPREDICTABLE).
            enc::stxr(size, rng.below(2) == 1, if a == d || a == n { (d.max(n) + 1) % 31 } else { a }, d, n),
            enc::ldar(size, d, n),
            enc::stlr(size, d, n),
            enc::dmb([9, 11, 15][rng.below(3) as usize]),
            enc::subs_ext(0, 31, n, m, [0, 1, 4, 5][rng.below(4) as usize]),
            enc::orn_reg(sf, d, 31, m),
            enc::orn_reg(sf, d, n, m),
            enc::csel(sf, d, n, m, cond),
            enc::cset(sf, d, cond),
            enc::b_uncond(rng.below(1 << 20) as i32 - (1 << 19)),
            enc::bl(rng.below(1 << 20) as i32 - (1 << 19)),
            enc::b_cond(cond, rng.below(1 << 18) as i32 - (1 << 17)),
            enc::cbz(sf, d, rng.below(1 << 18) as i32 - (1 << 17), rng.below(2) == 1),
            enc::blr(n),
            enc::ret(30),
            enc::ret(n),
            enc::svc(rng.below(65536) as u32),
            enc::brk(rng.below(65536) as u32),
            enc::adrp(d),
            enc::stp_pre(d, n, 31, rng.below(128) as i32 - 64),
            enc::ldp_post(d, n, 31, rng.below(128) as i32 - 64),
            enc::fadd(ptype, d, n, m),
            enc::fsub(ptype, d, n, m),
            enc::fmul(ptype, d, n, m),
            enc::fdiv(ptype, d, n, m),
            enc::fmov_reg(ptype, d, n),
            enc::fneg(ptype, d, n),
            enc::fcvt(ptype, 1 - ptype, d, n),
            enc::fcmp(ptype, n, m),
            enc::fcvtzs(sf, ptype, d, n),
            enc::fcvtzu(sf, ptype, d, n),
            enc::scvtf(sf, ptype, d, n),
            enc::ucvtf(sf, ptype, d, n),
            enc::fmov_from_gpr(ptype, ptype, d, n),
            enc::fp_ldst_uimm(rng.below(2) == 1, 2 + ptype, d, n, rng.below(4096) as u32),
            enc::q_ldst_uimm(rng.below(2) == 1, d, n, rng.below(4096) as u32),
            enc::simd_mov(d, n),
        ]);
        let esize = [8, 16, 32, 64][rng.below(4) as usize];
        let lane = rng.below(u64::from(128 / esize)) as u32;
        out.extend([
            enc::neon_dup(esize, d, n),
            enc::neon_dup_lane(esize, lane, d, n),
            enc::neon_umov(esize, lane, d, n),
            enc::neon_ins_gpr(esize, lane, d, n),
            enc::neon_ins_elem(esize, lane, d, n),
            enc::neon_shift(crate::target::aarch64::isel::neon::NeonOp::Shl, esize, d, n, rng.below(u64::from(esize)) as u32),
            enc::neon_shift(crate::target::aarch64::isel::neon::NeonOp::Ushr, esize, d, n, 1 + rng.below(u64::from(esize)) as u32),
            enc::neon_shift(crate::target::aarch64::isel::neon::NeonOp::Sshr, esize, d, n, 1 + rng.below(u64::from(esize)) as u32),
        ]);
        {
            use crate::target::aarch64::isel::neon::NeonOp as N;
            let int3 = [N::Add, N::Sub, N::And, N::Bic, N::Orr, N::Eor, N::Cmeq, N::Cmgt, N::Cmge, N::Cmhi, N::Cmhs, N::Sshl, N::Ushl, N::Sqadd, N::Uqadd, N::Sqsub, N::Uqsub];
            let no64 = [N::Mul, N::Smax, N::Smin, N::Umax, N::Umin];
            let float3 = [N::Fadd, N::Fsub, N::Fmul, N::Fdiv, N::Fcmeq, N::Fcmge, N::Fcmgt];
            let fesize = [32, 64][rng.below(2) as usize];
            let small = [8, 16, 32][rng.below(3) as usize];
            out.push(enc::neon3(int3[rng.below(int3.len() as u64) as usize], esize, d, n, m));
            out.push(enc::neon3(no64[rng.below(no64.len() as u64) as usize], small, d, n, m));
            out.push(enc::neon3(float3[rng.below(float3.len() as u64) as usize], fesize, d, n, m));
            out.push(enc::neon3(N::Tbl, 8, d, n, m));
            out.push(enc::neon2([N::Neg, N::Not][rng.below(2) as usize], esize, d, n));
            out.push(enc::neon2([N::Fneg, N::Scvtf, N::Ucvtf, N::Fcvtzs, N::Fcvtzu][rng.below(5) as usize], fesize, d, n));
            out.push(enc::neon2([N::Addv, N::Smaxv, N::Sminv, N::Umaxv, N::Uminv][rng.below(5) as usize], small, d, n));
            out.push(enc::neon2(N::Addp, 64, d, n));
        }
    }
    out
}

#[test]
fn encoder_round_trip() {
    let mut rng = Rng(0xa64);
    let corpus = encoder_corpus(&mut rng);
    let mut lines = Vec::new();
    for &w in &corpus {
        // Decoded at address 0, a PC-relative target is the offset from the
        // instruction, which the assembler takes back as `.+off`.
        let inst = decode(TargetArch::AArch64, &w.to_le_bytes(), 0, &Options::default());
        assert!(inst.known && decode_word(w, 0).is_some(), "{w:#010x} did not decode: {}", inst.text());
        let mut t = inst.clone();
        if let (Some(k), Some(target)) = (inst.target_operand, inst.target) {
            t.operands[k] = format!(".{:+}", target as i64);
        }
        lines.push(t.text().replace('\t', " "));
    }
    // Re-assemble with rsasm; what it does not take goes to llvm-mc.
    let (mut identical, mut fixpoint, mut by_llvm) = (0, 0, 0);
    let mut leftovers: Vec<(u32, String)> = Vec::new();
    let check = |ours: u32, theirs: u32, line: &str, identical: &mut usize, fixpoint: &mut usize| {
        if ours == theirs {
            *identical += 1;
        } else {
            // Another encoding of the same text: the text must be a fixpoint.
            assert_eq!(text(ours, 0), text(theirs, 0), "`{line}`: {ours:#010x} re-assembled as {theirs:#010x}");
            *fixpoint += 1;
        }
    };
    for (&w, line) in corpus.iter().zip(&lines) {
        match rsasm_word(line) {
            Some(theirs) => check(w, theirs, line, &mut identical, &mut fixpoint),
            None => leftovers.push((w, line.clone())),
        }
    }
    let rsasm_count = identical + fixpoint;
    if !leftovers.is_empty() {
        let src: String = leftovers.iter().map(|(_, l)| format!("{l}\n")).collect();
        if let Some(obj) = llvm_mc(&src) {
            let words = text_words(&obj).expect("llvm-mc object");
            assert_eq!(words.len(), leftovers.len());
            for ((w, line), theirs) in leftovers.iter().zip(words) {
                check(*w, theirs, line, &mut identical, &mut fixpoint);
                by_llvm += 1;
            }
        } else {
            eprintln!("{} words not re-assembled: rsasm does not take them and llvm-mc is absent", leftovers.len());
        }
    }
    eprintln!(
        "aarch64 encoder round trip: {} words ({rsasm_count} via rsasm, {by_llvm} via llvm-mc): {identical} identical, {fixpoint} text fixpoints",
        corpus.len()
    );
    assert!(identical * 10 >= (identical + fixpoint) * 9, "most words must re-assemble identically");
}

/// Straight-line disassembly of a compiled function decodes every word.
#[test]
fn compiled_code_is_fully_known() {
    for src in [INTS, FLOATS, VECTORS, ATOMICS] {
        let obj = compile(TargetArch::AArch64, src);
        let text = obj.sections().iter().find(|s| s.name == ".text").unwrap();
        for (addr, inst) in disassemble(TargetArch::AArch64, &text.bytes, 0, &Options::default()) {
            assert!(inst.known, "{addr:#x}: {}", inst.text());
        }
    }
}

/// Random words, disassembled by both: wherever we decode an instruction,
/// llvm-objdump must print the same text (an encoding we do not know is
/// data, which is never wrong).
#[test]
fn random_words_match_llvm_objdump() {
    use crate::mc::object::{ObjectModule, Section, SectionKind, Symbol, SymbolBinding, SymbolType};
    let mut rng = Rng(0x05ee_da64);
    let mut obj = ObjectModule::new("random");
    let s = obj.add_section(Section::new(".text", SectionKind::Text, 4));
    let n = 40_000;
    let mut bytes = Vec::with_capacity(4 * n);
    for _ in 0..n {
        bytes.extend_from_slice(&(rng.next() as u32).to_le_bytes());
    }
    obj.section_mut(s).bytes = bytes;
    obj.add_symbol(Symbol::defined("f", SymbolBinding::Global, SymbolType::Func, s, 0, 0));
    let file = object_file(TargetArch::AArch64, &obj, ObjectFormat::Elf);
    let Some(theirs) = super::objdump(&file, &[]) else {
        eprintln!("skipping: no llvm-objdump");
        return;
    };
    let theirs = &theirs[".text"];
    let (mut ours_known, mut theirs_known, mut agree) = (0, 0, 0);
    let mut bad = Vec::new();
    let mut missing: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for (addr, t) in theirs {
        let off = *addr as usize;
        let w = &obj.sections()[0].bytes[off..off + 4];
        let inst = decode(TargetArch::AArch64, w, *addr, &Options::default());
        theirs_known += usize::from(!t.contains("<unknown>"));
        if !inst.known {
            if !t.contains("<unknown>") {
                let m = t.split_whitespace().next().unwrap_or("").to_owned();
                *missing.entry(m).or_insert(0usize) += 1;
            }
            continue;
        }
        ours_known += 1;
        let a = super::normalize(TargetArch::AArch64, &inst.text());
        let b = super::normalize(TargetArch::AArch64, t);
        // LLVM names hundreds of system registers; an unnamed one prints in
        // the architectural `S<op0>_<op1>_C<n>_C<m>_<op2>` form, which is as
        // correct: compare everything else.
        let generic_sysreg = |x: &str| x.split(',').any(|o| o.starts_with("s2_") || o.starts_with("s3_"));
        let sysreg_only = (a.starts_with("mrs ") || a.starts_with("msr ")) && generic_sysreg(&a) && {
            let strip = |x: &str| x.split(',').filter(|o| !o.starts_with('s') || o.starts_with("sp")).map(str::to_owned).collect::<Vec<_>>();
            strip(&a.replacen("mrs ", "", 1).replacen("msr ", "", 1)) == strip(&b.replacen("mrs ", "", 1).replacen("msr ", "", 1))
        };
        if a == b || sysreg_only {
            agree += 1;
        } else {
            bad.push(format!("{:#010x}: ours `{}` | llvm `{t}`", u32::from_le_bytes([w[0], w[1], w[2], w[3]]), inst.text()));
        }
    }
    eprintln!("aarch64 random words: {n} words, llvm decodes {theirs_known}, we decode {ours_known}, {agree} agree, {} differ", bad.len());
    let mut top: Vec<(usize, String)> = missing.into_iter().map(|(m, c)| (c, m)).collect();
    top.sort_unstable_by(|a, b| b.cmp(a));
    eprintln!("  most frequent encodings llvm decodes and we do not: {:?}", &top[..top.len().min(40)]);
    assert!(bad.is_empty(), "{} differences:\n{}", bad.len(), bad.join("\n"));
}
