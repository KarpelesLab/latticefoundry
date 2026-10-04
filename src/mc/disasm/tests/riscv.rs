//! RISC-V decoder tests: round trips against the encoder, and llvm-objdump
//! differential tests.
//!
//! - **Golden texts**: exact output for hand-checked encodings (base ISA,
//!   pseudoinstructions, M, A, F, D, Zicsr, C, Zba/Zbb).
//! - **Encoder round trip**: a fuzzed corpus built with the backend's own
//!   instruction builders (`target::riscv::encode`) decodes to exactly the
//!   instruction (mnemonic and operands) the builder was asked for, and the
//!   decoded text re-assembles with our assembler (`rsasm`) to the same
//!   bytes — or, where `rsasm` lacks a pseudoinstruction or picks another
//!   encoding, to bytes that decode to the same text.
//! - **llvm-objdump differentials** (skipped without LLVM): every 16-bit
//!   compressed encoding; random 32-bit words over every major opcode; a
//!   hand-written corpus assembled by `llvm-mc`; and objects LF compiles.

use super::corpus::{FLOATS, INTS};
use super::{Rng, assert_clean, compile, differential, llvm_tool, normalize, object_file, objdump, scratch};
use crate::mc::disasm::riscv::{Operand, RvInst, decode_inst};
use crate::mc::disasm::{Options, decode, disassemble};
use crate::mc::object::{ObjectModule, Section, SectionKind};
use crate::target::riscv::encode as enc;
use crate::target::{ObjectFormat, TargetArch};

/// The features llvm-objdump must enable to decode everything we do.
const MATTR: &str = "--mattr=+m,+a,+f,+d,+c,+zba,+zbb";

fn text_at(bytes: &[u8], addr: u64) -> String {
    let i = decode(TargetArch::Riscv64, bytes, addr, &Options::default());
    assert!(i.known, "{bytes:02x?} did not decode");
    i.text()
}

fn word(w: u32) -> String {
    text_at(&w.to_le_bytes(), 0)
}

fn half(h: u16) -> String {
    text_at(&h.to_le_bytes(), 0)
}

#[test]
fn golden_texts() {
    let cases: &[(u32, &str)] = &[
        (enc::add(10, 11, 12), "add\ta0, a1, a2"),
        (enc::addi(10, 11, -3), "addi\ta0, a1, -0x3"),
        (enc::addi(10, 0, 5), "li\ta0, 0x5"),
        (enc::addi(0, 0, 0), "nop"),
        (enc::mv(10, 11), "mv\ta0, a1"),
        (enc::addiw(10, 11, 0), "sext.w\ta0, a1"),
        (enc::xori(10, 11, -1), "not\ta0, a1"),
        (enc::sub(10, 0, 11), "neg\ta0, a1"),
        (enc::sltiu(10, 11, 1), "seqz\ta0, a1"),
        (enc::sltu(10, 0, 11), "snez\ta0, a1"),
        (enc::andi(10, 11, 255), "zext.b\ta0, a1"),
        (enc::slli(10, 11, 63), "slli\ta0, a1, 0x3f"),
        (enc::srai(10, 11, 3), "srai\ta0, a1, 0x3"),
        (enc::load(8, 10, 2, -8), "ld\ta0, -0x8(sp)"),
        (enc::load(1, 10, 11, 0), "lbu\ta0, 0x0(a1)"),
        (enc::store(4, 10, 11, 2047), "sw\ta0, 0x7ff(a1)"),
        (enc::lui(10, 0xfffff), "lui\ta0, 0xfffff"),
        (enc::auipc(10, 1), "auipc\ta0, 0x1"),
        (enc::ret(), "ret"),
        (enc::jalr(0, 10, 0), "jr\ta0"),
        (enc::jalr(1, 10, 0), "jalr\ta0"),
        (enc::jalr(1, 10, 4), "jalr\t0x4(a0)"),
        (enc::jalr(10, 11, 8), "jalr\ta0, 0x8(a1)"),
        (enc::ecall(), "ecall"),
        (enc::ebreak(), "ebreak"),
        (enc::mul(10, 11, 12), "mul\ta0, a1, a2"),
        (enc::remu(10, 11, 12), "remu\ta0, a1, a2"),
        (enc::lr(4, true, false, 10, 11), "lr.w.aq\ta0, (a1)"),
        (enc::sc(8, false, true, 10, 12, 11), "sc.d.rl\ta0, a2, (a1)"),
        (enc::amo(0, true, true, 10, 12, 11, 8), "amoadd.d.aqrl\ta0, a2, (a1)"),
        (enc::fence(0, 3, 3), "fence\trw, rw"),
        (enc::fence(0, 15, 15), "fence"),
        (enc::fence(8, 3, 3), "fence.tso"),
        (0x0000_100f, "fence.i"),
        (0x02c5_f553, "fadd.d\tfa0, fa1, fa2"),
        (0x02c5_8553, "fadd.d\tfa0, fa1, fa2, rne"),
        (0x6ac5_f543, "fmadd.d\tfa0, fa1, fa2, fa3"),
        (0x68c5_b54b, "fnmsub.s\tfa0, fa1, fa2, fa3, rup"),
        (0xc225_1553, "fcvt.l.d\ta0, fa0, rtz"),
        (0xd205_0553, "fcvt.d.w\tfa0, a0"),
        (0xd205_7553, "fcvt.d.w\tfa0, a0, dyn"),
        (0x4015_8553, "fcvt.s.d\tfa0, fa1, rne"),
        (0x22b5_8553, "fmv.d\tfa0, fa1"),
        (0x22b5_9553, "fneg.d\tfa0, fa1"),
        (0x20b5_a553, "fabs.s\tfa0, fa1"),
        (0xe205_0553, "fmv.x.d\ta0, fa0"),
        (0xf005_0553, "fmv.w.x\tfa0, a0"),
        (0xa2b5_2553, "feq.d\ta0, fa0, fa1"),
        (0xe205_1553, "fclass.d\ta0, fa0"),
        (0x00a1_2427, "fsw\tfa0, 0x8(sp)"),
        (0x0010_2573, "frflags\ta0"),
        (0x0025_1073, "fsrm\ta0"),
        (0x0025_9573, "fsrm\ta0, a1"),
        (0x0020_d073, "fsrmi\t0x1"),
        (0xc000_2573, "rdcycle\ta0"),
        (0x3000_2573, "csrr\ta0, mstatus"),
        (0x3005_1073, "csrw\tmstatus, a0"),
        (0x3004_6073, "csrsi\tmstatus, 0x8"),
        (0x7c00_2573, "csrr\ta0, 0x7c0"),
        (0x3005_a573, "csrrs\ta0, mstatus, a1"),
        (0x1050_0073, "wfi"),
        (0x1205_0073, "sfence.vma\ta0"),
        (0x0805_853b, "zext.w\ta0, a1"),
        (0x20c5_a533, "sh1add\ta0, a1, a2"),
        (0x6005_9513, "clz\ta0, a1"),
        (0x6b85_d513, "rev8\ta0, a1"),
        (0x0805_c53b, "zext.h\ta0, a1"),
    ];
    for &(w, want) in cases {
        assert_eq!(word(w), want, "{w:#010x}");
    }
    // Branches and jumps print absolute targets.
    assert_eq!(text_at(&enc::beq(10, 11, 8).to_le_bytes(), 0x100), "beq\ta0, a1, 0x108");
    assert_eq!(text_at(&enc::bne(10, 0, -8).to_le_bytes(), 0x100), "bnez\ta0, 0xf8");
    assert_eq!(text_at(&enc::bcmp(5, 0, 10, 4).to_le_bytes(), 0), "blez\ta0, 0x4");
    assert_eq!(text_at(&enc::bcmp(4, 0, 10, 4).to_le_bytes(), 0), "bgtz\ta0, 0x4");
    assert_eq!(text_at(&enc::jal(0, 16).to_le_bytes(), 0x20), "j\t0x30");
    assert_eq!(text_at(&enc::jal(1, -16).to_le_bytes(), 0x20), "jal\t0x10");
    assert_eq!(text_at(&enc::jal(10, 4).to_le_bytes(), 0), "jal\ta0, 0x4");
    let i = decode(TargetArch::Riscv64, &enc::jal(1, 64).to_le_bytes(), 0x1000, &Options::default());
    assert_eq!((i.target, i.target_operand), (Some(0x1040), Some(0)));

    // Compressed instructions print as the instruction they expand to.
    let c: &[(u16, &str)] = &[
        (0x0505, "addi\ta0, a0, 0x1"),
        (0x55f5, "li\ta1, -0x3"),
        (0x852e, "mv\ta0, a1"),
        (0x8082, "ret"),
        (0x41c8, "lw\ta0, 0x4(a1)"),
        (0x6422, "ld\ts0, 0x8(sp)"),
        (0x713d, "addi\tsp, sp, -0x20"),
        (0x0808, "addi\ta0, sp, 0x10"),
        (0x0001, "nop"),
        (0x9002, "ebreak"),
        (0x667d, "lui\ta2, 0x1f"),
        (0x757d, "lui\ta0, 0xfffff"),
        (0x050e, "slli\ta0, a0, 0x3"),
        (0x952e, "add\ta0, a0, a1"),
        (0x9502, "jalr\ta0"),
        (0x8502, "jr\ta0"),
        (0x9d0d, "subw\ta0, a0, a1"),
        (0x2588, "fld\tfa0, 0x8(a1)"),
        (0xa82e, "fsd\tfa1, 0x10(sp)"),
        (0xe82a, "sd\ta0, 0x10(sp)"),
        (0xc188, "sw\ta0, 0x0(a1)"),
    ];
    for &(h, want) in c {
        assert_eq!(half(h), want, "{h:#06x}");
    }
    assert_eq!(text_at(&0xa001u16.to_le_bytes(), 0x16), "j\t0x16");
    assert_eq!(text_at(&0xc101u16.to_le_bytes(), 0x18), "beqz\ta0, 0x18");
    assert_eq!(decode(TargetArch::Riscv64, &0x952eu16.to_le_bytes(), 0, &Options::default()).len, 2);
}

#[test]
fn unknown_encodings_are_data() {
    let opts = Options::default();
    // The all-zero halfword is the defined illegal instruction, `unimp`;
    // other zero `c.addi4spn` immediates are reserved.
    let i = decode(TargetArch::Riscv64, &[0, 0, 0, 0], 0, &opts);
    assert_eq!((i.known, i.len, i.text()), (true, 2, "unimp".to_owned()));
    let i = decode(TargetArch::Riscv64, &[0x04, 0x00], 0, &opts);
    assert_eq!((i.known, i.len, i.text()), (false, 2, ".short\t0x0004".to_owned()));
    // A reserved 32-bit encoding.
    let i = decode(TargetArch::Riscv64, &0xffff_ffffu32.to_le_bytes(), 0, &opts);
    assert!(!i.known);
    // Reserved rounding mode 5.
    assert!(decode_inst(&0x02c5_d553u32.to_le_bytes(), 0).is_none());
    // A truncated 32-bit instruction.
    let i = decode(TargetArch::Riscv64, &[0x13, 0x05], 0, &opts);
    assert_eq!((i.known, i.len), (false, 2));
}

// ===========================================================================
// Round trip against the encoder
// ===========================================================================

/// A fuzzed corpus of `(word, the instruction it encodes)` from the
/// backend's instruction builders.
fn encoder_corpus(rng: &mut Rng, n: usize) -> Vec<(u32, RvInst)> {
    use Operand::{Addr, Fence, Imm, Mem, Target, X};
    let r = |rng: &mut Rng| rng.below(32) as u32;
    let x = |v: u32| X(v as u8);
    let mk = |m: &str, ops: Vec<Operand>| RvInst { len: 4, mnemonic: m.to_owned(), ops };
    let mut out = Vec::new();
    while out.len() < n {
        let (d, a, b) = (r(rng), r(rng), r(rng));
        let imm12 = (rng.below(4096) as i32) - 2048;
        let shamt = rng.below(64) as u32;
        let br = ((rng.below(4096) as i32) - 2048) * 2;
        let jo = ((rng.below(1 << 20) as i32) - (1 << 19)) * 2;
        let size = [1u64, 2, 4, 8][rng.below(4) as usize];
        let item = match rng.below(14) {
            0 => {
                type RFn = fn(u32, u32, u32) -> u32;
                let ops: [(RFn, &str); 18] = [
                    (enc::add, "add"),
                    (enc::sub, "sub"),
                    (enc::sll, "sll"),
                    (enc::slt, "slt"),
                    (enc::sltu, "sltu"),
                    (enc::xor, "xor"),
                    (enc::srl, "srl"),
                    (enc::sra, "sra"),
                    (enc::or, "or"),
                    (enc::and, "and"),
                    (enc::mul, "mul"),
                    (enc::mulh, "mulh"),
                    (enc::div, "div"),
                    (enc::divu, "divu"),
                    (enc::rem, "rem"),
                    (enc::remu, "remu"),
                    (enc::add, "add"),
                    (enc::sub, "sub"),
                ];
                let (f, m) = ops[rng.below(18) as usize];
                (f(d, a, b), mk(m, vec![x(d), x(a), x(b)]))
            }
            1 => {
                type IFn = fn(u32, u32, i32) -> u32;
                let ops: [(IFn, &str); 6] = [
                    (enc::addi, "addi"),
                    (enc::addiw, "addiw"),
                    (enc::andi, "andi"),
                    (enc::ori, "ori"),
                    (enc::xori, "xori"),
                    (enc::sltiu, "sltiu"),
                ];
                let (f, m) = ops[rng.below(6) as usize];
                (f(d, a, imm12), mk(m, vec![x(d), x(a), Imm(i64::from(imm12))]))
            }
            2 => {
                type SFn = fn(u32, u32, u32) -> u32;
                let ops: [(SFn, &str); 3] = [(enc::slli, "slli"), (enc::srli, "srli"), (enc::srai, "srai")];
                let (f, m) = ops[rng.below(3) as usize];
                (f(d, a, shamt), mk(m, vec![x(d), x(a), Imm(i64::from(shamt))]))
            }
            3 => {
                let m = match size {
                    1 => "lbu",
                    2 => "lhu",
                    4 => "lwu",
                    _ => "ld",
                };
                (enc::load(size, d, a, imm12), mk(m, vec![x(d), Mem(a as u8, i64::from(imm12))]))
            }
            4 => {
                let m = ["", "sb", "sh", "", "sw", "", "", "", "sd"][size as usize];
                (enc::store(size, d, a, imm12), mk(m, vec![x(d), Mem(a as u8, i64::from(imm12))]))
            }
            5 => {
                let imm20 = rng.below(1 << 20) as u32;
                if rng.below(2) == 0 {
                    (enc::lui(d, imm20), mk("lui", vec![x(d), Imm(i64::from(imm20))]))
                } else {
                    (enc::auipc(d, imm20), mk("auipc", vec![x(d), Imm(i64::from(imm20))]))
                }
            }
            6 => (enc::jal(d, jo), mk("jal", vec![x(d), Target(jo as i64 as u64)])),
            7 => {
                let f3 = [0u32, 1, 4, 5, 6, 7][rng.below(6) as usize];
                let m = ["beq", "bne", "", "", "blt", "bge", "bltu", "bgeu"][f3 as usize];
                (enc::bcmp(f3, a, b, br), mk(m, vec![x(a), x(b), Target(br as i64 as u64)]))
            }
            8 => (enc::jalr(d, a, imm12), mk("jalr", vec![x(d), Mem(a as u8, i64::from(imm12))])),
            9 => {
                let funct5 = [0u32, 1, 4, 8, 12, 16, 20, 24, 28][rng.below(9) as usize];
                let base = match funct5 {
                    0 => "amoadd",
                    1 => "amoswap",
                    4 => "amoxor",
                    8 => "amoor",
                    12 => "amoand",
                    16 => "amomin",
                    20 => "amomax",
                    24 => "amominu",
                    _ => "amomaxu",
                };
                let (aq, rl) = (rng.below(2) == 1, rng.below(2) == 1);
                let size = if rng.below(2) == 0 { 4 } else { 8 };
                let m = format!("{base}.{}{}", if size == 4 { "w" } else { "d" }, order(aq, rl));
                (enc::amo(funct5, aq, rl, d, b, a, size), mk(&m, vec![x(d), x(b), Addr(a as u8)]))
            }
            10 => {
                let (aq, rl) = (rng.below(2) == 1, rng.below(2) == 1);
                let size = if rng.below(2) == 0 { 4 } else { 8 };
                let w = if size == 4 { "w" } else { "d" };
                if rng.below(2) == 0 {
                    (enc::lr(size, aq, rl, d, a), mk(&format!("lr.{w}{}", order(aq, rl)), vec![x(d), Addr(a as u8)]))
                } else {
                    (enc::sc(size, aq, rl, d, b, a), mk(&format!("sc.{w}{}", order(aq, rl)), vec![x(d), x(b), Addr(a as u8)]))
                }
            }
            11 => {
                let (p, s) = (rng.below(16) as u8, rng.below(16) as u8);
                let inst = match (p, s) {
                    (15, 15) => mk("fence", vec![]),
                    _ => mk("fence", vec![Fence(p), Fence(s)]),
                };
                (enc::fence(0, u32::from(p), u32::from(s)), inst)
            }
            12 => match rng.below(3) {
                0 => (enc::ecall(), mk("ecall", vec![])),
                1 => (enc::ebreak(), mk("ebreak", vec![])),
                _ => (enc::fence(8, 3, 3), mk("fence.tso", vec![])),
            },
            _ => (enc::mv(d, a), mk("addi", vec![x(d), x(a), Imm(0)])),
        };
        out.push(item);
    }
    out
}

fn order(aq: bool, rl: bool) -> &'static str {
    match (aq, rl) {
        (false, false) => "",
        (false, true) => ".rl",
        (true, false) => ".aq",
        (true, true) => ".aqrl",
    }
}

/// Assemble one line with rsasm (no compression) and return the `.text`
/// bytes, or `None` when rsasm rejects it.
fn rsasm(line: &str) -> Option<Vec<u8>> {
    let src = format!(".option norvc\n{line}\n");
    let opts = crate::mc::asm::AsmOptions::new(TargetArch::Riscv64);
    let elf = crate::mc::asm::assemble(&[crate::mc::asm::AsmSource { name: "rt.s", text: &src }], &opts).ok()?;
    let bin = crate::mc::disasm::objfile::read(&elf).ok()?;
    bin.sections.into_iter().find(|s| s.name == ".text").map(|s| s.bytes)
}

/// The assembly text of a decoded instruction at address 0 with any
/// branch target written relative to `.`, as an assembler reads it.
fn reassemblable(inst: &crate::mc::disasm::Inst) -> String {
    let mut i = inst.clone();
    if let (Some(t), Some(k)) = (i.target, i.target_operand) {
        let off = t as i64;
        i.operands[k] = if off < 0 { format!(". - {:#x}", off.unsigned_abs()) } else { format!(". + {off:#x}") };
    }
    i.text().replace('\t', " ")
}

#[test]
fn encoder_round_trip() {
    let mut rng = Rng(0x005e_ed0f_5ca1_ab1e);
    let corpus = encoder_corpus(&mut rng, 3000);
    let (mut exact, mut fixpoint, mut structural_only) = (0, 0, 0);
    for (w, want) in &corpus {
        let bytes = w.to_le_bytes();
        let got = decode_inst(&bytes, 0).unwrap_or_else(|| panic!("{w:#010x} ({want:?}) did not decode"));
        assert_eq!(&got, want, "{w:#010x}");
        let inst = decode(TargetArch::Riscv64, &bytes, 0, &Options::default());
        assert!(inst.known && inst.len == 4);
        // Re-encode: the canonical text, or else the raw instruction's.
        let canonical = reassemblable(&inst);
        let raw = reassemblable(&got.to_inst());
        match rsasm(&canonical).or_else(|| rsasm(&raw)) {
            Some(b) if b == bytes => exact += 1,
            Some(b) => {
                let again = decode(TargetArch::Riscv64, &b, 0, &Options::default());
                assert_eq!(again.text(), inst.text(), "{w:#010x}: `{canonical}` re-assembled to {b:02x?}");
                fixpoint += 1;
            }
            None => structural_only += 1,
        }
    }
    // `li` sequences for random constants decode instruction by instruction.
    for _ in 0..300 {
        let v = rng.next() as i64 >> rng.below(64);
        let rd = 1 + rng.below(31) as u32;
        let bytes = enc::emit_li_bytes(rd, v);
        let insts = disassemble(TargetArch::Riscv64, &bytes, 0, &Options::default());
        assert!(insts.iter().all(|(_, i)| i.known && i.len == 4), "li {v:#x}: {insts:?}");
        let first = &insts[0].1.mnemonic;
        assert!(matches!(first.as_str(), "lui" | "li"), "li {v:#x} starts with {first}");
    }
    eprintln!(
        "riscv encoder round trip: {} instructions decoded structurally; re-assembled by rsasm: {exact} byte-exact, \
         {fixpoint} text fixpoint, {structural_only} not assemblable by rsasm",
        corpus.len()
    );
    assert!(exact > corpus.len() / 2, "too few byte-exact round trips: {exact}");
}

// ===========================================================================
// llvm-objdump differentials
// ===========================================================================

/// `code` as the `.text` of an ELF RISC-V object.
fn elf_of(code: &[u8]) -> Vec<u8> {
    let mut obj = ObjectModule::new("raw");
    let s = obj.add_section(Section::new(".text", SectionKind::Text, 4));
    obj.section_mut(s).bytes = code.to_vec();
    object_file(TargetArch::Riscv64, &obj, ObjectFormat::Elf)
}

/// Compare our straight disassembly of `code` (at address 0) with
/// llvm-objdump's, `<unknown>` matching our data directives. Returns
/// `(compared, mismatches)`; `None` without llvm-objdump.
fn compare_raw(code: &[u8]) -> Option<(usize, Vec<String>)> {
    let theirs = objdump(&elf_of(code), &[MATTR])?;
    let theirs = theirs.get(".text").cloned().unwrap_or_default();
    let ours: std::collections::BTreeMap<u64, crate::mc::disasm::Inst> =
        disassemble(TargetArch::Riscv64, code, 0, &Options::default()).into_iter().collect();
    let mut bad = Vec::new();
    for (addr, text) in &theirs {
        let Some(inst) = ours.get(addr) else {
            bad.push(format!("{addr:#x}: llvm `{text}` has no counterpart"));
            continue;
        };
        let at = *addr as usize;
        let raw = &code[at..(at + inst.len.max(2)).min(code.len())];
        let ok = if text.starts_with("<unknown>") {
            !inst.known
        } else {
            inst.known && normalize(TargetArch::Riscv64, &inst.text()) == normalize(TargetArch::Riscv64, text)
        };
        if !ok {
            bad.push(format!("{addr:#x} {raw:02x?}: ours `{}` | llvm `{text}`", inst.text()));
        }
    }
    Some((theirs.len(), bad))
}

fn report(what: &str, compared: usize, bad: &[String]) {
    eprintln!("{what}: {compared} instructions compared with llvm-objdump, {} mismatches", bad.len());
    assert!(bad.is_empty(), "{what}:\n{}", bad.iter().take(200).cloned().collect::<Vec<_>>().join("\n"));
    assert!(compared > 0);
}

/// Every 16-bit encoding (the three compressed quadrants) against
/// llvm-objdump.
#[test]
fn every_compressed_encoding_matches_llvm() {
    let mut code = Vec::new();
    for h in 0..=0xffffu16 {
        if h & 3 != 3 {
            code.extend_from_slice(&h.to_le_bytes());
        }
    }
    let Some((n, bad)) = compare_raw(&code) else {
        eprintln!("skipping: no llvm-objdump");
        return;
    };
    report("riscv compressed (all 49152 encodings)", n, &bad);
}

/// Every CSR number, named or not, against llvm-objdump.
#[test]
fn every_csr_matches_llvm() {
    let mut code = Vec::new();
    for csr in 0..4096u32 {
        // csrrs a0, csr, a1
        code.extend_from_slice(&(csr << 20 | 11 << 15 | 2 << 12 | 10 << 7 | 0x73).to_le_bytes());
    }
    let Some((n, bad)) = compare_raw(&code) else {
        eprintln!("skipping: no llvm-objdump");
        return;
    };
    report("riscv CSR names (all 4096)", n, &bad);
}

/// Random 32-bit words over every major opcode against llvm-objdump.
#[test]
fn random_words_match_llvm() {
    const OPCODES: [u32; 22] = [
        0x03, 0x07, 0x0f, 0x13, 0x17, 0x1b, 0x23, 0x27, 0x2f, 0x33, 0x37, 0x3b, 0x43, 0x47, 0x4b, 0x4f, 0x53, 0x63,
        0x67, 0x6f, 0x73, 0x53,
    ];
    let mut rng = Rng(0xfeed_beef_1234_5678);
    let mut code = Vec::new();
    for k in 0..60_000 {
        let mut w = (rng.next() as u32 & !0x7f) | OPCODES[k % OPCODES.len()];
        // Bias the funct7/funct5 fields towards defined values half the time.
        if rng.below(2) == 0 {
            let f7 = [0x00u32, 0x01, 0x20, 0x04, 0x05, 0x10, 0x30, 0x08, 0x09, 0x0c, 0x0d][rng.below(11) as usize];
            w = (w & 0x01ff_ffff) | f7 << 25;
        }
        if w & 0x7f == 0x53 && rng.below(2) == 0 {
            let f7 = [0x00u32, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d, 0x10, 0x11, 0x14, 0x15, 0x20, 0x21, 0x2c,
                0x2d, 0x50, 0x51, 0x60, 0x61, 0x68, 0x69, 0x70, 0x71, 0x78, 0x79][rng.below(26) as usize];
            w = (w & 0x01ff_ffff) | f7 << 25;
            if rng.below(2) == 0 {
                w = (w & !(0x1f << 20)) | (rng.below(4) as u32) << 20;
            }
        }
        code.extend_from_slice(&w.to_le_bytes());
    }
    let Some((n, bad)) = compare_raw(&code) else {
        eprintln!("skipping: no llvm-objdump");
        return;
    };
    report("riscv random 32-bit words", n, &bad);
}

/// Hand-written instructions of every extension, assembled by llvm-mc.
const ASM_CORPUS: &str = r"
add a0, a1, a2
sub t0, t1, t2
sll s2, s3, s4
slt a0, a1, a2
sltu a0, a1, a2
xor a0, a1, a2
srl a0, a1, a2
sra a0, a1, a2
or a0, a1, a2
and a0, a1, a2
addw a0, a1, a2
subw a0, a1, a2
sllw a0, a1, a2
srlw a0, a1, a2
sraw a0, a1, a2
addi a0, a1, -2048
addi a0, a1, 2047
slti a0, a1, -1
sltiu a0, a1, 5
xori a0, a1, 3
ori a0, a1, 0
andi a0, a1, 1
slli a0, a1, 1
srli a0, a1, 33
srai a0, a1, 63
addiw a0, a1, 1
slliw a0, a1, 31
srliw a0, a1, 1
sraiw a0, a1, 17
lui a0, 0x80000
auipc t1, 0xfffff
lb a0, -1(a1)
lh a0, 2(a1)
lw a0, 4(a1)
ld a0, 8(a1)
lbu a0, 0(a1)
lhu a0, 6(a1)
lwu a0, 12(a1)
sb a0, -1(a1)
sh a0, 2(a1)
sw a0, 4(a1)
sd a0, 8(a1)
beq a0, a1, 16
bne a0, a1, -16
blt a0, a1, 16
bge a0, a1, 16
bltu a0, a1, 16
bgeu a0, a1, 16
jal ra, 2048
jal zero, -2048
jal t0, 4
jalr ra, 0(t0)
jalr zero, 12(t0)
jalr t1, -4(t0)
nop
li a0, 7
mv a0, a1
not a0, a1
neg a0, a1
negw a0, a1
sext.w a0, a1
seqz a0, a1
snez a0, a1
sltz a0, a1
sgtz a0, a1
beqz a0, 8
bnez a0, 8
blez a0, 8
bgez a0, 8
bltz a0, 8
bgtz a0, 8
ret
fence
fence rw, rw
fence r, rw
fence w, w
fence io, iorw
fence.tso
fence.i
ecall
ebreak
mul a0, a1, a2
mulh a0, a1, a2
mulhsu a0, a1, a2
mulhu a0, a1, a2
div a0, a1, a2
divu a0, a1, a2
rem a0, a1, a2
remu a0, a1, a2
mulw a0, a1, a2
divw a0, a1, a2
divuw a0, a1, a2
remw a0, a1, a2
remuw a0, a1, a2
lr.w a0, (a1)
lr.d.aqrl a0, (a1)
sc.w.aq a0, a2, (a1)
sc.d a0, a2, (a1)
amoswap.w.aq a0, a2, (a1)
amoadd.w a0, a2, (a1)
amoxor.d.rl a0, a2, (a1)
amoand.w a0, a2, (a1)
amoor.d a0, a2, (a1)
amomin.w a0, a2, (a1)
amomax.d a0, a2, (a1)
amominu.w a0, a2, (a1)
amomaxu.d.aqrl a0, a2, (a1)
flw fa0, 4(a1)
fsw fa0, 4(a1)
fld fs0, -8(sp)
fsd fs11, 2040(sp)
fadd.s fa0, fa1, fa2
fsub.s fa0, fa1, fa2, rtz
fmul.s fa0, fa1, fa2, rdn
fdiv.s fa0, fa1, fa2, rup
fsqrt.s fa0, fa1, rmm
fadd.d ft0, ft1, ft2
fsub.d ft8, ft9, ft10
fmul.d fa0, fa1, fa2
fdiv.d fa0, fa1, fa2, rne
fsqrt.d fa0, fa1
fmadd.s fa0, fa1, fa2, fa3
fmsub.s fa0, fa1, fa2, fa3, rtz
fnmsub.d fa0, fa1, fa2, fa3
fnmadd.d fa0, fa1, fa2, fa3, rdn
fsgnj.s fa0, fa1, fa2
fsgnjn.s fa0, fa1, fa2
fsgnjx.d fa0, fa1, fa2
fmv.s fa0, fa1
fneg.s fa0, fa1
fabs.d fa0, fa1
fmin.s fa0, fa1, fa2
fmax.d fa0, fa1, fa2
feq.s a0, fa1, fa2
flt.d a0, fa1, fa2
fle.s a0, fa1, fa2
fclass.s a0, fa1
fclass.d a0, fa1
fcvt.w.s a0, fa1
fcvt.wu.s a0, fa1, rtz
fcvt.l.s a0, fa1, rdn
fcvt.lu.s a0, fa1
fcvt.w.d a0, fa1, rtz
fcvt.wu.d a0, fa1
fcvt.l.d a0, fa1, rtz
fcvt.lu.d a0, fa1, rmm
fcvt.s.w fa0, a1
fcvt.s.wu fa0, a1, rtz
fcvt.s.l fa0, a1
fcvt.s.lu fa0, a1
fcvt.d.w fa0, a1
fcvt.d.wu fa0, a1
fcvt.d.l fa0, a1, rtz
fcvt.d.lu fa0, a1
fcvt.s.d fa0, fa1
fcvt.d.s fa0, fa1
fmv.x.w a0, fa1
fmv.w.x fa0, a1
fmv.x.d a0, fa1
fmv.d.x fa0, a1
csrrw a0, mscratch, a1
csrrs a0, mstatus, a1
csrrc a0, mie, a1
csrrwi a0, mscratch, 5
csrrsi a0, mstatus, 31
csrrci a0, mie, 1
csrr a0, mepc
csrw mtvec, a0
csrs mstatus, a0
csrc mstatus, a0
csrwi mscratch, 3
csrsi mie, 1
csrci mie, 2
frflags a0
fsflags a1
fsflags a0, a1
frrm a0
fsrm a1
frcsr a0
fscsr a0, a1
fsrmi 3
fsflagsi a0, 1
rdcycle a0
rdtime a0
rdinstret a0
wfi
mret
sret
sfence.vma
sfence.vma a0, a1
add.uw a0, a1, a2
zext.w a0, a1
sh1add a0, a1, a2
sh2add a0, a1, a2
sh3add a0, a1, a2
sh1add.uw a0, a1, a2
sh3add.uw a0, a1, a2
slli.uw a0, a1, 40
andn a0, a1, a2
orn a0, a1, a2
xnor a0, a1, a2
clz a0, a1
clzw a0, a1
ctz a0, a1
ctzw a0, a1
cpop a0, a1
cpopw a0, a1
max a0, a1, a2
maxu a0, a1, a2
min a0, a1, a2
minu a0, a1, a2
sext.b a0, a1
sext.h a0, a1
zext.h a0, a1
rol a0, a1, a2
rolw a0, a1, a2
ror a0, a1, a2
rorw a0, a1, a2
rori a0, a1, 7
roriw a0, a1, 7
orc.b a0, a1
rev8 a0, a1
c.addi4spn a0, sp, 1020
c.fld fa0, 248(a1)
c.lw a0, 124(a1)
c.ld a0, 248(a1)
c.fsd fa0, 0(a1)
c.sw a0, 64(a1)
c.sd a0, 128(a1)
c.nop
c.addi a0, -32
c.addiw a0, 31
c.li a0, -1
c.addi16sp sp, 496
c.lui a0, 0x1
c.srli a0, 1
c.srai a0, 63
c.andi a0, -1
c.sub a0, a1
c.xor a0, a1
c.or a0, a1
c.and a0, a1
c.subw a0, a1
c.addw a0, a1
c.j -2048
c.beqz a0, -256
c.bnez a0, 254
c.slli a0, 63
c.fldsp fa0, 504(sp)
c.lwsp a0, 252(sp)
c.ldsp a0, 504(sp)
c.jr a0
c.mv a0, a1
c.ebreak
c.jalr a1
c.add a0, a1
c.fsdsp fa0, 504(sp)
c.swsp a0, 252(sp)
c.sdsp a0, 504(sp)
";

#[test]
fn llvm_mc_corpus_matches_llvm_objdump() {
    let Some(mc) = llvm_tool("llvm-mc") else {
        eprintln!("skipping: no llvm-mc");
        return;
    };
    let dir = scratch("rv-mc");
    let src = dir.join("corpus.s");
    let obj = dir.join("corpus.o");
    std::fs::write(&src, ASM_CORPUS).unwrap();
    let out = std::process::Command::new(mc)
        .args(["--triple=riscv64", "-mattr=+m,+a,+f,+d,+c,+zba,+zbb", "-filetype=obj", "-o"])
        .arg(&obj)
        .arg(&src)
        .output()
        .expect("run llvm-mc");
    assert!(out.status.success(), "llvm-mc: {}", String::from_utf8_lossy(&out.stderr));
    let file = std::fs::read(&obj).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let Some(rep) = differential(TargetArch::Riscv64, &file, &[MATTR], &Options::default(), &|s| s) else {
        eprintln!("skipping: no llvm-objdump");
        return;
    };
    assert!(rep.compared >= ASM_CORPUS.lines().filter(|l| !l.trim().is_empty()).count());
    assert_clean("riscv llvm-mc corpus", &rep);
}

/// Objects LF compiles for RISC-V disassemble exactly as llvm-objdump does.
#[test]
fn compiled_objects_match_llvm_objdump() {
    let mut total = 0;
    for (name, src) in [("ints", INTS), ("floats", FLOATS)] {
        // The backend may not compile everything yet (floating point is in
        // progress): compare what it does compile.
        let compiled = std::panic::catch_unwind(|| compile(TargetArch::Riscv64, src));
        let Ok(obj) = compiled else {
            eprintln!("riscv differential: the backend cannot compile `{name}` yet; skipped");
            continue;
        };
        let file = object_file(TargetArch::Riscv64, &obj, ObjectFormat::Elf);
        let Some(rep) = differential(TargetArch::Riscv64, &file, &[MATTR], &Options::default(), &|s| s) else {
            eprintln!("skipping: no llvm-objdump");
            return;
        };
        assert_clean(&format!("riscv compiled `{name}`"), &rep);
        total += rep.compared;
    }
    assert!(total > 0, "no RISC-V object compiled");
}
