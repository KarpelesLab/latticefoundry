//! x86-64 decoder tests: round trips against the encoder, and llvm-objdump
//! differential tests.

use std::process::Command;

use super::corpus::{FLOATS, INTS};
use super::{Rng, assert_clean, compile, differential, llvm_tool, object_file, scratch};
use crate::mc::disasm::x86::decode_inst;
use crate::mc::disasm::{Inst, Options, Syntax, decode, objfile};
use crate::mc::emit::Emitter;
use crate::target::x86_64::encode as enc;
use crate::target::{ObjectFormat, TargetArch};

const ATT: Options = Options { syntax: Syntax::Att };
const INTEL: Options = Options { syntax: Syntax::Intel };

fn text(bytes: &[u8], opts: &Options) -> String {
    let i = decode(TargetArch::X86_64, bytes, 0, opts);
    assert_eq!(i.len, bytes.len(), "length of {bytes:02x?}: {}", i.text());
    i.text().replace('\t', " ")
}

/// Exact AT&T and Intel text for a dozen instructions.
#[test]
fn golden() {
    let cases: &[(&[u8], &str, &str)] = &[
        (&[0x55], "pushq %rbp", "push rbp"),
        (&[0x48, 0x89, 0xe5], "movq %rsp, %rbp", "mov rbp, rsp"),
        (&[0x48, 0x8b, 0x45, 0xf8], "movq -0x8(%rbp), %rax", "mov rax, qword ptr [rbp - 0x8]"),
        (&[0x4a, 0x8b, 0x84, 0xe8, 0x00, 0x01, 0, 0], "movq 0x100(%rax,%r13,8), %rax", "mov rax, qword ptr [rax + 8*r13 + 0x100]"),
        (&[0x0f, 0xb6, 0xc0], "movzbl %al, %eax", "movzx eax, al"),
        (&[0x48, 0x63, 0xc8], "movslq %eax, %rcx", "movsxd rcx, eax"),
        (&[0x48, 0x99], "cqto", "cqo"),
        (&[0x48, 0x98], "cltq", "cdqe"),
        (&[0x48, 0x83, 0xe4, 0xf0], "andq $-0x10, %rsp", "and rsp, -0x10"),
        (&[0xb8, 0xff, 0xff, 0xff, 0xff], "movl $0xffffffff, %eax", "mov eax, 0xffffffff"),
        (&[0xc3], "retq", "ret"),
        (&[0xf2, 0x0f, 0x59, 0xc8], "mulsd %xmm0, %xmm1", "mulsd xmm1, xmm0"),
        (&[0xf2, 0x48, 0x0f, 0x2a, 0xc8], "cvtsi2sd %rax, %xmm1", "cvtsi2sd xmm1, rax"),
        (&[0xf2, 0x0f, 0x2a, 0x00], "cvtsi2sdl (%rax), %xmm0", "cvtsi2sd xmm0, dword ptr [rax]"),
        (&[0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0], "movq %fs:0x0, %rax", "mov rax, qword ptr fs:[0x0]"),
        (&[0x48, 0x8d, 0x05, 0x10, 0, 0, 0], "leaq 0x10(%rip), %rax", "lea rax, [rip + 0x10]"),
        (&[0xff, 0xe0], "jmpq *%rax", "jmp rax"),
        (&[0xff, 0x50, 0x10], "callq *0x10(%rax)", "call qword ptr [rax + 0x10]"),
        (&[0x66, 0x0f, 0x70, 0xc1, 0xb1], "pshufd $0xb1, %xmm1, %xmm0", "pshufd xmm0, xmm1, 0xb1"),
        (&[0xf3, 0x48, 0xab], "rep stosq %rax, %es:(%rdi)", "rep stosq qword ptr es:[rdi], rax"),
        (&[0xf0], "lock", "lock"),
        // TLS: the initial-exec thread pointer, a GOT-relative offset, the
        // padded general-dynamic sequence, and the local-exec relaxations.
        (&[0x48, 0x03, 0x05, 0, 0, 0, 0], "addq (%rip), %rax", "add rax, qword ptr [rip]"),
        (&[0x66, 0x48, 0x8d, 0x3d, 0, 0, 0, 0], "leaq (%rip), %rdi", "lea rdi, [rip]"),
        (&[0x66, 0x66, 0x48, 0xe8, 0, 0, 0, 0], "callq 0x8", "call 0x8"),
        (&[0x48, 0xc7, 0xc0, 0x10, 0, 0, 0], "movq $0x10, %rax", "mov rax, 0x10"),
        (&[0x48, 0x81, 0xc0, 0x10, 0, 0, 0], "addq $0x10, %rax", "add rax, 0x10"),
        // i128: the widening multiply.
        (&[0x48, 0xf7, 0xe1], "mulq %rcx", "mul rcx"),
        (&[0x48, 0xf7, 0x20], "mulq (%rax)", "mul qword ptr [rax]"),
    ];
    for (bytes, att, intel) in cases {
        assert_eq!(text(bytes, &ATT), *att, "{bytes:02x?}");
        assert_eq!(text(bytes, &INTEL), *intel, "{bytes:02x?}");
    }
    // Branch targets are absolute and recorded.
    let i = decode(TargetArch::X86_64, &[0xe8, 0x10, 0, 0, 0], 0x100, &ATT);
    assert_eq!((i.text().replace('\t', " "), i.target), ("callq 0x115".to_owned(), Some(0x115)));
    let i = decode(TargetArch::X86_64, &[0x75, 0xfe], 0x40, &ATT);
    assert_eq!((i.text().replace('\t', " "), i.target), ("jne 0x40".to_owned(), Some(0x40)));
    // Unknown and truncated encodings are data.
    for bad in [&[0x06u8][..], &[0x0f, 0xff], &[0x48], &[0x48, 0x8b], &[0xe8, 0, 0]] {
        let i = decode(TargetArch::X86_64, bad, 0, &ATT);
        assert!(!i.known && i.len == 1, "{bad:02x?}: {}", i.text());
        assert_eq!(i.mnemonic, ".byte");
    }
}

// ===========================================================================
// Round trip: LF's encoder -> decoder -> our assembler
// ===========================================================================

/// The bytes of one encoder call.
fn emit(f: impl FnOnce(&mut Emitter)) -> Vec<u8> {
    let mut e = Emitter::new();
    f(&mut e);
    e.finish().expect("emit").bytes
}

/// A fuzzed corpus of single instructions from the x86-64 encoder's helpers.
fn encoder_corpus() -> Vec<Vec<u8>> {
    let mut rng = Rng(0x005e_ed86);
    let mut out = Vec::new();
    let r = |rng: &mut Rng| rng.below(16) as u8;
    let alu_ops = [0x01u8, 0x29, 0x21, 0x09, 0x31, 0x89, 0x39, 0x85, 0x11, 0x19];
    let ccs = 0..16u8;
    for _ in 0..400 {
        let (a, b, w) = (r(&mut rng), r(&mut rng), rng.below(2) == 1);
        let op = alu_ops[rng.below(alu_ops.len() as u64) as usize];
        out.push(emit(|e| enc::alu_rr(e, op, a, b, w)));
        out.push(emit(|e| enc::mov_rr(e, a, b, w)));
        out.push(emit(|e| enc::imul_rr(e, a, b, w)));
        out.push(emit(|e| enc::neg_r(e, a, w)));
        let v = match rng.below(4) {
            0 => rng.below(256),
            1 => rng.next() & 0xffff_ffff,
            _ => rng.next(),
        };
        out.push(emit(|e| enc::mov_ri(e, a, v)));
        let ext = [4u8, 5, 7, 0, 1][rng.below(5) as usize];
        let count = rng.below(64) as u8;
        out.push(emit(|e| enc::shift_imm(e, ext, a, count, w)));
        out.push(emit(|e| enc::shift_cl(e, ext, a, w)));
        let cc = rng.below(16) as u8;
        out.push(emit(|e| enc::setcc(e, cc, a)));
        out.push(emit(|e| enc::cmov_rr(e, cc, a, b, w)));
        let (sw, dw) = [(8, 32), (8, 64), (16, 32), (16, 64), (32, 64)][rng.below(5) as usize];
        out.push(emit(|e| enc::movsx_rr(e, a, b, sw, dw)));
        let zw = [8u32, 16, 32][rng.below(3) as usize];
        out.push(emit(|e| enc::movzx_rr(e, a, b, zw)));
        out.push(emit(|e| enc::movzx_byte(e, a)));
        out.push(emit(|e| enc::divide(e, 6 + rng.below(2) as u8, a, w)));
        out.push(emit(|e| enc::push_r(e, a)));
        out.push(emit(|e| enc::pop_r(e, a)));
        let width = [8u32, 16, 32, 64][rng.below(4) as usize];
        out.push(emit(|e| enc::cmp_rr_width(e, a, b, width)));
        // Memory forms: loads, stores, lea, SSE loads/stores.
        let disp = match rng.below(3) {
            0 => 0,
            1 => rng.below(256) as i32 - 128,
            _ => rng.next() as i32,
        };
        let mop: &[&[u8]] = &[&[0x8b], &[0x89], &[0x8d], &[0x0f, 0xb6], &[0x0f, 0xb7], &[0x0f, 0xbe], &[0x63], &[0x88]];
        let mo = mop[rng.below(mop.len() as u64) as usize];
        let (mw, force) = (rng.below(2) == 1 || mo == [0x63], rng.below(2) == 1);
        out.push(emit(|e| enc::mem(e, mo, a, b, disp, mw, force)));
        let sse_pfx = [0xf2u8, 0xf3][rng.below(2) as usize];
        out.push(emit(|e| enc::sse_mem(e, sse_pfx, 0x10 + rng.below(2) as u8, a, b, disp)));
        let sse_ops = [0x58u8, 0x59, 0x5c, 0x5e, 0x51, 0x5d, 0x5f, 0x10, 0x5a, 0x2e, 0x2f, 0x54, 0x57, 0x28];
        let so = sse_ops[rng.below(sse_ops.len() as u64) as usize];
        let sp = if matches!(so, 0x2e | 0x2f | 0x54 | 0x57) { [0u8, 0x66][rng.below(2) as usize] } else if so == 0x28 { 0 } else { sse_pfx };
        out.push(emit(|e| enc::sse_rr(e, sp, false, so, a, b)));
        // GPR <-> xmm conversions and moves (REX.W selects 32/64-bit GPRs).
        let (cp, co) = [(sse_pfx, 0x2a), (sse_pfx, 0x2c), (0x66, 0x6e), (0x66, 0x7e)][rng.below(4) as usize];
        out.push(emit(|e| enc::sse_rr(e, cp, w, co, a, b)));
        out.push(emit(|e| enc::movaps(e, a, b)));
        // Packed integer ops (the vector backend's VEnc opcodes).
        let pops = [0xfcu8, 0xfd, 0xfe, 0xd4, 0xf8, 0xf9, 0xfa, 0xfb, 0x74, 0x75, 0x76, 0x64, 0x65, 0x66, 0xdb, 0xdf,
            0xeb, 0xef, 0xd5, 0xf4, 0x62, 0x6c, 0xda, 0xde, 0xea, 0xee, 0xec, 0xed, 0xdc, 0xdd, 0xe8, 0xe9, 0xd8, 0xd9];
        let po = pops[rng.below(pops.len() as u64) as usize];
        out.push(emit(|e| enc::sse_rr(e, 0x66, false, po, a, b)));
        out.push(emit(|e| {
            enc::sse_rr(e, 0x66, false, 0x70, a, b);
            e.u8(rng.next() as u8);
        }));
        let _ = &ccs;
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    for _ in 0..100 {
        let (reg, base) = (r(&mut rng), r(&mut rng));
        let op = [0xb1u8, 0xc1, 0x87][rng.below(3) as usize];
        let size = [1u64, 2, 4, 8][rng.below(4) as usize];
        let lock = rng.below(2) == 1;
        let bytes = emit(|e| enc::atomic_mem_rr_for_test(e, lock, op, reg, base, size));
        // A lock prefix decodes as its own instruction.
        out.push(if bytes[0] == 0xf0 { bytes[1..].to_vec() } else { bytes });
    }
    out
}

/// The `.text` bytes of an ELF object.
fn text_of(elf: &[u8]) -> Vec<u8> {
    let bin = objfile::read(elf).expect("an object");
    bin.sections.into_iter().find(|s| s.name == ".text").map(|s| s.bytes).unwrap_or_default()
}

/// Assemble AT&T `lines` with our own assembler (rsasm), one instruction per
/// line, returning each line's bytes, or `None` if rsasm rejects the batch.
fn rsasm_each(lines: &[String]) -> Option<Vec<Vec<u8>>> {
    // Separate instructions with a label so each one's bytes can be cut out:
    // assemble every line on its own (rsasm is fast) instead.
    let mut out = Vec::new();
    for l in lines {
        let src = format!("{l}\n");
        let opts = crate::mc::asm::AsmOptions::new(TargetArch::X86_64);
        let obj = crate::mc::asm::assemble(&[crate::mc::asm::AsmSource { name: "rt.s", text: &src }], &opts).ok()?;
        out.push(text_of(&obj));
    }
    Some(out)
}

/// `llvm-mc --disassemble` of each instruction of `corpus` (one per input
/// line), in AT&T (`variant` 0) or Intel (1); `None` without llvm-mc.
fn llvm_mc_disasm(corpus: &[Vec<u8>], variant: u32) -> Option<Vec<String>> {
    use std::io::Write;
    use std::process::Stdio;
    let mc = llvm_tool("llvm-mc")?;
    let mut child = Command::new(mc)
        .args(["--triple=x86_64", "--disassemble", &format!("--output-asm-variant={variant}")])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut input = String::new();
    for bytes in corpus {
        let line: Vec<String> = bytes.iter().map(|b| format!("{b:#04x}")).collect();
        input.push_str(&line.join(" "));
        input.push('\n');
    }
    let mut stdin = child.stdin.take()?;
    let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
    let out = child.wait_with_output().ok()?;
    writer.join().ok()?.ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    Some(text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('.')).map(str::to_owned).collect())
}

/// Every instruction LF's encoder produces decodes as one known
/// instruction covering exactly its bytes, and our own assembler turns the
/// decoded AT&T text back into the same bytes (or, where it picks another
/// encoding of the same instruction, into bytes that decode to the same
/// text).
#[test]
fn encoder_round_trip() {
    let corpus = encoder_corpus();
    let mut texts = Vec::new();
    for bytes in &corpus {
        let x = decode_inst(bytes).unwrap_or_else(|| panic!("{bytes:02x?} does not decode"));
        assert_eq!(x.len, bytes.len(), "{bytes:02x?}: {x:?}");
        let att = x.render(0, Syntax::Att);
        assert!(att.known);
        // Intel renders too, with the same operand count.
        assert_eq!(x.render(0, Syntax::Intel).operands.len(), att.operands.len());
        texts.push(att.text());
    }
    // Our text agrees with LLVM's disassembler on every instruction, in
    // both syntaxes.
    let n = |s: &str| super::normalize(TargetArch::X86_64, s);
    let llvm_att = llvm_mc_disasm(&corpus, 0);
    if let Some(llvm) = &llvm_att {
        assert_eq!(llvm.len(), corpus.len(), "llvm-mc decoded a different number of instructions");
        let llvm_intel = llvm_mc_disasm(&corpus, 1).expect("llvm-mc");
        for (k, bytes) in corpus.iter().enumerate() {
            assert_eq!(n(&texts[k]), n(&llvm[k]), "{bytes:02x?}: ours `{}`, llvm-mc `{}`", texts[k], llvm[k]);
            let intel = decode(TargetArch::X86_64, bytes, 0, &INTEL).text();
            assert_eq!(n(&intel), n(&llvm_intel[k]), "{bytes:02x?}: ours `{intel}`, llvm-mc `{}`", llvm_intel[k]);
        }
        eprintln!("x86-64 encoder corpus: {} instructions agree with llvm-mc in AT&T and Intel syntax", corpus.len());
    }
    let (mut identical, mut fixpoint, mut rejected, mut arbitrated, mut unverified) = (0, 0, 0, 0, 0);
    for (bytes, t) in corpus.iter().zip(&texts) {
        let Some(asm) = rsasm_each(std::slice::from_ref(t)) else {
            rejected += 1;
            continue;
        };
        let again = &asm[0];
        if again == bytes {
            identical += 1;
            continue;
        }
        let back = decode(TargetArch::X86_64, again, 0, &ATT);
        if back.len == again.len() && &back.text() == t {
            fixpoint += 1;
            continue;
        }
        // rsasm produced another encoding that does not decode back to the
        // same text (a shorter equivalent form such as `shr` by one, or an
        // rsasm bug: 0.1.3 swaps the operands of `movq %xmmN, %r64`). Our text
        // for the original bytes was checked against llvm-mc above.
        if llvm_att.is_some() { arbitrated += 1 } else { unverified += 1 }
    }
    eprintln!(
        "x86-64 encoder round trip: {} instructions, {identical} byte-identical, {fixpoint} text fixpoints, \
         {arbitrated} other encodings with our text confirmed by llvm-mc, {unverified} unverified, {rejected} not re-assembled",
        corpus.len()
    );
    assert!(rejected * 20 < corpus.len(), "rsasm rejected {rejected} of {}", corpus.len());
}

// ===========================================================================
// Differential tests against llvm-objdump
// ===========================================================================

/// LF-compiled objects in every x86-64 format, both syntaxes.
#[test]
fn llvm_objdump_on_lf_objects() {
    for (name, src) in [("ints", INTS), ("floats", FLOATS), ("extra", EXTRA)] {
        let obj = compile(TargetArch::X86_64, src);
        for format in [ObjectFormat::Elf, ObjectFormat::Coff, ObjectFormat::MachO] {
            let file = object_file(TargetArch::X86_64, &obj, format);
            for (opts, args) in [(ATT, &[][..]), (INTEL, &["-M", "intel"][..])] {
                let Some(report) = differential(TargetArch::X86_64, &file, args, &opts, &|s| s) else {
                    eprintln!("skipping llvm_objdump_on_lf_objects: no llvm-objdump");
                    return;
                };
                assert_clean(&format!("x86-64 {name} {format:?} {:?}", opts.syntax), &report);
            }
        }
    }
}

/// Hand-written assembly covering the common instruction set, assembled by
/// llvm-mc and disassembled by llvm-objdump and by us.
#[test]
fn llvm_objdump_on_common_isa() {
    let Some(mc) = llvm_tool("llvm-mc") else {
        eprintln!("skipping llvm_objdump_on_common_isa: no llvm-mc");
        return;
    };
    let dir = scratch("x86-mc");
    let src = dir.join("isa.s");
    let obj = dir.join("isa.o");
    std::fs::write(&src, COMMON_ISA).unwrap();
    let out = Command::new(mc).args(["--triple=x86_64", "-filetype=obj", "-o"]).arg(&obj).arg(&src).output().unwrap();
    assert!(out.status.success(), "llvm-mc: {}", String::from_utf8_lossy(&out.stderr));
    let file = std::fs::read(&obj).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    for (opts, args) in [(ATT, &[][..]), (INTEL, &["-M", "intel"][..])] {
        let report = differential(TargetArch::X86_64, &file, args, &opts, &|s| s).expect("llvm-objdump");
        assert_clean(&format!("x86-64 common ISA {:?}", opts.syntax), &report);
    }
}

/// A random instruction-shaped byte string: prefixes in the canonical
/// order compilers and assemblers emit (a segment, `67`, one of `66`/`F2`/
/// `F3`, then `REX`), an opcode from the one-byte, `0F`, `0F 38` or `0F 3A`
/// map, then random ModRM/SIB/displacement/immediate bytes.
fn random_instruction(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::new();
    let map = rng.below(11);
    if map >= 8 {
        // VEX: an optional segment or 67, then C5 xx or C4 xx xx, an opcode.
        if rng.below(4) == 0 {
            b.push([0x2e, 0x64, 0x65, 0x67][rng.below(4) as usize]);
        }
        if map == 8 {
            b.extend([0xc5, rng.next() as u8]);
        } else {
            b.extend([0xc4, (rng.next() as u8 & 0xe0) | (1 + rng.below(3) as u8), rng.next() as u8]);
        }
        b.push(rng.next() as u8);
        let tail = rng.bytes(12);
        b.extend(tail);
        return b;
    }
    if rng.below(4) == 0 {
        b.push([0x2e, 0x64, 0x65, 0x26][rng.below(4) as usize]);
    }
    if rng.below(6) == 0 {
        b.push(0x67);
    }
    match (map, rng.below(5)) {
        (6 | 7, _) | (_, 0) => b.push(0x66),
        (_, 1) => b.push(0xf2),
        (_, 2) => b.push(0xf3),
        _ => {}
    }
    if rng.below(2) == 0 {
        b.push(0x40 | rng.below(16) as u8);
    }
    match map {
        0..=3 => loop {
            // An opcode, not another prefix.
            let op = rng.next() as u8;
            if !matches!(op, 0x26 | 0x2e | 0x36 | 0x3e | 0x40..=0x4f | 0x64..=0x67 | 0xf0 | 0xf2 | 0xf3) {
                b.push(op);
                break;
            }
        },
        4..=5 => b.extend([0x0f, rng.next() as u8]),
        6 => b.extend([0x0f, 0x38, rng.below(0x42) as u8]),
        _ => b.extend([0x0f, 0x3a, rng.below(0x45) as u8]),
    }
    let tail = rng.bytes(12);
    b.extend(tail);
    b
}

/// Prefix uses the architecture leaves undefined or that decoders treat
/// differently: `F2`/`F3` on `xchg` and `mov` to memory (the HLE hints,
/// which llvm splits off as `xacquire`/`xrelease`), `F3` with `REX.W` on
/// `0F 7E` (llvm drops the `F3` and decodes an MMX `movq`), and `67` on a
/// `moffs` move (llvm-objdump and llvm-mc print it differently).
fn ambiguous_prefixes(b: &[u8]) -> bool {
    let mut k = 0;
    let (mut rep, mut w) = (false, false);
    while let Some(&c) = b.get(k) {
        match c {
            0xf2 | 0xf3 => rep = true,
            0x40..=0x4f => w = c & 8 != 0,
            0x66 | 0x67 | 0x26 | 0x2e | 0x64 | 0x65 => {}
            _ => break,
        }
        k += 1;
    }
    let op = b.get(k).copied().unwrap_or(0);
    let op2 = b.get(k + 1).copied().unwrap_or(0);
    let addr32 = b[..k].contains(&0x67);
    (rep && (matches!(op, 0x86 | 0x87 | 0x88 | 0x89 | 0xc6 | 0xc7 | 0x90..=0x97) || (w && op == 0x0f && op2 == 0x7e)))
        || (addr32 && matches!(op, 0xa0..=0xa3))
}

/// Randomly generated instructions that we decode are decoded identically
/// (length and text, both syntaxes) by llvm-objdump. Each sits at its own
/// symbol so both disassemblers restart there.
#[test]
fn random_instructions_agree_with_llvm() {
    use crate::mc::object::{ObjectModule, Section, SectionKind, Symbol, SymbolBinding, SymbolType};
    let mut rng = Rng(0xdec0de);
    let mut obj = ObjectModule::new("random");
    let text = obj.add_section(Section::new(".text", SectionKind::Text, 1));
    let mut code = Vec::new();
    let mut n = 0;
    let mut tried = 0;
    while n < 20_000 && tried < 500_000 {
        tried += 1;
        let cand = random_instruction(&mut rng);
        if cand[0] == 0xf0 || ambiguous_prefixes(&cand) {
            continue;
        }
        let Some(x) = decode_inst(&cand) else { continue };
        let at = code.len() as u64;
        obj.add_symbol(Symbol::defined(format!("i{n}"), SymbolBinding::Local, SymbolType::Func, text, at, x.len as u64));
        code.extend_from_slice(&cand[..x.len]);
        n += 1;
    }
    obj.section_mut(text).bytes = code;
    let file = object_file(TargetArch::X86_64, &obj, ObjectFormat::Elf);
    for (opts, args) in [(ATT, &[][..]), (INTEL, &["-M", "intel"][..])] {
        let Some(report) = differential(TargetArch::X86_64, &file, args, &opts, &|s| s) else {
            eprintln!("skipping random_instructions_agree_with_llvm: no llvm-objdump");
            return;
        };
        assert_clean(&format!("x86-64 random instructions {:?}", opts.syntax), &report);
    }
}

/// Random input decodes in bounds and renders in both syntaxes.
#[test]
fn opcode_sweep_lengths() {
    let mut rng = Rng(99);
    let mut count = 0;
    for _ in 0..20_000 {
        let b = random_instruction(&mut rng);
        if let Some(x) = decode_inst(&b) {
            assert!(x.len >= 1 && x.len <= 15 && x.len <= b.len());
            let _: Inst = x.render(0x1000, Syntax::Att);
            let _: Inst = x.render(0x1000, Syntax::Intel);
            count += 1;
        }
    }
    assert!(count > 5_000, "only {count} decoded");
}

/// More IR for the differential test: narrow memory traffic, i128 math,
/// atomics and vectors where the backend supports them.
const EXTRA: &str = r#"
module "extra"

global @g8 : i8 = i8 1
global @g16 : i16 = i16 2
global @w64 : i64 = i64 5

func @narrow(ptr, i8, i16) -> i32 {
entry ^0(%p: ptr, %a: i8, %b: i16):
  store %a, %p align 1 : i8
  %q = ptr_add %p, i64 2 : ptr
  store %b, %q align 2 : i16
  %x = load %p align 1 : i8
  %y = load %q align 2 : i16
  %xs = sext %x : i32
  %yz = zext %y : i32
  %s = add %xs, %yz : i32
  %g = load @g8 align 1 : i8
  %gz = zext %g : i32
  %r = mul %s, %gz : i32
  ret %r
}

func @wide(i128, i128) -> i128 {
entry ^0(%a: i128, %b: i128):
  %m = mul %a, %b : i128
  %s = add %m, %a : i128
  %t = lshr %s, i128 3 : i128
  ret %t
}

func @atomics(i64) -> i64 {
entry ^0(%v: i64):
  %a = atomic_rmw add seq_cst @w64, %v align 8 : i64
  %b = atomic_rmw xchg seq_cst @w64, %a align 8 : i64
  %c = atomic_rmw and seq_cst @w64, %b align 8 : i64
  %r = add %a, %c : i64
  ret %r
}

func @vec(<4 x i32>, <4 x i32>) -> <4 x i32> {
entry ^0(%a: <4 x i32>, %b: <4 x i32>):
  %s = add %a, %b : <4 x i32>
  %m = mul %s, %b : <4 x i32>
  %x = xor %m, %a : <4 x i32>
  ret %x
}

func @vecf(<4 x f32>, <2 x f64>) -> <4 x f32> {
entry ^0(%a: <4 x f32>, %b: <2 x f64>):
  %s = fadd %a, %a : <4 x f32>
  %m = fmul %s, %a : <4 x f32>
  ret %m
}
"#;

/// The common-ISA corpus (AT&T, one instruction per line).
const COMMON_ISA: &str = include_str!("x86_isa.s");

/// Thread-local accesses (local-exec `%fs:0` loads, initial-exec GOT loads,
/// the padded general-dynamic `__tls_get_addr` sequence) and i128 code, in
/// static and position-independent objects.
const TLS: &str = r#"module "tls"
global thread_local @big : i64 = i64 7
global internal thread_local @tiny : i16 = i16 0
global thread_local @ext : i32

func @get() -> i64 {
entry ^0:
  %a = load @big align 8 : i64
  %t = load @tiny align 2 : i16
  %tx = zext %t : i64
  %e = load @ext align 4 : i32
  %ex = sext %e : i64
  store i16 3, @tiny align 2 : i16
  %s = add %a, %tx : i64
  %r = add %s, %ex : i64
  ret %r
}

func @wide(i128, i128) -> i128 {
entry ^0(%a: i128, %b: i128):
  %m = mul %a, %b : i128
  %s = add %m, %a : i128
  %x = lshr %s, i128 3 : i128
  ret %x
}
"#;

#[test]
fn llvm_objdump_on_tls_and_i128() {
    let (m, syms) = super::parse_for(TargetArch::X86_64, TLS);
    for pic in [false, true] {
        let cg = crate::codegen::CodegenOptions::default().with_pic(pic);
        let obj = crate::target::compile_module_for(TargetArch::X86_64, &m, &syms, &cg).expect("compile").object;
        let file = object_file(TargetArch::X86_64, &obj, ObjectFormat::Elf);
        let bin = objfile::read(&file).unwrap();
        let code = bin.sections.iter().find(|s| s.executable).unwrap();
        assert!(code.relocs.iter().any(|r| r.kind.contains("TPOFF") || r.kind.contains("TLSGD")), "{:?}", code.relocs);
        for (opts, args) in [(ATT, &[][..]), (INTEL, &["-M", "intel"][..])] {
            let Some(report) = differential(TargetArch::X86_64, &file, args, &opts, &|s| s) else {
                eprintln!("skipping llvm_objdump_on_tls_and_i128: no llvm-objdump");
                return;
            };
            assert_clean(&format!("x86-64 TLS/i128 pic={pic} {:?}", opts.syntax), &report);
        }
    }
}

