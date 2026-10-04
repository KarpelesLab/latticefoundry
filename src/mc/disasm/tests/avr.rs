//! AVR decoder tests: round trips against the encoder, and llvm-objdump
//! differential tests.

use super::corpus::{FLOATS, INTS};
use super::{Rng, assert_clean, compile, differential, llvm_tool, object_file, scratch};
use crate::mc::disasm::avr::{AvrInst, Op, Ptr, decode_inst, encode_inst, render};
use crate::mc::disasm::{Options, decode};
use crate::mc::object::{ObjectModule, Section, SectionKind};
use crate::target::avr::encode as enc;
use crate::target::{ObjectFormat, TargetArch};

/// Decode `bytes` at `addr` as text.
fn text(bytes: &[u8], addr: u64) -> String {
    decode(TargetArch::Avr, bytes, addr, &Options::default()).text()
}

fn w(word: u16) -> [u8; 2] {
    word.to_le_bytes()
}

#[test]
fn golden_texts() {
    // From the AVR Instruction Set Manual's encodings; the texts are what
    // llvm-objdump --mcpu=atmega328p prints.
    let cases: &[(&[u8], &str)] = &[
        (&[0x12, 0x0c], "add\tr1, r2"),
        (&[0x33, 0x0c], "lsl\tr3"),
        (&[0x44, 0x1c], "rol\tr4"),
        (&[0x01, 0x96], "adiw\tr24, 0x1"),
        (&[0xff, 0x96], "adiw\tr30, 0x3f"),
        (&[0x1f, 0x4f], "sbci\tr17, 0xff"),
        (&[0x13, 0x97], "sbiw\tr26, 0x3"),
        (&[0x11, 0x20], "tst\tr1"),
        (&[0x11, 0x24], "clr\tr1"),
        (&[0x8a, 0x94], "dec\tr8"),
        (&[0x01, 0x02], "muls\tr16, r17"),
        (&[0x89, 0x03], "fmulsu\tr16, r17"),
        (&[0xff, 0xcf], "rjmp\t.-2"),
        (&[0x01, 0xc0], "rjmp\t.+2"),
        (&[0x09, 0x94], "ijmp"),
        (&[0x0c, 0x94, 0x1a, 0x09], "jmp\t0x1234"),
        (&[0xff, 0x94, 0xff, 0xff], "call\t0x3ffffe"),
        (&[0x08, 0x95], "ret"),
        (&[0x13, 0xfc], "sbrc\tr1, 0x3"),
        (&[0xfa, 0x99], "sbic\t0x1f, 0x2"),
        (&[0xfb, 0xf3], "brvs\t.-2"),
        (&[0xf8, 0xf3], "brlo\t.-2"),
        (&[0xf8, 0xf7], "brsh\t.-2"),
        (&[0xfc, 0xf7], "brge\t.-2"),
        (&[0xcb, 0x01], "movw\tr24, r22"),
        (&[0x80, 0x91, 0x00, 0x01], "lds\tr24, 0x100"),
        (&[0x1e, 0x90], "ld\tr1, -X"),
        (&[0x18, 0x80], "ldd\tr1, Y+0"),
        (&[0x17, 0xac], "ldd\tr1, Z+63"),
        (&[0x1d, 0x92], "st\tX+, r1"),
        (&[0x19, 0x82], "std\tY+1, r1"),
        (&[0x30, 0x92, 0x00, 0x02], "sts\t0x200, r3"),
        (&[0xc8, 0x95], "lpm"),
        (&[0x15, 0x90], "lpm\tr1, Z+"),
        (&[0x17, 0x90], "elpm\tr1, Z+"),
        (&[0xcd, 0xb7], "in\tr28, 0x3d"),
        (&[0xde, 0xbf], "out\t0x3e, r29"),
        (&[0x1f, 0x92], "push\tr1"),
        (&[0x2f, 0x90], "pop\tr2"),
        (&[0x38, 0x94], "sev"),
        (&[0xc8, 0x94], "cls"),
        (&[0xf8, 0x94], "cli"),
        (&[0xfb, 0x9a], "sbi\t0x1f, 0x3"),
        (&[0x12, 0xfa], "bst\tr1, 0x2"),
        (&[0x98, 0x95], "break"),
        (&[0x00, 0x00], "nop"),
        (&[0x0f, 0xef], "ldi\tr16, 0xff"),
        (&[0x54, 0x92], "xch\tZ, r5"),
        (&[0x5b, 0x94], "des\t0x5"),
    ];
    for (bytes, want) in cases {
        assert_eq!(text(bytes, 0), *want, "{bytes:02x?}");
    }
    // Reserved encodings and truncated two-word instructions are data.
    assert_eq!(text(&[0x01, 0x00], 0), ".short\t0x0001");
    assert_eq!(text(&[0x08, 0xff], 0), ".short\t0xff08");
    assert_eq!(text(&[0x08, 0x92], 0), ".short\t0x9208");
    assert_eq!(text(&[0x0c, 0x94], 0), ".short\t0x940c");
    assert_eq!(text(&[0x0c], 0), ".byte\t0x0c");
}

#[test]
fn branch_targets() {
    let i = decode(TargetArch::Avr, &w(enc::rjmp(-3)), 0x100, &Options::default());
    assert_eq!((i.text(), i.target, i.target_operand), ("rjmp\t.-6".to_owned(), Some(0xfc), Some(0)));
    let i = decode(TargetArch::Avr, &w(enc::brbs(1, 5)), 0x10, &Options::default());
    assert_eq!((i.text(), i.target), ("breq\t.+10".to_owned(), Some(0x1c)));
    let c = enc::call(0x1234);
    let bytes = [w(c[0]), w(c[1])].concat();
    let i = decode(TargetArch::Avr, &bytes, 0, &Options::default());
    assert_eq!((i.text(), i.target, i.len), ("call\t0x2468".to_owned(), Some(0x2468), 4));
}

/// A random register in `lo..32`.
fn reg(rng: &mut Rng, lo: u8) -> u8 {
    lo + rng.below(u64::from(32 - lo)) as u8
}

/// Every helper of LF's AVR encoder, with random operands, decodes to the
/// typed instruction it encodes, renders as expected, and re-encodes to the
/// same words.
#[test]
fn encoder_round_trip() {
    let mut rng = Rng(0xa5a5_1234_5678_9abc);
    let mut count = 0;
    type Two = fn(u8, u8) -> u16;
    let two: [(Two, Op, &str); 11] = [
        (enc::add, Op::Add, "add"),
        (enc::adc, Op::Adc, "adc"),
        (enc::sub, Op::Sub, "sub"),
        (enc::sbc, Op::Sbc, "sbc"),
        (enc::and, Op::And, "and"),
        (enc::or, Op::Or, "or"),
        (enc::eor, Op::Eor, "eor"),
        (enc::mov, Op::Mov, "mov"),
        (enc::cp, Op::Cp, "cp"),
        (enc::cpc, Op::Cpc, "cpc"),
        (enc::mul, Op::Mul, "mul"),
    ];
    let imm: [(Two, Op, &str); 6] = [
        (enc::ldi, Op::Ldi, "ldi"),
        (enc::cpi, Op::Cpi, "cpi"),
        (enc::subi, Op::Subi, "subi"),
        (enc::sbci, Op::Sbci, "sbci"),
        (enc::andi, Op::Andi, "andi"),
        (enc::ori, Op::Ori, "ori"),
    ];
    type One = fn(u8) -> u16;
    let one: [(One, Op, &str); 9] = [
        (enc::com, Op::Com, "com"),
        (enc::dec, Op::Dec, "dec"),
        (enc::neg, Op::Neg, "neg"),
        (enc::swap, Op::Swap, "swap"),
        (enc::asr, Op::Asr, "asr"),
        (enc::lsr, Op::Lsr, "lsr"),
        (enc::ror, Op::Ror, "ror"),
        (enc::pop, Op::Pop, "pop"),
        (enc::lpm, Op::Lpm, "lpm"),
    ];
    let mut check = |words: &[u16], want: AvrInst, want_text: String| {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let got = decode_inst(&bytes).unwrap_or_else(|| panic!("{words:04x?} does not decode"));
        assert_eq!(got, want, "{words:04x?}");
        assert_eq!(render(&got, 0x200).text(), want_text, "{words:04x?}");
        let (a, b) = encode_inst(&got).expect("re-encodes");
        assert_eq!(a, words[0]);
        if got.len == 4 {
            assert_eq!(b, words[1]);
        }
        count += 1;
    };
    let mk = |op, d, r| AvrInst { d, r, ..AvrInst::new(op) };
    for _ in 0..500 {
        let (d, r) = (reg(&mut rng, 0), reg(&mut rng, 0));
        for (f, op, name) in two {
            let alias = match (op, d == r) {
                (Op::Add, true) => Some("lsl"),
                (Op::Adc, true) => Some("rol"),
                (Op::And, true) => Some("tst"),
                (Op::Eor, true) => Some("clr"),
                _ => None,
            };
            let t = alias.map_or_else(|| format!("{name}\tr{d}, r{r}"), |a| format!("{a}\tr{d}"));
            check(&[f(d, r)], mk(op, d, r), t);
        }
        check(&[enc::lsl(d)], mk(Op::Add, d, d), format!("lsl\tr{d}"));
        check(&[enc::rol(d)], mk(Op::Adc, d, d), format!("rol\tr{d}"));
        check(&[enc::tst(d)], mk(Op::And, d, d), format!("tst\tr{d}"));
        let (h, k) = (reg(&mut rng, 16), rng.next() as u8);
        for (f, op, name) in imm {
            check(&[f(h, k)], AvrInst { d: h, k: i32::from(k), ..AvrInst::new(op) }, format!("{name}\tr{h}, {k:#x}"));
        }
        for (f, op, name) in one {
            let t = if op == Op::Lpm { format!("lpm\tr{d}, Z") } else { format!("{name}\tr{d}") };
            let ptr = if op == Op::Lpm { Ptr::Z } else { Ptr::None };
            check(&[f(d)], AvrInst { d, ptr, ..AvrInst::new(op) }, t);
        }
        check(&[enc::lpm_inc(d)], AvrInst { d, ptr: Ptr::ZInc, ..AvrInst::new(Op::Lpm) }, format!("lpm\tr{d}, Z+"));
        check(&[enc::push(r)], mk(Op::Push, 0, r), format!("push\tr{r}"));
        check(&[enc::st_x_inc(r)], AvrInst { r, ptr: Ptr::XInc, ..AvrInst::new(Op::St) }, format!("st\tX+, r{r}"));
        let (pd, ps) = (rng.below(16) as u8 * 2, rng.below(16) as u8 * 2);
        check(&[enc::movw(pd, ps)], mk(Op::Movw, pd, ps), format!("movw\tr{pd}, r{ps}"));
        let (p, k6) = (24 + 2 * rng.below(4) as u8, rng.below(64) as u8);
        check(&[enc::adiw(p, k6)], AvrInst { d: p, k: i32::from(k6), ..AvrInst::new(Op::Adiw) }, format!("adiw\tr{p}, {k6:#x}"));
        check(&[enc::sbiw(p, k6)], AvrInst { d: p, k: i32::from(k6), ..AvrInst::new(Op::Sbiw) }, format!("sbiw\tr{p}, {k6:#x}"));
        let (y, q) = (rng.below(2) == 1, rng.below(64) as u8);
        let (pp, pn) = if y { (Ptr::Y, "Y") } else { (Ptr::Z, "Z") };
        check(&[enc::ldd(d, y, q)], AvrInst { d, k: i32::from(q), ptr: pp, ..AvrInst::new(Op::Ldd) }, format!("ldd\tr{d}, {pn}+{q}"));
        check(&[enc::std(y, q, r)], AvrInst { r, k: i32::from(q), ptr: pp, ..AvrInst::new(Op::Std) }, format!("std\t{pn}+{q}, r{r}"));
        let a = rng.below(64) as u8;
        check(&[enc::in_(d, a)], AvrInst { d, a, ..AvrInst::new(Op::In) }, format!("in\tr{d}, {a:#x}"));
        check(&[enc::out(a, r)], AvrInst { r, a, ..AvrInst::new(Op::Out) }, format!("out\t{a:#x}, r{r}"));
        let (s, bk) = (rng.below(8) as u8, rng.below(128) as i32 - 64);
        let rel = |words: i32| if words < 0 { format!(".-{}", -2 * words) } else { format!(".+{}", 2 * words) };
        const SET: [&str; 8] = ["brlo", "breq", "brmi", "brvs", "brlt", "brhs", "brts", "brie"];
        const CLR: [&str; 8] = ["brsh", "brne", "brpl", "brvc", "brge", "brhc", "brtc", "brid"];
        check(&[enc::brbs(s, bk)], AvrInst { k: bk, b: s, ..AvrInst::new(Op::Brbs) }, format!("{}\t{}", SET[usize::from(s)], rel(bk)));
        check(&[enc::brbc(s, bk)], AvrInst { k: bk, b: s, ..AvrInst::new(Op::Brbc) }, format!("{}\t{}", CLR[usize::from(s)], rel(bk)));
        let set = rng.below(2) == 1;
        let op = if set { Op::Brbs } else { Op::Brbc };
        let name = if set { SET[usize::from(s)] } else { CLR[usize::from(s)] };
        check(&[enc::br(s, set, bk)], AvrInst { k: bk, b: s, ..AvrInst::new(op) }, format!("{name}\t{}", rel(bk)));
        let jk = rng.below(4096) as i32 - 2048;
        check(&[enc::rjmp(jk)], AvrInst { k: jk, ..AvrInst::new(Op::Rjmp) }, format!("rjmp\t{}", rel(jk)));
        let far = rng.below(1 << 22) as u32;
        check(&enc::jmp(far), AvrInst { k: far as i32, len: 4, ..AvrInst::new(Op::Jmp) }, format!("jmp\t{:#x}", far * 2));
        check(&enc::call(far), AvrInst { k: far as i32, len: 4, ..AvrInst::new(Op::Call) }, format!("call\t{:#x}", far * 2));
        let b = rng.below(8) as u8;
        check(&[enc::sbrc(r, b)], AvrInst { r, b, ..AvrInst::new(Op::Sbrc) }, format!("sbrc\tr{r}, {b:#x}"));
        check(&[enc::bst(d, b)], AvrInst { d, b, ..AvrInst::new(Op::Bst) }, format!("bst\tr{d}, {b:#x}"));
        check(&[enc::bld(d, b)], AvrInst { d, b, ..AvrInst::new(Op::Bld) }, format!("bld\tr{d}, {b:#x}"));
    }
    for (word, t) in [(enc::ICALL, "icall"), (enc::RET, "ret"), (enc::CLI, "cli"), (enc::BREAK, "break")] {
        let i = decode_inst(&w(word)).unwrap();
        check(&[word], i, t.to_owned());
    }
    eprintln!("avr encoder round trip: {count} instructions");
}

/// Every 16-bit word: decoding never panics, and every word that decodes
/// re-encodes to itself (the decoder ignores no bit). Also counts the
/// defined encodings.
#[test]
fn every_word_round_trips() {
    let mut known = 0;
    for word in 0..=u16::MAX {
        let mut bytes = word.to_le_bytes().to_vec();
        bytes.extend_from_slice(&0xbeefu16.to_le_bytes());
        let Some(i) = decode_inst(&bytes) else {
            assert!(!decode(TargetArch::Avr, &bytes, 0, &Options::default()).known);
            continue;
        };
        known += 1;
        let (a, b) = encode_inst(&i).unwrap_or_else(|| panic!("{word:04x} ({i:?}) does not re-encode"));
        assert_eq!(a, word, "{i:?}");
        if i.len == 4 {
            assert_eq!(b, 0xbeef);
        }
        // Lone first words of two-word instructions are not instructions.
        if i.len == 4 {
            assert!(decode_inst(&word.to_le_bytes()).is_none());
        }
    }
    eprintln!("avr: {known} of 65536 words are defined encodings");
    // The reserved slots of the manual's opcode map.
    for word in [0x0001, 0x00ff, 0x9003, 0x9008, 0x900b, 0x9203, 0x9208, 0x920b, 0x9404, 0x9528, 0x9409 | 0x20, 0x95b8, 0xf808, 0xff0f] {
        assert!(decode_inst(&w(word)).is_none(), "{word:04x} is reserved");
    }
    assert!(known > 60_000, "{known}");
}

/// An AVR ELF object holding `bytes` as `.text`.
fn elf_of(bytes: Vec<u8>) -> Vec<u8> {
    let mut obj = ObjectModule::new("words");
    let s = obj.add_section(Section::new(".text", SectionKind::Text, 2));
    obj.section_mut(s).bytes = bytes;
    crate::target::avr::write_elf(&obj).expect("AVR ELF")
}

/// Every defined encoding the ATmega2560 has (the AVR5 set plus `eijmp`,
/// `eicall`, `elpm`; the XMEGA-only `xch`/`las`/`lac`/`lat`/`des`/`spm Z+`
/// are left out), each once, decoded by llvm-objdump and by us.
#[test]
fn objdump_every_encoding() {
    let mut rng = Rng(99);
    let mut bytes = Vec::new();
    for word in 0..=u16::MAX {
        let second = rng.next() as u16;
        let mut b = word.to_le_bytes().to_vec();
        b.extend_from_slice(&second.to_le_bytes());
        let Some(i) = decode_inst(&b) else { continue };
        if matches!(i.op, Op::Xch | Op::Las | Op::Lac | Op::Lat | Op::Des) || (i.op == Op::Spm && i.ptr == Ptr::ZInc) {
            continue;
        }
        bytes.extend_from_slice(&b[..usize::from(i.len)]);
    }
    let file = elf_of(bytes);
    let Some(report) = differential(TargetArch::Avr, &file, &["--mcpu=atmega2560"], &Options::default(), &|s| s) else {
        eprintln!("skipping objdump_every_encoding: no llvm-objdump");
        return;
    };
    assert_clean("avr every encoding", &report);
}

/// Objects LF compiles for AVR (calls are relocated `call`s; soft-float
/// helpers for the floating-point corpus).
#[test]
fn objdump_compiled_objects() {
    for (name, src) in [("ints", INTS), ("floats", FLOATS)] {
        let obj = compile(TargetArch::Avr, src);
        let file = object_file(TargetArch::Avr, &obj, ObjectFormat::Elf);
        let Some(report) = differential(TargetArch::Avr, &file, &["--mcpu=atmega328p"], &Options::default(), &|s| s) else {
            eprintln!("skipping objdump_compiled_objects: no llvm-objdump");
            return;
        };
        assert_clean(&format!("avr {name}"), &report);
    }
}

/// Hand-written assembly covering the instruction set (with every alias
/// spelling), assembled by llvm-mc and decoded by both disassemblers.
#[test]
fn llvm_mc_assembly() {
    let Some(mc) = llvm_tool("llvm-mc") else {
        eprintln!("skipping llvm_mc_assembly: no llvm-mc");
        return;
    };
    let mut src = String::from(HAND);
    // Plus random register/immediate forms.
    let mut rng = Rng(5);
    for _ in 0..300 {
        let (d, r, h) = (reg(&mut rng, 0), reg(&mut rng, 0), reg(&mut rng, 16));
        let k = rng.next() as u8;
        let a = rng.below(64);
        let q = rng.below(64);
        let b = rng.below(8);
        let p = 24 + 2 * rng.below(4);
        src.push_str(&format!(
            "sub r{d}, r{r}\nsbci r{h}, {k}\nldd r{d}, Y+{q}\nstd Z+{q}, r{r}\nin r{d}, {a}\nout {a}, r{r}\nsbrs r{d}, {b}\nbld r{r}, {b}\nadiw r{p}, {q}\nlds r{d}, {}\nsts {}, r{r}\n",
            rng.below(65536),
            rng.below(65536)
        ));
    }
    let dir = scratch("avr-mc");
    let (s, o) = (dir.join("in.s"), dir.join("in.o"));
    std::fs::write(&s, &src).unwrap();
    let out = std::process::Command::new(mc)
        .args(["--triple=avr", "-mcpu=atmega2560", "-filetype=obj", "-o"])
        .arg(&o)
        .arg(&s)
        .output()
        .expect("run llvm-mc");
    assert!(out.status.success(), "llvm-mc: {}", String::from_utf8_lossy(&out.stderr));
    let file = std::fs::read(&o).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let report = differential(TargetArch::Avr, &file, &["--mcpu=atmega2560"], &Options::default(), &|s| s)
        .expect("llvm-objdump next to llvm-mc");
    assert_clean("avr llvm-mc corpus", &report);
}

const HAND: &str = "
add r1, r2
add r3, r3
adc r4, r4
adc r4, r5
adiw r24, 1
adiw r30, 63
sub r1, r2
subi r16, 5
sbc r1, r2
sbci r17, 255
sbiw r26, 3
and r1, r2
and r1, r1
andi r16, 15
or r1, r2
ori r16, 3
eor r1, r2
eor r1, r1
com r5
neg r6
inc r7
dec r8
ser r16
mul r1, r2
muls r16, r17
mulsu r16, r17
fmul r16, r17
fmuls r16, r17
fmulsu r16, r17
ijmp
eijmp
jmp 0x1234
icall
eicall
call 0x100
ret
reti
cpse r1, r2
cp r1, r2
cpc r1, r2
cpi r16, 0x10
sbrc r1, 3
sbrs r1, 4
sbic 0x1f, 2
sbis 0x10, 7
mov r1, r2
movw r24, r22
ldi r24, 5
lds r24, 0x100
ld r1, X
ld r1, X+
ld r1, -X
ld r1, Y
ld r1, Y+
ld r1, -Y
ldd r1, Y+5
ld r1, Z
ld r1, Z+
ld r1, -Z
ldd r1, Z+63
st X, r1
st X+, r1
st -X, r1
st Y, r1
st Y+, r1
st -Y, r1
std Y+1, r1
st Z, r1
st Z+, r1
st -Z, r1
std Z+10, r1
sts 0x200, r3
lpm
lpm r1, Z
lpm r1, Z+
elpm
elpm r1, Z
elpm r1, Z+
spm
in r28, 0x3d
out 0x3e, r29
push r1
pop r2
lsl r1
lsr r1
rol r1
ror r1
asr r1
swap r1
bset 3
bclr 4
sbi 0x1f, 3
cbi 0x1f, 3
bst r1, 2
bld r1, 3
sec
clc
sen
cln
sez
clz
sei
cli
ses
cls
sev
clv
set
clt
seh
clh
break
nop
sleep
wdr
tst r5
clr r6
sbr r16, 3
cbr r16, 3
rjmp .+4
rcall .-8
breq .+2
brne .-2
brcs .+6
brcc .+6
brmi .+2
brpl .+2
brge .+2
brlt .+2
brhs .+2
brhc .+2
brts .+2
brtc .+2
brvs .+2
brvc .+2
brie .+2
brid .+2
brbs 3, .+2
brbc 5, .-4
";
