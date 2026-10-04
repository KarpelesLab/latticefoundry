//! Thumb-2 decoder tests: round trips against the encoder, golden texts
//! (including IT blocks), and llvm-objdump differential tests over objects
//! LF compiles and over a broad hand-written instruction corpus assembled by
//! llvm-mc.

use std::process::Command;

use super::corpus::{FLOATS, INTS};
use super::{Rng, assert_clean, compile, differential, llvm_tool, normalize, object_file, scratch};
use crate::mc::disasm::{Options, State, disassemble, thumb};
use crate::target::thumb::encode::{self as enc, T};
use crate::target::{ObjectFormat, TargetArch};

const REGS: [&str; 16] = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "sp", "lr", "pc"];
const CONDS: [&str; 14] = ["eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le"];

fn r(n: u32) -> &'static str {
    REGS[n as usize]
}

/// Decode `t` at `addr` (outside any IT block) and compare with `want`
/// (whitespace and number spelling normalized).
fn check(t: T, addr: u64, want: &str) {
    let bytes = t.bytes();
    let inst = thumb::decode(&bytes, addr);
    assert!(inst.known, "{bytes:02x?}: unknown, want `{want}`");
    assert_eq!(inst.len, t.len(), "{bytes:02x?}: length, want `{want}`");
    assert_eq!(
        normalize(TargetArch::Thumb, &inst.text()),
        normalize(TargetArch::Thumb, want),
        "{bytes:02x?} decoded as `{}`",
        inst.text()
    );
}

fn list(mask: u32) -> String {
    let v: Vec<&str> = (0..16).filter(|k| mask >> k & 1 != 0).map(r).collect();
    format!("{{{}}}", v.join(", "))
}

/// A fuzzed corpus from the encoder's own builders: every builder, random
/// operands, decoded back and compared field by field (through the text).
#[test]
fn encoder_round_trip() {
    let mut rng = Rng(0x7468_756d_6232);
    let mut n = 0usize;
    for _ in 0..400 {
        let (d, m, k) = (rng.below(8) as u32, rng.below(8) as u32, rng.below(8) as u32);
        let imm5 = 1 + rng.below(31) as u32;
        let kind = rng.below(3) as u32;
        check(enc::shift_imm16(kind, d, m, imm5), 0, &format!("{}s {}, {}, #{imm5}", ["lsl", "lsr", "asr"][kind as usize], r(d), r(m)));
        let sub = rng.below(2) == 1;
        let op = if sub { "subs" } else { "adds" };
        check(enc::addsub_reg16(sub, d, m, k), 0, &format!("{op} {}, {}, {}", r(d), r(m), r(k)));
        check(enc::addsub_imm3(sub, d, m, k), 0, &format!("{op} {}, {}, #{k}", r(d), r(m)));
        let imm8 = rng.below(256) as u32;
        check(enc::movs_imm8(d, imm8), 0, &format!("movs {}, #{imm8}", r(d)));
        check(enc::cmp_imm8(d, imm8), 0, &format!("cmp {}, #{imm8}", r(d)));
        check(enc::addsub_imm8(sub, d, imm8), 0, &format!("{op} {}, #{imm8}", r(d)));
        let dp = [(0, "ands"), (1, "eors"), (2, "lsls"), (3, "lsrs"), (4, "asrs"), (8, "tst"), (10, "cmp"), (12, "orrs"), (14, "bics"), (15, "mvns")];
        let (code, name) = dp[rng.below(dp.len() as u64) as usize];
        check(enc::dp16(code, d, m), 0, &format!("{name} {}, {}", r(d), r(m)));
        check(enc::dp16(9, d, m), 0, &format!("rsbs {}, {}, #0", r(d), r(m)));
        check(enc::dp16(13, d, m), 0, &format!("muls {}, {}, {}", r(d), r(m), r(d)));
        let (hd, hm) = (rng.below(13) as u32, rng.below(13) as u32);
        check(enc::add_hi(hd, hm), 0, &format!("add {}, {}", r(hd), r(hm)));
        check(enc::cmp_hi(hd | 8, hm), 0, &format!("cmp {}, {}", r(hd | 8), r(hm)));
        check(enc::mov_reg16(hd, hm), 0, &format!("mov {}, {}", r(hd), r(hm)));
        check(enc::blx(hm), 0, &format!("blx {}", r(hm)));
        check(enc::bx(14), 0, "bx lr");
        let load = rng.below(2) == 1;
        let (size, name) = [(4, "ldr"), (2, "ldrh"), (1, "ldrb")][rng.below(3) as usize];
        let name = if load { name.to_owned() } else { name.replace("ld", "st") };
        let off = rng.below(32) as u32 * size;
        let mem = if off == 0 { format!("[{}]", r(m)) } else { format!("[{}, #{off}]", r(m)) };
        check(enc::ldst_imm16(load, size, d, m, off), 0, &format!("{name} {}, {mem}", r(d)));
        let off4 = rng.below(256) as u32 * 4;
        let spm = if off4 == 0 { "[sp]".to_owned() } else { format!("[sp, #{off4}]") };
        check(enc::ldst_sp16(load, d, off4), 0, &format!("{} {}, {spm}", if load { "ldr" } else { "str" }, r(d)));
        check(enc::add_rd_sp16(d, off4), 0, &format!("add {}, sp, #{off4}", r(d)));
        let off_sp = rng.below(128) as u32 * 4;
        check(enc::addsub_sp16(sub, off_sp), 0, &format!("{} sp, #{off_sp}", if sub { "sub" } else { "add" }));
        let ek = rng.below(4) as u32;
        check(enc::ext16(ek, d, m), 0, &format!("{} {}, {}", ["sxth", "sxtb", "uxth", "uxtb"][ek as usize], r(d), r(m)));
        let l8 = rng.below(256) as u32;
        let lr = rng.below(2) == 1;
        if l8 != 0 || lr {
            check(enc::push16(l8, lr), 0, &format!("push {}", list(l8 | u32::from(lr) << 14)));
            check(enc::pop16(l8, lr), 0, &format!("pop {}", list(l8 | u32::from(lr) << 15)));
        }
        check(enc::udf(imm8), 0, &format!("udf #{imm8}"));
        check(enc::svc(imm8), 0, &format!("svc #{imm8}"));
        let addr = rng.below(0x10000) * 2 + 0x1000;
        let c = rng.below(14) as u32;
        let boff = (rng.below(256) as i32 - 128) * 2;
        check(enc::bcond16(c, boff), addr, &format!("b{} {:#x}", CONDS[c as usize], (addr as i64 + 4 + i64::from(boff)) as u64));
        let boff = (rng.below(2048) as i32 - 1024) * 2;
        check(enc::b16(boff), addr, &format!("b {:#x}", (addr as i64 + 4 + i64::from(boff)) as u64));
        n += 27;
    }
    for _ in 0..400 {
        let (d, a, b, x) = (rng.below(13) as u32, rng.below(13) as u32, rng.below(13) as u32, rng.below(13) as u32);
        // Modified immediates: every representable shape.
        let v = match rng.below(4) {
            0 => rng.below(256) as u32,
            1 => (rng.below(255) as u32 + 1) * 0x0101_0101,
            2 => (0x80 | rng.below(128) as u32) << rng.below(24),
            _ => (rng.below(255) as u32 + 1) * 0x0001_0001,
        };
        let imm12 = enc::mod_imm(v).expect("representable");
        let s = rng.below(2) == 1;
        let sx = if s { "s" } else { "" };
        let ops = [(0, "and", ""), (1, "bic", ""), (2, "orr", ""), (3, "orn", ""), (4, "eor", ""), (8, "add", ".w"), (13, "sub", ".w"), (14, "rsb", ".w")];
        let (code, name, wq) = ops[rng.below(ops.len() as u64) as usize];
        // (`d` = 15 with `s` would be a test instruction; avoid it here.)
        check(enc::dp_modimm(code, s, d, a, imm12), 0, &format!("{name}{sx}{wq} {}, {}, #{v}", r(d), r(a)));
        check(enc::dp_modimm(2, s, d, 15, imm12), 0, &format!("mov{sx}.w {}, #{v}", r(d)));
        check(enc::dp_modimm(3, s, d, 15, imm12), 0, &format!("mvn{sx} {}, #{v}", r(d)));
        check(enc::dp_modimm(13, true, 15, a, imm12), 0, &format!("cmp.w {}, #{v}", r(a)));
        check(enc::dp_modimm(0, true, 15, a, imm12), 0, &format!("tst.w {}, #{v}", r(a)));
        let i12 = rng.below(4096) as u32;
        check(enc::dp_plainimm(0, d, a, i12), 0, &format!("addw {}, {}, #{i12}", r(d), r(a)));
        check(enc::dp_plainimm(10, d, a, i12), 0, &format!("subw {}, {}, #{i12}", r(d), r(a)));
        let i16 = rng.below(65536) as u32;
        check(enc::movw(d, i16), 0, &format!("movw {}, #{i16}", r(d)));
        check(enc::movt(d, i16), 0, &format!("movt {}, #{i16}", r(d)));
        let lsb = rng.below(32) as u32;
        let width = 1 + rng.below(u64::from(32 - lsb)) as u32;
        let signed = rng.below(2) == 1;
        check(enc::bfx(signed, d, a, lsb, width), 0, &format!("{} {}, {}, #{lsb}, #{width}", if signed { "sbfx" } else { "ubfx" }, r(d), r(a)));
        let ty = rng.below(3) as u32;
        let sh5 = rng.below(32) as u32;
        let shift = match (ty, sh5) {
            (0, 0) => String::new(),
            (0, n) => format!(", lsl #{n}"),
            (1, 0) => ", lsr #32".to_owned(),
            (1, n) => format!(", lsr #{n}"),
            (_, 0) => ", asr #32".to_owned(),
            (_, n) => format!(", asr #{n}"),
        };
        let rops = [(0, "and", ".w"), (1, "bic", ".w"), (2, "orr", ".w"), (3, "orn", ""), (4, "eor", ".w"), (8, "add", ".w"), (10, "adc", ".w"), (11, "sbc", ".w"), (13, "sub", ".w"), (14, "rsb", "")];
        let (code, name, wq) = rops[rng.below(rops.len() as u64) as usize];
        check(enc::dp_reg(code, s, d, a, b, ty, sh5), 0, &format!("{name}{sx}{wq} {}, {}, {}{shift}", r(d), r(a), r(b)));
        check(enc::dp_reg(3, s, d, 15, b, ty, sh5), 0, &format!("mvn{sx}.w {}, {}{shift}", r(d), r(b)));
        let movsh = match (ty, sh5) {
            (0, 0) => format!("mov{sx}.w {}, {}", r(d), r(b)),
            (t, n) => format!("{}{sx}.w {}, {}, #{}", ["lsl", "lsr", "asr"][t as usize], r(d), r(b), if n == 0 { 32 } else { n }),
        };
        check(enc::dp_reg(2, s, d, 15, b, ty, sh5), 0, &movsh);
        check(enc::dp_reg(13, true, 15, a, b, ty, sh5), 0, &format!("cmp.w {}, {}{shift}", r(a), r(b)));
        check(enc::shift_reg32(ty, d, a, b), 0, &format!("{}.w {}, {}, {}", ["lsl", "lsr", "asr"][ty as usize], r(d), r(a), r(b)));
        let ek = [0, 1, 4, 5][rng.below(4) as usize];
        let en = match ek {
            0 => "sxth",
            1 => "uxth",
            4 => "sxtb",
            _ => "uxtb",
        };
        check(enc::ext32(ek, d, b), 0, &format!("{en}.w {}, {}", r(d), r(b)));
        check(enc::mul32(d, a, b), 0, &format!("mul {}, {}, {}", r(d), r(a), r(b)));
        check(enc::mls(d, a, b, x), 0, &format!("mls {}, {}, {}, {}", r(d), r(a), r(b), r(x)));
        check(enc::div(signed, d, a, b), 0, &format!("{} {}, {}, {}", if signed { "sdiv" } else { "udiv" }, r(d), r(a), r(b)));
        let load = rng.below(2) == 1;
        let (size, name) = [(4, "ldr"), (2, "ldrh"), (1, "ldrb")][rng.below(3) as usize];
        let name = if load { name.to_owned() } else { name.replace("ld", "st") };
        let i12 = rng.below(4096) as u32;
        let mem = if i12 == 0 { format!("[{}]", r(a)) } else { format!("[{}, #{i12}]", r(a)) };
        check(enc::ldst_imm12(load, size, d, a, i12), 0, &format!("{name}.w {}, {mem}", r(d)));
        let i8 = rng.below(256) as u32;
        check(enc::ldst_neg8(load, size, d, a, i8), 0, &format!("{name} {}, [{}, #-{i8}]", r(d), r(a)));
        let off = rng.below(256) as u32 * 4;
        let mem = if off == 0 { format!("[{}]", r(a)) } else { format!("[{}, #{off}]", r(a)) };
        check(enc::ldst_dual(load, d, x, a, off), 0, &format!("{} {}, {}, {mem}", if load { "ldrd" } else { "strd" }, r(d), r(x)));
        let lst = rng.below(0x1fff) as u32 | 0x4000 | 0x10;
        check(enc::push32(lst), 0, &format!("push.w {}", list(lst)));
        let lst = (lst & !0x4000) | 0x8000;
        check(enc::pop32(lst), 0, &format!("pop.w {}", list(lst)));
        let addr = 0x8000 + rng.below(0x1000) * 2;
        let off = (rng.below(1 << 23) as i32 - (1 << 22)) * 2;
        let link = rng.below(2) == 1;
        let target = (addr as i64 + 4 + i64::from(off)) as u64 & 0xffff_ffff;
        check(enc::b32(link, off), addr, &format!("{} {target:#x}", if link { "bl" } else { "b.w" }));
        let off = (rng.below(1 << 19) as i32 - (1 << 18)) * 2;
        let c = rng.below(14) as u32;
        let target = (addr as i64 + 4 + i64::from(off)) as u64 & 0xffff_ffff;
        check(enc::bcond32(c, off), addr, &format!("b{}.w {target:#x}", CONDS[c as usize]));
        n += 26;
    }
    check(enc::dmb_sy(), 0, "dmb sy");
    eprintln!("thumb encoder round trip: {n} instructions decoded back exactly");
}

/// IT blocks: the condition suffixes follow the mask, and 16-bit
/// flag-setting forms lose their `s` inside the block.
#[test]
fn it_blocks() {
    let mut code = Vec::new();
    for t in [
        enc::it(0, enc::ite_mask(0)),      // ite eq
        enc::movs_imm8(0, 1),              // moveq
        enc::movs_imm8(0, 0),              // movne
        enc::addsub_reg16(false, 1, 2, 3), // adds (outside the block)
        enc::it(11, 0b1111),               // itttt lt
        enc::addsub_imm8(true, 1, 4),
        enc::dp16(0, 1, 2),
        enc::mov_reg16(8, 1),
        enc::b32(true, 0x100),
        enc::it(1, 0b1100), // itt ne
        enc::dp_reg(8, false, 0, 1, 2, 0, 0),
        enc::addsub_reg16(false, 1, 2, 3),
    ] {
        code.extend(t.bytes());
    }
    let insts = disassemble(TargetArch::Thumb, &code, 0x100, &Options::default());
    let texts: Vec<String> = insts.iter().map(|(_, i)| normalize(TargetArch::Thumb, &i.text())).collect();
    let want = [
        "ite eq",
        "moveq r0,#1",
        "movne r0,#0",
        "adds r1,r2,r3",
        "itttt lt",
        "sublt r1,#4",
        "andlt r1,r2",
        "movlt r8,r1",
        "bllt 532",
        "itt ne",
        "addne.w r0,r1,r2",
        "addne r1,r2,r3",
    ];
    assert_eq!(texts, want);
    // The state is explicit, too.
    let mut st = State::default();
    let _ = thumb::decode_in(&enc::it(0, 0b1000).bytes(), 0, &mut st);
    assert_eq!(st.it, 0x08);
    let i = thumb::decode_in(&enc::movs_imm8(2, 3).bytes(), 2, &mut st);
    assert_eq!(i.text(), "moveq\tr2, #0x3");
    assert_eq!(st.it, 0);
}

#[test]
fn golden_texts() {
    let cases: [(&[u8], &str); 12] = [
        (&[0x2d, 0xe9, 0xf0, 0x43], "push.w\t{r4, r5, r6, r7, r8, r9, lr}"),
        (&[0x83, 0xb0], "sub\tsp, #0xc"),
        (&[0x00, 0x23], "movs\tr3, #0x0"),
        (&[0xcd, 0xf8, 0x00, 0x90], "str.w\tr9, [sp]"),
        (&[0x41, 0xea, 0x08, 0x03], "orr.w\tr3, r1, r8"),
        (&[0xbf, 0xf3, 0x5f, 0x8f], "dmb\tsy"),
        (&[0x70, 0x47], "bx\tlr"),
        (&[0xd0, 0xe8, 0x01, 0xf0], "tbb\t[r0, r1]"),
        (&[0x52, 0xe8, 0x00, 0x1f], "ldrex\tr1, [r2]"),
        (&[0xef, 0xf3, 0x10, 0x80], "mrs\tr0, primask"),
        (&[0x72, 0xb6], "cpsid\ti"),
        (&[0x02, 0x48], "ldr\tr0, [pc, #0x8]"),
    ];
    for (bytes, want) in cases {
        assert_eq!(thumb::decode(bytes, 0).text(), want, "{bytes:02x?}");
    }
    // Unknown encodings (here a VFP `vadd.f32`) are data.
    let vadd = thumb::decode(&[0x30, 0xee, 0x00, 0x0a], 0);
    assert!(!vadd.known && vadd.len == 4, "{vadd:?}");
    assert_eq!(vadd.text(), ".inst.w\t0xee300a00");
    // A truncated 32-bit instruction is a halfword of data.
    let cut = thumb::decode(&[0x2d, 0xe9], 0);
    assert_eq!((cut.len, cut.known), (2, false));
    // Branch targets are absolute and marked for symbolization.
    let bl = thumb::decode(&enc::b32(true, -8).bytes(), 0x1000);
    assert_eq!((bl.target, bl.target_operand), (Some(0xffc), Some(0)));
}

/// `llvm-objdump --triple=thumbv7m-none-eabi`.
const TRIPLE: &str = "--triple=thumbv7m-none-eabi";

/// LF-compiled objects disassemble exactly like llvm-objdump.
#[test]
fn differential_compiled() {
    let mut total = 0;
    for (name, src) in [("ints", INTS), ("floats", FLOATS), ("wide", WIDE)] {
        let obj = compile(TargetArch::Thumb, src);
        let file = object_file(TargetArch::Thumb, &obj, ObjectFormat::Elf);
        let Some(report) = differential(TargetArch::Thumb, &file, &[TRIPLE], &Options::default(), &|s| s) else {
            eprintln!("skipping thumb differential_compiled: no llvm-objdump");
            return;
        };
        assert_clean(&format!("thumb {name}"), &report);
        total += report.compared;
    }
    eprintln!("thumb compiled objects: {total} instructions identical to llvm-objdump");
}

/// i64 arithmetic and shifts, bit operations, division and a dense switch.
const WIDE: &str = r#"
module "wide"

func @mul64(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %m = mul %a, %b : i64
  %s = shl %m, i64 13 : i64
  %t = lshr %a, %b : i64
  %u = ashr %b, i64 40 : i64
  %x = xor %s, %t : i64
  %r = add %x, %u : i64
  ret %r
}

func @div64(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %q = sdiv %a, %b : i64
  %r = urem %a, %b : i64
  %s = sub %q, %r : i64
  ret %s
}

func @bits(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %x = and %a, i32 16711935 : i32
  %y = or %b, i32 -16777216 : i32
  %c = icmp ule %x, %y : i1
  %s = select %c, %x, %y : i32
  %d = sdiv %s, %b : i32
  %e = srem %a, i32 7 : i32
  %f = add %d, %e : i32
  ret %f
}

func @big(i32) -> i32 {
entry ^0(%x: i32):
  switch %x, ^9 [0: ^1, 1: ^2, 2: ^3, 3: ^4, 4: ^5, 5: ^6, 6: ^7, 7: ^8]
^1:
  ret i32 10
^2:
  ret i32 20
^3:
  ret i32 30
^4:
  ret i32 40
^5:
  ret i32 50
^6:
  ret i32 60
^7:
  ret i32 70
^8:
  ret i32 80
^9:
  ret i32 -1
}
"#;

/// A broad hand-written corpus (every instruction class, addressing mode,
/// IT blocks, hints, barriers, system registers), assembled by llvm-mc and
/// compared instruction by instruction with llvm-objdump.
#[test]
fn differential_llvm_mc_corpus() {
    let Some(mc) = llvm_tool("llvm-mc") else {
        eprintln!("skipping thumb differential_llvm_mc_corpus: no llvm-mc");
        return;
    };
    let dir = scratch("thumb-mc");
    let src = dir.join("in.s");
    let obj = dir.join("in.o");
    std::fs::write(&src, format!(".syntax unified\n.thumb\nf:\n{MC_CORPUS}")).unwrap();
    let out = Command::new(mc).args([TRIPLE, "-filetype=obj", "-o"]).arg(&obj).arg(&src).output().expect("run llvm-mc");
    assert!(out.status.success(), "llvm-mc: {}", String::from_utf8_lossy(&out.stderr));
    let file = std::fs::read(&obj).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let Some(report) = differential(TargetArch::Thumb, &file, &[TRIPLE], &Options::default(), &|s| s) else {
        eprintln!("skipping thumb differential_llvm_mc_corpus: no llvm-objdump");
        return;
    };
    assert_clean("thumb llvm-mc corpus", &report);
}

/// Random streams decode in step: 2- or 4-byte instructions covering every
/// byte.
#[test]
fn random_streams_stay_in_step() {
    let mut rng = Rng(99);
    for _ in 0..200 {
        let bytes = rng.bytes(256);
        let insts = disassemble(TargetArch::Thumb, &bytes, 0x2000, &Options::default());
        let mut at = 0x2000;
        for (a, i) in &insts {
            assert_eq!(*a, at);
            assert!(i.len == 2 || i.len == 4);
            at += i.len as u64;
        }
    }
}

/// The hand-written corpus (unified syntax, assembled after a label `f`).
const MC_CORPUS: &str = r#"
lsls r0, r1, #2
movs r0, r1
adds r0, r1, r2
adds r0, r1, #1
adds r0, #100
cmp r0, #5
ands r0, r1
muls r0, r1, r0
rsbs r0, r1, #0
tst r0, r1
add r0, r9
add sp, r1
add r0, sp, r0
mov r8, r0
cmp r8, r1
bx lr
blx r3
ldr r0, [pc, #8]
ldr r0, [r1]
ldr r0, [r1, #4]
ldrb r0, [r1, r2]
ldrsh r0, [r1, r2]
ldr r0, [sp, #8]
str r0, [sp]
adr r0, f
add r0, sp, #8
add sp, #8
sub sp, #12
sxth r0, r1
push {r4, r5, r7, lr}
pop {r4, pc}
rev r0, r1
cpsid i
bkpt #3
nop
it eq
moveq r0, #1
ite ne
addne r0, r1, r2
addeq r0, r1
stm r0!, {r1, r2}
ldm r0!, {r1, r2}
ldm r0, {r0, r1}
udf #5
svc #1
cbz r0, 1f
1:
beq f
b f
push.w {r4, r5, r6, r7, r8, lr}
pop.w {r4, r5, r6, r7, r8, pc}
push.w {r4, lr}
ldm.w r0, {r1, r2, r8}
ldmdb r0!, {r1, r2}
stmdb r1, {r2, r3}
strex r0, r1, [r2, #4]
ldrex r1, [r2]
ldrd r0, r1, [r2, #8]
ldrd r0, r1, [r2], #-8
strd r0, r1, [r2, #8]!
tbb [r0, r1]
tbh [r0, r1, lsl #1]
ldrexb r0, [r1]
strexh r0, r1, [r2]
and.w r0, r1, r2, lsl #3
ands.w r0, r1, r2
tst.w r0, r1
mov.w r0, r1
movs.w r0, r1
lsl.w r0, r1, #3
lsrs.w r0, r1, #32
rrx r0, r1
mvn.w r0, r1
mvn r0, r1, ror #4
orn r0, r1, r2
add.w r0, r1, r2, asr #4
cmp.w r0, r1
sub.w r0, r1, r2
rsb r0, r1, r2
and r0, r1, #255
orr r0, r1, #0x10001
mov.w r0, #0x80000000
mvn r0, #0
cmp.w r0, #256
add.w r0, r1, #4
adds.w r0, r1, #4
sub.w sp, sp, #8
addw r0, r1, #4095
subw r0, r1, #1
movw r0, #0x1234
movt r0, #0xabcd
bfi r0, r1, #4, #8
bfc r0, #4, #8
ubfx r0, r1, #3, #5
sbfx r0, r1, #0, #32
ssat r0, #8, r1
usat r0, #8, r1, lsl #4
b.w f
bne.w f
bl f
msr apsr_nzcvq, r0
mrs r0, primask
nop.w
dmb sy
dsb sy
isb sy
dmb ish
strb.w r0, [r1, #4095]
str r0, [r1, #-4]
str r0, [r1, #4]!
str r0, [r1], #4
ldr.w r0, [r1, r2, lsl #2]
ldrsb.w r0, [r1, #4]
ldr.w r0, [pc, #-8]
ldrt r0, [r1, #4]
pld [r0]
lsl.w r0, r1, r2
asrs.w r0, r1, r2
sxth.w r0, r1
uxtb r0, r1, ror #8
clz r0, r1
rbit r0, r1
rev.w r0, r1
mul r0, r1, r2
mla r0, r1, r2, r3
mls r0, r1, r2, r3
smull r0, r1, r2, r3
umlal r0, r1, r2, r3
sdiv r0, r1, r2
udiv r0, r1, r2
ldr r0, [r1, #0]
ldr.w r0, [r1]
cbz r0, 2f
cbnz r1, 2f
nop
nop
2:
adr r0, 3f
nop
.p2align 2
3:
rsb.w r0, r1, #4
rsbs r0, r1, #4
cmn r0, #4
cmn.w r0, r1
tst r0, #4
teq r0, #4
teq r0, r1
eor r0, r1, #4
eor.w r0, r1, r2
eors.w r0, r1, r2
bic r0, r1, #4
bic.w r0, r1, r2
adc r0, r1, #4
adc.w r0, r1, r2
sbc r0, r1, #4
sbc.w r0, r1, r2
orr.w r0, r1, r2
orrs.w r0, r1, r2
mov r0, #4
movs.w r0, #4
mvns r0, #4
ror.w r0, r1, #4
ror.w r0, r1, r2
asr.w r0, r1, #4
lsr.w r0, r1, #4
ldrh.w r0, [r1, #4]
ldrsh r0, [r1, #4]
ldrsh r0, [r1, #-4]
ldrb r0, [r1, #-4]
ldrb r0, [r1], #-4
ldrh r0, [r1, #4]!
ldrsh.w r0, [r1, r2]
ldrb.w r0, [r1, r2, lsl #1]
strh.w r0, [r1, r2]
strh.w r0, [r1, #8]
ldrb.w r0, [pc, #4]
ldrsh.w r0, [pc, #-4]
ldrd r0, r1, [pc, #8]
ldrd r0, r1, [r2]
strd r0, r1, [r2, #-8]
ldrbt r0, [r1, #4]
strht r0, [r1]
ldr r0, [r1, #-0]
pli [r0, #4]
pld [r0, r1]
sxtb r0, r1
uxth r0, r1
uxtb r0, r1
sxtb.w r0, r1, ror #16
uxth.w r0, r9
rev16 r0, r1
revsh r0, r1
rev16.w r0, r1
revsh.w r0, r1
cpsie i
cpsid f
yield
wfi
wfe
sev
yield.w
wfi.w
itt eq
moveq r0, r1
addeq r0, #1
itete gt
movgt r0, #1
movle r0, #2
addgt r0, r1, r2
suble r0, r1, r2
it ne
bne 2b
it lt
bllt f
itt hs
ldrhs r0, [r1]
lslhs r0, r1, #2
it lo
addlo.w r0, r1, r2
mov r0, r1
mov r0, sp
mov sp, r0
add r1, pc
mov pc, lr
add sp, sp, #4
sub sp, sp, #4
add r0, sp, #1024
sub r0, sp, #4
add.w r0, sp, r1, lsl #2
ldm r0, {r0, r1}
ldmia.w r0!, {r1, r2}
stmia.w r0!, {r1, r2, r8}
stm.w r0, {r1, r2}
stmdb r0!, {r1, r2}
ldmdb sp!, {r1, r2}
push.w {r1}
pop.w {r1}
dmb
dsb ish
clrex
strexb r0, r1, [r2]
strex r0, r1, [r2]
mrs r0, apsr
mrs r0, msp
msr basepri, r0
msr primask, r0
bkpt #0
udf.w #300
smlal r0, r1, r2, r3
umull r0, r1, r2, r3
mul r1, r1, r2
movs r0, #0
mvn r0, r1
uxtb.w r0, r1
ssat r0, #16, r1, asr #3
usat r0, #31, r1
bfi r0, r1, #0, #32
sbfx r0, r1, #31, #1
and r0, r1, #0x00ff00ff
and r0, r1, #0xff00ff00
and r0, r1, #0xabababab
and r0, r1, #0x3fc
tst.w r0, r1, lsl #2
cmp.w r0, r1, lsr #3
neg r0, r1
mov.w r0, r1, lsl #0
add.w r0, r1, #0x10000
addw r0, pc, #4
subw r0, pc, #4
add r0, pc, #8
"#;

/// encode∘decode through an independent assembler: the decoded text of a
/// fuzzed encoder corpus, re-assembled by rsasm's Thumb backend, gives back
/// the same bytes (or, where the assembler picks another encoding of the
/// same instruction, text that decodes identically).
#[test]
fn rsasm_reassembles_decoded_text() {
    let mut rng = Rng(0x5eed);
    let mut insts: Vec<T> = Vec::new();
    for _ in 0..200 {
        let (d, m, k) = (rng.below(8) as u32, rng.below(8) as u32, rng.below(8) as u32);
        let (a, b, x) = (rng.below(13) as u32, rng.below(13) as u32, rng.below(13) as u32);
        let s = rng.below(2) == 1;
        let imm8 = rng.below(256) as u32;
        let v = (rng.below(255) as u32 + 1) << rng.below(20);
        insts.extend([
            enc::shift_imm16(rng.below(3) as u32, d, m, 1 + rng.below(31) as u32),
            enc::addsub_reg16(s, d, m, k),
            enc::addsub_imm3(s, d, m, k),
            enc::movs_imm8(d, imm8),
            enc::cmp_imm8(d, imm8),
            enc::addsub_imm8(s, d, imm8),
            enc::dp16([0, 1, 2, 3, 4, 8, 10, 12, 14, 15][rng.below(10) as usize], d, m),
            enc::add_hi(a, b),
            enc::mov_reg16(a, b),
            enc::ldst_imm16(s, 4, d, m, rng.below(32) as u32 * 4),
            enc::ldst_imm16(s, 1, d, m, rng.below(32) as u32),
            enc::ldst_sp16(s, d, rng.below(256) as u32 * 4),
            enc::ext16(rng.below(4) as u32, d, m),
            enc::push16(imm8 | 1, s),
            enc::svc(imm8),
        ]);
        if let Some(i12) = enc::mod_imm(v) {
            insts.push(enc::dp_modimm([0, 1, 2, 4, 8, 13, 14][rng.below(7) as usize], s, a, b, i12));
        }
        insts.extend([
            enc::dp_plainimm(0, a, b, rng.below(4096) as u32),
            enc::movw(a, rng.below(65536) as u32),
            enc::movt(a, rng.below(65536) as u32),
            enc::bfx(s, a, b, 3, 7),
            enc::dp_reg([0, 1, 2, 4, 8, 13][rng.below(6) as usize], s, a, b, x, rng.below(3) as u32, 1 + rng.below(31) as u32),
            enc::shift_reg32(rng.below(3) as u32, a, b, x),
            enc::ext32([0, 1, 4, 5][rng.below(4) as usize], a, b),
            enc::mul32(a, b, x),
            enc::mls(a, b, x, d),
            enc::div(s, a, b, x),
            enc::ldst_imm12(s, [1, 2, 4][rng.below(3) as usize], a, b, rng.below(4096) as u32),
            enc::ldst_neg8(s, 4, a, b, 1 + rng.below(255) as u32),
            enc::ldst_dual(s, 2, 3, b, rng.below(256) as u32 * 4),
            enc::push32(0x4ff0),
            enc::pop32(0x8ff0),
        ]);
    }
    insts.push(enc::dmb_sy());
    let bytes: Vec<u8> = insts.iter().flat_map(|t| t.bytes()).collect();
    let decoded = disassemble(TargetArch::Thumb, &bytes, 0, &Options::default());
    assert!(decoded.iter().all(|(_, i)| i.known));
    let text: String = decoded.iter().map(|(_, i)| format!("\t{}\n", i.text())).collect();
    let src = format!(".syntax unified\n.thumb\n{text}");
    let opts = crate::mc::asm::AsmOptions::new(TargetArch::Thumb);
    let elf = match crate::mc::asm::assemble(&[crate::mc::asm::AsmSource { name: "rt.s", text: &src }], &opts) {
        Ok(elf) => elf,
        Err(e) => panic!("rsasm rejected the decoded text:\n{e}"),
    };
    let bin = crate::mc::disasm::objfile::read(&elf).expect("read rsasm's object");
    let again = &bin.sections.iter().find(|s| s.name == ".text").expect(".text").bytes;
    let mut same = 0;
    let mut equivalent = 0;
    let redecoded = disassemble(TargetArch::Thumb, again, 0, &Options::default());
    if again == &bytes {
        same = decoded.len();
    } else {
        // Compare instruction texts in order (encodings may differ in size).
        assert_eq!(decoded.len(), redecoded.len(), "instruction counts");
        for ((_, a), (_, b)) in decoded.iter().zip(&redecoded) {
            let (ta, tb) = (normalize(TargetArch::Thumb, &a.text()), normalize(TargetArch::Thumb, &b.text()));
            assert_eq!(ta, tb, "re-assembled differently");
            if a.len == b.len { same += 1 } else { equivalent += 1 }
        }
    }
    eprintln!("thumb rsasm round trip: {same} identical, {equivalent} re-encoded equivalently, of {}", decoded.len());
}

/// Random encodings across the whole 16- and 32-bit space (emitted with
/// `.inst.n` / `.inst.w`, IT instructions excluded), compared with
/// llvm-objdump: both sides agree on every instruction, and on which
/// encodings are not instructions (llvm's `<unknown>`, our `.inst`).
#[test]
fn differential_random_encodings() {
    let Some(mc) = llvm_tool("llvm-mc") else {
        eprintln!("skipping thumb differential_random_encodings: no llvm-mc");
        return;
    };
    let mut rng = Rng(0x4242_4242_4242);
    let mut src = String::from(".syntax unified\n.thumb\nf:\n");
    for k in 0..60000 {
        if k % 3 == 0 {
            let mut h = rng.next() as u32 & 0xffff;
            while h >> 11 >= 0b11101 || (h >> 8 == 0xbf && h & 15 != 0) {
                h = rng.next() as u32 & 0xffff;
            }
            src.push_str(&format!(".inst.n {h:#06x}\n"));
        } else {
            let (mut h1, mut h2) = (0, 0);
            while h1 == 0 || idiosyncratic(h1, h2) {
                h1 = 0xe800 + (rng.next() as u32 % 0x1800);
                h2 = rng.next() as u32 & 0xffff;
            }
            src.push_str(&format!(".inst.w {:#010x}\n", h1 << 16 | h2));
        }
    }
    let dir = scratch("thumb-rand");
    let (s, o) = (dir.join("in.s"), dir.join("in.o"));
    std::fs::write(&s, src).unwrap();
    let out = Command::new(mc).args([TRIPLE, "-filetype=obj", "-o"]).arg(&o).arg(&s).output().expect("run llvm-mc");
    assert!(out.status.success(), "llvm-mc: {}", String::from_utf8_lossy(&out.stderr));
    let file = std::fs::read(&o).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    // Unknown on both sides; and llvm's own names for two UDF immediates
    // (its `trap` and Windows' `__brkdiv0`), which the manual spells `udf`.
    let unknown = |s: String| match s.as_str() {
        "" => "unknown".to_owned(),
        _ if s.starts_with(".inst") => "unknown".to_owned(),
        "trap" => "udf #254".to_owned(),
        "__brkdiv0" => "udf #249".to_owned(),
        _ => s,
    };
    let Some(report) = differential(TargetArch::Thumb, &file, &[TRIPLE], &Options::default(), &unknown) else {
        eprintln!("skipping thumb differential_random_encodings: no llvm-objdump");
        return;
    };
    assert_clean("thumb random encodings", &report);
}

/// UNPREDICTABLE 32-bit encodings whose rendering by llvm-objdump follows
/// its own conventions rather than the architecture, left out of the random
/// differential: MRS/MSR with a SYSm value ARMv7-M does not define (llvm
/// accepts or rejects them by its banked-register tables), CLZ/RBIT/REV*
/// whose two copies of Rm differ, and BFI/BFC with msb < lsb (we decode
/// neither of the latter two).
fn idiosyncratic(h1: u32, h2: u32) -> bool {
    let sysm = h2 & 0xff;
    let defined = matches!(sysm, 0..=3 | 5..=9 | 16..=20);
    let msr_mrs = h1 & 0xffe0 == 0xf380 || h1 & 0xffe0 == 0xf3e0;
    let misc = h1 & 0xfff0 == 0xfa90 || h1 & 0xfff0 == 0xfab0;
    let bfi = h1 & 0xfbf0 == 0xf360;
    let lsb = (h2 >> 10) & 0x1c | (h2 >> 6) & 3;
    (msr_mrs && h2 & 0x8000 != 0 && h2 & 0x5000 == 0 && !defined)
        || (misc && h2 & 0xf000 == 0xf000 && h1 & 15 != h2 & 15)
        || (bfi && h2 & 0x8000 == 0 && h2 & 0x1f < lsb)
}
