//! Encoder, object and firmware tests for the Thumb-2 backend.
//!
//! This host can neither run Arm code nor has an `arm-none-eabi` toolchain,
//! so the encoder is checked three ways:
//!
//! - **golden encodings**, hand-checked against the ARMv7-M ARM, which need
//!   no tools;
//! - a **differential corpus against `llvm-mc --triple=thumbv7m`** covering
//!   every encoding form the encoder emits (16- and 32-bit, IT blocks,
//!   branches of both sizes), skipped when `llvm-mc` is absent;
//! - whole compiled functions **disassembled by `llvm-objdump`**, and run on
//!   the machine-code simulator ([`super::sim`]) by the differential suite
//!   ([`super::diff_tests`]).

use super::encode::*;
use crate::support::StrInterner;

/// `llvm-mc --triple=thumbv7m --show-encoding` over `asm` (one instruction per
/// line): the bytes of each instruction, in order. `None` when `llvm-mc` is
/// unavailable or rejects the input.
pub(super) fn llvm_mc(asm: &str) -> Option<Vec<Vec<u8>>> {
    use std::io::Write;
    let mut child = std::process::Command::new("llvm-mc")
        .args(["--triple=thumbv7m", "--show-encoding"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(asm.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        eprintln!("llvm-mc: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut all = Vec::new();
    for line in text.lines() {
        let Some(pos) = line.find("encoding: [") else { continue };
        let start = pos + "encoding: [".len();
        let end = line[start..].find(']')? + start;
        let mut bytes = Vec::new();
        for tok in line[start..end].split(',') {
            let tok = tok.trim().trim_start_matches("0x");
            bytes.push(u8::from_str_radix(tok, 16).ok()?);
        }
        all.push(bytes);
    }
    Some(all)
}

pub(super) fn have_tool(cmd: &str) -> bool {
    std::process::Command::new(cmd)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Every encoding form the encoder emits, with its assembly.
fn corpus() -> Vec<(T, String)> {
    let mut c: Vec<(T, String)> = Vec::new();
    let mut add = |t: T, s: &str| c.push((t, s.to_owned()));
    // 16-bit shifts, add/sub, mov/cmp immediates.
    add(shift_imm16(0, 1, 2, 3), "lsls r1, r2, #3");
    add(shift_imm16(1, 7, 0, 31), "lsrs r7, r0, #31");
    add(shift_imm16(2, 3, 4, 1), "asrs r3, r4, #1");
    add(addsub_reg16(false, 0, 1, 2), "adds r0, r1, r2");
    add(addsub_reg16(true, 7, 6, 5), "subs r7, r6, r5");
    add(addsub_imm3(false, 1, 2, 7), "adds r1, r2, #7");
    add(addsub_imm3(true, 3, 3, 1), "subs r3, r3, #1");
    add(movs_imm8(5, 200), "movs r5, #200");
    add(cmp_imm8(2, 255), "cmp r2, #255");
    add(addsub_imm8(false, 4, 100), "adds r4, #100");
    add(addsub_imm8(true, 6, 8), "subs r6, #8");
    // The data-processing (register) group.
    add(dp16(0, 1, 2), "ands r1, r2");
    add(dp16(1, 3, 4), "eors r3, r4");
    add(dp16(2, 0, 7), "lsls r0, r7");
    add(dp16(3, 5, 6), "lsrs r5, r6");
    add(dp16(4, 2, 3), "asrs r2, r3");
    add(dp16(9, 0, 1), "rsbs r0, r1, #0");
    add(dp16(10, 6, 7), "cmp r6, r7");
    add(dp16(12, 4, 5), "orrs r4, r5");
    add(dp16(13, 2, 3), "muls r2, r3, r2");
    add(dp16(15, 1, 0), "mvns r1, r0");
    // High-register forms.
    add(add_hi(12, 13), "add r12, sp");
    add(add_hi(8, 1), "add r8, r1");
    add(cmp_hi(1, 12), "cmp r1, r12");
    add(cmp_hi(9, 2), "cmp r9, r2");
    add(mov_reg16(0, 1), "mov r0, r1");
    add(mov_reg16(8, 14), "mov r8, lr");
    add(mov_reg16(3, 12), "mov r3, r12");
    add(blx(4), "blx r4");
    add(blx(9), "blx r9");
    add(bx(14), "bx lr");
    // 16-bit loads and stores.
    add(ldst_imm16(true, 4, 0, 1, 0), "ldr r0, [r1]");
    add(ldst_imm16(false, 4, 2, 3, 124), "str r2, [r3, #124]");
    add(ldst_imm16(true, 1, 4, 5, 31), "ldrb r4, [r5, #31]");
    add(ldst_imm16(false, 1, 6, 7, 1), "strb r6, [r7, #1]");
    add(ldst_imm16(true, 2, 0, 0, 62), "ldrh r0, [r0, #62]");
    add(ldst_imm16(false, 2, 1, 2, 2), "strh r1, [r2, #2]");
    add(ldst_sp16(true, 3, 1020), "ldr r3, [sp, #1020]");
    add(ldst_sp16(false, 7, 4), "str r7, [sp, #4]");
    add(add_rd_sp16(2, 16), "add r2, sp, #16");
    add(addsub_sp16(false, 508), "add sp, #508");
    add(addsub_sp16(true, 8), "sub sp, #8");
    // Extensions.
    add(ext16(0, 1, 2), "sxth r1, r2");
    add(ext16(1, 3, 4), "sxtb r3, r4");
    add(ext16(2, 5, 6), "uxth r5, r6");
    add(ext16(3, 7, 0), "uxtb r7, r0");
    add(ext32(0, 8, 1), "sxth.w r8, r1");
    add(ext32(1, 1, 9), "uxth.w r1, r9");
    add(ext32(4, 10, 11), "sxtb.w r10, r11");
    add(ext32(5, 14, 2), "uxtb.w lr, r2");
    // Push / pop.
    add(push16(0b1111_0000, true), "push {r4, r5, r6, r7, lr}");
    add(push16(0, true), "push {lr}");
    add(pop16(0b0001_0000, true), "pop {r4, pc}");
    add(push32(1 << 4 | 1 << 8 | 1 << 14), "push.w {r4, r8, lr}");
    add(pop32(1 << 4 | 1 << 9 | 1 << 10 | 1 << 11 | 1 << 15), "pop.w {r4, r9, r10, r11, pc}");
    // Traps and barriers.
    add(udf(0), "udf #0");
    add(svc(0), "svc #0");
    add(dmb_sy(), "dmb sy");
    // Modified immediates (all four replication patterns and rotations).
    for (v, d) in [(0x55u32, 1u32), (0x0034_0034, 2), (0x7800_7800, 3), (0xabab_abab, 4), (0xff00_0000, 8), (0x0003_fc00, 9), (4096, 12)] {
        let m = mod_imm(v).expect("encodable");
        add(dp_modimm(2, false, d, 15, m), &format!("mov.w r{d}, #{v}"));
        add(dp_modimm(8, false, d, 5, m), &format!("add.w r{d}, r5, #{v}"));
        add(dp_modimm(13, false, d, 6, m), &format!("sub.w r{d}, r6, #{v}"));
        add(dp_modimm(0, false, d, 7, m), &format!("and r{d}, r7, #{v}"));
        add(dp_modimm(2, false, d, 1, m), &format!("orr r{d}, r1, #{v}"));
        add(dp_modimm(4, false, d, 2, m), &format!("eor r{d}, r2, #{v}"));
        add(dp_modimm(1, false, d, 3, m), &format!("bic r{d}, r3, #{v}"));
        add(dp_modimm(3, false, d, 4, m), &format!("orn r{d}, r4, #{v}"));
        add(dp_modimm(14, false, d, 0, m), &format!("rsb.w r{d}, r0, #{v}"));
        add(dp_modimm(13, true, 15, d, m), &format!("cmp.w r{d}, #{v}"));
        add(dp_modimm(8, true, 15, d, m), &format!("cmn.w r{d}, #{v}"));
        add(dp_modimm(3, false, d, 15, m), &format!("mvn r{d}, #{v}"));
    }
    assert_eq!(mod_imm(0x101), None);
    assert_eq!(mod_imm(0x1234_5678), None);
    add(dp_modimm(0, true, 15, 9, 1), "tst.w r9, #1");
    add(dp_modimm(13, true, 12, 12, 1), "subs.w r12, r12, #1");
    add(dp_modimm(13, false, 13, 13, mod_imm(4096).unwrap()), "sub.w sp, sp, #4096");
    add(dp_modimm(8, false, 13, 13, mod_imm(0x10000).unwrap()), "add.w sp, sp, #65536");
    // Plain immediates.
    add(dp_plainimm(0, 1, 2, 4095), "addw r1, r2, #4095");
    add(dp_plainimm(10, 9, 10, 1234), "subw r9, r10, #1234");
    add(dp_plainimm(0, 4, 13, 2000), "addw r4, sp, #2000");
    add(dp_plainimm(10, 13, 13, 1001), "subw sp, sp, #1001");
    add(movw(0, 0x1234), "movw r0, #0x1234");
    add(movw(12, 0xffff), "movw r12, #0xffff");
    add(movt(3, 0x8765), "movt r3, #0x8765");
    add(movt(14, 1), "movt lr, #1");
    add(bfx(false, 1, 2, 0, 1), "ubfx r1, r2, #0, #1");
    add(bfx(true, 8, 3, 0, 24), "sbfx r8, r3, #0, #24");
    add(bfx(false, 4, 11, 0, 31), "ubfx r4, r11, #0, #31");
    // Register forms.
    for (op, name) in [(0u32, "and.w"), (2, "orr.w"), (4, "eor.w"), (8, "add.w"), (13, "sub.w")] {
        add(dp_reg(op, false, 8, 1, 2, 0, 0), &format!("{name} r8, r1, r2"));
        add(dp_reg(op, false, 3, 9, 14, 0, 0), &format!("{name} r3, r9, lr"));
    }
    add(dp_reg(8, false, 0, 13, 12, 0, 0), "add.w r0, sp, r12");
    add(dp_reg(13, false, 1, 12, 2, 0, 0), "sub.w r1, r12, r2");
    add(dp_reg(3, false, 9, 15, 1, 0, 0), "mvn.w r9, r1");
    add(dp_reg(2, false, 8, 15, 1, 0, 5), "lsl.w r8, r1, #5");
    add(dp_reg(2, false, 2, 15, 9, 1, 31), "lsr.w r2, r9, #31");
    add(dp_reg(2, false, 10, 15, 10, 2, 16), "asr.w r10, r10, #16");
    add(shift_reg32(0, 8, 1, 2), "lsl.w r8, r1, r2");
    add(shift_reg32(1, 1, 9, 2), "lsr.w r1, r9, r2");
    add(shift_reg32(2, 3, 4, 11), "asr.w r3, r4, r11");
    add(mul32(9, 1, 2), "mul r9, r1, r2");
    add(mul32(0, 1, 2), "mul r0, r1, r2");
    add(mls(1, 12, 3, 4), "mls r1, r12, r3, r4");
    add(div(true, 0, 1, 2), "sdiv r0, r1, r2");
    add(div(false, 9, 10, 11), "udiv r9, r10, r11");
    add(div(true, 12, 4, 14), "sdiv r12, r4, lr");
    // 32-bit loads and stores.
    for (size, suf) in [(4u32, ""), (2, "h"), (1, "b")] {
        add(ldst_imm12(true, size, 8, 1, 4000), &format!("ldr{suf}.w r8, [r1, #4000]"));
        add(ldst_imm12(false, size, 1, 9, 12), &format!("str{suf}.w r1, [r9, #12]"));
        add(ldst_imm12(true, size, 2, 13, 2048), &format!("ldr{suf}.w r2, [sp, #2048]"));
        add(ldst_neg8(true, size, 3, 4, 4), &format!("ldr{suf} r3, [r4, #-4]"));
        add(ldst_neg8(false, size, 11, 0, 255), &format!("str{suf} r11, [r0, #-255]"));
    }
    add(ldst_imm12(false, 4, 12, 13, 0), "str.w r12, [sp]");
    add(ldst_dual(true, 0, 1, 2, 0), "ldrd r0, r1, [r2]");
    add(ldst_dual(false, 4, 8, 9, 0), "strd r4, r8, [r9]");
    // IT blocks.
    for cond in [0u32, 1, 2, 3, 8, 9, 10, 11, 12, 13] {
        c.push((it(cond, ite_mask(cond)), format!("ite {}", cond_name(cond))));
        c.push((movs_imm8(0, 1), format!("mov{} r0, #1", cond_name(cond))));
        c.push((movs_imm8(0, 0), format!("mov{} r0, #0", cond_name(cond ^ 1))));
    }
    c.push((it(0, 0b1000), "it eq".into()));
    c.push((dp_modimm(2, false, 8, 15, 1), "moveq.w r8, #1".into()));
    c.push((it(1, ite_mask(1)), "ite ne".into()));
    c.push((mov_reg16(9, 12), "movne r9, r12".into()));
    c.push((mov_reg16(9, 2), "moveq r9, r2".into()));
    // Branches of both sizes, forwards and backwards.
    for off in [-2048i32, -4, 0, 2, 2046] {
        c.push((b16(off), format!("b #{off}")));
    }
    for off in [-256i32, -2, 4, 254] {
        c.push((bcond16(0, off), format!("beq #{off}")));
        c.push((bcond16(12, off), format!("bgt #{off}")));
    }
    for off in [-(1i32 << 24), -4, 4094, 1 << 20, (1 << 24) - 2] {
        c.push((b32(false, off), format!("b.w #{off}")));
        c.push((b32(true, off), format!("bl #{off}")));
    }
    for off in [-(1i32 << 20), -100, 256, (1 << 20) - 2] {
        c.push((bcond32(1, off), format!("bne.w #{off}")));
        c.push((bcond32(11, off), format!("blt.w #{off}")));
    }
    c
}

fn cond_name(c: u32) -> &'static str {
    ["eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le"][c as usize]
}

#[test]
fn golden_encodings() {
    // Hand-checked from the ARMv7-M ARM.
    assert_eq!(addsub_reg16(false, 0, 1, 2).bytes(), [0x88, 0x18]); // adds r0, r1, r2
    assert_eq!(movw(0, 0x1234).bytes(), [0x41, 0xf2, 0x34, 0x20]); // movw r0, #0x1234
    assert_eq!(div(true, 0, 1, 2).bytes(), [0x91, 0xfb, 0xf2, 0xf0]); // sdiv r0, r1, r2
    assert_eq!(it(0, 0b1000).bytes(), [0x08, 0xbf]); // it eq
    assert_eq!(ldst_imm12(true, 4, 8, 1, 4000).bytes(), [0xd1, 0xf8, 0xa0, 0x8f]); // ldr.w r8, [r1, #4000]
    assert_eq!(push16(0b1111_0000, true).bytes(), [0xf0, 0xb5]); // push {r4-r7, lr}
    assert_eq!(b32(true, -4).bytes(), [0xff, 0xf7, 0xfe, 0xff]); // bl .
    assert_eq!(b16(-4).bytes(), [0xfe, 0xe7]); // b .
    assert_eq!(mod_imm(0x00ab_00ab), Some(0x1ab));
    assert_eq!(mod_imm(0x8000_0000), Some(0x400)); // 0x80 ror 8: rot 8, bits 0
}

#[test]
fn differential_encoding_matches_llvm_mc() {
    if !have_tool("llvm-mc") {
        eprintln!("skipping differential_encoding_matches_llvm_mc: no llvm-mc");
        return;
    }
    let corpus = corpus();
    let asm: String = corpus.iter().map(|(_, s)| format!("{s}\n")).collect();
    let got = llvm_mc(&asm).expect("llvm-mc accepts the corpus");
    assert_eq!(got.len(), corpus.len(), "one encoding per corpus line");
    for ((t, s), want) in corpus.iter().zip(&got) {
        assert_eq!(&t.bytes(), want, "`{s}`: ours {:02x?}, llvm-mc {want:02x?}", t.bytes());
    }
    eprintln!("differential encoder gate: {} Thumb-2 encodings matched llvm-mc", corpus.len());
}

/// The immediate materializations: `mov_imm` of many values evaluates (on the
/// simulator) to the value.
#[test]
fn constants_materialize_exactly() {
    let mut vals: Vec<u32> = vec![0, 1, 255, 256, 0xffff, 0x10000, 0x1234_5678, u32::MAX, 0x8000_0000, 0xff00_ff00, 0xfffe_0000];
    let mut x = 0x9e37_79b9u32;
    for _ in 0..40 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        vals.push(x);
        vals.push(x >> 20);
    }
    for v in vals {
        for d in [0u32, 7, 8, 9, 12] {
            let got = super::sim::run_snippet(|a| a.mov_imm(d, v), d).expect("runs");
            assert_eq!(got, v, "mov_imm r{d}, {v:#x}");
        }
    }
}

// ===========================================================================
// Whole functions: objects, disassembly, symbols
// ===========================================================================

pub(super) fn parse(src: &str) -> (crate::ir::Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = crate::ir::text::parse_module(src, crate::support::diagnostics::FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse: {e:?}"));
    m.set_data_layout(super::data_layout());
    if let Err(d) = crate::verify::verify_module(&m) {
        panic!("input does not verify: {d:?}");
    }
    (m, syms)
}

/// A scratch directory for files the external tools read.
pub(super) fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("lf-thumb-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

const SMALL: &str = "\
module \"small\"
global @counter : i32 = i32 7
func @add3(i32, i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32, %c: i32):
  %s = add %a, %b : i32
  %t = add %s, %c : i32
  ret %t
}
func @max(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %c = icmp sgt %a, %b : i1
  %r = select %c, %a, %b : i32
  ret %r
}
func @bump() -> i32 {
entry ^0:
  %v = load @counter align 4 : i32
  %w = call @add3(%v, i32 1, i32 2) : i32
  store %w, @counter align 4 : i32
  ret %w
}
";

#[test]
fn object_is_arm_elf32_with_thumb_symbols_and_relocations() {
    let (m, syms) = parse(SMALL);
    let obj = super::compile_module(&m, &syms);
    let text = obj.sections().iter().position(|s| s.name == ".text").expect(".text");
    // Function symbols carry the Thumb bit; `$t` marks the code.
    for f in ["add3", "max", "bump"] {
        let s = obj.symbol(obj.symbol_id(f).expect(f));
        match s.value {
            crate::mc::object::SymbolValue::Defined { offset, .. } => assert_eq!(offset & 1, 1, "{f}"),
            _ => panic!("{f} undefined"),
        }
    }
    assert!(obj.symbol_id("$t").is_some());
    let kinds: Vec<_> = obj.relocations().iter().map(|r| r.kind).collect();
    use crate::mc::object::RelocKind::*;
    assert!(kinds.contains(&ThumbCall) && kinds.contains(&ThumbMovwAbsNc) && kinds.contains(&ThumbMovtAbs));
    assert!(obj.sections()[text].bytes.len().is_multiple_of(2));

    let elf = crate::mc::write_object(&obj, crate::target::Triple::parse("thumbv7m-none-eabi").unwrap()).unwrap();
    assert_eq!(&elf[..4], b"\x7fELF");
    assert_eq!(elf[4], 1, "ELFCLASS32");
    assert_eq!(elf[5], 1, "little-endian");
    assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), 40, "EM_ARM");
    assert_eq!(u32::from_le_bytes([elf[36], elf[37], elf[38], elf[39]]), 0x0500_0200, "EABI5, soft-float");

    if !have_tool("llvm-readobj") {
        eprintln!("skipping the llvm-readobj check: no llvm-readobj");
        return;
    }
    let dir = scratch_dir("readobj");
    let path = dir.join("small.o");
    std::fs::write(&path, &elf).unwrap();
    let out = std::process::Command::new("llvm-readobj")
        .args(["-h", "-r", "--symbols"])
        .arg(&path)
        .output()
        .expect("llvm-readobj runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    for want in [
        "EM_ARM",
        "elf32-littlearm",
        "R_ARM_THM_CALL add3",
        "R_ARM_THM_MOVW_ABS_NC counter",
        "R_ARM_THM_MOVT_ABS counter",
        "Flags [ (0x5000200)",
    ] {
        assert!(text.contains(want), "llvm-readobj lacks `{want}`:\n{text}");
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// Disassemble `.text` of an Arm ELF object with `llvm-objdump`.
pub(super) fn objdump(elf: &[u8], tag: &str) -> Option<String> {
    if !have_tool("llvm-objdump") {
        return None;
    }
    let dir = scratch_dir(tag);
    let path = dir.join("f.o");
    std::fs::write(&path, elf).ok()?;
    let out = std::process::Command::new("llvm-objdump")
        .args(["-d", "-r", "--triple=thumbv7m", "--no-show-raw-insn"])
        .arg(&path)
        .output()
        .ok()?;
    let _ = std::fs::remove_dir_all(dir);
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn objdump_disassembles_the_compiled_code() {
    let (m, syms) = parse(SMALL);
    let obj = super::compile_module(&m, &syms);
    let elf = crate::mc::elf::write_with(&obj, &crate::mc::elf::ElfTarget::ARM).unwrap();
    let Some(text) = objdump(&elf, "objdump") else {
        eprintln!("skipping: no llvm-objdump");
        return;
    };
    // The disassembler decodes every halfword (no `<unknown>`), sees the
    // prologue/epilogue, the IT block of `max`, the call, and the address
    // materialization.
    assert!(!text.contains("unknown"), "{text}");
    for want in ["push", "pop", "ite", "bl", "movw", "movt", "R_ARM_THM_CALL", "R_ARM_THM_MOVW_ABS_NC"] {
        assert!(text.contains(want), "no `{want}` in:\n{text}");
    }
}

#[test]
fn stack_report_matches_the_frame() {
    let src = "\
module \"frames\"
func @leaf(i32) -> i32 {
entry ^0(%a: i32):
  %b = add %a, i32 1 : i32
  ret %b
}
func @big(i32) -> i32 {
entry ^0(%a: i32):
  %buf = alloca [5000 x i8] : ptr
  %p = ptr_add %buf, i32 100 : ptr
  store i8 3, %p align 1 : i8
  %v = load %p align 1 : i8
  %w = zext %v : i32
  %r = call @leaf(%w) : i32
  ret %r
}
";
    let (m, syms) = parse(src);
    let compiled = super::compile_module_with(&m, &syms, &crate::codegen::CodegenOptions::default());
    let leaf = compiled.stack.get("leaf").expect("leaf");
    assert_eq!(leaf.frame_size, 8, "push {{lr}} padded to 8: {leaf:?}");
    assert_eq!(leaf.return_address, 0);
    let big = compiled.stack.get("big").expect("big");
    assert!(big.frame_size >= 5000 && big.frame_size.is_multiple_of(8), "{big:?}");
    assert!(big.probed);
    assert_eq!(big.direct_callees, vec!["leaf".to_owned()]);
    let depth = compiled.stack.worst_case_depth("big", &crate::codegen::StackAssumptions::new()).unwrap();
    assert_eq!(depth.bytes, big.frame_size + leaf.frame_size);

    // The probed prologue touches every page: a `str.w ip, [sp]` per 4 KiB.
    let obj = compiled.object;
    let elf = crate::mc::elf::write_with(&obj, &crate::mc::elf::ElfTarget::ARM).unwrap();
    if let Some(text) = objdump(&elf, "probes") {
        assert!(text.contains("sub.w\tsp, sp, #0x1000"), "{text}");
        assert!(text.contains("str.w\tr12, [sp]"), "{text}");
    }
    // Without probes, one adjustment.
    let unprobed = super::compile_module_with(&m, &syms, &crate::codegen::CodegenOptions::default().with_stack_probes(false));
    assert!(!unprobed.stack.get("big").unwrap().probed);
}

#[test]
fn registry_and_writer_dispatch() {
    use crate::target::{TargetArch, Triple};
    let t = Triple::parse("thumbv7em-none-eabi").unwrap();
    assert_eq!(t.arch, TargetArch::Thumb);
    let (m, syms) = parse(SMALL);
    let c = crate::target::compile_module_for(TargetArch::Thumb, &m, &syms, &crate::codegen::CodegenOptions::default())
        .unwrap();
    let bytes = crate::mc::write_object(&c.object, t).unwrap();
    assert_eq!(u16::from_le_bytes([bytes[18], bytes[19]]), 40);
    // PIC is refused with an error, not a panic.
    let pic = crate::codegen::CodegenOptions::default().with_pic(true);
    assert!(crate::target::compile_module_for(TargetArch::Thumb, &m, &syms, &pic).is_err());
    // The .lfo form round-trips the Thumb relocation kinds.
    let lfo = crate::mc::lfo::encode(&c.object);
    let back = crate::mc::lfo::decode(&lfo).expect("decodes");
    assert_eq!(back.relocations(), c.object.relocations());
}

// ===========================================================================
// Firmware: startup object, qld link, flashable image
// ===========================================================================

const FIRMWARE: &str = "\
module \"blink\"
global @initialized : i32 = i32 1234
global @zeroed : [4 x i32] = [4 x i32] (i32 0, i32 0, i32 0, i32 0)
func @main() -> i32 {
entry ^0:
  %a = load @initialized align 4 : i32
  %p = ptr_add @zeroed, i32 12 : ptr
  %b = load %p align 4 : i32
  %c = add %a, %b : i32
  store %c, %p align 4 : i32
  %d = mul %c, i32 3 : i32
  ret %d
}
";

/// Load an ELF32 executable's segments into simulator memory and run it from
/// the reset vector until it parks in a `b .` loop.
fn run_firmware(elf: &[u8]) -> super::sim::Cpu {
    let segs = crate::link::raw::load_segments(elf).expect("segments");
    let mut mem = super::sim::Memory::default();
    for s in &segs {
        mem.write_bytes(s.addr as u32, &s.data);
    }
    let mut cpu = super::sim::Cpu::new(mem, Default::default());
    let (sp, reset) = (cpu.mem.read(0, 4), cpu.mem.read(4, 4));
    cpu.r[13] = sp;
    assert_eq!(reset & 1, 1, "the reset vector has the Thumb bit");
    cpu.r[15] = reset & !1;
    cpu.run().expect("runs to the idle loop");
    cpu
}

#[test]
fn firmware_links_with_qld_and_boots() {
    let (m, syms) = parse(FIRMWARE);
    let obj = super::compile_module(&m, &syms);
    let startup = super::firmware::startup_object("main", 8);
    let dir = scratch_dir("firmware");
    let out = dir.join("blink.elf");
    let layout = super::firmware::MemoryLayout::default();
    super::firmware::link_elf(&[obj, startup], &layout, &[], &out).expect("qld links");
    let elf = std::fs::read(&out).unwrap();
    assert_eq!(&elf[..4], b"\x7fELF");
    assert_eq!(elf[4], 1, "ELF32");
    assert_eq!(u16::from_le_bytes([elf[16], elf[17]]), 2, "ET_EXEC");
    assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), 40, "EM_ARM");
    let entry = super::firmware::elf32_entry(&elf).unwrap();
    assert_eq!(entry & 1, 1, "e_entry is a Thumb address");

    // The reset handler copies .data, zeroes .bss, and calls main.
    let cpu = run_firmware(&elf);
    assert_eq!(cpu.r[0], 1234 * 3, "main's result is left in r0");

    // The Intel HEX image starts with the vector table at the flash origin.
    let segs = crate::link::raw::load_segments(&elf).unwrap();
    let hex = crate::link::raw::to_ihex(&segs, Some(entry)).unwrap();
    assert!(hex.starts_with(":10000000"), "{hex}");
    assert!(hex.ends_with(":00000001FF\r\n"));
    let (base, bin) = crate::link::raw::to_binary(&segs, 0xff).unwrap();
    assert_eq!(base, layout.flash_origin);
    assert_eq!(u32::from_le_bytes(bin[0..4].try_into().unwrap()), 0x2001_0000, "initial sp = end of RAM");
    if have_tool("llvm-readobj") {
        let o = std::process::Command::new("llvm-readobj").args(["-h", "-l"]).arg(&out).output().unwrap();
        let text = String::from_utf8_lossy(&o.stdout);
        assert!(text.contains("EM_ARM") && text.contains("PT_LOAD"), "{text}");
    }
    let _ = std::fs::remove_dir_all(dir);
}
