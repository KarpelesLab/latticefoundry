//! Instruction selection keeps constant-time code constant-time
//! (`docs/ir-design.md` §6d), on every target.
//!
//! - **No introduced branches.** For every function, the number of MIR
//!   instructions that may branch on a register value
//!   (`may_branch_on_data`) equals the number of IR `cond_br`/`switch`
//!   terminators: instruction selection adds no data-dependent control flow
//!   of its own to the operations the constant-time verifier allows on
//!   secrets. A straight-line function over every allowed operation compiles
//!   to machine code with no conditional branch at all, checked on the bytes
//!   (AArch64 and RISC-V by decoding the fixed-width words, x86-64 with
//!   `llvm-mc --disassemble` when it is installed).
//! - **`select` is branchless**: `cmov` on x86-64, `csel` on AArch64, the
//!   mask blend on RISC-V, an `IT`-predicated pair of `mov`s on Thumb.
//! - **Thumb** is checked on the module its backend actually selects from
//!   (after vector scalarization, soft-float lowering and 64-bit
//!   legalization, whose expansions are all selects and compares), and its
//!   bytes are scanned for conditional branches (`b<cond>`, `b<cond>.w`,
//!   `cbz`/`cbnz`) with the Thumb length rule.
//! - **AVR** likewise, on its prepared module (vector scalarization, soft
//!   float with libgcc names, 16-bit legalization), with its words scanned
//!   for conditional branches *and skips* (`brbs`/`brbc`, `sbrc`/`sbrs`,
//!   `sbic`/`sbis`, `cpse`). AVR picks per operation: secret-derived operands
//!   get branch-free compares (the flag read out of `SREG`), barrel-shifter
//!   variable shifts and shift-pair sign extension, public ones the compact
//!   branching forms (which the audit counts separately).
//! - **Branchy lowerings are rejected on secrets**: every opcode whose lowering
//!   is inherently branchy or variable-time (the `u64`↔float fix-ups, the
//!   atomic retry loops, `dyn_alloca`'s probe loop, division) has its secret
//!   operands rejected by the verifier.
//! - **It runs**: a Montgomery-ladder conditional swap and a constant-time
//!   memcmp compiled at `-O2` verify constant-time and compute the right
//!   answer natively on x86-64.

use crate::codegen::mir::MachineFunction;
use crate::ir::inst::InstKind;
use crate::ir::{FuncId, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize};
use crate::verify::{CtPolicy, ct_violations, verify_module};

use super::aarch64::{A64Op, AArch64Target};
use super::avr::{AvrOp, AvrTarget};
use super::riscv::{RiscvTarget, RvOp};
use super::thumb::{ThOp, ThumbTarget};
use super::x86_64::{X86Op, X86_64Target};

/// Every operation the verifier allows on a secret, at several widths, in one
/// straight-line function (integer only, so every target compiles it).
const ALLOWED_LF: &str = r#"
module "allowed"

func @ops(secret i64, secret i64, i64, secret i8, secret i16, secret i32) -> secret i64 {
entry ^0(%a: i64, %b: i64, %p: i64, %c8: i8, %c16: i16, %c32: i32):
  %add = add %a, %b : i64
  %sub = sub %add, %p : i64
  %mul = mul %sub, %b : i64
  %and = and %mul, %a : i64
  %or = or %and, %p : i64
  %xor = xor %or, %b : i64
  %sh = and %b, i64 63 : i64
  %shl = shl %xor, %sh : i64
  %lshr = lshr %shl, %sh : i64
  %ashr = ashr %lshr, %sh : i64
  %eq = icmp eq %a, %b : i1
  %ult = icmp ult %ashr, %p : i1
  %slt = icmp slt %a, %ashr : i1
  %s1 = select %eq, %a, %b : i64
  %s2 = select %ult, %s1, %ashr : i64
  %s3 = select %slt, %s2, %p : i64
  %n8 = add %c8, i8 3 : i8
  %c8b = icmp ult %n8, i8 200 : i1
  %s8 = select %c8b, %n8, %c8 : i8
  %z8 = zext %s8 : i64
  %n16 = mul %c16, i16 7 : i16
  %c16b = icmp sgt %n16, %c16 : i1
  %x16 = sext %n16 : i64
  %t16 = select %c16b, %x16, %z8 : i64
  %n32 = lshr %c32, i32 5 : i32
  %c32b = icmp sle %n32, i32 1000 : i1
  %e32 = zext %n32 : i64
  %t32 = select %c32b, %e32, %t16 : i64
  %tr = trunc %t32 : i1
  %t1 = select %tr, %s3, %t32 : i64
  %fr = freeze %t1 : i64
  %d = declassify %fr : i64
  %m = mul %d, %d : i64
  %r = xor %fr, %m : i64
  ret %r
}
"#;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    if let Err(d) = verify_module(&m) {
        panic!("verify: {d:#?}");
    }
    (m, syms)
}

/// The IR `cond_br`/`switch` count of a function.
fn ir_branches(m: &Module, f: FuncId) -> usize {
    let func = m.function(f);
    func.blocks()
        .filter_map(|(_, b)| b.terminator())
        .filter(|&t| matches!(func.inst(t).kind, InstKind::CondBr { .. } | InstKind::Switch(_)))
        .count()
}

/// The MIR instructions of `mf` that may branch on data, per a target's audit.
fn mir_branches(mf: &MachineFunction, may: impl Fn(&crate::codegen::mir::MachineInst) -> bool) -> usize {
    mf.block_ids().flat_map(|b| mf.block(b).insts.iter()).filter(|i| may(i)).count()
}

/// Every defined function of `m`, per target, keeps exactly its IR branches.
fn assert_isel_adds_no_branches(m: &Module, syms: &StrInterner, what: &str) {
    let x86 = X86_64Target::new();
    let a64 = AArch64Target::new();
    let rv = RiscvTarget::new();
    for i in 0..m.function_count() {
        let f = FuncId::from_index(i);
        if m.function(f).is_declaration() {
            continue;
        }
        let want = ir_branches(m, f);
        let mf = x86.select_with_syms(m, f, syms);
        let got = mir_branches(&mf, |mi| X86Op::decode(mi.opcode).may_branch_on_data(&mi.operands));
        assert_eq!(got, want, "{what}: x86-64 function #{i}");
        let mf = a64.select(m, f);
        let got = mir_branches(&mf, |mi| A64Op::decode(mi.opcode).may_branch_on_data(&mi.operands));
        assert_eq!(got, want, "{what}: aarch64 function #{i}");
        let mf = rv.select(m, f);
        let got = mir_branches(&mf, |mi| RvOp::decode(mi.opcode).may_branch_on_data(&mi.operands));
        assert_eq!(got, want, "{what}: riscv64 function #{i}");
    }
    assert_thumb_adds_no_branches(m, syms, what);
    assert_avr_adds_no_branches(m, syms, what);
}

/// Whether an AVR MIR instruction is the compact (branching) form of a
/// lowering that also has a constant-time form: isel picks those only for
/// public operands (checked by `avr_picks_the_constant_time_form_per_operand`).
fn avr_compact(mi: &crate::codegen::mir::MachineInst) -> bool {
    let op = AvrOp::decode(mi.opcode);
    matches!(op, AvrOp::SetCmp | AvrOp::ShlV | AvrOp::LshrV | AvrOp::AshrV | AvrOp::Ext)
        && op.may_branch_on_data(&mi.operands)
}

/// The AVR backend selects from its prepared module; every defined function
/// there keeps exactly its IR branches (besides the compact forms on public
/// operands), and preparing adds none.
fn assert_avr_adds_no_branches(m: &Module, syms: &StrInterner, what: &str) {
    let (pm, _, helpers) =
        super::avr::prepare::prepare(m, syms, &super::avr::Device::ATMEGA328P).expect("prepares for AVR");
    let target = AvrTarget::new(true).with_helpers(helpers);
    for i in 0..pm.function_count() {
        let f = FuncId::from_index(i);
        if pm.function(f).is_declaration() {
            continue;
        }
        let mf = target.select(&pm, f);
        let got =
            mir_branches(&mf, |mi| AvrOp::decode(mi.opcode).may_branch_on_data(&mi.operands) && !avr_compact(mi));
        assert_eq!(got, ir_branches(&pm, f), "{what}: avr function #{i}");
        if i < m.function_count() {
            assert_eq!(ir_branches(&pm, f), ir_branches(m, f), "{what}: preparing for AVR adds no branch (#{i})");
        }
    }
}

/// The conditional control transfers in AVR code (walking it with the
/// 2-word `jmp`/`call`/`lds`/`sts` rule): `brbs`/`brbc` (every `br<cond>`),
/// and the skips `sbrc`/`sbrs`, `sbic`/`sbis` and `cpse`, whose timing
/// depends on the tested value.
fn avr_cond_branches(bytes: &[u8]) -> Vec<(usize, u16)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 1 < bytes.len() {
        let w = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        let branch = w & 0xf800 == 0xf000 // brbs / brbc
            || w & 0xfc08 == 0xfc00 // sbrc / sbrs
            || w & 0xfd00 == 0x9900 // sbic / sbis
            || w & 0xfc00 == 0x1000; // cpse
        if branch {
            out.push((at, w));
        }
        let two = w & 0xfe0c == 0x940c || w & 0xfe0f == 0x9000 || w & 0xfe0f == 0x9200;
        at += if two { 4 } else { 2 };
    }
    out
}

/// The code of every function the AVR backend compiles from `m`.
fn avr_text(m: &Module, syms: &StrInterner) -> Vec<u8> {
    let obj = super::avr::compile_module(m, syms);
    obj.sections().iter().filter(|s| s.name == ".text").flat_map(|s| s.bytes.iter().copied()).collect()
}

/// The Thumb backend selects from its prepared module; every defined
/// function there keeps exactly its IR branches.
fn assert_thumb_adds_no_branches(m: &Module, syms: &StrInterner, what: &str) {
    let topts = super::thumb::ThumbOptions::default();
    let (pm, ps) = super::thumb::prepare_module(m, syms, &topts).expect("prepares");
    let target = ThumbTarget::new().with_helpers(super::thumb::isel::Helpers::resolve(&pm, &ps));
    for i in 0..pm.function_count() {
        let f = FuncId::from_index(i);
        if pm.function(f).is_declaration() {
            continue;
        }
        let mf = target.select(&pm, f, &ps);
        let got = mir_branches(&mf, |mi| ThOp::decode(mi.opcode).may_branch_on_data(&mi.operands));
        assert_eq!(got, ir_branches(&pm, f), "{what}: thumb function #{i}");
        if i < m.function_count() {
            assert_eq!(ir_branches(&pm, f), ir_branches(m, f), "{what}: preparing adds no branch (#{i})");
        }
    }
}

/// The conditional branches in Thumb code (walking it with the 16/32-bit
/// length rule): `b<cond>` (T1), `b<cond>.w` (T3), `cbz`/`cbnz`.
fn thumb_cond_branches(bytes: &[u8]) -> Vec<(usize, u16)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 1 < bytes.len() {
        let h = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        let wide = matches!(h >> 11, 0b11101..=0b11111);
        if wide {
            let h2 = u16::from_le_bytes([bytes[at + 2], bytes[at + 3]]);
            let cond = (h >> 6) & 0xf;
            if h >> 11 == 0b11110 && h2 & 0xd000 == 0x8000 && cond < 0xe {
                out.push((at, h));
            }
            at += 4;
        } else {
            let bcc = h >> 12 == 0xd && ((h >> 8) & 0xf) < 0xe;
            let cbz = h & 0xf500 == 0xb100;
            if bcc || cbz {
                out.push((at, h));
            }
            at += 2;
        }
    }
    out
}

/// Whether the little-endian A64 word `w` is a conditional branch
/// (`b.cond`, `cbz`/`cbnz`, `tbz`/`tbnz`).
fn a64_is_cond_branch(w: u32) -> bool {
    (w & 0xFF00_0010) == 0x5400_0000 // b.cond
        || (w & 0x7E00_0000) == 0x3400_0000 // cbz / cbnz
        || (w & 0x7E00_0000) == 0x3600_0000 // tbz / tbnz
}

/// Whether the RV32/64 base word `w` is a conditional branch (`BRANCH` major
/// opcode: `beq`/`bne`/`blt`/`bge`/`bltu`/`bgeu`).
fn rv_is_cond_branch(w: u32) -> bool {
    w & 0x7F == 0x63
}

fn words(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c))
}

/// Disassemble x86-64 bytes with `llvm-mc`; `None` when it is not installed.
fn x86_disasm(bytes: &[u8]) -> Option<String> {
    use std::io::Write;
    let hex: Vec<String> = bytes.iter().map(|b| format!("0x{b:02x}")).collect();
    let mut child = std::process::Command::new("llvm-mc")
        .arg("--triple=x86_64")
        .arg("--disassemble")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(hex.join(",").as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

#[test]
fn branch_scanners_recognize_branches() {
    assert!(a64_is_cond_branch(0x5400_0001)); // b.ne
    assert!(a64_is_cond_branch(0xB400_0000)); // cbz x0
    assert!(a64_is_cond_branch(0x3500_0000)); // cbnz w0
    assert!(a64_is_cond_branch(0x3600_0000)); // tbz w0, #0
    assert!(!a64_is_cond_branch(0x9A80_1000)); // csel x0, x0, x0, ne
    assert!(!a64_is_cond_branch(0x1400_0000)); // b
    assert!(rv_is_cond_branch(0x0000_0063)); // beq x0, x0, 0
    // Thumb: beq (T1), bne.w (T3), cbz; not b (T2), bl, or an IT block.
    assert_eq!(thumb_cond_branches(&[0x00, 0xd0]).len(), 1);
    assert_eq!(thumb_cond_branches(&[0x40, 0xf0, 0x00, 0x80]).len(), 1);
    assert_eq!(thumb_cond_branches(&[0x08, 0xb1]).len(), 1);
    assert!(thumb_cond_branches(&[0x00, 0xe0, 0xff, 0xf7, 0xfe, 0xff, 0x08, 0xbf]).is_empty());
    assert!(!rv_is_cond_branch(0x0000_006F)); // jal x0, 0
    // AVR: breq, brlt, sbrc, sbis, cpse; not rjmp, jmp (whose second word
    // looks like anything), an `in` of SREG, or bst/bld.
    for w in [0xf001u16, 0xf004, 0xfd83, 0x9b00, 0x1001] {
        assert_eq!(avr_cond_branches(&w.to_le_bytes()).len(), 1, "{w:#06x}");
    }
    assert!(avr_cond_branches(&[0xff, 0xcf, 0x0c, 0x94, 0x01, 0xf0, 0xef, 0xb7, 0x80, 0xfb, 0x10, 0xf8]).is_empty());
    if let Some(text) = x86_disasm(&[0x74, 0x00, 0x48, 0x0F, 0x45, 0xC1]) {
        assert!(text.contains("je") && text.contains("cmovne"), "{text}");
    }
}

#[test]
fn allowed_operations_compile_without_branches_on_every_target() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let (mut m, syms) = parse(ALLOWED_LF);
        optimize(&mut m, level);
        verify_module(&m).unwrap_or_else(|d| panic!("{d:#?}"));
        assert_isel_adds_no_branches(&m, &syms, &format!("allowed/{level:?}"));
        let f = FuncId::from_index(0);
        assert_eq!(ir_branches(&m, f), 0);

        let a64 = super::aarch64::compile_function(&m, f, &syms);
        assert!(!a64.bytes.is_empty());
        let bad: Vec<u32> = words(&a64.bytes).filter(|&w| a64_is_cond_branch(w)).collect();
        assert!(bad.is_empty(), "aarch64 conditional branches {bad:08x?} at {level:?}");

        let rv = super::riscv::compile_function(&m, f);
        let bad: Vec<u32> = words(&rv.bytes).filter(|&w| rv_is_cond_branch(w)).collect();
        assert!(bad.is_empty(), "riscv conditional branches {bad:08x?} at {level:?}");

        let th = super::thumb::compile_function(&m, f, &syms);
        assert!(!th.bytes.is_empty());
        let bad = thumb_cond_branches(&th.bytes);
        assert!(bad.is_empty(), "thumb conditional branches {bad:04x?} at {level:?}");

        // AVR: the only function of the module (its 64-bit multiply is a call
        // to __muldi3, compiled elsewhere).
        let avr = avr_text(&m, &syms);
        assert!(!avr.is_empty());
        let bad = avr_cond_branches(&avr);
        assert!(bad.is_empty(), "avr conditional branches/skips {bad:04x?} at {level:?}");

        let x86 = super::x86_64::compile_function(&m, f, &syms);
        match x86_disasm(&x86.bytes) {
            Some(text) => {
                let jcc: Vec<&str> = text
                    .lines()
                    .map(str::trim)
                    .filter(|l| l.starts_with('j') && !l.starts_with("jmp"))
                    .collect();
                assert!(jcc.is_empty(), "x86-64 conditional jumps {jcc:?} at {level:?}:\n{text}");
                assert!(text.contains("cmov"), "select lowers to cmov:\n{text}");
            }
            None => eprintln!("skipping the x86-64 byte scan: no llvm-mc"),
        }
    }
}

#[test]
fn pic_and_win64_lowerings_add_no_branches() {
    use crate::codegen::{CodegenOptions, RelocModel};
    use crate::target::TargetOs;
    // Secret data behind globals and parameters: PIC reaches the globals
    // through the GOT, Win64 passes the fifth and sixth parameters on the
    // stack. Neither changes the straight-line shape.
    let src = format!(
        "{ALLOWED_LF}
global secret @key : i64 = i64 42

global @scale : i64 = i64 3

func @via_globals(secret i64) -> secret i64 {{
entry ^0(%s: i64):
  %k = load @key align 8 : i64
  %c = load @scale align 8 : i64
  %m = mul %k, %c : i64
  %x = xor %m, %s : i64
  %lt = icmp ult %x, %k : i1
  %r = select %lt, %x, %k : i64
  ret %r
}}
"
    );
    let (m, syms) = parse(&src);
    let win = X86_64Target::for_os(TargetOs::Windows);
    for i in 0..m.function_count() {
        let f = FuncId::from_index(i);
        let mf = win.select_with_syms(&m, f, &syms);
        let n = mir_branches(&mf, |mi| X86Op::decode(mi.opcode).may_branch_on_data(&mi.operands));
        assert_eq!(n, 0, "win64 function #{i}");
    }
    let configs = [
        ("pic", CodegenOptions::default().with_reloc_model(RelocModel::Pic)),
        ("pie", CodegenOptions::default().with_reloc_model(RelocModel::Pie)),
        ("win64", CodegenOptions::default().with_os(TargetOs::Windows)),
    ];
    for (what, opts) in configs {
        let obj = super::x86_64::compile_module_with(&m, &syms, &opts).object;
        let text: Vec<u8> = obj
            .sections()
            .iter()
            .filter(|s| s.name == ".text")
            .flat_map(|s| s.bytes.iter().copied())
            .collect();
        assert!(!text.is_empty(), "{what}");
        match x86_disasm(&text) {
            Some(asm) => {
                let jcc: Vec<&str> = asm
                    .lines()
                    .map(str::trim)
                    .filter(|l| l.starts_with('j') && !l.starts_with("jmp"))
                    .collect();
                assert!(jcc.is_empty(), "{what}: conditional jumps {jcc:?}:\n{asm}");
            }
            None => eprintln!("skipping the {what} byte scan: no llvm-mc"),
        }
    }
    // AArch64 under PIC/PIE: `adrp`+`ldr` GOT loads, still no branch.
    for model in [RelocModel::Pic, RelocModel::Pie] {
        let obj = super::aarch64::compile_module_with(&m, &syms, &CodegenOptions::default().with_reloc_model(model)).object;
        let text = &obj.sections().iter().find(|s| s.name == ".text").unwrap().bytes;
        let bad: Vec<u32> = words(text).filter(|&w| a64_is_cond_branch(w)).collect();
        assert!(bad.is_empty(), "aarch64 {model:?}: conditional branches {bad:08x?}");
        assert!(obj.relocations().iter().any(|r| r.kind == crate::mc::object::RelocKind::Aarch64Ld64GotLo12Nc)
            || model == RelocModel::Pie);
    }
}

/// Thread-local addressing (`docs/ir-design.md` §4c) is straight-line under
/// every x86-64 TLS model — local-exec, initial-exec and the general-dynamic
/// `__tls_get_addr` call — so secret thread-locals stay constant-time.
#[test]
fn tls_addressing_adds_no_branches() {
    use crate::codegen::{CodegenOptions, RelocModel};
    let src = "module \"tls\"
global secret thread_local @key : i64 = i64 42
global secret thread_local @ext : i64

func @mix(secret i64) -> secret i64 {
entry ^0(%s: i64):
  %k = load @key align 8 : i64
  %e = load @ext align 8 : i64
  %m = xor %k, %e : i64
  %x = add %m, %s : i64
  %lt = icmp ult %x, %k : i1
  %r = select %lt, %x, %k : i64
  store %r, @key align 8 : i64
  ret %r
}
";
    let (m, syms) = parse(src);
    let f = FuncId::from_index(0);
    assert!(ct_violations(&m, f, CtPolicy::DEFAULT).is_empty());
    for model in [RelocModel::Static, RelocModel::Pie, RelocModel::Pic] {
        let mf = X86_64Target::new().with_reloc_model(model).select_with_syms(&m, f, &syms);
        let n = mir_branches(&mf, |mi| X86Op::decode(mi.opcode).may_branch_on_data(&mi.operands));
        assert_eq!(n, 0, "{model:?}");
        let opts = CodegenOptions::default().with_reloc_model(model);
        let obj = super::x86_64::compile_module_with(&m, &syms, &opts).object;
        let text = &obj.sections().iter().find(|s| s.name == ".text").unwrap().bytes;
        if let Some(asm) = x86_disasm(text) {
            let jcc: Vec<&str> =
                asm.lines().map(str::trim).filter(|l| l.starts_with('j') && !l.starts_with("jmp")).collect();
            assert!(jcc.is_empty(), "{model:?}: conditional jumps {jcc:?}:\n{asm}");
        }
    }
}

/// 128-bit integers on x86-64 (legalized into 64-bit parts, with an inline
/// multiply and register pairs at the boundary) stay branch-free: adds with
/// carry chains, shifts by a secret amount (funnel shifts and a `select`
/// ladder), lexicographic compares and selects all lower to straight-line
/// code, which the constant-time verifier accepts on the prepared module too.
#[test]
fn i128_lowering_adds_no_branches() {
    let src = r#"module "w"
func @ops(secret i128, secret i128, i128) -> secret i128 {
entry ^0(%a: i128, %b: i128, %p: i128):
  %add = add %a, %b : i128
  %sub = sub %add, %p : i128
  %mul = mul %sub, %b : i128
  %and = and %mul, %a : i128
  %xor = xor %and, %p : i128
  %sh = and %b, i128 127 : i128
  %shl = shl %xor, %sh : i128
  %lshr = lshr %shl, %sh : i128
  %ashr = ashr %lshr, %sh : i128
  %eq = icmp eq %a, %b : i1
  %ult = icmp ult %ashr, %p : i1
  %slt = icmp slt %a, %ashr : i1
  %s1 = select %eq, %a, %b : i128
  %s2 = select %ult, %s1, %ashr : i128
  %s3 = select %slt, %s2, %p : i128
  %t = trunc %s3 : i64
  %x = sext %t : i128
  %r = add %x, %s3 : i128
  ret %r
}
"#;
    let (m, syms) = parse(src);
    let f = FuncId::from_index(0);
    assert!(ct_violations(&m, f, CtPolicy::DEFAULT).is_empty());
    let (p, names) = super::x86_64::prepare_module(&m, &syms).expect("prepare");
    let pf = (0..p.function_count()).map(FuncId::from_index).find(|&g| names.resolve(p.function(g).name) == "ops").unwrap();
    let v = ct_violations(&p, pf, CtPolicy::DEFAULT);
    assert!(v.is_empty(), "{v:?}");
    assert_eq!(ir_branches(&p, pf), 0);
    let mf = X86_64Target::new().select_with_syms(&p, pf, &names);
    let n = mir_branches(&mf, |mi| X86Op::decode(mi.opcode).may_branch_on_data(&mi.operands));
    assert_eq!(n, 0);
    let obj = super::x86_64::compile_module(&m, &syms);
    let text = &obj.sections().iter().find(|s| s.name == ".text").unwrap().bytes;
    if let Some(asm) = x86_disasm(text) {
        let jcc: Vec<&str> =
            asm.lines().map(str::trim).filter(|l| l.starts_with('j') && !l.starts_with("jmp")).collect();
        assert!(jcc.is_empty(), "conditional jumps {jcc:?}:\n{asm}");
    }
}

#[test]
fn select_lowers_branchless_on_every_target() {
    let src = r#"module "sel"
func @sel(secret i1, secret i64, secret i64) -> secret i64 {
entry ^0(%c: i1, %a: i64, %b: i64):
  %r = select %c, %a, %b : i64
  ret %r
}
"#;
    let (m, syms) = parse(src);
    let f = FuncId::from_index(0);
    let ops = |mf: &MachineFunction| -> Vec<u32> {
        mf.block_ids().flat_map(|b| mf.block(b).insts.iter().map(|i| i.opcode.0)).collect()
    };
    let x = ops(&X86_64Target::new().select_with_syms(&m, f, &syms));
    assert!(x.contains(&X86Op::Cmovne.opcode().0));
    let a = ops(&AArch64Target::new().select(&m, f));
    // AArch64: `cmp cond, #0` + `csel` (a flag-setting compare, no branch).
    assert!(a.contains(&A64Op::CselNe.opcode().0) && !a.contains(&A64Op::BrCond.opcode().0));
    // RISC-V selects with the branchless `f ^ ((t ^ f) & -c)`.
    let r = ops(&RiscvTarget::new().select(&m, f));
    assert!(r.contains(&RvOp::Xor.opcode().0) && r.contains(&RvOp::And.opcode().0));
    assert!(!r.contains(&RvOp::BrCond.opcode().0));
    // Thumb: `tst c, #1; ite ne; mov; mov` — conditional execution, no branch.
    let t = ops(&ThumbTarget::new().select(&m, f, &syms));
    assert!(t.contains(&ThOp::Select.opcode().0) && !t.contains(&ThOp::BrCond.opcode().0));
    assert!(!ThOp::Select.may_branch_on_data(&[]) && !ThOp::SetCmp.may_branch_on_data(&[]));
    assert!(thumb_cond_branches(&super::thumb::compile_function(&m, f, &syms).bytes).is_empty());
    // AVR: a mask from the condition (`0 - c`) and `f ^ ((t ^ f) & mask)` on
    // each 16-bit part — no branch, and no skip.
    let (pm, _, h) = super::avr::prepare::prepare(&m, &syms, &super::avr::Device::ATMEGA328P).unwrap();
    let v = ops(&AvrTarget::new(true).with_helpers(h).select(&pm, f));
    assert!(v.contains(&AvrOp::Mask.opcode().0) && v.contains(&AvrOp::Xor.opcode().0));
    assert!(!v.contains(&AvrOp::BrCond.opcode().0) && !v.contains(&AvrOp::CmpBr.opcode().0));
    for op in [AvrOp::Mask, AvrOp::Xor, AvrOp::And] {
        assert!(!op.may_branch_on_data(&[]), "{op:?}");
    }
    assert!(avr_cond_branches(&avr_text(&m, &syms)).is_empty());
    assert_isel_adds_no_branches(&m, &syms, "sel");
}

/// AVR picks per operation: an operation on a secret-derived operand gets
/// the constant-time form, the same operation on public operands the compact
/// one (which branches or skips) — within one function.
#[test]
fn avr_picks_the_constant_time_form_per_operand() {
    let src = r#"module "mix"
func @mix(secret i16, i16, secret i4, i4) -> secret i16 {
entry ^0(%s: i16, %p: i16, %s4: i4, %p4: i4):
  %c1 = icmp ult %s, %p : i1
  %c2 = icmp ult %p, i16 5 : i1
  %k = and %p, i16 7 : i16
  %h1 = shl %s, %k : i16
  %h2 = lshr %p, %k : i16
  %x1 = sext %s4 : i16
  %x2 = sext %p4 : i16
  %z1 = zext %c1 : i16
  %z2 = zext %c2 : i16
  %a = add %z1, %z2 : i16
  %b = add %a, %h1 : i16
  %c = add %b, %h2 : i16
  %d = add %c, %x1 : i16
  %r = add %d, %x2 : i16
  ret %r
}
"#;
    use crate::codegen::mir::{MachineInst, MachineOperand};
    let (m, syms) = parse(src);
    let (pm, _, h) = super::avr::prepare::prepare(&m, &syms, &super::avr::Device::ATMEGA328P).unwrap();
    let mf = AvrTarget::new(true).with_helpers(h).select(&pm, FuncId::from_index(0));
    let insts: Vec<&MachineInst> = mf.block_ids().flat_map(|b| mf.block(b).insts.iter()).collect();
    let count = |ops: &[AvrOp], compact: bool| {
        insts.iter().filter(|i| ops.contains(&AvrOp::decode(i.opcode)) && avr_compact(i) == compact).count()
    };
    assert_eq!((count(&[AvrOp::SetCmp], false), count(&[AvrOp::SetCmp], true)), (1, 1), "compares");
    let shifts = [AvrOp::ShlV, AvrOp::LshrV];
    assert_eq!((count(&shifts, false), count(&shifts, true)), (1, 1), "variable shifts");
    // Only a sub-byte sign extension has two forms.
    let imm = |i: &MachineInst, k: usize| match &i.operands[k] {
        MachineOperand::Imm(v) => v.to_u64(),
        _ => None,
    };
    let sext4 = |compact: bool| {
        insts
            .iter()
            .filter(|i| AvrOp::decode(i.opcode) == AvrOp::Ext && imm(i, 3) == Some(1) && imm(i, 2) == Some(4))
            .filter(|i| avr_compact(i) == compact)
            .count()
    };
    assert_eq!((sext4(false), sext4(true)), (1, 1), "sign extensions");
}

/// On AVR, soft-float arithmetic, all division and multiplication wider than
/// 16 bits are calls to the runtime. As on Thumb, the verifier rejects the
/// float and division operations on secrets in the source, and the helper
/// calls take public parameters, so a secret reaching one — a secret 32- or
/// 64-bit multiply, which [`CtPolicy::DEFAULT`] allows — is a violation in
/// the prepared module: verify that (or use [`CtPolicy::STRICT`]) for AVR
/// code. An 8- or 16-bit multiply is `mul` on AVR5; on a core without it,
/// it is a call the prepared IR does not show, to a runtime helper that runs
/// a fixed number of branch-free iterations (the 32/64-bit helpers stop early,
/// but a secret never reaches them without a verifier violation).
#[test]
fn avr_helper_calls_are_rejected_on_secrets() {
    let dev = super::avr::Device::ATMEGA328P;
    let cases: [(&str, &str, &str, bool); 7] = [
        ("soft-float add", "f32", "  %r = fadd %x, %x : f32\n  %o = bitcast %r : i32\n  %w = zext %o : i64\n  ret %w\n", true),
        ("soft-float compare", "f32", "  %c = fcmp olt %x, %x : i1\n  %w = zext %c : i64\n  ret %w\n", true),
        ("soft-float conversion", "f32", "  %r = fptosi %x : i64\n  ret %r\n", true),
        ("16-bit division", "i16", "  %r = udiv i16 1000, %x : i16\n  %w = zext %r : i64\n  ret %w\n", true),
        ("32-bit remainder", "i32", "  %r = srem i32 1000, %x : i32\n  %w = sext %r : i64\n  ret %w\n", true),
        ("32-bit multiply", "i32", "  %r = mul %x, %x : i32\n  %w = zext %r : i64\n  ret %w\n", false),
        ("64-bit multiply", "i64", "  %r = mul %x, %x : i64\n  ret %r\n", false),
    ];
    for (what, ty, body, source_rejects) in cases {
        for secret in [false, true] {
            let kw = if secret { "secret " } else { "" };
            let src = format!("module \"b\"\nfunc @f({kw}{ty}) -> {kw}i64 {{\nentry ^0(%x: {ty}):\n{body}}}\n");
            let mut syms = StrInterner::new();
            let m = crate::ir::text::parse_module(&src, FileId::new(0), &mut syms)
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let f = FuncId::from_index(0);
            let v = ct_violations(&m, f, CtPolicy::DEFAULT);
            assert_eq!(!v.is_empty(), secret && source_rejects, "{what}: the source verdict");
            let (pm, _, _) = super::avr::prepare::prepare(&m, &syms, &dev).expect("prepares");
            let pv = ct_violations(&pm, f, CtPolicy::DEFAULT);
            assert_eq!(!pv.is_empty(), secret, "{what}: the prepared module's verdict");
        }
    }
    // The documented exception: a secret 16-bit multiply stays a `mul` in the
    // prepared IR (allowed), and on AVR5 it is the `mul` instruction.
    let src = "module \"m\"\nfunc @f(secret i16) -> secret i16 {\nentry ^0(%x: i16):\n  %r = mul %x, %x : i16\n  ret %r\n}\n";
    let (m, syms) = parse(src);
    let (pm, _, _) = super::avr::prepare::prepare(&m, &syms, &dev).unwrap();
    assert!(ct_violations(&pm, FuncId::from_index(0), CtPolicy::DEFAULT).is_empty());
    assert!(avr_cond_branches(&avr_text(&m, &syms)).is_empty());
}

/// On Thumb, soft-float arithmetic, division and 64-bit multiplication are
/// calls to run-time helpers or divide instructions. The verifier rejects the
/// floating-point and division operations on secrets in the source IR; and
/// every helper call the preparation introduces takes public parameters, so a
/// secret reaching one is a violation in the prepared module as well — which
/// is how a secret 64-bit multiply (allowed by [`CtPolicy::DEFAULT`], but an
/// `__aeabi_lmul` call on Thumb) is caught: verify the prepared module, or use
/// [`CtPolicy::STRICT`] for Cortex-M code.
#[test]
fn thumb_helper_calls_are_rejected_on_secrets() {
    let cases: [(&str, &str, &str, bool); 6] = [
        ("soft-float add", "f32", "  %r = fadd %x, %x : f32\n  %o = bitcast %r : i32\n  %w = zext %o : i64\n  ret %w\n", true),
        ("soft-float compare", "f64", "  %c = fcmp olt %x, %x : i1\n  %w = zext %c : i64\n  ret %w\n", true),
        ("soft-float conversion", "f64", "  %r = fptosi %x : i64\n  ret %r\n", true),
        ("32-bit division", "i32", "  %r = udiv i32 1000, %x : i32\n  %w = zext %r : i64\n  ret %w\n", true),
        ("64-bit remainder", "i64", "  %r = srem i64 1000, %x : i64\n  ret %r\n", true),
        ("64-bit multiply", "i64", "  %r = mul %x, %x : i64\n  ret %r\n", false),
    ];
    for (what, ty, body, source_rejects) in cases {
        for secret in [false, true] {
            let kw = if secret { "secret " } else { "" };
            let src = format!("module \"b\"\nfunc @f({kw}{ty}) -> {kw}i64 {{\nentry ^0(%x: {ty}):\n{body}}}\n");
            let mut syms = StrInterner::new();
            let m = crate::ir::text::parse_module(&src, FileId::new(0), &mut syms)
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let f = FuncId::from_index(0);
            let v = ct_violations(&m, f, CtPolicy::DEFAULT);
            assert_eq!(!v.is_empty(), secret && source_rejects, "{what}: the source verdict");
            for hw_div in [true, false] {
                let topts = super::thumb::ThumbOptions::default().with_hw_div(hw_div);
                let (pm, _) = super::thumb::prepare_module(&m, &syms, &topts).expect("prepares");
                let pv = ct_violations(&pm, f, CtPolicy::DEFAULT);
                assert_eq!(!pv.is_empty(), secret, "{what}: the prepared module's verdict");
            }
        }
    }
}

#[test]
fn branchy_lowerings_are_rejected_on_secrets() {
    // Each op's x86-64 lowering branches on its operand (or loops over it);
    // the verifier must reject a secret there. The public variant compiles
    // to MIR the audit flags, the secret variant is a violation.
    let cases: [(&str, &str, &str); 5] = [
        ("u64 to float", "i64", "  %r = uitofp %x : f64\n  %o = bitcast %r : i64\n  ret %o\n"),
        ("float to u64", "f64", "  %r = fptoui %x : i64\n  ret %r\n"),
        ("dyn_alloca", "i64", "  %r = dyn_alloca %x align 16 : ptr\n  %o = ptrtoint %r : i64\n  ret %o\n"),
        (
            "atomic rmw",
            "i64",
            "  %s = alloca i64 : ptr\n  %o = atomic_rmw umax seq_cst %s, %x align 8 : i64\n  ret %o\n",
        ),
        ("division", "i64", "  %r = udiv i64 1000, %x : i64\n  ret %r\n"),
    ];
    let x86 = X86_64Target::new();
    let a64 = AArch64Target::new();
    for (what, ty, body) in cases {
        for secret in [false, true] {
            let kw = if secret { "secret " } else { "" };
            let src = format!(
                "module \"b\"\nfunc @f({kw}{ty}) -> {kw}i64 {{\nentry ^0(%x: {ty}):\n{body}}}\n"
            );
            let mut syms = StrInterner::new();
            let m = crate::ir::text::parse_module(&src, FileId::new(0), &mut syms)
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let f = FuncId::from_index(0);
            let v = ct_violations(&m, f, CtPolicy::DEFAULT);
            if secret {
                assert!(!v.is_empty(), "{what}: a secret operand must be rejected");
            } else {
                assert!(v.is_empty(), "{what}: public is fine");
                if what != "division" {
                    let mf = x86.select_with_syms(&m, f, &syms);
                    let n = mir_branches(&mf, |mi| {
                        X86Op::decode(mi.opcode).may_branch_on_data(&mi.operands)
                    });
                    assert!(n > 0, "{what}: the x86-64 audit knows this lowering branches");
                }
                // AArch64 converts with single instructions; its `dyn_alloca`
                // probe loop and exclusive-monitor loops branch.
                if matches!(what, "dyn_alloca" | "atomic rmw") {
                    let mf = a64.select_with_syms(&m, f, &syms);
                    let n = mir_branches(&mf, |mi| A64Op::decode(mi.opcode).may_branch_on_data(&mi.operands));
                    assert!(n > 0, "{what}: the aarch64 audit knows this lowering branches");
                }
            }
        }
    }
}

#[test]
fn ladder_and_memcmp_isel_adds_no_branches() {
    for level in [OptLevel::O0, OptLevel::O2, OptLevel::O3] {
        let (mut m, syms) = parse(crate::transform::ct_tests::LADDER_LF);
        optimize(&mut m, level);
        verify_module(&m).unwrap_or_else(|d| panic!("{d:#?}"));
        assert_isel_adds_no_branches(&m, &syms, &format!("ladder/{level:?}"));
    }
}

// ---------------------------------------------------------------------------
// wasm32: lowered straight from the IR, so there is no MIR to audit; the
// emitted function bodies are decoded and scanned instead.
// ---------------------------------------------------------------------------

/// Secret narrow values (whose results the wasm backend masks) and a secret
/// `i128` (split into `i64` parts, variable shifts through a `select`
/// ladder): neither may introduce a branch.
const NARROW_WIDE_LF: &str = r#"
module "narrowwide"

func @narrow(secret i8, secret i16, secret i1, secret i24) -> secret i32 {
entry ^0(%a: i8, %b: i16, %c: i1, %d: i24):
  %x = add %a, i8 100 : i8
  %y = mul %b, i16 300 : i16
  %s = ashr %d, i24 3 : i24
  %xe = sext %x : i32
  %ye = zext %y : i32
  %se = sext %s : i32
  %t = add %xe, %ye : i32
  %u = sub %t, %se : i32
  %k = select %c, %u, %t : i32
  ret %k
}

func @wide(secret i128, secret i128, i128) -> secret i128 {
entry ^0(%a: i128, %b: i128, %p: i128):
  %s = add %a, %b : i128
  %d = sub %s, %p : i128
  %m = and %b, i128 127 : i128
  %x = shl %d, %m : i128
  %y = lshr %x, %m : i128
  %z = ashr %y, %m : i128
  %lt = icmp slt %z, %a : i1
  %eq = icmp eq %z, %b : i1
  %r1 = select %lt, %z, %s : i128
  %r2 = select %eq, %r1, %d : i128
  ret %r2
}
"#;

/// `src` parsed with the wasm32 layout, optimized at `level`, compiled, and
/// each defined function's wasm opcodes with its IR branch count.
fn wasm_bodies(src: &str, level: OptLevel) -> Vec<(String, Vec<u32>, usize)> {
    use crate::target::wasm32::binary::decode::opcodes;
    let (mut m, syms) = parse(src);
    m.set_data_layout(crate::target::wasm32::data_layout());
    optimize(&mut m, level);
    verify_module(&m).unwrap_or_else(|d| panic!("{d:#?}"));
    for f in (0..m.function_count()).map(FuncId::from_index) {
        let v = ct_violations(&m, f, CtPolicy::default());
        assert!(v.is_empty(), "the program is constant-time: {v:?}");
    }
    let c =crate::target::wasm32::compile(&m, &syms, &crate::codegen::CodegenOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    (0..m.function_count())
        .map(FuncId::from_index)
        .filter(|&f| !m.function(f).is_declaration())
        .map(|f| {
            let name = syms.resolve(m.function(f).name).to_owned();
            let ops = opcodes(&c.object.function_expr(&name));
            (name, ops, ir_branches(&m, f))
        })
        .collect()
}

#[test]
fn wasm32_straight_line_code_has_no_branches() {
    use crate::target::wasm32::binary::decode::is_branchy;
    for level in [OptLevel::O0, OptLevel::O2] {
        for src in [ALLOWED_LF, NARROW_WIDE_LF] {
            for (name, ops, ir) in wasm_bodies(src, level) {
                assert_eq!(ir, 0, "{name} is straight-line");
                let bad: Vec<u32> = ops.iter().copied().filter(|&o| is_branchy(o)).collect();
                assert!(bad.is_empty(), "{name} at {level:?}: branchy opcodes {bad:x?} in {ops:x?}");
                if name != "narrow" || level == OptLevel::O0 {
                    assert!(ops.contains(&0x1b), "{name} at {level:?}: the selects are wasm `select`s");
                }
            }
        }
    }
}

#[test]
fn wasm32_select_is_branchless() {
    let src = r#"module "sel"
func @sel(secret i1, secret i64, secret i64) -> secret i64 {
entry ^0(%c: i1, %a: i64, %b: i64):
  %r = select %c, %a, %b : i64
  ret %r
}
"#;
    let bodies = wasm_bodies(src, OptLevel::O0);
    let (_, ops, _) = &bodies[0];
    // local.get ×3 (the i1 masked on entry first), select, return.
    assert!(ops.contains(&0x1b), "{ops:x?}");
    assert!(!ops.iter().any(|&o| crate::target::wasm32::binary::decode::is_branchy(o)), "{ops:x?}");
}

/// The ladder and memcmp keep only their public loop control: every `if`,
/// `br_if` and `br_table` in the wasm comes from an IR `cond_br`/`switch`.
#[test]
fn wasm32_ladder_and_memcmp_add_no_branches() {
    for level in [OptLevel::O0, OptLevel::O2, OptLevel::O3] {
        for (name, ops, ir) in wasm_bodies(crate::transform::ct_tests::LADDER_LF, level) {
            let conds = ops.iter().filter(|&&o| matches!(o, 0x04 | 0x0d | 0x0e)).count();
            assert!(conds <= ir, "{name} at {level:?}: {conds} conditional branches for {ir} IR branches");
        }
    }
}

// ---------------------------------------------------------------------------
// Native execution on x86-64.
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use super::*;

    /// The Rust model of `@cswap` over two limbs.
    fn cswap(a: &mut [u64; 2], b: &mut [u64; 2], bit: u64) {
        let mask = 0u64.wrapping_sub(bit);
        for i in 0..2 {
            let d = (a[i] ^ b[i]) & mask;
            a[i] ^= d;
            b[i] ^= d;
        }
    }

    /// The Rust model of `@ladder`.
    fn ladder(r0: &mut [u64; 2], r1: &mut [u64; 2], k: u64, bits: u64) {
        let mut prev = 0;
        for j in (0..bits).rev() {
            let bit = (k >> j) & 1;
            cswap(r0, r1, bit ^ prev);
            let (a0, a1, b0, b1) = (r0[0], r0[1], r1[0], r1[1]);
            r1[0] = a0.wrapping_add(b0);
            r0[1] = a1.wrapping_mul(2);
            r1[1] = b0.wrapping_mul(b1) << 1;
            prev = bit;
        }
        cswap(r0, r1, prev);
    }

    /// `@main`: run the ladder on fixed inputs and two memcmps, and fold the
    /// declassified results into an exit code.
    fn main_lf(k: u64, bits: u64) -> String {
        format!(
            r#"
    global constant @s1 : [16 x i8] = [16 x i8] "constant-time!!!"

    global constant @s2 : [16 x i8] = [16 x i8] "constant-time!!!"

    global constant @s3 : [16 x i8] = [16 x i8] "constant-tame!!!"

    func @main() -> i64 {{
    entry ^0:
      %a = alloca [2 x i64] : ptr
      %b = alloca [2 x i64] : ptr
      %a1 = ptr_add %a, i64 8 : ptr
      %b1 = ptr_add %b, i64 8 : ptr
      store i64 3, %a align 8 : i64
      store i64 5, %a1 align 8 : i64
      store i64 7, %b align 8 : i64
      store i64 11, %b1 align 8 : i64
      call @ladder(%a, %b, i64 {k}, i64 {bits}) : void
      %x0 = load secret %a align 8 : i64
      %x1 = load secret %a1 align 8 : i64
      %y0 = load secret %b align 8 : i64
      %y1 = load secret %b1 align 8 : i64
      %m1 = mul %x1, i64 3 : i64
      %m2 = mul %y0, i64 5 : i64
      %m3 = mul %y1, i64 7 : i64
      %h1 = xor %x0, %m1 : i64
      %h2 = xor %h1, %m2 : i64
      %h3 = xor %h2, %m3 : i64
      %h4 = lshr %h3, i64 17 : i64
      %h5 = xor %h3, %h4 : i64
      %h = and %h5, i64 63 : i64
      %e = call @ct_memcmp(@s1, @s2, i64 16) : i64
      %d = call @ct_memcmp(@s1, @s3, i64 16) : i64
      %e6 = shl %e, i64 6 : i64
      %d7 = shl %d, i64 7 : i64
      %r1 = or %h, %e6 : i64
      %r2 = or %r1, %d7 : i64
      %r = declassify %r2 : i64
      ret %r
    }}
    "#
        )
    }

    fn run_native(m: &Module, syms: &StrInterner, tag: &str) -> i32 {
        use crate::link::{ImageOptions, link_executable, write_executable};
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let obj = crate::target::x86_64::compile_module(m, syms);
        let image = link_executable(vec![obj], &ImageOptions::default()).expect("link");
        let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("lf_ct_{tag}_{}_{uniq}", std::process::id()));
        write_executable(path.to_str().unwrap(), &image).expect("write executable");
        let status = loop {
            match std::process::Command::new(&path).status() {
                Ok(s) => break s,
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec: {e}"),
            }
        };
        let _ = std::fs::remove_file(&path);
        status.code().unwrap_or_else(|| panic!("{tag} died: {status:?}"))
    }

    #[test]
    fn montgomery_ladder_and_memcmp_run_correctly_at_o2() {
        for (k, bits) in [(0b1011_0110u64, 8u64), (0xDEAD_BEEF, 32), (1, 1), (0, 5)] {
            let src = format!("{}{}", crate::transform::ct_tests::LADDER_LF, main_lf(k, bits));
            let (mut r0, mut r1) = ([3u64, 5], [7u64, 11]);
            ladder(&mut r0, &mut r1, k, bits);
            let h3 = r0[0] ^ r0[1].wrapping_mul(3) ^ r1[0].wrapping_mul(5) ^ r1[1].wrapping_mul(7);
            let h = (h3 ^ (h3 >> 17)) & 63;
            let want = (h | (1 << 7)) as i32; // s1 == s2 (0 << 6), s1 != s3 (1 << 7)
            for level in [OptLevel::O0, OptLevel::O2] {
                let (mut m, syms) = parse(&src);
                optimize(&mut m, level);
                verify_module(&m).unwrap_or_else(|d| panic!("{d:#?}"));
                for i in 0..m.function_count() {
                    assert!(ct_violations(&m, FuncId::from_index(i), CtPolicy::DEFAULT).is_empty());
                }
                assert_isel_adds_no_branches(&m, &syms, &format!("ladder+main/{level:?}"));
                assert_eq!(run_native(&m, &syms, "ladder"), want, "k={k:#x} bits={bits} at {level:?}");
            }
        }
    }
}
