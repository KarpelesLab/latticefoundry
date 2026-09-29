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
//!   mask blend on RISC-V.
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
use super::riscv::{RiscvTarget, RvOp};
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
    assert!(!rv_is_cond_branch(0x0000_006F)); // jal x0, 0
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
    assert!(a.contains(&A64Op::Csel.opcode().0));
    // RISC-V selects with the branchless `f ^ ((t ^ f) & -c)`.
    let r = ops(&RiscvTarget::new().select(&m, f));
    assert!(r.contains(&RvOp::Xor.opcode().0) && r.contains(&RvOp::And.opcode().0));
    assert!(!r.contains(&RvOp::BrCond.opcode().0));
    assert_isel_adds_no_branches(&m, &syms, "sel");
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
