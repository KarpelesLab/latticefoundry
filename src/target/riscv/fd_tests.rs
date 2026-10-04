//! The F and D extensions and the LP64D convention: every new encoding
//! differentially checked against `llvm-mc`, every compiled function decoded
//! by `llvm-objdump`, and the argument placement of the psABI checked on the
//! selected MIR (which register each part of each argument lands in).

use super::encode::*;
use super::isel::RvOp;
use crate::codegen::mir::{MachineOperand, PReg, Reg, RegClass};
use crate::ir::FuncId;

use super::diff_tests::parse;

/// Assemble `asm` with `llvm-mc` for RV64IMAFD (`+c` too with `compressed`),
/// returning the concatenated encodings; `None` without `llvm-mc`.
pub(super) fn llvm_mc_with(asm: &str, compressed: bool) -> Option<Vec<u8>> {
    use std::io::Write;
    let attrs = if compressed { "-mattr=+m,+a,+f,+d,+c" } else { "-mattr=+m,+a,+f,+d" };
    let mut child = std::process::Command::new("llvm-mc")
        .args(["--triple=riscv64", attrs, "--show-encoding"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(asm.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut bytes = Vec::new();
    for part in text.split("encoding: [").skip(1) {
        let end = part.find(']')?;
        for tok in part[..end].split(',') {
            bytes.push(u8::from_str_radix(tok.trim().trim_start_matches("0x"), 16).ok()?);
        }
    }
    Some(bytes)
}

fn llvm_mc(asm: &str) -> Option<Vec<u8>> {
    llvm_mc_with(asm, false)
}

/// Every F/D instruction form the encoder emits, in both formats, against
/// `llvm-mc`.
#[test]
fn fd_encodings_match_llvm_mc() {
    if llvm_mc("ret").is_none() {
        eprintln!("skipping fd_encodings_match_llvm_mc: no llvm-mc");
        return;
    }
    let mut corpus: Vec<(u32, String)> = Vec::new();
    for (w, sfx) in [(32u32, "s"), (64, "d")] {
        for (f5, name) in [(0u32, "fadd"), (1, "fsub"), (2, "fmul"), (3, "fdiv")] {
            corpus.push((fp_op(f5, w, RM_DYN, 10, 11, 12), format!("{name}.{sfx} fa0, fa1, fa2")));
            corpus.push((fp_op(f5, w, RM_DYN, 31, 0, 27), format!("{name}.{sfx} ft11, ft0, fs11")));
        }
        for (k, name) in ["fmadd", "fmsub", "fnmsub", "fnmadd"].iter().enumerate() {
            corpus.push((fmadd(k as u32, w, 10, 11, 12, 13), format!("{name}.{sfx} fa0, fa1, fa2, fa3")));
            corpus.push((fmadd(k as u32, w, 28, 29, 30, 31), format!("{name}.{sfx} ft8, ft9, ft10, ft11")));
        }
        for (f3, name) in [(0u32, "fsgnj"), (1, "fsgnjn"), (2, "fsgnjx")] {
            corpus.push((fsgnj(w, f3, 10, 11, 12), format!("{name}.{sfx} fa0, fa1, fa2")));
        }
        for (f3, name) in [(2u32, "feq"), (1, "flt"), (0, "fle")] {
            corpus.push((fcmp(w, f3, 10, 11, 12), format!("{name}.{sfx} a0, fa1, fa2")));
            corpus.push((fcmp(w, f3, 5, 18, 9), format!("{name}.{sfx} t0, fs2, fs1")));
        }
        for (signed, iw, ik) in [(true, 32u32, "w"), (false, 32, "wu"), (true, 64, "l"), (false, 64, "lu")] {
            corpus.push((fcvt_int_from_float(signed, iw, w, 10, 11), format!("fcvt.{ik}.{sfx} a0, fa1, rtz")));
            corpus.push((fcvt_float_from_int(signed, iw, w, 10, 11), format!("fcvt.{sfx}.{ik} fa0, a1")));
        }
        let x = if w == 32 { "w" } else { "d" };
        corpus.push((fmv_x_f(w, 10, 11), format!("fmv.x.{x} a0, fa1")));
        corpus.push((fmv_f_x(w, 10, 11), format!("fmv.{x}.x fa0, a1")));
        corpus.push((fmv_f_x(w, 10, 0), format!("fmv.{x}.x fa0, zero")));
        let size = u64::from(w / 8);
        let (l, s) = if w == 32 { ("flw", "fsw") } else { ("fld", "fsd") };
        corpus.push((fload(size, 10, 2, 16), format!("{l} fa0, 16(sp)")));
        corpus.push((fload(size, 27, 11, -2048), format!("{l} fs11, -2048(a1)")));
        corpus.push((fstore(size, 10, 2, 2040), format!("{s} fa0, 2040(sp)")));
        corpus.push((fstore(size, 8, 31, -8), format!("{s} fs0, -8(t6)")));
    }
    corpus.push((fcvt_ff(64, 32, 10, 11), "fcvt.d.s fa0, fa1".into()));
    corpus.push((fcvt_ff(32, 64, 10, 11), "fcvt.s.d fa0, fa1".into()));
    for (word, asm) in &corpus {
        let want = llvm_mc(asm).unwrap_or_else(|| panic!("llvm-mc failed on `{asm}`"));
        assert_eq!(word.to_le_bytes().to_vec(), want, "`{asm}`: ours {:02x?}", word.to_le_bytes());
    }
    eprintln!("F/D encodings: {} instructions matched llvm-mc", corpus.len());
    assert!(corpus.len() >= 80);
}

/// Disassemble `bytes` (raw RV64IMAFD code) with `llvm-objdump`.
pub(super) fn objdump(bytes: &[u8], compressed: bool) -> Option<String> {
    use std::io::Write;
    let path = std::env::temp_dir().join(format!("lf_rvfd_{}_{}.bin", std::process::id(), bytes.len()));
    std::fs::File::create(&path).ok()?.write_all(bytes).ok()?;
    let attrs = if compressed { "--mattr=+m,+a,+f,+d,+c" } else { "--mattr=+m,+a,+f,+d" };
    let out = std::process::Command::new("llvm-objdump")
        .args(["-D", "--triple=riscv64", attrs, "-b", "binary", "-m", "riscv"])
        .arg(&path)
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&path);
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// A module exercising every float lowering at once.
pub(super) const FLOAT_KITCHEN: &str = r#"
module "kitchen"
global @g : f32 = f32 0x40490fdb
func @k(f64, f32, i64, i32) -> f64 {
entry ^0(%a: f64, %b: f32, %n: i64, %m: i32):
  %e = fpext %b : f64
  %s = fadd %a, %e : f64
  %t = fsub %s, f64 0x3ff8000000000000 : f64
  %u = fmul %t, %a : f64
  %v = fdiv %u, %e : f64
  %w = frem %v, %a : f64
  %ng = fneg %w : f64
  %c = fcmp ueq %ng, %a : i1
  %o = fcmp one %ng, %a : i1
  %sl = select %c, %ng, %a : f64
  %i = fptosi %sl : i64
  %j = fptoui %b : i32
  %k = sitofp %n : f64
  %l = uitofp %m : f32
  %x = fadd %l, %b : f32
  %y = frem %x, %b : f32
  %gv = load @g align 4 : f32
  %z = fmul %y, %gv : f32
  store %z, @g align 4 : f32
  %zz = fptrunc %k : f32
  %bits = bitcast %zz : i32
  %bb = bitcast %n : f64
  %q = fadd %bb, %k : f64
  %fi = sitofp %i : f64
  %fj = uitofp %j : f64
  %r1 = fadd %q, %fi : f64
  %r2 = fadd %r1, %fj : f64
  %ob = zext %o : i64
  %bz = zext %bits : i64
  %t2 = add %ob, %bz : i64
  %ft = sitofp %t2 : f64
  %r3 = fadd %r2, %ft : f64
  ret %r3
}
"#;

/// Every compiled function decodes completely as RV64IMAFD.
#[test]
fn compiled_float_code_decodes_with_llvm_objdump() {
    let (m, syms) = parse(FLOAT_KITCHEN);
    let obj = super::compile_module(&m, &syms);
    let text = &obj.sections()[0].bytes;
    let Some(dis) = objdump(text, false) else {
        eprintln!("skipping compiled_float_code_decodes_with_llvm_objdump: no llvm-objdump");
        return;
    };
    assert!(!dis.contains("unknown") && !dis.contains("<invalid"), "{dis}");
    for needle in ["fadd.d", "fsub.d", "fmul.d", "fdiv.d", "fneg.d", "feq.d", "flt.d", "fcvt.d.s", "fcvt.l.d", "fcvt.wu.s", "fcvt.d.l", "fcvt.s.wu", "fmv.x.d", "fmv.d.x", "flw", "fsw", "fcvt.s.d"] {
        assert!(dis.contains(needle), "`{needle}` missing:\n{dis}");
    }
}

// ===========================================================================
// LP64D argument placement, on the selected MIR
// ===========================================================================

/// The physical argument registers a function's `Call` uses (in operand
/// order) and the stack stores before it (`LeaSp` offsets).
fn call_uses(src: &str, caller: &str) -> (Vec<PReg>, Vec<u64>) {
    let (m, syms) = parse(src);
    let idx = m.functions().position(|f| syms.resolve(f.name) == caller).unwrap();
    let t = super::RiscvTarget::for_module(&m, Some(&syms), &crate::codegen::CodegenOptions::default());
    let mf = t.select(&m, FuncId::from_index(idx));
    let insts: Vec<_> = mf.block_ids().flat_map(|b| mf.block(b).insts.clone()).collect();
    let call = insts.iter().find(|i| RvOp::decode(i.opcode) == RvOp::Call).expect("a call");
    let uses = call
        .operands
        .iter()
        .skip(1)
        .filter_map(|o| match o {
            MachineOperand::Use(Reg::Physical(p)) => Some(*p),
            _ => None,
        })
        .collect();
    let stack = insts
        .iter()
        .filter(|i| RvOp::decode(i.opcode) == RvOp::LeaSp)
        .map(|i| match &i.operands[1] {
            MachineOperand::Imm(v) => v.to_u64().unwrap(),
            _ => unreachable!(),
        })
        .collect();
    (uses, stack)
}

fn x(n: u16) -> PReg {
    PReg::new(RegClass::Gpr, n)
}
fn f(n: u16) -> PReg {
    PReg::new(RegClass::Fp, n)
}

/// The psABI's floating-point convention at call sites: floats in `fa*`,
/// then integer registers, then the stack; integers independently in `a*`;
/// structs flattened per their fields; variadic floats in integer registers.
#[test]
fn call_sites_place_arguments_per_the_psabi() {
    // Nine doubles and two longs: fa0-fa7, then a0 (the ninth double), a1-a2.
    let src = "module \"m\"\nfunc @c(f64, f64, f64, f64, f64, f64, f64, f64, f64, i64, i64) -> void\n\
        func @t(f64, i64) -> void {\nentry ^0(%d: f64, %i: i64):\n  \
        call @c(%d, %d, %d, %d, %d, %d, %d, %d, %d, %i, %i) : void\n  ret\n}\n";
    let (uses, stack) = call_uses(src, "t");
    let mut want: Vec<PReg> = (10..18).map(f).collect();
    want.extend([x(10), x(11), x(12)]);
    assert_eq!(uses, want);
    assert!(stack.is_empty());

    // Eight longs then a float and a long: the float still gets fa0, the
    // ninth long goes to the stack.
    let src = "module \"m\"\nfunc @c(i64, i64, i64, i64, i64, i64, i64, i64, f32, i64) -> void\n\
        func @t(i64, f32) -> void {\nentry ^0(%i: i64, %s: f32):\n  \
        call @c(%i, %i, %i, %i, %i, %i, %i, %i, %s, %i) : void\n  ret\n}\n";
    let (uses, stack) = call_uses(src, "t");
    let mut want: Vec<PReg> = (10..18).map(x).collect();
    want.push(f(10));
    assert_eq!(uses, want);
    assert_eq!(stack, [0]);

    // Structs: { float, int } → fa0 + a0; { double, double } → fa1, fa2;
    // { int, double } → a1 + fa3; { float, float, float } → a2 (8 bytes) +
    // a3; { long, long, long } → a pointer in a4.
    let src = "module \"m\"\nfunc @c({f32, i32}, {f64, f64}, {i32, f64}, {f32, f32, f32}, {i64, i64, i64}) -> void\n\
        func @t() -> void {\nentry ^0:\n  %a = alloca {f32, i32} : ptr\n  %b = alloca {f64, f64} : ptr\n  \
        %c = alloca {i32, f64} : ptr\n  %d = alloca {f32, f32, f32} : ptr\n  %e = alloca {i64, i64, i64} : ptr\n  \
        call @c(%a, %b, %c, %d, %e) : void\n  ret\n}\n";
    let (uses, stack) = call_uses(src, "t");
    assert_eq!(uses, [f(10), x(10), f(11), f(12), x(11), f(13), x(12), x(13), x(14)]);
    assert!(stack.is_empty());

    // A variadic callee: the named double in fa0, the variadic ones in
    // integer registers.
    let src = "module \"m\"\nfunc @v(f64, ...) -> void\n\
        func @t(f64) -> void {\nentry ^0(%d: f64):\n  call @v(%d, %d, %d) : void\n  ret\n}\n";
    let (uses, _) = call_uses(src, "t");
    assert_eq!(uses, [f(10), x(10), x(11)]);

    // A struct returned in memory takes a0 for its address.
    let src = "module \"m\"\nfunc @r(i64) -> {i64, i64, i64}\n\
        func @t(i64) -> i64 {\nentry ^0(%i: i64):\n  %s = call @r(%i) : {i64, i64, i64}\n  ret i64 0\n}\n";
    let (uses, _) = call_uses(src, "t");
    assert_eq!(uses, [x(10), x(11)]);
}
