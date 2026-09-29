//! Execution and encoding tests for **SSE2 vectors** on x86-64
//! (`docs/ir-design.md` §6c).
//!
//! Vector programs are compiled at `-O0` and `-O2` by our backend, linked by
//! our static linker, run on the bare kernel, and every result is compared with
//! the reference executor's result on the *unoptimized* IR. Machine-code shape
//! checks pin the SSE2 lowering (and that ops without an SSE2 form are
//! scalarized), the encoder is diffed against `llvm-mc`, and a vector crosses
//! the System V ABI to and from gcc-compiled C using `__m128i`.

use crate::codegen::legalize::legalized;
use crate::codegen::mir::MachineFunction;
use crate::ir::{FuncId, Module};
use crate::link::{ImageOptions, link_executable, write_executable};
use crate::support::StrInterner;
use crate::target::vector_fixtures::{
    FLOAT_SRC, INPUTS, INT_OPS, cases, compare_src, int_arith_src, lanes_src,
    Case, Rng, assert_matches, parse, random_inputs, random_program, reference, with_stdout_main,
};
use crate::transform::pipeline::{OptLevel, optimize};

use super::isel::{Sse2Legality, X86Op, X86_64Target};

const LEVELS: [OptLevel; 2] = [OptLevel::O0, OptLevel::O2];

/// Compile + link `m`, run it, and return its stdout as `i64` results.
fn run_native(m: &Module, syms: &StrInterner, tag: &str) -> Vec<u64> {
    let obj = super::compile_module(m, syms);
    let image = link_executable(vec![obj], &ImageOptions::default()).expect("link should succeed");
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("lf_vec_{tag}_{}_{uniq}", std::process::id()));
    write_executable(path.to_str().unwrap(), &image).expect("write executable");
    let out = loop {
        match std::process::Command::new(&path).stdout(std::process::Stdio::piped()).output() {
            Ok(o) => break o,
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
            Err(e) => panic!("exec our native binary: {e}"),
        }
    };
    let _ = std::fs::remove_file(&path);
    assert_eq!(out.status.code(), Some(0), "{tag}: the program must exit 0 ({:?})", out.status);
    out.stdout.chunks(8).map(|c| u64::from_le_bytes(c.try_into().expect("8-byte result"))).collect()
}

/// Run every case of `src` natively at each level and compare with the
/// reference results.
fn check_native(src: &str, cases: &[Case], tag: &str) {
    let want = reference(src, cases);
    let full = with_stdout_main(src, cases);
    for level in LEVELS {
        let (mut m, syms) = parse(&full);
        optimize(&mut m, level);
        crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify after {level:?}: {e:?}"));
        let got = run_native(&m, &syms, tag);
        assert_matches(&format!("{tag} at {level:?}"), cases, &got, &want);
    }
}

#[test]
fn integer_vector_arithmetic_matches_the_reference() {
    let src = int_arith_src();
    let names: Vec<String> = ["i8", "i16", "i32", "i64"]
        .iter()
        .flat_map(|t| INT_OPS.map(|o| format!("{o}_{t}")))
        .collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    check_native(&src, &cases(&refs, &INPUTS), "int");
}

/// Every integer and float compare predicate, blended by `select`, plus mask
/// logic and mask <-> integer casts.
#[test]
fn vector_compares_and_masks_match_the_reference() {
    let src = compare_src();
    let (m, syms) = parse(&src);
    let names: Vec<String> = m.functions().map(|f| syms.resolve(f.name).to_owned()).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    check_native(&src, &cases(&refs, &INPUTS), "cmp");
}

#[test]
fn float_vector_arithmetic_matches_the_reference() {
    check_native(FLOAT_SRC, &cases(&["farith"], &INPUTS), "flt");
}

#[test]
fn lane_moves_match_the_reference() {
    let src = lanes_src();
    let (m, syms) = parse(&src);
    let names: Vec<String> = m.functions().map(|f| syms.resolve(f.name).to_owned()).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    check_native(&src, &cases(&refs, &INPUTS[1..4]), "lane");
}

/// Illegal types (256-bit and 64-bit vectors, `<3 x i32>`), reductions, memory
/// (aligned and unaligned), calls passing/returning vectors (including more
/// than eight vector arguments, so some go on the stack), and vectors live
/// across calls (spilled as 16 bytes).
const MIXED_SRC: &str = r#"
module "vmix"
global @mem : [3 x <16 x i8>] = [3 x <16 x i8>] (<16 x i8> (i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16), <16 x i8> (i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31, i8 32), <16 x i8> (i8 33, i8 34, i8 35, i8 36, i8 37, i8 38, i8 39, i8 40, i8 41, i8 42, i8 43, i8 44, i8 45, i8 46, i8 47, i8 48))

func @many(<4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>) -> <4 x i32> {
entry ^0(%v0: <4 x i32>, %v1: <4 x i32>, %v2: <4 x i32>, %v3: <4 x i32>, %v4: <4 x i32>, %v5: <4 x i32>, %v6: <4 x i32>, %v7: <4 x i32>, %v8: <4 x i32>, %v9: <4 x i32>):
  %s1 = add %v0, %v1 : <4 x i32>
  %s2 = sub %s1, %v2 : <4 x i32>
  %s3 = xor %s2, %v3 : <4 x i32>
  %s4 = add %s3, %v4 : <4 x i32>
  %s5 = mul %s4, %v5 : <4 x i32>
  %s6 = add %s5, %v6 : <4 x i32>
  %s7 = or %s6, %v7 : <4 x i32>
  %s8 = sub %s7, %v8 : <4 x i32>
  %s9 = add %s8, %v9 : <4 x i32>
  ret %s9
}

func @wide(<8 x i32>, <8 x i32>) -> <8 x i32> {
entry ^0(%x: <8 x i32>, %y: <8 x i32>):
  %m = mul %x, %y : <8 x i32>
  %s = shufflevector %m, %x, [8, 1, 10, 3, 12, 5, 14, 7] : <8 x i32>
  ret %s
}

func @narrow(<2 x i32>, <3 x i32>) -> <2 x i32> {
entry ^0(%x: <2 x i32>, %y: <3 x i32>):
  %e = extractelement %y, 2 : i32
  %s = splat %e : <2 x i32>
  %r = add %x, %s : <2 x i32>
  ret %r
}

func @mixed(i64, i64, i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64):
  %p0 = insertelement <2 x i64> poison, %a, 0 : <2 x i64>
  %x0 = insertelement %p0, %b, 1 : <2 x i64>
  %p1 = insertelement <2 x i64> poison, %c, 0 : <2 x i64>
  %y0 = insertelement %p1, %d, 1 : <2 x i64>
  %x = bitcast %x0 : <4 x i32>
  %y = bitcast %y0 : <4 x i32>
  %k = call @many(%x, %y, %x, %y, %x, %y, %x, %y, %x, %y) : <4 x i32>
  %w1 = shufflevector %x, %y, [0, 1, 2, 3, 4, 5, 6, 7] : <8 x i32>
  %w2 = shufflevector %y, %k, [4, 5, 6, 7, 0, 1, 2, 3] : <8 x i32>
  %w = call @wide(%w1, %w2) : <8 x i32>
  %ra = reduce add %w : i32
  %rx = reduce smax %w : i32
  %n2 = shufflevector %k, %k, [3, 0] : <2 x i32>
  %n3 = shufflevector %x, %x, [1, 2, 3] : <3 x i32>
  %n = call @narrow(%n2, %n3) : <2 x i32>
  %nw = bitcast %n : i64
  %u1 = load @mem align 16 : <16 x i8>
  %mp = ptr_add @mem, i64 3 : ptr
  %u2 = load %mp align 1 : <16 x i8>
  %u3 = add %u1, %u2 : <16 x i8>
  %buf = alloca [3 x <16 x i8>] : ptr
  %sp = ptr_add %buf, i64 17 : ptr
  store %u3, %sp align 1 : <16 x i8>
  %sp2 = ptr_add %buf, i64 32 : ptr
  %xb = bitcast %x : <16 x i8>
  store %xb, %sp2 align 16 : <16 x i8>
  %u4 = load %sp align 1 : <16 x i8>
  %u5 = load %sp2 align 16 : <16 x i8>
  %u6 = xor %u4, %u5 : <16 x i8>
  %ru = reduce add %u6 : i8
  %kf = bitcast %k : <2 x i64>
  %k0 = extractelement %kf, 0 : i64
  %k1 = extractelement %kf, 1 : i64
  %ra64 = zext %ra : i64
  %rx64 = sext %rx : i64
  %ru64 = zext %ru : i64
  %t1 = xor %k0, %k1 : i64
  %t2 = add %t1, %ra64 : i64
  %t3 = mul %t2, i64 31 : i64
  %t4 = add %t3, %rx64 : i64
  %t5 = xor %t4, %nw : i64
  %t6 = mul %t5, i64 17 : i64
  %t7 = add %t6, %ru64 : i64
  ret %t7
}

func @id(i64) -> i64 {
entry ^0(%x: i64):
  ret %x
}

; Many GPR values live across calls force callee-saved pushes (an odd number of
; them misaligned 16-byte frame slots before the frame-layout fix).
func @framed(i64, i64, i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64):
  %slot = alloca <4 x i32> : ptr
  %e = add %a, i64 1 : i64
  %f = add %b, i64 2 : i64
  %g = add %c, i64 3 : i64
  %h = add %d, i64 4 : i64
  %i = xor %a, %d : i64
  %z = call @id(%a) : i64
  %p0 = insertelement <2 x i64> poison, %z, 0 : <2 x i64>
  %x0 = insertelement %p0, %b, 1 : <2 x i64>
  %v = bitcast %x0 : <4 x i32>
  store %v, %slot align 16 : <4 x i32>
  %w = load %slot align 16 : <4 x i32>
  %w2 = add %w, %v : <4 x i32>
  %q = bitcast %w2 : <2 x i64>
  %l = extractelement %q, 0 : i64
  %s1 = add %l, %e : i64
  %s2 = add %s1, %f : i64
  %s3 = add %s2, %g : i64
  %s4 = add %s3, %h : i64
  %s5 = xor %s4, %i : i64
  ret %s5
}

func @fred(i64, i64, i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64):
  %p0 = insertelement <2 x i64> poison, %a, 0 : <2 x i64>
  %x0 = insertelement %p0, %b, 1 : <2 x i64>
  %i = bitcast %x0 : <4 x i32>
  %small = and %i, <4 x i32> (i32 4095, i32 4095, i32 4095, i32 4095) : <4 x i32>
  %f = sitofp %small : <4 x f32>
  %s = reduce fadd %f : f32
  %p = reduce fmul %f : f32
  %si = bitcast %s : i32
  %pi = bitcast %p : i32
  %sw = zext %si : i64
  %pw = zext %pi : i64
  %m = shl %pw, i64 32 : i64
  %r = or %m, %sw : i64
  ret %r
}
"#;

#[test]
fn illegal_types_reductions_memory_and_calls_match_the_reference() {
    check_native(MIXED_SRC, &cases(&["mixed", "fred", "framed"], &INPUTS), "mix");
}

#[test]
fn random_vector_programs_match_the_reference() {
    // Several independent random programs, each with several functions, run
    // on several random inputs.
    let mut rng = Rng(0xd1ff_5eed);
    for p in 0..6u64 {
        let (src, names) = random_program(0x1000 + p, 6, 10, true);
        let mut cs = Vec::new();
        for n in &names {
            for _ in 0..4 {
                cs.push((n.clone(), random_inputs(&mut rng)));
            }
        }
        check_native(&src, &cs, &format!("rand{p}"));
    }
}

// ---------------------------------------------------------------------------
// Instruction-selection shape
// ---------------------------------------------------------------------------

/// The opcodes of function `name` after SSE2 legalization and isel.
fn mir_ops(src: &str, name: &str) -> Vec<X86Op> {
    let (m, syms) = parse(src);
    let legal = legalized(&m, &Sse2Legality);
    let idx = legal.functions().position(|f| syms.resolve(f.name) == name).expect("function");
    let mf: MachineFunction = X86_64Target::new().select_with_syms(&legal, FuncId::from_index(idx), &syms);
    mf.block_ids().flat_map(|b| mf.block(b).insts.iter().map(|i| X86Op::decode(i.opcode)).collect::<Vec<_>>()).collect()
}

#[test]
fn legal_ops_select_sse2_and_the_rest_is_scalarized() {
    let src = int_arith_src();
    let count = |ops: &[X86Op], op: X86Op| ops.iter().filter(|&&o| o == op).count();
    // Every test function folds its result with one scalar `imul`.
    let base = count(&mir_ops(&src, "and_i32"), X86Op::Imul);
    // add <4 x i32>: one packed op, no scalar add.
    let ops = mir_ops(&src, "add_i32");
    assert!(ops.contains(&X86Op::VOp) && !ops.contains(&X86Op::Add), "{ops:?}");
    // mul <4 x i32> without SSE4.1: pmuludq-based, no scalar imul.
    let ops = mir_ops(&src, "mul_i32");
    assert!(ops.contains(&X86Op::VOp) && count(&ops, X86Op::Imul) == base, "{ops:?}");
    // A uniform constant shift is one psll/psrl/psra.
    assert!(mir_ops(&src, "shl_i16").contains(&X86Op::VShiftI));
    // udiv has no SSE2 form: scalarized to one `div` per lane.
    let ops = mir_ops(&src, "udiv_i32");
    assert_eq!(ops.iter().filter(|&&o| o == X86Op::Div).count(), 4, "{ops:?}");
    // mul <16 x i8> has no SSE2 form (no pmullb): 16 scalar multiplies.
    let ops = mir_ops(&src, "mul_i8");
    assert_eq!(count(&ops, X86Op::Imul), base + 16, "{ops:?}");
    // A byte shift has no SSE2 form either.
    assert!(!mir_ops(&src, "shl_i8").contains(&X86Op::VShiftI));
    // pminub / paddsw are direct; smin <4 x i32> is a vector compare + blend
    // (no scalar code); signed i64 compares have no SSE2 form (scalarized).
    for f in ["umin_i8", "sadd_sat_i16", "smin_i32", "uadd_sat_i32"] {
        let ops = mir_ops(&src, f);
        assert!(count(&ops, X86Op::SetccCmp) == 0 && count(&ops, X86Op::Cmovne) == 0, "{f}: {ops:?}");
    }
    assert!(count(&mir_ops(&src, "smin_i64"), X86Op::SetccCmp) == 2);
}

// ---------------------------------------------------------------------------
// Encoding differential vs llvm-mc
// ---------------------------------------------------------------------------

/// Assemble AT&T lines with `llvm-mc` and return the concatenated bytes, or
/// `None` if `llvm-mc` is unavailable.
fn llvm_mc(lines: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let mut child = std::process::Command::new("llvm-mc")
        .args(["--triple=x86_64", "--show-encoding"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(lines.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut bytes = Vec::new();
    for line in text.lines() {
        let Some(at) = line.find("encoding: [") else { continue };
        let rest = &line[at + "encoding: [".len()..];
        let end = rest.find(']')?;
        for tok in rest[..end].split(',') {
            bytes.push(u8::from_str_radix(tok.trim().trim_start_matches("0x"), 16).ok()?);
        }
    }
    Some(bytes)
}

/// Encode one MIR instruction over physical registers.
fn encode_one(op: X86Op, operands: Vec<crate::codegen::mir::MachineOperand>) -> Vec<u8> {
    use crate::codegen::mir::{MachineFunction, MachineInst};
    let mut mf = MachineFunction::new("t", 0);
    let b = mf.add_block();
    mf.set_entry(b);
    mf.block_mut(b).insts.push(MachineInst::new(op.opcode(), operands));
    let target = X86_64Target::new();
    let layout = super::encode::layout_frame(&mf, &target);
    let name = |_: u32| String::from("f");
    super::encode::encode_function(&mf, &layout, &name, &name).bytes
}

#[test]
fn sse2_encodings_match_llvm_mc() {
    use super::isel::vector::VEnc;
    use crate::codegen::mir::{MachineOperand, Reg};
    use crate::codegen::mir::PReg;
    use crate::codegen::mir::RegClass;
    if llvm_mc("pxor %xmm0, %xmm0\n").is_none() {
        eprintln!("skipping sse2_encodings_match_llvm_mc: no llvm-mc");
        return;
    }
    let x = |n: u16| PReg::new(RegClass::Fp, n);
    let g = |n: u16| PReg::new(RegClass::Gpr, n);
    let d = |p: PReg| MachineOperand::Def(Reg::Physical(p));
    let u = |p: PReg| MachineOperand::Use(Reg::Physical(p));
    let i = |v: u64| MachineOperand::Imm(puremp::Int::from_u64(v));
    type C = (X86Op, Vec<MachineOperand>, &'static str);
    let mut cases: Vec<C> = Vec::new();
    // Every packed two-operand op the lowering uses, on low and high registers.
    let two: [(u8, u8, &str); 36] = [
        (0x66, 0xFC, "paddb"), (0x66, 0xFD, "paddw"), (0x66, 0xFE, "paddd"), (0x66, 0xD4, "paddq"),
        (0x66, 0xF8, "psubb"), (0x66, 0xF9, "psubw"), (0x66, 0xFA, "psubd"), (0x66, 0xFB, "psubq"),
        (0x66, 0xD5, "pmullw"), (0x66, 0xF4, "pmuludq"), (0x66, 0xDB, "pand"), (0x66, 0xDF, "pandn"),
        (0x66, 0xEB, "por"), (0x66, 0xEF, "pxor"), (0x66, 0x74, "pcmpeqb"), (0x66, 0x75, "pcmpeqw"),
        (0x66, 0x76, "pcmpeqd"), (0x66, 0x64, "pcmpgtb"), (0x66, 0x65, "pcmpgtw"), (0x66, 0x66, "pcmpgtd"),
        (0x66, 0x62, "punpckldq"), (0x66, 0x6C, "punpcklqdq"), (0x00, 0x58, "addps"), (0x66, 0x58, "addpd"),
        (0x00, 0x5C, "subps"), (0x66, 0x5C, "subpd"), (0x00, 0x59, "mulps"), (0x66, 0x59, "mulpd"),
        (0x00, 0x5E, "divps"), (0x66, 0x5E, "divpd"), (0x00, 0x57, "xorps"), (0x66, 0x57, "xorpd"),
        (0xF2, 0x10, "movsd"), (0x66, 0x14, "unpcklpd"), (0x00, 0x5B, "cvtdq2ps"), (0xF3, 0x5B, "cvttps2dq"),
    ];
    let mut exps: Vec<String> = Vec::new();
    for (k, &(pfx, opc, mn)) in two.iter().enumerate() {
        let (a, b) = if k % 2 == 0 { (3, 9) } else { (12, 1) };
        if opc == 0x5B {
            cases.push((X86Op::VUnary, vec![d(x(a)), u(x(b)), i(VEnc::op(pfx, opc, false))], ""));
        } else {
            // d == a, so the op is emitted alone.
            cases.push((X86Op::VOp, vec![d(x(a)), u(x(a)), u(x(b)), i(VEnc::op(pfx, opc, false))], ""));
        }
        exps.push(format!("{mn} %xmm{b}, %xmm{a}\n"));
    }
    // Min/max and saturating forms.
    for (k, (opc, mn)) in [
        (0xDAu8, "pminub"), (0xDE, "pmaxub"), (0xEA, "pminsw"), (0xEE, "pmaxsw"), (0xEC, "paddsb"),
        (0xED, "paddsw"), (0xDC, "paddusb"), (0xDD, "paddusw"), (0xE8, "psubsb"), (0xE9, "psubsw"),
        (0xD8, "psubusb"), (0xD9, "psubusw"),
    ]
    .into_iter()
    .enumerate()
    {
        let (a, b) = if k % 2 == 0 { (14u16, 2u16) } else { (5, 10) };
        cases.push((X86Op::VOp, vec![d(x(a)), u(x(a)), u(x(b)), i(VEnc::op(0x66, opc, true))], ""));
        exps.push(format!("{mn} %xmm{b}, %xmm{a}\n"));
    }
    // Immediate forms.
    cases.push((X86Op::VOp, vec![d(x(2)), u(x(2)), u(x(10)), i(VEnc::op_imm(0x00, 0xC2, 5))], ""));
    exps.push(String::from("cmpnltps %xmm10, %xmm2\n"));
    cases.push((X86Op::VOp, vec![d(x(11)), u(x(11)), u(x(4)), i(VEnc::op_imm(0x66, 0xC2, 3))], ""));
    exps.push(String::from("cmpunordpd %xmm4, %xmm11\n"));
    cases.push((X86Op::VOp, vec![d(x(0)), u(x(0)), u(x(7)), i(VEnc::op_imm(0x00, 0xC6, 0x1B))], ""));
    exps.push(String::from("shufps $0x1b, %xmm7, %xmm0\n"));
    cases.push((X86Op::VOp, vec![d(x(8)), u(x(8)), u(x(9)), i(VEnc::op_imm(0x66, 0xC6, 1))], ""));
    exps.push(String::from("shufpd $1, %xmm9, %xmm8\n"));
    cases.push((X86Op::VUnary, vec![d(x(5)), u(x(13)), i(VEnc::op_imm(0x66, 0x70, 0xB1))], ""));
    exps.push(String::from("pshufd $0xb1, %xmm13, %xmm5\n"));
    // Shifts by an immediate.
    for (opc, ext, cnt, mn, reg) in [(0x71u8, 6u8, 3u8, "psllw", 1u16), (0x72, 2, 31, "psrld", 9), (0x73, 6, 63, "psllq", 4), (0x71, 4, 15, "psraw", 12), (0x72, 4, 7, "psrad", 0), (0x73, 2, 1, "psrlq", 15)] {
        let enc = u64::from(opc) | (u64::from(ext) << 8) | (u64::from(cnt) << 16);
        cases.push((X86Op::VShiftI, vec![d(x(reg)), u(x(reg)), i(enc)], ""));
        exps.push(format!("{mn} ${cnt}, %xmm{reg}\n"));
    }
    // Moves between files, word insert/extract, loads/stores, constants.
    cases.push((X86Op::MovGprToX, vec![d(x(3)), u(g(9)), i(0)], ""));
    exps.push(String::from("movd %r9d, %xmm3\n"));
    cases.push((X86Op::MovGprToX, vec![d(x(10)), u(g(0)), i(1)], ""));
    exps.push(String::from("movq %rax, %xmm10\n"));
    cases.push((X86Op::MovXToGpr, vec![d(g(12)), u(x(2)), i(0)], ""));
    exps.push(String::from("movd %xmm2, %r12d\n"));
    cases.push((X86Op::MovXToGpr, vec![d(g(1)), u(x(14)), i(1)], ""));
    exps.push(String::from("movq %xmm14, %rcx\n"));
    cases.push((X86Op::Pinsrw, vec![d(x(6)), u(x(6)), u(g(10)), i(5)], ""));
    exps.push(String::from("pinsrw $5, %r10d, %xmm6\n"));
    cases.push((X86Op::Pextrw, vec![d(g(2)), u(x(11)), i(7)], ""));
    exps.push(String::from("pextrw $7, %xmm11, %edx\n"));
    cases.push((X86Op::VLoad, vec![d(x(9)), u(g(4)), i(0)], ""));
    exps.push(String::from("movdqu (%rsp), %xmm9\n"));
    cases.push((X86Op::VLoad, vec![d(x(1)), u(g(13)), i(1)], ""));
    exps.push(String::from("movdqa (%r13), %xmm1\n"));
    cases.push((X86Op::VStore, vec![u(g(7)), u(x(12)), i(0)], ""));
    exps.push(String::from("movdqu %xmm12, (%rdi)\n"));
    cases.push((X86Op::VStore, vec![u(g(0)), u(x(3)), i(1)], ""));
    exps.push(String::from("movdqa %xmm3, (%rax)\n"));
    cases.push((X86Op::LoadVConst, vec![d(x(4)), i(0), i(0)], ""));
    exps.push(String::from("pxor %xmm4, %xmm4\n"));
    cases.push((X86Op::LoadVConst, vec![d(x(9)), i(u64::MAX), i(u64::MAX)], ""));
    exps.push(String::from("pcmpeqd %xmm9, %xmm9\n"));
    cases.push((X86Op::LoadVConst, vec![d(x(2)), i(0x1122_3344_5566_7788), i(0x99)], ""));
    exps.push(String::from("movabsq $0x1122334455667788, %r11\nmovq %r11, %xmm2\nmovl $0x99, %r11d\nmovq %r11, %xmm15\npunpcklqdq %xmm15, %xmm2\n"));
    cases.push((X86Op::MovRR, vec![d(x(10)), u(x(3))], ""));
    exps.push(String::from("movaps %xmm3, %xmm10\n"));
    // Two-address expansions: d != a (copy first), and a non-commutative op
    // with d == b (through a scratch).
    cases.push((X86Op::VOp, vec![d(x(0)), u(x(1)), u(x(2)), i(VEnc::op(0x66, 0xFA, false))], ""));
    exps.push(String::from("movaps %xmm1, %xmm0\npsubd %xmm2, %xmm0\n"));
    cases.push((X86Op::VOp, vec![d(x(2)), u(x(1)), u(x(2)), i(VEnc::op(0x66, 0xFA, false))], ""));
    exps.push(String::from("movaps %xmm2, %xmm15\nmovaps %xmm1, %xmm2\npsubd %xmm15, %xmm2\n"));

    assert_eq!(cases.len(), exps.len());
    let n = cases.len();
    for ((op, ops, _), want) in cases.into_iter().zip(&exps) {
        let mine = encode_one(op, ops);
        let theirs = llvm_mc(want).unwrap_or_else(|| panic!("llvm-mc rejects {want}"));
        assert_eq!(mine, theirs, "{op:?} encodes {mine:02x?}, llvm-mc gives {theirs:02x?} for:\n{want}");
    }
    assert!(n > 60, "{n} encodings checked");
}

// ---------------------------------------------------------------------------
// System V ABI: __m128i to and from gcc-compiled C
// ---------------------------------------------------------------------------

#[test]
fn m128i_crosses_the_abi_to_and_from_gcc() {
    let Some(cc) = ["gcc", "cc"].into_iter().find(|c| {
        std::process::Command::new(c)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }) else {
        eprintln!("skipping m128i ABI test: no C compiler");
        return;
    };
    // C side: takes two __m128i and a __m128d, returns a __m128i; and calls
    // back into our `lf_vadd` with vectors.
    let c_src = r#"
#include <emmintrin.h>
#include <stdint.h>
__m128i c_mix(__m128i a, __m128i b, __m128d f) {
    __m128i s = _mm_add_epi32(a, b);
    __m128i t = _mm_castpd_si128(_mm_mul_pd(f, f));
    return _mm_xor_si128(s, t);
}
extern __m128i lf_vadd(__m128i a, __m128i b, long k);
int main(void) {
    __m128i a = _mm_set_epi32(4, 3, 2, 1);
    __m128i b = _mm_set_epi32(40, 30, 20, 10);
    __m128i r = lf_vadd(a, b, 7);
    int32_t out[4];
    _mm_storeu_si128((__m128i *)out, r);
    /* lf_vadd = c_mix(a*k... see the IR): check lane by lane. */
    return (out[0] == 11 * 7 && out[1] == 22 * 7 && out[2] == 33 * 7 && out[3] == 44 * 7) ? 0 : 1;
}
"#;
    let lf_src = r#"
module "abi"
func @c_mix(<4 x i32>, <4 x i32>, <2 x f64>) -> <4 x i32>
func @lf_vadd(<4 x i32>, <4 x i32>, i64) -> <4 x i32> {
entry ^0(%a: <4 x i32>, %b: <4 x i32>, %k: i64):
  %zero = bitcast <2 x i64> (i64 0, i64 0) : <2 x f64>
  %s = call @c_mix(%a, %b, %zero) : <4 x i32>
  %k32 = trunc %k : i32
  %ks = splat %k32 : <4 x i32>
  %r = mul %s, %ks : <4 x i32>
  ret %r
}
"#;
    let dir = std::env::temp_dir().join(format!("lf_m128_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c_path = dir.join("m.c");
    std::fs::write(&c_path, c_src).unwrap();
    for level in LEVELS {
        let (mut m, syms) = parse(lf_src);
        optimize(&mut m, level);
        let obj = super::compile_module(&m, &syms);
        let o_path = dir.join(format!("lf_{level:?}.o"));
        std::fs::write(&o_path, crate::mc::elf::write(&obj)).unwrap();
        let exe = dir.join(format!("m_{level:?}"));
        let st = std::process::Command::new(cc)
            .args(["-O1", "-msse2", "-o"])
            .arg(&exe)
            .arg(&c_path)
            .arg(&o_path)
            .status()
            .expect("run the C compiler");
        assert!(st.success(), "C compile/link failed");
        let st = std::process::Command::new(&exe).status().expect("run the linked program");
        assert_eq!(st.code(), Some(0), "__m128i round trip at {level:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
