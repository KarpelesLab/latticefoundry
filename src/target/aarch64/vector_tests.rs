//! Vectors on AArch64 (`docs/ir-design.md` §6c). Programs are legalized either
//! for NEON ([`NeonLegality`], the compile path) or with everything scalarized
//! ([`ScalarOnly`]), instruction-selected, and run on the MIR interpreter (this
//! host cannot execute A64 code); every result must match the reference
//! executor on the original vector IR, and the whole encoding pipeline must
//! accept the program. The NEON encodings are diffed against `llvm-mc`.

use super::encode::{neon_dup, neon_dup_lane, neon_ins_elem, neon_ins_gpr, neon_shift, neon_umov, neon2, neon3, q_ldst_uimm, simd_mov};
use super::interp;
use super::isel::neon::NeonOp;
use super::isel::{A64Op, AArch64Target, NeonLegality};
use crate::codegen::legalize::{ScalarOnly, VectorLegality, legalized, uses_vectors};
use crate::ir::FuncId;
use crate::target::vector_fixtures::{
    Case, EDGES_SRC, FLOAT_SRC, INPUTS, INT_OPS, PRESSURE_SRC, Rng, assert_matches, cases, compare_src, int_arith_src, lanes_src,
    masks_src, parse, random_inputs, random_program, reference,
};

use puremp::Int;

/// Run `cases` of `src` on the interpreter after legalizing for `legality`.
fn run_cases(src: &str, cases: &[Case], legality: &dyn VectorLegality) -> Vec<u64> {
    let (m, syms) = parse(src);
    let legal = legalized(&m, legality);
    crate::verify::verify_module(&legal).unwrap_or_else(|e| panic!("legalized: {e:?}"));
    let target = AArch64Target::new();
    let funcs: Vec<_> = (0..legal.function_count()).map(|i| target.select(&legal, FuncId::from_index(i))).collect();
    for k in 0..legal.function_count() {
        assert!(!super::compile_function(&m, FuncId::from_index(k), &syms).bytes.is_empty());
    }
    cases
        .iter()
        .map(|(name, args)| {
            let idx = legal.functions().position(|f| syms.resolve(f.name) == name).expect("case function");
            let a: Vec<Int> = args.iter().map(|&x| Int::from_i64(x)).collect();
            let v = interp::run(&target, &funcs, idx, &a)
                .unwrap_or_else(|e| panic!("@{name}{args:?}: {e}"))
                .expect("a result");
            v.to_u64().expect("a 64-bit pattern")
        })
        .collect()
}

fn check(src: &str, cs: &[Case], what: &str) {
    let want = reference(src, cs);
    assert_matches(&format!("{what} (NEON)"), cs, &run_cases(src, cs, &NeonLegality), &want);
}

#[test]
fn scalarized_random_vector_programs_match_the_reference() {
    let mut rng = Rng(0xa64);
    for p in 0..4u64 {
        let (src, names) = random_program(0x2000 + p, 5, 8, true);
        let (m, _) = parse(&src);
        assert!(!uses_vectors(&legalized(&m, &ScalarOnly)), "every vector is scalarized");
        let mut cs = Vec::new();
        for n in &names {
            for _ in 0..3 {
                cs.push((n.clone(), random_inputs(&mut rng)));
            }
        }
        let want = reference(&src, &cs);
        assert_matches(&format!("aarch64 scalar rand{p}"), &cs, &run_cases(&src, &cs, &ScalarOnly), &want);
    }
}

#[test]
fn neon_random_vector_programs_match_the_reference() {
    let mut rng = Rng(0x0e0e);
    for p in 0..8u64 {
        let (src, names) = random_program(0x4000 + p, 6, 10, true);
        let mut cs = Vec::new();
        for n in &names {
            for _ in 0..3 {
                cs.push((n.clone(), random_inputs(&mut rng)));
            }
        }
        check(&src, &cs, &format!("rand{p}"));
    }
}

#[test]
fn neon_integer_arithmetic_matches_the_reference() {
    let names: Vec<String> =
        ["i8", "i16", "i32", "i64"].iter().flat_map(|t| INT_OPS.map(|o| format!("{o}_{t}"))).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    check(&int_arith_src(), &cases(&refs, &INPUTS), "int");
}

#[test]
fn neon_compares_floats_and_lane_moves_match_the_reference() {
    for (src, tag) in [(compare_src(), "cmp"), (FLOAT_SRC.to_string(), "flt"), (lanes_src(), "lane"), (masks_src(), "mask")] {
        let (m, syms) = parse(&src);
        let names: Vec<String> = m.functions().map(|f| syms.resolve(f.name).to_owned()).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        check(&src, &cases(&refs, &INPUTS), tag);
    }
}

#[test]
fn neon_selects_vector_ops_and_scalarizes_the_rest() {
    let src = int_arith_src();
    let (m, syms) = parse(&src);
    let legal = legalized(&m, &NeonLegality);
    let ops_of = |name: &str| -> Vec<A64Op> {
        let idx = legal.functions().position(|f| syms.resolve(f.name) == name).expect("function");
        let mf = AArch64Target::new().select(&legal, FuncId::from_index(idx));
        mf.block_ids().flat_map(|b| mf.block(b).insts.iter().map(|i| A64Op::decode(i.opcode)).collect::<Vec<_>>()).collect()
    };
    let count = |ops: &[A64Op], op: A64Op| ops.iter().filter(|&&o| o == op).count();
    // Byte multiply, variable-free shifts, min/max and saturation are single
    // NEON ops; the fold at the end is the only scalar multiply.
    for f in ["mul_i8", "shl_i8", "ashr_i64", "umin_i16", "sadd_sat_i64", "uadd_sat_i8"] {
        let ops = ops_of(f);
        assert!(count(&ops, A64Op::NeonOp3) + count(&ops, A64Op::NeonShift) >= 1, "{f}: {ops:?}");
        assert_eq!(count(&ops, A64Op::Mul), 1, "{f}: {ops:?}");
    }
    // No NEON divide: scalarized. No 64-bit `smin` either, but its expansion
    // (a compare and a blend) stays vector code.
    assert_eq!(count(&ops_of("udiv_i32"), A64Op::Udiv), 4);
    let ops = ops_of("smin_i64");
    assert!(count(&ops, A64Op::CmpCset) == 0 && count(&ops, A64Op::NeonOp3) >= 4, "{ops:?}");
}

// ---------------------------------------------------------------------------
// Encoding differential vs llvm-mc
// ---------------------------------------------------------------------------

/// Assemble `asm` (one instruction) with `llvm-mc`, or `None` without it.
fn llvm_mc(asm: &str) -> Option<u32> {
    use std::io::Write;
    let mut child = std::process::Command::new("llvm-mc")
        .args(["--triple=aarch64", "--show-encoding"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(format!("{asm}\n").as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let start = text.find("encoding: [")? + "encoding: [".len();
    let end = text[start..].find(']')? + start;
    let bytes: Vec<u8> = text[start..end]
        .split(',')
        .filter_map(|t| u8::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
        .collect();
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

#[test]
fn neon_encodings_match_llvm_mc() {
    if llvm_mc("add v0.16b, v1.16b, v2.16b").is_none() {
        eprintln!("skipping neon_encodings_match_llvm_mc: no llvm-mc");
        return;
    }
    let arr = |es: u32| match es {
        8 => "16b",
        16 => "8h",
        32 => "4s",
        _ => "2d",
    };
    let mut cases: Vec<(u32, String)> = Vec::new();
    // Three-register ops at every lane size they exist for.
    let int3: [(NeonOp, &str, &[u32]); 21] = [
        (NeonOp::Add, "add", &[8, 16, 32, 64]),
        (NeonOp::Sub, "sub", &[8, 16, 32, 64]),
        (NeonOp::Mul, "mul", &[8, 16, 32]),
        (NeonOp::Cmeq, "cmeq", &[8, 16, 32, 64]),
        (NeonOp::Cmgt, "cmgt", &[8, 16, 32, 64]),
        (NeonOp::Cmge, "cmge", &[8, 16, 32, 64]),
        (NeonOp::Cmhi, "cmhi", &[8, 16, 32, 64]),
        (NeonOp::Cmhs, "cmhs", &[8, 16, 32, 64]),
        (NeonOp::Sshl, "sshl", &[8, 16, 32, 64]),
        (NeonOp::Ushl, "ushl", &[8, 16, 32, 64]),
        (NeonOp::Smax, "smax", &[8, 16, 32]),
        (NeonOp::Smin, "smin", &[8, 16, 32]),
        (NeonOp::Umax, "umax", &[8, 16, 32]),
        (NeonOp::Umin, "umin", &[8, 16, 32]),
        (NeonOp::Sqadd, "sqadd", &[8, 16, 32, 64]),
        (NeonOp::Uqadd, "uqadd", &[8, 16, 32, 64]),
        (NeonOp::Sqsub, "sqsub", &[8, 16, 32, 64]),
        (NeonOp::Uqsub, "uqsub", &[8, 16, 32, 64]),
        (NeonOp::Fadd, "fadd", &[32, 64]),
        (NeonOp::Fsub, "fsub", &[32, 64]),
        (NeonOp::Fmul, "fmul", &[32, 64]),
    ];
    for (k, (op, mn, sizes)) in int3.iter().enumerate() {
        for &es in *sizes {
            let (d, n, m) = (k as u32 % 32, (k as u32 * 7 + 3) % 32, (es + k as u32) % 32);
            let a = arr(es);
            cases.push((neon3(*op, es, d, n, m), format!("{mn} v{d}.{a}, v{n}.{a}, v{m}.{a}")));
        }
    }
    for (op, mn) in [(NeonOp::Fdiv, "fdiv"), (NeonOp::Fcmeq, "fcmeq"), (NeonOp::Fcmge, "fcmge"), (NeonOp::Fcmgt, "fcmgt")] {
        for es in [32, 64] {
            let a = arr(es);
            cases.push((neon3(op, es, 30, 5, 17), format!("{mn} v30.{a}, v5.{a}, v17.{a}")));
        }
    }
    for (op, mn) in [(NeonOp::And, "and"), (NeonOp::Bic, "bic"), (NeonOp::Orr, "orr"), (NeonOp::Eor, "eor")] {
        cases.push((neon3(op, 8, 1, 22, 9), format!("{mn} v1.16b, v22.16b, v9.16b")));
    }
    cases.push((neon3(NeonOp::Tbl, 8, 4, 12, 28), "tbl v4.16b, {v12.16b}, v28.16b".into()));
    // Two-register and across-lane ops.
    for es in [8, 16, 32, 64] {
        let a = arr(es);
        cases.push((neon2(NeonOp::Neg, es, 3, 19), format!("neg v3.{a}, v19.{a}")));
    }
    cases.push((neon2(NeonOp::Not, 8, 7, 8), "mvn v7.16b, v8.16b".into()));
    for es in [32, 64] {
        let a = arr(es);
        for (op, mn) in [(NeonOp::Fneg, "fneg"), (NeonOp::Scvtf, "scvtf"), (NeonOp::Ucvtf, "ucvtf"), (NeonOp::Fcvtzs, "fcvtzs"), (NeonOp::Fcvtzu, "fcvtzu")] {
            cases.push((neon2(op, es, 11, 25), format!("{mn} v11.{a}, v25.{a}")));
        }
    }
    for (es, reg) in [(8, 'b'), (16, 'h'), (32, 's')] {
        let a = arr(es);
        for (op, mn) in [(NeonOp::Addv, "addv"), (NeonOp::Smaxv, "smaxv"), (NeonOp::Sminv, "sminv"), (NeonOp::Umaxv, "umaxv"), (NeonOp::Uminv, "uminv")] {
            cases.push((neon2(op, es, 2, 14), format!("{mn} {reg}2, v14.{a}")));
        }
    }
    cases.push((neon2(NeonOp::Addp, 64, 6, 21), "addp d6, v21.2d".into()));
    // Shifts by immediate, at the extremes of their ranges.
    for es in [8u32, 16, 32, 64] {
        let a = arr(es);
        for amt in [0, 1, es - 1] {
            cases.push((neon_shift(NeonOp::Shl, es, 9, 10, amt), format!("shl v9.{a}, v10.{a}, #{amt}")));
        }
        for amt in [1, es / 2, es] {
            cases.push((neon_shift(NeonOp::Ushr, es, 9, 10, amt), format!("ushr v9.{a}, v10.{a}, #{amt}")));
            cases.push((neon_shift(NeonOp::Sshr, es, 27, 0, amt), format!("sshr v27.{a}, v0.{a}, #{amt}")));
        }
    }
    // Lane moves.
    for (es, lanes) in [(8u32, 16u32), (16, 8), (32, 4), (64, 2)] {
        let a = arr(es);
        let (sfx, r) = match es {
            8 => ("b", "w"),
            16 => ("h", "w"),
            32 => ("s", "w"),
            _ => ("d", "x"),
        };
        cases.push((neon_dup(es, 5, 20), format!("dup v5.{a}, {r}20")));
        for lane in [0, lanes - 1] {
            cases.push((neon_dup_lane(es, lane, 13, 31), format!("dup v13.{a}, v31.{sfx}[{lane}]")));
            cases.push((neon_umov(es, lane, 7, 18), format!("umov {r}7, v18.{sfx}[{lane}]")));
            cases.push((neon_ins_gpr(es, lane, 2, 16), format!("mov v2.{sfx}[{lane}], {r}16")));
            cases.push((neon_ins_elem(es, lane, 29, 8), format!("mov v29.{sfx}[{lane}], v8.{sfx}[0]")));
        }
    }
    cases.push((simd_mov(3, 30), "mov v3.16b, v30.16b".into()));
    cases.push((q_ldst_uimm(true, 12, 31, 0), "ldr q12, [sp]".into()));
    cases.push((q_ldst_uimm(false, 12, 4, 3), "str q12, [x4, #48]".into()));
    cases.push((q_ldst_uimm(true, 0, 16, 4095), "ldr q0, [x16, #65520]".into()));

    let n = cases.len();
    for (word, asm) in cases {
        let theirs = llvm_mc(&asm).unwrap_or_else(|| panic!("llvm-mc rejects `{asm}`"));
        assert_eq!(word, theirs, "`{asm}`: ours {word:#010x}, llvm-mc {theirs:#010x}");
    }
    assert!(n > 150, "{n} NEON encodings checked");
}


#[test]
fn selects_under_register_pressure_allocate_and_run() {
    let cs: Vec<Case> = [[1i64, 2, 1, 0], [-7, 1 << 40, 0, 1], [0, 0, 3, 3]]
        .iter()
        .map(|a| ("pressure".to_string(), a.to_vec()))
        .collect();
    let want = reference(PRESSURE_SRC, &cs);
    for legality in [&NeonLegality as &dyn VectorLegality, &ScalarOnly] {
        assert_matches("pressure", &cs, &run_cases(PRESSURE_SRC, &cs, legality), &want);
    }
}

#[test]
fn mask_stores_float_selects_and_wrapped_shifts() {
    // (Stack-passed arguments need a modeled stack pointer, which the
    // pre-allocation interpreter lacks; x86-64 runs that case natively.)
    let cs = cases(&["store_mask", "fsel", "wrapshift"], &INPUTS);
    let want = reference(EDGES_SRC, &cs);
    for legality in [&NeonLegality as &dyn VectorLegality, &ScalarOnly] {
        assert_matches("edges", &cs, &run_cases(EDGES_SRC, &cs, legality), &want);
    }
}
