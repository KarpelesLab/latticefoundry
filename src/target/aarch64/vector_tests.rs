//! Vectors on AArch64 (`docs/ir-design.md` §6c): no NEON lowering yet, so the
//! generic legalizer scalarizes every vector op. Random vector programs
//! (integer and float) are legalized, instruction-selected, and run on the
//! MIR interpreter, and each result must match the reference executor on the
//! original vector IR; the whole encoding pipeline must also accept them.

use super::interp;
use super::isel::AArch64Target;
use crate::codegen::legalize::{ScalarOnly, legalized, uses_vectors};
use crate::ir::FuncId;
use crate::target::vector_fixtures::{Case, Rng, assert_matches, parse, random_inputs, random_program, reference};

use puremp::Int;

/// Run `cases` of `src` on the interpreter after scalarization.
fn run_cases(src: &str, cases: &[Case]) -> Vec<u64> {
    let (m, syms) = parse(src);
    let legal = legalized(&m, &ScalarOnly);
    assert!(!uses_vectors(&legal), "every vector is scalarized for AArch64");
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
            let v = interp::run(&target, &funcs, idx, &a).expect("interpretation succeeds").expect("a result");
            v.to_u64().expect("a 64-bit pattern")
        })
        .collect()
}

#[test]
fn scalarized_random_vector_programs_match_the_reference() {
    let mut rng = Rng(0xa64);
    for p in 0..4u64 {
        let (src, names) = random_program(0x2000 + p, 5, 8, true);
        let mut cs = Vec::new();
        for n in &names {
            for _ in 0..3 {
                cs.push((n.clone(), random_inputs(&mut rng)));
            }
        }
        let want = reference(&src, &cs);
        let got = run_cases(&src, &cs);
        assert_matches(&format!("aarch64 rand{p}"), &cs, &got, &want);
    }
}
