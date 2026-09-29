//! SIMD vectors on wasm32 (`docs/ir-design.md` §6e): wasm32 declares no legal
//! vector type, so the generic legalizer scalarizes every vector op before
//! lowering. The shared vector fixtures run under node, at `-O0` and `-O2`,
//! and every result must match the reference executor on the original,
//! unoptimized vector IR.

use super::node::{self, Call};
use super::no_node;
use crate::codegen::CodegenOptions;
use crate::target::vector_fixtures::{
    Case, EDGES_SRC, FLOAT_SRC, INPUTS, INT_OPS, PRESSURE_SRC, Rng, assert_matches, cases, compare_src,
    int_arith_src, lanes_src, masks_src, parse, random_inputs, random_program, reference,
};
use crate::transform::pipeline::{OptLevel, optimize};

/// Compile `src` for wasm32 at `-O0` and `-O2`, run `cases` under node, and
/// compare with the reference. `false` when node is not installed.
fn check(src: &str, cases: &[Case], tag: &str) -> bool {
    if node::node().is_none() {
        return false;
    }
    let want = reference(src, cases);
    for level in [OptLevel::O0, OptLevel::O2] {
        let (mut m, syms) = parse(src);
        m.set_data_layout(crate::target::wasm32::data_layout());
        optimize(&mut m, level);
        crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("{tag}: verify after {level:?}: {e:?}"));
        let c = crate::target::wasm32::compile(&m, &syms, &CodegenOptions::default())
            .unwrap_or_else(|e| panic!("{tag}: {e}"));
        let wasm = c.object.to_linked(&Default::default()).unwrap_or_else(|e| panic!("{tag}: {e}"));
        let calls: Vec<Call> = cases
            .iter()
            .map(|(name, args)| Call {
                func: name.clone(),
                args: args.iter().map(|&a| ("i64", a as u64)).collect(),
                rets: vec!["i64"],
            })
            .collect();
        let out = node::run(tag, &wasm, &calls).expect("node");
        let got: Vec<u64> = out
            .into_iter()
            .zip(cases)
            .map(|(r, (name, args))| {
                let r = r.unwrap_or_else(|t| panic!("{tag} @{name}{args:?} at {level:?} trapped: {t}"));
                r[0]
            })
            .collect();
        assert_matches(&format!("wasm32 {tag} at {level:?}"), cases, &got, &want);
    }
    true
}

/// The names of every function of `src`.
fn names(src: &str) -> Vec<String> {
    let (m, syms) = parse(src);
    m.functions().filter(|f| !f.is_declaration()).map(|f| syms.resolve(f.name).to_owned()).collect()
}

fn check_all(src: &str, inputs: &[[i64; 4]], tag: &str) -> bool {
    let ns = names(src);
    let refs: Vec<&str> = ns.iter().map(String::as_str).collect();
    check(src, &cases(&refs, inputs), tag)
}

#[test]
fn integer_vector_arithmetic() {
    let src = int_arith_src();
    let ns: Vec<String> = ["i8", "i16", "i32", "i64"].iter().flat_map(|t| INT_OPS.map(|o| format!("{o}_{t}"))).collect();
    let refs: Vec<&str> = ns.iter().map(String::as_str).collect();
    if !check(&src, &cases(&refs, &INPUTS), "vint") {
        no_node("integer_vector_arithmetic");
    }
}

#[test]
fn compares_masks_lanes_floats_and_edges() {
    let ok = check_all(&compare_src(), &INPUTS, "vcmp")
        && check_all(&lanes_src(), &INPUTS[1..4], "vlane")
        && check_all(&masks_src(), &INPUTS, "vmask")
        && check(FLOAT_SRC, &cases(&["farith"], &INPUTS), "vflt")
        && check(PRESSURE_SRC, &cases(&["pressure"], &INPUTS), "vpress")
        && check(EDGES_SRC, &cases(&["stack_masks", "store_mask", "fsel", "wrapshift"], &INPUTS), "vedge");
    if !ok {
        no_node("compares_masks_lanes_floats_and_edges");
    }
}

#[test]
fn random_vector_programs() {
    let mut rng = Rng(0x0a5_3200);
    for p in 0..6u64 {
        let (src, ns) = random_program(0x5000 + p, 6, 10, true);
        let mut cs = Vec::new();
        for n in &ns {
            for _ in 0..4 {
                cs.push((n.clone(), random_inputs(&mut rng)));
            }
        }
        if !check(&src, &cs, &format!("vrand{p}")) {
            return no_node("random_vector_programs");
        }
    }
}
