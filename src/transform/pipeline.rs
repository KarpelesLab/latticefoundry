//! The `-O` optimization pipeline (ROADMAP Phase 4).
//!
//! This module composes the individual transforms of [`crate::transform`] into
//! ordered, iterated pipelines selected by an [`OptLevel`] (`-O0`..`-O3`), the
//! way `lf build -O2` and `lf-opt -O2` drive them. It is deliberately a *real*
//! pipeline — a fixed, inspectable sequence with the higher levels re-running the
//! core clean-up group to a bounded fixpoint — not a single mega-pass.
//!
//! Every pass in the pipeline is individually a **verified refinement** (tenet
//! T3 / bet B2): each transform rebuilds a function into a semantically-refining
//! one and keeps the structural verifier happy. Composition preserves both
//! properties, so `optimize(m, level)` maps a verifying module to a verifying
//! module that refines it. The pipeline is **deterministic** (tenet T5): the pass
//! list is fixed per level and each pass is a deterministic functional rebuild,
//! so the same input always yields the same output.
//!
//! ## The pipelines
//!
//! - **O0** — nothing (the caller still verifies; this is the identity).
//! - **O1** — `mem2reg → sccp → simplify_cfg → dce`: promote memory to SSA, fold
//!   constants and prune dead edges, tidy the CFG, and drop the dead code that
//!   exposes.
//! - **O2** — `mem2reg`, then the clean-up group
//!   `sccp → egraph → simplify_cfg → dce → licm` iterated twice, then one round of
//!   `inline`, then `sccp → egraph → dce` to clean up after inlining (so
//!   cross-call constants fold).
//! - **O3** — O2 with a deeper fixpoint (three clean-up rounds), a second
//!   inlining round, and more post-inline clean-up — the level where interprocedural
//!   work (including cross-module inlining after LTO) pays off most.

use crate::ir::Module;
use crate::pass::{ModulePass, PassManager};
use crate::transform::{
    Dce, FunctionTransformPass, Inline, Licm, Mem2Reg, SimplifyCfg,
    egraph::EqSatPass, sccp::SccpPass,
};

/// The optimization level, selected on the driver command line by `-O0`..`-O3`.
///
/// Higher levels run strictly more work: the pass list of a lower level is a
/// prefix-in-spirit of the higher one, and every level preserves program
/// behavior (it is a checked refinement).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub enum OptLevel {
    /// No optimization — the identity pipeline. The default for `lf build`.
    #[default]
    O0,
    /// Light clean-up: mem2reg, SCCP, CFG simplification, DCE.
    O1,
    /// The workhorse level: the clean-up group iterated to a bounded fixpoint,
    /// plus inlining and post-inline clean-up.
    O2,
    /// O2 with a deeper fixpoint and a second inlining round.
    O3,
}

impl OptLevel {
    /// Parse a level from a driver token: `-O0`/`-O1`/`-O2`/`-O3`, or a bare
    /// `-O` (treated as `-O1`). Returns `None` if `tok` is not an `-O` flag.
    pub fn parse_flag(tok: &str) -> Option<OptLevel> {
        match tok {
            "-O" | "-O1" => Some(OptLevel::O1),
            "-O0" => Some(OptLevel::O0),
            "-O2" => Some(OptLevel::O2),
            "-O3" => Some(OptLevel::O3),
            _ => None,
        }
    }

    /// The short name of this level (`O0`..`O3`).
    pub fn name(self) -> &'static str {
        match self {
            OptLevel::O0 => "O0",
            OptLevel::O1 => "O1",
            OptLevel::O2 => "O2",
            OptLevel::O3 => "O3",
        }
    }
}

/// Build a fresh boxed instance of the pass named `name`, or `None` if the name
/// is unknown. Drives `lf-opt -p pass,pass,...` and the pipeline builders below.
///
/// Recognized names: `mem2reg`, `sccp`, `simplify_cfg` (aka `simplifycfg`,
/// `scfg`), `dce`, `egraph` (aka `eqsat`), `licm`, `inline`.
pub fn pass_by_name(name: &str) -> Option<Box<dyn ModulePass>> {
    let pass: Box<dyn ModulePass> = match name {
        "mem2reg" => Box::new(FunctionTransformPass::new(Mem2Reg)),
        "sccp" => Box::new(SccpPass),
        "simplify_cfg" | "simplifycfg" | "scfg" => {
            Box::new(FunctionTransformPass::new(SimplifyCfg))
        }
        "dce" => Box::new(FunctionTransformPass::new(Dce)),
        "egraph" | "eqsat" => Box::new(EqSatPass),
        "licm" => Box::new(FunctionTransformPass::new(Licm)),
        "inline" => Box::new(Inline::new()),
        _ => return None,
    };
    Some(pass)
}

/// The exact, ordered pass sequence for `level`, with the higher levels' fixpoint
/// iterations unrolled into the list. Running this list in order *is*
/// [`optimize`]; it is exposed so drivers and tests can inspect and print the
/// pipeline.
pub fn pipeline_for(level: OptLevel) -> Vec<Box<dyn ModulePass>> {
    // The core clean-up group, re-run to a (bounded, unrolled) fixpoint.
    fn cleanup(out: &mut Vec<Box<dyn ModulePass>>) {
        for n in ["sccp", "egraph", "simplify_cfg", "dce", "licm"] {
            out.push(pass_by_name(n).expect("known pass"));
        }
    }
    let mut out: Vec<Box<dyn ModulePass>> = Vec::new();
    match level {
        OptLevel::O0 => {}
        OptLevel::O1 => {
            for n in ["mem2reg", "sccp", "simplify_cfg", "dce"] {
                out.push(pass_by_name(n).expect("known pass"));
            }
        }
        OptLevel::O2 => {
            out.push(pass_by_name("mem2reg").expect("known pass"));
            for _ in 0..2 {
                cleanup(&mut out);
            }
            out.push(pass_by_name("inline").expect("known pass"));
            for n in ["sccp", "egraph", "dce"] {
                out.push(pass_by_name(n).expect("known pass"));
            }
        }
        OptLevel::O3 => {
            out.push(pass_by_name("mem2reg").expect("known pass"));
            for _ in 0..3 {
                cleanup(&mut out);
            }
            out.push(pass_by_name("inline").expect("known pass"));
            for _ in 0..2 {
                cleanup(&mut out);
            }
            out.push(pass_by_name("inline").expect("known pass"));
            for n in ["sccp", "egraph", "simplify_cfg", "dce"] {
                out.push(pass_by_name(n).expect("known pass"));
            }
        }
    }
    out
}

/// A human-readable, arrow-separated description of the pipeline for `level`
/// (the pass names in run order). `O0` is reported as `"(none)"`.
pub fn pipeline_description(level: OptLevel) -> String {
    let passes = pipeline_for(level);
    if passes.is_empty() {
        return "(none)".to_owned();
    }
    passes.iter().map(|p| p.name().to_owned()).collect::<Vec<_>>().join(" → ")
}

/// Run the `-O`-level pipeline over `module` in place.
///
/// The module is expected to already verify (`Structural` tier); after
/// optimization it still verifies and refines the input. `O0` is the identity.
pub fn optimize(module: &mut Module, level: OptLevel) {
    run_passes(module, pipeline_for(level));
}

/// Run an explicit, already-constructed pass list over `module` in order, sharing
/// one [`PassManager`] (so analysis invalidation is handled between passes).
pub fn run_passes(module: &mut Module, passes: Vec<Box<dyn ModulePass>>) {
    if passes.is_empty() {
        return;
    }
    let mut pm = PassManager::new();
    for p in passes {
        pm.add(p);
    }
    pm.run(module);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::inst::{Flags, InstKind};
    use crate::ir::text;
    use crate::ir::{FuncId, Function, Module};
    use crate::support::StrInterner;
    use crate::verify::{RefinementResult, check_refinement, verify_module};

    /// `f(x:i32) = (2+3)*x + (x - x)` — a pure, single-block function whose
    /// optimum is `5*x`. Both const folding and identity simplification apply, so
    /// the optimizer must shrink it.
    fn build_foldable() -> (Module, StrInterner, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("fold");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let x = b.param(entry, 0);
            let two = b.const_i64(i32t, 2);
            let three = b.const_i64(i32t, 3);
            let a = b.add(two, three, Flags::NONE);
            let prod = b.mul(a, x, Flags::NONE);
            let zero = b.sub(x, x, Flags::NONE);
            let r = b.add(prod, zero, Flags::NONE);
            b.ret(Some(r));
        }
        (m, syms, f)
    }

    /// Count the non-terminator instructions of a function.
    fn body_insts(f: &Function) -> usize {
        f.blocks().map(|(_, b)| b.insts().len()).sum()
    }

    #[test]
    fn o2_verifies_and_shrinks() {
        let (mut m, _syms, f) = build_foldable();
        let before = body_insts(m.function(f));
        optimize(&mut m, OptLevel::O2);
        assert!(verify_module(&m).is_ok(), "optimized module must verify");
        let after = body_insts(m.function(f));
        assert!(after < before, "O2 must shrink {before} -> got {after}");
    }

    #[test]
    fn o1_verifies() {
        let (mut m, _syms, f) = build_foldable();
        optimize(&mut m, OptLevel::O1);
        assert!(verify_module(&m).is_ok());
        let _ = f;
    }

    #[test]
    fn optimized_function_refines_original() {
        // Spot-check on the pure-integer subset the refinement checker supports:
        // the e-graph optimizer's rebuild must be a proven refinement of the input.
        // `map_function` builds the optimized body against the *same* interning
        // tables as the original, so both are checkable together.
        let (mut m, _syms, f) = build_foldable();
        let mut t = crate::transform::EqSat::new();
        t.analyze(m.function(f), m.types(), m.consts());
        let (fresh, changed) = m.map_function(f, |old, b| {
            use crate::transform::FunctionTransform;
            t.run(old, b)
        });
        assert_eq!(changed, crate::pass::Changed::Yes, "the e-graph pass should fire");
        match check_refinement(m.types(), m.consts(), m.function(f), &fresh) {
            RefinementResult::Refines => {}
            other => panic!("optimized function must refine original, got {other:?}"),
        }
    }

    #[test]
    fn each_level_is_deterministic() {
        for level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
            let (mut a, sa, _) = build_foldable();
            let (mut b, sb, _) = build_foldable();
            optimize(&mut a, level);
            optimize(&mut b, level);
            assert_eq!(
                text::print_module(&a, &sa),
                text::print_module(&b, &sb),
                "{} must be deterministic",
                level.name()
            );
        }
    }

    #[test]
    fn pipeline_description_lists_passes() {
        assert_eq!(pipeline_description(OptLevel::O0), "(none)");
        assert!(pipeline_description(OptLevel::O1).contains("mem2reg"));
        assert!(pipeline_description(OptLevel::O2).contains("inline"));
    }

    #[test]
    fn o2_folds_to_a_single_multiply() {
        // The fully-optimized body should contain exactly one arithmetic op (the
        // `5*x`) — no residual add/sub — proving const-fold + identity elimination.
        let (mut m, _syms, f) = build_foldable();
        optimize(&mut m, OptLevel::O2);
        let func = m.function(f);
        let muls = func
            .blocks()
            .flat_map(|(_, b)| b.insts().iter())
            .filter(|&&i| matches!(func.inst(i).kind, InstKind::Bin(_)))
            .count();
        assert_eq!(muls, 1, "expected a single residual multiply");
    }

    // --- End-to-end native execution: O0 vs O2 preserve behavior --------------
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    mod native {
        use super::*;
        use crate::ir::inst::IntPred;
        use crate::link::{ImageOptions, link_executable, write_executable};

        /// Compile `module` to a native ELF, run it, and return its exit code.
        fn build_and_run(module: &Module, syms: &StrInterner, tag: &str) -> i32 {
            let obj = crate::target::x86_64::compile_module(module, syms);
            let image = link_executable(vec![obj], &ImageOptions::default()).expect("link");
            use std::sync::atomic::{AtomicU64, Ordering};
            static SEQ: AtomicU64 = AtomicU64::new(0);
            let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("lf_pipe_{tag}_{}_{uniq}", std::process::id()));
            let path_str = path.to_str().unwrap().to_owned();
            write_executable(&path_str, &image).expect("write exe");
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
            status.code().expect("exited via code")
        }

        /// `main() -> i64`: a dead branch, a constant-folded sum, and a helper call
        /// `dbl(20)` — so O2 has const-fold, dead-code, and inlining to do. Returns
        /// `21 + dbl(20) = 21 + 40 = 61 ... wait computed below`.
        fn opt_program() -> (Module, StrInterner) {
            let mut syms = StrInterner::new();
            let mut m = Module::new("optprog");
            let i64t = m.types_mut().int(64);
            let sig = m.types_mut().func(vec![i64t], i64t, false);
            // dbl(y) = y + y
            let dbl = m.declare_function(syms.intern("dbl"), sig);
            {
                let mut b = m.build(dbl);
                let entry = b.create_entry_block();
                let y = b.param(entry, 0);
                let s = b.add(y, y, Flags::NONE);
                b.ret(Some(s));
            }
            // main() = { if (1<0) unreachable-ish else t = 20+1; d = dbl(20); ret t + d + (d-d) }
            let msig = m.types_mut().func(vec![], i64t, false);
            let main = m.declare_function(syms.intern("main"), msig);
            {
                let mut b = m.build(main);
                b.create_entry_block();
                let then_b = b.create_block(&[]);
                let cont = b.create_block(&[i64t]);
                let zero = b.const_i64(i64t, 0);
                let one = b.const_i64(i64t, 1);
                // dead branch: 1 < 0 is always false
                let cond = b.icmp(IntPred::Slt, one, zero);
                let twenty = b.const_i64(i64t, 20);
                let dead = b.const_i64(i64t, 999);
                b.cond_br(cond, then_b, &[], cont, &[twenty]);
                b.switch_to(then_b);
                b.br(cont, &[dead]);
                b.switch_to(cont);
                let base = b.param(cont, 0); // 20
                let t = b.add(base, one, Flags::NONE); // 21
                let dref = b.func_ref(dbl);
                let d = b.call(dref, &[twenty], i64t).unwrap(); // dbl(20) = 40
                let sum = b.add(t, d, Flags::NONE); // 61
                let zero2 = b.sub(d, d, Flags::NONE); // 0
                let r = b.add(sum, zero2, Flags::NONE); // 61
                b.ret(Some(r));
            }
            (m, syms)
        }

        #[test]
        fn o0_and_o2_agree_on_exit_code() {
            let (mut m0, s0) = opt_program();
            optimize(&mut m0, OptLevel::O0);
            let e0 = build_and_run(&m0, &s0, "o0");

            let (mut m2, s2) = opt_program();
            optimize(&mut m2, OptLevel::O2);
            assert!(verify_module(&m2).is_ok());
            let e2 = build_and_run(&m2, &s2, "o2");

            assert_eq!(e0, 61, "O0 program result");
            assert_eq!(e2, 61, "O2 must preserve behavior");
        }
    }
}
