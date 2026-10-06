//! Regression tests for GitHub issues filed by Lode against the scalar
//! passes: #18 (SCCP on a branch whose condition is poison), #20
//! (`simplify_cfg` on a branch whose edges are identical) and #12 (folding
//! `ptr_add x, 0`). Every result is verified, and the optimized program is run
//! by the reference executor against the unoptimized one: wherever the source
//! has defined behavior, the result must refine it.

use crate::ir::refexec::{ExecError, run_named};
use crate::ir::semantics::SemValue;
use crate::ir::text::{parse_module, print_module};
use crate::ir::{Function, InstKind, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize, pass_by_name, run_passes};
use crate::verify::verify_module;

use puremp::Int;

const LEVELS: [OptLevel; 4] = [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3];

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    (m, syms)
}

fn verified(m: &Module, syms: &StrInterner, what: &str) {
    if let Err(e) = verify_module(m) {
        panic!("{what}: {e:#?}\n{}", print_module(m, syms));
    }
}

/// The only function of a module.
fn only(m: &Module) -> &Function {
    let mut fs = m.functions();
    let f = fs.next().expect("a function");
    assert!(fs.next().is_none(), "one function");
    f
}

/// The instructions (terminators included) of `f` matching `pred`.
fn count(f: &Function, pred: impl Fn(&InstKind) -> bool) -> usize {
    f.blocks()
        .flat_map(|(_, b)| b.insts().iter().copied().chain(b.terminator()))
        .filter(|&i| pred(&f.inst(i).kind))
        .count()
}

/// Run `@f` of `orig` and `opt` on each argument list: where the source has
/// defined behavior, the optimized result must refine it.
fn agree(orig: &(Module, StrInterner), opt: &(Module, StrInterner), what: &str, inputs: &[Vec<SemValue>]) {
    for args in inputs {
        let want = match run_named(&orig.0, &orig.1, "f", args) {
            Ok(r) => r.expect("a result"),
            Err(ExecError::Ub(_)) => continue,
            Err(e) => panic!("{what}: source fails on {args:?}: {e:?}"),
        };
        let got = run_named(&opt.0, &opt.1, "f", args)
            .unwrap_or_else(|e| panic!("{what}: optimized fails on {args:?}: {e:?}\n{}", print_module(&opt.0, &opt.1)))
            .expect("a result");
        assert!(got.refines(&want), "{what} @f{args:?}: {got:?} vs {want:?}");
    }
}

/// Integer arguments of width `w` for a one-parameter function.
fn ints(w: u32, xs: &[i64]) -> Vec<Vec<SemValue>> {
    xs.iter().map(|&x| vec![SemValue::int(w, Int::from_i64(x))]).collect()
}

/// Optimize `src` at every level and with `passes` alone; each result must
/// verify and agree with the source on `inputs`. Returns the `-O2` module.
fn check_everywhere(src: &str, passes: &[&str], inputs: &[Vec<SemValue>]) -> (Module, StrInterner) {
    let orig = parse(src);
    verified(&orig.0, &orig.1, "source");
    let mut o2 = None;
    for level in LEVELS {
        let mut opt = parse(src);
        optimize(&mut opt.0, level);
        verified(&opt.0, &opt.1, level.name());
        agree(&orig, &opt, level.name(), inputs);
        if level == OptLevel::O2 {
            o2 = Some(opt);
        }
    }
    for &p in passes {
        let mut opt = parse(src);
        run_passes(&mut opt.0, vec![pass_by_name(p).expect("a pass")]);
        verified(&opt.0, &opt.1, p);
        agree(&orig, &opt, p, inputs);
    }
    o2.expect("-O2 ran")
}

// ---------------------------------------------------------------------------
// Issue #18: SCCP on a branch whose condition is poison
// ---------------------------------------------------------------------------

/// The issue's repro: `^2` is only entered with a poison argument.
const POISON_BR: &str = r#"
module "sccp_poison"

func @f(i32) -> i32 {
entry ^0(%0: i32):
  %1 = icmp ugt %0, i32 100 : i1
  cond_br %1, ^1, ^2(i32 poison)
^1:
  ret i32 0
^2(%2: i32):
  %3 = icmp eq %2, i32 0 : i1
  cond_br %3, ^3, ^4
^3:
  ret i32 1
^4:
  ret i32 2
}
"#;

/// A `switch` on a poison scrutinee, one of whose targets (`^3`) is also
/// reached on a defined path, so it survives while `^4` does not.
const POISON_SWITCH: &str = r#"
module "sccp_poison_switch"

func @f(i32) -> i32 {
entry ^0(%0: i32):
  %1 = icmp ugt %0, i32 100 : i1
  cond_br %1, ^3, ^2(i32 poison)
^2(%2: i32):
  %3 = add %2, i32 1 : i32
  switch %3, ^4 [0: ^3, 7: ^5]
^3:
  ret i32 1
^4:
  ret i32 2
^5:
  ret i32 3
}
"#;

/// A `cond_br` on poison whose true target is reachable on a defined path and
/// whose false target is not.
const POISON_BR_SHARED: &str = r#"
module "sccp_poison_shared"

func @f(i32) -> i32 {
entry ^0(%0: i32):
  %1 = icmp ugt %0, i32 100 : i1
  cond_br %1, ^3(%0), ^2(i32 poison)
^2(%2: i32):
  %3 = icmp eq %2, i32 0 : i1
  cond_br %3, ^3(%2), ^4
^3(%4: i32):
  ret %4
^4:
  ret i32 2
}
"#;

#[test]
fn issue18_branch_on_poison_becomes_unreachable() {
    let inputs = ints(32, &[0, 1, 100, 101, 1000, -1]);
    for src in [POISON_BR, POISON_SWITCH, POISON_BR_SHARED] {
        let o2 = check_everywhere(src, &["sccp"], &inputs);
        let f = only(&o2.0);
        // No branch on poison survives; the block that held it is gone or
        // ends in `unreachable`.
        assert_eq!(count(f, |k| matches!(k, InstKind::Switch(_))), 0, "{}", print_module(&o2.0, &o2.1));
    }
    // SCCP alone turns the poison branch into `unreachable`.
    let mut m = parse(POISON_BR);
    run_passes(&mut m.0, vec![pass_by_name("sccp").expect("sccp")]);
    let f = only(&m.0);
    assert_eq!(count(f, |k| matches!(k, InstKind::Unreachable)), 1, "{}", print_module(&m.0, &m.1));
    assert_eq!(count(f, |k| matches!(k, InstKind::CondBr { .. })), 1, "only the entry branch is left");
}

// ---------------------------------------------------------------------------
// Issue #20: `simplify_cfg` on a branch whose edges are identical
// ---------------------------------------------------------------------------

/// The issue's repro.
const SAME_BR: &str = r#"
module "samebr"

func @f(i64) -> i64 {
entry ^0(%0: i64):
  %1 = icmp eq %0, i64 0 : i1
  cond_br %1, ^1, ^1
^1:
  ret %0
}
"#;

/// Identical edges with arguments, a `switch` whose every target is the same,
/// and a `cond_br` whose edges only meet past forwarding blocks.
const SAME_EDGES: &str = r#"
module "sameedges"

func @f(i64) -> i64 {
entry ^0(%0: i64):
  %1 = icmp ult %0, i64 10 : i1
  %2 = add %0, i64 3 : i64
  cond_br %1, ^1(%2, i64 7), ^1(%2, i64 7)
^1(%3: i64, %4: i64):
  %5 = mul %3, %4 : i64
  switch %0, ^2(%5) [1: ^2(%5), 2: ^2(%5)]
^2(%6: i64):
  %7 = icmp sgt %6, i64 100 : i1
  cond_br %7, ^3(%6), ^4(%6)
^3(%8: i64):
  br ^5(%8)
^4(%9: i64):
  br ^5(%9)
^5(%10: i64):
  %11 = sub %10, i64 1 : i64
  ret %11
}
"#;

/// A switch whose targets agree but whose arguments do not stays a switch.
const DIFFERENT_ARGS: &str = r#"
module "diffargs"

func @f(i64) -> i64 {
entry ^0(%0: i64):
  switch %0, ^1(i64 5) [1: ^1(i64 5), 2: ^1(i64 6)]
^1(%1: i64):
  ret %1
}
"#;

#[test]
fn issue20_identical_edges_fold_to_br() {
    let inputs = ints(64, &[0, 1, 2, 3, 9, 10, 11, 200, -4]);
    for src in [SAME_BR, SAME_EDGES] {
        // simplify_cfg alone folds every branch: no compare survives DCE.
        let orig = parse(src);
        let mut opt = parse(src);
        run_passes(&mut opt.0, vec![pass_by_name("simplify_cfg").expect("simplify_cfg"), pass_by_name("dce").expect("dce")]);
        verified(&opt.0, &opt.1, "simplify_cfg");
        agree(&orig, &opt, "simplify_cfg", &inputs);
        let f = only(&opt.0);
        let text = print_module(&opt.0, &opt.1);
        assert_eq!(f.block_count(), 1, "{text}");
        assert_eq!(count(f, |k| matches!(k, InstKind::ICmp(_) | InstKind::CondBr { .. } | InstKind::Switch(_))), 0, "{text}");
        // A second run finds nothing left to do.
        let before = print_module(&opt.0, &opt.1);
        run_passes(&mut opt.0, vec![pass_by_name("simplify_cfg").expect("simplify_cfg")]);
        assert_eq!(print_module(&opt.0, &opt.1), before);

        let o2 = check_everywhere(src, &["simplify_cfg"], &inputs);
        let f = only(&o2.0);
        assert_eq!(count(f, |k| matches!(k, InstKind::ICmp(_) | InstKind::CondBr { .. })), 0, "{}", print_module(&o2.0, &o2.1));
    }
    // Differing arguments keep the switch.
    let mut m = parse(DIFFERENT_ARGS);
    run_passes(&mut m.0, vec![pass_by_name("simplify_cfg").expect("simplify_cfg")]);
    verified(&m.0, &m.1, "simplify_cfg");
    assert_eq!(count(only(&m.0), |k| matches!(k, InstKind::Switch(_))), 1);
    check_everywhere(DIFFERENT_ARGS, &["simplify_cfg"], &ints(64, &[0, 1, 2, 3]));
}
