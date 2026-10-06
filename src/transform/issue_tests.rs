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

// ---------------------------------------------------------------------------
// Issue #12: folding `ptr_add x, 0` and chains of constant offsets
// ---------------------------------------------------------------------------

/// The issue's repro.
const PTR_ADD_ZERO: &str = r#"
module "ptradd0"

func @f(ptr) -> i8 {
entry ^0(%0: ptr):
  %1 = ptr_add inbounds %0, i64 0 : ptr
  %2 = ptr_add inbounds %1, i64 0 : ptr
  %3 = load %2 align 1 : i8
  ret %3
}
"#;

/// A buffer on the stack walked by chains of constant offsets: zero ones (one
/// proven zero by SCCP rather than written so), chains that sum, chains that
/// cancel out, mixed `inbounds`, an `i32` offset under an `i64` one (which
/// must not combine) and an `i8` chain whose sum would overflow its type.
const PTR_ADD_CHAINS: &str = r#"
module "ptrchains"

func @f(i64) -> i64 {
entry ^0(%x: i64):
  %buf = alloca [512 x i8] : ptr
  %z = sub i64 7, i64 7 : i64
  %p0 = ptr_add %buf, %z : ptr
  %p1 = ptr_add inbounds %p0, i64 8 : ptr
  %p2 = ptr_add inbounds %p1, i64 16 : ptr
  store %x, %p2 align 1 : i64
  %q1 = ptr_add inbounds %buf, i64 24 : ptr
  %q2 = ptr_add %q1, i64 -24 : ptr
  %q3 = ptr_add inbounds %q2, i64 24 : ptr
  %a = load %q3 align 1 : i64
  %r1 = ptr_add inbounds %buf, i32 100 : ptr
  %r2 = ptr_add inbounds %r1, i64 4 : ptr
  store %a, %r2 align 1 : i64
  %s1 = ptr_add inbounds %buf, i8 100 : ptr
  %s2 = ptr_add inbounds %s1, i8 100 : ptr
  %s3 = ptr_add inbounds %s2, i8 0 : ptr
  store %x, %s3 align 1 : i64
  %t = ptr_add inbounds %buf, i64 104 : ptr
  %b = load %t align 1 : i64
  %u = ptr_add inbounds %buf, i64 200 : ptr
  %c = load %u align 1 : i64
  %s = add %b, %c : i64
  %y = add %a, %s : i64
  ret %y
}
"#;

/// The `ptr_add` instructions of `f`, as `(inbounds, offset operand)`.
fn ptr_adds(f: &Function) -> Vec<(bool, crate::ir::ValueId)> {
    f.blocks()
        .flat_map(|(_, b)| b.insts().iter())
        .filter_map(|&i| match f.inst(i).kind {
            InstKind::PtrAdd { inbounds } => Some((inbounds, f.inst(i).operands()[1])),
            _ => None,
        })
        .collect()
}

#[test]
fn issue12_ptr_add_zero_folds() {
    let mut m = parse(PTR_ADD_ZERO);
    run_passes(&mut m.0, vec![pass_by_name("sccp").expect("sccp")]);
    verified(&m.0, &m.1, "sccp");
    let f = only(&m.0);
    assert!(ptr_adds(f).is_empty(), "{}", print_module(&m.0, &m.1));
    // The load reads straight from the parameter.
    let load = f.blocks().flat_map(|(_, b)| b.insts().iter()).find(|&&i| matches!(f.inst(i).kind, InstKind::Load { .. }));
    let param = f.block(f.entry().expect("entry")).params()[0];
    assert_eq!(f.inst(*load.expect("a load")).operands()[0], param);
    for level in [OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let mut m = parse(PTR_ADD_ZERO);
        optimize(&mut m.0, level);
        verified(&m.0, &m.1, level.name());
        assert!(ptr_adds(only(&m.0)).is_empty(), "{level:?}\n{}", print_module(&m.0, &m.1));
    }
}

#[test]
fn issue12_constant_offset_chains_combine() {
    let inputs = ints(64, &[0, 1, -1, 0x0102_0304_0506_0708, i64::MIN]);
    check_everywhere(PTR_ADD_CHAINS, &["sccp"], &inputs);
    // After SCCP and DCE, one `ptr_add` is left per address that combines.
    let mut m = parse(PTR_ADD_CHAINS);
    run_passes(&mut m.0, vec![pass_by_name("sccp").expect("sccp"), pass_by_name("dce").expect("dce")]);
    verified(&m.0, &m.1, "sccp");
    let text = print_module(&m.0, &m.1);
    let f = only(&m.0);
    let adds = ptr_adds(f);
    let consts = m.0.consts();
    let offsets: Vec<(bool, i64)> = adds
        .iter()
        .map(|&(ib, o)| match &f.value(o).def {
            crate::ir::ValueDef::Const(c) => match consts.get(*c) {
                crate::ir::value::Const::Int { value, .. } => (ib, value.to_i64().expect("small")),
                other => panic!("{other:?}"),
            },
            other => panic!("non-constant offset {other:?}\n{text}"),
        })
        .collect();
    // %p2 → buf+24 (not inbounds: %p0 was not); %q3 → buf+24 (not inbounds);
    // %r1, %r2 stay (i32 then i64); %s1, %s2 stay (100 + 100 overflows i8),
    // and %s3 is %s2; %t and %u are untouched.
    assert_eq!(
        offsets,
        [(false, 24), (false, 24), (true, 100), (true, 4), (true, 100), (true, 100), (true, 104), (true, 200)],
        "{text}"
    );
    // A second SCCP run finds nothing more to combine.
    run_passes(&mut m.0, vec![pass_by_name("sccp").expect("sccp")]);
    assert_eq!(print_module(&m.0, &m.1), text);
}
