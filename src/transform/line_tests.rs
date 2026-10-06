//! Regression tests for GitHub issue #19: the optimizer and the codegen
//! legalizers keep source-line provenance. Every pass, run alone and inside the
//! `-O1`..`-O3` pipelines, must keep each function's `decl_line` and give every
//! instruction it leaves behind a line drawn from the input's lines (a
//! replacement inherits the line of what it replaces, an inlined instruction
//! keeps the callee's, a hoisted one its own).

use std::collections::BTreeSet;

use crate::codegen::legalize::{ScalarOnly, legalize_vectors};
use crate::codegen::legalize_int::{LegalizeOptions, legalize_ints};
use crate::codegen::softfloat::{SoftFloatAbi, lower_soft_float};
use crate::ir::inst::Flags;
use crate::ir::text::{parse_module, print_module};
use crate::ir::{FuncId, Function, Module};
use crate::pass::{Changed, ModulePass};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize, pass_by_name};
use crate::transform::sanitize::{SanitizeKinds, SanitizeOptions, SanitizeRuntime, sanitize_module};
use crate::transform::superopt::superoptimize;
use crate::transform::yield_points::{YieldConfig, YieldPoints};
use crate::verify::verify_module;

/// One instruction per source line, exercising every optimization pass:
/// a promotable slot (mem2reg), constant folding and a constant branch
/// (sccp, simplify_cfg), a dead op (dce), algebra (egraph), a loop-invariant
/// multiply (licm), an inlinable callee (inline), an unreferenced internal
/// function (dfe), a single-predecessor merge (simplify_cfg) and a division
/// (sanitize).
const FIXTURE: &str = r#"module "lines"
func internal @helper(i64) -> i64 {
entry ^0(%a: i64):
  %h1 = mul %a, i64 2 : i64
  %h2 = add %h1, i64 0 : i64
  ret %h2
}
func internal @unused() -> i64 {
entry ^0:
  ret i64 7
}
func @main(i64, i64) -> i64 {
entry ^0(%x: i64, %n: i64):
  %slot = alloca i64 : ptr
  store %x, %slot align 8 : i64
  %two = add i64 1, i64 1 : i64
  %dead = mul %x, i64 9 : i64
  %c = icmp eq %two, i64 2 : i1
  cond_br %c, ^1(i64 0, i64 0), ^4
^1(%i: i64, %acc: i64):
  %done = icmp sge %i, %n : i1
  cond_br %done, ^3, ^2
^2:
  %v = load %slot align 8 : i64
  %inv = mul %x, %two : i64
  %s = sub %v, %v : i64
  %t = add %inv, %s : i64
  %acc2 = add %acc, %t : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %acc2)
^3:
  %r = call @helper(%acc) : i64
  %q = sdiv %r, %n : i64
  br ^5(%q)
^4:
  ret i64 -1
^5(%out: i64):
  ret %out
}
"#;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    (m, syms)
}

/// Every instruction line of `f`.
fn fn_lines(f: &Function) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for (_, bl) in f.blocks() {
        out.extend(bl.insts().iter().chain(bl.terminator().iter()).filter_map(|&i| f.inst_line(i)));
    }
    out
}

/// Every instruction line of every function of `m`.
fn all_lines(m: &Module) -> BTreeSet<u32> {
    m.functions().flat_map(fn_lines).collect()
}

/// `after` (the result of running `what` over `before`) verifies, keeps the
/// `decl_line` of every function that survives, and gives every instruction of
/// those functions a line taken from `before`.
fn check_lines(what: &str, before: &Module, after: &Module, syms: &StrInterner) {
    if let Err(e) = verify_module(after) {
        panic!("{what}: fails to verify: {e:?}\n{}", print_module(after, syms));
    }
    let allowed = all_lines(before);
    let mut seen = 0usize;
    for f in after.functions().filter(|f| !f.is_declaration()) {
        let Some(orig) = before.functions().find(|g| g.name == f.name) else {
            continue; // a function the pass introduced
        };
        assert_eq!(f.decl_line, orig.decl_line, "{what}: decl_line of {} lost", syms.resolve(f.name));
        for (_, bl) in f.blocks() {
            for &i in bl.insts().iter().chain(bl.terminator().iter()) {
                let line = f.inst_line(i);
                assert!(
                    line.is_some_and(|l| allowed.contains(&l)),
                    "{what}: instruction {:?} of {} has line {line:?}, not one of {allowed:?}\n{}",
                    f.inst(i).kind,
                    syms.resolve(f.name),
                    print_module(after, syms),
                );
                seen += 1;
            }
        }
    }
    assert!(seen > 0, "{what}: no instructions left to check");
}

#[test]
fn fixture_has_a_line_and_decl_line_everywhere() {
    let (m, syms) = parse(FIXTURE);
    for f in m.functions() {
        assert!(f.decl_line.is_some());
    }
    check_lines("parse", &m, &m, &syms);
}

#[test]
fn every_optimization_pass_keeps_lines() {
    for name in ["mem2reg", "sccp", "egraph", "simplify_cfg", "dce", "licm", "inline", "dfe"] {
        let (before, syms) = parse(FIXTURE);
        let mut m = before.clone();
        let mut pass = pass_by_name(name).expect("known pass");
        assert_eq!(pass.run(&mut m), Changed::Yes, "{name} has nothing to do on the fixture");
        check_lines(name, &before, &m, &syms);
    }
}

#[test]
fn every_pass_keeps_lines_after_mem2reg() {
    // The SSA form mem2reg exposes is what the later passes really see.
    let (mut before, syms) = parse(FIXTURE);
    pass_by_name("mem2reg").expect("known pass").run(&mut before);
    for name in ["sccp", "egraph", "simplify_cfg", "dce", "licm", "inline", "dfe"] {
        let mut m = before.clone();
        pass_by_name(name).expect("known pass").run(&mut m);
        check_lines(name, &before, &m, &syms);
    }
}

#[test]
fn every_pipeline_keeps_lines() {
    for level in [OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let (before, syms) = parse(FIXTURE);
        let mut m = before.clone();
        optimize(&mut m, level);
        check_lines(level.name(), &before, &m, &syms);
    }
}

#[test]
fn inlined_instructions_keep_the_callee_lines() {
    let (before, syms) = parse(FIXTURE);
    let helper = before.functions().find(|f| syms.resolve(f.name) == "helper").expect("helper");
    let callee_lines = fn_lines(helper);
    let mut m = before.clone();
    pass_by_name("mem2reg").expect("known pass").run(&mut m);
    pass_by_name("inline").expect("known pass").run(&mut m);
    let main = m.functions().find(|f| syms.resolve(f.name) == "main").expect("main");
    let main_lines = fn_lines(main);
    assert!(callee_lines.is_subset(&main_lines), "callee lines {callee_lines:?} missing from {main_lines:?}");
}

#[test]
fn licm_hoisted_instruction_keeps_its_line() {
    let (before, syms) = parse(FIXTURE);
    let line_of_inv = FIXTURE.lines().position(|l| l.contains("%inv = mul")).expect("fixture line") as u32 + 1;
    let mut m = before.clone();
    pass_by_name("licm").expect("known pass").run(&mut m);
    let main = m.functions().find(|f| syms.resolve(f.name) == "main").expect("main");
    let entry = main.entry().expect("body");
    let hoisted = main
        .blocks()
        .filter(|&(b, _)| b != entry)
        .flat_map(|(_, bl)| bl.insts().iter().copied())
        .find(|&i| matches!(main.inst(i).kind, crate::ir::InstKind::Bin(crate::ir::inst::BinOp::Mul)))
        .expect("the invariant multiply survives");
    assert_eq!(main.inst_line(hoisted), Some(line_of_inv));
}

#[test]
fn yield_points_and_sanitize_keep_lines() {
    let (before, mut syms) = parse(FIXTURE);
    let mut m = before.clone();
    let config = YieldConfig::new(syms.intern("__yield_flag"), syms.intern("__yield"));
    assert_eq!(YieldPoints::new(config).run(&mut m), Changed::Yes);
    check_lines("yield_points", &before, &m, &syms);

    let mut m = before.clone();
    let opts = SanitizeOptions {
        kinds: SanitizeKinds::parse("integer").expect("known kind"),
        runtime: SanitizeRuntime::External,
        ..SanitizeOptions::default()
    };
    let stats = sanitize_module(&mut m, &mut syms, "lines.lf", &opts).expect("sanitizes");
    assert!(stats.total() > 0, "sanitize inserted no checks");
    check_lines("sanitize", &before, &m, &syms);
}

#[test]
fn superopt_keeps_lines() {
    let src = r#"module "so"
func @f(i8) -> i8 {
entry ^0(%x: i8):
  %y = mul %x, i8 2 : i8
  ret %y
}
"#;
    let (before, syms) = parse(src);
    let mut m = before.clone();
    let f = FuncId::from_index(0);
    let fresh = superoptimize(&mut m, f).expect("x*2 has a cheaper form");
    m.replace_function(f, fresh);
    check_lines("superopt", &before, &m, &syms);
}

#[test]
fn codegen_legalizers_keep_lines() {
    let wide = r#"module "wide"
func @w(i128, i128) -> i128 {
entry ^0(%a: i128, %b: i128):
  %s = add %a, %b : i128
  %t = xor %s, %a : i128
  ret %t
}
"#;
    let (before, mut syms) = parse(wide);
    let mut m = before.clone();
    legalize_ints(&mut m, &mut syms, &LegalizeOptions::new(64)).expect("legalizes");
    check_lines("legalize_int", &before, &m, &syms);

    let vec = r#"module "vec"
func @v(<4 x i32>, <4 x i32>) -> i32 {
entry ^0(%a: <4 x i32>, %b: <4 x i32>):
  %s = add %a, %b : <4 x i32>
  %e = extractelement %s, 2 : i32
  ret %e
}
"#;
    let (before, syms) = parse(vec);
    let mut m = before.clone();
    legalize_vectors(&mut m, &ScalarOnly);
    check_lines("legalize_vectors", &before, &m, &syms);

    let float = r#"module "sf"
func @f(f32, f32) -> i1 {
entry ^0(%a: f32, %b: f32):
  %s = fadd %a, %b : f32
  %c = fcmp ult %s, %b : i1
  ret %c
}
"#;
    let (before, mut syms) = parse(float);
    let mut m = before.clone();
    lower_soft_float(&mut m, &mut syms, SoftFloatAbi::Aeabi).expect("lowers");
    check_lines("softfloat", &before, &m, &syms);
}

/// The builder-API reproduction from the issue.
#[test]
fn issue_19_repro() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        b.set_decl_line(3);
        b.set_line(4);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let two = b.const_i64(i64t, 2);
        let three = b.const_i64(i64t, 3);
        let five = b.add(two, three, Flags::default());
        b.set_line(5);
        let y = b.mul(x, five, Flags::default());
        b.ret(Some(y));
    }
    let lines = |m: &Module| {
        let func = m.function(f);
        let lines: Vec<_> = func
            .blocks()
            .flat_map(|(_, bl)| bl.insts().iter().copied().chain(bl.terminator()))
            .map(|i| func.inst_line(i))
            .collect();
        (func.decl_line, lines)
    };
    assert_eq!(lines(&m), (Some(3), vec![Some(4), Some(5), Some(5)]));
    for level in [OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let mut o = m.clone();
        optimize(&mut o, level);
        assert_eq!(lines(&o), (Some(3), vec![Some(5), Some(5)]), "{}", level.name());
    }
}
