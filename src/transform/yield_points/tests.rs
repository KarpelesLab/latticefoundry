//! Pass-level tests for [`super::YieldPoints`].

use super::*;
use crate::ir::text::{parse_module, print_module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::run_passes;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    (m, syms)
}

fn config(syms: &mut StrInterner) -> YieldConfig {
    YieldConfig::new(syms.intern("preempt_flag"), syms.intern("rt_yield"))
}

/// Run the pass; return the printed module and the number of checks (volatile
/// loads of the flag) per function name.
fn run(src: &str, max_cost: Option<u64>) -> (String, Vec<(String, usize)>) {
    let (mut m, mut syms) = parse(src);
    let mut cfg = config(&mut syms);
    if let Some(c) = max_cost {
        cfg.max_cost = c;
    }
    run_passes(&mut m, vec![Box::new(YieldPoints::new(cfg))]);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify after: {e:?}\n{}", print_module(&m, &syms)));
    let flag = find_global(&m, cfg.flag);
    let mut counts = Vec::new();
    for f in m.functions() {
        if f.is_declaration() {
            continue;
        }
        let n = f
            .blocks()
            .flat_map(|(_, b)| b.insts().to_vec())
            .filter(|&i| {
                let inst = f.inst(i);
                matches!(inst.kind, InstKind::Load { volatile: true, .. })
                    && flag.is_some_and(|g| matches!(f.value(inst.operands()[0]).def, ValueDef::Global(x) if x == g))
            })
            .count();
        counts.push((syms.resolve(f.name).to_owned(), n));
    }
    (print_module(&m, &syms), counts)
}

fn count(counts: &[(String, usize)], name: &str) -> usize {
    counts.iter().find(|(n, _)| n == name).map_or_else(|| panic!("no function {name}"), |c| c.1)
}

const LOOPS: &str = r#"
module "loops"
global @data : i64 = i64 0

; A loop whose exit depends on memory: no trip bound.
func @unbounded(ptr) -> i64 {
entry ^0(%p: ptr):
  br ^1(i64 0)
^1(%acc: i64):
  %v = load %p align 8 : i64
  %acc1 = add %acc, %v : i64
  %z = icmp eq %v, i64 0 : i1
  cond_br %z, ^2, ^1(%acc1)
^2:
  ret %acc1
}

; for i in 0..10: a small counted loop.
func @small() -> i64 {
entry ^0:
  br ^1(i64 0, i64 0)
^1(%i: i64, %s: i64):
  %c = icmp slt %i, i64 10 : i1
  cond_br %c, ^2, ^3
^2:
  %s1 = add %s, %i : i64
  %i1 = add %i, i64 1 : i64
  br ^1(%i1, %s1)
^3:
  ret %s
}

; for i in 0..1_000_000: counted, but too costly.
func @big() -> i64 {
entry ^0:
  br ^1(i64 0, i64 0)
^1(%i: i64, %s: i64):
  %c = icmp slt %i, i64 1000000 : i1
  cond_br %c, ^2, ^3
^2:
  %s1 = add %s, %i : i64
  %i1 = add %i, i64 1 : i64
  br ^1(%i1, %s1)
^3:
  ret %s
}

; for i in 0..(n & 15): bounded through the ranges domain; the exit test is
; on the stepped value in the latch (do-while shape).
func @masked(i64) -> i64 {
entry ^0(%n: i64):
  %m = and %n, i64 15 : i64
  br ^1(i64 0, i64 0)
^1(%i: i64, %s: i64):
  %s1 = add %s, %i : i64
  %i1 = add %i, i64 1 : i64
  %d = icmp uge %i1, %m : i1
  cond_br %d, ^2, ^1(%i1, %s1)
^2:
  ret %s1
}

; for i in 0..n with n unknown: unbounded.
func @param_bound(i64) -> i64 {
entry ^0(%n: i64):
  br ^1(i64 0, i64 0)
^1(%i: i64, %s: i64):
  %c = icmp slt %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %s1 = add %s, %i : i64
  %i1 = add %i, i64 1 : i64
  br ^1(%i1, %s1)
^3:
  ret %s
}

; An unbounded outer loop around a small counted inner loop: one check, on
; the outer loop's back edge only.
func @nested(ptr) -> i64 {
entry ^0(%p: ptr):
  br ^1(i64 0)
^1(%acc: i64):
  br ^2(i64 0, %acc)
^2(%j: i64, %a: i64):
  %a1 = add %a, %j : i64
  %j1 = add %j, i64 1 : i64
  %jd = icmp eq %j1, i64 8 : i1
  cond_br %jd, ^3, ^2(%j1, %a1)
^3:
  %v = load %p align 8 : i64
  %z = icmp eq %v, i64 0 : i1
  cond_br %z, ^4, ^1(%a1)
^4:
  ret %a1
}

; No loop at all.
func @straight(i64) -> i64 {
entry ^0(%x: i64):
  %y = add %x, i64 1 : i64
  ret %y
}
"#;

#[test]
fn checks_only_loops_without_a_small_proven_cost() {
    let (text, counts) = run(LOOPS, None);
    assert_eq!(count(&counts, "unbounded"), 1, "{text}");
    assert_eq!(count(&counts, "small"), 0, "{text}");
    assert_eq!(count(&counts, "big"), 1, "{text}");
    assert_eq!(count(&counts, "masked"), 0, "{text}");
    assert_eq!(count(&counts, "param_bound"), 1, "{text}");
    assert_eq!(count(&counts, "nested"), 1, "{text}");
    assert_eq!(count(&counts, "straight"), 0, "{text}");
    // The flag and the yield function were declared.
    assert!(text.contains("global detached @preempt_flag : i32"), "{text}");
    assert!(text.contains("func @rt_yield() -> void\n"), "{text}");
    assert!(text.contains("call @rt_yield() : void"), "{text}");
}

#[test]
fn threshold_is_configurable() {
    // With a huge threshold the million-iteration loop is "cheap enough".
    let (_, counts) = run(LOOPS, Some(u64::MAX / 2));
    assert_eq!(count(&counts, "big"), 0);
    assert_eq!(count(&counts, "unbounded"), 1, "no bound at all still gets a check");
    // With threshold 0 even the 10-iteration loop gets one.
    let (_, counts) = run(LOOPS, Some(0));
    assert_eq!(count(&counts, "small"), 1);
    assert_eq!(count(&counts, "masked"), 1);
}

#[test]
fn loop_reports_carry_trip_bounds_and_costs() {
    let (m, mut syms) = parse(LOOPS);
    let cfg = config(&mut syms);
    let by_name = |name: &str| {
        let fid = FuncId::from_index(m.functions().position(|f| syms.resolve(f.name) == name).unwrap());
        analyze_loops(&m, fid, &cfg)
    };
    let small = by_name("small");
    assert_eq!(small.len(), 1);
    // 10 staying tests + the failing one.
    assert_eq!(small[0].trip_bound, Some(11));
    assert!(matches!(small[0].cost, Cost::Bounded(c) if c < 200), "{:?}", small[0].cost);
    let masked = by_name("masked");
    // m <= 15 (known bits of `and %n, 15`): i = 0..=14, i.e. 15 header runs.
    assert_eq!(masked[0].trip_bound, Some(15));
    let unbounded = by_name("unbounded");
    assert_eq!(unbounded[0].trip_bound, None);
    assert_eq!(unbounded[0].cost, Cost::Unbounded);
    let nested = by_name("nested");
    assert_eq!(nested.len(), 2);
    let inner = nested.iter().find(|r| r.trip_bound.is_some()).expect("inner loop is counted");
    assert!(!inner.needs_check);
    assert!(nested.iter().any(|r| r.needs_check && r.cost == Cost::Unbounded));
}

#[test]
fn pass_is_idempotent() {
    let (mut m, mut syms) = parse(LOOPS);
    let cfg = config(&mut syms);
    run_passes(&mut m, vec![Box::new(YieldPoints::new(cfg))]);
    let once = print_module(&m, &syms);
    let mut p = YieldPoints::new(cfg);
    assert_eq!(p.run(&mut m), Changed::No);
    assert_eq!(print_module(&m, &syms), once);
}

#[test]
fn cost_lattice_laws() {
    let b = Cost::Bounded;
    assert_eq!(b(3).join(b(5)), b(5));
    assert_eq!(b(3).join(Cost::Unbounded), Cost::Unbounded);
    assert_eq!(b(3).plus(b(4)), b(7));
    assert_eq!(b(u64::MAX).plus(b(1)), Cost::Unbounded);
    assert_eq!(b(3).times(4), b(12));
    assert_eq!(b(u64::MAX).times(2), Cost::Unbounded);
    assert!(b(10).at_most(10) && !b(11).at_most(10) && !Cost::Unbounded.at_most(u64::MAX));
}

/// The pass runs after the `-O` pipeline; the result still verifies and the
/// inserted load stays inside the loop (nothing hoists it).
#[test]
fn optimize_with_yield_points_keeps_the_check_in_the_loop() {
    let (mut m, mut syms) = parse(LOOPS);
    let cfg = config(&mut syms);
    optimize_with_yield_points(&mut m, OptLevel::O2, cfg);
    crate::verify::verify_module(&m).unwrap();
    let fid = FuncId::from_index(m.functions().position(|f| syms.resolve(f.name) == "unbounded").unwrap());
    let reports = analyze_loops(&m, fid, &cfg);
    assert!(!reports.is_empty() && reports.iter().all(|r| !r.needs_check), "already checked");
}
