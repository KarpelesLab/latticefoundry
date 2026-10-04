//! Regression tests for the post-inlining clean-up (GitHub issues #7 and #10):
//! dead-function elimination ([`DeadFunctionElim`]) and the [`Module`]
//! function-removal API it is built on, the `-O2` CFG clean-up after inlining,
//! SCCP's propagation of a symbol address through block parameters, and DCE's
//! removal of unused block parameters. Every result is verified and executed
//! by the reference executor against the unoptimized program.

use crate::ir::refexec::run_named;
use crate::ir::semantics::SemValue;
use crate::ir::text::{parse_module, print_module};
use crate::ir::value::{AddrTarget, Const, ValueDef};
use crate::ir::{
    FuncAttrs, FuncId, Function, Global, GlobalAttrs, InstKind, Linkage, Module, Referrer,
    RemoveFunctionError, Visibility,
};
use crate::pass::ModulePass;
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize, pass_by_name, run_passes};
use crate::transform::{Dce, DeadFunctionElim, FunctionTransformPass, Mem2Reg};
use crate::verify::{RefinementResult, check_refinement, verify_module};

use puremp::Int;

const LEVELS: [OptLevel; 4] = [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3];

/// Issue #7: `@helper` is inlined into `@main`, `@unused` is never referenced.
const DEAD_LF: &str = r#"
module "dead"
func internal @helper(i64) -> i64 {
entry ^0(%0: i64):
  %1 = add %0, i64 1 : i64
  ret %1
}
func internal @unused() -> i64 {
entry ^0:
  ret i64 7
}
func @main() -> i64 {
entry ^0:
  %0 = call @helper(i64 41) : i64
  ret %0
}
"#;

/// Issue #10: after `mem2reg` the loop header gets a parameter for `%slot`
/// that nothing reads.
const M2R_LF: &str = r#"
module "m2r"
func @g(i64) -> i64 {
entry ^0(%n0: i64):
  %slot = alloca i64 : ptr
  br ^1(%n0)
^1(%n: i64):
  %z = icmp eq %n, i64 0 : i1
  cond_br %z, ^2, ^3
^2:
  ret i64 0
^3:
  store %n, %slot align 8 : i64
  %t = load %slot align 8 : i64
  %n2 = sub %t, i64 1 : i64
  br ^1(%n2)
}
func @main() -> i64 {
entry ^0:
  %r = call @g(i64 3) : i64
  ret %r
}
"#;

/// A join block whose pointer parameter receives `@msg` on both edges (the
/// shape inlining a call with a global argument leaves behind).
const GJOIN_LF: &str = r#"
module "gjoin"
global internal constant @msg : [4 x i8] = [4 x i8] "ABCD"
global internal constant @other : [4 x i8] = [4 x i8] "abcd"
func @pick(i64) -> i64 {
entry ^0(%n: i64):
  %z = icmp eq %n, i64 0 : i1
  cond_br %z, ^1, ^2
^1:
  br ^3(@msg, i64 1)
^2:
  br ^3(@msg, i64 2)
^3(%p: ptr, %k: i64):
  %q = ptr_add %p, %k : ptr
  %c = load %q align 1 : i8
  %w = zext %c : i64
  ret %w
}
func @mixed(i64) -> i64 {
entry ^0(%n: i64):
  %z = icmp eq %n, i64 0 : i1
  cond_br %z, ^1, ^2
^1:
  br ^3(@msg)
^2:
  br ^3(@other)
^3(%p: ptr):
  %c = load %p align 1 : i8
  %w = zext %c : i64
  ret %w
}
func @looped(i64) -> i64 {
entry ^0(%n: i64):
  br ^1(@msg, %n)
^1(%p: ptr, %i: i64):
  %z = icmp eq %i, i64 0 : i1
  cond_br %z, ^2, ^3
^3:
  %i2 = sub %i, i64 1 : i64
  br ^1(%p, %i2)
^2:
  %c = load %p align 1 : i8
  %w = zext %c : i64
  ret %w
}
"#;

/// Dead block parameters through every kind of edge: a `switch`, a `cond_br`
/// and a loop that threads a value only to itself.
const PARAMS_LF: &str = r#"
module "params"
func @sw(i64) -> i64 {
entry ^0(%x: i64):
  %d = mul %x, i64 3 : i64
  switch %x, ^1(%x, %d) [1: ^2(%d, %x), 2: ^1(%x, %d)]
^1(%a: i64, %b: i64):
  %r = add %a, i64 10 : i64
  ret %r
^2(%c: i64, %e: i64):
  %t = icmp ne %e, i64 0 : i1
  cond_br %t, ^3(%c, %e), ^3(%e, %e)
^3(%f: i64, %g: i64):
  ret %g
}
func @spin(i64) -> i64 {
entry ^0(%n: i64):
  %junk = mul %n, %n : i64
  br ^1(%n, %junk, i64 0)
^1(%i: i64, %acc: i64, %s: i64):
  %z = icmp eq %i, i64 0 : i1
  cond_br %z, ^2, ^3
^3:
  %i2 = sub %i, i64 1 : i64
  %acc2 = add %acc, %i : i64
  %s2 = add %s, i64 2 : i64
  br ^1(%i2, %acc2, %s2)
^2:
  ret %s
}
"#;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    (m, syms)
}

fn func_id(m: &Module, syms: &StrInterner, name: &str) -> FuncId {
    m.func_ids().find(|&f| syms.resolve(m.function(f).name) == name).unwrap_or_else(|| panic!("no @{name}"))
}

fn names(m: &Module, syms: &StrInterner) -> Vec<String> {
    m.functions().map(|f| syms.resolve(f.name).to_owned()).collect()
}

fn verified(m: &Module, syms: &StrInterner, what: &str) {
    if let Err(e) = verify_module(m) {
        panic!("{what}: {e:#?}\n{}", print_module(m, syms));
    }
}

fn i64v(n: i64) -> SemValue {
    SemValue::int(64, Int::from_i64(n))
}

/// Run `name(args)` on `orig` and `opt`; the optimized result must refine the
/// original, and the original must produce `expect`.
fn agree(orig: &(Module, StrInterner), opt: &(Module, StrInterner), name: &str, args: &[i64], expect: i64) {
    let args: Vec<SemValue> = args.iter().map(|&a| i64v(a)).collect();
    let want = run_named(&orig.0, &orig.1, name, &args).expect("source runs").expect("a result");
    let got = run_named(&opt.0, &opt.1, name, &args).expect("optimized runs").expect("a result");
    assert_eq!(want, i64v(expect), "@{name}{args:?} reference result");
    assert!(got.refines(&want), "@{name}{args:?}: {got:?} vs {want:?}");
}

/// The non-entry block parameters nothing uses, over every function.
fn unused_params(m: &Module) -> usize {
    m.functions()
        .map(|f| {
            f.blocks()
                .filter(|&(b, _)| Some(b) != f.entry())
                .flat_map(|(_, blk)| blk.params().iter())
                .filter(|&&p| f.uses_of(p).is_empty())
                .count()
        })
        .sum()
}

/// The number of `mul` instructions in a function.
fn muls(f: &Function) -> usize {
    f.blocks()
        .flat_map(|(_, b)| b.insts().iter())
        .filter(|&&i| matches!(f.inst(i).kind, InstKind::Bin(crate::ir::BinOp::Mul)))
        .count()
}

/// Blocks whose only content is an unconditional `br` (left by inlining).
fn jump_only_blocks(f: &Function) -> usize {
    f.blocks()
        .filter(|(_, b)| {
            b.insts().is_empty() && b.terminator().is_some_and(|t| matches!(f.inst(t).kind, InstKind::Br(_)))
        })
        .count()
}

// ---------------------------------------------------------------------------
// Issue #7
// ---------------------------------------------------------------------------

#[test]
fn issue7_repro_o2_is_one_function_returning_a_constant() {
    let orig = parse(DEAD_LF);
    for level in [OptLevel::O2, OptLevel::O3] {
        let mut opt = parse(DEAD_LF);
        optimize(&mut opt.0, level);
        verified(&opt.0, &opt.1, level.name());
        assert_eq!(names(&opt.0, &opt.1), ["main"], "{level:?}: the inlined and unused helpers go");
        let main = opt.0.function(func_id(&opt.0, &opt.1, "main"));
        assert_eq!(main.block_count(), 1, "{level:?}: no jump-only blocks\n{}", print_module(&opt.0, &opt.1));
        agree(&orig, &opt, "main", &[], 42);
    }
    // -O1 does not inline, but still drops the never-referenced function.
    let mut o1 = parse(DEAD_LF);
    optimize(&mut o1.0, OptLevel::O1);
    verified(&o1.0, &o1.1, "O1");
    assert_eq!(names(&o1.0, &o1.1), ["helper", "main"]);
    agree(&orig, &o1, "main", &[], 42);
}

#[test]
fn o2_cleans_the_cfg_after_inlining() {
    // A callee with a branch: inlining splits the caller; -O2 must merge the
    // jump-only blocks back.
    const SRC: &str = r#"
module "cfg"
func internal @abs(i64) -> i64 {
entry ^0(%x: i64):
  %neg = icmp slt %x, i64 0 : i1
  cond_br %neg, ^1, ^2(%x)
^1:
  %m = sub i64 0, %x : i64
  br ^2(%m)
^2(%r: i64):
  ret %r
}
func @main(i64) -> i64 {
entry ^0(%v: i64):
  %a = call @abs(%v) : i64
  %b = add %a, i64 1 : i64
  ret %b
}
"#;
    let orig = parse(SRC);
    let mut opt = parse(SRC);
    optimize(&mut opt.0, OptLevel::O2);
    verified(&opt.0, &opt.1, "O2");
    assert_eq!(names(&opt.0, &opt.1), ["main"]);
    let main = opt.0.function(func_id(&opt.0, &opt.1, "main"));
    assert_eq!(jump_only_blocks(main), 0, "{}", print_module(&opt.0, &opt.1));
    for (v, want) in [(-5, 6), (0, 1), (7, 8)] {
        agree(&orig, &opt, "main", &[v], want);
    }
}

#[test]
fn every_level_keeps_the_program_meaning() {
    for src in [DEAD_LF, M2R_LF, GJOIN_LF, PARAMS_LF] {
        let orig = parse(src);
        for level in LEVELS {
            let mut opt = parse(src);
            optimize(&mut opt.0, level);
            verified(&opt.0, &opt.1, level.name());
            for f in orig.0.functions() {
                let name = orig.1.resolve(f.name).to_owned();
                let nparams = match orig.0.types().get(f.sig) {
                    crate::ir::types::Type::Func(ft) => ft.params.len(),
                    _ => unreachable!(),
                };
                for x in [0i64, 1, 2, 3, 9] {
                    let args: Vec<SemValue> = (0..nparams).map(|_| i64v(x)).collect();
                    let want = run_named(&orig.0, &orig.1, &name, &args).expect("source runs");
                    if opt.0.functions().all(|g| opt.1.resolve(g.name) != name) {
                        continue; // an internal function that was removed
                    }
                    let got = run_named(&opt.0, &opt.1, &name, &args).expect("optimized runs");
                    let (want, got) = (want.expect("result"), got.expect("result"));
                    assert!(got.refines(&want), "{level:?} @{name}({x}): {got:?} vs {want:?}");
                }
            }
        }
    }
}

#[test]
fn dfe_removes_transitive_orphans_and_dead_cycles() {
    const SRC: &str = r#"
module "orphans"
func internal @leaf() -> i64 {
entry ^0:
  ret i64 1
}
func internal @mid() -> i64 {
entry ^0:
  %a = call @leaf() : i64
  ret %a
}
func internal @top() -> i64 {
entry ^0:
  %a = call @mid() : i64
  ret %a
}
func internal @ping(i64) -> i64 {
entry ^0(%n: i64):
  %a = call @pong(%n) : i64
  ret %a
}
func internal @pong(i64) -> i64 {
entry ^0(%n: i64):
  %a = call @ping(%n) : i64
  ret %a
}
func internal @used(i64) -> i64 {
entry ^0(%n: i64):
  %a = add %n, i64 5 : i64
  ret %a
}
func @main(i64) -> i64 {
entry ^0(%n: i64):
  %a = call @used(%n) : i64
  ret %a
}
"#;
    let orig = parse(SRC);
    let mut opt = parse(SRC);
    let mut p = DeadFunctionElim::new();
    assert_eq!(p.run(&mut opt.0), crate::pass::Changed::Yes);
    verified(&opt.0, &opt.1, "dfe");
    assert_eq!(names(&opt.0, &opt.1), ["used", "main"]);
    // The call in @main follows @used to its new id.
    let main = opt.0.function(func_id(&opt.0, &opt.1, "main"));
    let callee = main
        .blocks()
        .flat_map(|(_, b)| b.insts().iter())
        .find(|&&i| matches!(main.inst(i).kind, InstKind::Call))
        .map(|&i| main.inst(i).operands()[0])
        .expect("the call stays");
    assert_eq!(main.value(callee).def, ValueDef::Func(func_id(&opt.0, &opt.1, "used")));
    agree(&orig, &opt, "main", &[4], 9);
    // A second run has nothing left to do.
    assert_eq!(p.run(&mut opt.0), crate::pass::Changed::No);
}

#[test]
fn dfe_keeps_public_weak_declared_and_address_taken_functions() {
    const SRC: &str = r#"
module "keep"
func @public() -> i64 {
entry ^0:
  ret i64 1
}
func weak hidden @weakly() -> void {
entry ^0:
  ret
}
func @external(i64) -> i64
func internal @dead() -> i64 {
entry ^0:
  ret i64 2
}
func internal @in_table() -> i64 {
entry ^0:
  %a = call @callee_of_table() : i64
  ret %a
}
func internal @callee_of_table() -> i64 {
entry ^0:
  ret i64 3
}
global internal constant @tab : [2 x ptr] = [2 x ptr] (ptr @in_table, ptr @public)
"#;
    let (mut m, syms) = parse(SRC);
    run_passes(&mut m, vec![pass_by_name("dfe").expect("dfe")]);
    verified(&m, &syms, "dfe");
    assert_eq!(names(&m, &syms), ["public", "weakly", "external", "in_table", "callee_of_table"]);
    // The table's address constants follow the renumbering.
    let tab = m.global(crate::ir::GlobalId::from_index(0)).init.expect("init");
    let Const::Aggregate { elems, .. } = m.consts().get(tab) else { panic!("aggregate") };
    let targets: Vec<FuncId> = elems
        .iter()
        .map(|&e| match m.consts().get(e) {
            Const::Addr { target: AddrTarget::Func(f), .. } => *f,
            c => panic!("unexpected {c:?}"),
        })
        .collect();
    assert_eq!(targets, [func_id(&m, &syms, "in_table"), func_id(&m, &syms, "public")]);
    let text = print_module(&m, &syms);
    assert!(text.contains("ptr @in_table, ptr @public"), "{text}");
}

#[test]
fn dfe_keeps_a_function_aliased_by_a_global_of_its_name() {
    // A frontend may take a function's address through a body-less global of
    // the same symbol name (lf-cc does); that reference is invisible to the IR.
    let (mut m, mut syms) = parse(DEAD_LF);
    let ptr = m.types_mut().ptr();
    let name = syms.intern("unused");
    m.define_global(Global { name, ty: ptr, init: None }, GlobalAttrs { linkage: Linkage::Internal, ..GlobalAttrs::DETACHED });
    run_passes(&mut m, vec![pass_by_name("dfe").expect("dfe")]);
    assert_eq!(names(&m, &syms), ["helper", "unused", "main"]);
}

// ---------------------------------------------------------------------------
// The Module API
// ---------------------------------------------------------------------------

#[test]
fn module_api_enumerates_and_finds_functions() {
    let (m, syms) = parse(DEAD_LF);
    let ids: Vec<FuncId> = m.func_ids().collect();
    assert_eq!(ids.len(), 3);
    assert_eq!(ids, (0..3).map(FuncId::from_index).collect::<Vec<_>>());
    let main = m.function_by_name(m.function(ids[2]).name).expect("main");
    assert_eq!(main, ids[2]);
    assert_eq!(syms.resolve(m.function(main).name), "main");
    assert_eq!(m.referenced_functions(main), [ids[0]]);
    assert!(m.referenced_functions(ids[0]).is_empty());
}

#[test]
fn module_remove_function_refuses_live_references() {
    let (mut m, syms) = parse(DEAD_LF);
    let helper = func_id(&m, &syms, "helper");
    let main = func_id(&m, &syms, "main");
    assert_eq!(
        m.remove_function(helper),
        Err(RemoveFunctionError::StillReferenced { func: helper, by: Referrer::Function(main) })
    );
    assert_eq!(m.remove_function(FuncId::from_index(9)), Err(RemoveFunctionError::OutOfRange(FuncId::from_index(9))));
    assert_eq!(m.function_count(), 3, "a refused removal changes nothing");

    // Removing the caller together with its callee is fine; ids compact.
    let remap = m.remove_functions(&[helper, main]).expect("removable together");
    assert_eq!(remap, [None, Some(FuncId::from_index(0)), None]);
    assert_eq!(names(&m, &syms), ["unused"]);
    verified(&m, &syms, "after removal");

    // A global initializer pins a function.
    const SRC: &str = r#"
module "g"
func internal @f() -> void {
entry ^0:
  ret
}
global @p : ptr = ptr @f
"#;
    let (mut m, _) = parse(SRC);
    let err = m.remove_function(FuncId::from_index(0)).expect_err("pinned by @p");
    assert_eq!(
        err,
        RemoveFunctionError::StillReferenced {
            func: FuncId::from_index(0),
            by: Referrer::Global(crate::ir::GlobalId::from_index(0)),
        }
    );
    assert!(!err.to_string().is_empty());
}

#[test]
fn module_remove_function_renumbers_and_tombstones_stale_references() {
    let (mut m, syms) = parse(DEAD_LF);
    let unused = func_id(&m, &syms, "unused");
    let helper = func_id(&m, &syms, "helper");
    let main = func_id(&m, &syms, "main");
    let ptr = m.types_mut().ptr();
    // An address constant nothing uses, and a use-less `func_ref` in @main.
    let stale = m.intern_const(Const::Addr { ty: ptr, target: AddrTarget::Func(unused), offset: 0 });
    let live = m.intern_const(Const::Addr { ty: ptr, target: AddrTarget::Func(main), offset: 8 });
    let poison = m.intern_const(Const::Poison(ptr));
    let stale_ref = {
        let mut b = m.build(main);
        b.func_ref(unused)
    };

    let remap = m.remove_function(unused).expect("unreferenced");
    assert_eq!(remap, [Some(FuncId::from_index(0)), None, Some(FuncId::from_index(1))]);
    let main = remap[main.index()].expect("main survives");
    let helper = remap[helper.index()].expect("helper survives");
    assert_eq!(names(&m, &syms), ["helper", "main"]);
    verified(&m, &syms, "after removal");

    // Constants: the stale one is a tombstone, the live one follows @main, and
    // interning still deduplicates to the original handles.
    assert_eq!(m.consts().get(stale), &Const::Poison(ptr));
    assert_eq!(m.consts().get(live), &Const::Addr { ty: ptr, target: AddrTarget::Func(main), offset: 8 });
    assert_eq!(m.intern_const(Const::Poison(ptr)), poison);
    assert_eq!(m.intern_const(Const::Addr { ty: ptr, target: AddrTarget::Func(main), offset: 8 }), live);

    // The use-less reference is a poison now, the call still targets @helper,
    // and a fresh `func_ref` dedups to the existing value.
    let f = m.function(main);
    assert!(matches!(f.value(stale_ref).def, ValueDef::Const(c) if m.consts().get(c) == &Const::Poison(ptr)));
    let call_ref = f
        .blocks()
        .flat_map(|(_, b)| b.insts().iter())
        .find(|&&i| matches!(f.inst(i).kind, InstKind::Call))
        .map(|&i| f.inst(i).operands()[0])
        .expect("call");
    assert_eq!(f.value(call_ref).def, ValueDef::Func(helper));
    let again = m.build(main).func_ref(helper);
    assert_eq!(again, call_ref);
    let orig = parse(DEAD_LF);
    agree(&orig, &(m, syms), "main", &[], 42);
}

#[test]
fn module_remove_function_keeps_attributes_with_the_function() {
    let (mut m, syms) = parse(DEAD_LF);
    let main = func_id(&m, &syms, "main");
    m.set_func_attrs(main, FuncAttrs::new(Linkage::Weak, Visibility::Hidden));
    m.set_ret_secret(main, true);
    let remap = m.remove_function(func_id(&m, &syms, "unused")).expect("removable");
    let main = remap[main.index()].expect("kept");
    assert_eq!(m.func_attrs(main).linkage, Linkage::Weak);
    assert!(m.func_attrs(main).secret_ret);
}

// ---------------------------------------------------------------------------
// SCCP: symbol addresses through block parameters
// ---------------------------------------------------------------------------

#[test]
fn sccp_propagates_a_global_address_passed_on_every_edge() {
    let orig = parse(GJOIN_LF);
    let mut opt = parse(GJOIN_LF);
    run_passes(&mut opt.0, vec![pass_by_name("sccp").unwrap(), pass_by_name("dce").unwrap()]);
    verified(&opt.0, &opt.1, "sccp,dce");
    let text = print_module(&opt.0, &opt.1);
    // @pick and @looped: the pointer parameter is gone, @msg used directly.
    for name in ["pick", "looped"] {
        let f = opt.0.function(func_id(&opt.0, &opt.1, name));
        for (b, blk) in f.blocks() {
            if Some(b) != f.entry() {
                assert!(
                    blk.params().iter().all(|&p| !matches!(opt.0.types().get(f.value_type(p)), crate::ir::types::Type::Ptr)),
                    "@{name} keeps a pointer parameter\n{text}"
                );
            }
        }
    }
    // @mixed receives two different globals: the parameter stays.
    let mixed = opt.0.function(func_id(&opt.0, &opt.1, "mixed"));
    assert_eq!(mixed.blocks().map(|(_, b)| b.params().len()).sum::<usize>(), 2, "{text}");
    agree(&orig, &opt, "pick", &[0], 66);
    agree(&orig, &opt, "pick", &[5], 67);
    agree(&orig, &opt, "mixed", &[0], 65);
    agree(&orig, &opt, "mixed", &[1], 97);
    agree(&orig, &opt, "looped", &[3], 65);
}

// ---------------------------------------------------------------------------
// Issue #10: unused block parameters
// ---------------------------------------------------------------------------

#[test]
fn issue10_repro_header_keeps_one_parameter() {
    let orig = parse(M2R_LF);
    // mem2reg alone leaves the dead parameter; DCE removes it.
    let mut m = parse(M2R_LF);
    let g = func_id(&m.0, &m.1, "g");
    run_passes(&mut m.0, vec![Box::new(FunctionTransformPass::new(Mem2Reg))]);
    assert!(unused_params(&m.0) > 0, "mem2reg leaves a dead header parameter");
    run_passes(&mut m.0, vec![Box::new(FunctionTransformPass::new(Dce))]);
    verified(&m.0, &m.1, "mem2reg,dce");
    assert_eq!(unused_params(&m.0), 0, "{}", print_module(&m.0, &m.1));
    let header = m.0.function(g).blocks().map(|(_, b)| b.params().len()).collect::<Vec<_>>();
    assert_eq!(header, [1, 1, 0, 0], "entry(%n0), ^1(%n)");
    for n in [0, 1, 3, 7] {
        agree(&orig, &m, "g", &[n], 0);
    }

    for level in [OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let mut opt = parse(M2R_LF);
        optimize(&mut opt.0, level);
        verified(&opt.0, &opt.1, level.name());
        assert_eq!(unused_params(&opt.0), 0, "{level:?}\n{}", print_module(&opt.0, &opt.1));
        let g = opt.0.function(func_id(&opt.0, &opt.1, "g"));
        assert_eq!(g.blocks().map(|(_, b)| b.params().len()).max(), Some(1));
        agree(&orig, &opt, "g", &[3], 0);
        agree(&orig, &opt, "main", &[], 0);
    }
}

#[test]
fn dce_drops_dead_parameters_on_every_edge_kind() {
    let orig = parse(PARAMS_LF);
    let mut opt = parse(PARAMS_LF);
    run_passes(&mut opt.0, vec![pass_by_name("dce").unwrap()]);
    verified(&opt.0, &opt.1, "dce");
    let text = print_module(&opt.0, &opt.1);
    assert_eq!(unused_params(&opt.0), 0, "{text}");
    let sw = opt.0.function(func_id(&opt.0, &opt.1, "sw"));
    // ^1 keeps %a, ^2 keeps %e (the condition), ^3 keeps %g.
    assert_eq!(sw.blocks().map(|(_, b)| b.params().len()).collect::<Vec<_>>(), [1, 1, 1, 1], "{text}");
    // The `%d = mul` fed only dropped arguments.
    assert_eq!(muls(sw), 0, "{text}");
    // @spin threads %acc only to itself: the whole cycle (and %junk) goes.
    let spin = opt.0.function(func_id(&opt.0, &opt.1, "spin"));
    assert_eq!(spin.blocks().map(|(_, b)| b.params().len()).collect::<Vec<_>>(), [1, 2, 0, 0], "{text}");
    assert_eq!(muls(spin), 0, "{text}");
    for x in [0, 1, 2, 5] {
        let want_sw = if x == 1 { 1 } else { x + 10 };
        agree(&orig, &opt, "sw", &[x], want_sw);
        agree(&orig, &opt, "spin", &[x], 2 * x);
    }
    // Idempotent.
    let before = print_module(&opt.0, &opt.1);
    run_passes(&mut opt.0, vec![pass_by_name("dce").unwrap()]);
    assert_eq!(print_module(&opt.0, &opt.1), before);
}

#[test]
fn dce_keeps_entry_parameters() {
    const SRC: &str = r#"
module "e"
func @f(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  ret i64 1
}
"#;
    let (mut m, syms) = parse(SRC);
    run_passes(&mut m, vec![pass_by_name("dce").unwrap()]);
    verified(&m, &syms, "dce");
    let f = m.function(FuncId::from_index(0));
    assert_eq!(f.block(f.entry().unwrap()).params().len(), 2);
}

#[test]
fn dce_parameter_removal_is_a_proven_refinement() {
    // Acyclic, pure integer: in the refinement checker's subset.
    const SRC: &str = r#"
module "r"
func @f(i32, i1) -> i32 {
entry ^0(%x: i32, %c: i1):
  %y = mul %x, i32 7 : i32
  cond_br %c, ^1(%x, %y), ^1(%y, %x)
^1(%a: i32, %b: i32):
  %r = add %a, i32 1 : i32
  ret %r
}
"#;
    let (mut m, _syms) = parse(SRC);
    let f = FuncId::from_index(0);
    let mut t = Dce;
    let (fresh, c) = m.map_function(f, |old, b| crate::transform::FunctionTransform::run(&mut t, old, b));
    assert_eq!(c, crate::pass::Changed::Yes);
    assert_eq!(fresh.blocks().map(|(_, b)| b.params().len()).collect::<Vec<_>>(), [2, 1]);
    match check_refinement(m.types(), m.consts(), m.function(f), &fresh) {
        RefinementResult::Refines => {}
        other => panic!("DCE must refine, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Native execution and image size (x86-64 Linux)
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use super::*;
    use crate::link::{ImageOptions, link_executable, write_executable};

    fn image(m: &Module, syms: &StrInterner) -> Vec<u8> {
        let obj = crate::target::x86_64::compile_module(m, syms);
        link_executable(vec![obj], &ImageOptions::default()).expect("link")
    }

    fn run(image: &[u8], tag: &str) -> i32 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("lf_cleanup_{tag}_{}_{uniq}", std::process::id()));
        write_executable(path.to_str().unwrap(), image).expect("write exe");
        let status = loop {
            match std::process::Command::new(&path).status() {
                Ok(s) => break s,
                Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
                Err(e) => panic!("exec: {e}"),
            }
        };
        let _ = std::fs::remove_file(&path);
        status.code().expect("exited via code")
    }

    #[test]
    fn issue_repros_run_and_o2_is_not_larger_than_o1() {
        for (src, exit) in [(DEAD_LF, 42), (M2R_LF, 0)] {
            let mut sizes = Vec::new();
            for level in LEVELS {
                let (mut m, syms) = parse(src);
                optimize(&mut m, level);
                let img = image(&m, &syms);
                assert_eq!(run(&img, level.name()), exit, "{level:?}");
                sizes.push(img.len());
            }
            if src == DEAD_LF {
                assert!(sizes[2] <= sizes[1], "-O2 ({}) must not be larger than -O1 ({})", sizes[2], sizes[1]);
                assert!(sizes[2] < sizes[0], "-O2 must shrink the -O0 image");
            }
        }
    }
}
