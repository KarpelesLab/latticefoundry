//! Tests for the `inline_asm` / `asm_output` instructions (`docs/ir-design.md`
//! §6j): text and binary round trips, the builder, the verifier, the
//! analyses' and passes' conservative treatment, the refinement and
//! constant-time checkers, and the targets without an inline-asm lowering.

use crate::ir::inst::{AsmInput, AsmOutput, Flags, InlineAsm, InstKind};
use crate::ir::{FuncId, Function, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

/// A module exercising every operand form.
const ASM_LF: &str = r#"module "asm"

global @g : i64 = i64 7

func @f(i64, i64, ptr) -> i64 {
entry ^0(%0: i64, %1: i64, %2: ptr):
  %3 = inline_asm volatile "rdtsc" outs("=a" i32, "=d" i32) : i32
  %4 = asm_output %3, 1 : i32
  %5 = inline_asm "addq %2, %0" outs("=r" [sum] i64) ins("0" (%0), "r" [b] (%1)) clobbers("cc") : i64
  %6 = inline_asm "xchgq %0, %1" outs("+r" i64 (%5), "+m" (%2)) clobbers("memory") : i64
  inline_asm volatile "movq %1, %0\n\tnop # \"q\"" outs("=m" (%2)) ins("r" (%6)) : void
  %7 = inline_asm "leaq %c1(%%rip), %0" outs("=r" ptr) ins("i" (@g), "n" (i64 -3), "m" (%2)) : ptr
  %8 = inline_asm "" outs("+x" f64 (f64 0x4045000000000000), "=r" i64) : f64
  %9 = asm_output %8, 1 : i64
  %10 = zext %4 : i64
  %11 = add %10, %9 : i64
  ret %11
}
"#;

fn parse(src: &str, syms: &mut StrInterner) -> Module {
    crate::ir::text::parse_module(src, FileId::new(0), syms).unwrap_or_else(|e| panic!("parse: {e:?}"))
}

fn count(f: &Function, pred: impl Fn(&InstKind) -> bool) -> usize {
    f.blocks().map(|(_, b)| b.insts().iter().filter(|&&i| pred(&f.inst(i).kind)).count()).sum()
}

fn is_asm(k: &InstKind) -> bool {
    matches!(k, InstKind::InlineAsm(_))
}

#[test]
fn text_and_binary_round_trip() {
    let mut syms = StrInterner::new();
    let m = parse(ASM_LF, &mut syms);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let text = crate::ir::text::print_module(&m, &syms);
    for needle in [
        r#"inline_asm volatile "rdtsc" outs("=a" i32, "=d" i32) : i32"#,
        r#"= asm_output %3, 1 : i32"#,
        r#"outs("=r" [sum] i64) ins("0" (%0), "r" [b] (%1)) clobbers("cc") : i64"#,
        r#"outs("+r" i64 (%5), "+m" (%2)) clobbers("memory") : i64"#,
        r#"inline_asm volatile "movq %1, %0\n\tnop # \"q\"" outs("=m" (%2)) ins("r" (%6)) : void"#,
        r#"ins("i" (@g), "n" (i64 -3), "m" (%2)) : ptr"#,
    ] {
        assert!(text.contains(needle), "missing `{needle}` in\n{text}");
    }
    let again = parse(&text, &mut syms);
    assert_eq!(crate::ir::text::print_module(&again, &syms), text);

    let bytes = crate::ir::binary::encode(&m, &syms);
    let mut back = StrInterner::new();
    let m2 = crate::ir::binary::decode(&bytes, &mut back).expect("decode");
    assert_eq!(crate::ir::binary::encode(&m2, &back), bytes);
    assert_eq!(crate::ir::text::print_module(&m2, &back), text);
    // Truncations of the stream are errors, never panics.
    for cut in (bytes.len() / 2..bytes.len()).step_by(7) {
        assert!(crate::ir::binary::decode(&bytes[..cut], &mut StrInterner::new()).is_err());
    }
}

#[test]
fn builder_and_operand_layout() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("b");
    let i64t = m.types_mut().int(64);
    let i32t = m.types_mut().int(32);
    let ptr = m.types_mut().ptr();
    let sig = m.types_mut().func(vec![i64t, ptr], i64t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    let asm = InlineAsm {
        template: "cpuid".into(),
        outputs: vec![
            AsmOutput { constraint: "=a".into(), name: None, ty: Some(i32t) },
            AsmOutput { constraint: "+m".into(), name: None, ty: None },
            AsmOutput { constraint: "+b".into(), name: Some("x".into()), ty: Some(i64t) },
        ],
        inputs: vec![AsmInput { constraint: "0".into(), name: None }],
        clobbers: vec!["memory".into()],
        volatile: false,
    };
    assert_eq!(
        asm.operand_slots(),
        [crate::ir::AsmSlot::Output(1), crate::ir::AsmSlot::Output(2), crate::ir::AsmSlot::Input(0)]
    );
    assert_eq!(asm.result_output(), Some(0));
    assert_eq!(asm.register_outputs().collect::<Vec<_>>(), [0, 2]);
    assert_eq!(asm.tied_output("0"), Some(0));
    assert_eq!(asm.tied_output("[x]"), Some(2));
    assert!(asm.may_access_memory() && asm.has_side_effect());
    {
        let mut b = m.build(f);
        let e = b.create_entry_block();
        let (a, p) = (b.param(e, 0), b.param(e, 1));
        let a32 = b.cast(crate::ir::CastOp::Trunc, a, i32t);
        let r = b.inline_asm(asm, &[p, a, a32]).expect("a register output");
        let x = b.asm_output(r, 2, i64t);
        b.ret(Some(x));
    }
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let text = crate::ir::text::print_module(&m, &syms);
    assert!(text.contains(r#"outs("=a" i32, "+m" (%1), "+b" [x] i64 (%0)) ins("0" (%2)) clobbers("memory") : i32"#), "{text}");

    // Purity: only a non-volatile asm without memory access is removable.
    let pure = InlineAsm { template: "x".into(), outputs: vec![], inputs: vec![], clobbers: vec!["cc".into()], volatile: false };
    assert!(!InstKind::InlineAsm(Box::new(pure.clone())).has_side_effect());
    let vol = InlineAsm { volatile: true, ..pure.clone() };
    assert!(InstKind::InlineAsm(Box::new(vol)).has_side_effect());
    let mem = InlineAsm { clobbers: vec!["memory".into()], ..pure };
    assert!(InstKind::InlineAsm(Box::new(mem)).has_side_effect());
    assert!(!InstKind::AsmOutput(1).has_side_effect());
    assert!(InlineAsm::is_indirect("=m") && InlineAsm::is_indirect("+&o") && !InlineAsm::is_indirect("rm"));
    assert!(InlineAsm::is_immediate_only("i") && InlineAsm::is_immediate_only("n") && !InlineAsm::is_immediate_only("ri"));
}

/// The types a malformed-asm case is built from.
#[derive(Clone, Copy)]
struct Tys {
    i64t: crate::ir::TypeId,
    i32t: crate::ir::TypeId,
}

type Extra = Box<dyn FnOnce(&mut crate::ir::builder::FunctionBuilder<'_>, Option<crate::ir::ValueId>, Tys)>;

/// Build `@f(i64, ptr) -> i64` whose body is the asm `make` builds, with
/// operands `ops` (by index: 0 = the `i64` parameter, 1 = the pointer, 2 = a
/// constant 5) and result type `ty` (`None` = void), then `extra`, then
/// `ret 0`; return the verifier's messages.
fn verify_asm(make: impl FnOnce(Tys) -> InlineAsm, ops: &[usize], ty: Option<&str>, extra: Option<Extra>) -> Vec<String> {
    let mut syms = StrInterner::new();
    let mut m = Module::new("v");
    let i64t = m.types_mut().int(64);
    let i32t = m.types_mut().int(32);
    let ptr = m.types_mut().ptr();
    let tys = Tys { i64t, i32t };
    let sig = m.types_mut().func(vec![i64t, ptr], i64t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    let rty = ty.map(|t| if t == "i32" { i32t } else { i64t });
    {
        let mut b = m.build(f);
        let e = b.create_entry_block();
        let vals = [b.param(e, 0), b.param(e, 1), b.const_int(i64t, puremp::Int::from_i64(5))];
        let operands: Vec<_> = ops.iter().map(|&i| vals[i]).collect();
        let r = b.append_inst(InstKind::InlineAsm(Box::new(make(tys))), operands, Flags::NONE, rty);
        if let Some(extra) = extra {
            extra(&mut b, r, tys);
        }
        let z = b.const_int(i64t, puremp::Int::ZERO);
        b.ret(Some(z));
    }
    match crate::verify::verify_module(&m) {
        Ok(()) => Vec::new(),
        Err(d) => d.into_iter().map(|d| d.message).collect(),
    }
}

/// An asm with `outputs` (constraint, `Some("i64"/"i32")` type or `None`)
/// and input constraints `inputs`.
fn asm_of(t: Tys, outputs: &[(&str, Option<&str>)], inputs: &[&str]) -> InlineAsm {
    InlineAsm {
        template: "nop".into(),
        outputs: outputs
            .iter()
            .map(|&(c, ty)| AsmOutput {
                constraint: c.into(),
                name: None,
                ty: ty.map(|s| if s == "i32" { t.i32t } else { t.i64t }),
            })
            .collect(),
        inputs: inputs.iter().map(|&c| AsmInput { constraint: c.into(), name: None }).collect(),
        clobbers: vec![],
        volatile: true,
    }
}

#[test]
fn verifier_rejects_malformed_asm() {
    type Case = (&'static [(&'static str, Option<&'static str>)], &'static [&'static str], &'static [usize], Option<&'static str>, &'static str);
    let cases: &[Case] = &[
        (&[("r", Some("i64"))], &[], &[], Some("i64"), "must start with `=` or `+`"),
        (&[("=r", None)], &[], &[], None, "needs a type"),
        (&[("=m", Some("i64"))], &[], &[1], None, "produces no value, but has a type"),
        (&[], &["=r"], &[0], None, "may not start with `=`"),
        (&[], &["i"], &[0], None, "must be a constant"),
        (&[], &["m"], &[0], None, "takes a pointer"),
        (&[("=m", None)], &["0"], &[1, 0], None, "must match a register output"),
        (&[], &["3"], &[0], None, "must match a register output"),
        (&[], &["r"], &[], None, "takes 1 operand(s)"),
        (&[("=r", Some("i64"))], &[], &[], Some("i32"), "first register output"),
        (&[("=r", Some("i64"))], &[], &[], None, "first register output"),
        (&[("+r", Some("i32"))], &[], &[0], Some("i32"), "incoming value is i64"),
    ];
    for &(outs, ins, ops, ty, want) in cases {
        let diags = verify_asm(|t| asm_of(t, outs, ins), ops, ty, None);
        assert!(diags.iter().any(|d| d.contains(want)), "{outs:?} {ins:?}: wanted `{want}` in {diags:?}");
    }
    // Duplicate names.
    let diags = verify_asm(
        |t| {
            let mut a = asm_of(t, &[("=r", Some("i64"))], &["r"]);
            a.outputs[0].name = Some("x".into());
            a.inputs[0].name = Some("x".into());
            a
        },
        &[0],
        Some("i64"),
        None,
    );
    assert!(diags.iter().any(|d| d.contains("used twice")), "{diags:?}");
    // `asm_output`: a register output of its asm, at its type.
    let two = |t: Tys| asm_of(t, &[("=a", Some("i64")), ("=d", Some("i32"))], &[]);
    let wrong_type: Extra = Box::new(|b, r, t| {
        b.asm_output(r.unwrap(), 1, t.i64t);
    });
    let diags = verify_asm(two, &[], Some("i64"), Some(wrong_type));
    assert!(diags.iter().any(|d| d.contains("asm_output 1 has type i64 but the output is i32")), "{diags:?}");
    let wrong_index: Extra = Box::new(|b, r, t| {
        b.asm_output(r.unwrap(), 5, t.i64t);
    });
    let diags = verify_asm(two, &[], Some("i64"), Some(wrong_index));
    assert!(diags.iter().any(|d| d.contains("does not name a register output")), "{diags:?}");
    let not_asm: Extra = Box::new(|b, r, t| {
        let s = b.add(r.unwrap(), r.unwrap(), Flags::NONE);
        b.asm_output(s, 1, t.i32t);
    });
    let diags = verify_asm(two, &[], Some("i64"), Some(not_asm));
    assert!(diags.iter().any(|d| d.contains("must be the result of an inline_asm")), "{diags:?}");
    let fine: Extra = Box::new(|b, r, t| {
        b.asm_output(r.unwrap(), 1, t.i32t);
    });
    let ok = verify_asm(two, &[], Some("i64"), Some(fine));
    assert!(ok.is_empty(), "{ok:?}");
    // A well-formed asm with every operand kind verifies.
    let ok = verify_asm(|t| asm_of(t, &[("=r", Some("i64"))], &["0", "i", "m"]), &[0, 2, 1], Some("i64"), None);
    assert!(ok.is_empty(), "{ok:?}");
}

#[test]
fn the_parser_checks_operands_against_constraints() {
    let mut syms = StrInterner::new();
    for (body, want) in [
        (r#"%3 = inline_asm "x" outs("=r" i64 (%0)) : i64"#, "do not match its constraints"),
        (r#"inline_asm "x" ins("r") : void"#, "needs an operand"),
        (r#"%3 = inline_asm "x" outs("=r" i64) : i32"#, "first register output"),
    ] {
        let src = format!("module \"p\"\nfunc @f(i64, i64) -> i64 {{\nentry ^0(%0: i64, %1: i64):\n  {body}\n  ret %0\n}}\n");
        let err = crate::ir::text::parse_module(&src, FileId::new(0), &mut syms).expect_err(body);
        assert!(err.iter().any(|d| d.message.contains(want)), "{body}: {err:?}");
    }
}

/// `@f`, with a pure unused asm, a volatile unused one, a pure one in a loop
/// whose operands are loop-invariant, and a pure one on constants.
const PASSES_LF: &str = r#"module "passes"
func @f(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %dead = inline_asm "imulq %1, %0" outs("=r" i64) ins("r" (%a)) : i64
  %kept = inline_asm volatile "nop" outs("=r" i64) : i64
  %c = inline_asm "movq %1, %0" outs("=r" i64) ins("i" (i64 4)) : i64
  br ^1(i64 0, %c)
^1(%i: i64, %acc: i64):
  %done = icmp sge %i, %b : i1
  cond_br %done, ^3, ^2
^2:
  %inv = inline_asm "leaq 1(%1), %0" outs("=r" i64) ins("r" (%a)) : i64
  %acc2 = add %acc, %inv : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %acc2)
^3:
  ret %acc
}

func @g(i64) -> i64 {
entry ^0(%x: i64):
  %y = call @f(%x, %x) : i64
  ret %y
}
"#;

#[test]
fn passes_treat_asm_conservatively() {
    use crate::transform::pipeline::{OptLevel, optimize, pass_by_name, run_passes};
    let mut syms = StrInterner::new();
    let base = parse(PASSES_LF, &mut syms);
    crate::verify::verify_module(&base).unwrap();
    let f = FuncId::from_index(0);
    assert_eq!(count(base.function(f), is_asm), 4);

    // DCE drops the unused pure asm and keeps the volatile one.
    let mut m = base.clone();
    run_passes(&mut m, vec![pass_by_name("dce").unwrap()]);
    crate::verify::verify_module(&m).unwrap();
    let text = crate::ir::text::print_module(&m, &syms);
    assert!(!text.contains("imulq") && text.contains("\"nop\""), "{text}");

    // SCCP folds nothing out of an asm, even on constant inputs.
    let mut m = base.clone();
    run_passes(&mut m, vec![pass_by_name("sccp").unwrap()]);
    crate::verify::verify_module(&m).unwrap();
    assert!(crate::ir::text::print_module(&m, &syms).contains("movq %1, %0"));

    // LICM leaves the loop's asm in the loop.
    let mut m = base.clone();
    run_passes(&mut m, vec![pass_by_name("licm").unwrap()]);
    crate::verify::verify_module(&m).unwrap();
    let func = m.function(f);
    let in_loop = func.blocks().any(|(_, b)| {
        b.insts().iter().any(|&i| matches!(&func.inst(i).kind, InstKind::InlineAsm(a) if a.template.contains("leaq")))
            && b.insts().iter().any(|&i| matches!(func.inst(i).kind, InstKind::Bin(crate::ir::BinOp::Add)))
            && b.terminator().is_some_and(|t| matches!(func.inst(t).kind, InstKind::Br(_)))
    });
    assert!(in_loop, "the loop asm was moved:\n{}", crate::ir::text::print_module(&m, &syms));

    // Every pipeline (inlining @f into @g included) keeps the module valid
    // and the effectful asm in place.
    for level in [OptLevel::O1, OptLevel::O2, OptLevel::O3] {
        let mut m = base.clone();
        optimize(&mut m, level);
        crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("{level:?}: {e:?}"));
        let text = crate::ir::text::print_module(&m, &syms);
        assert!(text.contains("inline_asm volatile \"nop\""), "{level:?}:\n{text}");
    }
    for name in ["egraph", "simplify_cfg", "mem2reg", "inline"] {
        let mut m = base.clone();
        run_passes(&mut m, vec![pass_by_name(name).unwrap()]);
        crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
}

#[test]
fn analyses_know_nothing_about_asm_outputs() {
    use crate::analysis::domains::ConstLattice;
    use crate::analysis::solver::solve;
    let mut syms = StrInterner::new();
    let m = parse(PASSES_LF, &mut syms);
    let f = FuncId::from_index(0);
    let r = solve::<ConstLattice>(m.function(f), m.types(), m.consts());
    let func = m.function(f);
    for (bid, b) in func.blocks() {
        for &i in b.insts() {
            if let (InstKind::InlineAsm(_), Some(v)) = (&func.inst(i).kind, func.inst(i).result())
                && r.is_reachable(bid)
            {
                assert!(!matches!(r.value(v), ConstLattice::Const(_)), "an asm output is not a constant");
            }
        }
    }
}

#[test]
fn refinement_is_unknown_with_asm() {
    let mut syms = StrInterner::new();
    let src = r#"module "r"
func @a(i64) -> i64 {
entry ^0(%x: i64):
  %y = inline_asm "" outs("+r" i64 (%x)) : i64
  ret %y
}
func @b(i64) -> i64 {
entry ^0(%x: i64):
  %y = inline_asm "" outs("+r" i64 (%x)) : i64
  ret %y
}
func @c(i64) -> i64 {
entry ^0(%x: i64):
  inline_asm "nop" : void
  ret %x
}
"#;
    let m = parse(src, &mut syms);
    let tier = crate::verify::refinement::RefinementTier;
    let r = tier.check(&m, FuncId::from_index(0), FuncId::from_index(1));
    assert!(matches!(r, crate::verify::refinement::RefinementResult::Unknown(_)), "{r:?}");
    // Even a pure asm with no output is opaque.
    let r = tier.check(&m, FuncId::from_index(2), FuncId::from_index(2));
    assert!(matches!(r, crate::verify::refinement::RefinementResult::Unknown(_)), "{r:?}");
}

#[test]
fn constant_time_rejects_secret_asm_operands() {
    use crate::verify::constant_time::{CtPolicy, CtRole, ct_violations};
    let mut syms = StrInterner::new();
    let src = r#"module "ct"
func @f(secret i64, i64) -> secret i64 {
entry ^0(%s: i64, %p: i64):
  %a = inline_asm "addq %1, %0" outs("=r" i64) ins("r" (%p), "r" (%s)) : i64
  %b = inline_asm "" outs("+r" i64 (%s)) : i64
  %c = add %a, %b : i64
  ret %c
}
"#;
    let m = parse(src, &mut syms);
    let v = ct_violations(&m, FuncId::from_index(0), CtPolicy::DEFAULT);
    assert_eq!(v.len(), 1, "{v:?}");
    assert_eq!(v[0].role, CtRole::AsmOperand);
    // The outputs carry the operands' taint, so branching on one is caught.
    let src2 = r#"module "ct2"
func @f(secret i64) -> i64 {
entry ^0(%s: i64):
  %b = inline_asm "" outs("+r" i64 (%s)) : i64
  %z = icmp eq %b, i64 0 : i1
  cond_br %z, ^1, ^2
^1:
  ret i64 1
^2:
  ret i64 2
}
"#;
    let m2 = parse(src2, &mut syms);
    let v = ct_violations(&m2, FuncId::from_index(0), CtPolicy::DEFAULT);
    assert!(v.iter().any(|x| x.role == CtRole::BranchCondition), "{v:?}");
    assert!(crate::verify::verify_module(&m2).is_err());
}

#[test]
fn merging_modules_remaps_output_types() {
    let mut syms = StrInterner::new();
    let a = parse("module \"a\"\nfunc @pad(f32, <4 x i8>) -> void {\nentry ^0(%x: f32, %y: <4 x i8>):\n  ret\n}\n", &mut syms);
    let b = parse(ASM_LF, &mut syms);
    let text_b = crate::ir::text::print_module(&b, &syms);
    let merged = crate::ir::merge_modules([a, b], "m").expect("merge");
    crate::verify::verify_module(&merged).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let text = crate::ir::text::print_module(&merged, &syms);
    let body = |t: &str| t[t.find("func @f").unwrap()..].to_owned();
    assert_eq!(body(&text), body(&text_b));
}

#[test]
fn other_targets_reject_inline_asm_cleanly() {
    use crate::codegen::CodegenOptions;
    use crate::target::{TargetArch, compile_module_for};
    let mut syms = StrInterner::new();
    let m = parse(
        "module \"o\"\nfunc @f(i64) -> i64 {\nentry ^0(%x: i64):\n  %y = inline_asm \"nop\" outs(\"+r\" i64 (%x)) : i64\n  ret %y\n}\n",
        &mut syms,
    );
    for arch in [TargetArch::AArch64, TargetArch::Riscv64, TargetArch::Thumb, TargetArch::Avr, TargetArch::Wasm32] {
        let err = compile_module_for(arch, &m, &syms, &CodegenOptions::default()).err().unwrap_or_else(|| panic!("{arch} accepted asm"));
        let msg = err.to_string();
        assert!(msg.contains("inline asm is not supported on this target") && msg.contains("`f`"), "{arch}: {msg}");
    }
    assert!(compile_module_for(TargetArch::X86_64, &m, &syms, &CodegenOptions::default()).is_ok());
    // The x86-64 check reports a bad statement with its function's name.
    let bad = parse(
        "module \"o\"\nfunc @h(i64) -> i64 {\nentry ^0(%x: i64):\n  %y = inline_asm \"frobnicate %0\" outs(\"+r\" i64 (%x)) : i64\n  ret %y\n}\n",
        &mut syms,
    );
    let err = crate::target::x86_64::check_inline_asm(&bad, &syms).unwrap_err();
    assert!(err.contains("function `h`: inline asm:"), "{err}");
    for (body, want) in [
        (r#"outs("=a" i64, "=a" i64)"#, "two outputs need register %rax"),
        (r#"outs("=r" i64) ins("A" (%x))"#, "`A`"),
        (r#"outs("=r" i64) clobbers("rbp")"#, "frame pointer"),
        (r#"outs("=r" i64) clobbers("bogus")"#, "unknown register `bogus`"),
        (r#"outs("=a" i64) clobbers("eax")"#, "both an operand and clobbered"),
        (r#"outs("=&a" i64) ins("a" (%x))"#, "early-clobber output 0 and an input"),
        (r#"outs("=r" i64) ins("N" (i64 300))"#, "out of range"),
        (r#"outs("=r" i64) ins("I" (%x))"#, "needs a constant operand"),
        (r#"outs("=r" i64) ins("x" (i128 5))"#, "128-bit integer operand"),
        (r#"outs("=r" i64) ins("r" (i128 5))"#, "does not fit a general register"),
        (r#"outs("=r" i64) ins("{r10}" (%x), "{r10}" (i64 3))"#, "two inputs need register %r10"),
    ] {
        let src = format!("module \"o\"\nfunc @h(i64) -> i64 {{\nentry ^0(%x: i64):\n  %y = inline_asm \"nop\" {body} : i64\n  ret %y\n}}\n");
        let mut s = StrInterner::new();
        let m = parse(&src, &mut s);
        let err = crate::target::x86_64::check_inline_asm(&m, &s).unwrap_err();
        assert!(err.contains(want), "{body}: {err}");
    }
}
