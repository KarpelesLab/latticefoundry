//! The optimization pipeline on **vector** code (`docs/ir-design.md` §6c):
//! every level (mem2reg, SCCP, LICM, the e-graph, inlining, CFG
//! simplification, DCE) must keep a vector program's meaning. Each program is
//! run by the reference executor (`crate::ir::refexec`) before and after
//! optimization on several inputs, and the results must agree.

use crate::ir::refexec::run_named;
use crate::ir::semantics::SemValue;
use crate::ir::text::parse_module;
use crate::ir::{InstKind, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize};

use puremp::Int;

const LEVELS: [OptLevel; 4] = [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3];

/// A loop over vectors with a call, a loop-invariant splat, a dead vector op,
/// a vector alloca (mem2reg), shuffles, compares/selects and a reduction.
const PROGRAM: &str = r#"
module "vp"
global @buf : <4 x i32> = <4 x i32> (i32 5, i32 -6, i32 7, i32 -8)

func @helper(<4 x i32>, <4 x i32>) -> <4 x i32> {
entry ^0(%a: <4 x i32>, %b: <4 x i32>):
  %m = mul %a, %b : <4 x i32>
  %c = icmp sgt %m, <4 x i32> (i32 0, i32 0, i32 0, i32 0) : <4 x i1>
  %r = select %c, %m, %a : <4 x i32>
  ret %r
}

func @main(i32) -> i32 {
entry ^0(%n: i32):
  %slot = alloca <4 x i32> : ptr
  %init = load @buf align 16 : <4 x i32>
  store %init, %slot align 16 : <4 x i32>
  br ^1(i32 0, <4 x i32> (i32 1, i32 2, i32 3, i32 4))
^1(%i: i32, %acc: <4 x i32>):
  %k = splat %n : <4 x i32>
  %kk = add %k, <4 x i32> (i32 1, i32 -1, i32 3, i32 0) : <4 x i32>
  %v = load %slot align 16 : <4 x i32>
  %h = call @helper(%acc, %kk) : <4 x i32>
  %s = add %h, %v : <4 x i32>
  %sh = shufflevector %s, %acc, [1, 2, 3, 4] : <4 x i32>
  %dead = mul %sh, %sh : <4 x i32>
  %e = extractelement %sh, 0 : i32
  %w = insertelement %sh, %i, 3 : <4 x i32>
  %i2 = add %i, i32 1 : i32
  %cmp = icmp slt %i2, i32 5 : i1
  cond_br %cmp, ^1(%i2, %w), ^2(%w, %e)
^2(%fin: <4 x i32>, %e2: i32):
  %r = reduce add %fin : i32
  %bc = bitcast %fin : <2 x i64>
  %hi = extractelement %bc, 1 : i64
  %t = trunc %hi : i32
  %x = xor %r, %e2 : i32
  %y = add %x, %t : i32
  ret %y
}

func @fmain(f64) -> f64 {
entry ^0(%x: f64):
  %v = splat %x : <2 x f64>
  %w = fmul %v, <2 x f64> (f64 0x4000000000000000, f64 0xbff0000000000000) : <2 x f64>
  %n = fneg %w : <2 x f64>
  %c = fcmp olt %n, %w : <2 x i1>
  %s = select %c, %n, %w : <2 x f64>
  %d = fdiv %s, <2 x f64> (f64 0x4008000000000000, f64 0x3ff0000000000000) : <2 x f64>
  %r = reduce fadd %d : f64
  ret %r
}
"#;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    (m, syms)
}

fn count(m: &Module, pred: impl Fn(&InstKind) -> bool) -> usize {
    m.functions()
        .map(|f| f.blocks().map(|(_, b)| b.insts().iter().filter(|&&i| pred(&f.inst(i).kind)).count()).sum::<usize>())
        .sum()
}

#[test]
fn every_level_preserves_vector_program_results() {
    let (orig, syms) = parse(PROGRAM);
    let inputs: Vec<i64> = vec![0, 1, -3, 1000, i64::from(i32::MAX)];
    let fin: Vec<f64> = vec![0.0, 1.5, -2.25, 1e300];
    for level in LEVELS {
        let (mut m, s2) = parse(PROGRAM);
        optimize(&mut m, level);
        crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify after {level:?}: {e:?}"));
        for &n in &inputs {
            let arg = [SemValue::int(32, Int::from_i64(n))];
            let want = run_named(&orig, &syms, "main", &arg).expect("source runs");
            let got = run_named(&m, &s2, "main", &arg).expect("optimized runs");
            let (want, got) = (want.expect("a result"), got.expect("a result"));
            assert!(got.refines(&want), "{level:?} n={n}: {got:?} vs {want:?}");
        }
        for &x in &fin {
            let arg = [SemValue::Float(crate::ir::FloatBits::F64(x.to_bits()))];
            let want = run_named(&orig, &syms, "fmain", &arg).expect("source runs").expect("result");
            let got = run_named(&m, &s2, "fmain", &arg).expect("optimized runs").expect("result");
            assert!(got.refines(&want), "{level:?} x={x}: {got:?} vs {want:?}");
        }
        if level >= OptLevel::O2 {
            // The vector-returning helper was inlined into its caller, and the
            // dead vector multiply is gone.
            let main = m.functions().position(|f| s2.resolve(f.name) == "main").unwrap();
            let mf = m.function(crate::ir::FuncId::from_index(main));
            assert!(
                !mf.blocks().any(|(_, b)| b.insts().iter().any(|&i| matches!(mf.inst(i).kind, InstKind::Call))),
                "{level:?} should inline @helper"
            );
        }
    }
    // DCE removes the unused `%dead` product at O1 and above.
    let (mut m, _) = parse(PROGRAM);
    let before = count(&m, |k| matches!(k, InstKind::Bin(crate::ir::BinOp::Mul)));
    optimize(&mut m, OptLevel::O1);
    let after = count(&m, |k| matches!(k, InstKind::Bin(crate::ir::BinOp::Mul)));
    assert!(after < before, "DCE should drop the dead vector mul ({before} -> {after})");
}

#[test]
fn sccp_folds_through_poison_lanes_soundly() {
    // Extracting a poison lane folds to poison; extracting a defined lane of an
    // insert is left alone (vector constants are not folded) but stays correct.
    let src = r#"
module "p"
func @f(i32) -> i32 {
entry ^0(%s: i32):
  %v = insertelement <4 x i32> poison, %s, 1 : <4 x i32>
  %a = extractelement %v, 1 : i32
  %b = extractelement <4 x i32> poison, 2 : i32
  %f = freeze %b : i32
  %r = add %a, %f : i32
  ret %r
}
"#;
    let (orig, syms) = parse(src);
    for level in LEVELS {
        let (mut m, s2) = parse(src);
        optimize(&mut m, level);
        crate::verify::verify_module(&m).unwrap();
        for n in [0i64, 7, -1] {
            let arg = [SemValue::int(32, Int::from_i64(n))];
            let want = run_named(&orig, &syms, "f", &arg).unwrap().unwrap();
            let got = run_named(&m, &s2, "f", &arg).unwrap().unwrap();
            assert!(got.refines(&want), "{level:?}: {got:?} vs {want:?}");
        }
    }
}
