//! Tests for the wasm32 backend.
//!
//! - **Differential execution under node**: IR programs are compiled to a
//!   self-contained module, run under node (`WebAssembly.instantiate`), and
//!   every result is compared with the [reference interpreter](refinterp),
//!   which evaluates the *original* IR with the reference evaluator. Inputs on
//!   which the IR has undefined behavior are skipped, and a poison reference
//!   result accepts anything. Skipped (with a message) when node is absent.
//! - **Encoding**: section layout, LEB128 forms, relocatable-object structure,
//!   instruction bytes cross-checked with `llvm-mc`, and `wasm-ld` linking our
//!   objects (each skipped when its tool is absent).
//!
//! The structurizer's own tests live in [`super::structure`].

mod behavior;
mod encoding;
mod node;
mod programs;
mod random;
mod refinterp;

use super::{compile, data_layout};
use crate::codegen::CodegenOptions;
use crate::ir::types::{FloatKind, Type, TypeId};
use crate::ir::value::FloatBits;
use crate::ir::{Module, SemValue, text};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

use node::Call;
use refinterp::{Interp, Stop, bits_u64};

use puremp::Int;

/// Parse `src`, give it the wasm32 layout, and verify it.
pub(crate) fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = text::parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|d| panic!("parse: {d:?}\n{src}"));
    m.set_data_layout(data_layout());
    if let Err(d) = crate::verify::verify_module(&m) {
        panic!("verify: {d:?}\n{src}");
    }
    (m, syms)
}

/// Compile `m` to a self-contained module.
pub(crate) fn linked(m: &Module, syms: &StrInterner) -> Vec<u8> {
    let c = compile(m, syms, &CodegenOptions::default()).unwrap_or_else(|e| panic!("{e}"));
    c.object.to_linked(&Default::default()).unwrap_or_else(|e| panic!("{e}"))
}

/// The wasm types (and part count) a value of IR type `ty` travels as.
fn wasm_types(m: &Module, ty: TypeId) -> Vec<&'static str> {
    match m.types().get(ty) {
        Type::Void => vec![],
        Type::Int(b) if *b <= 32 => vec!["i32"],
        Type::Int(b) if *b <= 64 => vec!["i64"],
        Type::Int(b) => vec!["i64"; (*b / 64) as usize],
        Type::Float(FloatKind::F32) => vec!["f32"],
        Type::Float(FloatKind::F64) => vec!["f64"],
        _ => vec!["i32"],
    }
}

/// A reference value of IR type `ty` from raw bits.
fn sem(m: &Module, ty: TypeId, raw: u128) -> SemValue {
    match m.types().get(ty) {
        Type::Int(b) => SemValue::int(*b, Int::from_u128(raw)),
        Type::Float(FloatKind::F32) => SemValue::Float(FloatBits::F32(raw as u32)),
        Type::Float(FloatKind::F64) => SemValue::Float(FloatBits::F64(raw as u64)),
        _ => SemValue::Ptr(Int::from_u128(raw & 0xffff_ffff)),
    }
}

/// Split raw bits into the wasm values carrying them.
fn parts(tys: &[&'static str], raw: u128) -> Vec<u64> {
    if tys.len() > 1 {
        (0..tys.len()).map(|k| if k < 2 { (raw >> (64 * k)) as u64 } else { 0 }).collect()
    } else {
        vec![raw as u64]
    }
}

fn is_nan(t: &str, bits: u64) -> bool {
    match t {
        "f32" => f32::from_bits(bits as u32).is_nan(),
        "f64" => f64::from_bits(bits).is_nan(),
        _ => false,
    }
}

/// How many calls a differential run compared (and skipped as UB).
#[derive(Default, Debug)]
pub(crate) struct Tally {
    pub(crate) compared: usize,
    pub(crate) skipped: usize,
}

/// Compile `src`, run every `(function, raw args)` case under node and under
/// the reference interpreter, and compare. Returns `None` without node.
pub(crate) fn differential(tag: &str, src: &str, cases: &[(&str, Vec<u128>)]) -> Option<Tally> {
    let (m, syms) = parse(src);
    let wasm = linked(&m, &syms);
    differential_module(tag, &m, &syms, &wasm, cases)
}

/// [`differential`], but the wasm comes from the program after the `-O2`
/// pipeline, while the reference still runs the original: a check of the
/// optimizer and the backend together.
pub(crate) fn differential_optimized(tag: &str, src: &str, cases: &[(&str, Vec<u128>)]) -> Option<Tally> {
    let (m, syms) = parse(src);
    let (mut opt, opt_syms) = parse(src);
    crate::transform::pipeline::optimize(&mut opt, crate::transform::pipeline::OptLevel::O2);
    if let Err(d) = crate::verify::verify_module(&opt) {
        panic!("{tag}: the optimized module does not verify: {d:?}");
    }
    let wasm = linked(&opt, &opt_syms);
    differential_module(&format!("{tag}-O2"), &m, &syms, &wasm, cases)
}

/// [`differential`] on an already compiled module.
pub(crate) fn differential_module(
    tag: &str,
    m: &Module,
    syms: &StrInterner,
    wasm: &[u8],
    cases: &[(&str, Vec<u128>)],
) -> Option<Tally> {
    node::node()?;
    let mut calls = Vec::new();
    let mut expected = Vec::new();
    let mut tally = Tally::default();
    // One interpreter for all the calls, as node runs them all on one
    // instance: global state carries over from call to call in both.
    let mut interp = Interp::new(m, syms);
    for (name, raw) in cases {
        let f = m
            .functions()
            .find(|f| syms.resolve(f.name) == *name)
            .unwrap_or_else(|| panic!("no function {name}"));
        let Type::Func(ft) = m.types().get(f.sig).clone() else { unreachable!() };
        assert_eq!(ft.params.len(), raw.len(), "{name}: argument count");
        let mut args = Vec::new();
        let mut sargs = Vec::new();
        for (&p, &r) in ft.params.iter().zip(raw) {
            let tys = wasm_types(m, p);
            let v = sem(m, p, r);
            // The bits the IR value has (masked to its width).
            let bits = match &v {
                SemValue::Int { bits, .. } | SemValue::Ptr(bits) => bits.to_u128().unwrap(),
                other => u128::from(bits_u64(other).unwrap()),
            };
            args.extend(tys.iter().copied().zip(parts(&tys, bits)));
            sargs.push(v);
        }
        let want = match interp.run(name, sargs) {
            Ok(v) => v,
            Err(Stop::Ub(_)) => {
                tally.skipped += 1;
                continue;
            }
            Err(Stop::Budget) => panic!("{name}{raw:?}: the reference ran out of budget"),
        };
        let rets = wasm_types(m, ft.ret);
        calls.push(Call { func: (*name).to_owned(), args, rets: rets.clone() });
        expected.push((format!("{name}{raw:x?}"), rets, want));
    }
    let got = node::run(tag, wasm, &calls)?;
    for ((what, rets, want), got) in expected.into_iter().zip(got) {
        let got = got.unwrap_or_else(|t| panic!("{what}: trapped: {t}"));
        let want = match want {
            None => {
                assert!(got.is_empty(), "{what}: void function returned {got:x?}");
                tally.compared += 1;
                continue;
            }
            Some(SemValue::Poison) => continue,
            Some(v) => v,
        };
        let bits = match &want {
            SemValue::Int { bits, .. } | SemValue::Ptr(bits) => bits.to_u128().unwrap(),
            other => u128::from(bits_u64(other).unwrap()),
        };
        let want_parts = parts(&rets, bits);
        if rets.len() == 1 && is_nan(rets[0], want_parts[0]) {
            assert!(is_nan(rets[0], got[0]), "{what}: want NaN, got {:#x}", got[0]);
        } else {
            assert_eq!(got, want_parts, "{what}: wasm {got:x?} != reference {want_parts:x?}");
        }
        tally.compared += 1;
    }
    eprintln!("differential {tag}: {} calls compared, {} skipped (UB in the IR)", tally.compared, tally.skipped);
    Some(tally)
}

/// Skip message for tests that need node.
pub(crate) fn no_node(test: &str) {
    eprintln!("skipping {test}: node is not installed");
}

#[test]
fn smoke_add() {
    let src = r#"
module "smoke"
func @add(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %r = add %a, %b : i32
  ret %r
}
"#;
    let Some(t) = differential("smoke", src, &[("add", vec![1, 2]), ("add", vec![0xffff_ffff, 5])]) else {
        return no_node("smoke_add");
    };
    assert_eq!(t.compared, 2);
}
