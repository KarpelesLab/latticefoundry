//! Tests for the sanitizer pass: each check fires exactly when the reference
//! semantics ([`crate::ir::semantics::eval`]) says the operation is poison or
//! undefined, the output verifies, the static elisions hold, and secrets are
//! left alone.

use super::*;
use crate::ir::refexec::{ExecError, run_named};
use crate::ir::semantics::{EvalOutcome, SemValue, eval};
use crate::ir::text::{parse_module, print_module};
use crate::support::diagnostics::FileId;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    (m, syms)
}

/// Parse, sanitize with `opts`, verify; return the module and the stats.
fn sanitized(src: &str, opts: &SanitizeOptions) -> (Module, StrInterner, SanitizeStats) {
    let (mut m, mut syms) = parse(src);
    let stats = sanitize_module(&mut m, &mut syms, "t.lf", opts).unwrap_or_else(|e| panic!("sanitize: {e}"));
    crate::verify::verify_module(&m)
        .unwrap_or_else(|e| panic!("verify after: {e:?}\n{}", print_module(&m, &syms)));
    (m, syms, stats)
}

/// Every kind, trapping (no runtime: the refexec stops at the `unreachable`).
fn trap_all(kinds: SanitizeKinds) -> SanitizeOptions {
    SanitizeOptions { kinds, trap: kinds, recover: true, runtime: SanitizeRuntime::External }
}

/// Whether a run stopped at a check's trap (the sanitized `unreachable`).
fn fired(r: &Result<Option<SemValue>, ExecError>) -> bool {
    match r {
        Ok(_) => false,
        Err(ExecError::Ub(msg)) if msg == "reached unreachable" => true,
        Err(e) => panic!("the sanitized program misbehaved: {e:?}"),
    }
}

/// Whether the reference semantics make the operation poison or UB.
fn bad_per_semantics(m: &Module, ty: TypeId, kind: &InstKind, flags: Flags, ops: &[SemValue]) -> bool {
    match eval(m.types(), ty, kind, &flags, ops) {
        EvalOutcome::UndefinedBehavior => true,
        EvalOutcome::Value(v) => v == SemValue::Poison,
    }
}

/// The one binary instruction of `@f`.
fn the_inst(m: &Module, syms: &StrInterner) -> (InstKind, Flags, TypeId) {
    let f = m.functions().find(|f| syms.resolve(f.name) == "f").expect("@f");
    let (_, blk) = f.blocks().next().expect("a block");
    let inst = f.inst(blk.insts()[0]);
    (inst.kind.clone(), inst.flags, inst.ty)
}

/// Check `op flags` on `iW` exhaustively (or on `samples`): the sanitized
/// function traps exactly when the semantics say poison/UB.
fn check_binop(op: &str, flags: &str, w: u32, samples: Option<&[i128]>) {
    let t = format!("i{w}");
    let src = format!(
        "module \"t\"\nfunc @f({t}, {t}) -> {t} {{\nentry ^0(%a: {t}, %b: {t}):\n  %r = {op} {flags} %a, %b : {t}\n  ret %r\n}}\n"
    );
    let (orig, osyms) = parse(&src);
    let (kind, fl, ty) = the_inst(&orig, &osyms);
    let (m, syms, _) = sanitized(&src, &trap_all(SanitizeKinds::ALL));
    let all: Vec<i128> = match samples {
        Some(s) => s.to_vec(),
        None => (0..(1i128 << w)).collect(),
    };
    let mut fires = 0;
    for &a in &all {
        for &b in &all {
            let args = [SemValue::int(w, Int::from_i128(a)), SemValue::int(w, Int::from_i128(b))];
            let want = bad_per_semantics(&orig, ty, &kind, fl, &args);
            let got = fired(&run_named(&m, &syms, "f", &args));
            assert_eq!(got, want, "{op} {flags} i{w} on ({a}, {b}): check fired = {got}, semantics bad = {want}");
            fires += usize::from(got);
        }
    }
    // Every flagged or faulting op has some failing input.
    if !flags.is_empty() || matches!(op, "udiv" | "sdiv" | "urem" | "srem" | "shl" | "lshr" | "ashr") {
        assert!(fires > 0, "{op} {flags}: no input failed");
    }
}

#[test]
fn integer_checks_match_the_semantics_exhaustively_on_i8() {
    for (op, flags) in [
        ("add", "nsw"),
        ("add", "nuw"),
        ("sub", "nsw"),
        ("sub", "nuw"),
        ("mul", "nsw"),
        ("mul", "nuw"),
        ("mul", "nsw nuw"),
        ("shl", ""),
        ("shl", "nsw"),
        ("shl", "nuw"),
        ("lshr", "exact"),
        ("ashr", "exact"),
        ("udiv", "exact"),
        ("sdiv", "exact"),
        ("urem", ""),
        ("srem", ""),
    ] {
        check_binop(op, flags, 8, None);
    }
}

#[test]
fn integer_checks_match_the_semantics_on_other_widths() {
    // i1 and i3 exhaustively; i32, i64 (division-based mul check) and i128
    // (add/sub/shift/div) on boundary samples.
    for w in [1, 3] {
        for (op, flags) in [("add", "nsw"), ("sub", "nuw"), ("mul", "nsw"), ("shl", "nsw"), ("sdiv", "exact"), ("srem", "")] {
            check_binop(op, flags, w, None);
        }
    }
    for w in [32u32, 64] {
        let max = (1i128 << (w - 1)) - 1;
        let min = -(1i128 << (w - 1));
        let s = [0, 1, 2, 3, -1, -2, 7, max, max - 1, min, min + 1, max / 2 + 1, min / 2, 1 << (w / 2), (1 << (w / 2)) - 1, w as i128, w as i128 - 1];
        for (op, flags) in [("add", "nsw"), ("add", "nuw"), ("sub", "nsw"), ("mul", "nsw"), ("mul", "nuw"), ("shl", "nsw"), ("shl", "nuw"), ("ashr", "exact"), ("sdiv", "exact"), ("udiv", ""), ("srem", "")] {
            check_binop(op, flags, w, Some(&s));
        }
    }
    let s = [0, 1, -1, 127, 128, i128::MAX, i128::MIN, i128::MIN + 1, 1 << 64, -(1 << 64)];
    for (op, flags) in [("add", "nsw"), ("sub", "nuw"), ("shl", ""), ("sdiv", ""), ("udiv", "exact")] {
        check_binop(op, flags, 128, Some(&s));
    }
}

/// Check `fptosi`/`fptoui` from `from` to `iW` on the given float bit patterns.
fn check_fp_to_int(cast: &str, from: &str, w: u32, patterns: &[FloatBits]) {
    let src = format!(
        "module \"t\"\nfunc @f({from}) -> i{w} {{\nentry ^0(%x: {from}):\n  %r = {cast} %x : i{w}\n  ret %r\n}}\n"
    );
    let (orig, osyms) = parse(&src);
    let (kind, fl, ty) = the_inst(&orig, &osyms);
    let (m, syms, stats) = sanitized(&src, &trap_all(SanitizeKinds::ALL));
    assert_eq!(stats.count(UbKind::FloatCast), 1);
    for &p in patterns {
        let args = [SemValue::Float(p)];
        let want = bad_per_semantics(&orig, ty, &kind, fl, &args);
        let got = fired(&run_named(&m, &syms, "f", &args));
        assert_eq!(got, want, "{cast} {from} -> i{w} on {p:?}");
    }
}

#[test]
fn float_cast_checks_match_the_semantics() {
    // Every binary16 pattern to the widths around its range, and every
    // seventh to more widths.
    let all16: Vec<FloatBits> = (0..=u16::MAX).map(FloatBits::F16).collect();
    let some16: Vec<FloatBits> = (0..=u16::MAX).step_by(7).map(FloatBits::F16).collect();
    for w in [8, 16, 17] {
        check_fp_to_int("fptosi", "f16", w, &all16);
        check_fp_to_int("fptoui", "f16", w, &all16);
    }
    for w in [1, 11, 12, 32] {
        check_fp_to_int("fptosi", "f16", w, &some16);
        check_fp_to_int("fptoui", "f16", w, &some16);
    }
    // f32 and f64 on the values around each width's bounds.
    for w in [1u32, 8, 16, 23, 24, 25, 31, 32, 33, 53, 54, 63, 64, 65, 127, 128] {
        let mut f64s = vec![0.0, -0.0, 0.5, -0.5, 0.99, -0.99, 1.0, -1.0, -1.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX, f64::MIN];
        for k in [w.saturating_sub(1), w, w + 1] {
            let p = 2f64.powi(k as i32);
            f64s.extend([p, -p, p - 1.0, -p + 1.0, p + 1.0, -p - 1.0, p - 0.5, -p - 0.5, -p + 0.5, p * (1.0 - f64::EPSILON), -p * (1.0 + f64::EPSILON)]);
        }
        let p32: Vec<FloatBits> = f64s
            .iter()
            .flat_map(|&x| {
                let f = x as f32;
                [FloatBits::F32(f.to_bits()), FloatBits::F32(f.to_bits().wrapping_add(1)), FloatBits::F32(f.to_bits().wrapping_sub(1))]
            })
            .collect();
        let p64: Vec<FloatBits> = f64s
            .iter()
            .flat_map(|&x| [FloatBits::F64(x.to_bits()), FloatBits::F64(x.to_bits().wrapping_add(1)), FloatBits::F64(x.to_bits().wrapping_sub(1))])
            .collect();
        for cast in ["fptosi", "fptoui"] {
            check_fp_to_int(cast, "f32", w, &p32);
            check_fp_to_int(cast, "f64", w, &p64);
        }
    }
}

/// Run `@f(i64)` of a sanitized module, reporting whether a check fired.
fn fires_on(m: &Module, syms: &StrInterner, x: i64) -> bool {
    fired(&run_named(m, syms, "f", &[SemValue::int(64, Int::from_i64(x))]))
}

#[test]
fn bounds_checks_follow_the_object() {
    // A 16-byte global and a 16-byte stack slot: `ptr_add inbounds` may reach
    // one past the end; an i32 access must end inside.
    for base in ["@g", "%s"] {
        let src = format!(
            "module \"t\"\nglobal @g : [16 x i8] = [16 x i8] poison\n\
             func @f(i64) -> i64 {{\nentry ^0(%i: i64):\n  %s = alloca [16 x i8] : ptr\n  \
             %p = ptr_add inbounds {base}, %i : ptr\n  ret i64 0\n}}\n"
        );
        let (m, syms, stats) = sanitized(&src, &trap_all(SanitizeKinds::of(&[UbKind::PointerBounds])));
        assert_eq!(stats.count(UbKind::PointerBounds), 1, "{base}");
        for (i, bad) in [(0, false), (15, false), (16, false), (17, true), (-1, true), (i64::MIN, true)] {
            assert_eq!(fires_on(&m, &syms, i), bad, "ptr_add {base} + {i}");
        }
        let src = format!(
            "module \"t\"\nglobal @g : [16 x i8] = [16 x i8] poison\n\
             func @f(i64) -> i64 {{\nentry ^0(%i: i64):\n  %s = alloca [16 x i8] : ptr\n  \
             %p = ptr_add {base}, %i : ptr\n  %q = ptr_add %p, i64 2 : ptr\n  store i32 7, %q align 1 : i32\n  ret i64 0\n}}\n"
        );
        let (m, syms, stats) = sanitized(&src, &trap_all(SanitizeKinds::of(&[UbKind::ObjectBounds])));
        assert_eq!(stats.count(UbKind::ObjectBounds), 1, "{base}");
        for (i, bad) in [(-2, false), (10, false), (11, true), (-3, true), (100, true)] {
            assert_eq!(fires_on(&m, &syms, i), bad, "store at {base} + {i} + 2");
        }
    }
    // A dynamic stack allocation is measured by its run-time size.
    let src = "module \"t\"\nfunc @f(i64) -> i64 {\nentry ^0(%n: i64):\n  %s = dyn_alloca %n align 8 : ptr\n  \
               %p = ptr_add inbounds %s, i64 8 : ptr\n  %v = load %p align 8 : i64\n  ret %v\n}\n";
    // (The reference executor does not model `dyn_alloca`; the end-to-end
    // tests run these checks.)
    let (_, _, stats) = sanitized(src, &trap_all(SanitizeKinds::of(&[UbKind::PointerBounds, UbKind::ObjectBounds])));
    assert_eq!((stats.count(UbKind::PointerBounds), stats.count(UbKind::ObjectBounds)), (1, 1));
}

#[test]
fn null_and_alignment_checks() {
    let src = "module \"t\"\nglobal @g : [16 x i8] = [16 x i8] poison\n\
               func @f(i64) -> i64 {\nentry ^0(%i: i64):\n  %z = icmp eq %i, i64 99 : i1\n  \
               %q = ptr_add @g, %i : ptr\n  %p = select %z, ptr null, %q : ptr\n  \
               store i32 1, %p align 4 : i32\n  ret i64 0\n}\n";
    let opts = trap_all(SanitizeKinds::of(&[UbKind::NullPointer, UbKind::Misaligned]));
    let (m, syms, stats) = sanitized(src, &opts);
    assert_eq!((stats.count(UbKind::NullPointer), stats.count(UbKind::Misaligned)), (1, 1));
    for (i, bad) in [(0, false), (4, false), (8, false), (1, true), (2, true), (7, true), (99, true)] {
        assert_eq!(fires_on(&m, &syms, i), bad, "store at @g + {i}");
    }
}

#[test]
fn unreachable_is_reported() {
    let src = "module \"t\"\nfunc @f(i64) -> i64 {\nentry ^0(%i: i64):\n  %z = icmp eq %i, i64 3 : i1\n  \
               cond_br %z, ^1, ^2\n^1:\n  unreachable\n^2:\n  ret %i\n}\n";
    let (m, syms, stats) = sanitized(src, &trap_all(SanitizeKinds::of(&[UbKind::Unreachable])));
    assert_eq!(stats.count(UbKind::Unreachable), 1);
    assert!(!fires_on(&m, &syms, 2));
    assert!(fires_on(&m, &syms, 3));
    // The trap records the kind before trapping.
    let text = print_module(&m, &syms);
    assert!(text.contains("store volatile i32 13, @__lf_ub_trap_kind"), "{text}");
}

#[test]
fn reporting_mode_calls_the_handler_and_drops_covered_flags() {
    let src = "module \"t\"\nfunc @f(i32, i32) -> i32 {\nentry ^0(%a: i32, %b: i32):\n  \
               %r = add nsw %a, %b : i32\n  %s = shl nuw %r, %b : i32\n  %d = sdiv exact %s, %b : i32\n  ret %d\n}\n";
    let opts = SanitizeOptions { runtime: SanitizeRuntime::External, ..SanitizeOptions::default() };
    let (m, syms, stats) = sanitized(src, &opts);
    let text = print_module(&m, &syms);
    assert!(text.contains("func @__lf_ub_report(i32, ptr, i64, i64) -> void\n"), "declared only:\n{text}");
    assert!(text.contains("call @__lf_ub_report("), "{text}");
    assert!(!text.contains("add nsw") && !text.contains("shl nuw") && !text.contains("sdiv exact"), "{text}");
    assert_eq!(stats.count(UbKind::SignedOverflow), 1);
    assert_eq!(stats.count(UbKind::ShiftExponent), 1);
    assert_eq!(stats.count(UbKind::ShiftBase), 1);
    assert_eq!(stats.count(UbKind::DivByZero), 1);
    assert_eq!(stats.count(UbKind::DivOverflow), 1);
    assert_eq!(stats.count(UbKind::Inexact), 1);
    // One location record per (function, line, kind), naming the file.
    assert!(text.contains("@__lf_ub_loc"), "{text}");
    assert!(text.contains("@__lf_ub_file0"), "{text}");
}

#[test]
fn statically_safe_operations_get_no_check() {
    let src = "module \"t\"\nglobal @g : [4 x i32] = [4 x i32] poison\n\
               func @f(i32, ptr) -> i32 {\nentry ^0(%a: i32, %p: ptr):\n  \
               %s = alloca i32 : ptr\n  store %a, %s align 4 : i32\n  %v = load %s align 4 : i32\n  \
               %e = ptr_add inbounds @g, i64 12 : ptr\n  %w = load %e align 4 : i32\n  \
               %x = shl %v, i32 3 : i32\n  %y = sdiv %x, i32 7 : i32\n  %z = udiv %y, i32 -1 : i32\n  \
               %k = add nsw i32 1, i32 2 : i32\n  %r = add %z, %w : i32\n  %t = add %r, %k : i32\n  ret %t\n}\n";
    let (_, _, stats) = sanitized(src, &trap_all(SanitizeKinds::ALL));
    assert_eq!(stats.total(), 0, "{stats:?}");
}

#[test]
fn the_output_composes_with_the_optimizer() {
    let src = "module \"t\"\nfunc @f(i64, i64) -> i64 {\nentry ^0(%a: i64, %b: i64):\n  \
               %s = alloca [8 x i64] : ptr\n  %p = ptr_add inbounds %s, %b : ptr\n  store %a, %p align 8 : i64\n  \
               %v = load %p align 8 : i64\n  %m = mul nsw %v, %a : i64\n  %d = srem %m, %b : i64\n  ret %d\n}\n";
    for opt in [crate::transform::OptLevel::O1, crate::transform::OptLevel::O2, crate::transform::OptLevel::O3] {
        let (mut m, syms, stats) = sanitized(src, &trap_all(SanitizeKinds::ALL));
        assert!(stats.total() >= 5, "{stats:?}");
        crate::transform::optimize(&mut m, opt);
        crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("{opt:?}: {e:?}\n{}", print_module(&m, &syms)));
        // The checks survive optimization (the trap's volatile store is
        // observable) and still fire.
        let call = |a: i64, b: i64| {
            run_named(&m, &syms, "f", &[SemValue::int(64, Int::from_i64(a)), SemValue::int(64, Int::from_i64(b))])
        };
        assert!(!fired(&call(3, 8)), "{opt:?}: a correct call");
        assert!(fired(&call(3, 0)), "{opt:?}: srem by zero");
        assert!(fired(&call(1 << 40, 16)), "{opt:?}: mul overflow");
        assert!(fired(&call(3, 64)), "{opt:?}: out of bounds");
    }
}

#[test]
fn secret_operands_are_not_checked_and_ct_verification_still_passes() {
    let src = "module \"t\"\n\
               func @ct(secret i32, i32, ptr) -> secret i32 {\nentry ^0(%k: i32, %n: i32, %p: ptr):\n  \
               %a = add nsw %k, i32 1 : i32\n  %b = add nsw %n, i32 1 : i32\n  %c = shl %k, %n : i32\n  \
               store secret %a, %p align 4 : i32\n  %d = udiv %n, %b : i32\n  %e = add %c, %d : i32\n  ret %e\n}\n";
    let (m0, _) = parse(src);
    crate::verify::constant_time::verify_module_ct(&m0).expect("constant-time before");
    let (m, syms, stats) = sanitized(src, &trap_all(SanitizeKinds::ALL));
    // `%a = add nsw %k` and the shift (its amount is public, the value secret)
    // read a secret; `%b`, the division and the store's address are public.
    assert_eq!(stats.skipped_secret, 2, "{stats:?}");
    assert_eq!(stats.count(UbKind::SignedOverflow), 1);
    assert_eq!(stats.count(UbKind::DivByZero), 1);
    assert_eq!(stats.count(UbKind::NullPointer), 1);
    crate::verify::constant_time::verify_module_ct(&m)
        .unwrap_or_else(|e| panic!("constant-time after: {e:?}\n{}", print_module(&m, &syms)));
    let text = print_module(&m, &syms);
    assert!(text.contains("add nsw %0, i32 1"), "the secret add keeps its flag:\n{text}");
}

#[test]
fn the_runtime_links_and_verifies() {
    for abi in [LinuxSyscalls::X86_64, LinuxSyscalls::Generic] {
        for recover in [true, false] {
            let src = "module \"t\"\nfunc @main() -> i32 {\nentry ^0:\n  %r = add nsw i32 2147483647, i32 1 : i32\n  ret %r\n}\n";
            let opts = SanitizeOptions { runtime: SanitizeRuntime::Linux(abi), recover, ..SanitizeOptions::default() };
            let (m, syms, _) = sanitized(src, &opts);
            let text = print_module(&m, &syms);
            assert!(text.contains("func weak @__lf_ub_report(i32, ptr, i64, i64) -> void {"), "{text}");
            let (write, exit) = if abi == LinuxSyscalls::X86_64 { (1, 231) } else { (64, 94) };
            assert!(text.contains(&format!("syscall i64 {write}, i64 2")), "{text}");
            assert!(text.contains(&format!("syscall i64 {exit}, i64 1")), "{text}");
            // The runtime itself is not instrumented, and a second run adds
            // nothing to it.
            assert!(!text.contains("@__lf_ub_report(i32 1"), "{text}");
        }
    }
}

#[test]
fn kind_names_parse() {
    assert_eq!(SanitizeKinds::parse("undefined"), Ok(SanitizeKinds::ALL));
    let s = SanitizeKinds::parse("shift,null").unwrap();
    assert!(s.contains(UbKind::ShiftExponent) && s.contains(UbKind::ShiftBase) && s.contains(UbKind::NullPointer));
    assert!(!s.contains(UbKind::SignedOverflow));
    let s = SanitizeKinds::parse("signed-integer-overflow").unwrap();
    assert_eq!(s, SanitizeKinds::of(&[UbKind::SignedOverflow, UbKind::DivOverflow]));
    assert!(SanitizeKinds::parse("address").is_err());
    for k in UbKind::ALL {
        assert!(SanitizeKinds::from_name(k.name()).is_some_and(|s| s.contains(k)), "{}", k.name());
        assert_eq!(UbKind::from_code(k.code()), Some(k));
    }
    assert_eq!(UbKind::from_code(0), None);
}
