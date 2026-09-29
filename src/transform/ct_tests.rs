//! Constant-time preservation (`docs/ir-design.md` §6d): every pass, every
//! `-O` pipeline and random pass orders must map constant-time code to
//! constant-time code. Each test runs the transforms over secret-heavy
//! functions — hand-written classics and a deterministic random generator —
//! and re-runs the constant-time verifier (inside `verify_module`) after every
//! single pass.

use crate::ir::{FuncId, Module};
use crate::pass::ModulePass;
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, pass_by_name, pipeline_for};
use crate::verify::{CtPolicy, ct_violations, verify_module};

/// Every individual pass `pass_by_name` knows.
pub(crate) const PASSES: [&str; 7] =
    ["mem2reg", "sccp", "simplify_cfg", "dce", "egraph", "licm", "inline"];

/// A conditional swap of two `n`-limb numbers by a secret bit (the core of a
/// Montgomery ladder), a ladder driving it bit by bit over a secret scalar,
/// and a constant-time comparison. Shared with the x86-64 execution test.
pub(crate) const LADDER_LF: &str = r#"
module "ladder"

func @cswap(ptr, ptr, secret i64, i64) -> void {
entry ^0(%a: ptr, %b: ptr, %bit: i64, %n: i64):
  %mask = sub i64 0, %bit : i64
  br ^1(i64 0)
^1(%i: i64):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %off = mul %i, i64 8 : i64
  %pa = ptr_add %a, %off : ptr
  %pb = ptr_add %b, %off : ptr
  %x = load secret %pa align 8 : i64
  %y = load secret %pb align 8 : i64
  %t = xor %x, %y : i64
  %d = and %t, %mask : i64
  %x2 = xor %x, %d : i64
  %y2 = xor %y, %d : i64
  store secret %x2, %pa align 8 : i64
  store secret %y2, %pb align 8 : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2)
^3:
  ret
}

func @ladder(ptr, ptr, secret i64, i64) -> void {
entry ^0(%r0: ptr, %r1: ptr, %k: i64, %bits: i64):
  br ^1(%bits, i64 0)
^1(%j: i64, %prev: i64):
  %more = icmp ne %j, i64 0 : i1
  cond_br %more, ^2, ^3
^2:
  %j1 = sub %j, i64 1 : i64
  %sh = lshr %k, %j1 : i64
  %bit = and %sh, i64 1 : i64
  %swap = xor %bit, %prev : i64
  call @cswap(%r0, %r1, %swap, i64 2) : void
  %p0 = ptr_add %r0, i64 8 : ptr
  %p1 = ptr_add %r1, i64 8 : ptr
  %a0 = load secret %r0 align 8 : i64
  %a1 = load secret %p0 align 8 : i64
  %b0 = load secret %r1 align 8 : i64
  %b1 = load secret %p1 align 8 : i64
  %s0 = add %a0, %b0 : i64
  %d1 = mul %a1, i64 2 : i64
  %d2 = add %d1, i64 0 : i64
  %m0 = mul %b0, %b1 : i64
  %m1 = shl %m0, i64 1 : i64
  store secret %s0, %r1 align 8 : i64
  store secret %d2, %p0 align 8 : i64
  store secret %m1, %p1 align 8 : i64
  br ^1(%j1, %bit)
^3:
  call @cswap(%r0, %r1, %prev, i64 2) : void
  ret
}

func @ct_memcmp(ptr, ptr, i64) -> secret i64 {
entry ^0(%a: ptr, %b: ptr, %n: i64):
  br ^1(i64 0, i8 0)
^1(%i: i64, %acc: i8):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %pa = ptr_add %a, %i : ptr
  %pb = ptr_add %b, %i : ptr
  %x = load secret %pa align 1 : i8
  %y = load secret %pb align 1 : i8
  %d = xor %x, %y : i8
  %acc2 = or %acc, %d : i8
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %acc2)
^3:
  %nz = icmp ne %acc, i8 0 : i1
  %r = zext %nz : i64
  ret %r
}
"#;

/// Selects, masks, local memory (mem2reg food), constants (SCCP / e-graph
/// food), loops (LICM food), a declassified branch and a helper to inline.
const MIXED_LF: &str = r#"
module "mixed"

global secret @key : [4 x i64] = [4 x i64] (i64 11, i64 22, i64 33, i64 44)

func @ct_select(secret i1, secret i64, secret i64) -> secret i64 {
entry ^0(%c: i1, %a: i64, %b: i64):
  %r = select %c, %a, %b : i64
  ret %r
}

func @ct_min(secret i64, secret i64) -> secret i64 {
entry ^0(%a: i64, %b: i64):
  %c = icmp slt %a, %b : i1
  %r = call @ct_select(%c, %a, %b) : i64
  ret %r
}

func @locals(secret i64, i64) -> secret i64 {
entry ^0(%s: i64, %p: i64):
  %x = alloca i64 : ptr
  store %s, %x align 8 : i64
  %c = icmp ult %p, i64 10 : i1
  cond_br %c, ^1, ^2
^1:
  %v = load %x align 8 : i64
  %v2 = mul %v, i64 3 : i64
  store %v2, %x align 8 : i64
  br ^3
^2:
  store %p, %x align 8 : i64
  br ^3
^3:
  %r = load %x align 8 : i64
  %z = xor %r, %r : i64
  %q = add %r, %z : i64
  ret %q
}

func @folds(secret i64, i64, ptr) -> secret i64 {
entry ^0(%s: i64, %p: i64, %t: ptr):
  %a = add %s, i64 0 : i64
  %b = mul %a, i64 8 : i64
  %c = sub %b, %b : i64
  %d = or %c, %s : i64
  %p2 = mul %p, i64 4 : i64
  %p3 = add %p2, i64 0 : i64
  %e = ptr_add %t, %p3 : ptr
  %f = load %e align 8 : i64
  %g = udiv %f, i64 7 : i64
  %one = add i64 1, i64 0 : i64
  %h = icmp eq %one, i64 1 : i1
  cond_br %h, ^1, ^2
^1:
  %u = add %d, %g : i64
  ret %u
^2:
  ret %s
}

func @hoist(secret i64, i64) -> secret i64 {
entry ^0(%s: i64, %n: i64):
  br ^1(i64 0, i64 0)
^1(%i: i64, %acc: i64):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %inv = mul %s, i64 5 : i64
  %invp = mul %n, i64 3 : i64
  %m = and %inv, %invp : i64
  %acc2 = xor %acc, %m : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %acc2)
^3:
  ret %acc
}

func @verdict(ptr, ptr) -> i64 {
entry ^0(%a: ptr, %b: ptr):
  %k = load @key align 8 : i64
  %k2 = ptr_add @key, i64 8 : ptr
  %kk = load %k2 align 8 : i64
  %m = call @ct_min(%k, %kk) : i64
  %e = icmp eq %m, i64 11 : i1
  %ez = zext %e : i64
  %d = declassify %ez : i64
  %c = icmp ne %d, i64 0 : i1
  cond_br %c, ^1, ^2
^1:
  %q = udiv i64 100, %d : i64
  ret %q
^2:
  ret i64 0
}
"#;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    (m, syms)
}

/// Assert `m` verifies, including the constant-time check.
fn assert_ct(m: &Module, syms: &StrInterner, what: &str) {
    if let Err(d) = verify_module(m) {
        panic!("{what}: {d:#?}\n{}", crate::ir::text::print_module(m, syms));
    }
    assert!(m.has_secrets(), "{what}: secrecy must survive");
}

/// Run `passes` one at a time over a fresh parse of `src`, verifying after
/// every one.
fn run_checked(src: &str, passes: Vec<Box<dyn ModulePass>>, what: &str) -> (Module, StrInterner) {
    let (mut m, syms) = parse(src);
    assert_ct(&m, &syms, &format!("{what}: input"));
    for (k, mut p) in passes.into_iter().enumerate() {
        let name = p.name().to_owned();
        p.run(&mut m);
        assert_ct(&m, &syms, &format!("{what}: after pass {k} ({name})"));
    }
    (m, syms)
}

fn fixtures() -> Vec<(&'static str, String)> {
    vec![("ladder", LADDER_LF.to_owned()), ("mixed", MIXED_LF.to_owned())]
}

#[test]
fn every_pass_preserves_constant_time_on_the_fixtures() {
    for (name, src) in fixtures() {
        for pass in PASSES {
            run_checked(&src, vec![pass_by_name(pass).unwrap()], &format!("{name}/{pass}"));
        }
    }
}

#[test]
fn every_pipeline_preserves_constant_time_on_the_fixtures() {
    for (name, src) in fixtures() {
        for level in [OptLevel::O1, OptLevel::O2, OptLevel::O3] {
            run_checked(&src, pipeline_for(level), &format!("{name}/{}", level.name()));
        }
    }
}

#[test]
fn optimization_really_happens_under_secrets() {
    // The check is not vacuous: O2 inlines, promotes and folds secret code.
    let (m, syms) = run_checked(MIXED_LF, pipeline_for(OptLevel::O2), "mixed/O2");
    let text = crate::ir::text::print_module(&m, &syms);
    let body = |name: &str| {
        let start = text.find(&format!("func @{name}(")).unwrap();
        let end = text[start..].find("\n}\n").map_or(text.len(), |e| start + e);
        text[start..end].to_owned()
    };
    assert!(!body("locals").contains("alloca"), "mem2reg promoted the slot:\n{}", body("locals"));
    assert!(!body("ct_min").contains("call"), "inlined:\n{}", body("ct_min"));
    assert!(!body("folds").contains("cond_br"), "the constant branch folded:\n{}", body("folds"));
    assert!(body("verdict").contains("declassify"), "declassify survives:\n{}", body("verdict"));
}

/// A tiny deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Generate a random constant-time module: a helper with a secret parameter
/// (inlining food) and a function mixing secret and public arithmetic,
/// selects on secret and public conditions, a stack slot holding secrets, a
/// public-indexed array, a division by a public value, a public diamond
/// whose block arguments carry secrets, a counted loop, and a declassified
/// branch. Every construct is constant-time by construction.
fn random_module(seed: u64) -> String {
    let mut r = Rng(seed | 1);
    let mut s = String::from("module \"rand\"\n\n");
    s += "func @helper(secret i64, i64) -> secret i64 {\nentry ^0(%hs: i64, %hp: i64):\n";
    s += "  %h1 = mul %hs, %hp : i64\n  %h2 = xor %h1, %hs : i64\n  %hc = icmp ult %h2, %hp : i1\n";
    s += "  %h3 = select %hc, %h2, %hp : i64\n  ret %h3\n}\n\n";
    s += "func @f(secret i64, secret i64, i64, i64) -> secret i64 {\n";
    s += "entry ^0(%s0: i64, %s1: i64, %p0: i64, %p1: i64):\n";
    s += "  %slot = alloca i64 : ptr\n  %arr = alloca [8 x i64] : ptr\n";
    s += "  store %s0, %slot align 8 : i64\n";
    let mut secret = vec!["%s0".to_owned(), "%s1".to_owned()];
    let mut public = vec!["%p0".to_owned(), "%p1".to_owned()];
    let mut n = 0usize;
    let fresh = |n: &mut usize| {
        *n += 1;
        format!("%v{n}")
    };
    let ops = ["add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr"];
    let mut block = 0usize;
    for _ in 0..(12 + r.below(12)) {
        let pick = |r: &mut Rng, v: &Vec<String>| v[r.below(v.len())].clone();
        match r.below(12) {
            // Arithmetic on anything; secret if any operand is.
            0..=3 => {
                let op = ops[r.below(ops.len())];
                let a_sec = r.below(2) == 0;
                let a = if a_sec { pick(&mut r, &secret) } else { pick(&mut r, &public) };
                let b = if r.below(3) == 0 {
                    format!("i64 {}", r.below(9))
                } else if r.below(2) == 0 {
                    pick(&mut r, &secret)
                } else {
                    pick(&mut r, &public)
                };
                let v = fresh(&mut n);
                s += &format!("  {v} = {op} {a}, {b} : i64\n");
                if a_sec || secret.contains(&b) { secret.push(v) } else { public.push(v) }
            }
            // A comparison feeding a select (secret condition allowed).
            4 => {
                let a = pick(&mut r, &secret);
                let b = pick(&mut r, &public);
                let c = fresh(&mut n);
                let v = fresh(&mut n);
                let t = pick(&mut r, &public);
                s += &format!("  {c} = icmp ult {a}, {b} : i1\n  {v} = select {c}, {t}, {a} : i64\n");
                secret.push(v);
            }
            // Public-only work: a division and an array access by a public index.
            5 => {
                let a = pick(&mut r, &public);
                let b = pick(&mut r, &public);
                let d = fresh(&mut n);
                let q = fresh(&mut n);
                s += &format!("  {d} = or {b}, i64 1 : i64\n  {q} = udiv {a}, {d} : i64\n");
                let i = fresh(&mut n);
                let o = fresh(&mut n);
                let e = fresh(&mut n);
                let x = fresh(&mut n);
                s += &format!("  {i} = and {q}, i64 7 : i64\n  {o} = mul {i}, i64 8 : i64\n");
                s += &format!("  {e} = ptr_add %arr, {o} : ptr\n  {x} = load {e} align 8 : i64\n");
                public.extend([d, q, i, x]);
            }
            // The stack slot holds secrets.
            6 => {
                let a = pick(&mut r, &secret);
                let v = fresh(&mut n);
                s += &format!("  store {a}, %slot align 8 : i64\n  {v} = load %slot align 8 : i64\n");
                secret.push(v);
            }
            // A helper call (secret argument to the secret parameter).
            7 => {
                let a = pick(&mut r, &secret);
                let b = pick(&mut r, &public);
                let v = fresh(&mut n);
                s += &format!("  {v} = call @helper({a}, {b}) : i64\n");
                secret.push(v);
            }
            // A public diamond whose join carries a secret.
            8 => {
                let c = fresh(&mut n);
                let p = pick(&mut r, &public);
                let a = pick(&mut r, &secret);
                let b = pick(&mut r, &secret);
                let j = fresh(&mut n);
                let (t, e, m) = (block + 1, block + 2, block + 3);
                s += &format!("  {c} = icmp slt {p}, i64 {} : i1\n", r.below(50));
                s += &format!("  cond_br {c}, ^{t}, ^{e}\n^{t}:\n  br ^{m}({a})\n^{e}:\n  br ^{m}({b})\n");
                s += &format!("^{m}({j}: i64):\n");
                block = m;
                secret.push(j);
            }
            // A counted loop accumulating a secret.
            9 => {
                let a = pick(&mut r, &secret);
                let (h, body, exit) = (block + 1, block + 2, block + 3);
                let (i, acc, c, i2, acc2) =
                    (fresh(&mut n), fresh(&mut n), fresh(&mut n), fresh(&mut n), fresh(&mut n));
                s += &format!("  br ^{h}(i64 0, {a})\n^{h}({i}: i64, {acc}: i64):\n");
                s += &format!("  {c} = icmp ult {i}, i64 {} : i1\n  cond_br {c}, ^{body}, ^{exit}\n", 1 + r.below(4));
                s += &format!("^{body}:\n  {acc2} = add {acc}, {a} : i64\n  {i2} = add {i}, i64 1 : i64\n");
                s += &format!("  br ^{h}({i2}, {acc2})\n^{exit}:\n");
                block = exit;
                public.push(i);
                secret.push(acc);
            }
            // Declassify, then branch and divide on the result.
            10 => {
                let a = pick(&mut r, &secret);
                let d = fresh(&mut n);
                let c = fresh(&mut n);
                let q = fresh(&mut n);
                s += &format!("  {d} = declassify {a} : i64\n  {c} = icmp eq {d}, i64 0 : i1\n");
                let (t, m) = (block + 1, block + 2);
                s += &format!("  cond_br {c}, ^{t}, ^{m}\n^{t}:\n  br ^{m}\n^{m}:\n");
                s += &format!("  {q} = urem {d}, i64 10 : i64\n");
                block = m;
                public.extend([d, q]);
            }
            // Casts on secrets.
            _ => {
                let a = pick(&mut r, &secret);
                let t = fresh(&mut n);
                let z = fresh(&mut n);
                s += &format!("  {t} = trunc {a} : i32\n  {z} = sext {t} : i64\n");
                secret.push(z);
            }
        }
    }
    // Combine everything into the (secret) result.
    let mut acc = secret[0].clone();
    for v in secret.iter().skip(1).chain(public.iter()) {
        let t = fresh(&mut n);
        s += &format!("  {t} = xor {acc}, {v} : i64\n");
        acc = t;
    }
    s += &format!("  ret {acc}\n}}\n");
    s
}

#[test]
fn random_constant_time_functions_survive_every_pass() {
    for seed in 1..=40u64 {
        let src = random_module(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for pass in PASSES {
            run_checked(&src, vec![pass_by_name(pass).unwrap()], &format!("seed {seed}/{pass}"));
        }
    }
}

#[test]
fn random_constant_time_functions_survive_pipelines_and_random_orders() {
    for seed in 1..=25u64 {
        let src = random_module(seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
        for level in [OptLevel::O1, OptLevel::O2, OptLevel::O3] {
            run_checked(&src, pipeline_for(level), &format!("seed {seed}/{}", level.name()));
        }
        let mut r = Rng(seed * 7919 + 1);
        let order: Vec<Box<dyn ModulePass>> =
            (0..12).map(|_| pass_by_name(PASSES[r.below(PASSES.len())]).unwrap()).collect();
        run_checked(&src, order, &format!("seed {seed}/random order"));
    }
}

#[test]
fn the_superoptimizer_keeps_constant_time() {
    use crate::transform::superopt::{Budget, superoptimize_ct};
    // Single-block, pure: the superoptimizer's subset. Its candidates may use
    // shifts, which the strict policy forbids on a secret amount.
    let src = r#"module "so"
func @f(secret i64, i64) -> secret i64 {
entry ^0(%s: i64, %p: i64):
  %a = mul %s, i64 2 : i64
  %b = add %a, i64 0 : i64
  ret %b
}
"#;
    let (mut m, _syms) = parse(src);
    let f = FuncId::from_index(0);
    let budget = Budget::default();
    for policy in [CtPolicy::DEFAULT, CtPolicy::STRICT] {
        let before = ct_violations(&m, f, policy).len();
        if let Some(cand) = superoptimize_ct(&mut m, f, &budget, policy) {
            let old = m.swap_function(f, cand);
            assert!(ct_violations(&m, f, policy).len() <= before, "{policy:?}");
            m.replace_function(f, old);
        }
    }
}

/// Wide secret arithmetic for the integer legalizer: every operation it
/// expands in place (bitwise, add/sub carry chains, constant and variable
/// shifts, comparisons, selects, casts, loads/stores per part).
const WIDE_LF: &str = r#"
module "wide"

func @wide(secret i128, secret i128, i64, ptr) -> secret i128 {
entry ^0(%a: i128, %b: i128, %n: i64, %p: ptr):
  %x = xor %a, %b : i128
  %s = add %x, %a : i128
  %d = sub %s, %b : i128
  %k = trunc %b : i8
  %k2 = zext %k : i128
  %sh = shl %d, %k2 : i128
  %sr = lshr %sh, i128 13 : i128
  %sa = ashr %sr, %k2 : i128
  %c = icmp ult %sa, %a : i1
  %c2 = icmp eq %sa, %b : i1
  %c3 = xor %c, %c2 : i1
  %m = select %c3, %sa, %d : i128
  %t = trunc %m : i64
  %e = sext %t : i128
  %l = load secret %p align 16 : i128
  %r = or %e, %l : i128
  store secret %r, %p align 16 : i128
  %np = mul %n, i64 3 : i64
  %c4 = icmp ult %np, i64 100 : i1
  cond_br %c4, ^1, ^2
^1:
  ret %r
^2:
  ret %m
}
"#;

/// The number of `cond_br`/`switch` terminators of a module.
fn branch_count(m: &Module) -> usize {
    m.functions()
        .flat_map(|f| {
            f.blocks().filter_map(|(_, b)| b.terminator()).map(move |t| &f.inst(t).kind)
        })
        .filter(|k| matches!(k, crate::ir::InstKind::CondBr { .. } | crate::ir::InstKind::Switch(_)))
        .count()
}

#[test]
fn integer_legalization_keeps_constant_time() {
    use crate::codegen::legalize_int::{LegalizeOptions, legalize_ints};
    for part in [8u32, 16, 32, 64] {
        let (mut m, mut syms) = parse(WIDE_LF);
        assert_ct(&m, &syms, "wide: input");
        let branches = branch_count(&m);
        let loads = |m: &Module| -> (usize, usize) {
            let f = m.function(FuncId::from_index(0));
            let mut all = 0;
            let mut secret = 0;
            for (_, b) in f.blocks() {
                for &i in b.insts() {
                    if let crate::ir::InstKind::Load { secret: s, .. } = f.inst(i).kind {
                        all += 1;
                        secret += usize::from(s);
                    }
                }
            }
            (all, secret)
        };
        legalize_ints(&mut m, &mut syms, &LegalizeOptions::new(part))
            .unwrap_or_else(|e| panic!("legalize at {part}: {e:?}"));
        // Splitting introduces no branch (carries and compares are
        // `icmp`/`select`), every part access keeps `secret`, and the result
        // is still constant-time.
        assert_eq!(branch_count(&m), branches, "part width {part}");
        let (all, secret) = loads(&m);
        assert_eq!(all, secret, "every part load stays secret at {part}");
        assert!(all >= (128 / part) as usize, "the load was split at {part}");
        assert_ct(&m, &syms, &format!("wide legalized at {part}"));
    }
}

#[test]
fn a_wide_secret_multiply_becoming_a_libcall_is_rejected() {
    use crate::codegen::legalize_int::{LegalizeOptions, legalize_ints};
    // A libcall of unknown timing may not see a secret: after splitting, the
    // secret operands flow into `__multi3`'s public parameters.
    let src = r#"module "mul"
func @f(secret i128, secret i128) -> secret i128 {
entry ^0(%a: i128, %b: i128):
  %r = mul %a, %b : i128
  ret %r
}
"#;
    let (mut m, mut syms) = parse(src);
    assert_ct(&m, &syms, "mul: input");
    legalize_ints(&mut m, &mut syms, &LegalizeOptions::new(64)).expect("legalize");
    let v = ct_violations(&m, FuncId::from_index(0), CtPolicy::DEFAULT);
    assert!(
        v.iter().any(|v| matches!(v.role, crate::verify::CtRole::PublicParameter(_))),
        "{v:?}"
    );
}

#[test]
fn yield_points_branch_only_on_the_public_flag() {
    use crate::transform::yield_points::{YieldConfig, optimize_with_yield_points};
    // The ladder's loops have unbounded trip counts, so each gets a check; the
    // check loads the preemption flag and branches on it, which is public even
    // though the function also writes secrets through pointers.
    for level in [OptLevel::O0, OptLevel::O2] {
        let (mut m, mut syms) = parse(LADDER_LF);
        let config = YieldConfig::new(syms.intern("preempt_flag"), syms.intern("lf_yield"));
        let before = branch_count(&m);
        optimize_with_yield_points(&mut m, level, config);
        let checks = m
            .functions()
            .flat_map(|f| (0..f.inst_count()).map(move |i| f.inst(crate::ir::InstId::from_index(i)).kind.clone()))
            .filter(|k| matches!(k, crate::ir::InstKind::Load { volatile: true, .. }))
            .count();
        assert!(checks > 0 && branch_count(&m) > before, "checks were inserted at {level:?}");
        assert_ct(&m, &syms, &format!("ladder with yield points at {level:?}"));
    }
}
