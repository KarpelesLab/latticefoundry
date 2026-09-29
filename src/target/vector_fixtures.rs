//! Shared vector test fixtures (`docs/ir-design.md` §6c): a harness that turns
//! a set of pure `@tN(i64, …) -> i64` test functions into a program whose
//! `@main` calls each on fixed inputs and reports every result, the matching
//! reference results (from the reference executor on the *unoptimized* IR),
//! and a random generator of well-typed vector programs.
//!
//! Each backend runs the same fixtures its own way (x86-64 natively, AArch64
//! and RISC-V on their MIR interpreters after scalarization) and compares with
//! the reference executor, so the lowering of every vector op is checked
//! against the semantics.

use crate::ir::refexec::run_named;
use crate::ir::semantics::SemValue;
use crate::ir::Module;
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

use puremp::Int;

/// A test case: a function name and its `i64` arguments.
pub(crate) type Case = (String, Vec<i64>);

/// Parse and verify `src`.
pub(crate) fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse .lf: {e:?}\n{src}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    (m, syms)
}

/// The reference result of each case on the unoptimized `src`: `Some(bits)`,
/// or `None` if the result is poison (any value refines it). A case whose
/// reference run has undefined behavior is a bug in the test.
pub(crate) fn reference(src: &str, cases: &[Case]) -> Vec<Option<u64>> {
    let (m, syms) = parse(src);
    cases
        .iter()
        .map(|(name, args)| {
            let a: Vec<SemValue> = args.iter().map(|&x| SemValue::int(64, Int::from_i64(x))).collect();
            match run_named(&m, &syms, name, &a) {
                Ok(Some(SemValue::Int { bits, .. })) => Some(bits.to_u64().expect("an i64")),
                Ok(Some(SemValue::Poison)) => None,
                other => panic!("reference run of @{name}{args:?}: {other:?}"),
            }
        })
        .collect()
}

/// `src` plus a `@main` that calls every case and writes the `i64` results to
/// stdout (Linux `write`), returning 0. (Only the native x86-64 Linux tests
/// run such a program.)
#[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
pub(crate) fn with_stdout_main(src: &str, cases: &[Case]) -> String {
    let n = cases.len().max(1);
    let mut s = String::from(src);
    s += &format!("\nfunc @main() -> i64 {{\nentry ^0:\n  %buf = alloca [{n} x i64] : ptr\n");
    for (k, (name, args)) in cases.iter().enumerate() {
        let a: Vec<String> = args.iter().map(|x| format!("i64 {x}")).collect();
        s += &format!("  %r{k} = call @{name}({}) : i64\n", a.join(", "));
        s += &format!("  %p{k} = ptr_add %buf, i64 {} : ptr\n", 8 * k);
        s += &format!("  store %r{k}, %p{k} align 8 : i64\n");
    }
    s += &format!("  %w = syscall i64 1, i64 1, %buf, i64 {} : i64\n  ret i64 0\n}}\n", 8 * cases.len());
    s
}

/// Compare backend results with the reference ones (poison references accept
/// anything), naming the first mismatching case.
#[track_caller]
pub(crate) fn assert_matches(what: &str, cases: &[Case], got: &[u64], want: &[Option<u64>]) {
    assert_eq!(got.len(), want.len(), "{what}: result count");
    for (k, ((name, args), (&g, w))) in cases.iter().zip(got.iter().zip(want)).enumerate() {
        if let Some(w) = w {
            assert_eq!(g, *w, "{what}: case #{k} @{name}{args:?}: got {g:#x}, reference {w:#x}");
        }
    }
}

// ---------------------------------------------------------------------------
// Random vector programs
// ---------------------------------------------------------------------------

/// A deterministic SplitMix64 generator.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

/// A 128-bit vector type of the generator: `(lane type, lanes)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct VTy {
    lane: &'static str,
    n: u32,
}

impl VTy {
    fn name(self) -> String {
        format!("<{} x {}>", self.n, self.lane)
    }
    fn is_float(self) -> bool {
        self.lane.starts_with('f')
    }
    fn width(self) -> u32 {
        self.lane[1..].parse().expect("a lane width")
    }
    fn mask(self) -> String {
        format!("<{} x i1>", self.n)
    }
}

const TYPES: [VTy; 6] = [
    VTy { lane: "i8", n: 16 },
    VTy { lane: "i16", n: 8 },
    VTy { lane: "i32", n: 4 },
    VTy { lane: "i64", n: 2 },
    VTy { lane: "f32", n: 4 },
    VTy { lane: "f64", n: 2 },
];

/// A random well-typed vector program of `count` test functions
/// `@g0..`, each `(i64, i64, i64, i64) -> i64`, avoiding undefined behavior
/// (divisors are forced nonzero and positive, shift amounts are masked or
/// uniform constants). `floats` allows float vectors (a target without FP
/// support excludes them). Returns the source and the function names.
pub(crate) fn random_program(seed: u64, count: usize, steps: usize, floats: bool) -> (String, Vec<String>) {
    let mut rng = Rng(seed);
    let mut src = String::from("module \"rand\"\n");
    let mut names = Vec::new();
    for f in 0..count {
        let name = format!("g{f}");
        src += &random_function(&mut rng, &name, steps, floats);
        names.push(name);
    }
    (src, names)
}

/// A constant splat of `x` in the lanes of `t`, as a vector constant operand.
fn splat_const(t: VTy, x: i64) -> String {
    let lane = if t.is_float() {
        // Only used for integer types.
        unreachable!("integer splat on a float type")
    } else {
        format!("{} {x}", t.lane)
    };
    format!("{} ({})", t.name(), vec![lane; t.n as usize].join(", "))
}

fn random_function(rng: &mut Rng, name: &str, steps: usize, floats: bool) -> String {
    let types: Vec<VTy> = TYPES.iter().copied().filter(|t| floats || !t.is_float()).collect();
    let mut b = String::new();
    let mut k = 0usize;
    let mut fresh = |prefix: &str| {
        k += 1;
        format!("%{prefix}{k}")
    };
    b += &format!("func @{name}(i64, i64, i64, i64) -> i64 {{\nentry ^0(%a: i64, %b: i64, %c: i64, %d: i64):\n");
    // Two seed bit patterns, as <2 x i64>.
    let mut pool: Vec<(String, VTy)> = Vec::new();
    for (x, y) in [("%a", "%b"), ("%c", "%d")] {
        let p = fresh("s");
        b += &format!("  {p} = insertelement <2 x i64> poison, {x}, 0 : <2 x i64>\n");
        let q = fresh("s");
        b += &format!("  {q} = insertelement {p}, {y}, 1 : <2 x i64>\n");
        pool.push((q, TYPES[3]));
    }
    // Reinterpret the seeds as the working types.
    let conv = |b: &mut String, v: &str, from: VTy, to: VTy, fresh: &mut dyn FnMut(&str) -> String| -> String {
        if from == to {
            return v.to_string();
        }
        let r = fresh("bc");
        *b += &format!("  {r} = bitcast {v} : {}\n", to.name());
        r
    };
    for _ in 0..steps {
        let t = *rng.pick(&types);
        let (v1, t1) = pool[rng.below(pool.len() as u64) as usize].clone();
        let (v2, t2) = pool[rng.below(pool.len() as u64) as usize].clone();
        let x = conv(&mut b, &v1, t1, t, &mut fresh);
        let y = conv(&mut b, &v2, t2, t, &mut fresh);
        let r = fresh("v");
        let w = t.width();
        let choice = rng.below(12);
        if t.is_float() {
            match choice {
                0..=3 => {
                    let op = *rng.pick(&["fadd", "fsub", "fmul", "fdiv"]);
                    b += &format!("  {r} = {op} {x}, {y} : {}\n", t.name());
                }
                4 | 5 => {
                    let pred = *rng.pick(&[
                        "oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno",
                        "false", "true",
                    ]);
                    let m = fresh("m");
                    b += &format!("  {m} = fcmp {pred} {x}, {y} : {}\n", t.mask());
                    b += &format!("  {r} = select {m}, {x}, {y} : {}\n", t.name());
                }
                6 => b += &format!("  {r} = fneg {x} : {}\n", t.name()),
                7 | 8 => {
                    let mask: Vec<String> = (0..t.n).map(|_| rng.below(u64::from(2 * t.n)).to_string()).collect();
                    b += &format!("  {r} = shufflevector {x}, {y}, [{}] : {}\n", mask.join(", "), t.name());
                }
                9 => {
                    let (i, j) = (rng.below(u64::from(t.n)), rng.below(u64::from(t.n)));
                    let e = fresh("e");
                    b += &format!("  {e} = extractelement {x}, {i} : {}\n", t.lane);
                    if rng.below(2) == 0 {
                        b += &format!("  {r} = insertelement {y}, {e}, {j} : {}\n", t.name());
                    } else {
                        b += &format!("  {r} = splat {e} : {}\n", t.name());
                    }
                }
                10 if t.lane == "f32" => {
                    // A round trip through <4 x i32>: fptosi may be poison for
                    // big lanes, so clamp first with a compare/select.
                    let a = fresh("abs");
                    b += &format!("  {a} = fmul {x}, <4 x f32> (f32 0x3a800000, f32 0x3a800000, f32 0x3a800000, f32 0x3a800000) : <4 x f32>\n");
                    let big = fresh("m");
                    b += &format!("  {big} = fcmp olt {a}, <4 x f32> (f32 0x4e000000, f32 0x4e000000, f32 0x4e000000, f32 0x4e000000) : <4 x i1>\n");
                    let small = fresh("m");
                    b += &format!("  {small} = fcmp ogt {a}, <4 x f32> (f32 0xce000000, f32 0xce000000, f32 0xce000000, f32 0xce000000) : <4 x i1>\n");
                    let ok = fresh("m");
                    b += &format!("  {ok} = and {big}, {small} : <4 x i1>\n");
                    let safe = fresh("sf");
                    b += &format!("  {safe} = select {ok}, {a}, <4 x f32> (f32 0x3f800000, f32 0x3f800000, f32 0x3f800000, f32 0x3f800000) : <4 x f32>\n");
                    let i = fresh("i");
                    b += &format!("  {i} = fptosi {safe} : <4 x i32>\n");
                    b += &format!("  {r} = sitofp {i} : <4 x f32>\n");
                }
                _ => b += &format!("  {r} = fsub {y}, {x} : {}\n", t.name()),
            }
        } else {
            match choice {
                0..=2 => {
                    let op = *rng.pick(&[
                        "add", "sub", "mul", "and", "or", "xor", "smin", "smax", "umin", "umax", "sadd_sat",
                        "uadd_sat", "ssub_sat", "usub_sat",
                    ]);
                    b += &format!("  {r} = {op} {x}, {y} : {}\n", t.name());
                }
                3 => {
                    // A uniform constant shift (legal on SSE2 for 16..64-bit lanes).
                    let op = *rng.pick(&["shl", "lshr", "ashr"]);
                    let amt = rng.below(u64::from(w)) as i64;
                    b += &format!("  {r} = {op} {x}, {} : {}\n", splat_const(t, amt), t.name());
                }
                4 => {
                    // A per-lane shift, masked below the width (never poison).
                    let op = *rng.pick(&["shl", "lshr", "ashr"]);
                    let m = fresh("amt");
                    b += &format!("  {m} = and {y}, {} : {}\n", splat_const(t, i64::from(w - 1)), t.name());
                    b += &format!("  {r} = {op} {x}, {m} : {}\n", t.name());
                }
                5 => {
                    // Division by a positive, nonzero divisor (no UB).
                    let op = *rng.pick(&["udiv", "urem", "sdiv", "srem"]);
                    let pos = fresh("dv");
                    let maxpos = if w == 64 { i64::MAX } else { (1i64 << (w - 1)) - 1 };
                    b += &format!("  {pos} = and {y}, {} : {}\n", splat_const(t, maxpos), t.name());
                    let nz = fresh("dv");
                    b += &format!("  {nz} = or {pos}, {} : {}\n", splat_const(t, 1), t.name());
                    b += &format!("  {r} = {op} {x}, {nz} : {}\n", t.name());
                }
                6 | 7 => {
                    let pred = *rng.pick(&["eq", "ne", "ugt", "uge", "ult", "ule", "sgt", "sge", "slt", "sle"]);
                    let m = fresh("m");
                    b += &format!("  {m} = icmp {pred} {x}, {y} : {}\n", t.mask());
                    match rng.below(3) {
                        0 => b += &format!("  {r} = select {m}, {x}, {y} : {}\n", t.name()),
                        1 => b += &format!("  {r} = sext {m} : {}\n", t.name()),
                        _ => {
                            let m2 = fresh("m");
                            b += &format!("  {m2} = trunc {y} : {}\n", t.mask());
                            let m3 = fresh("m");
                            let op = *rng.pick(&["and", "or", "xor"]);
                            b += &format!("  {m3} = {op} {m}, {m2} : {}\n", t.mask());
                            b += &format!("  {r} = zext {m3} : {}\n", t.name());
                        }
                    }
                }
                8 | 9 => {
                    let mask: Vec<String> = (0..t.n).map(|_| rng.below(u64::from(2 * t.n)).to_string()).collect();
                    b += &format!("  {r} = shufflevector {x}, {y}, [{}] : {}\n", mask.join(", "), t.name());
                }
                10 => {
                    let (i, j) = (rng.below(u64::from(t.n)), rng.below(u64::from(t.n)));
                    let e = fresh("e");
                    b += &format!("  {e} = extractelement {x}, {i} : {}\n", t.lane);
                    if rng.below(2) == 0 {
                        b += &format!("  {r} = insertelement {y}, {e}, {j} : {}\n", t.name());
                    } else {
                        b += &format!("  {r} = splat {e} : {}\n", t.name());
                    }
                }
                _ => {
                    // A reduction folded back in as a splat.
                    let op = *rng.pick(&["add", "mul", "and", "or", "xor", "smin", "smax", "umin", "umax"]);
                    let e = fresh("e");
                    b += &format!("  {e} = reduce {op} {x} : {}\n", t.lane);
                    let s = fresh("sp");
                    b += &format!("  {s} = splat {e} : {}\n", t.name());
                    b += &format!("  {r} = xor {s}, {y} : {}\n", t.name());
                }
            }
        }
        // Canonicalize NaN lanes (hardware and host NaN payloads may differ).
        let r = if t.is_float() {
            let nan = fresh("m");
            b += &format!("  {nan} = fcmp uno {r}, {r} : {}\n", t.mask());
            let c = fresh("v");
            let zero = if t.lane == "f32" { "f32 0x00000000" } else { "f64 0x0000000000000000" };
            b += &format!("  {c} = select {nan}, {} ({}), {r} : {}\n", t.name(), vec![zero; t.n as usize].join(", "), t.name());
            c
        } else {
            r
        };
        pool.push((r, t));
    }
    // Fold the last three values into one i64.
    let mut acc = String::from("i64 0");
    for (v, t) in pool.iter().rev().take(3).cloned().collect::<Vec<_>>() {
        let q = conv(&mut b, &v, t, TYPES[3], &mut fresh);
        let (l, h) = (fresh("l"), fresh("h"));
        b += &format!("  {l} = extractelement {q}, 0 : i64\n  {h} = extractelement {q}, 1 : i64\n");
        let x = fresh("x");
        b += &format!("  {x} = xor {l}, {h} : i64\n");
        let m = fresh("x");
        b += &format!("  {m} = mul {acc}, i64 31 : i64\n");
        let n = fresh("x");
        b += &format!("  {n} = add {m}, {x} : i64\n");
        acc = n;
    }
    b += &format!("  ret {acc}\n}}\n");
    b
}

/// Random 64-bit inputs, biased toward interesting bit patterns.
pub(crate) fn random_inputs(rng: &mut Rng) -> Vec<i64> {
    (0..4)
        .map(|_| match rng.below(5) {
            0 => 0,
            1 => -1,
            2 => i64::MIN,
            3 => (rng.next() & 0x00FF_00FF_00FF_00FF) as i64,
            _ => rng.next() as i64,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Targeted vector programs: `@name(i64, i64, i64, i64) -> i64` test functions
// ---------------------------------------------------------------------------

pub(crate) fn cases(names: &[&str], inputs: &[[i64; 4]]) -> Vec<Case> {
    names
        .iter()
        .flat_map(|n| inputs.iter().map(move |a| (n.to_string(), a.to_vec())))
        .collect()
}

pub(crate) const INPUTS: [[i64; 4]; 5] = [
    [0, 0, 0, 0],
    [1, 2, 3, 4],
    [-1, i64::MIN, i64::MAX, 0x0123_4567_89ab_cdef],
    [0x7f80_0001_ff00_8000, -0x5555_5555_5555_5556, 0x0102_0304_0506_0708, -42],
    [0x4000_0000_3f80_0000, 0x4059_0000_0000_0000u64 as i64, 0xc000_0000_bf80_0000u64 as i64, 0x7ff8_0000_0000_0000],
];

/// Builds the two `<2 x i64>` seeds from the four arguments.
const SEEDS: &str = "  %p0 = insertelement <2 x i64> poison, %a, 0 : <2 x i64>
  %x0 = insertelement %p0, %b, 1 : <2 x i64>
  %p1 = insertelement <2 x i64> poison, %c, 0 : <2 x i64>
  %y0 = insertelement %p1, %d, 1 : <2 x i64>
";

/// Fold a `<2 x i64>` `%out` into the `i64` result.
const FOLD: &str = "  %l = extractelement %out, 0 : i64
  %h = extractelement %out, 1 : i64
  %m = mul %h, i64 1000003 : i64
  %r = xor %l, %m : i64
  ret %r
";

pub(crate) fn func(name: &str, body: &str) -> String {
    // A `bitcast` to the same type is not a conversion: use `freeze` (the
    // identity on these defined values) for the `<2 x i64>` cases.
    let body = body
        .lines()
        .map(|l| match l.split_once(" = bitcast ") {
            Some((lhs, rhs)) if rhs.ends_with(": <2 x i64>") && is_q(lhs, rhs, body) => {
                format!("{lhs} = freeze {rhs}\n")
            }
            _ => format!("{l}\n"),
        })
        .collect::<String>();
    format!("func @{name}(i64, i64, i64, i64) -> i64 {{\nentry ^0(%a: i64, %b: i64, %c: i64, %d: i64):\n{SEEDS}{body}{FOLD}}}\n")
}

/// Whether the source of `lhs = bitcast rhs` in `body` is already `<2 x i64>`.
fn is_q(_lhs: &str, rhs: &str, body: &str) -> bool {
    let src = rhs.split(" : ").next().unwrap_or("").trim();
    if src == "%x0" || src == "%y0" {
        return true;
    }
    body.lines().any(|l| l.trim_start().starts_with(&format!("{src} = ")) && l.ends_with(": <2 x i64>"))
}

pub(crate) const INT_OPS: [&str; 18] = [
    "add", "sub", "mul", "and", "or", "xor", "udiv", "shl", "lshr", "ashr", "smin", "smax", "umin", "umax",
    "sadd_sat", "uadd_sat", "ssub_sat", "usub_sat",
];

/// Integer arithmetic on every SSE2 integer lane width.
pub(crate) fn int_arith_src() -> String {
    let mut s = String::from("module \"vint\"\n");
    for (t, n) in [("i8", 16), ("i16", 8), ("i32", 4), ("i64", 2)] {
        let ty = format!("<{n} x {t}>");
        for op in INT_OPS {
            let body = if op == "udiv" {
                // A divisor forced odd (nonzero): scalarized on SSE2.
                format!(
                    "  %x = bitcast %x0 : {ty}\n  %y = bitcast %y0 : {ty}\n  %one = splat {t} 1 : {ty}\n  %yy = or %y, %one : {ty}\n  %v = udiv %x, %yy : {ty}\n  %out = bitcast %v : <2 x i64>\n"
                )
            } else if matches!(op, "shl" | "lshr" | "ashr") {
                // A uniform constant amount (SSE2 psll/psrl/psra where they exist).
                let k = match t { "i8" => 3, "i16" => 5, "i32" => 13, _ => 37 };
                let amt = format!("{ty} ({})", vec![format!("{t} {k}"); n].join(", "));
                format!("  %x = bitcast %x0 : {ty}\n  %v = {op} %x, {amt} : {ty}\n  %out = bitcast %v : <2 x i64>\n")
            } else {
                format!("  %x = bitcast %x0 : {ty}\n  %y = bitcast %y0 : {ty}\n  %v = {op} %x, %y : {ty}\n  %out = bitcast %v : <2 x i64>\n")
            };
            s += &func(&format!("{op}_{t}"), &body);
        }
    }
    s
}


pub(crate) fn compare_src() -> String {
    let mut s = String::from("module \"vcmp\"\n");
    for (t, n) in [("i8", 16), ("i16", 8), ("i32", 4), ("i64", 2)] {
        let ty = format!("<{n} x {t}>");
        let mt = format!("<{n} x i1>");
        for p in ["eq", "ne", "ugt", "uge", "ult", "ule", "sgt", "sge", "slt", "sle"] {
            s += &func(
                &format!("icmp_{p}_{t}"),
                &format!("  %x = bitcast %x0 : {ty}\n  %y = bitcast %y0 : {ty}\n  %z = shufflevector %x, %y, [{}] : {ty}\n  %m = icmp {p} %x, %z : {mt}\n  %v = select %m, %x, %y : {ty}\n  %w = sext %m : {ty}\n  %u = xor %v, %w : {ty}\n  %out = bitcast %u : <2 x i64>\n",
                    (0..n).map(|i| if i % 3 == 0 { (i + n).to_string() } else { i.to_string() }).collect::<Vec<_>>().join(", ")
                ),
            );
        }
        s += &func(
            &format!("masks_{t}"),
            &format!("  %x = bitcast %x0 : {ty}\n  %y = bitcast %y0 : {ty}\n  %m = trunc %x : {mt}\n  %k = trunc %y : {mt}\n  %a1 = and %m, %k : {mt}\n  %o1 = or %m, %k : {mt}\n  %x1 = xor %a1, %o1 : {mt}\n  %z = zext %x1 : {ty}\n  %s = sext %a1 : {ty}\n  %u = sub %z, %s : {ty}\n  %out = bitcast %u : <2 x i64>\n"),
        );
    }
    for (t, n) in [("f32", 4), ("f64", 2)] {
        let ty = format!("<{n} x {t}>");
        let mt = format!("<{n} x i1>");
        for p in ["false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno", "true"] {
            s += &func(
                &format!("fcmp_{p}_{t}"),
                &format!("  %x = bitcast %x0 : {ty}\n  %y = bitcast %y0 : {ty}\n  %z = shufflevector %x, %y, [{}] : {ty}\n  %m = fcmp {p} %x, %z : {mt}\n  %w = sext %m : <{n} x i{}>\n  %out = bitcast %w : <2 x i64>\n",
                    (0..n).map(|i| if i % 2 == 0 { i.to_string() } else { (i + n).to_string() }).collect::<Vec<_>>().join(", "),
                    128 / n
                ),
            );
        }
    }
    s
}


pub(crate) const FLOAT_SRC: &str = r#"
module "vflt"
func @farith(i64, i64, i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64):
  %p0 = insertelement <2 x i64> poison, %a, 0 : <2 x i64>
  %x0 = insertelement %p0, %b, 1 : <2 x i64>
  %p1 = insertelement <2 x i64> poison, %c, 0 : <2 x i64>
  %y0 = insertelement %p1, %d, 1 : <2 x i64>
  %i = bitcast %x0 : <4 x i32>
  %j = bitcast %y0 : <4 x i32>
  %small = and %i, <4 x i32> (i32 65535, i32 65535, i32 65535, i32 65535) : <4 x i32>
  %f = sitofp %small : <4 x f32>
  %g = sitofp %j : <4 x f32>
  %s = fadd %f, %g : <4 x f32>
  %t = fsub %s, %f : <4 x f32>
  %u = fmul %t, <4 x f32> (f32 0x3fc00000, f32 0xc0000000, f32 0x3f000000, f32 0x41200000) : <4 x f32>
  %v = fdiv %u, <4 x f32> (f32 0x40400000, f32 0x3f800000, f32 0xc0800000, f32 0x3e800000) : <4 x f32>
  %n = fneg %v : <4 x f32>
  %back = fptosi %f : <4 x i32>
  %d64 = bitcast %y0 : <2 x f64>
  %e = fadd %d64, <2 x f64> (f64 0x3ff0000000000000, f64 0xc000000000000000) : <2 x f64>
  %e2 = fmul %e, %e : <2 x f64>
  %e3 = fdiv %e2, <2 x f64> (f64 0x4008000000000000, f64 0x4010000000000000) : <2 x f64>
  %e4 = fneg %e3 : <2 x f64>
  %e5 = fsub %e4, %e : <2 x f64>
  %nan = fcmp uno %e5, %e5 : <2 x i1>
  %e6 = select %nan, <2 x f64> (f64 0x0, f64 0x0), %e5 : <2 x f64>
  %ni = bitcast %n : <2 x i64>
  %bi = bitcast %back : <2 x i64>
  %ei = bitcast %e6 : <2 x i64>
  %o1 = xor %ni, %bi : <2 x i64>
  %out = add %o1, %ei : <2 x i64>
  %l = extractelement %out, 0 : i64
  %h = extractelement %out, 1 : i64
  %m = mul %h, i64 1000003 : i64
  %r = xor %l, %m : i64
  ret %r
}
"#;


/// Every lane of every legal type through extract, insert and splat, and the
/// direct shuffle patterns plus scalarized ones (byte/word lanes).
pub(crate) fn lanes_src() -> String {
    let mut s = String::from("module \"vlane\"\n");
    for (t, n) in [("i8", 16u32), ("i16", 8), ("i32", 4), ("i64", 2), ("f32", 4), ("f64", 2)] {
        let ty = format!("<{n} x {t}>");
        for lane in 0..n {
            let other = (lane * 5 + 3) % n;
            s += &func(
                &format!("lane_{t}_{lane}"),
                &format!("  %x = bitcast %x0 : {ty}\n  %y = bitcast %y0 : {ty}\n  %e = extractelement %x, {lane} : {t}\n  %i = insertelement %y, %e, {other} : {ty}\n  %sp = splat %e : {ty}\n  %sh = shufflevector %i, %sp, [{}] : {ty}\n  %out = bitcast %sh : <2 x i64>\n",
                    (0..n).map(|k| ((k * 7 + lane) % (2 * n)).to_string()).collect::<Vec<_>>().join(", ")
                ),
            );
        }
    }
    // Direct SSE2 shuffle forms: pshufd (one source), shufps (0-1 / 2-3), shufpd.
    for (name, ty, mask) in [
        ("pshufd_a", "<4 x i32>", "[3, 2, 1, 0]"),
        ("pshufd_b", "<4 x i32>", "[4, 4, 7, 5]"),
        ("shufps_ab", "<4 x f32>", "[1, 3, 4, 6]"),
        ("shufps_ba", "<4 x i32>", "[7, 5, 2, 0]"),
        ("q_swap", "<2 x i64>", "[1, 0]"),
        ("q_ab", "<2 x f64>", "[1, 2]"),
        ("q_ba", "<2 x i64>", "[3, 0]"),
    ] {
        s += &func(
            name,
            &format!("  %x = bitcast %x0 : {ty}\n  %y = bitcast %y0 : {ty}\n  %sh = shufflevector %x, %y, {mask} : {ty}\n  %out = bitcast %sh : <2 x i64>\n"),
        );
    }
    s
}
