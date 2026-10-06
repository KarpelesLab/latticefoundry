//! Shared test fixtures for two-word values (`docs/ir-design.md` §3b and §6):
//! the `i128` operation suite every 64-bit backend is checked on against the
//! reference evaluator, and Lode-style programs returning two-word results —
//! a `throws(E) -> usize` result as a `{i64, i64}` struct, and an `i128` — with
//! the C side of their ABI round trips.

use std::fmt::Write as _;

use crate::codegen::MachineFunction;
use crate::codegen::target::MachineTarget;
use crate::ir::refexec::run_named;
use crate::ir::semantics::SemValue;
use crate::ir::{FuncId, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize};

use puremp::Int;

/// Every `i128` operation of the differential suite, as `(name, body)`: the
/// body computes `%r : i128` from the parameters `%a`, `%b`.
pub(crate) const OPS: &[(&str, &str)] = &[
    ("add", "%r = add %a, %b : i128"),
    ("sub", "%r = sub %a, %b : i128"),
    ("neg", "%r = sub i128 0, %a : i128"),
    ("mul", "%r = mul %a, %b : i128"),
    ("mulc", "%r = mul %a, i128 1000000000000000000000 : i128"),
    ("and", "%r = and %a, %b : i128"),
    ("or", "%r = or %a, %b : i128"),
    ("xor", "%r = xor %a, %b : i128"),
    ("shl", "%s = and %b, i128 127 : i128\n  %r = shl %a, %s : i128"),
    ("lshr", "%s = and %b, i128 127 : i128\n  %r = lshr %a, %s : i128"),
    ("ashr", "%s = and %b, i128 127 : i128\n  %r = ashr %a, %s : i128"),
    ("shl67", "%r = shl %a, i128 67 : i128"),
    ("lshr64", "%r = lshr %a, i128 64 : i128"),
    ("ashr100", "%r = ashr %a, i128 100 : i128"),
    ("ashr3", "%r = ashr %a, i128 3 : i128"),
    ("udiv", "%r = udiv %a, %b : i128"),
    ("sdiv", "%r = sdiv %a, %b : i128"),
    ("urem", "%r = urem %a, %b : i128"),
    ("srem", "%r = srem %a, %b : i128"),
    ("eq", "%c = icmp eq %a, %b : i1\n  %r = zext %c : i128"),
    ("ne", "%c = icmp ne %a, %b : i1\n  %r = zext %c : i128"),
    ("ult", "%c = icmp ult %a, %b : i1\n  %r = zext %c : i128"),
    ("ule", "%c = icmp ule %a, %b : i1\n  %r = zext %c : i128"),
    ("ugt", "%c = icmp ugt %a, %b : i1\n  %r = zext %c : i128"),
    ("uge", "%c = icmp uge %a, %b : i1\n  %r = zext %c : i128"),
    ("slt", "%c = icmp slt %a, %b : i1\n  %r = zext %c : i128"),
    ("sle", "%c = icmp sle %a, %b : i1\n  %r = zext %c : i128"),
    ("sgt", "%c = icmp sgt %a, %b : i1\n  %r = zext %c : i128"),
    ("sge", "%c = icmp sge %a, %b : i1\n  %r = zext %c : i128"),
    ("select", "%c = icmp slt %a, %b : i1\n  %r = select %c, %b, %a : i128"),
    ("smin", "%r = smin %a, %b : i128"),
    ("umax", "%r = umax %a, %b : i128"),
    ("const", "%r = add %a, i128 85070591730234615865843651857942052864 : i128"),
    ("sext64", "%t = trunc %a : i64\n  %r = sext %t : i128"),
    ("zext32", "%t = trunc %b : i32\n  %r = zext %t : i128"),
    ("sext8", "%t = trunc %a : i8\n  %u = sext %t : i128\n  %r = add %u, %b : i128"),
    (
        "mix",
        "%t = add %a, %b : i128\n  %u = mul %t, %a : i128\n  %v = lshr %u, i128 13 : i128\n  %r = xor %v, %b : i128",
    ),
    (
        "memory",
        "%p = alloca i128 : ptr\n  store %a, %p align 16 : i128\n  %q = load %p align 16 : i128\n  %r = sub %q, %b : i128",
    ),
    (
        "volatile",
        "%p = alloca i128 : ptr\n  store volatile %b, %p align 16 : i128\n  %q = load volatile %p align 16 : i128\n  %r = xor %q, %a : i128",
    ),
    ("sitofp", "%f = sitofp %a : f64\n  %x = bitcast %f : i64\n  %r = zext %x : i128"),
    ("uitofp32", "%f = uitofp %b : f32\n  %x = bitcast %f : i32\n  %r = zext %x : i128"),
    ("fptosi", "%t = trunc %a : i64\n  %f = sitofp %t : f64\n  %g = fmul %f, f64 0x430c6bf526340000 : f64\n  %r = fptosi %g : i128"),
    ("fptoui32", "%t = trunc %b : i32\n  %f = uitofp %t : f32\n  %g = fmul %f, f32 0x501502f9 : f32\n  %r = fptoui %g : i128"),
    ("ptr", "%p = inttoptr %a : ptr\n  %q = ptr_add %p, %b : ptr\n  %r = ptrtoint %q : i128"),
    ("vec", "%v = bitcast %a : <2 x i64>\n  %w = add %v, %v : <2 x i64>\n  %r = bitcast %w : i128"),
    (
        "switch",
        "switch %a, ^1 [5: ^2, 18446744073709551616: ^3, -1: ^2]\n^1:\n  ret %b\n^2:\n  ret i128 7\n^3:\n  %r = add %b, i128 1 : i128",
    ),
];

/// The operations that call libgcc (division, the float conversions): a
/// freestanding emulator run has no libgcc to link.
pub(crate) const LIBGCC_OPS: &[&str] = &["udiv", "sdiv", "urem", "srem", "sitofp", "uitofp32", "fptosi", "fptoui32"];

/// The operations only x86-64 lowers: a 128-bit `switch` and a bitcast to a
/// vector.
pub(crate) const X86_ONLY_OPS: &[&str] = &["vec", "switch"];

/// The suite's operations as one module: `@t_<name>(i128, i128) -> i128` per
/// operation `keep` accepts.
pub(crate) fn suite_src(keep: &dyn Fn(&str) -> bool) -> String {
    let mut s = String::from("module \"wide\"\n");
    for (name, body) in OPS.iter().filter(|(n, _)| keep(n)) {
        s.push_str(&format!(
            "func @t_{name}(i128, i128) -> i128 {{\nentry ^0(%a: i128, %b: i128):\n  {body}\n  ret %r\n}}\n"
        ));
    }
    s
}

/// Parse and verify `src`.
pub(crate) fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|d| panic!("verify: {d:#?}"));
    (m, syms)
}

/// Parse `src`, optimize it at `level` and verify the result.
pub(crate) fn parse_at(src: &str, level: OptLevel) -> (Module, StrInterner) {
    let (mut m, syms) = parse(src);
    optimize(&mut m, level);
    crate::verify::verify_module(&m).unwrap_or_else(|d| panic!("{level:?} verify: {d:#?}"));
    (m, syms)
}

/// A small deterministic generator (xorshift64*), so the suite needs no
/// dependency and every run checks the same cases.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A 128-bit operand: an edge value, a small (possibly negative) one, a
    /// value straddling the 64-bit boundary, or random bits.
    pub(crate) fn operand(&mut self) -> u128 {
        const EDGES: [u128; 12] = [
            0,
            1,
            2,
            u128::MAX,
            u128::MAX - 1,
            1 << 63,
            1 << 64,
            (1 << 64) - 1,
            1 << 127,
            (1 << 127) - 1,
            0xFFFF_FFFF_FFFF_FFFF_0000_0000_0000_0000,
            0x8000_0000_0000_0000_8000_0000_0000_0000,
        ];
        match self.next() % 5 {
            0 => EDGES[(self.next() % EDGES.len() as u64) as usize],
            1 => (self.next() % 1000) as u128,
            2 => ((self.next() % 1000) as i128).wrapping_neg() as u128,
            3 => (u128::from(self.next() % 4) << 64) | u128::from(self.next()),
            _ => (u128::from(self.next()) << 64) | u128::from(self.next()),
        }
    }
}

/// Whether the reference semantics is defined for `op(a, b)` (division by
/// zero and `MIN / -1` are undefined behavior).
pub(crate) fn defined(op: &str, a: u128, b: u128) -> bool {
    match op {
        "udiv" | "urem" => b != 0,
        "sdiv" | "srem" => b != 0 && !(a == 1 << 127 && b == u128::MAX),
        _ => true,
    }
}

/// The reference evaluator's `t_<op>(a, b)` on the suite module `m`.
pub(crate) fn reference(m: &Module, syms: &StrInterner, op: &str, a: u128, b: u128) -> u128 {
    let sem = |v: u128| SemValue::int(128, Int::from_u128(v));
    let want = run_named(m, syms, &format!("t_{op}"), &[sem(a), sem(b)])
        .unwrap_or_else(|e| panic!("reference t_{op}({a:#x}, {b:#x}): {e:?}"));
    let Some(SemValue::Int { bits, .. }) = want else { panic!("t_{op}: {want:?}") };
    bits.to_u128().expect("a 128-bit result")
}

/// A self-checking program for a freestanding run: the suite's operations
/// (except libgcc's and the x86-only ones) and a `@main() -> i64` (in its
/// own module, so no optimization of the suite sees the operands) calling
/// each on `per_op` random operand pairs against the reference evaluator's
/// results, returning a bitmask of the failing operations (bit = index in
/// the returned name list).
pub(crate) fn emu_programs(per_op: usize) -> (String, String, Vec<&'static str>) {
    let keep = |n: &str| !LIBGCC_OPS.contains(&n) && !X86_ONLY_OPS.contains(&n);
    let suite = suite_src(&keep);
    let (m, syms) = parse(&suite);
    let names: Vec<&'static str> = OPS.iter().map(|(n, _)| *n).filter(|n| keep(n)).collect();
    assert!(names.len() < 64);
    let mut rng = Rng(0x0BAD_5EED_1234_5678);
    let mut main = String::from("module \"wide_main\"\n");
    for n in &names {
        writeln!(main, "func @t_{n}(i128, i128) -> i128").unwrap();
    }
    main.push_str("func @main() -> i64 {\nentry ^0:\n");
    let mut acc = "i64 0".to_owned();
    let mut k = 0;
    for (bit, n) in names.iter().enumerate() {
        for _ in 0..per_op {
            let (a, b) = (rng.operand(), rng.operand());
            let w = reference(&m, &syms, n, a, b);
            writeln!(main, "  %r{k} = call @t_{n}(i128 {a}, i128 {b}) : i128").unwrap();
            writeln!(main, "  %c{k} = icmp ne %r{k}, i128 {w} : i1").unwrap();
            writeln!(main, "  %z{k} = zext %c{k} : i64").unwrap();
            writeln!(main, "  %s{k} = shl %z{k}, i64 {bit} : i64").unwrap();
            writeln!(main, "  %m{k} = or {acc}, %s{k} : i64").unwrap();
            acc = format!("%m{k}");
            k += 1;
        }
    }
    writeln!(main, "  ret {acc}\n}}").unwrap();
    (suite, main, names)
}

/// The names in `names` whose bit is set in `mask`.
pub(crate) fn failing<'a>(mask: u64, names: &[&'a str]) -> Vec<&'a str> {
    names.iter().enumerate().filter(|(i, _)| mask & (1 << i) != 0).map(|(_, n)| *n).collect()
}

/// Lode-style functions returning two words: `parse_digit` is a
/// `throws(E) -> usize` (`{error, value}`: error 22 for a non-digit), built
/// in branches; `pair` a plain `{i64, i64}`; `wide_pair` the same two words
/// as an `i128` (low word first).
pub(crate) const LODE_CALLEE: &str = r#"module "lode_callee"
func @parse_digit(i64) -> {i64, i64} {
entry ^0(%c: i64):
  %r = alloca {i64, i64} : ptr
  %lo = icmp ult %c, i64 48 : i1
  %hi = icmp ugt %c, i64 57 : i1
  %bad = or %lo, %hi : i1
  cond_br %bad, ^1, ^2
^1:
  store i64 22, %r align 8 : i64
  %p1 = ptr_add inbounds %r, i64 8 : ptr
  store i64 0, %p1 align 8 : i64
  br ^3
^2:
  store i64 0, %r align 8 : i64
  %p2 = ptr_add inbounds %r, i64 8 : ptr
  %v = sub %c, i64 48 : i64
  store %v, %p2 align 8 : i64
  br ^3
^3:
  ret %r
}

func @pair(i64, i64) -> {i64, i64} {
entry ^0(%a: i64, %b: i64):
  %r = alloca {i64, i64} : ptr
  store %a, %r align 8 : i64
  %p = ptr_add inbounds %r, i64 8 : ptr
  store %b, %p align 8 : i64
  ret %r
}

func @wide_pair(i64, i64) -> i128 {
entry ^0(%a: i64, %b: i64):
  %x = zext %a : i128
  %y = zext %b : i128
  %h = shl %y, i128 64 : i128
  %r = or %x, %h : i128
  ret %r
}
"#;

/// Lode-style callers, compiled separately from [`LODE_CALLEE`] (so the
/// calls stay calls): `sum_digits` branches on each result's error word
/// (returning `-error`, else the sum); `try_sum` propagates the error as its
/// own `throws` result; `wide_sum` adds an `i128` result's two words.
pub(crate) const LODE_CALLER: &str = r#"module "lode_caller"
func @parse_digit(i64) -> {i64, i64}
func @wide_pair(i64, i64) -> i128

func @sum_digits(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %ra = call @parse_digit(%a) : {i64, i64}
  %ea = load %ra align 8 : i64
  %fa = icmp ne %ea, i64 0 : i1
  cond_br %fa, ^3(%ea), ^1
^1:
  %pa = ptr_add inbounds %ra, i64 8 : ptr
  %va = load %pa align 8 : i64
  %rb = call @parse_digit(%b) : {i64, i64}
  %eb = load %rb align 8 : i64
  %fb = icmp ne %eb, i64 0 : i1
  cond_br %fb, ^3(%eb), ^2
^2:
  %pb = ptr_add inbounds %rb, i64 8 : ptr
  %vb = load %pb align 8 : i64
  %s = add %va, %vb : i64
  ret %s
^3(%e: i64):
  %n = sub i64 0, %e : i64
  ret %n
}

func @try_sum(i64, i64) -> {i64, i64} {
entry ^0(%a: i64, %b: i64):
  %out = alloca {i64, i64} : ptr
  %ra = call @parse_digit(%a) : {i64, i64}
  %ea = load %ra align 8 : i64
  %fa = icmp ne %ea, i64 0 : i1
  cond_br %fa, ^3(%ea), ^1
^1:
  %pa = ptr_add inbounds %ra, i64 8 : ptr
  %va = load %pa align 8 : i64
  %rb = call @parse_digit(%b) : {i64, i64}
  %eb = load %rb align 8 : i64
  %fb = icmp ne %eb, i64 0 : i1
  cond_br %fb, ^3(%eb), ^2
^2:
  %pb = ptr_add inbounds %rb, i64 8 : ptr
  %vb = load %pb align 8 : i64
  %s = add %va, %vb : i64
  store i64 0, %out align 8 : i64
  %po = ptr_add inbounds %out, i64 8 : ptr
  store %s, %po align 8 : i64
  br ^4
^3(%e: i64):
  store %e, %out align 8 : i64
  %pe = ptr_add inbounds %out, i64 8 : ptr
  store i64 0, %pe align 8 : i64
  br ^4
^4:
  ret %out
}

func @wide_sum(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %w = call @wide_pair(%a, %b) : i128
  %lo = trunc %w : i64
  %h = lshr %w, i128 64 : i128
  %hi = trunc %h : i64
  %s = add %lo, %hi : i64
  ret %s
}
"#;

/// A `@main() -> i64` checking [`LODE_CALLEE`] and [`LODE_CALLER`],
/// returning a bitmask of the failing checks (0: all pass; 7 bits, so it
/// survives as a process exit status).
pub(crate) const LODE_MAIN: &str = r#"module "lode_main"
func @pair(i64, i64) -> {i64, i64}
func @sum_digits(i64, i64) -> i64
func @try_sum(i64, i64) -> {i64, i64}
func @wide_sum(i64, i64) -> i64

func @main() -> i64 {
entry ^0:
  %a = call @sum_digits(i64 51, i64 52) : i64
  %ca = icmp ne %a, i64 7 : i1
  %b = call @sum_digits(i64 51, i64 65) : i64
  %cb = icmp ne %b, i64 -22 : i1
  %c = call @sum_digits(i64 120, i64 48) : i64
  %cc = icmp ne %c, i64 -22 : i1
  %t = call @try_sum(i64 57, i64 57) : {i64, i64}
  %te = load %t align 8 : i64
  %tp = ptr_add inbounds %t, i64 8 : ptr
  %tv = load %tp align 8 : i64
  %ct1 = icmp ne %te, i64 0 : i1
  %ct2 = icmp ne %tv, i64 18 : i1
  %u = call @try_sum(i64 49, i64 1) : {i64, i64}
  %ue = load %u align 8 : i64
  %up = ptr_add inbounds %u, i64 8 : ptr
  %uv = load %up align 8 : i64
  %cu1 = icmp ne %ue, i64 22 : i1
  %cu2 = icmp ne %uv, i64 0 : i1
  %p = call @pair(i64 5, i64 -3) : {i64, i64}
  %pa = load %p align 8 : i64
  %pp = ptr_add inbounds %p, i64 8 : ptr
  %pb = load %pp align 8 : i64
  %cp1 = icmp ne %pa, i64 5 : i1
  %cp2 = icmp ne %pb, i64 -3 : i1
  %w = call @wide_sum(i64 -1, i64 2) : i64
  %cw = icmp ne %w, i64 1 : i1
  %ct = or %ct1, %ct2 : i1
  %cu = or %cu1, %cu2 : i1
  %cp = or %cp1, %cp2 : i1
  %b0 = zext %ca : i64
  %b1 = zext %cb : i64
  %b2 = zext %cc : i64
  %b3 = zext %ct : i64
  %b4 = zext %cu : i64
  %b5 = zext %cp : i64
  %b6 = zext %cw : i64
  %s1 = shl %b1, i64 1 : i64
  %s2 = shl %b2, i64 2 : i64
  %s3 = shl %b3, i64 3 : i64
  %s4 = shl %b4, i64 4 : i64
  %s5 = shl %b5, i64 5 : i64
  %s6 = shl %b6, i64 6 : i64
  %o1 = or %b0, %s1 : i64
  %o2 = or %o1, %s2 : i64
  %o3 = or %o2, %s3 : i64
  %o4 = or %o3, %s4 : i64
  %o5 = or %o4, %s5 : i64
  %o6 = or %o5, %s6 : i64
  ret %o6
}
"#;

/// The C side of the two-word ABI round trips (freestanding): the same
/// functions as [`LODE_CALLEE`] (`c_*`), and `cmain`, which calls ours —
/// [`LODE_CALLER`]'s and [`ABI_LF`]'s — and C's, returning a bitmask of the
/// failing checks.
pub(crate) const ABI_C: &str = r#"
typedef struct { long err; unsigned long val; } R;
typedef __int128 i128;
typedef unsigned __int128 u128;

R parse_digit(long c);
R pair(long a, long b);
i128 wide_pair(long a, long b);
long sum_digits(long a, long b);
R try_sum(long a, long b);
long wide_sum(long a, long b);
i128 lf_mix(long, i128, long, i128, i128, long, i128);
i128 lf_late(long, long, long, long, long, long, long, i128, long);
R lf_calls_c(long a, long b);

R c_parse_digit(long c) {
    R r;
    if ((unsigned long)c < 48 || (unsigned long)c > 57) { r.err = 22; r.val = 0; }
    else { r.err = 0; r.val = (unsigned long)c - 48; }
    return r;
}
R c_pair(long a, long b) { R r = { a, (unsigned long)b }; return r; }
i128 c_wide_pair(long a, long b) { return (i128)(((u128)(unsigned long)b << 64) | (unsigned long)a); }
i128 c_mix(long a, i128 b, long c, i128 d, i128 e, long f, i128 g) {
    return (i128)((u128)b * 3 - (u128)d ^ ((u128)e << 1)) + g + a + (i128)(unsigned long)c - f;
}
i128 c_late(long a, long b, long c, long d, long e, long f, long g, i128 x, long y) {
    (void)a; (void)b; (void)c; (void)d; (void)f; (void)g;
    return (i128)((u128)x + ((u128)(unsigned long)y << 64)) - e;
}

static i128 mk(unsigned long hi, unsigned long lo) { return (i128)(((u128)hi << 64) | lo); }

int cmain(void) {
    int bad = 0;
    for (long c = 40; c < 70; c++) {
        R a = parse_digit(c), b = c_parse_digit(c);
        if (a.err != b.err || a.val != b.val) bad |= 1;
    }
    R p = pair(-9, 77);
    if (p.err != -9 || p.val != 77) bad |= 2;
    if (wide_pair(-2, 5) != c_wide_pair(-2, 5)) bad |= 4;
    if (sum_digits('7', '8') != 15 || sum_digits('7', '!') != -22) bad |= 8;
    R t = try_sum('9', '1'), u = try_sum('a', '1');
    if (t.err != 0 || t.val != 10 || u.err != 22 || u.val != 0) bad |= 16;
    if (wide_sum(3, 4) != 7) bad |= 32;
    i128 vals[] = { 0, 1, -1, mk(0x8000000000000000ul, 0), mk(0x0123456789abcdeful, 0xfedcba9876543210ul),
                    mk(0xfffffffful, 0xfffffffffffffffful), mk(0x7ffffffffffffffful, 0x1ul) };
    int n = sizeof vals / sizeof vals[0];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            i128 b = vals[i], d = vals[j], e = vals[(i + j) % n], g = vals[(i * 3 + j) % n];
            long a = (long)(i * 1000003 - j), c = (long)(j * 77 - 5), f = (long)i - 100;
            if (lf_mix(a, b, c, d, e, f, g) != c_mix(a, b, c, d, e, f, g)) bad |= 64;
            if (lf_late(1, 2, 3, 4, f, 6, 7, b, c) != c_late(1, 2, 3, 4, f, 6, 7, b, c)) bad |= 128;
        }
    R q = lf_calls_c('4', -6);
    if (q.err != 0 || q.val != 4) bad |= 256;
    return bad;
}
"#;

/// Our side of the `i128` argument round trips: register pairs, then the
/// stack once fewer than two argument registers are left (`lf_late`'s `x`
/// is the 8th and 9th word), and calls into C (`lf_calls_c` adds `c_mix`'s
/// low word, a `c_parse_digit` result and a `c_pair` result, and returns a
/// `throws` struct).
pub(crate) const ABI_LF: &str = r#"module "abi"
func @c_mix(i64, i128, i64, i128, i128, i64, i128) -> i128
func @c_late(i64, i64, i64, i64, i64, i64, i64, i128, i64) -> i128
func @c_parse_digit(i64) -> {i64, i64}
func @c_pair(i64, i64) -> {i64, i64}
func @c_wide_pair(i64, i64) -> i128

func @lf_mix(i64, i128, i64, i128, i128, i64, i128) -> i128 {
entry ^0(%a: i64, %b: i128, %c: i64, %d: i128, %e: i128, %f: i64, %g: i128):
  %a2 = sext %a : i128
  %c2 = zext %c : i128
  %f2 = sext %f : i128
  %s1 = mul %b, i128 3 : i128
  %s2 = sub %s1, %d : i128
  %s3 = shl %e, i128 1 : i128
  %s4 = xor %s2, %s3 : i128
  %s5 = add %s4, %g : i128
  %s6 = add %s5, %a2 : i128
  %s7 = add %s6, %c2 : i128
  %r = sub %s7, %f2 : i128
  ret %r
}

func @lf_late(i64, i64, i64, i64, i64, i64, i64, i128, i64) -> i128 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64, %e: i64, %f: i64, %g: i64, %x: i128, %y: i64):
  %y2 = zext %y : i128
  %s = shl %y2, i128 64 : i128
  %t = add %x, %s : i128
  %e2 = sext %e : i128
  %r = sub %t, %e2 : i128
  ret %r
}

func @lf_calls_c(i64, i64) -> {i64, i64} {
entry ^0(%a: i64, %b: i64):
  %out = alloca {i64, i64} : ptr
  %m = call @c_mix(i64 0, i128 0, i64 0, i128 0, i128 0, i64 0, i128 0) : i128
  %l = call @c_late(i64 1, i64 2, i64 3, i64 4, i64 0, i64 6, i64 7, %m, i64 0) : i128
  %lw = trunc %l : i64
  %w = call @c_wide_pair(i64 1, i64 2) : i128
  %wl = trunc %w : i64
  %d = call @c_parse_digit(%a) : {i64, i64}
  %de = load %d align 8 : i64
  %dp = ptr_add inbounds %d, i64 8 : ptr
  %dv = load %dp align 8 : i64
  %p = call @c_pair(%b, i64 7) : {i64, i64}
  %pa = load %p align 8 : i64
  %pp = ptr_add inbounds %p, i64 8 : ptr
  %pb = load %pp align 8 : i64
  %s1 = add %dv, %pa : i64
  %s2 = add %s1, %pb : i64
  %s3 = add %s2, %lw : i64
  %s4 = sub %s3, %wl : i64
  store %de, %out align 8 : i64
  %po = ptr_add inbounds %out, i64 8 : ptr
  store %s4, %po align 8 : i64
  ret %out
}
"#;

/// The stack slots — struct storage or spills — that function `name` of
/// `src`, optimized at `-O2`, needs after register allocation, `select`
/// lowering it (from the module and its names) for `target`. Zero means the
/// function's values never leave registers (the callee-saved registers and
/// the frame record aside).
pub(crate) fn frame_slots(
    src: &str,
    name: &str,
    target: &dyn MachineTarget,
    select: &dyn Fn(&Module, &StrInterner, &str) -> MachineFunction,
) -> usize {
    let (m, syms) = parse_at(src, OptLevel::O2);
    let mut mf = select(&m, &syms, name);
    crate::codegen::regalloc::allocate(&mut mf, target);
    mf.frame().len()
}

/// The index of function `name` in `m`.
pub(crate) fn func_id(m: &Module, syms: &StrInterner, name: &str) -> FuncId {
    let i = m.functions().position(|f| syms.resolve(f.name) == name).unwrap_or_else(|| panic!("no function @{name}"));
    FuncId::from_index(i)
}

/// The text of a caught panic.
pub(crate) fn panic_text(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned())).unwrap_or_default()
}

/// The functions of [`LODE_CALLEE`] and [`LODE_CALLER`] whose two-word
/// results must not touch memory at `-O2`.
pub(crate) const LODE_FUNCS: &[&str] = &["parse_digit", "pair", "wide_pair", "sum_digits", "try_sum", "wide_sum"];
