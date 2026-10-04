//! 128-bit integers on x86-64 (`docs/ir-design.md` §3b): what the
//! preparation leaves for instruction selection, the inline multiply, and — on
//! an x86-64 Linux host with gcc — a randomized differential suite of every
//! `i128` operation (reference evaluator vs. native code, at `-O0` and `-O2`)
//! and System V ABI round trips with gcc's `__int128` in both directions.

use crate::codegen::legalize_int::illegal_int_ops;
use crate::ir::text::parse_module;
use crate::ir::{FuncId, Module};
use crate::mc::object::{ObjectModule, RelocKind};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

use super::{MUL128_PSEUDO, prepare_module};

/// Every `i128` operation of the differential suite, as `(name, body)`: the
/// body computes `%r : i128` from the parameters `%a`, `%b`.
const OPS: &[(&str, &str)] = &[
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

/// The suite as one module: `@t_<name>(i128, i128) -> i128` per operation.
fn suite_src() -> String {
    let mut s = String::from("module \"wide\"\n");
    for (name, body) in OPS {
        s.push_str(&format!(
            "func @t_{name}(i128, i128) -> i128 {{\nentry ^0(%a: i128, %b: i128):\n  {body}\n  ret %r\n}}\n"
        ));
    }
    s
}

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|d| panic!("verify: {d:#?}"));
    (m, syms)
}

/// After preparation only the ABI boundary is wide; a 128-bit multiply calls
/// the inline-expanded pseudo-helper, division calls libgcc, and the float
/// conversions have their helpers declared.
#[test]
fn prepare_leaves_only_the_boundary_wide() {
    let (m, syms) = parse(&suite_src());
    let (p, names) = prepare_module(&m, &syms).expect("prepare");
    for i in 0..p.function_count() {
        let f = FuncId::from_index(i);
        if p.function(f).is_declaration() {
            continue;
        }
        assert!(illegal_int_ops(&p, f, 64).is_empty(), "{}", names.resolve(p.function(f).name));
    }
    let decls: Vec<&str> =
        p.functions().filter(|f| f.is_declaration()).map(|f| names.resolve(f.name)).collect();
    for want in [MUL128_PSEUDO, "__udivti3", "__divti3", "__umodti3", "__modti3", "__floattidf", "__floatuntisf", "__fixdfti", "__fixunssfti"] {
        assert!(decls.contains(&want), "{want} declared: {decls:?}");
    }
    assert!(!decls.contains(&"__multi3"));
}

/// The compiled object calls libgcc for division and the float conversions,
/// but never for multiplication, which is inline (`mul` + two `imul`s).
#[test]
fn multiply_is_inline_and_division_calls_libgcc() {
    let (m, syms) = parse(&suite_src());
    let obj: ObjectModule = super::compile_module(&m, &syms);
    let called: Vec<&str> = obj
        .relocations()
        .iter()
        .filter(|r| r.kind == RelocKind::Plt32)
        .map(|r| obj.symbol(r.symbol).name.as_str())
        .collect();
    for want in ["__udivti3", "__divti3", "__umodti3", "__modti3", "__floattidf", "__fixunssfti"] {
        assert!(called.contains(&want), "{want}: {called:?}");
    }
    assert!(!called.iter().any(|n| n.contains("mul")), "{called:?}");
    // `mul r/m64` (REX.W F7 /4) appears in the multiply.
    let text = &obj.sections().iter().find(|s| s.name == ".text").unwrap().bytes;
    assert!(text.windows(3).any(|w| (w[0] & 0xF8) == 0x48 && w[1] == 0xF7 && (w[2] & 0xF8) == 0xE0));
}

/// Integers wider than 128 bits have no register convention.
#[test]
fn wider_than_128_is_rejected_at_the_boundary() {
    let src = "module \"w\"\nfunc @f(i256) -> i256 {\nentry ^0(%a: i256):\n  ret %a\n}\n";
    let (m, syms) = parse(src);
    let r = std::panic::catch_unwind(|| super::compile_module(&m, &syms));
    let msg = r.expect_err("i256 is rejected");
    let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("wider than 128 bits"), "{msg}");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use super::*;
    use crate::ir::refexec::run_named;
    use crate::ir::semantics::SemValue;
    use crate::transform::pipeline::{OptLevel, optimize};
    use puremp::Int;
    use std::fmt::Write as _;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    /// A small deterministic generator (xorshift64*), so the suite needs no
    /// dependency and every run checks the same cases.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        /// A 128-bit operand: an edge value, a small (possibly negative) one, a
        /// value straddling the 64-bit boundary, or random bits.
        fn operand(&mut self) -> u128 {
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

    fn have_gcc() -> bool {
        Command::new("gcc").arg("--version").output().is_ok_and(|o| o.status.success())
    }

    fn scratch(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("lf-i128-{name}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn run(cmd: &mut Command) -> Output {
        loop {
            match cmd.output() {
                Ok(o) => return o,
                Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
                Err(e) => panic!("exec: {e}"),
            }
        }
    }

    /// Compile `m` to `dir/<name>.o`.
    fn object(dir: &Path, name: &str, m: &Module, syms: &StrInterner) -> PathBuf {
        let p = dir.join(format!("{name}.o"));
        std::fs::write(&p, crate::mc::elf::write(&super::super::compile_module(m, syms))).unwrap();
        p
    }

    /// Build `c_src` with gcc against `objs` (and libgcc's `__int128`
    /// helpers), run it, and return its output (asserting success).
    fn gcc_run(dir: &Path, c_src: &str, objs: &[PathBuf]) -> String {
        let c = dir.join("main.c");
        std::fs::write(&c, c_src).unwrap();
        let exe = dir.join("main");
        let out = Command::new("gcc").arg("-O1").arg(&c).args(objs).arg("-o").arg(&exe).output().unwrap();
        assert!(out.status.success(), "gcc: {}", String::from_utf8_lossy(&out.stderr));
        let out = run(&mut Command::new(&exe));
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success(), "{stdout}{}", String::from_utf8_lossy(&out.stderr));
        stdout
    }

    fn sem(v: u128) -> SemValue {
        SemValue::int(128, Int::from_u128(v))
    }

    /// Whether the reference semantics is defined for `op(a, b)` (division by
    /// zero and `MIN / -1` are undefined behavior).
    fn defined(op: &str, a: u128, b: u128) -> bool {
        match op {
            "udiv" | "urem" => b != 0,
            "sdiv" | "srem" => b != 0 && !(a == 1 << 127 && b == u128::MAX),
            _ => true,
        }
    }

    /// Every operation on random operands: the reference evaluator's result
    /// is the expected value the native code (compiled at `-O0`, and at `-O2`
    /// after the optimizer) must reproduce, called from C through the System V
    /// `__int128` convention.
    #[test]
    fn i128_ops_match_the_reference_evaluator() {
        if !have_gcc() {
            eprintln!("skipping: gcc not found");
            return;
        }
        let (m, syms) = parse(&suite_src());
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let mut cases = String::new();
        let mut count = 0;
        for (k, (name, _)) in OPS.iter().enumerate() {
            let mut done = 0;
            while done < 24 {
                let (a, b) = (rng.operand(), rng.operand());
                if !defined(name, a, b) {
                    continue;
                }
                let want = run_named(&m, &syms, &format!("t_{name}"), &[sem(a), sem(b)])
                    .unwrap_or_else(|e| panic!("reference t_{name}({a:#x}, {b:#x}): {e:?}"));
                let Some(SemValue::Int { bits, .. }) = want else { panic!("t_{name}: {want:?}") };
                let w = bits.to_u128().unwrap();
                let h = |v: u128| format!("{:#x}ULL, {:#x}ULL", v as u64, (v >> 64) as u64);
                writeln!(cases, "    {{ {k}, {}, {}, {} }},", h(a), h(b), h(w)).unwrap();
                done += 1;
                count += 1;
            }
        }
        let mut decls = String::new();
        let mut table = String::new();
        for (name, _) in OPS {
            writeln!(decls, "__int128 t_{name}(__int128, __int128);").unwrap();
            writeln!(table, "    t_{name},").unwrap();
        }
        let names: Vec<String> = OPS.iter().map(|(n, _)| format!("\"{n}\"")).collect();
        let c = format!(
            r#"#include <stdio.h>
#include <stdint.h>
{decls}
typedef __int128 (*fn)(__int128, __int128);
static const fn fns[] = {{
{table}}};
static const char *names[] = {{ {names} }};
struct case_ {{ int op; uint64_t alo, ahi, blo, bhi, wlo, whi; }};
static const struct case_ cases[] = {{
{cases}}};
static __int128 mk(uint64_t lo, uint64_t hi) {{ return (__int128)(((unsigned __int128)hi << 64) | lo); }}
int main(void) {{
    int bad = 0;
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {{
        const struct case_ *c = &cases[i];
        unsigned __int128 got = fns[c->op](mk(c->alo, c->ahi), mk(c->blo, c->bhi));
        if ((uint64_t)got != c->wlo || (uint64_t)(got >> 64) != c->whi) {{
            if (bad++ < 20)
                printf("%s(%#llx:%#llx, %#llx:%#llx) = %#llx:%#llx, want %#llx:%#llx\n", names[c->op],
                       (unsigned long long)c->ahi, (unsigned long long)c->alo, (unsigned long long)c->bhi,
                       (unsigned long long)c->blo, (unsigned long long)(got >> 64), (unsigned long long)got,
                       (unsigned long long)c->whi, (unsigned long long)c->wlo);
        }}
    }}
    printf("%d mismatches of %u\n", bad, (unsigned)(sizeof cases / sizeof cases[0]));
    return bad != 0;
}}
"#,
            names = names.join(", ")
        );
        assert!(count > 1000);
        for level in [OptLevel::O0, OptLevel::O2] {
            let dir = scratch("ops");
            let mut mm = m.clone();
            optimize(&mut mm, level);
            crate::verify::verify_module(&mm).unwrap_or_else(|d| panic!("{level:?} verify: {d:#?}"));
            let obj = object(&dir, "wide", &mm, &syms);
            let out = gcc_run(&dir, &c, &[obj]);
            eprintln!("{level:?}: {}", out.trim());
            assert!(out.contains("0 mismatches"), "{level:?}: {out}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// System V `__int128` passing both ways with gcc: arguments in register
    /// pairs and, once fewer than two registers are left, in 16-byte-aligned
    /// stack slots (with a later `i64` still taking the last register), and
    /// results in `rax:rdx`.
    const ABI_LF: &str = r#"module "abi"
func @c_mix(i64, i128, i64, i128, i128, i64, i128) -> i128
func @c_late(i64, i64, i64, i64, i64, i128, i64) -> i128

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

func @lf_late(i64, i64, i64, i64, i64, i128, i64) -> i128 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64, %e: i64, %x: i128, %y: i64):
  %y2 = zext %y : i128
  %s = shl %y2, i128 64 : i128
  %t = add %x, %s : i128
  %e2 = sext %e : i128
  %r = sub %t, %e2 : i128
  ret %r
}

func @lf_calls_c(i128, i128) -> i128 {
entry ^0(%p: i128, %q: i128):
  %m = call @c_mix(i64 -7, %p, i64 9, %q, i128 -1, i64 5, i128 340282366920938463463374607431768211455) : i128
  %l = call @c_late(i64 1, i64 2, i64 3, i64 4, i64 5, %m, i64 -2) : i128
  %r = add %l, %p : i128
  ret %r
}
"#;

    const ABI_C: &str = r#"#include <stdio.h>
#include <stdint.h>
typedef __int128 i128;
typedef unsigned __int128 u128;

i128 lf_mix(int64_t, i128, int64_t, i128, i128, int64_t, i128);
i128 lf_late(int64_t, int64_t, int64_t, int64_t, int64_t, i128, int64_t);
i128 lf_calls_c(i128, i128);

/* The same computations in C, compiled by gcc. */
i128 c_mix(int64_t a, i128 b, int64_t c, i128 d, i128 e, int64_t f, i128 g) {
    return (i128)((u128)b * 3 - (u128)d ^ ((u128)e << 1)) + g + a + (i128)(uint64_t)c - f;
}
i128 c_late(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, i128 x, int64_t y) {
    (void)a; (void)b; (void)c; (void)d;
    return (i128)((u128)x + ((u128)(uint64_t)y << 64)) - e;
}

static i128 mk(uint64_t hi, uint64_t lo) { return (i128)(((u128)hi << 64) | lo); }

int main(void) {
    int bad = 0;
    i128 vals[] = { 0, 1, -1, mk(0x8000000000000000ull, 0), mk(0x0123456789abcdefull, 0xfedcba9876543210ull),
                    mk(0xffffffffull, 0xffffffffffffffffull), mk(0x7fffffffffffffffull, 0x1ull) };
    int n = sizeof vals / sizeof vals[0];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            i128 b = vals[i], d = vals[j], e = vals[(i + j) % n], g = vals[(i * 3 + j) % n];
            int64_t a = (int64_t)(i * 1000003 - j), c = (int64_t)(j * 77 - 5), f = (int64_t)i - 100;
            if (lf_mix(a, b, c, d, e, f, g) != c_mix(a, b, c, d, e, f, g)) bad |= 1;
            if (lf_late(1, 2, 3, 4, f, b, c) != c_late(1, 2, 3, 4, f, b, c)) bad |= 2;
            i128 m = c_mix(-7, b, 9, d, -1, 5, -1);
            if (lf_calls_c(b, d) != c_late(1, 2, 3, 4, 5, m, -2) + b) bad |= 4;
        }
    printf("bad=%d\n", bad);
    return bad;
}
"#;

    #[test]
    fn i128_abi_round_trips_with_gcc() {
        if !have_gcc() {
            eprintln!("skipping: gcc not found");
            return;
        }
        for level in [OptLevel::O0, OptLevel::O2] {
            let (mut m, syms) = parse(ABI_LF);
            optimize(&mut m, level);
            let dir = scratch("abi");
            let obj = object(&dir, "abi", &m, &syms);
            let out = gcc_run(&dir, ABI_C, &[obj]);
            assert!(out.contains("bad=0"), "{level:?}: {out}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
