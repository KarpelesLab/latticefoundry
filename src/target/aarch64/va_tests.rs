//! Variadic functions under AAPCS64 (Linux) and Darwin arm64.
//!
//! The callees implement `<stdarg.h>` as a C front end would, in IR over the
//! AAPCS64 `va_list` and the two frame-address hooks (see the `isel` module
//! docs): `va_start` fills `__stack`, `__gr_top`, `__vr_top`, `__gr_offs`
//! and `__vr_offs`; `va_arg` takes the next general or SIMD/FP register from
//! the save area while its offset is negative and otherwise walks `__stack`;
//! `va_copy` copies the 32-byte struct. Each program is checked three ways:
//!
//! - on the MIR interpreter (isel semantics: the save area, `__stack` past the
//!   named stack arguments, the caller's stack arguments);
//! - compiled, linked by `qld` into a static executable and run on the A64
//!   emulator (`super::emu`), so the prologue's register saves, the frame
//!   layout and the encodings are exercised for real;
//! - linked against C compiled by `clang --target=aarch64-linux-gnu` (when
//!   it is installed): clang's callers call ours and ours call clang's, so
//!   both sides of the ABI meet an independent implementation.

use std::fmt::Write as _;

use super::emu;
use super::interp;
use super::isel::{A64Op, AArch64Target};
use crate::codegen::CodegenOptions;
use crate::ir::{FuncId, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::target::TargetOs;

/// The register class a `va_arg` reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// `long`: a general register (`__gr_offs`, `__gr_top`).
    Gr,
    /// `double`: a SIMD/FP register (`__vr_offs`, `__vr_top`).
    Vr,
}

/// A small IR text generator with fresh block numbers.
struct Gen {
    text: String,
    next_block: u32,
}

impl Gen {
    fn new() -> Gen {
        Gen { text: String::new(), next_block: 10 }
    }

    fn block(&mut self) -> u32 {
        self.next_block += 1;
        self.next_block
    }

    fn line(&mut self, s: &str) {
        self.text.push_str(s);
        self.text.push('\n');
    }

    /// `va_start(ap)` for a function whose named arguments took `gr` general
    /// and `vr` SIMD/FP registers.
    fn va_start(&mut self, ap: &str, gr: i64, vr: i64) {
        let gr_offs = -(8 - gr) * 8;
        let vr_offs = -(8 - vr) * 16;
        let _ = write!(
            self.text,
            "  %{ap}_rsa = call @__lf_va_reg_save_area() : ptr
  %{ap}_stk = call @__lf_va_overflow_area() : ptr
  store %{ap}_stk, %{ap} align 8 : ptr
  %{ap}_gt = ptr_add %{ap}_rsa, i64 64 : ptr
  %{ap}_gta = ptr_add %{ap}, i64 8 : ptr
  store %{ap}_gt, %{ap}_gta align 8 : ptr
  %{ap}_vt = ptr_add %{ap}_rsa, i64 192 : ptr
  %{ap}_vta = ptr_add %{ap}, i64 16 : ptr
  store %{ap}_vt, %{ap}_vta align 8 : ptr
  %{ap}_goa = ptr_add %{ap}, i64 24 : ptr
  store i32 {gr_offs}, %{ap}_goa align 4 : i32
  %{ap}_voa = ptr_add %{ap}, i64 28 : ptr
  store i32 {vr_offs}, %{ap}_voa align 4 : i32
"
        );
    }

    /// `va_copy(dst, src)`: the 32-byte struct.
    fn va_copy(&mut self, dst: &str, src: &str) {
        for k in 0..4 {
            let _ = write!(
                self.text,
                "  %cp{k}s = ptr_add %{src}, i64 {o} : ptr
  %cp{k}d = ptr_add %{dst}, i64 {o} : ptr
  %cp{k} = load %cp{k}s align 8 : i64
  store %cp{k}, %cp{k}d align 8 : i64
",
                o = 8 * k
            );
        }
    }

    /// `%dst = va_arg(ap, long | double)`, the AAPCS64 algorithm: if the
    /// offset is already non-negative, or becomes positive once advanced, the
    /// argument is on the stack (`__stack`, 8-byte slots); else it is at
    /// `top + offset`. Leaves a new current block where `%dst` is defined.
    fn va_arg(&mut self, ap: &str, class: Class, dst: &str) {
        let (offs_at, top_at, step, ty) = match class {
            Class::Gr => (24, 8, 8, "i64"),
            Class::Vr => (28, 16, 16, "f64"),
        };
        let (r0, r1, s, m) = (self.block(), self.block(), self.block(), self.block());
        let _ = write!(
            self.text,
            "  %{dst}_po = ptr_add %{ap}, i64 {offs_at} : ptr
  %{dst}_o = load %{dst}_po align 4 : i32
  %{dst}_ge = icmp sge %{dst}_o, i32 0 : i1
  cond_br %{dst}_ge, ^{s}, ^{r0}
^{r0}:
  %{dst}_n = add %{dst}_o, i32 {step} : i32
  store %{dst}_n, %{dst}_po align 4 : i32
  %{dst}_past = icmp sgt %{dst}_n, i32 0 : i1
  cond_br %{dst}_past, ^{s}, ^{r1}
^{r1}:
  %{dst}_pt = ptr_add %{ap}, i64 {top_at} : ptr
  %{dst}_top = load %{dst}_pt align 8 : ptr
  %{dst}_ox = sext %{dst}_o : i64
  %{dst}_ra = ptr_add %{dst}_top, %{dst}_ox : ptr
  %{dst}_rv = load %{dst}_ra align 8 : {ty}
  br ^{m}(%{dst}_rv)
^{s}:
  %{dst}_sp = load %{ap} align 8 : ptr
  %{dst}_sn = ptr_add %{dst}_sp, i64 8 : ptr
  store %{dst}_sn, %{ap} align 8 : ptr
  %{dst}_sv = load %{dst}_sp align 8 : {ty}
  br ^{m}(%{dst}_sv)
^{m}(%{dst}: {ty}):
"
        );
    }

    /// Darwin's `va_arg`: `va_list` is a pointer to the next 8-byte stack slot.
    fn va_arg_darwin(&mut self, ap: &str, ty: &str, dst: &str) {
        let _ = write!(
            self.text,
            "  %{dst}_p = load %{ap} align 8 : ptr
  %{dst}_n = ptr_add %{dst}_p, i64 8 : ptr
  store %{dst}_n, %{ap} align 8 : ptr
  %{dst} = load %{dst}_p align 8 : {ty}
"
        );
    }
}

/// The `va_list` struct type (`__stack`, `__gr_top`, `__vr_top`, `__gr_offs`,
/// `__vr_offs`: 32 bytes, 8-aligned).
const VA_LIST: &str = "{ptr, ptr, ptr, i32, i32}";

/// A callee summing `n` variadic values whose classes follow `pattern`
/// (cyclically), as `f64` when any is a double, else as `i64`. `named_fp`
/// adds a leading named `f64` parameter (added to the sum), so the SIMD/FP
/// save area starts past it. With `copy`, the list is walked twice, the
/// second time through a `va_copy` made right after `va_start`.
fn sum_callee(name: &str, pattern: &[Class], named_fp: bool, copy: bool) -> String {
    let float = pattern.contains(&Class::Vr) || named_fp;
    let acc_ty = if float { "f64" } else { "i64" };
    let zero = if float { "f64 0x0" } else { "i64 0" };
    let mut g = Gen::new();
    let params = if named_fp { "f64, i32, ..." } else { "i32, ..." };
    let entry_params = if named_fp { "%base: f64, %n: i32" } else { "%n: i32" };
    g.line(&format!("func @{name}({params}) -> {acc_ty} {{"));
    g.line(&format!("entry ^0({entry_params}):"));
    g.line(&format!("  %ap = alloca {VA_LIST} : ptr"));
    g.line(&format!("  %ap2 = alloca {VA_LIST} : ptr"));
    g.va_start("ap", 1, i64::from(named_fp));
    g.va_copy("ap2", "ap");
    let start = if named_fp { "%base" } else { zero };
    let walks: &[&str] = if copy { &["ap", "ap2"] } else { &["ap"] };
    let mut acc_in = start.to_owned();
    for (w, ap) in walks.iter().enumerate() {
        let (head, body, done) = (g.block(), g.block(), g.block());
        g.line(&format!("  br ^{head}({acc_in}, i32 0)"));
        g.line(&format!("^{head}(%acc{w}: {acc_ty}, %i{w}: i32):"));
        g.line(&format!("  %more{w} = icmp slt %i{w}, %n : i1"));
        g.line(&format!("  cond_br %more{w}, ^{body}, ^{done}(%acc{w})"));
        g.line(&format!("^{body}:"));
        // One value per pattern position: dispatch on i mod len.
        let len = pattern.len() as i64;
        let k = format!("%k{w}");
        g.line(&format!("  {k} = srem %i{w}, i32 {len} : i32"));
        let join = g.block();
        let arms: Vec<u32> = pattern.iter().map(|_| g.block()).collect();
        let cases: Vec<String> = arms.iter().enumerate().map(|(c, b)| format!("{c}: ^{b}")).collect();
        g.line(&format!("  switch {k}, ^{} [{}]", arms[0], cases.join(", ")));
        for (c, (&b, &class)) in arms.iter().zip(pattern).enumerate() {
            g.line(&format!("^{b}:"));
            let v = format!("v{w}_{c}");
            g.va_arg(ap, class, &v);
            let val = match (class, float) {
                (Class::Vr, _) | (Class::Gr, false) => format!("%{v}"),
                (Class::Gr, true) => {
                    g.line(&format!("  %{v}f = sitofp %{v} : f64"));
                    format!("%{v}f")
                }
            };
            g.line(&format!("  br ^{join}({val})"));
        }
        g.line(&format!("^{join}(%x{w}: {acc_ty}):"));
        let add = if float { "fadd" } else { "add" };
        g.line(&format!("  %acc{w}n = {add} %acc{w}, %x{w} : {acc_ty}"));
        g.line(&format!("  %i{w}n = add %i{w}, i32 1 : i32"));
        g.line(&format!("  br ^{head}(%acc{w}n, %i{w}n)"));
        g.line(&format!("^{done}(%res{w}: {acc_ty}):"));
        acc_in = format!("%res{w}");
    }
    g.line(&format!("  ret {acc_in}"));
    g.line("}");
    g.text
}

/// `f64 <bits>` for an IR constant.
fn fc(x: f64) -> String {
    format!("f64 {:#x}", x.to_bits())
}

/// One check of `main`: call `callee(args)`, compare with `want`, and fold a
/// failure into bit `bit` of the result.
struct Check {
    callee: &'static str,
    args: Vec<String>,
    want: String,
    ty: &'static str,
}

/// `int main(void)` running every check, returning the failure bitmask.
fn main_func(checks: &[Check]) -> String {
    let mut s = String::from("func @main() -> i32 {\nentry ^0:\n");
    let mut acc = "i32 0".to_owned();
    for (k, c) in checks.iter().enumerate() {
        let ty = c.ty;
        let _ = writeln!(s, "  %r{k} = call @{}({}) : {ty}", c.callee, c.args.join(", "));
        if ty == "f64" {
            let _ = writeln!(s, "  %bad{k} = fcmp une %r{k}, {} : i1", c.want);
        } else {
            let _ = writeln!(s, "  %bad{k} = icmp ne %r{k}, {} : i1", c.want);
        }
        let _ = writeln!(s, "  %z{k} = zext %bad{k} : i32");
        let _ = writeln!(s, "  %s{k} = shl %z{k}, i32 {k} : i32");
        let _ = writeln!(s, "  %a{k} = or {acc}, %s{k} : i32");
        acc = format!("%a{k}");
    }
    let _ = writeln!(s, "  ret {acc}\n}}");
    s
}

/// The variadic callees.
fn callees() -> String {
    use Class::{Gr, Vr};
    [
        sum_callee("isum", &[Gr], false, false),
        sum_callee("dsum", &[Vr], false, false),
        sum_callee("mixed", &[Gr, Vr], false, false),
        sum_callee("fmixed", &[Vr, Vr, Gr], true, false),
        sum_callee("copysum", &[Gr], false, true),
    ]
    .concat()
}

/// The checks `main` runs against the callees named `prefix`-something
/// (`""` for ours, `"c_"` for clang's): with up to 20 anonymous arguments,
/// so both register banks overflow onto the stack.
fn checks(prefix: &'static str) -> Vec<Check> {
    let name = |s: &str| -> &'static str { Box::leak(format!("{prefix}{s}").into_boxed_str()) };
    let ints = |n: i64| (1..=n).map(|k| format!("i64 {}", k * 3 - 7)).collect::<Vec<_>>();
    let isum = |n: i64| (1..=n).map(|k| k * 3 - 7).sum::<i64>();
    let dbls = |n: i64| (1..=n).map(|k| fc(k as f64 * 1.5)).collect::<Vec<_>>();
    let dsum = |n: i64| (1..=n).map(|k| k as f64 * 1.5).sum::<f64>();
    let mut out = Vec::new();
    for n in [0i64, 3, 7, 12] {
        let mut args = vec![format!("i32 {n}")];
        args.extend(ints(n));
        out.push(Check { callee: name("isum"), args, want: format!("i64 {}", isum(n)), ty: "i64" });
    }
    for n in [1i64, 8, 11] {
        let mut args = vec![format!("i32 {n}")];
        args.extend(dbls(n));
        out.push(Check { callee: name("dsum"), args, want: fc(dsum(n)), ty: "f64" });
    }
    // mixed: i64, f64 alternating, 20 values (10 of each: both banks spill).
    let mut args = vec!["i32 20".to_owned()];
    let mut want = 0.0;
    for k in 0..20i64 {
        if k % 2 == 0 {
            args.push(format!("i64 {}", k - 5));
            want += (k - 5) as f64;
        } else {
            args.push(fc(k as f64 * 0.25));
            want += k as f64 * 0.25;
        }
    }
    out.push(Check { callee: name("mixed"), args, want: fc(want), ty: "f64" });
    // fmixed: a named double first, then f64, f64, i64, ... (15 values).
    let mut args = vec![fc(100.5), "i32 15".to_owned()];
    let mut want = 100.5;
    for k in 0..15i64 {
        if k % 3 == 2 {
            args.push(format!("i64 {k}"));
            want += k as f64;
        } else {
            args.push(fc(k as f64 + 0.5));
            want += k as f64 + 0.5;
        }
    }
    out.push(Check { callee: name("fmixed"), args, want: fc(want), ty: "f64" });
    // va_copy: the list walked twice.
    let mut args = vec!["i32 10".to_owned()];
    args.extend(ints(10));
    out.push(Check { callee: name("copysum"), args, want: format!("i64 {}", 2 * isum(10)), ty: "i64" });
    out
}

const HOOKS: &str = "func @__lf_va_reg_save_area() -> ptr\nfunc @__lf_va_overflow_area() -> ptr\n";

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    (m, syms)
}

/// The whole program: callees, hooks, `main` over our callees.
fn program() -> String {
    format!("module \"va\"\n{HOOKS}{}{}", callees(), main_func(&checks("")))
}

/// Run `main` of `src` on the MIR interpreter under `target`.
fn interp_main(src: &str, target: &AArch64Target) -> u64 {
    let (m, syms) = parse(src);
    let funcs: Vec<_> =
        (0..m.functions().count()).map(|i| target.select_with_syms(&m, FuncId::from_index(i), &syms)).collect();
    let main = m.functions().position(|f| syms.resolve(f.name) == "main").unwrap();
    let v = interp::run(target, &funcs, main, &[]).expect("interpretation succeeds").expect("a result");
    v.to_u64().unwrap()
}

#[test]
fn variadic_sums_on_the_interpreter() {
    assert_eq!(interp_main(&program(), &AArch64Target::new()), 0, "failing checks (bitmask)");
}

/// Compile `src`, link it with `extra` objects into a static executable, and
/// run it on the emulator: the exit status.
fn run_linked(src: &str, extra: &[std::path::PathBuf], tag: &str) -> u64 {
    let (m, syms) = parse(src);
    let obj = super::compile_module_with(&m, &syms, &CodegenOptions::default()).object;
    let dir = std::env::temp_dir().join(format!("lf-a64-va-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("a.out");
    let extra: Vec<String> = extra.iter().map(|p| p.display().to_string()).collect();
    super::link::link_executable(vec![obj], "main", &extra, &exe).expect("qld links the program");
    let elf = std::fs::read(&exe).unwrap();
    let (code, _) = emu::run_executable(&elf).unwrap_or_else(|e| panic!("{tag}: {e}"));
    let _ = std::fs::remove_dir_all(&dir);
    code
}

#[test]
fn variadic_sums_run_linked() {
    assert_eq!(run_linked(&program(), &[], "own"), 0, "failing checks (bitmask)");
}

/// The C side of the cross-check: the same callees (named `c_*`), and a
/// `cmain` calling ours with the same arguments.
const C_SRC: &str = r#"
#include <stdarg.h>
long c_isum(int n, ...) {
    va_list ap; va_start(ap, n); long s = 0;
    for (int i = 0; i < n; i++) s += va_arg(ap, long);
    va_end(ap); return s;
}
double c_dsum(int n, ...) {
    va_list ap; va_start(ap, n); double s = 0;
    for (int i = 0; i < n; i++) s += va_arg(ap, double);
    va_end(ap); return s;
}
double c_mixed(int n, ...) {
    va_list ap; va_start(ap, n); double s = 0;
    for (int i = 0; i < n; i++) s += (i % 2 == 0) ? (double)va_arg(ap, long) : va_arg(ap, double);
    va_end(ap); return s;
}
double c_fmixed(double base, int n, ...) {
    va_list ap; va_start(ap, n); double s = base;
    for (int i = 0; i < n; i++) s += (i % 3 == 2) ? (double)va_arg(ap, long) : va_arg(ap, double);
    va_end(ap); return s;
}
long c_copysum(int n, ...) {
    va_list ap, ap2; va_start(ap, n); va_copy(ap2, ap); long s = 0;
    for (int i = 0; i < n; i++) s += va_arg(ap, long);
    for (int i = 0; i < n; i++) s += va_arg(ap2, long);
    va_end(ap2); va_end(ap); return s;
}
long isum(int n, ...); double dsum(int n, ...); double mixed(int n, ...);
double fmixed(double base, int n, ...); long copysum(int n, ...);
int cmain(void) {
    int bad = 0;
    if (isum(0) != 0) bad |= 1;
    if (isum(12, -4L, -1L, 2L, 5L, 8L, 11L, 14L, 17L, 20L, 23L, 26L, 29L) != 150) bad |= 2;
    if (dsum(11, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5) != 99.0) bad |= 4;
    if (mixed(6, 1L, 0.5, 2L, 0.25, 3L, 8.0) != 14.75) bad |= 8;
    if (fmixed(1.0, 4, 0.5, 1.5, 7L, 2.0) != 12.0) bad |= 16;
    if (copysum(9, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L) != 90) bad |= 32;
    return bad;
}
"#;

/// Compile [`C_SRC`] with clang for AArch64 Linux, if clang can.
fn clang_object(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    std::fs::create_dir_all(dir).ok()?;
    let src = dir.join("c.c");
    std::fs::write(&src, C_SRC).ok()?;
    let obj = dir.join("c.o");
    let out = std::process::Command::new("clang")
        .args(["--target=aarch64-linux-gnu", "-ffreestanding", "-fno-stack-protector", "-O1", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .output()
        .ok()?;
    out.status.success().then_some(obj)
}

#[test]
fn variadic_abi_matches_clang() {
    let dir = std::env::temp_dir().join(format!("lf-a64-va-clang-{}", std::process::id()));
    let Some(obj) = clang_object(&dir) else {
        eprintln!("skipping variadic_abi_matches_clang: clang cannot target aarch64-linux-gnu");
        return;
    };
    // Our main calls clang's callees, then clang's `cmain` calls ours.
    let mut checks = checks("c_");
    checks.push(Check { callee: "cmain", args: Vec::new(), want: "i32 0".into(), ty: "i32" });
    let decls = "func @c_isum(i32, ...) -> i64\nfunc @c_dsum(i32, ...) -> f64\nfunc @c_mixed(i32, ...) -> f64\n\
                 func @c_fmixed(f64, i32, ...) -> f64\nfunc @c_copysum(i32, ...) -> i64\nfunc @cmain() -> i32\n";
    let src = format!("module \"va\"\n{HOOKS}{decls}{}{}", callees(), main_func(&checks));
    assert_eq!(run_linked(&src, &[obj], "clang"), 0, "failing checks (bitmask)");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn variadic_prologue_saves_the_argument_registers() {
    // The save area: x0..x7 at 0..56 and the whole q0..q7 at 64..176, and
    // the hooks resolve to frame addresses, not calls.
    let (m, syms) = parse(&format!("module \"va\"\n{HOOKS}{}", sum_callee("isum", &[Class::Gr], false, false)));
    let f = m.functions().position(|f| syms.resolve(f.name) == "isum").unwrap();
    let mf = AArch64Target::new().select_with_syms(&m, FuncId::from_index(f), &syms);
    let ops: Vec<A64Op> = mf.block_ids().flat_map(|b| mf.block(b).insts.iter()).map(|i| A64Op::decode(i.opcode)).collect();
    assert_eq!(ops.iter().filter(|&&o| o == A64Op::NeonStore).count(), 8, "q0..q7 saved");
    assert!(ops.iter().filter(|&&o| o == A64Op::Store).count() >= 8, "x0..x7 saved");
    assert!(!ops.contains(&A64Op::Call), "the hooks are not calls");
    assert!(ops.contains(&A64Op::LeaFpOff), "__stack is x29-relative");
    // A non-variadic function saves nothing.
    let (m, syms) = parse("module \"f\"\nfunc @f(i64) -> i64 {\nentry ^0(%x: i64):\n  ret %x\n}\n");
    let mf = AArch64Target::new().select_with_syms(&m, FuncId::from_index(0), &syms);
    assert!(mf.block_ids().flat_map(|b| mf.block(b).insts.iter()).all(|i| A64Op::decode(i.opcode) != A64Op::NeonStore));
}

// ---------------------------------------------------------------------------
// Darwin arm64: anonymous arguments on the stack, `va_list` a plain pointer.
// ---------------------------------------------------------------------------

/// A Darwin callee summing `n` `i64`s.
fn darwin_isum() -> String {
    let mut g = Gen::new();
    g.line("func @isum(i32, ...) -> i64 {");
    g.line("entry ^0(%n: i32):");
    g.line("  %ap = alloca ptr : ptr");
    g.line("  %stk = call @__lf_va_overflow_area() : ptr");
    g.line("  store %stk, %ap align 8 : ptr");
    let (head, body, done) = (g.block(), g.block(), g.block());
    g.line(&format!("  br ^{head}(i64 0, i32 0)"));
    g.line(&format!("^{head}(%acc: i64, %i: i32):"));
    g.line("  %more = icmp slt %i, %n : i1");
    g.line(&format!("  cond_br %more, ^{body}, ^{done}(%acc)"));
    g.line(&format!("^{body}:"));
    g.va_arg_darwin("ap", "i64", "v");
    g.line("  %acc2 = add %acc, %v : i64");
    g.line("  %i2 = add %i, i32 1 : i32");
    g.line(&format!("  br ^{head}(%acc2, %i2)"));
    g.line(&format!("^{done}(%r: i64):"));
    g.line("  ret %r");
    g.line("}");
    g.text
}

#[test]
fn darwin_passes_anonymous_arguments_on_the_stack() {
    let src = format!(
        "module \"va\"\n{HOOKS}{}func @main() -> i64 {{\nentry ^0:\n  %r = call @isum(i32 3, i64 10, i64 20, i64 12) : i64\n  ret %r\n}}\n",
        darwin_isum()
    );
    let darwin = AArch64Target::for_os(TargetOs::Darwin);
    assert_eq!(interp_main(&src, &darwin), 42);
    // The caller puts only the named `n` in a register: the three anonymous
    // values go to the outgoing area.
    let (m, syms) = parse(&src);
    let main = m.functions().position(|f| syms.resolve(f.name) == "main").unwrap();
    let mf = darwin.select_with_syms(&m, FuncId::from_index(main), &syms);
    let call = mf.block_ids().flat_map(|b| mf.block(b).insts.iter()).find(|i| A64Op::decode(i.opcode) == A64Op::Call).unwrap();
    let arg_uses = call.uses().count();
    assert_eq!(arg_uses, 1, "only x0 carries an argument: {call:?}");
    assert_eq!(mf.frame().outgoing(), 32, "three 8-byte slots, rounded to 16");
    // The same caller under Linux passes all four in registers.
    let mf = AArch64Target::new().select_with_syms(&m, FuncId::from_index(main), &syms);
    let call = mf.block_ids().flat_map(|b| mf.block(b).insts.iter()).find(|i| A64Op::decode(i.opcode) == A64Op::Call).unwrap();
    assert_eq!(call.uses().count(), 4);
    assert_eq!(mf.frame().outgoing(), 0);
}
