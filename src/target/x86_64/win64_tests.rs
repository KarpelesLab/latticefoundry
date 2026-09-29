//! Execution tests for the Microsoft x64 ("Win64") calling convention.
//!
//! Windows binaries cannot run here, but the convention can: code compiled
//! for `x86_64-windows` is packaged as an ELF object and linked into a Linux
//! program whose other side also speaks Win64 — either gcc's own
//! `__attribute__((ms_abi))` implementation (an independent reference for
//! argument placement, aggregates and variadics), or a hand-written assembly
//! harness that plants sentinels in every Win64 callee-saved register
//! (including the full 128 bits of `xmm6..xmm15`) and checks them after the
//! call. Skipped when no C compiler is present.

use crate::codegen::CodegenOptions;
use crate::ir::inst::Flags;
use crate::ir::Module;
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::target::TargetOs;

fn win64_object(module: &Module, syms: &StrInterner) -> Vec<u8> {
    let opts = CodegenOptions::default().with_os(TargetOs::Windows);
    let obj = super::compile_module_with(module, syms, &opts).object;
    crate::mc::elf::write(&obj)
}

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms).expect("parse .lf");
    (m, syms)
}

fn find_cc() -> Option<&'static str> {
    ["cc", "gcc"].into_iter().find(|cc| {
        std::process::Command::new(cc)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-win64-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run `exe`, retrying a transient ETXTBSY from a concurrent fork.
fn run(exe: &std::path::Path) -> i32 {
    loop {
        match std::process::Command::new(exe).status() {
            Ok(s) => return s.code().expect("exited via a signal"),
            Err(e) if e.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(e) => panic!("run {}: {e}", exe.display()),
        }
    }
}

/// Link our Win64 object with a C driver (gcc, `ms_abi` prototypes) and run it.
fn run_with_c(module: &Module, syms: &StrInterner, tag: &str, c: &str) -> Option<i32> {
    let cc = find_cc()?;
    let dir = scratch(tag);
    let (obj, src, exe) = (dir.join("w.o"), dir.join("main.c"), dir.join("t"));
    std::fs::write(&obj, win64_object(module, syms)).unwrap();
    std::fs::write(&src, c).unwrap();
    let status = std::process::Command::new(cc)
        .arg(&src)
        .arg(&obj)
        .arg("-o")
        .arg(&exe)
        .arg("-Wl,-z,noexecstack")
        .status()
        .expect("run cc");
    assert!(status.success(), "linking with {cc} failed");
    let code = run(&exe);
    let _ = std::fs::remove_dir_all(&dir);
    Some(code)
}

/// `win_entry(a..f)` keeps values live across calls (so they sit in
/// callee-saved registers), passes floats, and calls the external `ext7`
/// with five integer, one float and one more integer argument (three of
/// them on the stack). Returns 212 for `(1, 2, 3, 4, 5, 6)`.
const SCALARS: &str = r#"
module "w"
func @ext7(i64, i64, i64, i64, i64, f64, i64) -> i64
func @helper(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %t = mul %a, i64 3 : i64
  %r = add %t, %b : i64
  ret %r
}
func @fhelp(f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64):
  %r = fmul %a, %b : f64
  ret %r
}
func @win_entry(i64, i64, i64, i64, i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64, %e: i64, %f: i64):
  %x1 = call @helper(%a, %b) : i64
  %x2 = call @helper(%c, %d) : i64
  %x3 = call @helper(%e, %f) : i64
  %fa = sitofp %a : f64
  %fb = sitofp %b : f64
  %fc = sitofp %c : f64
  %p1 = call @fhelp(%fa, %fb) : f64
  %p2 = call @fhelp(%fc, %fb) : f64
  %fs = fadd %p1, %p2 : f64
  %fs2 = fadd %fs, %fa : f64
  %fs3 = fadd %fs2, %fc : f64
  %ff = fptosi %fs3 : i64
  %y = call @ext7(i64 1, i64 2, i64 3, i64 4, i64 5, f64 0x4018000000000000, i64 7) : i64
  %s1 = add %a, %b : i64
  %s2 = add %s1, %c : i64
  %s3 = add %s2, %d : i64
  %s4 = add %s3, %e : i64
  %s5 = add %s4, %f : i64
  %t1 = add %s5, %x1 : i64
  %t2 = add %t1, %x2 : i64
  %t3 = add %t2, %x3 : i64
  %t4 = add %t3, %ff : i64
  %t5 = add %t4, %y : i64
  ret %t5
}
func @mixf(i64, f64, i64, f64, f64, i64) -> f64 {
entry ^0(%a: i64, %b: f64, %c: i64, %d: f64, %e: f64, %f: i64):
  %fa = sitofp %a : f64
  %b2 = fadd %b, %b : f64
  %fc = sitofp %c : f64
  %c3 = fmul %fc, f64 0x4008000000000000 : f64
  %d4 = fmul %d, f64 0x4010000000000000 : f64
  %e5 = fmul %e, f64 0x4014000000000000 : f64
  %ff = sitofp %f : f64
  %f6 = fmul %ff, f64 0x4018000000000000 : f64
  %s1 = fadd %fa, %b2 : f64
  %s2 = fadd %s1, %c3 : f64
  %s3 = fadd %s2, %d4 : f64
  %s4 = fadd %s3, %e5 : f64
  %s5 = fadd %s4, %f6 : f64
  ret %s5
}
"#;

#[test]
fn win64_scalars_interoperate_with_gcc_ms_abi() {
    let (m, syms) = parse(SCALARS);
    let c = r#"
        #include <stdint.h>
        #define MS __attribute__((ms_abi))
        MS int64_t ext7(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, double f, int64_t g) {
            return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * (int64_t)f + 7 * g;
        }
        MS int64_t win_entry(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
        MS double mixf(int64_t, double, int64_t, double, double, int64_t);
        int main(void) {
            if (win_entry(1, 2, 3, 4, 5, 6) != 212) return 1;
            if (mixf(1, 2.5, 3, 4.5, 5.5, 6) != 96.5) return 2;
            return 0;
        }
    "#;
    match run_with_c(&m, &syms, "scalars", c) {
        Some(code) => assert_eq!(code, 0, "Win64 scalar interop with gcc ms_abi failed ({code})"),
        None => eprintln!("skipping: no C compiler"),
    }
}

/// The assembly harness: `_start` plants sentinels in every Win64
/// callee-saved register, calls `win_entry(1..6)` Win64-style (shadow space,
/// two stack arguments), checks the sentinels, and exits with the result;
/// `ext7` is a Win64 callee that writes its home slots (proving the caller
/// reserved them) and clobbers every volatile register.
fn harness() -> String {
    let gprs = [("rbx", 0x1111), ("rsi", 0x2222), ("rdi", 0x3333), ("rbp", 0x4444), ("r12", 0x5555), ("r13", 0x6666), ("r14", 0x7777), ("r15", 0x8888)];
    let mut s = String::from("        .text\n        .globl _start\n_start:\n");
    for (r, v) in gprs {
        s += &format!("        movq ${v}, %{r}\n");
    }
    for x in 6..=15 {
        // Low qword 0x600x, high qword 0x700x: all 128 bits are checked.
        s += &format!(
            "        movq ${lo}, %rax\n        movq %rax, %xmm{x}\n        movq ${hi}, %rax\n        movq %rax, %xmm0\n        punpcklqdq %xmm0, %xmm{x}\n",
            lo = 0x6000 + x,
            hi = 0x7000 + x
        );
    }
    s += "        subq $48, %rsp\n        movq $5, 32(%rsp)\n        movq $6, 40(%rsp)\n";
    s += "        movq $1, %rcx\n        movq $2, %rdx\n        movq $3, %r8\n        movq $4, %r9\n";
    s += "        call win_entry\n        addq $48, %rsp\n        movq %rax, %r10\n";
    let mut code = 200;
    for (r, v) in gprs {
        s += &format!("        cmpq ${v}, %{r}\n        movl ${code}, %r11d\n        jne bad\n");
        code += 1;
    }
    for x in 6..=15 {
        s += &format!(
            "        movdqu %xmm{x}, -16(%rsp)\n        movl ${code}, %r11d\n        cmpq ${lo}, -16(%rsp)\n        jne bad\n        cmpq ${hi}, -8(%rsp)\n        jne bad\n",
            lo = 0x6000 + x,
            hi = 0x7000 + x
        );
        code += 1;
    }
    s += "        movq %r10, %r11\nbad:\n        movq %r11, %rdi\n        movl $60, %eax\n        syscall\n";
    s += r#"
        .globl ext7
ext7:
        movq %rcx, 8(%rsp)
        movq %rdx, 16(%rsp)
        movq %r8, 24(%rsp)
        movq %r9, 32(%rsp)
        movq 8(%rsp), %rax
        movq 16(%rsp), %r10
        leaq (%rax,%r10,2), %rax
        imulq $3, 24(%rsp), %r10
        addq %r10, %rax
        imulq $4, 32(%rsp), %r10
        addq %r10, %rax
        imulq $5, 40(%rsp), %r10
        addq %r10, %rax
        cvttsd2si 48(%rsp), %r10
        imulq $6, %r10, %r10
        addq %r10, %rax
        imulq $7, 56(%rsp), %r10
        addq %r10, %rax
        movq $-1, %rcx
        movq $-1, %rdx
        movq $-1, %r8
        movq $-1, %r9
        movq $-1, %r10
        movq $-1, %r11
        pcmpeqd %xmm0, %xmm0
        pcmpeqd %xmm1, %xmm1
        pcmpeqd %xmm2, %xmm2
        pcmpeqd %xmm3, %xmm3
        pcmpeqd %xmm4, %xmm4
        pcmpeqd %xmm5, %xmm5
        ret
"#;
    s
}

#[test]
fn win64_preserves_callee_saved_registers() {
    use crate::mc::asm::{AsmOptions, AsmSource, assemble};
    use crate::target::TargetArch;
    let (m, syms) = parse(SCALARS);
    let asm = assemble(&[AsmSource { name: "harness.s", text: &harness() }], &AsmOptions::new(TargetArch::X86_64))
        .expect("assemble the harness");
    let dir = scratch("harness");
    let (a, w, exe) = (dir.join("h.o"), dir.join("w.o"), dir.join("t"));
    std::fs::write(&a, asm).unwrap();
    std::fs::write(&w, win64_object(&m, &syms)).unwrap();
    let args: Vec<std::ffi::OsString> =
        vec!["-static".into(), "-o".into(), exe.clone().into(), a.into(), w.into()];
    crate::link::gnu::link_gnu("test", &args).expect("link the harness");
    let code = run(&exe);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(code, 212, "exit 200+ names the clobbered callee-saved register");
}

/// Build the aggregate fixtures with the IR builder (a struct value is a
/// pointer to its storage at this level; see the System V struct tests).
fn build_structs() -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("s");
    let i32t = m.types_mut().int(32);
    let i64t = m.types_mut().int(64);
    let p = m.types_mut().struct_(vec![i32t, i32t]); // 8 bytes: by value
    let t = m.types_mut().struct_(vec![i64t, i64t, i64t]); // 24: by reference
    let v = m.types_mut().struct_(vec![i64t, i64t]); // 16: by reference / sret

    // P addP(P a, P b) = {a.x + b.x, a.y + b.y}
    let sig = m.types_mut().func(vec![p, p], p, false);
    let add_p = m.declare_function(syms.intern("addP"), sig);
    {
        let mut b = m.build(add_p);
        let e = b.create_entry_block();
        let (a, c) = (b.param(e, 0), b.param(e, 1));
        let r = b.alloca(p);
        for k in 0..2 {
            let ap = b.struct_field(a, p, k);
            let x = b.load(i32t, ap, 4);
            let cp = b.struct_field(c, p, k);
            let y = b.load(i32t, cp, 4);
            let s = b.add(x, y, Flags::NONE);
            let rp = b.struct_field(r, p, k);
            b.store(i32t, rp, s, 4);
        }
        b.ret(Some(r));
    }
    // i64 sumT(T t, i64 k) = t.a + t.b + t.c + k
    let sig = m.types_mut().func(vec![t, i64t], i64t, false);
    let sum_t = m.declare_function(syms.intern("sumT"), sig);
    {
        let mut b = m.build(sum_t);
        let e = b.create_entry_block();
        let (tv, k) = (b.param(e, 0), b.param(e, 1));
        let mut acc = k;
        for f in 0..3 {
            let fp = b.struct_field(tv, t, f);
            let x = b.load(i64t, fp, 8);
            acc = b.add(acc, x, Flags::NONE);
        }
        b.ret(Some(acc));
    }
    // V mkV(i64 x, i64 y) = {x, y}
    let sig = m.types_mut().func(vec![i64t, i64t], v, false);
    let mk_v = m.declare_function(syms.intern("mkV"), sig);
    {
        let mut b = m.build(mk_v);
        let e = b.create_entry_block();
        let (x, y) = (b.param(e, 0), b.param(e, 1));
        let r = b.alloca(v);
        let rx = b.struct_field(r, v, 0);
        b.store(i64t, rx, x, 8);
        let ry = b.struct_field(r, v, 1);
        b.store(i64t, ry, y, 8);
        b.ret(Some(r));
    }
    // External C (ms_abi): P cP(P); i64 cT(T, P); V cV(i64).
    let sig = m.types_mut().func(vec![p], p, false);
    let c_p = m.declare_function(syms.intern("cP"), sig);
    let sig = m.types_mut().func(vec![t, p], i64t, false);
    let c_t = m.declare_function(syms.intern("cT"), sig);
    let sig = m.types_mut().func(vec![i64t], v, false);
    let c_v = m.declare_function(syms.intern("cV"), sig);
    // i64 useC(P p, T t, i64 x) = cT(t, cP(p)) + cV(x).x + cV(x).y
    // (struct arguments are struct-typed values: parameters or call results).
    let sig = m.types_mut().func(vec![p, t, i64t], i64t, false);
    let use_c = m.declare_function(syms.intern("useC"), sig);
    {
        let mut b = m.build(use_c);
        let e = b.create_entry_block();
        let (pa, ta, x) = (b.param(e, 0), b.param(e, 1), b.param(e, 2));
        let cpf = b.func_ref(c_p);
        let pr = b.call(cpf, &[pa], p).unwrap();
        let ctf = b.func_ref(c_t);
        let r1 = b.call(ctf, &[ta, pr], i64t).unwrap();
        let cvf = b.func_ref(c_v);
        let vr = b.call(cvf, &[x], v).unwrap();
        let vx_p = b.struct_field(vr, v, 0);
        let vx = b.load(i64t, vx_p, 8);
        let vy_p = b.struct_field(vr, v, 1);
        let vy = b.load(i64t, vy_p, 8);
        let s = b.add(r1, vx, Flags::NONE);
        let s = b.add(s, vy, Flags::NONE);
        b.ret(Some(s));
    }
    (m, syms)
}

#[test]
fn win64_aggregates_interoperate_with_gcc_ms_abi() {
    let (m, syms) = build_structs();
    let c = r#"
        #define MS __attribute__((ms_abi))
        typedef struct { int x, y; } P;
        typedef struct { long long a, b, c; } T;
        typedef struct { long long x, y; } V;
        MS P addP(P, P);
        MS long long sumT(T, long long);
        MS V mkV(long long, long long);
        MS long long useC(P, T, long long);
        MS P cP(P a) { P r = { a.x * 10, a.y * 10 }; return r; }
        MS long long cT(T t, P p) { return t.a + t.b + t.c + p.x + p.y; }
        MS V cV(long long k) { V v = { k, k * 2 }; return v; }
        int main(void) {
            P r = addP((P){3, 4}, (P){10, 20});
            if (r.x != 13 || r.y != 24) return 1;
            T t = { 100, 20, 3 };
            if (sumT(t, 5) != 128) return 2;
            if (t.a != 100) return 3;
            V v = mkV(7, 9);
            if (v.x != 7 || v.y != 9) return 4;
            if (useC((P){2, 3}, (T){2, 2, 2}, 2) != 62) return 5;
            return 0;
        }
    "#;
    match run_with_c(&m, &syms, "structs", c) {
        Some(code) => assert_eq!(code, 0, "Win64 aggregate interop with gcc ms_abi failed ({code})"),
        None => eprintln!("skipping: no C compiler"),
    }
}

/// A variadic callee walking its arguments from `__lf_va_overflow_area`, and
/// a variadic call passing doubles (which must also land in the GPRs).
const VARIADIC: &str = r#"
module "v"
func @__lf_va_overflow_area() -> ptr
func @cvf(i32, ...) -> f64
func @vsum(i64, ...) -> i64 {
entry ^0(%n: i64):
  %p = call @__lf_va_overflow_area() : ptr
  br ^1(%p, %n, i64 0)
^1(%q: ptr, %k: i64, %acc: i64):
  %done = icmp eq %k, i64 0 : i1
  cond_br %done, ^2(%acc), ^3
^3:
  %v = load %q align 8 : i64
  %acc2 = add %acc, %v : i64
  %q2 = ptr_add %q, i64 8 : ptr
  %k2 = sub %k, i64 1 : i64
  br ^1(%q2, %k2, %acc2)
^2(%r: i64):
  ret %r
}
func @callvf() -> f64 {
entry ^0:
  %r = call @cvf(i32 5, f64 0x3ff8000000000000, f64 0x4004000000000000, f64 0x4010000000000000, f64 0x4000000000000000, f64 0x3ff0000000000000) : f64
  ret %r
}
"#;

#[test]
fn win64_variadics_interoperate_with_gcc_ms_abi() {
    let (m, syms) = parse(VARIADIC);
    let c = r#"
        #define MS __attribute__((ms_abi))
        MS long long vsum(long long n, ...);
        MS double callvf(void);
        MS double cvf(int n, ...) {
            __builtin_ms_va_list ap;
            __builtin_ms_va_start(ap, n);
            double s = 0;
            for (int i = 0; i < n; i++) s += __builtin_va_arg(ap, double);
            __builtin_ms_va_end(ap);
            return s;
        }
        int main(void) {
            if (vsum(5, 1LL, 2LL, 3LL, 4LL, 5LL) != 15) return 1;
            if (vsum(0) != 0) return 2;
            if (vsum(2, 40LL, 2LL) != 42) return 3;
            if (callvf() != 11.0) return 4;
            return 0;
        }
    "#;
    match run_with_c(&m, &syms, "variadic", c) {
        Some(code) => assert_eq!(code, 0, "Win64 variadic interop with gcc ms_abi failed ({code})"),
        None => eprintln!("skipping: no C compiler"),
    }
}

#[test]
fn win64_reads_the_first_argument_from_rcx() {
    // Structural: a Win64 function reads its first argument from rcx.
    let (m, syms) = parse(
        "module \"s\"\nfunc @id(i64) -> i64 {\nentry ^0(%a: i64):\n  ret %a\n}\n",
    );
    let opts = CodegenOptions::default().with_os(TargetOs::Windows);
    let obj = super::compile_module_with(&m, &syms, &opts).object;
    let text = &obj.sections()[0].bytes;
    // mov rax, rcx (48 89 c8) somewhere after the prologue.
    assert!(text.windows(3).any(|w| w == [0x48, 0x89, 0xc8]), "{text:02x?}");
}
