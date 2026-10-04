//! Execution tests for `inline_asm` on x86-64 (`docs/ir-design.md` §6i).
//!
//! Each program is `.lf` text with GCC-style inline asm, verified, run
//! through the `-O0`/`-O2` pipelines, compiled by our backend (the template
//! instantiated with the allocated registers and assembled by rsasm), loaded
//! by the in-process JIT and called. The `(i64, i64) -> i64` entry points
//! keep every call type-correct.

use crate::jit::Jit;
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize};

/// Parse and verify `src`, optimize it at `level`, check its asm, compile it
/// with the JIT and call `@f(a, b)`.
fn run(src: &str, level: OptLevel, a: i64, b: i64) -> i64 {
    let mut syms = StrInterner::new();
    let mut m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse .lf: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    optimize(&mut m, level);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify after {level:?}: {e:?}"));
    super::check_inline_asm(&m, &syms).unwrap_or_else(|e| panic!("check: {e}"));
    let jit = Jit::new().compile(&m, &syms).expect("jit compile");
    let f = jit.get_fn_i64_i64_i64("f").expect("@f");
    f(a, b)
}

/// [`run`] at `-O0` and `-O2`, asserting both give `want`.
fn check(src: &str, a: i64, b: i64, want: i64) {
    for level in [OptLevel::O0, OptLevel::O2] {
        assert_eq!(run(src, level, a, b), want, "at {level:?}");
    }
}

/// A function header taking `%a` and `%b`.
fn func(body: &str) -> String {
    format!("module \"t\"\nfunc @f(i64, i64) -> i64 {{\nentry ^0(%a: i64, %b: i64):\n{body}\n}}\n")
}

#[test]
fn add_with_a_matching_constraint() {
    let src = func(
        r#"  %r = inline_asm "addq %2, %0" outs("=r" i64) ins("0"(%a), "r"(%b)) : i64
  ret %r"#,
    );
    check(&src, 40, 2, 42);
    check(&src, -7, 7, 0);
}

#[test]
fn rdtsc_into_a_pair_of_outputs() {
    // `rdtsc` twice: the 64-bit counter `edx:eax` never goes backwards.
    let src = func(
        r#"  %lo = inline_asm volatile "rdtsc" outs("=a" i32, "=d" i32) : i32
  %hi = asm_output %lo, 1 : i32
  %l = zext %lo : i64
  %h = zext %hi : i64
  %hs = shl %h, i64 32 : i64
  %t1 = or %hs, %l : i64
  %lo2 = inline_asm volatile "rdtsc" outs("=a" i32, "=d" i32) : i32
  %hi2 = asm_output %lo2, 1 : i32
  %l2 = zext %lo2 : i64
  %h2 = zext %hi2 : i64
  %hs2 = shl %h2, i64 32 : i64
  %t2 = or %hs2, %l2 : i64
  %up = icmp uge %t2, %t1 : i1
  %nz = icmp ne %t1, i64 0 : i1
  %ok = and %up, %nz : i1
  %r = zext %ok : i64
  ret %r"#,
    );
    check(&src, 0, 0, 1);
}

#[test]
fn cpuid_with_four_outputs() {
    // Leaf 0: the vendor string's first four letters in ebx (`rbx` is
    // callee-saved: the JIT's Rust caller relies on it being restored).
    let src = func(
        r#"  %leaf = trunc %a : i32
  %sub = trunc %b : i32
  %eax = inline_asm "cpuid" outs("=a" i32, "=b" i32, "=c" i32, "=d" i32) ins("0"(%leaf), "2"(%sub)) : i32
  %ebx = asm_output %eax, 1 : i32
  %ecx = asm_output %eax, 2 : i32
  %edx = asm_output %eax, 3 : i32
  %x = zext %ebx : i64
  %y = zext %eax : i64
  %ys = shl %y, i64 32 : i64
  %r = or %ys, %x : i64
  ret %r"#,
    );
    for level in [OptLevel::O0, OptLevel::O2] {
        let r = run(&src, level, 0, 0);
        let max_leaf = (r >> 32) as u32;
        let vendor = (r as u32).to_le_bytes();
        assert!(max_leaf >= 1, "cpuid max leaf {max_leaf}");
        assert!(vendor.iter().all(u8::is_ascii_alphabetic), "vendor bytes {vendor:?}");
    }
}

#[test]
fn xchg_and_cmpxchg_atomics() {
    // `xchg` swaps a register with memory: returns old * 1000 + new.
    let xchg = func(
        r#"  %p = alloca i64 : ptr
  store %a, %p align 8 : i64
  %old = inline_asm volatile "xchgq %0, %1" outs("+r" i64 (%b), "+m" (%p)) clobbers("memory") : i64
  %new = load %p align 8 : i64
  %m = mul %old, i64 1000 : i64
  %r = add %m, %new : i64
  ret %r"#,
    );
    check(&xchg, 7, 9, 7009);
    // `lock cmpxchg`: compare `rax` with memory, store on a match; returns
    // prev * 1000 + memory afterwards.
    let cas = func(
        r#"  %p = alloca i64 : ptr
  store i64 5, %p align 8 : i64
  %prev = inline_asm volatile "lock; cmpxchgq %2, %1" outs("=a" i64, "+m" (%p)) ins("r"(%b), "0"(%a)) clobbers("memory", "cc") : i64
  %now = load %p align 8 : i64
  %m = mul %prev, i64 1000 : i64
  %r = add %m, %now : i64
  ret %r"#,
    );
    check(&cas, 5, 8, 5008);
    check(&cas, 4, 8, 5005);
}

#[test]
fn bswap_32_and_64() {
    let src = func(
        r#"  %x = trunc %a : i32
  %s = inline_asm "bswap %0" outs("=r" i32) ins("0"(%x)) : i32
  %y = inline_asm "bswapq %0" outs("+r" i64 (%b)) : i64
  %sz = zext %s : i64
  %r = xor %sz, %y : i64
  ret %r"#,
    );
    check(&src, 0x1122_3344, 0, 0x4433_2211);
    check(&src, 0, 0x0102_0304_0506_0708, 0x0807_0605_0403_0201);
}

#[test]
fn memory_operands() {
    // An `m` input read and an `=m` output written by an output-less asm.
    let src = func(
        r#"  %p = alloca i64 : ptr
  %q = alloca i64 : ptr
  store %a, %p align 8 : i64
  %v = inline_asm "movq %1, %0" outs("=r" i64) ins("m"(%p)) : i64
  inline_asm volatile "movq %1, %0" outs("=m" (%q)) ins("r"(%b)) : void
  %w = load %q align 8 : i64
  %r = sub %v, %w : i64
  ret %r"#,
    );
    check(&src, 50, 8, 42);
    // `%H` names the next eightbyte of a memory operand; `%a` an address.
    let pair = func(
        r#"  %p = alloca [2 x i64] : ptr
  store %a, %p align 8 : i64
  %p1 = ptr_add %p, i64 8 : ptr
  store %b, %p1 align 8 : i64
  %v = inline_asm "movq %H1, %0" outs("=r" i64) ins("m"(%p)) : i64
  %u = inline_asm "movq %a1, %0" outs("=r" i64) ins("r"(%p)) : i64
  %r = sub %v, %u : i64
  ret %r"#,
    );
    check(&pair, 3, 10, 7);
}

#[test]
fn xmm_operands() {
    // Scalar doubles in xmm registers.
    let src = func(
        r#"  %x = bitcast %a : f64
  %y = bitcast %b : f64
  %s = inline_asm "addsd %2, %0" outs("=x" f64) ins("0"(%x), "x"(%y)) : f64
  %r = fptosi %s : i64
  ret %r"#,
    );
    check(&src, 2.5f64.to_bits() as i64, 39.5f64.to_bits() as i64, 42);
    // A vector: `paddd` on four lanes.
    let vec = func(
        r#"  %x = trunc %a : i32
  %y = trunc %b : i32
  %vx = splat %x : <4 x i32>
  %vy = splat %y : <4 x i32>
  %vs = inline_asm "paddd %1, %0" outs("+x" <4 x i32> (%vx)) ins("x"(%vy)) : <4 x i32>
  %s = reduce add %vs : i32
  %r = sext %s : i64
  ret %r"#,
    );
    check(&vec, 10, 1, 44);
}

#[test]
fn early_clobber_output() {
    // `%0` is written before `%1` and `%2` are read: an early-clobber output
    // must not share a register with them.
    let src = func(
        r#"  %r = inline_asm "movq $100, %0; addq %1, %0; addq %2, %0" outs("=&r" i64) ins("r"(%a), "r"(%b)) : i64
  ret %r"#,
    );
    check(&src, 20, 3, 123);
}

#[test]
fn clobbered_registers_are_preserved_around_the_asm() {
    // Values live across an asm that clobbers every general register (the
    // callee-saved ones included, which the prologue then saves) and xmm0-15.
    let src = func(
        r#"  %x = mul %a, i64 3 : i64
  %y = add %b, i64 11 : i64
  inline_asm volatile "xorl %%eax, %%eax; xorl %%ebx, %%ebx; xorl %%ecx, %%ecx; xorl %%edx, %%edx; xorl %%esi, %%esi; xorl %%edi, %%edi; xorl %%r8d, %%r8d; xorl %%r9d, %%r9d; xorl %%r10d, %%r10d; xorl %%r11d, %%r11d; xorl %%r12d, %%r12d; xorl %%r13d, %%r13d; xorl %%r14d, %%r14d; xorl %%r15d, %%r15d; pxor %%xmm0, %%xmm0; pxor %%xmm15, %%xmm15" clobbers("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "xmm0", "xmm15", "cc") : void
  %z = mul %x, %y : i64
  ret %z"#,
    );
    check(&src, 5, 3, 210);
}

#[test]
fn named_operands_and_modifiers() {
    // `%[name]`, and the `k`/`w`/`b`/`q` register sizes.
    let src = func(
        r#"  %r = inline_asm "movzbl %b[in], %k[out]; movw %w[in], %w[t]; movzwl %w[t], %k[t]; shlq $8, %q[t]; orq %q[t], %q[out]" outs("=&r" [out] i64, "=&r" [t] i64) ins("r" [in] (%a)) : i64
  ret %r"#,
    );
    check(&src, 0x1234, 0, 0x12_3434);
    // An immediate (`i`, printed with `$`) and `%c` (bare), `%n` (negated).
    let imm = func(
        r#"  %r = inline_asm "leaq %c2(%1), %0; addq %3, %0; subq $%n2, %0" outs("=r" i64) ins("r"(%a), "i"(i64 10), "n"(i64 5)) : i64
  ret %r"#,
    );
    check(&imm, 100, 0, 125);
}

#[test]
fn labels_inside_the_template() {
    // `1:`/`1f`/`2f` numeric labels and `%=` labels, in two asm statements
    // that use the same names.
    let src = func(
        r#"  %x = inline_asm "testq %1, %1; jz 1f; movq $1, %0; jmp 2f; 1: movq $2, %0; 2:" outs("=r" i64) ins("r"(%a)) clobbers("cc") : i64
  %y = inline_asm "testq %1, %1; jz .Lz%=; movq $10, %0; jmp .Le%=; .Lz%=: movq $20, %0; .Le%=:" outs("=r" i64) ins("r"(%b)) clobbers("cc") : i64
  %r = add %x, %y : i64
  ret %r"#,
    );
    check(&src, 0, 0, 22);
    check(&src, 1, 0, 21);
    check(&src, 0, 1, 12);
    // A backward loop: count `b` down to zero, adding `a` each time.
    let lp = func(
        r#"  %r = inline_asm "xorl %k0, %k0; testq %1, %1; jz 2f; 1: addq %2, %0; decq %1; jnz 1b; 2:" outs("=&r" i64, "+r" i64 (%b)) ins("r"(%a)) clobbers("cc") : i64
  ret %r"#,
    );
    check(&lp, 6, 7, 42);
}

#[test]
fn relocations_from_the_template() {
    // A call to another function of the module (`R_X86_64_PLT32`), a
    // RIP-relative load of a global named by an `i` operand (`PC32`), and an
    // absolute reference to the template's own label (an `R_X86_64_64`
    // against the function itself).
    let src = r#"module "t"
global @g : i64 = i64 1000
func @twice(i64) -> i64 {
entry ^0(%x: i64):
  %r = add %x, %x : i64
  ret %r
}
func @f(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %c = inline_asm volatile "movq %1, %%rdi; call twice; movq %%rax, %0" outs("=r" i64) ins("r"(%a)) clobbers("rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "memory", "cc") : i64
  %g = inline_asm "movq %c1(%%rip), %0" outs("=r" i64) ins("i"(@g)) : i64
  %d = inline_asm "movabsq $1f, %0; leaq 1f(%%rip), %1; subq %1, %0; 1:" outs("=&r" i64, "=&r" i64) : i64
  %s = add %c, %g : i64
  %t = add %s, %d : i64
  ret %t
}
"#;
    for level in [OptLevel::O0, OptLevel::O2] {
        assert_eq!(run(src, level, 21, 0), 1042, "at {level:?}");
    }
}

#[test]
fn asm_in_a_loop_with_block_arguments() {
    // Pure asm in a loop body: LICM must not hoist them, and the first one's
    // operand changes every iteration.
    let src = func(
        r#"  br ^1(i64 0, %a)
^1(%i: i64, %acc: i64):
  %done = icmp sge %i, %b : i1
  cond_br %done, ^3, ^2
^2:
  %n = inline_asm "leaq 3(%1), %0" outs("=r" i64) ins("r"(%acc)) : i64
  %k = inline_asm "movq $2, %0" outs("=r" i64) : i64
  %m = add %n, %k : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %m)
^3:
  ret %acc"#,
    );
    check(&src, 1, 4, 21);
}

#[test]
fn empty_template_value_barrier() {
    // `asm("" : "+r"(x))` launders a value through a register: no bytes.
    let src = func(
        r#"  %x = add %a, %b : i64
  %y = inline_asm "" outs("+r" i64 (%x)) : i64
  %r = add %y, i64 1 : i64
  ret %r"#,
    );
    check(&src, 40, 1, 42);
}
