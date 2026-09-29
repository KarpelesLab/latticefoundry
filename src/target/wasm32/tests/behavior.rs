//! Differential programs beyond single operations: control flow (loops,
//! switches, an irreducible CFG, recursion), memory and globals, calls
//! (direct, indirect, host imports), atomics, and `i128` values.

use super::{differential, differential_optimized, no_node};

/// `(function, args)` cases from a compact list.
fn cases(list: &[(&'static str, &[u128])]) -> Vec<(&'static str, Vec<u128>)> {
    list.iter().map(|(n, a)| (*n, a.to_vec())).collect()
}

pub(super) const CONTROL: &str = r#"
module "control"

; A counted loop with two loop-carried values.
func @sum(i32) -> i32 {
entry ^0(%n: i32):
  br ^1(i32 0, i32 0)
^1(%i: i32, %acc: i32):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %acc2 = add %acc, %i : i32
  %i2 = add %i, i32 1 : i32
  br ^1(%i2, %acc2)
^3:
  ret %acc
}

; Block arguments that swap: the edge copy must be parallel.
func @swap(i32, i32, i32) -> i32 {
entry ^0(%x: i32, %y: i32, %k: i32):
  br ^1(%x, %y, %k)
^1(%a: i32, %b: i32, %n: i32):
  %z = icmp eq %n, i32 0 : i1
  cond_br %z, ^3, ^2
^2:
  %n2 = sub %n, i32 1 : i32
  %b2 = mul %b, i32 3 : i32
  br ^1(%b2, %a, %n2)
^3:
  %s = shl %a, i32 16 : i32
  %r = or %s, %b : i32
  ret %r
}

; Nested loops with a break out of the inner one and a merge after.
func @nested(i32) -> i64 {
entry ^0(%n: i32):
  br ^1(i32 0, i64 0)
^1(%i: i32, %acc: i64):
  %c = icmp slt %i, %n : i1
  cond_br %c, ^2(i32 0, %acc), ^6
^2(%j: i32, %a2: i64):
  %cj = icmp slt %j, %i : i1
  cond_br %cj, ^3, ^5(%a2)
^3:
  %p = mul %i, %j : i32
  %big = icmp ugt %p, i32 200 : i1
  cond_br %big, ^5(%a2), ^4
^4:
  %pe = zext %p : i64
  %a3 = add %a2, %pe : i64
  %j2 = add %j, i32 1 : i32
  br ^2(%j2, %a3)
^5(%a4: i64):
  %i2 = add %i, i32 1 : i32
  br ^1(%i2, %a4)
^6:
  ret %acc
}

; A diamond whose join takes different arguments from each side.
func @diamond(i8, i8) -> i8 {
entry ^0(%a: i8, %b: i8):
  %c = icmp sgt %a, %b : i1
  cond_br %c, ^1, ^2
^1:
  %d = sub %a, %b : i8
  br ^3(%d)
^2:
  %e = sub %b, %a : i8
  %f = mul %e, i8 3 : i8
  br ^3(%f)
^3(%r: i8):
  ret %r
}

; Both arms of a cond_br to the same block with different arguments.
func @same_target(i1, i32) -> i32 {
entry ^0(%c: i1, %x: i32):
  %y = add %x, i32 7 : i32
  cond_br %c, ^1(%x), ^1(%y)
^1(%r: i32):
  ret %r
}

; A dense switch (br_table), with shared targets and edge arguments.
func @dense(i32) -> i32 {
entry ^0(%x: i32):
  switch %x, ^9(i32 99) [0: ^1, 1: ^2, 2: ^3, 3: ^1, 5: ^2, 6: ^9(i32 66), 4: ^9(%x)]
^1:
  ret i32 10
^2:
  ret i32 20
^3:
  ret i32 30
^9(%v: i32):
  ret %v
}

; A dense switch on an offset range of i8 (negative case values).
func @dense8(i8) -> i32 {
entry ^0(%x: i8):
  switch %x, ^4 [-3: ^1, -2: ^2, -1: ^3, 0: ^1, 1: ^2, 2: ^3]
^1:
  ret i32 1
^2:
  ret i32 2
^3:
  ret i32 3
^4:
  ret i32 4
}

; A sparse switch (compare chain) on i64.
func @sparse(i64) -> i32 {
entry ^0(%x: i64):
  switch %x, ^4 [1000000: ^1, -5: ^2, 81985529216486895: ^3, 7: ^1]
^1:
  ret i32 1
^2:
  ret i32 2
^3:
  ret i32 3
^4:
  ret i32 0
}

; A dense switch on i64, inside a loop, merging afterwards.
func @switch_loop(i64) -> i64 {
entry ^0(%n: i64):
  br ^1(i64 0, i64 0)
^1(%i: i64, %acc: i64):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^8
^2:
  %m = urem %i, i64 5 : i64
  switch %m, ^7(i64 1) [0: ^3, 1: ^4, 2: ^5, 3: ^7(i64 1000)]
^3:
  br ^7(i64 3)
^4:
  br ^7(i64 5)
^5:
  %t = mul %i, %i : i64
  br ^7(%t)
^7(%d: i64):
  %acc2 = add %acc, %d : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2, %acc2)
^8:
  ret %acc
}

; An irreducible CFG: the cycle ^1 <-> ^2 is entered at both nodes.
func @irreducible(i32, i32) -> i32 {
entry ^0(%n: i32, %s: i32):
  %c = icmp eq %s, i32 0 : i1
  cond_br %c, ^1(i32 0), ^2(%s)
^1(%a: i32):
  %a2 = add %a, i32 1 : i32
  %d = icmp ult %a2, %n : i1
  cond_br %d, ^2(%a2), ^3(%a2)
^2(%b: i32):
  %b2 = add %b, i32 3 : i32
  %e = icmp ult %b2, i32 50 : i1
  cond_br %e, ^1(%b2), ^3(%b2)
^3(%r: i32):
  ret %r
}

; Recursion.
func @fib(i32) -> i32 {
entry ^0(%n: i32):
  %c = icmp ult %n, i32 2 : i1
  cond_br %c, ^1, ^2
^1:
  ret %n
^2:
  %a = sub %n, i32 1 : i32
  %b = sub %n, i32 2 : i32
  %x = call @fib(%a) : i32
  %y = call @fib(%b) : i32
  %r = add %x, %y : i32
  ret %r
}

func @is_even(i32) -> i1 {
entry ^0(%n: i32):
  %z = icmp eq %n, i32 0 : i1
  cond_br %z, ^1, ^2
^1:
  ret i1 1
^2:
  %m = sub %n, i32 1 : i32
  %r = call @is_odd(%m) : i1
  ret %r
}

func @is_odd(i32) -> i1 {
entry ^0(%n: i32):
  %z = icmp eq %n, i32 0 : i1
  cond_br %z, ^1, ^2
^1:
  ret i1 0
^2:
  %m = sub %n, i32 1 : i32
  %r = call @is_even(%m) : i1
  ret %r
}

; An `unreachable` terminator and a block no edge reaches.
func @late_entry(i32) -> i32 {
entry ^0(%x: i32):
  %c = icmp eq %x, i32 12345 : i1
  cond_br %c, ^2, ^1(%x)
^1(%v: i32):
  %w = mul %v, i32 2 : i32
  ret %w
^2:
  unreachable
^3:
  ret i32 0
}
"#;

/// Loops, parallel block-argument copies, nested loops with breaks, dense and
/// sparse switches at several widths, an irreducible CFG, and recursion.
#[test]
fn control_flow() {
    let mut list: Vec<(&str, Vec<u128>)> = Vec::new();
    for n in [0u128, 1, 2, 10, 100] {
        list.push(("sum", vec![n]));
        list.push(("nested", vec![n]));
        list.push(("switch_loop", vec![n]));
        list.push(("irreducible", vec![n, 0]));
        list.push(("irreducible", vec![n, 7]));
        list.push(("fib", vec![n % 20]));
        list.push(("is_even", vec![n]));
        list.push(("is_odd", vec![n]));
        list.push(("late_entry", vec![n]));
    }
    for k in 0..6u128 {
        list.push(("swap", vec![0x1234, 0x9, k]));
    }
    for x in 0..10u128 {
        list.push(("dense", vec![x]));
        list.push(("dense", vec![x.wrapping_neg() & 0xffff_ffff]));
        list.push(("dense8", vec![x.wrapping_sub(5) & 0xff]));
        list.push(("same_target", vec![x & 1, x * 1000]));
    }
    for x in [1000000u128, 5u128.wrapping_neg() & u128::from(u64::MAX), 81985529216486895, 7, 8, 0] {
        list.push(("sparse", vec![x]));
    }
    for (a, b) in [(1u128, 2u128), (100, 3), (0x80, 0x7f), (0xff, 1), (50, 50)] {
        list.push(("diamond", vec![a, b]));
    }
    let Some(t) = differential("control", CONTROL, &list) else { return no_node("control_flow") };
    assert_eq!(t.skipped, 0, "{t:?}");
    // The same calls against the program after the -O2 pipeline.
    let t = differential_optimized("control", CONTROL, &list).expect("node");
    assert_eq!(t.skipped, 0, "{t:?}");
}

pub(super) const MEMORY: &str = r#"
module "memory"

global @table : [8 x i32] = [8 x i32] (i32 3, i32 1, i32 4, i32 1, i32 5, i32 9, i32 2, i32 6)
global constant @msg : [6 x i8] = [6 x i8] "hello\n"
global @counter : i64 = i64 0
global @zeros : [16 x i16] = [16 x i16] (i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0)
global @rec : {i8, i32, ptr, i64} = {i8, i32, ptr, i64} (i8 -2, i32 77, ptr @table + 12, i64 -9)
global constant @pp : ptr = ptr @rec

func @table_sum() -> i32 {
entry ^0:
  br ^1(i32 0, i32 0)
^1(%i: i32, %acc: i32):
  %c = icmp ult %i, i32 8 : i1
  cond_br %c, ^2, ^3
^2:
  %o = mul %i, i32 4 : i32
  %p = ptr_add @table, %o : ptr
  %v = load %p align 4 : i32
  %acc2 = add %acc, %v : i32
  %i2 = add %i, i32 1 : i32
  br ^1(%i2, %acc2)
^3:
  ret %acc
}

func @msg_byte(i32) -> i8 {
entry ^0(%i: i32):
  %p = ptr_add @msg, %i : ptr
  %v = load %p align 1 : i8
  ret %v
}

; State that persists across calls.
func @bump(i64) -> i64 {
entry ^0(%d: i64):
  %v = load @counter align 8 : i64
  %w = add %v, %d : i64
  store %w, @counter align 8 : i64
  ret %w
}

; Address constants in data: follow @pp -> @rec -> @table + 12.
func @chase() -> i32 {
entry ^0:
  %r = load @pp align 4 : ptr
  %f = ptr_add %r, i32 8 : ptr
  %t = load %f align 4 : ptr
  %v = load %t align 4 : i32
  %b = load %r align 1 : i8
  %bx = sext %b : i32
  %s = add %v, %bx : i32
  ret %s
}

; Narrow and odd-width stores and loads through a stack frame.
func @widths(i64) -> i64 {
entry ^0(%x: i64):
  %s = alloca [4 x i64] : ptr
  %t8 = trunc %x : i8
  %t16 = trunc %x : i16
  %t24 = trunc %x : i24
  %t48 = trunc %x : i48
  store %x, %s align 8 : i64
  %p1 = ptr_add %s, i32 8 : ptr
  store %t8, %p1 align 1 : i8
  %p2 = ptr_add %s, i32 9 : ptr
  store %t24, %p2 align 1 : i24
  %p3 = ptr_add %s, i32 12 : ptr
  store %t16, %p3 align 2 : i16
  %p4 = ptr_add %s, i32 16 : ptr
  store %t48, %p4 align 8 : i48
  %l24 = load %p2 align 1 : i24
  %l48 = load %p4 align 8 : i48
  %l8 = load %p1 align 1 : i8
  %l16 = load %p3 align 2 : i16
  %e24 = sext %l24 : i64
  %e48 = zext %l48 : i64
  %e8 = sext %l8 : i64
  %e16 = zext %l16 : i64
  %a = add %e24, %e48 : i64
  %b = xor %e8, %e16 : i64
  %r = add %a, %b : i64
  ret %r
}

; Floats in memory.
func @fmem(f64, f32) -> f64 {
entry ^0(%a: f64, %b: f32):
  %s = alloca {f32, f64} : ptr
  store %b, %s align 4 : f32
  %p = ptr_add %s, i32 8 : ptr
  store %a, %p align 8 : f64
  %lb = load %s align 4 : f32
  %la = load %p align 8 : f64
  %e = fpext %lb : f64
  %r = fadd %la, %e : f64
  ret %r
}

; A runtime-sized stack buffer.
func @dyn(i32) -> i32 {
entry ^0(%n: i32):
  %bytes = mul %n, i32 4 : i32
  %buf = dyn_alloca %bytes align 8 : ptr
  br ^1(i32 0)
^1(%i: i32):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^3(i32 0, i32 0)
^2:
  %o = mul %i, i32 4 : i32
  %p = ptr_add %buf, %o : ptr
  %sq = mul %i, %i : i32
  store %sq, %p align 4 : i32
  %i2 = add %i, i32 1 : i32
  br ^1(%i2)
^3(%j: i32, %acc: i32):
  %d = icmp ult %j, %n : i1
  cond_br %d, ^4, ^5
^4:
  %o2 = mul %j, i32 4 : i32
  %q = ptr_add %buf, %o2 : ptr
  %v = load %q align 4 : i32
  %acc2 = add %acc, %v : i32
  %j2 = add %j, i32 1 : i32
  br ^3(%j2, %acc2)
^5:
  ret %acc
}

; Recursion with a frame per activation (the shadow stack must unwind).
func @depth(i32) -> i32 {
entry ^0(%n: i32):
  %slot = alloca i32 : ptr
  store %n, %slot align 4 : i32
  %z = icmp eq %n, i32 0 : i1
  cond_br %z, ^1, ^2
^1:
  ret i32 0
^2:
  %m = sub %n, i32 1 : i32
  %r = call @depth(%m) : i32
  %v = load %slot align 4 : i32
  %s = add %r, %v : i32
  ret %s
}

; Volatile accesses are plain accesses.
func @vol(i16) -> i16 {
entry ^0(%x: i16):
  %p = ptr_add @zeros, i32 6 : ptr
  store volatile %x, %p align 2 : i16
  %v = load volatile %p align 2 : i16
  %w = add %v, i16 1 : i16
  ret %w
}

; pointer <-> integer.
func @ptrint(i32) -> i32 {
entry ^0(%o: i32):
  %p = ptr_add @table, %o : ptr
  %i = ptrtoint %p : i32
  %b = ptrtoint @table : i32
  %d = sub %i, %b : i32
  %q = inttoptr %i : ptr
  %e = icmp eq %q, %p : i1
  %ez = zext %e : i32
  %r = add %d, %ez : i32
  ret %r
}
"#;

/// Globals (arrays, strings, structs, address constants), state across
/// calls, narrow/odd-width/float memory accesses, `alloca`, `dyn_alloca`,
/// recursion through the shadow stack, volatile accesses, pointer casts.
#[test]
fn memory_and_globals() {
    let mut list = cases(&[
        ("table_sum", &[]),
        ("chase", &[]),
        ("bump", &[5]),
        ("bump", &[7]),
        ("bump", &[u64::MAX as u128]),
        ("dyn", &[0]),
        ("dyn", &[1]),
        ("dyn", &[33]),
        ("depth", &[0]),
        ("depth", &[10]),
        ("depth", &[1000]),
        ("vol", &[0x7fff]),
        ("ptrint", &[12]),
        ("fmem", &[0x3ff4_0000_0000_0000, 0x4020_0000]),
    ]);
    for i in 0..6 {
        list.push(("msg_byte", vec![i]));
    }
    for x in [0u128, 1, 0xffff_ffff_ffff_ffff, 0x8000_0000_0000_0000, 0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210] {
        list.push(("widths", vec![x]));
    }
    let Some(t) = differential("memory", MEMORY, &list) else { return no_node("memory_and_globals") };
    assert_eq!(t.skipped, 0, "{t:?}");
    // The same calls against the program after the -O2 pipeline.
    let t = differential_optimized("memory", MEMORY, &list).expect("node");
    assert_eq!(t.skipped, 0, "{t:?}");
}

pub(super) const CALLS: &str = r#"
module "calls"

func @host_mul3(i32) -> i32
func @host_i64(i64) -> i64
func @host_sloppy8() -> i8
func @host_half(f64) -> f64
func @host_void() -> void

func internal @twice(i32) -> i32 {
entry ^0(%x: i32):
  %r = add %x, %x : i32
  ret %r
}

func internal @neg(i32) -> i32 {
entry ^0(%x: i32):
  %r = sub i32 0, %x : i32
  ret %r
}

global constant @ops : [3 x ptr] = [3 x ptr] (ptr @twice, ptr @neg, ptr @host_mul3)

; Indirect calls through a table in data.
func @apply(i32, i32) -> i32 {
entry ^0(%k: i32, %x: i32):
  %o = mul %k, i32 4 : i32
  %p = ptr_add @ops, %o : ptr
  %f = load %p align 4 : ptr
  %r = call %f(%x) : i32
  ret %r
}

; A function pointer taken in code and called.
func @via_ref(i32) -> i32 {
entry ^0(%x: i32):
  %c = icmp slt %x, i32 0 : i1
  %f = select %c, @neg, @twice : ptr
  %r = call %f(%x) : i32
  ret %r
}

; Imports from the host.
func @host(i32, i64, f64) -> i64 {
entry ^0(%a: i32, %b: i64, %c: f64):
  call @host_void() : void
  %x = call @host_mul3(%a) : i32
  %y = call @host_i64(%b) : i64
  %h = call @host_half(%c) : f64
  %hi = fptosi %h : i64
  %xe = sext %x : i64
  %s = add %xe, %y : i64
  %r = add %s, %hi : i64
  ret %r
}

; A host returning garbage above an i8: the result must be masked.
func @sloppy() -> i32 {
entry ^0:
  %v = call @host_sloppy8() : i8
  %e = zext %v : i32
  ret %e
}

; Narrow parameters and results between our own functions.
func @narrow_callee(i8, i16, i1) -> i8 {
entry ^0(%a: i8, %b: i16, %c: i1):
  %bt = trunc %b : i8
  %s = add %a, %bt : i8
  %n = sub i8 0, %s : i8
  %r = select %c, %s, %n : i8
  ret %r
}

func @narrow_caller(i32) -> i32 {
entry ^0(%x: i32):
  %a = trunc %x : i8
  %s = lshr %x, i32 8 : i32
  %b = trunc %s : i16
  %c = trunc %x : i1
  %r = call @narrow_callee(%a, %b, %c) : i8
  %e = sext %r : i32
  ret %e
}
"#;

/// Direct, indirect (through data and through code), and host calls, and the
/// masking of narrow values crossing call boundaries.
#[test]
fn calls() {
    let mut list = Vec::new();
    for k in 0..3u128 {
        for x in [0u128, 5, 0xffff_fff0] {
            list.push(("apply", vec![k, x]));
        }
    }
    for x in [3u128, 0xffff_fffd] {
        list.push(("via_ref", vec![x]));
    }
    list.push(("host", vec![7, 0xffff_ffff_ffff_fff0, 0x4022_0000_0000_0000]));
    list.push(("sloppy", vec![]));
    for x in [0u128, 0x1234_5678, 0xffff_ffff, 0x0080_7f01, 0x00ff_80fe] {
        list.push(("narrow_caller", vec![x]));
    }
    let Some(t) = differential("calls", CALLS, &list) else { return no_node("calls") };
    assert_eq!(t.skipped, 0, "{t:?}");
    // The same calls against the program after the -O2 pipeline.
    let t = differential_optimized("calls", CALLS, &list).expect("node");
    assert_eq!(t.skipped, 0, "{t:?}");
}

pub(super) const ATOMICS: &str = r#"
module "atomics"

global @w32 : i32 = i32 100
global @w8 : i8 = i8 -3
global @w16 : i16 = i16 1000
global @w64 : i64 = i64 -5

func @rmw32(i32, i32) -> i32 {
entry ^0(%op: i32, %v: i32):
  store i32 100, @w32 align 4 : i32
  switch %op, ^12 [0: ^1, 1: ^2, 2: ^3, 3: ^4, 4: ^5, 5: ^6, 6: ^7, 7: ^8, 8: ^9, 9: ^10, 10: ^11]
^1:
  %a = atomic_rmw xchg seq_cst @w32, %v align 4 : i32
  br ^13(%a)
^2:
  %b = atomic_rmw add seq_cst @w32, %v align 4 : i32
  br ^13(%b)
^3:
  %c = atomic_rmw sub acq_rel @w32, %v align 4 : i32
  br ^13(%c)
^4:
  %d = atomic_rmw and relaxed @w32, %v align 4 : i32
  br ^13(%d)
^5:
  %e = atomic_rmw nand seq_cst @w32, %v align 4 : i32
  br ^13(%e)
^6:
  %f = atomic_rmw or seq_cst @w32, %v align 4 : i32
  br ^13(%f)
^7:
  %g = atomic_rmw xor seq_cst @w32, %v align 4 : i32
  br ^13(%g)
^8:
  %h = atomic_rmw max seq_cst @w32, %v align 4 : i32
  br ^13(%h)
^9:
  %i = atomic_rmw min seq_cst @w32, %v align 4 : i32
  br ^13(%i)
^10:
  %j = atomic_rmw umax seq_cst @w32, %v align 4 : i32
  br ^13(%j)
^11:
  %k = atomic_rmw umin seq_cst @w32, %v align 4 : i32
  br ^13(%k)
^12:
  ret i32 -1
^13(%old: i32):
  %new = atomic_load seq_cst @w32 align 4 : i32
  %s = shl %new, i32 8 : i32
  %r = xor %s, %old : i32
  ret %r
}

func @rmw8(i32, i8) -> i32 {
entry ^0(%op: i32, %v: i8):
  store i8 -3, @w8 align 1 : i8
  switch %op, ^6 [0: ^1, 1: ^2, 2: ^3, 3: ^4, 4: ^5]
^1:
  %a = atomic_rmw add seq_cst @w8, %v align 1 : i8
  br ^7(%a)
^2:
  %b = atomic_rmw max seq_cst @w8, %v align 1 : i8
  br ^7(%b)
^3:
  %c = atomic_rmw umin seq_cst @w8, %v align 1 : i8
  br ^7(%c)
^4:
  %d = atomic_rmw nand seq_cst @w8, %v align 1 : i8
  br ^7(%d)
^5:
  %e = cmpxchg seq_cst seq_cst @w8, i8 -3, %v align 1 : i8
  br ^7(%e)
^6:
  ret i32 -1
^7(%old: i8):
  %new = atomic_load acquire @w8 align 1 : i8
  %ne = zext %new : i32
  %oe = zext %old : i32
  %s = shl %ne, i32 8 : i32
  %r = or %s, %oe : i32
  ret %r
}

func @cas64(i64, i64) -> i64 {
entry ^0(%exp: i64, %new: i64):
  store i64 -5, @w64 align 8 : i64
  fence seq_cst
  %old = cmpxchg acq_rel acquire @w64, %exp, %new align 8 : i64
  %now = atomic_load seq_cst @w64 align 8 : i64
  %r = add %old, %now : i64
  ret %r
}

func @min16(i16) -> i16 {
entry ^0(%v: i16):
  atomic_store release i16 1000, @w16 align 2 : i16
  %o = atomic_rmw min seq_cst @w16, %v align 2 : i16
  %n = atomic_load seq_cst @w16 align 2 : i16
  %r = sub %n, %o : i16
  ret %r
}
"#;

/// Atomics (threads proposal): native rmw ops, the compare-exchange loops of
/// the others, narrow and 64-bit accesses, compare-exchange and fences.
#[test]
fn atomics() {
    let mut list = Vec::new();
    for op in 0..12u128 {
        for v in [0u128, 7, 0xffff_fff9, 0x8000_0000, 100] {
            list.push(("rmw32", vec![op, v]));
        }
    }
    for op in 0..5u128 {
        for v in [0u128, 1, 0xfd, 0x7f, 0x80, 0xfe] {
            list.push(("rmw8", vec![op, v]));
        }
    }
    for (e, n) in [(0xffff_ffff_ffff_fffbu128, 42u128), (5, 42)] {
        list.push(("cas64", vec![e, n]));
    }
    for v in [0u128, 999, 1001, 0xffff, 0x8000] {
        list.push(("min16", vec![v]));
    }
    let Some(t) = differential("atomics", ATOMICS, &list) else { return no_node("atomics") };
    assert_eq!(t.skipped, 0, "{t:?}");
    // The same calls against the program after the -O2 pipeline.
    let t = differential_optimized("atomics", ATOMICS, &list).expect("node");
    assert_eq!(t.skipped, 0, "{t:?}");
}

pub(super) const WIDE: &str = r#"
module "wide"

func @add128(i128, i128) -> i128 {
entry ^0(%a: i128, %b: i128):
  %r = add %a, %b : i128
  ret %r
}

func @sub128(i128, i128) -> i128 {
entry ^0(%a: i128, %b: i128):
  %r = sub %a, %b : i128
  ret %r
}

func @shifts128(i128, i128) -> i128 {
entry ^0(%a: i128, %s: i128):
  %m = and %s, i128 127 : i128
  %x = shl %a, %m : i128
  %y = lshr %a, %m : i128
  %z = ashr %a, %m : i128
  %xy = xor %x, %y : i128
  %r = or %xy, %z : i128
  ret %r
}

func @cmp128(i128, i128) -> i32 {
entry ^0(%a: i128, %b: i128):
  %lt = icmp slt %a, %b : i1
  %ult = icmp ult %a, %b : i1
  %eq = icmp eq %a, %b : i1
  %l = zext %lt : i32
  %u = zext %ult : i32
  %e = zext %eq : i32
  %u2 = shl %u, i32 1 : i32
  %e2 = shl %e, i32 2 : i32
  %t = or %l, %u2 : i32
  %r = or %t, %e2 : i32
  ret %r
}

func @widen(i64, i32) -> i128 {
entry ^0(%a: i64, %b: i32):
  %x = sext %a : i128
  %y = zext %b : i128
  %s = shl %y, i128 64 : i128
  %r = add %x, %s : i128
  ret %r
}

func @narrow(i128) -> i64 {
entry ^0(%a: i128):
  %h = lshr %a, i128 64 : i128
  %lo = trunc %a : i64
  %hi = trunc %h : i64
  %r = xor %lo, %hi : i64
  ret %r
}

; Wide values through calls and memory.
func @callwide(i128) -> i128 {
entry ^0(%a: i128):
  %s = alloca i128 : ptr
  store %a, %s align 16 : i128
  %l = load %s align 16 : i128
  %r = call @add128(%l, i128 1) : i128
  ret %r
}
"#;

/// `i128` through legalization: arithmetic, variable shifts, comparisons,
/// extension and truncation, and wide parameters and results (two `i64`
/// parameters, a multi-value result).
#[test]
fn wide_integers() {
    let vals: [u128; 8] = [
        0,
        1,
        u128::MAX,
        1 << 64,
        (1 << 64) - 1,
        1 << 127,
        0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        0x8000_0000_0000_0001_0000_0000_ffff_ffff,
    ];
    let mut list = Vec::new();
    for &a in &vals {
        for &b in &vals {
            list.push(("add128", vec![a, b]));
            list.push(("sub128", vec![a, b]));
            list.push(("cmp128", vec![a, b]));
        }
        for s in [0u128, 1, 63, 64, 65, 100, 127] {
            list.push(("shifts128", vec![a, s]));
        }
        list.push(("narrow", vec![a]));
        list.push(("callwide", vec![a]));
        list.push(("widen", vec![a & u128::from(u64::MAX), a >> 96]));
    }
    let Some(t) = differential("wide", WIDE, &list) else { return no_node("wide_integers") };
    assert_eq!(t.skipped, 0, "{t:?}");
    // The same calls against the program after the -O2 pipeline.
    let t = differential_optimized("wide", WIDE, &list).expect("node");
    assert_eq!(t.skipped, 0, "{t:?}");
}
