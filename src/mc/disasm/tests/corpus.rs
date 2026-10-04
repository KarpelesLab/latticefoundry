//! IR programs compiled for every target to give the differential tests
//! realistic objects: integer arithmetic of every width, shifts, compares and
//! selects, loops and switches, direct and indirect calls, globals, stack
//! memory, and floating point with conversions.

/// Integer code.
pub(super) const INTS: &str = r#"
module "ints"

global @counter : i64 = i64 0
global @table : [8 x i32] = [8 x i32] (i32 3, i32 1, i32 4, i32 1, i32 5, i32 9, i32 2, i32 6)

func @ext(i32) -> i32

func @arith64(i64, i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64):
  %m = mul %a, %b : i64
  %s = ashr %a, i64 3 : i64
  %x = xor %m, %s : i64
  %u = lshr %b, %c : i64
  %o = or %x, %u : i64
  %d = sdiv %a, %b : i64
  %r = srem %o, %b : i64
  %q = udiv %d, %c : i64
  %t = sub %r, %q : i64
  %k = and %t, i64 255 : i64
  %l = shl %k, %c : i64
  %n = add %l, i64 -12345 : i64
  ret %n
}

func @arith32(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %m = mul %a, %b : i32
  %s = shl %a, i32 5 : i32
  %x = xor %m, %s : i32
  %u = lshr %x, i32 7 : i32
  %d = udiv %u, %b : i32
  %r = urem %a, %b : i32
  %o = or %d, %r : i32
  %n = sub i32 0, %o : i32
  ret %n
}

func @narrow(i8, i16, i8) -> i32 {
entry ^0(%a: i8, %b: i16, %c: i8):
  %ax = sext %a : i32
  %bx = sext %b : i32
  %cx = zext %c : i32
  %p = mul %ax, %bx : i32
  %q = add %p, %cx : i32
  %t = trunc %q : i16
  %tx = zext %t : i32
  ret %tx
}

func @compare(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %c1 = icmp slt %a, %b : i1
  %c2 = icmp ugt %a, %b : i1
  %c3 = icmp eq %a, i64 42 : i1
  %s1 = select %c1, %a, %b : i64
  %s2 = select %c2, %s1, i64 7 : i64
  %z = zext %c3 : i64
  %r = add %s2, %z : i64
  ret %r
}

func @sum(i32) -> i32 {
entry ^0(%n: i32):
  br ^1(i32 0, i32 0)
^1(%i: i32, %acc: i32):
  %c = icmp ult %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %acc2 = add %acc, %i : i32
  %big = icmp sgt %acc2, i32 1000 : i1
  cond_br %big, ^4, ^5(%acc2)
^4:
  %e = call @ext(%acc2) : i32
  br ^5(%e)
^5(%a3: i32):
  %i2 = add %i, i32 1 : i32
  br ^1(%i2, %a3)
^3:
  ret %acc
}

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

func @bump(i64) -> i64 {
entry ^0(%d: i64):
  %v = load @counter align 8 : i64
  %w = add %v, %d : i64
  store %w, @counter align 8 : i64
  ret %w
}

func @table_at(i32) -> i32 {
entry ^0(%i: i32):
  %o = mul %i, i32 4 : i32
  %p = ptr_add @table, %o : ptr
  %v = load %p align 4 : i32
  ret %v
}

func @frame(i32) -> i32 {
entry ^0(%x: i32):
  %s = alloca [4 x i32] : ptr
  store %x, %s align 4 : i32
  %p = ptr_add %s, i32 8 : ptr
  store i32 77, %p align 4 : i32
  %a = load %s align 4 : i32
  %b = load %p align 4 : i32
  %r = add %a, %b : i32
  ret %r
}

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

func @via_ref(i32) -> i32 {
entry ^0(%x: i32):
  %c = icmp slt %x, i32 0 : i1
  %f = select %c, @neg, @twice : ptr
  %r = call %f(%x) : i32
  %s = call @twice(%r) : i32
  ret %s
}
"#;

/// Floating-point code.
pub(super) const FLOATS: &str = r#"
module "floats"

func @fd(f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64):
  %m = fmul %a, %b : f64
  %d = fdiv %a, %b : f64
  %s = fadd %m, %d : f64
  %i = fptosi %a : i64
  %f = sitofp %i : f64
  %r = fsub %s, %f : f64
  %n = fneg %r : f64
  ret %n
}

func @ff(f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32):
  %m = fmul %a, %b : f32
  %d = fdiv %a, %b : f32
  %s = fsub %m, %d : f32
  %i = fptosi %b : i32
  %f = sitofp %i : f32
  %r = fadd %s, %f : f32
  ret %r
}

func @conv(f32, f64, i64) -> f64 {
entry ^0(%a: f32, %b: f64, %c: i64):
  %e = fpext %a : f64
  %t = fptrunc %b : f32
  %u = fptoui %t : i32
  %w = uitofp %u : f64
  %x = uitofp %c : f64
  %s = fadd %e, %w : f64
  %r = fadd %s, %x : f64
  ret %r
}

func @fcmp(f64, f64) -> i32 {
entry ^0(%a: f64, %b: f64):
  %c1 = fcmp olt %a, %b : i1
  %c2 = fcmp oeq %a, %b : i1
  %c3 = fcmp oge %a, %b : i1
  %x1 = zext %c1 : i32
  %x2 = zext %c2 : i32
  %x3 = zext %c3 : i32
  %y = shl %x2, i32 1 : i32
  %z = shl %x3, i32 2 : i32
  %s = or %x1, %y : i32
  %r = or %s, %z : i32
  ret %r
}
"#;
