//! The AVR runtime library, written in LF IR and compiled by this backend.
//!
//! It provides every helper the backend calls:
//!
//! - integer multiply, division and remainder: `__lf_{mul,udiv,div,umod,mod}_i8`
//!   and `_i16` (the narrow helpers isel calls), and the libgcc-named
//!   `__{mul,udiv,div,umod,mod}si3` / `…di3` that
//!   [`legalize_ints`](crate::codegen::legalize_int::legalize_ints) calls for 32
//!   and 64 bits — shift-and-add and restoring shift-and-subtract loops, all
//!   `T f(T, T)` under the normal calling convention (the signed forms are
//!   written on the unsigned ones; division by zero returns all ones);
//! - IEEE 754 binary32 soft float: `__addsf3`, `__subsf3`, `__mulsf3`,
//!   `__divsf3` (round to nearest, ties to even; subnormals, infinities and
//!   NaN handled; a NaN result is quiet), the comparisons `__eqsf2`, `__nesf2`,
//!   `__ltsf2`, `__lesf2`, `__gtsf2`, `__gesf2`, `__unordsf2` (returning an
//!   `i16`), and the conversions `__fix{,uns}sf{si,di}` (truncating) and
//!   `__float{,un}{si,di}sf`.
//!
//! `f64` helpers (`__adddf3`, ...) and `fmodf` are **not** provided.
//!
//! [`members`] compiles each function as its own object — an archive member —
//! so [`super::link`] pulls in only the helpers a program uses (and what they
//! use in turn). Compiling a helper on its own also keeps the legalization of
//! its body independent of the others: `__mulsf3`'s 64-bit multiply is a call
//! to `__muldi3`, another member.

use crate::ir::{FuncId, Function, Module};
use crate::mc::object::ObjectModule;
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

/// The integer helper names for a width: `(mul, udiv, div, umod, mod)`.
fn int_names(bits: u32) -> [String; 5] {
    use crate::codegen::legalize_int::libgcc_libcall;
    use crate::ir::inst::BinOp;
    [BinOp::Mul, BinOp::UDiv, BinOp::SDiv, BinOp::URem, BinOp::SRem].map(|op| libgcc_libcall(op, bits))
}

/// The integer helpers of one width.
fn int_helpers(bits: u32) -> String {
    let [mul, udiv, div, umod, modn] = int_names(bits);
    let t = format!("i{bits}");
    let unsigned = |name: &str, rem: bool| {
        let out = if rem { "%rr" } else { "%qq" };
        format!(
            r#"
func @{name}({t}, {t}) -> {t} {{
entry ^0(%n: {t}, %d: {t}):
  br ^1(%n, {t} 0, {t} 0, {t} 0)
^1(%x: {t}, %q: {t}, %r: {t}, %i: {t}):
  %top = lshr %x, {t} {m1} : {t}
  %rtop = lshr %r, {t} {m1} : {t}
  %r2 = shl %r, {t} 1 : {t}
  %r3 = or %r2, %top : {t}
  %x2 = shl %x, {t} 1 : {t}
  %q2 = shl %q, {t} 1 : {t}
  %ge0 = icmp uge %r3, %d : i1
  %big = trunc %rtop : i1
  %ge = or %ge0, %big : i1
  %rs = sub %r3, %d : {t}
  %r4 = select %ge, %rs, %r3 : {t}
  %qb = zext %ge : {t}
  %q3 = or %q2, %qb : {t}
  %i2 = add %i, {t} 1 : {t}
  %done = icmp eq %i2, {t} {bits} : i1
  cond_br %done, ^2(%q3, %r4), ^1(%x2, %q3, %r4, %i2)
^2(%qq: {t}, %rr: {t}):
  ret {out}
}}
"#,
            m1 = bits - 1
        )
    };
    let signed = |name: &str, inner: &str, rem: bool| {
        let neg = if rem { "%na" } else { "%nq" };
        format!(
            r#"
func @{name}({t}, {t}) -> {t} {{
entry ^0(%a: {t}, %b: {t}):
  %na = icmp slt %a, {t} 0 : i1
  %nb = icmp slt %b, {t} 0 : i1
  %ma = sub {t} 0, %a : {t}
  %mb = sub {t} 0, %b : {t}
  %ua = select %na, %ma, %a : {t}
  %ub = select %nb, %mb, %b : {t}
  %q = call @{inner}(%ua, %ub) : {t}
  %mq = sub {t} 0, %q : {t}
  %nq = xor %na, %nb : i1
  %r = select {neg}, %mq, %q : {t}
  ret %r
}}
"#
        )
    };
    let mut s = format!(
        r#"
func @{mul}({t}, {t}) -> {t} {{
entry ^0(%a: {t}, %b: {t}):
  br ^1(%a, %b, {t} 0)
^1(%x: {t}, %y: {t}, %acc: {t}):
  %bit = and %y, {t} 1 : {t}
  %t = trunc %bit : i1
  %s = add %acc, %x : {t}
  %acc2 = select %t, %s, %acc : {t}
  %x2 = shl %x, {t} 1 : {t}
  %y2 = lshr %y, {t} 1 : {t}
  %done = icmp eq %y2, {t} 0 : i1
  cond_br %done, ^2(%acc2), ^1(%x2, %y2, %acc2)
^2(%r: {t}):
  ret %r
}}
"#
    );
    s.push_str(&unsigned(&udiv, false));
    s.push_str(&unsigned(&umod, true));
    s.push_str(&signed(&div, &udiv, false));
    s.push_str(&signed(&modn, &umod, true));
    s
}

/// The binary32 soft-float helpers.
const SOFT_FLOAT: &str = r#"
func @__lf_clz32(i32) -> i32 {
entry ^0(%x: i32):
  %h16 = lshr %x, i32 16 : i32
  %c16 = icmp eq %h16, i32 0 : i1
  %s16 = shl %x, i32 16 : i32
  %x1 = select %c16, %s16, %x : i32
  %n1 = select %c16, i32 16, i32 0 : i32
  %h8 = lshr %x1, i32 24 : i32
  %c8 = icmp eq %h8, i32 0 : i1
  %s8 = shl %x1, i32 8 : i32
  %x2 = select %c8, %s8, %x1 : i32
  %a8 = select %c8, i32 8, i32 0 : i32
  %n2 = add %n1, %a8 : i32
  %h4 = lshr %x2, i32 28 : i32
  %c4 = icmp eq %h4, i32 0 : i1
  %s4 = shl %x2, i32 4 : i32
  %x3 = select %c4, %s4, %x2 : i32
  %a4 = select %c4, i32 4, i32 0 : i32
  %n3 = add %n2, %a4 : i32
  %h2 = lshr %x3, i32 30 : i32
  %c2 = icmp eq %h2, i32 0 : i1
  %s2 = shl %x3, i32 2 : i32
  %x4 = select %c2, %s2, %x3 : i32
  %a2 = select %c2, i32 2, i32 0 : i32
  %n4 = add %n3, %a2 : i32
  %h1 = lshr %x4, i32 31 : i32
  %c1 = icmp eq %h1, i32 0 : i1
  %a1 = zext %c1 : i32
  %n5 = add %n4, %a1 : i32
  %z = icmp eq %x, i32 0 : i1
  %r = select %z, i32 32, %n5 : i32
  ret %r
}

func @__lf_shr_sticky32(i32, i32) -> i32 {
entry ^0(%x: i32, %n: i32):
  %big = icmp uge %n, i32 32 : i1
  cond_br %big, ^1, ^2
^1:
  %nz = icmp ne %x, i32 0 : i1
  %r = zext %nz : i32
  ret %r
^2:
  %z = icmp eq %n, i32 0 : i1
  cond_br %z, ^3, ^4
^3:
  ret %x
^4:
  %s = lshr %x, %n : i32
  %m = sub i32 32, %n : i32
  %lost = shl %x, %m : i32
  %st = icmp ne %lost, i32 0 : i1
  %stz = zext %st : i32
  %r2 = or %s, %stz : i32
  ret %r2
}

func @__lf_round_pack(i32, i32, i32) -> i32 {
entry ^0(%s: i32, %e: i32, %m: i32):
  %ovf = icmp sge %e, i32 255 : i1
  cond_br %ovf, ^1, ^2
^1:
  %inf = or %s, i32 0x7f800000 : i32
  ret %inf
^2:
  %den = icmp sle %e, i32 0 : i1
  cond_br %den, ^3, ^4(%e, %m)
^3:
  %sh = sub i32 1, %e : i32
  %m2 = call @__lf_shr_sticky32(%m, %sh) : i32
  br ^4(i32 0, %m2)
^4(%e2: i32, %m3: i32):
  %rgs = and %m3, i32 7 : i32
  %frac = lshr %m3, i32 3 : i32
  %fr = and %frac, i32 0x7fffff : i32
  %ex = shl %e2, i32 23 : i32
  %r0 = or %fr, %ex : i32
  %r1 = or %r0, %s : i32
  %up = icmp ugt %rgs, i32 4 : i1
  %half = icmp eq %rgs, i32 4 : i1
  %odd0 = and %r1, i32 1 : i32
  %odd = trunc %odd0 : i1
  %tie = and %half, %odd : i1
  %inc = or %up, %tie : i1
  %incv = zext %inc : i32
  %r = add %r1, %incv : i32
  ret %r
}

func @__lf_sf_exp(i32) -> i32 {
entry ^0(%a: i32):
  %e = lshr %a, i32 23 : i32
  %e8 = and %e, i32 0xff : i32
  %z = icmp eq %e8, i32 0 : i1
  cond_br %z, ^1, ^2
^2:
  ret %e8
^1:
  %m = and %a, i32 0x7fffff : i32
  %lz = call @__lf_clz32(%m) : i32
  %sh = sub %lz, i32 8 : i32
  %r = sub i32 1, %sh : i32
  ret %r
}

func @__lf_sf_sig(i32) -> i32 {
entry ^0(%a: i32):
  %m = and %a, i32 0x7fffff : i32
  %e = and %a, i32 0x7f800000 : i32
  %z = icmp eq %e, i32 0 : i1
  cond_br %z, ^1, ^2
^2:
  %r = or %m, i32 0x800000 : i32
  ret %r
^1:
  %lz = call @__lf_clz32(%m) : i32
  %sh = sub %lz, i32 8 : i32
  %r2 = shl %m, %sh : i32
  ret %r2
}

func @__lf_addsf_special(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %aa = and %a, i32 0x7fffffff : i32
  %ba = and %b, i32 0x7fffffff : i32
  %an = icmp ugt %aa, i32 0x7f800000 : i1
  cond_br %an, ^1, ^2
^1:
  %qa = or %a, i32 0x400000 : i32
  ret %qa
^2:
  %bn = icmp ugt %ba, i32 0x7f800000 : i1
  cond_br %bn, ^3, ^4
^3:
  %qb = or %b, i32 0x400000 : i32
  ret %qb
^4:
  %ai = icmp eq %aa, i32 0x7f800000 : i1
  cond_br %ai, ^5, ^6
^5:
  %bi = icmp eq %ba, i32 0x7f800000 : i1
  %x = xor %a, %b : i32
  %opp = icmp ne %x, i32 0 : i1
  %nan = and %bi, %opp : i1
  %r5 = select %nan, i32 0x7fc00000, %a : i32
  ret %r5
^6:
  %bi2 = icmp eq %ba, i32 0x7f800000 : i1
  cond_br %bi2, ^7, ^8
^7:
  ret %b
^8:
  %az = icmp eq %aa, i32 0 : i1
  %bz = icmp eq %ba, i32 0 : i1
  %both = and %a, %b : i32
  %r8 = select %bz, %both, %b : i32
  %r9 = select %az, %r8, %a : i32
  ret %r9
}

func @__addsf3(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %aa = and %a, i32 0x7fffffff : i32
  %ba = and %b, i32 0x7fffffff : i32
  %am = sub %aa, i32 1 : i32
  %bm = sub %ba, i32 1 : i32
  %asp = icmp uge %am, i32 0x7f7fffff : i1
  %bsp = icmp uge %bm, i32 0x7f7fffff : i1
  %sp = or %asp, %bsp : i1
  cond_br %sp, ^1, ^2
^1:
  %r = call @__lf_addsf_special(%a, %b) : i32
  ret %r
^2:
  %sw = icmp ugt %ba, %aa : i1
  %x = select %sw, %b, %a : i32
  %y = select %sw, %a, %b : i32
  %xe = call @__lf_sf_exp(%x) : i32
  %ye = call @__lf_sf_exp(%y) : i32
  %xs = call @__lf_sf_sig(%x) : i32
  %ys = call @__lf_sf_sig(%y) : i32
  %xs3 = shl %xs, i32 3 : i32
  %ys3 = shl %ys, i32 3 : i32
  %al = sub %xe, %ye : i32
  %ys4 = call @__lf_shr_sticky32(%ys3, %al) : i32
  %sign = and %x, i32 0x80000000 : i32
  %ab = xor %a, %b : i32
  %subtract = icmp slt %ab, i32 0 : i1
  cond_br %subtract, ^3, ^4
^3:
  %d = sub %xs3, %ys4 : i32
  %dz = icmp eq %d, i32 0 : i1
  cond_br %dz, ^5, ^6
^5:
  ret i32 0
^6:
  %lz = call @__lf_clz32(%d) : i32
  %sh = sub %lz, i32 5 : i32
  %d2 = shl %d, %sh : i32
  %e2 = sub %xe, %sh : i32
  %r2 = call @__lf_round_pack(%sign, %e2, %d2) : i32
  ret %r2
^4:
  %s = add %xs3, %ys4 : i32
  %ov = icmp uge %s, i32 0x8000000 : i1
  %s1 = lshr %s, i32 1 : i32
  %lsb = and %s, i32 1 : i32
  %s2 = or %s1, %lsb : i32
  %s3 = select %ov, %s2, %s : i32
  %one = zext %ov : i32
  %e3 = add %xe, %one : i32
  %r3 = call @__lf_round_pack(%sign, %e3, %s3) : i32
  ret %r3
}

func @__subsf3(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %nb = xor %b, i32 0x80000000 : i32
  %r = call @__addsf3(%a, %nb) : i32
  ret %r
}

func @__lf_mulsf_special(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %s0 = xor %a, %b : i32
  %sign = and %s0, i32 0x80000000 : i32
  %aa = and %a, i32 0x7fffffff : i32
  %ba = and %b, i32 0x7fffffff : i32
  %an = icmp ugt %aa, i32 0x7f800000 : i1
  cond_br %an, ^1, ^2
^1:
  %qa = or %a, i32 0x400000 : i32
  ret %qa
^2:
  %bn = icmp ugt %ba, i32 0x7f800000 : i1
  cond_br %bn, ^3, ^4
^3:
  %qb = or %b, i32 0x400000 : i32
  ret %qb
^4:
  %ai = icmp eq %aa, i32 0x7f800000 : i1
  %bi = icmp eq %ba, i32 0x7f800000 : i1
  %az = icmp eq %aa, i32 0 : i1
  %bz = icmp eq %ba, i32 0 : i1
  %i1 = and %ai, %bz : i1
  %i2 = and %bi, %az : i1
  %inv = or %i1, %i2 : i1
  %anyinf = or %ai, %bi : i1
  %infr = or %sign, i32 0x7f800000 : i32
  %r0 = select %anyinf, %infr, %sign : i32
  %r = select %inv, i32 0x7fc00000, %r0 : i32
  ret %r
}

func @__mulsf3(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %s0 = xor %a, %b : i32
  %sign = and %s0, i32 0x80000000 : i32
  %aa = and %a, i32 0x7fffffff : i32
  %ba = and %b, i32 0x7fffffff : i32
  %am = sub %aa, i32 1 : i32
  %bm = sub %ba, i32 1 : i32
  %asp = icmp uge %am, i32 0x7f7fffff : i1
  %bsp = icmp uge %bm, i32 0x7f7fffff : i1
  %sp = or %asp, %bsp : i1
  cond_br %sp, ^1, ^2
^1:
  %r = call @__lf_mulsf_special(%a, %b) : i32
  ret %r
^2:
  %ae = call @__lf_sf_exp(%a) : i32
  %be = call @__lf_sf_exp(%b) : i32
  %as = call @__lf_sf_sig(%a) : i32
  %bs = call @__lf_sf_sig(%b) : i32
  %a64 = zext %as : i64
  %b64 = zext %bs : i64
  %p = mul %a64, %b64 : i64
  %hi = lshr %p, i64 47 : i64
  %hb = trunc %hi : i1
  %e0 = add %ae, %be : i32
  %e1 = sub %e0, i32 127 : i32
  %inc = zext %hb : i32
  %e = add %e1, %inc : i32
  %q21 = lshr %p, i64 21 : i64
  %q20 = lshr %p, i64 20 : i64
  %q = select %hb, %q21, %q20 : i64
  %l21 = and %p, i64 0x1fffff : i64
  %l20 = and %p, i64 0xfffff : i64
  %lost = select %hb, %l21, %l20 : i64
  %st = icmp ne %lost, i64 0 : i1
  %st32 = zext %st : i32
  %q32 = trunc %q : i32
  %m = or %q32, %st32 : i32
  %r2 = call @__lf_round_pack(%sign, %e, %m) : i32
  ret %r2
}

func @__lf_divsf_special(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %s0 = xor %a, %b : i32
  %sign = and %s0, i32 0x80000000 : i32
  %aa = and %a, i32 0x7fffffff : i32
  %ba = and %b, i32 0x7fffffff : i32
  %an = icmp ugt %aa, i32 0x7f800000 : i1
  cond_br %an, ^1, ^2
^1:
  %qa = or %a, i32 0x400000 : i32
  ret %qa
^2:
  %bn = icmp ugt %ba, i32 0x7f800000 : i1
  cond_br %bn, ^3, ^4
^3:
  %qb = or %b, i32 0x400000 : i32
  ret %qb
^4:
  %ai = icmp eq %aa, i32 0x7f800000 : i1
  %bi = icmp eq %ba, i32 0x7f800000 : i1
  %az = icmp eq %aa, i32 0 : i1
  %bz = icmp eq %ba, i32 0 : i1
  %i1 = and %ai, %bi : i1
  %i2 = and %az, %bz : i1
  %inv = or %i1, %i2 : i1
  %toinf = or %ai, %bz : i1
  %infr = or %sign, i32 0x7f800000 : i32
  %r0 = select %toinf, %infr, %sign : i32
  %r = select %inv, i32 0x7fc00000, %r0 : i32
  ret %r
}

func @__divsf3(i32, i32) -> i32 {
entry ^0(%a: i32, %b: i32):
  %s0 = xor %a, %b : i32
  %sign = and %s0, i32 0x80000000 : i32
  %aa = and %a, i32 0x7fffffff : i32
  %ba = and %b, i32 0x7fffffff : i32
  %am = sub %aa, i32 1 : i32
  %bm = sub %ba, i32 1 : i32
  %asp = icmp uge %am, i32 0x7f7fffff : i1
  %bsp = icmp uge %bm, i32 0x7f7fffff : i1
  %sp = or %asp, %bsp : i1
  cond_br %sp, ^1, ^2
^1:
  %r = call @__lf_divsf_special(%a, %b) : i32
  ret %r
^2:
  %ae = call @__lf_sf_exp(%a) : i32
  %be = call @__lf_sf_exp(%b) : i32
  %as = call @__lf_sf_sig(%a) : i32
  %bs = call @__lf_sf_sig(%b) : i32
  %lt = icmp ult %as, %bs : i1
  %as2 = shl %as, i32 1 : i32
  %r0 = select %lt, %as2, %as : i32
  %e0 = sub %ae, %be : i32
  %e1 = add %e0, i32 127 : i32
  %dec = zext %lt : i32
  %e = sub %e1, %dec : i32
  br ^3(%r0, i32 0, i32 0)
^3(%rr: i32, %q: i32, %i: i32):
  %ge = icmp uge %rr, %bs : i1
  %rs = sub %rr, %bs : i32
  %r1 = select %ge, %rs, %rr : i32
  %q1 = shl %q, i32 1 : i32
  %qb = zext %ge : i32
  %q2 = or %q1, %qb : i32
  %r2 = shl %r1, i32 1 : i32
  %i2 = add %i, i32 1 : i32
  %done = icmp eq %i2, i32 27 : i1
  cond_br %done, ^4(%q2, %r2), ^3(%r2, %q2, %i2)
^4(%qq: i32, %rem: i32):
  %st = icmp ne %rem, i32 0 : i1
  %st32 = zext %st : i32
  %m = or %qq, %st32 : i32
  %res = call @__lf_round_pack(%sign, %e, %m) : i32
  ret %res
}

func @__lf_cmpsf(i32, i32) -> i16 {
entry ^0(%a: i32, %b: i32):
  %aa = and %a, i32 0x7fffffff : i32
  %ba = and %b, i32 0x7fffffff : i32
  %an = icmp ugt %aa, i32 0x7f800000 : i1
  %bn = icmp ugt %ba, i32 0x7f800000 : i1
  %un = or %an, %bn : i1
  cond_br %un, ^1, ^2
^1:
  ret i16 2
^2:
  %o = or %aa, %ba : i32
  %z = icmp eq %o, i32 0 : i1
  cond_br %z, ^3, ^4
^3:
  ret i16 0
^4:
  %n = and %a, %b : i32
  %pos = icmp sge %n, i32 0 : i1
  %lt = icmp slt %a, %b : i1
  %gt = icmp sgt %a, %b : i1
  %eq = icmp eq %a, %b : i1
  %less = select %pos, %lt, %gt : i1
  %m1 = select %less, i16 -1, i16 1 : i16
  %r = select %eq, i16 0, %m1 : i16
  ret %r
}

func @__eqsf2(i32, i32) -> i16 {
entry ^0(%a: i32, %b: i32):
  %c = call @__lf_cmpsf(%a, %b) : i16
  %u = icmp eq %c, i16 2 : i1
  %r = select %u, i16 1, %c : i16
  ret %r
}

func @__nesf2(i32, i32) -> i16 {
entry ^0(%a: i32, %b: i32):
  %c = call @__lf_cmpsf(%a, %b) : i16
  %u = icmp eq %c, i16 2 : i1
  %r = select %u, i16 1, %c : i16
  ret %r
}

func @__ltsf2(i32, i32) -> i16 {
entry ^0(%a: i32, %b: i32):
  %c = call @__lf_cmpsf(%a, %b) : i16
  %u = icmp eq %c, i16 2 : i1
  %r = select %u, i16 1, %c : i16
  ret %r
}

func @__lesf2(i32, i32) -> i16 {
entry ^0(%a: i32, %b: i32):
  %c = call @__lf_cmpsf(%a, %b) : i16
  %u = icmp eq %c, i16 2 : i1
  %r = select %u, i16 1, %c : i16
  ret %r
}

func @__gtsf2(i32, i32) -> i16 {
entry ^0(%a: i32, %b: i32):
  %c = call @__lf_cmpsf(%a, %b) : i16
  %u = icmp eq %c, i16 2 : i1
  %r = select %u, i16 -1, %c : i16
  ret %r
}

func @__gesf2(i32, i32) -> i16 {
entry ^0(%a: i32, %b: i32):
  %c = call @__lf_cmpsf(%a, %b) : i16
  %u = icmp eq %c, i16 2 : i1
  %r = select %u, i16 -1, %c : i16
  ret %r
}

func @__unordsf2(i32, i32) -> i16 {
entry ^0(%a: i32, %b: i32):
  %c = call @__lf_cmpsf(%a, %b) : i16
  %u = icmp eq %c, i16 2 : i1
  %r = zext %u : i16
  ret %r
}

func @__fixsfsi(i32) -> i32 {
entry ^0(%a: i32):
  %e0 = lshr %a, i32 23 : i32
  %e = and %e0, i32 0xff : i32
  %small = icmp ult %e, i32 127 : i1
  cond_br %small, ^1, ^2
^1:
  ret i32 0
^2:
  %sh = sub %e, i32 127 : i32
  %big = icmp uge %sh, i32 31 : i1
  cond_br %big, ^3, ^4
^3:
  ret i32 0x80000000
^4:
  %m0 = and %a, i32 0x7fffff : i32
  %m = or %m0, i32 0x800000 : i32
  %left = icmp uge %sh, i32 23 : i1
  cond_br %left, ^5, ^6
^5:
  %ls = sub %sh, i32 23 : i32
  %v = shl %m, %ls : i32
  br ^7(%v)
^6:
  %rs = sub i32 23, %sh : i32
  %v2 = lshr %m, %rs : i32
  br ^7(%v2)
^7(%v3: i32):
  %neg = icmp slt %a, i32 0 : i1
  %nv = sub i32 0, %v3 : i32
  %r = select %neg, %nv, %v3 : i32
  ret %r
}

func @__fixunssfsi(i32) -> i32 {
entry ^0(%a: i32):
  %e0 = lshr %a, i32 23 : i32
  %e = and %e0, i32 0xff : i32
  %small = icmp ult %e, i32 127 : i1
  %neg = icmp slt %a, i32 0 : i1
  %zero = or %small, %neg : i1
  cond_br %zero, ^1, ^2
^1:
  ret i32 0
^2:
  %sh = sub %e, i32 127 : i32
  %big = icmp uge %sh, i32 32 : i1
  cond_br %big, ^3, ^4
^3:
  ret i32 0xffffffff
^4:
  %m0 = and %a, i32 0x7fffff : i32
  %m = or %m0, i32 0x800000 : i32
  %left = icmp uge %sh, i32 23 : i1
  cond_br %left, ^5, ^6
^5:
  %ls = sub %sh, i32 23 : i32
  %v = shl %m, %ls : i32
  ret %v
^6:
  %rs = sub i32 23, %sh : i32
  %v2 = lshr %m, %rs : i32
  ret %v2
}

func @__fixsfdi(i32) -> i64 {
entry ^0(%a: i32):
  %e0 = lshr %a, i32 23 : i32
  %e = and %e0, i32 0xff : i32
  %small = icmp ult %e, i32 127 : i1
  cond_br %small, ^1, ^2
^1:
  ret i64 0
^2:
  %sh = sub %e, i32 127 : i32
  %big = icmp uge %sh, i32 63 : i1
  cond_br %big, ^3, ^4
^3:
  ret i64 0x8000000000000000
^4:
  %m0 = and %a, i32 0x7fffff : i32
  %m1 = or %m0, i32 0x800000 : i32
  %m = zext %m1 : i64
  %left = icmp uge %sh, i32 23 : i1
  cond_br %left, ^5, ^6
^5:
  %ls = sub %sh, i32 23 : i32
  %ls64 = zext %ls : i64
  %v = shl %m, %ls64 : i64
  br ^7(%v)
^6:
  %rs = sub i32 23, %sh : i32
  %rs64 = zext %rs : i64
  %v2 = lshr %m, %rs64 : i64
  br ^7(%v2)
^7(%v3: i64):
  %neg = icmp slt %a, i32 0 : i1
  %nv = sub i64 0, %v3 : i64
  %r = select %neg, %nv, %v3 : i64
  ret %r
}

func @__fixunssfdi(i32) -> i64 {
entry ^0(%a: i32):
  %e0 = lshr %a, i32 23 : i32
  %e = and %e0, i32 0xff : i32
  %small = icmp ult %e, i32 127 : i1
  %neg = icmp slt %a, i32 0 : i1
  %zero = or %small, %neg : i1
  cond_br %zero, ^1, ^2
^1:
  ret i64 0
^2:
  %sh = sub %e, i32 127 : i32
  %big = icmp uge %sh, i32 64 : i1
  cond_br %big, ^3, ^4
^3:
  ret i64 0xffffffffffffffff
^4:
  %m0 = and %a, i32 0x7fffff : i32
  %m1 = or %m0, i32 0x800000 : i32
  %m = zext %m1 : i64
  %left = icmp uge %sh, i32 23 : i1
  cond_br %left, ^5, ^6
^5:
  %ls = sub %sh, i32 23 : i32
  %ls64 = zext %ls : i64
  %v = shl %m, %ls64 : i64
  ret %v
^6:
  %rs = sub i32 23, %sh : i32
  %rs64 = zext %rs : i64
  %v2 = lshr %m, %rs64 : i64
  ret %v2
}

func @__floatunsisf(i32) -> i32 {
entry ^0(%x: i32):
  %z = icmp eq %x, i32 0 : i1
  cond_br %z, ^1, ^2
^1:
  ret i32 0
^2:
  %lz = call @__lf_clz32(%x) : i32
  %msb = sub i32 31, %lz : i32
  %e = add %msb, i32 127 : i32
  %big = icmp ugt %msb, i32 26 : i1
  cond_br %big, ^3, ^4
^3:
  %sh = sub %msb, i32 26 : i32
  %m = call @__lf_shr_sticky32(%x, %sh) : i32
  br ^5(%m)
^4:
  %sh2 = sub i32 26, %msb : i32
  %m2 = shl %x, %sh2 : i32
  br ^5(%m2)
^5(%m3: i32):
  %r = call @__lf_round_pack(i32 0, %e, %m3) : i32
  ret %r
}

func @__floatsisf(i32) -> i32 {
entry ^0(%x: i32):
  %neg = icmp slt %x, i32 0 : i1
  %nx = sub i32 0, %x : i32
  %ax = select %neg, %nx, %x : i32
  %u = call @__floatunsisf(%ax) : i32
  %s = select %neg, i32 0x80000000, i32 0 : i32
  %r = or %u, %s : i32
  ret %r
}

func @__floatundisf(i64) -> i32 {
entry ^0(%x: i64):
  %hi = lshr %x, i64 32 : i64
  %hi32 = trunc %hi : i32
  %hz = icmp eq %hi32, i32 0 : i1
  cond_br %hz, ^1, ^2
^1:
  %lo = trunc %x : i32
  %r = call @__floatunsisf(%lo) : i32
  ret %r
^2:
  %lz = call @__lf_clz32(%hi32) : i32
  %msb = sub i32 63, %lz : i32
  %e = add %msb, i32 127 : i32
  %sh = sub %msb, i32 26 : i32
  %sh64 = zext %sh : i64
  %q = lshr %x, %sh64 : i64
  %one = shl i64 1, %sh64 : i64
  %mask = sub %one, i64 1 : i64
  %lost = and %x, %mask : i64
  %st = icmp ne %lost, i64 0 : i1
  %st32 = zext %st : i32
  %q32 = trunc %q : i32
  %m = or %q32, %st32 : i32
  %r2 = call @__lf_round_pack(i32 0, %e, %m) : i32
  ret %r2
}

func @__floatdisf(i64) -> i32 {
entry ^0(%x: i64):
  %neg = icmp slt %x, i64 0 : i1
  %nx = sub i64 0, %x : i64
  %ax = select %neg, %nx, %x : i64
  %u = call @__floatundisf(%ax) : i32
  %s = select %neg, i32 0x80000000, i32 0 : i32
  %r = or %u, %s : i32
  ret %r
}
"#;

/// The runtime as one LF IR module text (the AVR data layout).
pub fn source(device: &super::Device) -> String {
    let mut s = format!("module \"avr_rt\"\ntarget \"avr\"\ndatalayout \"{}\"\n", super::data_layout().to_spec());
    for bits in [8, 16, 32, 64] {
        s.push_str(&int_helpers(bits));
    }
    s.push_str(&decimal(SOFT_FLOAT));
    let _ = device;
    s
}

/// `src` with every `0x…` literal written in decimal (operand constants in
/// the text form are decimal; hex keeps the float masks readable here).
fn decimal(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find("0x") {
        out.push_str(&rest[..i]);
        let hex: String = rest[i + 2..].chars().take_while(char::is_ascii_hexdigit).collect();
        let v = u64::from_str_radix(&hex, 16).expect("a hex literal");
        out.push_str(&v.to_string());
        rest = &rest[i + 2 + hex.len()..];
    }
    out.push_str(rest);
    out
}

/// The parsed runtime module.
///
/// # Panics
///
/// If the runtime source does not parse or verify (a bug).
pub fn module(device: &super::Device, syms: &mut StrInterner) -> Module {
    let m = crate::ir::text::parse_module(&source(device), FileId::new(0), syms)
        .unwrap_or_else(|e| panic!("the AVR runtime does not parse: {e:?}"));
    if let Err(e) = crate::verify::verify_module(&m) {
        panic!("the AVR runtime does not verify: {e:?}");
    }
    m
}

/// The runtime compiled for `device`, one object per function (see the
/// [module docs](self)).
pub fn members(device: &super::Device) -> Vec<ObjectModule> {
    compiled(device).into_iter().map(|c| c.object).collect()
}

/// Like [`members`], with each member's stack-usage report (for a
/// whole-program stack bound through the helpers).
///
/// The result depends only on the device's core (the multiplier), so it is
/// compiled once per core and process and cloned afterwards.
pub fn compiled(device: &super::Device) -> Vec<crate::codegen::CompiledModule> {
    use std::sync::{Mutex, OnceLock};
    type Cache = Mutex<Vec<(bool, Vec<crate::codegen::CompiledModule>)>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    if let Some((_, c)) = cache.lock().expect("the runtime cache").iter().find(|(m, _)| *m == device.has_mul) {
        return c.clone();
    }
    let c = compile_members(device);
    cache.lock().expect("the runtime cache").push((device.has_mul, c.clone()));
    c
}

fn compile_members(device: &super::Device) -> Vec<crate::codegen::CompiledModule> {
    let mut syms = StrInterner::new();
    let m = module(device, &mut syms);
    let mut out = Vec::new();
    for fi in 0..m.function_count() {
        if m.function(FuncId::from_index(fi)).is_declaration() {
            continue;
        }
        let (mut one, s) = super::prepare::copy_module(&m, &syms).unwrap_or_else(|e| panic!("{e}"));
        for other in 0..one.function_count() {
            if other != fi {
                let f = one.function(FuncId::from_index(other));
                let decl = Function::new(f.name, f.sig);
                one.replace_function(FuncId::from_index(other), decl);
            }
        }
        let opts = crate::codegen::CodegenOptions::default();
        out.push(super::encode::compile_module_for_device(&one, &s, &opts, device));
    }
    out
}
