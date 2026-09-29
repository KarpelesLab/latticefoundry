//! Shared test fixtures for the per-target atomic tests: an independent oracle
//! for every `atomic_rmw` operation (the standard library's own atomics), and
//! `.lf` programs that check each operation's old value and final memory.

/// One `atomic_rmw` check: `(bytes, op, init, v, old, new)` — the access width
/// in bytes, the IR operation name, the initial memory value and the operand
/// (as unsigned bit patterns of the width), and the expected returned old
/// value and final memory value.
pub(crate) type RmwCase = (u32, &'static str, u64, u64, u64, u64);

/// The IR spelling of the integer type of a byte width.
pub(crate) fn ty_name(bytes: u32) -> &'static str {
    match bytes {
        1 => "i8",
        2 => "i16",
        4 => "i32",
        _ => "i64",
    }
}

/// Every `atomic_rmw` operation on three `(init, v)` pairs straddling the sign
/// bit and the wrap-around of a `bytes`-wide integer, with the expected old
/// and new values computed by the standard library's atomics
/// (`swap`/`fetch_add`/`fetch_nand`/`fetch_max`/...) on the same bit patterns.
pub(crate) fn rmw_cases(bytes: u32) -> Vec<RmwCase> {
    use std::sync::atomic::Ordering::SeqCst;
    use std::sync::atomic::{AtomicI64, AtomicU64};
    let bits = 8 * bytes;
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let sext = |x: u64| -> i64 { ((x << (64 - bits)) as i64) >> (64 - bits) };
    let top = 1u64 << (bits - 1);
    let pairs = [(top - 1, top + 1), (5 & mask, mask - 5), (mask, 3)];
    let mut out = Vec::new();
    for (init, v) in pairs {
        // Unsigned ops run on the zero-extended patterns, masked back.
        let u = |f: &dyn Fn(&AtomicU64, u64) -> u64| {
            let a = AtomicU64::new(init);
            let old = f(&a, v);
            (old & mask, a.load(SeqCst) & mask)
        };
        // Signed min/max run on the sign-extended values.
        let i = |f: &dyn Fn(&AtomicI64, i64) -> i64| {
            let a = AtomicI64::new(sext(init));
            let old = f(&a, sext(v));
            (old as u64 & mask, a.load(SeqCst) as u64 & mask)
        };
        let table: [(&'static str, (u64, u64)); 11] = [
            ("xchg", u(&|a, v| a.swap(v, SeqCst))),
            ("add", u(&|a, v| a.fetch_add(v, SeqCst))),
            ("sub", u(&|a, v| a.fetch_sub(v, SeqCst))),
            ("and", u(&|a, v| a.fetch_and(v, SeqCst))),
            ("nand", u(&|a, v| a.fetch_nand(v, SeqCst))),
            ("or", u(&|a, v| a.fetch_or(v, SeqCst))),
            ("xor", u(&|a, v| a.fetch_xor(v, SeqCst))),
            ("max", i(&|a, v| a.fetch_max(v, SeqCst))),
            ("min", i(&|a, v| a.fetch_min(v, SeqCst))),
            ("umax", u(&|a, v| a.fetch_max(v, SeqCst))),
            ("umin", u(&|a, v| a.fetch_min(v, SeqCst))),
        ];
        for (op, (old, new)) in table {
            out.push((bytes, op, init, v, old, new));
        }
    }
    out
}

/// A `main() -> i64` running every case on its own stack slot (so machine
/// interpreters without a model of globals can run it): store `init`, apply
/// the rmw, check the returned old value and the reloaded memory. Returns the
/// 1-based index of the last failing case, or 0.
pub(crate) fn rmw_slot_program(cases: &[RmwCase]) -> String {
    let mut s = String::from("module \"rmw\"\nfunc @main() -> i64 {\nentry ^0:\n  %f0 = add i64 0, i64 0 : i64\n");
    for (k, &(bytes, op, init, v, old, new)) in cases.iter().enumerate() {
        let t = ty_name(bytes);
        s += &format!("  %p{k} = alloca {t} : ptr\n");
        s += &format!("  store {t} {init}, %p{k} align {bytes} : {t}\n");
        s += &format!("  %old{k} = atomic_rmw {op} seq_cst %p{k}, {t} {v} align {bytes} : {t}\n");
        s += &format!("  %new{k} = load %p{k} align {bytes} : {t}\n");
        s += &format!("  %eo{k} = icmp ne %old{k}, {t} {old} : i1\n");
        s += &format!("  %en{k} = icmp ne %new{k}, {t} {new} : i1\n");
        s += &format!("  %bad{k} = or %eo{k}, %en{k} : i1\n");
        s += &format!("  %f{} = select %bad{k}, i64 {}, %f{k} : i64\n", k + 1, k + 1);
    }
    s += &format!("  ret %f{}\n}}\n", cases.len());
    s
}

/// A `main() -> i64` checking `cmpxchg` success and failure at every integer
/// width on stack slots: returns a bitmask of failed checks (0 = all passed).
pub(crate) const CMPXCHG_SLOTS: &str = r#"
module "cas"
func @check8() -> i64 {
entry ^0:
  %p = alloca i8 : ptr
  store i8 -3, %p align 1 : i8
  %o1 = cmpxchg seq_cst seq_cst %p, i8 -3, i8 100 align 1 : i8
  %o2 = cmpxchg acquire relaxed %p, i8 4, i8 -3 align 1 : i8
  %v = load %p align 1 : i8
  %c1 = icmp eq %o1, i8 -3 : i1
  %c2 = icmp eq %o2, i8 100 : i1
  %c3 = icmp eq %v, i8 100 : i1
  %a = and %c1, %c2 : i1
  %b = and %a, %c3 : i1
  %r = select %b, i64 0, i64 1 : i64
  ret %r
}
func @check16() -> i64 {
entry ^0:
  %p = alloca i16 : ptr
  store i16 -300, %p align 2 : i16
  %o1 = cmpxchg acq_rel acquire %p, i16 -300, i16 3000 align 2 : i16
  %o2 = cmpxchg seq_cst seq_cst %p, i16 7, i16 1 align 2 : i16
  %v = load %p align 2 : i16
  %c1 = icmp eq %o1, i16 -300 : i1
  %c2 = icmp eq %o2, i16 3000 : i1
  %c3 = icmp eq %v, i16 3000 : i1
  %a = and %c1, %c2 : i1
  %b = and %a, %c3 : i1
  %r = select %b, i64 0, i64 2 : i64
  ret %r
}
func @check32() -> i64 {
entry ^0:
  %p = alloca i32 : ptr
  store i32 10, %p align 4 : i32
  %o1 = cmpxchg release relaxed %p, i32 10, i32 -20 align 4 : i32
  %o2 = cmpxchg relaxed relaxed %p, i32 11, i32 5 align 4 : i32
  %v = load %p align 4 : i32
  %c1 = icmp eq %o1, i32 10 : i1
  %c2 = icmp eq %o2, i32 -20 : i1
  %c3 = icmp eq %v, i32 -20 : i1
  %a = and %c1, %c2 : i1
  %b = and %a, %c3 : i1
  %r = select %b, i64 0, i64 4 : i64
  ret %r
}
func @check64() -> i64 {
entry ^0:
  %p = alloca i64 : ptr
  store i64 -1, %p align 8 : i64
  %o1 = cmpxchg seq_cst acquire %p, i64 -1, i64 123456789012 align 8 : i64
  %o2 = cmpxchg seq_cst acquire %p, i64 1, i64 2 align 8 : i64
  %v = load %p align 8 : i64
  %c1 = icmp eq %o1, i64 -1 : i1
  %c2 = icmp eq %o2, i64 123456789012 : i1
  %c3 = icmp eq %v, i64 123456789012 : i1
  %a = and %c1, %c2 : i1
  %b = and %a, %c3 : i1
  %r = select %b, i64 0, i64 8 : i64
  ret %r
}
func @main() -> i64 {
entry ^0:
  %a = call @check8() : i64
  %b = call @check16() : i64
  %c = call @check32() : i64
  %d = call @check64() : i64
  %ab = or %a, %b : i64
  %cd = or %c, %d : i64
  %r = or %ab, %cd : i64
  ret %r
}
"#;

/// A function using every atomic form (for "it compiles and encodes" checks on
/// each backend): `f(ptr, i64) -> i64`.
pub(crate) const ALL_FORMS: &str = r#"
module "forms"
func @f(ptr, i64) -> i64 {
entry ^0(%p: ptr, %v: i64):
  %a = atomic_load relaxed %p align 8 : i64
  %b = atomic_load acquire %p align 8 : i64
  %c = atomic_load seq_cst %p align 8 : i64
  atomic_store relaxed %v, %p align 8 : i64
  atomic_store release %v, %p align 8 : i64
  atomic_store seq_cst %v, %p align 8 : i64
  fence acquire
  fence release
  fence acq_rel
  fence seq_cst
  %x1 = atomic_rmw xchg relaxed %p, %v align 8 : i64
  %x2 = atomic_rmw add acquire %p, %v align 8 : i64
  %x3 = atomic_rmw sub release %p, %v align 8 : i64
  %x4 = atomic_rmw nand acq_rel %p, %v align 8 : i64
  %x5 = atomic_rmw umin seq_cst %p, %v align 8 : i64
  %t = trunc %v : i8
  %x6 = atomic_rmw max seq_cst %p, %t align 1 : i8
  %x7 = cmpxchg seq_cst relaxed %p, %a, %v align 8 : i64
  %h = trunc %v : i16
  %x8 = cmpxchg acquire acquire %p, %h, %h align 2 : i16
  %v8 = load volatile %p align 1 : i8
  store volatile %v8, %p align 1 : i8
  %s1 = add %a, %b : i64
  %s2 = add %s1, %c : i64
  %s3 = add %s2, %x1 : i64
  %s4 = add %s3, %x7 : i64
  ret %s4
}
"#;
