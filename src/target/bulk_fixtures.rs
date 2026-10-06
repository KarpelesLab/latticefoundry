//! Shared bulk-memory test fixtures (`docs/ir-design.md` §6k): self-contained
//! `@name(i64 n, i64 doff, i64 soff, i64 byte) -> i64` functions that fill a
//! 768-byte stack buffer with a pattern, run one `memcpy` / `memmove` /
//! `memset` on it, and return a checksum of the whole buffer. Each backend
//! runs them its own way (machine code on its emulator, its MIR interpreter)
//! against the reference executor ([`super::vector_fixtures::reference`]), so
//! the lowering of every length, alignment and overlap is checked against
//! the semantics. The functions take the same four `i64`s as the vector
//! fixtures, so the backends' existing harnesses run them unchanged.

use std::fmt::Write as _;

use super::vector_fixtures::Case;

/// Bytes in the buffer: a copy's source is its upper half.
const BUF: u64 = 768;

/// One test function. `len` is `None` for the variable length `%n`.
fn func(out: &mut String, name: &str, op: &str, align: u32, len: Option<u64>) {
    let _ = writeln!(out, "func @{name}(i64, i64, i64, i64) -> i64 {{");
    out.push_str("entry ^0(%n: i64, %doff: i64, %soff: i64, %byte: i64):\n");
    let _ = writeln!(out, "  %a = alloca [{} x i64] : ptr", BUF / 8);
    // The pattern: word i = 0x0123456789abcdef + i * 0x0f1e2d3c4b5a6978.
    out.push_str("  br ^1(i64 0, i64 81985529216486895)\n");
    out.push_str("^1(%i: i64, %w: i64):\n");
    let _ = writeln!(out, "  %more = icmp ult %i, i64 {BUF} : i1");
    out.push_str("  cond_br %more, ^2, ^3\n^2:\n  %p = ptr_add %a, %i : ptr\n  store %w, %p align 8 : i64\n");
    out.push_str("  %i2 = add %i, i64 8 : i64\n  %w2 = add %w, i64 1089641669211335032 : i64\n  br ^1(%i2, %w2)\n");
    out.push_str("^3:\n  %d = ptr_add %a, %doff : ptr\n");
    let n = len.map_or("%n".to_string(), |c| format!("i64 {c}"));
    match op {
        "memset" => {
            out.push_str("  %b = trunc %byte : i8\n");
            let _ = writeln!(out, "  memset %d, %b, {n} align {align}");
        }
        "memcpy" => {
            let _ = writeln!(out, "  %s0 = ptr_add %a, i64 {} : ptr", BUF / 2);
            out.push_str("  %s = ptr_add %s0, %soff : ptr\n");
            let _ = writeln!(out, "  memcpy %d, %s, {n} align {align}");
        }
        _ => {
            out.push_str("  %s = ptr_add %a, %soff : ptr\n");
            let _ = writeln!(out, "  memmove %d, %s, {n} align {align}");
        }
    }
    // Checksum: h = rotl(h ^ byte, 5) over every byte (no multiply, so the
    // 8-bit target needs no helper).
    out.push_str("  br ^4(i64 0, i64 7)\n^4(%j: i64, %h: i64):\n");
    let _ = writeln!(out, "  %go = icmp ult %j, i64 {BUF} : i1");
    out.push_str("  cond_br %go, ^5, ^6\n^5:\n  %q = ptr_add %a, %j : ptr\n  %v = load %q align 1 : i8\n");
    out.push_str("  %vw = zext %v : i64\n  %x = xor %h, %vw : i64\n  %hl = shl %x, i64 5 : i64\n");
    out.push_str("  %hr = lshr %x, i64 59 : i64\n  %h2 = or %hl, %hr : i64\n  %j2 = add %j, i64 1 : i64\n");
    out.push_str("  br ^4(%j2, %h2)\n^6:\n  ret %h\n}\n\n");
}

/// The fixture module and its cases: for each op, a variable-length
/// function at alignment 1 and 8, and a constant-length one per entry of
/// `consts`, each run at the lengths `ns` (the constant ones at their own)
/// and at unaligned and aligned offsets (overlapping both ways for
/// `memmove`).
pub(crate) fn bulk(ns: &[u64], consts: &[u64]) -> (String, Vec<Case>) {
    let mut src = String::from("module \"bulk\"\n\n");
    let mut cases: Vec<Case> = Vec::new();
    for op in ["memcpy", "memmove", "memset"] {
        let offs: &[(i64, i64)] =
            if op == "memmove" { &[(0, 9), (9, 0), (3, 3), (40, 1)] } else { &[(0, 0), (3, 13), (13, 1)] };
        let aligned: &[(i64, i64)] = if op == "memmove" { &[(0, 16), (16, 8)] } else { &[(8, 16), (0, 32)] };
        for (align, offs) in [(1u32, offs), (8, aligned)] {
            let name = format!("{}_v{align}", &op[3..]);
            func(&mut src, &name, op, align, None);
            for &n in ns {
                for &(d, s) in offs {
                    cases.push((name.clone(), vec![n as i64, d, s, 0x15a]));
                }
            }
            for &c in consts {
                let name = format!("{}_c{c}_a{align}", &op[3..]);
                func(&mut src, &name, op, align, Some(c));
                for &(d, s) in offs {
                    cases.push((name.clone(), vec![0, d, s, 0x3c]));
                }
            }
        }
    }
    (src, cases)
}

/// A sample of lengths: the small ones, the word and vector boundaries, and
/// up to 300 (the x86-64 tests run every one in-process).
pub(crate) fn some_lengths() -> Vec<u64> {
    (0..=20).chain([23, 31, 32, 33, 47, 63, 64, 65, 100, 127, 255, 256, 257, 300]).collect()
}

/// The constant lengths: inline expansions of every shape, and loops.
pub(crate) const CONSTS: [u64; 10] = [1, 3, 7, 8, 15, 16, 24, 40, 64, 300];
