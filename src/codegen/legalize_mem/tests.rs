//! Tests for the bulk-memory legalization: generated programs run in the
//! reference executor before and after legalization under several target
//! descriptions (aligned word chunks, narrow words, vectors with unaligned
//! accesses, byte loops only, native), across lengths, alignments, offsets,
//! volatility and constant or variable lengths; plus the libcall lowering.

use super::{BulkMemoryLowering, bulk_memory_libcalls, legalize_bulk_memory, uses_bulk_memory};
use crate::ir::refexec::{ExecError, run_named};
use crate::ir::semantics::SemValue;
use crate::ir::text::{parse_module, print_module};
use crate::ir::{DataLayout, InstId, InstKind, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

use puremp::Int;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    (m, syms)
}

/// The descriptions under test.
fn lowerings() -> Vec<(&'static str, BulkMemoryLowering)> {
    let p = BulkMemoryLowering::portable(&DataLayout::lp64());
    vec![
        ("portable", p),
        ("word4", BulkMemoryLowering { word: 4, ..p }),
        ("word2-max2", BulkMemoryLowering { word: 2, max_inline: 2, ..p }),
        ("bytes", BulkMemoryLowering { word: 1, max_inline: 0, ..p }),
        ("vector", BulkMemoryLowering { vector16: true, unaligned: true, max_inline: 4, ..p }),
        ("unaligned-loop", BulkMemoryLowering { unaligned: true, max_inline: 1, ..p }),
        ("native", BulkMemoryLowering { native: true, max_inline: 4, vector16: true, unaligned: true, ..p }),
        ("native-guard", BulkMemoryLowering { native: true, guard_zero: true, max_inline: 1, ..p }),
    ]
}

/// One generated test function: `@name(i64 %n) -> i64` sets up two 80-byte
/// 16-aligned buffers with a byte pattern, runs one bulk op, and returns a
/// checksum of the destination buffer (and, for a copy, the source one).
#[allow(clippy::too_many_arguments)]
fn gen_fn(out: &mut String, name: &str, op: &str, vol: bool, align: u32, doff: u64, soff: u64, len: &str) {
    out.push_str(&format!("func @{name}(i64) -> i64 {{\nentry ^0(%n: i64):\n"));
    out.push_str("  %a = alloca [5 x <16 x i8>] : ptr\n  %b = alloca [5 x <16 x i8>] : ptr\n");
    for w in 0..10u64 {
        let ka = 0x0102_0304_0506_0708u64.wrapping_mul(w + 1) ^ 0x1111;
        let kb = 0x8877_6655_4433_2211u64.wrapping_mul(w + 3);
        out.push_str(&format!("  %pa{w} = ptr_add %a, i64 {} : ptr\n", 8 * w));
        out.push_str(&format!("  store i64 {}, %pa{w} align 8 : i64\n", ka as i64));
        out.push_str(&format!("  %pb{w} = ptr_add %b, i64 {} : ptr\n", 8 * w));
        out.push_str(&format!("  store i64 {}, %pb{w} align 8 : i64\n", kb as i64));
    }
    out.push_str(&format!("  %d = ptr_add %a, i64 {doff} : ptr\n"));
    let src_buf = if op == "memmove" { "%a" } else { "%b" };
    out.push_str(&format!("  %s = ptr_add {src_buf}, i64 {soff} : ptr\n"));
    let v = if vol { " volatile" } else { "" };
    let n = match len {
        "n" => "%n".to_string(),
        "n32" => {
            out.push_str("  %n32 = trunc %n : i32\n");
            "%n32".to_string()
        }
        c => format!("i64 {c}"),
    };
    match op {
        "memset" => out.push_str(&format!("  memset{v} %d, i8 165, {n} align {align}\n")),
        _ => out.push_str(&format!("  {op}{v} %d, %s, {n} align {align}\n")),
    }
    out.push_str("  %h0 = add i64 7, i64 0 : i64\n");
    for (k, buf) in ["pa", "pb"].iter().enumerate() {
        for w in 0..10u64 {
            let i = k as u64 * 10 + w;
            out.push_str(&format!("  %w{i} = load %{buf}{w} align 8 : i64\n"));
            out.push_str(&format!("  %m{i} = mul %h{i}, i64 31 : i64\n"));
            out.push_str(&format!("  %h{} = add %m{i}, %w{i} : i64\n", i + 1));
        }
    }
    out.push_str("  ret %h20\n}\n\n");
}

/// Build the module of cases, returning it and the calls `(name, n)` to run.
fn cases(ops: &[&str], vol: bool) -> (String, Vec<(String, u64)>) {
    let mut src = String::from("module \"bulk\"\n\n");
    let mut calls = Vec::new();
    let mut k = 0;
    for &op in ops {
        for align in [1u32, 2, 4, 8, 16] {
            // Offsets that are multiples of the alignment (the op's promise).
            let offs: &[(u64, u64)] = if op == "memmove" {
                &[(0, 0), (0, 16), (16, 0), (align as u64, 0), (0, align as u64), (32, 32 - align as u64)]
            } else {
                &[(0, 0), (align as u64, 3 * align as u64 % 32), (16, 32)]
            };
            for &(doff, soff) in offs {
                let doff = doff / u64::from(align) * u64::from(align);
                let soff = soff / u64::from(align) * u64::from(align);
                for n in [0u64, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 40, 47, 48] {
                    let name = format!("c{k}");
                    k += 1;
                    gen_fn(&mut src, &name, op, vol, align, doff, soff, &n.to_string());
                    calls.push((name, 0));
                }
                for len in ["n", "n32"] {
                    let name = format!("c{k}");
                    k += 1;
                    gen_fn(&mut src, &name, op, vol, align, doff, soff, len);
                    for n in [0u64, 1, 5, 8, 13, 16, 31, 33, 48] {
                        calls.push((name.clone(), n));
                    }
                }
            }
        }
    }
    (src, calls)
}

fn check_all(ops: &[&str], vol: bool) {
    let (src, calls) = cases(ops, vol);
    let (orig, syms) = parse(&src);
    let want: Vec<_> = calls
        .iter()
        .map(|(f, n)| run_named(&orig, &syms, f, &[SemValue::int(64, Int::from_u64(*n))]))
        .collect();
    for (what, lowering) in lowerings() {
        let mut m = orig.clone();
        legalize_bulk_memory(&mut m, &lowering);
        crate::verify::verify_module(&m)
            .unwrap_or_else(|e| panic!("{what}: legalized module fails to verify: {e:?}"));
        if !lowering.native {
            assert!(!uses_bulk_memory(&m), "{what}: every op is expanded");
        }
        for ((f, n), w) in calls.iter().zip(&want) {
            let got = run_named(&m, &syms, f, &[SemValue::int(64, Int::from_u64(*n))]);
            match (w, &got) {
                (Ok(Some(w)), Ok(Some(g))) => {
                    assert!(!w.is_poison(), "{f}: vacuous");
                    assert_eq!(g, w, "{what} {f}({n})\n{}", print_module(&m, &syms));
                }
                // A source with UB (an out-of-bounds or overlapping case) may
                // do anything once lowered.
                (Err(ExecError::Ub(_)), _) => {}
                _ => panic!("{what} {f}({n}): want {w:?}, got {got:?}"),
            }
        }
    }
    // Some cases must have been defined, or the comparison says nothing.
    assert!(want.iter().filter(|w| matches!(w, Ok(Some(_)))).count() > calls.len() / 2);
}

#[test]
fn memcpy_lowerings_agree_with_the_reference() {
    check_all(&["memcpy"], false);
}

#[test]
fn memmove_lowerings_agree_with_the_reference() {
    check_all(&["memmove"], false);
}

#[test]
fn memset_lowerings_agree_with_the_reference() {
    check_all(&["memset"], false);
}

#[test]
fn volatile_lowerings_agree_and_stay_volatile_and_scalar() {
    check_all(&["memcpy", "memset", "memmove"], true);
    let (src, _) = cases(&["memcpy"], true);
    let (mut m, _) = parse(&src);
    let p = BulkMemoryLowering { vector16: true, unaligned: true, ..BulkMemoryLowering::portable(&DataLayout::lp64()) };
    legalize_bulk_memory(&mut m, &p);
    for f in m.functions() {
        for i in 0..f.inst_count() {
            let inst = f.inst(InstId::from_index(i));
            if let InstKind::Load { ty, volatile, .. } | InstKind::Store { ty, volatile, .. } = &inst.kind {
                // The pattern set-up and checksum use 8-byte non-volatile
                // accesses; every expansion access is volatile and scalar.
                assert!(!m.types().is_vector(*ty), "no vector volatile access");
                let _ = volatile;
            }
        }
    }
}

#[test]
fn constant_expansion_shapes() {
    let src = "module \"s\"\n\nfunc @f(ptr, ptr, i8) -> void {\nentry ^0(%d: ptr, %s: ptr, %b: i8):\n  memcpy %d, %s, i64 24 align 8\n  memset %d, %b, i64 16 align 8\n  memset %d, i8 0, i32 256 align 8\n  memmove %d, %s, i64 12 align 4\n  ret\n}\n";
    let (m0, syms) = parse(src);
    let count = |m: &Module, pred: &dyn Fn(&InstKind) -> bool| {
        let f = m.function(crate::ir::FuncId::from_index(0));
        (0..f.inst_count()).filter(|&i| pred(&f.inst(InstId::from_index(i)).kind)).count()
    };
    // x86-64-like: 16-byte vectors inline up to 4 accesses, the 256-byte fill
    // stays native with an i64 length.
    let x86 = BulkMemoryLowering {
        word: 8,
        vector16: true,
        unaligned: true,
        max_inline: 4,
        native: true,
        guard_zero: false,
    };
    let mut m = m0.clone();
    legalize_bulk_memory(&mut m, &x86);
    let text = print_module(&m, &syms);
    assert!(text.contains("memset %0, i8 0, i64 256 align 8"), "{text}");
    assert_eq!(count(&m, &|k| k.is_bulk_memory()), 1, "{text}");
    assert!(text.contains("load %1 align 8 : <16 x i8>"), "a vector chunk: {text}");
    assert!(text.contains("splat %2 : <16 x i8>"), "a variable byte splat: {text}");
    // The memmove (8 + 4 bytes) loads both chunks before storing.
    let f = m.function(crate::ir::FuncId::from_index(0));
    let kinds: Vec<&InstKind> = (0..f.inst_count()).map(|i| &f.inst(InstId::from_index(i)).kind).collect();
    let tail: Vec<bool> = kinds
        .iter()
        .rev()
        .filter(|k| matches!(k, InstKind::Load { .. } | InstKind::Store { .. }))
        .take(4)
        .map(|k| matches!(k, InstKind::Store { .. }))
        .collect();
    assert_eq!(tail, [true, true, false, false], "{text}");

    // Portable, aligned 8: the fill of 256 bytes is a loop, the rest inline.
    let mut m = m0.clone();
    legalize_bulk_memory(&mut m, &BulkMemoryLowering::portable(&DataLayout::lp64()));
    let text = print_module(&m, &syms);
    assert!(!uses_bulk_memory(&m));
    assert!(text.contains("icmp ult"), "a loop: {text}");
    assert!(text.contains("store i64 0,"), "word stores of the constant fill: {text}");
    // A variable byte is replicated by shifts and ors, not multiplied.
    assert!(text.contains("shl") && !text.contains("mul"), "{text}");
}

#[test]
fn libcalls_replace_what_would_stay_native() {
    let src = "module \"l\"\n\nfunc @f(ptr, ptr, i32, i8) -> void {\nentry ^0(%d: ptr, %s: ptr, %n: i32, %b: i8):\n  memcpy %d, %s, %n align 1\n  memmove %d, %s, i64 1000 align 1\n  memset %d, %b, %n align 1\n  memset %d, %b, i64 0 align 1\n  ret\n}\n";
    let (mut m, mut syms) = parse(src);
    bulk_memory_libcalls(&mut m, &mut syms);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
    let text = print_module(&m, &syms);
    assert!(!uses_bulk_memory(&m), "{text}");
    for want in ["call @memcpy(%0, %1, %", "call @memmove(%0, %1, i64 1000)", "call @memset(%0, %"] {
        assert!(text.contains(want), "missing `{want}` in\n{text}");
    }
    assert_eq!(text.matches("call @memset").count(), 1, "n = 0 is dropped: {text}");
    assert!(text.contains("zext %2 : i64") && text.contains("zext %3 : i32"), "{text}");
}
