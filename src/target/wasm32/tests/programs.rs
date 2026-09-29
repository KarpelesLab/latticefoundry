//! The differential test programs: IR run under node and under the reference
//! interpreter, compared result by result (see [`super::differential`]).

use super::{differential, no_node};

/// A one-block function `@name(params) -> ret` with `body` (which names the
/// parameters `%a`, `%b`, `%c`, ...).
fn func(name: &str, params: &[&str], ret: &str, body: &str) -> String {
    let names = ["a", "b", "c", "d", "e", "f"];
    let sig: Vec<&str> = params.to_vec();
    let ps: Vec<String> = params.iter().zip(names).map(|(t, n)| format!("%{n}: {t}")).collect();
    format!("func @{name}({}) -> {ret} {{\nentry ^0({}):\n{body}\n}}\n", sig.join(", "), ps.join(", "))
}

/// Edge values of a `w`-bit integer, plus a few pseudo-random ones.
fn samples(w: u32) -> Vec<u128> {
    let mask = if w >= 128 { u128::MAX } else { (1u128 << w) - 1 };
    let top = 1u128 << (w - 1);
    let mut v = vec![0, 1, 2, 3, mask, mask - 1, top, top - 1, top + 1, 5, 0x5a];
    let mut x = 0x9e37_79b9_7f4a_7c15_2545_f491_4f6c_dd1du128;
    for _ in 0..5 {
        x = x.wrapping_mul(0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645).wrapping_add(0x5851_f42d_4c95_7f2d);
        v.push(x >> 17);
    }
    let mut out: Vec<u128> = v.into_iter().map(|x| x & mask).collect();
    out.sort_unstable();
    out.dedup();
    out
}

const WIDTHS: [u32; 12] = [1, 7, 8, 13, 16, 24, 31, 32, 33, 48, 63, 64];

/// Every integer binary operation and comparison at every width, including the
/// narrow and odd ones whose wrap-around the backend must mask.
#[test]
fn int_ops_all_widths() {
    let bins = ["add", "sub", "mul", "udiv", "sdiv", "urem", "srem", "and", "or", "xor", "shl", "lshr", "ashr"];
    let preds = ["eq", "ne", "ugt", "uge", "ult", "ule", "sgt", "sge", "slt", "sle"];
    let mut total = 0;
    for w in WIDTHS {
        let t = format!("i{w}");
        let mut src = String::from("module \"ints\"\n");
        let mut cases = Vec::new();
        let s = samples(w);
        for op in bins {
            let name = format!("{op}_{w}");
            src += &func(&name, &[&t, &t], &t, &format!("  %r = {op} %a, %b : {t}\n  ret %r"));
            for &a in &s {
                for &b in &s {
                    let b = if matches!(op, "shl" | "lshr" | "ashr") {
                        b % u128::from(w)
                    } else {
                        b
                    };
                    cases.push((name.clone(), vec![a, b]));
                }
            }
        }
        for p in preds {
            let name = format!("icmp_{p}_{w}");
            src += &func(&name, &[&t, &t], "i1", &format!("  %r = icmp {p} %a, %b : i1\n  ret %r"));
            for &a in &s {
                for &b in &s {
                    cases.push((name.clone(), vec![a, b]));
                }
            }
        }
        // select, freeze, and a use of the result inside the same block
        // (exercising the inlined stack expressions).
        let name = format!("mix_{w}");
        src += &func(
            &name,
            &[&t, &t],
            &t,
            &format!(
                "  %s = add %a, %b : {t}\n  %m = mul %s, %a : {t}\n  %c = icmp slt %m, %b : i1\n  %x = xor %m, %b : {t}\n  %f = freeze %x : {t}\n  %r = select %c, %f, %s : {t}\n  ret %r"
            ),
        );
        for &a in &s {
            for &b in &s {
                cases.push((name.clone(), vec![a, b]));
            }
        }
        let refs: Vec<(&str, Vec<u128>)> = cases.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
        let Some(t) = differential(&format!("ints{w}"), &src, &refs) else { return no_node("int_ops_all_widths") };
        total += t.compared;
    }
    eprintln!("int_ops_all_widths: {total} calls compared");
    assert!(total > 20_000, "{total}");
}

/// Truncation and extension between every pair of widths, and the narrow
/// arithmetic whose results must stay masked (the "narrow values" bug class).
#[test]
fn casts_between_widths() {
    let mut src = String::from("module \"casts\"\n");
    let mut cases = Vec::new();
    for from in WIDTHS {
        for to in WIDTHS {
            let (ft, tt) = (format!("i{from}"), format!("i{to}"));
            let ops: &[&str] = if to < from {
                &["trunc"]
            } else if to > from {
                &["zext", "sext"]
            } else {
                &[]
            };
            for op in ops {
                let name = format!("{op}_{from}_{to}");
                // Compute in the source width first so garbage bits would show.
                src += &func(&name, &[&ft], &tt, &format!("  %x = add %a, %a : {ft}\n  %r = {op} %x : {tt}\n  ret %r"));
                for a in samples(from) {
                    cases.push((name.clone(), vec![a]));
                }
            }
        }
    }
    let refs: Vec<(&str, Vec<u128>)> = cases.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    let Some(t) = differential("casts", &src, &refs) else { return no_node("casts_between_widths") };
    assert!(t.compared > 1500, "{t:?}");
}

const F32S: [f32; 16] =
    [0.0, -0.0, 1.0, -1.5, 2.5, 3.0, 7.0, 0.1, 1e10, -1e10, 300.7, -129.9, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-40];
const F64S: [f64; 16] = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    2.5,
    3.0,
    7.0,
    0.1,
    1e10,
    -1e10,
    70000.25,
    -40000.75,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    5e-324,
];

/// Float arithmetic, `frem` (through the `fmod` import), comparisons with
/// every predicate, and conversions.
#[test]
fn float_ops() {
    let mut src = String::from("module \"floats\"\n");
    let mut cases: Vec<(String, Vec<u128>)> = Vec::new();
    for (t, vals) in [
        ("f32", F32S.iter().map(|v| u128::from(v.to_bits())).collect::<Vec<_>>()),
        ("f64", F64S.iter().map(|v| u128::from(v.to_bits())).collect::<Vec<_>>()),
    ] {
        for op in ["fadd", "fsub", "fmul", "fdiv", "frem"] {
            let name = format!("{op}_{t}");
            src += &func(&name, &[t, t], t, &format!("  %r = {op} %a, %b : {t}\n  ret %r"));
            for &a in &vals {
                for &b in &vals {
                    cases.push((name.clone(), vec![a, b]));
                }
            }
        }
        let name = format!("fneg_{t}");
        src += &func(&name, &[t], t, &format!("  %r = fneg %a : {t}\n  ret %r"));
        for &a in &vals {
            cases.push((name.clone(), vec![a]));
        }
        for p in [
            "false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno", "true",
        ] {
            let name = format!("fcmp_{p}_{t}");
            src += &func(&name, &[t, t], "i1", &format!("  %r = fcmp {p} %a, %b : i1\n  ret %r"));
            for &a in &vals {
                for &b in &vals {
                    cases.push((name.clone(), vec![a, b]));
                }
            }
        }
        for w in [1u32, 8, 16, 24, 32, 48, 64] {
            let it = format!("i{w}");
            for op in ["fptosi", "fptoui"] {
                let name = format!("{op}_{t}_{w}");
                src += &func(&name, &[t], &it, &format!("  %r = {op} %a : {it}\n  ret %r"));
                for &a in &vals {
                    cases.push((name.clone(), vec![a]));
                }
            }
            for op in ["sitofp", "uitofp"] {
                let name = format!("{op}_{w}_{t}");
                src += &func(&name, &[&it], t, &format!("  %r = {op} %a : {t}\n  ret %r"));
                for a in samples(w) {
                    cases.push((name.clone(), vec![a]));
                }
            }
        }
        let (other, it) = if t == "f32" { ("f64", "i32") } else { ("f32", "i64") };
        let conv = if t == "f32" { "fpext" } else { "fptrunc" };
        let name = format!("{conv}_{t}");
        src += &func(&name, &[t], other, &format!("  %r = {conv} %a : {other}\n  ret %r"));
        let name2 = format!("bitcast_{t}");
        src += &func(&name2, &[t], it, &format!("  %r = bitcast %a : {it}\n  ret %r"));
        let name3 = format!("bitcast_back_{t}");
        src += &func(&name3, &[it], t, &format!("  %r = bitcast %a : {t}\n  ret %r"));
        for &a in &vals {
            cases.push((name.clone(), vec![a]));
            cases.push((name2.clone(), vec![a]));
            if !(t == "f32" && f32::from_bits(a as u32).is_nan()) && !(t == "f64" && f64::from_bits(a as u64).is_nan()) {
                cases.push((name3.clone(), vec![a]));
            }
        }
    }
    let refs: Vec<(&str, Vec<u128>)> = cases.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    let Some(t) = differential("floats", &src, &refs) else { return no_node("float_ops") };
    assert!(t.compared > 5000, "{t:?}");
}
