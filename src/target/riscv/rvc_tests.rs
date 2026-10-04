//! The C extension: the compressor ([`super::encode::compress`]) against
//! `llvm-mc -mattr=+c` instruction by instruction (the same 16-bit form, or
//! none where `llvm-mc` keeps 32 bits), every compressed form executed
//! against its 32-bit original on the simulator (whose expander is written
//! independently), whole programs compiled compressed and run against the
//! reference, and the object flag.

use std::collections::HashMap;

use crate::codegen::CodegenOptions;

use super::diff_tests::{Harness, f64s, parse, samples};
use super::encode::*;
use super::fd_tests::llvm_mc_with;
use super::sim::{Cpu, EXIT, Image, Memory, expand_compressed};

const X: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
];
const F: [&str; 32] = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
    "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
];

/// 32-bit instructions with their assembly: every form the compressor
/// handles, at the edges of each 16-bit form's registers and immediates, and
/// some it must leave alone.
fn corpus() -> Vec<(u32, String)> {
    let mut c: Vec<(u32, String)> = Vec::new();
    let regs = [0u32, 1, 2, 5, 8, 9, 10, 15, 16, 31];
    let imms = [-2048i32, -513, -512, -33, -32, -16, -1, 0, 1, 4, 8, 16, 31, 32, 252, 496, 504, 1020, 2047];
    for &rd in &regs {
        for &rs in &regs {
            for &k in &imms {
                c.push((addi(rd, rs, k), format!("addi {}, {}, {k}", X[rd as usize], X[rs as usize])));
            }
            for (f, name) in [(add as fn(u32, u32, u32) -> u32, "add"), (sub, "sub"), (and, "and"), (or, "or"), (xor, "xor"), (mul, "mul")] {
                for &rt in &[0u32, 8, 10, 15, 16] {
                    c.push((f(rd, rs, rt), format!("{name} {}, {}, {}", X[rd as usize], X[rs as usize], X[rt as usize])));
                }
            }
        }
        for &k in &[-33i32, -32, 0, 31, 32] {
            c.push((addiw(rd, rd, k), format!("addiw {0}, {0}, {k}", X[rd as usize])));
            c.push((andi(rd, rd, k), format!("andi {0}, {0}, {k}", X[rd as usize])));
            c.push((addiw(rd, 10, k), format!("addiw {}, a0, {k}", X[rd as usize])));
        }
        for &sh in &[0u32, 1, 31, 32, 63] {
            c.push((slli(rd, rd, sh), format!("slli {0}, {0}, {sh}", X[rd as usize])));
            c.push((srli(rd, rd, sh), format!("srli {0}, {0}, {sh}", X[rd as usize])));
            c.push((srai(rd, rd, sh), format!("srai {0}, {0}, {sh}", X[rd as usize])));
            c.push((slli(rd, 10, sh), format!("slli {}, a0, {sh}", X[rd as usize])));
        }
        for &u in &[0u32, 1, 31, 32, 0xfffe0, 0xfffff] {
            c.push((lui(rd, u), format!("lui {}, {u}", X[rd as usize])));
        }
        c.push((r_type(0, 9, rd, 0, rd, 0x3B), format!("addw {0}, {0}, s1", X[rd as usize])));
        c.push((r_type(0, rd, 9, 0, rd, 0x3B), format!("addw {0}, s1, {0}", X[rd as usize])));
        c.push((r_type(0x20, 9, rd, 0, rd, 0x3B), format!("subw {0}, {0}, s1", X[rd as usize])));
        for &rs in &[0u32, 1, 10] {
            c.push((jalr(rd, rs, 0), format!("jalr {}, 0({})", X[rd as usize], X[rs as usize])));
        }
        c.push((jalr(0, rd, 8), format!("jalr zero, 8({})", X[rd as usize])));
    }
    // Loads and stores: sp-relative and compressed-register forms.
    let offs = [-8i32, 0, 4, 8, 12, 124, 128, 248, 252, 256, 504, 512];
    for &r in &[0u32, 1, 8, 10, 15, 16] {
        for &base in &[2u32, 8, 15, 16] {
            for &o in &offs {
                let (rn, bn) = (X[r as usize], X[base as usize]);
                c.push((load(8, r, base, o), format!("ld {rn}, {o}({bn})")));
                c.push((store(8, r, base, o), format!("sd {rn}, {o}({bn})")));
                c.push((i_type(o, base, 2, r, 0x03), format!("lw {rn}, {o}({bn})")));
                c.push((store(4, r, base, o), format!("sw {rn}, {o}({bn})")));
                c.push((load(4, r, base, o), format!("lwu {rn}, {o}({bn})")));
                c.push((load(1, r, base, o), format!("lbu {rn}, {o}({bn})")));
                let fname = F[r as usize];
                c.push((fload(8, r, base, o), format!("fld {fname}, {o}({bn})")));
                c.push((fstore(8, r, base, o), format!("fsd {fname}, {o}({bn})")));
                c.push((fload(4, r, base, o), format!("flw {fname}, {o}({bn})")));
            }
        }
    }
    c.push((ebreak(), "ebreak".into()));
    c.push((ecall(), "ecall".into()));
    c.push((beq(10, 11, 8), "beq a0, a1, 8".into()));
    c.push((fp_op(0, 64, RM_DYN, 10, 11, 12), "fadd.d fa0, fa1, fa2".into()));
    c
}

#[test]
fn compressed_forms_match_llvm_mc() {
    let corpus = corpus();
    // One `llvm-mc` run over the whole corpus (one instruction per line).
    let asm: String = corpus.iter().map(|(_, a)| format!("{a}\n")).collect();
    let Some(want) = llvm_mc_with(&asm, true) else {
        eprintln!("skipping compressed_forms_match_llvm_mc: no llvm-mc");
        return;
    };
    let mut at = 0usize;
    let mut compressed = 0;
    let mut bad: Vec<String> = Vec::new();
    for (w, a) in &corpus {
        let ours: Vec<u8> = match compress(*w) {
            Some(h) => {
                compressed += 1;
                h.to_le_bytes().to_vec()
            }
            None => w.to_le_bytes().to_vec(),
        };
        // llvm-mc's encoding of this line: 2 bytes when its low bits are not 11.
        let len = if want[at] & 3 == 3 { 4 } else { 2 };
        if ours != want[at..at + len] {
            bad.push(format!("`{a}`: ours {ours:02x?}, llvm-mc {:02x?}", &want[at..at + len]));
        }
        at += len;
    }
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
    assert_eq!(at, want.len());
    eprintln!("C extension: {} instructions matched llvm-mc, {compressed} compressed", corpus.len());
    assert!(compressed > 300);
    // Jumps and branches stay 32-bit (`c.j`/`c.beqz` would need relaxation:
    // their displacements are fixed up after layout).
    for w in [jal(0, 8), jal(1, 8), beq(8, 0, 8), bne(9, 0, -8)] {
        assert_eq!(compress(w), None);
    }
}

/// Run one instruction (32-bit `w`, or the halfword `h`) on the simulator
/// from a fixed pseudo-random state, returning the registers and the bytes
/// the access could touch.
fn exec_one(w: Option<u32>, h: Option<u16>, seed: u64) -> ([u64; 32], [u64; 32], Vec<u8>) {
    // The instruction ends exactly at the simulator's exit address, so the
    // run stops right after it whatever it writes.
    let mut mem = Memory::default();
    let code = if let Some(h) = h {
        mem.write(EXIT - 2, 2, u64::from(h));
        EXIT - 2
    } else {
        mem.write(EXIT - 4, 4, u64::from(w.unwrap()));
        EXIT - 4
    };
    let image = Image { mem, symbols: HashMap::new(), helpers: HashMap::new(), text: vec![(code, EXIT)] };
    let mut cpu = Cpu::new(&image);
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        s
    };
    for r in 1..32 {
        cpu.x[r] = 0x7000_0000 + (next() & 0xfff0);
    }
    for r in 0..32 {
        cpu.f[r] = next();
    }
    for k in 0..64u64 {
        cpu.mem.write(0x7000_0000 + 16 * k, 8, next());
    }
    let sp = 0x7000_8000;
    cpu.call(code, &[], &[], sp).unwrap_or_else(|e| panic!("{w:?} {h:?}: {e}"));
    let snapshot: Vec<u8> = (0..0x1_0000u64).step_by(8).flat_map(|o| cpu.mem.read(0x7000_0000 + o, 8).to_le_bytes()).collect();
    (cpu.x, cpu.f, snapshot)
}

/// Each compressed form executes exactly as its 32-bit original (the
/// simulator expands it with its own decoder): same registers, same memory.
#[test]
fn compressed_forms_execute_as_their_originals() {
    let mut n = 0;
    for (w, a) in corpus() {
        let Some(h) = compress(w) else { continue };
        let e = expand_compressed(h).unwrap_or_else(|| panic!("`{a}`: {h:#06x} does not expand"));
        if e == w {
            n += 1;
            continue;
        }
        // Jumps and ebreak expand canonically; the rest must agree in effect.
        assert!(w & 0x7f != 0x67 && w != ebreak(), "`{a}`: {h:#06x} expands to {e:#010x}");
        for seed in [1u64, 2, 3] {
            assert_eq!(exec_one(Some(w), None, seed), exec_one(None, Some(h), seed), "`{a}` vs {h:#06x}");
        }
        n += 1;
    }
    assert!(n > 300);
}

/// Whole programs compiled with the C extension run as before: integers at
/// several widths, floats, calls with stack arguments, `dyn_alloca` (whose
/// probe loop stays 32-bit), atomics (likewise), and the object says RVC.
#[test]
fn compressed_programs_match_the_reference() {
    let rvc = super::RiscvOptions::default().with_compressed(true);
    let mut n = 0;
    for bits in [8u32, 32, 64] {
        let t = format!("i{bits}");
        let mut src = String::from("module \"ops\"\n");
        for op in ["add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr", "udiv", "srem"] {
            src += &format!("func @{op}({t}, {t}) -> {t} {{\nentry ^0(%a: {t}, %b: {t}):\n  %r = {op} %a, %b : {t}\n  ret %r\n}}\n");
        }
        let h = Harness::with_riscv(&src, &CodegenOptions::default(), &rvc);
        let s = samples(bits);
        let pairs: Vec<Vec<u64>> = s.iter().flat_map(|&a| s.iter().map(move |&b| vec![a, b % u64::from(bits)])).collect();
        for op in ["add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr"] {
            n += h.check(op, &pairs);
        }
        let all: Vec<Vec<u64>> = s.iter().flat_map(|&a| s.iter().map(move |&b| vec![a, b])).collect();
        n += h.check("udiv", &all) + h.check("srem", &all);
    }
    let fsrc = "module \"f\"\nfunc @f(f64, f64) -> f64 {\nentry ^0(%a: f64, %b: f64):\n  %s = fadd %a, %b : f64\n  \
                %m = fmul %s, %a : f64\n  %d = fdiv %m, %b : f64\n  %r = frem %d, %a : f64\n  ret %r\n}\n";
    let h = Harness::with_riscv(fsrc, &CodegenOptions::default(), &rvc);
    let v = f64s();
    let fpairs: Vec<Vec<u64>> = v.iter().flat_map(|&a| v.iter().map(move |&b| vec![a, b])).collect();
    n += h.check("f", &fpairs);
    assert!(n > 2000, "{n}");

    // The struct, dyn_alloca and atomics programs of the other suites.
    let atomics = r#"
module "at"
func @rmw(i64, i64) -> i64 {
entry ^0(%x: i64, %y: i64):
  %p = alloca [2 x i64] : ptr
  store %x, %p align 8 : i64
  %q = ptr_add %p, i64 3 : ptr
  store i8 7, %q align 1 : i8
  %o = atomic_rmw nand seq_cst %p, %y align 8 : i64
  %b = atomic_rmw max acquire %q, i8 9 align 1 : i8
  %c = cmpxchg seq_cst seq_cst %p, %o, %x align 8 : i64
  %v = load %p align 8 : i64
  %bz = zext %b : i64
  %s = add %v, %bz : i64
  %s2 = add %s, %c : i64
  ret %s2
}
"#;
    let h = Harness::with_riscv(atomics, &CodegenOptions::default(), &rvc);
    assert_eq!(h.check("rmw", &[vec![5, 3], vec![u64::MAX, 0x1234]]), 2);

    let (m, syms) = parse(atomics);
    let obj = super::compile_module_riscv(&m, &syms, &CodegenOptions::default(), &rvc).object;
    assert!(super::uses_compressed(&obj));
    let elf = super::write_elf(&obj).unwrap();
    assert_eq!(u32::from_le_bytes(elf[48..52].try_into().unwrap()), 0x5, "RVC | FLOAT_ABI_DOUBLE");
    let plain = super::write_elf(&super::compile_module(&m, &syms)).unwrap();
    assert_eq!(u32::from_le_bytes(plain[48..52].try_into().unwrap()), 0x4);
    if let Some(dis) = super::fd_tests::objdump(&obj.sections()[0].bytes, true) {
        assert!(!dis.contains("unknown") && dis.contains("c."), "{dis}");
    }
}

/// `riscv64gc` (or any architecture string naming C) turns compression on.
#[test]
fn triples_select_the_c_extension() {
    for (t, c) in [("riscv64-linux", false), ("riscv64gc-unknown-linux-gnu", true), ("riscv64imac", true), ("riscv64g-linux", false)] {
        assert_eq!(super::RiscvOptions::for_triple(t).compressed, c, "{t}");
    }
}
