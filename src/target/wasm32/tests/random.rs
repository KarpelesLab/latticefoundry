//! Random programs: arbitrary control flow (loops, irreducible cycles,
//! switches, edges passing permuted block arguments) over random arithmetic
//! on `i8`/`i32`/`i64`, each run under node and the reference interpreter.
//!
//! Every program terminates: a *fuel* counter travels along every edge and a
//! guard block before each body exits once it reaches zero. Division is only
//! by a positive odd divisor and shifts are masked, so no input is undefined
//! behavior or poison.

use super::{differential, differential_optimized, no_node};
use crate::target::wasm32::structure::{CfgNode, Shape, is_reducible};

/// A deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
        &v[self.below(v.len())]
    }
}

const TYPES: [&str; 3] = ["i32", "i64", "i8"];

/// One function body under construction: the text, and the values of each
/// type available at the current point.
struct Gen {
    out: String,
    vals: [Vec<String>; 3],
    next: usize,
}

impl Gen {
    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("%v{}", self.next)
    }

    fn operand(&mut self, rng: &mut Rng, t: usize) -> String {
        if rng.below(6) == 0 {
            let c: i64 = [0, 1, -1, 7, 100, -128, 255][rng.below(7)];
            format!("{} {c}", TYPES[t])
        } else {
            rng.pick(&self.vals[t]).clone()
        }
    }

    fn emit(&mut self, t: usize, line: String) -> String {
        let v = self.fresh();
        self.out += &format!("  {v} = {line}\n");
        self.vals[t].push(v.clone());
        v
    }

    /// One random computation, adding its result to the pool.
    fn op(&mut self, rng: &mut Rng) {
        let t = rng.below(3);
        let ty = TYPES[t];
        let bits = [32, 64, 8][t];
        match rng.below(9) {
            7 | 8 => {
                // Memory: a slot of the global `@mem` or of the stack frame
                // `%slots`, picked by a masked index; a store, or a load of `t`.
                let base = *rng.pick(&["@mem", "%slots"]);
                let i = self.operand(rng, 0);
                let m = self.emit(0, format!("and {i}, i32 15 : i32"));
                let o = self.emit(0, format!("mul {m}, i32 8 : i32"));
                let p = self.fresh();
                self.out += &format!("  {p} = ptr_add {base}, {o} : ptr\n");
                let align = [4, 8, 1][t];
                if rng.below(2) == 0 {
                    let v = self.operand(rng, t);
                    self.out += &format!("  store {v}, {p} align {align} : {ty}\n");
                } else {
                    self.emit(t, format!("load {p} align {align} : {ty}"));
                }
            }
            0 | 1 => {
                let op = *rng.pick(&["add", "sub", "mul", "and", "or", "xor"]);
                let (a, b) = (self.operand(rng, t), self.operand(rng, t));
                self.emit(t, format!("{op} {a}, {b} : {ty}"));
            }
            2 => {
                let op = *rng.pick(&["shl", "lshr", "ashr"]);
                let (a, b) = (self.operand(rng, t), self.operand(rng, t));
                let m = self.emit(t, format!("and {b}, {ty} {} : {ty}", bits - 1));
                self.emit(t, format!("{op} {a}, {m} : {ty}"));
            }
            3 => {
                let op = *rng.pick(&["udiv", "urem", "sdiv", "srem"]);
                let (a, b) = (self.operand(rng, t), self.operand(rng, t));
                let m = self.emit(t, format!("and {b}, {ty} 63 : {ty}"));
                let d = self.emit(t, format!("or {m}, {ty} 1 : {ty}"));
                self.emit(t, format!("{op} {a}, {d} : {ty}"));
            }
            4 => {
                // A cast into `t` from another type.
                let s = (t + 1 + rng.below(2)) % 3;
                let from_bits = [32, 64, 8][s];
                let a = self.operand(rng, s);
                let op = if from_bits > bits { "trunc" } else if rng.below(2) == 0 { "zext" } else { "sext" };
                self.emit(t, format!("{op} {a} : {ty}"));
            }
            _ => {
                let pred = *rng.pick(&["eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"]);
                let s = rng.below(3);
                let (x, y) = (self.operand(rng, s), self.operand(rng, s));
                let c = self.fresh();
                self.out += &format!("  {c} = icmp {pred} {x}, {y} : i1\n");
                let (a, b) = (self.operand(rng, t), self.operand(rng, t));
                self.emit(t, format!("select {c}, {a}, {b} : {ty}"));
            }
        }
    }

    /// Arguments for a guard block: fuel, then one value of each type.
    fn args(&mut self, rng: &mut Rng, fuel: &str) -> String {
        let (a, b, c) = (self.operand(rng, 0), self.operand(rng, 1), self.operand(rng, 2));
        format!("({fuel}, {a}, {b}, {c})")
    }
}

/// A random function `@f(i32, i64, i8) -> i64` with `n` nodes.
fn program(seed: u64, n: usize) -> String {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let exit = 2 * n + 1;
    let mut g = Gen { out: String::new(), vals: Default::default(), next: 0 };
    g.out += "func @f(i32, i64, i8) -> i64 {\nentry ^0(%a: i32, %b: i64, %c: i8):\n";
    // A zeroed stack frame the bodies may use (uninitialized, it would read
    // as poison).
    g.out += "  %slots = alloca [16 x i64] : ptr\n";
    for i in 0..16 {
        g.out += &format!("  %z{i}p = ptr_add %slots, i32 {} : ptr\n  store i64 0, %z{i}p align 8 : i64\n", i * 8);
    }
    g.out += "  br ^1(i32 40, %a, %b, %c)\n";
    for k in 0..n {
        let (guard, body) = (2 * k + 1, 2 * k + 2);
        // The guard: exit with a digest of the values once the fuel is out.
        g.out += &format!("^{guard}(%f{k}: i32, %x{k}: i32, %y{k}: i64, %z{k}: i8):\n");
        g.out += &format!("  %d{k} = icmp eq %f{k}, i32 0 : i1\n  %g{k} = sub %f{k}, i32 1 : i32\n");
        g.out += &format!("  %xe{k} = zext %x{k} : i64\n  %ze{k} = sext %z{k} : i64\n");
        g.out += &format!("  %s{k} = add %xe{k}, %ze{k} : i64\n  %r{k} = xor %s{k}, %y{k} : i64\n");
        g.out += &format!("  cond_br %d{k}, ^{exit}(%r{k}), ^{body}\n");
        // The body: random computations, then a random terminator.
        g.out += &format!("^{body}:\n");
        g.vals = [vec![format!("%x{k}")], vec![format!("%y{k}")], vec![format!("%z{k}")]];
        for _ in 0..2 + rng.below(6) {
            g.op(&mut rng);
        }
        let fuel = format!("%g{k}");
        let target = |rng: &mut Rng| 2 * rng.below(n) + 1;
        match rng.below(4) {
            0 => {
                let (t, a) = (target(&mut rng), g.args(&mut rng, &fuel));
                g.out += &format!("  br ^{t}{a}\n");
            }
            1 | 2 => {
                let s = rng.below(3);
                let (x, y) = (g.operand(&mut rng, s), g.operand(&mut rng, s));
                let c = g.fresh();
                let pred = *rng.pick(&["eq", "ne", "ult", "slt", "sge"]);
                g.out += &format!("  {c} = icmp {pred} {x}, {y} : i1\n");
                let (t1, a1) = (target(&mut rng), g.args(&mut rng, &fuel));
                let (t2, a2) = (target(&mut rng), g.args(&mut rng, &fuel));
                g.out += &format!("  cond_br {c}, ^{t1}{a1}, ^{t2}{a2}\n");
            }
            _ => {
                let s = if rng.below(2) == 0 { 0 } else { 2 };
                let v = g.operand(&mut rng, s);
                let m = g.emit(s, format!("and {v}, {} 7 : {}", TYPES[s], TYPES[s]));
                let (t, a) = (target(&mut rng), g.args(&mut rng, &fuel));
                let mut cases = Vec::new();
                for v in 0..1 + rng.below(6) {
                    if rng.below(3) != 0 {
                        let (t, a) = (target(&mut rng), g.args(&mut rng, &fuel));
                        cases.push(format!("{v}: ^{t}{a}"));
                    }
                }
                g.out += &format!("  switch {m}, ^{t}{a} [{}]\n", cases.join(", "));
            }
        }
    }
    g.out += &format!("^{exit}(%res: i64):\n  ret %res\n}}\n");
    let zeros = vec!["i64 0"; 16].join(", ");
    format!("module \"random{seed}\"\nglobal @mem : [16 x i64] = [16 x i64] ({zeros})\n{}", g.out)
}

/// Many random programs, each on a handful of inputs, at `-O0` and `-O2`.
#[test]
fn random_programs() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut total = 0;
    let mut irreducible = 0;
    for seed in 0..60u64 {
        let n = 1 + (seed as usize % 7);
        let src = program(seed, n);
        // Count the programs whose CFG needs the dispatch loop.
        let (m, _) = super::parse(&src);
        let f = m.functions().next().expect("one function");
        let cfg = crate::analysis::cfg::ControlFlowGraph::new(f);
        let graph: Vec<CfgNode> = (0..f.block_count())
            .map(|b| {
                let arms = cfg.successors(b).to_vec();
                let shape = if arms.len() > 1 { Shape::Table } else { Shape::Direct };
                CfgNode { arms, shape }
            })
            .collect();
        if !is_reducible(&graph) {
            irreducible += 1;
        }
        let cases: Vec<(&str, Vec<u128>)> = (0..6)
            .map(|_| ("f", vec![u128::from(rng.next() as u32), u128::from(rng.next()), u128::from(rng.next() as u8)]))
            .collect();
        let Some(t) = differential(&format!("random{seed}"), &src, &cases) else { return no_node("random_programs") };
        assert_eq!(t.skipped, 0, "{src}");
        total += t.compared;
        let t = differential_optimized(&format!("random{seed}"), &src, &cases).expect("node");
        assert_eq!(t.skipped, 0, "{src}");
    }
    assert_eq!(total, 360);
    assert!(irreducible >= 5, "only {irreducible} irreducible programs");
    eprintln!("random_programs: {irreducible} of 60 irreducible");
}
