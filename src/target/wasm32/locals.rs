//! Local assignment: which wasm local holds each SSA value.
//!
//! Every value that lives in a local (a block parameter, or a result that is
//! used and not recomputed in place) needs one local per wasm value of its
//! type, and two values may share a local when their live ranges do not
//! overlap. In SSA form two values interfere exactly when one is live at the
//! other's definition, and every value live at a definition is defined earlier
//! in dominance order, so a greedy pass in dominator-tree preorder that gives
//! each definition the lowest local free among the values live at that point
//! assigns every value without conflict (the classic result that SSA
//! interference graphs are chordal).
//!
//! The liveness it relies on follows the emitted code, not just the IR:
//!
//! - a value recomputed in place (inlined) is read where its *root* user is
//!   emitted, so its leaves are uses at that point;
//! - a block's parameters are written on the incoming edges, after the edge's
//!   arguments are all read, and are defined at the block's start: they
//!   interfere with everything live into the block (and with each other), even
//!   when unused;
//! - a terminator reads all its edges' arguments at the end of its block
//!   (a conservative choice: each arm reads only its own).
//!
//! The entry block's parameters are the function's wasm parameters, fixed
//! locals outside the shared pools.

use super::binary::ValType;
use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::{BlockId, Function, ValueDef, ValueId};

/// A dense set of values.
#[derive(Clone, PartialEq, Eq)]
struct Set(Vec<u64>);

impl Set {
    fn new(n: usize) -> Set {
        Set(vec![0; n.div_ceil(64)])
    }

    fn contains(&self, v: usize) -> bool {
        self.0[v / 64] >> (v % 64) & 1 == 1
    }

    fn insert(&mut self, v: usize) {
        self.0[v / 64] |= 1 << (v % 64);
    }

    fn remove(&mut self, v: usize) {
        self.0[v / 64] &= !(1 << (v % 64));
    }

    /// Add every member of `other`; whether anything was new.
    fn union_with(&mut self, other: &Set) -> bool {
        let mut changed = false;
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            let n = *a | *b;
            changed |= n != *a;
            *a = n;
        }
        changed
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(w, &bits)| {
            (0..64).filter(move |b| bits >> b & 1 == 1).map(move |b| w * 64 + b)
        })
    }
}

/// The result: each value's locals, and the types of the pooled locals, which
/// follow the parameters in index order.
pub(crate) struct Assignment {
    /// Per value: its local(s) (empty when it has none).
    pub(crate) loc: Vec<Vec<u32>>,
    /// The type of each pooled local (local `nparams + i`).
    pub(crate) locals: Vec<ValType>,
}

const TYPES: [ValType; 4] = [ValType::I32, ValType::I64, ValType::F32, ValType::F64];

fn type_index(t: ValType) -> usize {
    TYPES.iter().position(|&x| x == t).expect("a number type")
}

/// The locals of the values an instruction reads where it is emitted.
struct Reads<'a> {
    f: &'a Function,
    inline: &'a [bool],
    held: Vec<bool>,
}

impl Reads<'_> {
    fn of(&self, operands: &[ValueId], out: &mut Vec<usize>) {
        for &o in operands {
            if self.inline[o.index()] {
                let ValueDef::Inst(i) = self.f.value(o).def else { unreachable!("only results are inlined") };
                self.of(self.f.inst(i).operands(), out);
            } else if self.held[o.index()] {
                out.push(o.index());
            }
        }
    }
}

/// Assign locals to `f`'s values. `fixed[v]` are the parameter locals of the
/// entry block's parameters; `wants[v]` the wasm types of every other value
/// that needs locals; `inline[v]` marks values recomputed at their use.
pub(crate) fn assign(
    f: &Function,
    nparams: u32,
    fixed: &[Vec<u32>],
    wants: &[Vec<ValType>],
    inline: &[bool],
) -> Assignment {
    let nv = f.value_count();
    let nb = f.block_count();
    let held: Vec<bool> = (0..nv).map(|v| !wants[v].is_empty() || !fixed[v].is_empty()).collect();
    let reads = Reads { f, inline, held };

    // Per block: what each instruction reads (non-inlined ones only) and the
    // terminator's reads.
    let block_reads = |b: BlockId| -> (Vec<Vec<usize>>, Vec<usize>) {
        let block = f.block(b);
        let insts = block
            .insts()
            .iter()
            .map(|&i| {
                let inst = f.inst(i);
                let mut out = Vec::new();
                if !inst.result().is_some_and(|r| inline[r.index()]) {
                    reads.of(inst.operands(), &mut out);
                }
                out
            })
            .collect();
        let mut term = Vec::new();
        if let Some(t) = block.terminator() {
            reads.of(f.inst(t).operands(), &mut term);
        }
        (insts, term)
    };
    let all_reads: Vec<(Vec<Vec<usize>>, Vec<usize>)> = (0..nb).map(|b| block_reads(BlockId::from_index(b))).collect();
    let result_of = |b: usize, k: usize| -> Option<usize> {
        let i = f.block(BlockId::from_index(b)).insts()[k];
        f.inst(i).result().map(|r| r.index()).filter(|&r| reads.held[r])
    };

    // Liveness to a fixpoint: live_in(b) = reads before defs, from live_out(b)
    // = ∪ live_in(s) \ params(s).
    let cfg = ControlFlowGraph::new(f);
    let mut live_in = vec![Set::new(nv); nb];
    let mut live_out = vec![Set::new(nv); nb];
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..nb).rev() {
            let mut out = Set::new(nv);
            for &s in cfg.successors(b) {
                let mut li = live_in[s].clone();
                for &p in f.block(BlockId::from_index(s)).params() {
                    li.remove(p.index());
                }
                out.union_with(&li);
            }
            let mut live = out.clone();
            live_out[b] = out;
            let (insts, term) = &all_reads[b];
            for &r in term {
                live.insert(r);
            }
            for k in (0..insts.len()).rev() {
                if let Some(r) = result_of(b, k) {
                    live.remove(r);
                }
                for &r in &insts[k] {
                    live.insert(r);
                }
            }
            if live != live_in[b] {
                live_in[b] = live;
                changed = true;
            }
        }
    }

    // Greedy coloring in dominator-tree preorder.
    let doms = Dominators::new(f, &cfg);
    let mut children = vec![Vec::new(); nb];
    for b in 0..nb {
        if let Some(d) = doms.idom(b)
            && d != b
        {
            children[d].push(b);
        }
    }
    let mut color: Vec<Vec<u32>> = vec![Vec::new(); nv];
    let mut pool = [0u32; 4];
    let entry = f.entry().expect("defined").index();
    let mut stack = vec![entry];
    while let Some(b) = stack.pop() {
        stack.extend(children[b].iter().rev());
        let block = f.block(BlockId::from_index(b));
        // Colors taken by what is live into the block (the parameters aside).
        let mut taken: [Vec<bool>; 4] = Default::default();
        let take = |taken: &mut [Vec<bool>; 4], t: usize, c: u32| {
            let v = &mut taken[t];
            if v.len() <= c as usize {
                v.resize(c as usize + 1, false);
            }
            v[c as usize] = true;
        };
        let params: Vec<usize> = block.params().iter().map(|p| p.index()).collect();
        for v in live_in[b].iter() {
            if params.contains(&v) || !fixed[v].is_empty() {
                continue;
            }
            for (&c, &t) in color[v].iter().zip(&wants[v]) {
                take(&mut taken, type_index(t), c);
            }
        }
        let alloc = |taken: &mut [Vec<bool>; 4], pool: &mut [u32; 4], v: usize, color: &mut Vec<Vec<u32>>| {
            let cs: Vec<u32> = wants[v]
                .iter()
                .map(|&t| {
                    let ti = type_index(t);
                    let c = taken[ti].iter().position(|&x| !x).unwrap_or(taken[ti].len()) as u32;
                    take(taken, ti, c);
                    pool[ti] = pool[ti].max(c + 1);
                    c
                })
                .collect();
            color[v] = cs;
        };
        let free = |taken: &mut [Vec<bool>; 4], v: usize, color: &Vec<Vec<u32>>| {
            for (&c, &t) in color[v].iter().zip(&wants[v]) {
                taken[type_index(t)][c as usize] = false;
            }
        };
        // Parameters: defined together at the start; one never read dies at once.
        if b != entry {
            for &p in &params {
                alloc(&mut taken, &mut pool, p, &mut color);
            }
            for &p in &params {
                if !live_in[b].contains(p) {
                    free(&mut taken, p, &color);
                }
            }
        }
        // Which values die at each instruction, from a backward walk.
        let (insts, term) = &all_reads[b];
        let mut live = live_out[b].clone();
        for &r in term {
            live.insert(r);
        }
        let mut dies: Vec<Vec<usize>> = vec![Vec::new(); insts.len()];
        let mut dead_def = vec![false; insts.len()];
        for k in (0..insts.len()).rev() {
            if let Some(r) = result_of(b, k) {
                dead_def[k] = !live.contains(r);
                live.remove(r);
            }
            for &r in &insts[k] {
                if !live.contains(r) {
                    live.insert(r);
                    dies[k].push(r);
                }
            }
        }
        for k in 0..insts.len() {
            // Operands are all read before the result is written, so a dying
            // operand's local may hold the result.
            for &r in &dies[k] {
                if fixed[r].is_empty() {
                    free(&mut taken, r, &color);
                }
            }
            if let Some(r) = result_of(b, k) {
                alloc(&mut taken, &mut pool, r, &mut color);
                if dead_def[k] {
                    free(&mut taken, r, &color);
                }
            }
        }
    }

    // Values in unreachable blocks are never emitted: any local of the right
    // type will do.
    for v in 0..nv {
        if color[v].is_empty() && !wants[v].is_empty() {
            color[v] = wants[v]
                .iter()
                .map(|&t| {
                    let ti = type_index(t);
                    pool[ti] = pool[ti].max(1);
                    0
                })
                .collect();
        }
    }

    // Pools in type order after the parameters.
    let mut base = [0u32; 4];
    let mut locals = Vec::new();
    for ti in 0..4 {
        base[ti] = nparams + locals.len() as u32;
        locals.extend(std::iter::repeat_n(TYPES[ti], pool[ti] as usize));
    }
    let loc = (0..nv)
        .map(|v| {
            if !fixed[v].is_empty() {
                fixed[v].clone()
            } else {
                color[v].iter().zip(&wants[v]).map(|(&c, &t)| base[type_index(t)] + c).collect()
            }
        })
        .collect();
    Assignment { loc, locals }
}
