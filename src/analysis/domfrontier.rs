//! Dominance frontiers and iterated dominance frontiers (Cytron et al., 1991).
//!
//! The **dominance frontier** of a block `b`, `DF(b)`, is the set of blocks `y`
//! such that `b` dominates some predecessor of `y` but does *not* strictly
//! dominate `y` itself — informally, the blocks "just past" the region `b`
//! dominates. Its **iterated** closure over a set of definition blocks is
//! exactly the set of join points where φ-functions must be placed for a
//! variable defined in those blocks; in our block-argument SSA form (there are
//! no φ-nodes, see `docs/ir-design.md` §2) that means the blocks that need a new
//! **block parameter**. This is the engine `mem2reg` places block parameters
//! with ([`crate::transform::mem2reg`]).
//!
//! The computation is the classic one built on the immediate-dominator tree
//! ([`Dominators`]): for every join block `y` (two or more predecessors), walk up
//! the dominator tree from each predecessor until reaching `idom(y)`, adding `y`
//! to the frontier of every block on the way. Everything is indexed by dense
//! block indices and the frontier lists are sorted, so results are deterministic
//! (tenet T5).

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::Function;

/// The dominance frontier of every block in one function's CFG.
///
/// `frontier(b)` is `DF(b)`, sorted by block index. Unreachable blocks have an
/// empty frontier (they cannot be a dominator of any reachable predecessor).
#[derive(Debug, Clone)]
pub struct DominanceFrontiers {
    df: Vec<Vec<usize>>,
}

impl DominanceFrontiers {
    /// Compute the dominance frontiers of `func`, building the CFG and dominator
    /// tree internally. Convenience for callers that do not already hold them.
    pub fn of(func: &Function) -> DominanceFrontiers {
        let cfg = ControlFlowGraph::new(func);
        let doms = Dominators::new(func, &cfg);
        DominanceFrontiers::compute(&cfg, &doms, func.block_count())
    }

    /// Compute the dominance frontiers from a pre-built `cfg` and dominator tree
    /// `doms` over `n` blocks.
    pub fn compute(cfg: &ControlFlowGraph, doms: &Dominators, n: usize) -> DominanceFrontiers {
        let mut df: Vec<Vec<usize>> = vec![Vec::new(); n];
        for y in 0..n {
            let preds = cfg.predecessors(y);
            // A block enters a frontier only at a join point (≥ 2 predecessors).
            if preds.len() < 2 {
                continue;
            }
            let Some(idom_y) = doms.idom(y) else {
                continue; // y unreachable
            };
            for &p in preds {
                if !doms.is_reachable(p) {
                    continue;
                }
                // Walk up the dominator tree from p to idom(y); every block on
                // the way (p included) has y on its dominance frontier.
                let mut runner = p;
                while runner != idom_y {
                    df[runner].push(y);
                    match doms.idom(runner) {
                        Some(next) => runner = next,
                        None => break,
                    }
                }
            }
        }
        for f in &mut df {
            f.sort_unstable();
            f.dedup();
        }
        DominanceFrontiers { df }
    }

    /// The dominance frontier of block `b`, sorted by block index.
    #[inline]
    pub fn frontier(&self, b: usize) -> &[usize] {
        &self.df[b]
    }

    /// The **iterated** dominance frontier of a set of definition blocks: the
    /// least set closed under taking dominance frontiers. These are exactly the
    /// blocks that need a block parameter for a variable defined in `defs`.
    ///
    /// The result is sorted by block index (deterministic). Duplicate or
    /// unreachable entries in `defs` are harmless.
    pub fn iterated(&self, defs: &[usize]) -> Vec<usize> {
        let n = self.df.len();
        let mut in_idf = vec![false; n];
        let mut worklist: Vec<usize> = defs.to_vec();
        let mut result = Vec::new();
        while let Some(x) = worklist.pop() {
            if x >= n {
                continue;
            }
            for &y in &self.df[x] {
                if !in_idf[y] {
                    in_idf[y] = true;
                    result.push(y);
                    // y's own frontier may extend the set further.
                    worklist.push(y);
                }
            }
        }
        result.sort_unstable();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BlockId, FuncId, Module};
    use crate::support::StrInterner;

    /// A diamond: entry → {then, els} → merge.
    fn diamond() -> (Module, FuncId, [BlockId; 4]) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("df-diamond");
        let i1 = m.types_mut().bool();
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i1], i32t, false);
        let f = m.declare_function(syms.intern("d"), sig);
        let (entry, then_b, els_b, merge);
        {
            let mut b = m.build(f);
            entry = b.create_entry_block();
            then_b = b.create_block(&[]);
            els_b = b.create_block(&[]);
            merge = b.create_block(&[]);
            let c = b.param(entry, 0);
            b.switch_to(entry);
            b.cond_br(c, then_b, &[], els_b, &[]);
            b.switch_to(then_b);
            b.br(merge, &[]);
            b.switch_to(els_b);
            b.br(merge, &[]);
            b.switch_to(merge);
            let z = b.const_i64(i32t, 0);
            b.ret(Some(z));
        }
        (m, f, [entry, then_b, els_b, merge])
    }

    #[test]
    fn diamond_frontiers() {
        let (m, f, [entry, then_b, els_b, merge]) = diamond();
        let df = DominanceFrontiers::of(m.function(f));
        // then and els both flow into the join `merge`, which neither strictly
        // dominates: DF(then) = DF(els) = {merge}.
        assert_eq!(df.frontier(then_b.index()), &[merge.index()]);
        assert_eq!(df.frontier(els_b.index()), &[merge.index()]);
        // entry dominates everything, so its frontier is empty; merge has none.
        assert!(df.frontier(entry.index()).is_empty());
        assert!(df.frontier(merge.index()).is_empty());
        // IDF of the two arms is exactly the join point.
        assert_eq!(df.iterated(&[then_b.index(), els_b.index()]), vec![merge.index()]);
    }

    #[test]
    fn loop_frontiers() {
        // entry → header → {body → header (back edge), exit}
        let mut syms = StrInterner::new();
        let mut m = Module::new("df-loop");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("l"), sig);
        let (entry, header, body, exit);
        {
            let mut b = m.build(f);
            entry = b.create_entry_block();
            header = b.create_block(&[]);
            body = b.create_block(&[]);
            exit = b.create_block(&[]);
            b.switch_to(entry);
            b.br(header, &[]);
            b.switch_to(header);
            let t = b.const_bool(true);
            b.cond_br(t, body, &[], exit, &[]);
            b.switch_to(body);
            b.br(header, &[]);
            b.switch_to(exit);
            let z = b.const_i64(i32t, 0);
            b.ret(Some(z));
        }
        let df = DominanceFrontiers::of(m.function(f));
        // The back edge body → header makes header its own frontier's member for
        // both header and body (the classic loop result): DF(header) = {header},
        // DF(body) = {header}.
        assert_eq!(df.frontier(header.index()), &[header.index()]);
        assert_eq!(df.frontier(body.index()), &[header.index()]);
        assert!(df.frontier(entry.index()).is_empty());
        assert!(df.frontier(exit.index()).is_empty());
        // A variable defined in entry and body needs a parameter at the header.
        assert_eq!(df.iterated(&[entry.index(), body.index()]), vec![header.index()]);
    }
}
