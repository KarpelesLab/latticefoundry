//! Control-flow structure the fixpoint engine needs: per-block successor and
//! predecessor lists, and a dominator tree with loop-header detection.
//!
//! The dominator tree uses the classic iterative Cooper–Harvey–Kennedy "A
//! Simple, Fast Dominance Algorithm" (2001) — a reverse-postorder fixpoint with
//! the `idom` intersection walk. The verifier carries its own private copy for
//! SSA-dominance checking; the algorithm is settled CS (used freely per the
//! clean-room policy) and re-derived here so the analysis layer does not reach
//! into `verify` internals.
//!
//! Everything is indexed by dense `BlockId` indices and built in block order, so
//! results are deterministic (tenet T5).

use crate::ir::{BlockId, Function};

/// The successor and predecessor structure of one function's CFG, over the
/// reachable and unreachable blocks alike (edges to out-of-range targets are
/// dropped, matching the verifier's tolerance of malformed successors).
#[derive(Debug)]
pub struct ControlFlowGraph {
    succ: Vec<Vec<usize>>,
    preds: Vec<Vec<usize>>,
}

impl ControlFlowGraph {
    /// Build the CFG of `func`.
    pub fn new(func: &Function) -> ControlFlowGraph {
        let n = func.block_count();
        let succ: Vec<Vec<usize>> = (0..n)
            .map(|b| match func.block(BlockId::from_index(b)).terminator() {
                Some(t) => func
                    .inst(t)
                    .successors()
                    .into_iter()
                    .map(|s| s.index())
                    .filter(|&s| s < n)
                    .collect(),
                None => Vec::new(),
            })
            .collect();
        let mut preds = vec![Vec::new(); n];
        for (b, ss) in succ.iter().enumerate() {
            for &s in ss {
                preds[s].push(b);
            }
        }
        ControlFlowGraph { succ, preds }
    }

    /// The successor block indices of block `b`, in edge order.
    #[inline]
    pub fn successors(&self, b: usize) -> &[usize] {
        &self.succ[b]
    }

    /// The predecessor block indices of block `b`.
    #[inline]
    pub fn predecessors(&self, b: usize) -> &[usize] {
        &self.preds[b]
    }
}

/// The immediate-dominator tree of one function's CFG, plus reachability and
/// loop-header identification.
#[derive(Debug)]
pub struct Dominators {
    /// `idom[b]` is the immediate dominator of `b`, or `None` if unreachable.
    /// The entry is its own immediate dominator.
    idom: Vec<Option<usize>>,
    reachable: Vec<bool>,
    loop_header: Vec<bool>,
}

impl Dominators {
    /// Compute the dominator tree of `func` from its `cfg`. A function with no
    /// entry yields an all-unreachable tree.
    pub fn new(func: &Function, cfg: &ControlFlowGraph) -> Dominators {
        let n = func.block_count();
        let Some(entry) = func.entry().map(|e| e.index()) else {
            return Dominators {
                idom: vec![None; n],
                reachable: vec![false; n],
                loop_header: vec![false; n],
            };
        };

        // Iterative DFS postorder over the reachable blocks.
        let mut visited = vec![false; n];
        let mut post = Vec::new();
        let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
        visited[entry] = true;
        while let Some(&(node, idx)) = stack.last() {
            let succ = cfg.successors(node);
            if idx < succ.len() {
                stack.last_mut().unwrap().1 = idx + 1;
                let next = succ[idx];
                if !visited[next] {
                    visited[next] = true;
                    stack.push((next, 0));
                }
            } else {
                post.push(node);
                stack.pop();
            }
        }
        let reachable = visited;

        // Reverse postorder and its inverse numbering.
        let rpo: Vec<usize> = post.iter().rev().copied().collect();
        let mut rpo_index = vec![usize::MAX; n];
        for (i, &b) in rpo.iter().enumerate() {
            rpo_index[b] = i;
        }

        // Cooper–Harvey–Kennedy fixpoint over reachable predecessors.
        let mut idom = vec![None; n];
        idom[entry] = Some(entry);
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &rpo {
                if b == entry {
                    continue;
                }
                let mut new_idom: Option<usize> = None;
                for &p in cfg.predecessors(b) {
                    if reachable[p] && idom[p].is_some() {
                        new_idom = Some(match new_idom {
                            None => p,
                            Some(cur) => intersect(&idom, &rpo_index, p, cur),
                        });
                    }
                }
                if new_idom != idom[b] {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        // A block `h` is a loop header iff some reachable predecessor `p` is
        // dominated by `h` (the edge `p → h` is a back edge).
        let mut dom = Dominators { idom, reachable, loop_header: vec![false; n] };
        for b in 0..n {
            if !dom.reachable[b] {
                continue;
            }
            for &p in cfg.predecessors(b) {
                if dom.reachable[p] && dom.dominates(b, p) {
                    dom.loop_header[b] = true;
                    break;
                }
            }
        }
        dom
    }

    /// Whether block `b` is reachable from the entry.
    #[inline]
    pub fn is_reachable(&self, b: usize) -> bool {
        self.reachable.get(b).copied().unwrap_or(false)
    }

    /// The immediate dominator of block `b` — the entry is its own immediate
    /// dominator; an unreachable or out-of-range block has `None`. This is the
    /// hook the dominance-frontier computation ([`crate::analysis::domfrontier`])
    /// climbs.
    #[inline]
    pub fn idom(&self, b: usize) -> Option<usize> {
        self.idom.get(b).copied().flatten()
    }

    /// Whether block `h` is a loop header (the target of a back edge).
    #[inline]
    pub fn is_loop_header(&self, h: usize) -> bool {
        self.loop_header.get(h).copied().unwrap_or(false)
    }

    /// Whether block `a` dominates block `b` (reflexively). `false` if either is
    /// unreachable or out of range.
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        if a >= self.idom.len() || b >= self.idom.len() || !self.reachable[a] || !self.reachable[b] {
            return false;
        }
        let mut x = b;
        loop {
            if x == a {
                return true;
            }
            match self.idom[x] {
                Some(i) if i != x => x = i,
                _ => return false,
            }
        }
    }
}

/// Walk two dominator fingers up to their common ancestor (Cooper–Harvey–
/// Kennedy). `rpo_index` orders blocks so a block's idom has a strictly smaller
/// index, which makes the walk terminate.
fn intersect(idom: &[Option<usize>], rpo_index: &[usize], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while rpo_index[a] > rpo_index[b] {
            a = idom[a].expect("processed node has an immediate dominator");
        }
        while rpo_index[b] > rpo_index[a] {
            b = idom[b].expect("processed node has an immediate dominator");
        }
    }
    a
}
