//! Control-flow graph reachability and a dominator tree.
//!
//! The verifier's SSA-dominance check (`docs/ir-design.md` §2 block arguments
//! replace φ-nodes, but the *availability* rule is the same as any SSA form: a
//! value may only be used where its definition dominates the use) needs, per
//! function, which blocks are reachable from the entry and the immediate
//! dominator of each reachable block.
//!
//! [`DomTree::build`] computes both with the classic iterative
//! Cooper–Harvey–Kennedy "A Simple, Fast Dominance Algorithm" (2001): a
//! reverse-postorder fixpoint over the CFG using the `idom` intersection walk.
//! It is `O(N · E · d)` in the worst case but near-linear in practice and needs
//! no auxiliary data structures beyond the reverse-postorder numbering — a good
//! fit for a verifier that runs on every debug build.

use crate::ir::{BlockId, Function};

/// The immediate-dominator tree of one function's CFG, plus reachability.
///
/// Block indices are the dense `BlockId` indices (`0..func.block_count()`).
#[derive(Debug)]
pub(crate) struct DomTree {
    /// `idom[b]` is the immediate dominator of block `b`, or `None` when `b` is
    /// unreachable. The entry block is its own immediate dominator.
    idom: Vec<Option<usize>>,
    /// Whether each block is reachable from the entry.
    reachable: Vec<bool>,
}

impl DomTree {
    /// Compute the dominator tree of `func`. A function with no entry block
    /// yields an all-unreachable tree (every query returns `false`).
    pub(crate) fn build(func: &Function) -> DomTree {
        let n = func.block_count();
        let Some(entry) = func.entry().map(|e| e.index()) else {
            return DomTree { idom: vec![None; n], reachable: vec![false; n] };
        };

        // Successor lists, restricted to in-range targets (dangling successors
        // are a separate structural error and must not derail dominance).
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

        // Iterative DFS to a postorder of the reachable blocks.
        let mut visited = vec![false; n];
        let mut post = Vec::new();
        let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
        visited[entry] = true;
        while let Some(&(node, idx)) = stack.last() {
            if idx < succ[node].len() {
                stack.last_mut().unwrap().1 = idx + 1;
                let next = succ[node][idx];
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

        // Predecessors, over reachable blocks only.
        let mut preds = vec![Vec::new(); n];
        for b in 0..n {
            if !reachable[b] {
                continue;
            }
            for &s in &succ[b] {
                preds[s].push(b);
            }
        }

        // Cooper–Harvey–Kennedy fixpoint.
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
                for &p in &preds[b] {
                    if idom[p].is_some() {
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

        DomTree { idom, reachable }
    }

    /// Whether block `b` is reachable from the entry.
    pub(crate) fn is_reachable(&self, b: usize) -> bool {
        self.reachable.get(b).copied().unwrap_or(false)
    }

    /// Whether block `a` dominates block `b` (reflexively: a block dominates
    /// itself). Returns `false` if either block is unreachable or out of range.
    pub(crate) fn dominates(&self, a: usize, b: usize) -> bool {
        if a >= self.idom.len() || b >= self.idom.len() {
            return false;
        }
        if !self.reachable[a] || !self.reachable[b] {
            return false;
        }
        let mut x = b;
        loop {
            if x == a {
                return true;
            }
            match self.idom[x] {
                Some(i) if i != x => x = i,
                // Reached the entry (its own idom) without meeting `a`.
                _ => return false,
            }
        }
    }
}

/// Walk two dominator fingers up to their common ancestor, per Cooper–Harvey–
/// Kennedy. `rpo_index` orders blocks so a block's idom always has a strictly
/// smaller index, which makes the walk terminate.
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
