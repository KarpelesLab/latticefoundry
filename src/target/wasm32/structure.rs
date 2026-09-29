//! Control-flow **structuring**: turning a function's control-flow graph into
//! WebAssembly's nested `block` / `loop` / `if` / `br` form.
//!
//! WebAssembly has no `goto`: a branch names an *enclosing* construct by its
//! nesting depth, and jumps to the end of a `block` (forward) or to the start of
//! a `loop` (backward). Every reducible CFG can be expressed that way without
//! duplicating code; an irreducible one (a cycle with more than one entry)
//! cannot, and falls back to a dispatch loop.
//!
//! # The reducible case: dominator-tree placement
//!
//! The placement below is derived from the dominator tree and a reverse
//! postorder (RPO) numbering, both classic results (the dominators are computed
//! with the Cooper–Harvey–Kennedy iteration):
//!
//! - An edge `u → v` is **backward** when `rpo(v) ≤ rpo(u)`. In a reducible CFG
//!   every backward edge goes to a node that dominates its source: a **loop
//!   header**. The header's code is wrapped in a `loop`, and a backward edge is
//!   a `br` to it.
//! - A node with two or more *forward* in-edges is a **merge node**. It cannot
//!   be emitted inline at a branch (that would duplicate it), so its immediate
//!   dominator `d` places it: `d`'s code is wrapped in a `block` that ends just
//!   before the merge node's code, and every forward edge to it is a `br` out of
//!   that block. When `d` dominates several merge nodes they get nested blocks,
//!   the one latest in RPO outermost, so that each block's end falls into its
//!   node and every branch to it comes from inside the block.
//! - A node with a single forward in-edge is emitted **inline**, right where
//!   that edge is taken (its immediate dominator is the edge's source).
//!
//! Every reachable node is thus emitted exactly once, and each branch's target
//! is found by searching the stack of enclosing constructs, which gives its
//! `br` depth directly.
//!
//! # Terminator shapes
//!
//! A node leaves through *arms* (its distinct outgoing edges) in one of three
//! [`Shape`]s, which fix how many anonymous constructs sit between an arm's
//! code and the enclosing context (and so the depths of branches inside it):
//!
//! - [`Shape::Direct`] — one arm, no construct;
//! - [`Shape::IfElse`] — two arms in the `then` and `else` of an `if` (one
//!   construct around each); the condition selects arm 0 when nonzero;
//! - [`Shape::Table`] — `n` arms behind `n` nested blocks: the dispatch sits in
//!   the innermost block and reaches arm `i` with `br i`; arm `i`'s code follows
//!   the end of block `i`, still inside the `n - 1 - i` outer blocks.
//!
//! # The irreducible case: a dispatch loop
//!
//! If some backward edge targets a node that does not dominate its source, the
//! whole function becomes a [`Structured::Dispatch`]: a label variable holds
//! the index of the node to run next; a `loop` around `n` nested blocks and a
//! `br_table` on the label selects the node, and every edge sets the label and
//! branches back to the loop. It costs one indirect branch per edge but handles
//! any CFG.

/// How a node's terminator selects among its arms (see the [module
/// docs](self)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    /// No arms (`return`, `unreachable`) or one arm, taken unconditionally.
    Direct,
    /// Two arms: `if` (arm 0) / `else` (arm 1).
    IfElse,
    /// Any number of arms behind nested blocks (a `br_table` or a `br_if`
    /// chain dispatches to them).
    Table,
}

impl Shape {
    /// How many anonymous constructs enclose arm `i` of `n`.
    pub fn frames(self, i: usize, n: usize) -> u32 {
        match self {
            Shape::Direct => 0,
            Shape::IfElse => 1,
            Shape::Table => (n - 1 - i) as u32,
        }
    }
}

/// One node of the input graph: its arms (the target node of each distinct
/// outgoing edge, in arm order) and how it dispatches to them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CfgNode {
    /// The target of each arm.
    pub arms: Vec<usize>,
    /// The dispatch shape ([`Shape::IfElse`] needs exactly two arms).
    pub shape: Shape,
}

/// What an arm does after its edge's block arguments are assigned.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Arm {
    /// Branch to the construct at this depth (relative to the arm's code).
    Br(u32),
    /// Continue with the target's code, emitted right here.
    Inline(Vec<Structured>),
    /// Fall through to the next item of the enclosing sequence, which is the
    /// target's code: the arm of a [`Shape::Direct`] node whose target would
    /// be inlined and needs no construct of its own. Chains of such nodes are
    /// thus flat sequences rather than nested arms, which keeps the recursion
    /// of both the structurizer and the emitter bounded by the real nesting.
    Next,
    /// Dispatch mode: set the label variable to `target` (a dispatch index)
    /// and branch to the dispatch loop at `depth`.
    Goto {
        /// The dispatch index of the target node.
        target: u32,
        /// The depth of the dispatch `loop`, relative to the arm's code.
        depth: u32,
    },
}

/// The structured form of a function (or part of one).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Structured {
    /// A `block … end`: a branch to it continues after its end.
    Block(Vec<Structured>),
    /// A `loop … end`: a branch to it restarts it.
    Loop(Vec<Structured>),
    /// A node's own code, then its terminator dispatching to `arms` (one per
    /// input arm, in order) in the node's [`Shape`].
    Node {
        /// The input node.
        node: usize,
        /// What each arm does.
        arms: Vec<Arm>,
    },
    /// The irreducible fallback (see the [module docs](self)): the label
    /// starts at 0; `order[i]` is the node at dispatch index `i` (the entry
    /// first), and `arms[i]` are its arms, each an [`Arm::Goto`].
    Dispatch {
        /// Dispatch index → node.
        order: Vec<usize>,
        /// The arms of each node, by dispatch index.
        arms: Vec<Vec<Arm>>,
    },
}

/// One enclosing construct during structuring.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Frame {
    /// A `loop` whose header is this node (backward edges target it).
    Loop(usize),
    /// A `block` whose end falls into this merge node.
    Block(usize),
    /// A construct no branch targets (an `if`, a dispatch block).
    Anon,
}

/// The analysis the placement needs, over the nodes reachable from node 0.
struct Graph<'a> {
    nodes: &'a [CfgNode],
    /// RPO number of each node (`usize::MAX` when unreachable).
    rpo: Vec<usize>,
    /// The reachable nodes in RPO.
    order: Vec<usize>,
    /// Immediate dominator (the entry is its own).
    idom: Vec<usize>,
    loop_header: Vec<bool>,
    merge: Vec<bool>,
    /// Dominator-tree children that are merge nodes, latest RPO first.
    merge_children: Vec<Vec<usize>>,
}

/// Structure the graph `nodes` (entry: node 0). Nodes unreachable from the
/// entry are left out of the result.
///
/// # Panics
///
/// If `nodes` is empty, an arm names a missing node, or an [`Shape::IfElse`]
/// node does not have exactly two arms.
pub fn structurize(nodes: &[CfgNode]) -> Vec<Structured> {
    assert!(!nodes.is_empty(), "structurize: empty graph");
    for (i, n) in nodes.iter().enumerate() {
        assert!(n.arms.iter().all(|&t| t < nodes.len()), "structurize: node {i} has an arm to a missing node");
        assert!(n.shape != Shape::IfElse || n.arms.len() == 2, "structurize: if/else node {i} needs two arms");
    }
    let g = Graph::new(nodes);
    if !g.is_reducible() {
        return vec![g.dispatch()];
    }
    let mut ctx = Vec::new();
    g.do_tree(0, &mut ctx)
}

/// Whether the graph `nodes` (entry: node 0) is reducible, i.e. whether
/// [`structurize`] places it without a dispatch loop.
pub fn is_reducible(nodes: &[CfgNode]) -> bool {
    Graph::new(nodes).is_reducible()
}

impl<'a> Graph<'a> {
    fn new(nodes: &'a [CfgNode]) -> Graph<'a> {
        let n = nodes.len();
        // Iterative DFS postorder from the entry.
        let mut visited = vec![false; n];
        let mut post = Vec::with_capacity(n);
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        visited[0] = true;
        while let Some(top) = stack.last_mut() {
            let (node, idx) = *top;
            if let Some(&next) = nodes[node].arms.get(idx) {
                top.1 += 1;
                if !visited[next] {
                    visited[next] = true;
                    stack.push((next, 0));
                }
            } else {
                post.push(node);
                stack.pop();
            }
        }
        let order: Vec<usize> = post.into_iter().rev().collect();
        let mut rpo = vec![usize::MAX; n];
        for (i, &b) in order.iter().enumerate() {
            rpo[b] = i;
        }

        // Predecessors among reachable nodes (an arm list may repeat a target).
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &u in &order {
            for &v in &nodes[u].arms {
                preds[v].push(u);
            }
        }

        // Cooper–Harvey–Kennedy: iterate idom[b] = ⋂ idom over processed preds.
        let mut idom = vec![usize::MAX; n];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in order.iter().skip(1) {
                let mut new = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    new = if new == usize::MAX { p } else { intersect(&idom, &rpo, p, new) };
                }
                if new != idom[b] {
                    idom[b] = new;
                    changed = true;
                }
            }
        }

        let mut loop_header = vec![false; n];
        let mut forward_in = vec![0usize; n];
        for &u in &order {
            for &v in &nodes[u].arms {
                if rpo[v] <= rpo[u] {
                    loop_header[v] = true;
                } else {
                    forward_in[v] += 1;
                }
            }
        }
        let merge: Vec<bool> = forward_in.iter().map(|&c| c >= 2).collect();
        let mut merge_children = vec![Vec::new(); n];
        for &b in order.iter().skip(1) {
            if merge[b] {
                merge_children[idom[b]].push(b);
            }
        }
        for kids in &mut merge_children {
            kids.sort_by_key(|&k| std::cmp::Reverse(rpo[k]));
        }
        Graph { nodes, rpo, order, idom, loop_header, merge, merge_children }
    }

    /// Whether `a` dominates `b` (both reachable).
    fn dominates(&self, a: usize, mut b: usize) -> bool {
        loop {
            if a == b {
                return true;
            }
            if b == 0 {
                return false;
            }
            b = self.idom[b];
        }
    }

    /// Reducible iff every backward edge targets a dominator of its source.
    fn is_reducible(&self) -> bool {
        self.order.iter().all(|&u| {
            self.nodes[u].arms.iter().all(|&v| self.rpo[v] > self.rpo[u] || self.dominates(v, u))
        })
    }

    fn do_tree(&self, x: usize, ctx: &mut Vec<Frame>) -> Vec<Structured> {
        let merges = &self.merge_children[x];
        if self.loop_header[x] {
            ctx.push(Frame::Loop(x));
            let body = self.node_within(x, merges, ctx);
            ctx.pop();
            vec![Structured::Loop(body)]
        } else {
            self.node_within(x, merges, ctx)
        }
    }

    /// `x`'s code with the merge children `ys` (latest RPO first) placed after
    /// it, each behind a block that `x`'s code branches out of.
    fn node_within(&self, x: usize, ys: &[usize], ctx: &mut Vec<Frame>) -> Vec<Structured> {
        let Some((&y, rest)) = ys.split_first() else {
            // `x`, then as long as it falls straight into a node that is
            // neither a loop header nor places merge nodes, that node.
            let mut out = Vec::new();
            let mut cur = x;
            loop {
                let node = &self.nodes[cur];
                if let [y] = node.arms[..]
                    && node.shape == Shape::Direct
                    && self.rpo[y] > self.rpo[cur]
                    && !self.merge[y]
                    && !self.loop_header[y]
                    && self.merge_children[y].is_empty()
                {
                    out.push(Structured::Node { node: cur, arms: vec![Arm::Next] });
                    cur = y;
                    continue;
                }
                out.push(self.node(cur, ctx));
                return out;
            }
        };
        ctx.push(Frame::Block(y));
        let inner = self.node_within(x, rest, ctx);
        ctx.pop();
        let mut out = vec![Structured::Block(inner)];
        out.extend(self.do_tree(y, ctx));
        out
    }

    fn node(&self, x: usize, ctx: &mut Vec<Frame>) -> Structured {
        let node = &self.nodes[x];
        let n = node.arms.len();
        let mut arms = Vec::with_capacity(n);
        for (i, &y) in node.arms.iter().enumerate() {
            let extra = node.shape.frames(i, n);
            for _ in 0..extra {
                ctx.push(Frame::Anon);
            }
            arms.push(self.do_branch(x, y, ctx));
            ctx.truncate(ctx.len() - extra as usize);
        }
        Structured::Node { node: x, arms }
    }

    fn do_branch(&self, x: usize, y: usize, ctx: &mut Vec<Frame>) -> Arm {
        let backward = self.rpo[y] <= self.rpo[x];
        if backward || self.merge[y] {
            let want = if backward { Frame::Loop(y) } else { Frame::Block(y) };
            let pos = ctx.iter().rposition(|&f| f == want).expect("structurize: branch target is not in scope");
            Arm::Br((ctx.len() - 1 - pos) as u32)
        } else {
            Arm::Inline(self.do_tree(y, ctx))
        }
    }

    /// The dispatch-loop form of the whole graph.
    fn dispatch(&self) -> Structured {
        let n = self.order.len();
        let arms = self
            .order
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let node = &self.nodes[x];
                let k = node.arms.len();
                node.arms
                    .iter()
                    .enumerate()
                    .map(|(a, &y)| Arm::Goto {
                        target: self.rpo[y] as u32,
                        depth: (n - 1 - i) as u32 + node.shape.frames(a, k),
                    })
                    .collect()
            })
            .collect();
        Structured::Dispatch { order: self.order.clone(), arms }
    }
}

/// The Cooper–Harvey–Kennedy finger walk to the nearest common dominator.
fn intersect(idom: &[usize], rpo: &[usize], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while rpo[a] > rpo[b] {
            a = idom[a];
        }
        while rpo[b] > rpo[a] {
            b = idom[b];
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What running a structured program (or walking a CFG) did: the nodes
    /// visited, in order.
    type Trace = Vec<usize>;

    /// How a structured sequence finished.
    enum Flow {
        Normal,
        Br(u32),
        Exit,
    }

    /// An interpreter for [`Structured`] programs: `choose(node, visit)` picks
    /// the arm a node takes on its `visit`-th execution (so the same choices
    /// can drive the CFG walk). Stops after `budget` node visits.
    struct Sim<'a> {
        nodes: &'a [CfgNode],
        choose: &'a dyn Fn(usize, usize) -> usize,
        visits: Vec<usize>,
        trace: Trace,
        budget: usize,
    }

    impl Sim<'_> {
        /// Visit `node`: record it and pick an arm, or `None` to stop (an exit
        /// node, or the budget ran out).
        fn visit(&mut self, node: usize) -> Option<usize> {
            if self.trace.len() >= self.budget {
                return None;
            }
            self.trace.push(node);
            let k = self.nodes[node].arms.len();
            let v = self.visits[node];
            self.visits[node] += 1;
            if k == 0 { None } else { Some((self.choose)(node, v) % k) }
        }

        fn seq(&mut self, items: &[Structured]) -> Flow {
            for item in items {
                match self.item(item) {
                    Flow::Normal => {}
                    other => return other,
                }
            }
            Flow::Normal
        }

        fn item(&mut self, item: &Structured) -> Flow {
            match item {
                Structured::Block(body) => match self.seq(body) {
                    Flow::Br(0) | Flow::Normal => Flow::Normal,
                    Flow::Br(d) => Flow::Br(d - 1),
                    Flow::Exit => Flow::Exit,
                },
                Structured::Loop(body) => loop {
                    match self.seq(body) {
                        Flow::Br(0) => continue,
                        Flow::Br(d) => return Flow::Br(d - 1),
                        Flow::Normal => return Flow::Normal,
                        Flow::Exit => return Flow::Exit,
                    }
                },
                Structured::Node { node, arms } => {
                    let Some(a) = self.visit(*node) else { return Flow::Exit };
                    let shape = self.nodes[*node].shape;
                    let extra = shape.frames(a, arms.len());
                    let flow = match &arms[a] {
                        Arm::Br(d) => Flow::Br(*d),
                        Arm::Inline(body) => self.seq(body),
                        // The only arm allowed to continue with what follows.
                        Arm::Next => {
                            assert_eq!(shape, Shape::Direct, "fall-through from a non-direct node");
                            return Flow::Normal;
                        }
                        Arm::Goto { .. } => panic!("goto outside a dispatch loop"),
                    };
                    match flow {
                        Flow::Br(d) => {
                            assert!(d >= extra, "branch to an anonymous construct");
                            Flow::Br(d - extra)
                        }
                        Flow::Normal => panic!("arm {a} of node {node} fell through"),
                        Flow::Exit => Flow::Exit,
                    }
                }
                Structured::Dispatch { order, arms } => {
                    let n = order.len();
                    let mut label = 0usize;
                    loop {
                        let node = order[label];
                        let Some(a) = self.visit(node) else { return Flow::Exit };
                        let Arm::Goto { target, depth } = arms[label][a] else { panic!("non-goto arm in dispatch") };
                        let extra = self.nodes[node].shape.frames(a, arms[label].len());
                        assert_eq!(depth, (n - 1 - label) as u32 + extra, "dispatch depth");
                        label = target as usize;
                    }
                }
            }
        }
    }

    /// Walk the CFG itself with the same choices.
    fn walk(nodes: &[CfgNode], choose: &dyn Fn(usize, usize) -> usize, budget: usize) -> Trace {
        let mut visits = vec![0usize; nodes.len()];
        let mut trace = Vec::new();
        let mut at = 0usize;
        while trace.len() < budget {
            trace.push(at);
            let k = nodes[at].arms.len();
            if k == 0 {
                break;
            }
            let a = choose(at, visits[at]) % k;
            visits[at] += 1;
            at = nodes[at].arms[a];
        }
        trace
    }

    fn run(nodes: &[CfgNode], program: &[Structured], choose: &dyn Fn(usize, usize) -> usize, budget: usize) -> Trace {
        let mut sim = Sim { nodes, choose, visits: vec![0; nodes.len()], trace: Vec::new(), budget };
        match sim.seq(program) {
            Flow::Exit => {}
            Flow::Normal => panic!("program fell off its end"),
            Flow::Br(d) => panic!("branch out of the function (depth {d})"),
        }
        sim.trace
    }

    /// Each node emitted exactly once (reducible form), counted recursively.
    fn count_nodes(items: &[Structured], seen: &mut Vec<usize>) {
        for item in items {
            match item {
                Structured::Block(b) | Structured::Loop(b) => count_nodes(b, seen),
                Structured::Node { node, arms } => {
                    seen.push(*node);
                    for a in arms {
                        if let Arm::Inline(b) = a {
                            count_nodes(b, seen);
                        }
                    }
                }
                Structured::Dispatch { order, .. } => seen.extend(order),
            }
        }
    }

    fn node(arms: &[usize]) -> CfgNode {
        let shape = if arms.len() == 2 { Shape::IfElse } else if arms.len() > 2 { Shape::Table } else { Shape::Direct };
        CfgNode { arms: arms.to_vec(), shape }
    }

    fn check_all_choices(nodes: &[CfgNode]) {
        let program = structurize(nodes);
        let mut seen = Vec::new();
        count_nodes(&program, &mut seen);
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), seen.len(), "a node was emitted twice: {program:?}");
        for seed in 0..40u64 {
            let choose = move |node: usize, visit: usize| {
                let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ ((node as u64) << 32) ^ visit as u64;
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (x >> 33) as usize
            };
            let want = walk(nodes, &choose, 64);
            let got = run(nodes, &program, &choose, 64);
            assert_eq!(got, want, "seed {seed}: {program:?}");
        }
    }

    #[test]
    fn straight_line_and_diamond() {
        // 0 -> 1 -> 2 (exit): a flat sequence.
        let g = [node(&[1]), node(&[2]), node(&[])];
        let p = structurize(&g);
        assert_eq!(
            p,
            vec![
                Structured::Node { node: 0, arms: vec![Arm::Next] },
                Structured::Node { node: 1, arms: vec![Arm::Next] },
                Structured::Node { node: 2, arms: vec![] },
            ]
        );
        check_all_choices(&g);
        // A long chain does not nest (nor recurse).
        let mut chain: Vec<CfgNode> = (1..20_000).map(|i| node(&[i])).collect();
        chain.push(node(&[]));
        let p = structurize(&chain);
        assert_eq!(p.len(), 20_000);
        // Diamond: 0 -> {1, 2} -> 3. 3 is a merge node placed after a block.
        let g = [node(&[1, 2]), node(&[3]), node(&[3]), node(&[])];
        let p = structurize(&g);
        assert_eq!(
            p,
            vec![
                Structured::Block(vec![Structured::Node {
                    node: 0,
                    arms: vec![
                        Arm::Inline(vec![Structured::Node { node: 1, arms: vec![Arm::Br(1)] }]),
                        Arm::Inline(vec![Structured::Node { node: 2, arms: vec![Arm::Br(1)] }]),
                    ]
                }]),
                Structured::Node { node: 3, arms: vec![] },
            ]
        );
        check_all_choices(&g);
    }

    #[test]
    fn while_loop() {
        // 0 -> 1 (header) -> {2 (body) -> 1, 3 (exit)}
        let g = [node(&[1]), node(&[2, 3]), node(&[1]), node(&[])];
        let p = structurize(&g);
        assert_eq!(
            p,
            vec![Structured::Node {
                node: 0,
                arms: vec![Arm::Inline(vec![Structured::Loop(vec![Structured::Node {
                    node: 1,
                    arms: vec![
                        Arm::Inline(vec![Structured::Node { node: 2, arms: vec![Arm::Br(1)] }]),
                        Arm::Inline(vec![Structured::Node { node: 3, arms: vec![] }]),
                    ]
                }])])]
            }]
        );
        check_all_choices(&g);
    }

    #[test]
    fn nested_loops_breaks_and_continues() {
        // Outer loop 1..5 with an inner loop 2..3, a `continue` from 3 to 1
        // and a `break` from 2 to 5.
        let g = [
            node(&[1]),       // 0
            node(&[2, 5]),    // 1: outer header
            node(&[3, 5]),    // 2: inner header, break out
            node(&[2, 4, 1]), // 3: inner latch, fall out, continue outer
            node(&[1]),       // 4: outer latch
            node(&[]),        // 5: exit
        ];
        assert!(is_reducible(&g));
        check_all_choices(&g);
    }

    #[test]
    fn self_loops_and_duplicate_arms() {
        // A self loop, and a node whose two arms reach the same target.
        let g = [node(&[0, 1]), node(&[2, 2]), node(&[])];
        check_all_choices(&g);
        let t = CfgNode { arms: vec![1, 2, 1, 3], shape: Shape::Table };
        let g = [t, node(&[3]), node(&[3]), node(&[0, 4]), node(&[])];
        check_all_choices(&g);
    }

    #[test]
    fn irreducible_uses_a_dispatch_loop() {
        // 0 -> {1, 2}, 1 <-> 2: the cycle has two entries.
        let g = [node(&[1, 2]), node(&[2, 3]), node(&[1, 3]), node(&[])];
        assert!(!is_reducible(&g));
        let p = structurize(&g);
        assert!(matches!(p.as_slice(), [Structured::Dispatch { .. }]), "{p:?}");
        check_all_choices(&g);
    }

    #[test]
    fn unreachable_nodes_are_dropped() {
        let g = [node(&[2]), node(&[2]), node(&[])];
        let p = structurize(&g);
        let mut seen = Vec::new();
        count_nodes(&p, &mut seen);
        seen.sort_unstable();
        assert_eq!(seen, [0, 2]);
    }

    /// Random graphs, reducible and not: the structured program visits the
    /// same nodes as the CFG under every choice sequence tried.
    #[test]
    fn random_graphs_simulate_like_their_cfg() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let (mut reducible, mut irreducible) = (0, 0);
        for _ in 0..2000 {
            let n = 2 + (next() % 11) as usize;
            let nodes: Vec<CfgNode> = (0..n)
                .map(|i| {
                    // The last node always exits so most graphs terminate.
                    let k = if i == n - 1 { 0 } else { (next() % 4) as usize };
                    let arms: Vec<usize> = (0..k).map(|_| (next() % n as u64) as usize).collect();
                    let shape = match arms.len() {
                        2 if next() % 2 == 0 => Shape::IfElse,
                        0 | 1 => Shape::Direct,
                        _ => Shape::Table,
                    };
                    CfgNode { arms, shape }
                })
                .collect();
            if is_reducible(&nodes) {
                reducible += 1;
            } else {
                irreducible += 1;
            }
            check_all_choices(&nodes);
        }
        assert!(reducible > 500 && irreducible > 40, "coverage: {reducible} reducible, {irreducible} irreducible");
    }
}
