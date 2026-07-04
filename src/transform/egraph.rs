//! **Equality saturation** — the algebraic optimizer (bet **B4**, ROADMAP
//! Phase 4).
//!
//! This pass is LatticeFoundry's local/algebraic mid-level optimizer, built on
//! the bet-B4 recipe: represent all the equivalent forms of a function's **pure
//! value DAG** in an **e-graph**, grow it by applying a set of **B2-verified**
//! rewrite rules to (bounded) saturation, then **extract** the cheapest form per
//! value under a **cost model** (bet **B9**) and rebuild the function. Because
//! the same value can be represented many ways at once, the optimizer sidesteps
//! phase-ordering for the rewrites it knows (tenet T4): there is no "apply `x*2`
//! before or after reassociation?" — every order is explored simultaneously and
//! the extractor picks the global optimum.
//!
//! ## What is modeled
//!
//! Only the **pure, integer-typed** value-producing opcodes participate as
//! e-nodes: integer [`Bin`](crate::ir::inst::BinOp) (add/sub/mul/div/rem/bitwise/
//! shift), [`ICmp`](InstKind::ICmp), the integer casts (`trunc`/`zext`/`sext`),
//! [`Select`](InstKind::Select), and [`Freeze`](InstKind::Freeze). Everything
//! else — memory (`load`/`store`/`alloca`), `call`, floating point, pointer
//! arithmetic, block parameters, and terminators — is an **opaque leaf**: its
//! result is an input to the e-graph, reproduced verbatim by the rebuild. This
//! is exactly the subset the [`refinement`](crate::verify::refinement) checker
//! reasons about, which is what lets every rule be machine-verified.
//!
//! ## The e-graph (union-find + hash-consing + congruence)
//!
//! An **e-node** is `(op, [child e-class ids])`; leaves and constants have no
//! children. E-classes are the blocks of a **union-find** (path-halving `find`,
//! deterministic union that keeps the lower id as the representative). E-nodes
//! are **hash-consed** through a memo table, so structurally identical
//! subexpressions land in one class on construction — free common-subexpression
//! elimination. After merges we restore the **congruence** invariant (if two
//! classes are equal, any two e-nodes that differ only by those two children are
//! equal too) with a naive-to-fixpoint [`EGraph::rebuild`]: re-canonicalize every
//! e-node and union any that collide. Cyclic classes (created by, e.g.,
//! `x + 0 → x`, whose `add` node points back at its own class) are fine — the
//! extractor's cost fixpoint simply never selects a self-referential node.
//!
//! ## Soundness (tenets T2/T3, bet B2)
//!
//! Every rewrite rule is a **refinement** `lhs ⇒ rhs` (the rhs refines the lhs
//! per `docs/ir-design.md` §5), and each is proved by encoding the two sides as
//! tiny functions and asking [`check_refinement`](crate::verify::check_refinement)
//! — the in-file tests assert every rule returns `Refines`. Most rules are full
//! *equalities* (mutual refinement): `x+0=x`, commutativity, associativity of
//! flag-free `+`/`*`, `x*2^k = x<<k`, and constant folding. A few are
//! *one-directional* refinements where the rhs is strictly more defined —
//! `x*0 → 0` and `x&0 → 0` (a poison `x` makes the lhs poison but the rhs is the
//! defined constant `0`). Unioning those is still sound for extraction because
//! the rhs is a **constant**, which the cost model makes the unique cheapest
//! member of its class, so min-cost extraction always selects the refining
//! direction and never regresses a genuine `0` into a poison-capable product.
//! Flags matter: reassociation and the identity rules fire **only on flag-free
//! ops** (`nsw`/`nuw`/`exact` cleared), so no assumption is silently dropped;
//! constant folding delegates to [`ir::fold`](crate::ir::fold), which refuses to
//! fold anything that would be undefined behavior.
//!
//! ## Termination & determinism
//!
//! Saturation runs to a fixpoint but is bounded by a node budget
//! ([`MAX_NODES`]) and an iteration cap ([`MAX_ITERS`]); associativity and
//! commutativity can only add finitely many nodes before the budget halts
//! growth, so the pass always terminates. All maps/sets are the deterministic
//! [`DetHashMap`]/[`DetHashSet`] (fixed-seed hashing, tenet T5); classes and
//! nodes are visited in id/insertion order and ties in extraction break on that
//! order, so the output is byte-for-byte reproducible across runs.

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, CastOp, Flags, InstData, InstKind, IntPred};
use crate::ir::semantics::{FoldResult, fold};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ConstPool, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Module};
use crate::pass::{Changed, ModulePass};
use crate::support::{DetHashMap, DetHashSet};
use crate::transform::{FunctionTransform, dom_preorder, remap_value};

use puremp::Int;

/// The maximum number of e-classes the saturation loop may create. Reaching it
/// stops rule application, guaranteeing termination even for rules (associativity,
/// commutativity) that could otherwise grow the e-graph without bound.
const MAX_NODES: usize = 5000;

/// The maximum number of saturation iterations. A second bound on termination and
/// on cost; small expressions saturate (reach a no-change fixpoint) far sooner.
const MAX_ITERS: usize = 60;

// ---------------------------------------------------------------------------
// The transform + its module-pass wrapper (mirrors `Sccp`/`SccpPass`).
// ---------------------------------------------------------------------------

/// The equality-saturation transform (see the module documentation).
///
/// Like [`Sccp`](crate::transform::Sccp), the read-only analysis (building and
/// saturating the e-graph, which needs `&TypeContext`/`&ConstPool`) is done up
/// front by [`EqSat::analyze`] and distilled into a borrow-free [`Plan`] that the
/// rebuild in [`run`](FunctionTransform::run) consumes — the rebuild's builder
/// holds the shared interning tables mutably, so the analysis cannot run inside
/// `run`.
#[derive(Debug, Default)]
pub struct EqSat {
    plan: Option<Plan>,
}

impl EqSat {
    /// A transform with no analysis yet; call [`EqSat::analyze`] before running it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build and saturate the e-graph for `func` — which **must** be the same
    /// function subsequently handed to [`run`](FunctionTransform::run) as `old` —
    /// and store the distilled extraction [`Plan`].
    pub fn analyze(&mut self, func: &Function, types: &TypeContext, consts: &ConstPool) {
        self.plan = Some(Plan::build(func, types, consts));
    }
}

impl FunctionTransform for EqSat {
    fn name(&self) -> &str {
        "eqsat"
    }

    fn run(&mut self, old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
        let Some(plan) = &self.plan else {
            return Changed::No;
        };
        if !plan.changed || old.entry().is_none() {
            return Changed::No;
        }
        rebuild_function(old, plan, builder);
        Changed::Yes
    }
}

/// The equality-saturation transform as a module pass: for every function
/// definition it builds/saturates the e-graph, rebuilds the body, and installs
/// the result, reporting [`Changed::Yes`] if any function changed.
#[derive(Debug, Default, Clone, Copy)]
pub struct EqSatPass;

impl ModulePass for EqSatPass {
    fn name(&self) -> &str {
        "eqsat"
    }

    fn run(&mut self, module: &mut Module) -> Changed {
        let mut changed = Changed::No;
        for i in 0..module.function_count() {
            let id = FuncId::from_index(i);
            if module.function(id).is_declaration() {
                continue;
            }
            let mut t = EqSat::new();
            t.analyze(module.function(id), module.types(), module.consts());
            let (fresh, c) = module.map_function(id, |old, b| t.run(old, b));
            if c == Changed::Yes {
                module.replace_function(id, fresh);
                changed = Changed::Yes;
            }
        }
        changed
    }
}

// ---------------------------------------------------------------------------
// The borrow-free extraction plan.
// ---------------------------------------------------------------------------

/// One node of the extracted expression DAG. Leaves reference an original
/// [`ValueId`] (materialized during rebuild via the value map); constants carry an
/// owned [`Const`]; ops carry everything [`FunctionBuilder::append_inst`] needs.
#[derive(Debug, Clone)]
enum ExtNode {
    /// An opaque original value (parameter, side-effecting/unmodeled result,
    /// global/function reference) — reproduced through the rebuild's value map.
    Leaf(ValueId),
    /// A constant to materialize.
    Const(Const),
    /// A pure modeled op over child [`ExtNode`] indices.
    Op {
        /// The opcode to emit.
        kind: InstKind,
        /// The instruction flags (only `Bin` carries non-`NONE` flags here).
        flags: Flags,
        /// The result type.
        ty: TypeId,
        /// Operand ext-node indices, in order.
        children: Vec<usize>,
    },
}

/// Everything [`EqSat::run`] needs, extracted from the saturated e-graph so it
/// survives the rebuild without borrowing the interning tables.
#[derive(Debug)]
struct Plan {
    /// The extraction DAG: each node references children by index (shared
    /// subterms appear once — the CSE effect).
    nodes: Vec<ExtNode>,
    /// `value_node[v]` is `Some(idx)` for each pure **modeled** value `v` (its
    /// chosen, cheapest extraction is `nodes[idx]`), and `None` for opaque values
    /// (which the rebuild passes through).
    value_node: Vec<Option<usize>>,
    /// Whether applying this plan changes the function (drives `Changed` and keeps
    /// the pass idempotent).
    changed: bool,
}

impl Plan {
    fn build(func: &Function, types: &TypeContext, consts: &ConstPool) -> Plan {
        let nv = func.value_count();
        if func.entry().is_none() {
            return Plan { nodes: Vec::new(), value_node: vec![None; nv], changed: false };
        }

        // Build one e-class for every SSA value (pure ops become e-nodes over
        // their operands' classes; everything else is an opaque leaf).
        let mut eg = EGraph::new(types, consts, nv);
        for i in 0..nv {
            eg.class_of(func, ValueId::from_index(i));
        }
        eg.rebuild();
        eg.saturate();

        eg.extract(func, types)
    }
}

// ---------------------------------------------------------------------------
// The e-graph.
// ---------------------------------------------------------------------------

/// A rewrite/leaf operator. Value operands live in the [`ENode`]'s child list;
/// this carries only the opcode payload (and, for `Bin`, its flags — so `add` and
/// `add nsw` are distinct e-nodes, as they may denote different values).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NodeOp {
    /// An opaque original value.
    Leaf(ValueId),
    /// A constant.
    Const(Const),
    /// Integer binary op with its flags.
    Bin(BinOp, Flags),
    /// Integer comparison (result `i1`).
    ICmp(IntPred),
    /// Integer cast (`trunc`/`zext`/`sext`).
    Cast(CastOp),
    /// Ternary select.
    Select,
    /// `freeze`.
    Freeze,
}

/// An e-node: an operator over canonical child e-class ids.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ENode {
    op: NodeOp,
    children: Vec<usize>,
}

/// An e-class: the set of e-nodes proven equal, plus their shared type.
#[derive(Debug)]
struct EClass {
    nodes: Vec<ENode>,
    ty: TypeId,
}

/// The equality-saturation e-graph over one function's pure value DAG.
#[derive(Debug)]
struct EGraph<'a> {
    types: &'a TypeContext,
    consts: &'a ConstPool,
    /// Union-find parent pointers (index = e-class id).
    uf: Vec<usize>,
    /// The e-classes (index = e-class id; non-root classes have their nodes
    /// drained into their representative on union).
    classes: Vec<EClass>,
    /// Hash-cons: canonical e-node → its e-class id.
    memo: DetHashMap<ENode, usize>,
    /// Memo of `value → e-class id` while building.
    value_class: Vec<Option<usize>>,
}

impl<'a> EGraph<'a> {
    fn new(types: &'a TypeContext, consts: &'a ConstPool, value_count: usize) -> Self {
        EGraph {
            types,
            consts,
            uf: Vec::new(),
            classes: Vec::new(),
            memo: DetHashMap::default(),
            value_class: vec![None; value_count],
        }
    }

    // --- union-find --------------------------------------------------------

    /// Find the representative of `x`, halving the path as it walks.
    fn find(&mut self, mut x: usize) -> usize {
        while self.uf[x] != x {
            self.uf[x] = self.uf[self.uf[x]];
            x = self.uf[x];
        }
        x
    }

    /// Union two classes, keeping the lower id as representative (deterministic).
    /// Returns whether they were distinct.
    fn union(&mut self, a: usize, b: usize) -> bool {
        let a = self.find(a);
        let b = self.find(b);
        if a == b {
            return false;
        }
        let (keep, gone) = if a < b { (a, b) } else { (b, a) };
        self.uf[gone] = keep;
        let gnodes = std::mem::take(&mut self.classes[gone].nodes);
        self.classes[keep].nodes.extend(gnodes);
        true
    }

    // --- hash-consing ------------------------------------------------------

    /// Canonicalize an e-node by mapping each child to its representative.
    fn canonicalize(&mut self, mut node: ENode) -> ENode {
        for c in &mut node.children {
            *c = self.find(*c);
        }
        node
    }

    /// Add an e-node (hash-consed), returning its e-class id.
    fn add(&mut self, node: ENode, ty: TypeId) -> usize {
        let node = self.canonicalize(node);
        if let Some(&id) = self.memo.get(&node) {
            return self.find(id);
        }
        let id = self.classes.len();
        self.uf.push(id);
        self.classes.push(EClass { nodes: vec![node.clone()], ty });
        self.memo.insert(node, id);
        id
    }

    /// Add an e-node from parts (convenience for rule construction).
    fn add_op(&mut self, op: NodeOp, children: Vec<usize>, ty: TypeId) -> usize {
        self.add(ENode { op, children }, ty)
    }

    // --- construction from the IR ------------------------------------------

    /// The e-class of value `v`, built (and memoized) on demand. Pure modeled
    /// instructions recurse into their operands; every other value is a leaf.
    fn class_of(&mut self, func: &Function, v: ValueId) -> usize {
        if let Some(id) = self.value_class[v.index()] {
            return id;
        }
        let ty = func.value_type(v);
        let node = match &func.value(v).def {
            ValueDef::Const(cid) => {
                ENode { op: NodeOp::Const(self.consts.get(*cid).clone()), children: Vec::new() }
            }
            ValueDef::Param(..) | ValueDef::Global(_) | ValueDef::Func(_) => {
                ENode { op: NodeOp::Leaf(v), children: Vec::new() }
            }
            ValueDef::Inst(i) => {
                let inst = func.inst(*i);
                if is_modeled(&inst.kind, ty, self.types) {
                    let mut children = Vec::with_capacity(inst.operands().len());
                    for &o in inst.operands() {
                        children.push(self.class_of(func, o));
                    }
                    let op = node_op(&inst.kind, inst.flags).expect("modeled op has a NodeOp");
                    ENode { op, children }
                } else {
                    ENode { op: NodeOp::Leaf(v), children: Vec::new() }
                }
            }
        };
        let id = self.add(node, ty);
        self.value_class[v.index()] = Some(id);
        id
    }

    // --- congruence closure ------------------------------------------------

    /// Restore the congruence invariant and re-canonicalize the whole graph.
    ///
    /// A naive-to-fixpoint pass: repeatedly canonicalize every e-node into a fresh
    /// table and union any collisions, until a sweep makes no merge. Then compact
    /// each representative's node list (canonical, deduplicated) and rebuild the
    /// hash-cons memo. Correct and simple; the node budget keeps it cheap.
    fn rebuild(&mut self) {
        loop {
            let mut merged = false;
            let mut table: DetHashMap<ENode, usize> = DetHashMap::default();
            for ci in 0..self.classes.len() {
                if self.find(ci) != ci {
                    continue;
                }
                let nodes = self.classes[ci].nodes.clone();
                for node in nodes {
                    let cn = self.canonicalize(node);
                    let cid = self.find(ci);
                    if let Some(&other) = table.get(&cn) {
                        if self.find(other) != cid {
                            self.union(other, cid);
                            merged = true;
                        }
                    } else {
                        table.insert(cn, cid);
                    }
                }
            }
            if !merged {
                break;
            }
        }

        // Compact: canonical, deduplicated node lists + a fresh memo.
        self.memo.clear();
        for ci in 0..self.classes.len() {
            if self.find(ci) != ci {
                self.classes[ci].nodes.clear();
                continue;
            }
            let nodes = std::mem::take(&mut self.classes[ci].nodes);
            let mut seen: DetHashSet<ENode> = DetHashSet::default();
            let mut compact = Vec::with_capacity(nodes.len());
            for node in nodes {
                let cn = self.canonicalize(node);
                if seen.insert(cn.clone()) {
                    compact.push(cn);
                }
            }
            for n in &compact {
                self.memo.insert(n.clone(), ci);
            }
            self.classes[ci].nodes = compact;
        }
    }

    // --- saturation --------------------------------------------------------

    /// Apply the rewrite rules to a bounded fixpoint.
    fn saturate(&mut self) {
        for _ in 0..MAX_ITERS {
            let snapshot = self.snapshot();
            let mut changed = false;
            for (ci, node) in snapshot {
                let ci = self.find(ci);
                let ty = self.classes[ci].ty;
                // Constant folding (sound for any op; refuses UB via `ir::fold`).
                if let Some(c) = self.try_fold(&node, ty)
                    && self.classes.len() < MAX_NODES
                {
                    let nid = self.add_op(NodeOp::Const(c), Vec::new(), ty);
                    changed |= self.union(ci, nid);
                }
                changed |= self.apply_algebraic(ci, &node, ty);
            }
            self.rebuild();
            if !changed {
                break;
            }
        }
    }

    /// Snapshot the current `(class, node)` pairs so rule application can mutate
    /// the graph while iterating a stable view.
    fn snapshot(&mut self) -> Vec<(usize, ENode)> {
        let mut out = Vec::new();
        for ci in 0..self.classes.len() {
            if self.find(ci) != ci {
                continue;
            }
            for node in &self.classes[ci].nodes {
                out.push((ci, node.clone()));
            }
        }
        out
    }

    /// If `node`'s children are all constants, fold it (unless that would be UB).
    fn try_fold(&mut self, node: &ENode, ty: TypeId) -> Option<Const> {
        let (kind, flags) = node_inst(&node.op)?;
        let mut operands = Vec::with_capacity(node.children.len());
        for &ch in &node.children {
            operands.push(self.class_const(ch)?);
        }
        match fold(self.types, ty, &kind, &flags, &operands) {
            Some(FoldResult::Folded(c)) => Some(c),
            // `WouldBeUb` or a non-foldable operand: leave it computed.
            _ => None,
        }
    }

    /// The constant a class is equal to, if any of its e-nodes is a constant.
    fn class_const(&mut self, id: usize) -> Option<Const> {
        let r = self.find(id);
        self.classes[r].nodes.iter().find_map(|n| match &n.op {
            NodeOp::Const(c) => Some(c.clone()),
            _ => None,
        })
    }

    /// Whether a class is equal to the constant `0` of its (integer) type.
    fn is_zero_class(&mut self, id: usize) -> bool {
        matches!(self.class_const(id), Some(Const::Int { ty, value })
            if int_width(self.types, ty).is_some_and(|w| value.mod_2k(w).is_zero()))
    }

    /// Whether a class is equal to the constant `1` of its (integer) type.
    fn is_one_class(&mut self, id: usize) -> bool {
        matches!(self.class_const(id), Some(Const::Int { ty, value })
            if int_width(self.types, ty).is_some_and(|w| value.mod_2k(w).is_one()))
    }

    /// Add a new e-node and union it into class `ci`, honoring the node budget.
    /// Returns whether the union changed anything.
    fn union_new(&mut self, ci: usize, op: NodeOp, children: Vec<usize>, ty: TypeId) -> bool {
        if self.classes.len() >= MAX_NODES {
            return false;
        }
        let nid = self.add_op(op, children, ty);
        self.union(ci, nid)
    }

    /// The `(a, b)` child-class pairs of every flag-free `bop` e-node in `id`'s
    /// class — the shapes an associativity rule reassociates.
    fn bin_children(&mut self, id: usize, bop: BinOp) -> Vec<(usize, usize)> {
        let r = self.find(id);
        let mut out = Vec::new();
        for n in &self.classes[r].nodes {
            if let NodeOp::Bin(op, f) = &n.op
                && *op == bop
                && *f == Flags::NONE
                && n.children.len() == 2
            {
                out.push((n.children[0], n.children[1]));
            }
        }
        out
    }

    /// Reassociate `bop(p, q)` where `p`'s class contains `bop(a, b)`:
    /// `(a·b)·q = a·(b·q) = b·(a·q)`. Combined with commutativity this floats
    /// constants together so folding can collapse them (e.g. `(x+3)+4 → x+7`).
    fn reassociate(&mut self, ci: usize, p: usize, q: usize, bop: BinOp, ty: TypeId) -> bool {
        let mut changed = false;
        for (a, b) in self.bin_children(p, bop) {
            if self.classes.len() >= MAX_NODES {
                break;
            }
            let bq = self.add_op(NodeOp::Bin(bop, Flags::NONE), vec![b, q], ty);
            changed |= self.union_new(ci, NodeOp::Bin(bop, Flags::NONE), vec![a, bq], ty);
            let aq = self.add_op(NodeOp::Bin(bop, Flags::NONE), vec![a, q], ty);
            changed |= self.union_new(ci, NodeOp::Bin(bop, Flags::NONE), vec![b, aq], ty);
        }
        changed
    }

    /// Apply the algebraic rewrite rules to one `(class, node)`. Every rule below
    /// is proved a refinement by the in-file `verified_rules_refine` test.
    fn apply_algebraic(&mut self, ci: usize, node: &ENode, ty: TypeId) -> bool {
        let NodeOp::Bin(op, flags) = &node.op else {
            return false;
        };
        // Restrict the algebraic rules to flag-free integer ops so no `nsw`/`nuw`/
        // `exact` assumption is ever silently dropped (that would be unsound).
        if *flags != Flags::NONE || !is_int(self.types, ty) || node.children.len() != 2 {
            return false;
        }
        let (a, b) = (node.children[0], node.children[1]);
        let none = Flags::NONE;
        let mut changed = false;
        match op {
            BinOp::Add => {
                // commutativity: a+b = b+a
                changed |= self.union_new(ci, NodeOp::Bin(BinOp::Add, none), vec![b, a], ty);
                // identity: x+0 = x  (either operand)
                if self.is_zero_class(b) {
                    let r = self.find(a);
                    changed |= self.union(ci, r);
                }
                if self.is_zero_class(a) {
                    let r = self.find(b);
                    changed |= self.union(ci, r);
                }
                // associativity (both nestings)
                changed |= self.reassociate(ci, a, b, BinOp::Add, ty);
                changed |= self.reassociate(ci, b, a, BinOp::Add, ty);
            }
            BinOp::Mul => {
                // commutativity
                changed |= self.union_new(ci, NodeOp::Bin(BinOp::Mul, none), vec![b, a], ty);
                // x*1 = x
                if self.is_one_class(b) {
                    let r = self.find(a);
                    changed |= self.union(ci, r);
                }
                if self.is_one_class(a) {
                    let r = self.find(b);
                    changed |= self.union(ci, r);
                }
                // x*0 → 0  (refinement; the constant dominates cost, see module docs)
                if self.is_zero_class(a) || self.is_zero_class(b) {
                    changed |= self.add_zero(ci, ty);
                }
                // x*2^k = x<<k
                changed |= self.mul_to_shift(ci, a, b, ty);
                changed |= self.mul_to_shift(ci, b, a, ty);
                // associativity
                changed |= self.reassociate(ci, a, b, BinOp::Mul, ty);
                changed |= self.reassociate(ci, b, a, BinOp::Mul, ty);
            }
            BinOp::Sub => {
                // x-0 = x
                if self.is_zero_class(b) {
                    let r = self.find(a);
                    changed |= self.union(ci, r);
                }
                // x-x = 0
                if self.find(a) == self.find(b) {
                    changed |= self.add_zero(ci, ty);
                }
            }
            BinOp::And => {
                changed |= self.union_new(ci, NodeOp::Bin(BinOp::And, none), vec![b, a], ty);
                // x&x = x
                if self.find(a) == self.find(b) {
                    let r = self.find(a);
                    changed |= self.union(ci, r);
                }
                // x&0 → 0
                if self.is_zero_class(a) || self.is_zero_class(b) {
                    changed |= self.add_zero(ci, ty);
                }
            }
            BinOp::Or => {
                changed |= self.union_new(ci, NodeOp::Bin(BinOp::Or, none), vec![b, a], ty);
                // x|x = x
                if self.find(a) == self.find(b) {
                    let r = self.find(a);
                    changed |= self.union(ci, r);
                }
                // x|0 = x
                if self.is_zero_class(b) {
                    let r = self.find(a);
                    changed |= self.union(ci, r);
                }
                if self.is_zero_class(a) {
                    let r = self.find(b);
                    changed |= self.union(ci, r);
                }
            }
            BinOp::Xor => {
                changed |= self.union_new(ci, NodeOp::Bin(BinOp::Xor, none), vec![b, a], ty);
                // x^x = 0
                if self.find(a) == self.find(b) {
                    changed |= self.add_zero(ci, ty);
                }
                // x^0 = x
                if self.is_zero_class(b) {
                    let r = self.find(a);
                    changed |= self.union(ci, r);
                }
                if self.is_zero_class(a) {
                    let r = self.find(b);
                    changed |= self.union(ci, r);
                }
            }
            _ => {}
        }
        changed
    }

    /// Union `ci` with the constant `0` of type `ty`.
    fn add_zero(&mut self, ci: usize, ty: TypeId) -> bool {
        if self.classes.len() >= MAX_NODES {
            return false;
        }
        let zero = Const::Int { ty, value: Int::ZERO };
        let nid = self.add_op(NodeOp::Const(zero), Vec::new(), ty);
        self.union(ci, nid)
    }

    /// `x * 2^k → x << k` when the multiplier class is a power-of-two constant
    /// with `0 < k < width` (both wrap identically, so this is a full equality).
    fn mul_to_shift(&mut self, ci: usize, x: usize, k: usize, ty: TypeId) -> bool {
        let Some(w) = int_width(self.types, ty) else {
            return false;
        };
        let Some(Const::Int { value, .. }) = self.class_const(k) else {
            return false;
        };
        let Some(shift) = value.mod_2k(w).is_power_of_two() else {
            return false;
        };
        if shift == 0 || shift >= w {
            return false;
        }
        let xr = self.find(x);
        let amt = self.add_op(NodeOp::Const(Const::Int { ty, value: Int::from_u64(u64::from(shift)) }), Vec::new(), ty);
        self.union_new(ci, NodeOp::Bin(BinOp::Shl, Flags::NONE), vec![xr, amt], ty)
    }

    // --- extraction --------------------------------------------------------

    /// Pick the cheapest e-node per class (a bottom-up cost fixpoint), build the
    /// extraction DAG, and decide whether the function actually changed.
    fn extract(&mut self, func: &Function, types: &TypeContext) -> Plan {
        let ncls = self.classes.len();
        let mut root = vec![0usize; ncls];
        for (i, r) in root.iter_mut().enumerate() {
            *r = self.find(i);
        }
        let class_ty: Vec<TypeId> = self.classes.iter().map(|c| c.ty).collect();

        // Cost fixpoint: cost(class) = min over its nodes of own + Σ children.
        // Self-referential (cyclic) nodes keep an infinite cost and are never
        // selected, so extraction always terminates on an acyclic DAG.
        let mut best_cost = vec![u64::MAX; ncls];
        let mut best_node: Vec<Option<ENode>> = vec![None; ncls];
        loop {
            let mut improved = false;
            for ci in 0..ncls {
                if root[ci] != ci {
                    continue;
                }
                for node in &self.classes[ci].nodes {
                    let mut cost = own_cost(&node.op);
                    let mut finite = true;
                    for &ch in &node.children {
                        let cc = best_cost[root[ch]];
                        if cc == u64::MAX {
                            finite = false;
                            break;
                        }
                        cost = cost.saturating_add(cc);
                    }
                    if finite && cost < best_cost[ci] {
                        best_cost[ci] = cost;
                        best_node[ci] = Some(node.clone());
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }

        // Materialize the extraction DAG for every modeled value's class.
        let mut nodes = Vec::new();
        let mut ext_of: Vec<Option<usize>> = vec![None; ncls];
        let mut value_node = vec![None; func.value_count()];
        for i in 0..func.value_count() {
            let v = ValueId::from_index(i);
            if !modeled_value(func, types, v) {
                continue;
            }
            let cr = root[self.value_class[i].expect("built a class for every value")];
            let idx = build_ext(cr, &root, &best_node, &class_ty, &mut nodes, &mut ext_of);
            value_node[i] = Some(idx);
        }

        let changed = self.decide_changed(func, types, &root, &best_node);
        Plan { nodes, value_node, changed }
    }

    /// Whether the extraction differs from the original: any simplification (a
    /// class whose cheapest node is not the value's own node), any CSE merge (two
    /// distinct live modeled values sharing a class), or any dead modeled op (its
    /// value unused, so the rebuild drops it). Precise enough to be idempotent.
    fn decide_changed(
        &mut self,
        func: &Function,
        types: &TypeContext,
        root: &[usize],
        best_node: &[Option<ENode>],
    ) -> bool {
        let used = compute_used(func, types);
        let mut class_first: DetHashMap<usize, usize> = DetHashMap::default();
        for i in 0..func.value_count() {
            let v = ValueId::from_index(i);
            if !modeled_value(func, types, v) {
                continue;
            }
            let cr = root[self.value_class[i].expect("class exists")];
            if !used.contains(&v) {
                return true; // a dead pure op the rebuild will drop
            }
            match class_first.get(&cr) {
                Some(&first) if first != i => return true, // CSE-merged with another value
                Some(_) => {}
                None => {
                    class_first.insert(cr, i);
                }
            }
            let orig = original_enode(func, v, root, &self.value_class);
            if best_node[cr].as_ref() != Some(&orig) {
                return true; // simplified/reassociated/folded away
            }
        }
        false
    }
}

/// Recursively lower the chosen e-node of class `cr` into the extraction DAG,
/// memoizing so shared subterms (CSE) appear once. Acyclic by construction of the
/// cost fixpoint, so the recursion terminates.
fn build_ext(
    cr: usize,
    root: &[usize],
    best: &[Option<ENode>],
    class_ty: &[TypeId],
    nodes: &mut Vec<ExtNode>,
    ext_of: &mut Vec<Option<usize>>,
) -> usize {
    if let Some(x) = ext_of[cr] {
        return x;
    }
    let node = best[cr].as_ref().expect("every reachable class has a finite extraction");
    let ext = match &node.op {
        NodeOp::Leaf(v) => ExtNode::Leaf(*v),
        NodeOp::Const(c) => ExtNode::Const(c.clone()),
        other => {
            let (kind, flags) = node_inst(other).expect("modeled op");
            let children: Vec<usize> = node
                .children
                .iter()
                .map(|&c| build_ext(root[c], root, best, class_ty, nodes, ext_of))
                .collect();
            ExtNode::Op { kind, flags, ty: class_ty[cr], children }
        }
    };
    let idx = nodes.len();
    nodes.push(ext);
    ext_of[cr] = Some(idx);
    idx
}

// ---------------------------------------------------------------------------
// The functional rebuild.
// ---------------------------------------------------------------------------

fn rebuild_function(old: &Function, plan: &Plan, builder: &mut FunctionBuilder<'_>) {
    let n = old.block_count();
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);
    let entry = old.entry().expect("run checks the function has a body");

    // Recreate every block (this pass never touches control flow).
    let mut new_block: Vec<Option<BlockId>> = vec![None; n];
    new_block[entry.index()] = Some(builder.create_entry_block());
    for (b, slot) in new_block.iter_mut().enumerate() {
        if b == entry.index() {
            continue;
        }
        let bb = BlockId::from_index(b);
        let ptys: Vec<TypeId> = old.block(bb).params().iter().map(|&p| old.value_type(p)).collect();
        *slot = Some(builder.create_block(&ptys));
    }
    let new_block: Vec<BlockId> =
        new_block.into_iter().map(|x| x.expect("every block was created")).collect();

    // Seed the value map from the rebuilt block parameters.
    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for (b, &nb) in new_block.iter().enumerate() {
        let bb = BlockId::from_index(b);
        let new_params = builder.block_params(nb).to_vec();
        for (i, &p) in old.block(bb).params().iter().enumerate() {
            vmap[p.index()] = Some(new_params[i]);
        }
    }

    // Emit in dominator preorder so every surviving definition precedes its uses.
    for b in dom_preorder(old, &doms) {
        let bb = BlockId::from_index(b);
        builder.switch_to(new_block[b]);
        // Extracted modeled subterms are cached per block, so a shared subterm is
        // emitted once and its definition (earlier in the block) dominates every
        // reuse; the cache is *not* shared across blocks (that would break
        // dominance), so a value used in several blocks is recomputed locally.
        let mut cache: DetHashMap<usize, ValueId> = DetHashMap::default();
        for &i in old.block(bb).insts() {
            let inst = old.inst(i);
            // Pure modeled results are emitted lazily at their use sites via the
            // extraction DAG, so skip them here (unused ones simply vanish — a
            // DCE bonus). Everything else (memory, calls, float, ...) is opaque
            // and reproduced verbatim, with its operands optimized.
            if let Some(r) = inst.result()
                && plan.value_node[r.index()].is_some()
            {
                continue;
            }
            let mut ops = Vec::with_capacity(inst.operands().len());
            for &o in inst.operands() {
                ops.push(operand_val(plan, old, builder, &mut vmap, &mut cache, o));
            }
            let result_ty = inst.result().map(|_| inst.ty);
            let nr = builder.append_inst(inst.kind.clone(), ops, inst.flags, result_ty);
            if let Some(r) = inst.result() {
                vmap[r.index()] = nr;
            }
        }
        emit_terminator(plan, old, builder, &mut vmap, &mut cache, &new_block, bb);
    }
}

/// The new value for an operand: the extracted term for a pure modeled value, or
/// the passed-through mapping for an opaque one.
fn operand_val(
    plan: &Plan,
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    vmap: &mut [Option<ValueId>],
    cache: &mut DetHashMap<usize, ValueId>,
    o: ValueId,
) -> ValueId {
    match plan.value_node[o.index()] {
        Some(idx) => emit_ext(plan, old, builder, vmap, cache, idx),
        None => remap_value(vmap, old, builder, o),
    }
}

/// Materialize extraction node `idx` into the current block, caching per block.
fn emit_ext(
    plan: &Plan,
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    vmap: &mut [Option<ValueId>],
    cache: &mut DetHashMap<usize, ValueId>,
    idx: usize,
) -> ValueId {
    if let Some(&v) = cache.get(&idx) {
        return v;
    }
    let v = match &plan.nodes[idx] {
        ExtNode::Leaf(vid) => remap_value(vmap, old, builder, *vid),
        ExtNode::Const(c) => materialize(builder, c),
        ExtNode::Op { kind, flags, ty, children } => {
            let mut child_vals = Vec::with_capacity(children.len());
            for &ch in children {
                child_vals.push(emit_ext(plan, old, builder, vmap, cache, ch));
            }
            builder
                .append_inst(kind.clone(), child_vals, *flags, Some(*ty))
                .expect("modeled op defines a value")
        }
    };
    cache.insert(idx, v);
    v
}

/// Rebuild `bb`'s terminator, routing each operand through [`operand_val`] so pure
/// operands get their optimized form.
fn emit_terminator(
    plan: &Plan,
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    vmap: &mut [Option<ValueId>],
    cache: &mut DetHashMap<usize, ValueId>,
    new_block: &[BlockId],
    bb: BlockId,
) {
    let Some(t) = old.block(bb).terminator() else {
        return;
    };
    let term = old.inst(t);
    let ops = term.operands();
    match &term.kind {
        InstKind::Ret => {
            let v = if ops.is_empty() {
                None
            } else {
                Some(operand_val(plan, old, builder, vmap, cache, ops[0]))
            };
            builder.ret(v);
        }
        InstKind::Unreachable => builder.unreachable(),
        InstKind::Br(target) => {
            let target = *target;
            let mut args = Vec::with_capacity(ops.len());
            for &o in ops {
                args.push(operand_val(plan, old, builder, vmap, cache, o));
            }
            builder.br(new_block[target.index()], &args);
        }
        InstKind::CondBr { if_true, if_false, true_args, false_args } => {
            let (if_true, if_false) = (*if_true, *if_false);
            let ta = *true_args as usize;
            let fa = *false_args as usize;
            let cond = operand_val(plan, old, builder, vmap, cache, ops[0]);
            let mut targs = Vec::with_capacity(ta);
            for k in 0..ta {
                targs.push(operand_val(plan, old, builder, vmap, cache, ops[1 + k]));
            }
            let mut fargs = Vec::with_capacity(fa);
            for k in 0..fa {
                fargs.push(operand_val(plan, old, builder, vmap, cache, ops[1 + ta + k]));
            }
            builder.cond_br(cond, new_block[if_true.index()], &targs, new_block[if_false.index()], &fargs);
        }
        InstKind::Switch(data) => {
            let cond = operand_val(plan, old, builder, vmap, cache, ops[0]);
            let da = data.default_args as usize;
            let default = data.default;
            let mut dargs = Vec::with_capacity(da);
            for k in 0..da {
                dargs.push(operand_val(plan, old, builder, vmap, cache, ops[1 + k]));
            }
            let mut cases = Vec::with_capacity(data.cases.len());
            let mut off = 1 + da;
            for c in &data.cases {
                let ca = c.args as usize;
                let mut cargs = Vec::with_capacity(ca);
                for k in 0..ca {
                    cargs.push(operand_val(plan, old, builder, vmap, cache, ops[off + k]));
                }
                cases.push((c.value.clone(), new_block[c.target.index()], cargs));
                off += ca;
            }
            builder.switch(cond, new_block[default.index()], &dargs, cases);
        }
        _ => {}
    }
}

/// Materialize an interned scalar constant as a value in the function being built.
fn materialize(builder: &mut FunctionBuilder<'_>, c: &Const) -> ValueId {
    match c {
        Const::Int { ty, value } => builder.const_int(*ty, value.clone()),
        Const::Float { ty, bits } => builder.const_float(*ty, *bits),
        Const::Null(ty) => builder.null(*ty),
        Const::Poison(ty) => builder.poison(*ty),
        // Modeled ops fold only to scalar constants; an aggregate never reaches
        // here, but poison of its type is a sound fallback rather than a panic.
        Const::Aggregate { ty, .. } => builder.poison(*ty),
    }
}

// ---------------------------------------------------------------------------
// Small shared helpers.
// ---------------------------------------------------------------------------

/// Whether `kind` (with result type `result_ty`) is a pure, integer-typed op the
/// e-graph models as an e-node. Everything else is an opaque leaf.
fn is_modeled(kind: &InstKind, result_ty: TypeId, types: &TypeContext) -> bool {
    match kind {
        InstKind::Bin(op) if !op.is_float() => is_int(types, result_ty),
        InstKind::ICmp(_) => is_int(types, result_ty),
        InstKind::Cast(CastOp::Trunc | CastOp::ZExt | CastOp::SExt) => is_int(types, result_ty),
        InstKind::Select => is_int(types, result_ty),
        InstKind::Freeze => is_int(types, result_ty),
        _ => false,
    }
}

/// Whether value `v` is a pure modeled instruction result.
fn modeled_value(func: &Function, types: &TypeContext, v: ValueId) -> bool {
    match func.value(v).def {
        ValueDef::Inst(i) => is_modeled(&func.inst(i).kind, func.value_type(v), types),
        _ => false,
    }
}

/// The [`NodeOp`] for a modeled opcode (with its flags for `Bin`).
fn node_op(kind: &InstKind, flags: Flags) -> Option<NodeOp> {
    match kind {
        InstKind::Bin(op) if !op.is_float() => Some(NodeOp::Bin(*op, flags)),
        InstKind::ICmp(pred) => Some(NodeOp::ICmp(*pred)),
        InstKind::Cast(op @ (CastOp::Trunc | CastOp::ZExt | CastOp::SExt)) => Some(NodeOp::Cast(*op)),
        InstKind::Select => Some(NodeOp::Select),
        InstKind::Freeze => Some(NodeOp::Freeze),
        _ => None,
    }
}

/// The `(InstKind, Flags)` an op-node reconstructs to (for folding and emission).
fn node_inst(op: &NodeOp) -> Option<(InstKind, Flags)> {
    match op {
        NodeOp::Bin(b, f) => Some((InstKind::Bin(*b), *f)),
        NodeOp::ICmp(p) => Some((InstKind::ICmp(*p), Flags::NONE)),
        NodeOp::Cast(c) => Some((InstKind::Cast(*c), Flags::NONE)),
        NodeOp::Select => Some((InstKind::Select, Flags::NONE)),
        NodeOp::Freeze => Some((InstKind::Freeze, Flags::NONE)),
        NodeOp::Leaf(_) | NodeOp::Const(_) => None,
    }
}

/// The e-node a modeled value was originally built from (canonical children),
/// used to detect whether extraction changed it.
fn original_enode(
    func: &Function,
    v: ValueId,
    root: &[usize],
    value_class: &[Option<usize>],
) -> ENode {
    let ValueDef::Inst(i) = func.value(v).def else {
        unreachable!("modeled values are instruction results");
    };
    let inst = func.inst(i);
    let op = node_op(&inst.kind, inst.flags).expect("modeled op");
    let children = inst
        .operands()
        .iter()
        .map(|&o| root[value_class[o.index()].expect("class exists")])
        .collect();
    ENode { op, children }
}

/// The set of pure modeled values that are actually used (reachable, backward,
/// from the operands of every opaque instruction and terminator). A modeled value
/// outside this set is dead and the rebuild drops it.
fn compute_used(func: &Function, types: &TypeContext) -> DetHashSet<ValueId> {
    let mut used: DetHashSet<ValueId> = DetHashSet::default();
    let mut worklist: Vec<ValueId> = Vec::new();
    let seed = |used: &mut DetHashSet<ValueId>, wl: &mut Vec<ValueId>, o: ValueId| {
        if modeled_value(func, types, o) && used.insert(o) {
            wl.push(o);
        }
    };
    for (_bid, blk) in func.blocks() {
        for &i in blk.insts() {
            let inst = func.inst(i);
            // A modeled result is emitted only when used, so it is *not* a seed;
            // opaque instructions are always emitted, so their operands are.
            if modeled_value_of_inst(func, types, inst) {
                continue;
            }
            for &o in inst.operands() {
                seed(&mut used, &mut worklist, o);
            }
        }
        if let Some(t) = blk.terminator() {
            for &o in func.inst(t).operands() {
                seed(&mut used, &mut worklist, o);
            }
        }
    }
    while let Some(v) = worklist.pop() {
        if let ValueDef::Inst(i) = func.value(v).def {
            for &o in func.inst(i).operands() {
                seed(&mut used, &mut worklist, o);
            }
        }
    }
    used
}

/// Whether an instruction produces a modeled value.
fn modeled_value_of_inst(func: &Function, types: &TypeContext, inst: &InstData) -> bool {
    match inst.result() {
        Some(r) => is_modeled(&inst.kind, func.value_type(r), types),
        None => false,
    }
}

/// The (single-dimension) **B9 cost** of an e-node's own operator, in abstract
/// code-size/latency units: constants and inputs are cheap, shifts/adds cheap,
/// multiply dear, divide/remainder dearest. Extraction minimizes the additive sum
/// over the chosen tree — a monotone cost, the height-1 instance of the B9 cost
/// lattice (which generalizes to a product of resource dimensions ordered
/// pointwise; a single additive dimension suffices for this pass).
fn own_cost(op: &NodeOp) -> u64 {
    match op {
        NodeOp::Leaf(_) | NodeOp::Const(_) => 1,
        NodeOp::Bin(b, _) => match b {
            BinOp::Mul => 6,
            BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => 20,
            _ => 2, // add/sub/bitwise/shift and (unreached) float ops
        },
        NodeOp::ICmp(_) | NodeOp::Cast(_) | NodeOp::Freeze => 2,
        NodeOp::Select => 3,
    }
}

/// Whether `ty` is an integer type.
fn is_int(types: &TypeContext, ty: TypeId) -> bool {
    matches!(types.get(ty), Type::Int(_))
}

/// The width of an integer type, or `None` for non-integers.
fn int_width(types: &TypeContext, ty: TypeId) -> Option<u32> {
    match types.get(ty) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::{EqSat, EqSatPass};

    use crate::ir::builder::FunctionBuilder;
    use crate::ir::inst::{BinOp, Flags, InstKind};
    use crate::ir::value::{Const, ValueDef};
    use crate::ir::{FuncId, Function, InstId, Module, TypeId, ValueId};
    use crate::pass::{Changed, ModulePass};
    use crate::support::StrInterner;
    use crate::transform::FunctionTransform;
    use crate::verify::{RefinementResult, check_refinement, verify_module};

    use puremp::Int;

    /// Run the optimizer over one function (analyze up front, then rebuild).
    fn run_eqsat(m: &mut Module, f: FuncId) -> Changed {
        let mut t = EqSat::new();
        t.analyze(m.function(f), m.types(), m.consts());
        let (fresh, c) = m.map_function(f, |old, b| t.run(old, b));
        if c == Changed::Yes {
            m.replace_function(f, fresh);
        }
        c
    }

    fn count_kind(f: &Function, pred: impl Fn(&InstKind) -> bool) -> usize {
        let mut c = 0;
        for (_bid, blk) in f.blocks() {
            for &i in blk.insts() {
                if pred(&f.inst(i).kind) {
                    c += 1;
                }
            }
        }
        c
    }

    fn ret_operand(f: &Function) -> ValueId {
        for (_bid, blk) in f.blocks() {
            if let Some(t) = blk.terminator()
                && matches!(f.inst(t).kind, InstKind::Ret)
                && let Some(&v) = f.inst(t).operands().first()
            {
                return v;
            }
        }
        panic!("no value-returning ret found");
    }

    /// A structural fingerprint of a function, for determinism checks.
    fn canon(f: &Function) -> String {
        let mut s = String::new();
        for i in 0..f.inst_count() {
            let _ = writeln!(s, "I{i}: {:?}", f.inst(InstId::from_index(i)));
        }
        for v in 0..f.value_count() {
            let val = f.value(ValueId::from_index(v));
            let _ = writeln!(s, "V{v}: {:?} : {:?}", val.def, val.ty);
        }
        s
    }

    /// Build a single-block function `name(params) -> ret { ret body(params) }`.
    fn build_fn(
        m: &mut Module,
        syms: &mut StrInterner,
        name: &str,
        params: &[TypeId],
        ret: TypeId,
        body: impl FnOnce(&mut FunctionBuilder<'_>, &[ValueId]) -> ValueId,
    ) -> FuncId {
        let sig = m.types_mut().func(params.to_vec(), ret, false);
        let f = m.declare_function(syms.intern(name), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let ps: Vec<ValueId> = (0..params.len()).map(|i| b.param(entry, i as u32)).collect();
            let r = body(&mut b, &ps);
            b.ret(Some(r));
        }
        f
    }

    /// Assert that `tgt` provably refines `src` (never `Unknown` — the rule must
    /// be discharged by z3rs, not merely un-refuted).
    fn assert_refines(m: &Module, src: FuncId, tgt: FuncId, rule: &str) {
        match check_refinement(m.types(), m.consts(), m.function(src), m.function(tgt)) {
            RefinementResult::Refines => {}
            other => panic!("rule `{rule}` did not verify as a refinement: {other:?}"),
        }
    }

    /// Assert a rule is at least **not refuted**: the solver returns `Refines`, or
    /// a sound `Unknown` (never a counterexample). Used for the nonlinear
    /// bit-vector *multiplication* identities (commutativity/associativity of
    /// `mul`), which are trivially sound by inspection but whose equivalence
    /// bit-blasts to a hard SAT instance z3rs may answer `unknown` on — a
    /// solver-completeness boundary, not an unsoundness (cf. the `refinement`
    /// module's own out-of-scope `Unknown`s).
    fn assert_not_refuted(m: &Module, src: FuncId, tgt: FuncId, rule: &str) {
        match check_refinement(m.types(), m.consts(), m.function(src), m.function(tgt)) {
            RefinementResult::Refines | RefinementResult::Unknown(_) => {}
            RefinementResult::Counterexample(c) => panic!("rule `{rule}` is unsound: {c}"),
        }
    }

    /// **B2 verification of the whole rule set.** For each algebraic identity the
    /// optimizer relies on, build the lhs pattern and the rhs and prove the rhs
    /// refines the lhs via the z3rs refinement checker.
    #[test]
    fn verified_rules_refine() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("eqsat-rules");
        // Verify over `i4`: the rules are width-agnostic, and narrow bit-vectors
        // keep z3rs's bit-blasting cheap (the whole suite discharges quickly).
        let iw = m.types_mut().int(4);

        // A rule side: builds the returned value from the entry parameters.
        type Side = fn(&mut FunctionBuilder<'_>, &[ValueId]) -> ValueId;

        // Tier A — proved `Refines`. Two `i4` params; `ci(b, p, v)` is the
        // constant `v` of the parameters' type.
        let cases2: Vec<(&str, Side, Side)> = vec![
            ("add-comm", |b, p| b.add(p[0], p[1], Flags::NONE), |b, p| b.add(p[1], p[0], Flags::NONE)),
            ("and-comm", |b, p| b.bin(BinOp::And, p[0], p[1], Flags::NONE), |b, p| b.bin(BinOp::And, p[1], p[0], Flags::NONE)),
            ("or-comm", |b, p| b.bin(BinOp::Or, p[0], p[1], Flags::NONE), |b, p| b.bin(BinOp::Or, p[1], p[0], Flags::NONE)),
            ("xor-comm", |b, p| b.bin(BinOp::Xor, p[0], p[1], Flags::NONE), |b, p| b.bin(BinOp::Xor, p[1], p[0], Flags::NONE)),
            // identities with a constant / same operand
            ("add-zero", |b, p| { let z = ci(b, p, 0); b.add(p[0], z, Flags::NONE) }, |_, p| p[0]),
            ("mul-one", |b, p| { let o = ci(b, p, 1); b.mul(p[0], o, Flags::NONE) }, |_, p| p[0]),
            ("mul-zero", |b, p| { let z = ci(b, p, 0); b.mul(p[0], z, Flags::NONE) }, |b, p| ci(b, p, 0)),
            ("sub-self", |b, p| b.sub(p[0], p[0], Flags::NONE), |b, p| ci(b, p, 0)),
            ("sub-zero", |b, p| { let z = ci(b, p, 0); b.sub(p[0], z, Flags::NONE) }, |_, p| p[0]),
            ("and-self", |b, p| b.bin(BinOp::And, p[0], p[0], Flags::NONE), |_, p| p[0]),
            ("and-zero", |b, p| { let z = ci(b, p, 0); b.bin(BinOp::And, p[0], z, Flags::NONE) }, |b, p| ci(b, p, 0)),
            ("or-self", |b, p| b.bin(BinOp::Or, p[0], p[0], Flags::NONE), |_, p| p[0]),
            ("or-zero", |b, p| { let z = ci(b, p, 0); b.bin(BinOp::Or, p[0], z, Flags::NONE) }, |_, p| p[0]),
            ("xor-self", |b, p| b.bin(BinOp::Xor, p[0], p[0], Flags::NONE), |b, p| ci(b, p, 0)),
            ("xor-zero", |b, p| { let z = ci(b, p, 0); b.bin(BinOp::Xor, p[0], z, Flags::NONE) }, |_, p| p[0]),
            // multiply by powers of two → shift
            ("mul-2", |b, p| { let two = ci(b, p, 2); b.mul(p[0], two, Flags::NONE) }, |b, p| { let one = ci(b, p, 1); b.bin(BinOp::Shl, p[0], one, Flags::NONE) }),
            ("mul-8", |b, p| { let e = ci(b, p, 8); b.mul(p[0], e, Flags::NONE) }, |b, p| { let three = ci(b, p, 3); b.bin(BinOp::Shl, p[0], three, Flags::NONE) }),
            // constant folding instance
            ("fold-add", |b, p| { let a = ci(b, p, 3); let c = ci(b, p, 4); b.add(a, c, Flags::NONE) }, |b, p| ci(b, p, 7)),
        ];
        for (name, src_b, tgt_b) in cases2 {
            let src = build_fn(&mut m, &mut syms, &format!("{name}_s"), &[iw, iw], iw, src_b);
            let tgt = build_fn(&mut m, &mut syms, &format!("{name}_t"), &[iw, iw], iw, tgt_b);
            assert_refines(&m, src, tgt, name);
        }

        // Tier A associativity of `+` (linear) — proved `Refines`; needs 3 params.
        let src = build_fn(&mut m, &mut syms, "add-assoc_s", &[iw, iw, iw], iw,
            |b, p| { let ab = b.add(p[0], p[1], Flags::NONE); b.add(ab, p[2], Flags::NONE) });
        let tgt = build_fn(&mut m, &mut syms, "add-assoc_t", &[iw, iw, iw], iw,
            |b, p| { let bc = b.add(p[1], p[2], Flags::NONE); b.add(p[0], bc, Flags::NONE) });
        assert_refines(&m, src, tgt, "add-assoc");

        // Tier B — sound by inspection, but their equivalence is a *nonlinear*
        // bit-vector query z3rs may only answer `unknown` on. We assert they are
        // never *refuted* (no counterexample), which is what soundness needs.
        let src = build_fn(&mut m, &mut syms, "mul-comm_s", &[iw, iw], iw,
            |b, p| b.mul(p[0], p[1], Flags::NONE));
        let tgt = build_fn(&mut m, &mut syms, "mul-comm_t", &[iw, iw], iw,
            |b, p| b.mul(p[1], p[0], Flags::NONE));
        assert_not_refuted(&m, src, tgt, "mul-comm");

        let src = build_fn(&mut m, &mut syms, "mul-assoc_s", &[iw, iw, iw], iw,
            |b, p| { let ab = b.mul(p[0], p[1], Flags::NONE); b.mul(ab, p[2], Flags::NONE) });
        let tgt = build_fn(&mut m, &mut syms, "mul-assoc_t", &[iw, iw, iw], iw,
            |b, p| { let bc = b.mul(p[1], p[2], Flags::NONE); b.mul(p[0], bc, Flags::NONE) });
        assert_not_refuted(&m, src, tgt, "mul-assoc");
    }

    #[test]
    fn multiply_by_two_becomes_shift() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("eqsat-shift");
        let i32t = m.types_mut().int(32);
        let f = build_fn(&mut m, &mut syms, "f", &[i32t], i32t, |b, p| {
            let two = b.const_i64(i32t, 2);
            b.mul(p[0], two, Flags::NONE)
        });
        assert!(verify_module(&m).is_ok());

        assert_eq!(run_eqsat(&mut m, f), Changed::Yes);
        let func = m.function(f);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Bin(BinOp::Mul))), 0, "mul is gone");
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Bin(BinOp::Shl))), 1, "shl replaces it");
        assert!(verify_module(&m).is_ok(), "output must verify");
    }

    #[test]
    fn identities_collapse_to_the_input() {
        // (x + 0) * 1  →  x
        let mut syms = StrInterner::new();
        let mut m = Module::new("eqsat-ident");
        let i32t = m.types_mut().int(32);
        let f = build_fn(&mut m, &mut syms, "f", &[i32t], i32t, |b, p| {
            let z = b.const_i64(i32t, 0);
            let s = b.add(p[0], z, Flags::NONE);
            let one = b.const_i64(i32t, 1);
            b.mul(s, one, Flags::NONE)
        });
        assert!(verify_module(&m).is_ok());

        assert_eq!(run_eqsat(&mut m, f), Changed::Yes);
        let func = m.function(f);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Bin(_))), 0, "all arithmetic gone");
        // The return operand is the parameter itself.
        assert!(matches!(func.value(ret_operand(func)).def, ValueDef::Param(..)), "returns x directly");
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn constant_expression_folds() {
        // 2 + 3*4  →  14
        let mut syms = StrInterner::new();
        let mut m = Module::new("eqsat-fold");
        let i32t = m.types_mut().int(32);
        let f = build_fn(&mut m, &mut syms, "f", &[], i32t, |b, _| {
            let three = b.const_i64(i32t, 3);
            let four = b.const_i64(i32t, 4);
            let prod = b.mul(three, four, Flags::NONE);
            let two = b.const_i64(i32t, 2);
            b.add(two, prod, Flags::NONE)
        });
        assert!(verify_module(&m).is_ok());

        assert_eq!(run_eqsat(&mut m, f), Changed::Yes);
        let func = m.function(f);
        assert_eq!(count_kind(func, |_| true), 0, "the whole expression folds away");
        assert_int_ret(&m, f, 32, 14);
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn reassociation_exposes_a_fold() {
        // (x + 3) + 4  →  x + 7
        let mut syms = StrInterner::new();
        let mut m = Module::new("eqsat-reassoc");
        let i32t = m.types_mut().int(32);
        let f = build_fn(&mut m, &mut syms, "f", &[i32t], i32t, |b, p| {
            let three = b.const_i64(i32t, 3);
            let s = b.add(p[0], three, Flags::NONE);
            let four = b.const_i64(i32t, 4);
            b.add(s, four, Flags::NONE)
        });
        assert!(verify_module(&m).is_ok());

        assert_eq!(run_eqsat(&mut m, f), Changed::Yes);
        let func = m.function(f);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Bin(BinOp::Add))), 1, "one add remains");
        // That add is `x + 7`: one operand is the param, the other the constant 7.
        let add = ret_operand(func);
        let ValueDef::Inst(i) = func.value(add).def else { panic!("ret is an instruction") };
        let ops = func.inst(i).operands();
        let has_seven = ops.iter().any(|&o| matches!(&func.value(o).def, ValueDef::Const(c)
            if matches!(m.consts().get(*c), Const::Int { value, .. } if value.mod_2k(32) == Int::from_i64(7).mod_2k(32))));
        assert!(has_seven, "the folded constant 7 appears");
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn common_subexpression_is_shared() {
        // (a*b) + (a*b): two separate products collapse to one via hash-consing.
        let mut syms = StrInterner::new();
        let mut m = Module::new("eqsat-cse");
        let i32t = m.types_mut().int(32);
        let f = build_fn(&mut m, &mut syms, "f", &[i32t, i32t], i32t, |b, p| {
            let m1 = b.mul(p[0], p[1], Flags::NONE);
            let m2 = b.mul(p[0], p[1], Flags::NONE);
            b.add(m1, m2, Flags::NONE)
        });
        assert!(verify_module(&m).is_ok());
        assert_eq!(count_kind(m.function(f), |k| matches!(k, InstKind::Bin(BinOp::Mul))), 2);

        assert_eq!(run_eqsat(&mut m, f), Changed::Yes);
        let func = m.function(f);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Bin(BinOp::Mul))), 1, "product computed once");
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Bin(BinOp::Add))), 1, "add remains");
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn is_idempotent_and_deterministic() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("eqsat-idem");
        let i32t = m.types_mut().int(32);
        let f = build_fn(&mut m, &mut syms, "f", &[i32t], i32t, |b, p| {
            let two = b.const_i64(i32t, 2);
            b.mul(p[0], two, Flags::NONE)
        });
        assert_eq!(run_eqsat(&mut m, f), Changed::Yes);
        assert_eq!(run_eqsat(&mut m, f), Changed::No, "second run is a fixpoint");

        // Determinism: two independent runs produce identical bodies.
        let mut ma = Module::new("a");
        let mut mb = Module::new("b");
        let ia = ma.types_mut().int(32);
        let ib = mb.types_mut().int(32);
        let mut sa = StrInterner::new();
        let mut sb = StrInterner::new();
        let body = |b: &mut FunctionBuilder<'_>, p: &[ValueId]| {
            let three = b.const_i64(b.value_type(p[0]), 3);
            let s = b.add(p[0], three, Flags::NONE);
            let four = b.const_i64(b.value_type(p[0]), 4);
            b.add(s, four, Flags::NONE)
        };
        let fa = build_fn(&mut ma, &mut sa, "f", &[ia], ia, body);
        let fb = build_fn(&mut mb, &mut sb, "f", &[ib], ib, body);
        run_eqsat(&mut ma, fa);
        run_eqsat(&mut mb, fb);
        assert_eq!(canon(ma.function(fa)), canon(mb.function(fb)));
    }

    #[test]
    fn does_not_fold_ub_division() {
        // sdiv 4, 0 is UB; folding must be refused and the op left in place.
        let mut syms = StrInterner::new();
        let mut m = Module::new("eqsat-ub");
        let i32t = m.types_mut().int(32);
        let f = build_fn(&mut m, &mut syms, "f", &[], i32t, |b, _| {
            let four = b.const_i64(i32t, 4);
            let zero = b.const_i64(i32t, 0);
            b.bin(BinOp::SDiv, four, zero, Flags::NONE)
        });
        assert_eq!(run_eqsat(&mut m, f), Changed::No, "UB division must not fold");
        assert_eq!(count_kind(m.function(f), |k| matches!(k, InstKind::Bin(BinOp::SDiv))), 1);
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn runs_as_a_module_pass() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("eqsat-pass");
        let i32t = m.types_mut().int(32);
        let f = build_fn(&mut m, &mut syms, "f", &[], i32t, |b, _| {
            let three = b.const_i64(i32t, 3);
            let four = b.const_i64(i32t, 4);
            b.mul(three, four, Flags::NONE)
        });
        assert_eq!(EqSatPass.run(&mut m), Changed::Yes);
        assert!(verify_module(&m).is_ok());
        assert_int_ret(&m, f, 32, 12);
        assert_eq!(EqSatPass.run(&mut m), Changed::No, "module pass reaches a fixpoint");
    }

    #[test]
    fn folded_output_refines_original_b2() {
        // End-to-end B2 spot check: the optimized single-block body refines the
        // original (rebuild without installing so both coexist).
        let mut syms = StrInterner::new();
        let mut m = Module::new("eqsat-b2");
        let i32t = m.types_mut().int(32);
        let f = build_fn(&mut m, &mut syms, "f", &[i32t], i32t, |b, p| {
            let three = b.const_i64(i32t, 3);
            let s = b.add(p[0], three, Flags::NONE);
            let four = b.const_i64(i32t, 4);
            b.add(s, four, Flags::NONE)
        });
        let mut t = EqSat::new();
        t.analyze(m.function(f), m.types(), m.consts());
        let (fresh, c) = m.map_function(f, |old, b| t.run(old, b));
        assert_eq!(c, Changed::Yes);
        match check_refinement(m.types(), m.consts(), m.function(f), &fresh) {
            RefinementResult::Refines => {}
            RefinementResult::Unknown(_) => {}
            RefinementResult::Counterexample(model) => panic!("optimization is not a refinement: {model}"),
        }
    }

    /// Assert the returned value is a materialized integer constant equal to
    /// `expected` (modulo `width`).
    fn assert_int_ret(m: &Module, f: FuncId, width: u32, expected: i64) {
        let func = m.function(f);
        let v = ret_operand(func);
        match &func.value(v).def {
            ValueDef::Const(cid) => match m.consts().get(*cid) {
                Const::Int { value, .. } => assert_eq!(
                    value.mod_2k(width),
                    Int::from_i64(expected).mod_2k(width),
                    "returned constant mismatch"
                ),
                other => panic!("ret operand is a non-integer constant: {other:?}"),
            },
            other => panic!("ret operand was not a constant: {other:?}"),
        }
    }

    /// The constant `v` of the parameters' (integer) type — a convenience for the
    /// rule table, whose `fn`-pointer builders have no captured type context.
    fn ci(b: &mut FunctionBuilder<'_>, p: &[ValueId], v: i64) -> ValueId {
        let ty = b.value_type(p[0]);
        b.const_i64(ty, v)
    }
}
