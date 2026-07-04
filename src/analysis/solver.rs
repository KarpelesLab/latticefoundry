//! The generic monotone fixpoint engine (bet **B8**): a **sparse, SSA-based**
//! worklist solver parameterized by any [`AbstractDomain`].
//!
//! # Algorithm
//!
//! This is sparse conditional constant propagation (Wegman–Zadeck) generalized
//! to an arbitrary domain. It computes, at once, two interdependent fixpoints:
//!
//! - **block reachability** — which blocks are executable, refined by pruning
//!   branch edges a domain proves infeasible ([`AbstractDomain::edge_feasible`]);
//! - **a per-`ValueId` abstract state** — the abstract value of every SSA value.
//!
//! It is *sparse* because it propagates along SSA def→use edges (via the
//! function's use lists) rather than recomputing whole blocks, and *conditional*
//! because a value is only evaluated once its defining block is proven
//! reachable. Two worklists drive it: a **block** worklist (blocks newly proven
//! executable, whose instructions must be visited) and an **SSA** worklist
//! (values whose abstract state changed, whose uses must be re-evaluated).
//!
//! ## Block-argument edges
//!
//! There are no φ-nodes; a terminator carries a per-edge argument list that maps
//! positionally to the target block's parameters (`docs/ir-design.md`). When an
//! edge `pred → succ` is executable, each target parameter absorbs the matching
//! argument value: `param ← param ⊔ arg`. Re-evaluating the terminator whenever
//! an argument's abstract value changes keeps the parameter states current.
//!
//! ## Widening
//!
//! At a **loop header** (a block that is the target of a back edge, from
//! [`Dominators`]) a parameter update applies [`AbstractDomain::widen`] instead
//! of a plain join, so a tall or infinite-height domain still terminates. For a
//! finite-height domain (flat constants) `widen` defaults to `join`, so the
//! extra step is a no-op.
//!
//! ## Determinism
//!
//! Every collection that affects the result is a `Vec` indexed by dense ids or a
//! deterministic worklist; there is no reliance on `std::collections::HashMap`
//! iteration order (tenet T5).

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::analysis::domain::{AbstractDomain, DomainCtx, EdgeGuard};
use crate::ir::inst::{InstData, InstKind};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{ConstPool, ValueDef, ValueId};
use crate::ir::{BlockId, Function};

use puremp::Int;

/// The computed fixpoint of an analysis over one function: an abstract value per
/// [`ValueId`] and executability per [`BlockId`].
///
/// A value defined in an unreachable block, or not yet constrained, is
/// [`AbstractDomain::bottom`]; a block that the solver never proved executable
/// is not reachable.
#[derive(Debug)]
pub struct FixpointResult<D: AbstractDomain> {
    values: Vec<D>,
    block_reachable: Vec<bool>,
}

impl<D: AbstractDomain> FixpointResult<D> {
    /// The abstract value of SSA value `v`.
    pub fn value(&self, v: ValueId) -> &D {
        &self.values[v.index()]
    }

    /// Whether block `b` was proven executable.
    pub fn is_reachable(&self, b: BlockId) -> bool {
        self.block_reachable.get(b.index()).copied().unwrap_or(false)
    }

    /// The number of SSA values covered.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }
}

/// Run the fixpoint engine for domain `D` over `func`, resolving constants
/// through `consts` and types through `types`.
pub fn solve<D: AbstractDomain>(
    func: &Function,
    types: &TypeContext,
    consts: &ConstPool,
) -> FixpointResult<D> {
    let n_vals = func.value_count();
    let n_blocks = func.block_count();

    // A declaration (or entryless function) has no reachable code.
    let Some(entry) = func.entry() else {
        return FixpointResult {
            values: vec![D::bottom(); n_vals],
            block_reachable: vec![false; n_blocks],
        };
    };

    let mut engine = Engine::new(func, types, consts, entry);
    engine.run(entry);
    FixpointResult { values: engine.vals, block_reachable: engine.block_exec }
}

/// The mutable working state of one solver run.
struct Engine<'a, D: AbstractDomain> {
    func: &'a Function,
    types: &'a TypeContext,
    cfg: ControlFlowGraph,
    doms: Dominators,
    /// `inst_block[i]` is the block an instruction belongs to (`None` for
    /// dangling/never-placed instructions).
    inst_block: Vec<Option<usize>>,
    vals: Vec<D>,
    block_exec: Vec<bool>,
    /// `edge_exec[b][k]` — whether the `k`-th outgoing edge of block `b` is
    /// executable.
    edge_exec: Vec<Vec<bool>>,
    block_wl: Vec<usize>,
    ssa_wl: Vec<ValueId>,
}

impl<'a, D: AbstractDomain> Engine<'a, D> {
    fn new(
        func: &'a Function,
        types: &'a TypeContext,
        consts: &'a ConstPool,
        entry: BlockId,
    ) -> Self {
        let n_vals = func.value_count();
        let n_blocks = func.block_count();
        let cfg = ControlFlowGraph::new(func);
        let doms = Dominators::new(func, &cfg);
        let ctx = DomainCtx::new(types);

        // Map every placed instruction to its block.
        let mut inst_block = vec![None; func.inst_count()];
        for (bid, block) in func.blocks() {
            for &inst in block.insts() {
                inst_block[inst.index()] = Some(bid.index());
            }
            if let Some(t) = block.terminator() {
                inst_block[t.index()] = Some(bid.index());
            }
        }

        // Seed the initial abstract state of every value.
        let mut vals = Vec::with_capacity(n_vals);
        for i in 0..n_vals {
            let v = ValueId::from_index(i);
            let init = match &func.value(v).def {
                ValueDef::Const(cid) => D::abstract_const(ctx, consts.get(*cid)),
                // A global or function address is an opaque pointer: unknown.
                ValueDef::Global(_) | ValueDef::Func(_) => D::top(),
                // Entry-block parameters are the function inputs: unknown.
                // Other block parameters are determined by their incoming edges.
                ValueDef::Param(block, _) => {
                    if *block == entry {
                        D::top()
                    } else {
                        D::bottom()
                    }
                }
                // Instruction results start undetermined and are computed once
                // their block is proven executable.
                ValueDef::Inst(_) => D::bottom(),
            };
            vals.push(init);
        }

        let edge_exec = (0..n_blocks).map(|b| vec![false; cfg.successors(b).len()]).collect();

        Engine {
            func,
            types,
            cfg,
            doms,
            inst_block,
            vals,
            block_exec: vec![false; n_blocks],
            edge_exec,
            block_wl: Vec::new(),
            ssa_wl: Vec::new(),
        }
    }

    fn run(&mut self, entry: BlockId) {
        self.mark_block_exec(entry.index());
        loop {
            if let Some(b) = self.block_wl.pop() {
                self.visit_block(b);
            } else if let Some(v) = self.ssa_wl.pop() {
                self.process_ssa(v);
            } else {
                break;
            }
        }
    }

    /// Mark block `b` executable and schedule a visit of its body.
    fn mark_block_exec(&mut self, b: usize) {
        if !self.block_exec[b] {
            self.block_exec[b] = true;
            self.block_wl.push(b);
        }
    }

    /// Visit every value-producing instruction of `b`, then its terminator.
    fn visit_block(&mut self, b: usize) {
        let block = self.func.block(BlockId::from_index(b));
        for &inst in block.insts() {
            self.visit_inst(inst);
        }
        self.visit_terminator(b);
    }

    /// Recompute the abstract result of a value-producing instruction; if it
    /// changed, propagate to its uses.
    fn visit_inst(&mut self, inst: crate::ir::InstId) {
        let data = self.func.inst(inst);
        let Some(result) = data.result() else {
            return; // no result (store, void call): nothing to compute
        };
        let operands: Vec<D> =
            data.operands().iter().map(|&op| self.vals[op.index()].clone()).collect();
        let ctx = DomainCtx::new(self.types);
        let new = D::transfer(ctx, data, &operands);
        if new != self.vals[result.index()] {
            self.vals[result.index()] = new;
            self.ssa_wl.push(result);
        }
    }

    /// Re-evaluate the terminator of block `b`: prune infeasible edges, mark the
    /// feasible ones executable, and push each target's block-argument values
    /// into the target parameters.
    fn visit_terminator(&mut self, b: usize) {
        let Some(term_id) = self.func.block(BlockId::from_index(b)).terminator() else {
            return;
        };
        let term = self.func.inst(term_id);
        for si in self.feasible_edges(term) {
            let newly = !self.edge_exec[b][si];
            if newly {
                self.edge_exec[b][si] = true;
            }
            self.update_target_params(b, term, si);
            let target = self.cfg.successors(b)[si];
            self.mark_block_exec(target);
        }
    }

    /// The feasible outgoing edge indices of terminator `term`, in successor
    /// order, according to the domain's view of the branch condition.
    fn feasible_edges(&self, term: &InstData) -> Vec<usize> {
        match &term.kind {
            InstKind::Br(_) => vec![0],
            InstKind::CondBr { .. } => {
                let cond = &self.vals[term.operands()[0].index()];
                let mut f = Vec::with_capacity(2);
                if cond.edge_feasible(&EdgeGuard::CondIs(true)) {
                    f.push(0);
                }
                if cond.edge_feasible(&EdgeGuard::CondIs(false)) {
                    f.push(1);
                }
                f
            }
            InstKind::Switch(data) => {
                let cond_v = term.operands()[0];
                let cond = &self.vals[cond_v.index()];
                let width = int_width(self.types, self.func.value_type(cond_v)).unwrap_or(0);
                let case_values: Vec<Int> = data.cases.iter().map(|c| c.value.clone()).collect();
                let mut f = Vec::with_capacity(1 + data.cases.len());
                // Successor 0 is the default edge.
                if cond.edge_feasible(&EdgeGuard::CondNotAnyOf { values: &case_values, width }) {
                    f.push(0);
                }
                for (i, case) in data.cases.iter().enumerate() {
                    if cond.edge_feasible(&EdgeGuard::CondEquals { value: &case.value, width }) {
                        f.push(i + 1);
                    }
                }
                f
            }
            // `ret` / `unreachable`: no successors.
            _ => Vec::new(),
        }
    }

    /// Merge the block-argument values of edge `si` of `term` (in block `b`)
    /// into the target block's parameters, widening at loop headers.
    fn update_target_params(&mut self, b: usize, term: &InstData, si: usize) {
        let target = self.cfg.successors(b)[si];
        let widen = self.doms.is_loop_header(target);
        let args = edge_args(term, si);
        let params: Vec<ValueId> = self.func.block(BlockId::from_index(target)).params().to_vec();
        for (i, &param) in params.iter().enumerate() {
            let arg_val = match args.get(i) {
                Some(&arg) => self.vals[arg.index()].clone(),
                // A malformed edge with too few arguments is over-approximated
                // as top rather than derailing the analysis.
                None => D::top(),
            };
            let old = self.vals[param.index()].clone();
            let joined = old.join(&arg_val);
            let new = if widen { old.widen(&joined) } else { joined };
            if new != old {
                self.vals[param.index()] = new;
                self.ssa_wl.push(param);
            }
        }
    }

    /// Propagate a changed value to every use in an executable block.
    fn process_ssa(&mut self, v: ValueId) {
        for u in self.func.uses_of(v) {
            let inst = u.inst;
            let Some(b) = self.inst_block[inst.index()] else {
                continue;
            };
            if !self.block_exec[b] {
                continue;
            }
            if self.func.inst(inst).is_terminator() {
                self.visit_terminator(b);
            } else {
                self.visit_inst(inst);
            }
        }
    }
}

/// The block-argument slice of edge `si` of terminator `term`, matching the
/// target block's parameter list positionally.
fn edge_args(term: &InstData, si: usize) -> &[ValueId] {
    let ops = term.operands();
    match &term.kind {
        InstKind::Br(_) => ops,
        InstKind::CondBr { true_args, false_args, .. } => {
            let ta = *true_args as usize;
            let fa = *false_args as usize;
            if si == 0 {
                &ops[1..1 + ta]
            } else {
                &ops[1 + ta..1 + ta + fa]
            }
        }
        InstKind::Switch(data) => {
            // Operand layout: [cond, default args, case0 args, case1 args, ...].
            let da = data.default_args as usize;
            if si == 0 {
                &ops[1..1 + da]
            } else {
                let mut off = 1 + da;
                let case = si - 1;
                for c in &data.cases[..case] {
                    off += c.args as usize;
                }
                let len = data.cases[case].args as usize;
                &ops[off..off + len]
            }
        }
        _ => &[],
    }
}

/// The bit width of an integer type, or `None` if `ty` is not an integer type.
fn int_width(types: &TypeContext, ty: TypeId) -> Option<u32> {
    match types.get(ty) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}
