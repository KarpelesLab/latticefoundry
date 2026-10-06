//! **DCE** — dead-code elimination.
//!
//! An instruction is *dead* when its result is used by nothing that ultimately
//! contributes to the function's observable behavior, and the instruction itself
//! has **no side effects**. Per the reference semantics the side-effecting /
//! always-live opcodes are `store`, `call`, `syscall`, `alloca`/`dyn_alloca`,
//! every atomic (`atomic_load`, `atomic_store`, `atomic_rmw`, `cmpxchg`) and
//! `fence`, a *volatile* `load` (see [`InstKind::has_side_effect`]), and every
//! terminator (control flow); everything else — arithmetic, comparisons, casts,
//! `select`, `freeze`, `ptr_add`, and a plain `load` — is a pure value whose
//! only reason to exist is its result.
//!
//! Liveness is a backward reachability fixpoint: seed the live set with the
//! side-effecting instructions and terminators, then repeatedly mark the
//! definitions of any live instruction's operands live, until it stabilizes.
//! Removing one dead value can expose its operands as dead, which the fixpoint
//! captures. The function is then rebuilt keeping only the live instructions;
//! dead results are referenced only by other dead instructions, so nothing that
//! survives can observe a dropped value.
//!
//! **Block parameters** take part in the same fixpoint. A branch is always
//! live, but its edge arguments are not operands like the others: an argument
//! is live only if the parameter it feeds is, and a parameter is live only if
//! a live instruction (a branch condition, a returned value, ...) or a live
//! parameter's argument uses it. So a parameter nothing uses — for instance one
//! `mem2reg` placed at a loop header for a slot whose value is dead there — is
//! dropped together with its argument on every incoming `br`, `cond_br` and
//! `switch` edge, and a loop that only threads a value around to itself drops
//! the whole cycle. The entry block's parameters are the function's parameters
//! and are always kept. Dropping an argument only removes a data flow into a
//! value nothing reads, so this is a refinement and adds no control flow.

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::InstKind;
use crate::ir::value::{ValueDef, ValueId};
use crate::ir::{BlockId, Function, InstId};
use crate::pass::Changed;
use crate::transform::{FunctionTransform, dom_preorder, edge_args, rebuild_terminator_keeping, remap_value};

/// The dead-code-elimination transform (see the module documentation).
#[derive(Debug, Default, Clone, Copy)]
pub struct Dce;

impl FunctionTransform for Dce {
    fn name(&self) -> &str {
        "dce"
    }

    fn run(&mut self, old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
        eliminate(old, builder)
    }
}

/// Whether an opcode has a side effect that keeps it live regardless of use
/// (stores, calls, syscalls, allocations, atomics, fences, volatile loads).
fn has_side_effect(kind: &InstKind) -> bool {
    kind.has_side_effect()
}

/// A node of the liveness graph: an instruction, or a block parameter (by
/// its value id).
#[derive(Clone, Copy)]
enum Node {
    Inst(InstId),
    Param(ValueId),
}

/// The liveness state of one function (see the module documentation).
struct Liveness {
    inst: Vec<bool>,
    param: Vec<bool>,
    worklist: Vec<Node>,
}

impl Liveness {
    fn mark_inst(&mut self, i: InstId) {
        if !self.inst[i.index()] {
            self.inst[i.index()] = true;
            self.worklist.push(Node::Inst(i));
        }
    }

    /// Mark the definition of operand `v` live (constants and symbol
    /// references have no definition to keep).
    fn mark_value(&mut self, old: &Function, v: ValueId) {
        match old.value(v).def {
            ValueDef::Inst(d) => self.mark_inst(d),
            ValueDef::Param(..) if !self.param[v.index()] => {
                self.param[v.index()] = true;
                self.worklist.push(Node::Param(v));
            }
            _ => {}
        }
    }
}

fn eliminate(old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
    let n = old.block_count();
    let Some(entry) = old.entry() else {
        return Changed::No;
    };

    // Every edge into each block: (terminator, successor index).
    let mut incoming: Vec<Vec<(InstId, usize)>> = vec![Vec::new(); n];
    for (_bid, blk) in old.blocks() {
        if let Some(t) = blk.terminator() {
            for (si, succ) in old.inst(t).successors().into_iter().enumerate() {
                incoming[succ.index()].push((t, si));
            }
        }
    }

    // Seed liveness with side-effecting instructions, every terminator, and the
    // entry block's parameters (the function's own parameters).
    let mut lv = Liveness {
        inst: vec![false; old.inst_count()],
        param: vec![false; old.value_count()],
        worklist: Vec::new(),
    };
    for &p in old.block(entry).params() {
        lv.mark_value(old, p);
    }
    for (_bid, blk) in old.blocks() {
        for &i in blk.insts() {
            if has_side_effect(&old.inst(i).kind) {
                lv.mark_inst(i);
            }
        }
        if let Some(t) = blk.terminator() {
            lv.mark_inst(t);
        }
    }

    // Backward fixpoint. A live instruction's operands are live — except a
    // branch's edge arguments, which are live only as far as the parameter they
    // feed is. A live parameter makes its argument on every incoming edge live.
    while let Some(node) = lv.worklist.pop() {
        match node {
            Node::Inst(i) => {
                let inst = old.inst(i);
                let ops = inst.operands();
                let non_edge = match inst.kind {
                    InstKind::Br(_) => &ops[..0],
                    InstKind::CondBr { .. } | InstKind::Switch(_) => &ops[..1],
                    _ => ops,
                };
                for &op in non_edge {
                    lv.mark_value(old, op);
                }
            }
            Node::Param(p) => {
                let ValueDef::Param(b, k) = old.value(p).def else {
                    continue;
                };
                for &(t, si) in &incoming[b.index()] {
                    let args = edge_args(old.inst(t), si);
                    if let Some(&a) = args.get(k as usize) {
                        lv.mark_value(old, a);
                    }
                }
            }
        }
    }

    // Nothing dead ⇒ no rebuild (keep the original body).
    let mut removed = 0usize;
    for (_bid, blk) in old.blocks() {
        removed += blk.insts().iter().filter(|i| !lv.inst[i.index()]).count();
        removed += blk.params().iter().filter(|p| !lv.param[p.index()]).count();
    }
    if removed == 0 {
        return Changed::No;
    }

    // Rebuild: identical blocks and edges, dead instructions and dead block
    // parameters (with their edge arguments) dropped.
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);
    let entry_idx = entry.index();

    let mut new_block: Vec<Option<BlockId>> = vec![None; n];
    new_block[entry_idx] = Some(builder.create_entry_block());
    for (b, slot) in new_block.iter_mut().enumerate() {
        if b == entry_idx {
            continue;
        }
        let bb = BlockId::from_index(b);
        let ptys: Vec<_> = old
            .block(bb)
            .params()
            .iter()
            .filter(|p| lv.param[p.index()])
            .map(|&p| old.value_type(p))
            .collect();
        *slot = Some(builder.create_block(&ptys));
    }
    let new_block: Vec<BlockId> =
        new_block.into_iter().map(|x| x.expect("every block was created")).collect();

    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for (b, &nb) in new_block.iter().enumerate() {
        let bb = BlockId::from_index(b);
        let new_params = builder.block_params(nb).to_vec();
        let live_params = old.block(bb).params().iter().filter(|p| lv.param[p.index()]);
        for (&op, &np) in live_params.zip(new_params.iter()) {
            vmap[op.index()] = Some(np);
        }
    }

    // Emit in dominator preorder so every surviving definition precedes its uses.
    let keep = |t: BlockId, k: usize| lv.param[old.block(t).params()[k].index()];
    for b in dom_preorder(old, &doms) {
        let bb = BlockId::from_index(b);
        builder.switch_to(new_block[b]);
        let insts = old.block(bb).insts().to_vec();
        for i in insts {
            if !lv.inst[i.index()] {
                continue;
            }
            builder.set_line_from(old, i);
            let inst = old.inst(i);
            let mut ops = Vec::with_capacity(inst.operands().len());
            for &o in inst.operands() {
                ops.push(remap_value(&mut vmap, old, builder, o));
            }
            let result_ty = inst.result().map(|_| inst.ty);
            let nr = builder.append_inst(inst.kind.clone(), ops, inst.flags, result_ty);
            if let Some(r) = inst.result() {
                vmap[r.index()] = nr;
            }
        }
        rebuild_terminator_keeping(&mut vmap, old, builder, &new_block, bb, keep, |_, _, _| {});
    }

    Changed::Yes
}
