//! **DCE** — dead-code elimination.
//!
//! An instruction is *dead* when its result is used by nothing that ultimately
//! contributes to the function's observable behavior, and the instruction itself
//! has **no side effects**. Per the reference semantics the side-effecting /
//! always-live opcodes are `store`, `call`, and `alloca` (memory effects) and
//! every terminator (control flow); everything else — arithmetic, comparisons,
//! casts, `select`, `freeze`, `ptr_add`, and `load` — is a pure value whose only
//! reason to exist is its result.
//!
//! Liveness is a backward reachability fixpoint: seed the live set with the
//! side-effecting instructions and terminators, then repeatedly mark the
//! definitions of any live instruction's operands live, until it stabilizes.
//! Removing one dead value can expose its operands as dead, which the fixpoint
//! captures. The function is then rebuilt keeping only the live instructions;
//! dead results are referenced only by other dead instructions, so nothing that
//! survives can observe a dropped value.

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::InstKind;
use crate::ir::value::{ValueDef, ValueId};
use crate::ir::{BlockId, Function};
use crate::pass::Changed;
use crate::transform::{FunctionTransform, dom_preorder, rebuild_terminator, remap_value};

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

/// Whether an opcode has a side effect that keeps it live regardless of use.
fn has_side_effect(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Alloca { .. }
            | InstKind::DynAlloca { .. }
            | InstKind::Store { .. }
            | InstKind::Call
    )
}

/// Mark instruction `i` live and enqueue it, if it was not already.
fn mark(i: crate::ir::InstId, live: &mut [bool], worklist: &mut Vec<crate::ir::InstId>) {
    if !live[i.index()] {
        live[i.index()] = true;
        worklist.push(i);
    }
}

fn eliminate(old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
    let n = old.block_count();
    if old.entry().is_none() {
        return Changed::No;
    }

    // Seed liveness with side-effecting instructions and every terminator.
    let mut live = vec![false; old.inst_count()];
    let mut worklist: Vec<crate::ir::InstId> = Vec::new();
    for (_bid, blk) in old.blocks() {
        for &i in blk.insts() {
            if has_side_effect(&old.inst(i).kind) {
                mark(i, &mut live, &mut worklist);
            }
        }
        if let Some(t) = blk.terminator() {
            mark(t, &mut live, &mut worklist);
        }
    }

    // Backward fixpoint: a live instruction's operands' definitions are live.
    while let Some(i) = worklist.pop() {
        let inst = old.inst(i);
        for &op in inst.operands() {
            if let ValueDef::Inst(d) = old.value(op).def {
                mark(d, &mut live, &mut worklist);
            }
        }
    }

    // Nothing dead ⇒ no rebuild (keep the original body).
    let mut removed = 0usize;
    for (_bid, blk) in old.blocks() {
        for &i in blk.insts() {
            if !live[i.index()] {
                removed += 1;
            }
        }
    }
    if removed == 0 {
        return Changed::No;
    }

    // Rebuild: identical blocks/parameters/edges, dead instructions dropped.
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);
    let entry_idx = old.entry().expect("has an entry").index();

    let mut new_block: Vec<Option<BlockId>> = vec![None; n];
    new_block[entry_idx] = Some(builder.create_entry_block());
    for (b, slot) in new_block.iter_mut().enumerate() {
        if b == entry_idx {
            continue;
        }
        let bb = BlockId::from_index(b);
        let ptys: Vec<_> = old.block(bb).params().iter().map(|&p| old.value_type(p)).collect();
        *slot = Some(builder.create_block(&ptys));
    }
    let new_block: Vec<BlockId> =
        new_block.into_iter().map(|x| x.expect("every block was created")).collect();

    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for (b, &nb) in new_block.iter().enumerate() {
        let bb = BlockId::from_index(b);
        let new_params = builder.block_params(nb).to_vec();
        for (i, &op) in old.block(bb).params().iter().enumerate() {
            vmap[op.index()] = Some(new_params[i]);
        }
    }

    // Emit in dominator preorder so every surviving definition precedes its uses.
    for b in dom_preorder(old, &doms) {
        let bb = BlockId::from_index(b);
        builder.switch_to(new_block[b]);
        let insts = old.block(bb).insts().to_vec();
        for i in insts {
            if !live[i.index()] {
                continue;
            }
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
        rebuild_terminator(&mut vmap, old, builder, &new_block, bb, |_, _, _| {});
    }

    Changed::Yes
}
