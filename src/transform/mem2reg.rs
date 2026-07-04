//! **mem2reg** — promote memory `alloca` slots to SSA values.
//!
//! An `alloca` is *promotable* when it is used **only** as the whole-slot address
//! of `load`/`store` instructions (accessing exactly the allocated type) and its
//! address never otherwise escapes — it is never a `ptr_add` base, a `call`
//! argument, a stored *value*, a returned value, or any other operand. Such a
//! slot behaves exactly like a local variable, so its loads and stores can be
//! replaced by direct SSA data flow, deleting the memory traffic entirely.
//!
//! ## Algorithm (Cytron et al., adapted to block arguments)
//!
//! There are no φ-nodes in this IR (`docs/ir-design.md` §2); an SSA merge is a
//! **block parameter** whose incoming values are supplied per-edge by each
//! predecessor's terminator. mem2reg therefore:
//!
//! 1. **Places parameters.** For each promoted slot, the blocks in the *iterated
//!    dominance frontier* of its store blocks each gain a new block parameter of
//!    the slot's element type (the block-argument analog of φ-placement). The
//!    entry block never qualifies (it has no predecessors), so its
//!    signature-fixed parameter list is preserved.
//! 2. **Renames.** Walking the dominator tree in preorder, it threads a *current
//!    reaching definition* per slot: entering a block that has a parameter for a
//!    slot makes that parameter the reaching definition; a `store` replaces it
//!    with the stored value; a `load` is deleted and every use of its result is
//!    rewritten to the reaching definition. A load with no reaching definition on
//!    its path reads uninitialized memory, which the `alloca` semantics define as
//!    **poison** — so it is rewritten to a `poison` value of the slot type.
//! 3. **Threads edge arguments.** At each terminator, every successor that has a
//!    promoted parameter receives, as an extra block argument on that edge, the
//!    reaching definition at the end of this block (or `poison`).
//!
//! The promoted `alloca`/`load`/`store` instructions are dropped; every other
//! instruction is copied verbatim with remapped operands. The result is verified
//! and refines the input (loads observe exactly the value the corresponding store
//! wrote, or poison for an uninitialized read).

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::analysis::domfrontier::DominanceFrontiers;
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::InstKind;
use crate::ir::types::TypeId;
use crate::ir::value::ValueId;
use crate::ir::{BlockId, Function};
use crate::pass::Changed;
use crate::transform::{FunctionTransform, rebuild_terminator, remap_value};

/// The mem2reg transform (see the module documentation).
#[derive(Debug, Default, Clone, Copy)]
pub struct Mem2Reg;

impl FunctionTransform for Mem2Reg {
    fn name(&self) -> &str {
        "mem2reg"
    }

    fn run(&mut self, old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
        promote(old, builder)
    }
}

/// Read-only analysis data shared across the rename walk. The mutable rename
/// state (`vmap`, `cur_def`) is kept in separate locals so their borrows stay
/// disjoint from these fields.
#[derive(Debug)]
struct Plan {
    /// The element type of each promoted slot (the type of its parameter).
    slots: Vec<TypeId>,
    /// `value_slot[alloca_value] = Some(slot)` for a promoted alloca's result.
    value_slot: Vec<Option<usize>>,
    /// Old block index → its block id in the rebuilt function.
    new_block: Vec<BlockId>,
    /// `promoted_params[b]` = the slots that block `b` gains a parameter for, in
    /// slot order (the order the extra parameters/arguments are appended).
    promoted_params: Vec<Vec<usize>>,
    /// `slot_param_val[b][s]` = the rebuilt parameter value for slot `s` in block
    /// `b`, when that block has one.
    slot_param_val: Vec<Vec<Option<ValueId>>>,
}

/// Whether the alloca whose result is `av` (of element type `elem_ty`) is
/// promotable: every use is a whole-slot `load`/`store` of `elem_ty` through the
/// address operand, never an escaping use.
fn is_promotable(old: &Function, av: ValueId, elem_ty: TypeId) -> bool {
    for u in old.uses_of(av) {
        match old.inst(u.inst).kind {
            // A load's only operand is the address; must access the slot type.
            InstKind::Load { ty, .. } => {
                if u.operand != 0 || ty != elem_ty {
                    return false;
                }
            }
            // A store uses the address at operand 0 and the value at operand 1;
            // the address must be operand 0 (else the pointer is stored, i.e.
            // escapes) and must write the slot type.
            InstKind::Store { ty, .. } => {
                if u.operand != 0 || ty != elem_ty {
                    return false;
                }
            }
            // Any other use (ptr_add base, call argument, ...) takes the address.
            _ => return false,
        }
    }
    true
}

fn promote(old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
    let n = old.block_count();
    let Some(entry) = old.entry() else {
        return Changed::No;
    };
    let entry_idx = entry.index();
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);

    // Locate every instruction in its block (for store-block identification).
    let mut inst_block = vec![usize::MAX; old.inst_count()];
    for (bid, blk) in old.blocks() {
        for &i in blk.insts() {
            inst_block[i.index()] = bid.index();
        }
        if let Some(t) = blk.terminator() {
            inst_block[t.index()] = bid.index();
        }
    }

    // Discover promotable allocas in a deterministic order.
    let mut slots: Vec<TypeId> = Vec::new();
    let mut value_slot: Vec<Option<usize>> = vec![None; old.value_count()];
    for (_bid, blk) in old.blocks() {
        for &i in blk.insts() {
            let inst = old.inst(i);
            if let InstKind::Alloca { elem_ty } = inst.kind {
                let av = inst.result().expect("alloca defines a value");
                if is_promotable(old, av, elem_ty) {
                    value_slot[av.index()] = Some(slots.len());
                    slots.push(elem_ty);
                }
            }
        }
    }
    if slots.is_empty() {
        return Changed::No;
    }

    // Place block parameters: a slot needs one in every block of the iterated
    // dominance frontier of the blocks that store to it (reachable only).
    let df = DominanceFrontiers::compute(&cfg, &doms, n);
    let mut has_param = vec![vec![false; slots.len()]; n];
    for (av_idx, slot) in value_slot.iter().enumerate() {
        let Some(s) = *slot else {
            continue;
        };
        let av = ValueId::from_index(av_idx);
        let mut defs = Vec::new();
        for u in old.uses_of(av) {
            if matches!(old.inst(u.inst).kind, InstKind::Store { .. }) && u.operand == 0 {
                let b = inst_block[u.inst.index()];
                if b != usize::MAX && doms.is_reachable(b) {
                    defs.push(b);
                }
            }
        }
        defs.sort_unstable();
        defs.dedup();
        for y in df.iterated(&defs) {
            if y != entry_idx && doms.is_reachable(y) {
                has_param[y][s] = true;
            }
        }
    }
    let promoted_params: Vec<Vec<usize>> = (0..n)
        .map(|b| (0..slots.len()).filter(|&s| has_param[b][s]).collect())
        .collect();

    // Create the rebuilt blocks: original parameters, then the promoted ones.
    let mut new_block: Vec<Option<BlockId>> = vec![None; n];
    new_block[entry_idx] = Some(builder.create_entry_block());
    for b in 0..n {
        if b == entry_idx {
            continue;
        }
        let bb = BlockId::from_index(b);
        let mut ptys: Vec<TypeId> =
            old.block(bb).params().iter().map(|&p| old.value_type(p)).collect();
        for &s in &promoted_params[b] {
            ptys.push(slots[s]);
        }
        new_block[b] = Some(builder.create_block(&ptys));
    }
    let new_block: Vec<BlockId> =
        new_block.into_iter().map(|x| x.expect("every block was created")).collect();

    // Seed the value map with the rebuilt block parameters, and record each
    // promoted parameter value for the rename walk.
    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    let mut slot_param_val = vec![vec![None; slots.len()]; n];
    for b in 0..n {
        let bb = BlockId::from_index(b);
        let base = old.block(bb).params().len();
        let new_params = builder.block_params(new_block[b]).to_vec();
        for (i, &op) in old.block(bb).params().iter().enumerate() {
            vmap[op.index()] = Some(new_params[i]);
        }
        for (k, &s) in promoted_params[b].iter().enumerate() {
            slot_param_val[b][s] = Some(new_params[base + k]);
        }
    }

    let plan = Plan { slots, value_slot, new_block, promoted_params, slot_param_val };

    // Rename in dominator-tree preorder, threading reaching definitions with a
    // save/restore discipline so siblings do not see each other's stores.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 0..n {
        if b != entry_idx
            && doms.is_reachable(b)
            && let Some(ip) = doms.idom(b)
        {
            children[ip].push(b);
        }
    }
    for c in &mut children {
        c.sort_unstable();
    }

    let mut cur_def: Vec<Option<ValueId>> = vec![None; plan.slots.len()];
    process_block(&plan, &mut vmap, old, builder, &mut cur_def, entry_idx);
    // Frame: (block, next child, reaching state captured *before* the block ran).
    let mut stack: Vec<(usize, usize, Vec<Option<ValueId>>)> =
        vec![(entry_idx, 0, vec![None; plan.slots.len()])];
    while let Some(&(b, ci, _)) = stack.last() {
        if ci < children[b].len() {
            let c = children[b][ci];
            stack.last_mut().expect("nonempty").1 += 1;
            let snapshot = cur_def.clone();
            process_block(&plan, &mut vmap, old, builder, &mut cur_def, c);
            stack.push((c, 0, snapshot));
        } else {
            cur_def = stack.pop().expect("nonempty").2;
        }
    }

    // Unreachable blocks never execute; copy them for well-formedness with an
    // empty reaching state (loads and promoted edge arguments become poison).
    for b in 0..n {
        if !doms.is_reachable(b) {
            let mut dead = vec![None; plan.slots.len()];
            process_block(&plan, &mut vmap, old, builder, &mut dead, b);
        }
    }

    Changed::Yes
}

/// Rebuild block `b`: apply its promoted-parameter overrides to `cur_def`, emit
/// its (non-promoted) instructions with loads/stores folded into the reaching
/// definitions, then rebuild its terminator threading the reaching definitions as
/// per-edge block arguments.
fn process_block(
    plan: &Plan,
    vmap: &mut [Option<ValueId>],
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    cur_def: &mut [Option<ValueId>],
    b: usize,
) {
    let bb = BlockId::from_index(b);
    builder.switch_to(plan.new_block[b]);

    // Entering the block, its promoted parameters become the reaching defs.
    for (dst, param) in cur_def.iter_mut().zip(plan.slot_param_val[b].iter()) {
        if let Some(pv) = *param {
            *dst = Some(pv);
        }
    }

    let insts = old.block(bb).insts().to_vec();
    for i in insts {
        let inst = old.inst(i);
        match &inst.kind {
            InstKind::Alloca { .. } => {
                let av = inst.result().expect("alloca has a result");
                if plan.value_slot[av.index()].is_some() {
                    // Promoted slot: drop the alloca entirely.
                } else {
                    let ty = inst.ty;
                    let kind = inst.kind.clone();
                    let flags = inst.flags;
                    let nr = builder.append_inst(kind, Vec::new(), flags, Some(ty));
                    vmap[av.index()] = nr;
                }
            }
            InstKind::Load { .. } => {
                let ptr = inst.operands()[0];
                if let Some(s) = plan.value_slot[ptr.index()] {
                    let res = inst.result().expect("load has a result");
                    let reaching = match cur_def[s] {
                        Some(v) => v,
                        None => builder.poison(plan.slots[s]),
                    };
                    vmap[res.index()] = Some(reaching);
                } else {
                    copy_generic(vmap, old, builder, inst);
                }
            }
            InstKind::Store { .. } => {
                let ptr = inst.operands()[0];
                if let Some(s) = plan.value_slot[ptr.index()] {
                    let valop = inst.operands()[1];
                    cur_def[s] = Some(remap_value(vmap, old, builder, valop));
                } else {
                    copy_generic(vmap, old, builder, inst);
                }
            }
            _ => copy_generic(vmap, old, builder, inst),
        }
    }

    let extra = |b: &mut FunctionBuilder<'_>, target: BlockId, args: &mut Vec<ValueId>| {
        for &s in &plan.promoted_params[target.index()] {
            let v = match cur_def[s] {
                Some(v) => v,
                None => b.poison(plan.slots[s]),
            };
            args.push(v);
        }
    };
    rebuild_terminator(vmap, old, builder, &plan.new_block, bb, extra);
}

/// Copy an instruction verbatim with remapped operands, recording its result.
fn copy_generic(
    vmap: &mut [Option<ValueId>],
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    inst: &crate::ir::inst::InstData,
) {
    let mut ops = Vec::with_capacity(inst.operands().len());
    for &o in inst.operands() {
        ops.push(remap_value(vmap, old, builder, o));
    }
    let result_ty = inst.result().map(|_| inst.ty);
    let nr = builder.append_inst(inst.kind.clone(), ops, inst.flags, result_ty);
    if let Some(r) = inst.result() {
        vmap[r.index()] = nr;
    }
}
