//! **LICM** — loop-invariant code motion.
//!
//! An instruction inside a loop is *loop-invariant* when it recomputes the same
//! value on every iteration. Hoisting such a computation to a point that runs
//! once before the loop (a **preheader**) removes redundant work without changing
//! meaning. This pass is a functional rebuild (tenet T5): it reads the old
//! [`Function`] and constructs a fresh, refining one, moving invariant
//! computations into freshly created preheader blocks.
//!
//! ## Finding loops
//!
//! From the dominator tree and the CFG we find every **back edge** `n -> h`
//! (an edge whose target `h` dominates its source `n`). The **natural loop** of
//! that back edge is `h` together with every block that can reach `n` without
//! passing through `h`; back edges that share a header are unioned into one loop.
//! A loop's **header** dominates all of its blocks, so any value that is live
//! into the loop and defined outside it dominates the header (and hence the
//! preheader we insert just above the header).
//!
//! ## Preheaders
//!
//! For each loop we actually hoist into, we materialize a fresh preheader block
//! carrying a *copy* of the header's parameter list. Every **out-of-loop**
//! predecessor of the header is redirected to branch to the preheader instead,
//! passing exactly the block arguments it used to pass to the header
//! (`docs/ir-design.md` §2: loop-carried values are block parameters, not
//! φ-nodes); the preheader forwards its own parameters on to the header. Back
//! edges (in-loop predecessors) keep branching straight to the header, so the
//! loop-carried dataflow is untouched. A loop whose header is the function entry
//! is skipped (the entry cannot be given a predecessor), which only forgoes an
//! optimization and never affects soundness.
//!
//! ## Invariance (a fixpoint)
//!
//! A **pure** instruction in loop `L` is invariant relative to `L` iff every
//! operand is either a constant / global / function reference, or is defined
//! *outside* `L` by a definition that dominates `L`'s header, or is itself an
//! instruction hoisted out of `L` (to `L`'s preheader or an enclosing loop's).
//! The last clause is recursive, so we iterate to a fixpoint. Because "defined
//! outside the outer loop" implies "defined outside the inner loop," an
//! instruction invariant w.r.t. an outer loop is invariant w.r.t. every inner
//! one; each instruction is therefore hoisted to the preheader of the *outermost*
//! loop it is invariant in — an instruction invariant only w.r.t. an inner loop
//! is hoisted just out of that inner loop, not further.
//!
//! ## Speculation safety
//!
//! Hoisting makes an instruction run unconditionally whenever the loop is
//! entered, even if inside the loop it ran only conditionally. In this IR every
//! pure operation is **poison-on-error, never UB** (`docs/ir-design.md` §5, §7):
//! `nsw`/`nuw` overflow, over-wide shifts, `exact` violations, out-of-range
//! float→int casts, etc. all yield *poison*, and computing a poison value that is
//! then discarded is harmless. So speculatively executing any such pure op in the
//! preheader is a sound refinement. The **only** pure ops with genuine undefined
//! behavior are integer `udiv`/`sdiv`/`urem`/`srem` (division by zero, and
//! `INT_MIN / -1`), so hoisting one could introduce UB on a path that never
//! divided. A [`FunctionTransform`] cannot inspect interned constant *values*
//! (it is handed only the old function and a builder, not the constant pool), so
//! we cannot prove a divisor nonzero here; we therefore **conservatively never
//! hoist integer div/rem**. Floating-point `fdiv`/`frem` have no UB (they yield
//! ±inf/NaN) and are hoisted like any other pure op.
//!
//! The result verifies and refines the input: each invariant value is computed
//! once in the preheader and every in-loop use now reads that single definition,
//! which — being invariant — equals what the loop would have recomputed.

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, InstKind};
use crate::ir::types::TypeId;
use crate::ir::value::{ValueDef, ValueId};
use crate::ir::{BlockId, Function, InstId};
use crate::pass::Changed;
use crate::transform::{FunctionTransform, dom_preorder, remap_value};

/// The loop-invariant-code-motion transform (see the module documentation).
#[derive(Debug, Default, Clone, Copy)]
pub struct Licm;

impl FunctionTransform for Licm {
    fn name(&self) -> &str {
        "licm"
    }

    fn run(&mut self, old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
        hoist(old, builder)
    }
}

/// One natural loop: its header block, a membership mask and block list over the
/// old CFG, its size (for nest ordering), and whether it may host a preheader
/// (its header is not the function entry).
#[derive(Debug)]
struct LoopInfo {
    header: usize,
    mask: Vec<bool>,
    blocks: Vec<usize>,
    size: usize,
    target: bool,
}

/// Whether an opcode is a pure value producer that LICM may relocate. Excludes
/// terminators, the memory/effect ops (`load`/`store`/`call`/`alloca`), and —
/// for speculation safety — integer division/remainder (the only pure ops that
/// can trigger undefined behavior); see the module docs.
fn is_hoistable_kind(kind: &InstKind) -> bool {
    match kind {
        InstKind::Bin(b) => !matches!(b, BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem),
        InstKind::Unary(_)
        | InstKind::ICmp(_)
        | InstKind::FCmp(_)
        | InstKind::Cast(_)
        | InstKind::Select
        | InstKind::Freeze
        | InstKind::PtrAdd { .. } => true,
        _ => false,
    }
}

/// Discover every natural loop, unioning back edges that share a header.
fn find_loops(
    old: &Function,
    cfg: &ControlFlowGraph,
    doms: &Dominators,
    entry_idx: usize,
) -> Vec<LoopInfo> {
    let n = old.block_count();
    let mut loops = Vec::new();
    for h in 0..n {
        if !doms.is_reachable(h) {
            continue;
        }
        // Latches: reachable predecessors `p` of `h` with `h` dominating `p`.
        let mut latches = Vec::new();
        for &p in cfg.predecessors(h) {
            if doms.is_reachable(p) && doms.dominates(h, p) {
                latches.push(p);
            }
        }
        if latches.is_empty() {
            continue;
        }
        // Natural loop: `h`, plus everything reaching a latch without crossing
        // `h`. `mask[h]` starts set, so the backward walk never expands past it.
        let mut mask = vec![false; n];
        mask[h] = true;
        let mut work = Vec::new();
        for &p in &latches {
            if !mask[p] {
                mask[p] = true;
                work.push(p);
            }
        }
        while let Some(x) = work.pop() {
            for &pp in cfg.predecessors(x) {
                if doms.is_reachable(pp) && !mask[pp] {
                    mask[pp] = true;
                    work.push(pp);
                }
            }
        }
        let blocks: Vec<usize> = (0..n).filter(|&b| mask[b]).collect();
        let size = blocks.len();
        loops.push(LoopInfo { header: h, mask, blocks, size, target: h != entry_idx });
    }
    loops
}

/// Whether loop `dl` contains all of loop `l`'s blocks (i.e. `dl` is `l` itself
/// or an enclosing loop), so a value in `dl`'s preheader is available outside `l`.
fn loop_superset(loops: &[LoopInfo], dl: usize, l: usize) -> bool {
    dl == l || loops[l].blocks.iter().all(|&x| loops[dl].mask[x])
}

/// Whether instruction `i` is invariant relative to loop `l`, given the current
/// hoist destinations of every instruction.
fn invariant_relative(
    old: &Function,
    loops: &[LoopInfo],
    doms: &Dominators,
    inst_block: &[usize],
    destination: &[Option<usize>],
    i: usize,
    l: usize,
) -> bool {
    let header = loops[l].header;
    for &op in old.inst(InstId::from_index(i)).operands() {
        match &old.value(op).def {
            // Available everywhere.
            ValueDef::Const(_) | ValueDef::Global(_) | ValueDef::Func(_) => {}
            // A block parameter: usable in the preheader iff it is defined
            // outside the loop by a block dominating the header.
            ValueDef::Param(pb, _) => {
                let pb = pb.index();
                if loops[l].mask[pb] || !doms.dominates(pb, header) {
                    return false;
                }
            }
            // An instruction result: either defined outside the loop (and
            // dominating the header), or itself hoisted out of this loop.
            ValueDef::Inst(di) => {
                let dbi = inst_block[di.index()];
                if dbi == usize::MAX {
                    return false;
                }
                if !loops[l].mask[dbi] {
                    if !doms.dominates(dbi, header) {
                        return false;
                    }
                } else {
                    match destination[di.index()] {
                        Some(dl) if loop_superset(loops, dl, l) => {}
                        _ => return false,
                    }
                }
            }
        }
    }
    true
}

/// Copy instruction `i` into the current insertion block with remapped operands,
/// recording its result in `vmap`.
fn emit_inst(
    vmap: &mut [Option<ValueId>],
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    i: InstId,
) {
    let inst = old.inst(i);
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

/// Rebuild the terminator of old block `bb`, remapping operands through `vmap`
/// and each successor through `resolve` (which redirects out-of-loop edges to
/// preheaders). LICM adds no extra block arguments.
fn emit_term(
    vmap: &mut [Option<ValueId>],
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    bb: BlockId,
    resolve: impl Fn(BlockId) -> BlockId,
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
                Some(remap_value(vmap, old, builder, ops[0]))
            };
            builder.ret(v);
        }
        InstKind::Unreachable => builder.unreachable(),
        InstKind::Br(target) => {
            let mut args = Vec::with_capacity(ops.len());
            for &o in ops {
                args.push(remap_value(vmap, old, builder, o));
            }
            builder.br(resolve(*target), &args);
        }
        InstKind::CondBr { if_true, if_false, true_args, false_args } => {
            let ta = *true_args as usize;
            let fa = *false_args as usize;
            let cond = remap_value(vmap, old, builder, ops[0]);
            let mut targs = Vec::with_capacity(ta);
            for k in 0..ta {
                targs.push(remap_value(vmap, old, builder, ops[1 + k]));
            }
            let mut fargs = Vec::with_capacity(fa);
            for k in 0..fa {
                fargs.push(remap_value(vmap, old, builder, ops[1 + ta + k]));
            }
            builder.cond_br(cond, resolve(*if_true), &targs, resolve(*if_false), &fargs);
        }
        InstKind::Switch(data) => {
            let cond = remap_value(vmap, old, builder, ops[0]);
            let da = data.default_args as usize;
            let mut dargs = Vec::with_capacity(da);
            for k in 0..da {
                dargs.push(remap_value(vmap, old, builder, ops[1 + k]));
            }
            let mut cases = Vec::with_capacity(data.cases.len());
            let mut off = 1 + da;
            for c in &data.cases {
                let ca = c.args as usize;
                let mut cargs = Vec::with_capacity(ca);
                for k in 0..ca {
                    cargs.push(remap_value(vmap, old, builder, ops[off + k]));
                }
                cases.push((c.value.clone(), resolve(c.target), cargs));
                off += ca;
            }
            builder.switch(cond, resolve(data.default), &dargs, cases);
        }
        _ => {}
    }
}

fn hoist(old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
    let n = old.block_count();
    let Some(entry) = old.entry() else {
        return Changed::No;
    };
    let entry_idx = entry.index();
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);

    let loops = find_loops(old, &cfg, &doms, entry_idx);
    if loops.is_empty() {
        return Changed::No;
    }

    // header block index → the loop it heads (unique).
    let mut header_to_loop = vec![None; n];
    for (li, lp) in loops.iter().enumerate() {
        header_to_loop[lp.header] = Some(li);
    }

    // For each block, the loops containing it, outermost (largest) first.
    let mut block_loops: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, slot) in block_loops.iter_mut().enumerate() {
        let mut v: Vec<usize> = (0..loops.len()).filter(|&li| loops[li].mask[b]).collect();
        v.sort_by(|&a, &c| {
            loops[c].size.cmp(&loops[a].size).then(loops[a].header.cmp(&loops[c].header))
        });
        *slot = v;
    }

    // Locate every instruction, its intra-block position, and whether it is a
    // hoistable-kind op that lives inside some loop.
    let mut inst_block = vec![usize::MAX; old.inst_count()];
    let mut intra_pos = vec![0usize; old.inst_count()];
    let mut hoistable = vec![false; old.inst_count()];
    for (bid, blk) in old.blocks() {
        let b = bid.index();
        for (pos, &i) in blk.insts().iter().enumerate() {
            inst_block[i.index()] = b;
            intra_pos[i.index()] = pos;
            if !block_loops[b].is_empty() && is_hoistable_kind(&old.inst(i).kind) {
                hoistable[i.index()] = true;
            }
        }
        if let Some(t) = blk.terminator() {
            inst_block[t.index()] = b;
        }
    }

    // Fixpoint: each hoistable instruction's destination is the outermost target
    // loop it is invariant relative to (or None). Destinations only move outward
    // as operands become hoistable, so the iteration converges monotonically.
    let mut destination: Vec<Option<usize>> = vec![None; old.inst_count()];
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..old.inst_count() {
            if !hoistable[i] {
                continue;
            }
            let b = inst_block[i];
            let mut nd = None;
            for &l in &block_loops[b] {
                if loops[l].target
                    && invariant_relative(old, &loops, &doms, &inst_block, &destination, i, l)
                {
                    nd = Some(l);
                    break;
                }
            }
            if destination[i] != nd {
                destination[i] = nd;
                changed = true;
            }
        }
    }

    // Nothing to hoist ⇒ keep the original body (no rebuild).
    if destination.iter().all(Option::is_none) {
        return Changed::No;
    }

    // Group hoisted instructions per destination loop, ordered so definitions
    // precede uses: dominator preorder of their blocks, then intra-block order.
    let order = dom_preorder(old, &doms);
    let mut dom_rank = vec![usize::MAX; n];
    for (r, &b) in order.iter().enumerate() {
        dom_rank[b] = r;
    }
    let mut hoist_in_loop: Vec<Vec<InstId>> = vec![Vec::new(); loops.len()];
    for (i, dest) in destination.iter().enumerate() {
        if let Some(l) = *dest {
            hoist_in_loop[l].push(InstId::from_index(i));
        }
    }
    for list in &mut hoist_in_loop {
        list.sort_by(|&x, &y| {
            let (bx, by) = (inst_block[x.index()], inst_block[y.index()]);
            dom_rank[bx]
                .cmp(&dom_rank[by])
                .then(intra_pos[x.index()].cmp(&intra_pos[y.index()]))
        });
    }

    // --- rebuild -----------------------------------------------------------
    // Create every original block first (entry from the signature, the rest with
    // their original parameter types), so branch targets exist before emission.
    let mut new_block: Vec<Option<BlockId>> = vec![None; n];
    new_block[entry_idx] = Some(builder.create_entry_block());
    for (b, slot) in new_block.iter_mut().enumerate() {
        if b == entry_idx {
            continue;
        }
        let bb = BlockId::from_index(b);
        let ptys: Vec<TypeId> =
            old.block(bb).params().iter().map(|&p| old.value_type(p)).collect();
        *slot = Some(builder.create_block(&ptys));
    }
    let new_block: Vec<BlockId> =
        new_block.into_iter().map(|x| x.expect("every block was created")).collect();

    // A preheader per loop we hoist into, carrying a copy of the header's params.
    let mut header_ph: Vec<Option<BlockId>> = vec![None; n];
    for (li, lp) in loops.iter().enumerate() {
        if hoist_in_loop[li].is_empty() {
            continue;
        }
        let hb = BlockId::from_index(lp.header);
        let ptys: Vec<TypeId> =
            old.block(hb).params().iter().map(|&p| old.value_type(p)).collect();
        header_ph[lp.header] = Some(builder.create_block(&ptys));
    }

    // Seed the value map with the rebuilt block parameters (preheader parameters
    // are forwarded directly and never referenced by old values).
    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for (b, &nb) in new_block.iter().enumerate() {
        let bb = BlockId::from_index(b);
        let new_params = builder.block_params(nb).to_vec();
        for (i, &op) in old.block(bb).params().iter().enumerate() {
            vmap[op.index()] = Some(new_params[i]);
        }
    }

    // Emit in dominator preorder, emitting each preheader just before its header.
    for &b in &order {
        if let Some(ph) = header_ph[b] {
            builder.switch_to(ph);
            let l = header_to_loop[b].expect("header heads a loop");
            for &i in &hoist_in_loop[l] {
                emit_inst(&mut vmap, old, builder, i);
            }
            let phparams = builder.block_params(ph).to_vec();
            builder.br(new_block[b], &phparams);
        }

        builder.switch_to(new_block[b]);
        for &i in old.block(BlockId::from_index(b)).insts() {
            if destination[i.index()].is_some() {
                continue; // hoisted into a preheader
            }
            emit_inst(&mut vmap, old, builder, i);
        }
        let resolve = |succ: BlockId| -> BlockId {
            let s = succ.index();
            if let Some(ph) = header_ph[s] {
                let l = header_to_loop[s].expect("header heads a loop");
                if !loops[l].mask[b] {
                    return ph; // out-of-loop predecessor ⇒ go through the preheader
                }
            }
            new_block[s]
        };
        emit_term(&mut vmap, old, builder, BlockId::from_index(b), resolve);
    }

    Changed::Yes
}

#[cfg(test)]
mod tests {
    use super::Licm;

    use std::fmt::Write as _;

    use crate::analysis::cfg::{ControlFlowGraph, Dominators};
    use crate::ir::inst::{BinOp, Flags, InstKind, IntPred};
    use crate::ir::{BlockId, FuncId, Function, InstId, Module};
    use crate::pass::Changed;
    use crate::support::StrInterner;
    use crate::transform::FunctionTransform;
    use crate::verify::verify_module;

    // --- helpers -----------------------------------------------------------

    fn run_licm(m: &mut Module, f: FuncId) -> Changed {
        let mut t = Licm;
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

    /// The block containing instruction `target`.
    fn block_of(f: &Function, target: InstId) -> BlockId {
        for (bid, blk) in f.blocks() {
            if blk.insts().contains(&target) || blk.terminator() == Some(target) {
                return bid;
            }
        }
        panic!("instruction not found in any block");
    }

    /// The first instruction matching `pred`, with its block.
    fn find(f: &Function, pred: impl Fn(&InstKind) -> bool) -> Option<(BlockId, InstId)> {
        for (bid, blk) in f.blocks() {
            for &i in blk.insts() {
                if pred(&f.inst(i).kind) {
                    return Some((bid, i));
                }
            }
        }
        None
    }

    fn is_mul(k: &InstKind) -> bool {
        matches!(k, InstKind::Bin(BinOp::Mul))
    }

    /// Every loop header (a block that is the target of a back edge).
    fn loop_headers(f: &Function, cfg: &ControlFlowGraph, doms: &Dominators) -> Vec<usize> {
        let mut hs = Vec::new();
        for b in 0..f.block_count() {
            if !doms.is_reachable(b) {
                continue;
            }
            if cfg.predecessors(b).iter().any(|&p| doms.is_reachable(p) && doms.dominates(b, p)) {
                hs.push(b);
            }
        }
        hs
    }

    fn canon(f: &Function) -> String {
        let mut s = String::new();
        for i in 0..f.inst_count() {
            let _ = writeln!(s, "I{i}: {:?}", f.inst(InstId::from_index(i)));
        }
        for (bid, b) in f.blocks() {
            let _ = writeln!(
                s,
                "B{}: p={:?} i={:?} t={:?}",
                bid.index(),
                b.params(),
                b.insts(),
                b.terminator()
            );
        }
        s
    }

    // --- builders ----------------------------------------------------------

    /// i32 f(i32 a, i32 b):
    ///   entry: br header(0)
    ///   header(iv): t = mul a,b; c = icmp slt iv,t; cond_br c, body, exit
    ///   body: iv2 = add iv,1; br header(iv2)
    ///   exit: ret iv
    /// `t` is invariant (a,b are parameters) and must hoist to a preheader.
    fn build_simple_hoist() -> (Module, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("licm-simple");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let a = b.param(entry, 0);
            let bb = b.param(entry, 1);
            let header = b.create_block(&[i32t]);
            let body = b.create_block(&[]);
            let exit = b.create_block(&[]);

            b.switch_to(entry);
            let zero = b.const_i64(i32t, 0);
            b.br(header, &[zero]);

            b.switch_to(header);
            let iv = b.param(header, 0);
            let t = b.mul(a, bb, Flags::NONE);
            let c = b.icmp(IntPred::Slt, iv, t);
            b.cond_br(c, body, &[], exit, &[]);

            b.switch_to(body);
            let one = b.const_i64(i32t, 1);
            let iv2 = b.add(iv, one, Flags::NONE);
            b.br(header, &[iv2]);

            b.switch_to(exit);
            b.ret(Some(iv));
        }
        (m, f)
    }

    // --- tests -------------------------------------------------------------

    #[test]
    fn hoists_loop_invariant_to_preheader() {
        let (mut m, f) = build_simple_hoist();
        assert!(verify_module(&m).is_ok());
        assert_eq!(count_kind(m.function(f), is_mul), 1);

        let c = run_licm(&mut m, f);
        assert_eq!(c, Changed::Yes);
        assert!(verify_module(&m).is_ok(), "licm output must verify");

        let func = m.function(f);
        assert_eq!(count_kind(func, is_mul), 1, "mul must not be duplicated");
        let (mul_block, _) = find(func, is_mul).expect("mul present");

        let cfg = ControlFlowGraph::new(func);
        let doms = Dominators::new(func, &cfg);
        let headers = loop_headers(func, &cfg, &doms);
        assert_eq!(headers.len(), 1, "one loop header");
        let header = headers[0];

        // The mul now lives in the preheader: a block that strictly dominates the
        // header (so it runs once before the loop), i.e. not the header itself.
        assert_ne!(mul_block.index(), header, "mul must leave the loop header");
        assert!(
            doms.dominates(mul_block.index(), header),
            "the hoisted mul must dominate the loop header (preheader)"
        );

        // The in-loop compare still consumes the hoisted value: its operand is
        // defined outside the loop (dominates the header, is not the header).
        let (_, cmp) = find(func, |k| matches!(k, InstKind::ICmp(_))).expect("icmp present");
        let uses_hoisted = func.inst(cmp).operands().iter().any(|&op| {
            if let crate::ir::value::ValueDef::Inst(di) = func.value(op).def {
                let db = block_of(func, di).index();
                db != header && doms.dominates(db, header)
            } else {
                false
            }
        });
        assert!(uses_hoisted, "the loop-body compare must read the hoisted definition");
    }

    #[test]
    fn idempotent_and_deterministic() {
        // Second run finds nothing new to hoist.
        let (mut m, f) = build_simple_hoist();
        assert_eq!(run_licm(&mut m, f), Changed::Yes);
        assert_eq!(run_licm(&mut m, f), Changed::No, "a second run must be a no-op");
        assert!(verify_module(&m).is_ok());

        // Determinism: identical inputs yield structurally identical outputs.
        let (mut m1, f1) = build_simple_hoist();
        let (mut m2, f2) = build_simple_hoist();
        run_licm(&mut m1, f1);
        run_licm(&mut m2, f2);
        assert_eq!(canon(m1.function(f1)), canon(m2.function(f2)));
    }

    #[test]
    fn does_not_hoist_loop_carried() {
        // x = add iv,1 depends on the loop-carried header parameter.
        let mut syms = StrInterner::new();
        let mut m = Module::new("licm-carried");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        let header;
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let nparam = b.param(entry, 0);
            header = b.create_block(&[i32t]);
            let body = b.create_block(&[]);
            let exit = b.create_block(&[]);

            b.switch_to(entry);
            let zero = b.const_i64(i32t, 0);
            b.br(header, &[zero]);

            b.switch_to(header);
            let iv = b.param(header, 0);
            let one = b.const_i64(i32t, 1);
            let x = b.add(iv, one, Flags::NONE);
            let c = b.icmp(IntPred::Slt, x, nparam);
            b.cond_br(c, body, &[], exit, &[]);

            b.switch_to(body);
            b.br(header, &[x]);

            b.switch_to(exit);
            b.ret(Some(iv));
        }
        let c = run_licm(&mut m, f);
        assert_eq!(c, Changed::No, "a loop-carried computation is not invariant");
        let func = m.function(f);
        let (add_block, _) =
            find(func, |k| matches!(k, InstKind::Bin(BinOp::Add))).expect("add present");
        assert_eq!(add_block, header, "the loop-carried add stays in the loop");
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn does_not_hoist_side_effects() {
        // load / store / call all have invariant operands but must not be hoisted.
        let mut syms = StrInterner::new();
        let mut m = Module::new("licm-effects");
        let i32t = m.types_mut().int(32);
        let ptr = m.types_mut().ptr();
        let void = m.types_mut().void();
        let g_sig = m.types_mut().func(vec![ptr], void, false);
        let g = m.declare_function(syms.intern("g"), g_sig);
        let sig = m.types_mut().func(vec![ptr, i32t], void, false);
        let f = m.declare_function(syms.intern("f"), sig);
        let header;
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let p = b.param(entry, 0);
            let nparam = b.param(entry, 1);
            header = b.create_block(&[i32t]);
            let body = b.create_block(&[]);
            let exit = b.create_block(&[]);

            b.switch_to(entry);
            let zero = b.const_i64(i32t, 0);
            b.br(header, &[zero]);

            b.switch_to(header);
            let iv = b.param(header, 0);
            let v = b.load(i32t, p, 4);
            b.store(i32t, p, v, 4);
            let gref = b.func_ref(g);
            b.call(gref, &[p], void);
            let c = b.icmp(IntPred::Slt, iv, nparam);
            b.cond_br(c, body, &[], exit, &[]);

            b.switch_to(body);
            let one = b.const_i64(i32t, 1);
            let iv2 = b.add(iv, one, Flags::NONE);
            b.br(header, &[iv2]);

            b.switch_to(exit);
            b.ret(None);
        }
        let c = run_licm(&mut m, f);
        assert_eq!(c, Changed::No, "effectful ops must not be hoisted");
        let func = m.function(f);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Load { .. })), 1);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Store { .. })), 1);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Call)), 1);
        let (load_b, _) = find(func, |k| matches!(k, InstKind::Load { .. })).unwrap();
        assert_eq!(load_b, header, "the load stays inside the loop");
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn does_not_hoist_div_by_possibly_zero() {
        // sdiv a,b has invariant operands but a possibly-zero divisor: unsafe to
        // speculate (division by zero is UB), so it must not be hoisted.
        let mut syms = StrInterner::new();
        let mut m = Module::new("licm-div");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        let header;
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let a = b.param(entry, 0);
            let bb = b.param(entry, 1);
            header = b.create_block(&[i32t]);
            let body = b.create_block(&[]);
            let exit = b.create_block(&[]);

            b.switch_to(entry);
            let zero = b.const_i64(i32t, 0);
            b.br(header, &[zero]);

            b.switch_to(header);
            let iv = b.param(header, 0);
            let q = b.bin(BinOp::SDiv, a, bb, Flags::NONE);
            let c = b.icmp(IntPred::Slt, iv, a);
            b.cond_br(c, body, &[], exit, &[]);

            b.switch_to(body);
            let one = b.const_i64(i32t, 1);
            let iv2 = b.add(iv, one, Flags::NONE);
            b.br(header, &[iv2]);

            b.switch_to(exit);
            b.ret(Some(q));
        }
        let c = run_licm(&mut m, f);
        assert_eq!(c, Changed::No, "integer div/rem is not speculatively hoisted");
        let func = m.function(f);
        let (div_b, _) = find(func, |k| matches!(k, InstKind::Bin(BinOp::SDiv))).unwrap();
        assert_eq!(div_b, header, "the sdiv stays inside the loop");
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn nested_hoists_only_out_of_inner_loop() {
        // t = mul i,i is invariant w.r.t. the inner loop (i is the OUTER header
        // parameter, defined outside the inner loop) but variant w.r.t. the outer
        // loop. It must be hoisted to the inner preheader — inside the outer loop.
        let mut syms = StrInterner::new();
        let mut m = Module::new("licm-nested");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let nparam = b.param(entry, 0);
            let h_o = b.create_block(&[i32t]); // outer header, param i
            let h_ob = b.create_block(&[]); // outer body (enters inner)
            let h_i = b.create_block(&[i32t]); // inner header, param j
            let h_ib = b.create_block(&[]); // inner body
            let latch = b.create_block(&[]); // outer latch
            let exit = b.create_block(&[]);

            b.switch_to(entry);
            let zero = b.const_i64(i32t, 0);
            b.br(h_o, &[zero]);

            b.switch_to(h_o);
            let i = b.param(h_o, 0);
            let co = b.icmp(IntPred::Slt, i, nparam);
            b.cond_br(co, h_ob, &[], exit, &[]);

            b.switch_to(h_ob);
            let zero2 = b.const_i64(i32t, 0);
            b.br(h_i, &[zero2]);

            b.switch_to(h_i);
            let j = b.param(h_i, 0);
            let t = b.mul(i, i, Flags::NONE); // invariant to inner, variant to outer
            let ci = b.icmp(IntPred::Slt, j, t);
            b.cond_br(ci, h_ib, &[], latch, &[]);

            b.switch_to(h_ib);
            let one = b.const_i64(i32t, 1);
            let j2 = b.add(j, one, Flags::NONE);
            b.br(h_i, &[j2]);

            b.switch_to(latch);
            let one2 = b.const_i64(i32t, 1);
            let i2 = b.add(i, one2, Flags::NONE);
            b.br(h_o, &[i2]);

            b.switch_to(exit);
            b.ret(Some(i));
        }
        let c = run_licm(&mut m, f);
        assert_eq!(c, Changed::Yes);
        assert!(verify_module(&m).is_ok());

        let func = m.function(f);
        assert_eq!(count_kind(func, is_mul), 1, "mul not duplicated");
        let (mul_block, _) = find(func, is_mul).expect("mul present");

        let cfg = ControlFlowGraph::new(func);
        let doms = Dominators::new(func, &cfg);
        let headers = loop_headers(func, &cfg, &doms);
        assert_eq!(headers.len(), 2, "two loop headers survive");
        let (outer, inner) = if doms.dominates(headers[0], headers[1]) {
            (headers[0], headers[1])
        } else {
            (headers[1], headers[0])
        };

        let mb = mul_block.index();
        // Hoisted out of the inner loop: dominates the inner header (preheader).
        assert!(doms.dominates(mb, inner), "mul must dominate the inner header");
        // Still inside the outer loop: dominated by the outer header...
        assert!(doms.dominates(outer, mb), "mul must stay inside the outer loop");
        // ...but NOT hoisted out of the outer loop (does not dominate its header).
        assert!(!doms.dominates(mb, outer), "mul must not be hoisted out of the outer loop");
    }
}
