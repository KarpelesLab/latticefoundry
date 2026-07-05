//! The transform (optimizer) layer — ROADMAP Phase 4.
//!
//! A **transform** rewrites one function into a semantically-refining one (tenet
//! T3 / bet B2: a transform's output must *refine* its input, and every output
//! must pass the structural verifier). Following tenet T5 (the IR is
//! id/arena-based and immutable-friendly) the transforms here are written as
//! **functional rebuilds**: a pass reads the old [`Function`] and constructs a
//! fresh one via a [`FunctionBuilder`], applying the transformation as it goes,
//! rather than performing in-place surgery on the arena. [`Module::map_function`]
//! is the primitive that makes this cheap and borrow-clean.
//!
//! Two foundational structural passes are built on this substrate:
//!
//! - [`Mem2Reg`] — promote `alloca` slots that are never address-taken into SSA
//!   values, placing **block parameters** at join points via iterated dominance
//!   frontiers (our block-argument analog of φ-placement, `docs/ir-design.md`
//!   §2) and renaming loads to their reaching definitions.
//! - [`Dce`] — dead-code elimination: drop side-effect-free instructions whose
//!   results are unused, iterated to a fixpoint.
//!
//! A [`FunctionTransform`] is adapted to the module-level [`ModulePass`] pipeline
//! (and its analysis-invalidation) by [`FunctionTransformPass`].

pub mod dce;
pub mod egraph;
pub mod inline;
pub mod licm;
pub mod mem2reg;
pub mod sccp;
pub mod simplify_cfg;
pub mod superopt;

#[cfg(test)]
mod tests;

pub use dce::Dce;
pub use egraph::EqSat;
pub use inline::Inline;
pub use licm::Licm;
pub use mem2reg::Mem2Reg;
pub use sccp::Sccp;
pub use simplify_cfg::SimplifyCfg;

use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::InstKind;
use crate::ir::value::{ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Module};
use crate::pass::{Changed, ModulePass};

/// A transformation over a single function, expressed as a functional rebuild.
///
/// [`run`](FunctionTransform::run) reads `old` and reconstructs an equivalent
/// (refining) body into `builder`, a [`FunctionBuilder`] over a fresh function
/// that shares the module's interning tables. It returns whether it changed
/// anything; on [`Changed::No`] the freshly built body is discarded and the
/// original retained (so an inapplicable transform costs only its analysis).
pub trait FunctionTransform {
    /// A short, stable name used in pass pipelines and diagnostics.
    fn name(&self) -> &str;

    /// Rebuild `old` into `builder`, reporting whether the body changed.
    fn run(&mut self, old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed;
}

/// Adapts a [`FunctionTransform`] into a module-level [`ModulePass`]: it runs the
/// transform over every function definition, installing each rebuilt body, and
/// reports [`Changed::Yes`] if any function changed (so the
/// [`PassManager`](crate::pass::PassManager) invalidates cached analyses).
#[derive(Debug, Default)]
pub struct FunctionTransformPass<T> {
    transform: T,
}

impl<T> FunctionTransformPass<T> {
    /// Wrap a function transform as a module pass.
    pub fn new(transform: T) -> Self {
        Self { transform }
    }
}

impl<T: FunctionTransform> ModulePass for FunctionTransformPass<T> {
    fn name(&self) -> &str {
        self.transform.name()
    }

    fn run(&mut self, module: &mut Module) -> Changed {
        let mut changed = Changed::No;
        for i in 0..module.function_count() {
            let id = FuncId::from_index(i);
            if module.function(id).is_declaration() {
                continue;
            }
            let (fresh, c) = module.map_function(id, |old, b| self.transform.run(old, b));
            if c == Changed::Yes {
                module.replace_function(id, fresh);
                changed = Changed::Yes;
            }
        }
        changed
    }
}

// ---------------------------------------------------------------------------
// Shared rebuild helpers used by the structural transforms.
// ---------------------------------------------------------------------------

/// Map an *old*-function value to its *new*-function equivalent, materializing
/// constants / global / function references on demand and caching the result in
/// `vmap` (indexed by old [`ValueId`]).
///
/// Block parameters and instruction results must already be present in `vmap`
/// (the caller populates parameters up front and instruction results as it emits
/// them, in an order where every definition precedes its uses). The only case
/// where an instruction result or parameter is *not* yet mapped is inside
/// unreachable code, whose values never execute — such a stray operand is mapped
/// to a `poison` of the right type, which the verifier accepts (it does not
/// check dominance in unreachable blocks) and which is a sound refinement.
pub(crate) fn remap_value(
    vmap: &mut [Option<ValueId>],
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    v: ValueId,
) -> ValueId {
    if let Some(nv) = vmap[v.index()] {
        return nv;
    }
    let nv = match &old.value(v).def {
        ValueDef::Const(c) => builder.use_const(*c),
        ValueDef::Global(g) => builder.global_ref(*g),
        ValueDef::Func(f) => builder.func_ref(*f),
        ValueDef::Param(..) | ValueDef::Inst(..) => builder.poison(old.value_type(v)),
    };
    vmap[v.index()] = Some(nv);
    nv
}

/// Rebuild the terminator of old block `bb` into the current insertion block,
/// remapping its operands through `vmap` and mapping successor blocks through
/// `new_block`. For each outgoing edge, `extra` is invoked to append any
/// additional block arguments (e.g. mem2reg's reaching definitions); it receives
/// the builder, the *old* successor id, and the edge's argument list to extend.
pub(crate) fn rebuild_terminator(
    vmap: &mut [Option<ValueId>],
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    new_block: &[BlockId],
    bb: BlockId,
    mut extra: impl FnMut(&mut FunctionBuilder<'_>, BlockId, &mut Vec<ValueId>),
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
            let target = *target;
            let mut args = Vec::with_capacity(ops.len());
            for &o in ops {
                args.push(remap_value(vmap, old, builder, o));
            }
            extra(builder, target, &mut args);
            builder.br(new_block[target.index()], &args);
        }
        InstKind::CondBr { if_true, if_false, true_args, false_args } => {
            let (if_true, if_false) = (*if_true, *if_false);
            let ta = *true_args as usize;
            let fa = *false_args as usize;
            let cond = remap_value(vmap, old, builder, ops[0]);
            let mut targs = Vec::with_capacity(ta);
            for k in 0..ta {
                targs.push(remap_value(vmap, old, builder, ops[1 + k]));
            }
            extra(builder, if_true, &mut targs);
            let mut fargs = Vec::with_capacity(fa);
            for k in 0..fa {
                fargs.push(remap_value(vmap, old, builder, ops[1 + ta + k]));
            }
            extra(builder, if_false, &mut fargs);
            builder.cond_br(
                cond,
                new_block[if_true.index()],
                &targs,
                new_block[if_false.index()],
                &fargs,
            );
        }
        InstKind::Switch(data) => {
            let cond = remap_value(vmap, old, builder, ops[0]);
            let da = data.default_args as usize;
            let default = data.default;
            let mut dargs = Vec::with_capacity(da);
            for k in 0..da {
                dargs.push(remap_value(vmap, old, builder, ops[1 + k]));
            }
            extra(builder, default, &mut dargs);
            let mut cases = Vec::with_capacity(data.cases.len());
            let mut off = 1 + da;
            for c in &data.cases {
                let ca = c.args as usize;
                let mut cargs = Vec::with_capacity(ca);
                for k in 0..ca {
                    cargs.push(remap_value(vmap, old, builder, ops[off + k]));
                }
                extra(builder, c.target, &mut cargs);
                cases.push((c.value.clone(), new_block[c.target.index()], cargs));
                off += ca;
            }
            builder.switch(cond, new_block[default.index()], &dargs, cases);
        }
        // A non-terminator in the terminator slot is a malformed input the
        // verifier rejects; nothing to rebuild.
        _ => {}
    }
}

/// A dominator-tree preorder of the reachable blocks (parents before children),
/// followed by the unreachable blocks in index order. Emitting blocks in this
/// order guarantees every SSA definition is rebuilt before its uses in
/// well-formed reachable code, since a dominator precedes the blocks it
/// dominates.
pub(crate) fn dom_preorder(
    old: &Function,
    doms: &crate::analysis::cfg::Dominators,
) -> Vec<usize> {
    let n = old.block_count();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let entry = old.entry();
    for b in 0..n {
        if Some(BlockId::from_index(b)) == entry {
            continue;
        }
        if doms.is_reachable(b)
            && let Some(ip) = doms.idom(b)
        {
            children[ip].push(b);
        }
    }
    for c in &mut children {
        c.sort_unstable();
    }
    let mut order = Vec::with_capacity(n);
    if let Some(e) = entry {
        let mut stack = vec![e.index()];
        while let Some(b) = stack.pop() {
            order.push(b);
            for &c in children[b].iter().rev() {
                stack.push(c);
            }
        }
    }
    for b in 0..n {
        if !doms.is_reachable(b) {
            order.push(b);
        }
    }
    order
}
