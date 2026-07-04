//! **Inline** — interprocedural function inlining (ROADMAP Phase 4).
//!
//! At a call site `%r = call @g(a0, a1, ...)` where `g` is a small, non-recursive
//! *defined* function, this pass replaces the call with a copy of `g`'s body,
//! wiring the call's arguments to `g`'s entry parameters and the call's result to
//! `g`'s returns. Because this IR uses **block arguments, not φ-nodes**
//! (`docs/ir-design.md` §2), the splice is uniform and merge-clean:
//!
//! - `g`'s blocks are copied into the caller as fresh blocks, every value and
//!   block id remapped, preserving their parameter lists and per-edge argument
//!   lists faithfully.
//! - The call **splits** its block. Instructions after the call move into a fresh
//!   *continuation* block that takes the call's result as its **one block
//!   parameter** (none for a `void` call). The split point branches to the copy of
//!   `g`'s entry block, passing the call arguments as that block's parameters.
//! - Each `ret v` in the inlined body becomes `br continuation(v)` (and a bare
//!   `ret` becomes `br continuation()`), so `g`'s multiple returns *merge* through
//!   the continuation's block parameter — the phi-free merge this IR is built
//!   around. An `unreachable` stays `unreachable`.
//!
//! Poison/UB semantics are preserved exactly: every instruction is copied
//! verbatim (same opcode, flags, operands after remapping), the argument→parameter
//! and return→parameter data flow is an identity substitution, and no operation is
//! reordered across an effect — so the result is a refinement of the original
//! (tenet T3 / bet B2).
//!
//! ## Candidate selection and the cost model
//!
//! A call is inlined only when its callee is a *known* function reference
//! ([`ValueDef::Func`], so indirect/opaque callees are skipped), is a **definition**
//! (has a body), is **not the caller itself** (the direct-recursion guard), is not
//! **variadic** (fixed-parameter mapping only), and is **small enough** — its
//! instruction count (terminators included) is at most [`Inline::threshold`].
//!
//! ## Termination
//!
//! One [`Inline::run`] performs a **single level** of inlining: callee bodies are
//! read from the module as it was at the start of the run (via
//! [`Module::map_function_reading`]), so a call that a callee *contains* is copied
//! into the caller and only becomes an inlining candidate on a *later* run. The
//! direct-recursion guard makes even that bounded: a mutually recursive cycle
//! `f → g → f` collapses to a *direct* self-call after one round (inlining `g`
//! into `f` drags `g`'s call-to-`f` in), which the guard then refuses — so the
//! pass always terminates.

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{InstData, InstKind};
use crate::ir::types::{Type, TypeId};
use crate::ir::value::{ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Module};
use crate::pass::{Changed, ModulePass};
use crate::transform::{dom_preorder, rebuild_terminator, remap_value};

/// Default callee size ceiling (instructions, terminators included) for inlining.
pub const DEFAULT_THRESHOLD: usize = 16;

/// The function-inlining pass (see the module documentation). It is a
/// [`ModulePass`] because inlining is interprocedural: rebuilding a caller reads
/// its callees' bodies.
#[derive(Debug, Clone, Copy)]
pub struct Inline {
    /// Callees larger than this many instructions are left alone.
    threshold: usize,
}

impl Default for Inline {
    fn default() -> Self {
        Self { threshold: DEFAULT_THRESHOLD }
    }
}

impl Inline {
    /// An inliner with the [default threshold](DEFAULT_THRESHOLD).
    pub fn new() -> Self {
        Self::default()
    }

    /// An inliner with a custom callee-size threshold.
    pub fn with_threshold(threshold: usize) -> Self {
        Self { threshold }
    }

    /// The current callee-size threshold.
    pub fn threshold(&self) -> usize {
        self.threshold
    }
}

impl ModulePass for Inline {
    fn name(&self) -> &str {
        "inline"
    }

    fn run(&mut self, module: &mut Module) -> Changed {
        let mut changed = Changed::No;
        for i in 0..module.function_count() {
            let id = FuncId::from_index(i);
            if module.function(id).is_declaration() {
                continue;
            }
            // Decide which calls to inline against the *original* module, so the
            // whole run is one level of inlining (see the module docs).
            let decisions = self.plan(module, id);
            if decisions.iter().all(Option::is_none) {
                continue;
            }
            let (fresh, _) = module.map_function_reading(id, |caller, funcs, b| {
                rebuild(caller, funcs, &decisions, b);
            });
            module.replace_function(id, fresh);
            changed = Changed::Yes;
        }
        changed
    }
}

impl Inline {
    /// The inlining decision for every instruction of `caller`: `decisions[i]` is
    /// `Some(callee)` when instruction `i` is a `call` to be inlined. Only calls in
    /// reachable blocks are considered.
    fn plan(&self, module: &Module, caller_id: FuncId) -> Vec<Option<FuncId>> {
        let caller = module.function(caller_id);
        let mut decisions = vec![None; caller.inst_count()];
        let cfg = ControlFlowGraph::new(caller);
        let doms = Dominators::new(caller, &cfg);
        for (bid, blk) in caller.blocks() {
            if !doms.is_reachable(bid.index()) {
                continue;
            }
            for &i in blk.insts() {
                let inst = caller.inst(i);
                if matches!(inst.kind, InstKind::Call)
                    && let Some(callee_id) = self.inlinable_callee(module, caller_id, inst)
                {
                    decisions[i.index()] = Some(callee_id);
                }
            }
        }
        decisions
    }

    /// The callee to inline for `call`, or `None` if the call is not a suitable
    /// candidate (indirect, recursive, external, variadic, or too large).
    fn inlinable_callee(
        &self,
        module: &Module,
        caller_id: FuncId,
        call: &InstData,
    ) -> Option<FuncId> {
        // Operand 0 is the callee reference; it must name a known function.
        let callee_ref = *call.operands().first()?;
        let callee_id = match &module.function(caller_id).value(callee_ref).def {
            ValueDef::Func(f) => *f,
            _ => return None,
        };
        // Direct-recursion guard: never inline a function into itself.
        if callee_id == caller_id {
            return None;
        }
        let callee = module.function(callee_id);
        if callee.is_declaration() {
            return None; // an external declaration has no body to inline
        }
        // Only fixed-arity callees: variadic argument passing has no faithful
        // parameter mapping.
        match module.types().get(callee.sig) {
            Type::Func(ft) if !ft.variadic => {}
            _ => return None,
        }
        // Cost model: small enough by instruction count.
        if callee_size(callee) > self.threshold {
            return None;
        }
        Some(callee_id)
    }
}

/// A callee's size for the cost model: its instruction count with terminators.
fn callee_size(callee: &Function) -> usize {
    let mut n = 0;
    for (_bid, blk) in callee.blocks() {
        n += blk.insts().len();
        if blk.terminator().is_some() {
            n += 1;
        }
    }
    n
}

/// Rebuild `caller`, splicing each selected callee body in place of its call.
fn rebuild(
    caller: &Function,
    funcs: &[Function],
    decisions: &[Option<FuncId>],
    builder: &mut FunctionBuilder<'_>,
) {
    let n = caller.block_count();
    let entry = caller.entry().expect("a definition has an entry block");
    let cfg = ControlFlowGraph::new(caller);
    let doms = Dominators::new(caller, &cfg);

    // Create the *head* block of every caller block up front. Incoming edges
    // target these, so they keep the caller block's original parameter list; a
    // block that a call splits keeps emitting into freshly created continuation
    // blocks from here.
    let mut new_head: Vec<Option<BlockId>> = vec![None; n];
    new_head[entry.index()] = Some(builder.create_entry_block());
    for (b, slot) in new_head.iter_mut().enumerate() {
        if b == entry.index() {
            continue;
        }
        let bb = BlockId::from_index(b);
        let ptys: Vec<TypeId> =
            caller.block(bb).params().iter().map(|&p| caller.value_type(p)).collect();
        *slot = Some(builder.create_block(&ptys));
    }
    let new_head: Vec<BlockId> =
        new_head.into_iter().map(|x| x.expect("every head was created")).collect();

    // Seed the caller value map from the rebuilt head parameters.
    let mut vmap: Vec<Option<ValueId>> = vec![None; caller.value_count()];
    for (b, &nb) in new_head.iter().enumerate() {
        let bb = BlockId::from_index(b);
        let new_params = builder.block_params(nb).to_vec();
        for (i, &p) in caller.block(bb).params().iter().enumerate() {
            vmap[p.index()] = Some(new_params[i]);
        }
    }

    // Emit in dominator preorder so every surviving definition precedes its uses.
    for b in dom_preorder(caller, &doms) {
        let bb = BlockId::from_index(b);
        builder.switch_to(new_head[b]);
        for &i in caller.block(bb).insts() {
            if let Some(callee_id) = decisions[i.index()] {
                let call = caller.inst(i);
                // The continuation takes the call's result as its lone parameter.
                let cont_params: Vec<TypeId> = call.result().map(|_| call.ty).into_iter().collect();
                let cont = builder.create_block(&cont_params);
                if let Some(r) = call.result() {
                    vmap[r.index()] = Some(builder.block_params(cont)[0]);
                }
                // Map the call arguments (operands after the callee reference).
                let mut args = Vec::with_capacity(call.operands().len().saturating_sub(1));
                for &a in &call.operands()[1..] {
                    args.push(remap_value(&mut vmap, caller, builder, a));
                }
                splice_callee(&funcs[callee_id.index()], builder, &args, cont);
                // Continue emitting this caller block's tail into the continuation.
                builder.switch_to(cont);
            } else {
                copy_generic(&mut vmap, caller, builder, caller.inst(i));
            }
        }
        rebuild_terminator(&mut vmap, caller, builder, &new_head, bb, |_, _, _| {});
    }
}

/// Splice `callee`'s body into the function under construction: branch from the
/// current block to a copy of `callee`'s entry (passing `args`), and route every
/// `ret` to `cont` (carrying the returned value, if any).
fn splice_callee(
    callee: &Function,
    builder: &mut FunctionBuilder<'_>,
    args: &[ValueId],
    cont: BlockId,
) {
    let cn = callee.block_count();
    let entry = callee.entry().expect("an inlined callee is a definition");

    // Fresh copy of every callee block, preserving parameter lists.
    let mut callee_new: Vec<BlockId> = Vec::with_capacity(cn);
    for cb in 0..cn {
        let bb = BlockId::from_index(cb);
        let ptys: Vec<TypeId> =
            callee.block(bb).params().iter().map(|&p| callee.value_type(p)).collect();
        callee_new.push(builder.create_block(&ptys));
    }

    // Enter the inlined body: the caller arguments become the entry parameters.
    builder.br(callee_new[entry.index()], args);

    // Seed the callee value map from the copied block parameters.
    let mut cmap: Vec<Option<ValueId>> = vec![None; callee.value_count()];
    for (cb, &nb) in callee_new.iter().enumerate() {
        let bb = BlockId::from_index(cb);
        let new_params = builder.block_params(nb).to_vec();
        for (i, &p) in callee.block(bb).params().iter().enumerate() {
            cmap[p.index()] = Some(new_params[i]);
        }
    }

    // Copy callee blocks in dominator preorder; `ret` becomes `br cont(...)`.
    let cfg = ControlFlowGraph::new(callee);
    let doms = Dominators::new(callee, &cfg);
    for cb in dom_preorder(callee, &doms) {
        let bb = BlockId::from_index(cb);
        builder.switch_to(callee_new[cb]);
        for &ci in callee.block(bb).insts() {
            copy_generic(&mut cmap, callee, builder, callee.inst(ci));
        }
        rebuild_callee_terminator(&mut cmap, callee, builder, &callee_new, bb, cont);
    }
}

/// Rebuild a callee block's terminator during a splice: `ret` merges into `cont`
/// via its block argument; every other terminator is copied faithfully (its
/// successors remapped through the callee's fresh blocks).
fn rebuild_callee_terminator(
    cmap: &mut [Option<ValueId>],
    callee: &Function,
    builder: &mut FunctionBuilder<'_>,
    callee_new: &[BlockId],
    bb: BlockId,
    cont: BlockId,
) {
    let Some(t) = callee.block(bb).terminator() else {
        return;
    };
    let term = callee.inst(t);
    if matches!(term.kind, InstKind::Ret) {
        let mut cargs = Vec::new();
        if let Some(&v) = term.operands().first() {
            cargs.push(remap_value(cmap, callee, builder, v));
        }
        builder.br(cont, &cargs);
    } else {
        // Br / CondBr / Switch / Unreachable: the shared rebuilder handles these,
        // reading the callee's structure and mapping successors to their copies.
        rebuild_terminator(cmap, callee, builder, callee_new, bb, |_, _, _| {});
    }
}

/// Copy an instruction verbatim with remapped operands, recording its result.
fn copy_generic(
    vmap: &mut [Option<ValueId>],
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    inst: &InstData,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::Inline;

    use crate::analysis::domains::ConstLattice;
    use crate::analysis::solver::solve;
    use crate::ir::inst::{BinOp, Flags, InstKind, IntPred};
    use crate::ir::value::{Const, ValueDef};
    use crate::ir::{FuncId, Function, InstId, Module, ValueId};
    use crate::pass::{Changed, ModulePass};
    use crate::support::StrInterner;
    use crate::verify::verify_module;

    use puremp::Int;

    /// Count instructions of a whole function matching `pred` (terminators too).
    fn count_kind(f: &Function, pred: impl Fn(&InstKind) -> bool) -> usize {
        let mut c = 0;
        for (_bid, blk) in f.blocks() {
            for &i in blk.insts() {
                if pred(&f.inst(i).kind) {
                    c += 1;
                }
            }
            if let Some(t) = blk.terminator()
                && pred(&f.inst(t).kind)
            {
                c += 1;
            }
        }
        c
    }

    fn n_calls(f: &Function) -> usize {
        count_kind(f, |k| matches!(k, InstKind::Call))
    }

    /// The abstract constant of the operand of whichever block returns a value,
    /// per the constant-propagation analysis (the end-to-end value check).
    fn ret_value_const(m: &Module, f: FuncId) -> ConstLattice {
        let func = m.function(f);
        let r = solve::<ConstLattice>(func, m.types(), m.consts());
        for (_bid, blk) in func.blocks() {
            if let Some(t) = blk.terminator()
                && matches!(func.inst(t).kind, InstKind::Ret)
                && let Some(&v) = func.inst(t).operands().first()
            {
                return r.value(v).clone();
            }
        }
        panic!("no value-returning ret found");
    }

    fn assert_ret_int(m: &Module, f: FuncId, width: u32, expected: i64) {
        match ret_value_const(m, f) {
            ConstLattice::Const(Const::Int { value, .. }) => assert_eq!(
                value.mod_2k(width),
                Int::from_i64(expected).mod_2k(width),
                "returned constant mismatch"
            ),
            other => panic!("expected constant int {expected}, got {other:?}"),
        }
    }

    /// A structural fingerprint of a function, for determinism checks.
    fn canon(f: &Function) -> String {
        let mut s = String::new();
        for i in 0..f.inst_count() {
            let _ = writeln!(s, "I{i}: {:?}", f.inst(InstId::from_index(i)));
        }
        for (bid, b) in f.blocks() {
            let _ = writeln!(
                s,
                "B{}: params={:?} insts={:?} term={:?}",
                bid.index(),
                b.params(),
                b.insts(),
                b.terminator()
            );
        }
        for v in 0..f.value_count() {
            let val = f.value(ValueId::from_index(v));
            let _ = writeln!(s, "V{v}: {:?} : {:?}", val.def, val.ty);
        }
        s
    }

    /// `g(a, b) = a + b`, and `f(x, y) = g(x, y)`.
    fn build_leaf_caller() -> (Module, FuncId, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("inline-leaf");
        let i32t = m.types_mut().int(32);
        let g_sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
        let g = m.declare_function(syms.intern("g"), g_sig);
        {
            let mut b = m.build(g);
            let e = b.create_entry_block();
            let a = b.param(e, 0);
            let bb = b.param(e, 1);
            let r = b.add(a, bb, Flags::NONE);
            b.ret(Some(r));
        }
        let f_sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), f_sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let x = b.param(e, 0);
            let y = b.param(e, 1);
            let gref = b.func_ref(g);
            let r = b.call(gref, &[x, y], i32t).expect("g returns i32");
            b.ret(Some(r));
        }
        (m, f, g)
    }

    #[test]
    fn inlines_leaf_callee() {
        let (mut m, f, _g) = build_leaf_caller();
        assert!(verify_module(&m).is_ok());
        assert_eq!(n_calls(m.function(f)), 1);

        let c = Inline::new().run(&mut m);
        assert_eq!(c, Changed::Yes);

        let func = m.function(f);
        assert_eq!(n_calls(func), 0, "the call must be gone");
        assert_eq!(
            count_kind(func, |k| matches!(k, InstKind::Bin(BinOp::Add))),
            1,
            "the add is spliced in"
        );
        assert!(verify_module(&m).is_ok(), "inline output must verify");
    }

    /// `g(a, b, c) = if c { a } else { b }` — two blocks, two `ret`s (a diamond).
    /// `f(x, y, c) = g(x, y, c)`.
    fn build_diamond_caller() -> (Module, FuncId, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("inline-diamond");
        let i1 = m.types_mut().bool();
        let i32t = m.types_mut().int(32);
        let g_sig = m.types_mut().func(vec![i32t, i32t, i1], i32t, false);
        let g = m.declare_function(syms.intern("g"), g_sig);
        {
            let mut b = m.build(g);
            let e = b.create_entry_block();
            let then_b = b.create_block(&[]);
            let els_b = b.create_block(&[]);
            let a = b.param(e, 0);
            let bb = b.param(e, 1);
            let cnd = b.param(e, 2);
            b.switch_to(e);
            b.cond_br(cnd, then_b, &[], els_b, &[]);
            b.switch_to(then_b);
            b.ret(Some(a));
            b.switch_to(els_b);
            b.ret(Some(bb));
        }
        let f_sig = m.types_mut().func(vec![i32t, i32t, i1], i32t, false);
        let f = m.declare_function(syms.intern("f"), f_sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let x = b.param(e, 0);
            let y = b.param(e, 1);
            let cnd = b.param(e, 2);
            let gref = b.func_ref(g);
            let r = b.call(gref, &[x, y, cnd], i32t).expect("g returns i32");
            b.ret(Some(r));
        }
        (m, f, g)
    }

    #[test]
    fn inlines_multi_block_two_returns_merge_via_continuation() {
        let (mut m, f, _g) = build_diamond_caller();
        assert!(verify_module(&m).is_ok());

        let c = Inline::new().run(&mut m);
        assert_eq!(c, Changed::Yes);

        let func = m.function(f);
        assert_eq!(n_calls(func), 0, "the call is inlined");
        assert_eq!(
            count_kind(func, |k| matches!(k, InstKind::CondBr { .. })),
            1,
            "the callee's cond_br is spliced in"
        );
        // The two `ret`s merged into one continuation return of a block parameter.
        let mut ret_op = None;
        for (_bid, blk) in func.blocks() {
            if let Some(t) = blk.terminator()
                && matches!(func.inst(t).kind, InstKind::Ret)
            {
                ret_op = Some(func.inst(t).operands()[0]);
            }
        }
        let ret_op = ret_op.expect("a value-returning ret");
        assert!(
            matches!(func.value(ret_op).def, ValueDef::Param(..)),
            "the merged return is a continuation block parameter"
        );
        assert!(verify_module(&m).is_ok(), "diamond inline output must verify");
    }

    #[test]
    fn does_not_inline_callee_above_threshold() {
        // A callee with several instructions, inlined only if the threshold allows.
        let mut syms = StrInterner::new();
        let mut m = Module::new("inline-big");
        let i32t = m.types_mut().int(32);
        let g_sig = m.types_mut().func(vec![i32t], i32t, false);
        let g = m.declare_function(syms.intern("g"), g_sig);
        {
            let mut b = m.build(g);
            let e = b.create_entry_block();
            let mut acc = b.param(e, 0);
            for k in 0..8 {
                let c = b.const_i64(i32t, k);
                acc = b.add(acc, c, Flags::NONE);
            }
            b.ret(Some(acc));
        }
        let f_sig = m.types_mut().func(vec![i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), f_sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let x = b.param(e, 0);
            let gref = b.func_ref(g);
            let r = b.call(gref, &[x], i32t).expect("g returns i32");
            b.ret(Some(r));
        }

        // Threshold 4 is well below the callee's size (8 adds + ret): no inlining.
        let c = Inline::with_threshold(4).run(&mut m);
        assert_eq!(c, Changed::No, "a callee above threshold must not inline");
        assert_eq!(n_calls(m.function(f)), 1, "the call survives");
        assert!(verify_module(&m).is_ok());

        // A generous threshold inlines it.
        let c = Inline::with_threshold(64).run(&mut m);
        assert_eq!(c, Changed::Yes);
        assert_eq!(n_calls(m.function(f)), 0);
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn does_not_inline_direct_recursion() {
        // f(n) = { if n<=0 { ret 0 } else { ret f(n) } } — a direct self-call.
        let mut syms = StrInterner::new();
        let mut m = Module::new("inline-rec");
        let i32t = m.types_mut().int(32);
        let f_sig = m.types_mut().func(vec![i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), f_sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let base = b.create_block(&[]);
            let rec = b.create_block(&[]);
            let n = b.param(e, 0);
            let zero = b.const_i64(i32t, 0);
            let cond = b.icmp(IntPred::Sle, n, zero);
            b.cond_br(cond, base, &[], rec, &[]);
            b.switch_to(base);
            let z = b.const_i64(i32t, 0);
            b.ret(Some(z));
            b.switch_to(rec);
            let fref = b.func_ref(f);
            let r = b.call(fref, &[n], i32t).expect("f returns i32");
            b.ret(Some(r));
        }
        let c = Inline::new().run(&mut m);
        assert_eq!(c, Changed::No, "a directly-recursive call must not inline");
        assert_eq!(n_calls(m.function(f)), 1, "the self-call is left alone (terminates)");
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn does_not_inline_indirect_call() {
        // f(p, x) = call *p(x) — an indirect call through a pointer parameter.
        let mut syms = StrInterner::new();
        let mut m = Module::new("inline-indirect");
        let i32t = m.types_mut().int(32);
        let ptr = m.types_mut().ptr();
        let f_sig = m.types_mut().func(vec![ptr, i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), f_sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let p = b.param(e, 0);
            let x = b.param(e, 1);
            let r = b.call(p, &[x], i32t).expect("returns i32");
            b.ret(Some(r));
        }
        let c = Inline::new().run(&mut m);
        assert_eq!(c, Changed::No, "an indirect callee is not a known function");
        assert_eq!(n_calls(m.function(f)), 1);
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn end_to_end_inlined_result_folds_to_constant() {
        // g(a, b) = a + b; caller() = g(2, 3). After inlining, the constant
        // analysis must prove the caller's return equals 5.
        let mut syms = StrInterner::new();
        let mut m = Module::new("inline-const");
        let i32t = m.types_mut().int(32);
        let g_sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
        let g = m.declare_function(syms.intern("g"), g_sig);
        {
            let mut b = m.build(g);
            let e = b.create_entry_block();
            let a = b.param(e, 0);
            let bb = b.param(e, 1);
            let r = b.add(a, bb, Flags::NONE);
            b.ret(Some(r));
        }
        let f_sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("caller"), f_sig);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let two = b.const_i64(i32t, 2);
            let three = b.const_i64(i32t, 3);
            let gref = b.func_ref(g);
            let r = b.call(gref, &[two, three], i32t).expect("g returns i32");
            b.ret(Some(r));
        }

        // Before inlining the return is opaque (a call result): not a constant.
        assert!(ret_value_const(&m, f).is_top(), "call result is unknown pre-inline");

        let c = Inline::new().run(&mut m);
        assert_eq!(c, Changed::Yes);
        assert!(verify_module(&m).is_ok(), "inline output must verify");
        // The analysis threads 2 and 3 through the spliced add and the merge
        // block parameter, solving the return to 5.
        assert_ret_int(&m, f, 32, 5);
    }

    #[test]
    fn is_deterministic() {
        let (mut a, fa, _) = build_diamond_caller();
        let (mut b, fb, _) = build_diamond_caller();
        Inline::new().run(&mut a);
        Inline::new().run(&mut b);
        assert_eq!(canon(a.function(fa)), canon(b.function(fb)));
    }

    #[test]
    fn second_run_is_a_no_op_on_a_leaf() {
        // After inlining the only call, a second run finds nothing to do.
        let (mut m, _f, _g) = build_leaf_caller();
        assert_eq!(Inline::new().run(&mut m), Changed::Yes);
        assert_eq!(Inline::new().run(&mut m), Changed::No, "no calls left to inline");
    }
}
