//! **SCCP** — sparse conditional constant propagation, the *transform* half.
//!
//! Phase 3 gave us the constant-propagation *analysis* (the [`ConstLattice`]
//! domain on the one lattice engine, [`crate::analysis::solver`]); this pass is
//! the *rewrite* that turns that knowledge into a smaller, faster — and, by
//! construction, refining (tenet T3 / bet B2) — function. It does two things the
//! analysis proved:
//!
//! 1. **Materializes proven constants.** Any SSA value whose fixpoint result is
//!    [`ConstLattice::Const`] is replaced by that constant, and its uses are
//!    rewired to reference the literal rather than recomputing the instruction.
//!    The now-unused pure computation is simply not emitted (a stronger-than-DCE
//!    outcome that falls out of the functional rebuild); side-effecting ops are
//!    never dropped — and they can never be `Const` anyway, since the domain's
//!    transfer function maps `alloca`/`load`/`call` to ⊤.
//! 2. **Prunes infeasible edges and unreachable blocks.** Where the analysis
//!    proved a `cond_br`/`switch` condition constant, the taken edge is the only
//!    feasible one; the terminator is simplified to an unconditional `br` and any
//!    block left unreachable (per the solver's block reachability, which already
//!    accounts for edge feasibility) is dropped.
//!
//! ## Soundness around poison / UB (`docs/ir-design.md` §5)
//!
//! Only [`ConstLattice::Const`] values are folded. The domain's transfer
//! delegates to [`ir::fold`](crate::ir::fold), which returns ⊤ (never a `Const`)
//! for anything that *would be* undefined behavior (`FoldResult::WouldBeUb`, e.g.
//! `sdiv x, 0`) or that has no scalar-constant representation. So a value that
//! could be UB is ⊤ and is left computed — SCCP never folds UB away and never
//! materializes a ⊥ (unreachable) value. Folding a value the analysis proved to
//! be `poison` to a `poison` literal is sound (poison refines poison), and the
//! side-effect guard keeps observable behavior intact.
//!
//! ## Consuming the analysis under the rebuild's borrows
//!
//! A [`FunctionTransform`] runs inside
//! [`Module::map_function`](crate::ir::Module::map_function), where the
//! [`FunctionBuilder`] holds `&mut` of the shared type/constant tables for the
//! whole rebuild — so the read-only fixpoint (which needs `&TypeContext` /
//! `&ConstPool`) cannot be computed *inside* [`Sccp::run`]. Instead
//! [`Sccp::analyze`] runs the solver up front — on the very function that will be
//! rebuilt — and distills it into a borrow-free [`Plan`] that `run` consumes.
//! [`SccpPass`] wires this together as a module pass; the in-crate tests drive it
//! the same way.

use crate::analysis::domains::ConstLattice;
use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::analysis::solver::{FixpointResult, solve};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{InstData, InstKind};
use crate::ir::types::{TypeContext, TypeId};
use crate::ir::value::{Const, ConstPool, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Module};
use crate::pass::{Changed, ModulePass};
use crate::transform::{FunctionTransform, dom_preorder, rebuild_terminator, remap_value};

/// The SCCP constant-folding transform (see the module documentation).
///
/// It carries the [`Plan`] distilled from a prior [`Sccp::analyze`]; without one,
/// [`run`](FunctionTransform::run) is a no-op (there is nothing to consume).
#[derive(Debug, Default)]
pub struct Sccp {
    plan: Option<Plan>,
}

impl Sccp {
    /// A transform with no analysis yet; call [`Sccp::analyze`] before running it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run the constant-propagation fixpoint over `func` — which **must** be the
    /// same function this transform is subsequently handed as `old` — and store
    /// the distilled rewrite [`Plan`]. The solver needs read access to the shared
    /// interning tables, which the rebuild's builder holds mutably, so the
    /// analysis is performed here, before the rebuild borrows begin.
    pub fn analyze(&mut self, func: &Function, types: &TypeContext, consts: &ConstPool) {
        self.plan = Some(Plan::build(func, types, consts));
    }
}

impl FunctionTransform for Sccp {
    fn name(&self) -> &str {
        "sccp"
    }

    fn run(&mut self, old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
        let Some(plan) = &self.plan else {
            return Changed::No;
        };
        if !plan.changed {
            return Changed::No;
        }
        let Some(entry) = old.entry() else {
            return Changed::No;
        };
        rebuild(old, plan, builder, entry);
        Changed::Yes
    }
}

/// The SCCP transform as a module pass: for every function definition it runs the
/// analysis, rebuilds the body, and installs the result, reporting
/// [`Changed::Yes`] if any function changed (so the pass manager invalidates
/// cached analyses).
#[derive(Debug, Default, Clone, Copy)]
pub struct SccpPass;

impl ModulePass for SccpPass {
    fn name(&self) -> &str {
        "sccp"
    }

    fn run(&mut self, module: &mut Module) -> Changed {
        let mut changed = Changed::No;
        for i in 0..module.function_count() {
            let id = FuncId::from_index(i);
            if module.function(id).is_declaration() {
                continue;
            }
            let mut t = Sccp::new();
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
// The rewrite plan: a borrow-free distillation of the fixpoint result.
// ---------------------------------------------------------------------------

/// How a block's terminator is rewritten.
#[derive(Debug, Clone, Copy)]
enum TermChoice {
    /// Keep every edge (an unknown branch condition, or a non-branch terminator).
    KeepAll,
    /// The condition is a known constant: replace the terminator with an
    /// unconditional `br` along successor edge index `.0`.
    Single(usize),
}

/// Everything [`Sccp::run`] needs, extracted from the fixpoint so it survives the
/// rebuild without borrowing the interning tables.
#[derive(Debug)]
struct Plan {
    /// `value_const[v]` is `Some(c)` when value `v` is proven to equal the scalar
    /// constant `c` and folding it is worthwhile — i.e. `v` is a block parameter
    /// or a pure instruction result, not an already-literal constant or an opaque
    /// reference. Aggregates are never folded (they are ⊤).
    value_const: Vec<Option<Const>>,
    /// `reachable[b]` mirrors the solver's block reachability (which already
    /// prunes infeasible edges). Unreachable blocks are dropped.
    reachable: Vec<bool>,
    /// Per-block terminator rewrite.
    term_choice: Vec<TermChoice>,
    /// Whether applying this plan changes the function at all (drives `Changed`
    /// and keeps the pass idempotent).
    changed: bool,
}

impl Plan {
    fn build(func: &Function, types: &TypeContext, consts: &ConstPool) -> Plan {
        let res = solve::<ConstLattice>(func, types, consts);
        let nv = func.value_count();
        let nb = func.block_count();

        // Which values to fold to a literal.
        let mut value_const: Vec<Option<Const>> = vec![None; nv];
        for (i, slot) in value_const.iter_mut().enumerate() {
            let v = ValueId::from_index(i);
            if let Some(c) = res.value(v).as_const() {
                // Aggregates have no scalar-constant materialization; the domain
                // reports ⊤ for them, but guard anyway.
                if matches!(c, Const::Aggregate { .. }) {
                    continue;
                }
                match func.value(v).def {
                    // A pure instruction result: fold it and drop the compute.
                    // (Side-effecting ops are always ⊤, never reached here, but
                    // the guard keeps the invariant local and obvious.)
                    ValueDef::Inst(inst) => {
                        if !has_side_effect(&func.inst(inst).kind) {
                            *slot = Some(c.clone());
                        }
                    }
                    // A block parameter: rewire its uses to the literal.
                    ValueDef::Param(..) => *slot = Some(c.clone()),
                    // Already a literal constant / an opaque reference: nothing to
                    // do (a `Const` operand materializes itself on remap).
                    ValueDef::Const(_) | ValueDef::Global(_) | ValueDef::Func(_) => {}
                }
            }
        }

        // Block reachability (mirroring the solver, which already prunes
        // infeasible edges) and per-terminator pruning.
        let reachable: Vec<bool> =
            (0..nb).map(|b| res.is_reachable(BlockId::from_index(b))).collect();
        let term_choice: Vec<TermChoice> = (0..nb)
            .map(|b| {
                if reachable[b] {
                    terminator_choice(func, types, &res, BlockId::from_index(b))
                } else {
                    TermChoice::KeepAll
                }
            })
            .collect();

        // A change is any pruned edge, any dropped (unreachable) block, any folded
        // instruction, or any folded parameter that actually has uses to rewire.
        // Excluding use-less parameter folds is what makes the pass idempotent: a
        // parameter proven constant persists across the rebuild, but after its
        // uses are rewired there is nothing left to change on a second run.
        let pruned = term_choice.iter().any(|c| matches!(c, TermChoice::Single(_)));
        let dropped = reachable.iter().any(|&r| !r);
        let folded = value_const.iter().enumerate().any(|(i, vc)| {
            if vc.is_none() {
                return false;
            }
            let v = ValueId::from_index(i);
            matches!(func.value(v).def, ValueDef::Inst(_)) || !func.uses_of(v).is_empty()
        });
        let changed = pruned || dropped || folded;

        Plan { value_const, reachable, term_choice, changed }
    }
}

/// Decide how a reachable block's terminator is rewritten, given the fixpoint.
fn terminator_choice(
    func: &Function,
    types: &TypeContext,
    res: &FixpointResult<ConstLattice>,
    bb: BlockId,
) -> TermChoice {
    let Some(t) = func.block(bb).terminator() else {
        return TermChoice::KeepAll;
    };
    let term = func.inst(t);
    match &term.kind {
        InstKind::CondBr { .. } => {
            let cond = term.operands()[0];
            if let Some(Const::Int { value, .. }) = res.value(cond).as_const() {
                // The condition is an `i1`; its truth is "non-zero". Successor 0 is
                // `if_true`, successor 1 is `if_false`.
                let is_true = !value.mod_2k(1).is_zero();
                TermChoice::Single(if is_true { 0 } else { 1 })
            } else {
                TermChoice::KeepAll
            }
        }
        InstKind::Switch(data) => {
            let cond = term.operands()[0];
            if let Some(Const::Int { value, .. }) = res.value(cond).as_const()
                && let Some(width) = types.get(func.value_type(cond)).bit_width()
            {
                let cv = value.mod_2k(width);
                // Successor 0 is the default; case `i` is successor `i + 1`.
                let mut chosen = 0;
                for (i, case) in data.cases.iter().enumerate() {
                    if case.value.mod_2k(width) == cv {
                        chosen = i + 1;
                        break;
                    }
                }
                TermChoice::Single(chosen)
            } else {
                TermChoice::KeepAll
            }
        }
        _ => TermChoice::KeepAll,
    }
}

/// Whether an opcode has a side effect that forbids dropping it (mirrors DCE).
fn has_side_effect(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Alloca { .. }
            | InstKind::DynAlloca { .. }
            | InstKind::Store { .. }
            | InstKind::Call
    )
}

// ---------------------------------------------------------------------------
// The functional rebuild.
// ---------------------------------------------------------------------------

fn rebuild(old: &Function, plan: &Plan, builder: &mut FunctionBuilder<'_>, entry: BlockId) {
    let n = old.block_count();
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);

    // Create only the reachable blocks (entry first, then in index order). An
    // unreachable block keeps a `None` slot and is never emitted.
    let mut new_block: Vec<Option<BlockId>> = vec![None; n];
    new_block[entry.index()] = Some(builder.create_entry_block());
    for (b, slot) in new_block.iter_mut().enumerate() {
        if b == entry.index() || !plan.reachable[b] {
            continue;
        }
        let bb = BlockId::from_index(b);
        let ptys: Vec<TypeId> = old.block(bb).params().iter().map(|&p| old.value_type(p)).collect();
        *slot = Some(builder.create_block(&ptys));
    }
    let entry_new = new_block[entry.index()].expect("entry block was created");

    // Seed the value map from the rebuilt block parameters, folding any parameter
    // proven constant to its literal (its uses then reference the literal; the
    // now-unused parameter is left in place so edge arities stay consistent).
    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for (b, slot) in new_block.iter().enumerate() {
        let Some(nb) = *slot else {
            continue;
        };
        let bb = BlockId::from_index(b);
        let new_params = builder.block_params(nb).to_vec();
        for (i, &p) in old.block(bb).params().iter().enumerate() {
            if let Some(c) = &plan.value_const[p.index()] {
                vmap[p.index()] = Some(materialize(builder, c));
            } else {
                vmap[p.index()] = Some(new_params[i]);
            }
        }
    }

    // A total successor map for the shared terminator rebuilder: unreachable slots
    // get a harmless placeholder, which a kept (unpruned) terminator never targets
    // — a live edge always leads to a reachable, hence created, block.
    let succ_map: Vec<BlockId> = new_block.iter().map(|o| o.unwrap_or(entry_new)).collect();

    // Emit in dominator preorder so every surviving definition precedes its uses.
    for b in dom_preorder(old, &doms) {
        let Some(nb) = new_block[b] else {
            continue;
        };
        builder.switch_to(nb);
        let bb = BlockId::from_index(b);
        for &i in old.block(bb).insts() {
            let inst = old.inst(i);
            // A proven-constant result: fold it away, mapping its uses to the
            // literal, and do not emit the (pure) computation.
            if let Some(r) = inst.result()
                && let Some(c) = &plan.value_const[r.index()]
            {
                let cv = materialize(builder, c);
                vmap[r.index()] = Some(cv);
                continue;
            }
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
        match plan.term_choice[b] {
            TermChoice::Single(edge) => {
                emit_single_edge(&mut vmap, old, builder, &new_block, bb, edge);
            }
            TermChoice::KeepAll => {
                rebuild_terminator(&mut vmap, old, builder, &succ_map, bb, |_, _, _| {});
            }
        }
    }
}

/// Emit the pruned terminator of `bb` as an unconditional `br` along successor
/// edge `edge`, threading exactly that edge's block-argument list.
fn emit_single_edge(
    vmap: &mut [Option<ValueId>],
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    new_block: &[Option<BlockId>],
    bb: BlockId,
    edge: usize,
) {
    let t = old.block(bb).terminator().expect("reachable block is terminated");
    let term = old.inst(t);
    let succ = term.successors()[edge];
    let args = edge_args(term, edge);
    let mut mapped = Vec::with_capacity(args.len());
    for &o in args {
        mapped.push(remap_value(vmap, old, builder, o));
    }
    let target = new_block[succ.index()].expect("a feasible edge targets a reachable block");
    builder.br(target, &mapped);
}

/// The block-argument slice of successor edge `si` of `term`, matching the target
/// block's parameter list positionally (the layout the solver documents).
fn edge_args(term: &InstData, si: usize) -> &[ValueId] {
    let ops = term.operands();
    match &term.kind {
        InstKind::Br(_) => ops,
        InstKind::CondBr { true_args, false_args, .. } => {
            let ta = *true_args as usize;
            let fa = *false_args as usize;
            if si == 0 { &ops[1..1 + ta] } else { &ops[1 + ta..1 + ta + fa] }
        }
        InstKind::Switch(data) => {
            let da = data.default_args as usize;
            if si == 0 {
                &ops[1..1 + da]
            } else {
                let mut off = 1 + da;
                for c in &data.cases[..si - 1] {
                    off += c.args as usize;
                }
                let len = data.cases[si - 1].args as usize;
                &ops[off..off + len]
            }
        }
        _ => &[],
    }
}

/// Materialize an interned scalar constant as a value in the function being built.
fn materialize(builder: &mut FunctionBuilder<'_>, c: &Const) -> ValueId {
    match c {
        Const::Int { ty, value } => builder.const_int(*ty, value.clone()),
        Const::Float { ty, bits } => builder.const_float(*ty, *bits),
        Const::Null(ty) => builder.null(*ty),
        Const::Poison(ty) => builder.poison(*ty),
        // Aggregates are ⊤ in the domain and are never queued for folding.
        Const::Aggregate { .. } => unreachable!("aggregate constants are never folded"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::{Sccp, SccpPass};

    use crate::ir::inst::{BinOp, Flags, InstKind};
    use crate::ir::value::{Const, ValueDef};
    use crate::ir::{FuncId, Function, InstId, Module, ValueId};
    use crate::pass::{Changed, ModulePass};
    use crate::support::StrInterner;
    use crate::transform::FunctionTransform;
    use crate::verify::{check_refinement, RefinementResult, verify_module};

    use puremp::Int;

    /// Run SCCP over one function: analyze up front (the solver needs the shared
    /// tables), then rebuild through `map_function`, installing on change.
    fn run_sccp(m: &mut Module, f: FuncId) -> Changed {
        let mut t = Sccp::new();
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

    /// The value operand of whichever block ends in `ret <value>`.
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

    /// Assert the returned value is a materialized integer constant equal to
    /// `expected` (modulo `width`) — i.e. SCCP folded it to a literal.
    fn assert_ret_folds_to(m: &Module, f: FuncId, width: u32, expected: i64) {
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
            other => panic!("ret operand was not folded to a constant: {other:?}"),
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

    /// `f() -> i32 { ret 2 + 3 * 4 }` — a chain of constant arithmetic.
    fn build_const_chain() -> (Module, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("sccp-chain");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let three = b.const_i64(i32t, 3);
            let four = b.const_i64(i32t, 4);
            let m1 = b.mul(three, four, Flags::NONE);
            let two = b.const_i64(i32t, 2);
            let r = b.add(two, m1, Flags::NONE);
            b.ret(Some(r));
        }
        (m, f)
    }

    #[test]
    fn folds_constant_arithmetic_chain() {
        let (mut m, f) = build_const_chain();
        assert!(verify_module(&m).is_ok());
        assert_eq!(count_kind(m.function(f), |k| matches!(k, InstKind::Bin(_))), 2);

        let c = run_sccp(&mut m, f);
        assert_eq!(c, Changed::Yes);

        let func = m.function(f);
        assert_eq!(count_kind(func, |_| true), 0, "the whole chain folds away");
        assert!(verify_module(&m).is_ok(), "sccp output must verify");
        assert_ret_folds_to(&m, f, 32, 14);
    }

    #[test]
    fn does_not_fold_parameter_dependent_value() {
        // f(x) -> i32 { ret x + 5 } — `x` is unknown, so nothing folds.
        let mut syms = StrInterner::new();
        let mut m = Module::new("sccp-param");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let x = b.param(entry, 0);
            let five = b.const_i64(i32t, 5);
            let r = b.add(x, five, Flags::NONE);
            b.ret(Some(r));
        }
        let c = run_sccp(&mut m, f);
        assert_eq!(c, Changed::No, "a param-dependent value must not fold");
        let func = m.function(f);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Bin(BinOp::Add))), 1, "the add stays");
        assert!(matches!(func.value(ret_operand(func)).def, ValueDef::Inst(_)), "ret stays computed");
        assert!(verify_module(&m).is_ok());
    }

    /// `f() -> i32 { br cond ? then : else; then: ret 1+2; else: ret 20 }` with a
    /// constant-true condition: the false arm and its block are pruned.
    fn build_const_cond() -> (Module, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("sccp-cond");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let _entry = b.create_entry_block();
            let then_b = b.create_block(&[]);
            let els_b = b.create_block(&[]);
            let t = b.const_bool(true);
            b.cond_br(t, then_b, &[], els_b, &[]);
            b.switch_to(then_b);
            let one = b.const_i64(i32t, 1);
            let two = b.const_i64(i32t, 2);
            let r = b.add(one, two, Flags::NONE);
            b.ret(Some(r));
            b.switch_to(els_b);
            let twenty = b.const_i64(i32t, 20);
            b.ret(Some(twenty));
        }
        (m, f)
    }

    #[test]
    fn prunes_dead_branch_and_folds_live_path() {
        let (mut m, f) = build_const_cond();
        assert_eq!(m.function(f).block_count(), 3);
        assert!(verify_module(&m).is_ok());

        let c = run_sccp(&mut m, f);
        assert_eq!(c, Changed::Yes);

        let func = m.function(f);
        assert_eq!(func.block_count(), 2, "the dead `else` block is dropped");
        assert_eq!(
            count_kind(func, |k| matches!(k, InstKind::CondBr { .. })),
            0,
            "the constant cond_br is simplified to a br"
        );
        assert!(verify_module(&m).is_ok(), "sccp output must verify");
        assert_ret_folds_to(&m, f, 32, 3);
    }

    #[test]
    fn does_not_fold_ub_division() {
        // f() -> i32 { ret sdiv 4, 0 } — folding would be UB, so the analysis
        // yields Top and SCCP leaves the sdiv in place.
        let mut syms = StrInterner::new();
        let mut m = Module::new("sccp-ub");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let four = b.const_i64(i32t, 4);
            let zero = b.const_i64(i32t, 0);
            let d = b.bin(BinOp::SDiv, four, zero, Flags::NONE);
            b.ret(Some(d));
        }
        let c = run_sccp(&mut m, f);
        assert_eq!(c, Changed::No, "a would-be-UB division must not fold");
        let func = m.function(f);
        assert_eq!(
            count_kind(func, |k| matches!(k, InstKind::Bin(BinOp::SDiv))),
            1,
            "the sdiv survives"
        );
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn switch_on_constant_prunes_to_case() {
        // f() -> i32 { switch 2 [ 1: a, 2: b, default: d ] } with constant 2 folds
        // to the matching case, folding that case's arithmetic.
        let mut syms = StrInterner::new();
        let mut m = Module::new("sccp-switch");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let _entry = b.create_entry_block();
            let case1 = b.create_block(&[]);
            let case2 = b.create_block(&[]);
            let dflt = b.create_block(&[]);
            let two = b.const_i64(i32t, 2);
            b.switch(
                two,
                dflt,
                &[],
                vec![(Int::from_i64(1), case1, vec![]), (Int::from_i64(2), case2, vec![])],
            );
            b.switch_to(case1);
            let x = b.const_i64(i32t, 100);
            b.ret(Some(x));
            b.switch_to(case2);
            let ten = b.const_i64(i32t, 10);
            let five = b.const_i64(i32t, 5);
            let r = b.add(ten, five, Flags::NONE);
            b.ret(Some(r));
            b.switch_to(dflt);
            let z = b.const_i64(i32t, 0);
            b.ret(Some(z));
        }
        let c = run_sccp(&mut m, f);
        assert_eq!(c, Changed::Yes);
        let func = m.function(f);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Switch(_))), 0, "switch is pruned");
        assert_eq!(func.block_count(), 2, "only entry and the taken case remain");
        assert!(verify_module(&m).is_ok());
        assert_ret_folds_to(&m, f, 32, 15);
    }

    #[test]
    fn is_idempotent_and_deterministic() {
        // Idempotence: a second run finds nothing left to fold or prune.
        let (mut m, f) = build_const_cond();
        assert_eq!(run_sccp(&mut m, f), Changed::Yes);
        assert_eq!(run_sccp(&mut m, f), Changed::No, "second run must be a no-op");

        // Determinism: two independent runs produce identical bodies.
        let (mut a, fa) = build_const_chain();
        let (mut b, fb) = build_const_chain();
        run_sccp(&mut a, fa);
        run_sccp(&mut b, fb);
        assert_eq!(canon(a.function(fa)), canon(b.function(fb)));
    }

    #[test]
    fn runs_as_a_module_pass() {
        let (mut m, f) = build_const_chain();
        let c = SccpPass.run(&mut m);
        assert_eq!(c, Changed::Yes);
        assert!(verify_module(&m).is_ok());
        assert_ret_folds_to(&m, f, 32, 14);
        // A second module-pass run is a fixpoint.
        assert_eq!(SccpPass.run(&mut m), Changed::No);
    }

    #[test]
    fn folded_function_refines_original_b2() {
        // B2 spot-check: the folded, single-block pure-integer body must refine
        // the original. We rebuild without installing, so both functions coexist.
        let (mut m, f) = build_const_chain();
        let mut t = Sccp::new();
        t.analyze(m.function(f), m.types(), m.consts());
        let (fresh, c) = m.map_function(f, |old, b| t.run(old, b));
        assert_eq!(c, Changed::Yes);

        match check_refinement(m.types(), m.consts(), m.function(f), &fresh) {
            RefinementResult::Refines => {}
            // A budget-exhausted / unavailable solver is a sound non-answer.
            RefinementResult::Unknown(_) => {}
            RefinementResult::Counterexample(model) => {
                panic!("sccp folding is not a refinement: {model}");
            }
        }
    }
}

