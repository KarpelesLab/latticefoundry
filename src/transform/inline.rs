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
//!   parameter** (none for a `void` call). `g`'s entry block (which nothing
//!   branches to) is copied straight into the split block, with the call
//!   arguments substituted for its parameters, so an argument reaches its uses
//!   directly rather than through a block argument.
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
//! ## Candidate selection
//!
//! A call is a candidate only when its callee is a *known* function reference
//! ([`ValueDef::Func`], so indirect/opaque callees are skipped), is a
//! **definition** (has a body), is **not the caller itself** (the
//! direct-recursion guard), is not **variadic** (fixed-parameter mapping only),
//! is not **weak** (another object may replace a weak body at link time), and
//! does not use **`dyn_alloca`** (its storage lives until the function returns;
//! an inlined copy would hold it until the *caller* returns, and grow the stack
//! on every iteration of a loop around the call).
//!
//! ## The cost model
//!
//! Sizes are measured in the **code-size dimension of the B9 cost lattice**
//! (`docs/design-tenets.md`): [`inst_size`] gives each instruction a cost in
//! abstract units of roughly one machine instruction — `alloca`, `freeze`,
//! `bitcast` and `declassify` are free, plain arithmetic, compares, loads and
//! stores cost 1, a division 3, and a `call` its whole call sequence
//! ([`call_cost`]). The e-graph's extraction cost (`own_cost` in
//! [`egraph`](crate::transform::egraph)) is the latency-flavoured sibling of the
//! same lattice. A candidate is then inlined when the first of these rules
//! that applies says so:
//!
//! 1. **Hints** ([`InlineHint`], `docs/ir-design.md` §4b): `inline(never)` is
//!    never inlined; `inline(always)` always is (still never into itself).
//! 2. **Stack guard.** A callee whose `alloca`s exceed [`STACK_LIMIT`] bytes is
//!    not inlined without a hint (its frame would join the caller's, which may
//!    be recursive).
//! 3. **Free inlining.** The callee's *inlined size* ([`inlined_size`]: its body
//!    without the entry parameters, with every `ret` after the first turned
//!    into a branch) is at most the [`call_cost`] of the call it replaces —
//!    argument setup, the call, receiving the result, the caller-saved
//!    registers the call clobbers, and the address of each caller slot passed
//!    as an out-parameter (which mem2reg then promotes, issue #13). Inlining
//!    such a callee does not grow the code, so it is not budgeted.
//! 4. **Single call site.** An internal function whose only reference in the
//!    whole module is this call (no taken address, no global initializer, no
//!    same-named global) is inlined up to [`SINGLE_SITE_LIMIT`] units, as long
//!    as the caller stays within [`MAX_CALLER_SIZE`]: dead-function elimination
//!    then deletes the original, so the module does not grow.
//! 5. **Budgeted.** Otherwise a callee of at most [`Inline::threshold`] units
//!    is inlined while its growth (inlined size minus call cost) fits both the
//!    **caller's budget** (`max(CALLER_BUDGET_MIN, caller size / 2)` per run)
//!    and the **module's budget** (`max(MODULE_BUDGET_MIN, module size / 16)`
//!    per run). The budgets go to the smallest growth first, ties broken in
//!    caller and instruction order, so the outcome is deterministic (tenet T5).
//!
//! **Constants at the call site.** When some arguments are constants and the
//! callee is not already free to inline (and has at most [`CONST_FOLD_LIMIT`]
//! units), its inlined size is re-estimated with
//! the one abstract-interpretation engine (bet B8): the constant lattice is
//! solved over the callee with those parameters seeded to the constants
//! ([`solve_with`]), and instructions that fold to a constant, branches on a
//! constant, and blocks that become unreachable cost nothing.
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
//! pass always terminates, hints or not, and a recursive function is never
//! inlined into itself.

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::analysis::domains::ConstLattice;
use crate::analysis::solver::{FixpointResult, SolveHooks, solve_with};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, CastOp, InstData, InstId, InstKind};
use crate::ir::types::{Type, TypeId};
use crate::ir::value::{AddrTarget, Const, ConstId, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, GlobalId, InlineHint, Linkage, Module};
use crate::pass::{Changed, ModulePass};
use crate::transform::{dom_preorder, rebuild_terminator, remap_value};

/// Default size limit (code-size units, see [`inst_size`]) of a callee inlined
/// against the budgets (rule 5 of the module documentation).
pub const DEFAULT_THRESHOLD: usize = 32;

/// The largest single-call-site internal callee inlined (rule 4), in units.
pub const SINGLE_SITE_LIMIT: u32 = 400;

/// The size (units) up to which a caller may grow through single-call-site
/// inlining (rule 4); it bounds the compile time of very large functions.
pub const MAX_CALLER_SIZE: u32 = 4000;

/// The minimum per-run growth budget of one caller (rule 5), in units.
pub const CALLER_BUDGET_MIN: u32 = 64;

/// The minimum per-run growth budget of the whole module (rule 5), in units.
pub const MODULE_BUDGET_MIN: u32 = 256;

/// The most `alloca` storage (bytes) a callee may bring into its caller without
/// an `inline(always)` hint (rule 2).
pub const STACK_LIMIT: u64 = 512;

/// The largest callee (units) whose size is re-estimated with the call's
/// constant arguments; the estimate runs the constant lattice over the callee.
pub const CONST_FOLD_LIMIT: u32 = 200;

/// Cost of the `call` instruction itself.
const CALL_INSN: u32 = 1;
/// Cost of the caller-saved registers a call clobbers: values live across it
/// are spilled or kept in callee-saved registers the prologue must save.
const CALL_CLOBBER: u32 = 2;
/// Extra cost of an argument that is the address of a caller `alloca`: the
/// address computation, and the reload of the slot after the call.
const OUT_SLOT: u32 = 2;

/// The function-inlining pass (see the module documentation). It is a
/// [`ModulePass`] because inlining is interprocedural: rebuilding a caller reads
/// its callees' bodies.
#[derive(Debug, Clone, Copy)]
pub struct Inline {
    /// Callees larger than this many units are only inlined when free, hinted
    /// or single-call-site.
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

    /// An inliner with a custom budgeted-callee size limit (in [`inst_size`]
    /// units).
    pub fn with_threshold(threshold: usize) -> Self {
        Self { threshold }
    }

    /// The current budgeted-callee size limit.
    pub fn threshold(&self) -> usize {
        self.threshold
    }
}

impl ModulePass for Inline {
    fn name(&self) -> &str {
        "inline"
    }

    fn run(&mut self, module: &mut Module) -> Changed {
        let facts = ModuleFacts::new(module);
        let n = module.function_count();
        // Decide which calls to inline against the *original* module, so the
        // whole run is one level of inlining (see the module docs).
        let mut decisions: Vec<Vec<Option<FuncId>>> = vec![Vec::new(); n];
        let mut caller_left = vec![0u32; n];
        let mut budgeted: Vec<Budgeted> = Vec::new();
        for (i, f) in module.functions().enumerate() {
            if f.is_declaration() {
                continue;
            }
            decisions[i] = self.plan(module, &facts, FuncId::from_index(i), &mut budgeted);
            caller_left[i] = CALLER_BUDGET_MIN.max(function_size(f) / 2);
        }
        // Rule 5: hand the budgets to the smallest growth first (ties in
        // caller and instruction order, so the outcome is deterministic).
        budgeted.sort_by_key(|c| (c.growth, c.caller.index(), c.call.index()));
        let mut module_left = MODULE_BUDGET_MIN.max(facts.module_size / 16);
        for c in budgeted {
            let left = &mut caller_left[c.caller.index()];
            if c.growth <= *left && c.growth <= module_left {
                *left -= c.growth;
                module_left -= c.growth;
                decisions[c.caller.index()][c.call.index()] = Some(c.callee);
            }
        }
        // Rebuild every caller against the unmodified module, and install the
        // results only afterwards: a callee spliced into a caller is always its
        // body from the start of the run, never one this run already grew.
        let mut fresh = Vec::new();
        for (i, decisions) in decisions.iter().enumerate() {
            if decisions.iter().all(Option::is_none) {
                continue;
            }
            let id = FuncId::from_index(i);
            let (f, _) = module.map_function_reading(id, |caller, funcs, b| {
                rebuild(caller, funcs, decisions, b);
            });
            fresh.push((id, f));
        }
        let changed = if fresh.is_empty() { Changed::No } else { Changed::Yes };
        for (id, f) in fresh {
            module.replace_function(id, f);
        }
        changed
    }
}

/// A call that rule 5 (the budgeted rule) would inline, waiting for budget.
struct Budgeted {
    /// The code growth inlining it costs (inlined size minus call cost).
    growth: u32,
    caller: FuncId,
    call: InstId,
    callee: FuncId,
}

/// Per-function facts the cost model needs, computed once per run from the
/// module as it was at the start of the run.
struct ModuleFacts {
    /// [`inlined_size`] of every definition (0 for declarations).
    size: Vec<u32>,
    /// [`function_size`] of every function, summed.
    module_size: u32,
    /// Bytes of `alloca` storage in each function.
    alloca_bytes: Vec<u64>,
    /// Whether the function uses `dyn_alloca`.
    dyn_alloca: Vec<bool>,
    /// Whether the function is internal and referenced exactly once in the
    /// module, by the callee operand of a call.
    single_site: Vec<bool>,
}

impl ModuleFacts {
    fn new(module: &Module) -> ModuleFacts {
        let n = module.function_count();
        let mut facts = ModuleFacts {
            size: vec![0; n],
            module_size: 0,
            alloca_bytes: vec![0; n],
            dyn_alloca: vec![false; n],
            single_site: vec![false; n],
        };
        // References: `calls[g]` counts call-operand uses of `g`, `other[g]`
        // every other reference (a taken address, an address constant).
        let mut calls = vec![0u32; n];
        let mut other = vec![false; n];
        for (fi, f) in module.functions().enumerate() {
            if f.is_declaration() {
                continue;
            }
            facts.size[fi] = inlined_size(f, None);
            facts.module_size = facts.module_size.saturating_add(function_size(f));
            for (_, blk) in f.blocks() {
                for &i in blk.insts() {
                    match f.inst(i).kind {
                        InstKind::Alloca { elem_ty } => {
                            facts.alloca_bytes[fi] += module.types().size_of(elem_ty);
                        }
                        InstKind::DynAlloca { .. } => facts.dyn_alloca[fi] = true,
                        _ => {}
                    }
                }
            }
            for v in 0..f.value_count() {
                let v = ValueId::from_index(v);
                match f.value(v).def {
                    ValueDef::Func(g) => {
                        for u in f.uses_of(v) {
                            if u.operand == 0 && matches!(f.inst(u.inst).kind, InstKind::Call) {
                                calls[g.index()] += 1;
                            } else {
                                other[g.index()] = true;
                            }
                        }
                    }
                    ValueDef::Const(c) if !f.uses_of(v).is_empty() => {
                        mark_const_funcs(module, c, &mut other);
                    }
                    _ => {}
                }
            }
        }
        for g in 0..module.global_count() {
            for f in module.global_referenced_functions(GlobalId::from_index(g)) {
                other[f.index()] = true;
            }
        }
        let global_names: std::collections::HashSet<_> = module.globals().map(|g| g.name).collect();
        for (fi, f) in module.functions().enumerate() {
            facts.single_site[fi] = f.attrs.linkage == Linkage::Internal
                && !f.is_declaration()
                && calls[fi] == 1
                && !other[fi]
                && !global_names.contains(&f.name);
        }
        facts
    }
}

/// Mark every function an (aggregate) constant addresses.
fn mark_const_funcs(module: &Module, c: ConstId, out: &mut [bool]) {
    let mut stack = vec![c];
    while let Some(c) = stack.pop() {
        match module.consts().get(c) {
            Const::Addr { target: AddrTarget::Func(f), .. } => out[f.index()] = true,
            Const::Aggregate { elems, .. } => stack.extend(elems.iter().copied()),
            _ => {}
        }
    }
}

impl Inline {
    /// The inlining decision for every instruction of `caller` under rules
    /// 1–4: `decisions[i]` is `Some(callee)` when instruction `i` is a `call` to
    /// be inlined. A call only rule 5 would inline is pushed onto `budgeted`
    /// instead. Only calls in reachable blocks are considered.
    fn plan(
        &self,
        module: &Module,
        facts: &ModuleFacts,
        caller_id: FuncId,
        budgeted: &mut Vec<Budgeted>,
    ) -> Vec<Option<FuncId>> {
        let caller = module.function(caller_id);
        let mut decisions = vec![None; caller.inst_count()];
        let cfg = ControlFlowGraph::new(caller);
        let doms = Dominators::new(caller, &cfg);
        let caller_size = function_size(caller);
        let mut grown = 0u32;
        for (bid, blk) in caller.blocks() {
            if !doms.is_reachable(bid.index()) {
                continue;
            }
            for &i in blk.insts() {
                let inst = caller.inst(i);
                if !matches!(inst.kind, InstKind::Call) {
                    continue;
                }
                let Some(callee_id) = inlinable_callee(module, facts, caller_id, inst) else {
                    continue;
                };
                let callee = module.function(callee_id);
                let take = match callee.attrs.inline {
                    InlineHint::Never => false,
                    InlineHint::Always => true,
                    InlineHint::Auto => {
                        let cc = call_cost(caller, inst);
                        let mut size = facts.size[callee_id.index()];
                        if size > cc && size <= CONST_FOLD_LIMIT {
                            size = size.min(const_folded_size(module, caller, inst, callee));
                        }
                        let growth = size.saturating_sub(cc);
                        if facts.alloca_bytes[callee_id.index()] > STACK_LIMIT {
                            false // rule 2
                        } else if growth == 0 {
                            true // rule 3: does not grow the code
                        } else if facts.single_site[callee_id.index()]
                            && size <= SINGLE_SITE_LIMIT
                            && caller_size.saturating_add(grown).saturating_add(growth) <= MAX_CALLER_SIZE
                        {
                            // Rule 4: the original is deleted afterwards.
                            grown += growth;
                            true
                        } else {
                            if size <= self.threshold as u32 {
                                budgeted.push(Budgeted { growth, caller: caller_id, call: i, callee: callee_id });
                            }
                            false
                        }
                    }
                };
                if take {
                    decisions[i.index()] = Some(callee_id);
                }
            }
        }
        decisions
    }
}

/// The callee to consider for `call`, or `None` if the call can never be
/// inlined (indirect, recursive, external, variadic, weak, or using
/// `dyn_alloca`).
fn inlinable_callee(module: &Module, facts: &ModuleFacts, caller_id: FuncId, call: &InstData) -> Option<FuncId> {
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
    // A weak definition may be replaced by another object's at link time.
    if callee.attrs.linkage == Linkage::Weak {
        return None;
    }
    if facts.dyn_alloca[callee_id.index()] {
        return None;
    }
    // Only fixed-arity callees: variadic argument passing has no faithful
    // parameter mapping.
    match module.types().get(callee.sig) {
        Type::Func(ft) if !ft.variadic => {}
        _ => return None,
    }
    Some(callee_id)
}

// ---------------------------------------------------------------------------
// The code-size cost (the B9 lattice's code-size dimension)
// ---------------------------------------------------------------------------

/// The code-size cost of one instruction, in abstract units of roughly one
/// machine instruction (see the module documentation). Terminators included;
/// a `call` costs its whole call sequence ([`call_cost`] without the
/// out-parameter term, which depends on the caller).
pub fn inst_size(inst: &InstData) -> u32 {
    match &inst.kind {
        InstKind::Alloca { .. }
        | InstKind::Freeze
        | InstKind::Declassify
        | InstKind::AsmOutput(_)
        | InstKind::Cast(CastOp::Bitcast) => 0,
        InstKind::Bin(BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem) => 3,
        InstKind::Select | InstKind::CondBr { .. } | InstKind::AtomicRmw { .. } | InstKind::ShuffleVector(_) => 2,
        InstKind::CmpXchg { .. } | InstKind::Reduce(_) => 3,
        InstKind::DynAlloca { .. } | InstKind::InlineAsm(_) => 4,
        InstKind::Call => call_sequence(inst),
        InstKind::Syscall => 1 + inst.operands().len() as u32,
        InstKind::Switch(sw) => 2 + sw.cases.len() as u32,
        _ => 1,
    }
}

/// The cost of a call sequence without out-parameter slots: the call, one
/// move per argument, receiving the result, and the clobbered registers.
fn call_sequence(call: &InstData) -> u32 {
    let args = call.operands().len().saturating_sub(1) as u32;
    CALL_INSN + args + u32::from(call.result().is_some()) + CALL_CLOBBER
}

/// What inlining the call `call` of `caller` saves at the call site: its call
/// sequence ([`inst_size`]) plus, for each argument that is the address of one
/// of the caller's `alloca` slots (an out-parameter), the address computation
/// and the reload that promoting the slot removes.
pub fn call_cost(caller: &Function, call: &InstData) -> u32 {
    let slots = call.operands()[1..]
        .iter()
        .filter(|&&a| match caller.value(a).def {
            ValueDef::Inst(i) => matches!(caller.inst(i).kind, InstKind::Alloca { .. }),
            _ => false,
        })
        .count() as u32;
    call_sequence(call) + slots * OUT_SLOT
}

/// The code size of a whole function: [`inst_size`] summed over every
/// instruction and terminator.
pub fn function_size(f: &Function) -> u32 {
    let mut n = 0u32;
    for (_, blk) in f.blocks() {
        for &i in blk.insts() {
            n = n.saturating_add(inst_size(f.inst(i)));
        }
        if let Some(t) = blk.terminator() {
            n = n.saturating_add(inst_size(f.inst(t)));
        }
    }
    n
}

/// The size of `callee`'s body once inlined: [`function_size`] where the first
/// `ret` is free (it becomes the fall-through into the continuation) and each
/// further one a branch. With `consts` (a constant-lattice solution seeded with
/// the call's constant arguments), instructions that fold to a constant,
/// branches on a constant and unreachable blocks are free too.
pub fn inlined_size(callee: &Function, consts: Option<&FixpointResult<ConstLattice>>) -> u32 {
    let is_const = |v: ValueId| consts.is_some_and(|r| matches!(r.value(v), ConstLattice::Const(_)));
    let mut n = 0u32;
    let mut rets = 0u32;
    for (bid, blk) in callee.blocks() {
        if consts.is_some_and(|r| !r.is_reachable(bid)) {
            continue;
        }
        for &i in blk.insts() {
            let inst = callee.inst(i);
            let folds = !inst.kind.has_side_effect() && inst.result().is_some_and(is_const);
            if !folds {
                n = n.saturating_add(inst_size(inst));
            }
        }
        let Some(t) = blk.terminator() else { continue };
        let term = callee.inst(t);
        n = n.saturating_add(match term.kind {
            InstKind::Ret => {
                rets += 1;
                u32::from(rets > 1)
            }
            InstKind::CondBr { .. } | InstKind::Switch(_) if is_const(term.operands()[0]) => 0,
            _ => inst_size(term),
        });
    }
    n
}

/// Seeds the callee's entry parameters with the call's constant arguments.
struct ArgSeeds {
    entry: BlockId,
    args: Vec<Option<ConstLattice>>,
}

impl SolveHooks<ConstLattice> for ArgSeeds {
    fn seed(&self, _v: ValueId, def: &ValueDef) -> Option<ConstLattice> {
        match def {
            ValueDef::Param(b, i) if *b == self.entry => self.args.get(*i as usize).cloned().flatten(),
            _ => None,
        }
    }
}

/// [`inlined_size`] of `callee` at the call `call` of `caller`, folding what
/// the call's constant arguments decide. Without a scalar constant argument
/// this is the plain inlined size.
fn const_folded_size(module: &Module, caller: &Function, call: &InstData, callee: &Function) -> u32 {
    let args: Vec<Option<ConstLattice>> = call.operands()[1..]
        .iter()
        .map(|&a| match caller.value(a).def {
            ValueDef::Const(c) => match module.consts().get(c) {
                Const::Aggregate { .. } | Const::Addr { .. } => None,
                k => Some(ConstLattice::Const(k.clone())),
            },
            _ => None,
        })
        .collect();
    let Some(entry) = callee.entry() else { return 0 };
    if args.iter().all(Option::is_none) {
        return inlined_size(callee, None);
    }
    let hooks = ArgSeeds { entry, args };
    let r = solve_with::<ConstLattice>(callee, module.types(), module.consts(), &hooks);
    inlined_size(callee, Some(&r))
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
            // Caller code keeps its lines; the call-entry glue (argument
            // coercion) takes the call's line.
            builder.set_line_from(caller, i);
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

/// Splice `callee`'s body into the function under construction: its entry block
/// continues the current block with `args` for its parameters, and every
/// `ret` branches to `cont` (carrying the returned value, if any).
fn splice_callee(
    callee: &Function,
    builder: &mut FunctionBuilder<'_>,
    args: &[ValueId],
    cont: BlockId,
) {
    let cn = callee.block_count();
    let entry = callee.entry().expect("an inlined callee is a definition");
    let mut cmap: Vec<Option<ValueId>> = vec![None; callee.value_count()];

    // The callee's entry block has no predecessors (the verifier forbids a
    // branch to an entry block), so its body continues the caller's current
    // block, and its parameters *are* the call arguments: no block argument
    // stands between an argument and its uses. That keeps a caller slot passed
    // as an out-parameter a direct load/store address, which mem2reg can then
    // promote (issue #13). A call may pass a `ptr` for an aggregate parameter
    // (the struct-by-value convention makes them interchangeable at a call);
    // inside the body the parameter is strictly typed, so such an argument is
    // bitcast to the parameter's type.
    let here = builder.current_block().expect("splicing at an insertion point");
    for (&p, &a) in callee.block(entry).params().iter().zip(args) {
        cmap[p.index()] = Some(coerce_address(builder, a, callee.value_type(p)));
    }

    // Fresh copy of every other callee block, preserving parameter lists.
    let mut callee_new: Vec<BlockId> = Vec::with_capacity(cn);
    for cb in 0..cn {
        let bb = BlockId::from_index(cb);
        if bb == entry {
            callee_new.push(here);
            continue;
        }
        let ptys: Vec<TypeId> =
            callee.block(bb).params().iter().map(|&p| callee.value_type(p)).collect();
        let nb = builder.create_block(&ptys);
        let new_params = builder.block_params(nb).to_vec();
        for (i, &p) in callee.block(bb).params().iter().enumerate() {
            cmap[p.index()] = Some(new_params[i]);
        }
        callee_new.push(nb);
    }

    // Copy callee blocks in dominator preorder; `ret` becomes `br cont(...)`.
    let cfg = ControlFlowGraph::new(callee);
    let doms = Dominators::new(callee, &cfg);
    for cb in dom_preorder(callee, &doms) {
        let bb = BlockId::from_index(cb);
        builder.switch_to(callee_new[cb]);
        for &ci in callee.block(bb).insts() {
            // Inlined instructions keep the callee's own lines.
            builder.set_line_from(callee, ci);
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
    builder.set_line_from(callee, t);
    let term = callee.inst(t);
    if matches!(term.kind, InstKind::Ret) {
        let mut cargs = Vec::new();
        if let Some(&v) = term.operands().first() {
            let v = remap_value(cmap, callee, builder, v);
            // A `ret` of a `ptr` from an aggregate-returning function (the
            // address of the struct value) meets the continuation's aggregate
            // parameter.
            let want = builder.value_type(builder.block_params(cont)[0]);
            cargs.push(coerce_address(builder, v, want));
        }
        builder.br(cont, &cargs);
    } else {
        // Br / CondBr / Switch / Unreachable: the shared rebuilder handles these,
        // reading the callee's structure and mapping successors to their copies.
        rebuild_terminator(cmap, callee, builder, callee_new, bb, |_, _, _| {});
    }
}

/// `v` as a value of type `want`. In verified IR the only type difference a
/// call/return boundary admits is `ptr` against an aggregate (both denote an
/// address); inlining turns that boundary into a strictly typed block
/// argument, so the address is bitcast.
fn coerce_address(builder: &mut FunctionBuilder<'_>, v: ValueId, want: TypeId) -> ValueId {
    if builder.value_type(v) == want { v } else { builder.cast(CastOp::Bitcast, v, want) }
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

    /// A struct-by-value callee — an aggregate parameter, and an aggregate
    /// result returned as the `ptr` of a local — called with a `ptr` argument
    /// (all address-compatible at a call boundary). Inlining turns those
    /// boundaries into strictly typed block arguments, which must still verify.
    #[test]
    fn inlines_struct_by_value_callee() {
        use crate::ir::inst::CastOp;
        let mut syms = StrInterner::new();
        let mut m = Module::new("inline-struct");
        let i32t = m.types_mut().int(32);
        let pair = m.types_mut().struct_(vec![i32t, i32t]);
        let g_sig = m.types_mut().func(vec![pair], pair, false);
        let g = m.declare_function(syms.intern("g"), g_sig);
        {
            let mut b = m.build(g);
            let e = b.create_entry_block();
            let v = b.param(e, 0);
            let r = b.alloca(pair);
            let x_p = b.struct_field(v, pair, 0);
            let x = b.load(i32t, x_p, 4);
            let rx = b.struct_field(r, pair, 0);
            b.store(i32t, rx, x, 4);
            b.ret(Some(r));
        }
        let f_sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("f"), f_sig);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let tmp = b.alloca(pair);
            let gref = b.func_ref(g);
            // One call passes the storage `ptr` directly, one bitcasts it.
            let r1 = b.call(gref, &[tmp], pair).expect("g returns a pair");
            let agg = b.cast(CastOp::Bitcast, tmp, pair);
            let r2 = b.call(gref, &[agg], pair).expect("g returns a pair");
            let p1 = b.struct_field(r1, pair, 0);
            let a = b.load(i32t, p1, 4);
            let p2 = b.struct_field(r2, pair, 0);
            let c = b.load(i32t, p2, 4);
            let s = b.add(a, c, Flags::NONE);
            b.ret(Some(s));
        }
        assert!(verify_module(&m).is_ok());
        assert_eq!(Inline::new().run(&mut m), Changed::Yes);
        assert_eq!(n_calls(m.function(f)), 0, "both calls are inlined");
        if let Err(diags) = verify_module(&m) {
            panic!("inline output must verify: {:?}", diags.iter().map(|d| &d.message).collect::<Vec<_>>());
        }
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

    // --- The cost model (issue #15) ------------------------------------------

    fn parse(src: &str) -> (Module, StrInterner) {
        let mut syms = StrInterner::new();
        let m = crate::ir::text::parse_module(src, crate::support::diagnostics::FileId::new(0), &mut syms)
            .unwrap_or_else(|e| panic!("parse: {e:?}"));
        verify_module(&m).unwrap_or_else(|e| panic!("source verifies: {e:?}"));
        (m, syms)
    }

    fn func<'m>(m: &'m Module, syms: &StrInterner, name: &str) -> &'m Function {
        m.functions().find(|f| syms.resolve(f.name) == name).unwrap_or_else(|| panic!("no @{name}"))
    }

    fn calls_in(m: &Module, syms: &StrInterner, name: &str) -> usize {
        n_calls(func(m, syms, name))
    }

    /// `@g(x)` adding 1 to `x` `adds` times: inlined size `adds`.
    fn adder(name: &str, attrs: &str, adds: usize) -> String {
        let mut s = format!("func {attrs}@{name}(i64) -> i64 {{\nentry ^0(%0: i64):\n");
        for k in 0..adds {
            let _ = writeln!(s, "  %{} = add %{k}, i64 1 : i64", k + 1);
        }
        let _ = write!(s, "  ret %{adds}\n}}\n");
        s
    }

    /// `@f(x) = g(x)`.
    const CALL_G: &str = "func @f(i64) -> i64 {\nentry ^0(%0: i64):\n  %1 = call @g(%0) : i64\n  ret %1\n}\n";

    #[test]
    fn a_callee_no_bigger_than_its_call_is_always_inlined() {
        // Threshold 0 disables the budgeted rule: only free inlining is left.
        let (mut m, f, _g) = build_leaf_caller();
        assert_eq!(Inline::with_threshold(0).run(&mut m), Changed::Yes);
        assert_eq!(n_calls(m.function(f)), 0);
        // `g(x) = x + 1 + ... + 1` (6 adds) costs more than `call g(x)` (5).
        for (adds, inlined) in [(5, true), (6, false)] {
            let (mut m, syms) = parse(&format!("module \"m\"\n{}{CALL_G}", adder("g", "", adds)));
            Inline::with_threshold(0).run(&mut m);
            assert_eq!(calls_in(&m, &syms, "f") == 0, inlined, "{adds} adds");
        }
    }

    /// The issue #13 callee writes its status through an out-parameter: the
    /// caller slots it addresses make the call cost more than the body.
    #[test]
    fn out_parameter_slots_count_toward_the_call_cost() {
        let src = "module \"m\"\n\
            func @try_dec(ptr, ptr) -> void {\nentry ^0(%0: ptr, %1: ptr):\n  %2 = load %1 align 8 : i64\n  \
            %3 = icmp eq %2, i64 0 : i1\n  cond_br %3, ^1, ^2\n^1:\n  store i8 1, %0 align 1 : i8\n  ret\n\
            ^2:\n  %4 = sub nuw %2, i64 1 : i64\n  store %4, %1 align 8 : i64\n  store i8 0, %0 align 1 : i8\n  ret\n}\n\
            func @slots(i64) -> i8 {\nentry ^0(%0: i64):\n  %1 = alloca i64 : ptr\n  store %0, %1 align 8 : i64\n  \
            %2 = alloca i8 : ptr\n  call @try_dec(%2, %1) : void\n  %3 = load %2 align 1 : i8\n  ret %3\n}\n\
            func @ptrs(ptr, ptr) -> void {\nentry ^0(%0: ptr, %1: ptr):\n  call @try_dec(%0, %1) : void\n  ret\n}\n";
        let (mut m, syms) = parse(src);
        Inline::with_threshold(0).run(&mut m);
        assert_eq!(calls_in(&m, &syms, "slots"), 0, "out-parameter slots: free");
        assert_eq!(calls_in(&m, &syms, "ptrs"), 1, "plain pointers: the body is bigger than the call");
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn hints_override_the_cost_model() {
        let (mut m, syms) = parse(&format!("module \"m\"\n{}{CALL_G}", adder("g", "inline(always) ", 100)));
        assert_eq!(Inline::with_threshold(0).run(&mut m), Changed::Yes, "always: any size");
        assert_eq!(calls_in(&m, &syms, "f"), 0);
        let (mut m, syms) = parse(&format!("module \"m\"\n{}{CALL_G}", adder("g", "internal inline(never) ", 0)));
        assert_eq!(Inline::new().run(&mut m), Changed::No, "never: even a free single-site callee");
        assert_eq!(calls_in(&m, &syms, "f"), 1);
        // A recursive inline(always) function is still never inlined into itself.
        let rec = "module \"m\"\nfunc inline(always) @r(i64) -> i64 {\nentry ^0(%0: i64):\n  \
                   %1 = call @r(%0) : i64\n  ret %1\n}\n";
        let (mut m, _) = parse(rec);
        assert_eq!(Inline::new().run(&mut m), Changed::No);
    }

    /// Mutually recursive `inline(always)` functions: each run inlines one
    /// level, which leaves direct self-calls the guard refuses, so repeated runs
    /// reach a fixpoint instead of unrolling forever.
    #[test]
    fn always_inline_recursion_is_bounded() {
        let src = "module \"m\"\n\
            func inline(always) @even(i64) -> i1 {\nentry ^0(%0: i64):\n  %1 = icmp eq %0, i64 0 : i1\n  \
            cond_br %1, ^1, ^2\n^1:\n  ret i1 1\n^2:\n  %2 = sub %0, i64 1 : i64\n  %3 = call @odd(%2) : i1\n  ret %3\n}\n\
            func inline(always) @odd(i64) -> i1 {\nentry ^0(%0: i64):\n  %1 = icmp eq %0, i64 0 : i1\n  \
            cond_br %1, ^1, ^2\n^1:\n  ret i1 0\n^2:\n  %2 = sub %0, i64 1 : i64\n  %3 = call @even(%2) : i1\n  ret %3\n}\n";
        let (mut m, syms) = parse(src);
        assert_eq!(Inline::new().run(&mut m), Changed::Yes);
        assert_eq!(Inline::new().run(&mut m), Changed::No, "only self-calls are left");
        assert!(verify_module(&m).is_ok());
        assert_eq!(calls_in(&m, &syms, "even"), 1);
        assert_eq!(calls_in(&m, &syms, "odd"), 1);
    }

    #[test]
    fn a_single_call_site_internal_callee_is_inlined_whatever_its_size() {
        let two_calls = "func @f2(i64) -> i64 {\nentry ^0(%0: i64):\n  %1 = call @g(%0) : i64\n  ret %1\n}\n";
        for (attrs, extra, inlined) in
            [("internal ", "", true), ("internal ", two_calls, false), ("", "", false)]
        {
            let src = format!("module \"m\"\n{}{CALL_G}{extra}", adder("g", attrs, 100));
            let (mut m, syms) = parse(&src);
            Inline::new().run(&mut m);
            assert_eq!(calls_in(&m, &syms, "f") == 0, inlined, "{attrs:?} {}", extra.is_empty());
            assert!(verify_module(&m).is_ok());
        }
    }

    /// 60 calls of a 20-unit external callee (growth 15 each) in a 301-unit
    /// caller: the caller's budget (301 / 2 = 150) admits exactly 10.
    #[test]
    fn the_budgets_bound_growth() {
        let mut caller = String::from("func @f(i64) -> i64 {\nentry ^0(%0: i64):\n");
        for k in 0..60 {
            let _ = writeln!(caller, "  %{} = call @g(%{k}) : i64", k + 1);
        }
        caller.push_str("  ret %60\n}\n");
        let (mut m, syms) = parse(&format!("module \"m\"\n{}{caller}", adder("g", "", 20)));
        assert_eq!(super::function_size(func(&m, &syms, "f")), 301);
        Inline::new().run(&mut m);
        assert_eq!(calls_in(&m, &syms, "f"), 50);
        assert!(verify_module(&m).is_ok());
    }

    /// `g(x, c) = c ? <40 adds> : x`: with a constant `c` the dead arm costs
    /// nothing, so `g(x, 0)` is free to inline while `g(x, 1)` and `g(x, y)`
    /// are not.
    #[test]
    fn constant_arguments_fold_the_estimate() {
        let mut g = String::from("func @g(i64, i1) -> i64 {\nentry ^0(%0: i64, %1: i1):\n  cond_br %1, ^1, ^2\n^1:\n");
        for k in 0..40 {
            let _ = writeln!(g, "  %{} = add %{}, i64 1 : i64", k + 2, if k == 0 { 0 } else { k + 1 });
        }
        g.push_str("  ret %41\n^2:\n  ret %0\n}\n");
        let callers = "func @zero(i64) -> i64 {\nentry ^0(%0: i64):\n  %1 = call @g(%0, i1 0) : i64\n  ret %1\n}\n\
            func @one(i64) -> i64 {\nentry ^0(%0: i64):\n  %1 = call @g(%0, i1 1) : i64\n  ret %1\n}\n\
            func @var(i64, i1) -> i64 {\nentry ^0(%0: i64, %1: i1):\n  %2 = call @g(%0, %1) : i64\n  ret %2\n}\n";
        let (mut m, syms) = parse(&format!("module \"m\"\n{g}{callers}"));
        Inline::with_threshold(0).run(&mut m);
        assert_eq!(calls_in(&m, &syms, "zero"), 0);
        assert_eq!(calls_in(&m, &syms, "one"), 1);
        assert_eq!(calls_in(&m, &syms, "var"), 1);
        assert!(verify_module(&m).is_ok());
    }

    #[test]
    fn weak_dyn_alloca_and_large_frame_callees_are_kept() {
        let body = |attrs: &str, inst: &str| {
            format!("module \"m\"\nfunc {attrs}@g(i64) -> i64 {{\nentry ^0(%0: i64):\n{inst}  ret %0\n}}\n{CALL_G}")
        };
        for (attrs, inst, inlined) in [
            ("", "", true),
            ("weak ", "", false),
            ("", "  %1 = dyn_alloca %0 align 16 : ptr\n", false),
            ("", "  %1 = alloca [1024 x i8] : ptr\n  store i8 0, %1 align 1 : i8\n", false),
            ("inline(always) ", "  %1 = alloca [1024 x i8] : ptr\n  store i8 0, %1 align 1 : i8\n", true),
        ] {
            let (mut m, syms) = parse(&body(attrs, inst));
            Inline::new().run(&mut m);
            assert_eq!(calls_in(&m, &syms, "f") == 0, inlined, "{attrs:?} {inst:?}");
        }
    }

    /// Lode's `StackBuf.push_front` shape (issue #15): a 21-instruction helper
    /// that shifts the buffer and reports success through a status byte. At
    /// -O2 every call is inlined, the status slots are promoted, and the
    /// results are unchanged.
    #[test]
    fn stackbuf_push_front_is_inlined_and_its_status_promoted() {
        use crate::ir::refexec::run_named;
        use crate::ir::semantics::SemValue;
        use crate::transform::pipeline::{OptLevel, optimize};
        let src = format!("module \"stackbuf\"\n{PUSH_FRONT}");
        let (orig, syms) = parse(&src);
        let pf = func(&orig, &syms, "push_front");
        let insts: usize = pf.blocks().map(|(_, b)| b.insts().len() + 1).sum();
        assert_eq!(insts, 21);
        for level in [OptLevel::O2, OptLevel::O3] {
            let mut m = orig.clone();
            optimize(&mut m, level);
            verify_module(&m).unwrap_or_else(|e| panic!("{level:?} verifies: {e:?}"));
            for name in ["push_one", "push_two"] {
                let f = func(&m, &syms, name);
                let left = count_kind(f, |k| matches!(k, InstKind::Call | InstKind::Alloca { .. }));
                assert_eq!(left, 0, "{level:?} @{name}:\n{}", crate::ir::text::print_module(&m, &syms));
            }
            for (a, b) in [(1, 2), (-5, 7), (0, 0)] {
                let args = [SemValue::int(64, Int::from_i64(a)), SemValue::int(64, Int::from_i64(b))];
                let want = run_named(&orig, &syms, "drive", &args).expect("source runs").expect("result");
                let got = run_named(&m, &syms, "drive", &args).expect("optimized runs").expect("result");
                assert!(got.refines(&want), "{level:?} ({a}, {b}): {got:?} vs {want:?}");
            }
        }
    }

    /// `push_front(buf, v, status)` on `{len: i64, cap: i64, data: [cap x i64]}`.
    const PUSH_FRONT: &str = "\
func @push_front(ptr, i64, ptr) -> void {
entry ^0(%0: ptr, %1: i64, %2: ptr):
  %3 = load %0 align 8 : i64
  %4 = ptr_add %0, i64 8 : ptr
  %5 = load %4 align 8 : i64
  %6 = icmp uge %3, %5 : i1
  %7 = ptr_add %0, i64 16 : ptr
  %8 = shl %3, i64 3 : i64
  %9 = ptr_add %7, %8 : ptr
  cond_br %6, ^1, ^2(%9)
^1:
  store i8 1, %2 align 1 : i8
  ret
^2(%10: ptr):
  %11 = icmp eq %10, %7 : i1
  cond_br %11, ^4, ^3
^3:
  %12 = ptr_add %10, i64 -8 : ptr
  %13 = load %12 align 8 : i64
  store %13, %10 align 8 : i64
  br ^2(%12)
^4:
  store %1, %7 align 8 : i64
  %14 = add %3, i64 1 : i64
  store %14, %0 align 8 : i64
  store i8 0, %2 align 1 : i8
  ret
}
func @push_two(ptr, i64, i64) -> i64 {
entry ^0(%0: ptr, %1: i64, %2: i64):
  %3 = alloca i8 : ptr
  call @push_front(%0, %1, %3) : void
  %4 = load %3 align 1 : i8
  %5 = icmp ne %4, i8 0 : i1
  cond_br %5, ^1, ^2
^1:
  ret i64 -1
^2:
  %6 = alloca i8 : ptr
  call @push_front(%0, %2, %6) : void
  %7 = load %6 align 1 : i8
  %8 = icmp ne %7, i8 0 : i1
  cond_br %8, ^1, ^3
^3:
  %9 = load %0 align 8 : i64
  ret %9
}
func @push_one(ptr, i64) -> i64 {
entry ^0(%0: ptr, %1: i64):
  %2 = alloca i8 : ptr
  call @push_front(%0, %1, %2) : void
  %3 = load %2 align 1 : i8
  %4 = zext %3 : i64
  ret %4
}
func @drive(i64, i64) -> i64 {
entry ^0(%0: i64, %1: i64):
  %2 = alloca {i64, i64, [3 x i64]} : ptr
  store i64 0, %2 align 8 : i64
  %3 = ptr_add %2, i64 8 : ptr
  store i64 3, %3 align 8 : i64
  %4 = call @push_two(%2, %0, %1) : i64
  %5 = call @push_one(%2, %1) : i64
  %6 = call @push_two(%2, %0, %0) : i64
  %7 = ptr_add %2, i64 16 : ptr
  %8 = load %7 align 8 : i64
  %9 = ptr_add %2, i64 32 : ptr
  %10 = load %9 align 8 : i64
  %11 = mul %4, i64 1000 : i64
  %12 = mul %5, i64 100 : i64
  %13 = add %11, %12 : i64
  %14 = add %13, %6 : i64
  %15 = mul %8, i64 7 : i64
  %16 = add %14, %15 : i64
  %17 = sub %16, %10 : i64
  ret %17
}
";
}
