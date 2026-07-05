//! z3rs-driven **superoptimization** (bet **B5**, tenet **T3**; ROADMAP Phase 10).
//!
//! A Souper-style *bounded synthesizer* for the pure single-block integer
//! subset. Given a small target function it enumerates candidate instruction
//! sequences over the integer opcode set, **prunes** them hard, and accepts one
//! only when [`check_refinement`](crate::verify::check_refinement) — the `z3rs`
//! refinement oracle (bet **B2**) — *proves* it a refinement. The cheapest such
//! proven-equivalent, when strictly cheaper than the input, is the result. Every
//! accepted rewrite carries a `z3rs` `Refines` verdict, so the synthesizer grows
//! a **verified rewrite database** the equality-saturation optimizer
//! ([`egraph`](crate::transform::egraph), bet **B4**) can consume — the compiler
//! compounds in capability with a proof trail, never trusting a rewrite that is
//! only *concretely* equal on samples.
//!
//! ## The subset
//!
//! Exactly the subset the refinement checker reasons about: a single block of
//! **pure, integer-typed** value-producing ops ending in `ret <value>` — integer
//! [`Bin`](crate::ir::inst::BinOp) (add/sub/mul/div/rem/bitwise/shift),
//! [`ICmp`](crate::ir::inst::InstKind::ICmp), and
//! [`Select`](crate::ir::inst::InstKind::Select) over integer parameters. A
//! target outside the subset (memory, calls, floats, pointers, multi-block
//! control flow) is simply not optimized (`None`); it is never mis-handled.
//!
//! ## Synthesis & pruning
//!
//! The target is lifted to an [`Expr`] tree ([`Spec::from_function`]). Candidates
//! are enumerated as [`Expr`] trees of increasing size over the target's integer
//! parameters and a small deterministic set of constant templates, using the
//! candidate opcode set [`CAND_OPS`] (all flag-free). Two prunes make the search
//! affordable:
//!
//! 1. **Cost bound.** Each candidate is scored with a small additive cost model
//!    ([`cost_of`] over [`bin_own_cost`]) mirroring the e-graph's B9 cost
//!    (constants/inputs cheapest, shifts cheaper than add, multiply dear,
//!    divide/remainder dearest). Only candidates *strictly cheaper* than the best
//!    so far are considered, and whole size classes are skipped once their
//!    minimum possible cost cannot beat the best (`2·size + 1`).
//! 2. **Concrete pre-filter.** Before ever calling the solver, the candidate and
//!    the target are evaluated on a batch of edge-case + pseudo-random concrete
//!    inputs with [`crate::ir::eval`] ([`concretely_refines`]). The check mirrors
//!    the *refinement* relation pointwise (poison source licenses any target;
//!    a defined source pins the target; a target that introduces UB where the
//!    source was UB-free is rejected), so it never rejects a true refinement but
//!    cheaply discards the overwhelming majority of candidates.
//!
//! ## z3rs is the gate, not the samples
//!
//! A candidate that survives the concrete pre-filter is **only a hypothesis**:
//! agreeing on samples is necessary, not sufficient. It is built into a `src`
//! (target) / `tgt` (candidate) function pair and handed to
//! [`check_refinement`]; the rewrite is accepted **iff** the verdict is
//! [`Refines`](crate::verify::RefinementResult::Refines). `Unknown` and
//! `Counterexample` are never accepted. The in-file test
//! `solver_is_the_gate_not_samples` constructs a near-miss (`x` vs `x & 0xFE`)
//! that agrees on a hand-picked sample set yet is refuted by the solver, and
//! asserts the solver rejects it — proving samples alone are not trusted.
//!
//! ## The rewrite database & how B4 ingests it
//!
//! [`discover_rules`] runs the synthesizer over a set of seed functions and
//! returns a [`RewriteDatabase`]: for each seed where a strictly-cheaper proven
//! equivalent was found, it records a [`RewriteRule`] whose `lhs`/`rhs` are two
//! functions (built into the database's own [`Module`], sharing its interning
//! tables) plus the recorded `z3rs` `Refines` proof and the two costs. Each rule
//! is *independently re-verified* with [`check_refinement`] before it is
//! admitted, so the database contains only proven refinements.
//!
//! The shape is deliberately the same algebraic `lhs ⇒ rhs` the e-graph already
//! uses: to ingest a rule, the e-graph would lift `lhs` and `rhs` to
//! [`NodeOp`](crate::transform::egraph) e-node patterns exactly as it lifts its
//! built-in rules (parameters become pattern variables, `Bin`/`ICmp`/`Select`
//! map one-to-one), then, wherever `lhs` matches an e-class, add `rhs` and union
//! the two classes. The recorded `Refines` proof is precisely the B2 obligation
//! that licenses that union; the e-graph's min-cost extraction then picks the
//! cheaper `rhs` (this file does not modify `egraph.rs` — it only produces rules
//! in the compatible shape).
//!
//! ## Termination & determinism
//!
//! Every search runs under a hard [`Budget`]: `max_ops` (sequence length),
//! `max_candidates` (candidates examined), `max_solver_calls` (solver calls), and
//! `max_samples` (concrete inputs). Enumeration is strictly ordered — sizes
//! ascending, opcodes in [`CAND_OPS`] order, parameters before constants — and
//! never iterates a hash map, so the result is byte-for-byte reproducible across
//! runs (asserted by `deterministic_across_runs`). A target with no cheaper
//! proven equivalent returns `None`/no rule without blowing up.

use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, Flags, InstKind, IntPred};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ConstPool, ValueDef, ValueId};
use crate::ir::{EvalOutcome, FuncId, Function, Module, SemValue, eval};
use crate::support::StrInterner;
use crate::verify::{RefinementResult, check_refinement};

use puremp::Int;

// ---------------------------------------------------------------------------
// Budgets (all termination bounds live here; see the module docs).
// ---------------------------------------------------------------------------

/// The candidate opcode set the synthesizer enumerates (all flag-free). Ordered;
/// enumeration follows this order, which fixes tie-breaks deterministically.
/// Division/remainder are intentionally excluded from *generation* (cost-20,
/// never cheaper, and UB-prone), though they are handled when lifting a target.
const CAND_OPS: [BinOp; 9] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::And,
    BinOp::Or,
    BinOp::Xor,
    BinOp::Shl,
    BinOp::LShr,
    BinOp::AShr,
];

/// The hard resource budget bounding a single synthesis run. All four bounds
/// together guarantee termination regardless of the target.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// Maximum number of instructions (internal op nodes) in a candidate.
    pub max_ops: usize,
    /// Maximum number of candidates examined before the search stops.
    pub max_candidates: usize,
    /// Maximum number of `z3rs` refinement queries issued.
    pub max_solver_calls: usize,
    /// Number of concrete inputs used by the pre-filter.
    pub max_samples: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Budget { max_ops: 2, max_candidates: 20_000, max_solver_calls: 400, max_samples: 48 }
    }
}

// ---------------------------------------------------------------------------
// The candidate / target expression IR.
// ---------------------------------------------------------------------------

/// A pure straight-line integer expression over typed parameters — the internal
/// representation of both a lifted target and an enumerated candidate. It is
/// table-independent (widths are plain `u32`, constants are [`Int`]), so the same
/// tree can be evaluated concretely and rebuilt into any [`Module`].
#[derive(Clone, Debug, PartialEq, Eq)]
enum Expr {
    /// The `i`-th function parameter.
    Param(u32),
    /// An integer constant of the given width (its unsigned representative).
    Const {
        /// The bit width of the constant.
        width: u32,
        /// The value (interpreted modulo `2^width`).
        value: Int,
    },
    /// A flag-carrying integer binary op over two same-width children.
    Bin(BinOp, Flags, Box<Expr>, Box<Expr>),
    /// An integer comparison (result `i1`) over two same-width children.
    ICmp(IntPred, Box<Expr>, Box<Expr>),
    /// A ternary select: `[cond(i1), if_true, if_false]`.
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
}

/// The abstract description of a target: parameter widths, return width, and the
/// expression that computes the returned value.
#[derive(Clone, Debug)]
struct Spec {
    param_widths: Vec<u32>,
    ret_width: u32,
    target: Expr,
}

impl Spec {
    /// Lift a function to a [`Spec`], or `None` if it is outside the pure
    /// single-block integer subset.
    fn from_function(types: &TypeContext, consts: &ConstPool, func: &Function) -> Option<Spec> {
        if func.block_count() != 1 {
            return None;
        }
        let entry = func.entry()?;
        let block = func.block(entry);
        let term = func.inst(block.terminator()?);
        if !matches!(term.kind, InstKind::Ret) {
            return None;
        }
        let ops = term.operands();
        if ops.len() != 1 {
            return None;
        }
        let ret_val = ops[0];

        let params = block.params();
        let mut param_widths = Vec::with_capacity(params.len());
        for &p in params {
            param_widths.push(int_width(types, func.value_type(p))?);
        }
        let ret_width = int_width(types, func.value_type(ret_val))?;
        let target = expr_from_value(func, types, consts, ret_val, params)?;
        Some(Spec { param_widths, ret_width, target })
    }
}

/// Lift the value DAG rooted at `v` to an [`Expr`], or `None` if it uses a
/// construct outside the subset.
fn expr_from_value(
    func: &Function,
    types: &TypeContext,
    consts: &ConstPool,
    v: ValueId,
    params: &[ValueId],
) -> Option<Expr> {
    if let Some(i) = params.iter().position(|&p| p == v) {
        return Some(Expr::Param(i as u32));
    }
    match &func.value(v).def {
        ValueDef::Const(cid) => match consts.get(*cid) {
            Const::Int { ty, value } => {
                Some(Expr::Const { width: int_width(types, *ty)?, value: value.clone() })
            }
            _ => None,
        },
        ValueDef::Inst(i) => {
            let inst = func.inst(*i);
            let sub = |o: ValueId| expr_from_value(func, types, consts, o, params);
            let ops = inst.operands();
            match &inst.kind {
                InstKind::Bin(op) if !op.is_float() && ops.len() == 2 => {
                    Some(Expr::Bin(*op, inst.flags, Box::new(sub(ops[0])?), Box::new(sub(ops[1])?)))
                }
                InstKind::ICmp(pred) if ops.len() == 2 => {
                    Some(Expr::ICmp(*pred, Box::new(sub(ops[0])?), Box::new(sub(ops[1])?)))
                }
                InstKind::Select if ops.len() == 3 => Some(Expr::Select(
                    Box::new(sub(ops[0])?),
                    Box::new(sub(ops[1])?),
                    Box::new(sub(ops[2])?),
                )),
                _ => None,
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Cost model (a small additive B9-style cost; see the e-graph's `own_cost`).
// ---------------------------------------------------------------------------

/// The own cost of a binary opcode: shifts cheapest (a constant shift is a
/// genuinely cheap operation), add/sub/bitwise next, multiply dear, div/rem
/// dearest. Distinguishing shifts from adds is what makes `x*2 ⇒ x<<1` a strict,
/// uniquely-cheapest improvement.
fn bin_own_cost(op: BinOp) -> u64 {
    match op {
        BinOp::Shl | BinOp::LShr | BinOp::AShr => 1,
        BinOp::Mul => 6,
        BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => 20,
        _ => 2, // add / sub / and / or / xor (and unreached float ops)
    }
}

/// The additive cost of an expression tree (leaves cost 1; shared subterms are
/// counted per occurrence, matching the e-graph's tree cost).
fn cost_of(e: &Expr) -> u64 {
    match e {
        Expr::Param(_) | Expr::Const { .. } => 1,
        Expr::Bin(op, _, l, r) => bin_own_cost(*op) + cost_of(l) + cost_of(r),
        Expr::ICmp(_, l, r) => 2 + cost_of(l) + cost_of(r),
        Expr::Select(c, t, f) => 3 + cost_of(c) + cost_of(t) + cost_of(f),
    }
}

// ---------------------------------------------------------------------------
// Small integer helpers.
// ---------------------------------------------------------------------------

/// The integer width of a type, or `None` if it is not an integer type.
fn int_width(types: &TypeContext, ty: TypeId) -> Option<u32> {
    match types.get(ty) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}

/// The all-ones bit pattern (`2^width − 1`, i.e. `-1`) of a width.
fn all_ones(width: u32) -> Int {
    Int::from_u64(1).mul_2k(width).sub(&Int::ONE)
}

/// The sign-bit pattern (`2^(width−1)`, i.e. `INT_MIN`) of a width.
fn sign_bit(width: u32) -> Int {
    if width == 0 { Int::ZERO } else { Int::from_u64(1).mul_2k(width - 1) }
}

/// Push `w` into `v` only if absent (a tiny order-preserving dedup).
fn push_unique(v: &mut Vec<u32>, w: u32) {
    if !v.contains(&w) {
        v.push(w);
    }
}

// ---------------------------------------------------------------------------
// The width→TypeId map for building / evaluating against a module's tables.
// ---------------------------------------------------------------------------

/// Interned integer types keyed by width, for one [`Module`]'s tables.
#[derive(Debug)]
struct TypeMap {
    by_width: Vec<(u32, TypeId)>,
}

impl TypeMap {
    /// Intern every integer width the spec/expr needs (parameters, return, `i1`
    /// for comparisons, and all constant widths) into `module`.
    fn build(module: &mut Module, param_widths: &[u32], ret_width: u32, expr: &Expr) -> TypeMap {
        let mut widths: Vec<u32> = Vec::new();
        push_unique(&mut widths, 1);
        push_unique(&mut widths, ret_width);
        for &w in param_widths {
            push_unique(&mut widths, w);
        }
        collect_const_widths(expr, &mut widths);
        let by_width = widths.into_iter().map(|w| (w, module.types_mut().int(w))).collect();
        TypeMap { by_width }
    }

    /// The interned type of a width (which must have been collected at build).
    fn ty(&self, w: u32) -> TypeId {
        self.by_width
            .iter()
            .find(|(x, _)| *x == w)
            .map(|(_, t)| *t)
            .expect("width interned in TypeMap")
    }
}

/// Collect the widths of every constant leaf in `e` (other widths are covered by
/// the parameter/return set).
fn collect_const_widths(e: &Expr, out: &mut Vec<u32>) {
    match e {
        Expr::Param(_) => {}
        Expr::Const { width, .. } => push_unique(out, *width),
        Expr::Bin(_, _, l, r) | Expr::ICmp(_, l, r) => {
            collect_const_widths(l, out);
            collect_const_widths(r, out);
        }
        Expr::Select(c, t, f) => {
            collect_const_widths(c, out);
            collect_const_widths(t, out);
            collect_const_widths(f, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Concrete evaluation & the concrete pre-filter.
// ---------------------------------------------------------------------------

/// The result width of an expression (a `Bin` shares its left child's width; an
/// `ICmp` is `i1`; a `Select` shares its arms' width).
fn expr_width(spec: &Spec, e: &Expr) -> u32 {
    match e {
        Expr::Param(i) => spec.param_widths[*i as usize],
        Expr::Const { width, .. } => *width,
        Expr::Bin(_, _, l, _) => expr_width(spec, l),
        Expr::ICmp(..) => 1,
        Expr::Select(_, t, _) => expr_width(spec, t),
    }
}

/// Evaluate an expression on concrete parameter values with the reference
/// interpreter [`crate::ir::eval`]. A UB sub-result poisons the whole evaluation
/// with [`EvalOutcome::UndefinedBehavior`] (there is no defined value to feed the
/// parent).
fn eval_expr(
    types: &TypeContext,
    tm: &TypeMap,
    spec: &Spec,
    e: &Expr,
    inputs: &[SemValue],
) -> EvalOutcome {
    match e {
        Expr::Param(i) => EvalOutcome::Value(inputs[*i as usize].clone()),
        Expr::Const { width, value } => EvalOutcome::Value(SemValue::int(*width, value.clone())),
        Expr::Bin(op, flags, l, r) => {
            let a = match eval_expr(types, tm, spec, l, inputs) {
                EvalOutcome::Value(v) => v,
                ub => return ub,
            };
            let b = match eval_expr(types, tm, spec, r, inputs) {
                EvalOutcome::Value(v) => v,
                ub => return ub,
            };
            eval(types, tm.ty(expr_width(spec, e)), &InstKind::Bin(*op), flags, &[a, b])
        }
        Expr::ICmp(pred, l, r) => {
            let a = match eval_expr(types, tm, spec, l, inputs) {
                EvalOutcome::Value(v) => v,
                ub => return ub,
            };
            let b = match eval_expr(types, tm, spec, r, inputs) {
                EvalOutcome::Value(v) => v,
                ub => return ub,
            };
            eval(types, tm.ty(1), &InstKind::ICmp(*pred), &Flags::NONE, &[a, b])
        }
        Expr::Select(c, t, f) => {
            let cv = match eval_expr(types, tm, spec, c, inputs) {
                EvalOutcome::Value(v) => v,
                ub => return ub,
            };
            let tv = match eval_expr(types, tm, spec, t, inputs) {
                EvalOutcome::Value(v) => v,
                ub => return ub,
            };
            let fv = match eval_expr(types, tm, spec, f, inputs) {
                EvalOutcome::Value(v) => v,
                ub => return ub,
            };
            eval(types, tm.ty(expr_width(spec, e)), &InstKind::Select, &Flags::NONE, &[cv, tv, fv])
        }
    }
}

/// Whether `cand` refines `spec.target` on every sample, mirroring the refinement
/// relation pointwise. This is a *necessary* condition for a true refinement (so
/// it never rejects a real one) but not sufficient — the solver is the gate.
fn concretely_refines(
    types: &TypeContext,
    tm: &TypeMap,
    spec: &Spec,
    cand: &Expr,
    samples: &[Vec<SemValue>],
    target_out: &[EvalOutcome],
) -> bool {
    for (inputs, s) in samples.iter().zip(target_out) {
        let sv = match s {
            // Source UB: the obligation is conditional on it, so no constraint.
            EvalOutcome::UndefinedBehavior => continue,
            EvalOutcome::Value(v) => v,
        };
        match eval_expr(types, tm, spec, cand, inputs) {
            // Target must not introduce UB where the source was UB-free.
            EvalOutcome::UndefinedBehavior => return false,
            EvalOutcome::Value(tv) => {
                if sv.is_poison() {
                    continue; // a poison source licenses any (defined) target
                }
                if tv.is_poison() || *sv != tv {
                    return false; // a defined source pins the target
                }
            }
        }
    }
    true
}

/// Deterministic concrete inputs: edge cases (0, ±1, ±2, 3, `INT_MIN`,
/// `INT_MAX`, `-1`) rotated per parameter, then an LCG-driven pseudo-random tail.
fn gen_samples(param_widths: &[u32], n: usize) -> Vec<Vec<SemValue>> {
    let edges: Vec<Vec<Int>> = param_widths.iter().map(|&w| edge_values(w)).collect();
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mut inp = Vec::with_capacity(param_widths.len());
        for (pi, &w) in param_widths.iter().enumerate() {
            let e = &edges[pi];
            let raw = if k < e.len() {
                e[(k + pi) % e.len()].clone()
            } else {
                let mut s = (k as u64).wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                s ^= (pi as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                Int::from_u64(s)
            };
            inp.push(SemValue::int(w, raw));
        }
        out.push(inp);
    }
    out
}

/// The edge-case value patterns for a width.
fn edge_values(width: u32) -> Vec<Int> {
    let mut v = Vec::new();
    for m in [0i64, 1, 2, 3, -1, -2] {
        v.push(Int::from_i64(m).mod_2k(width));
    }
    v.push(sign_bit(width)); // INT_MIN
    v.push(sign_bit(width).sub(&Int::ONE)); // INT_MAX
    v.push(all_ones(width)); // -1 pattern
    v
}

// ---------------------------------------------------------------------------
// Building expressions into a module (for verification and for the result).
// ---------------------------------------------------------------------------

/// Recursively materialize `e` into the function under construction.
fn build_expr(b: &mut FunctionBuilder<'_>, e: &Expr, params: &[ValueId], tm: &TypeMap) -> ValueId {
    match e {
        Expr::Param(i) => params[*i as usize],
        Expr::Const { width, value } => b.const_int(tm.ty(*width), value.clone()),
        Expr::Bin(op, flags, l, r) => {
            let lv = build_expr(b, l, params, tm);
            let rv = build_expr(b, r, params, tm);
            b.bin(*op, lv, rv, *flags)
        }
        Expr::ICmp(pred, l, r) => {
            let lv = build_expr(b, l, params, tm);
            let rv = build_expr(b, r, params, tm);
            b.icmp(*pred, lv, rv)
        }
        Expr::Select(c, t, f) => {
            let cv = build_expr(b, c, params, tm);
            let tv = build_expr(b, t, params, tm);
            let fv = build_expr(b, f, params, tm);
            b.select(cv, tv, fv)
        }
    }
}

/// Declare and build a single-block `(params...) -> ret` function computing `expr`
/// in `module`, returning its id.
fn build_fn_from_expr(
    module: &mut Module,
    syms: &mut StrInterner,
    name: &str,
    param_widths: &[u32],
    ret_width: u32,
    expr: &Expr,
) -> FuncId {
    let tm = TypeMap::build(module, param_widths, ret_width, expr);
    let param_tys: Vec<TypeId> = param_widths.iter().map(|&w| tm.ty(w)).collect();
    let ret_ty = tm.ty(ret_width);
    let sig = module.types_mut().func(param_tys, ret_ty, false);
    let sym = syms.intern(name);
    let id = module.declare_function(sym, sig);
    {
        let mut b = module.build(id);
        let entry = b.create_entry_block();
        let params: Vec<ValueId> = b.block_params(entry).to_vec();
        let r = build_expr(&mut b, expr, &params, &tm);
        b.ret(Some(r));
    }
    id
}

/// The final gate: build `src` = target and `tgt` = candidate as functions of a
/// fresh module (shared tables) and ask `z3rs` whether the candidate refines the
/// target.
fn prove_refines(spec: &Spec, cand: &Expr) -> RefinementResult {
    let mut m = Module::new("superopt-verify");
    let mut syms = StrInterner::new();
    let src =
        build_fn_from_expr(&mut m, &mut syms, "src", &spec.param_widths, spec.ret_width, &spec.target);
    let tgt = build_fn_from_expr(&mut m, &mut syms, "tgt", &spec.param_widths, spec.ret_width, cand);
    check_refinement(m.types(), m.consts(), m.function(src), m.function(tgt))
}

// ---------------------------------------------------------------------------
// Enumeration & the search.
// ---------------------------------------------------------------------------

/// The width-`width` leaves: parameters of that width (in order) then the
/// constant templates.
fn candidate_leaves(spec: &Spec, width: u32) -> Vec<Expr> {
    let mut leaves = Vec::new();
    for (i, &w) in spec.param_widths.iter().enumerate() {
        if w == width {
            leaves.push(Expr::Param(i as u32));
        }
    }
    for value in const_templates(width) {
        leaves.push(Expr::Const { width, value });
    }
    leaves
}

/// The deterministic constant leaf set for a width: small magnitudes, all-ones,
/// and the sign bit, masked to the width and deduplicated.
fn const_templates(width: u32) -> Vec<Int> {
    let mut mags: Vec<Int> = Vec::new();
    for k in [0u64, 1, 2, 3, 4, 8, 16] {
        mags.push(Int::from_u64(k));
    }
    mags.push(all_ones(width));
    mags.push(sign_bit(width));
    let mut out: Vec<Int> = Vec::new();
    for v in mags {
        let m = v.mod_2k(width);
        if !out.contains(&m) {
            out.push(m);
        }
    }
    out
}

/// Build all size-`size` candidates from the smaller levels, in deterministic
/// order, capped at `cap` entries.
fn build_level(size: usize, levels: &[Vec<Expr>], cap: usize) -> Vec<Expr> {
    let mut cur = Vec::new();
    'outer: for &op in &CAND_OPS {
        for k in 0..size {
            let rs = size - 1 - k;
            for l in &levels[k] {
                for r in &levels[rs] {
                    if cur.len() >= cap {
                        break 'outer;
                    }
                    cur.push(Expr::Bin(op, Flags::NONE, Box::new(l.clone()), Box::new(r.clone())));
                }
            }
        }
    }
    cur
}

/// A proven-equivalent candidate found by the search.
#[derive(Debug)]
struct Found {
    expr: Expr,
    cost: u64,
    proof: RefinementResult,
}

/// The core bounded search. Returns the cheapest candidate that provably refines
/// `spec.target` with cost `< max_cost_excl` (where `max_cost_excl` is the target
/// cost for a *strict* improvement, or one more than it to allow an equal-cost
/// re-derivation), or `None`.
fn synthesize(spec: &Spec, budget: &Budget, strict: bool) -> Option<Found> {
    let target_cost = cost_of(&spec.target);
    let mut best_cost = if strict { target_cost } else { target_cost + 1 };
    if best_cost == 0 {
        return None;
    }

    // Scratch tables for the concrete pre-filter (built once, then read-only).
    let mut scratch = Module::new("superopt-scratch");
    let tm = TypeMap::build(&mut scratch, &spec.param_widths, spec.ret_width, &spec.target);
    let types = scratch.types();

    let samples = gen_samples(&spec.param_widths, budget.max_samples);
    let target_out: Vec<EvalOutcome> =
        samples.iter().map(|inp| eval_expr(types, &tm, spec, &spec.target, inp)).collect();

    let mut levels: Vec<Vec<Expr>> = vec![candidate_leaves(spec, spec.ret_width)];
    let mut best: Option<Found> = None;
    let mut solver_calls = 0usize;
    let mut seen = 0usize;

    for size in 0..=budget.max_ops {
        if size >= 1 {
            // The cheapest size-`size` tree costs `2·size + 1`; if that cannot
            // beat the best, neither can any larger size (cost is monotone).
            if 2 * (size as u64) + 1 >= best_cost {
                break;
            }
            let level = build_level(size, &levels, budget.max_candidates);
            levels.push(level);
        }

        for cand in &levels[size] {
            seen += 1;
            if seen > budget.max_candidates {
                return best;
            }
            let c = cost_of(cand);
            if c >= best_cost {
                continue; // cost bound
            }
            if !concretely_refines(types, &tm, spec, cand, &samples, &target_out) {
                continue; // concrete pre-filter
            }
            if solver_calls >= budget.max_solver_calls {
                continue; // out of solver budget: never accept unverified
            }
            solver_calls += 1;
            let verdict = prove_refines(spec, cand);
            if verdict.is_refines() {
                best_cost = c;
                best = Some(Found { expr: cand.clone(), cost: c, proof: verdict });
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Public API: single-function superoptimization.
// ---------------------------------------------------------------------------

/// Superoptimize one small function with the default [`Budget`]: search for a
/// **proven, strictly-cheaper** equivalent of `module.function(func)` over the
/// pure single-block integer subset. Returns the freshly built replacement
/// function (sharing `module`'s tables, ready for
/// [`Module::replace_function`](crate::ir::Module::replace_function)), or `None`
/// if the target is out of subset or already optimal.
pub fn superoptimize(module: &mut Module, func: FuncId) -> Option<Function> {
    superoptimize_with(module, func, &Budget::default())
}

/// [`superoptimize`] under an explicit [`Budget`].
pub fn superoptimize_with(module: &mut Module, func: FuncId, budget: &Budget) -> Option<Function> {
    let spec = Spec::from_function(module.types(), module.consts(), module.function(func))?;
    let found = synthesize(&spec, budget, true)?;
    if !found.proof.is_refines() {
        return None; // defensive: only a proven refinement is ever returned
    }

    // Rebuild the winning candidate into a fresh function of `module`.
    let tm = TypeMap::build(module, &spec.param_widths, spec.ret_width, &found.expr);
    let expr = found.expr;
    let (fresh, ()) = module.map_function(func, |_old, b| {
        let entry = b.create_entry_block();
        let params: Vec<ValueId> = b.block_params(entry).to_vec();
        let r = build_expr(b, &expr, &params, &tm);
        b.ret(Some(r));
    });
    Some(fresh)
}

// ---------------------------------------------------------------------------
// Public API: the verified rewrite database (bet B5 → B4).
// ---------------------------------------------------------------------------

/// One proven rewrite `lhs ⇒ rhs`, with its recorded `z3rs` proof and costs.
///
/// `lhs`/`rhs` are functions of the owning [`RewriteDatabase`]'s [`Module`]
/// (resolve them with [`RewriteDatabase::lhs`]/[`RewriteDatabase::rhs`]). `proof`
/// is always [`RefinementResult::Refines`] — a rule is never admitted otherwise.
#[derive(Debug)]
pub struct RewriteRule {
    lhs: FuncId,
    rhs: FuncId,
    lhs_cost: u64,
    rhs_cost: u64,
    proof: RefinementResult,
}

impl RewriteRule {
    /// The left-hand (original) pattern function id.
    pub fn lhs_id(&self) -> FuncId {
        self.lhs
    }

    /// The right-hand (cheaper, proven-equivalent) function id.
    pub fn rhs_id(&self) -> FuncId {
        self.rhs
    }

    /// The cost of the left-hand side under the synthesizer's cost model.
    pub fn lhs_cost(&self) -> u64 {
        self.lhs_cost
    }

    /// The cost of the right-hand side (`< lhs_cost` for every admitted rule).
    pub fn rhs_cost(&self) -> u64 {
        self.rhs_cost
    }

    /// The recorded refinement proof (always [`RefinementResult::Refines`]).
    pub fn proof(&self) -> &RefinementResult {
        &self.proof
    }
}

/// A growing collection of proven `(lhs, rhs)` rewrite rules — the verified
/// database bet B5 produces for the e-graph (bet B4) to consume. It owns its own
/// [`Module`] holding every rule's `lhs`/`rhs` functions against shared interning
/// tables, so the database is self-contained and portable.
#[derive(Debug)]
pub struct RewriteDatabase {
    module: Module,
    syms: StrInterner,
    rules: Vec<RewriteRule>,
}

impl RewriteDatabase {
    /// An empty database.
    fn new() -> Self {
        RewriteDatabase {
            module: Module::new("superopt-rules"),
            syms: StrInterner::new(),
            rules: Vec::new(),
        }
    }

    /// The module owning the rule functions (and their shared tables).
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// The proven rewrite rules, in discovery order.
    pub fn rules(&self) -> &[RewriteRule] {
        &self.rules
    }

    /// The left-hand-side function of a rule.
    pub fn lhs(&self, rule: &RewriteRule) -> &Function {
        self.module.function(rule.lhs)
    }

    /// The right-hand-side function of a rule.
    pub fn rhs(&self, rule: &RewriteRule) -> &Function {
        self.module.function(rule.rhs)
    }

    /// The number of rules in the database.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

/// Run the synthesizer over each seed function of `module` and collect the proven
/// rewrite rules into a fresh [`RewriteDatabase`]. A seed outside the subset, or
/// with no strictly-cheaper proven equivalent, contributes no rule. Each rule is
/// **independently re-verified** with [`check_refinement`] before admission, so
/// the database contains only refinements a `z3rs` proof licenses.
pub fn discover_rules(module: &Module, funcs: &[FuncId], budget: &Budget) -> RewriteDatabase {
    let mut db = RewriteDatabase::new();
    for &f in funcs {
        let Some(spec) = Spec::from_function(module.types(), module.consts(), module.function(f))
        else {
            continue;
        };
        let Some(found) = synthesize(&spec, budget, true) else {
            continue;
        };
        if !found.proof.is_refines() {
            continue;
        }

        let n = db.rules.len();
        let lhs_name = format!("lhs{n}");
        let rhs_name = format!("rhs{n}");
        let lhs = build_fn_from_expr(
            &mut db.module,
            &mut db.syms,
            &lhs_name,
            &spec.param_widths,
            spec.ret_width,
            &spec.target,
        );
        let rhs = build_fn_from_expr(
            &mut db.module,
            &mut db.syms,
            &rhs_name,
            &spec.param_widths,
            spec.ret_width,
            &found.expr,
        );
        // Independent re-verification against the database's own tables.
        let proof = check_refinement(
            db.module.types(),
            db.module.consts(),
            db.module.function(lhs),
            db.module.function(rhs),
        );
        if !proof.is_refines() {
            continue;
        }
        db.rules.push(RewriteRule {
            lhs,
            rhs,
            lhs_cost: cost_of(&spec.target),
            rhs_cost: found.cost,
            proof,
        });
    }
    db
}

#[cfg(test)]
mod tests;
