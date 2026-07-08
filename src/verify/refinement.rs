//! The **`Refinement`** verification tier (bet **B2**, tenet **T3**): every
//! optimization is a *checked refinement*.
//!
//! This module is the refinement checker. It takes a rewrite `src ⇒ tgt` — two
//! functions with the same signature, each an **acyclic** (loop-free) region of
//! **pure** value-producing ops and control flow (`ret`/`br`/`cond_br`/`switch`/
//! `unreachable`) — encodes both into SMT-LIB2 **QF_BV**, and asks
//! [`z3rs`](crate::verify::smt)
//! whether `tgt` refines `src`. `unsat` of the negated obligation means the
//! rewrite is a sound refinement; `sat` yields a counterexample input; anything
//! else (`unknown`, a solver error, an unsupported construct) is reported as a
//! *sound non-answer* ([`RefinementResult::Unknown`]) and **never** mistaken for
//! a proof (per the tiers in `docs/design-tenets.md` §2).
//!
//! ## The value model: a bit-vector paired with a poison bit
//!
//! Each SSA integer value is encoded as a pair of SMT terms that exactly mirror
//! the concrete [`SemValue`](crate::ir::SemValue) model of
//! [`crate::ir::semantics`]:
//!
//! - `val : (_ BitVec N)` — the `N`-bit two's-complement bit pattern, and
//! - `poison : Bool` — whether the value is [poison](crate::ir::Const::Poison).
//!
//! Every opcode contributes a **value relation** (the bit-vector it computes)
//! and a **poison relation**. Poison propagates from any operand
//! (`res_poison = op0_poison ∨ op1_poison ∨ …`), plus each set flag adds a
//! *flag-violation* disjunct encoded in QF_BV to match `ir::semantics` exactly:
//!
//! - `nsw` / `nuw` overflow of `add`/`sub`/`mul`/`shl` — detected by redoing the
//!   op in a widened bit-vector (sign- or zero-extended) and checking the exact
//!   result differs from the (sign/zero)-extension of the `N`-bit wrapped
//!   result: i.e. the reduction modulo `2^N` lost information.
//! - `exact` on `udiv`/`sdiv`/`lshr`/`ashr` — a nonzero remainder / a
//!   shifted-out set bit, tested with a shift/round-trip identity.
//! - over-wide shift (`amount ≥ N`) — a `bvuge` against the width.
//!
//! `select` and `freeze` have the bespoke poison rules of the reference
//! semantics: a poison *condition* poisons `select` but a non-selected poison
//! arm does not; `freeze` clears poison, mapping a poison operand to a *fresh
//! unconstrained* bit-vector (so a frozen poison may be **any** value — which is
//! exactly what makes replacing a defined value by `freeze(poison)` fail
//! refinement).
//!
//! ## UB as a precondition
//!
//! Division/remainder by zero and `sdiv`/`srem` of `INT_MIN` by `-1` are the
//! only **undefined behavior** in this subset (`docs/ir-design.md` §10). Per the
//! refinement contract, the obligation is *conditional on the source being
//! UB-free*, so each `div`/`rem` in `src` contributes a UB condition that the
//! query **assumes does not occur**; symmetrically, a `div`/`rem` in `tgt`
//! contributes a UB condition that, if reachable, *breaks* refinement (the
//! target must not introduce UB the source ruled out). UB fires only on
//! non-poison operands, matching the poison-before-UB order of `ir::semantics`.
//!
//! ## The refinement relation
//!
//! With `s = src_ret` and `t = tgt_ret` (same width), a single value refines
//! per `docs/ir-design.md` §5:
//!
//! ```text
//! refines(s, t)  ≡  s.poison ∨ (¬t.poison ∧ s.val = t.val)
//! ```
//!
//! i.e. a poison source licenses any target, a defined source pins the target.
//! The whole obligation is
//!
//! ```text
//! ∀ inputs.  ¬src_ub  ⇒  ( ¬tgt_ub ∧ refines(src_ret, tgt_ret) )
//! ```
//!
//! and we hand the solver its **negation**; `unsat` ⇒ the rewrite is sound.
//!
//! ## Multi-block (acyclic) control flow
//!
//! Beyond the single-block subset, an entire **DAG** function is encoded (loops —
//! any back-edge — are detected and reported as `Unknown`, never encoded):
//!
//! - **Reachability guards.** Each block `b` gets a boolean `reach_b`
//!   (`reach_entry = true`; otherwise the disjunction of its incoming edges). An
//!   edge `s → t` is *taken* iff `reach_s` holds and the terminator's local guard
//!   does: `br` is unconditional; `cond_br` takes the true edge iff the condition
//!   bit is `1`; `switch` takes the matching case, else the default. Branching or
//!   switching on a **poison** condition, and *reaching* an `unreachable`, are
//!   **UB** (guarded by `reach_b`) — matching `ir::semantics`.
//! - **Block parameters as merges.** A non-entry block parameter's `(val,poison)`
//!   is a deterministic `ite` chain over its incoming taken edges (our SSA phi):
//!   exactly one edge is taken on any concrete path, so the chain reads that
//!   edge's argument. Entry parameters are the shared symbolic inputs.
//! - **Per-block effect.** Instructions encode as in the single-block case, but a
//!   block's UB conditions are guarded by `reach_b` (an unreachable op cannot
//!   fault); unreachable values never flow into the result or a live merge.
//! - **The result.** A value-returning function may `ret` from several blocks; the
//!   observed result is the `ite` over the reachable `ret` blocks (exactly one is
//!   reachable on any path).
//!
//! ## Scope
//!
//! In scope: integer (`Int(N)`) pure ops — `add`/`sub`/`mul`, `udiv`/`sdiv`/
//! `urem`/`srem`, `and`/`or`/`xor`, `shl`/`lshr`/`ashr`, `icmp`, `select`,
//! `freeze`, and the integer casts `trunc`/`zext`/`sext`; all their flags; plus
//! acyclic control flow with block-argument merges. Out of scope (cleanly
//! reported as [`RefinementResult::Unknown`], never a false `Refines`): floating
//! point, pointers/memory, `call`, **loops** (back-edges), and non-integer casts.

use std::collections::HashMap;

use crate::ir::inst::{BinOp, CastOp, Flags, InstData, InstKind, IntPred};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ConstPool, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Module};

use super::smt::z3rs;

/// The verdict of a refinement check.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RefinementResult {
    /// `tgt` provably refines `src`: the rewrite is sound (`unsat` of the
    /// negated obligation).
    Refines,
    /// The rewrite is **unsound**; the string is the solver's model of an input
    /// that distinguishes the two functions (`sat`).
    Counterexample(String),
    /// No proof was produced — an `unknown` from the solver, a solver error, or
    /// an out-of-scope construct. Per the verification tiers this is a *sound
    /// non-answer*: it is never treated as a proof of refinement.
    Unknown(String),
}

impl RefinementResult {
    /// Whether this result is a proof of refinement.
    pub fn is_refines(&self) -> bool {
        matches!(self, RefinementResult::Refines)
    }
}

/// The `Refinement`-tier entry point the driver (`lf-opt`) calls to discharge a
/// rewrite obligation over two functions of a [`Module`].
///
/// This is the thin, module-facing wrapper around [`check_refinement`]: it
/// resolves the two [`FuncId`]s against the module's shared type/constant tables
/// and runs the check.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefinementTier;

impl RefinementTier {
    /// Check that `tgt` refines `src` (both functions of `module`).
    pub fn check(self, module: &Module, src: FuncId, tgt: FuncId) -> RefinementResult {
        check_refinement(module.types(), module.consts(), module.function(src), module.function(tgt))
    }
}

/// Check that `tgt` **refines** `src` for the single-block pure-integer subset,
/// discharging the obligation through `z3rs`.
///
/// `types` and `consts` are the shared interning tables the two functions were
/// built against (from their owning [`Module`]). Returns [`RefinementResult`];
/// anything outside the supported subset yields [`RefinementResult::Unknown`],
/// never a false [`RefinementResult::Refines`].
pub fn check_refinement(
    types: &TypeContext,
    consts: &ConstPool,
    src: &Function,
    tgt: &Function,
) -> RefinementResult {
    match build_query(types, consts, src, tgt) {
        Ok(query) => decide(&query.script, &query.input_names),
        Err(EncErr::Unsupported(reason)) => {
            RefinementResult::Unknown(format!("unsupported: {reason}"))
        }
        Err(EncErr::Loops) => RefinementResult::Unknown("loops unsupported".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Errors and small value types.
// ---------------------------------------------------------------------------

/// Why encoding could not proceed (a *sound non-answer*, never a false proof).
#[derive(Clone, Debug)]
enum EncErr {
    /// An out-of-scope construct was encountered while encoding.
    Unsupported(String),
    /// The function is not a DAG (it has a back-edge / loop). Unbounded loops are
    /// deliberately not encoded; the check reports [`RefinementResult::Unknown`].
    Loops,
}

fn unsupported(what: impl Into<String>) -> EncErr {
    EncErr::Unsupported(what.into())
}

/// The encoded `(value, poison)` SMT terms of one SSA value, plus its width.
#[derive(Clone)]
struct Sym {
    /// An SMT term of sort `(_ BitVec width)` — the bit pattern.
    val: String,
    /// An SMT term of sort `Bool` — whether the value is poison.
    poison: String,
    /// The bit width `N`.
    width: u32,
}

/// A fully built SMT-LIB2 query plus the shared input names (for `get-value`).
struct Query {
    script: String,
    input_names: Vec<String>,
}

/// The encoding of one function: its return value and its UB conditions.
struct EncodedFn {
    ret: Sym,
    ub: Vec<String>,
}

// ---------------------------------------------------------------------------
// Query construction.
// ---------------------------------------------------------------------------

fn build_query(
    types: &TypeContext,
    consts: &ConstPool,
    src: &Function,
    tgt: &Function,
) -> Result<Query, EncErr> {
    let src_params = entry_params(src)?;
    let tgt_params = entry_params(tgt)?;

    // Signatures must match: same parameter widths and same return width. We
    // compare the concrete entry-parameter types (the function's parameters).
    if src_params.len() != tgt_params.len() {
        return Err(unsupported("parameter count mismatch between src and tgt"));
    }

    let mut enc = Enc { types, consts, out: String::new(), fresh: 0 };
    enc.line("(set-logic QF_BV)");

    // Shared symbolic inputs: one bit-vector + one poison bool per parameter,
    // universally quantified (they are free constants in the check-sat).
    let mut inputs = Vec::with_capacity(src_params.len());
    let mut input_names = Vec::new();
    for (i, &p) in src_params.iter().enumerate() {
        let w = int_width(types, src.value_type(p))
            .ok_or_else(|| unsupported("non-integer parameter"))?;
        let tw = int_width(types, tgt.value_type(tgt_params[i]))
            .ok_or_else(|| unsupported("non-integer parameter"))?;
        if w != tw {
            return Err(unsupported("parameter width mismatch between src and tgt"));
        }
        let vn = format!("in{i}");
        let pn = format!("in{i}_p");
        enc.declare_bv(&vn, w);
        enc.declare_bool(&pn);
        inputs.push(Sym { val: vn.clone(), poison: pn.clone(), width: w });
        input_names.push(vn);
        input_names.push(pn);
    }

    let src_enc = enc.encode_function(src, "s", &inputs)?;
    let tgt_enc = enc.encode_function(tgt, "t", &inputs)?;

    if src_enc.ret.width != tgt_enc.ret.width {
        return Err(unsupported("return width mismatch between src and tgt"));
    }

    // refines(s, t) ≡ s.poison ∨ (¬t.poison ∧ s.val = t.val)
    let refines = format!(
        "(or {sp} (and (not {tp}) (= {sv} {tv})))",
        sp = src_enc.ret.poison,
        tp = tgt_enc.ret.poison,
        sv = src_enc.ret.val,
        tv = tgt_enc.ret.val,
    );
    let src_ub = or_terms(&src_enc.ub);
    let tgt_ub = or_terms(&tgt_enc.ub);

    // Negated obligation: src is UB-free, yet tgt is UB or fails to refine.
    enc.assert(&format!("(and (not {src_ub}) (or {tgt_ub} (not {refines})))"));
    enc.line("(check-sat)");

    Ok(Query { script: enc.out, input_names })
}

/// Disjoin a list of boolean terms, collapsing to `false`/the single term.
fn or_terms(terms: &[String]) -> String {
    match terms {
        [] => "false".to_string(),
        [only] => only.clone(),
        many => format!("(or {})", many.join(" ")),
    }
}

/// The function's parameters: its entry block's parameter list.
fn entry_params(func: &Function) -> Result<Vec<ValueId>, EncErr> {
    let entry = func.entry().ok_or_else(|| unsupported("function has no body"))?;
    Ok(func.block(entry).params().to_vec())
}

/// The integer width of a function's return type, or `None` if non-integer.
fn func_ret_width(types: &TypeContext, func: &Function) -> Option<u32> {
    match types.get(func.sig) {
        Type::Func(ft) => int_width(types, ft.ret),
        _ => None,
    }
}

/// A reverse-postorder over the blocks **reachable** from the entry (so every
/// predecessor precedes its successors — a valid topological order for a DAG),
/// or [`EncErr::Loops`] if a back-edge is found (the function is not acyclic).
///
/// Uses an iterative colored DFS: a successor still on the DFS stack (gray) marks
/// a back-edge. Successor targets out of range are dropped (matching the CFG
/// builder's tolerance; the structural verifier rejects them earlier).
fn reverse_postorder(func: &Function) -> Result<Vec<BlockId>, EncErr> {
    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White,
        Gray,
        Black,
    }
    let n = func.block_count();
    let entry = func.entry().ok_or_else(|| unsupported("function has no body"))?.index();
    let mut color = vec![Color::White; n];
    let mut post = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
    color[entry] = Color::Gray;
    while let Some(&(node, idx)) = stack.last() {
        let succ: Vec<usize> = match func.block(BlockId::from_index(node)).terminator() {
            Some(t) => func.inst(t).successors().into_iter().map(BlockId::index).filter(|&s| s < n).collect(),
            None => return Err(unsupported("block is not terminated")),
        };
        if idx < succ.len() {
            stack.last_mut().expect("nonempty").1 = idx + 1;
            let next = succ[idx];
            match color[next] {
                Color::White => {
                    color[next] = Color::Gray;
                    stack.push((next, 0));
                }
                Color::Gray => return Err(EncErr::Loops),
                Color::Black => {}
            }
        } else {
            color[node] = Color::Black;
            post.push(node);
            stack.pop();
        }
    }
    Ok(post.into_iter().rev().map(BlockId::from_index).collect())
}

/// One incoming CFG edge into a block: the boolean term for "this edge is taken"
/// and the block arguments it supplies (matching the target's parameters).
#[derive(Clone)]
struct Edge {
    /// SMT `Bool` term: `reach_src ∧ (the terminator's local guard)`.
    taken: String,
    /// The values passed on this edge, positionally matching the target's params.
    args: Vec<ValueId>,
}

// ---------------------------------------------------------------------------
// The encoder.
// ---------------------------------------------------------------------------

/// Accumulates one SMT-LIB2 script across both functions, minting fresh names.
struct Enc<'a> {
    types: &'a TypeContext,
    consts: &'a ConstPool,
    out: String,
    fresh: u32,
}

impl Enc<'_> {
    fn line(&mut self, s: &str) {
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn declare_bv(&mut self, name: &str, width: u32) {
        self.line(&format!("(declare-fun {name} () {})", bv_sort(width)));
    }

    fn declare_bool(&mut self, name: &str) {
        self.line(&format!("(declare-fun {name} () Bool)"));
    }

    fn assert(&mut self, term: &str) {
        self.line(&format!("(assert {term})"));
    }

    /// A fresh, unconstrained bit-vector constant of the given width.
    fn fresh_bv(&mut self, width: u32) -> String {
        let name = format!("fresh{}", self.fresh);
        self.fresh += 1;
        self.declare_bv(&name, width);
        name
    }

    /// Bind a result value to named `val`/`poison` symbols and assert their
    /// defining relations, returning the [`Sym`].
    fn bind(&mut self, prefix: &str, res: ValueId, width: u32, val: &str, poison: &str) -> Sym {
        let vn = format!("{prefix}v{}", res.index());
        let pn = format!("{prefix}p{}", res.index());
        self.declare_bv(&vn, width);
        self.declare_bool(&pn);
        self.assert(&format!("(= {vn} {val})"));
        self.assert(&format!("(= {pn} {poison})"));
        Sym { val: vn, poison: pn, width }
    }

    /// Encode a whole **acyclic** function over the shared `inputs`: reachability
    /// guards, block-parameter merges, per-block pure ops, and the multi-`ret`
    /// result. Returns [`EncErr::Loops`] if the function has a back-edge.
    fn encode_function(
        &mut self,
        func: &Function,
        prefix: &str,
        inputs: &[Sym],
    ) -> Result<EncodedFn, EncErr> {
        let entry = func.entry().ok_or_else(|| unsupported("function has no body"))?;
        // Reachable blocks in topological order; also our loop check.
        let order = reverse_postorder(func)?;
        let mut reachable = vec![false; func.block_count()];
        for &b in &order {
            reachable[b.index()] = true;
        }

        let mut map: HashMap<ValueId, Sym> = HashMap::new();
        let mut ub: Vec<String> = Vec::new();
        // Incoming edges collected as predecessors are processed (topo order
        // guarantees every predecessor is handled before its target).
        let mut incoming: HashMap<usize, Vec<Edge>> = HashMap::new();
        // The (reach, value) of each `ret` block, in encounter order.
        let mut rets: Vec<(String, Sym)> = Vec::new();

        for &b in &order {
            let bi = b.index();
            let block = func.block(b);
            let edges = incoming.get(&bi).cloned().unwrap_or_default();

            // 1. Reachability guard for this block.
            let reach = self.bind_reach(prefix, bi, bi == entry.index(), &edges);

            // 2. Block parameters: entry params are the inputs; others are merges.
            if bi == entry.index() {
                for (i, &p) in block.params().iter().enumerate() {
                    map.insert(p, inputs[i].clone());
                }
            } else {
                for (j, &p) in block.params().iter().enumerate() {
                    let sym = self.merge_param(func, &mut map, prefix, p, j, &edges)?;
                    map.insert(p, sym);
                }
            }

            // 3. The block's pure instructions (UB guarded by reachability).
            for &inst_id in block.insts() {
                let inst = func.inst(inst_id);
                if let Some(res) = inst.result() {
                    let sym = self.encode_inst(func, prefix, inst, res, &mut map, &mut ub, &reach)?;
                    map.insert(res, sym);
                }
            }

            // 4. The terminator: outgoing edges, branch/switch UB, and `ret`s.
            let term_id = block.terminator().ok_or_else(|| unsupported("block is not terminated"))?;
            let term = func.inst(term_id).clone();
            self.encode_terminator(
                func, &term, &reach, &mut map, &mut incoming, &mut ub, &mut rets, &reachable,
            )?;
        }

        let ret = self.merge_rets(func, prefix, &rets)?;
        Ok(EncodedFn { ret, ub })
    }

    /// Declare and define block `bi`'s reachability boolean, returning its name.
    /// The entry is always reachable; any other block is reachable iff some
    /// incoming edge is taken.
    fn bind_reach(&mut self, prefix: &str, bi: usize, is_entry: bool, edges: &[Edge]) -> String {
        let name = format!("{prefix}reach{bi}");
        self.declare_bool(&name);
        let def = if is_entry {
            "true".to_string()
        } else {
            or_terms(&edges.iter().map(|e| e.taken.clone()).collect::<Vec<_>>())
        };
        self.assert(&format!("(= {name} {def})"));
        name
    }

    /// Merge the `j`-th block parameter across its `incoming` edges as a
    /// deterministic `ite` chain (an SSA phi): on any concrete path exactly one
    /// edge is taken, so the chain reads that edge's argument. A parameter of an
    /// (effectively) unreachable block with no encoded predecessors is bound to a
    /// fresh unconstrained value that never reaches the observable result.
    fn merge_param(
        &mut self,
        func: &Function,
        map: &mut HashMap<ValueId, Sym>,
        prefix: &str,
        p: ValueId,
        j: usize,
        edges: &[Edge],
    ) -> Result<Sym, EncErr> {
        let w = int_width(self.types, func.value_type(p))
            .ok_or_else(|| unsupported("non-integer block parameter"))?;
        let vn = format!("{prefix}bv{}", p.index());
        let pn = format!("{prefix}bp{}", p.index());
        self.declare_bv(&vn, w);
        self.declare_bool(&pn);
        if edges.is_empty() {
            let fresh = self.fresh_bv(w);
            self.assert(&format!("(= {vn} {fresh})"));
            self.assert(&format!("(= {pn} false)"));
            return Ok(Sym { val: vn, poison: pn, width: w });
        }
        // Resolve each edge's j-th argument to a Sym, keeping the edge's guard.
        let mut arms: Vec<(String, Sym)> = Vec::with_capacity(edges.len());
        for e in edges {
            let a = *e.args.get(j).ok_or_else(|| unsupported("edge argument arity mismatch"))?;
            let sym = self.resolve(func, map, a)?;
            arms.push((e.taken.clone(), sym));
        }
        let (val, poison) = ite_chain(&arms);
        self.assert(&format!("(= {vn} {val})"));
        self.assert(&format!("(= {pn} {poison})"));
        Ok(Sym { val: vn, poison: pn, width: w })
    }

    /// Encode a terminator: register its outgoing (taken, args) edges on their
    /// targets, contribute branch/switch/unreachable **UB** (guarded by `reach`),
    /// and record `ret` blocks. Edges into (statically) unreachable targets are
    /// dropped — their guard would be `false` anyway.
    #[allow(clippy::too_many_arguments)]
    fn encode_terminator(
        &mut self,
        func: &Function,
        term: &InstData,
        reach: &str,
        map: &mut HashMap<ValueId, Sym>,
        incoming: &mut HashMap<usize, Vec<Edge>>,
        ub: &mut Vec<String>,
        rets: &mut Vec<(String, Sym)>,
        reachable: &[bool],
    ) -> Result<(), EncErr> {
        let mut add_edge = |target: BlockId, taken: String, args: Vec<ValueId>| {
            if reachable.get(target.index()).copied().unwrap_or(false) {
                incoming.entry(target.index()).or_default().push(Edge { taken, args });
            }
        };
        match &term.kind {
            InstKind::Ret => {
                let ops = term.operands();
                match ops.len() {
                    1 => {
                        let sym = self.resolve(func, map, ops[0])?;
                        rets.push((reach.to_string(), sym));
                    }
                    0 => return Err(unsupported("void `ret` (no value to refine)")),
                    _ => return Err(unsupported("`ret` with multiple operands")),
                }
            }
            InstKind::Br(target) => {
                add_edge(*target, reach.to_string(), term.operands().to_vec());
            }
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let ops = term.operands();
                let cond = self.resolve(func, map, ops[0])?;
                let nt = *true_args as usize;
                let nf = *false_args as usize;
                if 1 + nt + nf != ops.len() {
                    return Err(unsupported("cond_br argument arity mismatch"));
                }
                let true_args_v = ops[1..1 + nt].to_vec();
                let false_args_v = ops[1 + nt..].to_vec();
                let cond_true = format!("(= {} (_ bv1 1))", cond.val);
                // Branching on poison is UB (only where this block is reached).
                ub.push(format!("(and {reach} {})", cond.poison));
                add_edge(*if_true, format!("(and {reach} {cond_true})"), true_args_v);
                add_edge(*if_false, format!("(and {reach} (not {cond_true}))"), false_args_v);
            }
            InstKind::Switch(data) => {
                let ops = term.operands();
                let cond = self.resolve(func, map, ops[0])?;
                let w = cond.width;
                // Operand layout: [cond, <default args>, <case0 args>, ...].
                let mut off = 1usize;
                let da = data.default_args as usize;
                if off + da > ops.len() {
                    return Err(unsupported("switch default argument arity mismatch"));
                }
                let default_args = ops[off..off + da].to_vec();
                off += da;
                // Switching on poison is UB.
                ub.push(format!("(and {reach} {})", cond.poison));
                let mut case_eqs: Vec<String> = Vec::with_capacity(data.cases.len());
                for case in &data.cases {
                    let ca = case.args as usize;
                    if off + ca > ops.len() {
                        return Err(unsupported("switch case argument arity mismatch"));
                    }
                    let args = ops[off..off + ca].to_vec();
                    off += ca;
                    let eq = format!("(= {} {})", cond.val, bv_lit(&case.value, w));
                    add_edge(case.target, format!("(and {reach} {eq})"), args);
                    case_eqs.push(eq);
                }
                let none = if case_eqs.is_empty() {
                    "true".to_string()
                } else {
                    format!("(not (or {}))", case_eqs.join(" "))
                };
                add_edge(data.default, format!("(and {reach} {none})"), default_args);
            }
            // Reaching `unreachable` at runtime is UB (asserts the path is dead).
            InstKind::Unreachable => ub.push(reach.to_string()),
            _ => return Err(unsupported("non-terminator in the terminator slot")),
        }
        Ok(())
    }

    /// The observed function result: an `ite` over the reachable `ret` blocks
    /// (exactly one is reachable on any concrete path). With no `ret` block every
    /// path is UB or non-returning, so the (never-observed) result is fresh.
    fn merge_rets(
        &mut self,
        func: &Function,
        prefix: &str,
        rets: &[(String, Sym)],
    ) -> Result<Sym, EncErr> {
        let vn = format!("{prefix}ret_v");
        let pn = format!("{prefix}ret_p");
        if rets.is_empty() {
            let w = func_ret_width(self.types, func)
                .ok_or_else(|| unsupported("no `ret` and non-integer return type"))?;
            self.declare_bv(&vn, w);
            self.declare_bool(&pn);
            let fresh = self.fresh_bv(w);
            self.assert(&format!("(= {vn} {fresh})"));
            self.assert(&format!("(= {pn} false)"));
            return Ok(Sym { val: vn, poison: pn, width: w });
        }
        let w = rets[0].1.width;
        for (_, s) in rets {
            if s.width != w {
                return Err(unsupported("`ret` blocks disagree on return width"));
            }
        }
        self.declare_bv(&vn, w);
        self.declare_bool(&pn);
        let (val, poison) = ite_chain(rets);
        self.assert(&format!("(= {vn} {val})"));
        self.assert(&format!("(= {pn} {poison})"));
        Ok(Sym { val: vn, poison: pn, width: w })
    }

    /// Resolve an operand to its [`Sym`], materializing (and caching) constants.
    fn resolve(
        &mut self,
        func: &Function,
        map: &mut HashMap<ValueId, Sym>,
        v: ValueId,
    ) -> Result<Sym, EncErr> {
        if let Some(s) = map.get(&v) {
            return Ok(s.clone());
        }
        let sym = match func.value(v).def {
            ValueDef::Const(cid) => {
                let c = self.consts.get(cid).clone();
                self.encode_const(&c)?
            }
            // Params are pre-registered and instruction results are inserted as
            // they are encoded, so anything else here is out of scope.
            _ => return Err(unsupported("value is not a constant, parameter, or pure result")),
        };
        map.insert(v, sym.clone());
        Ok(sym)
    }

    /// Encode a constant into a [`Sym`].
    fn encode_const(&mut self, c: &Const) -> Result<Sym, EncErr> {
        match c {
            Const::Int { ty, value } => {
                let w = int_width(self.types, *ty)
                    .ok_or_else(|| unsupported("non-integer integer-constant type"))?;
                Ok(Sym { val: bv_lit(value, w), poison: "false".to_string(), width: w })
            }
            Const::Poison(ty) => {
                let w = int_width(self.types, *ty)
                    .ok_or_else(|| unsupported("poison constant of a non-integer type"))?;
                // Poison of an integer type: a fresh unconstrained value, marked
                // poison. (Its bit pattern is never observed unless frozen.)
                let fv = self.fresh_bv(w);
                Ok(Sym { val: fv, poison: "true".to_string(), width: w })
            }
            Const::Float { .. } => Err(unsupported("floating-point constant")),
            Const::Null(_) => Err(unsupported("pointer constant")),
            Const::Aggregate { .. } => Err(unsupported("aggregate constant")),
        }
    }

    /// Encode one value-producing instruction, returning its result [`Sym`].
    ///
    /// Any undefined-behavior condition the op contributes is guarded by `reach`:
    /// an op in an unreachable block cannot fault (its effect is not observed).
    #[allow(clippy::too_many_arguments)]
    fn encode_inst(
        &mut self,
        func: &Function,
        prefix: &str,
        inst: &InstData,
        res: ValueId,
        map: &mut HashMap<ValueId, Sym>,
        ub: &mut Vec<String>,
        reach: &str,
    ) -> Result<Sym, EncErr> {
        let mut ops = Vec::with_capacity(inst.operands().len());
        for &o in inst.operands() {
            ops.push(self.resolve(func, map, o)?);
        }
        let rw = int_width(self.types, inst.ty)
            .ok_or_else(|| unsupported("non-integer result type"))?;

        // UB conditions are collected locally, then guarded by `reach`.
        let mut inst_ub: Vec<String> = Vec::new();
        let (val, poison) = match &inst.kind {
            InstKind::Bin(op) => enc_bin(*op, &inst.flags, &ops, rw, &mut inst_ub)?,
            InstKind::ICmp(pred) => enc_icmp(*pred, &ops)?,
            InstKind::Select => enc_select(&ops)?,
            InstKind::Cast(op) => enc_cast(*op, &ops, rw)?,
            InstKind::Freeze => {
                let a = &ops[0];
                let fresh = self.fresh_bv(rw);
                (format!("(ite {} {fresh} {})", a.poison, a.val), "false".to_string())
            }
            InstKind::Unary(_) => return Err(unsupported("float unary op (fneg)")),
            InstKind::FCmp(_) => return Err(unsupported("float comparison (fcmp)")),
            InstKind::PtrAdd { .. } => return Err(unsupported("pointer arithmetic (ptr_add)")),
            InstKind::Alloca { .. }
            | InstKind::DynAlloca { .. }
            | InstKind::Load { .. }
            | InstKind::Store { .. }
            | InstKind::Call => return Err(unsupported("memory / call op")),
            InstKind::Ret
            | InstKind::Br(_)
            | InstKind::CondBr { .. }
            | InstKind::Switch(_)
            | InstKind::Unreachable => return Err(unsupported("terminator in the body")),
        };
        for cond in inst_ub {
            ub.push(format!("(and {reach} {cond})"));
        }
        Ok(self.bind(prefix, res, rw, &val, &poison))
    }
}

/// Build the `(val, poison)` `ite` chain that reads the arm whose guard holds,
/// deterministically folding from the last arm (its default) backwards. Exactly
/// one guard is true on any concrete path, so the choice is unambiguous there.
fn ite_chain(arms: &[(String, Sym)]) -> (String, String) {
    let (_, last) = arms.last().expect("ite_chain requires at least one arm");
    let mut val = last.val.clone();
    let mut poison = last.poison.clone();
    for (guard, sym) in arms[..arms.len() - 1].iter().rev() {
        val = format!("(ite {guard} {} {val})", sym.val);
        poison = format!("(ite {guard} {} {poison})", sym.poison);
    }
    (val, poison)
}

// ---------------------------------------------------------------------------
// Per-opcode encodings (value term, poison term).
// ---------------------------------------------------------------------------

/// Binary integer op. Pushes any UB condition onto `ub`; returns `(val, poison)`.
fn enc_bin(
    op: BinOp,
    flags: &Flags,
    ops: &[Sym],
    w: u32,
    ub: &mut Vec<String>,
) -> Result<(String, String), EncErr> {
    if op.is_float() {
        return Err(unsupported("floating-point arithmetic"));
    }
    let a = &ops[0];
    let b = &ops[1];
    let mut poison = vec![a.poison.clone(), b.poison.clone()];

    let val = match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul => {
            let (opc, ext) = match op {
                BinOp::Add => ("bvadd", 1),
                BinOp::Sub => ("bvsub", 1),
                BinOp::Mul => ("bvmul", w),
                _ => unreachable!(),
            };
            let res = format!("({opc} {} {})", a.val, b.val);
            if flags.nsw {
                poison.push(ext_overflow(opc, &a.val, &b.val, &res, ext, true));
            }
            if flags.nuw {
                poison.push(ext_overflow(opc, &a.val, &b.val, &res, ext, false));
            }
            res
        }
        BinOp::And => format!("(bvand {} {})", a.val, b.val),
        BinOp::Or => format!("(bvor {} {})", a.val, b.val),
        BinOp::Xor => format!("(bvxor {} {})", a.val, b.val),
        BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => {
            let signed = matches!(op, BinOp::SDiv | BinOp::SRem);
            let zero = bv_zero(w);
            let div_zero = format!("(= {} {zero})", b.val);
            let cond = if signed {
                let int_min = bv_int_min(w);
                let minus_one = bv_all_ones(w);
                format!(
                    "(and (not {ap}) (not {bp}) (or {div_zero} (and (= {av} {int_min}) (= {bv} {minus_one}))))",
                    ap = a.poison,
                    bp = b.poison,
                    av = a.val,
                    bv = b.val,
                )
            } else {
                format!("(and (not {ap}) (not {bp}) {div_zero})", ap = a.poison, bp = b.poison)
            };
            ub.push(cond);
            if flags.exact {
                // `exact` is meaningful only on udiv/sdiv: a nonzero remainder
                // is poison.
                let rem = match op {
                    BinOp::UDiv => Some(format!("(bvurem {} {})", a.val, b.val)),
                    BinOp::SDiv => Some(format!("(bvsrem {} {})", a.val, b.val)),
                    _ => None,
                };
                if let Some(rem) = rem {
                    poison.push(format!("(distinct {rem} {zero})"));
                }
            }
            match op {
                BinOp::UDiv => format!("(bvudiv {} {})", a.val, b.val),
                BinOp::SDiv => format!("(bvsdiv {} {})", a.val, b.val),
                BinOp::URem => format!("(bvurem {} {})", a.val, b.val),
                BinOp::SRem => format!("(bvsrem {} {})", a.val, b.val),
                _ => unreachable!(),
            }
        }
        BinOp::Shl | BinOp::LShr | BinOp::AShr => {
            let width_lit = bv_lit(&puremp::Int::from_u64(u64::from(w)), w);
            poison.push(format!("(bvuge {} {width_lit})", b.val));
            match op {
                BinOp::Shl => {
                    let res = format!("(bvshl {} {})", a.val, b.val);
                    if flags.nuw {
                        // No unsigned wrap: shifting back recovers the input.
                        poison.push(format!("(distinct (bvlshr {res} {}) {})", b.val, a.val));
                    }
                    if flags.nsw {
                        // No signed wrap: arithmetic round-trip recovers the input.
                        poison.push(format!("(distinct (bvashr {res} {}) {})", b.val, a.val));
                    }
                    res
                }
                BinOp::LShr => {
                    let res = format!("(bvlshr {} {})", a.val, b.val);
                    if flags.exact {
                        poison.push(format!("(distinct (bvshl {res} {}) {})", b.val, a.val));
                    }
                    res
                }
                BinOp::AShr => {
                    let res = format!("(bvashr {} {})", a.val, b.val);
                    if flags.exact {
                        poison.push(format!("(distinct (bvshl {res} {}) {})", b.val, a.val));
                    }
                    res
                }
                _ => unreachable!(),
            }
        }
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
            return Err(unsupported("floating-point arithmetic"));
        }
    };
    Ok((val, or_terms(&poison)))
}

/// Redo `opc` in a widened bit-vector (sign/zero-extended by `ext` bits) and
/// report whether reducing modulo `2^N` lost information — i.e. the flagged
/// wrap actually occurred. Matches the exact-then-reduce check in `ir::semantics`.
fn ext_overflow(opc: &str, a: &str, b: &str, wrapped: &str, ext: u32, signed: bool) -> String {
    let extend = if signed { "sign_extend" } else { "zero_extend" };
    let ea = format!("((_ {extend} {ext}) {a})");
    let eb = format!("((_ {extend} {ext}) {b})");
    let full = format!("({opc} {ea} {eb})");
    let wrapped_ext = format!("((_ {extend} {ext}) {wrapped})");
    format!("(distinct {full} {wrapped_ext})")
}

/// Integer comparison. Result is a 1-bit value; poison propagates from operands.
fn enc_icmp(pred: IntPred, ops: &[Sym]) -> Result<(String, String), EncErr> {
    let a = &ops[0];
    let b = &ops[1];
    let cmp = match pred {
        IntPred::Eq => format!("(= {} {})", a.val, b.val),
        IntPred::Ne => format!("(distinct {} {})", a.val, b.val),
        IntPred::Ugt => format!("(bvugt {} {})", a.val, b.val),
        IntPred::Uge => format!("(bvuge {} {})", a.val, b.val),
        IntPred::Ult => format!("(bvult {} {})", a.val, b.val),
        IntPred::Ule => format!("(bvule {} {})", a.val, b.val),
        IntPred::Sgt => format!("(bvsgt {} {})", a.val, b.val),
        IntPred::Sge => format!("(bvsge {} {})", a.val, b.val),
        IntPred::Slt => format!("(bvslt {} {})", a.val, b.val),
        IntPred::Sle => format!("(bvsle {} {})", a.val, b.val),
    };
    let val = format!("(ite {cmp} (_ bv1 1) (_ bv0 1))");
    Ok((val, or_terms(&[a.poison.clone(), b.poison.clone()])))
}

/// `select cond, t, f`. A poison condition poisons the result; a non-selected
/// poison arm does not (matches `ir::semantics::eval_select`).
fn enc_select(ops: &[Sym]) -> Result<(String, String), EncErr> {
    let cond = &ops[0];
    let t = &ops[1];
    let f = &ops[2];
    let cond_true = format!("(= {} (_ bv1 1))", cond.val);
    let val = format!("(ite {cond_true} {} {})", t.val, f.val);
    let poison =
        format!("(or {} (ite {cond_true} {} {}))", cond.poison, t.poison, f.poison);
    Ok((val, poison))
}

/// Integer casts `trunc`/`zext`/`sext`. Other casts are out of scope.
fn enc_cast(op: CastOp, ops: &[Sym], rw: u32) -> Result<(String, String), EncErr> {
    let a = &ops[0];
    let val = match op {
        CastOp::Trunc => {
            if rw > a.width {
                return Err(unsupported("trunc to a wider type"));
            }
            format!("((_ extract {} 0) {})", rw - 1, a.val)
        }
        CastOp::ZExt => {
            if rw < a.width {
                return Err(unsupported("zext to a narrower type"));
            }
            format!("((_ zero_extend {}) {})", rw - a.width, a.val)
        }
        CastOp::SExt => {
            if rw < a.width {
                return Err(unsupported("sext to a narrower type"));
            }
            format!("((_ sign_extend {}) {})", rw - a.width, a.val)
        }
        CastOp::FpTrunc
        | CastOp::FpExt
        | CastOp::FpToUi
        | CastOp::FpToSi
        | CastOp::UiToFp
        | CastOp::SiToFp => return Err(unsupported("floating-point cast")),
        CastOp::PtrToInt | CastOp::IntToPtr => return Err(unsupported("pointer cast")),
        CastOp::Bitcast => return Err(unsupported("bitcast")),
    };
    Ok((val, a.poison.clone()))
}

// ---------------------------------------------------------------------------
// SMT-LIB2 literal / sort helpers.
// ---------------------------------------------------------------------------

fn bv_sort(width: u32) -> String {
    format!("(_ BitVec {width})")
}

/// A bit-vector literal of the low `width` bits of `value` (its unsigned
/// representative in `[0, 2^width)`), as `(_ bvDEC width)`.
fn bv_lit(value: &puremp::Int, width: u32) -> String {
    let masked = value.mod_2k(width); // non-negative, `[0, 2^width)`
    format!("(_ bv{masked} {width})")
}

fn bv_zero(width: u32) -> String {
    format!("(_ bv0 {width})")
}

/// `INT_MIN` bit pattern for width `N`: `2^(N-1)` (top bit set).
fn bv_int_min(width: u32) -> String {
    let v = puremp::Int::ONE.mul_2k(width - 1);
    format!("(_ bv{v} {width})")
}

/// The all-ones pattern (`-1` / `2^N − 1`) for width `N`.
fn bv_all_ones(width: u32) -> String {
    let v = puremp::Int::ONE.mul_2k(width).sub(&puremp::Int::ONE);
    format!("(_ bv{v} {width})")
}

/// The integer width of `ty`, or `None` if it is not an integer type.
fn int_width(types: &TypeContext, ty: TypeId) -> Option<u32> {
    match types.get(ty) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Solver.
// ---------------------------------------------------------------------------

/// Run the SMT-LIB2 `script` (ending in `check-sat`) and map its verdict to a
/// [`RefinementResult`]. On `sat`, a second run appends a `get-value` for the
/// shared inputs to render the counterexample (`get-value` is illegal after an
/// `unsat`, so it cannot be issued unconditionally).
fn decide(script: &str, input_names: &[String]) -> RefinementResult {
    match z3rs::cmd_context::run_smt2(script) {
        Ok(lines) => match lines.first().map(String::as_str) {
            Some("unsat") => RefinementResult::Refines,
            Some("sat") => RefinementResult::Counterexample(extract_model(script, input_names)),
            Some("unknown") => {
                RefinementResult::Unknown("solver returned unknown (budget/incompleteness)".to_string())
            }
            other => RefinementResult::Unknown(format!("unexpected solver response: {other:?}")),
        },
        Err(e) => RefinementResult::Unknown(format!("solver error: {e}")),
    }
}

/// Re-run a satisfiable `script` with a `get-value` over the inputs to render a
/// human-readable counterexample; falls back to a placeholder on any hiccup.
fn extract_model(script: &str, input_names: &[String]) -> String {
    if input_names.is_empty() {
        return "(no inputs; the two functions differ on the empty input)".to_string();
    }
    let with_get = format!("{script}(get-value ({}))\n", input_names.join(" "));
    match z3rs::cmd_context::run_smt2(&with_get) {
        Ok(lines) => lines.get(1).cloned().unwrap_or_else(|| "(model unavailable)".to_string()),
        Err(_) => "(model unavailable)".to_string(),
    }
}

#[cfg(test)]
mod tests;
