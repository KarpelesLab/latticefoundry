//! **simplify_cfg** — sound control-flow-graph cleanups (ROADMAP Phase 4).
//!
//! A single functional-rebuild pass (tenet T5) that performs the standard,
//! refinement-preserving CFG simplifications, iterated to a fixpoint *within one
//! `run`* so the output is stable (a second `run` reports [`Changed::No`]):
//!
//! 1. **Unreachable-block removal.** Blocks not reachable from the entry — over
//!    the CFG as refined by the folds below — are dropped. The verifier tolerates
//!    their absence and reachable code never observes their values.
//! 2. **Constant branch folding.** A `cond_br` whose condition is a constant `i1`
//!    becomes an unconditional `br` to the taken edge (carrying that edge's block
//!    arguments); the untaken edge disappears. A `switch` whose condition is a
//!    constant that *exactly* matches a case folds to that case's edge. (A
//!    constant that matches no case is left alone — folding to the default would
//!    require ruling out a modular-congruent match, which we cannot do
//!    conservatively, so we don't.)
//! 3. **Single-predecessor merge (straightening).** If a block `B` has exactly one
//!    predecessor `P`, `P`'s terminator is an unconditional `br` to `B`, and `B`
//!    is not the entry, `B` is spliced into `P`: `B`'s instructions follow `P`'s,
//!    and `B`'s block *parameters* are bound to the arguments `P` passed on the
//!    edge (the block-argument analog of φ-straightening). Chains straighten
//!    transitively in one pass.
//! 4. **Empty forwarding-block bypass.** A parameter-free/pure forwarding block
//!    `^F(p…): br C(args…)` — no instructions, an unconditional branch whose
//!    every argument is either one of `F`'s own parameters or a constant/reference
//!    — is transparent: every edge into `F` is redirected to `C`, substituting the
//!    incoming arguments for `F`'s parameters. Because the substituted arguments
//!    are exactly what the predecessor already supplied (available at the
//!    predecessor) or constants (available everywhere), the redirect is always
//!    sound regardless of dominance. Forwarding blocks whose arguments reference
//!    values defined elsewhere are conservatively left in place.
//!
//! ## How block arguments are threaded
//!
//! The whole transform is computed as a plan over the *old* function and emitted
//! once. Each old terminator is first normalized to an [`Term`] with its outgoing
//! edges resolved through forwarding blocks (substituting parameters as it goes)
//! and its constant branches folded. Reachability, predecessor counts, and merge
//! eligibility are computed on this resolved graph, so folds and bypasses expose
//! further merges in the *same* pass. Emission walks the surviving "root" blocks
//! in reverse-postorder of the resolved graph (so every SSA definition precedes
//! its uses); an absorbed block's parameters are bound in the value map to the
//! merged edge's arguments, and a forwarded edge's arguments are the resolved,
//! predecessor-available values.
//!
//! ## Note on constant reading
//!
//! A [`FunctionTransform`] receives only the old [`Function`] and a builder, not
//! the module's [`ConstPool`], so a constant's numeric value cannot be decoded
//! directly. To fold a branch we instead round-trip through the builder's
//! module-wide interning: `use_const(cid)` and `const_bool(true)` yield the *same*
//! new value id iff `cid` is the canonical `i1` `true` constant. This is exact and
//! never yields a false positive (a distinct constant interns to a distinct id),
//! so the fold stays sound; an unrecognized constant is simply not folded.

use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::InstKind;
use crate::ir::value::{ValueDef, ValueId};
use crate::ir::{BlockId, Function};
use crate::pass::Changed;
use crate::transform::{FunctionTransform, remap_value};

/// The CFG-simplification transform (see the module documentation).
#[derive(Debug, Default, Clone, Copy)]
pub struct SimplifyCfg;

impl FunctionTransform for SimplifyCfg {
    fn name(&self) -> &str {
        "simplify_cfg"
    }

    fn run(&mut self, old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
        simplify(old, builder)
    }
}

/// A block's terminator after constant folding, over *old* block indices and
/// *old* value ids. Outgoing-edge arguments are resolved through forwarding
/// blocks lazily at use sites (see [`resolve_edge`]).
#[derive(Debug, Clone)]
enum Term {
    /// `ret` with an optional value operand.
    Ret(Option<ValueId>),
    /// `unreachable`.
    Unreachable,
    /// A block with no terminator (malformed input); emitted as `unreachable`.
    Missing,
    /// Unconditional branch to a block, with its edge arguments.
    Br(usize, Vec<ValueId>),
    /// Conditional branch on `cond` to `t`/`f` with their edge arguments.
    Cond { cond: ValueId, t: usize, ta: Vec<ValueId>, f: usize, fa: Vec<ValueId> },
    /// Multi-way branch on `cond`; `default`/`da` and the match arms.
    Switch { cond: ValueId, default: usize, da: Vec<ValueId>, cases: Vec<(puremp::Int, usize, Vec<ValueId>)> },
}

/// The immutable analysis result consulted during emission.
#[derive(Debug)]
struct Plan {
    /// The folded terminator of each old block.
    eff: Vec<Term>,
    /// Whether each old block is a transparent forwarding block (bypassed).
    is_forwarder: Vec<bool>,
    /// Whether each old block is merged into its unique predecessor.
    absorbable: Vec<bool>,
    /// Old block index → its block id in the rebuilt function (roots only).
    new_block: Vec<Option<BlockId>>,
}

fn simplify(old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
    let n = old.block_count();
    let Some(entry) = old.entry() else {
        return Changed::No;
    };
    let entry_idx = entry.index();

    // Phase 1: constant-fold every terminator.
    let (eff, any_fold) = compute_eff(old, builder);

    // Phase 2: classify transparent forwarding blocks (empty, unconditional
    // branch whose arguments are own-parameters or constants), never the entry.
    let mut is_forwarder = vec![false; n];
    for b in 0..n {
        if b == entry_idx || !old.block(BlockId::from_index(b)).insts().is_empty() {
            continue;
        }
        if let Term::Br(c, args) = &eff[b]
            && *c != b
            && args.iter().all(|&a| forward_safe(old, b, a))
        {
            is_forwarder[b] = true;
        }
    }
    // A forwarding cycle has no non-forwarder exit; demote such blocks so every
    // resolution terminates (conservative and always sound).
    loop {
        let mut changed = false;
        for b in 0..n {
            if !is_forwarder[b] {
                continue;
            }
            let mut cur = b;
            let mut seen = vec![false; n];
            let mut ok = false;
            loop {
                if !is_forwarder[cur] {
                    ok = true;
                    break;
                }
                if seen[cur] {
                    break;
                }
                seen[cur] = true;
                match &eff[cur] {
                    Term::Br(c, _) => cur = *c,
                    _ => break,
                }
            }
            if !ok {
                is_forwarder[b] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Phase 3: resolved successor lists (edges through forwarders), reachability.
    let mut rsucc: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 0..n {
        if is_forwarder[b] {
            continue;
        }
        for (tgt, args) in term_edges(&eff[b]) {
            let (rt, _) = resolve_edge(&eff, &is_forwarder, old, tgt, &args);
            rsucc[b].push(rt);
        }
    }

    let mut visited = vec![false; n];
    let mut post: Vec<usize> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(entry_idx, 0)];
    visited[entry_idx] = true;
    while let Some(&(node, idx)) = stack.last() {
        if idx < rsucc[node].len() {
            stack.last_mut().expect("nonempty").1 = idx + 1;
            let next = rsucc[node][idx];
            if !visited[next] {
                visited[next] = true;
                stack.push((next, 0));
            }
        } else {
            post.push(node);
            stack.pop();
        }
    }
    let reachable = visited;

    // Phase 4: predecessor counts and single-predecessor merge eligibility.
    let mut incoming: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];
    for b in 0..n {
        if is_forwarder[b] || !reachable[b] {
            continue;
        }
        let uncond = matches!(eff[b], Term::Br(..));
        for &t in &rsucc[b] {
            incoming[t].push((b, uncond));
        }
    }
    let mut absorbable = vec![false; n];
    for t in 0..n {
        if !reachable[t] || t == entry_idx || is_forwarder[t] {
            continue;
        }
        if incoming[t].len() == 1 && incoming[t][0].1 && incoming[t][0].0 != t {
            absorbable[t] = true;
        }
    }

    let is_root: Vec<bool> =
        (0..n).map(|b| reachable[b] && !is_forwarder[b] && !absorbable[b]).collect();
    let num_roots = is_root.iter().filter(|&&x| x).count();

    // Nothing dropped/merged/bypassed and nothing folded ⇒ identical function.
    if num_roots == n && !any_fold {
        return Changed::No;
    }

    // Phase 5: create the surviving blocks, then emit in resolved reverse-postorder.
    let mut new_block: Vec<Option<BlockId>> = vec![None; n];
    new_block[entry_idx] = Some(builder.create_entry_block());
    for b in 0..n {
        if b == entry_idx || !is_root[b] {
            continue;
        }
        let bb = BlockId::from_index(b);
        let ptys: Vec<_> = old.block(bb).params().iter().map(|&p| old.value_type(p)).collect();
        new_block[b] = Some(builder.create_block(&ptys));
    }

    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for b in 0..n {
        if !is_root[b] {
            continue;
        }
        let bb = BlockId::from_index(b);
        let nb = new_block[b].expect("root has a new block");
        let nps = builder.block_params(nb).to_vec();
        for (i, &p) in old.block(bb).params().iter().enumerate() {
            vmap[p.index()] = Some(nps[i]);
        }
    }

    let plan = Plan { eff, is_forwarder, absorbable, new_block };
    for &b in post.iter().rev() {
        if is_root[b] {
            emit_chain(&plan, &mut vmap, old, builder, b);
        }
    }

    Changed::Yes
}

/// Whether value `a`, used as a branch argument of block `b`, is safe to
/// substitute when bypassing `b`: a constant/reference (available everywhere) or
/// one of `b`'s own parameters (replaced by the incoming argument).
fn forward_safe(old: &Function, b: usize, a: ValueId) -> bool {
    match old.value(a).def {
        ValueDef::Const(_) | ValueDef::Global(_) | ValueDef::Func(_) => true,
        ValueDef::Param(bb, _) => bb.index() == b,
        ValueDef::Inst(_) => false,
    }
}

/// Follow a chain of forwarding blocks from `target`, substituting each
/// forwarder's parameters with the incoming arguments, until a non-forwarder
/// block is reached. Returns that block and the argument list to pass it.
fn resolve_edge(
    eff: &[Term],
    is_forwarder: &[bool],
    old: &Function,
    mut target: usize,
    args: &[ValueId],
) -> (usize, Vec<ValueId>) {
    let mut cur_args = args.to_vec();
    let mut steps = 0usize;
    while is_forwarder[target] && steps <= eff.len() {
        steps += 1;
        let Term::Br(c, bargs) = &eff[target] else {
            break;
        };
        let mut new_args = Vec::with_capacity(bargs.len());
        for &barg in bargs {
            if let ValueDef::Param(bb, idx) = old.value(barg).def
                && bb.index() == target
            {
                new_args.push(cur_args[idx as usize]);
            } else {
                new_args.push(barg);
            }
        }
        target = *c;
        cur_args = new_args;
    }
    (target, cur_args)
}

/// The outgoing (target, arguments) edges of a folded terminator, in edge order.
fn term_edges(t: &Term) -> Vec<(usize, Vec<ValueId>)> {
    match t {
        Term::Br(x, a) => vec![(*x, a.clone())],
        Term::Cond { t, ta, f, fa, .. } => vec![(*t, ta.clone()), (*f, fa.clone())],
        Term::Switch { default, da, cases, .. } => {
            let mut v = Vec::with_capacity(1 + cases.len());
            v.push((*default, da.clone()));
            for (_, tgt, a) in cases {
                v.push((*tgt, a.clone()));
            }
            v
        }
        Term::Ret(_) | Term::Unreachable | Term::Missing => Vec::new(),
    }
}

/// Compute each block's constant-folded terminator, and whether any fold applied.
fn compute_eff(old: &Function, builder: &mut FunctionBuilder<'_>) -> (Vec<Term>, bool) {
    let n = old.block_count();
    let mut eff = Vec::with_capacity(n);
    let mut any_fold = false;
    for b in 0..n {
        let bb = BlockId::from_index(b);
        let Some(term) = old.block(bb).terminator() else {
            eff.push(Term::Missing);
            continue;
        };
        let inst = old.inst(term);
        let ops = inst.operands();
        let t = match &inst.kind {
            InstKind::Ret => Term::Ret(ops.first().copied()),
            InstKind::Unreachable => Term::Unreachable,
            InstKind::Br(target) => Term::Br(target.index(), ops.to_vec()),
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let ta_n = *true_args as usize;
                let fa_n = *false_args as usize;
                let cond = ops[0];
                let ta = ops[1..1 + ta_n].to_vec();
                let fa = ops[1 + ta_n..1 + ta_n + fa_n].to_vec();
                match const_i1(old, builder, cond) {
                    Some(true) => {
                        any_fold = true;
                        Term::Br(if_true.index(), ta)
                    }
                    Some(false) => {
                        any_fold = true;
                        Term::Br(if_false.index(), fa)
                    }
                    None => Term::Cond { cond, t: if_true.index(), ta, f: if_false.index(), fa },
                }
            }
            InstKind::Switch(data) => {
                let cond = ops[0];
                let da_n = data.default_args as usize;
                let da = ops[1..1 + da_n].to_vec();
                let mut off = 1 + da_n;
                let mut cases = Vec::with_capacity(data.cases.len());
                for c in &data.cases {
                    let cn = c.args as usize;
                    let cargs = ops[off..off + cn].to_vec();
                    off += cn;
                    cases.push((c.value.clone(), c.target.index(), cargs));
                }
                match switch_fold(old, builder, cond, &cases) {
                    Some((tgt, args)) => {
                        any_fold = true;
                        Term::Br(tgt, args)
                    }
                    None => Term::Switch { cond, default: data.default.index(), da, cases },
                }
            }
            // A non-terminator in the terminator slot is malformed input.
            _ => Term::Missing,
        };
        eff.push(t);
    }
    (eff, any_fold)
}

/// If `v` is the canonical `i1` `true`/`false` constant, return its boolean value.
///
/// Uses module-wide constant interning: `use_const(cid)` and `const_bool(x)`
/// dedup to the same new value id iff `cid` is the constant for `x`. Any other
/// (or non-constant) value returns `None` and is left unfolded (sound).
fn const_i1(old: &Function, builder: &mut FunctionBuilder<'_>, v: ValueId) -> Option<bool> {
    let ValueDef::Const(cid) = old.value(v).def else {
        return None;
    };
    let a = builder.use_const(cid);
    let t = builder.const_bool(true);
    let f = builder.const_bool(false);
    if a == t {
        Some(true)
    } else if a == f {
        Some(false)
    } else {
        None
    }
}

/// If switch condition `cond` is a constant that *exactly* equals a case value,
/// return that case's (target, arguments). Exact interning equality never
/// misidentifies a case (distinct constants intern distinctly), so a fold is
/// always sound; a constant matching no case is reported as `None` (unfolded).
fn switch_fold(
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    cond: ValueId,
    cases: &[(puremp::Int, usize, Vec<ValueId>)],
) -> Option<(usize, Vec<ValueId>)> {
    let ValueDef::Const(cid) = old.value(cond).def else {
        return None;
    };
    let ty = old.value_type(cond);
    let a = builder.use_const(cid);
    for (val, tgt, args) in cases {
        let cv = builder.const_int(ty, val.clone());
        if a == cv {
            return Some((*tgt, args.clone()));
        }
    }
    None
}

/// Emit a root block and the straight-line chain of blocks merged into it.
fn emit_chain(
    plan: &Plan,
    vmap: &mut [Option<ValueId>],
    old: &Function,
    builder: &mut FunctionBuilder<'_>,
    root: usize,
) {
    builder.switch_to(plan.new_block[root].expect("root block"));
    let mut cur = root;
    let mut steps = 0usize;
    loop {
        // A reachable merge chain is acyclic; this bound only guards against bugs.
        steps += 1;
        if steps > plan.eff.len() + 1 {
            builder.unreachable();
            return;
        }

        let bb = BlockId::from_index(cur);
        let insts = old.block(bb).insts().to_vec();
        for i in insts {
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

        match &plan.eff[cur] {
            Term::Ret(v) => {
                let nv = v.map(|x| remap_value(vmap, old, builder, x));
                builder.ret(nv);
                return;
            }
            Term::Unreachable | Term::Missing => {
                builder.unreachable();
                return;
            }
            Term::Br(t, args) => {
                let (ft, fargs) = resolve_edge(&plan.eff, &plan.is_forwarder, old, *t, args);
                if plan.absorbable[ft] {
                    // Splice `ft` into this chain: bind its parameters to the edge.
                    let params = old.block(BlockId::from_index(ft)).params().to_vec();
                    for (i, &p) in params.iter().enumerate() {
                        let nv = remap_value(vmap, old, builder, fargs[i]);
                        vmap[p.index()] = Some(nv);
                    }
                    cur = ft;
                    continue;
                }
                let nargs: Vec<_> =
                    fargs.iter().map(|&a| remap_value(vmap, old, builder, a)).collect();
                builder.br(plan.new_block[ft].expect("branch target is a root"), &nargs);
                return;
            }
            Term::Cond { cond, t, ta, f, fa } => {
                let (t2, ta2) = resolve_edge(&plan.eff, &plan.is_forwarder, old, *t, ta);
                let (f2, fa2) = resolve_edge(&plan.eff, &plan.is_forwarder, old, *f, fa);
                let nc = remap_value(vmap, old, builder, *cond);
                let nta: Vec<_> = ta2.iter().map(|&a| remap_value(vmap, old, builder, a)).collect();
                let nfa: Vec<_> = fa2.iter().map(|&a| remap_value(vmap, old, builder, a)).collect();
                builder.cond_br(
                    nc,
                    plan.new_block[t2].expect("true target is a root"),
                    &nta,
                    plan.new_block[f2].expect("false target is a root"),
                    &nfa,
                );
                return;
            }
            Term::Switch { cond, default, da, cases } => {
                let (d2, da2) = resolve_edge(&plan.eff, &plan.is_forwarder, old, *default, da);
                let nc = remap_value(vmap, old, builder, *cond);
                let nda: Vec<_> = da2.iter().map(|&a| remap_value(vmap, old, builder, a)).collect();
                let mut ncases = Vec::with_capacity(cases.len());
                for (val, tgt, args) in cases {
                    let (t2, a2) = resolve_edge(&plan.eff, &plan.is_forwarder, old, *tgt, args);
                    let na: Vec<_> = a2.iter().map(|&a| remap_value(vmap, old, builder, a)).collect();
                    ncases.push((val.clone(), plan.new_block[t2].expect("case target is a root"), na));
                }
                builder.switch(nc, plan.new_block[d2].expect("default target is a root"), &nda, ncases);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::SimplifyCfg;
    use crate::analysis::domains::ConstLattice;
    use crate::analysis::solver::solve;
    use crate::ir::inst::{Flags, InstKind};
    use crate::ir::value::Const;
    use crate::ir::{FuncId, Function, InstId, Module, ValueId};
    use crate::pass::Changed;
    use crate::support::StrInterner;
    use crate::transform::FunctionTransform;
    use crate::verify::verify_module;

    use puremp::Int;

    /// Run the pass once, installing the result iff it changed.
    fn run_simplify(m: &mut Module, f: FuncId) -> Changed {
        let mut t = SimplifyCfg;
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
            if let Some(t) = blk.terminator()
                && pred(&f.inst(t).kind)
            {
                c += 1;
            }
        }
        c
    }

    /// The abstract constant of the operand of whichever block returns a value.
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
            ConstLattice::Const(Const::Int { value, .. }) => {
                assert_eq!(
                    value.mod_2k(width),
                    Int::from_i64(expected).mod_2k(width),
                    "returned constant mismatch"
                );
            }
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

    // -----------------------------------------------------------------------
    // Builders
    // -----------------------------------------------------------------------

    /// entry: ret 0 ; ^dead: ret 1  (dead is unreachable)
    fn build_unreachable() -> (Module, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("scfg-unreach");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let _entry = b.create_entry_block();
            let dead = b.create_block(&[]);
            let zero = b.const_i64(i32t, 0);
            b.ret(Some(zero));
            b.switch_to(dead);
            let one = b.const_i64(i32t, 1);
            b.ret(Some(one));
        }
        (m, f)
    }

    /// entry: cond_br true, ^a, ^b ; ^a: ret 10 ; ^b: ret 20
    fn build_const_cond() -> (Module, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("scfg-cond");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let _entry = b.create_entry_block();
            let a = b.create_block(&[]);
            let bl = b.create_block(&[]);
            let t = b.const_bool(true);
            b.cond_br(t, a, &[], bl, &[]);
            b.switch_to(a);
            let ten = b.const_i64(i32t, 10);
            b.ret(Some(ten));
            b.switch_to(bl);
            let twenty = b.const_i64(i32t, 20);
            b.ret(Some(twenty));
        }
        (m, f)
    }

    /// entry(a): br ^b(a) ; ^b(x): y = x + 1 ; ret y
    fn build_merge() -> (Module, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("scfg-merge");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let bb = b.create_block(&[i32t]);
            let a = b.param(entry, 0);
            b.switch_to(entry);
            b.br(bb, &[a]);
            b.switch_to(bb);
            let x = b.param(bb, 0);
            let one = b.const_i64(i32t, 1);
            let y = b.add(x, one, Flags::NONE);
            b.ret(Some(y));
        }
        (m, f)
    }

    /// entry: cond_br true, ^then, ^else
    /// ^then: br ^merge(10) ; ^else: br ^merge(20) ; ^merge(x): ret x
    fn build_diamond() -> (Module, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("scfg-diamond");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let _entry = b.create_entry_block();
            let then_b = b.create_block(&[]);
            let els_b = b.create_block(&[]);
            let merge = b.create_block(&[i32t]);
            let t = b.const_bool(true);
            b.cond_br(t, then_b, &[], els_b, &[]);
            b.switch_to(then_b);
            let ten = b.const_i64(i32t, 10);
            b.br(merge, &[ten]);
            b.switch_to(els_b);
            let twenty = b.const_i64(i32t, 20);
            b.br(merge, &[twenty]);
            b.switch_to(merge);
            let x = b.param(merge, 0);
            b.ret(Some(x));
        }
        (m, f)
    }

    /// entry(c): cond_br c, ^a, ^b
    /// ^a: br ^fwd(10) ; ^b: br ^fwd(20) ; ^fwd(x): br ^exit(x) ; ^exit(y): ret y
    ///
    /// `^a`, `^b`, and `^fwd` are all empty forwarding blocks and get bypassed;
    /// `^exit` keeps two predecessors so it is not merged.
    fn build_bypass() -> (Module, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("scfg-bypass");
        let i1 = m.types_mut().bool();
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i1], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let a = b.create_block(&[]);
            let bl = b.create_block(&[]);
            let fwd = b.create_block(&[i32t]);
            let exit = b.create_block(&[i32t]);
            let c = b.param(entry, 0);
            b.switch_to(entry);
            b.cond_br(c, a, &[], bl, &[]);
            b.switch_to(a);
            let ten = b.const_i64(i32t, 10);
            b.br(fwd, &[ten]);
            b.switch_to(bl);
            let twenty = b.const_i64(i32t, 20);
            b.br(fwd, &[twenty]);
            b.switch_to(fwd);
            let x = b.param(fwd, 0);
            b.br(exit, &[x]);
            b.switch_to(exit);
            let y = b.param(exit, 0);
            b.ret(Some(y));
        }
        (m, f)
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn removes_unreachable_block() {
        let (mut m, f) = build_unreachable();
        assert_eq!(m.function(f).block_count(), 2);
        assert!(verify_module(&m).is_ok());
        let c = run_simplify(&mut m, f);
        assert_eq!(c, Changed::Yes);
        assert_eq!(m.function(f).block_count(), 1, "the unreachable block is dropped");
        assert!(verify_module(&m).is_ok());
        assert_ret_int(&m, f, 32, 0);
    }

    #[test]
    fn folds_constant_conditional_branch() {
        let (mut m, f) = build_const_cond();
        let before = ret_value_const(&m, f);
        assert!(verify_module(&m).is_ok());
        let c = run_simplify(&mut m, f);
        assert_eq!(c, Changed::Yes);
        let func = m.function(f);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::CondBr { .. })), 0, "cond_br folded away");
        assert!(func.block_count() < 3, "the untaken block ^b is dropped");
        assert!(verify_module(&m).is_ok());
        assert_ret_int(&m, f, 32, 10);
        // Behavior preserved: the constant-analysis result is unchanged.
        assert_eq!(before, ret_value_const(&m, f));
    }

    #[test]
    fn merges_single_predecessor_with_block_args() {
        let (mut m, f) = build_merge();
        let before = ret_value_const(&m, f);
        assert_eq!(m.function(f).block_count(), 2);
        let c = run_simplify(&mut m, f);
        assert_eq!(c, Changed::Yes);
        let func = m.function(f);
        assert_eq!(func.block_count(), 1, "^b is spliced into its unique predecessor");
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Bin(_))), 1, "the add survives the merge");
        assert!(verify_module(&m).is_ok());
        // The block parameter x was correctly bound to the argument a.
        assert_eq!(before, ret_value_const(&m, f));
    }

    #[test]
    fn diamond_folds_to_straight_line() {
        let (mut m, f) = build_diamond();
        assert_eq!(m.function(f).block_count(), 4);
        let before = ret_value_const(&m, f);
        assert!(matches!(before, ConstLattice::Const(_)), "true arm pins the result to 10");
        let c = run_simplify(&mut m, f);
        assert_eq!(c, Changed::Yes);
        let func = m.function(f);
        assert_eq!(func.block_count(), 1, "the diamond collapses to one block");
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::CondBr { .. } | InstKind::Br(_))), 0);
        assert!(verify_module(&m).is_ok());
        assert_ret_int(&m, f, 32, 10);
        assert_eq!(before, ret_value_const(&m, f));
    }

    #[test]
    fn bypasses_empty_forwarding_blocks() {
        let (mut m, f) = build_bypass();
        assert_eq!(m.function(f).block_count(), 5);
        let before = ret_value_const(&m, f);
        let c = run_simplify(&mut m, f);
        assert_eq!(c, Changed::Yes);
        let func = m.function(f);
        assert_eq!(func.block_count(), 2, "^a, ^b, ^fwd are all bypassed");
        assert!(verify_module(&m).is_ok());
        // Both conditional edges now target the single surviving ^exit block.
        let entry = func.entry().expect("entry");
        let term = func.block(entry).terminator().expect("entry terminator");
        if let InstKind::CondBr { if_true, if_false, .. } = func.inst(term).kind {
            assert_eq!(if_true, if_false, "both arms forward to ^exit");
            assert_ne!(if_true, entry);
        } else {
            panic!("entry should still be a cond_br on the unknown condition");
        }
        assert_eq!(before, ret_value_const(&m, f));
    }

    #[test]
    fn is_idempotent() {
        for build in [build_diamond as fn() -> (Module, FuncId), build_merge, build_bypass, build_const_cond] {
            let (mut m, f) = build();
            let first = run_simplify(&mut m, f);
            assert_eq!(first, Changed::Yes);
            assert!(verify_module(&m).is_ok());
            let second = run_simplify(&mut m, f);
            assert_eq!(second, Changed::No, "a second run finds nothing more to simplify");
        }
    }

    #[test]
    fn already_simple_is_unchanged() {
        // A single straight-line block has nothing to simplify.
        let mut syms = StrInterner::new();
        let mut m = Module::new("scfg-simple");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32t], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let x = b.param(entry, 0);
            let one = b.const_i64(i32t, 1);
            let y = b.add(x, one, Flags::NONE);
            b.ret(Some(y));
        }
        assert_eq!(run_simplify(&mut m, f), Changed::No);
        assert_eq!(m.function(f).block_count(), 1);
    }

    #[test]
    fn is_deterministic() {
        for build in [build_diamond as fn() -> (Module, FuncId), build_merge, build_bypass] {
            let (mut m1, f1) = build();
            let (mut m2, f2) = build();
            run_simplify(&mut m1, f1);
            run_simplify(&mut m2, f2);
            assert_eq!(canon(m1.function(f1)), canon(m2.function(f2)));
        }
    }

    #[test]
    fn folds_constant_switch() {
        // entry: switch 1i32 [default ^d, 1 -> ^one] ; ^one: ret 7 ; ^d: ret 9
        let mut syms = StrInterner::new();
        let mut m = Module::new("scfg-switch");
        let i32t = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![], i32t, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let _entry = b.create_entry_block();
            let one_b = b.create_block(&[]);
            let def_b = b.create_block(&[]);
            let sel = b.const_i64(i32t, 1);
            b.switch(sel, def_b, &[], vec![(Int::from_i64(1), one_b, vec![])]);
            b.switch_to(one_b);
            let seven = b.const_i64(i32t, 7);
            b.ret(Some(seven));
            b.switch_to(def_b);
            let nine = b.const_i64(i32t, 9);
            b.ret(Some(nine));
        }
        let c = run_simplify(&mut m, f);
        assert_eq!(c, Changed::Yes);
        let func = m.function(f);
        assert_eq!(count_kind(func, |k| matches!(k, InstKind::Switch(_))), 0, "switch folded away");
        assert_eq!(func.block_count(), 1);
        assert!(verify_module(&m).is_ok());
        assert_ret_int(&m, f, 32, 7);
        assert_eq!(run_simplify(&mut m, f), Changed::No);
    }
}
