//! **Yield-point insertion** for preemptible green threads (issue #3).
//!
//! A green-thread runtime that preempts cooperatively needs every loop that can
//! run for a long time to check, now and then, whether the scheduler wants the
//! CPU back. This pass inserts that check on the **back edges** of such loops:
//!
//! ```text
//!   latch ──► ^Y(args):  %f = load volatile @flag : i32
//!                        %c = icmp ne %f, i32 0
//!                        cond_br %c, ^Ycall, ^header(args)
//!             ^Ycall:    call @yield_fn()
//!                        br ^header(args)
//! ```
//!
//! The flag is a per-thread (or global) `i32` "preempt requested" word that a
//! timer signal handler or another thread sets; the yield function (supplied by
//! the runtime, e.g. one that clears the flag and calls `lf_ctx_switch`) runs
//! only when it is set, so the fast path is one load and one predictable branch
//! per iteration.
//!
//! # Which loops (a first slice of bet B9)
//!
//! Loops whose total cost is *proven small* are skipped. Cost lives in a tiny
//! lattice, [`Cost`]: `Bounded(n)` (at most `n` abstract cost units) below
//! `Unbounded` (⊤). Each instruction has a static cost ([`inst_cost`]); a loop
//! costs `trip_bound × (its own blocks + its inner loops' costs)`, with
//! saturating arithmetic that goes to ⊤ on overflow, and a loop with no proven
//! trip bound costs ⊤. A loop gets a check unless its cost is `Bounded(n)` with
//! `n <= max_cost`. So an enclosing loop of an unbounded loop is itself
//! unbounded and gets a check too (conservative, never unsound: every cycle
//! whose cost is not bounded passes a check).
//!
//! The trip bound comes from the **ranges** and **known-bits** domains (bet B8): the loop must have
//! an exit test, in a block that every iteration passes (one dominating all
//! latches), of the form `icmp pred iv, n` (or with `iv.next`, or swapped)
//! where `iv` is a header parameter stepped by a positive constant on every back
//! edge and `n` is loop-invariant; the bound is computed from the proven ranges of
//! `n` and of `iv`'s initial values, so `n = and %x, 15` bounds a loop as well as
//! a literal does. Anything else is treated as unbounded.
//!
//! Calls count as [`CALL_COST`]: a callee's own loops carry their own checks;
//! unbounded **recursion** is not covered by loop yield points.
//!
//! # Configuration and use
//!
//! [`YieldConfig`] names the flag global and the yield function by symbol
//! (declared in the module if absent: an external `i32` global and a
//! `() -> void` function) and sets the cost threshold. The pass is **not** part
//! of any `-O` pipeline: it adds an observable call, so it is not a refinement
//! of its input and runs only on request — append [`YieldPoints`] to a pass list
//! or call [`optimize_with_yield_points`], which runs it after the `-O`
//! pipeline so that loops are in their final shape and nothing hoists the check.
//! It is idempotent: a loop that already loads the flag volatilely is skipped.

use puremp::Int;

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::analysis::domains::known_bits::KnownBits;
use crate::analysis::domains::ranges::Range;
use crate::analysis::solver::solve;
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, InstId, InstKind, IntPred};
use crate::ir::types::{Type, TypeId};
use crate::ir::value::{Const, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Global, GlobalId, Module};
use crate::pass::{Changed, ModulePass};
use crate::support::Sym;
use crate::transform::pipeline::{OptLevel, pipeline_for, run_passes};
use crate::transform::{block_line, dom_preorder, rebuild_terminator, remap_value};

/// An element of the cost lattice: at most `n` cost units, or unbounded (⊤).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cost {
    /// At most this many abstract cost units.
    Bounded(u64),
    /// No bound is known (⊤).
    Unbounded,
}

impl Cost {
    /// The least upper bound (the larger cost).
    pub fn join(self, other: Cost) -> Cost {
        match (self, other) {
            (Cost::Bounded(a), Cost::Bounded(b)) => Cost::Bounded(a.max(b)),
            _ => Cost::Unbounded,
        }
    }
    /// Sequential composition: saturating sum, ⊤ on overflow.
    pub fn plus(self, other: Cost) -> Cost {
        match (self, other) {
            (Cost::Bounded(a), Cost::Bounded(b)) => a.checked_add(b).map_or(Cost::Unbounded, Cost::Bounded),
            _ => Cost::Unbounded,
        }
    }
    /// Repetition `n` times: ⊤ on overflow.
    pub fn times(self, n: u64) -> Cost {
        match self {
            Cost::Bounded(a) => a.checked_mul(n).map_or(Cost::Unbounded, Cost::Bounded),
            Cost::Unbounded => Cost::Unbounded,
        }
    }
    /// Whether this cost is proven to be at most `limit`.
    pub fn at_most(self, limit: u64) -> bool {
        matches!(self, Cost::Bounded(a) if a <= limit)
    }
}

/// The cost of a call (the callee's own loops carry their own checks).
pub const CALL_COST: u64 = 10;

/// The static cost of one instruction, in abstract units (roughly cycles).
pub fn inst_cost(kind: &InstKind) -> u64 {
    match kind {
        InstKind::Bin(op) => match op {
            BinOp::Mul => 3,
            BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => 20,
            BinOp::Add
            | BinOp::Sub
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor
            | BinOp::Shl
            | BinOp::LShr
            | BinOp::AShr => 1,
            _ => 5, // floating point
        },
        InstKind::Load { .. } | InstKind::Store { .. } => 4,
        InstKind::AtomicLoad { .. }
        | InstKind::AtomicStore { .. }
        | InstKind::AtomicRmw { .. }
        | InstKind::CmpXchg { .. }
        | InstKind::Fence(_) => 20,
        InstKind::Call => CALL_COST,
        InstKind::Syscall => 100,
        InstKind::DynAlloca { .. } => 10,
        _ => 1,
    }
}

/// What the pass inserts and when.
#[derive(Clone, Copy, Debug)]
pub struct YieldConfig {
    /// The `i32` global whose non-zero value requests a yield.
    pub flag: Sym,
    /// The `() -> void` function called when the flag is set.
    pub yield_fn: Sym,
    /// Loops whose proven total cost is at most this many units are skipped.
    pub max_cost: u64,
}

impl YieldConfig {
    /// The default threshold: loops proven to cost at most this many units
    /// (about as many cycles) need no check.
    pub const DEFAULT_MAX_COST: u64 = 10_000;

    /// A configuration with the default cost threshold.
    pub fn new(flag: Sym, yield_fn: Sym) -> YieldConfig {
        YieldConfig { flag, yield_fn, max_cost: Self::DEFAULT_MAX_COST }
    }
}

/// The yield-point insertion pass (see the [module docs](self)).
#[derive(Debug)]
pub struct YieldPoints {
    config: YieldConfig,
}

impl YieldPoints {
    /// A pass inserting checks per `config`.
    pub fn new(config: YieldConfig) -> YieldPoints {
        YieldPoints { config }
    }
}

/// One natural loop of a function.
#[derive(Clone, Debug)]
pub struct LoopReport {
    /// The loop header.
    pub header: BlockId,
    /// The proven trip bound (header executions per entry), if any.
    pub trip_bound: Option<u64>,
    /// The loop's total cost per entry.
    pub cost: Cost,
    /// Whether the pass puts a yield check on its back edges.
    pub needs_check: bool,
}

struct LoopInfo {
    header: usize,
    mask: Vec<bool>,
    latches: Vec<usize>,
    size: usize,
}

/// Discover every natural loop (back edges sharing a header are merged).
fn find_loops(f: &Function, cfg: &ControlFlowGraph, doms: &Dominators) -> Vec<LoopInfo> {
    let n = f.block_count();
    let mut loops = Vec::new();
    for h in 0..n {
        if !doms.is_reachable(h) {
            continue;
        }
        let latches: Vec<usize> = cfg
            .predecessors(h)
            .iter()
            .copied()
            .filter(|&p| doms.is_reachable(p) && doms.dominates(h, p))
            .collect();
        if latches.is_empty() {
            continue;
        }
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
        let size = mask.iter().filter(|&&m| m).count();
        loops.push(LoopInfo { header: h, mask, latches, size });
    }
    loops
}

/// The integer bit width of `ty`, if it is an integer type.
fn int_width(m: &Module, ty: TypeId) -> Option<u32> {
    match m.types().get(ty) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}

/// The value of an integer constant operand.
fn const_int(m: &Module, f: &Function, v: ValueId) -> Option<Int> {
    match &f.value(v).def {
        ValueDef::Const(c) => match m.consts().get(*c) {
            Const::Int { value, .. } => Some(value.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// The signed `[lo, hi]` of `v`: exact for a constant, else the intersection
/// of what the ranges and known-bits domains prove.
fn bounds_of(m: &Module, f: &Function, ranges: &[Range], known: &[KnownBits], v: ValueId) -> Option<(i128, i128)> {
    if let Some(c) = const_int(m, f, v) {
        let w = int_width(m, f.value_type(v))?;
        // Constants are stored as the mathematical representative; read signed.
        let c = c.to_i128()?;
        let c = if w < 128 && c >= (1i128 << (w - 1)) { c - (1i128 << w) } else { c };
        return Some((c, c));
    }
    let from_ranges = ranges[v.index()].bounds().and_then(|(_, lo, hi)| Some((lo.to_i128()?, hi.to_i128()?)));
    // Known bits: with the sign bit known zero the value is in [ones, ~zeros].
    let from_bits = match &known[v.index()] {
        KnownBits::Bits { width, zeros, ones } if (1..=64).contains(width) => {
            let (z, o) = (zeros.to_u128()?, ones.to_u128()?);
            let all = (1u128 << width) - 1;
            if z >> (width - 1) & 1 == 1 { Some((o as i128, (all & !z) as i128)) } else { None }
        }
        _ => None,
    };
    match (from_ranges, from_bits) {
        (Some((a, b)), Some((c, d))) => Some((a.max(c), b.min(d))),
        (x, None) | (None, x) => x,
    }
}

/// Per-function facts shared by the loop analyses.
struct FnFacts<'a> {
    m: &'a Module,
    f: &'a Function,
    ranges: Vec<Range>,
    known: Vec<KnownBits>,
    inst_block: Vec<usize>,
    doms: Dominators,
    cfg: ControlFlowGraph,
}

impl FnFacts<'_> {
    /// The block defining `v` (`None` for constants/globals/functions).
    fn def_block(&self, v: ValueId) -> Option<usize> {
        match &self.f.value(v).def {
            ValueDef::Param(b, _) => Some(b.index()),
            ValueDef::Inst(i) => Some(self.inst_block[i.index()]),
            _ => None,
        }
    }
    fn invariant(&self, lp: &LoopInfo, v: ValueId) -> bool {
        self.def_block(v).is_none_or(|b| b != usize::MAX && !lp.mask[b])
    }
    /// The argument `k` passed along the edge `from → to` (every such edge, when
    /// a terminator branches to `to` more than once; `None` if they differ).
    fn edge_arg(&self, from: usize, to: usize, k: usize) -> Option<ValueId> {
        let t = self.f.block(BlockId::from_index(from)).terminator()?;
        let inst = self.f.inst(t);
        let ops = inst.operands();
        let mut found: Option<ValueId> = None;
        let mut take = |target: BlockId, args: &[ValueId]| -> bool {
            if target.index() != to {
                return true;
            }
            let Some(&a) = args.get(k) else { return false };
            match found {
                Some(prev) if prev != a => false,
                _ => {
                    found = Some(a);
                    true
                }
            }
        };
        let ok = match &inst.kind {
            InstKind::Br(tb) => take(*tb, ops),
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let ta = *true_args as usize;
                let fa = *false_args as usize;
                take(*if_true, &ops[1..1 + ta]) && take(*if_false, &ops[1 + ta..1 + ta + fa])
            }
            InstKind::Switch(data) => {
                let da = data.default_args as usize;
                let mut ok = take(data.default, &ops[1..1 + da]);
                let mut off = 1 + da;
                for c in &data.cases {
                    let ca = c.args as usize;
                    ok &= take(c.target, &ops[off..off + ca]);
                    off += ca;
                }
                ok
            }
            _ => false,
        };
        if ok { found } else { None }
    }

    /// If `v` is a header parameter stepped by a positive constant on every
    /// back edge, return `(param index, step)`.
    fn induction(&self, lp: &LoopInfo, p: ValueId) -> Option<(usize, i128)> {
        let ValueDef::Param(b, k) = self.f.value(p).def else { return None };
        if b.index() != lp.header {
            return None;
        }
        let k = k as usize;
        let mut step: Option<i128> = None;
        for &l in &lp.latches {
            let next = self.edge_arg(l, lp.header, k)?;
            let ValueDef::Inst(i) = self.f.value(next).def else { return None };
            let inst = self.f.inst(i);
            if !matches!(inst.kind, InstKind::Bin(BinOp::Add)) {
                return None;
            }
            let ops = inst.operands();
            let c = if ops[0] == p {
                const_int(self.m, self.f, ops[1])?
            } else if ops[1] == p {
                const_int(self.m, self.f, ops[0])?
            } else {
                return None;
            };
            let c = c.to_i128()?;
            if c <= 0 || step.is_some_and(|s| s != c) {
                return None;
            }
            step = Some(c);
        }
        Some((k, step?))
    }

    /// `[lo, hi]` over the initial values of header parameter `k` (the
    /// arguments on the non-latch edges into the header).
    fn init_bounds(&self, lp: &LoopInfo, k: usize) -> Option<(i128, i128)> {
        let mut acc: Option<(i128, i128)> = None;
        for &p in self.cfg.predecessors(lp.header) {
            if lp.mask[p] || !self.doms.is_reachable(p) {
                continue;
            }
            let v = self.edge_arg(p, lp.header, k)?;
            let (lo, hi) = bounds_of(self.m, self.f, &self.ranges, &self.known, v)?;
            acc = Some(match acc {
                None => (lo, hi),
                Some((a, b)) => (a.min(lo), b.max(hi)),
            });
        }
        acc
    }

    /// A proven bound on the header executions of `lp`, from one exit test.
    fn trip_bound(&self, lp: &LoopInfo) -> Option<u64> {
        let mut best: Option<u64> = None;
        for b in 0..self.f.block_count() {
            if !lp.mask[b] || !lp.latches.iter().all(|&l| self.doms.dominates(b, l)) {
                continue;
            }
            if let Some(t) = self.exit_test_bound(lp, b) {
                best = Some(best.map_or(t, |x| x.min(t)));
            }
        }
        best
    }

    fn exit_test_bound(&self, lp: &LoopInfo, b: usize) -> Option<u64> {
        let t = self.f.block(BlockId::from_index(b)).terminator()?;
        let term = self.f.inst(t);
        let InstKind::CondBr { if_true, if_false, .. } = term.kind else { return None };
        let (tin, fin) = (lp.mask[if_true.index()], lp.mask[if_false.index()]);
        if tin == fin {
            return None;
        }
        let cond = term.operands()[0];
        let ValueDef::Inst(ci) = self.f.value(cond).def else { return None };
        let cinst = self.f.inst(ci);
        let InstKind::ICmp(mut pred) = cinst.kind else { return None };
        // Normalize to "stay in the loop while `a pred b`".
        if !tin {
            pred = negate(pred);
        }
        let (mut a, mut bnd) = (cinst.operands()[0], cinst.operands()[1]);
        let mut iv = self.iv_operand(lp, a);
        if iv.is_none() {
            std::mem::swap(&mut a, &mut bnd);
            pred = swap(pred);
            iv = self.iv_operand(lp, a);
        }
        let (k, step, offset) = iv?;
        if !self.invariant(lp, bnd) {
            return None;
        }
        let width = int_width(self.m, self.f.value_type(a))?;
        if width > 64 {
            return None;
        }
        let (ilo, ihi) = self.init_bounds(lp, k)?;
        let (_, nhi) = bounds_of(self.m, self.f, &self.ranges, &self.known, bnd)?;
        let (nlo, _) = bounds_of(self.m, self.f, &self.ranges, &self.known, bnd)?;
        let signed_max = (1i128 << (width - 1)) - 1;
        let (unsigned, inclusive) = match pred {
            IntPred::Slt => (false, false),
            IntPred::Sle => (false, true),
            IntPred::Ult => (true, false),
            IntPred::Ule => (true, true),
            IntPred::Ne if step == 1 => {
                // `iv != n` from below: iv must start at or below n.
                if ihi + offset > nlo {
                    return None;
                }
                (false, false)
            }
            _ => return None,
        };
        // Unsigned tests read these signed bounds as unsigned only when both
        // sides are non-negative.
        if unsigned && (ilo < 0 || nlo < 0) {
            return None;
        }
        // The stepped value must not wrap before the test fails.
        if nhi + step > signed_max {
            return None;
        }
        // Tested value at header execution j (0-based): init + offset + j*step.
        // It stays while <= limit, so j <= (limit - init - offset) / step.
        let limit = if inclusive { nhi } else { nhi - 1 };
        let span = limit - (ilo + offset);
        if span < 0 {
            return Some(1);
        }
        let trips = span / step + 1;
        // The test is evaluated once per iteration; a test in a later block than
        // the header allows one more header execution than tested iterations.
        u64::try_from(trips + 1).ok()
    }

    /// `(param index, step, offset)` when `v` is the induction parameter
    /// (`offset` 0) or its stepped value `iv + step` (`offset` = step).
    fn iv_operand(&self, lp: &LoopInfo, v: ValueId) -> Option<(usize, i128, i128)> {
        if let Some((k, s)) = self.induction(lp, v) {
            return Some((k, s, 0));
        }
        let ValueDef::Inst(i) = self.f.value(v).def else { return None };
        let inst = self.f.inst(i);
        if !matches!(inst.kind, InstKind::Bin(BinOp::Add)) {
            return None;
        }
        for (x, y) in [(0, 1), (1, 0)] {
            let ops = inst.operands();
            if let Some((k, s)) = self.induction(lp, ops[x])
                && const_int(self.m, self.f, ops[y]).and_then(|c| c.to_i128()) == Some(s)
            {
                return Some((k, s, s));
            }
        }
        None
    }
}

fn negate(p: IntPred) -> IntPred {
    match p {
        IntPred::Eq => IntPred::Ne,
        IntPred::Ne => IntPred::Eq,
        IntPred::Ugt => IntPred::Ule,
        IntPred::Uge => IntPred::Ult,
        IntPred::Ult => IntPred::Uge,
        IntPred::Ule => IntPred::Ugt,
        IntPred::Sgt => IntPred::Sle,
        IntPred::Sge => IntPred::Slt,
        IntPred::Slt => IntPred::Sge,
        IntPred::Sle => IntPred::Sgt,
    }
}

/// The predicate with its operands exchanged (`a < b` ⇔ `b > a`).
fn swap(p: IntPred) -> IntPred {
    match p {
        IntPred::Eq => IntPred::Eq,
        IntPred::Ne => IntPred::Ne,
        IntPred::Ugt => IntPred::Ult,
        IntPred::Uge => IntPred::Ule,
        IntPred::Ult => IntPred::Ugt,
        IntPred::Ule => IntPred::Uge,
        IntPred::Sgt => IntPred::Slt,
        IntPred::Sge => IntPred::Sle,
        IntPred::Slt => IntPred::Sgt,
        IntPred::Sle => IntPred::Sge,
    }
}

/// Whether the loop already loads `flag` volatilely (a previous run's check).
fn has_check(f: &Function, lp: &LoopInfo, flag: Option<GlobalId>) -> bool {
    let Some(g) = flag else { return false };
    f.blocks().any(|(bid, blk)| {
        lp.mask[bid.index()]
            && blk.insts().iter().any(|&i| {
                let inst = f.inst(i);
                matches!(inst.kind, InstKind::Load { volatile: true, .. })
                    && matches!(f.value(inst.operands()[0]).def, ValueDef::Global(x) if x == g)
            })
    })
}

/// Analyze every loop of `func`: trip bounds, costs, and whether it would get a
/// yield check under `config`.
pub fn analyze_loops(m: &Module, func: FuncId, config: &YieldConfig) -> Vec<LoopReport> {
    let flag = find_global(m, config.flag);
    let f = m.function(func);
    let Some(_) = f.entry() else { return Vec::new() };
    let cfg = ControlFlowGraph::new(f);
    let doms = Dominators::new(f, &cfg);
    let loops = find_loops(f, &cfg, &doms);
    if loops.is_empty() {
        return Vec::new();
    }
    let res = solve::<Range>(f, m.types(), m.consts());
    let ranges: Vec<Range> = (0..f.value_count()).map(|i| res.value(ValueId::from_index(i)).clone()).collect();
    let kb = solve::<KnownBits>(f, m.types(), m.consts());
    let known: Vec<KnownBits> = (0..f.value_count()).map(|i| kb.value(ValueId::from_index(i)).clone()).collect();
    let mut inst_block = vec![usize::MAX; f.inst_count()];
    for (bid, blk) in f.blocks() {
        for &i in blk.insts() {
            inst_block[i.index()] = bid.index();
        }
        if let Some(t) = blk.terminator() {
            inst_block[t.index()] = bid.index();
        }
    }
    let facts = FnFacts { m, f, ranges, known, inst_block, doms, cfg };

    // Innermost first: a loop's cost includes its (already computed) children.
    let mut order: Vec<usize> = (0..loops.len()).collect();
    order.sort_by_key(|&l| (loops[l].size, loops[l].header));
    let mut cost: Vec<Option<Cost>> = vec![None; loops.len()];
    let mut trips: Vec<Option<u64>> = vec![None; loops.len()];
    for &l in &order {
        let lp = &loops[l];
        // Direct children: loops strictly inside `lp` with no loop in between.
        let inside = |c: usize| c != l && loops[c].mask.iter().zip(&lp.mask).all(|(&x, &y)| !x || y);
        let children: Vec<usize> = (0..loops.len())
            .filter(|&c| inside(c) && !(0..loops.len()).any(|d| d != c && inside(d) && loops[c].mask.iter().zip(&loops[d].mask).all(|(&x, &y)| !x || y)))
            .collect();
        let mut body = Cost::Bounded(0);
        for (bid, blk) in f.blocks() {
            let b = bid.index();
            if !lp.mask[b] || children.iter().any(|&c| loops[c].mask[b]) {
                continue;
            }
            let mut n = blk.insts().iter().map(|&i| inst_cost(&f.inst(i).kind)).sum::<u64>();
            n += 1; // the terminator
            body = body.plus(Cost::Bounded(n));
        }
        for &c in &children {
            body = body.plus(cost[c].expect("children are computed first"));
        }
        let trip = facts.trip_bound(lp);
        trips[l] = trip;
        cost[l] = Some(match trip {
            Some(t) => body.times(t),
            None => Cost::Unbounded,
        });
    }

    let mut out: Vec<LoopReport> = loops
        .iter()
        .enumerate()
        .map(|(l, lp)| {
            let c = cost[l].expect("every loop is computed");
            LoopReport {
                header: BlockId::from_index(lp.header),
                trip_bound: trips[l],
                cost: c,
                needs_check: !c.at_most(config.max_cost) && !has_check(f, lp, flag),
            }
        })
        .collect();
    out.sort_by_key(|r| r.header.index());
    out
}

fn find_global(m: &Module, name: Sym) -> Option<GlobalId> {
    m.globals().position(|g| g.name == name).map(GlobalId::from_index)
}

fn find_function(m: &Module, name: Sym) -> Option<FuncId> {
    m.functions().position(|f| f.name == name).map(FuncId::from_index)
}

/// Rebuild `old` with a yield check on every back edge into the headers in
/// `checked`.
fn rebuild(
    old: &Function,
    b: &mut FunctionBuilder<'_>,
    checked: &[bool],
    flag: GlobalId,
    yield_fn: FuncId,
    i32t: TypeId,
    void: TypeId,
) {
    let n = old.block_count();
    let entry = old.entry().expect("a definition has an entry").index();
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);

    let mut new_block: Vec<BlockId> = Vec::with_capacity(n);
    for bi in 0..n {
        if bi == entry {
            new_block.push(b.create_entry_block());
        } else {
            let tys: Vec<TypeId> = old.block(BlockId::from_index(bi)).params().iter().map(|&p| old.value_type(p)).collect();
            new_block.push(b.create_block(&tys));
        }
    }
    // A check block per checked header, with the header's parameter types.
    let mut check_block: Vec<Option<BlockId>> = vec![None; n];
    for h in 0..n {
        if checked[h] {
            let tys: Vec<TypeId> = old.block(BlockId::from_index(h)).params().iter().map(|&p| old.value_type(p)).collect();
            check_block[h] = Some(b.create_block(&tys));
        }
    }

    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for (bi, &nb) in new_block.iter().enumerate() {
        let np = b.block_params(nb).to_vec();
        for (k, &p) in old.block(BlockId::from_index(bi)).params().iter().enumerate() {
            vmap[p.index()] = Some(np[k]);
        }
    }

    for bi in dom_preorder(old, &doms) {
        let bb = BlockId::from_index(bi);
        b.switch_to(new_block[bi]);
        for &i in old.block(bb).insts() {
            copy_inst(&mut vmap, old, b, i);
        }
        // Back edges (this block → a checked header it is dominated by) go
        // through the header's check block.
        let mut targets = new_block.clone();
        for h in 0..n {
            if let Some(cb) = check_block[h]
                && doms.dominates(h, bi)
                && cfg.successors(bi).contains(&h)
            {
                targets[h] = cb;
            }
        }
        rebuild_terminator(&mut vmap, old, b, &targets, bb, |_, _, _| {});
    }

    for h in 0..n {
        let Some(cb) = check_block[h] else { continue };
        let args = b.block_params(cb).to_vec();
        b.switch_to(cb);
        // The yield check stands for its loop header: it takes the header's line.
        b.set_line(block_line(old, BlockId::from_index(h)));
        let fp = b.global_ref(flag);
        let fv = b.load_volatile(i32t, fp, 4);
        let zero = b.const_i64(i32t, 0);
        let c = b.icmp(IntPred::Ne, fv, zero);
        let call_block = b.create_block(&[]);
        b.cond_br(c, call_block, &[], new_block[h], &args);
        b.switch_to(call_block);
        let callee = b.func_ref(yield_fn);
        b.call(callee, &[], void);
        b.br(new_block[h], &args);
    }
}

fn copy_inst(vmap: &mut [Option<ValueId>], old: &Function, b: &mut FunctionBuilder<'_>, i: InstId) {
    b.set_line_from(old, i);
    let inst = old.inst(i);
    let ops: Vec<ValueId> = inst.operands().iter().map(|&o| remap_value(vmap, old, b, o)).collect();
    let result_ty = inst.result().map(|_| inst.ty);
    let nr = b.append_inst(inst.kind.clone(), ops, inst.flags, result_ty);
    if let Some(r) = inst.result() {
        vmap[r.index()] = nr;
    }
}

impl ModulePass for YieldPoints {
    fn name(&self) -> &str {
        "yield_points"
    }

    fn run(&mut self, module: &mut Module) -> Changed {
        // Plan first (immutable), then declare what is missing, then rebuild.
        let mut plans: Vec<(FuncId, Vec<bool>)> = Vec::new();
        for fi in 0..module.function_count() {
            let fid = FuncId::from_index(fi);
            let f = module.function(fid);
            if f.is_declaration() || f.name == self.config.yield_fn {
                continue;
            }
            let reports = analyze_loops(module, fid, &self.config);
            if reports.iter().any(|r| r.needs_check) {
                let mut checked = vec![false; f.block_count()];
                for r in reports.iter().filter(|r| r.needs_check) {
                    checked[r.header.index()] = true;
                }
                plans.push((fid, checked));
            }
        }
        if plans.is_empty() {
            return Changed::No;
        }
        let i32t = module.types_mut().int(32);
        let void = module.types_mut().void();
        let flag = find_global(module, self.config.flag)
            .unwrap_or_else(|| module.add_global(Global { name: self.config.flag, ty: i32t, init: None }));
        let yield_fn = find_function(module, self.config.yield_fn).unwrap_or_else(|| {
            let sig = module.types_mut().func(Vec::new(), void, false);
            module.declare_function(self.config.yield_fn, sig)
        });
        for (fid, checked) in plans {
            let (fresh, ()) = module.map_function(fid, |old, b| rebuild(old, b, &checked, flag, yield_fn, i32t, void));
            module.replace_function(fid, fresh);
        }
        Changed::Yes
    }
}

/// Run the `-O` pipeline for `level`, then [`YieldPoints`] with `config`.
pub fn optimize_with_yield_points(module: &mut Module, level: OptLevel, config: YieldConfig) {
    let mut passes = pipeline_for(level);
    passes.push(Box::new(YieldPoints::new(config)));
    run_passes(module, passes);
}

#[cfg(test)]
mod tests;
