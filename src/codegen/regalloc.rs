//! Register allocation: a correct **linear-scan** allocator over live intervals
//! (ROADMAP Phase 5; graph-coloring is a later refinement).
//!
//! The pipeline is:
//!
//! 1. **Liveness** ([`compute_liveness`]) — a backward dataflow over the machine
//!    CFG giving per-block live-in/live-out sets of vregs, plus a linear
//!    numbering of instruction *points*.
//! 2. **Live intervals** ([`build_intervals`]) — one conservative
//!    `[start, end]` range per vreg (its first def / block-live-in to its last
//!    use / block-live-out). Two vregs *interfere* iff their intervals overlap.
//! 3. **Linear scan** — process intervals by increasing start, keeping an
//!    *active* set expired by end point; assign each vreg the lowest-numbered
//!    free physical register of its class that also avoids the **fixed**
//!    intervals of physical registers (ABI argument/return registers and the
//!    caller-saved registers a `call` clobbers). When none is free, **spill**
//!    the interval that ends furthest away.
//! 4. **Rewrite** — replace every vreg operand with its physical register;
//!    around each use/def of a spilled vreg insert a reload/spill using the
//!    target's reserved scratch registers. After this, no vreg remains.
//!
//! Everything is dense-id/`Vec`-indexed and processed in deterministic order
//! (tenets T5/T6): the same MIR always yields the same allocation.

use crate::codegen::mir::{
    MBlockId, MachineFunction, MachineInst, MachineOperand, PReg, Reg, StackSlot, VReg,
};
use crate::codegen::target::MachineTarget;
use crate::support::{DetHashMap, DetHashSet};

/// Per-function liveness: live-in/out sets and the instruction numbering.
#[derive(Debug)]
pub struct Liveness {
    /// `live_in[b]` / `live_out[b]`: the vregs live entering/leaving block `b`
    /// (block index = `MBlockId::index`).
    pub live_in: Vec<DetHashSet<VReg>>,
    /// See [`Liveness::live_in`].
    pub live_out: Vec<DetHashSet<VReg>>,
    block_start: Vec<usize>,
    block_end: Vec<usize>,
    num_points: usize,
    /// The block indices in program-point order.
    order: Vec<usize>,
}

impl Liveness {
    /// The first program point of block `b`.
    #[inline]
    pub fn block_start(&self, b: usize) -> usize {
        self.block_start[b]
    }

    /// The last program point of block `b`.
    #[inline]
    pub fn block_end(&self, b: usize) -> usize {
        self.block_end[b]
    }

    /// The total number of program points.
    #[inline]
    pub fn num_points(&self) -> usize {
        self.num_points
    }
}

/// A vreg's live interval: the half-open reasoning is folded into an inclusive
/// `[start, end]` over program points.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Interval {
    /// First point the vreg is live.
    pub start: usize,
    /// Last point the vreg is live.
    pub end: usize,
}

impl Interval {
    /// Whether two intervals overlap (share at least one point).
    #[inline]
    pub fn overlaps(self, other: Interval) -> bool {
        self.start <= other.end && other.start <= self.end
    }
}

/// The computed live intervals, one optional entry per vreg (by index).
#[derive(Debug)]
pub struct Intervals {
    per_vreg: Vec<Option<Interval>>,
}

impl Intervals {
    /// The interval of a vreg, if it is live anywhere.
    #[inline]
    pub fn get(&self, v: VReg) -> Option<Interval> {
        self.per_vreg[v.index()]
    }
}

/// Where a vreg ended up.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Assign {
    /// Allocated to a physical register.
    Reg(PReg),
    /// Spilled to a stack slot.
    Spill(StackSlot),
}

/// The result of allocation: the per-vreg assignment and the intervals it was
/// derived from (both indexed by vreg index). Exposed so callers/tests can check
/// the allocation is interference-free.
#[derive(Debug)]
pub struct Allocation {
    /// `assign[v]` is where vreg `v` was placed (`None` if `v` is never live).
    pub assign: Vec<Option<Assign>>,
    /// The live intervals used for allocation.
    pub intervals: Intervals,
    /// The number of vregs spilled.
    pub spills: usize,
}

impl Allocation {
    /// The physical register a vreg was colored with, if it was not spilled.
    pub fn color(&self, v: VReg) -> Option<PReg> {
        match self.assign.get(v.index()).copied().flatten() {
            Some(Assign::Reg(p)) => Some(p),
            _ => None,
        }
    }
}

/// Compute liveness of `mf`. Blocks are numbered in arena order; each block's
/// points are contiguous.
pub fn compute_liveness(mf: &MachineFunction) -> Liveness {
    let order: Vec<MBlockId> = mf.block_ids().collect();
    compute_liveness_ordered(mf, &order)
}

/// Compute liveness of `mf`, numbering program points block by block in
/// `order` (any blocks it leaves out follow in arena order). Liveness itself
/// does not depend on the order; the live intervals derived from it are
/// tighter when blocks that run one after the other are numbered so (a block
/// next to its fall-through successor, an edge-copy block next to the block
/// it jumps to).
pub fn compute_liveness_ordered(mf: &MachineFunction, order: &[MBlockId]) -> Liveness {
    let n = mf.num_blocks();
    let mut seq: Vec<usize> = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    for b in order.iter().map(|b| b.index()).chain(0..n) {
        if b < n && !seen[b] {
            seen[b] = true;
            seq.push(b);
        }
    }
    let mut block_start = vec![0usize; n];
    let mut block_end = vec![0usize; n];
    let mut point = 0usize;
    for &b in &seq {
        let bid = block_id(b);
        block_start[b] = point;
        let len = mf.block(bid).insts.len().max(1);
        point += len;
        block_end[b] = point - 1;
    }
    let num_points = point;

    // Upward-exposed uses and defs per block.
    let mut use_set: Vec<DetHashSet<VReg>> = vec![DetHashSet::default(); n];
    let mut def_set: Vec<DetHashSet<VReg>> = vec![DetHashSet::default(); n];
    for b in 0..n {
        let bid = block_id(b);
        let mut defined: DetHashSet<VReg> = DetHashSet::default();
        for inst in &mf.block(bid).insts {
            for r in inst.uses() {
                if let Reg::Virtual(v) = r
                    && !defined.contains(&v)
                {
                    use_set[b].insert(v);
                }
            }
            for r in inst.defs() {
                if let Reg::Virtual(v) = r {
                    defined.insert(v);
                }
            }
        }
        def_set[b] = defined;
    }

    // Successor indices per block.
    let succ: Vec<Vec<usize>> = (0..n)
        .map(|b| mf.block(block_id(b)).successors().iter().map(|s| s.index()).collect())
        .collect();

    let mut live_in: Vec<DetHashSet<VReg>> = vec![DetHashSet::default(); n];
    let mut live_out: Vec<DetHashSet<VReg>> = vec![DetHashSet::default(); n];
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            let mut new_out: DetHashSet<VReg> = DetHashSet::default();
            for &s in &succ[b] {
                for &v in &live_in[s] {
                    new_out.insert(v);
                }
            }
            let mut new_in = use_set[b].clone();
            for &v in &new_out {
                if !def_set[b].contains(&v) {
                    new_in.insert(v);
                }
            }
            if new_out != live_out[b] {
                live_out[b] = new_out;
                changed = true;
            }
            if new_in != live_in[b] {
                live_in[b] = new_in;
                changed = true;
            }
        }
    }

    Liveness { live_in, live_out, block_start, block_end, num_points, order: seq }
}

/// Extend a vreg's interval to cover point `p`.
fn extend(per: &mut [Option<Interval>], v: VReg, p: usize) {
    match &mut per[v.index()] {
        Some(iv) => {
            iv.start = iv.start.min(p);
            iv.end = iv.end.max(p);
        }
        slot @ None => *slot = Some(Interval { start: p, end: p }),
    }
}

/// Build one conservative live interval per vreg from `mf` and its `liveness`.
pub fn build_intervals(mf: &MachineFunction, liveness: &Liveness) -> Intervals {
    let mut per_vreg: Vec<Option<Interval>> = vec![None; mf.num_vregs()];

    for b in 0..mf.num_blocks() {
        let bid = block_id(b);
        let base = liveness.block_start(b);
        for (j, inst) in mf.block(bid).insts.iter().enumerate() {
            let p = base + j;
            for r in inst.defs().chain(inst.uses()) {
                if let Reg::Virtual(v) = r {
                    extend(&mut per_vreg, v, p);
                }
            }
        }
        for &v in &liveness.live_in[b] {
            extend(&mut per_vreg, v, liveness.block_start(b));
        }
        for &v in &liveness.live_out[b] {
            extend(&mut per_vreg, v, liveness.block_end(b));
        }
    }
    Intervals { per_vreg }
}

/// One live range of a physical register (see [`fixed_ranges`]), with the
/// copies that open and close it.
#[derive(Clone, Copy, Debug)]
struct FixedRange {
    iv: Interval,
    /// The vreg the range's value was copied from (`mov preg, v` opened it).
    from: Option<VReg>,
    /// The vreg the range's value is copied into at its last point
    /// (`mov v, preg` closes it).
    to: Option<VReg>,
}

/// The live **intervals** of each physical register that appears as a fixed
/// (`Reg::Physical`) def/use operand.
///
/// This replaces the old point-based model, which recorded only the discrete
/// points a physical register was touched and was therefore *unsound*: a
/// register live *across a gap* — e.g. an ABI argument register written by an
/// arg-move and read later at the `call` — was marked at its def point and its
/// use point but not across the span between them, so a vreg whose interval sat
/// strictly inside that span was wrongly considered free to clobber it.
///
/// Instead we compute standard physical-register liveness for the pre-colored
/// operands isel emits (ABI argument/return registers, call clobbers, div's
/// `rax`/`rdx`, shift's `rcx`, ...): a register is live from a **definition** to
/// its **last use before the next definition** of that same register. A `call`,
/// which both reads an argument register and redefines it as the return
/// register, closes the incoming range and opens a fresh one at that point. A
/// use with no preceding definition in the function — an incoming convention
/// such as an argument register read by the entry prologue — conservatively
/// extends the interval back to the function start (point `0`); this is a sound
/// over-approximation and, since every such use in this codegen sits at the very
/// start, does not over-constrain in practice.
///
/// Determinism (tenet T5): each physical register's events are collected in
/// ascending program-point order (blocks in point order, instructions in list
/// order) and its intervals are derived independently from that event list, so
/// the result is a pure function of the MIR regardless of map iteration order.
#[cfg(test)]
fn fixed_intervals(mf: &MachineFunction, liveness: &Liveness) -> DetHashMap<PReg, Vec<Interval>> {
    let none = |_: &MachineInst| false;
    fixed_ranges(mf, liveness, &none)
        .into_iter()
        .map(|(p, rs)| (p, rs.into_iter().map(|r| r.iv).collect()))
        .collect()
}

/// [`fixed_intervals`] with the copies at the ends of each range, `is_move`
/// telling which instructions are register copies.
fn fixed_ranges(
    mf: &MachineFunction,
    liveness: &Liveness,
    is_move: &dyn Fn(&MachineInst) -> bool,
) -> DetHashMap<PReg, Vec<FixedRange>> {
    // Per physical register, the touched points in ascending order, each tagged
    // with whether that point defines and/or uses the register, and the vreg
    // on the other side when the point is a register copy. Instructions are
    // walked in point order, so each register's list is already sorted.
    let mut events: DetHashMap<PReg, Vec<(usize, bool, bool, Option<VReg>)>> = DetHashMap::default();
    for &b in &liveness.order {
        let bid = block_id(b);
        let base = liveness.block_start(b);
        for (j, inst) in mf.block(bid).insts.iter().enumerate() {
            let p = base + j;
            // The vreg a `mov preg, v` / `mov v, preg` copies with.
            let partner = match (is_move(inst), inst.operands.as_slice()) {
                (true, [MachineOperand::Def(Reg::Physical(_)), MachineOperand::Use(Reg::Virtual(v))])
                | (true, [MachineOperand::Def(Reg::Virtual(v)), MachineOperand::Use(Reg::Physical(_))]) => Some(*v),
                _ => None,
            };
            // Aggregate this instruction's physical def/use flags per register so
            // a register touched by several operands (e.g. a `call` that both
            // reads and redefines a register) yields a single event at this point.
            let mut local: Vec<(PReg, bool, bool)> = Vec::new();
            for (r, is_def) in inst.defs().map(|r| (r, true)).chain(inst.uses().map(|r| (r, false))) {
                if let Reg::Physical(pr) = r {
                    match local.iter_mut().find(|e| e.0 == pr) {
                        Some(e) => {
                            e.1 |= is_def;
                            e.2 |= !is_def;
                        }
                        None => local.push((pr, is_def, !is_def)),
                    }
                }
            }
            for (pr, is_def, is_use) in local {
                events.entry(pr).or_default().push((p, is_def, is_use, partner));
            }
        }
    }

    let mut out: DetHashMap<PReg, Vec<FixedRange>> = DetHashMap::default();
    for (&pr, evs) in &events {
        let mut ivs: Vec<FixedRange> = Vec::new();
        // The start of the currently-open live range (`None` if none is open),
        // the last point that range was extended to, the copy source that
        // opened it, and the copy destination at `last` (when `last` is a pure
        // use by a copy).
        let mut open_start: Option<usize> = None;
        let mut last = 0usize;
        let mut from: Option<VReg> = None;
        let mut to: Option<VReg> = None;
        for &(p, is_def, is_use, partner) in evs {
            if is_use {
                // A use consumes the current value; if none is open it comes from
                // before this region, so extend the range back to the start.
                let start = open_start.unwrap_or(0);
                if is_def {
                    // The same point reads the old value and writes a new one
                    // (a `call` using an argument register it also redefines):
                    // close the incoming range and open a fresh one here.
                    ivs.push(FixedRange { iv: Interval { start, end: p }, from, to: None });
                    open_start = Some(p);
                    from = None;
                    to = None;
                } else {
                    open_start = Some(start);
                    to = partner;
                }
                last = p;
            } else {
                // A pure definition ends any open range at its last use and starts
                // a new range at this point.
                if let Some(s) = open_start {
                    ivs.push(FixedRange { iv: Interval { start: s, end: last }, from, to });
                }
                open_start = Some(p);
                last = p;
                from = partner;
                to = None;
            }
        }
        if let Some(s) = open_start {
            ivs.push(FixedRange { iv: Interval { start: s, end: last }, from, to });
        }
        out.insert(pr, ivs);
    }
    out
}

fn block_id(index: usize) -> crate::codegen::mir::MBlockId {
    crate::support::Id::from_index(index)
}

/// How [`allocate_with`] allocates, beyond the plain linear scan of
/// [`allocate`].
#[derive(Clone, Debug, Default)]
pub struct AllocOptions {
    /// The block order program points are numbered in (see
    /// [`compute_liveness_ordered`]); `None` for arena order.
    pub order: Option<Vec<MBlockId>>,
    /// Precise allocation: live ranges with holes instead of one interval per
    /// vreg, register **hints**, and copy **coalescing**:
    ///
    /// - two vregs interfere only where their live ranges (one segment per
    ///   stretch of a block they are live in) overlap, so a value dead in the
    ///   middle of a loop leaves its register to others there;
    /// - at an instruction where one vreg dies and another is born, both may
    ///   share a register when the target says the instruction allows it
    ///   ([`MachineTarget::tied_use`]: a copy, or a two-address operation
    ///   whose destination may be its first source);
    /// - a vreg copied into a physical register (an argument register before
    ///   a call or `syscall`) may live in that register, making the copy a
    ///   no-op, as long as it is not redefined while the register's range is
    ///   live; one copied out of a physical register (a parameter, a call's
    ///   result) may take it over the same way;
    /// - each vreg prefers the register it is copied to or from most often,
    ///   copies inside loops counting for more, and otherwise the register of
    ///   a vreg it is copied or tied to.
    pub precise: bool,
}

/// Allocate registers for `mf` over `target`, rewriting it in place. Returns the
/// [`Allocation`] (assignments + intervals) for inspection. After this call no
/// virtual register remains in `mf`.
pub fn allocate(mf: &mut MachineFunction, target: &dyn MachineTarget) -> Allocation {
    allocate_with(mf, target, &AllocOptions::default())
}

/// The loop depth of each block (by index): the number of natural loops of
/// the CFG from `mf`'s entry that contain it.
fn loop_depths(mf: &MachineFunction) -> Vec<u32> {
    let n = mf.num_blocks();
    let succ: Vec<Vec<usize>> =
        (0..n).map(|b| mf.block(block_id(b)).successors().iter().map(|s| s.index()).collect()).collect();
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, ss) in succ.iter().enumerate() {
        for &s in ss {
            pred[s].push(b);
        }
    }
    let mut depth = vec![0u32; n];
    let Some(entry) = mf.entry() else { return depth };
    // Iterative DFS: an edge to a block on the stack is a back edge.
    let mut state = vec![0u8; n]; // 0 new, 1 on the stack, 2 done
    let mut back: Vec<(usize, usize)> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(entry.index(), 0)];
    state[entry.index()] = 1;
    while let Some(top) = stack.last_mut() {
        let b = top.0;
        if let Some(&s) = succ[b].get(top.1) {
            top.1 += 1;
            match state[s] {
                0 => {
                    state[s] = 1;
                    stack.push((s, 0));
                }
                1 => back.push((b, s)),
                _ => {}
            }
        } else {
            state[b] = 2;
            stack.pop();
        }
    }
    for (tail, head) in back {
        // The natural loop: the head plus everything reaching the tail without
        // passing through the head.
        let mut body = vec![false; n];
        body[head] = true;
        let mut work = vec![tail];
        while let Some(b) = work.pop() {
            if !body[b] {
                body[b] = true;
                work.extend(pred[b].iter().copied());
            }
        }
        for (b, &inside) in body.iter().enumerate() {
            if inside {
                depth[b] += 1;
            }
        }
    }
    depth
}

/// The precise live ranges of every vreg: per vreg, sorted disjoint
/// segments, each covering a stretch of one block from a definition (or the
/// block's start) to a last use (or the block's end).
fn live_segments(mf: &MachineFunction, liveness: &Liveness) -> Vec<Vec<Interval>> {
    let mut segs: Vec<Vec<Interval>> = vec![Vec::new(); mf.num_vregs()];
    for &b in &liveness.order {
        let bid = block_id(b);
        let base = liveness.block_start(b);
        // Walk backward: `open[v]` is the end of v's segment being built.
        let mut open: DetHashMap<VReg, usize> = DetHashMap::default();
        for &v in &liveness.live_out[b] {
            open.insert(v, liveness.block_end(b));
        }
        for (j, inst) in mf.block(bid).insts.iter().enumerate().rev() {
            let p = base + j;
            for r in inst.defs() {
                if let Reg::Virtual(v) = r {
                    let end = open.remove(&v).unwrap_or(p);
                    segs[v.index()].push(Interval { start: p, end });
                }
            }
            for r in inst.uses() {
                if let Reg::Virtual(v) = r {
                    open.entry(v).or_insert(p);
                }
            }
        }
        for (v, end) in open {
            segs[v.index()].push(Interval { start: base, end });
        }
    }
    // Sorting makes the result independent of the map's iteration order.
    for s in &mut segs {
        s.sort_by_key(|iv| (iv.start, iv.end));
        let mut merged: Vec<Interval> = Vec::with_capacity(s.len());
        for &iv in s.iter() {
            match merged.last_mut() {
                Some(m) if iv.start <= m.end => m.end = m.end.max(iv.end),
                _ => merged.push(iv),
            }
        }
        *s = merged;
    }
    segs
}

/// Where two sorted, disjoint segment lists overlap, piece by piece.
fn overlaps(a: &[Interval], b: &[Interval]) -> Vec<Interval> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        let s = a[i].start.max(b[j].start);
        let e = a[i].end.min(b[j].end);
        if s <= e {
            out.push(Interval { start: s, end: e });
        }
        if a[i].end < b[j].end {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Allocate registers for `mf` over `target` under `opts`, rewriting it in
/// place (see [`allocate`]).
pub fn allocate_with(mf: &mut MachineFunction, target: &dyn MachineTarget, opts: &AllocOptions) -> Allocation {
    let liveness = match &opts.order {
        Some(order) => compute_liveness_ordered(mf, order),
        None => compute_liveness(mf),
    };
    let intervals = build_intervals(mf, &liveness);
    let precise = opts.precise;
    let is_move = |inst: &MachineInst| precise && target.is_move(inst.opcode);
    let fixed = fixed_ranges(mf, &liveness, &is_move);
    let nv = mf.num_vregs();

    // Live ranges: precise segments, or the one hull interval per vreg.
    let segs: Vec<Vec<Interval>> = if precise {
        live_segments(mf, &liveness)
    } else {
        (0..nv).map(|i| intervals.per_vreg[i].into_iter().collect()).collect()
    };

    // Per point: the (def, use) vreg pair the instruction lets share a
    // register; per vreg: its definition points and its hints.
    let mut tie: DetHashMap<usize, (VReg, VReg)> = DetHashMap::default();
    let mut def_points: Vec<Vec<usize>> = vec![Vec::new(); nv];
    let mut fixed_hint: Vec<Vec<(PReg, u64)>> = vec![Vec::new(); nv];
    let mut partner: Vec<Vec<(VReg, u64)>> = vec![Vec::new(); nv];
    if precise {
        let depth = loop_depths(mf);
        for &b in &liveness.order {
            let base = liveness.block_start(b);
            let weight = 8u64.pow(depth[b].min(6));
            for (j, inst) in mf.block(block_id(b)).insts.iter().enumerate() {
                let p = base + j;
                for r in inst.defs() {
                    if let Reg::Virtual(v) = r {
                        def_points[v.index()].push(p);
                    }
                }
                if let Some(k) = target.tied_use(inst)
                    && let (Some(MachineOperand::Def(Reg::Virtual(d))), Some(MachineOperand::Use(Reg::Virtual(u)))) =
                        (inst.operands.first(), inst.operands.get(k))
                {
                    tie.insert(p, (*d, *u));
                    if d != u {
                        partner[d.index()].push((*u, weight));
                        partner[u.index()].push((*d, weight));
                    }
                }
                if target.is_move(inst.opcode) {
                    let (v, pr) = match inst.operands.as_slice() {
                        [MachineOperand::Def(Reg::Physical(pr)), MachineOperand::Use(Reg::Virtual(v))]
                        | [MachineOperand::Def(Reg::Virtual(v)), MachineOperand::Use(Reg::Physical(pr))] => (*v, *pr),
                        _ => continue,
                    };
                    match fixed_hint[v.index()].iter_mut().find(|h| h.0 == pr) {
                        Some(h) => h.1 += weight,
                        None => fixed_hint[v.index()].push((pr, weight)),
                    }
                }
            }
        }
    }

    // Precompute per-vreg class so the scan does not borrow `mf` immutably while
    // it mutates the frame for spill slots.
    let classes: Vec<_> = (0..nv).map(|i| mf.vreg_class(VReg::from_index(i))).collect();

    // Intervals sorted by (start, vreg index) for a deterministic scan.
    let mut order: Vec<(usize, usize, VReg)> = (0..nv)
        .filter_map(|i| intervals.per_vreg[i].map(|iv| (iv.start, iv.end, VReg::from_index(i))))
        .collect();
    order.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.index().cmp(&b.2.index())));

    let mut assign: Vec<Option<Assign>> = vec![None; nv];
    // Active: (end, vreg, preg), kept small.
    let mut active: Vec<(usize, VReg, PReg)> = Vec::new();
    let mut spills = 0usize;

    // Whether vregs `v` and `u` interfere: their live ranges overlap anywhere
    // but at a point whose instruction ties the one born there to the one
    // dying there.
    let interferes = |v: VReg, u: VReg| -> bool {
        overlaps(&segs[v.index()], &segs[u.index()]).iter().any(|o| {
            if o.start != o.end {
                return true;
            }
            let p = o.start;
            let Some(&(d, x)) = tie.get(&p) else { return true };
            let born = segs[d.index()].iter().any(|s| s.start == p);
            let dies = segs[x.index()].iter().any(|s| s.end == p);
            !(((d == v && x == u) || (d == u && x == v)) && born && dies)
        })
    };

    // A vreg may take `pr` only if its live range overlaps none of `pr`'s
    // live intervals (range-based, not point membership: this is what catches a
    // physical register live across a gap between its def and a distant use),
    // except where the interval only carries `v`'s own value (see
    // [`AllocOptions::precise`]).
    let fixed_conflict = |pr: PReg, v: VReg| -> bool {
        let Some(ranges) = fixed.get(&pr) else { return false };
        let vs = &segs[v.index()];
        ranges.iter().any(|r| {
            let mut iv = r.iv;
            if r.from == Some(v) {
                // `mov pr, v` opened it: compatible unless v changes while pr
                // is still live (a def at the end that copies pr back is fine).
                let redefined = def_points[v.index()]
                    .iter()
                    .any(|&d| d > iv.start && d <= iv.end && !(d == iv.end && r.to == Some(v)));
                if !redefined {
                    return false;
                }
            } else if r.to == Some(v) {
                // `mov v, pr` closes it: v may be born in pr right there.
                if iv.end == iv.start {
                    return false;
                }
                iv.end -= 1;
            }
            !overlaps(vs, &[iv]).is_empty()
        })
    };

    for &(start, end, v) in &order {
        // Expire intervals that ended before this one starts.
        active.retain(|&(a_end, _, _)| a_end >= start);
        let class = classes[v.index()];

        // The active vregs each register would collide with.
        let conflicts = |active: &[(usize, VReg, PReg)], pr: PReg| -> Vec<usize> {
            active
                .iter()
                .enumerate()
                .filter(|&(_, &(_, u, upr))| upr == pr && interferes(v, u))
                .map(|(i, _)| i)
                .collect()
        };
        let free: Vec<PReg> = target
            .allocatable(class)
            .iter()
            .copied()
            .filter(|&pr| !fixed_conflict(pr, v) && conflicts(&active, pr).is_empty())
            .collect();
        // The best-hinted free register, else the first free one.
        let mut chosen = free.first().copied();
        if precise {
            let mut best = 0u64;
            for &pr in &free {
                let mut score: u64 = fixed_hint[v.index()].iter().filter(|h| h.0 == pr).map(|h| h.1).sum();
                for &(u, w) in &partner[v.index()] {
                    if assign[u.index()] == Some(Assign::Reg(pr)) {
                        score += w;
                    }
                }
                if score > best {
                    best = score;
                    chosen = Some(pr);
                }
            }
        }

        match chosen {
            Some(pr) => {
                assign[v.index()] = Some(Assign::Reg(pr));
                active.push((end, v, pr));
            }
            None => {
                // Spill the furthest-ending interval among the current one and
                // the active intervals that alone keep a register (without a
                // fixed conflict) from it.
                let mut victim: Option<(usize, usize)> = None; // (end, active index)
                for &pr in target.allocatable(class) {
                    if fixed_conflict(pr, v) {
                        continue;
                    }
                    if let [i] = conflicts(&active, pr)[..] {
                        let a_end = active[i].0;
                        match victim {
                            Some((be, _)) if be >= a_end => {}
                            _ => victim = Some((a_end, i)),
                        }
                    }
                }
                let (size, align) = target.spill_slot(class);
                let slot = mf.frame_mut().add_slot(size, align);
                spills += 1;
                match victim {
                    Some((a_end, i)) if a_end > end => {
                        let (_, av, apr) = active[i];
                        assign[av.index()] = Some(Assign::Spill(slot));
                        active[i] = (end, v, apr);
                        assign[v.index()] = Some(Assign::Reg(apr));
                    }
                    _ => {
                        assign[v.index()] = Some(Assign::Spill(slot));
                    }
                }
            }
        }
    }

    rewrite(mf, target, &assign, &classes);
    Allocation { assign, intervals, spills }
}

/// Replace vreg operands with physical registers, inserting spill/reload code
/// (using the target's reserved scratch registers) around spilled operands.
fn rewrite(
    mf: &mut MachineFunction,
    target: &dyn MachineTarget,
    assign: &[Option<Assign>],
    classes: &[crate::codegen::mir::RegClass],
) {
    let block_ids: Vec<_> = mf.block_ids().collect();
    for bid in block_ids {
        let old = std::mem::take(&mut mf.block_mut(bid).insts);
        let mut new_insts: Vec<MachineInst> = Vec::with_capacity(old.len());
        for inst in old {
            let mut pre: Vec<MachineInst> = Vec::new();
            let mut post: Vec<MachineInst> = Vec::new();
            let mut new_ops: Vec<MachineOperand> = Vec::with_capacity(inst.operands.len());
            let mut reload_scratch: DetHashMap<VReg, PReg> = DetHashMap::default();
            let mut next_scratch = 0usize;

            for op in &inst.operands {
                match op {
                    MachineOperand::Use(Reg::Virtual(v)) => match assign[v.index()] {
                        Some(Assign::Reg(pr)) => new_ops.push(MachineOperand::Use(Reg::Physical(pr))),
                        Some(Assign::Spill(slot)) => {
                            let sc = if let Some(&s) = reload_scratch.get(v) {
                                s
                            } else {
                                let s = target.scratch(classes[v.index()])[next_scratch];
                                next_scratch += 1;
                                pre.push(target.emit_reload(s, slot));
                                reload_scratch.insert(*v, s);
                                s
                            };
                            new_ops.push(MachineOperand::Use(Reg::Physical(sc)));
                        }
                        None => unreachable!("used vreg was never assigned"),
                    },
                    MachineOperand::Def(Reg::Virtual(v)) => match assign[v.index()] {
                        Some(Assign::Reg(pr)) => new_ops.push(MachineOperand::Def(Reg::Physical(pr))),
                        Some(Assign::Spill(slot)) => {
                            let s = target.scratch(classes[v.index()])[next_scratch];
                            next_scratch += 1;
                            new_ops.push(MachineOperand::Def(Reg::Physical(s)));
                            post.push(target.emit_spill(slot, s));
                        }
                        None => unreachable!("defined vreg was never assigned"),
                    },
                    other => new_ops.push(other.clone()),
                }
            }

            new_insts.append(&mut pre);
            // Preserve the source line so debug-line info survives allocation.
            new_insts.push(MachineInst::new(inst.opcode, new_ops).with_line(inst.line));
            new_insts.append(&mut post);
        }
        mf.block_mut(bid).insts = new_insts;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, RegClass};
    use crate::codegen::vtarget::{VOp, VirtualTarget};

    use puremp::Int;

    fn gpr(n: u16) -> PReg {
        PReg::new(RegClass::Gpr, n)
    }

    fn def_phys(pr: PReg) -> MachineOperand {
        MachineOperand::Def(Reg::Physical(pr))
    }

    fn use_phys(pr: PReg) -> MachineOperand {
        MachineOperand::Use(Reg::Physical(pr))
    }

    fn def_v(v: VReg) -> MachineOperand {
        MachineOperand::Def(Reg::Virtual(v))
    }

    fn use_v(v: VReg) -> MachineOperand {
        MachineOperand::Use(Reg::Virtual(v))
    }

    fn imm(n: i64) -> MachineOperand {
        MachineOperand::Imm(Int::from_i64(n))
    }

    fn inst(op: VOp, ops: Vec<MachineOperand>) -> MachineInst {
        MachineInst::new(op.opcode(), ops)
    }

    /// Build the "live-across-a-gap" shape: a physical register `r0` is defined
    /// early (point 0) and used late (point 3), while an independent vreg `mid`
    /// is defined and used strictly inside that span (points 1..2). A later vreg
    /// `late` (point 4) lies entirely after `r0`'s last use.
    ///
    /// Returns the function plus the vregs `(mid, mid2, out, late)`.
    fn build_gap() -> (MachineFunction, VReg, VReg, VReg, VReg) {
        let mut mf = MachineFunction::new("gap", 0);
        let mid = mf.new_vreg(RegClass::Gpr);
        let mid2 = mf.new_vreg(RegClass::Gpr);
        let out = mf.new_vreg(RegClass::Gpr);
        let late = mf.new_vreg(RegClass::Gpr);
        let r0 = gpr(0);

        let b = mf.add_block();
        mf.set_entry(b);
        let block = mf.block_mut(b);
        // p0: r0 <- 1                        (physical def, opens r0's range)
        block.insts.push(inst(VOp::Li, vec![def_phys(r0), imm(1)]));
        // p1: mid <- 2                       (independent vreg def, inside the gap)
        block.insts.push(inst(VOp::Li, vec![def_v(mid), imm(2)]));
        // p2: mid2 = mid + mid               (uses mid, still inside the gap)
        block.insts.push(inst(VOp::Add, vec![def_v(mid2), use_v(mid), use_v(mid), imm(64)]));
        // p3: out <- r0                      (physical use, closes r0's range at 3)
        block.insts.push(inst(VOp::Move, vec![def_v(out), use_phys(r0)]));
        // p4: late <- 3                      (entirely after r0's last use)
        block.insts.push(inst(VOp::Li, vec![def_v(late), imm(3)]));
        // p5: ret out
        block.insts.push(inst(VOp::Ret, vec![use_v(out)]));

        (mf, mid, mid2, out, late)
    }

    #[test]
    fn physical_register_live_range_spans_the_gap() {
        let (mf, ..) = build_gap();
        let liveness = compute_liveness(&mf);
        let fixed = fixed_intervals(&mf, &liveness);
        // r0 is one contiguous live interval from its def (0) to its last use (3),
        // covering the gap — not just the two discrete endpoints.
        assert_eq!(fixed.get(&gpr(0)).map(Vec::as_slice), Some(&[Interval { start: 0, end: 3 }][..]));
    }

    #[test]
    fn vreg_in_the_gap_avoids_the_fixed_register() {
        let (mut mf, mid, mid2, out, late) = build_gap();
        let target = VirtualTarget::new();
        let alloc = allocate(&mut mf, &target);
        let r0 = gpr(0);

        // The independent vregs whose intervals sit inside r0's live range must
        // NOT be colored r0 — the old point-based model missed exactly this.
        assert_ne!(alloc.color(mid), Some(r0), "mid overlaps r0's live range");
        assert_ne!(alloc.color(mid2), Some(r0), "mid2 overlaps r0's live range");
        // `out` is defined at the very point r0 is last used, so it overlaps too.
        assert_ne!(alloc.color(out), Some(r0), "out overlaps r0's live range at its last use");
        // `late` lies entirely after r0's last use, so r0 is free to be reused.
        assert_eq!(alloc.color(late), Some(r0), "late must reuse r0 after its last use");

        // Sanity: the gap vregs really do overlap the fixed range, confirming the
        // avoidance above was meaningful and not vacuous.
        let iv = alloc.intervals.get(mid).unwrap();
        assert!(iv.overlaps(Interval { start: 0, end: 3 }));
    }

    #[test]
    fn allocation_is_deterministic() {
        let target = VirtualTarget::new();
        let colors = || {
            let (mut mf, mid, mid2, out, late) = build_gap();
            let a = allocate(&mut mf, &target);
            [a.color(mid), a.color(mid2), a.color(out), a.color(late)]
        };
        assert_eq!(colors(), colors(), "identical MIR yields an identical allocation");
    }
}
