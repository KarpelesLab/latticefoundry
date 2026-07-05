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
    MachineFunction, MachineInst, MachineOperand, PReg, Reg, StackSlot, VReg,
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
    let n = mf.num_blocks();
    let mut block_start = vec![0usize; n];
    let mut block_end = vec![0usize; n];
    let mut point = 0usize;
    for b in 0..n {
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

    Liveness { live_in, live_out, block_start, block_end, num_points }
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

/// The points at which each physical register is fixed (defined or used).
fn fixed_points(mf: &MachineFunction, liveness: &Liveness) -> DetHashMap<PReg, Vec<usize>> {
    let mut map: DetHashMap<PReg, Vec<usize>> = DetHashMap::default();
    for b in 0..mf.num_blocks() {
        let bid = block_id(b);
        let base = liveness.block_start(b);
        for (j, inst) in mf.block(bid).insts.iter().enumerate() {
            let p = base + j;
            for r in inst.defs().chain(inst.uses()) {
                if let Reg::Physical(pr) = r {
                    map.entry(pr).or_default().push(p);
                }
            }
        }
    }
    map
}

fn block_id(index: usize) -> crate::codegen::mir::MBlockId {
    crate::support::Id::from_index(index)
}

/// Allocate registers for `mf` over `target`, rewriting it in place. Returns the
/// [`Allocation`] (assignments + intervals) for inspection. After this call no
/// virtual register remains in `mf`.
pub fn allocate(mf: &mut MachineFunction, target: &dyn MachineTarget) -> Allocation {
    let liveness = compute_liveness(mf);
    let intervals = build_intervals(mf, &liveness);
    let fixed = fixed_points(mf, &liveness);

    // Precompute per-vreg class so the scan does not borrow `mf` immutably while
    // it mutates the frame for spill slots.
    let classes: Vec<_> = (0..mf.num_vregs()).map(|i| mf.vreg_class(VReg::from_index(i))).collect();

    // Intervals sorted by (start, vreg index) for a deterministic scan.
    let mut order: Vec<(usize, usize, VReg)> = (0..mf.num_vregs())
        .filter_map(|i| intervals.per_vreg[i].map(|iv| (iv.start, iv.end, VReg::from_index(i))))
        .collect();
    order.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.index().cmp(&b.2.index())));

    let mut assign: Vec<Option<Assign>> = vec![None; mf.num_vregs()];
    // Active: (end, vreg, preg), kept small.
    let mut active: Vec<(usize, VReg, PReg)> = Vec::new();
    let mut spills = 0usize;

    let fixed_conflict = |pr: PReg, s: usize, e: usize| -> bool {
        fixed.get(&pr).is_some_and(|pts| pts.iter().any(|&p| p >= s && p <= e))
    };

    for &(start, end, v) in &order {
        // Expire intervals that ended before this one starts.
        active.retain(|&(a_end, _, _)| a_end >= start);
        let class = classes[v.index()];

        let used: DetHashSet<PReg> = active.iter().map(|&(_, _, p)| p).collect();
        let mut chosen = None;
        for &pr in target.allocatable(class) {
            if !used.contains(&pr) && !fixed_conflict(pr, start, end) {
                chosen = Some(pr);
                break;
            }
        }

        match chosen {
            Some(pr) => {
                assign[v.index()] = Some(Assign::Reg(pr));
                active.push((end, v, pr));
            }
            None => {
                // Spill the furthest-ending interval among the current one and
                // the active intervals whose register does not fixed-conflict.
                let mut victim: Option<(usize, usize)> = None; // (end, active index)
                for (i, &(a_end, a_v, a_pr)) in active.iter().enumerate() {
                    if classes[a_v.index()] == class && !fixed_conflict(a_pr, start, end) {
                        match victim {
                            Some((be, _)) if be >= a_end => {}
                            _ => victim = Some((a_end, i)),
                        }
                    }
                }
                let slot = mf.frame_mut().add_slot(8, 8);
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
