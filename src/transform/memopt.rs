//! **memopt** — memory optimizations that understand the bulk-memory ops
//! (`memcpy` / `memmove` / `memset`, `docs/ir-design.md` §6k) as well as
//! loads and stores.
//!
//! Every address is traced to a **base** and a constant byte offset where
//! possible: a stack slot (`alloca`), a global, or an opaque pointer value
//! (a parameter, a loaded or returned pointer), through `ptr_add`s with
//! constant offsets (a variable offset keeps the base but forgets the offset).
//! A slot whose address is only ever used as the address of loads, stores
//! and bulk ops (never stored, passed, returned, or touched by an atomic or
//! inline asm) does not **escape**: nothing but those accesses can reach it.
//! Two accesses may alias unless they are in different slots or globals, at
//! disjoint constant ranges of the same base, or one is in a slot that does
//! not escape and the other goes through an opaque pointer.
//!
//! On that model the pass runs three rewrites, each a refinement of the
//! reference semantics:
//!
//! 1. **Scalar replacement** of slots that bulk ops touch (SROA-style). A
//!    non-escaping entry-block slot whose accesses are all at constant
//!    offsets — loads and stores of first-class types that agree on every
//!    byte range, and bulk ops of constant length — is split into one slot
//!    per accessed range (the bytes only bulk ops touch get integer slices),
//!    and each bulk op on it becomes a load and a store per slice, so a
//!    struct copy of a few words becomes scalar traffic that `mem2reg` then
//!    promotes. A slice both filled by a copy from other memory and copied
//!    out to other memory is left alone (a copy moves poison byte by byte,
//!    a load of a wider type would poison the whole slice), as is a slot a
//!    bulk op uses as both source and destination; a bulk op between two
//!    candidate slots splits one per round, and the pass runs a few rounds.
//! 2. **Forwarding** within a block: facts about what a byte range holds —
//!    a stored (or loaded) value, a `memset` byte, or a copy of another range
//!    — survive until a write that may alias the range (or the copy's source)
//!    or an opaque effect (a call, a syscall, an atomic, a fence, a volatile
//!    access or a memory-touching inline asm, which keep only facts about
//!    non-escaping slots). A load of exactly a stored value becomes that
//!    value; a load from a constant fill becomes the replicated constant; a
//!    load from a copy reads the original. A constant-length `memcpy` of
//!    exactly a stored value becomes a store of the value, of a filled range
//!    a `memset`, of a copied range a copy from the original (a `memmove`
//!    unless the two ranges provably do not overlap). A bulk op of length 0
//!    is deleted.
//! 3. **Dead writes** within a block: a store or constant-length bulk write
//!    whose whole range is overwritten later in the block before anything
//!    may read it is deleted, and so is any write to a non-escaping slot
//!    after which the block returns without reading it.
//!
//! Volatile accesses are never rewritten, merged or deleted, and they (like
//! atomics and fences) are barriers for the facts and dead-write ranges of
//! escaping memory.

use std::collections::HashMap;

use puremp::Int;

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, CastOp, Flags, InstKind};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, GlobalId, InstId, Module};
use crate::pass::{Changed, ModulePass};
use crate::transform::{dom_preorder, rebuild_terminator, remap_value};

/// The memory-optimization pass (see the module documentation).
#[derive(Debug, Default, Clone, Copy)]
pub struct MemOpt;

/// Scalar-replacement rounds per function.
const SROA_ROUNDS: usize = 4;
/// The largest slot scalar replacement splits, in bytes.
const SROA_MAX_BYTES: u64 = 256;
/// The most slices a slot is split into.
const SROA_MAX_SLICES: usize = 32;

impl ModulePass for MemOpt {
    fn name(&self) -> &str {
        "memopt"
    }

    fn run(&mut self, module: &mut Module) -> Changed {
        let mut changed = Changed::No;
        for i in 0..module.function_count() {
            let id = FuncId::from_index(i);
            if module.function(id).is_declaration() {
                continue;
            }
            if optimize_function(module, id) {
                changed = Changed::Yes;
            }
        }
        changed
    }
}

/// Run the three rewrites on function `id`; whether anything changed.
pub(crate) fn optimize_function(module: &mut Module, id: FuncId) -> bool {
    let mut changed = false;
    for _ in 0..SROA_ROUNDS {
        let plan = {
            let f = module.function(id);
            let mem = MemInfo::new(module, f);
            sroa_plan(module, f, &mem)
        };
        if plan.slots.is_empty() {
            break;
        }
        let (fresh, ()) = module.map_function(id, |old, b| apply_sroa(old, b, &plan));
        module.replace_function(id, fresh);
        changed = true;
    }
    let rw = {
        let f = module.function(id);
        let mem = MemInfo::new(module, f);
        let mut rw = forward_plan(module, f, &mem);
        dead_write_plan(module, f, &mem, &mut rw);
        rw
    };
    if !rw.is_empty() {
        let (fresh, ()) = module.map_function(id, |old, b| apply_rewrites(old, b, &rw));
        module.replace_function(id, fresh);
        changed = true;
    }
    changed
}

// ---------------------------------------------------------------------------
// The memory model.
// ---------------------------------------------------------------------------

/// What an address is relative to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Base {
    /// A stack slot (its `alloca`).
    Slot(InstId),
    /// A global, named directly.
    Global(GlobalId),
    /// An opaque pointer value.
    Value(ValueId),
}

/// An address: a base and, when constant, the byte offset from it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Loc {
    base: Base,
    off: Option<i64>,
}

/// Per-function address and escape facts.
struct MemInfo {
    loc: Vec<Loc>,
    escapes: HashMap<InstId, bool>,
}

/// The integer constant `v` is, sign-extended from its width.
fn const_int(module: &Module, f: &Function, v: ValueId) -> Option<i64> {
    let ValueDef::Const(c) = f.value(v).def else { return None };
    let Const::Int { ty, value } = module.consts().get(c) else { return None };
    let w = module.types().bit_width(*ty)?;
    let u = value.mod_2k(w.min(64)).to_u64()?;
    Some(if w < 64 && w > 0 && (u >> (w - 1)) & 1 == 1 { (u | !((1u64 << w) - 1)) as i64 } else { u as i64 })
}

/// The unsigned constant `v` is.
fn const_u64(module: &Module, f: &Function, v: ValueId) -> Option<u64> {
    let ValueDef::Const(c) = f.value(v).def else { return None };
    match module.consts().get(c) {
        Const::Int { value, .. } => value.to_u64(),
        _ => None,
    }
}

impl MemInfo {
    fn new(module: &Module, f: &Function) -> MemInfo {
        let n = f.value_count();
        let mut loc: Vec<Option<Loc>> = vec![None; n];
        fn resolve(module: &Module, f: &Function, v: ValueId, loc: &mut [Option<Loc>], depth: u32) -> Loc {
            if let Some(l) = loc[v.index()] {
                return l;
            }
            let l = match f.value(v).def {
                ValueDef::Global(g) => Loc { base: Base::Global(g), off: Some(0) },
                ValueDef::Inst(i) if depth < 64 => match f.inst(i).kind {
                    InstKind::Alloca { .. } => Loc { base: Base::Slot(i), off: Some(0) },
                    InstKind::PtrAdd { .. } => {
                        let ops = f.inst(i).operands();
                        let b = resolve(module, f, ops[0], loc, depth + 1);
                        let c = const_int(module, f, ops[1]);
                        Loc { base: b.base, off: b.off.zip(c).and_then(|(o, c)| o.checked_add(c)) }
                    }
                    _ => Loc { base: Base::Value(v), off: Some(0) },
                },
                _ => Loc { base: Base::Value(v), off: Some(0) },
            };
            loc[v.index()] = Some(l);
            l
        }
        for i in 0..n {
            resolve(module, f, ValueId::from_index(i), &mut loc, 0);
        }
        let loc: Vec<Loc> = loc.into_iter().map(|l| l.expect("resolved")).collect();
        let mut escapes = HashMap::new();
        for i in 0..f.inst_count() {
            let id = InstId::from_index(i);
            if let (InstKind::Alloca { .. }, Some(r)) = (&f.inst(id).kind, f.inst(id).result()) {
                escapes.insert(id, slot_escapes(f, r));
            }
        }
        MemInfo { loc, escapes }
    }

    fn loc(&self, v: ValueId) -> Loc {
        self.loc[v.index()]
    }

    /// Whether memory at `base` may be reached by code other than this
    /// function's own direct accesses (a callee, another thread, an opaque
    /// pointer).
    fn escapable(&self, base: Base) -> bool {
        match base {
            Base::Slot(i) => self.escapes.get(&i).copied().unwrap_or(true),
            _ => true,
        }
    }

    /// Whether `[a, a + asz)` and `[b, b + bsz)` may overlap (`None` sizes are
    /// unbounded).
    fn may_alias(&self, a: Loc, asz: Option<u64>, b: Loc, bsz: Option<u64>) -> bool {
        if a.base == b.base {
            return match (a.off, b.off) {
                (Some(x), Some(y)) => {
                    let end = |o: i64, s: Option<u64>| s.map_or(i128::MAX, |s| i128::from(o) + i128::from(s));
                    i128::from(x) < end(y, bsz) && i128::from(y) < end(x, asz)
                }
                _ => true,
            };
        }
        match (a.base, b.base) {
            (Base::Value(_), Base::Value(_)) => true,
            (Base::Value(_), other) | (other, Base::Value(_)) => self.escapable(other),
            // Two different slots or globals, or a slot and a global.
            _ => false,
        }
    }
}

/// Whether slot address `v` (an `alloca` result) is used as anything but the
/// address of a load, store or bulk op, or the base of a `ptr_add`.
fn slot_escapes(f: &Function, v: ValueId) -> bool {
    let mut work = vec![v];
    let mut seen = vec![false; f.value_count()];
    while let Some(p) = work.pop() {
        if std::mem::replace(&mut seen[p.index()], true) {
            continue;
        }
        for u in f.uses_of(p) {
            let data = f.inst(u.inst);
            match &data.kind {
                InstKind::Load { .. } if u.operand == 0 => {}
                InstKind::Store { .. } if u.operand == 0 => {}
                InstKind::MemCopy { .. } if u.operand < 2 => {}
                InstKind::MemSet { .. } if u.operand == 0 => {}
                InstKind::PtrAdd { .. } if u.operand == 0 => match data.result() {
                    Some(r) => work.push(r),
                    None => return true,
                },
                _ => return true,
            }
        }
    }
    false
}

/// The alignment known for byte `off` of a range aligned to `align`.
fn align_at(align: u32, off: u64) -> u32 {
    let align = align.max(1);
    if off == 0 { align } else { align.min(1u32 << off.trailing_zeros().min(31)) }
}

/// The parts of a bulk op: `(dst, mid, n, align, volatile, is_set)`.
fn bulk_parts(f: &Function, i: InstId) -> Option<(ValueId, ValueId, ValueId, u32, bool, bool)> {
    let inst = f.inst(i);
    let ops = inst.operands();
    match inst.kind {
        InstKind::MemCopy { align, volatile, .. } => Some((ops[0], ops[1], ops[2], align, volatile, false)),
        InstKind::MemSet { align, volatile } => Some((ops[0], ops[1], ops[2], align, volatile, true)),
        _ => None,
    }
}

/// Whether `kind` may touch memory in ways the model does not track
/// precisely (a call, syscall, atomic, fence, volatile access or memory
/// inline asm): it may read and write any escapable memory.
fn opaque_effect(kind: &InstKind) -> bool {
    match kind {
        InstKind::Call | InstKind::Syscall => true,
        InstKind::InlineAsm(asm) => asm.may_access_memory() || asm.has_side_effect(),
        k => k.is_atomic() || k.is_volatile(),
    }
}

// ---------------------------------------------------------------------------
// Scalar replacement.
// ---------------------------------------------------------------------------

/// One slice of a split slot: its byte offset and type.
#[derive(Clone, Copy, Debug)]
struct Slice {
    off: u64,
    /// The slice type; `None` for an integer of `size` bytes (bytes only
    /// bulk ops touch).
    ty: Option<TypeId>,
    size: u64,
}

/// The split of the slots replaced this round.
#[derive(Default, Debug)]
struct SroaPlan {
    /// Split slot → its slices (sorted by offset).
    slots: HashMap<InstId, Vec<Slice>>,
}

/// Whether `ty` is a first-class value a slice may hold.
fn first_class(types: &TypeContext, ty: TypeId) -> bool {
    matches!(types.get(ty), Type::Int(_) | Type::Float(_) | Type::Ptr | Type::PtrIn(_) | Type::Vector(..))
}

/// One access to a candidate slot.
enum Access {
    Value { off: u64, ty: TypeId },
    Bulk { inst: InstId, off: u64, n: u64, dst: bool, copy: bool },
}

fn sroa_plan(module: &Module, f: &Function, mem: &MemInfo) -> SroaPlan {
    let types = module.types();
    let mut plan = SroaPlan::default();
    let Some(entry) = f.entry() else { return plan };
    // Bulk ops already claimed by a slot split this round.
    let mut claimed: Vec<InstId> = Vec::new();
    for &a in f.block(entry).insts() {
        let InstKind::Alloca { elem_ty } = f.inst(a).kind else { continue };
        if mem.escapable(Base::Slot(a)) {
            continue;
        }
        let size = types.size_of(elem_ty);
        if size == 0 || size > SROA_MAX_BYTES {
            continue;
        }
        let Some(accesses) = slot_accesses(module, f, mem, a, size) else { continue };
        // A slot accessed whole, as its own type, is mem2reg's already.
        if accesses.iter().all(|x| matches!(*x, Access::Value { off: 0, ty } if ty == elem_ty)) {
            continue;
        }
        if accesses.iter().any(|x| matches!(x, Access::Bulk { inst, .. } if claimed.contains(inst))) {
            continue;
        }
        let Some(slices) = slices_for(module, &accesses, size) else { continue };
        for x in &accesses {
            if let Access::Bulk { inst, .. } = x {
                claimed.push(*inst);
            }
        }
        plan.slots.insert(a, slices);
    }
    plan
}

/// Every access to slot `a` (of `size` bytes), or `None` when one is not at
/// a constant in-bounds offset, is volatile, or has a variable length.
fn slot_accesses(module: &Module, f: &Function, mem: &MemInfo, a: InstId, size: u64) -> Option<Vec<Access>> {
    let types = module.types();
    let mut out = Vec::new();
    let r = f.inst(a).result()?;
    let mut work = vec![r];
    let in_bounds = |off: i64, len: u64| off >= 0 && (off as u64).checked_add(len).is_some_and(|e| e <= size);
    while let Some(p) = work.pop() {
        for u in f.uses_of(p) {
            let data = f.inst(u.inst);
            let off = mem.loc(p).off?;
            match &data.kind {
                InstKind::PtrAdd { .. } => {
                    let r = data.result()?;
                    mem.loc(r).off?;
                    work.push(r);
                }
                InstKind::Load { ty, volatile: false, .. } | InstKind::Store { ty, volatile: false, .. } => {
                    if !first_class(types, *ty) || !in_bounds(off, types.size_of(*ty)) {
                        return None;
                    }
                    out.push(Access::Value { off: off as u64, ty: *ty });
                }
                InstKind::MemCopy { volatile: false, .. } | InstKind::MemSet { volatile: false, .. } => {
                    let (d, m, n, ..) = bulk_parts(f, u.inst)?;
                    let n = const_u64(module, f, n)?;
                    let copy = matches!(data.kind, InstKind::MemCopy { .. });
                    // Both ends in this slot: left alone.
                    if copy && mem.loc(d).base == mem.loc(m).base {
                        return None;
                    }
                    if !in_bounds(off, n) {
                        return None;
                    }
                    out.push(Access::Bulk { inst: u.inst, off: off as u64, n, dst: u.operand == 0, copy });
                }
                _ => return None,
            }
        }
    }
    Some(out)
}

/// The slices of a slot of `size` bytes with `accesses`, or `None` when the
/// accesses do not split cleanly.
fn slices_for(module: &Module, accesses: &[Access], size: u64) -> Option<Vec<Slice>> {
    let types = module.types();
    let mut slices: Vec<Slice> = Vec::new();
    for x in accesses {
        if let Access::Value { off, ty } = *x {
            let sz = types.size_of(ty);
            match slices.iter().find(|s| s.off < off + sz && off < s.off + s.size) {
                Some(s) if s.off == off && s.ty == Some(ty) => {}
                Some(_) => return None,
                None => slices.push(Slice { off, ty: Some(ty), size: sz }),
            }
        }
    }
    // The boundaries every slice must respect: each bulk range's ends.
    let mut cuts: Vec<u64> = vec![0, size];
    for x in accesses {
        if let Access::Bulk { off, n, .. } = *x {
            cuts.push(off);
            cuts.push(off + n);
        }
    }
    for s in &slices {
        if cuts.iter().any(|&c| s.off < c && c < s.off + s.size) {
            return None;
        }
        cuts.push(s.off);
        cuts.push(s.off + s.size);
    }
    cuts.sort_unstable();
    cuts.dedup();
    // Integer slices for the bytes only bulk ops touch.
    let touched = |b: u64| accesses.iter().any(|x| matches!(*x, Access::Bulk { off, n, .. } if off <= b && b < off + n));
    let mut extra: Vec<Slice> = Vec::new();
    for w in cuts.windows(2) {
        let (mut lo, hi) = (w[0], w[1]);
        while lo < hi {
            if slices.iter().any(|s| s.off <= lo && lo < s.off + s.size) {
                lo += 1;
                continue;
            }
            let mut sz = 8u64;
            while sz > 1 && (lo % sz != 0 || lo + sz > hi || slices.iter().any(|s| s.off < lo + sz && lo < s.off + s.size)) {
                sz /= 2;
            }
            if touched(lo) {
                extra.push(Slice { off: lo, ty: None, size: sz });
            }
            lo += sz;
        }
    }
    slices.extend(extra);
    slices.sort_by_key(|s| s.off);
    if slices.len() > SROA_MAX_SLICES {
        return None;
    }
    // A slice filled by a copy from elsewhere and copied out again keeps
    // per-byte poison only in memory: leave such slots alone.
    for s in &slices {
        let covered = |want_dst: bool| {
            accesses.iter().any(|x| {
                matches!(*x, Access::Bulk { off, n, dst, copy: true, .. }
                    if dst == want_dst && off <= s.off && s.off + s.size <= off + n)
            })
        };
        if s.size > 1 && covered(true) && covered(false) {
            return None;
        }
    }
    Some(slices)
}

/// The pre-interned types the rewrites need (the builder interns on demand).
fn int_ty(b: &mut FunctionBuilder<'_>, bytes: u64) -> TypeId {
    b.types_mut().int((8 * bytes) as u32)
}

/// `byte` replicated across a value of type `ty` (`size` bytes).
fn splat_value(b: &mut FunctionBuilder<'_>, byte: ValueId, ty: TypeId, size: u64) -> ValueId {
    let ity = int_ty(b, size);
    let known = b.const_of(byte).and_then(|c| match b.consts().get(c) {
        Const::Int { value, .. } => value.to_u64(),
        _ => None,
    });
    let iv = match known {
        Some(k) => {
            let mut x = Int::ZERO;
            for _ in 0..size {
                x = x.mul_2k(8).add(&Int::from_u64(k & 0xff));
            }
            b.const_int(ity, x)
        }
        None if size == 1 => byte,
        None => {
            let mut v = b.cast(CastOp::ZExt, byte, ity);
            let mut shift = 8u64;
            while shift < size * 8 {
                let amt = b.const_int(ity, Int::from_u64(shift));
                let hi = b.bin(BinOp::Shl, v, amt, Flags::NONE);
                v = b.bin(BinOp::Or, v, hi, Flags::NONE);
                shift *= 2;
            }
            v
        }
    };
    if ity == ty {
        return iv;
    }
    let op = if b.types().is_ptr(ty) { CastOp::IntToPtr } else { CastOp::Bitcast };
    b.cast(op, iv, ty)
}

/// `p + delta` (or `p` itself).
fn offset(b: &mut FunctionBuilder<'_>, p: ValueId, delta: i64) -> ValueId {
    if delta == 0 {
        return p;
    }
    let pbits = b.types().data_layout().pointer_bits(0);
    let ity = b.types_mut().int(pbits);
    let d = b.const_int(ity, Int::from_i64(delta));
    b.ptr_add(p, d, false)
}

/// Rebuild `old` with the slots of `plan` split.
fn apply_sroa(old: &Function, b: &mut FunctionBuilder<'_>, plan: &SroaPlan) {
    // The (old) values derived from a split slot: slot and offset.
    let mut derived: HashMap<ValueId, (InstId, u64)> = HashMap::new();
    for &a in plan.slots.keys() {
        let r = old.inst(a).result().expect("an alloca result");
        let mut work = vec![(r, 0u64)];
        while let Some((p, off)) = work.pop() {
            derived.insert(p, (a, off));
            for u in old.uses_of(p) {
                let data = old.inst(u.inst);
                if matches!(data.kind, InstKind::PtrAdd { .. }) && u.operand == 0 {
                    let c = data.operands()[1];
                    let delta = match old.value(c).def {
                        ValueDef::Const(_) => b.const_of_old(old, c),
                        _ => None,
                    }
                    .expect("constant offsets only");
                    work.push((data.result().expect("ptr_add result"), (off as i64 + delta) as u64));
                }
            }
        }
    }
    let mut slice_vals: HashMap<InstId, Vec<(Slice, ValueId)>> = HashMap::new();
    rebuild_with(old, b, |b, vmap, i, first_in_entry| {
        if first_in_entry {
            // The slices, at the top of the entry block.
            let mut keys: Vec<InstId> = plan.slots.keys().copied().collect();
            keys.sort_unstable();
            for a in keys {
                b.set_line_from(old, a);
                let vals = plan.slots[&a]
                    .iter()
                    .map(|s| {
                        let ty = s.ty.unwrap_or_else(|| int_ty(b, s.size));
                        (Slice { ty: Some(ty), ..*s }, b.alloca(ty))
                    })
                    .collect();
                slice_vals.insert(a, vals);
            }
        }
        let inst = old.inst(i);
        if let Some(r) = inst.result()
            && derived.contains_key(&r)
        {
            return true; // the split slot and its addresses
        }
        let slice_at = |p: ValueId, off_extra: u64| -> Option<(Slice, ValueId)> {
            let &(a, off) = derived.get(&p)?;
            slice_vals[&a].iter().copied().find(|(s, _)| s.off == off + off_extra)
        };
        match &inst.kind {
            InstKind::Load { ty, align, volatile, secret } if derived.contains_key(&inst.operands()[0]) => {
                let (s, sv) = slice_at(inst.operands()[0], 0).expect("a slice per access");
                let natural = b.types().align_of(s.ty.expect("resolved")).max(1) as u32;
                let kind = InstKind::Load { ty: *ty, align: (*align).min(natural), volatile: *volatile, secret: *secret };
                let r = b.append_inst(kind, vec![sv], Flags::NONE, Some(*ty));
                vmap[inst.result().expect("load result").index()] = r;
                true
            }
            InstKind::Store { ty, align, volatile, secret } if derived.contains_key(&inst.operands()[0]) => {
                let (s, sv) = slice_at(inst.operands()[0], 0).expect("a slice per access");
                let natural = b.types().align_of(s.ty.expect("resolved")).max(1) as u32;
                let v = remap_value(vmap, old, b, inst.operands()[1]);
                let kind = InstKind::Store { ty: *ty, align: (*align).min(natural), volatile: *volatile, secret: *secret };
                b.append_inst(kind, vec![sv, v], Flags::NONE, None);
                true
            }
            InstKind::MemCopy { .. } | InstKind::MemSet { .. } => {
                let (d, m, n, align, _, is_set) = bulk_parts(old, i).expect("a bulk op");
                let (slot_side, other, dst_is_slot) = if derived.contains_key(&d) {
                    (d, m, true)
                } else if !is_set && derived.contains_key(&m) {
                    (m, d, false)
                } else {
                    return false;
                };
                let n = b.const_of_old(old, n).expect("a constant length") as u64;
                let &(a, base_off) = derived.get(&slot_side).expect("derived");
                let slices: Vec<(Slice, ValueId)> = slice_vals[&a]
                    .iter()
                    .copied()
                    .filter(|(s, _)| base_off <= s.off && s.off + s.size <= base_off + n)
                    .collect();
                let other = remap_value(vmap, old, b, other);
                for (s, sv) in slices {
                    let rel = s.off - base_off;
                    let natural = b.types().align_of(s.ty.expect("resolved")).max(1) as u32;
                    let ty = s.ty.expect("resolved");
                    if is_set {
                        let v = splat_value(b, other, ty, s.size);
                        b.store(ty, sv, v, natural);
                    } else {
                        let p = offset(b, other, rel as i64);
                        let a_other = align_at(align, rel);
                        if dst_is_slot {
                            let v = b.load(ty, p, a_other);
                            b.store(ty, sv, v, natural);
                        } else {
                            let v = b.load(ty, sv, natural);
                            b.store(ty, p, v, a_other);
                        }
                    }
                }
                true
            }
            _ => false,
        }
    });
}

/// Helpers on the builder for old-function constants.
trait OldConst {
    /// The signed value of integer constant `v` of `old`.
    fn const_of_old(&self, old: &Function, v: ValueId) -> Option<i64>;
}

impl OldConst for FunctionBuilder<'_> {
    fn const_of_old(&self, old: &Function, v: ValueId) -> Option<i64> {
        let ValueDef::Const(c) = old.value(v).def else { return None };
        let Const::Int { ty, value } = self.consts().get(c) else { return None };
        let w = self.types().bit_width(*ty)?;
        let u = value.mod_2k(w.min(64)).to_u64()?;
        Some(if w < 64 && w > 0 && (u >> (w - 1)) & 1 == 1 { (u | !((1u64 << w) - 1)) as i64 } else { u as i64 })
    }
}

// ---------------------------------------------------------------------------
// Forwarding and dead writes.
// ---------------------------------------------------------------------------

/// A rewrite of one old instruction.
#[derive(Clone, Debug)]
enum Rewrite {
    /// Drop it.
    Delete,
    /// A load whose result is this old value.
    LoadVal(ValueId),
    /// A load whose result is this integer constant.
    LoadConst(TypeId, Int),
    /// A load from `ptr + delta` instead, at this alignment.
    LoadFrom { ptr: ValueId, delta: i64, align: u32 },
    /// A `memcpy` that becomes `store v` of `ty` to its destination.
    StoreVal { v: ValueId, ty: TypeId },
    /// A `memcpy` that becomes a `memset` of `byte`.
    SetFrom { byte: ValueId },
    /// A `memcpy` that copies from `ptr + delta` (a `memmove` unless
    /// `disjoint`), with source alignment `align`.
    CopyFrom { ptr: ValueId, delta: i64, align: u32, disjoint: bool },
}

/// What a byte range is known to hold.
#[derive(Clone, Copy, Debug)]
enum Content {
    /// The value `v` of type `ty` (exactly the range).
    Val { v: ValueId, ty: TypeId },
    /// Every byte is `byte` (an `i8` value).
    Fill { byte: ValueId },
    /// A copy of the range at `src` (an old pointer value whose location
    /// is `src_loc`), whose start was aligned to `align`.
    Copy { src: ValueId, src_base: Base, src_off: i64, align: u32 },
}

#[derive(Clone, Copy, Debug)]
struct Fact {
    base: Base,
    off: i64,
    size: u64,
    content: Content,
}

impl Fact {
    fn covers(&self, base: Base, off: i64, size: u64) -> bool {
        self.base == base && self.off <= off && i128::from(off) + i128::from(size) <= i128::from(self.off) + i128::from(self.size)
    }
}

/// Drop the facts a write of `[at, at + size)` may invalidate.
fn kill(facts: &mut Vec<Fact>, mem: &MemInfo, at: Loc, size: Option<u64>) {
    facts.retain(|f| {
        let here = Loc { base: f.base, off: Some(f.off) };
        if mem.may_alias(here, Some(f.size), at, size) {
            return false;
        }
        if let Content::Copy { src_base, src_off, .. } = f.content {
            let from = Loc { base: src_base, off: Some(src_off) };
            if mem.may_alias(from, Some(f.size), at, size) {
                return false;
            }
        }
        true
    });
}

/// Drop the facts an opaque effect may invalidate.
fn kill_escapable(facts: &mut Vec<Fact>, mem: &MemInfo) {
    facts.retain(|f| {
        !mem.escapable(f.base) && !matches!(f.content, Content::Copy { src_base, .. } if mem.escapable(src_base))
    });
}

/// The forwarding rewrites of `f` (block-local).
fn forward_plan(module: &Module, f: &Function, mem: &MemInfo) -> HashMap<InstId, Rewrite> {
    let types = module.types();
    let mut rw = HashMap::new();
    for (_, block) in f.blocks() {
        let mut facts: Vec<Fact> = Vec::new();
        for &i in block.insts() {
            let inst = f.inst(i);
            let ops = inst.operands();
            match &inst.kind {
                InstKind::Load { ty, align, volatile: false, .. } => {
                    let l = mem.loc(ops[0]);
                    let size = types.size_of(*ty);
                    let Some(off) = l.off else { continue };
                    let hit = facts.iter().rev().find(|x| x.covers(l.base, off, size)).copied();
                    let r = inst.result().expect("a load result");
                    match hit.map(|h| (h, h.content)) {
                        Some((h, Content::Val { v, ty: vt })) if vt == *ty && h.off == off && h.size == size => {
                            rw.insert(i, Rewrite::LoadVal(v));
                            continue;
                        }
                        Some((_, Content::Fill { byte })) => {
                            if let (Some(k), Type::Int(w)) = (const_u64(module, f, byte), types.get(*ty)) {
                                let mut x = Int::ZERO;
                                for _ in 0..size {
                                    x = x.mul_2k(8).add(&Int::from_u64(k & 0xff));
                                }
                                rw.insert(i, Rewrite::LoadConst(*ty, x.mod_2k(*w)));
                                continue;
                            }
                            if size == 1 && matches!(types.get(*ty), Type::Int(8)) {
                                rw.insert(i, Rewrite::LoadVal(byte));
                                continue;
                            }
                        }
                        Some((h, Content::Copy { src, src_off, align: ca, src_base })) => {
                            let delta = off - h.off;
                            let src_loc_off = src_off + delta;
                            let a = (*align).min(align_at(ca, delta as u64));
                            let base_off = mem.loc(src).off.unwrap_or(0);
                            rw.insert(i, Rewrite::LoadFrom { ptr: src, delta: src_loc_off - base_off, align: a });
                            // The load now reads the source; remember its value there too.
                            facts.push(Fact { base: src_base, off: src_loc_off, size, content: Content::Val { v: r, ty: *ty } });
                            continue;
                        }
                        _ => {}
                    }
                    facts.push(Fact { base: l.base, off, size, content: Content::Val { v: r, ty: *ty } });
                }
                InstKind::Store { ty, volatile: false, .. } => {
                    let l = mem.loc(ops[0]);
                    let size = types.size_of(*ty);
                    kill(&mut facts, mem, l, Some(size));
                    if let Some(off) = l.off {
                        facts.push(Fact { base: l.base, off, size, content: Content::Val { v: ops[1], ty: *ty } });
                    }
                }
                InstKind::MemSet { volatile: false, .. } => {
                    let l = mem.loc(ops[0]);
                    let n = const_u64(module, f, ops[2]);
                    if n == Some(0) {
                        rw.insert(i, Rewrite::Delete);
                        continue;
                    }
                    kill(&mut facts, mem, l, n);
                    if let (Some(off), Some(n)) = (l.off, n) {
                        facts.push(Fact { base: l.base, off, size: n, content: Content::Fill { byte: ops[1] } });
                    }
                }
                InstKind::MemCopy { align, volatile: false, .. } => {
                    let (d, s) = (mem.loc(ops[0]), mem.loc(ops[1]));
                    let n = const_u64(module, f, ops[2]);
                    if n == Some(0) {
                        rw.insert(i, Rewrite::Delete);
                        continue;
                    }
                    // What the source range is known to hold.
                    let mut content: Option<Content> = None;
                    if let (Some(n), Some(soff)) = (n, s.off) {
                        let hit = facts.iter().rev().find(|x| x.covers(s.base, soff, n)).copied();
                        match hit.map(|h| (h, h.content)) {
                            Some((h, Content::Val { v, ty })) if h.off == soff && h.size == n && first_class(types, ty) => {
                                rw.insert(i, Rewrite::StoreVal { v, ty });
                                content = Some(Content::Val { v, ty });
                            }
                            Some((_, Content::Fill { byte })) => {
                                rw.insert(i, Rewrite::SetFrom { byte });
                                content = Some(Content::Fill { byte });
                            }
                            Some((h, Content::Copy { src, src_base, src_off, align: ca })) => {
                                let delta = soff - h.off;
                                let from = Loc { base: src_base, off: Some(src_off + delta) };
                                let disjoint = !mem.may_alias(from, Some(n), d, Some(n));
                                let a = (*align).min(align_at(ca, delta as u64));
                                let base_off = mem.loc(src).off.unwrap_or(0);
                                rw.insert(
                                    i,
                                    Rewrite::CopyFrom { ptr: src, delta: src_off + delta - base_off, align: a, disjoint },
                                );
                                content = Some(Content::Copy { src, src_base, src_off: src_off + delta, align: a });
                            }
                            _ => {}
                        }
                        if content.is_none() && s.base != d.base {
                            content = Some(Content::Copy { src: ops[1], src_base: s.base, src_off: soff, align: *align });
                        }
                    }
                    kill(&mut facts, mem, d, n);
                    if let (Some(off), Some(n), Some(c)) = (d.off, n, content) {
                        facts.push(Fact { base: d.base, off, size: n, content: c });
                    }
                }
                k if opaque_effect(k) => {
                    kill_escapable(&mut facts, mem);
                    // A volatile or atomic access may also write its own
                    // address (whose base may be a non-escaping slot).
                    if (k.is_volatile() || k.is_atomic())
                        && let Some(&p) = ops.first()
                    {
                        kill(&mut facts, mem, mem.loc(p), None);
                    }
                }
                _ => {}
            }
        }
    }
    rw
}

/// Byte-range sets per base, for the dead-write walk.
#[derive(Default)]
struct Covered {
    ranges: Vec<(Base, i64, i64)>,
}

impl Covered {
    fn contains(&self, base: Base, lo: i64, hi: i64) -> bool {
        // The union of the ranges of `base` must cover `[lo, hi)`.
        let mut v: Vec<(i64, i64)> = self.ranges.iter().filter(|r| r.0 == base).map(|r| (r.1, r.2)).collect();
        v.sort_unstable();
        let mut at = lo;
        for (a, b) in v {
            if a > at {
                break;
            }
            at = at.max(b);
            if at >= hi {
                return true;
            }
        }
        at >= hi
    }

    fn add(&mut self, base: Base, lo: i64, hi: i64) {
        self.ranges.push((base, lo, hi));
    }

    /// Forget what a read of `[at, at + size)` may observe.
    fn read(&mut self, mem: &MemInfo, at: Loc, size: Option<u64>) {
        let mut out = Vec::new();
        for &(base, lo, hi) in &self.ranges {
            let here = Loc { base, off: Some(lo) };
            let len = hi.abs_diff(lo);
            if !mem.may_alias(here, Some(len), at, size) {
                out.push((base, lo, hi));
                continue;
            }
            // Same base and known offsets: keep the parts outside the read.
            if base == at.base
                && let Some(o) = at.off
            {
                let end = size.map_or(i64::MAX, |s| o.saturating_add(s as i64));
                if lo < o {
                    out.push((base, lo, hi.min(o)));
                }
                if end < hi {
                    out.push((base, lo.max(end), hi));
                }
            }
        }
        self.ranges = out;
    }

    fn read_escapable(&mut self, mem: &MemInfo) {
        self.ranges.retain(|r| !mem.escapable(r.0));
    }
}

/// The dead writes of `f` (block-local), added to `rw` (skipping
/// instructions it already rewrites).
fn dead_write_plan(module: &Module, f: &Function, mem: &MemInfo, rw: &mut HashMap<InstId, Rewrite>) {
    let types = module.types();
    // The non-escaping slots (dead once the function returns).
    let mut locals: Vec<InstId> = mem.escapes.iter().filter(|(_, e)| !**e).map(|(i, _)| *i).collect();
    locals.sort_unstable();
    // A non-escaping slot nothing reads: its writes and the slot itself go.
    for &s in &locals {
        if let Some(writes) = write_only(f, s) {
            for w in writes {
                rw.insert(w, Rewrite::Delete);
            }
            rw.insert(s, Rewrite::Delete);
        }
    }
    for (_, block) in f.blocks() {
        let mut cov = Covered::default();
        if let Some(t) = block.terminator()
            && matches!(f.inst(t).kind, InstKind::Ret)
        {
            for &s in &locals {
                cov.add(Base::Slot(s), i64::MIN, i64::MAX);
            }
        }
        for &i in block.insts().iter().rev() {
            let inst = f.inst(i);
            let ops = inst.operands();
            if rw.contains_key(&i) {
                // A rewritten instruction is handled conservatively: it may
                // read and write what it touched before.
                let first = ops.first().map(|&p| mem.loc(p));
                if let Some(l) = first {
                    cov.read(mem, l, None);
                }
                if let Some(&s) = ops.get(1)
                    && matches!(inst.kind, InstKind::MemCopy { .. })
                {
                    cov.read(mem, mem.loc(s), None);
                }
                continue;
            }
            match &inst.kind {
                InstKind::Store { ty, volatile: false, .. } => {
                    let l = mem.loc(ops[0]);
                    let size = types.size_of(*ty) as i64;
                    if let Some(off) = l.off {
                        if size > 0 && cov.contains(l.base, off, off + size) {
                            rw.insert(i, Rewrite::Delete);
                            continue;
                        }
                        cov.add(l.base, off, off + size);
                    }
                }
                InstKind::MemSet { volatile: false, .. } | InstKind::MemCopy { volatile: false, .. } => {
                    let l = mem.loc(ops[0]);
                    let n = const_u64(module, f, ops[2]).and_then(|n| i64::try_from(n).ok());
                    if let (Some(off), Some(n)) = (l.off, n) {
                        if n > 0 && cov.contains(l.base, off, off.saturating_add(n)) {
                            rw.insert(i, Rewrite::Delete);
                            continue;
                        }
                        cov.add(l.base, off, off.saturating_add(n));
                    }
                    if matches!(inst.kind, InstKind::MemCopy { .. }) {
                        cov.read(mem, mem.loc(ops[1]), n.map(|n| n as u64));
                    }
                }
                InstKind::Load { ty, volatile: false, .. } => {
                    cov.read(mem, mem.loc(ops[0]), Some(types.size_of(*ty)));
                }
                k if opaque_effect(k) => {
                    // Anything the model does not track may read any
                    // escapable memory, and a volatile or atomic access its
                    // own address (a copy also its source).
                    cov.read_escapable(mem);
                    if (k.is_volatile() || k.is_atomic())
                        && let Some(&p) = ops.first()
                    {
                        cov.read(mem, mem.loc(p), None);
                    }
                    if let (InstKind::MemCopy { .. }, Some(&s)) = (k, ops.get(1)) {
                        cov.read(mem, mem.loc(s), None);
                    }
                }
                _ => {}
            }
        }
    }
}

/// The (non-volatile) writes of non-escaping slot `s` when nothing ever
/// reads it, or `None`.
fn write_only(f: &Function, s: InstId) -> Option<Vec<InstId>> {
    let mut writes = Vec::new();
    let mut work = vec![f.inst(s).result()?];
    while let Some(p) = work.pop() {
        for u in f.uses_of(p) {
            let data = f.inst(u.inst);
            match &data.kind {
                InstKind::PtrAdd { .. } => work.push(data.result()?),
                InstKind::Store { volatile: false, .. } | InstKind::MemSet { volatile: false, .. } => writes.push(u.inst),
                InstKind::MemCopy { volatile: false, .. } if u.operand == 0 => writes.push(u.inst),
                _ => return None,
            }
        }
    }
    Some(writes)
}

/// Rebuild `old` applying `rw`.
fn apply_rewrites(old: &Function, b: &mut FunctionBuilder<'_>, rw: &HashMap<InstId, Rewrite>) {
    rebuild_with(old, b, |b, vmap, i, _| {
        let Some(r) = rw.get(&i) else { return false };
        let inst = old.inst(i);
        let ops = inst.operands();
        match r {
            Rewrite::Delete => {}
            Rewrite::LoadVal(v) => {
                let nv = remap_value(vmap, old, b, *v);
                vmap[inst.result().expect("a load result").index()] = Some(nv);
            }
            Rewrite::LoadConst(ty, x) => {
                let c = b.const_int(*ty, x.clone());
                vmap[inst.result().expect("a load result").index()] = Some(c);
            }
            Rewrite::LoadFrom { ptr, delta, align } => {
                let InstKind::Load { ty, secret, .. } = inst.kind else { unreachable!() };
                let p = remap_value(vmap, old, b, *ptr);
                let p = offset(b, p, *delta);
                let kind = InstKind::Load { ty, align: *align, volatile: false, secret };
                let nr = b.append_inst(kind, vec![p], Flags::NONE, Some(ty));
                vmap[inst.result().expect("a load result").index()] = nr;
            }
            Rewrite::StoreVal { v, ty } => {
                let InstKind::MemCopy { align, .. } = inst.kind else { unreachable!() };
                let d = remap_value(vmap, old, b, ops[0]);
                let nv = remap_value(vmap, old, b, *v);
                b.store(*ty, d, nv, align);
            }
            Rewrite::SetFrom { byte } => {
                let InstKind::MemCopy { align, .. } = inst.kind else { unreachable!() };
                let d = remap_value(vmap, old, b, ops[0]);
                let nb = remap_value(vmap, old, b, *byte);
                let n = remap_value(vmap, old, b, ops[2]);
                b.memset(d, nb, n, align);
            }
            Rewrite::CopyFrom { ptr, delta, align, disjoint } => {
                let InstKind::MemCopy { align: own, .. } = inst.kind else { unreachable!() };
                let d = remap_value(vmap, old, b, ops[0]);
                let p = remap_value(vmap, old, b, *ptr);
                let p = offset(b, p, *delta);
                let n = remap_value(vmap, old, b, ops[2]);
                // One alignment covers both pointers.
                let a = own.min(*align);
                if *disjoint {
                    b.memcpy(d, p, n, a);
                } else {
                    b.memmove(d, p, n, a);
                }
            }
        }
        true
    });
}

/// Rebuild `old`; `hook(b, vmap, inst, first_in_entry)` may emit an old
/// instruction's replacement itself (returning `true`) — it is called with
/// `first_in_entry` set once, before the entry block's first instruction.
fn rebuild_with(
    old: &Function,
    b: &mut FunctionBuilder<'_>,
    mut hook: impl FnMut(&mut FunctionBuilder<'_>, &mut Vec<Option<ValueId>>, InstId, bool) -> bool,
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
        let mut first = bi == entry;
        for &i in old.block(bb).insts() {
            b.set_line_from(old, i);
            if hook(b, &mut vmap, i, std::mem::take(&mut first)) {
                continue;
            }
            let inst = old.inst(i);
            let ops: Vec<ValueId> = inst.operands().iter().map(|&o| remap_value(&mut vmap, old, b, o)).collect();
            let result_ty = inst.result().map(|_| inst.ty);
            let nr = b.append_inst(inst.kind.clone(), ops, inst.flags, result_ty);
            if let Some(r) = inst.result() {
                vmap[r.index()] = nr;
            }
        }
        rebuild_terminator(&mut vmap, old, b, &new_block, bb, |_, _, _| {});
    }
}

#[cfg(test)]
mod tests;
