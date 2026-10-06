//! **SROA** — scalar replacement of aggregates: split an aggregate `alloca`
//! accessed only at constant offsets into one `alloca` per accessed field, so
//! [`Mem2Reg`](super::Mem2Reg) can promote each field to an SSA value.
//!
//! An aggregate value is the address of its storage (`docs/ir-design.md` §6),
//! so a front end builds a struct in an `alloca` and reads its fields back
//! with `load`s through `ptr_add`s; mem2reg alone cannot promote such a slot
//! (its address is a `ptr_add` base). This pass splits it when every use of
//! the `alloca` is
//!
//! - a non-volatile `load` or `store` *through* the address (never the stored
//!   value), directly or behind a chain of `ptr_add`s by constant offsets
//!   and `bitcast`s to a pointer or aggregate type, of a non-aggregate type
//!   that lies inside the allocation, or
//! - a `ret` of the address itself, in a function returning the `alloca`'s
//!   type (a by-value struct return);
//!
//! and the accessed byte ranges are pairwise identical or disjoint, and every
//! load reads a range a store wrote on every path to it (a `ret` copies the
//! slot, it does not read it: unwritten bytes stay undefined in the copy). A
//! slot read before it is written is left in memory: the read is poison
//! either way, but a front end may count on it being a stable unknown value
//! (C's indeterminate value), which the optimizer cannot fold away while it
//! stays in memory. Each
//! distinct range becomes an `alloca` of its accessed type, placed where the
//! original was. Accesses of one range by an integer and a float of the same
//! width share the range's first type through a `bitcast`; any other mix of
//! types still splits, with the slot left for memory accesses (mem2reg then
//! skips it).
//!
//! A `ret` of the aggregate becomes a store of every field into a fresh
//! `alloca` of the return type right before the `ret`, which returns that
//! instead (a slot already in that form is left alone). The fields themselves then promote, and the return slot is the
//! canonical construction a backend recognizes: an `alloca` written only by
//! stores in the block of its single `ret`, which the x86-64 System V,
//! AArch64 and RISC-V backends return straight from registers (`rax:rdx`,
//! `x0:x1`, `a0:a1`) without a stack slot. Bytes no access touches were
//! never written, so they are as undefined in the copy as in the original.
//!
//! Alignment: a split access asks for at most its type's natural alignment
//! (the new slot's), never more.
//!
//! A slot a bulk-memory op (`memcpy`/`memset`) touches is not this pass's:
//! [`memopt`](super::memopt) splits those (and runs first in the
//! pipeline), so the two cover disjoint slots — memopt the ones bulk ops
//! copy or fill, this pass the ones accessed field by field, returned, or
//! viewed through an address `bitcast` (an inlined struct return).

use std::collections::HashMap;

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{CastOp, InstId, InstKind};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ValueDef, ValueId};
use crate::ir::{BlockId, Function};
use crate::pass::Changed;
use crate::transform::{FunctionTransform, dom_preorder, rebuild_terminator, remap_value};

/// The SROA transform (see the module documentation).
#[derive(Debug, Default, Clone, Copy)]
pub struct Sroa;

impl FunctionTransform for Sroa {
    fn name(&self) -> &str {
        "sroa"
    }

    fn run(&mut self, old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
        split(old, builder)
    }
}

/// One accessed byte range of a split aggregate, and its slot's type.
#[derive(Clone, Copy, Debug)]
struct Range {
    off: u64,
    size: u64,
    ty: TypeId,
}

/// A splittable `alloca`: its ranges, in offset order.
#[derive(Debug)]
struct Candidate {
    ranges: Vec<Range>,
}

/// What a use of a split aggregate becomes.
#[derive(Clone, Copy, Debug)]
enum Rewrite {
    /// A load or store of range `range` of candidate `cand`.
    Access { cand: usize, range: usize },
    /// A `ptr_add` into a candidate: dropped (every use is rewritten).
    Drop,
}

/// Whether `ty` is an aggregate (an address in the IR).
fn is_aggregate(types: &TypeContext, ty: TypeId) -> bool {
    matches!(types.get(ty), Type::Struct(_) | Type::Array(..))
}

/// Whether a value of type `ty` is an address: a pointer or an aggregate.
fn is_address(types: &TypeContext, ty: TypeId) -> bool {
    types.get(ty).is_ptr() || is_aggregate(types, ty)
}

/// The constant integer value of `v`, if it is one (sign-extended).
fn const_offset(old: &Function, builder: &FunctionBuilder<'_>, v: ValueId) -> Option<i64> {
    let ValueDef::Const(c) = old.value(v).def else { return None };
    match builder.consts().get(c) {
        Const::Int { value, .. } => value.to_i64().or_else(|| value.to_u64().map(|u| u as i64)),
        _ => None,
    }
}

/// Analyze the `alloca` defining `av` (of type `elem_ty`): its accesses as
/// `(inst, offset, type)`, its `ptr_add`s, and its `ret`s, or `None` when
/// some use cannot be split.
#[allow(clippy::type_complexity)]
fn analyze(
    old: &Function,
    builder: &FunctionBuilder<'_>,
    av: ValueId,
    elem_ty: TypeId,
    ret_ty: Option<TypeId>,
) -> Option<(Vec<(InstId, u64, TypeId)>, Vec<InstId>, Vec<InstId>)> {
    let types = builder.types();
    let size = types.size_of(elem_ty);
    let mut accesses = Vec::new();
    let mut adds = Vec::new();
    let mut rets = Vec::new();
    let mut work: Vec<(ValueId, i64)> = vec![(av, 0)];
    while let Some((v, off)) = work.pop() {
        for u in old.uses_of(v) {
            let inst = old.inst(u.inst);
            let fits = |ty: TypeId| {
                !is_aggregate(types, ty)
                    && off >= 0
                    && (off as u64).checked_add(types.size_of(ty)).is_some_and(|e| e <= size)
                    && types.size_of(ty) > 0
            };
            match inst.kind {
                InstKind::Load { ty, volatile: false, .. } if u.operand == 0 && fits(ty) => {
                    accesses.push((u.inst, off as u64, ty));
                }
                InstKind::Store { ty, volatile: false, .. } if u.operand == 0 && fits(ty) => {
                    accesses.push((u.inst, off as u64, ty));
                }
                InstKind::PtrAdd { .. } if u.operand == 0 => {
                    let c = const_offset(old, builder, inst.operands()[1])?;
                    let r = inst.result().expect("ptr_add defines a value");
                    adds.push(u.inst);
                    work.push((r, off.checked_add(c)?));
                }
                // The address viewed as an aggregate (the inliner's view of
                // a returned struct): the same storage.
                InstKind::Cast(CastOp::Bitcast) if is_address(types, inst.ty) => {
                    adds.push(u.inst);
                    work.push((inst.result().expect("bitcast defines a value"), off));
                }
                InstKind::Ret if off == 0 && ret_ty == Some(elem_ty) => rets.push(u.inst),
                _ => return None,
            }
        }
    }
    Some((accesses, adds, rets))
}

/// The ranges of one candidate's accesses, or `None` when two overlap
/// without being identical. Accesses of one range by different types keep
/// the first type seen (in instruction order).
fn ranges(types: &TypeContext, accesses: &mut [(InstId, u64, TypeId)]) -> Option<Vec<Range>> {
    accesses.sort_by_key(|&(i, off, ty)| (off, types.size_of(ty), i.index()));
    let mut out: Vec<Range> = Vec::new();
    let mut first: Vec<usize> = Vec::new();
    for &(i, off, ty) in accesses.iter() {
        let size = types.size_of(ty);
        match out.last() {
            Some(r) if r.off == off && r.size == size => {
                let k = out.len() - 1;
                if i.index() < first[k] {
                    first[k] = i.index();
                    out[k].ty = ty;
                }
            }
            Some(r) if off < r.off + r.size => return None,
            _ => {
                out.push(Range { off, size, ty });
                first.push(i.index());
            }
        }
    }
    Some(out)
}

/// Whether every load of a candidate reads a range some store wrote on every
/// path to it (a forward must-analysis over the reachable blocks): only then
/// does the split create no value the memory did not hold. A load of a slot
/// before it is written reads poison either way, but a front end may rely on
/// such a read being a stable, merely unknown value (C's indeterminate
/// value), as long as it stays in memory: promoted, it would be a `poison`
/// the optimizer may fold away along with the code that tests it.
#[allow(clippy::too_many_arguments)]
fn written_before_read(
    old: &Function,
    cfg: &ControlFlowGraph,
    doms: &Dominators,
    block_of: &HashMap<InstId, usize>,
    pos: &HashMap<InstId, usize>,
    accesses: &[(InstId, u64, TypeId)],
    rs: &[Range],
    types: &TypeContext,
) -> bool {
    let n = old.block_count();
    let nr = rs.len();
    // Per block, its accesses in order: (position, is store, range).
    let mut by_block: Vec<Vec<(usize, bool, usize)>> = vec![Vec::new(); n];
    for &(i, off, ty) in accesses {
        let size = types.size_of(ty);
        let r = rs.iter().position(|r| r.off == off && r.size == size).expect("range of an access");
        let store = matches!(old.inst(i).kind, InstKind::Store { .. });
        by_block[block_of[&i]].push((pos[&i], store, r));
    }
    for v in &mut by_block {
        v.sort_unstable();
    }
    let entry = old.entry().map_or(0, |e| e.index());
    let reachable: Vec<usize> = (0..n).filter(|&b| doms.is_reachable(b)).collect();
    let block_in = |out: &[Vec<bool>], b: usize| -> Vec<bool> {
        if b == entry {
            return vec![false; nr];
        }
        let mut acc = vec![true; nr];
        for &p in cfg.predecessors(b) {
            if doms.is_reachable(p) {
                for (a, o) in acc.iter_mut().zip(&out[p]) {
                    *a &= *o;
                }
            }
        }
        acc
    };
    let mut out: Vec<Vec<bool>> = vec![vec![true; nr]; n];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &reachable {
            let mut cur = block_in(&out, b);
            for &(_, store, r) in &by_block[b] {
                cur[r] |= store;
            }
            if cur != out[b] {
                out[b] = cur;
                changed = true;
            }
        }
    }
    reachable.iter().all(|&b| {
        let mut cur = block_in(&out, b);
        by_block[b].iter().all(|&(_, store, r)| {
            let ok = store || cur[r];
            cur[r] |= store;
            ok
        })
    })
}

/// How a value of type `from` is reinterpreted as `to` (same size), if the
/// pass converts between them: integer ↔ float by `bitcast`.
fn conversion(types: &TypeContext, from: TypeId, to: TypeId) -> Option<CastOp> {
    let (a, b) = (types.get(from), types.get(to));
    let scalar = |t: &Type| matches!(t, Type::Int(_) | Type::Float(_));
    (scalar(a) && scalar(b) && a.is_float() != b.is_float()).then_some(CastOp::Bitcast)
}

fn split(old: &Function, builder: &mut FunctionBuilder<'_>) -> Changed {
    let Some(entry) = old.entry() else {
        return Changed::No;
    };
    let ret_ty = match builder.types().get(old.sig) {
        Type::Func(ft) => Some(ft.ret),
        _ => None,
    };

    // Each instruction's block and position in it.
    let mut block_of: HashMap<InstId, usize> = HashMap::new();
    let mut pos: HashMap<InstId, usize> = HashMap::new();
    for (b, blk) in old.blocks() {
        for (k, &i) in blk.insts().iter().chain(blk.terminator().as_ref()).enumerate() {
            block_of.insert(i, b.index());
            pos.insert(i, k);
        }
    }
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);

    // Discover the candidates.
    let mut cands: Vec<Candidate> = Vec::new();
    let mut alloca_cand: HashMap<usize, usize> = HashMap::new();
    let mut rewrite: HashMap<usize, Rewrite> = HashMap::new();
    let mut ret_cand: HashMap<usize, usize> = HashMap::new();
    for (_bid, blk) in old.blocks() {
        for &i in blk.insts() {
            let inst = old.inst(i);
            let InstKind::Alloca { elem_ty } = inst.kind else { continue };
            if !is_aggregate(builder.types(), elem_ty) {
                continue;
            }
            let av = inst.result().expect("alloca defines a value");
            let Some((mut accesses, adds, rets)) = analyze(old, builder, av, elem_ty, ret_ty) else {
                continue;
            };
            if accesses.is_empty() {
                continue;
            }
            // Already a return slot (written only right before its one
            // `ret`): what this pass makes, and what a backend returns from
            // registers. Splitting it again would only rebuild it.
            let canonical = matches!(rets[..], [r] if accesses.iter().all(|&(ai, ..)| {
                matches!(old.inst(ai).kind, InstKind::Store { .. }) && block_of.get(&ai) == block_of.get(&r)
            }));
            if canonical {
                continue;
            }
            let Some(rs) = ranges(builder.types(), &mut accesses) else { continue };
            if !written_before_read(old, &cfg, &doms, &block_of, &pos, &accesses, &rs, builder.types()) {
                continue;
            }
            let c = cands.len();
            for &(ai, off, ty) in &accesses {
                let size = builder.types().size_of(ty);
                let range = rs.iter().position(|r| r.off == off && r.size == size).expect("range of an access");
                rewrite.insert(ai.index(), Rewrite::Access { cand: c, range });
            }
            for a in adds {
                rewrite.insert(a.index(), Rewrite::Drop);
            }
            for r in rets {
                ret_cand.insert(r.index(), c);
            }
            alloca_cand.insert(i.index(), c);
            cands.push(Candidate { ranges: rs });
        }
    }
    if cands.is_empty() {
        return Changed::No;
    }

    // Rebuild with identical blocks and edges.
    let n = old.block_count();
    let entry_idx = entry.index();
    let mut new_block: Vec<Option<BlockId>> = vec![None; n];
    new_block[entry_idx] = Some(builder.create_entry_block());
    for (b, slot) in new_block.iter_mut().enumerate() {
        if b == entry_idx {
            continue;
        }
        let bb = BlockId::from_index(b);
        let ptys: Vec<TypeId> = old.block(bb).params().iter().map(|&p| old.value_type(p)).collect();
        *slot = Some(builder.create_block(&ptys));
    }
    let new_block: Vec<BlockId> = new_block.into_iter().map(|x| x.expect("every block was created")).collect();
    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for (b, &nb) in new_block.iter().enumerate() {
        let bb = BlockId::from_index(b);
        let new_params = builder.block_params(nb).to_vec();
        for (&op, &np) in old.block(bb).params().iter().zip(new_params.iter()) {
            vmap[op.index()] = Some(np);
        }
    }

    // `slots[c][r]`: the new `alloca` of range `r` of candidate `c`;
    // `ret_slot[ret inst]`: the return slot made for that `ret`.
    let mut slots: Vec<Vec<ValueId>> = vec![Vec::new(); cands.len()];
    let mut ret_slot: HashMap<usize, ValueId> = HashMap::new();
    let i64t = builder.types_mut().int(64);

    for b in dom_preorder(old, &doms) {
        let bb = BlockId::from_index(b);
        builder.switch_to(new_block[b]);
        let insts = old.block(bb).insts().to_vec();
        for i in insts {
            let inst = old.inst(i);
            if let Some(&c) = alloca_cand.get(&i.index()) {
                slots[c] = cands[c].ranges.iter().map(|r| builder.alloca(r.ty)).collect();
                let InstKind::Alloca { elem_ty } = inst.kind else { unreachable!() };
                // One return slot per `ret` of this aggregate.
                let mut rets: Vec<usize> = ret_cand.iter().filter(|&(_, &rc)| rc == c).map(|(&r, _)| r).collect();
                rets.sort_unstable();
                for r in rets {
                    let s = builder.alloca(elem_ty);
                    ret_slot.insert(r, s);
                }
                continue;
            }
            match rewrite.get(&i.index()) {
                Some(Rewrite::Drop) => continue,
                Some(&Rewrite::Access { cand, range }) => {
                    let slot = slots[cand][range];
                    let rty = cands[cand].ranges[range].ty;
                    let natural = builder.types().align_of(rty).max(1) as u32;
                    match inst.kind {
                        InstKind::Load { ty, align, volatile, secret } => {
                            // Read the slot's type and convert, or reinterpret the
                            // memory as the accessed type.
                            let cast = if ty == rty { None } else { conversion(builder.types(), rty, ty) };
                            let lty = if cast.is_some() { rty } else { ty };
                            let kind = InstKind::Load { ty: lty, align: align.min(natural), volatile, secret };
                            let v = builder.append_inst(kind, vec![slot], inst.flags, Some(lty)).expect("load result");
                            let v = match cast {
                                Some(op) => builder.cast(op, v, ty),
                                None => v,
                            };
                            vmap[inst.result().expect("load result").index()] = Some(v);
                        }
                        InstKind::Store { ty, align, volatile, secret } => {
                            let mut val = remap_value(&mut vmap, old, builder, inst.operands()[1]);
                            let cast = if ty == rty { None } else { conversion(builder.types(), ty, rty) };
                            let sty = match cast {
                                Some(op) => {
                                    val = builder.cast(op, val, rty);
                                    rty
                                }
                                None => ty,
                            };
                            let kind = InstKind::Store { ty: sty, align: align.min(natural), volatile, secret };
                            builder.append_inst(kind, vec![slot, val], inst.flags, None);
                        }
                        _ => unreachable!("only loads and stores are rewritten"),
                    }
                    continue;
                }
                None => {}
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
        // A `ret` of a split aggregate: copy the fields into its return slot.
        if let Some(t) = old.block(bb).terminator()
            && let Some(&c) = ret_cand.get(&t.index())
        {
            let dst = ret_slot[&t.index()];
            for (k, r) in cands[c].ranges.iter().enumerate() {
                let natural = builder.types().align_of(r.ty).max(1) as u32;
                let v = builder.load(r.ty, slots[c][k], natural);
                let p = if r.off == 0 {
                    dst
                } else {
                    let off = builder.const_i64(i64t, r.off as i64);
                    builder.ptr_add(dst, off, true)
                };
                builder.store(r.ty, p, v, natural);
            }
            builder.ret(Some(dst));
            continue;
        }
        rebuild_terminator(&mut vmap, old, builder, &new_block, bb, |_, _, _| {});
    }
    Changed::Yes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::text;
    use crate::pass::ModulePass;
    use crate::support::StrInterner;
    use crate::support::diagnostics::FileId;
    use crate::transform::{FunctionTransformPass, Mem2Reg};
    use crate::verify::verify_module;

    fn parse(src: &str, syms: &mut StrInterner) -> crate::ir::Module {
        text::parse_module(src, FileId::new(0), syms).expect("parses")
    }

    fn run(src: &str) -> String {
        let mut syms = StrInterner::new();
        let mut m = parse(src, &mut syms);
        verify_module(&m).expect("input verifies");
        FunctionTransformPass::new(Sroa).run(&mut m);
        verify_module(&m).expect("sroa output verifies");
        FunctionTransformPass::new(Mem2Reg).run(&mut m);
        verify_module(&m).expect("mem2reg output verifies");
        text::print_module(&m, &syms)
    }

    /// A `{i64, i64}` built in branches and returned: the fields promote, and
    /// the return slot is written only right before the `ret`.
    #[test]
    fn struct_return_fields_promote() {
        let out = run(r#"module "t"
func @parse(i64) -> {i64, i64} {
entry ^0(%0: i64):
  %1 = alloca {i64, i64} : ptr
  %2 = icmp slt %0, i64 0 : i1
  cond_br %2, ^1, ^2
^1:
  store i64 1, %1 align 8 : i64
  %3 = ptr_add inbounds %1, i64 8 : ptr
  store i64 0, %3 align 8 : i64
  br ^3
^2:
  store i64 0, %1 align 8 : i64
  %4 = ptr_add inbounds %1, i64 8 : ptr
  %5 = mul %0, i64 3 : i64
  store %5, %4 align 8 : i64
  br ^3
^3:
  ret %1
}
"#);
        // Only the return slot remains in memory; the join takes both fields
        // as block parameters.
        assert_eq!(out.matches("alloca").count(), 1, "{out}");
        assert!(out.contains("^3(%"), "{out}");
        assert!(!out.contains("load"), "{out}");
    }

    /// A struct copied out of a call result and branched on: the local copy
    /// disappears, leaving only loads of the call result.
    #[test]
    fn copied_call_result_promotes() {
        let out = run(r#"module "t"
func @parse(i64) -> {i64, i64}

func @use(i64) -> i64 {
entry ^0(%0: i64):
  %1 = alloca {i64, i64} : ptr
  %2 = call @parse(%0) : {i64, i64}
  %3 = load %2 align 1 : i64
  store %3, %1 align 1 : i64
  %4 = ptr_add inbounds %2, i64 8 : ptr
  %5 = ptr_add inbounds %1, i64 8 : ptr
  %6 = load %4 align 1 : i64
  store %6, %5 align 1 : i64
  %7 = load %1 align 8 : i64
  %8 = icmp ne %7, i64 0 : i1
  cond_br %8, ^1, ^2
^1:
  ret i64 -1
^2:
  %9 = ptr_add inbounds %1, i64 8 : ptr
  %10 = load %9 align 8 : i64
  ret %10
}
"#);
        assert!(!out.contains("alloca"), "{out}");
        assert!(!out.contains("store"), "{out}");
    }

    /// A slot read before anything writes it (C's indeterminate value, here
    /// `counts[0]` tested before it is set) stays in memory, so the test is
    /// not folded away as a branch on poison; one written on every path
    /// before its read is split.
    #[test]
    fn read_before_write_is_kept() {
        let src = r#"module "t"
func @f(i32) -> i32 {
entry ^0(%x: i32):
  %c = alloca [3 x i32] : ptr
  %v = load %c align 4 : i32
  %t = icmp sge %v, i32 0 : i1
  store %x, %c align 4 : i32
  %r = select %t, i32 1, i32 2 : i32
  ret %r
}

func @g(i1, i32) -> i32 {
entry ^0(%k: i1, %x: i32):
  %c = alloca [3 x i32] : ptr
  cond_br %k, ^1, ^2
^1:
  store %x, %c align 4 : i32
  br ^3
^2:
  store i32 5, %c align 4 : i32
  br ^3
^3:
  %v = load %c align 4 : i32
  ret %v
}
"#;
        let out = run(src);
        let (f, g) = out.split_at(out.find("func @g").expect("g"));
        assert!(f.contains("alloca"), "{out}");
        assert!(!g.contains("alloca"), "{out}");
    }

    /// The return slot the split leaves is the canonical form: a second run
    /// changes nothing.
    #[test]
    fn return_slot_is_left_alone() {
        let src = r#"module "t"
func @pair(i64, i64) -> {i64, i64} {
entry ^0(%a: i64, %b: i64):
  %r = alloca {i64, i64} : ptr
  store %a, %r align 8 : i64
  %p = ptr_add inbounds %r, i64 8 : ptr
  store %b, %p align 8 : i64
  ret %r
}
"#;
        let mut syms = StrInterner::new();
        let mut m = parse(src, &mut syms);
        assert_eq!(FunctionTransformPass::new(Sroa).run(&mut m), Changed::No);
    }

    /// An escaping aggregate (its address passed to a call) is left alone, as
    /// is one accessed through overlapping ranges.
    #[test]
    fn escaping_or_overlapping_is_kept() {
        let src = r#"module "t"
func @sink(ptr) -> void

func @f(i64) -> i64 {
entry ^0(%0: i64):
  %1 = alloca {i64, i64} : ptr
  store %0, %1 align 8 : i64
  call @sink(%1) : void
  %2 = alloca {i64, i64} : ptr
  store %0, %2 align 8 : i64
  %3 = ptr_add inbounds %2, i64 4 : ptr
  %4 = load %3 align 4 : i64
  ret %4
}
"#;
        let mut syms = StrInterner::new();
        let mut m = parse(src, &mut syms);
        assert_eq!(FunctionTransformPass::new(Sroa).run(&mut m), Changed::No);
    }

    /// An integer and a float view of one field share a slot through a
    /// `bitcast`, so the field still promotes.
    #[test]
    fn int_float_views_share_a_slot() {
        let out = run(r#"module "t"
func @f(f64) -> i64 {
entry ^0(%0: f64):
  %1 = alloca {f64, i64} : ptr
  store %0, %1 align 8 : f64
  %2 = load %1 align 8 : i64
  ret %2
}
"#);
        assert!(!out.contains("alloca"), "{out}");
        assert!(out.contains("bitcast"), "{out}");
    }

    /// A struct-returning callee inlined at `-O2`: the inliner hands the
    /// caller the callee's return slot (viewed through a `bitcast`), which
    /// SROA then splits, so no `alloca` is left — and the result agrees with
    /// the reference evaluator before and after.
    #[test]
    fn inlined_struct_return_promotes_at_o2() {
        use crate::ir::refexec::run_named;
        use crate::ir::semantics::SemValue;
        use crate::transform::pipeline::{OptLevel, optimize};
        let src = r#"module "t"
func @parse(i64) -> {i64, i64} {
entry ^0(%0: i64):
  %1 = alloca {i64, i64} : ptr
  %2 = icmp slt %0, i64 0 : i1
  cond_br %2, ^1, ^2
^1:
  store i64 1, %1 align 8 : i64
  %3 = ptr_add inbounds %1, i64 8 : ptr
  store i64 0, %3 align 8 : i64
  br ^3
^2:
  store i64 0, %1 align 8 : i64
  %4 = ptr_add inbounds %1, i64 8 : ptr
  %5 = mul %0, i64 3 : i64
  store %5, %4 align 8 : i64
  br ^3
^3:
  ret %1
}

func @use(i64) -> i64 {
entry ^0(%0: i64):
  %1 = alloca {i64, i64} : ptr
  %2 = call @parse(%0) : {i64, i64}
  %3 = load %2 align 1 : i64
  store %3, %1 align 1 : i64
  %4 = ptr_add inbounds %2, i64 8 : ptr
  %5 = ptr_add inbounds %1, i64 8 : ptr
  %6 = load %4 align 1 : i64
  store %6, %5 align 1 : i64
  %7 = load %1 align 8 : i64
  %8 = icmp ne %7, i64 0 : i1
  cond_br %8, ^1, ^2
^1:
  ret i64 -1
^2:
  %9 = ptr_add inbounds %1, i64 8 : ptr
  %10 = load %9 align 8 : i64
  %11 = add %10, i64 1 : i64
  ret %11
}
"#;
        let mut syms = StrInterner::new();
        let before = parse(src, &mut syms);
        let mut after = before.clone();
        optimize(&mut after, OptLevel::O2);
        verify_module(&after).expect("verifies");
        let out = text::print_module(&after, &syms);
        let body = &out[out.find("func @use").expect("use survives")..];
        assert!(!body.contains("alloca") && !body.contains("call"), "{out}");
        for x in [-5i64, 0, 7] {
            let arg = [SemValue::int(64, puremp::Int::from_i64(x))];
            let want = run_named(&before, &syms, "use", &arg).expect("runs");
            let got = run_named(&after, &syms, "use", &arg).expect("runs");
            assert_eq!(got, want, "use({x})");
        }
    }
}
