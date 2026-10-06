//! Register-returned aggregates without a stack slot (`docs/ir-design.md` §6,
//! "Aggregate returns in registers").
//!
//! An aggregate value is the address of its storage, so a by-value struct
//! return is, in the IR, an address on both sides of the call: the callee
//! builds the struct in an `alloca` and returns that address, and the caller
//! receives an address it loads the fields from. When the ABI returns the
//! struct in registers (`rax:rdx`, `x0:x1`, `a0:a1`, their floating-point
//! counterparts), the memory is only a detour, and two shapes let a backend
//! skip it entirely:
//!
//! - **Callee** — `ret %s`, where `%s` is an `alloca` used by nothing but this
//!   `ret` and by non-volatile `store`s *in the `ret`'s block* (directly or
//!   through `ptr_add`s by constant offsets), each storing exactly one
//!   register part, at most one store per part. The stored values go straight
//!   into the return registers; the `alloca`, its `ptr_add`s and the stores
//!   emit nothing. (A part no store covers was never written: its register
//!   is left as it is, which the undefined bytes allow.) This is the shape
//!   [`crate::transform::Sroa`] leaves after promoting the fields.
//! - **Caller** — `%r = call ...` returning the aggregate, where every use of
//!   `%r` is a non-volatile `load` (directly or through constant `ptr_add`s)
//!   of a field lying inside one register part: an integer or pointer in an
//!   integer part (any byte offset in it), or a float at the start of a
//!   floating-point part. Nothing writes the result, so each load is the
//!   returned register's bits: a move, or a logical shift right for a field
//!   above the part's low byte (the backends keep a narrow integer in the low
//!   bits of a register, its upper bits unspecified). No result slot is
//!   allocated.
//!
//! [`analyze`] finds both shapes for a function, given the target's register
//! parts ([`RetPart`]) for a returned aggregate type;
//! [`Lower`](crate::codegen::isel::Lower) applies the plan around the
//! target's rules (see [`TargetIsel::ret_parts`](crate::codegen::isel::TargetIsel::ret_parts)).
//! Anything else keeps the slot, which is always correct.

use std::collections::HashMap;

use crate::ir::inst::{InstId, InstKind};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ValueDef, ValueId};
use crate::ir::{ConstPool, Function};

/// One register of a register-returned aggregate: the bytes
/// `off .. off + size` of the struct, in a floating-point register (`fp`) or
/// an integer one. A target lists the parts in its return-register order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RetPart {
    /// The byte offset of the part in the aggregate.
    pub off: u64,
    /// The number of bytes the register carries.
    pub size: u64,
    /// Whether the part travels in a floating-point register.
    pub fp: bool,
}

/// A load of a field of a fused call result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PartLoad {
    /// The call's result value.
    pub call: ValueId,
    /// The register part the field lies in.
    pub part: usize,
    /// The field's bit offset in the part (a multiple of 8; 0 for a float).
    pub shift: u32,
}

/// Where a function's register-returned aggregates skip memory.
#[derive(Debug, Default)]
pub struct AggPlan {
    /// Call results returned in registers that need no slot, with their parts.
    pub calls: HashMap<ValueId, Vec<RetPart>>,
    /// The loads reading those results, by instruction.
    pub loads: HashMap<InstId, PartLoad>,
    /// Return slots: per `alloca` value, the value stored into each part of
    /// the function's return registers (`None`: never written).
    pub rets: HashMap<ValueId, Vec<Option<ValueId>>>,
    /// Instructions that emit nothing: the return slots' `alloca`s and
    /// stores, and the `ptr_add`s into a fused aggregate.
    pub skip: Vec<InstId>,
}

impl AggPlan {
    /// Whether the plan changes nothing.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty() && self.rets.is_empty()
    }
}

fn is_aggregate(types: &TypeContext, ty: TypeId) -> bool {
    matches!(types.get(ty), Type::Struct(_) | Type::Array(..))
}

/// The constant value of `v` as a signed offset, if it is an integer constant.
fn const_offset(f: &Function, consts: &ConstPool, v: ValueId) -> Option<i64> {
    let ValueDef::Const(c) = f.value(v).def else { return None };
    match consts.get(c) {
        Const::Int { value, .. } => value.to_i64().or_else(|| value.to_u64().map(|u| u as i64)),
        _ => None,
    }
}

/// The memory uses reachable from `root` through `ptr_add`s by constant
/// offsets: `(instruction, byte offset, is store)`, plus the `ptr_add`s, or
/// `None` when some use is anything else (the address escapes, a volatile
/// access, a store *of* the address, ...). A `ret` of `root` itself is
/// accepted when `ret_ok`, and reported in the last element.
#[allow(clippy::type_complexity)]
fn memory_uses(
    f: &Function,
    consts: &ConstPool,
    root: ValueId,
    ret_ok: bool,
) -> Option<(Vec<(InstId, i64, bool)>, Vec<InstId>, Vec<InstId>)> {
    let mut accesses = Vec::new();
    let mut adds = Vec::new();
    let mut rets = Vec::new();
    let mut work = vec![(root, 0i64)];
    while let Some((v, off)) = work.pop() {
        for u in f.uses_of(v) {
            let inst = f.inst(u.inst);
            match inst.kind {
                InstKind::Load { volatile: false, .. } if u.operand == 0 => accesses.push((u.inst, off, false)),
                InstKind::Store { volatile: false, .. } if u.operand == 0 => accesses.push((u.inst, off, true)),
                InstKind::PtrAdd { .. } if u.operand == 0 => {
                    let c = const_offset(f, consts, inst.operands()[1])?;
                    adds.push(u.inst);
                    work.push((inst.result().expect("ptr_add defines a value"), off.checked_add(c)?));
                }
                InstKind::Ret if ret_ok && v == root => rets.push(u.inst),
                _ => return None,
            }
        }
    }
    Some((accesses, adds, rets))
}

/// The part a field of type `ty` at byte `off` lies in, and its bit offset
/// there; `exact` asks for a field starting the part (a stored field).
fn part_of(types: &TypeContext, parts: &[RetPart], off: i64, ty: TypeId, exact: bool) -> Option<(usize, u32)> {
    let off = u64::try_from(off).ok()?;
    let size = types.size_of(ty);
    let t = types.get(ty);
    let int = matches!(t, Type::Int(_)) || t.is_ptr();
    if size == 0 || !(int || t.is_float()) {
        return None;
    }
    let k = parts.iter().position(|p| p.off <= off && off + size <= p.off + p.size)?;
    let p = parts[k];
    if p.fp != t.is_float() || (p.fp || exact) && off != p.off || size > 8 {
        return None;
    }
    Some((k, ((off - p.off) * 8) as u32))
}

/// Find the slot-free aggregate returns of `f` (see the module docs).
/// `parts_of` gives the target's register parts for a returned aggregate
/// type, or `None` when it is returned in memory.
pub fn analyze(
    f: &Function,
    types: &TypeContext,
    consts: &ConstPool,
    parts_of: &dyn Fn(TypeId) -> Option<Vec<RetPart>>,
) -> AggPlan {
    let mut plan = AggPlan::default();
    let mut block_of: HashMap<InstId, usize> = HashMap::new();
    for (b, blk) in f.blocks() {
        for &i in blk.insts() {
            block_of.insert(i, b.index());
        }
        if let Some(t) = blk.terminator() {
            block_of.insert(t, b.index());
        }
    }
    let ret_ty = match types.get(f.sig) {
        Type::Func(ft) => Some(ft.ret),
        _ => None,
    };
    let ret_parts = ret_ty.filter(|&t| is_aggregate(types, t)).and_then(parts_of);

    for (_b, blk) in f.blocks() {
        for &i in blk.insts() {
            let inst = f.inst(i);
            match inst.kind {
                // Caller: a call result read only by field loads.
                InstKind::Call => {
                    let Some(r) = inst.result() else { continue };
                    let ty = f.value_type(r);
                    if !is_aggregate(types, ty) {
                        continue;
                    }
                    let Some(parts) = parts_of(ty) else { continue };
                    let Some((accesses, adds, _)) = memory_uses(f, consts, r, false) else { continue };
                    let mut loads = Vec::with_capacity(accesses.len());
                    let ok = accesses.iter().all(|&(li, off, store)| {
                        let l = f.inst(li);
                        let InstKind::Load { ty, .. } = l.kind else { return false };
                        match part_of(types, &parts, off, ty, false) {
                            Some((part, shift)) if !store => {
                                loads.push((li, PartLoad { call: r, part, shift }));
                                true
                            }
                            _ => false,
                        }
                    });
                    if !ok {
                        continue;
                    }
                    plan.loads.extend(loads);
                    plan.skip.extend(adds);
                    plan.calls.insert(r, parts);
                }
                // Callee: a return slot written only right before its `ret`.
                InstKind::Alloca { .. } => {
                    let Some(parts) = ret_parts.as_ref() else { continue };
                    let s = inst.result().expect("alloca defines a value");
                    let Some((accesses, adds, rets)) = memory_uses(f, consts, s, true) else { continue };
                    let [ret] = rets[..] else { continue };
                    let ret_block = block_of.get(&ret).copied();
                    let mut vals: Vec<Option<ValueId>> = vec![None; parts.len()];
                    let mut stores = Vec::with_capacity(accesses.len());
                    let ok = accesses.iter().all(|&(si, off, store)| {
                        let st = f.inst(si);
                        let InstKind::Store { ty, .. } = st.kind else { return false };
                        if !store || block_of.get(&si).copied() != ret_block {
                            return false;
                        }
                        match part_of(types, parts, off, ty, true) {
                            Some((k, _)) if vals[k].is_none() => {
                                vals[k] = Some(st.operands()[1]);
                                stores.push(si);
                                true
                            }
                            _ => false,
                        }
                    });
                    if !ok {
                        continue;
                    }
                    plan.skip.push(i);
                    plan.skip.extend(adds);
                    plan.skip.extend(stores);
                    plan.rets.insert(s, vals);
                }
                _ => {}
            }
        }
    }
    plan
}
