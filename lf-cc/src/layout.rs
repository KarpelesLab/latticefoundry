//! Data layout for the C type system, shared by sema, the parser, and lowering.
//!
//! The C aggregate types (`struct`, `union`, arrays) reference a shared
//! [`Records`] registry of `struct`/`union` definitions (a struct or union is a
//! [`crate::ast::CType::Record`] carrying only an index into that registry, which
//! is what lets a type refer to itself through a pointer, e.g. a linked-list
//! node). This module answers the layout questions — `size_of`, `align_of`, and
//! `field_offset` — with a natural-alignment layout identical to the IR's default
//! (`latticefoundry::ir::types::TypeContext`), and maps a [`CType`] to an interned
//! IR [`TypeId`] for lowering. Keeping one implementation here guarantees sema's
//! `sizeof`/offsets agree with the storage the backend reserves.

use latticefoundry::ir::types::{FloatKind, TypeContext, TypeId};

use crate::ast::{CType, Field, FloatTy, RecordId, RecordKind, Records};

/// Round `value` up to the next multiple of `align` (a power of two ≥ 1).
fn round_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align.max(1)) * align.max(1)
}

/// The effective alignment of a record member: its natural alignment raised to
/// any explicit `_Alignas`/`alignas` request (alignment may only increase).
fn field_align(recs: &Records, f: &Field) -> u64 {
    let natural = align_of(recs, &f.ty);
    match f.align {
        Some(a) => natural.max(a),
        None => natural,
    }
}

/// The size in bytes of a C type under the natural-alignment layout.
pub fn size_of(recs: &Records, ty: &CType) -> u64 {
    match ty {
        CType::Void => 1,
        CType::Bool => 1,
        CType::Int(i) => u64::from(i.width) / 8,
        CType::Float(f) => u64::from(f.bits()) / 8,
        CType::Pointer(_) => 8,
        CType::Array(elem, n) => stride_of(recs, elem) * *n,
        CType::Record(id) => record_size(recs, *id),
        // `sizeof` a function designator is 1 (a GCC extension); function
        // pointers are `Pointer` and take the pointer size above.
        CType::Func(_) => 1,
    }
}

/// The alignment in bytes of a C type.
pub fn align_of(recs: &Records, ty: &CType) -> u64 {
    match ty {
        CType::Void | CType::Bool => 1,
        CType::Int(i) => (u64::from(i.width) / 8).clamp(1, 8),
        CType::Float(f) => u64::from(f.bits()) / 8,
        CType::Pointer(_) => 8,
        CType::Array(elem, _) => align_of(recs, elem),
        CType::Record(id) => record_align(recs, *id),
        CType::Func(_) => 1,
    }
}

/// The byte distance between consecutive array elements of type `ty` (its size
/// rounded up to its alignment).
pub fn stride_of(recs: &Records, ty: &CType) -> u64 {
    round_up(size_of(recs, ty), align_of(recs, ty))
}

fn record_align(recs: &Records, id: RecordId) -> u64 {
    let def = recs.get(id);
    def.fields.iter().map(|f| field_align(recs, f)).max().unwrap_or(1).max(1)
}

fn record_size(recs: &Records, id: RecordId) -> u64 {
    let def = recs.get(id);
    match def.kind {
        RecordKind::Struct => {
            let mut offset = 0u64;
            let mut align = 1u64;
            for f in &def.fields {
                let a = field_align(recs, f);
                align = align.max(a);
                offset = round_up(offset, a) + size_of(recs, &f.ty);
            }
            round_up(offset, align.max(1)).max(1)
        }
        RecordKind::Union => {
            let mut size = 0u64;
            for f in &def.fields {
                size = size.max(size_of(recs, &f.ty));
            }
            round_up(size, record_align(recs, id)).max(1)
        }
    }
}

/// The byte offset of field `idx` within record `id` (`0` for every union
/// member, cumulative with padding for a struct).
pub fn field_offset(recs: &Records, id: RecordId, idx: usize) -> u64 {
    let def = recs.get(id);
    if def.kind == RecordKind::Union {
        return 0;
    }
    let mut offset = 0u64;
    for (i, f) in def.fields.iter().enumerate() {
        let a = field_align(recs, f);
        offset = round_up(offset, a);
        if i == idx {
            return offset;
        }
        offset += size_of(recs, &f.ty);
    }
    offset
}

/// Resolve a member `name` within record `id`, descending into anonymous
/// `struct`/`union` members (C11). Returns the member's byte offset (composed
/// through any enclosing anonymous members) and its type.
pub fn resolve_member(recs: &Records, id: RecordId, name: &str) -> Option<(u64, CType)> {
    let def = recs.get(id);
    // A directly-named member wins over descending into anonymous members.
    for (i, f) in def.fields.iter().enumerate() {
        if !f.anonymous && f.name == name {
            return Some((field_offset(recs, id, i), f.ty.clone()));
        }
    }
    for (i, f) in def.fields.iter().enumerate() {
        if f.anonymous
            && let CType::Record(sub) = &f.ty
            && let Some((off, ty)) = resolve_member(recs, *sub, name)
        {
            return Some((field_offset(recs, id, i) + off, ty));
        }
    }
    None
}

/// Intern the IR [`TypeId`] modelling `ty` into `cx`, given the record registry.
///
/// Scalars map to their IR counterpart; arrays to `Array`; a struct to a
/// `Struct` of its field types in order; a union to a `Struct` whose layout has
/// the union's size and alignment (an alignment-sized integer head plus byte
/// padding) — union member access never reads this shape, it only needs the
/// storage `alloca` reserves.
pub fn ir_type(cx: &mut TypeContext, recs: &Records, ty: &CType) -> TypeId {
    match ty {
        CType::Void => cx.void(),
        CType::Bool => cx.int(8),
        CType::Int(i) => cx.int(u32::from(i.width)),
        CType::Float(FloatTy::F32) => cx.float(FloatKind::F32),
        CType::Float(FloatTy::F64) => cx.float(FloatKind::F64),
        CType::Pointer(_) => cx.ptr(),
        CType::Array(elem, n) => {
            let e = ir_type(cx, recs, elem);
            cx.array(e, *n)
        }
        CType::Record(id) => ir_record(cx, recs, *id),
        // A function type only ever appears behind a pointer in lowered code.
        CType::Func(_) => cx.ptr(),
    }
}

fn ir_record(cx: &mut TypeContext, recs: &Records, id: RecordId) -> TypeId {
    let def = recs.get(id);
    match def.kind {
        RecordKind::Struct => {
            // A member with an explicit `alignas` can raise the struct's size and
            // alignment past what a natural struct of the field types would give;
            // model such a struct by a byte blob of the right size/alignment (as
            // for unions) so the storage reserved by an `alloca` stays adequate.
            if def.fields.iter().any(|f| f.align.is_some()) {
                let size = record_size(recs, id);
                let align = record_align(recs, id).min(8);
                let head = cx.int((align * 8) as u32);
                return if size > align {
                    let i8t = cx.int(8);
                    let pad = cx.array(i8t, size - align);
                    cx.struct_(vec![head, pad])
                } else {
                    cx.struct_(vec![head])
                };
            }
            let field_tys = def.fields.clone();
            let ids: Vec<TypeId> = field_tys.iter().map(|f| ir_type(cx, recs, &f.ty)).collect();
            cx.struct_(ids)
        }
        RecordKind::Union => {
            let size = record_size(recs, id);
            let align = record_align(recs, id);
            let head = cx.int((align * 8) as u32);
            if size > align {
                let i8t = cx.int(8);
                let pad = cx.array(i8t, size - align);
                cx.struct_(vec![head, pad])
            } else {
                cx.struct_(vec![head])
            }
        }
    }
}
