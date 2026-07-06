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

/// The placement of a bit-field within its storage unit, computed by the
/// little-endian System V (Itanium) bit-field allocation. A bit-field is read
/// and written through a load/read-modify-write store of an integer of
/// `unit_bits` bits located at the member's byte offset; the field occupies
/// `width` bits starting `bit_offset` bits above the storage unit's least
/// significant bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitPlacement {
    /// The storage unit's width in bits (the bit-field's declared type width;
    /// `_Bool` uses an 8-bit unit).
    pub unit_bits: u16,
    /// The bit-field's least-significant-bit offset within the storage unit.
    pub bit_offset: u32,
    /// The bit-field's width in bits.
    pub width: u32,
    /// Whether the bit-field is signed (selecting sign vs. zero extension on read).
    pub signed: bool,
}

/// The `(storage-unit bits, maximum declared width)` of a bit-field of declared
/// type `ty`. The storage unit is the natural size of the declared type in bits
/// (`_Bool` occupies an 8-bit unit but permits only a single value bit).
fn bitfield_unit_bits(ty: &CType) -> u16 {
    match ty {
        CType::Bool => 8,
        CType::Int(i) => i.width,
        _ => 32,
    }
}

/// The computed layout of one record: per-field byte offsets (for a bit-field,
/// its storage unit's byte offset) and optional bit placements, plus the record's
/// overall size and alignment.
struct RecordLayout {
    /// Byte offset of each field (the storage-unit offset for a bit-field).
    offsets: Vec<u64>,
    /// Bit placement of each field, `Some` only for bit-fields.
    bits: Vec<Option<BitPlacement>>,
    size: u64,
    align: u64,
}

/// Compute a record's layout: byte offsets, bit-field placements, size, and
/// alignment, following the little-endian System V bit-field rules gcc uses on
/// x86-64 Linux (`-mno-ms-bitfields`).
fn record_layout(recs: &Records, id: RecordId) -> RecordLayout {
    let def = recs.get(id);
    let n = def.fields.len();
    let mut offsets = vec![0u64; n];
    let mut bits: Vec<Option<BitPlacement>> = vec![None; n];
    let mut align = 1u64;

    match def.kind {
        RecordKind::Struct => {
            // The running offset is tracked in bits so bit-fields pack sub-byte.
            let mut offset_bits = 0u64;
            for (i, f) in def.fields.iter().enumerate() {
                match f.bit_width {
                    Some(0) => {
                        // Unnamed `:0`: close the current unit by rounding to the
                        // next storage-unit boundary of the declared type. It does
                        // *not* raise the record's alignment (matching gcc).
                        let unit = u64::from(bitfield_unit_bits(&f.ty));
                        offset_bits = round_up(offset_bits, unit);
                    }
                    Some(w) => {
                        let w = u64::from(w);
                        let unit = u64::from(bitfield_unit_bits(&f.ty));
                        // A bit-field starts a new unit rather than straddle an
                        // aligned container of its declared type.
                        if offset_bits % unit + w > unit {
                            offset_bits = round_up(offset_bits, unit);
                        }
                        let unit_index = offset_bits / unit;
                        let unit_start = unit_index * unit;
                        offsets[i] = unit_start / 8;
                        bits[i] = Some(BitPlacement {
                            unit_bits: bitfield_unit_bits(&f.ty),
                            bit_offset: (offset_bits - unit_start) as u32,
                            width: w as u32,
                            signed: f.ty.is_signed(),
                        });
                        align = align.max(align_of(recs, &f.ty));
                        offset_bits += w;
                    }
                    None => {
                        let a = field_align(recs, f);
                        align = align.max(a);
                        offset_bits = round_up(offset_bits, a * 8);
                        offsets[i] = offset_bits / 8;
                        offset_bits += size_of(recs, &f.ty) * 8;
                    }
                }
            }
            let size = round_up(offset_bits, align.max(1) * 8) / 8;
            RecordLayout { offsets, bits, size: size.max(1), align: align.max(1) }
        }
        RecordKind::Union => {
            // Every member (bit-field or not) overlays at offset 0.
            let mut size = 0u64;
            for (i, f) in def.fields.iter().enumerate() {
                if let Some(w) = f.bit_width {
                    if w > 0 {
                        bits[i] = Some(BitPlacement {
                            unit_bits: bitfield_unit_bits(&f.ty),
                            bit_offset: 0,
                            width: w,
                            signed: f.ty.is_signed(),
                        });
                        align = align.max(align_of(recs, &f.ty));
                    }
                } else {
                    align = align.max(field_align(recs, f));
                }
                size = size.max(size_of(recs, &f.ty));
            }
            let align = align.max(1);
            RecordLayout { offsets, bits, size: round_up(size, align).max(1), align }
        }
    }
}

fn record_align(recs: &Records, id: RecordId) -> u64 {
    record_layout(recs, id).align
}

fn record_size(recs: &Records, id: RecordId) -> u64 {
    record_layout(recs, id).size
}

/// The byte offset of field `idx` within record `id` (`0` for every union
/// member, cumulative with padding for a struct; the storage-unit offset for a
/// bit-field).
pub fn field_offset(recs: &Records, id: RecordId, idx: usize) -> u64 {
    record_layout(recs, id).offsets[idx]
}

/// The byte offset and bit placement of field `idx` within record `id`. The bit
/// placement is `Some` only for a bit-field member.
pub fn field_placement(recs: &Records, id: RecordId, idx: usize) -> (u64, Option<BitPlacement>) {
    let layout = record_layout(recs, id);
    (layout.offsets[idx], layout.bits[idx])
}

/// Resolve a member `name` within record `id`, descending into anonymous
/// `struct`/`union` members (C11). Returns the member's byte offset (composed
/// through any enclosing anonymous members) and its type.
pub fn resolve_member(recs: &Records, id: RecordId, name: &str) -> Option<(u64, CType)> {
    resolve_member_bits(recs, id, name).map(|(off, ty, _)| (off, ty))
}

/// Like [`resolve_member`], but also returns the bit placement when the resolved
/// member is a bit-field. The byte offset is composed through any enclosing
/// anonymous members; the bit placement's `bit_offset` is unchanged (an
/// anonymous member is always byte-aligned).
pub fn resolve_member_bits(
    recs: &Records,
    id: RecordId,
    name: &str,
) -> Option<(u64, CType, Option<BitPlacement>)> {
    let layout = record_layout(recs, id);
    let def = recs.get(id);
    // A directly-named member wins over descending into anonymous members.
    for (i, f) in def.fields.iter().enumerate() {
        if !f.anonymous && f.name == name {
            return Some((layout.offsets[i], f.ty.clone(), layout.bits[i]));
        }
    }
    for (i, f) in def.fields.iter().enumerate() {
        if f.anonymous
            && let CType::Record(sub) = &f.ty
            && let Some((off, ty, bp)) = resolve_member_bits(recs, *sub, name)
        {
            return Some((layout.offsets[i] + off, ty, bp));
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
            // A member with an explicit `alignas`, or any bit-field member (whose
            // sub-word packing a struct-of-field-types cannot express), can make a
            // natural struct of the field types mis-sized; model such a struct by a
            // byte blob of the right size/alignment (as for unions) so the storage
            // reserved by an `alloca` stays adequate.
            if def.fields.iter().any(|f| f.align.is_some() || f.bit_width.is_some()) {
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
