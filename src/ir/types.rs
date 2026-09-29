//! The LatticeFoundry IR type system.
//!
//! Types are **interned** (hash-consed) from day one (tenet T5): the
//! [`TypeContext`] hands out small `Copy` [`TypeId`] handles with structural
//! identity, so two structurally equal types always share one id and compare in
//! constant time. Types reference one another *by id*, never by owning boxes,
//! which keeps the type graph flat and arena-friendly.
//!
//! Pointers are **opaque**: a pointer carries no pointee type. The accessed
//! type lives on the memory operation that dereferences the pointer (see
//! [`crate::ir::inst`]). This is the design LLVM converged on after years of
//! typed-pointer pain, and we start there. See `docs/ir-design.md` §3.
//!
//! A pointer lives in an **address space**: [`Type::Ptr`] is the default space
//! `0`, [`Type::PtrIn`] any other (`ptr addrspace(N)` in the text form). Sizes
//! and alignments follow the context's [`DataLayout`], LP64 unless the module
//! sets another (`docs/ir-design.md` §3a).

use std::collections::HashMap;

use crate::ir::datalayout::DataLayout;

/// A `Copy` handle to an interned [`Type`] within a [`TypeContext`].
///
/// Structural identity: equal types always intern to equal ids, so id equality
/// *is* type equality. The wrapped index is an implementation detail.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct TypeId(u32);

impl TypeId {
    /// The dense index this id addresses within its [`TypeContext`].
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn from_index(i: usize) -> Self {
        TypeId(i as u32)
    }
}

/// IEEE-754 floating-point formats supported by the IR.
///
/// Wider or exotic formats (bf16, fp128, x87 80-bit) are added only when a
/// target needs them; the semantics of these three are host-independent and
/// exact.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatKind {
    /// Half precision (IEEE-754 binary16).
    F16,
    /// Single precision (IEEE-754 binary32).
    F32,
    /// Double precision (IEEE-754 binary64).
    F64,
}

impl FloatKind {
    /// The width in bits of this format.
    #[inline]
    pub fn bit_width(self) -> u32 {
        match self {
            FloatKind::F16 => 16,
            FloatKind::F32 => 32,
            FloatKind::F64 => 64,
        }
    }
}

/// A LatticeFoundry IR type.
///
/// Composite types reference their components by [`TypeId`] rather than owning
/// them, so the whole type graph lives flat inside a [`TypeContext`]. Fixed-width
/// SIMD vectors are [`Type::Vector`]; scalable vectors are deferred until a
/// scalable-vector target (SVE/RVV) is real (`docs/ir-design.md` §3, §6e).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Type {
    /// The unit / no-value type, produced by e.g. a bare `ret`.
    Void,
    /// An arbitrary-width integer such as `i1`, `i32`, or `i128`. Integers are
    /// sign-agnostic; signedness is a property of the *operation*, not the type.
    Int(u32),
    /// An IEEE-754 floating-point value.
    Float(FloatKind),
    /// An opaque (untyped) pointer into the default address space `0`.
    Ptr,
    /// An opaque pointer into address space `n`, **`n ≥ 1`** (`ptr
    /// addrspace(n)`). A separate variant, rather than a field on [`Type::Ptr`],
    /// so that code written before address spaces keeps compiling; interning
    /// normalizes `PtrIn(0)` to [`Type::Ptr`], so each space has exactly one
    /// type. Use [`Type::is_ptr`] / [`Type::addr_space`] to handle both.
    PtrIn(u32),
    /// A fixed-length array `[N x T]`.
    Array(TypeId, u64),
    /// An anonymous aggregate of fields, laid out in declaration order.
    Struct(Vec<TypeId>),
    /// A function type `(params...) -> ret`.
    Func(FuncType),
    /// A fixed-length SIMD vector `<N x T>` of `N ≥ 1` lanes, a first-class
    /// **value** (unlike an array, which is an address; `docs/ir-design.md` §6e).
    /// The element type is `i1`, `i8`, `i16`, `i32`, `i64` or a float type (the
    /// verifier enforces this). Operations act lane-wise and poison is tracked
    /// per lane.
    Vector(TypeId, u32),
}

impl Type {
    /// Whether this is any integer type.
    #[inline]
    pub fn is_integer(&self) -> bool {
        matches!(self, Type::Int(_))
    }

    /// Whether this is any floating-point type.
    #[inline]
    pub fn is_float(&self) -> bool {
        matches!(self, Type::Float(_))
    }

    /// Whether this is a pointer type, in any address space.
    #[inline]
    pub fn is_ptr(&self) -> bool {
        matches!(self, Type::Ptr | Type::PtrIn(_))
    }

    /// The address space of a pointer type (`0` for [`Type::Ptr`]), or `None`
    /// for a non-pointer.
    #[inline]
    pub fn addr_space(&self) -> Option<u32> {
        match self {
            Type::Ptr => Some(0),
            Type::PtrIn(n) => Some(*n),
            _ => None,
        }
    }

    /// Whether this is a fixed-length vector type.
    #[inline]
    pub fn is_vector(&self) -> bool {
        matches!(self, Type::Vector(..))
    }

    /// The width in bits of a scalar (integer or float) type, if it has one.
    #[inline]
    pub fn bit_width(&self) -> Option<u32> {
        match self {
            Type::Int(w) => Some(*w),
            Type::Float(k) => Some(k.bit_width()),
            _ => None,
        }
    }
}

/// The signature of a function type: parameter types, a return type, and
/// whether the function is variadic. Components are referenced by [`TypeId`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FuncType {
    /// Fixed parameter types, in order.
    pub params: Vec<TypeId>,
    /// The return type (the interned `Void` type for functions returning nothing).
    pub ret: TypeId,
    /// Whether the function accepts trailing variadic arguments.
    pub variadic: bool,
}

/// The size and alignment of a type under a [`DataLayout`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Layout {
    /// Size in bytes (excluding trailing tail padding for a bare value).
    pub size: u64,
    /// Alignment in bytes (always a power of two, at least 1).
    pub align: u64,
}

/// The interning context for IR types.
///
/// This is the single owner of every [`Type`] in a module. It deduplicates on
/// insertion, so [`TypeContext::intern`] of two structurally equal types yields
/// the same [`TypeId`]. Convenience constructors (`int`, `ptr`, `array`, ...)
/// intern in one step.
///
/// The context also carries the module's [`DataLayout`] (LP64 by default), which
/// every size/alignment query follows.
#[derive(Clone, Debug, Default)]
pub struct TypeContext {
    types: Vec<Type>,
    dedup: HashMap<Type, TypeId>,
    layout: DataLayout,
}

impl TypeContext {
    /// Create an empty type context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a type, returning its stable handle. Equal types intern equal.
    pub fn intern(&mut self, ty: Type) -> TypeId {
        // One type per address space: `PtrIn(0)` *is* `Ptr`.
        let ty = if ty == Type::PtrIn(0) { Type::Ptr } else { ty };
        if let Some(&id) = self.dedup.get(&ty) {
            return id;
        }
        let id = TypeId::from_index(self.types.len());
        self.types.push(ty.clone());
        self.dedup.insert(ty, id);
        id
    }

    /// Resolve a handle back to its type.
    #[inline]
    pub fn get(&self, id: TypeId) -> &Type {
        &self.types[id.index()]
    }

    /// Number of distinct types interned so far.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Whether nothing has been interned yet.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Iterate the interned types in id order (id `0`, `1`, ...). The `n`-th item
    /// is the type of [`TypeId`] `n`, which is what lets a consumer rebuild an
    /// old→new id map by position (used when merging modules for LTO).
    pub fn iter(&self) -> impl Iterator<Item = &Type> {
        self.types.iter()
    }

    // --- convenience constructors -------------------------------------------

    /// The `void` type.
    pub fn void(&mut self) -> TypeId {
        self.intern(Type::Void)
    }

    /// An integer type of the given bit width (`int(1)` is the boolean type).
    pub fn int(&mut self, bits: u32) -> TypeId {
        self.intern(Type::Int(bits))
    }

    /// The boolean type, `i1`.
    pub fn bool(&mut self) -> TypeId {
        self.int(1)
    }

    /// A floating-point type of the given format.
    pub fn float(&mut self, kind: FloatKind) -> TypeId {
        self.intern(Type::Float(kind))
    }

    /// The opaque pointer type (address space 0).
    pub fn ptr(&mut self) -> TypeId {
        self.intern(Type::Ptr)
    }

    /// The opaque pointer type of address space `addr_space` (`ptr` for 0).
    pub fn ptr_in(&mut self, addr_space: u32) -> TypeId {
        self.intern(Type::PtrIn(addr_space))
    }

    /// A fixed-length array type `[len x elem]`.
    pub fn array(&mut self, elem: TypeId, len: u64) -> TypeId {
        self.intern(Type::Array(elem, len))
    }

    /// An anonymous struct type over the given field types.
    pub fn struct_(&mut self, fields: Vec<TypeId>) -> TypeId {
        self.intern(Type::Struct(fields))
    }

    /// A fixed-length vector type `<lanes x elem>`.
    pub fn vector(&mut self, elem: TypeId, lanes: u32) -> TypeId {
        self.intern(Type::Vector(elem, lanes))
    }

    /// A function type. Pass `ret = void()` for a procedure.
    pub fn func(&mut self, params: Vec<TypeId>, ret: TypeId, variadic: bool) -> TypeId {
        self.intern(Type::Func(FuncType { params, ret, variadic }))
    }

    // --- queries ------------------------------------------------------------

    /// Whether the referenced type is an integer type.
    pub fn is_integer(&self, id: TypeId) -> bool {
        self.get(id).is_integer()
    }

    /// The scalar bit width of the referenced type, if any.
    pub fn bit_width(&self, id: TypeId) -> Option<u32> {
        self.get(id).bit_width()
    }

    /// Whether the referenced type is a pointer (any address space).
    pub fn is_ptr(&self, id: TypeId) -> bool {
        self.get(id).is_ptr()
    }

    /// The address space of a pointer type, or `None` for a non-pointer.
    pub fn addr_space(&self, id: TypeId) -> Option<u32> {
        self.get(id).addr_space()
    }

    /// The width in bits of a pointer type under the data layout, or `None`
    /// for a non-pointer.
    pub fn pointer_bits(&self, id: TypeId) -> Option<u32> {
        self.addr_space(id).map(|a| self.layout.pointer_bits(a))
    }

    /// The bit width of an integer or pointer type (a pointer counts at its
    /// address space's width), or `None` for anything else. This is the width
    /// at which `icmp` compares and `ptrtoint`/`inttoptr` convert.
    pub fn int_or_ptr_bits(&self, id: TypeId) -> Option<u32> {
        match self.get(id) {
            Type::Int(w) => Some(*w),
            t => t.addr_space().map(|a| self.layout.pointer_bits(a)),
        }
    }

    /// Whether the referenced type is a vector type.
    pub fn is_vector(&self, id: TypeId) -> bool {
        self.get(id).is_vector()
    }

    /// The `(element type, lane count)` of a vector type, or `None` for any
    /// other type.
    pub fn vector_parts(&self, id: TypeId) -> Option<(TypeId, u32)> {
        match self.get(id) {
            Type::Vector(elem, n) => Some((*elem, *n)),
            _ => None,
        }
    }

    /// The lane type of a vector, or the type itself for a non-vector: the type
    /// a lane-wise operation computes on.
    pub fn scalar_of(&self, id: TypeId) -> TypeId {
        self.vector_parts(id).map_or(id, |(e, _)| e)
    }

    /// The total bit width of a scalar or vector type (`lanes × lane width`
    /// for a vector), or `None` for pointers, aggregates and the like. This is
    /// the width a `bitcast` must preserve.
    pub fn total_bits(&self, id: TypeId) -> Option<u64> {
        match self.get(id) {
            Type::Vector(elem, n) => self.bit_width(*elem).map(|w| u64::from(w) * u64::from(*n)),
            t => t.bit_width().map(u64::from),
        }
    }

    // --- data layout --------------------------------------------------------
    //
    // Every size/alignment query follows the context's `DataLayout`, which the
    // module sets (LP64 unless told otherwise). The builder's offset helpers
    // (`struct_field` / `array_elem`), `alloca`, the verifier and global-data
    // emission all go through these.

    /// The data layout the size/alignment queries follow.
    #[inline]
    pub fn data_layout(&self) -> &DataLayout {
        &self.layout
    }

    /// Replace the data layout. Types are layout-independent, so every interned
    /// id stays valid; only sizes, alignments and offsets change.
    pub fn set_data_layout(&mut self, layout: DataLayout) {
        self.layout = layout;
    }

    /// The [`Layout`] (size and alignment) of a type under the data layout.
    pub fn layout(&self, id: TypeId) -> Layout {
        let dl = &self.layout;
        match self.get(id) {
            Type::Void => Layout { size: 0, align: 1 },
            Type::Int(bits) => {
                let size = u64::from(bits.div_ceil(8));
                Layout { size, align: dl.int_align(*bits) }
            }
            Type::Float(k) => {
                let size = u64::from(k.bit_width() / 8);
                Layout { size, align: dl.float_align(k.bit_width()) }
            }
            // A function reference is a pointer into the program address space.
            Type::Ptr | Type::PtrIn(_) | Type::Func(_) => {
                let space = self.get(id).addr_space().unwrap_or(dl.program_addr_space());
                let p = dl.pointer_or_default(space);
                Layout { size: p.bytes(), align: p.align }
            }
            Type::Array(elem, len) => {
                let stride = self.stride(*elem);
                let align = self.layout(*elem).align;
                Layout { size: stride * *len, align }
            }
            Type::Struct(fields) => {
                let mut offset = 0u64;
                let mut align = 1u64;
                for &f in fields {
                    let l = self.layout(f);
                    align = align.max(l.align);
                    offset = round_up(offset, l.align) + l.size;
                }
                Layout { size: round_up(offset, align), align }
            }
            // Lanes are packed at the element's size (an `i1` lane takes one
            // byte); the vector is aligned to its size rounded up to a power of
            // two, capped at 16 bytes (the SSE/NEON register width).
            Type::Vector(elem, n) => {
                let el = self.layout(*elem);
                let size = el.size * u64::from(*n);
                let align = size.max(1).next_power_of_two().min(16).max(el.align);
                Layout { size, align }
            }
        }
    }

    /// The size in bytes of a type.
    pub fn size_of(&self, id: TypeId) -> u64 {
        self.layout(id).size
    }

    /// The alignment in bytes of a type.
    pub fn align_of(&self, id: TypeId) -> u64 {
        self.layout(id).align
    }

    /// The stride of an array element: its size rounded up to its alignment.
    /// This is the byte distance between consecutive elements.
    pub fn stride(&self, id: TypeId) -> u64 {
        let l = self.layout(id);
        round_up(l.size, l.align)
    }

    /// The byte offset and field type of struct field `idx`.
    ///
    /// Panics if `id` is not a struct type or `idx` is out of range; callers in
    /// the builder validate this against the type they were handed.
    pub fn field_offset(&self, id: TypeId, idx: u32) -> (u64, TypeId) {
        let Type::Struct(fields) = self.get(id) else {
            panic!("field_offset on a non-struct type");
        };
        let mut offset = 0u64;
        for (i, &f) in fields.iter().enumerate() {
            let l = self.layout(f);
            offset = round_up(offset, l.align);
            if i as u32 == idx {
                return (offset, f);
            }
            offset += l.size;
        }
        panic!("struct field index {idx} out of range");
    }

    /// The element type of an array type, or `None` for non-arrays.
    pub fn array_elem(&self, id: TypeId) -> Option<TypeId> {
        match self.get(id) {
            Type::Array(elem, _) => Some(*elem),
            _ => None,
        }
    }
}

/// Round `value` up to the next multiple of `align` (a power of two ≥ 1).
#[inline]
fn round_up(value: u64, align: u64) -> u64 {
    debug_assert!(align >= 1);
    value.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_gives_equal_ids_for_equal_types() {
        let mut cx = TypeContext::new();
        let a = cx.int(32);
        let b = cx.int(32);
        let c = cx.int(64);
        assert_eq!(a, b, "equal types must intern to equal ids");
        assert_ne!(a, c);

        let arr1 = cx.array(a, 4);
        let arr2 = cx.array(b, 4);
        assert_eq!(arr1, arr2, "structural equality reaches through composites");
    }

    #[test]
    fn scalar_queries() {
        let mut cx = TypeContext::new();
        let i1 = cx.bool();
        let i64_ = cx.int(64);
        let f32 = cx.float(FloatKind::F32);
        let p = cx.ptr();
        assert_eq!(cx.bit_width(i1), Some(1));
        assert_eq!(cx.bit_width(i64_), Some(64));
        assert_eq!(cx.bit_width(f32), Some(32));
        assert_eq!(cx.bit_width(p), None);
        assert!(cx.is_integer(i64_));
    }

    #[test]
    fn struct_layout_and_field_offsets() {
        let mut cx = TypeContext::new();
        let i8_ = cx.int(8);
        let i32_ = cx.int(32);
        // struct { i8, i32 }: i8 at 0, then pad to 4, i32 at 4; size 8, align 4.
        let s = cx.struct_(vec![i8_, i32_]);
        assert_eq!(cx.size_of(s), 8);
        assert_eq!(cx.align_of(s), 4);
        assert_eq!(cx.field_offset(s, 0), (0, i8_));
        assert_eq!(cx.field_offset(s, 1), (4, i32_));
    }

    #[test]
    fn vector_types_and_layout() {
        let mut cx = TypeContext::new();
        let i32_ = cx.int(32);
        let f64_ = cx.float(FloatKind::F64);
        let v4 = cx.vector(i32_, 4);
        assert_eq!(v4, cx.vector(i32_, 4));
        assert_eq!(cx.vector_parts(v4), Some((i32_, 4)));
        assert_eq!(cx.scalar_of(v4), i32_);
        assert_eq!(cx.scalar_of(i32_), i32_);
        assert_eq!(cx.total_bits(v4), Some(128));
        assert_eq!(cx.bit_width(v4), None, "a vector is not a scalar");
        assert_eq!((cx.size_of(v4), cx.align_of(v4)), (16, 16));
        let v3 = cx.vector(i32_, 3);
        assert_eq!((cx.size_of(v3), cx.align_of(v3), cx.stride(v3)), (12, 16, 16));
        let v2d = cx.vector(f64_, 2);
        assert_eq!(cx.total_bits(v2d), Some(128));
        let v8 = cx.vector(i32_, 8);
        assert_eq!((cx.size_of(v8), cx.align_of(v8)), (32, 16));
        let i1 = cx.bool();
        let m4 = cx.vector(i1, 4);
        assert_eq!((cx.size_of(m4), cx.total_bits(m4)), (4, Some(4)));
    }

    #[test]
    fn array_stride() {
        let mut cx = TypeContext::new();
        let i32_ = cx.int(32);
        let arr = cx.array(i32_, 3);
        assert_eq!(cx.stride(i32_), 4);
        assert_eq!(cx.size_of(arr), 12);
    }

    /// `{ i8, ptr, i64, f64, fn }` and scalars under LP64, ILP32 (with a 4-byte
    /// `i64`, as i386 System V), and an AVR-like 16-bit, byte-aligned layout.
    #[test]
    fn layouts_follow_the_data_layout() {
        let mut cx = TypeContext::new();
        let i8_ = cx.int(8);
        let i16_ = cx.int(16);
        let i64_ = cx.int(64);
        let f64_ = cx.float(FloatKind::F64);
        let p = cx.ptr();
        let void = cx.void();
        let fnty = cx.func(vec![], void, false);
        let s = cx.struct_(vec![i8_, p, i64_, f64_, fnty]);
        let arr = cx.array(p, 3);

        // LP64: 8-byte pointers; i8@0 ptr@8 i64@16 f64@24 fn@32, size 40.
        assert_eq!((cx.size_of(p), cx.align_of(p)), (8, 8));
        assert_eq!(cx.field_offset(s, 4).0, 32);
        assert_eq!((cx.size_of(s), cx.align_of(s)), (40, 8));
        assert_eq!(cx.size_of(arr), 24);

        // ILP32 with i64/f64 4-aligned: i8@0 ptr@4 i64@8 f64@16 fn@24, size 28.
        let ilp32 = DataLayout::ilp32().with_int_align(64, 4).unwrap().with_float_align(64, 4).unwrap();
        cx.set_data_layout(ilp32);
        assert_eq!((cx.size_of(p), cx.align_of(p)), (4, 4));
        assert_eq!((cx.size_of(fnty), cx.align_of(fnty)), (4, 4));
        assert_eq!((cx.size_of(i64_), cx.align_of(i64_)), (8, 4));
        assert_eq!(cx.field_offset(s, 2).0, 8);
        assert_eq!(cx.field_offset(s, 4).0, 24);
        assert_eq!((cx.size_of(s), cx.align_of(s)), (28, 4));
        assert_eq!(cx.size_of(arr), 12);

        // AVR-like: 16-bit pointers, everything byte-aligned, 24-bit flash
        // pointers in address space 1 which holds the functions.
        let avr = DataLayout::parse("p:16:8-p1:24:8-i16:8-i32:8-i64:8-f32:8-f64:8-S8-n8-P1").unwrap();
        cx.set_data_layout(avr);
        let flash = cx.ptr_in(1);
        assert_eq!((cx.size_of(p), cx.align_of(p)), (2, 1));
        assert_eq!((cx.size_of(flash), cx.align_of(flash)), (3, 1));
        assert_eq!(cx.size_of(fnty), 3, "a function reference is a program-space pointer");
        assert_eq!((cx.size_of(i16_), cx.align_of(i16_)), (2, 1));
        assert_eq!(cx.field_offset(s, 1).0, 1);
        assert_eq!(cx.field_offset(s, 2).0, 3);
        assert_eq!(cx.field_offset(s, 4).0, 19);
        assert_eq!((cx.size_of(s), cx.align_of(s)), (22, 1));
        assert_eq!(cx.stride(p), 2);
        assert_eq!(cx.size_of(arr), 6);
        assert_eq!(cx.pointer_bits(flash), Some(24));
        assert_eq!(cx.int_or_ptr_bits(p), Some(16));
    }

    #[test]
    fn address_spaces_intern_to_one_type_each() {
        let mut cx = TypeContext::new();
        let p = cx.ptr();
        assert_eq!(cx.ptr_in(0), p, "addrspace(0) is plain ptr");
        assert_eq!(cx.intern(Type::PtrIn(0)), p);
        let q = cx.ptr_in(1);
        assert_ne!(q, p);
        assert_eq!(cx.ptr_in(1), q);
        assert_eq!(cx.addr_space(p), Some(0));
        assert_eq!(cx.addr_space(q), Some(1));
        assert!(cx.is_ptr(q) && cx.get(q).is_ptr());
        let i8_ = cx.int(8);
        assert_eq!(cx.addr_space(i8_), None);
    }
}
