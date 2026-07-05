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

use std::collections::HashMap;

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
/// them, so the whole type graph lives flat inside a [`TypeContext`]. Vectors
/// and scalable vectors are deliberately deferred until a SIMD target is real
/// (`docs/ir-design.md` §3).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Type {
    /// The unit / no-value type, produced by e.g. a bare `ret`.
    Void,
    /// An arbitrary-width integer such as `i1`, `i32`, or `i128`. Integers are
    /// sign-agnostic; signedness is a property of the *operation*, not the type.
    Int(u32),
    /// An IEEE-754 floating-point value.
    Float(FloatKind),
    /// An opaque (untyped) pointer. Address spaces are deferred; the single
    /// default address space is implied.
    Ptr,
    /// A fixed-length array `[N x T]`.
    Array(TypeId, u64),
    /// An anonymous aggregate of fields, laid out in declaration order.
    Struct(Vec<TypeId>),
    /// A function type `(params...) -> ret`.
    Func(FuncType),
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

/// The size and alignment of a type under the default data layout.
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
#[derive(Debug, Default)]
pub struct TypeContext {
    types: Vec<Type>,
    dedup: HashMap<Type, TypeId>,
}

impl TypeContext {
    /// Create an empty type context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a type, returning its stable handle. Equal types intern equal.
    pub fn intern(&mut self, ty: Type) -> TypeId {
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

    /// The opaque pointer type.
    pub fn ptr(&mut self) -> TypeId {
        self.intern(Type::Ptr)
    }

    /// A fixed-length array type `[len x elem]`.
    pub fn array(&mut self, elem: TypeId, len: u64) -> TypeId {
        self.intern(Type::Array(elem, len))
    }

    /// An anonymous struct type over the given field types.
    pub fn struct_(&mut self, fields: Vec<TypeId>) -> TypeId {
        self.intern(Type::Struct(fields))
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

    // --- data layout --------------------------------------------------------
    //
    // A provisional, target-independent default layout used by the builder's
    // offset helpers (`struct_field` / `array_elem`) and by `alloca`. It models
    // a 64-bit byte-addressed machine with natural alignment. A real per-target
    // data layout replaces this in a later phase; nothing in the IR *encodes*
    // these numbers, so swapping the layout is a local change.

    /// The [`Layout`] (size and alignment) of a type under the default layout.
    pub fn layout(&self, id: TypeId) -> Layout {
        match self.get(id) {
            Type::Void => Layout { size: 0, align: 1 },
            Type::Int(bits) => {
                let size = u64::from(bits.div_ceil(8));
                Layout { size, align: align_for(size) }
            }
            Type::Float(k) => {
                let size = u64::from(k.bit_width() / 8);
                Layout { size, align: size.max(1) }
            }
            // Pointers and function references are pointer-sized on a 64-bit machine.
            Type::Ptr | Type::Func(_) => Layout { size: 8, align: 8 },
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

/// The natural alignment for an integer of the given byte size: the size
/// rounded up to a power of two, capped at 8 (the machine word).
#[inline]
fn align_for(size: u64) -> u64 {
    if size == 0 {
        return 1;
    }
    size.next_power_of_two().min(8)
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
    fn array_stride() {
        let mut cx = TypeContext::new();
        let i32_ = cx.int(32);
        let arr = cx.array(i32_, 3);
        assert_eq!(cx.stride(i32_), 4);
        assert_eq!(cx.size_of(arr), 12);
    }
}
