//! Values and constants.
//!
//! Every SSA value has a [`ValueId`] and a [`TypeId`]. A value is produced by
//! exactly one of the [`ValueDef`] kinds — an instruction result, a block
//! parameter (block arguments, *not* φ-nodes), an interned constant, or a
//! reference to a global or a function. Value ids index a per-function table
//! (see [`crate::ir::Function`]); use/def edges are first-class there, which is
//! what makes replace-all-uses and most rewrites cheap.
//!
//! Constants are **interned** module-wide (tenet T5). Integer constants are
//! arbitrary-precision, backed by [`puremp::Int`] — there is no bespoke bignum.
//! Floating-point constants are stored as their exact IEEE-754 bit pattern, so
//! they are host-independent and hash/compare bit-for-bit (which distinguishes
//! signed zeros and NaN payloads, as the semantics require).
//!
//! The value model has **no `undef`**: a value is either defined or
//! [`poison`](Const::Poison). See `docs/ir-design.md` §5.

use std::collections::HashMap;

use crate::ir::types::TypeId;
use crate::ir::{BlockId, FuncId, GlobalId, InstId};

/// A `Copy` handle to an SSA value within a function's value table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ValueId(u32);

impl ValueId {
    /// The dense index this id addresses.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub(crate) fn from_index(i: usize) -> Self {
        ValueId(i as u32)
    }
}

/// A `Copy` handle to an interned [`Const`] within a [`ConstPool`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ConstId(u32);

impl ConstId {
    /// The dense index this id addresses.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn from_index(i: usize) -> Self {
        ConstId(i as u32)
    }
}

/// What produces a value. Every [`ValueId`] resolves to exactly one of these.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ValueDef {
    /// The result of an instruction.
    Inst(InstId),
    /// The `idx`-th parameter of a block (a block argument).
    Param(BlockId, u32),
    /// An interned constant materialized as a value.
    Const(ConstId),
    /// A reference to a module global (its address, a pointer).
    Global(GlobalId),
    /// A reference to a function (its address, a pointer / callable).
    Func(FuncId),
}

/// A value: its defining site plus its type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Value {
    /// How this value is produced.
    pub def: ValueDef,
    /// The value's type.
    pub ty: TypeId,
}

/// The exact IEEE-754 bit pattern of a floating-point constant.
///
/// Storing raw bits (rather than a decoded float) makes constants hashable and
/// bit-exact; the evaluator decodes them into `puremp::Float` when it needs to
/// compute. The width is implied by the constant's [`TypeId`] but recorded here
/// too so the pattern is self-describing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatBits {
    /// binary16 bit pattern.
    F16(u16),
    /// binary32 bit pattern.
    F32(u32),
    /// binary64 bit pattern.
    F64(u64),
}

/// An interned constant value.
///
/// The `TypeId` on each variant is the constant's type; it is part of the
/// interning key, so `i8 0` and `i32 0` are distinct constants.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Const {
    /// An arbitrary-precision integer constant of an integer type. The value is
    /// interpreted modulo `2^width`; the stored `puremp::Int` is the exact
    /// mathematical representative (two's-complement is applied by operations).
    Int {
        /// The integer type of this constant.
        ty: TypeId,
        /// The arbitrary-precision value.
        value: puremp::Int,
    },
    /// A floating-point constant of a float type, stored as raw IEEE bits.
    Float {
        /// The float type of this constant.
        ty: TypeId,
        /// The exact bit pattern.
        bits: FloatBits,
    },
    /// The null pointer of a pointer type.
    Null(TypeId),
    /// **Poison** of any type: a deferred error that taints any operation that
    /// depends on it, until a `freeze` pins it to a concrete value. There is no
    /// `undef` (see `docs/ir-design.md` §5).
    Poison(TypeId),
    /// An aggregate (array or struct) constant, element-wise.
    Aggregate {
        /// The array or struct type of this constant.
        ty: TypeId,
        /// The element/field constants, in order.
        elems: Vec<ConstId>,
    },
}

impl Const {
    /// The type of this constant.
    pub fn type_id(&self) -> TypeId {
        match self {
            Const::Int { ty, .. }
            | Const::Float { ty, .. }
            | Const::Null(ty)
            | Const::Poison(ty)
            | Const::Aggregate { ty, .. } => *ty,
        }
    }
}

/// The module-wide interning pool for constants (tenet T5).
///
/// Structurally equal constants share one [`ConstId`]. This is what lets pure
/// value nodes be hash-consed later (bets B6/B7) without reworking the core.
#[derive(Debug, Default)]
pub struct ConstPool {
    consts: Vec<Const>,
    dedup: HashMap<Const, ConstId>,
}

impl ConstPool {
    /// Create an empty constant pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a constant, returning its stable handle. Equal constants intern
    /// to equal handles.
    pub fn intern(&mut self, c: Const) -> ConstId {
        if let Some(&id) = self.dedup.get(&c) {
            return id;
        }
        let id = ConstId::from_index(self.consts.len());
        self.consts.push(c.clone());
        self.dedup.insert(c, id);
        id
    }

    /// Resolve a handle back to its constant.
    #[inline]
    pub fn get(&self, id: ConstId) -> &Const {
        &self.consts[id.index()]
    }

    /// The type of the interned constant.
    pub fn type_of(&self, id: ConstId) -> TypeId {
        self.get(id).type_id()
    }

    /// Number of distinct constants interned so far.
    pub fn len(&self) -> usize {
        self.consts.len()
    }

    /// Whether nothing has been interned yet.
    pub fn is_empty(&self) -> bool {
        self.consts.is_empty()
    }

    /// Iterate the interned constants in id order (id `0`, `1`, ...). The `n`-th
    /// item is the constant of [`ConstId`] `n` — used to rebuild an old→new id map
    /// by position when merging modules for LTO.
    pub fn iter(&self) -> impl Iterator<Item = &Const> {
        self.consts.iter()
    }
}
