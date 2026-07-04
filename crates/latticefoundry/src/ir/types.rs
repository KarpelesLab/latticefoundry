//! The LatticeFoundry IR type system. See ROADMAP Phase 1.

/// IEEE-754 floating-point formats supported by the IR.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatKind {
    /// Half precision (binary16).
    F16,
    /// Single precision (binary32).
    F32,
    /// Double precision (binary64).
    F64,
}

/// A LatticeFoundry IR type.
///
/// Pointers are opaque: the pointee type is carried by the operations that
/// dereference them rather than by the pointer itself. This keeps the type
/// graph acyclic and simplifies the verifier.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Type {
    /// The unit / no-value type, produced by e.g. a bare `ret`.
    Void,
    /// An arbitrary-width integer such as `i1`, `i32`, or `i128`.
    Int(u32),
    /// An IEEE-754 floating-point value.
    Float(FloatKind),
    /// An opaque (untyped) pointer.
    Ptr,
    /// A fixed-length array `[N x T]`.
    Array(Box<Type>, u64),
    /// An anonymous aggregate of fields.
    Struct(Vec<Type>),
    /// A function type `(params...) -> ret`.
    Func(Box<FuncType>),
}

impl Type {
    /// The canonical boolean type, `i1`.
    pub const BOOL: Type = Type::Int(1);

    /// Construct an integer type of the given bit width.
    pub fn int(bits: u32) -> Type {
        Type::Int(bits)
    }

    /// Whether this is any integer type.
    pub fn is_integer(&self) -> bool {
        matches!(self, Type::Int(_))
    }

    /// The width in bits of a scalar type, if it has one.
    pub fn bit_width(&self) -> Option<u32> {
        match self {
            Type::Int(w) => Some(*w),
            Type::Float(FloatKind::F16) => Some(16),
            Type::Float(FloatKind::F32) => Some(32),
            Type::Float(FloatKind::F64) => Some(64),
            _ => None,
        }
    }
}

/// The signature of a function type: parameter types, a return type, and
/// whether the function is variadic.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FuncType {
    /// Fixed parameter types, in order.
    pub params: Vec<Type>,
    /// The return type (`Type::Void` for functions returning nothing).
    pub ret: Type,
    /// Whether the function accepts trailing variadic arguments.
    pub variadic: bool,
}

impl FuncType {
    /// A non-variadic signature.
    pub fn new(params: Vec<Type>, ret: Type) -> Self {
        Self { params, ret, variadic: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_widths() {
        assert_eq!(Type::BOOL.bit_width(), Some(1));
        assert_eq!(Type::int(64).bit_width(), Some(64));
        assert_eq!(Type::Float(FloatKind::F32).bit_width(), Some(32));
        assert_eq!(Type::Ptr.bit_width(), None);
        assert!(Type::int(8).is_integer());
    }
}
