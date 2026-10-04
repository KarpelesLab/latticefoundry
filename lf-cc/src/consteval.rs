//! Integer constant expressions, evaluated with their C types.
//!
//! The parser (array bounds, `case` labels, enumerators, `constexpr`,
//! `_Static_assert`) and sema (static initializers) both fold integer constant
//! expressions. Each value carries its C type, so the folding follows the
//! language: the integer promotions and usual arithmetic conversions pick the
//! type an operation is performed in, the result wraps to that type's width,
//! and unsigned division, right shifts and comparisons are unsigned. Values up
//! to 128 bits wide (`__int128`) are exact; nothing here can overflow.
//!
//! A value is kept *normalized* to its type: a signed one sign-extended from
//! its width, an unsigned one below 128 bits zero-extended. An `unsigned
//! __int128` above `i128::MAX` is held as its bit pattern (a negative `i128`).

use crate::ast::{BinaryOp, CType, Expr, ExprKind, IntTy, UnaryOp};
use crate::sema::{promote, usual_arith};

/// An integer constant and its (integer or `_Bool`) C type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CInt {
    /// The value, normalized to `ty` (see the module docs).
    pub value: i128,
    /// The value's type.
    pub ty: CType,
}

impl CInt {
    /// The constant `value` converted to integer type `ty` (wrapping to its
    /// width). A non-integer `ty` keeps the value as an `int`-or-wider type
    /// chosen by [`CInt::natural`].
    pub fn new(value: i128, ty: &CType) -> CInt {
        let ty = ty.unqual();
        if ty.is_integer() {
            CInt { value: reduce_const_to_type(value, ty), ty: ty.clone() }
        } else {
            CInt::natural(value)
        }
    }

    /// An `int` constant.
    pub fn int(value: i128) -> CInt {
        CInt::new(value, &CType::int())
    }

    /// A `size_t` (`unsigned long`) constant.
    pub fn size(value: u64) -> CInt {
        CInt::new(i128::from(value), &CType::Int(IntTy::new(64, false)))
    }

    /// A constant of the first of `int`, `long`, `unsigned long` and
    /// `__int128` that holds `value` (the type of an enumerator whose value
    /// does not fit an `int`, as gcc gives it its enum's underlying type).
    pub fn natural(value: i128) -> CInt {
        let ty = if i32::try_from(value).is_ok() {
            CType::int()
        } else if i64::try_from(value).is_ok() {
            CType::long()
        } else if u64::try_from(value).is_ok() {
            CType::Int(IntTy::new(64, false))
        } else {
            CType::Int(IntTy::new(128, true))
        };
        CInt { value, ty }
    }

    /// The value as the unsigned bit pattern of its two's complement.
    fn bits(&self) -> u128 {
        self.value as u128
    }
}

/// Convert constant `c` to type `ty`, as a cast does: to an integer type it
/// wraps to the type's width (`_Bool` tests against zero); to a pointer it is
/// an address-sized `unsigned long`; any other type leaves it unchanged.
pub fn cast(c: CInt, ty: &CType) -> CInt {
    let ty = ty.unqual();
    if ty.is_integer() {
        CInt::new(c.value, ty)
    } else if ty.is_pointer() {
        CInt::new(c.value, &CType::Int(IntTy::new(64, false)))
    } else {
        c
    }
}

/// Apply unary operator `op` (`-`, `+`, `~`, `!`).
pub fn unary(op: UnaryOp, a: CInt) -> Option<CInt> {
    match op {
        UnaryOp::Neg => {
            let ty = promote(&a.ty);
            Some(CInt::new(a.value.wrapping_neg(), &ty))
        }
        UnaryOp::Plus => {
            let ty = promote(&a.ty);
            Some(CInt::new(a.value, &ty))
        }
        UnaryOp::BitNot => {
            let ty = promote(&a.ty);
            Some(CInt::new(!a.value, &ty))
        }
        UnaryOp::LNot => Some(CInt::int(i128::from(a.value == 0))),
        _ => None,
    }
}

/// Apply binary operator `op`. Division or remainder by zero, a negative shift
/// count and one of 128 or more are not constants (`None`).
pub fn binary(op: BinaryOp, a: CInt, b: CInt) -> Option<CInt> {
    use BinaryOp::*;
    match op {
        LAnd => return Some(CInt::int(i128::from(a.value != 0 && b.value != 0))),
        LOr => return Some(CInt::int(i128::from(a.value != 0 || b.value != 0))),
        Shl | Shr => {
            let ty = promote(&a.ty);
            let a = CInt::new(a.value, &ty);
            let n = u32::try_from(b.value).ok().filter(|&n| n < 128)?;
            let v = match op {
                Shl => (a.bits() << n) as i128,
                _ if ty.is_signed() => a.value >> n,
                _ => (a.bits() >> n) as i128,
            };
            return Some(CInt::new(v, &ty));
        }
        _ => {}
    }
    let ty = usual_arith(&a.ty, &b.ty);
    let signed = ty.is_signed();
    let (a, b) = (CInt::new(a.value, &ty), CInt::new(b.value, &ty));
    let cmp = |r: bool| Some(CInt::int(i128::from(r)));
    let v = match op {
        Add => a.bits().wrapping_add(b.bits()) as i128,
        Sub => a.bits().wrapping_sub(b.bits()) as i128,
        Mul => a.bits().wrapping_mul(b.bits()) as i128,
        Div | Rem if b.value == 0 => return None,
        Div if signed => a.value.wrapping_div(b.value),
        Div => (a.bits() / b.bits()) as i128,
        Rem if signed => a.value.wrapping_rem(b.value),
        Rem => (a.bits() % b.bits()) as i128,
        BitAnd => a.value & b.value,
        BitOr => a.value | b.value,
        BitXor => a.value ^ b.value,
        Eq => return cmp(a.value == b.value),
        Ne => return cmp(a.value != b.value),
        Lt if signed => return cmp(a.value < b.value),
        Le if signed => return cmp(a.value <= b.value),
        Gt if signed => return cmp(a.value > b.value),
        Ge if signed => return cmp(a.value >= b.value),
        Lt => return cmp(a.bits() < b.bits()),
        Le => return cmp(a.bits() <= b.bits()),
        Gt => return cmp(a.bits() > b.bits()),
        Ge => return cmp(a.bits() >= b.bits()),
        _ => return None,
    };
    Some(CInt::new(v, &ty))
}

/// What an integer constant expression may refer to beyond literals and
/// operators: named constants, object sizes and the evaluator-specific forms.
pub trait ConstEnv {
    /// The value of identifier `name` (an enumerator or `constexpr` object).
    fn ident(&self, name: &str) -> Option<CInt>;
    /// `sizeof` of a type.
    fn size_of_type(&self, ty: &CType) -> u64;
    /// `_Alignof` of a type.
    fn align_of_type(&self, ty: &CType) -> u64;
    /// `sizeof expr`: the size of the operand's static type.
    fn size_of_expr(&self, e: &Expr) -> Option<u64>;
    /// Any other form this evaluator can fold (`&&label`, `offsetof`'s
    /// address arithmetic), or `None`.
    fn other(&self, _e: &Expr) -> Option<CInt> {
        None
    }
}

/// Fold the integer constant expression `e`, or `None` if it is not one.
pub fn eval(e: &Expr, env: &impl ConstEnv) -> Option<CInt> {
    match &e.kind {
        ExprKind::IntLit(v, ty) => Some(CInt::new(*v, ty)),
        ExprKind::Ident(name) => env.ident(name),
        ExprKind::Unary(op @ (UnaryOp::Neg | UnaryOp::Plus | UnaryOp::BitNot | UnaryOp::LNot), inner) => {
            unary(*op, eval(inner, env)?)
        }
        ExprKind::Binary(op, l, r) => {
            let a = eval(l, env)?;
            // `&&` and `||` do not evaluate an operand that cannot matter.
            match op {
                BinaryOp::LAnd if a.value == 0 => return Some(CInt::int(0)),
                BinaryOp::LOr if a.value != 0 => return Some(CInt::int(1)),
                _ => {}
            }
            binary(*op, a, eval(r, env)?)
        }
        ExprKind::Cond(c, t, f) => {
            let (taken, other) = if eval(c, env)?.value != 0 { (t, f) } else { (f, t) };
            let v = eval(taken, env)?;
            // The result has the arms' common type; the arm not taken need not
            // be a constant (`x ? 1 / x : 0`), and then the taken one's type is
            // used.
            Some(match eval(other, env) {
                Some(o) => CInt::new(v.value, &usual_arith(&v.ty, &o.ty)),
                None => v,
            })
        }
        ExprKind::Cast(ty, inner) => Some(cast(eval(inner, env)?, ty)),
        ExprKind::SizeofType(ty) => Some(CInt::size(env.size_of_type(ty))),
        ExprKind::SizeofExpr(inner) => Some(CInt::size(env.size_of_expr(inner)?)),
        ExprKind::AlignofType(ty) => Some(CInt::size(env.align_of_type(ty))),
        _ => env.other(e),
    }
}

/// Reduce an integer constant `v` to the representable value of integer type
/// `ty`: mask to the type's value-bit count (`N` for a `_BitInt(N)`), then
/// sign-extend for a signed type. `_Bool` normalizes to 0/1. Non-integer types
/// and 128-bit ones (whose every bit pattern is an `i128`) are returned
/// unchanged.
pub fn reduce_const_to_type(v: i128, ty: &CType) -> i128 {
    let (bits, signed) = match ty.unqual() {
        CType::Bool => return i128::from(v != 0),
        CType::Int(i) => (u32::from(i.value_bits()), i.signed),
        _ => return v,
    };
    if bits == 0 || bits >= 128 {
        return v;
    }
    let mask = (1i128 << bits) - 1;
    let m = v & mask;
    if signed && (m & (1i128 << (bits - 1))) != 0 { m - (1i128 << bits) } else { m }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u128_ty() -> CType {
        CType::Int(IntTy::new(128, false))
    }

    fn i128_ty() -> CType {
        CType::Int(IntTy::new(128, true))
    }

    #[test]
    fn wide_values_wrap_without_overflow() {
        let one = CInt::new(1, &u128_ty());
        let top = binary(BinaryOp::Shl, one.clone(), CInt::int(127)).unwrap();
        assert_eq!(top.bits(), 1u128 << 127);
        // `~(unsigned __int128)0 >> 64` is the low-half mask, not -1.
        let all = unary(UnaryOp::BitNot, CInt::new(0, &u128_ty())).unwrap();
        let low = binary(BinaryOp::Shr, all.clone(), CInt::int(64)).unwrap();
        assert_eq!(low.bits(), u128::from(u64::MAX));
        // Unsigned division of the all-ones value.
        let third = binary(BinaryOp::Div, all.clone(), CInt::int(3)).unwrap();
        assert_eq!(third.bits(), u128::MAX / 3);
        // Products and sums wrap at 128 bits.
        let sq = binary(BinaryOp::Mul, all.clone(), all.clone()).unwrap();
        assert_eq!(sq.bits(), 1);
        let min = CInt::new(i128::MIN, &i128_ty());
        assert_eq!(binary(BinaryOp::Div, min.clone(), CInt::int(-1)).unwrap().value, i128::MIN);
        assert_eq!(unary(UnaryOp::Neg, min.clone()).unwrap().value, i128::MIN);
        assert_eq!(binary(BinaryOp::Sub, min, CInt::int(1)).unwrap().value, i128::MAX);
        // Unsigned comparison: the all-ones value is the greatest.
        assert_eq!(binary(BinaryOp::Gt, all, CInt::int(1)).unwrap().value, 1);
        assert!(binary(BinaryOp::Shl, one.clone(), CInt::int(128)).is_none());
        assert!(binary(BinaryOp::Shl, one, CInt::int(-1)).is_none());
    }

    #[test]
    fn narrow_values_follow_their_types() {
        let neg = unary(UnaryOp::Neg, CInt::int(1)).unwrap();
        let u = cast(neg, &CType::uint());
        assert_eq!(u.value, i128::from(u32::MAX));
        assert_eq!(binary(BinaryOp::Shr, u.clone(), CInt::int(1)).unwrap().value, i128::from(i32::MAX));
        // `-1 < 1U` is false: the comparison is unsigned.
        let lt = binary(BinaryOp::Lt, CInt::int(-1), CInt::new(1, &CType::uint())).unwrap();
        assert_eq!(lt.value, 0);
        // `(unsigned char)300` and `(_Bool)2`.
        let uc = CType::Int(IntTy::new(8, false));
        assert_eq!(cast(CInt::int(300), &uc).value, 44);
        assert_eq!(cast(CInt::int(2), &CType::Bool).value, 1);
        // An `int` product wraps to 32 bits.
        let big = binary(BinaryOp::Mul, CInt::int(65536), CInt::int(65536)).unwrap();
        assert_eq!(big.value, 0);
    }
}
