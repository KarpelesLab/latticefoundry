//! Helpers shared by the backends that hold 128-bit vectors in SIMD registers
//! (x86-64 SSE2, AArch64 NEON): the legal type set, the register image of a
//! vector constant, and uniform-constant detection (`docs/ir-design.md` §6c).
//!
//! Both backends keep the same six types — `<16 x i8>`, `<8 x i16>`,
//! `<4 x i32>`, `<2 x i64>`, `<4 x f32>`, `<2 x f64>` — and the masks
//! `<16/8/4/2 x i1>`, a mask `<N x i1>` occupying the register as `N` lanes of
//! `128 / N` bits each, all-ones (true) or all-zeros (false): the form their
//! compare instructions produce and their bitwise blends consume.

use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ConstPool, FloatBits, ValueDef, ValueId};
use crate::ir::Function;

/// The shape of a legal 128-bit vector type: the lane kind and count.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Lanes {
    /// Integer lanes of this width.
    Int(u32),
    /// Float lanes of this width (32 or 64).
    Float(u32),
    /// Mask lanes (`i1`); each occupies `128 / n` bits.
    Mask,
}

/// The `(kind, lane count, container bits per lane)` of a legal vector type.
pub(crate) fn shape(types: &TypeContext, ty: TypeId) -> Option<(Lanes, u32, u32)> {
    let (elem, n) = types.vector_parts(ty)?;
    let kind = match (types.get(elem), n) {
        (Type::Int(8), 16) | (Type::Int(16), 8) | (Type::Int(32), 4) | (Type::Int(64), 2) => {
            Lanes::Int(types.bit_width(elem).expect("an int"))
        }
        (Type::Float(k), 4) if k.bit_width() == 32 => Lanes::Float(32),
        (Type::Float(k), 2) if k.bit_width() == 64 => Lanes::Float(64),
        (Type::Int(1), 2 | 4 | 8 | 16) => Lanes::Mask,
        _ => return None,
    };
    Some((kind, n, 128 / n))
}

/// A constant vector operand whose defined lanes are all the same integer
/// (poison lanes allow any value): that integer.
pub(crate) fn uniform_const(consts: &ConstPool, func: &Function, v: ValueId) -> Option<u64> {
    let ValueDef::Const(c) = func.value(v).def else {
        return None;
    };
    match consts.get(c) {
        Const::Poison(_) => Some(0),
        Const::Aggregate { elems, .. } => {
            let mut out: Option<u64> = None;
            for &e in elems {
                match consts.get(e) {
                    Const::Poison(_) => {}
                    Const::Int { value, .. } => {
                        let x = value.to_u64().unwrap_or(u64::MAX);
                        if out.is_some_and(|o| o != x) {
                            return None;
                        }
                        out = Some(x);
                    }
                    _ => return None,
                }
            }
            Some(out.unwrap_or(0))
        }
        _ => None,
    }
}

/// The 128-bit image of a vector constant as `(low, high)` quadwords, in the
/// register convention (a true mask lane is all-ones across its container;
/// poison lanes are zero).
pub(crate) fn const_bits(types: &TypeContext, consts: &ConstPool, c: &Const) -> (u64, u64) {
    let Some((_, n, cw)) = shape(types, c.type_id()) else {
        return (0, 0);
    };
    let Const::Aggregate { elems, .. } = c else {
        return (0, 0);
    };
    let mut bytes = [0u8; 16];
    let per = (cw / 8) as usize;
    for (i, &e) in elems.iter().enumerate().take(n as usize) {
        let raw: u64 = match consts.get(e) {
            Const::Int { ty, value } => {
                let w = types.bit_width(*ty).unwrap_or(64);
                let bits = value.mod_2k(w).to_u64().unwrap_or(0);
                if w == 1 { if bits != 0 { u64::MAX } else { 0 } } else { bits }
            }
            Const::Float { bits: FloatBits::F32(b), .. } => u64::from(*b),
            Const::Float { bits: FloatBits::F64(b), .. } => *b,
            Const::Float { bits: FloatBits::F16(b), .. } => u64::from(*b),
            _ => 0,
        };
        for k in 0..per {
            bytes[i * per + k] = (raw >> (8 * k)) as u8;
        }
    }
    let lo = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"));
    let hi = u64::from_le_bytes(bytes[8..].try_into().expect("8 bytes"));
    (lo, hi)
}
