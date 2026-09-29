//! Preparing a module for AVR instruction selection.
//!
//! The backend compiles a private copy of the module, rewritten by the
//! target-independent passes, in this order:
//!
//! 1. **Data layout.** A module that still has the default LP64 layout is given
//!    the AVR layout with functions in space 0 ([`super::data_layout_p0`]: its
//!    function references were typed that way); one with another layout must
//!    already agree with AVR's on the pointer widths.
//! 2. **Vector legalization** ([`crate::codegen::legalize`] with no legal
//!    vector type): every vector operation is scalarized.
//! 3. **Soft float** ([`crate::codegen::softfloat`], libgcc helper names with a
//!    16-bit `int`): `f32`/`f64` become their bit patterns — signatures
//!    included, which is AVR's soft-float calling convention — and the
//!    operations runtime calls.
//! 4. **Integer legalization** ([`legalize_ints`]) at a part width of 16:
//!    every integer wider than 16 bits becomes 16-bit parts, and a wide
//!    `mul`/`udiv`/`sdiv`/`urem`/`srem` a call to `__mulsi3`, `__udivdi3`, ...
//!
//! It then declares the helpers that isel calls for the 8- and 16-bit
//! `udiv`/`sdiv`/`urem`/`srem` (always: AVR has no divider) and `mul` (on a
//! core without the multiplier): `__lf_udiv_i16`, `__lf_mul_i8`, ... — the
//! names [`libgcc_libcall`] gives those widths, each `T f(T, T)` under the
//! ordinary calling convention. The runtime ([`super::runtime`]) defines
//! them all.

use crate::codegen::legalize_int::{LegalizeOptions, legalize_ints, libgcc_libcall};
use crate::ir::inst::{BinOp, InstKind};
use crate::ir::types::Type;
use crate::ir::{DataLayout, FuncId, Module};
use crate::support::{DetHashMap, StrInterner};

use super::isel::container;

/// The helper functions isel calls for narrow operations, by `(op, container
/// width)`, as function indices of the prepared module.
pub(crate) type Helpers = DetHashMap<(BinOp, u32), u32>;

/// The soft-float helper names: libgcc's, with AVR's 16-bit `int` as the
/// comparison helpers' result.
pub(crate) const SOFT_FLOAT: crate::codegen::softfloat::SoftFloatAbi =
    crate::codegen::softfloat::SoftFloatAbi::Libgcc { int_bits: 16 };

/// A deep copy of `module` (through the binary form) with a fresh interner.
pub(crate) fn copy_module(module: &Module, syms: &StrInterner) -> Result<(Module, StrInterner), String> {
    let bytes = crate::ir::binary::encode(module, syms);
    let mut s = StrInterner::new();
    let m = crate::ir::binary::decode(&bytes, &mut s).map_err(|e| format!("cannot copy the module: {e}"))?;
    Ok((m, s))
}

/// Prepare a copy of `module` for `device` (see the [module docs](self)).
///
/// # Errors
///
/// A module whose layout disagrees with AVR's, `f16` values, or an integer
/// width legalization cannot split.
pub(crate) fn prepare(
    module: &Module,
    syms: &StrInterner,
    device: &super::Device,
) -> Result<(Module, StrInterner, Helpers), String> {
    let (mut m, mut s) = copy_module(module, syms)?;
    let avr = super::data_layout();
    if *m.data_layout() == DataLayout::lp64() {
        m.set_data_layout(super::data_layout_p0());
    } else {
        let dl = m.data_layout();
        if dl.pointer_bits(0) != 16 || dl.pointers().iter().any(|&(_, p)| p.bits != 16) {
            return Err(format!(
                "the module's data layout `{}` has pointers that are not 16 bits (AVR: `{}`)",
                dl.to_spec(),
                avr.to_spec()
            ));
        }
    }
    // Vectors first (AVR has no vector registers: everything is scalarized),
    // so the soft-float pass sees scalar float lanes and legalization scalar
    // wide integers.
    crate::codegen::legalize::legalize_vectors(&mut m, &crate::codegen::legalize::ScalarOnly);
    crate::codegen::softfloat::lower_soft_float(&mut m, &mut s, SOFT_FLOAT).map_err(|e| e.to_string())?;
    legalize_ints(&mut m, &mut s, &LegalizeOptions::new(16)).map_err(|e| e.to_string())?;
    let helpers = declare_helpers(&mut m, &mut s, device.has_mul);
    Ok((m, s, helpers))
}

/// Declare (or find) the narrow-operation helpers the module needs.
fn declare_helpers(m: &mut Module, s: &mut StrInterner, has_mul: bool) -> Helpers {
    let mut needed: Vec<(BinOp, u32)> = Vec::new();
    for fi in 0..m.function_count() {
        let f = m.function(FuncId::from_index(fi));
        for (_, b) in f.blocks() {
            for &i in b.insts() {
                let inst = f.inst(i);
                let InstKind::Bin(op) = inst.kind else { continue };
                let &Type::Int(bits) = m.types().get(inst.ty) else { continue };
                let wanted = matches!(op, BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem)
                    || (op == BinOp::Mul && !has_mul);
                if wanted && bits <= 16 && !needed.contains(&(op, container(bits))) {
                    needed.push((op, container(bits)));
                }
            }
        }
    }
    let mut out = Helpers::default();
    for (op, cw) in needed {
        let name = libgcc_libcall(op, cw);
        let existing = (0..m.function_count()).map(FuncId::from_index).find(|&f| s.resolve(m.function(f).name) == name);
        let fid = existing.unwrap_or_else(|| {
            let t = m.types_mut().int(cw);
            let sig = m.types_mut().func(vec![t, t], t, false);
            m.declare_function(s.intern(&name), sig)
        });
        out.insert((op, cw), fid.index() as u32);
    }
    out
}
