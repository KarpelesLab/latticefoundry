//! The x86-64 backend (ROADMAP Phase 7): register file + System V ABI, the
//! integer instruction-selection rules, the stack-frame/prologue construction,
//! and a from-spec machine-code encoder that produces relocatable ELF64 objects.
//!
//! The backend plugs into the Phase-5 code-generation framework — it implements
//! both [`crate::codegen::target::MachineTarget`] (register file, ABI,
//! move/spill builders) and [`crate::codegen::isel::TargetIsel`] (per-opcode
//! lowering) — and emits into the Phase-6 machine-code layer
//! ([`crate::mc::emit`] / [`crate::mc::object`] / [`crate::mc::elf`]).
//!
//! Scope covers the integer subset sufficient to compile and *run* real
//! functions — arithmetic/bitwise/shift/divide, comparisons and branches,
//! loads/stores and `alloca`, and calls under the SysV ABI — plus **scalar SSE
//! floating-point** (F32/F64): `addsd`/…/`divsd` and the `ss` forms, `fneg` via
//! a sign-bit `xorpd`, `fcmp` via `ucomis`+`setcc` (with the ordered/unordered
//! parity fixup), the `cvt*` conversions, `movsd`/`movss` loads/stores/spills,
//! and float argument/return passing in `xmm0..xmm7`/`xmm0`. F16 is widened or
//! deferred (it is not an x86 scalar type); wider SIMD and AArch64 FP are
//! follow-ups.
//!
//! Calls follow System V by default, or the Microsoft x64 convention for
//! Windows targets (`CodegenOptions::with_os(TargetOs::Windows)`; see
//! [`isel`]).
//!
//! Submodules:
//!
//! - `regs` — the 16 GPRs as physical registers, the allocatable/scratch split,
//!   and the SysV calling convention;
//! - [`isel`] — the [`X86Op`] opcode set and the lowering rules, including
//!   the register pairs of 128-bit integers ([`prepare_module`] splits every
//!   other wide operation into 64-bit parts first);
//! - [`runtime`] — the green-thread context-switching runtime (save/restore/
//!   switch, signal-`ucontext` mapping, `rt_sigaction` + restorer), emitted as
//!   machine code for a front end to link in;
//! - [`encode`] — the REX/ModRM/SIB encoder, frame layout + prologue/epilogue,
//!   and the `compile_function`/`compile_module` drivers.

pub mod encode;
pub mod isel;
#[cfg(test)]
mod wide_tests;
pub mod runtime;
pub(crate) mod regs;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod stack_tests;
#[cfg(test)]
mod pic_tests;
#[cfg(test)]
mod tls_tests;
#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod syscall_tests;
#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod data_tests;
#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod atomic_tests;
#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod runtime_tests;
#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod win64_tests;
#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod vector_tests;
#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod asm_tests;

pub use encode::{
    DebugSource, compile_function, compile_module, compile_module_debug, compile_module_debug_with,
    compile_module_with, compile_to_elf,
};
pub use isel::{MUL128_PSEUDO, Sse2Legality, X86Op, X86_64Target, check_inline_asm};

use crate::codegen::legalize_int::{LegalizeError, LegalizeOptions, legalize_ints, libgcc_libcall};
use crate::ir::inst::{BinOp, CastOp, InstKind};
use crate::ir::types::Type;
use crate::ir::{Module, ValueId};
use crate::support::StrInterner;

/// The name of the helper implementing a wide `op` on `bits`-bit integers on
/// x86-64: [`MUL128_PSEUDO`] for a 128-bit `mul` (expanded inline by
/// instruction selection), libgcc's names otherwise (`__divti3`, `__udivti3`,
/// `__modti3`, `__umodti3`).
pub fn x86_64_libcall(op: BinOp, bits: u32) -> String {
    match (op, bits) {
        (BinOp::Mul, 128) => MUL128_PSEUDO.into(),
        _ => libgcc_libcall(op, bits),
    }
}

/// Whether some function of `module` computes with an integer wider than 64
/// bits.
fn has_wide_ints(module: &Module) -> bool {
    module.functions().any(|f| {
        (0..f.value_count())
            .any(|v| matches!(module.types().get(f.value_type(ValueId::from_index(v))), Type::Int(b) if *b > 64))
    })
}

/// Prepare a copy of `module` for x86-64 instruction selection of integers
/// wider than 64 bits (`docs/ir-design.md` §3b): declare the libgcc helpers
/// that convert between `i128` and floats (`__floattidf`, `__fixunssfti`, …)
/// for the conversions the module makes, then split every wide integer into
/// 64-bit parts ([`legalize_ints`] at `W = 64`, with [`x86_64_libcall`]
/// names), leaving only the ABI boundary wide for the backend's register
/// pairs. Returns the new module with its own interner; the compile entry
/// points do this themselves whenever a module needs it.
///
/// # Errors
///
/// A [`LegalizeError`] for a width the split cannot handle (one that is not
/// a multiple of 64).
pub fn prepare_module(module: &Module, syms: &StrInterner) -> Result<(Module, StrInterner), LegalizeError> {
    let bytes = crate::ir::binary::encode(module, syms);
    let mut names = StrInterner::new();
    let mut m = crate::ir::binary::decode(&bytes, &mut names).expect("a module round-trips through .lfb");
    // Min/max and saturating operations become compares and selects first.
    crate::codegen::legalize::legalize_vectors(&mut m, &Sse2Legality);
    declare_float_helpers(&mut m, &mut names);
    legalize_ints(&mut m, &mut names, &LegalizeOptions { part_bits: 64, libcall_name: x86_64_libcall })?;
    // The multiply placeholder is expanded inline into constant-time `mul` and
    // `imul`: it takes secrets and its result carries their taint (§6d).
    let pseudo = m.functions().position(|f| f.is_declaration() && names.resolve(f.name) == MUL128_PSEUDO);
    if let Some(i) = pseudo {
        let f = crate::ir::FuncId::from_index(i);
        let mut attrs = m.function(f).attrs.clone();
        attrs.set_param_secret(0, true);
        attrs.set_param_secret(1, true);
        attrs.secret_ret = true;
        m.set_func_attrs(f, attrs);
    }
    Ok((m, names))
}

/// The prepared copy of `module` when it has integers wider than 64 bits.
///
/// # Panics
///
/// When [`prepare_module`] fails: the backend cannot compile such a width.
pub(crate) fn prepared_if_wide(module: &Module, syms: &StrInterner) -> Option<(Module, StrInterner)> {
    has_wide_ints(module).then(|| {
        prepare_module(module, syms).unwrap_or_else(|e| panic!("x86-64 backend: wide-integer legalization: {e}"))
    })
}

/// Declare `(i128) -> float` / `(float) -> i128` helpers for every wide
/// integer/float conversion in `module` (see [`isel::float_helper`]).
fn declare_float_helpers(module: &mut Module, names: &mut StrInterner) {
    let mut need: Vec<(&'static str, crate::ir::types::TypeId, crate::ir::types::TypeId)> = Vec::new();
    for f in module.functions() {
        for (_, b) in f.blocks() {
            for &i in b.insts() {
                let inst = f.inst(i);
                let InstKind::Cast(op) = inst.kind else { continue };
                let src = f.value_type(inst.operands()[0]);
                let (int_ty, float_ty) = match op {
                    CastOp::SiToFp | CastOp::UiToFp => (src, inst.ty),
                    CastOp::FpToSi | CastOp::FpToUi => (inst.ty, src),
                    _ => continue,
                };
                if !matches!(module.types().get(int_ty), Type::Int(b) if *b > 64) {
                    continue;
                }
                let fbits = module.types().bit_width(float_ty).unwrap_or(64);
                let Some(name) = isel::float_helper(op, fbits) else { continue };
                let sig = if matches!(op, CastOp::SiToFp | CastOp::UiToFp) { (int_ty, float_ty) } else { (float_ty, int_ty) };
                if !need.iter().any(|n| n.0 == name) {
                    need.push((name, sig.0, sig.1));
                }
            }
        }
    }
    for (name, param, ret) in need {
        if module.functions().any(|f| names.resolve(f.name) == name) {
            continue;
        }
        let sig = module.types_mut().func(vec![param], ret, false);
        module.declare_function(names.intern(name), sig);
    }
}
