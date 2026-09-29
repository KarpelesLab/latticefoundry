//! The 32-bit Arm **Thumb-2** backend for Cortex-M (ARMv7-M: Cortex-M3; also
//! ARMv7E-M: Cortex-M4/M7, whose DSP and FPU extensions it does not use),
//! with the AAPCS base (soft-float) procedure call standard.
//!
//! The pipeline is the framework's usual one — the reusable lowering driver
//! ([`crate::codegen::isel`]) and linear-scan allocator
//! ([`crate::codegen::regalloc`]) drive a target that implements
//! [`crate::codegen::target::MachineTarget`] and
//! [`crate::codegen::isel::TargetIsel`] — preceded by two IR-to-IR steps a
//! 32-bit core without an FPU needs ([`prepare_module`]), after vector code
//! has been scalarized ([`crate::codegen::legalize`]; there are no vector
//! registers):
//!
//! 1. **soft-float** ([`softfloat`]): floating-point values become the
//!    integers holding their bits, and every floating-point operation a call to
//!    a helper of the *Run-time ABI for the Arm Architecture* (`__aeabi_fadd`,
//!    `__aeabi_dmul`, `__aeabi_fcmplt`, `__aeabi_f2iz`, `__aeabi_i2d`, …). That
//!    is the AAPCS base standard: a `float` travels like an `i32`, a `double`
//!    like an `i64`;
//! 2. **64-bit legalization** ([`crate::codegen::legalize_int`] at `W = 32`):
//!    every integer wider than 32 bits is split into words; `mul`, `sdiv`,
//!    `udiv` and the remainders of 64-bit values call `__aeabi_lmul` and
//!    `__aeabi_ldivmod` / `__aeabi_uldivmod`.
//!
//! What remains wide (the ABI boundary: `i64` parameters, arguments, results
//! and returns, and the legalizer's split/join shapes) lives in register pairs
//! in [`isel`].
//!
//! # The target
//!
//! - **Data layout** ([`data_layout`]): ILP32, little-endian, natural
//!   alignment up to 8 bytes (so `i64`/`double` are 8-aligned, as the AAPCS
//!   requires), an 8-byte stack, native `i8`..`i32`.
//! - **Registers** (the private `regs` module): `r0`–`r12`, `sp`, `lr`, `pc`; `r0`–`r3` carry
//!   arguments and results, `r4`–`r11` are callee-saved.
//! - **AAPCS** ([`isel`]): `r0`–`r3` then the stack, doubleword alignment for
//!   64-bit values (even register pairs, 8-aligned stack slots), composites
//!   copied into registers and split across `r3` and the stack, results in
//!   `r0` / `r0:r1` / memory through a hidden pointer in `r0`, an 8-byte
//!   aligned stack at every call.
//! - **Encoding** ([`encode`]): 16-bit forms whenever the registers and
//!   immediate allow, 32-bit Thumb-2 otherwise; `IT` blocks for compare-and-set
//!   and select; `movw`/`movt` for constants and addresses (no literal pools);
//!   relaxed branches; `sdiv`/`udiv` (or the AEABI division helpers with
//!   [`ThumbOptions::with_hw_div`] off).
//! - **Objects**: ELF32, `EM_ARM`, EABI version 5 with the soft-float flag,
//!   `REL` relocations with implicit addends — `R_ARM_THM_CALL` for `bl`,
//!   `R_ARM_THM_MOVW_ABS_NC` / `R_ARM_THM_MOVT_ABS` for addresses,
//!   `R_ARM_ABS32` for data ([`crate::mc::elf::ElfTarget::ARM`]). Function
//!   symbols carry the Thumb bit and `.text` starts with a `$t` mapping
//!   symbol.
//! - **Stack usage and probes** as on the other targets
//!   ([`crate::codegen::stack`]): the report is read off the frame layout
//!   (`push {r4.., lr}` plus `sub sp`); probes are emitted for large frames
//!   (a Cortex-M has no guard page unless its MPU provides one, so they only
//!   help there, but cost nothing otherwise).
//! - **Firmware** ([`firmware`]): a Cortex-M vector table and reset handler
//!   generated for an entry symbol, a linker script, and a link through `qld`
//!   into an ELF executable whose segments `lf build --oformat binary|ihex`
//!   turns into a flashable image.
//!
//! # The runtime a program links with
//!
//! Soft-float, 64-bit multiply/divide, and (without hardware divide) 32-bit
//! division are calls to the standard RTABI helpers, which every Arm
//! toolchain's runtime library provides (libgcc's `libgcc.a` for
//! `arm-none-eabi`, LLVM's compiler-rt `builtins`); `frem` calls the C
//! library's `fmodf`/`fmod`. Link one of them (`lf build … -lgcc -L<dir>`).
//!
//! # Not provided (documented choices)
//!
//! - **Hard float (FPv4-SP, Cortex-M4F).** The FPv4-SP extension has 32
//!   single-precision registers `s0`–`s31` (aliased as 16 `d` registers for
//!   loads and stores only) and no double-precision arithmetic. A hard-float
//!   variant would keep `f32` in an `Fp` register class (`vadd.f32`,
//!   `vmul.f32`, `vdiv.f32`, `vcmp.f32` + `vmrs APSR_nzcv, fpscr`,
//!   `vcvt.s32.f32`, `vldr`/`vstr`), still lower every `f64` operation to the
//!   `__aeabi_d*` helpers, and follow the AAPCS **VFP variant**: float
//!   arguments in `s0`–`s15` (back-filling), results in `s0`, `s16`–`s31`
//!   callee-saved, `EF_ARM_ABI_FLOAT_HARD` in `e_flags`. Code built that way
//!   does not link with soft-float code. Only the soft-float ABI is
//!   implemented, and it runs unchanged on a Cortex-M4F.
//! - **ARMv6-M (Cortex-M0/M0+/M1).** Thumb-1 plus a handful of 32-bit
//!   instructions (`bl`, `dmb`, `mrs`/`msr`): no `IT`, no `movw`/`movt`, no
//!   `sdiv`/`udiv`, no 32-bit data-processing forms, only `r0`–`r7` for most
//!   operations. This backend's encoder uses all of those, so `thumbv6m` is
//!   not accepted; [`ThumbOptions::with_hw_div`] (division through
//!   `__aeabi_idiv`) is the one piece that already carries over.
//! - **The green-thread context runtime** of the other targets (save/restore
//!   and switch as machine code) is not provided: on Cortex-M, context
//!   switching is the PendSV exception's job, whose hardware-stacked frame
//!   (`r0`–`r3`, `r12`, `lr`, `pc`, `xPSR`) plus a software `push {r4-r11}`
//!   is the save area.
//! - **DWARF**, `dyn_alloca`, atomic read-modify-write / compare-exchange
//!   (`ldrex`/`strex` loops) and position-independent code.
//!
//! Implemented from the ARMv7-M Architecture Reference Manual, the AAPCS, the
//! Run-time ABI and *ELF for the Arm Architecture*.

pub mod encode;
pub mod firmware;
pub mod isel;
pub(crate) mod regs;
#[doc(no_inline)]
pub use crate::codegen::softfloat;

#[cfg(test)]
mod interp;
#[cfg(test)]
mod sim;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod diff_tests;

pub use encode::{
    ThumbOptions, compile_function, compile_module, compile_module_thumb, compile_module_with,
};
pub use isel::{ThOp, ThumbTarget};

use crate::codegen::legalize_int::{LegalizeError, LegalizeOptions, legalize_ints, libgcc_libcall};
use crate::ir::inst::{BinOp, InstKind};
use crate::ir::types::Type;
use crate::ir::{DataLayout, FuncId, Module};
use crate::support::StrInterner;

/// The Thumb data layout: ILP32 little-endian with the AAPCS alignments
/// (`e-p:32:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-S64-n8:16:32`).
pub fn data_layout() -> DataLayout {
    DataLayout::ilp32()
}

/// The name of the helper implementing a wide `op` on `bits`-bit integers:
/// the RTABI's `__aeabi_lmul`, `__aeabi_ldivmod` and `__aeabi_uldivmod` for
/// 64 bits (the remainders are placeholders the isel turns into the divmod
/// helpers, see [`isel::LMOD_PSEUDO`]), libgcc's names otherwise.
pub fn aeabi_libcall(op: BinOp, bits: u32) -> String {
    match (op, bits) {
        (BinOp::Mul, 64) => "__aeabi_lmul".into(),
        (BinOp::SDiv, 64) => "__aeabi_ldivmod".into(),
        (BinOp::UDiv, 64) => "__aeabi_uldivmod".into(),
        (BinOp::SRem, 64) => isel::LMOD_PSEUDO.into(),
        (BinOp::URem, 64) => isel::ULMOD_PSEUDO.into(),
        _ => libgcc_libcall(op, bits),
    }
}

/// Why a module cannot be compiled for Thumb.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PrepareError {
    /// The module could not be copied (it does not round-trip through the
    /// binary IR form).
    Copy(String),
    /// Soft-float lowering failed.
    SoftFloat(softfloat::SoftFloatError),
    /// Wide-integer legalization failed.
    Legalize(LegalizeError),
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrepareError::Copy(e) => write!(f, "cannot copy the module: {e}"),
            PrepareError::SoftFloat(e) => write!(f, "soft-float lowering: {e}"),
            PrepareError::Legalize(e) => write!(f, "64-bit legalization: {e}"),
        }
    }
}

impl std::error::Error for PrepareError {}

/// Prepare a copy of `module` for Thumb instruction selection: give it the
/// Thumb [`data_layout`] (a module compiled for this target should already
/// carry it; one with the default LP64 layout is re-laid out, which suits code
/// that does not bake LP64 offsets into constants), scalarize every vector
/// ([`crate::codegen::legalize`] with no legal vector type, as on RISC-V),
/// lower floating point to soft-float calls ([`softfloat`]), legalize integers
/// wider than 32 bits
/// ([`legalize_ints`], with [`aeabi_libcall`] names), and declare the division
/// helpers the isel calls directly. Returns the new module with its own
/// interner. Function attributes (linkage, visibility, secrecy) are preserved.
///
/// # Errors
///
/// A [`PrepareError`] when a step cannot handle the module.
pub fn prepare_module(
    module: &Module,
    syms: &StrInterner,
    topts: &ThumbOptions,
) -> Result<(Module, StrInterner), PrepareError> {
    let bytes = crate::ir::binary::encode(module, syms);
    let mut s = StrInterner::new();
    let mut m = crate::ir::binary::decode(&bytes, &mut s).map_err(|e| PrepareError::Copy(e.to_string()))?;
    if m.data_layout() != &data_layout() {
        m.set_data_layout(data_layout());
    }
    // Vectors first (Thumb has no vector registers: everything is scalarized,
    // min/max and saturating ops expanded), so the soft-float pass sees the
    // scalarized float lanes and legalization the scalarized wide integers.
    crate::codegen::legalize::legalize_vectors(&mut m, &crate::codegen::legalize::ScalarOnly);
    softfloat::lower_soft_float(&mut m, &mut s, softfloat::SoftFloatAbi::Aeabi).map_err(PrepareError::SoftFloat)?;
    let opts = LegalizeOptions { part_bits: 32, libcall_name: aeabi_libcall };
    legalize_ints(&mut m, &mut s, &opts).map_err(PrepareError::Legalize)?;

    // The helpers the isel calls by index.
    let has = |m: &Module, s: &StrInterner, name: &str| m.functions().any(|f| s.resolve(f.name) == name);
    let mut need: Vec<&str> = Vec::new();
    if has(&m, &s, isel::LMOD_PSEUDO) {
        need.push("__aeabi_ldivmod");
    }
    if has(&m, &s, isel::ULMOD_PSEUDO) {
        need.push("__aeabi_uldivmod");
    }
    let (i64t, i32t) = (m.types_mut().int(64), m.types_mut().int(32));
    let wide_sig = m.types_mut().func(vec![i64t, i64t], i64t, false);
    let word_sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
    for name in need {
        if !has(&m, &s, name) {
            m.declare_function(s.intern(name), wide_sig);
        }
    }
    if !topts.hw_div && narrow_division(&m) {
        for name in ["__aeabi_idiv", "__aeabi_uidiv", "__aeabi_idivmod", "__aeabi_uidivmod"] {
            if !has(&m, &s, name) {
                m.declare_function(s.intern(name), word_sig);
            }
        }
    }
    Ok((m, s))
}

/// Whether any function divides or takes a remainder of a value of at most 32
/// bits.
fn narrow_division(m: &Module) -> bool {
    (0..m.function_count()).any(|i| {
        let f = m.function(FuncId::from_index(i));
        f.blocks().any(|(_, b)| {
            b.insts().iter().any(|&id| {
                let inst = f.inst(id);
                matches!(inst.kind, InstKind::Bin(BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem))
                    && matches!(m.types().get(inst.ty), Type::Int(w) if *w <= 32)
            })
        })
    })
}
