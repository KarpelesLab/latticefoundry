//! Target registry and target-description tables. See ROADMAP Phase 7.
//!
//! Each target contributes its register file, calling conventions, instruction
//! encodings, and lowering rules. Targets register themselves here so drivers
//! can select one by triple.

pub mod aarch64;
pub mod riscv;
pub mod triple;
pub mod x86_64;

pub(crate) mod rt_words;

use crate::codegen::{CodegenOptions, CompiledModule, RelocModel};
use crate::ir::Module;
use crate::support::StrInterner;

#[cfg(test)]
pub(crate) mod atomic_fixtures;

#[doc(inline)]
pub use triple::{CallConvKind, ObjectFormat, TargetOs, Triple};

/// A supported target architecture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TargetArch {
    /// 64-bit x86 (the bring-up target).
    X86_64,
    /// 64-bit ARM.
    AArch64,
    /// 64-bit RISC-V.
    Riscv64,
}

impl TargetArch {
    /// The canonical short name for this architecture.
    pub fn name(self) -> &'static str {
        match self {
            TargetArch::X86_64 => "x86_64",
            TargetArch::AArch64 => "aarch64",
            TargetArch::Riscv64 => "riscv64",
        }
    }
}

impl std::fmt::Display for TargetArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A code-generation request a target cannot honor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodegenError {
    /// The target does not implement the requested relocation model.
    UnsupportedRelocModel {
        /// The target.
        arch: TargetArch,
        /// The requested model.
        model: RelocModel,
    },
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::UnsupportedRelocModel { arch, model } => write!(
                f,
                "the {arch} backend does not generate position-independent code yet \
                 (relocation model {model:?}); only x86_64 supports PIC/PIE"
            ),
        }
    }
}

impl std::error::Error for CodegenError {}

/// Check that `arch`'s backend can honor `opts`. Position-independent code
/// ([`RelocModel::Pie`] / [`RelocModel::Pic`]) is x86-64 only for now: the
/// AArch64 and RISC-V backends would need GOT-indirect address sequences
/// (`adrp`+`ldr` with `R_AARCH64_ADR_GOT_PAGE`/`R_AARCH64_LD64_GOT_LO12_NC`;
/// `auipc`+`ld` with `R_RISCV_GOT_HI20`/`R_RISCV_PCREL_LO12_I`).
///
/// # Errors
///
/// [`CodegenError::UnsupportedRelocModel`] for PIC on a target without it.
pub fn check_options(arch: TargetArch, opts: &CodegenOptions) -> Result<(), CodegenError> {
    let model = opts.reloc_model;
    if model.is_pic() && arch != TargetArch::X86_64 {
        return Err(CodegenError::UnsupportedRelocModel { arch, model });
    }
    Ok(())
}

/// Compile `module` for `arch` under `opts`: the target-generic, fallible entry
/// point over each backend's `compile_module_with`. With
/// [`RelocModel::Pic`] (e.g. `CodegenOptions::default().with_pic(true)`) the
/// result is a position-independent relocatable object ready for a shared
/// library (link it with [`crate::link::gnu::shared_library_args`]).
///
/// # Errors
///
/// When `arch`'s backend cannot honor `opts` (see [`check_options`]).
pub fn compile_module_for(
    arch: TargetArch,
    module: &Module,
    syms: &StrInterner,
    opts: &CodegenOptions,
) -> Result<CompiledModule, CodegenError> {
    check_options(arch, opts)?;
    Ok(match arch {
        TargetArch::X86_64 => x86_64::compile_module_with(module, syms, opts),
        TargetArch::AArch64 => aarch64::compile_module_with(module, syms, opts),
        TargetArch::Riscv64 => riscv::compile_module_with(module, syms, opts),
    })
}
