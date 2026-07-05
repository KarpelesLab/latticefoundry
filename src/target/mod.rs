//! Target registry and target-description tables. See ROADMAP Phase 7.
//!
//! Each target contributes its register file, calling conventions, instruction
//! encodings, and lowering rules. Targets register themselves here so drivers
//! can select one by triple.

pub mod aarch64;
pub mod x86_64;

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
