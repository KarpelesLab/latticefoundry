//! Target-independent code generation.
//!
//! Lowers the SSA [`ir`](crate::ir) to a machine-level IR (MIR), then runs
//! instruction selection and register allocation to produce target machine
//! instructions ready for the [`mc`](crate::mc) layer (ROADMAP Phase 5). The
//! layer is split into:
//!
//! - [`mir`] — the target-abstract MIR data model (registers, operands, blocks,
//!   functions, stack frame);
//! - [`target`] — the [`MachineTarget`] interface a
//!   backend implements to describe its register file, calling convention, and
//!   move/spill builders, *without* committing to encodings;
//! - [`isel`] — the reusable instruction-selection framework (block-argument
//!   lowering, the ABI seam, constant materialization) parameterized by a
//!   target's per-opcode rules ([`isel::TargetIsel`]);
//! - [`vtarget`] — an abstract RISC-like virtual target that implements both
//!   traits, so the framework can be exercised end to end without a real ISA;
//! - [`regalloc`] — a correct linear-scan register allocator with spilling;
//! - [`interp`] — a small MIR interpreter over the virtual target, the
//!   executable semantics isel + regalloc are validated against;
//! - [`data`] — the target-independent emission of global data (initializer
//!   serialization, `.rodata`/`.data`/`.bss` placement, data relocations) that
//!   each backend's `compile_module` calls with its absolute-pointer reloc kind;
//! - [`legalize_int`] — wide-integer legalization: splits integer operations
//!   wider than a target's native width into part-width operations (and
//!   libcalls), the shared IR-to-IR step before isel on 32-, 16- and 8-bit
//!   targets;
//! - [`linkage`] — symbol binding decisions shared by the backends: which
//!   symbols bind locally under a [`RelocModel`] (direct vs. GOT addressing),
//!   and the IR linkage/visibility applied to the object's symbols;
//! - [`options`] — the [`CodegenOptions`] knobs (stack probes, relocation
//!   model) and the
//!   [`CompiledModule`] result of the backends' `compile_module_with`;
//! - [`stack`] — per-function [`StackUsage`] reports taken
//!   from the frame layouts, worst-case stack-depth analysis over the call
//!   graph, and the stack-probing contract.
//!
//! Real ISAs (x86-64, AArch64, RISC-V) and instruction *encoding* are Phases
//! 6–7; this phase produces MIR, not bytes.

pub mod data;
pub mod interp;
pub mod isel;
pub mod legalize;
pub mod legalize_int;
pub mod linkage;
pub mod mir;
pub mod options;
pub mod regalloc;
pub mod softfloat;
pub(crate) mod simd128;
pub mod stack;
pub mod target;
pub mod unwind;
pub mod vtarget;

/// The diagnostic every backend without an inline-asm lowering gives for an
/// `inline_asm` (`docs/ir-design.md` §6j): only x86-64 assembles templates.
pub const INLINE_ASM_UNSUPPORTED: &str = "inline asm is not supported on this target";

/// The first function of `module` that contains an `inline_asm`, with its
/// name (for "not supported on this target" diagnostics).
pub fn first_inline_asm(module: &crate::ir::Module, syms: &crate::support::StrInterner) -> Option<String> {
    module.functions().find_map(|f| {
        let has = f.blocks().any(|(_, b)| {
            b.insts().iter().any(|&i| matches!(f.inst(i).kind, crate::ir::InstKind::InlineAsm(_)))
        });
        has.then(|| syms.resolve(f.name).to_owned())
    })
}

pub use mir::MachineFunction;
pub use options::{CodegenOptions, CompiledModule, RelocModel};
pub use unwind::UnwindTables;
pub use stack::{
    StackAnalysis, StackAssumptions, StackBlocker, StackBound, StackBoundError, StackReport, StackUsage,
};
pub use target::MachineTarget;
pub use vtarget::VirtualTarget;

#[cfg(test)]
mod tests;
