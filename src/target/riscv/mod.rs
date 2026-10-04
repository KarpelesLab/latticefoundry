//! The RISC-V RV64IMAFD backend: register files + the LP64D ABI, the
//! instruction-selection rules, the stack-frame/prologue construction, a
//! from-spec RV64 machine-code encoder, and the ELF object output.
//!
//! This is the framework's **third** CPU target, after x86-64 and AArch64,
//! demonstrating that the code-generation framework is genuinely retargetable:
//! the same reusable lowering driver ([`crate::codegen::isel`]) and linear-scan
//! allocator ([`crate::codegen::regalloc`]) drive all three, and only the ISA
//! details differ. The backend implements both
//! [`crate::codegen::target::MachineTarget`] (register file, ABI, move/spill
//! builders) and [`crate::codegen::isel::TargetIsel`] (per-opcode lowering).
//!
//! Scope: the RV64I base with the M (multiply/divide), A (atomics), F and D
//! (single- and double-precision floating point) extensions; the LP64D
//! calling convention (floats in `fa0`–`fa7`, by-value structs flattened per
//! the psABI, stack-passed arguments, variadic calls); global data and the
//! psABI relocations (`R_RISCV_CALL_PLT`, `R_RISCV_PCREL_HI20`/`LO12_I`,
//! `R_RISCV_64`) in an `EM_RISCV` ELF64 object ([`write_elf`]) that qld
//! links. Deferred (and noted at their sites): the RV64 word forms as a
//! cheaper `i32` lowering, `f16` (Zfh), integers wider than 64 bits, and the
//! callee side of variadic functions.
//!
//! Submodules:
//!
//! - `regs` — the 32 GPRs (`x0`–`x31`, `x0` hardwired zero) and 32 FPRs, the
//!   allocatable/scratch split, and the LP64D register roles;
//! - `abi` — the LP64D argument classification and assignment;
//! - [`isel`] — the [`RvOp`] opcode set and the lowering rules;
//! - [`runtime`] — the green-thread context-switching runtime (save/restore/
//!   switch/init), emitted as machine code for a front end to link in;
//! - [`encode`] — the fixed-width bitfield encoder (R/I/S/B/U/J/R4 formats),
//!   frame layout + prologue/epilogue, relocations, and the
//!   `compile_function`/`compile_module` drivers.
//!
//! Since this host cannot run RISC-V code, the tests validate it three ways:
//! encodings against `llvm-mc`, isel on a MIR interpreter, and the machine
//! code on an instruction-set simulator (its own decoder) linked by a test
//! linker or by qld, differentially against the IR's reference semantics.

pub mod encode;
pub mod isel;
pub mod runtime;
pub(crate) mod regs;
pub(crate) mod abi;

#[cfg(test)]
mod interp;
#[cfg(test)]
mod sim;
#[cfg(test)]
mod diff_tests;
#[cfg(test)]
mod fd_tests;
#[cfg(test)]
mod link_tests;
#[cfg(test)]
mod dyn_tests;
#[cfg(test)]
mod pic_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod stack_tests;
#[cfg(test)]
mod runtime_tests;
#[cfg(test)]
mod vector_tests;

pub use encode::{compile_function, compile_module, compile_module_with};
pub use isel::{RiscvTarget, RvOp};

// ===========================================================================
// ELF object output (RISC-V ELF psABI)
// ===========================================================================

use crate::mc::elf::{ElfClass, ElfTarget, RelocFormat};
use crate::mc::object::RelocKind;

/// The `e_machine` value for RISC-V.
pub const EM_RISCV: u16 = 243;
/// RISC-V `e_flags`: the code uses the compressed (C) extension.
pub const EF_RISCV_RVC: u32 = 0x1;
/// RISC-V `e_flags`: the double-precision hard-float ABI (LP64D), in the
/// `EF_RISCV_FLOAT_ABI` field (bits 1–2).
pub const EF_RISCV_FLOAT_ABI_DOUBLE: u32 = 0x4;

// RISC-V relocation numbers (RISC-V ELF psABI, "Relocations").
const R_RISCV_32: u32 = 1;
const R_RISCV_64: u32 = 2;
const R_RISCV_CALL_PLT: u32 = 19;
const R_RISCV_GOT_HI20: u32 = 20;
const R_RISCV_PCREL_HI20: u32 = 23;
const R_RISCV_PCREL_LO12_I: u32 = 24;
const R_RISCV_PCREL_LO12_S: u32 = 25;
const R_RISCV_32_PCREL: u32 = 57;

/// The RISC-V relocation number of a generic kind: `R_RISCV_64`/`R_RISCV_32`
/// for data, `R_RISCV_32_PCREL`, and the instruction relocations
/// (`R_RISCV_CALL_PLT`, `R_RISCV_PCREL_HI20`, `R_RISCV_PCREL_LO12_I`/`_S`,
/// `R_RISCV_GOT_HI20`). `None` for anything else.
fn elf_reloc_type(kind: RelocKind) -> Option<u32> {
    Some(match kind {
        RelocKind::Abs64 => R_RISCV_64,
        RelocKind::Abs32 => R_RISCV_32,
        RelocKind::Pc32 => R_RISCV_32_PCREL,
        RelocKind::RiscvCallPlt => R_RISCV_CALL_PLT,
        RelocKind::RiscvGotHi20 => R_RISCV_GOT_HI20,
        RelocKind::RiscvPcrelHi20 => R_RISCV_PCREL_HI20,
        RelocKind::RiscvPcrelLo12I => R_RISCV_PCREL_LO12_I,
        RelocKind::RiscvPcrelLo12S => R_RISCV_PCREL_LO12_S,
        _ => return None,
    })
}

/// ELF64, little-endian, `EM_RISCV`, `RELA`, `e_flags` = the LP64D
/// double-float ABI (the code is RV64GC-compatible and uses no compressed
/// instructions).
pub const ELF: ElfTarget = ElfTarget {
    class: ElfClass::Elf64,
    endian: crate::ir::Endian::Little,
    machine: EM_RISCV,
    flags: EF_RISCV_FLOAT_ABI_DOUBLE,
    reloc_format: RelocFormat::Rela,
    reloc_type: elf_reloc_type,
};

/// The startup object of a static Linux executable: `_start` calls `entry`
/// (a `() -> i64` function) and exits with its result (`exit_group`, Linux
/// RISC-V syscall 94). The kernel hands `_start` an aligned `sp`; the code
/// uses no global pointer (`gp`), since nothing is relaxed against it.
pub fn startup_object(entry: &str) -> crate::mc::object::ObjectModule {
    let src = format!(
        "module \"rv-start\"\nfunc @\"{entry}\"() -> i64\nfunc @_start() -> void {{\nentry ^0:\n  \
         %r = call @\"{entry}\"() : i64\n  %x = syscall i64 94, %r : i64\n  unreachable\n}}\n"
    );
    let mut syms = crate::support::StrInterner::new();
    let m = crate::ir::text::parse_module(&src, crate::support::diagnostics::FileId::new(0), &mut syms)
        .expect("the startup module parses");
    compile_module(&m, &syms)
}

/// Link `obj` (and the [`startup_object`] calling `entry`) into a static
/// RISC-V Linux executable at `output`, with qld (`-m elf64lriscv -static`).
/// The objects travel through temporary files next to `output`.
///
/// # Errors
///
/// When an object cannot be written or the link fails.
pub fn link_static_executable(
    obj: &crate::mc::object::ObjectModule,
    entry: &str,
    output: &str,
) -> Result<(), String> {
    let start = startup_object(entry);
    let (tmp, tmp_start) = (format!("{output}.lf-tmp.o"), format!("{output}.lf-start.o"));
    for (path, o) in [(&tmp, obj), (&tmp_start, &start)] {
        let bytes = write_elf(o).map_err(|e| format!("cannot write a RISC-V ELF object: {e}"))?;
        std::fs::write(path, bytes).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    let args = ["-m", "elf64lriscv", "-static", "-e", "_start", "-o", output, &tmp_start, &tmp];
    let result = crate::link::gnu::link_gnu("lf", &args).map_err(|e| format!("link error: {e}"));
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&tmp_start);
    result
}

/// Link `obj` (compiled position-independent) into a RISC-V shared library
/// at `output` with qld ([`crate::link::gnu::shared_library_args`]: `-shared
/// -z text`, an optional `soname`, `extra` arguments such as `-L`/`-l`).
/// There is no RISC-V C runtime on the host, so undefined symbols are left
/// for the dynamic loader.
///
/// # Errors
///
/// When the object cannot be written or the link fails.
pub fn link_shared(
    obj: &crate::mc::object::ObjectModule,
    soname: Option<&str>,
    extra: &[String],
    output: &std::path::Path,
) -> Result<(), String> {
    let tmp = std::path::PathBuf::from(format!("{}.lf-tmp.o", output.display()));
    let bytes = write_elf(obj).map_err(|e| format!("cannot write a RISC-V ELF object: {e}"))?;
    std::fs::write(&tmp, bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    let mut args: Vec<std::ffi::OsString> = vec!["-m".into(), "elf64lriscv".into()];
    args.extend(crate::link::gnu::shared_library_args(None, &[&tmp], soname, extra, output));
    let result = crate::link::gnu::link_gnu("lf", &args).map_err(|e| format!("link error: {e}"));
    let _ = std::fs::remove_file(&tmp);
    result
}

/// Serialize a RISC-V object (from [`compile_module`]) as an ELF64
/// relocatable file for the LP64D ABI.
///
/// # Errors
///
/// A relocation kind RISC-V has no number for.
pub fn write_elf(obj: &crate::mc::object::ObjectModule) -> Result<Vec<u8>, crate::mc::elf::ElfError> {
    crate::mc::elf::write_with(obj, &ELF)
}
