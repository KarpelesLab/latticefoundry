//! Code-generation options shared by every backend, and the result type of the
//! option-taking compile entry points.
//!
//! Each target's `compile_module` keeps its original signature (default options,
//! returning just the [`ObjectModule`]); the `compile_module_with` siblings take
//! a [`CodegenOptions`] and return a [`CompiledModule`], which carries the object
//! *and* the per-function [`StackReport`] computed from the same frame layouts
//! the prologues were built from.

use crate::codegen::stack::StackReport;
use crate::codegen::unwind::UnwindTables;
use crate::mc::object::ObjectModule;
use crate::target::TargetOs;

/// Target-independent code-generation options.
///
/// Construct with [`CodegenOptions::default`] (the safe defaults) and adjust with
/// the builder methods; the struct is `#[non_exhaustive]` so new knobs can be
/// added without breaking callers.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CodegenOptions {
    /// Emit **stack probes** (default `true`). With probes on, every function
    /// whose stack-pointer adjustment is at least
    /// [`STACK_PROBE_INTERVAL`](crate::codegen::stack::STACK_PROBE_INTERVAL)
    /// bytes moves the stack pointer one interval at a time and touches each
    /// step, and every `dyn_alloca` does the same at run time, so a stack
    /// overflow always faults on the guard page (a deterministic `SIGSEGV`)
    /// instead of silently jumping past it into other memory. See
    /// [`crate::codegen::stack`] for the exact invariant and the per-target
    /// sequences. Turn it off only where the worst-case stack depth is proven to
    /// fit (e.g. with [`StackReport::worst_case_depth`]).
    pub stack_probes: bool,
    /// The **relocation model** (default [`RelocModel::Static`]): whether the
    /// code must run at any load address, and which symbols may be preempted
    /// by another component at run time. See [`RelocModel`] and
    /// [`crate::codegen::linkage`].
    pub reloc_model: RelocModel,
    /// The operating system the code runs on (default [`TargetOs::Linux`]).
    /// With the backend's architecture it forms the [`Triple`] whose calling
    /// convention ([`Triple::call_conv`]) every compiled function follows: on x86-64, [`TargetOs::Windows`] selects the Microsoft x64
    /// convention and every other OS System V. It does not change the object
    /// *format* — pick the writer with [`crate::mc::write_object`].
    ///
    /// [`Triple`]: crate::target::Triple
    /// [`Triple::call_conv`]: crate::target::Triple::call_conv
    pub os: TargetOs,
    /// The **unwind tables** to emit (default `None`: the OS's own, see
    /// [`UnwindTables::default_for`] — `.pdata`/`.xdata` on Windows, compact
    /// unwind on Darwin, none elsewhere). [`UnwindTables::EhFrame`] adds DWARF
    /// `.eh_frame` call-frame information to an ELF object. See
    /// [`crate::codegen::unwind`]; x86-64 emits every kind, AArch64 the
    /// compact unwind of frames it can encode.
    pub unwind: Option<UnwindTables>,
    /// The **function alignment** in bytes (default `None`: the target's own,
    /// 16 on x86-64 and 4 on AArch64, RISC-V and Thumb; AVR and WebAssembly
    /// do not align functions). Each function's first instruction is placed
    /// at a multiple of it within `.text`, the gap before it filled with the
    /// target's no-op, and the `.text` section asks for that alignment too so
    /// a linker keeps it. The value is rounded up to a power of two, and up to
    /// the target's instruction alignment where it has one (4 on AArch64,
    /// RISC-V and Thumb): see [`CodegenOptions::function_alignment_for`].
    ///
    /// The tradeoff is size against speed. `1` on x86-64 packs functions back
    /// to back, saving up to 15 bytes of `nop`s per function, which matters
    /// for small programs; the 16-byte default keeps each function's entry on
    /// a fetch block, so a hot function or loop head is not split across two
    /// 16-byte fetch windows (or two cache lines) for want of padding.
    pub function_alignment: Option<u64>,
    /// Lower the bulk-memory ops a target would otherwise select inline for a
    /// long or variable length (`docs/ir-design.md` §6k) to calls to the C
    /// library's `memcpy`, `memmove` and `memset` instead (default `false`).
    /// Only for hosted code linked against a libc. Short constant lengths
    /// stay inline either way. Honored by x86-64, where the inline form is
    /// `rep movsb` / `rep stosb`; the other targets always use their loops.
    pub bulk_memory_libcalls: bool,
}

/// How position-dependent the generated code may be, and so how it addresses
/// symbols (`docs/ir-design.md` §4b).
///
/// Every model addresses code and data RIP-relatively where the target allows
/// it; they differ in which symbols are reached *through the GOT*.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum RelocModel {
    /// A position-dependent (or statically linked) executable: every symbol is
    /// addressed directly and resolved at static link time. The default, and
    /// what LatticeFoundry's own static linker consumes.
    #[default]
    Static,
    /// Position-independent code for an **executable** (`-fPIE`): symbols the
    /// module defines bind locally (an executable is never preempted), so only
    /// references to symbols defined elsewhere — possibly in a shared library —
    /// go through the GOT.
    Pie,
    /// Position-independent code for a **shared library** (`-fPIC`): a
    /// default- or protected-visibility symbol may resolve outside the
    /// library, so its address comes from the GOT and calls to it go through
    /// the PLT. Only `internal` and `hidden` symbols are addressed directly.
    Pic,
}

impl RelocModel {
    /// Whether code must be position-independent (`Pie` or `Pic`).
    pub fn is_pic(self) -> bool {
        self != RelocModel::Static
    }
}

impl Default for CodegenOptions {
    fn default() -> CodegenOptions {
        CodegenOptions {
            stack_probes: true,
            reloc_model: RelocModel::Static,
            os: TargetOs::Linux,
            unwind: None,
            function_alignment: None,
            bulk_memory_libcalls: false,
        }
    }
}

impl CodegenOptions {
    /// Call the C library for long or variable bulk-memory ops (see
    /// [`CodegenOptions::bulk_memory_libcalls`]).
    pub fn with_bulk_memory_libcalls(mut self, on: bool) -> CodegenOptions {
        self.bulk_memory_libcalls = on;
        self
    }

    /// Enable or disable stack probes (see [`CodegenOptions::stack_probes`]).
    pub fn with_stack_probes(mut self, on: bool) -> CodegenOptions {
        self.stack_probes = on;
        self
    }

    /// Set the relocation model (see [`CodegenOptions::reloc_model`]).
    pub fn with_reloc_model(mut self, model: RelocModel) -> CodegenOptions {
        self.reloc_model = model;
        self
    }

    /// Shorthand: position-independent code for a shared library
    /// ([`RelocModel::Pic`]) when `on`, else [`RelocModel::Static`].
    pub fn with_pic(self, on: bool) -> CodegenOptions {
        self.with_reloc_model(if on { RelocModel::Pic } else { RelocModel::Static })
    }

    /// Target the operating system `os` (see [`CodegenOptions::os`]).
    pub fn with_os(mut self, os: TargetOs) -> CodegenOptions {
        self.os = os;
        self
    }

    /// Emit the unwind tables `tables` (see [`CodegenOptions::unwind`]).
    pub fn with_unwind_tables(mut self, tables: UnwindTables) -> CodegenOptions {
        self.unwind = Some(tables);
        self
    }

    /// Align every function to `align` bytes (see
    /// [`CodegenOptions::function_alignment`]); `1` packs them back to back.
    pub fn with_function_alignment(mut self, align: u64) -> CodegenOptions {
        self.function_alignment = Some(align);
        self
    }

    /// The function alignment a backend uses: the explicit
    /// [`CodegenOptions::function_alignment`] rounded up to a power of two, or
    /// the target's `default`, and never below the target's instruction
    /// alignment `min`.
    pub fn function_alignment_for(&self, default: u64, min: u64) -> u64 {
        let a = self.function_alignment.map_or(default, |a| a.max(1).checked_next_power_of_two().unwrap_or(default));
        a.max(min.max(1))
    }

    /// The unwind tables this compilation emits: the explicit choice, or the
    /// OS's default.
    pub fn unwind_tables(&self) -> UnwindTables {
        self.unwind.unwrap_or(UnwindTables::default_for(self.os))
    }
}

/// The result of an option-taking `compile_module_with`: the relocatable object
/// plus the stack-usage report of every function defined in it.
#[derive(Clone, Debug)]
pub struct CompiledModule {
    /// The relocatable object (identical to what `compile_module` returns for the
    /// same options).
    pub object: ObjectModule,
    /// Per-function stack usage, in definition order.
    pub stack: StackReport,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::object::{SymbolType, SymbolValue};
    use crate::target::TargetArch;

    #[test]
    fn function_alignment_for_rounds_and_clamps() {
        let o = CodegenOptions::default();
        assert_eq!(o.function_alignment_for(16, 1), 16);
        assert_eq!(o.function_alignment_for(4, 4), 4);
        let a = |n| CodegenOptions::default().with_function_alignment(n);
        assert_eq!(a(1).function_alignment_for(16, 1), 1);
        assert_eq!(a(0).function_alignment_for(16, 1), 1);
        assert_eq!(a(3).function_alignment_for(16, 1), 4);
        assert_eq!(a(1).function_alignment_for(4, 4), 4);
        assert_eq!(a(32).function_alignment_for(4, 4), 32);
    }

    /// Every backend that aligns functions honors the option, down to its
    /// instruction alignment.
    #[test]
    fn backends_align_functions_as_asked() {
        const SRC: &str = "module \"a\"\n\
            func @f(i64) -> i64 {\nentry ^0(%x: i64):\n  %y = add %x, i64 1 : i64\n  ret %y\n}\n\
            func @g(i64) -> i64 {\nentry ^0(%x: i64):\n  %y = call @f(%x) : i64\n  %z = mul %y, %x : i64\n  ret %z\n}\n\
            func @h() -> i64 {\nentry ^0:\n  %r = call @g(i64 3) : i64\n  ret %r\n}\n";
        let mut syms = crate::support::StrInterner::new();
        let m = crate::ir::text::parse_module(SRC, crate::support::diagnostics::FileId::new(0), &mut syms).unwrap();
        for (arch, default, min) in [
            (TargetArch::X86_64, 16, 1),
            (TargetArch::AArch64, 4, 4),
            (TargetArch::Riscv64, 4, 4),
            (TargetArch::Thumb, 4, 4),
        ] {
            for asked in [None, Some(1), Some(8), Some(32)] {
                let opts = match asked {
                    Some(a) => CodegenOptions::default().with_function_alignment(a),
                    None => CodegenOptions::default(),
                };
                let want = asked.unwrap_or(default).max(min);
                let obj = crate::target::compile_module_for(arch, &m, &syms, &opts).unwrap().object;
                let text = obj.sections().iter().position(|s| s.name == ".text").unwrap();
                assert_eq!(obj.sections()[text].align, want, "{arch:?} {asked:?}");
                let mut offs: Vec<(u64, u64)> = obj
                    .symbols()
                    .iter()
                    .filter(|s| s.kind == SymbolType::Func)
                    .filter_map(|s| match s.value {
                        // Thumb function symbols carry the Thumb bit.
                        SymbolValue::Defined { section, offset } if section.index() == text => {
                            Some((if arch == TargetArch::Thumb { offset & !1 } else { offset }, s.size))
                        }
                        _ => None,
                    })
                    .collect();
                offs.sort();
                assert_eq!(offs.len(), 3, "{arch:?}");
                for w in offs.windows(2) {
                    assert_eq!(w[0].0 % want, 0, "{arch:?} {asked:?}: {offs:?}");
                    assert_eq!(w[1].0, (w[0].0 + w[0].1).next_multiple_of(want), "{arch:?} {asked:?}: {offs:?}");
                }
            }
        }
    }
}
