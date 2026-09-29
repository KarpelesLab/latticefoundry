//! Code-generation options shared by every backend, and the result type of the
//! option-taking compile entry points.
//!
//! Each target's `compile_module` keeps its original signature (default options,
//! returning just the [`ObjectModule`]); the `compile_module_with` siblings take
//! a [`CodegenOptions`] and return a [`CompiledModule`], which carries the object
//! *and* the per-function [`StackReport`] computed from the same frame layouts
//! the prologues were built from.

use crate::codegen::stack::StackReport;
use crate::mc::object::ObjectModule;

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
}

impl Default for CodegenOptions {
    fn default() -> CodegenOptions {
        CodegenOptions { stack_probes: true }
    }
}

impl CodegenOptions {
    /// Enable or disable stack probes (see [`CodegenOptions::stack_probes`]).
    pub fn with_stack_probes(mut self, on: bool) -> CodegenOptions {
        self.stack_probes = on;
        self
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
