//! Pass and analysis infrastructure. See ROADMAP Phase 3.
//!
//! Transformations implement [`ModulePass`] and are sequenced by a
//! [`PassManager`]. Analyses (dominators, CFG, liveness, ...) are layered on
//! top in Phase 3 and cached across passes.

use crate::ir::Module;

/// Whether a pass mutated the IR. Drives fixpoint iteration and cache
/// invalidation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Changed {
    /// The pass modified the module.
    Yes,
    /// The pass left the module unchanged.
    No,
}

/// A transformation over an entire module.
pub trait ModulePass {
    /// A short, stable name used in pass pipelines and diagnostics.
    fn name(&self) -> &str;

    /// Run the pass, reporting whether it changed anything.
    fn run(&mut self, module: &mut Module) -> Changed;
}

/// An ordered collection of passes executed over a module.
#[derive(Default)]
pub struct PassManager {
    passes: Vec<Box<dyn ModulePass>>,
}

impl std::fmt::Debug for PassManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassManager")
            .field("passes", &self.passes.iter().map(|p| p.name()).collect::<Vec<_>>())
            .finish()
    }
}

impl PassManager {
    /// Create an empty pass pipeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a pass to the pipeline.
    pub fn add(&mut self, pass: Box<dyn ModulePass>) {
        self.passes.push(pass);
    }

    /// Run every pass in order over `module`.
    pub fn run(&mut self, module: &mut Module) {
        for pass in &mut self.passes {
            let _ = pass.run(module);
        }
    }
}
