//! **DFE** — dead-function elimination.
//!
//! Inlining copies a small callee into each of its callers, but the callee's own
//! definition stays in the module, and so does any function nothing ever
//! called. Every backend emits every definition, and the linker cannot drop
//! a function from a section that also holds live code, so an unreferenced
//! function costs its full size in the image. This module pass removes the
//! **internal** functions ([`Linkage::Internal`]) that nothing can reach.
//!
//! ## What is live
//!
//! Liveness is reachability over the IR reference graph, from these roots:
//!
//! - every function that is not internal (external and weak definitions can be
//!   called from other objects, and a declaration is a reference, not a body);
//! - every function whose address appears in a **global initializer**
//!   ([`Module::global_referenced_functions`]), live or not — globals are not
//!   removed here;
//! - every function that shares its **symbol name** with a global. A frontend
//!   may reach a function through a symbol-name alias instead of a `func_ref`
//!   (a body-less global of the same name), and such a reference is invisible
//!   to the IR graph.
//!
//! From a live function, every function its body references
//! ([`Module::referenced_functions`]: used `func_ref` values, i.e. call targets
//! and taken addresses, and used address constants) is live too. Everything
//! else is unreachable and removed in one [`Module::remove_functions`] call.
//!
//! Reachability is the fixpoint of "remove a function with no remaining
//! references" — removing one function can orphan the functions only it
//! called — and is strictly stronger: a cycle of internal functions that only
//! call each other is removed as well.
//!
//! A function referenced only by symbol name from an **inline-asm template**
//! is not seen; as with C's `static` functions, such a function must not be
//! internal (or must be referenced from the IR).
//!
//! ## Soundness
//!
//! Only unreachable code is deleted, so the behavior of every live function is
//! unchanged (a trivial refinement, tenet T3 / bet B2), and no instruction of a
//! surviving function is touched, so the transform cannot add or remove a
//! branch (the constant-time discipline is unaffected). Removal compacts the
//! function list and renumbers the surviving [`FuncId`]s (see
//! [`Module::remove_functions`]); the pass manager invalidates cached analyses
//! when the pass reports a change.

use crate::ir::{FuncId, Linkage, Module};
use crate::pass::{Changed, ModulePass};

/// The dead-function-elimination pass (see the module documentation).
#[derive(Debug, Default, Clone, Copy)]
pub struct DeadFunctionElim;

impl DeadFunctionElim {
    /// A dead-function-elimination pass.
    pub fn new() -> Self {
        Self
    }
}

impl ModulePass for DeadFunctionElim {
    fn name(&self) -> &str {
        "dfe"
    }

    fn run(&mut self, module: &mut Module) -> Changed {
        let dead = unreachable_functions(module);
        if dead.is_empty() {
            return Changed::No;
        }
        module
            .remove_functions(&dead)
            .expect("an unreachable function has no live referrer");
        Changed::Yes
    }
}

/// The internal functions of `module` that are unreachable from its roots (see
/// the module documentation), ascending.
pub fn unreachable_functions(module: &Module) -> Vec<FuncId> {
    let n = module.function_count();
    let mut live = vec![false; n];
    let mut work: Vec<FuncId> = Vec::new();
    let mark = |f: FuncId, live: &mut Vec<bool>, work: &mut Vec<FuncId>| {
        if !live[f.index()] {
            live[f.index()] = true;
            work.push(f);
        }
    };

    let global_names: std::collections::HashSet<_> = module.globals().map(|g| g.name).collect();
    for f in module.func_ids() {
        let func = module.function(f);
        if func.attrs.linkage != Linkage::Internal || global_names.contains(&func.name) {
            mark(f, &mut live, &mut work);
        }
    }
    for g in 0..module.global_count() {
        for f in module.global_referenced_functions(crate::ir::GlobalId::from_index(g)) {
            mark(f, &mut live, &mut work);
        }
    }
    while let Some(f) = work.pop() {
        for g in module.referenced_functions(f) {
            mark(g, &mut live, &mut work);
        }
    }
    module.func_ids().filter(|f| !live[f.index()]).collect()
}
