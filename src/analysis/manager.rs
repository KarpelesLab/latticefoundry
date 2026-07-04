//! The analysis manager: run an analysis over a function, **cache** its result,
//! and **invalidate** the cache when a pass reports it mutated the IR.
//!
//! An analysis is a [`FunctionAnalysis`]: a stable id plus a `run` that computes
//! a result for one function. [`AnalysisCache`] memoizes results keyed by
//! `(analysis id, function)`, type-erased through [`Any`] so one cache holds
//! results of every domain. Invalidation is coarse but real — a mutating pass
//! drops the affected entries (or all of them) — and ties into the existing
//! [`Changed`](crate::pass::Changed) signal via
//! [`crate::pass::PassManager`], which clears the cache after any pass that
//! reports [`Changed::Yes`](crate::pass::Changed::Yes).
//!
//! The design is deliberately compatible with the parallel/incremental goals
//! (tenets T5/T6): keys are dense ids, the map is a [`DetHashMap`] (deterministic
//! iteration), and results are owned (no borrow of the mutated module survives an
//! invalidation).

use std::any::Any;
use std::fmt;

use crate::analysis::domains::ConstLattice;
use crate::analysis::solver::{self, FixpointResult};
use crate::ir::value::ConstPool;
use crate::ir::{FuncId, Function, TypeContext};
use crate::support::DetHashMap;

/// An analysis computable over a single function, producing a cacheable result.
pub trait FunctionAnalysis {
    /// The result type this analysis produces (owned, so it can outlive the
    /// borrow of the function it was computed from).
    type Result: 'static;

    /// A short, stable identifier used as part of the cache key.
    fn id(&self) -> &'static str;

    /// Compute the analysis result for `func`.
    fn run(&self, func: &Function, types: &TypeContext, consts: &ConstPool) -> Self::Result;
}

/// The key identifying a cached result: an analysis id and a function index.
type CacheKey = (&'static str, u32);

/// A memoizing cache of analysis results, invalidated on IR mutation.
#[derive(Default)]
pub struct AnalysisCache {
    entries: DetHashMap<CacheKey, Box<dyn Any>>,
    computations: usize,
}

impl fmt::Debug for AnalysisCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut keys: Vec<&CacheKey> = self.entries.keys().collect();
        keys.sort_unstable();
        f.debug_struct("AnalysisCache")
            .field("cached", &keys)
            .field("computations", &self.computations)
            .finish()
    }
}

impl AnalysisCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The result of `analysis` for `func`, computing and caching it on the
    /// first request and returning the cached value thereafter.
    pub fn get_or_compute<A: FunctionAnalysis>(
        &mut self,
        analysis: &A,
        func_id: FuncId,
        func: &Function,
        types: &TypeContext,
        consts: &ConstPool,
    ) -> &A::Result {
        let key = (analysis.id(), func_id.index() as u32);
        if !self.entries.contains_key(&key) {
            let result = analysis.run(func, types, consts);
            self.computations += 1;
            self.entries.insert(key, Box::new(result));
        }
        self.entries
            .get(&key)
            .expect("just inserted")
            .downcast_ref::<A::Result>()
            .expect("cache entry has the analysis's result type")
    }

    /// Whether a result for `analysis` on `func_id` is currently cached.
    pub fn is_cached<A: FunctionAnalysis>(&self, analysis: &A, func_id: FuncId) -> bool {
        self.entries.contains_key(&(analysis.id(), func_id.index() as u32))
    }

    /// Drop every cached result for one function (its IR changed).
    pub fn invalidate_function(&mut self, func_id: FuncId) {
        let idx = func_id.index() as u32;
        self.entries.retain(|k, _| k.1 != idx);
    }

    /// Drop every cached result (a module-wide change).
    pub fn invalidate_all(&mut self) {
        self.entries.clear();
    }

    /// How many results have actually been computed (cache misses). A stable
    /// witness that caching and invalidation behave, for tests and diagnostics.
    pub fn computations(&self) -> usize {
        self.computations
    }

    /// How many results are currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no results.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The constant-propagation analysis over the one lattice engine
/// ([`ConstLattice`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstantPropagation;

impl FunctionAnalysis for ConstantPropagation {
    type Result = FixpointResult<ConstLattice>;

    fn id(&self) -> &'static str {
        "constant-propagation"
    }

    fn run(&self, func: &Function, types: &TypeContext, consts: &ConstPool) -> Self::Result {
        solver::solve::<ConstLattice>(func, types, consts)
    }
}
