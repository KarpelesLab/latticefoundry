//! **Secret taint**: which values are derived from a secret (the analysis
//! behind constant-time preservation, `docs/ir-design.md` §6d; a first, narrow
//! step toward bet B10).
//!
//! Secrets enter a function in exactly five ways, all declared in the IR:
//!
//! - a **secret parameter** ([`FuncAttrs`](crate::ir::FuncAttrs));
//! - a **`load secret`** (the front end read memory it typed `secret[T]`);
//! - a load whose address is based on a **secret global**
//!   ([`GlobalAttrs::secret`](crate::ir::GlobalAttrs::secret));
//! - the result of a **direct call** to a function whose return is secret;
//! - a load of memory this function itself wrote a secret into (below).
//!
//! and they leave only through [`declassify`](crate::ir::InstKind::Declassify).
//! Everything else propagates: a pure operation (arithmetic, comparison, cast,
//! `select`, `freeze`, `ptr_add`) is secret if any operand is, and a block
//! parameter is secret if any executable edge passes it a secret argument.
//!
//! ## One engine
//!
//! Value taint is the two-point lattice `Public ⊑ Secret` (plus ⊥ for "not yet
//! reached"): an [`AbstractDomain`] like any other, solved by the one sparse
//! fixpoint engine ([`solve_with`], bet B8). The module context the domain's
//! transfer cannot see — which parameters and globals are secret, which callee
//! returns a secret, what memory holds secrets — comes in through
//! [`SolveHooks`]. γ is trivial (taint is not a property of the value's bits:
//! every non-⊥ element contains every value), so the soundness harness holds
//! vacuously; what matters is monotonicity, which the join-of-operands transfer
//! has by construction.
//!
//! **No implicit flows.** Taint tracks data flow only. That is sound *because*
//! the constant-time verifier forbids a secret-derived branch or switch
//! condition: with no secret-dependent control flow there is no control
//! dependence for a secret to leak through. (After a `declassify` the value is
//! public by decree, which is the point of the escape hatch.)
//!
//! ## Memory (conservative)
//!
//! Every address is traced to a **root** ([`MemRoot`]): a stack slot
//! (`alloca`/`dyn_alloca`) whose address never escapes (it is only ever used,
//! possibly through `ptr_add`/`bitcast`/`freeze`, as the address of a load,
//! store or atomic), a global named directly, or `Unknown` (a parameter, a
//! loaded or returned pointer, an escaping slot, a `select` of pointers, ...).
//! A flow-insensitive memory summary records which roots may hold a secret:
//!
//! - initially, every secret global;
//! - every root a secret-derived value is stored to (by `store`, or the value
//!   operand of an atomic), and every root a `store secret` writes;
//! - `Unknown` memory as soon as a call or syscall receives a secret argument
//!   (the callee may stash it through any pointer it can reach).
//!
//! A load (or atomic read) from a root is secret if the root may hold a secret:
//! a stack slot only through its own stores; a global if it is secret or this
//! function stores a secret to it by name; `Unknown` through a store to
//! `Unknown` or to any global. Because loads feed stores, the summary and the
//! value taint are iterated together to a fixpoint (both only grow, so it
//! terminates).
//!
//! A write through an unknown pointer does **not** taint a global read by name:
//! such a write is a `store secret` (an unflagged one is rejected by the
//! verifier) or happens in a callee, and both declare *secret memory*, which
//! the modular contract below says is read with `load secret` or lives in a
//! `secret` global. This keeps a public global (a green-thread preemption flag,
//! a counter) public in a function that also writes secrets through pointers.
//!
//! Across calls memory is **modular**: secrets that cross a function boundary
//! through memory are declared at both ends (`store secret` / `load secret`,
//! or a secret global), which the verifier enforces on the storing side (a
//! secret-derived value may only be stored unflagged into a non-escaping stack
//! slot or a secret global). Function-pointer types carry no secrecy, so an
//! indirect call's result is public and its arguments must be public.

use crate::analysis::domain::{AbstractDomain, DomainCtx};
use crate::analysis::solver::{FixpointResult, SolveHooks, edge_args, solve_with};
use crate::ir::inst::{CastOp, InstData, InstId, InstKind};
use crate::ir::value::{Const, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, GlobalId, Module, SemValue};

// ---------------------------------------------------------------------------
// The domain.
// ---------------------------------------------------------------------------

/// The secret-taint lattice: `Bottom ⊑ Public ⊑ Secret`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Taint {
    /// Not yet reached (unreachable code).
    Bottom,
    /// Derived from public data only.
    Public,
    /// Possibly derived from a secret.
    Secret,
}

impl Taint {
    /// Whether this is [`Taint::Secret`].
    pub fn is_secret(self) -> bool {
        self == Taint::Secret
    }

    /// `Secret` if `b`, else `Public`.
    pub fn from_secret(b: bool) -> Taint {
        if b { Taint::Secret } else { Taint::Public }
    }

    fn rank(self) -> u8 {
        match self {
            Taint::Bottom => 0,
            Taint::Public => 1,
            Taint::Secret => 2,
        }
    }
}

impl AbstractDomain for Taint {
    fn bottom() -> Self {
        Taint::Bottom
    }

    fn top() -> Self {
        Taint::Secret
    }

    fn join(&self, other: &Self) -> Self {
        if self.rank() >= other.rank() { *self } else { *other }
    }

    fn le(&self, other: &Self) -> bool {
        self.rank() <= other.rank()
    }

    fn contains(&self, _v: &SemValue) -> bool {
        // Taint is not a property of a value's bits: every reached element
        // stands for every value.
        *self != Taint::Bottom
    }

    fn abstract_const(_ctx: DomainCtx<'_>, _c: &Const) -> Self {
        Taint::Public
    }

    /// The context-free transfer: the join of the operands (any secret operand
    /// makes the result secret), except that `declassify` is always public and
    /// a `load secret` always secret. Memory and calls are refined by the
    /// analysis hooks ([`SecretTaint`]).
    fn transfer(_ctx: DomainCtx<'_>, inst: &InstData, operands: &[Self]) -> Self {
        match &inst.kind {
            InstKind::Declassify => Taint::Public,
            InstKind::Load { secret: true, .. } => Taint::Secret,
            _ => operands.iter().fold(Taint::Public, |acc, o| acc.join(o)),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory roots.
// ---------------------------------------------------------------------------

/// Where an address points, as far as the secret analysis can tell.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemRoot {
    /// A stack slot (`alloca`/`dyn_alloca`) whose address never escapes.
    Stack(InstId),
    /// A global, named directly.
    Global(GlobalId),
    /// Anything else: a parameter, a loaded or returned pointer, an escaping
    /// slot, a merge of pointers.
    Unknown,
}

/// The flow-insensitive summary of which memory may hold a secret.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct MemState {
    /// `stack[i]`: the non-escaping slot defined by instruction `i` may hold a
    /// secret.
    stack: Vec<bool>,
    /// `globals[g]`: global `g` may hold a secret (secret globals, and globals
    /// this function stores a secret to).
    globals: Vec<bool>,
    /// Some global was *stored* a secret by this function (so an `Unknown`
    /// pointer, which may alias it, may read one).
    stored_global: bool,
    /// `Unknown` memory may hold a secret.
    unknown: bool,
}

impl MemState {
    fn root_secret(&self, root: MemRoot) -> bool {
        match root {
            MemRoot::Stack(i) => self.stack[i.index()],
            MemRoot::Global(g) => self.globals[g.index()],
            MemRoot::Unknown => self.unknown || self.stored_global,
        }
    }

    fn taint(&mut self, root: MemRoot) {
        match root {
            MemRoot::Stack(i) => self.stack[i.index()] = true,
            MemRoot::Global(g) => {
                self.globals[g.index()] = true;
                self.stored_global = true;
            }
            MemRoot::Unknown => self.unknown = true,
        }
    }
}

/// Where a secret-derived value's secrecy comes from: one step of the chain a
/// diagnostic prints (see [`SecretTaint::origin`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Origin {
    /// Parameter `index` of the function is declared secret.
    SecretParam(usize),
    /// A `load secret`.
    SecretLoad(InstId),
    /// A load (or atomic read) of memory that may hold a secret.
    SecretMemory(InstId, MemRoot),
    /// A direct call to a function whose return is secret.
    SecretReturn(InstId, FuncId),
    /// An instruction with a secret-derived operand `from`.
    Operand(InstId, ValueId),
    /// A block parameter passed the secret-derived argument `from` on an edge.
    BlockArg(BlockId, ValueId),
}

// ---------------------------------------------------------------------------
// The analysis.
// ---------------------------------------------------------------------------

/// The secret-taint fixpoint of one function (see the module docs).
#[derive(Debug)]
pub struct SecretTaint {
    values: FixpointResult<Taint>,
    roots: Vec<Option<MemRoot>>,
    mem: MemState,
    secret_params: Vec<bool>,
}

/// The module context plugged into the engine.
struct Hooks<'a> {
    module: &'a Module,
    func: &'a Function,
    entry: Option<BlockId>,
    secret_params: &'a [bool],
    roots: &'a [Option<MemRoot>],
    mem: &'a MemState,
}

impl SolveHooks<Taint> for Hooks<'_> {
    fn seed(&self, _v: ValueId, def: &ValueDef) -> Option<Taint> {
        Some(match def {
            ValueDef::Param(b, i) if Some(*b) == self.entry => {
                Taint::from_secret(self.secret_params.get(*i as usize).copied().unwrap_or(false))
            }
            // Addresses and constants are public.
            _ => Taint::Public,
        })
    }

    fn transfer(&self, _inst: InstId, data: &InstData, operands: &[Taint]) -> Option<Taint> {
        let ops = data.operands();
        let root = |i: usize| ops.get(i).and_then(|v| self.roots[v.index()]).unwrap_or(MemRoot::Unknown);
        Some(match &data.kind {
            InstKind::Load { secret, .. } => Taint::from_secret(*secret || self.mem.root_secret(root(0))),
            InstKind::AtomicLoad { .. } | InstKind::AtomicRmw { .. } | InstKind::CmpXchg { .. } => {
                Taint::from_secret(self.mem.root_secret(root(0)))
            }
            InstKind::Call => {
                let secret_ret = match self.func.value(ops[0]).def {
                    ValueDef::Func(f) => self.module.func_attrs(f).secret_ret,
                    _ => false,
                };
                Taint::from_secret(secret_ret)
            }
            InstKind::Syscall | InstKind::Alloca { .. } | InstKind::DynAlloca { .. } => {
                Taint::Public
            }
            // An inline asm's outputs derive from its operands and, when it
            // may read memory, from whatever (escaped) memory holds; its
            // pointer operands escape, so that memory is the unknown root.
            InstKind::InlineAsm(asm) => {
                let from_ops = operands.iter().any(|t| t.is_secret());
                Taint::from_secret(from_ops || (asm.may_access_memory() && self.mem.root_secret(MemRoot::Unknown)))
            }
            _ => return None,
        })
    }
}

impl SecretTaint {
    /// Compute the secret taint of function `func` of `module`.
    pub fn compute(module: &Module, func: FuncId) -> SecretTaint {
        let f = module.function(func);
        let attrs = module.func_attrs(func);
        let nparams = f.entry().map_or(0, |e| f.block(e).params().len());
        let secret_params: Vec<bool> = (0..nparams).map(|i| attrs.is_param_secret(i)).collect();
        let roots = compute_roots(f);
        let mut mem = MemState {
            stack: vec![false; f.inst_count()],
            globals: (0..module.global_count())
                .map(|g| module.global_attrs(GlobalId::from_index(g)).secret)
                .collect(),
            stored_global: false,
            unknown: false,
        };
        loop {
            let hooks = Hooks {
                module,
                func: f,
                entry: f.entry(),
                secret_params: &secret_params,
                roots: &roots,
                mem: &mem,
            };
            let values = solve_with(f, module.types(), module.consts(), &hooks);
            let next = summarize_memory(f, &values, &roots, &mem);
            if next == mem {
                return SecretTaint { values, roots, mem, secret_params };
            }
            mem = next;
        }
    }

    /// Whether value `v` may be derived from a secret.
    pub fn is_secret(&self, v: ValueId) -> bool {
        self.values.value(v).is_secret()
    }

    /// The taint of value `v` ([`Taint::Bottom`] in unreachable code).
    pub fn taint(&self, v: ValueId) -> Taint {
        *self.values.value(v)
    }

    /// `result[v]` is whether value `v` may be derived from a secret.
    pub fn secret_values(&self) -> Vec<bool> {
        (0..self.values.value_count()).map(|i| self.is_secret(ValueId::from_index(i))).collect()
    }

    /// Whether any value of the function is secret-derived.
    pub fn any_secret(&self) -> bool {
        (0..self.values.value_count()).any(|i| self.is_secret(ValueId::from_index(i)))
    }

    /// Whether block `b` is reachable (the analysis visits every edge; only
    /// blocks unreachable from the entry are not).
    pub fn is_reachable(&self, b: BlockId) -> bool {
        self.values.is_reachable(b)
    }

    /// The memory root of pointer value `v`.
    pub fn root_of(&self, v: ValueId) -> MemRoot {
        self.roots[v.index()].unwrap_or(MemRoot::Unknown)
    }

    /// Whether memory rooted at `root` may hold a secret.
    pub fn memory_secret(&self, root: MemRoot) -> bool {
        self.mem.root_secret(root)
    }

    /// Why the secret-derived value `v` is secret: one step back toward a
    /// source (`None` if `v` is not secret-derived). Follow
    /// [`Origin::Operand`] / [`Origin::BlockArg`] to walk the chain; the other
    /// variants are sources. Each step moves strictly closer to a source, so
    /// the walk ends (it never circles a loop).
    pub fn origin(&self, module: &Module, func: FuncId, v: ValueId) -> Option<Origin> {
        let f = module.function(func);
        let steps = self.steps(f, v);
        if let Some(src) = steps.source {
            return Some(src);
        }
        let d = self.depth(f, v)?;
        steps.from.into_iter().find(|o| {
            let prev = match *o {
                Origin::Operand(_, p) | Origin::BlockArg(_, p) => p,
                _ => return false,
            };
            self.depth(f, prev).is_some_and(|pd| pd < d)
        })
    }

    /// The full chain from `v` back to a secret source (empty if `v` is
    /// public): `origin(v)`, then the origin of the value it names, and so on.
    pub fn explain(&self, module: &Module, func: FuncId, v: ValueId) -> Vec<Origin> {
        let mut chain = Vec::new();
        let mut cur = v;
        while let Some(o) = self.origin(module, func, cur) {
            chain.push(o);
            match o {
                Origin::Operand(_, p) | Origin::BlockArg(_, p) => cur = p,
                _ => break,
            }
        }
        chain
    }

    /// The immediate reasons `v` may be secret: a source, and/or the
    /// secret-derived operands / block arguments it depends on.
    fn steps(&self, f: &Function, v: ValueId) -> Steps {
        let mut s = Steps { source: None, from: Vec::new() };
        if !self.is_secret(v) {
            return s;
        }
        match f.value(v).def {
            ValueDef::Param(b, i) if Some(b) == f.entry() => {
                if self.secret_params.get(i as usize).copied().unwrap_or(false) {
                    s.source = Some(Origin::SecretParam(i as usize));
                }
            }
            ValueDef::Param(b, i) => {
                for (_, block) in f.blocks() {
                    let Some(t) = block.terminator() else { continue };
                    let term = f.inst(t);
                    for (si, succ) in term.successors().into_iter().enumerate() {
                        if succ != b {
                            continue;
                        }
                        if let Some(&arg) = edge_args(term, si).get(i as usize)
                            && self.is_secret(arg)
                        {
                            s.from.push(Origin::BlockArg(b, arg));
                        }
                    }
                }
            }
            ValueDef::Inst(i) => {
                let data = f.inst(i);
                let ops = data.operands();
                match &data.kind {
                    InstKind::Load { secret: true, .. } => s.source = Some(Origin::SecretLoad(i)),
                    InstKind::Load { .. }
                    | InstKind::AtomicLoad { .. }
                    | InstKind::AtomicRmw { .. }
                    | InstKind::CmpXchg { .. } => {
                        s.source = Some(Origin::SecretMemory(i, self.root_of(ops[0])));
                    }
                    InstKind::Call => {
                        if let ValueDef::Func(callee) = f.value(ops[0]).def {
                            s.source = Some(Origin::SecretReturn(i, callee));
                        }
                    }
                    _ => {
                        s.from = ops
                            .iter()
                            .filter(|&&o| self.is_secret(o))
                            .map(|&o| Origin::Operand(i, o))
                            .collect();
                    }
                }
            }
            _ => {}
        }
        s
    }

    /// The length of the shortest chain from `v` back to a secret source, or
    /// `None` if `v` is public. Computed on demand by a bounded breadth-first
    /// search backwards over [`SecretTaint::steps`] (diagnostics only).
    fn depth(&self, f: &Function, v: ValueId) -> Option<u32> {
        let mut seen = vec![false; f.value_count()];
        let mut frontier = vec![v];
        seen[v.index()] = true;
        let mut d = 0u32;
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for &x in &frontier {
                let s = self.steps(f, x);
                if s.source.is_some() {
                    return Some(d);
                }
                for o in s.from {
                    if let Origin::Operand(_, p) | Origin::BlockArg(_, p) = o
                        && !std::mem::replace(&mut seen[p.index()], true)
                    {
                        next.push(p);
                    }
                }
            }
            frontier = next;
            d += 1;
        }
        None
    }
}

/// See [`SecretTaint::steps`].
struct Steps {
    source: Option<Origin>,
    from: Vec<Origin>,
}

/// Recompute the memory summary from the current value taint (monotone: it
/// starts from `prev` and only adds).
fn summarize_memory(
    f: &Function,
    values: &FixpointResult<Taint>,
    roots: &[Option<MemRoot>],
    prev: &MemState,
) -> MemState {
    let mut mem = prev.clone();
    let root = |v: ValueId| roots[v.index()].unwrap_or(MemRoot::Unknown);
    let secret = |v: ValueId| values.value(v).is_secret();
    for (bid, block) in f.blocks() {
        if !values.is_reachable(bid) {
            continue;
        }
        for &i in block.insts() {
            let data = f.inst(i);
            let ops = data.operands();
            match &data.kind {
                InstKind::Store { secret: flagged, .. } if *flagged || secret(ops[1]) => {
                    mem.taint(root(ops[0]));
                }
                InstKind::AtomicStore { .. } | InstKind::AtomicRmw { .. } if secret(ops[1]) => {
                    mem.taint(root(ops[0]));
                }
                InstKind::CmpXchg { .. } if secret(ops[2]) => mem.taint(root(ops[0])),
                InstKind::Call if ops[1..].iter().any(|&a| secret(a)) => {
                    mem.taint(MemRoot::Unknown);
                }
                InstKind::Syscall if ops.iter().any(|&a| secret(a)) => mem.taint(MemRoot::Unknown),
                InstKind::InlineAsm(asm) if asm.may_access_memory() && ops.iter().any(|&a| secret(a)) => {
                    mem.taint(MemRoot::Unknown);
                }
                _ => {}
            }
        }
    }
    mem
}

/// The memory root of every value (`None` for values that are not addresses
/// of anything the analysis tracks; treated as `Unknown` when dereferenced).
fn compute_roots(f: &Function) -> Vec<Option<MemRoot>> {
    let n = f.value_count();
    let mut roots: Vec<Option<MemRoot>> = vec![None; n];
    // A function's instructions may sit anywhere in the arena, so resolve each
    // root recursively with a memo (bounded depth: a pathological chain simply
    // becomes `Unknown`, which is conservative).
    fn resolve(f: &Function, v: ValueId, roots: &mut [Option<MemRoot>], depth: u32) -> MemRoot {
        if let Some(r) = roots[v.index()] {
            return r;
        }
        let r = match f.value(v).def {
            ValueDef::Global(g) => MemRoot::Global(g),
            ValueDef::Inst(i) if depth < 64 => {
                let data = f.inst(i);
                match &data.kind {
                    InstKind::Alloca { .. } | InstKind::DynAlloca { .. } => {
                        if slot_escapes(f, v) { MemRoot::Unknown } else { MemRoot::Stack(i) }
                    }
                    InstKind::PtrAdd { .. }
                    | InstKind::Cast(CastOp::Bitcast)
                    | InstKind::Freeze => resolve(f, data.operands()[0], roots, depth + 1),
                    _ => MemRoot::Unknown,
                }
            }
            _ => MemRoot::Unknown,
        };
        roots[v.index()] = Some(r);
        r
    }
    for i in 0..n {
        resolve(f, ValueId::from_index(i), &mut roots, 0);
    }
    roots
}

/// Whether the address `v` of a stack slot (or a pointer derived from it by
/// `ptr_add`/`bitcast`/`freeze`) is used as anything but the address of a load,
/// store or atomic — i.e. whether other code may reach the slot.
fn slot_escapes(f: &Function, v: ValueId) -> bool {
    let mut work = vec![v];
    let mut seen = vec![false; f.value_count()];
    while let Some(p) = work.pop() {
        if std::mem::replace(&mut seen[p.index()], true) {
            continue;
        }
        for u in f.uses_of(p) {
            let data = f.inst(u.inst);
            let derived = match &data.kind {
                InstKind::Load { .. } | InstKind::AtomicLoad { .. } => u.operand == 0,
                InstKind::Store { .. }
                | InstKind::AtomicStore { .. }
                | InstKind::AtomicRmw { .. }
                | InstKind::CmpXchg { .. } => {
                    if u.operand == 0 {
                        continue;
                    }
                    return true; // the address itself is stored / exchanged
                }
                InstKind::PtrAdd { .. } | InstKind::Cast(CastOp::Bitcast) | InstKind::Freeze
                    if u.operand == 0 =>
                {
                    match data.result() {
                        Some(r) => {
                            work.push(r);
                            continue;
                        }
                        None => return true,
                    }
                }
                _ => return true,
            };
            if !derived {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests;
