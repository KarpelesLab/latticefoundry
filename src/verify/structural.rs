//! The structural + semantic well-formedness checker (the `Structural`
//! verification tier of `docs/design-tenets.md` §2).
//!
//! This is the cheap, solver-free tier: it establishes the invariants every
//! later stage (and the `Refinement`/`z3rs` tier) is entitled to assume. It
//! never stops at the first problem — a single call collects *all* violations as
//! [`Diagnostic`]s so a caller sees the whole picture in one pass.
//!
//! The checks, grouped:
//!
//! - **Blocks & terminators** — every block ends in exactly one terminator, and
//!   no terminator sits anywhere but the terminator slot.
//! - **Control-flow integrity** — every successor `BlockId` exists, and nothing
//!   branches back into the entry block (whose parameters are the function
//!   parameters, not an edge's arguments).
//! - **Block-argument arity & typing** — each edge supplies exactly as many
//!   arguments as the target has parameters, matched by type (our replacement
//!   for φ-node operand checks).
//! - **SSA dominance** — every operand's definition dominates its use (see
//!   [`super::cfg`] for the dominator tree).
//! - **Type agreement** — per-opcode operand/result typing, `call`
//!   arity/signature agreement, `cond_br` on `i1`, `switch` on an integer,
//!   `select` arms, cast compatibility, and `load`/`store` sanity.
//! - **Values** — constants are well-typed and referenced functions/globals
//!   exist.
//! - **Vectors** (`docs/ir-design.md` §6e) — every `<N x T>` has `N ≥ 1` lanes
//!   of `i1`/`i8`/`i16`/`i32`/`i64` or a float type; lane-wise ops check their
//!   scalar rule on the lane types and agree on the lane count (`icmp`/`fcmp`
//!   give `<N x i1>`, a vector `select` condition is `<N x i1>`); lane indices
//!   and shuffle-mask entries are in range; `bitcast` preserves the total bit
//!   width; vector loads/stores are never `volatile`.
//!
//! The `Refinement` tier (per-opcode poison/UB refinement obligations discharged
//! by `z3rs`) is layered on top later and is deliberately *not* implemented
//! here; this module leaves the module untouched and only reads it, so that seam
//! stays clean.
//!
//! ## Aggregate values are addresses (the struct-by-value convention)
//!
//! A value of **aggregate type** (`Struct`/`Array`) *denotes the address of its
//! storage* — its runtime representation is a pointer to that storage. This is
//! the convention the backends already emit and gcc links against (see
//! `build_struct_int` in `src/target/x86_64/tests.rs`): a struct value is an SSA
//! value of struct type whose machine value is a pointer. Concretely, aggregate
//! types and `ptr` are **interchangeable**
//!
//! - as the *base* of address arithmetic (`ptr_add` / `struct_field` /
//!   `array_elem`) and as the *address* operand of `load` / `store`, and
//! - across the *call / return* boundary (a `ptr` may be passed where an
//!   aggregate parameter is declared, an aggregate value where a `ptr` parameter
//!   is declared, and likewise for `ret` vs. the return type), and
//! - through `bitcast` (`ptr` -> aggregate and back: the identity on the
//!   address). A backend classifies a call argument by the argument *value's*
//!   type, so a frontend passing a struct by value bitcasts the address of the
//!   storage it filled to the struct type.
//!
//! **Scalars stay strictly typed** — only the pointer ↔ aggregate pairing is
//! newly compatible. The single predicate that encodes this is
//! [`addr_compatible`]; the base / address sites use [`is_aggregate`] alongside
//! [`is_ptr`].
//!
//! ## Address spaces
//!
//! A pointer lives in an address space (`ptr` is space 0, `ptr addrspace(N)`
//! space `N`; `docs/ir-design.md` §3a). The rules:
//!
//! - every address space a type mentions, and every global's space, must be
//!   declared by the module's [`DataLayout`](crate::ir::DataLayout);
//! - pointers of different spaces are **different types**: they never unify at
//!   a call/return boundary, a block argument, a `select` or an `icmp`, and
//!   there is **no `addrspacecast`** — `bitcast` between pointer types is
//!   rejected. Code that really means to reinterpret an address in another
//!   space says so with `ptrtoint` + `inttoptr`;
//! - `ptr_add` stays in its base's space (an aggregate base is a space-0
//!   address), and `alloca` / `dyn_alloca` produce space-0 pointers;
//! - a reference to a global is a pointer into the global's space, a reference
//!   to a function a pointer into the layout's program address space, and an
//!   indirect callee must be a pointer into the program address space;
//! - only a space-0 `ptr` is interchangeable with an aggregate value.

use crate::ir::inst::{AtomicOrdering, BinOp, CastOp, InstData, InstId, InstKind, RmwOp, UnaryOp};
use crate::ir::types::{FloatKind, Type, TypeId};
use crate::ir::value::{AddrTarget, Const, ConstId, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, GlobalId, Module};
use crate::support::diagnostics::Diagnostic;

use super::cfg::DomTree;

/// Verify one function of `module` in isolation, returning every structural /
/// type violation as an error [`Diagnostic`]. An empty result means the function
/// is well-formed at the structural tier.
///
/// External declarations (functions with no body) are checked only for a
/// well-formed signature.
pub fn verify_function(module: &Module, func: FuncId) -> Vec<Diagnostic> {
    let mut ctx = Ctx::new(module, func);
    ctx.run();
    ctx.diags
}

/// Verify every global's initializer, returning each violation as an error
/// [`Diagnostic`]: the initializer's type must equal the global's type, every
/// aggregate must match its array/struct shape element-by-element, scalar leaves
/// must be well-typed (`int`/`float`/`null`), and an address constant
/// ([`Const::Addr`]) must be pointer-typed and name an existing global or
/// function. Poison is accepted anywhere (it serializes as zero bytes).
pub fn verify_globals(module: &Module) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    for (gi, g) in module.globals().enumerate() {
        let space = module.global_addr_space(GlobalId::from_index(gi));
        if module.data_layout().pointer(space).is_none() {
            diags.push(global_err(
                gi,
                format!("lives in address space {space}, which the data layout does not declare"),
            ));
        }
        if let Some(s) = undeclared_space(module, g.ty) {
            diags.push(global_err(
                gi,
                format!("type {} uses address space {s}, which the data layout does not declare", render_type(module, g.ty)),
            ));
        }
        let Some(init) = g.init else { continue };
        let ity = module.consts().type_of(init);
        if ity != g.ty {
            diags.push(global_err(
                gi,
                format!(
                    "initializer has type {} but the global has type {}",
                    render_type(module, ity),
                    render_type(module, g.ty)
                ),
            ));
            continue;
        }
        check_init_const(module, gi, init, &mut diags);
    }
    diags
}

/// An error diagnostic about global `gi`.
fn global_err(gi: usize, msg: String) -> Diagnostic {
    Diagnostic::error(format!("global #{gi}: {msg}"))
}

/// Recursively check one global-initializer constant (see [`verify_globals`]).
fn check_init_const(m: &Module, gi: usize, cid: ConstId, diags: &mut Vec<Diagnostic>) {
    match m.consts().get(cid) {
        Const::Int { ty, .. } if !is_int(m, *ty) => diags.push(global_err(
            gi,
            format!("integer constant has non-integer type {}", render_type(m, *ty)),
        )),
        Const::Float { ty, .. } if !is_float(m, *ty) => diags.push(global_err(
            gi,
            format!("float constant has non-float type {}", render_type(m, *ty)),
        )),
        Const::Null(ty) if !is_ptr(m, *ty) => diags.push(global_err(
            gi,
            format!("null constant has non-pointer type {}", render_type(m, *ty)),
        )),
        Const::Int { .. } | Const::Float { .. } | Const::Null(_) | Const::Poison(_) => {}
        Const::Addr { ty, target, .. } => {
            if !is_ptr(m, *ty) {
                diags.push(global_err(
                    gi,
                    format!("address constant has non-pointer type {}", render_type(m, *ty)),
                ));
            }
            let exists = match target {
                AddrTarget::Global(g) => g.index() < m.global_count(),
                AddrTarget::Func(f) => f.index() < m.function_count(),
            };
            if !exists {
                diags.push(global_err(
                    gi,
                    format!("address constant names a nonexistent symbol ({target:?})"),
                ));
            } else if let Some(space) = m.types().addr_space(*ty) {
                // The address of a symbol is a pointer into the symbol's space.
                let want = match target {
                    AddrTarget::Global(g) => m.global_addr_space(*g),
                    AddrTarget::Func(_) => m.data_layout().program_addr_space(),
                };
                if space != want {
                    diags.push(global_err(
                        gi,
                        format!(
                            "address constant has type {} but its symbol lives in address space {want}",
                            render_type(m, *ty)
                        ),
                    ));
                }
            }
        }
        Const::Aggregate { ty, elems } => {
            let want: Vec<TypeId> = match m.types().get(*ty) {
                Type::Array(elem, n) if elems.len() as u64 == *n => vec![*elem; elems.len()],
                Type::Vector(elem, n) if elems.len() == *n as usize => vec![*elem; elems.len()],
                Type::Struct(fields) if elems.len() == fields.len() => fields.clone(),
                Type::Array(..) | Type::Struct(_) | Type::Vector(..) => {
                    diags.push(global_err(
                        gi,
                        format!(
                            "aggregate constant has {} element(s) but type {} disagrees",
                            elems.len(),
                            render_type(m, *ty)
                        ),
                    ));
                    return;
                }
                _ => {
                    diags.push(global_err(
                        gi,
                        format!("aggregate constant has non-aggregate type {}", render_type(m, *ty)),
                    ));
                    return;
                }
            };
            for (i, (&e, &w)) in elems.iter().zip(want.iter()).enumerate() {
                let et = m.consts().type_of(e);
                if et != w {
                    diags.push(global_err(
                        gi,
                        format!(
                            "aggregate element #{i} has type {} but expected {}",
                            render_type(m, et),
                            render_type(m, w)
                        ),
                    ));
                } else {
                    check_init_const(m, gi, e, diags);
                }
            }
        }
    }
}

/// Per-function verification state: the module and function under test, cheap
/// precomputed sizes, the dominator tree, an instruction-location map, and the
/// growing diagnostic list.
struct Ctx<'a> {
    module: &'a Module,
    func: &'a Function,
    func_id: FuncId,
    block_count: usize,
    func_count: usize,
    global_count: usize,
    /// The function's parameter types and return type, if its signature is a
    /// `Func` type; `None` (with a diagnostic already emitted) otherwise.
    sig: Option<(Vec<TypeId>, TypeId)>,
    domtree: DomTree,
    /// `inst_loc[i]` is `(block index, order)` of instruction `i`, where `order`
    /// is `0` for block parameters, `pos + 1` for the `pos`-th non-terminator,
    /// and `insts.len() + 1` for the terminator. `None` if the instruction is
    /// not placed in any block.
    inst_loc: Vec<Option<(usize, usize)>>,
    diags: Vec<Diagnostic>,
}

impl<'a> Ctx<'a> {
    fn new(module: &'a Module, func_id: FuncId) -> Ctx<'a> {
        let func = module.function(func_id);
        let block_count = func.block_count();
        let func_count = module.functions().count();
        let global_count = module.globals().count();

        let sig = match module.types().get(func.sig) {
            Type::Func(ft) => Some((ft.params.clone(), ft.ret)),
            _ => None,
        };

        // Locate every instruction in its block for dominance ordering.
        let mut inst_loc = vec![None; func.inst_count()];
        for b in 0..block_count {
            let block = func.block(BlockId::from_index(b));
            for (pos, &inst) in block.insts().iter().enumerate() {
                inst_loc[inst.index()] = Some((b, pos + 1));
            }
            if let Some(t) = block.terminator() {
                inst_loc[t.index()] = Some((b, block.insts().len() + 1));
            }
        }

        let domtree = DomTree::build(func);

        Ctx {
            module,
            func,
            func_id,
            block_count,
            func_count,
            global_count,
            sig,
            domtree,
            inst_loc,
            diags: Vec::new(),
        }
    }

    /// Record an error diagnostic, prefixed with the function under test.
    fn err(&mut self, msg: impl std::fmt::Display) {
        self.diags.push(Diagnostic::error(format!("function #{}: {}", self.func_id.index(), msg)));
    }

    fn run(&mut self) {
        if self.sig.is_none() {
            self.err("function signature is not a function type");
        }
        if let Some(bad) = first_bad_vector(self.module, self.func.sig) {
            self.err(format!("signature uses an invalid vector type {}", render_type(self.module, bad)));
        }
        // An external declaration (no blocks) needs no body checks.
        if self.func.is_declaration() {
            self.check_values();
            return;
        }
        self.check_entry();
        self.check_blocks();
        self.check_dominance();
        self.check_values();
    }

    // --- entry-block rules --------------------------------------------------

    fn check_entry(&mut self) {
        let func = self.func;
        match func.entry() {
            None => self.err("function has a body but no entry block"),
            Some(entry) => {
                let Some((params, _)) = self.sig.clone() else { return };
                let got = func.block(entry).params().to_vec();
                if got.len() != params.len() {
                    self.err(format!(
                        "entry block has {} parameter(s) but the signature declares {}",
                        got.len(),
                        params.len(),
                    ));
                    return;
                }
                for (i, (&pv, &want)) in got.iter().zip(params.iter()).enumerate() {
                    let have = func.value_type(pv);
                    if have != want {
                        let (a, b) = (render_type(self.module, have), render_type(self.module, want));
                        self.err(format!(
                            "entry parameter #{i} has type {a} but the signature declares {b}"
                        ));
                    }
                }
            }
        }
    }

    // --- blocks, terminators, per-instruction typing ------------------------

    fn check_blocks(&mut self) {
        let func = self.func;
        for b in 0..self.block_count {
            let block = func.block(BlockId::from_index(b));

            // Non-terminator slots must not hold terminators; type-check each.
            for &inst in block.insts() {
                if func.inst(inst).is_terminator() {
                    self.err(format!(
                        "block #{b}: terminator instruction #{} appears before the end of the block",
                        inst.index()
                    ));
                } else {
                    self.check_inst(b, inst);
                }
            }

            match block.terminator() {
                None => self.err(format!("block #{b} has no terminator")),
                Some(t) => {
                    if !func.inst(t).is_terminator() {
                        self.err(format!(
                            "block #{b}: terminator slot holds non-terminator instruction #{}",
                            t.index()
                        ));
                    }
                    self.check_inst(b, t);
                }
            }
        }
    }

    /// Type-check one instruction (terminator or not). Control-flow edge checks
    /// live in [`Ctx::check_edges`], invoked from here for terminators.
    fn check_inst(&mut self, block: usize, inst: InstId) {
        let func = self.func;
        let module = self.module;
        let data = func.inst(inst);
        let ops = data.operands();
        let ty = data.ty;

        match &data.kind {
            InstKind::Bin(op) => {
                if !self.arity(inst, ops, 2) {
                    return;
                }
                let (l, r) = (func.value_type(ops[0]), func.value_type(ops[1]));
                let float = op.is_float();
                let lane = module.types().scalar_of(ty);
                if float {
                    self.want_float(inst, lane, "binary result");
                } else {
                    self.want_int(inst, lane, "binary result");
                }
                if l != ty || r != ty {
                    self.err(format!(
                        "instruction #{}: {} operands ({}, {}) must match the result type {}",
                        inst.index(),
                        bin_name(*op),
                        render_type(module, l),
                        render_type(module, r),
                        render_type(module, ty),
                    ));
                }
            }
            InstKind::Unary(UnaryOp::FNeg) => {
                if !self.arity(inst, ops, 1) {
                    return;
                }
                self.want_float(inst, module.types().scalar_of(ty), "fneg result");
                let o = func.value_type(ops[0]);
                if o != ty {
                    self.type_mismatch(inst, "fneg operand", o, ty);
                }
            }
            InstKind::ICmp(_) => {
                if !self.arity(inst, ops, 2) {
                    return;
                }
                let (l, r) = (func.value_type(ops[0]), func.value_type(ops[1]));
                self.want_bool_like(inst, ty, l, "icmp result");
                let lane = module.types().scalar_of(l);
                if l != r {
                    self.type_mismatch(inst, "icmp operands", l, r);
                } else if !is_int(module, lane) && !is_ptr(module, l) {
                    self.err(format!(
                        "instruction #{}: icmp operands must be integer or pointer, found {}",
                        inst.index(),
                        render_type(module, l),
                    ));
                }
            }
            InstKind::FCmp(_) => {
                if !self.arity(inst, ops, 2) {
                    return;
                }
                let (l, r) = (func.value_type(ops[0]), func.value_type(ops[1]));
                self.want_bool_like(inst, ty, l, "fcmp result");
                if l != r {
                    self.type_mismatch(inst, "fcmp operands", l, r);
                } else if !is_float(module, module.types().scalar_of(l)) {
                    self.err(format!(
                        "instruction #{}: fcmp operands must be floating-point, found {}",
                        inst.index(),
                        render_type(module, l),
                    ));
                }
            }
            InstKind::Cast(op) => {
                if !self.arity(inst, ops, 1) {
                    return;
                }
                let from = func.value_type(ops[0]);
                self.check_cast(inst, *op, from, ty);
            }
            InstKind::Alloca { .. } => {
                self.arity(inst, ops, 0);
                if !is_ptr0(module, ty) {
                    self.err(format!(
                        "instruction #{}: alloca result must be a pointer (address space 0), found {}",
                        inst.index(),
                        render_type(module, ty),
                    ));
                }
            }
            InstKind::DynAlloca { align } => {
                if *align == 0 || !align.is_power_of_two() {
                    self.err(format!(
                        "instruction #{}: dyn_alloca alignment {align} must be a nonzero power of two",
                        inst.index()
                    ));
                }
                if !self.arity(inst, ops, 1) {
                    return;
                }
                let n = func.value_type(ops[0]);
                if !is_int(module, n) {
                    self.err(format!(
                        "instruction #{}: dyn_alloca size operand must be an integer, found {}",
                        inst.index(),
                        render_type(module, n),
                    ));
                }
                if !is_ptr0(module, ty) {
                    self.err(format!(
                        "instruction #{}: dyn_alloca result must be a pointer (address space 0), found {}",
                        inst.index(),
                        render_type(module, ty),
                    ));
                }
            }
            InstKind::Load { ty: acc, align, volatile, .. } => {
                self.check_align(inst, "load", *align);
                self.check_type_wf(inst, *acc);
                if *volatile && module.types().is_vector(*acc) {
                    self.err(format!("instruction #{}: a vector load cannot be volatile", inst.index()));
                }
                if !self.arity(inst, ops, 1) {
                    return;
                }
                let p = func.value_type(ops[0]);
                if !is_ptr(module, p) && !is_aggregate(module, p) {
                    self.err(format!(
                        "instruction #{}: load address operand must be a pointer or an aggregate value, found {}",
                        inst.index(),
                        render_type(module, p),
                    ));
                }
                if *acc != ty {
                    self.type_mismatch(inst, "load result vs. accessed type", ty, *acc);
                }
            }
            InstKind::Store { ty: acc, align, volatile, .. } => {
                self.check_align(inst, "store", *align);
                self.check_type_wf(inst, *acc);
                if *volatile && module.types().is_vector(*acc) {
                    self.err(format!("instruction #{}: a vector store cannot be volatile", inst.index()));
                }
                if !self.arity(inst, ops, 2) {
                    return;
                }
                let p = func.value_type(ops[0]);
                if !is_ptr(module, p) && !is_aggregate(module, p) {
                    self.err(format!(
                        "instruction #{}: store address operand must be a pointer or an aggregate value, found {}",
                        inst.index(),
                        render_type(module, p),
                    ));
                }
                let v = func.value_type(ops[1]);
                if v != *acc {
                    self.type_mismatch(inst, "stored value vs. accessed type", v, *acc);
                }
            }
            InstKind::AtomicLoad { ty: acc, align, ordering } => {
                self.check_atomic_type(inst, "atomic_load", *acc, *align, true);
                if !ordering.valid_for_load() {
                    self.bad_ordering(inst, "atomic_load", *ordering, "relaxed, acquire or seq_cst");
                }
                if !self.arity(inst, ops, 1) {
                    return;
                }
                self.check_atomic_addr(inst, "atomic_load", ops[0]);
                if *acc != ty {
                    self.type_mismatch(inst, "atomic_load result vs. accessed type", ty, *acc);
                }
            }
            InstKind::AtomicStore { ty: acc, align, ordering } => {
                self.check_atomic_type(inst, "atomic_store", *acc, *align, true);
                if !ordering.valid_for_store() {
                    self.bad_ordering(inst, "atomic_store", *ordering, "relaxed, release or seq_cst");
                }
                if !self.arity(inst, ops, 2) {
                    return;
                }
                self.check_atomic_addr(inst, "atomic_store", ops[0]);
                let v = func.value_type(ops[1]);
                if v != *acc {
                    self.type_mismatch(inst, "atomic_store value vs. accessed type", v, *acc);
                }
            }
            InstKind::AtomicRmw { op, ty: acc, align, .. } => {
                // Only `xchg` moves a pointer; the arithmetic ops are integer-only.
                let allow_ptr = *op == RmwOp::Xchg;
                self.check_atomic_type(inst, "atomic_rmw", *acc, *align, allow_ptr);
                if !self.arity(inst, ops, 2) {
                    return;
                }
                self.check_atomic_addr(inst, "atomic_rmw", ops[0]);
                let v = func.value_type(ops[1]);
                if v != *acc {
                    self.type_mismatch(inst, "atomic_rmw operand vs. accessed type", v, *acc);
                }
                if *acc != ty {
                    self.type_mismatch(inst, "atomic_rmw result vs. accessed type", ty, *acc);
                }
            }
            InstKind::CmpXchg { ty: acc, align, failure, .. } => {
                self.check_atomic_type(inst, "cmpxchg", *acc, *align, true);
                if !failure.valid_for_load() {
                    self.bad_ordering(inst, "cmpxchg failure", *failure, "relaxed, acquire or seq_cst");
                }
                if !self.arity(inst, ops, 3) {
                    return;
                }
                self.check_atomic_addr(inst, "cmpxchg", ops[0]);
                for (what, &o) in [("expected", &ops[1]), ("new", &ops[2])] {
                    let v = func.value_type(o);
                    if v != *acc {
                        self.type_mismatch(inst, &format!("cmpxchg {what} vs. accessed type"), v, *acc);
                    }
                }
                if *acc != ty {
                    self.type_mismatch(inst, "cmpxchg result vs. accessed type", ty, *acc);
                }
            }
            InstKind::Fence(ordering) => {
                if !ordering.valid_for_fence() {
                    self.bad_ordering(inst, "fence", *ordering, "acquire, release, acq_rel or seq_cst");
                }
                self.arity(inst, ops, 0);
            }
            InstKind::PtrAdd { .. } => {
                if !self.arity(inst, ops, 2) {
                    return;
                }
                // The base is an address: a pointer, or an aggregate value
                // (which denotes the address of its storage — see the module
                // docs on the struct-by-value convention).
                let base = func.value_type(ops[0]);
                if !is_ptr(module, base) && !is_aggregate(module, base) {
                    self.err(format!(
                        "instruction #{}: ptr_add base must be a pointer or an aggregate value, found {}",
                        inst.index(),
                        render_type(module, base),
                    ));
                }
                let off = func.value_type(ops[1]);
                if !is_int(module, off) {
                    self.err(format!(
                        "instruction #{}: ptr_add byte offset must be an integer, found {}",
                        inst.index(),
                        render_type(module, off),
                    ));
                }
                if !is_ptr(module, ty) {
                    self.err(format!(
                        "instruction #{}: ptr_add result must be a pointer",
                        inst.index()
                    ));
                } else if is_ptr(module, base) || is_aggregate(module, base) {
                    // Address arithmetic never leaves its address space.
                    let want = module.types().addr_space(base).unwrap_or(0);
                    let got = module.types().addr_space(ty).unwrap_or(0);
                    if got != want {
                        self.err(format!(
                            "instruction #{}: ptr_add result {} must stay in its base's address space {want}",
                            inst.index(),
                            render_type(module, ty),
                        ));
                    }
                }
            }
            InstKind::Select => {
                if !self.arity(inst, ops, 3) {
                    return;
                }
                let cond = func.value_type(ops[0]);
                // `i1` chooses a whole arm; `<N x i1>` chooses per lane of an
                // `N`-lane result.
                let per_lane = match (module.types().vector_parts(cond), module.types().vector_parts(ty)) {
                    (Some((c, n)), Some((_, m))) => is_bool(module, c) && n == m,
                    _ => false,
                };
                if !is_bool(module, cond) && !per_lane {
                    self.err(format!(
                        "instruction #{}: select condition must be i1 (or <N x i1> for an N-lane result), found {}",
                        inst.index(),
                        render_type(module, cond),
                    ));
                }
                let (t, f) = (func.value_type(ops[1]), func.value_type(ops[2]));
                if t != f {
                    self.type_mismatch(inst, "select arms", t, f);
                } else if t != ty {
                    self.type_mismatch(inst, "select result vs. arms", ty, t);
                }
            }
            InstKind::Freeze => {
                if !self.arity(inst, ops, 1) {
                    return;
                }
                let o = func.value_type(ops[0]);
                if o != ty {
                    self.type_mismatch(inst, "freeze operand vs. result", o, ty);
                }
            }
            InstKind::Declassify => {
                if !self.arity(inst, ops, 1) {
                    return;
                }
                let o = func.value_type(ops[0]);
                if o != ty {
                    self.type_mismatch(inst, "declassify operand vs. result", o, ty);
                }
            }
            InstKind::ExtractElement { lane } => {
                if !self.arity(inst, ops, 1) {
                    return;
                }
                let v = func.value_type(ops[0]);
                match module.types().vector_parts(v) {
                    Some((elem, n)) => {
                        self.check_lane(inst, "extractelement", *lane, n);
                        if elem != ty {
                            self.type_mismatch(inst, "extractelement result vs. element type", ty, elem);
                        }
                    }
                    None => self.not_vector(inst, "extractelement operand", v),
                }
            }
            InstKind::InsertElement { lane } => {
                if !self.arity(inst, ops, 2) {
                    return;
                }
                let v = func.value_type(ops[0]);
                match module.types().vector_parts(v) {
                    Some((elem, n)) => {
                        self.check_lane(inst, "insertelement", *lane, n);
                        let x = func.value_type(ops[1]);
                        if x != elem {
                            self.type_mismatch(inst, "insertelement value vs. element type", x, elem);
                        }
                        if v != ty {
                            self.type_mismatch(inst, "insertelement result vs. vector", ty, v);
                        }
                    }
                    None => self.not_vector(inst, "insertelement operand", v),
                }
            }
            InstKind::ShuffleVector(mask) => {
                if !self.arity(inst, ops, 2) {
                    return;
                }
                let (a, b) = (func.value_type(ops[0]), func.value_type(ops[1]));
                if a != b {
                    self.type_mismatch(inst, "shufflevector operands", a, b);
                    return;
                }
                let Some((elem, n)) = module.types().vector_parts(a) else {
                    self.not_vector(inst, "shufflevector operand", a);
                    return;
                };
                if let Some(&bad) = mask.iter().find(|&&m| u64::from(m) >= 2 * u64::from(n)) {
                    self.err(format!(
                        "instruction #{}: shufflevector mask index {bad} is out of range for two {n}-lane operands",
                        inst.index()
                    ));
                }
                let want_len = module.types().vector_parts(ty);
                if mask.is_empty() || want_len != Some((elem, mask.len() as u32)) {
                    self.err(format!(
                        "instruction #{}: shufflevector result must be <{} x {}>, found {}",
                        inst.index(),
                        mask.len(),
                        render_type(module, elem),
                        render_type(module, ty),
                    ));
                }
            }
            InstKind::Splat => {
                if !self.arity(inst, ops, 1) {
                    return;
                }
                let x = func.value_type(ops[0]);
                match module.types().vector_parts(ty) {
                    Some((elem, _)) if elem == x => {}
                    Some((elem, _)) => self.type_mismatch(inst, "splat operand vs. element type", x, elem),
                    None => self.not_vector(inst, "splat result", ty),
                }
            }
            InstKind::Reduce(op) => {
                if !self.arity(inst, ops, 1) {
                    return;
                }
                let v = func.value_type(ops[0]);
                let Some((elem, _)) = module.types().vector_parts(v) else {
                    self.not_vector(inst, "reduce operand", v);
                    return;
                };
                if elem != ty {
                    self.type_mismatch(inst, "reduce result vs. element type", ty, elem);
                }
                if op.is_float() {
                    self.want_float(inst, elem, "reduce lane");
                } else {
                    self.want_int(inst, elem, "reduce lane");
                }
            }
            InstKind::Call => self.check_call(inst, data),
            InstKind::Syscall => {
                // `[nr, args...]`: the number plus 0..=6 arguments (the Linux
                // register ABI on every target carries at most six).
                if ops.is_empty() || ops.len() > 7 {
                    self.err(format!(
                        "instruction #{}: syscall takes a number and 0..=6 arguments, found {} operand(s)",
                        inst.index(),
                        ops.len(),
                    ));
                }
                // Each operand fills one 64-bit register exactly: `i64` or
                // `ptr` (front ends extend narrower integers explicitly).
                for (i, &o) in ops.iter().enumerate() {
                    let ot = func.value_type(o);
                    if !is_ptr(module, ot) && !is_int_width(module, ot, 64) {
                        let what = if i == 0 { "number".to_string() } else { format!("argument {}", i - 1) };
                        self.err(format!(
                            "instruction #{}: syscall {what} must be i64 or ptr, found {}",
                            inst.index(),
                            render_type(module, ot),
                        ));
                    }
                }
                if !is_int_width(module, ty, 64) {
                    self.err(format!(
                        "instruction #{}: syscall result must be i64, found {}",
                        inst.index(),
                        render_type(module, ty),
                    ));
                }
            }
            InstKind::Ret => self.check_ret(inst, ops),
            InstKind::Br(_) | InstKind::CondBr { .. } | InstKind::Switch(_) => {
                self.check_terminator_conds(inst, data);
                self.check_edges(block, data);
            }
            InstKind::Unreachable => {
                self.arity(inst, ops, 0);
            }
        }
    }

    fn check_cast(&mut self, inst: InstId, op: CastOp, from: TypeId, to: TypeId) {
        let m = self.module;
        let types = m.types();
        let (fv, tv) = (types.vector_parts(from), types.vector_parts(to));
        let ok = if op == CastOp::Bitcast && (fv.is_some() || tv.is_some()) {
            // A vector bitcast reinterprets the packed lane bits: both sides are
            // integer/float scalars or vectors of the same total width.
            let bits_ok = |t: TypeId| {
                types.is_vector(t) || matches!(types.get(t), Type::Int(_) | Type::Float(_))
            };
            from != to
                && bits_ok(from)
                && bits_ok(to)
                && types.total_bits(from).is_some()
                && types.total_bits(from) == types.total_bits(to)
        } else {
            match (fv, tv) {
                // A lane-wise conversion: equal lane counts, the scalar rule on
                // the lanes.
                (Some((fe, fnum)), Some((te, tnum))) => fnum == tnum && scalar_cast_ok(m, op, fe, te),
                (None, None) => scalar_cast_ok(m, op, from, to),
                _ => false,
            }
        };
        if !ok {
            self.err(format!(
                "instruction #{}: {} from {} to {} is not a valid conversion",
                inst.index(),
                cast_name(op),
                render_type(m, from),
                render_type(m, to),
            ));
        }
    }

    /// A lane index must address one of the vector's `n` lanes.
    fn check_lane(&mut self, inst: InstId, op: &str, lane: u32, n: u32) {
        if lane >= n {
            self.err(format!(
                "instruction #{}: {op} lane {lane} is out of range for a {n}-lane vector",
                inst.index()
            ));
        }
    }

    fn not_vector(&mut self, inst: InstId, what: &str, t: TypeId) {
        self.err(format!(
            "instruction #{}: {what} must be a vector, found {}",
            inst.index(),
            render_type(self.module, t),
        ));
    }

    /// A comparison result: `i1` for scalar operands, `<N x i1>` for `N`-lane
    /// vector operands.
    fn want_bool_like(&mut self, inst: InstId, ty: TypeId, operand: TypeId, what: &str) {
        let types = self.module.types();
        let ok = match (types.vector_parts(operand), types.vector_parts(ty)) {
            (Some((_, n)), Some((b, m))) => n == m && is_bool(self.module, b),
            (None, None) => is_bool(self.module, ty),
            _ => false,
        };
        if !ok {
            let want = match types.vector_parts(operand) {
                Some((_, n)) => format!("<{n} x i1>"),
                None => "i1".to_string(),
            };
            self.err(format!(
                "instruction #{}: {what} must be {want}, found {}",
                inst.index(),
                render_type(self.module, ty),
            ));
        }
    }

    /// Every vector type reachable from `t` must be well-formed (see
    /// [`vector_wf`]).
    fn check_type_wf(&mut self, inst: InstId, t: TypeId) {
        if let Some(bad) = first_bad_vector(self.module, t) {
            self.err(format!(
                "instruction #{}: invalid vector type {} (lanes must number 1..=65536 and be i1, i8, i16, i32, i64 or a float)",
                inst.index(),
                render_type(self.module, bad),
            ));
        }
    }
}

/// The scalar (non-vector) cast rule: the conversions each [`CastOp`] allows.
fn scalar_cast_ok(m: &Module, op: CastOp, from: TypeId, to: TypeId) -> bool {
    {
        match op {
            CastOp::Trunc => matches!(
                (int_width(m, from), int_width(m, to)),
                (Some(a), Some(b)) if a > b
            ),
            CastOp::ZExt | CastOp::SExt => matches!(
                (int_width(m, from), int_width(m, to)),
                (Some(a), Some(b)) if a < b
            ),
            CastOp::FpTrunc => matches!(
                (float_width(m, from), float_width(m, to)),
                (Some(a), Some(b)) if a > b
            ),
            CastOp::FpExt => matches!(
                (float_width(m, from), float_width(m, to)),
                (Some(a), Some(b)) if a < b
            ),
            CastOp::FpToUi | CastOp::FpToSi => is_float(m, from) && is_int(m, to),
            CastOp::UiToFp | CastOp::SiToFp => is_int(m, from) && is_float(m, to),
            CastOp::PtrToInt => is_ptr(m, from) && is_int(m, to),
            CastOp::IntToPtr => is_int(m, from) && is_ptr(m, to),
            CastOp::Bitcast => {
                // Same-size reinterpretation; a pointer counts at its address
                // space's width from the data layout, so ptr<->iN(ptr width)
                // agree. Two pointer types never bitcast: they differ only in
                // address space, and there is no `addrspacecast` (see the
                // module docs). An aggregate value *is* the address of its
                // storage, so a space-0 `ptr` <-> aggregate is the identity on
                // that address: it is how a frontend turns storage it filled
                // into a by-value struct argument.
                (from != to
                    && !(is_ptr(m, from) && is_ptr(m, to))
                    && bit_size(m, from) == bit_size(m, to)
                    && bit_size(m, from).is_some())
                    || (is_ptr0(m, from) && is_aggregate(m, to))
                    || (is_aggregate(m, from) && is_ptr0(m, to))
            }
        }
    }
}

impl Ctx<'_> {

    fn check_call(&mut self, inst: InstId, data: &InstData) {
        let func = self.func;
        let module = self.module;
        let ops = data.operands();
        if ops.is_empty() {
            self.err(format!("instruction #{}: call has no callee operand", inst.index()));
            return;
        }
        let callee = ops[0];
        let args = &ops[1..];
        match &func.value(callee).def {
            ValueDef::Func(fid) if fid.index() < self.func_count => {
                let sig = module.function(*fid).sig;
                let Type::Func(ft) = module.types().get(sig) else {
                    self.err(format!(
                        "instruction #{}: callee function #{} has a non-function signature",
                        inst.index(),
                        fid.index(),
                    ));
                    return;
                };
                let arity_ok =
                    if ft.variadic { args.len() >= ft.params.len() } else { args.len() == ft.params.len() };
                if !arity_ok {
                    self.err(format!(
                        "instruction #{}: call passes {} argument(s) but callee #{} expects {}{}",
                        inst.index(),
                        args.len(),
                        fid.index(),
                        ft.params.len(),
                        if ft.variadic { "+" } else { "" },
                    ));
                }
                for (i, (&a, &p)) in args.iter().zip(ft.params.iter()).enumerate() {
                    let at = func.value_type(a);
                    // A `ptr` and an aggregate type are interchangeable across
                    // the ABI (the struct-by-value convention); scalars stay
                    // strict.
                    if !addr_compatible(module, at, p) {
                        let (x, y) = (render_type(module, at), render_type(module, p));
                        self.err(format!(
                            "instruction #{}: call argument #{i} has type {x} but callee expects {y}",
                            inst.index(),
                        ));
                    }
                }
                let ret_void = matches!(module.types().get(ft.ret), Type::Void);
                match data.result() {
                    None => {
                        if !ret_void {
                            self.err(format!(
                                "instruction #{}: call to non-void function #{} produces no result value",
                                inst.index(),
                                fid.index(),
                            ));
                        }
                    }
                    Some(r) => {
                        let rt = func.value_type(r);
                        if ret_void {
                            self.err(format!(
                                "instruction #{}: call to void function #{} produces a result value",
                                inst.index(),
                                fid.index(),
                            ));
                        } else if rt != ft.ret {
                            self.type_mismatch(inst, "call result vs. callee return", rt, ft.ret);
                        }
                    }
                }
            }
            ValueDef::Func(fid) => self.err(format!(
                "instruction #{}: call references nonexistent function #{}",
                inst.index(),
                fid.index(),
            )),
            _ => {
                // Indirect call: signature unknown, but the callee must be a
                // pointer into the program address space.
                let ct = func.value_type(callee);
                let program = module.data_layout().program_addr_space();
                if module.types().addr_space(ct) != Some(program) {
                    self.err(format!(
                        "instruction #{}: indirect call callee must be a pointer into the program address space {program}, found {}",
                        inst.index(),
                        render_type(module, ct),
                    ));
                }
            }
        }
    }

    fn check_ret(&mut self, inst: InstId, ops: &[ValueId]) {
        let Some((_, ret)) = self.sig.clone() else { return };
        let func = self.func;
        let ret_void = matches!(self.module.types().get(ret), Type::Void);
        if ret_void {
            if !ops.is_empty() {
                self.err(format!(
                    "instruction #{}: void function returns a value",
                    inst.index()
                ));
            }
        } else if ops.len() != 1 {
            self.err(format!(
                "instruction #{}: ret must supply exactly one value for a non-void function",
                inst.index()
            ));
        } else {
            let rt = func.value_type(ops[0]);
            // A `ptr` value satisfies an aggregate return type (and vice versa)
            // under the struct-by-value convention; scalars stay strict.
            if !addr_compatible(self.module, rt, ret) {
                self.type_mismatch(inst, "returned value vs. return type", rt, ret);
            }
        }
    }

    /// Per-terminator operand-typing conditions (the `i1` / integer condition
    /// rules); edge checks are separate.
    fn check_terminator_conds(&mut self, inst: InstId, data: &InstData) {
        let func = self.func;
        let module = self.module;
        let ops = data.operands();
        match &data.kind {
            InstKind::CondBr { .. } => {
                if ops.is_empty() {
                    self.err(format!("instruction #{}: cond_br has no condition", inst.index()));
                } else {
                    let c = func.value_type(ops[0]);
                    if !is_bool(module, c) {
                        self.err(format!(
                            "instruction #{}: cond_br condition must be i1, found {}",
                            inst.index(),
                            render_type(module, c),
                        ));
                    }
                }
            }
            InstKind::Switch(_) => {
                if ops.is_empty() {
                    self.err(format!("instruction #{}: switch has no condition", inst.index()));
                } else {
                    let c = func.value_type(ops[0]);
                    if !is_int(module, c) {
                        self.err(format!(
                            "instruction #{}: switch condition must be an integer, found {}",
                            inst.index(),
                            render_type(module, c),
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    /// Successor existence, the entry-block-predecessor rule, and per-edge
    /// block-argument arity and typing.
    fn check_edges(&mut self, block: usize, data: &InstData) {
        let func = self.func;
        let module = self.module;
        let ops = data.operands();
        let entry = func.entry();

        for (target, start, count) in edge_args(&data.kind, ops.len()) {
            if target.index() >= self.block_count {
                self.err(format!(
                    "block #{block}: terminator branches to nonexistent block #{}",
                    target.index()
                ));
                continue;
            }
            if entry == Some(target) {
                self.err(format!(
                    "block #{block}: terminator branches to the entry block #{}, whose parameters are the function parameters",
                    target.index()
                ));
            }
            let params = func.block(target).params().to_vec();
            if count != params.len() {
                self.err(format!(
                    "block #{block}: edge to block #{} passes {count} argument(s) but the block has {} parameter(s)",
                    target.index(),
                    params.len(),
                ));
            }
            let common = count.min(params.len());
            for (i, &param) in params.iter().take(common).enumerate() {
                let op_idx = start + i;
                if op_idx >= ops.len() {
                    break;
                }
                let arg_ty = func.value_type(ops[op_idx]);
                let param_ty = func.value_type(param);
                if arg_ty != param_ty {
                    let (a, b) = (render_type(module, arg_ty), render_type(module, param_ty));
                    self.err(format!(
                        "block #{block}: argument #{i} to block #{} has type {a} but the parameter is {b}",
                        target.index(),
                    ));
                }
            }
        }
    }

    // --- SSA dominance ------------------------------------------------------

    fn check_dominance(&mut self) {
        let func = self.func;
        for iid in 0..func.inst_count() {
            let inst = InstId::from_index(iid);
            let Some((ublock, useq)) = self.inst_loc[iid] else { continue };
            // Uses inside unreachable code cannot violate anything at runtime.
            if !self.domtree.is_reachable(ublock) {
                continue;
            }
            let data = func.inst(inst);
            let ops = data.operands().to_vec();
            for &op in &ops {
                self.check_use(inst, ublock, useq, op);
            }
        }
    }

    fn check_use(&mut self, inst: InstId, ublock: usize, useq: usize, op: ValueId) {
        let func = self.func;
        match &func.value(op).def {
            ValueDef::Const(_) | ValueDef::Global(_) | ValueDef::Func(_) => {}
            ValueDef::Param(dblock, _) => {
                let db = dblock.index();
                if db >= self.block_count {
                    return; // reported by check_values
                }
                // A parameter is defined at the top of its block, so it
                // dominates every use in that block and in dominated blocks.
                let ok = db == ublock || self.domtree.dominates(db, ublock);
                if !ok {
                    self.not_dominated(inst, ublock, op);
                }
            }
            ValueDef::Inst(dinst) => match self.inst_loc[dinst.index()] {
                None => self.err(format!(
                    "instruction #{}: operand is defined by instruction #{}, which is not placed in any block",
                    inst.index(),
                    dinst.index(),
                )),
                Some((dblock, dseq)) => {
                    let ok = if dblock == ublock {
                        dseq < useq
                    } else {
                        self.domtree.dominates(dblock, ublock)
                    };
                    if !ok {
                        self.not_dominated(inst, ublock, op);
                    }
                }
            },
        }
    }

    fn not_dominated(&mut self, inst: InstId, ublock: usize, op: ValueId) {
        self.err(format!(
            "instruction #{} in block #{ublock} uses value {} whose definition does not dominate the use (use before def or across non-dominating paths)",
            inst.index(),
            op.index(),
        ));
    }

    // --- values: constants and references -----------------------------------

    /// A reference to a symbol living in address space `space` must be typed
    /// as a pointer into that space.
    fn check_ref_type(&mut self, v: ValueId, ty: TypeId, space: u32, what: &str) {
        if self.module.types().addr_space(ty) != Some(space) {
            self.err(format!(
                "value {}: a {what} reference must be a pointer into address space {space}, found {}",
                v.index(),
                render_type(self.module, ty),
            ));
        }
    }

    fn check_values(&mut self) {
        let func = self.func;
        let module = self.module;
        for vi in 0..func.value_count() {
            let v = ValueId::from_index(vi);
            let val = func.value(v).clone();
            if let Some(s) = undeclared_space(module, val.ty) {
                self.err(format!(
                    "value {}: type {} uses address space {s}, which the data layout does not declare",
                    v.index(),
                    render_type(module, val.ty),
                ));
            }
            if let Some(bad) = first_bad_vector(module, val.ty) {
                self.err(format!(
                    "value {}: invalid vector type {} (lanes must number 1..=65536 and be i1, i8, i16, i32, i64 or a float)",
                    v.index(),
                    render_type(module, bad),
                ));
            }
            match &val.def {
                ValueDef::Const(cid) => {
                    let c = module.consts().get(*cid).clone();
                    if c.type_id() != val.ty {
                        self.type_mismatch_val(v, "constant value vs. constant type", val.ty, c.type_id());
                    }
                    self.check_const(v, &c);
                }
                ValueDef::Func(fid) => {
                    if fid.index() >= self.func_count {
                        self.err(format!(
                            "value {}: references nonexistent function #{}",
                            v.index(),
                            fid.index()
                        ));
                    } else {
                        let want = module.data_layout().program_addr_space();
                        self.check_ref_type(v, val.ty, want, "function");
                    }
                }
                ValueDef::Global(gid) => {
                    if gid.index() >= self.global_count {
                        self.err(format!(
                            "value {}: references nonexistent global #{}",
                            v.index(),
                            gid.index()
                        ));
                    } else {
                        self.check_ref_type(v, val.ty, module.global_addr_space(*gid), "global");
                    }
                }
                ValueDef::Param(b, idx) => {
                    if b.index() >= self.block_count {
                        self.err(format!(
                            "value {}: parameter of nonexistent block #{}",
                            v.index(),
                            b.index()
                        ));
                    } else {
                        let ps = func.block(*b).params();
                        if (*idx as usize) >= ps.len() || ps[*idx as usize] != v {
                            self.err(format!(
                                "value {}: block-parameter definition is inconsistent with its block",
                                v.index()
                            ));
                        }
                    }
                }
                ValueDef::Inst(iid) => {
                    if iid.index() >= func.inst_count() {
                        self.err(format!(
                            "value {}: defined by nonexistent instruction #{}",
                            v.index(),
                            iid.index()
                        ));
                    } else {
                        let d = func.inst(*iid);
                        if d.result() != Some(v) {
                            self.err(format!(
                                "value {}: instruction #{} does not define it",
                                v.index(),
                                iid.index()
                            ));
                        } else if d.ty != val.ty {
                            self.type_mismatch_val(v, "value type vs. defining instruction", val.ty, d.ty);
                        }
                    }
                }
            }
        }
    }

    fn check_const(&mut self, v: ValueId, c: &Const) {
        let m = self.module;
        match c {
            Const::Int { ty, .. } => {
                if !is_int(m, *ty) {
                    self.err(format!("value {}: integer constant has non-integer type {}", v.index(), render_type(m, *ty)));
                }
            }
            Const::Float { ty, .. } => {
                if !is_float(m, *ty) {
                    self.err(format!("value {}: float constant has non-float type {}", v.index(), render_type(m, *ty)));
                }
            }
            Const::Null(ty) => {
                if !is_ptr(m, *ty) {
                    self.err(format!("value {}: null constant has non-pointer type {}", v.index(), render_type(m, *ty)));
                }
            }
            Const::Poison(_) => {}
            Const::Aggregate { ty, elems } => match m.types().get(*ty) {
                Type::Array(elem, n) => {
                    if elems.len() as u64 != *n {
                        self.err(format!(
                            "value {}: array constant has {} element(s) but type expects {}",
                            v.index(),
                            elems.len(),
                            n
                        ));
                    }
                    let elem = *elem;
                    for (i, &e) in elems.iter().enumerate() {
                        let et = m.consts().type_of(e);
                        if et != elem {
                            self.err(format!(
                                "value {}: array element #{i} has type {} but expected {}",
                                v.index(),
                                render_type(m, et),
                                render_type(m, elem),
                            ));
                        }
                    }
                }
                Type::Struct(fields) => {
                    let fields = fields.clone();
                    if elems.len() != fields.len() {
                        self.err(format!(
                            "value {}: struct constant has {} field(s) but type expects {}",
                            v.index(),
                            elems.len(),
                            fields.len()
                        ));
                    }
                    for (i, (&e, &f)) in elems.iter().zip(fields.iter()).enumerate() {
                        let et = m.consts().type_of(e);
                        if et != f {
                            self.err(format!(
                                "value {}: struct field #{i} has type {} but expected {}",
                                v.index(),
                                render_type(m, et),
                                render_type(m, f),
                            ));
                        }
                    }
                }
                // A vector constant is a first-class operand: one scalar
                // (integer / float / poison) constant per lane.
                Type::Vector(elem, n) => {
                    let (elem, n) = (*elem, *n);
                    if elems.len() != n as usize {
                        self.err(format!(
                            "value {}: vector constant has {} lane(s) but type expects {n}",
                            v.index(),
                            elems.len(),
                        ));
                    }
                    for (i, &e) in elems.iter().enumerate() {
                        let lc = m.consts().get(e);
                        let scalar =
                            matches!(lc, Const::Int { .. } | Const::Float { .. } | Const::Poison(_));
                        if lc.type_id() != elem || !scalar {
                            self.err(format!(
                                "value {}: vector lane #{i} must be a {} constant",
                                v.index(),
                                render_type(m, elem),
                            ));
                        } else {
                            self.check_const(v, lc);
                        }
                    }
                }
                _ => self.err(format!(
                    "value {}: aggregate constant has non-aggregate type {}",
                    v.index(),
                    render_type(m, *ty)
                )),
            },
            Const::Addr { .. } => self.err(format!(
                "value {}: address constants are only allowed in global initializers",
                v.index()
            )),
        }
    }

    // --- small typing helpers -----------------------------------------------

    /// Check an operand count, reporting a mismatch. Returns whether it held.
    fn arity(&mut self, inst: InstId, ops: &[ValueId], want: usize) -> bool {
        if ops.len() != want {
            self.err(format!(
                "instruction #{}: expected {want} operand(s), found {}",
                inst.index(),
                ops.len()
            ));
            false
        } else {
            true
        }
    }

    fn want_int(&mut self, inst: InstId, ty: TypeId, what: &str) {
        if !is_int(self.module, ty) {
            self.err(format!(
                "instruction #{}: {what} must be an integer, found {}",
                inst.index(),
                render_type(self.module, ty),
            ));
        }
    }

    fn want_float(&mut self, inst: InstId, ty: TypeId, what: &str) {
        if !is_float(self.module, ty) {
            self.err(format!(
                "instruction #{}: {what} must be floating-point, found {}",
                inst.index(),
                render_type(self.module, ty),
            ));
        }
    }

    fn check_align(&mut self, inst: InstId, op: &str, align: u32) {
        if align == 0 || !align.is_power_of_two() {
            self.err(format!(
                "instruction #{}: {op} alignment {align} must be a nonzero power of two",
                inst.index()
            ));
        }
    }

    /// An atomic access type must be `i8`/`i16`/`i32`/`i64` (or `ptr` where
    /// `allow_ptr`), with an alignment that is a power of two and at least the
    /// type's size (natural alignment).
    fn check_atomic_type(&mut self, inst: InstId, op: &str, acc: TypeId, align: u32, allow_ptr: bool) {
        let m = self.module;
        let ok_int = matches!(m.types().get(acc), Type::Int(8 | 16 | 32 | 64));
        if !(ok_int || (allow_ptr && is_ptr(m, acc))) {
            let allowed = if allow_ptr { "i8, i16, i32, i64 or ptr" } else { "i8, i16, i32 or i64" };
            self.err(format!(
                "instruction #{}: {op} accesses {}, but atomics support only {allowed}",
                inst.index(),
                render_type(m, acc),
            ));
            return;
        }
        self.check_align(inst, op, align);
        let size = m.types().size_of(acc);
        if u64::from(align) < size {
            self.err(format!(
                "instruction #{}: {op} alignment {align} is below the natural alignment {size} of {}",
                inst.index(),
                render_type(m, acc),
            ));
        }
    }

    /// An atomic's address operand must be a `ptr`.
    fn check_atomic_addr(&mut self, inst: InstId, op: &str, addr: ValueId) {
        let p = self.func.value_type(addr);
        if !is_ptr(self.module, p) {
            self.err(format!(
                "instruction #{}: {op} address operand must be a pointer, found {}",
                inst.index(),
                render_type(self.module, p),
            ));
        }
    }

    fn bad_ordering(&mut self, inst: InstId, op: &str, o: AtomicOrdering, allowed: &str) {
        self.err(format!(
            "instruction #{}: {op} ordering `{}` is invalid (allowed: {allowed})",
            inst.index(),
            o.name(),
        ));
    }

    fn type_mismatch(&mut self, inst: InstId, what: &str, a: TypeId, b: TypeId) {
        let (x, y) = (render_type(self.module, a), render_type(self.module, b));
        self.err(format!("instruction #{}: {what}: {x} vs. {y}", inst.index()));
    }

    fn type_mismatch_val(&mut self, v: ValueId, what: &str, a: TypeId, b: TypeId) {
        let (x, y) = (render_type(self.module, a), render_type(self.module, b));
        self.err(format!("value {}: {what}: {x} vs. {y}", v.index()));
    }
}

// --- free type helpers ------------------------------------------------------

/// The `(target, operand start, arg count)` of every outgoing edge of a
/// terminator, in `successors()` order. Non-branch terminators yield none.
fn edge_args(kind: &InstKind, num_operands: usize) -> Vec<(BlockId, usize, usize)> {
    match kind {
        InstKind::Br(t) => vec![(*t, 0, num_operands)],
        InstKind::CondBr { if_true, if_false, true_args, false_args } => {
            let ta = *true_args as usize;
            let fa = *false_args as usize;
            vec![(*if_true, 1, ta), (*if_false, 1 + ta, fa)]
        }
        InstKind::Switch(data) => {
            let mut out = Vec::with_capacity(1 + data.cases.len());
            let mut cursor = 1 + data.default_args as usize;
            out.push((data.default, 1, data.default_args as usize));
            for c in &data.cases {
                out.push((c.target, cursor, c.args as usize));
                cursor += c.args as usize;
            }
            out
        }
        _ => Vec::new(),
    }
}

fn render_type(m: &Module, t: TypeId) -> String {
    match m.types().get(t) {
        Type::Void => "void".to_string(),
        Type::Int(w) => format!("i{w}"),
        Type::Float(FloatKind::F16) => "f16".to_string(),
        Type::Float(FloatKind::F32) => "f32".to_string(),
        Type::Float(FloatKind::F64) => "f64".to_string(),
        Type::Ptr => "ptr".to_string(),
        Type::PtrIn(space) => format!("ptr addrspace({space})"),
        Type::Array(e, n) => format!("[{n} x {}]", render_type(m, *e)),
        Type::Struct(fs) => {
            let inner: Vec<String> = fs.iter().map(|&f| render_type(m, f)).collect();
            format!("{{{}}}", inner.join(", "))
        }
        Type::Func(_) => "func".to_string(),
        Type::Vector(e, n) => format!("<{n} x {}>", render_type(m, *e)),
    }
}

/// Whether a vector type is well-formed: `1..=65536` lanes of `i1`, `i8`,
/// `i16`, `i32`, `i64` or a float type.
fn vector_wf(m: &Module, t: TypeId) -> bool {
    match m.types().get(t) {
        Type::Vector(elem, n) => {
            (1..=65536).contains(n)
                && matches!(m.types().get(*elem), Type::Int(1 | 8 | 16 | 32 | 64) | Type::Float(_))
        }
        _ => true,
    }
}

/// The first ill-formed vector type within `t` (itself, or an array/struct
/// component or signature part), if any.
fn first_bad_vector(m: &Module, t: TypeId) -> Option<TypeId> {
    if !vector_wf(m, t) {
        return Some(t);
    }
    match m.types().get(t) {
        Type::Array(e, _) => first_bad_vector(m, *e),
        Type::Struct(fs) => fs.iter().find_map(|&f| first_bad_vector(m, f)),
        Type::Func(ft) => {
            ft.params.iter().find_map(|&p| first_bad_vector(m, p)).or_else(|| first_bad_vector(m, ft.ret))
        }
        _ => None,
    }
}

fn is_int(m: &Module, t: TypeId) -> bool {
    matches!(m.types().get(t), Type::Int(_))
}

fn is_int_width(m: &Module, t: TypeId, width: u32) -> bool {
    matches!(m.types().get(t), Type::Int(w) if *w == width)
}

fn is_bool(m: &Module, t: TypeId) -> bool {
    matches!(m.types().get(t), Type::Int(1))
}

fn is_float(m: &Module, t: TypeId) -> bool {
    matches!(m.types().get(t), Type::Float(_))
}

/// A pointer in any address space.
fn is_ptr(m: &Module, t: TypeId) -> bool {
    m.types().is_ptr(t)
}

/// A pointer in the default address space 0 (`ptr`).
fn is_ptr0(m: &Module, t: TypeId) -> bool {
    matches!(m.types().get(t), Type::Ptr)
}

/// The first address space mentioned by `t` (through arrays, structs and
/// function signatures) that the module's data layout does not declare.
fn undeclared_space(m: &Module, t: TypeId) -> Option<u32> {
    match m.types().get(t) {
        Type::PtrIn(s) => m.data_layout().pointer(*s).is_none().then_some(*s),
        Type::Array(e, _) => undeclared_space(m, *e),
        Type::Struct(fs) => fs.iter().find_map(|&f| undeclared_space(m, f)),
        Type::Func(ft) => {
            ft.params.iter().chain(std::iter::once(&ft.ret)).find_map(|&p| undeclared_space(m, p))
        }
        _ => None,
    }
}

/// An aggregate type (`Struct`/`Array`). A value of such a type denotes the
/// address of its storage, so it is usable as a pointer (see the module docs).
fn is_aggregate(m: &Module, t: TypeId) -> bool {
    matches!(m.types().get(t), Type::Struct(_) | Type::Array(..))
}

/// Whether types `a` and `b` are **address-compatible** under the struct-by-value
/// convention: they are equal (two pointers are equal iff they share an address
/// space), or one is a space-0 `ptr` and the other an aggregate (whose value
/// *is* an address). This is the *only* relaxation over exact type equality —
/// scalars (`Int`/`Float`) remain strictly typed. Used at the ABI boundaries
/// (`call` arguments and `ret`).
fn addr_compatible(m: &Module, a: TypeId, b: TypeId) -> bool {
    a == b
        || (is_ptr0(m, a) && is_aggregate(m, b))
        || (is_aggregate(m, a) && is_ptr0(m, b))
}

fn int_width(m: &Module, t: TypeId) -> Option<u32> {
    match m.types().get(t) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}

fn float_width(m: &Module, t: TypeId) -> Option<u32> {
    match m.types().get(t) {
        Type::Float(k) => Some(k.bit_width()),
        _ => None,
    }
}

/// The bit size of a bit-reinterpretable type: scalar width, or the data
/// layout's width of the pointer's address space. Aggregates and functions have
/// no single bit width here.
fn bit_size(m: &Module, t: TypeId) -> Option<u32> {
    match m.types().get(t) {
        Type::Int(w) => Some(*w),
        Type::Float(k) => Some(k.bit_width()),
        _ => m.types().pointer_bits(t),
    }
}

fn bin_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::UDiv => "udiv",
        BinOp::SDiv => "sdiv",
        BinOp::URem => "urem",
        BinOp::SRem => "srem",
        BinOp::And => "and",
        BinOp::Or => "or",
        BinOp::Xor => "xor",
        BinOp::Shl => "shl",
        BinOp::LShr => "lshr",
        BinOp::AShr => "ashr",
        BinOp::FAdd => "fadd",
        BinOp::FSub => "fsub",
        BinOp::FMul => "fmul",
        BinOp::FDiv => "fdiv",
        BinOp::FRem => "frem",
    }
}

fn cast_name(op: CastOp) -> &'static str {
    match op {
        CastOp::Trunc => "trunc",
        CastOp::ZExt => "zext",
        CastOp::SExt => "sext",
        CastOp::FpTrunc => "fptrunc",
        CastOp::FpExt => "fpext",
        CastOp::FpToUi => "fptoui",
        CastOp::FpToSi => "fptosi",
        CastOp::UiToFp => "uitofp",
        CastOp::SiToFp => "sitofp",
        CastOp::PtrToInt => "ptrtoint",
        CastOp::IntToPtr => "inttoptr",
        CastOp::Bitcast => "bitcast",
    }
}

// ---------------------------------------------------------------------------
// Tests for the struct-by-value (aggregate-value-as-address) convention.
//
// These prove the four relaxed sites accept the backend's gcc-ABI form (an
// aggregate value used as an address / across the call & return boundary), while
// a genuine scalar mismatch is still rejected — i.e. only pointer ↔ aggregate is
// newly compatible.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod struct_by_value_tests {
    use crate::ir::inst::Flags;
    use crate::verify::verify_module;
    use crate::ir::{FuncId, Module};
    use crate::support::StrInterner;

    /// `struct P { i32 x, y; } addP(struct P, struct P)` returning
    /// `{a.x+b.x, a.y+b.y}`, exactly the shape of `build_struct_int` in the
    /// x86-64 backend tests: reads each field via `struct_field` (whose base is
    /// the struct-typed *parameter value*) + `load`, `alloca`s a result, stores
    /// into it, and `ret`s the `alloca` pointer where the return type is the
    /// struct. Exercises: `ptr_add`/`struct_field` base = aggregate, `load`
    /// address = aggregate, `store` address = ptr, and `ret` ptr vs. aggregate.
    fn build_addp() -> (Module, StrInterner, FuncId) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("t");
        let i32t = m.types_mut().int(32);
        let p = m.types_mut().struct_(vec![i32t, i32t]);
        let sig = m.types_mut().func(vec![p, p], p, false);
        let f = m.declare_function(syms.intern("addP"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let a = b.param(entry, 0);
            let bb = b.param(entry, 1);
            let ax_p = b.struct_field(a, p, 0);
            let ax = b.load(i32t, ax_p, 4);
            let ay_p = b.struct_field(a, p, 1);
            let ay = b.load(i32t, ay_p, 4);
            let bx_p = b.struct_field(bb, p, 0);
            let bx = b.load(i32t, bx_p, 4);
            let by_p = b.struct_field(bb, p, 1);
            let by = b.load(i32t, by_p, 4);
            let sx = b.add(ax, bx, Flags::NONE);
            let sy = b.add(ay, by, Flags::NONE);
            let r = b.alloca(p);
            let rx = b.struct_field(r, p, 0);
            b.store(i32t, rx, sx, 4);
            let ry = b.struct_field(r, p, 1);
            b.store(i32t, ry, sy, 4);
            b.ret(Some(r)); // ptr value, aggregate return type
        }
        (m, syms, f)
    }

    #[test]
    fn struct_by_value_addp_verifies() {
        let (m, _syms, _f) = build_addp();
        assert!(
            verify_module(&m).is_ok(),
            "the gcc-ABI struct-by-value form must verify: {:?}",
            verify_module(&m).err()
        );
    }

    #[test]
    fn ptr_passed_where_struct_param_expected_verifies() {
        // A caller that `alloca`s two `P`s (pointers) and calls `addP` passing
        // those pointers where the parameters are declared as the struct type —
        // the call-argument ptr ↔ aggregate relaxation — then returns the struct
        // result. Verifies clean.
        let (mut m, mut syms, addp) = build_addp();
        let i32t = m.types_mut().int(32);
        let p = m.types_mut().struct_(vec![i32t, i32t]);
        let sig = m.types_mut().func(vec![], p, false);
        let caller = m.declare_function(syms.intern("call_addP"), sig);
        {
            let mut b = m.build(caller);
            b.create_entry_block();
            let a = b.alloca(p); // ptr
            let bb = b.alloca(p); // ptr
            let cref = b.func_ref(addp);
            let r = b.call(cref, &[a, bb], p).expect("addP returns a value");
            b.ret(Some(r));
        }
        assert!(verify_module(&m).is_ok(), "ptr-as-struct-arg must verify: {:?}", verify_module(&m).err());
    }

    #[test]
    fn scalar_return_mismatch_still_rejected() {
        // Returning an `i32` where the return type is `i64` is a real scalar
        // mismatch — the relaxation must NOT cover it.
        let mut syms = StrInterner::new();
        let mut m = Module::new("t");
        let i32t = m.types_mut().int(32);
        let i64t = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![], i64t, false);
        let f = m.declare_function(syms.intern("bad_ret"), sig);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let c = b.const_i64(i32t, 7); // i32 constant
            b.ret(Some(c));
        }
        let diags = verify_module(&m).expect_err("i32 vs i64 return must be rejected");
        assert!(
            diags.iter().any(|d| d.message.contains("returned value vs. return type")),
            "expected a return-type mismatch, got: {:?}",
            diags.iter().map(|d| d.message.as_str()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn scalar_call_arg_mismatch_still_rejected() {
        // Passing an `i32` where the callee expects an `i64` is a real scalar
        // mismatch — still rejected.
        let mut syms = StrInterner::new();
        let mut m = Module::new("t");
        let i32t = m.types_mut().int(32);
        let i64t = m.types_mut().int(64);
        let callee_sig = m.types_mut().func(vec![i64t], i64t, false);
        let callee = m.declare_function(syms.intern("callee"), callee_sig);
        {
            let mut b = m.build(callee);
            let entry = b.create_entry_block();
            let y = b.param(entry, 0);
            b.ret(Some(y));
        }
        let caller_sig = m.types_mut().func(vec![], i64t, false);
        let caller = m.declare_function(syms.intern("caller"), caller_sig);
        {
            let mut b = m.build(caller);
            b.create_entry_block();
            let bad = b.const_i64(i32t, 1); // i32 arg
            let cref = b.func_ref(callee);
            let r = b.call(cref, &[bad], i64t).expect("callee returns a value");
            b.ret(Some(r));
        }
        let diags = verify_module(&m).expect_err("i32 arg vs i64 param must be rejected");
        assert!(
            diags.iter().any(|d| d.message.contains("call argument #0")),
            "expected a call-argument type mismatch, got: {:?}",
            diags.iter().map(|d| d.message.as_str()).collect::<Vec<_>>()
        );
    }
}

// ---------------------------------------------------------------------------
// Tests for the dynamic (runtime-sized) stack allocation op `dyn_alloca`.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod dyn_alloca_tests {
    use crate::ir::Module;
    use crate::support::StrInterner;
    use crate::verify::verify_module;

    #[test]
    fn valid_dyn_alloca_verifies() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("t");
        let i64t = m.types_mut().int(64);
        let ptr = m.types_mut().ptr();
        let sig = m.types_mut().func(vec![i64t], ptr, false);
        let f = m.declare_function(syms.intern("d"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let n = b.param(e, 0);
            let p = b.dyn_alloca(n, 16);
            b.ret(Some(p));
        }
        assert!(
            verify_module(&m).is_ok(),
            "valid dyn_alloca must verify: {:?}",
            verify_module(&m).err()
        );
    }

    #[test]
    fn non_integer_size_rejected() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("t");
        let i8t = m.types_mut().int(8);
        let ptr = m.types_mut().ptr();
        let sig = m.types_mut().func(vec![], ptr, false);
        let f = m.declare_function(syms.intern("bad"), sig);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let a = b.alloca(i8t); // a pointer value
            let p = b.dyn_alloca(a, 16); // size operand is a pointer -> invalid
            b.ret(Some(p));
        }
        let diags = verify_module(&m).expect_err("non-integer size must be rejected");
        assert!(
            diags.iter().any(|d| d.message.contains("dyn_alloca size operand must be an integer")),
            "expected a size-type diagnostic, got: {:?}",
            diags.iter().map(|d| d.message.as_str()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn non_power_of_two_align_rejected() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("t");
        let i64t = m.types_mut().int(64);
        let ptr = m.types_mut().ptr();
        let sig = m.types_mut().func(vec![i64t], ptr, false);
        let f = m.declare_function(syms.intern("a3"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let n = b.param(e, 0);
            let p = b.dyn_alloca(n, 3); // align 3 is not a power of two
            b.ret(Some(p));
        }
        let diags = verify_module(&m).expect_err("non-power-of-two align must be rejected");
        assert!(
            diags.iter().any(|d| d.message.contains("must be a nonzero power of two")),
            "expected an alignment diagnostic, got: {:?}",
            diags.iter().map(|d| d.message.as_str()).collect::<Vec<_>>()
        );
    }
}

// ---------------------------------------------------------------------------
// Tests for the operating-system call op `syscall`.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod syscall_tests {
    use crate::ir::builder::FunctionBuilder;
    use crate::ir::inst::{CastOp, Flags, InstKind};
    use crate::ir::types::{FloatKind, TypeId};
    use crate::ir::{Module, ValueId};
    use crate::support::StrInterner;
    use crate::verify::verify_module;

    /// Build `s(i64, ptr, i32, f64) -> i64` whose body is `body(b, params,
    /// [i64, ptr])` (its value is returned) and collect the verifier's messages
    /// (empty when the module verifies).
    fn check(
        body: impl FnOnce(&mut FunctionBuilder<'_>, &[ValueId], [TypeId; 2]) -> ValueId,
    ) -> Vec<String> {
        let mut syms = StrInterner::new();
        let mut m = Module::new("t");
        let i64t = m.types_mut().int(64);
        let i32t = m.types_mut().int(32);
        let f64t = m.types_mut().float(FloatKind::F64);
        let ptr = m.types_mut().ptr();
        let sig = m.types_mut().func(vec![i64t, ptr, i32t, f64t], i64t, false);
        let f = m.declare_function(syms.intern("s"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let params: Vec<ValueId> = (0..4).map(|i| b.param(e, i)).collect();
            let r = body(&mut b, &params, [i64t, ptr]);
            b.ret(Some(r));
        }
        match verify_module(&m) {
            Ok(()) => Vec::new(),
            Err(diags) => diags.into_iter().map(|d| d.message).collect(),
        }
    }

    #[test]
    fn valid_syscalls_verify() {
        let diags = check(|b, p, _| {
            b.syscall(p[0], &[]);
            b.syscall(p[0], &[p[0], p[1], p[0], p[1], p[0], p[1]])
        });
        assert!(diags.is_empty(), "{diags:?}");
    }

    #[test]
    fn more_than_six_arguments_rejected() {
        let diags = check(|b, p, _| b.syscall(p[0], &[p[0]; 7]));
        assert!(
            diags.iter().any(|d| d.contains("syscall takes a number and 0..=6 arguments, found 8")),
            "{diags:?}"
        );
    }

    #[test]
    fn missing_number_rejected() {
        let diags = check(|b, _, [i64t, _]| {
            b.append_inst(InstKind::Syscall, vec![], Flags::NONE, Some(i64t)).unwrap()
        });
        assert!(diags.iter().any(|d| d.contains("found 0 operand(s)")), "{diags:?}");
    }

    #[test]
    fn narrow_or_float_operands_rejected() {
        // An i32 number and an f64 argument: neither is i64/ptr (there is no
        // implicit extension — front ends `zext`/`sext` explicitly).
        let diags = check(|b, p, _| b.syscall(p[2], &[p[3]]));
        assert!(
            diags.iter().any(|d| d.contains("syscall number must be i64 or ptr, found i32")),
            "{diags:?}"
        );
        assert!(
            diags.iter().any(|d| d.contains("syscall argument 0 must be i64 or ptr, found f64")),
            "{diags:?}"
        );
    }

    #[test]
    fn non_i64_result_rejected() {
        let diags = check(|b, p, [i64t, ptr]| {
            let r = b.append_inst(InstKind::Syscall, vec![p[0]], Flags::NONE, Some(ptr)).unwrap();
            b.cast(CastOp::PtrToInt, r, i64t)
        });
        assert!(diags.iter().any(|d| d.contains("syscall result must be i64, found ptr")), "{diags:?}");
    }
}

// ---------------------------------------------------------------------------
// Tests for volatile accesses, atomics and fences.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod atomic_tests {
    use crate::support::StrInterner;
    use crate::support::diagnostics::FileId;
    use crate::verify::verify_module;

    /// Parse a one-block function `f(ptr %p, i32 %v, i64 %w) -> void` whose
    /// body is `body` and return the verifier's messages (empty when valid).
    /// Parsing must succeed: these are type/ordering errors only the verifier
    /// catches.
    fn check(body: &str) -> Vec<String> {
        let src = format!(
            "module \"t\"\nfunc @f(ptr, i32, i64) -> void {{\nentry ^0(%p: ptr, %v: i32, %w: i64):\n{body}\n  ret\n}}\n"
        );
        let mut syms = StrInterner::new();
        let m = crate::ir::text::parse_module(&src, FileId::new(0), &mut syms)
            .unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
        match verify_module(&m) {
            Ok(()) => Vec::new(),
            Err(diags) => diags.into_iter().map(|d| d.message).collect(),
        }
    }

    fn assert_rejects(body: &str, needle: &str) {
        let diags = check(body);
        assert!(diags.iter().any(|d| d.contains(needle)), "`{body}`: expected `{needle}` in {diags:?}");
    }

    #[test]
    fn valid_forms_verify() {
        let mut syms = StrInterner::new();
        let m = crate::ir::tests::atomics_module(&mut syms);
        assert!(verify_module(&m).is_ok());
        let diags = check(
            "  %a = load volatile %p align 1 : i32\n  store volatile %v, %p align 2 : i32\n  %b = atomic_load acquire %p align 4 : i32\n  %c = atomic_rmw umax relaxed %p, %v align 8 : i32\n  %d = cmpxchg release seq_cst %p, %w, %w align 8 : i64\n  fence release",
        );
        assert!(diags.is_empty(), "{diags:?}");
    }

    #[test]
    fn load_cannot_release_and_store_cannot_acquire() {
        assert_rejects("  %a = atomic_load release %p align 4 : i32", "atomic_load ordering `release` is invalid");
        assert_rejects("  %a = atomic_load acq_rel %p align 4 : i32", "atomic_load ordering `acq_rel` is invalid");
        assert_rejects("  atomic_store acquire %v, %p align 4 : i32", "atomic_store ordering `acquire` is invalid");
        assert_rejects("  atomic_store acq_rel %v, %p align 4 : i32", "atomic_store ordering `acq_rel` is invalid");
    }

    #[test]
    fn cmpxchg_failure_must_be_a_load_ordering_and_fence_not_relaxed() {
        assert_rejects(
            "  %a = cmpxchg seq_cst release %p, %v, %v align 4 : i32",
            "cmpxchg failure ordering `release` is invalid",
        );
        assert_rejects(
            "  %a = cmpxchg seq_cst acq_rel %p, %v, %v align 4 : i32",
            "cmpxchg failure ordering `acq_rel` is invalid",
        );
        assert_rejects("  fence relaxed", "fence ordering `relaxed` is invalid");
    }

    #[test]
    fn under_aligned_atomics_are_rejected() {
        assert_rejects(
            "  %a = atomic_load seq_cst %p align 2 : i32",
            "atomic_load alignment 2 is below the natural alignment 4 of i32",
        );
        assert_rejects(
            "  atomic_store relaxed %w, %p align 4 : i64",
            "atomic_store alignment 4 is below the natural alignment 8 of i64",
        );
        assert_rejects("  %a = atomic_rmw add relaxed %p, %v align 3 : i32", "must be a nonzero power of two");
        assert_rejects(
            "  %a = cmpxchg seq_cst seq_cst %p, %w, %w align 1 : i64",
            "cmpxchg alignment 1 is below the natural alignment 8 of i64",
        );
    }

    #[test]
    fn unsupported_atomic_types_are_rejected() {
        assert_rejects(
            "  %a = atomic_load relaxed %p align 16 : i128",
            "atomic_load accesses i128, but atomics support only i8, i16, i32, i64 or ptr",
        );
        assert_rejects(
            "  %a = atomic_load relaxed %p align 1 : i1",
            "atomics support only i8, i16, i32, i64 or ptr",
        );
        assert_rejects(
            "  %a = atomic_load relaxed %p align 8 : f64",
            "atomic_load accesses f64",
        );
        // Arithmetic rmw on a pointer: only xchg moves pointers.
        assert_rejects(
            "  %a = atomic_rmw add relaxed %p, %p align 8 : ptr",
            "atomic_rmw accesses ptr, but atomics support only i8, i16, i32 or i64",
        );
        assert!(check("  %a = atomic_rmw xchg relaxed %p, %p align 8 : ptr").is_empty());
    }

    #[test]
    fn operand_types_must_match_the_access() {
        assert_rejects("  atomic_store relaxed %w, %p align 8 : i32", "atomic_store value vs. accessed type");
        assert_rejects("  %a = atomic_rmw add relaxed %p, %w align 8 : i32", "atomic_rmw operand vs. accessed type");
        assert_rejects(
            "  %a = cmpxchg seq_cst relaxed %p, %v, %w align 8 : i64",
            "cmpxchg expected vs. accessed type",
        );
        assert_rejects(
            "  %a = cmpxchg seq_cst relaxed %p, %w, %v align 8 : i64",
            "cmpxchg new vs. accessed type",
        );
        // The address must be a pointer (not an aggregate, unlike plain load).
        assert_rejects("  %a = atomic_load relaxed %w align 8 : i64", "atomic_load address operand must be a pointer");
    }
}

#[cfg(test)]
mod addrspace_tests {
    use crate::support::StrInterner;
    use crate::support::diagnostics::FileId;
    use crate::verify::verify_module;

    /// A 16-bit layout with a program address space 1 and a data space 2.
    const LAYOUT: &str = "e-p:16:8-p1:16:8-p2:32:8-n8:16-P1";

    /// Parse a module with [`LAYOUT`], `globals`, and a one-block function
    /// `f(ptr %p, ptr addrspace(1) %q, ptr addrspace(2) %r, i16 %i)`, whose
    /// body is `body`; return the verifier's messages (empty when valid).
    fn check(globals: &str, body: &str) -> Vec<String> {
        let src = format!(
            "module \"t\"\ndatalayout \"{LAYOUT}\"\n{globals}\nfunc @f(ptr, ptr addrspace(1), ptr addrspace(2), i16) -> void {{\nentry ^0(%p: ptr, %q: ptr addrspace(1), %r: ptr addrspace(2), %i: i16):\n{body}\n  ret\n}}\n"
        );
        let mut syms = StrInterner::new();
        let m = crate::ir::text::parse_module(&src, FileId::new(0), &mut syms)
            .unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
        match verify_module(&m) {
            Ok(()) => Vec::new(),
            Err(diags) => diags.into_iter().map(|d| d.message).collect(),
        }
    }

    fn assert_rejects(globals: &str, body: &str, needle: &str) {
        let diags = check(globals, body);
        assert!(diags.iter().any(|d| d.contains(needle)), "expected `{needle}` for `{body}`, got {diags:?}");
    }

    #[test]
    fn valid_address_space_code_verifies() {
        let globals = "global constant addrspace(1) @tbl : [4 x i8] = [4 x i8] (i8 1, i8 2, i8 3, i8 4)\nglobal @pt : ptr addrspace(1) = ptr addrspace(1) @tbl + 2\nglobal @pf : ptr addrspace(1) = ptr addrspace(1) @f";
        let body = "  %a = ptr_add @tbl, %i : ptr addrspace(1)\n  %b = load %a align 1 : i8\n  %c = load %r align 1 : i16\n  store %b, %p align 1 : i8\n  %d = ptrtoint %q : i16\n  %e = inttoptr %d : ptr\n  %x = icmp eq %q, %a : i1\n  %y = ptrtoint %r : i32\n  %z = bitcast %y : ptr addrspace(2)";
        assert_eq!(check(globals, body), Vec::<String>::new());
    }

    #[test]
    fn pointer_casts_between_spaces_are_rejected() {
        // There is no addrspacecast: a bitcast between pointer types is invalid.
        assert_rejects("", "  %a = bitcast %p : ptr addrspace(1)", "bitcast from ptr to ptr addrspace(1)");
        // Bit sizes follow the layout: a 16-bit pointer does not bitcast to i64.
        assert_rejects("", "  %a = bitcast %p : i64", "bitcast from ptr to i64");
        assert!(check("", "  %a = bitcast %p : i16").is_empty());
    }

    #[test]
    fn spaces_never_unify() {
        assert_rejects("", "  %x = icmp eq %p, %q : i1", "icmp operands");
        assert_rejects("", "  %s = select i1 1, %p, %q : ptr", "select arms");
    }

    /// Build `f(ptr addrspace(1) %q, i16 %i)` whose body appends the raw
    /// instruction `kind` with `operands(q, i)` and result type `result(types)`
    /// (bypassing the builder's typing helpers), and verify it.
    fn check_raw(
        kind: crate::ir::InstKind,
        operands: impl FnOnce(crate::ir::ValueId, crate::ir::ValueId) -> Vec<crate::ir::ValueId>,
        result: impl FnOnce(&mut crate::ir::TypeContext) -> crate::ir::TypeId,
    ) -> Vec<String> {
        let mut syms = StrInterner::new();
        let mut m = crate::ir::Module::new("t");
        m.set_data_layout(crate::ir::DataLayout::parse(LAYOUT).unwrap());
        let q = m.types_mut().ptr_in(1);
        let i16t = m.types_mut().int(16);
        let void = m.types_mut().void();
        let rt = result(m.types_mut());
        let sig = m.types_mut().func(vec![q, i16t], void, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let (qv, iv) = (b.param(e, 0), b.param(e, 1));
            b.append_inst(kind, operands(qv, iv), crate::ir::Flags::NONE, Some(rt));
            b.ret(None);
        }
        match verify_module(&m) {
            Ok(()) => Vec::new(),
            Err(diags) => diags.into_iter().map(|d| d.message).collect(),
        }
    }

    #[test]
    fn ptr_add_stays_in_its_space() {
        use crate::ir::InstKind;
        let add = InstKind::PtrAdd { inbounds: false };
        assert!(check_raw(add.clone(), |q, i| vec![q, i], |t| t.ptr_in(1)).is_empty());
        let d = check_raw(add, |q, i| vec![q, i], |t| t.ptr());
        assert!(d.iter().any(|d| d.contains("must stay in its base's address space 1")), "{d:?}");
    }

    #[test]
    fn allocas_live_in_space_zero_and_calls_in_the_program_space() {
        use crate::ir::InstKind;
        assert!(check("", "  %a = alloca i8 : ptr").is_empty());
        let d = check_raw(InstKind::DynAlloca { align: 1 }, |_, i| vec![i], |t| t.ptr_in(2));
        assert!(d.iter().any(|d| d.contains("dyn_alloca result must be a pointer (address space 0)")), "{d:?}");
        // An indirect call goes through a program-space pointer.
        assert!(check("", "  call %q() : void").is_empty());
        assert_rejects("", "  call %p() : void", "program address space 1");
    }

    #[test]
    fn undeclared_spaces_and_mismatched_addresses_are_rejected() {
        assert_rejects("", "  %a = inttoptr %i : ptr addrspace(7)", "address space 7, which the data layout does not declare");
        assert_rejects("global addrspace(9) @g : i8 = i8 0", "", "lives in address space 9");
        // The address of a space-1 global is a space-1 pointer.
        assert_rejects(
            "global addrspace(1) @g : i8 = i8 0\nglobal @h : ptr = ptr @g",
            "",
            "its symbol lives in address space 1",
        );
    }
}
