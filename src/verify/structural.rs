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
//!   is declared, and likewise for `ret` vs. the return type).
//!
//! **Scalars stay strictly typed** — only the pointer ↔ aggregate pairing is
//! newly compatible. The single predicate that encodes this is
//! [`addr_compatible`]; the base / address sites use [`is_aggregate`] alongside
//! [`is_ptr`].

use crate::ir::inst::{BinOp, CastOp, InstData, InstId, InstKind, UnaryOp};
use crate::ir::types::{FloatKind, Type, TypeId};
use crate::ir::value::{Const, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Module};
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
                if float {
                    self.want_float(inst, ty, "binary result");
                } else {
                    self.want_int(inst, ty, "binary result");
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
                self.want_float(inst, ty, "fneg result");
                let o = func.value_type(ops[0]);
                if o != ty {
                    self.type_mismatch(inst, "fneg operand", o, ty);
                }
            }
            InstKind::ICmp(_) => {
                if !self.arity(inst, ops, 2) {
                    return;
                }
                self.want_bool(inst, ty, "icmp result");
                let (l, r) = (func.value_type(ops[0]), func.value_type(ops[1]));
                if l != r {
                    self.type_mismatch(inst, "icmp operands", l, r);
                } else if !is_int(module, l) && !is_ptr(module, l) {
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
                self.want_bool(inst, ty, "fcmp result");
                let (l, r) = (func.value_type(ops[0]), func.value_type(ops[1]));
                if l != r {
                    self.type_mismatch(inst, "fcmp operands", l, r);
                } else if !is_float(module, l) {
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
                if !is_ptr(module, ty) {
                    self.err(format!(
                        "instruction #{}: alloca result must be a pointer, found {}",
                        inst.index(),
                        render_type(module, ty),
                    ));
                }
            }
            InstKind::Load { ty: acc, align } => {
                self.check_align(inst, "load", *align);
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
            InstKind::Store { ty: acc, align } => {
                self.check_align(inst, "store", *align);
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
                }
            }
            InstKind::Select => {
                if !self.arity(inst, ops, 3) {
                    return;
                }
                let cond = func.value_type(ops[0]);
                if !is_bool(module, cond) {
                    self.err(format!(
                        "instruction #{}: select condition must be i1, found {}",
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
            InstKind::Call => self.check_call(inst, data),
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
        let ok = match op {
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
                // Same-size reinterpretation; pointers count as machine-word
                // sized via the layout, so ptr<->ptr and ptr<->iN(word) agree.
                from != to && bit_size(m, from) == bit_size(m, to) && bit_size(m, from).is_some()
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
                // pointer value.
                let ct = func.value_type(callee);
                if !is_ptr(module, ct) {
                    self.err(format!(
                        "instruction #{}: indirect call callee must be a pointer, found {}",
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

    fn check_values(&mut self) {
        let func = self.func;
        let module = self.module;
        for vi in 0..func.value_count() {
            let v = ValueId::from_index(vi);
            let val = func.value(v).clone();
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
                    }
                }
                ValueDef::Global(gid) => {
                    if gid.index() >= self.global_count {
                        self.err(format!(
                            "value {}: references nonexistent global #{}",
                            v.index(),
                            gid.index()
                        ));
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
                _ => self.err(format!(
                    "value {}: aggregate constant has non-aggregate type {}",
                    v.index(),
                    render_type(m, *ty)
                )),
            },
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

    fn want_bool(&mut self, inst: InstId, ty: TypeId, what: &str) {
        if !is_bool(self.module, ty) {
            self.err(format!(
                "instruction #{}: {what} must be i1, found {}",
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
        Type::Array(e, n) => format!("[{n} x {}]", render_type(m, *e)),
        Type::Struct(fs) => {
            let inner: Vec<String> = fs.iter().map(|&f| render_type(m, f)).collect();
            format!("{{{}}}", inner.join(", "))
        }
        Type::Func(_) => "func".to_string(),
    }
}

fn is_int(m: &Module, t: TypeId) -> bool {
    matches!(m.types().get(t), Type::Int(_))
}

fn is_bool(m: &Module, t: TypeId) -> bool {
    matches!(m.types().get(t), Type::Int(1))
}

fn is_float(m: &Module, t: TypeId) -> bool {
    matches!(m.types().get(t), Type::Float(_))
}

fn is_ptr(m: &Module, t: TypeId) -> bool {
    matches!(m.types().get(t), Type::Ptr)
}

/// An aggregate type (`Struct`/`Array`). A value of such a type denotes the
/// address of its storage, so it is usable as a pointer (see the module docs).
fn is_aggregate(m: &Module, t: TypeId) -> bool {
    matches!(m.types().get(t), Type::Struct(_) | Type::Array(..))
}

/// Whether types `a` and `b` are **address-compatible** under the struct-by-value
/// convention: they are equal, both pointers, or one is `ptr` and the other an
/// aggregate (whose value *is* an address). This is the *only* relaxation over
/// exact type equality — scalars (`Int`/`Float`) remain strictly typed. Used at
/// the ABI boundaries (`call` arguments and `ret`).
fn addr_compatible(m: &Module, a: TypeId, b: TypeId) -> bool {
    a == b
        || (is_ptr(m, a) && is_ptr(m, b))
        || (is_ptr(m, a) && is_aggregate(m, b))
        || (is_aggregate(m, a) && is_ptr(m, b))
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

/// The bit size of a bit-reinterpretable type: scalar width, or the machine word
/// (64) for a pointer. Aggregates and functions have no single bit width here.
fn bit_size(m: &Module, t: TypeId) -> Option<u32> {
    match m.types().get(t) {
        Type::Int(w) => Some(*w),
        Type::Float(k) => Some(k.bit_width()),
        Type::Ptr => Some(64),
        _ => None,
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
