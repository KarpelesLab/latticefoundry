//! Lowering the typed C tree to LatticeFoundry IR.
//!
//! Walks [`crate::sema::Program`] and builds a [`latticefoundry::ir::Module`].
//! Locals and parameters are lowered as `alloca` + `load`/`store` so the
//! framework's `mem2reg` promotes them to SSA. Signedness (carried by the C
//! type) selects signed vs. unsigned IR ops and the extend kind; pointer
//! arithmetic scales the index by the pointee size. Control flow, short-circuit
//! `&&`/`||`, the ternary, and lvalue handling are lowered to explicit blocks
//! and block arguments.

use std::collections::HashMap;

use latticefoundry::ir::builder::FunctionBuilder;
use latticefoundry::ir::inst::{BinOp, CastOp, Flags, FloatPred, IntPred};
use latticefoundry::ir::types::TypeId;
use latticefoundry::ir::value::{FloatBits, ValueId};
use latticefoundry::ir::{BlockId, Const, FuncId, GlobalId, Global, Module};
use latticefoundry::support::StrInterner;
use latticefoundry::support::puremp;

use crate::ast::{BinaryOp, CType, FloatTy, Records};
use crate::layout;
use crate::sema::{LocalInfo, Program, TExpr, TExprKind, TFunc, TStmt};

/// A precomputed set of the small, fixed set of IR types the frontend needs.
#[derive(Clone, Copy, Debug)]
struct Tys {
    void: TypeId,
    ptr: TypeId,
    i1: TypeId,
    i8: TypeId,
    i16: TypeId,
    i32: TypeId,
    i64: TypeId,
    f32: TypeId,
    f64: TypeId,
}

impl Tys {
    fn for_int(&self, width: u16) -> TypeId {
        match width {
            8 => self.i8,
            16 => self.i16,
            32 => self.i32,
            64 => self.i64,
            _ => self.i32,
        }
    }

    /// The IR type of a scalar/pointer C type (aggregates are resolved through
    /// the precomputed [`FnLower::agg_types`] map, since they need interning).
    fn of(&self, ty: &CType) -> TypeId {
        match ty {
            CType::Void => self.void,
            CType::Bool => self.i8,
            CType::Int(i) => self.for_int(i.width),
            CType::Float(FloatTy::F32) => self.f32,
            CType::Float(FloatTy::F64) => self.f64,
            CType::Pointer(_) => self.ptr,
            CType::Array(..) | CType::Record(_) => self.ptr,
            CType::Func(_) => self.ptr,
        }
    }
}

/// A byte-offset → 1-based line-number map for debug info.
#[derive(Debug)]
struct LineMap {
    starts: Vec<u32>,
}

impl LineMap {
    fn new(src: &str) -> LineMap {
        let mut starts = vec![0u32];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                starts.push((i + 1) as u32);
            }
        }
        LineMap { starts }
    }

    fn line(&self, offset: u32) -> u32 {
        // Number of line-starts at or before `offset`.
        self.starts.partition_point(|&s| s <= offset) as u32
    }
}

/// Lower a checked [`Program`] to an IR [`Module`], returning the module and the
/// symbol interner whose handles the module's names refer to (the codegen and
/// linker need the same interner).
pub fn lower(program: &Program, source: &str, module_name: &str, debug: bool) -> (Module, StrInterner) {
    let mut module = Module::new(module_name.to_owned());
    let mut syms = StrInterner::new();
    let linemap = LineMap::new(source);

    let tys = {
        let cx = module.types_mut();
        Tys {
            void: cx.void(),
            ptr: cx.ptr(),
            i1: cx.int(1),
            i8: cx.int(8),
            i16: cx.int(16),
            i32: cx.int(32),
            i64: cx.int(64),
            f32: cx.float(latticefoundry::ir::types::FloatKind::F32),
            f64: cx.float(latticefoundry::ir::types::FloatKind::F64),
        }
    };

    // Declare every function signature in order so a sig index equals its FuncId.
    let mut func_ids: Vec<FuncId> = Vec::with_capacity(program.sigs.len());
    for sig in &program.sigs {
        let params: Vec<TypeId> = sig.params.iter().map(|p| tys.of(p)).collect();
        let ret = tys.of(&sig.ret);
        let ft = module.types_mut().func(params, ret, sig.variadic);
        let name = syms.intern(&sig.name);
        func_ids.push(module.declare_function(name, ft));
    }

    // Add globals. Their storage bytes are emitted by the driver's
    // `emit_globals`; the IR init here exists only so each global is well-typed.
    let mut global_ids: Vec<GlobalId> = Vec::with_capacity(program.globals.len());
    for g in &program.globals {
        let ty = layout::ir_type(module.types_mut(), &program.records, &g.ty);
        let init = if g.ty.is_pointer() {
            module.intern_const(Const::Null(ty))
        } else if let Some(fty) = g.ty.float_ty() {
            // A float global's storage bytes (its IEEE image) are emitted by the
            // driver's `emit_globals`; this IR init only has to be well-typed.
            let bits = decode_float_le(fty, &g.bytes);
            module.intern_const(Const::Float { ty, bits })
        } else if g.ty.is_scalar() {
            let value = decode_le(&g.bytes);
            module.intern_const(Const::Int { ty, value: puremp::Int::from_i64(value) })
        } else {
            module.intern_const(Const::Poison(ty))
        };
        let name = syms.intern(&g.name);
        global_ids.push(module.add_global(Global { name, ty, init: Some(init) }));
    }

    // For each function, an externally-defined ("init: None") global aliasing its
    // symbol name. Taking a function's address (`FuncPtr`) materializes through
    // this global's `global_ref` (a RIP-relative `lea` relocated to the function
    // symbol), because the backend materializes a bare `func_ref` *value* as zero
    // — it only honours `func_ref` as a direct call target. The synthetic global
    // is never emitted as data (it is not in `program.globals`), so at link time
    // the reference resolves to the function definition of the same name.
    let ptr_ty = tys.ptr;
    let mut func_addr_globals: Vec<GlobalId> = Vec::with_capacity(program.sigs.len());
    for sig in &program.sigs {
        let name = syms.intern(&sig.name);
        func_addr_globals.push(module.add_global(Global { name, ty: ptr_ty, init: None }));
    }

    // Precompute the interned IR type of every local/parameter aggregate, so the
    // per-function builder (which cannot re-borrow the type context) can size its
    // `alloca`s.
    let mut agg_types: HashMap<CType, TypeId> = HashMap::new();
    // Blob (`[i8 x N]`) types for over-aligned locals, keyed by blob length: an
    // over-aligned object is allocated as a byte blob and its base rounded up.
    let mut blob_types: HashMap<u64, TypeId> = HashMap::new();
    {
        let cx = module.types_mut();
        let i8 = cx.int(8);
        for f in &program.funcs {
            for local in &f.locals {
                if local.ty.is_aggregate() {
                    let id = layout::ir_type(cx, &program.records, &local.ty);
                    agg_types.insert(local.ty.clone(), id);
                }
                if let Some(a) = local.align.filter(|&a| a > 1) {
                    let len = layout::size_of(&program.records, &local.ty) + a;
                    blob_types.entry(len).or_insert_with(|| cx.array(i8, len));
                }
            }
        }
    }

    // Build each defined function's body.
    for f in &program.funcs {
        let fid = func_ids[f.sig_index];
        let decl_line = linemap.line(f.decl_line);
        let builder = module.build(fid);
        let mut fl = FnLower {
            b: builder,
            slots: Vec::new(),
            break_targets: Vec::new(),
            continue_targets: Vec::new(),
            switch_blocks: Vec::new(),
            label_blocks: vec![None; f.n_labels as usize],
            terminated: false,
            func_ids: &func_ids,
            global_ids: &global_ids,
            func_addr_globals: &func_addr_globals,
            locals: &f.locals,
            tys,
            agg_types: &agg_types,
            blob_types: &blob_types,
            records: &program.records,
            debug,
            linemap: &linemap,
        };
        fl.lower_function(f, decl_line);
    }

    (module, syms)
}

/// Decode up to 8 little-endian bytes into an `i64` (for scalar global inits).
fn decode_le(bytes: &[u8]) -> i64 {
    let mut buf = [0u8; 8];
    for (i, b) in bytes.iter().take(8).enumerate() {
        buf[i] = *b;
    }
    i64::from_le_bytes(buf)
}

/// Decode a float global's little-endian IEEE image into [`FloatBits`].
fn decode_float_le(fty: FloatTy, bytes: &[u8]) -> FloatBits {
    match fty {
        FloatTy::F32 => {
            let mut buf = [0u8; 4];
            for (i, b) in bytes.iter().take(4).enumerate() {
                buf[i] = *b;
            }
            FloatBits::F32(u32::from_le_bytes(buf))
        }
        FloatTy::F64 => {
            let mut buf = [0u8; 8];
            for (i, b) in bytes.iter().take(8).enumerate() {
                buf[i] = *b;
            }
            FloatBits::F64(u64::from_le_bytes(buf))
        }
    }
}

/// Build the IEEE [`FloatBits`] of value `v` in float C type `ty`.
fn float_bits(ty: &CType, v: f64) -> FloatBits {
    match ty.float_ty() {
        Some(FloatTy::F32) => FloatBits::F32((v as f32).to_bits()),
        _ => FloatBits::F64(v.to_bits()),
    }
}

/// The IR floating-point binary opcode for a C arithmetic operator.
fn float_binop(op: BinaryOp) -> BinOp {
    match op {
        BinaryOp::Add => BinOp::FAdd,
        BinaryOp::Sub => BinOp::FSub,
        BinaryOp::Mul => BinOp::FMul,
        BinaryOp::Div => BinOp::FDiv,
        _ => unreachable!("not a floating arithmetic operator: {op:?}"),
    }
}

/// The ordered IR floating-point predicate for a C comparison operator. C
/// comparisons are ordered (false if either operand is NaN) except `!=`, which
/// is `une` (true if either operand is NaN, matching C's `!=`).
fn fcmp_pred(op: BinaryOp) -> FloatPred {
    match op {
        BinaryOp::Eq => FloatPred::Oeq,
        BinaryOp::Ne => FloatPred::Une,
        BinaryOp::Lt => FloatPred::Olt,
        BinaryOp::Le => FloatPred::Ole,
        BinaryOp::Gt => FloatPred::Ogt,
        BinaryOp::Ge => FloatPred::Oge,
        _ => unreachable!("not a comparison operator: {op:?}"),
    }
}

/// Per-function lowering state.
struct FnLower<'a> {
    b: FunctionBuilder<'a>,
    /// `slots[obj]` is the `alloca` pointer for object `obj`.
    slots: Vec<ValueId>,
    /// Stack of `break` targets (pushed by loops and switches).
    break_targets: Vec<BlockId>,
    /// Stack of `continue` targets (pushed by loops only).
    continue_targets: Vec<BlockId>,
    /// Stack of per-switch block tables (mark id → block), innermost last.
    switch_blocks: Vec<Vec<BlockId>>,
    /// Blocks for named labels (label id → block), created lazily.
    label_blocks: Vec<Option<BlockId>>,
    terminated: bool,
    func_ids: &'a [FuncId],
    global_ids: &'a [GlobalId],
    /// External aliases (one per function) for materializing function addresses.
    func_addr_globals: &'a [GlobalId],
    locals: &'a [LocalInfo],
    tys: Tys,
    /// Interned IR types of aggregate local/parameter types (for `alloca`).
    agg_types: &'a HashMap<CType, TypeId>,
    /// Interned `[i8 x N]` blob types for over-aligned locals, keyed by length.
    blob_types: &'a HashMap<u64, TypeId>,
    /// The struct/union registry, for layout queries during lowering.
    records: &'a Records,
    debug: bool,
    linemap: &'a LineMap,
}

impl FnLower<'_> {
    fn lower_function(&mut self, f: &TFunc, decl_line: u32) {
        if self.debug {
            self.b.set_decl_line(decl_line);
        }
        let entry = self.b.create_entry_block();
        self.terminated = false;

        // Allocate storage for every object up front (in the entry block).
        self.slots = Vec::with_capacity(self.locals.len());
        for local in self.locals {
            let slot = match local.align.filter(|&a| a > 1) {
                // Over-aligned: allocate a byte blob and round its base up so the
                // object's address meets the requested alignment (the framework's
                // `alloca` only guarantees natural alignment, up to 8 bytes).
                Some(a) => {
                    let len = layout::size_of(self.records, &local.ty) + a;
                    let raw = self.b.alloca(self.blob_types[&len]);
                    self.align_ptr(raw, a)
                }
                None => {
                    let ty = self.ir_of(&local.ty);
                    self.b.alloca(ty)
                }
            };
            self.slots.push(slot);
        }
        // Store the incoming parameter values into their slots.
        let params: Vec<ValueId> = self.b.block_params(entry).to_vec();
        for (i, &obj) in f.params.iter().enumerate() {
            let ty = self.tys.of(&self.locals[obj].ty);
            let align = align_of(&self.locals[obj].ty);
            self.b.store(ty, self.slots[obj], params[i], align);
        }

        self.lower_block(&f.body);

        if !self.terminated {
            // Fall off the end: return 0 (or nothing for void).
            match &f.ret {
                CType::Void => self.b.ret(None),
                other => {
                    let ty = self.tys.of(other);
                    let zero = if other.is_pointer() {
                        self.b.null(ty)
                    } else if other.is_float() {
                        self.b.const_float(ty, float_bits(other, 0.0))
                    } else {
                        self.b.const_i64(ty, 0)
                    };
                    self.b.ret(Some(zero));
                }
            }
            self.terminated = true;
        }
    }

    fn switch(&mut self, bb: BlockId) {
        self.b.switch_to(bb);
        self.terminated = false;
    }

    /// The IR type of a C type (aggregates via the precomputed map).
    fn ir_of(&self, ty: &CType) -> TypeId {
        if ty.is_aggregate() { self.agg_types[ty] } else { self.tys.of(ty) }
    }

    /// The size in bytes of a pointer's pointee (`1` if not a pointer).
    fn pointee_size(&self, ty: &CType) -> u64 {
        match ty.pointee() {
            Some(p) => layout::size_of(self.records, p),
            None => 1,
        }
    }

    fn set_line(&mut self, span: latticefoundry::support::diagnostics::Span) {
        if self.debug {
            self.b.set_line(self.linemap.line(span.start));
        }
    }

    // --- statements --------------------------------------------------------

    fn lower_block(&mut self, stmts: &[TStmt]) {
        // Statements are not skipped when the block is already terminated: a later
        // `case`/`default`/label is a jump target that must still be lowered even
        // if the code textually before it is unreachable. Straight-line dead code
        // is redirected into a fresh unreachable block by `ensure_live`.
        for s in stmts {
            self.lower_stmt(s);
        }
    }

    /// Ensure there is a live (non-terminated) current block to emit into. Called
    /// before lowering any statement that emits directly; if control has already
    /// terminated (dead code), start a fresh unreachable block so later labels
    /// remain reachable and the dead code stays well-formed.
    fn ensure_live(&mut self) {
        if self.terminated {
            let bb = self.b.create_block(&[]);
            self.switch(bb);
        }
    }

    fn lower_stmt(&mut self, s: &TStmt) {
        // Blocks and label markers don't emit at entry (blocks recurse; markers
        // switch to their own block), so they must not force a fresh dead block.
        match s {
            TStmt::Block(_) | TStmt::CaseMark(_) | TStmt::Labeled(..) => {}
            _ => self.ensure_live(),
        }
        match s {
            TStmt::Expr(None) => {}
            TStmt::Expr(Some(e)) => {
                self.lower_effect(e);
            }
            TStmt::Block(stmts) => self.lower_block(stmts),
            TStmt::InitLocal(id, v) => {
                self.set_line(v.span);
                let val = self.lower_rvalue(v);
                let ty = self.tys.of(&self.locals[*id].ty);
                let align = align_of(&self.locals[*id].ty);
                let slot = self.slots[*id];
                self.b.store(ty, slot, val, align);
            }
            TStmt::InitAggregate { obj, size, stores } => {
                let base = self.slots[*obj];
                self.zero_fill(base, *size);
                for (offset, value) in stores {
                    self.set_line(value.span);
                    let val = self.lower_rvalue(value);
                    let addr = self.offset_ptr(base, *offset);
                    let ty = self.tys.of(&value.ty);
                    let align = align_of(&value.ty);
                    self.b.store(ty, addr, val, align);
                }
            }
            TStmt::If(cond, then, els) => self.lower_if(cond, then, els.as_deref()),
            TStmt::While(cond, body) => self.lower_while(cond, body),
            TStmt::DoWhile(body, cond) => self.lower_do_while(body, cond),
            TStmt::For(init, cond, step, body) => {
                self.lower_for(init.as_deref(), cond.as_ref(), step.as_ref(), body)
            }
            TStmt::Return(v) => {
                match v {
                    Some(e) => {
                        self.set_line(e.span);
                        let val = self.lower_rvalue(e);
                        self.b.ret(Some(val));
                    }
                    None => self.b.ret(None),
                }
                self.terminated = true;
            }
            TStmt::Break => {
                if let Some(&brk) = self.break_targets.last() {
                    self.b.br(brk, &[]);
                    self.terminated = true;
                }
            }
            TStmt::Continue => {
                if let Some(&cont) = self.continue_targets.last() {
                    self.b.br(cont, &[]);
                    self.terminated = true;
                }
            }
            TStmt::Switch { value, cases, default, nmarks, body } => {
                self.lower_switch(value, cases, *default, *nmarks, body);
            }
            TStmt::CaseMark(id) => {
                let bb = self.switch_blocks.last().expect("case mark outside a switch")
                    [*id as usize];
                if !self.terminated {
                    self.b.br(bb, &[]);
                }
                self.switch(bb);
            }
            TStmt::Labeled(id, inner) => {
                let bb = self.label_block(*id);
                if !self.terminated {
                    self.b.br(bb, &[]);
                }
                self.switch(bb);
                self.lower_stmt(inner);
            }
            TStmt::Goto(id) => {
                let bb = self.label_block(*id);
                self.b.br(bb, &[]);
                self.terminated = true;
            }
        }
    }

    /// The IR block for a named label (created on first reference so forward
    /// `goto`s and the label itself share one block).
    fn label_block(&mut self, id: u32) -> BlockId {
        match self.label_blocks[id as usize] {
            Some(bb) => bb,
            None => {
                let bb = self.b.create_block(&[]);
                self.label_blocks[id as usize] = Some(bb);
                bb
            }
        }
    }

    /// Lower a `switch`: emit the multi-way branch (jump table / value→block map)
    /// with the default edge, then lower the body. Case/`default` marks in the
    /// body (possibly nested) start each arm's block; falling out of one arm into
    /// the next case's block is C's fall-through. `break` targets the exit block.
    fn lower_switch(
        &mut self,
        value: &TExpr,
        cases: &[(i128, u32)],
        default: Option<u32>,
        nmarks: u32,
        body: &TStmt,
    ) {
        let vv = self.lower_rvalue(value);
        let blocks: Vec<BlockId> = (0..nmarks).map(|_| self.b.create_block(&[])).collect();
        let exit = self.b.create_block(&[]);
        let default_bb = match default {
            Some(id) => blocks[id as usize],
            None => exit,
        };
        let case_list: Vec<(puremp::Int, BlockId, Vec<ValueId>)> = cases
            .iter()
            .map(|(v, id)| (puremp::Int::from_i64(*v as i64), blocks[*id as usize], Vec::new()))
            .collect();
        self.b.switch(vv, default_bb, &[], case_list);
        // The body is entered only through case/default marks (jump targets), not
        // by falling into it, so mark control terminated; any code before the
        // first mark is unreachable and `ensure_live` isolates it.
        self.terminated = true;
        self.switch_blocks.push(blocks);
        self.break_targets.push(exit);
        self.lower_stmt(body);
        if !self.terminated {
            self.b.br(exit, &[]);
        }
        self.switch_blocks.pop();
        self.break_targets.pop();
        self.switch(exit);
    }

    fn lower_if(&mut self, cond: &TExpr, then: &TStmt, els: Option<&TStmt>) {
        let c = self.truth_of(cond);
        let then_bb = self.b.create_block(&[]);
        let join_bb = self.b.create_block(&[]);
        let else_bb = if els.is_some() { self.b.create_block(&[]) } else { join_bb };
        self.b.cond_br(c, then_bb, &[], else_bb, &[]);

        self.switch(then_bb);
        self.lower_stmt(then);
        if !self.terminated {
            self.b.br(join_bb, &[]);
        }

        if let Some(e) = els {
            self.switch(else_bb);
            self.lower_stmt(e);
            if !self.terminated {
                self.b.br(join_bb, &[]);
            }
        }
        self.switch(join_bb);
    }

    fn lower_while(&mut self, cond: &TExpr, body: &TStmt) {
        let head = self.b.create_block(&[]);
        let body_bb = self.b.create_block(&[]);
        let exit = self.b.create_block(&[]);
        self.b.br(head, &[]);
        self.switch(head);
        let c = self.truth_of(cond);
        self.b.cond_br(c, body_bb, &[], exit, &[]);
        self.switch(body_bb);
        self.continue_targets.push(head);
        self.break_targets.push(exit);
        self.lower_stmt(body);
        if !self.terminated {
            self.b.br(head, &[]);
        }
        self.continue_targets.pop();
        self.break_targets.pop();
        self.switch(exit);
    }

    fn lower_do_while(&mut self, body: &TStmt, cond: &TExpr) {
        let body_bb = self.b.create_block(&[]);
        let cond_bb = self.b.create_block(&[]);
        let exit = self.b.create_block(&[]);
        self.b.br(body_bb, &[]);
        self.switch(body_bb);
        self.continue_targets.push(cond_bb);
        self.break_targets.push(exit);
        self.lower_stmt(body);
        if !self.terminated {
            self.b.br(cond_bb, &[]);
        }
        self.continue_targets.pop();
        self.break_targets.pop();
        self.switch(cond_bb);
        let c = self.truth_of(cond);
        self.b.cond_br(c, body_bb, &[], exit, &[]);
        self.switch(exit);
    }

    fn lower_for(
        &mut self,
        init: Option<&TStmt>,
        cond: Option<&TExpr>,
        step: Option<&TExpr>,
        body: &TStmt,
    ) {
        if let Some(i) = init {
            self.lower_stmt(i);
        }
        let head = self.b.create_block(&[]);
        let body_bb = self.b.create_block(&[]);
        let step_bb = self.b.create_block(&[]);
        let exit = self.b.create_block(&[]);
        self.b.br(head, &[]);
        self.switch(head);
        match cond {
            Some(c) => {
                let cv = self.truth_of(c);
                self.b.cond_br(cv, body_bb, &[], exit, &[]);
            }
            None => self.b.br(body_bb, &[]),
        }
        self.switch(body_bb);
        self.continue_targets.push(step_bb);
        self.break_targets.push(exit);
        self.lower_stmt(body);
        if !self.terminated {
            self.b.br(step_bb, &[]);
        }
        self.continue_targets.pop();
        self.break_targets.pop();
        self.switch(step_bb);
        if let Some(s) = step {
            self.lower_effect(s);
        }
        self.b.br(head, &[]);
        self.switch(exit);
    }

    // --- expressions -------------------------------------------------------

    /// Evaluate an expression for its side effects, discarding the value.
    fn lower_effect(&mut self, e: &TExpr) {
        match &e.kind {
            TExprKind::Call(callee, args) if matches!(e.ty, CType::Void) => {
                self.lower_call(callee, args, e);
            }
            TExprKind::Convert(inner) if matches!(e.ty, CType::Void) => {
                self.lower_effect(inner);
            }
            TExprKind::Comma(a, b) => {
                self.lower_effect(a);
                self.lower_effect(b);
            }
            _ => {
                self.lower_rvalue(e);
            }
        }
    }

    /// Lower an expression to a pointer to its storage (it must be an lvalue).
    fn lower_lvalue(&mut self, e: &TExpr) -> ValueId {
        match &e.kind {
            TExprKind::Obj(id) => self.slots[*id],
            TExprKind::Global(idx) => self.b.global_ref(self.global_ids[*idx]),
            TExprKind::Deref(inner) => self.lower_rvalue(inner),
            TExprKind::Field { base, offset } => {
                let a = self.lower_lvalue(base);
                self.offset_ptr(a, *offset)
            }
            TExprKind::CompoundLiteral { obj, zero_size, stores } => {
                self.init_compound(*obj, *zero_size, stores);
                self.slots[*obj]
            }
            _ => unreachable!("lower_lvalue on a non-lvalue expression"),
        }
    }

    /// Initialize a compound literal's object in place: zero-fill an aggregate,
    /// then perform its scalar stores.
    fn init_compound(&mut self, obj: usize, zero_size: u64, stores: &[(u64, TExpr)]) {
        let base = self.slots[obj];
        if zero_size > 0 {
            self.zero_fill(base, zero_size);
        }
        for (offset, value) in stores {
            self.set_line(value.span);
            let val = self.lower_rvalue(value);
            let addr = self.offset_ptr(base, *offset);
            let ty = self.tys.of(&value.ty);
            let align = align_of(&value.ty);
            self.b.store(ty, addr, val, align);
        }
    }

    /// Round a pointer up to an alignment `a` (a power of two): compute
    /// `(p + (a-1)) & ~(a-1)` in the integer domain and cast back to a pointer.
    fn align_ptr(&mut self, raw: ValueId, a: u64) -> ValueId {
        let i64t = self.tys.i64;
        let raw_int = self.b.cast(CastOp::PtrToInt, raw, i64t);
        let bias = self.b.const_i64(i64t, (a - 1) as i64);
        let summed = self.b.bin(BinOp::Add, raw_int, bias, Flags::NONE);
        let mask = self.b.const_i64(i64t, !((a - 1) as i64));
        let masked = self.b.bin(BinOp::And, summed, mask, Flags::NONE);
        self.b.cast(CastOp::IntToPtr, masked, self.tys.ptr)
    }

    /// Displace a pointer by a constant byte offset (in-bounds).
    fn offset_ptr(&mut self, base: ValueId, offset: u64) -> ValueId {
        if offset == 0 {
            return base;
        }
        let off = self.b.const_i64(self.tys.i64, offset as i64);
        self.b.ptr_add(base, off, true)
    }

    /// Zero `size` bytes starting at `base` (8-byte then 1-byte stores).
    fn zero_fill(&mut self, base: ValueId, size: u64) {
        let mut o = 0u64;
        while o + 8 <= size {
            let addr = self.offset_ptr(base, o);
            let z = self.b.const_i64(self.tys.i64, 0);
            self.b.store(self.tys.i64, addr, z, 1);
            o += 8;
        }
        while o < size {
            let addr = self.offset_ptr(base, o);
            let z = self.b.const_i64(self.tys.i8, 0);
            self.b.store(self.tys.i8, addr, z, 1);
            o += 1;
        }
    }

    /// Copy `size` bytes from `src` to `dst` (8-byte then 1-byte load/stores).
    fn copy_bytes(&mut self, dst: ValueId, src: ValueId, size: u64) {
        let mut o = 0u64;
        while o + 8 <= size {
            let s = self.offset_ptr(src, o);
            let d = self.offset_ptr(dst, o);
            let v = self.b.load(self.tys.i64, s, 1);
            self.b.store(self.tys.i64, d, v, 1);
            o += 8;
        }
        while o < size {
            let s = self.offset_ptr(src, o);
            let d = self.offset_ptr(dst, o);
            let v = self.b.load(self.tys.i8, s, 1);
            self.b.store(self.tys.i8, d, v, 1);
            o += 1;
        }
    }

    /// Lower an expression to its (rvalue) IR value.
    fn lower_rvalue(&mut self, e: &TExpr) -> ValueId {
        self.set_line(e.span);
        match &e.kind {
            TExprKind::Const(v) => {
                if e.ty.is_pointer() {
                    let ty = self.tys.ptr;
                    self.b.null(ty)
                } else if e.ty.is_float() {
                    // Only arises from an implicit `return;` zero in a float
                    // function (see sema); materialize a float zero of that type.
                    let ty = self.tys.of(&e.ty);
                    self.b.const_float(ty, float_bits(&e.ty, *v as f64))
                } else {
                    let ty = self.tys.of(&e.ty);
                    self.b.const_i64(ty, *v as i64)
                }
            }
            TExprKind::FConst(v) => {
                let ty = self.tys.of(&e.ty);
                let bits = float_bits(&e.ty, *v);
                self.b.const_float(ty, bits)
            }
            TExprKind::Obj(_)
            | TExprKind::Global(_)
            | TExprKind::Deref(_)
            | TExprKind::Field { .. }
            | TExprKind::CompoundLiteral { .. } => {
                let addr = self.lower_lvalue(e);
                let ty = self.tys.of(&e.ty);
                let align = align_of(&e.ty);
                self.b.load(ty, addr, align)
            }
            TExprKind::Decay(inner) => self.lower_lvalue(inner),
            TExprKind::CopyAssign { dst, src, size } => {
                let d = self.lower_lvalue(dst);
                let s = self.lower_lvalue(src);
                self.copy_bytes(d, s, *size);
                d
            }
            TExprKind::Convert(inner) => {
                let v = self.lower_rvalue(inner);
                self.convert(v, &inner.ty, &e.ty)
            }
            TExprKind::Arith(op, l, r) => {
                let lv = self.lower_rvalue(l);
                let rv = self.lower_rvalue(r);
                let binop = if e.ty.is_float() {
                    float_binop(*op)
                } else {
                    arith_binop(*op, e.ty.is_signed())
                };
                self.b.bin(binop, lv, rv, Flags::NONE)
            }
            TExprKind::Shift(op, l, r) => {
                let lv = self.lower_rvalue(l);
                let rv0 = self.lower_rvalue(r);
                // The IR requires the shift amount to share the value type.
                let rv = self.int_resize(rv0, width_of(&r.ty), r.ty.is_signed(), width_of(&l.ty));
                let binop = match op {
                    BinaryOp::Shl => BinOp::Shl,
                    _ if l.ty.is_signed() => BinOp::AShr,
                    _ => BinOp::LShr,
                };
                self.b.bin(binop, lv, rv, Flags::NONE)
            }
            TExprKind::Cmp(op, l, r) => {
                let lv = self.lower_rvalue(l);
                let rv = self.lower_rvalue(r);
                let bit = if l.ty.is_float() {
                    self.b.fcmp(fcmp_pred(*op), lv, rv, Flags::NONE)
                } else {
                    let pred = cmp_pred(*op, l.ty.is_signed());
                    self.b.icmp(pred, lv, rv)
                };
                self.b.cast(CastOp::ZExt, bit, self.tys.i32)
            }
            TExprKind::Neg(inner) => {
                let v = self.lower_rvalue(inner);
                if e.ty.is_float() {
                    self.b.fneg(v, Flags::NONE)
                } else {
                    let ty = self.tys.of(&e.ty);
                    let zero = self.b.const_i64(ty, 0);
                    self.b.sub(zero, v, Flags::NONE)
                }
            }
            TExprKind::BitNot(inner) => {
                let v = self.lower_rvalue(inner);
                let ty = self.tys.of(&e.ty);
                let ones = self.b.const_i64(ty, -1);
                self.b.bin(BinOp::Xor, v, ones, Flags::NONE)
            }
            TExprKind::LogNot(inner) => {
                let v = self.lower_rvalue(inner);
                let bit = if inner.ty.is_float() {
                    // `!x` is `x == 0.0`; NaN is not equal to 0, so `!NaN == 0`.
                    let zero = self.fzero(&inner.ty);
                    self.b.fcmp(FloatPred::Oeq, v, zero, Flags::NONE)
                } else {
                    let zero = self.zero_of(&inner.ty);
                    self.b.icmp(IntPred::Eq, v, zero)
                };
                self.b.cast(CastOp::ZExt, bit, self.tys.i32)
            }
            TExprKind::LogAnd(l, r) => self.lower_logical(l, r, true),
            TExprKind::LogOr(l, r) => self.lower_logical(l, r, false),
            TExprKind::Assign(lval, rval) => {
                let addr = self.lower_lvalue(lval);
                let v = self.lower_rvalue(rval);
                let ty = self.tys.of(&lval.ty);
                let align = align_of(&lval.ty);
                self.b.store(ty, addr, v, align);
                v
            }
            TExprKind::Compound { lvalue, rhs, op, compute_ty } => {
                self.lower_compound(lvalue, rhs, *op, compute_ty, &e.ty)
            }
            TExprKind::Call(callee, args) => {
                self.lower_call(callee, args, e).expect("non-void call has a result")
            }
            TExprKind::Cond(c, t, f) => self.lower_ternary(c, t, f, &e.ty),
            TExprKind::Comma(a, b) => {
                self.lower_effect(a);
                self.lower_rvalue(b)
            }
            TExprKind::AddrOf(inner) => self.lower_lvalue(inner),
            TExprKind::PtrArith { ptr, index, elem_size, sub } => {
                let p = self.lower_rvalue(ptr);
                let idx = self.lower_rvalue(index);
                let scale = self.b.const_i64(self.tys.i64, *elem_size as i64);
                let mut off = self.b.mul(idx, scale, Flags::NONE);
                if *sub {
                    let zero = self.b.const_i64(self.tys.i64, 0);
                    off = self.b.sub(zero, off, Flags::NONE);
                }
                self.b.ptr_add(p, off, true)
            }
            TExprKind::PtrDiff { lhs, rhs, elem_size } => {
                let a = self.lower_rvalue(lhs);
                let bb = self.lower_rvalue(rhs);
                let ai = self.b.cast(CastOp::PtrToInt, a, self.tys.i64);
                let bi = self.b.cast(CastOp::PtrToInt, bb, self.tys.i64);
                let diff = self.b.sub(ai, bi, Flags::NONE);
                let es = self.b.const_i64(self.tys.i64, *elem_size as i64);
                self.b.bin(BinOp::SDiv, diff, es, Flags::NONE)
            }
            TExprKind::IncDec { target, inc, post, scale } => {
                self.lower_incdec(target, *inc, *post, *scale)
            }
            TExprKind::FuncPtr(idx) => self.b.global_ref(self.func_addr_globals[*idx]),
            TExprKind::FuncRef(_) => {
                unreachable!("function designator not decayed to a function pointer")
            }
        }
    }

    fn lower_logical(&mut self, l: &TExpr, r: &TExpr, is_and: bool) -> ValueId {
        let lb = self.truth_of(l);
        let rhs_bb = self.b.create_block(&[]);
        let join_bb = self.b.create_block(&[self.tys.i1]);
        if is_and {
            // false short-circuits to `false`.
            let f = self.b.const_bool(false);
            self.b.cond_br(lb, rhs_bb, &[], join_bb, &[f]);
        } else {
            // true short-circuits to `true`.
            let t = self.b.const_bool(true);
            self.b.cond_br(lb, join_bb, &[t], rhs_bb, &[]);
        }
        self.switch(rhs_bb);
        let rb = self.truth_of(r);
        self.b.br(join_bb, &[rb]);
        self.switch(join_bb);
        let p = self.b.param(join_bb, 0);
        self.b.cast(CastOp::ZExt, p, self.tys.i32)
    }

    fn lower_ternary(&mut self, c: &TExpr, t: &TExpr, f: &TExpr, ty: &CType) -> ValueId {
        let cond = self.truth_of(c);
        let result_ty = self.tys.of(ty);
        let then_bb = self.b.create_block(&[]);
        let else_bb = self.b.create_block(&[]);
        let join_bb = self.b.create_block(&[result_ty]);
        self.b.cond_br(cond, then_bb, &[], else_bb, &[]);

        self.switch(then_bb);
        let tv = self.lower_rvalue(t);
        self.b.br(join_bb, &[tv]);

        self.switch(else_bb);
        let fv = self.lower_rvalue(f);
        self.b.br(join_bb, &[fv]);

        self.switch(join_bb);
        self.b.param(join_bb, 0)
    }

    fn lower_compound(
        &mut self,
        lvalue: &TExpr,
        rhs: &TExpr,
        op: BinaryOp,
        compute_ty: &CType,
        result_ty: &CType,
    ) -> ValueId {
        let addr = self.lower_lvalue(lvalue);
        let lty = self.tys.of(&lvalue.ty);
        let align = align_of(&lvalue.ty);
        let old = self.b.load(lty, addr, align);

        let new = if lvalue.ty.is_pointer() {
            // Pointer compound: scale the index by the pointee size.
            let idx0 = self.lower_rvalue(rhs);
            let idx = self.convert(idx0, &rhs.ty, &CType::long());
            let elem = self.tys.i64;
            let psize = self.pointee_size(&lvalue.ty);
            let scale = self.b.const_i64(elem, psize as i64);
            let mut off = self.b.mul(idx, scale, Flags::NONE);
            if op == BinaryOp::Sub {
                let zero = self.b.const_i64(elem, 0);
                off = self.b.sub(zero, off, Flags::NONE);
            }
            self.b.ptr_add(old, off, true)
        } else {
            let oldc = self.convert(old, &lvalue.ty, compute_ty);
            let rv0 = self.lower_rvalue(rhs);
            let res = if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
                let amt = self.int_resize(
                    rv0,
                    width_of(&rhs.ty),
                    rhs.ty.is_signed(),
                    width_of(compute_ty),
                );
                let binop = match op {
                    BinaryOp::Shl => BinOp::Shl,
                    _ if compute_ty.is_signed() => BinOp::AShr,
                    _ => BinOp::LShr,
                };
                self.b.bin(binop, oldc, amt, Flags::NONE)
            } else {
                let rc = self.convert(rv0, &rhs.ty, compute_ty);
                let binop = if compute_ty.is_float() {
                    float_binop(op)
                } else {
                    arith_binop(op, compute_ty.is_signed())
                };
                self.b.bin(binop, oldc, rc, Flags::NONE)
            };
            self.convert(res, compute_ty, &lvalue.ty)
        };
        self.b.store(lty, addr, new, align);
        // The value of a compound assignment is the new value in the lvalue type.
        let _ = result_ty;
        new
    }

    fn lower_incdec(&mut self, target: &TExpr, inc: bool, post: bool, scale: u64) -> ValueId {
        let addr = self.lower_lvalue(target);
        let ty = self.tys.of(&target.ty);
        let align = align_of(&target.ty);
        let old = self.b.load(ty, addr, align);
        let new = if target.ty.is_pointer() {
            let delta = if inc { scale as i64 } else { -(scale as i64) };
            let off = self.b.const_i64(self.tys.i64, delta);
            self.b.ptr_add(old, off, true)
        } else if target.ty.is_float() {
            let one = self.b.const_float(ty, float_bits(&target.ty, 1.0));
            let op = if inc { BinOp::FAdd } else { BinOp::FSub };
            self.b.bin(op, old, one, Flags::NONE)
        } else {
            let one = self.b.const_i64(ty, 1);
            if inc {
                self.b.add(old, one, Flags::NONE)
            } else {
                self.b.sub(old, one, Flags::NONE)
            }
        };
        self.b.store(ty, addr, new, align);
        if post { old } else { new }
    }

    fn lower_call(&mut self, callee: &TExpr, args: &[TExpr], call: &TExpr) -> Option<ValueId> {
        // A bare function designator (`FuncPtr`) as the callee is a *direct* call,
        // lowered to `func_ref` (which the backend encodes as a direct call). Any
        // other callee is an indirect call through the loaded pointer value.
        let callee_val = match &callee.kind {
            TExprKind::FuncPtr(idx) => self.b.func_ref(self.func_ids[*idx]),
            _ => self.lower_rvalue(callee),
        };
        let arg_vals: Vec<ValueId> = args.iter().map(|a| self.lower_rvalue(a)).collect();
        let ret_ty = self.tys.of(&call.ty);
        self.b.call(callee_val, &arg_vals, ret_ty)
    }

    /// Evaluate a scalar expression and reduce it to an `i1` truth value. In a
    /// controlling context a float `x` means `x != 0.0`; `une` makes NaN true,
    /// matching C's `!=`.
    fn truth_of(&mut self, e: &TExpr) -> ValueId {
        let v = self.lower_rvalue(e);
        if e.ty.is_float() {
            let zero = self.fzero(&e.ty);
            self.b.fcmp(FloatPred::Une, v, zero, Flags::NONE)
        } else {
            let zero = self.zero_of(&e.ty);
            self.b.icmp(IntPred::Ne, v, zero)
        }
    }

    fn zero_of(&mut self, ty: &CType) -> ValueId {
        if ty.is_pointer() {
            let p = self.tys.ptr;
            self.b.null(p)
        } else {
            let t = self.tys.of(ty);
            self.b.const_i64(t, 0)
        }
    }

    /// A `+0.0` constant of the given float C type.
    fn fzero(&mut self, ty: &CType) -> ValueId {
        let t = self.tys.of(ty);
        let bits = float_bits(ty, 0.0);
        self.b.const_float(t, bits)
    }

    // --- conversions -------------------------------------------------------

    /// Convert `v` (of C type `from`) to C type `to`, emitting the right IR cast.
    fn convert(&mut self, v: ValueId, from: &CType, to: &CType) -> ValueId {
        if from == to {
            return v;
        }
        match (from, to) {
            (_, CType::Void) => v,
            (_, CType::Bool) => {
                // `x != 0` in the source type; floats use an ordered `!=` (`une`).
                let bit = if from.is_float() {
                    let zero = self.fzero(from);
                    self.b.fcmp(FloatPred::Une, v, zero, Flags::NONE)
                } else {
                    let zero = self.zero_of(from);
                    self.b.icmp(IntPred::Ne, v, zero)
                };
                self.b.cast(CastOp::ZExt, bit, self.tys.i8)
            }
            (CType::Pointer(_), CType::Pointer(_)) => v,
            (CType::Pointer(_), CType::Int(t)) => {
                let iv = self.b.cast(CastOp::PtrToInt, v, self.tys.i64);
                self.int_resize(iv, 64, false, t.width)
            }
            // float ↔ float: extend to a wider format, or truncate (rounds).
            (CType::Float(a), CType::Float(b)) => {
                let to_ty = self.tys.of(to);
                if b.bits() > a.bits() {
                    self.b.cast(CastOp::FpExt, v, to_ty)
                } else {
                    self.b.cast(CastOp::FpTrunc, v, to_ty)
                }
            }
            // float → integer: truncate toward zero, by the target's signedness.
            (CType::Float(_), _) if to.is_integer() => {
                let to_ty = self.tys.of(to);
                let op = if to.is_signed() { CastOp::FpToSi } else { CastOp::FpToUi };
                self.b.cast(op, v, to_ty)
            }
            // integer → float: convert by the source's signedness (rounds).
            (_, CType::Float(_)) if from.is_integer() => {
                let to_ty = self.tys.of(to);
                let op = if from.is_signed() { CastOp::SiToFp } else { CastOp::UiToFp };
                self.b.cast(op, v, to_ty)
            }
            (_, CType::Pointer(_)) => {
                // integer (or bool) → pointer
                let iv = self.int_resize(v, width_of(from), from.is_signed(), 64);
                self.b.cast(CastOp::IntToPtr, iv, self.tys.ptr)
            }
            _ => {
                // integer/bool → integer
                self.int_resize(v, width_of(from), from.is_signed(), width_of(to))
            }
        }
    }

    /// Resize an integer value from `from_w` bits (with the given signedness) to
    /// `to_w` bits, choosing trunc / zext / sext as appropriate.
    fn int_resize(&mut self, v: ValueId, from_w: u16, from_signed: bool, to_w: u16) -> ValueId {
        if from_w == to_w {
            return v;
        }
        let to_ty = self.tys.for_int(to_w);
        if to_w > from_w {
            let op = if from_signed { CastOp::SExt } else { CastOp::ZExt };
            self.b.cast(op, v, to_ty)
        } else {
            self.b.cast(CastOp::Trunc, v, to_ty)
        }
    }
}

fn arith_binop(op: BinaryOp, signed: bool) -> BinOp {
    match op {
        BinaryOp::Add => BinOp::Add,
        BinaryOp::Sub => BinOp::Sub,
        BinaryOp::Mul => BinOp::Mul,
        BinaryOp::Div => {
            if signed {
                BinOp::SDiv
            } else {
                BinOp::UDiv
            }
        }
        BinaryOp::Rem => {
            if signed {
                BinOp::SRem
            } else {
                BinOp::URem
            }
        }
        BinaryOp::BitAnd => BinOp::And,
        BinaryOp::BitOr => BinOp::Or,
        BinaryOp::BitXor => BinOp::Xor,
        _ => unreachable!("not an arithmetic/bitwise operator: {op:?}"),
    }
}

fn cmp_pred(op: BinaryOp, signed: bool) -> IntPred {
    match op {
        BinaryOp::Eq => IntPred::Eq,
        BinaryOp::Ne => IntPred::Ne,
        BinaryOp::Lt => {
            if signed {
                IntPred::Slt
            } else {
                IntPred::Ult
            }
        }
        BinaryOp::Le => {
            if signed {
                IntPred::Sle
            } else {
                IntPred::Ule
            }
        }
        BinaryOp::Gt => {
            if signed {
                IntPred::Sgt
            } else {
                IntPred::Ugt
            }
        }
        BinaryOp::Ge => {
            if signed {
                IntPred::Sge
            } else {
                IntPred::Uge
            }
        }
        _ => unreachable!("not a comparison operator: {op:?}"),
    }
}

/// The width in bits of an integer/`_Bool` C type (`_Bool` = 8), else 0.
fn width_of(ty: &CType) -> u16 {
    match ty {
        CType::Bool => 8,
        CType::Int(i) => i.width,
        _ => 0,
    }
}

/// The alignment in bytes to attach to a load/store of this type. Only scalar
/// types reach `load`/`store`; aggregates are copied byte-wise.
fn align_of(ty: &CType) -> u32 {
    match ty {
        CType::Void | CType::Bool => 1,
        CType::Int(i) => (i.width / 8) as u32,
        CType::Float(f) => u32::from(f.bits() / 8),
        CType::Pointer(_) => 8,
        CType::Array(..) | CType::Record(_) => 1,
        CType::Func(_) => 1,
    }
}

