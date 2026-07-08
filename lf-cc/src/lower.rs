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
use crate::layout::BitPlacement;
use crate::sema::{AggStore, LocalInfo, Program, TExpr, TExprKind, TFunc, TStmt};

/// The size in bytes of a System V `__va_list_tag` (`va_copy` copies this many).
const VA_LIST_SIZE: u64 = 24;

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
    // A struct/union return is lowered by hidden-pointer (`sret`) convention: the
    // caller allocates the result and passes its address as a hidden leading
    // pointer parameter, and the function returns `void`. A struct/union *value*
    // parameter or argument is likewise passed by pointer to a caller-made copy
    // (`tys.of` already maps a record to a pointer). This keeps every crossing of
    // the ABI a scalar/pointer operation, which the IR verifier accepts.
    let mut func_ids: Vec<FuncId> = Vec::with_capacity(program.sigs.len());
    for sig in &program.sigs {
        let mut params: Vec<TypeId> = Vec::with_capacity(sig.params.len() + 1);
        if sig.ret.is_record() {
            params.push(tys.ptr); // hidden sret pointer
        }
        params.extend(sig.params.iter().map(|p| tys.of(p)));
        let ret = if sig.ret.is_record() { tys.void } else { tys.of(&sig.ret) };
        let ft = module.types_mut().func(params, ret, sig.variadic);
        let name = syms.intern(&sig.name);
        func_ids.push(module.declare_function(name, ft));
    }

    // The System V variadic frame-address intrinsics, declared once as external
    // `ptr @name()`. `va_start` calls these inside a variadic function; the
    // backend recognizes them by name and replaces each call with the frame
    // address of the register save area / overflow argument area.
    let (va_reg_save, va_overflow) = {
        let hook_sig = module.types_mut().func(Vec::new(), tys.ptr, false);
        let rs = module.declare_function(syms.intern("__lf_va_reg_save_area"), hook_sig);
        let ov = module.declare_function(syms.intern("__lf_va_overflow_area"), hook_sig);
        (rs, ov)
    };

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
            sret: None,
            va_gp: 0,
            va_fp: 0,
            va_reg_save,
            va_overflow,
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
    /// The hidden `sret` pointer of the function being lowered, if it returns a
    /// struct/union by value (the caller-allocated result storage).
    sret: Option<ValueId>,
    /// `va_start` seeds: `gp_offset = 8 * va_gp`, `fp_offset = 48 + 16 * va_fp`,
    /// where the counts are the enclosing function's named GPR/SSE arguments.
    va_gp: u32,
    va_fp: u32,
    /// The variadic frame-address intrinsics (`ptr @__lf_va_*()`).
    va_reg_save: FuncId,
    va_overflow: FuncId,
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
        // A struct/union return uses the hidden `sret` pointer (the first IR
        // parameter); the real parameters follow it.
        let params: Vec<ValueId> = self.b.block_params(entry).to_vec();
        let base = if f.ret.is_record() {
            self.sret = Some(params[0]);
            1
        } else {
            0
        };
        // The `va_start` seed counts: how many named GPR/SSE argument registers
        // the enclosing function consumes (an sret pointer takes one GPR).
        let mut gp = base as u32;
        let mut fp = 0u32;
        // Store the incoming parameter values into their slots. A struct/union
        // value parameter arrives as a pointer to the caller's copy; copy it into
        // the parameter object's own storage so the body owns a private copy.
        for (i, &obj) in f.params.iter().enumerate() {
            let pty = self.locals[obj].ty.clone();
            let incoming = params[base + i];
            if pty.is_record() {
                let size = layout::size_of(self.records, &pty);
                let dst = self.slots[obj];
                self.copy_bytes(dst, incoming, size);
                if gp < 6 {
                    gp += 1;
                }
            } else if pty.is_float() {
                let ty = self.tys.of(&pty);
                let align = align_of(&pty);
                self.b.store(ty, self.slots[obj], incoming, align);
                if fp < 8 {
                    fp += 1;
                }
            } else {
                let ty = self.tys.of(&pty);
                let align = align_of(&pty);
                self.b.store(ty, self.slots[obj], incoming, align);
                if gp < 6 {
                    gp += 1;
                }
            }
        }
        self.va_gp = gp;
        self.va_fp = fp;

        self.lower_block(&f.body);

        if !self.terminated {
            // Fall off the end: return 0 (or nothing for void / a struct return).
            match &f.ret {
                CType::Void => self.b.ret(None),
                r if r.is_record() => self.b.ret(None),
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
            TStmt::CopyInit { obj, src, size } => {
                self.set_line(src.span);
                let dst = self.slots[*obj];
                let s = self.lower_struct_addr(src);
                self.copy_bytes(dst, s, *size);
            }
            TStmt::InitAggregate { obj, size, stores } => {
                let base = self.slots[*obj];
                self.zero_fill(base, *size);
                for st in stores {
                    self.set_line(st.value.span);
                    let val = self.lower_rvalue(&st.value);
                    let addr = self.offset_ptr(base, st.offset);
                    match &st.bits {
                        Some(bp) => {
                            self.bitfield_write_at(addr, bp, val);
                        }
                        None => {
                            let ty = self.tys.of(&st.value.ty);
                            let align = align_of(&st.value.ty);
                            self.b.store(ty, addr, val, align);
                        }
                    }
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
                    // A struct/union return copies the value into the caller's
                    // `sret` storage and returns no register value.
                    Some(e) if e.ty.is_record() => {
                        self.set_line(e.span);
                        let sret = self.sret.expect("struct return has an sret pointer");
                        let src = self.lower_struct_addr(e);
                        let size = layout::size_of(self.records, &e.ty);
                        self.copy_bytes(sret, src, size);
                        self.b.ret(None);
                    }
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
        // Widen the controlling value to 64 bits (sign- or zero-extended per its
        // type) so it matches on the full register the backend's switch lowering
        // compares against — the case immediates are compared at 64-bit width, so
        // e.g. a signed `int` value of `-1` must sit as `0xFFFF_FFFF_FFFF_FFFF`,
        // not `0x0000_0000_FFFF_FFFF`, to hit `case -1:`.
        let raw = self.lower_rvalue(value);
        let vv = self.int_resize(raw, width_of(&value.ty), value.ty.is_signed(), 64);
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
            // A conditional whose result is discarded — commonly a void-typed
            // ternary such as `cond ? (void)0 : abort()` (the `assert` macro).
            // Lower each arm for effect so a void-returning arm is not forced
            // through the value path (which requires a result).
            TExprKind::Cond(c, t, f) => {
                let cond = self.truth_of(c);
                let then_bb = self.b.create_block(&[]);
                let else_bb = self.b.create_block(&[]);
                let join_bb = self.b.create_block(&[]);
                self.b.cond_br(cond, then_bb, &[], else_bb, &[]);
                self.switch(then_bb);
                self.lower_effect(t);
                self.b.br(join_bb, &[]);
                self.switch(else_bb);
                self.lower_effect(f);
                self.b.br(join_bb, &[]);
                self.switch(join_bb);
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
            // A bit-field is not addressable; it is read/written through its
            // storage unit by the dedicated bit-field paths, never as an lvalue
            // address (`&` of a bit-field is rejected in sema).
            TExprKind::BitField { .. } => unreachable!("bit-field has no address"),
            _ => unreachable!("lower_lvalue on a non-lvalue expression"),
        }
    }

    /// Initialize a compound literal's object in place: zero-fill an aggregate,
    /// then perform its scalar stores.
    fn init_compound(&mut self, obj: usize, zero_size: u64, stores: &[AggStore]) {
        let base = self.slots[obj];
        if zero_size > 0 {
            self.zero_fill(base, zero_size);
        }
        for st in stores {
            self.set_line(st.value.span);
            let val = self.lower_rvalue(&st.value);
            let addr = self.offset_ptr(base, st.offset);
            match &st.bits {
                Some(bp) => {
                    self.bitfield_write_at(addr, bp, val);
                }
                None => {
                    let ty = self.tys.of(&st.value.ty);
                    let align = align_of(&st.value.ty);
                    self.b.store(ty, addr, val, align);
                }
            }
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
            TExprKind::BitField { base, offset, bits } => {
                let addr = self.bitfield_unit_addr(base, *offset);
                self.bitfield_read_at(addr, bits)
            }
            TExprKind::Decay(inner) => self.lower_lvalue(inner),
            TExprKind::CopyAssign { dst, src, size } => {
                // Evaluate the source address first (it may be a struct-returning
                // call that must run before the destination is written).
                let s = self.lower_struct_addr(src);
                let d = self.lower_lvalue(dst);
                self.copy_bytes(d, s, *size);
                d
            }
            TExprKind::Convert(inner) => {
                let v = self.lower_rvalue(inner);
                let c = self.convert(v, &inner.ty, &e.ty);
                self.normalize_bitint(c, &e.ty)
            }
            TExprKind::Arith(op, l, r) => {
                let lv = self.lower_rvalue(l);
                let rv = self.lower_rvalue(r);
                let binop = if e.ty.is_float() {
                    float_binop(*op)
                } else {
                    arith_binop(*op, e.ty.is_signed())
                };
                let res = self.b.bin(binop, lv, rv, Flags::NONE);
                self.normalize_bitint(res, &e.ty)
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
                let res = self.b.bin(binop, lv, rv, Flags::NONE);
                self.normalize_bitint(res, &e.ty)
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
                    let res = self.b.sub(zero, v, Flags::NONE);
                    self.normalize_bitint(res, &e.ty)
                }
            }
            TExprKind::BitNot(inner) => {
                let v = self.lower_rvalue(inner);
                let ty = self.tys.of(&e.ty);
                let ones = self.b.const_i64(ty, -1);
                let res = self.b.bin(BinOp::Xor, v, ones, Flags::NONE);
                self.normalize_bitint(res, &e.ty)
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
                if let TExprKind::BitField { base, offset, bits } = &lval.kind {
                    let addr = self.bitfield_unit_addr(base, *offset);
                    let v = self.lower_rvalue(rval);
                    self.bitfield_write_at(addr, bits, v)
                } else {
                    let addr = self.lower_lvalue(lval);
                    let v = self.lower_rvalue(rval);
                    let ty = self.tys.of(&lval.ty);
                    let align = align_of(&lval.ty);
                    self.b.store(ty, addr, v, align);
                    v
                }
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
            TExprKind::VaStart(ap) => self.lower_va_start(ap),
            TExprKind::VaArg(ap) => self.lower_va_arg(ap, &e.ty),
            TExprKind::VaEnd => self.void_value(),
            TExprKind::VaCopy(dst, src) => {
                let d = self.lower_rvalue(dst);
                let s = self.lower_rvalue(src);
                self.copy_bytes(d, s, VA_LIST_SIZE);
                self.void_value()
            }
        }
    }

    /// A throwaway value for a `void`-typed builtin used in an rvalue position
    /// (its result is discarded).
    fn void_value(&mut self) -> ValueId {
        self.b.const_i64(self.tys.i32, 0)
    }

    /// Call one of the variadic frame-address intrinsics, returning its `ptr`.
    fn call_va_hook(&mut self, hook: FuncId) -> ValueId {
        let f = self.b.func_ref(hook);
        self.b.call(f, &[], self.tys.ptr).expect("va hook returns a pointer")
    }

    /// Lower `__builtin_va_start(ap, ...)`: seed the `__va_list_tag` at `ap` per
    /// the System V layout — `gp_offset`/`fp_offset` from the enclosing function's
    /// named argument counts, and the register-save / overflow area addresses from
    /// the backend intrinsics.
    fn lower_va_start(&mut self, ap: &TExpr) -> ValueId {
        let p = self.lower_rvalue(ap);
        let gp_off = 8 * self.va_gp;
        let fp_off = 48 + 16 * self.va_fp;
        let gp = self.b.const_i64(self.tys.i32, i64::from(gp_off));
        self.b.store(self.tys.i32, p, gp, 4);
        let fp_ptr = self.offset_ptr(p, 4);
        let fp = self.b.const_i64(self.tys.i32, i64::from(fp_off));
        self.b.store(self.tys.i32, fp_ptr, fp, 4);
        let ov = self.call_va_hook(self.va_overflow);
        let ov_ptr = self.offset_ptr(p, 8);
        self.b.store(self.tys.ptr, ov_ptr, ov, 8);
        let rsa = self.call_va_hook(self.va_reg_save);
        let rsa_ptr = self.offset_ptr(p, 16);
        self.b.store(self.tys.ptr, rsa_ptr, rsa, 8);
        self.void_value()
    }

    /// Lower `__builtin_va_arg(ap, T)` using the System V walk: an INTEGER-class
    /// `T` (int/pointer) reads from `reg_save_area + gp_offset` while
    /// `gp_offset < 48` (stride 8), else from `overflow_arg_area` (stride 8); an
    /// SSE-class `T` (float/double) reads from `reg_save_area + fp_offset` while
    /// `fp_offset < 176` (stride 16), else the overflow area. The scalar value is
    /// then loaded from the chosen address.
    fn lower_va_arg(&mut self, ap: &TExpr, ty: &CType) -> ValueId {
        let p = self.lower_rvalue(ap);
        let is_sse = ty.is_float();
        let (off_field, max, stride) =
            if is_sse { (4u64, 176i64, 16i64) } else { (0u64, 48i64, 8i64) };
        let off_ptr = self.offset_ptr(p, off_field);
        let cur = self.b.load(self.tys.i32, off_ptr, 4);
        let maxc = self.b.const_i64(self.tys.i32, max);
        let is_reg = self.b.icmp(IntPred::Ult, cur, maxc);
        let reg_bb = self.b.create_block(&[]);
        let ov_bb = self.b.create_block(&[]);
        let join = self.b.create_block(&[self.tys.ptr]);
        self.b.cond_br(is_reg, reg_bb, &[], ov_bb, &[]);

        // Register save area: addr = reg_save_area + cur; cur += stride.
        self.switch(reg_bb);
        let rsa_ptr = self.offset_ptr(p, 16);
        let rsa = self.b.load(self.tys.ptr, rsa_ptr, 8);
        let cur64 = self.b.cast(CastOp::ZExt, cur, self.tys.i64);
        let addr_r = self.b.ptr_add(rsa, cur64, true);
        let stridec = self.b.const_i64(self.tys.i32, stride);
        let new_off = self.b.add(cur, stridec, Flags::NONE);
        self.b.store(self.tys.i32, off_ptr, new_off, 4);
        self.b.br(join, &[addr_r]);

        // Overflow area: addr = overflow_arg_area; overflow_arg_area += 8.
        self.switch(ov_bb);
        let ov_slot = self.offset_ptr(p, 8);
        let ov = self.b.load(self.tys.ptr, ov_slot, 8);
        let eight = self.b.const_i64(self.tys.i64, 8);
        let new_ov = self.b.ptr_add(ov, eight, true);
        self.b.store(self.tys.ptr, ov_slot, new_ov, 8);
        self.b.br(join, &[ov]);

        self.switch(join);
        let addr = self.b.param(join, 0);
        let ity = self.tys.of(ty);
        let align = align_of(ty);
        self.b.load(ity, addr, align)
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
        // A bit-field compound assignment reads through its storage unit, computes
        // in `compute_ty`, and writes back with a masked read-modify-write.
        if let TExprKind::BitField { base, offset, bits } = &lvalue.kind {
            let unit_addr = self.bitfield_unit_addr(base, *offset);
            let old = self.bitfield_read_at(unit_addr, bits);
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
                let binop = arith_binop(op, compute_ty.is_signed());
                self.b.bin(binop, oldc, rc, Flags::NONE)
            };
            let newv = self.convert(res, compute_ty, &lvalue.ty);
            let _ = result_ty;
            return self.bitfield_write_at(unit_addr, bits, newv);
        }

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
        // A `_BitInt` lvalue wraps to its N-bit range before the write-back.
        let new = self.normalize_bitint(new, &lvalue.ty);
        self.b.store(lty, addr, new, align);
        // The value of a compound assignment is the new value in the lvalue type.
        let _ = result_ty;
        new
    }

    fn lower_incdec(&mut self, target: &TExpr, inc: bool, post: bool, scale: u64) -> ValueId {
        // A bit-field `++`/`--` reads its (extended) value, adjusts by one, and
        // writes back through a masked read-modify-write.
        if let TExprKind::BitField { base, offset, bits } = &target.kind {
            let unit_addr = self.bitfield_unit_addr(base, *offset);
            let old = self.bitfield_read_at(unit_addr, bits);
            let ty = self.tys.of(&target.ty);
            let one = self.b.const_i64(ty, 1);
            let new = if inc {
                self.b.add(old, one, Flags::NONE)
            } else {
                self.b.sub(old, one, Flags::NONE)
            };
            let stored = self.bitfield_write_at(unit_addr, bits, new);
            return if post { old } else { stored };
        }
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
            let stepped = if inc {
                self.b.add(old, one, Flags::NONE)
            } else {
                self.b.sub(old, one, Flags::NONE)
            };
            self.normalize_bitint(stepped, &target.ty)
        };
        self.b.store(ty, addr, new, align);
        if post { old } else { new }
    }

    /// The address of a bit-field's storage unit: the base aggregate lvalue's
    /// address displaced by the storage unit's byte offset.
    fn bitfield_unit_addr(&mut self, base: &TExpr, offset: u64) -> ValueId {
        let a = self.lower_lvalue(base);
        self.offset_ptr(a, offset)
    }

    /// Read a bit-field from its storage unit at `addr`: load the native unit,
    /// then isolate and sign/zero-extend the field to the storage-unit
    /// (declared-type) width.
    ///
    /// The mask/shift arithmetic is performed in a work width of at least 32 bits
    /// (the IR backend does not reliably truncate sub-word shift results), then
    /// truncated back to the declared-type width.
    fn bitfield_read_at(&mut self, addr: ValueId, bp: &BitPlacement) -> ValueId {
        let unit_bits = bp.unit_bits;
        let unit_ty = self.tys.for_int(unit_bits);
        let align = u32::from(unit_bits / 8).max(1);
        let raw = self.b.load(unit_ty, addr, align);
        let w = work_bits(unit_bits);
        // Zero-extend the loaded unit into the work width; the bits above the field
        // are shifted out during extraction, so their value is irrelevant.
        let raw_w = self.int_resize(raw, unit_bits, false, w);
        let work_ty = self.tys.for_int(w);
        let left = u32::from(w) - bp.bit_offset - bp.width;
        let right = u32::from(w) - bp.width;
        let lsh = self.b.const_i64(work_ty, i64::from(left));
        let hi = self.b.bin(BinOp::Shl, raw_w, lsh, Flags::NONE);
        let rsh = self.b.const_i64(work_ty, i64::from(right));
        let op = if bp.signed { BinOp::AShr } else { BinOp::LShr };
        let ext = self.b.bin(op, hi, rsh, Flags::NONE);
        // Truncate the extracted (already sign/zero-extended) value back to the
        // declared-type width, which equals the storage unit's width.
        self.int_resize(ext, w, bp.signed, unit_bits)
    }

    /// Write `value` (already in the field's declared type) into a bit-field at
    /// storage-unit address `addr` via a read-modify-write: clear the field's bits
    /// in the loaded unit, insert the masked value, and store the unit back.
    /// Returns the stored field value re-extended to the declared type (the value
    /// of a bit-field assignment expression, which wraps modulo `2^width`).
    ///
    /// As in [`Self::bitfield_read_at`], the bit manipulation is done in a work
    /// width of at least 32 bits and truncated to the unit width for the store.
    fn bitfield_write_at(&mut self, addr: ValueId, bp: &BitPlacement, value: ValueId) -> ValueId {
        let unit_bits = bp.unit_bits;
        let unit_ty = self.tys.for_int(unit_bits);
        let align = u32::from(unit_bits / 8).max(1);
        let w = work_bits(unit_bits);
        let work_ty = self.tys.for_int(w);

        let old = self.b.load(unit_ty, addr, align);
        let old_w = self.int_resize(old, unit_bits, false, w);
        let value_w = self.int_resize(value, unit_bits, false, w);

        let mask = self.b.const_i64(work_ty, bitfield_mask(bp.width, w));
        let off = self.b.const_i64(work_ty, i64::from(bp.bit_offset));
        let mask_sh = self.b.bin(BinOp::Shl, mask, off, Flags::NONE);
        let all_ones = self.b.const_i64(work_ty, -1);
        let inv = self.b.bin(BinOp::Xor, mask_sh, all_ones, Flags::NONE);
        let cleared = self.b.bin(BinOp::And, old_w, inv, Flags::NONE);
        let vmask = self.b.bin(BinOp::And, value_w, mask, Flags::NONE);
        let vsh = self.b.bin(BinOp::Shl, vmask, off, Flags::NONE);
        let newv_w = self.b.bin(BinOp::Or, cleared, vsh, Flags::NONE);
        let newv = self.int_resize(newv_w, w, false, unit_bits);
        self.b.store(unit_ty, addr, newv, align);

        // The value of the assignment is the field read back: sign/zero-extend the
        // masked low bits to the declared-type width.
        let result_w = if bp.signed {
            let sh = self.b.const_i64(work_ty, i64::from(u32::from(w) - bp.width));
            let hi = self.b.bin(BinOp::Shl, vmask, sh, Flags::NONE);
            self.b.bin(BinOp::AShr, hi, sh, Flags::NONE)
        } else {
            vmask
        };
        self.int_resize(result_w, w, bp.signed, unit_bits)
    }

    fn lower_call(&mut self, callee: &TExpr, args: &[TExpr], call: &TExpr) -> Option<ValueId> {
        // A bare function designator (`FuncPtr`) as the callee is a *direct* call,
        // lowered to `func_ref` (which the backend encodes as a direct call). Any
        // other callee is an indirect call through the loaded pointer value.
        let callee_val = match &callee.kind {
            TExprKind::FuncPtr(idx) => self.b.func_ref(self.func_ids[*idx]),
            _ => self.lower_rvalue(callee),
        };
        let ret_is_record = call.ty.is_record();
        let mut arg_vals: Vec<ValueId> = Vec::with_capacity(args.len() + 1);
        // A struct/union return: allocate the caller's result storage and pass its
        // address as the hidden leading argument; the call's value is that address.
        let ret_slot = if ret_is_record {
            let ty = self.ir_of(&call.ty);
            let slot = self.b.alloca(ty);
            arg_vals.push(slot);
            Some(slot)
        } else {
            None
        };
        for a in args {
            if a.ty.is_record() {
                // Pass a struct/union argument by pointer to a fresh copy, so the
                // callee cannot mutate the caller's object (value semantics).
                let ty = self.ir_of(&a.ty);
                let tmp = self.b.alloca(ty);
                let src = self.lower_struct_addr(a);
                let size = layout::size_of(self.records, &a.ty);
                self.copy_bytes(tmp, src, size);
                arg_vals.push(tmp);
            } else {
                arg_vals.push(self.lower_rvalue(a));
            }
        }
        let ret_ty = if ret_is_record { self.tys.void } else { self.tys.of(&call.ty) };
        let res = self.b.call(callee_val, &arg_vals, ret_ty);
        if ret_is_record { ret_slot } else { res }
    }

    /// Lower a `struct`/`union`-typed expression to a pointer to its storage. Used
    /// wherever a whole record value is consumed by a copy (assignment, return,
    /// argument passing): an lvalue yields its address, a struct-returning call
    /// yields its result slot.
    fn lower_struct_addr(&mut self, e: &TExpr) -> ValueId {
        match &e.kind {
            TExprKind::Obj(_)
            | TExprKind::Global(_)
            | TExprKind::Deref(_)
            | TExprKind::Field { .. }
            | TExprKind::CompoundLiteral { .. } => self.lower_lvalue(e),
            TExprKind::Call(callee, args) => {
                self.lower_call(callee, args, e).expect("struct call yields a result pointer")
            }
            // A whole-struct assignment yields the destination address as its value.
            TExprKind::CopyAssign { .. } => self.lower_rvalue(e),
            TExprKind::Comma(a, b) => {
                self.lower_effect(a);
                self.lower_struct_addr(b)
            }
            _ => unreachable!("not a struct-addressable expression: {:?}", e.kind),
        }
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

    /// Reduce a freshly-computed value to a valid `_BitInt(N)` bit pattern within
    /// its standard storage container: for an unsigned `_BitInt(N)` mask to the
    /// low `N` bits (`v & (2^N - 1)`); for a signed one sign-extend from bit `N`
    /// (`(v << (W-N)) >>a (W-N)`). Ordinary integers, and a `_BitInt` whose value
    /// bits fill the whole storage width (N == 8/16/32/64), need no masking (the
    /// container wraps modulo `2^N` on its own). `v` is returned unchanged for a
    /// non-`_BitInt` type.
    fn normalize_bitint(&mut self, v: ValueId, ty: &CType) -> ValueId {
        let CType::Int(i) = ty else { return v };
        let Some(n) = i.bitint else { return v };
        if n >= i.width {
            return v;
        }
        if i.signed {
            // Sign-extend from bit `n` via `shl; ashr` in a work width of at least
            // 32 bits: the backend does not reliably truncate sub-word shifts (as
            // the bit-field lowering also observes), so the shifts run in the work
            // width and the result is resized back to the storage width.
            let w = work_bits(i.width);
            let wt = self.tys.for_int(w);
            let vw = self.int_resize(v, i.width, true, w);
            let sh = self.b.const_i64(wt, i64::from(w - n));
            let hi = self.b.bin(BinOp::Shl, vw, sh, Flags::NONE);
            let ext = self.b.bin(BinOp::AShr, hi, sh, Flags::NONE);
            self.int_resize(ext, w, true, i.width)
        } else {
            // Mask to the low `n` bits (a plain `and`, reliable at any width).
            let work = self.tys.for_int(i.width);
            let mask = ((1u128 << n) - 1) as i64;
            let m = self.b.const_i64(work, mask);
            self.b.bin(BinOp::And, v, m, Flags::NONE)
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

/// The work width (in bits) for a bit-field's mask/shift arithmetic: at least 32
/// bits, since the IR backend does not reliably truncate sub-word shift results.
fn work_bits(unit_bits: u16) -> u16 {
    if unit_bits <= 32 { 32 } else { 64 }
}

/// The low-`width`-bits mask as an `i64` for a `work_bits`-wide compute value. A
/// field as wide as the work width uses all-ones (`-1`), avoiding a
/// `1 << work_bits` overflow.
fn bitfield_mask(width: u32, work_bits: u16) -> i64 {
    if width >= u32::from(work_bits) {
        -1
    } else {
        ((1u64 << width) - 1) as i64
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

