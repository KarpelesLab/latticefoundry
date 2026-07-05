//! Lowering the typed C tree to LatticeFoundry IR.
//!
//! Walks [`crate::sema::Program`] and builds a [`latticefoundry::ir::Module`].
//! Locals and parameters are lowered as `alloca` + `load`/`store` so the
//! framework's `mem2reg` promotes them to SSA. Signedness (carried by the C
//! type) selects signed vs. unsigned IR ops and the extend kind; pointer
//! arithmetic scales the index by the pointee size. Control flow, short-circuit
//! `&&`/`||`, the ternary, and lvalue handling are lowered to explicit blocks
//! and block arguments.

use latticefoundry::ir::builder::FunctionBuilder;
use latticefoundry::ir::inst::{BinOp, CastOp, Flags, IntPred};
use latticefoundry::ir::types::TypeId;
use latticefoundry::ir::value::ValueId;
use latticefoundry::ir::{BlockId, Const, FuncId, GlobalId, Global, Module};
use latticefoundry::support::StrInterner;
use latticefoundry::support::puremp;

use crate::ast::{BinaryOp, CType};
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

    fn of(&self, ty: &CType) -> TypeId {
        match ty {
            CType::Void => self.void,
            CType::Bool => self.i8,
            CType::Int(i) => self.for_int(i.width),
            CType::Pointer(_) => self.ptr,
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

    // Add globals.
    let mut global_ids: Vec<GlobalId> = Vec::with_capacity(program.globals.len());
    for g in &program.globals {
        let ty = tys.of(&g.ty);
        let init = if g.ty.is_pointer() {
            module.intern_const(Const::Null(ty))
        } else {
            module.intern_const(Const::Int { ty, value: puremp::Int::from_i64(g.init as i64) })
        };
        let name = syms.intern(&g.name);
        global_ids.push(module.add_global(Global { name, ty, init: Some(init) }));
    }

    // Build each defined function's body.
    for f in &program.funcs {
        let fid = func_ids[f.sig_index];
        let decl_line = linemap.line(f.decl_line);
        let builder = module.build(fid);
        let mut fl = FnLower {
            b: builder,
            slots: Vec::new(),
            loops: Vec::new(),
            terminated: false,
            func_ids: &func_ids,
            global_ids: &global_ids,
            locals: &f.locals,
            tys,
            debug,
            linemap: &linemap,
        };
        fl.lower_function(f, decl_line);
    }

    (module, syms)
}

/// Per-function lowering state.
struct FnLower<'a> {
    b: FunctionBuilder<'a>,
    /// `slots[obj]` is the `alloca` pointer for object `obj`.
    slots: Vec<ValueId>,
    /// Stack of `(continue target, break target)` for the enclosing loops.
    loops: Vec<(BlockId, BlockId)>,
    terminated: bool,
    func_ids: &'a [FuncId],
    global_ids: &'a [GlobalId],
    locals: &'a [LocalInfo],
    tys: Tys,
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
            let ty = self.tys.of(&local.ty);
            let slot = self.b.alloca(ty);
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

    fn set_line(&mut self, span: latticefoundry::support::diagnostics::Span) {
        if self.debug {
            self.b.set_line(self.linemap.line(span.start));
        }
    }

    // --- statements --------------------------------------------------------

    fn lower_block(&mut self, stmts: &[TStmt]) {
        for s in stmts {
            if self.terminated {
                break;
            }
            self.lower_stmt(s);
        }
    }

    fn lower_stmt(&mut self, s: &TStmt) {
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
                if let Some(&(_, brk)) = self.loops.last() {
                    self.b.br(brk, &[]);
                    self.terminated = true;
                }
            }
            TStmt::Continue => {
                if let Some(&(cont, _)) = self.loops.last() {
                    self.b.br(cont, &[]);
                    self.terminated = true;
                }
            }
        }
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
        self.loops.push((head, exit));
        self.lower_stmt(body);
        if !self.terminated {
            self.b.br(head, &[]);
        }
        self.loops.pop();
        self.switch(exit);
    }

    fn lower_do_while(&mut self, body: &TStmt, cond: &TExpr) {
        let body_bb = self.b.create_block(&[]);
        let cond_bb = self.b.create_block(&[]);
        let exit = self.b.create_block(&[]);
        self.b.br(body_bb, &[]);
        self.switch(body_bb);
        self.loops.push((cond_bb, exit));
        self.lower_stmt(body);
        if !self.terminated {
            self.b.br(cond_bb, &[]);
        }
        self.loops.pop();
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
        self.loops.push((step_bb, exit));
        self.lower_stmt(body);
        if !self.terminated {
            self.b.br(step_bb, &[]);
        }
        self.loops.pop();
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
            _ => unreachable!("lower_lvalue on a non-lvalue expression"),
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
                } else {
                    let ty = self.tys.of(&e.ty);
                    self.b.const_i64(ty, *v as i64)
                }
            }
            TExprKind::Obj(_) | TExprKind::Global(_) | TExprKind::Deref(_) => {
                let addr = self.lower_lvalue(e);
                let ty = self.tys.of(&e.ty);
                let align = align_of(&e.ty);
                self.b.load(ty, addr, align)
            }
            TExprKind::Convert(inner) => {
                let v = self.lower_rvalue(inner);
                self.convert(v, &inner.ty, &e.ty)
            }
            TExprKind::Arith(op, l, r) => {
                let lv = self.lower_rvalue(l);
                let rv = self.lower_rvalue(r);
                let binop = arith_binop(*op, e.ty.is_signed());
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
                let pred = cmp_pred(*op, l.ty.is_signed());
                let bit = self.b.icmp(pred, lv, rv);
                self.b.cast(CastOp::ZExt, bit, self.tys.i32)
            }
            TExprKind::Neg(inner) => {
                let v = self.lower_rvalue(inner);
                let ty = self.tys.of(&e.ty);
                let zero = self.b.const_i64(ty, 0);
                self.b.sub(zero, v, Flags::NONE)
            }
            TExprKind::BitNot(inner) => {
                let v = self.lower_rvalue(inner);
                let ty = self.tys.of(&e.ty);
                let ones = self.b.const_i64(ty, -1);
                self.b.bin(BinOp::Xor, v, ones, Flags::NONE)
            }
            TExprKind::LogNot(inner) => {
                let v = self.lower_rvalue(inner);
                let zero = self.zero_of(&inner.ty);
                let bit = self.b.icmp(IntPred::Eq, v, zero);
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
            TExprKind::FuncRef(_) => unreachable!("function reference used as a value"),
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
            let scale = self.b.const_i64(elem, pointee_size(&lvalue.ty) as i64);
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
                let binop = arith_binop(op, compute_ty.is_signed());
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
        let TExprKind::FuncRef(idx) = &callee.kind else {
            unreachable!("callee is not a function reference");
        };
        let fref = self.b.func_ref(self.func_ids[*idx]);
        let arg_vals: Vec<ValueId> = args.iter().map(|a| self.lower_rvalue(a)).collect();
        let ret_ty = self.tys.of(&call.ty);
        self.b.call(fref, &arg_vals, ret_ty)
    }

    /// Evaluate a scalar expression and reduce it to an `i1` truth value.
    fn truth_of(&mut self, e: &TExpr) -> ValueId {
        let v = self.lower_rvalue(e);
        let zero = self.zero_of(&e.ty);
        self.b.icmp(IntPred::Ne, v, zero)
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

    // --- conversions -------------------------------------------------------

    /// Convert `v` (of C type `from`) to C type `to`, emitting the right IR cast.
    fn convert(&mut self, v: ValueId, from: &CType, to: &CType) -> ValueId {
        if from == to {
            return v;
        }
        match (from, to) {
            (_, CType::Void) => v,
            (_, CType::Bool) => {
                let zero = self.zero_of(from);
                let bit = self.b.icmp(IntPred::Ne, v, zero);
                self.b.cast(CastOp::ZExt, bit, self.tys.i8)
            }
            (CType::Pointer(_), CType::Pointer(_)) => v,
            (CType::Pointer(_), CType::Int(t)) => {
                let iv = self.b.cast(CastOp::PtrToInt, v, self.tys.i64);
                self.int_resize(iv, 64, false, t.width)
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

/// The alignment in bytes to attach to a load/store of this type.
fn align_of(ty: &CType) -> u32 {
    match ty {
        CType::Void => 1,
        CType::Bool => 1,
        CType::Int(i) => (i.width / 8) as u32,
        CType::Pointer(_) => 8,
    }
}

fn pointee_size(ty: &CType) -> u64 {
    match ty.pointee() {
        Some(p) => crate::sema::size_of(p),
        None => 1,
    }
}
