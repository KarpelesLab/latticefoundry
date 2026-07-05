//! Semantic analysis: name resolution, type checking, and the typed tree.
//!
//! Consumes the untyped [`crate::ast`] tree and produces a [`Program`] of typed
//! nodes in which every C conversion is explicit (integer promotions, the usual
//! arithmetic conversions, pointer scaling, and the implicit conversions on
//! assignment/return/argument are all inserted as [`TExprKind::Convert`] or
//! pointer-arithmetic nodes). Type errors are reported as spanned diagnostics.
//! Lowering ([`crate::lower`]) then walks the typed tree mechanically.

use std::collections::HashMap;

use latticefoundry::support::diagnostics::{Diagnostic, Span};

use crate::ast::{
    BinaryOp, CType, Expr, ExprKind, IntTy, Stmt, StmtKind, TopLevel, TranslationUnit, UnaryOp,
    VarDecl,
};

/// A function-local object with storage (a parameter or a local variable),
/// addressed by an [`ObjId`] within its function.
pub type ObjId = usize;

/// A typed translation unit ready for lowering.
#[derive(Clone, Debug, Default)]
pub struct Program {
    /// Defined functions, in source order.
    pub funcs: Vec<TFunc>,
    /// Every function signature (definitions and prototypes), used for calls.
    pub sigs: Vec<FuncSig>,
    /// Global variables.
    pub globals: Vec<TGlobal>,
}

/// A function signature (a definition or a prototype).
#[derive(Clone, Debug)]
pub struct FuncSig {
    /// The function name.
    pub name: String,
    /// The return type.
    pub ret: CType,
    /// The parameter types.
    pub params: Vec<CType>,
    /// Whether the function is variadic.
    pub variadic: bool,
    /// Whether a definition (body) was seen.
    pub defined: bool,
}

/// A global variable.
#[derive(Clone, Debug)]
pub struct TGlobal {
    /// The global's name.
    pub name: String,
    /// The global's type.
    pub ty: CType,
    /// The constant initializer value (zero if none).
    pub init: i128,
}

/// A defined function with a typed body.
#[derive(Clone, Debug)]
pub struct TFunc {
    /// The index of this function's signature in [`Program::sigs`].
    pub sig_index: usize,
    /// The function name.
    pub name: String,
    /// The return type.
    pub ret: CType,
    /// Every object with storage (parameters first, then locals), by [`ObjId`].
    pub locals: Vec<LocalInfo>,
    /// The [`ObjId`]s of the parameters, in order.
    pub params: Vec<ObjId>,
    /// The typed statement body.
    pub body: Vec<TStmt>,
    /// The 1-based declaration line (for debug info).
    pub decl_line: u32,
}

/// Storage-carrying object metadata.
#[derive(Clone, Debug)]
pub struct LocalInfo {
    /// The object's source name.
    pub name: String,
    /// The object's type.
    pub ty: CType,
}

/// A typed statement.
#[derive(Clone, Debug)]
pub enum TStmt {
    /// An expression evaluated for effect (or the empty statement).
    Expr(Option<TExpr>),
    /// A nested block.
    Block(Vec<TStmt>),
    /// `if (cond) then [else els]` — `cond` is a scalar tested against zero.
    If(TExpr, Box<TStmt>, Option<Box<TStmt>>),
    /// `while (cond) body`.
    While(TExpr, Box<TStmt>),
    /// `do body while (cond)`.
    DoWhile(Box<TStmt>, TExpr),
    /// `for (init; cond; step) body`.
    For(Option<Box<TStmt>>, Option<TExpr>, Option<TExpr>, Box<TStmt>),
    /// `return e` — `e` already converted to the function return type.
    Return(Option<TExpr>),
    /// `break`.
    Break,
    /// `continue`.
    Continue,
    /// Initialize a local object with a value already converted to its type.
    InitLocal(ObjId, TExpr),
}

/// A typed expression: a [`TExprKind`], its C type, and its source span.
#[derive(Clone, Debug)]
pub struct TExpr {
    /// The expression variant.
    pub kind: TExprKind,
    /// The expression's C type.
    pub ty: CType,
    /// The source span.
    pub span: Span,
}

/// A typed expression node. Operand conversions are already explicit.
#[derive(Clone, Debug)]
pub enum TExprKind {
    /// An integer constant.
    Const(i128),
    /// An lvalue reference to a local/parameter object.
    Obj(ObjId),
    /// An lvalue reference to a global (index into [`Program::globals`]).
    Global(usize),
    /// A reference to a function (index into [`Program::sigs`]); used as a callee.
    FuncRef(usize),
    /// Convert the inner value to this node's type.
    Convert(Box<TExpr>),
    /// Arithmetic/bitwise op on same-typed operands (`+ - * / % & | ^`).
    Arith(BinaryOp, Box<TExpr>, Box<TExpr>),
    /// Shift op (`<< >>`); result type is the left operand's type.
    Shift(BinaryOp, Box<TExpr>, Box<TExpr>),
    /// Comparison (`== != < <= > >=`); result is `int` 0/1.
    Cmp(BinaryOp, Box<TExpr>, Box<TExpr>),
    /// Assignment; rhs already converted to the lvalue type. Result = new value.
    Assign(Box<TExpr>, Box<TExpr>),
    /// Compound assignment `lvalue op= rhs`, computed in `compute_ty`.
    Compound { lvalue: Box<TExpr>, rhs: Box<TExpr>, op: BinaryOp, compute_ty: CType },
    /// A call: callee then already-converted arguments.
    Call(Box<TExpr>, Vec<TExpr>),
    /// Conditional; `then`/`els` already converted to the node type.
    Cond(Box<TExpr>, Box<TExpr>, Box<TExpr>),
    /// Comma; result is the right operand.
    Comma(Box<TExpr>, Box<TExpr>),
    /// Dereference `*p` — an lvalue of the pointee type.
    Deref(Box<TExpr>),
    /// Address-of `&lvalue`.
    AddrOf(Box<TExpr>),
    /// Short-circuiting `&&`; result `int` 0/1.
    LogAnd(Box<TExpr>, Box<TExpr>),
    /// Short-circuiting `||`; result `int` 0/1.
    LogOr(Box<TExpr>, Box<TExpr>),
    /// Logical negation `!e`; result `int` 0/1.
    LogNot(Box<TExpr>),
    /// Arithmetic negation `-e`.
    Neg(Box<TExpr>),
    /// Bitwise complement `~e`.
    BitNot(Box<TExpr>),
    /// Pointer arithmetic `ptr ± index` (index scaled by `elem_size`).
    PtrArith { ptr: Box<TExpr>, index: Box<TExpr>, elem_size: u64, sub: bool },
    /// Pointer difference `a - b`, divided by `elem_size`; result `long`.
    PtrDiff { lhs: Box<TExpr>, rhs: Box<TExpr>, elem_size: u64 },
    /// `++`/`--`; `inc` selects direction, `post` selects old-vs-new result.
    IncDec { target: Box<TExpr>, inc: bool, post: bool, scale: u64 },
}

impl TExpr {
    fn new(kind: TExprKind, ty: CType, span: Span) -> TExpr {
        TExpr { kind, ty, span }
    }

    /// Whether this typed expression designates an lvalue (has storage).
    pub fn is_lvalue(&self) -> bool {
        matches!(self.kind, TExprKind::Obj(_) | TExprKind::Global(_) | TExprKind::Deref(_))
    }
}

/// The size in bytes of a C type under the target data layout.
pub fn size_of(ty: &CType) -> u64 {
    match ty {
        CType::Void => 1,
        CType::Bool => 1,
        CType::Int(i) => u64::from(i.width) / 8,
        CType::Pointer(_) => 8,
    }
}

/// The integer promotion: `_Bool`/`char`/`short` become `int`; other types are
/// unchanged.
fn promote(ty: &CType) -> CType {
    match ty {
        CType::Bool => CType::int(),
        CType::Int(i) if i.width < 32 => CType::int(),
        other => other.clone(),
    }
}

/// The usual arithmetic conversions applied to two (already promoted) integer
/// types, yielding their common type.
fn usual_arith(a: &CType, b: &CType) -> CType {
    let a = promote(a);
    let b = promote(b);
    if a == b {
        return a;
    }
    let (wa, sa) = (a.int_width().unwrap_or(32), a.is_signed());
    let (wb, sb) = (b.int_width().unwrap_or(32), b.is_signed());
    if sa == sb {
        // Same signedness: the wider rank wins.
        return if wa >= wb { a } else { b };
    }
    // Mixed signedness.
    let (unsigned_t, uw, signed_t, sw) =
        if !sa { (a.clone(), wa, b.clone(), wb) } else { (b.clone(), wb, a.clone(), wa) };
    if uw >= sw {
        unsigned_t
    } else {
        // The signed type is strictly wider, so it represents all unsigned values.
        signed_t
    }
}

/// Type-check a translation unit, producing a typed [`Program`] or diagnostics.
pub fn check(unit: &TranslationUnit) -> Result<Program, Vec<Diagnostic>> {
    let mut checker = Checker::default();
    checker.run(unit);
    if checker.diags.is_empty() {
        Ok(Program { funcs: checker.funcs, sigs: checker.sigs, globals: checker.globals })
    } else {
        Err(checker.diags)
    }
}

#[derive(Default)]
struct Checker {
    sigs: Vec<FuncSig>,
    sig_index: HashMap<String, usize>,
    globals: Vec<TGlobal>,
    global_index: HashMap<String, usize>,
    funcs: Vec<TFunc>,
    diags: Vec<Diagnostic>,
}

impl Checker {
    fn error(&mut self, span: Span, msg: impl Into<String>) {
        self.diags.push(Diagnostic::error(msg).with_span(span));
    }

    fn run(&mut self, unit: &TranslationUnit) {
        // Pass 1: register every signature and global so bodies can forward- and
        // mutually-reference them.
        for item in &unit.items {
            match item {
                TopLevel::Proto(p) => {
                    let params = p.params.iter().map(|pp| pp.ty.clone()).collect();
                    self.register_sig(&p.name, p.ret.clone(), params, p.variadic, false, p.span);
                }
                TopLevel::Func(f) => {
                    let params = f.params.iter().map(|pp| pp.ty.clone()).collect();
                    self.register_sig(&f.name, f.ret.clone(), params, false, true, f.span);
                }
                TopLevel::Global(g) => self.register_global(g),
            }
        }
        // Pass 2: check each function body.
        for item in &unit.items {
            if let TopLevel::Func(f) = item {
                self.check_func(f);
            }
        }
    }

    fn register_sig(
        &mut self,
        name: &str,
        ret: CType,
        params: Vec<CType>,
        variadic: bool,
        defined: bool,
        span: Span,
    ) {
        if let Some(&idx) = self.sig_index.get(name) {
            let existing = &mut self.sigs[idx];
            if defined {
                if existing.defined {
                    self.error(span, format!("redefinition of function '{name}'"));
                    return;
                }
                existing.defined = true;
                existing.ret = ret;
                existing.params = params;
                existing.variadic = variadic;
            }
            return;
        }
        let idx = self.sigs.len();
        self.sigs.push(FuncSig { name: name.to_owned(), ret, params, variadic, defined });
        self.sig_index.insert(name.to_owned(), idx);
    }

    fn register_global(&mut self, g: &VarDecl) {
        if self.global_index.contains_key(&g.name) {
            self.error(g.span, format!("redefinition of global '{}'", g.name));
            return;
        }
        if matches!(g.ty, CType::Void) {
            self.error(g.span, "global cannot have type 'void'");
            return;
        }
        let init = match &g.init {
            Some(e) => self.const_eval_init(e, &g.ty),
            None => 0,
        };
        let idx = self.globals.len();
        self.global_index.insert(g.name.clone(), idx);
        self.globals.push(TGlobal { name: g.name.clone(), ty: g.ty.clone(), init });
    }

    fn const_eval_init(&mut self, e: &Expr, _ty: &CType) -> i128 {
        match const_eval(e) {
            Some(v) => v,
            None => {
                self.error(e.span, "global initializer must be a constant expression");
                0
            }
        }
    }

    fn check_func(&mut self, f: &crate::ast::FuncDef) {
        let sig_index = self.sig_index[&f.name];
        let ret = f.ret.clone();
        let mut ctx = FnCtx {
            locals: Vec::new(),
            params: Vec::new(),
            scopes: vec![HashMap::new()],
            ret_ty: ret.clone(),
            loop_depth: 0,
        };
        // Parameters become objects with storage in the outermost scope.
        for p in &f.params {
            if matches!(p.ty, CType::Void) {
                continue;
            }
            let name = p.name.clone().unwrap_or_default();
            let id = ctx.add_object(&name, p.ty.clone());
            ctx.params.push(id);
            if !name.is_empty() {
                ctx.scopes.last_mut().unwrap().insert(name, id);
            }
        }
        let mut body = Vec::new();
        for stmt in &f.body {
            if let Some(s) = self.check_stmt(&mut ctx, stmt) {
                body.push(s);
            }
        }
        let decl_line = f.span.start; // placeholder; refined to a real line by lower via source map
        self.funcs.push(TFunc {
            sig_index,
            name: f.name.clone(),
            ret,
            locals: ctx.locals,
            params: ctx.params,
            body,
            decl_line,
        });
    }

    // --- statements --------------------------------------------------------

    fn check_stmt(&mut self, ctx: &mut FnCtx, stmt: &Stmt) -> Option<TStmt> {
        match &stmt.kind {
            StmtKind::Expr(None) => Some(TStmt::Expr(None)),
            StmtKind::Expr(Some(e)) => {
                let te = self.check_expr(ctx, e)?;
                Some(TStmt::Expr(Some(te)))
            }
            StmtKind::Block(stmts) => {
                ctx.push_scope();
                let mut out = Vec::new();
                for s in stmts {
                    if let Some(ts) = self.check_stmt(ctx, s) {
                        out.push(ts);
                    }
                }
                ctx.pop_scope();
                Some(TStmt::Block(out))
            }
            StmtKind::Decl(decls) => self.check_local_decls(ctx, decls, stmt.span),
            StmtKind::If(cond, then, els) => {
                let c = self.check_cond(ctx, cond)?;
                let t = Box::new(self.check_stmt(ctx, then)?);
                let e = match els {
                    Some(s) => Some(Box::new(self.check_stmt(ctx, s)?)),
                    None => None,
                };
                Some(TStmt::If(c, t, e))
            }
            StmtKind::While(cond, body) => {
                let c = self.check_cond(ctx, cond)?;
                ctx.loop_depth += 1;
                let b = self.check_stmt(ctx, body);
                ctx.loop_depth -= 1;
                Some(TStmt::While(c, Box::new(b?)))
            }
            StmtKind::DoWhile(body, cond) => {
                ctx.loop_depth += 1;
                let b = self.check_stmt(ctx, body);
                ctx.loop_depth -= 1;
                let c = self.check_cond(ctx, cond)?;
                Some(TStmt::DoWhile(Box::new(b?), c))
            }
            StmtKind::For(init, cond, step, body) => {
                ctx.push_scope();
                let init_s = match init {
                    Some(s) => self.check_stmt(ctx, s).map(Box::new),
                    None => None,
                };
                let cond_e = match cond {
                    Some(c) => Some(self.check_cond(ctx, c)?),
                    None => None,
                };
                let step_e = match step {
                    Some(s) => Some(self.check_expr(ctx, s)?),
                    None => None,
                };
                ctx.loop_depth += 1;
                let b = self.check_stmt(ctx, body);
                ctx.loop_depth -= 1;
                ctx.pop_scope();
                Some(TStmt::For(init_s, cond_e, step_e, Box::new(b?)))
            }
            StmtKind::Return(None) => {
                if !matches!(ctx.ret_ty, CType::Void) {
                    // A missing value in a value-returning function: default to 0.
                    let zero = TExpr::new(TExprKind::Const(0), ctx.ret_ty.clone(), stmt.span);
                    return Some(TStmt::Return(Some(zero)));
                }
                Some(TStmt::Return(None))
            }
            StmtKind::Return(Some(e)) => {
                let te = self.check_expr(ctx, e)?;
                if matches!(ctx.ret_ty, CType::Void) {
                    self.error(stmt.span, "return with a value in a function returning void");
                    return Some(TStmt::Return(None));
                }
                let ret_ty = ctx.ret_ty.clone();
                let conv = self.convert(te, &ret_ty);
                Some(TStmt::Return(Some(conv)))
            }
            StmtKind::Break => {
                if ctx.loop_depth == 0 {
                    self.error(stmt.span, "'break' outside of a loop");
                }
                Some(TStmt::Break)
            }
            StmtKind::Continue => {
                if ctx.loop_depth == 0 {
                    self.error(stmt.span, "'continue' outside of a loop");
                }
                Some(TStmt::Continue)
            }
        }
    }

    fn check_local_decls(
        &mut self,
        ctx: &mut FnCtx,
        decls: &[VarDecl],
        _span: Span,
    ) -> Option<TStmt> {
        let mut out = Vec::new();
        for d in decls {
            if matches!(d.ty, CType::Void) {
                self.error(d.span, "variable cannot have type 'void'");
                continue;
            }
            let init_val = match &d.init {
                Some(e) => {
                    let te = self.check_expr(ctx, e)?;
                    Some(self.convert(te, &d.ty))
                }
                None => None,
            };
            // Declaring the object *after* checking the initializer matches C
            // scoping (the initializer cannot see the new name).
            if ctx.scopes.last().unwrap().contains_key(&d.name) {
                self.error(d.span, format!("redeclaration of '{}'", d.name));
            }
            let id = ctx.add_object(&d.name, d.ty.clone());
            ctx.scopes.last_mut().unwrap().insert(d.name.clone(), id);
            if let Some(v) = init_val {
                out.push(TStmt::InitLocal(id, v));
            }
        }
        // Wrap the (possibly several) initializers in a block-free sequence.
        match out.len() {
            0 => Some(TStmt::Expr(None)),
            1 => Some(out.pop().unwrap()),
            _ => Some(TStmt::Block(out)),
        }
    }

    /// Check a condition expression (must be scalar).
    fn check_cond(&mut self, ctx: &mut FnCtx, e: &Expr) -> Option<TExpr> {
        let te = self.check_expr(ctx, e)?;
        if !te.ty.is_scalar() {
            self.error(e.span, format!("condition must be a scalar, found '{}'", te.ty));
        }
        Some(te)
    }

    // --- expressions -------------------------------------------------------

    fn check_expr(&mut self, ctx: &mut FnCtx, e: &Expr) -> Option<TExpr> {
        let span = e.span;
        match &e.kind {
            ExprKind::IntLit(v, ty) => Some(TExpr::new(TExprKind::Const(*v), ty.clone(), span)),
            ExprKind::Ident(name) => self.check_ident(ctx, name, span),
            ExprKind::Unary(op, inner) => self.check_unary(ctx, *op, inner, span),
            ExprKind::Binary(op, l, r) => self.check_binary(ctx, *op, l, r, span),
            ExprKind::Assign(compound, l, r) => self.check_assign(ctx, *compound, l, r, span),
            ExprKind::Call(callee, args) => self.check_call(ctx, callee, args, span),
            ExprKind::Cast(ty, inner) => self.check_cast(ctx, ty, inner, span),
            ExprKind::Cond(c, t, f) => self.check_ternary(ctx, c, t, f, span),
            ExprKind::Comma(a, b) => {
                let ta = self.check_expr(ctx, a)?;
                let tb = self.check_expr(ctx, b)?;
                let ty = tb.ty.clone();
                Some(TExpr::new(TExprKind::Comma(Box::new(ta), Box::new(tb)), ty, span))
            }
            ExprKind::PreInc(inner) => self.check_incdec(ctx, inner, true, false, span),
            ExprKind::PreDec(inner) => self.check_incdec(ctx, inner, false, false, span),
            ExprKind::PostInc(inner) => self.check_incdec(ctx, inner, true, true, span),
            ExprKind::PostDec(inner) => self.check_incdec(ctx, inner, false, true, span),
            ExprKind::SizeofExpr(inner) => {
                let te = self.check_expr(ctx, inner)?;
                let sz = size_of(&te.ty) as i128;
                Some(TExpr::new(TExprKind::Const(sz), size_t(), span))
            }
            ExprKind::SizeofType(ty) => {
                let sz = size_of(ty) as i128;
                Some(TExpr::new(TExprKind::Const(sz), size_t(), span))
            }
        }
    }

    fn check_ident(&mut self, ctx: &mut FnCtx, name: &str, span: Span) -> Option<TExpr> {
        if let Some(id) = ctx.lookup(name) {
            let ty = ctx.locals[id].ty.clone();
            return Some(TExpr::new(TExprKind::Obj(id), ty, span));
        }
        if let Some(&idx) = self.global_index.get(name) {
            let ty = self.globals[idx].ty.clone();
            return Some(TExpr::new(TExprKind::Global(idx), ty, span));
        }
        if let Some(&idx) = self.sig_index.get(name) {
            // A function designator used outside a call: represent as its ref.
            return Some(TExpr::new(TExprKind::FuncRef(idx), CType::Void, span));
        }
        self.error(span, format!("use of undeclared identifier '{name}'"));
        None
    }

    fn check_unary(
        &mut self,
        ctx: &mut FnCtx,
        op: UnaryOp,
        inner: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        match op {
            UnaryOp::Plus => {
                let te = self.check_expr(ctx, inner)?;
                if !te.ty.is_integer() {
                    self.error(span, "unary '+' requires an integer operand");
                    return None;
                }
                let pt = promote(&te.ty);
                Some(self.convert(te, &pt))
            }
            UnaryOp::Neg => {
                let te = self.check_expr(ctx, inner)?;
                if !te.ty.is_integer() {
                    self.error(span, "unary '-' requires an integer operand");
                    return None;
                }
                let pt = promote(&te.ty);
                let c = self.convert(te, &pt);
                Some(TExpr::new(TExprKind::Neg(Box::new(c)), pt, span))
            }
            UnaryOp::BitNot => {
                let te = self.check_expr(ctx, inner)?;
                if !te.ty.is_integer() {
                    self.error(span, "unary '~' requires an integer operand");
                    return None;
                }
                let pt = promote(&te.ty);
                let c = self.convert(te, &pt);
                Some(TExpr::new(TExprKind::BitNot(Box::new(c)), pt, span))
            }
            UnaryOp::LNot => {
                let te = self.check_expr(ctx, inner)?;
                if !te.ty.is_scalar() {
                    self.error(span, "unary '!' requires a scalar operand");
                    return None;
                }
                Some(TExpr::new(TExprKind::LogNot(Box::new(te)), CType::int(), span))
            }
            UnaryOp::Deref => {
                let te = self.check_expr(ctx, inner)?;
                match te.ty.pointee().cloned() {
                    Some(CType::Void) => {
                        self.error(span, "cannot dereference a 'void *'");
                        None
                    }
                    Some(pointee) => {
                        Some(TExpr::new(TExprKind::Deref(Box::new(te)), pointee, span))
                    }
                    None => {
                        self.error(span, format!("cannot dereference non-pointer '{}'", te.ty));
                        None
                    }
                }
            }
            UnaryOp::AddrOf => {
                let te = self.check_expr(ctx, inner)?;
                if !te.is_lvalue() {
                    self.error(span, "cannot take the address of a non-lvalue");
                    return None;
                }
                let ty = CType::ptr_to(te.ty.clone());
                Some(TExpr::new(TExprKind::AddrOf(Box::new(te)), ty, span))
            }
        }
    }

    fn check_binary(
        &mut self,
        ctx: &mut FnCtx,
        op: BinaryOp,
        l: &Expr,
        r: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        // Logical operators short-circuit and produce int 0/1.
        if matches!(op, BinaryOp::LAnd | BinaryOp::LOr) {
            let lt = self.check_expr(ctx, l)?;
            let rt = self.check_expr(ctx, r)?;
            if !lt.ty.is_scalar() || !rt.ty.is_scalar() {
                self.error(span, "logical operator requires scalar operands");
            }
            let kind = if op == BinaryOp::LAnd {
                TExprKind::LogAnd(Box::new(lt), Box::new(rt))
            } else {
                TExprKind::LogOr(Box::new(lt), Box::new(rt))
            };
            return Some(TExpr::new(kind, CType::int(), span));
        }

        let lt = self.check_expr(ctx, l)?;
        let rt = self.check_expr(ctx, r)?;

        // Pointer arithmetic and comparisons.
        if lt.ty.is_pointer() || rt.ty.is_pointer() {
            return self.check_pointer_binary(op, lt, rt, span);
        }

        // Both integer from here.
        if !lt.ty.is_integer() || !rt.ty.is_integer() {
            self.error(span, "invalid operands to binary operator");
            return None;
        }

        match op {
            BinaryOp::Shl | BinaryOp::Shr => {
                let lp = promote(&lt.ty);
                let rp = promote(&rt.ty);
                let lc = self.convert(lt, &lp);
                let rc = self.convert(rt, &rp);
                let ty = lp;
                Some(TExpr::new(TExprKind::Shift(op, Box::new(lc), Box::new(rc)), ty, span))
            }
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt
            | BinaryOp::Ge => {
                let common = usual_arith(&lt.ty, &rt.ty);
                let lc = self.convert(lt, &common);
                let rc = self.convert(rt, &common);
                Some(TExpr::new(TExprKind::Cmp(op, Box::new(lc), Box::new(rc)), CType::int(), span))
            }
            _ => {
                let common = usual_arith(&lt.ty, &rt.ty);
                let lc = self.convert(lt, &common);
                let rc = self.convert(rt, &common);
                Some(TExpr::new(
                    TExprKind::Arith(op, Box::new(lc), Box::new(rc)),
                    common,
                    span,
                ))
            }
        }
    }

    fn check_pointer_binary(
        &mut self,
        op: BinaryOp,
        lt: TExpr,
        rt: TExpr,
        span: Span,
    ) -> Option<TExpr> {
        match op {
            BinaryOp::Add => {
                // ptr + int  or  int + ptr
                let (ptr, idx) = if lt.ty.is_pointer() { (lt, rt) } else { (rt, lt) };
                if !idx.ty.is_integer() {
                    self.error(span, "invalid operands to pointer addition");
                    return None;
                }
                let elem_size = size_of(ptr.ty.pointee().unwrap());
                let ptr_ty = ptr.ty.clone();
                let idx_c = self.convert(idx, &CType::long());
                Some(TExpr::new(
                    TExprKind::PtrArith {
                        ptr: Box::new(ptr),
                        index: Box::new(idx_c),
                        elem_size,
                        sub: false,
                    },
                    ptr_ty,
                    span,
                ))
            }
            BinaryOp::Sub if lt.ty.is_pointer() && rt.ty.is_pointer() => {
                let elem_size = size_of(lt.ty.pointee().unwrap());
                Some(TExpr::new(
                    TExprKind::PtrDiff {
                        lhs: Box::new(lt),
                        rhs: Box::new(rt),
                        elem_size,
                    },
                    CType::long(),
                    span,
                ))
            }
            BinaryOp::Sub => {
                // ptr - int
                if !lt.ty.is_pointer() || !rt.ty.is_integer() {
                    self.error(span, "invalid operands to pointer subtraction");
                    return None;
                }
                let elem_size = size_of(lt.ty.pointee().unwrap());
                let ptr_ty = lt.ty.clone();
                let idx_c = self.convert(rt, &CType::long());
                Some(TExpr::new(
                    TExprKind::PtrArith {
                        ptr: Box::new(lt),
                        index: Box::new(idx_c),
                        elem_size,
                        sub: true,
                    },
                    ptr_ty,
                    span,
                ))
            }
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt
            | BinaryOp::Ge => {
                // Pointer comparison: bring both to a common pointer type.
                let common = if lt.ty.is_pointer() { lt.ty.clone() } else { rt.ty.clone() };
                let lc = self.convert(lt, &common);
                let rc = self.convert(rt, &common);
                Some(TExpr::new(TExprKind::Cmp(op, Box::new(lc), Box::new(rc)), CType::int(), span))
            }
            _ => {
                self.error(span, "invalid operands to binary operator on pointers");
                None
            }
        }
    }

    fn check_assign(
        &mut self,
        ctx: &mut FnCtx,
        compound: Option<BinaryOp>,
        l: &Expr,
        r: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        let lt = self.check_expr(ctx, l)?;
        if !lt.is_lvalue() {
            self.error(l.span, "expression is not assignable (not an lvalue)");
            return None;
        }
        let rt = self.check_expr(ctx, r)?;
        let target_ty = lt.ty.clone();
        match compound {
            None => {
                let rc = self.convert(rt, &target_ty);
                Some(TExpr::new(TExprKind::Assign(Box::new(lt), Box::new(rc)), target_ty, span))
            }
            Some(op) => {
                // Determine the computation type.
                let compute_ty = if target_ty.is_pointer() {
                    // ptr += int / ptr -= int
                    if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
                        self.error(span, "invalid compound assignment on a pointer");
                        return None;
                    }
                    target_ty.clone()
                } else if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
                    // A compound shift computes in the promoted left-operand type.
                    promote(&target_ty)
                } else {
                    usual_arith(&target_ty, &rt.ty)
                };
                Some(TExpr::new(
                    TExprKind::Compound {
                        lvalue: Box::new(lt),
                        rhs: Box::new(rt),
                        op,
                        compute_ty,
                    },
                    target_ty,
                    span,
                ))
            }
        }
    }

    fn check_call(
        &mut self,
        ctx: &mut FnCtx,
        callee: &Expr,
        args: &[Expr],
        span: Span,
    ) -> Option<TExpr> {
        // Only direct calls to named functions are supported.
        let ExprKind::Ident(name) = &callee.kind else {
            self.error(callee.span, "only calls to named functions are supported");
            return None;
        };
        let Some(&idx) = self.sig_index.get(name) else {
            self.error(callee.span, format!("call to undeclared function '{name}'"));
            return None;
        };
        let sig = self.sigs[idx].clone();
        if args.len() < sig.params.len() || (!sig.variadic && args.len() != sig.params.len()) {
            self.error(
                span,
                format!(
                    "function '{name}' expects {} argument(s), found {}",
                    sig.params.len(),
                    args.len()
                ),
            );
        }
        let mut targs = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let ta = self.check_expr(ctx, a)?;
            let conv = if i < sig.params.len() {
                self.convert(ta, &sig.params[i])
            } else {
                // Variadic argument: default argument promotions.
                let pt = promote(&ta.ty);
                self.convert(ta, &pt)
            };
            targs.push(conv);
        }
        let callee_t = TExpr::new(TExprKind::FuncRef(idx), CType::Void, callee.span);
        Some(TExpr::new(TExprKind::Call(Box::new(callee_t), targs), sig.ret.clone(), span))
    }

    fn check_cast(
        &mut self,
        ctx: &mut FnCtx,
        ty: &CType,
        inner: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        let te = self.check_expr(ctx, inner)?;
        if matches!(ty, CType::Void) {
            // Cast to void: evaluate for effect; result is void.
            return Some(TExpr::new(TExprKind::Convert(Box::new(te)), CType::Void, span));
        }
        if !te.ty.is_scalar() {
            self.error(span, "cannot cast a non-scalar value");
            return None;
        }
        Some(self.convert(te, ty))
    }

    fn check_ternary(
        &mut self,
        ctx: &mut FnCtx,
        c: &Expr,
        t: &Expr,
        f: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        let cond = self.check_cond(ctx, c)?;
        let tt = self.check_expr(ctx, t)?;
        let ft = self.check_expr(ctx, f)?;
        let result_ty = if tt.ty.is_integer() && ft.ty.is_integer() {
            usual_arith(&tt.ty, &ft.ty)
        } else if tt.ty.is_pointer() {
            tt.ty.clone()
        } else if ft.ty.is_pointer() {
            ft.ty.clone()
        } else {
            tt.ty.clone()
        };
        let tc = self.convert(tt, &result_ty);
        let fc = self.convert(ft, &result_ty);
        Some(TExpr::new(
            TExprKind::Cond(Box::new(cond), Box::new(tc), Box::new(fc)),
            result_ty,
            span,
        ))
    }

    fn check_incdec(
        &mut self,
        ctx: &mut FnCtx,
        inner: &Expr,
        inc: bool,
        post: bool,
        span: Span,
    ) -> Option<TExpr> {
        let te = self.check_expr(ctx, inner)?;
        if !te.is_lvalue() {
            self.error(span, "operand of increment/decrement is not an lvalue");
            return None;
        }
        if !te.ty.is_scalar() {
            self.error(span, "operand of increment/decrement must be scalar");
            return None;
        }
        let scale = match te.ty.pointee() {
            Some(p) => size_of(p),
            None => 1,
        };
        let ty = te.ty.clone();
        Some(TExpr::new(
            TExprKind::IncDec { target: Box::new(te), inc, post, scale },
            ty,
            span,
        ))
    }

    /// Insert an explicit conversion of `e` to `to`, or return `e` unchanged if
    /// its type already matches.
    fn convert(&mut self, e: TExpr, to: &CType) -> TExpr {
        if &e.ty == to {
            return e;
        }
        let span = e.span;
        TExpr::new(TExprKind::Convert(Box::new(e)), to.clone(), span)
    }
}

/// `size_t` for this target: `unsigned long` (64-bit).
fn size_t() -> CType {
    CType::Int(IntTy { width: 64, signed: false })
}

/// The per-function checking context: scopes and the object table.
struct FnCtx {
    locals: Vec<LocalInfo>,
    params: Vec<ObjId>,
    scopes: Vec<HashMap<String, ObjId>>,
    ret_ty: CType,
    loop_depth: u32,
}

impl FnCtx {
    fn add_object(&mut self, name: &str, ty: CType) -> ObjId {
        let id = self.locals.len();
        self.locals.push(LocalInfo { name: name.to_owned(), ty });
        id
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn lookup(&self, name: &str) -> Option<ObjId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&id) = scope.get(name) {
                return Some(id);
            }
        }
        None
    }
}

/// A small constant evaluator over the untyped AST for global initializers.
fn const_eval(e: &Expr) -> Option<i128> {
    match &e.kind {
        ExprKind::IntLit(v, _) => Some(*v),
        ExprKind::Unary(op, inner) => {
            let v = const_eval(inner)?;
            match op {
                UnaryOp::Neg => Some(-v),
                UnaryOp::Plus => Some(v),
                UnaryOp::BitNot => Some(!v),
                UnaryOp::LNot => Some(i128::from(v == 0)),
                _ => None,
            }
        }
        ExprKind::Binary(op, l, r) => {
            let a = const_eval(l)?;
            let b = const_eval(r)?;
            match op {
                BinaryOp::Add => Some(a + b),
                BinaryOp::Sub => Some(a - b),
                BinaryOp::Mul => Some(a * b),
                BinaryOp::Div if b != 0 => Some(a / b),
                BinaryOp::Rem if b != 0 => Some(a % b),
                BinaryOp::BitAnd => Some(a & b),
                BinaryOp::BitOr => Some(a | b),
                BinaryOp::BitXor => Some(a ^ b),
                BinaryOp::Shl => Some(a << b),
                BinaryOp::Shr => Some(a >> b),
                _ => None,
            }
        }
        ExprKind::Cast(_, inner) => const_eval(inner),
        _ => None,
    }
}
