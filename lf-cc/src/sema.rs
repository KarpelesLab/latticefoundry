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
    BinaryOp, CType, Designator, Expr, ExprKind, FuncType, Init, IntTy, RecordId, Records, Stmt,
    StmtKind, TopLevel, TranslationUnit, UnaryOp, VarDecl,
};
use crate::layout;

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
    /// Global variables (including anonymous read-only string literals).
    pub globals: Vec<TGlobal>,
    /// The `struct`/`union` registry, needed by lowering for layout.
    pub records: Records,
}

/// The plain `char` type on this target (signed 8-bit), used for string data.
pub fn char_ty() -> CType {
    CType::Int(IntTy { width: 8, signed: true })
}

/// The result of checking a variable's initializer: a single scalar value, or a
/// list of scalar stores for an aggregate.
enum InitBuilt {
    /// A scalar/pointer initializer already converted to the variable's type.
    Scalar(TExpr),
    /// Aggregate stores (offsets relative to the object's base).
    Aggregate(Vec<AggStore>),
    /// A whole-`struct`/`union` copy from another value of the same record type
    /// (e.g. `struct P r = make_p();`): the source is copied byte-for-byte.
    StructCopy(TExpr),
}

/// One scalar store emitted for an aggregate initializer: a value to write at a
/// byte offset relative to the object's base, optionally targeting a bit-field
/// (which requires a masked read-modify-write rather than a plain store).
#[derive(Clone, Debug)]
pub struct AggStore {
    /// Byte offset within the object (the storage-unit offset for a bit-field).
    pub offset: u64,
    /// The value to store (already converted to the field/element type).
    pub value: TExpr,
    /// The bit placement, `Some` only when the target member is a bit-field.
    pub bits: Option<crate::layout::BitPlacement>,
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

/// A global variable (or an anonymous string-literal object). Its initializer is
/// fully materialized to a little-endian byte image, zero-padded to the type's
/// size; emitting it is a byte copy into a `.data`/`.rodata` section.
#[derive(Clone, Debug)]
pub struct TGlobal {
    /// The global's name.
    pub name: String,
    /// The global's type.
    pub ty: CType,
    /// The initializer image (already the full size of the object).
    pub bytes: Vec<u8>,
    /// Whether the object belongs in read-only data (string literals).
    pub readonly: bool,
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
    /// The number of named labels in the function (one IR block per label id).
    pub n_labels: u32,
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
    /// An explicit `_Alignas`/`alignas` alignment override (over-aligning the
    /// object's stack storage), if any.
    pub align: Option<u64>,
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
    /// `switch (value) body`. `value` is the controlling expression already
    /// converted to its integer-promoted type. `cases` maps each (converted)
    /// case constant to a mark id; `default` is the default mark id if present;
    /// `nmarks` is the number of case/default marks (the block-table size). The
    /// `body` contains [`TStmt::CaseMark`]s (possibly nested) marking each label.
    Switch {
        /// The controlling value (already integer-promoted).
        value: TExpr,
        /// `(case constant in the promoted type, mark id)` pairs, unique by value.
        cases: Vec<(i128, u32)>,
        /// The `default:` mark id, if the switch has one.
        default: Option<u32>,
        /// The number of case/default marks (the per-switch block-table size).
        nmarks: u32,
        /// The switch body.
        body: Box<TStmt>,
    },
    /// A `case`/`default` label marker with its per-switch mark id: lowering
    /// starts (or falls through into) the mark's block here.
    CaseMark(u32),
    /// A named label with its function-wide label id, prefixing `body`.
    Labeled(u32, Box<TStmt>),
    /// `goto` to a named label (its function-wide label id).
    Goto(u32),
    /// Initialize a scalar local object with a value already converted to its type.
    InitLocal(ObjId, TExpr),
    /// Initialize a `struct`/`union` local object by copying `size` bytes from a
    /// value of the same record type (`struct P r = expr;`).
    CopyInit {
        /// The object being initialized.
        obj: ObjId,
        /// The source struct value (an lvalue or a struct-returning call result).
        src: TExpr,
        /// The number of bytes to copy.
        size: u64,
    },
    /// Initialize an aggregate local object: zero its `size` bytes, then perform
    /// each scalar `(byte offset, value)` store.
    InitAggregate {
        /// The object being initialized.
        obj: ObjId,
        /// The object's size in bytes (the region to zero first).
        size: u64,
        /// The scalar stores, each already converted to its field/element type.
        stores: Vec<AggStore>,
    },
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
    /// A floating-point constant (exact value, already rounded to its precision).
    FConst(f64),
    /// An lvalue reference to a local/parameter object.
    Obj(ObjId),
    /// An lvalue reference to a global (index into [`Program::globals`]).
    Global(usize),
    /// A function *designator* (index into [`Program::sigs`]); its C type is
    /// [`CType::Func`]. Used as a value it decays to [`TExprKind::FuncPtr`].
    FuncRef(usize),
    /// A function pointer value: the address of the function at the given
    /// [`Program::sigs`] index (typed `Pointer(Func)`). Lowers to `func_ref`.
    FuncPtr(usize),
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
    /// A member lvalue: `base` (an aggregate lvalue) displaced by `offset` bytes.
    Field { base: Box<TExpr>, offset: u64 },
    /// A bit-field lvalue: the storage unit lives at `base` (an aggregate lvalue)
    /// displaced by `offset` bytes; the field occupies `bits.width` bits at
    /// `bits.bit_offset` within a `bits.unit_bits`-wide unit. Read via a masked,
    /// sign/zero-extended load; assigned via a read-modify-write store. Its C type
    /// (the node's `ty`) is the bit-field's declared type. It is a modifiable
    /// lvalue but is not addressable (`&` is a constraint violation).
    BitField { base: Box<TExpr>, offset: u64, bits: crate::layout::BitPlacement },
    /// Array-to-pointer decay: yield the address of `inner` (an array lvalue) as
    /// a pointer to its first element.
    Decay(Box<TExpr>),
    /// A whole-aggregate copy `dst = src` (both lvalues), copying `size` bytes.
    CopyAssign { dst: Box<TExpr>, src: Box<TExpr>, size: u64 },
    /// A compound literal (C99): an unnamed object `obj` initialized in place on
    /// first evaluation (zero-filling `zero_size` bytes for an aggregate, then
    /// performing the scalar `stores`). The expression designates that object (an
    /// lvalue), so it lowers like [`TExprKind::Obj`] once initialized.
    CompoundLiteral { obj: ObjId, zero_size: u64, stores: Vec<AggStore> },
    /// `__builtin_va_start(ap, ...)`: initialize the `va_list` at `ap` (a pointer
    /// to its `__va_list_tag`) from the enclosing function's argument frame.
    VaStart(Box<TExpr>),
    /// `__builtin_va_arg(ap, T)`: fetch the next variadic argument as type `T`
    /// (this node's `ty`); `ap` is a pointer to the `__va_list_tag`.
    VaArg(Box<TExpr>),
    /// `__builtin_va_end(ap)`: a no-op on this target (`ty` is `void`).
    VaEnd,
    /// `__builtin_va_copy(dst, src)`: copy the 24-byte `__va_list_tag` state from
    /// `src` to `dst` (both pointers to their tags).
    VaCopy(Box<TExpr>, Box<TExpr>),
}

impl TExpr {
    fn new(kind: TExprKind, ty: CType, span: Span) -> TExpr {
        TExpr { kind, ty, span }
    }

    /// Whether this typed expression designates an lvalue (has storage).
    pub fn is_lvalue(&self) -> bool {
        matches!(
            self.kind,
            TExprKind::Obj(_)
                | TExprKind::Global(_)
                | TExprKind::Deref(_)
                | TExprKind::Field { .. }
                | TExprKind::BitField { .. }
                | TExprKind::CompoundLiteral { .. }
        )
    }

    /// Whether this expression designates a bit-field (a modifiable lvalue that
    /// is not addressable).
    fn is_bitfield(&self) -> bool {
        matches!(self.kind, TExprKind::BitField { .. })
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

/// The usual arithmetic conversions applied to two arithmetic types, yielding
/// their common type. Floating types rank above every integer type: if either
/// operand is `double` the result is `double`; else if either is `float` the
/// result is `float`; otherwise the integer promotions and integer UAC apply.
fn usual_arith(a: &CType, b: &CType) -> CType {
    if a.is_float() || b.is_float() {
        let has_double =
            a.float_ty() == Some(crate::ast::FloatTy::F64) || b.float_ty() == Some(crate::ast::FloatTy::F64);
        return if has_double { CType::double() } else { CType::float() };
    }
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
    let mut checker = Checker {
        records: unit.records.clone(),
        enum_consts: unit.enum_consts.iter().cloned().collect(),
        ..Checker::default()
    };
    checker.run(unit);
    if checker.diags.is_empty() {
        Ok(Program {
            funcs: checker.funcs,
            sigs: checker.sigs,
            globals: checker.globals,
            records: checker.records,
        })
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
    /// The `struct`/`union` registry (from the parser).
    records: Records,
    /// Enumerator constants resolvable as integer constant expressions.
    enum_consts: HashMap<String, i128>,
    /// Deduplicated string-literal objects: bytes → global index.
    string_pool: HashMap<Vec<u8>, usize>,
}

impl Checker {
    fn error(&mut self, span: Span, msg: impl Into<String>) {
        self.diags.push(Diagnostic::error(msg).with_span(span));
    }

    /// The size in bytes of a C type under the target layout.
    fn size_of(&self, ty: &CType) -> u64 {
        layout::size_of(&self.records, ty)
    }

    /// Intern a string literal as an anonymous read-only global, returning its
    /// index in [`Program::globals`]. Identical literals are deduplicated.
    fn intern_string(&mut self, mut bytes: Vec<u8>) -> usize {
        bytes.push(0); // NUL terminator
        if let Some(&idx) = self.string_pool.get(&bytes) {
            return idx;
        }
        let idx = self.globals.len();
        let name = format!(".Lstr.{idx}");
        let ty = CType::Array(Box::new(char_ty()), bytes.len() as u64);
        self.string_pool.insert(bytes.clone(), idx);
        self.globals.push(TGlobal { name, ty, bytes, readonly: true });
        idx
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
                    self.register_sig(&f.name, f.ret.clone(), params, f.variadic, true, f.span);
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
        let mut ty = g.ty.clone();
        if let Some(init) = &g.init {
            ty = self.deduce_array_len(&ty, init);
        }
        let size = self.size_of(&ty) as usize;
        let mut bytes = vec![0u8; size];
        if let Some(init) = &g.init {
            self.build_global_bytes(&ty, init, 0, &mut bytes, g.span);
        }
        let idx = self.globals.len();
        self.global_index.insert(g.name.clone(), idx);
        self.globals.push(TGlobal { name: g.name.clone(), ty, bytes, readonly: false });
    }

    /// Evaluate a constant integer expression, resolving enumerators.
    fn const_eval(&self, e: &Expr) -> Option<i128> {
        const_eval_with(e, &self.enum_consts, &self.records)
    }

    /// Materialize an initializer to little-endian bytes at `off` within `bytes`
    /// (globals must have constant initializers).
    fn build_global_bytes(&mut self, ty: &CType, init: &Init, off: u64, bytes: &mut [u8], span: Span) {
        match ty {
            CType::Array(elem, n) => {
                // `char[] = "..."` writes the literal bytes directly.
                if let Init::Expr(e) = init
                    && let ExprKind::StrLit(s) = &e.kind
                    && matches!(**elem, CType::Int(IntTy { width: 8, .. }))
                {
                    write_string_bytes(bytes, off, s, *n);
                    return;
                }
                let stride = layout::stride_of(&self.records, elem);
                let items = match init {
                    Init::List(items) => items,
                    Init::Expr(_) => {
                        self.error(span, "array initializer must be a brace-enclosed list");
                        return;
                    }
                };
                let mut idx = 0u64;
                for item in items {
                    idx = apply_index_designators(&item.designators, idx);
                    if idx < *n {
                        self.build_global_bytes(elem, &item.init, off + idx * stride, bytes, span);
                    }
                    idx += 1;
                }
            }
            CType::Record(id) => {
                let id = *id;
                let items = match init {
                    Init::List(items) => items,
                    Init::Expr(_) => {
                        self.error(span, "struct/union initializer must be a brace-enclosed list");
                        return;
                    }
                };
                let mut field_idx = 0usize;
                for item in items {
                    field_idx = self.apply_field_designators(id, &item.designators, field_idx);
                    let nfields = self.records.get(id).fields.len();
                    // Unnamed bit-fields (padding, and `:0`) take no initializer.
                    while field_idx < nfields && is_unnamed_bitfield(&self.records, id, field_idx) {
                        field_idx += 1;
                    }
                    if field_idx < nfields {
                        let fty = self.records.get(id).fields[field_idx].ty.clone();
                        let (foff, bits) = layout::field_placement(&self.records, id, field_idx);
                        match bits {
                            // A bit-field initializer OR's its masked, shifted value
                            // into the storage unit's bytes (the image is zeroed).
                            Some(bp) => {
                                match init_scalar_expr(&item.init).and_then(|e| self.const_eval(e)) {
                                    Some(v) => write_bitfield_bytes(bytes, off + foff, v, bp),
                                    None => self.error(
                                        span,
                                        "bit-field initializer must be a constant expression",
                                    ),
                                }
                            }
                            None => {
                                self.build_global_bytes(&fty, &item.init, off + foff, bytes, span)
                            }
                        }
                    }
                    field_idx += 1;
                }
            }
            _ => {
                // Scalar: a bare expression, or a single-element brace list.
                let e = match init {
                    Init::Expr(e) => e,
                    Init::List(items) if items.len() == 1 => match &items[0].init {
                        Init::Expr(e) => e,
                        Init::List(_) => {
                            self.error(span, "invalid scalar initializer");
                            return;
                        }
                    },
                    Init::List(_) => {
                        self.error(span, "invalid scalar initializer");
                        return;
                    }
                };
                if let Some(fty) = ty.float_ty() {
                    match const_eval_float(e, &self.enum_consts) {
                        Some(v) => write_float_bytes(bytes, off, v, fty),
                        None => {
                            self.error(e.span, "global initializer must be a constant expression");
                        }
                    }
                } else {
                    match self.const_eval(e) {
                        Some(v) => write_int_bytes(bytes, off, v, self.size_of(ty)),
                        None => {
                            self.error(e.span, "global initializer must be a constant expression");
                        }
                    }
                }
            }
        }
    }

    /// Deduce the length of an incomplete array type (`T a[]`) from its
    /// initializer, or return `ty` unchanged.
    fn deduce_array_len(&self, ty: &CType, init: &Init) -> CType {
        if let CType::Array(elem, 0) = ty {
            let n = match init {
                Init::Expr(e) => match &e.kind {
                    ExprKind::StrLit(s) => s.len() as u64 + 1,
                    _ => 1,
                },
                Init::List(items) => {
                    let mut idx = 0u64;
                    let mut max = 0u64;
                    for item in items {
                        idx = apply_index_designators(&item.designators, idx);
                        max = max.max(idx + 1);
                        idx += 1;
                    }
                    max
                }
            };
            return CType::Array(elem.clone(), n.max(1));
        }
        ty.clone()
    }

    fn apply_field_designators(&self, id: RecordId, desigs: &[Designator], cur: usize) -> usize {
        match desigs.first() {
            Some(Designator::Field(name)) => {
                self.records.field(id, name).map(|(i, _)| i).unwrap_or(cur)
            }
            _ => cur,
        }
    }

    fn check_func(&mut self, f: &crate::ast::FuncDef) {
        let sig_index = self.sig_index[&f.name];
        let ret = f.ret.clone();
        // Collect every label in the function up front so `goto` may reference a
        // label that appears later (forward references); labels have function
        // scope, not block scope. Duplicate labels are diagnosed here.
        let mut labels = HashMap::new();
        for stmt in &f.body {
            self.collect_labels(stmt, &mut labels);
        }
        let n_labels = labels.len() as u32;
        let mut ctx = FnCtx {
            locals: Vec::new(),
            params: Vec::new(),
            scopes: vec![HashMap::new()],
            ret_ty: ret.clone(),
            loop_depth: 0,
            switch_depth: 0,
            switches: Vec::new(),
            labels,
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
            n_labels,
            decl_line,
        });
    }

    /// Recursively register the labels declared anywhere within `stmt` into
    /// `labels`, assigning each a fresh id and diagnosing duplicates.
    fn collect_labels(&mut self, stmt: &Stmt, labels: &mut HashMap<String, u32>) {
        match &stmt.kind {
            StmtKind::Label(name, body) => {
                let next = labels.len() as u32;
                if labels.insert(name.clone(), next).is_some() {
                    self.error(stmt.span, format!("duplicate label '{name}'"));
                }
                self.collect_labels(body, labels);
            }
            StmtKind::Block(stmts) => {
                for s in stmts {
                    self.collect_labels(s, labels);
                }
            }
            StmtKind::If(_, then, els) => {
                self.collect_labels(then, labels);
                if let Some(e) = els {
                    self.collect_labels(e, labels);
                }
            }
            StmtKind::While(_, body)
            | StmtKind::DoWhile(body, _)
            | StmtKind::Switch(_, body)
            | StmtKind::Case(_, body)
            | StmtKind::Default(body) => self.collect_labels(body, labels),
            StmtKind::For(init, _, _, body) => {
                if let Some(i) = init {
                    self.collect_labels(i, labels);
                }
                self.collect_labels(body, labels);
            }
            _ => {}
        }
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
                if !matches!(ctx.ret_ty, CType::Void) && !ctx.ret_ty.is_record() {
                    // A missing value in a value-returning function: default to 0.
                    let zero = TExpr::new(TExprKind::Const(0), ctx.ret_ty.clone(), stmt.span);
                    return Some(TStmt::Return(Some(zero)));
                }
                Some(TStmt::Return(None))
            }
            StmtKind::Return(Some(e)) => {
                let te = self.check_rvalue(ctx, e)?;
                if matches!(ctx.ret_ty, CType::Void) {
                    self.error(stmt.span, "return with a value in a function returning void");
                    return Some(TStmt::Return(None));
                }
                let ret_ty = ctx.ret_ty.clone();
                let conv = self.convert(te, &ret_ty);
                Some(TStmt::Return(Some(conv)))
            }
            StmtKind::Break => {
                if ctx.loop_depth == 0 && ctx.switch_depth == 0 {
                    self.error(stmt.span, "'break' outside of a loop or switch");
                }
                Some(TStmt::Break)
            }
            StmtKind::Continue => {
                if ctx.loop_depth == 0 {
                    self.error(stmt.span, "'continue' outside of a loop");
                }
                Some(TStmt::Continue)
            }
            StmtKind::Switch(expr, body) => self.check_switch(ctx, expr, body),
            StmtKind::Case(value, body) => self.check_case(ctx, *value, body, stmt.span),
            StmtKind::Default(body) => self.check_default(ctx, body, stmt.span),
            StmtKind::Label(name, body) => {
                // The id was assigned during the function-wide label pre-scan.
                let id = ctx.labels[name];
                let b = self.check_stmt(ctx, body)?;
                Some(TStmt::Labeled(id, Box::new(b)))
            }
            StmtKind::Goto(name) => match ctx.labels.get(name) {
                Some(&id) => Some(TStmt::Goto(id)),
                None => {
                    self.error(stmt.span, format!("use of undeclared label '{name}'"));
                    None
                }
            },
        }
    }

    /// Check a `switch`: the controlling expression is integer-promoted, and its
    /// body's `case`/`default` labels are collected (with duplicate detection).
    fn check_switch(&mut self, ctx: &mut FnCtx, expr: &Expr, body: &Stmt) -> Option<TStmt> {
        let ce = self.check_rvalue(ctx, expr)?;
        if !ce.ty.is_integer() {
            self.error(expr.span, format!("switch quantity must be an integer, found '{}'", ce.ty));
        }
        let prom = promote(&ce.ty);
        let value = self.convert(ce, &prom);
        ctx.switches.push(SwitchCollector { prom, cases: Vec::new(), default: None, nmarks: 0 });
        ctx.switch_depth += 1;
        let tbody = self.check_stmt(ctx, body);
        ctx.switch_depth -= 1;
        let coll = ctx.switches.pop().unwrap();
        Some(TStmt::Switch {
            value,
            cases: coll.cases,
            default: coll.default,
            nmarks: coll.nmarks,
            body: Box::new(tbody?),
        })
    }

    fn check_case(&mut self, ctx: &mut FnCtx, value: i128, body: &Stmt, span: Span) -> Option<TStmt> {
        let id = match ctx.switches.last_mut() {
            Some(coll) => {
                let canon = convert_case(value, &coll.prom);
                if coll.cases.iter().any(|(v, _)| *v == canon) {
                    self.error(span, format!("duplicate case value '{value}'"));
                }
                let id = coll.nmarks;
                coll.nmarks += 1;
                coll.cases.push((canon, id));
                id
            }
            None => {
                self.error(span, "'case' label not within a switch");
                let b = self.check_stmt(ctx, body)?;
                return Some(b);
            }
        };
        let b = self.check_stmt(ctx, body)?;
        Some(TStmt::Block(vec![TStmt::CaseMark(id), b]))
    }

    fn check_default(&mut self, ctx: &mut FnCtx, body: &Stmt, span: Span) -> Option<TStmt> {
        let id = match ctx.switches.last_mut() {
            Some(coll) => {
                if coll.default.is_some() {
                    self.error(span, "multiple 'default' labels in one switch");
                }
                let id = coll.nmarks;
                coll.nmarks += 1;
                coll.default = Some(id);
                id
            }
            None => {
                self.error(span, "'default' label not within a switch");
                let b = self.check_stmt(ctx, body)?;
                return Some(b);
            }
        };
        let b = self.check_stmt(ctx, body)?;
        Some(TStmt::Block(vec![TStmt::CaseMark(id), b]))
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
            // Deduce an incomplete array length from its initializer.
            let mut ty = d.ty.clone();
            if let Some(init) = &d.init {
                ty = self.deduce_array_len(&ty, init);
            }
            if matches!(ty, CType::Array(_, 0)) {
                self.error(d.span, "array size is required (variable-length arrays are unsupported)");
            }
            if let CType::Record(id) = &ty
                && !self.records.get(*id).complete
            {
                self.error(d.span, "variable has incomplete struct/union type");
            }
            // Check the initializer *before* the name is in scope (C scoping).
            let init_built = match &d.init {
                Some(init) => self.build_init(ctx, &ty, init, d.span),
                None => None,
            };
            if ctx.scopes.last().unwrap().contains_key(&d.name) {
                self.error(d.span, format!("redeclaration of '{}'", d.name));
            }
            let id = ctx.add_object_aligned(&d.name, ty.clone(), d.align);
            ctx.scopes.last_mut().unwrap().insert(d.name.clone(), id);
            match init_built {
                Some(InitBuilt::Scalar(v)) => out.push(TStmt::InitLocal(id, v)),
                Some(InitBuilt::Aggregate(stores)) => {
                    let size = self.size_of(&ty);
                    out.push(TStmt::InitAggregate { obj: id, size, stores });
                }
                Some(InitBuilt::StructCopy(src)) => {
                    let size = self.size_of(&ty);
                    out.push(TStmt::CopyInit { obj: id, src, size });
                }
                None => {}
            }
        }
        // Wrap the (possibly several) initializers in a block-free sequence.
        match out.len() {
            0 => Some(TStmt::Expr(None)),
            1 => Some(out.pop().unwrap()),
            _ => Some(TStmt::Block(out)),
        }
    }

    /// Check an initializer against `ty`, producing either a single scalar value
    /// or a flat list of `(offset, value)` aggregate stores.
    fn build_init(
        &mut self,
        ctx: &mut FnCtx,
        ty: &CType,
        init: &Init,
        span: Span,
    ) -> Option<InitBuilt> {
        // A `struct`/`union` object may be initialized from a single expression of
        // the same record type (`struct P r = expr;`) — a whole-object copy.
        if ty.is_record()
            && let Init::Expr(e) = init
        {
            let te = self.check_rvalue(ctx, e)?;
            if &te.ty == ty {
                return Some(InitBuilt::StructCopy(te));
            }
            self.error(span, "invalid initializer for a struct/union object");
            return None;
        }
        if ty.is_aggregate() {
            let mut stores = Vec::new();
            self.build_agg_stores(ctx, ty, init, 0, &mut stores, span)?;
            Some(InitBuilt::Aggregate(stores))
        } else {
            Some(InitBuilt::Scalar(self.build_scalar_init(ctx, ty, init, span)?))
        }
    }

    /// Check a scalar initializer (a bare expression, or a single-element brace
    /// list), converting it to the target type.
    fn build_scalar_init(
        &mut self,
        ctx: &mut FnCtx,
        ty: &CType,
        init: &Init,
        span: Span,
    ) -> Option<TExpr> {
        let e = match init {
            Init::Expr(e) => e,
            Init::List(items) if items.len() == 1 && items[0].designators.is_empty() => {
                match &items[0].init {
                    Init::Expr(e) => e,
                    Init::List(_) => {
                        self.error(span, "too many braces around a scalar initializer");
                        return None;
                    }
                }
            }
            Init::List(_) => {
                self.error(span, "invalid brace initializer for a scalar");
                return None;
            }
        };
        let te = self.check_rvalue(ctx, e)?;
        Some(self.convert(te, ty))
    }

    /// Accumulate the scalar stores for an aggregate initializer.
    fn build_agg_stores(
        &mut self,
        ctx: &mut FnCtx,
        ty: &CType,
        init: &Init,
        base: u64,
        out: &mut Vec<AggStore>,
        span: Span,
    ) -> Option<()> {
        match ty {
            CType::Array(elem, n) => {
                // `char[]` initialized from a string literal.
                if let Init::Expr(e) = init
                    && let ExprKind::StrLit(s) = &e.kind
                    && matches!(**elem, CType::Int(IntTy { width: 8, .. }))
                {
                    for (i, &b) in s.iter().enumerate() {
                        if (i as u64) < *n {
                            out.push(AggStore {
                                offset: base + i as u64,
                                value: self.char_const(i128::from(b), elem, span),
                                bits: None,
                            });
                        }
                    }
                    return Some(());
                }
                let items = match init {
                    Init::List(items) => items,
                    Init::Expr(_) => {
                        self.error(span, "array initializer must be a brace-enclosed list");
                        return None;
                    }
                };
                let stride = layout::stride_of(&self.records, elem);
                let mut idx = 0u64;
                for item in items {
                    idx = apply_index_designators(&item.designators, idx);
                    if idx < *n {
                        self.build_member_init(
                            ctx,
                            elem,
                            &item.init,
                            base + idx * stride,
                            out,
                            span,
                        )?;
                    }
                    idx += 1;
                }
                Some(())
            }
            CType::Record(id) => {
                let id = *id;
                let items = match init {
                    Init::List(items) => items,
                    Init::Expr(_) => {
                        self.error(span, "struct/union initializer must be a brace-enclosed list");
                        return None;
                    }
                };
                let mut field_idx = 0usize;
                for item in items {
                    field_idx = self.apply_field_designators(id, &item.designators, field_idx);
                    let nfields = self.records.get(id).fields.len();
                    // Unnamed bit-fields (padding, and `:0`) take no initializer.
                    while field_idx < nfields && is_unnamed_bitfield(&self.records, id, field_idx) {
                        field_idx += 1;
                    }
                    if field_idx < nfields {
                        let fty = self.records.get(id).fields[field_idx].ty.clone();
                        let (foff, bits) =
                            layout::field_placement(&self.records, id, field_idx);
                        match bits {
                            // A bit-field member is always scalar: emit a masked
                            // read-modify-write store directly.
                            Some(bp) => {
                                let v = self.build_scalar_init(ctx, &fty, &item.init, span)?;
                                out.push(AggStore {
                                    offset: base + foff,
                                    value: v,
                                    bits: Some(bp),
                                });
                            }
                            None => self
                                .build_member_init(ctx, &fty, &item.init, base + foff, out, span)?,
                        }
                    }
                    field_idx += 1;
                }
                Some(())
            }
            _ => None,
        }
    }

    /// Initialize one (non-bit-field) array element or struct member: a scalar
    /// store, or a nested aggregate.
    fn build_member_init(
        &mut self,
        ctx: &mut FnCtx,
        ty: &CType,
        init: &Init,
        base: u64,
        out: &mut Vec<AggStore>,
        span: Span,
    ) -> Option<()> {
        if ty.is_aggregate() {
            self.build_agg_stores(ctx, ty, init, base, out, span)
        } else {
            let v = self.build_scalar_init(ctx, ty, init, span)?;
            out.push(AggStore { offset: base, value: v, bits: None });
            Some(())
        }
    }

    /// A character constant of type `ty` (used for string-literal element stores).
    fn char_const(&self, value: i128, ty: &CType, span: Span) -> TExpr {
        TExpr::new(TExprKind::Const(value), ty.clone(), span)
    }

    /// Check a condition expression (must be scalar).
    fn check_cond(&mut self, ctx: &mut FnCtx, e: &Expr) -> Option<TExpr> {
        let te = self.check_rvalue(ctx, e)?;
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
            ExprKind::FloatLit(v, ty) => Some(TExpr::new(TExprKind::FConst(*v), ty.clone(), span)),
            ExprKind::Ident(name) => self.check_ident(ctx, name, span),
            ExprKind::Unary(op, inner) => self.check_unary(ctx, *op, inner, span),
            ExprKind::Binary(op, l, r) => self.check_binary(ctx, *op, l, r, span),
            ExprKind::Assign(compound, l, r) => self.check_assign(ctx, *compound, l, r, span),
            ExprKind::Call(callee, args) => self.check_call(ctx, callee, args, span),
            ExprKind::Cast(ty, inner) => self.check_cast(ctx, ty, inner, span),
            ExprKind::Cond(c, t, f) => self.check_ternary(ctx, c, t, f, span),
            ExprKind::Comma(a, b) => {
                let ta = self.check_expr(ctx, a)?;
                let tb = self.check_rvalue(ctx, b)?;
                let ty = tb.ty.clone();
                Some(TExpr::new(TExprKind::Comma(Box::new(ta), Box::new(tb)), ty, span))
            }
            ExprKind::PreInc(inner) => self.check_incdec(ctx, inner, true, false, span),
            ExprKind::PreDec(inner) => self.check_incdec(ctx, inner, false, false, span),
            ExprKind::PostInc(inner) => self.check_incdec(ctx, inner, true, true, span),
            ExprKind::PostDec(inner) => self.check_incdec(ctx, inner, false, true, span),
            ExprKind::SizeofExpr(inner) => {
                let te = self.check_expr(ctx, inner)?;
                let sz = self.size_of(&te.ty) as i128;
                Some(TExpr::new(TExprKind::Const(sz), size_t(), span))
            }
            ExprKind::SizeofType(ty) => {
                let sz = self.size_of(ty) as i128;
                Some(TExpr::new(TExprKind::Const(sz), size_t(), span))
            }
            ExprKind::StrLit(bytes) => {
                let idx = self.intern_string(bytes.clone());
                let ty = self.globals[idx].ty.clone();
                // A string literal is an lvalue array object (it decays elsewhere).
                Some(TExpr::new(TExprKind::Global(idx), ty, span))
            }
            ExprKind::Index(base, index) => self.check_index(ctx, base, index, span),
            ExprKind::Member(base, name, arrow) => {
                self.check_member(ctx, base, name, *arrow, span)
            }
            ExprKind::AlignofType(ty) => {
                let a = layout::align_of(&self.records, ty) as i128;
                Some(TExpr::new(TExprKind::Const(a), size_t(), span))
            }
            ExprKind::Generic(controlling, assocs) => {
                self.check_generic(ctx, controlling, assocs, span)
            }
            ExprKind::CompoundLiteral(ty, init) => {
                self.check_compound_literal(ctx, ty, init, span)
            }
            ExprKind::VaStart(ap, last) => self.check_va_start(ctx, ap, last, span),
            ExprKind::VaArg(ap, ty) => self.check_va_arg(ctx, ap, ty, span),
            ExprKind::VaEnd(ap) => self.check_va_end(ctx, ap, span),
            ExprKind::VaCopy(dst, src) => self.check_va_copy(ctx, dst, src, span),
        }
    }

    /// Type-check `__builtin_va_start(ap, last)`: `ap` decays to a pointer to its
    /// `__va_list_tag`; `last` (the last named parameter) is evaluated only to
    /// validate the call and is otherwise unused. Result type `void`.
    fn check_va_start(
        &mut self,
        ctx: &mut FnCtx,
        ap: &Expr,
        last: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        let ap_ptr = self.check_va_list_ptr(ctx, ap, span)?;
        // `last` is required by the interface but carries no lowering information.
        let _ = self.check_expr(ctx, last)?;
        Some(TExpr::new(TExprKind::VaStart(Box::new(ap_ptr)), CType::Void, span))
    }

    /// Type-check `__builtin_va_arg(ap, T)`: the result is a value of type `T`.
    fn check_va_arg(&mut self, ctx: &mut FnCtx, ap: &Expr, ty: &CType, span: Span) -> Option<TExpr> {
        let ap_ptr = self.check_va_list_ptr(ctx, ap, span)?;
        if !(ty.is_integer() || ty.is_pointer() || ty.is_float()) {
            self.error(span, "va_arg supports only integer, pointer, and floating types");
            return None;
        }
        Some(TExpr::new(TExprKind::VaArg(Box::new(ap_ptr)), ty.clone(), span))
    }

    /// Type-check `__builtin_va_end(ap)`: a no-op returning `void`.
    fn check_va_end(&mut self, ctx: &mut FnCtx, ap: &Expr, span: Span) -> Option<TExpr> {
        let _ = self.check_va_list_ptr(ctx, ap, span)?;
        Some(TExpr::new(TExprKind::VaEnd, CType::Void, span))
    }

    /// Type-check `__builtin_va_copy(dst, src)`: copy the traversal state.
    fn check_va_copy(
        &mut self,
        ctx: &mut FnCtx,
        dst: &Expr,
        src: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        let d = self.check_va_list_ptr(ctx, dst, span)?;
        let s = self.check_va_list_ptr(ctx, src, span)?;
        Some(TExpr::new(TExprKind::VaCopy(Box::new(d), Box::new(s)), CType::Void, span))
    }

    /// Evaluate a `va_list` argument to a pointer to its `__va_list_tag`. A
    /// `va_list` is `__va_list_tag[1]`, so as a value it decays to a pointer to
    /// its element; a pointer operand is accepted as-is.
    fn check_va_list_ptr(&mut self, ctx: &mut FnCtx, e: &Expr, span: Span) -> Option<TExpr> {
        let te = self.check_rvalue(ctx, e)?;
        if !te.ty.is_pointer() {
            self.error(span, "expected a 'va_list' argument");
            return None;
        }
        Some(te)
    }

    /// Check a `_Generic` selection: type (but do not evaluate) the controlling
    /// expression, select the association whose type matches its type after
    /// lvalue/array/function conversion, and return that association's checked
    /// expression as the result.
    fn check_generic(
        &mut self,
        ctx: &mut FnCtx,
        controlling: &Expr,
        assocs: &[crate::ast::GenericAssoc],
        span: Span,
    ) -> Option<TExpr> {
        // The controlling expression is typed (with lvalue/array/function
        // conversion applied) but never evaluated.
        let ctrl = self.check_rvalue(ctx, controlling)?;
        let cty = ctrl.ty;
        // Diagnose duplicate/compatible association types and multiple defaults.
        let mut default_count = 0usize;
        for (i, a) in assocs.iter().enumerate() {
            match &a.ty {
                None => default_count += 1,
                Some(t) => {
                    if assocs[..i].iter().any(|b| b.ty.as_ref() == Some(t)) {
                        self.error(span, format!("_Generic has two associations for type '{t}'"));
                    }
                }
            }
        }
        if default_count > 1 {
            self.error(span, "_Generic has more than one 'default' association");
        }
        // Select an exact type match, else the default.
        let mut selected: Option<usize> = None;
        for (i, a) in assocs.iter().enumerate() {
            if a.ty.as_ref() == Some(&cty) {
                selected = Some(i);
                break;
            }
        }
        if selected.is_none() {
            selected = assocs.iter().position(|a| a.ty.is_none());
        }
        match selected {
            Some(i) => self.check_expr(ctx, &assocs[i].expr),
            None => {
                self.error(
                    span,
                    format!("no _Generic association matches the controlling type '{cty}'"),
                );
                None
            }
        }
    }

    /// Check a compound literal `(type-name){ init }`: create an unnamed object of
    /// the (array-length-deduced) type, build its initializer, and yield an
    /// lvalue that initializes the object in place when evaluated.
    fn check_compound_literal(
        &mut self,
        ctx: &mut FnCtx,
        ty: &CType,
        init: &Init,
        span: Span,
    ) -> Option<TExpr> {
        if matches!(ty, CType::Void) {
            self.error(span, "compound literal cannot have type 'void'");
            return None;
        }
        let cty = self.deduce_array_len(ty, init);
        if let CType::Record(id) = &cty
            && !self.records.get(*id).complete
        {
            self.error(span, "compound literal has incomplete struct/union type");
            return None;
        }
        let obj = ctx.add_object("", cty.clone());
        let (zero_size, stores) = if cty.is_aggregate() {
            let mut stores = Vec::new();
            self.build_agg_stores(ctx, &cty, init, 0, &mut stores, span)?;
            (self.size_of(&cty), stores)
        } else {
            let v = self.build_scalar_init(ctx, &cty, init, span)?;
            (0u64, vec![AggStore { offset: 0, value: v, bits: None }])
        };
        Some(TExpr::new(TExprKind::CompoundLiteral { obj, zero_size, stores }, cty, span))
    }

    /// Check an expression and apply array-to-pointer decay (the "value of" an
    /// array is a pointer to its first element).
    fn check_rvalue(&mut self, ctx: &mut FnCtx, e: &Expr) -> Option<TExpr> {
        let te = self.check_expr(ctx, e)?;
        Some(self.decay(te))
    }

    /// Decay an array-typed lvalue to a pointer to its first element, or a
    /// function designator to a function pointer.
    fn decay(&mut self, te: TExpr) -> TExpr {
        if matches!(te.ty, CType::Func(_)) {
            let TExpr { kind, ty, span } = te;
            return match kind {
                // `f` → &f (a function pointer).
                TExprKind::FuncRef(idx) => TExpr::new(TExprKind::FuncPtr(idx), CType::ptr_to(ty), span),
                // `*fp` (a dereferenced function pointer) → the pointer itself.
                TExprKind::Deref(inner) => *inner,
                other => TExpr { kind: other, ty, span },
            };
        }
        match te.ty.decayed() {
            Some(ptr_ty) => {
                let span = te.span;
                TExpr::new(TExprKind::Decay(Box::new(te)), ptr_ty, span)
            }
            None => te,
        }
    }

    fn check_index(
        &mut self,
        ctx: &mut FnCtx,
        base: &Expr,
        index: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        // `a[i]` is `*(a + i)`, where either operand may be the pointer.
        let a = self.check_rvalue(ctx, base)?;
        let b = self.check_rvalue(ctx, index)?;
        let (ptr, idx) = if a.ty.is_pointer() { (a, b) } else { (b, a) };
        if !ptr.ty.is_pointer() || !idx.ty.is_integer() {
            self.error(span, "invalid subscript: need a pointer/array and an integer");
            return None;
        }
        let elem = ptr.ty.pointee().cloned().unwrap();
        if matches!(elem, CType::Void) {
            self.error(span, "cannot subscript a pointer to 'void'");
            return None;
        }
        let elem_size = self.size_of(&elem);
        let ptr_ty = ptr.ty.clone();
        let idx_c = self.convert(idx, &CType::long());
        let addr = TExpr::new(
            TExprKind::PtrArith {
                ptr: Box::new(ptr),
                index: Box::new(idx_c),
                elem_size,
                sub: false,
            },
            ptr_ty,
            span,
        );
        Some(TExpr::new(TExprKind::Deref(Box::new(addr)), elem, span))
    }

    fn check_member(
        &mut self,
        ctx: &mut FnCtx,
        base: &Expr,
        name: &str,
        arrow: bool,
        span: Span,
    ) -> Option<TExpr> {
        // `s.m`: `s` is a record lvalue. `p->m`: `p` is a pointer to a record;
        // form the `*p` lvalue first.
        let record_lvalue = if arrow {
            let bt = self.check_rvalue(ctx, base)?;
            let inner = match bt.ty.pointee().cloned() {
                Some(t) => t,
                None => {
                    self.error(span, "'->' requires a pointer to a struct/union");
                    return None;
                }
            };
            if !inner.is_record() {
                self.error(span, "'->' requires a pointer to a struct/union");
                return None;
            }
            TExpr::new(TExprKind::Deref(Box::new(bt)), inner, span)
        } else {
            let bt = self.check_expr(ctx, base)?;
            if !bt.ty.is_record() {
                self.error(span, "'.' requires a struct/union operand");
                return None;
            }
            if !bt.is_lvalue() {
                self.error(span, "'.' requires an lvalue struct/union");
                return None;
            }
            bt
        };
        let CType::Record(id) = &record_lvalue.ty else { unreachable!() };
        let id = *id;
        // Resolve the member, descending through anonymous struct/union members.
        let Some((offset, fty, bits)) = layout::resolve_member_bits(&self.records, id, name) else {
            self.error(span, format!("no member named '{name}' in the struct/union"));
            return None;
        };
        let kind = match bits {
            Some(bits) => TExprKind::BitField { base: Box::new(record_lvalue), offset, bits },
            None => TExprKind::Field { base: Box::new(record_lvalue), offset },
        };
        Some(TExpr::new(kind, fty, span))
    }

    fn check_ident(&mut self, ctx: &mut FnCtx, name: &str, span: Span) -> Option<TExpr> {
        if let Some(id) = ctx.lookup(name) {
            let ty = ctx.locals[id].ty.clone();
            return Some(TExpr::new(TExprKind::Obj(id), ty, span));
        }
        if let Some(&value) = self.enum_consts.get(name) {
            return Some(TExpr::new(TExprKind::Const(value), CType::int(), span));
        }
        if let Some(&idx) = self.global_index.get(name) {
            let ty = self.globals[idx].ty.clone();
            return Some(TExpr::new(TExprKind::Global(idx), ty, span));
        }
        if let Some(&idx) = self.sig_index.get(name) {
            // A function designator: its type is the function type. Used as a
            // value it decays to a function pointer (see `decay`).
            let sig = &self.sigs[idx];
            let fty = CType::Func(Box::new(FuncType {
                ret: sig.ret.clone(),
                params: sig.params.clone(),
                variadic: sig.variadic,
            }));
            return Some(TExpr::new(TExprKind::FuncRef(idx), fty, span));
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
                let te = self.check_rvalue(ctx, inner)?;
                if !te.ty.is_arithmetic() {
                    self.error(span, "unary '+' requires an arithmetic operand");
                    return None;
                }
                let pt = promote(&te.ty);
                Some(self.convert(te, &pt))
            }
            UnaryOp::Neg => {
                let te = self.check_rvalue(ctx, inner)?;
                if !te.ty.is_arithmetic() {
                    self.error(span, "unary '-' requires an arithmetic operand");
                    return None;
                }
                let pt = promote(&te.ty);
                let c = self.convert(te, &pt);
                Some(TExpr::new(TExprKind::Neg(Box::new(c)), pt, span))
            }
            UnaryOp::BitNot => {
                let te = self.check_rvalue(ctx, inner)?;
                if !te.ty.is_integer() {
                    self.error(span, "unary '~' requires an integer operand");
                    return None;
                }
                let pt = promote(&te.ty);
                let c = self.convert(te, &pt);
                Some(TExpr::new(TExprKind::BitNot(Box::new(c)), pt, span))
            }
            UnaryOp::LNot => {
                let te = self.check_rvalue(ctx, inner)?;
                if !te.ty.is_scalar() {
                    self.error(span, "unary '!' requires a scalar operand");
                    return None;
                }
                Some(TExpr::new(TExprKind::LogNot(Box::new(te)), CType::int(), span))
            }
            UnaryOp::Deref => {
                let te = self.check_rvalue(ctx, inner)?;
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
                // `&function` yields a function pointer (same value the designator
                // decays to); `&(*fp)` folds back to the pointer `fp`.
                if matches!(te.ty, CType::Func(_)) {
                    let TExpr { kind, ty, span: sp } = te;
                    return match kind {
                        TExprKind::FuncRef(idx) => {
                            Some(TExpr::new(TExprKind::FuncPtr(idx), CType::ptr_to(ty), sp))
                        }
                        TExprKind::Deref(inner) => Some(*inner),
                        other => Some(TExpr { kind: other, ty, span: sp }),
                    };
                }
                if te.is_bitfield() {
                    self.error(span, "cannot take the address of a bit-field");
                    return None;
                }
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
            let lt = self.check_rvalue(ctx, l)?;
            let rt = self.check_rvalue(ctx, r)?;
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

        let lt = self.check_rvalue(ctx, l)?;
        let rt = self.check_rvalue(ctx, r)?;

        // Pointer arithmetic and comparisons.
        if lt.ty.is_pointer() || rt.ty.is_pointer() {
            return self.check_pointer_binary(op, lt, rt, span);
        }

        // Both arithmetic (integer or floating) from here.
        if !lt.ty.is_arithmetic() || !rt.ty.is_arithmetic() {
            self.error(span, "invalid operands to binary operator");
            return None;
        }
        let float_operand = lt.ty.is_float() || rt.ty.is_float();

        match op {
            BinaryOp::Shl | BinaryOp::Shr => {
                if float_operand {
                    self.error(span, "invalid operands to shift (integer operands required)");
                    return None;
                }
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
            // `%` and the bitwise operators forbid floating operands (a C
            // constraint violation): `%` requires integers, and `& | ^` too.
            BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor
                if float_operand =>
            {
                let sym = match op {
                    BinaryOp::Rem => "%",
                    BinaryOp::BitAnd => "&",
                    BinaryOp::BitOr => "|",
                    _ => "^",
                };
                self.error(
                    span,
                    format!("invalid operands to binary '{sym}' (floating-point operands are not allowed)"),
                );
                None
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
                let elem_size = self.size_of(ptr.ty.pointee().unwrap());
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
                let elem_size = self.size_of(lt.ty.pointee().unwrap());
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
                let elem_size = self.size_of(lt.ty.pointee().unwrap());
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
        if lt.ty.is_array() {
            self.error(l.span, "an array is not assignable");
            return None;
        }
        let target_ty = lt.ty.clone();
        // Whole struct/union assignment copies the object's bytes. The source may
        // be any expression of the same record type (a struct lvalue, or a
        // struct-returning call — both designate readable storage at lowering).
        if compound.is_none() && target_ty.is_record() {
            let rt = self.check_expr(ctx, r)?;
            if rt.ty != target_ty {
                self.error(span, "incompatible struct/union assignment");
                return None;
            }
            let size = self.size_of(&target_ty);
            return Some(TExpr::new(
                TExprKind::CopyAssign { dst: Box::new(lt), src: Box::new(rt), size },
                target_ty,
                span,
            ));
        }
        let rt = self.check_rvalue(ctx, r)?;
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
                    if target_ty.is_float() || rt.ty.is_float() {
                        self.error(span, "invalid operands to shift (integer operands required)");
                        return None;
                    }
                    // A compound shift computes in the promoted left-operand type.
                    promote(&target_ty)
                } else if matches!(
                    op,
                    BinaryOp::Rem | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor
                ) && (target_ty.is_float() || rt.ty.is_float())
                {
                    self.error(
                        span,
                        "invalid operands to this compound assignment (floating-point operands are not allowed)",
                    );
                    return None;
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
        // The callee decays to a function pointer: a bare function designator
        // (direct call) or any pointer-to-function value (indirect call).
        let ct = self.check_rvalue(ctx, callee)?;
        let ft = match &ct.ty {
            CType::Pointer(inner) => match &**inner {
                CType::Func(ft) => ft.clone(),
                _ => {
                    self.error(callee.span, "called object is not a function or function pointer");
                    return None;
                }
            },
            _ => {
                self.error(callee.span, "called object is not a function or function pointer");
                return None;
            }
        };
        if args.len() < ft.params.len() || (!ft.variadic && args.len() != ft.params.len()) {
            self.error(
                span,
                format!("function expects {} argument(s), found {}", ft.params.len(), args.len()),
            );
        }
        let mut targs = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let ta = self.check_rvalue(ctx, a)?;
            let conv = if i < ft.params.len() {
                self.convert(ta, &ft.params[i])
            } else {
                // Variadic argument: default argument promotions.
                let pt = promote(&ta.ty);
                self.convert(ta, &pt)
            };
            targs.push(conv);
        }
        Some(TExpr::new(TExprKind::Call(Box::new(ct), targs), ft.ret.clone(), span))
    }

    fn check_cast(
        &mut self,
        ctx: &mut FnCtx,
        ty: &CType,
        inner: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        let te = self.check_rvalue(ctx, inner)?;
        if matches!(ty, CType::Void) {
            // Cast to void: evaluate for effect; result is void.
            return Some(TExpr::new(TExprKind::Convert(Box::new(te)), CType::Void, span));
        }
        if !te.ty.is_scalar() {
            self.error(span, "cannot cast a non-scalar value");
            return None;
        }
        if !ty.is_scalar() {
            self.error(span, "cannot cast to a non-scalar type");
            return None;
        }
        // A pointer is never converted to or from a floating-point type.
        if (ty.is_pointer() && te.ty.is_float()) || (ty.is_float() && te.ty.is_pointer()) {
            self.error(span, "cannot cast between a pointer and a floating-point type");
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
        let tt = self.check_rvalue(ctx, t)?;
        let ft = self.check_rvalue(ctx, f)?;
        let result_ty = if tt.ty.is_arithmetic() && ft.ty.is_arithmetic() {
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
            Some(p) => self.size_of(p),
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

/// Whether field `idx` of record `id` is an unnamed bit-field (padding or a
/// `:0` unit terminator), which takes no positional initializer.
fn is_unnamed_bitfield(recs: &Records, id: RecordId, idx: usize) -> bool {
    let f = &recs.get(id).fields[idx];
    f.bit_width.is_some() && f.name.is_empty() && !f.anonymous
}

/// The scalar initializer expression of `init`: a bare expression, or the single
/// element of a one-element brace list. Used for a bit-field global initializer.
fn init_scalar_expr(init: &Init) -> Option<&Expr> {
    match init {
        Init::Expr(e) => Some(e),
        Init::List(items) if items.len() == 1 => match &items[0].init {
            Init::Expr(e) => Some(e),
            Init::List(_) => None,
        },
        Init::List(_) => None,
    }
}

/// OR a bit-field's constant value into a global's little-endian byte image: the
/// low `width` bits of `v`, shifted to the field's bit offset within its storage
/// unit at `unit_off`. The image is pre-zeroed, so an OR suffices.
fn write_bitfield_bytes(bytes: &mut [u8], unit_off: u64, v: i128, bp: crate::layout::BitPlacement) {
    let width = bp.width;
    let mask: u128 = if width >= 128 { u128::MAX } else { (1u128 << width) - 1 };
    let field = (v as u128 & mask) << bp.bit_offset;
    let unit_bytes = (bp.unit_bits / 8) as u64;
    let le = field.to_le_bytes();
    for i in 0..unit_bytes {
        if let (Some(dst), Some(src)) =
            (bytes.get_mut((unit_off + i) as usize), le.get(i as usize))
        {
            *dst |= *src;
        }
    }
}

/// Convert a case constant to the switch's promoted controlling type, yielding
/// the canonical in-range value used for duplicate detection and matching (the
/// low `width` bits interpreted with the type's signedness).
fn convert_case(v: i128, ty: &CType) -> i128 {
    let width = ty.int_width().unwrap_or(32);
    if width >= 128 {
        return v;
    }
    let masked = v & ((1i128 << width) - 1);
    if ty.is_signed() && masked & (1i128 << (width - 1)) != 0 {
        masked - (1i128 << width)
    } else {
        masked
    }
}

/// A switch being checked: its promoted controlling type and the case/default
/// marks collected from the body (which may be nested arbitrarily deep).
struct SwitchCollector {
    /// The integer-promoted type of the controlling expression.
    prom: CType,
    /// `(converted case constant, mark id)` pairs, in source order.
    cases: Vec<(i128, u32)>,
    /// The `default:` mark id, once seen.
    default: Option<u32>,
    /// The number of marks allocated so far (the next mark id).
    nmarks: u32,
}

/// The per-function checking context: scopes and the object table.
struct FnCtx {
    locals: Vec<LocalInfo>,
    params: Vec<ObjId>,
    scopes: Vec<HashMap<String, ObjId>>,
    ret_ty: CType,
    loop_depth: u32,
    /// Nesting depth of enclosing `switch` statements (for `break` validity).
    switch_depth: u32,
    /// The stack of enclosing switches; the innermost collects `case`/`default`.
    switches: Vec<SwitchCollector>,
    /// Function-wide label names → label id (labels have their own namespace).
    labels: HashMap<String, u32>,
}

impl FnCtx {
    fn add_object(&mut self, name: &str, ty: CType) -> ObjId {
        self.add_object_aligned(name, ty, None)
    }

    fn add_object_aligned(&mut self, name: &str, ty: CType, align: Option<u64>) -> ObjId {
        let id = self.locals.len();
        self.locals.push(LocalInfo { name: name.to_owned(), ty, align });
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

/// A constant evaluator over the untyped AST for global initializers, resolving
/// enumerator constants and `sizeof`.
fn const_eval_with(e: &Expr, enums: &HashMap<String, i128>, recs: &Records) -> Option<i128> {
    let rec = |x: &Expr| const_eval_with(x, enums, recs);
    match &e.kind {
        ExprKind::IntLit(v, _) => Some(*v),
        ExprKind::Ident(name) => enums.get(name).copied(),
        ExprKind::Unary(op, inner) => {
            let v = rec(inner)?;
            match op {
                UnaryOp::Neg => Some(-v),
                UnaryOp::Plus => Some(v),
                UnaryOp::BitNot => Some(!v),
                UnaryOp::LNot => Some(i128::from(v == 0)),
                _ => None,
            }
        }
        ExprKind::Binary(op, l, r) => {
            let a = rec(l)?;
            let b = rec(r)?;
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
        ExprKind::Cond(c, t, f) => {
            if rec(c)? != 0 { rec(t) } else { rec(f) }
        }
        ExprKind::Cast(_, inner) => rec(inner),
        ExprKind::SizeofType(ty) => Some(layout::size_of(recs, ty) as i128),
        ExprKind::AlignofType(ty) => Some(layout::align_of(recs, ty) as i128),
        _ => None,
    }
}

/// Evaluate a constant expression to an `f64` for a floating-point global
/// initializer: floating and integer literals, enumerators, unary `+`/`-`, the
/// arithmetic operators `+ - * /`, casts, and the conditional operator.
fn const_eval_float(e: &Expr, enums: &HashMap<String, i128>) -> Option<f64> {
    let rec = |x: &Expr| const_eval_float(x, enums);
    match &e.kind {
        ExprKind::FloatLit(v, _) => Some(*v),
        ExprKind::IntLit(v, _) => Some(*v as f64),
        ExprKind::Ident(name) => enums.get(name).map(|&v| v as f64),
        ExprKind::Unary(op, inner) => {
            let v = rec(inner)?;
            match op {
                UnaryOp::Neg => Some(-v),
                UnaryOp::Plus => Some(v),
                _ => None,
            }
        }
        ExprKind::Binary(op, l, r) => {
            let a = rec(l)?;
            let b = rec(r)?;
            match op {
                BinaryOp::Add => Some(a + b),
                BinaryOp::Sub => Some(a - b),
                BinaryOp::Mul => Some(a * b),
                BinaryOp::Div => Some(a / b),
                _ => None,
            }
        }
        ExprKind::Cond(c, t, f) => {
            if rec(c)? != 0.0 { rec(t) } else { rec(f) }
        }
        ExprKind::Cast(ty, inner) => {
            // A cast to a float type rounds; a cast to an integer truncates.
            let v = rec(inner)?;
            match ty.float_ty() {
                Some(crate::ast::FloatTy::F32) => Some(f64::from(v as f32)),
                Some(crate::ast::FloatTy::F64) => Some(v),
                None => Some(v.trunc()),
            }
        }
        _ => None,
    }
}

/// Write a floating-point value into `bytes` at `off` as its little-endian IEEE
/// bit pattern (binary32 for `float`, binary64 for `double`).
fn write_float_bytes(bytes: &mut [u8], off: u64, v: f64, fty: crate::ast::FloatTy) {
    match fty {
        crate::ast::FloatTy::F32 => {
            let le = (v as f32).to_le_bytes();
            for (i, &src) in le.iter().enumerate() {
                if let Some(dst) = bytes.get_mut(off as usize + i) {
                    *dst = src;
                }
            }
        }
        crate::ast::FloatTy::F64 => {
            let le = v.to_le_bytes();
            for (i, &src) in le.iter().enumerate() {
                if let Some(dst) = bytes.get_mut(off as usize + i) {
                    *dst = src;
                }
            }
        }
    }
}

/// The array index selected by an initializer item's designator chain (its first
/// `[index]` designator), or the running `cur` for a positional item.
fn apply_index_designators(desigs: &[Designator], cur: u64) -> u64 {
    match desigs.first() {
        Some(Designator::Index(i)) => *i as u64,
        _ => cur,
    }
}

/// Write the low `size` bytes (little-endian) of `v` into `bytes` at `off`.
fn write_int_bytes(bytes: &mut [u8], off: u64, v: i128, size: u64) {
    let le = v.to_le_bytes();
    for i in 0..size as usize {
        if let (Some(dst), Some(src)) = (bytes.get_mut(off as usize + i), le.get(i)) {
            *dst = *src;
        }
    }
}

/// Write string bytes into `bytes` at `off`, truncated to `n` (the NUL and any
/// remaining bytes are already zero from the caller's zero-fill).
fn write_string_bytes(bytes: &mut [u8], off: u64, s: &[u8], n: u64) {
    for (i, &b) in s.iter().enumerate() {
        if (i as u64) < n
            && let Some(dst) = bytes.get_mut(off as usize + i)
        {
            *dst = b;
        }
    }
}
