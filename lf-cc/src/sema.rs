//! Semantic analysis: name resolution, type checking, and the typed tree.
//!
//! Consumes the untyped [`crate::ast`] tree and produces a [`Program`] of typed
//! nodes in which every C conversion is explicit (integer promotions, the usual
//! arithmetic conversions, pointer scaling, and the implicit conversions on
//! assignment/return/argument are all inserted as [`TExprKind::Convert`] or
//! pointer-arithmetic nodes). Type errors are reported as spanned diagnostics.
//! Lowering ([`crate::lower`]) then walks the typed tree mechanically.

use std::collections::{HashMap, HashSet};

use latticefoundry::support::diagnostics::{Diagnostic, Span};

use crate::ast::{
    AsmStmt, BinaryOp, CType, Designator, Expr, ExprKind, FuncType, Init, IntTy, Quals, RecordId,
    Records, Stmt, StmtKind, Storage, StrKind, SymAttrs, TopLevel, TranslationUnit, UnaryOp, VarDecl,
};
use latticefoundry::ir::Visibility;
use crate::consteval::{self, CInt, ConstEnv};
use crate::cstd::CStd;
use crate::layout;

#[path = "sema_atomic.rs"]
mod atomic;
#[path = "sema_vector.rs"]
mod vector;

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
    /// The templates of the file-scope `asm("...")` declarations, in source
    /// order. They are assembled separately (see `lf_cc::assemble_toplevel_asm`)
    /// and linked alongside the translation unit's object.
    pub toplevel_asm: Vec<String>,
}

/// The diagnostic for an `_Atomic` aggregate object.
const ATOMIC_SCALARS_ONLY: &str = "_Atomic is only supported on scalar (integer, floating or pointer) types";


/// The plain `char` type on this target (signed 8-bit), used for string data.
pub fn char_ty() -> CType {
    CType::Int(IntTy::new(8, true))
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
    /// The function's *symbol* name: its C name, or the GNU asm label given on
    /// a declaration (`int f(int) __asm__("g");` makes this `g`). C-level
    /// lookup goes through the checker's name index, never through this field.
    pub name: String,
    /// The return type.
    pub ret: CType,
    /// The parameter types.
    pub params: Vec<CType>,
    /// Whether the function is variadic.
    pub variadic: bool,
    /// Whether a definition (body) was seen.
    pub defined: bool,
    /// Whether the function has internal linkage (`static`): its symbol is local.
    pub is_static: bool,
    /// An explicit `visibility` attribute on any declaration of the function.
    pub visibility: Option<Visibility>,
    /// Whether any declaration carries `__attribute__((weak))`.
    pub weak: bool,
    /// Whether the definition is a C99 *inline definition* (every file-scope
    /// declaration is `inline` without `extern`): it provides no external
    /// definition, so it is emitted as a private copy that only this
    /// translation unit's calls use.
    pub inline_def: bool,
}

/// A global variable (or an anonymous string-literal object). Its initializer is
/// fully materialized to a little-endian byte image, zero-padded to the type's
/// size; emitting it is a byte copy into a `.data`/`.rodata` section.
#[derive(Clone, Debug)]
pub struct TGlobal {
    /// The global's symbol name (its C name, or its GNU asm label).
    pub name: String,
    /// The global's type (unqualified).
    pub ty: CType,
    /// The object's `volatile`/`_Atomic` qualifiers.
    pub quals: Quals,
    /// The initializer image (already the full size of the object).
    pub bytes: Vec<u8>,
    /// Whether the object belongs in read-only data (string literals).
    pub readonly: bool,
    /// Whether this translation unit provides a *definition* (storage) for the
    /// object. `false` marks a pure external reference (`extern T x;` with no
    /// definition in this TU): no storage is emitted and the symbol is left
    /// undefined for the linker to resolve.
    pub defined: bool,
    /// Whether the object has internal linkage (`static`): its symbol is local.
    pub is_static: bool,
    /// An explicit `visibility` attribute on any declaration of the object.
    pub visibility: Option<Visibility>,
    /// Whether any declaration carries `__attribute__((weak))`.
    pub weak: bool,
    /// Whether the current definition is only *tentative* (a definition without
    /// an initializer). A tentative definition may be superseded by a later
    /// initialized definition; two initialized definitions collide.
    pub tentative: bool,
    /// Relocations that patch address-valued fields of the initializer image
    /// (e.g. a pointer initialized with a string-literal or another object's
    /// address). Each entry patches 8 bytes at `offset` to `symbol + addend`.
    pub relocs: Vec<GlobalReloc>,
    /// Whether the object is thread-local (`_Thread_local`, `__thread`, C23
    /// `thread_local`): one instance per thread, in the TLS sections.
    pub thread_local: bool,
}

/// A relocation within a global's initializer image: the 8-byte field at
/// `offset` holds the address of `symbol` plus `addend`.
#[derive(Clone, Debug)]
pub struct GlobalReloc {
    /// The byte offset of the pointer field within the global's image.
    pub offset: u64,
    /// The name of the symbol whose address is stored.
    pub symbol: String,
    /// A constant byte addend added to the symbol's address.
    pub addend: i64,
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
    /// The object's type (unqualified).
    pub ty: CType,
    /// The object's `volatile`/`_Atomic` qualifiers.
    pub quals: Quals,
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
    /// GNU computed `goto *target`: `target` (a `long`) holds some label's
    /// dispatch number (see `label_value`); branch to whichever of the listed
    /// label ids it names.
    GotoIndirect(TExpr, Vec<u32>),
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
    /// The expression's C type (never qualified).
    pub ty: CType,
    /// For an lvalue, the qualifiers of the object it designates: accesses
    /// through a `volatile` lvalue are volatile, through an `_Atomic` one
    /// atomic (`seq_cst`). Empty for every rvalue.
    pub quals: Quals,
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
    /// `alloca(n)` / `__builtin_alloca(n)`: allocate `n` bytes on the stack
    /// (native `dyn_alloca`), yielding a `void *`. `n` is a byte count.
    DynAlloca(Box<TExpr>),
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
    /// A GNU statement expression `({ ... })`: run the statements, then yield
    /// the value of the final expression statement (absent for a `void` one).
    StmtExpr(Vec<TStmt>, Option<Box<TExpr>>),
    /// An atomic load of `*ptr` (`__atomic_load_n`); the node's type is the
    /// object's.
    AtomicLoad { ptr: Box<TExpr>, order: MemOrder },
    /// An atomic store of `value` (already of the object's type) to `*ptr`;
    /// `void`.
    AtomicStore { ptr: Box<TExpr>, value: Box<TExpr>, order: MemOrder },
    /// An atomic read-modify-write `*ptr = *ptr op value`, yielding the old
    /// value (`fetch_old`) or the new one, typed like the object. On a pointer
    /// object `value` is a `long` byte offset (GCC does not scale it).
    AtomicRmw { op: AtomicOp, ptr: Box<TExpr>, value: Box<TExpr>, order: MemOrder, fetch_old: bool },
    /// A strong compare-exchange of `*ptr`: if it holds the expected value,
    /// store `desired`. With `by_ref`, `expected` points to the expected value
    /// and receives the value found on failure; otherwise it is the expected
    /// value. Yields the `_Bool` success flag, or the value found (`want_old`).
    AtomicCas {
        ptr: Box<TExpr>,
        expected: Box<TExpr>,
        desired: Box<TExpr>,
        by_ref: bool,
        success: MemOrder,
        failure: MemOrder,
        want_old: bool,
    },
    /// `__atomic_thread_fence(order)`; `void`.
    AtomicFence(MemOrder),
    /// A scalar (already of the element type) broadcast to every lane of this
    /// node's vector type.
    VecSplat(Box<TExpr>),
    /// `__builtin_shufflevector`/`__builtin_shuffle`: lane `k` of the result
    /// is lane `mask[k]` of the concatenation of the two (same-typed) vectors.
    VecShuffle(Box<TExpr>, Box<TExpr>, Vec<u32>),
    /// `__builtin_convertvector`: the vector converted lane by lane to this
    /// node's vector type (same lane count).
    VecConvert(Box<TExpr>),
}

/// A C11 memory order (`memory_order_*` / `__ATOMIC_*`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemOrder {
    /// `relaxed`.
    Relaxed,
    /// `consume` (implemented as `acquire`).
    Consume,
    /// `acquire`.
    Acquire,
    /// `release`.
    Release,
    /// `acq_rel`.
    AcqRel,
    /// `seq_cst`.
    SeqCst,
}

/// The operation of an atomic read-modify-write builtin.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AtomicOp {
    /// Exchange: the new value is the operand.
    Xchg,
    /// `old + v` (wrapping).
    Add,
    /// `old - v` (wrapping).
    Sub,
    /// `old & v`.
    And,
    /// `old | v`.
    Or,
    /// `old ^ v`.
    Xor,
    /// `~(old & v)`.
    Nand,
}

impl TExpr {
    fn new(kind: TExprKind, ty: CType, span: Span) -> TExpr {
        TExpr { kind, ty, quals: Quals::NONE, span }
    }

    /// This (lvalue) expression with the qualifiers `q` added.
    fn with_quals(mut self, q: Quals) -> TExpr {
        self.quals = self.quals.union(q);
        self
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
/// unchanged. A `_BitInt(N)` is *not* an integer-promotion candidate (C23): it
/// keeps its own type even when narrower than `int`.
pub(crate) fn promote(ty: &CType) -> CType {
    match ty {
        CType::Bool => CType::int(),
        CType::Int(i) if i.bitint.is_none() && i.width < 32 => CType::int(),
        other => other.clone(),
    }
}

/// The number of value bits of an integer type: `N` for a `_BitInt(N)`, else the
/// storage width (`_Bool` counts as one bit).
fn value_bits(ty: &CType) -> u16 {
    match ty {
        CType::Bool => 1,
        CType::Int(i) => i.value_bits(),
        _ => 32,
    }
}

/// The integer conversion rank used by the usual arithmetic conversions, as a
/// value where a greater number is a greater rank. Rank is dominated by the
/// number of value bits; at equal value bits a standard integer type outranks a
/// bit-precise (`_BitInt`) type (C23 6.3.1.1).
fn int_rank(ty: &CType) -> u32 {
    let standard = !matches!(ty, CType::Int(i) if i.bitint.is_some());
    (u32::from(value_bits(ty)) << 1) | u32::from(standard)
}

/// The unsigned integer type corresponding to a signed integer type (same
/// width / `_BitInt` value-bit count).
fn to_unsigned(ty: &CType) -> CType {
    match ty {
        CType::Int(i) => CType::Int(IntTy { signed: false, ..*i }),
        _ => CType::uint(),
    }
}

/// The usual arithmetic conversions applied to two arithmetic types, yielding
/// their common type. Floating types rank above every integer type: if either
/// operand is `double` the result is `double`; else if either is `float` the
/// result is `float`; otherwise the integer promotions and integer UAC apply.
pub(crate) fn usual_arith(a: &CType, b: &CType) -> CType {
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
    let sa = a.is_signed();
    let sb = b.is_signed();
    if sa == sb {
        // Same signedness: the greater rank wins.
        return if int_rank(&a) >= int_rank(&b) { a } else { b };
    }
    // Mixed signedness: identify the signed and unsigned operands.
    let (u, s) = if sa { (b, a) } else { (a, b) };
    if int_rank(&u) >= int_rank(&s) {
        // Unsigned type has rank >= the signed type: convert to it.
        u
    } else if value_bits(&s) > value_bits(&u) {
        // The signed type can represent every value of the unsigned type.
        s
    } else {
        // Otherwise: the unsigned type corresponding to the signed operand.
        to_unsigned(&s)
    }
}

/// Type-check a translation unit, producing a typed [`Program`] or diagnostics.
pub fn check(unit: &TranslationUnit, std: CStd) -> Result<Program, Vec<Diagnostic>> {
    // `constexpr` objects are named compile-time constants: they resolve as their
    // (typed) value in expressions and as their integer value in constant
    // expressions (alongside enumerators).
    let mut enum_consts: HashMap<String, i128> = unit.enum_consts.iter().cloned().collect();
    let mut constexprs: HashMap<String, (i128, CType)> = HashMap::new();
    for (name, value, ty) in &unit.constexprs {
        enum_consts.insert(name.clone(), *value);
        constexprs.insert(name.clone(), (*value, ty.clone()));
    }
    let mut checker = Checker {
        records: unit.records.clone(),
        enum_consts,
        constexprs,
        std,
        ..Checker::default()
    };
    checker.run(unit);
    if checker.diags.is_empty() {
        Ok(Program {
            funcs: checker.funcs,
            sigs: checker.sigs,
            globals: checker.globals,
            records: checker.records,
            toplevel_asm: checker.toplevel_asm,
        })
    } else {
        Err(checker.diags)
    }
}

#[derive(Default)]
struct Checker {
    sigs: Vec<FuncSig>,
    /// C name → index in `sigs`. Two C names may share one signature when an
    /// asm label binds them to the same symbol.
    sig_index: HashMap<String, usize>,
    /// Symbol name → index in `sigs`, so a declaration whose C name or asm
    /// label names an already-declared *symbol* joins that entity.
    sig_by_symbol: HashMap<String, usize>,
    globals: Vec<TGlobal>,
    /// C name → index in `globals` (file-scope and block-scope `extern` objects).
    global_index: HashMap<String, usize>,
    /// Symbol name → index in `globals` (see `sig_by_symbol`).
    global_by_symbol: HashMap<String, usize>,
    /// File-scope asm templates, in source order.
    toplevel_asm: Vec<String>,
    funcs: Vec<TFunc>,
    diags: Vec<Diagnostic>,
    /// The `struct`/`union` registry (from the parser).
    records: Records,
    /// Enumerator constants resolvable as integer constant expressions.
    enum_consts: HashMap<String, i128>,
    /// C23 `constexpr` objects: name → (reduced value, declared type). Resolved
    /// as a typed compile-time constant in expressions.
    constexprs: HashMap<String, (i128, CType)>,
    /// Deduplicated string-literal objects: (element bytes, encoding) → global
    /// index. The encoding is part of the key so two literals with identical
    /// bytes but distinct element types (`L"…"` vs `U"…"`) are not merged.
    string_pool: HashMap<(Vec<u8>, StrKind), usize>,
    /// The active language dialect (gates implicit function declarations, etc.).
    std: CStd,
    /// Nonzero while checking an unevaluated operand (`sizeof`), where naming a
    /// value of an unsupported type (`_Float128`, `_Complex`) is harmless.
    unevaluated: u32,
    /// The labels (name → id) of the function being checked, so `&&label` can
    /// also appear in the constant initializer of a `static` local (a dispatch
    /// table).
    cur_labels: HashMap<String, u32>,
}

/// The value `&&label` takes for label `id`: a small nonzero dispatch number
/// (not a machine address). `goto *p` branches on it through a multi-way
/// branch over the function's labels, so it only needs to be distinct per
/// label; like any other pointer value, it may be stored, copied, compared
/// and indexed from a table.
fn label_value(id: u32) -> i128 {
    i128::from(id) + 1
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
    fn intern_string(&mut self, mut bytes: Vec<u8>, kind: StrKind) -> usize {
        let width = kind.elem_width();
        // Append the terminating NUL *element* (`width` zero bytes).
        bytes.extend(std::iter::repeat_n(0u8, width as usize));
        let key = (bytes.clone(), kind);
        if let Some(&idx) = self.string_pool.get(&key) {
            return idx;
        }
        let idx = self.globals.len();
        let name = format!(".Lstr.{idx}");
        let ty = CType::Array(Box::new(kind.elem_type()), bytes.len() as u64 / width);
        self.string_pool.insert(key, idx);
        self.globals.push(TGlobal {
            name,
            ty,
            quals: Quals::NONE,
            bytes,
            readonly: true,
            defined: true,
            is_static: true,
            visibility: None,
            weak: false,
            tentative: false,
            relocs: Vec::new(),
            thread_local: false,
        });
        idx
    }

    fn run(&mut self, unit: &TranslationUnit) {
        // `static inline` definitions nothing refers to — the system-header
        // idiom (`__bswap_32`, `__uint16_identity`, ...) — are dropped, as gcc
        // drops them: they would contribute no code, and their bodies may use
        // constructs this compiler does not implement.
        let inline_defs = c99_inline_definitions(unit, self.std);
        let unused_inline = unused_static_inlines(unit, &inline_defs);
        // Pass 1: register every signature and global so bodies can forward- and
        // mutually-reference them.
        for (i, item) in unit.items.iter().enumerate() {
            if unused_inline.contains(&i) {
                continue;
            }
            match item {
                TopLevel::Proto(p) => {
                    let params = p.params.iter().map(|pp| pp.ty.clone()).collect();
                    self.register_sig(
                        &p.name,
                        p.ret.clone(),
                        params,
                        p.variadic,
                        false,
                        p.is_static,
                        p.asm_label.as_deref(),
                        p.span,
                    );
                    self.apply_sig_attrs(&p.name, p.attrs);
                }
                TopLevel::Func(f) => {
                    let params = f.params.iter().map(|pp| pp.ty.clone()).collect();
                    self.register_sig(
                        &f.name,
                        f.ret.clone(),
                        params,
                        f.variadic,
                        true,
                        f.is_static,
                        f.asm_label.as_deref(),
                        f.span,
                    );
                    self.apply_sig_attrs(&f.name, f.attrs);
                    if inline_defs.contains(f.name.as_str())
                        && let Some(&idx) = self.sig_index.get(&f.name)
                    {
                        self.sigs[idx].inline_def = true;
                    }
                }
                TopLevel::Global(g) => self.register_global(g),
                TopLevel::Asm(text) => self.toplevel_asm.push(text.clone()),
            }
        }
        // Pass 2: check each function body.
        for (i, item) in unit.items.iter().enumerate() {
            if let TopLevel::Func(f) = item
                && !unused_inline.contains(&i)
            {
                self.check_func(f);
            }
        }
    }

    /// Merge a declaration's symbol attributes into the signature registered
    /// for the C name `name`: a `visibility` attribute (the last one wins) and
    /// `weak`.
    fn apply_sig_attrs(&mut self, name: &str, attrs: SymAttrs) {
        if let Some(&idx) = self.sig_index.get(name) {
            let sig = &mut self.sigs[idx];
            if attrs.visibility.is_some() {
                sig.visibility = attrs.visibility;
            }
            sig.weak |= attrs.weak;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn register_sig(
        &mut self,
        name: &str,
        ret: CType,
        params: Vec<CType>,
        variadic: bool,
        defined: bool,
        is_static: bool,
        asm_label: Option<&str>,
        span: Span,
    ) {
        // The entity this declaration refers to: a prior declaration of the same
        // C name, or else one already bound to the same *symbol* (a GNU asm label
        // can give two C names one link-level identity: `int my_abs(int)
        // __asm__("abs");` next to `int abs(int);`).
        let symbol = asm_label.unwrap_or(name);
        let prior = self.sig_index.get(name).or_else(|| self.sig_by_symbol.get(symbol)).copied();
        if let Some(idx) = prior {
            self.sig_index.insert(name.to_owned(), idx);
            if let Some(label) = asm_label
                && self.sigs[idx].name != label
            {
                // A label on a redeclaration renames the entity, as long as no
                // earlier declaration fixed a different label or emitted it.
                if self.sigs[idx].name != name || self.sigs[idx].defined {
                    self.error(
                        span,
                        format!(
                            "asm label '{label}' for '{name}' conflicts with its earlier symbol '{}'",
                            self.sigs[idx].name
                        ),
                    );
                    return;
                }
                self.sig_by_symbol.remove(name);
                self.sigs[idx].name = label.to_owned();
                self.sig_by_symbol.insert(label.to_owned(), idx);
            }
            let existing = &mut self.sigs[idx];
            // Any `static` declaration of the name gives the whole entity internal
            // linkage.
            if is_static {
                existing.is_static = true;
            }
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
        self.sigs.push(FuncSig {
            name: symbol.to_owned(),
            ret,
            params,
            variadic,
            defined,
            is_static,
            visibility: None,
            weak: false,
            inline_def: false,
        });
        self.sig_index.insert(name.to_owned(), idx);
        self.sig_by_symbol.insert(symbol.to_owned(), idx);
    }

    fn register_global(&mut self, g: &VarDecl) {
        if matches!(g.ty, CType::Void) {
            self.error(g.span, "global cannot have type 'void'");
            return;
        }
        // An object of a type whose values cannot be computed with may be
        // declared `extern`, but not given storage here.
        if g.storage != Storage::Extern
            && let Some(name) = g.ty.unsupported_value()
        {
            self.error(g.span, format!("objects of type '{name}' are not supported"));
            return;
        }
        // Classify this file-scope declaration by its storage class and whether it
        // carries an initializer (C11 6.9.2):
        //   - `extern T x;` (no init)        -> a *declaration* only (references a
        //                                        definition elsewhere / later).
        //   - `T x;` / `static T x;` (no init) -> a *tentative definition*.
        //   - `... x = init;`                -> an external *definition*.
        let has_init = g.init.is_some();
        let is_decl_only = g.storage == Storage::Extern && !has_init;
        let is_static = g.storage == Storage::Static;

        let quals = g.ty.quals();
        let mut ty = g.ty.unqual().clone();
        if quals.atomic && !ty.is_scalar() {
            self.error(g.span, ATOMIC_SCALARS_ONLY);
            return;
        }
        if let Some(init) = &g.init {
            ty = self.deduce_array_len(&ty, init);
        }

        // Merge with any prior declaration of the same name — or of the same
        // symbol, when an asm label binds this C name to an existing one.
        let symbol = g.asm_label.as_deref().unwrap_or(&g.name);
        let prior =
            self.global_index.get(&g.name).or_else(|| self.global_by_symbol.get(symbol)).copied();
        if let Some(idx) = prior {
            self.global_index.insert(g.name.clone(), idx);
            if !self.relabel_global(idx, g) || !self.same_thread_storage(idx, g) {
                return;
            }
            // A second full definition (both with initializers) is an error.
            if has_init && self.globals[idx].defined && !self.globals[idx].tentative {
                self.error(g.span, format!("redefinition of global '{}'", g.name));
                return;
            }
            // Adopt a more complete type (e.g. `extern int a[];` then `int a[10];`).
            if ty_is_more_complete(&ty, &self.globals[idx].ty) {
                self.globals[idx].ty = ty.clone();
            }
            // A definition (initialized or tentative) upgrades a prior declaration
            // and materializes the storage image.
            if !is_decl_only {
                let final_ty = self.globals[idx].ty.clone();
                let (bytes, relocs) = self.materialize_global(&final_ty, g);
                self.globals[idx].bytes = bytes;
                self.globals[idx].relocs = relocs;
                self.globals[idx].defined = true;
                self.globals[idx].tentative = !has_init;
            }
            if is_static {
                self.globals[idx].is_static = true;
            }
            self.globals[idx].quals = self.globals[idx].quals.union(quals);
            if g.attrs.visibility.is_some() {
                self.globals[idx].visibility = g.attrs.visibility;
            }
            self.globals[idx].weak |= g.attrs.weak;
            return;
        }

        // First declaration of this name.
        let (bytes, relocs) = if is_decl_only {
            (Vec::new(), Vec::new())
        } else {
            self.materialize_global(&ty, g)
        };
        let idx = self.globals.len();
        self.global_index.insert(g.name.clone(), idx);
        self.global_by_symbol.insert(symbol.to_owned(), idx);
        self.globals.push(TGlobal {
            name: symbol.to_owned(),
            ty,
            quals,
            bytes,
            readonly: false,
            defined: !is_decl_only,
            is_static,
            visibility: g.attrs.visibility,
            weak: g.attrs.weak,
            tentative: !has_init && !is_decl_only,
            relocs,
            thread_local: g.thread_local,
        });
    }

    /// Check that redeclaration `g` of global `idx` agrees on thread storage
    /// (C11 6.7.1p3: every declaration of a thread-local object says so).
    fn same_thread_storage(&mut self, idx: usize, g: &VarDecl) -> bool {
        if self.globals[idx].thread_local == g.thread_local {
            return true;
        }
        let msg = if g.thread_local {
            format!("thread-local declaration of '{}' follows a non-thread-local declaration", g.name)
        } else {
            format!("non-thread-local declaration of '{}' follows a thread-local declaration", g.name)
        };
        self.error(g.span, msg);
        false
    }

    /// Apply the asm label of redeclaration `g` to the existing global `idx`.
    /// Returns `false` (after reporting) when it conflicts with an earlier label
    /// or with a definition already emitted under the old symbol.
    fn relabel_global(&mut self, idx: usize, g: &VarDecl) -> bool {
        let Some(label) = g.asm_label.as_deref() else { return true };
        if self.globals[idx].name == label {
            return true;
        }
        if self.globals[idx].name != g.name || self.globals[idx].defined {
            let msg = format!(
                "asm label '{label}' for '{}' conflicts with its earlier symbol '{}'",
                g.name, self.globals[idx].name
            );
            self.error(g.span, msg);
            return false;
        }
        self.global_by_symbol.remove(&g.name);
        self.globals[idx].name = label.to_owned();
        self.global_by_symbol.insert(label.to_owned(), idx);
        true
    }

    /// Build the little-endian storage image (and any address relocations) for a
    /// defined global of type `ty` from its optional initializer.
    fn materialize_global(&mut self, ty: &CType, g: &VarDecl) -> (Vec<u8>, Vec<GlobalReloc>) {
        let size = self.size_of(ty) as usize;
        let mut bytes = vec![0u8; size];
        let mut relocs = Vec::new();
        if let Some(init) = &g.init {
            self.build_global_bytes(ty, init, 0, &mut bytes, &mut relocs, g.span);
        }
        (bytes, relocs)
    }

    /// Evaluate a constant integer expression, resolving enumerators. A map of
    /// file-scope object types lets `sizeof <global>` (e.g. `sizeof table` for an
    /// array whose length is deduced from its initializer) reduce to a constant.
    fn const_eval(&self, e: &Expr) -> Option<i128> {
        // Keyed by C name (a global's `name` is its symbol, which an asm label
        // may have changed).
        let gtypes: HashMap<&str, &CType> =
            self.global_index.iter().map(|(n, &i)| (n.as_str(), &self.globals[i].ty)).collect();
        let env = SemaConsts {
            enums: &self.enum_consts,
            constexprs: &self.constexprs,
            recs: &self.records,
            gtypes,
            labels: &self.cur_labels,
        };
        consteval::eval(e, &env).map(|c| c.value)
    }

    /// Materialize an initializer to little-endian bytes at `off` within `bytes`
    /// (globals must have constant initializers). Address-valued pointer fields
    /// (a string literal or another object's address) append a [`GlobalReloc`] to
    /// `relocs` instead of writing a numeric value.
    fn build_global_bytes(
        &mut self,
        ty: &CType,
        init: &Init,
        off: u64,
        bytes: &mut [u8],
        relocs: &mut Vec<GlobalReloc>,
        span: Span,
    ) {
        let ty = ty.unqual();
        match ty {
            CType::Vector(elem, n) => {
                let Init::List(items) = init else {
                    self.error(span, "a vector global's initializer must be a brace-enclosed list");
                    return;
                };
                let stride = self.size_of(elem);
                let mut idx = 0u64;
                for item in items {
                    idx = apply_index_designators(&item.designators, idx);
                    if idx < u64::from(*n) {
                        self.build_global_bytes(elem, &item.init, off + idx * stride, bytes, relocs, span);
                    }
                    idx += 1;
                }
            }
            CType::Array(elem, n) => {
                // `char[] = "..."` (or a wide array from `L"…"`/`u"…"`/`U"…"`)
                // writes the literal element bytes directly, when the array's
                // element type matches the literal's element width.
                if let Init::Expr(e) = init
                    && let ExprKind::StrLit(s, kind) = &e.kind
                    && matches!(elem.unqual(), CType::Int(IntTy { bitint: None, .. }))
                    && layout::size_of(&self.records, elem) == kind.elem_width()
                {
                    let limit = (*n).saturating_mul(kind.elem_width());
                    write_string_bytes(bytes, off, s, limit);
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
                        self.build_global_bytes(
                            elem,
                            &item.init,
                            off + idx * stride,
                            bytes,
                            relocs,
                            span,
                        );
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
                            None => self.build_global_bytes(
                                &fty,
                                &item.init,
                                off + foff,
                                bytes,
                                relocs,
                                span,
                            ),
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
                } else if let Some(v) = self.const_eval(e) {
                    // Reduce to the type's range so a `_BitInt(N)` global holds a
                    // valid N-bit pattern (loads do not re-normalize).
                    write_int_bytes(bytes, off, reduce_to_type(v, ty), self.size_of(ty));
                } else if ty.is_pointer()
                    && let Some((sym, addend)) = self.const_addr(e)
                {
                    // An address constant (a string literal, or `&object` / a
                    // decayed array or function name, plus a constant offset). The
                    // pointer field is a relocation the linker fills in; `sym` is
                    // `None` for a pure integer address (e.g. a null pointer), which
                    // stays a plain numeric value.
                    match sym {
                        Some(symbol) => relocs.push(GlobalReloc { offset: off, symbol, addend }),
                        None => write_int_bytes(bytes, off, addend as i128, self.size_of(ty)),
                    }
                } else {
                    self.error(e.span, "global initializer must be a constant expression");
                }
            }
        }
    }

    /// Evaluate an initializer expression as an address constant: `Some((symbol,
    /// addend))` where `symbol` is the target object's symbol name (`None` for a
    /// pure integer address such as a null pointer) and `addend` a constant byte
    /// offset. Handles string literals, decayed array/function names, `&object`
    /// (into a member or a constant array element), and casts. Returns `None` if
    /// `e` is not an address constant.
    fn const_addr(&mut self, e: &Expr) -> Option<(Option<String>, i64)> {
        match &e.kind {
            // An integer that is not otherwise a constant-expression (rare here):
            // treat as a pure numeric address with no symbol.
            ExprKind::IntLit(v, _) => Some((None, *v as i64)),
            ExprKind::StrLit(bytes, kind) => {
                let idx = self.intern_string(bytes.clone(), *kind);
                Some((Some(self.globals[idx].name.clone()), 0))
            }
            // A cast does not change the represented address.
            ExprKind::Cast(_, inner) => self.const_addr(inner),
            // An identifier of array or function type decays to its address.
            ExprKind::Ident(name) => {
                if let Some(&idx) = self.sig_index.get(name) {
                    return Some((Some(self.sigs[idx].name.clone()), 0));
                }
                if let Some(&idx) = self.global_index.get(name)
                    && matches!(self.globals[idx].ty, CType::Array(..))
                {
                    return self.static_address_of(idx, e.span).map(|sym| (Some(sym), 0));
                }
                None
            }
            // `&lvalue`: the address of a named object (optionally into a member or
            // a constant array element).
            ExprKind::Unary(UnaryOp::AddrOf, inner) => {
                let (sym, off, _ty) = self.const_lvalue(inner)?;
                Some((Some(sym), off))
            }
            _ => None,
        }
    }

    /// The symbol of global `idx` for an address constant, or `None` (after
    /// reporting) for a thread-local object, whose address differs per thread
    /// and so is not a link-time constant.
    fn static_address_of(&mut self, idx: usize, span: Span) -> Option<String> {
        let g = &self.globals[idx];
        if g.thread_local {
            let msg = format!(
                "the address of thread-local variable '{}' is not a constant (it differs per thread)",
                g.name
            );
            self.error(span, msg);
            return None;
        }
        Some(g.name.clone())
    }

    /// Resolve a constant lvalue designating part of a named object to its
    /// `(symbol, byte-offset, type)`. Descends through constant array subscripts
    /// and struct/union members from a root global identifier.
    fn const_lvalue(&mut self, e: &Expr) -> Option<(String, i64, CType)> {
        match &e.kind {
            ExprKind::Ident(name) => {
                let &idx = self.global_index.get(name)?;
                let sym = self.static_address_of(idx, e.span)?;
                Some((sym, 0, self.globals[idx].ty.clone()))
            }
            ExprKind::Index(base, idx) => {
                let (sym, off, bty) = self.const_lvalue(base)?;
                let elem = match &bty {
                    CType::Array(elem, _) | CType::Pointer(elem) => (**elem).clone(),
                    _ => return None,
                };
                let n = self.const_eval(idx)? as i64;
                let stride = layout::stride_of(&self.records, &elem) as i64;
                Some((sym, off + n * stride, elem))
            }
            ExprKind::Member(base, field, false) => {
                let (sym, off, bty) = self.const_lvalue(base)?;
                if let CType::Record(id) = &bty {
                    let (foff, fty) = layout::resolve_member(&self.records, *id, field)?;
                    return Some((sym, off + foff as i64, fty));
                }
                None
            }
            ExprKind::Cast(_, inner) => self.const_lvalue(inner),
            _ => None,
        }
    }

    /// Deduce the length of an incomplete array type (`T a[]`) from its
    /// initializer, or return `ty` unchanged.
    fn deduce_array_len(&self, ty: &CType, init: &Init) -> CType {
        if let CType::Array(elem, 0) = ty {
            let n = match init {
                Init::Expr(e) => match &e.kind {
                    ExprKind::StrLit(s, kind) => s.len() as u64 / kind.elem_width() + 1,
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
        for ty in f.params.iter().map(|p| &p.ty).chain(std::iter::once(&f.ret)) {
            if let Some(name) = ty.unsupported_value() {
                self.error(
                    f.span,
                    format!("function '{}' passes or returns a '{name}', which is not supported", f.name),
                );
                return;
            }
        }
        let sig_index = self.sig_index[&f.name];
        let ret = f.ret.unqual().clone();
        // Collect every label in the function up front so `goto` may reference a
        // label that appears later (forward references); labels have function
        // scope, not block scope. Duplicate labels are diagnosed here.
        let mut labels = HashMap::new();
        for stmt in &f.body {
            self.collect_labels(stmt, &mut labels);
        }
        let n_labels = labels.len() as u32;
        self.cur_labels = labels.clone();
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
                ctx.scopes.last_mut().unwrap().insert(name, Binding::Local(id));
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
            name: self.sigs[sig_index].name.clone(),
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
            | StmtKind::CaseRange(_, _, body)
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

    /// Check a GNU `asm` statement. The backends encode machine code directly
    /// (no textual assembly in the function pipeline), so the only form accepted
    /// inside a function is the *compiler barrier*: an empty template with no
    /// outputs and no goto labels, e.g. `__asm__ volatile ("" ::: "memory")` or
    /// `asm("" : : "r"(x))`. It emits no instructions; its input operands are
    /// evaluated for their side effects. (The current optimizer never moves
    /// memory operations across statements in a way this barrier would forbid,
    /// so treating it as a no-op is sound for now.) Everything else is rejected
    /// with a clear diagnostic.
    fn check_asm(&mut self, ctx: &mut FnCtx, asm: &AsmStmt, span: Span) -> Option<TStmt> {
        let barrier = asm.template.trim().is_empty()
            && asm.outputs.is_empty()
            && asm.labels.is_empty()
            && !asm.is_goto;
        if !barrier {
            self.error(span, "inline assembly with operands/instructions is not supported yet");
            return None;
        }
        let mut out = Vec::with_capacity(asm.inputs.len());
        for op in &asm.inputs {
            let te = self.check_expr(ctx, &op.expr)?;
            out.push(TStmt::Expr(Some(te)));
        }
        Some(TStmt::Block(out))
    }

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
            StmtKind::Case(value, body) => self.check_case(ctx, *value, *value, body, stmt.span),
            StmtKind::CaseRange(lo, hi, body) => self.check_case(ctx, *lo, *hi, body, stmt.span),
            StmtKind::Default(body) => self.check_default(ctx, body, stmt.span),
            StmtKind::Label(name, body) => {
                // The id was assigned during the function-wide label pre-scan.
                // (A label inside a statement expression is not pre-scanned.)
                let Some(&id) = ctx.labels.get(name) else {
                    self.error(stmt.span, "a label inside a statement expression is not supported");
                    return None;
                };
                let b = self.check_stmt(ctx, body)?;
                Some(TStmt::Labeled(id, Box::new(b)))
            }
            StmtKind::Asm(asm) => self.check_asm(ctx, asm, stmt.span),
            StmtKind::GotoIndirect(e) => {
                let te = self.check_rvalue(ctx, e)?;
                if !te.ty.is_pointer() {
                    self.error(e.span, "the operand of 'goto *' must be a pointer ('&&label')");
                    return None;
                }
                let targets = ctx.labels.values().copied().collect();
                Some(TStmt::GotoIndirect(self.convert(te, &CType::long()), targets))
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

    /// Check `case lo:` (`lo == hi`) or a GNU range `case lo ... hi:`, whose
    /// every value branches to the same mark.
    fn check_case(
        &mut self,
        ctx: &mut FnCtx,
        lo: i128,
        hi: i128,
        body: &Stmt,
        span: Span,
    ) -> Option<TStmt> {
        let id = match ctx.switches.last_mut() {
            Some(coll) => {
                let id = coll.nmarks;
                coll.nmarks += 1;
                let mut dup = None;
                for value in lo..=hi {
                    let canon = convert_case(value, &coll.prom);
                    if dup.is_none() && coll.cases.iter().any(|(v, _)| *v == canon) {
                        dup = Some(value);
                    }
                    coll.cases.push((canon, id));
                }
                if let Some(value) = dup {
                    self.error(span, format!("duplicate case value '{value}'"));
                }
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
            // A thread-local object has static (per-thread) storage duration,
            // so at block scope it must be `static` or `extern` (C11 6.7.1p3).
            if d.thread_local && matches!(d.ty.unqual(), CType::Func(_)) {
                self.error(d.span, format!("function '{}' declared thread-local", d.name));
                continue;
            }
            if d.thread_local && d.storage == Storage::None {
                self.error(
                    d.span,
                    format!("block-scope thread-local variable '{}' must be 'static' or 'extern'", d.name),
                );
                continue;
            }
            if let Some(name) = d.ty.unsupported_value() {
                self.error(d.span, format!("objects of type '{name}' are not supported"));
                continue;
            }
            // Deduce an incomplete array length from its initializer. The
            // object's qualifiers are kept apart from its type.
            let quals = d.ty.quals();
            let mut ty = d.ty.unqual().clone();
            if quals.atomic && !ty.is_scalar() {
                self.error(d.span, ATOMIC_SCALARS_ONLY);
                continue;
            }
            if let Some(init) = &d.init {
                ty = self.deduce_array_len(&ty, init);
            }
            // A block-scope function declaration (`extern char *getlogin();`, or a
            // bare `void foo(int);`) declares an external function, not an object.
            // Register its signature — like a file-scope prototype — so calls to
            // the name resolve to a function designator rather than a stack slot.
            if let CType::Func(ft) = &ty {
                self.register_sig(
                    &d.name,
                    ft.ret.clone(),
                    ft.params.clone(),
                    ft.variadic,
                    false,
                    false,
                    d.asm_label.as_deref(),
                    d.span,
                );
                continue;
            }
            // A block-scope `extern` declaration references an object with
            // external linkage defined elsewhere (C11 6.2.2p4). It gets no local
            // storage: the name resolves to the shared program global, created
            // declaration-only if not yet seen. An incomplete array type here
            // (`extern char default_shell[];`) is legal — it is only a reference.
            if d.storage == Storage::Extern {
                let symbol = d.asm_label.as_deref().unwrap_or(&d.name);
                let prior = self
                    .global_index
                    .get(&d.name)
                    .or_else(|| self.global_by_symbol.get(symbol))
                    .copied();
                let idx = if let Some(i) = prior {
                    if !self.relabel_global(i, d) || !self.same_thread_storage(i, d) {
                        continue;
                    }
                    if ty_is_more_complete(&ty, &self.globals[i].ty) {
                        self.globals[i].ty = ty.clone();
                    }
                    if d.attrs.visibility.is_some() {
                        self.globals[i].visibility = d.attrs.visibility;
                    }
                    self.globals[i].weak |= d.attrs.weak;
                    self.globals[i].quals = self.globals[i].quals.union(quals);
                    i
                } else {
                    let i = self.globals.len();
                    self.global_index.insert(d.name.clone(), i);
                    self.global_by_symbol.insert(symbol.to_owned(), i);
                    self.globals.push(TGlobal {
                        name: symbol.to_owned(),
                        ty: ty.clone(),
                        quals,
                        bytes: Vec::new(),
                        readonly: false,
                        defined: false,
                        is_static: false,
                        visibility: d.attrs.visibility,
                        weak: d.attrs.weak,
                        tentative: false,
                        relocs: Vec::new(),
                        thread_local: d.thread_local,
                    });
                    i
                };
                ctx.scopes.last_mut().unwrap().insert(d.name.clone(), Binding::Static(idx));
                continue;
            }
            if matches!(ty, CType::Array(_, 0)) {
                self.error(d.span, "array size is required (variable-length arrays are unsupported)");
            }
            if let CType::Record(id) = &ty
                && !self.records.get(*id).complete
            {
                self.error(d.span, "variable has incomplete struct/union type");
            }
            // A `static` block-scope object has static storage duration: it is
            // backed by a program global (initialized once, at load), not by a
            // stack slot, and its name resolves to that global.
            if d.storage == Storage::Static {
                if ctx.scopes.last().unwrap().contains_key(&d.name) {
                    self.error(d.span, format!("redeclaration of '{}'", d.name));
                }
                let idx = self.make_static_local(&d.name, &ty, quals, d);
                ctx.scopes.last_mut().unwrap().insert(d.name.clone(), Binding::Static(idx));
                continue;
            }
            // An identifier's scope begins right after its declarator (C11
            // 6.2.1p7), so its own initializer already sees it — `T *p =
            // malloc(sizeof *p);`.
            if ctx.scopes.last().unwrap().contains_key(&d.name) {
                self.error(d.span, format!("redeclaration of '{}'", d.name));
            }
            // An object whose type wants more than the stack's natural 8-byte
            // alignment (a 16-byte vector, an over-aligned record) is
            // over-aligned explicitly, so gcc-compiled code may rely on it.
            let natural = layout::align_of(&self.records, &ty);
            let align = d.align.or((natural > 8).then_some(natural));
            let id = ctx.add_object_aligned(&d.name, ty.clone().qualified(quals), align);
            ctx.scopes.last_mut().unwrap().insert(d.name.clone(), Binding::Local(id));
            let init_built = match &d.init {
                Some(init) => self.build_init(ctx, &ty, init, d.span),
                None => None,
            };
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

    /// Create the backing global for a `static` block-scope object of type `ty`,
    /// returning its index in [`Program::globals`]. Its initializer must be a
    /// constant expression (like a file-scope object); it is materialized to the
    /// object's storage image once. The symbol is given a unique name and
    /// internal linkage (it is not visible outside the function).
    fn make_static_local(&mut self, name: &str, ty: &CType, quals: Quals, d: &VarDecl) -> usize {
        // Materialize the constant initializer first (it may intern string
        // literals, which append globals), then take the next global index.
        let size = self.size_of(ty) as usize;
        let mut bytes = vec![0u8; size];
        let mut relocs = Vec::new();
        if let Some(init) = &d.init {
            self.build_global_bytes(ty, init, 0, &mut bytes, &mut relocs, d.span);
        }
        let idx = self.globals.len();
        // A unique, internally-linked symbol name (the index disambiguates two
        // functions that each declare a `static` of the same source name) —
        // unless an asm label names the symbol explicitly.
        let sym = match &d.asm_label {
            Some(label) => label.clone(),
            None => format!("{name}.static.{idx}"),
        };
        self.globals.push(TGlobal {
            name: sym,
            ty: ty.clone(),
            quals,
            bytes,
            readonly: false,
            defined: true,
            is_static: true,
            visibility: None,
            weak: false,
            tentative: false,
            relocs,
            thread_local: d.thread_local,
        });
        idx
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
        if braced_aggregate(ty, init) {
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
        match ty.unqual() {
            CType::Array(elem, n) => {
                // `char[]` (or a wide `wchar_t[]`/`char16_t[]`/`char32_t[]`)
                // initialized from a matching string literal: one store per
                // element, decoded from the literal's little-endian element bytes.
                let elem = elem.unqual();
                if let Init::Expr(e) = init
                    && let ExprKind::StrLit(s, kind) = &e.kind
                    && matches!(elem, CType::Int(IntTy { bitint: None, .. }))
                    && layout::size_of(&self.records, elem) == kind.elem_width()
                {
                    let w = kind.elem_width() as usize;
                    for (i, chunk) in s.chunks(w).enumerate() {
                        if (i as u64) < *n {
                            let mut buf = [0u8; 8];
                            buf[..chunk.len()].copy_from_slice(chunk);
                            let value = i128::from(u64::from_le_bytes(buf));
                            out.push(AggStore {
                                offset: base + (i * w) as u64,
                                value: self.char_const(value, elem, span),
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
            // A vector initialized lane by lane, like an array.
            CType::Vector(elem, n) => {
                let Init::List(items) = init else {
                    self.error(span, "a vector initializer must be a brace-enclosed list");
                    return None;
                };
                let elem = (**elem).clone();
                let stride = self.size_of(&elem);
                let mut idx = 0u64;
                for item in items {
                    idx = apply_index_designators(&item.designators, idx);
                    if idx < u64::from(*n) {
                        self.build_member_init(ctx, &elem, &item.init, base + idx * stride, out, span)?;
                    } else {
                        self.error(span, "excess elements in a vector initializer");
                        return None;
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
        let ty = ty.unqual();
        if braced_aggregate(ty, init) {
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

    /// Type-check an expression. Every subexpression passes through here, so it
    /// is also where a value of a type lf-cc cannot compute with (`_Float128`,
    /// `_Complex`) is rejected: such types may be *named* by declarations (the
    /// glibc headers declare many `_Float128` functions) but not evaluated.
    fn check_expr(&mut self, ctx: &mut FnCtx, e: &Expr) -> Option<TExpr> {
        let te = self.check_expr_kind(ctx, e)?;
        if self.unevaluated == 0
            && let Some(name) = te.ty.unsupported_value()
        {
            self.error(
                e.span,
                format!("values of type '{name}' are not supported (it may only be declared)"),
            );
            return None;
        }
        // An access to an `_Atomic` object is a real atomic operation, and
        // x86-64 has none wider than 8 bytes without `cmpxchg16b`.
        if self.unevaluated == 0 && te.quals.atomic && self.size_of(&te.ty) > 8 {
            self.error(e.span, format!("_Atomic objects of type '{}' (wider than 8 bytes) are not supported", te.ty));
            return None;
        }
        Some(te)
    }

    fn check_expr_kind(&mut self, ctx: &mut FnCtx, e: &Expr) -> Option<TExpr> {
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
                self.unevaluated += 1;
                let te = self.check_expr(ctx, inner);
                self.unevaluated -= 1;
                let te = te?;
                let sz = self.size_of(&te.ty) as i128;
                Some(TExpr::new(TExprKind::Const(sz), size_t(), span))
            }
            ExprKind::SizeofType(ty) => {
                let sz = self.size_of(ty) as i128;
                Some(TExpr::new(TExprKind::Const(sz), size_t(), span))
            }
            ExprKind::StrLit(bytes, kind) => {
                let idx = self.intern_string(bytes.clone(), *kind);
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
            ExprKind::StmtExpr(stmts) => self.check_stmt_expr(ctx, stmts, span),
            ExprKind::ConvertVector(v, ty) => self.check_convertvector(ctx, v, ty, span),
            ExprKind::LabelAddr(name) => {
                let Some(&id) = ctx.labels.get(name) else {
                    self.error(span, format!("use of undeclared label '{name}'"));
                    return None;
                };
                let v = TExpr::new(TExprKind::Const(label_value(id)), CType::long(), span);
                Some(TExpr::new(TExprKind::Convert(Box::new(v)), CType::ptr_to(CType::Void), span))
            }
        }
    }

    /// Check a GNU statement expression: its statements in a fresh scope, with a
    /// trailing expression statement supplying the value (decayed like any
    /// rvalue, except that a struct/union value stays an lvalue to copy from).
    fn check_stmt_expr(&mut self, ctx: &mut FnCtx, stmts: &[Stmt], span: Span) -> Option<TExpr> {
        ctx.push_scope();
        let mut out = Vec::new();
        let mut value = None;
        let mut ok = true;
        for (i, s) in stmts.iter().enumerate() {
            if i + 1 == stmts.len()
                && let StmtKind::Expr(Some(e)) = &s.kind
            {
                match self.check_expr(ctx, e) {
                    Some(te) if te.ty.is_record() => value = Some(te),
                    Some(te) => value = Some(self.decay(te)),
                    None => ok = false,
                }
                continue;
            }
            if let Some(ts) = self.check_stmt(ctx, s) {
                out.push(ts);
            }
        }
        ctx.pop_scope();
        if !ok {
            return None;
        }
        let ty = value.as_ref().map_or(CType::Void, |v| v.ty.clone());
        Some(TExpr::new(TExprKind::StmtExpr(out, value.map(Box::new)), ty, span))
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
        let ty = ty.unqual();
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
        let quals = ty.quals();
        let ty = ty.unqual();
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
        let obj = ctx.add_object("", cty.clone().qualified(quals));
        let (zero_size, stores) = if braced_aggregate(&cty, init) {
            let mut stores = Vec::new();
            self.build_agg_stores(ctx, &cty, init, 0, &mut stores, span)?;
            (self.size_of(&cty), stores)
        } else {
            let v = self.build_scalar_init(ctx, &cty, init, span)?;
            (0u64, vec![AggStore { offset: 0, value: v, bits: None }])
        };
        Some(TExpr::new(TExprKind::CompoundLiteral { obj, zero_size, stores }, cty, span).with_quals(quals))
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
            let TExpr { kind, ty, quals, span } = te;
            return match kind {
                // `f` → &f (a function pointer).
                TExprKind::FuncRef(idx) => TExpr::new(TExprKind::FuncPtr(idx), CType::ptr_to(ty), span),
                // `*fp` (a dereferenced function pointer) → the pointer itself.
                TExprKind::Deref(inner) => *inner,
                other => TExpr { kind: other, ty, quals, span },
            };
        }
        // An array object's qualifiers belong to its elements: a `volatile`
        // array decays to a pointer to `volatile` elements.
        let decayed = match &te.ty {
            CType::Array(elem, _) => Some(CType::ptr_to((**elem).clone().qualified(te.quals))),
            other => other.decayed(),
        };
        match decayed {
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
        if a.ty.is_vector() {
            return self.check_vector_index(ctx, a, b, span);
        }
        let (ptr, idx) = if a.ty.is_pointer() { (a, b) } else { (b, a) };
        if !ptr.ty.is_pointer() || !idx.ty.is_integer() {
            self.error(span, "invalid subscript: need a pointer/array and an integer");
            return None;
        }
        let qelem = ptr.ty.pointee().cloned().unwrap();
        let (elem, elem_quals) = (qelem.unqual().clone(), qelem.quals());
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
        Some(TExpr::new(TExprKind::Deref(Box::new(addr)), elem, span).with_quals(elem_quals))
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
            let (inner, inner_quals) = (inner.unqual().clone(), inner.quals());
            if !inner.is_record() {
                self.error(span, "'->' requires a pointer to a struct/union");
                return None;
            }
            TExpr::new(TExprKind::Deref(Box::new(bt)), inner, span).with_quals(inner_quals)
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
        // A member of a qualified struct is qualified like it (and by its own
        // declared qualifiers).
        let quals = record_lvalue.quals.union(fty.quals());
        let fty = fty.unqual().clone();
        let kind = match bits {
            Some(bits) => TExprKind::BitField { base: Box::new(record_lvalue), offset, bits },
            None => TExprKind::Field { base: Box::new(record_lvalue), offset },
        };
        Some(TExpr::new(kind, fty, span).with_quals(quals))
    }

    fn check_ident(&mut self, ctx: &mut FnCtx, name: &str, span: Span) -> Option<TExpr> {
        match ctx.lookup(name) {
            Some(Binding::Local(id)) => {
                let ty = ctx.locals[id].ty.clone();
                let quals = ctx.locals[id].quals;
                return Some(TExpr::new(TExprKind::Obj(id), ty, span).with_quals(quals));
            }
            // A `static` block-scope object resolves to its backing global.
            Some(Binding::Static(idx)) => {
                let ty = self.globals[idx].ty.clone();
                let quals = self.globals[idx].quals;
                return Some(TExpr::new(TExprKind::Global(idx), ty, span).with_quals(quals));
            }
            None => {}
        }
        // A `constexpr` object resolves to its (typed) compile-time value; being a
        // pure constant, it is not a modifiable lvalue (so assignment is rejected).
        if let Some((value, ty)) = self.constexprs.get(name) {
            return Some(TExpr::new(TExprKind::Const(*value), ty.clone(), span));
        }
        if let Some(&value) = self.enum_consts.get(name) {
            return Some(TExpr::new(TExprKind::Const(value), CType::int(), span));
        }
        if let Some(&idx) = self.global_index.get(name) {
            let ty = self.globals[idx].ty.clone();
            let quals = self.globals[idx].quals;
            return Some(TExpr::new(TExprKind::Global(idx), ty, span).with_quals(quals));
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
                if te.ty.is_vector() {
                    return Some(te);
                }
                if !te.ty.is_arithmetic() {
                    self.error(span, "unary '+' requires an arithmetic operand");
                    return None;
                }
                let pt = promote(&te.ty);
                Some(self.convert(te, &pt))
            }
            UnaryOp::Neg => {
                let te = self.check_rvalue(ctx, inner)?;
                if te.ty.is_vector() {
                    let ty = te.ty.clone();
                    return Some(TExpr::new(TExprKind::Neg(Box::new(te)), ty, span));
                }
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
                if te.ty.vector_parts().is_some_and(|(e, _)| e.is_integer()) {
                    let ty = te.ty.clone();
                    return Some(TExpr::new(TExprKind::BitNot(Box::new(te)), ty, span));
                }
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
                    Some(p) if matches!(p.unqual(), CType::Void) => {
                        self.error(span, "cannot dereference a 'void *'");
                        None
                    }
                    Some(pointee) => {
                        let (ty, quals) = (pointee.unqual().clone(), pointee.quals());
                        Some(TExpr::new(TExprKind::Deref(Box::new(te)), ty, span).with_quals(quals))
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
                    let TExpr { kind, ty, quals, span: sp } = te;
                    return match kind {
                        TExprKind::FuncRef(idx) => {
                            Some(TExpr::new(TExprKind::FuncPtr(idx), CType::ptr_to(ty), sp))
                        }
                        TExprKind::Deref(inner) => Some(*inner),
                        other => Some(TExpr { kind: other, ty, quals, span: sp }),
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
                // `&x` of a `volatile`/`_Atomic` object points to a qualified type.
                let ty = CType::ptr_to(te.ty.clone().qualified(te.quals));
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

        // GCC vector operations (element-wise).
        if lt.ty.is_vector() || rt.ty.is_vector() {
            return self.check_vector_binary(op, lt, rt, span);
        }

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
        if target_ty.is_vector() || rt.ty.is_vector() {
            return self.check_vector_assign(compound, lt, rt, span);
        }
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
        // `alloca(n)` / `__builtin_alloca(n)`: lowered to the native `dyn_alloca`
        // (a stack allocation freed at function return), not an external call.
        // GNU treats `alloca` as a builtin even when a prototype is in scope.
        if let ExprKind::Ident(name) = &callee.kind
            && (name == "alloca" || name == "__builtin_alloca")
            && args.len() == 1
        {
            let n = self.check_rvalue(ctx, &args[0])?;
            let n = self.convert(n, &size_t());
            return Some(TExpr::new(
                TExprKind::DynAlloca(Box::new(n)),
                CType::ptr_to(CType::Void),
                span,
            ));
        }
        // The GNU atomic builtins (`__atomic_*`, `__sync_*`), unless the program
        // declares a function of that name itself.
        if let ExprKind::Ident(name) = &callee.kind
            && (name.starts_with("__atomic_") || name.starts_with("__sync_"))
            && !self.ident_in_scope(ctx, name)
        {
            return self.check_atomic_builtin(ctx, name, callee.span, args, span);
        }
        // GNU `__builtin_<name>` calls that are not one of the specially-parsed
        // forms (the `va_*` family become dedicated AST nodes in the parser) and
        // that are not explicitly declared are treated as aliases for the
        // library function `<name>` (e.g. `__builtin_memcpy` -> `memcpy`,
        // `__builtin_alloca` -> `alloca`), plus a few pure-compiler builtins.
        if let ExprKind::Ident(name) = &callee.kind
            && !self.ident_in_scope(ctx, name)
        {
            if name.starts_with("__builtin_")
                && let Some(t) = self.try_builtin_alias(ctx, name, callee.span, args, span)
            {
                return Some(t);
            }
            // Implicit function declaration: calling an undeclared function in a
            // pre-C99 dialect (`c89`/`gnu89`, which gcc accepts) declares it
            // implicitly as `extern int name()` — an unprototyped (K&R) function,
            // modelled here as returning `int` with an unchecked argument list.
            if !self.std.is_c99() {
                self.register_sig(name, CType::int(), Vec::new(), true, false, false, None, callee.span);
            }
        }
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
            if self.unevaluated == 0
                && let Some(name) = ft.params.get(i).and_then(CType::unsupported_value)
            {
                self.error(a.span, format!("passing a '{name}' argument is not supported"));
                return None;
            }
            let conv = if i < ft.params.len() {
                self.convert(ta, &ft.params[i])
            } else {
                // Variadic argument: the default argument promotions — the
                // integer promotions, and `float` to `double`.
                let pt = if ta.ty.float_ty() == Some(crate::ast::FloatTy::F32) {
                    CType::double()
                } else {
                    promote(&ta.ty)
                };
                self.convert(ta, &pt)
            };
            targs.push(conv);
        }
        Some(TExpr::new(TExprKind::Call(Box::new(ct), targs), ft.ret.clone(), span))
    }

    /// Whether `name` currently resolves to any ordinary identifier (a local or
    /// `static` object, a `constexpr`, an enumerator, a file-scope object, or a
    /// declared function). Used by `check_call` to detect a call to an
    /// as-yet-undeclared function.
    /// `v = x` / `v op= x` where the target or the value is a vector: both
    /// must be vectors of one size (a scalar right operand of a compound
    /// assignment is broadcast, as in a binary operation).
    fn check_vector_assign(&mut self, compound: Option<BinaryOp>, lt: TExpr, rt: TExpr, span: Span) -> Option<TExpr> {
        let target_ty = lt.ty.clone();
        if !target_ty.is_vector() {
            self.error(span, format!("cannot assign a '{}' to a '{target_ty}'", rt.ty));
            return None;
        }
        match compound {
            None => {
                if !rt.ty.is_vector() {
                    self.error(span, format!("cannot assign a '{}' to a '{target_ty}'", rt.ty));
                    return None;
                }
                let rc = self.vector_operand(rt, &target_ty, span)?;
                Some(TExpr::new(TExprKind::Assign(Box::new(lt), Box::new(rc)), target_ty, span))
            }
            Some(op) => {
                // Type the operation as the binary operator would.
                let probe = self.check_vector_binary(op, lt.clone(), rt, span)?;
                let (TExprKind::Arith(_, _, r) | TExprKind::Shift(_, _, r)) = probe.kind else {
                    self.error(span, "invalid compound assignment on a vector");
                    return None;
                };
                Some(TExpr::new(
                    TExprKind::Compound { lvalue: Box::new(lt), rhs: r, op, compute_ty: target_ty.clone() },
                    target_ty,
                    span,
                ))
            }
        }
    }

    fn ident_in_scope(&self, ctx: &FnCtx, name: &str) -> bool {
        ctx.lookup(name).is_some()
            || self.constexprs.contains_key(name)
            || self.enum_consts.contains_key(name)
            || self.global_index.contains_key(name)
            || self.sig_index.contains_key(name)
    }

    /// Handle a call to an undeclared `__builtin_<base>` identifier. The `va_*`
    /// builtins never reach here (the parser lowers them to dedicated nodes), so
    /// this covers the library-alias builtins and a couple of pure-compiler ones.
    /// Returns `None` if `name` is not a builtin this frontend models (letting
    /// the caller fall through to the normal "undeclared identifier" path).
    fn try_builtin_alias(
        &mut self,
        ctx: &mut FnCtx,
        name: &str,
        callee_span: Span,
        args: &[Expr],
        span: Span,
    ) -> Option<TExpr> {
        let base = name.strip_prefix("__builtin_")?;
        match base {
            // `__builtin_expect(exp, c)`: no branch-prediction hint at -O0; the
            // value is `exp` (typed `long`, per the GCC prototype).
            "expect" => {
                let e = self.check_rvalue(ctx, args.first()?)?;
                return Some(self.convert(e, &CType::long()));
            }
            // `__builtin_constant_p(x)`: conservatively 0 (not a constant).
            "constant_p" => return Some(TExpr::new(TExprKind::Const(0), CType::int(), span)),
            "expect_with_probability" => {
                let e = self.check_rvalue(ctx, args.first()?)?;
                return Some(self.convert(e, &CType::long()));
            }
            // `__builtin_unreachable()`: no code; `__builtin_trap()` aborts.
            "unreachable" => {
                let zero = TExpr::new(TExprKind::Const(0), CType::int(), span);
                return Some(TExpr::new(TExprKind::Convert(Box::new(zero)), CType::Void, span));
            }
            "trap" => {
                if !self.sig_index.contains_key("abort") {
                    self.register_sig("abort", CType::Void, Vec::new(), false, false, false, None, callee_span);
                }
                let callee = Expr { kind: ExprKind::Ident("abort".to_owned()), span: callee_span };
                return self.check_call(ctx, &callee, &[], span);
            }
            // `__builtin_object_size(p, type)`: the size is never known here —
            // "unknown" is (size_t)-1 for types 0/1 and 0 for types 2/3. The
            // pointer operand is not evaluated.
            "object_size" | "dynamic_object_size" => {
                let kind = args.get(1).and_then(|a| self.const_eval(a)).unwrap_or(0);
                let v = if kind & 2 != 0 { 0 } else { i128::from(u64::MAX) };
                return Some(TExpr::new(TExprKind::Const(v), size_t(), span));
            }
            // `__builtin_prefetch(addr, ...)`: a hint; evaluate the operands only.
            "prefetch" => {
                let mut acc = TExpr::new(TExprKind::Const(0), CType::int(), span);
                for a in args {
                    let t = self.check_rvalue(ctx, a)?;
                    let ty = t.ty.clone();
                    acc = TExpr::new(TExprKind::Comma(Box::new(acc), Box::new(t)), ty, span);
                }
                return Some(TExpr::new(TExprKind::Convert(Box::new(acc)), CType::Void, span));
            }
            "assume_aligned" => return self.check_rvalue(ctx, args.first()?),
            "popcount" | "popcountl" | "popcountll" | "clz" | "clzl" | "clzll" | "ctz" | "ctzl"
            | "ctzll" | "ffs" | "ffsl" | "ffsll" | "parity" | "parityl" | "parityll" => {
                return self.builtin_bit_count(ctx, base, args, span);
            }
            "shufflevector" => return self.builtin_shufflevector(ctx, args, span),
            "shuffle" => return self.builtin_shuffle(ctx, args, span),
            "bswap16" => return self.builtin_bswap(ctx, args, 16, span),
            "bswap32" => return self.builtin_bswap(ctx, args, 32, span),
            "bswap64" => return self.builtin_bswap(ctx, args, 64, span),
            "isnan" | "isinf" | "isinf_sign" | "isfinite" | "finite" | "isnormal" | "signbit"
            | "signbitf" | "signbitl" | "fpclassify" | "isgreater" | "isgreaterequal"
            | "isless" | "islessequal" | "islessgreater" | "isunordered" => {
                return self.builtin_float_classify(ctx, base, args, span);
            }
            "va_arg_pack" | "va_arg_pack_len" => {
                self.error(
                    span,
                    format!("'{name}' is only meaningful in always-inline functions and is not supported"),
                );
                return None;
            }
            _ => {}
        }
        if let Some((v, ty)) = float_builtin_const_typed(name) {
            // The argument of `__builtin_nan("")` (a payload string) is ignored.
            return Some(TExpr::new(TExprKind::FConst(v), ty, span));
        }
        // Math library aliases get their true prototype (a variadic implicit
        // declaration would pass a `float` argument promoted to `double`).
        // `long double` is `double` in this subset, so the `l` forms call the
        // `double` functions.
        if let Some((func, fty, nparams)) = math_builtin(base) {
            if !self.sig_index.contains_key(func) {
                let params = vec![fty.clone(); nparams];
                self.register_sig(func, fty, params, false, false, false, None, callee_span);
            }
            let new_callee = Expr { kind: ExprKind::Ident(func.to_owned()), span: callee_span };
            return self.check_call(ctx, &new_callee, args, span);
        }
        // Library-alias builtins: call the libc function `base`, declaring it
        // implicitly if no prototype is in scope. Pointer-returning functions get
        // a `void *` result so the value is usable as a pointer under LP64.
        if !self.sig_index.contains_key(base) {
            let ret = if matches!(
                base,
                "alloca"
                    | "memcpy"
                    | "memmove"
                    | "memset"
                    | "strcpy"
                    | "strncpy"
                    | "strcat"
                    | "strncat"
                    | "strchr"
                    | "strrchr"
                    | "strstr"
                    | "strpbrk"
                    | "strdup"
            ) {
                CType::ptr_to(CType::Void)
            } else {
                CType::int()
            };
            self.register_sig(base, ret, Vec::new(), true, false, false, None, callee_span);
        }
        let new_callee = Expr { kind: ExprKind::Ident(base.to_owned()), span: callee_span };
        self.check_call(ctx, &new_callee, args, span)
    }

    /// Bind `value` to a fresh unnamed local, returning its initialization and
    /// an lvalue naming it (for builtins that use an operand more than once).
    fn bind_temp(&mut self, ctx: &mut FnCtx, value: TExpr) -> (TStmt, TExpr) {
        let ty = value.ty.clone();
        let span = value.span;
        let id = ctx.add_object("", ty.clone());
        (TStmt::InitLocal(id, value), TExpr::new(TExprKind::Obj(id), ty, span))
    }

    /// `__builtin_bswapN(x)`: reverse the bytes of an `N`-bit unsigned value,
    /// built from shifts and masks.
    fn builtin_bswap(
        &mut self,
        ctx: &mut FnCtx,
        args: &[Expr],
        bits: u16,
        span: Span,
    ) -> Option<TExpr> {
        let [arg] = args else {
            self.error(span, "__builtin_bswap takes exactly one argument");
            return None;
        };
        let ret_ty = CType::Int(IntTy::new(bits, false));
        // Compute in at least 32 bits (the promoted width).
        let work = CType::Int(IntTy::new(bits.max(32), false));
        let v = self.check_rvalue(ctx, arg)?;
        if !v.ty.is_integer() {
            self.error(arg.span, "__builtin_bswap requires an integer argument");
            return None;
        }
        let v = self.convert(v, &ret_ty);
        let v = self.convert(v, &work);
        let (init, t) = self.bind_temp(ctx, v);
        let c = |v: i128, ty: &CType| TExpr::new(TExprKind::Const(v), ty.clone(), span);
        let int = CType::int();
        let nbytes = i128::from(bits / 8);
        let mut acc: Option<TExpr> = None;
        for i in 0..nbytes {
            // byte i (from the bottom) moves to position nbytes-1-i.
            let shifted = TExpr::new(
                TExprKind::Shift(BinaryOp::Shr, Box::new(t.clone()), Box::new(c(8 * i, &int))),
                work.clone(),
                span,
            );
            let byte = TExpr::new(
                TExprKind::Arith(BinaryOp::BitAnd, Box::new(shifted), Box::new(c(0xff, &work))),
                work.clone(),
                span,
            );
            let placed = TExpr::new(
                TExprKind::Shift(
                    BinaryOp::Shl,
                    Box::new(byte),
                    Box::new(c(8 * (nbytes - 1 - i), &int)),
                ),
                work.clone(),
                span,
            );
            acc = Some(match acc {
                None => placed,
                Some(a) => TExpr::new(
                    TExprKind::Arith(BinaryOp::BitOr, Box::new(a), Box::new(placed)),
                    work.clone(),
                    span,
                ),
            });
        }
        let result = self.convert(acc?, &ret_ty);
        Some(TExpr::new(TExprKind::StmtExpr(vec![init], Some(Box::new(result))), ret_ty, span))
    }

    /// `__builtin_{popcount,clz,ctz,ffs,parity}{,l,ll}`, computed branch-free
    /// in the operand's unsigned width: a SWAR population count, with `clz` as
    /// the population count of the complement of the right-smeared value, `ctz`
    /// as that of `(x & -x) - 1`, `ffs` as `ctz + 1` (0 for 0), and `parity` as
    /// the count's low bit. (`clz`/`ctz` of 0 are undefined in C; this yields
    /// the width.)
    fn builtin_bit_count(
        &mut self,
        ctx: &mut FnCtx,
        base: &str,
        args: &[Expr],
        span: Span,
    ) -> Option<TExpr> {
        let [arg] = args else {
            self.error(span, format!("__builtin_{base} takes exactly one argument"));
            return None;
        };
        let wide = base.ends_with('l');
        let bits: u16 = if wide { 64 } else { 32 };
        let family = base.trim_end_matches('l');
        let u = CType::Int(IntTy::new(bits, false));
        let v = self.check_rvalue(ctx, arg)?;
        if !v.ty.is_integer() {
            self.error(arg.span, format!("__builtin_{base} requires an integer argument"));
            return None;
        }
        // `ffs` takes a signed operand; the bit pattern is what matters.
        let v = self.convert(v, &CType::Int(IntTy::new(bits, family != "ffs")));
        let v = self.convert(v, &u);
        let mut stmts = Vec::new();
        let c = |k: u128| TExpr::new(TExprKind::Const(k as i128), u.clone(), span);
        let int_c = |k: i128| TExpr::new(TExprKind::Const(k), CType::int(), span);
        let bin = |op: BinaryOp, a: TExpr, b: TExpr| {
            TExpr::new(TExprKind::Arith(op, Box::new(a), Box::new(b)), u.clone(), span)
        };
        let shr = |a: TExpr, k: i128| {
            TExpr::new(TExprKind::Shift(BinaryOp::Shr, Box::new(a), Box::new(int_c(k))), u.clone(), span)
        };
        let (init, x) = self.bind_temp(ctx, v);
        stmts.push(init);
        // The operand whose set bits are counted.
        let counted = match family {
            "clz" => {
                let mut s = x.clone();
                let mut k = 1;
                while k < i128::from(bits) {
                    let (init, t) = self.bind_temp(ctx, bin(BinaryOp::BitOr, s.clone(), shr(s, k)));
                    stmts.push(init);
                    s = t;
                    k *= 2;
                }
                let ones = c(if wide { u128::from(u64::MAX) } else { u128::from(u32::MAX) });
                bin(BinaryOp::BitXor, s, ones)
            }
            "ctz" | "ffs" => {
                let neg = bin(BinaryOp::Sub, c(0), x.clone());
                let low = bin(BinaryOp::BitAnd, x.clone(), neg);
                bin(BinaryOp::Sub, low, c(1))
            }
            _ => x.clone(),
        };
        // SWAR population count.
        let m1 = if wide { 0x5555_5555_5555_5555u128 } else { 0x5555_5555 };
        let m2 = if wide { 0x3333_3333_3333_3333u128 } else { 0x3333_3333 };
        let m4 = if wide { 0x0f0f_0f0f_0f0f_0f0fu128 } else { 0x0f0f_0f0f };
        let h01 = if wide { 0x0101_0101_0101_0101u128 } else { 0x0101_0101 };
        let (init, a) = self.bind_temp(ctx, counted);
        stmts.push(init);
        let a1 = bin(BinaryOp::Sub, a.clone(), bin(BinaryOp::BitAnd, shr(a, 1), c(m1)));
        let (init, b) = self.bind_temp(ctx, a1);
        stmts.push(init);
        let b1 = bin(
            BinaryOp::Add,
            bin(BinaryOp::BitAnd, b.clone(), c(m2)),
            bin(BinaryOp::BitAnd, shr(b, 2), c(m2)),
        );
        let (init, d) = self.bind_temp(ctx, b1);
        stmts.push(init);
        let d1 = bin(BinaryOp::BitAnd, bin(BinaryOp::Add, d.clone(), shr(d, 4)), c(m4));
        let count = shr(bin(BinaryOp::Mul, d1, c(h01)), i128::from(bits) - 8);
        let count = self.convert(count, &CType::int());
        let result = match family {
            "parity" => TExpr::new(
                TExprKind::Arith(BinaryOp::BitAnd, Box::new(count), Box::new(int_c(1))),
                CType::int(),
                span,
            ),
            "ffs" => {
                let is_zero = TExpr::new(
                    TExprKind::Cmp(BinaryOp::Eq, Box::new(x), Box::new(c(0))),
                    CType::int(),
                    span,
                );
                let plus1 = TExpr::new(
                    TExprKind::Arith(BinaryOp::Add, Box::new(count), Box::new(int_c(1))),
                    CType::int(),
                    span,
                );
                TExpr::new(TExprKind::Cond(Box::new(is_zero), Box::new(int_c(0)), Box::new(plus1)), CType::int(), span)
            }
            _ => count,
        };
        Some(TExpr::new(TExprKind::StmtExpr(stmts, Some(Box::new(result))), CType::int(), span))
    }

    /// The floating-point classification builtins behind `<math.h>`'s
    /// `isnan`/`isinf`/`isfinite`/`isnormal`/`signbit`/`fpclassify` and the quiet
    /// comparisons (`isgreater`, ..., `isunordered`), expanded inline.
    fn builtin_float_classify(
        &mut self,
        ctx: &mut FnCtx,
        base: &str,
        args: &[Expr],
        span: Span,
    ) -> Option<TExpr> {
        let int = CType::int();
        let ci = |v: i128| TExpr::new(TExprKind::Const(v), CType::int(), span);
        let cmp = |op: BinaryOp, a: &TExpr, b: TExpr| {
            TExpr::new(TExprKind::Cmp(op, Box::new(a.clone()), Box::new(b)), CType::int(), span)
        };
        let two = matches!(
            base,
            "isgreater" | "isgreaterequal" | "isless" | "islessequal" | "islessgreater"
                | "isunordered"
        );
        let (value_args, lead): (&[Expr], &[Expr]) = if base == "fpclassify" {
            if args.len() != 6 {
                self.error(span, "__builtin_fpclassify takes six arguments");
                return None;
            }
            (&args[5..], &args[..5])
        } else {
            (args, &[])
        };
        if value_args.len() != if two { 2 } else { 1 } {
            self.error(span, format!("wrong number of arguments to __builtin_{base}"));
            return None;
        }
        let mut inits = Vec::new();
        let mut temps = Vec::new();
        let mut vals = Vec::new();
        for a in value_args {
            let v = self.check_rvalue(ctx, a)?;
            if !v.ty.is_arithmetic() {
                self.error(a.span, format!("__builtin_{base} requires a floating-point argument"));
                return None;
            }
            vals.push(v);
        }
        // A two-operand comparison works in the operands' common type; an
        // integer operand of a one-operand test is taken as `double`.
        let common = if two {
            let c = usual_arith(&vals[0].ty, &vals[1].ty);
            if c.is_float() { c } else { CType::double() }
        } else if vals[0].ty.is_float() {
            vals[0].ty.clone()
        } else {
            CType::double()
        };
        for v in vals {
            let v = self.convert(v, &common);
            let (init, t) = self.bind_temp(ctx, v);
            inits.push(init);
            temps.push(t);
        }
        let fc = |v: f64| TExpr::new(TExprKind::FConst(v), common.clone(), span);
        let t = temps[0].clone();
        let isnan = |t: &TExpr| cmp(BinaryOp::Ne, t, t.clone());
        let isinf_pos = cmp(BinaryOp::Eq, &t, fc(f64::INFINITY));
        let isinf_neg = cmp(BinaryOp::Eq, &t, fc(f64::NEG_INFINITY));
        let isinf = TExpr::new(
            TExprKind::LogOr(Box::new(isinf_pos.clone()), Box::new(isinf_neg.clone())),
            int.clone(),
            span,
        );
        // x - x is 0 for a finite x and NaN for an infinity or a NaN.
        let diff =
            TExpr::new(TExprKind::Arith(BinaryOp::Sub, Box::new(t.clone()), Box::new(t.clone())), common.clone(), span);
        let isfinite = cmp(BinaryOp::Eq, &diff, fc(0.0));
        let min = if common.float_ty() == Some(crate::ast::FloatTy::F32) {
            f64::from(f32::MIN_POSITIVE)
        } else {
            f64::MIN_POSITIVE
        };
        let big = TExpr::new(
            TExprKind::LogOr(
                Box::new(cmp(BinaryOp::Ge, &t, fc(min))),
                Box::new(cmp(BinaryOp::Le, &t, fc(-min))),
            ),
            int.clone(),
            span,
        );
        let isnormal =
            TExpr::new(TExprKind::LogAnd(Box::new(isfinite.clone()), Box::new(big)), int.clone(), span);
        let cond = |c: TExpr, a: TExpr, b: TExpr| {
            TExpr::new(TExprKind::Cond(Box::new(c), Box::new(a), Box::new(b)), CType::int(), span)
        };
        let result = match base {
            "isnan" => isnan(&t),
            "isinf" => isinf,
            "isinf_sign" => cond(isinf_pos, ci(1), cond(isinf_neg, ci(-1), ci(0))),
            "isfinite" | "finite" => isfinite,
            "isnormal" => isnormal,
            "signbit" | "signbitf" | "signbitl" => {
                // Read the sign bit through the object representation.
                let bits = if common.float_ty() == Some(crate::ast::FloatTy::F32) { 32 } else { 64 };
                let ity = CType::Int(IntTy::new(bits, true));
                let addr = TExpr::new(TExprKind::AddrOf(Box::new(t.clone())), CType::ptr_to(common.clone()), span);
                let iaddr = TExpr::new(TExprKind::Convert(Box::new(addr)), CType::ptr_to(ity.clone()), span);
                let word = TExpr::new(TExprKind::Deref(Box::new(iaddr)), ity.clone(), span);
                cmp(BinaryOp::Lt, &word, TExpr::new(TExprKind::Const(0), ity, span))
            }
            "fpclassify" => {
                let mut k = Vec::new();
                for a in lead {
                    let v = self.check_rvalue(ctx, a)?;
                    k.push(self.convert(v, &int));
                }
                let zero = cmp(BinaryOp::Eq, &t, fc(0.0));
                cond(
                    isnan(&t),
                    k[0].clone(),
                    cond(isinf, k[1].clone(), cond(isnormal, k[2].clone(), cond(zero, k[4].clone(), k[3].clone()))),
                )
            }
            _ => {
                let u = temps[1].clone();
                let unordered = TExpr::new(
                    TExprKind::LogOr(Box::new(isnan(&t)), Box::new(isnan(&u))),
                    int.clone(),
                    span,
                );
                match base {
                    "isgreater" => cmp(BinaryOp::Gt, &t, u),
                    "isgreaterequal" => cmp(BinaryOp::Ge, &t, u),
                    "isless" => cmp(BinaryOp::Lt, &t, u),
                    "islessequal" => cmp(BinaryOp::Le, &t, u),
                    "islessgreater" => TExpr::new(
                        TExprKind::LogOr(
                            Box::new(cmp(BinaryOp::Lt, &t, u.clone())),
                            Box::new(cmp(BinaryOp::Gt, &t, u)),
                        ),
                        int.clone(),
                        span,
                    ),
                    _ => unordered,
                }
            }
        };
        Some(TExpr::new(TExprKind::StmtExpr(inits, Some(Box::new(result))), int, span))
    }

    fn check_cast(
        &mut self,
        ctx: &mut FnCtx,
        ty: &CType,
        inner: &Expr,
        span: Span,
    ) -> Option<TExpr> {
        // A cast yields an rvalue: qualifiers on its type are meaningless.
        let ty = ty.unqual();
        let te = self.check_rvalue(ctx, inner)?;
        if matches!(ty, CType::Void) {
            // Cast to void: evaluate for effect; result is void.
            return Some(TExpr::new(TExprKind::Convert(Box::new(te)), CType::Void, span));
        }
        // A vector casts to/from a vector or an integer of the same size,
        // reinterpreting the bits.
        if ty.is_vector() || te.ty.is_vector() {
            let ok = (ty.is_vector() || ty.is_integer())
                && (te.ty.is_vector() || te.ty.is_integer())
                && self.size_of(ty) == self.size_of(&te.ty);
            if !ok {
                self.error(span, format!("cannot cast '{}' to '{ty}' (a vector cast needs equal sizes)", te.ty));
                return None;
            }
            return Some(self.convert(te, ty));
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
        if (tt.ty.is_vector() || ft.ty.is_vector()) && tt.ty != ft.ty {
            self.error(span, format!("conditional arms of types '{}' and '{}' do not match", tt.ty, ft.ty));
            return None;
        }
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
        let to = to.unqual();
        if &e.ty == to {
            return e;
        }
        let span = e.span;
        // Only a same-size vector or integer converts to/from a vector (a bit
        // reinterpretation); anything else is a type error, not a conversion.
        if (e.ty.is_vector() || to.is_vector())
            && !matches!(to, CType::Void)
            && !((e.ty.is_vector() || e.ty.is_integer())
                && (to.is_vector() || to.is_integer())
                && self.size_of(&e.ty) == self.size_of(to))
        {
            self.error(span, format!("cannot convert '{}' to '{to}'", e.ty));
            return e;
        }
        TExpr::new(TExprKind::Convert(Box::new(e)), to.clone(), span)
    }
}

/// `size_t` for this target: `unsigned long` (64-bit).
fn size_t() -> CType {
    CType::Int(IntTy::new(64, false))
}

/// Whether type `new_ty` is a more complete version of `old_ty` for the purpose
/// of merging redundant file-scope declarations: it replaces an incomplete
/// array (`T[]`, modelled as length 0) with a sized one (`extern int a[];` then
/// `int a[10];`).
fn ty_is_more_complete(new_ty: &CType, old_ty: &CType) -> bool {
    matches!((old_ty, new_ty), (CType::Array(_, 0), CType::Array(_, n)) if *n != 0)
}

/// Whether an object of type `ty` is initialized element by element from
/// `init`: an array or record, or a vector given a brace-enclosed list.
fn braced_aggregate(ty: &CType, init: &Init) -> bool {
    ty.is_aggregate() || (ty.is_vector() && matches!(init, Init::List(_)))
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
/// What a block-scope name denotes: an object with automatic storage (a stack
/// local or parameter, by [`ObjId`]) or one with static storage duration (a
/// `static` local, held as a program global by index).
#[derive(Clone, Copy, Debug)]
enum Binding {
    /// A stack local or parameter.
    Local(ObjId),
    /// A `static` block-scope object: an index into [`Program::globals`].
    Static(usize),
}

struct FnCtx {
    locals: Vec<LocalInfo>,
    params: Vec<ObjId>,
    scopes: Vec<HashMap<String, Binding>>,
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

    /// Add an object of (possibly qualified) type `ty`; its qualifiers are
    /// recorded apart from its (unqualified) type.
    fn add_object_aligned(&mut self, name: &str, ty: CType, align: Option<u64>) -> ObjId {
        let id = self.locals.len();
        let quals = ty.quals();
        let ty = ty.unqual().clone();
        self.locals.push(LocalInfo { name: name.to_owned(), ty, quals, align });
        id
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn lookup(&self, name: &str) -> Option<Binding> {
        for scope in self.scopes.iter().rev() {
            if let Some(&b) = scope.get(name) {
                return Some(b);
            }
        }
        None
    }
}

/// Sema's view of an integer constant expression: enumerators and `constexpr`
/// objects, `sizeof` a file-scope object or an expression whose type the AST
/// fixes on its own, and `&&label`.
struct SemaConsts<'a> {
    enums: &'a HashMap<String, i128>,
    constexprs: &'a HashMap<String, (i128, CType)>,
    recs: &'a Records,
    gtypes: HashMap<&'a str, &'a CType>,
    labels: &'a HashMap<String, u32>,
}

impl ConstEnv for SemaConsts<'_> {
    fn ident(&self, name: &str) -> Option<CInt> {
        if let Some((v, ty)) = self.constexprs.get(name) {
            return Some(CInt::new(*v, ty));
        }
        self.enums.get(name).map(|&v| CInt::natural(v))
    }

    fn size_of_type(&self, ty: &CType) -> u64 {
        layout::size_of(self.recs, ty)
    }

    fn align_of_type(&self, ty: &CType) -> u64 {
        layout::align_of(self.recs, ty)
    }

    /// `sizeof expr` is a constant expression: its operand is unevaluated and
    /// only its static type is measured. In a global initializer the operand
    /// is one whose type the AST fixes on its own (a string literal, a cast, a
    /// nested `sizeof`, …) — see `ast_static_type`.
    fn size_of_expr(&self, e: &Expr) -> Option<u64> {
        ast_static_type(e, &self.gtypes).map(|ty| layout::size_of(self.recs, &ty))
    }

    fn other(&self, e: &Expr) -> Option<CInt> {
        match &e.kind {
            // `&&label` (GNU labels as values) is the label's dispatch number.
            ExprKind::LabelAddr(name) => {
                self.labels.get(name).map(|&id| CInt::new(label_value(id), &CType::long()))
            }
            _ => None,
        }
    }
}

/// The static type of an expression, computed from the AST alone (no symbol
/// table). Covers the operand forms `sizeof expr` realistically takes in a
/// constant initializer — string literals, casts, compound literals, and the
/// literals/`sizeof` results whose type the node itself carries. Returns `None`
/// when the type would require variable/typedef context the caller lacks.
fn ast_static_type(e: &Expr, gtypes: &HashMap<&str, &CType>) -> Option<CType> {
    match &e.kind {
        ExprKind::IntLit(_, ty) | ExprKind::FloatLit(_, ty) => Some(ty.clone()),
        ExprKind::StrLit(bytes, kind) => {
            let n = bytes.len() as u64 / kind.elem_width() + 1;
            Some(CType::Array(Box::new(kind.elem_type()), n))
        }
        ExprKind::Cast(ty, _) | ExprKind::CompoundLiteral(ty, _) | ExprKind::VaArg(_, ty) => {
            Some(ty.clone())
        }
        ExprKind::SizeofType(_) | ExprKind::SizeofExpr(_) | ExprKind::AlignofType(_) => {
            Some(size_t())
        }
        // A file-scope object (typically `sizeof array` for a length-deduced
        // global array), resolved via the global-type map.
        ExprKind::Ident(name) => gtypes.get(name.as_str()).map(|t| (*t).clone()),
        ExprKind::Cond(_, t, f) => {
            ast_static_type(t, gtypes).or_else(|| ast_static_type(f, gtypes))
        }
        ExprKind::Comma(_, b) => ast_static_type(b, gtypes),
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
                Some(_) => Some(v),
                None => Some(v.trunc()),
            }
        }
        // `HUGE_VAL`, `INFINITY` and `NAN` expand to these builtin calls.
        ExprKind::Call(callee, _) => match &callee.kind {
            ExprKind::Ident(n) => float_builtin_const(n),
            _ => None,
        },
        _ => None,
    }
}

/// The value of a constant-valued floating builtin (`__builtin_huge_val`,
/// `__builtin_inff`, `__builtin_nan("")`, ...) and its type, or `None`.
fn float_builtin_const_typed(name: &str) -> Option<(f64, CType)> {
    let base = name.strip_prefix("__builtin_")?;
    let (stem, ty) = match base.strip_suffix('f') {
        Some(s) if matches!(s, "huge_val" | "inf" | "nan" | "nans") => (s, CType::float()),
        _ => (base.strip_suffix('l').filter(|s| matches!(*s, "huge_val" | "inf" | "nan" | "nans")).unwrap_or(base), CType::double()),
    };
    let v = match stem {
        "huge_val" | "inf" => f64::INFINITY,
        "nan" | "nans" => f64::NAN,
        _ => return None,
    };
    Some((v, ty))
}

/// For a `__builtin_<base>` that aliases a `<math.h>` function, the library
/// function to call, its floating type, and its parameter count. The `...f`
/// forms are `float`; the `...l` forms call the `double` function, since
/// `long double` is `double` in this subset.
fn math_builtin(base: &str) -> Option<(&str, CType, usize)> {
    const UNARY: &[&str] = &[
        "fabs", "sqrt", "floor", "ceil", "trunc", "round", "rint", "nearbyint", "exp", "exp2",
        "expm1", "log", "log2", "log10", "log1p", "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cbrt", "erf", "erfc", "tgamma",
        "lgamma", "logb",
    ];
    const BINARY: &[&str] = &[
        "copysign", "fmod", "pow", "atan2", "fmin", "fmax", "hypot", "fdim", "nextafter",
        "remainder",
    ];
    let arity = |f: &str| {
        if UNARY.contains(&f) {
            Some(1)
        } else if BINARY.contains(&f) {
            Some(2)
        } else {
            None
        }
    };
    if let Some(n) = arity(base) {
        return Some((base, CType::double(), n));
    }
    if let Some(stem) = base.strip_suffix('f')
        && let Some(n) = arity(stem)
    {
        return Some((base, CType::float(), n));
    }
    if let Some(stem) = base.strip_suffix('l')
        && let Some(n) = arity(stem)
    {
        return Some((stem, CType::double(), n));
    }
    None
}

/// [`float_builtin_const_typed`] without the type.
fn float_builtin_const(name: &str) -> Option<f64> {
    float_builtin_const_typed(name).map(|(v, _)| v)
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
        _ => {
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

/// Reduce an integer constant `v` to the representable value of integer type
/// `ty`: mask to the type's value-bit count (`N` for a `_BitInt(N)`), then
/// sign-extend for a signed type. Non-`_BitInt` types are returned unchanged
/// (the byte-writer truncates them to their storage size anyway).
fn reduce_to_type(v: i128, ty: &CType) -> i128 {
    let CType::Int(i) = ty else { return v };
    let Some(n) = i.bitint else { return v };
    let bits = u32::from(n);
    if bits == 0 || bits >= 128 {
        return v;
    }
    let mask = (1i128 << bits) - 1;
    let m = v & mask;
    if i.signed && (m & (1i128 << (bits - 1))) != 0 {
        m - (1i128 << bits)
    } else {
        m
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
fn write_string_bytes(bytes: &mut [u8], off: u64, s: &[u8], limit: u64) {
    for (i, &b) in s.iter().enumerate() {
        if (i as u64) < limit
            && let Some(dst) = bytes.get_mut(off as usize + i)
        {
            *dst = b;
        }
    }
}

/// The names of the functions whose definition in `unit` is a C99 *inline
/// definition* (C11 6.7.4p7): a non-`static` function every file-scope
/// declaration of which carries `inline` and none `extern`. Such a definition
/// provides no external definition of the function. Under GNU89 inline
/// semantics (`-std=gnu89`/`c89`, or the `gnu_inline` attribute) a plain
/// `inline` definition is an ordinary external definition instead.
fn c99_inline_definitions(unit: &TranslationUnit, std: CStd) -> HashSet<String> {
    if !std.is_c99() {
        return HashSet::new();
    }
    // name -> (has an inline definition, every declaration is inline and not extern)
    let mut state: HashMap<&str, (bool, bool)> = HashMap::new();
    for item in &unit.items {
        let (name, inline_only, def) = match item {
            TopLevel::Func(f) if !f.is_static => {
                let c99 = f.is_inline && !f.is_extern && !f.attrs.gnu_inline;
                (f.name.as_str(), c99, c99)
            }
            TopLevel::Proto(p) if !p.is_static => (p.name.as_str(), p.is_inline && !p.is_extern, false),
            _ => continue,
        };
        let e = state.entry(name).or_insert((false, true));
        e.0 |= def;
        e.1 &= inline_only;
    }
    state.into_iter().filter(|(_, (def, only))| *def && *only).map(|(n, _)| n.to_owned()).collect()
}

/// The indices (in `unit.items`) of the `static inline` function definitions
/// (and of the C99 inline definitions named in `inline_defs`) that nothing in
/// the translation unit refers to. References are found by name from every
/// other function body and global initializer, then transitively through the
/// bodies of referenced inline functions.
fn unused_static_inlines(unit: &TranslationUnit, inline_defs: &HashSet<String>) -> HashSet<usize> {
    let is_candidate = |f: &crate::ast::FuncDef| {
        (f.is_static && f.is_inline) || (!f.is_static && inline_defs.contains(&f.name))
    };
    let candidates: HashMap<&str, usize> = unit
        .items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| match item {
            TopLevel::Func(f) if is_candidate(f) => Some((f.name.as_str(), i)),
            _ => None,
        })
        .collect();
    if candidates.is_empty() {
        return HashSet::new();
    }
    let mut names: HashSet<String> = HashSet::new();
    for item in &unit.items {
        match item {
            TopLevel::Func(f) if !is_candidate(f) => {
                for s in &f.body {
                    idents_in_stmt(s, &mut names);
                }
            }
            TopLevel::Global(g) => {
                if let Some(init) = &g.init {
                    idents_in_init(init, &mut names);
                }
            }
            _ => {}
        }
    }
    // Close over the bodies of the referenced inline functions.
    let mut used: HashSet<usize> = HashSet::new();
    loop {
        let newly: Vec<usize> = candidates
            .iter()
            .filter(|(n, i)| names.contains(**n) && !used.contains(*i))
            .map(|(_, &i)| i)
            .collect();
        if newly.is_empty() {
            break;
        }
        for i in newly {
            used.insert(i);
            if let TopLevel::Func(f) = &unit.items[i] {
                for s in &f.body {
                    idents_in_stmt(s, &mut names);
                }
            }
        }
    }
    candidates.values().copied().filter(|i| !used.contains(i)).collect()
}

/// Collect every identifier named by expressions within `s`.
fn idents_in_stmt(s: &Stmt, out: &mut HashSet<String>) {
    match &s.kind {
        StmtKind::Expr(e) | StmtKind::Return(e) => {
            if let Some(e) = e {
                idents_in_expr(e, out);
            }
        }
        StmtKind::Decl(decls) => {
            for d in decls {
                if let Some(init) = &d.init {
                    idents_in_init(init, out);
                }
            }
        }
        StmtKind::Block(stmts) => {
            for s in stmts {
                idents_in_stmt(s, out);
            }
        }
        StmtKind::If(c, t, e) => {
            idents_in_expr(c, out);
            idents_in_stmt(t, out);
            if let Some(e) = e {
                idents_in_stmt(e, out);
            }
        }
        StmtKind::While(c, b) | StmtKind::DoWhile(b, c) | StmtKind::Switch(c, b) => {
            idents_in_expr(c, out);
            idents_in_stmt(b, out);
        }
        StmtKind::For(init, c, step, b) => {
            if let Some(i) = init {
                idents_in_stmt(i, out);
            }
            for e in [c, step].into_iter().flatten() {
                idents_in_expr(e, out);
            }
            idents_in_stmt(b, out);
        }
        StmtKind::Case(_, b)
        | StmtKind::CaseRange(_, _, b)
        | StmtKind::Default(b)
        | StmtKind::Label(_, b) => {
            idents_in_stmt(b, out);
        }
        StmtKind::Asm(asm) => {
            for op in asm.outputs.iter().chain(asm.inputs.iter()) {
                idents_in_expr(&op.expr, out);
            }
        }
        StmtKind::GotoIndirect(e) => idents_in_expr(e, out),
        StmtKind::Break | StmtKind::Continue | StmtKind::Goto(_) => {}
    }
}

/// Collect every identifier named within an initializer.
fn idents_in_init(init: &Init, out: &mut HashSet<String>) {
    match init {
        Init::Expr(e) => idents_in_expr(e, out),
        Init::List(items) => {
            for it in items {
                idents_in_init(&it.init, out);
            }
        }
    }
}

/// Collect every identifier named within `e`.
fn idents_in_expr(e: &Expr, out: &mut HashSet<String>) {
    match &e.kind {
        ExprKind::Ident(n) => {
            out.insert(n.clone());
        }
        ExprKind::IntLit(..)
        | ExprKind::FloatLit(..)
        | ExprKind::StrLit(..)
        | ExprKind::SizeofType(_)
        | ExprKind::AlignofType(_)
        | ExprKind::LabelAddr(_) => {}
        ExprKind::Unary(_, a)
        | ExprKind::Cast(_, a)
        | ExprKind::PreInc(a)
        | ExprKind::PreDec(a)
        | ExprKind::PostInc(a)
        | ExprKind::PostDec(a)
        | ExprKind::SizeofExpr(a)
        | ExprKind::Member(a, _, _)
        | ExprKind::VaArg(a, _)
        | ExprKind::ConvertVector(a, _)
        | ExprKind::VaEnd(a) => idents_in_expr(a, out),
        ExprKind::Binary(_, a, b)
        | ExprKind::Assign(_, a, b)
        | ExprKind::Comma(a, b)
        | ExprKind::Index(a, b)
        | ExprKind::VaStart(a, b)
        | ExprKind::VaCopy(a, b) => {
            idents_in_expr(a, out);
            idents_in_expr(b, out);
        }
        ExprKind::Cond(a, b, c) => {
            idents_in_expr(a, out);
            idents_in_expr(b, out);
            idents_in_expr(c, out);
        }
        ExprKind::Call(f, args) => {
            idents_in_expr(f, out);
            for a in args {
                idents_in_expr(a, out);
            }
        }
        ExprKind::Generic(c, assocs) => {
            idents_in_expr(c, out);
            for a in assocs {
                idents_in_expr(&a.expr, out);
            }
        }
        ExprKind::CompoundLiteral(_, init) => idents_in_init(init, out),
        ExprKind::StmtExpr(stmts) => {
            for s in stmts {
                idents_in_stmt(s, out);
            }
        }
    }
}
