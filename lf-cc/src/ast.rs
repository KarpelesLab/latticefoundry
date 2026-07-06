//! The C abstract syntax tree and the C type model.
//!
//! The parser ([`crate::parse`]) produces these untyped nodes directly from the
//! token stream; [`crate::sema`] then type-checks them and produces its own
//! typed tree. Every node carries a source [`Span`] so diagnostics can point at
//! the offending construct.

use std::fmt;

use latticefoundry::support::diagnostics::Span;

/// A C type in the subset: `void`, `_Bool`, the integer types (tracked as an
/// explicit width plus signedness), pointers, arrays, and aggregates
/// (`struct`/`union`, referenced by index into a shared [`Records`] registry).
///
/// Floating-point types (`float`, `double`) are tracked as a [`FloatTy`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum CType {
    /// `void`.
    Void,
    /// `_Bool`.
    Bool,
    /// An integer type of a given width in bits and signedness.
    Int(IntTy),
    /// A floating-point type (`float` → `F32`, `double` → `F64`). `long double`
    /// is modelled as `F64` in this subset.
    Float(FloatTy),
    /// `T *` — a pointer to `T`.
    Pointer(Box<CType>),
    /// `T[N]` — a fixed-length array of `N` elements of `T`. `N == 0` marks an
    /// incomplete array (`T a[]`) whose length is deduced from its initializer.
    Array(Box<CType>, u64),
    /// A `struct` or `union` type, identified by its [`RecordId`] in the shared
    /// [`Records`] registry (so a record can refer to itself through a pointer).
    Record(RecordId),
    /// A function type (return type, parameter types, variadic flag). A function
    /// *designator* has this type; used as a value it decays to `Pointer(Func)`
    /// (a function pointer), which is the only form that reaches storage.
    Func(Box<FuncType>),
}

/// The type of a function: its return type, parameter types, and whether it is
/// variadic. A function pointer is `CType::Pointer(Box::new(CType::Func(..)))`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FuncType {
    /// The return type.
    pub ret: CType,
    /// The parameter types (already adjusted: arrays/functions decayed).
    pub params: Vec<CType>,
    /// Whether the function is variadic (`...`).
    pub variadic: bool,
}

/// The (width, signedness) of a C integer type. Width is in bits: 8 (`char`),
/// 16 (`short`), 32 (`int`), 64 (`long`/`long long`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IntTy {
    /// Width in bits.
    pub width: u16,
    /// Whether the type is signed.
    pub signed: bool,
}

/// The precision of a floating-point type: `float` (IEEE binary32) or `double`
/// (IEEE binary64). `long double` is treated as `F64` in this subset.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatTy {
    /// Single precision (`float`, 4 bytes).
    F32,
    /// Double precision (`double`, 8 bytes).
    F64,
}

impl FloatTy {
    /// The width in bits of this format (32 or 64).
    pub fn bits(self) -> u16 {
        match self {
            FloatTy::F32 => 32,
            FloatTy::F64 => 64,
        }
    }
}

/// An index into a [`Records`] registry, naming one `struct`/`union` definition.
pub type RecordId = usize;

/// Whether a record is a `struct` (fields laid out sequentially) or a `union`
/// (all members overlaid at offset 0).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RecordKind {
    /// A `struct`.
    Struct,
    /// A `union`.
    Union,
}

/// One member of a record.
#[derive(Clone, Debug)]
pub struct Field {
    /// The member name (empty for an anonymous struct/union member, or for an
    /// unnamed bit-field).
    pub name: String,
    /// The member type.
    pub ty: CType,
    /// Whether this is an anonymous struct/union member (C11): its own members
    /// are accessed as if they belonged to the enclosing record.
    pub anonymous: bool,
    /// An explicit `_Alignas`/`alignas` alignment override for this member, if
    /// any (raising the member's alignment in [`crate::layout`]).
    pub align: Option<u64>,
    /// If this is a bit-field (`type name : width`), its declared width in bits.
    /// `Some(0)` marks an unnamed `:0` bit-field (which forces the next member to
    /// the next storage-unit boundary of its declared type). `None` marks an
    /// ordinary (non-bit-field) member. The concrete bit placement (storage-unit
    /// byte offset, unit width, and bit offset) is computed in [`crate::layout`].
    pub bit_width: Option<u32>,
}

/// A `struct`/`union` definition. A forward declaration (`struct T;`) or a use
/// before definition is `complete == false` with empty `fields`.
#[derive(Clone, Debug)]
pub struct RecordDef {
    /// Whether this is a `struct` or a `union`.
    pub kind: RecordKind,
    /// The tag name, if the record is tagged.
    pub tag: Option<String>,
    /// The members in declaration order (empty while incomplete).
    pub fields: Vec<Field>,
    /// Whether a full definition (a member list) has been seen.
    pub complete: bool,
}

/// The registry of every `struct`/`union` definition in a translation unit,
/// shared by the parser (which populates it), sema, and lowering.
#[derive(Clone, Debug, Default)]
pub struct Records {
    /// Definitions, indexed by [`RecordId`].
    pub defs: Vec<RecordDef>,
}

impl Records {
    /// Borrow the definition of a record.
    pub fn get(&self, id: RecordId) -> &RecordDef {
        &self.defs[id]
    }

    /// Look up a member by name, returning its field index and definition.
    pub fn field(&self, id: RecordId, name: &str) -> Option<(usize, &Field)> {
        self.defs[id].fields.iter().enumerate().find(|(_, f)| f.name == name)
    }
}

impl CType {
    /// The signed `int` type (`i32`).
    pub fn int() -> CType {
        CType::Int(IntTy { width: 32, signed: true })
    }

    /// The `unsigned int` type.
    pub fn uint() -> CType {
        CType::Int(IntTy { width: 32, signed: false })
    }

    /// The `long` type (`i64`, signed).
    pub fn long() -> CType {
        CType::Int(IntTy { width: 64, signed: true })
    }

    /// The `float` type (IEEE binary32).
    pub fn float() -> CType {
        CType::Float(FloatTy::F32)
    }

    /// The `double` type (IEEE binary64).
    pub fn double() -> CType {
        CType::Float(FloatTy::F64)
    }

    /// A pointer to `pointee`.
    pub fn ptr_to(pointee: CType) -> CType {
        CType::Pointer(Box::new(pointee))
    }

    /// Whether this is any integer type (including `_Bool`).
    pub fn is_integer(&self) -> bool {
        matches!(self, CType::Int(_) | CType::Bool)
    }

    /// Whether this is a floating-point type (`float`/`double`).
    pub fn is_float(&self) -> bool {
        matches!(self, CType::Float(_))
    }

    /// The floating-point precision of a float type, if this is one.
    pub fn float_ty(&self) -> Option<FloatTy> {
        match self {
            CType::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Whether this is an arithmetic type (integer or floating-point).
    pub fn is_arithmetic(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// Whether this is a pointer type.
    pub fn is_pointer(&self) -> bool {
        matches!(self, CType::Pointer(_))
    }

    /// Whether this is an array type.
    pub fn is_array(&self) -> bool {
        matches!(self, CType::Array(..))
    }

    /// Whether this is a `struct`/`union` type.
    pub fn is_record(&self) -> bool {
        matches!(self, CType::Record(_))
    }

    /// Whether this is a function type (a function designator's type).
    pub fn is_function(&self) -> bool {
        matches!(self, CType::Func(_))
    }

    /// Whether this is an aggregate (array or record) type.
    pub fn is_aggregate(&self) -> bool {
        self.is_array() || self.is_record()
    }

    /// Whether this is a scalar type usable in a condition (integer, pointer, or
    /// floating-point).
    pub fn is_scalar(&self) -> bool {
        self.is_integer() || self.is_pointer() || self.is_float()
    }

    /// The pointee type of a pointer, if this is one.
    pub fn pointee(&self) -> Option<&CType> {
        match self {
            CType::Pointer(inner) => Some(inner),
            _ => None,
        }
    }

    /// The element type of an array, if this is one.
    pub fn array_elem(&self) -> Option<&CType> {
        match self {
            CType::Array(inner, _) => Some(inner),
            _ => None,
        }
    }

    /// If this is an array or function type, the pointer type it decays to (an
    /// array decays to a pointer-to-element; a function to a pointer-to-function).
    pub fn decayed(&self) -> Option<CType> {
        match self {
            CType::Array(elem, _) => Some(CType::ptr_to((**elem).clone())),
            CType::Func(_) => Some(CType::ptr_to(self.clone())),
            _ => None,
        }
    }

    /// The bit width of an integer or `_Bool` type (`_Bool` counts as 8-bit
    /// storage), or `None` for `void`/pointers.
    pub fn int_width(&self) -> Option<u16> {
        match self {
            CType::Bool => Some(8),
            CType::Int(i) => Some(i.width),
            _ => None,
        }
    }

    /// Whether an integer or `_Bool` type is signed. Pointers/void return
    /// `false`.
    pub fn is_signed(&self) -> bool {
        matches!(self, CType::Int(i) if i.signed)
    }

    /// The integer conversion rank used by the usual arithmetic conversions.
    /// For this subset the rank is simply the width, with `_Bool` lowest.
    pub fn rank(&self) -> u16 {
        match self {
            CType::Bool => 1,
            CType::Int(i) => i.width,
            _ => 0,
        }
    }
}

impl fmt::Display for CType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CType::Void => write!(f, "void"),
            CType::Bool => write!(f, "_Bool"),
            CType::Int(i) => {
                let base = match (i.width, i.signed) {
                    (8, true) => "signed char",
                    (8, false) => "unsigned char",
                    (16, true) => "short",
                    (16, false) => "unsigned short",
                    (32, true) => "int",
                    (32, false) => "unsigned int",
                    (64, true) => "long",
                    (64, false) => "unsigned long",
                    _ => "int",
                };
                write!(f, "{base}")
            }
            CType::Float(FloatTy::F32) => write!(f, "float"),
            CType::Float(FloatTy::F64) => write!(f, "double"),
            CType::Pointer(inner) => write!(f, "{inner} *"),
            CType::Array(elem, n) => write!(f, "{elem}[{n}]"),
            CType::Record(_) => write!(f, "struct/union"),
            CType::Func(ft) => {
                write!(f, "{} (", ft.ret)?;
                for (i, p) in ft.params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{p}")?;
                }
                if ft.variadic {
                    write!(f, ", ...")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// A binary operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinaryOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Rem,
    /// `&`
    BitAnd,
    /// `|`
    BitOr,
    /// `^`
    BitXor,
    /// `<<`
    Shl,
    /// `>>`
    Shr,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `&&`
    LAnd,
    /// `||`
    LOr,
}

/// A prefix unary operator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnaryOp {
    /// `-e`
    Neg,
    /// `+e`
    Plus,
    /// `!e`
    LNot,
    /// `~e`
    BitNot,
    /// `*e` (dereference)
    Deref,
    /// `&e` (address-of)
    AddrOf,
}

/// An expression node: a [`ExprKind`] plus its source span.
#[derive(Clone, Debug)]
pub struct Expr {
    /// The expression variant.
    pub kind: ExprKind,
    /// The source span covering the expression.
    pub span: Span,
}

/// The variants of a C expression.
#[derive(Clone, Debug)]
pub enum ExprKind {
    /// An integer (or character) constant with its C type derived from the
    /// literal's suffix/value.
    IntLit(i128, CType),
    /// A floating-point constant (its exact value as `f64`, already rounded to
    /// the target precision) with its C type (`float`/`double`).
    FloatLit(f64, CType),
    /// An identifier reference.
    Ident(String),
    /// A prefix unary operation.
    Unary(UnaryOp, Box<Expr>),
    /// A binary operation.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// An assignment: plain (`None`) or compound (`Some(op)`), `lhs op= rhs`.
    Assign(Option<BinaryOp>, Box<Expr>, Box<Expr>),
    /// A function call: callee and argument list.
    Call(Box<Expr>, Vec<Expr>),
    /// A cast `(T)e`.
    Cast(CType, Box<Expr>),
    /// A conditional `c ? t : f`.
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
    /// A comma expression `a, b`.
    Comma(Box<Expr>, Box<Expr>),
    /// Prefix `++e`.
    PreInc(Box<Expr>),
    /// Prefix `--e`.
    PreDec(Box<Expr>),
    /// Postfix `e++`.
    PostInc(Box<Expr>),
    /// Postfix `e--`.
    PostDec(Box<Expr>),
    /// `sizeof e`.
    SizeofExpr(Box<Expr>),
    /// `sizeof(T)`.
    SizeofType(CType),
    /// A string literal's decoded bytes (already NUL-terminated is *not*
    /// assumed; sema appends the terminator).
    StrLit(Vec<u8>),
    /// Array subscript `base[index]`.
    Index(Box<Expr>, Box<Expr>),
    /// Member access: `base.name` (`arrow == false`) or `base->name`
    /// (`arrow == true`).
    Member(Box<Expr>, String, bool),
    /// `_Alignof(type-name)` / `alignof(type-name)` (C11/C23): a `size_t`
    /// constant equal to the type's alignment.
    AlignofType(CType),
    /// `_Generic(controlling, type1: e1, ..., default: ed)` (C11): generic
    /// selection. The controlling expression is typed but not evaluated; the
    /// association whose type matches supplies the result.
    Generic(Box<Expr>, Vec<GenericAssoc>),
    /// A compound literal `(type-name){ initializer-list }` (C99): an unnamed
    /// object of the given type with the given initializer, usable as an lvalue.
    CompoundLiteral(CType, Box<Init>),
    /// `__builtin_va_start(ap, last)` — initialize the `va_list` `ap` (the second
    /// operand, the last named parameter, is only used to validate the call and is
    /// otherwise unused: the argument counts come from the enclosing function).
    VaStart(Box<Expr>, Box<Expr>),
    /// `__builtin_va_arg(ap, type)` — fetch the next variadic argument of `type`.
    VaArg(Box<Expr>, CType),
    /// `__builtin_va_end(ap)` — finish traversing `ap` (a no-op on this target).
    VaEnd(Box<Expr>),
    /// `__builtin_va_copy(dst, src)` — copy the traversal state of `src` to `dst`.
    VaCopy(Box<Expr>, Box<Expr>),
}

/// One association of a `_Generic` selection: a type (`None` for `default`) and
/// the expression selected when the controlling type matches it.
#[derive(Clone, Debug)]
pub struct GenericAssoc {
    /// The association type, or `None` for the `default` association.
    pub ty: Option<CType>,
    /// The result expression for this association.
    pub expr: Expr,
}

/// A C initializer: either a single expression or a brace-enclosed list whose
/// items may carry designators (`.field` / `[index]`).
#[derive(Clone, Debug)]
pub enum Init {
    /// A scalar (or full-aggregate string) initializer expression.
    Expr(Expr),
    /// A brace-enclosed initializer list.
    List(Vec<InitItem>),
}

/// One entry of a brace initializer list: an optional designator chain and the
/// initializer applied at that position.
#[derive(Clone, Debug)]
pub struct InitItem {
    /// The designator chain (`.field` / `[index]`), empty for positional items.
    pub designators: Vec<Designator>,
    /// The initializer value at this position.
    pub init: Init,
}

/// A single designator in a designated initializer.
#[derive(Clone, Debug)]
pub enum Designator {
    /// `.field`.
    Field(String),
    /// `[index]` (a constant expression).
    Index(i128),
}

/// A statement node: a [`StmtKind`] plus its source span.
#[derive(Clone, Debug)]
pub struct Stmt {
    /// The statement variant.
    pub kind: StmtKind,
    /// The source span covering the statement.
    pub span: Span,
}

/// The variants of a C statement.
#[derive(Clone, Debug)]
pub enum StmtKind {
    /// An expression statement, or the empty statement (`None`).
    Expr(Option<Expr>),
    /// One or more local variable declarations sharing a base type.
    Decl(Vec<VarDecl>),
    /// A brace-delimited block introducing a new scope.
    Block(Vec<Stmt>),
    /// `if (cond) then [else els]`.
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    /// `while (cond) body`.
    While(Expr, Box<Stmt>),
    /// `do body while (cond);`.
    DoWhile(Box<Stmt>, Expr),
    /// `for (init; cond; step) body`.
    For(Option<Box<Stmt>>, Option<Expr>, Option<Expr>, Box<Stmt>),
    /// `return [e];`.
    Return(Option<Expr>),
    /// `break;`.
    Break,
    /// `continue;`.
    Continue,
    /// `switch (expr) body`.
    Switch(Expr, Box<Stmt>),
    /// `case const-expr: stmt` — the constant is folded at parse time.
    Case(i128, Box<Stmt>),
    /// `default: stmt`.
    Default(Box<Stmt>),
    /// `label: stmt` — a named label (function-scoped, in its own namespace).
    Label(String, Box<Stmt>),
    /// `goto label;`.
    Goto(String),
}

/// A single declared variable (in a local declaration or a global).
#[derive(Clone, Debug)]
pub struct VarDecl {
    /// The declared name.
    pub name: String,
    /// The declared type.
    pub ty: CType,
    /// The initializer, if any.
    pub init: Option<Init>,
    /// An explicit `_Alignas`/`alignas` alignment override, if any.
    pub align: Option<u64>,
    /// The source span of the declarator.
    pub span: Span,
}

/// A function parameter.
#[derive(Clone, Debug)]
pub struct Param {
    /// The parameter name, if named.
    pub name: Option<String>,
    /// The parameter type.
    pub ty: CType,
    /// The source span of the parameter.
    pub span: Span,
}

/// A function definition (a prototype with a body).
#[derive(Clone, Debug)]
pub struct FuncDef {
    /// The function name.
    pub name: String,
    /// The return type.
    pub ret: CType,
    /// The parameter list.
    pub params: Vec<Param>,
    /// Whether the function is variadic (`...` after its named parameters).
    pub variadic: bool,
    /// The function body (a list of statements).
    pub body: Vec<Stmt>,
    /// The source span of the function's declarator (its name).
    pub span: Span,
}

/// A function prototype / declaration (no body).
#[derive(Clone, Debug)]
pub struct FuncProto {
    /// The function name.
    pub name: String,
    /// The return type.
    pub ret: CType,
    /// The parameter types.
    pub params: Vec<Param>,
    /// Whether the prototype is variadic (`...`).
    pub variadic: bool,
    /// The source span of the declarator.
    pub span: Span,
}

/// A top-level declaration in a translation unit.
#[derive(Clone, Debug)]
pub enum TopLevel {
    /// A function definition.
    Func(FuncDef),
    /// A function prototype.
    Proto(FuncProto),
    /// A global variable declaration.
    Global(VarDecl),
}

/// A whole translation unit: the ordered top-level declarations of one file,
/// plus the shared `struct`/`union` registry and the enum-constant table the
/// parser accumulated.
#[derive(Clone, Debug, Default)]
pub struct TranslationUnit {
    /// The top-level items in source order.
    pub items: Vec<TopLevel>,
    /// Every `struct`/`union` definition, referenced by [`CType::Record`].
    pub records: Records,
    /// Enumerator constants (`name`, value), in declaration order.
    pub enum_consts: Vec<(String, i128)>,
}
