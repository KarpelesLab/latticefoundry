//! The C abstract syntax tree and the C type model.
//!
//! The parser ([`crate::parse`]) produces these untyped nodes directly from the
//! token stream; [`crate::sema`] then type-checks them and produces its own
//! typed tree. Every node carries a source [`Span`] so diagnostics can point at
//! the offending construct.

use std::fmt;

use latticefoundry::support::diagnostics::Span;

/// A C scalar type in the freestanding subset: `void`, `_Bool`, the integer
/// types (tracked as an explicit width plus signedness), and pointers.
///
/// Aggregates (structs, arrays) and floating-point are out of the v1 subset.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CType {
    /// `void`.
    Void,
    /// `_Bool`.
    Bool,
    /// An integer type of a given width in bits and signedness.
    Int(IntTy),
    /// `T *` — a pointer to `T`.
    Pointer(Box<CType>),
}

/// The (width, signedness) of a C integer type. Width is in bits: 8 (`char`),
/// 16 (`short`), 32 (`int`), 64 (`long`/`long long`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IntTy {
    /// Width in bits.
    pub width: u16,
    /// Whether the type is signed.
    pub signed: bool,
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

    /// A pointer to `pointee`.
    pub fn ptr_to(pointee: CType) -> CType {
        CType::Pointer(Box::new(pointee))
    }

    /// Whether this is any integer type (including `_Bool`).
    pub fn is_integer(&self) -> bool {
        matches!(self, CType::Int(_) | CType::Bool)
    }

    /// Whether this is a pointer type.
    pub fn is_pointer(&self) -> bool {
        matches!(self, CType::Pointer(_))
    }

    /// Whether this is a scalar type usable in a condition (integer or pointer).
    pub fn is_scalar(&self) -> bool {
        self.is_integer() || self.is_pointer()
    }

    /// The pointee type of a pointer, if this is one.
    pub fn pointee(&self) -> Option<&CType> {
        match self {
            CType::Pointer(inner) => Some(inner),
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
            CType::Pointer(inner) => write!(f, "{inner} *"),
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
}

/// A single declared variable (in a local declaration or a global).
#[derive(Clone, Debug)]
pub struct VarDecl {
    /// The declared name.
    pub name: String,
    /// The declared type.
    pub ty: CType,
    /// The initializer expression, if any.
    pub init: Option<Expr>,
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

/// A whole translation unit: the ordered top-level declarations of one file.
#[derive(Clone, Debug, Default)]
pub struct TranslationUnit {
    /// The top-level items in source order.
    pub items: Vec<TopLevel>,
}
