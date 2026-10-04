//! A recursive-descent parser for the freestanding C subset.
//!
//! Written directly from the C grammar (clean room, tenet T1): declaration
//! specifiers and declarators, statements, and a precedence-climbing expression
//! parser that implements C's operator precedence and associativity. It consumes
//! the [`crate::lex`] token stream and produces the untyped [`crate::ast`] tree.
//! Errors are reported as [`Diagnostic`]s with the offending token's span; the
//! parser bails on the first error.

use std::collections::HashMap;

use latticefoundry::support::diagnostics::{Diagnostic, Span};

use crate::ast::{
    AsmOperand, AsmStmt, BinaryOp, CType, Designator, Expr, ExprKind, Field, FloatTy, FuncDef, FuncProto, FuncType,
    GenericAssoc, Init, InitItem, IntTy, Param, Quals, RecordDef, RecordId, RecordKind, Records, Stmt,
    StmtKind, Storage, StrKind, SymAttrs, TopLevel, TranslationUnit, UnaryOp, VarDecl,
};
use crate::consteval::{self, CInt, ConstEnv, reduce_const_to_type};
use crate::cstd::CStd;
use crate::layout;
use crate::lex::{Keyword, Punct, Token, TokenKind};
use latticefoundry::ir::Visibility;

type PResult<T> = Result<T, Diagnostic>;

/// Parse a token stream into a [`TranslationUnit`], gating language features by
/// the selected `std`.
pub fn parse(tokens: Vec<Token>, std: CStd) -> Result<TranslationUnit, Vec<Diagnostic>> {
    let mut parser = Parser {
        tokens,
        pos: 0,
        std,
        records: Records::default(),
        tags: HashMap::new(),
        enum_map: HashMap::new(),
        enum_tag_signed: HashMap::new(),
        enum_consts: Vec::new(),
        constexprs: Vec::new(),
        scopes: vec![HashMap::new()],
        last_alignas: None,
        last_constexpr: false,
        spec_attrs: Attrs::default(),
        decl_attrs: Attrs::default(),
        pending_attrs: Attrs::default(),
        spec_inline: false,
        spec_thread: false,
        spec_quals: Quals::NONE,
        va_list_record: None,
        cur_func: None,
        in_params: 0,
        vla_ok: false,
        vla_len: None,
        transparent_params: Vec::new(),
        extension: 0,
        named_fn_params: None,
    };
    match parser.parse_unit() {
        Ok(items) => Ok(TranslationUnit {
            items,
            records: parser.records,
            enum_consts: parser.enum_consts,
            constexprs: parser.constexprs,
        }),
        Err(d) => Err(vec![d]),
    }
}

/// How a name is bound in a parser scope, for the typedef-name disambiguation.
#[derive(Clone, Debug)]
enum NameKind {
    /// A `typedef` name resolving to a type.
    Typedef(CType),
    /// An ordinary identifier (variable/parameter/function), which shadows any
    /// outer `typedef` of the same name. Carries the object's type when known,
    /// so `typeof(name)` can resolve it.
    Ordinary(Option<CType>),
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    std: CStd,
    /// The `struct`/`union` registry being populated.
    records: Records,
    /// Tag name → record id (a single translation-unit-wide tag namespace).
    tags: HashMap<String, RecordId>,
    /// Enumerator name → value, for constant-expression evaluation.
    enum_map: HashMap<String, i128>,
    /// Enum tag → whether its underlying type is signed (`int`, i.e. it has a
    /// negative enumerator) vs. unsigned (`unsigned int`). Lets a later tag-only
    /// reference (`enum foo x : 2;`) recover the right signedness — important for
    /// `enum`-typed bit-fields.
    enum_tag_signed: HashMap<String, bool>,
    /// Enumerator constants in declaration order, handed to sema.
    enum_consts: Vec<(String, i128)>,
    /// C23 `constexpr` objects (`name`, reduced value, declared type), handed to
    /// sema as named compile-time constants.
    constexprs: Vec<(String, i128, CType)>,
    /// Scoped name bindings for typedef-name disambiguation.
    scopes: Vec<HashMap<String, NameKind>>,
    /// The pending `_Alignas`/`alignas` alignment from the declaration specifiers
    /// currently being parsed (consumed by the declarators they apply to).
    last_alignas: Option<u64>,
    /// Whether the declaration specifiers just parsed included C23 `constexpr`.
    last_constexpr: bool,
    /// The GNU attributes found among the declaration specifiers just parsed
    /// (they apply to every declarator of the declaration).
    spec_attrs: Attrs,
    /// The GNU attributes trailing the declarator just parsed (after its
    /// suffixes and around an asm label), applying to that declarator only.
    decl_attrs: Attrs,
    /// Attributes met among leading storage-class specifiers, not yet folded
    /// into a declaration-specifier sequence.
    pending_attrs: Attrs,
    /// Whether the current declaration's specifiers included `inline`. Reset at
    /// the start of each top-level/block declaration.
    spec_inline: bool,
    /// Whether the current declaration's specifiers included `_Thread_local` /
    /// `__thread`. Reset like `spec_inline`.
    spec_thread: bool,
    /// The `volatile`/`_Atomic` qualifiers among the declaration specifiers
    /// being parsed (saved and restored around nested specifier sequences).
    spec_quals: Quals,
    /// The record modelling the System V `__va_list_tag` behind the builtin
    /// `__builtin_va_list` type, created on first use.
    va_list_record: Option<RecordId>,
    /// The name of the function whose body is being parsed (for `__func__`).
    cur_func: Option<String>,
    /// Nesting depth of parameter lists being parsed (a parameter array bound
    /// need not be constant).
    in_params: u32,
    /// Set while parsing the declarator of a block-scope object, whose
    /// outermost array bound may be a run-time value (a variable-length
    /// array); that bound is parked in [`vla_len`](Self::vla_len).
    vla_ok: bool,
    /// The run-time bound of the variable-length array declarator just parsed.
    vla_len: Option<Expr>,
    /// The transparent-union parameters (index, union type) of the parameter
    /// list parsed last.
    transparent_params: Vec<(usize, CType)>,
    /// Nonzero while parsing a declaration or expression marked with GNU
    /// `__extension__`, which lifts the pedantic C-standard gates.
    extension: u32,
    /// The named parameters of the function declarator parsed last (for a
    /// definition through a grouped declarator: `void (*f(int x))(void) {`).
    named_fn_params: Option<Vec<Param>>,
}

/// The GNU `__attribute__((...))` properties lf-cc gives meaning to. Every other
/// attribute (`nonnull`, `format`, `nothrow`, `leaf`, `pure`, `malloc`, ...) is
/// parsed and ignored, which is always permitted.
#[derive(Clone, Debug, Default)]
struct Attrs {
    /// `mode(QI|HI|SI|DI|TI|word|pointer|byte)`: the integer width in bits.
    mode: Option<u16>,
    /// `aligned` / `aligned(N)`: the requested minimum alignment.
    aligned: Option<u64>,
    /// `packed` on a record type.
    packed: bool,
    /// `gnu_inline` on a function (GNU89 `extern inline` semantics).
    gnu_inline: bool,
    /// `transparent_union` on a union type.
    transparent_union: bool,
    /// `vector_size(N)`: the type is a GCC vector of `N` bytes.
    vector_size: Option<u64>,
    /// `visibility("default"|"hidden"|"protected"|"internal")`.
    visibility: Option<Visibility>,
    /// `weak`.
    weak: bool,
}

/// The declaration-wide properties shared by every declarator of one
/// file-scope declaration.
struct DeclCtx {
    /// The attributes among the declaration specifiers.
    sattrs: Attrs,
    /// An `_Alignas` among the declaration specifiers.
    align: Option<u64>,
    /// The linkage-affecting storage class.
    storage: Storage,
}

/// A parsed function declarator at file scope: a definition, or a prototype.
enum FnDecl {
    Def(FuncDef),
    Proto(FuncProto),
    /// A GNU `extern inline` definition, reduced to its prototype (the body
    /// has been consumed, so no `;` follows).
    InlineOnly(FuncProto),
}

/// The stricter of two optional alignment requests.
fn max_align(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

impl Attrs {
    /// Fold the attributes of `other` into `self` (later wins; alignment keeps
    /// the strictest request).
    fn merge(&mut self, other: Attrs) {
        if other.mode.is_some() {
            self.mode = other.mode;
        }
        if let Some(a) = other.aligned {
            self.aligned = Some(self.aligned.map_or(a, |b| b.max(a)));
        }
        self.packed |= other.packed;
        self.gnu_inline |= other.gnu_inline;
        self.transparent_union |= other.transparent_union;
        if other.vector_size.is_some() {
            self.vector_size = other.vector_size;
        }
        if other.visibility.is_some() {
            self.visibility = other.visibility;
        }
        self.weak |= other.weak;
    }

    /// The symbol attributes (visibility, weak) among these attributes.
    fn sym(&self) -> SymAttrs {
        SymAttrs { visibility: self.visibility, weak: self.weak, gnu_inline: self.gnu_inline }
    }
}

/// The parser's integer constant expressions: enumerators and `constexpr`
/// objects by name, `sizeof` from the parser's symbol table, and the classic
/// `offsetof` address arithmetic.
impl ConstEnv for Parser {
    fn ident(&self, name: &str) -> Option<CInt> {
        let v = *self.enum_map.get(name)?;
        Some(match self.constexprs.iter().rev().find(|(n, ..)| n == name) {
            Some((_, _, ty)) => CInt::new(v, ty),
            None => CInt::natural(v),
        })
    }

    fn size_of_type(&self, ty: &CType) -> u64 {
        layout::size_of(&self.records, ty)
    }

    fn align_of_type(&self, ty: &CType) -> u64 {
        layout::align_of(&self.records, ty)
    }

    fn size_of_expr(&self, e: &Expr) -> Option<u64> {
        Some(layout::size_of(&self.records, &self.expr_type(e)?))
    }

    fn other(&self, e: &Expr) -> Option<CInt> {
        match &e.kind {
            // The classic `offsetof`: `(size_t) &((T *) 0)->m` — the address
            // of a member reached from a constant pointer is a constant.
            ExprKind::Unary(UnaryOp::AddrOf, inner) => {
                Some(CInt::new(self.const_lvalue_addr(inner)?, &CType::Int(IntTy::new(64, false))))
            }
            _ => None,
        }
    }
}

impl Parser {
    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn peek_span(&self) -> Span {
        self.tokens[self.pos].span
    }

    fn peek_at(&self, ahead: usize) -> &TokenKind {
        let idx = (self.pos + ahead).min(self.tokens.len() - 1);
        &self.tokens[idx].kind
    }

    fn bump(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek(), TokenKind::Eof)
    }

    fn is_punct(&self, p: Punct) -> bool {
        matches!(self.peek(), TokenKind::Punct(x) if *x == p)
    }

    fn is_kw(&self, k: Keyword) -> bool {
        matches!(self.peek(), TokenKind::Keyword(x) if *x == k)
    }

    fn eat_punct(&mut self, p: Punct) -> bool {
        if self.is_punct(p) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn eat_kw(&mut self, k: Keyword) -> bool {
        if self.is_kw(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn err<T>(&self, msg: impl Into<String>) -> PResult<T> {
        Err(Diagnostic::error(msg).with_span(self.peek_span()))
    }

    fn expect_punct(&mut self, p: Punct, what: &str) -> PResult<Span> {
        if self.is_punct(p) {
            Ok(self.bump().span)
        } else {
            self.err(format!("expected {what}"))
        }
    }

    fn expect_ident(&mut self) -> PResult<(String, Span)> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let span = self.bump().span;
                Ok((name, span))
            }
            _ => self.err("expected identifier"),
        }
    }

    // --- scopes & typedef names --------------------------------------------

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare_ordinary(&mut self, name: &str, ty: Option<CType>) {
        self.scopes.last_mut().unwrap().insert(name.to_owned(), NameKind::Ordinary(ty));
    }

    fn declare_typedef(&mut self, name: &str, ty: CType) {
        self.scopes.last_mut().unwrap().insert(name.to_owned(), NameKind::Typedef(ty));
    }

    /// The type a name resolves to if the innermost binding is a `typedef`.
    fn typedef_type(&self, name: &str) -> Option<CType> {
        for scope in self.scopes.iter().rev() {
            match scope.get(name) {
                Some(NameKind::Typedef(ty)) => return Some(ty.clone()),
                Some(NameKind::Ordinary(_)) => return None,
                None => {}
            }
        }
        None
    }

    /// The declared type of an ordinary identifier, for `typeof`.
    fn var_type(&self, name: &str) -> Option<CType> {
        for scope in self.scopes.iter().rev() {
            match scope.get(name) {
                Some(NameKind::Ordinary(ty)) => return ty.clone(),
                Some(NameKind::Typedef(_)) => return None,
                None => {}
            }
        }
        None
    }

    fn is_typedef_name(&self, name: &str) -> bool {
        self.typedef_type(name).is_some()
    }

    // --- records (struct/union) & enums ------------------------------------

    fn tag_record(&mut self, tag: &str, kind: RecordKind) -> RecordId {
        if let Some(&id) = self.tags.get(tag) {
            return id;
        }
        let id = self.records.defs.len();
        self.records.defs.push(RecordDef {
            kind,
            tag: Some(tag.to_owned()),
            fields: Vec::new(),
            complete: false,
            packed: false,
            align: None,
            transparent: false,
        });
        self.tags.insert(tag.to_owned(), id);
        id
    }

    fn anon_record(&mut self, kind: RecordKind) -> RecordId {
        let id = self.records.defs.len();
        self.records.defs.push(RecordDef {
            kind,
            tag: None,
            fields: Vec::new(),
            complete: false,
            packed: false,
            align: None,
            transparent: false,
        });
        id
    }

    /// The builtin `__builtin_va_list` type: the System V AMD64 `va_list`,
    /// `struct { unsigned gp_offset, fp_offset; void *overflow_arg_area,
    /// *reg_save_area; }[1]`. It is layout-identical to the `va_list` of
    /// lf-cc's own `<stdarg.h>` and to gcc's, so a `va_list` crosses into libc
    /// (`vprintf`, `vsnprintf`, ...) unchanged; like any array parameter it is
    /// passed as a pointer to its single element.
    fn builtin_va_list(&mut self) -> CType {
        let id = match self.va_list_record {
            Some(id) => id,
            None => {
                let id = self.anon_record(RecordKind::Struct);
                let field = |name: &str, ty: CType| Field {
                    name: name.to_owned(),
                    ty,
                    anonymous: false,
                    align: None,
                    bit_width: None,
                };
                let vp = CType::ptr_to(CType::Void);
                self.records.defs[id].fields = vec![
                    field("gp_offset", CType::uint()),
                    field("fp_offset", CType::uint()),
                    field("overflow_arg_area", vp.clone()),
                    field("reg_save_area", vp),
                ];
                self.records.defs[id].complete = true;
                self.va_list_record = Some(id);
                id
            }
        };
        CType::Array(Box::new(CType::Record(id)), 1)
    }

    // --- top level ---------------------------------------------------------

    fn parse_unit(&mut self) -> PResult<Vec<TopLevel>> {
        let mut items = Vec::new();
        while !self.at_eof() {
            // An attribute specifier sequence may precede a top-level declaration
            // (`__attribute__((visibility("hidden"))) int f(void);`); it applies
            // to the declaration's specifiers.
            self.pending_attrs = Attrs::default();
            let leading = self.parse_attributes()?;
            self.pending_attrs.merge(leading);
            if self.at_eof() {
                break;
            }
            // A file-scope `_Static_assert` is a declaration with no external
            // effect in this subset; consume and drop it.
            if self.is_kw(Keyword::StaticAssert) {
                self.parse_static_assert()?;
                continue;
            }
            // A file-scope `asm("...");` (GNU): basic asm text the translation
            // unit contributes verbatim to its assembly output.
            if self.is_kw(Keyword::Asm) {
                let span = self.peek_span();
                let stmt = self.parse_asm_stmt()?;
                self.expect_punct(Punct::Semi, "';' after file-scope asm")?;
                if stmt.extended || stmt.is_goto {
                    return Err(Diagnostic::error(
                        "a file-scope asm declaration takes no operands (basic asm only)",
                    )
                    .with_span(span));
                }
                items.push(TopLevel::Asm(stmt.template));
                continue;
            }
            // `#pragma weak name`, as the preprocessor passes it on.
            if matches!(self.peek_at(0), TokenKind::Ident(n) if n == "__pragma_weak") {
                self.bump();
                let (name, _) = self.expect_ident()?;
                self.expect_punct(Punct::Semi, "';'")?;
                items.push(TopLevel::PragmaWeak(name));
                continue;
            }
            // `__extension__ typedef long long ...;` / `__extension__ extern ...`.
            let ext = self.skip_extension();
            self.extension += u32::from(ext);
            let r = self.parse_top_level();
            self.extension -= u32::from(ext);
            items.extend(r?);
            self.pending_attrs = Attrs::default();
        }
        Ok(items)
    }

    /// Whether the cursor is at GNU `__extension__`.
    fn at_extension(&self) -> bool {
        matches!(self.peek(), TokenKind::Ident(n) if n == "__extension__")
    }

    /// Consume any `__extension__` markers at the cursor; true if there were any.
    fn skip_extension(&mut self) -> bool {
        let mut any = false;
        while self.at_extension() {
            self.bump();
            any = true;
        }
        any
    }

    /// Whether the cursor is at `__extension__` (possibly repeated) introducing
    /// a declaration rather than an expression.
    fn extension_decl_ahead(&self) -> bool {
        let mut k = 0;
        while matches!(self.peek_at(k), TokenKind::Ident(n) if n == "__extension__") {
            k += 1;
        }
        k > 0
            && match self.peek_at(k) {
                TokenKind::Keyword(kw) => self.keyword_starts_decl(*kw),
                TokenKind::Ident(n) => {
                    self.ident_starts_decl(n)
                        && !matches!(self.peek_at(k + 1), TokenKind::Punct(Punct::Colon))
                }
                _ => false,
            }
    }

    /// Parse a `_Static_assert ( const-expr [, "msg"] ) ;` (C11) or
    /// `static_assert ( const-expr [, "msg"] ) ;` (C23, message optional)
    /// declaration: evaluate the integer constant expression and, if it is zero,
    /// report a diagnostic that includes the message. A true assertion produces
    /// no code.
    fn parse_static_assert(&mut self) -> PResult<()> {
        self.bump(); // _Static_assert / static_assert
        self.expect_punct(Punct::LParen, "'(' after _Static_assert")?;
        let cond_span = self.peek_span();
        let cond = self.parse_conditional()?;
        let value = self.eval_const_expr(&cond).ok_or_else(|| {
            Diagnostic::error("static assertion expression is not an integer constant expression")
                .with_span(cond_span)
        })?;
        let mut message: Option<String> = None;
        if self.eat_punct(Punct::Comma) {
            // The message may be several adjacent literals (`"verify (" #R ")"`).
            let mut text = Vec::new();
            while let TokenKind::Str(s, _) = self.peek().clone() {
                self.bump();
                text.extend_from_slice(&s);
                message = Some(String::new());
            }
            match message {
                Some(_) => message = Some(String::from_utf8_lossy(&text).into_owned()),
                None => return self.err("expected a string message in _Static_assert"),
            }
        }
        self.expect_punct(Punct::RParen, "')' to close _Static_assert")?;
        self.expect_punct(Punct::Semi, "';' after _Static_assert")?;
        if value == 0 {
            let msg = match message {
                Some(m) => format!("static assertion failed: {m}"),
                None => "static assertion failed".to_owned(),
            };
            return Err(Diagnostic::error(msg).with_span(cond_span));
        }
        Ok(())
    }

    fn parse_top_level(&mut self) -> PResult<Vec<TopLevel>> {
        // A stray `;` at file scope is an empty declaration (GNU accepts it).
        if self.eat_punct(Punct::Semi) {
            return Ok(Vec::new());
        }
        self.spec_inline = false;
        self.spec_thread = false;
        if self.eat_kw(Keyword::Typedef) {
            return self.parse_typedef();
        }
        // Storage-class specifiers (extern/static) determine a file-scope object's
        // linkage; they may appear before and/or after the type specifiers.
        let storage = self.consume_storage()?;
        if self.eat_kw(Keyword::Typedef) {
            return self.parse_typedef();
        }
        let base = self.parse_decl_specs_impl(true)?;
        // Capture the specifier attributes now: parsing a parameter list below
        // re-enters the declaration-specifier parser.
        let sattrs = self.spec_attrs.clone();
        let align = self.last_alignas.take();
        let is_constexpr = self.last_constexpr;
        let storage = merge_storage(storage, self.consume_storage()?);

        // A C23 `constexpr` object is a named compile-time constant (no storage).
        if is_constexpr {
            self.parse_constexpr_decls(base)?;
            return Ok(Vec::new());
        }

        // A bare `struct S { ... };` / `enum E { ... };` declares only a type.
        if self.eat_punct(Punct::Semi) {
            return Ok(Vec::new());
        }

        let decl = DeclCtx { sattrs, align, storage };
        let mut items = Vec::new();
        // First declarator: pointers, then a name.
        let ty0 = self.parse_pointers(base.clone())?;

        if self.is_punct(Punct::LParen) {
            // A grouped declarator: an object (`ret (*name)(...)`, `ret
            // (*name[N])(...)`), a prototype (`ret (*f(args))(args);`), or the
            // definition of a function returning a function pointer (`void
            // (*f(int x))(void) { ... }`, SQLite's xDlSym methods).
            self.named_fn_params = None;
            let (name, ty, span) = self.parse_named_declarator(ty0)?;
            let (asm_label, ty, attrs) = self.finish_declarator(ty, span, &decl.sattrs)?;
            if let CType::Func(ft) = &ty
                && self.is_punct(Punct::LBrace)
            {
                let params = match self.named_fn_params.take() {
                    Some(ps) => ps,
                    None => ft.params.iter().map(|t| Param { name: None, ty: t.clone(), span }).collect(),
                };
                self.declare_ordinary(&name, Some(ty.clone()));
                self.cur_func = Some(name.clone());
                self.push_scope();
                for p in &params {
                    if let Some(n) = &p.name {
                        self.declare_ordinary(n, Some(p.ty.clone()));
                    }
                }
                let body = self.parse_block_stmts();
                self.pop_scope();
                return Ok(vec![TopLevel::Func(FuncDef {
                    name,
                    ret: ft.ret.clone(),
                    params,
                    variadic: ft.variadic,
                    is_static: decl.storage == Storage::Static,
                    is_inline: self.spec_inline,
                    is_extern: decl.storage == Storage::Extern,
                    body: body?,
                    asm_label,
                    attrs: attrs.sym(),
                    span,
                })]);
            }
            items.push(self.file_scope_finished_declarator(name, ty, span, &decl, asm_label, &attrs)?);
        } else {
            let (name, name_span) = self.expect_ident()?;
            self.declare_ordinary(&name, None);
            if self.is_punct(Punct::LParen) {
                if self.spec_thread {
                    return Err(Diagnostic::error(format!("function '{name}' declared thread-local"))
                        .with_span(name_span));
                }
                match self.parse_function_declarator(name, name_span, ty0, &decl)? {
                    FnDecl::Def(f) => return Ok(vec![TopLevel::Func(f)]),
                    FnDecl::InlineOnly(p) => return Ok(vec![TopLevel::Proto(p)]),
                    FnDecl::Proto(p) => items.push(TopLevel::Proto(p)),
                }
            } else {
                let ty = self.parse_array_suffix(ty0)?;
                items.push(self.file_scope_declarator(name, ty, name_span, &decl)?);
            }
        }
        while self.eat_punct(Punct::Comma) {
            let (name, ty, span) = self.parse_named_declarator(base.clone())?;
            items.push(self.file_scope_declarator(name, ty, span, &decl)?);
        }
        self.expect_punct(Punct::Semi, "';' after global declaration")?;
        Ok(items)
    }

    /// Parse the parameter list (the `(` at the cursor) of a function declarator
    /// named `name` returning `ret`, then its trailing extensions, and either a
    /// body (a definition) or nothing (a prototype; the caller consumes `;`/`,`).
    ///
    /// An `extern inline` definition under GNU inline semantics (the
    /// `gnu_inline` attribute, as glibc's `__extern_inline` spells it, or the
    /// `gnu89` dialect) provides an inline body only: no out-of-line definition
    /// is emitted, calls bind to the external symbol, so it is returned as a
    /// prototype (exactly what gcc does when it does not inline).
    fn parse_function_declarator(
        &mut self,
        name: String,
        name_span: Span,
        ret: CType,
        decl: &DeclCtx,
    ) -> PResult<FnDecl> {
        let is_static = decl.storage == Storage::Static;
        let is_inline = self.spec_inline;
        // Qualifiers on a return type are meaningless (the value is an rvalue).
        let ret = ret.unqual().clone();
        self.cur_func = Some(name.clone());
        self.push_scope();
        // An old-style (K&R) function definition opens with an *identifier
        // list*: `ret name(id, id, ...) decl-list { body }`. We recognize it
        // by the first token after `(` being an identifier that does not name
        // a type (a typedef-name would begin a prototype parameter instead).
        // Empty `()`, `(void)`, and prototype parameter lists are handled by
        // the normal path below.
        if self.at_kr_identifier_list() {
            let params = self.parse_kr_definition_params()?;
            for p in &params {
                if let Some(n) = &p.name {
                    self.declare_ordinary(n, Some(p.ty.clone()));
                }
            }
            let body = self.parse_block_stmts()?;
            self.pop_scope();
            return Ok(FnDecl::Def(FuncDef {
                name,
                ret,
                params,
                variadic: false,
                is_static,
                is_inline,
                is_extern: decl.storage == Storage::Extern,
                body,
                asm_label: None,
                attrs: decl.sattrs.sym(),
                span: name_span,
            }));
        }
        let (mut params, variadic) = self.parse_param_list()?;
        let prologue = self.rebuild_transparent_params(&mut params);
        for p in &params {
            if let Some(n) = &p.name {
                self.declare_ordinary(n, Some(p.ty.clone()));
            }
        }
        // A trailing declarator attribute — `f(void) __attribute__((noreturn));`
        // (GNU) or `[[noreturn]]` — sits between the parameter list and the
        // `;`/`{`. bzip2's NORETURN and glibc prototypes rely on this position.
        // So does a GNU asm label (`f(int) __asm__("sym")`, glibc's
        // `__REDIRECT`), mixed with attributes on either side of it.
        let (asm_label, ret, attrs) = self.finish_declarator(ret, name_span, &decl.sattrs)?;
        let fty = CType::Func(Box::new(FuncType {
            ret: ret.clone(),
            params: params.iter().map(|p| p.ty.clone()).collect(),
            variadic,
        }));
        if self.is_punct(Punct::LBrace) {
            let mut body = prologue;
            body.extend(self.parse_block_stmts()?);
            self.pop_scope();
            self.declare_ordinary(&name, Some(fty));
            let gnu_inline = attrs.gnu_inline || !self.std.is_c99();
            if is_inline && decl.storage == Storage::Extern && gnu_inline {
                return Ok(FnDecl::InlineOnly(FuncProto {
                    name,
                    ret,
                    params,
                    variadic,
                    is_static,
                    is_inline,
                    is_extern: true,
                    asm_label,
                    attrs: attrs.sym(),
                    span: name_span,
                }));
            }
            return Ok(FnDecl::Def(FuncDef {
                name,
                ret,
                params,
                variadic,
                is_static,
                is_inline,
                is_extern: decl.storage == Storage::Extern,
                body,
                asm_label,
                attrs: attrs.sym(),
                span: name_span,
            }));
        }
        self.pop_scope();
        self.declare_ordinary(&name, Some(fty));
        Ok(FnDecl::Proto(FuncProto {
            name,
            ret,
            params,
            variadic,
            is_static,
            is_inline,
            is_extern: decl.storage == Storage::Extern,
            asm_label,
            attrs: attrs.sym(),
            span: name_span,
        }))
    }

    /// Complete one file-scope declarator `name` of type `ty`: its trailing
    /// extensions and optional initializer. A declarator of function type (`int
    /// a, f(int);`, or a grouped `void (*signal(int, h))(int);`) declares a
    /// function prototype rather than an object.
    fn file_scope_declarator(
        &mut self,
        name: String,
        ty: CType,
        span: Span,
        decl: &DeclCtx,
    ) -> PResult<TopLevel> {
        let (asm_label, ty, attrs) = self.finish_declarator(ty, span, &decl.sattrs)?;
        self.file_scope_finished_declarator(name, ty, span, decl, asm_label, &attrs)
    }

    /// [`Self::file_scope_declarator`] once the trailing extensions are parsed.
    fn file_scope_finished_declarator(
        &mut self,
        name: String,
        ty: CType,
        span: Span,
        decl: &DeclCtx,
        asm_label: Option<String>,
        attrs: &Attrs,
    ) -> PResult<TopLevel> {
        self.declare_ordinary(&name, Some(ty.clone()));
        if let CType::Func(ft) = &ty {
            if self.spec_thread {
                return Err(Diagnostic::error(format!("function '{name}' declared thread-local")).with_span(span));
            }
            let params =
                ft.params.iter().map(|t| Param { name: None, ty: t.clone(), span }).collect();
            return Ok(TopLevel::Proto(FuncProto {
                name,
                ret: ft.ret.clone(),
                params,
                variadic: ft.variadic,
                is_static: decl.storage == Storage::Static,
                is_inline: self.spec_inline,
                is_extern: decl.storage == Storage::Extern,
                asm_label,
                attrs: attrs.sym(),
                span,
            }));
        }
        let init = if self.eat_punct(Punct::Assign) { Some(self.parse_initializer()?) } else { None };
        if let Some(i) = &init {
            self.declare_ordinary(&name, Some(self.deduce_array_symbol_type(&ty, i)));
        }
        let align = max_align(decl.align, attrs.aligned);
        Ok(TopLevel::Global(VarDecl {
            name,
            ty,
            init,
            align,
            storage: decl.storage,
            asm_label,
            thread_local: self.spec_thread,
            attrs: attrs.sym(),
            vla_len: None,
            span,
        }))
    }

    /// Parse the declarators of a `typedef` (the `typedef` keyword already
    /// consumed), registering each name as a typedef in the current scope.
    fn parse_typedef(&mut self) -> PResult<Vec<TopLevel>> {
        let base = self.parse_decl_specs()?;
        if self.spec_thread {
            return self.err("a typedef cannot be thread-local");
        }
        let sattrs = self.spec_attrs.clone();
        loop {
            let (name, ty, span) = self.parse_named_declarator(base.clone())?;
            let (_label, ty, attrs) = self.finish_declarator(ty, span, &sattrs)?;
            if let CType::Record(id) = &ty {
                // glibc's `__SOCKADDR_ARG`: `typedef union { ... } name
                // __attribute__ ((__transparent_union__));`.
                if attrs.transparent_union {
                    self.records.defs[*id].transparent = true;
                }
                // An alignment on the typedef of an untagged record is an
                // alignment of that record (it has no other name).
                if let Some(a) = attrs.aligned
                    && self.records.defs[*id].tag.is_none()
                {
                    let def = &mut self.records.defs[*id];
                    def.align = max_align(def.align, Some(a));
                }
            }
            self.declare_typedef(&name, ty);
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        self.expect_punct(Punct::Semi, "';' after typedef")?;
        Ok(Vec::new())
    }

    /// Parse zero or more `[const-expr]` / `[]` array suffixes onto `base`,
    /// building the C array type (`base[A][B]` is array-A-of-array-B-of-base).
    /// An empty `[]` yields an incomplete array (length `0`, deduced later).
    fn parse_array_suffix(&mut self, base: CType) -> PResult<CType> {
        let mut dims = Vec::new();
        while self.is_punct(Punct::LBracket) {
            self.bump();
            // A parameter's array declarator may carry qualifiers and `static`
            // (`char *argv[__restrict]`, `int a[static 4]`); they only matter to
            // the parameter's pointer adjustment, which ignores them.
            while self.is_kw(Keyword::Static)
                || self.is_kw(Keyword::Const)
                || self.is_kw(Keyword::Volatile)
                || self.is_kw(Keyword::Restrict)
            {
                self.bump();
            }
            if self.is_punct(Punct::RBracket) {
                dims.push(0u64);
            } else if self.in_params > 0
                && dims.is_empty()
                && (self.is_punct(Punct::Star)
                    && matches!(self.peek_at(1), TokenKind::Punct(Punct::RBracket)))
            {
                // `[*]`: a VLA parameter of unspecified size.
                self.bump();
                dims.push(0u64);
            } else {
                let sp = self.peek_span();
                let e = self.parse_assign()?;
                match self.eval_const_expr(&e) {
                    Some(n) if n < 0 => return self.err("array size must be non-negative"),
                    Some(n) => dims.push(n as u64),
                    // The outermost bound of a parameter array may be any
                    // expression (`regmatch_t m[__restrict n]`): the parameter is
                    // adjusted to a pointer, so the bound is irrelevant.
                    None if self.in_params > 0 && dims.is_empty() => dims.push(0u64),
                    // A block-scope object's outermost bound may be a run-time
                    // value: a variable-length array (C99 6.7.6.2).
                    None if self.vla_ok && dims.is_empty() && self.vla_len.is_none() => {
                        self.vla_len = Some(e);
                        dims.push(0u64);
                    }
                    None => {
                        return Err(Diagnostic::error(
                            "expected a constant integer expression (variable-length arrays are unsupported)",
                        )
                        .with_span(sp));
                    }
                }
            }
            self.expect_punct(Punct::RBracket, "']' after array size")?;
        }
        let mut ty = base;
        for &d in dims.iter().rev() {
            ty = CType::Array(Box::new(ty), d);
        }
        Ok(ty)
    }

    /// Parse and ignore any attribute specifier sequences at the cursor: C23
    /// `[[ ... ]]` sequences and GNU `__attribute__((...))`. Standard attributes
    /// are accepted and ignored per C23 (an implementation may ignore any
    /// attribute it does not recognize).
    fn skip_attributes(&mut self) -> PResult<()> {
        self.parse_attributes().map(|_| ())
    }

    /// Whether the cursor is at a GNU `__attribute__` (or `__attribute`).
    fn at_gnu_attribute(&self) -> bool {
        matches!(self.peek(), TokenKind::Ident(n) if n == "__attribute__" || n == "__attribute")
            && matches!(self.peek_at(1), TokenKind::Punct(Punct::LParen))
    }

    /// Whether the cursor is at any attribute specifier (GNU, or a C23 `[[`).
    fn at_attribute(&self) -> bool {
        self.at_gnu_attribute()
            || (self.is_punct(Punct::LBracket)
                && matches!(self.peek_at(1), TokenKind::Punct(Punct::LBracket)))
    }

    /// Parse any attribute specifier sequences at the cursor — C23 `[[ ... ]]`
    /// and GNU `__attribute__((...))` — returning the GNU attributes that carry
    /// meaning for lf-cc (see [`Attrs`]). The GNU spelling lives in the reserved
    /// namespace and system headers use it under every `-std`, so it is accepted
    /// in all dialects.
    fn parse_attributes(&mut self) -> PResult<Attrs> {
        let mut attrs = Attrs::default();
        loop {
            // C23 `[[ attribute-list ]]`.
            if self.is_punct(Punct::LBracket)
                && matches!(self.peek_at(1), TokenKind::Punct(Punct::LBracket))
            {
                if !self.std.attributes() {
                    return self.err(
                        "attribute specifier sequences '[[...]]' are a C23 feature (use -std=c23)",
                    );
                }
                self.bump();
                self.bump(); // consume `[[`
                let mut depth = 2u32;
                loop {
                    match self.peek() {
                        TokenKind::Punct(Punct::LBracket) => depth += 1,
                        TokenKind::Punct(Punct::RBracket) => depth -= 1,
                        TokenKind::Eof => return self.err("unterminated attribute specifier"),
                        _ => {}
                    }
                    self.bump();
                    if depth == 0 {
                        break;
                    }
                }
                continue;
            }
            // GNU `__attribute__ (( attribute-list ))`.
            if self.at_gnu_attribute() {
                self.bump(); // __attribute__
                self.expect_punct(Punct::LParen, "'(' after __attribute__")?;
                self.expect_punct(Punct::LParen, "'((' after __attribute__")?;
                loop {
                    if self.is_punct(Punct::RParen) {
                        break;
                    }
                    if !self.is_punct(Punct::Comma) {
                        self.parse_one_gnu_attribute(&mut attrs)?;
                    }
                    if !self.eat_punct(Punct::Comma) {
                        break;
                    }
                }
                self.expect_punct(Punct::RParen, "')' to close __attribute__")?;
                self.expect_punct(Punct::RParen, "'))' to close __attribute__")?;
                continue;
            }
            return Ok(attrs);
        }
    }

    /// Parse one entry of a GNU attribute list — a name (an identifier, or a
    /// keyword such as `const`) with an optional parenthesized argument list —
    /// recording it in `attrs` when lf-cc gives it meaning.
    fn parse_one_gnu_attribute(&mut self, attrs: &mut Attrs) -> PResult<()> {
        let name = match self.peek().clone() {
            TokenKind::Ident(n) => n,
            TokenKind::Keyword(_) => String::new(),
            _ => return self.err("expected an attribute name"),
        };
        self.bump();
        // `__name__` and `name` spell the same attribute.
        let bare = strip_dunder(&name).to_owned();
        if !self.is_punct(Punct::LParen) {
            match bare.as_str() {
                // A bare `aligned` requests the target's largest useful alignment.
                "aligned" => attrs.aligned = Some(attrs.aligned.map_or(16, |a| a.max(16))),
                "packed" => attrs.packed = true,
                "gnu_inline" => attrs.gnu_inline = true,
                "transparent_union" => attrs.transparent_union = true,
                "weak" => attrs.weak = true,
                _ => {}
            }
            return Ok(());
        }
        match bare.as_str() {
            "visibility" => {
                self.bump(); // (
                let sp = self.peek_span();
                let v = self.parse_asm_string("a visibility")?;
                attrs.visibility = Some(match v.as_str() {
                    "default" => Visibility::Default,
                    // ELF `STV_INTERNAL` is `STV_HIDDEN` plus a processor-specific
                    // promise; as gcc does on x86-64, treat it as hidden.
                    "hidden" | "internal" => Visibility::Hidden,
                    "protected" => Visibility::Protected,
                    _ => {
                        return Err(Diagnostic::error(format!(
                            "unknown visibility '{v}' (expected default, hidden, protected or internal)"
                        ))
                        .with_span(sp));
                    }
                });
                self.expect_punct(Punct::RParen, "')' after the visibility")?;
            }
            "mode" => {
                self.bump(); // (
                let (m, sp) = self.expect_ident()?;
                let width = match strip_dunder(&m) {
                    "QI" | "byte" => 8,
                    "HI" => 16,
                    "SI" => 32,
                    "DI" | "word" | "pointer" | "unwind_word" => 64,
                    "TI" => 128,
                    _ => {
                        return Err(Diagnostic::error(format!(
                            "unsupported machine mode '{m}' in a mode attribute"
                        ))
                        .with_span(sp));
                    }
                };
                attrs.mode = Some(width);
                self.expect_punct(Punct::RParen, "')' after the machine mode")?;
            }
            "aligned" | "vector_size" => {
                self.bump(); // (
                let sp = self.peek_span();
                let n = self.parse_const_expr()?;
                if bare == "aligned" {
                    if n <= 0 || (n & (n - 1)) != 0 {
                        return Err(Diagnostic::error(
                            "requested alignment is not a positive power of 2",
                        )
                        .with_span(sp));
                    }
                    let n = n as u64;
                    attrs.aligned = Some(attrs.aligned.map_or(n, |a| a.max(n)));
                } else {
                    if n <= 0 {
                        return Err(Diagnostic::error("the vector size must be positive").with_span(sp));
                    }
                    attrs.vector_size = Some(n as u64);
                }
                self.expect_punct(Punct::RParen, "')' after the attribute argument")?;
            }
            _ => self.skip_balanced_parens()?,
        }
        Ok(())
    }

    /// Skip a balanced `( ... )` group starting at the `(` at the cursor.
    fn skip_balanced_parens(&mut self) -> PResult<()> {
        let mut depth = 0u32;
        loop {
            match self.peek() {
                TokenKind::Punct(Punct::LParen) => depth += 1,
                TokenKind::Punct(Punct::RParen) => depth -= 1,
                TokenKind::Eof => return self.err("unterminated parenthesized group"),
                _ => {}
            }
            self.bump();
            if depth == 0 {
                return Ok(());
            }
        }
    }

    /// Apply the type-changing attributes of `attrs` to a declared type:
    /// `mode(...)` resizes an integer type (keeping its signedness), and
    /// `vector_size(N)` makes it a GCC vector of `N` bytes.
    fn apply_type_attrs(&self, ty: CType, attrs: &Attrs, span: Span) -> PResult<CType> {
        if let CType::Qual(inner, q) = ty {
            return Ok(self.apply_type_attrs(*inner, attrs, span)?.qualified(q));
        }
        let ty = match (attrs.mode, ty) {
            (Some(w), CType::Int(i)) => CType::Int(IntTy::new(w, i.signed)),
            (Some(w), CType::Bool) => CType::Int(IntTy::new(w, false)),
            (_, ty) => ty,
        };
        match attrs.vector_size {
            Some(bytes) => self.vector_type(ty, bytes, span),
            None => Ok(ty),
        }
    }

    /// The GCC vector type of `bytes` bytes of `elem` elements: the element
    /// must be an integer (not `_Bool` or a `_BitInt`) or `float`/`double`,
    /// and the size a power-of-two multiple of the element size.
    fn vector_type(&self, elem: CType, bytes: u64, span: Span) -> PResult<CType> {
        let valid_elem = matches!(&elem, CType::Int(i) if i.bitint.is_none() && i.width <= 64)
            || matches!(elem, CType::Float(FloatTy::F32 | FloatTy::F64));
        if !valid_elem {
            return Err(Diagnostic::error(format!(
                "invalid vector element type '{elem}' (an integer or floating type is required)"
            ))
            .with_span(span));
        }
        let esize = layout::size_of(&self.records, &elem);
        if !bytes.is_multiple_of(esize) || !(bytes / esize).is_power_of_two() {
            return Err(Diagnostic::error(format!(
                "the vector size {bytes} is not a power-of-two multiple of the element size {esize}"
            ))
            .with_span(span));
        }
        Ok(CType::Vector(Box::new(elem), (bytes / esize) as u32))
    }

    /// Finish a declarator: parse its trailing GNU extensions (attributes and an
    /// optional asm label), then apply the declaration-specifier and declarator
    /// attributes to its type. Returns `(asm label, adjusted type, attributes)`.
    fn finish_declarator(
        &mut self,
        ty: CType,
        span: Span,
        sattrs: &Attrs,
    ) -> PResult<(Option<String>, CType, Attrs)> {
        let label = self.parse_declarator_extensions()?;
        let dattrs = std::mem::take(&mut self.decl_attrs);
        // The declaration specifiers' attributes already shaped the base type;
        // the declarator's own apply to the declared type.
        let ty = self.apply_type_attrs(ty, &dattrs, span)?;
        let mut attrs = sattrs.clone();
        attrs.merge(dattrs);
        Ok((label, ty, attrs))
    }

    /// Parse the GNU extensions that may trail a declarator: any mix of
    /// attribute specifiers and at most one asm label `asm ("symbol")`, as in
    /// `int f(int) __asm__ ("" "g") __THROW __wur;`. Returns the label, if any.
    fn parse_declarator_extensions(&mut self) -> PResult<Option<String>> {
        let mut label: Option<String> = None;
        loop {
            let a = self.parse_attributes()?;
            self.decl_attrs.merge(a);
            if !self.is_kw(Keyword::Asm) {
                return Ok(label);
            }
            if label.is_some() {
                return self.err("a declarator may carry only one asm label");
            }
            self.bump(); // asm
            self.expect_punct(Punct::LParen, "'(' after asm")?;
            let sym = self.parse_asm_string("an asm label")?;
            if sym.is_empty() {
                return self.err("an asm label must name a symbol");
            }
            self.expect_punct(Punct::RParen, "')' after asm label")?;
            label = Some(sym);
        }
    }

    /// Parse one or more adjacent narrow string literals (concatenated, as in
    /// translation phase 6) for an asm template, label, constraint, or clobber.
    fn parse_asm_string(&mut self, what: &str) -> PResult<String> {
        let mut bytes: Vec<u8> = Vec::new();
        let mut any = false;
        while let TokenKind::Str(s, kind) = self.peek().clone() {
            if kind != StrKind::Narrow {
                return self.err(format!("{what} must be a narrow string literal"));
            }
            bytes.extend_from_slice(&s);
            self.bump();
            any = true;
        }
        if !any {
            return self.err(format!("expected a string literal for {what}"));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Parse a GNU asm statement or file-scope asm declaration (the `asm`
    /// keyword at the cursor) up to its closing `)`; the caller consumes the `;`:
    ///
    /// `asm qualifiers ( template [: outputs [: inputs [: clobbers [: labels]]]] )`
    ///
    /// The qualifiers are any of `volatile`, `inline`, and `goto`. A `::` is two
    /// `:` tokens here, so `asm("" ::: "memory")` needs no special casing.
    fn parse_asm_stmt(&mut self) -> PResult<AsmStmt> {
        self.bump(); // asm
        let mut stmt = AsmStmt {
            template: String::new(),
            extended: false,
            is_volatile: false,
            is_inline: false,
            is_goto: false,
            outputs: Vec::new(),
            inputs: Vec::new(),
            clobbers: Vec::new(),
            labels: Vec::new(),
        };
        loop {
            match self.peek() {
                TokenKind::Keyword(Keyword::Volatile) => stmt.is_volatile = true,
                TokenKind::Keyword(Keyword::Inline) => stmt.is_inline = true,
                // `inline` is a plain identifier under C89, but still an asm
                // qualifier in GNU C.
                TokenKind::Ident(n) if n == "inline" => stmt.is_inline = true,
                TokenKind::Keyword(Keyword::Goto) => stmt.is_goto = true,
                _ => break,
            }
            self.bump();
        }
        self.expect_punct(Punct::LParen, "'(' after asm")?;
        stmt.template = self.parse_asm_string("the asm template")?;
        // Each section is optional from the right; a present section may be
        // empty (`asm("" : : "r"(x))`).
        if self.eat_punct(Punct::Colon) {
            stmt.extended = true;
            stmt.outputs = self.parse_asm_operands()?;
            if self.eat_punct(Punct::Colon) {
                stmt.inputs = self.parse_asm_operands()?;
                if self.eat_punct(Punct::Colon) {
                    if matches!(self.peek(), TokenKind::Str(..)) {
                        loop {
                            stmt.clobbers.push(self.parse_asm_string("an asm clobber")?);
                            if !self.eat_punct(Punct::Comma) {
                                break;
                            }
                        }
                    }
                    if self.eat_punct(Punct::Colon) && matches!(self.peek(), TokenKind::Ident(_)) {
                        loop {
                            stmt.labels.push(self.expect_ident()?.0);
                            if !self.eat_punct(Punct::Comma) {
                                break;
                            }
                        }
                    }
                }
            }
        }
        self.expect_punct(Punct::RParen, "')' to close asm")?;
        Ok(stmt)
    }

    /// Parse a (possibly empty) comma-separated list of extended-asm operands,
    /// `[name] "constraint" (expr)`, stopping before a `:` or `)`.
    fn parse_asm_operands(&mut self) -> PResult<Vec<AsmOperand>> {
        let mut ops = Vec::new();
        if self.is_punct(Punct::Colon) || self.is_punct(Punct::RParen) {
            return Ok(ops);
        }
        loop {
            let name = if self.eat_punct(Punct::LBracket) {
                let (n, _) = self.expect_ident()?;
                self.expect_punct(Punct::RBracket, "']' after asm operand name")?;
                Some(n)
            } else {
                None
            };
            let constraint = self.parse_asm_string("an asm operand constraint")?;
            self.expect_punct(Punct::LParen, "'(' before asm operand expression")?;
            let expr = self.parse_expr()?;
            self.expect_punct(Punct::RParen, "')' after asm operand expression")?;
            ops.push(AsmOperand { name, constraint, expr });
            if !self.eat_punct(Punct::Comma) {
                return Ok(ops);
            }
        }
    }

    /// Consume any leading storage-class / function specifiers, returning the
    /// linkage-affecting one (`extern`/`static`) if present. `register`, `auto`,
    /// `inline`, and `_Noreturn` do not affect linkage and yield [`Storage::None`].
    ///
    /// `inline` and `_Thread_local`/`__thread` are recorded in `spec_inline` /
    /// `spec_thread`, and interleaved GNU attributes (`extern __inline
    /// __attribute__ ((__gnu_inline__)) int f ...`) are handed on to the
    /// declaration-specifier parser through `pending_attrs`.
    fn consume_storage(&mut self) -> PResult<Storage> {
        let mut storage = Storage::None;
        loop {
            if self.at_attribute() {
                let a = self.parse_attributes()?;
                self.pending_attrs.merge(a);
                continue;
            }
            match self.peek() {
                TokenKind::Keyword(Keyword::Extern) => storage = Storage::Extern,
                TokenKind::Keyword(Keyword::Static) => storage = Storage::Static,
                TokenKind::Keyword(Keyword::Inline) => self.spec_inline = true,
                TokenKind::Keyword(Keyword::Register | Keyword::Auto | Keyword::Noreturn) => {}
                TokenKind::Ident(n) if n == "__extension__" => {}
                TokenKind::Ident(n) if self.is_thread_spec(n) => {
                    self.spec_thread = true;
                }
                _ => break,
            }
            self.bump();
        }
        Ok(storage)
    }

    fn parse_param_list(&mut self) -> PResult<(Vec<Param>, bool)> {
        self.in_params += 1;
        let r = self.parse_param_list_inner();
        self.in_params -= 1;
        r
    }

    fn parse_param_list_inner(&mut self) -> PResult<(Vec<Param>, bool)> {
        self.expect_punct(Punct::LParen, "'('")?;
        let mut params = Vec::new();
        let mut variadic = false;
        let mut transparent: Vec<(usize, CType)> = Vec::new();
        self.transparent_params = Vec::new();
        // Empty parentheses `()` declare a function whose parameters are
        // *unspecified* (an old-style / K&R declarator), NOT a function taking no
        // arguments. Calls to it are unchecked and the arguments undergo the
        // default argument promotions. We model this as an empty, variadic
        // parameter list: `check_call` then imposes no arity or type constraints
        // and applies the promotions, and the lowered IR function is variadic so
        // the verifier accepts calls carrying arguments. An explicit `(void)`
        // (below) is the distinct "takes no arguments" prototype.
        if self.eat_punct(Punct::RParen) {
            return Ok((params, true));
        }
        // `(void)` means an explicit empty parameter list (a prototype taking no
        // arguments); calls with any argument are rejected.
        if self.is_kw(Keyword::Void) && matches!(self.peek_at(1), TokenKind::Punct(Punct::RParen)) {
            self.bump();
            self.bump();
            return Ok((params, variadic));
        }
        loop {
            if self.eat_punct(Punct::Ellipsis) {
                variadic = true;
                break;
            }
            // A prototype parameter may carry the `register` storage-class
            // specifier (the only one permitted on a parameter), e.g.
            // `int f(register int x)`. Consume and ignore it.
            self.consume_storage()?;
            let base = self.parse_decl_specs()?;
            let sattrs = self.spec_attrs.clone();
            let (name, ty, span) = self.declarator(base)?;
            // A trailing attribute on the parameter declarator, e.g.
            // `int desc __attribute__((unused))` (GNU) or `int x [[maybe_unused]]`.
            let (_label, ty, _attrs) = self.finish_declarator(ty, span, &sattrs)?;
            // A parameter of array or function type decays to a pointer; its
            // top-level qualifiers are not part of the function's type.
            let ty = ty.unqual().clone();
            let ty = ty.decayed().unwrap_or(ty);
            // A parameter of transparent-union type (GNU) is passed exactly like
            // the union's first member, and accepts an argument of any member
            // type; model it as that first member's type (glibc's
            // `__SOCKADDR_ARG` and `__CONST_SOCKADDR_ARG` are pointer unions). A
            // function *definition* rebuilds the union from it on entry (see
            // `transparent_params`).
            let ty = match &ty {
                CType::Record(id)
                    if self.records.get(*id).transparent
                        && !self.records.get(*id).fields.is_empty() =>
                {
                    transparent.push((params.len(), ty.clone()));
                    self.records.get(*id).fields[0].ty.clone()
                }
                _ => ty,
            };
            params.push(Param { name, ty, span });
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        self.expect_punct(Punct::RParen, "')' to close parameter list")?;
        self.transparent_params = transparent;
        Ok((params, variadic))
    }

    /// For a function definition whose parameter list (just parsed) had
    /// transparent-union parameters: rename each such parameter to a hidden
    /// name holding the first member, and return the declarations that rebuild
    /// the union under the parameter's own name at the top of the body.
    fn rebuild_transparent_params(&mut self, params: &mut [Param]) -> Vec<Stmt> {
        let mut prologue = Vec::new();
        for (idx, uty) in std::mem::take(&mut self.transparent_params) {
            let p = &mut params[idx];
            let Some(name) = p.name.clone() else { continue };
            let hidden = format!("{name}.transparent");
            p.name = Some(hidden.clone());
            let span = p.span;
            self.declare_ordinary(&name, Some(uty.clone()));
            let init = Init::List(vec![InitItem {
                designators: Vec::new(),
                init: Init::Expr(Expr { kind: ExprKind::Ident(hidden), span }),
            }]);
            prologue.push(Stmt {
                kind: StmtKind::Decl(vec![VarDecl {
                    name,
                    ty: uty,
                    init: Some(init),
                    align: None,
                    storage: Storage::None,
                    asm_label: None,
                    thread_local: false,
                    attrs: SymAttrs::default(),
                    vla_len: None,
                    span,
                }]),
                span,
            });
        }
        prologue
    }

    /// Whether the `(` at the cursor opens an old-style (K&R) identifier list —
    /// the parameter names of an old-style function definition — rather than a
    /// prototype parameter list. This holds when the first token after `(` is an
    /// identifier that does not name a type (a typedef-name would begin a
    /// prototype parameter). Empty `()` and `(void)` are not identifier lists.
    fn at_kr_identifier_list(&self) -> bool {
        matches!(self.peek_at(1), TokenKind::Ident(n) if !self.ident_starts_decl(n))
    }

    /// Whether the identifier `name` begins a declaration: a typedef-name, a
    /// builtin type name that lexes as an identifier (`_BitInt`, `__int128`,
    /// `_Float128`, `__builtin_va_list`, `__typeof__`, ...), C23 `constexpr`, a
    /// thread-storage specifier, or a GNU `__attribute__`.
    fn ident_starts_decl(&self, name: &str) -> bool {
        is_builtin_type_ident(name)
            || matches!(name, "constexpr" | "__attribute__" | "__attribute")
            || self.is_thread_spec(name)
            || self.is_typedef_name(name)
    }

    /// Whether `name` is the thread storage-class specifier: C11
    /// `_Thread_local`, GNU `__thread`, or (a keyword from C23) `thread_local`.
    fn is_thread_spec(&self, name: &str) -> bool {
        matches!(name, "_Thread_local" | "__thread") || (name == "thread_local" && self.std.is_c23())
    }

    /// Parse the parameters of an old-style (K&R) function definition: the
    /// identifier list `(id, id, ...)` at the cursor, followed by the
    /// declaration-list that gives those identifiers their types. An identifier
    /// that the declaration-list does not mention defaults to `int`. The
    /// parameters are returned in identifier-list (declaration) order, with
    /// array/function types decayed to pointers as for prototype parameters.
    fn parse_kr_definition_params(&mut self) -> PResult<Vec<Param>> {
        self.expect_punct(Punct::LParen, "'('")?;
        let mut names: Vec<(String, Span)> = Vec::new();
        loop {
            let (n, sp) = self.expect_ident()?;
            names.push((n, sp));
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        self.expect_punct(Punct::RParen, "')' to close identifier list")?;
        // The declaration-list types the named parameters. It runs until the
        // function body's `{`. Each declaration may declare several parameters
        // (`int a, b;`) and may carry a `register` storage-class specifier.
        let mut types: HashMap<String, (CType, Span)> = HashMap::new();
        while !self.is_punct(Punct::LBrace) && !self.at_eof() {
            self.consume_storage()?;
            let base = self.parse_decl_specs()?;
            let sattrs = self.spec_attrs.clone();
            self.consume_storage()?;
            if self.eat_punct(Punct::Semi) {
                continue;
            }
            loop {
                let (pname, pty, psp) = self.parse_named_declarator(base.clone())?;
                let (_label, pty, _attrs) = self.finish_declarator(pty, psp, &sattrs)?;
                let pty = pty.unqual().clone();
                let pty = pty.decayed().unwrap_or(pty);
                if !names.iter().any(|(n, _)| *n == pname) {
                    return Err(Diagnostic::error(format!(
                        "'{pname}' is not one of the function's parameters"
                    ))
                    .with_span(psp));
                }
                if types.insert(pname.clone(), (pty, psp)).is_some() {
                    return Err(Diagnostic::error(format!(
                        "redeclaration of parameter '{pname}'"
                    ))
                    .with_span(psp));
                }
                if !self.eat_punct(Punct::Comma) {
                    break;
                }
            }
            self.expect_punct(Punct::Semi, "';' after parameter declaration")?;
        }
        // Assemble the parameters in identifier-list order; an undeclared
        // identifier defaults to `int`.
        let params = names
            .into_iter()
            .map(|(n, sp)| match types.remove(&n) {
                Some((ty, tsp)) => Param { name: Some(n), ty, span: tsp },
                None => Param { name: Some(n), ty: CType::int(), span: sp },
            })
            .collect();
        Ok(params)
    }

    // --- types -------------------------------------------------------------

    /// Whether the current token begins a declaration (a type or storage
    /// specifier keyword, or a typedef-name identifier).
    fn at_type_specifier(&self) -> bool {
        match self.peek() {
            TokenKind::Keyword(kw) => self.keyword_starts_decl(*kw),
            // C23 `_BitInt`/`constexpr` are lexed as identifiers; treat them as
            // declaration starts so a declaration using them is recognized (the
            // C23 gate is applied in `parse_decl_specs`).
            TokenKind::Ident(name) => self.ident_starts_decl(name),
            _ => false,
        }
    }

    /// Whether the keyword `kw` begins a declaration.
    fn keyword_starts_decl(&self, kw: Keyword) -> bool {
        matches!(
            kw,
            Keyword::Void
                | Keyword::Bool
                | Keyword::Char
                | Keyword::Short
                | Keyword::Int
                | Keyword::Long
                | Keyword::Float
                | Keyword::Double
                | Keyword::Signed
                | Keyword::Unsigned
                | Keyword::Const
                | Keyword::Volatile
                | Keyword::Atomic
                | Keyword::Restrict
                | Keyword::Inline
                | Keyword::Noreturn
                | Keyword::Alignas
                | Keyword::Typeof
                | Keyword::TypeofUnqual
                | Keyword::Extern
                | Keyword::Static
                | Keyword::Register
                | Keyword::Auto
                | Keyword::Struct
                | Keyword::Union
                | Keyword::Enum
                | Keyword::Typedef
        )
    }

    fn parse_decl_specs(&mut self) -> PResult<CType> {
        self.parse_decl_specs_impl(false)
    }

    /// Parse declaration specifiers. When `allow_implicit_int` is set and no type
    /// specifier is present, the type defaults to `int` (the pre-C99 / K&R
    /// "implicit int" rule, a GNU extension retained in later dialects) provided
    /// a declarator plausibly follows. Only declaration contexts (file scope and
    /// K&R parameter declaration-lists) enable this, so parenthesised
    /// expressions are never misread as casts.
    ///
    /// GNU attributes may appear anywhere among the specifiers; they (and any
    /// met among preceding storage-class specifiers) are left in `spec_attrs`,
    /// and a `mode` attribute among them already resizes the returned type.
    fn parse_decl_specs_impl(&mut self, allow_implicit_int: bool) -> PResult<CType> {
        let start = self.peek_span();
        let mut attrs = std::mem::take(&mut self.pending_attrs);
        let outer_quals = std::mem::replace(&mut self.spec_quals, Quals::NONE);
        let ty = self.parse_decl_specs_body(allow_implicit_int, &mut attrs);
        let quals = std::mem::replace(&mut self.spec_quals, outer_quals);
        let ty = self.apply_type_attrs(ty?, &attrs, start)?;
        self.spec_attrs = attrs;
        Ok(ty.qualified(quals))
    }

    fn parse_decl_specs_body(
        &mut self,
        allow_implicit_int: bool,
        attrs: &mut Attrs,
    ) -> PResult<CType> {
        let start = self.peek_span();
        self.last_alignas = None;
        self.last_constexpr = false;
        let mut longs = 0u8;
        let mut has_short = false;
        let mut has_char = false;
        let mut has_int = false;
        let mut has_void = false;
        let mut has_bool = false;
        let mut has_float = false;
        let mut has_double = false;
        let mut signed_spec: Option<bool> = None;
        let mut saw_any = false;
        let mut explicit: Option<CType> = None;
        // A pending `_BitInt(N)` value-bit count (the signedness is applied at the
        // end, since `unsigned`/`signed` may appear on either side of `_BitInt`).
        let mut bitint_n: Option<u16> = None;
        // GNU `__int128` (combines with `signed`/`unsigned`).
        let mut has_int128 = false;
        // C99 `_Complex` (combines with `float`/`double`/`long double`).
        let mut has_complex = false;
        // An `__extension__` among the specifiers lifts the pedantic gates.
        let mut marked = false;

        loop {
            if self.at_attribute() {
                let a = self.parse_attributes()?;
                attrs.merge(a);
                continue;
            }
            if self.at_extension() {
                self.bump();
                marked = true;
                continue;
            }
            let numeric_seen = has_void
                || has_bool
                || has_char
                || has_short
                || has_int
                || has_float
                || has_double
                || has_int128
                || longs > 0
                || signed_spec.is_some();
            // C23 `constexpr` (a declaration specifier) and `_BitInt` (a type
            // specifier) are lexed as ordinary identifiers (they are not global
            // keywords); recognize them here and gate them on C23. So are the
            // GNU/TS 18661 builtin type names and the thread-storage specifiers.
            if let TokenKind::Ident(name) = self.peek().clone() {
                let sp = self.peek_span();
                match name.as_str() {
                    n if self.is_thread_spec(n) => {
                        self.bump();
                        self.spec_thread = true;
                        saw_any = true;
                        continue;
                    }
                    "__int128" => {
                        self.bump();
                        has_int128 = true;
                        saw_any = true;
                        continue;
                    }
                    "_Complex" | "__complex__" => {
                        self.bump();
                        has_complex = true;
                        saw_any = true;
                        continue;
                    }
                    _ => {}
                }
                // A builtin type name that the program (or glibc, below GCC 7) has
                // `typedef`ed is resolved as that typedef instead.
                if explicit.is_none() && !numeric_seen && !self.is_typedef_name(&name) {
                    let ty = match name.as_str() {
                        "__int128_t" => Some(CType::Int(IntTy::new(128, true))),
                        "__uint128_t" => Some(CType::Int(IntTy::new(128, false))),
                        "_Float32" => Some(CType::float()),
                        // `long double` is modelled as `double` in this subset, so
                        // `_Float64x` (and the x87 `__float80`) follow it.
                        "_Float64" | "_Float32x" | "_Float64x" | "__float80" => {
                            Some(CType::double())
                        }
                        "_Float128" | "__float128" => Some(CType::Float(FloatTy::F128)),
                        "_Float16" | "_Float128x" | "__ibm128" => {
                            return Err(Diagnostic::error(format!(
                                "the floating type '{name}' is not supported"
                            ))
                            .with_span(sp));
                        }
                        "__builtin_va_list" => {
                            self.bump();
                            explicit = Some(self.builtin_va_list());
                            saw_any = true;
                            continue;
                        }
                        "__typeof__" | "__typeof" => {
                            explicit = Some(self.parse_typeof()?);
                            saw_any = true;
                            continue;
                        }
                        _ => None,
                    };
                    if let Some(ty) = ty {
                        self.bump();
                        explicit = Some(ty);
                        saw_any = true;
                        continue;
                    }
                }
            }
            if let TokenKind::Ident(name) = self.peek() {
                if name == "constexpr" {
                    if !self.std.is_c23() {
                        return self
                            .err("`constexpr` is a C23 feature (use -std=c23 or later)");
                    }
                    self.bump();
                    self.last_constexpr = true;
                    saw_any = true;
                    continue;
                }
                if name == "_BitInt" {
                    if !self.std.is_c23() {
                        return self
                            .err("`_BitInt` is a C23 feature (use -std=c23 or later)");
                    }
                    if bitint_n.is_some() {
                        return self.err("duplicate `_BitInt` type specifier");
                    }
                    let sp = self.peek_span();
                    self.bump();
                    self.expect_punct(Punct::LParen, "'(' after _BitInt")?;
                    let n = self.parse_const_expr()?;
                    self.expect_punct(Punct::RParen, "')' after _BitInt width")?;
                    if n < 1 {
                        return Err(Diagnostic::error(
                            "the width of a `_BitInt` must be at least 1",
                        )
                        .with_span(sp));
                    }
                    if n > 64 {
                        return Err(Diagnostic::error(format!(
                            "`_BitInt({n})` is unsupported (>64-bit `_BitInt` is not implemented)"
                        ))
                        .with_span(sp));
                    }
                    bitint_n = Some(n as u16);
                    saw_any = true;
                    continue;
                }
            }
            // A `struct`/`union`/`enum` specifier, or a typedef-name, supplies the
            // whole type; it may not combine with the numeric specifiers.
            match self.peek() {
                TokenKind::Keyword(Keyword::Struct) if explicit.is_none() && !numeric_seen => {
                    explicit = Some(self.parse_record(RecordKind::Struct)?);
                    saw_any = true;
                    continue;
                }
                TokenKind::Keyword(Keyword::Union) if explicit.is_none() && !numeric_seen => {
                    explicit = Some(self.parse_record(RecordKind::Union)?);
                    saw_any = true;
                    continue;
                }
                TokenKind::Keyword(Keyword::Enum) if explicit.is_none() && !numeric_seen => {
                    explicit = Some(self.parse_enum()?);
                    saw_any = true;
                    continue;
                }
                TokenKind::Keyword(Keyword::Typeof | Keyword::TypeofUnqual)
                    if explicit.is_none() && !numeric_seen =>
                {
                    explicit = Some(self.parse_typeof()?);
                    saw_any = true;
                    continue;
                }
                TokenKind::Ident(name) if explicit.is_none() && !numeric_seen => {
                    match self.typedef_type(name) {
                        Some(ty) => {
                            explicit = Some(ty);
                            saw_any = true;
                            self.bump();
                            continue;
                        }
                        None => break,
                    }
                }
                _ => {}
            }
            match self.peek() {
                TokenKind::Keyword(Keyword::Const | Keyword::Restrict | Keyword::Noreturn) => {
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Volatile) => {
                    self.spec_quals.volatile = true;
                    self.bump();
                }
                // `_Atomic ( type-name )`: the atomic version of a type (a type
                // specifier); a bare `_Atomic` is the qualifier.
                TokenKind::Keyword(Keyword::Atomic)
                    if matches!(self.peek_at(1), TokenKind::Punct(Punct::LParen)) =>
                {
                    if explicit.is_some() || numeric_seen {
                        return self.err("`_Atomic(type-name)` cannot combine with another type specifier");
                    }
                    self.bump();
                    self.bump(); // (
                    let inner = self.parse_type_name()?;
                    self.expect_punct(Punct::RParen, "')' after the _Atomic type name")?;
                    explicit = Some(inner.qualified(Quals { volatile: false, atomic: true }));
                    saw_any = true;
                }
                TokenKind::Keyword(Keyword::Atomic) => {
                    self.spec_quals.atomic = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Inline) => {
                    self.spec_inline = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Alignas) => {
                    let prev = self.last_alignas;
                    let a = self.parse_alignas()?;
                    self.last_alignas = Some(prev.map_or(a, |p| p.max(a)));
                }
                TokenKind::Keyword(Keyword::Void) => {
                    has_void = true;
                    saw_any = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Bool) => {
                    has_bool = true;
                    saw_any = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Char) => {
                    has_char = true;
                    saw_any = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Short) => {
                    has_short = true;
                    saw_any = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Int) => {
                    has_int = true;
                    saw_any = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Long) => {
                    longs += 1;
                    saw_any = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Float) => {
                    has_float = true;
                    saw_any = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Double) => {
                    has_double = true;
                    saw_any = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Signed) => {
                    signed_spec = Some(true);
                    saw_any = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Unsigned) => {
                    signed_spec = Some(false);
                    saw_any = true;
                    self.bump();
                }
                _ => break,
            }
        }

        if !saw_any {
            // Implicit `int`: a pre-C99 declaration with no type specifier, e.g.
            // the K&R function definition `foo(x) int x; { ... }`. Permitted only
            // when a declarator plausibly follows (an identifier or `*`/`(`), and
            // only in the pre-C99 dialects gcc accepts it in (`c89`/`gnu89`).
            if allow_implicit_int
                && !self.std.is_c99()
                && matches!(
                    self.peek(),
                    TokenKind::Ident(_) | TokenKind::Punct(Punct::Star | Punct::LParen)
                )
            {
                return Ok(CType::int());
            }
            return Err(Diagnostic::error("expected a type").with_span(start));
        }
        // `_Complex float` / `_Complex double` / `_Complex long double` (a bare
        // `_Complex` is `double _Complex`, as in gcc). Complex values are not
        // implemented: the types serve declarations (<complex.h> prototypes).
        if has_complex {
            let fty = match &explicit {
                Some(CType::Float(FloatTy::F32)) => Some(FloatTy::C32),
                Some(CType::Float(FloatTy::F64)) => Some(FloatTy::C64),
                Some(CType::Float(FloatTy::F128)) => Some(FloatTy::C128),
                Some(CType::Float(f)) if f.is_complex() => Some(*f),
                Some(_) => None,
                None if has_char || has_short || has_int || has_bool || has_void || has_int128
                    || signed_spec.is_some()
                    || (longs > 0 && !has_double) =>
                {
                    None
                }
                None if has_float => Some(FloatTy::C32),
                None => Some(FloatTy::C64),
            };
            return match fty {
                Some(f) => Ok(CType::Float(f)),
                None => Err(Diagnostic::error("complex integer types are not supported")
                    .with_span(start)),
            };
        }
        if let Some(ty) = explicit {
            return Ok(ty);
        }
        // A `_BitInt(N)`: only `signed`/`unsigned` may accompany it.
        if let Some(n) = bitint_n {
            if has_void || has_bool || has_char || has_short || has_int || has_float
                || has_double || longs > 0
            {
                return Err(Diagnostic::error(
                    "`_BitInt` may not be combined with another type specifier",
                )
                .with_span(start));
            }
            let signed = signed_spec.unwrap_or(true);
            if signed && n < 2 {
                return Err(Diagnostic::error(
                    "the width of a signed `_BitInt` must be at least 2 (one bit is the sign)",
                )
                .with_span(start));
            }
            return Ok(CType::Int(IntTy::bit_int(n, signed)));
        }
        // GNU `__int128` / `unsigned __int128`: only `signed`/`unsigned` (and a
        // redundant `int`) may accompany it.
        if has_int128 {
            if has_void || has_bool || has_char || has_short || has_float || has_double || longs > 0
            {
                return Err(Diagnostic::error("invalid combination of type specifiers")
                    .with_span(start));
            }
            return Ok(CType::Int(IntTy::new(128, signed_spec.unwrap_or(true))));
        }
        if longs >= 2 && !self.std.has_long_long() && self.extension == 0 && !marked {
            return Err(Diagnostic::error(
                "`long long` is a C99 feature (use -std=c99 or later)",
            )
            .with_span(start));
        }
        if has_void {
            return Ok(CType::Void);
        }
        if has_bool {
            return Ok(CType::Bool);
        }
        // Floating types. `float`, `double`, and `long double` (modelled as
        // `double`); the numeric integer specifiers may not combine with them.
        if has_float || has_double {
            if has_char || has_short || has_int || has_bool || signed_spec.is_some()
                || (has_float && (has_double || longs > 0))
                || (has_double && longs > 1)
            {
                return Err(Diagnostic::error("invalid combination of type specifiers")
                    .with_span(start));
            }
            return Ok(if has_double { CType::double() } else { CType::float() });
        }
        if has_char {
            // Plain `char` is signed on this target; `signed`/`unsigned` override.
            let signed = signed_spec.unwrap_or(true);
            return Ok(CType::Int(IntTy::new(8, signed)));
        }
        // `int` is the default; `short`/`long` override the width.
        let _ = has_int;
        let width = if has_short {
            16
        } else if longs >= 1 {
            64
        } else {
            32
        };
        let signed = signed_spec.unwrap_or(true);
        Ok(CType::Int(IntTy::new(width, signed)))
    }

    /// Consume leading `*` (with optional qualifiers and GNU attributes) and wrap
    /// `base`. Attributes directly before the first `*` are accepted too.
    fn parse_pointers(&mut self, mut base: CType) -> PResult<CType> {
        self.skip_attributes()?;
        while self.eat_punct(Punct::Star) {
            // Pointer qualifiers and attributes (`char *__restrict p`, `int
            // *volatile p`, `void * __attribute__((aligned(8))) p`); `volatile`
            // and `_Atomic` qualify the pointer object itself.
            let mut quals = Quals::NONE;
            loop {
                if self.is_kw(Keyword::Const) || self.is_kw(Keyword::Restrict) {
                    self.bump();
                } else if self.eat_kw(Keyword::Volatile) {
                    quals.volatile = true;
                } else if self.eat_kw(Keyword::Atomic) {
                    quals.atomic = true;
                } else if self.at_attribute() {
                    self.skip_attributes()?;
                } else {
                    break;
                }
            }
            base = CType::ptr_to(base).qualified(quals);
        }
        Ok(base)
    }

    /// Parse a type-name (used in casts and `sizeof`): specifiers plus an
    /// abstract declarator (pointer levels, array/function suffixes, and grouped
    /// declarators such as `(*)(int)`).
    fn parse_type_name(&mut self) -> PResult<CType> {
        let base = self.parse_decl_specs()?;
        let (_name, ty, _span) = self.declarator(base)?;
        Ok(ty)
    }

    /// Parse `typeof ( type-name | expression )` / `typeof_unqual ( ... )`
    /// (C23/GNU), yielding the operand's type. The expression is not evaluated,
    /// only typed. `typeof_unqual` strips qualifiers, which this subset does not
    /// model, so both spellings resolve to the same underlying type here.
    fn parse_typeof(&mut self) -> PResult<CType> {
        self.bump(); // typeof / typeof_unqual
        // A nested type-name parse would clobber any pending `alignas`; preserve it.
        let saved_alignas = self.last_alignas;
        self.expect_punct(Punct::LParen, "'(' after typeof")?;
        let ty = if self.at_type_specifier() {
            self.parse_type_name()?
        } else {
            let e = self.parse_expr()?;
            match self.expr_type(&e) {
                Some(t) => t,
                None => {
                    return self.err("cannot determine the type of this `typeof` operand");
                }
            }
        };
        self.expect_punct(Punct::RParen, "')' after typeof operand")?;
        self.last_alignas = saved_alignas;
        Ok(ty)
    }

    /// Parse `_Alignas ( type-name | const-expr )` / `alignas ( ... )`
    /// (C11/C23), returning the requested alignment in bytes.
    fn parse_alignas(&mut self) -> PResult<u64> {
        self.bump(); // _Alignas / alignas
        self.expect_punct(Punct::LParen, "'(' after alignas")?;
        let val = if self.at_type_specifier() {
            let ty = self.parse_type_name()?;
            layout::align_of(&self.records, &ty)
        } else {
            let n = self.parse_const_expr()?;
            if n < 0 {
                return self.err("alignment must be non-negative");
            }
            n as u64
        };
        self.expect_punct(Punct::RParen, "')' after alignas")?;
        Ok(val.max(1))
    }

    /// The static type of an expression for `typeof`, computed without evaluating
    /// it. Covers the forms a `typeof` operand realistically takes in this subset;
    /// returns `None` when the parser cannot determine the type (e.g. a call to a
    /// bare function name, whose signature the parser does not track).
    fn expr_type(&self, e: &Expr) -> Option<CType> {
        match &e.kind {
            ExprKind::IntLit(_, ty) | ExprKind::FloatLit(_, ty) => Some(ty.clone()),
            ExprKind::Ident(name) => {
                if let Some(t) = self.var_type(name) {
                    Some(t)
                } else if self.enum_map.contains_key(name) {
                    Some(CType::int())
                } else {
                    None
                }
            }
            ExprKind::StrLit(bytes, kind) => {
                let width = kind.elem_width();
                let n = bytes.len() as u64 / width + 1;
                Some(CType::Array(Box::new(kind.elem_type()), n))
            }
            ExprKind::Cast(ty, _) => Some(ty.clone()),
            ExprKind::CompoundLiteral(ty, _) => Some(ty.clone()),
            ExprKind::SizeofType(_) | ExprKind::SizeofExpr(_) | ExprKind::AlignofType(_) => {
                Some(size_t_ty())
            }
            ExprKind::Unary(op, inner) => {
                let it = self.expr_type(inner)?;
                match op {
                    UnaryOp::Deref => deref_target(&it),
                    UnaryOp::AddrOf => Some(CType::ptr_to(it)),
                    UnaryOp::LNot => Some(CType::int()),
                    UnaryOp::Neg | UnaryOp::Plus | UnaryOp::BitNot => Some(promote_ty(&it)),
                }
            }
            ExprKind::Binary(op, l, r) => {
                let lt = self.expr_type(l)?;
                let rt = self.expr_type(r)?;
                binary_type(*op, &lt, &rt)
            }
            ExprKind::Assign(_, l, _) => self.expr_type(l),
            ExprKind::Cond(_, t, f) => {
                let tt = self.expr_type(t)?;
                let ft = self.expr_type(f)?;
                if tt.is_arithmetic() && ft.is_arithmetic() {
                    Some(usual_arith_ty(&tt, &ft))
                } else if tt.is_pointer() {
                    Some(tt)
                } else {
                    Some(ft)
                }
            }
            ExprKind::Comma(_, b) => self.expr_type(b),
            ExprKind::PreInc(i) | ExprKind::PreDec(i) | ExprKind::PostInc(i)
            | ExprKind::PostDec(i) => self.expr_type(i),
            ExprKind::Index(base, _) => {
                let bt = self.expr_type(base)?;
                deref_target(&bt)
            }
            ExprKind::Member(base, name, arrow) => {
                let bt = self.expr_type(base)?;
                let rec = if *arrow { bt.pointee().cloned()? } else { bt };
                if let CType::Record(id) = *rec.unqual() {
                    layout::resolve_member(&self.records, id, name).map(|(_, ty)| ty)
                } else {
                    None
                }
            }
            ExprKind::Call(callee, _) => {
                let ct = self.expr_type(callee)?;
                match ct {
                    CType::Pointer(inner) => match *inner {
                        CType::Func(ft) => Some(ft.ret),
                        _ => None,
                    },
                    CType::Func(ft) => Some(ft.ret),
                    _ => None,
                }
            }
            ExprKind::Generic(..) => None,
            // `va_arg` yields its type operand; the others are `void`.
            ExprKind::VaArg(_, ty) | ExprKind::ConvertVector(_, ty) => Some(ty.clone()),
            ExprKind::VaStart(..) | ExprKind::VaEnd(_) | ExprKind::VaCopy(..) => Some(CType::Void),
            ExprKind::LabelAddr(_) => Some(CType::ptr_to(CType::Void)),
            ExprKind::StmtExpr(stmts) => match stmts.last() {
                Some(Stmt { kind: StmtKind::Expr(Some(e)), .. }) => self.expr_type(e),
                _ => Some(CType::Void),
            },
        }
    }

    // --- declarators -------------------------------------------------------

    /// Parse a concrete declarator (must name something), returning the declared
    /// name, its full type built from `base`, and the name's span.
    fn parse_named_declarator(&mut self, base: CType) -> PResult<(String, CType, Span)> {
        let (name, ty, span) = self.declarator(base)?;
        match name {
            Some(n) => Ok((n, ty, span)),
            None => Err(Diagnostic::error("expected a name in this declarator").with_span(span)),
        }
    }

    /// The core recursive declarator parser. Handles leading pointers, grouped
    /// (parenthesized) declarators — including the `(*name)` function-pointer and
    /// `(*name[N])` array-of-pointer forms — and trailing array/function suffixes.
    /// Returns `(name?, type, name-span)`; `name` is `None` for an abstract
    /// declarator (a parameter or type-name with no identifier).
    fn declarator(&mut self, base: CType) -> PResult<(Option<String>, CType, Span)> {
        // Leading pointers wrap the type the inner declarator ultimately builds on.
        let base = self.parse_pointers(base)?;
        if self.is_punct(Punct::LParen) && self.grouped_declarator_ahead() {
            self.bump(); // '('
            let inner_start = self.pos;
            // First pass: skip the inner declarator to find its matching ')'.
            self.skip_grouped_declarator()?;
            self.expect_punct(Punct::RParen, "')' in declarator")?;
            // Apply the suffixes that follow the group to `base`, forming the type
            // the inner declarator derives from.
            let outer = self.declarator_suffixes(base)?;
            let resume = self.pos;
            // Second pass: re-parse the inner declarator with the correct base.
            self.pos = inner_start;
            let (name, ty, span) = self.declarator(outer)?;
            self.pos = resume;
            // Attributes after the whole grouped declarator apply to it too.
            let a = self.parse_attributes()?;
            self.decl_attrs.merge(a);
            Ok((name, ty, span))
        } else {
            let (name, span) = match self.peek().clone() {
                TokenKind::Ident(n) if !self.at_gnu_attribute() => {
                    let sp = self.bump().span;
                    (Some(n), sp)
                }
                _ => (None, self.peek_span()),
            };
            // Attributes between the name and its suffixes (`int x
            // __attribute__((unused))[4]` is rare; `name attrs (params)` rarer).
            let pre = self.parse_attributes()?;
            let ty = if name.is_some() && self.is_punct(Punct::LParen) {
                // The name's own parameter list: keep the parameter names, which
                // a function definition through a grouped declarator needs.
                let (params, variadic) = self.parse_param_list()?;
                let param_tys = params.iter().map(|p| p.ty.clone()).collect();
                self.named_fn_params = Some(params);
                CType::Func(Box::new(FuncType { ret: base.unqual().clone(), params: param_tys, variadic }))
            } else {
                self.declarator_suffixes(base)?
            };
            // The declarator's own trailing attributes (`int x
            // __attribute__((aligned(8)))`, `typedef int T __attribute__((mode(DI)))`).
            // They replace whatever a nested parameter declarator left behind.
            let mut attrs = pre;
            attrs.merge(self.parse_attributes()?);
            self.decl_attrs = attrs;
            Ok((name, ty, span))
        }
    }

    /// Parse trailing declarator suffixes: a function parameter list (yielding a
    /// [`CType::Func`]) or array dimensions.
    fn declarator_suffixes(&mut self, base: CType) -> PResult<CType> {
        if self.is_punct(Punct::LParen) {
            let (params, variadic) = self.parse_param_list()?;
            let param_tys: Vec<CType> = params.into_iter().map(|p| p.ty).collect();
            let ret = base.unqual().clone();
            return Ok(CType::Func(Box::new(FuncType { ret, params: param_tys, variadic })));
        }
        self.parse_array_suffix(base)
    }

    /// Whether the `(` at the cursor opens a nested (grouped) declarator rather
    /// than a function parameter list: it does when it is followed by `*`, another
    /// `(`, or an identifier that is not a typedef-name (the declared name).
    fn grouped_declarator_ahead(&self) -> bool {
        match self.peek_at(1) {
            TokenKind::Punct(Punct::Star | Punct::LParen) => true,
            TokenKind::Ident(name) => {
                name == "__attribute__" || name == "__attribute" || !self.ident_starts_decl(name)
            }
            _ => false,
        }
    }

    /// Skip over the tokens of a grouped declarator's inner declarator, stopping
    /// at the `)` that closes the group (tracking nested `()`/`[]`).
    fn skip_grouped_declarator(&mut self) -> PResult<()> {
        let mut paren = 0u32;
        let mut bracket = 0u32;
        loop {
            match self.peek() {
                TokenKind::Punct(Punct::LParen) => paren += 1,
                TokenKind::Punct(Punct::RParen) => {
                    if paren == 0 && bracket == 0 {
                        return Ok(());
                    }
                    paren = paren.saturating_sub(1);
                }
                TokenKind::Punct(Punct::LBracket) => bracket += 1,
                TokenKind::Punct(Punct::RBracket) => bracket = bracket.saturating_sub(1),
                TokenKind::Eof => return self.err("unterminated declarator"),
                _ => {}
            }
            self.bump();
        }
    }

    /// Parse a `struct`/`union` specifier (the keyword at the cursor), returning
    /// the record type. A body `{ ... }` completes the record's definition.
    fn parse_record(&mut self, kind: RecordKind) -> PResult<CType> {
        self.bump(); // struct / union
        // `struct __attribute__((packed)) tag { ... }`: attributes may precede
        // the tag, and follow the closing brace; both apply to the record type.
        let mut attrs = self.parse_attributes()?;
        let tag = match self.peek().clone() {
            TokenKind::Ident(name) => {
                self.bump();
                Some(name)
            }
            _ => None,
        };
        attrs.merge(self.parse_attributes()?);
        let has_body = self.is_punct(Punct::LBrace);
        if tag.is_none() && !has_body {
            return self.err("expected a tag name or '{' after struct/union");
        }
        let id = match &tag {
            Some(t) => self.tag_record(t, kind),
            None => self.anon_record(kind),
        };
        if has_body {
            self.parse_record_body(id)?;
            attrs.merge(self.parse_attributes()?);
            let def = &mut self.records.defs[id];
            def.packed |= attrs.packed;
            def.transparent |= attrs.transparent_union;
            def.align = max_align(def.align, attrs.aligned);
        }
        Ok(CType::Record(id))
    }

    fn parse_record_body(&mut self, id: RecordId) -> PResult<()> {
        self.expect_punct(Punct::LBrace, "'{' to open struct/union body")?;
        let mut fields = Vec::new();
        while !self.is_punct(Punct::RBrace) && !self.at_eof() {
            // A stray `;` (GNU accepts an empty member declaration).
            if self.eat_punct(Punct::Semi) {
                continue;
            }
            // A `_Static_assert` member declaration (C11) declares nothing.
            if self.is_kw(Keyword::StaticAssert) {
                self.parse_static_assert()?;
                continue;
            }
            // `__extension__ union { ... };` (glibc) lifts the pedantic gates for
            // this member declaration.
            let ext = u32::from(self.skip_extension());
            self.extension += ext;
            let r = self.parse_member_decl(&mut fields);
            self.extension -= ext;
            r?;
        }
        self.expect_punct(Punct::RBrace, "'}' to close struct/union body")?;
        self.records.defs[id].fields = fields;
        self.records.defs[id].complete = true;
        Ok(())
    }

    /// Parse one member declaration of a struct/union body (through its `;`),
    /// appending the members it declares to `fields`.
    fn parse_member_decl(&mut self, fields: &mut Vec<Field>) -> PResult<()> {
        let base = self.parse_decl_specs()?;
        let sattrs = self.spec_attrs.clone();
        let align = max_align(self.last_alignas.take(), sattrs.aligned);
        // A member declaration with no declarator: an anonymous struct/union
        // member (an untagged `struct`/`union`, C11), or a nested tagged type
        // declaration (which declares no member).
        if self.is_punct(Punct::Semi) {
            if let CType::Record(rid) = &base
                && self.records.get(*rid).tag.is_none()
            {
                if !self.std.anonymous_members() && self.extension == 0 {
                    return self.err(
                        "anonymous struct/union members are a C11 feature (use -std=c11 or later)",
                    );
                }
                fields.push(Field {
                    name: String::new(),
                    ty: base.clone(),
                    anonymous: true,
                    align,
                    bit_width: None,
                });
            }
            self.bump(); // ';'
            return Ok(());
        }
        // An unnamed bit-field: `type : const-expr ;` (including `: 0`), which
        // has no declarator, only a colon and a width.
        if self.is_punct(Punct::Colon) {
            let bit_width = self.parse_bitfield_width(&base)?;
            self.skip_attributes()?;
            fields.push(Field {
                name: String::new(),
                ty: base.clone(),
                anonymous: false,
                align,
                bit_width: Some(bit_width),
            });
            self.expect_punct(Punct::Semi, "';' after bit-field")?;
            return Ok(());
        }
        loop {
            let (name, ty, name_span) = self.parse_named_declarator(base.clone())?;
            // Attributes may sit before the bit-field width and after it.
            let (_label, ty, mut attrs) = self.finish_declarator(ty, name_span, &sattrs)?;
            let bit_width = if self.is_punct(Punct::Colon) {
                let w = self.parse_bitfield_width(&ty)?;
                // A named bit-field of width 0 is a constraint violation.
                if w == 0 {
                    return Err(Diagnostic::error("named bit-field cannot have zero width")
                        .with_span(name_span));
                }
                attrs.merge(self.parse_attributes()?);
                Some(w)
            } else {
                None
            };
            let align = max_align(align, attrs.aligned);
            fields.push(Field { name, ty, anonymous: false, align, bit_width });
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        self.expect_punct(Punct::Semi, "';' after struct/union member")?;
        Ok(())
    }

    /// Parse the `: const-expr` of a bit-field declared with base type `ty` (the
    /// leading `:` is at the cursor). Validate that `ty` is an integer type and
    /// that the width is in `0..=bits(ty)` (where `_Bool` counts as one bit).
    fn parse_bitfield_width(&mut self, ty: &CType) -> PResult<u32> {
        let colon = self.expect_punct(Punct::Colon, "':' in bit-field")?;
        let ty = ty.unqual();
        if !ty.is_integer() {
            return Err(Diagnostic::error("bit-field has a non-integer type").with_span(colon));
        }
        if ty.int_width().is_some_and(|w| w > 64) {
            return Err(Diagnostic::error(format!("bit-fields of type '{ty}' are not supported"))
                .with_span(colon));
        }
        let width_span = self.peek_span();
        let value = self.parse_const_expr()?;
        if value < 0 {
            return Err(Diagnostic::error("bit-field width must be non-negative")
                .with_span(width_span));
        }
        // `_Bool` bit-fields hold a single value bit; other integer types allow up
        // to their full storage width.
        let max = match ty {
            CType::Bool => 1u32,
            _ => u32::from(ty.int_width().unwrap_or(32)),
        };
        if value as u128 > u128::from(max) {
            return Err(Diagnostic::error(format!(
                "width of bit-field ({value}) exceeds the width of its type ({max} bits)"
            ))
            .with_span(width_span));
        }
        Ok(value as u32)
    }

    /// Parse an `enum` specifier. An `enum` has type `int`; a body registers its
    /// enumerator constants (auto-incrementing, or explicit `= const-expr`).
    fn parse_enum(&mut self) -> PResult<CType> {
        self.bump(); // enum
        self.skip_attributes()?;
        let mut tag: Option<String> = None;
        if let TokenKind::Ident(name) = self.peek() {
            tag = Some(name.clone());
            self.bump(); // tag
        }
        self.skip_attributes()?;
        if self.eat_punct(Punct::LBrace) {
            let mut next = 0i128;
            let mut any_negative = false;
            while !self.is_punct(Punct::RBrace) && !self.at_eof() {
                let (name, _span) = self.expect_ident()?;
                // An enumerator may carry attributes (`A __attribute__((deprecated))`).
                self.skip_attributes()?;
                if self.eat_punct(Punct::Assign) {
                    next = self.parse_const_expr()?;
                }
                if next < 0 {
                    any_negative = true;
                }
                self.enum_map.insert(name.clone(), next);
                self.enum_consts.push((name, next));
                next += 1;
                if !self.eat_punct(Punct::Comma) {
                    break;
                }
            }
            self.expect_punct(Punct::RBrace, "'}' to close enum body")?;
            self.skip_attributes()?;
            // An enumerated type's underlying integer type, matching gcc: `int`
            // when any enumerator is negative, otherwise `unsigned int`. This is
            // observable in a `_Generic` selection and, crucially, in the sign of
            // an `enum`-typed bit-field (a 2-bit field holding value 3 must read
            // back as 3, not −1).
            let signed = any_negative;
            if let Some(t) = &tag {
                self.enum_tag_signed.insert(t.clone(), signed);
            }
            return Ok(CType::Int(IntTy::new(32, signed)));
        }
        // A tag-only reference (`enum foo x;`) without a visible body: recover the
        // underlying signedness from the tag's definition if it has been seen,
        // else fall back to plain `int`.
        let signed = tag.as_deref().and_then(|t| self.enum_tag_signed.get(t)).copied().unwrap_or(true);
        Ok(CType::Int(IntTy::new(32, signed)))
    }

    /// Parse the declarator list of a C23 `constexpr` declaration (`base` is the
    /// already-parsed base type). Each object must be an integer-category type
    /// with a constant-expression initializer; it is registered as a named
    /// compile-time constant (usable in constant expressions and, having no
    /// storage, not a modifiable lvalue) rather than emitted as an object.
    fn parse_constexpr_decls(&mut self, base: CType) -> PResult<()> {
        let (name, ty, span) = self.parse_named_declarator(base)?;
        if !ty.is_integer() {
            return Err(Diagnostic::error(format!(
                "`constexpr` object '{name}' has type `{ty}`; only integer-category \
                 `constexpr` objects are supported"
            ))
            .with_span(span));
        }
        if !self.eat_punct(Punct::Assign) {
            return Err(Diagnostic::error(format!(
                "`constexpr` object '{name}' requires a constant initializer"
            ))
            .with_span(span));
        }
        let init_span = self.peek_span();
        let init = self.parse_initializer()?;
        let Init::Expr(e) = init else {
            return Err(Diagnostic::error(
                "a `constexpr` integer object requires a scalar constant initializer",
            )
            .with_span(init_span));
        };
        let value = self.eval_const_expr(&e).ok_or_else(|| {
            Diagnostic::error(format!(
                "the initializer of `constexpr` object '{name}' is not a constant expression"
            ))
            .with_span(init_span)
        })?;
        let reduced = reduce_const_to_type(value, &ty);
        // Register for constant-expression evaluation in later declarators,
        // array sizes, `case` labels, `_Static_assert`, and sema.
        self.enum_map.insert(name.clone(), reduced);
        self.declare_ordinary(&name, Some(ty.clone()));
        self.constexprs.push((name, reduced, ty));
        // C23 allows only a single declarator with `constexpr` (matching gcc).
        if self.is_punct(Punct::Comma) {
            return self.err("`constexpr` may only be used with a single declarator");
        }
        self.expect_punct(Punct::Semi, "';' after constexpr declaration")?;
        Ok(())
    }

    /// Parse a constant expression (a conditional-expression) and fold it to an
    /// integer, resolving enumerator constants and `sizeof`.
    fn parse_const_expr(&mut self) -> PResult<i128> {
        let span = self.peek_span();
        let e = self.parse_conditional()?;
        self.eval_const_expr(&e)
            .ok_or_else(|| Diagnostic::error("expected a constant integer expression").with_span(span))
    }

    /// Complete an incomplete array type (`T x[] = …`) from its initializer, so
    /// the parser's symbol table records the deduced length. A later `sizeof x`
    /// in a constant expression (typically another object's array bound, e.g.
    /// `char opts[sizeof switches / sizeof switches[0]]`) then measures the whole
    /// array rather than a zero-length one.
    fn deduce_array_symbol_type(&self, ty: &CType, init: &Init) -> CType {
        if let CType::Array(elem, 0) = ty {
            let n = match init {
                Init::List(items) => {
                    let mut pos: u64 = 0;
                    let mut max: u64 = 0;
                    for it in items {
                        if let Some(Designator::Index(k)) = it.designators.first()
                            && *k >= 0
                        {
                            pos = *k as u64;
                        }
                        pos += 1;
                        max = max.max(pos);
                    }
                    max
                }
                Init::Expr(e) => match &e.kind {
                    ExprKind::StrLit(bytes, kind) => bytes.len() as u64 / kind.elem_width() + 1,
                    _ => 0,
                },
            };
            if n > 0 {
                return CType::Array(elem.clone(), n);
            }
        }
        ty.clone()
    }

    /// Fold a parsed expression to a constant integer, or `None` if it is not a
    /// constant expression the parser can evaluate. The folding is typed (see
    /// [`consteval`]); the result is the value normalized to its type.
    fn eval_const_expr(&self, e: &Expr) -> Option<i128> {
        consteval::eval(e, self).map(|c| c.value)
    }

    /// The constant address of an lvalue reached from a constant pointer through
    /// member access and subscripts (`((T *) 0)->a.b[2]`), or `None`.
    fn const_lvalue_addr(&self, e: &Expr) -> Option<i128> {
        match &e.kind {
            ExprKind::Member(base, name, arrow) => {
                let (addr, rec) = if *arrow {
                    (self.eval_const_expr(base)?, self.expr_type(base)?.pointee()?.clone())
                } else {
                    (self.const_lvalue_addr(base)?, self.expr_type(base)?)
                };
                let CType::Record(id) = *rec.unqual() else { return None };
                let (off, _) = layout::resolve_member(&self.records, id, name)?;
                Some(addr + i128::from(off))
            }
            ExprKind::Index(base, idx) => {
                let i = self.eval_const_expr(idx)?;
                let (addr, elem) = match self.expr_type(base)? {
                    CType::Pointer(elem) => (self.eval_const_expr(base)?, *elem),
                    CType::Array(elem, _) => (self.const_lvalue_addr(base)?, *elem),
                    _ => return None,
                };
                Some(addr + i * i128::from(layout::stride_of(&self.records, &elem)))
            }
            ExprKind::Unary(UnaryOp::Deref, p) => self.eval_const_expr(p),
            _ => None,
        }
    }

    // --- statements --------------------------------------------------------

    fn parse_block_stmts(&mut self) -> PResult<Vec<Stmt>> {
        self.expect_punct(Punct::LBrace, "'{'")?;
        self.push_scope();
        let mut stmts = Vec::new();
        let mut seen_stmt = false;
        while !self.is_punct(Punct::RBrace) && !self.at_eof() {
            // Attribute specifier sequences may precede a declaration or statement.
            if let Err(e) = self.skip_attributes() {
                self.pop_scope();
                return Err(e);
            }
            if self.is_punct(Punct::RBrace) || self.at_eof() {
                break;
            }
            // A file-scope-style `_Static_assert` may also appear in a block.
            if self.is_kw(Keyword::StaticAssert) {
                self.parse_static_assert()?;
                continue;
            }
            // A label (`ident :`) is a statement even when its name is a
            // typedef-name, so it must not be mistaken for a declaration.
            let is_decl = (self.at_type_specifier() || self.extension_decl_ahead()) && !self.label_ahead();
            if is_decl && seen_stmt && !self.std.mixed_declarations() {
                self.pop_scope();
                return self.err(
                    "declarations after statements are a C99 feature (use -std=c99 or later)",
                );
            }
            if !is_decl {
                seen_stmt = true;
            }
            match self.parse_stmt() {
                Ok(s) => stmts.push(s),
                Err(e) => {
                    self.pop_scope();
                    return Err(e);
                }
            }
        }
        self.pop_scope();
        self.expect_punct(Punct::RBrace, "'}' to close block")?;
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> PResult<Stmt> {
        // An attribute specifier sequence may precede a statement (e.g.
        // `[[fallthrough]];`, `[[maybe_unused]] int x;`).
        self.skip_attributes()?;
        let start = self.peek_span();
        // `__extension__ long long x;` — a marked block-scope declaration.
        if self.extension_decl_ahead() {
            self.skip_extension();
            self.extension += 1;
            let r = self.parse_local_decl();
            self.extension -= 1;
            return r;
        }
        // A named label `ident :` (its own namespace; may shadow a typedef name).
        if self.label_ahead() {
            let TokenKind::Ident(name) = self.peek().clone() else { unreachable!() };
            self.bump(); // ident
            self.bump(); // ':'
            let body = Box::new(self.parse_labeled_body()?);
            let span = start.merge(body.span);
            return Ok(self.stmt(StmtKind::Label(name, body), span));
        }
        if self.is_punct(Punct::LBrace) {
            let stmts = self.parse_block_stmts()?;
            return Ok(self.stmt(StmtKind::Block(stmts), start));
        }
        if self.at_type_specifier() {
            return self.parse_local_decl();
        }
        match self.peek() {
            TokenKind::Keyword(Keyword::If) => self.parse_if(),
            TokenKind::Keyword(Keyword::While) => self.parse_while(),
            TokenKind::Keyword(Keyword::Do) => self.parse_do_while(),
            TokenKind::Keyword(Keyword::For) => self.parse_for(),
            TokenKind::Keyword(Keyword::Switch) => self.parse_switch(),
            TokenKind::Keyword(Keyword::Case) => {
                self.bump();
                let value = self.parse_const_expr()?;
                // GNU case range: `case lo ... hi:`.
                let high = if self.eat_punct(Punct::Ellipsis) {
                    let sp = self.peek_span();
                    let hi = self.parse_const_expr()?;
                    if hi < value {
                        return Err(Diagnostic::error("empty case range").with_span(sp));
                    }
                    if hi - value >= 65536 {
                        return Err(Diagnostic::error(
                            "case ranges spanning more than 65536 values are not supported",
                        )
                        .with_span(sp));
                    }
                    Some(hi)
                } else {
                    None
                };
                self.expect_punct(Punct::Colon, "':' after case label")?;
                let body = Box::new(self.parse_labeled_body()?);
                let span = start.merge(body.span);
                Ok(match high {
                    Some(hi) => self.stmt(StmtKind::CaseRange(value, hi, body), span),
                    None => self.stmt(StmtKind::Case(value, body), span),
                })
            }
            TokenKind::Keyword(Keyword::Default) => {
                self.bump();
                self.expect_punct(Punct::Colon, "':' after default label")?;
                let body = Box::new(self.parse_labeled_body()?);
                let span = start.merge(body.span);
                Ok(self.stmt(StmtKind::Default(body), span))
            }
            TokenKind::Keyword(Keyword::Asm) => {
                let stmt = self.parse_asm_stmt()?;
                let end = self.expect_punct(Punct::Semi, "';' after asm statement")?;
                Ok(self.stmt(StmtKind::Asm(stmt), start.merge(end)))
            }
            TokenKind::Keyword(Keyword::Goto) => {
                self.bump();
                // GNU computed goto: `goto *expr;`.
                if self.eat_punct(Punct::Star) {
                    let target = self.parse_expr()?;
                    let end = self.expect_punct(Punct::Semi, "';' after goto")?;
                    return Ok(self.stmt(StmtKind::GotoIndirect(target), start.merge(end)));
                }
                let (name, _) = self.expect_ident()?;
                let end = self.expect_punct(Punct::Semi, "';' after goto")?;
                Ok(self.stmt(StmtKind::Goto(name), start.merge(end)))
            }
            TokenKind::Keyword(Keyword::Return) => {
                self.bump();
                let value = if self.is_punct(Punct::Semi) { None } else { Some(self.parse_expr()?) };
                let end = self.expect_punct(Punct::Semi, "';' after return")?;
                Ok(self.stmt(StmtKind::Return(value), start.merge(end)))
            }
            TokenKind::Keyword(Keyword::Break) => {
                self.bump();
                let end = self.expect_punct(Punct::Semi, "';' after break")?;
                Ok(self.stmt(StmtKind::Break, start.merge(end)))
            }
            TokenKind::Keyword(Keyword::Continue) => {
                self.bump();
                let end = self.expect_punct(Punct::Semi, "';' after continue")?;
                Ok(self.stmt(StmtKind::Continue, start.merge(end)))
            }
            TokenKind::Punct(Punct::Semi) => {
                let end = self.bump().span;
                Ok(self.stmt(StmtKind::Expr(None), end))
            }
            _ => {
                let expr = self.parse_expr()?;
                let end = self.expect_punct(Punct::Semi, "';' after expression")?;
                Ok(self.stmt(StmtKind::Expr(Some(expr)), start.merge(end)))
            }
        }
    }

    fn parse_local_decl(&mut self) -> PResult<Stmt> {
        let start = self.peek_span();
        self.spec_inline = false;
        self.spec_thread = false;
        if self.eat_kw(Keyword::Typedef) {
            self.parse_typedef()?;
            return Ok(self.stmt(StmtKind::Expr(None), start));
        }
        let storage = self.consume_storage()?;
        if self.eat_kw(Keyword::Typedef) {
            self.parse_typedef()?;
            return Ok(self.stmt(StmtKind::Expr(None), start));
        }
        let base = self.parse_decl_specs()?;
        let sattrs = self.spec_attrs.clone();
        let align = self.last_alignas.take();
        let is_constexpr = self.last_constexpr;
        let storage = merge_storage(storage, self.consume_storage()?);
        // A C23 `constexpr` object is a named compile-time constant (no storage);
        // it contributes no executable statement.
        if is_constexpr {
            self.parse_constexpr_decls(base)?;
            return Ok(self.stmt(StmtKind::Expr(None), start));
        }
        // A bare `struct S { ... };` at block scope declares only a type.
        if self.is_punct(Punct::Semi) {
            let end = self.bump().span;
            return Ok(self.stmt(StmtKind::Expr(None), start.merge(end)));
        }
        let thread_local = self.spec_thread;
        let mut decls = Vec::new();
        loop {
            self.vla_ok = true;
            self.vla_len = None;
            let declarator = self.parse_named_declarator(base.clone());
            self.vla_ok = false;
            let (name, ty, name_span) = declarator?;
            let vla_len = self.vla_len.take().map(Box::new);
            if vla_len.is_some() && !matches!(ty, CType::Array(_, 0)) {
                return Err(Diagnostic::error(
                    "a variable-length array is only supported as the type of an object itself \
                     (not behind a pointer or function declarator)",
                )
                .with_span(name_span));
            }
            let (asm_label, ty, attrs) = self.finish_declarator(ty, name_span, &sattrs)?;
            self.declare_ordinary(&name, Some(ty.clone()));
            let init =
                if self.eat_punct(Punct::Assign) { Some(self.parse_initializer()?) } else { None };
            if let Some(i) = &init {
                // `static const T tbl[] = {...}; char a[sizeof tbl / sizeof tbl[0]];`
                self.declare_ordinary(&name, Some(self.deduce_array_symbol_type(&ty, i)));
            }
            let align = max_align(align, attrs.aligned);
            decls.push(VarDecl {
                name,
                ty,
                init,
                align,
                storage,
                asm_label,
                thread_local,
                attrs: attrs.sym(),
                vla_len,
                span: name_span,
            });
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        let end = self.expect_punct(Punct::Semi, "';' after declaration")?;
        Ok(self.stmt(StmtKind::Decl(decls), start.merge(end)))
    }

    /// Whether the cursor is at a named label: an identifier followed by `:`.
    fn label_ahead(&self) -> bool {
        matches!(self.peek(), TokenKind::Ident(_))
            && matches!(self.peek_at(1), TokenKind::Punct(Punct::Colon))
    }

    /// Parse the statement a label prefixes. A label directly before the block's
    /// closing `}` is treated as prefixing an empty statement.
    fn parse_labeled_body(&mut self) -> PResult<Stmt> {
        if self.is_punct(Punct::RBrace) {
            return Ok(self.stmt(StmtKind::Expr(None), self.peek_span()));
        }
        self.parse_stmt()
    }

    fn parse_switch(&mut self) -> PResult<Stmt> {
        let start = self.peek_span();
        self.bump(); // switch
        self.expect_punct(Punct::LParen, "'(' after switch")?;
        let cond = self.parse_expr()?;
        self.expect_punct(Punct::RParen, "')' after switch condition")?;
        let body = Box::new(self.parse_stmt()?);
        let span = start.merge(body.span);
        Ok(self.stmt(StmtKind::Switch(cond, body), span))
    }

    fn parse_if(&mut self) -> PResult<Stmt> {
        let start = self.peek_span();
        self.bump(); // if
        self.expect_punct(Punct::LParen, "'(' after if")?;
        let cond = self.parse_expr()?;
        self.expect_punct(Punct::RParen, "')' after if condition")?;
        let then = Box::new(self.parse_stmt()?);
        let els = if self.eat_kw(Keyword::Else) { Some(Box::new(self.parse_stmt()?)) } else { None };
        let end = els.as_ref().map(|s| s.span).unwrap_or(then.span);
        Ok(self.stmt(StmtKind::If(cond, then, els), start.merge(end)))
    }

    fn parse_while(&mut self) -> PResult<Stmt> {
        let start = self.peek_span();
        self.bump(); // while
        self.expect_punct(Punct::LParen, "'(' after while")?;
        let cond = self.parse_expr()?;
        self.expect_punct(Punct::RParen, "')' after while condition")?;
        let body = Box::new(self.parse_stmt()?);
        let span = start.merge(body.span);
        Ok(self.stmt(StmtKind::While(cond, body), span))
    }

    fn parse_do_while(&mut self) -> PResult<Stmt> {
        let start = self.peek_span();
        self.bump(); // do
        let body = Box::new(self.parse_stmt()?);
        if !self.eat_kw(Keyword::While) {
            return self.err("expected 'while' after do-body");
        }
        self.expect_punct(Punct::LParen, "'(' after do-while")?;
        let cond = self.parse_expr()?;
        self.expect_punct(Punct::RParen, "')' after do-while condition")?;
        let end = self.expect_punct(Punct::Semi, "';' after do-while")?;
        Ok(self.stmt(StmtKind::DoWhile(body, cond), start.merge(end)))
    }

    fn parse_for(&mut self) -> PResult<Stmt> {
        let start = self.peek_span();
        self.bump(); // for
        self.expect_punct(Punct::LParen, "'(' after for")?;

        // init clause: a declaration, an expression, or empty.
        let init: Option<Box<Stmt>> = if self.is_punct(Punct::Semi) {
            self.bump();
            None
        } else if self.at_type_specifier() {
            if !self.std.for_loop_decls() {
                return self
                    .err("a declaration in `for` is a C99 feature (use -std=c99 or later)");
            }
            Some(Box::new(self.parse_local_decl()?))
        } else {
            let sp = self.peek_span();
            let e = self.parse_expr()?;
            let end = self.expect_punct(Punct::Semi, "';' after for-init")?;
            Some(Box::new(self.stmt(StmtKind::Expr(Some(e)), sp.merge(end))))
        };

        let cond = if self.is_punct(Punct::Semi) { None } else { Some(self.parse_expr()?) };
        self.expect_punct(Punct::Semi, "';' after for-condition")?;

        let step = if self.is_punct(Punct::RParen) { None } else { Some(self.parse_expr()?) };
        self.expect_punct(Punct::RParen, "')' after for-clauses")?;

        let body = Box::new(self.parse_stmt()?);
        let span = start.merge(body.span);
        Ok(self.stmt(StmtKind::For(init, cond, step, body), span))
    }

    fn stmt(&self, kind: StmtKind, span: Span) -> Stmt {
        Stmt { kind, span }
    }

    // --- expressions -------------------------------------------------------

    fn parse_expr(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_assign()?;
        while self.is_punct(Punct::Comma) {
            self.bump();
            let rhs = self.parse_assign()?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr { kind: ExprKind::Comma(Box::new(lhs), Box::new(rhs)), span };
        }
        Ok(lhs)
    }

    fn parse_assign(&mut self) -> PResult<Expr> {
        let lhs = self.parse_conditional()?;
        let op = match self.peek() {
            TokenKind::Punct(Punct::Assign) => Some(None),
            TokenKind::Punct(Punct::PlusEq) => Some(Some(BinaryOp::Add)),
            TokenKind::Punct(Punct::MinusEq) => Some(Some(BinaryOp::Sub)),
            TokenKind::Punct(Punct::StarEq) => Some(Some(BinaryOp::Mul)),
            TokenKind::Punct(Punct::SlashEq) => Some(Some(BinaryOp::Div)),
            TokenKind::Punct(Punct::PercentEq) => Some(Some(BinaryOp::Rem)),
            TokenKind::Punct(Punct::AmpEq) => Some(Some(BinaryOp::BitAnd)),
            TokenKind::Punct(Punct::PipeEq) => Some(Some(BinaryOp::BitOr)),
            TokenKind::Punct(Punct::CaretEq) => Some(Some(BinaryOp::BitXor)),
            TokenKind::Punct(Punct::ShlEq) => Some(Some(BinaryOp::Shl)),
            TokenKind::Punct(Punct::ShrEq) => Some(Some(BinaryOp::Shr)),
            _ => None,
        };
        match op {
            Some(compound) => {
                self.bump();
                let rhs = self.parse_assign()?; // right-associative
                let span = lhs.span.merge(rhs.span);
                Ok(Expr { kind: ExprKind::Assign(compound, Box::new(lhs), Box::new(rhs)), span })
            }
            None => Ok(lhs),
        }
    }

    fn parse_conditional(&mut self) -> PResult<Expr> {
        let cond = self.parse_binary(0)?;
        if self.eat_punct(Punct::Question) {
            let then = self.parse_expr()?;
            self.expect_punct(Punct::Colon, "':' in conditional expression")?;
            let els = self.parse_assign()?;
            let span = cond.span.merge(els.span);
            return Ok(Expr {
                kind: ExprKind::Cond(Box::new(cond), Box::new(then), Box::new(els)),
                span,
            });
        }
        Ok(cond)
    }

    /// Precedence-climbing parse of the binary operators (levels 0..=9 below).
    fn parse_binary(&mut self, min_prec: u8) -> PResult<Expr> {
        let mut lhs = self.parse_cast()?;
        while let Some((op, prec)) = self.peek_binop() {
            if prec < min_prec {
                break;
            }
            self.bump();
            let rhs = self.parse_binary(prec + 1)?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr { kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)), span };
        }
        Ok(lhs)
    }

    fn peek_binop(&self) -> Option<(BinaryOp, u8)> {
        let TokenKind::Punct(p) = self.peek() else {
            return None;
        };
        Some(match p {
            Punct::PipePipe => (BinaryOp::LOr, 0),
            Punct::AmpAmp => (BinaryOp::LAnd, 1),
            Punct::Pipe => (BinaryOp::BitOr, 2),
            Punct::Caret => (BinaryOp::BitXor, 3),
            Punct::Amp => (BinaryOp::BitAnd, 4),
            Punct::EqEq => (BinaryOp::Eq, 5),
            Punct::Ne => (BinaryOp::Ne, 5),
            Punct::Lt => (BinaryOp::Lt, 6),
            Punct::Le => (BinaryOp::Le, 6),
            Punct::Gt => (BinaryOp::Gt, 6),
            Punct::Ge => (BinaryOp::Ge, 6),
            Punct::Shl => (BinaryOp::Shl, 7),
            Punct::Shr => (BinaryOp::Shr, 7),
            Punct::Plus => (BinaryOp::Add, 8),
            Punct::Minus => (BinaryOp::Sub, 8),
            Punct::Star => (BinaryOp::Mul, 9),
            Punct::Slash => (BinaryOp::Div, 9),
            Punct::Percent => (BinaryOp::Rem, 9),
            _ => return None,
        })
    }

    /// A cast `(type-name) cast-expression`, or a unary expression.
    fn parse_cast(&mut self) -> PResult<Expr> {
        if self.is_punct(Punct::LParen) && self.type_name_follows_lparen() {
            let start = self.peek_span();
            self.bump(); // (
            let ty = self.parse_type_name()?;
            self.expect_punct(Punct::RParen, "')' to close cast")?;
            // `(type-name){ init }` is a compound literal (an lvalue), not a cast.
            if self.is_punct(Punct::LBrace) {
                if !self.std.compound_literals() {
                    return self.err(
                        "compound literals are a C99 feature (use -std=c99 or later)",
                    );
                }
                let init = self.parse_init_list()?;
                let span = start.merge(self.peek_span());
                let lit =
                    Expr { kind: ExprKind::CompoundLiteral(ty, Box::new(init)), span };
                // A compound literal is a postfix-expression: `(int[]){1,2}[i]`.
                return self.parse_postfix_tail(lit);
            }
            let inner = self.parse_cast()?;
            let span = start.merge(inner.span);
            return Ok(Expr { kind: ExprKind::Cast(ty, Box::new(inner)), span });
        }
        self.parse_unary()
    }

    /// Whether a `(` at the cursor is followed by a type-name (so this is a cast
    /// or a `sizeof(type)` rather than a parenthesized expression).
    fn type_name_follows_lparen(&self) -> bool {
        match self.peek_at(1) {
            TokenKind::Keyword(
                Keyword::Void
                | Keyword::Bool
                | Keyword::Char
                | Keyword::Short
                | Keyword::Int
                | Keyword::Long
                | Keyword::Float
                | Keyword::Double
                | Keyword::Signed
                | Keyword::Unsigned
                | Keyword::Const
                | Keyword::Volatile
                | Keyword::Atomic
                | Keyword::Restrict
                | Keyword::Typeof
                | Keyword::TypeofUnqual
                | Keyword::Struct
                | Keyword::Union
                | Keyword::Enum,
            ) => true,
            // `sizeof(_BitInt(N))` / a `(_BitInt(N))` cast (C23).
            TokenKind::Ident(name) => is_builtin_type_ident(name) || self.is_typedef_name(name),
            _ => false,
        }
    }

    fn parse_unary(&mut self) -> PResult<Expr> {
        let start = self.peek_span();
        // GNU `__extension__` as a unary prefix: `__extension__ ({ ... })`.
        if self.at_extension() {
            self.skip_extension();
            self.extension += 1;
            let r = self.parse_cast();
            self.extension -= 1;
            return r;
        }
        // GNU labels as values: `&&label`.
        if self.is_punct(Punct::AmpAmp)
            && let TokenKind::Ident(name) = self.peek_at(1).clone()
        {
            self.bump();
            let end = self.bump().span;
            return Ok(Expr { kind: ExprKind::LabelAddr(name), span: start.merge(end) });
        }
        if let TokenKind::Punct(p) = self.peek() {
            let unop = match p {
                Punct::Minus => Some(UnaryOp::Neg),
                Punct::Plus => Some(UnaryOp::Plus),
                Punct::Bang => Some(UnaryOp::LNot),
                Punct::Tilde => Some(UnaryOp::BitNot),
                Punct::Star => Some(UnaryOp::Deref),
                Punct::Amp => Some(UnaryOp::AddrOf),
                _ => None,
            };
            if let Some(op) = unop {
                self.bump();
                let inner = self.parse_cast()?;
                let span = start.merge(inner.span);
                return Ok(Expr { kind: ExprKind::Unary(op, Box::new(inner)), span });
            }
            if *p == Punct::PlusPlus || *p == Punct::MinusMinus {
                let is_inc = *p == Punct::PlusPlus;
                self.bump();
                let inner = self.parse_unary()?;
                let span = start.merge(inner.span);
                let kind =
                    if is_inc { ExprKind::PreInc(Box::new(inner)) } else { ExprKind::PreDec(Box::new(inner)) };
                return Ok(Expr { kind, span });
            }
        }
        if self.is_kw(Keyword::Sizeof) {
            return self.parse_sizeof();
        }
        if self.is_kw(Keyword::Alignof)
            || matches!(self.peek(), TokenKind::Ident(n) if n == "__alignof__" || n == "__alignof")
        {
            return self.parse_alignof();
        }
        self.parse_postfix()
    }

    fn parse_sizeof(&mut self) -> PResult<Expr> {
        let start = self.peek_span();
        self.bump(); // sizeof
        if self.is_punct(Punct::LParen) && self.type_name_follows_lparen() {
            self.bump(); // (
            let ty = self.parse_type_name()?;
            let end = self.expect_punct(Punct::RParen, "')' after sizeof type")?;
            return Ok(Expr { kind: ExprKind::SizeofType(ty), span: start.merge(end) });
        }
        let inner = self.parse_unary()?;
        let span = start.merge(inner.span);
        Ok(Expr { kind: ExprKind::SizeofExpr(Box::new(inner)), span })
    }

    /// `_Alignof ( type-name )` / `alignof ( type-name )`: a `size_t` constant
    /// equal to the type's alignment.
    fn parse_alignof(&mut self) -> PResult<Expr> {
        let start = self.peek_span();
        self.bump(); // _Alignof / alignof / __alignof__
        // GNU also takes an expression operand: `__alignof__ (x)`, `__alignof__ x`.
        if !(self.is_punct(Punct::LParen) && self.type_name_follows_lparen()) {
            let e = self.parse_unary()?;
            let Some(ty) = self.expr_type(&e) else {
                return Err(Diagnostic::error("cannot determine the type of this alignof operand")
                    .with_span(e.span));
            };
            let span = start.merge(e.span);
            return Ok(Expr { kind: ExprKind::AlignofType(ty), span });
        }
        self.expect_punct(Punct::LParen, "'(' after _Alignof")?;
        let ty = self.parse_type_name()?;
        let end = self.expect_punct(Punct::RParen, "')' after _Alignof type")?;
        Ok(Expr { kind: ExprKind::AlignofType(ty), span: start.merge(end) })
    }

    /// `_Generic ( controlling-expr , assoc-list )` (C11): a generic selection.
    /// Each association is `type-name : assign-expr` or `default : assign-expr`.
    fn parse_generic(&mut self) -> PResult<Expr> {
        let start = self.peek_span();
        self.bump(); // _Generic
        self.expect_punct(Punct::LParen, "'(' after _Generic")?;
        let controlling = Box::new(self.parse_assign()?);
        self.expect_punct(Punct::Comma, "',' after the _Generic controlling expression")?;
        let mut assocs = Vec::new();
        loop {
            if self.eat_kw(Keyword::Default) {
                self.expect_punct(Punct::Colon, "':' after 'default'")?;
                let expr = self.parse_assign()?;
                assocs.push(GenericAssoc { ty: None, expr });
            } else {
                let ty = self.parse_type_name()?;
                self.expect_punct(Punct::Colon, "':' after a _Generic association type")?;
                let expr = self.parse_assign()?;
                assocs.push(GenericAssoc { ty: Some(ty), expr });
            }
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        let end = self.expect_punct(Punct::RParen, "')' to close _Generic")?;
        Ok(Expr { kind: ExprKind::Generic(controlling, assocs), span: start.merge(end) })
    }

    /// Parse a `__builtin_va_*` primary. The current token is the builtin's
    /// identifier (already known to be followed by `(`). `va_arg`'s second
    /// operand is a type-name; every other builtin takes ordinary expressions.
    fn parse_va_builtin(&mut self, name: &str) -> PResult<Expr> {
        let start = self.bump().span; // the builtin identifier
        self.expect_punct(Punct::LParen, "'(' after a va_* builtin")?;
        let kind = match name {
            "__builtin_va_start" => {
                let ap = self.parse_assign()?;
                self.expect_punct(Punct::Comma, "',' in __builtin_va_start")?;
                let last = self.parse_assign()?;
                ExprKind::VaStart(Box::new(ap), Box::new(last))
            }
            "__builtin_va_arg" => {
                let ap = self.parse_assign()?;
                self.expect_punct(Punct::Comma, "',' in __builtin_va_arg")?;
                let ty = self.parse_type_name()?;
                ExprKind::VaArg(Box::new(ap), ty)
            }
            "__builtin_va_end" => {
                let ap = self.parse_assign()?;
                ExprKind::VaEnd(Box::new(ap))
            }
            // "__builtin_va_copy"
            _ => {
                let dst = self.parse_assign()?;
                self.expect_punct(Punct::Comma, "',' in __builtin_va_copy")?;
                let src = self.parse_assign()?;
                ExprKind::VaCopy(Box::new(dst), Box::new(src))
            }
        };
        let end = self.expect_punct(Punct::RParen, "')' to close a va_* builtin")?;
        Ok(Expr { kind, span: start.merge(end) })
    }

    /// `__builtin_offsetof ( type-name , member-designator )` (the identifier at
    /// the cursor): folded to its `size_t` byte offset. The designator is a
    /// member name followed by any mix of `.member` and `[const-expr]`.
    fn parse_builtin_offsetof(&mut self) -> PResult<Expr> {
        let start = self.bump().span; // __builtin_offsetof
        self.expect_punct(Punct::LParen, "'(' after __builtin_offsetof")?;
        let mut ty = self.parse_type_name()?;
        self.expect_punct(Punct::Comma, "',' in __builtin_offsetof")?;
        let mut offset = 0u64;
        let mut member = Some(self.expect_ident()?);
        loop {
            if let Some((name, sp)) = member.take() {
                let CType::Record(id) = *ty.unqual() else {
                    return Err(Diagnostic::error("__builtin_offsetof of a member of a non-record type")
                        .with_span(sp));
                };
                let Some((off, fty)) = layout::resolve_member(&self.records, id, &name) else {
                    return Err(Diagnostic::error(format!("no member named '{name}'")).with_span(sp));
                };
                offset += off;
                ty = fty;
            }
            if self.eat_punct(Punct::Dot) {
                member = Some(self.expect_ident()?);
            } else if self.is_punct(Punct::LBracket) {
                let sp = self.bump().span;
                let idx = self.parse_const_expr()?;
                self.expect_punct(Punct::RBracket, "']' in __builtin_offsetof")?;
                let CType::Array(elem, _) = ty.unqual().clone() else {
                    return Err(Diagnostic::error("subscript of a non-array in __builtin_offsetof")
                        .with_span(sp));
                };
                offset = offset.wrapping_add((idx as u64).wrapping_mul(layout::stride_of(&self.records, &elem)));
                ty = *elem;
            } else {
                break;
            }
        }
        let end = self.expect_punct(Punct::RParen, "')' to close __builtin_offsetof")?;
        Ok(Expr { kind: ExprKind::IntLit(i128::from(offset), size_t_ty()), span: start.merge(end) })
    }

    /// `__builtin_types_compatible_p ( type-name , type-name )`: the `int`
    /// constant 1 when the two types are compatible (qualifiers are not
    /// modelled, so this is type identity), else 0.
    fn parse_builtin_types_compatible(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        self.expect_punct(Punct::LParen, "'(' after __builtin_types_compatible_p")?;
        let a = self.parse_type_name()?;
        self.expect_punct(Punct::Comma, "',' in __builtin_types_compatible_p")?;
        let b = self.parse_type_name()?;
        let end = self.expect_punct(Punct::RParen, "')' to close __builtin_types_compatible_p")?;
        // Top-level qualifiers do not affect compatibility (gcc ignores them).
        let same = a.unqual() == b.unqual();
        Ok(Expr { kind: ExprKind::IntLit(i128::from(same), CType::int()), span: start.merge(end) })
    }

    /// `__builtin_choose_expr ( const-expr , e1 , e2 )`: selects `e1` when the
    /// constant is nonzero, else `e2`, at parse time (the other is discarded).
    fn parse_builtin_choose_expr(&mut self) -> PResult<Expr> {
        self.bump();
        self.expect_punct(Punct::LParen, "'(' after __builtin_choose_expr")?;
        let c = self.parse_const_expr()?;
        self.expect_punct(Punct::Comma, "',' in __builtin_choose_expr")?;
        let a = self.parse_assign()?;
        self.expect_punct(Punct::Comma, "',' in __builtin_choose_expr")?;
        let b = self.parse_assign()?;
        self.expect_punct(Punct::RParen, "')' to close __builtin_choose_expr")?;
        Ok(if c != 0 { a } else { b })
    }

    /// `__builtin_convertvector ( expr , type-name )`: the vector `expr`
    /// converted element-wise to the vector type `type-name`.
    fn parse_builtin_convertvector(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        self.expect_punct(Punct::LParen, "'(' after __builtin_convertvector")?;
        let e = self.parse_assign()?;
        self.expect_punct(Punct::Comma, "',' in __builtin_convertvector")?;
        let ty = self.parse_type_name()?;
        let end = self.expect_punct(Punct::RParen, "')' to close __builtin_convertvector")?;
        Ok(Expr { kind: ExprKind::ConvertVector(Box::new(e), ty), span: start.merge(end) })
    }

    fn parse_postfix(&mut self) -> PResult<Expr> {
        let expr = self.parse_primary()?;
        self.parse_postfix_tail(expr)
    }

    /// Apply postfix operators (`()`, `[]`, `.`/`->`, `++`/`--`) to an already
    /// parsed primary or compound-literal head.
    fn parse_postfix_tail(&mut self, expr: Expr) -> PResult<Expr> {
        let mut expr = expr;
        loop {
            if self.is_punct(Punct::LParen) {
                self.bump();
                let mut args = Vec::new();
                if !self.is_punct(Punct::RParen) {
                    loop {
                        args.push(self.parse_assign()?);
                        if !self.eat_punct(Punct::Comma) {
                            break;
                        }
                    }
                }
                let end = self.expect_punct(Punct::RParen, "')' to close call")?;
                let span = expr.span.merge(end);
                expr = Expr { kind: ExprKind::Call(Box::new(expr), args), span };
            } else if self.is_punct(Punct::LBracket) {
                self.bump();
                let index = self.parse_expr()?;
                let end = self.expect_punct(Punct::RBracket, "']' to close subscript")?;
                let span = expr.span.merge(end);
                expr = Expr { kind: ExprKind::Index(Box::new(expr), Box::new(index)), span };
            } else if self.is_punct(Punct::Dot) || self.is_punct(Punct::Arrow) {
                let arrow = self.is_punct(Punct::Arrow);
                self.bump();
                let (name, end) = self.expect_ident()?;
                let span = expr.span.merge(end);
                expr = Expr { kind: ExprKind::Member(Box::new(expr), name, arrow), span };
            } else if self.is_punct(Punct::PlusPlus) || self.is_punct(Punct::MinusMinus) {
                let is_inc = self.is_punct(Punct::PlusPlus);
                let end = self.bump().span;
                let span = expr.span.merge(end);
                let kind = if is_inc {
                    ExprKind::PostInc(Box::new(expr))
                } else {
                    ExprKind::PostDec(Box::new(expr))
                };
                expr = Expr { kind, span };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> PResult<Expr> {
        match self.peek().clone() {
            TokenKind::IntLit(value, ty) => {
                let span = self.bump().span;
                Ok(Expr { kind: ExprKind::IntLit(value, ty), span })
            }
            TokenKind::FloatLit(value, ty) => {
                let span = self.bump().span;
                Ok(Expr { kind: ExprKind::FloatLit(value, ty), span })
            }
            TokenKind::Ident(name) => {
                // The variadic compiler builtins look like calls but `va_arg` takes
                // a type-name argument, so they are parsed as dedicated primaries.
                if is_va_builtin(&name)
                    && matches!(self.peek_at(1), TokenKind::Punct(Punct::LParen))
                {
                    return self.parse_va_builtin(&name);
                }
                if matches!(self.peek_at(1), TokenKind::Punct(Punct::LParen)) {
                    match name.as_str() {
                        "__builtin_offsetof" => return self.parse_builtin_offsetof(),
                        "__builtin_types_compatible_p" => {
                            return self.parse_builtin_types_compatible();
                        }
                        "__builtin_choose_expr" => return self.parse_builtin_choose_expr(),
                        "__builtin_convertvector" => return self.parse_builtin_convertvector(),
                        _ => {}
                    }
                }
                // `__func__` (C99) and its GNU spellings name the enclosing
                // function as a string literal (outside a function: "").
                if matches!(name.as_str(), "__func__" | "__FUNCTION__" | "__PRETTY_FUNCTION__")
                    && self.var_type(&name).is_none()
                {
                    let span = self.bump().span;
                    let bytes = self.cur_func.clone().unwrap_or_default().into_bytes();
                    return Ok(Expr { kind: ExprKind::StrLit(bytes, StrKind::Narrow), span });
                }
                let span = self.bump().span;
                Ok(Expr { kind: ExprKind::Ident(name), span })
            }
            // A GNU statement expression `({ ... })`.
            TokenKind::Punct(Punct::LParen)
                if matches!(self.peek_at(1), TokenKind::Punct(Punct::LBrace)) =>
            {
                let start = self.bump().span; // (
                let stmts = self.parse_block_stmts()?;
                let end = self.expect_punct(Punct::RParen, "')' to close a statement expression")?;
                Ok(Expr { kind: ExprKind::StmtExpr(stmts), span: start.merge(end) })
            }
            TokenKind::Punct(Punct::LParen) => {
                self.bump();
                let inner = self.parse_expr()?;
                self.expect_punct(Punct::RParen, "')' to close parenthesized expression")?;
                Ok(inner)
            }
            TokenKind::Str(..) => {
                // Adjacent string literals concatenate into one literal. Element
                // values (not raw bytes) are concatenated so a narrow literal
                // adjacent to a wide one is re-encoded at the wider element width.
                let mut span = self.peek_span();
                let mut kind = StrKind::Narrow;
                let mut pieces: Vec<(Vec<u8>, StrKind)> = Vec::new();
                let mut first = true;
                while let TokenKind::Str(s, k) = self.peek().clone() {
                    kind = if first { k } else { kind.concat(k) };
                    first = false;
                    pieces.push((s, k));
                    span = span.merge(self.bump().span);
                }
                let out_width = kind.elem_width();
                let mut bytes = Vec::new();
                for (piece, pk) in pieces {
                    let pw = pk.elem_width() as usize;
                    if pw == out_width as usize {
                        bytes.extend_from_slice(&piece);
                    } else {
                        // Widen a narrower piece element-by-element (zero-extend).
                        for chunk in piece.chunks(pw) {
                            let mut v = [0u8; 8];
                            v[..chunk.len()].copy_from_slice(chunk);
                            let elem = u64::from_le_bytes(v);
                            bytes.extend_from_slice(&elem.to_le_bytes()[..out_width as usize]);
                        }
                    }
                }
                Ok(Expr { kind: ExprKind::StrLit(bytes, kind), span })
            }
            TokenKind::Keyword(Keyword::Generic) => self.parse_generic(),
            _ => self.err("expected an expression"),
        }
    }

    // --- initializers ------------------------------------------------------

    /// Parse an initializer: either a brace-enclosed list or an assignment
    /// expression.
    fn parse_initializer(&mut self) -> PResult<Init> {
        if self.is_punct(Punct::LBrace) {
            self.parse_init_list()
        } else {
            Ok(Init::Expr(self.parse_assign()?))
        }
    }

    fn parse_init_list(&mut self) -> PResult<Init> {
        self.expect_punct(Punct::LBrace, "'{' to open initializer list")?;
        let mut items = Vec::new();
        while !self.is_punct(Punct::RBrace) && !self.at_eof() {
            let designators = self.parse_designators()?;
            let init = self.parse_initializer()?;
            items.push(InitItem { designators, init });
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        self.expect_punct(Punct::RBrace, "'}' to close initializer list")?;
        Ok(Init::List(items))
    }

    /// Parse an optional designator chain (`.field` / `[index]` ... `=`).
    fn parse_designators(&mut self) -> PResult<Vec<Designator>> {
        if !self.is_punct(Punct::Dot) && !self.is_punct(Punct::LBracket) {
            return Ok(Vec::new());
        }
        if !self.std.for_loop_decls() {
            // `for_loop_decls` tracks C99; designated initializers are also C99.
            return self
                .err("designated initializers are a C99 feature (use -std=c99 or later)");
        }
        let mut chain = Vec::new();
        loop {
            if self.eat_punct(Punct::Dot) {
                let (name, _span) = self.expect_ident()?;
                chain.push(Designator::Field(name));
            } else if self.eat_punct(Punct::LBracket) {
                let idx = self.parse_const_expr()?;
                self.expect_punct(Punct::RBracket, "']' after array designator")?;
                chain.push(Designator::Index(idx));
            } else {
                break;
            }
        }
        self.expect_punct(Punct::Assign, "'=' after designator")?;
        Ok(chain)
    }
}

/// `size_t` for this target: `unsigned long` (64-bit).
fn size_t_ty() -> CType {
    CType::Int(IntTy::new(64, false))
}

/// The integer promotion of a type (for `typeof` typing): `_Bool`/`char`/`short`
/// become `int`; other types (including `_BitInt`, which is not promoted) are
/// unchanged.
fn promote_ty(ty: &CType) -> CType {
    match ty {
        CType::Bool => CType::int(),
        CType::Int(i) if i.bitint.is_none() && i.width < 32 => CType::int(),
        other => other.clone(),
    }
}

/// The usual arithmetic conversions on two arithmetic types (for `typeof`).
fn usual_arith_ty(a: &CType, b: &CType) -> CType {
    if a.is_float() || b.is_float() {
        let has_double =
            a.float_ty() == Some(FloatTy::F64) || b.float_ty() == Some(FloatTy::F64);
        return if has_double { CType::double() } else { CType::float() };
    }
    let a = promote_ty(a);
    let b = promote_ty(b);
    if a == b {
        return a;
    }
    let (wa, sa) = (a.int_width().unwrap_or(32), a.is_signed());
    let (wb, sb) = (b.int_width().unwrap_or(32), b.is_signed());
    if sa == sb {
        return if wa >= wb { a } else { b };
    }
    let (unsigned_t, uw, signed_t, sw) =
        if !sa { (a.clone(), wa, b.clone(), wb) } else { (b.clone(), wb, a.clone(), wa) };
    if uw >= sw { unsigned_t } else { signed_t }
}

/// The type of a binary expression, for `typeof` typing.
fn binary_type(op: BinaryOp, lt: &CType, rt: &CType) -> Option<CType> {
    match op {
        BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        | BinaryOp::LAnd | BinaryOp::LOr => Some(CType::int()),
        BinaryOp::Shl | BinaryOp::Shr => Some(promote_ty(lt)),
        BinaryOp::Add | BinaryOp::Sub => {
            if lt.is_pointer() {
                if rt.is_pointer() { Some(CType::long()) } else { Some(lt.clone()) }
            } else if rt.is_pointer() {
                Some(rt.clone())
            } else {
                Some(usual_arith_ty(lt, rt))
            }
        }
        _ => Some(usual_arith_ty(lt, rt)),
    }
}

/// The element type reached by dereferencing/indexing a pointer or array type.
fn deref_target(ty: &CType) -> Option<CType> {
    match ty {
        CType::Pointer(p) => Some((**p).clone()),
        CType::Array(e, _) => Some((**e).clone()),
        _ => None,
    }
}

/// Combine two storage-class results from a declaration (specifiers may appear
/// on either side of the type). A concrete class (`extern`/`static`) overrides
/// [`Storage::None`]; if both are concrete the later one wins (a genuine
/// conflict like `extern static` is ill-formed but not diagnosed here).
fn merge_storage(a: Storage, b: Storage) -> Storage {
    match (a, b) {
        (s, Storage::None) => s,
        (_, s) => s,
    }
}

/// Whether `name` is a builtin type name that the lexer leaves as an identifier
/// (so it can serve as a type specifier, or begin a type-name in a cast).
fn is_builtin_type_ident(name: &str) -> bool {
    matches!(
        name,
        "_BitInt"
            | "__int128"
            | "__int128_t"
            | "__uint128_t"
            | "_Float16"
            | "_Float32"
            | "_Float64"
            | "_Float128"
            | "_Float32x"
            | "_Float64x"
            | "_Float128x"
            | "__float128"
            | "__float80"
            | "__ibm128"
            | "_Complex"
            | "__complex__"
            | "__builtin_va_list"
            | "__typeof__"
            | "__typeof"
    )
}

/// `name` without a GNU `__...__` wrapping (`__aligned__` → `aligned`), the
/// reserved-namespace spelling of attribute names and machine modes.
fn strip_dunder(name: &str) -> &str {
    name.strip_prefix("__").and_then(|n| n.strip_suffix("__")).unwrap_or(name)
}

/// Whether `name` is one of the variadic compiler builtins the frontend expands
/// (the `<stdarg.h>` `va_*` macros map onto these).
fn is_va_builtin(name: &str) -> bool {
    matches!(
        name,
        "__builtin_va_start" | "__builtin_va_arg" | "__builtin_va_end" | "__builtin_va_copy"
    )
}
