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
    BinaryOp, CType, Designator, Expr, ExprKind, Field, FloatTy, FuncDef, FuncProto, FuncType,
    GenericAssoc, Init, InitItem, IntTy, Param, RecordDef, RecordId, RecordKind, Records, Stmt,
    StmtKind, Storage, TopLevel, TranslationUnit, UnaryOp, VarDecl,
};
use crate::cstd::CStd;
use crate::layout;
use crate::lex::{Keyword, Punct, Token, TokenKind};

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
        enum_consts: Vec::new(),
        constexprs: Vec::new(),
        scopes: vec![HashMap::new()],
        last_alignas: None,
        last_constexpr: false,
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
        });
        self.tags.insert(tag.to_owned(), id);
        id
    }

    fn anon_record(&mut self, kind: RecordKind) -> RecordId {
        let id = self.records.defs.len();
        self.records.defs.push(RecordDef { kind, tag: None, fields: Vec::new(), complete: false });
        id
    }

    // --- top level ---------------------------------------------------------

    fn parse_unit(&mut self) -> PResult<Vec<TopLevel>> {
        let mut items = Vec::new();
        while !self.at_eof() {
            // An attribute specifier sequence may precede a top-level declaration.
            self.skip_attributes()?;
            if self.at_eof() {
                break;
            }
            // A file-scope `_Static_assert` is a declaration with no external
            // effect in this subset; consume and drop it.
            if self.is_kw(Keyword::StaticAssert) {
                self.parse_static_assert()?;
                continue;
            }
            items.extend(self.parse_top_level()?);
        }
        Ok(items)
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
            match self.peek().clone() {
                TokenKind::Str(s) => {
                    self.bump();
                    message = Some(String::from_utf8_lossy(&s).into_owned());
                }
                _ => return self.err("expected a string message in _Static_assert"),
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
        if self.eat_kw(Keyword::Typedef) {
            return self.parse_typedef();
        }
        // Storage-class specifiers (extern/static) determine a file-scope object's
        // linkage; they may appear before and/or after the type specifiers.
        let storage = self.consume_storage();
        if self.eat_kw(Keyword::Typedef) {
            return self.parse_typedef();
        }
        let base = self.parse_decl_specs()?;
        let align = self.last_alignas.take();
        let is_constexpr = self.last_constexpr;
        let storage = merge_storage(storage, self.consume_storage());

        // A C23 `constexpr` object is a named compile-time constant (no storage).
        if is_constexpr {
            self.parse_constexpr_decls(base)?;
            return Ok(Vec::new());
        }

        // A bare `struct S { ... };` / `enum E { ... };` declares only a type.
        if self.eat_punct(Punct::Semi) {
            return Ok(Vec::new());
        }

        // First declarator: pointers, then a name.
        let ty0 = self.parse_pointers(base.clone());

        // A grouped declarator (`ret (*name)(...)` or `ret (*name[N])(...)`) is
        // always an object declaration — never a function definition — so parse
        // it (and any comma-separated siblings) as globals.
        if self.is_punct(Punct::LParen) {
            let mut items = Vec::new();
            let (name, ty, span) = self.parse_named_declarator(ty0)?;
            self.declare_ordinary(&name, Some(ty.clone()));
            let init =
                if self.eat_punct(Punct::Assign) { Some(self.parse_initializer()?) } else { None };
            items.push(TopLevel::Global(VarDecl { name, ty, init, align, storage, span }));
            while self.eat_punct(Punct::Comma) {
                let (name, ty, span) = self.parse_named_declarator(base.clone())?;
                self.declare_ordinary(&name, Some(ty.clone()));
                let init = if self.eat_punct(Punct::Assign) {
                    Some(self.parse_initializer()?)
                } else {
                    None
                };
                items.push(TopLevel::Global(VarDecl { name, ty, init, align, storage, span }));
            }
            self.expect_punct(Punct::Semi, "';' after global declaration")?;
            return Ok(items);
        }

        let (name, name_span) = self.expect_ident()?;
        self.declare_ordinary(&name, None);

        if self.is_punct(Punct::LParen) {
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
                return Ok(vec![TopLevel::Func(FuncDef {
                    name,
                    ret: ty0,
                    params,
                    variadic: false,
                    is_static: storage == Storage::Static,
                    body,
                    span: name_span,
                })]);
            }
            let (params, variadic) = self.parse_param_list()?;
            for p in &params {
                if let Some(n) = &p.name {
                    self.declare_ordinary(n, Some(p.ty.clone()));
                }
            }
            // A trailing declarator attribute — `f(void) __attribute__((noreturn));`
            // (GNU) or `[[noreturn]]` — sits between the parameter list and the
            // `;`/`{`. bzip2's NORETURN and glibc prototypes rely on this position.
            self.skip_attributes()?;
            if self.is_punct(Punct::LBrace) {
                let body = self.parse_block_stmts()?;
                self.pop_scope();
                return Ok(vec![TopLevel::Func(FuncDef {
                    name,
                    ret: ty0,
                    params,
                    variadic,
                    is_static: storage == Storage::Static,
                    body,
                    span: name_span,
                })]);
            }
            self.pop_scope();
            self.expect_punct(Punct::Semi, "';' or function body after prototype")?;
            return Ok(vec![TopLevel::Proto(FuncProto {
                name,
                ret: ty0,
                params,
                variadic,
                is_static: storage == Storage::Static,
                span: name_span,
            })]);
        }

        // One or more global variables (each may have an initializer).
        let mut items = Vec::new();
        let ty = self.parse_array_suffix(ty0)?;
        self.declare_ordinary(&name, Some(ty.clone()));
        let init = if self.eat_punct(Punct::Assign) { Some(self.parse_initializer()?) } else { None };
        items.push(TopLevel::Global(VarDecl { name, ty, init, align, storage, span: name_span }));
        while self.eat_punct(Punct::Comma) {
            let (name, ty, span) = self.parse_named_declarator(base.clone())?;
            self.declare_ordinary(&name, Some(ty.clone()));
            let init =
                if self.eat_punct(Punct::Assign) { Some(self.parse_initializer()?) } else { None };
            items.push(TopLevel::Global(VarDecl { name, ty, init, align, storage, span }));
        }
        self.expect_punct(Punct::Semi, "';' after global declaration")?;
        Ok(items)
    }

    /// Parse the declarators of a `typedef` (the `typedef` keyword already
    /// consumed), registering each name as a typedef in the current scope.
    fn parse_typedef(&mut self) -> PResult<Vec<TopLevel>> {
        let base = self.parse_decl_specs()?;
        loop {
            let (name, ty, _span) = self.parse_named_declarator(base.clone())?;
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
            if self.is_punct(Punct::RBracket) {
                dims.push(0u64);
            } else {
                let n = self.parse_const_expr()?;
                if n < 0 {
                    return self.err("array size must be non-negative");
                }
                dims.push(n as u64);
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
    /// `[[ ... ]]` sequences and (under the GNU dialects) `__attribute__((...))`.
    /// Standard attributes are accepted and ignored per C23 (an implementation
    /// may ignore any attribute it does not recognize).
    fn skip_attributes(&mut self) -> PResult<()> {
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
            // GNU `__attribute__((...))`.
            if self.std.is_gnu()
                && matches!(self.peek(), TokenKind::Ident(n) if n == "__attribute__")
                && matches!(self.peek_at(1), TokenKind::Punct(Punct::LParen))
            {
                self.bump(); // __attribute__
                let mut depth = 0u32;
                loop {
                    match self.peek() {
                        TokenKind::Punct(Punct::LParen) => depth += 1,
                        TokenKind::Punct(Punct::RParen) => depth -= 1,
                        TokenKind::Eof => return self.err("unterminated __attribute__"),
                        _ => {}
                    }
                    self.bump();
                    if depth == 0 {
                        break;
                    }
                }
                continue;
            }
            return Ok(());
        }
    }

    /// Consume any leading storage-class / function specifiers, returning the
    /// linkage-affecting one (`extern`/`static`) if present. `register`, `auto`,
    /// `inline`, and `_Noreturn` do not affect linkage and yield [`Storage::None`].
    fn consume_storage(&mut self) -> Storage {
        let mut storage = Storage::None;
        loop {
            if self.is_kw(Keyword::Extern) {
                storage = Storage::Extern;
            } else if self.is_kw(Keyword::Static) {
                storage = Storage::Static;
            } else if !(self.is_kw(Keyword::Register)
                || self.is_kw(Keyword::Auto)
                || self.is_kw(Keyword::Inline)
                || self.is_kw(Keyword::Noreturn))
            {
                break;
            }
            self.bump();
        }
        storage
    }

    fn parse_param_list(&mut self) -> PResult<(Vec<Param>, bool)> {
        self.expect_punct(Punct::LParen, "'('")?;
        let mut params = Vec::new();
        let mut variadic = false;
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
            let base = self.parse_decl_specs()?;
            let (name, ty, span) = self.declarator(base)?;
            // A parameter of array or function type decays to a pointer.
            let ty = ty.decayed().unwrap_or(ty);
            params.push(Param { name, ty, span });
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        self.expect_punct(Punct::RParen, "')' to close parameter list")?;
        Ok((params, variadic))
    }

    /// Whether the `(` at the cursor opens an old-style (K&R) identifier list —
    /// the parameter names of an old-style function definition — rather than a
    /// prototype parameter list. This holds when the first token after `(` is an
    /// identifier that does not name a type (a typedef-name would begin a
    /// prototype parameter). Empty `()` and `(void)` are not identifier lists.
    fn at_kr_identifier_list(&self) -> bool {
        matches!(self.peek_at(1),
            TokenKind::Ident(n)
                if !self.is_typedef_name(n) && n != "_BitInt" && n != "constexpr")
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
            self.consume_storage();
            let base = self.parse_decl_specs()?;
            self.consume_storage();
            if self.eat_punct(Punct::Semi) {
                continue;
            }
            loop {
                let (pname, pty, psp) = self.parse_named_declarator(base.clone())?;
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
                | Keyword::Typedef,
            ) => true,
            // C23 `_BitInt`/`constexpr` are lexed as identifiers; treat them as
            // declaration starts so a declaration using them is recognized (the
            // C23 gate is applied in `parse_decl_specs`).
            TokenKind::Ident(name) => {
                name == "_BitInt" || name == "constexpr" || self.is_typedef_name(name)
            }
            _ => false,
        }
    }

    fn parse_decl_specs(&mut self) -> PResult<CType> {
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

        loop {
            // C23 `constexpr` (a declaration specifier) and `_BitInt` (a type
            // specifier) are lexed as ordinary identifiers (they are not global
            // keywords); recognize them here and gate them on C23.
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
            let numeric_seen = has_void
                || has_bool
                || has_char
                || has_short
                || has_int
                || has_float
                || has_double
                || longs > 0
                || signed_spec.is_some();
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
                TokenKind::Keyword(
                    Keyword::Const
                    | Keyword::Volatile
                    | Keyword::Restrict
                    | Keyword::Inline
                    | Keyword::Noreturn,
                ) => {
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
            return Err(Diagnostic::error("expected a type").with_span(start));
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
        if longs >= 2 && !self.std.has_long_long() {
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

    /// Consume leading `*` (with optional qualifiers) and wrap `base`.
    fn parse_pointers(&mut self, mut base: CType) -> CType {
        while self.eat_punct(Punct::Star) {
            // Skip pointer qualifiers.
            while self.is_kw(Keyword::Const)
                || self.is_kw(Keyword::Volatile)
                || self.is_kw(Keyword::Restrict)
            {
                self.bump();
            }
            base = CType::ptr_to(base);
        }
        base
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
            ExprKind::StrLit(bytes) => {
                Some(CType::Array(Box::new(CType::Int(IntTy::new(8, true))), bytes.len() as u64 + 1))
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
                if let CType::Record(id) = rec {
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
            ExprKind::VaArg(_, ty) => Some(ty.clone()),
            ExprKind::VaStart(..) | ExprKind::VaEnd(_) | ExprKind::VaCopy(..) => Some(CType::Void),
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
        let base = self.parse_pointers(base);
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
            Ok((name, ty, span))
        } else {
            let (name, span) = match self.peek().clone() {
                TokenKind::Ident(n) => {
                    let sp = self.bump().span;
                    (Some(n), sp)
                }
                _ => (None, self.peek_span()),
            };
            let ty = self.declarator_suffixes(base)?;
            Ok((name, ty, span))
        }
    }

    /// Parse trailing declarator suffixes: a function parameter list (yielding a
    /// [`CType::Func`]) or array dimensions.
    fn declarator_suffixes(&mut self, base: CType) -> PResult<CType> {
        if self.is_punct(Punct::LParen) {
            let (params, variadic) = self.parse_param_list()?;
            let param_tys: Vec<CType> = params.into_iter().map(|p| p.ty).collect();
            return Ok(CType::Func(Box::new(FuncType { ret: base, params: param_tys, variadic })));
        }
        self.parse_array_suffix(base)
    }

    /// Whether the `(` at the cursor opens a nested (grouped) declarator rather
    /// than a function parameter list: it does when it is followed by `*`, another
    /// `(`, or an identifier that is not a typedef-name (the declared name).
    fn grouped_declarator_ahead(&self) -> bool {
        match self.peek_at(1) {
            TokenKind::Punct(Punct::Star | Punct::LParen) => true,
            TokenKind::Ident(name) => !self.is_typedef_name(name),
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
        let tag = match self.peek().clone() {
            TokenKind::Ident(name) => {
                self.bump();
                Some(name)
            }
            _ => None,
        };
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
        }
        Ok(CType::Record(id))
    }

    fn parse_record_body(&mut self, id: RecordId) -> PResult<()> {
        self.expect_punct(Punct::LBrace, "'{' to open struct/union body")?;
        let mut fields = Vec::new();
        while !self.is_punct(Punct::RBrace) && !self.at_eof() {
            let base = self.parse_decl_specs()?;
            let align = self.last_alignas.take();
            // A member declaration with no declarator: an anonymous struct/union
            // member (an untagged `struct`/`union`, C11), or a nested tagged type
            // declaration (which declares no member).
            if self.is_punct(Punct::Semi) {
                if let CType::Record(rid) = &base
                    && self.records.get(*rid).tag.is_none()
                {
                    if !self.std.anonymous_members() {
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
                continue;
            }
            // An unnamed bit-field: `type : const-expr ;` (including `: 0`), which
            // has no declarator, only a colon and a width.
            if self.is_punct(Punct::Colon) {
                let bit_width = self.parse_bitfield_width(&base)?;
                fields.push(Field {
                    name: String::new(),
                    ty: base.clone(),
                    anonymous: false,
                    align,
                    bit_width: Some(bit_width),
                });
                self.expect_punct(Punct::Semi, "';' after bit-field")?;
                continue;
            }
            loop {
                let (name, ty, name_span) = self.parse_named_declarator(base.clone())?;
                let bit_width = if self.is_punct(Punct::Colon) {
                    let w = self.parse_bitfield_width(&ty)?;
                    // A named bit-field of width 0 is a constraint violation.
                    if w == 0 {
                        return Err(Diagnostic::error(
                            "named bit-field cannot have zero width",
                        )
                        .with_span(name_span));
                    }
                    Some(w)
                } else {
                    None
                };
                fields.push(Field { name, ty, anonymous: false, align, bit_width });
                if !self.eat_punct(Punct::Comma) {
                    break;
                }
            }
            self.expect_punct(Punct::Semi, "';' after struct/union member")?;
        }
        self.expect_punct(Punct::RBrace, "'}' to close struct/union body")?;
        self.records.defs[id].fields = fields;
        self.records.defs[id].complete = true;
        Ok(())
    }

    /// Parse the `: const-expr` of a bit-field declared with base type `ty` (the
    /// leading `:` is at the cursor). Validate that `ty` is an integer type and
    /// that the width is in `0..=bits(ty)` (where `_Bool` counts as one bit).
    fn parse_bitfield_width(&mut self, ty: &CType) -> PResult<u32> {
        let colon = self.expect_punct(Punct::Colon, "':' in bit-field")?;
        if !ty.is_integer() {
            return Err(Diagnostic::error("bit-field has a non-integer type").with_span(colon));
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
        if let TokenKind::Ident(_) = self.peek() {
            self.bump(); // tag (ignored; an enum is an int)
        }
        if self.eat_punct(Punct::LBrace) {
            let mut next = 0i128;
            while !self.is_punct(Punct::RBrace) && !self.at_eof() {
                let (name, _span) = self.expect_ident()?;
                if self.eat_punct(Punct::Assign) {
                    next = self.parse_const_expr()?;
                }
                self.enum_map.insert(name.clone(), next);
                self.enum_consts.push((name, next));
                next += 1;
                if !self.eat_punct(Punct::Comma) {
                    break;
                }
            }
            self.expect_punct(Punct::RBrace, "'}' to close enum body")?;
        }
        Ok(CType::int())
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

    /// Fold a parsed expression to a constant integer, or `None` if it is not a
    /// constant expression the parser can evaluate.
    fn eval_const_expr(&self, e: &Expr) -> Option<i128> {
        match &e.kind {
            ExprKind::IntLit(v, _) => Some(*v),
            ExprKind::Ident(name) => self.enum_map.get(name).copied(),
            ExprKind::Unary(op, inner) => {
                let v = self.eval_const_expr(inner)?;
                match op {
                    UnaryOp::Neg => Some(-v),
                    UnaryOp::Plus => Some(v),
                    UnaryOp::BitNot => Some(!v),
                    UnaryOp::LNot => Some(i128::from(v == 0)),
                    _ => None,
                }
            }
            ExprKind::Binary(op, l, r) => {
                let a = self.eval_const_expr(l)?;
                let b = self.eval_const_expr(r)?;
                eval_binop(*op, a, b)
            }
            ExprKind::Cond(c, t, f) => {
                let cv = self.eval_const_expr(c)?;
                if cv != 0 { self.eval_const_expr(t) } else { self.eval_const_expr(f) }
            }
            ExprKind::Cast(_, inner) => self.eval_const_expr(inner),
            ExprKind::SizeofType(ty) => Some(layout::size_of(&self.records, ty) as i128),
            ExprKind::AlignofType(ty) => Some(layout::align_of(&self.records, ty) as i128),
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
            let is_decl = self.at_type_specifier() && !self.label_ahead();
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
                self.expect_punct(Punct::Colon, "':' after case label")?;
                let body = Box::new(self.parse_labeled_body()?);
                let span = start.merge(body.span);
                Ok(self.stmt(StmtKind::Case(value, body), span))
            }
            TokenKind::Keyword(Keyword::Default) => {
                self.bump();
                self.expect_punct(Punct::Colon, "':' after default label")?;
                let body = Box::new(self.parse_labeled_body()?);
                let span = start.merge(body.span);
                Ok(self.stmt(StmtKind::Default(body), span))
            }
            TokenKind::Keyword(Keyword::Goto) => {
                self.bump();
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
        if self.eat_kw(Keyword::Typedef) {
            self.parse_typedef()?;
            return Ok(self.stmt(StmtKind::Expr(None), start));
        }
        let storage = self.consume_storage();
        if self.eat_kw(Keyword::Typedef) {
            self.parse_typedef()?;
            return Ok(self.stmt(StmtKind::Expr(None), start));
        }
        let base = self.parse_decl_specs()?;
        let align = self.last_alignas.take();
        let is_constexpr = self.last_constexpr;
        let storage = merge_storage(storage, self.consume_storage());
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
        let mut decls = Vec::new();
        loop {
            let (name, ty, name_span) = self.parse_named_declarator(base.clone())?;
            self.declare_ordinary(&name, Some(ty.clone()));
            let init =
                if self.eat_punct(Punct::Assign) { Some(self.parse_initializer()?) } else { None };
            decls.push(VarDecl { name, ty, init, align, storage, span: name_span });
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
                | Keyword::Restrict
                | Keyword::Typeof
                | Keyword::TypeofUnqual
                | Keyword::Struct
                | Keyword::Union
                | Keyword::Enum,
            ) => true,
            // `sizeof(_BitInt(N))` / a `(_BitInt(N))` cast (C23).
            TokenKind::Ident(name) => name == "_BitInt" || self.is_typedef_name(name),
            _ => false,
        }
    }

    fn parse_unary(&mut self) -> PResult<Expr> {
        let start = self.peek_span();
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
        if self.is_kw(Keyword::Alignof) {
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
        self.bump(); // _Alignof / alignof
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
                let span = self.bump().span;
                Ok(Expr { kind: ExprKind::Ident(name), span })
            }
            TokenKind::Punct(Punct::LParen) => {
                self.bump();
                let inner = self.parse_expr()?;
                self.expect_punct(Punct::RParen, "')' to close parenthesized expression")?;
                Ok(inner)
            }
            TokenKind::Str(_) => {
                // Adjacent string literals concatenate into one literal.
                let mut bytes = Vec::new();
                let mut span = self.peek_span();
                while let TokenKind::Str(s) = self.peek().clone() {
                    bytes.extend_from_slice(&s);
                    span = span.merge(self.bump().span);
                }
                Ok(Expr { kind: ExprKind::StrLit(bytes), span })
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

/// Reduce an integer constant `v` to the representable value of integer type
/// `ty`: mask to the type's value-bit count (`N` for a `_BitInt(N)`), then
/// sign-extend for a signed type. `_Bool` normalizes to 0/1. Non-integer types
/// are returned unchanged.
fn reduce_const_to_type(v: i128, ty: &CType) -> i128 {
    let (bits, signed) = match ty {
        CType::Bool => return i128::from(v != 0),
        CType::Int(i) => (u32::from(i.value_bits()), i.signed),
        _ => return v,
    };
    if bits == 0 || bits >= 128 {
        return v;
    }
    let mask = (1i128 << bits) - 1;
    let m = v & mask;
    if signed && (m & (1i128 << (bits - 1))) != 0 {
        m - (1i128 << bits)
    } else {
        m
    }
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

/// Whether `name` is one of the variadic compiler builtins the frontend expands
/// (the `<stdarg.h>` `va_*` macros map onto these).
fn is_va_builtin(name: &str) -> bool {
    matches!(
        name,
        "__builtin_va_start" | "__builtin_va_arg" | "__builtin_va_end" | "__builtin_va_copy"
    )
}

/// Fold a binary operator over two constant integers (constant-expression
/// evaluation for array sizes, enum values, and designators).
fn eval_binop(op: BinaryOp, a: i128, b: i128) -> Option<i128> {
    Some(match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div if b != 0 => a / b,
        BinaryOp::Rem if b != 0 => a % b,
        BinaryOp::BitAnd => a & b,
        BinaryOp::BitOr => a | b,
        BinaryOp::BitXor => a ^ b,
        BinaryOp::Shl => a << b,
        BinaryOp::Shr => a >> b,
        BinaryOp::Eq => i128::from(a == b),
        BinaryOp::Ne => i128::from(a != b),
        BinaryOp::Lt => i128::from(a < b),
        BinaryOp::Le => i128::from(a <= b),
        BinaryOp::Gt => i128::from(a > b),
        BinaryOp::Ge => i128::from(a >= b),
        BinaryOp::LAnd => i128::from(a != 0 && b != 0),
        BinaryOp::LOr => i128::from(a != 0 || b != 0),
        _ => return None,
    })
}
