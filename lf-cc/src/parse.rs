//! A recursive-descent parser for the freestanding C subset.
//!
//! Written directly from the C grammar (clean room, tenet T1): declaration
//! specifiers and declarators, statements, and a precedence-climbing expression
//! parser that implements C's operator precedence and associativity. It consumes
//! the [`crate::lex`] token stream and produces the untyped [`crate::ast`] tree.
//! Errors are reported as [`Diagnostic`]s with the offending token's span; the
//! parser bails on the first error.

use latticefoundry::support::diagnostics::{Diagnostic, Span};

use crate::ast::{
    BinaryOp, CType, Expr, ExprKind, FuncDef, FuncProto, IntTy, Param, Stmt, StmtKind, TopLevel,
    TranslationUnit, UnaryOp, VarDecl,
};
use crate::lex::{Keyword, Punct, Token, TokenKind};

type PResult<T> = Result<T, Diagnostic>;

/// Parse a token stream into a [`TranslationUnit`].
pub fn parse(tokens: Vec<Token>) -> Result<TranslationUnit, Vec<Diagnostic>> {
    let mut parser = Parser { tokens, pos: 0 };
    match parser.parse_unit() {
        Ok(unit) => Ok(unit),
        Err(d) => Err(vec![d]),
    }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
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

    // --- top level ---------------------------------------------------------

    fn parse_unit(&mut self) -> PResult<TranslationUnit> {
        let mut items = Vec::new();
        while !self.at_eof() {
            items.push(self.parse_top_level()?);
        }
        Ok(TranslationUnit { items })
    }

    fn parse_top_level(&mut self) -> PResult<TopLevel> {
        // Storage-class specifiers (extern/static) are consumed and ignored for
        // linkage purposes in this single-TU subset.
        self.consume_storage();
        let base = self.parse_decl_specs()?;
        self.consume_storage();

        // A declarator: pointers, then a name, then either function params or a
        // variable initializer.
        let ty = self.parse_pointers(base.clone());
        let (name, name_span) = self.expect_ident()?;

        if self.is_punct(Punct::LParen) {
            let (params, variadic) = self.parse_param_list()?;
            if self.is_punct(Punct::LBrace) {
                let body = self.parse_block_stmts()?;
                return Ok(TopLevel::Func(FuncDef { name, ret: ty, params, body, span: name_span }));
            }
            self.expect_punct(Punct::Semi, "';' or function body after prototype")?;
            return Ok(TopLevel::Proto(FuncProto {
                name,
                ret: ty,
                params,
                variadic,
                span: name_span,
            }));
        }

        // A global variable (possibly with an initializer). Only the first
        // declarator is kept per item; commas start a fresh VarDecl below.
        let init = if self.eat_punct(Punct::Assign) { Some(self.parse_assign()?) } else { None };
        let decl = VarDecl { name, ty, init, span: name_span };
        // Additional comma-separated globals reuse the base type.
        if self.is_punct(Punct::Comma) {
            // Emit the first as its own item by returning it; but to keep one
            // item per declarator we only support a single global per statement
            // here for simplicity. Chain the rest by recursion is awkward, so
            // require a semicolon.
            return self.err("multiple declarators in one global declaration are not supported");
        }
        self.expect_punct(Punct::Semi, "';' after global declaration")?;
        Ok(TopLevel::Global(decl))
    }

    fn consume_storage(&mut self) {
        while self.is_kw(Keyword::Extern) || self.is_kw(Keyword::Static) {
            self.bump();
        }
    }

    fn parse_param_list(&mut self) -> PResult<(Vec<Param>, bool)> {
        self.expect_punct(Punct::LParen, "'('")?;
        let mut params = Vec::new();
        let mut variadic = false;
        if self.eat_punct(Punct::RParen) {
            return Ok((params, variadic));
        }
        // `(void)` means an explicit empty parameter list.
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
            let ty = self.parse_pointers(base);
            let (name, span) = match self.peek().clone() {
                TokenKind::Ident(n) => {
                    let sp = self.bump().span;
                    (Some(n), sp)
                }
                _ => (None, self.peek_span()),
            };
            params.push(Param { name, ty, span });
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        self.expect_punct(Punct::RParen, "')' to close parameter list")?;
        Ok((params, variadic))
    }

    // --- types -------------------------------------------------------------

    /// Whether the current token begins a declaration (a type or storage
    /// specifier keyword).
    fn at_type_specifier(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Keyword(
                Keyword::Void
                    | Keyword::Bool
                    | Keyword::Char
                    | Keyword::Short
                    | Keyword::Int
                    | Keyword::Long
                    | Keyword::Signed
                    | Keyword::Unsigned
                    | Keyword::Const
                    | Keyword::Volatile
                    | Keyword::Extern
                    | Keyword::Static
            )
        )
    }

    fn parse_decl_specs(&mut self) -> PResult<CType> {
        let start = self.peek_span();
        let mut longs = 0u8;
        let mut has_short = false;
        let mut has_char = false;
        let mut has_int = false;
        let mut has_void = false;
        let mut has_bool = false;
        let mut signed_spec: Option<bool> = None;
        let mut saw_any = false;

        loop {
            match self.peek() {
                TokenKind::Keyword(Keyword::Const | Keyword::Volatile) => {
                    self.bump();
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
        if has_void {
            return Ok(CType::Void);
        }
        if has_bool {
            return Ok(CType::Bool);
        }
        if has_char {
            // Plain `char` is signed on this target; `signed`/`unsigned` override.
            let signed = signed_spec.unwrap_or(true);
            return Ok(CType::Int(IntTy { width: 8, signed }));
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
        Ok(CType::Int(IntTy { width, signed }))
    }

    /// Consume leading `*` (with optional qualifiers) and wrap `base`.
    fn parse_pointers(&mut self, mut base: CType) -> CType {
        while self.eat_punct(Punct::Star) {
            // Skip pointer qualifiers.
            while self.is_kw(Keyword::Const) || self.is_kw(Keyword::Volatile) {
                self.bump();
            }
            base = CType::ptr_to(base);
        }
        base
    }

    /// Parse a type-name (used in casts and `sizeof`): specifiers plus an
    /// abstract declarator (pointer levels only in this subset).
    fn parse_type_name(&mut self) -> PResult<CType> {
        let base = self.parse_decl_specs()?;
        Ok(self.parse_pointers(base))
    }

    // --- statements --------------------------------------------------------

    fn parse_block_stmts(&mut self) -> PResult<Vec<Stmt>> {
        self.expect_punct(Punct::LBrace, "'{'")?;
        let mut stmts = Vec::new();
        while !self.is_punct(Punct::RBrace) && !self.at_eof() {
            stmts.push(self.parse_stmt()?);
        }
        self.expect_punct(Punct::RBrace, "'}' to close block")?;
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> PResult<Stmt> {
        let start = self.peek_span();
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
        self.consume_storage();
        let base = self.parse_decl_specs()?;
        let mut decls = Vec::new();
        loop {
            let ty = self.parse_pointers(base.clone());
            let (name, name_span) = self.expect_ident()?;
            let init = if self.eat_punct(Punct::Assign) { Some(self.parse_assign()?) } else { None };
            decls.push(VarDecl { name, ty, init, span: name_span });
            if !self.eat_punct(Punct::Comma) {
                break;
            }
        }
        let end = self.expect_punct(Punct::Semi, "';' after declaration")?;
        Ok(self.stmt(StmtKind::Decl(decls), start.merge(end)))
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
            let inner = self.parse_cast()?;
            let span = start.merge(inner.span);
            return Ok(Expr { kind: ExprKind::Cast(ty, Box::new(inner)), span });
        }
        self.parse_unary()
    }

    /// Whether a `(` at the cursor is followed by a type-name (so this is a cast
    /// or a `sizeof(type)` rather than a parenthesized expression).
    fn type_name_follows_lparen(&self) -> bool {
        matches!(
            self.peek_at(1),
            TokenKind::Keyword(
                Keyword::Void
                    | Keyword::Bool
                    | Keyword::Char
                    | Keyword::Short
                    | Keyword::Int
                    | Keyword::Long
                    | Keyword::Signed
                    | Keyword::Unsigned
                    | Keyword::Const
                    | Keyword::Volatile
            )
        )
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

    fn parse_postfix(&mut self) -> PResult<Expr> {
        let mut expr = self.parse_primary()?;
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
            TokenKind::Ident(name) => {
                let span = self.bump().span;
                Ok(Expr { kind: ExprKind::Ident(name), span })
            }
            TokenKind::Punct(Punct::LParen) => {
                self.bump();
                let inner = self.parse_expr()?;
                self.expect_punct(Punct::RParen, "')' to close parenthesized expression")?;
                Ok(inner)
            }
            _ => self.err("expected an expression"),
        }
    }
}
