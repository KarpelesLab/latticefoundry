//! A hand-written lexer for the freestanding C subset.
//!
//! Turns source text into a flat [`Token`] stream with byte-offset spans. It is
//! written directly from the C token grammar (clean room, tenet T1): keywords,
//! identifiers, integer and character constants, and the operator/punctuator
//! set the parser needs. Whitespace and both comment forms are skipped. There is
//! no preprocessor: `#`-lines are rejected with a diagnostic.

use latticefoundry::support::diagnostics::{Diagnostic, FileId, Span};

use crate::ast::{CType, IntTy};

/// A lexical token: its [`TokenKind`] and the source [`Span`] it covers.
#[derive(Clone, Debug)]
pub struct Token {
    /// The token classification and payload.
    pub kind: TokenKind,
    /// The source span the token covers.
    pub span: Span,
}

/// The classification (and payload) of a [`Token`].
#[derive(Clone, PartialEq, Debug)]
pub enum TokenKind {
    /// An identifier (not a keyword).
    Ident(String),
    /// An integer or character constant, with the C type its literal implies.
    IntLit(i128, CType),
    /// A floating-point constant (its exact value, already rounded to the target
    /// precision) with the C type its suffix implies (`float`/`double`).
    FloatLit(f64, CType),
    /// A string literal, carrying its already-decoded contents (produced by the
    /// preprocessor; the expression grammar of this subset rejects it, but
    /// directives such as `_Static_assert` accept one).
    Str(String),
    /// A language keyword.
    Keyword(Keyword),
    /// A punctuator or operator.
    Punct(Punct),
    /// End of input.
    Eof,
}

/// The keywords recognized by the subset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Keyword {
    /// `void`
    Void,
    /// `_Bool`
    Bool,
    /// `char`
    Char,
    /// `short`
    Short,
    /// `int`
    Int,
    /// `long`
    Long,
    /// `float`
    Float,
    /// `double`
    Double,
    /// `signed`
    Signed,
    /// `unsigned`
    Unsigned,
    /// `const` (accepted, ignored)
    Const,
    /// `volatile` (accepted, ignored)
    Volatile,
    /// `extern`
    Extern,
    /// `static`
    Static,
    /// `if`
    If,
    /// `else`
    Else,
    /// `while`
    While,
    /// `do`
    Do,
    /// `for`
    For,
    /// `return`
    Return,
    /// `break`
    Break,
    /// `continue`
    Continue,
    /// `switch`
    Switch,
    /// `case`
    Case,
    /// `default`
    Default,
    /// `goto`
    Goto,
    /// `sizeof`
    Sizeof,
    /// `struct`
    Struct,
    /// `union`
    Union,
    /// `enum`
    Enum,
    /// `typedef`
    Typedef,
    /// `restrict` (C99; accepted as a type qualifier and ignored)
    Restrict,
    /// `inline` (C99; accepted as a function specifier and ignored)
    Inline,
    /// `_Noreturn` (C11; accepted as a function specifier and ignored)
    Noreturn,
    /// `_Alignof` (C11)
    Alignof,
    /// `_Static_assert` (C11)
    StaticAssert,
    /// `_Generic` (C11)
    Generic,
}

/// The punctuators and operators recognized by the subset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(missing_docs)]
pub enum Punct {
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Semi,
    Comma,
    Ellipsis,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    Pipe,
    Caret,
    Tilde,
    Bang,
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    AmpAmp,
    PipePipe,
    Assign,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,
    PlusPlus,
    MinusMinus,
    Question,
    Colon,
    Arrow,
    Dot,
}

fn keyword_from(word: &str) -> Option<Keyword> {
    Some(match word {
        "void" => Keyword::Void,
        "_Bool" => Keyword::Bool,
        "char" => Keyword::Char,
        "short" => Keyword::Short,
        "int" => Keyword::Int,
        "long" => Keyword::Long,
        "float" => Keyword::Float,
        "double" => Keyword::Double,
        "signed" => Keyword::Signed,
        "unsigned" => Keyword::Unsigned,
        "const" => Keyword::Const,
        "volatile" => Keyword::Volatile,
        "extern" => Keyword::Extern,
        "static" => Keyword::Static,
        "if" => Keyword::If,
        "else" => Keyword::Else,
        "while" => Keyword::While,
        "do" => Keyword::Do,
        "for" => Keyword::For,
        "return" => Keyword::Return,
        "break" => Keyword::Break,
        "continue" => Keyword::Continue,
        "switch" => Keyword::Switch,
        "case" => Keyword::Case,
        "default" => Keyword::Default,
        "goto" => Keyword::Goto,
        "sizeof" => Keyword::Sizeof,
        "struct" => Keyword::Struct,
        "union" => Keyword::Union,
        "enum" => Keyword::Enum,
        "typedef" => Keyword::Typedef,
        _ => return None,
    })
}

/// The lexer state: the source bytes and a cursor.
#[derive(Debug)]
struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    file: FileId,
}

/// Tokenize `src`, returning the token stream (terminated by an `Eof` token) or
/// the collected lexical diagnostics.
pub fn lex(src: &str, file: FileId) -> Result<Vec<Token>, Vec<Diagnostic>> {
    let mut lexer = Lexer { src: src.as_bytes(), pos: 0, file };
    let mut tokens = Vec::new();
    let mut errors = Vec::new();
    loop {
        match lexer.next_token() {
            Ok(tok) => {
                let is_eof = tok.kind == TokenKind::Eof;
                tokens.push(tok);
                if is_eof {
                    break;
                }
            }
            Err(d) => {
                errors.push(d);
                // Skip one byte to make progress and keep collecting errors.
                if lexer.pos < lexer.src.len() {
                    lexer.pos += 1;
                } else {
                    break;
                }
            }
        }
    }
    if errors.is_empty() { Ok(tokens) } else { Err(errors) }
}

impl Lexer<'_> {
    fn span(&self, start: usize, end: usize) -> Span {
        Span::new(self.file, start as u32, end as u32)
    }

    fn error(&self, start: usize, end: usize, msg: impl Into<String>) -> Diagnostic {
        Diagnostic::error(msg).with_span(self.span(start, end))
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.src.get(self.pos + 1).copied()
    }

    fn skip_trivia(&mut self) -> Result<(), Diagnostic> {
        loop {
            match self.peek() {
                Some(b' ' | b'\t' | b'\r' | b'\n' | 0x0c) => self.pos += 1,
                Some(b'/') if self.peek2() == Some(b'/') => {
                    while let Some(c) = self.peek() {
                        if c == b'\n' {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                Some(b'/') if self.peek2() == Some(b'*') => {
                    let start = self.pos;
                    self.pos += 2;
                    loop {
                        match self.peek() {
                            Some(b'*') if self.peek2() == Some(b'/') => {
                                self.pos += 2;
                                break;
                            }
                            Some(_) => self.pos += 1,
                            None => {
                                return Err(
                                    self.error(start, self.pos, "unterminated block comment")
                                );
                            }
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn next_token(&mut self) -> Result<Token, Diagnostic> {
        self.skip_trivia()?;
        let start = self.pos;
        let Some(c) = self.peek() else {
            return Ok(Token { kind: TokenKind::Eof, span: self.span(start, start) });
        };

        if c == b'#' {
            return Err(self.error(
                start,
                start + 1,
                "preprocessor directives are not supported in this freestanding subset",
            ));
        }

        if c == b'_' || c.is_ascii_alphabetic() {
            return Ok(self.lex_ident());
        }
        // A digit, or a `.` immediately followed by a digit (e.g. `.5`), begins a
        // numeric constant.
        if c.is_ascii_digit() || (c == b'.' && self.peek2().is_some_and(|d| d.is_ascii_digit())) {
            return self.lex_number();
        }
        if c == b'\'' {
            return self.lex_char();
        }
        if c == b'"' {
            return Err(self.error(
                start,
                start + 1,
                "string literals are out of scope in this C subset",
            ));
        }
        self.lex_punct()
    }

    fn lex_ident(&mut self) -> Token {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == b'_' || c.is_ascii_alphanumeric() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let word = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        let kind = match keyword_from(word) {
            Some(kw) => TokenKind::Keyword(kw),
            None => TokenKind::Ident(word.to_owned()),
        };
        Token { kind, span: self.span(start, self.pos) }
    }

    fn lex_number(&mut self) -> Result<Token, Diagnostic> {
        let start = self.pos;
        let (radix, digits_start) = if self.peek() == Some(b'0')
            && matches!(self.peek2(), Some(b'x' | b'X'))
        {
            self.pos += 2;
            (16, self.pos)
        } else if self.peek() == Some(b'0') {
            (8, self.pos)
        } else {
            (10, self.pos)
        };

        while let Some(c) = self.peek() {
            let ok = match radix {
                16 => c.is_ascii_hexdigit(),
                8 => (b'0'..=b'7').contains(&c),
                _ => c.is_ascii_digit(),
            };
            if ok {
                self.pos += 1;
            } else {
                break;
            }
        }
        // A `.`, an exponent (`e`/`E` for decimal, `p`/`P` for hex), or a float
        // suffix makes this a floating constant; consume the rest and parse it.
        let is_float = matches!(self.peek(), Some(b'.'))
            || (radix == 10 && matches!(self.peek(), Some(b'e' | b'E')))
            || (radix == 16 && matches!(self.peek(), Some(b'p' | b'P')));
        if is_float {
            return self.lex_float(start, radix == 16);
        }

        let digits = std::str::from_utf8(&self.src[digits_start..self.pos]).unwrap_or("");
        // An "0" alone lexes with digits_start == pos for octal; treat as 0.
        let text = if radix == 8 && digits.is_empty() { "0" } else { digits };

        // Parse suffix: any combination of u/U and l/L (l, ll).
        let mut unsigned = false;
        let mut long = false;
        let suffix_start = self.pos;
        loop {
            match self.peek() {
                Some(b'u' | b'U') => {
                    unsigned = true;
                    self.pos += 1;
                }
                Some(b'l' | b'L') => {
                    long = true;
                    self.pos += 1;
                    if matches!(self.peek(), Some(b'l' | b'L')) {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
        // Reject a stray identifier tail (e.g. `123abc`).
        if let Some(c) = self.peek()
            && (c == b'_' || c.is_ascii_alphanumeric())
        {
            return Err(self.error(start, self.pos + 1, "invalid digit in integer constant"));
        }

        let value = match i128::from_str_radix(text, radix) {
            Ok(v) => v,
            Err(_) => {
                return Err(self.error(start, self.pos, "integer constant out of range"));
            }
        };
        let _ = suffix_start;
        let ty = integer_literal_type(value, radix != 10, unsigned, long);
        Ok(Token { kind: TokenKind::IntLit(value, ty), span: self.span(start, self.pos) })
    }

    /// Finish lexing a floating constant that begins at `start` (the cursor sits
    /// on the `.`, exponent marker, or already past the whole-number digits). The
    /// lexer, unlike the preprocessor, is used only in unit tests, so hex floats
    /// are accepted unconditionally here (the preprocessor gates them by `--std`).
    fn lex_float(&mut self, start: usize, hex: bool) -> Result<Token, Diagnostic> {
        // Fractional part.
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while self
                .peek()
                .is_some_and(|c| if hex { c.is_ascii_hexdigit() } else { c.is_ascii_digit() })
            {
                self.pos += 1;
            }
        }
        // Exponent (`eNN`/`pNN`, with optional sign).
        let (lo, hi) = if hex { (b'p', b'P') } else { (b'e', b'E') };
        if matches!(self.peek(), Some(c) if c == lo || c == hi) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        // One optional type suffix (`f`/`F`/`l`/`L`).
        if matches!(self.peek(), Some(b'f' | b'F' | b'l' | b'L')) {
            self.pos += 1;
        }
        // Reject a stray identifier/digit tail (e.g. `1.5fx`).
        if let Some(c) = self.peek()
            && (c == b'_' || c.is_ascii_alphanumeric())
        {
            return Err(self.error(start, self.pos + 1, "invalid suffix on floating constant"));
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        match parse_float_literal(text, true) {
            Ok((value, ty)) => {
                Ok(Token { kind: TokenKind::FloatLit(value, ty), span: self.span(start, self.pos) })
            }
            Err(msg) => Err(self.error(start, self.pos, msg)),
        }
    }

    fn lex_char(&mut self) -> Result<Token, Diagnostic> {
        let start = self.pos;
        self.pos += 1; // opening quote
        let value: i128 = match self.peek() {
            Some(b'\\') => {
                self.pos += 1;
                let esc = self.peek().ok_or_else(|| {
                    self.error(start, self.pos, "unterminated character constant")
                })?;
                self.pos += 1;
                match esc {
                    b'n' => 10,
                    b't' => 9,
                    b'r' => 13,
                    b'0' => 0,
                    b'\\' => 92,
                    b'\'' => 39,
                    b'"' => 34,
                    b'a' => 7,
                    b'b' => 8,
                    b'f' => 12,
                    b'v' => 11,
                    other => i128::from(other),
                }
            }
            Some(b'\'') => {
                return Err(self.error(start, self.pos + 1, "empty character constant"));
            }
            Some(c) => {
                self.pos += 1;
                i128::from(c)
            }
            None => {
                return Err(self.error(start, self.pos, "unterminated character constant"));
            }
        };
        if self.peek() != Some(b'\'') {
            return Err(self.error(start, self.pos, "unterminated character constant"));
        }
        self.pos += 1; // closing quote
        // A character constant has type `int` in C.
        Ok(Token { kind: TokenKind::IntLit(value, CType::int()), span: self.span(start, self.pos) })
    }

    fn lex_punct(&mut self) -> Result<Token, Diagnostic> {
        let start = self.pos;
        let c = self.peek().expect("caller ensured a byte");
        let two = self.peek2();
        // Three-character punctuators.
        if c == b'.' && two == Some(b'.') && self.src.get(self.pos + 2) == Some(&b'.') {
            self.pos += 3;
            return Ok(self.punct(Punct::Ellipsis, start));
        }
        if (c == b'<' && two == Some(b'<') && self.src.get(self.pos + 2) == Some(&b'='))
            || (c == b'>' && two == Some(b'>') && self.src.get(self.pos + 2) == Some(&b'='))
        {
            let p = if c == b'<' { Punct::ShlEq } else { Punct::ShrEq };
            self.pos += 3;
            return Ok(self.punct(p, start));
        }
        // Two-character punctuators.
        if let Some(t) = two {
            let two_char = match (c, t) {
                (b'<', b'<') => Some(Punct::Shl),
                (b'>', b'>') => Some(Punct::Shr),
                (b'<', b'=') => Some(Punct::Le),
                (b'>', b'=') => Some(Punct::Ge),
                (b'=', b'=') => Some(Punct::EqEq),
                (b'!', b'=') => Some(Punct::Ne),
                (b'&', b'&') => Some(Punct::AmpAmp),
                (b'|', b'|') => Some(Punct::PipePipe),
                (b'+', b'=') => Some(Punct::PlusEq),
                (b'-', b'=') => Some(Punct::MinusEq),
                (b'*', b'=') => Some(Punct::StarEq),
                (b'/', b'=') => Some(Punct::SlashEq),
                (b'%', b'=') => Some(Punct::PercentEq),
                (b'&', b'=') => Some(Punct::AmpEq),
                (b'|', b'=') => Some(Punct::PipeEq),
                (b'^', b'=') => Some(Punct::CaretEq),
                (b'+', b'+') => Some(Punct::PlusPlus),
                (b'-', b'-') => Some(Punct::MinusMinus),
                (b'-', b'>') => Some(Punct::Arrow),
                _ => None,
            };
            if let Some(p) = two_char {
                self.pos += 2;
                return Ok(self.punct(p, start));
            }
        }
        // One-character punctuators.
        let one = match c {
            b'(' => Punct::LParen,
            b')' => Punct::RParen,
            b'{' => Punct::LBrace,
            b'}' => Punct::RBrace,
            b'[' => Punct::LBracket,
            b']' => Punct::RBracket,
            b';' => Punct::Semi,
            b',' => Punct::Comma,
            b'+' => Punct::Plus,
            b'-' => Punct::Minus,
            b'*' => Punct::Star,
            b'/' => Punct::Slash,
            b'%' => Punct::Percent,
            b'&' => Punct::Amp,
            b'|' => Punct::Pipe,
            b'^' => Punct::Caret,
            b'~' => Punct::Tilde,
            b'!' => Punct::Bang,
            b'<' => Punct::Lt,
            b'>' => Punct::Gt,
            b'=' => Punct::Assign,
            b'?' => Punct::Question,
            b':' => Punct::Colon,
            b'.' => Punct::Dot,
            _ => {
                return Err(self.error(start, start + 1, format!("unexpected character '{}'", c as char)));
            }
        };
        self.pos += 1;
        Ok(self.punct(one, start))
    }

    fn punct(&self, p: Punct, start: usize) -> Token {
        Token { kind: TokenKind::Punct(p), span: self.span(start, self.pos) }
    }
}

/// Determine the C type of an integer literal from its value, radix, and
/// suffix, per the C rules restricted to this subset (32-bit `int`, 64-bit
/// `long`). Decimal literals never pick an unsigned type unless suffixed `u`;
/// hex/octal literals may.
pub(crate) fn integer_literal_type(
    value: i128,
    hex_or_octal: bool,
    unsigned: bool,
    long: bool,
) -> CType {
    let fits_i32 = value <= i128::from(i32::MAX);
    let fits_u32 = value <= i128::from(u32::MAX);
    let fits_i64 = value <= i128::from(i64::MAX);

    let width = if long {
        64
    } else if unsigned {
        if fits_u32 { 32 } else { 64 }
    } else if hex_or_octal {
        // int, unsigned int, long, unsigned long
        if fits_i32 {
            32
        } else if fits_u32 || fits_i64 {
            // unsigned int (fits_u32) or long (fits_i64)
            if fits_u32 { 32 } else { 64 }
        } else {
            64
        }
    } else {
        // decimal: int, long, long long — always signed
        if fits_i32 { 32 } else { 64 }
    };

    let signed = if unsigned {
        false
    } else if hex_or_octal {
        // stays signed unless the signed type of that width can't hold it
        match width {
            32 => fits_i32,
            _ => fits_i64,
        }
    } else {
        true
    };

    CType::Int(IntTy { width, signed })
}

/// Whether a preprocessing-number spelling denotes a floating (as opposed to an
/// integer) constant: it has a `.`, a decimal exponent (`e`/`E`, for non-hex),
/// or a binary exponent (`p`/`P`, for hex). Digit separators must already be
/// stripped by the caller.
pub(crate) fn is_float_ppnumber(text: &str) -> bool {
    let is_hex = text.len() >= 2 && matches!(&text.as_bytes()[..2], b"0x" | b"0X");
    if text.contains('.') {
        return true;
    }
    if is_hex {
        text.contains(['p', 'P'])
    } else {
        text.contains(['e', 'E'])
    }
}

/// Parse a floating constant spelling (decimal or hexadecimal, with an optional
/// `f`/`F`/`l`/`L` suffix) into its exact value and C type. `long double` (`l`
/// suffix) is modelled as `double`. A `float` (`f` suffix) value is rounded to
/// binary32 and stored back as the exact `f64` of that binary32. `hex_float_ok`
/// gates C99 hexadecimal floating constants.
pub(crate) fn parse_float_literal(text: &str, hex_float_ok: bool) -> Result<(f64, CType), String> {
    let (num, is_f32) = match text.as_bytes().last() {
        Some(b'f' | b'F') => (&text[..text.len() - 1], true),
        Some(b'l' | b'L') => (&text[..text.len() - 1], false),
        _ => (text, false),
    };
    let is_hex = num.len() >= 2 && matches!(&num.as_bytes()[..2], b"0x" | b"0X");
    let value = if is_hex {
        if !hex_float_ok {
            return Err(
                "hexadecimal floating constants are a C99 feature (use -std=c99 or later)".to_owned(),
            );
        }
        parse_hex_float(&num[2..])?
    } else {
        num.parse::<f64>().map_err(|_| "invalid floating constant".to_owned())?
    };
    let ty = if is_f32 { CType::float() } else { CType::double() };
    // Round to the target precision so the stored value is bit-exact.
    let value = if is_f32 { f64::from(value as f32) } else { value };
    Ok((value, ty))
}

/// Parse the body of a hexadecimal floating constant (the text after `0x`):
/// `[hexint][.hexfrac]p[±]decexp`, evaluating `mantissa × 2^exp` in `f64`.
fn parse_hex_float(body: &str) -> Result<f64, String> {
    let err = || "invalid hexadecimal floating constant".to_owned();
    let (mant, exp) = body.split_once(['p', 'P']).ok_or_else(|| {
        "hexadecimal floating constant requires a 'p' binary exponent".to_owned()
    })?;
    let (int_part, frac_part) = mant.split_once('.').unwrap_or((mant, ""));
    if int_part.is_empty() && frac_part.is_empty() {
        return Err(err());
    }
    let mut value = 0.0f64;
    for c in int_part.chars() {
        value = value * 16.0 + f64::from(c.to_digit(16).ok_or_else(err)?);
    }
    let mut scale = 1.0f64 / 16.0;
    for c in frac_part.chars() {
        value += f64::from(c.to_digit(16).ok_or_else(err)?) * scale;
        scale /= 16.0;
    }
    let e: i32 = exp.parse().map_err(|_| err())?;
    Ok(value * 2f64.powi(e))
}
