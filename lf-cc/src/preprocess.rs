//! A clean-room C preprocessor (translation phases 1–4/6).
//!
//! Written directly from the C standard (tenet T1): it consumes the raw source
//! of a translation unit, executes `#`-directives, expands macros (object- and
//! function-like, with `#`, `##`, and variadics) using a hide-set algorithm that
//! suppresses self-reference, evaluates conditional groups, and splices in
//! `#include`d files. Its output is the final [`Token`] stream the
//! [`crate::parse`]r consumes, so the preprocessor sits between raw text and the
//! parser without either of them knowing about the other.
//!
//! Provenance: every emitted token carries a [`Span`] into the *main* source
//! (file 0) so the existing diagnostic renderer and DWARF line map keep working
//! unchanged; tokens produced from an included file or a macro body are attributed
//! to their triggering construct in the main source, while `__LINE__`/`__FILE__`
//! report the true presumed location tracked during expansion.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use latticefoundry::support::diagnostics::{Diagnostic, FileId, Span};

use crate::ast::CType;
use crate::cstd::CStd;
use crate::lex::{self, Keyword, Punct, Token, TokenKind};

/// A `-D` / `-U` command-line macro operation, applied in order before the main
/// file is processed.
#[derive(Clone, Debug)]
pub enum MacroOp {
    /// `-D name` or `-D name=value` (an empty value defaults to `1`).
    Define(String),
    /// `-U name`.
    Undef(String),
}

/// Options controlling preprocessing.
#[derive(Clone, Debug)]
pub struct PpOptions {
    /// The selected C standard/dialect.
    pub std: CStd,
    /// `-I` search directories (searched for both `"…"` and `<…>` includes).
    pub include_dirs: Vec<PathBuf>,
    /// `-D` / `-U` command-line macros, in order.
    pub cmdline: Vec<MacroOp>,
    /// The main source file's name (used for `__FILE__`, diagnostics, and to
    /// resolve `"…"` includes relative to its directory).
    pub main_file_name: String,
}

impl Default for PpOptions {
    fn default() -> Self {
        PpOptions {
            std: CStd::default(),
            include_dirs: Vec::new(),
            cmdline: Vec::new(),
            main_file_name: "input.c".to_owned(),
        }
    }
}

/// Preprocess `main_source` into the final token stream for the parser.
pub fn preprocess(main_source: &str, opts: &PpOptions) -> Result<Vec<Token>, Vec<Diagnostic>> {
    let mut pp = Pp::new(opts.std, &opts.include_dirs, &opts.main_file_name, main_source.len());
    pp.define_predefined();
    pp.apply_cmdline(&opts.cmdline);

    let main_idx = 0u32;
    let toks = pp.lex_file(main_source, main_idx);
    pp.process_file(main_idx, toks);

    if pp.diags.iter().any(Diagnostic::is_error) {
        return Err(pp.diags);
    }
    let out = std::mem::take(&mut pp.out);
    let tokens = pp.finalize(out);
    if pp.diags.iter().any(Diagnostic::is_error) {
        return Err(pp.diags);
    }
    Ok(tokens)
}

/// The maximum `#include` nesting depth (cycle guard).
const INCLUDE_DEPTH_LIMIT: usize = 200;

/// A preprocessing token.
#[derive(Clone, Debug)]
struct PpTok {
    kind: PpKind,
    /// True presumed line for `__LINE__` (physical line; `#line`/`__LINE__`
    /// apply the active delta at use).
    line: u32,
    /// Owning pp-file index (0 = main); used for span remapping and `__FILE__`.
    file: u32,
    /// First token of a logical line (directive detection).
    bol: bool,
    /// Whitespace/comment preceded this token (stringize spacing).
    space_before: bool,
    /// Byte span within the owning file (remapped to the main source at finalize).
    span: Span,
    /// Blue-paint hide set: macros that must not re-expand this token.
    hideset: BTreeSet<String>,
}

/// The classification of a [`PpTok`].
#[derive(Clone, Debug, PartialEq)]
enum PpKind {
    Ident(String),
    /// A preprocessing number (raw spelling; parsed to an integer at finalize).
    Number(String),
    /// A character constant (raw spelling, including quotes).
    Char(String),
    /// A string literal (raw spelling, including quotes).
    Str(String),
    Punct(Punct),
    Hash,
    HashHash,
    /// A placemarker: the empty operand of `##`.
    Placemarker,
}

impl PpKind {
    /// The textual spelling of a token (for stringize and paste).
    fn spelling(&self) -> String {
        match self {
            PpKind::Ident(s) | PpKind::Number(s) | PpKind::Char(s) | PpKind::Str(s) => s.clone(),
            PpKind::Punct(p) => punct_spelling(*p).to_owned(),
            PpKind::Hash => "#".to_owned(),
            PpKind::HashHash => "##".to_owned(),
            PpKind::Placemarker => String::new(),
        }
    }
}

/// A macro definition.
#[derive(Clone, Debug)]
struct Macro {
    /// Parameter names, or `None` for an object-like macro.
    params: Option<Vec<String>>,
    /// Whether the (function-like) macro is variadic (`...`).
    variadic: bool,
    /// The replacement list.
    body: Vec<PpTok>,
}

/// One frame of the conditional-inclusion stack.
#[derive(Clone, Copy, Debug)]
struct Cond {
    /// Whether the current branch is emitting tokens.
    active: bool,
    /// Whether any branch of this `#if` has been taken.
    taken: bool,
    /// Whether the enclosing group is active.
    parent_active: bool,
    /// Whether `#else` has been seen.
    seen_else: bool,
}

/// The preprocessor state.
#[derive(Debug)]
struct Pp {
    std: CStd,
    include_dirs: Vec<PathBuf>,
    macros: HashMap<String, Macro>,
    /// Display names per pp-file (index = file id).
    filenames: Vec<String>,
    /// Filesystem paths per pp-file (for resolving relative includes).
    file_paths: Vec<PathBuf>,
    /// Main-source byte offset each pp-file's tokens are attributed to.
    include_site: Vec<u32>,
    /// Canonical paths guarded by `#pragma once`.
    pragma_once: HashSet<PathBuf>,
    out: Vec<PpTok>,
    diags: Vec<Diagnostic>,
    depth: usize,
    /// `#line` adjustment for the current file (presumed = physical + delta).
    line_delta: i64,
    /// `#line` filename override for the current file.
    file_override: Option<String>,
    main_len: u32,
}

impl Pp {
    fn new(std: CStd, include_dirs: &[PathBuf], main_name: &str, main_len: usize) -> Pp {
        Pp {
            std,
            include_dirs: include_dirs.to_vec(),
            macros: HashMap::new(),
            filenames: vec![main_name.to_owned()],
            file_paths: vec![PathBuf::from(main_name)],
            include_site: vec![0],
            pragma_once: HashSet::new(),
            out: Vec::new(),
            diags: Vec::new(),
            depth: 0,
            line_delta: 0,
            file_override: None,
            main_len: main_len as u32,
        }
    }

    fn error(&mut self, msg: impl Into<String>, span: Span) {
        self.diags.push(Diagnostic::error(msg).with_span(self.remap(span)));
    }

    /// Remap a token span (in its owning file) to the main source (file 0).
    fn remap(&self, sp: Span) -> Span {
        let idx = sp.file.index() as usize;
        if idx == 0 {
            Span::new(FileId::new(0), sp.start, sp.end)
        } else {
            let off = self.include_site.get(idx).copied().unwrap_or(0);
            Span::point(FileId::new(0), off)
        }
    }

    // --- predefined & command-line macros -----------------------------------

    fn define_object(&mut self, name: &str, value: &str) {
        let toks = self.lex_file(value, 0);
        // Re-tag as originating from the command line (file 0, offset 0).
        let body: Vec<PpTok> = toks
            .into_iter()
            .filter(|t| !matches!(t.kind, PpKind::Placemarker))
            .map(|mut t| {
                t.span = Span::point(FileId::new(0), 0);
                t.bol = false;
                t
            })
            .collect();
        self.macros.insert(name.to_owned(), Macro { params: None, variadic: false, body });
    }

    fn define_predefined(&mut self) {
        self.define_object("__STDC__", "1");
        self.define_object("__STDC_HOSTED__", "0");
        if let Some(v) = self.std.stdc_version() {
            self.define_object("__STDC_VERSION__", &format!("{v}L"));
        }
        // Target predefined macros for a freestanding x86-64 ELF/Linux target.
        for name in ["__x86_64__", "__amd64__", "__LP64__", "__linux__", "__unix__", "__ELF__"] {
            self.define_object(name, "1");
        }
        self.define_object("__DATE__", "\"Jan  1 2020\"");
        self.define_object("__TIME__", "\"00:00:00\"");
        if self.std.is_gnu() {
            self.define_object("__GNUC__", "13");
            self.define_object("__GNUC_MINOR__", "0");
            self.define_object("__GNUC_PATCHLEVEL__", "0");
        }
    }

    fn apply_cmdline(&mut self, ops: &[MacroOp]) {
        for op in ops {
            match op {
                MacroOp::Define(spec) => {
                    let (name, body) = match spec.split_once('=') {
                        Some((n, v)) => (n.to_owned(), v.to_owned()),
                        None => (spec.clone(), "1".to_owned()),
                    };
                    // Support function-like `-D 'f(x)=body'` by re-lexing.
                    let synth = format!("{name} {body}");
                    let toks = self.lex_file(&synth, 0);
                    let cleaned: Vec<PpTok> = toks
                        .into_iter()
                        .map(|mut t| {
                            t.span = Span::point(FileId::new(0), 0);
                            t
                        })
                        .collect();
                    self.add_define(&cleaned);
                }
                MacroOp::Undef(name) => {
                    self.macros.remove(name);
                }
            }
        }
    }

    // --- pp-lexer -----------------------------------------------------------

    fn lex_file(&mut self, text: &str, file_idx: u32) -> Vec<PpTok> {
        let bytes = text.as_bytes();
        let mut pos = 0usize;
        let mut line = 1u32;
        let mut bol = true;
        let mut out = Vec::new();

        loop {
            let space = self.skip_ws(bytes, &mut pos, &mut line, &mut bol, file_idx);
            if pos >= bytes.len() {
                break;
            }
            let start = pos;
            let c = bytes[pos];
            let kind = if c.is_ascii_digit()
                || (c == b'.' && bytes.get(pos + 1).is_some_and(u8::is_ascii_digit))
            {
                self.lex_ppnumber(bytes, &mut pos)
            } else if c == b'_' || c.is_ascii_alphabetic() {
                self.lex_ident(bytes, &mut pos)
            } else if c == b'"' {
                match self.lex_quoted(bytes, &mut pos, b'"') {
                    Some(raw) => PpKind::Str(raw),
                    None => {
                        self.diags.push(
                            Diagnostic::error("unterminated string literal")
                                .with_span(self.remap(Span::new(FileId::new(file_idx), start as u32, pos as u32))),
                        );
                        continue;
                    }
                }
            } else if c == b'\'' {
                match self.lex_quoted(bytes, &mut pos, b'\'') {
                    Some(raw) => PpKind::Char(raw),
                    None => {
                        self.diags.push(
                            Diagnostic::error("unterminated character constant")
                                .with_span(self.remap(Span::new(FileId::new(file_idx), start as u32, pos as u32))),
                        );
                        continue;
                    }
                }
            } else if let Some((k, len)) = match_punct(&bytes[pos..]) {
                pos += len;
                k
            } else {
                self.diags.push(
                    Diagnostic::error(format!("unexpected character '{}'", c as char))
                        .with_span(self.remap(Span::new(FileId::new(file_idx), start as u32, (start + 1) as u32))),
                );
                pos += 1;
                continue;
            };

            out.push(PpTok {
                kind,
                line,
                file: file_idx,
                bol,
                space_before: space,
                span: Span::new(FileId::new(file_idx), start as u32, pos as u32),
                hideset: BTreeSet::new(),
            });
            bol = false;
        }
        out
    }

    /// Skip whitespace, comments, and line splices; return whether anything was
    /// skipped (a preceding-whitespace flag). Updates `line` and `bol`.
    fn skip_ws(
        &mut self,
        bytes: &[u8],
        pos: &mut usize,
        line: &mut u32,
        bol: &mut bool,
        file_idx: u32,
    ) -> bool {
        let mut space = false;
        loop {
            let Some(&c) = bytes.get(*pos) else { return space };
            match c {
                b' ' | b'\t' | 0x0c | b'\r' => {
                    *pos += 1;
                    space = true;
                }
                b'\n' => {
                    *pos += 1;
                    *line += 1;
                    *bol = true;
                    space = true;
                }
                b'\\' if bytes.get(*pos + 1) == Some(&b'\n') => {
                    *pos += 2;
                    *line += 1;
                }
                b'\\' if bytes.get(*pos + 1) == Some(&b'\r')
                    && bytes.get(*pos + 2) == Some(&b'\n') =>
                {
                    *pos += 3;
                    *line += 1;
                }
                b'/' if bytes.get(*pos + 1) == Some(&b'/') => {
                    if !self.std.line_comments() {
                        let start = *pos;
                        self.diags.push(
                            Diagnostic::error(
                                "'//' line comments are a C99 feature (use -std=c99 or later)",
                            )
                            .with_span(self.remap(Span::new(
                                FileId::new(file_idx),
                                start as u32,
                                (start + 2) as u32,
                            ))),
                        );
                    }
                    *pos += 2;
                    while let Some(&d) = bytes.get(*pos) {
                        if d == b'\n' {
                            break;
                        }
                        *pos += 1;
                    }
                    space = true;
                }
                b'/' if bytes.get(*pos + 1) == Some(&b'*') => {
                    let start = *pos;
                    *pos += 2;
                    let mut closed = false;
                    while *pos < bytes.len() {
                        if bytes[*pos] == b'*' && bytes.get(*pos + 1) == Some(&b'/') {
                            *pos += 2;
                            closed = true;
                            break;
                        }
                        if bytes[*pos] == b'\n' {
                            *line += 1;
                            *bol = true;
                        }
                        *pos += 1;
                    }
                    if !closed {
                        self.diags.push(
                            Diagnostic::error("unterminated block comment").with_span(
                                self.remap(Span::new(FileId::new(file_idx), start as u32, *pos as u32)),
                            ),
                        );
                    }
                    space = true;
                }
                _ => return space,
            }
        }
    }

    fn lex_ident(&self, bytes: &[u8], pos: &mut usize) -> PpKind {
        let start = *pos;
        while let Some(&c) = bytes.get(*pos) {
            if c == b'_' || c.is_ascii_alphanumeric() {
                *pos += 1;
            } else {
                break;
            }
        }
        PpKind::Ident(String::from_utf8_lossy(&bytes[start..*pos]).into_owned())
    }

    fn lex_ppnumber(&self, bytes: &[u8], pos: &mut usize) -> PpKind {
        let start = *pos;
        // Initial digit or '.'.
        *pos += 1;
        while let Some(&c) = bytes.get(*pos) {
            if c.is_ascii_alphanumeric() || c == b'_' {
                *pos += 1;
                if matches!(c, b'e' | b'E' | b'p' | b'P')
                    && matches!(bytes.get(*pos), Some(b'+' | b'-'))
                {
                    *pos += 1;
                }
            } else if c == b'.' {
                *pos += 1;
            } else if c == b'\''
                && self.std.digit_separators()
                && bytes.get(*pos + 1).is_some_and(u8::is_ascii_alphanumeric)
            {
                *pos += 2;
            } else {
                break;
            }
        }
        PpKind::Number(String::from_utf8_lossy(&bytes[start..*pos]).into_owned())
    }

    /// Lex a quoted literal starting at `pos` (which is on the opening `quote`).
    /// Returns the raw spelling including quotes, or `None` if unterminated.
    fn lex_quoted(&self, bytes: &[u8], pos: &mut usize, quote: u8) -> Option<String> {
        let start = *pos;
        *pos += 1;
        loop {
            let &c = bytes.get(*pos)?;
            if c == b'\n' {
                return None;
            }
            if c == b'\\' {
                *pos += 1;
                bytes.get(*pos)?;
                *pos += 1;
                continue;
            }
            *pos += 1;
            if c == quote {
                return Some(String::from_utf8_lossy(&bytes[start..*pos]).into_owned());
            }
        }
    }

    // --- file processing ----------------------------------------------------

    fn process_file(&mut self, file_idx: u32, toks: Vec<PpTok>) {
        let dir = self
            .file_paths
            .get(file_idx as usize)
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let mut cond: Vec<Cond> = Vec::new();
        let mut run: Vec<PpTok> = Vec::new();
        let mut i = 0usize;

        while i < toks.len() {
            let start = i;
            let mut j = i + 1;
            while j < toks.len() && !toks[j].bol {
                j += 1;
            }
            let line = &toks[start..j];
            i = j;

            let active = cond.last().map(|c| c.active).unwrap_or(true);
            if line[0].bol && matches!(line[0].kind, PpKind::Hash) {
                if !run.is_empty() {
                    let r = std::mem::take(&mut run);
                    let e = self.expand(r);
                    self.out.extend(e);
                }
                self.handle_directive(file_idx, &dir, line, &mut cond);
            } else if active {
                run.extend(line.iter().cloned());
            }
        }

        if !run.is_empty() {
            let e = self.expand(std::mem::take(&mut run));
            self.out.extend(e);
        }
        if !cond.is_empty() {
            self.diags.push(Diagnostic::error("unterminated `#if` (missing `#endif`)"));
        }
    }

    fn handle_directive(&mut self, file_idx: u32, dir: &Path, line: &[PpTok], cond: &mut Vec<Cond>) {
        let active = cond.last().map(|c| c.active).unwrap_or(true);
        let dname: &str = match line.get(1).map(|t| &t.kind) {
            Some(PpKind::Ident(n)) => n.as_str(),
            Some(PpKind::Number(_)) => "\0linemarker",
            None => "",
            _ => "\0bad",
        };
        let dspan = line[0].span;
        match dname {
            "" => {}
            "if" => {
                let parent = active;
                let taken = parent && self.eval_if(&line[2..], &line[0]);
                cond.push(Cond { active: taken, taken, parent_active: parent, seen_else: false });
            }
            "ifdef" | "ifndef" => {
                let parent = active;
                let defined = self.first_ident_defined(&line[2..], dspan);
                let want = if dname == "ifdef" { defined } else { !defined };
                let taken = parent && want;
                cond.push(Cond { active: taken, taken, parent_active: parent, seen_else: false });
            }
            "elif" => {
                let Some(&Cond { seen_else, taken, parent_active, .. }) = cond.last() else {
                    self.error("`#elif` without `#if`", dspan);
                    return;
                };
                if seen_else {
                    self.error("`#elif` after `#else`", dspan);
                    return;
                }
                let (new_active, new_taken) = if taken {
                    (false, true)
                } else if parent_active {
                    let v = self.eval_if(&line[2..], &line[0]);
                    (v, v)
                } else {
                    (false, false)
                };
                let c = cond.last_mut().expect("checked");
                c.active = new_active;
                c.taken = new_taken;
            }
            "else" => {
                let Some(c) = cond.last_mut() else {
                    self.error("`#else` without `#if`", dspan);
                    return;
                };
                if c.seen_else {
                    self.error("`#else` after `#else`", dspan);
                    return;
                }
                c.seen_else = true;
                c.active = c.parent_active && !c.taken;
                c.taken = true;
            }
            "endif" => {
                if cond.pop().is_none() {
                    self.error("`#endif` without `#if`", dspan);
                }
            }
            _ if !active => {}
            "define" => self.add_define(&line[2..]),
            "undef" => {
                if let Some(PpKind::Ident(n)) = line.get(2).map(|t| &t.kind) {
                    self.macros.remove(n);
                } else {
                    self.error("`#undef` expects an identifier", dspan);
                }
            }
            "include" => self.do_include(file_idx, dir, &line[2..], dspan),
            "error" => {
                let msg = spell_line(&line[2..]);
                self.error(format!("#error {msg}"), dspan);
            }
            "warning" => {
                let msg = spell_line(&line[2..]);
                self.diags
                    .push(Diagnostic::warning(format!("#warning {msg}")).with_span(self.remap(dspan)));
            }
            "pragma" => {
                if let Some(PpKind::Ident(n)) = line.get(2).map(|t| &t.kind)
                    && n == "once"
                    && let Some(p) = self.file_paths.get(file_idx as usize).cloned()
                {
                    let canon = std::fs::canonicalize(&p).unwrap_or(p);
                    self.pragma_once.insert(canon);
                }
            }
            "line" => self.handle_line(&line[2..], line[0].line, dspan),
            "\0linemarker" => self.handle_line(&line[1..], line[0].line, dspan),
            other => self.error(format!("invalid preprocessing directive #{other}"), dspan),
        }
    }

    fn handle_line(&mut self, args: &[PpTok], phys_line: u32, dspan: Span) {
        let expanded = self.expand(args.to_vec());
        let Some(first) = expanded.first() else {
            self.error("`#line` expects a line number", dspan);
            return;
        };
        let n = match &first.kind {
            PpKind::Number(t) => match t.parse::<i64>() {
                Ok(v) => v,
                Err(_) => {
                    self.error("`#line` expects a decimal line number", dspan);
                    return;
                }
            },
            _ => {
                self.error("`#line` expects a line number", dspan);
                return;
            }
        };
        self.line_delta = n - (phys_line as i64 + 1);
        if let Some(PpKind::Str(raw)) = expanded.get(1).map(|t| &t.kind) {
            self.file_override = Some(decode_string(raw));
        }
    }

    fn first_ident_defined(&mut self, args: &[PpTok], dspan: Span) -> bool {
        match args.first().map(|t| &t.kind) {
            Some(PpKind::Ident(n)) => self.is_defined(n),
            _ => {
                self.error("expected an identifier after `#ifdef`/`#ifndef`", dspan);
                false
            }
        }
    }

    fn is_defined(&self, name: &str) -> bool {
        name == "__LINE__" || name == "__FILE__" || self.macros.contains_key(name)
    }

    fn do_include(&mut self, file_idx: u32, dir: &Path, args: &[PpTok], dspan: Span) {
        let (name, angled) = match self.parse_header_name(args) {
            Some(v) => v,
            None => {
                self.error("`#include` expects \"file\" or <file>", dspan);
                return;
            }
        };
        let Some(path) = self.resolve_include(&name, angled, dir) else {
            self.error(format!("cannot find include file {name:?}"), dspan);
            return;
        };
        let canon = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if self.pragma_once.contains(&canon) {
            return;
        }
        if self.depth >= INCLUDE_DEPTH_LIMIT {
            self.error("`#include` nested too deeply (cyclic include?)", dspan);
            return;
        }
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                self.error(format!("cannot read include file {path:?}: {e}"), dspan);
                return;
            }
        };

        let attribution =
            if file_idx == 0 { dspan.start } else { self.include_site[file_idx as usize] };
        let new_idx = self.filenames.len() as u32;
        self.filenames.push(path.display().to_string());
        self.file_paths.push(path);
        self.include_site.push(attribution);

        let toks = self.lex_file(&text, new_idx);
        let saved_delta = self.line_delta;
        let saved_override = self.file_override.take();
        self.line_delta = 0;
        self.depth += 1;
        self.process_file(new_idx, toks);
        self.depth -= 1;
        self.line_delta = saved_delta;
        self.file_override = saved_override;
    }

    /// Interpret the tokens after `#include` as a header name; macro-expand if it
    /// is neither `"…"` nor `<…>`.
    fn parse_header_name(&mut self, args: &[PpTok]) -> Option<(String, bool)> {
        if let Some(first) = args.first() {
            match &first.kind {
                PpKind::Str(raw) => return Some((decode_string(raw), false)),
                PpKind::Punct(Punct::Lt) => return Some((join_angle(&args[1..]), true)),
                _ => {}
            }
        }
        let expanded = self.expand(args.to_vec());
        match expanded.first().map(|t| &t.kind) {
            Some(PpKind::Str(raw)) => Some((decode_string(raw), false)),
            Some(PpKind::Punct(Punct::Lt)) => Some((join_angle(&expanded[1..]), true)),
            _ => None,
        }
    }

    fn resolve_include(&self, name: &str, angled: bool, dir: &Path) -> Option<PathBuf> {
        if !angled {
            let cand = dir.join(name);
            if cand.is_file() {
                return Some(cand);
            }
        }
        for inc in &self.include_dirs {
            let cand = inc.join(name);
            if cand.is_file() {
                return Some(cand);
            }
        }
        None
    }

    fn add_define(&mut self, toks: &[PpTok]) {
        let name = match toks.first().map(|t| &t.kind) {
            Some(PpKind::Ident(n)) => n.clone(),
            _ => {
                if let Some(t) = toks.first() {
                    self.error("macro name must be an identifier", t.span);
                }
                return;
            }
        };
        let rest = &toks[1..];
        if let Some(first) = rest.first()
            && matches!(first.kind, PpKind::Punct(Punct::LParen))
            && !first.space_before
        {
            if let Some((params, variadic, body_start)) = self.parse_params(rest) {
                let body = clean_body(&rest[body_start..]);
                self.macros.insert(name, Macro { params: Some(params), variadic, body });
            }
            return;
        }
        let body = clean_body(rest);
        self.macros.insert(name, Macro { params: None, variadic: false, body });
    }

    /// Parse a function-like parameter list starting at `rest[0] == '('`.
    /// Returns the parameters, whether variadic, and the body start index.
    fn parse_params(&mut self, rest: &[PpTok]) -> Option<(Vec<String>, bool, usize)> {
        let mut params = Vec::new();
        let mut variadic = false;
        let mut i = 1usize; // skip '('
        if matches!(rest.get(i).map(|t| &t.kind), Some(PpKind::Punct(Punct::RParen))) {
            return Some((params, variadic, i + 1));
        }
        loop {
            match rest.get(i).map(|t| &t.kind) {
                Some(PpKind::Punct(Punct::Ellipsis)) => {
                    variadic = true;
                    i += 1;
                    break;
                }
                Some(PpKind::Ident(n)) => {
                    params.push(n.clone());
                    i += 1;
                }
                _ => {
                    let sp = rest.get(i).map(|t| t.span).unwrap_or(rest[0].span);
                    self.error("expected a macro parameter name", sp);
                    return None;
                }
            }
            match rest.get(i).map(|t| &t.kind) {
                Some(PpKind::Punct(Punct::Comma)) => i += 1,
                Some(PpKind::Punct(Punct::RParen)) => break,
                _ => {
                    let sp = rest.get(i).map(|t| t.span).unwrap_or(rest[0].span);
                    self.error("expected ',' or ')' in macro parameter list", sp);
                    return None;
                }
            }
        }
        if !matches!(rest.get(i).map(|t| &t.kind), Some(PpKind::Punct(Punct::RParen))) {
            let sp = rest.get(i).map(|t| t.span).unwrap_or(rest[0].span);
            self.error("expected ')' to close macro parameter list", sp);
            return None;
        }
        Some((params, variadic, i + 1))
    }

    // --- macro expansion ----------------------------------------------------

    fn expand(&mut self, input: Vec<PpTok>) -> Vec<PpTok> {
        let mut input: VecDeque<PpTok> = VecDeque::from(input);
        let mut out: Vec<PpTok> = Vec::new();
        let mut guard = 0usize;

        while let Some(t) = input.pop_front() {
            guard += 1;
            if guard > 5_000_000 {
                self.diags.push(Diagnostic::error("macro expansion did not terminate"));
                break;
            }
            let PpKind::Ident(name) = &t.kind else {
                out.push(t);
                continue;
            };
            let name = name.clone();
            if t.hideset.contains(&name) {
                out.push(t);
                continue;
            }
            if name == "__LINE__" {
                out.push(self.make_line_tok(&t));
                continue;
            }
            if name == "__FILE__" {
                out.push(self.make_file_tok(&t));
                continue;
            }
            let Some(mac) = self.macros.get(&name).cloned() else {
                out.push(t);
                continue;
            };
            match &mac.params {
                None => {
                    let mut hs = t.hideset.clone();
                    hs.insert(name.clone());
                    let repl = self.subst(&mac.body, &[], &[], false, &hs, &t);
                    for tok in repl.into_iter().rev() {
                        input.push_front(tok);
                    }
                }
                Some(params) => {
                    if !matches!(input.front().map(|x| &x.kind), Some(PpKind::Punct(Punct::LParen))) {
                        out.push(t);
                        continue;
                    }
                    let Some((args, close_hs)) =
                        self.gather_args(&mut input, params.len(), mac.variadic, &t)
                    else {
                        out.push(t);
                        continue;
                    };
                    let mut hs: BTreeSet<String> =
                        t.hideset.intersection(&close_hs).cloned().collect();
                    hs.insert(name.clone());
                    let repl = self.subst(&mac.body, params, &args, mac.variadic, &hs, &t);
                    for tok in repl.into_iter().rev() {
                        input.push_front(tok);
                    }
                }
            }
        }
        out
    }

    /// Collect the arguments of a function-like macro call. On entry the front of
    /// `input` is the opening `(`. Returns the argument token lists (one per named
    /// parameter, plus a trailing list for `...`) and the closing `)`'s hide set.
    fn gather_args(
        &mut self,
        input: &mut VecDeque<PpTok>,
        nparams: usize,
        variadic: bool,
        inv: &PpTok,
    ) -> Option<(Vec<Vec<PpTok>>, BTreeSet<String>)> {
        input.pop_front(); // consume '('
        let mut args: Vec<Vec<PpTok>> = Vec::new();
        let mut cur: Vec<PpTok> = Vec::new();
        let mut depth = 0usize;
        let close_hs;
        loop {
            let Some(tok) = input.pop_front() else {
                self.error("unterminated macro argument list", inv.span);
                return None;
            };
            match &tok.kind {
                PpKind::Punct(Punct::LParen) => {
                    depth += 1;
                    cur.push(tok);
                }
                PpKind::Punct(Punct::RParen) => {
                    if depth == 0 {
                        close_hs = tok.hideset.clone();
                        break;
                    }
                    depth -= 1;
                    cur.push(tok);
                }
                PpKind::Punct(Punct::Comma)
                    if depth == 0 && !(variadic && args.len() == nparams) =>
                {
                    args.push(std::mem::take(&mut cur));
                }
                _ => cur.push(tok),
            }
        }
        args.push(cur);

        // Normalize `F()` for a zero-parameter, non-variadic macro to zero args.
        if nparams == 0 && !variadic && args.len() == 1 && args[0].is_empty() {
            args.clear();
        }
        // Ensure a (possibly empty) variadic slot exists.
        if variadic && args.len() == nparams {
            args.push(Vec::new());
        }

        let expected = nparams;
        let got = if variadic { args.len().saturating_sub(1) } else { args.len() };
        if (variadic && got < expected) || (!variadic && args.len() != expected) {
            self.error(
                format!(
                    "macro invoked with {} argument(s) but expects {}{}",
                    if variadic { got } else { args.len() },
                    expected,
                    if variadic { " or more" } else { "" }
                ),
                inv.span,
            );
            return None;
        }
        Some((args, close_hs))
    }

    /// Substitute a macro body: parameter replacement, `#` stringize, `##` paste,
    /// then apply the hide set `hs` and invocation provenance to the result.
    fn subst(
        &mut self,
        body: &[PpTok],
        params: &[String],
        args: &[Vec<PpTok>],
        variadic: bool,
        hs: &BTreeSet<String>,
        inv: &PpTok,
    ) -> Vec<PpTok> {
        let param_index = |name: &str| -> Option<usize> {
            if let Some(p) = params.iter().position(|p| p == name) {
                Some(p)
            } else if variadic && name == "__VA_ARGS__" {
                Some(params.len())
            } else {
                None
            }
        };
        let arg_of = |t: &PpKind| -> Option<&Vec<PpTok>> {
            if let PpKind::Ident(n) = t {
                param_index(n).and_then(|i| args.get(i))
            } else {
                None
            }
        };

        let mut res: Vec<PpTok> = Vec::new();
        let mut i = 0usize;
        while i < body.len() {
            let t = &body[i];

            // `#` stringize (function-like only).
            if (!params.is_empty() || variadic)
                && matches!(t.kind, PpKind::Hash)
                && let Some(arg) = body.get(i + 1).and_then(|n| arg_of(&n.kind))
            {
                res.push(stringize(arg, inv));
                i += 2;
                continue;
            }

            // `##` paste.
            if matches!(t.kind, PpKind::HashHash)
                && let Some(next) = body.get(i + 1)
            {
                let is_va = matches!(&next.kind, PpKind::Ident(n) if n == "__VA_ARGS__");
                let rhs: Vec<PpTok> = match arg_of(&next.kind) {
                    Some(a) => a.clone(),
                    None => vec![next.clone()],
                };
                self.paste_into(&mut res, rhs, is_va, inv);
                i += 2;
                continue;
            }

            // Parameter immediately followed by `##`: substitute unexpanded.
            if body.get(i + 1).is_some_and(|n| matches!(n.kind, PpKind::HashHash))
                && let Some(arg) = arg_of(&t.kind)
            {
                if arg.is_empty() {
                    res.push(placemarker(inv));
                } else {
                    res.extend(arg.iter().cloned());
                }
                i += 1;
                continue;
            }

            // Plain parameter: substitute the fully-expanded argument.
            if let Some(arg) = arg_of(&t.kind) {
                let expanded = self.expand(arg.clone());
                res.extend(expanded);
                i += 1;
                continue;
            }

            res.push(t.clone());
            i += 1;
        }

        for tok in &mut res {
            if matches!(tok.kind, PpKind::Placemarker) {
                continue;
            }
            let mut new_hs = tok.hideset.clone();
            new_hs.extend(hs.iter().cloned());
            tok.hideset = new_hs;
            tok.span = inv.span;
            tok.line = inv.line;
            tok.file = inv.file;
            tok.bol = false;
        }
        res.retain(|t| !matches!(t.kind, PpKind::Placemarker));
        res
    }

    fn paste_into(&mut self, res: &mut Vec<PpTok>, rhs: Vec<PpTok>, is_va: bool, inv: &PpTok) {
        // GNU `, ## __VA_ARGS__` comma elision when the variadic args are empty.
        if is_va && rhs.is_empty() {
            if matches!(res.last().map(|t| &t.kind), Some(PpKind::Punct(Punct::Comma))) {
                res.pop();
            }
            return;
        }
        let rhs_empty = rhs.is_empty() || rhs.iter().all(|t| matches!(t.kind, PpKind::Placemarker));
        let lhs = res.pop();
        match (lhs, rhs_empty) {
            (None, _) => res.extend(rhs),
            (Some(l), true) => {
                if !matches!(l.kind, PpKind::Placemarker) {
                    res.push(l);
                }
            }
            (Some(l), false) => {
                let mut it = rhs.into_iter();
                let first = it.next().expect("non-empty");
                if matches!(l.kind, PpKind::Placemarker) {
                    res.push(first);
                } else {
                    let pasted = self.paste_tokens(&l, &first, inv);
                    res.push(pasted);
                }
                res.extend(it);
            }
        }
    }

    fn paste_tokens(&mut self, a: &PpTok, b: &PpTok, inv: &PpTok) -> PpTok {
        let spelling = format!("{}{}", a.kind.spelling(), b.kind.spelling());
        let lexed = self.lex_file(&spelling, inv.file);
        let real: Vec<&PpTok> =
            lexed.iter().filter(|t| !matches!(t.kind, PpKind::Placemarker)).collect();
        if real.len() != 1 {
            self.error(
                format!("pasting \"{}\" and \"{}\" does not form a valid token", a.kind.spelling(), b.kind.spelling()),
                inv.span,
            );
        }
        let kind = real.first().map(|t| t.kind.clone()).unwrap_or(PpKind::Placemarker);
        PpTok {
            kind,
            line: inv.line,
            file: inv.file,
            bol: false,
            space_before: false,
            span: inv.span,
            hideset: a.hideset.intersection(&b.hideset).cloned().collect(),
        }
    }

    fn make_line_tok(&self, t: &PpTok) -> PpTok {
        let value = (t.line as i64 + self.line_delta).max(0);
        PpTok { kind: PpKind::Number(value.to_string()), bol: false, ..t.clone() }
    }

    fn make_file_tok(&self, t: &PpTok) -> PpTok {
        let name = self
            .file_override
            .clone()
            .unwrap_or_else(|| self.filenames.get(t.file as usize).cloned().unwrap_or_default());
        PpTok { kind: PpKind::Str(format!("\"{}\"", escape_string(&name))), bol: false, ..t.clone() }
    }

    // --- #if constant expression --------------------------------------------

    fn eval_if(&mut self, toks: &[PpTok], at: &PpTok) -> bool {
        let replaced = self.replace_defined(toks, at.span);
        let expanded = self.expand(replaced);
        match self.eval_const_expr(&expanded, at.span) {
            Ok(v) => v != 0,
            Err(d) => {
                self.diags.push(d);
                false
            }
        }
    }

    /// Replace `defined X` / `defined(X)` with `1` or `0` before expansion.
    fn replace_defined(&mut self, toks: &[PpTok], at: Span) -> Vec<PpTok> {
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < toks.len() {
            let t = &toks[i];
            if matches!(&t.kind, PpKind::Ident(n) if n == "defined") {
                let (name, consumed) = match toks.get(i + 1).map(|x| &x.kind) {
                    Some(PpKind::Ident(n)) => (Some(n.clone()), 2),
                    Some(PpKind::Punct(Punct::LParen)) => {
                        match (toks.get(i + 2).map(|x| &x.kind), toks.get(i + 3).map(|x| &x.kind)) {
                            (Some(PpKind::Ident(n)), Some(PpKind::Punct(Punct::RParen))) => {
                                (Some(n.clone()), 4)
                            }
                            _ => (None, 1),
                        }
                    }
                    _ => (None, 1),
                };
                match name {
                    Some(n) => {
                        let v = if self.is_defined(&n) { "1" } else { "0" };
                        out.push(number_tok(v, t));
                        i += consumed;
                        continue;
                    }
                    None => {
                        self.error("operator `defined` requires an identifier", at);
                        i += consumed;
                        continue;
                    }
                }
            }
            out.push(t.clone());
            i += 1;
        }
        out
    }

    fn eval_const_expr(&self, toks: &[PpTok], at: Span) -> Result<i128, Diagnostic> {
        let mut items: Vec<EItem> = Vec::new();
        for t in toks {
            let item = match &t.kind {
                PpKind::Number(s) => match eval_number(s, self.std) {
                    Ok((v, _)) => EItem::Num(v),
                    Err(m) => return Err(Diagnostic::error(m).with_span(self.remap(t.span))),
                },
                PpKind::Char(raw) => EItem::Num(eval_char(raw)),
                PpKind::Ident(n) => {
                    // Remaining identifiers evaluate to 0 (C23 `true` is 1).
                    EItem::Num(i128::from(self.std.is_c23() && n == "true"))
                }
                PpKind::Punct(p) => EItem::Op(*p),
                PpKind::Str(_) => {
                    return Err(Diagnostic::error("string literal in `#if` expression")
                        .with_span(self.remap(t.span)));
                }
                PpKind::Hash | PpKind::HashHash | PpKind::Placemarker => {
                    return Err(Diagnostic::error("invalid token in `#if` expression")
                        .with_span(self.remap(t.span)));
                }
            };
            items.push(item);
        }
        let mut ev = Ev { items: &items, pos: 0, at: self.remap(at) };
        let v = ev.expr()?;
        if ev.pos != ev.items.len() {
            return Err(Diagnostic::error("trailing tokens in `#if` expression").with_span(ev.at));
        }
        Ok(v)
    }

    // --- finalize -----------------------------------------------------------

    fn finalize(&mut self, toks: Vec<PpTok>) -> Vec<Token> {
        let mut out = Vec::with_capacity(toks.len() + 1);
        for t in toks {
            let span = self.remap(t.span);
            let kind = match &t.kind {
                PpKind::Ident(name) => match self.classify_ident(name, span) {
                    Some(k) => k,
                    None => continue,
                },
                PpKind::Number(s) => match eval_number(s, self.std) {
                    Ok((v, ty)) => TokenKind::IntLit(v, ty),
                    Err(m) => {
                        self.diags.push(Diagnostic::error(m).with_span(span));
                        continue;
                    }
                },
                PpKind::Char(raw) => TokenKind::IntLit(eval_char(raw), CType::int()),
                PpKind::Str(raw) => TokenKind::Str(decode_string(raw)),
                PpKind::Punct(p) => TokenKind::Punct(*p),
                PpKind::Hash | PpKind::HashHash => {
                    self.diags.push(Diagnostic::error("stray '#' in program").with_span(span));
                    continue;
                }
                PpKind::Placemarker => continue,
            };
            out.push(Token { kind, span });
        }
        out.push(Token { kind: TokenKind::Eof, span: Span::point(FileId::new(0), self.main_len) });
        out
    }

    /// Classify an identifier into a keyword/literal token, applying the standard
    /// gating. Returns `None` (dropping the token) only when a gating error is
    /// recorded.
    fn classify_ident(&mut self, name: &str, span: Span) -> Option<TokenKind> {
        // Base C89 keywords.
        if let Some(kw) = base_keyword(name) {
            return Some(TokenKind::Keyword(kw));
        }
        match name {
            "_Bool" => {
                if self.std.has_bool_type() {
                    Some(TokenKind::Keyword(Keyword::Bool))
                } else {
                    self.diags.push(
                        Diagnostic::error("`_Bool` is a C99 feature (use -std=c99 or later)")
                            .with_span(span),
                    );
                    None
                }
            }
            "restrict" if self.std.inline_restrict() => Some(TokenKind::Keyword(Keyword::Restrict)),
            "inline" if self.std.inline_restrict() => Some(TokenKind::Keyword(Keyword::Inline)),
            "_Noreturn" => self.gate_reserved(name, self.std.static_assert_generic(), "C11", Keyword::Noreturn, span),
            "_Alignof" => self.gate_reserved(name, self.std.static_assert_generic(), "C11", Keyword::Alignof, span),
            "_Static_assert" => self.gate_reserved(name, self.std.static_assert_generic(), "C11", Keyword::StaticAssert, span),
            "_Generic" => self.gate_reserved(name, self.std.static_assert_generic(), "C11", Keyword::Generic, span),
            "bool" if self.std.bool_keyword() => Some(TokenKind::Keyword(Keyword::Bool)),
            "true" if self.std.bool_keyword() => Some(TokenKind::IntLit(1, CType::int())),
            "false" if self.std.bool_keyword() => Some(TokenKind::IntLit(0, CType::int())),
            "nullptr" if self.std.nullptr_keyword() => Some(TokenKind::IntLit(0, CType::int())),
            _ => Some(TokenKind::Ident(name.to_owned())),
        }
    }

    fn gate_reserved(
        &mut self,
        name: &str,
        available: bool,
        since: &str,
        kw: Keyword,
        span: Span,
    ) -> Option<TokenKind> {
        if available {
            Some(TokenKind::Keyword(kw))
        } else {
            self.diags.push(
                Diagnostic::error(format!(
                    "`{name}` is a {since} feature (use -std={} or later)",
                    since.to_ascii_lowercase()
                ))
                .with_span(span),
            );
            None
        }
    }
}

// --- free helpers -----------------------------------------------------------

/// The evaluator's simplified token.
#[derive(Clone, Copy, Debug)]
enum EItem {
    Num(i128),
    Op(Punct),
}

/// A recursive-descent evaluator for `#if` constant integer expressions.
struct Ev<'a> {
    items: &'a [EItem],
    pos: usize,
    at: Span,
}

impl Ev<'_> {
    fn peek(&self) -> Option<Punct> {
        match self.items.get(self.pos) {
            Some(EItem::Op(p)) => Some(*p),
            _ => None,
        }
    }

    fn err(&self, msg: &str) -> Diagnostic {
        Diagnostic::error(msg.to_owned()).with_span(self.at)
    }

    fn expr(&mut self) -> Result<i128, Diagnostic> {
        self.ternary()
    }

    fn ternary(&mut self) -> Result<i128, Diagnostic> {
        let c = self.binary(0)?;
        if self.peek() == Some(Punct::Question) {
            self.pos += 1;
            let t = self.expr()?;
            if self.peek() != Some(Punct::Colon) {
                return Err(self.err("expected ':' in `#if` conditional"));
            }
            self.pos += 1;
            let e = self.ternary()?;
            Ok(if c != 0 { t } else { e })
        } else {
            Ok(c)
        }
    }

    fn binary(&mut self, min_prec: u8) -> Result<i128, Diagnostic> {
        let mut lhs = self.unary()?;
        while let Some((prec, op)) = self.peek().and_then(binop_prec) {
            if prec < min_prec {
                break;
            }
            self.pos += 1;
            let rhs = self.binary(prec + 1)?;
            lhs = apply_binop(op, lhs, rhs, self)?;
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<i128, Diagnostic> {
        match self.peek() {
            Some(Punct::Minus) => {
                self.pos += 1;
                Ok(self.unary()?.wrapping_neg())
            }
            Some(Punct::Plus) => {
                self.pos += 1;
                self.unary()
            }
            Some(Punct::Bang) => {
                self.pos += 1;
                Ok(i128::from(self.unary()? == 0))
            }
            Some(Punct::Tilde) => {
                self.pos += 1;
                Ok(!self.unary()?)
            }
            _ => self.primary(),
        }
    }

    fn primary(&mut self) -> Result<i128, Diagnostic> {
        match self.items.get(self.pos) {
            Some(EItem::Num(v)) => {
                self.pos += 1;
                Ok(*v)
            }
            Some(EItem::Op(Punct::LParen)) => {
                self.pos += 1;
                let v = self.expr()?;
                if self.peek() != Some(Punct::RParen) {
                    return Err(self.err("expected ')' in `#if` expression"));
                }
                self.pos += 1;
                Ok(v)
            }
            _ => Err(self.err("expected a value in `#if` expression")),
        }
    }
}

fn binop_prec(p: Punct) -> Option<(u8, Punct)> {
    let prec = match p {
        Punct::PipePipe => 0,
        Punct::AmpAmp => 1,
        Punct::Pipe => 2,
        Punct::Caret => 3,
        Punct::Amp => 4,
        Punct::EqEq | Punct::Ne => 5,
        Punct::Lt | Punct::Le | Punct::Gt | Punct::Ge => 6,
        Punct::Shl | Punct::Shr => 7,
        Punct::Plus | Punct::Minus => 8,
        Punct::Star | Punct::Slash | Punct::Percent => 9,
        _ => return None,
    };
    Some((prec, p))
}

fn apply_binop(op: Punct, a: i128, b: i128, ev: &Ev<'_>) -> Result<i128, Diagnostic> {
    Ok(match op {
        Punct::PipePipe => i128::from(a != 0 || b != 0),
        Punct::AmpAmp => i128::from(a != 0 && b != 0),
        Punct::Pipe => a | b,
        Punct::Caret => a ^ b,
        Punct::Amp => a & b,
        Punct::EqEq => i128::from(a == b),
        Punct::Ne => i128::from(a != b),
        Punct::Lt => i128::from(a < b),
        Punct::Le => i128::from(a <= b),
        Punct::Gt => i128::from(a > b),
        Punct::Ge => i128::from(a >= b),
        Punct::Shl => a.wrapping_shl(b as u32),
        Punct::Shr => a.wrapping_shr(b as u32),
        Punct::Plus => a.wrapping_add(b),
        Punct::Minus => a.wrapping_sub(b),
        Punct::Star => a.wrapping_mul(b),
        Punct::Slash => {
            if b == 0 {
                return Err(ev.err("division by zero in `#if` expression"));
            }
            a.wrapping_div(b)
        }
        Punct::Percent => {
            if b == 0 {
                return Err(ev.err("division by zero in `#if` expression"));
            }
            a.wrapping_rem(b)
        }
        _ => return Err(ev.err("unsupported operator in `#if` expression")),
    })
}

/// The C89 keyword set (revision-independent base).
fn base_keyword(word: &str) -> Option<Keyword> {
    Some(match word {
        "void" => Keyword::Void,
        "char" => Keyword::Char,
        "short" => Keyword::Short,
        "int" => Keyword::Int,
        "long" => Keyword::Long,
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
        "sizeof" => Keyword::Sizeof,
        _ => return None,
    })
}

/// Strip leading whitespace flags from a macro body (does not alter tokens).
fn clean_body(rest: &[PpTok]) -> Vec<PpTok> {
    let mut body = rest.to_vec();
    if let Some(first) = body.first_mut() {
        first.space_before = false;
        first.bol = false;
    }
    body
}

fn number_tok(text: &str, from: &PpTok) -> PpTok {
    PpTok { kind: PpKind::Number(text.to_owned()), bol: false, ..from.clone() }
}

fn placemarker(inv: &PpTok) -> PpTok {
    PpTok {
        kind: PpKind::Placemarker,
        line: inv.line,
        file: inv.file,
        bol: false,
        space_before: false,
        span: inv.span,
        hideset: BTreeSet::new(),
    }
}

/// Build a stringized string token from an argument's tokens.
fn stringize(arg: &[PpTok], inv: &PpTok) -> PpTok {
    let mut inner = String::new();
    for (i, t) in arg.iter().enumerate() {
        if matches!(t.kind, PpKind::Placemarker) {
            continue;
        }
        if i != 0 && t.space_before {
            inner.push(' ');
        }
        let sp = t.kind.spelling();
        match &t.kind {
            PpKind::Str(_) | PpKind::Char(_) => {
                for ch in sp.chars() {
                    if ch == '"' || ch == '\\' {
                        inner.push('\\');
                    }
                    inner.push(ch);
                }
            }
            _ => inner.push_str(&sp),
        }
    }
    PpTok {
        kind: PpKind::Str(format!("\"{inner}\"")),
        line: inv.line,
        file: inv.file,
        bol: false,
        space_before: false,
        span: inv.span,
        hideset: BTreeSet::new(),
    }
}

/// Join `<…>` header-name tokens into a single path string.
fn join_angle(toks: &[PpTok]) -> String {
    let mut s = String::new();
    for t in toks {
        if matches!(t.kind, PpKind::Punct(Punct::Gt)) {
            break;
        }
        s.push_str(&t.kind.spelling());
    }
    s
}

/// Spell a run of tokens (for `#error`/`#warning` messages).
fn spell_line(toks: &[PpTok]) -> String {
    let mut s = String::new();
    for (i, t) in toks.iter().enumerate() {
        if i != 0 && t.space_before {
            s.push(' ');
        }
        s.push_str(&t.kind.spelling());
    }
    s
}

/// Decode a string literal spelling (strip quotes, unescape).
fn decode_string(raw: &str) -> String {
    let inner = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(raw);
    unescape(inner)
}

fn unescape(inner: &str) -> String {
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some('\\') => out.push('\\'),
            Some('\'') => out.push('\''),
            Some('"') => out.push('"'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

fn escape_string(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c == '"' || c == '\\' {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Evaluate a character constant's spelling (including quotes) to its value.
fn eval_char(raw: &str) -> i128 {
    let inner = raw.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')).unwrap_or(raw);
    let bytes = inner.as_bytes();
    if bytes.is_empty() {
        return 0;
    }
    if bytes[0] == b'\\' {
        return match bytes.get(1).copied() {
            Some(b'n') => 10,
            Some(b't') => 9,
            Some(b'r') => 13,
            Some(b'0') => 0,
            Some(b'\\') => 92,
            Some(b'\'') => 39,
            Some(b'"') => 34,
            Some(b'a') => 7,
            Some(b'b') => 8,
            Some(b'f') => 12,
            Some(b'v') => 11,
            Some(b'x') => {
                let hex: String = inner[2..].chars().take_while(|c| c.is_ascii_hexdigit()).collect();
                i128::from_str_radix(&hex, 16).unwrap_or(0)
            }
            Some(other) => i128::from(other),
            None => 0,
        };
    }
    i128::from(bytes[0])
}

/// Parse a preprocessing number into an integer value and its C type, honoring
/// C23 binary literals and digit separators (both gated by `std`).
fn eval_number(text: &str, std: CStd) -> Result<(i128, CType), String> {
    if text.contains('\'') && !std.digit_separators() {
        return Err("digit separators are a C23 feature (use -std=c23)".to_owned());
    }
    let s: String =
        if std.digit_separators() { text.chars().filter(|&c| c != '\'').collect() } else { text.to_owned() };
    let lower = s.to_ascii_lowercase();
    let b = s.as_bytes();

    let (radix, digits_start) = if lower.starts_with("0x") {
        (16u32, 2usize)
    } else if lower.starts_with("0b") {
        if !std.binary_literals() {
            return Err("binary literals are a C23 feature (use -std=c23)".to_owned());
        }
        (2, 2)
    } else if b.len() > 1 && b[0] == b'0' {
        (8, 1)
    } else {
        (10, 0)
    };

    let mut i = digits_start;
    while i < b.len() {
        let c = b[i];
        let ok = match radix {
            16 => c.is_ascii_hexdigit(),
            8 => (b'0'..=b'7').contains(&c),
            2 => c == b'0' || c == b'1',
            _ => c.is_ascii_digit(),
        };
        if ok {
            i += 1;
        } else {
            break;
        }
    }

    if let Some(&c) = b.get(i) {
        let is_float = c == b'.'
            || (radix == 10 && matches!(c, b'e' | b'E'))
            || (radix == 16 && matches!(c, b'p' | b'P'));
        if is_float {
            return Err("floating-point constants are out of scope in this C subset".to_owned());
        }
    }

    let digits = &s[digits_start..i];
    let mut unsigned = false;
    let mut long = false;
    let mut j = i;
    while j < b.len() {
        match b[j] {
            b'u' | b'U' => {
                if unsigned {
                    return Err("invalid integer suffix".to_owned());
                }
                unsigned = true;
                j += 1;
            }
            b'l' | b'L' => {
                long = true;
                j += 1;
                if matches!(b.get(j), Some(b'l' | b'L')) {
                    j += 1;
                }
            }
            _ => return Err(format!("invalid suffix on integer constant: {}", &s[i..])),
        }
    }

    let for_parse = if radix == 8 && digits.is_empty() { "0" } else { digits };
    if for_parse.is_empty() {
        return Err("invalid integer constant".to_owned());
    }
    let value = i128::from_str_radix(for_parse, radix)
        .map_err(|_| "integer constant out of range".to_owned())?;
    Ok((value, lex::integer_literal_type(value, radix != 10, unsigned, long)))
}

/// Match a punctuator at the start of `s`, returning its kind and byte length.
fn match_punct(s: &[u8]) -> Option<(PpKind, usize)> {
    let c = *s.first()?;
    let c1 = s.get(1).copied();
    let c2 = s.get(2).copied();

    // Three-character punctuators.
    if c == b'.' && c1 == Some(b'.') && c2 == Some(b'.') {
        return Some((PpKind::Punct(Punct::Ellipsis), 3));
    }
    if c == b'<' && c1 == Some(b'<') && c2 == Some(b'=') {
        return Some((PpKind::Punct(Punct::ShlEq), 3));
    }
    if c == b'>' && c1 == Some(b'>') && c2 == Some(b'=') {
        return Some((PpKind::Punct(Punct::ShrEq), 3));
    }

    // Two-character punctuators.
    if c == b'#' && c1 == Some(b'#') {
        return Some((PpKind::HashHash, 2));
    }
    if let Some(t) = c1 {
        let two = match (c, t) {
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
        if let Some(p) = two {
            return Some((PpKind::Punct(p), 2));
        }
    }

    // One-character punctuators.
    if c == b'#' {
        return Some((PpKind::Hash, 1));
    }
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
        _ => return None,
    };
    Some((PpKind::Punct(one), 1))
}

/// The textual spelling of a punctuator (for stringize/paste).
fn punct_spelling(p: Punct) -> &'static str {
    match p {
        Punct::LParen => "(",
        Punct::RParen => ")",
        Punct::LBrace => "{",
        Punct::RBrace => "}",
        Punct::LBracket => "[",
        Punct::RBracket => "]",
        Punct::Semi => ";",
        Punct::Comma => ",",
        Punct::Ellipsis => "...",
        Punct::Plus => "+",
        Punct::Minus => "-",
        Punct::Star => "*",
        Punct::Slash => "/",
        Punct::Percent => "%",
        Punct::Amp => "&",
        Punct::Pipe => "|",
        Punct::Caret => "^",
        Punct::Tilde => "~",
        Punct::Bang => "!",
        Punct::Shl => "<<",
        Punct::Shr => ">>",
        Punct::Lt => "<",
        Punct::Le => "<=",
        Punct::Gt => ">",
        Punct::Ge => ">=",
        Punct::EqEq => "==",
        Punct::Ne => "!=",
        Punct::AmpAmp => "&&",
        Punct::PipePipe => "||",
        Punct::Assign => "=",
        Punct::PlusEq => "+=",
        Punct::MinusEq => "-=",
        Punct::StarEq => "*=",
        Punct::SlashEq => "/=",
        Punct::PercentEq => "%=",
        Punct::AmpEq => "&=",
        Punct::PipeEq => "|=",
        Punct::CaretEq => "^=",
        Punct::ShlEq => "<<=",
        Punct::ShrEq => ">>=",
        Punct::PlusPlus => "++",
        Punct::MinusMinus => "--",
        Punct::Question => "?",
        Punct::Colon => ":",
        Punct::Arrow => "->",
        Punct::Dot => ".",
    }
}
