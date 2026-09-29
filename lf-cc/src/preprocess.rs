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
//! Provenance: every file the translation unit reads (the main source, each
//! `#include`d header — once per inclusion — and each builtin header) is given
//! its own disjoint range of one *virtual offset space*, recorded in a
//! [`SourceMap`]. Every emitted token carries a [`Span`] (always `FileId(0)`)
//! whose offsets lie in the range of the file the token was spelled in, so a
//! diagnostic anywhere downstream resolves to the right file, line and column
//! (plus its include chain) through the map. The main source occupies
//! `[0, len]`, so offsets into it are plain byte offsets exactly as before, and
//! span arithmetic (`Span::merge`) never meets two different file ids. Tokens
//! produced by a macro expansion are attributed to the invocation site;
//! `__LINE__`/`__FILE__` report the true presumed location tracked during
//! expansion.
//!
//! Header search follows the conventional layered model: for `"…"` the
//! including file's directory, then the `-iquote` directories; then (for both
//! forms) `-I`, `-isystem`, the builtin compiler headers
//! ([`crate::headers`]), the host's standard system directories, and finally
//! `-idirafter`. `#include_next` resumes the search after the directory the
//! current file was found in, which is how the builtin headers and a C
//! library's headers layer on top of each other.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use latticefoundry::support::diagnostics::{Diagnostic, FileId, Severity, Span};

use crate::ast::{CType, StrKind};
use crate::cstd::CStd;
use crate::headers::builtin_header;
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
    /// `-iquote` directories: searched for `"…"` includes only, after the
    /// including file's own directory and before [`include_dirs`](Self::include_dirs).
    pub quote_dirs: Vec<PathBuf>,
    /// `-isystem` directories: searched after `-I` and before the builtin
    /// headers.
    pub system_dirs: Vec<PathBuf>,
    /// The standard system include directories (searched after the builtin
    /// headers). Empty by default; the driver fills it with
    /// [`default_system_include_dirs`] unless `-nostdinc` is given.
    pub stdinc_dirs: Vec<PathBuf>,
    /// `-idirafter` directories: searched last of all.
    pub after_dirs: Vec<PathBuf>,
    /// `-D` / `-U` command-line macros, in order.
    pub cmdline: Vec<MacroOp>,
    /// The main source file's name (used for `__FILE__`, diagnostics, and to
    /// resolve `"…"` includes relative to its directory).
    pub main_file_name: String,
    /// Consult the builtin compiler headers (`<stddef.h>`, `<stdint.h>`, …),
    /// which sit on the search chain between `-isystem` and the standard system
    /// directories. Enabled by default; the driver's `-nostdinc` clears it.
    pub builtin_headers: bool,
    /// A hosted implementation (`__STDC_HOSTED__ == 1`). Off by default for the
    /// library (a freestanding translation unit); the driver turns it on unless
    /// `-ffreestanding` is given. When hosted, the builtin `<limits.h>` and
    /// `<stdint.h>` layer over the C library's own (via `#include_next`), and a
    /// `stdc-predef.h` found on the system search chain is pre-included.
    pub hosted: bool,
    /// Optimization is enabled (`-O1` and up): predefines `__OPTIMIZE__` once the
    /// parser accepts the statement expressions C library headers then use.
    pub optimize: bool,
}

impl Default for PpOptions {
    fn default() -> Self {
        PpOptions {
            std: CStd::default(),
            include_dirs: Vec::new(),
            quote_dirs: Vec::new(),
            system_dirs: Vec::new(),
            stdinc_dirs: Vec::new(),
            after_dirs: Vec::new(),
            cmdline: Vec::new(),
            main_file_name: "input.c".to_owned(),
            builtin_headers: true,
            hosted: false,
            optimize: false,
        }
    }
}

/// The host's standard system include directories, in search order: the
/// site-local `/usr/local/include`, the Debian-style multiarch directory for
/// this target (`/usr/include/x86_64-linux-gnu`), then `/usr/include`. Only the
/// directories that exist are returned.
pub fn default_system_include_dirs() -> Vec<PathBuf> {
    ["/usr/local/include", "/usr/include/x86_64-linux-gnu", "/usr/include"]
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .collect()
}

/// Preprocess `main_source` into the final token stream for the parser.
pub fn preprocess(main_source: &str, opts: &PpOptions) -> Result<Vec<Token>, Vec<Diagnostic>> {
    preprocess_mapped(main_source, opts).0
}

/// Like [`preprocess`], but also return the [`SourceMap`] that resolves every
/// token and diagnostic span to its file, line and column (it is returned
/// whether or not preprocessing succeeded, so errors can be rendered).
pub fn preprocess_mapped(
    main_source: &str,
    opts: &PpOptions,
) -> (Result<Vec<Token>, Vec<Diagnostic>>, SourceMap) {
    let mut pp = Pp::new(opts, main_source);
    pp.define_predefined(opts);
    pp.apply_cmdline(&opts.cmdline);

    if opts.hosted {
        pp.preinclude("stdc-predef.h");
    }

    let main_idx = 0u32;
    let toks = pp.lex_file(main_source, main_idx);
    pp.process_file(main_idx, toks);

    let result = if pp.diags.iter().any(Diagnostic::is_error) {
        Err(std::mem::take(&mut pp.diags))
    } else {
        let out = std::mem::take(&mut pp.out);
        let tokens = pp.finalize(out);
        if pp.diags.iter().any(Diagnostic::is_error) {
            Err(std::mem::take(&mut pp.diags))
        } else {
            Ok(tokens)
        }
    };
    (result, pp.map)
}

/// The maximum `#include` nesting depth (cycle guard).
const INCLUDE_DEPTH_LIMIT: usize = 200;

/// The source files of one translation unit laid out in a single virtual offset
/// space (see the [module docs](self)): resolves a token/diagnostic [`Span`]
/// offset back to its file, line, column, and include chain.
#[derive(Clone, Default)]
pub struct SourceMap {
    /// The files in allocation order (ascending, disjoint `base` ranges).
    files: Vec<SourceFile>,
}

/// One file (one inclusion of it) in a [`SourceMap`].
#[derive(Clone)]
struct SourceFile {
    /// The display name (the path as found, or `<name>` for a builtin header).
    name: String,
    /// The file's text (shared between repeated inclusions of one file).
    text: Arc<str>,
    /// The virtual offset of the file's first byte; it spans `[base, base+len]`.
    base: u32,
    /// The virtual offset of the `#include` directive that brought this file
    /// in (`None` for the main source and pre-included files).
    included_at: Option<u32>,
}

impl std::fmt::Debug for SourceMap {
    // The file texts (whole system headers) are deliberately left out.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.files.iter().map(|s| (&s.name, s.base))).finish()
    }
}

/// A resolved source position (1-based line and column).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceLocation {
    /// The file's display name.
    pub file: String,
    /// The 1-based line number.
    pub line: u32,
    /// The 1-based column (in bytes).
    pub column: u32,
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

impl SourceMap {
    /// A map holding only a main source named `name` (what the lexer-level
    /// entry points and callers without a preprocessor run can use).
    pub fn single(name: &str, text: &str) -> SourceMap {
        let mut map = SourceMap::default();
        map.add(name.to_owned(), Arc::from(text), None);
        map
    }

    /// Register a file; returns its base offset, or `None` if the virtual
    /// offset space (4 GiB) is exhausted.
    fn add(&mut self, name: String, text: Arc<str>, included_at: Option<u32>) -> Option<u32> {
        let base = match self.files.last() {
            Some(last) => last.base.checked_add(u32::try_from(last.text.len()).ok()?)?.checked_add(1)?,
            None => 0,
        };
        base.checked_add(u32::try_from(text.len()).ok()?)?;
        self.files.push(SourceFile { name, text, base, included_at });
        Some(base)
    }

    /// The number of files (inclusions) recorded.
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether no file is recorded.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    fn file_at(&self, offset: u32) -> Option<&SourceFile> {
        let idx = self.files.partition_point(|f| f.base <= offset).checked_sub(1)?;
        self.files.get(idx)
    }

    /// Resolve a virtual offset to its file, line and column.
    pub fn locate(&self, offset: u32) -> Option<SourceLocation> {
        let f = self.file_at(offset)?;
        let local = (offset - f.base) as usize;
        let before = f.text.as_bytes().get(..local.min(f.text.len()))?;
        let line = 1 + before.iter().filter(|&&b| b == b'\n').count() as u32;
        let line_start = before.iter().rposition(|&b| b == b'\n').map_or(0, |p| p + 1);
        let column = 1 + (before.len() - line_start) as u32;
        Some(SourceLocation { file: f.name.clone(), line, column })
    }

    /// The include chain of the file containing `offset`, innermost first: the
    /// location of each `#include` directive that led to it.
    pub fn include_chain(&self, offset: u32) -> Vec<SourceLocation> {
        let mut chain = Vec::new();
        let mut cur = self.file_at(offset).and_then(|f| f.included_at);
        while let Some(site) = cur {
            if chain.len() > INCLUDE_DEPTH_LIMIT {
                break;
            }
            let Some(loc) = self.locate(site) else { break };
            chain.push(loc);
            cur = self.file_at(site).and_then(|f| f.included_at);
        }
        chain
    }

    /// Render diagnostics as `file:line:col: severity: message` lines (plus
    /// their notes), each preceded by an `In file included from …` chain when
    /// it lies in an included file. Spanless diagnostics are attributed to the
    /// main file.
    pub fn render(&self, diags: &[Diagnostic]) -> String {
        let main = self.files.first().map(|f| f.name.as_str()).unwrap_or("<input>");
        let mut out = String::new();
        for d in diags {
            let sev = severity_name(d.severity);
            match d.span.and_then(|s| self.locate(s.start).map(|l| (s, l))) {
                Some((span, loc)) => {
                    for (i, site) in self.include_chain(span.start).iter().enumerate() {
                        let lead = if i == 0 { "In file included from" } else { "                 from" };
                        out.push_str(&format!("{lead} {}:{}:\n", site.file, site.line));
                    }
                    out.push_str(&format!("{loc}: {sev}: {}\n", d.message));
                }
                None => out.push_str(&format!("{main}: {sev}: {}\n", d.message)),
            }
            for n in &d.notes {
                match n.span.and_then(|s| self.locate(s.start)) {
                    Some(loc) => out.push_str(&format!("{loc}: note: {}\n", n.message)),
                    None => out.push_str(&format!("{main}: note: {}\n", n.message)),
                }
            }
        }
        out
    }
}

fn severity_name(sev: Severity) -> &'static str {
    match sev {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
    }
}


/// A preprocessing token.
#[derive(Clone, Debug)]
struct PpTok {
    kind: PpKind,
    /// True presumed line for `__LINE__` (physical line; `#line`/`__LINE__`
    /// apply the active delta at use).
    line: u32,
    /// Owning pp-file index (0 = main); used for `__FILE__`.
    file: u32,
    /// First token of a logical line (directive detection).
    bol: bool,
    /// Whitespace/comment preceded this token (stringize spacing).
    space_before: bool,
    /// Span in the virtual offset space (see [`SourceMap`]).
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

/// The recognized parameters of a C23 `#embed` directive.
#[derive(Default, Debug)]
struct EmbedParams {
    /// `limit(N)`: embed at most `N` bytes (`limit(0)` yields an empty resource).
    limit: Option<i128>,
    /// `prefix(tokens…)`: emitted before the bytes of a non-empty embed.
    prefix: Option<Vec<PpTok>>,
    /// `suffix(tokens…)`: emitted after the bytes of a non-empty embed.
    suffix: Option<Vec<PpTok>>,
    /// `if_empty(tokens…)`: emitted instead when the resource is empty (or
    /// `limit(0)`), in place of the bytes and any prefix/suffix.
    if_empty: Option<Vec<PpTok>>,
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

/// One entry of the header search chain.
#[derive(Clone, Debug, PartialEq)]
enum SearchDir {
    /// A directory on disk.
    Dir(PathBuf),
    /// The builtin compiler headers ([`builtin_header`]).
    Builtin,
}

/// Where an `#include` resolved.
#[derive(Debug)]
enum Found {
    /// A file on disk, and the index of the search-chain entry it was found in
    /// (`None` when found relative to the including file's directory).
    Disk(PathBuf, Option<usize>),
    /// A builtin header (its text), found at search-chain entry `usize`.
    Builtin(&'static str, usize),
}

/// The preprocessor state.
#[derive(Debug)]
struct Pp {
    std: CStd,
    /// The header search chain: the `-iquote` entries first, then the entries
    /// shared by both include forms, starting at `angle_start`.
    search: Vec<SearchDir>,
    /// Index into `search` where the `<…>` search begins.
    angle_start: usize,
    macros: HashMap<String, Macro>,
    /// Display names per pp-file (index = pp-file id), used for `__FILE__`.
    filenames: Vec<String>,
    /// The directory `"…"` includes of each pp-file are resolved against
    /// (`None` for a builtin header).
    file_dirs: Vec<Option<PathBuf>>,
    /// The canonical path of each pp-file on disk (`#pragma once`, guards).
    file_canon: Vec<Option<PathBuf>>,
    /// The search-chain entry each pp-file was found in (`#include_next`).
    found_in: Vec<Option<usize>>,
    /// The virtual base offset of each pp-file (see [`SourceMap`]).
    file_base: Vec<u32>,
    /// Every file read, laid out in the virtual offset space.
    map: SourceMap,
    /// Canonical paths guarded by `#pragma once`.
    pragma_once: HashSet<PathBuf>,
    /// Files whose whole content is wrapped in `#ifndef GUARD … #endif`: once
    /// `GUARD` is defined, a re-inclusion is a no-op and is skipped unread.
    guards: HashMap<PathBuf, String>,
    /// File texts already read (shared by repeated inclusions).
    texts: HashMap<PathBuf, Arc<str>>,
    out: Vec<PpTok>,
    diags: Vec<Diagnostic>,
    depth: usize,
    /// `#line` adjustment for the current file (presumed = physical + delta).
    line_delta: i64,
    /// `#line` filename override for the current file.
    file_override: Option<String>,
    main_len: u32,
    /// The next `__COUNTER__` value.
    counter: u64,
    /// The pp-file whose lines are being processed (for `__has_include`).
    cur_file: u32,
}

/// A file about to be entered (see `Pp::enter_file`).
#[derive(Debug)]
struct NewFile {
    /// Display name.
    name: String,
    text: Arc<str>,
    /// Directory for its `"…"` includes (`None` for a builtin header).
    dir: Option<PathBuf>,
    /// Canonical path on disk.
    canon: Option<PathBuf>,
    /// The search-chain entry it was found in.
    found: Option<usize>,
}

impl Pp {
    fn new(opts: &PpOptions, main_source: &str) -> Pp {
        // Assemble the search chain, dropping later duplicates of a directory
        // (as a `-I` naming a system directory would otherwise shadow the
        // system directory's position in the chain, a `-I` duplicate of an
        // `-isystem` or standard directory is dropped in favour of the latter).
        let system: Vec<&PathBuf> =
            opts.system_dirs.iter().chain(&opts.stdinc_dirs).chain(&opts.after_dirs).collect();
        let is_system = |d: &PathBuf| system.iter().any(|s| same_dir(s, d));
        let mut search: Vec<SearchDir> =
            opts.quote_dirs.iter().cloned().map(SearchDir::Dir).collect();
        let angle_start = search.len();
        let push = |search: &mut Vec<SearchDir>, d: SearchDir| {
            let dup = search[angle_start..].iter().any(|s| match (s, &d) {
                (SearchDir::Dir(a), SearchDir::Dir(b)) => same_dir(a, b),
                (a, b) => a == b,
            });
            if !dup {
                search.push(d);
            }
        };
        for d in opts.include_dirs.iter().filter(|d| !is_system(d)) {
            push(&mut search, SearchDir::Dir(d.clone()));
        }
        for d in &opts.system_dirs {
            push(&mut search, SearchDir::Dir(d.clone()));
        }
        if opts.builtin_headers {
            push(&mut search, SearchDir::Builtin);
        }
        for d in opts.stdinc_dirs.iter().chain(&opts.after_dirs) {
            push(&mut search, SearchDir::Dir(d.clone()));
        }

        let main_name = opts.main_file_name.as_str();
        let map = SourceMap::single(main_name, main_source);
        let main_dir = Path::new(main_name).parent().map(Path::to_path_buf);
        Pp {
            std: opts.std,
            search,
            angle_start,
            macros: HashMap::new(),
            filenames: vec![main_name.to_owned()],
            file_dirs: vec![main_dir],
            file_canon: vec![std::fs::canonicalize(main_name).ok()],
            found_in: vec![None],
            file_base: vec![0],
            map,
            pragma_once: HashSet::new(),
            guards: HashMap::new(),
            texts: HashMap::new(),
            out: Vec::new(),
            diags: Vec::new(),
            depth: 0,
            line_delta: 0,
            file_override: None,
            main_len: main_source.len() as u32,
            counter: 0,
            cur_file: 0,
        }
    }

    fn error(&mut self, msg: impl Into<String>, span: Span) {
        self.diags.push(Diagnostic::error(msg).with_span(span));
    }

    /// A span over `[start, end)` of pp-file `file_idx`, in the virtual offset
    /// space.
    fn fspan(&self, file_idx: u32, start: usize, end: usize) -> Span {
        let base = self.file_base.get(file_idx as usize).copied().unwrap_or(0);
        Span::new(FileId::new(0), base + start as u32, base + end as u32)
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

    /// Define a function-like macro from its `name(params) body` spelling.
    fn define_function(&mut self, spec: &str) {
        let toks: Vec<PpTok> = self
            .lex_file(spec, 0)
            .into_iter()
            .map(|mut t| {
                t.span = Span::point(FileId::new(0), 0);
                t
            })
            .collect();
        self.add_define(&toks);
    }

    /// The predefined macros: the standard's (`__STDC__`, `__STDC_HOSTED__`,
    /// `__STDC_VERSION__`), and the target/ABI description a C library's
    /// headers are written against — the x86-64 SysV LP64 data model on
    /// GNU/Linux ELF.
    ///
    /// GNU compatibility level: the GNU dialects predefine `__GNUC__` 4,
    /// `__GNUC_MINOR__` 2, `__GNUC_PATCHLEVEL__` 1 — the conservative baseline
    /// of GNU C (statement expressions, `__typeof__`, `__attribute__`, asm
    /// labels, `__builtin_expect`/`__builtin_constant_p`, `__extension__`, …)
    /// that other GNU-compatible compilers also claim. Headers gate newer
    /// compiler capabilities on the version number: glibc, for instance,
    /// switches to `__builtin_bswap*` at GCC 4.3/4.8, declares the
    /// `_Float128`/`__float128` interfaces from GCC 4.3/4.4, and treats
    /// `_Float32`/`_Float64`/`_Float32x`/`_Float64x` as built-in keywords (not
    /// its own typedefs) from GCC 7. lf-cc implements none of those, so
    /// claiming a newer version would steer the headers into constructs it
    /// cannot compile. Individual newer capabilities are advertised instead
    /// through `__has_builtin`/`__has_attribute`/`__has_feature`.
    ///
    /// Deliberately *not* predefined: `__SIZEOF_INT128__` (no `__int128`),
    /// `__SSE__`/`__SSE2__`/`__MMX__` (no vector types or intrinsics; the
    /// headers keyed on them include `<*intrin.h>`), `__GCC_ATOMIC_*` and
    /// `__GCC_HAVE_SYNC_COMPARE_AND_SWAP_*` (no `__atomic_*`/`__sync_*`
    /// builtins), `__STDC_UTF_16__` (`u""` literals are not UTF-16-encoded
    /// beyond the BMP), `__STDC_EMBED_*__` (no `__has_embed`), and
    /// `__PIC__`/`__PIE__`.
    fn define_predefined(&mut self, opts: &PpOptions) {
        let gnu = self.std.is_gnu();
        self.define_object("__STDC__", "1");
        self.define_object("__STDC_HOSTED__", if opts.hosted { "1" } else { "0" });
        if let Some(v) = self.std.stdc_version() {
            self.define_object("__STDC_VERSION__", &format!("{v}L"));
        }
        if !gnu {
            self.define_object("__STRICT_ANSI__", "1");
        }
        self.define_object("__STDC_UTF_32__", "1");

        // Target: x86-64, LP64, GNU/Linux, ELF. The non-reserved spellings
        // (`linux`, `unix`) only in the GNU dialects, where they are permitted.
        for name in [
            "__x86_64__", "__x86_64", "__amd64__", "__amd64", "__LP64__", "_LP64", "__linux__",
            "__linux", "__gnu_linux__", "__unix__", "__unix", "__ELF__",
        ] {
            self.define_object(name, "1");
        }
        if gnu {
            self.define_object("linux", "1");
            self.define_object("unix", "1");
        }

        // The data model: sizes, widths, byte order.
        let long_double = &LDBL_FORMAT;
        for (name, val) in [
            ("__CHAR_BIT__", "8"),
            ("__SIZEOF_SHORT__", "2"),
            ("__SIZEOF_INT__", "4"),
            ("__SIZEOF_LONG__", "8"),
            ("__SIZEOF_LONG_LONG__", "8"),
            ("__SIZEOF_POINTER__", "8"),
            ("__SIZEOF_SIZE_T__", "8"),
            ("__SIZEOF_PTRDIFF_T__", "8"),
            ("__SIZEOF_WCHAR_T__", "4"),
            ("__SIZEOF_WINT_T__", "4"),
            ("__SIZEOF_FLOAT__", "4"),
            ("__SIZEOF_DOUBLE__", "8"),
            ("__SIZEOF_LONG_DOUBLE__", long_double.size),
            ("__ORDER_LITTLE_ENDIAN__", "1234"),
            ("__ORDER_BIG_ENDIAN__", "4321"),
            ("__ORDER_PDP_ENDIAN__", "3412"),
            ("__BYTE_ORDER__", "__ORDER_LITTLE_ENDIAN__"),
            ("__FLOAT_WORD_ORDER__", "__ORDER_LITTLE_ENDIAN__"),
            // Plain `char` is signed: `__CHAR_UNSIGNED__` is not defined.
            ("__SCHAR_MAX__", "0x7f"),
            ("__SHRT_MAX__", "0x7fff"),
            ("__INT_MAX__", "0x7fffffff"),
            ("__LONG_MAX__", "0x7fffffffffffffffL"),
            ("__LONG_LONG_MAX__", "0x7fffffffffffffffLL"),
            ("__WCHAR_MAX__", "0x7fffffff"),
            ("__WCHAR_MIN__", "(-__WCHAR_MAX__ - 1)"),
            ("__WINT_MAX__", "0xffffffffU"),
            ("__WINT_MIN__", "0U"),
            ("__PTRDIFF_MAX__", "0x7fffffffffffffffL"),
            ("__SIZE_MAX__", "0xffffffffffffffffUL"),
            ("__INTMAX_MAX__", "0x7fffffffffffffffL"),
            ("__UINTMAX_MAX__", "0xffffffffffffffffUL"),
            ("__INTPTR_MAX__", "0x7fffffffffffffffL"),
            ("__UINTPTR_MAX__", "0xffffffffffffffffUL"),
            ("__SIG_ATOMIC_MAX__", "0x7fffffff"),
            ("__SIG_ATOMIC_MIN__", "(-__SIG_ATOMIC_MAX__ - 1)"),
            ("__SCHAR_WIDTH__", "8"),
            ("__SHRT_WIDTH__", "16"),
            ("__INT_WIDTH__", "32"),
            ("__LONG_WIDTH__", "64"),
            ("__LONG_LONG_WIDTH__", "64"),
            ("__PTRDIFF_WIDTH__", "64"),
            ("__SIG_ATOMIC_WIDTH__", "32"),
            ("__SIZE_WIDTH__", "64"),
            ("__WCHAR_WIDTH__", "32"),
            ("__WINT_WIDTH__", "32"),
            ("__INTMAX_WIDTH__", "64"),
            ("__INTPTR_WIDTH__", "64"),
            // The types the C library builds its own typedefs on
            // (`typedef __SIZE_TYPE__ size_t;` and friends).
            ("__SIZE_TYPE__", "long unsigned int"),
            ("__PTRDIFF_TYPE__", "long int"),
            ("__WCHAR_TYPE__", "int"),
            ("__WINT_TYPE__", "unsigned int"),
            ("__INTMAX_TYPE__", "long int"),
            ("__UINTMAX_TYPE__", "long unsigned int"),
            ("__CHAR16_TYPE__", "short unsigned int"),
            ("__CHAR32_TYPE__", "unsigned int"),
            ("__SIG_ATOMIC_TYPE__", "int"),
            ("__INTPTR_TYPE__", "long int"),
            ("__UINTPTR_TYPE__", "long unsigned int"),
        ] {
            self.define_object(name, val);
        }
        // Exact-, least- and fast-width integer types, their limits and widths,
        // and the constant-suffix macros.
        for (bits, sty, uty, smax, umax, sfx, usfx) in [
            ("8", "signed char", "unsigned char", "0x7f", "0xff", "", ""),
            ("16", "short int", "short unsigned int", "0x7fff", "0xffff", "", ""),
            ("32", "int", "unsigned int", "0x7fffffff", "0xffffffffU", "", "U"),
            ("64", "long int", "long unsigned int", "0x7fffffffffffffffL", "0xffffffffffffffffUL", "L", "UL"),
        ] {
            self.define_object(&format!("__INT{bits}_TYPE__"), sty);
            self.define_object(&format!("__UINT{bits}_TYPE__"), uty);
            self.define_object(&format!("__INT{bits}_MAX__"), smax);
            self.define_object(&format!("__UINT{bits}_MAX__"), umax);
            self.define_object(&format!("__INT_LEAST{bits}_TYPE__"), sty);
            self.define_object(&format!("__UINT_LEAST{bits}_TYPE__"), uty);
            self.define_object(&format!("__INT_LEAST{bits}_MAX__"), smax);
            self.define_object(&format!("__UINT_LEAST{bits}_MAX__"), umax);
            self.define_object(&format!("__INT_LEAST{bits}_WIDTH__"), bits);
            // LP64: the fast types of 16 bits and more are `long`.
            let (fs, fu, fsmax, fumax, fw) = if bits == "8" {
                (sty, uty, smax, umax, bits)
            } else {
                ("long int", "long unsigned int", "0x7fffffffffffffffL", "0xffffffffffffffffUL", "64")
            };
            self.define_object(&format!("__INT_FAST{bits}_TYPE__"), fs);
            self.define_object(&format!("__UINT_FAST{bits}_TYPE__"), fu);
            self.define_object(&format!("__INT_FAST{bits}_MAX__"), fsmax);
            self.define_object(&format!("__UINT_FAST{bits}_MAX__"), fumax);
            self.define_object(&format!("__INT_FAST{bits}_WIDTH__"), fw);
            let paste = |s: &str| if s.is_empty() { "c".to_owned() } else { format!("c ## {s}") };
            self.define_function(&format!("__INT{bits}_C(c) {}", paste(sfx)));
            self.define_function(&format!("__UINT{bits}_C(c) {}", paste(usfx)));
        }
        self.define_function("__INTMAX_C(c) c ## L");
        self.define_function("__UINTMAX_C(c) c ## UL");

        // Floating types: IEEE-754 binary32 and binary64, and `long double` as
        // lf-cc implements it (see [`LDBL_FORMAT`]).
        self.define_object("__FLT_RADIX__", "2");
        self.define_object("__FLT_EVAL_METHOD__", "0");
        self.define_object("__FLT_EVAL_METHOD_TS_18661_3__", "0");
        self.define_object("__FINITE_MATH_ONLY__", "0");
        self.define_object("__DECIMAL_DIG__", long_double.decimal_dig);
        for (p, f) in [("FLT", &FLT_FORMAT), ("DBL", &DBL_FORMAT), ("LDBL", long_double)] {
            for (field, val) in [
                ("MANT_DIG", f.mant_dig),
                ("DIG", f.dig),
                ("MIN_EXP", f.min_exp),
                ("MIN_10_EXP", f.min_10_exp),
                ("MAX_EXP", f.max_exp),
                ("MAX_10_EXP", f.max_10_exp),
                ("DECIMAL_DIG", f.decimal_dig),
                ("MAX", f.max),
                ("NORM_MAX", f.max),
                ("MIN", f.min),
                ("EPSILON", f.epsilon),
                ("DENORM_MIN", f.denorm_min),
                ("HAS_DENORM", "1"),
                ("HAS_INFINITY", "1"),
                ("HAS_QUIET_NAN", "1"),
                ("IS_IEC_60559", "1"),
            ] {
                let v = match field {
                    "MAX" | "NORM_MAX" | "MIN" | "EPSILON" | "DENORM_MIN" => format!("{val}{}", f.suffix),
                    _ => val.to_owned(),
                };
                self.define_object(&format!("__{p}_{field}__"), &v);
            }
        }

        // The prefix the ABI prepends to C names at the symbol level: none on
        // ELF. glibc builds its asm labels from it (`__ASMNAME`), so an undefined
        // macro would leak its own name into every redirected symbol.
        self.define_object("__USER_LABEL_PREFIX__", "");
        self.define_object("__REGISTER_PREFIX__", "");
        self.define_object("__DATE__", "\"Jan  1 2020\"");
        self.define_object("__TIME__", "\"00:00:00\"");
        let base = escape_string(&opts.main_file_name);
        self.define_object("__BASE_FILE__", &format!("\"{base}\""));

        // `__OPTIMIZE__` makes C library headers swap functions for
        // optimized macro forms — glibc's <ctype.h> `tolower`/`toupper` become
        // GNU statement expressions `({ … })` — so it is only claimed once the
        // parser accepts those (see `STATEMENT_EXPRESSIONS`).
        if opts.optimize && STATEMENT_EXPRESSIONS {
            self.define_object("__OPTIMIZE__", "1");
        }
        // `__NO_INLINE__` is gcc's "no function is inlined" signal. C libraries
        // key their `extern __inline __attribute__((__gnu_inline__))`
        // definitions (glibc: `__USE_EXTERN_INLINES`) on its absence; lf-cc
        // ignores `gnu_inline` and would emit such a body as a strong external
        // definition, so the headers must never offer them.
        self.define_object("__NO_INLINE__", "1");

        if gnu {
            self.define_object("__GNUC__", "4");
            self.define_object("__GNUC_MINOR__", "2");
            self.define_object("__GNUC_PATCHLEVEL__", "1");
            self.define_object("__VERSION__", "\"4.2.1 Compatible lf-cc\"");
            // The `inline` semantics in force: ISO C99 from C99 on, GNU89 before.
            if self.std.is_c99() {
                self.define_object("__GNUC_STDC_INLINE__", "1");
            } else {
                self.define_object("__GNUC_GNU_INLINE__", "1");
            }
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
                    self.define_function(&format!("{name} {body}"));
                }
                MacroOp::Undef(name) => {
                    self.macros.remove(name);
                }
            }
        }
    }

    /// Pre-include `name` (gcc-style `stdc-predef.h`) from the standard system
    /// directories, if present there: it predefines the C library's view of
    /// the implementation (`__STDC_IEC_559__`, `__STDC_ISO_10646__`, …).
    fn preinclude(&mut self, name: &str) {
        let Some(builtin) = self.search.iter().position(|d| *d == SearchDir::Builtin) else {
            return;
        };
        let start = builtin + 1;
        let found = self.search.iter().enumerate().skip(start).find_map(|(i, d)| match d {
            SearchDir::Dir(dir) if dir.join(name).is_file() => Some((dir.join(name), i)),
            _ => None,
        });
        if let Some((path, idx)) = found {
            self.include_disk(path, Some(idx), None, Span::point(FileId::new(0), 0));
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
            } else if c == b'_' || c == b'$' || c.is_ascii_alphabetic() {
                let id = self.lex_ident(bytes, &mut pos);
                // An encoding-prefixed character/string literal: `L'c'` / `L"s"`
                // (wide), and `u'c'`/`U'c'`/`u8"s"` (Unicode). The prefix is an
                // identifier immediately followed by a quote; keep it as part of
                // the literal's spelling so `eval_char`/`decode_string` see it.
                if let PpKind::Ident(p) = &id
                    && matches!(p.as_str(), "L" | "u" | "U" | "u8")
                    && let Some(&q) = bytes.get(pos)
                    && (q == b'\'' || q == b'"')
                {
                    let prefix = p.clone();
                    match self.lex_quoted(bytes, &mut pos, q) {
                        Some(raw) => {
                            let full = format!("{prefix}{raw}");
                            if q == b'\'' { PpKind::Char(full) } else { PpKind::Str(full) }
                        }
                        None => {
                            self.diags.push(
                                Diagnostic::error("unterminated literal")
                                    .with_span(self.fspan(file_idx, start, pos)),
                            );
                            continue;
                        }
                    }
                } else {
                    id
                }
            } else if c == b'"' {
                match self.lex_quoted(bytes, &mut pos, b'"') {
                    Some(raw) => PpKind::Str(raw),
                    None => {
                        self.diags.push(
                            Diagnostic::error("unterminated string literal")
                                .with_span(self.fspan(file_idx, start, pos)),
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
                                .with_span(self.fspan(file_idx, start, pos)),
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
                        .with_span(self.fspan(file_idx, start, start + 1)),
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
                span: self.fspan(file_idx, start, pos),
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
                            .with_span(self.fspan(file_idx, start, start + 2)),
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
                                self.fspan(file_idx, start, *pos),
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
            // GNU C accepts `$` as an identifier character (on by default, like
            // gcc's -fdollars-in-identifiers); real headers use it in names such
            // as VMS `<lib$routines.h>` inside conditional groups.
            if c == b'_' || c == b'$' || c.is_ascii_alphanumeric() {
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
        let saved_file = std::mem::replace(&mut self.cur_file, file_idx);
        let mut cond: Vec<Cond> = Vec::new();
        let mut run: Vec<PpTok> = Vec::new();
        let mut i = 0usize;
        // Multiple-inclusion guard detection: a file whose every token lies
        // inside one `#ifndef G` … `#endif` group (with no `#else`/`#elif` on
        // it) is a no-op once `G` is defined, so a later inclusion can be
        // skipped without even reading it.
        let guard = guard_macro(&toks);
        let mut guard_ok = guard.is_some();
        let mut guard_closed = false;

        while i < toks.len() {
            let start = i;
            let mut j = i + 1;
            while j < toks.len() && !toks[j].bol {
                j += 1;
            }
            let line = &toks[start..j];
            i = j;
            if guard_closed {
                guard_ok = false;
            }

            let active = cond.last().map(|c| c.active).unwrap_or(true);
            if line[0].bol && matches!(line[0].kind, PpKind::Hash) {
                if !run.is_empty() {
                    let r = std::mem::take(&mut run);
                    let e = self.expand(r);
                    self.out.extend(e);
                }
                if cond.len() == 1
                    && matches!(line.get(1).map(|t| &t.kind), Some(PpKind::Ident(n))
                        if matches!(n.as_str(), "else" | "elif" | "elifdef" | "elifndef"))
                {
                    guard_ok = false;
                }
                let depth_before = cond.len();
                self.handle_directive(file_idx, line, &mut cond);
                if depth_before == 1 && cond.is_empty() {
                    guard_closed = true;
                }
            } else if active {
                run.extend(line.iter().cloned());
            }
        }

        if !run.is_empty() {
            let e = self.expand(std::mem::take(&mut run));
            self.out.extend(e);
        }
        if !cond.is_empty() {
            let at = toks.last().map(|t| t.span).unwrap_or_else(|| self.fspan(file_idx, 0, 0));
            self.error("unterminated `#if` (missing `#endif`)", at);
        } else if guard_ok
            && guard_closed
            && let Some(g) = guard
            && let Some(Some(canon)) = self.file_canon.get(file_idx as usize)
        {
            self.guards.insert(canon.clone(), g);
        }
        self.cur_file = saved_file;
    }

    fn handle_directive(&mut self, file_idx: u32, line: &[PpTok], cond: &mut Vec<Cond>) {
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
            "include" => self.do_include(file_idx, &line[2..], dspan, false),
            "include_next" => self.do_include(file_idx, &line[2..], dspan, true),
            "embed" => self.do_embed(file_idx, &line[2..], &line[0]),
            "error" => {
                let msg = spell_line(&line[2..]);
                self.error(format!("#error {msg}"), dspan);
            }
            "warning" => {
                let msg = spell_line(&line[2..]);
                self.diags
                    .push(Diagnostic::warning(format!("#warning {msg}")).with_span(dspan));
            }
            "pragma" => {
                if let Some(PpKind::Ident(n)) = line.get(2).map(|t| &t.kind)
                    && n == "once"
                    && let Some(Some(canon)) = self.file_canon.get(file_idx as usize).cloned()
                {
                    self.pragma_once.insert(canon);
                }
            }
            // `#ident "…"` / `#sccs "…"`: version strings for the object file's
            // comment section; accepted and dropped.
            "ident" | "sccs" => {}
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
        matches!(
            name,
            "__LINE__"
                | "__FILE__"
                | "__COUNTER__"
                | "__INCLUDE_LEVEL__"
                | "__has_include"
                | "__has_include_next"
                | "__has_builtin"
                | "__has_attribute"
                | "__has_c_attribute"
                | "__has_feature"
                | "__has_extension"
        ) || self.macros.contains_key(name)
    }

    /// Execute `#include` (or, with `next`, `#include_next`).
    fn do_include(&mut self, file_idx: u32, args: &[PpTok], dspan: Span, next: bool) {
        let (name, angled) = match self.parse_header_name(args) {
            Some(v) => v,
            None => {
                let d = if next { "#include_next" } else { "#include" };
                self.error(format!("`{d}` expects \"file\" or <file>"), dspan);
                return;
            }
        };
        match self.find_include(&name, angled, file_idx, next) {
            Some(Found::Disk(path, idx)) => self.include_disk(path, idx, Some(dspan.start), dspan),
            Some(Found::Builtin(text, idx)) => self.include_builtin(&name, text, idx, dspan),
            None => self.error(format!("cannot find include file {name:?}"), dspan),
        }
    }

    /// Resolve a header name along the search chain. `"…"` looks in the
    /// including file's directory first, then the whole chain; `<…>` starts at
    /// the angle-bracket part of the chain. With `next` (`#include_next`, and
    /// `__has_include_next`) the search resumes after the chain entry the
    /// current file was found in; a file not found through the chain (the main
    /// source, or a `"…"` include resolved next to its includer) searches as a
    /// plain `#include` would.
    fn find_include(&self, name: &str, angled: bool, cur: u32, next: bool) -> Option<Found> {
        let mut start = if angled { self.angle_start } else { 0 };
        let mut own_dir = !angled;
        if next && let Some(Some(i)) = self.found_in.get(cur as usize) {
            start = start.max(i + 1);
            own_dir = false;
        }
        if own_dir && let Some(Some(dir)) = self.file_dirs.get(cur as usize) {
            let cand = dir.join(name);
            if cand.is_file() {
                return Some(Found::Disk(cand, None));
            }
        }
        for (i, d) in self.search.iter().enumerate().skip(start) {
            match d {
                SearchDir::Dir(dir) => {
                    let cand = dir.join(name);
                    if cand.is_file() {
                        return Some(Found::Disk(cand, Some(i)));
                    }
                }
                SearchDir::Builtin => {
                    if let Some(text) = builtin_header(name) {
                        return Some(Found::Builtin(text, i));
                    }
                }
            }
        }
        None
    }

    /// Include a header file from disk, found at search-chain entry `found`.
    fn include_disk(&mut self, path: PathBuf, found: Option<usize>, included_at: Option<u32>, dspan: Span) {
        let canon = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if self.pragma_once.contains(&canon) {
            return;
        }
        if let Some(g) = self.guards.get(&canon)
            && self.macros.contains_key(g)
        {
            return;
        }
        let text = match self.texts.get(&canon) {
            Some(t) => t.clone(),
            None => match std::fs::read(&path) {
                // Invalid UTF-8 (say, a Latin-1 comment) is replaced, not fatal;
                // spans then index the replaced text the source map stores.
                Ok(bytes) => {
                    let t: Arc<str> = Arc::from(String::from_utf8_lossy(&bytes).as_ref());
                    self.texts.insert(canon.clone(), t.clone());
                    t
                }
                Err(e) => {
                    self.error(format!("cannot read include file {path:?}: {e}"), dspan);
                    return;
                }
            },
        };
        let dir = path.parent().map(Path::to_path_buf);
        let file = NewFile { name: path.display().to_string(), text, dir, canon: Some(canon), found };
        self.enter_file(file, included_at, dspan);
    }

    /// Include builtin header `name` (found at search-chain entry `found`) as a
    /// virtual file named `<name>`; its own include guard makes re-inclusion
    /// idempotent, and `__FILE__`, `#line` and diagnostics behave exactly as for
    /// an on-disk header.
    fn include_builtin(&mut self, name: &str, text: &'static str, found: usize, dspan: Span) {
        let display = format!("<{name}>");
        let key = PathBuf::from(&display);
        let text = self.texts.entry(key).or_insert_with(|| Arc::from(text)).clone();
        let file = NewFile { name: display, text, dir: None, canon: None, found: Some(found) };
        self.enter_file(file, Some(dspan.start), dspan);
    }

    /// Register a new pp-file (one inclusion of `file`, by the directive at
    /// virtual offset `included_at`) and process it.
    fn enter_file(&mut self, file: NewFile, included_at: Option<u32>, dspan: Span) {
        if self.depth >= INCLUDE_DEPTH_LIMIT {
            self.error("`#include` nested too deeply (cyclic include?)", dspan);
            return;
        }
        let Some(base) = self.map.add(file.name.clone(), file.text.clone(), included_at) else {
            self.error("translation unit too large (4 GiB of source)", dspan);
            return;
        };
        let new_idx = self.filenames.len() as u32;
        self.filenames.push(file.name);
        self.file_dirs.push(file.dir);
        self.file_canon.push(file.canon);
        self.found_in.push(file.found);
        self.file_base.push(base);

        let toks = self.lex_file(&file.text, new_idx);
        let saved_delta = self.line_delta;
        let saved_override = self.file_override.take();
        self.line_delta = 0;
        self.depth += 1;
        self.process_file(new_idx, toks);
        self.depth -= 1;
        self.line_delta = saved_delta;
        self.file_override = saved_override;
    }

    /// Execute a C23 `#embed` directive: locate the resource (the same search as
    /// `#include` — the including file's directory then `-I` for `"…"`, `-I` only
    /// for `<…>`), read its raw bytes, and splice a comma-separated list of the
    /// byte values (each `0..=255`) into the token stream. The optional parameters
    /// `limit(N)`, `prefix(…)`, `suffix(…)`, and `if_empty(…)` are honored per the
    /// standard (see [`EmbedParams`]).
    fn do_embed(&mut self, file_idx: u32, args: &[PpTok], site: &PpTok) {
        let dspan = site.span;
        if !self.std.is_c23() {
            self.error("`#embed` is a C23 feature (use -std=c23 or later)", dspan);
            return;
        }
        // A literal header name (`"…"` / `<…>`) is used verbatim, exactly like
        // `#include`; anything else has the whole line macro-expanded first so a
        // macro can produce the resource name or the parameters.
        let work: Vec<PpTok> = match args.first().map(|t| &t.kind) {
            Some(PpKind::Str(_) | PpKind::Punct(Punct::Lt)) => args.to_vec(),
            _ => self.expand(args.to_vec()),
        };
        let Some((name, angled, rest)) = self.parse_embed_header(&work, dspan) else {
            return;
        };
        let Some(params) = self.parse_embed_params(rest, dspan) else {
            return;
        };
        let Some(path) = self.resolve_embed(&name, angled, file_idx) else {
            self.error(format!("cannot find embed resource {name:?}"), dspan);
            return;
        };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                self.error(format!("cannot read embed resource {path:?}: {e}"), dspan);
                return;
            }
        };

        // Apply `limit(N)` (a negative bound clamps to zero, an oversized one to
        // the file length), then decide the empty vs non-empty expansion.
        let limited: &[u8] = match params.limit {
            Some(l) => {
                let n = l.clamp(0, bytes.len() as i128) as usize;
                &bytes[..n]
            }
            None => &bytes,
        };

        let mut result: Vec<PpTok> = Vec::new();
        if limited.is_empty() {
            if let Some(toks) = params.if_empty {
                result.extend(toks);
            }
        } else {
            if let Some(toks) = params.prefix {
                result.extend(toks);
            }
            for (k, &b) in limited.iter().enumerate() {
                if k != 0 {
                    result.push(self.embed_tok(PpKind::Punct(Punct::Comma), site));
                }
                result.push(self.embed_tok(PpKind::Number(b.to_string()), site));
            }
            if let Some(toks) = params.suffix {
                result.extend(toks);
            }
        }
        // Rescan the spliced tokens like any other run so a `prefix`/`suffix`/
        // `if_empty` macro is expanded (the byte numbers pass through untouched).
        let expanded = self.expand(result);
        self.out.extend(expanded);
    }

    /// A synthetic `#embed`-produced token, attributed to the directive site.
    fn embed_tok(&self, kind: PpKind, site: &PpTok) -> PpTok {
        PpTok {
            kind,
            line: site.line,
            file: site.file,
            bol: false,
            space_before: true,
            span: site.span,
            hideset: BTreeSet::new(),
        }
    }

    /// Split an `#embed` argument line into its header name and the remaining
    /// parameter tokens. Returns `(name, angled, rest)`, or `None` after reporting
    /// a diagnostic.
    fn parse_embed_header<'a>(
        &mut self,
        args: &'a [PpTok],
        dspan: Span,
    ) -> Option<(String, bool, &'a [PpTok])> {
        match args.first().map(|t| &t.kind) {
            Some(PpKind::Str(raw)) => Some((decode_string(raw), false, &args[1..])),
            Some(PpKind::Punct(Punct::Lt)) => {
                let mut i = 1usize;
                while i < args.len() && !matches!(args[i].kind, PpKind::Punct(Punct::Gt)) {
                    i += 1;
                }
                if i >= args.len() {
                    self.error("missing '>' in `#embed <...>`", dspan);
                    return None;
                }
                Some((join_angle(&args[1..i]), true, &args[i + 1..]))
            }
            _ => {
                self.error("`#embed` expects \"file\" or <file>", dspan);
                None
            }
        }
    }

    /// Parse the `#embed` parameter sequence (`name` or `name(tokens…)`). Returns
    /// the recognized parameters, or `None` after reporting a diagnostic.
    fn parse_embed_params(&mut self, rest: &[PpTok], dspan: Span) -> Option<EmbedParams> {
        let mut p = EmbedParams::default();
        let mut i = 0usize;
        while i < rest.len() {
            let PpKind::Ident(name) = &rest[i].kind else {
                self.error("expected an `#embed` parameter name", rest[i].span);
                return None;
            };
            let name = name.clone();
            let nspan = rest[i].span;
            i += 1;
            // An optional balanced parenthesized argument list.
            let mut inner: Option<Vec<PpTok>> = None;
            if matches!(rest.get(i).map(|t| &t.kind), Some(PpKind::Punct(Punct::LParen))) {
                i += 1;
                let start = i;
                let mut depth = 1usize;
                while i < rest.len() {
                    match &rest[i].kind {
                        PpKind::Punct(Punct::LParen) => depth += 1,
                        PpKind::Punct(Punct::RParen) => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                if depth != 0 {
                    self.error(format!("unterminated `{name}(...)` `#embed` parameter"), nspan);
                    return None;
                }
                inner = Some(rest[start..i].to_vec());
                i += 1; // consume ')'
            }
            match name.as_str() {
                "limit" => {
                    let toks = inner.unwrap_or_default();
                    if toks.is_empty() {
                        self.error("`#embed` `limit` requires a value", nspan);
                        return None;
                    }
                    let expanded = self.expand(toks);
                    match self.eval_const_expr(&expanded, dspan) {
                        Ok(v) => p.limit = Some(v),
                        Err(d) => {
                            self.diags.push(d);
                            return None;
                        }
                    }
                }
                "prefix" => p.prefix = Some(inner.unwrap_or_default()),
                "suffix" => p.suffix = Some(inner.unwrap_or_default()),
                "if_empty" => p.if_empty = Some(inner.unwrap_or_default()),
                other => {
                    self.error(format!("unsupported `#embed` parameter `{other}`"), nspan);
                    return None;
                }
            }
        }
        Some(p)
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

    /// Resolve an `#embed` resource name: the `#include` search, ignoring the
    /// builtin headers (which are not resources).
    fn resolve_embed(&self, name: &str, angled: bool, cur: u32) -> Option<PathBuf> {
        match self.find_include(name, angled, cur, false) {
            Some(Found::Disk(path, _)) => Some(path),
            _ => None,
        }
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
            if let Some((params, variadic, va_name, body_start)) = self.parse_params(rest) {
                let mut body = clean_body(&rest[body_start..]);
                // GNU named variadic parameter (`args...`): its uses in the body
                // denote the variable arguments, exactly like `__VA_ARGS__`.
                if let Some(va) = va_name {
                    for t in &mut body {
                        if matches!(&t.kind, PpKind::Ident(n) if *n == va) {
                            t.kind = PpKind::Ident("__VA_ARGS__".to_owned());
                        }
                    }
                }
                self.macros.insert(name, Macro { params: Some(params), variadic, body });
            }
            return;
        }
        let body = clean_body(rest);
        self.macros.insert(name, Macro { params: None, variadic: false, body });
    }

    /// Parse a function-like parameter list starting at `rest[0] == '('`.
    /// Returns the parameters, whether variadic, the name of a GNU named
    /// variadic parameter (`args...`), and the body start index.
    #[allow(clippy::type_complexity)]
    fn parse_params(&mut self, rest: &[PpTok]) -> Option<(Vec<String>, bool, Option<String>, usize)> {
        let mut params = Vec::new();
        let mut variadic = false;
        let mut va_name = None;
        let mut i = 1usize; // skip '('
        if matches!(rest.get(i).map(|t| &t.kind), Some(PpKind::Punct(Punct::RParen))) {
            return Some((params, variadic, va_name, i + 1));
        }
        loop {
            match rest.get(i).map(|t| &t.kind) {
                Some(PpKind::Punct(Punct::Ellipsis)) => {
                    variadic = true;
                    i += 1;
                    break;
                }
                Some(PpKind::Ident(n))
                    if matches!(rest.get(i + 1).map(|t| &t.kind), Some(PpKind::Punct(Punct::Ellipsis))) =>
                {
                    variadic = true;
                    va_name = Some(n.clone());
                    i += 2;
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
        Some((params, variadic, va_name, i + 1))
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
            if name == "__COUNTER__" {
                let n = self.counter;
                self.counter += 1;
                out.push(number_tok(&n.to_string(), &t));
                continue;
            }
            if name == "__INCLUDE_LEVEL__" {
                out.push(number_tok(&self.depth.to_string(), &t));
                continue;
            }
            // The C99 `_Pragma ( string-literal )` operator: a pragma produced by
            // macro expansion. No pragma lf-cc honors is meaningful here (they
            // are diagnostics/optimization controls), so the operator is
            // consumed and dropped.
            if name == "_Pragma"
                && matches!(input.front().map(|x| &x.kind), Some(PpKind::Punct(Punct::LParen)))
                && matches!(input.get(1).map(|x| &x.kind), Some(PpKind::Str(_)))
                && matches!(input.get(2).map(|x| &x.kind), Some(PpKind::Punct(Punct::RParen)))
            {
                input.drain(..3);
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
        // The `defined` and `__has_*` operators are evaluated before macro
        // expansion (their operands are names, not expressions), and once more
        // afterwards for the ones a macro expansion produced (glibc's
        // `__glibc_has_attribute (x)` expands to `__has_attribute (x)`).
        let replaced = self.replace_defined(toks, at.span);
        let expanded = self.expand(replaced);
        let expanded = self.replace_defined(&expanded, at.span);
        match self.eval_const_expr(&expanded, at.span) {
            Ok(v) => v != 0,
            Err(d) => {
                self.diags.push(d);
                false
            }
        }
    }

    /// Replace `defined X` / `defined(X)` with `1` or `0`, and each `__has_*`
    /// query (`__has_include`, `__has_include_next`, `__has_builtin`,
    /// `__has_attribute`, `__has_c_attribute`, `__has_feature`,
    /// `__has_extension`) with its value.
    fn replace_defined(&mut self, toks: &[PpTok], at: Span) -> Vec<PpTok> {
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < toks.len() {
            let t = &toks[i];
            let PpKind::Ident(op) = &t.kind else {
                out.push(t.clone());
                i += 1;
                continue;
            };
            if op == "defined" {
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
            let is_query = matches!(
                op.as_str(),
                "__has_include"
                    | "__has_include_next"
                    | "__has_builtin"
                    | "__has_attribute"
                    | "__has_c_attribute"
                    | "__has_feature"
                    | "__has_extension"
            );
            if !is_query {
                out.push(t.clone());
                i += 1;
                continue;
            }
            // The parenthesized operand: everything up to the matching `)`.
            let Some(PpKind::Punct(Punct::LParen)) = toks.get(i + 1).map(|x| &x.kind) else {
                self.error(format!("missing '(' after `{op}`"), t.span);
                i += 1;
                continue;
            };
            let mut j = i + 2;
            let mut depth = 0usize;
            while let Some(x) = toks.get(j) {
                match x.kind {
                    PpKind::Punct(Punct::LParen) => depth += 1,
                    PpKind::Punct(Punct::RParen) if depth == 0 => break,
                    PpKind::Punct(Punct::RParen) => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            if j >= toks.len() {
                self.error(format!("missing ')' after `{op}` operand"), t.span);
                return out;
            }
            let operand = &toks[i + 2..j];
            let value = match op.as_str() {
                "__has_include" | "__has_include_next" => {
                    let next = op == "__has_include_next";
                    match self.parse_header_name(operand) {
                        Some((name, angled)) => {
                            i128::from(self.find_include(&name, angled, self.cur_file, next).is_some())
                        }
                        None => {
                            self.error(format!("`{op}` expects \"file\" or <file>"), t.span);
                            0
                        }
                    }
                }
                _ => {
                    let name = spell_joined(operand);
                    match op.as_str() {
                        "__has_builtin" => i128::from(has_builtin(&name)),
                        "__has_attribute" => i128::from(has_gnu_attribute(&name)),
                        "__has_c_attribute" => has_c_attribute(&name),
                        _ => i128::from(has_feature(&name, self.std)),
                    }
                }
            };
            out.push(number_tok(&value.to_string(), t));
            i = j + 1;
        }
        out
    }

    fn eval_const_expr(&self, toks: &[PpTok], at: Span) -> Result<i128, Diagnostic> {
        let mut items: Vec<EItem> = Vec::new();
        for t in toks {
            let item = match &t.kind {
                PpKind::Number(s) => match eval_number(s, self.std) {
                    // Out-of-range values were rejected by `eval_number`; the
                    // constant's type decides signed vs unsigned arithmetic.
                    Ok(NumVal::Int(v, ty)) => {
                        EItem::Num(PpVal { bits: v as u64, unsigned: ty.is_integer() && !ty.is_signed() })
                    }
                    Ok(NumVal::Float(..)) => {
                        return Err(Diagnostic::error(
                            "floating constant in a preprocessor `#if` expression",
                        )
                        .with_span(t.span));
                    }
                    Err(m) => return Err(Diagnostic::error(m).with_span(t.span)),
                },
                PpKind::Char(raw) => EItem::Num(PpVal::signed(eval_char(raw) as i64)),
                PpKind::Ident(n) => {
                    // Remaining identifiers evaluate to 0 (C23 `true` is 1).
                    EItem::Num(PpVal::truth(self.std.is_c23() && n == "true"))
                }
                PpKind::Punct(p) => EItem::Op(*p),
                PpKind::Str(_) => {
                    return Err(Diagnostic::error("string literal in `#if` expression")
                        .with_span(t.span));
                }
                PpKind::Hash | PpKind::HashHash | PpKind::Placemarker => {
                    return Err(Diagnostic::error("invalid token in `#if` expression")
                        .with_span(t.span));
                }
            };
            items.push(item);
        }
        let mut ev = Ev { items: &items, pos: 0, at, skip: 0 };
        let v = ev.expr()?;
        if ev.pos != ev.items.len() {
            return Err(Diagnostic::error("trailing tokens in `#if` expression").with_span(ev.at));
        }
        Ok(v.value())
    }

    // --- finalize -----------------------------------------------------------

    fn finalize(&mut self, toks: Vec<PpTok>) -> Vec<Token> {
        let mut out = Vec::with_capacity(toks.len() + 1);
        for t in toks {
            let span = t.span;
            let kind = match &t.kind {
                PpKind::Ident(name) => match self.classify_ident(name, span) {
                    Some(k) => k,
                    None => continue,
                },
                PpKind::Number(s) => match eval_number(s, self.std) {
                    Ok(NumVal::Int(v, ty)) => TokenKind::IntLit(v, ty),
                    Ok(NumVal::Float(v, ty)) => TokenKind::FloatLit(v, ty),
                    Err(m) => {
                        self.diags.push(Diagnostic::error(m).with_span(span));
                        continue;
                    }
                },
                PpKind::Char(raw) => TokenKind::IntLit(eval_char(raw), CType::int()),
                PpKind::Str(raw) => {
                    let kind = str_kind_of(raw);
                    TokenKind::Str(decode_string_elements(raw, kind), kind)
                }
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
    /// gating. Returns `None` (dropping the token) when a gating error is
    /// recorded, and for GNU `__extension__`, which only silences pedantic
    /// diagnostics and so has no meaning to this compiler.
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
            // GNU spellings live in the reserved `__` namespace and are available
            // as extensions under every standard (not std-gated), so real C code
            // (e.g. bzip2's `static __inline__ void ...`) parses in any mode.
            "__inline__" | "__inline" => Some(TokenKind::Keyword(Keyword::Inline)),
            "__restrict__" | "__restrict" => Some(TokenKind::Keyword(Keyword::Restrict)),
            "__const__" | "__const" => Some(TokenKind::Keyword(Keyword::Const)),
            "__volatile__" | "__volatile" => Some(TokenKind::Keyword(Keyword::Volatile)),
            "__signed__" | "__signed" => Some(TokenKind::Keyword(Keyword::Signed)),
            // GNU `asm`: the reserved spellings everywhere, the plain keyword only
            // under the GNU dialects (in ISO modes `asm` is an ordinary identifier).
            "__asm__" | "__asm" => Some(TokenKind::Keyword(Keyword::Asm)),
            "__extension__" => None,
            "asm" if self.std.is_gnu() => Some(TokenKind::Keyword(Keyword::Asm)),
            "_Noreturn" => self.gate_reserved(name, self.std.static_assert_generic(), "C11", Keyword::Noreturn, span),
            "_Alignof" => self.gate_reserved(name, self.std.static_assert_generic(), "C11", Keyword::Alignof, span),
            "_Alignas" => self.gate_reserved(name, self.std.static_assert_generic(), "C11", Keyword::Alignas, span),
            "_Static_assert" => self.gate_reserved(name, self.std.static_assert_generic(), "C11", Keyword::StaticAssert, span),
            "_Generic" => self.gate_reserved(name, self.std.static_assert_generic(), "C11", Keyword::Generic, span),
            // C23 keyword spellings (plain identifiers under earlier standards).
            "alignof" if self.std.keyword_alignas() => Some(TokenKind::Keyword(Keyword::Alignof)),
            "alignas" if self.std.keyword_alignas() => Some(TokenKind::Keyword(Keyword::Alignas)),
            "static_assert" if self.std.keyword_alignas() => Some(TokenKind::Keyword(Keyword::StaticAssert)),
            "noreturn" if self.std.keyword_noreturn() => Some(TokenKind::Keyword(Keyword::Noreturn)),
            // `typeof`/`typeof_unqual`: C23 keywords, also a GNU extension.
            "typeof" if self.std.typeof_specifier() => Some(TokenKind::Keyword(Keyword::Typeof)),
            "typeof_unqual" if self.std.typeof_specifier() => {
                Some(TokenKind::Keyword(Keyword::TypeofUnqual))
            }
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

/// A value in a `#if` expression: per C (6.10.1), every integer is evaluated
/// as `intmax_t` or `uintmax_t` — 64 bits on this target — so `bits` holds the
/// two's-complement representation and `unsigned` selects the interpretation.
#[derive(Clone, Copy, Debug)]
struct PpVal {
    bits: u64,
    unsigned: bool,
}

impl PpVal {
    fn signed(v: i64) -> PpVal {
        PpVal { bits: v as u64, unsigned: false }
    }

    fn truth(b: bool) -> PpVal {
        PpVal::signed(i64::from(b))
    }

    fn is_true(self) -> bool {
        self.bits != 0
    }

    /// The mathematical value.
    fn value(self) -> i128 {
        if self.unsigned { i128::from(self.bits) } else { i128::from(self.bits as i64) }
    }
}

/// The evaluator's simplified token.
#[derive(Clone, Copy, Debug)]
enum EItem {
    Num(PpVal),
    Op(Punct),
}

/// A recursive-descent evaluator for `#if` constant integer expressions.
struct Ev<'a> {
    items: &'a [EItem],
    pos: usize,
    at: Span,
    /// Nesting depth of operands that are not evaluated (the untaken arm of
    /// `?:`, the right side of a short-circuited `&&`/`||`): a division by
    /// zero there is not an error.
    skip: u32,
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

    fn expr(&mut self) -> Result<PpVal, Diagnostic> {
        self.ternary()
    }

    /// Parse an operand that is evaluated only when `live`.
    fn operand<T>(&mut self, live: bool, f: impl FnOnce(&mut Self) -> Result<T, Diagnostic>) -> Result<T, Diagnostic> {
        if !live {
            self.skip += 1;
        }
        let r = f(self);
        if !live {
            self.skip -= 1;
        }
        r
    }

    fn ternary(&mut self) -> Result<PpVal, Diagnostic> {
        let c = self.binary(0)?;
        if self.peek() == Some(Punct::Question) {
            self.pos += 1;
            let t = self.operand(c.is_true(), Self::expr)?;
            if self.peek() != Some(Punct::Colon) {
                return Err(self.err("expected ':' in `#if` conditional"));
            }
            self.pos += 1;
            let e = self.operand(!c.is_true(), Self::ternary)?;
            // The result has the common type of the two arms.
            let unsigned = t.unsigned || e.unsigned;
            let v = if c.is_true() { t } else { e };
            Ok(PpVal { bits: v.bits, unsigned })
        } else {
            Ok(c)
        }
    }

    fn binary(&mut self, min_prec: u8) -> Result<PpVal, Diagnostic> {
        let mut lhs = self.unary()?;
        while let Some((prec, op)) = self.peek().and_then(binop_prec) {
            if prec < min_prec {
                break;
            }
            self.pos += 1;
            let live = match op {
                Punct::AmpAmp => lhs.is_true(),
                Punct::PipePipe => !lhs.is_true(),
                _ => true,
            };
            let rhs = self.operand(live, |ev| ev.binary(prec + 1))?;
            lhs = apply_binop(op, lhs, rhs, self)?;
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<PpVal, Diagnostic> {
        match self.peek() {
            Some(Punct::Minus) => {
                self.pos += 1;
                let v = self.unary()?;
                Ok(PpVal { bits: v.bits.wrapping_neg(), ..v })
            }
            Some(Punct::Plus) => {
                self.pos += 1;
                self.unary()
            }
            Some(Punct::Bang) => {
                self.pos += 1;
                Ok(PpVal::truth(!self.unary()?.is_true()))
            }
            Some(Punct::Tilde) => {
                self.pos += 1;
                let v = self.unary()?;
                Ok(PpVal { bits: !v.bits, ..v })
            }
            _ => self.primary(),
        }
    }

    fn primary(&mut self) -> Result<PpVal, Diagnostic> {
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

/// Apply a binary operator with the usual arithmetic conversions of
/// `intmax_t`/`uintmax_t` (either operand unsigned makes both unsigned; a shift
/// has its left operand's type; comparisons and logical operators yield a
/// signed 0/1).
fn apply_binop(op: Punct, a: PpVal, b: PpVal, ev: &Ev<'_>) -> Result<PpVal, Diagnostic> {
    let unsigned = a.unsigned || b.unsigned;
    let (x, y) = (a.bits, b.bits);
    let (sx, sy) = (x as i64, y as i64);
    let arith = |bits: u64| PpVal { bits, unsigned };
    let (lt, eq) = if unsigned { (x < y, x == y) } else { (sx < sy, sx == sy) };
    Ok(match op {
        Punct::PipePipe => PpVal::truth(a.is_true() || b.is_true()),
        Punct::AmpAmp => PpVal::truth(a.is_true() && b.is_true()),
        Punct::Pipe => arith(x | y),
        Punct::Caret => arith(x ^ y),
        Punct::Amp => arith(x & y),
        Punct::EqEq => PpVal::truth(eq),
        Punct::Ne => PpVal::truth(!eq),
        Punct::Lt => PpVal::truth(lt),
        Punct::Le => PpVal::truth(lt || eq),
        Punct::Gt => PpVal::truth(!lt && !eq),
        Punct::Ge => PpVal::truth(!lt),
        Punct::Shl | Punct::Shr => {
            // An out-of-range count is undefined; take the limit of the shift.
            let count = if b.unsigned || sy >= 0 { y.min(64) as u32 } else { 64 };
            let bits = match (op, a.unsigned) {
                (Punct::Shl, _) => x.checked_shl(count).unwrap_or(0),
                (_, true) => x.checked_shr(count).unwrap_or(0),
                _ => sx.checked_shr(count).unwrap_or(if sx < 0 { -1 } else { 0 }) as u64,
            };
            PpVal { bits, unsigned: a.unsigned }
        }
        Punct::Plus => arith(x.wrapping_add(y)),
        Punct::Minus => arith(x.wrapping_sub(y)),
        Punct::Star => arith(x.wrapping_mul(y)),
        Punct::Slash | Punct::Percent => {
            if y == 0 {
                if ev.skip > 0 {
                    return Ok(arith(0));
                }
                return Err(ev.err("division by zero in `#if` expression"));
            }
            let bits = match (op, unsigned) {
                (Punct::Slash, true) => x / y,
                (Punct::Slash, false) => sx.wrapping_div(sy) as u64,
                (_, true) => x % y,
                _ => sx.wrapping_rem(sy) as u64,
            };
            arith(bits)
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
        "float" => Keyword::Float,
        "double" => Keyword::Double,
        "signed" => Keyword::Signed,
        "unsigned" => Keyword::Unsigned,
        "const" => Keyword::Const,
        "volatile" => Keyword::Volatile,
        "extern" => Keyword::Extern,
        "static" => Keyword::Static,
        "register" => Keyword::Register,
        "auto" => Keyword::Auto,
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

/// Decode a string literal spelling (strip quotes, unescape) to text.
/// Strip a leading encoding prefix (`L`, `u`, `U`, or `u8`) from a
/// character/string-literal spelling, leaving the quoted body. A prefix is only
/// recognised when it is immediately followed by a quote.
fn strip_encoding_prefix(raw: &str) -> &str {
    for p in ["u8", "L", "u", "U"] {
        if let Some(rest) = raw.strip_prefix(p)
            && (rest.starts_with('\'') || rest.starts_with('"'))
        {
            return rest;
        }
    }
    raw
}

fn decode_string(raw: &str) -> String {
    let raw = strip_encoding_prefix(raw);
    let inner = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(raw);
    unescape(inner)
}

/// Classify a string-literal spelling by its encoding prefix.
fn str_kind_of(raw: &str) -> StrKind {
    if raw.starts_with("u8") {
        StrKind::Narrow
    } else if raw.starts_with('L') {
        StrKind::Wide
    } else if raw.starts_with('u') {
        StrKind::Char16
    } else if raw.starts_with('U') {
        StrKind::Char32
    } else {
        StrKind::Narrow
    }
}

/// Decode a string-literal spelling to its element bytes, encoded little-endian
/// at the element width for its `kind`. For a narrow string this is exactly the
/// byte sequence [`decode_string_bytes`] produces; for a wide/`u`/`U` string
/// each element occupies `kind.elem_width()` bytes.
fn decode_string_elements(raw: &str, kind: StrKind) -> Vec<u8> {
    let raw = strip_encoding_prefix(raw);
    let inner = raw.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(raw);
    let width = kind.elem_width();
    if width == 1 {
        return unescape_bytes(inner);
    }
    let mut out = Vec::new();
    for elem in unescape_elements(inner, width) {
        out.extend_from_slice(&elem.to_le_bytes()[..width as usize]);
    }
    out
}

/// Decode a wide string-literal body to its sequence of element values (masked
/// to `width` bytes). A source character contributes its Unicode scalar value as
/// one element; escapes are interpreted at the element width.
fn unescape_elements(inner: &str, width: u64) -> Vec<u32> {
    let mask: u64 = if width >= 4 { 0xffff_ffff } else { (1u64 << (width * 8)) - 1 };
    let m = |v: u64| (v & mask) as u32;
    let mut out: Vec<u32> = Vec::new();
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(m(u64::from(c)));
            continue;
        }
        match chars.next() {
            Some('n') => out.push(m(u64::from(b'\n'))),
            Some('t') => out.push(m(u64::from(b'\t'))),
            Some('r') => out.push(m(u64::from(b'\r'))),
            Some('a') => out.push(m(7)),
            Some('b') => out.push(m(8)),
            Some('f') => out.push(m(12)),
            Some('v') => out.push(m(11)),
            Some('\\') => out.push(m(u64::from(b'\\'))),
            Some('\'') => out.push(m(u64::from(b'\''))),
            Some('"') => out.push(m(u64::from(b'"'))),
            Some('?') => out.push(m(u64::from(b'?'))),
            // `\xhh…`: a hexadecimal escape, masked to the element width.
            Some('x') => {
                let mut v: u64 = 0;
                let mut any = false;
                while let Some(h) = chars.peek().and_then(|c| c.to_digit(16)) {
                    v = v.wrapping_mul(16).wrapping_add(u64::from(h));
                    any = true;
                    chars.next();
                }
                out.push(if any { m(v) } else { m(u64::from(b'x')) });
            }
            // `\uNNNN` / `\UNNNNNNNN`: a universal character name.
            Some(u @ ('u' | 'U')) => {
                let ndigits = if u == 'u' { 4 } else { 8 };
                let mut v: u64 = 0;
                let mut n = 0;
                while n < ndigits {
                    match chars.peek().and_then(|c| c.to_digit(16)) {
                        Some(h) => {
                            v = v.wrapping_mul(16).wrapping_add(u64::from(h));
                            chars.next();
                            n += 1;
                        }
                        None => break,
                    }
                }
                out.push(m(v));
            }
            // `\ooo`: an octal escape of one to three octal digits.
            Some(d @ '0'..='7') => {
                let mut v = u64::from(d.to_digit(8).unwrap());
                let mut n = 1;
                while n < 3 {
                    match chars.peek().and_then(|c| c.to_digit(8)) {
                        Some(o) => {
                            v = v * 8 + u64::from(o);
                            chars.next();
                            n += 1;
                        }
                        None => break,
                    }
                }
                out.push(m(v));
            }
            Some(other) => out.push(m(u64::from(other))),
            None => out.push(m(u64::from(b'\\'))),
        }
    }
    out
}

/// Decode a string-literal body to its exact C byte sequence, interpreting the
/// escape sequences. Octal (`\ooo`) and hexadecimal (`\xhh…`) escapes yield the
/// exact byte value; a source character outside ASCII contributes its raw UTF-8
/// bytes (a C string literal is a byte array).
fn unescape_bytes(inner: &str) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut chars = inner.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            // A non-ASCII source character contributes its UTF-8 bytes verbatim.
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        match chars.next() {
            Some('n') => out.push(b'\n'),
            Some('t') => out.push(b'\t'),
            Some('r') => out.push(b'\r'),
            Some('a') => out.push(7),
            Some('b') => out.push(8),
            Some('f') => out.push(12),
            Some('v') => out.push(11),
            Some('\\') => out.push(b'\\'),
            Some('\'') => out.push(b'\''),
            Some('"') => out.push(b'"'),
            Some('?') => out.push(b'?'),
            // `\xhh…`: a hexadecimal escape (one or more hex digits).
            Some('x') => {
                let mut v: u32 = 0;
                let mut any = false;
                while let Some(h) = chars.peek().and_then(|c| c.to_digit(16)) {
                    v = v.wrapping_mul(16).wrapping_add(h);
                    any = true;
                    chars.next();
                }
                if any {
                    out.push((v & 0xff) as u8);
                } else {
                    out.push(b'x');
                }
            }
            // `\ooo`: an octal escape of one to three octal digits.
            Some(d @ '0'..='7') => {
                let mut v = d.to_digit(8).unwrap();
                let mut n = 1;
                while n < 3 {
                    match chars.peek().and_then(|c| c.to_digit(8)) {
                        Some(o) => {
                            v = v * 8 + o;
                            chars.next();
                            n += 1;
                        }
                        None => break,
                    }
                }
                out.push((v & 0xff) as u8);
            }
            Some(other) => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(other.encode_utf8(&mut buf).as_bytes());
            }
            None => out.push(b'\\'),
        }
    }
    out
}

/// Decode a string-literal body to a `String` (lossily for non-UTF-8 bytes).
/// Used where a textual value is needed (an `#include` filename, a `#pragma`
/// argument): those are ASCII in practice.
fn unescape(inner: &str) -> String {
    String::from_utf8_lossy(&unescape_bytes(inner)).into_owned()
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
    let raw = strip_encoding_prefix(raw);
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
            Some(b'\\') => 92,
            Some(b'\'') => 39,
            Some(b'"') => 34,
            Some(b'a') => 7,
            Some(b'b') => 8,
            Some(b'f') => 12,
            Some(b'v') => 11,
            Some(b'?') => 63,
            // `\xhh…`: a hexadecimal escape (one or more hex digits), truncated
            // to a single byte.
            Some(b'x') => {
                let hex: String = inner[2..].chars().take_while(|c| c.is_ascii_hexdigit()).collect();
                i128::from_str_radix(&hex, 16).unwrap_or(0) & 0xff
            }
            // `\ooo`: an octal escape of one to three octal digits (`\0` is the
            // common case), truncated to a single byte.
            Some(d @ b'0'..=b'7') => {
                let mut val = i128::from(d - b'0');
                let mut n = 2;
                while n < 4 {
                    match bytes.get(n).copied() {
                        Some(o @ b'0'..=b'7') => {
                            val = val * 8 + i128::from(o - b'0');
                            n += 1;
                        }
                        _ => break,
                    }
                }
                val & 0xff
            }
            Some(other) => i128::from(other),
            None => 0,
        };
    }
    i128::from(bytes[0])
}

/// A parsed preprocessing number: an integer or a floating constant, each with
/// its C type.
enum NumVal {
    /// An integer constant value and its type.
    Int(i128, CType),
    /// A floating constant value (exact, already rounded to precision) and type.
    Float(f64, CType),
}

/// Parse a preprocessing number into a value and its C type, honoring C23 binary
/// literals and digit separators, and C99 (hex) floating constants (all gated by
/// `std`).
fn eval_number(text: &str, std: CStd) -> Result<NumVal, String> {
    if text.contains('\'') && !std.digit_separators() {
        return Err("digit separators are a C23 feature (use -std=c23)".to_owned());
    }
    let s: String =
        if std.digit_separators() { text.chars().filter(|&c| c != '\'').collect() } else { text.to_owned() };
    // A `.`, an exponent, or a float suffix makes this a floating constant.
    if lex::is_float_ppnumber(&s) {
        let (v, ty) = lex::parse_float_literal(&s, std.is_c99())?;
        return Ok(NumVal::Float(v, ty));
    }
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
    Ok(NumVal::Int(value, lex::integer_literal_type(value, radix != 10, unsigned, long)))
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

// --- search chain, guards and feature queries --------------------------------

/// Whether two search directories name the same directory (canonically, when
/// both exist).
fn same_dir(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// The include-guard macro of a file whose first line is `#ifndef G`,
/// `#if !defined G` or `#if !defined(G)` (whether the guard group really
/// encloses the whole file is checked while the file is processed).
fn guard_macro(toks: &[PpTok]) -> Option<String> {
    let first = toks.first()?;
    if !first.bol || !matches!(first.kind, PpKind::Hash) {
        return None;
    }
    let line_end = toks.iter().skip(1).position(|t| t.bol).map_or(toks.len(), |p| p + 1);
    let kinds: Vec<&PpKind> = toks[1..line_end].iter().map(|t| &t.kind).collect();
    let ident = |k: &PpKind, s: &str| matches!(k, PpKind::Ident(n) if n == s);
    let name = |k: &PpKind| match k {
        PpKind::Ident(n) => Some(n.clone()),
        _ => None,
    };
    match kinds.as_slice() {
        [d, g] if ident(d, "ifndef") => name(g),
        [d, PpKind::Punct(Punct::Bang), def, g] if ident(d, "if") && ident(def, "defined") => name(g),
        [d, PpKind::Punct(Punct::Bang), def, PpKind::Punct(Punct::LParen), g, PpKind::Punct(Punct::RParen)]
            if ident(d, "if") && ident(def, "defined") =>
        {
            name(g)
        }
        _ => None,
    }
}

/// Spell a query operand's tokens with no separators (`gnu :: packed` →
/// `gnu::packed`).
fn spell_joined(toks: &[PpTok]) -> String {
    toks.iter().map(|t| t.kind.spelling()).collect()
}

/// Strip the `__name__` decoration an attribute or builtin may be spelled
/// with.
fn undecorate(name: &str) -> &str {
    name.strip_prefix("__").and_then(|n| n.strip_suffix("__")).unwrap_or(name)
}

/// `__has_builtin(name)`: whether lf-cc implements the builtin `name` with its
/// documented meaning. Kept in sync with what the parser and `sema` accept:
/// the `va_*` family (dedicated AST nodes), `alloca`, `expect`, `constant_p`
/// (conservatively 0 — still a correct implementation), and the library-alias
/// builtins whose aliased function has the builtin's exact result type (the
/// pointer-returning string/memory functions and the `int`-returning
/// comparisons). Anything else — in particular builtins `sema` would silently
/// alias to an implicitly-declared `int` function — answers 0.
fn has_builtin(name: &str) -> bool {
    matches!(
        name,
        "__builtin_va_start"
            | "__builtin_va_arg"
            | "__builtin_va_end"
            | "__builtin_va_copy"
            | "__builtin_alloca"
            | "__builtin_expect"
            | "__builtin_constant_p"
            | "__builtin_memcpy"
            | "__builtin_memmove"
            | "__builtin_memset"
            | "__builtin_strcpy"
            | "__builtin_strncpy"
            | "__builtin_strcat"
            | "__builtin_strncat"
            | "__builtin_strchr"
            | "__builtin_strrchr"
            | "__builtin_strstr"
            | "__builtin_strpbrk"
            | "__builtin_memcmp"
            | "__builtin_strcmp"
            | "__builtin_strncmp"
    ) || (BUILTIN_VA_LIST_TYPE && name == "__builtin_va_list")
}

/// Whether the parser implements the `__builtin_va_list` type keyword. The
/// builtin `<stdarg.h>` spells `__gnuc_va_list` with it when available and
/// otherwise declares the psABI `struct __va_list_tag[1]` itself.
const BUILTIN_VA_LIST_TYPE: bool = false;

/// Whether the parser accepts GNU statement expressions `({ … })`, which C
/// library headers use in their `__OPTIMIZE__` macro forms. Until it does,
/// `-O1`+ does not predefine `__OPTIMIZE__`.
const STATEMENT_EXPRESSIONS: bool = false;

/// `__has_attribute(name)` (GNU `__attribute__` names, bare or `__x__`-
/// decorated): true only for attributes whose meaning lf-cc provides — which,
/// since attributes are parsed and then ignored, means the attributes that are
/// pure diagnostics or optimization hints (ignoring them is a correct
/// implementation). Attributes that change layout, linkage, or code (`aligned`,
/// `packed`, `section`, `alias`, `weak`, `cleanup`, `constructor`, `mode`,
/// `vector_size`, `gnu_inline`, `transparent_union`, …) answer 0.
fn has_gnu_attribute(name: &str) -> bool {
    let name = name.strip_prefix("gnu::").unwrap_or(name);
    matches!(
        undecorate(name),
        "noreturn"
            | "unused"
            | "used"
            | "maybe_unused"
            | "deprecated"
            | "unavailable"
            | "warn_unused_result"
            | "nonnull"
            | "returns_nonnull"
            | "nothrow"
            | "leaf"
            | "pure"
            | "const"
            | "malloc"
            | "alloc_size"
            | "alloc_align"
            | "format"
            | "format_arg"
            | "sentinel"
            | "cold"
            | "hot"
            | "noinline"
            | "noclone"
            | "noipa"
            | "always_inline"
            | "artificial"
            | "access"
            | "fallthrough"
            | "may_alias"
            | "warning"
            | "error"
            | "no_instrument_function"
            | "externally_visible"
            | "returns_twice"
    )
}

/// `__has_c_attribute(name)`: the C23 standard attributes lf-cc accepts (as
/// hints it may ignore), valued by the standard's date codes; `gnu::` ones per
/// [`has_gnu_attribute`]; 0 otherwise.
fn has_c_attribute(name: &str) -> i128 {
    if let Some(gnu) = name.strip_prefix("gnu::") {
        return i128::from(has_gnu_attribute(gnu));
    }
    match undecorate(name) {
        "deprecated" | "fallthrough" | "maybe_unused" => 201904,
        "nodiscard" => 202003,
        "noreturn" | "_Noreturn" => 202202,
        "unsequenced" | "reproducible" => 202207,
        _ => 0,
    }
}

/// `__has_feature(name)` / `__has_extension(name)`: the C language features
/// lf-cc implements under the selected standard; everything else (sanitizers,
/// modules, …) answers 0.
fn has_feature(name: &str, std: CStd) -> bool {
    match name {
        "c_alignas" | "c_alignof" | "c_static_assert" | "c_generic_selections" => std.is_c11(),
        _ => false,
    }
}

/// The characteristics of a binary floating format, spelled as the values of
/// the `__FLT_*__`/`__DBL_*__`/`__LDBL_*__` predefined macros (which the
/// builtin `<float.h>` is written in terms of).
#[derive(Debug)]
struct FloatFormat {
    /// `sizeof` of the type.
    size: &'static str,
    mant_dig: &'static str,
    dig: &'static str,
    min_exp: &'static str,
    min_10_exp: &'static str,
    max_exp: &'static str,
    max_10_exp: &'static str,
    decimal_dig: &'static str,
    max: &'static str,
    min: &'static str,
    epsilon: &'static str,
    denorm_min: &'static str,
    /// The constant suffix of the type's literals.
    suffix: &'static str,
}

/// IEEE-754 binary32 (`float`).
const FLT_FORMAT: FloatFormat = FloatFormat {
    size: "4",
    mant_dig: "24",
    dig: "6",
    min_exp: "(-125)",
    min_10_exp: "(-37)",
    max_exp: "128",
    max_10_exp: "38",
    decimal_dig: "9",
    max: "3.40282346638528859811704183484516925e+38",
    min: "1.17549435082228750796873653722224568e-38",
    epsilon: "1.19209289550781250000000000000000000e-7",
    denorm_min: "1.40129846432481707092372958328991613e-45",
    suffix: "F",
};

/// IEEE-754 binary64 (`double`).
const DBL_FORMAT: FloatFormat = FloatFormat {
    size: "8",
    mant_dig: "53",
    dig: "15",
    min_exp: "(-1021)",
    min_10_exp: "(-307)",
    max_exp: "1024",
    max_10_exp: "308",
    decimal_dig: "17",
    max: "1.79769313486231570814527423731704357e+308",
    min: "2.22507385850720138309023271733240406e-308",
    epsilon: "2.22044604925031308084726333618164062e-16",
    denorm_min: "4.94065645841246544176568792868221372e-324",
    suffix: "",
};

/// `long double` as lf-cc implements it today: binary64, 8 bytes (see
/// `ast::FloatTy`). The predefined `__LDBL_*__`/`__SIZEOF_LONG_DOUBLE__`
/// macros and so `<float.h>` describe this type, not the psABI's x87 format.
const LDBL_FORMAT: FloatFormat = FloatFormat { suffix: "L", ..DBL_FORMAT };
