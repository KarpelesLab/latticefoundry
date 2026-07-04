//! Presentation-agnostic diagnostics: spans, severities, and a collecting sink.
//!
//! These are the shared error-reporting types the parser and verifier build on
//! (Phase 2). They deliberately carry *no* rendering: a [`Diagnostic`] holds a
//! severity, a message, an optional source [`Span`], and a list of attached
//! notes, and a [`Diagnostics`] sink accumulates them and tracks whether any
//! error was seen. How a diagnostic is turned into text (colours, carets, source
//! snippets) is a separate concern layered on top later.
//!
//! Source locations are byte offsets into a file identified by a [`FileId`]
//! handle; resolving that handle to a path or to line/column is the job of a
//! source map that owns the files, not of these types.

/// A cheap, copyable handle identifying a source file within a source map.
///
/// Like the arena ids in [`crate::support::arena`], this is an opaque index; the
/// source map that owns the files maps it back to a path and contents.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct FileId(u32);

impl FileId {
    /// Wrap a raw file index.
    #[inline]
    pub fn new(index: u32) -> Self {
        FileId(index)
    }

    /// The raw index backing this handle.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }
}

/// A half-open byte range `[start, end)` within a single source [`FileId`].
///
/// Offsets are byte positions (not char or line/column), which keeps spans cheap
/// to produce and combine; a source map converts them to human coordinates for
/// display.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Span {
    /// The file this span points into.
    pub file: FileId,
    /// Inclusive start byte offset.
    pub start: u32,
    /// Exclusive end byte offset.
    pub end: u32,
}

impl Span {
    /// Create a span over `[start, end)` in `file`.
    #[inline]
    pub fn new(file: FileId, start: u32, end: u32) -> Self {
        debug_assert!(start <= end, "span start must not exceed end");
        Span { file, start, end }
    }

    /// An empty span (a caret position) at `offset` in `file`.
    #[inline]
    pub fn point(file: FileId, offset: u32) -> Self {
        Span {
            file,
            start: offset,
            end: offset,
        }
    }

    /// The length of the span in bytes.
    #[inline]
    pub fn len(self) -> u32 {
        self.end - self.start
    }

    /// Whether the span covers no bytes (`start == end`).
    #[inline]
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }

    /// Whether `offset` falls within `[start, end)`.
    #[inline]
    pub fn contains(self, offset: u32) -> bool {
        self.start <= offset && offset < self.end
    }

    /// The smallest span covering both `self` and `other`.
    ///
    /// Both spans must refer to the same file; this panics otherwise.
    #[inline]
    pub fn merge(self, other: Span) -> Span {
        assert_eq!(self.file, other.file, "cannot merge spans from different files");
        Span {
            file: self.file,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// How serious a diagnostic is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Severity {
    /// A problem that makes the result invalid.
    Error,
    /// A problem worth surfacing that does not invalidate the result.
    Warning,
    /// Extra context, typically attached to another diagnostic.
    Note,
}

impl Severity {
    /// Whether this severity denotes an error.
    #[inline]
    pub fn is_error(self) -> bool {
        matches!(self, Severity::Error)
    }
}

/// A subordinate note attached to a [`Diagnostic`], optionally located.
#[derive(Clone, Debug)]
pub struct Note {
    /// The note text.
    pub message: String,
    /// An optional source location the note refers to.
    pub span: Option<Span>,
}

/// A single diagnostic: a severity, a message, an optional primary [`Span`], and
/// any number of attached [`Note`]s.
///
/// Build one with [`Diagnostic::error`] / [`Diagnostic::warning`] and refine it
/// fluently with [`Diagnostic::with_span`] and [`Diagnostic::with_note`].
#[derive(Clone, Debug)]
pub struct Diagnostic {
    /// How serious the diagnostic is.
    pub severity: Severity,
    /// The primary human-readable message.
    pub message: String,
    /// The primary source location, if any.
    pub span: Option<Span>,
    /// Subordinate notes giving further context.
    pub notes: Vec<Note>,
}

impl Diagnostic {
    /// Create a diagnostic with an explicit severity.
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Diagnostic {
            severity,
            message: message.into(),
            span: None,
            notes: Vec::new(),
        }
    }

    /// Create an [`Severity::Error`] diagnostic.
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }

    /// Create a [`Severity::Warning`] diagnostic.
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }

    /// Attach a primary source span (builder style).
    #[must_use]
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Attach an unlocated note (builder style).
    #[must_use]
    pub fn with_note(mut self, message: impl Into<String>) -> Self {
        self.notes.push(Note {
            message: message.into(),
            span: None,
        });
        self
    }

    /// Attach a located note (builder style).
    #[must_use]
    pub fn with_note_at(mut self, message: impl Into<String>, span: Span) -> Self {
        self.notes.push(Note {
            message: message.into(),
            span: Some(span),
        });
        self
    }

    /// Whether this diagnostic is an error.
    #[inline]
    pub fn is_error(&self) -> bool {
        self.severity.is_error()
    }
}

/// A sink that collects [`Diagnostic`]s and tracks whether any error was seen.
///
/// This is the shared reporting channel a phase threads through its work: it
/// pushes diagnostics as it finds problems, and callers query [`has_errors`] to
/// decide whether to proceed.
///
/// [`has_errors`]: Diagnostics::has_errors
#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    items: Vec<Diagnostic>,
    error_count: usize,
}

impl Diagnostics {
    /// Create an empty sink.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a diagnostic, updating the error count.
    pub fn emit(&mut self, diagnostic: Diagnostic) {
        if diagnostic.is_error() {
            self.error_count += 1;
        }
        self.items.push(diagnostic);
    }

    /// Convenience: record an error with just a message.
    pub fn error(&mut self, message: impl Into<String>) {
        self.emit(Diagnostic::error(message));
    }

    /// Convenience: record a warning with just a message.
    pub fn warning(&mut self, message: impl Into<String>) {
        self.emit(Diagnostic::warning(message));
    }

    /// Whether any [`Severity::Error`] has been recorded.
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// The number of errors recorded.
    #[inline]
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// The total number of diagnostics recorded (all severities).
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether no diagnostics have been recorded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Iterate over the recorded diagnostics in emission order.
    pub fn iter(&self) -> impl Iterator<Item = &Diagnostic> {
        self.items.iter()
    }

    /// Consume the sink, yielding the collected diagnostics.
    pub fn into_vec(self) -> Vec<Diagnostic> {
        self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file() -> FileId {
        FileId::new(0)
    }

    #[test]
    fn span_helpers() {
        let s = Span::new(file(), 4, 10);
        assert_eq!(s.len(), 6);
        assert!(!s.is_empty());
        assert!(s.contains(4));
        assert!(s.contains(9));
        assert!(!s.contains(10));

        let p = Span::point(file(), 3);
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);

        let merged = Span::new(file(), 2, 5).merge(Span::new(file(), 8, 12));
        assert_eq!(merged, Span::new(file(), 2, 12));
    }

    #[test]
    #[should_panic(expected = "different files")]
    fn merge_across_files_panics() {
        let _ = Span::new(FileId::new(0), 0, 1).merge(Span::new(FileId::new(1), 0, 1));
    }

    #[test]
    fn diagnostic_builder() {
        let span = Span::new(file(), 1, 2);
        let d = Diagnostic::error("type mismatch")
            .with_span(span)
            .with_note("expected i32")
            .with_note_at("found i64 here", span);

        assert_eq!(d.severity, Severity::Error);
        assert!(d.is_error());
        assert_eq!(d.message, "type mismatch");
        assert_eq!(d.span, Some(span));
        assert_eq!(d.notes.len(), 2);
        assert_eq!(d.notes[0].message, "expected i32");
        assert_eq!(d.notes[0].span, None);
        assert_eq!(d.notes[1].span, Some(span));

        assert!(!Diagnostic::warning("unused value").is_error());
    }

    #[test]
    fn sink_tracks_errors() {
        let mut sink = Diagnostics::new();
        assert!(sink.is_empty());
        assert!(!sink.has_errors());

        sink.warning("a warning");
        assert!(!sink.has_errors());
        assert_eq!(sink.error_count(), 0);

        sink.error("first error");
        sink.emit(Diagnostic::error("second error").with_span(Span::point(file(), 0)));

        assert!(sink.has_errors());
        assert_eq!(sink.error_count(), 2);
        assert_eq!(sink.len(), 3);

        let severities: Vec<Severity> = sink.iter().map(|d| d.severity).collect();
        assert_eq!(
            severities,
            vec![Severity::Warning, Severity::Error, Severity::Error]
        );

        let all = sink.into_vec();
        assert_eq!(all.len(), 3);
    }
}
