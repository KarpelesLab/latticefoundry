//! The C standard-version model (`--std`).
//!
//! [`CStd`] records which revision of ISO C (and whether the GNU dialect) the
//! frontend is compiling against, and answers feature-availability questions the
//! lexer, preprocessor, and parser use to gate language features. The mapping is
//! written from the standards themselves (clean room, tenet T1): the feature
//! predicates below encode "which revision introduced this construct".
//!
//! GNU dialects (`gnuNN`) are, for this subset, feature-equivalent to the
//! matching ISO revision (`cNN`) plus a handful of GNU predefined macros
//! (`__GNUC__` and friends); see [`crate::preprocess`].

/// A selected C standard revision and dialect.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CStd {
    /// ISO C89 / C90 (ANSI C).
    C89,
    /// ISO C99.
    C99,
    /// ISO C11.
    C11,
    /// ISO C17 / C18.
    C17,
    /// ISO C23.
    C23,
    /// GNU C89 (C89 features + GNU predefined macros).
    Gnu89,
    /// GNU C99.
    Gnu99,
    /// GNU C11.
    Gnu11,
    /// GNU C17.
    Gnu17,
    /// GNU C23.
    Gnu23,
}

impl Default for CStd {
    /// `gnu17`, matching the default of a typical `gcc`, so the differential
    /// suite compiles the same dialect on both sides.
    fn default() -> Self {
        CStd::Gnu17
    }
}

impl CStd {
    /// Parse a `--std=` / `-std=` value (e.g. `c99`, `gnu11`, `c17`, `iso9899:1999`).
    /// Returns `None` for an unrecognized name.
    pub fn parse(name: &str) -> Option<CStd> {
        Some(match name {
            "c89" | "c90" | "iso9899:1990" | "iso9899:199409" => CStd::C89,
            "c99" | "c9x" | "iso9899:1999" => CStd::C99,
            "c11" | "c1x" | "iso9899:2011" => CStd::C11,
            "c17" | "c18" | "iso9899:2017" | "iso9899:2018" => CStd::C17,
            "c23" | "c2x" => CStd::C23,
            "gnu89" | "gnu90" => CStd::Gnu89,
            "gnu99" | "gnu9x" => CStd::Gnu99,
            "gnu11" | "gnu1x" => CStd::Gnu11,
            "gnu17" | "gnu18" => CStd::Gnu17,
            "gnu23" | "gnu2x" => CStd::Gnu23,
            _ => return None,
        })
    }

    /// The canonical name of this standard (as accepted by [`CStd::parse`]).
    pub fn name(self) -> &'static str {
        match self {
            CStd::C89 => "c89",
            CStd::C99 => "c99",
            CStd::C11 => "c11",
            CStd::C17 => "c17",
            CStd::C23 => "c23",
            CStd::Gnu89 => "gnu89",
            CStd::Gnu99 => "gnu99",
            CStd::Gnu11 => "gnu11",
            CStd::Gnu17 => "gnu17",
            CStd::Gnu23 => "gnu23",
        }
    }

    /// The ISO revision year-code used for ordering: 1989/1999/2011/2017/2023.
    /// GNU dialects share the year of their ISO base.
    fn year(self) -> u32 {
        match self {
            CStd::C89 | CStd::Gnu89 => 1989,
            CStd::C99 | CStd::Gnu99 => 1999,
            CStd::C11 | CStd::Gnu11 => 2011,
            CStd::C17 | CStd::Gnu17 => 2017,
            CStd::C23 | CStd::Gnu23 => 2023,
        }
    }

    /// Whether this is a GNU dialect (`gnuNN`).
    pub fn is_gnu(self) -> bool {
        matches!(self, CStd::Gnu89 | CStd::Gnu99 | CStd::Gnu11 | CStd::Gnu17 | CStd::Gnu23)
    }

    /// Whether the standard is at least C99.
    pub fn is_c99(self) -> bool {
        self.year() >= 1999
    }

    /// Whether the standard is at least C11.
    pub fn is_c11(self) -> bool {
        self.year() >= 2011
    }

    /// Whether the standard is at least C23.
    pub fn is_c23(self) -> bool {
        self.year() >= 2023
    }

    // --- feature predicates (each names the construct it gates) --------------

    /// `//` line comments (C99+, and all GNU dialects as an extension).
    pub fn line_comments(self) -> bool {
        self.is_c99() || self.is_gnu()
    }

    /// The `_Bool` type keyword (C99+).
    pub fn has_bool_type(self) -> bool {
        self.is_c99()
    }

    /// `long long` (C99+, and a GNU extension in every GNU dialect — gcc accepts
    /// it under `-std=gnu89` without a diagnostic).
    pub fn has_long_long(self) -> bool {
        self.is_c99() || self.is_gnu()
    }

    /// `restrict` / `inline` keywords (C99+, and accepted in every GNU dialect —
    /// gcc recognizes both under `-std=gnu89` as extensions).
    pub fn inline_restrict(self) -> bool {
        self.is_c99() || self.is_gnu()
    }

    /// Declarations intermixed with statements in a block (C99+, and a GNU
    /// extension in every GNU dialect — gcc accepts it under `-std=gnu89`).
    pub fn mixed_declarations(self) -> bool {
        self.is_c99() || self.is_gnu()
    }

    /// A declaration in the `for` initializer clause (C99+).
    pub fn for_loop_decls(self) -> bool {
        self.is_c99()
    }

    /// `_Static_assert` / `_Generic` / `_Alignof` / `_Alignas` / `_Noreturn`
    /// (the underscore-spelled keywords, C11+).
    pub fn static_assert_generic(self) -> bool {
        self.is_c11()
    }

    /// Compound literals `(T){ ... }` (C99+, and a GNU extension everywhere).
    pub fn compound_literals(self) -> bool {
        self.is_c99() || self.is_gnu()
    }

    /// Anonymous `struct`/`union` members (C11+, and a GNU extension everywhere).
    pub fn anonymous_members(self) -> bool {
        self.is_c11() || self.is_gnu()
    }

    /// The keyword-spelled `alignof` / `alignas` / `static_assert` / `bool`
    /// helpers that C23 promoted from `<stdalign.h>`/`<assert.h>` macros to real
    /// keywords (`_Alignof`/`_Alignas`/`_Static_assert` remain available under
    /// C11 via [`CStd::static_assert_generic`]).
    pub fn keyword_alignas(self) -> bool {
        self.is_c23()
    }

    /// The `noreturn` keyword spelling (C23; `_Noreturn` is the C11 spelling).
    pub fn keyword_noreturn(self) -> bool {
        self.is_c23()
    }

    /// `typeof` / `typeof_unqual` type specifiers (C23; `typeof` is also a GNU
    /// extension available in every GNU dialect).
    pub fn typeof_specifier(self) -> bool {
        self.is_c23() || self.is_gnu()
    }

    /// Attribute specifier sequences `[[...]]` (C23).
    pub fn attributes(self) -> bool {
        self.is_c23()
    }

    /// The `bool` / `true` / `false` keywords (C23; earlier they are ordinary
    /// identifiers, typically provided by `<stdbool.h>` macros).
    pub fn bool_keyword(self) -> bool {
        self.is_c23()
    }

    /// The `nullptr` keyword (C23).
    pub fn nullptr_keyword(self) -> bool {
        self.is_c23()
    }

    /// Binary integer literals `0b…` (C23).
    pub fn binary_literals(self) -> bool {
        self.is_c23()
    }

    /// Digit separators `'` in numeric literals (C23).
    pub fn digit_separators(self) -> bool {
        self.is_c23()
    }

    /// The value of the predefined `__STDC_VERSION__` macro (the `L`-suffixed
    /// long), or `None` for C89 where the macro is not defined.
    pub fn stdc_version(self) -> Option<u64> {
        match self.year() {
            1989 => None,
            1999 => Some(199901),
            2011 => Some(201112),
            2017 => Some(201710),
            _ => Some(202311),
        }
    }
}
