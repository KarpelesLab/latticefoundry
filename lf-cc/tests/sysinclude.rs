//! Header search and system-header interoperability: the include search chain
//! (`-iquote`/`-I`/`-isystem`/builtin/standard/`-idirafter`), `#include_next`,
//! the builtin headers' `__need_*` protocol and hosted layering, the `__has_*`
//! queries, the predefined macros, `#if` arithmetic, and diagnostics located in
//! the header they arise in. The last group compiles against the host's real
//! `/usr/include` and skips when it is absent.

use std::path::{Path, PathBuf};
use std::process::Command;

use lf_cc::cstd::CStd;
use lf_cc::lex::{Token, TokenKind};
use lf_cc::preprocess::{self, PpOptions};

// --- helpers ---------------------------------------------------------------

/// A per-test scratch directory, removed when the test finishes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lf-cc-sysinc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    /// Write `text` to `rel` (creating parent directories) and return its path.
    fn write(&self, rel: &str, text: &str) -> PathBuf {
        let p = self.0.join(rel);
        std::fs::create_dir_all(p.parent().expect("has a parent")).expect("mkdir");
        std::fs::write(&p, text).expect("write file");
        p
    }

    fn dir(&self, rel: &str) -> PathBuf {
        let p = self.0.join(rel);
        std::fs::create_dir_all(&p).expect("mkdir");
        p
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn opts() -> PpOptions {
    PpOptions { main_file_name: "t.c".to_owned(), ..PpOptions::default() }
}

fn pp_with(src: &str, o: &PpOptions) -> Vec<Token> {
    match preprocess::preprocess(src, o) {
        Ok(t) => t,
        Err(d) => panic!("preprocessing failed: {d:?}"),
    }
}

fn int_values(toks: &[Token]) -> Vec<i128> {
    toks.iter()
        .filter_map(|t| match t.kind {
            TokenKind::IntLit(v, _) => Some(v),
            _ => None,
        })
        .collect()
}

/// The integer values of the expressions in `src` (one per line after the
/// includes), preprocessed with `o`.
fn values(src: &str, o: &PpOptions) -> Vec<i128> {
    int_values(&pp_with(src, o))
}

/// Preprocess, parse and type-check; return the rendered diagnostics on error.
fn check(src: &str, o: &PpOptions) -> Result<(), String> {
    lf_cc::check_source_mapped(src, o).map(|_| ()).map_err(|(d, map)| map.render(&d))
}

// --- search chain & #include_next -------------------------------------------

#[test]
fn include_next_walks_the_chain() {
    let s = Scratch::new("next");
    let a = s.dir("a");
    let b = s.dir("b");
    let c = s.dir("c");
    s.write("a/n.h", "#define A_SEEN 1\n#include_next <n.h>\n");
    s.write("b/n.h", "#define B_SEEN 2\n#include_next <n.h>\n");
    s.write(
        "c/n.h",
        "#define C_SEEN 3\n#if __has_include_next(<n.h>)\n#error chain should end here\n#endif\n",
    );
    let o = PpOptions { include_dirs: vec![a, b, c], ..opts() };
    assert_eq!(values("#include <n.h>\nA_SEEN B_SEEN C_SEEN", &o), vec![1, 2, 3]);
    // `"…"` include_next from a header also resumes after its directory.
    s.write("a/q.h", "#include_next \"q.h\"\n");
    s.write("b/q.h", "#define Q 4\n");
    assert_eq!(values("#include \"q.h\"\nQ", &o), vec![4]);
}

#[test]
fn search_order_of_the_directory_kinds() {
    let s = Scratch::new("order");
    let quote = s.dir("quote");
    let inc = s.dir("inc");
    let sys = s.dir("sys");
    let std_dir = s.dir("std");
    let after = s.dir("after");
    // Each directory defines WHO for a header present in it and everything
    // after it; the first one on the chain must win.
    for (dir, who, names) in [
        ("quote", 1, &["h1.h"][..]),
        ("inc", 2, &["h1.h", "h2.h"][..]),
        ("sys", 3, &["h1.h", "h2.h", "h3.h"][..]),
        ("std", 4, &["h1.h", "h2.h", "h3.h", "h4.h"][..]),
        ("after", 5, &["h1.h", "h2.h", "h3.h", "h4.h", "h5.h"][..]),
    ] {
        for n in names {
            s.write(&format!("{dir}/{n}"), &format!("{who}\n"));
        }
    }
    let o = PpOptions {
        quote_dirs: vec![quote],
        include_dirs: vec![inc],
        system_dirs: vec![sys],
        stdinc_dirs: vec![std_dir],
        after_dirs: vec![after],
        ..opts()
    };
    let src = "#include \"h1.h\"\n#include \"h2.h\"\n#include <h1.h>\n#include <h3.h>\n\
               #include <h4.h>\n#include <h5.h>\n";
    // `-iquote` only serves "…"; <h1.h> therefore comes from -I.
    assert_eq!(values(src, &o), vec![1, 2, 2, 3, 4, 5]);
}

#[test]
fn builtin_headers_sit_between_isystem_and_the_standard_dirs() {
    let s = Scratch::new("builtin-pos");
    let sys = s.dir("sys");
    let std_dir = s.dir("std");
    s.write("std/stdbool.h", "#define FROM_STD_DIR 1\n");
    s.write("sys/stdalign.h", "#define FROM_ISYSTEM 1\n");
    let o = PpOptions { system_dirs: vec![sys], stdinc_dirs: vec![std_dir], ..opts() };
    // The builtin <stdbool.h> shadows the standard directory's...
    assert_eq!(values("#include <stdbool.h>\n#ifdef FROM_STD_DIR\n9\n#endif\ntrue", &o), vec![1]);
    // ...but -isystem shadows the builtin <stdalign.h>.
    assert_eq!(values("#include <stdalign.h>\nFROM_ISYSTEM", &o), vec![1]);
}

#[test]
fn hosted_limits_and_stdint_layer_over_the_c_library() {
    let s = Scratch::new("layer");
    let std_dir = s.dir("std");
    // A stand-in C library: its <limits.h> must see the compiler's limits
    // already in place (it would otherwise chain back to them).
    s.write(
        "std/limits.h",
        "#ifndef _GCC_LIMITS_H_\n#error compiler limits not marked\n#endif\n#define LIBC_PATH_MAX 4096\n",
    );
    s.write("std/stdint.h", "#define LIBC_STDINT 1\ntypedef long int64_t;\n");
    let hosted = PpOptions { stdinc_dirs: vec![std_dir.clone()], hosted: true, ..opts() };
    assert_eq!(values("#include <limits.h>\nINT_MAX LIBC_PATH_MAX", &hosted), vec![2147483647, 4096]);
    assert_eq!(values("#include <stdint.h>\nLIBC_STDINT\n#ifdef INT8_MAX\n9\n#endif", &hosted), vec![1]);
    // Freestanding: the builtin headers stand alone.
    let free = PpOptions { stdinc_dirs: vec![std_dir], hosted: false, ..opts() };
    assert_eq!(
        values("#include <limits.h>\n#include <stdint.h>\n#ifdef LIBC_PATH_MAX\n9\n#endif\nINT8_MAX", &free),
        vec![127]
    );
}

#[test]
fn nostdinc_drops_builtins() {
    let o = PpOptions { builtin_headers: false, ..opts() };
    let err = preprocess::preprocess("#include <stddef.h>\n", &o).expect_err("no builtin <stddef.h>");
    assert!(err.iter().any(|d| d.message.contains("stddef.h")), "{err:?}");
}

#[test]
fn include_guard_optimization_respects_undef() {
    let s = Scratch::new("guard");
    s.write("g.h", "#ifndef G_H\n#define G_H\n42\n#endif\n");
    // Not a pure guard: a token follows the #endif.
    s.write("ng.h", "#ifndef NG_H\n#define NG_H\n#endif\n7\n");
    let o = PpOptions { include_dirs: vec![s.path().to_path_buf()], ..opts() };
    let src = "#include \"g.h\"\n#include \"g.h\"\n#undef G_H\n#include \"g.h\"\n\
               #include \"ng.h\"\n#include \"ng.h\"\n";
    assert_eq!(values(src, &o), vec![42, 42, 7, 7]);
}

// --- the __need_* protocol ------------------------------------------------------

#[test]
fn stddef_need_protocol() {
    let src = "#define __need_size_t\n#include <stddef.h>\n\
               #if defined NULL || defined __need_size_t || defined offsetof\n#error leaked\n#endif\n\
               size_t a;\n\
               #define __need_NULL\n#define __need_wint_t\n#include <stddef.h>\n\
               #if !defined NULL || !defined _WINT_T || defined __need_NULL\n#error missing\n#endif\n\
               wint_t w;\n\
               #include <stddef.h>\n\
               ptrdiff_t p; wchar_t c; size_t b; void *q = NULL;\n\
               int main(void){ return (int)offsetof(struct { int x; int y; }, y); }\n";
    check(src, &opts()).expect("the __need_* sequence compiles");
    // max_align_t is a C11 addition.
    check("#include <stddef.h>\nmax_align_t m;\n", &PpOptions { std: CStd::C11, ..opts() })
        .expect("C11 max_align_t");
}

#[test]
fn stdarg_need_va_list_protocol() {
    let src = "#define __need___va_list\n#include <stdarg.h>\n\
               #if defined va_start || defined __need___va_list\n#error leaked\n#endif\n\
               int vf(const char *f, __gnuc_va_list ap);\n\
               #include <stdarg.h>\n\
               int sum(int n, ...){ va_list ap; int s = 0; va_start(ap, n);\n\
               while (n--) s += va_arg(ap, int); va_end(ap); return s; }\n\
               int main(void){ return sum(3, 1, 2, 3); }\n";
    check(src, &opts()).expect("the __need___va_list sequence compiles");
    // A C library that declared va_list itself (guarded by _VA_LIST_DEFINED)
    // does not get a second typedef.
    let src2 = "#define __need___va_list\n#include <stdarg.h>\n\
                typedef __gnuc_va_list va_list;\n#define _VA_LIST_DEFINED\n\
                #include <stdarg.h>\nint main(void){ return 0; }\n";
    check(src2, &opts()).expect("va_list declared once");
}

// --- __has_* queries ----------------------------------------------------------

/// Whether `#if expr` is taken under `o`.
fn taken(expr: &str, o: &PpOptions) -> bool {
    values(&format!("#if {expr}\n1\n#else\n0\n#endif\n"), o) == vec![1]
}

#[test]
fn has_queries() {
    let s = Scratch::new("has");
    s.write("present.h", "\n");
    let o = PpOptions { include_dirs: vec![s.path().to_path_buf()], ..opts() };
    for op in [
        "__has_include",
        "__has_include_next",
        "__has_builtin",
        "__has_attribute",
        "__has_c_attribute",
        "__has_feature",
        "__has_extension",
    ] {
        assert!(taken(&format!("defined {op} && defined({op})"), &o), "{op} is defined");
    }
    let yes = [
        "__has_include(<stddef.h>)",
        "__has_include(\"present.h\")",
        "__has_include(<present.h>)",
        "__has_builtin(__builtin_expect)",
        "__has_builtin(__builtin_va_arg)",
        "__has_builtin(__builtin_memcpy)",
        "__has_attribute(noreturn)",
        "__has_attribute(__format__)",
        "__has_attribute(__nonnull__)",
        "__has_c_attribute(nodiscard) == 202003",
        "__has_c_attribute(deprecated) == 201904",
        "__has_c_attribute(gnu::unused)",
    ];
    let no = [
        "__has_include(<nonexistent/nope.h>)",
        "__has_builtin(__builtin_bswap32)",
        "__has_builtin(__builtin_add_overflow)",
        "__has_builtin(__builtin_strlen)",
        "__has_attribute(aligned)",
        "__has_attribute(__packed__)",
        "__has_attribute(__gnu_inline__)",
        "__has_attribute(__mode__)",
        "__has_c_attribute(no_such_attribute)",
        "__has_feature(address_sanitizer)",
    ];
    for q in yes {
        assert!(taken(q, &o), "{q} should hold");
    }
    for q in no {
        assert!(!taken(q, &o), "{q} should not hold");
    }
    // A macro-named header, and a query produced by macro expansion.
    assert!(taken("__has_include(HDR)", &PpOptions {
        cmdline: vec![lf_cc::MacroOp::Define("HDR=<present.h>".to_owned())],
        ..o.clone()
    }));
    let src = "#define HAS_ATTR(x) __has_attribute(x)\n#if HAS_ATTR(__cold__)\n1\n#endif\n";
    assert_eq!(values(src, &o), vec![1]);
    // Feature queries follow the selected standard.
    assert!(taken("__has_feature(c_static_assert)", &PpOptions { std: CStd::C11, ..o.clone() }));
    assert!(!taken("__has_feature(c_static_assert)", &PpOptions { std: CStd::C99, ..o }));
}

// --- predefined macros ----------------------------------------------------------

#[test]
fn predefined_macros_describe_lf_cc() {
    // The data-model macros agree with the sizes sema actually uses.
    let src = "\
_Static_assert(sizeof(short) == __SIZEOF_SHORT__, \"short\");
_Static_assert(sizeof(int) == __SIZEOF_INT__, \"int\");
_Static_assert(sizeof(long) == __SIZEOF_LONG__, \"long\");
_Static_assert(sizeof(long long) == __SIZEOF_LONG_LONG__, \"long long\");
_Static_assert(sizeof(void *) == __SIZEOF_POINTER__, \"pointer\");
_Static_assert(sizeof(float) == __SIZEOF_FLOAT__, \"float\");
_Static_assert(sizeof(double) == __SIZEOF_DOUBLE__, \"double\");
_Static_assert(sizeof(long double) == __SIZEOF_LONG_DOUBLE__, \"long double\");
_Static_assert(sizeof(__SIZE_TYPE__) == __SIZEOF_SIZE_T__, \"size_t\");
_Static_assert(sizeof(__WCHAR_TYPE__) == __SIZEOF_WCHAR_T__, \"wchar_t\");
_Static_assert(sizeof(__WINT_TYPE__) == __SIZEOF_WINT_T__, \"wint_t\");
_Static_assert(sizeof(__INT64_TYPE__) == 8 && sizeof(__UINT8_TYPE__) == 1, \"exact\");
_Static_assert(__INT64_C(1) << 40 == 1099511627776L, \"INT64_C\");
_Static_assert(__INT_MAX__ == 2147483647 && __LONG_MAX__ == 9223372036854775807L, \"max\");
_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && __CHAR_BIT__ == 8, \"order\");
int main(void) { return 0; }
";
    for std in [CStd::Gnu17, CStd::C11, CStd::C23] {
        check(src, &PpOptions { std, ..opts() }).unwrap_or_else(|e| panic!("{std:?}: {e}"));
    }
    let probe = |o: &PpOptions, m: &str| -> Vec<i128> {
        values(&format!("#ifdef {m}\n1\n#else\n0\n#endif\n"), o)
    };
    let gnu = opts();
    assert_eq!(values("__GNUC__ __GNUC_MINOR__ __STDC_HOSTED__", &gnu), vec![4, 2, 0]);
    assert_eq!(probe(&gnu, "__STRICT_ANSI__"), vec![0]);
    assert_eq!(probe(&gnu, "__OPTIMIZE__"), vec![0]);
    assert_eq!(probe(&gnu, "__NO_INLINE__"), vec![1]);
    assert_eq!(probe(&gnu, "__SIZEOF_INT128__"), vec![0], "no __int128");
    assert_eq!(probe(&gnu, "__SSE2__"), vec![0], "no vector intrinsics");
    let iso = PpOptions { std: CStd::C17, ..opts() };
    assert_eq!(probe(&iso, "__STRICT_ANSI__"), vec![1]);
    assert_eq!(probe(&iso, "linux"), vec![0], "non-reserved names only in GNU modes");
    assert_eq!(probe(&gnu, "linux"), vec![1]);
    let hosted_opt = PpOptions { hosted: true, optimize: true, ..opts() };
    // `__OPTIMIZE__` waits for statement-expression support (glibc's <ctype.h>
    // macro forms need it), so optimizing does not yet define it.
    assert_eq!(values("__STDC_HOSTED__", &hosted_opt), vec![1]);
    assert_eq!(probe(&hosted_opt, "__OPTIMIZE__"), vec![0]);
    // __COUNTER__ and __INCLUDE_LEVEL__.
    assert_eq!(values("__COUNTER__ __COUNTER__ __INCLUDE_LEVEL__", &gnu), vec![0, 1, 0]);
}

/// The target data-model macros match the host gcc's (skipped without gcc).
#[test]
fn predefined_data_model_matches_gcc() {
    let Ok(out) = Command::new("gcc").args(["-dM", "-E", "-x", "c", "/dev/null"]).output() else {
        eprintln!("gcc not available; skipping");
        return;
    };
    let gcc = String::from_utf8_lossy(&out.stdout).into_owned();
    let names = [
        "__CHAR_BIT__", "__SIZEOF_SHORT__", "__SIZEOF_INT__", "__SIZEOF_LONG__",
        "__SIZEOF_LONG_LONG__", "__SIZEOF_POINTER__", "__SIZEOF_SIZE_T__", "__SIZEOF_PTRDIFF_T__",
        "__SIZEOF_WCHAR_T__", "__SIZEOF_WINT_T__", "__SIZEOF_FLOAT__", "__SIZEOF_DOUBLE__",
        "__SCHAR_MAX__", "__SHRT_MAX__", "__INT_MAX__", "__LONG_MAX__", "__LONG_LONG_MAX__",
        "__WCHAR_MAX__", "__WINT_MAX__", "__PTRDIFF_MAX__", "__SIZE_MAX__", "__INTMAX_MAX__",
        "__UINTMAX_MAX__", "__INTPTR_MAX__", "__UINTPTR_MAX__", "__INT8_MAX__", "__INT16_MAX__",
        "__INT32_MAX__", "__INT64_MAX__", "__UINT8_MAX__", "__UINT16_MAX__", "__UINT32_MAX__",
        "__UINT64_MAX__", "__INT_FAST16_MAX__", "__UINT_FAST32_MAX__", "__INT_LEAST8_MAX__",
        "__ORDER_LITTLE_ENDIAN__", "__FLT_MANT_DIG__", "__DBL_MANT_DIG__", "__FLT_MAX_EXP__",
        "__DBL_MIN_EXP__", "__DBL_DIG__", "__FLT_DIG__", "__INT_WIDTH__", "__LONG_WIDTH__",
        "__SIZE_WIDTH__",
    ];
    let o = PpOptions { std: CStd::Gnu17, ..opts() };
    for name in names {
        let Some(line) = gcc.lines().find(|l| l.starts_with(&format!("#define {name} "))) else {
            continue;
        };
        let gcc_val = line[format!("#define {name} ").len()..].trim();
        let want = values(&format!("{gcc_val}\n"), &o);
        let got = values(&format!("{name}\n"), &o);
        assert_eq!(got, want, "{name}: gcc says {gcc_val}");
    }
    for name in ["__SIZE_TYPE__", "__PTRDIFF_TYPE__", "__WCHAR_TYPE__", "__WINT_TYPE__",
        "__INTMAX_TYPE__", "__UINTMAX_TYPE__", "__INT64_TYPE__", "__UINT32_TYPE__",
        "__INTPTR_TYPE__", "__CHAR16_TYPE__", "__CHAR32_TYPE__", "__SIG_ATOMIC_TYPE__"]
    {
        let Some(line) = gcc.lines().find(|l| l.starts_with(&format!("#define {name} "))) else {
            continue;
        };
        let gcc_val = line[format!("#define {name} ").len()..].trim();
        // Compare as types: the typedefs must be compatible.
        let src = format!("typedef {gcc_val} a_t; typedef {name} a_t; int main(void){{ return 0; }}\n");
        check(&src, &o).unwrap_or_else(|e| panic!("{name} vs gcc's '{gcc_val}': {e}"));
    }
}

// --- #if arithmetic, macros ------------------------------------------------------

#[test]
fn if_arithmetic_uses_intmax_and_uintmax() {
    let src = "\
#if -1 > 0u
1
#endif
#if ~0UL == 0xffffffffffffffff
2
#endif
#if -1 < 0 && (-1) / 2 == 0 && -7 % 3 == -1
3
#endif
#if 0xffffffffffffffff > 0 && 18446744073709551615u / 2 == 0x7fffffffffffffff
4
#endif
#if 0 && (1 / 0)
#else
5
#endif
#if 1 || (1 / 0)
6
#endif
#if (1 ? 2 : (1 / 0)) == 2 && (0 ? -1 : 0u) == 0 && (1 ? -1 : 0u) > 0
7
#endif
";
    assert_eq!(values(src, &opts()), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn gnu_named_variadic_and_pragma_operator() {
    let src = "#define SUM(first, rest...) first + rest\n\
               #define STR(args...) #args\n\
               #define OPT(fmt, args...) f(fmt, ## args)\n\
               _Pragma(\"GCC diagnostic push\") SUM(1, 2, 3)\n";
    assert_eq!(values(src, &opts()), vec![1, 2, 3]);
    let toks = pp_with("#define STR(args...) #args\nSTR(a, b)", &opts());
    assert!(matches!(&toks[0].kind, TokenKind::Str(s, _) if s == b"a, b"), "{toks:?}");
}

// --- diagnostic locations ---------------------------------------------------------

#[test]
fn errors_in_headers_point_into_the_header() {
    let s = Scratch::new("diag");
    s.write("inc/outer.h", "int ok1;\n#include \"inner.h\"\n");
    s.write("inc/inner.h", "/* line 1 */\nint ok2;\nint bad = ;\n");
    s.write("inc/sema.h", "\nint f(void) { return undeclared_thing; }\n");
    s.write("inc/pp.h", "\n\n#error from the header\n");
    let o = PpOptions { include_dirs: vec![s.path().join("inc")], ..opts() };

    // A parse error: header path, line and column, and the include chain.
    let e = check("int a;\n#include \"outer.h\"\nint main(void){return 0;}\n", &o)
        .expect_err("syntax error in inner.h");
    let inner = s.path().join("inc/inner.h").display().to_string();
    let outer = s.path().join("inc/outer.h").display().to_string();
    assert!(e.contains(&format!("{inner}:3:11: error")), "{e}");
    assert!(e.contains(&format!("In file included from {outer}:2:")), "{e}");
    assert!(e.contains("from t.c:2:"), "{e}");

    // A type error inside a header function.
    let e = check("#include \"sema.h\"\nint main(void){return 0;}\n", &o).expect_err("sema error");
    let sema = s.path().join("inc/sema.h").display().to_string();
    assert!(e.contains(&format!("{sema}:2:")), "{e}");

    // A preprocessor error.
    let e = check("\n#include \"pp.h\"\n", &o).expect_err("#error");
    let pph = s.path().join("inc/pp.h").display().to_string();
    assert!(e.contains(&format!("{pph}:3:1: error: #error from the header")), "{e}");
    assert!(e.contains("In file included from t.c:2:"), "{e}");

    // Errors in the main file keep plain main-file offsets.
    let (d, map) = lf_cc::check_source_mapped("int x = ;\n", &opts()).expect_err("error");
    let span = d[0].span.expect("spanned");
    assert_eq!(map.locate(span.start).map(|l| (l.file, l.line)), Some(("t.c".to_owned(), 1)));
}

/// The driver prints header diagnostics at the header's own location.
#[test]
fn driver_reports_header_locations() {
    let s = Scratch::new("drvdiag");
    s.write("h.h", "\n\nstruct s { int a };\n");
    s.write("m.c", "#include \"h.h\"\nint main(void){return 0;}\n");
    let out = Command::new(env!("CARGO_BIN_EXE_lf-cc"))
        .args(["-c", "m.c"])
        .current_dir(s.path())
        .output()
        .expect("run lf-cc");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(err.contains("In file included from m.c:1:"), "{err}");
    assert!(err.contains("h.h:3:"), "{err}");
}

// --- the host's real /usr/include ---------------------------------------------------

/// With the driver's defaults (standard directories searched, hosted), the
/// real `<stdio.h>` and `<limits.h>` resolve and compile. Skipped when the
/// host has no `/usr/include/stdio.h`.
#[test]
fn default_search_finds_system_headers() {
    if !Path::new("/usr/include/stdio.h").is_file() || !Path::new("/usr/include/limits.h").is_file() {
        eprintln!("no /usr/include C library headers; skipping");
        return;
    }
    let dirs = lf_cc::default_system_include_dirs();
    assert!(dirs.contains(&PathBuf::from("/usr/include")), "{dirs:?}");
    let o = PpOptions { stdinc_dirs: dirs, hosted: true, ..opts() };
    check(
        "#include <stdio.h>\n#include <limits.h>\n\
         int main(void){ printf(\"%d\\n\", INT_MAX); return CHAR_BIT == 8 ? 0 : 1; }\n",
        &o,
    )
    .unwrap_or_else(|e| panic!("real <stdio.h>/<limits.h> should compile:\n{e}"));
    // The hosted <limits.h> reaches the C library's POSIX limits too.
    let posix = values("#include <limits.h>\n#ifdef _POSIX_ARG_MAX\n1\n#else\n0\n#endif\n", &o);
    let glibc = std::fs::read_to_string("/usr/include/limits.h").unwrap_or_default().contains("_LIBC_LIMITS_H_");
    if glibc {
        assert_eq!(posix, vec![1]);
    }
    // `-nostdinc`-equivalent options do not see them.
    let bare = PpOptions { builtin_headers: false, ..opts() };
    assert!(preprocess::preprocess("#include <stdio.h>\n", &bare).is_err());
}
