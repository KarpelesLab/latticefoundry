//! Preprocessor and `--std` tests: token-level unit checks for macro expansion,
//! conditionals, includes, and command-line macros, plus standard-gating checks
//! and an end-to-end differential-vs-gcc suite of preprocessor-heavy programs.

use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::transform::pipeline::OptLevel;
use lf_cc::cstd::CStd;
use lf_cc::lex::{Punct, Token, TokenKind};
use lf_cc::preprocess::{self, MacroOp, PpOptions};
use lf_cc::{check_source_with, CStd as CStdReexport};

// --- helpers ---------------------------------------------------------------

fn opts(std: CStd) -> PpOptions {
    PpOptions { std, main_file_name: "test.c".to_owned(), ..PpOptions::default() }
}

/// Preprocess `src` under `std` and return the final token stream.
fn pp(src: &str, std: CStd) -> Vec<Token> {
    preprocess::preprocess(src, &opts(std)).expect("preprocessing should succeed")
}

/// Preprocess under the default standard (gnu17).
fn pp_def(src: &str) -> Vec<Token> {
    pp(src, CStd::default())
}

/// A flat, readable rendering of a token stream for assertions.
fn render(toks: &[Token]) -> String {
    let mut s = String::new();
    for t in toks {
        match &t.kind {
            TokenKind::Ident(n) => s.push_str(n),
            TokenKind::Keyword(_) => s.push_str("<kw>"),
            TokenKind::IntLit(v, _) => s.push_str(&v.to_string()),
            TokenKind::FloatLit(v, _) => s.push_str(&v.to_string()),
            TokenKind::Str(v) => {
                s.push('"');
                s.push_str(v);
                s.push('"');
            }
            TokenKind::Punct(p) => s.push_str(punct(*p)),
            TokenKind::Eof => {}
        }
        s.push(' ');
    }
    s.trim_end().to_owned()
}

fn punct(p: Punct) -> &'static str {
    match p {
        Punct::LParen => "(",
        Punct::RParen => ")",
        Punct::Plus => "+",
        Punct::Star => "*",
        Punct::Comma => ",",
        Punct::Semi => ";",
        _ => "?",
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

// --- object / function-like macros -----------------------------------------

#[test]
fn object_like_macro() {
    let toks = pp_def("#define N 5\nN + N");
    assert_eq!(int_values(&toks), vec![5, 5]);
}

#[test]
fn function_like_macro() {
    let toks = pp_def("#define ADD(a,b) ((a)+(b))\nADD(1,2)");
    assert_eq!(render(&toks), "( ( 1 ) + ( 2 ) )");
}

#[test]
fn nested_and_rescanned() {
    let toks = pp_def("#define A B\n#define B 7\nA");
    assert_eq!(int_values(&toks), vec![7]);
}

#[test]
fn self_reference_is_suppressed() {
    // The classic blue-paint case: `f(a)` -> `f(2*(a))` must not recurse.
    let toks = pp_def("#define f(a) f(2*(a))\nf(9)");
    assert_eq!(render(&toks), "f ( 2 * ( 9 ) )");
}

#[test]
fn mutual_recursion_terminates() {
    // `x` and `y` reference each other; expansion must not loop.
    let toks = pp_def("#define x y\n#define y x\nx");
    assert_eq!(render(&toks), "x");
}

#[test]
fn stringize_operator() {
    let toks = pp_def("#define STR(x) #x\nSTR(a + b)");
    // A single string token whose contents preserve inter-token spacing.
    assert_eq!(render(&toks), "\"a + b\"");
}

#[test]
fn token_paste_operator() {
    let toks = pp_def("#define CAT(a,b) a##b\nCAT(foo, bar)");
    assert_eq!(render(&toks), "foobar");
}

#[test]
fn paste_forms_number() {
    let toks = pp_def("#define J(a,b) a##b\nJ(12, 34)");
    assert_eq!(int_values(&toks), vec![1234]);
}

#[test]
fn variadic_macro() {
    let toks = pp_def("#define CALL(f, ...) f(__VA_ARGS__)\nCALL(g, 1, 2, 3)");
    assert_eq!(render(&toks), "g ( 1 , 2 , 3 )");
}

#[test]
fn variadic_empty_with_comma_elision() {
    let toks = pp_def("#define LOG(fmt, ...) f(fmt, ##__VA_ARGS__)\nLOG(1)");
    // The trailing comma is elided when the variadic arguments are empty.
    assert_eq!(render(&toks), "f ( 1 )");
}

#[test]
fn undef_stops_expansion() {
    let toks = pp_def("#define N 5\n#undef N\nN");
    assert_eq!(render(&toks), "N");
}

// --- conditionals ----------------------------------------------------------

#[test]
fn if_defined_selection() {
    let toks = pp_def("#define FOO 1\n#if defined(FOO)\n10\n#else\n20\n#endif");
    assert_eq!(int_values(&toks), vec![10]);
}

#[test]
fn if_arithmetic_and_elif() {
    let src = "#define V 3\n#if V > 5\n1\n#elif V > 2\n2\n#else\n3\n#endif";
    assert_eq!(int_values(&pp_def(src)), vec![2]);
}

#[test]
fn ifndef_guard() {
    let toks = pp_def("#ifndef X\n42\n#endif");
    assert_eq!(int_values(&toks), vec![42]);
}

#[test]
fn undefined_identifier_is_zero_in_if() {
    let toks = pp_def("#if UNDEFINED_THING\n1\n#else\n2\n#endif");
    assert_eq!(int_values(&toks), vec![2]);
}

#[test]
fn nested_conditionals_skip_inactive() {
    let src = "#if 0\n#if 1\n1\n#endif\nbad\n#else\n99\n#endif";
    assert_eq!(int_values(&pp_def(src)), vec![99]);
}

#[test]
fn error_directive_fails() {
    let err = preprocess::preprocess("#error boom\n", &opts(CStd::default())).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("#error") && d.message.contains("boom")));
}

#[test]
fn error_in_inactive_branch_is_ignored() {
    let toks = pp_def("#if 0\n#error should not fire\n#endif\n7");
    assert_eq!(int_values(&toks), vec![7]);
}

// --- command-line macros ---------------------------------------------------

#[test]
fn command_line_define() {
    let mut o = opts(CStd::default());
    o.cmdline.push(MacroOp::Define("FLAG=41".to_owned()));
    let toks = preprocess::preprocess("FLAG", &o).unwrap();
    assert_eq!(int_values(&toks), vec![41]);
}

#[test]
fn command_line_define_defaults_to_one() {
    let mut o = opts(CStd::default());
    o.cmdline.push(MacroOp::Define("ON".to_owned()));
    let toks = preprocess::preprocess("#if ON\n5\n#endif", &o).unwrap();
    assert_eq!(int_values(&toks), vec![5]);
}

#[test]
fn command_line_undef() {
    let mut o = opts(CStd::default());
    o.cmdline.push(MacroOp::Define("N=9".to_owned()));
    o.cmdline.push(MacroOp::Undef("N".to_owned()));
    let toks = preprocess::preprocess("N", &o).unwrap();
    assert_eq!(render(&toks), "N");
}

// --- __LINE__ / __FILE__ / #line -------------------------------------------

#[test]
fn line_macro() {
    // Line 1 is the blank line before; the literal is on line 2.
    let toks = pp_def("\n__LINE__");
    assert_eq!(int_values(&toks), vec![2]);
}

#[test]
fn line_directive_overrides() {
    let toks = pp_def("#line 100\n__LINE__");
    assert_eq!(int_values(&toks), vec![100]);
}

#[test]
fn file_macro() {
    let toks = pp_def("__FILE__");
    assert_eq!(render(&toks), "\"test.c\"");
}

// --- predefined macros -----------------------------------------------------

#[test]
fn stdc_version_per_std() {
    let cases = [
        (CStd::C99, 199901),
        (CStd::C11, 201112),
        (CStd::C17, 201710),
        (CStd::C23, 202311),
    ];
    for (std, want) in cases {
        let toks = pp("__STDC_VERSION__", std);
        assert_eq!(int_values(&toks), vec![want], "std {:?}", std);
    }
    // C89 does not define __STDC_VERSION__ (identifier passes through).
    let toks = pp("__STDC_VERSION__", CStd::C89);
    assert_eq!(render(&toks), "__STDC_VERSION__");
}

#[test]
fn target_predefined_macros() {
    for m in ["__x86_64__", "__LP64__", "__linux__", "__ELF__"] {
        let src = format!("#ifdef {m}\n1\n#else\n0\n#endif");
        assert_eq!(int_values(&pp_def(&src)), vec![1], "{m}");
    }
    // __GNUC__ only in GNU dialects.
    assert_eq!(int_values(&pp("#ifdef __GNUC__\n1\n#else\n0\n#endif", CStd::Gnu17)), vec![1]);
    assert_eq!(int_values(&pp("#ifdef __GNUC__\n1\n#else\n0\n#endif", CStd::C17)), vec![0]);
}

// --- #include --------------------------------------------------------------

#[test]
fn include_quoted_header() {
    let dir = tempdir("inc-quoted");
    std::fs::write(dir.join("h.h"), "#define FROM_HEADER 123\n").unwrap();
    let mut o = opts(CStd::default());
    o.include_dirs.push(dir.clone());
    let toks = preprocess::preprocess("#include \"h.h\"\nFROM_HEADER", &o).unwrap();
    assert_eq!(int_values(&toks), vec![123]);
}

#[test]
fn include_guard_prevents_double() {
    let dir = tempdir("inc-guard");
    std::fs::write(
        dir.join("g.h"),
        "#ifndef G_H\n#define G_H\n#define ONCE 1\n#endif\n",
    )
    .unwrap();
    let mut o = opts(CStd::default());
    o.include_dirs.push(dir.clone());
    // Including twice must not redefine or error.
    let toks =
        preprocess::preprocess("#include <g.h>\n#include <g.h>\nONCE", &o).unwrap();
    assert_eq!(int_values(&toks), vec![1]);
}

#[test]
fn missing_include_errors() {
    let err = preprocess::preprocess("#include \"nope.h\"\n", &opts(CStd::default())).unwrap_err();
    assert!(err.iter().any(|d| d.message.contains("cannot find include file")));
}

// --- std gating (checked via the full frontend) ----------------------------

fn check_ok(src: &str, std: CStd) -> bool {
    check_source_with(src, &opts(std)).is_ok()
}

fn check_err_contains(src: &str, std: CStd, needle: &str) {
    let err = check_source_with(src, &opts(std)).unwrap_err();
    assert!(
        err.iter().any(|d| d.message.contains(needle)),
        "expected a diagnostic containing {needle:?}, got: {:?}",
        err.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn c89_rejects_line_comments() {
    check_err_contains("int f(){ return 1; }\n// a comment\n", CStd::C89, "C99");
    assert!(check_ok("int f(){ return 1; }\n// a comment\n", CStd::C99));
}

#[test]
fn c89_rejects_long_long() {
    check_err_contains("long long f(){ return 1; }", CStd::C89, "long long");
    assert!(check_ok("long long f(){ return 1; }", CStd::C99));
}

#[test]
fn c89_rejects_bool() {
    check_err_contains("int f(){ _Bool b = 1; return b; }", CStd::C89, "_Bool");
    assert!(check_ok("int f(){ _Bool b = 1; return b; }", CStd::C99));
}

#[test]
fn c89_rejects_mixed_declarations() {
    let src = "int f(){ int x = 1; x = x + 1; int y = 2; return y; }";
    check_err_contains(src, CStd::C89, "C99");
    assert!(check_ok(src, CStd::C99));
}

#[test]
fn c89_rejects_for_loop_declaration() {
    let src = "int f(){ int s = 0; for (int i = 0; i < 3; i++) s = s + i; return s; }";
    check_err_contains(src, CStd::C89, "C99");
    assert!(check_ok(src, CStd::C99));
}

#[test]
fn c23_keywords_gated() {
    // `bool`/`true` are ordinary identifiers before C23, keywords in C23.
    assert!(check_ok("int f(){ bool b = 1; return b; }", CStd::C23));
    assert!(check_ok("int f(){ int nullptr = 3; return nullptr; }", CStd::C17));
}

#[test]
fn reexport_is_the_same_type() {
    let _a: CStdReexport = CStd::C11;
}

// --- differential vs gcc ---------------------------------------------------

fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-cc-pp-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn which(prog: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(prog)).find(|c| c.is_file())
}

fn run_exit(bin: &Path) -> i32 {
    Command::new(bin).status().expect("run binary").code().expect("exit code")
}

/// A preprocessor differential case.
struct Case {
    name: &'static str,
    src: &'static str,
    std: CStd,
    defines: Vec<(&'static str, &'static str)>,
    /// `(filename, contents)` headers written into a temp `-I` directory.
    headers: Vec<(&'static str, &'static str)>,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "func_macros_paste",
            src: "#define CAT(a,b) a##b\n#define ADD(x,y) ((x)+(y))\n\
                   int CAT(ma,in)(void){ return ADD(20, 22); }",
            std: CStd::Gnu17,
            defines: vec![],
            headers: vec![],
        },
        Case {
            name: "conditional_compilation",
            src: "#define LEVEL 2\n\
                   #if LEVEL == 1\n#define R 10\n#elif LEVEL == 2\n#define R 42\n#else\n#define R 0\n#endif\n\
                   int main(void){ return R; }",
            std: CStd::Gnu17,
            defines: vec![],
            headers: vec![],
        },
        Case {
            name: "ifdef_and_cmdline_define",
            src: "#ifdef WANT\nint main(void){ return WANT + 1; }\n#else\nint main(void){ return 0; }\n#endif",
            std: CStd::Gnu17,
            defines: vec![("WANT", "40")],
            headers: vec![],
        },
        Case {
            name: "variadic_sum",
            src: "#define SUM3(a,b,c) sum_impl(a,b,c)\n#define APPLY(f, ...) f(__VA_ARGS__)\n\
                   int sum_impl(int a,int b,int c){ return a+b+c; }\n\
                   int main(void){ return APPLY(SUM3, 10, 15, 20) - 3; }",
            std: CStd::Gnu17,
            defines: vec![],
            headers: vec![],
        },
        Case {
            name: "included_header",
            src: "#include \"mathy.h\"\n\
                   int square(int x){ return x*x; }\n\
                   int main(void){ return square(6) + CONST_OFFSET; }",
            std: CStd::Gnu17,
            defines: vec![],
            headers: vec![(
                "mathy.h",
                "#ifndef MATHY_H\n#define MATHY_H\n#define CONST_OFFSET 6\nint square(int x);\n#endif\n\
                 #ifdef MATHY_IMPL\n#endif\n",
            )],
        },
        Case {
            name: "square_impl",
            src: "int square(int x){ return x*x; }\n#include \"proto.h\"\n\
                   int main(void){ return use(); }",
            std: CStd::Gnu17,
            defines: vec![],
            headers: vec![(
                "proto.h",
                "int square(int x);\n#define use() (square(5) + 5)\n",
            )],
        },
        Case {
            name: "stdc_version_c17",
            src: "int main(void){ return __STDC_VERSION__ == 201710L ? 0 : 1; }",
            std: CStd::C17,
            defines: vec![],
            headers: vec![],
        },
        Case {
            name: "stdc_version_c99",
            src: "int main(void){ return __STDC_VERSION__ == 199901L ? 7 : 1; }",
            std: CStd::C99,
            defines: vec![],
            headers: vec![],
        },
    ]
}

fn std_flag(std: CStd) -> String {
    format!("-std={}", std.name())
}

#[test]
fn preprocessor_differential_against_gcc() {
    let dir = tempdir("diff");
    let gcc = which("gcc").or_else(|| which("cc"));
    let mut matched = 0usize;
    let mut ran = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for c in cases() {
        // Build the PpOptions and -I dir with any headers.
        let inc = dir.join(c.name);
        std::fs::create_dir_all(&inc).unwrap();
        for (fname, content) in &c.headers {
            std::fs::write(inc.join(fname), content).unwrap();
        }
        let o = PpOptions {
            std: c.std,
            include_dirs: vec![inc.clone()],
            cmdline: c.defines.iter().map(|(n, v)| MacroOp::Define(format!("{n}={v}"))).collect(),
            main_file_name: format!("{}.c", c.name),
        };

        let o0 = build_run(&dir, c.name, c.src, &o, OptLevel::O0);
        let o2 = build_run(&dir, c.name, c.src, &o, OptLevel::O2);
        ran += 1;

        if o0 != o2 {
            failures.push(format!("{}: lf-cc -O0={o0} != -O2={o2}", c.name));
            continue;
        }

        if let Some(gcc) = &gcc {
            let cfile = dir.join(format!("{}.c", c.name));
            std::fs::write(&cfile, c.src).unwrap();
            let bin = dir.join(format!("{}.gcc", c.name));
            let mut cmd = Command::new(gcc);
            cmd.arg(std_flag(c.std)).arg("-O0").arg("-w").arg(format!("-I{}", inc.display()));
            for (n, v) in &c.defines {
                cmd.arg(format!("-D{n}={v}"));
            }
            cmd.arg("-o").arg(&bin).arg(&cfile);
            let status = cmd.status().expect("run gcc");
            assert!(status.success(), "gcc failed on {}", c.name);
            let g = run_exit(&bin);
            if g != o0 {
                failures.push(format!("{}: lf-cc={o0} != gcc={g}", c.name));
                continue;
            }
            matched += 1;
        }
    }

    eprintln!(
        "pp-differential: {ran} programs ran; {matched} matched gcc ({})",
        if gcc.is_some() { "present" } else { "absent" }
    );
    assert!(failures.is_empty(), "mismatches:\n{}", failures.join("\n"));
}

fn build_run(dir: &Path, name: &str, src: &str, o: &PpOptions, opt: OptLevel) -> i32 {
    let image = lf_cc::build_image_with(src, &format!("{name}.c"), o, opt, false)
        .unwrap_or_else(|e| panic!("lf-cc failed on {name}: {e:?}"));
    let bin = dir.join(format!("{name}.lf.{}", opt.name()));
    latticefoundry::link::write_executable(bin.to_str().unwrap(), &image).expect("write exe");
    run_exit(&bin)
}
