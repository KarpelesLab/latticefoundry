//! GNU `asm` support: asm labels on declarations (the glibc `__REDIRECT`
//! mechanism), asm statements inside functions (see
//! `inline_asm.rs` for those with operands), and file-scope asm (collected, assembled with our own
//! assembler, and linked beside the C object).
//!
//! Tests that link against the host C library use `qld` through
//! `latticefoundry::link::gnu` and skip when no host C runtime is found; the
//! gcc-interop tests additionally skip when `gcc` is not installed.

use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::link::gnu::{HostCrt, host_c_link_args, link_gnu};
use latticefoundry::transform::pipeline::OptLevel;
use lf_cc::{BuildError, CStd, PpOptions};

// --- helpers ---------------------------------------------------------------

/// A fresh scratch directory for one test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-cc-asm-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn which(prog: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(prog))
        .find(|c| c.is_file())
        .map(|c| c.to_string_lossy().into_owned())
}

fn pp(std: &str) -> PpOptions {
    PpOptions { std: CStd::parse(std).expect("known std"), ..PpOptions::default() }
}

/// The first diagnostic message of a front-end failure.
fn frontend_error(src: &str, std: &str) -> String {
    match lf_cc::check_source_with(src, &pp(std)) {
        Ok(_) => panic!("expected a front-end error for:\n{src}"),
        Err(diags) => diags[0].message.clone(),
    }
}

/// The symbol name of every function signature: the names the object will
/// reference or define.
fn sig_symbols(src: &str) -> Vec<String> {
    let program = lf_cc::check_source_with(src, &pp("gnu17"))
        .unwrap_or_else(|d| panic!("check failed: {d:?}"));
    program.sigs.iter().map(|s| s.name.clone()).collect()
}

/// Compile `src` with lf-cc to an object in `dir`, returning its path and the
/// path of the assembled file-scope asm object (if any).
fn lf_object(dir: &Path, name: &str, src: &str, opts: &PpOptions, opt: OptLevel) -> Vec<PathBuf> {
    let input = format!("{name}.c");
    let out = lf_cc::compile_object_with(src, &input, opts, opt, false)
        .unwrap_or_else(|e| panic!("lf-cc failed on '{name}': {e:?}"));
    let obj = dir.join(format!("{name}.o"));
    std::fs::write(&obj, &out.object).unwrap();
    let mut objs = vec![obj];
    if let Some(asm_obj) = lf_cc::assemble_toplevel_asm(&out.toplevel_asm, &input)
        .unwrap_or_else(|e| panic!("assembling '{name}' asm: {e:?}"))
    {
        let p = dir.join(format!("{name}.asm.o"));
        std::fs::write(&p, asm_obj).unwrap();
        objs.push(p);
    }
    objs
}

/// Link `objects` against the host C library with qld and run the result,
/// returning its exit code and stdout.
fn link_and_run(crt: &HostCrt, dir: &Path, name: &str, objects: &[PathBuf]) -> (i32, String) {
    let exe = dir.join(name);
    let refs: Vec<&Path> = objects.iter().map(PathBuf::as_path).collect();
    let args = host_c_link_args(crt, &refs, &[], &exe);
    link_gnu("asm-test", &args).unwrap_or_else(|e| panic!("link of '{name}' failed: {e}"));
    let out = Command::new(&exe).output().expect("run executable");
    let code = out.status.code().expect("exited normally");
    (code, String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Build `src` alone (hosted) at -O0 and -O2 and assert both exit with `want`.
fn run_hosted(name: &str, src: &str, want: i32) {
    let Some(crt) = HostCrt::discover() else {
        eprintln!("skipping {name}: no host C runtime");
        return;
    };
    let dir = scratch(name);
    for opt in [OptLevel::O0, OptLevel::O2] {
        let tag = format!("{name}_{}", opt.name());
        let objs = lf_object(&dir, &tag, src, &pp("gnu17"), opt);
        let (code, _) = link_and_run(&crt, &dir, &tag, &objs);
        assert_eq!(code, want, "{tag}");
    }
}

// --- asm labels: front end -------------------------------------------------

#[test]
fn label_renames_the_symbol_but_not_the_c_name() {
    let syms = sig_symbols(
        "int my_abs(int) __asm__(\"abs\");\n\
         int main(void){ return my_abs(-1); }",
    );
    assert!(syms.contains(&"abs".to_owned()), "{syms:?}");
    assert!(!syms.contains(&"my_abs".to_owned()), "{syms:?}");
}

#[test]
fn label_spellings_concatenation_and_attributes() {
    // `asm`, `__asm`, `__asm__`; adjacent literals; attributes on both sides.
    let syms = sig_symbols(
        "int a(int) asm(\"sym_a\");\n\
         int b(int) __asm (\"sym\" \"_\" \"b\");\n\
         int c(int) __attribute__((__nothrow__)) __asm__ (\"\" \"sym_c\") __attribute__((pure));\n\
         extern int d(int) __attribute__((const)) __asm__(\"sym_d\") __attribute__((leaf));\n\
         int main(void){ return a(1) + b(2) + c(3) + d(4); }",
    );
    for s in ["sym_a", "sym_b", "sym_c", "sym_d"] {
        assert!(syms.contains(&s.to_owned()), "missing {s}: {syms:?}");
    }
}

#[test]
fn plain_asm_keyword_is_gnu_only() {
    // Under ISO C `asm` is an ordinary identifier, so the label does not parse;
    // the reserved spelling works under every standard.
    let msg = frontend_error("int f(int) asm(\"g\");", "c11");
    assert!(!msg.is_empty());
    let program = lf_cc::check_source_with("int f(int) __asm__(\"g\");", &pp("c11")).unwrap();
    assert_eq!(program.sigs[0].name, "g");
    // `asm` is still usable as a name in ISO mode.
    lf_cc::check_source_with("int asm = 3; int main(void){ return asm; }", &pp("c11")).unwrap();
}

#[test]
fn label_on_objects_and_block_scope_declarations() {
    let program = lf_cc::check_source_with(
        "extern int counter asm(\"real_counter\");\n\
         int table[4] __asm__(\"real_table\") = {1, 2, 3, 4};\n\
         int *p = &counter;\n\
         int main(void){\n\
           extern int other asm(\"real_other\");\n\
           int ext(void) __asm__(\"real_ext\");\n\
           static int s asm(\"real_static\") = 5;\n\
           return counter + other + ext() + s + (int)(sizeof table / sizeof table[0]);\n\
         }",
        &pp("gnu17"),
    )
    .unwrap();
    let globals: Vec<&str> = program.globals.iter().map(|g| g.name.as_str()).collect();
    for s in ["real_counter", "real_table", "real_other", "real_static"] {
        assert!(globals.contains(&s), "missing {s}: {globals:?}");
    }
    assert!(program.sigs.iter().any(|s| s.name == "real_ext"));
    // A global initialized with the address of a labeled object relocates
    // against the label.
    let p = program.globals.iter().find(|g| g.name == "p").unwrap();
    assert_eq!(p.relocs[0].symbol, "real_counter");
}

#[test]
fn two_c_names_bound_to_one_symbol_share_a_declaration() {
    // glibc declares both the redirected name and its target; they must not
    // become two IR functions of the same symbol.
    let syms = sig_symbols(
        "int my_abs(int) __asm__(\"abs\");\n\
         int abs(int);\n\
         int main(void){ return abs(-1) + my_abs(-2); }",
    );
    assert_eq!(syms.iter().filter(|s| *s == "abs").count(), 1, "{syms:?}");
}

#[test]
fn conflicting_labels_are_an_error() {
    let msg = frontend_error("int f(int) asm(\"a\");\nint f(int) asm(\"b\");", "gnu17");
    assert!(msg.contains("conflicts"), "{msg}");
    let msg = frontend_error("int f(int) asm(\"a\") asm(\"b\");", "gnu17");
    assert!(msg.contains("only one asm label"), "{msg}");
}

// --- asm labels: end to end ------------------------------------------------

#[test]
fn label_redirects_a_libc_call() {
    run_hosted(
        "redirect_libc",
        "int my_abs(int) __asm__(\"abs\");\n\
         long my_labs(long) asm(\"\" \"labs\");\n\
         int (*fp)(int) = my_abs;\n\
         int main(void){ return my_abs(-20) + (int)my_labs(-20L) + fp(-2); }",
        42,
    );
}

#[test]
fn labeled_declaration_then_unlabeled_definition() {
    // The definition inherits the label from the earlier declaration: the
    // function is emitted as `real_twice`, and calls (direct and through a
    // pointer) reach it under that name.
    run_hosted(
        "decl_then_def",
        "int twice(int) __asm__(\"real_twice\");\n\
         int value asm(\"real_value\");\n\
         int twice(int x){ return 2 * x; }\n\
         int value = 11;\n\
         int (*fp)(int) = twice;\n\
         int main(void){ return twice(value) + fp(10); }",
        42,
    );
}

#[test]
fn label_interoperates_with_gcc_objects() {
    // lf-cc defines `hidden_add` under the label `real_add`; a gcc-compiled
    // translation unit calls `real_add` and reads `real_base` by those names,
    // and lf-cc calls gcc's `gcc_side` through a labeled declaration.
    let Some(crt) = HostCrt::discover() else {
        eprintln!("skipping: no host C runtime");
        return;
    };
    let Some(gcc) = which("gcc") else {
        eprintln!("skipping: gcc not installed");
        return;
    };
    let dir = scratch("gcc_interop");
    let lf_src = "int hidden_add(int a, int b) __asm__(\"real_add\");\n\
                  int hidden_add(int a, int b){ return a + b; }\n\
                  int base asm(\"real_base\") = 30;\n\
                  int from_gcc(void) __asm__(\"gcc_side\");\n\
                  int main(void){ return from_gcc(); }";
    let gcc_src = "extern int real_add(int, int);\n\
                   extern int real_base;\n\
                   int gcc_side(void){ return real_add(real_base, 12); }";
    let gcc_c = dir.join("side.c");
    let gcc_o = dir.join("side.o");
    std::fs::write(&gcc_c, gcc_src).unwrap();
    let status = Command::new(gcc).args(["-c", "-O0", "-o"]).arg(&gcc_o).arg(&gcc_c).status().unwrap();
    assert!(status.success());
    let mut objs = lf_object(&dir, "main", lf_src, &pp("gnu17"), OptLevel::O0);
    objs.push(gcc_o);
    let (code, _) = link_and_run(&crt, &dir, "interop", &objs);
    assert_eq!(code, 42);
}

// --- glibc shapes ----------------------------------------------------------

#[test]
fn redirect_macro_shapes_parse() {
    // The shapes glibc's `__REDIRECT`/`__REDIRECT_NTH` macros expand to, written
    // out here: the label built from `__USER_LABEL_PREFIX__` (predefined empty)
    // and a stringized alias, with attributes before and after.
    let src = "#define STR(x) #x\n\
               #define XSTR(x) STR(x)\n\
               #define ASMNAME(c) XSTR(__USER_LABEL_PREFIX__) c\n\
               #define REDIR(name, proto, alias) name proto __asm__ (ASMNAME(#alias))\n\
               #define REDIR_NTH(name, proto, alias) \\\n\
                  name proto __asm__ (ASMNAME(#alias)) __attribute__((__nothrow__, __leaf__))\n\
               extern int REDIR (r1, (int __x), abs) __attribute__((__const__));\n\
               extern long int REDIR_NTH (r2, (long int __x), labs);\n\
               extern char *REDIR_NTH (r3, (const char *__s, int __c), strchr) \
                  __attribute__((__pure__)) __attribute__((__nonnull__ (1)));\n\
               extern int r4 (int __sig) __asm__ (\"__xpg_sigpause\");\n\
               int main(void){ return r1(-1) + (int)r2(-1L) + (r3(\"x\", 'x') != 0); }";
    let program = lf_cc::check_source_with(src, &pp("gnu17")).unwrap();
    let names: Vec<&str> = program.sigs.iter().map(|s| s.name.as_str()).collect();
    for s in ["abs", "labs", "strchr", "__xpg_sigpause"] {
        assert!(names.contains(&s), "missing {s}: {names:?}");
    }
}

#[test]
fn real_glibc_redirect_macros() {
    // The host's own <sys/cdefs.h> `__REDIRECT` machinery, end to end.
    if !Path::new("/usr/include/sys/cdefs.h").is_file() {
        eprintln!("skipping: no /usr/include/sys/cdefs.h");
        return;
    }
    let src = "#include <features.h>\n\
               #include <sys/cdefs.h>\n\
               extern int __REDIRECT (my_abs, (int __x), abs) __attribute_const__;\n\
               extern long int __REDIRECT_NTH (my_labs, (long int __x), labs) __wur;\n\
               int main(void){ return my_abs(-40) + (int)my_labs(-2L); }";
    let program = lf_cc::check_source_with(
        src,
        &PpOptions {
            // The host's system directories exactly as the driver searches them
        // (Debian/Ubuntu keep `bits/*` in the multiarch directory).
        stdinc_dirs: lf_cc::default_system_include_dirs(),
        hosted: true,
            builtin_headers: false,
            ..PpOptions::default()
        },
    )
    .unwrap_or_else(|d| panic!("{d:?}"));
    let names: Vec<&str> = program.sigs.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"abs") && names.contains(&"labs"), "{names:?}");
    assert!(!names.iter().any(|n| n.contains("my_") || n.contains("PREFIX")), "{names:?}");
}

#[test]
fn gnu_extension_keyword_is_ignored() {
    // glibc marks `long long` prototypes and the like with `__extension__`.
    run_hosted(
        "extension",
        "__extension__ extern long long int llabs(long long int);\n\
         __extension__ typedef unsigned long long u64;\n\
         int main(void){ u64 v = 2; return (int)llabs(-40LL) + (int)v; }",
        42,
    );
}

#[test]
fn real_glibc_string_h() {
    // The host's real <string.h> (not the builtin stub headers): asm labels,
    // `__extension__`, and attribute-laden prototypes all have to parse.
    let Some(crt) = HostCrt::discover() else {
        eprintln!("skipping: no host C runtime");
        return;
    };
    if !Path::new("/usr/include/string.h").is_file() {
        eprintln!("skipping: no /usr/include/string.h");
        return;
    }
    let opts = PpOptions {
        // The host's system directories exactly as the driver searches them
        // (Debian/Ubuntu keep `bits/*` in the multiarch directory).
        stdinc_dirs: lf_cc::default_system_include_dirs(),
        hosted: true,
        // The compiler-provided headers (`<stddef.h>`) come from lf-cc itself.
        main_file_name: "strlen.c".to_owned(),
        ..PpOptions::default()
    };
    let dir = scratch("string_h");
    let src = "#include <string.h>\n\
               int main(void){ return (int)strlen(\"forty-two characters, give or take a few..\"); }";
    let objs = lf_object(&dir, "strlen", src, &opts, OptLevel::O0);
    let (code, _) = link_and_run(&crt, &dir, "strlen", &objs);
    assert_eq!(code, 42);
}

// --- asm statements --------------------------------------------------------

#[test]
fn compiler_barriers_are_accepted() {
    run_hosted(
        "barriers",
        "int g;\n\
         int main(void){\n\
           int x = 40;\n\
           asm(\"\");\n\
           __asm__ volatile (\"\" ::: \"memory\");\n\
           __asm__ __volatile__ (\"\" : : : \"memory\", \"cc\");\n\
           __asm volatile (\"\" : : \"r\"(x), \"m\"(g));\n\
           asm inline (\"\");\n\
           asm volatile (\"\" : : \"g\"(x++));\n\
           g = 1; asm volatile(\"\" ::: \"memory\"); g++;\n\
           return x + g - 1;\n\
         }",
        42,
    );
}

#[test]
fn full_extended_asm_syntax_parses() {
    // Real-world extended asm parses (named operands, multiple sections) and
    // compiles (see `tests/inline_asm.rs` for execution); `asm goto` parses
    // and is then rejected with a clear diagnostic.
    for body in [
        "int r = 0, a = 1; asm(\"addl %1, %0\" : \"=r\"(r) : \"r\"(a), \"0\"(r));",
        "int r, a = 1; __asm__ __volatile__ (\"mov %[in], %[out]\" : [out] \"=r\" (r) : [in] \"r\" (a) : \"cc\");",
        "asm volatile (\"nop\");",
        "asm (\"pause\" \"\\n\\t\" \"pause\");",
    ] {
        let src = format!("int main(void){{ {body} return 0; }}");
        lf_cc::compile_object_with(&src, "t.c", &pp("gnu17"), OptLevel::O0, false)
            .unwrap_or_else(|e| panic!("{body}: {e:?}"));
    }
    let src = "int main(void){ int a = 1; asm goto (\"jmp %l[done]\" : : \"r\"(a) : \"memory\" : done); done: return 0; }";
    let msg = frontend_error(src, "gnu17");
    assert!(msg.contains("`asm goto` is not supported"), "{msg}");
}

// --- file-scope asm --------------------------------------------------------

#[test]
fn toplevel_asm_is_collected_in_order() {
    let program = lf_cc::check_source_with(
        "asm(\".text\");\n\
         int x = 1;\n\
         __asm__(\".globl f\\n\" \"f: ret\");\n",
        &pp("gnu17"),
    )
    .unwrap();
    assert_eq!(program.toplevel_asm, vec![".text".to_owned(), ".globl f\nf: ret".to_owned()]);
    let msg = frontend_error("asm(\"nop\" : : \"r\"(1));", "gnu17");
    assert!(msg.contains("file-scope asm"), "{msg}");
}

#[test]
fn toplevel_asm_rejected_where_it_would_be_dropped() {
    let src = "asm(\".globl f\\nf: ret\"); int main(void){ return 0; }";
    let opts = pp("gnu17");
    for r in [
        lf_cc::build_object_with(src, "t.c", &opts, OptLevel::O0, false).map(|_| ()),
        lf_cc::build_image_with(src, "t.c", &opts, OptLevel::O0, false).map(|_| ()),
    ] {
        match r {
            Err(BuildError::Backend(m)) => assert!(m.contains("file-scope asm"), "{m}"),
            other => panic!("expected a backend error, got {other:?}"),
        }
    }
    assert!(lf_cc::assemble_toplevel_asm(&[], "t.c").unwrap().is_none());
    let err = lf_cc::assemble_toplevel_asm(&["bogus_insn %rax".to_owned()], "t.c").unwrap_err();
    assert!(matches!(err, BuildError::Backend(ref m) if m.contains("file-scope asm")), "{err:?}");
}

#[test]
fn toplevel_asm_function_links_and_runs() {
    // A function written in file-scope asm, called from C (and calling back
    // into C), assembled with rsasm and linked with the C object by qld.
    let Some(crt) = HostCrt::discover() else {
        eprintln!("skipping: no host C runtime");
        return;
    };
    let dir = scratch("toplevel");
    let src = "int printf(const char *, ...);\n\
               int add_asm(int a, int b);\n\
               int c_helper(int v){ return v + 2; }\n\
               __asm__(\".text\\n\"\n\
                       \".globl add_asm\\n\"\n\
                       \".type add_asm, @function\\n\"\n\
                       \"add_asm:\\n\"\n\
                       \"    leal (%rdi,%rsi), %eax\\n\"\n\
                       \"    ret\\n\");\n\
               asm(\".globl via_c\\n\"\n\
                   \"via_c:\\n\"\n\
                   \"    subq $8, %rsp\\n\"\n\
                   \"    call c_helper\\n\"\n\
                   \"    addq $8, %rsp\\n\"\n\
                   \"    ret\");\n\
               int via_c(int);\n\
               int main(void){ printf(\"%d\\n\", add_asm(30, 10)); return via_c(add_asm(30, 10)); }";
    for opt in [OptLevel::O0, OptLevel::O2] {
        let tag = format!("toplevel_{}", opt.name());
        let objs = lf_object(&dir, &tag, src, &pp("gnu17"), opt);
        assert_eq!(objs.len(), 2, "expected a C object and an asm object");
        let (code, out) = link_and_run(&crt, &dir, &tag, &objs);
        assert_eq!((code, out.as_str()), (42, "40\n"), "{tag}");
    }
}
