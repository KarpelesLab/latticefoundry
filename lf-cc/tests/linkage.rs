//! Symbol linkage, visibility, position-independent code and shared objects:
//! `static` → internal linkage, `__attribute__((visibility(...)))` and
//! `-fvisibility=`, `__attribute__((weak))`, `-fPIC`/`-fPIE`, `-shared` (with
//! `-Wl,-soname,...`) and `-pie`, plus C99 plain-`inline` semantics across
//! translation units.
//!
//! The hosted tests need the host C runtime and skip without it; the ones that
//! cross-check with gcc (a gcc-built program `dlopen`ing an lf-cc shared
//! object, an lf-cc `-fPIC` object in a gcc-linked library) also skip without
//! gcc.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use latticefoundry::link::gnu::HostCrt;
use latticefoundry::mc::object::SymbolBinding;
use latticefoundry::transform::pipeline::OptLevel;
use lf_cc::PpOptions;

const LF_CC: &str = env!("CARGO_BIN_EXE_lf-cc");

/// A per-test scratch directory, removed when the test finishes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lf-cc-linktest-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    fn write(&self, name: &str, text: &str) {
        std::fs::write(self.0.join(name), text).expect("write source");
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// Run `lf-cc` with `args` in this directory; panic with its stderr on failure.
    fn lf_cc(&self, args: &[&str]) {
        let out = Command::new(LF_CC).args(args).current_dir(&self.0).output().expect("run lf-cc");
        assert!(out.status.success(), "lf-cc {args:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    }

    /// Run `gcc` with `args` in this directory; panic with its stderr on failure.
    fn gcc(&self, gcc: &Path, args: &[&str]) {
        let out = Command::new(gcc).args(args).current_dir(&self.0).output().expect("run gcc");
        assert!(out.status.success(), "gcc {args:?} failed:\n{}", String::from_utf8_lossy(&out.stderr));
    }

    /// Run the executable `exe` (in this directory) with `args`.
    fn run(&self, exe: &str, args: &[&str]) -> Output {
        let path = self.0.join(exe);
        for _ in 0..50 {
            match Command::new(&path).args(args).current_dir(&self.0).output() {
                // A freshly written binary may briefly be busy (ETXTBSY).
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(20))
                }
                other => return other.expect("run executable"),
            }
        }
        panic!("{} stayed busy", path.display());
    }

    fn run_stdout(&self, exe: &str, args: &[&str]) -> String {
        let out = self.run(exe, args);
        assert!(out.status.success(), "{exe} failed: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).into_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn host_crt() -> bool {
    let found = HostCrt::discover().is_some();
    if !found {
        eprintln!("skipping: no host C runtime (crt1.o) found");
    }
    found
}

fn which(prog: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(prog)).find(|p| p.is_file())
}

/// The ELF `e_type` of a file (2 = executable, 3 = shared object / PIE).
fn elf_type(path: &Path) -> u16 {
    let bytes = std::fs::read(path).expect("read ELF");
    u16::from_le_bytes([bytes[16], bytes[17]])
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// A library: a counter, an exported datum, a hidden function, a `static`
/// helper, a protected function, a pointer table, a function returning a
/// function pointer, and a call into libc.
const LIB_C: &str = r#"
#include <stdio.h>
#include <string.h>
static int counter;
int shared_value = 40;
__attribute__((visibility("hidden"))) int hidden_helper(int x) { return x * 2; }
static int local_helper(int x) { return x + 1; }
__attribute__((visibility("protected"))) int protected_fn(int x) { return x - 1; }
const char *names[] = { "zero", "one", "two" };
int lib_add(int a, int b) { counter++; return hidden_helper(a) + local_helper(b) + shared_value + protected_fn(1); }
int lib_count(void) { return counter; }
const char *lib_name(int i) { return names[i]; }
int (*lib_fp(void))(int, int) { return lib_add; }
int lib_fmt(const char *s) { char buf[32]; snprintf(buf, sizeof buf, "<%s>", s); return (int)strlen(buf); }
"#;

/// A program that `dlopen`s the library named on its command line.
const HOST_C: &str = r#"
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
    int (*add)(int, int) = (int (*)(int, int))dlsym(h, "lib_add");
    int (*count)(void) = (int (*)(void))dlsym(h, "lib_count");
    const char *(*name)(int) = (const char *(*)(int))dlsym(h, "lib_name");
    int (*(*fp)(void))(int, int) = (int (*(*)(void))(int, int))dlsym(h, "lib_fp");
    int (*fmt)(const char *) = (int (*)(const char *))dlsym(h, "lib_fmt");
    int *value = (int *)dlsym(h, "shared_value");
    printf("add=%d\n", add(1, 2));
    *value = 100;
    printf("add=%d\n", add(3, 4));
    printf("count=%d name=%s same_fp=%d fmt=%d\n", count(), name(2), fp() == add, fmt("abc"));
    printf("hidden=%d static=%d protected=%d\n", dlsym(h, "hidden_helper") != 0,
           dlsym(h, "local_helper") != 0, dlsym(h, "protected_fn") != 0);
    return 0;
}
"#;

const HOST_OUT: &str = "add=45\nadd=111\ncount=2 name=two same_fp=1 fmt=5\nhidden=0 static=0 protected=1\n";

#[test]
fn shared_library_is_dlopened_by_lf_cc_and_gcc_programs() {
    if !host_crt() {
        return;
    }
    let s = Scratch::new("shared");
    s.write("lib.c", LIB_C);
    s.write("host.c", HOST_C);
    // `-shared` compiles the source as PIC by default; `-Wl,-soname,` names it.
    s.lf_cc(&["-shared", "-O2", "-Wl,-soname,libdemo.so.1", "lib.c", "-o", "libdemo.so"]);
    assert_eq!(elf_type(&s.path("libdemo.so")), 3);
    let lib = std::fs::read(s.path("libdemo.so")).unwrap();
    assert!(contains(&lib, b"libdemo.so.1"), "DT_SONAME is recorded");
    let so = s.path("libdemo.so");
    let so = so.to_str().unwrap();

    s.lf_cc(&["host.c", "-ldl", "-o", "host_lf"]);
    assert_eq!(s.run_stdout("host_lf", &[so]), HOST_OUT);
    if let Some(gcc) = which("gcc") {
        s.gcc(&gcc, &["host.c", "-o", "host_gcc", "-ldl"]);
        assert_eq!(s.run_stdout("host_gcc", &[so]), HOST_OUT);
    }

    // `-fPIC -c` then `-shared` from the object: the same library.
    s.lf_cc(&["-fPIC", "-c", "lib.c", "-o", "lib.o"]);
    s.lf_cc(&["-shared", "lib.o", "-o", "libobj.so"]);
    let so2 = s.path("libobj.so");
    assert_eq!(s.run_stdout("host_lf", &[so2.to_str().unwrap()]), HOST_OUT);
}

#[test]
fn fvisibility_hidden_exports_only_default_visibility_symbols() {
    if !host_crt() {
        return;
    }
    let s = Scratch::new("fvis");
    s.write(
        "lib.c",
        "int internal_api(int x) { return x + 1; }\n\
         int internal_data = 5;\n\
         __attribute__((visibility(\"default\"))) int public_api(int x) { return internal_api(x) * internal_data; }\n",
    );
    s.write(
        "host.c",
        "#include <dlfcn.h>\n#include <stdio.h>\n\
         int main(int argc, char **argv) {\n\
           void *h = dlopen(argv[1], RTLD_NOW);\n\
           if (!h) return 1;\n\
           int (*f)(int) = (int (*)(int))dlsym(h, \"public_api\");\n\
           printf(\"%d %d %d\\n\", f ? f(1) : -1, dlsym(h, \"internal_api\") != 0, dlsym(h, \"internal_data\") != 0);\n\
           return 0;\n\
         }\n",
    );
    s.lf_cc(&["-shared", "-fPIC", "-fvisibility=hidden", "lib.c", "-o", "libvis.so"]);
    s.lf_cc(&["host.c", "-ldl", "-o", "host"]);
    let so = s.path("libvis.so");
    assert_eq!(s.run_stdout("host", &[so.to_str().unwrap()]), "10 0 0\n");
}

#[test]
fn pic_object_links_into_a_gcc_shared_library() {
    if !host_crt() {
        return;
    }
    let Some(gcc) = which("gcc") else {
        eprintln!("skipping: gcc is not installed");
        return;
    };
    let s = Scratch::new("gccso");
    s.write("lib.c", LIB_C);
    s.write("host.c", HOST_C);
    s.lf_cc(&["-fPIC", "-O2", "-c", "lib.c", "-o", "lib.o"]);
    s.gcc(&gcc, &["-shared", "lib.o", "-o", "libgcc.so"]);
    s.gcc(&gcc, &["host.c", "-o", "host", "-ldl"]);
    let so = s.path("libgcc.so");
    assert_eq!(s.run_stdout("host", &[so.to_str().unwrap()]), HOST_OUT);
}

const PIE_MAIN_C: &str = r#"
#include <stdio.h>
#include <stdlib.h>
extern int counter;
int bump(int by);
static const char *greeting = "pie";
int main(void) {
    int (*fp)(int) = bump;
    char *buf = malloc(16);
    snprintf(buf, 16, "%s:%d", greeting, fp(2) + bump(3));
    puts(buf);
    free(buf);
    return counter;
}
"#;

const PIE_UTIL_C: &str = "int counter = 1;\nstatic int twice(int x) { return 2 * x; }\nint bump(int by) { counter += by; return twice(counter); }\n";

#[test]
fn position_independent_executables_run() {
    if !host_crt() {
        return;
    }
    let s = Scratch::new("pie");
    s.write("main.c", PIE_MAIN_C);
    s.write("util.c", PIE_UTIL_C);
    // `-pie` compiles the sources as PIE code by default.
    s.lf_cc(&["-pie", "main.c", "util.c", "-o", "prog"]);
    assert_eq!(elf_type(&s.path("prog")), 3, "a PIE is ET_DYN");
    let out = s.run("prog", &[]);
    assert_eq!(String::from_utf8_lossy(&out.stdout), "pie:18\n");
    assert_eq!(out.status.code(), Some(6));
    // Separately compiled `-fPIE` objects, linked with `-pie`.
    s.lf_cc(&["-fPIE", "-O2", "-c", "main.c", "util.c"]);
    s.lf_cc(&["-pie", "main.o", "util.o", "-o", "prog2"]);
    assert_eq!(String::from_utf8_lossy(&s.run("prog2", &[]).stdout), "pie:18\n");
    // `-fPIC` code is valid in an executable too.
    s.lf_cc(&["-fPIC", "-pie", "main.c", "util.c", "-o", "prog3"]);
    assert_eq!(String::from_utf8_lossy(&s.run("prog3", &[]).stdout), "pie:18\n");
    if let Some(gcc) = which("gcc") {
        s.gcc(&gcc, &["-pie", "main.o", "util.o", "-o", "prog_gcc"]);
        assert_eq!(String::from_utf8_lossy(&s.run("prog_gcc", &[]).stdout), "pie:18\n");
    }
}

#[test]
fn pic_macros_follow_the_flags() {
    let s = Scratch::new("picmacro");
    s.write(
        "m.c",
        "int main(void) {\n\
         #if defined __PIE__\n  return __PIC__ * 10 + __PIE__;\n\
         #elif defined __PIC__\n  return __PIC__ * 10;\n\
         #else\n  return 0;\n#endif\n}\n",
    );
    for (flag, want) in [("-fPIC", 20), ("-fpic", 10), ("-fPIE", 22), ("-fpie", 11), ("-fno-pic", 0)] {
        let exe = format!("m{}", flag.trim_start_matches('-'));
        s.lf_cc(&["-nostdlib", flag, "m.c", "-o", &exe]);
        assert_eq!(s.run(&exe, &[]).status.code(), Some(want), "{flag}");
    }
}

const WEAK_A_C: &str = r#"
#include <stdio.h>
int weak_fn(void) __attribute__((weak));
int weak_val __attribute__((weak)) = 7;
extern int missing_weak(void) __attribute__((weak));
extern int missing_data __attribute__((weak));
int main(void) {
    printf("%d %d %d %d\n", weak_fn ? weak_fn() : -1, weak_val, missing_weak ? 1 : 0, &missing_data ? 1 : 0);
    return 0;
}
"#;

#[test]
fn weak_definitions_yield_and_weak_references_may_stay_unresolved() {
    if !host_crt() {
        return;
    }
    let s = Scratch::new("weak");
    s.write("a.c", WEAK_A_C);
    s.write("b.c", "int weak_fn(void) { return 42; }\nint weak_val = 9;\n");
    // Alone: the weak definition stands and the weak references are null.
    s.lf_cc(&["a.c", "-o", "alone"]);
    assert_eq!(s.run_stdout("alone", &[]), "-1 7 0 0\n");
    // With strong definitions: they win over the weak ones.
    s.lf_cc(&["a.c", "b.c", "-o", "both"]);
    assert_eq!(s.run_stdout("both", &[]), "42 9 0 0\n");
    s.lf_cc(&["-pie", "a.c", "b.c", "-o", "both_pie"]);
    assert_eq!(s.run_stdout("both_pie", &[]), "42 9 0 0\n");
}

/// The binding of the object symbol `name` in `src` compiled alone.
fn symbol_binding(src: &str, name: &str) -> Option<(SymbolBinding, bool)> {
    let opts = PpOptions { std: lf_cc::CStd::parse("gnu17").unwrap(), ..PpOptions::default() };
    let unit = lf_cc::compile_module_with(src, "t.c", &opts, OptLevel::O0, false).expect("compiles");
    let obj = unit.module;
    let id = obj.symbol_id(name)?;
    let sym = obj.symbol(id);
    Some((sym.binding, sym.is_undefined()))
}

#[test]
fn static_functions_and_objects_have_internal_linkage_in_the_ir() {
    let src = "static int helper(int x) { return x; }\n\
               static int table[2] = {1, 2};\n\
               __attribute__((visibility(\"hidden\"))) int hid(void) { return 0; }\n\
               __attribute__((weak)) int wk(void) { return 1; }\n\
               int api(int i) { return helper(table[i]) + hid() + wk(); }\n";
    let (module, syms) = lf_cc::compile_to_ir(src, "t.c", false).expect("compiles");
    let text = latticefoundry::ir::text::print_module(&module, &syms);
    assert!(text.contains("func internal @helper"), "{text}");
    assert!(text.contains("func hidden @hid"), "{text}");
    assert!(text.contains("func weak @wk"), "{text}");
    assert!(text.contains("func @api"), "{text}");
    assert_eq!(symbol_binding(src, "helper"), Some((SymbolBinding::Local, false)));
    assert_eq!(symbol_binding(src, "table"), Some((SymbolBinding::Local, false)));
    assert_eq!(symbol_binding(src, "wk"), Some((SymbolBinding::Weak, false)));
    assert_eq!(symbol_binding(src, "api"), Some((SymbolBinding::Global, false)));
}

/// The object symbols of `src` compiled at `opt`.
fn defined_symbols(src: &str, opt: OptLevel) -> Vec<String> {
    let opts = PpOptions { std: lf_cc::CStd::parse("gnu17").unwrap(), ..PpOptions::default() };
    let unit = lf_cc::compile_module_with(src, "t.c", &opts, opt, false).expect("compiles");
    let obj = unit.module;
    obj.symbols().iter().filter(|s| !s.is_undefined()).map(|s| s.name.clone()).collect()
}

#[test]
fn function_addresses_are_ir_function_references_not_alias_globals() {
    // Taking a function's address is a `func_ref`: no body-less `global @f :
    // ptr` shadows the function, and a `static` function named only by data
    // is kept referenced through an address constant.
    let src = "static int f(int x) { return x; }\n\
               static int g(int x) { return -x; }\n\
               static int (*tab[])(int) = { g };\n\
               int (*get(void))(int) { return f; }\n\
               int use(int i) { return tab[0](i); }\n";
    let (module, syms) = lf_cc::compile_to_ir(src, "t.c", false).expect("compiles");
    let text = latticefoundry::ir::text::print_module(&module, &syms);
    assert!(!text.contains("@f : ptr") && !text.contains("@g : ptr"), "{text}");
    assert!(text.contains("func internal @f"), "{text}");
    assert!(text.contains("ptr @g"), "the table's function is referenced by an address constant:\n{text}");
}

#[test]
fn unused_static_functions_are_removed_at_o2() {
    let src = "static int unused(int x) { return x * 3 + 1; }\n\
               static int unused_cycle_a(int x);\n\
               static int unused_cycle_b(int x) { return x ? unused_cycle_a(x - 1) : 0; }\n\
               static int unused_cycle_a(int x) { return x ? unused_cycle_b(x - 1) : 1; }\n\
               static int via_data(int x) { return x - 1; }\n\
               static int via_value(int x) { return x + 2; }\n\
               static int (*const tab[])(int) = { via_data };\n\
               int (*get(void))(int) { return via_value; }\n\
               int api(int i) { return tab[0](i); }\n";
    let o0 = defined_symbols(src, OptLevel::O0);
    assert!(o0.iter().any(|s| s == "unused"), "-O0 keeps every function: {o0:?}");
    let o2 = defined_symbols(src, OptLevel::O2);
    for gone in ["unused", "unused_cycle_a", "unused_cycle_b"] {
        assert!(!o2.iter().any(|s| s == gone), "{gone} should be removed at -O2: {o2:?}");
    }
    for kept in ["via_data", "via_value", "get", "api"] {
        assert!(o2.iter().any(|s| s == kept), "{kept} must stay: {o2:?}");
    }
}

#[test]
fn single_call_static_helper_is_inlined_at_o2() {
    // Too big for the "small callee" rule: only the single-call-site rule
    // inlines it, so its definition disappears. Called twice, it stays.
    let helper = "static int helper(int *a, int n) {\n\
                    int s = 0;\n\
                    for (int i = 0; i < n; i++) { s += a[i] * (i + 1); if (s > 1000) s -= a[i] ^ i; }\n\
                    for (int i = n - 1; i >= 0; i--) { s ^= a[i] << (i & 7); s += s / 3; }\n\
                    for (int i = 0; i < n; i += 2) { s -= a[i] * a[i]; s = s % 100003; }\n\
                    return s;\n\
                  }\n";
    let once = format!("{helper}int api(int *a, int n) {{ return helper(a, n) + 1; }}\n");
    let syms = defined_symbols(&once, OptLevel::O2);
    assert!(!syms.iter().any(|s| s == "helper"), "single-call helper should be inlined: {syms:?}");
    let twice = format!("{helper}int api(int *a, int n) {{ return helper(a, n) + helper(a, n - 1); }}\n");
    let syms = defined_symbols(&twice, OptLevel::O2);
    assert!(syms.iter().any(|s| s == "helper"), "a helper called twice is not duplicated: {syms:?}");
}

/// The header both translation units include: a C99 plain `inline` definition.
const INLINE_H: &str = "inline int sq(int x) { return x * x; }\ninline int cube(int x) { return x * sq(x); }\n";

#[test]
fn c99_inline_definitions_emit_no_external_symbol() {
    // An inline definition alone provides no external definition: its private
    // copy serves this unit's calls, and no global `sq` is defined.
    let only_inline = format!("{INLINE_H}int use(int v) {{ return sq(v); }}\n");
    assert_eq!(symbol_binding(&only_inline, "sq"), None);
    assert_eq!(symbol_binding(&only_inline, "sq.inline"), Some((SymbolBinding::Local, false)));
    // Unused inline definitions are not compiled at all.
    assert_eq!(symbol_binding(&only_inline, "cube.inline"), None);
    // An `extern` declaration (or a declaration without `inline`) anywhere in
    // the unit makes the definition an external one.
    for decl in ["extern int sq(int);", "int sq(int);", "extern inline int sq(int);"] {
        let src = format!("{INLINE_H}{decl}\nint use(int v) {{ return sq(v); }}\n");
        assert_eq!(symbol_binding(&src, "sq"), Some((SymbolBinding::Global, false)), "{decl}");
    }
    // `extern inline` on the definition itself, and `static inline`.
    let ext = "extern inline int sq(int x) { return x * x; }\n";
    assert_eq!(symbol_binding(ext, "sq"), Some((SymbolBinding::Global, false)));
    let st = "static inline int sq(int x) { return x * x; } int use(int v) { return sq(v); }\n";
    assert_eq!(symbol_binding(st, "sq"), Some((SymbolBinding::Local, false)));
    // GNU89 inline semantics: a plain `inline` definition is external.
    let gnu89 = PpOptions { std: lf_cc::CStd::parse("gnu89").unwrap(), ..PpOptions::default() };
    let unit = lf_cc::compile_module_with(&only_inline, "t.c", &gnu89, OptLevel::O0, false).unwrap();
    let id = unit.module.symbol_id("sq").expect("gnu89 defines sq");
    assert_eq!(unit.module.symbol(id).binding, SymbolBinding::Global);
}

#[test]
fn c99_inline_across_two_translation_units() {
    if !host_crt() {
        return;
    }
    let s = Scratch::new("inline2");
    s.write("sq.h", INLINE_H);
    // Both units include the inline definitions; only `ext.c` declares them
    // `extern`, so only it provides the external definitions. Taking `sq`'s
    // address in `main.c` refers to that external definition.
    s.write(
        "main.c",
        "#include <stdio.h>\n#include \"sq.h\"\n\
         int other(void);\nint (*sq_from_ext(void))(int);\n\
         int main(void) {\n\
           int (*p)(int) = sq;\n\
           printf(\"%d %d %d %d\\n\", sq(5), cube(2), other(), p == sq_from_ext());\n\
           return 0;\n\
         }\n",
    );
    s.write(
        "ext.c",
        "#include \"sq.h\"\nextern int sq(int);\nextern int cube(int);\n\
         int other(void) { return sq(4) + cube(3); }\n\
         int (*sq_from_ext(void))(int) { return sq; }\n",
    );
    for opt in ["-O0", "-O2"] {
        let exe = format!("prog{opt}");
        s.lf_cc(&["-std=c11", opt, "main.c", "ext.c", "-o", &exe]);
        assert_eq!(s.run_stdout(&exe, &[]), "25 8 43 1\n", "{opt}");
    }
    if let Some(gcc) = which("gcc") {
        let std = common::gcc_std_flag(&gcc, "c11");
        s.gcc(&gcc, &[&std, "main.c", "ext.c", "-o", "prog_gcc"]);
        assert_eq!(s.run_stdout("prog_gcc", &[]), "25 8 43 1\n");
        // Mixed: gcc's external definitions, lf-cc's inline-only unit.
        s.lf_cc(&["-std=c11", "-c", "main.c"]);
        s.gcc(&gcc, &[&std, "-c", "ext.c"]);
        s.lf_cc(&["main.o", "ext.o", "-o", "prog_mixed"]);
        assert_eq!(s.run_stdout("prog_mixed", &[]), "25 8 43 1\n");
    }
}
