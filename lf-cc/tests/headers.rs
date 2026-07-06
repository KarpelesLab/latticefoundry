//! Differential tests for the builtin freestanding standard headers.
//!
//! Each program `#include`s one (or more) of the builtin headers lf-cc provides
//! (`<stddef.h>`, `<stdint.h>`, `<stdbool.h>`, `<limits.h>`, `<stdalign.h>`,
//! `<iso646.h>`, `<stdnoreturn.h>`, `<float.h>`), is compiled by lf-cc at -O0 and
//! -O2, run, and its exit code compared against the *same* program compiled by
//! `gcc -std=<matching>` (which uses gcc's own system headers). Agreement of the
//! observed exit codes confirms our LP64 typedefs/macros carry the same values.
//! If `gcc` is not installed the gcc comparison is skipped but the lf-cc
//! -O0-vs-O2 self-consistency check still runs. Every `main` returns `0..256`.
//!
//! Note: the programs deliberately avoid two constructs that hit *unrelated*
//! pre-existing lf-cc codegen limitations (negative-`long` signed comparison and
//! casting a negative constant to a wider unsigned type, e.g. `(size_t)-1`); the
//! header *values* are still exercised through equivalent working idioms (e.g.
//! `(size_t)0 - 1` for the maximum `size_t`).

use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::link::write_executable;
use latticefoundry::transform::pipeline::OptLevel;
use lf_cc::headers::builtin_header;
use lf_cc::{CStd, PpOptions};

/// The corpus: `(name, std, source)`. Each `main` returns a value in `0..256`.
fn programs() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        // <stdint.h>: exact-width types in arithmetic; INT32_MAX, UINT64_MAX&0xFF,
        // SIZE_MAX (== max size_t), INT64_C(1)<<40 truncated to int, intptr_t.
        (
            "stdint",
            "gnu17",
            "#include <stddef.h>\n#include <stdint.h>\n\
             int main(void){ int32_t a=INT32_MAX; uint64_t b=UINT64_MAX; intptr_t p=10; \
             int r=0; \
             if(a==2147483647) r+=1; \
             if((b & 0xFF)==255) r+=2; \
             if(SIZE_MAX==(size_t)0-1) r+=4; \
             int64_t big=INT64_C(1)<<40; if((int)big==0) r+=8; \
             if((intptr_t)(p+5)==15) r+=16; \
             return r; }",
        ),
        // <stdint.h>: the _C-suffix macros form the right typed constants and the
        // exact-width limit macros hold their LP64 values.
        (
            "stdint_limits",
            "gnu17",
            "#include <stdint.h>\n\
             int main(void){ int r=0; \
             if(INT8_MAX==127 && INT8_MIN==-128) r+=1; \
             if(UINT32_C(1)==1u && INT16_MAX==32767) r+=2; \
             if(INTMAX_MAX==9223372036854775807L) r+=4; \
             uint64_t m=UINT64_MAX; if((m>>60)==15) r+=8; \
             return r; }",
        ),
        // <stddef.h>: NULL, sizeof(size_t)==8, offsetof used as an array index.
        (
            "stddef",
            "gnu17",
            "#include <stddef.h>\n\
             struct S{ int a; int m; }; int arr[8]={0,1,2,3,4,5,6,7}; \
             int main(void){ int r=0; \
             if(NULL==(void*)0) r+=1; \
             if(sizeof(size_t)==8) r+=2; \
             r+=arr[offsetof(struct S, m)]; \
             return r; }",
        ),
        // <limits.h>: INT_MAX, CHAR_BIT==8, LONG_MAX>>60.
        (
            "limits",
            "gnu17",
            "#include <limits.h>\n\
             int main(void){ int r=0; \
             if(CHAR_BIT==8) r+=1; \
             if(INT_MAX==2147483647) r+=2; \
             r+=(int)(LONG_MAX>>60); \
             if(CHAR_MIN==-128 && SCHAR_MAX==127) r+=16; \
             return r; }",
        ),
        // <stdbool.h> pre-C23 (macros): bool b = true; return b ? 7 : 0.
        (
            "stdbool_c11",
            "c11",
            "#include <stdbool.h>\n\
             int main(void){ bool b = true; return (b && !false) \
             ? __bool_true_false_are_defined + 6 : 0; }",
        ),
        // <stdbool.h> under C23 (bool/true/false are keywords; header is a no-op
        // except the feature probe): still usable.
        (
            "stdbool_c23",
            "c23",
            "#include <stdbool.h>\n\
             int main(void){ bool b = true; return (b && !false) \
             ? __bool_true_false_are_defined + 6 : 0; }",
        ),
        // <stdalign.h> pre-C23 (macros): alignof(double), alignas(16) via macro.
        (
            "stdalign_c11",
            "c11",
            "#include <stdalign.h>\n\
             int main(void){ alignas(16) char buf[8]; int a=(int)alignof(double); \
             return a + ((((unsigned long)&buf[0]) % 16 == 0) ? 40 : 0); }",
        ),
        // <stdalign.h> under C23 (alignas/alignof are keywords): still usable.
        (
            "stdalign_c23",
            "c23",
            "#include <stdalign.h>\n\
             int main(void){ alignas(16) char buf[8]; \
             return (((unsigned long)&buf[0]) % 16 == 0) \
             ? (int)alignof(double) + 34 : 0; }",
        ),
        // <iso646.h>: (1 and 0) or (not 0).
        (
            "iso646",
            "gnu17",
            "#include <iso646.h>\n\
             int main(void){ return ((1 and 0) or (not 0)) + (5 bitand 6) + 37; }",
        ),
        // <stdnoreturn.h> pre-C23: `noreturn` spelling accepted on a declaration.
        (
            "stdnoreturn_c11",
            "c11",
            "#include <stdnoreturn.h>\n\
             noreturn void fail(void); void fail(void){ for(;;){} } \
             int main(void){ return 42; }",
        ),
        // <float.h>: IEEE-754 binary32/64 characteristics.
        (
            "float",
            "gnu17",
            "#include <float.h>\n\
             int main(void){ int r=0; \
             if(FLT_RADIX==2) r+=1; \
             if(FLT_MANT_DIG==24 && DBL_MANT_DIG==53) r+=2; \
             if(DBL_DIG==15 && FLT_DIG==6) r+=4; \
             double e=DBL_EPSILON; if(e>0.0 && e<1.0) r+=8; \
             float fe=FLT_EPSILON; if(fe>0.0f && fe<1.0f) r+=16; \
             return r; }",
        ),
    ]
}

struct Harness {
    dir: PathBuf,
    gcc: Option<String>,
}

impl Harness {
    fn new() -> Harness {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("lf-cc-hdrtest-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let gcc = which("gcc").or_else(|| which("cc"));
        Harness { dir, gcc }
    }

    /// Compile with lf-cc under `std` at `opt`, run, and return the exit code.
    fn lf_run(&self, name: &str, src: &str, opt: OptLevel, std: &str) -> i32 {
        let input = format!("{name}.c");
        let pp = PpOptions {
            std: CStd::parse(std).expect("known --std"),
            main_file_name: input.clone(),
            ..PpOptions::default()
        };
        let image = lf_cc::build_image_with(src, &input, &pp, opt, false)
            .unwrap_or_else(|e| panic!("lf-cc failed to build '{name}' (--std={std}): {e:?}"));
        let bin = self.dir.join(format!("{name}.lf.{}", opt.name()));
        write_executable(bin.to_str().unwrap(), &image).expect("write executable");
        run_exit(&bin)
    }

    /// Compile with `gcc -std=<std> -O0` and run, if gcc is available.
    fn gcc_run(&self, name: &str, src: &str, std: &str) -> Option<i32> {
        let gcc = self.gcc.as_ref()?;
        let c = self.dir.join(format!("{name}.c"));
        std::fs::write(&c, src).expect("write source");
        let bin = self.dir.join(format!("{name}.gcc"));
        let status = Command::new(gcc)
            .args(["-O0", "-w", &format!("-std={std}"), "-o"])
            .arg(&bin)
            .arg(&c)
            .status()
            .expect("run gcc");
        assert!(status.success(), "gcc failed to compile '{name}' (-std={std})");
        Some(run_exit(&bin))
    }
}

fn which(prog: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(prog);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    None
}

fn run_exit(bin: &Path) -> i32 {
    let status = Command::new(bin).status().expect("run binary");
    if let Some(code) = status.code() {
        return code;
    }
    use std::os::unix::process::ExitStatusExt;
    panic!("{} killed by signal {:?}", bin.display(), status.signal());
}

#[test]
fn headers_differential_against_gcc() {
    let h = Harness::new();
    let mut matched_gcc = 0usize;
    let mut ran = 0usize;
    let total = programs().len();
    let mut failures: Vec<String> = Vec::new();

    for (name, std, src) in programs() {
        let o0 = h.lf_run(name, src, OptLevel::O0, std);
        let o2 = h.lf_run(name, src, OptLevel::O2, std);
        ran += 1;
        if o0 != o2 {
            failures.push(format!("{name} (--std={std}): lf-cc -O0={o0} != -O2={o2}"));
            continue;
        }
        if let Some(g) = h.gcc_run(name, src, std) {
            if o0 != g {
                failures.push(format!("{name} (--std={std}): lf-cc={o0} != gcc={g}"));
                continue;
            }
            matched_gcc += 1;
        }
    }

    eprintln!(
        "headers differential: {ran}/{total} programs ran; {matched_gcc} matched gcc's exit \
         code (gcc {})",
        if h.gcc.is_some() { "present" } else { "absent — comparison skipped" }
    );
    assert!(failures.is_empty(), "header differential mismatches:\n{}", failures.join("\n"));
    assert_eq!(ran, total, "every program should compile and run");
}

/// A `#include "local.h"` from the `-I` path still resolves (builtin headers are
/// only a *fallback* for names not found on disk).
#[test]
fn local_include_still_works() {
    let h = Harness::new();
    let hdr = h.dir.join("local.h");
    std::fs::write(&hdr, "#define LOCAL_ANSWER 42\n").expect("write local.h");

    let src = "#include \"local.h\"\nint main(void){ return LOCAL_ANSWER; }";
    let pp = PpOptions {
        main_file_name: "local_test.c".to_owned(),
        include_dirs: vec![h.dir.clone()],
        ..PpOptions::default()
    };
    let image = lf_cc::build_image_with(src, "local_test.c", &pp, OptLevel::O0, false)
        .expect("build with a local include");
    let bin = h.dir.join("local_test");
    write_executable(bin.to_str().unwrap(), &image).expect("write executable");
    assert_eq!(run_exit(&bin), 42);
}

/// A genuinely-missing `<nope.h>` is still an error (no phantom builtin).
#[test]
fn missing_angle_include_errors() {
    let src = "#include <nope.h>\nint main(void){ return 0; }";
    let pp = PpOptions { main_file_name: "missing.c".to_owned(), ..PpOptions::default() };
    let err = lf_cc::check_source_with(src, &pp).expect_err("a missing header must error");
    assert!(
        err.iter().any(|d| d.is_error() && d.message.contains("nope.h")),
        "expected a 'cannot find include file' diagnostic, got: {err:?}"
    );
}

/// The builtin-header table exposes the freestanding set and nothing else.
#[test]
fn builtin_header_table() {
    for name in [
        "stddef.h",
        "stdint.h",
        "stdbool.h",
        "limits.h",
        "stdalign.h",
        "iso646.h",
        "stdnoreturn.h",
        "float.h",
        "stdarg.h",
    ] {
        assert!(builtin_header(name).is_some(), "{name} should be a builtin header");
    }
    assert!(builtin_header("stdio.h").is_none(), "hosted headers are not provided");
    assert!(builtin_header("nope.h").is_none());
}

/// `-nostdinc` (i.e. `builtin_headers = false`) makes `<stdint.h>` unresolved,
/// while the default (enabled) resolves it.
#[test]
fn nostdinc_disables_builtin_headers() {
    let src = "#include <stdint.h>\nint main(void){ return sizeof(int32_t); }";

    let enabled = PpOptions { main_file_name: "n.c".to_owned(), ..PpOptions::default() };
    assert!(enabled.builtin_headers, "builtin headers are on by default");
    lf_cc::check_source_with(src, &enabled).expect("<stdint.h> resolves via the builtin table");

    let disabled = PpOptions {
        main_file_name: "n.c".to_owned(),
        builtin_headers: false,
        ..PpOptions::default()
    };
    let err = lf_cc::check_source_with(src, &disabled)
        .expect_err("with -nostdinc, <stdint.h> is unresolved");
    assert!(
        err.iter().any(|d| d.is_error() && d.message.contains("stdint.h")),
        "expected an unresolved-include error, got: {err:?}"
    );
}
