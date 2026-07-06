//! Differential ABI tests: struct-by-value across calls and variadic functions.
//!
//! Each program is compiled with `lf-cc` at -O0 and -O2 and, when `gcc` is
//! available, with `gcc -O0`; the process exit codes (kept in `0..256`) are
//! compared. lf-cc uses its own internally-consistent by-reference/`sret` struct
//! ABI, so a whole self-contained program still observes the same result gcc's
//! native ABI does. Variadic functions use the backend's System V register save
//! area / overflow area (`va_start`/`va_arg`/`va_end`).

use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::link::write_executable;
use latticefoundry::transform::pipeline::OptLevel;

/// The ABI corpus: `(name, source)`, each `main` returning a value in `0..256`.
fn programs() -> Vec<(&'static str, &'static str)> {
    vec![
        // --- struct-by-value -------------------------------------------------
        // Two-int struct (one INTEGER eightbyte): pass two by value, return one,
        // sum its fields. 13 + 24 = 37.
        (
            "struct_addp",
            "struct P{int x,y;}; \
             struct P addp(struct P a, struct P b){ struct P r; r.x=a.x+b.x; r.y=a.y+b.y; return r; } \
             int main(){ struct P a={3,4}, b={10,20}; struct P r=addp(a,b); return r.x+r.y; }",
        ),
        // A struct passed by value is NOT mutated in the caller (copy semantics).
        (
            "struct_copy_semantics",
            "struct P{int x,y;}; \
             void clobber(struct P p){ p.x=999; p.y=999; } \
             int main(){ struct P a={40,2}; clobber(a); return a.x+a.y; }",
        ),
        // An all-double struct (two SSE eightbytes): field-wise add, sum -> int.
        (
            "struct_double",
            "struct V{double x,y;}; \
             struct V addv(struct V a, struct V b){ struct V r; r.x=a.x+b.x; r.y=a.y+b.y; return r; } \
             int main(){ struct V a={1.5,2.25}, b={0.5,0.75}; struct V r=addv(a,b); \
             return (int)(r.x*10 + r.y*10); }",
        ),
        // A mixed int+double struct (INTEGER + SSE eightbytes).
        (
            "struct_mixed",
            "struct M{int i; double d;}; \
             struct M addm(struct M a, struct M b){ struct M r; r.i=a.i+b.i; r.d=a.d+b.d; return r; } \
             int main(){ struct M a={3,1.5}, b={4,2.25}; struct M r=addm(a,b); \
             return r.i + (int)(r.d*4); }",
        ),
        // A >16-byte struct (MEMORY class): passed on the stack, returned via sret.
        (
            "struct_big",
            "struct Big{long a,b,c;}; \
             struct Big addbig(struct Big a, struct Big b){ \
             struct Big r; r.a=a.a+b.a; r.b=a.b+b.b; r.c=a.c+b.c; return r; } \
             int main(){ struct Big a={1,2,3}, b={10,20,30}; struct Big r=addbig(a,b); \
             return (int)(r.a+r.b+r.c); }",
        ),
        // Returning a struct and using it directly (member of a call result via a
        // temporary), plus struct assignment from a call result.
        (
            "struct_return_use",
            "struct P{int x,y;}; \
             struct P mk(int a, int b){ struct P p; p.x=a; p.y=b; return p; } \
             int main(){ struct P q; q = mk(15, 25); struct P r = mk(1, 1); \
             return q.x + q.y + r.x + r.y - 5; }",
        ),
        // A struct argument that is itself a struct-returning call result.
        (
            "struct_call_arg",
            "struct P{int x,y;}; \
             struct P mk(int a,int b){ struct P p; p.x=a; p.y=b; return p; } \
             int sum(struct P p){ return p.x+p.y; } \
             int main(){ return sum(mk(30, 12)); }",
        ),
        // --- variadic --------------------------------------------------------
        // sum n int varargs, all in registers. Exercises the `__builtin_*` names
        // directly (with `va_list` from the header).
        (
            "va_int_sum",
            "#include <stdarg.h>\n\
             int sumn(int n, ...){ va_list ap; __builtin_va_start(ap, n); \
             int s=0; for(int i=0;i<n;i++) s += __builtin_va_arg(ap, int); \
             __builtin_va_end(ap); return s; } \
             int main(){ return sumn(4, 10, 20, 30, 40); }",
        ),
        // More than 6 int varargs: the tail spills to the overflow area.
        (
            "va_int_overflow",
            "#include <stdarg.h>\n\
             int sumn(int n, ...){ va_list ap; va_start(ap, n); \
             int s=0; for(int i=0;i<n;i++) s += va_arg(ap, int); \
             va_end(ap); return s; } \
             int main(){ return sumn(9, 1,2,3,4,5,6,7,8,9); }",
        ),
        // double varargs summed (SSE register save area + overflow).
        (
            "va_double_sum",
            "#include <stdarg.h>\n\
             double dsum(int n, ...){ va_list ap; va_start(ap, n); \
             double s=0; for(int i=0;i<n;i++) s += va_arg(ap, double); \
             va_end(ap); return s; } \
             int main(){ return (int)dsum(5, 1.0,2.0,4.0,8.0,16.0); }",
        ),
        // Real <stdarg.h> with the va_* macros.
        (
            "va_stdarg_header",
            "#include <stdarg.h>\n\
             int sumn(int n, ...){ va_list ap; va_start(ap, n); \
             int s=0; for(int i=0;i<n;i++) s += va_arg(ap, int); \
             va_end(ap); return s; } \
             int main(){ return sumn(6, 5,10,15,20,25,30) - 63; }",
        ),
        // Mixed named + variadic, with a pointer named parameter before `...`.
        (
            "va_mixed_named",
            "#include <stdarg.h>\n\
             int addbase(int base, int n, ...){ va_list ap; va_start(ap, n); \
             int s=base; for(int i=0;i<n;i++) s += va_arg(ap, int); \
             va_end(ap); return s; } \
             int main(){ return addbase(6, 3, 10, 11, 12); }",
        ),
    ]
}

struct Harness {
    dir: PathBuf,
    gcc: Option<String>,
}

impl Harness {
    fn new() -> Harness {
        let dir = std::env::temp_dir().join(format!("lf-cc-abitest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let gcc = which("gcc").or_else(|| which("cc"));
        Harness { dir, gcc }
    }

    fn lf_run(&self, name: &str, src: &str, opt: OptLevel) -> i32 {
        let image = lf_cc::build_image(src, &format!("{name}.c"), opt, false)
            .unwrap_or_else(|e| panic!("lf-cc failed to build '{name}': {e:?}"));
        let bin = self.dir.join(format!("{name}.lf.{}", opt.name()));
        write_executable(bin.to_str().unwrap(), &image).expect("write executable");
        run_exit(&bin)
    }

    fn gcc_run(&self, name: &str, src: &str) -> Option<i32> {
        let gcc = self.gcc.as_ref()?;
        let c = self.dir.join(format!("{name}.c"));
        std::fs::write(&c, src).expect("write source");
        let bin = self.dir.join(format!("{name}.gcc"));
        let status = Command::new(gcc)
            .args(["-O0", "-w", "-o"])
            .arg(&bin)
            .arg(&c)
            .status()
            .expect("run gcc");
        assert!(status.success(), "gcc failed to compile '{name}'");
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
fn abi_differential() {
    let h = Harness::new();
    let mut matched_gcc = 0usize;
    let mut ran = 0usize;
    let total = programs().len();
    let mut failures: Vec<String> = Vec::new();

    for (name, src) in programs() {
        let o0 = h.lf_run(name, src, OptLevel::O0);
        let o2 = h.lf_run(name, src, OptLevel::O2);
        ran += 1;
        if o0 != o2 {
            failures.push(format!("{name}: lf-cc -O0={o0} != -O2={o2}"));
            continue;
        }
        if let Some(g) = h.gcc_run(name, src) {
            if o0 != g {
                failures.push(format!("{name}: lf-cc={o0} != gcc={g}"));
                continue;
            }
            matched_gcc += 1;
        }
    }

    eprintln!(
        "abi: {ran}/{total} programs ran; {matched_gcc} matched gcc's exit code (gcc {})",
        if h.gcc.is_some() { "present" } else { "absent — comparison skipped" }
    );
    assert!(failures.is_empty(), "abi mismatches:\n{}", failures.join("\n"));
    assert_eq!(ran, total, "every program should compile and run");
}
