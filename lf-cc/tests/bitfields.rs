//! Differential tests for struct/union bit-fields: compile a suite of C programs
//! with `lf-cc` (at -O0 and -O2), run each native binary, and assert its process
//! exit code matches the same program compiled with `gcc -O0`.
//!
//! Every program keeps its result in `0..256` so the process exit code is an
//! unambiguous observation. If `gcc` is not installed the gcc comparison is
//! skipped, but the -O0-vs-O2 self-consistency check still runs. A final unit
//! test covers the two bit-field constraint diagnostics (`&` of a bit-field and
//! an over-wide width), which do not need gcc.

use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::link::write_executable;
use latticefoundry::transform::pipeline::OptLevel;

/// The bit-field corpus: `(name, source)`. Each `main` returns a value in
/// `0..256`. Several programs embed `sizeof(struct ...)` in the returned value,
/// so the gcc comparison also checks that lf-cc's bit-field layout (size and
/// alignment) matches gcc's.
fn programs() -> Vec<(&'static str, &'static str)> {
    vec![
        // Packing three unsigned fields into one 32-bit unit: round-trip the set
        // values, exercise the full range, and check that an out-of-range value
        // wraps modulo 2^width.
        (
            "pack_unsigned",
            "struct A { unsigned a:3, b:5, c:1; }; \
             int main(){ struct A s; int r=0; \
             s.a=5; s.b=20; s.c=1; \
             if (s.a==5 && s.b==20 && s.c==1) r+=1; \
             s.a=7; s.b=31; s.c=1; \
             if (s.a==7 && s.b==31 && s.c==1) r+=2; \
             s.a=8; s.b=33; \
             if (s.a==0 && s.b==1) r+=4; \
             return r + 100; }",
        ),
        // Write a whole 32-bit pattern through a union alias, then read the three
        // packed fields back individually (verifies little-endian bit order).
        (
            "pack_pattern",
            "struct A { unsigned a:3, b:5, c:1; }; \
             union U { struct A s; unsigned u; }; \
             int main(){ union U v; v.u = 0; v.s.a=7; v.s.b=5; v.s.c=1; \
             /* a=bits0-2=7, b=bits3-7=5, c=bit8=1 => 0b1 00101 111 = 0x12F */ \
             int lo = (v.u == 0x12F); \
             v.u = 0x1FF; \
             int back = (v.s.a==7) + (v.s.b==31)*2 + (v.s.c==1)*4; \
             return lo + back*8 + 90; }",
        ),
        // Signed bit-fields sign-extend on read; assigning an out-of-range value
        // wraps into the signed range (7 -> 7, 8 -> -8, -1 -> -1).
        (
            "signed_extend",
            "struct S { int x:4; }; \
             int main(){ struct S s; \
             s.x=-1; int a=(s.x==-1); \
             s.x=7;  int b=(s.x==7); \
             s.x=8;  int c=(s.x==-8); \
             s.x=-9; int d=(s.x==7); \
             return a + b*2 + c*4 + d*8 + 100; }",
        ),
        // An unnamed `:0` field closes the current unit: the field after it lands
        // in a fresh storage unit. Verified via round-trip and sizeof (== gcc).
        (
            "zero_width",
            "struct Z { unsigned a:5; unsigned :0; unsigned b:5; }; \
             int main(){ struct Z z; z.a=31; z.b=17; \
             int ok = (z.a==31) + (z.b==17)*2; \
             return ok + (int)sizeof(struct Z) + 100; }",
        ),
        // Bit-fields of different declared types (int, char, long) interleaved with
        // an ordinary member; round-trip each and fold sizeof(struct) into the
        // result so the gcc comparison checks the layout size.
        (
            "mixed_types",
            "struct M { int a:5; char c; long b:33; int d; }; \
             int main(){ struct M m; \
             m.a=-3; m.c=77; m.b=4000000000L; m.d=42; \
             int ok = (m.a==-3) + (m.c==77)*2 + (m.b==4000000000L)*4 + (m.d==42)*8; \
             return ok + (int)sizeof(struct M) + 60; }",
        ),
        // A `_Bool` bit-field is 0/1 (any non-zero assignment becomes 1), packed
        // ahead of an `int` bit-field in the same unit.
        (
            "bool_field",
            "struct B { _Bool f:1; int x:3; }; \
             int main(){ struct B b; \
             b.f=1; b.x=2; int r=(b.f==1)+(b.x==2)*2; \
             b.f=3; r+=(b.f==1)*4; \
             b.f=0; r+=(b.f==0)*8; \
             return r + 100; }",
        ),
        // Compound assignment, `++`, and `--` on bit-fields go through a masked
        // read-modify-write and wrap in the field's width.
        (
            "rmw_ops",
            "struct C { int a:4; unsigned b:5; }; \
             int main(){ struct C c; c.a=3; c.b=10; \
             c.a += 5;   /* 8 -> -8 in 4 signed bits */ \
             c.b <<= 1;  /* 20 */ \
             c.a++;      /* -7 */ \
             c.b--;      /* 19 */ \
             int r = (c.a==-7) + (c.b==19)*2; \
             return r + 100; }",
        ),
        // Bit-fields inside a union: each starts at bit 0 of offset 0, so writing
        // the wider field and reading the narrower one observes the low bits.
        (
            "union_bits",
            "union U { unsigned a:4; unsigned b:8; }; \
             int main(){ union U u; u.b=0xAB; \
             int hi = u.a; /* low 4 bits of 0xAB == 0xB */ \
             return hi + (int)sizeof(union U) + 100; }",
        ),
        // A straddling field: a 20-bit field followed by another 20-bit field of
        // the same type starts a new 32-bit unit (they cannot share one int).
        (
            "straddle",
            "struct T { unsigned a:20; unsigned b:20; }; \
             int main(){ struct T t; t.a=0xFFFFF; t.b=0x12345; \
             int ok = (t.a==0xFFFFF) + (t.b==0x12345)*2; \
             return ok + (int)sizeof(struct T) + 100; }",
        ),
        // A brace initializer for a struct with bit-fields (a `:0` in the middle is
        // skipped), plus a global bit-field object with a constant initializer.
        (
            "init_bits",
            "struct A { unsigned a:3, b:5; }; \
             struct G { int x:4; unsigned :0; int y:4; }; \
             struct G g = { -2, 5 }; \
             int main(){ struct A s = { 6, 21 }; \
             int ok = (s.a==6) + (s.b==21)*2 + (g.x==-2)*4 + (g.y==5)*8; \
             return ok + 100; }",
        ),
    ]
}

struct Harness {
    dir: PathBuf,
    gcc: Option<String>,
}

impl Harness {
    fn new() -> Harness {
        let dir = std::env::temp_dir().join(format!("lf-cc-bftest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let gcc = which("gcc").or_else(|| which("cc"));
        Harness { dir, gcc }
    }

    /// Compile with lf-cc at `opt`, run, and return the exit code.
    fn lf_run(&self, name: &str, src: &str, opt: OptLevel) -> i32 {
        let image = lf_cc::build_image(src, &format!("{name}.c"), opt, false)
            .unwrap_or_else(|e| panic!("lf-cc failed to build '{name}': {e:?}"));
        let bin = self.dir.join(format!("{name}.lf.{}", opt.name()));
        write_executable(bin.to_str().unwrap(), &image).expect("write executable");
        run_exit(&bin)
    }

    /// Compile with gcc -O0 and run, if gcc is available.
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
fn bitfields_against_gcc() {
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
        "bitfields: {ran}/{total} programs ran; {matched_gcc} matched gcc's exit code (gcc {})",
        if h.gcc.is_some() { "present" } else { "absent — comparison skipped" }
    );
    assert!(failures.is_empty(), "bit-field mismatches:\n{}", failures.join("\n"));
    assert_eq!(ran, total, "every program should compile and run");
}

/// The two bit-field constraint diagnostics that do not need gcc: taking the
/// address of a bit-field, and declaring a bit-field wider than its type.
#[test]
fn bitfield_constraint_diagnostics() {
    // `&s.bf` — a bit-field is not addressable.
    let addr = "struct S { int a:4; }; int main(){ struct S s; int *p = &s.a; return *p; }";
    let err = lf_cc::check_source(addr).expect_err("&bit-field must be rejected");
    assert!(
        err.iter().any(|d| d.message.contains("address of a bit-field")),
        "expected an address-of-bit-field diagnostic, got: {err:?}"
    );

    // A width wider than the declared type.
    let wide = "struct S { int a:40; }; int main(){ return 0; }";
    let err = lf_cc::check_source(wide).expect_err("over-wide bit-field must be rejected");
    assert!(
        err.iter().any(|d| d.message.contains("exceeds the width")),
        "expected an over-wide bit-field diagnostic, got: {err:?}"
    );

    // A named zero-width bit-field is a constraint violation.
    let zero = "struct S { int a:0; }; int main(){ return 0; }";
    let err = lf_cc::check_source(zero).expect_err("named :0 must be rejected");
    assert!(
        err.iter().any(|d| d.message.contains("zero width")),
        "expected a zero-width-named-bit-field diagnostic, got: {err:?}"
    );
}
