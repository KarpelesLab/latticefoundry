//! Tests for the C23 `#embed` preprocessor directive.
//!
//! Two layers:
//!  * a differential harness that compiles a `#embed` program with `lf-cc`
//!    (`--std=c23`, at -O0 and -O2) and, when the local `gcc` supports `#embed`,
//!    with `gcc -std=c23`, then compares the process exit codes. If gcc lacks
//!    `#embed` (or is absent), the lf-cc runs are still checked against a known
//!    expected value and against each other.
//!  * unit tests on the raw expansion (bytes → integer token list), the pre-C23
//!    rejection, and the missing-resource error, driven through
//!    [`lf_cc::preprocess::preprocess`] directly.
//!
//! Every program keeps its result in `0..256` so the exit code is an unambiguous
//! observation. Resource files are written into a per-test temp directory; the
//! `.c` file lives beside them so a `"…"` `#embed` resolves locally, exactly like
//! an `#include`.

use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::link::write_executable;
use latticefoundry::transform::pipeline::OptLevel;
use lf_cc::lex::TokenKind;
use lf_cc::{CStd, PpOptions};

/// A per-test scratch directory plus the resolved (embed-capable) gcc, if any.
struct Harness {
    dir: PathBuf,
    /// A `gcc`/`cc` that accepts `#embed` under `-std=c23`, if one is installed.
    gcc: Option<String>,
}

impl Harness {
    fn new() -> Harness {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("lf-cc-embedtest-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let gcc = which("gcc").or_else(|| which("cc")).filter(|g| gcc_has_embed(g, &dir));
        Harness { dir, gcc }
    }

    /// Write a resource file into the scratch directory.
    fn write_resource(&self, name: &str, bytes: &[u8]) {
        std::fs::write(self.dir.join(name), bytes).expect("write resource");
    }

    /// Compile `src` with lf-cc under `--std=c23` at `opt`, run, return the exit
    /// code. `name.c` is placed in the scratch dir so `"…"` embeds resolve there.
    fn lf_run(&self, name: &str, src: &str, opt: OptLevel) -> i32 {
        let input = self.dir.join(format!("{name}.c"));
        let pp = PpOptions {
            std: CStd::parse("c23").expect("c23"),
            include_dirs: vec![self.dir.clone()],
            main_file_name: input.to_string_lossy().into_owned(),
            ..PpOptions::default()
        };
        let image = lf_cc::build_image_with(src, &input.to_string_lossy(), &pp, opt, false)
            .unwrap_or_else(|e| panic!("lf-cc failed to build '{name}': {e:?}"));
        let bin = self.dir.join(format!("{name}.lf.{}", opt.name()));
        write_executable(bin.to_str().unwrap(), &image).expect("write executable");
        run_exit(&bin)
    }

    /// Compile with `gcc -std=c23 -O0` and run, if an embed-capable gcc exists.
    fn gcc_run(&self, name: &str, src: &str) -> Option<i32> {
        let gcc = self.gcc.as_ref()?;
        let c = self.dir.join(format!("{name}.gcc.c"));
        std::fs::write(&c, src).expect("write source");
        let bin = self.dir.join(format!("{name}.gcc"));
        // `<…>` embeds search gcc's dedicated `--embed-dir` path (not `-I`).
        let status = Command::new(gcc)
            .args(["-std=c23", "-O0", "-w"])
            .arg(format!("--embed-dir={}", self.dir.display()))
            .arg("-o")
            .arg(&bin)
            .arg(&c)
            .status()
            .expect("run gcc");
        assert!(status.success(), "gcc failed to compile '{name}'");
        Some(run_exit(&bin))
    }
}

/// One differential case: a name, the C source, the resources it needs, and the
/// known-correct exit code (used when gcc cannot corroborate).
struct Case {
    name: &'static str,
    src: &'static str,
    resources: &'static [(&'static str, &'static [u8])],
    expected: i32,
}

fn cases() -> Vec<Case> {
    vec![
        // A 3-byte resource used as an array initializer; check element values.
        Case {
            name: "basic_values",
            src: "int main(void){ unsigned char d[] = {\n#embed \"res.bin\"\n};\n\
                  return d[0] + d[2] - 100; }",
            resources: &[("res.bin", b"hi!")],
            expected: 104 + 33 - 100, // 'h'=104, '!'=33 → 37
        },
        // sizeof of the embedded array equals the byte count.
        Case {
            name: "sizeof_count",
            src: "int main(void){ unsigned char d[] = {\n#embed \"five.bin\"\n};\n\
                  return (int)sizeof(d); }",
            resources: &[("five.bin", b"ABCDE")],
            expected: 5,
        },
        // limit(3) truncates a 10-byte resource to 3 bytes.
        Case {
            name: "limit_three",
            src: "int main(void){ unsigned char d[] = {\n#embed \"ten.bin\" limit(3)\n};\n\
                  return (int)sizeof(d); }",
            resources: &[("ten.bin", b"0123456789")],
            expected: 3,
        },
        // An empty resource with if_empty(7) yields the single value 7.
        Case {
            name: "if_empty_value",
            src: "int main(void){ int x =\n#embed \"empty.bin\" if_empty(7)\n; return x; }",
            resources: &[("empty.bin", b"")],
            expected: 7,
        },
        // limit(0) is treated as empty, so if_empty fires.
        Case {
            name: "limit_zero_if_empty",
            src: "int main(void){ int x =\n#embed \"ten.bin\" limit(0) if_empty(9)\n; return x; }",
            resources: &[("ten.bin", b"0123456789")],
            expected: 9,
        },
        // prefix/suffix wrap a non-empty embed: {5, 10, 20, 7} → sizeof 4.
        Case {
            name: "prefix_suffix",
            src: "int main(void){ unsigned char d[] = {\n#embed \"two.bin\" prefix(5,) suffix(,7)\n};\n\
                  return (int)sizeof(d) * 10 + d[0] + d[3]; }",
            resources: &[("two.bin", &[10, 20])],
            expected: 4 * 10 + 5 + 7, // 52
        },
        // prefix/suffix are suppressed on an empty resource; only if_empty fires.
        Case {
            name: "empty_suppresses_prefix",
            src: "int main(void){ int x =\n\
                  #embed \"empty.bin\" prefix(1) suffix(2) if_empty(5)\n; return x; }",
            resources: &[("empty.bin", b"")],
            expected: 5,
        },
        // The classic compound-literal first-byte idiom.
        Case {
            name: "compound_first",
            src: "int main(void){ return (int[]){\n#embed \"res.bin\"\n}[0]; }",
            resources: &[("res.bin", b"hi!")],
            expected: 104,
        },
        // The <…> spelling resolves via the include path (the scratch dir).
        Case {
            name: "angle_form",
            src: "int main(void){ unsigned char d[] = {\n#embed <res.bin>\n};\n return d[1]; }",
            resources: &[("res.bin", b"hi!")],
            expected: 105, // 'i'
        },
        // A macro supplies the limit value; the argument line is macro-expanded.
        Case {
            name: "macro_limit",
            src: "#define N 4\nint main(void){ unsigned char d[] = {\n#embed \"ten.bin\" limit(N)\n};\n\
                  return (int)sizeof(d); }",
            resources: &[("ten.bin", b"0123456789")],
            expected: 4,
        },
    ]
}

#[test]
fn embed_differential() {
    let h = Harness::new();
    let mut failures: Vec<String> = Vec::new();
    let mut matched_gcc = 0usize;
    let total = cases().len();

    for case in cases() {
        for (fname, bytes) in case.resources {
            h.write_resource(fname, bytes);
        }
        let o0 = h.lf_run(case.name, case.src, OptLevel::O0);
        let o2 = h.lf_run(case.name, case.src, OptLevel::O2);
        if o0 != o2 {
            failures.push(format!("{}: lf-cc -O0={o0} != -O2={o2}", case.name));
            continue;
        }
        if o0 != case.expected {
            failures.push(format!("{}: lf-cc={o0} != expected={}", case.name, case.expected));
            continue;
        }
        if let Some(g) = h.gcc_run(case.name, case.src) {
            if g != o0 {
                failures.push(format!("{}: lf-cc={o0} != gcc={g}", case.name));
                continue;
            }
            matched_gcc += 1;
        }
    }

    eprintln!(
        "embed_differential: {total} programs ran; {matched_gcc} matched gcc's exit code \
         (embed-capable gcc {})",
        if h.gcc.is_some() { "present" } else { "absent — comparison skipped" }
    );
    assert!(failures.is_empty(), "embed mismatches:\n{}", failures.join("\n"));
}

// --- unit tests on the raw expansion & diagnostics --------------------------

/// Preprocess `src` under `--std=<std>`, resolving `"…"` embeds relative to
/// `dir`, and return the token stream (or diagnostics).
fn preprocess_in(
    dir: &Path,
    std: &str,
    src: &str,
) -> Result<Vec<lf_cc::lex::Token>, Vec<latticefoundry::support::diagnostics::Diagnostic>> {
    let input = dir.join("unit.c");
    let pp = PpOptions {
        std: CStd::parse(std).expect("known std"),
        include_dirs: vec![dir.to_path_buf()],
        main_file_name: input.to_string_lossy().into_owned(),
        ..PpOptions::default()
    };
    lf_cc::preprocess::preprocess(src, &pp)
}

/// The integer values of the `IntLit` tokens in a stream, in order.
fn int_values(toks: &[lf_cc::lex::Token]) -> Vec<i128> {
    toks.iter()
        .filter_map(|t| match &t.kind {
            TokenKind::IntLit(v, _) => Some(*v),
            _ => None,
        })
        .collect()
}

fn scratch() -> PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("lf-cc-embedunit-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn expansion_bytes_to_integer_tokens() {
    let dir = scratch();
    std::fs::write(dir.join("res.bin"), b"hi!").expect("write");
    let toks = preprocess_in(&dir, "c23", "int a[] = {\n#embed \"res.bin\"\n};\n").expect("ok");
    // 'h'=104, 'i'=105, '!'=33.
    assert_eq!(int_values(&toks), vec![104, 105, 33]);
}

#[test]
fn expansion_full_byte_range() {
    let dir = scratch();
    let bytes: Vec<u8> = (0..=255u16).map(|b| b as u8).collect();
    std::fs::write(dir.join("all.bin"), &bytes).expect("write");
    let toks = preprocess_in(&dir, "c23", "int a[] = {\n#embed \"all.bin\"\n};\n").expect("ok");
    let expected: Vec<i128> = (0..=255i128).collect();
    assert_eq!(int_values(&toks), expected);
}

#[test]
fn expansion_limit_truncates() {
    let dir = scratch();
    std::fs::write(dir.join("ten.bin"), b"0123456789").expect("write");
    let toks =
        preprocess_in(&dir, "c23", "int a[] = {\n#embed \"ten.bin\" limit(3)\n};\n").expect("ok");
    // '0'=48, '1'=49, '2'=50.
    assert_eq!(int_values(&toks), vec![48, 49, 50]);
}

#[test]
fn expansion_empty_yields_nothing() {
    let dir = scratch();
    std::fs::write(dir.join("empty.bin"), b"").expect("write");
    let toks = preprocess_in(&dir, "c23", "int a[] = {\n#embed \"empty.bin\"\n};\n").expect("ok");
    assert_eq!(int_values(&toks), Vec::<i128>::new());
}

#[test]
fn expansion_prefix_suffix() {
    let dir = scratch();
    std::fs::write(dir.join("two.bin"), [10u8, 20]).expect("write");
    let toks = preprocess_in(
        &dir,
        "c23",
        "int a[] = {\n#embed \"two.bin\" prefix(1,) suffix(,9)\n};\n",
    )
    .expect("ok");
    assert_eq!(int_values(&toks), vec![1, 10, 20, 9]);
}

#[test]
fn expansion_if_empty_on_empty() {
    let dir = scratch();
    std::fs::write(dir.join("empty.bin"), b"").expect("write");
    let toks = preprocess_in(&dir, "c23", "int a = \n#embed \"empty.bin\" if_empty(0)\n;\n")
        .expect("ok");
    assert_eq!(int_values(&toks), vec![0]);
}

#[test]
fn pre_c23_is_rejected() {
    let dir = scratch();
    std::fs::write(dir.join("res.bin"), b"hi!").expect("write");
    let err = preprocess_in(&dir, "c17", "int a[] = {\n#embed \"res.bin\"\n};\n")
        .expect_err("must reject");
    assert!(
        err.iter().any(|d| d.message.contains("#embed") && d.message.contains("C23")),
        "expected a C23-feature diagnostic, got: {:?}",
        err.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn missing_resource_errors() {
    let dir = scratch();
    let err = preprocess_in(&dir, "c23", "int a[] = {\n#embed \"nope.bin\"\n};\n")
        .expect_err("must error");
    assert!(
        err.iter().any(|d| d.message.contains("cannot find embed resource")),
        "expected a missing-resource diagnostic, got: {:?}",
        err.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn macro_produces_filename() {
    let dir = scratch();
    std::fs::write(dir.join("res.bin"), b"hi!").expect("write");
    // The resource name comes from a macro; the argument line is expanded.
    let src = "#define RES \"res.bin\"\nint a[] = {\n#embed RES\n};\n";
    let toks = preprocess_in(&dir, "c23", src).expect("ok");
    assert_eq!(int_values(&toks), vec![104, 105, 33]);
}

// --- helpers ---------------------------------------------------------------

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

/// Probe whether `gcc` accepts `#embed` under `-std=c23` (older gccs do not).
fn gcc_has_embed(gcc: &str, dir: &Path) -> bool {
    let res = dir.join("probe.bin");
    if std::fs::write(&res, b"A").is_err() {
        return false;
    }
    let c = dir.join("probe.c");
    let src = format!("int main(void){{ unsigned char d[]={{\n#embed \"{}\"\n}}; return d[0]; }}", res.display());
    if std::fs::write(&c, src).is_err() {
        return false;
    }
    let bin = dir.join("probe.bin.out");
    Command::new(gcc)
        .args(["-std=c23", "-w", "-o"])
        .arg(&bin)
        .arg(&c)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn run_exit(bin: &Path) -> i32 {
    let status = Command::new(bin).status().expect("run binary");
    if let Some(code) = status.code() {
        return code;
    }
    use std::os::unix::process::ExitStatusExt;
    panic!("{} killed by signal {:?}", bin.display(), status.signal());
}
