//! Two-word results on x86-64 System V (issue #16): a `{i64, i64}` struct
//! returned in `rax:rdx` without a stack slot on either side of the call
//! (`codegen::aggret`, after `transform::Sroa`), next to the `i128` the
//! backend already returns there; checked on the machine code and, on an
//! x86-64 Linux host with gcc, both ways against gcc's code.

use crate::ir::Module;
use crate::mc::disasm::{Options, Syntax, decode};
use crate::mc::object::{ObjectModule, SymbolValue};
use crate::support::StrInterner;
use crate::target::TargetArch;
use crate::target::wide_fixtures::{LODE_CALLEE, LODE_CALLER, LODE_FUNCS, frame_slots, func_id, parse_at};
use crate::transform::pipeline::OptLevel;

use super::isel::X86_64Target;

/// Compile `src` at `level`.
fn object(src: &str, level: OptLevel) -> ObjectModule {
    let (m, syms) = parse_at(src, level);
    super::compile_module(&m, &syms)
}

/// The Intel-syntax instructions of the defined function `name` in `obj`.
fn listing(obj: &ObjectModule, name: &str) -> Vec<String> {
    let sym = obj.symbols().iter().find(|s| s.name == name).unwrap_or_else(|| panic!("no symbol {name}"));
    let SymbolValue::Defined { section, offset } = sym.value else { panic!("{name} undefined") };
    let bytes = &obj.section(section).bytes[offset as usize..(offset + sym.size) as usize];
    let opts = Options { syntax: Syntax::Intel };
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let i = decode(TargetArch::X86_64, &bytes[at..], at as u64, &opts);
        assert!(i.known, "{name}: undecodable bytes at {at}");
        out.push(i.text().replace('\t', " "));
        at += i.len;
    }
    out
}

/// At `-O2` neither side of a two-word result needs a stack slot: the
/// callee builds it in `rax:rdx`, the caller reads the fields from there.
#[test]
fn two_word_results_need_no_stack_slot_at_o2() {
    let target = X86_64Target::new();
    let select = |m: &Module, syms: &StrInterner, name: &str| {
        let wide = super::prepared_if_wide(m, syms);
        let (m, syms) = wide.as_ref().map_or((m, syms), |(m, s)| (m, s));
        target.select_with_syms(m, func_id(m, syms, name), syms)
    };
    for name in LODE_FUNCS {
        let src = if ["parse_digit", "pair", "wide_pair"].contains(name) { LODE_CALLEE } else { LODE_CALLER };
        assert_eq!(frame_slots(src, name, &target, &select), 0, "{name}");
    }
}

/// The machine code of the `throws(E) -> usize` round trip at `-O2`: no
/// memory operand on the stack in the callee or the caller, which tests the
/// error word in `rax` right after the call and reads the value from `rdx`.
#[test]
fn throws_result_travels_in_rax_rdx() {
    let callee = object(LODE_CALLEE, OptLevel::O2);
    let caller = object(LODE_CALLER, OptLevel::O2);
    for (obj, name) in [(&callee, "parse_digit"), (&callee, "pair"), (&caller, "sum_digits"), (&caller, "try_sum")] {
        let code = listing(obj, name);
        let stack: Vec<&String> = code.iter().filter(|t| t.contains("[rsp") || t.contains("[rbp")).collect();
        assert!(stack.is_empty(), "{name}: {stack:?}\n{}", code.join("\n"));
    }
    let code = listing(&caller, "sum_digits");
    let call = code.iter().position(|t| t.starts_with("call")).expect("a call");
    assert!(code[call + 1].starts_with("test rax, rax"), "{}", code.join("\n"));
    assert!(code.iter().any(|t| t.contains("rdx")), "{}", code.join("\n"));
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use super::*;
    use crate::target::wide_fixtures::{ABI_C, ABI_LF, LODE_MAIN};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn have_gcc() -> bool {
        Command::new("gcc").arg("--version").output().is_ok_and(|o| o.status.success())
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lf-x64-pair-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Write `src` compiled at `level` to `dir/<name>.o`.
    fn object_file(dir: &Path, name: &str, src: &str, level: OptLevel) -> PathBuf {
        let p = dir.join(format!("{name}.o"));
        std::fs::write(&p, crate::mc::elf::write(&object(src, level))).unwrap();
        p
    }

    /// Link `inputs` (C sources and objects) with gcc and run the program:
    /// its exit status and output.
    fn gcc_run(dir: &Path, inputs: &[PathBuf]) -> (i32, String) {
        let exe = dir.join("main");
        let out = Command::new("gcc").arg("-O1").args(inputs).arg("-o").arg(&exe).output().unwrap();
        assert!(out.status.success(), "gcc: {}", String::from_utf8_lossy(&out.stderr));
        let out = loop {
            match Command::new(&exe).output() {
                Ok(o) => break o,
                Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
                Err(e) => panic!("exec: {e}"),
            }
        };
        (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// The Lode-style programs, linked by gcc against the C runtime, at
    /// `-O0` and `-O2`: `main`'s exit status is its failing-check bitmask.
    #[test]
    fn lode_two_word_results_run() {
        if !have_gcc() {
            eprintln!("skipping: gcc not found");
            return;
        }
        for level in [OptLevel::O0, OptLevel::O2] {
            let dir = scratch(&format!("lode{level:?}"));
            let objs = vec![
                object_file(&dir, "callee", LODE_CALLEE, level),
                object_file(&dir, "caller", LODE_CALLER, level),
                object_file(&dir, "main", LODE_MAIN, OptLevel::O0),
            ];
            let (code, _) = gcc_run(&dir, &objs);
            assert_eq!(code, 0, "{level:?}: failing checks (bitmask)");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// `struct { long err; unsigned long val; }` and `__int128` results and
    /// arguments, both ways with gcc's code, at `-O0` and `-O2`.
    #[test]
    fn two_word_abi_matches_gcc() {
        if !have_gcc() {
            eprintln!("skipping: gcc not found");
            return;
        }
        for level in [OptLevel::O0, OptLevel::O2] {
            let dir = scratch(&format!("abi{level:?}"));
            let c = dir.join("c.c");
            std::fs::write(&c, ABI_C).unwrap();
            let main = dir.join("main.c");
            std::fs::write(
                &main,
                "#include <stdio.h>\nint cmain(void);\nint main(void) { int b = cmain(); printf(\"bad=%d\\n\", b); return b != 0; }\n",
            )
            .unwrap();
            let inputs = vec![
                main,
                c,
                object_file(&dir, "callee", LODE_CALLEE, level),
                object_file(&dir, "caller", LODE_CALLER, level),
                object_file(&dir, "abi", ABI_LF, level),
            ];
            let (code, out) = gcc_run(&dir, &inputs);
            assert!(code == 0 && out.contains("bad=0"), "{level:?}: {out}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
