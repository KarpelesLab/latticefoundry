//! Two-word values on AArch64 (issue #16): `i128` in register pairs
//! (`codegen::wide`, AAPCS64's `__int128` convention) and `{i64, i64}`
//! results in `x0:x1` without a stack slot (`codegen::aggret`). Programs run
//! on the A64 emulator; the ABI is checked against clang's code when clang
//! can target `aarch64-linux-gnu`.

use crate::mc::object::{ObjectModule, RelocKind};
use crate::target::wide_fixtures::{
    ABI_C, ABI_LF, LIBGCC_OPS, LODE_CALLEE, LODE_CALLER, LODE_FUNCS, LODE_MAIN, emu_programs, failing, frame_slots,
    func_id, panic_text, parse, parse_at, suite_src,
};
use crate::transform::pipeline::OptLevel;

use crate::ir::Module;
use crate::support::StrInterner;

use super::emu;
use super::isel::AArch64Target;

/// Compile `src` at `level`.
fn object(src: &str, level: OptLevel) -> ObjectModule {
    let (m, syms) = parse_at(src, level);
    super::compile_module(&m, &syms)
}

/// Link `objs` (and the `extra` object files) into a static executable
/// entered at `main` and run it on the emulator: the exit status.
fn run(objs: Vec<ObjectModule>, extra: &[std::path::PathBuf], tag: &str) -> u64 {
    let dir = std::env::temp_dir().join(format!("lf-a64-wide-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("a.out");
    let extra: Vec<String> = extra.iter().map(|p| p.display().to_string()).collect();
    super::link::link_executable(objs, "main", &extra, &exe).expect("qld links the program");
    let elf = std::fs::read(&exe).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let (code, _) = emu::run_executable(&elf).unwrap_or_else(|e| panic!("{tag}: {e}"));
    code
}

/// Every `i128` operation that needs no libgcc, on random operands, against
/// the reference evaluator, with the suite at `-O0` and at `-O2`.
#[test]
fn i128_ops_match_the_reference_evaluator() {
    let (suite, main, names) = emu_programs(8);
    for level in [OptLevel::O0, OptLevel::O2] {
        let code = run(vec![object(&suite, level), object(&main, OptLevel::O0)], &[], &format!("ops{level:?}"));
        assert_eq!(code, 0, "{level:?}: failing {:?}", failing(code, &names));
    }
}

/// Division and the float conversions call libgcc's `__int128` helpers; the
/// multiply is inline (`umulh`), never `__multi3`.
#[test]
fn i128_division_calls_libgcc_and_multiply_is_inline() {
    let src = suite_src(&|n| LIBGCC_OPS.contains(&n) || n == "mul");
    let obj = object(&src, OptLevel::O0);
    let called: Vec<&str> =
        obj.relocations().iter().filter(|r| r.kind == RelocKind::Aarch64Call26).map(|r| obj.symbol(r.symbol).name.as_str()).collect();
    for want in ["__udivti3", "__divti3", "__umodti3", "__modti3", "__floattidf", "__floatuntisf", "__fixdfti", "__fixunssfti"] {
        assert!(called.contains(&want), "{want}: {called:?}");
    }
    assert!(!called.iter().any(|n| n.contains("mul")), "{called:?}");
}

/// A `switch` on an `i128` and an integer wider than 128 bits at the ABI are
/// rejected with a diagnostic, not miscompiled.
#[test]
fn unsupported_wide_forms_are_rejected() {
    for (src, msg) in [
        (suite_src(&|n| n == "switch"), "switch on an integer wider than 64 bits"),
        ("module \"w\"\nfunc @f(i256) -> i256 {\nentry ^0(%a: i256):\n  ret %a\n}\n".to_owned(), "wider than 128 bits"),
    ] {
        let (m, syms) = parse(&src);
        let r = std::panic::catch_unwind(|| super::compile_module(&m, &syms));
        let e = panic_text(&*r.expect_err("rejected"));
        assert!(e.contains(msg), "{e}");
    }
}

/// Lode-style two-word results — a `throws(E) -> usize` struct built in
/// branches and branched on, a plain pair, an `i128` — across separately
/// compiled modules, at `-O0` and `-O2`.
#[test]
fn lode_two_word_results_run() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let objs = vec![object(LODE_CALLEE, level), object(LODE_CALLER, level), object(LODE_MAIN, OptLevel::O0)];
        assert_eq!(run(objs, &[], &format!("lode{level:?}")), 0, "{level:?}: failing checks (bitmask)");
    }
}

/// At `-O2` neither side of a two-word result needs a stack slot: the
/// callee builds it in `x0:x1`, the caller reads the fields from there.
#[test]
fn two_word_results_need_no_stack_slot_at_o2() {
    let target = AArch64Target::new();
    let select = |m: &Module, syms: &StrInterner, name: &str| {
        let wide = crate::codegen::wide::prepared_if_wide(m, syms, "aarch64");
        let (m, syms) = wide.as_ref().map_or((m, syms), |(m, s)| (m, s));
        target.select_with_syms(m, func_id(m, syms, name), syms)
    };
    for name in LODE_FUNCS {
        let src = if ["parse_digit", "pair", "wide_pair"].contains(name) { LODE_CALLEE } else { LODE_CALLER };
        assert_eq!(frame_slots(src, name, &target, &select), 0, "{name}");
    }
}

/// Compile [`ABI_C`] with clang for AArch64 Linux, if clang can.
fn clang_object(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    std::fs::create_dir_all(dir).ok()?;
    let src = dir.join("c.c");
    std::fs::write(&src, ABI_C).ok()?;
    let obj = dir.join("c.o");
    let out = std::process::Command::new("clang")
        .args(["--target=aarch64-linux-gnu", "-ffreestanding", "-fno-stack-protector", "-O1", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .output()
        .ok()?;
    out.status.success().then_some(obj)
}

/// `i128` arguments (register pairs, the stack past `x7`) and results, and
/// `{long, long}` results, both ways with clang's code, at `-O0` and `-O2`.
#[test]
fn two_word_abi_matches_clang() {
    let dir = std::env::temp_dir().join(format!("lf-a64-wide-clang-{}", std::process::id()));
    let Some(c) = clang_object(&dir) else {
        eprintln!("skipping two_word_abi_matches_clang: clang cannot target aarch64-linux-gnu");
        return;
    };
    let main = "module \"m\"\nfunc @cmain() -> i32\nfunc @main() -> i64 {\nentry ^0:\n  %r = call @cmain() : i32\n  %z = zext %r : i64\n  ret %z\n}\n";
    for level in [OptLevel::O0, OptLevel::O2] {
        let objs = vec![object(LODE_CALLEE, level), object(LODE_CALLER, level), object(ABI_LF, level), object(main, OptLevel::O0)];
        assert_eq!(run(objs, std::slice::from_ref(&c), &format!("clang{level:?}")), 0, "{level:?}: failing checks (bitmask)");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
