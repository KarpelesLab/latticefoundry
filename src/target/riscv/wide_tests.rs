//! Two-word values on RISC-V (issue #16): `i128` as the psABI's 2×XLEN
//! scalar in a register pair (`codegen::wide`) and `{i64, i64}` results in
//! `a0:a1` without a stack slot (`codegen::aggret`). Programs run on the
//! simulator; the LP64D convention is checked against clang's code when
//! clang can target riscv64.

use std::path::Path;

use crate::codegen::CodegenOptions;
use crate::ir::Module;
use crate::mc::object::{ObjectModule, RelocKind};
use crate::support::StrInterner;
use crate::target::wide_fixtures::{
    ABI_C, ABI_LF, LIBGCC_OPS, LODE_CALLEE, LODE_CALLER, LODE_FUNCS, LODE_MAIN, emu_programs, failing, frame_slots,
    func_id, panic_text, parse, parse_at, suite_src,
};
use crate::transform::pipeline::OptLevel;

use super::encode::target_for;
use super::sim::{Cpu, STACK_TOP, link, load_elf};

/// Compile `src` at `level`.
fn object(src: &str, level: OptLevel) -> ObjectModule {
    let (m, syms) = parse_at(src, level);
    super::compile_module(&m, &syms)
}

/// Link `objs` in the simulator and run `main`: its result.
fn run(objs: &[ObjectModule]) -> u64 {
    let refs: Vec<&ObjectModule> = objs.iter().collect();
    let image = link(&refs).expect("the objects link");
    let entry = *image.symbols.get("main").expect("a main");
    let mut cpu = Cpu::new(&image);
    cpu.call(entry, &[], &[], STACK_TOP - 4096).unwrap_or_else(|e| panic!("main: {e}"));
    cpu.x[10]
}

/// Every `i128` operation that needs no libgcc, on random operands, against
/// the reference evaluator, with the suite at `-O0` and at `-O2`.
#[test]
fn i128_ops_match_the_reference_evaluator() {
    let (suite, main, names) = emu_programs(8);
    for level in [OptLevel::O0, OptLevel::O2] {
        let code = run(&[object(&suite, level), object(&main, OptLevel::O0)]);
        assert_eq!(code, 0, "{level:?}: failing {:?}", failing(code, &names));
    }
}

/// Division and the float conversions call libgcc's `__int128` helpers; the
/// multiply is inline (`mulhu`), never `__multi3`.
#[test]
fn i128_division_calls_libgcc_and_multiply_is_inline() {
    let src = suite_src(&|n| LIBGCC_OPS.contains(&n) || n == "mul");
    let obj = object(&src, OptLevel::O0);
    let called: Vec<&str> = obj
        .relocations()
        .iter()
        .filter(|r| r.kind == RelocKind::RiscvCallPlt)
        .map(|r| obj.symbol(r.symbol).name.as_str())
        .collect();
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
        let objs = [object(LODE_CALLEE, level), object(LODE_CALLER, level), object(LODE_MAIN, OptLevel::O0)];
        assert_eq!(run(&objs), 0, "{level:?}: failing checks (bitmask)");
    }
}

/// At `-O2` neither side of a two-word result needs a stack slot: the
/// callee builds it in `a0:a1`, the caller reads the fields from there.
#[test]
fn two_word_results_need_no_stack_slot_at_o2() {
    let opts = CodegenOptions::default();
    for name in LODE_FUNCS {
        let src = if ["parse_digit", "pair", "wide_pair"].contains(name) { LODE_CALLEE } else { LODE_CALLER };
        let select = |m: &Module, syms: &StrInterner, name: &str| {
            let wide = crate::codegen::wide::prepared_if_wide(m, syms, "riscv64");
            let (m, syms) = wide.as_ref().map_or((m, syms), |(m, s)| (m, s));
            let f = func_id(m, syms, name);
            target_for(m, f, Some(syms), &opts).select(m, f)
        };
        let target = super::isel::RiscvTarget::new();
        assert_eq!(frame_slots(src, name, &target, &select), 0, "{name}");
    }
}

/// Run a tool, returning whether it ran and succeeded.
fn tool(cmd: &str, args: &[&str]) -> bool {
    std::process::Command::new(cmd).args(args).output().is_ok_and(|o| o.status.success())
}

/// `i128` arguments (register pairs, one split between `a7` and the stack,
/// the stack) and results, and `{long, long}` results, both ways with
/// clang's LP64D code, at `-O0` and `-O2`.
#[test]
fn two_word_abi_matches_clang() {
    let dir = std::env::temp_dir().join(format!("lf-rv-wide-clang-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (c_src, c_obj) = (dir.join("c.c"), dir.join("c.o"));
    std::fs::write(&c_src, ABI_C).unwrap();
    let args = [
        "--target=riscv64-unknown-linux-gnu",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-O1",
        "-fno-pic",
        "-mno-relax",
        "-ffreestanding",
        "-c",
        c_src.to_str().unwrap(),
        "-o",
        c_obj.to_str().unwrap(),
    ];
    if !tool("clang", &args) {
        eprintln!("skipping two_word_abi_matches_clang: no clang with a riscv64 target");
        return;
    }
    let main = "module \"m\"\nfunc @cmain() -> i32\nfunc @main() -> i64 {\nentry ^0:\n  %r = call @cmain() : i32\n  %z = zext %r : i64\n  ret %z\n}\n";
    for level in [OptLevel::O0, OptLevel::O2] {
        let mut paths = vec![c_obj.clone()];
        for (k, (src, lvl)) in [(LODE_CALLEE, level), (LODE_CALLER, level), (ABI_LF, level), (main, OptLevel::O0)].into_iter().enumerate() {
            let p = dir.join(format!("lf{k}.o"));
            std::fs::write(&p, super::write_elf(&object(src, lvl)).expect("an ELF object")).unwrap();
            paths.push(p);
        }
        let exe = dir.join("x");
        let bytes = qld_static(&paths, &exe);
        let image = load_elf(&bytes).unwrap();
        let entry = *image.symbols.get("main").expect("a main");
        let mut cpu = Cpu::new(&image);
        cpu.call(entry, &[], &[], STACK_TOP - 4096).unwrap_or_else(|e| panic!("{level:?}: {e}"));
        assert_eq!(cpu.x[10], 0, "{level:?}: failing checks (bitmask)");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Link `objects` statically with qld, returning the executable's bytes.
fn qld_static(objects: &[std::path::PathBuf], out: &Path) -> Vec<u8> {
    let mut args: Vec<String> = ["-m", "elf64lriscv", "-static", "-e", "main", "-o"].map(Into::into).to_vec();
    args.push(out.to_str().unwrap().into());
    args.extend(objects.iter().map(|p| p.to_str().unwrap().to_owned()));
    crate::link::gnu::link_gnu("qld", &args).unwrap_or_else(|e| panic!("qld: {e}"));
    std::fs::read(out).unwrap()
}
