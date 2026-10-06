//! Bulk memory on AArch64 (`docs/ir-design.md` §6k): the shared fixtures
//! ([`crate::target::bulk_fixtures`]) compiled to machine code and run on the
//! A64 emulator, lengths from 0 to 300 at unaligned and aligned offsets, and
//! on the MIR interpreter, against the reference executor; plus the shapes
//! (`q`-register chunks inline, a word loop beyond).

use super::emu::{Emu, Stop};
use super::encode::compile_module_with;
use super::isel::{AArch64Target, NeonLegality};
use crate::codegen::CodegenOptions;
use crate::codegen::legalize::legalized;
use crate::codegen::legalize_mem::uses_bulk_memory;
use crate::ir::FuncId;
use crate::mc::disasm::{Options, Syntax, decode};
use crate::mc::object::{ObjectModule, SymbolValue};
use crate::target::TargetArch;
use crate::target::bulk_fixtures::{CONSTS, bulk, some_lengths};
use crate::target::vector_fixtures::{assert_matches, parse, reference};

use puremp::Int;

fn func_bytes<'a>(obj: &'a ObjectModule, name: &str) -> &'a [u8] {
    let sym = obj.symbols().iter().find(|s| s.name == name).expect("function symbol");
    let SymbolValue::Defined { section, offset } = sym.value else { panic!("{name} undefined") };
    &obj.section(section).bytes[offset as usize..(offset + sym.size) as usize]
}

#[test]
fn fixtures_run_as_machine_code() {
    let (src, cases) = bulk(&some_lengths(), &CONSTS);
    let want = reference(&src, &cases);
    let (m, syms) = parse(&src);
    let obj = compile_module_with(&m, &syms, &CodegenOptions::default()).object;
    let top = 0x7000_0000_0000u64;
    let mut got = Vec::new();
    for (name, args) in &cases {
        let code = func_bytes(&obj, name);
        let mut emu = Emu::new();
        emu.map(0x1_0000, code.len() as u64);
        emu.poke(0x1_0000, code);
        emu.map_stack(top, 1 << 16, false);
        let a: Vec<u64> = args.iter().map(|&x| x as u64).collect();
        let stop = emu.call(0x1_0000, &a, 50_000_000).unwrap_or_else(|e| panic!("@{name}{args:?}: {e}"));
        assert_eq!(stop, Stop::Returned, "@{name}{args:?}");
        got.push(emu.x[0]);
    }
    assert_matches("aarch64 bulk (machine code)", &cases, &got, &want);
}

#[test]
fn fixtures_run_on_the_mir_interpreter() {
    let (src, cases) = bulk(&some_lengths(), &CONSTS);
    let want = reference(&src, &cases);
    let (m, syms) = parse(&src);
    let legal = legalized(&m, &NeonLegality);
    assert!(!uses_bulk_memory(&legal), "every op is expanded on AArch64");
    let target = AArch64Target::new();
    let funcs: Vec<_> = (0..legal.function_count()).map(|i| target.select(&legal, FuncId::from_index(i))).collect();
    let got: Vec<u64> = cases
        .iter()
        .map(|(name, args)| {
            let idx = legal.functions().position(|f| syms.resolve(f.name) == name).expect("case function");
            let a: Vec<Int> = args.iter().map(|&x| Int::from_i64(x)).collect();
            super::interp::run(&target, &funcs, idx, &a)
                .unwrap_or_else(|e| panic!("@{name}{args:?}: {e}"))
                .expect("a result")
                .to_u64()
                .expect("64 bits")
        })
        .collect();
    assert_matches("aarch64 bulk (MIR)", &cases, &got, &want);
}

#[test]
fn shapes() {
    let src = r#"module "s"
func @small(ptr, ptr) -> void {
entry ^0(%d: ptr, %s: ptr):
  memcpy %d, %s, i64 40 align 8
  ret
}
func @big(ptr, i8) -> void {
entry ^0(%d: ptr, %b: i8):
  memset %d, %b, i64 4096 align 8
  ret
}
"#;
    let (m, syms) = parse(src);
    let obj = compile_module_with(&m, &syms, &CodegenOptions::default()).object;
    let dis = |name: &str| {
        let bytes = func_bytes(&obj, name);
        let opts = Options { syntax: Syntax::Intel };
        (0..bytes.len() / 4)
            .map(|k| decode(TargetArch::AArch64, &bytes[4 * k..], 4 * k as u64, &opts).text())
            .collect::<Vec<_>>()
            .join("\n")
    };
    let small = dis("small");
    // 16 + 16 + 8: two q-register pairs and one x-register pair, no loop.
    assert_eq!(small.matches("\tq0, [").count(), 4, "{small}");
    assert!(!small.contains("b."), "{small}");
    let big = dis("big");
    assert!(big.lines().any(|l| l.starts_with("b.") || l.starts_with("cb")), "a loop: {big}");
}
