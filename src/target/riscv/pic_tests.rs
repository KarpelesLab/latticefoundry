//! Position-independent code on RISC-V (`docs/ir-design.md` §4b): under
//! `Pic`/`Pie`, a symbol that may bind outside the component is addressed
//! through its GOT entry (`auipc`+`ld`, `R_RISCV_GOT_HI20` +
//! `R_RISCV_PCREL_LO12_I`), a locally bound one PC-relatively, calls stay
//! `R_RISCV_CALL_PLT`, and constants holding addresses move to
//! `.data.rel.ro`. Checked on the object's relocations, by running PIC code
//! (GOT included) on the simulator, and by linking shared libraries and PIEs
//! with qld and running them from the loaded file.

use std::collections::HashMap;

use crate::codegen::{CodegenOptions, RelocModel};
use crate::mc::object::{ObjectModule, RelocKind, SectionKind};

use super::diff_tests::parse;
use super::sim::{Cpu, STACK_TOP, link, load_elf};

const SRC: &str = r#"
module "pic"
global @data : i64 = i64 40
global hidden @hdata : i64 = i64 2
global internal @idata : i64 = i64 3
global @ext : i64
global constant @ptrs : [2 x ptr] = [2 x ptr] (ptr @data, ptr @hdata)
func @get() -> i64 {
entry ^0:
  %a = load @data align 8 : i64
  %b = load @hdata align 8 : i64
  %c = load @idata align 8 : i64
  %p = ptr_add @ptrs, i64 8 : ptr
  %q = load %p align 8 : ptr
  %d = load %q align 8 : i64
  %s1 = add %a, %b : i64
  %s2 = add %s1, %c : i64
  %s3 = add %s2, %d : i64
  ret %s3
}
func hidden @twice(i64) -> i64 {
entry ^0(%x: i64):
  %r = add %x, %x : i64
  ret %r
}
func @apply(i64) -> i64 {
entry ^0(%x: i64):
  %f = select i1 1, @twice, @get : ptr
  %r = call %f(%x) : i64
  %g = call @get() : i64
  %s = add %r, %g : i64
  ret %s
}
func @use_ext() -> i64 {
entry ^0:
  %e = load @ext align 8 : i64
  ret %e
}
"#;

/// The high-part relocation (`GotHi20` or `PcrelHi20`) on each symbol the
/// code addresses.
fn hi_kinds(obj: &ObjectModule) -> HashMap<String, RelocKind> {
    obj.relocations()
        .iter()
        .filter(|r| matches!(r.kind, RelocKind::RiscvGotHi20 | RelocKind::RiscvPcrelHi20))
        .map(|r| (obj.symbol(r.symbol).name.clone(), r.kind))
        .collect()
}

#[test]
fn preemptible_symbols_go_through_the_got() {
    let (m, syms) = parse(SRC);
    let compile = |model| super::compile_module_with(&m, &syms, &CodegenOptions::default().with_reloc_model(model)).object;
    let (stat, pie, pic) = (compile(RelocModel::Static), compile(RelocModel::Pie), compile(RelocModel::Pic));
    let (got, pc) = (RelocKind::RiscvGotHi20, RelocKind::RiscvPcrelHi20);
    // Static: everything PC-relative.
    assert!(hi_kinds(&stat).values().all(|&k| k == pc));
    // PIC: default-visibility symbols (defined or not) through the GOT;
    // hidden and internal ones PC-relative.
    let k = hi_kinds(&pic);
    assert_eq!((k["data"], k["ext"], k["get"], k["ptrs"]), (got, got, got, got));
    assert_eq!((k["hdata"], k["idata"], k["twice"]), (pc, pc, pc));
    // PIE: everything this module defines binds locally; only `ext` needs
    // the GOT.
    let k = hi_kinds(&pie);
    assert_eq!(k["ext"], got);
    assert!(k.iter().filter(|(n, _)| *n != "ext").all(|(_, &v)| v == pc), "{k:?}");
    for obj in [&pie, &pic] {
        // No absolute relocation in code; the address table moved to
        // `.data.rel.ro`; calls stay CALL_PLT.
        for r in obj.relocations() {
            let sec = obj.section(r.section);
            if sec.kind == SectionKind::Text {
                assert!(r.kind.is_riscv(), "{:?} in code", r.kind);
            }
        }
        let relro = obj.sections().iter().find(|s| s.name == ".data.rel.ro").expect(".data.rel.ro");
        assert_eq!(relro.bytes.len(), 16);
        assert!(obj.relocations().iter().any(|r| r.kind == RelocKind::RiscvCallPlt));
    }
    // A GOT load is `auipc` + `ld`, and its low part points at the `auipc`.
    let text = &pic.sections()[0].bytes;
    for r in pic.relocations().iter().filter(|r| r.kind == RelocKind::RiscvGotHi20) {
        let w = u32::from_le_bytes(text[r.offset as usize + 4..r.offset as usize + 8].try_into().unwrap());
        assert_eq!((w & 0x7f, (w >> 12) & 7), (0x03, 3), "an ld follows the auipc");
    }
}

/// PIC code runs: the simulator's linker gives every GOT-addressed symbol a
/// GOT slot holding its address.
#[test]
fn pic_code_runs_through_its_got() {
    let (m, syms) = parse(SRC);
    for model in [RelocModel::Pic, RelocModel::Pie, RelocModel::Static] {
        let obj = super::compile_module_with(&m, &syms, &CodegenOptions::default().with_reloc_model(model)).object;
        let mut ext = ObjectModule::new("ext");
        let d = ext.add_section(crate::mc::object::Section::new(".data", SectionKind::Data, 8));
        ext.section_mut(d).bytes = 1000u64.to_le_bytes().to_vec();
        ext.add_symbol(crate::mc::object::Symbol::defined(
            "ext",
            crate::mc::object::SymbolBinding::Global,
            crate::mc::object::SymbolType::Object,
            d,
            0,
            8,
        ));
        let image = link(&[&obj, &ext]).unwrap();
        let run = |name: &str, args: &[u64]| {
            let mut cpu = Cpu::new(&image);
            cpu.call(image.symbols[name], args, &[], STACK_TOP - 4096).unwrap();
            cpu.x[10]
        };
        assert_eq!(run("get", &[]), 47, "{model:?}");
        assert_eq!(run("apply", &[5]), 57, "{model:?}");
        assert_eq!(run("use_ext", &[]), 1000, "{model:?}");
    }
}

/// qld links PIC objects into a shared library (no text relocations; the
/// GOT and address table filled by dynamic relocations) and a PIE; loaded
/// from the file with its dynamic relocations applied, both run.
#[test]
fn shared_libraries_and_pies_link_with_qld_and_run() {
    let dir = std::env::temp_dir().join(format!("lf-rv-pic-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = SRC.replace("global @ext : i64\n", "global @ext : i64 = i64 1000\n");
    let (m, syms) = parse(&src);
    for (model, flags, out) in [(RelocModel::Pic, ["-shared", "-z"], "libpic.so"), (RelocModel::Pie, ["-pie", "--no-dynamic-linker"], "pie")] {
        let obj = super::compile_module_with(&m, &syms, &CodegenOptions::default().with_reloc_model(model)).object;
        let (o, so) = (dir.join(format!("{out}.o")), dir.join(out));
        std::fs::write(&o, super::write_elf(&obj).unwrap()).unwrap();
        let mut args: Vec<&str> = vec!["-m", "elf64lriscv"];
        args.extend(flags);
        if flags[0] == "-shared" {
            args.push("text");
        }
        args.extend(["-e", "apply", "-o", so.to_str().unwrap(), o.to_str().unwrap()]);
        crate::link::gnu::link_gnu("qld", &args).unwrap_or_else(|e| panic!("qld {model:?}: {e}"));
        let bytes = std::fs::read(&so).unwrap();
        assert_eq!(u16::from_le_bytes([bytes[16], bytes[17]]), 3, "ET_DYN");
        if let Ok(out) = std::process::Command::new("llvm-readobj").args(["-d", "-r"]).arg(&so).output() {
            let text = String::from_utf8_lossy(&out.stdout);
            assert!(!text.contains("TEXTREL"), "{text}");
            assert!(text.contains("R_RISCV_64") || text.contains("R_RISCV_RELATIVE"), "{text}");
        }
        let image = load_elf(&bytes).unwrap();
        let run = |name: &str, args: &[u64]| {
            let mut cpu = Cpu::new(&image);
            cpu.call(image.symbols[name], args, &[], STACK_TOP - 4096).unwrap();
            cpu.x[10]
        };
        assert_eq!(run("get", &[]), 47, "{model:?}");
        assert_eq!(run("apply", &[5]), 57, "{model:?}");
        assert_eq!(run("use_ext", &[]), 1000, "{model:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
