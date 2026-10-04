//! Position-independent code generation on x86-64: which relocation each kind
//! of symbol reference produces under the `Static`, `Pie` and `Pic` models, the
//! instruction bytes of a GOT load, `.data.rel.ro` placement, and that no
//! absolute 32-bit relocation appears in PIC output.

use crate::codegen::{CodegenOptions, RelocModel};
use crate::ir::text::parse_module;
use crate::mc::object::{ObjectModule, RelocKind, SymbolVisibility};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::target::x86_64::compile_module_with;
use crate::target::{CodegenError, TargetArch, compile_module_for};

const SRC: &str = r#"module "pic"
global @counter : i64 = i64 1
global hidden @secret : i64 = i64 2
global internal @priv : i64 = i64 3
global protected @prot : i64 = i64 4
global @ext : i64
global constant @table : [2 x ptr] = [2 x ptr] (ptr @counter, ptr @cb)
global constant @plain : i64 = i64 9

func @strlen(ptr) -> i64
func hidden @helper() -> i64
func @cb() -> i64 {
entry ^0:
  ret i64 1
}
func internal @local() -> i64 {
entry ^0:
  ret i64 2
}

func @uses() -> i64 {
entry ^0:
  %a = load @counter align 8 : i64
  %b = load @secret align 8 : i64
  %c = load @priv align 8 : i64
  %d = load @prot align 8 : i64
  %e = load @ext align 8 : i64
  %f = call @cb() : i64
  %g = call @local() : i64
  %h = call @strlen(@table) : i64
  %i = call @helper() : i64
  %p = call @takes(@cb) : i64
  %q = call @takes(@local) : i64
  %r = call @takes(@strlen) : i64
  %s = add %a, %b : i64
  ret %s
}
func @takes(ptr) -> i64 {
entry ^0(%x: ptr):
  ret i64 0
}
"#;

fn compile(model: RelocModel) -> ObjectModule {
    let mut syms = StrInterner::new();
    let m = parse_module(SRC, FileId::new(0), &mut syms).expect("parse");
    crate::verify::verify_module(&m).expect("verify");
    compile_module_with(&m, &syms, &CodegenOptions::default().with_reloc_model(model)).object
}

/// The relocation kinds `.text` applies to `name`, in order.
fn text_relocs(obj: &ObjectModule, name: &str) -> Vec<RelocKind> {
    obj.relocations()
        .iter()
        .filter(|r| obj.section(r.section).name == ".text" && obj.symbol(r.symbol).name == name)
        .map(|r| r.kind)
        .collect()
}

#[test]
fn pic_addresses_preemptible_symbols_through_the_got() {
    let obj = compile(RelocModel::Pic);
    use RelocKind::{GotPcRel, Pc32, Plt32};
    // Default-visibility and protected data, and undefined data: GOT.
    assert_eq!(text_relocs(&obj, "counter"), [GotPcRel]);
    assert_eq!(text_relocs(&obj, "prot"), [GotPcRel]);
    assert_eq!(text_relocs(&obj, "ext"), [GotPcRel]);
    assert_eq!(text_relocs(&obj, "table"), [GotPcRel]);
    // Hidden and internal data: direct RIP-relative.
    assert_eq!(text_relocs(&obj, "secret"), [Pc32]);
    assert_eq!(text_relocs(&obj, "priv"), [Pc32]);
    // Calls always go through PLT32; an address taken of a preemptible or
    // external function comes from the GOT, of an internal one directly.
    assert_eq!(text_relocs(&obj, "cb"), [Plt32, GotPcRel]);
    assert_eq!(text_relocs(&obj, "local"), [Plt32, Pc32]);
    assert_eq!(text_relocs(&obj, "strlen"), [Plt32, GotPcRel]);
    assert_eq!(text_relocs(&obj, "helper"), [Plt32]);
    // No absolute 32-bit addressing anywhere.
    assert!(obj.relocations().iter().all(|r| !matches!(r.kind, RelocKind::Abs32 | RelocKind::Abs32S)));
    // Absolute pointers only in (writable) data sections, never in text.
    for r in obj.relocations() {
        if r.kind == RelocKind::Abs64 {
            let sec = &obj.section(r.section).name;
            assert!(sec == ".data.rel.ro" || sec == ".data", "Abs64 in {sec}");
        }
    }
}

#[test]
fn pic_got_load_is_a_rip_relative_mov() {
    let obj = compile(RelocModel::Pic);
    let text = obj.sections().iter().find(|s| s.name == ".text").unwrap();
    let r = obj
        .relocations()
        .iter()
        .find(|r| obj.symbol(r.symbol).name == "counter")
        .expect("a reloc against counter");
    assert_eq!(r.addend, -4, "GOTPCREL is relative to the end of the disp32 field");
    let at = r.offset as usize;
    // REX.W (+R) 8B /r with ModRM mod=00 rm=101 (RIP-relative).
    let (rex, op, modrm) = (text.bytes[at - 3], text.bytes[at - 2], text.bytes[at - 1]);
    assert_eq!(rex & 0xF8, 0x48, "REX.W");
    assert_eq!(op, 0x8B, "mov r64, r/m64");
    assert_eq!(modrm & 0xC7, 0x05, "RIP-relative");
}

#[test]
fn pie_binds_module_definitions_locally() {
    let obj = compile(RelocModel::Pie);
    use RelocKind::{GotPcRel, Pc32, Plt32};
    assert_eq!(text_relocs(&obj, "counter"), [Pc32]);
    assert_eq!(text_relocs(&obj, "prot"), [Pc32]);
    assert_eq!(text_relocs(&obj, "cb"), [Plt32, Pc32]);
    // Only what is defined elsewhere goes through the GOT.
    assert_eq!(text_relocs(&obj, "ext"), [GotPcRel]);
    assert_eq!(text_relocs(&obj, "strlen"), [Plt32, GotPcRel]);
}

#[test]
fn static_model_is_unchanged() {
    let obj = compile(RelocModel::Static);
    assert!(obj.relocations().iter().all(|r| r.kind != RelocKind::GotPcRel));
    let names: Vec<&str> = obj.sections().iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&".rodata") && !names.contains(&".data.rel.ro"), "{names:?}");
    assert!(!names.contains(&".note.GNU-stack"), "{names:?}");
    let table = obj.symbol(obj.symbol_id("table").unwrap());
    let crate::mc::object::SymbolValue::Defined { section, .. } = table.value else { panic!() };
    assert_eq!(obj.section(section).name, ".rodata");
}

#[test]
fn pic_moves_pointer_constants_to_data_rel_ro() {
    let obj = compile(RelocModel::Pic);
    let section_of = |n: &str| {
        let s = obj.symbol(obj.symbol_id(n).unwrap());
        let crate::mc::object::SymbolValue::Defined { section, .. } = s.value else { panic!("{n}") };
        obj.section(section).name.clone()
    };
    assert_eq!(section_of("table"), ".data.rel.ro");
    assert_eq!(section_of("plain"), ".rodata", "pointer-free constants stay read-only");
    assert!(obj.sections().iter().any(|s| s.name == ".note.GNU-stack" && s.bytes.is_empty()));
    assert_eq!(obj.symbol(obj.symbol_id("secret").unwrap()).visibility, SymbolVisibility::Hidden);
    assert_eq!(obj.symbol(obj.symbol_id("helper").unwrap()).visibility, SymbolVisibility::Hidden);
}

#[test]
fn elf_st_other_carries_visibility() {
    let obj = compile(RelocModel::Pic);
    let elf = crate::mc::elf::write(&obj);
    // Walk .symtab: find it via the section headers.
    let u16_at = |o: usize| u16::from_le_bytes([elf[o], elf[o + 1]]) as usize;
    let u32_at = |o: usize| u32::from_le_bytes(elf[o..o + 4].try_into().unwrap()) as usize;
    let u64_at = |o: usize| u64::from_le_bytes(elf[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shnum) = (u64_at(0x28), u16_at(0x3C));
    let sh = |i: usize| shoff + i * 64;
    let symtab = (0..shnum).find(|&i| u32_at(sh(i) + 4) == 2).expect(".symtab");
    let strtab = u32_at(sh(symtab) + 40);
    let (sym_off, sym_size) = (u64_at(sh(symtab) + 24), u64_at(sh(symtab) + 32));
    let str_off = u64_at(sh(strtab) + 24);
    let mut vis = std::collections::HashMap::new();
    for k in 0..sym_size / 24 {
        let e = sym_off + k * 24;
        let name_at = str_off + u32_at(e);
        let end = elf[name_at..].iter().position(|&b| b == 0).unwrap();
        let name = String::from_utf8(elf[name_at..name_at + end].to_vec()).unwrap();
        vis.insert(name, elf[e + 5] & 3);
    }
    assert_eq!(vis["secret"], 2, "STV_HIDDEN");
    assert_eq!(vis["helper"], 2, "STV_HIDDEN on an undefined reference");
    assert_eq!(vis["prot"], 3, "STV_PROTECTED");
    assert_eq!(vis["counter"], 0, "STV_DEFAULT");
}

#[test]
fn other_targets_reject_pic_with_a_clear_error() {
    let mut syms = StrInterner::new();
    let m = parse_module("module \"m\"\nfunc @f() -> void {\nentry ^0:\n  ret\n}\n", FileId::new(0), &mut syms)
        .unwrap();
    let pic = CodegenOptions::default().with_pic(true);
    let arch = TargetArch::Riscv64;
    let err = compile_module_for(arch, &m, &syms, &pic).unwrap_err();
    assert_eq!(err, CodegenError::UnsupportedRelocModel { arch, model: RelocModel::Pic });
    assert!(err.to_string().contains("position-independent"), "{err}");
    // Position-dependent code still compiles.
    assert!(compile_module_for(arch, &m, &syms, &CodegenOptions::default()).is_ok());
    // x86-64 and AArch64 generate it.
    assert!(compile_module_for(TargetArch::X86_64, &m, &syms, &pic).is_ok());
    assert!(compile_module_for(TargetArch::AArch64, &m, &syms, &pic).is_ok());
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::target::riscv::compile_module_with(&m, &syms, &pic)
    }));
    assert!(r.is_err(), "the infallible entry point refuses PIC too");
}
