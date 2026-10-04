//! The object readers and listings: every format LF writes reads back with
//! its architecture, code sections, labels and relocations.

use super::corpus::{FLOATS, INTS};
use super::{Rng, compile, object_file};
use crate::mc::disasm::listing::{ListingOptions, list};
use crate::mc::disasm::objfile::{self, Binary, FileFormat, LabelKind};
use crate::target::{ObjectFormat, TargetArch};

fn text<'a>(bin: &'a Binary, name: &str) -> &'a objfile::CodeSection {
    bin.sections.iter().find(|s| s.name == name).unwrap_or_else(|| {
        panic!("no section {name} in {:?}", bin.sections.iter().map(|s| &s.name).collect::<Vec<_>>())
    })
}

fn has_label(sec: &objfile::CodeSection, name: &str) -> bool {
    sec.labels.iter().any(|l| l.name == name)
}

fn has_reloc(sec: &objfile::CodeSection, kind: &str, symbol: &str) -> bool {
    sec.relocs.iter().any(|r| r.kind == kind && r.symbol == symbol)
}

#[test]
fn elf_x86_64() {
    let obj = compile(TargetArch::X86_64, INTS);
    let file = object_file(TargetArch::X86_64, &obj, ObjectFormat::Elf);
    let bin = objfile::read(&file).unwrap();
    assert_eq!(bin.format, FileFormat::Elf);
    assert_eq!(bin.arch, Some(TargetArch::X86_64));
    assert_eq!(bin.description, "elf64-x86-64");
    let t = text(&bin, ".text");
    assert!(t.executable);
    for f in ["arith64", "sum", "dense", "via_ref", "twice"] {
        assert!(has_label(t, f), "label {f}: {:?}", t.labels);
    }
    assert!(t.labels.iter().any(|l| l.name == "sum" && l.kind == LabelKind::Function));
    let call = t.relocs.iter().find(|r| r.symbol == "ext").expect("a relocation against ext");
    assert_eq!(call.kind, "R_X86_64_PLT32");
    assert_eq!(call.addend, Some(-4));
    assert_eq!(call.note(), "R_X86_64_PLT32 ext-0x4");
    assert!(!text(&bin, ".data").executable);
    // The listing labels functions and notes relocations inline.
    let out = list(&bin, TargetArch::X86_64, "ints.o", &ListingOptions::new());
    assert!(out.contains("file format elf64-x86-64"), "{out}");
    assert!(out.contains("Disassembly of section .text:"), "{out}");
    assert!(out.contains("<sum>:"), "{out}");
    assert!(out.contains("# R_X86_64_PLT32 ext-0x4"), "{out}");
    assert!(!out.contains("Disassembly of section .data:"), "{out}");
    let all = list(&bin, TargetArch::X86_64, "ints.o", &ListingOptions { all_sections: true, ..ListingOptions::new() });
    assert!(all.contains("Disassembly of section .data:"), "{all}");
}

#[test]
fn coff_and_macho() {
    for (arch, format, desc, prefix, call) in [
        (TargetArch::X86_64, ObjectFormat::Coff, "coff-x86-64", "", "IMAGE_REL_AMD64_REL32"),
        (TargetArch::AArch64, ObjectFormat::Coff, "coff-ARM64", "", "IMAGE_REL_ARM64_BRANCH26"),
        (TargetArch::X86_64, ObjectFormat::MachO, "mach-o x86-64", "_", "X86_64_RELOC_BRANCH"),
        (TargetArch::AArch64, ObjectFormat::MachO, "mach-o arm64", "_", "ARM64_RELOC_BRANCH26"),
    ] {
        // (The Mach-O writer adds the `_` prefix itself.)
        let obj = compile(arch, INTS);
        let file = object_file(arch, &obj, format);
        let bin = objfile::read(&file).unwrap();
        assert_eq!(bin.arch, Some(arch), "{desc}");
        assert_eq!(bin.description, desc);
        let code: Vec<_> = bin.sections.iter().filter(|s| s.executable).collect();
        assert_eq!(code.len(), 1, "{desc}: {:?}", bin.sections.iter().map(|s| &s.name).collect::<Vec<_>>());
        let t = code[0];
        assert!(has_label(t, &format!("{prefix}sum")), "{desc}: {:?}", t.labels);
        assert!(has_reloc(t, call, &format!("{prefix}ext")), "{desc}: {:?}", t.relocs);
    }
}

#[test]
fn elf32_thumb_and_avr() {
    let obj = compile(TargetArch::Thumb, INTS);
    let file = object_file(TargetArch::Thumb, &obj, ObjectFormat::Elf);
    let bin = objfile::read(&file).unwrap();
    assert_eq!((bin.arch, bin.description.as_str()), (Some(TargetArch::Thumb), "elf32-littlearm"));
    let t = text(&bin, ".text");
    // Function labels sit on the instruction (the Thumb bit cleared).
    assert!(t.labels.iter().filter(|l| l.kind == LabelKind::Function).all(|l| l.addr % 2 == 0));
    assert!(has_reloc(t, "R_ARM_THM_CALL", "ext"), "{:?}", t.relocs);

    let obj = compile(TargetArch::Avr, INTS);
    let file = object_file(TargetArch::Avr, &obj, ObjectFormat::Elf);
    let bin = objfile::read(&file).unwrap();
    assert_eq!((bin.arch, bin.description.as_str()), (Some(TargetArch::Avr), "elf32-avr"));
    let t = text(&bin, ".text");
    assert!(has_label(t, "sum"));
    assert!(has_reloc(t, "R_AVR_CALL", "ext"), "{:?}", t.relocs);
}

#[test]
fn elf64_riscv_and_aarch64() {
    for (arch, desc) in [(TargetArch::Riscv64, "elf64-littleriscv"), (TargetArch::AArch64, "elf64-littleaarch64")] {
        let obj = compile(arch, INTS);
        let file = object_file(arch, &obj, ObjectFormat::Elf);
        let bin = objfile::read(&file).unwrap();
        assert_eq!((bin.arch, bin.description.as_str()), (Some(arch), desc));
        assert!(has_label(text(&bin, ".text"), "sum"));
    }
}

#[test]
fn wasm_object() {
    let obj = compile(TargetArch::Wasm32, INTS);
    let file = object_file(TargetArch::Wasm32, &obj, ObjectFormat::Wasm);
    let bin = objfile::read(&file).unwrap();
    assert_eq!(bin.format, FileFormat::Wasm);
    assert_eq!(bin.arch, Some(TargetArch::Wasm32));
    let code = text(&bin, "CODE");
    assert!(code.executable);
    // One region (expression) per defined function, each labeled.
    assert_eq!(code.regions.len(), code.labels.len());
    assert!(has_label(code, "sum") && has_label(code, "via_ref"), "{:?}", code.labels);
    assert!(code.regions.iter().all(|r| r.start < r.end && r.end <= code.bytes.len() as u64));
    assert!(has_reloc(code, "R_WASM_FUNCTION_INDEX_LEB", "ext"), "{:?}", code.relocs);
}

#[test]
fn lfo_and_raw() {
    let obj = compile(TargetArch::AArch64, INTS);
    let bytes = crate::mc::lfo::encode(&obj);
    let bin = objfile::read(&bytes).unwrap();
    assert_eq!(bin.format, FileFormat::Lfo);
    // Guessed from the AArch64 relocation kinds.
    assert_eq!(bin.arch, Some(TargetArch::AArch64));
    let t = text(&bin, ".text");
    assert!(has_label(t, "sum"));
    assert!(t.relocs.iter().any(|r| r.kind == "Aarch64Call26" && r.symbol == "ext"));

    let raw = objfile::raw(&[0x90, 0xc3], TargetArch::X86_64, 0x1000);
    assert_eq!(raw.sections[0].addr, 0x1000);
    assert!(raw.sections[0].executable);
}

#[test]
fn linked_executable_has_addresses() {
    let obj = compile(TargetArch::X86_64, FLOATS);
    let opts = crate::link::ImageOptions { entry: "fd".to_owned(), debug: true, ..crate::link::ImageOptions::default() };
    let image = crate::link::link_executable(vec![obj], &opts).expect("link");
    let bin = objfile::read(&image).unwrap();
    let t = bin.sections.iter().find(|s| s.executable).expect("a code section");
    assert!(t.addr >= 0x40_0000, "{:#x}", t.addr);
    let fd = t.labels.iter().find(|l| l.name == "fd").expect("fd");
    assert!(fd.addr >= t.addr && fd.addr < t.end());
    // Without the section table, the executable segment is still found.
    let plain = crate::link::link_executable(vec![compile(TargetArch::X86_64, FLOATS)], &crate::link::ImageOptions {
        entry: "fd".to_owned(),
        ..crate::link::ImageOptions::default()
    })
    .expect("link");
    let bin = objfile::read(&plain).unwrap();
    assert!(bin.sections.iter().any(|s| s.executable && s.addr >= 0x40_0000));
}

#[test]
fn aarch64_pic_got_relocations() {
    let (m, syms) = super::parse_for(TargetArch::AArch64, INTS);
    let opts = crate::codegen::CodegenOptions::default().with_pic(true);
    let obj = crate::target::compile_module_for(TargetArch::AArch64, &m, &syms, &opts).expect("PIC compile").object;
    let file = object_file(TargetArch::AArch64, &obj, ObjectFormat::Elf);
    let bin = objfile::read(&file).unwrap();
    let t = text(&bin, ".text");
    assert!(t.relocs.iter().any(|r| r.kind == "R_AARCH64_ADR_GOT_PAGE"), "{:?}", t.relocs);
    assert!(t.relocs.iter().any(|r| r.kind == "R_AARCH64_LD64_GOT_LO12_NC"), "{:?}", t.relocs);
    let out = list(&bin, TargetArch::AArch64, "pic.o", &ListingOptions::new());
    assert!(out.contains("// R_AARCH64_ADR_GOT_PAGE"), "{out}");
    let lfo = objfile::read(&crate::mc::lfo::encode(&obj)).unwrap();
    assert_eq!(lfo.arch, Some(TargetArch::AArch64));
    // The GOT sequences decode as llvm-objdump does.
    let opts = crate::mc::disasm::Options::default();
    if let Some(r) = super::differential(TargetArch::AArch64, &file, &[], &opts, &|s| s) {
        super::assert_clean("aarch64 PIC ELF", &r);
    }
}

#[test]
fn unrecognized_and_corrupt_files() {
    assert!(objfile::read(b"hello world").is_err());
    assert!(objfile::read(b"").is_err());
    // Every truncation and random corruption of real objects reads without
    // panicking (an error, or whatever survived).
    let mut files = Vec::new();
    for (arch, format) in [
        (TargetArch::X86_64, ObjectFormat::Elf),
        (TargetArch::X86_64, ObjectFormat::Coff),
        (TargetArch::AArch64, ObjectFormat::MachO),
        (TargetArch::Thumb, ObjectFormat::Elf),
        (TargetArch::Wasm32, ObjectFormat::Wasm),
    ] {
        files.push(object_file(arch, &compile(arch, INTS), format));
    }
    files.push(crate::mc::lfo::encode(&compile(TargetArch::X86_64, INTS)));
    let mut rng = Rng(42);
    for f in &files {
        for cut in (0..f.len()).step_by(7) {
            let _ = objfile::read(&f[..cut]);
        }
        for _ in 0..300 {
            let mut g = f.clone();
            for _ in 0..1 + rng.below(8) {
                let at = rng.below(g.len() as u64) as usize;
                g[at] = rng.next() as u8;
            }
            if let Ok(bin) = objfile::read(&g) {
                let arch = bin.arch.unwrap_or(TargetArch::X86_64);
                let _ = list(&bin, arch, "fuzz", &ListingOptions::new());
            }
        }
    }
}

#[test]
fn start_and_stop() {
    let obj = compile(TargetArch::X86_64, INTS);
    let file = object_file(TargetArch::X86_64, &obj, ObjectFormat::Elf);
    let bin = objfile::read(&file).unwrap();
    let t = text(&bin, ".text");
    let sum = t.labels.iter().find(|l| l.name == "sum").unwrap().addr;
    let opts = ListingOptions { start: Some(sum), stop: Some(sum + 1), ..ListingOptions::new() };
    let out = list(&bin, TargetArch::X86_64, "x.o", &opts);
    assert!(out.contains("<sum>:"), "{out}");
    let insts = out.lines().filter(|l| l.starts_with(' ') && l.contains(": ")).count();
    assert_eq!(insts, 1, "exactly one instruction: {out}");
}
