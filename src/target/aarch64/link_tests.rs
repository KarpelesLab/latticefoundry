//! ELF objects, `qld` links, position-independent code, shared libraries and
//! DWARF on AArch64 Linux.
//!
//! - The relocatable object is ELF64 `EM_AARCH64` with `RELA` relocations:
//!   checked field by field here, and by `llvm-readobj` when it is installed.
//! - `qld` links it into a static executable (with the `_start` of
//!   `super::link`) that the A64 emulator runs.
//! - Under `RelocModel::Pic` a preemptible symbol is reached through the GOT
//!   (`adrp`+`ldr`, `R_AARCH64_ADR_GOT_PAGE` + `LD64_GOT_LO12_NC`) and a local
//!   one directly. The shared library `qld` builds from it has no text
//!   relocation; the tests load it into the emulator with a minimal dynamic
//!   loader (applying `R_AARCH64_RELATIVE`, `GLOB_DAT`, `JUMP_SLOT` and
//!   `ABS64`), call into it through its PLT, and interpose a symbol to prove
//!   the code really goes through the GOT.
//! - `-g` objects carry line tables `llvm-dwarfdump` reads.

use std::path::{Path, PathBuf};

use super::emu::{self, Emu, Stop};
use crate::codegen::{CodegenOptions, RelocModel};
use crate::ir::Module;
use crate::mc::elf::ElfTarget;
use crate::mc::object::{ObjectModule, RelocKind};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms).unwrap_or_else(|e| panic!("parse: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    (m, syms)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-a64-link-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run an LLVM tool (from `PATH`), `None` when it is not installed.
fn tool(name: &str, args: &[&str], file: &Path) -> Option<String> {
    let out = std::process::Command::new(name).args(args).arg(file).output().ok()?;
    assert!(out.status.success(), "{name} rejected {}: {}", file.display(), String::from_utf8_lossy(&out.stderr));
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// A program touching every relocation the backend emits: calls, global
/// data, a pointer table in data, a function pointer, an internal function.
const PROG: &str = r#"
module "prog"
global @counter : i64 = i64 5
global hidden @secret : i64 = i64 7
global constant @table : [2 x ptr] = [2 x ptr] (ptr @counter, ptr @twice)

func internal @twice(i64) -> i64 {
entry ^0(%x: i64):
  %r = add %x, %x : i64
  ret %r
}

func @ext(i64) -> i64

func @entry(i64) -> i64 {
entry ^0(%x: i64):
  %pc = load @table align 8 : ptr
  %c = load %pc align 8 : i64
  %tf = ptr_add @table, i64 8 : ptr
  %f = load %tf align 8 : ptr
  %t = call %f(%x) : i64
  %e = call @ext(%x) : i64
  %s = load @secret align 8 : i64
  %one = icmp eq %x, i64 3 : i1
  %g = select %one, @ext, @twice : ptr
  %h = call %g(%c) : i64
  store i64 99, @counter align 8 : i64
  %c2 = load @counter align 8 : i64
  %a0 = add %c, %t : i64
  %a1 = add %a0, %e : i64
  %a2 = add %a1, %s : i64
  %a3 = add %a2, %h : i64
  %a4 = add %a3, %c2 : i64
  ret %a4
}
"#;

/// `entry(3)` with `ext(x) = x + 100` and `counter` starting at `c`:
/// `c + 6 + 103 + 7 + (c + 100) + 99`.
fn want(c: u64) -> u64 {
    c + 6 + 103 + 7 + (c + 100) + 99
}

/// `ext` and a `main` calling `entry(3)`, for the static executables.
const MAIN: &str = r#"
module "main"
func @entry(i64) -> i64
func @ext(i64) -> i64 {
entry ^0(%x: i64):
  %r = add %x, i64 100 : i64
  ret %r
}
func @main() -> i32 {
entry ^0:
  %r = call @entry(i64 3) : i64
  %t = trunc %r : i32
  ret %t
}
"#;

fn compile(src: &str, model: RelocModel) -> ObjectModule {
    let (m, syms) = parse(src);
    super::compile_module_with(&m, &syms, &CodegenOptions::default().with_reloc_model(model)).object
}

/// The `RELA` entries of section `name` of an ELF64 object: (offset, type,
/// symbol name, addend).
fn rela(elf: &[u8], name: &str) -> Vec<(u64, u32, String, i64)> {
    let shoff = u64_at(elf, 40) as usize;
    let shnum = usize::from(u16_at(elf, 60));
    let sh = |i: usize| shoff + i * 64;
    let shstr = u64_at(elf, sh(usize::from(u16_at(elf, 62))) + 24) as usize;
    let cstr = |at: usize| {
        let end = elf[at..].iter().position(|&b| b == 0).unwrap();
        String::from_utf8(elf[at..at + end].to_vec()).unwrap()
    };
    let sec = (0..shnum).find(|&i| cstr(shstr + u32_at(elf, sh(i)) as usize) == name).expect("section");
    let (off, size) = (u64_at(elf, sh(sec) + 24) as usize, u64_at(elf, sh(sec) + 32) as usize);
    let symtab = u32_at(elf, sh(sec) + 40) as usize;
    let symoff = u64_at(elf, sh(symtab) + 24) as usize;
    let strtab = u64_at(elf, sh(u32_at(elf, sh(symtab) + 40) as usize) + 24) as usize;
    (0..size / 24)
        .map(|k| {
            let e = off + 24 * k;
            let info = u64_at(elf, e + 8);
            let sym = symoff + 24 * (info >> 32) as usize;
            (u64_at(elf, e), info as u32, cstr(strtab + u32_at(elf, sym) as usize), u64_at(elf, e + 16) as i64)
        })
        .collect()
}

#[test]
fn object_is_elf64_aarch64_with_rela_relocations() {
    let obj = compile(PROG, RelocModel::Static);
    let elf = crate::mc::elf::write_with(&obj, &ElfTarget::AARCH64).unwrap();
    assert_eq!(&elf[..6], b"\x7fELF\x02\x01", "ELF64, little-endian");
    assert_eq!((u16_at(&elf, 16), u16_at(&elf, 18), u32_at(&elf, 48)), (1, 183, 0), "ET_REL, EM_AARCH64, flags");
    let text = rela(&elf, ".rela.text");
    let ty = |sym: &str| -> Vec<u32> { text.iter().filter(|r| r.2 == sym).map(|r| r.1).collect() };
    assert_eq!(ty("ext"), [283, 275, 277], "bl (CALL26), then its address (ADR_PREL_PG_HI21 + ADD_ABS_LO12_NC)");
    assert_eq!(ty("counter"), [275, 277], "one address per block, cached");
    assert_eq!(ty("twice"), [275, 277]);
    let data = rela(&elf, ".rela.rodata");
    assert_eq!(data.iter().map(|r| (r.0, r.1, r.2.as_str())).collect::<Vec<_>>(), [(0, 257, "counter"), (8, 257, "twice")]);
    // The same through `write_object` for an aarch64-linux triple.
    let t = crate::target::Triple::new(crate::target::TargetArch::AArch64, crate::target::TargetOs::Linux);
    assert_eq!(crate::mc::write_object(&obj, t).unwrap(), elf);

    let dir = scratch("obj");
    let path = dir.join("prog.o");
    std::fs::write(&path, &elf).unwrap();
    if let Some(s) = tool("llvm-readobj", &["-h", "-r"], &path) {
        for want in ["EM_AARCH64", "R_AARCH64_CALL26 ext", "R_AARCH64_ADR_PREL_PG_HI21 counter",
                     "R_AARCH64_ADD_ABS_LO12_NC counter", "R_AARCH64_ABS64 twice"] {
            assert!(s.contains(want), "llvm-readobj lacks `{want}`:\n{s}");
        }
    }
    if let Some(s) = tool("llvm-objdump", &["-d", "-r"], &path) {
        assert!(s.contains("bl\t") && s.contains("adrp\t") && s.contains("blr\t"), "{s}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Link `objs` into a static executable with qld and run it on the emulator.
fn link_and_run(objs: Vec<ObjectModule>, tag: &str) -> (u64, Vec<u8>) {
    let dir = scratch(tag);
    let exe = dir.join("a.out");
    super::link::link_executable(objs, "main", &[], &exe).expect("qld links");
    let elf = std::fs::read(&exe).unwrap();
    // A static ELF64 AArch64 executable entered at `_start`.
    assert_eq!((u16_at(&elf, 16), u16_at(&elf, 18)), (2, 183), "ET_EXEC, EM_AARCH64");
    if let Some(s) = tool("llvm-objdump", &["-d"], &exe) {
        assert!(s.contains("<_start>:") && s.contains("svc\t#0"), "{s}");
    }
    let out = emu::run_executable(&elf).unwrap_or_else(|e| panic!("{tag}: {e}"));
    let _ = std::fs::remove_dir_all(&dir);
    out
}

#[test]
fn static_executable_runs() {
    let (code, _) = link_and_run(vec![compile(PROG, RelocModel::Static), compile(MAIN, RelocModel::Static)], "static");
    assert_eq!(code & 0xFFFF_FFFF, want(5), "exit status: entry(3)");
}

/// Under PIC the preemptible symbols (`counter`, `ext`, and `entry` were it
/// called) go through the GOT, the local ones (`secret`, `twice`) do not,
/// and calls stay `CALL26`.
#[test]
fn pic_reaches_preemptible_symbols_through_the_got() {
    let obj = compile(PROG, RelocModel::Pic);
    let elf = crate::mc::elf::write_with(&obj, &ElfTarget::AARCH64).unwrap();
    let text = rela(&elf, ".rela.text");
    let ty = |sym: &str| -> Vec<u32> { text.iter().filter(|r| r.2 == sym).map(|r| r.1).collect() };
    assert_eq!(ty("counter"), [311, 312], "GOT page + LD64_GOT_LO12_NC");
    assert_eq!(ty("ext"), [283, 311, 312], "the call stays CALL26; the address is a GOT load");
    assert_eq!(ty("secret"), [275, 277], "hidden: direct");
    assert_eq!(ty("twice"), [275, 277], "internal: direct");
    // The pointer table moves to `.data.rel.ro`, and the object asks for a
    // non-executable stack.
    assert!(obj.sections().iter().any(|s| s.name == ".data.rel.ro"));
    assert!(obj.sections().iter().any(|s| s.name == ".note.GNU-stack"));
    assert!(!obj.relocations().iter().any(|r| r.kind == RelocKind::Aarch64AdrPrelPgHi21
        && obj.symbol(r.symbol).name == "counter"));
    // The GOT load encodes as `adrp`+`ldr` (llvm-mc agrees).
    let word = |at: u64| -> u32 {
        let t = obj.sections().iter().find(|s| s.name == ".text").unwrap();
        u32::from_le_bytes(t.bytes[at as usize..at as usize + 4].try_into().unwrap())
    };
    let r = text.iter().find(|r| r.2 == "counter" && r.1 == 312).unwrap();
    let ldr = word(r.0);
    assert_eq!(ldr & 0xFFC0_0000, 0xF940_0000, "ldr Xt, [Xn, #0]: {ldr:#010x}");
    // PIE: definitions bind locally (direct), declarations do not.
    let pie = crate::mc::elf::write_with(&compile(PROG, RelocModel::Pie), &ElfTarget::AARCH64).unwrap();
    let text = rela(&pie, ".rela.text");
    assert!(text.iter().filter(|r| r.2 == "counter").all(|r| r.1 == 275 || r.1 == 277));
    assert!(text.iter().any(|r| r.2 == "ext" && r.1 == 311));
}

#[test]
fn pic_executable_runs() {
    let (code, _) = link_and_run(vec![compile(PROG, RelocModel::Pic), compile(MAIN, RelocModel::Pic)], "pic");
    assert_eq!(code & 0xFFFF_FFFF, want(5));
}

/// The dynamic section, symbols and relocations of a shared object.
struct Dso<'a> {
    elf: &'a [u8],
    dynamic: Vec<(u64, u64)>,
}

impl Dso<'_> {
    fn new(elf: &[u8]) -> Dso<'_> {
        let (phoff, phnum) = (u64_at(elf, 32) as usize, usize::from(u16_at(elf, 56)));
        let mut dynamic = Vec::new();
        for i in 0..phnum {
            let ph = phoff + 56 * i;
            if u32_at(elf, ph) == 2 {
                let (off, size) = (u64_at(elf, ph + 8) as usize, u64_at(elf, ph + 32) as usize);
                dynamic = (0..size / 16).map(|k| (u64_at(elf, off + 16 * k), u64_at(elf, off + 16 * k + 8))).collect();
            }
        }
        Dso { elf, dynamic }
    }

    fn tag(&self, t: u64) -> Option<u64> {
        self.dynamic.iter().find(|e| e.0 == t).map(|e| e.1)
    }

    /// The file offset of virtual address `va` (through the `PT_LOAD`s).
    fn off(&self, va: u64) -> usize {
        let (phoff, phnum) = (u64_at(self.elf, 32) as usize, usize::from(u16_at(self.elf, 56)));
        for i in 0..phnum {
            let ph = phoff + 56 * i;
            let (off, vaddr, filesz) = (u64_at(self.elf, ph + 8), u64_at(self.elf, ph + 16), u64_at(self.elf, ph + 32));
            if u32_at(self.elf, ph) == 1 && va >= vaddr && va < vaddr + filesz {
                return (va - vaddr + off) as usize;
            }
        }
        panic!("{va:#x} is not in a loadable segment")
    }

    /// Dynamic symbol `k`: (name, value, defined).
    fn sym(&self, k: u64) -> (String, u64, bool) {
        let e = self.off(self.tag(6).unwrap()) + 24 * k as usize;
        let strtab = self.off(self.tag(5).unwrap());
        let at = strtab + u32_at(self.elf, e) as usize;
        let end = self.elf[at..].iter().position(|&b| b == 0).unwrap();
        let name = String::from_utf8(self.elf[at..at + end].to_vec()).unwrap();
        (name, u64_at(self.elf, e + 8), u16_at(self.elf, e + 6) != 0)
    }

    /// Every dynamic relocation: (offset, type, symbol index, addend).
    fn relocs(&self) -> Vec<(u64, u32, u64, i64)> {
        let mut out = Vec::new();
        for (at, size) in [(7, 8), (23, 2)] {
            if let (Some(a), Some(n)) = (self.tag(at), self.tag(size)) {
                let o = self.off(a);
                for k in 0..(n / 24) as usize {
                    let e = o + 24 * k;
                    let info = u64_at(self.elf, e + 8);
                    out.push((u64_at(self.elf, e), info as u32, info >> 32, u64_at(self.elf, e + 16) as i64));
                }
            }
        }
        out
    }

    /// Whether `va` lies in an executable segment.
    fn in_text(&self, va: u64) -> bool {
        let (phoff, phnum) = (u64_at(self.elf, 32) as usize, usize::from(u16_at(self.elf, 56)));
        (0..phnum).any(|i| {
            let ph = phoff + 56 * i;
            let (vaddr, memsz) = (u64_at(self.elf, ph + 16), u64_at(self.elf, ph + 40));
            u32_at(self.elf, ph) == 1 && u32_at(self.elf, ph + 4) & 1 != 0 && va >= vaddr && va < vaddr + memsz
        })
    }
}

/// Build the shared library of [`PROG`] with qld.
fn shared_library(dir: &Path) -> Vec<u8> {
    let so = dir.join("libprog.so");
    super::link::link_shared(&[compile(PROG, RelocModel::Pic)], Some("libprog.so"), &[], &so).expect("qld links");
    std::fs::read(so).unwrap()
}

#[test]
fn shared_library_has_no_text_relocations() {
    let dir = scratch("so");
    let so = shared_library(&dir);
    assert_eq!((u16_at(&so, 16), u16_at(&so, 18)), (3, 183), "ET_DYN, EM_AARCH64");
    let dso = Dso::new(&so);
    assert!(dso.tag(22).is_none(), "no DT_TEXTREL");
    assert_eq!(dso.tag(30).unwrap_or(0) & 4, 0, "no DF_TEXTREL");
    let relocs = dso.relocs();
    assert!(!relocs.is_empty());
    for &(off, ty, _, _) in &relocs {
        assert!(matches!(ty, 257 | 1025 | 1026 | 1027), "dynamic relocation type {ty}");
        assert!(!dso.in_text(off), "relocation at {off:#x} patches code");
    }
    let names: Vec<(String, u32)> = relocs.iter().map(|&(_, ty, s, _)| (dso.sym(s).0, ty)).collect();
    assert!(names.contains(&("counter".into(), 1025)), "counter's GOT entry: {names:?}");
    assert!(names.contains(&("ext".into(), 1025)) || names.contains(&("ext".into(), 1026)), "{names:?}");
    assert!(names.iter().any(|(_, ty)| *ty == 1027), "the table's local entry is RELATIVE: {names:?}");
    // `entry` and `counter` are exported, `secret` (hidden) and `twice`
    // (internal) are not.
    let exported: Vec<String> = (1..64)
        .map_while(|k| (dso.off(dso.tag(6).unwrap()) + 24 * k < dso.off(dso.tag(5).unwrap())).then(|| dso.sym(k as u64)))
        .filter(|s| s.2)
        .map(|s| s.0)
        .collect();
    assert!(exported.contains(&"entry".into()) && exported.contains(&"counter".into()), "{exported:?}");
    assert!(!exported.contains(&"secret".into()) && !exported.contains(&"twice".into()), "{exported:?}");
    let path = dir.join("libprog.so");
    if let Some(s) = tool("llvm-readobj", &["--dynamic-table", "--dyn-relocations"], &path) {
        assert!(s.contains("SONAME") && s.contains("libprog.so"), "{s}");
        assert!(!s.contains("TEXTREL"), "{s}");
        assert!(s.contains("R_AARCH64_GLOB_DAT") && s.contains("R_AARCH64_RELATIVE"), "{s}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Load the library at `base` into `emu` (a minimal dynamic loader), with
/// `imports` resolving the undefined (or interposed) symbols, and return the
/// address of `entry`.
fn load(emu: &mut Emu, so: &[u8], base: u64, imports: &[(&str, u64)]) -> u64 {
    let dso = Dso::new(so);
    let (phoff, phnum) = (u64_at(so, 32) as usize, usize::from(u16_at(so, 56)));
    for i in 0..phnum {
        let ph = phoff + 56 * i;
        if u32_at(so, ph) == 1 {
            let (off, vaddr, filesz, memsz) =
                (u64_at(so, ph + 8) as usize, u64_at(so, ph + 16), u64_at(so, ph + 32) as usize, u64_at(so, ph + 40));
            emu.map(base + vaddr, memsz);
            emu.poke(base + vaddr, &so[off..off + filesz]);
        }
    }
    let resolve = |s: u64| -> u64 {
        let (name, value, defined) = dso.sym(s);
        if let Some(&(_, a)) = imports.iter().find(|i| i.0 == name) {
            return a;
        }
        assert!(defined, "unresolved symbol {name}");
        base + value
    };
    for (off, ty, s, addend) in dso.relocs() {
        let v = match ty {
            1027 => base.wrapping_add(addend as u64),
            257 | 1025 | 1026 => resolve(s).wrapping_add(addend as u64),
            t => panic!("relocation type {t}"),
        };
        emu.poke(base + off, &v.to_le_bytes());
    }
    let entry = (1..).map(|k| dso.sym(k)).find(|s| s.0 == "entry").unwrap();
    base + entry.1
}

#[test]
fn shared_library_runs_and_its_symbols_can_be_interposed() {
    let dir = scratch("so-run");
    let so = shared_library(&dir);
    // `ext(x) = x + 100`, outside the library.
    let ext = [super::encode::add_imm(1, 0, 0, 100), super::encode::ret(30)];
    let run = |imports: &[(&str, u64)], counter: Option<u64>| {
        let mut emu = Emu::new();
        emu.map(0x2000_0000, 0x2000);
        emu.poke(0x2000_0000, &ext.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>());
        if let Some(c) = counter {
            emu.poke(0x2000_1000, &c.to_le_bytes());
        }
        emu.map_stack(0x7000_0000_0000, 1 << 20, true);
        let entry = load(&mut emu, &so, 0x1000_0000, imports);
        assert_eq!(emu.call(entry, &[3], 1_000_000).unwrap(), Stop::Returned);
        (emu.x[0], emu.peek(0x2000_1000, 8).unwrap() as u64)
    };
    // The library's own `counter`.
    assert_eq!(run(&[("ext", 0x2000_0000)], None).0, want(5));
    // `counter` interposed (as an executable's copy would be): the
    // library's GOT entry and data relocation both see the interposer, and
    // its store lands there.
    let (r, interposed) = run(&[("ext", 0x2000_0000), ("counter", 0x2000_1000)], Some(1000));
    assert_eq!(r, want(1000));
    assert_eq!(interposed, 99, "the library stored through the GOT");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn debug_object_has_line_tables() {
    let (m, syms) = parse(PROG);
    let source = super::DebugSource { file_name: "prog.lf".into(), comp_dir: "/lf".into() };
    let obj = super::compile_module_debug_with(&m, &syms, &source, &CodegenOptions::default()).object;
    for name in [".debug_abbrev", ".debug_info", ".debug_str", ".debug_line"] {
        assert!(obj.sections().iter().any(|s| s.name == name), "{name}");
    }
    assert!(obj.relocations().iter().any(|r| r.kind == RelocKind::Abs64 && obj.symbol(r.symbol).name == "entry"));
    let dir = scratch("dwarf");
    let path = dir.join("prog.o");
    std::fs::write(&path, crate::mc::elf::write_with(&obj, &ElfTarget::AARCH64).unwrap()).unwrap();
    if let Some(s) = tool("llvm-dwarfdump", &["--debug-info", "--debug-line"], &path) {
        assert!(s.contains("DW_TAG_subprogram") && s.contains("\"entry\""), "{s}");
        assert!(s.contains("prog.lf") && s.contains("Line table"), "{s}");
        // Rows for the source lines of `entry` (the `.lf` lines 16..36).
        assert!(s.lines().any(|l| l.trim_start().starts_with("0x") && l.contains(" 20 ")), "{s}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
