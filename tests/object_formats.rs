//! PE/COFF and Mach-O objects from real backend output, cross-checked with
//! external tools and linked into executables by qld.
//!
//! - Every object is parsed by `llvm-readobj` / `llvm-objdump` (and COFF by
//!   GNU `objdump`) when they are installed; a missing tool skips its check.
//! - qld's MinGW flavor (`-m i386pep` / `-m arm64pe`) links the COFF objects
//!   into PE executables, and its ld64 flavor links the Mach-O objects into
//!   Mach-O executables. Neither can run here, so the tests check their
//!   structure: the headers, the entry point, and the call from `main` to
//!   `helper` resolved to `helper`'s address.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::codegen::CodegenOptions;
use latticefoundry::ir::text;
use latticefoundry::mc::object::ObjectModule;
use latticefoundry::mc::write_object;
use latticefoundry::support::StrInterner;
use latticefoundry::support::diagnostics::FileId;
use latticefoundry::target::{self, TargetArch, TargetOs, Triple};

const SRC: &str = r#"
module "fmt"
global @counter : i64 = i64 40
global constant @table : [2 x ptr] = [2 x ptr] (ptr @counter, ptr @helper)
func @helper(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %s = add %a, %b : i64
  ret %s
}
func @main() -> i64 {
entry ^0:
  %c = load @counter align 8 : i64
  %r = call @helper(%c, i64 2) : i64
  ret %r
}
"#;

fn compile(triple: Triple) -> ObjectModule {
    let mut syms = StrInterner::new();
    let m = text::parse_module(SRC, FileId::new(0), &mut syms).expect("parse");
    let opts = CodegenOptions::default().with_os(triple.os);
    match triple.arch {
        TargetArch::X86_64 => target::x86_64::compile_module_with(&m, &syms, &opts).object,
        TargetArch::AArch64 => target::aarch64::compile_module_with(&m, &syms, &opts).object,
        other => panic!("no backend for {other:?} in this test"),
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-objfmt-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Find an LLVM tool on `PATH` or in `/usr/lib/llvm/*/bin`.
fn llvm_tool(name: &str) -> Option<PathBuf> {
    let on_path = Command::new(name).arg("--version").output().is_ok_and(|o| o.status.success());
    if on_path {
        return Some(PathBuf::from(name));
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir("/usr/lib/llvm")
        .ok()?
        .flatten()
        .map(|e| e.path().join("bin").join(name))
        .filter(|p| p.is_file())
        .collect();
    dirs.sort();
    dirs.pop()
}

/// Run `tool args.. file`, returning stdout, or `None` (skip) when the tool
/// is missing. A tool that runs but rejects the file fails the test.
fn inspect(tool: &str, args: &[&str], file: &Path) -> Option<String> {
    let Some(path) = llvm_tool(tool) else {
        eprintln!("skipping {tool} check: not installed");
        return None;
    };
    let out = Command::new(&path).args(args).arg(file).output().expect("run tool");
    assert!(
        out.status.success(),
        "{tool} {args:?} rejected {}: {}",
        file.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn gnu_objdump_supports(fmt: &str) -> bool {
    Command::new("objdump")
        .arg("--help")
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains(fmt))
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
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

// ---------------------------------------------------------------------------
// COFF
// ---------------------------------------------------------------------------

fn coff_object(arch: TargetArch) -> Vec<u8> {
    let triple = Triple::new(arch, TargetOs::Windows);
    write_object(&compile(triple), triple).expect("write COFF")
}

#[test]
fn coff_x86_64_parses_with_llvm_and_binutils() {
    let dir = scratch("coff-x64");
    let obj = write(&dir, "t.obj", &coff_object(TargetArch::X86_64));
    if let Some(out) = inspect("llvm-readobj", &["--file-headers", "--sections", "--relocations", "--symbols"], &obj) {
        assert!(out.contains("IMAGE_FILE_MACHINE_AMD64"), "{out}");
        assert!(out.contains("IMAGE_REL_AMD64_REL32"), "{out}");
        assert!(out.contains("IMAGE_REL_AMD64_ADDR64"), "{out}");
        assert!(out.contains("Name: main"), "{out}");
        assert!(out.contains("Name: helper"), "{out}");
        assert!(out.contains(".rdata"), "{out}");
    }
    if let Some(out) = inspect("llvm-objdump", &["-dr"], &obj) {
        assert!(out.contains("<main>:"), "{out}");
        assert!(out.contains("IMAGE_REL_AMD64_REL32\thelper") || out.contains("IMAGE_REL_AMD64_REL32 helper"), "{out}");
    }
    if gnu_objdump_supports("pe-x86-64") {
        let out = Command::new("objdump").args(["-x", "-dr"]).arg(&obj).output().unwrap();
        assert!(out.status.success(), "objdump: {}", String::from_utf8_lossy(&out.stderr));
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains("pe-x86-64"), "{s}");
        assert!(s.contains("R_AMD64_REL32") || s.contains("IMAGE_REL_AMD64_REL32"), "{s}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn coff_arm64_parses_with_llvm() {
    let dir = scratch("coff-a64");
    let obj = write(&dir, "t.obj", &coff_object(TargetArch::AArch64));
    if let Some(out) = inspect("llvm-readobj", &["--file-headers", "--relocations", "--symbols"], &obj) {
        assert!(out.contains("IMAGE_FILE_MACHINE_ARM64"), "{out}");
        assert!(out.contains("IMAGE_REL_ARM64_BRANCH26"), "{out}");
        assert!(out.contains("IMAGE_REL_ARM64_PAGEBASE_REL21"), "{out}");
        assert!(out.contains("IMAGE_REL_ARM64_PAGEOFFSET_12A"), "{out}");
    }
    if let Some(out) = inspect("llvm-objdump", &["-dr"], &obj) {
        assert!(out.contains("<main>:") && out.contains("bl"), "{out}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Link a COFF object with qld's MinGW flavor into a PE executable (no
/// imports: `main` is the entry point, which Windows starts and whose return
/// value becomes the exit code).
fn link_pe(obj: &Path, out: &Path, emulation: &str) {
    let args: Vec<OsString> = vec![
        "-m".into(),
        emulation.into(),
        "--entry".into(),
        "main".into(),
        "--subsystem".into(),
        "console".into(),
        "-o".into(),
        out.into(),
        obj.into(),
    ];
    latticefoundry::link::gnu::link_gnu("test", &args).expect("qld PE link");
}

/// A PE section: name, RVA, file offset, virtual size.
type PeSection = (String, u32, u32, u32);

/// The PE's `(machine, entry RVA, image base, sections)`.
fn pe_summary(pe: &[u8]) -> (u16, u32, u64, Vec<PeSection>) {
    assert_eq!(&pe[0..2], b"MZ");
    let pe_off = u32_at(pe, 0x3c) as usize;
    assert_eq!(&pe[pe_off..pe_off + 4], b"PE\0\0");
    let coff = pe_off + 4;
    let machine = u16_at(pe, coff);
    let nsect = u16_at(pe, coff + 2) as usize;
    let opt_size = u16_at(pe, coff + 16) as usize;
    let opt = coff + 20;
    assert_eq!(u16_at(pe, opt), 0x20b, "PE32+");
    let entry = u32_at(pe, opt + 16);
    let image_base = u64_at(pe, opt + 24);
    let mut sections = Vec::new();
    for k in 0..nsect {
        let o = opt + opt_size + 40 * k;
        let name = String::from_utf8(pe[o..o + 8].iter().copied().take_while(|&c| c != 0).collect()).unwrap();
        sections.push((name, u32_at(pe, o + 12), u32_at(pe, o + 20), u32_at(pe, o + 8)));
    }
    (machine, entry, image_base, sections)
}

#[test]
fn coff_x86_64_links_into_a_pe_executable() {
    let dir = scratch("pe-x64");
    let obj = write(&dir, "t.obj", &coff_object(TargetArch::X86_64));
    let exe = dir.join("t.exe");
    link_pe(&obj, &exe, "i386pep");
    let pe = std::fs::read(&exe).unwrap();
    let (machine, entry, _base, sections) = pe_summary(&pe);
    assert_eq!(machine, 0x8664);
    let text = sections.iter().find(|s| s.0 == ".text").expect(".text");
    assert!(entry >= text.1 && entry < text.1 + text.3, "entry inside .text");
    // `main` starts with `push rbp` (55); follow its `call rel32` to `helper`.
    let file_of = |rva: u32| (rva - text.1 + text.2) as usize;
    assert_eq!(pe[file_of(entry)], 0x55);
    let code = &pe[file_of(entry)..file_of(entry) + 64];
    let at = code.iter().position(|&b| b == 0xe8).expect("call in main");
    let rel = i32::from_le_bytes(code[at + 1..at + 5].try_into().unwrap());
    let target = (entry as i64 + at as i64 + 5 + rel as i64) as u32;
    // `helper` is the first function in .text (16-byte aligned at its start).
    assert_eq!(target, text.1, "call resolved to helper");
    if let Some(out) = inspect("llvm-readobj", &["--file-headers"], &exe) {
        assert!(out.contains("IMAGE_FILE_MACHINE_AMD64"), "{out}");
        assert!(out.contains("IMAGE_SUBSYSTEM_WINDOWS_CUI"), "{out}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn coff_arm64_links_into_a_pe_executable() {
    let dir = scratch("pe-a64");
    let obj = write(&dir, "t.obj", &coff_object(TargetArch::AArch64));
    let exe = dir.join("t.exe");
    link_pe(&obj, &exe, "arm64pe");
    let pe = std::fs::read(&exe).unwrap();
    let (machine, entry, _base, sections) = pe_summary(&pe);
    assert_eq!(machine, 0xaa64);
    let text = sections.iter().find(|s| s.0 == ".text").expect(".text");
    assert!(entry >= text.1 && entry < text.1 + text.3);
    // Follow main's `bl` to helper.
    let file_of = |rva: u32| (rva - text.1 + text.2) as usize;
    let mut found = false;
    for k in 0..32 {
        let pc = entry + 4 * k;
        let w = u32_at(&pe, file_of(pc));
        if w & 0xfc00_0000 == 0x9400_0000 {
            let imm = ((w & 0x03ff_ffff) << 6) as i32 >> 6;
            assert_eq!((pc as i64 + 4 * imm as i64) as u32, text.1, "bl resolved to helper");
            found = true;
            break;
        }
    }
    assert!(found, "a bl in main");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Mach-O
// ---------------------------------------------------------------------------

fn macho_object(arch: TargetArch) -> Vec<u8> {
    let triple = Triple::new(arch, TargetOs::Darwin);
    write_object(&compile(triple), triple).expect("write Mach-O")
}

#[test]
fn macho_x86_64_parses_with_llvm() {
    let dir = scratch("macho-x64");
    let obj = write(&dir, "t.o", &macho_object(TargetArch::X86_64));
    if let Some(out) = inspect("llvm-readobj", &["--file-headers", "--sections", "--relocations", "--symbols", "--macho-dysymtab"], &obj) {
        assert!(out.contains("CpuType: X86-64 (0x1000007)"), "{out}");
        assert!(out.contains("FileType: Relocatable (0x1)"), "{out}");
        assert!(out.contains("nextdefsym: 4"), "{out}");
        assert!(out.contains("X86_64_RELOC_BRANCH"), "{out}");
        assert!(out.contains("X86_64_RELOC_SIGNED"), "{out}");
        assert!(out.contains("X86_64_RELOC_UNSIGNED"), "{out}");
        assert!(out.contains("_main"), "{out}");
        assert!(out.contains("_helper"), "{out}");
    }
    if let Some(out) = inspect("llvm-objdump", &["-dr", "--macho"], &obj) {
        assert!(out.contains("_main:"), "{out}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn macho_arm64_parses_with_llvm() {
    let dir = scratch("macho-a64");
    let obj = write(&dir, "t.o", &macho_object(TargetArch::AArch64));
    if let Some(out) = inspect("llvm-readobj", &["--file-headers", "--relocations", "--symbols"], &obj) {
        assert!(out.contains("CpuType: Arm64 (0x100000C)"), "{out}");
        assert!(out.contains("ARM64_RELOC_BRANCH26"), "{out}");
        assert!(out.contains("ARM64_RELOC_PAGE21"), "{out}");
        assert!(out.contains("ARM64_RELOC_PAGEOFF12"), "{out}");
    }
    if let Some(out) = inspect("llvm-objdump", &["-dr", "--macho"], &obj) {
        assert!(out.contains("_main:") && out.contains("bl"), "{out}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Link a Mach-O object with qld's ld64 flavor (no dylibs: a static
/// executable whose entry is `_main`) and return the image.
fn link_macho(obj: &Path, arch: &str) -> Vec<u8> {
    use qld_bridge::link_darwin_to_bytes;
    let args: Vec<OsString> = vec![
        "-arch".into(),
        arch.into(),
        "-platform_version".into(),
        "macos".into(),
        "11.0".into(),
        "11.0".into(),
        "-e".into(),
        "_main".into(),
        "-o".into(),
        "a.out".into(),
        obj.into(),
    ];
    link_darwin_to_bytes(&args)
}

mod qld_bridge {
    use std::ffi::OsString;

    pub(super) fn link_darwin_to_bytes(args: &[OsString]) -> Vec<u8> {
        let mut argv = vec![OsString::from("ld64.qld")];
        argv.extend_from_slice(args);
        let options = match qld::args::parse_darwin(&argv).expect("ld64 command line") {
            qld::ParseOutcome::Link(o) => o,
            other => panic!("not a link: {other:?}"),
        };
        let diags = qld::diag::Collect::new();
        match qld::link_to_memory(&options, &diags) {
            Ok(bytes) => bytes,
            Err(e) => panic!("ld64 link failed: {e}: {:?}", diags.take_sorted()),
        }
    }
}

/// `(LC_MAIN entryoff, __text file offset)` of a Mach-O executable.
fn macho_entry_and_text(b: &[u8]) -> (u64, u64) {
    const LC_MAIN: u32 = 0x8000_0028;
    const LC_SEGMENT_64: u32 = 0x19;
    let ncmds = u32_at(b, 16);
    let mut o = 32usize;
    let (mut entry, mut text) = (None, None);
    for _ in 0..ncmds {
        let (cmd, size) = (u32_at(b, o), u32_at(b, o + 4) as usize);
        if cmd == LC_MAIN {
            entry = Some(u64_at(b, o + 8));
        }
        if cmd == LC_SEGMENT_64 {
            for k in 0..u32_at(b, o + 64) as usize {
                let s = o + 72 + 80 * k;
                if b[s..s + 7] == *b"__text\0" {
                    text = Some(u32_at(b, s + 48) as u64);
                }
            }
        }
        o += size;
    }
    (entry.expect("LC_MAIN"), text.expect("__text"))
}

#[test]
fn macho_objects_link_into_executables() {
    for (arch, name) in [(TargetArch::X86_64, "x86_64"), (TargetArch::AArch64, "arm64")] {
        let dir = scratch(&format!("macho-exe-{name}"));
        let obj = write(&dir, "t.o", &macho_object(arch));
        let exe = link_macho(&obj, name);
        assert_eq!(u32_at(&exe, 0), 0xfeed_facf);
        assert_eq!(u32_at(&exe, 12), 2, "MH_EXECUTE");
        let path = write(&dir, "t", &exe);
        // LC_MAIN's entry offset is `_main`, and main's call reaches `_helper`
        // (the first function in __text).
        let (entryoff, helper_off) = macho_entry_and_text(&exe);
        if arch == TargetArch::X86_64 {
            assert_eq!(exe[entryoff as usize], 0x55, "main starts with push rbp");
            let code = &exe[entryoff as usize..entryoff as usize + 64];
            let at = code.iter().position(|&b| b == 0xe8).expect("call in main");
            let rel = i32::from_le_bytes(code[at + 1..at + 5].try_into().unwrap());
            assert_eq!(entryoff as i64 + at as i64 + 5 + rel as i64, helper_off as i64);
        } else {
            let mut found = false;
            for k in 0..32u64 {
                let pc = entryoff + 4 * k;
                let w = u32_at(&exe, pc as usize);
                if w & 0xfc00_0000 == 0x9400_0000 {
                    let imm = ((w & 0x03ff_ffff) << 6) as i32 >> 6;
                    assert_eq!(pc as i64 + 4 * imm as i64, helper_off as i64, "bl resolved to _helper");
                    found = true;
                    break;
                }
            }
            assert!(found, "a bl in main");
        }
        if let Some(out) = inspect("llvm-objdump", &["-d", "--macho"], &path) {
            assert!(out.contains("_main"), "{out}");
            assert!(out.contains("_helper"), "{out}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
