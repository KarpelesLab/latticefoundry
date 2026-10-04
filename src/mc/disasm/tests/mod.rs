//! Tests of the disassemblers: per-architecture round trips against LF's own
//! encoders, differential comparison with `llvm-objdump -d` over objects LF
//! compiles (skipped when LLVM is not installed), the object readers, and
//! robustness against random bytes.

mod aarch64;
mod avr;
mod corpus;
mod objects;
mod riscv;
mod thumb;
mod wasm;
mod x86;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use super::listing::{ListingOptions, instructions};
use super::objfile;
use super::{Inst, Options, decode, disassemble};
use crate::codegen::CodegenOptions;
use crate::ir::Module;
use crate::mc::object::ObjectModule;
use crate::support::StrInterner;
use crate::target::{ObjectFormat, TargetArch};

/// Every architecture with a decoder.
pub(super) const ARCHS: [TargetArch; 6] = [
    TargetArch::X86_64,
    TargetArch::AArch64,
    TargetArch::Riscv64,
    TargetArch::Thumb,
    TargetArch::Avr,
    TargetArch::Wasm32,
];

/// A small deterministic PRNG (xorshift64*).
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// A value in `0..n` (`n` > 0).
    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub(super) fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

// ===========================================================================
// External tools
// ===========================================================================

/// An LLVM tool: `/usr/lib/llvm/22/bin/<name>` if present, else `<name>` on
/// `PATH` if it runs; `None` when LLVM is not installed.
pub(super) fn llvm_tool(name: &str) -> Option<PathBuf> {
    let pinned = PathBuf::from("/usr/lib/llvm/22/bin").join(name);
    if pinned.exists() {
        return Some(pinned);
    }
    Command::new(name).arg("--version").output().ok().filter(|o| o.status.success()).map(|_| PathBuf::from(name))
}

/// A fresh scratch directory for files an external tool reads.
pub(super) fn scratch(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let d = std::env::temp_dir().join(format!(
        "lf-disasm-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

/// `llvm-objdump -d --no-show-raw-insn <args> <obj>`, parsed: per section
/// name, the `(address, instruction text)` pairs (labels, headers and blank
/// lines dropped; the text is everything after the address's tab).
/// `None` when llvm-objdump is not installed.
pub(super) fn objdump(obj: &[u8], args: &[&str]) -> Option<BTreeMap<String, Vec<(u64, String)>>> {
    let tool = llvm_tool("llvm-objdump")?;
    let dir = scratch("objdump");
    let path = dir.join("in.o");
    std::fs::write(&path, obj).expect("write object");
    let out = Command::new(tool).arg("-d").arg("--no-show-raw-insn").args(args).arg(&path).output().expect("run llvm-objdump");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(out.status.success(), "llvm-objdump {args:?} failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let mut map: BTreeMap<String, Vec<(u64, String)>> = BTreeMap::new();
    let mut section = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Disassembly of section ") {
            section = rest.trim_end_matches(':').to_owned();
            continue;
        }
        let Some((addr, rest)) = line.split_once(':') else { continue };
        let addr = addr.trim();
        if addr.is_empty() || !addr.bytes().all(|c| c.is_ascii_hexdigit()) || line.starts_with(|c: char| c.is_ascii_hexdigit()) {
            continue; // a label line (`0000… <name>:`) or a header
        }
        let Ok(a) = u64::from_str_radix(addr, 16) else { continue };
        let inst = rest.trim_start_matches(' ').trim_start_matches('\t').trim_end();
        map.entry(section.clone()).or_default().push((a, inst.to_owned()));
    }
    Some(map)
}

/// Canonical text for comparing our output with llvm-objdump's: lower case,
/// comments and `<symbol>` annotations dropped, whitespace collapsed, no
/// space after a comma, and every number (decimal or `0x` hex, after `#`,
/// `$` or anything else that is not part of a name) rewritten in decimal.
pub(super) fn normalize(arch: TargetArch, text: &str) -> String {
    let marker = super::comment_marker(arch);
    let mut t = text.to_ascii_lowercase();
    // Comments.
    if let Some(i) = t.find(marker) {
        // AArch64's `//` and Thumb's `@` are unambiguous; x86/RISC-V `#` and
        // AVR `;` too (AArch64 immediates use `#` but its marker is `//`).
        t.truncate(i);
    }
    // `<symbol+off>` annotations.
    while let (Some(a), Some(b)) = (t.find('<'), t.find('>')) {
        if b < a {
            break;
        }
        t.replace_range(a..=b, "");
    }
    let t: String = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let t = t.replace(", ", ",");
    // Numbers.
    let b = t.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let prev_word = i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.');
        let starts_num = c.is_ascii_digit() || (c == b'-' && b.get(i + 1).is_some_and(u8::is_ascii_digit));
        if starts_num && !prev_word {
            let neg = c == b'-';
            let mut j = if neg { i + 1 } else { i };
            let (radix, digits_from) = if b.get(j) == Some(&b'0') && b.get(j + 1) == Some(&b'x') { (16, j + 2) } else { (10, j) };
            j = digits_from;
            while j < b.len() && (if radix == 16 { b[j].is_ascii_hexdigit() } else { b[j].is_ascii_digit() }) {
                j += 1;
            }
            let digits = &t[digits_from..j];
            match u128::from_str_radix(digits, radix) {
                Ok(v) if !(j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_')) => {
                    if neg && v != 0 {
                        out.push('-');
                    }
                    out.push_str(&v.to_string());
                    i = j;
                    continue;
                }
                _ => {}
            }
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// Parse and verify `src`, giving it `arch`'s data layout where the backend
/// needs one other than LP64.
pub(super) fn parse_for(arch: TargetArch, src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = crate::ir::text::parse_module(src, crate::support::diagnostics::FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse: {e:?}"));
    match arch {
        TargetArch::Wasm32 => m.set_data_layout(crate::target::wasm32::data_layout()),
        TargetArch::Avr => m.set_data_layout(crate::target::avr::data_layout_p0()),
        TargetArch::Thumb => m.set_data_layout(crate::target::thumb::data_layout()),
        _ => {}
    }
    crate::verify::verify_module(&m).unwrap_or_else(|d| panic!("verify: {d:?}"));
    (m, syms)
}

/// Compile `src` for `arch` (after the `-O1` pipeline) into an object.
pub(super) fn compile(arch: TargetArch, src: &str) -> ObjectModule {
    let (mut m, syms) = parse_for(arch, src);
    crate::transform::pipeline::optimize(&mut m, crate::transform::pipeline::OptLevel::O1);
    crate::target::compile_module_for(arch, &m, &syms, &CodegenOptions::default())
        .unwrap_or_else(|e| panic!("compile for {arch}: {e}"))
        .object
}

/// A copy of `obj` without its relocations (for a format whose writer cannot
/// express them yet).
fn without_relocations(obj: &ObjectModule) -> ObjectModule {
    let mut out = ObjectModule::new(obj.name.clone());
    for s in obj.sections() {
        out.add_section(s.clone());
    }
    for s in obj.symbols() {
        out.add_symbol(s.clone());
    }
    out
}

/// The file bytes of `obj` in `format`. RISC-V and AArch64 ELF objects are
/// written through a test-local [`ElfTarget`](crate::mc::elf::ElfTarget)
/// while the library has no ELF writer for them, without the relocations it
/// cannot map.
pub(super) fn object_file(arch: TargetArch, obj: &ObjectModule, format: ObjectFormat) -> Vec<u8> {
    match crate::mc::format::write_object_as(obj, arch, format) {
        Ok(b) => b,
        Err(e) => {
            assert_eq!(format, ObjectFormat::Elf, "{arch} {format:?}: {e}");
            use crate::mc::elf::{ElfClass, ElfTarget, RelocFormat};
            let target = ElfTarget {
                class: ElfClass::Elf64,
                endian: crate::ir::Endian::Little,
                machine: if arch == TargetArch::AArch64 { 183 } else { 243 },
                // RISC-V: EF_RISCV_RVC | EF_RISCV_FLOAT_ABI_DOUBLE.
                flags: if arch == TargetArch::Riscv64 { 0x5 } else { 0 },
                reloc_format: RelocFormat::Rela,
                reloc_type: |_| None,
            };
            crate::mc::elf::write_with(&without_relocations(obj), &target).expect("test-local ELF")
        }
    }
}

/// The outcome of comparing our disassembly of `file` with llvm-objdump's.
#[derive(Debug, Default)]
pub(super) struct DiffReport {
    /// Instructions compared.
    pub compared: usize,
    /// Instructions whose text differed (`addr: ours | llvm`).
    pub mismatches: Vec<String>,
    /// Addresses only one side decoded an instruction at.
    pub misaligned: Vec<String>,
}

/// Disassemble every code section of `file` with both decoders and compare
/// instruction by instruction (after [`normalize`], and after `fixup`, which
/// maps both sides' normalized text for an architecture's spelling
/// differences that are not errors). `None` when llvm-objdump is absent.
pub(super) fn differential(
    arch: TargetArch,
    file: &[u8],
    objdump_args: &[&str],
    opts: &Options,
    fixup: &dyn Fn(String) -> String,
) -> Option<DiffReport> {
    let theirs = objdump(file, objdump_args)?;
    let bin = objfile::read(file).expect("read our object");
    let mut report = DiffReport::default();
    let lopts = ListingOptions { disasm: *opts, ..ListingOptions::new() };
    for sec in bin.sections.iter().filter(|s| s.executable) {
        let ours: BTreeMap<u64, Inst> = instructions(sec, arch, &lopts).into_iter().map(|l| (l.addr, l.inst)).collect();
        let Some(list) = theirs.get(&sec.name) else {
            report.misaligned.push(format!("section {} missing from llvm-objdump output", sec.name));
            continue;
        };
        for (addr, text) in list {
            let Some(inst) = ours.get(addr) else {
                report.misaligned.push(format!("{}+{addr:#x}: llvm `{text}` has no counterpart", sec.name));
                continue;
            };
            report.compared += 1;
            let a = fixup(normalize(arch, &inst.text()));
            let b = fixup(normalize(arch, text));
            if a != b {
                report.mismatches.push(format!("{}+{addr:#x}: ours `{}` | llvm `{text}`", sec.name, inst.text()));
            }
        }
    }
    Some(report)
}

/// Assert a [`DiffReport`] is clean, printing its counts.
pub(super) fn assert_clean(what: &str, report: &DiffReport) {
    eprintln!(
        "{what}: {} instructions compared with llvm-objdump, {} mismatches, {} misaligned",
        report.compared,
        report.mismatches.len(),
        report.misaligned.len()
    );
    assert!(
        report.mismatches.is_empty() && report.misaligned.is_empty(),
        "{what}: differences from llvm-objdump:\n{}\n{}",
        report.mismatches.join("\n"),
        report.misaligned.join("\n")
    );
    assert!(report.compared > 0, "{what}: nothing compared");
}

/// The llvm-objdump harness itself: its output parses into instructions at
/// the addresses our reader and walker find. (Each architecture's own test
/// module asserts the texts agree.)
#[test]
fn objdump_harness() {
    let obj = compile(TargetArch::X86_64, corpus::INTS);
    let file = object_file(TargetArch::X86_64, &obj, ObjectFormat::Elf);
    let Some(map) = objdump(&file, &[]) else {
        eprintln!("skipping objdump_harness: no llvm-objdump");
        return;
    };
    let text = map.get(".text").expect("a .text listing");
    assert!(text.len() > 20 && text[0].0 == 0, "{text:?}");
    let report = differential(TargetArch::X86_64, &file, &[], &Options::default(), &|s| s).expect("llvm-objdump");
    assert!(report.compared > 0, "{report:?}");
    let _ = assert_clean;
}

// ===========================================================================
// Robustness
// ===========================================================================

/// Random bytes never panic any decoder, and every decode makes progress
/// within bounds.
#[test]
fn random_bytes_never_panic() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for arch in ARCHS {
        for syntax in [super::Syntax::Att, super::Syntax::Intel] {
            let opts = Options::with_syntax(syntax);
            for round in 0..3000 {
                let len = 1 + rng.below(24) as usize;
                let bytes = rng.bytes(len);
                let inst = decode(arch, &bytes, rng.next() & !1, &opts);
                assert!(inst.len >= 1 && inst.len <= bytes.len(), "{arch} round {round}: len {} of {bytes:02x?}", inst.len);
                let _ = inst.text();
            }
            let blob = rng.bytes(4096);
            let insts = disassemble(arch, &blob, 0x1000, &opts);
            let total: usize = insts.iter().map(|(_, i)| i.len).sum();
            assert_eq!(total, blob.len(), "{arch}: a straight disassembly covers every byte");
        }
    }
}

/// Every one-, two- and (sampled) four-byte prefix decodes without
/// panicking, for every architecture.
#[test]
fn short_and_truncated_inputs() {
    for arch in ARCHS {
        let opts = Options::default();
        for b0 in 0..=255u8 {
            let _ = decode(arch, &[b0], 0, &opts);
            for b1 in (0..=255u8).step_by(3) {
                let i = decode(arch, &[b0, b1], 0, &opts);
                assert!(i.len >= 1 && i.len <= 2);
            }
        }
        let mut rng = Rng(7);
        for _ in 0..20_000 {
            let w = rng.next().to_le_bytes();
            for n in [3, 4, 5, 6, 8] {
                let i = decode(arch, &w[..n], 0x400, &opts);
                assert!(i.len >= 1 && i.len <= n);
            }
        }
        assert_eq!(decode(arch, &[], 0, &opts).len, 0);
    }
}

#[test]
fn data_directives() {
    let i = Inst::data(&[0x12, 0x34, 0x56, 0x78], 4, true);
    assert_eq!(i.text(), ".word\t0x78563412");
    assert_eq!(Inst::data(&[0xab], 4, true).text(), ".byte\t0xab");
    assert_eq!(Inst::data(&[0xab, 0xcd, 0xef], 4, true).text(), ".short\t0xcdab");
    assert!(!i.known);
}

#[test]
fn normalization() {
    assert_eq!(normalize(TargetArch::X86_64, "movq\t$0x10, -0x8(%rbp)  # foo"), "movq $16,-8(%rbp)");
    assert_eq!(normalize(TargetArch::AArch64, "stp\tx29, x30, [sp, #-0x10]!"), "stp x29,x30,[sp,#-16]!");
    assert_eq!(normalize(TargetArch::AArch64, "b\t0x40 <foo+0x8>"), "b 64");
    assert_eq!(normalize(TargetArch::Riscv64, "addi\ta0, a1, -12"), "addi a0,a1,-12");
}
