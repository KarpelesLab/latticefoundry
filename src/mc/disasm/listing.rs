//! Objdump-style listings of a [`Binary`]: each code section's instructions
//! with their addresses and bytes, symbol labels, branch targets resolved to
//! `<symbol+offset>`, and relocations noted inline:
//!
//! ```text
//! 0000000000000000 <main>:
//!        0: 55                               pushq   %rbp
//!        5: e8 00 00 00 00                   callq   helper  # R_X86_64_PLT32 helper-0x4
//! ```
//!
//! Decoding restarts at every label (so a function always starts on an
//! instruction boundary), honors Arm/AArch64 `$d` mapping symbols (data
//! between them prints as `.word`s), and only decodes a wasm code section's
//! function bodies.

use std::fmt::Write as _;

use super::objfile::{Binary, CodeSection, LabelKind, MappingKind};
use super::{Inst, Options, State, comment_marker, decode_in, min_unit};
use crate::target::TargetArch;

/// What to list.
#[derive(Clone, Copy, Debug, Default)]
pub struct ListingOptions {
    /// Decoder options (the x86 syntax).
    pub disasm: Options,
    /// Print each instruction's bytes (default on via [`ListingOptions::new`]).
    pub show_bytes: bool,
    /// Note relocations inline (default on via [`ListingOptions::new`]).
    pub relocs: bool,
    /// Only instructions at or after this address.
    pub start: Option<u64>,
    /// Only instructions before this address.
    pub stop: Option<u64>,
    /// Disassemble every section with contents, not only executable ones.
    pub all_sections: bool,
}

impl ListingOptions {
    /// The defaults: bytes shown, relocations noted, every code section.
    pub fn new() -> ListingOptions {
        ListingOptions { show_bytes: true, relocs: true, ..ListingOptions::default() }
    }
}

/// One decoded instruction of a section, located.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Located {
    /// Its address.
    pub addr: u64,
    /// The instruction.
    pub inst: Inst,
}

/// The instructions of `sec` decoded as `arch`, the way a listing walks them:
/// restarting at each label, `$d` regions as data, only the
/// [regions](CodeSection::regions) when the section has any, and only the
/// addresses within `opts.start..opts.stop`.
pub fn instructions(sec: &CodeSection, arch: TargetArch, opts: &ListingOptions) -> Vec<Located> {
    let mut out = Vec::new();
    let ranges: Vec<(u64, u64)> = if sec.regions.is_empty() {
        vec![(sec.addr, sec.end())]
    } else {
        sec.regions.iter().map(|r| (r.start, r.end)).collect()
    };
    for (start, end) in ranges {
        walk(sec, arch, opts, start, end, &mut |addr, inst| out.push(Located { addr, inst }));
    }
    out
}

/// Decode `start..end` of `sec`, calling `f` with each instruction.
fn walk(sec: &CodeSection, arch: TargetArch, opts: &ListingOptions, start: u64, end: u64, f: &mut dyn FnMut(u64, Inst)) {
    let lo = opts.start.map_or(start, |s| s.max(start));
    let hi = opts.stop.map_or(end, |s| s.min(end));
    let mut addr = start;
    let mut state = State::default();
    // Skip to `lo` by decoding (to stay in step) without reporting.
    while addr < hi {
        let boundary = next_boundary(sec, addr, end);
        if sec.labels.iter().any(|l| l.addr == addr) {
            state = State::default(); // no IT block spans a label
        }
        let from = (addr - sec.addr) as usize;
        let to = (boundary - sec.addr) as usize;
        let Some(bytes) = sec.bytes.get(from..to) else { break };
        let inst = if in_data(sec, addr) {
            let unit = if arch == TargetArch::Thumb || arch == TargetArch::AArch64 { 4 } else { min_unit(arch) };
            state = State::default();
            Inst::data(bytes, unit.min(bytes.len()), true)
        } else {
            decode_in(arch, bytes, addr, &opts.disasm, &mut state)
        };
        let len = inst.len.max(1) as u64;
        if addr >= lo {
            f(addr, inst);
        }
        addr += len;
    }
}

/// The first label, mapping-symbol or region boundary after `addr` (or `end`).
fn next_boundary(sec: &CodeSection, addr: u64, end: u64) -> u64 {
    sec.labels.iter().map(|l| l.addr).filter(|&a| a > addr && a < end).min().unwrap_or(end)
}

/// Whether the last mapping symbol at or before `addr` is `$d`.
fn in_data(sec: &CodeSection, addr: u64) -> bool {
    sec.labels
        .iter()
        .filter(|l| l.addr <= addr)
        .filter_map(|l| match l.kind {
            LabelKind::Mapping(k) => Some(k),
            _ => None,
        })
        .next_back()
        == Some(MappingKind::Data)
}

/// The label for `addr`: `name` or `name+0xoff`, from the nearest
/// non-mapping label at or before it in any section of `bin` containing it.
pub fn symbolize(bin: &Binary, addr: u64) -> Option<String> {
    let sec = bin.sections.iter().find(|s| addr >= s.addr && addr < s.end().max(s.addr + 1))?;
    let l = sec.labels.iter().rfind(|l| l.addr <= addr && !matches!(l.kind, LabelKind::Mapping(_)))?;
    Some(if l.addr == addr { l.name.clone() } else { format!("{}+{:#x}", l.name, addr - l.addr) })
}

/// The bytes of an instruction as a listing shows them: x86, AVR and wasm
/// byte by byte; AArch64 and RISC-V as their instruction words; Thumb as
/// halfwords.
fn bytes_column(arch: TargetArch, bytes: &[u8], inst: &Inst) -> String {
    let unit = match arch {
        TargetArch::AArch64 if inst.known => 4,
        TargetArch::Riscv64 if inst.known => bytes.len(),
        TargetArch::Thumb if inst.known => 2,
        _ => 1,
    };
    if unit == 1 || !bytes.len().is_multiple_of(unit) {
        return bytes.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
    }
    bytes
        .chunks(unit)
        .map(|c| c.iter().rev().map(|b| format!("{b:02x}")).collect::<String>())
        .collect::<Vec<_>>()
        .join(" ")
}

/// List every code section of `bin` (every section with `all_sections`) as
/// `arch`, headed by `file_name` and the format description.
pub fn list(bin: &Binary, arch: TargetArch, file_name: &str, opts: &ListingOptions) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "\n{file_name}:\tfile format {}\n", bin.description);
    let wide = matches!(arch, TargetArch::X86_64 | TargetArch::AArch64 | TargetArch::Riscv64);
    let aw = if wide { 16 } else { 8 };
    let bytes_w = match arch {
        TargetArch::X86_64 => 3 * 10,
        TargetArch::Wasm32 => 3 * 6,
        TargetArch::Avr => 3 * 4,
        TargetArch::Thumb => 10,
        _ => 8,
    };
    let marker = comment_marker(arch);
    for sec in &bin.sections {
        if !(sec.executable || opts.all_sections) || sec.bytes.is_empty() {
            continue;
        }
        if let Some(stop) = opts.stop
            && sec.addr >= stop
        {
            continue;
        }
        if let Some(start) = opts.start
            && sec.end() <= start
        {
            continue;
        }
        let _ = writeln!(out, "Disassembly of section {}:", sec.name);
        let ranges: Vec<(u64, u64, Option<&str>)> = if sec.regions.is_empty() {
            vec![(sec.addr, sec.end(), None)]
        } else {
            sec.regions.iter().map(|r| (r.start, r.end, r.note.as_deref())).collect()
        };
        let mut printed_label: Option<u64> = None;
        let mut prev_end = sec.addr;
        for (start, end, note) in ranges {
            // A region's function label sits before the region (at the body
            // header): print it first.
            if !sec.regions.is_empty() {
                for l in sec.labels.iter().filter(|l| l.addr >= prev_end && l.addr < start && !matches!(l.kind, LabelKind::Mapping(_))) {
                    if opts.start.is_none_or(|s| start >= s) && opts.stop.is_none_or(|s| start < s) {
                        let _ = writeln!(out, "\n{:0aw$x} <{}>:", l.addr, l.name);
                    }
                }
            }
            if let Some(note) = note
                && opts.start.is_none_or(|s| end > s)
                && opts.stop.is_none_or(|s| start < s)
            {
                let _ = writeln!(out, "{:>8}  {marker} {note}", "");
            }
            walk(sec, arch, opts, start, end, &mut |addr, inst| {
                if printed_label != Some(addr) {
                    for l in sec.labels.iter().filter(|l| l.addr == addr && !matches!(l.kind, LabelKind::Mapping(_))) {
                        let _ = writeln!(out, "\n{:0aw$x} <{}>:", addr, l.name);
                    }
                    printed_label = Some(addr);
                }
                let from = (addr - sec.addr) as usize;
                let raw = &sec.bytes[from..from + inst.len.min(sec.bytes.len() - from)];
                let relocs: Vec<_> = if opts.relocs {
                    sec.relocs.iter().filter(|r| r.addr >= addr && r.addr < addr + inst.len as u64).collect()
                } else {
                    Vec::new()
                };
                let mut shown = inst.clone();
                let mut suffix = String::new();
                if let (Some(t), Some(k)) = (inst.target, inst.target_operand) {
                    if let Some(r) = relocs.first() {
                        if let Some(op) = shown.operands.get_mut(k) {
                            *op = r.symbol.clone();
                        }
                    } else if let Some(sym) = symbolize(bin, t) {
                        suffix = format!(" <{sym}>");
                    }
                }
                let mut line = format!("{addr:8x}: ");
                if opts.show_bytes {
                    let _ = write!(line, "{:<bytes_w$} ", bytes_column(arch, raw, &inst));
                }
                let _ = write!(line, "\t{}{suffix}", shown.text());
                for r in relocs {
                    let _ = write!(line, "  {marker} {}", r.note());
                }
                let _ = writeln!(out, "{}", line.trim_end());
            });
            prev_end = end;
        }
        out.push('\n');
    }
    out
}
