//! A DWARF 4 debug-information emitter (ROADMAP Phase 10).
//!
//! Given a compile unit's metadata and a per-function code map, this builds the
//! four core DWARF sections a debugger needs to recover function names, address
//! ranges, and source lines for LatticeFoundry-compiled code:
//!
//! - [`.debug_abbrev`](build) — the abbreviation table describing the two DIE
//!   shapes we emit (a `DW_TAG_compile_unit` and a `DW_TAG_subprogram`);
//! - `.debug_info` — the compile-unit DIE (producer/name/comp_dir/low_pc/high_pc/
//!   stmt_list) and one subprogram DIE per function (name/low_pc/high_pc/
//!   decl_file/decl_line/external);
//! - `.debug_str` — the string table the DIEs' `DW_FORM_strp` attributes point
//!   into;
//! - `.debug_line` — a real line-number program (file/dir tables plus the state
//!   machine: `DW_LNE_set_address`, `DW_LNS_advance_line`/`advance_pc`, `copy`,
//!   `DW_LNE_end_sequence`).
//!
//! Every runtime address in the debug data (`DW_AT_low_pc`, and each
//! `DW_LNE_set_address` operand) is emitted as a zeroed field plus an
//! [`Abs64`](crate::mc::object::RelocKind::Abs64) relocation against the
//! function's symbol, so the linker fills the real address. High-pc values are
//! emitted as constant offsets (`DW_FORM_data8`), which need no relocation.
//!
//! The format is implemented from the published DWARF specification (tenet T1);
//! nothing here is copied from another toolchain. Output is deterministic.

use crate::mc::emit::{Emitted, Emitter, Ref};
use crate::mc::object::RelocKind;

// ---- DWARF constants (from the DWARF 4 specification) ----------------------

const DW_TAG_COMPILE_UNIT: u64 = 0x11;
const DW_TAG_SUBPROGRAM: u64 = 0x2e;

const DW_CHILDREN_NO: u8 = 0x00;
const DW_CHILDREN_YES: u8 = 0x01;

const DW_AT_NAME: u64 = 0x03;
const DW_AT_STMT_LIST: u64 = 0x10;
const DW_AT_LOW_PC: u64 = 0x11;
const DW_AT_HIGH_PC: u64 = 0x12;
const DW_AT_COMP_DIR: u64 = 0x1b;
const DW_AT_PRODUCER: u64 = 0x25;
const DW_AT_DECL_FILE: u64 = 0x3a;
const DW_AT_DECL_LINE: u64 = 0x3b;
const DW_AT_EXTERNAL: u64 = 0x3f;

const DW_FORM_ADDR: u64 = 0x01;
const DW_FORM_DATA8: u64 = 0x07;
const DW_FORM_FLAG: u64 = 0x0c;
const DW_FORM_STRP: u64 = 0x0e;
const DW_FORM_UDATA: u64 = 0x0f;
const DW_FORM_SEC_OFFSET: u64 = 0x17;

// Line-number program opcodes.
const DW_LNS_COPY: u8 = 1;
const DW_LNS_ADVANCE_PC: u8 = 2;
const DW_LNS_ADVANCE_LINE: u8 = 3;
const DW_LNE_END_SEQUENCE: u8 = 1;
const DW_LNE_SET_ADDRESS: u8 = 2;

const DWARF_VERSION: u16 = 4;
const ADDRESS_SIZE: u8 = 8;

/// Per-function debug facts the emitter needs.
#[derive(Clone, Debug)]
pub struct FuncDebug {
    /// The function's symbol name; address fields relocate against it.
    pub name: String,
    /// The 1-based source line the function is declared on.
    pub decl_line: u32,
    /// The length in bytes of the function's machine code (the `high_pc` offset).
    pub size: u64,
    /// Statement rows `(offset_within_function, source_line)`, ascending by
    /// offset. The first row is conventionally the function entry at offset 0.
    pub rows: Vec<(u64, u32)>,
}

/// A compile unit to emit debug info for.
#[derive(Clone, Debug)]
pub struct DebugUnit {
    /// The source file name (`DW_AT_name`), relative to `comp_dir`.
    pub file_name: String,
    /// The compilation directory (`DW_AT_comp_dir`).
    pub comp_dir: String,
    /// The producer string (`DW_AT_producer`).
    pub producer: String,
    /// The total size of the text range the unit covers (`high_pc` offset from
    /// the first function's `low_pc`).
    pub text_size: u64,
    /// The functions of the unit, in address (layout) order.
    pub funcs: Vec<FuncDebug>,
}

/// The four DWARF section blobs produced for a [`DebugUnit`]. `abbrev` and `str`
/// are plain bytes; `info` and `line` carry the address relocations.
#[derive(Clone, Debug)]
pub struct DwarfSections {
    /// `.debug_abbrev` bytes.
    pub abbrev: Vec<u8>,
    /// `.debug_str` bytes.
    pub str: Vec<u8>,
    /// `.debug_info` bytes and its address relocations.
    pub info: Emitted,
    /// `.debug_line` bytes and its address relocations.
    pub line: Emitted,
}

// ---- LEB128 helpers --------------------------------------------------------

/// Append an unsigned LEB128 value to `buf`.
fn uleb(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if v == 0 {
            break;
        }
    }
}

/// Append a signed LEB128 value to `buf`.
fn sleb(buf: &mut Vec<u8>, mut v: i64) {
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7; // arithmetic shift keeps the sign
        let sign_bit = byte & 0x40 != 0;
        let done = (v == 0 && !sign_bit) || (v == -1 && sign_bit);
        if !done {
            byte |= 0x80;
        }
        buf.push(byte);
        if done {
            break;
        }
    }
}

/// Append an unsigned LEB128 value to an [`Emitter`].
fn uleb_e(e: &mut Emitter, v: u64) {
    let mut b = Vec::new();
    uleb(&mut b, v);
    e.bytes(&b);
}

/// Append a signed LEB128 value to an [`Emitter`].
fn sleb_e(e: &mut Emitter, v: i64) {
    let mut b = Vec::new();
    sleb(&mut b, v);
    e.bytes(&b);
}

// ---- String table ----------------------------------------------------------

/// A `.debug_str` table: strings, each NUL-terminated. Offsets returned by `add`
/// are what `DW_FORM_strp` attributes reference.
#[derive(Default)]
struct DebugStr {
    buf: Vec<u8>,
}

impl DebugStr {
    /// Add `s` (deduplicating nothing; callers add each once) and return its
    /// byte offset within the table.
    fn add(&mut self, s: &str) -> u32 {
        let off = self.buf.len() as u32;
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
        off
    }
}

// ---- Emitter -----------------------------------------------------------------

/// Build the four DWARF sections for `unit`.
pub fn build(unit: &DebugUnit) -> DwarfSections {
    let mut strs = DebugStr::default();
    let producer_off = strs.add(&unit.producer);
    let name_off = strs.add(&unit.file_name);
    let comp_dir_off = strs.add(&unit.comp_dir);
    let func_name_off: Vec<u32> = unit.funcs.iter().map(|f| strs.add(&f.name)).collect();

    let abbrev = build_abbrev();
    let info = build_info(unit, producer_off, name_off, comp_dir_off, &func_name_off);
    let line = build_line(unit);

    DwarfSections { abbrev, str: strs.buf, info, line }
}

/// The abbreviation table: two abbreviations (compile unit, subprogram).
fn build_abbrev() -> Vec<u8> {
    let mut b = Vec::new();

    // Abbrev 1: DW_TAG_compile_unit, has children.
    uleb(&mut b, 1);
    uleb(&mut b, DW_TAG_COMPILE_UNIT);
    b.push(DW_CHILDREN_YES);
    let cu_attrs: &[(u64, u64)] = &[
        (DW_AT_PRODUCER, DW_FORM_STRP),
        (DW_AT_NAME, DW_FORM_STRP),
        (DW_AT_COMP_DIR, DW_FORM_STRP),
        (DW_AT_STMT_LIST, DW_FORM_SEC_OFFSET),
        (DW_AT_LOW_PC, DW_FORM_ADDR),
        (DW_AT_HIGH_PC, DW_FORM_DATA8),
    ];
    for &(at, form) in cu_attrs {
        uleb(&mut b, at);
        uleb(&mut b, form);
    }
    uleb(&mut b, 0);
    uleb(&mut b, 0);

    // Abbrev 2: DW_TAG_subprogram, no children.
    uleb(&mut b, 2);
    uleb(&mut b, DW_TAG_SUBPROGRAM);
    b.push(DW_CHILDREN_NO);
    let sp_attrs: &[(u64, u64)] = &[
        (DW_AT_EXTERNAL, DW_FORM_FLAG),
        (DW_AT_NAME, DW_FORM_STRP),
        (DW_AT_DECL_FILE, DW_FORM_UDATA),
        (DW_AT_DECL_LINE, DW_FORM_UDATA),
        (DW_AT_LOW_PC, DW_FORM_ADDR),
        (DW_AT_HIGH_PC, DW_FORM_DATA8),
    ];
    for &(at, form) in sp_attrs {
        uleb(&mut b, at);
        uleb(&mut b, form);
    }
    uleb(&mut b, 0);
    uleb(&mut b, 0);

    // Terminate the table.
    uleb(&mut b, 0);
    b
}

/// The compile-unit + subprogram DIEs of `.debug_info`. Address fields become
/// `Abs64` relocations against the corresponding function symbols.
fn build_info(
    unit: &DebugUnit,
    producer_off: u32,
    name_off: u32,
    comp_dir_off: u32,
    func_name_off: &[u32],
) -> Emitted {
    let mut e = Emitter::new();

    // Unit header: 4-byte unit_length placeholder, version, abbrev offset, addr.
    e.u32(0); // unit_length, patched at the end.
    e.u16(DWARF_VERSION);
    e.u32(0); // debug_abbrev_offset (this unit's abbrevs start at 0).
    e.u8(ADDRESS_SIZE);

    // Compile-unit DIE (abbrev 1).
    uleb_e(&mut e, 1);
    e.u32(producer_off); // DW_AT_producer, strp
    e.u32(name_off); // DW_AT_name, strp
    e.u32(comp_dir_off); // DW_AT_comp_dir, strp
    e.u32(0); // DW_AT_stmt_list, sec_offset (single line program at 0)
    // DW_AT_low_pc = address of the first function (relocated).
    if let Some(first) = unit.funcs.first() {
        e.reference(RelocKind::Abs64, Ref::Symbol(first.name.clone()), 0);
    } else {
        e.u64(0);
    }
    e.u64(unit.text_size); // DW_AT_high_pc, data8 (offset from low_pc)

    // One subprogram DIE per function (abbrev 2).
    for (f, &noff) in unit.funcs.iter().zip(func_name_off) {
        uleb_e(&mut e, 2);
        e.u8(1); // DW_AT_external, flag = true
        e.u32(noff); // DW_AT_name, strp
        uleb_e(&mut e, 1); // DW_AT_decl_file, udata (file index 1)
        uleb_e(&mut e, u64::from(f.decl_line)); // DW_AT_decl_line, udata
        e.reference(RelocKind::Abs64, Ref::Symbol(f.name.clone()), 0); // low_pc (reloc)
        e.u64(f.size); // DW_AT_high_pc, data8
    }

    // End of the compile unit's children.
    uleb_e(&mut e, 0);

    let mut out = e.finish().expect(".debug_info has no internal labels");
    let unit_length = out.bytes.len() as u32 - 4;
    patch_u32(&mut out.bytes, 0, unit_length); // unit_length
    out
}

/// The `.debug_line` line-number program. Each `DW_LNE_set_address` operand is an
/// `Abs64` relocation against the function's symbol.
fn build_line(unit: &DebugUnit) -> Emitted {
    let mut e = Emitter::new();

    e.u32(0); // unit_length placeholder (patched last)
    e.u16(DWARF_VERSION);
    e.u32(0); // header_length placeholder (patched after the header)
    let header_len_field_end = e.offset(); // bytes after here count toward header_length

    e.u8(1); // minimum_instruction_length
    e.u8(1); // maximum_operations_per_instruction (DWARF 4)
    e.u8(1); // default_is_stmt
    e.u8((-5i8) as u8); // line_base
    e.u8(14); // line_range
    e.u8(13); // opcode_base
    // standard_opcode_lengths for opcodes 1..=12.
    for &n in &[0u8, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1] {
        e.u8(n);
    }
    // include_directories: empty list, terminated by a single NUL.
    e.u8(0);
    // file_names: one entry (name, dir_index, mtime, size), then a NUL.
    e.bytes(unit.file_name.as_bytes());
    e.u8(0);
    uleb_e(&mut e, 0); // directory index (0 = comp_dir)
    uleb_e(&mut e, 0); // mtime
    uleb_e(&mut e, 0); // size
    e.u8(0); // end of file_names

    let program_start = e.offset();
    let header_length = (program_start - header_len_field_end) as u32;

    // Line-number program: one sequence per function.
    for f in &unit.funcs {
        // DW_LNE_set_address <low_pc>  (extended opcode).
        e.u8(0); // extended opcode marker
        uleb_e(&mut e, 1 + ADDRESS_SIZE as u64); // length = sub-opcode + address
        e.u8(DW_LNE_SET_ADDRESS);
        e.reference(RelocKind::Abs64, Ref::Symbol(f.name.clone()), 0);

        let mut cur_off: u64 = 0;
        let mut cur_line: i64 = 1;
        let mut emitted_row = false;
        for &(off, line) in &f.rows {
            if off > cur_off {
                e.u8(DW_LNS_ADVANCE_PC);
                uleb_e(&mut e, off - cur_off);
                cur_off = off;
            }
            let line = i64::from(line);
            if line != cur_line {
                e.u8(DW_LNS_ADVANCE_LINE);
                sleb_e(&mut e, line - cur_line);
                cur_line = line;
            }
            e.u8(DW_LNS_COPY);
            emitted_row = true;
        }
        // If the function had no rows, emit at least an entry row at line 1.
        if !emitted_row {
            e.u8(DW_LNS_COPY);
        }
        // Advance to the end of the function and close the sequence.
        if f.size > cur_off {
            e.u8(DW_LNS_ADVANCE_PC);
            uleb_e(&mut e, f.size - cur_off);
        }
        e.u8(0); // extended opcode marker
        uleb_e(&mut e, 1); // length
        e.u8(DW_LNE_END_SEQUENCE);
    }

    let mut out = e.finish().expect(".debug_line has no internal labels");
    patch_u32(&mut out.bytes, 6, header_length); // header_length follows u32+u16
    let unit_length = out.bytes.len() as u32 - 4;
    patch_u32(&mut out.bytes, 0, unit_length); // unit_length
    out
}

/// Overwrite the little-endian `u32` at `at` in `buf`.
fn patch_u32(buf: &mut [u8], at: usize, v: u32) {
    buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_unit() -> DebugUnit {
        DebugUnit {
            file_name: "prog.lf".to_owned(),
            comp_dir: "/work".to_owned(),
            producer: "LatticeFoundry".to_owned(),
            text_size: 0x30,
            funcs: vec![
                FuncDebug {
                    name: "main".to_owned(),
                    decl_line: 2,
                    size: 0x20,
                    rows: vec![(0, 2), (8, 3)],
                },
                FuncDebug {
                    name: "helper".to_owned(),
                    decl_line: 7,
                    size: 0x10,
                    rows: vec![(0, 7)],
                },
            ],
        }
    }

    fn read_u32(b: &[u8], o: usize) -> u32 {
        u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }

    #[test]
    fn uleb_sleb_roundtrip_known_values() {
        let mut b = Vec::new();
        uleb(&mut b, 0);
        assert_eq!(b, vec![0]);
        b.clear();
        uleb(&mut b, 624_485);
        assert_eq!(b, vec![0xe5, 0x8e, 0x26]);
        b.clear();
        sleb(&mut b, -2);
        assert_eq!(b, vec![0x7e]);
        b.clear();
        sleb(&mut b, 127);
        assert_eq!(b, vec![0xff, 0x00]);
    }

    #[test]
    fn info_unit_length_and_version() {
        let s = build(&sample_unit());
        let info = &s.info.bytes;
        // unit_length excludes its own 4 bytes.
        assert_eq!(read_u32(info, 0) as usize, info.len() - 4);
        assert_eq!(u16::from_le_bytes([info[4], info[5]]), DWARF_VERSION);
        assert_eq!(info[10], ADDRESS_SIZE); // after abbrev offset u32
        // Address fields relocate against the function symbols: CU low_pc + one
        // per subprogram = 3 relocations, all Abs64.
        assert_eq!(s.info.relocations.len(), 3);
        assert!(s.info.relocations.iter().all(|r| r.kind == RelocKind::Abs64));
        let names: Vec<&str> = s.info.relocations.iter().map(|r| r.symbol.as_str()).collect();
        assert_eq!(names, vec!["main", "main", "helper"]);
    }

    #[test]
    fn line_header_length_and_relocs() {
        let s = build(&sample_unit());
        let line = &s.line.bytes;
        assert_eq!(read_u32(line, 0) as usize, line.len() - 4);
        assert_eq!(u16::from_le_bytes([line[4], line[5]]), DWARF_VERSION);
        // header_length points just past the file table.
        let header_length = read_u32(line, 6) as usize;
        let program_start = 10 + header_length;
        assert!(program_start < line.len());
        // One DW_LNE_set_address relocation per function.
        assert_eq!(s.line.relocations.len(), 2);
        assert!(s.line.relocations.iter().all(|r| r.kind == RelocKind::Abs64));
    }

    #[test]
    fn abbrev_has_two_abbreviations() {
        let s = build(&sample_unit());
        // Starts with abbrev codes 1 and 2, ends with a 0 terminator.
        assert_eq!(s.abbrev[0], 1);
        assert_eq!(*s.abbrev.last().unwrap(), 0);
    }

    #[test]
    fn str_table_contains_names() {
        let s = build(&sample_unit());
        let text = String::from_utf8_lossy(&s.str);
        assert!(text.contains("LatticeFoundry"));
        assert!(text.contains("prog.lf"));
        assert!(text.contains("main"));
        assert!(text.contains("helper"));
    }

    #[test]
    fn deterministic() {
        let a = build(&sample_unit());
        let b = build(&sample_unit());
        assert_eq!(a.abbrev, b.abbrev);
        assert_eq!(a.str, b.str);
        assert_eq!(a.info.bytes, b.info.bytes);
        assert_eq!(a.line.bytes, b.line.bytes);
    }
}
