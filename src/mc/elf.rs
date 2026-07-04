//! An ELF64 relocatable-object (`ET_REL`) writer for x86-64, implemented from
//! the ELF specification (ROADMAP Phase 6).
//!
//! This turns the framework's target-independent [`ObjectModule`] into a
//! standard SysV-ABI ELF64 relocatable object that a system linker (or our own
//! future linker) can consume. It is a clean-room implementation of the file
//! format from its public specification — the `Elf64_*` structure layouts,
//! constants, and x86-64 relocation numbers are those of the published standard,
//! not copied from any toolchain (tenet T1).
//!
//! # What it emits
//!
//! - an [`Elf64_Ehdr`](struct docs below) with class `ELFCLASS64`, data
//!   `ELFDATA2LSB`, type `ET_REL`, machine `EM_X86_64`;
//! - one section header per user [`Section`], plus `.symtab`, `.strtab`, a
//!   `.rela.<name>` for every section that has relocations, and `.shstrtab`;
//! - a symbol table with the null symbol first, then all local symbols, then
//!   global/weak ones (as ELF requires), and `.symtab`'s `sh_info` set to the
//!   first non-local index;
//! - `Elf64_Rela` entries mapping each generic [`RelocKind`] to its x86-64
//!   relocation number.
//!
//! Output is little-endian and fully deterministic: sections, symbols, and
//! relocations are emitted in the module's insertion order.
//!
//! [`Section`]: crate::mc::object::Section

use crate::mc::object::{
    ObjectModule, RelocKind, SectionKind, SymbolBinding, SymbolType, SymbolValue,
};

// ===========================================================================
// ELF constants (from the ELF-64 specification)
// ===========================================================================

const EI_NIDENT: usize = 16;

const ELFMAG: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EV_CURRENT: u8 = 1;
const ELFOSABI_SYSV: u8 = 0;

const ET_REL: u16 = 1;
/// The `e_machine` value for x86-64.
pub const EM_X86_64: u16 = 62;

const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;

const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STB_WEAK: u8 = 2;

const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const STT_SECTION: u8 = 3;

const SHN_UNDEF: u16 = 0;

// x86-64 relocation type numbers.
const R_X86_64_64: u32 = 1;
const R_X86_64_PC32: u32 = 2;
const R_X86_64_PLT32: u32 = 4;
const R_X86_64_GOTPCREL: u32 = 9;
const R_X86_64_32: u32 = 10;
const R_X86_64_32S: u32 = 11;
const R_X86_64_PC64: u32 = 24;

/// The size in bytes of an `Elf64_Ehdr`.
const EHDR_SIZE: u64 = 64;
/// The size in bytes of an `Elf64_Shdr`.
const SHDR_SIZE: u64 = 64;
/// The size in bytes of an `Elf64_Sym`.
const SYM_SIZE: u64 = 24;
/// The size in bytes of an `Elf64_Rela`.
const RELA_SIZE: u64 = 24;

/// Map a generic [`RelocKind`] to its x86-64 ELF relocation number.
fn x86_64_reloc(kind: RelocKind) -> u32 {
    match kind {
        RelocKind::Abs64 => R_X86_64_64,
        RelocKind::Abs32 => R_X86_64_32,
        RelocKind::Abs32S => R_X86_64_32S,
        RelocKind::Pc32 => R_X86_64_PC32,
        RelocKind::Pc64 => R_X86_64_PC64,
        RelocKind::Plt32 => R_X86_64_PLT32,
        RelocKind::GotPcRel => R_X86_64_GOTPCREL,
    }
}

fn section_flags_type(kind: SectionKind) -> (u64, u32) {
    match kind {
        SectionKind::Text => (SHF_ALLOC | SHF_EXECINSTR, SHT_PROGBITS),
        SectionKind::Data => (SHF_ALLOC | SHF_WRITE, SHT_PROGBITS),
        SectionKind::Rodata => (SHF_ALLOC, SHT_PROGBITS),
        SectionKind::Bss => (SHF_ALLOC | SHF_WRITE, SHT_NOBITS),
    }
}

fn binding_code(b: SymbolBinding) -> u8 {
    match b {
        SymbolBinding::Local => STB_LOCAL,
        SymbolBinding::Global => STB_GLOBAL,
        SymbolBinding::Weak => STB_WEAK,
    }
}

fn symtype_code(t: SymbolType) -> u8 {
    match t {
        SymbolType::NoType => STT_NOTYPE,
        SymbolType::Object => STT_OBJECT,
        SymbolType::Func => STT_FUNC,
        SymbolType::Section => STT_SECTION,
    }
}

// ===========================================================================
// Small helpers
// ===========================================================================

/// A growable string table (`.strtab`/`.shstrtab`): a leading NUL byte, then
/// each added string NUL-terminated. Returns the offset of each string.
#[derive(Debug, Default)]
struct StringTable {
    buf: Vec<u8>,
}

impl StringTable {
    fn new() -> Self {
        StringTable { buf: vec![0] }
    }

    /// Add `s` and return its byte offset. The empty string maps to offset 0.
    fn add(&mut self, s: &str) -> u32 {
        if s.is_empty() {
            return 0;
        }
        let off = self.buf.len() as u32;
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
        off
    }
}

/// Append `n` zero bytes to `buf` until its length is a multiple of `align`.
fn pad_to(buf: &mut Vec<u8>, align: u64) {
    if align <= 1 {
        return;
    }
    let mask = align - 1;
    let rem = buf.len() as u64 & mask;
    if rem != 0 {
        let pad = (align - rem) as usize;
        buf.resize(buf.len() + pad, 0);
    }
}

/// Write one `Elf64_Shdr` (64 bytes) into `buf`.
#[allow(clippy::too_many_arguments)]
fn write_shdr(
    buf: &mut Vec<u8>,
    name: u32,
    kind: u32,
    flags: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,
) {
    buf.extend_from_slice(&name.to_le_bytes());
    buf.extend_from_slice(&kind.to_le_bytes());
    buf.extend_from_slice(&flags.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // sh_addr
    buf.extend_from_slice(&offset.to_le_bytes());
    buf.extend_from_slice(&size.to_le_bytes());
    buf.extend_from_slice(&link.to_le_bytes());
    buf.extend_from_slice(&info.to_le_bytes());
    buf.extend_from_slice(&addralign.to_le_bytes());
    buf.extend_from_slice(&entsize.to_le_bytes());
}

/// Write one `Elf64_Sym` (24 bytes) into `buf`.
fn write_sym(buf: &mut Vec<u8>, name: u32, info: u8, shndx: u16, value: u64, size: u64) {
    buf.extend_from_slice(&name.to_le_bytes());
    buf.push(info);
    buf.push(0); // st_other
    buf.extend_from_slice(&shndx.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
    buf.extend_from_slice(&size.to_le_bytes());
}

/// Write one `Elf64_Rela` (24 bytes) into `buf`.
fn write_rela(buf: &mut Vec<u8>, offset: u64, sym_index: u32, ty: u32, addend: i64) {
    let info = (u64::from(sym_index) << 32) | u64::from(ty);
    buf.extend_from_slice(&offset.to_le_bytes());
    buf.extend_from_slice(&info.to_le_bytes());
    buf.extend_from_slice(&addend.to_le_bytes());
}

// ===========================================================================
// Writer
// ===========================================================================

/// Serialize `obj` to a valid ELF64 x86-64 relocatable object image.
///
/// The returned bytes form a complete `ET_REL` file: an ELF header, the section
/// contents, a symbol and string table, `.rela.*` relocation sections, and the
/// section header table. Output is deterministic.
pub fn write(obj: &ObjectModule) -> Vec<u8> {
    let sections = obj.sections();
    let n = sections.len();

    // --- assign ELF section indices ---
    // 0 = null; 1..=n = user sections; then symtab, strtab, rela.*, shstrtab.
    let symtab_index = (n + 1) as u32;
    let strtab_index = (n + 2) as u32;
    let mut rela_index_of: Vec<Option<u32>> = vec![None; n];
    let mut next = n as u32 + 3;
    let has_relocs = |i: usize| obj.relocations().iter().any(|r| r.section.index() == i);
    for (i, slot) in rela_index_of.iter_mut().enumerate() {
        if has_relocs(i) {
            *slot = Some(next);
            next += 1;
        }
    }
    let shstrtab_index = next;
    next += 1;
    let total_sections = next as usize;

    // The ELF section index a user section lives at.
    let user_elf_index = |i: usize| (i + 1) as u16;

    // --- order symbols: null, then locals, then globals/weaks ---
    let mut order: Vec<usize> = Vec::with_capacity(obj.symbols().len());
    for (i, s) in obj.symbols().iter().enumerate() {
        if matches!(s.binding, SymbolBinding::Local) {
            order.push(i);
        }
    }
    let nlocal = order.len();
    for (i, s) in obj.symbols().iter().enumerate() {
        if !matches!(s.binding, SymbolBinding::Local) {
            order.push(i);
        }
    }
    // obj symbol index -> symtab index (null occupies slot 0).
    let mut symtab_of = vec![0u32; obj.symbols().len()];
    for (pos, &obj_idx) in order.iter().enumerate() {
        symtab_of[obj_idx] = (pos + 1) as u32;
    }
    let first_global = (nlocal + 1) as u32; // sh_info for .symtab

    // --- build string tables ---
    let mut strtab = StringTable::new();
    let mut sym_name_off: Vec<u32> = vec![0; obj.symbols().len()];
    for (i, s) in obj.symbols().iter().enumerate() {
        sym_name_off[i] = strtab.add(&s.name);
    }

    let mut shstrtab = StringTable::new();
    let mut sec_name_off: Vec<u32> = Vec::with_capacity(n);
    for s in sections {
        sec_name_off.push(shstrtab.add(&s.name));
    }
    let symtab_name_off = shstrtab.add(".symtab");
    let strtab_name_off = shstrtab.add(".strtab");
    let mut rela_name_off: Vec<u32> = vec![0; n];
    for (i, s) in sections.iter().enumerate() {
        if rela_index_of[i].is_some() {
            let name = format!(".rela{}", s.name);
            rela_name_off[i] = shstrtab.add(&name);
        }
    }
    let shstrtab_name_off = shstrtab.add(".shstrtab");

    // --- build .symtab content ---
    let mut symtab = Vec::new();
    write_sym(&mut symtab, 0, 0, SHN_UNDEF, 0, 0); // null symbol
    for &obj_idx in &order {
        let s = &obj.symbols()[obj_idx];
        let info = (binding_code(s.binding) << 4) | symtype_code(s.kind);
        let (shndx, value) = match s.value {
            SymbolValue::Defined { section, offset } => (user_elf_index(section.index()), offset),
            SymbolValue::Undefined => (SHN_UNDEF, 0),
        };
        write_sym(&mut symtab, sym_name_off[obj_idx], info, shndx, value, s.size);
    }

    // --- build .rela.* content per user section ---
    let mut rela_data: Vec<Vec<u8>> = vec![Vec::new(); n];
    for r in obj.relocations() {
        let i = r.section.index();
        let ty = x86_64_reloc(r.kind);
        let sym_index = symtab_of[r.symbol.index()];
        write_rela(&mut rela_data[i], r.offset, sym_index, ty, r.addend);
    }

    // --- lay out the file, recording each section's file offset ---
    let mut buf = vec![0u8; EHDR_SIZE as usize];

    let mut user_offset = vec![0u64; n];
    for (i, s) in sections.iter().enumerate() {
        pad_to(&mut buf, s.align.max(1));
        user_offset[i] = buf.len() as u64;
        if !s.is_nobits() {
            buf.extend_from_slice(&s.bytes);
        }
    }

    pad_to(&mut buf, 8);
    let symtab_offset = buf.len() as u64;
    buf.extend_from_slice(&symtab);

    let strtab_offset = buf.len() as u64;
    buf.extend_from_slice(&strtab.buf);

    let mut rela_offset = vec![0u64; n];
    for (i, data) in rela_data.iter().enumerate() {
        if rela_index_of[i].is_some() {
            pad_to(&mut buf, 8);
            rela_offset[i] = buf.len() as u64;
            buf.extend_from_slice(data);
        }
    }

    let shstrtab_offset = buf.len() as u64;
    buf.extend_from_slice(&shstrtab.buf);

    // --- section header table ---
    pad_to(&mut buf, 8);
    let shoff = buf.len() as u64;

    // 0: null section header.
    write_shdr(&mut buf, 0, SHT_NULL, 0, 0, 0, 0, 0, 0, 0);

    // 1..=n: user sections.
    for (i, s) in sections.iter().enumerate() {
        let (flags, kind) = section_flags_type(s.kind);
        write_shdr(
            &mut buf,
            sec_name_off[i],
            kind,
            flags,
            user_offset[i],
            s.size(),
            0,
            0,
            s.align.max(1),
            0,
        );
    }

    // .symtab
    write_shdr(
        &mut buf,
        symtab_name_off,
        SHT_SYMTAB,
        0,
        symtab_offset,
        symtab.len() as u64,
        strtab_index,
        first_global,
        8,
        SYM_SIZE,
    );

    // .strtab
    write_shdr(
        &mut buf,
        strtab_name_off,
        SHT_STRTAB,
        0,
        strtab_offset,
        strtab.buf.len() as u64,
        0,
        0,
        1,
        0,
    );

    // .rela.* sections, in user-section order.
    for (i, data) in rela_data.iter().enumerate() {
        if rela_index_of[i].is_some() {
            write_shdr(
                &mut buf,
                rela_name_off[i],
                SHT_RELA,
                0,
                rela_offset[i],
                data.len() as u64,
                symtab_index,
                u32::from(user_elf_index(i)),
                8,
                RELA_SIZE,
            );
        }
    }

    // .shstrtab
    write_shdr(
        &mut buf,
        shstrtab_name_off,
        SHT_STRTAB,
        0,
        shstrtab_offset,
        shstrtab.buf.len() as u64,
        0,
        0,
        1,
        0,
    );

    // --- fill in the ELF header ---
    let mut ident = [0u8; EI_NIDENT];
    ident[0..4].copy_from_slice(&ELFMAG);
    ident[4] = ELFCLASS64;
    ident[5] = ELFDATA2LSB;
    ident[6] = EV_CURRENT;
    ident[7] = ELFOSABI_SYSV;
    let mut ehdr = Vec::with_capacity(EHDR_SIZE as usize);
    ehdr.extend_from_slice(&ident);
    ehdr.extend_from_slice(&ET_REL.to_le_bytes());
    ehdr.extend_from_slice(&EM_X86_64.to_le_bytes());
    ehdr.extend_from_slice(&1u32.to_le_bytes()); // e_version
    ehdr.extend_from_slice(&0u64.to_le_bytes()); // e_entry
    ehdr.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
    ehdr.extend_from_slice(&shoff.to_le_bytes()); // e_shoff
    ehdr.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    ehdr.extend_from_slice(&(EHDR_SIZE as u16).to_le_bytes()); // e_ehsize
    ehdr.extend_from_slice(&0u16.to_le_bytes()); // e_phentsize
    ehdr.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
    ehdr.extend_from_slice(&(SHDR_SIZE as u16).to_le_bytes()); // e_shentsize
    ehdr.extend_from_slice(&(total_sections as u16).to_le_bytes()); // e_shnum
    ehdr.extend_from_slice(&(shstrtab_index as u16).to_le_bytes()); // e_shstrndx
    debug_assert_eq!(ehdr.len(), EHDR_SIZE as usize);
    buf[0..EHDR_SIZE as usize].copy_from_slice(&ehdr);

    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::object::{Section, Symbol};

    // ---- a tiny structural re-parser, used only to validate our own output ----

    fn rd_u16(b: &[u8], o: usize) -> u16 {
        u16::from_le_bytes([b[o], b[o + 1]])
    }
    fn rd_u32(b: &[u8], o: usize) -> u32 {
        u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }
    fn rd_u64(b: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
    }
    fn rd_i64(b: &[u8], o: usize) -> i64 {
        i64::from_le_bytes(b[o..o + 8].try_into().unwrap())
    }

    struct Shdr {
        name: u32,
        kind: u32,
        flags: u64,
        offset: u64,
        size: u64,
        link: u32,
        info: u32,
        entsize: u64,
    }

    fn parse_shdrs(b: &[u8]) -> Vec<Shdr> {
        let shoff = rd_u64(b, 40) as usize;
        let shentsize = rd_u16(b, 58) as usize;
        let shnum = rd_u16(b, 60) as usize;
        let mut out = Vec::new();
        for i in 0..shnum {
            let o = shoff + i * shentsize;
            out.push(Shdr {
                name: rd_u32(b, o),
                kind: rd_u32(b, o + 4),
                flags: rd_u64(b, o + 8),
                offset: rd_u64(b, o + 24),
                size: rd_u64(b, o + 32),
                link: rd_u32(b, o + 40),
                info: rd_u32(b, o + 44),
                entsize: rd_u64(b, o + 56),
            });
        }
        out
    }

    fn cstr(b: &[u8], base: usize, name: u32) -> String {
        let start = base + name as usize;
        let mut end = start;
        while b[end] != 0 {
            end += 1;
        }
        String::from_utf8_lossy(&b[start..end]).into_owned()
    }

    /// Build a tiny object: a `.text` with a few bytes, a global function symbol
    /// named `main`, an undefined `puts`, and a PLT32 call relocation to it.
    fn tiny_object() -> ObjectModule {
        let mut m = ObjectModule::new("tiny.o");
        let mut text = Section::new(".text", SectionKind::Text, 16);
        // push rbp; mov rbp,rsp; call rel32(=0); pop rbp; ret
        text.bytes = vec![0x55, 0x48, 0x89, 0xe5, 0xe8, 0, 0, 0, 0, 0x5d, 0xc3];
        let text_id = m.add_section(text);
        m.add_symbol(Symbol::defined(
            "main",
            SymbolBinding::Global,
            SymbolType::Func,
            text_id,
            0,
            11,
        ));
        let puts = m.add_symbol(Symbol::undefined("puts", SymbolBinding::Global));
        m.add_relocation(crate::mc::object::Relocation {
            section: text_id,
            offset: 5,
            symbol: puts,
            kind: RelocKind::Plt32,
            addend: -4,
        });
        m
    }

    #[test]
    fn header_fields_are_correct() {
        let b = write(&tiny_object());
        // Magic, class, data, version, osabi.
        assert_eq!(&b[0..4], &[0x7f, b'E', b'L', b'F']);
        assert_eq!(b[4], ELFCLASS64);
        assert_eq!(b[5], ELFDATA2LSB);
        assert_eq!(b[6], EV_CURRENT);
        assert_eq!(b[7], ELFOSABI_SYSV);
        // Type ET_REL, machine EM_X86_64=62.
        assert_eq!(rd_u16(&b, 16), ET_REL);
        assert_eq!(rd_u16(&b, 18), EM_X86_64);
        assert_eq!(rd_u16(&b, 18), 62);
        // ehsize/shentsize.
        assert_eq!(rd_u16(&b, 52), 64);
        assert_eq!(rd_u16(&b, 58), 64);
        // e_phnum == 0 for a relocatable object.
        assert_eq!(rd_u16(&b, 56), 0);
    }

    #[test]
    fn section_headers_are_consistent() {
        let b = write(&tiny_object());
        let shdrs = parse_shdrs(&b);
        // null + .text + .symtab + .strtab + .rela.text + .shstrtab = 6.
        assert_eq!(shdrs.len(), 6);
        assert_eq!(shdrs[0].kind, SHT_NULL);

        let shstrndx = rd_u16(&b, 62) as usize;
        let shstr_base = shdrs[shstrndx].offset as usize;
        let names: Vec<String> =
            shdrs.iter().map(|s| cstr(&b, shstr_base, s.name)).collect();
        assert_eq!(names[0], "");
        assert_eq!(names[1], ".text");
        assert!(names.contains(&".symtab".to_owned()));
        assert!(names.contains(&".strtab".to_owned()));
        assert!(names.contains(&".rela.text".to_owned()));
        assert!(names.contains(&".shstrtab".to_owned()));

        // .text flags: ALLOC | EXECINSTR, PROGBITS.
        assert_eq!(shdrs[1].kind, SHT_PROGBITS);
        assert_eq!(shdrs[1].flags, SHF_ALLOC | SHF_EXECINSTR);
        // Every section's [offset, offset+size) lies within the file (except
        // NOBITS, of which this object has none).
        for s in &shdrs {
            if s.kind != SHT_NULL && s.kind != SHT_NOBITS {
                assert!(s.offset + s.size <= b.len() as u64, "section out of bounds");
            }
        }
    }

    #[test]
    fn symtab_and_strtab_are_well_formed() {
        let b = write(&tiny_object());
        let shdrs = parse_shdrs(&b);
        let symtab = shdrs.iter().find(|s| s.kind == SHT_SYMTAB).unwrap();
        assert_eq!(symtab.entsize, SYM_SIZE);
        // sh_link points to a STRTAB section.
        assert_eq!(shdrs[symtab.link as usize].kind, SHT_STRTAB);
        let strtab = &shdrs[symtab.link as usize];

        // Symbols: null + main(local? no, global) + puts(global). main and puts
        // are both global, so 0 locals -> sh_info == 1.
        let count = symtab.size / SYM_SIZE;
        assert_eq!(count, 3);
        assert_eq!(symtab.info, 1, "first global symbol index");

        // Resolve names via the linked strtab.
        let strbase = strtab.offset as usize;
        let symbase = symtab.offset as usize;
        let mut found_main = false;
        let mut found_puts_undef = false;
        for i in 0..count as usize {
            let o = symbase + i * SYM_SIZE as usize;
            let name_off = rd_u32(&b, o);
            let name = cstr(&b, strbase, name_off);
            let shndx = rd_u16(&b, o + 6);
            let info = b[o + 4];
            if name == "main" {
                found_main = true;
                assert_eq!(info >> 4, STB_GLOBAL);
                assert_eq!(info & 0xf, STT_FUNC);
                assert_eq!(shndx, 1); // defined in .text (elf index 1)
            }
            if name == "puts" {
                found_puts_undef = true;
                assert_eq!(shndx, SHN_UNDEF);
            }
        }
        assert!(found_main && found_puts_undef);
    }

    #[test]
    fn rela_section_is_well_formed() {
        let b = write(&tiny_object());
        let shdrs = parse_shdrs(&b);
        let rela = shdrs.iter().find(|s| s.kind == SHT_RELA).unwrap();
        assert_eq!(rela.entsize, RELA_SIZE);
        assert_eq!(shdrs[rela.link as usize].kind, SHT_SYMTAB);
        // sh_info identifies the section being relocated (.text at index 1).
        assert_eq!(rela.info, 1);

        let count = rela.size / RELA_SIZE;
        assert_eq!(count, 1);
        let o = rela.offset as usize;
        let r_offset = rd_u64(&b, o);
        let r_info = rd_u64(&b, o + 8);
        let r_addend = rd_i64(&b, o + 16);
        assert_eq!(r_offset, 5);
        assert_eq!((r_info & 0xffff_ffff) as u32, R_X86_64_PLT32);
        assert_eq!(r_addend, -4);
        // The referenced symbol index must be a valid symtab entry.
        let sym_index = (r_info >> 32) as usize;
        let symtab = shdrs.iter().find(|s| s.kind == SHT_SYMTAB).unwrap();
        assert!((sym_index as u64) < symtab.size / SYM_SIZE);
    }

    #[test]
    fn all_reloc_kinds_map() {
        // Exhaustive mapping sanity, so a new kind can't silently fall through.
        assert_eq!(x86_64_reloc(RelocKind::Abs64), R_X86_64_64);
        assert_eq!(x86_64_reloc(RelocKind::Abs32), R_X86_64_32);
        assert_eq!(x86_64_reloc(RelocKind::Abs32S), R_X86_64_32S);
        assert_eq!(x86_64_reloc(RelocKind::Pc32), R_X86_64_PC32);
        assert_eq!(x86_64_reloc(RelocKind::Pc64), R_X86_64_PC64);
        assert_eq!(x86_64_reloc(RelocKind::Plt32), R_X86_64_PLT32);
        assert_eq!(x86_64_reloc(RelocKind::GotPcRel), R_X86_64_GOTPCREL);
    }

    #[test]
    fn multiple_sections_and_local_ordering() {
        // An object with data/rodata/bss and a mix of local and global symbols:
        // check locals precede globals and .bss is NOBITS occupying no file space.
        let mut m = ObjectModule::new("multi.o");
        let text = m.add_section(Section::new(".text", SectionKind::Text, 16));
        m.section_mut(text).bytes = vec![0x90, 0xc3];
        let data = m.add_section(Section::new(".data", SectionKind::Data, 8));
        m.section_mut(data).bytes = vec![0; 16];
        let rodata = m.add_section(Section::new(".rodata", SectionKind::Rodata, 1));
        m.section_mut(rodata).bytes = b"msg\0".to_vec();
        let bss = m.add_section(Section::bss(".bss", 16, 256));

        m.add_symbol(Symbol::defined(
            "g_main",
            SymbolBinding::Global,
            SymbolType::Func,
            text,
            0,
            2,
        ));
        m.add_symbol(Symbol::defined(
            "l_tmp",
            SymbolBinding::Local,
            SymbolType::Object,
            data,
            0,
            16,
        ));
        m.add_symbol(Symbol::defined(
            "w_cache",
            SymbolBinding::Weak,
            SymbolType::Object,
            bss,
            0,
            256,
        ));

        let b = write(&m);
        let shdrs = parse_shdrs(&b);
        let bss_shdr = shdrs
            .iter()
            .find(|s| s.kind == SHT_NOBITS)
            .expect(".bss present");
        assert_eq!(bss_shdr.size, 256);

        let symtab = shdrs.iter().find(|s| s.kind == SHT_SYMTAB).unwrap();
        // 1 local user symbol -> first global at index 2 (null + local).
        assert_eq!(symtab.info, 2);
        // All local-binding symbols must appear before the first global.
        let symbase = symtab.offset as usize;
        for i in 1..symtab.info as usize {
            let info = b[symbase + i * SYM_SIZE as usize + 4];
            assert_eq!(info >> 4, STB_LOCAL, "symbol {i} should be local");
        }
    }

    #[test]
    fn output_is_deterministic() {
        let a = write(&tiny_object());
        let b = write(&tiny_object());
        assert_eq!(a, b);
    }

    // ---- optional external cross-check with readelf / llvm-readobj ----

    fn tool_available(cmd: &str) -> bool {
        std::process::Command::new(cmd)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[test]
    fn readelf_parses_our_object() {
        let tool = if tool_available("readelf") {
            "readelf"
        } else if tool_available("llvm-readobj") {
            "llvm-readobj"
        } else {
            eprintln!("skipping: neither readelf nor llvm-readobj is available");
            return;
        };

        let bytes = write(&tiny_object());
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lf_elf_test_{}.o", std::process::id()));
        std::fs::write(&path, &bytes).expect("write temp object");

        let arg = if tool == "readelf" { "-a" } else { "--all" };
        let output = std::process::Command::new(tool)
            .arg(arg)
            .arg(&path)
            .output()
            .expect("run reader tool");
        let _ = std::fs::remove_file(&path);

        assert!(
            output.status.success(),
            "{tool} rejected our object: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        // The tool must recognize the type/machine and see our symbol + reloc.
        assert!(stdout.contains("REL") || stdout.contains("Relocatable"));
        assert!(stdout.contains("main"));
        assert!(stdout.contains("puts"));
    }
}
