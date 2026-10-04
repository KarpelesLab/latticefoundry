//! A Mach-O relocatable-object (`MH_OBJECT`) writer for x86-64 and arm64,
//! implemented from Apple's published Mach-O file-format reference
//! (`<mach-o/loader.h>`, `<mach-o/nlist.h>`, `<mach-o/reloc.h>` and the
//! per-architecture relocation headers).
//!
//! This turns the framework's target-independent [`ObjectModule`] into a
//! 64-bit Mach-O object that `ld64`, `ld64.lld` or our own `qld` (ld64
//! flavor) links. It is a clean-room implementation (tenet T1).
//!
//! # What it emits
//!
//! - a `mach_header_64` (`MH_MAGIC_64`, `CPU_TYPE_X86_64` / `CPU_TYPE_ARM64`,
//!   `MH_OBJECT`);
//! - one unnamed `LC_SEGMENT_64` holding a `section_64` per module section:
//!   `.text` → `__TEXT,__text` (pure instructions), `.data` → `__DATA,__data`,
//!   `.rodata` → `__TEXT,__const` (or `__DATA,__const` when it carries
//!   relocations, which `__TEXT` must not), `.bss` → `__DATA,__bss`
//!   (`S_ZEROFILL`, placed after every section with contents), `.debug_*` →
//!   `__DWARF,__debug_*` (`S_ATTR_DEBUG`);
//! - `LC_BUILD_VERSION` (macOS, minimum 11.0 by default — see
//!   [`MachOOptions`]), so linkers know the platform;
//! - `LC_SYMTAB` with `nlist_64` entries in the order `LC_DYSYMTAB` requires —
//!   local symbols, then external definitions, then undefined symbols (the
//!   last two sorted by name) — and `LC_DYSYMTAB` describing those ranges.
//!
//! Every symbol name gets the C ABI's leading underscore (`main` →
//! `_main`), so [`ObjectModule`] names stay the IR names. An ELF-style
//! section symbol (Mach-O has none) becomes a local `ltmp<n>` label at the
//! start of its section. A weak definition carries `N_WEAK_DEF`, a weak
//! reference `N_WEAK_REF`.
//!
//! # Relocations
//!
//! All relocations are `r_extern` (against a symbol-table entry). x86-64
//! keeps every addend in the relocated field; arm64 keeps it in the field
//! only for `UNSIGNED` and otherwise precedes the relocation with an
//! `ARM64_RELOC_ADDEND` carrying it:
//!
//! | [`RelocKind`] | x86-64 | arm64 |
//! |---|---|---|
//! | `Abs64` | `X86_64_RELOC_UNSIGNED` (8 bytes, `A`) | `ARM64_RELOC_UNSIGNED` (8 bytes, `A`) |
//! | `Abs32`, `Abs32S` | `X86_64_RELOC_UNSIGNED` (4 bytes) | `ARM64_RELOC_UNSIGNED` (4 bytes) |
//! | `Pc32` | `X86_64_RELOC_SIGNED` / `SIGNED_1/2/4` (`A + 4`) | — |
//! | `Plt32` | `X86_64_RELOC_BRANCH` (`A + 4`) | — |
//! | `GotPcRel` | `X86_64_RELOC_GOT` (`A + 4`) | — |
//! | `Aarch64Call26` | — | `ARM64_RELOC_BRANCH26` |
//! | `Aarch64AdrPrelPgHi21` | — | `ARM64_RELOC_PAGE21` |
//! | `Aarch64AddAbsLo12Nc` | — | `ARM64_RELOC_PAGEOFF12` |
//!
//! A PC-relative x86-64 field is relative to the end of the instruction; the
//! `SIGNED_n` forms say the instruction ends `n` bytes past the field (an
//! immediate follows it), which is what an ELF addend of `-4 - n` means, so the
//! writer picks `SIGNED_1/2/4` for addends `-5/-6/-8` and plain `SIGNED`
//! otherwise; either way the field holds `A + 4`. `Pc64` has no Mach-O
//! equivalent, nor do the kinds of the other architecture; each is a clear
//! [`ObjectWriteError`].
//!
//! Output is little-endian and deterministic.

use crate::mc::format::{ObjectWriteError, align_up, log2_align, read_u32, write_u32};
use crate::mc::object::{ObjectModule, RelocKind, SectionKind, SymbolBinding, SymbolType, SymbolValue};

// ===========================================================================
// Constants (from the Mach-O headers)
// ===========================================================================

const MH_MAGIC_64: u32 = 0xfeed_facf;
const MH_OBJECT: u32 = 0x1;
const CPU_ARCH_ABI64: u32 = 0x0100_0000;
/// `CPU_TYPE_X86_64`.
pub const CPU_TYPE_X86_64: u32 = 7 | CPU_ARCH_ABI64;
/// `CPU_TYPE_ARM64`.
pub const CPU_TYPE_ARM64: u32 = 12 | CPU_ARCH_ABI64;
const CPU_SUBTYPE_X86_64_ALL: u32 = 3;
const CPU_SUBTYPE_ARM64_ALL: u32 = 0;

const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x2;
const LC_DYSYMTAB: u32 = 0xb;
const LC_BUILD_VERSION: u32 = 0x32;

const HEADER_SIZE: usize = 32;
const SEGMENT_CMD_SIZE: usize = 72;
const SECTION_SIZE: usize = 80;
const SYMTAB_CMD_SIZE: usize = 24;
const DYSYMTAB_CMD_SIZE: usize = 80;
const BUILD_VERSION_CMD_SIZE: usize = 24;
const NLIST_SIZE: usize = 16;
const RELOC_SIZE: usize = 8;

const VM_PROT_ALL: u32 = 7;

const S_REGULAR: u32 = 0x0;
const S_ZEROFILL: u32 = 0x1;
const S_ATTR_PURE_INSTRUCTIONS: u32 = 0x8000_0000;
const S_ATTR_SOME_INSTRUCTIONS: u32 = 0x0000_0400;
const S_ATTR_DEBUG: u32 = 0x0200_0000;

const N_EXT: u8 = 0x01;
const N_UNDF: u8 = 0x0;
const N_SECT: u8 = 0xe;
const N_WEAK_REF: u16 = 0x0040;
const N_WEAK_DEF: u16 = 0x0080;

const X86_64_RELOC_UNSIGNED: u8 = 0;
const X86_64_RELOC_SIGNED: u8 = 1;
const X86_64_RELOC_BRANCH: u8 = 2;
const X86_64_RELOC_GOT: u8 = 4;
const X86_64_RELOC_SIGNED_1: u8 = 6;
const X86_64_RELOC_SIGNED_2: u8 = 7;
const X86_64_RELOC_SIGNED_4: u8 = 8;

const ARM64_RELOC_UNSIGNED: u8 = 0;
const ARM64_RELOC_BRANCH26: u8 = 2;
const ARM64_RELOC_PAGE21: u8 = 3;
const ARM64_RELOC_PAGEOFF12: u8 = 4;
const ARM64_RELOC_ADDEND: u8 = 10;

/// `PLATFORM_MACOS` in `LC_BUILD_VERSION`.
const PLATFORM_MACOS: u32 = 1;

/// The CPU a Mach-O object is written for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MachOCpu {
    /// x86-64 (`CPU_TYPE_X86_64`).
    X86_64,
    /// arm64 (`CPU_TYPE_ARM64`).
    Arm64,
}

impl MachOCpu {
    /// `(cputype, cpusubtype)`.
    pub fn codes(self) -> (u32, u32) {
        match self {
            MachOCpu::X86_64 => (CPU_TYPE_X86_64, CPU_SUBTYPE_X86_64_ALL),
            MachOCpu::Arm64 => (CPU_TYPE_ARM64, CPU_SUBTYPE_ARM64_ALL),
        }
    }
}

/// Options for [`write_with`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MachOOptions {
    /// The CPU.
    pub cpu: MachOCpu,
    /// The minimum macOS version recorded in `LC_BUILD_VERSION`, as
    /// `(major, minor, patch)`. Defaults to 11.0.0 (the first release with
    /// arm64).
    pub min_os: (u16, u8, u8),
}

impl MachOOptions {
    /// Defaults for `cpu`: macOS 11.0.
    pub fn new(cpu: MachOCpu) -> MachOOptions {
        MachOOptions { cpu, min_os: (11, 0, 0) }
    }
}

// ===========================================================================
// Mapping the neutral model onto Mach-O
// ===========================================================================

/// The `(sectname, segname, flags)` of a module section.
fn section_names(name: &str, kind: SectionKind, has_relocs: bool) -> (String, &'static str, u32) {
    match kind {
        SectionKind::Text => (
            "__text".to_owned(),
            "__TEXT",
            S_REGULAR | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS,
        ),
        SectionKind::Data => ("__data".to_owned(), "__DATA", S_REGULAR),
        // Read-only data with pointers needs fixing at load time, which
        // `__TEXT` does not allow: such data lives in `__DATA,__const`.
        SectionKind::Rodata => {
            ("__const".to_owned(), if has_relocs { "__DATA" } else { "__TEXT" }, S_REGULAR)
        }
        SectionKind::Bss => ("__bss".to_owned(), "__DATA", S_ZEROFILL),
        SectionKind::Debug => {
            let base = name.trim_start_matches('.');
            (format!("__{base}"), "__DWARF", S_REGULAR | S_ATTR_DEBUG)
        }
        // Rejected by `write_with` before this is reached (Mach-O thread-local
        // variables use `__thread_vars` descriptors, not implemented).
        SectionKind::TData => ("__thread_data".to_owned(), "__DATA", S_REGULAR),
        SectionKind::TBss => ("__thread_bss".to_owned(), "__DATA", S_ZEROFILL),
    }
}

/// Copy `s` into a 16-byte, NUL-padded name field (truncating).
fn name16(s: &str) -> [u8; 16] {
    let mut out = [0u8; 16];
    let n = s.len().min(16);
    out[..n].copy_from_slice(&s.as_bytes()[..n]);
    out
}

/// One `relocation_info` record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RelocInfo {
    address: u32,
    symbolnum: u32,
    pcrel: bool,
    length: u8,
    external: bool,
    ty: u8,
}

impl RelocInfo {
    fn encode(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.address.to_le_bytes());
        let word = (self.symbolnum & 0x00ff_ffff)
            | (u32::from(self.pcrel) << 24)
            | (u32::from(self.length & 3) << 25)
            | (u32::from(self.external) << 27)
            | (u32::from(self.ty & 0xf) << 28);
        out.extend_from_slice(&word.to_le_bytes());
    }
}

/// How a relocation's addend reaches the linker.
#[derive(Clone, Copy, Debug)]
enum Addend {
    /// Stored in the 8-byte field.
    Word64(i64),
    /// Stored in the 4-byte field.
    Word32(u32),
    /// Carried by a preceding `ARM64_RELOC_ADDEND` (the instruction's
    /// immediate field is cleared); `mask` selects the immediate bits.
    Arm64Pair { addend: i64, mask: u32 },
}

fn unsupported(kind: RelocKind, cpu: MachOCpu) -> ObjectWriteError {
    let why = match kind {
        RelocKind::Pc64 => " (Mach-O has no 64-bit PC-relative relocation)",
        _ => "",
    };
    ObjectWriteError::new(format!(
        "relocation {kind:?} cannot be expressed in a {cpu:?} Mach-O object{why}"
    ))
}

/// The Mach-O type, `r_pcrel`, `r_length` and addend form of a relocation.
fn map_reloc(kind: RelocKind, addend: i64, cpu: MachOCpu) -> Result<(u8, bool, u8, Addend), ObjectWriteError> {
    let too_big = || {
        ObjectWriteError::new(format!(
            "addend {addend} of a {kind:?} relocation does not fit its Mach-O field"
        ))
    };
    let abs32 = || -> Result<Addend, ObjectWriteError> {
        match kind {
            RelocKind::Abs32 if u32::try_from(addend).is_ok() => Ok(Addend::Word32(addend as u32)),
            RelocKind::Abs32S if i32::try_from(addend).is_ok() => Ok(Addend::Word32(addend as i32 as u32)),
            _ => Err(too_big()),
        }
    };
    let pc_field = || -> Result<Addend, ObjectWriteError> {
        let stored = addend.checked_add(4).filter(|v| i32::try_from(*v).is_ok()).ok_or_else(too_big)?;
        Ok(Addend::Word32(stored as i32 as u32))
    };
    match cpu {
        MachOCpu::X86_64 => match kind {
            RelocKind::Abs64 => Ok((X86_64_RELOC_UNSIGNED, false, 3, Addend::Word64(addend))),
            RelocKind::Abs32 | RelocKind::Abs32S => Ok((X86_64_RELOC_UNSIGNED, false, 2, abs32()?)),
            RelocKind::Pc32 => {
                let ty = match addend {
                    -5 => X86_64_RELOC_SIGNED_1,
                    -6 => X86_64_RELOC_SIGNED_2,
                    -8 => X86_64_RELOC_SIGNED_4,
                    _ => X86_64_RELOC_SIGNED,
                };
                Ok((ty, true, 2, pc_field()?))
            }
            RelocKind::Plt32 => Ok((X86_64_RELOC_BRANCH, true, 2, pc_field()?)),
            RelocKind::GotPcRel => Ok((X86_64_RELOC_GOT, true, 2, pc_field()?)),
            other => Err(unsupported(other, cpu)),
        },
        MachOCpu::Arm64 => {
            let pair = |mask: u32| -> Result<Addend, ObjectWriteError> {
                // `r_symbolnum` is a signed 24-bit field.
                if !(-(1 << 23)..(1 << 23)).contains(&addend) {
                    return Err(too_big());
                }
                Ok(Addend::Arm64Pair { addend, mask })
            };
            match kind {
                RelocKind::Abs64 => Ok((ARM64_RELOC_UNSIGNED, false, 3, Addend::Word64(addend))),
                RelocKind::Abs32 | RelocKind::Abs32S => Ok((ARM64_RELOC_UNSIGNED, false, 2, abs32()?)),
                RelocKind::Aarch64Call26 => Ok((ARM64_RELOC_BRANCH26, true, 2, pair(0x03ff_ffff)?)),
                RelocKind::Aarch64AdrPrelPgHi21 => {
                    Ok((ARM64_RELOC_PAGE21, true, 2, pair((3 << 29) | (0x7ffff << 5))?))
                }
                RelocKind::Aarch64AddAbsLo12Nc => Ok((ARM64_RELOC_PAGEOFF12, false, 2, pair(0xfff << 10)?)),
                other => Err(unsupported(other, cpu)),
            }
        }
    }
}

// ===========================================================================
// Writer
// ===========================================================================

/// One `nlist_64` to write.
struct Nlist {
    name: String,
    ty: u8,
    sect: u8,
    desc: u16,
    value: u64,
}

/// Serialize `obj` as a Mach-O object for `cpu` with the default
/// [`MachOOptions`].
///
/// # Errors
///
/// See [`write_with`].
pub fn write(obj: &ObjectModule, cpu: MachOCpu) -> Result<Vec<u8>, ObjectWriteError> {
    write_with(obj, &MachOOptions::new(cpu))
}

/// Serialize `obj` as a Mach-O `MH_OBJECT` under `opts`.
///
/// # Errors
///
/// Returns an [`ObjectWriteError`] for a relocation the CPU's Mach-O
/// relocation set cannot express (see the [module docs](self)), an addend
/// that does not fit, more than 255 sections, or a relocation in `.bss`.
pub fn write_with(obj: &ObjectModule, opts: &MachOOptions) -> Result<Vec<u8>, ObjectWriteError> {
    let cpu = opts.cpu;
    let sections = obj.sections();
    let n = sections.len();
    if n > 255 {
        return Err(ObjectWriteError::new("a Mach-O object holds at most 255 sections"));
    }
    if let Some(s) = sections.iter().find(|s| s.kind.is_tls()) {
        return Err(ObjectWriteError::new(format!(
            "section {} holds thread-local storage, which this Mach-O writer does not support",
            s.name
        )));
    }

    // --- section order: everything with contents, then the zero-fill ones ---
    let mut order: Vec<usize> = (0..n).filter(|&i| !sections[i].is_nobits()).collect();
    order.extend((0..n).filter(|&i| sections[i].is_nobits()));
    let mut ordinal = vec![0u8; n]; // 1-based n_sect
    for (k, &i) in order.iter().enumerate() {
        ordinal[i] = (k + 1) as u8;
    }

    // --- addresses in the object's single address space ---
    let mut addr = vec![0u64; n];
    let mut cursor = 0u64;
    let mut content_end = 0u64;
    for &i in &order {
        let s = &sections[i];
        cursor = align_up(cursor, s.align.max(1));
        addr[i] = cursor;
        cursor += s.size();
        if !s.is_nobits() {
            content_end = cursor;
        }
    }
    let vmsize = cursor;

    // --- symbols: locals, then external definitions, then undefined ---
    let mut locals: Vec<(usize, Nlist)> = Vec::new();
    let mut extdefs: Vec<(usize, Nlist)> = Vec::new();
    let mut undefs: Vec<(usize, Nlist)> = Vec::new();
    for (i, s) in obj.symbols().iter().enumerate() {
        match s.value {
            SymbolValue::Defined { section, offset } => {
                let si = section.index();
                let (name, binding) = if s.kind == SymbolType::Section {
                    (format!("ltmp{}", ordinal[si] - 1), SymbolBinding::Local)
                } else {
                    (format!("_{}", s.name), s.binding)
                };
                let nl = Nlist {
                    name,
                    ty: N_SECT | if binding == SymbolBinding::Local { 0 } else { N_EXT },
                    sect: ordinal[si],
                    desc: if binding == SymbolBinding::Weak { N_WEAK_DEF } else { 0 },
                    value: addr[si] + offset,
                };
                if binding == SymbolBinding::Local { locals.push((i, nl)) } else { extdefs.push((i, nl)) }
            }
            SymbolValue::Undefined => undefs.push((
                i,
                Nlist {
                    name: format!("_{}", s.name),
                    ty: N_UNDF | N_EXT,
                    sect: 0,
                    desc: if s.binding == SymbolBinding::Weak { N_WEAK_REF } else { 0 },
                    value: 0,
                },
            )),
        }
    }
    extdefs.sort_by(|a, b| a.1.name.cmp(&b.1.name));
    undefs.sort_by(|a, b| a.1.name.cmp(&b.1.name));
    let (nlocal, nextdef, nundef) = (locals.len(), extdefs.len(), undefs.len());
    let mut index_of = vec![0u32; obj.symbols().len()];
    let all: Vec<(usize, Nlist)> = locals.into_iter().chain(extdefs).chain(undefs).collect();
    for (k, (i, _)) in all.iter().enumerate() {
        index_of[*i] = k as u32;
    }

    // --- relocations, with addends placed in the fields or ADDEND pairs ---
    let mut contents: Vec<Vec<u8>> = sections.iter().map(|s| s.bytes.clone()).collect();
    let mut relocs: Vec<Vec<RelocInfo>> = vec![Vec::new(); n];
    for r in obj.relocations() {
        let si = r.section.index();
        if sections[si].is_nobits() {
            return Err(ObjectWriteError::new(format!(
                "relocation in the zero-fill section {}",
                sections[si].name
            )));
        }
        let (ty, pcrel, length, addend) = map_reloc(r.kind, r.addend, cpu)?;
        let at = r.offset as usize;
        let width = 1usize << length;
        if at + width > contents[si].len() {
            return Err(ObjectWriteError::new(format!(
                "relocation at offset {at:#x} runs past the end of section {}",
                sections[si].name
            )));
        }
        let address = u32::try_from(r.offset)
            .ok()
            .filter(|&a| a < 1 << 31)
            .ok_or_else(|| ObjectWriteError::new("relocation offset beyond 2 GiB"))?;
        match addend {
            Addend::Word64(v) => contents[si][at..at + 8].copy_from_slice(&v.to_le_bytes()),
            Addend::Word32(v) => write_u32(&mut contents[si], at, v),
            Addend::Arm64Pair { addend, mask } => {
                let insn = read_u32(&contents[si], at);
                write_u32(&mut contents[si], at, insn & !mask);
                if addend != 0 {
                    relocs[si].push(RelocInfo {
                        address,
                        symbolnum: (addend as u32) & 0x00ff_ffff,
                        pcrel: false,
                        length: 2,
                        external: false,
                        ty: ARM64_RELOC_ADDEND,
                    });
                }
            }
        }
        relocs[si].push(RelocInfo {
            address,
            symbolnum: index_of[r.symbol.index()],
            pcrel,
            length,
            external: true,
            ty,
        });
    }

    // --- string table ---
    let mut strtab: Vec<u8> = vec![0];
    let mut strx = Vec::with_capacity(all.len());
    for (_, nl) in &all {
        strx.push(strtab.len() as u32);
        strtab.extend_from_slice(nl.name.as_bytes());
        strtab.push(0);
    }
    strtab.resize(align_up(strtab.len() as u64, 8) as usize, 0);

    // --- file layout ---
    let sizeofcmds = SEGMENT_CMD_SIZE
        + n * SECTION_SIZE
        + BUILD_VERSION_CMD_SIZE
        + SYMTAB_CMD_SIZE
        + DYSYMTAB_CMD_SIZE;
    let data_start = align_up((HEADER_SIZE + sizeofcmds) as u64, 8);
    // Section contents sit at `data_start + addr`, so file offsets keep the
    // sections' relative placement.
    let mut offset = data_start + content_end;
    let mut reloff = vec![0u32; n];
    for &i in &order {
        if !relocs[i].is_empty() {
            offset = align_up(offset, 8);
            reloff[i] = offset as u32;
            offset += (relocs[i].len() * RELOC_SIZE) as u64;
        }
    }
    let symoff = align_up(offset, 8);
    let stroff = symoff + (all.len() * NLIST_SIZE) as u64;
    if stroff + strtab.len() as u64 > u64::from(u32::MAX) {
        return Err(ObjectWriteError::new("Mach-O object larger than 4 GiB"));
    }

    let mut buf = Vec::with_capacity((stroff as usize) + strtab.len());

    // mach_header_64.
    let (cputype, cpusubtype) = cpu.codes();
    for w in [MH_MAGIC_64, cputype, cpusubtype, MH_OBJECT, 4, sizeofcmds as u32, 0, 0] {
        buf.extend_from_slice(&w.to_le_bytes());
    }

    // LC_SEGMENT_64 + its sections.
    buf.extend_from_slice(&LC_SEGMENT_64.to_le_bytes());
    buf.extend_from_slice(&((SEGMENT_CMD_SIZE + n * SECTION_SIZE) as u32).to_le_bytes());
    buf.extend_from_slice(&[0u8; 16]); // segname: "" in an object
    buf.extend_from_slice(&0u64.to_le_bytes()); // vmaddr
    buf.extend_from_slice(&vmsize.to_le_bytes());
    buf.extend_from_slice(&data_start.to_le_bytes()); // fileoff
    buf.extend_from_slice(&content_end.to_le_bytes()); // filesize
    buf.extend_from_slice(&VM_PROT_ALL.to_le_bytes()); // maxprot
    buf.extend_from_slice(&VM_PROT_ALL.to_le_bytes()); // initprot
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // flags
    for &i in &order {
        let s = &sections[i];
        let (sect, seg, flags) = section_names(&s.name, s.kind, !relocs[i].is_empty());
        buf.extend_from_slice(&name16(&sect));
        buf.extend_from_slice(&name16(seg));
        buf.extend_from_slice(&addr[i].to_le_bytes());
        buf.extend_from_slice(&s.size().to_le_bytes());
        let file_off = if s.is_nobits() { 0 } else { (data_start + addr[i]) as u32 };
        buf.extend_from_slice(&file_off.to_le_bytes());
        buf.extend_from_slice(&log2_align(s.align).to_le_bytes());
        buf.extend_from_slice(&reloff[i].to_le_bytes());
        buf.extend_from_slice(&(relocs[i].len() as u32).to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&[0u8; 12]); // reserved1..3
    }

    // LC_BUILD_VERSION (no tools).
    let (major, minor, patch) = opts.min_os;
    let minos = (u32::from(major) << 16) | (u32::from(minor) << 8) | u32::from(patch);
    for w in [LC_BUILD_VERSION, BUILD_VERSION_CMD_SIZE as u32, PLATFORM_MACOS, minos, 0, 0] {
        buf.extend_from_slice(&w.to_le_bytes());
    }

    // LC_SYMTAB.
    for w in [
        LC_SYMTAB,
        SYMTAB_CMD_SIZE as u32,
        symoff as u32,
        all.len() as u32,
        stroff as u32,
        strtab.len() as u32,
    ] {
        buf.extend_from_slice(&w.to_le_bytes());
    }

    // LC_DYSYMTAB: only the three symbol ranges are used in an object.
    let mut dysym = [0u32; 20];
    dysym[0] = LC_DYSYMTAB;
    dysym[1] = DYSYMTAB_CMD_SIZE as u32;
    dysym[2] = 0; // ilocalsym
    dysym[3] = nlocal as u32;
    dysym[4] = nlocal as u32; // iextdefsym
    dysym[5] = nextdef as u32;
    dysym[6] = (nlocal + nextdef) as u32; // iundefsym
    dysym[7] = nundef as u32;
    for w in dysym {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    debug_assert_eq!(buf.len(), HEADER_SIZE + sizeofcmds);

    // Section contents.
    for &i in &order {
        if !sections[i].is_nobits() {
            buf.resize((data_start + addr[i]) as usize, 0);
            buf.extend_from_slice(&contents[i]);
        }
    }
    buf.resize((data_start + content_end) as usize, 0);

    // Relocations.
    for &i in &order {
        if !relocs[i].is_empty() {
            buf.resize(reloff[i] as usize, 0);
            for r in &relocs[i] {
                r.encode(&mut buf);
            }
        }
    }

    // Symbol table and strings.
    buf.resize(symoff as usize, 0);
    for (k, (_, nl)) in all.iter().enumerate() {
        buf.extend_from_slice(&strx[k].to_le_bytes());
        buf.push(nl.ty);
        buf.push(nl.sect);
        buf.extend_from_slice(&nl.desc.to_le_bytes());
        buf.extend_from_slice(&nl.value.to_le_bytes());
    }
    buf.extend_from_slice(&strtab);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::object::{Relocation, Section, Symbol};

    fn u32_at(b: &[u8], o: usize) -> u32 {
        read_u32(b, o)
    }
    fn u64_at(b: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
    }
    fn str16(b: &[u8]) -> String {
        String::from_utf8(b.iter().copied().take_while(|&c| c != 0).collect()).unwrap()
    }

    /// The load commands as `(cmd, offset)`.
    fn commands(b: &[u8]) -> Vec<(u32, usize)> {
        let mut out = Vec::new();
        let mut o = HEADER_SIZE;
        for _ in 0..u32_at(b, 16) {
            out.push((u32_at(b, o), o));
            o += u32_at(b, o + 4) as usize;
        }
        out
    }

    fn command(b: &[u8], cmd: u32) -> usize {
        commands(b).into_iter().find(|c| c.0 == cmd).expect("load command").1
    }

    #[derive(Debug)]
    struct Sect {
        sectname: String,
        segname: String,
        addr: u64,
        size: u64,
        offset: u32,
        align: u32,
        reloff: u32,
        nreloc: u32,
        flags: u32,
    }

    fn sections(b: &[u8]) -> Vec<Sect> {
        let seg = command(b, LC_SEGMENT_64);
        let n = u32_at(b, seg + 64) as usize;
        (0..n)
            .map(|k| {
                let o = seg + SEGMENT_CMD_SIZE + k * SECTION_SIZE;
                Sect {
                    sectname: str16(&b[o..o + 16]),
                    segname: str16(&b[o + 16..o + 32]),
                    addr: u64_at(b, o + 32),
                    size: u64_at(b, o + 40),
                    offset: u32_at(b, o + 48),
                    align: u32_at(b, o + 52),
                    reloff: u32_at(b, o + 56),
                    nreloc: u32_at(b, o + 60),
                    flags: u32_at(b, o + 64),
                }
            })
            .collect()
    }

    #[derive(Debug, PartialEq)]
    struct Sym {
        name: String,
        ty: u8,
        sect: u8,
        desc: u16,
        value: u64,
    }

    fn symbols(b: &[u8]) -> Vec<Sym> {
        let st = command(b, LC_SYMTAB);
        let (symoff, nsyms, stroff) = (u32_at(b, st + 8) as usize, u32_at(b, st + 12) as usize, u32_at(b, st + 16) as usize);
        (0..nsyms)
            .map(|k| {
                let o = symoff + k * NLIST_SIZE;
                let strx = u32_at(b, o) as usize;
                let end = b[stroff + strx..].iter().position(|&c| c == 0).unwrap();
                Sym {
                    name: String::from_utf8(b[stroff + strx..stroff + strx + end].to_vec()).unwrap(),
                    ty: b[o + 4],
                    sect: b[o + 5],
                    desc: u16::from_le_bytes([b[o + 6], b[o + 7]]),
                    value: u64_at(b, o + 8),
                }
            })
            .collect()
    }

    fn relocs(b: &[u8], s: &Sect) -> Vec<RelocInfo> {
        (0..s.nreloc as usize)
            .map(|k| {
                let o = s.reloff as usize + k * RELOC_SIZE;
                let w = u32_at(b, o + 4);
                RelocInfo {
                    address: u32_at(b, o),
                    symbolnum: w & 0x00ff_ffff,
                    pcrel: (w >> 24) & 1 == 1,
                    length: ((w >> 25) & 3) as u8,
                    external: (w >> 27) & 1 == 1,
                    ty: (w >> 28) as u8,
                }
            })
            .collect()
    }

    fn x86_object() -> ObjectModule {
        let mut m = ObjectModule::new("t.o");
        let bss = m.add_section(Section::bss(".bss", 16, 32));
        let text = m.add_section(Section::new(".text", SectionKind::Text, 16));
        // lea rdi,[rip+msg]; call puts; mov dword [rip+cnt], 1; ret
        m.section_mut(text).bytes = vec![
            0x48, 0x8d, 0x3d, 0, 0, 0, 0, 0xe8, 0, 0, 0, 0, 0xc7, 0x05, 0, 0, 0, 0, 1, 0, 0, 0, 0xc3,
        ];
        let rodata = m.add_section(Section::new(".rodata", SectionKind::Rodata, 1));
        m.section_mut(rodata).bytes = b"hi\0".to_vec();
        let data = m.add_section(Section::new(".data", SectionKind::Data, 8));
        m.section_mut(data).bytes = vec![0; 8];
        m.add_symbol(Symbol::defined("main", SymbolBinding::Global, SymbolType::Func, text, 0, 23));
        let msg = m.add_symbol(Symbol::defined("msg", SymbolBinding::Local, SymbolType::Object, rodata, 0, 3));
        m.add_symbol(Symbol::defined("ptr", SymbolBinding::Global, SymbolType::Object, data, 0, 8));
        let cnt = m.add_symbol(Symbol::defined("cnt", SymbolBinding::Weak, SymbolType::Object, bss, 0, 4));
        let puts = m.reference_symbol("puts");
        m.add_relocation(Relocation { section: text, offset: 3, symbol: msg, kind: RelocKind::Pc32, addend: -4 });
        m.add_relocation(Relocation { section: text, offset: 8, symbol: puts, kind: RelocKind::Plt32, addend: -4 });
        m.add_relocation(Relocation { section: text, offset: 14, symbol: cnt, kind: RelocKind::Pc32, addend: -8 });
        m.add_relocation(Relocation { section: data, offset: 0, symbol: msg, kind: RelocKind::Abs64, addend: 2 });
        m
    }

    #[test]
    fn header_and_load_commands() {
        let b = write(&x86_object(), MachOCpu::X86_64).unwrap();
        assert_eq!(u32_at(&b, 0), MH_MAGIC_64);
        assert_eq!(u32_at(&b, 4), CPU_TYPE_X86_64);
        assert_eq!(u32_at(&b, 8), CPU_SUBTYPE_X86_64_ALL);
        assert_eq!(u32_at(&b, 12), MH_OBJECT);
        let cmds: Vec<u32> = commands(&b).iter().map(|c| c.0).collect();
        assert_eq!(cmds, [LC_SEGMENT_64, LC_BUILD_VERSION, LC_SYMTAB, LC_DYSYMTAB]);
        let bv = command(&b, LC_BUILD_VERSION);
        assert_eq!(u32_at(&b, bv + 8), PLATFORM_MACOS);
        assert_eq!(u32_at(&b, bv + 12), 0x000b_0000, "minos 11.0.0");
    }

    #[test]
    fn sections_map_and_zerofill_goes_last() {
        let b = write(&x86_object(), MachOCpu::X86_64).unwrap();
        let s = sections(&b);
        let names: Vec<(&str, &str)> = s.iter().map(|x| (x.segname.as_str(), x.sectname.as_str())).collect();
        // .rodata has no relocations -> __TEXT,__const; .bss moved last.
        assert_eq!(
            names,
            [("__TEXT", "__text"), ("__TEXT", "__const"), ("__DATA", "__data"), ("__DATA", "__bss")]
        );
        assert_eq!(s[0].flags, S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS);
        assert_eq!(s[0].align, 4);
        assert_eq!(s[3].flags, S_ZEROFILL);
        assert_eq!((s[3].offset, s[3].size), (0, 32));
        // Addresses are ascending and aligned; file offsets track addresses.
        assert_eq!(s[0].addr, 0);
        assert_eq!(s[1].addr, 23);
        assert_eq!(s[2].addr, 32);
        assert_eq!(s[3].addr, 48);
        for x in &s[..3] {
            assert_eq!(x.offset as u64 - s[0].offset as u64, x.addr);
        }
        assert_eq!(&b[s[1].offset as usize..][..3], b"hi\0");
        // Segment: vmsize covers .bss, filesize stops before it.
        let seg = command(&b, LC_SEGMENT_64);
        assert_eq!(u64_at(&b, seg + 32), 80, "vmsize");
        assert_eq!(u64_at(&b, seg + 48), 40, "filesize");
    }

    #[test]
    fn symbols_are_partitioned_and_prefixed() {
        let b = write(&x86_object(), MachOCpu::X86_64).unwrap();
        let syms = symbols(&b);
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        // locals; extdefs sorted; undefs sorted.
        assert_eq!(names, ["_msg", "_cnt", "_main", "_ptr", "_puts"]);
        assert_eq!(syms[0], Sym { name: "_msg".into(), ty: N_SECT, sect: 2, desc: 0, value: 23 });
        assert_eq!(syms[1], Sym { name: "_cnt".into(), ty: N_SECT | N_EXT, sect: 4, desc: N_WEAK_DEF, value: 48 });
        assert_eq!(syms[2].ty, N_SECT | N_EXT);
        assert_eq!(syms[4], Sym { name: "_puts".into(), ty: N_EXT, sect: 0, desc: 0, value: 0 });
        let dy = command(&b, LC_DYSYMTAB);
        let r: Vec<u32> = (0..6).map(|k| u32_at(&b, dy + 8 + 4 * k)).collect();
        assert_eq!(r, [0, 1, 1, 3, 4, 1]);
    }

    #[test]
    fn x86_64_relocations() {
        let b = write(&x86_object(), MachOCpu::X86_64).unwrap();
        let s = sections(&b);
        let text = relocs(&b, &s[0]);
        let r = |address, symbolnum, pcrel, length, ty| RelocInfo { address, symbolnum, pcrel, length, external: true, ty };
        assert_eq!(
            text,
            [
                r(3, 0, true, 2, X86_64_RELOC_SIGNED),
                r(8, 4, true, 2, X86_64_RELOC_BRANCH),
                r(14, 1, true, 2, X86_64_RELOC_SIGNED_4),
            ]
        );
        let t = s[0].offset as usize;
        assert_eq!(&b[t + 3..t + 7], &[0; 4], "A + 4 = 0");
        assert_eq!(&b[t + 14..t + 18], &(-4i32).to_le_bytes(), "A + 4 = -4 with SIGNED_4");
        assert_eq!(&b[t + 18..t + 22], &1u32.to_le_bytes(), "immediate untouched");
        let data = relocs(&b, &s[2]);
        assert_eq!(data, [r(0, 0, false, 3, X86_64_RELOC_UNSIGNED)]);
        assert_eq!(u64_at(&b, s[2].offset as usize), 2);
    }

    #[test]
    fn rodata_with_pointers_moves_to_data_const() {
        let mut m = ObjectModule::new("r");
        let ro = m.add_section(Section::new(".rodata", SectionKind::Rodata, 8));
        m.section_mut(ro).bytes = vec![0; 8];
        let f = m.reference_symbol("f");
        m.add_relocation(Relocation { section: ro, offset: 0, symbol: f, kind: RelocKind::Abs64, addend: 0 });
        let b = write(&m, MachOCpu::X86_64).unwrap();
        let s = sections(&b);
        assert_eq!((s[0].segname.as_str(), s[0].sectname.as_str()), ("__DATA", "__const"));
    }

    #[test]
    fn arm64_relocations_use_addend_pairs() {
        let mut m = ObjectModule::new("a");
        let t = m.add_section(Section::new(".text", SectionKind::Text, 4));
        let words: [u32; 4] = [0x9000_0000, 0x9100_0000, 0x9400_0000, 0xd65f_03c0];
        m.section_mut(t).bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        m.add_symbol(Symbol::defined("f", SymbolBinding::Global, SymbolType::Func, t, 0, 16));
        let g = m.reference_symbol("g");
        let h = m.reference_symbol("h");
        m.add_relocation(Relocation { section: t, offset: 0, symbol: g, kind: RelocKind::Aarch64AdrPrelPgHi21, addend: 16 });
        m.add_relocation(Relocation { section: t, offset: 4, symbol: g, kind: RelocKind::Aarch64AddAbsLo12Nc, addend: 16 });
        m.add_relocation(Relocation { section: t, offset: 8, symbol: h, kind: RelocKind::Aarch64Call26, addend: 0 });
        let b = write(&m, MachOCpu::Arm64).unwrap();
        assert_eq!(u32_at(&b, 4), CPU_TYPE_ARM64);
        let s = sections(&b);
        let rs = relocs(&b, &s[0]);
        // f (extdef 0), g, h (undefs 1, 2).
        let addend = |address, v: u32| RelocInfo { address, symbolnum: v, pcrel: false, length: 2, external: false, ty: ARM64_RELOC_ADDEND };
        let r = |address, symbolnum, pcrel, ty| RelocInfo { address, symbolnum, pcrel, length: 2, external: true, ty };
        assert_eq!(
            rs,
            [
                addend(0, 16),
                r(0, 1, true, ARM64_RELOC_PAGE21),
                addend(4, 16),
                r(4, 1, false, ARM64_RELOC_PAGEOFF12),
                r(8, 2, true, ARM64_RELOC_BRANCH26),
            ]
        );
        let d = s[0].offset as usize;
        assert_eq!(read_u32(&b, d), 0x9000_0000, "adrp immediate cleared");
        assert_eq!(read_u32(&b, d + 8), 0x9400_0000);
    }

    #[test]
    fn section_symbols_become_ltmp_labels() {
        let mut m = ObjectModule::new("s");
        let t = m.add_section(Section::new(".text", SectionKind::Text, 1));
        m.section_mut(t).bytes = vec![0; 8];
        let sec = m.add_symbol(Symbol::defined(".text", SymbolBinding::Local, SymbolType::Section, t, 0, 0));
        m.add_relocation(Relocation { section: t, offset: 0, symbol: sec, kind: RelocKind::Abs64, addend: 4 });
        let b = write(&m, MachOCpu::X86_64).unwrap();
        let syms = symbols(&b);
        assert_eq!(syms[0].name, "ltmp0");
        assert_eq!(syms[0].ty, N_SECT);
    }

    #[test]
    fn unsupported_relocations_are_errors() {
        for (kind, cpu) in [
            (RelocKind::Pc64, MachOCpu::X86_64),
            (RelocKind::Aarch64Call26, MachOCpu::X86_64),
            (RelocKind::Pc32, MachOCpu::Arm64),
            (RelocKind::GotPcRel, MachOCpu::Arm64),
            (RelocKind::ThumbCall, MachOCpu::Arm64),
            (RelocKind::ThumbMovwAbsNc, MachOCpu::X86_64),
            (RelocKind::ThumbMovtAbs, MachOCpu::Arm64),
            (RelocKind::AvrCall, MachOCpu::X86_64),
            (RelocKind::Avr13Pcrel, MachOCpu::Arm64),
            (RelocKind::Avr16Pm, MachOCpu::X86_64),
            (RelocKind::AvrLo8Ldi, MachOCpu::Arm64),
            (RelocKind::AvrHi8Ldi, MachOCpu::X86_64),
            (RelocKind::AvrLo8LdiPm, MachOCpu::Arm64),
            (RelocKind::AvrHi8LdiPm, MachOCpu::X86_64),
        ] {
            let mut m = ObjectModule::new("e");
            let t = m.add_section(Section::new(".text", SectionKind::Text, 1));
            m.section_mut(t).bytes = vec![0; 8];
            let x = m.reference_symbol("x");
            m.add_relocation(Relocation { section: t, offset: 0, symbol: x, kind, addend: 0 });
            let e = write(&m, cpu).unwrap_err();
            assert!(e.message().contains(&format!("{kind:?}")), "{e}");
        }
    }

    #[test]
    fn output_is_deterministic() {
        assert_eq!(write(&x86_object(), MachOCpu::X86_64), write(&x86_object(), MachOCpu::X86_64));
    }
}
