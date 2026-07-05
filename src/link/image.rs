//! The static-executable linker core (ROADMAP Phase 8).
//!
//! Consumes one or more relocatable [`ObjectModule`]s and produces a **static
//! ELF64 executable** (`ET_EXEC`) for x86-64 Linux, with no dependence on a
//! system linker, a dynamic loader, or libc. The three jobs of a linker are all
//! here:
//!
//! 1. **Symbol resolution** ([`resolve_globals`]): a global table is built
//!    across every input object; each undefined reference is bound to a defined
//!    global/weak definition. A duplicate *strong* definition or an unresolved
//!    *strong* reference is an error; a weak definition yields to a strong one,
//!    and an unresolved *weak* reference resolves to address 0.
//! 2. **Section layout**: same-kind sections (`.text`/`.rodata`/`.data`/`.bss`)
//!    are concatenated across objects with alignment, assigned virtual
//!    addresses, and grouped into `PT_LOAD` segments with the right permission
//!    flags. The image keeps the invariant `vaddr == base + file_offset`, which
//!    trivially satisfies the ELF rule `p_offset ≡ p_vaddr (mod page)`.
//! 3. **Relocation processing**: with every address now known, each generic
//!    [`RelocKind`] is applied in place (`Abs64` = `S+A`, `Pc32`/`Plt32` =
//!    `S+A-P`, …). In a static link a `Plt32` to a defined symbol reduces to a
//!    plain PC-relative reference.
//!
//! The **entry point** is a synthesized `_start` (emitted from the well-known
//! x86-64 bytes) that calls the entry symbol — `main` by default — and passes
//! its return value to the `exit` syscall. `e_entry` is set to `_start`.
//!
//! The ELF64 executable is written from the published specification (tenet T1):
//! an `Elf64_Ehdr`, a program-header table, then the segment contents. Section
//! headers are omitted — they are optional for an executable and a static image
//! runs without them. Output is deterministic: layout follows the objects'
//! insertion order, and the resolution maps are used only for lookup.

use crate::mc::emit::Emitter;
use crate::mc::object::{
    ObjectModule, RelocKind, SectionKind, Symbol, SymbolBinding, SymbolId, SymbolType, SymbolValue,
};
use crate::support::hash::DetHashMap;

/// The default virtual load address of an `ET_EXEC` image.
const BASE_DEFAULT: u64 = 0x40_0000;
/// The page size used for segment alignment.
const PAGE: u64 = 0x1000;

const EHDR_SIZE: u64 = 64;
const PHDR_SIZE: u64 = 56;

// ELF constants (from the ELF-64 specification).
const ELFMAG: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EV_CURRENT: u8 = 1;
const ELFOSABI_SYSV: u8 = 0;
const ET_EXEC: u16 = 2;
const EM_X86_64: u16 = 62;

const PT_LOAD: u32 = 1;
const PF_X: u32 = 0x1;
const PF_W: u32 = 0x2;
const PF_R: u32 = 0x4;

/// Options controlling how an executable image is built.
#[derive(Clone, Debug)]
pub struct ImageOptions {
    /// The symbol the synthesized `_start` transfers control to (default
    /// `main`). Ignored if an input object already defines `_start`.
    pub entry: String,
    /// The virtual base address of the image.
    pub base: u64,
}

impl Default for ImageOptions {
    fn default() -> Self {
        ImageOptions { entry: "main".to_owned(), base: BASE_DEFAULT }
    }
}

/// Why a link failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LinkError {
    /// No input objects were supplied.
    NoInput,
    /// Two objects both provide a strong definition of the named symbol.
    DuplicateSymbol(String),
    /// A non-weak reference to the named symbol has no definition.
    UnresolvedSymbol(String),
    /// No `_start` symbol exists and none could be resolved for the entry.
    MissingEntry(String),
    /// A relocation kind that a static link cannot lower (e.g. one needing a
    /// GOT/PLT that this static linker does not synthesize).
    UnsupportedReloc(RelocKind),
    /// A relocated value did not fit its field.
    RelocOverflow {
        /// The symbol whose address was being applied.
        symbol: String,
        /// The image virtual address of the field.
        at: u64,
    },
}

impl std::fmt::Display for LinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkError::NoInput => write!(f, "no input objects"),
            LinkError::DuplicateSymbol(n) => write!(f, "duplicate strong definition of '{n}'"),
            LinkError::UnresolvedSymbol(n) => write!(f, "undefined reference to '{n}'"),
            LinkError::MissingEntry(n) => write!(f, "no entry symbol: '{n}' is undefined"),
            LinkError::UnsupportedReloc(k) => {
                write!(f, "unsupported relocation in static link: {k:?}")
            }
            LinkError::RelocOverflow { symbol, at } => {
                write!(f, "relocation of '{symbol}' at {at:#x} does not fit its field")
            }
        }
    }
}

impl std::error::Error for LinkError {}

/// The winning global/weak definition of a symbol.
#[derive(Clone, Copy, Debug)]
struct GlobalDef {
    obj: usize,
    sym: SymbolId,
    strong: bool,
}

/// Build the cross-object table of defined global/weak symbols.
///
/// A strong (global) definition overrides a weak one; two strong definitions of
/// the same name are a [`LinkError::DuplicateSymbol`]. Iteration is over the
/// objects' insertion order, so the choice among equally-weak definitions (first
/// wins) is deterministic.
fn resolve_globals(objects: &[ObjectModule]) -> Result<DetHashMap<String, GlobalDef>, LinkError> {
    let mut map: DetHashMap<String, GlobalDef> = DetHashMap::default();
    for (oi, obj) in objects.iter().enumerate() {
        for (si, sym) in obj.symbols().iter().enumerate() {
            if sym.is_undefined() || matches!(sym.binding, SymbolBinding::Local) {
                continue;
            }
            let strong = matches!(sym.binding, SymbolBinding::Global);
            let def = GlobalDef { obj: oi, sym: SymbolId::from_index(si), strong };
            match map.get(&sym.name) {
                None => {
                    map.insert(sym.name.clone(), def);
                }
                Some(existing) => {
                    if existing.strong && strong {
                        return Err(LinkError::DuplicateSymbol(sym.name.clone()));
                    }
                    // A strong definition displaces a weak one; otherwise keep
                    // the existing (first) definition.
                    if !existing.strong && strong {
                        map.insert(sym.name.clone(), def);
                    }
                }
            }
        }
    }
    Ok(map)
}

/// A placed section: its assigned virtual address, keyed by `(object, section)`.
type Placement = DetHashMap<(usize, usize), u64>;

/// One `PT_LOAD` segment record.
#[derive(Clone, Copy, Debug)]
struct Segment {
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
    flags: u32,
}

/// Round `x` up to a multiple of `align` (a power of two, or `1`).
fn align_up(x: u64, align: u64) -> u64 {
    if align <= 1 {
        return x;
    }
    let mask = align - 1;
    (x + mask) & !mask
}

/// The `(object, section)` pairs of a given kind, in object then section order.
fn group(objects: &[ObjectModule], kind: SectionKind) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for (oi, obj) in objects.iter().enumerate() {
        for (si, s) in obj.sections().iter().enumerate() {
            if s.kind == kind {
                v.push((oi, si));
            }
        }
    }
    v
}

/// Does any input object provide a definition of `_start`?
fn defines_start(objects: &[ObjectModule]) -> bool {
    objects.iter().any(|obj| {
        obj.symbols()
            .iter()
            .any(|s| s.name == "_start" && !s.is_undefined() && !matches!(s.binding, SymbolBinding::Local))
    })
}

/// Synthesize a minimal `crt0` object defining `_start`:
///
/// ```text
///   xor  ebp, ebp        ; 31 ed          — mark the outermost frame
///   call <entry>         ; e8 rel32       — relocation to the entry symbol
///   mov  edi, eax        ; 89 c7          — status = entry()'s return value
///   mov  eax, 60         ; b8 3c 00 00 00 — __NR_exit
///   syscall              ; 0f 05
/// ```
///
/// At process entry `rsp` is 16-aligned; the `call` pushes the (unused) return
/// address, so `entry` is reached with `rsp ≡ 8 (mod 16)` exactly as the SysV
/// ABI requires of a callee.
fn synth_start(entry: &str) -> ObjectModule {
    let mut e = Emitter::new();
    e.u8(0x31);
    e.u8(0xed); // xor ebp, ebp
    e.u8(0xe8); // call rel32
    e.reference_symbol(RelocKind::Plt32, entry.to_owned(), 0);
    e.u8(0x89);
    e.u8(0xc7); // mov edi, eax
    e.u8(0xb8);
    e.u32(60); // mov eax, 60
    e.u8(0x0f);
    e.u8(0x05); // syscall
    let emitted = e.finish().expect("crt0 has no unbound labels");
    let len = emitted.bytes.len() as u64;

    let mut obj = ObjectModule::new("<crt0>");
    let text = obj.add_emitted_section(".text", SectionKind::Text, 16, emitted);
    obj.add_symbol(Symbol::defined(
        "_start",
        SymbolBinding::Global,
        SymbolType::Func,
        text,
        0,
        len,
    ));
    obj
}

/// The address of a resolved global/weak definition.
fn def_address(objects: &[ObjectModule], placement: &Placement, def: GlobalDef) -> u64 {
    match objects[def.obj].symbol(def.sym).value {
        SymbolValue::Defined { section, offset } => {
            placement[&(def.obj, section.index())] + offset
        }
        SymbolValue::Undefined => unreachable!("a GlobalDef is always a definition"),
    }
}

/// Resolve the runtime address a relocation's target symbol denotes.
fn symbol_address(
    objects: &[ObjectModule],
    placement: &Placement,
    globals: &DetHashMap<String, GlobalDef>,
    obj: usize,
    sym: SymbolId,
) -> Result<u64, LinkError> {
    let s = objects[obj].symbol(sym);
    match s.value {
        SymbolValue::Defined { section, offset } => match s.binding {
            // A file-local definition is fixed within its own object.
            SymbolBinding::Local => Ok(placement[&(obj, section.index())] + offset),
            // A global/weak definition resolves through the global table so all
            // references agree on the single winning definition.
            SymbolBinding::Global | SymbolBinding::Weak => {
                let def = globals.get(&s.name).copied().expect("its own definition is present");
                Ok(def_address(objects, placement, def))
            }
        },
        SymbolValue::Undefined => match globals.get(&s.name).copied() {
            Some(def) => Ok(def_address(objects, placement, def)),
            // An unresolved weak reference is defined to be address 0.
            None if matches!(s.binding, SymbolBinding::Weak) => Ok(0),
            None => Err(LinkError::UnresolvedSymbol(s.name.clone())),
        },
    }
}

/// Write a little-endian value of `n` bytes into `buf` at `off`.
fn put(buf: &mut [u8], off: usize, bytes: &[u8]) {
    buf[off..off + bytes.len()].copy_from_slice(bytes);
}

/// Link `objects` into a static ELF64 executable image for x86-64 Linux.
///
/// If no input object defines `_start`, a `crt0` calling `opts.entry` is
/// synthesized and placed first. Returns the complete executable bytes.
pub fn link_executable(
    mut objects: Vec<ObjectModule>,
    opts: &ImageOptions,
) -> Result<Vec<u8>, LinkError> {
    if objects.is_empty() {
        return Err(LinkError::NoInput);
    }

    // 1. Provide an entry stub unless the program brings its own.
    if !defines_start(&objects) {
        objects.insert(0, synth_start(&opts.entry));
    }

    // 2. Resolve global symbols across all objects.
    let globals = resolve_globals(&objects)?;

    // 3. Determine which segments are present.
    let g_text = group(&objects, SectionKind::Text);
    let g_rodata = group(&objects, SectionKind::Rodata);
    let g_data = group(&objects, SectionKind::Data);
    let g_bss = group(&objects, SectionKind::Bss);

    let size_of = |g: &[(usize, usize)]| -> u64 {
        g.iter().map(|&(o, s)| objects[o].sections()[s].size()).sum()
    };
    let rodata_present = size_of(&g_rodata) > 0;
    let data_present = size_of(&g_data) > 0 || size_of(&g_bss) > 0;

    let phnum = 1 + u64::from(rodata_present) + u64::from(data_present);
    let headers_size = EHDR_SIZE + phnum * PHDR_SIZE;

    let base = opts.base;
    let mut buf: Vec<u8> = vec![0u8; headers_size as usize];
    let mut placement: Placement = DetHashMap::default();
    let mut segments: Vec<Segment> = Vec::new();

    // Place a group of file-backed sections at the current end of `buf`.
    let place_filebacked = |buf: &mut Vec<u8>, placement: &mut Placement, g: &[(usize, usize)]| {
        for &(oi, si) in g {
            let s = &objects[oi].sections()[si];
            let aligned = align_up(buf.len() as u64, s.align.max(1));
            buf.resize(aligned as usize, 0);
            placement.insert((oi, si), base + buf.len() as u64);
            buf.extend_from_slice(&s.bytes);
        }
    };

    // --- TEXT segment (R+X): headers + all .text, starting at file offset 0. ---
    place_filebacked(&mut buf, &mut placement, &g_text);
    let text_end = buf.len() as u64;
    segments.push(Segment {
        offset: 0,
        vaddr: base,
        filesz: text_end,
        memsz: text_end,
        flags: PF_R | PF_X,
    });

    // --- RODATA segment (R): its own page. ---
    if rodata_present {
        let seg_off = align_up(buf.len() as u64, PAGE);
        buf.resize(seg_off as usize, 0);
        place_filebacked(&mut buf, &mut placement, &g_rodata);
        let end = buf.len() as u64;
        segments.push(Segment {
            offset: seg_off,
            vaddr: base + seg_off,
            filesz: end - seg_off,
            memsz: end - seg_off,
            flags: PF_R,
        });
    }

    // --- DATA segment (R+W): initialized .data (file) then .bss (memory only). ---
    if data_present {
        let seg_off = align_up(buf.len() as u64, PAGE);
        buf.resize(seg_off as usize, 0);
        place_filebacked(&mut buf, &mut placement, &g_data);
        let file_end = buf.len() as u64;

        // `.bss` extends the memory image past the file image; assign each a
        // vaddr (== base + running memory offset) but emit no file bytes.
        let mut mem_cursor = file_end;
        for &(oi, si) in &g_bss {
            let s = &objects[oi].sections()[si];
            mem_cursor = align_up(mem_cursor, s.align.max(1));
            placement.insert((oi, si), base + mem_cursor);
            mem_cursor += s.bss_size;
        }

        segments.push(Segment {
            offset: seg_off,
            vaddr: base + seg_off,
            filesz: file_end - seg_off,
            memsz: mem_cursor - seg_off,
            flags: PF_R | PF_W,
        });
    }

    // 4. Apply relocations now that every section has an address.
    for (oi, obj) in objects.iter().enumerate() {
        for r in obj.relocations() {
            let sec_vaddr = placement[&(oi, r.section.index())];
            let p = sec_vaddr + r.offset;
            let file_off = (p - base) as usize;
            let s = symbol_address(&objects, &placement, &globals, oi, r.symbol)?;
            let a = r.addend;
            let name = || obj.symbol(r.symbol).name.clone();
            match r.kind {
                RelocKind::Abs64 => {
                    let v = s.wrapping_add(a as u64);
                    put(&mut buf, file_off, &v.to_le_bytes());
                }
                RelocKind::Pc64 => {
                    let v = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                    put(&mut buf, file_off, &v.to_le_bytes());
                }
                RelocKind::Abs32 | RelocKind::Abs32S => {
                    let v = (s as i64).wrapping_add(a);
                    if v < i32::MIN as i64 || v > u32::MAX as i64 {
                        return Err(LinkError::RelocOverflow { symbol: name(), at: p });
                    }
                    put(&mut buf, file_off, &(v as u32).to_le_bytes());
                }
                RelocKind::Pc32 | RelocKind::Plt32 => {
                    // A PLT call in a static link is just a PC-relative reference.
                    let v = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                    if v < i32::MIN as i64 || v > i32::MAX as i64 {
                        return Err(LinkError::RelocOverflow { symbol: name(), at: p });
                    }
                    put(&mut buf, file_off, &(v as i32).to_le_bytes());
                }
                RelocKind::GotPcRel => return Err(LinkError::UnsupportedReloc(r.kind)),
            }
        }
    }

    // 5. Determine the entry point.
    let entry = globals
        .get("_start")
        .copied()
        .map(|def| def_address(&objects, &placement, def))
        .ok_or_else(|| LinkError::MissingEntry("_start".to_owned()))?;

    // 6. Write the ELF header and program headers into the reserved prefix.
    write_headers(&mut buf, entry, &segments);

    Ok(buf)
}

/// Fill `buf`'s reserved prefix with the `Elf64_Ehdr` and the program headers.
fn write_headers(buf: &mut [u8], entry: u64, segments: &[Segment]) {
    // e_ident.
    put(buf, 0, &ELFMAG);
    buf[4] = ELFCLASS64;
    buf[5] = ELFDATA2LSB;
    buf[6] = EV_CURRENT;
    buf[7] = ELFOSABI_SYSV;
    // bytes 8..16 are the padding, already zero.

    let phnum = segments.len() as u16;
    put(buf, 16, &ET_EXEC.to_le_bytes()); // e_type
    put(buf, 18, &EM_X86_64.to_le_bytes()); // e_machine
    put(buf, 20, &1u32.to_le_bytes()); // e_version
    put(buf, 24, &entry.to_le_bytes()); // e_entry
    put(buf, 32, &EHDR_SIZE.to_le_bytes()); // e_phoff
    put(buf, 40, &0u64.to_le_bytes()); // e_shoff (no section headers)
    put(buf, 48, &0u32.to_le_bytes()); // e_flags
    put(buf, 52, &(EHDR_SIZE as u16).to_le_bytes()); // e_ehsize
    put(buf, 54, &(PHDR_SIZE as u16).to_le_bytes()); // e_phentsize
    put(buf, 56, &phnum.to_le_bytes()); // e_phnum
    put(buf, 58, &0u16.to_le_bytes()); // e_shentsize
    put(buf, 60, &0u16.to_le_bytes()); // e_shnum
    put(buf, 62, &0u16.to_le_bytes()); // e_shstrndx

    for (i, seg) in segments.iter().enumerate() {
        let o = (EHDR_SIZE + i as u64 * PHDR_SIZE) as usize;
        put(buf, o, &PT_LOAD.to_le_bytes()); // p_type
        put(buf, o + 4, &seg.flags.to_le_bytes()); // p_flags
        put(buf, o + 8, &seg.offset.to_le_bytes()); // p_offset
        put(buf, o + 16, &seg.vaddr.to_le_bytes()); // p_vaddr
        put(buf, o + 24, &seg.vaddr.to_le_bytes()); // p_paddr
        put(buf, o + 32, &seg.filesz.to_le_bytes()); // p_filesz
        put(buf, o + 40, &seg.memsz.to_le_bytes()); // p_memsz
        put(buf, o + 48, &PAGE.to_le_bytes()); // p_align
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::object::{Relocation, Section};

    fn rd_u16(b: &[u8], o: usize) -> u16 {
        u16::from_le_bytes([b[o], b[o + 1]])
    }
    fn rd_u32(b: &[u8], o: usize) -> u32 {
        u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }
    fn rd_u64(b: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
    }

    /// An object with a `main` that just `ret`s, for structural tests.
    fn ret_object() -> ObjectModule {
        let mut m = ObjectModule::new("t");
        let mut text = Section::new(".text", SectionKind::Text, 16);
        text.bytes = vec![0xc3]; // ret
        let tid = m.add_section(text);
        m.add_symbol(Symbol::defined(
            "main",
            SymbolBinding::Global,
            SymbolType::Func,
            tid,
            0,
            1,
        ));
        m
    }

    #[test]
    fn resolves_and_builds_valid_header() {
        let img = link_executable(vec![ret_object()], &ImageOptions::default()).unwrap();
        assert_eq!(&img[0..4], &ELFMAG);
        assert_eq!(img[4], ELFCLASS64);
        assert_eq!(img[5], ELFDATA2LSB);
        assert_eq!(rd_u16(&img, 16), ET_EXEC);
        assert_eq!(rd_u16(&img, 18), EM_X86_64);
        // At least one PT_LOAD, e_entry within the text segment.
        let phnum = rd_u16(&img, 56);
        assert!(phnum >= 1);
        let entry = rd_u64(&img, 24);
        assert!(entry >= BASE_DEFAULT);
        // The first program header is the R+X text load at offset 0.
        let po = EHDR_SIZE as usize;
        assert_eq!(rd_u32(&img, po), PT_LOAD);
        assert_eq!(rd_u32(&img, po + 4), PF_R | PF_X);
        assert_eq!(rd_u64(&img, po + 8), 0); // p_offset
        assert_eq!(rd_u64(&img, po + 16), BASE_DEFAULT); // p_vaddr
        assert_eq!(rd_u64(&img, po + 48), PAGE); // p_align
    }

    #[test]
    fn segment_offset_congruent_to_vaddr_mod_page() {
        // Give the program a rodata and a data+bss segment too.
        let mut m = ret_object();
        let mut ro = Section::new(".rodata", SectionKind::Rodata, 8);
        ro.bytes = vec![1, 2, 3, 4];
        m.add_section(ro);
        let mut da = Section::new(".data", SectionKind::Data, 8);
        da.bytes = vec![9; 12];
        m.add_section(da);
        m.add_section(Section::bss(".bss", 16, 256));

        let img = link_executable(vec![m], &ImageOptions::default()).unwrap();
        let phnum = rd_u16(&img, 56) as usize;
        assert_eq!(phnum, 3);
        for i in 0..phnum {
            let o = EHDR_SIZE as usize + i * PHDR_SIZE as usize;
            let off = rd_u64(&img, o + 8);
            let vaddr = rd_u64(&img, o + 16);
            let filesz = rd_u64(&img, o + 32);
            let memsz = rd_u64(&img, o + 40);
            assert_eq!(off % PAGE, vaddr % PAGE, "segment {i} violates the page rule");
            assert!(memsz >= filesz);
            // File-backed part must lie within the image.
            assert!(off + filesz <= img.len() as u64);
        }
        // The last segment (data) has memsz > filesz thanks to .bss.
        let last = EHDR_SIZE as usize + (phnum - 1) * PHDR_SIZE as usize;
        assert_eq!(rd_u32(&img, last + 4), PF_R | PF_W);
        assert!(rd_u64(&img, last + 40) > rd_u64(&img, last + 32));
    }

    #[test]
    fn duplicate_strong_symbol_is_an_error() {
        let err = link_executable(vec![ret_object(), ret_object()], &ImageOptions::default())
            .unwrap_err();
        assert_eq!(err, LinkError::DuplicateSymbol("main".to_owned()));
    }

    #[test]
    fn unresolved_strong_reference_is_an_error() {
        // main calls an undefined `helper`.
        let mut m = ObjectModule::new("t");
        let mut text = Section::new(".text", SectionKind::Text, 16);
        text.bytes = vec![0xe8, 0, 0, 0, 0, 0xc3]; // call rel32 helper; ret
        let tid = m.add_section(text);
        m.add_symbol(Symbol::defined("main", SymbolBinding::Global, SymbolType::Func, tid, 0, 6));
        let h = m.reference_symbol("helper");
        m.add_relocation(Relocation {
            section: tid,
            offset: 1,
            symbol: h,
            kind: RelocKind::Plt32,
            addend: -4,
        });
        let err = link_executable(vec![m], &ImageOptions::default()).unwrap_err();
        assert_eq!(err, LinkError::UnresolvedSymbol("helper".to_owned()));
    }

    #[test]
    fn weak_definition_yields_to_strong() {
        // obj A: weak main; obj B: strong main. No duplicate error; strong wins.
        let mut a = ObjectModule::new("a");
        let mut ta = Section::new(".text", SectionKind::Text, 16);
        ta.bytes = vec![0x90, 0xc3];
        let tida = a.add_section(ta);
        a.add_symbol(Symbol::defined("main", SymbolBinding::Weak, SymbolType::Func, tida, 0, 2));

        let img = link_executable(vec![a, ret_object()], &ImageOptions::default());
        assert!(img.is_ok(), "weak+strong must resolve, got {img:?}");
    }

    #[test]
    fn abs64_and_pc32_apply_correctly() {
        // A hand-built object with fully predictable layout: it defines `_start`
        // itself (so no crt0 stub is prepended) and has an Abs64 field and a
        // Pc32 field both pointing at `target`, which lives in .rodata. With two
        // segments (text, rodata) the headers are 64 + 2*56 = 176 bytes and the
        // 16-aligned .text lands at file offset 176.
        let mut m = ObjectModule::new("t");
        let mut text = Section::new(".text", SectionKind::Text, 16);
        // bytes: [0..8) abs64 slot, [8] call opcode, [9..13) pc32 slot, [13] ret
        text.bytes = vec![0; 14];
        text.bytes[8] = 0xe8;
        text.bytes[13] = 0xc3;
        let tid = m.add_section(text);
        let mut ro = Section::new(".rodata", SectionKind::Rodata, 1);
        ro.bytes = vec![0xaa, 0xbb, 0xcc, 0xdd];
        let rid = m.add_section(ro);
        m.add_symbol(Symbol::defined("_start", SymbolBinding::Global, SymbolType::Func, tid, 0, 14));
        let target = m.add_symbol(Symbol::defined(
            "target",
            SymbolBinding::Local,
            SymbolType::Object,
            rid,
            0,
            4,
        ));
        m.add_relocation(Relocation {
            section: tid,
            offset: 0,
            symbol: target,
            kind: RelocKind::Abs64,
            addend: 0,
        });
        m.add_relocation(Relocation {
            section: tid,
            offset: 9,
            symbol: target,
            kind: RelocKind::Pc32,
            addend: -4,
        });

        let img = link_executable(vec![m], &ImageOptions::default()).unwrap();

        // Deterministic layout.
        let headers = EHDR_SIZE + 2 * PHDR_SIZE; // 176, already 16-aligned
        let text_foff = align_up(headers, 16);
        assert_eq!(text_foff, headers);
        let text_end = text_foff + 14;
        let ro_foff = align_up(text_end, PAGE);
        let target_addr = BASE_DEFAULT + ro_foff;

        // Program headers agree with the computed rodata placement.
        let ro_ph = EHDR_SIZE as usize + PHDR_SIZE as usize;
        assert_eq!(rd_u64(&img, ro_ph + 8), ro_foff); // p_offset
        assert_eq!(rd_u64(&img, ro_ph + 16), target_addr); // p_vaddr
        assert_eq!(rd_u32(&img, ro_ph + 4), PF_R); // read-only

        // Abs64 field == S + A == target_addr.
        let abs = rd_u64(&img, text_foff as usize);
        assert_eq!(abs, target_addr);

        // Pc32 field == S + A - P, with A = -4 and P = the field's vaddr.
        let p = BASE_DEFAULT + text_foff + 9;
        let disp = i32::from_le_bytes([
            img[(text_foff + 9) as usize],
            img[(text_foff + 10) as usize],
            img[(text_foff + 11) as usize],
            img[(text_foff + 12) as usize],
        ]);
        assert_eq!(disp as i64, target_addr as i64 - 4 - p as i64);
        // e_entry points at _start (text offset 0).
        assert_eq!(rd_u64(&img, 24), BASE_DEFAULT + text_foff);
    }

    #[test]
    fn deterministic_output() {
        let a = link_executable(vec![ret_object()], &ImageOptions::default()).unwrap();
        let b = link_executable(vec![ret_object()], &ImageOptions::default()).unwrap();
        assert_eq!(a, b);
    }
}
