//! Firmware output: a linked image as a **raw binary** or an **Intel HEX**
//! file, for loaders and flash programmers that take the memory contents
//! rather than an ELF file.
//!
//! The input is the image the static linker core produces
//! ([`link_executable`]): its `PT_LOAD` segments are
//! read back ([`load_segments`]) into [`LoadSegment`]s — each a load address
//! and the bytes the file supplies there. The ELF and program headers that a
//! Linux image maps at the start of its first segment are not memory contents
//! the program uses, so they are left out; `.bss` (memory past a segment's
//! file size) is not part of either output either, as a loader zeroes it.
//! [`link_firmware`] links so that the first byte of code lands exactly at a
//! requested address.
//!
//! - [`to_binary`] lays the segments out from the lowest address, filling
//!   the gaps between them with a fill byte (so the output is the exact memory
//!   image of `[lowest, highest)`).
//! - [`to_ihex`] writes Intel HEX from its published specification: data
//!   records (type `00`) of at most 16 bytes that never cross a 64 KiB
//!   boundary, an extended linear address record (type `04`) whenever the
//!   upper 16 address bits change, an optional start linear address record
//!   (type `05`) for the entry point, and the end-of-file record (type `01`).
//!   Each record is `:` + byte count + 16-bit address + type + data +
//!   checksum (the two's complement of the sum of the preceding bytes), in
//!   upper-case hex, ending in CR LF.

use super::{ImageOptions, link_executable};
use crate::mc::object::ObjectModule;

/// The bytes a file supplies at one load address.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LoadSegment {
    /// The address of `data[0]`.
    pub addr: u64,
    /// The contents.
    pub data: Vec<u8>,
}

/// A firmware output format.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RawFormat {
    /// The raw memory image (see [`to_binary`]).
    Binary,
    /// Intel HEX (see [`to_ihex`]).
    Ihex,
}

impl RawFormat {
    /// Parse `binary` or `ihex` (GNU `--oformat` spellings).
    pub fn parse(s: &str) -> Option<RawFormat> {
        match s {
            "binary" | "bin" => Some(RawFormat::Binary),
            "ihex" | "hex" => Some(RawFormat::Ihex),
            _ => None,
        }
    }
}

fn u16_at(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn u64_at(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}

/// The file-backed contents of every `PT_LOAD` segment of a little-endian
/// ELF executable (32- or 64-bit), at their physical load addresses, without
/// the ELF and program headers a segment may map. Empty segments are dropped;
/// the result is sorted by address.
///
/// # Errors
///
/// Returns a message when `elf` is not a well-formed little-endian ELF file.
pub fn load_segments(elf: &[u8]) -> Result<Vec<LoadSegment>, String> {
    const PT_LOAD: u32 = 1;
    let bad = || "not a well-formed little-endian ELF executable".to_owned();
    if elf.len() < 52 || elf[0..4] != [0x7f, b'E', b'L', b'F'] || elf[5] != 1 {
        return Err(bad());
    }
    let is64 = match elf[4] {
        1 => false,
        2 => true,
        _ => return Err(bad()),
    };
    let (phoff, phentsize, phnum) = if is64 {
        (u64_at(elf, 32).ok_or_else(bad)?, u16_at(elf, 54).ok_or_else(bad)?, u16_at(elf, 56).ok_or_else(bad)?)
    } else {
        (u64::from(u32_at(elf, 28).ok_or_else(bad)?), u16_at(elf, 42).ok_or_else(bad)?, u16_at(elf, 44).ok_or_else(bad)?)
    };
    let ehsize = u64::from(u16_at(elf, if is64 { 52 } else { 40 }).ok_or_else(bad)?);
    // The headers occupy the front of the file up to the end of the program
    // header table (which follows the ELF header).
    let headers_end = ehsize.max(phoff + u64::from(phentsize) * u64::from(phnum));

    let mut out = Vec::new();
    for i in 0..u64::from(phnum) {
        let o = usize::try_from(phoff + i * u64::from(phentsize)).map_err(|_| bad())?;
        let (ty, offset, paddr, filesz) = if is64 {
            (u32_at(elf, o).ok_or_else(bad)?, u64_at(elf, o + 8).ok_or_else(bad)?, u64_at(elf, o + 24).ok_or_else(bad)?, u64_at(elf, o + 32).ok_or_else(bad)?)
        } else {
            (
                u32_at(elf, o).ok_or_else(bad)?,
                u64::from(u32_at(elf, o + 4).ok_or_else(bad)?),
                u64::from(u32_at(elf, o + 12).ok_or_else(bad)?),
                u64::from(u32_at(elf, o + 16).ok_or_else(bad)?),
            )
        };
        if ty != PT_LOAD || filesz == 0 {
            continue;
        }
        // Skip a leading part of the segment that is the headers themselves.
        let skip = headers_end.saturating_sub(offset).min(filesz);
        let start = usize::try_from(offset + skip).map_err(|_| bad())?;
        let end = usize::try_from(offset + filesz).map_err(|_| bad())?;
        let data = elf.get(start..end).ok_or_else(bad)?.to_vec();
        if !data.is_empty() {
            out.push(LoadSegment { addr: paddr + skip, data });
        }
    }
    out.sort_by_key(|s| s.addr);
    Ok(out)
}

/// A linked firmware image: its contents and its entry point.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Firmware {
    /// The loaded contents, sorted by address.
    pub segments: Vec<LoadSegment>,
    /// The entry point (`_start`).
    pub entry: u64,
}

/// The entry point (`e_entry`) of a little-endian ELF executable.
fn elf_entry(elf: &[u8]) -> Option<u64> {
    match elf.get(4)? {
        1 => u32_at(elf, 24).map(u64::from),
        2 => u64_at(elf, 24),
        _ => None,
    }
}

/// Link `objects` with the static linker core and return the loaded image
/// (see [`load_segments`]), placed so that the first byte of code (the start
/// of the first input's `.text`, i.e. the entry stub or the program's own
/// `_start`) is at `opts.base` rather than the ELF headers.
///
/// # Errors
///
/// A link error, or a base address too low to hold the headers the image
/// layout reserves in front of the code.
pub fn link_firmware(objects: Vec<ObjectModule>, opts: &ImageOptions) -> Result<Firmware, String> {
    // The headers' size depends only on how many segments the image has, so
    // a first link measures it and a second one shifts the base down by it.
    let probe = link_executable(objects.clone(), opts).map_err(|e| e.to_string())?;
    let first = load_segments(&probe)?;
    let code = first.first().map_or(opts.base, |s| s.addr);
    let header_bytes = code - opts.base;
    let base = opts.base.checked_sub(header_bytes).ok_or_else(|| {
        format!(
            "base address {:#x} is below the {header_bytes} bytes of headers the image reserves",
            opts.base
        )
    })?;
    let shifted = ImageOptions { base, ..opts.clone() };
    let image = link_executable(objects, &shifted).map_err(|e| e.to_string())?;
    let entry = elf_entry(&image).ok_or("the linked image has no ELF header")?;
    Ok(Firmware { segments: load_segments(&image)?, entry })
}

/// The raw memory image of `segments`: from the lowest address to the end of
/// the highest segment, gaps filled with `fill`. Returns the image's start
/// address with its bytes (an empty image for no segments starts at 0).
///
/// # Errors
///
/// Returns a message when two segments overlap.
pub fn to_binary(segments: &[LoadSegment], fill: u8) -> Result<(u64, Vec<u8>), String> {
    let mut sorted: Vec<&LoadSegment> = segments.iter().filter(|s| !s.data.is_empty()).collect();
    sorted.sort_by_key(|s| s.addr);
    let Some(first) = sorted.first() else { return Ok((0, Vec::new())) };
    let base = first.addr;
    let mut out: Vec<u8> = Vec::new();
    for s in sorted {
        let at = usize::try_from(s.addr - base).map_err(|_| "image too large".to_owned())?;
        if at < out.len() {
            return Err(format!("segments overlap at address {:#x}", s.addr));
        }
        out.resize(at, fill);
        out.extend_from_slice(&s.data);
    }
    Ok((base, out))
}

/// Append one Intel HEX record.
fn ihex_record(out: &mut String, addr16: u16, ty: u8, data: &[u8]) {
    use std::fmt::Write;
    debug_assert!(data.len() <= 255);
    let mut sum = data.len() as u8;
    sum = sum.wrapping_add((addr16 >> 8) as u8).wrapping_add(addr16 as u8).wrapping_add(ty);
    let _ = write!(out, ":{:02X}{:04X}{:02X}", data.len(), addr16, ty);
    for &b in data {
        let _ = write!(out, "{b:02X}");
        sum = sum.wrapping_add(b);
    }
    let _ = write!(out, "{:02X}\r\n", sum.wrapping_neg());
}

/// `segments` as an Intel HEX file (see the [module docs](self)), with a
/// start linear address record for `entry` if given.
///
/// # Errors
///
/// Returns a message when an address does not fit in 32 bits.
pub fn to_ihex(segments: &[LoadSegment], entry: Option<u64>) -> Result<String, String> {
    const MAX_DATA: usize = 16;
    let mut out = String::new();
    let mut upper: u32 = 0; // the current extended linear address (bits 31:16)
    let mut sorted: Vec<&LoadSegment> = segments.iter().collect();
    sorted.sort_by_key(|s| s.addr);
    for s in sorted {
        let end = s.addr + s.data.len() as u64;
        if end > 1 << 32 {
            return Err(format!("address {:#x} does not fit Intel HEX's 32 bits", end - 1));
        }
        let mut addr = s.addr as u32;
        let mut data = &s.data[..];
        while !data.is_empty() {
            let hi = addr >> 16;
            if hi != upper {
                ihex_record(&mut out, 0, 0x04, &(hi as u16).to_be_bytes());
                upper = hi;
            }
            // Stop at the next 64 KiB boundary so the offset never wraps.
            let to_boundary = 0x1_0000 - (addr & 0xffff) as usize;
            let n = data.len().min(MAX_DATA).min(to_boundary);
            ihex_record(&mut out, addr as u16, 0x00, &data[..n]);
            data = &data[n..];
            addr = addr.wrapping_add(n as u32);
        }
    }
    if let Some(e) = entry {
        let e = u32::try_from(e).map_err(|_| format!("entry point {e:#x} does not fit 32 bits"))?;
        ihex_record(&mut out, 0, 0x05, &e.to_be_bytes());
    }
    ihex_record(&mut out, 0, 0x01, &[]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(addr: u64, data: &[u8]) -> LoadSegment {
        LoadSegment { addr, data: data.to_vec() }
    }

    /// Independently decode an Intel HEX file: verify every record's checksum
    /// and length, and rebuild `(address, byte)` pairs plus the entry.
    fn decode_ihex(text: &str) -> (Vec<(u32, u8)>, Option<u32>) {
        let mut bytes = Vec::new();
        let mut upper = 0u32;
        let mut entry = None;
        let mut saw_eof = false;
        for line in text.split("\r\n").filter(|l| !l.is_empty()) {
            assert!(!saw_eof, "record after EOF");
            let raw: Vec<u8> = (1..line.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
                .collect();
            assert!(line.starts_with(':'));
            let total: u32 = raw.iter().map(|&b| u32::from(b)).sum();
            assert_eq!(total & 0xff, 0, "checksum of {line}");
            let n = raw[0] as usize;
            assert_eq!(raw.len(), n + 5, "length of {line}");
            let off = u32::from(u16::from_be_bytes([raw[1], raw[2]]));
            let data = &raw[4..4 + n];
            match raw[3] {
                0x00 => {
                    for (k, &b) in data.iter().enumerate() {
                        bytes.push(((upper << 16) + off + k as u32, b));
                    }
                }
                0x01 => saw_eof = true,
                0x04 => upper = u32::from(u16::from_be_bytes([data[0], data[1]])),
                0x05 => entry = Some(u32::from_be_bytes(data.try_into().unwrap())),
                t => panic!("unexpected record type {t}"),
            }
        }
        assert!(saw_eof, "no EOF record");
        (bytes, entry)
    }

    #[test]
    fn ihex_exact_records() {
        // The classic example: 4 bytes at 0x0100, then EOF.
        let out = to_ihex(&[seg(0x100, &[0x02, 0x33, 0x7a, 0x0f])], None).unwrap();
        // Sum 04+01+00+00+02+33+7A+0F = 0xC3 -> checksum 0x3D.
        assert_eq!(out, ":0401000002337A0F3D\r\n:00000001FF\r\n");
    }

    #[test]
    fn ihex_extended_linear_address_and_entry() {
        let out = to_ihex(&[seg(0x0800_0000, &[0xaa, 0xbb])], Some(0x0800_0000)).unwrap();
        assert_eq!(
            out,
            ":020000040800F2\r\n:02000000AABB99\r\n:0400000508000000EF\r\n:00000001FF\r\n"
        );
    }

    #[test]
    fn ihex_splits_at_16_bytes_and_64k_boundaries() {
        let data: Vec<u8> = (0..40u8).collect();
        // Starts 8 bytes below a 64 KiB boundary.
        let out = to_ihex(&[seg(0x1_fff8, &data)], None).unwrap();
        let lines: Vec<&str> = out.split("\r\n").filter(|l| !l.is_empty()).collect();
        assert_eq!(lines[0], ":020000040001F9");
        assert!(lines[1].starts_with(":08FFF800"), "{}", lines[1]);
        assert_eq!(lines[2], ":020000040002F8");
        assert!(lines[3].starts_with(":10000000"));
        assert!(lines[4].starts_with(":10001000"));
        assert_eq!(lines.last(), Some(&":00000001FF"));
        let (bytes, _) = decode_ihex(&out);
        let want: Vec<(u32, u8)> = (0..40u32).map(|k| (0x1_fff8 + k, k as u8)).collect();
        assert_eq!(bytes, want);
        assert!(to_ihex(&[seg(0xffff_ffff, &[1, 2])], None).is_err());
    }

    #[test]
    fn binary_fills_gaps() {
        let (base, bin) = to_binary(&[seg(0x1004, &[5, 6]), seg(0x1000, &[1, 2, 3])], 0xff).unwrap();
        assert_eq!(base, 0x1000);
        assert_eq!(bin, [1, 2, 3, 0xff, 5, 6]);
        assert!(to_binary(&[seg(0, &[1, 2]), seg(1, &[3])], 0).is_err());
        assert_eq!(to_binary(&[], 0).unwrap(), (0, Vec::new()));
    }

    /// `_start: mov eax, 42; ret` plus a `.data` word, linked for firmware.
    fn tiny_program() -> Vec<ObjectModule> {
        use crate::mc::object::{Section, SectionKind, Symbol, SymbolBinding, SymbolType};
        let mut m = ObjectModule::new("fw");
        let t = m.add_section(Section::new(".text", SectionKind::Text, 16));
        m.section_mut(t).bytes = vec![0xb8, 42, 0, 0, 0, 0xc3];
        m.add_symbol(Symbol::defined("_start", SymbolBinding::Global, SymbolType::Func, t, 0, 6));
        let d = m.add_section(Section::new(".data", SectionKind::Data, 4));
        m.section_mut(d).bytes = vec![0xde, 0xad, 0xbe, 0xef];
        m.add_symbol(Symbol::defined("word", SymbolBinding::Global, SymbolType::Object, d, 0, 4));
        vec![m]
    }

    #[test]
    fn firmware_places_code_at_the_base() {
        let opts = ImageOptions { base: 0x0800_0000, ..ImageOptions::default() };
        let fw = link_firmware(tiny_program(), &opts).unwrap();
        assert_eq!(fw.entry, 0x0800_0000, "the program's own _start is first");
        let segs = fw.segments;
        assert_eq!(segs[0].addr, 0x0800_0000);
        assert_eq!(&segs[0].data[..6], &[0xb8, 42, 0, 0, 0, 0xc3]);
        let data = segs.iter().find(|s| s.data == [0xde, 0xad, 0xbe, 0xef]).expect(".data");
        assert!(data.addr > 0x0800_0000);

        let (base, bin) = to_binary(&segs, 0).unwrap();
        assert_eq!(base, 0x0800_0000);
        assert_eq!(&bin[..6], &[0xb8, 42, 0, 0, 0, 0xc3]);
        assert_eq!(&bin[(data.addr - base) as usize..][..4], &[0xde, 0xad, 0xbe, 0xef]);

        let hex = to_ihex(&segs, Some(0x0800_0000)).unwrap();
        let (bytes, entry) = decode_ihex(&hex);
        assert_eq!(entry, Some(0x0800_0000));
        for (addr, b) in bytes {
            assert_eq!(bin[(u64::from(addr) - base) as usize], b, "at {addr:#x}");
        }

        let too_low = ImageOptions { base: 0x10, ..ImageOptions::default() };
        assert!(link_firmware(tiny_program(), &too_low).is_err());
    }

    #[test]
    fn load_segments_rejects_garbage() {
        assert!(load_segments(b"not an elf file at all, clearly not one no no no no").is_err());
    }

    /// Cross-check with GNU objcopy when present: converting our Intel HEX
    /// back to binary must reproduce our binary exactly.
    #[test]
    fn objcopy_agrees_with_our_ihex() {
        let ok = std::process::Command::new("objcopy")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if !ok {
            eprintln!("skipping: objcopy not available");
            return;
        }
        let opts = ImageOptions { base: 0x0002_0000, ..ImageOptions::default() };
        let segs = link_firmware(tiny_program(), &opts).unwrap().segments;
        let (_, bin) = to_binary(&segs, 0).unwrap();
        let hex = to_ihex(&segs, None).unwrap();
        let dir = std::env::temp_dir().join(format!("lf-raw-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (h, b) = (dir.join("t.hex"), dir.join("t.bin"));
        std::fs::write(&h, hex).unwrap();
        let st = std::process::Command::new("objcopy")
            .args(["-I", "ihex", "-O", "binary"])
            .arg(&h)
            .arg(&b)
            .status()
            .unwrap();
        assert!(st.success(), "objcopy rejected our Intel HEX");
        let theirs = std::fs::read(&b).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(theirs, bin);
    }
}
