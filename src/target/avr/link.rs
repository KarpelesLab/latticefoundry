//! Linking AVR objects into a flashable firmware image (a Harvard layout).
//!
//! ```text
//! flash (byte addresses)                 SRAM (data addresses)
//! 0x0000  startup: vectors, __init       ram_start  .data, .rodata   (__data_start)
//!         .text of every object                     ...               (__data_end = __bss_start)
//!         .progmem.data                             .bss              (__bss_end)
//!         initial values of .data/.rodata ...
//!         (__data_load_start)                        ↑ stack from RAMEND (__stack)
//! ```
//!
//! The startup object ([`super::startup`]) is placed first, so the vector
//! table sits at address 0; the other objects' code follows in input order,
//! then all flash data, then the load image of the initialized SRAM
//! sections, which the startup code copies to `__data_start`. `.bss` is zeroed
//! by the startup code. Symbols resolve as usual (a strong definition beats a
//! weak one; locals stay per object), and library members — the
//! [runtime](super::runtime) — are pulled in only when they define a symbol
//! still undefined, repeating until nothing more is needed (archive
//! semantics).
//!
//! The relocations applied (`S` = symbol address, `A` = addend, `P` = field
//! address, all byte addresses):
//!
//! | kind | field |
//! |---|---|
//! | `AvrCall` | the 22-bit word address `(S + A) / 2` of `call`/`jmp` |
//! | `Avr13Pcrel` | the 12-bit word displacement `(S + A − (P + 2)) / 2` of `rcall`/`rjmp` |
//! | `Abs16` / `Avr16Pm` | `S + A` / `(S + A) / 2`, 16 bits |
//! | `AvrLo8Ldi` / `AvrHi8Ldi` | bits 0–7 / 8–15 of `S + A` in an `ldi`-form `K` field |
//! | `AvrLo8LdiPm` / `AvrHi8LdiPm` | the same of `(S + A) / 2` |
//! | `Abs32` | `S + A`, 32 bits |

use crate::link::raw::LoadSegment;
use crate::mc::object::{ObjectModule, RelocKind, SectionKind, SymbolBinding, SymbolValue};
use crate::support::DetHashMap;

use super::Device;

/// A linked firmware image.
#[derive(Clone, Debug)]
pub struct Firmware {
    /// The flash contents from address 0.
    pub flash: Vec<u8>,
    /// Every global symbol's address (flash byte address for code and flash
    /// data, data-space address for SRAM data) — for tools and tests.
    pub symbols: DetHashMap<String, u32>,
    /// The first byte past `.bss` in SRAM (where the heap would start).
    pub ram_end: u32,
}

impl Firmware {
    /// The image as a single segment at flash address 0, for
    /// [`crate::link::raw::to_ihex`] / [`crate::link::raw::to_binary`].
    pub fn segments(&self) -> Vec<LoadSegment> {
        vec![LoadSegment { addr: 0, data: self.flash.clone() }]
    }

    /// The image as Intel HEX.
    pub fn to_ihex(&self) -> String {
        crate::link::raw::to_ihex(&self.segments(), None).expect("a 64 KiB image fits Intel HEX")
    }

    /// The address of a global symbol.
    pub fn symbol(&self, name: &str) -> Option<u32> {
        self.symbols.get(name).copied()
    }
}

/// Where a section is placed.
#[derive(Clone, Copy, Debug)]
enum Mem {
    /// In flash at a byte address.
    Flash(u32),
    /// In SRAM at an address, its initial bytes at a flash address.
    Ram { addr: u32, load: Option<u32> },
}

fn is_flash(obj: &ObjectModule, s: usize) -> bool {
    let sec = &obj.sections()[s];
    sec.kind == SectionKind::Text || sec.name.starts_with(".progmem")
}

/// Build a firmware image for `device`: the startup code calling `entry`,
/// `objects`, and the runtime library.
///
/// # Errors
///
/// See [`link`].
pub fn build(objects: Vec<ObjectModule>, device: &Device, entry: &str) -> Result<Firmware, String> {
    let mut all = vec![super::startup::object(device, entry)];
    all.extend(objects);
    link(all, super::runtime::members(device), device)
}

/// Link `objects` (and whichever `library` members they need) for `device`.
///
/// # Errors
///
/// Undefined or duplicate symbols, a relocation that does not fit (a
/// branch out of range, an odd code address), or an image larger than the
/// device's flash or SRAM.
pub fn link(mut objects: Vec<ObjectModule>, library: Vec<ObjectModule>, device: &Device) -> Result<Firmware, String> {
    // 1. Archive semantics: add members defining still-undefined symbols.
    let mut lib: Vec<Option<ObjectModule>> = library.into_iter().map(Some).collect();
    loop {
        let mut defined: std::collections::HashSet<String> = std::collections::HashSet::new();
        for o in &objects {
            for s in o.symbols() {
                if !s.is_undefined() && s.binding != SymbolBinding::Local {
                    defined.insert(s.name.clone());
                }
            }
        }
        let mut wanted: Vec<String> = Vec::new();
        for o in &objects {
            for r in o.relocations() {
                let s = o.symbol(r.symbol);
                if s.is_undefined() && !defined.contains(&s.name) && !is_linker_symbol(&s.name) {
                    wanted.push(s.name.clone());
                }
            }
        }
        let mut added = false;
        for slot in lib.iter_mut() {
            let hit = slot.as_ref().is_some_and(|m| {
                m.symbols().iter().any(|s| !s.is_undefined() && s.binding != SymbolBinding::Local && wanted.contains(&s.name))
            });
            if hit {
                objects.push(slot.take().expect("checked"));
                added = true;
            }
        }
        if !added {
            break;
        }
    }

    // 2. Place the sections.
    let mut place: DetHashMap<(usize, usize), Mem> = DetHashMap::default();
    let mut flash: Vec<u8> = Vec::new();
    for pass in [true, false] {
        // Code first (pass 1), then flash data (pass 2).
        for (oi, o) in objects.iter().enumerate() {
            for (si, s) in o.sections().iter().enumerate() {
                if is_flash(o, si) && (s.kind == SectionKind::Text) == pass {
                    if flash.len() % 2 == 1 {
                        flash.push(0);
                    }
                    place.insert((oi, si), Mem::Flash(flash.len() as u32));
                    flash.extend_from_slice(&s.bytes);
                }
            }
        }
    }
    if flash.len() % 2 == 1 {
        flash.push(0);
    }
    let load_start = flash.len() as u32;
    let mut ram = u32::from(device.ram_start);
    let data_start = ram;
    for (oi, o) in objects.iter().enumerate() {
        for (si, s) in o.sections().iter().enumerate() {
            if !is_flash(o, si) && !s.is_nobits() && s.kind != SectionKind::Debug {
                place.insert((oi, si), Mem::Ram { addr: ram, load: Some(flash.len() as u32) });
                flash.extend_from_slice(&s.bytes);
                ram += s.bytes.len() as u32;
            }
        }
    }
    let data_end = ram;
    for (oi, o) in objects.iter().enumerate() {
        for (si, s) in o.sections().iter().enumerate() {
            if s.is_nobits() {
                place.insert((oi, si), Mem::Ram { addr: ram, load: None });
                ram += s.size() as u32;
            }
        }
    }
    let bss_end = ram;
    if flash.len() as u32 > device.flash || flash.len() > 0x1_0000 {
        return Err(format!("the image ({} bytes) does not fit the {}'s flash ({} bytes)", flash.len(), device.name, device.flash));
    }
    if bss_end > u32::from(device.ram_end) {
        return Err(format!(
            "the data ({} bytes) does not fit the {}'s SRAM",
            bss_end - u32::from(device.ram_start),
            device.name
        ));
    }

    // 3. Resolve symbols.
    let addr_of = |m: Mem, off: u64| -> u32 {
        match m {
            Mem::Flash(a) | Mem::Ram { addr: a, .. } => a + off as u32,
        }
    };
    let mut globals: DetHashMap<String, (u32, bool)> = DetHashMap::default();
    for (name, v) in [
        ("__data_start", data_start),
        ("__data_end", data_end),
        ("__data_load_start", load_start),
        ("__bss_start", data_end),
        ("__bss_end", bss_end),
        ("__heap_start", bss_end),
        ("__stack", u32::from(device.ram_end)),
    ] {
        globals.insert(name.to_owned(), (v, true));
    }
    for (oi, o) in objects.iter().enumerate() {
        for s in o.symbols() {
            let SymbolValue::Defined { section, offset } = s.value else { continue };
            if s.binding == SymbolBinding::Local {
                continue;
            }
            let Some(&m) = place.get(&(oi, section.index())) else { continue };
            let a = addr_of(m, offset);
            let strong = s.binding == SymbolBinding::Global;
            match globals.get(&s.name) {
                Some(&(_, true)) if strong => return Err(format!("duplicate definition of `{}`", s.name)),
                Some(&(_, true)) => {}
                _ => {
                    globals.insert(s.name.clone(), (a, strong));
                }
            }
        }
    }

    // 4. Apply the relocations.
    for (oi, o) in objects.iter().enumerate() {
        for r in o.relocations() {
            let Some(&m) = place.get(&(oi, r.section.index())) else { continue };
            let sym = o.symbol(r.symbol);
            let s = match sym.value {
                SymbolValue::Defined { section, offset } if sym.binding == SymbolBinding::Local => {
                    let sm = place.get(&(oi, section.index())).ok_or_else(|| format!("`{}` is in an unplaced section", sym.name))?;
                    addr_of(*sm, offset)
                }
                _ => match globals.get(&sym.name) {
                    Some(&(a, _)) => a,
                    None if sym.binding == SymbolBinding::Weak => 0,
                    None => return Err(format!("undefined reference to `{}`", sym.name)),
                },
            };
            let (p, file) = match m {
                Mem::Flash(a) => (a + r.offset as u32, (a + r.offset as u32) as usize),
                Mem::Ram { addr, load: Some(l) } => (addr + r.offset as u32, (l + r.offset as u32) as usize),
                Mem::Ram { load: None, .. } => return Err("a relocation in .bss".to_owned()),
            };
            let v = i64::from(s) + r.addend;
            apply(&mut flash, file, p, r.kind, v).map_err(|e| format!("relocation against `{}` at {p:#x}: {e}", sym.name))?;
        }
    }
    let symbols = globals.into_iter().map(|(k, (a, _))| (k, a)).collect();
    Ok(Firmware { flash, symbols, ram_end: bss_end })
}

/// Whether the linker defines `name`.
fn is_linker_symbol(name: &str) -> bool {
    matches!(name, "__data_start" | "__data_end" | "__data_load_start" | "__bss_start" | "__bss_end" | "__heap_start" | "__stack")
}

fn word(buf: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([buf[at], buf[at + 1]])
}

fn put_word(buf: &mut [u8], at: usize, w: u16) {
    buf[at..at + 2].copy_from_slice(&w.to_le_bytes());
}

/// Patch one relocated field at `buf[at]` (field address `p`) with value `v`.
pub(crate) fn apply(buf: &mut [u8], at: usize, p: u32, kind: RelocKind, v: i64) -> Result<(), String> {
    let even = |v: i64| if v % 2 == 0 { Ok(v / 2) } else { Err(format!("odd code address {v:#x}")) };
    let fits16 = |v: i64| if (-0x8000..=0xffff).contains(&v) { Ok(v as u16) } else { Err(format!("{v:#x} does not fit 16 bits")) };
    let ldi = |buf: &mut [u8], k: u8| {
        let w = word(buf, at) & 0xf0f0;
        put_word(buf, at, w | ((u16::from(k) & 0xf0) << 4) | (u16::from(k) & 0xf));
    };
    match kind {
        RelocKind::AvrCall => {
            let k = even(v)?;
            if !(0..1 << 22).contains(&k) {
                return Err(format!("call target {v:#x} out of range"));
            }
            let [a, b] = super::encode::jmp(k as u32);
            let w0 = (word(buf, at) & !0x01f1) | (a & 0x01f1);
            put_word(buf, at, w0);
            put_word(buf, at + 2, b);
        }
        RelocKind::Avr13Pcrel => {
            let k = even(v - (i64::from(p) + 2))?;
            if !(-2048..=2047).contains(&k) {
                return Err(format!("rjmp/rcall displacement {k} out of range"));
            }
            let w = (word(buf, at) & 0xf000) | (k as u16 & 0x0fff);
            put_word(buf, at, w);
        }
        RelocKind::Abs16 => put_word(buf, at, fits16(v)?),
        RelocKind::Avr16Pm => put_word(buf, at, fits16(even(v)?)?),
        RelocKind::AvrLo8Ldi => ldi(buf, v as u8),
        RelocKind::AvrHi8Ldi => ldi(buf, (v >> 8) as u8),
        RelocKind::AvrLo8LdiPm => ldi(buf, even(v)? as u8),
        RelocKind::AvrHi8LdiPm => ldi(buf, (even(v)? >> 8) as u8),
        RelocKind::Abs32 => buf[at..at + 4].copy_from_slice(&(v as u32).to_le_bytes()),
        other => return Err(format!("{other:?} is not an AVR relocation")),
    }
    Ok(())
}
