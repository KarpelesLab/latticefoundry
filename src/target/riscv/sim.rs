//! An RV64IMAFD(C) instruction-set simulator for the code the backend emits,
//! the test harness's linker, and the C library helpers the code may call,
//! written in Rust.
//!
//! This host cannot execute RISC-V code, so the encoded bytes are validated by
//! running them here: the simulator **decodes the machine code** (it never
//! looks at MIR) with its own decoder, written from the RISC-V ISA manual
//! independently of the encoder, and executes it with the architectural
//! semantics: 64-bit registers, sign-extending `*w` forms and loads, the M
//! extension's division corner cases, NaN-boxed single-precision values in the
//! `f` registers, canonical NaNs from arithmetic, and the saturating
//! float-to-integer conversions. [`link`] lays compiled objects out in a flat
//! address space and applies their relocations (`R_RISCV_CALL_PLT`,
//! `R_RISCV_PCREL_HI20`/`LO12_I`/`LO12_S`, `R_RISCV_GOT_HI20` through a
//! synthesized GOT, `R_RISCV_64`); a call to an undefined symbol lands on a
//! trap address that runs the helper of that name ([`helper`]).
//!
//! An unsupported or reserved encoding, an `ebreak`, a jump outside the image,
//! a misaligned instruction fetch or an exhausted step budget is an error, so
//! a mis-encoded instruction cannot pass silently.

use std::collections::HashMap;

use crate::mc::object::{ObjectModule, RelocKind, SymbolBinding, SymbolValue};

/// The address a top-level call returns to: reaching it ends the run.
pub(super) const EXIT: u64 = 0x0000_0000_00ff_fff0;
/// Where the trap addresses of undefined symbols (runtime helpers) start.
const HELPER_BASE: u64 = 0x0000_0000_00f0_0000;
/// The initial stack pointer.
pub(super) const STACK_TOP: u64 = 0x0000_0040_0000_0000;
/// Where the linked image starts.
const IMAGE_BASE: u64 = 0x0000_0000_0001_0000;
/// Executed-instruction budget per run.
const STEP_BUDGET: u64 = 50_000_000;

// ===========================================================================
// Memory
// ===========================================================================

/// A sparse little-endian byte-addressed memory of 4 KiB pages.
#[derive(Clone, Default)]
pub(super) struct Memory {
    pages: HashMap<u64, Box<[u8; 4096]>>,
}

impl Memory {
    pub(super) fn read8(&self, a: u64) -> u8 {
        self.pages.get(&(a >> 12)).map_or(0, |p| p[(a & 0xfff) as usize])
    }
    pub(super) fn write8(&mut self, a: u64, v: u8) {
        self.pages.entry(a >> 12).or_insert_with(|| Box::new([0; 4096]))[(a & 0xfff) as usize] = v;
    }
    pub(super) fn read(&self, a: u64, size: u64) -> u64 {
        (0..size).fold(0, |acc, k| acc | u64::from(self.read8(a.wrapping_add(k))) << (8 * k))
    }
    pub(super) fn write(&mut self, a: u64, size: u64, v: u64) {
        for k in 0..size {
            self.write8(a.wrapping_add(k), (v >> (8 * k)) as u8);
        }
    }
    pub(super) fn write_bytes(&mut self, a: u64, bytes: &[u8]) {
        for (k, &b) in bytes.iter().enumerate() {
            self.write8(a.wrapping_add(k as u64), b);
        }
    }
}

// ===========================================================================
// The test harness's linker
// ===========================================================================

/// A linked image: memory with every section placed and relocated, the
/// global symbol addresses, the helper trap addresses by name, and where the
/// code lies (an instruction fetch outside it is an error).
#[derive(Clone)]
pub(super) struct Image {
    pub(super) mem: Memory,
    pub(super) symbols: HashMap<String, u64>,
    pub(super) helpers: HashMap<u64, String>,
    pub(super) text: Vec<(u64, u64)>,
}

fn sext(v: u64, bits: u32) -> i64 {
    ((v << (64 - bits)) as i64) >> (64 - bits)
}

/// The `hi20`/`lo12` split of a PC-relative displacement: `hi << 12` plus the
/// sign-extended `lo` equals `disp`.
fn hi_lo(disp: i64) -> (u32, i32) {
    let hi = ((disp + 0x800) >> 12) as u32 & 0xFFFFF;
    let lo = (disp - (sext(u64::from(hi), 20) << 12)) as i32;
    (hi, lo)
}

/// Lay `objects` out from [`IMAGE_BASE`] (sections in object order, each at
/// its alignment), resolve symbols (an object's local symbols first, then the
/// global ones; anything undefined becomes a helper trap address) and apply
/// every relocation.
pub(super) fn link(objects: &[&ObjectModule]) -> Result<Image, String> {
    let mut mem = Memory::default();
    let mut at = IMAGE_BASE;
    let mut bases: Vec<Vec<u64>> = Vec::new();
    let mut text = Vec::new();
    for obj in objects {
        let mut b = Vec::new();
        for s in obj.sections() {
            at = at.div_ceil(s.align.max(16)) * s.align.max(16);
            b.push(at);
            mem.write_bytes(at, &s.bytes);
            if s.kind == crate::mc::object::SectionKind::Text {
                text.push((at, at + s.bytes.len() as u64));
            }
            at += (s.bytes.len() as u64).max(s.size()).max(1);
        }
        bases.push(b);
    }
    // Global symbols.
    let mut symbols: HashMap<String, u64> = HashMap::new();
    for (oi, obj) in objects.iter().enumerate() {
        for sym in obj.symbols() {
            if let SymbolValue::Defined { section, offset } = sym.value
                && sym.binding != SymbolBinding::Local
            {
                symbols.insert(sym.name.clone(), bases[oi][section.index()] + offset);
            }
        }
    }
    let mut helpers: HashMap<u64, String> = HashMap::new();
    let mut helper_at: HashMap<String, u64> = HashMap::new();
    let mut got: HashMap<String, u64> = HashMap::new();
    let mut got_next = at.div_ceil(4096) * 4096 + 4096;
    for (oi, obj) in objects.iter().enumerate() {
        let local: HashMap<&str, u64> = obj
            .symbols()
            .iter()
            .filter_map(|s| match s.value {
                SymbolValue::Defined { section, offset } if s.binding == SymbolBinding::Local => {
                    Some((s.name.as_str(), bases[oi][section.index()] + offset))
                }
                _ => None,
            })
            .collect();
        let mut resolve = |name: &str| -> u64 {
            if let Some(&a) = local.get(name).or_else(|| symbols.get(name)) {
                return a;
            }
            *helper_at.entry(name.to_owned()).or_insert_with(|| {
                let a = HELPER_BASE + 16 * helpers.len() as u64;
                helpers.insert(a, name.to_owned());
                a
            })
        };
        // First the high parts (a low part finds its `auipc` by address).
        let mut hi_disp: HashMap<u64, i64> = HashMap::new();
        let mut patches: Vec<(u64, RelocKind, i64, u64)> = Vec::new();
        for r in obj.relocations() {
            let p = bases[oi][r.section.index()] + r.offset;
            let name = obj.symbol(r.symbol).name.clone();
            let s = resolve(&name);
            match r.kind {
                RelocKind::RiscvPcrelHi20 => {
                    hi_disp.insert(p, (s as i64).wrapping_add(r.addend).wrapping_sub(p as i64));
                }
                RelocKind::RiscvGotHi20 => {
                    let slot = *got.entry(name.clone()).or_insert_with(|| {
                        let g = got_next;
                        got_next += 8;
                        g
                    });
                    mem.write(slot, 8, s);
                    hi_disp.insert(p, (slot as i64).wrapping_sub(p as i64));
                }
                _ => {}
            }
            patches.push((p, r.kind, r.addend, s));
        }
        for (p, kind, addend, s) in patches {
            let word = |mem: &Memory, a: u64| mem.read(a, 4) as u32;
            match kind {
                RelocKind::Abs64 => mem.write(p, 8, s.wrapping_add(addend as u64)),
                RelocKind::Abs32 => mem.write(p, 4, s.wrapping_add(addend as u64)),
                RelocKind::RiscvCallPlt => {
                    let disp = (s as i64).wrapping_add(addend).wrapping_sub(p as i64);
                    let (hi, lo) = hi_lo(disp);
                    let w = word(&mem, p);
                    mem.write(p, 4, u64::from((w & 0xFFF) | (hi << 12)));
                    let w = word(&mem, p + 4);
                    mem.write(p + 4, 4, u64::from((w & 0x000F_FFFF) | ((lo as u32) << 20)));
                }
                RelocKind::RiscvPcrelHi20 | RelocKind::RiscvGotHi20 => {
                    let (hi, _) = hi_lo(hi_disp[&p]);
                    let w = word(&mem, p);
                    mem.write(p, 4, u64::from((w & 0xFFF) | (hi << 12)));
                }
                RelocKind::RiscvPcrelLo12I | RelocKind::RiscvPcrelLo12S => {
                    let disp = *hi_disp
                        .get(&s)
                        .ok_or_else(|| format!("PCREL_LO12 at {p:#x}: no HI20 at {s:#x}"))?;
                    let (_, lo) = hi_lo(disp);
                    let w = word(&mem, p);
                    let lo = lo as u32;
                    let w = if kind == RelocKind::RiscvPcrelLo12I {
                        (w & 0x000F_FFFF) | (lo << 20)
                    } else {
                        (w & 0x01FF_F07F) | ((lo >> 5) << 25) | ((lo & 31) << 7)
                    };
                    mem.write(p, 4, u64::from(w));
                }
                other => return Err(format!("unsupported relocation {other:?}")),
            }
        }
    }
    Ok(Image { mem, symbols, helpers, text })
}

/// Load a linked ELF64 RISC-V executable (from qld): every `PT_LOAD` segment
/// at its address (zero-filled past its file bytes), the executable ones as
/// code, and the symbol table's defined symbols by name.
pub(super) fn load_elf(bytes: &[u8]) -> Result<Image, String> {
    let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    if bytes.len() < 64 || &bytes[..4] != b"\x7fELF" || bytes[4] != 2 || u16_at(18) != 243 {
        return Err("not an ELF64 RISC-V file".into());
    }
    let mut mem = Memory::default();
    let mut text = Vec::new();
    let (phoff, phnum, phsz) = (u64_at(32) as usize, u16_at(56) as usize, u16_at(54) as usize);
    for k in 0..phnum {
        let p = phoff + k * phsz;
        if u32_at(p) != 1 {
            continue; // PT_LOAD only
        }
        let (flags, off, vaddr, filesz, memsz) =
            (u32_at(p + 4), u64_at(p + 8) as usize, u64_at(p + 16), u64_at(p + 32) as usize, u64_at(p + 40));
        mem.write_bytes(vaddr, &bytes[off..off + filesz]);
        for z in filesz as u64..memsz {
            mem.write8(vaddr + z, 0);
        }
        if flags & 1 != 0 {
            text.push((vaddr, vaddr + memsz));
        }
    }
    let mut symbols = HashMap::new();
    let (shoff, shnum, shsz) = (u64_at(40) as usize, u16_at(60) as usize, u16_at(58) as usize);
    for k in 0..shnum {
        let s = shoff + k * shsz;
        if u32_at(s + 4) != 2 {
            continue; // SHT_SYMTAB only
        }
        let (off, size, link) = (u64_at(s + 24) as usize, u64_at(s + 32) as usize, u32_at(s + 40) as usize);
        let str_off = u64_at(shoff + link * shsz + 24) as usize;
        for e in (off..off + size).step_by(24) {
            let (name, shndx, value) = (u32_at(e) as usize, u16_at(e + 6), u64_at(e + 8));
            if shndx == 0 || name == 0 {
                continue;
            }
            let end = bytes[str_off + name..].iter().position(|&b| b == 0).unwrap_or(0);
            let n = String::from_utf8_lossy(&bytes[str_off + name..str_off + name + end]).into_owned();
            symbols.entry(n).or_insert(value);
        }
    }
    Ok(Image { mem, symbols, helpers: HashMap::new(), text })
}

// ===========================================================================
// The C library helpers
// ===========================================================================

/// Run the helper `name` on the CPU state (arguments in `a0..`/`fa0..`,
/// results written back), as the C library would.
fn helper(name: &str, cpu: &mut Cpu) -> Result<(), String> {
    match name {
        "fmod" => {
            let (x, y) = (f64::from_bits(cpu.f[10]), f64::from_bits(cpu.f[11]));
            cpu.f[10] = (x % y).to_bits();
        }
        "fmodf" => {
            let (x, y) = (cpu.f32(10), cpu.f32(11));
            cpu.set_f32(10, x % y);
        }
        other => return Err(format!("call to an undefined function {other}")),
    }
    Ok(())
}

// ===========================================================================
// The CPU
// ===========================================================================

/// Bounds on the stack for probe tests: `[guard_lo, guard_lo + 4096)` is a
/// guard page (an access there is the defined `SIGSEGV`) and nothing is
/// mapped below it.
#[derive(Clone, Copy, Debug)]
pub(super) struct StackGuard {
    pub(super) guard_lo: u64,
}

/// Why a run stopped short of returning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Fault {
    /// An access inside the guard page.
    Guard,
    /// An access below the guard page: something jumped over it.
    Skipped,
    /// Anything else (a decode error, a bad jump, ...).
    Other(String),
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fault::Guard => f.write_str("guard page hit"),
            Fault::Skipped => f.write_str("access below the guard page"),
            Fault::Other(s) => f.write_str(s),
        }
    }
}

impl From<String> for Fault {
    fn from(s: String) -> Fault {
        Fault::Other(s)
    }
}

/// A Linux environment-call handler: number (`a7`) and `a0..a5`, returning
/// `a0`.
pub(super) type Syscall<'h> = Box<dyn FnMut(u64, [u64; 6]) -> Result<u64, String> + 'h>;

/// The architectural state.
pub(super) struct Cpu<'h> {
    pub(super) x: [u64; 32],
    pub(super) f: [u64; 32],
    pub(super) pc: u64,
    pub(super) mem: Memory,
    helpers: HashMap<u64, String>,
    text: Vec<(u64, u64)>,
    pub(super) syscall: Option<Syscall<'h>>,
    pub(super) guard: Option<StackGuard>,
    /// The lowest `sp` seen.
    pub(super) min_sp: u64,
    /// Every load's and store's address, when recording (for the probe
    /// invariant: a load touches the stack as a store does).
    pub(super) touches: Option<Vec<u64>>,
    /// Instructions executed.
    pub(super) steps: u64,
    /// Compressed (16-bit) instructions executed.
    pub(super) compressed: u64,
}

const CANON_F32: u32 = 0x7fc0_0000;
const CANON_F64: u64 = 0x7ff8_0000_0000_0000;

/// A 32-bit pattern NaN-boxed in a 64-bit `f` register.
fn box32(b: u32) -> u64 {
    0xffff_ffff_0000_0000 | u64::from(b)
}

impl<'h> Cpu<'h> {
    /// A CPU over the linked `image`.
    pub(super) fn new(image: &Image) -> Cpu<'h> {
        Cpu {
            x: [0; 32],
            f: [0; 32],
            pc: 0,
            mem: image.mem.clone(),
            helpers: image.helpers.clone(),
            text: image.text.clone(),
            syscall: None,
            guard: None,
            min_sp: STACK_TOP,
            touches: None,
            steps: 0,
            compressed: 0,
        }
    }

    /// The single-precision value in `f[r]` (an improperly NaN-boxed value
    /// reads as the canonical NaN).
    pub(super) fn f32(&self, r: usize) -> f32 {
        f32::from_bits(self.f32_bits(r))
    }
    fn f32_bits(&self, r: usize) -> u32 {
        let v = self.f[r];
        if v >> 32 == 0xffff_ffff { v as u32 } else { CANON_F32 }
    }
    pub(super) fn set_f32(&mut self, r: usize, v: f32) {
        self.f[r] = box32(v.to_bits());
    }

    /// Call `entry` with `a0..` = `xs` and `fa0..` = `fs` (raw register
    /// images) and `sp` = `sp`, running until it returns to [`EXIT`].
    pub(super) fn call(&mut self, entry: u64, xs: &[u64], fs: &[u64], sp: u64) -> Result<(), Fault> {
        for (i, &v) in xs.iter().enumerate() {
            self.x[10 + i] = v;
        }
        for (i, &v) in fs.iter().enumerate() {
            self.f[10 + i] = v;
        }
        self.x[1] = EXIT;
        self.x[2] = sp;
        self.min_sp = sp;
        self.pc = entry;
        self.run()
    }

    fn check_access(&self, a: u64) -> Result<(), Fault> {
        if let Some(g) = self.guard
            && (STACK_TOP - (1 << 32)..STACK_TOP).contains(&a)
        {
            if a < g.guard_lo {
                return Err(Fault::Skipped);
            }
            if a < g.guard_lo + 4096 {
                return Err(Fault::Guard);
            }
        }
        Ok(())
    }

    fn load(&mut self, a: u64, size: u64) -> Result<u64, Fault> {
        self.check_access(a)?;
        if let Some(t) = &mut self.touches {
            t.push(a);
        }
        Ok(self.mem.read(a, size))
    }

    fn store(&mut self, a: u64, size: u64, v: u64) -> Result<(), Fault> {
        self.check_access(a)?;
        if let Some(t) = &mut self.touches {
            t.push(a);
        }
        self.mem.write(a, size, v);
        Ok(())
    }

    fn set(&mut self, r: usize, v: u64) {
        if r != 0 {
            self.x[r] = v;
        }
    }

    fn run(&mut self) -> Result<(), Fault> {
        loop {
            if self.pc == EXIT {
                return Ok(());
            }
            self.steps += 1;
            if self.steps > STEP_BUDGET {
                return Err(Fault::Other("step budget exhausted".into()));
            }
            if let Some(name) = self.helpers.get(&self.pc).cloned() {
                helper(&name, self)?;
                self.pc = self.x[1];
                continue;
            }
            if self.pc & 1 != 0 || !self.text.iter().any(|&(lo, hi)| (lo..hi).contains(&self.pc)) {
                return Err(Fault::Other(format!("instruction fetch outside the code at {:#x}", self.pc)));
            }
            let lo16 = self.mem.read(self.pc, 2) as u32;
            if lo16 & 3 != 3 {
                self.compressed += 1;
                let w = expand_compressed(lo16 as u16)
                    .ok_or_else(|| Fault::Other(format!("illegal compressed instruction {lo16:#06x} at {:#x}", self.pc)))?;
                self.exec(w, 2)?;
            } else {
                let w = self.mem.read(self.pc, 4) as u32;
                self.exec(w, 4)?;
            }
            self.min_sp = self.min_sp.min(self.x[2]);
        }
    }

    /// Execute one (32-bit, or expanded compressed) instruction of `len`
    /// bytes at `pc`.
    fn exec(&mut self, w: u32, len: u64) -> Result<(), Fault> {
        let pc = self.pc;
        let mut next = pc + len;
        let op = w & 0x7F;
        let rd = ((w >> 7) & 31) as usize;
        let f3 = (w >> 12) & 7;
        let rs1 = ((w >> 15) & 31) as usize;
        let rs2 = ((w >> 20) & 31) as usize;
        let f7 = w >> 25;
        let imm_i = sext(u64::from(w >> 20), 12);
        let imm_s = sext(u64::from(((w >> 25) << 5) | ((w >> 7) & 31)), 12);
        let (a, b) = (self.x[rs1], self.x[rs2]);
        let bad = || Fault::Other(format!("unsupported instruction {w:#010x} at {pc:#x}"));
        match op {
            0x37 => self.set(rd, sext(u64::from(w & 0xFFFF_F000), 32) as u64), // lui
            0x17 => self.set(rd, pc.wrapping_add(sext(u64::from(w & 0xFFFF_F000), 32) as u64)), // auipc
            0x6F => {
                let imm = ((w >> 31) << 20) | (((w >> 12) & 0xFF) << 12) | (((w >> 20) & 1) << 11) | (((w >> 21) & 0x3FF) << 1);
                self.set(rd, next);
                next = pc.wrapping_add(sext(u64::from(imm), 21) as u64);
            }
            0x67 if f3 == 0 => {
                let t = a.wrapping_add(imm_i as u64) & !1;
                self.set(rd, next);
                next = t;
            }
            0x63 => {
                let imm = ((w >> 31) << 12) | (((w >> 7) & 1) << 11) | (((w >> 25) & 0x3F) << 5) | (((w >> 8) & 0xF) << 1);
                let take = match f3 {
                    0 => a == b,
                    1 => a != b,
                    4 => (a as i64) < (b as i64),
                    5 => (a as i64) >= (b as i64),
                    6 => a < b,
                    7 => a >= b,
                    _ => return Err(bad()),
                };
                if take {
                    next = pc.wrapping_add(sext(u64::from(imm), 13) as u64);
                }
            }
            0x03 => {
                let addr = a.wrapping_add(imm_i as u64);
                let v = match f3 {
                    0 => sext(self.load(addr, 1)?, 8) as u64,
                    1 => sext(self.load(addr, 2)?, 16) as u64,
                    2 => sext(self.load(addr, 4)?, 32) as u64,
                    3 => self.load(addr, 8)?,
                    4 => self.load(addr, 1)?,
                    5 => self.load(addr, 2)?,
                    6 => self.load(addr, 4)?,
                    _ => return Err(bad()),
                };
                self.set(rd, v);
            }
            0x23 => {
                let addr = a.wrapping_add(imm_s as u64);
                if f3 > 3 {
                    return Err(bad());
                }
                self.store(addr, 1 << f3, b)?;
            }
            0x07 => {
                let addr = a.wrapping_add(imm_i as u64);
                match f3 {
                    2 => self.f[rd] = box32(self.load(addr, 4)? as u32),
                    3 => self.f[rd] = self.load(addr, 8)?,
                    _ => return Err(bad()),
                }
            }
            0x27 => {
                let addr = a.wrapping_add(imm_s as u64);
                match f3 {
                    2 => self.store(addr, 4, self.f[rs2] & 0xffff_ffff)?,
                    3 => self.store(addr, 8, self.f[rs2])?,
                    _ => return Err(bad()),
                }
            }
            0x13 => {
                let sh = (w >> 20) & 0x3F;
                let v = match f3 {
                    0 => a.wrapping_add(imm_i as u64),
                    1 if w >> 26 == 0 => a << sh,
                    2 => u64::from((a as i64) < imm_i),
                    3 => u64::from(a < imm_i as u64),
                    4 => a ^ imm_i as u64,
                    5 if w >> 26 == 0x10 => ((a as i64) >> sh) as u64,
                    5 if w >> 26 == 0 => a >> sh,
                    6 => a | imm_i as u64,
                    7 => a & imm_i as u64,
                    _ => return Err(bad()),
                };
                self.set(rd, v);
            }
            0x1B => {
                let sh = (w >> 20) & 0x1F;
                let v = match (f3, f7) {
                    (0, _) => a.wrapping_add(imm_i as u64) as u32,
                    (1, 0) => (a as u32) << sh,
                    (5, 0) => (a as u32) >> sh,
                    (5, 0x20) => ((a as i32) >> sh) as u32,
                    _ => return Err(bad()),
                };
                self.set(rd, sext(u64::from(v), 32) as u64);
            }
            0x33 => {
                let v = match (f7, f3) {
                    (0, 0) => a.wrapping_add(b),
                    (0x20, 0) => a.wrapping_sub(b),
                    (0, 1) => a << (b & 63),
                    (0, 2) => u64::from((a as i64) < (b as i64)),
                    (0, 3) => u64::from(a < b),
                    (0, 4) => a ^ b,
                    (0, 5) => a >> (b & 63),
                    (0x20, 5) => ((a as i64) >> (b & 63)) as u64,
                    (0, 6) => a | b,
                    (0, 7) => a & b,
                    (1, 0) => a.wrapping_mul(b),
                    (1, 1) => ((i128::from(a as i64) * i128::from(b as i64)) >> 64) as u64,
                    (1, 2) => ((i128::from(a as i64) * (b as i128)) >> 64) as u64,
                    (1, 3) => ((u128::from(a) * u128::from(b)) >> 64) as u64,
                    (1, 4) => {
                        let (x, y) = (a as i64, b as i64);
                        if y == 0 { u64::MAX } else { x.wrapping_div(y) as u64 }
                    }
                    (1, 5) => a.checked_div(b).unwrap_or(u64::MAX),
                    (1, 6) => {
                        let (x, y) = (a as i64, b as i64);
                        if y == 0 { a } else { x.wrapping_rem(y) as u64 }
                    }
                    (1, 7) => if b == 0 { a } else { a % b },
                    _ => return Err(bad()),
                };
                self.set(rd, v);
            }
            0x3B => {
                let (x, y) = (a as u32, b as u32);
                let v: u32 = match (f7, f3) {
                    (0, 0) => x.wrapping_add(y),
                    (0x20, 0) => x.wrapping_sub(y),
                    (0, 1) => x << (y & 31),
                    (0, 5) => x >> (y & 31),
                    (0x20, 5) => ((x as i32) >> (y & 31)) as u32,
                    (1, 0) => x.wrapping_mul(y),
                    (1, 4) => if y == 0 { u32::MAX } else { (x as i32).wrapping_div(y as i32) as u32 },
                    (1, 5) => x.checked_div(y).unwrap_or(u32::MAX),
                    (1, 6) => if y == 0 { x } else { (x as i32).wrapping_rem(y as i32) as u32 },
                    (1, 7) => if y == 0 { x } else { x % y },
                    _ => return Err(bad()),
                };
                self.set(rd, sext(u64::from(v), 32) as u64);
            }
            0x0F => {} // fence / fence.tso: a single hart is always ordered
            0x73 => match w {
                0x0000_0073 => {
                    let hook = self.syscall.as_mut().ok_or_else(|| Fault::Other("ecall without a handler".into()))?;
                    let args = [self.x[10], self.x[11], self.x[12], self.x[13], self.x[14], self.x[15]];
                    let r = hook(self.x[17], args)?;
                    self.x[10] = r;
                }
                0x0010_0073 => return Err(Fault::Other(format!("ebreak at {pc:#x}"))),
                _ => return Err(bad()),
            },
            0x2F => self.amo(w, rd, rs1, rs2, f3)?,
            0x43 | 0x47 | 0x4B | 0x4F => self.fma(w, op, rd, rs1, rs2)?,
            0x53 => self.op_fp(w, rd, rs1, rs2, f3, f7)?,
            _ => return Err(bad()),
        }
        self.pc = next;
        Ok(())
    }

    /// The A extension (a single hart: every reservation succeeds).
    fn amo(&mut self, w: u32, rd: usize, rs1: usize, rs2: usize, f3: u32) -> Result<(), Fault> {
        let size = match f3 {
            2 => 4,
            3 => 8,
            _ => return Err(Fault::Other(format!("bad AMO width {w:#010x}"))),
        };
        let addr = self.x[rs1];
        let ext = |v: u64| if size == 4 { sext(v, 32) as u64 } else { v };
        let old = ext(self.load(addr, size)?);
        let src = self.x[rs2];
        let f5 = w >> 27;
        let (s32, o32) = (src as u32 as i32, old as u32 as i32);
        let new = match f5 {
            0b00010 => {
                // lr
                self.set(rd, old);
                return Ok(());
            }
            0b00011 => {
                // sc: always succeeds
                self.store(addr, size, src)?;
                self.set(rd, 0);
                return Ok(());
            }
            0b00001 => src,
            0b00000 => old.wrapping_add(src),
            0b00100 => old ^ src,
            0b01100 => old & src,
            0b01000 => old | src,
            0b10000 if size == 4 => o32.min(s32) as u64,
            0b10100 if size == 4 => o32.max(s32) as u64,
            0b11000 if size == 4 => u64::from((old as u32).min(src as u32)),
            0b11100 if size == 4 => u64::from((old as u32).max(src as u32)),
            0b10000 => (old as i64).min(src as i64) as u64,
            0b10100 => (old as i64).max(src as i64) as u64,
            0b11000 => old.min(src),
            0b11100 => old.max(src),
            _ => return Err(Fault::Other(format!("bad AMO {w:#010x}"))),
        };
        self.store(addr, size, new)?;
        self.set(rd, old);
        Ok(())
    }

    /// The fused multiply-adds (`fmadd`/`fmsub`/`fnmsub`/`fnmadd`).
    fn fma(&mut self, w: u32, op: u32, rd: usize, rs1: usize, rs2: usize) -> Result<(), Fault> {
        let rs3 = (w >> 27) as usize;
        let fmt = (w >> 25) & 3;
        check_rm((w >> 12) & 7, w)?;
        let (neg_prod, neg_add) = match op {
            0x43 => (false, false),
            0x47 => (false, true),
            0x4B => (true, false),
            _ => (true, true),
        };
        match fmt {
            0 => {
                let (x, y, z) = (self.f32(rs1), self.f32(rs2), self.f32(rs3));
                let x = if neg_prod { -x } else { x };
                let z = if neg_add { -z } else { z };
                self.f[rd] = box32(canon32(x.mul_add(y, z)));
            }
            1 => {
                let (x, y, z) = (f64::from_bits(self.f[rs1]), f64::from_bits(self.f[rs2]), f64::from_bits(self.f[rs3]));
                let x = if neg_prod { -x } else { x };
                let z = if neg_add { -z } else { z };
                self.f[rd] = canon64(x.mul_add(y, z));
            }
            _ => return Err(Fault::Other(format!("bad fma format {w:#010x}"))),
        }
        Ok(())
    }

    /// The OP-FP major opcode.
    fn op_fp(&mut self, w: u32, rd: usize, rs1: usize, rs2: usize, f3: u32, f7: u32) -> Result<(), Fault> {
        let bad = || Fault::Other(format!("unsupported FP instruction {w:#010x} at {:#x}", self.pc));
        let dbl = f7 & 1 == 1;
        if f7 & 2 != 0 {
            return Err(bad()); // fmt H/Q
        }
        let (x32, y32) = (self.f32(rs1), self.f32(rs2));
        let (x64, y64) = (f64::from_bits(self.f[rs1]), f64::from_bits(self.f[rs2]));
        match f7 >> 2 {
            0x00..=0x03 => {
                check_rm(f3, w)?;
                if dbl {
                    let r = match f7 >> 2 {
                        0 => x64 + y64,
                        1 => x64 - y64,
                        2 => x64 * y64,
                        _ => x64 / y64,
                    };
                    self.f[rd] = canon64(r);
                } else {
                    let r = match f7 >> 2 {
                        0 => x32 + y32,
                        1 => x32 - y32,
                        2 => x32 * y32,
                        _ => x32 / y32,
                    };
                    self.f[rd] = box32(canon32(r));
                }
            }
            0x0B if rs2 == 0 => {
                check_rm(f3, w)?;
                self.f[rd] = if dbl { canon64(x64.sqrt()) } else { box32(canon32(x32.sqrt())) };
            }
            0x04 => {
                // fsgnj / fsgnjn / fsgnjx
                let (sign, mag) = if dbl {
                    (1u64 << 63, self.f[rs1])
                } else {
                    (1u64 << 31, u64::from(self.f32_bits(rs1)))
                };
                let other = if dbl { self.f[rs2] } else { u64::from(self.f32_bits(rs2)) };
                let s = match f3 {
                    0 => other & sign,
                    1 => !other & sign,
                    2 => (mag ^ other) & sign,
                    _ => return Err(bad()),
                };
                let r = (mag & !sign) | s;
                self.f[rd] = if dbl { r } else { box32(r as u32) };
            }
            0x05 => {
                // fmin / fmax (IEEE 754-2019 minimumNumber / maximumNumber).
                let max = match f3 {
                    0 => false,
                    1 => true,
                    _ => return Err(bad()),
                };
                if dbl {
                    self.f[rd] = minmax64(x64, y64, max);
                } else {
                    self.f[rd] = box32(minmax32(x32, y32, max));
                }
            }
            0x08 => {
                // fcvt.s.d (rs2 = 1) / fcvt.d.s (rs2 = 0)
                check_rm(f3, w)?;
                match (dbl, rs2) {
                    (false, 1) => self.f[rd] = box32(canon32(x64 as f32)),
                    (true, 0) => self.f[rd] = canon64(f64::from(x32)),
                    _ => return Err(bad()),
                }
            }
            0x14 => {
                let (x, y) = if dbl { (x64, y64) } else { (f64::from(x32), f64::from(y32)) };
                let r = match f3 {
                    2 => x == y,
                    1 => x < y,
                    0 => x <= y,
                    _ => return Err(bad()),
                };
                self.set(rd, u64::from(r));
            }
            0x18 => {
                // fcvt.{w,wu,l,lu}.{s,d}
                let x = if dbl { x64 } else { f64::from(x32) };
                let rm = if f3 == 7 { 0 } else { f3 };
                let v = round(x, rm).ok_or_else(bad)?;
                let r = match rs2 {
                    0 => sext(sat(x, v, -(1i128 << 31), (1i128 << 31) - 1, (1i128 << 31) - 1) as u64, 32) as u64,
                    1 => sext(sat(x, v, 0, (1i128 << 32) - 1, (1i128 << 32) - 1) as u64, 32) as u64,
                    2 => sat(x, v, -(1i128 << 63), (1i128 << 63) - 1, (1i128 << 63) - 1) as u64,
                    3 => sat(x, v, 0, (1i128 << 64) - 1, (1i128 << 64) - 1) as u64,
                    _ => return Err(bad()),
                };
                self.set(rd, r);
            }
            0x1A => {
                // fcvt.{s,d}.{w,wu,l,lu}: exact or round-to-nearest-even
                check_rm(f3, w)?;
                let a = self.x[rs1];
                let n: i128 = match rs2 {
                    0 => i128::from(a as i32),
                    1 => i128::from(a as u32),
                    2 => i128::from(a as i64),
                    3 => i128::from(a),
                    _ => return Err(bad()),
                };
                // `as` rounds to nearest, ties to even.
                self.f[rd] = if dbl { (n as f64).to_bits() } else { box32((n as f32).to_bits()) };
            }
            0x1C if rs2 == 0 && f3 == 0 => {
                // fmv.x.w (sign-extended) / fmv.x.d
                let v = if dbl { self.f[rs1] } else { sext(self.f[rs1] & 0xffff_ffff, 32) as u64 };
                self.set(rd, v);
            }
            0x1C if rs2 == 0 && f3 == 1 => {
                let c = if dbl { classify(x64) } else { classify(f64::from(x32)) };
                self.set(rd, c);
            }
            0x1E if rs2 == 0 && f3 == 0 => {
                // fmv.w.x / fmv.d.x
                let a = self.x[rs1];
                self.f[rd] = if dbl { a } else { box32(a as u32) };
            }
            _ => return Err(bad()),
        }
        Ok(())
    }
}

/// Only round-to-nearest-even (static `rne`, or `dyn` with `frm` = 0) is
/// modeled for arithmetic.
fn check_rm(rm: u32, w: u32) -> Result<(), Fault> {
    if rm == 0 || rm == 7 { Ok(()) } else { Err(Fault::Other(format!("rounding mode {rm} in {w:#010x}"))) }
}

fn canon32(v: f32) -> u32 {
    if v.is_nan() { CANON_F32 } else { v.to_bits() }
}
fn canon64(v: f64) -> u64 {
    if v.is_nan() { CANON_F64 } else { v.to_bits() }
}

fn minmax64(x: f64, y: f64, max: bool) -> u64 {
    match (x.is_nan(), y.is_nan()) {
        (true, true) => CANON_F64,
        (true, false) => y.to_bits(),
        (false, true) => x.to_bits(),
        _ if x == y => {
            // -0 < +0
            let (a, b) = (x.to_bits(), y.to_bits());
            if max { a & b } else { a | b }
        }
        _ => (if (x > y) == max { x } else { y }).to_bits(),
    }
}
fn minmax32(x: f32, y: f32, max: bool) -> u32 {
    match (x.is_nan(), y.is_nan()) {
        (true, true) => CANON_F32,
        (true, false) => y.to_bits(),
        (false, true) => x.to_bits(),
        _ if x == y => {
            let (a, b) = (x.to_bits(), y.to_bits());
            if max { a & b } else { a | b }
        }
        _ => (if (x > y) == max { x } else { y }).to_bits(),
    }
}

/// Round `x` to an integer-valued float under a rounding mode (`None` for a
/// reserved mode).
fn round(x: f64, rm: u32) -> Option<f64> {
    Some(match rm {
        0 => x.round_ties_even(),
        1 => x.trunc(),
        2 => x.floor(),
        3 => x.ceil(),
        4 => x.round(),
        _ => return None,
    })
}

/// The saturating conversion of the rounded value `v` of `x` into
/// `[lo, hi]`, with `nan` for a NaN input.
fn sat(x: f64, v: f64, lo: i128, hi: i128, nan: i128) -> i128 {
    if x.is_nan() {
        return nan;
    }
    if v <= lo as f64 {
        return lo;
    }
    if v >= hi as f64 {
        return hi;
    }
    v as i128
}

/// `fclass`: the one-hot class of a value.
fn classify(x: f64) -> u64 {
    let neg = x.is_sign_negative();
    let bit = if x.is_nan() {
        // the harness only feeds quiet NaNs
        9
    } else if x.is_infinite() {
        if neg { 0 } else { 7 }
    } else if x == 0.0 {
        if neg { 3 } else { 4 }
    } else if x.is_subnormal() {
        if neg { 2 } else { 5 }
    } else if neg {
        1
    } else {
        6
    };
    1 << bit
}

// ===========================================================================
// The C extension: each compressed instruction as its 32-bit expansion
// ===========================================================================

/// Expand an RV64C instruction to the 32-bit instruction it stands for
/// (RISC-V ISA manual, "C" Standard Extension), or `None` for a reserved or
/// unsupported encoding.
pub(super) fn expand_compressed(h: u16) -> Option<u32> {
    let h = u32::from(h);
    let op = h & 3;
    let f3 = h >> 13;
    let rdp = 8 + ((h >> 2) & 7); // rd'/rs2'
    let rs1p = 8 + ((h >> 7) & 7); // rs1'/rd'
    let rd = (h >> 7) & 31;
    let rs2 = (h >> 2) & 31;
    let r = |f7: u32, rs2: u32, rs1: u32, f3: u32, rd: u32, opc: u32| (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opc;
    let i = |imm: i32, rs1: u32, f3: u32, rd: u32, opc: u32| (((imm as u32) & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opc;
    let s = |imm: i32, rs2: u32, rs1: u32, f3: u32, opc: u32| {
        let u = imm as u32;
        (((u >> 5) & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((u & 31) << 7) | opc
    };
    let sx = |v: u32, bits: u32| sext(u64::from(v), bits) as i32;
    match (op, f3) {
        (0, 0) => {
            // c.addi4spn
            let imm = ((h >> 7) & 0x30) | ((h >> 1) & 0x3C0) | ((h >> 4) & 4) | ((h >> 2) & 8);
            if imm == 0 {
                return None;
            }
            Some(i(imm as i32, 2, 0, rdp, 0x13))
        }
        (0, 1) | (0, 3) | (0, 5) | (0, 7) => {
            // c.fld / c.ld / c.fsd / c.sd (8-byte scaled offset)
            let imm = ((h >> 7) & 0x38) | ((h << 1) & 0xC0);
            Some(match f3 {
                1 => i(imm as i32, rs1p, 3, rdp, 0x07),
                3 => i(imm as i32, rs1p, 3, rdp, 0x03),
                5 => s(imm as i32, rdp, rs1p, 3, 0x27),
                _ => s(imm as i32, rdp, rs1p, 3, 0x23),
            })
        }
        (0, 2) | (0, 6) => {
            // c.lw / c.sw
            let imm = ((h >> 7) & 0x38) | ((h >> 4) & 4) | ((h << 1) & 0x40);
            Some(if f3 == 2 { i(imm as i32, rs1p, 2, rdp, 0x03) } else { s(imm as i32, rdp, rs1p, 2, 0x23) })
        }
        (1, 0) => {
            // c.addi (c.nop)
            let imm = sx(((h >> 7) & 0x20) | ((h >> 2) & 31), 6);
            Some(i(imm, rd, 0, rd, 0x13))
        }
        (1, 1) => {
            // c.addiw
            if rd == 0 {
                return None;
            }
            let imm = sx(((h >> 7) & 0x20) | ((h >> 2) & 31), 6);
            Some(i(imm, rd, 0, rd, 0x1B))
        }
        (1, 2) => {
            // c.li
            let imm = sx(((h >> 7) & 0x20) | ((h >> 2) & 31), 6);
            Some(i(imm, 0, 0, rd, 0x13))
        }
        (1, 3) => {
            let nz = ((h >> 7) & 0x20) | ((h >> 2) & 31);
            if nz == 0 {
                return None;
            }
            if rd == 2 {
                // c.addi16sp
                let imm = ((h >> 3) & 0x200) | ((h >> 2) & 0x10) | ((h << 1) & 0x40) | ((h << 4) & 0x180) | ((h << 3) & 0x20);
                Some(i(sx(imm, 10), 2, 0, 2, 0x13))
            } else {
                // c.lui
                let imm = sx(nz, 6) as u32;
                Some(((imm & 0xFFFFF) << 12) | (rd << 7) | 0x37)
            }
        }
        (1, 4) => {
            let f2 = (h >> 10) & 3;
            let shamt = ((h >> 7) & 0x20) | ((h >> 2) & 31);
            match f2 {
                0 => Some(i(shamt as i32, rs1p, 5, rs1p, 0x13)),            // c.srli
                1 => Some(i((shamt | 0x400) as i32, rs1p, 5, rs1p, 0x13)),  // c.srai
                2 => Some(i(sx(shamt, 6), rs1p, 7, rs1p, 0x13)),            // c.andi
                _ => {
                    let (f7, f3, opc) = match ((h >> 12) & 1, (h >> 5) & 3) {
                        (0, 0) => (0x20, 0, 0x33), // c.sub
                        (0, 1) => (0, 4, 0x33),    // c.xor
                        (0, 2) => (0, 6, 0x33),    // c.or
                        (0, 3) => (0, 7, 0x33),    // c.and
                        (1, 0) => (0x20, 0, 0x3B), // c.subw
                        (1, 1) => (0, 0, 0x3B),    // c.addw
                        _ => return None,
                    };
                    Some(r(f7, rdp, rs1p, f3, rs1p, opc))
                }
            }
        }
        (1, 5) => {
            // c.j
            let imm = ((h >> 1) & 0x800) | ((h >> 7) & 0x10) | ((h >> 1) & 0x300) | ((h << 2) & 0x400) | ((h >> 1) & 0x40) | ((h << 1) & 0x80) | ((h >> 2) & 0xE) | ((h << 3) & 0x20);
            let imm = sx(imm, 12) as u32;
            Some((((imm >> 20) & 1) << 31) | (((imm >> 1) & 0x3FF) << 21) | (((imm >> 11) & 1) << 20) | (((imm >> 12) & 0xFF) << 12) | 0x6F)
        }
        (1, 6) | (1, 7) => {
            // c.beqz / c.bnez
            let imm = ((h >> 4) & 0x100) | ((h >> 7) & 0x18) | ((h << 1) & 0xC0) | ((h >> 2) & 6) | ((h << 3) & 0x20);
            let u = sx(imm, 9) as u32;
            let bits = (((u >> 12) & 1) << 31) | (((u >> 5) & 0x3F) << 25) | (((u >> 1) & 0xF) << 8) | (((u >> 11) & 1) << 7);
            Some(bits | (rs1p << 15) | ((f3 - 6) << 12) | 0x63)
        }
        (2, 0) => {
            // c.slli
            let shamt = ((h >> 7) & 0x20) | ((h >> 2) & 31);
            Some(i(shamt as i32, rd, 1, rd, 0x13))
        }
        (2, 1) | (2, 3) => {
            // c.fldsp / c.ldsp
            let imm = ((h >> 7) & 0x20) | ((h >> 2) & 0x18) | ((h << 4) & 0x1C0);
            if f3 == 3 && rd == 0 {
                return None;
            }
            Some(i(imm as i32, 2, 3, rd, if f3 == 1 { 0x07 } else { 0x03 }))
        }
        (2, 2) => {
            // c.lwsp
            let imm = ((h >> 7) & 0x20) | ((h >> 2) & 0x1C) | ((h << 4) & 0xC0);
            if rd == 0 {
                return None;
            }
            Some(i(imm as i32, 2, 2, rd, 0x03))
        }
        (2, 4) => {
            let bit12 = (h >> 12) & 1;
            match (bit12, rd, rs2) {
                (0, 0, _) => None,
                (0, _, 0) => Some(i(0, rd, 0, 0, 0x67)),          // c.jr
                (0, _, _) => Some(r(0, rs2, 0, 0, rd, 0x33)),     // c.mv
                (1, 0, 0) => Some(0x0010_0073),                   // c.ebreak
                (1, _, 0) => Some(i(0, rd, 0, 1, 0x67)),          // c.jalr
                _ => Some(r(0, rs2, rd, 0, rd, 0x33)),            // c.add
            }
        }
        (2, 5) | (2, 7) => {
            // c.fsdsp / c.sdsp
            let imm = ((h >> 7) & 0x38) | ((h >> 1) & 0x1C0);
            Some(s(imm as i32, rs2, 2, 3, if f3 == 5 { 0x27 } else { 0x23 }))
        }
        (2, 6) => {
            // c.swsp
            let imm = ((h >> 7) & 0x3C) | ((h >> 1) & 0xC0);
            Some(s(imm as i32, rs2, 2, 2, 0x23))
        }
        _ => None,
    }
}
