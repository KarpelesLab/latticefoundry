//! A test-only A64 machine-code emulator, so linked AArch64 programs run on
//! this x86-64 host (there is no `qemu-aarch64` here).
//!
//! It loads a static ELF64 `EM_AARCH64` executable (its `PT_LOAD` segments),
//! maps a stack, and executes from the entry point until the program calls
//! `exit`/`exit_group`. It covers the base A64 integer instruction set
//! (data processing, loads/stores in every addressing mode, branches), the
//! exclusive and acquire/release accesses (single-threaded), scalar floating
//! point, and the few Advanced SIMD forms compilers use for scalar code
//! (`mov`/`orr` of a whole vector, `movi`, `ins`/`umov`/`dup` of a general
//! register): enough for everything this backend emits, and for the
//! straightforward C that `clang --target=aarch64-linux-gnu` compiles, which
//! the tests link against ours to check the ABI against an independent
//! implementation. Each form is decoded from the Arm ARM's encoding tables;
//! anything else stops the run with an error naming the word.
//!
//! The stack can have a **guard page** below it: an access there ends the run
//! with [`Stop::Guard`] (the `SIGSEGV` a real guard gives), an access below
//! it with [`Stop::Skipped`] (the silent corruption stack probes prevent). The
//! emulator also records every store into the stack mapping, for the probe
//! invariant checks.

use std::collections::HashMap;

/// Why a run stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Stop {
    /// `exit`/`exit_group` with this status.
    Exit(u64),
    /// Returned to the sentinel return address (a [`Emu::call`]).
    Returned,
    /// An access inside the guard page below the stack.
    Guard,
    /// An access below the guard page: a stack move skipped over it.
    Skipped,
}

const PAGE: u64 = 4096;
/// The return address [`Emu::call`] plants in `x30`.
const SENTINEL: u64 = 0xDEAD_BEEF_0000;

/// The emulated machine.
pub(super) struct Emu {
    /// `x0`–`x30`; index 31 is unused (`sp`/`xzr` are decoded per form).
    pub(super) x: [u64; 32],
    pub(super) sp: u64,
    pub(super) pc: u64,
    /// `v0`–`v31`, all 128 bits.
    pub(super) v: [u128; 32],
    n: bool,
    z: bool,
    c: bool,
    vf: bool,
    pages: HashMap<u64, Box<[u8; PAGE as usize]>>,
    /// The stack mapping `[lo, hi)` and its guard page `[lo - 4096, lo)`.
    stack: (u64, u64),
    guard: bool,
    /// Every store into the stack mapping (address), in order.
    pub(super) stack_writes: Vec<u64>,
    /// Bytes written to file descriptors 1 and 2.
    pub(super) output: Vec<u8>,
    /// The exclusive monitor's address, if armed.
    monitor: Option<u64>,
    /// Set by a decoder that met a form it does not implement.
    unsupported: Option<String>,
}

/// The memory-access outcome of one instruction.
enum Fault {
    Guard,
    Skipped,
    Unmapped(u64),
}

fn sext(v: u64, bits: u32) -> i64 {
    ((v << (64 - bits)) as i64) >> (64 - bits)
}

fn ones(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

fn ror(v: u64, r: u32, width: u32) -> u64 {
    let v = v & ones(width);
    let r = r % width;
    if r == 0 { v } else { ((v >> r) | (v << (width - r))) & ones(width) }
}

/// The Arm ARM's `DecodeBitMasks` (wmask, tmask) for the logical-immediate
/// and bitfield forms.
fn decode_bit_masks(n: u32, imms: u32, immr: u32, immediate: bool, width: u32) -> Option<(u64, u64)> {
    let combined = (n << 6) | (!imms & 0x3F);
    if combined == 0 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 {
        return None;
    }
    let levels = ones(len) as u32;
    if immediate && (imms & levels) == levels {
        return None;
    }
    let s = imms & levels;
    let r = immr & levels;
    let diff = s.wrapping_sub(r) & levels;
    let esize = 1u32 << len;
    let welem = ones(s + 1);
    let telem = ones(diff + 1);
    let rep = |e: u64| {
        let mut out = 0u64;
        let mut k = 0;
        while k < width {
            out |= (e & ones(esize)) << k;
            k += esize;
        }
        out & ones(width)
    };
    Some((rep(ror(welem, r, esize)), rep(telem)))
}

fn f32_of(bits: u128) -> f32 {
    f32::from_bits(bits as u32)
}
fn f64_of(bits: u128) -> f64 {
    f64::from_bits(bits as u64)
}

impl Emu {
    /// An empty machine: nothing mapped.
    pub(super) fn new() -> Emu {
        Emu {
            x: [0; 32],
            sp: 0,
            pc: 0,
            v: [0; 32],
            n: false,
            z: false,
            c: false,
            vf: false,
            pages: HashMap::new(),
            stack: (0, 0),
            guard: false,
            stack_writes: Vec::new(),
            output: Vec::new(),
            monitor: None,
            unsupported: None,
        }
    }

    /// Map `[base, base + size)` (page-rounded), zeroed.
    pub(super) fn map(&mut self, base: u64, size: u64) {
        let mut p = base & !(PAGE - 1);
        while p < base + size {
            self.pages.entry(p).or_insert_with(|| Box::new([0; PAGE as usize]));
            p += PAGE;
        }
    }

    /// Map a stack of `size` bytes ending at `top` and point `sp` at its top;
    /// with `guard`, the page below it is a guard page.
    pub(super) fn map_stack(&mut self, top: u64, size: u64, guard: bool) {
        self.map(top - size, size);
        self.stack = (top - size, top);
        self.guard = guard;
        self.sp = top;
    }

    /// Copy `bytes` to `addr` (which must be mapped).
    pub(super) fn poke(&mut self, addr: u64, bytes: &[u8]) {
        for (k, &b) in bytes.iter().enumerate() {
            let a = addr + k as u64;
            self.pages.get_mut(&(a & !(PAGE - 1))).expect("poke into mapped memory")[(a % PAGE) as usize] = b;
        }
    }

    fn check(&self, addr: u64, size: u64) -> Result<(), Fault> {
        for a in [addr, addr + size - 1] {
            if !self.pages.contains_key(&(a & !(PAGE - 1))) {
                let (lo, _) = self.stack;
                if self.guard && a < lo && a >= lo - PAGE {
                    return Err(Fault::Guard);
                }
                if self.guard && a < lo - PAGE && a >= lo.saturating_sub(1 << 30) {
                    return Err(Fault::Skipped);
                }
                return Err(Fault::Unmapped(a));
            }
        }
        Ok(())
    }

    fn read(&self, addr: u64, size: u64) -> Result<u128, Fault> {
        self.check(addr, size)?;
        let mut v = 0u128;
        for k in 0..size {
            let a = addr + k;
            v |= u128::from(self.pages[&(a & !(PAGE - 1))][(a % PAGE) as usize]) << (8 * k);
        }
        Ok(v)
    }

    fn write(&mut self, addr: u64, size: u64, v: u128) -> Result<(), Fault> {
        self.check(addr, size)?;
        if addr >= self.stack.0 && addr < self.stack.1 {
            self.stack_writes.push(addr);
        }
        if self.monitor.is_some_and(|m| m >> 4 == addr >> 4) {
            self.monitor = None;
        }
        for k in 0..size {
            let a = addr + k;
            self.pages.get_mut(&(a & !(PAGE - 1))).unwrap()[(a % PAGE) as usize] = (v >> (8 * k)) as u8;
        }
        Ok(())
    }

    /// Read `size` bytes at `addr` (for tests inspecting memory).
    pub(super) fn peek(&self, addr: u64, size: u64) -> Option<u128> {
        self.read(addr, size).ok()
    }

    /// Load the `PT_LOAD` segments of a little-endian ELF64 executable and
    /// set `pc` to its entry point.
    pub(super) fn load_elf(&mut self, elf: &[u8]) -> Result<(), String> {
        let u16_at = |o: usize| u16::from_le_bytes([elf[o], elf[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(elf[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(elf[o..o + 8].try_into().unwrap());
        if elf.get(..4) != Some(b"\x7fELF") || elf[4] != 2 || u16_at(18) != 183 {
            return Err("not an ELF64 AArch64 file".into());
        }
        let (phoff, phentsize, phnum) = (u64_at(32) as usize, u16_at(54) as usize, u16_at(56) as usize);
        for i in 0..phnum {
            let ph = phoff + i * phentsize;
            if u32_at(ph) != 1 {
                continue; // PT_LOAD only
            }
            let (off, vaddr, filesz, memsz) =
                (u64_at(ph + 8) as usize, u64_at(ph + 16), u64_at(ph + 32) as usize, u64_at(ph + 40));
            self.map(vaddr, memsz.max(1));
            self.poke(vaddr, &elf[off..off + filesz]);
        }
        self.pc = u64_at(24);
        Ok(())
    }

    /// Run until an exit, a fault, a return to the sentinel, or `budget`
    /// instructions.
    pub(super) fn run(&mut self, budget: u64) -> Result<Stop, String> {
        for _ in 0..budget {
            if self.pc == SENTINEL {
                return Ok(Stop::Returned);
            }
            let pc = self.pc;
            let w = match self.read(pc, 4) {
                Ok(w) => w as u32,
                Err(_) => return Err(format!("fetch from unmapped {pc:#x}")),
            };
            self.pc = pc + 4;
            match self.step(w) {
                Ok(None) => {}
                Ok(Some(stop)) => return Ok(stop),
                Err(Fault::Guard) => return Ok(Stop::Guard),
                Err(Fault::Skipped) => return Ok(Stop::Skipped),
                Err(Fault::Unmapped(a)) => {
                    return Err(format!("access to unmapped {a:#x} at pc {pc:#x} (word {w:#010x})"));
                }
            }
            if self.unsupported.is_some() {
                let what = self.unsupported.take().unwrap();
                return Err(format!("unsupported instruction {w:#010x} at {pc:#x}: {what}"));
            }
        }
        Err("instruction budget exhausted".into())
    }

    /// Call the function at `addr` with integer arguments `args` (in
    /// `x0`..), returning when it does (`x30` holds a sentinel).
    pub(super) fn call(&mut self, addr: u64, args: &[u64], budget: u64) -> Result<Stop, String> {
        for (k, &a) in args.iter().enumerate() {
            self.x[k] = a;
        }
        self.x[30] = SENTINEL;
        self.pc = addr;
        self.run(budget)
    }

    // --- register access ---------------------------------------------------

    /// A general register read: 31 is `xzr`.
    fn xr(&self, r: u32) -> u64 {
        if r == 31 { 0 } else { self.x[r as usize] }
    }
    /// A general register read where 31 is `sp`.
    fn xs(&self, r: u32) -> u64 {
        if r == 31 { self.sp } else { self.x[r as usize] }
    }
    /// A write of `v` (at `sf` width, zero-extended) where 31 is `xzr`.
    fn wr(&mut self, r: u32, v: u64, sf: bool) {
        if r != 31 {
            self.x[r as usize] = if sf { v } else { v & 0xFFFF_FFFF };
        }
    }
    /// A write where 31 is `sp`.
    fn ws(&mut self, r: u32, v: u64, sf: bool) {
        let v = if sf { v } else { v & 0xFFFF_FFFF };
        if r == 31 { self.sp = v } else { self.x[r as usize] = v }
    }

    fn cond(&self, cond: u32) -> bool {
        let r = match cond >> 1 {
            0 => self.z,
            1 => self.c,
            2 => self.n,
            3 => self.vf,
            4 => self.c && !self.z,
            5 => self.n == self.vf,
            6 => self.n == self.vf && !self.z,
            _ => true,
        };
        if cond & 1 == 1 && cond != 0xF { !r } else { r }
    }

    /// `a + b + carry` at `width`, setting NZCV when `set`.
    fn add_with_carry(&mut self, a: u64, b: u64, carry: bool, width: u32, set: bool) -> u64 {
        let m = ones(width);
        let (a, b) = (a & m, b & m);
        let wide = u128::from(a) + u128::from(b) + u128::from(carry);
        let r = (wide as u64) & m;
        if set {
            let sign = |v: u64| v >> (width - 1) & 1 == 1;
            self.n = sign(r);
            self.z = r == 0;
            self.c = wide > u128::from(m);
            self.vf = sign(a) == sign(b) && sign(r) != sign(a);
        }
        r
    }

    fn shift_reg(&self, v: u64, kind: u32, amount: u32, width: u32) -> u64 {
        let v = v & ones(width);
        match kind {
            0 => (v << amount) & ones(width),
            1 => v >> amount,
            2 => (sext(v, width) >> amount) as u64 & ones(width),
            _ => ror(v, amount, width),
        }
    }

    fn extend_reg(&self, v: u64, option: u32, shift: u32) -> u64 {
        let e = match option {
            0 => v & 0xFF,
            1 => v & 0xFFFF,
            2 => v & 0xFFFF_FFFF,
            3 => v,
            4 => sext(v & 0xFF, 8) as u64,
            5 => sext(v & 0xFFFF, 16) as u64,
            6 => sext(v & 0xFFFF_FFFF, 32) as u64,
            _ => v,
        };
        e << shift
    }

    fn unsupported(&mut self, what: &str) -> Result<Option<Stop>, Fault> {
        self.unsupported = Some(what.to_owned());
        Ok(None)
    }
}

/// What one instruction did: keep going, stop the run, or fault.
type Step = Result<Option<Stop>, Fault>;

impl Emu {
    /// Execute one instruction word (`pc` already points past it).
    fn step(&mut self, w: u32) -> Step {
        match (w >> 25) & 0xF {
            0b1000 | 0b1001 => self.dp_imm(w),
            0b1010 | 0b1011 => self.branch(w),
            0b0100 | 0b0110 | 0b1100 | 0b1110 => self.ldst(w),
            0b0101 | 0b1101 => self.dp_reg(w),
            0b0111 | 0b1111 => self.fp_simd(w),
            _ => self.unsupported("unallocated or SVE/SME encoding space"),
        }
    }

    // --- data processing, immediate ----------------------------------------

    fn dp_imm(&mut self, w: u32) -> Step {
        let (rd, rn) = (w & 31, (w >> 5) & 31);
        let sf = w >> 31 == 1;
        let width = if sf { 64 } else { 32 };
        match (w >> 23) & 0x3F {
            // adr / adrp
            0b100000 | 0b100001 => {
                let imm = sext(u64::from((((w >> 5) & 0x7FFFF) << 2) | ((w >> 29) & 3)), 21);
                let pc = self.pc - 4;
                let v = if sf {
                    (pc & !0xFFF).wrapping_add((imm << 12) as u64)
                } else {
                    pc.wrapping_add(imm as u64)
                };
                self.wr(rd, v, true);
            }
            // add / sub (immediate), optionally setting flags
            0b100010 => {
                let mut imm = u64::from((w >> 10) & 0xFFF);
                if w & (1 << 22) != 0 {
                    imm <<= 12;
                }
                let (sub, s) = (w & (1 << 30) != 0, w & (1 << 29) != 0);
                let a = self.xs(rn);
                let r = if sub {
                    self.add_with_carry(a, !imm, true, width, s)
                } else {
                    self.add_with_carry(a, imm, false, width, s)
                };
                if s { self.wr(rd, r, sf) } else { self.ws(rd, r, sf) }
            }
            // logical (immediate)
            0b100100 => {
                let Some((imm, _)) = decode_bit_masks((w >> 22) & 1, (w >> 10) & 0x3F, (w >> 16) & 0x3F, true, width)
                else {
                    return self.unsupported("reserved logical immediate");
                };
                let a = self.xr(rn);
                let opc = (w >> 29) & 3;
                let r = match opc {
                    1 => a | imm,
                    2 => a ^ imm,
                    _ => a & imm,
                } & ones(width);
                if opc == 3 {
                    self.set_nz(r, width);
                    self.wr(rd, r, sf);
                } else {
                    self.ws(rd, r, sf);
                }
            }
            // move wide: movn / movz / movk
            0b100101 => {
                let hw = (w >> 21) & 3;
                let imm = u64::from((w >> 5) & 0xFFFF) << (16 * hw);
                match (w >> 29) & 3 {
                    0 => self.wr(rd, !imm, sf),
                    2 => self.wr(rd, imm, sf),
                    3 => {
                        let old = self.xr(rd);
                        self.wr(rd, (old & !(0xFFFF << (16 * hw))) | imm, sf);
                    }
                    _ => return self.unsupported("move wide opc 01"),
                }
            }
            // bitfield: sbfm / bfm / ubfm
            0b100110 => {
                let (opc, immr, imms) = ((w >> 29) & 3, (w >> 16) & 0x3F, (w >> 10) & 0x3F);
                let Some((wmask, tmask)) = decode_bit_masks((w >> 22) & 1, imms, immr, false, width) else {
                    return self.unsupported("reserved bitfield immediate");
                };
                let src = self.xr(rn) & ones(width);
                let dst = if opc == 1 { self.xr(rd) & ones(width) } else { 0 };
                let bot = (dst & !wmask) | (ror(src, immr, width) & wmask);
                let top = if opc == 0 { if (src >> imms) & 1 == 1 { ones(width) } else { 0 } } else { dst };
                self.wr(rd, (top & !tmask) | (bot & tmask), sf);
            }
            // extr
            0b100111 => {
                let lsb = (w >> 10) & 0x3F;
                let (hi, lo) = (self.xr(rn) & ones(width), self.xr((w >> 16) & 31) & ones(width));
                let r = if lsb == 0 { lo } else { (lo >> lsb) | (hi << (width - lsb)) };
                self.wr(rd, r & ones(width), sf);
            }
            _ => return self.unsupported("data processing (immediate)"),
        }
        Ok(None)
    }

    fn set_nz(&mut self, r: u64, width: u32) {
        self.n = (r >> (width - 1)) & 1 == 1;
        self.z = r & ones(width) == 0;
        self.c = false;
        self.vf = false;
    }

    // --- branches, exceptions, system ---------------------------------------

    fn branch(&mut self, w: u32) -> Step {
        let pc = self.pc - 4;
        let rel = |imm: u32, bits: u32| pc.wrapping_add((sext(u64::from(imm), bits) * 4) as u64);
        if w & 0x7C00_0000 == 0x1400_0000 {
            // b / bl
            if w >> 31 == 1 {
                self.x[30] = self.pc;
            }
            self.pc = rel(w & 0x03FF_FFFF, 26);
        } else if w & 0x7E00_0000 == 0x3400_0000 {
            // cbz / cbnz
            let v = self.xr(w & 31) & if w >> 31 == 1 { u64::MAX } else { 0xFFFF_FFFF };
            if (v != 0) == (w & (1 << 24) != 0) {
                self.pc = rel((w >> 5) & 0x7FFFF, 19);
            }
        } else if w & 0x7E00_0000 == 0x3600_0000 {
            // tbz / tbnz
            let bit = ((w >> 31) << 5) | ((w >> 19) & 31);
            if ((self.xr(w & 31) >> bit) & 1 == 1) == (w & (1 << 24) != 0) {
                self.pc = rel((w >> 5) & 0x3FFF, 14);
            }
        } else if w & 0xFF00_0010 == 0x5400_0000 {
            // b.cond
            if self.cond(w & 0xF) {
                self.pc = rel((w >> 5) & 0x7FFFF, 19);
            }
        } else if w & 0xFFE0_001F == 0xD400_0001 {
            return self.syscall();
        } else if w & 0xFFE0_001F == 0xD420_0000 {
            return self.unsupported("brk (a trap)");
        } else if w & 0xFFFF_F01F == 0xD503_201F || w & 0xFFFF_F01F == 0xD503_301F {
            // hints (nop, ...) and barriers (dsb, dmb, isb, clrex)
            if w & 0xFFFF_F0FF == 0xD503_305F {
                self.monitor = None;
            }
        } else if w & 0xFF9F_FC1F == 0xD61F_0000 {
            // br / blr / ret
            let target = self.xr((w >> 5) & 31);
            if (w >> 21) & 3 == 1 {
                self.x[30] = self.pc;
            }
            self.pc = target;
        } else {
            return self.unsupported("branch / system");
        }
        Ok(None)
    }

    /// `svc #0`: `exit` (93) and `exit_group` (94) stop the run; `write`
    /// (64) to standard output or error is captured.
    fn syscall(&mut self) -> Step {
        match self.x[8] {
            93 | 94 => Ok(Some(Stop::Exit(self.x[0]))),
            64 if self.x[0] == 1 || self.x[0] == 2 => {
                let (buf, len) = (self.x[1], self.x[2]);
                for k in 0..len {
                    let b = self.read(buf + k, 1)? as u8;
                    self.output.push(b);
                }
                self.x[0] = len;
                Ok(None)
            }
            nr => self.unsupported(&format!("system call {nr}")),
        }
    }

    // --- loads and stores --------------------------------------------------

    fn ldst(&mut self, w: u32) -> Step {
        let rn = (w >> 5) & 31;
        if w & 0x3F00_0000 == 0x0800_0000 {
            return self.exclusive(w);
        }
        if w & 0x3B00_0000 == 0x1800_0000 {
            // load register (literal)
            let addr = (self.pc - 4).wrapping_add((sext(u64::from((w >> 5) & 0x7FFFF), 19) * 4) as u64);
            let rt = w & 31;
            if (w >> 26) & 1 == 1 {
                let bytes = 4u64 << (w >> 30);
                self.v[rt as usize] = self.read(addr, bytes)?;
            } else {
                match w >> 30 {
                    0 => self.wr(rt, self.read(addr, 4)? as u64, true),
                    1 => self.wr(rt, self.read(addr, 8)? as u64, true),
                    2 => self.wr(rt, sext(self.read(addr, 4)? as u64, 32) as u64, true),
                    _ => {} // prfm
                }
            }
            return Ok(None);
        }
        if w & 0x3A00_0000 == 0x2800_0000 {
            return self.pair(w);
        }
        let base = self.xs(rn);
        if w & 0x3B00_0000 == 0x3900_0000 {
            // unsigned scaled offset
            let off = u64::from((w >> 10) & 0xFFF) << self.scale(w);
            self.single(w, base.wrapping_add(off))?;
        } else if w & 0x3B20_0000 == 0x3800_0000 {
            // 9-bit signed offset: unscaled, post-index, unprivileged, pre-index
            let imm = sext(u64::from((w >> 12) & 0x1FF), 9) as u64;
            match (w >> 10) & 3 {
                1 => {
                    self.single(w, base)?;
                    self.ws(rn, base.wrapping_add(imm), true);
                }
                3 => {
                    let a = base.wrapping_add(imm);
                    self.single(w, a)?;
                    self.ws(rn, a, true);
                }
                _ => self.single(w, base.wrapping_add(imm))?,
            }
        } else if w & 0x3B20_0C00 == 0x3820_0800 {
            // register offset, extended and optionally scaled
            let shift = if (w >> 12) & 1 == 1 { self.scale(w) } else { 0 };
            let off = self.extend_reg(self.xr((w >> 16) & 31), (w >> 13) & 7, shift);
            self.single(w, base.wrapping_add(off))?;
        } else {
            return self.unsupported("load/store");
        }
        Ok(None)
    }

    /// log2 of the access size of a single-register load/store.
    fn scale(&self, w: u32) -> u32 {
        let size = w >> 30;
        if (w >> 26) & 1 == 1 && (w >> 23) & 1 == 1 { 4 } else { size }
    }

    /// One single-register transfer at `addr` (GPR or SIMD/FP, by `V`).
    fn single(&mut self, w: u32, addr: u64) -> Result<(), Fault> {
        let rt = w & 31;
        let opc = (w >> 22) & 3;
        let bytes = 1u64 << self.scale(w);
        if (w >> 26) & 1 == 1 {
            if opc & 1 == 1 {
                self.v[rt as usize] = self.read(addr, bytes)?;
            } else {
                let mask = if bytes == 16 { u128::MAX } else { (1u128 << (8 * bytes)) - 1 };
                self.write(addr, bytes, self.v[rt as usize] & mask)?;
            }
            return Ok(());
        }
        let bits = 8 * bytes as u32;
        match opc {
            0 => self.write(addr, bytes, u128::from(self.xr(rt)))?,
            1 => {
                let v = self.read(addr, bytes)? as u64;
                self.wr(rt, v, true);
            }
            2 if bytes == 8 => {} // prfm
            2 => {
                let v = self.read(addr, bytes)? as u64;
                self.wr(rt, sext(v, bits) as u64, true);
            }
            _ => {
                let v = self.read(addr, bytes)? as u64;
                self.wr(rt, sext(v, bits) as u64, false);
            }
        }
        Ok(())
    }

    /// `ldp`/`stp` (and `ldpsw`, `ldnp`/`stnp`), GPR or SIMD/FP.
    fn pair(&mut self, w: u32) -> Step {
        let (rt, rt2, rn) = (w & 31, (w >> 10) & 31, (w >> 5) & 31);
        let opc = w >> 30;
        let fp = (w >> 26) & 1 == 1;
        let load = (w >> 22) & 1 == 1;
        let scale = if fp { 2 + opc } else { 2 + (opc >> 1) };
        let bytes = 1u64 << scale;
        let off = (sext(u64::from((w >> 15) & 0x7F), 7) << scale) as u64;
        let base = self.xs(rn);
        let mode = (w >> 23) & 3;
        let addr = if mode == 1 { base } else { base.wrapping_add(off) };
        for (k, r) in [(0, rt), (1, rt2)] {
            let a = addr + k * bytes;
            if fp {
                if load {
                    self.v[r as usize] = self.read(a, bytes)?;
                } else {
                    let mask = if bytes == 16 { u128::MAX } else { (1u128 << (8 * bytes)) - 1 };
                    self.write(a, bytes, self.v[r as usize] & mask)?;
                }
            } else if load {
                let v = self.read(a, bytes)? as u64;
                let v = if opc == 1 { sext(v, 32) as u64 } else { v };
                self.wr(r, v, opc != 0);
            } else {
                self.write(a, bytes, u128::from(self.xr(r)))?;
            }
        }
        if mode == 1 || mode == 3 {
            self.ws(rn, base.wrapping_add(off), true);
        }
        Ok(None)
    }

    /// The exclusive and acquire/release single-register accesses. The
    /// machine is single-threaded: a store-exclusive succeeds while the
    /// monitor set by the matching load-exclusive is armed.
    fn exclusive(&mut self, w: u32) -> Step {
        let (rt, rn, rs) = (w & 31, (w >> 5) & 31, (w >> 16) & 31);
        let bytes = 1u64 << (w >> 30);
        let (o2, load, o1) = ((w >> 23) & 1, (w >> 22) & 1 == 1, (w >> 21) & 1);
        if o1 == 1 {
            return self.unsupported("exclusive pair / compare-and-swap");
        }
        let addr = self.xs(rn);
        if load {
            let v = self.read(addr, bytes)? as u64;
            self.wr(rt, v, true);
            if o2 == 0 {
                self.monitor = Some(addr);
            }
        } else if o2 == 0 {
            if self.monitor == Some(addr) {
                self.write(addr, bytes, u128::from(self.xr(rt)))?;
                self.monitor = None;
                self.wr(rs, 0, false);
            } else {
                self.wr(rs, 1, false);
            }
        } else {
            self.write(addr, bytes, u128::from(self.xr(rt)))?;
        }
        Ok(None)
    }

    // --- data processing, register -----------------------------------------

    fn dp_reg(&mut self, w: u32) -> Step {
        let (rd, rn, rm) = (w & 31, (w >> 5) & 31, (w >> 16) & 31);
        let sf = w >> 31 == 1;
        let width = if sf { 64 } else { 32 };
        let (sub, s) = (w & (1 << 30) != 0, w & (1 << 29) != 0);
        if w & 0x1F00_0000 == 0x0A00_0000 {
            // logical (shifted register), with optional inversion
            let mut b = self.shift_reg(self.xr(rm), (w >> 22) & 3, (w >> 10) & 0x3F, width);
            if (w >> 21) & 1 == 1 {
                b = !b & ones(width);
            }
            let a = self.xr(rn) & ones(width);
            let opc = (w >> 29) & 3;
            let r = match opc {
                1 => a | b,
                2 => a ^ b,
                _ => a & b,
            };
            if opc == 3 {
                self.set_nz(r, width);
            }
            self.wr(rd, r, sf);
        } else if w & 0x1F20_0000 == 0x0B00_0000 {
            // add / sub (shifted register)
            let b = self.shift_reg(self.xr(rm), (w >> 22) & 3, (w >> 10) & 0x3F, width);
            let a = self.xr(rn);
            let r = if sub {
                self.add_with_carry(a, !b, true, width, s)
            } else {
                self.add_with_carry(a, b, false, width, s)
            };
            self.wr(rd, r, sf);
        } else if w & 0x1F20_0000 == 0x0B20_0000 {
            // add / sub (extended register): `sp` allowed
            let b = self.extend_reg(self.xr(rm), (w >> 13) & 7, (w >> 10) & 7);
            let a = self.xs(rn);
            let r = if sub {
                self.add_with_carry(a, !b, true, width, s)
            } else {
                self.add_with_carry(a, b, false, width, s)
            };
            if s { self.wr(rd, r, sf) } else { self.ws(rd, r, sf) }
        } else if w & 0x1FE0_FC00 == 0x1A00_0000 {
            // adc / sbc
            let b = if sub { !self.xr(rm) } else { self.xr(rm) };
            let r = self.add_with_carry(self.xr(rn), b, self.c, width, s);
            self.wr(rd, r, sf);
        } else if w & 0x1FE0_0000 == 0x1A40_0000 {
            // ccmn / ccmp (register or immediate)
            if self.cond((w >> 12) & 0xF) {
                let b = if w & (1 << 11) != 0 { u64::from(rm) } else { self.xr(rm) };
                let a = self.xr(rn);
                if sub {
                    self.add_with_carry(a, !b, true, width, true);
                } else {
                    self.add_with_carry(a, b, false, width, true);
                }
            } else {
                let f = w & 0xF;
                (self.n, self.z, self.c, self.vf) = (f & 8 != 0, f & 4 != 0, f & 2 != 0, f & 1 != 0);
            }
        } else if w & 0x1FE0_0000 == 0x1A80_0000 {
            // csel / csinc / csinv / csneg
            let r = if self.cond((w >> 12) & 0xF) {
                self.xr(rn)
            } else {
                let b = self.xr(rm);
                match (sub, (w >> 10) & 1) {
                    (false, 0) => b,
                    (false, _) => b.wrapping_add(1),
                    (true, 0) => !b,
                    (true, _) => (!b).wrapping_add(1),
                }
            };
            self.wr(rd, r & ones(width), sf);
        } else if w & 0x5FE0_0000 == 0x1AC0_0000 {
            // data processing (2 source)
            let (a, b) = (self.xr(rn) & ones(width), self.xr(rm) & ones(width));
            let r = match (w >> 10) & 0x3F {
                0b000010 => a.checked_div(b).unwrap_or(0),
                0b000011 => {
                    let (sa, sb) = (sext(a, width), sext(b, width));
                    if sb == 0 { 0 } else { sa.wrapping_div(sb) as u64 }
                }
                0b001000 => (a << (b % u64::from(width))) & ones(width),
                0b001001 => a >> (b % u64::from(width)),
                0b001010 => (sext(a, width) >> (b % u64::from(width))) as u64,
                0b001011 => ror(a, (b % u64::from(width)) as u32, width),
                _ => return self.unsupported("data processing (2 source)"),
            };
            self.wr(rd, r & ones(width), sf);
        } else if w & 0x5FE0_0000 == 0x5AC0_0000 {
            // data processing (1 source)
            let a = self.xr(rn) & ones(width);
            let r = match (w >> 10) & 0x3F {
                0 => a.reverse_bits() >> (64 - width),
                1 => {
                    let mut r = 0;
                    for k in (0..width).step_by(16) {
                        r |= u64::from(((a >> k) as u16).swap_bytes()) << k;
                    }
                    r
                }
                2 if sf => ((a as u32).swap_bytes() as u64) | (u64::from(((a >> 32) as u32).swap_bytes()) << 32),
                2 | 3 => a.swap_bytes() >> (64 - width),
                4 => u64::from((a << (64 - width)).leading_zeros().min(width)),
                _ => return self.unsupported("data processing (1 source)"),
            };
            self.wr(rd, r, sf);
        } else if w & 0x1F00_0000 == 0x1B00_0000 {
            // data processing (3 source)
            let (a, b, ra) = (self.xr(rn), self.xr(rm), self.xr((w >> 10) & 31));
            let neg = w & (1 << 15) != 0;
            let acc = |p: u64| if neg { ra.wrapping_sub(p) } else { ra.wrapping_add(p) };
            let r = match (w >> 21) & 7 {
                0 => acc(a.wrapping_mul(b)),
                1 => acc((sext(a, 32).wrapping_mul(sext(b, 32))) as u64),
                5 => acc((a & 0xFFFF_FFFF).wrapping_mul(b & 0xFFFF_FFFF)),
                2 => ((i128::from(a as i64) * i128::from(b as i64)) >> 64) as u64,
                6 => ((u128::from(a) * u128::from(b)) >> 64) as u64,
                _ => return self.unsupported("data processing (3 source)"),
            };
            self.wr(rd, r & ones(width), sf);
        } else {
            return self.unsupported("data processing (register)");
        }
        Ok(None)
    }

    // --- scalar floating point and the scalar-use SIMD forms -----------------

    fn fget(&self, r: u32, dbl: bool) -> f64 {
        if dbl { f64_of(self.v[r as usize]) } else { f64::from(f32_of(self.v[r as usize])) }
    }

    fn fset64(&mut self, r: u32, x: f64) {
        self.v[r as usize] = u128::from(x.to_bits());
    }

    fn fset32(&mut self, r: u32, x: f32) {
        self.v[r as usize] = u128::from(x.to_bits());
    }

    fn fp_simd(&mut self, w: u32) -> Step {
        let (rd, rn, rm) = (w & 31, (w >> 5) & 31, (w >> 16) & 31);
        if w & 0x5F20_0000 == 0x1E20_0000 {
            let ptype = (w >> 22) & 3;
            if ptype > 1 {
                return self.unsupported("half-precision floating point");
            }
            let dbl = ptype == 1;
            if w & 0xFC00 == 0 {
                return self.fp_int(w, dbl);
            }
            if w & 0x3C00 == 0x2000 {
                // fcmp / fcmpe (with a register or zero)
                let a = self.fget(rn, dbl);
                let b = if w & 8 != 0 { 0.0 } else { self.fget(rm, dbl) };
                (self.n, self.z, self.c, self.vf) = if a.is_nan() || b.is_nan() {
                    (false, false, true, true)
                } else if a < b {
                    (true, false, false, false)
                } else if a == b {
                    (false, true, true, false)
                } else {
                    (false, false, true, false)
                };
                return Ok(None);
            }
            if w & 0x7C00 == 0x4000 {
                // one source: fmov, fabs, fneg, fsqrt, fcvt
                let a = self.fget(rn, dbl);
                match (w >> 15) & 0x3F {
                    0 => self.v[rd as usize] = self.v[rn as usize] & if dbl { u128::from(u64::MAX) } else { 0xFFFF_FFFF },
                    1..=3 if dbl => {
                        let r = match (w >> 15) & 3 {
                            1 => a.abs(),
                            2 => -a,
                            _ => a.sqrt(),
                        };
                        self.fset64(rd, r);
                    }
                    1..=3 => {
                        let a = a as f32;
                        let r = match (w >> 15) & 3 {
                            1 => a.abs(),
                            2 => -a,
                            _ => a.sqrt(),
                        };
                        self.fset32(rd, r);
                    }
                    4 => self.fset32(rd, a as f32),
                    5 => self.fset64(rd, a),
                    _ => return self.unsupported("floating-point data processing (1 source)"),
                }
                return Ok(None);
            }
            if w & 0x0C00 == 0x0800 {
                // two sources
                let op = (w >> 12) & 0xF;
                if dbl {
                    let (a, b) = (self.fget(rn, true), self.fget(rm, true));
                    let r = match op {
                        0 => a * b,
                        1 => a / b,
                        2 => a + b,
                        3 => a - b,
                        4 | 5 if a.is_nan() || b.is_nan() => f64::NAN,
                        4 | 6 => a.max(b),
                        5 | 7 => a.min(b),
                        8 => -(a * b),
                        _ => return self.unsupported("floating-point data processing (2 source)"),
                    };
                    self.fset64(rd, r);
                } else {
                    let (a, b) = (f32_of(self.v[rn as usize]), f32_of(self.v[rm as usize]));
                    let r = match op {
                        0 => a * b,
                        1 => a / b,
                        2 => a + b,
                        3 => a - b,
                        4 | 5 if a.is_nan() || b.is_nan() => f32::NAN,
                        4 | 6 => a.max(b),
                        5 | 7 => a.min(b),
                        8 => -(a * b),
                        _ => return self.unsupported("floating-point data processing (2 source)"),
                    };
                    self.fset32(rd, r);
                }
                return Ok(None);
            }
            if w & 0x0C00 == 0x0C00 {
                // fcsel
                let src = if self.cond((w >> 12) & 0xF) { rn } else { rm };
                self.v[rd as usize] = self.v[src as usize] & if dbl { u128::from(u64::MAX) } else { 0xFFFF_FFFF };
                return Ok(None);
            }
            if w & 0x1FE0 == 0x1000 {
                // fmov (immediate): VFPExpandImm
                let imm8 = (w >> 13) & 0xFF;
                let (sign, b6, e54, frac) = ((imm8 >> 7) & 1, (imm8 >> 6) & 1, (imm8 >> 4) & 3, imm8 & 0xF);
                let rep = |n: u32| if b6 == 1 { ones(n) } else { 0 };
                self.v[rd as usize] = if dbl {
                    let exp = (u64::from(b6 ^ 1) << 10) | (rep(8) << 2) | u64::from(e54);
                    u128::from((u64::from(sign) << 63) | (exp << 52) | (u64::from(frac) << 48))
                } else {
                    let exp = (u64::from(b6 ^ 1) << 7) | (rep(5) << 2) | u64::from(e54);
                    u128::from((u64::from(sign) << 31) | (exp << 23) | (u64::from(frac) << 19))
                };
                return Ok(None);
            }
            return self.unsupported("scalar floating point");
        }
        if w & 0x5F00_0000 == 0x1F00_0000 {
            // fmadd / fmsub / fnmadd / fnmsub (fused)
            let dbl = (w >> 22) & 3 == 1;
            let (a, b, c) = (self.fget(rn, dbl), self.fget(rm, dbl), self.fget((w >> 10) & 31, dbl));
            let (o1, o0) = ((w >> 21) & 1, (w >> 15) & 1);
            let (pa, pc) = match (o1, o0) {
                (0, 0) => (a, c),
                (0, _) => (-a, c),
                (_, 0) => (-a, -c),
                _ => (a, -c),
            };
            if dbl {
                self.fset64(rd, pa.mul_add(b, pc));
            } else {
                self.fset32(rd, (pa as f32).mul_add(b as f32, pc as f32));
            }
            return Ok(None);
        }
        if w & 0xDF3E_0C00 == 0x5E20_0800 {
            // scalar two-register miscellaneous: the int <-> float conversions
            // within the SIMD/FP file (`scvtf d0, d1`, `fcvtzs d0, d1`, ...)
            let (u, dbl) = ((w >> 29) & 1 == 1, (w >> 22) & 1 == 1);
            let width = if dbl { 64 } else { 32 };
            let raw = self.v[rn as usize] as u64 & ones(width);
            match ((w >> 23) & 1, (w >> 12) & 0x1F) {
                (0, 0b11101) => {
                    let x = if u { raw as f64 } else { sext(raw, width) as f64 };
                    if dbl { self.fset64(rd, x) } else { self.fset32(rd, x as f32) }
                }
                (1, 0b11011) => {
                    let x = self.fget(rn, dbl);
                    let r = match (u, dbl) {
                        (false, true) => x as i64 as u64,
                        (false, false) => u64::from(x as i32 as u32),
                        (true, true) => x as u64,
                        (true, false) => u64::from(x as u32),
                    };
                    self.v[rd as usize] = u128::from(r);
                }
                _ => return self.unsupported("Advanced SIMD scalar two-register"),
            }
            return Ok(None);
        }
        let q = (w >> 30) & 1 == 1;
        let lanes_mask = if q { u128::MAX } else { u128::from(u64::MAX) };
        if w & 0xBFE0_FC00 == 0x0EA0_1C00 {
            // orr Vd.T, Vn.T, Vm.T (`mov` when Vn = Vm)
            self.v[rd as usize] = (self.v[rn as usize] | self.v[rm as usize]) & lanes_mask;
            return Ok(None);
        }
        if w & 0x9FF8_0C00 == 0x0F00_0400 {
            // movi (the 8-bit, byte-mask and shifted 32-bit forms)
            let imm8 = u64::from((((w >> 16) & 7) << 5) | ((w >> 5) & 31));
            let (op, cmode) = ((w >> 29) & 1, (w >> 12) & 0xF);
            let lane = match (op, cmode) {
                (1, 0b1110) => (0..8).fold(0u64, |acc, k| acc | if imm8 >> k & 1 == 1 { 0xFF << (8 * k) } else { 0 }),
                (0, 0b1110) => imm8 * 0x0101_0101_0101_0101,
                (0, c) if c & 1 == 0 && c < 8 => (imm8 << (8 * (c >> 1))) * 0x1_0000_0001,
                _ => return self.unsupported("movi form"),
            };
            self.v[rd as usize] = (u128::from(lane) | (u128::from(lane) << 64)) & lanes_mask;
            return Ok(None);
        }
        let imm5 = (w >> 16) & 31;
        let esize = 8u32 << imm5.trailing_zeros().min(3);
        let index = imm5 >> (imm5.trailing_zeros() + 1);
        let emask = (1u128 << esize) - 1;
        if w & 0xFFE0_FC00 == 0x4E00_1C00 {
            // ins Vd.T[index], Rn
            let sh = esize * index;
            let v = self.v[rd as usize] & !(emask << sh);
            self.v[rd as usize] = v | ((u128::from(self.xr(rn)) & emask) << sh);
            return Ok(None);
        }
        if w & 0xBFE0_FC00 == 0x0E00_3C00 {
            // umov Rd, Vn.T[index]
            let v = (self.v[rn as usize] >> (esize * index)) & emask;
            self.wr(rd, v as u64, q);
            return Ok(None);
        }
        if w & 0xBFE0_FC00 == 0x0E00_0C00 {
            // dup Vd.T, Rn
            let e = u128::from(self.xr(rn)) & emask;
            let mut v = 0u128;
            for k in 0..(128 / esize) {
                v |= e << (k * esize);
            }
            self.v[rd as usize] = v & lanes_mask;
            return Ok(None);
        }
        self.unsupported("Advanced SIMD")
    }

    /// Conversions between floating point and integers, and `fmov` between
    /// the register files.
    fn fp_int(&mut self, w: u32, dbl: bool) -> Step {
        let (rd, rn) = (w & 31, (w >> 5) & 31);
        let sf = w >> 31 == 1;
        let width = if sf { 64 } else { 32 };
        match ((w >> 19) & 3, (w >> 16) & 7) {
            (0, 2) | (0, 3) => {
                let x = self.xr(rn) & ones(width);
                let v = if (w >> 16) & 7 == 2 { sext(x, width) as f64 } else { x as f64 };
                // Convert through the exact integer for f32 (a direct u64 -> f32
                // rounds once).
                if dbl {
                    self.fset64(rd, v);
                } else if (w >> 16) & 7 == 2 {
                    self.fset32(rd, sext(x, width) as f32);
                } else {
                    self.fset32(rd, x as f32);
                }
            }
            (3, 0) => {
                let a = self.fget(rn, dbl);
                let r = if sf { a as i64 as u64 } else { u64::from(a as i32 as u32) };
                self.wr(rd, r, sf);
            }
            (3, 1) => {
                let a = self.fget(rn, dbl);
                let r = if sf { a as u64 } else { u64::from(a as u32) };
                self.wr(rd, r, sf);
            }
            (0, 6) => {
                let v = self.v[rn as usize] as u64;
                self.wr(rd, if dbl { v } else { v & 0xFFFF_FFFF }, sf);
            }
            (0, 7) => {
                let x = self.xr(rn);
                self.v[rd as usize] = u128::from(if dbl { x } else { x & 0xFFFF_FFFF });
            }
            _ => return self.unsupported("floating-point / integer conversion"),
        }
        Ok(None)
    }
}

/// Run a linked static executable to its `exit`, on a fresh 8 MiB stack.
/// Returns the exit status and what it wrote to standard output/error.
pub(super) fn run_executable(elf: &[u8]) -> Result<(u64, Vec<u8>), String> {
    let mut emu = Emu::new();
    emu.load_elf(elf)?;
    emu.map_stack(0x7FFF_0000_0000, 8 << 20, true);
    match emu.run(50_000_000)? {
        Stop::Exit(code) => Ok((code, emu.output)),
        other => Err(format!("the program stopped with {other:?}")),
    }
}
