//! A Thumb-2 (ARMv7-M) instruction-set simulator for the subset the encoder
//! emits, the test harness's linker, and the Run-time ABI helpers written in
//! Rust.
//!
//! This host cannot execute Arm code, so the encoded bytes are validated by
//! running them here: the simulator **decodes the machine code** (it never
//! looks at MIR) with its own decoder, written from the ARMv7-M ARM
//! independently of the encoder, and executes it with the architectural flag
//! and `IT`-block semantics. [`link`] lays compiled objects out in a flat
//! address space and applies their relocations (`R_ARM_THM_CALL`,
//! `R_ARM_THM_MOVW_ABS_NC`/`MOVT_ABS`, `R_ARM_ABS32`); a call to an undefined
//! symbol lands on a trap address that runs the helper of that name
//! ([`aeabi`]) — the soft-float and division helpers of the RTABI,
//! implemented over IEEE bits with the host's floating-point arithmetic.
//!
//! An unsupported encoding, an `udf`, a branch to an even (Arm-state) address
//! or an exhausted step budget is an error, so a mis-encoded instruction
//! cannot pass silently.

use std::collections::HashMap;

use crate::mc::object::{ObjectModule, RelocKind, SectionKind, SymbolValue, write_thumb_field};

/// The address a top-level call returns to: reaching it ends the run.
pub(super) const EXIT: u32 = 0x00ff_fff0;
/// Where the trap addresses of undefined symbols (runtime helpers) start.
const HELPER_BASE: u32 = 0x00f0_0000;
/// The initial stack pointer.
pub(super) const STACK_TOP: u32 = 0x0080_0000;
/// Where the linked image starts.
const IMAGE_BASE: u32 = 0x0000_1000;
/// Executed-instruction budget per run.
const STEP_BUDGET: u64 = 20_000_000;

// ===========================================================================
// Memory
// ===========================================================================

/// A sparse little-endian byte-addressed memory of 4 KiB pages.
#[derive(Clone, Default)]
pub(super) struct Memory {
    pages: HashMap<u32, Box<[u8; 4096]>>,
}

impl Memory {
    pub(super) fn read8(&self, a: u32) -> u8 {
        self.pages.get(&(a >> 12)).map_or(0, |p| p[(a & 0xfff) as usize])
    }
    pub(super) fn write8(&mut self, a: u32, v: u8) {
        self.pages.entry(a >> 12).or_insert_with(|| Box::new([0; 4096]))[(a & 0xfff) as usize] = v;
    }
    pub(super) fn read(&self, a: u32, size: u32) -> u32 {
        (0..size).fold(0, |acc, k| acc | u32::from(self.read8(a.wrapping_add(k))) << (8 * k))
    }
    pub(super) fn write(&mut self, a: u32, size: u32, v: u32) {
        for k in 0..size {
            self.write8(a.wrapping_add(k), (v >> (8 * k)) as u8);
        }
    }
    pub(super) fn write_bytes(&mut self, a: u32, bytes: &[u8]) {
        for (k, &b) in bytes.iter().enumerate() {
            self.write8(a.wrapping_add(k as u32), b);
        }
    }
}

// ===========================================================================
// The test harness's linker
// ===========================================================================

/// A linked image: memory with every section placed and relocated, the symbol
/// addresses (a Thumb function's with bit 0 set), and the helper trap
/// addresses by name.
pub(super) struct Image {
    pub(super) mem: Memory,
    pub(super) symbols: HashMap<String, u32>,
    pub(super) helpers: HashMap<u32, String>,
}

/// Lay `objects` out from [`IMAGE_BASE`] (sections in object order, each at
/// its alignment) and apply their relocations. An undefined symbol gets a
/// helper trap address.
pub(super) fn link(objects: &[&ObjectModule]) -> Result<Image, String> {
    let mut mem = Memory::default();
    let mut base: Vec<Vec<u32>> = Vec::new();
    let mut at = IMAGE_BASE;
    for obj in objects {
        let mut b = Vec::new();
        for s in obj.sections() {
            if s.kind == SectionKind::Debug {
                b.push(0);
                continue;
            }
            let align = s.align.max(4) as u32;
            at = at.div_ceil(align) * align;
            b.push(at);
            if s.kind != SectionKind::Bss {
                mem.write_bytes(at, &s.bytes);
            }
            at += s.size() as u32;
        }
        base.push(b);
    }
    let mut symbols: HashMap<String, u32> = HashMap::new();
    for (oi, obj) in objects.iter().enumerate() {
        for sym in obj.symbols() {
            if let SymbolValue::Defined { section, offset } = sym.value {
                let addr = base[oi][section.index()] + offset as u32;
                if !sym.name.starts_with('$') {
                    let strong = sym.binding != crate::mc::object::SymbolBinding::Weak;
                    if strong || !symbols.contains_key(&sym.name) {
                        symbols.insert(sym.name.clone(), addr);
                    }
                }
            }
        }
    }
    let mut helpers: HashMap<u32, String> = HashMap::new();
    let mut next = HELPER_BASE;
    for obj in objects {
        for sym in obj.symbols() {
            if sym.is_undefined() && !symbols.contains_key(&sym.name) {
                symbols.insert(sym.name.clone(), next);
                helpers.insert(next, sym.name.clone());
                next += 16;
            }
        }
    }
    for (oi, obj) in objects.iter().enumerate() {
        for r in obj.relocations() {
            let p = base[oi][r.section.index()] + r.offset as u32;
            let name = &obj.symbol(r.symbol).name;
            let s = *symbols.get(name).ok_or_else(|| format!("unresolved {name}"))?;
            let v = (i64::from(s) + r.addend) as u32;
            let mut field = [0u8; 4];
            for (k, f) in field.iter_mut().enumerate() {
                *f = mem.read8(p + k as u32);
            }
            let ok = match r.kind {
                RelocKind::Abs32 => {
                    field = v.to_le_bytes();
                    true
                }
                RelocKind::ThumbCall => {
                    let disp = i64::from((s & !1) as i32 as u32) + r.addend - i64::from(p);
                    write_thumb_field(&mut field, 0, r.kind, disp)
                }
                RelocKind::ThumbMovwAbsNc => write_thumb_field(&mut field, 0, r.kind, i64::from(v & 0xffff)),
                RelocKind::ThumbMovtAbs => write_thumb_field(&mut field, 0, r.kind, i64::from(v >> 16)),
                k => return Err(format!("relocation {k:?} in a Thumb image")),
            };
            if !ok {
                return Err(format!("relocation {:?} against {name} does not fit", r.kind));
            }
            mem.write_bytes(p, &field);
        }
    }
    Ok(Image { mem, symbols, helpers })
}

// ===========================================================================
// The Run-time ABI helpers
// ===========================================================================

fn f32a(r: &[u32; 4], k: usize) -> f32 {
    f32::from_bits(r[k])
}
fn f64a(r: &[u32; 4], k: usize) -> f64 {
    f64::from_bits(u64::from(r[k]) | u64::from(r[k + 1]) << 32)
}
fn ret64(r: &mut [u32; 4], v: u64) {
    r[0] = v as u32;
    r[1] = (v >> 32) as u32;
}
fn u64a(r: &[u32; 4], k: usize) -> u64 {
    u64::from(r[k]) | u64::from(r[k + 1]) << 32
}

/// An IEEE binary16 pattern to `f64` (exact).
pub(super) fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f64::from(h & 0x3ff);
    sign * match e {
        0 => m * 2f64.powi(-24),
        31 if m == 0.0 => f64::INFINITY,
        31 => return f64::NAN,
        _ => (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// `x` rounded (to nearest, ties to even) to an IEEE binary16 pattern.
pub(super) fn f64_to_f16(x: f64) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    if x.is_nan() {
        return 0x7e00;
    }
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a.is_infinite() {
        return sign | 0x7c00;
    }
    // In units of the smallest subnormal (2^-24), below 2^-14 the grid is
    // uniform; above, the exponent picks the grid spacing.
    let e = a.log2().floor() as i32;
    let e = if 2f64.powi(e) > a { e - 1 } else if 2f64.powi(e + 1) <= a { e + 1 } else { e };
    let spacing_exp = if e < -14 { -24 } else { e - 10 };
    let q = a / 2f64.powi(spacing_exp); // exact: a power-of-two scaling
    let mut n = q.floor();
    let frac = q - n;
    if frac > 0.5 || (frac == 0.5 && n % 2.0 == 1.0) {
        n += 1.0;
    }
    let v = n * 2f64.powi(spacing_exp);
    if v >= 65520.0 || v >= 2f64.powi(16) {
        return sign | 0x7c00;
    }
    if v < 2f64.powi(-14) {
        return sign | n as u16; // subnormal (or rounded up to the smallest normal: 0x400)
    }
    let e = v.log2().floor() as i32;
    let e = if 2f64.powi(e) > v { e - 1 } else { e };
    let m = (v / 2f64.powi(e) - 1.0) * 1024.0;
    sign | (((e + 15) as u16) << 10) | m as u16
}

/// Run the helper `name` on the argument registers `r` (results in `r0`..).
pub(super) fn aeabi(name: &str, r: &mut [u32; 4]) -> Result<(), String> {
    let f2 = |r: &mut [u32; 4], op: fn(f32, f32) -> f32| r[0] = op(f32a(r, 0), f32a(r, 1)).to_bits();
    let d2 = |r: &mut [u32; 4], op: fn(f64, f64) -> f64| {
        let v = op(f64a(r, 0), f64a(r, 2)).to_bits();
        ret64(r, v);
    };
    let fc = |r: &mut [u32; 4], op: fn(f32, f32) -> bool| r[0] = u32::from(op(f32a(r, 0), f32a(r, 1)));
    let dc = |r: &mut [u32; 4], op: fn(f64, f64) -> bool| r[0] = u32::from(op(f64a(r, 0), f64a(r, 2)));
    match name {
        "__aeabi_fadd" => f2(r, |a, b| a + b),
        "__aeabi_fsub" => f2(r, |a, b| a - b),
        "__aeabi_fmul" => f2(r, |a, b| a * b),
        "__aeabi_fdiv" => f2(r, |a, b| a / b),
        "fmodf" => f2(r, |a, b| a % b),
        "__aeabi_dadd" => d2(r, |a, b| a + b),
        "__aeabi_dsub" => d2(r, |a, b| a - b),
        "__aeabi_dmul" => d2(r, |a, b| a * b),
        "__aeabi_ddiv" => d2(r, |a, b| a / b),
        "fmod" => d2(r, |a, b| a % b),
        "__aeabi_fcmpeq" => fc(r, |a, b| a == b),
        "__aeabi_fcmplt" => fc(r, |a, b| a < b),
        "__aeabi_fcmple" => fc(r, |a, b| a <= b),
        "__aeabi_fcmpge" => fc(r, |a, b| a >= b),
        "__aeabi_fcmpgt" => fc(r, |a, b| a > b),
        "__aeabi_fcmpun" => fc(r, |a, b| a.is_nan() || b.is_nan()),
        "__aeabi_dcmpeq" => dc(r, |a, b| a == b),
        "__aeabi_dcmplt" => dc(r, |a, b| a < b),
        "__aeabi_dcmple" => dc(r, |a, b| a <= b),
        "__aeabi_dcmpge" => dc(r, |a, b| a >= b),
        "__aeabi_dcmpgt" => dc(r, |a, b| a > b),
        "__aeabi_dcmpun" => dc(r, |a, b| a.is_nan() || b.is_nan()),
        "__aeabi_f2d" => {
            let v = f64::from(f32a(r, 0)).to_bits();
            ret64(r, v);
        }
        "__aeabi_d2f" => r[0] = (f64a(r, 0) as f32).to_bits(),
        "__aeabi_f2iz" => r[0] = f32a(r, 0) as i32 as u32,
        "__aeabi_f2uiz" => r[0] = f32a(r, 0) as u32,
        "__aeabi_f2lz" => ret64(r, f32a(r, 0) as i64 as u64),
        "__aeabi_f2ulz" => ret64(r, f32a(r, 0) as u64),
        "__aeabi_d2iz" => r[0] = f64a(r, 0) as i32 as u32,
        "__aeabi_d2uiz" => r[0] = f64a(r, 0) as u32,
        "__aeabi_d2lz" => ret64(r, f64a(r, 0) as i64 as u64),
        "__aeabi_d2ulz" => ret64(r, f64a(r, 0) as u64),
        "__aeabi_i2f" => r[0] = (r[0] as i32 as f32).to_bits(),
        "__aeabi_ui2f" => r[0] = (r[0] as f32).to_bits(),
        "__aeabi_l2f" => r[0] = (u64a(r, 0) as i64 as f32).to_bits(),
        "__aeabi_ul2f" => r[0] = (u64a(r, 0) as f32).to_bits(),
        "__aeabi_i2d" => ret64(r, f64::from(r[0] as i32).to_bits()),
        "__aeabi_ui2d" => ret64(r, f64::from(r[0]).to_bits()),
        "__aeabi_l2d" => ret64(r, (u64a(r, 0) as i64 as f64).to_bits()),
        "__aeabi_ul2d" => ret64(r, (u64a(r, 0) as f64).to_bits()),
        "__aeabi_h2f" => r[0] = (f16_to_f64(r[0] as u16) as f32).to_bits(),
        "__aeabi_f2h" => r[0] = u32::from(f64_to_f16(f64::from(f32a(r, 0)))),
        "__aeabi_d2h" => r[0] = u32::from(f64_to_f16(f64a(r, 0))),
        "__aeabi_lmul" => ret64(r, u64a(r, 0).wrapping_mul(u64a(r, 2))),
        "__aeabi_ldivmod" | "__aeabi_uldivmod" => {
            let (a, b) = (u64a(r, 0), u64a(r, 2));
            if b == 0 {
                return Err(format!("{name}: division by zero"));
            }
            let (q, m) = if name == "__aeabi_ldivmod" {
                let (a, b) = (a as i64, b as i64);
                (a.wrapping_div(b) as u64, a.wrapping_rem(b) as u64)
            } else {
                (a / b, a % b)
            };
            *r = [q as u32, (q >> 32) as u32, m as u32, (m >> 32) as u32];
        }
        "__aeabi_idiv" | "__aeabi_idivmod" | "__aeabi_uidiv" | "__aeabi_uidivmod" => {
            if r[1] == 0 {
                return Err(format!("{name}: division by zero"));
            }
            let (q, m) = if name.starts_with("__aeabi_idiv") {
                let (a, b) = (r[0] as i32, r[1] as i32);
                (a.wrapping_div(b) as u32, a.wrapping_rem(b) as u32)
            } else {
                (r[0] / r[1], r[0] % r[1])
            };
            r[0] = q;
            r[1] = m;
        }
        other => return Err(format!("no runtime helper `{other}`")),
    }
    Ok(())
}

// ===========================================================================
// The CPU
// ===========================================================================

/// The architectural state of the simulated core.
pub(super) struct Cpu {
    pub(super) r: [u32; 16],
    n: bool,
    z: bool,
    c: bool,
    v: bool,
    /// `ITSTATE` (`IT[7:0]`): the base condition in `[7:5]`… as the ARM ARM
    /// defines it; zero outside an IT block.
    it: u8,
    pub(super) mem: Memory,
    helpers: HashMap<u32, String>,
    steps: u64,
}

/// `AddWithCarry(x, y, carry_in)`: the sum and the carry and overflow flags.
fn add_with_carry(x: u32, y: u32, carry: bool) -> (u32, bool, bool) {
    let u = u64::from(x) + u64::from(y) + u64::from(carry);
    let s = i64::from(x as i32) + i64::from(y as i32) + i64::from(carry);
    let r = u as u32;
    (r, u >> 32 != 0, i64::from(r as i32) != s)
}

/// `ThumbExpandImm_C(imm12, carry_in)`.
fn expand_imm(imm12: u32, carry: bool) -> Result<(u32, bool), String> {
    if imm12 >> 10 == 0 {
        let b = imm12 & 0xff;
        let v = match (imm12 >> 8) & 3 {
            0 => b,
            1 if b != 0 => b << 16 | b,
            2 if b != 0 => b << 24 | b << 8,
            3 if b != 0 => b.wrapping_mul(0x0101_0101),
            _ => return Err(format!("unpredictable modified immediate {imm12:#x}")),
        };
        Ok((v, carry))
    } else {
        let unrot = 0x80 | (imm12 & 0x7f);
        let v = unrot.rotate_right(imm12 >> 7);
        Ok((v, v >> 31 != 0))
    }
}

/// `Shift_C(value, type, amount, carry_in)` for an immediate shift (`type`
/// 0 LSL, 1 LSR, 2 ASR; LSR/ASR #0 encode 32).
fn shift_imm(v: u32, ty: u32, imm5: u32, carry: bool) -> (u32, bool) {
    match (ty, imm5) {
        (0, 0) => (v, carry),
        (0, n) => (v << n, (v >> (32 - n)) & 1 != 0),
        (1, 0) => (0, v >> 31 != 0),
        (1, n) => (v >> n, (v >> (n - 1)) & 1 != 0),
        (2, 0) => (((v as i32) >> 31) as u32, v >> 31 != 0),
        (2, n) => (((v as i32) >> n) as u32, (v >> (n - 1)) & 1 != 0),
        _ => (v.rotate_right(imm5), v.rotate_right(imm5) >> 31 != 0),
    }
}

/// A shift by a register (its bottom byte), with carry out.
fn shift_reg(v: u32, ty: u32, amount: u32, carry: bool) -> (u32, bool) {
    let n = amount & 0xff;
    if n == 0 {
        return (v, carry);
    }
    match ty {
        0 => {
            if n < 32 {
                (v << n, (v >> (32 - n)) & 1 != 0)
            } else {
                (0, n == 32 && v & 1 != 0)
            }
        }
        1 => {
            if n < 32 {
                (v >> n, (v >> (n - 1)) & 1 != 0)
            } else {
                (0, n == 32 && v >> 31 != 0)
            }
        }
        _ => {
            if n < 32 {
                (((v as i32) >> n) as u32, (v >> (n - 1)) & 1 != 0)
            } else {
                (((v as i32) >> 31) as u32, v >> 31 != 0)
            }
        }
    }
}

impl Cpu {
    pub(super) fn new(mem: Memory, helpers: HashMap<u32, String>) -> Cpu {
        let mut r = [0u32; 16];
        r[13] = STACK_TOP;
        Cpu { r, n: false, z: false, c: false, v: false, it: 0, mem, helpers, steps: 0 }
    }

    fn cond(&self, c: u32) -> bool {
        match c {
            0 => self.z,
            1 => !self.z,
            2 => self.c,
            3 => !self.c,
            4 => self.n,
            5 => !self.n,
            6 => self.v,
            7 => !self.v,
            8 => self.c && !self.z,
            9 => !self.c || self.z,
            10 => self.n == self.v,
            11 => self.n != self.v,
            12 => !self.z && self.n == self.v,
            13 => self.z || self.n != self.v,
            _ => true,
        }
    }

    fn nz(&mut self, r: u32) {
        self.n = r >> 31 != 0;
        self.z = r == 0;
    }

    fn in_it(&self) -> bool {
        self.it & 0xf != 0
    }

    fn advance_it(&mut self) {
        if self.it & 0x7 == 0 {
            self.it = 0;
        } else {
            self.it = (self.it & 0xe0) | ((self.it << 1) & 0x1f);
        }
    }

    /// Call the Thumb function at `addr` (Thumb bit set) with word arguments
    /// in `r0`–`r3`, returning `r0`–`r3` when it returns.
    pub(super) fn call(&mut self, addr: u32, args: &[u32]) -> Result<[u32; 4], String> {
        assert!(args.len() <= 4, "stack arguments are not set up by the harness");
        for (k, &a) in args.iter().enumerate() {
            self.r[k] = a;
        }
        self.r[14] = EXIT | 1;
        self.branch_exchange(addr)?;
        self.run()?;
        Ok([self.r[0], self.r[1], self.r[2], self.r[3]])
    }

    /// Run from the current `pc` until the top-level return (or, when `halt`
    /// allowed, a branch to itself), with the step budget.
    pub(super) fn run(&mut self) -> Result<(), String> {
        loop {
            let pc = self.r[15];
            if pc == EXIT {
                return Ok(());
            }
            if let Some(name) = self.helpers.get(&pc).cloned() {
                let mut regs = [self.r[0], self.r[1], self.r[2], self.r[3]];
                aeabi(&name, &mut regs)?;
                self.r[..4].copy_from_slice(&regs);
                self.r[12] = 0xdead_beef; // helpers may clobber ip
                let lr = self.r[14];
                self.branch_exchange(lr)?;
                continue;
            }
            self.steps += 1;
            if self.steps > STEP_BUDGET {
                return Err("step budget exhausted".into());
            }
            if self.step()? {
                return Ok(());
            }
        }
    }

    /// `BXWritePC`: an M-profile core only runs Thumb code.
    fn branch_exchange(&mut self, target: u32) -> Result<(), String> {
        if target & 1 == 0 && target != EXIT {
            return Err(format!("interworking branch to Arm state at {target:#x}"));
        }
        self.r[15] = target & !1;
        Ok(())
    }

    fn fetch(&self, a: u32) -> u16 {
        self.mem.read(a, 2) as u16
    }

    /// Execute one instruction; `true` when the core halts on a branch to
    /// itself (the firmware's idle loop).
    fn step(&mut self) -> Result<bool, String> {
        let pc = self.r[15];
        let hw1 = u32::from(self.fetch(pc));
        let wide = matches!(hw1 >> 11, 0b11101..=0b11111);
        let (len, hw2) = if wide { (4, u32::from(self.fetch(pc + 2))) } else { (2, 0) };
        let in_it = self.in_it();
        let exec = if in_it { self.cond(u32::from(self.it >> 4)) } else { true };
        // IT itself (outside a block) is handled here: it never executes
        // conditionally.
        let next = pc + len;
        self.r[15] = next;
        if !exec {
            self.advance_it();
            return Ok(false);
        }
        let halted = if wide { self.exec32(pc, hw1, hw2)? } else { self.exec16(pc, hw1)? };
        if in_it {
            self.advance_it();
        }
        Ok(halted)
    }

    /// The value of a register operand as an instruction reads it (`pc`
    /// reads as the instruction's address + 4).
    fn rd(&self, n: u32, pc: u32) -> u32 {
        if n == 15 { pc + 4 } else { self.r[n as usize] }
    }

    fn set(&mut self, d: u32, v: u32) -> Result<(), String> {
        if d == 15 {
            return Err("write to pc from a data-processing instruction".into());
        }
        self.r[d as usize] = v;
        Ok(())
    }

    fn b_to(&mut self, pc: u32, off: i32) -> bool {
        let target = (pc as i64 + 4 + i64::from(off)) as u32;
        self.r[15] = target;
        target == pc
    }

    fn exec16(&mut self, pc: u32, h: u32) -> Result<bool, String> {
        let setflags = !self.in_it();
        let (d3, n3, m3) = (h & 7, (h >> 3) & 7, (h >> 6) & 7);
        match h >> 11 {
            0b00000..=0b00010 => {
                // LSL/LSR/ASR (immediate).
                let ty = (h >> 11) & 3;
                let (r, c) = shift_imm(self.r[n3 as usize], ty, (h >> 6) & 0x1f, self.c);
                self.set(d3, r)?;
                if setflags {
                    self.nz(r);
                    self.c = c;
                }
            }
            0b00011 => {
                // ADD/SUB register or 3-bit immediate.
                let sub = h & 0x200 != 0;
                let y = if h & 0x400 != 0 { m3 } else { self.r[m3 as usize] };
                let x = self.r[n3 as usize];
                let (r, c, v) = if sub { add_with_carry(x, !y, true) } else { add_with_carry(x, y, false) };
                self.set(d3, r)?;
                if setflags {
                    self.nz(r);
                    self.c = c;
                    self.v = v;
                }
            }
            0b00100..=0b00111 => {
                let dn = (h >> 8) & 7;
                let imm = h & 0xff;
                let x = self.r[dn as usize];
                match (h >> 11) & 3 {
                    0 => {
                        self.set(dn, imm)?;
                        if setflags {
                            self.nz(imm);
                        }
                    }
                    1 => {
                        let (r, c, v) = add_with_carry(x, !imm, true);
                        self.nz(r);
                        self.c = c;
                        self.v = v;
                    }
                    op => {
                        let (r, c, v) = if op == 2 { add_with_carry(x, imm, false) } else { add_with_carry(x, !imm, true) };
                        self.set(dn, r)?;
                        if setflags {
                            self.nz(r);
                            self.c = c;
                            self.v = v;
                        }
                    }
                }
            }
            0b01000 if h & 0x0400 == 0 => {
                // Data processing (register).
                let op = (h >> 6) & 0xf;
                let (dn, m) = (d3, n3);
                let (x, y) = (self.r[dn as usize], self.r[m as usize]);
                match op {
                    0 | 1 | 12 | 14 | 15 | 13 => {
                        let r = match op {
                            0 => x & y,
                            1 => x ^ y,
                            12 => x | y,
                            14 => x & !y,
                            15 => !y,
                            _ => x.wrapping_mul(y),
                        };
                        self.set(dn, r)?;
                        if setflags {
                            self.nz(r);
                        }
                    }
                    2..=4 => {
                        let (r, c) = shift_reg(x, op - 2, y, self.c);
                        self.set(dn, r)?;
                        if setflags {
                            self.nz(r);
                            self.c = c;
                        }
                    }
                    8 => self.nz(x & y),
                    9 => {
                        let (r, c, v) = add_with_carry(!y, 0, true);
                        self.set(dn, r)?;
                        if setflags {
                            self.nz(r);
                            self.c = c;
                            self.v = v;
                        }
                    }
                    10 | 11 => {
                        let (r, c, v) = if op == 10 { add_with_carry(x, !y, true) } else { add_with_carry(x, y, false) };
                        self.nz(r);
                        self.c = c;
                        self.v = v;
                    }
                    _ => return Err(format!("unsupported 16-bit data-processing op {op} at {pc:#x}")),
                }
            }
            0b01000 => {
                // Special data processing and branch/exchange.
                let m = (h >> 3) & 0xf;
                let dn = (h & 7) | ((h >> 4) & 8);
                match (h >> 8) & 3 {
                    0 => {
                        let r = self.rd(dn, pc).wrapping_add(self.rd(m, pc));
                        self.set(dn, r)?;
                    }
                    1 => {
                        let (r, c, v) = add_with_carry(self.rd(dn, pc), !self.rd(m, pc), true);
                        self.nz(r);
                        self.c = c;
                        self.v = v;
                    }
                    2 => {
                        let v = self.rd(m, pc);
                        self.set(dn, v)?;
                    }
                    _ => {
                        let target = self.r[m as usize];
                        if h & 0x80 != 0 {
                            self.r[14] = (pc + 2) | 1;
                        }
                        self.branch_exchange(target)?;
                    }
                }
            }
            0b01100..=0b10001 => {
                // Load/store (immediate): word, byte, halfword.
                let imm5 = (h >> 6) & 0x1f;
                let (size, off) = match h >> 12 {
                    0b0110 => (4, imm5 * 4),
                    0b0111 => (1, imm5),
                    _ => (2, imm5 * 2),
                };
                let addr = self.r[n3 as usize].wrapping_add(off);
                if h & 0x0800 != 0 {
                    let v = self.mem.read(addr, size);
                    self.set(d3, v)?;
                } else {
                    let v = self.r[d3 as usize];
                    self.mem.write(addr, size, v);
                }
            }
            0b10010 | 0b10011 => {
                let t = (h >> 8) & 7;
                let addr = self.r[13].wrapping_add((h & 0xff) * 4);
                if h & 0x0800 != 0 {
                    let v = self.mem.read(addr, 4);
                    self.set(t, v)?;
                } else {
                    let v = self.r[t as usize];
                    self.mem.write(addr, 4, v);
                }
            }
            0b10101 => {
                let d = (h >> 8) & 7;
                let v = self.r[13].wrapping_add((h & 0xff) * 4);
                self.set(d, v)?;
            }
            0b10110 | 0b10111 => self.misc16(pc, h)?,
            0b11010 | 0b11011 => {
                let cond = (h >> 8) & 0xf;
                match cond {
                    0xe => return Err(format!("udf #{} at {pc:#x}", h & 0xff)),
                    0xf => return Err(format!("svc #{} at {pc:#x}", h & 0xff)),
                    _ => {
                        if self.in_it() {
                            return Err("conditional branch inside an IT block".into());
                        }
                        if self.cond(cond) {
                            let off = ((h & 0xff) as i8 as i32) * 2;
                            return Ok(self.b_to(pc, off));
                        }
                    }
                }
            }
            0b11100 => {
                let off = (((h & 0x7ff) << 21) as i32) >> 20;
                return Ok(self.b_to(pc, off));
            }
            _ => return Err(format!("unsupported 16-bit instruction {h:#06x} at {pc:#x}")),
        }
        Ok(false)
    }

    fn misc16(&mut self, pc: u32, h: u32) -> Result<(), String> {
        let (d3, m3) = (h & 7, (h >> 3) & 7);
        match (h >> 8) & 0xf {
            0b0000 => {
                let off = (h & 0x7f) * 4;
                self.r[13] = if h & 0x80 != 0 { self.r[13].wrapping_sub(off) } else { self.r[13].wrapping_add(off) };
            }
            0b0010 => {
                let m = self.r[m3 as usize];
                let v = match (h >> 6) & 3 {
                    0 => m as i16 as i32 as u32,
                    1 => m as i8 as i32 as u32,
                    2 => m & 0xffff,
                    _ => m & 0xff,
                };
                self.set(d3, v)?;
            }
            0b0100 | 0b0101 => {
                let list = (h & 0xff) | if h & 0x100 != 0 { 1 << 14 } else { 0 };
                self.push(list);
            }
            0b1100 | 0b1101 => {
                let list = (h & 0xff) | if h & 0x100 != 0 { 1 << 15 } else { 0 };
                self.pop(list)?;
            }
            0b1111 => {
                if h & 0xf != 0 {
                    if self.in_it() {
                        return Err("IT inside an IT block".into());
                    }
                    self.it = (h & 0xff) as u8;
                    // The IT instruction itself does not advance the state.
                } // else a hint (nop)
            }
            _ => return Err(format!("unsupported 16-bit misc instruction {h:#06x} at {pc:#x}")),
        }
        Ok(())
    }

    fn push(&mut self, list: u32) {
        let n = list.count_ones();
        let mut addr = self.r[13].wrapping_sub(4 * n);
        self.r[13] = addr;
        for k in 0..16 {
            if list & (1 << k) != 0 {
                let v = self.r[k];
                self.mem.write(addr, 4, v);
                addr += 4;
            }
        }
    }

    fn pop(&mut self, list: u32) -> Result<(), String> {
        let mut addr = self.r[13];
        let mut new_pc = None;
        for k in 0..16 {
            if list & (1 << k) != 0 {
                let v = self.mem.read(addr, 4);
                if k == 15 {
                    new_pc = Some(v);
                } else {
                    self.r[k] = v;
                }
                addr += 4;
            }
        }
        self.r[13] = addr;
        if let Some(t) = new_pc {
            self.branch_exchange(t)?;
        }
        Ok(())
    }

    fn exec32(&mut self, pc: u32, h1: u32, h2: u32) -> Result<bool, String> {
        let n = h1 & 0xf;
        if h1 == 0xe92d && h2 & 0xa000 == 0 {
            self.push(h2);
            return Ok(false);
        }
        if h1 == 0xe8bd && h2 & 0x2000 == 0 {
            self.pop(h2)?;
            return Ok(false);
        }
        if h1 & 0xffe0 == 0xe9c0 {
            // LDRD/STRD (immediate, offset).
            let (t, t2) = (h2 >> 12, (h2 >> 8) & 0xf);
            let addr = self.rd(n, pc).wrapping_add((h2 & 0xff) * 4);
            if h1 & 0x10 != 0 {
                let (a, b) = (self.mem.read(addr, 4), self.mem.read(addr + 4, 4));
                self.set(t, a)?;
                self.set(t2, b)?;
            } else {
                let (a, b) = (self.r[t as usize], self.r[t2 as usize]);
                self.mem.write(addr, 4, a);
                self.mem.write(addr + 4, 4, b);
            }
            return Ok(false);
        }
        if h1 & 0xfe00 == 0xea00 {
            return self.dp_shifted(pc, h1, h2).map(|()| false);
        }
        if h1 & 0xf800 == 0xf000 && h2 & 0x8000 == 0 {
            return if h1 & 0x0200 == 0 { self.dp_modimm(pc, h1, h2) } else { self.dp_plain(pc, h1, h2) }.map(|()| false);
        }
        if h1 & 0xf800 == 0xf000 && h2 & 0x8000 != 0 {
            let s = (h1 >> 10) & 1;
            match h2 & 0xd000 {
                0x8000 => {
                    let cond = (h1 >> 6) & 0xf;
                    if cond >= 0xe {
                        if h1 == 0xf3bf && h2 & 0xfff0 == 0x8f50 {
                            return Ok(false); // dmb/dsb/isb: no effect here
                        }
                        return Err(format!("unsupported misc control {h1:#06x} {h2:#06x}"));
                    }
                    let (j1, j2) = ((h2 >> 13) & 1, (h2 >> 11) & 1);
                    let raw = s << 20 | j2 << 19 | j1 << 18 | (h1 & 0x3f) << 12 | (h2 & 0x7ff) << 1;
                    let off = ((raw << 11) as i32) >> 11;
                    if self.in_it() {
                        return Err("conditional branch inside an IT block".into());
                    }
                    if self.cond(cond) {
                        return Ok(self.b_to(pc, off));
                    }
                    return Ok(false);
                }
                0x9000 | 0xd000 => {
                    let (j1, j2) = ((h2 >> 13) & 1, (h2 >> 11) & 1);
                    let (i1, i2) = ((j1 ^ s) ^ 1, (j2 ^ s) ^ 1);
                    let raw = s << 24 | i1 << 23 | i2 << 22 | (h1 & 0x3ff) << 12 | (h2 & 0x7ff) << 1;
                    let off = ((raw << 7) as i32) >> 7;
                    if h2 & 0x4000 != 0 {
                        self.r[14] = (pc + 4) | 1;
                    }
                    return Ok(self.b_to(pc, off));
                }
                _ => return Err(format!("unsupported branch {h1:#06x} {h2:#06x}")),
            }
        }
        if h1 & 0xfe00 == 0xf800 {
            return self.ldst32(pc, h1, h2).map(|()| false);
        }
        if h1 & 0xff80 == 0xfa00 && h2 & 0xf0f0 == 0xf000 {
            // LSL/LSR/ASR (register).
            let ty = (h1 >> 5) & 3;
            let (d, m) = ((h2 >> 8) & 0xf, h2 & 0xf);
            let (r, c) = shift_reg(self.r[n as usize], ty, self.r[m as usize], self.c);
            self.set(d, r)?;
            if h1 & 0x10 != 0 {
                self.nz(r);
                self.c = c;
            }
            return Ok(false);
        }
        if matches!(h1, 0xfa0f | 0xfa1f | 0xfa4f | 0xfa5f) && h2 & 0xf0f0 == 0xf080 {
            let (d, m) = ((h2 >> 8) & 0xf, h2 & 0xf);
            let x = self.r[m as usize];
            let v = match h1 {
                0xfa0f => x as i16 as i32 as u32,
                0xfa1f => x & 0xffff,
                0xfa4f => x as i8 as i32 as u32,
                _ => x & 0xff,
            };
            self.set(d, v)?;
            return Ok(false);
        }
        if h1 & 0xfff0 == 0xfb00 {
            let (a, d, m) = (h2 >> 12, (h2 >> 8) & 0xf, h2 & 0xf);
            let p = self.r[n as usize].wrapping_mul(self.r[m as usize]);
            let v = match (h2 >> 4) & 0xf {
                0 if a == 15 => p,
                0 => self.r[a as usize].wrapping_add(p),
                1 => self.r[a as usize].wrapping_sub(p),
                _ => return Err(format!("unsupported multiply {h1:#06x} {h2:#06x}")),
            };
            self.set(d, v)?;
            return Ok(false);
        }
        if (h1 & 0xfff0 == 0xfb90 || h1 & 0xfff0 == 0xfbb0) && h2 & 0xf0f0 == 0xf0f0 {
            let (d, m) = ((h2 >> 8) & 0xf, h2 & 0xf);
            let (x, y) = (self.r[n as usize], self.r[m as usize]);
            // An ARMv7-M divide by zero yields 0 (DIV_0_TRP clear).
            let v = if y == 0 {
                0
            } else if h1 & 0x20 == 0 {
                (x as i32).wrapping_div(y as i32) as u32
            } else {
                x / y
            };
            self.set(d, v)?;
            return Ok(false);
        }
        Err(format!("unsupported 32-bit instruction {h1:#06x} {h2:#06x} at {pc:#x}"))
    }

    fn dp_shifted(&mut self, pc: u32, h1: u32, h2: u32) -> Result<(), String> {
        let op = (h1 >> 5) & 0xf;
        let s = h1 & 0x10 != 0;
        let n = h1 & 0xf;
        let (d, m) = ((h2 >> 8) & 0xf, h2 & 0xf);
        let imm5 = ((h2 >> 12) & 7) << 2 | (h2 >> 6) & 3;
        let ty = (h2 >> 4) & 3;
        let (y, sc) = shift_imm(self.rd(m, pc), ty, imm5, self.c);
        let x = if n == 15 { 0 } else { self.rd(n, pc) };
        self.alu(op, s, d, n, x, y, sc)
    }

    fn dp_modimm(&mut self, pc: u32, h1: u32, h2: u32) -> Result<(), String> {
        let op = (h1 >> 5) & 0xf;
        let s = h1 & 0x10 != 0;
        let n = h1 & 0xf;
        let d = (h2 >> 8) & 0xf;
        let imm12 = ((h1 >> 10) & 1) << 11 | ((h2 >> 12) & 7) << 8 | (h2 & 0xff);
        let (y, sc) = expand_imm(imm12, self.c)?;
        let x = if n == 15 { 0 } else { self.rd(n, pc) };
        self.alu(op, s, d, n, x, y, sc)
    }

    /// The shared data-processing operation of the register and
    /// modified-immediate forms (`n` = 15 selects `mov`/`mvn`, `d` = 15 with
    /// `s` a flag-only compare/test).
    #[allow(clippy::too_many_arguments)]
    fn alu(&mut self, op: u32, s: bool, d: u32, n: u32, x: u32, y: u32, sc: bool) -> Result<(), String> {
        let (r, c, v) = match op {
            0 => (x & y, sc, self.v),
            1 => (x & !y, sc, self.v),
            2 => (if n == 15 { y } else { x | y }, sc, self.v),
            3 => (if n == 15 { !y } else { x | !y }, sc, self.v),
            4 => (x ^ y, sc, self.v),
            8 => add_with_carry(x, y, false),
            13 => add_with_carry(x, !y, true),
            14 => add_with_carry(!x, y, true),
            _ => return Err(format!("unsupported data-processing op {op}")),
        };
        if !(s && d == 15) {
            self.set(d, r)?;
        }
        if s {
            self.nz(r);
            self.c = c;
            self.v = v;
        }
        Ok(())
    }

    fn dp_plain(&mut self, pc: u32, h1: u32, h2: u32) -> Result<(), String> {
        let op = (h1 >> 4) & 0x1f;
        let n = h1 & 0xf;
        let d = (h2 >> 8) & 0xf;
        let imm12 = ((h1 >> 10) & 1) << 11 | ((h2 >> 12) & 7) << 8 | (h2 & 0xff);
        match op {
            0 => {
                let v = self.rd(n, pc).wrapping_add(imm12);
                self.set(d, v)
            }
            10 => {
                let v = self.rd(n, pc).wrapping_sub(imm12);
                self.set(d, v)
            }
            4 => self.set(d, n << 12 | imm12),
            12 => {
                let v = (self.r[d as usize] & 0xffff) | (n << 12 | imm12) << 16;
                self.set(d, v)
            }
            20 | 28 => {
                let lsb = ((h2 >> 12) & 7) << 2 | (h2 >> 6) & 3;
                let width = (h2 & 0x1f) + 1;
                let x = self.r[n as usize] >> lsb;
                let v = if width == 32 {
                    x
                } else if op == 20 {
                    (((x << (32 - width)) as i32) >> (32 - width)) as u32
                } else {
                    x & ((1 << width) - 1)
                };
                self.set(d, v)
            }
            _ => Err(format!("unsupported plain-immediate op {op} at {pc:#x}")),
        }
    }

    fn ldst32(&mut self, pc: u32, h1: u32, h2: u32) -> Result<(), String> {
        let n = h1 & 0xf;
        let t = h2 >> 12;
        let size = match (h1 >> 5) & 3 {
            0 => 1,
            1 => 2,
            2 => 4,
            _ => return Err(format!("unsupported load/store {h1:#06x}")),
        };
        if h1 & 0x0100 != 0 {
            return Err("sign-extending loads are not emitted".into());
        }
        let load = h1 & 0x10 != 0;
        let base = self.rd(n, pc);
        let addr = if h1 & 0x80 != 0 {
            base.wrapping_add(h2 & 0xfff)
        } else {
            // T4: [n, #+/-imm8] with P/U/W; only the offset form is emitted.
            if h2 & 0x0f00 != 0x0c00 && h2 & 0x0f00 != 0x0e00 {
                return Err(format!("unsupported indexed load/store {h1:#06x} {h2:#06x}"));
            }
            if h2 & 0x200 != 0 { base.wrapping_add(h2 & 0xff) } else { base.wrapping_sub(h2 & 0xff) }
        };
        if load {
            let v = self.mem.read(addr, size);
            self.set(t, v)
        } else {
            let v = self.r[t as usize];
            self.mem.write(addr, size, v);
            Ok(())
        }
    }
}

/// Run a code snippet built by `f` (followed by `bx lr`) and return the value
/// of register `reg` afterwards.
pub(super) fn run_snippet(f: impl FnOnce(&mut super::encode::Asm), reg: u32) -> Result<u32, String> {
    let mut a = super::encode::Asm::default();
    f(&mut a);
    a.i(super::encode::bx(14));
    let code = a.finish();
    let mut mem = Memory::default();
    mem.write_bytes(IMAGE_BASE, &code.bytes);
    let mut cpu = Cpu::new(mem, HashMap::new());
    cpu.call(IMAGE_BASE | 1, &[])?;
    Ok(cpu.r[reg as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_conversions_round_to_nearest_even() {
        for (x, h) in [
            (1.0, 0x3c00u16),
            (-2.0, 0xc000),
            (65504.0, 0x7bff),
            (65520.0, 0x7c00),
            (5.960_464_477_539_063e-8, 0x0001),
            (6.103_515_625e-5, 0x0400),
            (0.1, 0x2e66),
            (1.0 + 1.0 / 2048.0, 0x3c00), // a tie: to even
            (1.0 + 3.0 / 2048.0, 0x3c02),
        ] {
            assert_eq!(f64_to_f16(x), h, "{x}");
        }
        for h in [0u16, 1, 0x3ff, 0x400, 0x3c00, 0x7bff, 0x8001, 0xfbff, 0x3555] {
            assert_eq!(f64_to_f16(f16_to_f64(h)), h, "{h:#x}");
        }
    }
}
