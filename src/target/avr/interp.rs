//! An instruction-level AVR interpreter over an encoded flash image, for the
//! tests (the host has no AVR simulator).
//!
//! It models what the generated code relies on, from the AVR Instruction Set
//! Manual: the 32 registers mapped at data addresses `0x00..0x20`, the I/O
//! space at `0x20..0x60` (`SPL` `0x5d`, `SPH` `0x5e`, `SREG` `0x5f`), SRAM up
//! to `RAMEND`, flash as a separate byte-addressed memory read by `lpm`, the
//! stack (`push` stores then decrements; `call` pushes the return address
//! low byte first), and every `SREG` flag the executed instructions define.
//! An instruction it does not know, an access outside memory, a `break`, or
//! running out of steps is an error.

use std::fmt;

use super::Device;

/// Why execution stopped abnormally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Fault(pub String);

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

const SREG: usize = 0x5f;
const SPL: usize = 0x5d;
const SPH: usize = 0x5e;
const C: u8 = 1 << 0;
const Z: u8 = 1 << 1;
const N: u8 = 1 << 2;
const V: u8 = 1 << 3;
const S: u8 = 1 << 4;
const H: u8 = 1 << 5;
const I: u8 = 1 << 7;

/// The machine state.
#[derive(Clone)]
pub(crate) struct Avr {
    /// Flash, byte-addressed.
    pub flash: Vec<u8>,
    /// The data space: registers, I/O, SRAM.
    pub data: Vec<u8>,
    /// The program counter, in words.
    pub pc: u32,
    /// Instructions executed.
    pub steps: u64,
    /// The lowest stack pointer seen (the deepest stack use).
    pub min_sp: u16,
    ram_end: u16,
}

impl fmt::Debug for Avr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Avr {{ pc: {:#x}, sp: {:#x}, regs: {:02x?} }}", self.pc * 2, self.sp(), &self.data[..32])
    }
}

impl Avr {
    /// A machine with `flash` loaded, in its reset state.
    pub(crate) fn new(flash: &[u8], device: &Device) -> Avr {
        let mut f = flash.to_vec();
        f.resize(0x1_0000, 0xff);
        let mut data = vec![0u8; usize::from(device.ram_end) + 1];
        // Uninitialized SRAM is not zero on real hardware: make reads of it
        // visible in tests.
        for (i, b) in data.iter_mut().enumerate().skip(usize::from(device.ram_start)) {
            *b = (i as u8).wrapping_mul(37) ^ 0x5a;
        }
        let mut m = Avr { flash: f, data, pc: 0, steps: 0, min_sp: device.ram_end, ram_end: device.ram_end };
        m.set_sp(device.ram_end);
        m
    }

    pub(crate) fn reg(&self, r: usize) -> u8 {
        self.data[r]
    }

    pub(crate) fn set_reg(&mut self, r: usize, v: u8) {
        self.data[r] = v;
    }

    pub(crate) fn sp(&self) -> u16 {
        u16::from(self.data[SPL]) | (u16::from(self.data[SPH]) << 8)
    }

    pub(crate) fn set_sp(&mut self, sp: u16) {
        self.data[SPL] = sp as u8;
        self.data[SPH] = (sp >> 8) as u8;
    }

    fn sreg(&self) -> u8 {
        self.data[SREG]
    }

    fn flag(&self, f: u8) -> bool {
        self.sreg() & f != 0
    }

    /// Set the flags in `mask` to those of `val`.
    fn set_flags(&mut self, mask: u8, val: u8) {
        self.data[SREG] = (self.data[SREG] & !mask) | (val & mask);
    }

    fn rd(&self, a: u32) -> Result<u8, Fault> {
        self.data.get(a as usize).copied().ok_or_else(|| Fault(format!("read of data address {a:#x} at pc {:#x}", self.pc * 2)))
    }

    fn wr(&mut self, a: u32, v: u8) -> Result<(), Fault> {
        let pc = self.pc;
        let slot = self.data.get_mut(a as usize).ok_or_else(|| Fault(format!("write of data address {a:#x} at pc {:#x}", pc * 2)))?;
        *slot = v;
        Ok(())
    }

    fn word(&self, w: u32) -> u16 {
        let a = (2 * w) as usize & 0xffff;
        u16::from_le_bytes([self.flash[a], self.flash[a + 1]])
    }

    fn pair(&self, r: usize) -> u16 {
        u16::from(self.data[r]) | (u16::from(self.data[r + 1]) << 8)
    }

    fn set_pair(&mut self, r: usize, v: u16) {
        self.data[r] = v as u8;
        self.data[r + 1] = (v >> 8) as u8;
    }

    pub(crate) fn push(&mut self, v: u8) -> Result<(), Fault> {
        let sp = self.sp();
        self.wr(u32::from(sp), v)?;
        let sp = sp.wrapping_sub(1);
        self.set_sp(sp);
        self.min_sp = self.min_sp.min(sp);
        if sp < 0x100 {
            return Err(Fault(format!("stack overflow (SP {sp:#x})")));
        }
        Ok(())
    }

    pub(crate) fn pop(&mut self) -> Result<u8, Fault> {
        let sp = self.sp().wrapping_add(1);
        if sp > self.ram_end {
            return Err(Fault("stack underflow".into()));
        }
        self.set_sp(sp);
        self.rd(u32::from(sp))
    }

    /// Push a return address (a word address) as `call` does.
    pub(crate) fn push_ret(&mut self, ret: u32) -> Result<(), Fault> {
        self.push(ret as u8)?;
        self.push((ret >> 8) as u8)
    }

    fn nzs(&mut self, r: u8, mask_extra: u8, extra: u8) {
        let n = r & 0x80 != 0;
        let v = extra & V != 0;
        let mut f = extra;
        if n {
            f |= N;
        }
        if n ^ v {
            f |= S;
        }
        if r == 0 {
            f |= Z;
        }
        self.set_flags(N | Z | S | mask_extra, f);
    }

    /// `d + r + c`, setting H S V N Z C.
    fn add8(&mut self, d: u8, r: u8, c: bool) -> u8 {
        let c = u16::from(c);
        let sum = u16::from(d) + u16::from(r) + c;
        let res = sum as u8;
        let mut f = 0;
        if sum > 0xff {
            f |= C;
        }
        if u16::from(d & 0xf) + u16::from(r & 0xf) + c > 0xf {
            f |= H;
        }
        if (!(d ^ r) & (d ^ res)) & 0x80 != 0 {
            f |= V;
        }
        self.nzs(res, C | H | V, f);
        res
    }

    /// `d - r - c`, setting H S V N Z C; `keep_z` keeps Z only if already set
    /// (`sbc`, `sbci`, `cpc`).
    fn sub8(&mut self, d: u8, r: u8, c: bool, keep_z: bool) -> u8 {
        let old_z = self.flag(Z);
        let c8 = u8::from(c);
        let res = d.wrapping_sub(r).wrapping_sub(c8);
        let mut f = 0;
        if u16::from(r) + u16::from(c8) > u16::from(d) {
            f |= C;
        }
        if (r & 0xf) + c8 > (d & 0xf) {
            f |= H;
        }
        if ((d ^ r) & (d ^ res)) & 0x80 != 0 {
            f |= V;
        }
        self.nzs(res, C | H | V, f);
        if keep_z && !(res == 0 && old_z) {
            self.set_flags(Z, 0);
        }
        res
    }

    fn logic(&mut self, res: u8) -> u8 {
        self.nzs(res, V, 0);
        res
    }

    /// Execute one instruction.
    pub(crate) fn step(&mut self) -> Result<(), Fault> {
        self.steps += 1;
        let pc = self.pc;
        let op = self.word(pc);
        let d5 = usize::from((op >> 4) & 0x1f);
        let r5 = usize::from(((op >> 5) & 0x10) | (op & 0xf));
        let d4 = 16 + usize::from((op >> 4) & 0xf);
        let k8 = (((op >> 4) & 0xf0) | (op & 0xf)) as u8;
        let mut next = pc + 1;
        let bad = || Fault(format!("unknown instruction {op:#06x} at {:#x}", pc * 2));
        match op >> 12 {
            0x0..=0x2 => {
                let (d, r) = (self.data[d5], self.data[r5]);
                match op >> 10 {
                    0b000000 => {
                        if op == 0 {
                        } else if op & 0xff00 == 0x0100 {
                            let (dd, rr) = (usize::from((op >> 4) & 0xf) * 2, usize::from(op & 0xf) * 2);
                            let v = self.pair(rr);
                            self.set_pair(dd, v);
                        } else {
                            return Err(bad());
                        }
                    }
                    0b000001 => {
                        self.sub8(d, r, self.flag(C), true);
                    }
                    0b000010 => self.data[d5] = self.sub8(d, r, self.flag(C), true),
                    0b000011 => self.data[d5] = self.add8(d, r, false),
                    0b000100 => {
                        // cpse
                        if d == r {
                            next += self.len(next);
                        }
                    }
                    0b000101 => {
                        self.sub8(d, r, false, false);
                    }
                    0b000110 => self.data[d5] = self.sub8(d, r, false, false),
                    0b000111 => self.data[d5] = self.add8(d, r, self.flag(C)),
                    0b001000 => self.data[d5] = self.logic(d & r),
                    0b001001 => self.data[d5] = self.logic(d ^ r),
                    0b001010 => self.data[d5] = self.logic(d | r),
                    0b001011 => self.data[d5] = r,
                    _ => return Err(bad()),
                }
            }
            0x3 => {
                let d = self.data[d4];
                self.sub8(d, k8, false, false);
            }
            0x4 => {
                let d = self.data[d4];
                self.data[d4] = self.sub8(d, k8, self.flag(C), true);
            }
            0x5 => {
                let d = self.data[d4];
                self.data[d4] = self.sub8(d, k8, false, false);
            }
            0x6 => {
                let v = self.data[d4] | k8;
                self.data[d4] = self.logic(v);
            }
            0x7 => {
                let v = self.data[d4] & k8;
                self.data[d4] = self.logic(v);
            }
            0x8 | 0xa => {
                // ldd/std Y+q, Z+q
                let q = u32::from(((op >> 8) & 0x20) | ((op >> 7) & 0x18) | (op & 7));
                let base = if op & 8 != 0 { self.pair(28) } else { self.pair(30) };
                let a = u32::from(base) + q;
                if op & 0x0200 != 0 {
                    let v = self.data[d5];
                    self.wr(a, v)?;
                } else {
                    self.data[d5] = self.rd(a)?;
                }
            }
            0x9 => self.exec9(op, d5, r5, &mut next)?,
            0xb => {
                let a = usize::from(((op >> 5) & 0x30) | (op & 0xf)) + 0x20;
                if op & 0x0800 != 0 {
                    self.data[a] = self.data[d5];
                } else {
                    self.data[d5] = self.data[a];
                }
            }
            0xc | 0xd => {
                let k = ((op & 0xfff) as i16) << 4 >> 4;
                if op >> 12 == 0xd {
                    self.push_ret(pc + 1)?;
                }
                if k == -1 {
                    return Err(Fault(format!("halted in a self-loop at {:#x}", pc * 2)));
                }
                next = (pc as i32 + 1 + i32::from(k)) as u32 & 0x7fff;
            }
            0xe => self.data[d4] = k8,
            0xf => {
                if op & 0x0800 == 0 {
                    let s = op & 7;
                    let k = (((op >> 3) & 0x7f) as i8) << 1 >> 1;
                    let set = self.sreg() & (1 << s) != 0;
                    let want = op & 0x0400 == 0;
                    if set == want {
                        next = (pc as i32 + 1 + i32::from(k)) as u32;
                    }
                } else if op & 0x0c08 == 0x0800 {
                    // bld / bst (the T flag)
                    let b = 1u8 << (op & 7);
                    if op & 0x0200 != 0 {
                        let t = self.data[d5] & b != 0;
                        self.set_flags(1 << 6, if t { 1 << 6 } else { 0 });
                    } else if self.flag(1 << 6) {
                        self.data[d5] |= b;
                    } else {
                        self.data[d5] &= !b;
                    }
                } else if op & 0x0c08 == 0x0c00 {
                    // sbrc / sbrs
                    let bit = self.data[d5] & (1 << (op & 7)) != 0;
                    let skip_if_set = op & 0x0200 != 0;
                    if bit == skip_if_set {
                        next += self.len(next);
                    }
                } else {
                    return Err(bad());
                }
            }
            _ => return Err(bad()),
        }
        self.pc = next;
        Ok(())
    }

    /// The length in words of the instruction at `w`.
    fn len(&self, w: u32) -> u32 {
        let op = self.word(w);
        if op & 0xfe0c == 0x940c || op & 0xfe0f == 0x9000 || op & 0xfe0f == 0x9200 { 2 } else { 1 }
    }

    fn exec9(&mut self, op: u16, d5: usize, r5: usize, next: &mut u32) -> Result<(), Fault> {
        let pc = self.pc;
        let bad = || Fault(format!("unknown instruction {op:#06x} at {:#x}", pc * 2));
        match (op >> 9) & 7 {
            0 => {
                // loads: ld/lpm/pop
                match op & 0xf {
                    0x4 | 0x5 => {
                        let z = self.pair(30);
                        self.data[d5] = self.flash[usize::from(z)];
                        if op & 1 != 0 {
                            self.set_pair(30, z.wrapping_add(1));
                        }
                    }
                    0x9 | 0x1 | 0xd => {
                        let p = match op & 0xf {
                            0x9 => 28,
                            0x1 => 30,
                            _ => 26,
                        };
                        let a = self.pair(p);
                        self.data[d5] = self.rd(u32::from(a))?;
                        self.set_pair(p, a.wrapping_add(1));
                    }
                    0xc => {
                        let a = self.pair(26);
                        self.data[d5] = self.rd(u32::from(a))?;
                    }
                    0xf => self.data[d5] = self.pop()?,
                    _ => return Err(bad()),
                }
            }
            1 => match op & 0xf {
                0x9 | 0x1 | 0xd => {
                    let p = match op & 0xf {
                        0x9 => 28,
                        0x1 => 30,
                        _ => 26,
                    };
                    let a = self.pair(p);
                    let v = self.data[d5];
                    self.wr(u32::from(a), v)?;
                    self.set_pair(p, a.wrapping_add(1));
                }
                0xc => {
                    let a = self.pair(26);
                    let v = self.data[d5];
                    self.wr(u32::from(a), v)?;
                }
                0xf => {
                    let v = self.data[d5];
                    self.push(v)?;
                }
                _ => return Err(bad()),
            },
            2 => {
                // one-operand ops, jmp/call, misc
                if op & 0xfe0c == 0x940c {
                    let k = (u32::from((op >> 4) & 0x1f) << 17) | (u32::from(op & 1) << 16) | u32::from(self.word(pc + 1));
                    if op & 2 != 0 {
                        self.push_ret(pc + 2)?;
                    }
                    *next = k;
                    return Ok(());
                }
                let d = self.data[d5];
                match op {
                    0x9508 => {
                        let hi = self.pop()?;
                        let lo = self.pop()?;
                        *next = u32::from(lo) | (u32::from(hi) << 8);
                        return Ok(());
                    }
                    0x9509 => {
                        self.push_ret(pc + 1)?;
                        *next = u32::from(self.pair(30));
                        return Ok(());
                    }
                    0x9409 => {
                        *next = u32::from(self.pair(30));
                        return Ok(());
                    }
                    0x94f8 => {
                        self.set_flags(I, 0);
                        return Ok(());
                    }
                    0x9478 => {
                        self.set_flags(I, I);
                        return Ok(());
                    }
                    0x9598 => return Err(Fault(format!("break at {:#x}", pc * 2))),
                    _ => {}
                }
                match op & 0xfe0f {
                    0x9400 => {
                        let r = !d;
                        self.nzs(r, V | C, C);
                        self.data[d5] = r;
                    }
                    0x9401 => {
                        let r = self.sub8(0, d, false, false);
                        self.data[d5] = r;
                    }
                    0x9402 => self.data[d5] = d.rotate_left(4),
                    0x9403 => {
                        let r = d.wrapping_add(1);
                        self.nzs(r, V, if d == 0x7f { V } else { 0 });
                        self.data[d5] = r;
                    }
                    0x9405..=0x9407 => {
                        let c = d & 1;
                        let r = match op & 0xf {
                            5 => (d >> 1) | (d & 0x80),
                            6 => d >> 1,
                            _ => (d >> 1) | (u8::from(self.flag(C)) << 7),
                        };
                        let n = r & 0x80 != 0;
                        let v = n ^ (c != 0);
                        self.nzs(r, V | C, if v { V } else { 0 } | c);
                        self.data[d5] = r;
                    }
                    0x940a => {
                        let r = d.wrapping_sub(1);
                        self.nzs(r, V, if d == 0x80 { V } else { 0 });
                        self.data[d5] = r;
                    }
                    _ => return Err(bad()),
                }
            }
            3 => {
                // adiw / sbiw
                let p = 24 + 2 * usize::from((op >> 4) & 3);
                let k = ((op >> 2) & 0x30) | (op & 0xf);
                let a = self.pair(p);
                let (r, c, v) = if op & 0x0100 == 0 {
                    let r = a.wrapping_add(k);
                    (r, u32::from(a) + u32::from(k) > 0xffff, (!a & r) & 0x8000 != 0)
                } else {
                    let r = a.wrapping_sub(k);
                    (r, k > a, (a & !r) & 0x8000 != 0)
                };
                self.set_pair(p, r);
                let n = r & 0x8000 != 0;
                let mut f = 0;
                if c {
                    f |= C;
                }
                if v {
                    f |= V;
                }
                if n {
                    f |= N;
                }
                if n ^ v {
                    f |= S;
                }
                if r == 0 {
                    f |= Z;
                }
                self.set_flags(C | V | N | S | Z, f);
            }
            6 | 7 => {
                // mul
                let p = u16::from(self.data[d5]) * u16::from(self.data[r5]);
                self.set_pair(0, p);
                let mut f = 0;
                if p & 0x8000 != 0 {
                    f |= C;
                }
                if p == 0 {
                    f |= Z;
                }
                self.set_flags(C | Z, f);
            }
            _ => return Err(bad()),
        }
        Ok(())
    }

    /// Run until the PC reaches the word address `stop`, for at most `budget`
    /// instructions.
    pub(crate) fn run_until(&mut self, stop: u32, budget: u64) -> Result<(), Fault> {
        let end = self.steps + budget;
        while self.pc != stop {
            if self.steps >= end {
                return Err(Fault(format!("out of steps at pc {:#x}", self.pc * 2)));
            }
            self.step()?;
            self.min_sp = self.min_sp.min(self.sp());
            if self.data[1] != 0 && self.pc == stop {
                return Err(Fault("r1 is not zero on return".into()));
            }
        }
        Ok(())
    }
}
