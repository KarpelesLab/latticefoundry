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
//!
//! Instructions are decoded by the disassembler's AVR decoder
//! ([`crate::mc::disasm::avr::decode_inst`]), so execution and `lf-dis`
//! read machine code the same way.

use std::fmt;

use super::Device;
use crate::mc::disasm::avr::{AvrInst, Op, Ptr, decode_inst};

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

    /// The instruction at word address `w`, decoded by the disassembler's
    /// decoder ([`decode_inst`]) — the one AVR decoder.
    fn fetch(&self, w: u32) -> Option<AvrInst> {
        let [a, b] = self.word(w).to_le_bytes();
        let [c, d] = self.word(w + 1).to_le_bytes();
        decode_inst(&[a, b, c, d])
    }

    /// Execute one instruction.
    pub(crate) fn step(&mut self) -> Result<(), Fault> {
        self.steps += 1;
        let pc = self.pc;
        let op = self.word(pc);
        let bad = || Fault(format!("unknown instruction {op:#06x} at {:#x}", pc * 2));
        let i = self.fetch(pc).ok_or_else(bad)?;
        let mut next = pc + u32::from(i.len) / 2;
        let (d, r) = (usize::from(i.d), usize::from(i.r));
        let (dv, rv) = (self.data[d], self.data[r]);
        let k8 = i.k as u8;
        match i.op {
            Op::Nop => {}
            Op::Movw => {
                let v = self.pair(r);
                self.set_pair(d, v);
            }
            Op::Cpc => {
                self.sub8(dv, rv, self.flag(C), true);
            }
            Op::Sbc => self.data[d] = self.sub8(dv, rv, self.flag(C), true),
            Op::Add => self.data[d] = self.add8(dv, rv, false),
            Op::Cpse => {
                if dv == rv {
                    next += self.len(next);
                }
            }
            Op::Cp => {
                self.sub8(dv, rv, false, false);
            }
            Op::Sub => self.data[d] = self.sub8(dv, rv, false, false),
            Op::Adc => self.data[d] = self.add8(dv, rv, self.flag(C)),
            Op::And => self.data[d] = self.logic(dv & rv),
            Op::Eor => self.data[d] = self.logic(dv ^ rv),
            Op::Or => self.data[d] = self.logic(dv | rv),
            Op::Mov => self.data[d] = rv,
            Op::Cpi => {
                self.sub8(dv, k8, false, false);
            }
            Op::Sbci => self.data[d] = self.sub8(dv, k8, self.flag(C), true),
            Op::Subi => self.data[d] = self.sub8(dv, k8, false, false),
            Op::Ori => self.data[d] = self.logic(dv | k8),
            Op::Andi => self.data[d] = self.logic(dv & k8),
            Op::Ldi => self.data[d] = k8,
            Op::Ldd | Op::Std => {
                let a = u32::from(self.pair(i.ptr.base())) + i.k as u32;
                if i.op == Op::Std {
                    self.wr(a, rv)?;
                } else {
                    self.data[d] = self.rd(a)?;
                }
            }
            Op::Lds => self.data[d] = self.rd(i.k as u32)?,
            Op::Sts => self.wr(i.k as u32, rv)?,
            Op::Ld | Op::St => {
                let p = i.ptr.base();
                let mut a = self.pair(p);
                if i.ptr.pre_dec() {
                    a = a.wrapping_sub(1);
                    self.set_pair(p, a);
                }
                if i.op == Op::St {
                    self.wr(u32::from(a), rv)?;
                } else {
                    self.data[d] = self.rd(u32::from(a))?;
                }
                if i.ptr.post_inc() {
                    self.set_pair(p, a.wrapping_add(1));
                }
            }
            Op::Lpm if i.ptr != Ptr::None => {
                let z = self.pair(30);
                self.data[d] = self.flash[usize::from(z)];
                if i.ptr.post_inc() {
                    self.set_pair(30, z.wrapping_add(1));
                }
            }
            Op::Pop => self.data[d] = self.pop()?,
            Op::Push => self.push(rv)?,
            Op::Jmp | Op::Call => {
                if i.op == Op::Call {
                    self.push_ret(pc + 2)?;
                }
                next = i.k as u32;
            }
            Op::Ret => {
                let hi = self.pop()?;
                let lo = self.pop()?;
                next = u32::from(lo) | (u32::from(hi) << 8);
            }
            Op::Icall => {
                self.push_ret(pc + 1)?;
                next = u32::from(self.pair(30));
            }
            Op::Ijmp => next = u32::from(self.pair(30)),
            // `cli` / `sei`: the interrupt flag.
            Op::Bclr if i.b == 7 => self.set_flags(I, 0),
            Op::Bset if i.b == 7 => self.set_flags(I, I),
            Op::Break => return Err(Fault(format!("break at {:#x}", pc * 2))),
            Op::Com => {
                let v = !dv;
                self.nzs(v, V | C, C);
                self.data[d] = v;
            }
            Op::Neg => self.data[d] = self.sub8(0, dv, false, false),
            Op::Swap => self.data[d] = dv.rotate_left(4),
            Op::Inc => {
                let v = dv.wrapping_add(1);
                self.nzs(v, V, if dv == 0x7f { V } else { 0 });
                self.data[d] = v;
            }
            Op::Asr | Op::Lsr | Op::Ror => {
                let c = dv & 1;
                let v = match i.op {
                    Op::Asr => (dv >> 1) | (dv & 0x80),
                    Op::Lsr => dv >> 1,
                    _ => (dv >> 1) | (u8::from(self.flag(C)) << 7),
                };
                let n = v & 0x80 != 0;
                let ov = n ^ (c != 0);
                self.nzs(v, V | C, if ov { V } else { 0 } | c);
                self.data[d] = v;
            }
            Op::Dec => {
                let v = dv.wrapping_sub(1);
                self.nzs(v, V, if dv == 0x80 { V } else { 0 });
                self.data[d] = v;
            }
            Op::Adiw | Op::Sbiw => {
                let k = i.k as u16;
                let a = self.pair(d);
                let (v, c, ov) = if i.op == Op::Adiw {
                    let v = a.wrapping_add(k);
                    (v, u32::from(a) + u32::from(k) > 0xffff, (!a & v) & 0x8000 != 0)
                } else {
                    let v = a.wrapping_sub(k);
                    (v, k > a, (a & !v) & 0x8000 != 0)
                };
                self.set_pair(d, v);
                let n = v & 0x8000 != 0;
                let mut f = 0;
                if c {
                    f |= C;
                }
                if ov {
                    f |= V;
                }
                if n {
                    f |= N;
                }
                if n ^ ov {
                    f |= S;
                }
                if v == 0 {
                    f |= Z;
                }
                self.set_flags(C | V | N | S | Z, f);
            }
            Op::Mul => {
                let p = u16::from(dv) * u16::from(rv);
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
            Op::In => self.data[d] = self.data[usize::from(i.a) + 0x20],
            Op::Out => self.data[usize::from(i.a) + 0x20] = rv,
            Op::Rjmp | Op::Rcall => {
                if i.op == Op::Rcall {
                    self.push_ret(pc + 1)?;
                }
                if i.k == -1 {
                    return Err(Fault(format!("halted in a self-loop at {:#x}", pc * 2)));
                }
                next = (pc as i32 + 1 + i.k) as u32 & 0x7fff;
            }
            Op::Brbs | Op::Brbc => {
                let set = self.sreg() & (1 << i.b) != 0;
                if set == (i.op == Op::Brbs) {
                    next = (pc as i32 + 1 + i.k) as u32;
                }
            }
            Op::Bld | Op::Bst => {
                // The T flag.
                let b = 1u8 << i.b;
                if i.op == Op::Bst {
                    let t = dv & b != 0;
                    self.set_flags(1 << 6, if t { 1 << 6 } else { 0 });
                } else if self.flag(1 << 6) {
                    self.data[d] |= b;
                } else {
                    self.data[d] &= !b;
                }
            }
            Op::Sbrc | Op::Sbrs => {
                let bit = rv & (1 << i.b) != 0;
                if bit == (i.op == Op::Sbrs) {
                    next += self.len(next);
                }
            }
            // Defined by the manual, but nothing the generated code uses.
            _ => return Err(bad()),
        }
        self.pc = next;
        Ok(())
    }

    /// The length in words of the instruction at `w` (an unknown word counts
    /// as one).
    fn len(&self, w: u32) -> u32 {
        self.fetch(w).map_or(1, |i| u32::from(i.len) / 2)
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
