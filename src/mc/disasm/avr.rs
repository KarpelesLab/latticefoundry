//! The AVR decoder, from the AVR Instruction Set Manual: the AVRe+
//! instruction set of the ATmega devices (AVR5, the ATmega328P), plus the
//! extended-address forms of the larger cores (`eijmp`, `eicall`, `elpm`)
//! and the XMEGA read-modify-write forms (`xch`, `las`, `lac`, `lat`, `des`,
//! `spm Z+`).
//!
//! [`decode_inst`] turns one 16-bit word (two for `jmp`, `call`, `lds`,
//! `sts`) into a typed [`AvrInst`]; [`decode`] renders it the way
//! `llvm-objdump --mcpu=atmega328p` does — the manual's syntax with its
//! preferred aliases (`lsl`, `rol`, `tst`, `clr`, the `br*` condition names,
//! the `se*`/`cl*` flag names), hex immediates, `ldd r24, Y+0` for a
//! displacement-free `ld` through Y or Z, relative branches as `.+N`/`.-N`
//! (bytes from the next instruction) and `jmp`/`call` targets as byte
//! addresses. The AVR interpreter of the tests executes the same
//! [`AvrInst`]s, so there is one AVR decoder.

use super::Inst;

/// An AVR operation (one per instruction-set-manual entry; aliases such as
/// `lsl` or `breq` are rendering choices over these).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[allow(missing_docs)]
pub enum Op {
    Nop,
    Movw,
    Muls,
    Mulsu,
    Fmul,
    Fmuls,
    Fmulsu,
    Cpc,
    Sbc,
    Add,
    Cpse,
    Cp,
    Sub,
    Adc,
    And,
    Eor,
    Or,
    Mov,
    Cpi,
    Sbci,
    Subi,
    Ori,
    Andi,
    /// `ldd Rd, Y+q` / `ldd Rd, Z+q` (also `ld Rd, Y`/`Z`, `q` = 0).
    Ldd,
    /// `std Y+q, Rr` / `std Z+q, Rr`.
    Std,
    Lds,
    Sts,
    /// `ld Rd, X`/`X+`/`-X`/`Y+`/`-Y`/`Z+`/`-Z`.
    Ld,
    /// `st X, Rr` and the other pointer modes.
    St,
    /// `lpm` (implied r0, Z), `lpm Rd, Z`, `lpm Rd, Z+`.
    Lpm,
    Elpm,
    Xch,
    Las,
    Lac,
    Lat,
    Pop,
    Push,
    Com,
    Neg,
    Swap,
    Inc,
    Asr,
    Lsr,
    Ror,
    Dec,
    Bset,
    Bclr,
    Ret,
    Reti,
    Sleep,
    Break,
    Wdr,
    /// `spm`, or `spm Z+` with [`Ptr::ZInc`].
    Spm,
    Ijmp,
    Eijmp,
    Icall,
    Eicall,
    Jmp,
    Call,
    Des,
    Adiw,
    Sbiw,
    Cbi,
    Sbic,
    Sbi,
    Sbis,
    Mul,
    In,
    Out,
    Rjmp,
    Rcall,
    Ldi,
    Brbs,
    Brbc,
    Bld,
    Bst,
    Sbrc,
    Sbrs,
}

/// A pointer-register addressing mode.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[allow(missing_docs)]
pub enum Ptr {
    /// No pointer operand.
    #[default]
    None,
    X,
    XInc,
    XDec,
    Y,
    YInc,
    YDec,
    Z,
    ZInc,
    ZDec,
}

impl Ptr {
    /// The operand as written (`X`, `X+`, `-X`, ...).
    pub fn text(self) -> &'static str {
        match self {
            Ptr::None => "",
            Ptr::X => "X",
            Ptr::XInc => "X+",
            Ptr::XDec => "-X",
            Ptr::Y => "Y",
            Ptr::YInc => "Y+",
            Ptr::YDec => "-Y",
            Ptr::Z => "Z",
            Ptr::ZInc => "Z+",
            Ptr::ZDec => "-Z",
        }
    }

    /// The low register of the pointer pair (26 for X, 28 for Y, 30 for Z).
    pub fn base(self) -> usize {
        match self {
            Ptr::None => 0,
            Ptr::X | Ptr::XInc | Ptr::XDec => 26,
            Ptr::Y | Ptr::YInc | Ptr::YDec => 28,
            Ptr::Z | Ptr::ZInc | Ptr::ZDec => 30,
        }
    }

    /// Whether the mode post-increments the pointer.
    pub fn post_inc(self) -> bool {
        matches!(self, Ptr::XInc | Ptr::YInc | Ptr::ZInc)
    }

    /// Whether the mode pre-decrements the pointer.
    pub fn pre_dec(self) -> bool {
        matches!(self, Ptr::XDec | Ptr::YDec | Ptr::ZDec)
    }
}

/// One decoded AVR instruction. Fields an operation does not use are zero
/// (and [`Ptr::None`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AvrInst {
    /// The operation.
    pub op: Op,
    /// `Rd`: the destination (or only) register; for `movw`, `adiw` and
    /// `sbiw` the low register of the pair.
    pub d: u8,
    /// `Rr`: the source register (the stored register of `st`, `std`,
    /// `sts`, `push`, `out`, `sbrc`/`sbrs`; the low register of `movw`'s
    /// source pair).
    pub r: u8,
    /// The constant: `K` (8-bit immediates, `adiw`/`sbiw`, `des`); the signed
    /// word displacement of `rjmp`/`rcall`/`brbs`/`brbc`; the word address of
    /// `jmp`/`call`; the data address of `lds`/`sts`; `q` of `ldd`/`std`.
    pub k: i32,
    /// The I/O address `A` (`in`, `out`, `sbi`, `cbi`, `sbic`, `sbis`).
    pub a: u8,
    /// The bit number `b` (`sbi`/`cbi`/`sbic`/`sbis`, `sbrc`/`sbrs`,
    /// `bst`/`bld`) or the SREG bit `s` (`bset`/`bclr`/`brbs`/`brbc`).
    pub b: u8,
    /// The pointer operand of `ld`, `st`, `ldd`, `std`, `lpm`, `elpm`,
    /// `xch`/`las`/`lac`/`lat` and `spm Z+`.
    pub ptr: Ptr,
    /// Length in bytes: 2, or 4 for `jmp`, `call`, `lds`, `sts`.
    pub len: u8,
}

impl AvrInst {
    /// An instruction of `op` with every operand zero.
    pub fn new(op: Op) -> AvrInst {
        AvrInst { op, d: 0, r: 0, k: 0, a: 0, b: 0, ptr: Ptr::None, len: 2 }
    }

    /// The branch target of a relative or absolute jump/call/branch, as a
    /// byte address, for an instruction at byte address `addr`.
    pub fn target(&self, addr: u64) -> Option<u64> {
        match self.op {
            Op::Rjmp | Op::Rcall | Op::Brbs | Op::Brbc => {
                Some(addr.wrapping_add(2).wrapping_add((i64::from(self.k) * 2) as u64))
            }
            Op::Jmp | Op::Call => Some(u64::from(self.k as u32) * 2),
            _ => None,
        }
    }
}

/// Decode the instruction at the start of `bytes` (little-endian words).
/// `None` for a reserved or unknown encoding, or a truncated two-word one.
pub fn decode_inst(bytes: &[u8]) -> Option<AvrInst> {
    let w = u16::from_le_bytes([*bytes.first()?, *bytes.get(1)?]);
    let second = || Some(u16::from_le_bytes([*bytes.get(2)?, *bytes.get(3)?]));
    use Op::*;
    let d5 = ((w >> 4) & 0x1f) as u8;
    let r5 = (((w >> 5) & 0x10) | (w & 0xf)) as u8;
    let d4 = 16 + ((w >> 4) & 0xf) as u8;
    let r4 = 16 + (w & 0xf) as u8;
    let k8 = i32::from(((w >> 4) & 0xf0) | (w & 0xf));
    let mk = |op, d: u8, r: u8| AvrInst { d, r, ..AvrInst::new(op) };
    let imm = |op, d: u8, k: i32| AvrInst { d, k, ..AvrInst::new(op) };
    let ptr = |op, reg: u8, p: Ptr, store: bool| {
        let i = AvrInst { ptr: p, ..AvrInst::new(op) };
        if store { AvrInst { r: reg, ..i } } else { AvrInst { d: reg, ..i } }
    };
    Some(match w >> 12 {
        0x0 => match (w >> 8) & 0xf {
            0x0 if w == 0 => AvrInst::new(Nop),
            0x0 => return None,
            0x1 => mk(Movw, ((w >> 4) & 0xf) as u8 * 2, (w & 0xf) as u8 * 2),
            0x2 => mk(Muls, d4, r4),
            0x3 => {
                let (d, r) = (16 + ((w >> 4) & 7) as u8, 16 + (w & 7) as u8);
                let op = match (w >> 7 & 1, w >> 3 & 1) {
                    (0, 0) => Mulsu,
                    (0, _) => Fmul,
                    (_, 0) => Fmuls,
                    _ => Fmulsu,
                };
                mk(op, d, r)
            }
            0x4..=0x7 => mk(Cpc, d5, r5),
            0x8..=0xb => mk(Sbc, d5, r5),
            _ => mk(Add, d5, r5),
        },
        0x1 => mk([Cpse, Cp, Sub, Adc][usize::from((w >> 10) & 3)], d5, r5),
        0x2 => mk([And, Eor, Or, Mov][usize::from((w >> 10) & 3)], d5, r5),
        0x3 => imm(Cpi, d4, k8),
        0x4 => imm(Sbci, d4, k8),
        0x5 => imm(Subi, d4, k8),
        0x6 => imm(Ori, d4, k8),
        0x7 => imm(Andi, d4, k8),
        0x8 | 0xa => {
            let q = i32::from(((w >> 8) & 0x20) | ((w >> 7) & 0x18) | (w & 7));
            let p = if w & 8 != 0 { Ptr::Y } else { Ptr::Z };
            if w & 0x0200 != 0 {
                AvrInst { r: d5, k: q, ptr: p, ..AvrInst::new(Std) }
            } else {
                AvrInst { d: d5, k: q, ptr: p, ..AvrInst::new(Ldd) }
            }
        }
        0x9 => match (w >> 9) & 7 {
            0 | 1 => {
                let store = w & 0x0200 != 0;
                match (w & 0xf, store) {
                    (0x0, false) => AvrInst { d: d5, k: i32::from(second()?), len: 4, ..AvrInst::new(Lds) },
                    (0x0, true) => AvrInst { r: d5, k: i32::from(second()?), len: 4, ..AvrInst::new(Sts) },
                    (0x1, _) => ptr(if store { St } else { Ld }, d5, Ptr::ZInc, store),
                    (0x2, _) => ptr(if store { St } else { Ld }, d5, Ptr::ZDec, store),
                    (0x4, false) => ptr(Lpm, d5, Ptr::Z, false),
                    (0x5, false) => ptr(Lpm, d5, Ptr::ZInc, false),
                    (0x6, false) => ptr(Elpm, d5, Ptr::Z, false),
                    (0x7, false) => ptr(Elpm, d5, Ptr::ZInc, false),
                    (0x4, true) => ptr(Xch, d5, Ptr::Z, false),
                    (0x5, true) => ptr(Las, d5, Ptr::Z, false),
                    (0x6, true) => ptr(Lac, d5, Ptr::Z, false),
                    (0x7, true) => ptr(Lat, d5, Ptr::Z, false),
                    (0x9, _) => ptr(if store { St } else { Ld }, d5, Ptr::YInc, store),
                    (0xa, _) => ptr(if store { St } else { Ld }, d5, Ptr::YDec, store),
                    (0xc, _) => ptr(if store { St } else { Ld }, d5, Ptr::X, store),
                    (0xd, _) => ptr(if store { St } else { Ld }, d5, Ptr::XInc, store),
                    (0xe, _) => ptr(if store { St } else { Ld }, d5, Ptr::XDec, store),
                    (0xf, false) => mk(Pop, d5, 0),
                    (0xf, true) => mk(Push, 0, d5),
                    _ => return None,
                }
            }
            2 => {
                if w & 0xc == 0xc {
                    // jmp (110k) / call (111k): a 22-bit word address.
                    let k = (u32::from((w >> 4) & 0x1f) << 17) | (u32::from(w & 1) << 16) | u32::from(second()?);
                    return Some(AvrInst { k: k as i32, len: 4, ..AvrInst::new(if w & 2 != 0 { Call } else { Jmp }) });
                }
                match w & 0xf {
                    0x0 => mk(Com, d5, 0),
                    0x1 => mk(Neg, d5, 0),
                    0x2 => mk(Swap, d5, 0),
                    0x3 => mk(Inc, d5, 0),
                    0x5 => mk(Asr, d5, 0),
                    0x6 => mk(Lsr, d5, 0),
                    0x7 => mk(Ror, d5, 0),
                    0xa => mk(Dec, d5, 0),
                    0x8 if w & 0x0100 == 0 => {
                        AvrInst { b: ((w >> 4) & 7) as u8, ..AvrInst::new(if w & 0x80 != 0 { Bclr } else { Bset }) }
                    }
                    0x8 => match w {
                        0x9508 => AvrInst::new(Ret),
                        0x9518 => AvrInst::new(Reti),
                        0x9588 => AvrInst::new(Sleep),
                        0x9598 => AvrInst::new(Break),
                        0x95a8 => AvrInst::new(Wdr),
                        0x95c8 => AvrInst::new(Lpm),
                        0x95d8 => AvrInst::new(Elpm),
                        0x95e8 => AvrInst::new(Spm),
                        0x95f8 => AvrInst { ptr: Ptr::ZInc, ..AvrInst::new(Spm) },
                        _ => return None,
                    },
                    0x9 => match w {
                        0x9409 => AvrInst::new(Ijmp),
                        0x9419 => AvrInst::new(Eijmp),
                        0x9509 => AvrInst::new(Icall),
                        0x9519 => AvrInst::new(Eicall),
                        _ => return None,
                    },
                    0xb if w & 0x0100 == 0 => AvrInst { k: i32::from((w >> 4) & 0xf), ..AvrInst::new(Des) },
                    _ => return None,
                }
            }
            3 => {
                let d = 24 + 2 * ((w >> 4) & 3) as u8;
                let k = i32::from(((w >> 2) & 0x30) | (w & 0xf));
                imm(if w & 0x0100 != 0 { Sbiw } else { Adiw }, d, k)
            }
            4 | 5 => {
                let op = [Cbi, Sbic, Sbi, Sbis][usize::from((w >> 8) & 3)];
                AvrInst { a: ((w >> 3) & 0x1f) as u8, b: (w & 7) as u8, ..AvrInst::new(op) }
            }
            _ => mk(Mul, d5, r5),
        },
        0xb => {
            let a = (((w >> 5) & 0x30) | (w & 0xf)) as u8;
            if w & 0x0800 != 0 {
                AvrInst { r: d5, a, ..AvrInst::new(Out) }
            } else {
                AvrInst { d: d5, a, ..AvrInst::new(In) }
            }
        }
        0xc | 0xd => {
            let k = i32::from(((w & 0xfff) as i16) << 4 >> 4);
            AvrInst { k, ..AvrInst::new(if w >> 12 == 0xd { Rcall } else { Rjmp }) }
        }
        0xe => imm(Ldi, d4, k8),
        _ => {
            if w & 0x0800 == 0 {
                let k = i32::from((((w >> 3) & 0x7f) as i8) << 1 >> 1);
                AvrInst { k, b: (w & 7) as u8, ..AvrInst::new(if w & 0x0400 != 0 { Brbc } else { Brbs }) }
            } else if w & 8 != 0 {
                return None;
            } else {
                let b = (w & 7) as u8;
                match (w >> 9) & 3 {
                    0 => AvrInst { d: d5, b, ..AvrInst::new(Bld) },
                    1 => AvrInst { d: d5, b, ..AvrInst::new(Bst) },
                    2 => AvrInst { r: d5, b, ..AvrInst::new(Sbrc) },
                    _ => AvrInst { r: d5, b, ..AvrInst::new(Sbrs) },
                }
            }
        }
    })
}

/// The 16-bit word(s) of an instruction: the inverse of [`decode_inst`]
/// (`None` for an operand out of its field's range). The second word is 0
/// for a one-word instruction.
pub fn encode_inst(i: &AvrInst) -> Option<(u16, u16)> {
    use Op::*;
    let reg = |r: u8| -> Option<u16> { (r < 32).then_some(u16::from(r)) };
    let hi = |r: u8| -> Option<u16> { (16..32).contains(&r).then_some(u16::from(r - 16)) };
    let rr = |base: u16, d: u8, r: u8| -> Option<u16> {
        let (d, r) = (reg(d)?, reg(r)?);
        Some(base | (r & 0x10) << 5 | d << 4 | (r & 0xf))
    };
    let kk = |base: u16, d: u8, k: i32| -> Option<u16> {
        let k = u16::try_from(k).ok().filter(|&k| k < 256)?;
        Some(base | (k & 0xf0) << 4 | hi(d)? << 4 | (k & 0xf))
    };
    let one = |w: u16| Some((w, 0));
    match i.op {
        Nop => one(0),
        Movw => (i.d.is_multiple_of(2) && i.r.is_multiple_of(2) && i.d < 32 && i.r < 32)
            .then(|| (0x0100 | u16::from(i.d / 2) << 4 | u16::from(i.r / 2), 0)),
        Muls => one(0x0200 | hi(i.d)? << 4 | hi(i.r)?),
        Mulsu | Fmul | Fmuls | Fmulsu => {
            let (d, r) = (hi(i.d)?, hi(i.r)?);
            if d > 7 || r > 7 {
                return None;
            }
            let bits = match i.op {
                Mulsu => 0x00,
                Fmul => 0x08,
                Fmuls => 0x80,
                _ => 0x88,
            };
            one(0x0300 | bits | d << 4 | r)
        }
        Cpc => one(rr(0x0400, i.d, i.r)?),
        Sbc => one(rr(0x0800, i.d, i.r)?),
        Add => one(rr(0x0c00, i.d, i.r)?),
        Cpse => one(rr(0x1000, i.d, i.r)?),
        Cp => one(rr(0x1400, i.d, i.r)?),
        Sub => one(rr(0x1800, i.d, i.r)?),
        Adc => one(rr(0x1c00, i.d, i.r)?),
        And => one(rr(0x2000, i.d, i.r)?),
        Eor => one(rr(0x2400, i.d, i.r)?),
        Or => one(rr(0x2800, i.d, i.r)?),
        Mov => one(rr(0x2c00, i.d, i.r)?),
        Mul => one(rr(0x9c00, i.d, i.r)?),
        Cpi => one(kk(0x3000, i.d, i.k)?),
        Sbci => one(kk(0x4000, i.d, i.k)?),
        Subi => one(kk(0x5000, i.d, i.k)?),
        Ori => one(kk(0x6000, i.d, i.k)?),
        Andi => one(kk(0x7000, i.d, i.k)?),
        Ldi => one(kk(0xe000, i.d, i.k)?),
        Ldd | Std => {
            let q = u16::try_from(i.k).ok().filter(|&q| q < 64)?;
            let y = match i.ptr {
                Ptr::Y => 8,
                Ptr::Z => 0,
                _ => return None,
            };
            let (r, s) = if i.op == Std { (reg(i.r)?, 0x0200) } else { (reg(i.d)?, 0) };
            one(0x8000 | (q & 0x20) << 8 | (q & 0x18) << 7 | s | r << 4 | y | (q & 7))
        }
        Lds | Sts => {
            let k = u16::try_from(i.k).ok()?;
            let (r, s) = if i.op == Sts { (reg(i.r)?, 0x0200) } else { (reg(i.d)?, 0) };
            Some((0x9000 | s | r << 4, k))
        }
        Ld | St => {
            let mode = match i.ptr {
                Ptr::ZInc => 0x1,
                Ptr::ZDec => 0x2,
                Ptr::YInc => 0x9,
                Ptr::YDec => 0xa,
                Ptr::X => 0xc,
                Ptr::XInc => 0xd,
                Ptr::XDec => 0xe,
                _ => return None,
            };
            let (r, s) = if i.op == St { (reg(i.r)?, 0x0200) } else { (reg(i.d)?, 0) };
            one(0x9000 | s | r << 4 | mode)
        }
        Lpm | Elpm if i.ptr == Ptr::None => one(if i.op == Lpm { 0x95c8 } else { 0x95d8 }),
        Lpm | Elpm => {
            let base = if i.op == Lpm { 0x9004 } else { 0x9006 };
            let inc = match i.ptr {
                Ptr::Z => 0,
                Ptr::ZInc => 1,
                _ => return None,
            };
            one(base | reg(i.d)? << 4 | inc)
        }
        Xch | Las | Lac | Lat => {
            let n = match i.op {
                Xch => 4,
                Las => 5,
                Lac => 6,
                _ => 7,
            };
            one(0x9200 | reg(i.d)? << 4 | n)
        }
        Pop => one(0x900f | reg(i.d)? << 4),
        Push => one(0x920f | reg(i.r)? << 4),
        Com => one(0x9400 | reg(i.d)? << 4),
        Neg => one(0x9401 | reg(i.d)? << 4),
        Swap => one(0x9402 | reg(i.d)? << 4),
        Inc => one(0x9403 | reg(i.d)? << 4),
        Asr => one(0x9405 | reg(i.d)? << 4),
        Lsr => one(0x9406 | reg(i.d)? << 4),
        Ror => one(0x9407 | reg(i.d)? << 4),
        Dec => one(0x940a | reg(i.d)? << 4),
        Bset | Bclr => (i.b < 8).then(|| (0x9408 | if i.op == Bclr { 0x80 } else { 0 } | u16::from(i.b) << 4, 0)),
        Ret => one(0x9508),
        Reti => one(0x9518),
        Sleep => one(0x9588),
        Break => one(0x9598),
        Wdr => one(0x95a8),
        Spm => one(if i.ptr == Ptr::ZInc { 0x95f8 } else { 0x95e8 }),
        Ijmp => one(0x9409),
        Eijmp => one(0x9419),
        Icall => one(0x9509),
        Eicall => one(0x9519),
        Jmp | Call => {
            let k = u32::try_from(i.k).ok().filter(|&k| k < 1 << 22)?;
            let w = 0x940c | if i.op == Call { 2 } else { 0 } | ((k >> 17) as u16) << 4 | ((k >> 16) & 1) as u16;
            Some((w, k as u16))
        }
        Des => (0..16).contains(&i.k).then_some((0x940b | (i.k as u16) << 4, 0)),
        Adiw | Sbiw => {
            let k = u16::try_from(i.k).ok().filter(|&k| k < 64)?;
            if !matches!(i.d, 24 | 26 | 28 | 30) {
                return None;
            }
            let base = if i.op == Sbiw { 0x9700 } else { 0x9600 };
            one(base | (k & 0x30) << 2 | u16::from((i.d - 24) / 2) << 4 | (k & 0xf))
        }
        Cbi | Sbic | Sbi | Sbis => {
            if i.a >= 32 || i.b >= 8 {
                return None;
            }
            let n = match i.op {
                Cbi => 0,
                Sbic => 1,
                Sbi => 2,
                _ => 3,
            };
            one(0x9800 | n << 8 | u16::from(i.a) << 3 | u16::from(i.b))
        }
        In | Out => {
            if i.a >= 64 {
                return None;
            }
            let a = u16::from(i.a);
            let (r, s) = if i.op == Out { (reg(i.r)?, 0x0800) } else { (reg(i.d)?, 0) };
            one(0xb000 | s | (a & 0x30) << 5 | r << 4 | (a & 0xf))
        }
        Rjmp | Rcall => {
            if !(-2048..2048).contains(&i.k) {
                return None;
            }
            one(if i.op == Rcall { 0xd000 } else { 0xc000 } | (i.k as u16 & 0xfff))
        }
        Brbs | Brbc => {
            if !(-64..64).contains(&i.k) || i.b >= 8 {
                return None;
            }
            one(0xf000 | if i.op == Brbc { 0x0400 } else { 0 } | (i.k as u16 & 0x7f) << 3 | u16::from(i.b))
        }
        Bld | Bst | Sbrc | Sbrs => {
            if i.b >= 8 {
                return None;
            }
            let (r, n) = match i.op {
                Bld => (i.d, 0),
                Bst => (i.d, 1),
                Sbrc => (i.r, 2),
                _ => (i.r, 3),
            };
            one(0xf800 | n << 9 | reg(r)? << 4 | u16::from(i.b))
        }
    }
}

/// The mnemonic of a status-register bit branch: `brbs`/`brbc` with each
/// bit's preferred name (as llvm-objdump prints them; `brlo`/`brsh` for the
/// carry flag).
fn branch_name(set: bool, s: u8) -> &'static str {
    const SET: [&str; 8] = ["brlo", "breq", "brmi", "brvs", "brlt", "brhs", "brts", "brie"];
    const CLEAR: [&str; 8] = ["brsh", "brne", "brpl", "brvc", "brge", "brhc", "brtc", "brid"];
    if set { SET[usize::from(s & 7)] } else { CLEAR[usize::from(s & 7)] }
}

/// The `bset`/`bclr` aliases: `sec`, `sez`, ... / `clc`, `clz`, ...
fn flag_name(set: bool, s: u8) -> &'static str {
    const SET: [&str; 8] = ["sec", "sez", "sen", "sev", "ses", "seh", "set", "sei"];
    const CLEAR: [&str; 8] = ["clc", "clz", "cln", "clv", "cls", "clh", "clt", "cli"];
    if set { SET[usize::from(s & 7)] } else { CLEAR[usize::from(s & 7)] }
}

fn hex(v: i64) -> String {
    format!("{v:#x}")
}

/// A relative displacement in bytes from the next instruction: `.+4`, `.-2`.
fn rel(words: i32) -> String {
    let bytes = words * 2;
    if bytes < 0 { format!(".-{}", -bytes) } else { format!(".+{bytes}") }
}

/// Render `i`, located at byte address `addr`.
pub fn render(i: &AvrInst, addr: u64) -> Inst {
    use Op::*;
    let len = usize::from(i.len);
    let r = |n: u8| format!("r{n}");
    let name = |op: Op| format!("{op:?}").to_ascii_lowercase();
    let rd_rr = |m: &str| Inst::new(len, m).op(r(i.d)).op(r(i.r));
    match i.op {
        Add if i.d == i.r => Inst::new(len, "lsl").op(r(i.d)),
        Adc if i.d == i.r => Inst::new(len, "rol").op(r(i.d)),
        And if i.d == i.r => Inst::new(len, "tst").op(r(i.d)),
        Eor if i.d == i.r => Inst::new(len, "clr").op(r(i.d)),
        Movw | Muls | Mulsu | Fmul | Fmuls | Fmulsu | Cpc | Sbc | Add | Cpse | Cp | Sub | Adc | And | Eor | Or | Mov
        | Mul => rd_rr(&name(i.op)),
        Cpi | Sbci | Subi | Ori | Andi | Ldi | Adiw | Sbiw => {
            Inst::new(len, name(i.op)).op(r(i.d)).op(hex(i64::from(i.k)))
        }
        Ldd => Inst::new(len, "ldd").op(r(i.d)).op(format!("{}+{}", i.ptr.text(), i.k)),
        Std => Inst::new(len, "std").op(format!("{}+{}", i.ptr.text(), i.k)).op(r(i.r)),
        Lds => Inst::new(len, "lds").op(r(i.d)).op(hex(i64::from(i.k))),
        Sts => Inst::new(len, "sts").op(hex(i64::from(i.k))).op(r(i.r)),
        Ld | Lpm | Elpm if i.ptr != Ptr::None => Inst::new(len, name(i.op)).op(r(i.d)).op(i.ptr.text()),
        St => Inst::new(len, "st").op(i.ptr.text()).op(r(i.r)),
        Xch | Las | Lac | Lat => Inst::new(len, name(i.op)).op("Z").op(r(i.d)),
        Spm if i.ptr == Ptr::ZInc => Inst::new(len, "spm").op("Z+"),
        Pop | Com | Neg | Swap | Inc | Asr | Lsr | Ror | Dec => Inst::new(len, name(i.op)).op(r(i.d)),
        Push => Inst::new(len, "push").op(r(i.r)),
        Bset | Bclr => Inst::new(len, flag_name(i.op == Bset, i.b)),
        Des => Inst::new(len, "des").op(hex(i64::from(i.k))),
        Cbi | Sbic | Sbi | Sbis => Inst::new(len, name(i.op)).op(hex(i64::from(i.a))).op(hex(i64::from(i.b))),
        In => Inst::new(len, "in").op(r(i.d)).op(hex(i64::from(i.a))),
        Out => Inst::new(len, "out").op(hex(i64::from(i.a))).op(r(i.r)),
        Bld | Bst => Inst::new(len, name(i.op)).op(r(i.d)).op(hex(i64::from(i.b))),
        Sbrc | Sbrs => Inst::new(len, name(i.op)).op(r(i.r)).op(hex(i64::from(i.b))),
        Jmp | Call => Inst::new(len, name(i.op)).target_op(i.target(addr).unwrap_or(0)),
        Rjmp | Rcall | Brbs | Brbc => {
            let m = match i.op {
                Rjmp => "rjmp",
                Rcall => "rcall",
                _ => branch_name(i.op == Brbs, i.b),
            };
            let mut inst = Inst::new(len, m).op(rel(i.k));
            inst.target = i.target(addr);
            inst.target_operand = Some(0);
            inst
        }
        Nop | Ret | Reti | Sleep | Break | Wdr | Spm | Ijmp | Eijmp | Icall | Eicall | Lpm | Elpm | Ld => {
            Inst::new(len, name(i.op))
        }
    }
}

/// Decode one AVR instruction from the start of `bytes` (non-empty),
/// located at byte address `addr`. An unknown or truncated encoding is a
/// `.short` (a `.byte` for a lone trailing byte).
pub fn decode(bytes: &[u8], addr: u64) -> Inst {
    match decode_inst(bytes) {
        Some(i) => render(&i, addr),
        None => Inst::data(bytes, 2, true),
    }
}
