//! The AArch64 (A64) decoder, written from the Arm Architecture Reference
//! Manual for A-profile (the "A64 Instruction Set Encoding" chapter and each
//! instruction's alias conditions).
//!
//! A 32-bit instruction word is decoded in two steps: [`decode_word`] walks
//! the manual's top-level encoding index (`op0` = bits 28:25, then each
//! group's own fields) into an [`A64Inst`] — a mnemonic plus typed
//! [`Operand`]s — choosing the manual's **preferred disassembly** alias
//! (`mov`, `cmp`, `lsl`, `ubfx`, `cset`, `mul`, ...); [`A64Inst::to_inst`]
//! renders it in the standard syntax (immediates in hex except shift
//! amounts and bitfield positions, as LLVM prints them).
//!
//! Covered: data processing (immediate and register, all of the base ISA),
//! branches, exception generation and the common system instructions
//! (hints, barriers, `mrs`/`msr` with the usual named registers, `dc`/`ic`),
//! every load/store class of the base ISA (unsigned/unscaled/pre/post/
//! register offsets, literal, pairs, exclusives and acquire/release, LSE
//! atomics), scalar floating point, and the Advanced SIMD subset of three-
//! same, two-register miscellaneous, across-lanes, copy, permute, extract,
//! table, shift-by-immediate, modified-immediate and single/multiple
//! structure load/store instructions. Anything else decodes as `.word`.
//!
//! To add an instruction, extend the match arm of its encoding group (each
//! group function is named after the manual's section).

use super::Inst;

// ===========================================================================
// The typed instruction
// ===========================================================================

/// A register operand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Reg {
    /// A general-purpose register; number 31 is the stack pointer when `sp`
    /// (else the zero register).
    Gp {
        /// The register number (0–31).
        n: u8,
        /// `X` (64-bit) rather than `W`.
        wide: bool,
        /// Number 31 means `sp`/`wsp` rather than `xzr`/`wzr`.
        sp: bool,
    },
    /// A scalar SIMD&FP register `b`/`h`/`s`/`d`/`q` `n`.
    Fp {
        /// The register number.
        n: u8,
        /// The width letter.
        size: char,
    },
    /// A vector register with an arrangement, `v0.4s`.
    Vec {
        /// The register number.
        n: u8,
        /// The arrangement (`8b`, `16b`, `4h`, `8h`, `2s`, `4s`, `1d`, `2d`).
        arr: &'static str,
    },
    /// A vector element, `v1.s[2]`.
    Elem {
        /// The register number.
        n: u8,
        /// The element size letter.
        size: char,
        /// The element index.
        index: u8,
    },
}

impl Reg {
    fn x(n: u32) -> Reg {
        Reg::Gp { n: n as u8, wide: true, sp: false }
    }
    fn gp(n: u32, wide: bool) -> Reg {
        Reg::Gp { n: n as u8, wide, sp: false }
    }
    fn gp_sp(n: u32, wide: bool) -> Reg {
        Reg::Gp { n: n as u8, wide, sp: true }
    }
    fn fp(n: u32, size: char) -> Reg {
        Reg::Fp { n: n as u8, size }
    }
    fn v(n: u32, arr: &'static str) -> Reg {
        Reg::Vec { n: n as u8, arr }
    }

    fn render(self) -> String {
        match self {
            Reg::Gp { n: 31, wide, sp: true } => (if wide { "sp" } else { "wsp" }).to_owned(),
            Reg::Gp { n: 31, wide, sp: false } => (if wide { "xzr" } else { "wzr" }).to_owned(),
            Reg::Gp { n, wide, .. } => format!("{}{n}", if wide { 'x' } else { 'w' }),
            Reg::Fp { n, size } => format!("{size}{n}"),
            Reg::Vec { n, arr } => format!("v{n}.{arr}"),
            Reg::Elem { n, size, index } => format!("v{n}.{size}[{index}]"),
        }
    }
}

/// How a memory operand is indexed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Index {
    /// `[base, #imm]` (the offset is omitted when zero).
    Offset(i64),
    /// `[base, #imm]!`.
    Pre(i64),
    /// `[base], #imm` (hex).
    Post(i64),
    /// `[base], #imm` with a decimal immediate (structure loads/stores).
    PostDec(i64),
    /// `[base], xm`.
    PostReg(u8),
    /// `[base, Rm{, extend {#amount}}]`.
    Reg {
        /// The index register.
        rm: u8,
        /// `X` index rather than `W`.
        wide: bool,
        /// The extend/shift name (`lsl`, `uxtw`, `sxtw`, `sxtx`), if printed.
        ext: Option<&'static str>,
        /// The shift amount, if printed.
        amount: Option<u32>,
    },
}

/// A memory operand `[Xn|SP ...]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Mem {
    /// The base register (31 = `sp`).
    pub base: u8,
    /// The indexing.
    pub index: Index,
}

/// An operand of an [`A64Inst`].
#[derive(Clone, PartialEq, Debug)]
pub enum Operand {
    /// A register.
    Reg(Reg),
    /// A register list `{ v0.16b, v1.16b }` with an optional element index.
    List(Vec<Reg>, Option<u8>),
    /// An immediate printed in hex: `#0x10`, `#-0x8`.
    Imm(i64),
    /// An unsigned immediate printed in hex (logical immediates).
    UImm(u64),
    /// An immediate printed in decimal: `#3`.
    Dec(i64),
    /// A shift or extend modifier: `lsl #3`, `uxtw`, `sxtw #2`, `msl #8`.
    Modifier(&'static str, Option<u32>),
    /// A condition (`eq`, `ne`, ...).
    Cond(u8),
    /// A memory operand.
    Mem(Mem),
    /// A branch or literal target (absolute address).
    Target(u64),
    /// A floating-point immediate (`#1.00000000`).
    Float(f64),
    /// Anything else, printed verbatim (system register names, barrier
    /// options, prefetch operations, `c7`).
    Text(String),
}

/// A decoded A64 instruction: the (preferred) mnemonic and its operands.
#[derive(Clone, PartialEq, Debug)]
pub struct A64Inst {
    /// The mnemonic, e.g. `add`, `b.ne`, `ldr`.
    pub mnemonic: String,
    /// The operands, in assembly order.
    pub operands: Vec<Operand>,
}

const CONDS: [&str; 16] = ["eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "nv"];

fn hex(v: i64) -> String {
    if v < 0 { format!("#-{:#x}", v.unsigned_abs()) } else { format!("#{v:#x}") }
}

impl Operand {
    fn render(&self) -> String {
        match self {
            Operand::Reg(r) => r.render(),
            Operand::List(regs, idx) => {
                let body: Vec<String> = regs.iter().map(|r| r.render()).collect();
                match idx {
                    Some(i) => format!("{{ {} }}[{i}]", body.join(", ")),
                    None => format!("{{ {} }}", body.join(", ")),
                }
            }
            Operand::Imm(v) => hex(*v),
            Operand::UImm(v) => format!("#{v:#x}"),
            Operand::Dec(v) => format!("#{v}"),
            Operand::Modifier(m, None) => (*m).to_owned(),
            Operand::Modifier(m, Some(a)) => format!("{m} #{a}"),
            Operand::Cond(c) => CONDS[usize::from(*c & 15)].to_owned(),
            Operand::Mem(m) => {
                let base = Reg::gp_sp(u32::from(m.base), true).render();
                match m.index {
                    Index::Offset(0) => format!("[{base}]"),
                    Index::Offset(i) => format!("[{base}, {}]", hex(i)),
                    Index::Pre(i) => format!("[{base}, {}]!", hex(i)),
                    Index::Post(i) => format!("[{base}], {}", hex(i)),
                    Index::PostDec(i) => format!("[{base}], #{i}"),
                    Index::PostReg(r) => format!("[{base}], x{r}"),
                    Index::Reg { rm, wide, ext, amount } => {
                        let mut s = format!("[{base}, {}", Reg::gp(u32::from(rm), wide).render());
                        if let Some(e) = ext {
                            s.push_str(", ");
                            s.push_str(e);
                            if let Some(a) = amount {
                                s.push_str(&format!(" #{a}"));
                            }
                        }
                        s.push(']');
                        s
                    }
                }
            }
            Operand::Target(t) => format!("{t:#x}"),
            Operand::Float(f) => format!("#{f:.8}"),
            Operand::Text(t) => t.clone(),
        }
    }
}

impl A64Inst {
    fn new(mnemonic: impl Into<String>, operands: Vec<Operand>) -> A64Inst {
        A64Inst { mnemonic: mnemonic.into(), operands }
    }

    /// The uniform [`Inst`] (4 bytes), with the branch target recorded.
    pub fn to_inst(&self) -> Inst {
        let mut inst = Inst::new(4, self.mnemonic.clone());
        for op in &self.operands {
            match op {
                Operand::Target(t) => inst = inst.target_op(*t),
                other => inst = inst.op(other.render()),
            }
        }
        inst
    }
}

/// Decode one A64 instruction from the start of `bytes` (non-empty), located
/// at address `addr`. Fewer than four bytes, or an encoding this decoder does
/// not know, decode as a data directive.
pub fn decode(bytes: &[u8], addr: u64) -> Inst {
    if bytes.len() < 4 {
        return Inst::data(bytes, 4, true);
    }
    let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    match decode_word(w, addr) {
        Some(i) => i.to_inst(),
        None => Inst::data(&bytes[..4], 4, true),
    }
}

// ===========================================================================
// Field helpers
// ===========================================================================

#[inline]
fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

#[inline]
fn bit(w: u32, b: u32) -> bool {
    (w >> b) & 1 != 0
}

fn sext(v: u32, width: u32) -> i64 {
    super::sext(u64::from(v), width)
}

fn r(reg: Reg) -> Operand {
    Operand::Reg(reg)
}

fn ins(m: impl Into<String>, ops: Vec<Operand>) -> Option<A64Inst> {
    Some(A64Inst::new(m, ops))
}

/// Decode a 32-bit A64 instruction word located at `addr`; `None` for an
/// unallocated or unsupported encoding.
pub fn decode_word(w: u32, addr: u64) -> Option<A64Inst> {
    let op0 = bits(w, 28, 25);
    match op0 {
        0b0000 => {
            if bits(w, 31, 16) == 0 {
                ins("udf", vec![Operand::UImm(u64::from(bits(w, 15, 0)))])
            } else {
                None
            }
        }
        0b1000 | 0b1001 => dp_imm(w, addr),
        0b1010 | 0b1011 => branch_sys(w, addr),
        0b0101 | 0b1101 => dp_reg(w),
        0b0111 | 0b1111 => simd_fp(w),
        _ if op0 & 0b0101 == 0b0100 => ldst(w, addr),
        _ => None,
    }
}

// ===========================================================================
// Data processing — immediate
// ===========================================================================

/// The ARM ARM `DecodeBitMasks` for a logical immediate (`imm` of `datasize`
/// bits), `None` for a reserved encoding.
pub fn decode_bit_masks(n: u32, imms: u32, immr: u32, datasize: u32) -> Option<u64> {
    let combined = (n << 6) | (!imms & 0x3f);
    if combined == 0 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 {
        return None;
    }
    let levels = (1u32 << len) - 1;
    if imms & levels == levels {
        return None;
    }
    let s = imms & levels;
    let rot = immr & levels;
    let esize = 1u32 << len;
    if esize > datasize {
        return None;
    }
    let emask = if esize == 64 { u64::MAX } else { (1u64 << esize) - 1 };
    let welem = if s + 1 == 64 { u64::MAX } else { (1u64 << (s + 1)) - 1 };
    let elem = if rot == 0 { welem } else { ((welem >> rot) | (welem << (esize - rot))) & emask };
    let mut v = 0u64;
    let mut at = 0;
    while at < datasize {
        v |= elem << at;
        at += esize;
    }
    Some(if datasize == 64 { v } else { v & 0xffff_ffff })
}

/// The ARM ARM `MoveWidePreferred`: whether a `movz`/`movn` could produce the
/// bitmask immediate (so `orr Rd, zr, #imm` is not shown as `mov`).
fn move_wide_preferred(sf: bool, n: u32, imms: u32, immr: u32) -> bool {
    let (s, r) = (imms as i32, immr as i32);
    let width = if sf { 64 } else { 32 };
    if sf && n != 1 {
        return false;
    }
    if !sf && !(n == 0 && imms & 0x20 == 0) {
        return false;
    }
    if s < 16 {
        return (-r).rem_euclid(16) <= 15 - s;
    }
    if s >= width - 15 {
        return r % 16 <= s - (width - 15);
    }
    false
}

/// A value as the signed integer of its register width.
fn signed_of(v: u64, wide: bool) -> i64 {
    if wide { v as i64 } else { i64::from(v as u32 as i32) }
}

fn dp_imm(w: u32, addr: u64) -> Option<A64Inst> {
    let sf = bit(w, 31);
    let rd = bits(w, 4, 0);
    let rn = bits(w, 9, 5);
    match bits(w, 25, 23) {
        0b000 | 0b001 => {
            // PC-relative addressing.
            let imm = sext((bits(w, 23, 5) << 2) | bits(w, 30, 29), 21);
            if bit(w, 31) {
                let page = (addr & !0xfff).wrapping_add((imm << 12) as u64);
                ins("adrp", vec![r(Reg::x(rd)), Operand::Target(page)])
            } else {
                ins("adr", vec![r(Reg::x(rd)), Operand::Target(addr.wrapping_add(imm as u64))])
            }
        }
        0b010 => {
            // Add/subtract (immediate).
            let (op, s) = (bit(w, 30), bit(w, 29));
            let sh = bit(w, 22);
            let imm = i64::from(bits(w, 21, 10));
            let shift = if sh { vec![Operand::Modifier("lsl", Some(12))] } else { vec![] };
            if s && rd == 31 {
                let mut ops = vec![r(Reg::gp_sp(rn, sf)), Operand::Imm(imm)];
                ops.extend(shift);
                return ins(if op { "cmp" } else { "cmn" }, ops);
            }
            if !op && !s && !sh && imm == 0 && (rd == 31 || rn == 31) {
                return ins("mov", vec![r(Reg::gp_sp(rd, sf)), r(Reg::gp_sp(rn, sf))]);
            }
            let m = match (op, s) {
                (false, false) => "add",
                (false, true) => "adds",
                (true, false) => "sub",
                (true, true) => "subs",
            };
            let dst = if s { Reg::gp(rd, sf) } else { Reg::gp_sp(rd, sf) };
            let mut ops = vec![r(dst), r(Reg::gp_sp(rn, sf)), Operand::Imm(imm)];
            ops.extend(shift);
            ins(m, ops)
        }
        0b100 => {
            // Logical (immediate).
            let opc = bits(w, 30, 29);
            let n = bits(w, 22, 22);
            if !sf && n == 1 {
                return None;
            }
            let (immr, imms) = (bits(w, 21, 16), bits(w, 15, 10));
            let imm = decode_bit_masks(n, imms, immr, if sf { 64 } else { 32 })?;
            match opc {
                0b11 if rd == 31 => ins("tst", vec![r(Reg::gp(rn, sf)), Operand::UImm(imm)]),
                0b01 if rn == 31 && !move_wide_preferred(sf, n, imms, immr) => {
                    ins("mov", vec![r(Reg::gp_sp(rd, sf)), Operand::Imm(signed_of(imm, sf))])
                }
                _ => {
                    let m = ["and", "orr", "eor", "ands"][opc as usize];
                    let dst = if opc == 0b11 { Reg::gp(rd, sf) } else { Reg::gp_sp(rd, sf) };
                    ins(m, vec![r(dst), r(Reg::gp(rn, sf)), Operand::UImm(imm)])
                }
            }
        }
        0b101 => {
            // Move wide (immediate).
            let opc = bits(w, 30, 29);
            let hw = bits(w, 22, 21);
            if !sf && hw >= 2 {
                return None;
            }
            let imm16 = u64::from(bits(w, 20, 5));
            let shift = 16 * hw;
            let mask = if sf { u64::MAX } else { 0xffff_ffff };
            match opc {
                0b00 | 0b10 => {
                    let movn = opc == 0;
                    let alias = !(imm16 == 0 && hw != 0) && !(movn && !sf && imm16 == 0xffff);
                    if alias {
                        let v = imm16 << shift;
                        let v = if movn { !v & mask } else { v };
                        return ins("mov", vec![r(Reg::gp(rd, sf)), Operand::Imm(signed_of(v, sf))]);
                    }
                    let mut ops = vec![r(Reg::gp(rd, sf)), Operand::Imm(imm16 as i64)];
                    if hw != 0 {
                        ops.push(Operand::Modifier("lsl", Some(shift)));
                    }
                    ins(if movn { "movn" } else { "movz" }, ops)
                }
                0b11 => {
                    let mut ops = vec![r(Reg::gp(rd, sf)), Operand::Imm(imm16 as i64)];
                    if hw != 0 {
                        ops.push(Operand::Modifier("lsl", Some(shift)));
                    }
                    ins("movk", ops)
                }
                _ => None,
            }
        }
        0b110 => bitfield(w),
        0b111 => {
            // Extract.
            if bits(w, 30, 29) != 0 || bits(w, 22, 22) != u32::from(sf) || bit(w, 21) {
                return None;
            }
            let rm = bits(w, 20, 16);
            let imms = bits(w, 15, 10);
            if !sf && imms >= 32 {
                return None;
            }
            if rn == rm {
                ins("ror", vec![r(Reg::gp(rd, sf)), r(Reg::gp(rn, sf)), Operand::Imm(i64::from(imms))])
            } else {
                ins(
                    "extr",
                    vec![r(Reg::gp(rd, sf)), r(Reg::gp(rn, sf)), r(Reg::gp(rm, sf)), Operand::Imm(i64::from(imms))],
                )
            }
        }
        _ => None,
    }
}

/// Bitfield (`SBFM`/`BFM`/`UBFM`) with its shift, extend and field aliases.
fn bitfield(w: u32) -> Option<A64Inst> {
    let sf = bit(w, 31);
    let opc = bits(w, 30, 29);
    if bits(w, 22, 22) != u32::from(sf) || opc == 0b11 {
        return None;
    }
    let (immr, imms) = (bits(w, 21, 16), bits(w, 15, 10));
    let width = if sf { 64 } else { 32 };
    if !sf && (immr >= 32 || imms >= 32) {
        return None;
    }
    let (rd, rn) = (bits(w, 4, 0), bits(w, 9, 5));
    let d = r(Reg::gp(rd, sf));
    let n = r(Reg::gp(rn, sf));
    let dec = |v: u32| Operand::Dec(i64::from(v));
    match opc {
        0b00 => {
            if imms == width - 1 {
                return ins("asr", vec![d, n, dec(immr)]);
            }
            if immr == 0 {
                let m = match imms {
                    7 => Some("sxtb"),
                    15 => Some("sxth"),
                    31 if sf => Some("sxtw"),
                    _ => None,
                };
                if let Some(m) = m {
                    return ins(m, vec![d, r(Reg::gp(rn, false))]);
                }
            }
            if imms < immr {
                return ins("sbfiz", vec![d, n, dec(width - immr), dec(imms + 1)]);
            }
            ins("sbfx", vec![d, n, dec(immr), dec(imms - immr + 1)])
        }
        0b01 => {
            if imms < immr {
                ins("bfi", vec![d, n, dec(width - immr), dec(imms + 1)])
            } else {
                ins("bfxil", vec![d, n, dec(immr), dec(imms - immr + 1)])
            }
        }
        _ => {
            if imms != width - 1 && imms + 1 == immr {
                return ins("lsl", vec![d, n, dec(width - 1 - imms)]);
            }
            if imms == width - 1 {
                return ins("lsr", vec![d, n, dec(immr)]);
            }
            if immr == 0 && !sf && (imms == 7 || imms == 15) {
                return ins(if imms == 7 { "uxtb" } else { "uxth" }, vec![d, n]);
            }
            if imms < immr {
                return ins("ubfiz", vec![d, n, dec(width - immr), dec(imms + 1)]);
            }
            ins("ubfx", vec![d, n, dec(immr), dec(imms - immr + 1)])
        }
    }
}

// ===========================================================================
// Branches, exception generation and system instructions
// ===========================================================================

fn branch_sys(w: u32, addr: u64) -> Option<A64Inst> {
    let rt = bits(w, 4, 0);
    if bits(w, 30, 26) == 0b00101 {
        let t = addr.wrapping_add((sext(bits(w, 25, 0), 26) * 4) as u64);
        return ins(if bit(w, 31) { "bl" } else { "b" }, vec![Operand::Target(t)]);
    }
    if bits(w, 31, 24) == 0b0101_0100 {
        if bit(w, 4) {
            return None;
        }
        let t = addr.wrapping_add((sext(bits(w, 23, 5), 19) * 4) as u64);
        return ins(format!("b.{}", CONDS[bits(w, 3, 0) as usize]), vec![Operand::Target(t)]);
    }
    if bits(w, 30, 25) == 0b011010 {
        let t = addr.wrapping_add((sext(bits(w, 23, 5), 19) * 4) as u64);
        let m = if bit(w, 24) { "cbnz" } else { "cbz" };
        return ins(m, vec![r(Reg::gp(rt, bit(w, 31))), Operand::Target(t)]);
    }
    if bits(w, 30, 25) == 0b011011 {
        let t = addr.wrapping_add((sext(bits(w, 18, 5), 14) * 4) as u64);
        let b = (bits(w, 31, 31) << 5) | bits(w, 23, 19);
        let m = if bit(w, 24) { "tbnz" } else { "tbz" };
        return ins(m, vec![r(Reg::gp(rt, bit(w, 31))), Operand::Imm(i64::from(b)), Operand::Target(t)]);
    }
    if bits(w, 31, 24) == 0b1101_0100 {
        return exception(w);
    }
    if bits(w, 31, 22) == 0b11_0101_0100 {
        return system(w);
    }
    if bits(w, 31, 25) == 0b110_1011 {
        // Unconditional branch (register).
        let (opc, op2, op3, op4) = (bits(w, 24, 21), bits(w, 20, 16), bits(w, 15, 10), bits(w, 4, 0));
        let rn = bits(w, 9, 5);
        if op2 != 0b11111 || op3 != 0 || op4 != 0 {
            return None;
        }
        return match opc {
            0b0000 => ins("br", vec![r(Reg::x(rn))]),
            0b0001 => ins("blr", vec![r(Reg::x(rn))]),
            0b0010 if rn == 30 => ins("ret", vec![]),
            0b0010 => ins("ret", vec![r(Reg::x(rn))]),
            0b0100 if rn == 31 => ins("eret", vec![]),
            0b0101 if rn == 31 => ins("drps", vec![]),
            _ => None,
        };
    }
    None
}

fn exception(w: u32) -> Option<A64Inst> {
    let (opc, imm16, op2, ll) = (bits(w, 23, 21), bits(w, 20, 5), bits(w, 4, 2), bits(w, 1, 0));
    if op2 != 0 {
        return None;
    }
    let m = match (opc, ll) {
        (0b000, 0b01) => "svc",
        (0b000, 0b10) => "hvc",
        (0b000, 0b11) => "smc",
        (0b001, 0b00) => "brk",
        (0b010, 0b00) => "hlt",
        (0b101, 0b01) => "dcps1",
        (0b101, 0b10) => "dcps2",
        (0b101, 0b11) => "dcps3",
        _ => return None,
    };
    if m.starts_with("dcps") && imm16 == 0 {
        return ins(m, vec![]);
    }
    let imm = if imm16 == 0 { Operand::Dec(0) } else { Operand::Imm(i64::from(imm16)) };
    ins(m, vec![imm])
}

/// The name of a barrier option (`CRm` of `dmb`/`dsb`).
fn barrier_option(crm: u32) -> Option<&'static str> {
    Some(match crm {
        1 => "oshld",
        2 => "oshst",
        3 => "osh",
        5 => "nshld",
        6 => "nshst",
        7 => "nsh",
        9 => "ishld",
        10 => "ishst",
        11 => "ish",
        13 => "ld",
        14 => "st",
        15 => "sy",
        _ => return None,
    })
}

/// The name of a system register `(op0, op1, CRn, CRm, op2)`, in the upper
/// case LLVM prints, or the generic `S<op0>_<op1>_C<n>_C<m>_<op2>` form.
fn sysreg_name(o0: u32, o1: u32, crn: u32, crm: u32, o2: u32) -> String {
    let name = match (o0, o1, crn, crm, o2) {
        (3, 3, 13, 0, 2) => "TPIDR_EL0",
        (3, 3, 13, 0, 3) => "TPIDRRO_EL0",
        (3, 0, 13, 0, 4) => "TPIDR_EL1",
        (3, 3, 4, 2, 0) => "NZCV",
        (3, 3, 4, 2, 1) => "DAIF",
        (3, 3, 4, 4, 0) => "FPCR",
        (3, 3, 4, 4, 1) => "FPSR",
        (3, 3, 14, 0, 0) => "CNTFRQ_EL0",
        (3, 3, 14, 0, 1) => "CNTPCT_EL0",
        (3, 3, 14, 0, 2) => "CNTVCT_EL0",
        (3, 3, 0, 0, 1) => "CTR_EL0",
        (3, 3, 0, 0, 7) => "DCZID_EL0",
        (3, 0, 0, 0, 0) => "MIDR_EL1",
        (3, 0, 0, 0, 5) => "MPIDR_EL1",
        (3, 0, 4, 1, 0) => "SP_EL0",
        (3, 0, 4, 2, 2) => "CurrentEL",
        (3, 0, 4, 0, 0) => "SPSR_EL1",
        (3, 0, 4, 0, 1) => "ELR_EL1",
        (3, 0, 1, 0, 0) => "SCTLR_EL1",
        (3, 0, 5, 2, 0) => "ESR_EL1",
        (3, 0, 6, 0, 0) => "FAR_EL1",
        (3, 0, 12, 0, 0) => "VBAR_EL1",
        (3, 0, 2, 0, 0) => "TTBR0_EL1",
        (3, 0, 2, 0, 1) => "TTBR1_EL1",
        (3, 0, 2, 0, 2) => "TCR_EL1",
        (3, 0, 10, 2, 0) => "MAIR_EL1",
        (3, 0, 0, 4, 0) => "ID_AA64PFR0_EL1",
        (3, 0, 0, 6, 0) => "ID_AA64ISAR0_EL1",
        (3, 0, 0, 7, 0) => "ID_AA64MMFR0_EL1",
        (3, 3, 2, 4, 0) => "RNDR",
        (3, 3, 2, 4, 1) => "RNDRRS",
        _ => return format!("S{o0}_{o1}_C{crn}_C{crm}_{o2}"),
    };
    name.to_owned()
}

fn system(w: u32) -> Option<A64Inst> {
    let l = bit(w, 21);
    let (o0, o1, crn, crm, o2) = (bits(w, 20, 19), bits(w, 18, 16), bits(w, 15, 12), bits(w, 11, 8), bits(w, 7, 5));
    let rt = bits(w, 4, 0);
    if o0 >= 2 {
        let name = Operand::Text(sysreg_name(o0, o1, crn, crm, o2));
        return if l { ins("mrs", vec![r(Reg::x(rt)), name]) } else { ins("msr", vec![name, r(Reg::x(rt))]) };
    }
    if l {
        if o0 == 1 {
            return ins(
                "sysl",
                vec![r(Reg::x(rt)), Operand::Imm(i64::from(o1)), Operand::Text(format!("c{crn}")), Operand::Text(format!("c{crm}")), Operand::Imm(i64::from(o2))],
            );
        }
        return None;
    }
    if o0 == 1 {
        let alias = match (o1, crn, crm, o2) {
            (3, 7, 4, 1) => Some(("dc", "zva")),
            (0, 7, 6, 1) => Some(("dc", "ivac")),
            (0, 7, 6, 2) => Some(("dc", "isw")),
            (3, 7, 10, 1) => Some(("dc", "cvac")),
            (0, 7, 10, 2) => Some(("dc", "csw")),
            (3, 7, 11, 1) => Some(("dc", "cvau")),
            (3, 7, 14, 1) => Some(("dc", "civac")),
            (0, 7, 14, 2) => Some(("dc", "cisw")),
            (3, 7, 12, 1) => Some(("dc", "cvap")),
            (0, 7, 5, 0) => Some(("ic", "iallu")),
            (0, 7, 1, 0) => Some(("ic", "ialluis")),
            (3, 7, 5, 1) => Some(("ic", "ivau")),
            _ => None,
        };
        if let Some((m, op)) = alias {
            let mut ops = vec![Operand::Text(op.to_owned())];
            if rt != 31 || m == "dc" || op == "ivau" {
                ops.push(r(Reg::x(rt)));
            }
            return ins(m, ops);
        }
        let mut ops =
            vec![Operand::Imm(i64::from(o1)), Operand::Text(format!("c{crn}")), Operand::Text(format!("c{crm}")), Operand::Imm(i64::from(o2))];
        if rt != 31 {
            ops.push(r(Reg::x(rt)));
        }
        return ins("sys", ops);
    }
    // op0 == 0: hints, barriers, PSTATE.
    if crn == 0b0010 && o1 == 0b011 && rt == 31 {
        let imm = (crm << 3) | o2;
        let m = match imm {
            0 => "nop",
            1 => "yield",
            2 => "wfe",
            3 => "wfi",
            4 => "sev",
            5 => "sevl",
            7 => "xpaclri",
            8 => "pacia1716",
            10 => "pacib1716",
            12 => "autia1716",
            14 => "autib1716",
            16 => "esb",
            20 => "csdb",
            24 => "paciaz",
            25 => "paciasp",
            26 => "pacibz",
            27 => "pacibsp",
            28 => "autiaz",
            29 => "autiasp",
            30 => "autibz",
            31 => "autibsp",
            32 => "bti r",
            34 => "bti c",
            36 => "bti j",
            38 => "bti jc",
            _ => return ins("hint", vec![Operand::Imm(i64::from(imm))]),
        };
        // `bti c` and friends carry their target as an operand.
        if let Some((m, op)) = m.split_once(' ') {
            return ins(m, vec![Operand::Text(op.to_owned())]);
        }
        return ins(m, vec![]);
    }
    if crn == 0b0011 && o1 == 0b011 && rt == 31 {
        return match o2 {
            0b010 if crm == 15 => ins("clrex", vec![]),
            0b010 => ins("clrex", vec![Operand::Dec(i64::from(crm))]),
            0b100 if crm == 0 => ins("ssbb", vec![]),
            0b100 if crm == 4 => ins("pssbb", vec![]),
            0b100 | 0b101 => {
                let m = if o2 == 0b100 { "dsb" } else { "dmb" };
                let opt = barrier_option(crm).map_or(Operand::Dec(i64::from(crm)), |s| Operand::Text(s.to_owned()));
                ins(m, vec![opt])
            }
            0b110 if crm == 15 => ins("isb", vec![]),
            0b110 => ins("isb", vec![Operand::Dec(i64::from(crm))]),
            0b111 if crm == 0 => ins("sb", vec![]),
            _ => None,
        };
    }
    if crn == 0b0100 && rt == 31 {
        let field = match (o1, o2) {
            (0b011, 0b110) => "DAIFSet",
            (0b011, 0b111) => "DAIFClr",
            (0b000, 0b101) => "SPSel",
            _ => return None,
        };
        return ins("msr", vec![Operand::Text(field.to_owned()), Operand::Imm(i64::from(crm))]);
    }
    None
}

// ===========================================================================
// Data processing — register
// ===========================================================================

const SHIFTS: [&str; 4] = ["lsl", "lsr", "asr", "ror"];
const EXTENDS: [&str; 8] = ["uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx"];

fn shift_op(ty: u32, amount: u32) -> Option<Operand> {
    (ty != 0 || amount != 0).then(|| Operand::Modifier(SHIFTS[ty as usize], Some(amount)))
}

fn dp_reg(w: u32) -> Option<A64Inst> {
    let sf = bit(w, 31);
    let (rd, rn, rm) = (bits(w, 4, 0), bits(w, 9, 5), bits(w, 20, 16));
    let d = r(Reg::gp(rd, sf));
    let n = r(Reg::gp(rn, sf));
    let m = r(Reg::gp(rm, sf));
    if !bit(w, 28) {
        if !bit(w, 24) {
            // Logical (shifted register).
            let (opc, ty, nbit, imm6) = (bits(w, 30, 29), bits(w, 23, 22), bit(w, 21), bits(w, 15, 10));
            if !sf && imm6 >= 32 {
                return None;
            }
            let sh = shift_op(ty, imm6);
            let mut tail = vec![m.clone()];
            tail.extend(sh.clone());
            if opc == 0b01 && rn == 31 {
                if !nbit && sh.is_none() {
                    return ins("mov", vec![d, m]);
                }
                if nbit {
                    let mut ops = vec![d];
                    ops.extend(tail);
                    return ins("mvn", ops);
                }
            }
            if opc == 0b11 && rd == 31 && !nbit {
                let mut ops = vec![n];
                ops.extend(tail);
                return ins("tst", ops);
            }
            let name = ["and", "bic", "orr", "orn", "eor", "eon", "ands", "bics"][(opc * 2 + u32::from(nbit)) as usize];
            let mut ops = vec![d, n];
            ops.extend(tail);
            return ins(name, ops);
        }
        let (op, s) = (bit(w, 30), bit(w, 29));
        if !bit(w, 21) {
            // Add/subtract (shifted register).
            let (ty, imm6) = (bits(w, 23, 22), bits(w, 15, 10));
            if ty == 3 || (!sf && imm6 >= 32) {
                return None;
            }
            let mut tail = vec![m];
            tail.extend(shift_op(ty, imm6));
            if s && rd == 31 {
                let mut ops = vec![n];
                ops.extend(tail);
                return ins(if op { "cmp" } else { "cmn" }, ops);
            }
            if op && rn == 31 {
                let mut ops = vec![d];
                ops.extend(tail);
                return ins(if s { "negs" } else { "neg" }, ops);
            }
            let name = match (op, s) {
                (false, false) => "add",
                (false, true) => "adds",
                (true, false) => "sub",
                (true, true) => "subs",
            };
            let mut ops = vec![d, n];
            ops.extend(tail);
            return ins(name, ops);
        }
        // Add/subtract (extended register).
        if bits(w, 23, 22) != 0 {
            return None;
        }
        let (option, imm3) = (bits(w, 15, 13), bits(w, 12, 10));
        if imm3 > 4 {
            return None;
        }
        // The 32-bit forms always name a `W` index register.
    let rm_wide = sf && option & 0b011 == 0b011;
        let mreg = r(Reg::gp(rm, rm_wide));
        let uses_sp = (rd == 31 && !s) || rn == 31;
        let lsl_form = if sf { 0b011 } else { 0b010 };
        let ext = if uses_sp && option == lsl_form {
            (imm3 != 0).then_some(Operand::Modifier("lsl", Some(imm3)))
        } else {
            Some(Operand::Modifier(EXTENDS[option as usize], (imm3 != 0).then_some(imm3)))
        };
        let nsp = r(Reg::gp_sp(rn, sf));
        if s && rd == 31 {
            let mut ops = vec![nsp, mreg];
            ops.extend(ext);
            return ins(if op { "cmp" } else { "cmn" }, ops);
        }
        let name = match (op, s) {
            (false, false) => "add",
            (false, true) => "adds",
            (true, false) => "sub",
            (true, true) => "subs",
        };
        let dst = if s { Reg::gp(rd, sf) } else { Reg::gp_sp(rd, sf) };
        let mut ops = vec![r(dst), nsp, mreg];
        ops.extend(ext);
        return ins(name, ops);
    }
    let op2 = bits(w, 24, 21);
    match op2 {
        0b0000 => {
            if bits(w, 15, 10) != 0 {
                return None;
            }
            let (op, s) = (bit(w, 30), bit(w, 29));
            if op && rn == 31 {
                return ins(if s { "ngcs" } else { "ngc" }, vec![d, m]);
            }
            let name = ["adc", "adcs", "sbc", "sbcs"][(u32::from(op) * 2 + u32::from(s)) as usize];
            ins(name, vec![d, n, m])
        }
        0b0010 => {
            // Conditional compare (register / immediate).
            if !bit(w, 29) || bit(w, 10) || bit(w, 4) {
                return None;
            }
            let name = if bit(w, 30) { "ccmp" } else { "ccmn" };
            let second = if bit(w, 11) { Operand::Imm(i64::from(rm)) } else { m };
            let nzcv = Operand::Imm(i64::from(bits(w, 3, 0)));
            ins(name, vec![n, second, nzcv, Operand::Cond(bits(w, 15, 12) as u8)])
        }
        0b0100 => {
            // Conditional select.
            if bit(w, 29) || bit(w, 11) {
                return None;
            }
            let (op, o2) = (bit(w, 30), bit(w, 10));
            let cond = bits(w, 15, 12);
            let inv = Operand::Cond((cond ^ 1) as u8);
            let al = cond >= 14;
            match (op, o2) {
                (false, false) => ins("csel", vec![d, n, m, Operand::Cond(cond as u8)]),
                (false, true) if !al && rn == 31 && rm == 31 => ins("cset", vec![d, inv]),
                (false, true) if !al && rn == rm => ins("cinc", vec![d, n, inv]),
                (false, true) => ins("csinc", vec![d, n, m, Operand::Cond(cond as u8)]),
                (true, false) if !al && rn == 31 && rm == 31 => ins("csetm", vec![d, inv]),
                (true, false) if !al && rn == rm => ins("cinv", vec![d, n, inv]),
                (true, false) => ins("csinv", vec![d, n, m, Operand::Cond(cond as u8)]),
                (true, true) if !al && rn == rm => ins("cneg", vec![d, n, inv]),
                (true, true) => ins("csneg", vec![d, n, m, Operand::Cond(cond as u8)]),
            }
        }
        0b0110 => {
            if bit(w, 29) {
                return None;
            }
            let opcode = bits(w, 15, 10);
            if bit(w, 30) {
                // Data processing (1 source).
                if rm != 0 {
                    return None;
                }
                let name = match (opcode, sf) {
                    (0b000000, _) => "rbit",
                    (0b000001, _) => "rev16",
                    (0b000010, false) => "rev",
                    (0b000010, true) => "rev32",
                    (0b000011, true) => "rev",
                    (0b000100, _) => "clz",
                    (0b000101, _) => "cls",
                    _ => return None,
                };
                return ins(name, vec![d, n]);
            }
            // Data processing (2 source).
            let name = match opcode {
                0b000010 => "udiv",
                0b000011 => "sdiv",
                0b001000 => "lsl",
                0b001001 => "lsr",
                0b001010 => "asr",
                0b001011 => "ror",
                0b010000..=0b010111 => {
                    let sz = opcode & 3;
                    if (sz == 3) != sf {
                        return None;
                    }
                    let c = if opcode & 4 != 0 { "c" } else { "" };
                    let name = format!("crc32{c}{}", ['b', 'h', 'w', 'x'][sz as usize]);
                    return ins(name, vec![r(Reg::gp(rd, false)), r(Reg::gp(rn, false)), r(Reg::gp(rm, sz == 3))]);
                }
                _ => return None,
            };
            ins(name, vec![d, n, m])
        }
        0b1000..=0b1111 => {
            // Data processing (3 source).
            if bits(w, 30, 29) != 0 {
                return None;
            }
            let (op31, o0, ra) = (bits(w, 23, 21), bit(w, 15), bits(w, 14, 10));
            let a = r(Reg::gp(ra, sf));
            match op31 {
                0b000 => {
                    if ra == 31 {
                        ins(if o0 { "mneg" } else { "mul" }, vec![d, n, m])
                    } else {
                        ins(if o0 { "msub" } else { "madd" }, vec![d, n, m, a])
                    }
                }
                0b001 | 0b101 if sf => {
                    let u = op31 == 0b101;
                    let (wn, wm) = (r(Reg::gp(rn, false)), r(Reg::gp(rm, false)));
                    let p = if u { "u" } else { "s" };
                    if ra == 31 {
                        ins(format!("{p}{}", if o0 { "mnegl" } else { "mull" }), vec![d, wn, wm])
                    } else {
                        ins(format!("{p}{}", if o0 { "msubl" } else { "maddl" }), vec![d, wn, wm, a])
                    }
                }
                0b010 | 0b110 if sf && !o0 => ins(if op31 == 0b010 { "smulh" } else { "umulh" }, vec![d, n, m]),
                _ => None,
            }
        }
        _ => None,
    }
}

// ===========================================================================
// Loads and stores
// ===========================================================================

/// The `prfm` operation name of `Rt`.
fn prfop(rt: u32) -> Operand {
    let (ty, target, policy) = (rt >> 3, (rt >> 1) & 3, rt & 1);
    if ty == 3 {
        return Operand::Imm(i64::from(rt));
    }
    Operand::Text(format!(
        "{}{}{}",
        ["pld", "pli", "pst"][ty as usize],
        ["l1", "l2", "l3", "slc"][target as usize],
        if policy == 0 { "keep" } else { "strm" }
    ))
}

fn mem(base: u32, index: Index) -> Operand {
    Operand::Mem(Mem { base: base as u8, index })
}

fn ldst(w: u32, addr: u64) -> Option<A64Inst> {
    let rt = bits(w, 4, 0);
    let rn = bits(w, 9, 5);
    let v = bit(w, 26);
    let size = bits(w, 31, 30);
    // Advanced SIMD structure loads/stores.
    if bits(w, 31, 31) == 0 && bits(w, 29, 25) == 0b00110 && v {
        return simd_ldst(w);
    }
    match bits(w, 29, 27) {
        0b001 if !v && !bit(w, 24) => exclusive(w),
        0b011 => {
            if bits(w, 25, 24) == 0b01 && !v && !bit(w, 21) && bits(w, 11, 10) == 0 {
                return rcpc_unscaled(w);
            }
            if bits(w, 25, 24) != 0 {
                return None;
            }
            // Load register (literal).
            let t = addr.wrapping_add((sext(bits(w, 23, 5), 19) * 4) as u64);
            let opc = bits(w, 31, 30);
            let (m, reg) = if v {
                let s = match opc {
                    0 => 's',
                    1 => 'd',
                    2 => 'q',
                    _ => return None,
                };
                ("ldr", r(Reg::fp(rt, s)))
            } else {
                match opc {
                    0 => ("ldr", r(Reg::gp(rt, false))),
                    1 => ("ldr", r(Reg::x(rt))),
                    2 => ("ldrsw", r(Reg::x(rt))),
                    _ => ("prfm", prfop(rt)),
                }
            };
            ins(m, vec![reg, Operand::Target(t)])
        }
        0b101 => pair(w),
        0b111 => {
            if !bit(w, 24) && bit(w, 21) && bits(w, 11, 10) == 0 && !v {
                return atomic(w);
            }
            let opc = bits(w, 23, 22);
            // Name, register and scale of the access.
            let (base_name, reg, scale) = ldst_kind(size, v, opc, rt)?;
            if bit(w, 24) {
                // Unsigned immediate offset.
                let off = i64::from(bits(w, 21, 10)) << scale;
                return ins(base_name.0, vec![reg, mem(rn, Index::Offset(off))]);
            }
            if !bit(w, 21) {
                let imm9 = sext(bits(w, 20, 12), 9);
                return match bits(w, 11, 10) {
                    0b00 => {
                        // Unscaled (`ldur`): `prfum`, else ld/st + u + rest.
                        let name = if base_name.0 == "prfm" { "prfum".to_owned() } else { base_name.unscaled() };
                        ins(name, vec![reg, mem(rn, Index::Offset(imm9))])
                    }
                    0b01 if base_name.0 != "prfm" => ins(base_name.0, vec![reg, mem(rn, Index::Post(imm9))]),
                    0b11 if base_name.0 != "prfm" => ins(base_name.0, vec![reg, mem(rn, Index::Pre(imm9))]),
                    0b10 if !v && base_name.0 != "prfm" => ins(base_name.unprivileged(), vec![reg, mem(rn, Index::Offset(imm9))]),
                    _ => None,
                };
            }
            match bits(w, 11, 10) {
                0b10 => {
                    // Register offset.
                    let option = bits(w, 15, 13);
                    if option & 0b010 == 0 {
                        return None;
                    }
                    let s = bit(w, 12);
                    let rm = bits(w, 20, 16);
                    let wide = option & 1 == 1;
                    let (ext, amount) = if option == 0b011 {
                        if s { (Some("lsl"), Some(scale)) } else { (None, None) }
                    } else {
                        (Some(EXTENDS[option as usize]), s.then_some(scale))
                    };
                    ins(base_name.0, vec![reg, mem(rn, Index::Reg { rm: rm as u8, wide, ext, amount })])
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// The RCpc unscaled forms `stlur`/`ldapur` (and their sign-extending
/// variants).
fn rcpc_unscaled(w: u32) -> Option<A64Inst> {
    let size = bits(w, 31, 30);
    let opc = bits(w, 23, 22);
    let (rn, rt) = (bits(w, 9, 5), bits(w, 4, 0));
    let suffix = ["b", "h", "", ""][size as usize];
    let (m, wide) = match opc {
        0 => (format!("stlur{suffix}"), size == 3),
        1 => (format!("ldapur{suffix}"), size == 3),
        2 if size == 2 => ("ldapursw".to_owned(), true),
        2 if size < 2 => (format!("ldapurs{suffix}"), true),
        3 if size < 2 => (format!("ldapurs{suffix}"), false),
        _ => return None,
    };
    ins(m, vec![r(Reg::gp(rt, wide)), mem(rn, Index::Offset(sext(bits(w, 20, 12), 9)))])
}

/// A load/store name with the pieces its unscaled and unprivileged forms
/// are spelled from (`ldrsb` → `ldursb`, `ldtrsb`).
#[derive(Clone, Copy)]
struct LdstName(&'static str);

impl LdstName {
    fn unscaled(self) -> String {
        format!("{}u{}", &self.0[..2], &self.0[2..])
    }
    fn unprivileged(self) -> String {
        format!("{}t{}", &self.0[..2], &self.0[2..])
    }
}

/// The mnemonic, transfer register and log2 scale of a single-register
/// load/store with fields `size`, `V`, `opc`.
fn ldst_kind(size: u32, v: bool, opc: u32, rt: u32) -> Option<(LdstName, Operand, u32)> {
    if v {
        let (sz, scale) = match (size, opc) {
            (0, 0 | 1) => ('b', 0),
            (0, 2 | 3) => ('q', 4),
            (1, 0 | 1) => ('h', 1),
            (2, 0 | 1) => ('s', 2),
            (3, 0 | 1) => ('d', 3),
            _ => return None,
        };
        let name = if opc & 1 == 1 { "ldr" } else { "str" };
        return Some((LdstName(name), r(Reg::fp(rt, sz)), scale));
    }
    let suffix = ["b", "h", "", ""][size as usize];
    let (name, wide): (String, bool) = match opc {
        0 => (format!("str{suffix}"), size == 3),
        1 => (format!("ldr{suffix}"), size == 3),
        2 if size == 3 => return Some((LdstName("prfm"), prfop(rt), 3)),
        2 if size == 2 => ("ldrsw".to_owned(), true),
        2 => (format!("ldrs{suffix}"), true),
        3 if size < 2 => (format!("ldrs{suffix}"), false),
        _ => return None,
    };
    let name: &'static str = match name.as_str() {
        "strb" => "strb",
        "strh" => "strh",
        "str" => "str",
        "ldrb" => "ldrb",
        "ldrh" => "ldrh",
        "ldr" => "ldr",
        "ldrsw" => "ldrsw",
        "ldrsb" => "ldrsb",
        "ldrsh" => "ldrsh",
        _ => return None,
    };
    Some((LdstName(name), r(Reg::gp(rt, wide)), size))
}

/// Load/store register pair (offset, pre-, post-index, no-allocate).
fn pair(w: u32) -> Option<A64Inst> {
    let (opc, v, ty, l) = (bits(w, 31, 30), bit(w, 26), bits(w, 24, 23), bit(w, 22));
    let (rt, rt2, rn) = (bits(w, 4, 0), bits(w, 14, 10), bits(w, 9, 5));
    let imm7 = sext(bits(w, 21, 15), 7);
    let (mk, scale, name): (fn(u32) -> Reg, u32, &str) = if v {
        match opc {
            0 => (|n| Reg::fp(n, 's'), 2, ""),
            1 => (|n| Reg::fp(n, 'd'), 3, ""),
            2 => (|n| Reg::fp(n, 'q'), 4, ""),
            _ => return None,
        }
    } else {
        match (opc, l) {
            (0, _) => (|n| Reg::gp(n, false), 2, ""),
            (1, true) if ty != 0 => (|n| Reg::x(n), 2, "ldpsw"),
            (2, _) => (|n| Reg::x(n), 3, ""),
            _ => return None,
        }
    };
    let off = imm7 << scale;
    let index = match ty {
        0b00 | 0b10 => Index::Offset(off),
        0b01 => Index::Post(off),
        _ => Index::Pre(off),
    };
    let m = if !name.is_empty() {
        name
    } else {
        match (ty == 0, l) {
            (true, true) => "ldnp",
            (true, false) => "stnp",
            (false, true) => "ldp",
            (false, false) => "stp",
        }
    };
    ins(m, vec![r(mk(rt)), r(mk(rt2)), mem(rn, index)])
}

/// Load/store exclusive, acquire/release and compare-and-swap.
fn exclusive(w: u32) -> Option<A64Inst> {
    let size = bits(w, 31, 30);
    let (o2, l, o1, o0) = (bit(w, 23), bit(w, 22), bit(w, 21), bit(w, 15));
    let (rs, rt2, rn, rt) = (bits(w, 20, 16), bits(w, 14, 10), bits(w, 9, 5), bits(w, 4, 0));
    let suffix = ["b", "h", "", ""][size as usize];
    let wide = size == 3;
    let base = mem(rn, Index::Offset(0));
    match (o2, o1) {
        (false, false) => {
            // Rs and Rt2 are should-be-one fields here (ignored, as by
            // hardware).
            if l {
                let m = format!("ld{}xr{suffix}", if o0 { "a" } else { "" });
                ins(m, vec![r(Reg::gp(rt, wide)), base])
            } else {
                let m = format!("st{}xr{suffix}", if o0 { "l" } else { "" });
                ins(m, vec![r(Reg::gp(rs, false)), r(Reg::gp(rt, wide)), base])
            }
        }
        (false, true) if size >= 2 => {
            if l {
                let m = format!("ld{}xp", if o0 { "a" } else { "" });
                ins(m, vec![r(Reg::gp(rt, wide)), r(Reg::gp(rt2, wide)), base])
            } else {
                let m = format!("st{}xp", if o0 { "l" } else { "" });
                ins(m, vec![r(Reg::gp(rs, false)), r(Reg::gp(rt, wide)), r(Reg::gp(rt2, wide)), base])
            }
        }
        (false, true) => {
            // CASP: register pairs, even-numbered.
            if rt2 != 31 || rs % 2 != 0 || rt % 2 != 0 {
                return None;
            }
            let pw = size == 1;
            let m = format!("casp{}{}", if l { "a" } else { "" }, if o0 { "l" } else { "" });
            ins(
                m,
                vec![r(Reg::gp(rs, pw)), r(Reg::gp(rs + 1, pw)), r(Reg::gp(rt, pw)), r(Reg::gp(rt + 1, pw)), base],
            )
        }
        (true, false) => {
            let m = match (l, o0) {
                (false, true) => format!("stlr{suffix}"),
                (false, false) => format!("stllr{suffix}"),
                (true, true) => format!("ldar{suffix}"),
                (true, false) => format!("ldlar{suffix}"),
            };
            ins(m, vec![r(Reg::gp(rt, wide)), base])
        }
        (true, true) => {
            if rt2 != 31 {
                return None;
            }
            let m = format!("cas{}{}{suffix}", if l { "a" } else { "" }, if o0 { "l" } else { "" });
            ins(m, vec![r(Reg::gp(rs, wide)), r(Reg::gp(rt, wide)), base])
        }
    }
}

/// LSE atomic memory operations (`ldadd`, `swp`, `ldapr`, the `st<op>`
/// aliases).
fn atomic(w: u32) -> Option<A64Inst> {
    let size = bits(w, 31, 30);
    let (a, rr) = (bit(w, 23), bit(w, 22));
    let (rs, o3, opc, rn, rt) = (bits(w, 20, 16), bit(w, 15), bits(w, 14, 12), bits(w, 9, 5), bits(w, 4, 0));
    let suffix = ["b", "h", "", ""][size as usize];
    let wide = size == 3;
    let order = format!("{}{}", if a { "a" } else { "" }, if rr { "l" } else { "" });
    let base = mem(rn, Index::Offset(0));
    if o3 {
        return match opc {
            0b000 => ins(format!("swp{order}{suffix}"), vec![r(Reg::gp(rs, wide)), r(Reg::gp(rt, wide)), base]),
            0b100 if a && !rr && rs == 31 => ins(format!("ldapr{suffix}"), vec![r(Reg::gp(rt, wide)), base]),
            _ => None,
        };
    }
    let op = ["add", "clr", "eor", "set", "smax", "smin", "umax", "umin"][opc as usize];
    if rt == 31 && !a {
        let l = if rr { "l" } else { "" };
        return ins(format!("st{op}{l}{suffix}"), vec![r(Reg::gp(rs, wide)), base]);
    }
    ins(format!("ld{op}{order}{suffix}"), vec![r(Reg::gp(rs, wide)), r(Reg::gp(rt, wide)), base])
}

/// Advanced SIMD load/store multiple and single structures.
fn simd_ldst(w: u32) -> Option<A64Inst> {
    let q = bit(w, 30);
    let l = bit(w, 22);
    let (rm, rn, rt) = (bits(w, 20, 16), bits(w, 9, 5), bits(w, 4, 0));
    let post = bit(w, 23);
    let size = bits(w, 11, 10);
    let lo = if l { "ld" } else { "st" };
    if !bit(w, 24) {
        // Multiple structures.
        if bit(w, 21) || (!post && rm != 0) {
            return None;
        }
        let (name, regs) = match bits(w, 15, 12) {
            0b0000 => ("4", 4),
            0b0010 => ("1", 4),
            0b0100 => ("3", 3),
            0b0110 => ("1", 3),
            0b0111 => ("1", 1),
            0b1000 => ("2", 2),
            0b1010 => ("1", 2),
            _ => return None,
        };
        if size == 3 && !q && name != "1" {
            return None;
        }
        let arr = arrangement(size, q)?;
        let list: Vec<Reg> = (0..regs).map(|k| Reg::v((rt + k) % 32, arr)).collect();
        let bytes = i64::from(regs) * if q { 16 } else { 8 };
        let index = if !post {
            Index::Offset(0)
        } else if rm == 31 {
            Index::PostDec(bytes)
        } else {
            Index::PostReg(rm as u8)
        };
        return ins(format!("{lo}{name}"), vec![Operand::List(list, None), mem(rn, index)]);
    }
    // Single structure (only the one-register forms).
    if bit(w, 21) || (!post && rm != 0) {
        return None;
    }
    let opcode = bits(w, 15, 13);
    let s = bit(w, 12);
    if opcode == 0b110 {
        // ld1r.
        if !l || s {
            return None;
        }
        let arr = arrangement(size, q)?;
        let bytes = 1i64 << size;
        let index = if !post {
            Index::Offset(0)
        } else if rm == 31 {
            Index::PostDec(bytes)
        } else {
            Index::PostReg(rm as u8)
        };
        return ins("ld1r", vec![Operand::List(vec![Reg::v(rt, arr)], None), mem(rn, index)]);
    }
    let qi = u32::from(q);
    let si = u32::from(s);
    let (esz, index, bytes) = match opcode {
        0b000 => ('b', (qi << 3) | (si << 2) | size, 1),
        0b010 if size & 1 == 0 => ('h', (qi << 2) | (si << 1) | (size >> 1), 2),
        0b100 if size == 0 => ('s', (qi << 1) | si, 4),
        0b100 if size == 1 && !s => ('d', qi, 8),
        _ => return None,
    };
    let idx = match (post, rm) {
        (false, _) => Index::Offset(0),
        (true, 31) => Index::PostDec(bytes),
        (true, m) => Index::PostReg(m as u8),
    };
    let arr = match esz {
        'b' => "b",
        'h' => "h",
        's' => "s",
        _ => "d",
    };
    ins(format!("{lo}1"), vec![Operand::List(vec![Reg::v(rt, arr)], Some(index as u8)), mem(rn, idx)])
}

// ===========================================================================
// SIMD and floating point
// ===========================================================================

/// The vector arrangement for `size` and `Q` (`None` for the reserved `1d`
/// of three-same/two-misc integer forms is left to the caller).
fn arrangement(size: u32, q: bool) -> Option<&'static str> {
    Some(match (size, q) {
        (0, false) => "8b",
        (0, true) => "16b",
        (1, false) => "4h",
        (1, true) => "8h",
        (2, false) => "2s",
        (2, true) => "4s",
        (3, false) => "1d",
        (3, true) => "2d",
        _ => return None,
    })
}

/// The width letter of a scalar FP `ftype`.
fn ftype_size(ftype: u32) -> Option<char> {
    match ftype {
        0 => Some('s'),
        1 => Some('d'),
        3 => Some('h'),
        _ => None,
    }
}

/// The ARM ARM `VFPExpandImm` of an 8-bit floating-point immediate.
pub fn fp_imm(imm8: u32) -> f64 {
    let sign = if imm8 & 0x80 != 0 { -1.0 } else { 1.0 };
    let cd = ((imm8 >> 4) & 3) as i32;
    let exp = if imm8 & 0x40 == 0 { 1 + cd } else { cd - 3 };
    let frac = 1.0 + f64::from(imm8 & 0xf) / 16.0;
    sign * frac * 2f64.powi(exp)
}

fn simd_fp(w: u32) -> Option<A64Inst> {
    let b28_24 = bits(w, 28, 24);
    if b28_24 == 0b11110 && !bit(w, 30) {
        return fp_scalar(w);
    }
    if b28_24 == 0b11111 && bits(w, 31, 29) == 0 {
        // Floating-point data processing (3 source).
        let sz = ftype_size(bits(w, 23, 22))?;
        let (o1, o0) = (bit(w, 21), bit(w, 15));
        let m = match (o1, o0) {
            (false, false) => "fmadd",
            (false, true) => "fmsub",
            (true, false) => "fnmadd",
            (true, true) => "fnmsub",
        };
        let f = |n| r(Reg::fp(n, sz));
        return ins(m, vec![f(bits(w, 4, 0)), f(bits(w, 9, 5)), f(bits(w, 20, 16)), f(bits(w, 14, 10))]);
    }
    if bit(w, 31) {
        return None;
    }
    if b28_24 == 0b01110 {
        return simd_vector(w);
    }
    if b28_24 == 0b01111 && !bit(w, 10) {
        return by_element(w);
    }
    if b28_24 == 0b01111 && bit(w, 10) && !bit(w, 23) {
        return if bits(w, 22, 19) == 0 { simd_modified_imm(w) } else { simd_shift_imm(w, false) };
    }
    if b28_24 == 0b11110 && bit(w, 30) {
        return simd_scalar(w);
    }
    if b28_24 == 0b11111 && bit(w, 30) && bit(w, 10) && !bit(w, 23) && bits(w, 22, 19) != 0 {
        return simd_shift_imm(w, true);
    }
    None
}

/// Scalar floating-point instructions (`M = 0`, `S = 0`, bits 28:24 `11110`).
fn fp_scalar(w: u32) -> Option<A64Inst> {
    let sf = bit(w, 31);
    let ftype = bits(w, 23, 22);
    let (rd, rn, rm) = (bits(w, 4, 0), bits(w, 9, 5), bits(w, 20, 16));
    if bit(w, 29) {
        return None;
    }
    if !bit(w, 21) {
        // Conversion between floating-point and fixed-point.
        let sz = ftype_size(ftype)?;
        let scale = bits(w, 15, 10);
        if !sf && scale < 32 {
            return None;
        }
        let fbits = Operand::Imm(i64::from(64 - scale));
        let (rmode, opcode) = (bits(w, 20, 19), bits(w, 18, 16));
        return match (rmode, opcode) {
            (0b11, 0b000) => ins("fcvtzs", vec![r(Reg::gp(rd, sf)), r(Reg::fp(rn, sz)), fbits]),
            (0b11, 0b001) => ins("fcvtzu", vec![r(Reg::gp(rd, sf)), r(Reg::fp(rn, sz)), fbits]),
            (0b00, 0b010) => ins("scvtf", vec![r(Reg::fp(rd, sz)), r(Reg::gp(rn, sf)), fbits]),
            (0b00, 0b011) => ins("ucvtf", vec![r(Reg::fp(rd, sz)), r(Reg::gp(rn, sf)), fbits]),
            _ => None,
        };
    }
    if bit(w, 31) && bits(w, 15, 10) != 0 {
        return None;
    }
    if bits(w, 15, 10) == 0 {
        // Conversion between floating-point and integer.
        let (rmode, opcode) = (bits(w, 20, 19), bits(w, 18, 16));
        if ftype == 2 {
            // fmov Xd, Vn.d[1] / fmov Vd.d[1], Xn.
            if !sf || rmode != 1 {
                return None;
            }
            let elem = |n: u32| Reg::Elem { n: n as u8, size: 'd', index: 1 };
            return match opcode {
                0b110 => ins("fmov", vec![r(Reg::x(rd)), r(elem(rn))]),
                0b111 => ins("fmov", vec![r(elem(rd)), r(Reg::x(rn))]),
                _ => None,
            };
        }
        let sz = ftype_size(ftype)?;
        let to_int = |m: &str| ins(m, vec![r(Reg::gp(rd, sf)), r(Reg::fp(rn, sz))]);
        return match (rmode, opcode) {
            (0b00, 0b000) => to_int("fcvtns"),
            (0b00, 0b001) => to_int("fcvtnu"),
            (0b00, 0b010) => ins("scvtf", vec![r(Reg::fp(rd, sz)), r(Reg::gp(rn, sf))]),
            (0b00, 0b011) => ins("ucvtf", vec![r(Reg::fp(rd, sz)), r(Reg::gp(rn, sf))]),
            (0b00, 0b100) => to_int("fcvtas"),
            (0b00, 0b101) => to_int("fcvtau"),
            (0b00, 0b110) if sz == 'h' || (sz == 'd') == sf => to_int("fmov"),
            (0b00, 0b111) if sz == 'h' || (sz == 'd') == sf => ins("fmov", vec![r(Reg::fp(rd, sz)), r(Reg::gp(rn, sf))]),
            (0b01, 0b000) => to_int("fcvtps"),
            (0b01, 0b001) => to_int("fcvtpu"),
            (0b10, 0b000) => to_int("fcvtms"),
            (0b10, 0b001) => to_int("fcvtmu"),
            (0b11, 0b000) => to_int("fcvtzs"),
            (0b11, 0b001) => to_int("fcvtzu"),
            _ => None,
        };
    }
    if bit(w, 31) {
        return None;
    }
    let sz = ftype_size(ftype)?;
    let f = |n| r(Reg::fp(n, sz));
    if bits(w, 14, 10) == 0b10000 {
        // Data processing (1 source).
        let opcode = bits(w, 20, 15);
        let m = match opcode {
            0b000000 => "fmov",
            0b000001 => "fabs",
            0b000010 => "fneg",
            0b000011 => "fsqrt",
            0b000100 | 0b000101 | 0b000111 => {
                let to = ftype_size(opcode & 3)?;
                if to == sz {
                    return None;
                }
                return ins("fcvt", vec![r(Reg::fp(rd, to)), f(rn)]);
            }
            0b001000 => "frintn",
            0b001001 => "frintp",
            0b001010 => "frintm",
            0b001011 => "frintz",
            0b001100 => "frinta",
            0b001110 => "frintx",
            0b001111 => "frinti",
            _ => return None,
        };
        return ins(m, vec![f(rd), f(rn)]);
    }
    if bits(w, 13, 10) == 0b1000 {
        // Compare.
        if bits(w, 15, 14) != 0 || bits(w, 2, 0) != 0 {
            return None;
        }
        let opc2 = bits(w, 4, 3);
        let m = if opc2 & 2 != 0 { "fcmpe" } else { "fcmp" };
        let second = if opc2 & 1 != 0 {
            if rm != 0 {
                return None;
            }
            Operand::Text("#0.0".to_owned())
        } else {
            f(rm)
        };
        return ins(m, vec![f(rn), second]);
    }
    if bits(w, 12, 10) == 0b100 {
        // Immediate.
        if bits(w, 9, 5) != 0 {
            return None;
        }
        return ins("fmov", vec![f(rd), Operand::Float(fp_imm(bits(w, 20, 13)))]);
    }
    match bits(w, 11, 10) {
        0b01 => {
            let m = if bit(w, 4) { "fccmpe" } else { "fccmp" };
            ins(m, vec![f(rn), f(rm), Operand::Imm(i64::from(bits(w, 3, 0))), Operand::Cond(bits(w, 15, 12) as u8)])
        }
        0b10 => {
            let m = match bits(w, 15, 12) {
                0b0000 => "fmul",
                0b0001 => "fdiv",
                0b0010 => "fadd",
                0b0011 => "fsub",
                0b0100 => "fmax",
                0b0101 => "fmin",
                0b0110 => "fmaxnm",
                0b0111 => "fminnm",
                0b1000 => "fnmul",
                _ => return None,
            };
            ins(m, vec![f(rd), f(rn), f(rm)])
        }
        0b11 => ins("fcsel", vec![f(rd), f(rn), f(rm), Operand::Cond(bits(w, 15, 12) as u8)]),
        _ => None,
    }
}

/// The element size letter and index of a copy-instruction `imm5`.
fn imm5_elem(imm5: u32) -> Option<(char, u32, u32)> {
    let low = imm5.trailing_zeros();
    Some(match low {
        0 => ('b', imm5 >> 1, 0),
        1 => ('h', imm5 >> 2, 1),
        2 => ('s', imm5 >> 3, 2),
        3 => ('d', imm5 >> 4, 3),
        _ => return None,
    })
}

/// Vector instructions with bits 28:24 `01110` (and bit 31 clear).
fn simd_vector(w: u32) -> Option<A64Inst> {
    let q = bit(w, 30);
    let u = bit(w, 29);
    let size = bits(w, 23, 22);
    let (rd, rn, rm) = (bits(w, 4, 0), bits(w, 9, 5), bits(w, 20, 16));
    if bit(w, 21) {
        if bit(w, 10) {
            return three_same(w);
        }
        if bits(w, 11, 10) == 0b10 {
            match bits(w, 20, 17) {
                0b0000 => return two_misc(w),
                0b1000 => return across(w),
                _ => return None,
            }
        }
        return three_different(w);
    }
    // bit 21 == 0.
    if bits(w, 23, 21) == 0 && bit(w, 10) && !bit(w, 15) && (!u || q) {
        // Copy.
        let imm5 = bits(w, 20, 16);
        let imm4 = bits(w, 14, 11);
        let (esz, index, sz) = imm5_elem(imm5)?;
        let elem = |n: u32, i: u32| Reg::Elem { n: n as u8, size: esz, index: i as u8 };
        if u {
            let i2 = imm4 >> sz;
            return ins("mov", vec![r(elem(rd, index)), r(elem(rn, i2))]);
        }
        return match imm4 {
            0b0000 => {
                let arr = arrangement(sz, q)?;
                if sz == 3 && !q {
                    return None;
                }
                ins("dup", vec![r(Reg::v(rd, arr)), r(elem(rn, index))])
            }
            0b0001 => {
                let arr = arrangement(sz, q)?;
                if sz == 3 && !q {
                    return None;
                }
                ins("dup", vec![r(Reg::v(rd, arr)), r(Reg::gp(rn, sz == 3))])
            }
            0b0011 if q => ins("mov", vec![r(elem(rd, index)), r(Reg::gp(rn, sz == 3))]),
            0b0101 => {
                if sz >= 2 && !(sz == 2 && q) {
                    return None;
                }
                ins("smov", vec![r(Reg::gp(rd, q)), r(elem(rn, index))])
            }
            0b0111 => {
                if (sz == 3) != q {
                    return None;
                }
                let m = if sz >= 2 { "mov" } else { "umov" };
                ins(m, vec![r(Reg::gp(rd, q)), r(elem(rn, index))])
            }
            _ => None,
        };
    }
    if !bit(w, 15) && bits(w, 11, 10) == 0b10 && !u {
        // Permute.
        let arr = arrangement(size, q)?;
        if size == 3 && !q {
            return None;
        }
        let m = match bits(w, 14, 12) {
            0b001 => "uzp1",
            0b010 => "trn1",
            0b011 => "zip1",
            0b101 => "uzp2",
            0b110 => "trn2",
            0b111 => "zip2",
            _ => return None,
        };
        return ins(m, vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arr)), r(Reg::v(rm, arr))]);
    }
    if u && size == 0 && !bit(w, 15) && !bit(w, 10) {
        // Extract.
        let imm4 = bits(w, 14, 11);
        if !q && imm4 >= 8 {
            return None;
        }
        let arr = if q { "16b" } else { "8b" };
        return ins("ext", vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arr)), r(Reg::v(rm, arr)), Operand::Imm(i64::from(imm4))]);
    }
    if !u && size == 0 && !bit(w, 15) && bits(w, 11, 10) == 0 {
        // Table lookup.
        let len = bits(w, 14, 13) + 1;
        let m = if bit(w, 12) { "tbx" } else { "tbl" };
        let arr = if q { "16b" } else { "8b" };
        let list: Vec<Reg> = (0..len).map(|k| Reg::v((rn + k) % 32, "16b")).collect();
        return ins(m, vec![r(Reg::v(rd, arr)), Operand::List(list, None), r(Reg::v(rm, arr))]);
    }
    None
}

/// AdvSIMD three same.
fn three_same(w: u32) -> Option<A64Inst> {
    let q = bit(w, 30);
    let u = bit(w, 29);
    let size = bits(w, 23, 22);
    let opcode = bits(w, 15, 11);
    let (rd, rn, rm) = (bits(w, 4, 0), bits(w, 9, 5), bits(w, 20, 16));
    if opcode >= 0b11000 {
        // Floating point.
        let a = size >> 1;
        let sz = size & 1;
        if sz == 1 && !q {
            return None;
        }
        let arr = if sz == 1 { "2d" } else if q { "4s" } else { "2s" };
        let m = match (u, a, opcode) {
            (false, 0, 0b11000) => "fmaxnm",
            (false, 0, 0b11001) => "fmla",
            (false, 0, 0b11010) => "fadd",
            (false, 0, 0b11011) => "fmulx",
            (false, 0, 0b11100) => "fcmeq",
            (false, 0, 0b11110) => "fmax",
            (false, 0, 0b11111) => "frecps",
            (false, 1, 0b11000) => "fminnm",
            (false, 1, 0b11001) => "fmls",
            (false, 1, 0b11010) => "fsub",
            (false, 1, 0b11110) => "fmin",
            (false, 1, 0b11111) => "frsqrts",
            (true, 0, 0b11000) => "fmaxnmp",
            (true, 0, 0b11010) => "faddp",
            (true, 0, 0b11011) => "fmul",
            (true, 0, 0b11100) => "fcmge",
            (true, 0, 0b11101) => "facge",
            (true, 0, 0b11110) => "fmaxp",
            (true, 0, 0b11111) => "fdiv",
            (true, 1, 0b11000) => "fminnmp",
            (true, 1, 0b11010) => "fabd",
            (true, 1, 0b11100) => "fcmgt",
            (true, 1, 0b11101) => "facgt",
            (true, 1, 0b11110) => "fminp",
            _ => return None,
        };
        return ins(m, vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arr)), r(Reg::v(rm, arr))]);
    }
    if opcode == 0b00011 {
        // Logical.
        let arr = if q { "16b" } else { "8b" };
        let m = match (u, size) {
            (false, 0) => "and",
            (false, 1) => "bic",
            (false, 2) if rn == rm => return ins("mov", vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arr))]),
            (false, 2) => "orr",
            (false, _) => "orn",
            (true, 0) => "eor",
            (true, 1) => "bsl",
            (true, 2) => "bit",
            (true, _) => "bif",
        };
        return ins(m, vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arr)), r(Reg::v(rm, arr))]);
    }
    let arr = arrangement(size, q)?;
    if size == 3 && !q {
        return None;
    }
    // The ops with a 64-bit lane form.
    let has_2d = matches!(opcode, 0b00001 | 0b00101 | 0b00110 | 0b00111 | 0b01000 | 0b01001 | 0b01010 | 0b01011 | 0b10000 | 0b10001 | 0b10111);
    if size == 3 && !has_2d {
        return None;
    }
    let m = match (u, opcode) {
        (false, 0b00000) => "shadd",
        (true, 0b00000) => "uhadd",
        (false, 0b00001) => "sqadd",
        (true, 0b00001) => "uqadd",
        (false, 0b00010) => "srhadd",
        (true, 0b00010) => "urhadd",
        (false, 0b00100) => "shsub",
        (true, 0b00100) => "uhsub",
        (false, 0b00101) => "sqsub",
        (true, 0b00101) => "uqsub",
        (false, 0b00110) => "cmgt",
        (true, 0b00110) => "cmhi",
        (false, 0b00111) => "cmge",
        (true, 0b00111) => "cmhs",
        (false, 0b01000) => "sshl",
        (true, 0b01000) => "ushl",
        (false, 0b01001) => "sqshl",
        (true, 0b01001) => "uqshl",
        (false, 0b01010) => "srshl",
        (true, 0b01010) => "urshl",
        (false, 0b01011) => "sqrshl",
        (true, 0b01011) => "uqrshl",
        (false, 0b01100) => "smax",
        (true, 0b01100) => "umax",
        (false, 0b01101) => "smin",
        (true, 0b01101) => "umin",
        (false, 0b01110) => "sabd",
        (true, 0b01110) => "uabd",
        (false, 0b01111) => "saba",
        (true, 0b01111) => "uaba",
        (false, 0b10000) => "add",
        (true, 0b10000) => "sub",
        (false, 0b10001) => "cmtst",
        (true, 0b10001) => "cmeq",
        (false, 0b10010) if size != 3 => "mla",
        (true, 0b10010) if size != 3 => "mls",
        (false, 0b10011) if size != 3 => "mul",
        (true, 0b10011) if size == 0 => "pmul",
        (false, 0b10100) if size != 3 => "smaxp",
        (true, 0b10100) if size != 3 => "umaxp",
        (false, 0b10101) if size != 3 => "sminp",
        (true, 0b10101) if size != 3 => "uminp",
        (false, 0b10110) if size == 1 || size == 2 => "sqdmulh",
        (true, 0b10110) if size == 1 || size == 2 => "sqrdmulh",
        (false, 0b10111) => "addp",
        _ => return None,
    };
    ins(m, vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arr)), r(Reg::v(rm, arr))])
}

/// AdvSIMD three different (the long/wide/narrow forms).
fn three_different(w: u32) -> Option<A64Inst> {
    let q = bit(w, 30);
    let u = bit(w, 29);
    let size = bits(w, 23, 22);
    if size == 3 || bits(w, 11, 10) != 0 {
        return None;
    }
    let (rd, rn, rm) = (bits(w, 4, 0), bits(w, 9, 5), bits(w, 20, 16));
    let narrow = arrangement(size, q)?;
    let wide = arrangement(size + 1, true)?;
    let p = if u { "u" } else { "s" };
    let two = if q { "2" } else { "" };
    let (base, d, n, m) = match bits(w, 15, 12) {
        0b0000 => ("addl", wide, narrow, narrow),
        0b0001 => ("addw", wide, wide, narrow),
        0b0010 => ("subl", wide, narrow, narrow),
        0b0011 => ("subw", wide, wide, narrow),
        0b0101 => ("abal", wide, narrow, narrow),
        0b0111 => ("abdl", wide, narrow, narrow),
        0b1000 => ("mlal", wide, narrow, narrow),
        0b1010 => ("mlsl", wide, narrow, narrow),
        0b1100 => ("mull", wide, narrow, narrow),
        _ => return None,
    };
    ins(format!("{p}{base}{two}"), vec![r(Reg::v(rd, d)), r(Reg::v(rn, n)), r(Reg::v(rm, m))])
}

/// AdvSIMD two-register miscellaneous.
fn two_misc(w: u32) -> Option<A64Inst> {
    let q = bit(w, 30);
    let u = bit(w, 29);
    let size = bits(w, 23, 22);
    let opcode = bits(w, 16, 12);
    let (rd, rn) = (bits(w, 4, 0), bits(w, 9, 5));
    // Floating point (opcodes 01100–01111 and 11xxx with size<1> the `a` bit).
    let fp_op = opcode >= 0b11000 || (0b01100..=0b01111).contains(&opcode);
    if fp_op && (size & 2 != 0 || opcode >= 0b11000) {
        let a = size >> 1;
        let sz = size & 1;
        if sz == 1 && !q {
            return None;
        }
        let arr = if sz == 1 { "2d" } else if q { "4s" } else { "2s" };
        let zero = Some(Operand::Text("#0.0".to_owned()));
        let (m, extra) = match (u, a, opcode) {
            (false, 0, 0b11000) => ("frintn", None),
            (false, 0, 0b11001) => ("frintm", None),
            (false, 0, 0b11010) => ("fcvtns", None),
            (false, 0, 0b11011) => ("fcvtms", None),
            (false, 0, 0b11100) => ("fcvtas", None),
            (false, 0, 0b11101) => ("scvtf", None),
            (false, 1, 0b01100) => ("fcmgt", zero),
            (false, 1, 0b01101) => ("fcmeq", zero),
            (false, 1, 0b01110) => ("fcmlt", zero),
            (false, 1, 0b01111) => ("fabs", None),
            (false, 1, 0b11000) => ("frintp", None),
            (false, 1, 0b11001) => ("frintz", None),
            (false, 1, 0b11010) => ("fcvtps", None),
            (false, 1, 0b11011) => ("fcvtzs", None),
            (false, 1, 0b11101) => ("frecpe", None),
            (true, 0, 0b11000) => ("frinta", None),
            (true, 0, 0b11001) => ("frintx", None),
            (true, 0, 0b11010) => ("fcvtnu", None),
            (true, 0, 0b11011) => ("fcvtmu", None),
            (true, 0, 0b11100) => ("fcvtau", None),
            (true, 0, 0b11101) => ("ucvtf", None),
            (true, 1, 0b01100) => ("fcmge", zero),
            (true, 1, 0b01101) => ("fcmle", zero),
            (true, 1, 0b01111) => ("fneg", None),
            (true, 1, 0b11001) => ("frinti", None),
            (true, 1, 0b11010) => ("fcvtpu", None),
            (true, 1, 0b11011) => ("fcvtzu", None),
            (true, 1, 0b11101) => ("frsqrte", None),
            (true, 1, 0b11111) => ("fsqrt", None),
            _ => return None,
        };
        let mut ops = vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arr))];
        ops.extend(extra);
        return ins(m, ops);
    }
    let arr = arrangement(size, q)?;
    // Narrowing and widening forms.
    match (u, opcode) {
        (false, 0b10010) | (false, 0b10100) | (true, 0b10010) | (true, 0b10100) => {
            if size == 3 {
                return None;
            }
            let base = match (u, opcode) {
                (false, 0b10010) => "xtn",
                (false, _) => "sqxtn",
                (true, 0b10010) => "sqxtun",
                (true, _) => "uqxtn",
            };
            let m = format!("{base}{}", if q { "2" } else { "" });
            return ins(m, vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arrangement(size + 1, true)?))]);
        }
        (false, 0b10110) | (true, 0b10110) if size < 2 => {
            // fcvtn / fcvtxn (narrowing float).
            let m = format!("{}{}", if u { "fcvtxn" } else { "fcvtn" }, if q { "2" } else { "" });
            let (to, from) = if size == 1 { (if q { "4s" } else { "2s" }, "2d") } else { (if q { "8h" } else { "4h" }, "4s") };
            return ins(m, vec![r(Reg::v(rd, to)), r(Reg::v(rn, from))]);
        }
        (false, 0b10111) if size < 2 => {
            let m = format!("fcvtl{}", if q { "2" } else { "" });
            let (to, from) = if size == 1 { ("2d", if q { "4s" } else { "2s" }) } else { ("4s", if q { "8h" } else { "4h" }) };
            return ins(m, vec![r(Reg::v(rd, to)), r(Reg::v(rn, from))]);
        }
        (_, 0b00010) | (_, 0b00110) => {
            if size == 3 {
                return None;
            }
            let base = if opcode == 0b00010 { "addlp" } else { "adalp" };
            let m = format!("{}{base}", if u { "u" } else { "s" });
            let wide = arrangement(size + 1, q)?;
            return ins(m, vec![r(Reg::v(rd, wide)), r(Reg::v(rn, arr))]);
        }
        _ => {}
    }
    let zero = Some(Operand::Dec(0));
    let (m, extra) = match (u, opcode) {
        (false, 0b00000) if size < 3 => ("rev64", None),
        (false, 0b00001) if size == 0 => ("rev16", None),
        (true, 0b00000) if size < 2 => ("rev32", None),
        (false, 0b00011) => ("suqadd", None),
        (true, 0b00011) => ("usqadd", None),
        (false, 0b00100) if size < 3 => ("cls", None),
        (true, 0b00100) if size < 3 => ("clz", None),
        (false, 0b00101) if size == 0 => ("cnt", None),
        (true, 0b00101) if size == 0 => ("mvn", None),
        (true, 0b00101) if size == 1 => ("rbit", None),
        (false, 0b00111) => ("sqabs", None),
        (true, 0b00111) => ("sqneg", None),
        (false, 0b01000) => ("cmgt", zero),
        (true, 0b01000) => ("cmge", zero),
        (false, 0b01001) => ("cmeq", zero),
        (true, 0b01001) => ("cmle", zero),
        (false, 0b01010) => ("cmlt", zero),
        (false, 0b01011) => ("abs", None),
        (true, 0b01011) => ("neg", None),
        _ => return None,
    };
    if size == 3 && !q {
        return None;
    }
    let arr = if m == "mvn" || m == "rbit" || m == "cnt" { if q { "16b" } else { "8b" } } else { arr };
    let mut ops = vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arr))];
    ops.extend(extra);
    ins(m, ops)
}

/// AdvSIMD across lanes.
fn across(w: u32) -> Option<A64Inst> {
    let q = bit(w, 30);
    let u = bit(w, 29);
    let size = bits(w, 23, 22);
    let opcode = bits(w, 16, 12);
    let (rd, rn) = (bits(w, 4, 0), bits(w, 9, 5));
    if opcode == 0b01100 || opcode == 0b01111 {
        // fmaxnmv / fmaxv / fminnmv / fminv (single precision, Q = 1).
        if !u || !q || size & 1 != 0 {
            return None;
        }
        let m = match (opcode, size >> 1) {
            (0b01100, 0) => "fmaxnmv",
            (0b01111, 0) => "fmaxv",
            (0b01100, _) => "fminnmv",
            _ => "fminv",
        };
        return ins(m, vec![r(Reg::fp(rd, 's')), r(Reg::v(rn, "4s"))]);
    }
    if size == 3 || (size == 2 && !q) {
        return None;
    }
    let arr = arrangement(size, q)?;
    let letters = ['b', 'h', 's', 'd'];
    let (m, dsize) = match (u, opcode) {
        (false, 0b00011) => ("saddlv", size + 1),
        (true, 0b00011) => ("uaddlv", size + 1),
        (false, 0b01010) => ("smaxv", size),
        (true, 0b01010) => ("umaxv", size),
        (false, 0b11010) => ("sminv", size),
        (true, 0b11010) => ("uminv", size),
        (false, 0b11011) => ("addv", size),
        _ => return None,
    };
    ins(m, vec![r(Reg::fp(rd, letters[dsize as usize])), r(Reg::v(rn, arr))])
}

/// AdvSIMD shift by immediate (vector, or scalar when `scalar`).
fn simd_shift_imm(w: u32, scalar: bool) -> Option<A64Inst> {
    let q = bit(w, 30) || scalar;
    let u = bit(w, 29);
    let immh = bits(w, 22, 19);
    let immhb = bits(w, 22, 16);
    let opcode = bits(w, 15, 11);
    let (rd, rn) = (bits(w, 4, 0), bits(w, 9, 5));
    let hs = 31 - immh.leading_zeros(); // 0..=3
    let esize = 8u32 << hs;
    if hs == 3 && !q {
        return None;
    }
    if scalar && hs != 3 {
        return None;
    }
    let arr = arrangement(hs, bit(w, 30))?;
    let reg = |n| if scalar { Reg::fp(n, 'd') } else { Reg::v(n, arr) };
    let right = i64::from(2 * esize - immhb);
    let left = i64::from(immhb - esize);
    let (m, amount) = match (u, opcode) {
        (false, 0b00000) => ("sshr", right),
        (true, 0b00000) => ("ushr", right),
        (false, 0b00010) => ("ssra", right),
        (true, 0b00010) => ("usra", right),
        (false, 0b00100) => ("srshr", right),
        (true, 0b00100) => ("urshr", right),
        (true, 0b01000) => ("sri", right),
        (false, 0b01010) => ("shl", left),
        (true, 0b01010) => ("sli", left),
        (false, 0b10100) | (true, 0b10100) if !scalar => {
            if hs == 3 {
                return None;
            }
            let m = format!("{}shll{}", if u { "u" } else { "s" }, if bit(w, 30) { "2" } else { "" });
            let wide = arrangement(hs + 1, true)?;
            return ins(m, vec![r(Reg::v(rd, wide)), r(Reg::v(rn, arr)), Operand::Imm(left)]);
        }
        (false, 0b10000) if !scalar => {
            if hs == 3 {
                return None;
            }
            let m = format!("shrn{}", if bit(w, 30) { "2" } else { "" });
            let wide = arrangement(hs + 1, true)?;
            return ins(m, vec![r(Reg::v(rd, arr)), r(Reg::v(rn, wide)), Operand::Imm(right)]);
        }
        _ => return None,
    };
    ins(m, vec![r(reg(rd)), r(reg(rn)), Operand::Imm(amount)])
}

/// AdvSIMD modified immediate (`movi`, `mvni`, `orr`, `bic`, `fmov`).
fn simd_modified_imm(w: u32) -> Option<A64Inst> {
    let q = bit(w, 30);
    let op = bit(w, 29);
    let cmode = bits(w, 15, 12);
    let rd = bits(w, 4, 0);
    if bit(w, 11) {
        return None;
    }
    let imm8 = (bits(w, 18, 16) << 5) | bits(w, 9, 5);
    let imm = Operand::Imm(i64::from(imm8));
    match cmode {
        0b0000..=0b0111 => {
            let arr = if q { "4s" } else { "2s" };
            let shift = 8 * ((cmode >> 1) & 3);
            let m = match (cmode & 1, op) {
                (0, false) => "movi",
                (0, true) => "mvni",
                (_, false) => "orr",
                (_, true) => "bic",
            };
            let mut ops = vec![r(Reg::v(rd, arr)), imm];
            if shift != 0 {
                ops.push(Operand::Modifier("lsl", Some(shift)));
            }
            ins(m, ops)
        }
        0b1000..=0b1011 => {
            let arr = if q { "8h" } else { "4h" };
            let shift = 8 * ((cmode >> 1) & 1);
            let m = match (cmode & 1, op) {
                (0, false) => "movi",
                (0, true) => "mvni",
                (_, false) => "orr",
                (_, true) => "bic",
            };
            let mut ops = vec![r(Reg::v(rd, arr)), imm];
            if shift != 0 {
                ops.push(Operand::Modifier("lsl", Some(shift)));
            }
            ins(m, ops)
        }
        0b1100 | 0b1101 => {
            let arr = if q { "4s" } else { "2s" };
            let shift = if cmode & 1 == 0 { 8 } else { 16 };
            ins(if op { "mvni" } else { "movi" }, vec![r(Reg::v(rd, arr)), imm, Operand::Modifier("msl", Some(shift))])
        }
        0b1110 => {
            if !op {
                return ins("movi", vec![r(Reg::v(rd, if q { "16b" } else { "8b" })), imm]);
            }
            let mut v = 0u64;
            for k in 0..8 {
                if imm8 & (1 << k) != 0 {
                    v |= 0xff << (8 * k);
                }
            }
            let text = if v == 0 { "#0000000000000000".to_owned() } else { format!("#{v:#018x}") };
            let dst = if q { Reg::v(rd, "2d") } else { Reg::fp(rd, 'd') };
            ins("movi", vec![r(dst), Operand::Text(text)])
        }
        _ => {
            // cmode 1111: fmov (vector, immediate).
            let arr = match (op, q) {
                (false, false) => "2s",
                (false, true) => "4s",
                (true, true) => "2d",
                (true, false) => return None,
            };
            ins("fmov", vec![r(Reg::v(rd, arr)), Operand::Float(fp_imm(imm8))])
        }
    }
}

/// The scalar AdvSIMD forms with bits 28:24 `11110` (and bit 30 set): the
/// 64-bit three-same integer ops and `addp`.
fn simd_scalar(w: u32) -> Option<A64Inst> {
    let u = bit(w, 29);
    let size = bits(w, 23, 22);
    let (rd, rn, rm) = (bits(w, 4, 0), bits(w, 9, 5), bits(w, 20, 16));
    if bit(w, 21) && bit(w, 10) {
        if size != 3 {
            return None;
        }
        let m = match (u, bits(w, 15, 11)) {
            (false, 0b10000) => "add",
            (true, 0b10000) => "sub",
            (false, 0b00110) => "cmgt",
            (true, 0b00110) => "cmhi",
            (false, 0b00111) => "cmge",
            (true, 0b00111) => "cmhs",
            (false, 0b10001) => "cmtst",
            (true, 0b10001) => "cmeq",
            (false, 0b01000) => "sshl",
            (true, 0b01000) => "ushl",
            _ => return None,
        };
        let d = |n| r(Reg::fp(n, 'd'));
        return ins(m, vec![d(rd), d(rn), d(rm)]);
    }
    if bits(w, 21, 17) == 0b11000 && bits(w, 11, 10) == 0b10 && !u && size == 3 && bits(w, 16, 12) == 0b11011 {
        return ins("addp", vec![r(Reg::fp(rd, 'd')), r(Reg::v(rn, "2d"))]);
    }
    None
}

/// AdvSIMD vector x indexed element (`mul v0.4s, v1.4s, v2.s[1]`, `fmla`,
/// the long `smull`/`umull` forms).
fn by_element(w: u32) -> Option<A64Inst> {
    let q = bit(w, 30);
    let u = bit(w, 29);
    let size = bits(w, 23, 22);
    let (l, mbit, h) = (bits(w, 21, 21), bits(w, 20, 20), bits(w, 11, 11));
    let opcode = bits(w, 15, 12);
    let (rd, rn) = (bits(w, 4, 0), bits(w, 9, 5));
    let float = matches!((u, opcode), (false, 0b0001) | (false, 0b0101) | (false, 0b1001) | (true, 0b1001));
    if float {
        let (esz, index, arr) = match size {
            0b10 => ('s', (h << 1) | l, if q { "4s" } else { "2s" }),
            0b11 if l == 0 && q => ('d', h, "2d"),
            _ => return None,
        };
        let rm = (mbit << 4) | bits(w, 19, 16);
        let m = match (u, opcode) {
            (false, 0b0001) => "fmla",
            (false, 0b0101) => "fmls",
            (false, 0b1001) => "fmul",
            _ => "fmulx",
        };
        let e = Reg::Elem { n: rm as u8, size: esz, index: index as u8 };
        return ins(m, vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arr)), r(e)]);
    }
    let (esz, index, rm) = match size {
        0b01 => ('h', (h << 2) | (l << 1) | mbit, bits(w, 19, 16)),
        0b10 => ('s', (h << 1) | l, (mbit << 4) | bits(w, 19, 16)),
        _ => return None,
    };
    let e = r(Reg::Elem { n: rm as u8, size: esz, index: index as u8 });
    let arr = arrangement(size, q)?;
    let same = |m: &str| ins(m, vec![r(Reg::v(rd, arr)), r(Reg::v(rn, arr)), e.clone()]);
    let long = |m: &str| {
        let wide = arrangement(size + 1, true)?;
        ins(format!("{m}{}", if q { "2" } else { "" }), vec![r(Reg::v(rd, wide)), r(Reg::v(rn, arr)), e.clone()])
    };
    match (u, opcode) {
        (false, 0b1000) => same("mul"),
        (true, 0b0000) => same("mla"),
        (true, 0b0100) => same("mls"),
        (false, 0b1100) => same("sqdmulh"),
        (false, 0b1101) => same("sqrdmulh"),
        (false, 0b1010) => long("smull"),
        (true, 0b1010) => long("umull"),
        (false, 0b0010) => long("smlal"),
        (true, 0b0010) => long("umlal"),
        (false, 0b0110) => long("smlsl"),
        (true, 0b0110) => long("umlsl"),
        _ => None,
    }
}
