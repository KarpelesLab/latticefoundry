//! The Thumb-2 (ARMv7-M) decoder, written from the ARMv7-M Architecture
//! Reference Manual (chapter A5, "The Thumb Instruction Set Encoding", and
//! the instruction pages of A7.7).
//!
//! An instruction is one or two little-endian halfwords; it is 32 bits wide
//! when its first halfword's top five bits are `0b11101`, `0b11110` or
//! `0b11111`. Decoding produces a typed [`ThumbInst`] (base mnemonic, flag
//! setting, condition, `.w` qualifier, operands, branch target), printed in
//! the Arm unified assembler language with the spellings `llvm-objdump`
//! uses: hex immediates (`#0x10`), `push {r4, lr}`, `ldr.w`, `it eq`, ...
//!
//! The decoder covers the whole ARMv7-M Thumb instruction set except the
//! coprocessor / floating-point space and the ARMv7E-M DSP extension (both
//! print as `.inst.w` data). IT blocks are tracked through [`State`]: an `IT`
//! instruction loads `ITSTATE`, every instruction of the block takes the
//! block's condition as its suffix (`addeq`), and a 16-bit data-processing
//! instruction that sets the flags outside a block (`adds`) does not inside
//! one (`add`).

use super::{Inst, State};

/// The register names, `r0`..`r12`, `sp`, `lr`, `pc`.
const REGS: [&str; 16] = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "sp", "lr", "pc"];

/// The condition-code suffixes (`al` prints as nothing).
const CONDS: [&str; 16] = ["eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "", ""];

fn reg(r: u32) -> &'static str {
    REGS[(r & 15) as usize]
}

/// The name of condition `c` (`al` for 14; used by `it`).
fn cond_name(c: u32) -> &'static str {
    match c & 15 {
        14 => "al",
        c => CONDS[c as usize],
    }
}

/// `#0x…`, with a sign for negative values.
fn imm(v: i64) -> String {
    if v < 0 { format!("#-{:#x}", v.unsigned_abs()) } else { format!("#{v:#x}") }
}

/// A register list `{r4, r5, lr}`.
fn reglist(mask: u32) -> String {
    let regs: Vec<&str> = (0..16).filter(|r| mask >> r & 1 != 0).map(reg).collect();
    format!("{{{}}}", regs.join(", "))
}

/// Sign-extend the low `bits` bits of `v`.
fn sext(v: u32, bits: u32) -> i64 {
    super::sext(u64::from(v), bits)
}

/// One decoded Thumb instruction.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ThumbInst {
    /// Size in bytes: 2 or 4.
    pub size: usize,
    /// The base mnemonic (`add`, `ldrb`, `push`, `it`, ...).
    pub op: String,
    /// Whether the mnemonic takes the flag-setting `s` suffix.
    pub sets_flags: bool,
    /// The condition suffix: a conditional branch's own, or the enclosing IT
    /// block's.
    pub cond: Option<u32>,
    /// Whether the mnemonic takes the `.w` (32-bit) qualifier.
    pub wide: bool,
    /// The operands, in printing order.
    pub operands: Vec<String>,
    /// A direct branch's absolute target.
    pub target: Option<u64>,
}

impl ThumbInst {
    fn new(size: usize, op: &str) -> ThumbInst {
        ThumbInst { size, op: op.to_owned(), sets_flags: false, cond: None, wide: false, operands: Vec::new(), target: None }
    }
    fn s(mut self, s: bool) -> ThumbInst {
        self.sets_flags = s;
        self
    }
    fn w(mut self) -> ThumbInst {
        self.wide = true;
        self
    }
    fn o(mut self, operand: impl Into<String>) -> ThumbInst {
        self.operands.push(operand.into());
        self
    }
    fn t(mut self, target: u64) -> ThumbInst {
        self.target = Some(target & 0xffff_ffff);
        self
    }

    /// The full mnemonic: base, `s`, condition, `.w`.
    pub fn mnemonic(&self) -> String {
        let mut m = self.op.clone();
        if self.sets_flags {
            m.push('s');
        }
        if let Some(c) = self.cond {
            m.push_str(CONDS[(c & 15) as usize]);
        }
        if self.wide {
            m.push_str(".w");
        }
        m
    }

    /// The uniform [`Inst`].
    pub fn to_inst(&self) -> Inst {
        let mut i = Inst::new(self.size, self.mnemonic()).ops(self.operands.iter().cloned());
        if let Some(t) = self.target {
            i = i.target_op(t);
        }
        i
    }
}

/// Decode one Thumb-2 instruction from the start of `bytes` (non-empty),
/// located at address `addr`, outside any IT block.
pub fn decode(bytes: &[u8], addr: u64) -> Inst {
    decode_in(bytes, addr, &mut State::default())
}

/// Decode one Thumb-2 instruction, reading and advancing the IT-block
/// `state`.
pub fn decode_in(bytes: &[u8], addr: u64, state: &mut State) -> Inst {
    if bytes.len() < 2 {
        *state = State::default();
        return Inst::data(bytes, 1, true);
    }
    let it = state.it;
    let in_it = it & 0xf != 0;
    // Advance ITSTATE past this instruction (ITAdvance); an IT instruction
    // replaces it below.
    state.it = if !in_it || it & 7 == 0 { 0 } else { (it & 0xe0) | ((it << 1) & 0x1f) };
    let hw1 = u32::from(u16::from_le_bytes([bytes[0], bytes[1]]));
    let decoded = if hw1 >> 11 >= 0b11101 {
        if bytes.len() < 4 {
            return Inst::data(bytes, 2, true);
        }
        let hw2 = u32::from(u16::from_le_bytes([bytes[2], bytes[3]]));
        match decode32(hw1, hw2, addr) {
            Some(t) => t,
            None => {
                let mut i = Inst::new(4, ".inst.w").op(format!("{:#010x}", hw1 << 16 | hw2));
                i.known = false;
                return i;
            }
        }
    } else {
        match decode16(hw1, addr, in_it) {
            Some(t) => t,
            None => {
                let mut i = Inst::new(2, ".inst.n").op(format!("{hw1:#06x}"));
                i.known = false;
                return i;
            }
        }
    };
    let mut decoded = decoded;
    if decoded.size == 2 && hw1 >> 8 == 0xbf && hw1 & 0xf != 0 {
        state.it = (hw1 & 0xff) as u8;
    } else if in_it && decoded.cond.is_none() && it >> 4 != 14 {
        decoded.cond = Some(u32::from(it >> 4));
    }
    decoded.to_inst()
}

/// Decode a full 32-bit instruction given as its two halfwords (`None` for an
/// encoding outside the supported set).
pub fn decode_inst(bytes: &[u8], addr: u64, in_it: bool) -> Option<ThumbInst> {
    let hw1 = u32::from(u16::from_le_bytes([*bytes.first()?, *bytes.get(1)?]));
    if hw1 >> 11 >= 0b11101 {
        let hw2 = u32::from(u16::from_le_bytes([*bytes.get(2)?, *bytes.get(3)?]));
        decode32(hw1, hw2, addr)
    } else {
        decode16(hw1, addr, in_it)
    }
}

/// `[rn]`, or `[rn, #off]` for a non-zero offset.
fn mem_imm(rn: u32, off: i64) -> String {
    if off == 0 { format!("[{}]", reg(rn)) } else { format!("[{}, {}]", reg(rn), imm(off)) }
}

// ===========================================================================
// 16-bit encodings (A5.2)
// ===========================================================================

fn decode16(hw: u32, addr: u64, in_it: bool) -> Option<ThumbInst> {
    let n = |op: &str| ThumbInst::new(2, op);
    let lo = |shift: u32| (hw >> shift) & 7;
    let s = !in_it;
    let pc = addr.wrapping_add(4);
    Some(match hw >> 10 {
        // Shift (immediate), add, subtract, move, compare.
        0b000000..=0b001111 => {
            let opc = (hw >> 9) & 0x1f;
            let (d, m3) = (lo(0), lo(3));
            let imm5 = (hw >> 6) & 0x1f;
            match opc {
                0b00000..=0b00011 if imm5 == 0 => n("mov").s(true).o(reg(d)).o(reg(m3)),
                0b00000..=0b00011 => n("lsl").s(s).o(reg(d)).o(reg(m3)).o(imm(imm5.into())),
                0b00100..=0b01011 => {
                    let amount = if imm5 == 0 { 32 } else { imm5 };
                    n(if opc < 0b01000 { "lsr" } else { "asr" }).s(s).o(reg(d)).o(reg(m3)).o(imm(amount.into()))
                }
                0b01100 | 0b01101 => n(if opc & 1 == 0 { "add" } else { "sub" }).s(s).o(reg(d)).o(reg(m3)).o(reg(lo(6))),
                0b01110 | 0b01111 => n(if opc & 1 == 0 { "add" } else { "sub" }).s(s).o(reg(d)).o(reg(m3)).o(imm(lo(6).into())),
                _ => {
                    let rd = lo(8);
                    let imm8 = i64::from(hw & 0xff);
                    match opc >> 2 {
                        0b100 => n("mov").s(s).o(reg(rd)).o(imm(imm8)),
                        0b101 => n("cmp").o(reg(rd)).o(imm(imm8)),
                        0b110 => n("add").s(s).o(reg(rd)).o(imm(imm8)),
                        _ => n("sub").s(s).o(reg(rd)).o(imm(imm8)),
                    }
                }
            }
        }
        // Data processing (register).
        0b010000 => {
            let (dn, m) = (reg(lo(0)), reg(lo(3)));
            match (hw >> 6) & 15 {
                0 => n("and").s(s).o(dn).o(m),
                1 => n("eor").s(s).o(dn).o(m),
                2 => n("lsl").s(s).o(dn).o(m),
                3 => n("lsr").s(s).o(dn).o(m),
                4 => n("asr").s(s).o(dn).o(m),
                5 => n("adc").s(s).o(dn).o(m),
                6 => n("sbc").s(s).o(dn).o(m),
                7 => n("ror").s(s).o(dn).o(m),
                8 => n("tst").o(dn).o(m),
                9 => n("rsb").s(s).o(dn).o(m).o("#0"),
                10 => n("cmp").o(dn).o(m),
                11 => n("cmn").o(dn).o(m),
                12 => n("orr").s(s).o(dn).o(m),
                13 => n("mul").s(s).o(dn).o(m).o(dn),
                14 => n("bic").s(s).o(dn).o(m),
                _ => n("mvn").s(s).o(dn).o(m),
            }
        }
        // Special data instructions and branch and exchange.
        0b010001 => {
            let dn = (hw >> 4) & 8 | lo(0);
            let m = (hw >> 3) & 15;
            match (hw >> 6) & 15 {
                0b0000..=0b0011 => {
                    if m == 13 {
                        n("add").o(reg(dn)).o("sp").o(reg(dn))
                    } else {
                        n("add").o(reg(dn)).o(reg(m))
                    }
                }
                0b0100..=0b0111 => n("cmp").o(reg(dn)).o(reg(m)),
                0b1000..=0b1011 => n("mov").o(reg(dn)).o(reg(m)),
                0b1100 | 0b1101 => n("bx").o(reg(m)),
                _ if hw & 7 == 0 => n("blx").o(reg(m)),
                _ => return None,
            }
        }
        // LDR (literal).
        0b010010 | 0b010011 => n("ldr").o(reg(lo(8))).o(format!("[pc, {}]", imm(i64::from(hw & 0xff) * 4))),
        // Load/store single data item.
        0b010100..=0b100111 => {
            let (t, b) = (reg(lo(0)), lo(3));
            match hw >> 12 {
                0b0101 => {
                    const OPS: [&str; 8] = ["str", "strh", "strb", "ldrsb", "ldr", "ldrh", "ldrb", "ldrsh"];
                    n(OPS[lo(9) as usize]).o(t).o(format!("[{}, {}]", reg(b), reg(lo(6))))
                }
                0b0110..=0b1000 => {
                    let load = hw & 0x800 != 0;
                    let imm5 = i64::from((hw >> 6) & 0x1f);
                    let (op, off) = match hw >> 12 {
                        0b0110 => (if load { "ldr" } else { "str" }, imm5 * 4),
                        0b0111 => (if load { "ldrb" } else { "strb" }, imm5),
                        _ => (if load { "ldrh" } else { "strh" }, imm5 * 2),
                    };
                    n(op).o(t).o(mem_imm(b, off))
                }
                _ => {
                    let load = hw & 0x800 != 0;
                    n(if load { "ldr" } else { "str" }).o(reg(lo(8))).o(mem_imm(13, i64::from(hw & 0xff) * 4))
                }
            }
        }
        // ADR, and ADD (SP plus immediate).
        0b101000 | 0b101001 => n("adr").o(reg(lo(8))).o(imm(i64::from(hw & 0xff) * 4)),
        0b101010 | 0b101011 => n("add").o(reg(lo(8))).o("sp").o(imm(i64::from(hw & 0xff) * 4)),
        // Miscellaneous 16-bit instructions.
        0b101100..=0b101111 => return misc16(hw, pc),
        // STM / LDM.
        0b110000..=0b110011 if hw & 0xff == 0 => return None,
        0b110000 | 0b110001 => {
            let rn = lo(8);
            n("stm").o(format!("{}!", reg(rn))).o(reglist(hw & 0xff))
        }
        0b110010 | 0b110011 => {
            let rn = lo(8);
            let wb = hw >> rn & 1 == 0;
            n("ldm").o(if wb { format!("{}!", reg(rn)) } else { reg(rn).to_owned() }).o(reglist(hw & 0xff))
        }
        // Conditional branch, UDF, SVC.
        0b110100..=0b110111 => {
            let c = (hw >> 8) & 15;
            let imm8 = hw & 0xff;
            match c {
                14 => n("udf").o(imm(imm8.into())),
                15 => n("svc").o(imm(imm8.into())),
                _ => {
                    let mut i = n("b").t(pc.wrapping_add_signed(sext(imm8 << 1, 9)));
                    i.cond = Some(c);
                    i
                }
            }
        }
        // Unconditional branch.
        0b111000 | 0b111001 => n("b").t(pc.wrapping_add_signed(sext((hw & 0x7ff) << 1, 12))),
        _ => return None,
    })
}

/// The miscellaneous 16-bit instructions (A5.2.5).
fn misc16(hw: u32, pc: u64) -> Option<ThumbInst> {
    let n = |op: &str| ThumbInst::new(2, op);
    let op = (hw >> 5) & 0x7f;
    Some(match op {
        0b0000000..=0b0000011 => n("add").o("sp").o(imm(i64::from(hw & 0x7f) * 4)),
        0b0000100..=0b0000111 => n("sub").o("sp").o(imm(i64::from(hw & 0x7f) * 4)),
        _ if matches!((hw >> 8) & 15, 0b0001 | 0b0011 | 0b1001 | 0b1011) => {
            let off = ((hw >> 9) & 1) << 6 | ((hw >> 3) & 0x1f) << 1;
            n(if hw & 0x800 != 0 { "cbnz" } else { "cbz" }).o(reg(hw & 7)).t(pc + u64::from(off))
        }
        0b0010000..=0b0010111 => {
            const OPS: [&str; 4] = ["sxth", "sxtb", "uxth", "uxtb"];
            n(OPS[((hw >> 6) & 3) as usize]).o(reg(hw & 7)).o(reg((hw >> 3) & 7))
        }
        0b0100000..=0b0101111 | 0b1100000..=0b1101111 if hw & 0x1ff == 0 => return None,
        0b0100000..=0b0101111 => {
            let list = (hw & 0xff) | ((hw >> 8) & 1) << 14;
            n("push").o(reglist(list))
        }
        0b0110011 => {
            if hw & 0x8 != 0 {
                return None;
            }
            let mut flags: String = [(4, 'a'), (2, 'i'), (1, 'f')].iter().filter(|(b, _)| hw & b != 0).map(|(_, c)| *c).collect();
            if flags.is_empty() {
                flags.push_str("none");
            }
            n(if hw & 0x10 != 0 { "cpsid" } else { "cpsie" }).o(flags)
        }
        0b1010000..=0b1010001 => n("rev").o(reg(hw & 7)).o(reg((hw >> 3) & 7)),
        0b1010010..=0b1010011 => n("rev16").o(reg(hw & 7)).o(reg((hw >> 3) & 7)),
        0b1010110..=0b1010111 => n("revsh").o(reg(hw & 7)).o(reg((hw >> 3) & 7)),
        0b1100000..=0b1101111 => {
            let list = (hw & 0xff) | ((hw >> 8) & 1) << 15;
            n("pop").o(reglist(list))
        }
        0b1110000..=0b1110111 => n("bkpt").o(imm(i64::from(hw & 0xff))),
        0b1111000..=0b1111111 => {
            let mask = hw & 15;
            if mask != 0 {
                // IT{x{y{z}}} firstcond: each later slot is `t` when its
                // mask bit equals firstcond<0>, else `e`.
                let fc = (hw >> 4) & 15;
                if fc == 15 || (fc == 14 && mask.count_ones() != 1) {
                    return None;
                }
                let count = 4 - mask.trailing_zeros();
                let mut suffix = String::new();
                for k in 1..count {
                    let bit = (mask >> (4 - k)) & 1;
                    suffix.push(if bit == fc & 1 { 't' } else { 'e' });
                }
                n(&format!("it{suffix}")).o(cond_name(fc))
            } else {
                match (hw >> 4) & 15 {
                    0 => n("nop"),
                    1 => n("yield"),
                    2 => n("wfe"),
                    3 => n("wfi"),
                    4 => n("sev"),
                    h => n("hint").o(imm(i64::from(h))),
                }
            }
        }
        _ => return None,
    })
}

// ===========================================================================
// 32-bit encodings (A5.3)
// ===========================================================================

/// ThumbExpandImm (A5.3.2). (A replicated pattern of a zero byte is
/// UNPREDICTABLE; it decodes as zero.)
fn expand_imm(imm12: u32) -> Option<u32> {
    let imm8 = imm12 & 0xff;
    if imm12 >> 10 == 0 {
        let v = match (imm12 >> 8) & 3 {
            0 => imm8,
            1 => imm8 << 16 | imm8,
            2 => imm8 << 24 | imm8 << 8,
            _ => imm8 * 0x0101_0101,
        };
        Some(v)
    } else {
        Some((0x80 | (imm12 & 0x7f)).rotate_right(imm12 >> 7))
    }
}

/// The text of an immediate shift (`, lsl #3`), or `None` for no shift.
fn shift_text(ty: u32, imm5: u32) -> Option<String> {
    match (ty, imm5) {
        (0, 0) => None,
        (0, n) => Some(format!("lsl #{n}")),
        (1, n) => Some(format!("lsr #{}", if n == 0 { 32 } else { n })),
        (2, n) => Some(format!("asr #{}", if n == 0 { 32 } else { n })),
        (_, 0) => Some("rrx".to_owned()),
        (_, n) => Some(format!("ror #{n}")),
    }
}

fn decode32(hw1: u32, hw2: u32, addr: u64) -> Option<ThumbInst> {
    let op1 = (hw1 >> 11) & 3;
    let op2 = (hw1 >> 4) & 0x7f;
    match op1 {
        1 => {
            if op2 & 0b110_0100 == 0 {
                ldst_multiple(hw1, hw2)
            } else if op2 & 0b110_0100 == 0b000_0100 {
                ldst_dual_excl(hw1, hw2)
            } else if op2 & 0b110_0000 == 0b010_0000 {
                dp_shifted(hw1, hw2)
            } else {
                coprocessor(hw1, hw2)
            }
        }
        2 => {
            if hw2 & 0x8000 == 0 {
                if op2 & 0b010_0000 == 0 { dp_modimm(hw1, hw2) } else { dp_plainimm(hw1, hw2) }
            } else {
                branch_misc(hw1, hw2, addr)
            }
        }
        3 => {
            if op2 & 0b111_0001 == 0 {
                store_single(hw1, hw2)
            } else if op2 & 0b110_0111 == 0b000_0001 || op2 & 0b110_0111 == 0b000_0011 || op2 & 0b110_0111 == 0b000_0101 {
                load_single(hw1, hw2)
            } else if op2 & 0b111_0000 == 0b010_0000 {
                dp_register(hw1, hw2)
            } else if op2 & 0b111_1000 == 0b011_0000 {
                multiply(hw1, hw2)
            } else if op2 & 0b111_1000 == 0b011_1000 {
                long_multiply(hw1, hw2)
            } else if op2 & 0b100_0000 != 0 {
                coprocessor(hw1, hw2)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn w(op: &str) -> ThumbInst {
    ThumbInst::new(4, op)
}

/// Load/store multiple (A5.3.5).
fn ldst_multiple(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    let wb = hw1 & 0x20 != 0;
    let load = hw1 & 0x10 != 0;
    let rn = hw1 & 15;
    let base = if wb { format!("{}!", reg(rn)) } else { reg(rn).to_owned() };
    if hw2 == 0 || (!load && hw2 & 0xa000 != 0) {
        return None; // UNPREDICTABLE: no registers, or SP or PC stored
    }
    let list = reglist(hw2);
    Some(match (hw1 >> 7) & 3 {
        0b01 if load && wb && rn == 13 => w("pop").w().o(list),
        0b01 => w(if load { "ldm" } else { "stm" }).w().o(base).o(list),
        0b10 if !load && wb && rn == 13 => w("push").w().o(list),
        0b10 => w(if load { "ldmdb" } else { "stmdb" }).o(base).o(list),
        _ => return None,
    })
}

/// Load/store dual or exclusive, table branch (A5.3.6).
fn ldst_dual_excl(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    let (p, u, wb, load) = (hw1 >> 8 & 1, hw1 >> 7 & 1, hw1 >> 5 & 1, hw1 & 0x10 != 0);
    let rn = hw1 & 15;
    let rt = hw2 >> 12;
    let rd = (hw2 >> 8) & 15;
    if p == 1 || wb == 1 {
        // LDRD / STRD (immediate).
        let off = i64::from(hw2 & 0xff) * 4;
        let off = if u == 1 { off } else { -off };
        let addr = match (p, wb) {
            (1, 0) => mem_imm(rn, off),
            (1, _) => format!("[{}, {}]!", reg(rn), imm(off)),
            _ => format!("[{}], {}", reg(rn), imm(off)),
        };
        let addr = if p == 1 && wb == 0 && off == 0 && u == 0 { format!("[{}, #-0x0]", reg(rn)) } else { addr };
        return Some(w(if load { "ldrd" } else { "strd" }).o(reg(rt)).o(reg(rd)).o(addr));
    }
    let op3 = (hw2 >> 4) & 15;
    Some(match ((hw1 >> 7) & 3, (hw1 >> 4) & 3) {
        (0, 0) => w("strex").o(reg(rd)).o(reg(rt)).o(mem_imm(rn, i64::from(hw2 & 0xff) * 4)),
        (0, 1) if rd == 15 => w("ldrex").o(reg(rt)).o(mem_imm(rn, i64::from(hw2 & 0xff) * 4)),
        (1, _) if rd != 15 && op3 != 0 && op3 != 1 => return None,
        (1, 0) => match op3 {
            4 => w("strexb").o(reg(hw2 & 15)).o(reg(rt)).o(mem_imm(rn, 0)),
            5 => w("strexh").o(reg(hw2 & 15)).o(reg(rt)).o(mem_imm(rn, 0)),
            _ => return None,
        },
        (1, 1) => match op3 {
            0 | 1 if rt != 15 || rd != 0 || hw2 & 15 == 13 => return None,
            0 => w("tbb").o(format!("[{}, {}]", reg(rn), reg(hw2 & 15))),
            1 => w("tbh").o(format!("[{}, {}, lsl #1]", reg(rn), reg(hw2 & 15))),
            4 | 5 if hw2 & 15 != 15 => return None,
            4 => w("ldrexb").o(reg(rt)).o(mem_imm(rn, 0)),
            5 => w("ldrexh").o(reg(rt)).o(mem_imm(rn, 0)),
            _ => return None,
        },
        _ => return None,
    })
}

/// Data processing (shifted register) (A5.3.11).
fn dp_shifted(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    let op = (hw1 >> 5) & 15;
    let s = hw1 & 0x10 != 0;
    let rn = hw1 & 15;
    let rd = (hw2 >> 8) & 15;
    let rm = hw2 & 15;
    let ty = (hw2 >> 4) & 3;
    let imm5 = (hw2 >> 10) & 0x1c | (hw2 >> 6) & 3;
    let shift = shift_text(ty, imm5);
    let with_shift = |mut i: ThumbInst| {
        if let Some(sh) = &shift {
            i = i.o(sh.clone());
        }
        i
    };
    let test = |name: &str| with_shift(w(name).w().o(reg(rn)).o(reg(rm)));
    let three = |name: &str, wide: bool| {
        let i = w(name).s(s).o(reg(rd)).o(reg(rn)).o(reg(rm));
        with_shift(if wide { i.w() } else { i })
    };
    Some(match op {
        0b0000 if rd == 15 && s => test("tst"),
        0b0000 => three("and", true),
        0b0001 => three("bic", true),
        // (Bit 15 of the second halfword should be zero; `llvm-objdump`
        // only enforces it for ORR and its MOV/shift aliases, not RRX.)
        0b0010 if hw2 & 0x8000 != 0 && !(rn == 15 && ty == 3 && imm5 == 0) => return None,
        0b0010 if rn == 15 => match (ty, imm5) {
            (0, 0) => w("mov").s(s).w().o(reg(rd)).o(reg(rm)),
            (3, 0) => w("rrx").s(s).o(reg(rd)).o(reg(rm)),
            _ => {
                const OPS: [&str; 4] = ["lsl", "lsr", "asr", "ror"];
                let amount = if imm5 == 0 { 32 } else { imm5 };
                w(OPS[ty as usize]).s(s).w().o(reg(rd)).o(reg(rm)).o(imm(amount.into()))
            }
        },
        0b0010 => three("orr", true),
        0b0011 if rn == 15 => with_shift(w("mvn").s(s).w().o(reg(rd)).o(reg(rm))),
        0b0011 => three("orn", false),
        0b0100 if rd == 15 && s => test("teq"),
        0b0100 => three("eor", true),
        0b1000 if rd == 15 && s => test("cmn"),
        0b1000 => three("add", true),
        0b1010 => three("adc", true),
        0b1011 => three("sbc", true),
        0b1101 if rd == 15 && s => test("cmp"),
        0b1101 => three("sub", true),
        0b1110 => three("rsb", false),
        _ => return None,
    })
}

/// Data processing (modified immediate) (A5.3.1).
fn dp_modimm(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    let op = (hw1 >> 5) & 15;
    let s = hw1 & 0x10 != 0;
    let rn = hw1 & 15;
    let rd = (hw2 >> 8) & 15;
    let imm12 = (hw1 >> 10 & 1) << 11 | (hw2 >> 12 & 7) << 8 | (hw2 & 0xff);
    let v = imm(i64::from(expand_imm(imm12)?));
    let test = |name: &str| w(name).w().o(reg(rn)).o(v.clone());
    let three = |name: &str, wide: bool| {
        let i = w(name).s(s).o(reg(rd)).o(reg(rn)).o(v.clone());
        if wide { i.w() } else { i }
    };
    Some(match op {
        0b0000 if rd == 15 && s => test("tst"),
        0b0000 => three("and", false),
        0b0001 => three("bic", false),
        0b0010 if rn == 15 => w("mov").s(s).w().o(reg(rd)).o(v.clone()),
        0b0010 => three("orr", false),
        0b0011 if rn == 15 => w("mvn").s(s).o(reg(rd)).o(v.clone()),
        0b0011 => three("orn", false),
        0b0100 if rd == 15 && s => test("teq"),
        0b0100 => three("eor", false),
        0b1000 if rd == 15 && s => test("cmn"),
        0b1000 => three("add", true),
        0b1010 => three("adc", false),
        0b1011 => three("sbc", false),
        0b1101 if rd == 15 && s => test("cmp"),
        0b1101 => three("sub", true),
        0b1110 => three("rsb", true),
        _ => return None,
    })
}

/// Data processing (plain binary immediate) (A5.3.3).
fn dp_plainimm(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    let op = (hw1 >> 4) & 0x1f;
    let rn = hw1 & 15;
    let rd = (hw2 >> 8) & 15;
    let imm12 = (hw1 >> 10 & 1) << 11 | (hw2 >> 12 & 7) << 8 | (hw2 & 0xff);
    let imm5 = (hw2 >> 10) & 0x1c | (hw2 >> 6) & 3;
    let low5 = hw2 & 0x1f;
    Some(match op {
        0b00000 if rn == 15 => w("adr").w().o(reg(rd)).o(imm(imm12.into())),
        0b00000 => w("addw").o(reg(rd)).o(reg(rn)).o(imm(imm12.into())),
        0b01010 if rn == 15 => w("adr").w().o(reg(rd)).o(imm(-i64::from(imm12))),
        0b01010 => w("subw").o(reg(rd)).o(reg(rn)).o(imm(imm12.into())),
        0b00100 | 0b01100 => {
            let imm16 = rn << 12 | imm12;
            w(if op == 0b00100 { "movw" } else { "movt" }).o(reg(rd)).o(imm(imm16.into()))
        }
        0b10000 | 0b10010 | 0b11000 | 0b11010 | 0b10110 if hw2 & 0x20 != 0 || hw1 & 0x400 != 0 => return None,
        0b10000 | 0b10010 | 0b11000 | 0b11010 => {
            let asr = op & 2 != 0;
            if asr && imm5 == 0 {
                return None; // SSAT16 / USAT16 (DSP)
            }
            let signed = op & 0b01000 == 0;
            let sat = if signed { low5 + 1 } else { low5 };
            let mut i = w(if signed { "ssat" } else { "usat" }).o(reg(rd)).o(imm(sat.into())).o(reg(rn));
            if asr {
                i = i.o(format!("asr #{imm5}"));
            } else if imm5 != 0 {
                i = i.o(format!("lsl #{imm5}"));
            }
            i
        }
        0b10100 | 0b11100 => {
            let width = low5 + 1;
            w(if op == 0b10100 { "sbfx" } else { "ubfx" }).o(reg(rd)).o(reg(rn)).o(imm(imm5.into())).o(imm(width.into()))
        }
        0b10110 => {
            if low5 < imm5 {
                return None;
            }
            let width = low5 - imm5 + 1;
            if rn == 15 {
                w("bfc").o(reg(rd)).o(format!("#{imm5}")).o(format!("#{width}"))
            } else {
                w("bfi").o(reg(rd)).o(reg(rn)).o(format!("#{imm5}")).o(format!("#{width}"))
            }
        }
        _ => return None,
    })
}

/// The special register named by `sysm` in MRS/MSR (B5.1.1), with the APSR
/// write mask for MSR.
fn special_reg(sysm: u32, mask: Option<u32>) -> Option<String> {
    let name = match sysm {
        0 => "apsr",
        1 => "iapsr",
        2 => "eapsr",
        3 => "xpsr",
        5 => "ipsr",
        6 => "epsr",
        7 => "iepsr",
        8 => "msp",
        9 => "psp",
        16 => "primask",
        17 => "basepri",
        18 => "basepri_max",
        19 => "faultmask",
        20 => "control",
        _ => return Some(sysm.to_string()),
    };
    match mask {
        // ARMv7-M without the DSP extension has no APSR.GE bits: any
        // non-zero write mask writes the flags.
        Some(0) if sysm < 4 => None,
        Some(_) if sysm < 4 => Some(format!("{name}_nzcvq")),
        // (The write mask only qualifies the APSR group.)
        _ => Some(name.to_owned()),
    }
}

/// A barrier option's name.
fn barrier(opt: u32) -> String {
    match opt {
        15 => "sy".to_owned(),
        14 => "st".to_owned(),
        11 => "ish".to_owned(),
        10 => "ishst".to_owned(),
        7 => "nsh".to_owned(),
        6 => "nshst".to_owned(),
        3 => "osh".to_owned(),
        2 => "oshst".to_owned(),
        o => format!("#{o:#x}"),
    }
}

/// Branches and miscellaneous control (A5.3.4).
fn branch_misc(hw1: u32, hw2: u32, addr: u64) -> Option<ThumbInst> {
    let op1 = (hw2 >> 12) & 7;
    let op = (hw1 >> 4) & 0x7f;
    let pc = addr.wrapping_add(4);
    let s = (hw1 >> 10) & 1;
    let (j1, j2) = ((hw2 >> 13) & 1, (hw2 >> 11) & 1);
    match op1 & 0b101 {
        0b000 if op & 0b011_1000 != 0b011_1000 => {
            let off = s << 20 | j2 << 19 | j1 << 18 | (hw1 & 0x3f) << 12 | (hw2 & 0x7ff) << 1;
            let mut i = w("b").w().t(pc.wrapping_add_signed(sext(off, 21)));
            i.cond = Some((hw1 >> 6) & 15);
            Some(i)
        }
        0b000 => match op {
            0b011_1000 | 0b011_1001 => {
                let mask = (hw2 >> 10) & 3;
                Some(w("msr").o(special_reg(hw2 & 0xff, Some(mask))?).o(reg(hw1 & 15)))
            }
            0b011_1010 => {
                if hw2 & 0x2f00 != 0 || hw1 & 15 != 15 {
                    return None;
                }
                Some(match hw2 & 0xff {
                    0 => w("nop").w(),
                    1 => w("yield").w(),
                    2 => w("wfe").w(),
                    3 => w("wfi").w(),
                    4 => w("sev").w(),
                    0x14 => w("csdb"),
                    o if o >= 0xf0 => w("dbg").o(imm(i64::from(o & 15))),
                    o => w("hint").w().o(imm(i64::from(o))),
                })
            }
            0b011_1011 if hw1 & 15 != 15 || hw2 & 0x0f00 != 0x0f00 => None,
            0b011_1011 => Some(match (hw2 >> 4) & 15 {
                2 => w("clrex"),
                4 if hw2 & 15 == 0 => w("ssbb"),
                4 if hw2 & 15 == 4 => w("pssbb"),
                4 => w("dsb").o(barrier(hw2 & 15)),
                5 => w("dmb").o(barrier(hw2 & 15)),
                6 => w("isb").o(if hw2 & 15 == 15 { "sy".to_owned() } else { format!("#{:#x}", hw2 & 15) }),
                _ => return None,
            }),
            0b011_1110 | 0b011_1111 => Some(w("mrs").o(reg((hw2 >> 8) & 15)).o(special_reg(hw2 & 0xff, None)?)),
            0b111_1111 if op1 == 0b010 => Some(w("udf").w().o(imm(i64::from((hw1 & 15) << 12 | (hw2 & 0xfff))))),
            _ => None,
        },
        0b001 | 0b101 => {
            let i1 = !(j1 ^ s) & 1;
            let i2 = !(j2 ^ s) & 1;
            let off = s << 24 | i1 << 23 | i2 << 22 | (hw1 & 0x3ff) << 12 | (hw2 & 0x7ff) << 1;
            let target = pc.wrapping_add_signed(sext(off, 25));
            Some(if op1 & 0b100 != 0 { w("bl").t(target) } else { w("b").w().t(target) })
        }
        _ => None,
    }
}

/// `[rn, #±imm8]` / `[rn, #±imm8]!` / `[rn], #±imm8` from P, U, W.
fn imm8_mode(rn: u32, hw2: u32) -> Option<(String, bool)> {
    let (p, u, wb) = (hw2 >> 10 & 1, hw2 >> 9 & 1, hw2 >> 8 & 1);
    let v = i64::from(hw2 & 0xff);
    let off = if u == 1 { v } else { -v };
    let neg_zero = |s: String| if u == 0 && v == 0 { format!("[{}, #-0x0]", reg(rn)) } else { s };
    Some(match (p, u, wb) {
        (1, 1, 0) => (mem_imm(rn, off), true), // unprivileged
        (1, 0, 0) => (neg_zero(mem_imm(rn, off)), false),
        (1, _, 1) => (format!("[{}, {}]!", reg(rn), imm(off)), false),
        (0, _, 1) => (format!("[{}], {}", reg(rn), imm(off)), false),
        _ => return None,
    })
}

/// Store single data item (A5.3.10).
fn store_single(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    let size = (hw1 >> 5) & 3;
    let name = ["strb", "strh", "str"].get(size as usize)?;
    let rn = hw1 & 15;
    let rt = reg(hw2 >> 12);
    if rn == 15 {
        return None;
    }
    if hw1 & 0x80 != 0 {
        return Some(w(name).w().o(rt).o(mem_imm(rn, i64::from(hw2 & 0xfff))));
    }
    if hw2 & 0x800 != 0 {
        let (m, unpriv) = imm8_mode(rn, hw2)?;
        return Some(w(&if unpriv { format!("{name}t") } else { (*name).to_owned() }).o(rt).o(m));
    }
    if hw2 & 0xfc0 == 0 {
        return Some(w(name).w().o(rt).o(reg_offset(rn, hw2)));
    }
    None
}

/// `[rn, rm]` or `[rn, rm, lsl #n]`.
fn reg_offset(rn: u32, hw2: u32) -> String {
    let sh = (hw2 >> 4) & 3;
    if sh == 0 { format!("[{}, {}]", reg(rn), reg(hw2 & 15)) } else { format!("[{}, {}, lsl #{sh}]", reg(rn), reg(hw2 & 15)) }
}

/// Load byte / halfword / word, and the memory hints (A5.3.7-9).
fn load_single(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    let size = (hw1 >> 5) & 3;
    let signed = hw1 & 0x100 != 0;
    let rn = hw1 & 15;
    let rt = hw2 >> 12;
    let name = match (size, signed) {
        (0, false) => "ldrb",
        (0, true) => "ldrsb",
        (1, false) => "ldrh",
        (1, true) => "ldrsh",
        (2, false) => "ldr",
        _ => return None,
    };
    // A load of a byte into the PC is a preload hint; of a halfword, an
    // unallocated hint.
    let hint = rt == 15 && size < 2;
    let hint_name = if signed { "pli" } else { "pld" };
    let indexed = hw1 & 0x80 == 0 && rn != 15 && hw2 & 0x800 != 0 && hw2 & 0x700 != 0x400;
    if hint && size == 1 && !indexed && !(rn == 15 && !signed) {
        return None;
    }
    if rn == 15 {
        let v = i64::from(hw2 & 0xfff);
        let off = if hw1 & 0x80 != 0 { v } else { -v };
        let m = if off == 0 && hw1 & 0x80 == 0 { "[pc, #-0x0]".to_owned() } else { format!("[pc, {}]", imm(off)) };
        return Some(if hint { w(hint_name).o(m) } else { w(name).w().o(reg(rt)).o(m) });
    }
    if hw1 & 0x80 != 0 {
        let m = mem_imm(rn, i64::from(hw2 & 0xfff));
        return Some(if hint { w(hint_name).o(m) } else { w(name).w().o(reg(rt)).o(m) });
    }
    if hw2 & 0x800 != 0 {
        let (m, unpriv) = imm8_mode(rn, hw2)?;
        if hint && hw2 & 0x700 == 0x400 {
            // The negative-offset form is a hint; the others load the PC.
            return Some(w(hint_name).o(m));
        }
        return Some(w(&if unpriv { format!("{name}t") } else { name.to_owned() }).o(reg(rt)).o(m));
    }
    if hw2 & 0xfc0 == 0 {
        let m = reg_offset(rn, hw2);
        return Some(if hint { w(hint_name).o(m) } else { w(name).w().o(reg(rt)).o(m) });
    }
    None
}

/// Data processing (register): shifts by a register, extends, and the
/// miscellaneous operations (A5.3.12).
fn dp_register(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    if hw2 & 0xf000 != 0xf000 {
        return None;
    }
    let op1 = (hw1 >> 4) & 15;
    let op2 = (hw2 >> 4) & 15;
    let rn = hw1 & 15;
    let rd = (hw2 >> 8) & 15;
    let rm = hw2 & 15;
    if op2 == 0 && op1 < 8 {
        const OPS: [&str; 4] = ["lsl", "lsr", "asr", "ror"];
        return Some(w(OPS[(op1 >> 1) as usize]).s(op1 & 1 != 0).w().o(reg(rd)).o(reg(rn)).o(reg(rm)));
    }
    if op2 & 8 != 0 && op1 < 8 {
        let name = match op1 {
            0 => "sxth",
            1 => "uxth",
            4 => "sxtb",
            5 => "uxtb",
            _ => return None,
        };
        if rn != 15 {
            return None; // SXTAH and friends (DSP)
        }
        let rot = (hw2 >> 4) & 3;
        let mut i = w(name).w().o(reg(rd)).o(reg(rm));
        if rot != 0 {
            i = i.o(format!("ror #{}", rot * 8));
        }
        return Some(i);
    }
    if op1 & 0b1100 == 0b1000 && op2 & 0b1100 == 0b1000 {
        return Some(match (op1 & 3, op2 & 3) {
            // (The source register is encoded twice; the first copy is
            // the one printed.)
            (1, 0) => w("rev").w().o(reg(rd)).o(reg(rn)),
            (1, 1) => w("rev16").w().o(reg(rd)).o(reg(rn)),
            (1, 2) => w("rbit").o(reg(rd)).o(reg(rn)),
            (1, 3) => w("revsh").w().o(reg(rd)).o(reg(rn)),
            (3, 0) => w("clz").o(reg(rd)).o(reg(rn)),
            _ => return None,
        });
    }
    None
}

/// Multiply and multiply-accumulate (A5.3.16).
fn multiply(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    let (rn, ra, rd, rm) = (hw1 & 15, hw2 >> 12, (hw2 >> 8) & 15, hw2 & 15);
    if hw2 & 0xc0 != 0 {
        return None;
    }
    Some(match ((hw1 >> 4) & 7, (hw2 >> 4) & 3) {
        (0, 0) if ra == 15 => w("mul").o(reg(rd)).o(reg(rn)).o(reg(rm)),
        (0, 0) => w("mla").o(reg(rd)).o(reg(rn)).o(reg(rm)).o(reg(ra)),
        (0, 1) => w("mls").o(reg(rd)).o(reg(rn)).o(reg(rm)).o(reg(ra)),
        _ => return None,
    })
}

/// Long multiply, long multiply-accumulate, and divide (A5.3.17).
fn long_multiply(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    let (rn, lo, hi, rm) = (hw1 & 15, hw2 >> 12, (hw2 >> 8) & 15, hw2 & 15);
    let long = |name: &str| w(name).o(reg(lo)).o(reg(hi)).o(reg(rn)).o(reg(rm));
    Some(match ((hw1 >> 4) & 7, (hw2 >> 4) & 15) {
        (0, 0) => long("smull"),
        (2, 0) => long("umull"),
        (4, 0) => long("smlal"),
        (6, 0) => long("umlal"),
        (1, 15) if lo == 15 => w("sdiv").o(reg(hi)).o(reg(rn)).o(reg(rm)),
        (3, 15) if lo == 15 => w("udiv").o(reg(hi)).o(reg(rn)).o(reg(rm)),
        _ => return None,
    })
}

/// The coprocessor instructions (A5.3.18): `ldc`/`stc`, `mcrr`/`mrrc`,
/// `cdp`, `mcr`/`mrc`, and their `2` forms (first halfword `0xfc..`-`0xff..`).
fn coprocessor(hw1: u32, hw2: u32) -> Option<ThumbInst> {
    let two = if hw1 & 0x1000 != 0 { "2" } else { "" };
    let op1 = (hw1 >> 4) & 0x3f;
    let coproc = (hw2 >> 8) & 15;
    // Coprocessors 10 and 11 are the floating-point extension's space;
    // only its unconditional, unindexed long `ldc2l`/`stc2l` forms remain
    // generic coprocessor instructions.
    let unindexed_long = two == "2" && op1 & 0b10_0000 == 0 && hw1 & 0x1e0 == 0x0c0;
    if (coproc == 10 || coproc == 11) && !unindexed_long {
        return None;
    }
    let cp = format!("p{coproc}");
    let c = |n: u32| format!("c{}", n & 15);
    let rn = hw1 & 15;
    let crd = hw2 >> 12;
    match op1 {
        0b00_0000 | 0b00_0001 => None,
        0b00_0100 | 0b00_0101 => {
            let name = if op1 & 1 == 0 { "mcrr" } else { "mrrc" };
            Some(w(&format!("{name}{two}")).o(cp).o(imm(i64::from((hw2 >> 4) & 15))).o(reg(crd)).o(reg(rn)).o(c(hw2)))
        }
        _ if op1 & 0b10_0000 == 0 => {
            let (p, u, d, wb, load) = (hw1 >> 8 & 1, hw1 >> 7 & 1, hw1 >> 6 & 1, hw1 >> 5 & 1, hw1 >> 4 & 1);
            let name = format!("{}{two}{}", if load == 1 { "ldc" } else { "stc" }, if d == 1 { "l" } else { "" });
            let v = i64::from(hw2 & 0xff);
            let off = if u == 1 { v * 4 } else { -v * 4 };
            let addr = match (p, wb) {
                (1, 0) if v == 0 && u == 1 => format!("[{}]", reg(rn)),
                (1, 0) if v == 0 => format!("[{}, #-0x0]", reg(rn)),
                (1, 0) => format!("[{}, {}]", reg(rn), imm(off)),
                (1, _) => format!("[{}, {}]!", reg(rn), imm(off)),
                (0, 1) => format!("[{}], {}", reg(rn), imm(off)),
                _ => format!("[{}], {{{v}}}", reg(rn)),
            };
            Some(w(&name).o(cp).o(c(crd)).o(addr))
        }
        _ if op1 & 0b11_0000 == 0b10_0000 => {
            let opc2 = imm(i64::from((hw2 >> 5) & 7));
            if hw2 & 0x10 == 0 {
                Some(w(&format!("cdp{two}")).o(cp).o(imm(i64::from((hw1 >> 4) & 15))).o(c(crd)).o(c(rn)).o(c(hw2)).o(opc2))
            } else {
                let load = op1 & 1 == 1;
                let rt = if load && crd == 15 { "apsr_nzcv" } else { reg(crd) };
                let name = if load { "mrc" } else { "mcr" };
                Some(w(&format!("{name}{two}")).o(cp).o(imm(i64::from((hw1 >> 5) & 7))).o(rt).o(c(rn)).o(c(hw2)).o(opc2))
            }
        }
        _ => None,
    }
}
