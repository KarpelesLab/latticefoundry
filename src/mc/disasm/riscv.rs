//! The RISC-V decoder: RV64GC — the base integer ISA (RV64I), M, A, F, D,
//! Zicsr, Zifencei, the compressed C extension, and the Zba/Zbb bit
//! manipulation extensions — written from *The RISC-V Instruction Set Manual*
//! (Volume I: Unprivileged ISA; Volume II for the privileged `sret`/`mret`/
//! `wfi`/`sfence.vma` and the CSR names).
//!
//! Decoding has two steps. [`decode_inst`] turns bytes into a typed
//! [`RvInst`]: the *real* instruction's mnemonic and operands, a compressed
//! instruction already expanded to the 32-bit instruction it stands for (the
//! manual defines every C instruction that way). [`RvInst::canonical`] then
//! applies the manual's pseudoinstructions (`li`, `mv`, `not`, `neg`,
//! `sext.w`, `seqz`, `beqz`, `j`, `ret`, `csrr`, `frflags`, `fmv.d`, ...),
//! which is how [`decode`] prints. Immediates print in hex, as llvm-objdump
//! does; a floating-point rounding mode prints only when it is not the
//! instruction's default (`dyn`, or `rne` for the exact conversions).
//!
//! Each major opcode is one match over its `funct3`/`funct7` fields, so a new
//! instruction is one more arm.

use super::{Inst, sext};

/// The ABI names of the integer registers.
pub const XREGS: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
];

/// The ABI names of the floating-point registers.
pub const FREGS: [&str; 32] = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
    "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
];

/// The rounding-mode names (`frm` field values 0–4 and 7; 5 and 6 are
/// reserved).
const RM: [&str; 8] = ["rne", "rtz", "rdn", "rup", "rmm", "", "", "dyn"];

const ZERO: u8 = 0;
const RA: u8 = 1;
const SP: u8 = 2;

/// One operand of a decoded instruction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Operand {
    /// An integer register.
    X(u8),
    /// A floating-point register.
    F(u8),
    /// An immediate (printed in hex, signed).
    Imm(i64),
    /// A memory operand `offset(base)` (loads, stores, `jalr`).
    Mem(u8, i64),
    /// The address register of an atomic: `(base)`.
    Addr(u8),
    /// An absolute branch or jump target.
    Target(u64),
    /// A CSR number (printed by name when it has one).
    Csr(u16),
    /// A floating-point rounding mode (`frm` field).
    Rm(u8),
    /// A `fence` predecessor or successor set (`iorw` bits).
    Fence(u8),
}

/// A decoded instruction: its length (2 for a compressed one, else 4), its
/// mnemonic and operands in assembly order.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RvInst {
    /// 2 or 4.
    pub len: usize,
    /// The mnemonic, with any `.aq`/`.rl` ordering suffix.
    pub mnemonic: String,
    /// The operands.
    pub ops: Vec<Operand>,
}

impl RvInst {
    fn new(len: usize, mnemonic: impl Into<String>, ops: Vec<Operand>) -> RvInst {
        RvInst { len, mnemonic: mnemonic.into(), ops }
    }

    /// The instruction rewritten into the pseudoinstruction the manual (and
    /// llvm-objdump) prefers for it, if any; otherwise unchanged.
    pub fn canonical(&self) -> RvInst {
        use Operand::{Csr, F, Imm, Mem, Target, X};
        let m = self.mnemonic.as_str();
        let ops = &self.ops[..];
        let p = |mn: &str, o: Vec<Operand>| Some(RvInst::new(self.len, mn, o));
        let alias = match (m, ops) {
            ("addi", [X(0), X(0), Imm(0)]) => p("nop", vec![]),
            ("addi", [X(d), X(0), Imm(i)]) => p("li", vec![X(*d), Imm(*i)]),
            ("addi", [X(d), X(s), Imm(0)]) => p("mv", vec![X(*d), X(*s)]),
            ("addiw", [X(d), X(s), Imm(0)]) => p("sext.w", vec![X(*d), X(*s)]),
            ("xori", [X(d), X(s), Imm(-1)]) => p("not", vec![X(*d), X(*s)]),
            ("andi", [X(d), X(s), Imm(255)]) => p("zext.b", vec![X(*d), X(*s)]),
            ("sltiu", [X(d), X(s), Imm(1)]) => p("seqz", vec![X(*d), X(*s)]),
            ("sub", [X(d), X(0), X(s)]) => p("neg", vec![X(*d), X(*s)]),
            ("subw", [X(d), X(0), X(s)]) => p("negw", vec![X(*d), X(*s)]),
            ("sltu", [X(d), X(0), X(s)]) => p("snez", vec![X(*d), X(*s)]),
            ("slt", [X(d), X(s), X(0)]) => p("sltz", vec![X(*d), X(*s)]),
            ("slt", [X(d), X(0), X(s)]) => p("sgtz", vec![X(*d), X(*s)]),
            ("add.uw", [X(d), X(s), X(0)]) => p("zext.w", vec![X(*d), X(*s)]),
            ("beq", [X(s), X(0), t]) => p("beqz", vec![X(*s), *t]),
            ("bne", [X(s), X(0), t]) => p("bnez", vec![X(*s), *t]),
            ("bge", [X(0), X(s), t]) => p("blez", vec![X(*s), *t]),
            ("bge", [X(s), X(0), t]) => p("bgez", vec![X(*s), *t]),
            ("blt", [X(s), X(0), t]) => p("bltz", vec![X(*s), *t]),
            ("blt", [X(0), X(s), t]) => p("bgtz", vec![X(*s), *t]),
            ("jal", [X(0), t @ Target(_)]) => p("j", vec![*t]),
            ("jal", [X(1), t @ Target(_)]) => p("jal", vec![*t]),
            ("jalr", [X(0), Mem(1, 0)]) => p("ret", vec![]),
            ("jalr", [X(0), Mem(s, 0)]) => p("jr", vec![X(*s)]),
            ("jalr", [X(1), Mem(s, 0)]) => p("jalr", vec![X(*s)]),
            ("jalr", [X(0), mem @ Mem(..)]) => p("jr", vec![*mem]),
            ("jalr", [X(1), mem @ Mem(..)]) => p("jalr", vec![*mem]),
            ("fsgnj.s" | "fsgnj.d", [F(d), F(a), F(b)]) if a == b => p(&format!("fmv.{}", &m[6..]), vec![F(*d), F(*a)]),
            ("fsgnjn.s" | "fsgnjn.d", [F(d), F(a), F(b)]) if a == b => p(&format!("fneg.{}", &m[7..]), vec![F(*d), F(*a)]),
            ("fsgnjx.s" | "fsgnjx.d", [F(d), F(a), F(b)]) if a == b => p(&format!("fabs.{}", &m[7..]), vec![F(*d), F(*a)]),
            ("csrrs", [X(d), Csr(c), X(0)]) => match c {
                0x001 => p("frflags", vec![X(*d)]),
                0x002 => p("frrm", vec![X(*d)]),
                0x003 => p("frcsr", vec![X(*d)]),
                0xc00 => p("rdcycle", vec![X(*d)]),
                0xc01 => p("rdtime", vec![X(*d)]),
                0xc02 => p("rdinstret", vec![X(*d)]),
                _ => p("csrr", vec![X(*d), Csr(*c)]),
            },
            ("csrrw", [X(d), Csr(c @ 1..=3), X(s)]) => {
                let name = ["", "fsflags", "fsrm", "fscsr"][*c as usize];
                if *d == 0 { p(name, vec![X(*s)]) } else { p(name, vec![X(*d), X(*s)]) }
            }
            ("csrrwi", [X(d), Csr(c @ 1..=2), i]) => {
                let name = ["", "fsflagsi", "fsrmi"][*c as usize];
                if *d == 0 { p(name, vec![*i]) } else { p(name, vec![X(*d), *i]) }
            }
            ("csrrw" | "csrrs" | "csrrc" | "csrrwi" | "csrrsi" | "csrrci", [X(0), Csr(c), s]) => {
                let short = format!("csr{}", &m[4..]);
                p(&short, vec![Csr(*c), *s])
            }
            _ => None,
        };
        alias.unwrap_or_else(|| self.clone())
    }

    /// Render as an [`Inst`] (operands in the manual's syntax, immediates in
    /// hex, a non-default rounding mode as a trailing operand).
    pub fn to_inst(&self) -> Inst {
        let mut inst = Inst::new(self.len, self.mnemonic.clone());
        let default_rm = if matches!(self.mnemonic.as_str(), "fcvt.d.w" | "fcvt.d.wu" | "fcvt.d.s") { 0 } else { 7 };
        for op in &self.ops {
            inst = match *op {
                Operand::X(r) => inst.op(XREGS[usize::from(r & 31)]),
                Operand::F(r) => inst.op(FREGS[usize::from(r & 31)]),
                Operand::Imm(v) => inst.op(hex(v)),
                Operand::Mem(b, off) => inst.op(format!("{}({})", hex(off), XREGS[usize::from(b & 31)])),
                Operand::Addr(b) => inst.op(format!("({})", XREGS[usize::from(b & 31)])),
                Operand::Target(t) => inst.target_op(t),
                Operand::Csr(c) => inst.op(csr_name(c).unwrap_or_else(|| format!("{c:#x}"))),
                Operand::Rm(rm) if rm == default_rm => inst,
                Operand::Rm(rm) => inst.op(RM[usize::from(rm & 7)]),
                Operand::Fence(set) => inst.op(fence_set(set)),
            };
        }
        inst
    }
}

/// A signed value in hex, as llvm-objdump prints RISC-V immediates.
fn hex(v: i64) -> String {
    if v < 0 { format!("-{:#x}", v.unsigned_abs()) } else { format!("{v:#x}") }
}

/// A `fence` set: some of `i`, `o`, `r`, `w`, or `0` for none.
fn fence_set(set: u8) -> String {
    let s: String = [(8, 'i'), (4, 'o'), (2, 'r'), (1, 'w')].iter().filter(|(b, _)| set & b != 0).map(|&(_, c)| c).collect();
    if s.is_empty() { "0".to_owned() } else { s }
}

/// The standard name of a CSR, from the privileged architecture's CSR
/// listing (unprivileged floating-point, vector and counter CSRs;
/// supervisor, hypervisor/virtual-supervisor, machine and debug CSRs). The
/// RV32-only halves (`cycleh`, `mstatush`, odd `pmpcfg`s, ...) have no name
/// on RV64.
pub fn csr_name(csr: u16) -> Option<String> {
    if let Some((family, k)) = csr_family(csr) {
        return Some(format!("{family}{k}"));
    }
    named_csr(csr).map(str::to_owned)
}

/// A CSR of a numbered family, as `(family, index)`.
pub fn csr_family(csr: u16) -> Option<(&'static str, u16)> {
    match csr {
        0xc03..=0xc1f => Some(("hpmcounter", csr - 0xc00)),
        0xb03..=0xb1f => Some(("mhpmcounter", csr - 0xb00)),
        0x323..=0x33f => Some(("mhpmevent", csr - 0x320)),
        0x3a0..=0x3af if csr.is_multiple_of(2) => Some(("pmpcfg", csr - 0x3a0)),
        0x3b0..=0x3ef => Some(("pmpaddr", csr - 0x3b0)),
        _ => None,
    }
}

fn named_csr(csr: u16) -> Option<&'static str> {
    Some(match csr {
        // Unprivileged.
        0x001 => "fflags",
        0x002 => "frm",
        0x003 => "fcsr",
        0x008 => "vstart",
        0x009 => "vxsat",
        0x00a => "vxrm",
        0x00f => "vcsr",
        0x011 => "ssp",
        0x015 => "seed",
        // Indirect CSR access (Smcsrind/Sscsrind), counter delegation
        // (Smcdeleg), control-transfer records (Smctr), QoS IDs (Ssqosid)
        // and counter filtering (Smcntrpmf).
        0x152 => "sireg2",
        0x153 => "sireg3",
        0x155 => "sireg4",
        0x156 => "sireg5",
        0x157 => "sireg6",
        0x252 => "vsireg2",
        0x253 => "vsireg3",
        0x255 => "vsireg4",
        0x256 => "vsireg5",
        0x257 => "vsireg6",
        0x352 => "mireg2",
        0x353 => "mireg3",
        0x355 => "mireg4",
        0x356 => "mireg5",
        0x357 => "mireg6",
        0x120 => "scountinhibit",
        0x14e => "sctrctl",
        0x14f => "sctrstatus",
        0x15f => "sctrdepth",
        0x24e => "vsctrctl",
        0x34e => "mctrctl",
        0x181 => "srmcfg",
        0x321 => "mcyclecfg",
        0x322 => "minstretcfg",
        0x017 => "jvt",
        0xc00 => "cycle",
        0xc01 => "time",
        0xc02 => "instret",
        0xc20 => "vl",
        0xc21 => "vtype",
        0xc22 => "vlenb",
        // Supervisor.
        0x100 => "sstatus",
        0x104 => "sie",
        0x105 => "stvec",
        0x106 => "scounteren",
        0x10a => "senvcfg",
        0x10c => "sstateen0",
        0x10d => "sstateen1",
        0x10e => "sstateen2",
        0x10f => "sstateen3",
        0x140 => "sscratch",
        0x141 => "sepc",
        0x142 => "scause",
        0x143 => "stval",
        0x144 => "sip",
        0x14d => "stimecmp",
        0x150 => "siselect",
        0x151 => "sireg",
        0x15c => "stopei",
        0x180 => "satp",
        0x5a8 => "scontext",
        0xda0 => "scountovf",
        0xdb0 => "stopi",
        // Hypervisor and virtual supervisor.
        0x600 => "hstatus",
        0x602 => "hedeleg",
        0x603 => "hideleg",
        0x604 => "hie",
        0x605 => "htimedelta",
        0x606 => "hcounteren",
        0x607 => "hgeie",
        0x608 => "hvien",
        0x609 => "hvictl",
        0x60a => "henvcfg",
        0x60c => "hstateen0",
        0x60d => "hstateen1",
        0x60e => "hstateen2",
        0x60f => "hstateen3",
        0x643 => "htval",
        0x644 => "hip",
        0x645 => "hvip",
        0x646 => "hviprio1",
        0x647 => "hviprio2",
        0x64a => "htinst",
        0x680 => "hgatp",
        0x6a8 => "hcontext",
        0xe12 => "hgeip",
        0x200 => "vsstatus",
        0x204 => "vsie",
        0x205 => "vstvec",
        0x240 => "vsscratch",
        0x241 => "vsepc",
        0x242 => "vscause",
        0x243 => "vstval",
        0x244 => "vsip",
        0x24d => "vstimecmp",
        0x250 => "vsiselect",
        0x251 => "vsireg",
        0x25c => "vstopei",
        0x280 => "vsatp",
        0xeb0 => "vstopi",
        // Machine.
        0xf11 => "mvendorid",
        0xf12 => "marchid",
        0xf13 => "mimpid",
        0xf14 => "mhartid",
        0xf15 => "mconfigptr",
        0x300 => "mstatus",
        0x301 => "misa",
        0x302 => "medeleg",
        0x303 => "mideleg",
        0x304 => "mie",
        0x305 => "mtvec",
        0x306 => "mcounteren",
        0x308 => "mvien",
        0x309 => "mvip",
        0x30a => "menvcfg",
        0x30c => "mstateen0",
        0x30d => "mstateen1",
        0x30e => "mstateen2",
        0x30f => "mstateen3",
        0x320 => "mcountinhibit",
        0x340 => "mscratch",
        0x341 => "mepc",
        0x342 => "mcause",
        0x343 => "mtval",
        0x344 => "mip",
        0x34a => "mtinst",
        0x34b => "mtval2",
        0x350 => "miselect",
        0x351 => "mireg",
        0x35c => "mtopei",
        0x740 => "mnscratch",
        0x741 => "mnepc",
        0x742 => "mncause",
        0x744 => "mnstatus",
        0x747 => "mseccfg",
        0xb00 => "mcycle",
        0xb02 => "minstret",
        0xfb0 => "mtopi",
        // Debug and trigger.
        0x7a0 => "tselect",
        0x7a1 => "tdata1",
        0x7a2 => "tdata2",
        0x7a3 => "tdata3",
        0x7a4 => "tinfo",
        0x7a5 => "tcontrol",
        0x7a8 => "mcontext",
        0x7aa => "mscontext",
        0x7b0 => "dcsr",
        0x7b1 => "dpc",
        0x7b2 => "dscratch0",
        0x7b3 => "dscratch1",
        _ => return None,
    })
}

/// Decode one RISC-V instruction from the start of `bytes` (non-empty),
/// located at address `addr`: a 16-bit compressed instruction when the low
/// two bits are not `11`, else a 32-bit one. Longer encodings and unknown
/// ones become `.short`/`.word` data.
pub fn decode(bytes: &[u8], addr: u64) -> Inst {
    match decode_inst(bytes, addr) {
        Some(i) => i.canonical().to_inst(),
        None => {
            let unit = if bytes.len() >= 4 && bytes[0] & 3 == 3 && bytes[0] & 0x1c != 0x1c { 4 } else { 2 };
            Inst::data(bytes, unit, true)
        }
    }
}

/// Decode the instruction at the start of `bytes` (at `addr`) into its
/// typed form, a compressed instruction expanded; `None` for an unknown,
/// reserved or truncated encoding.
pub fn decode_inst(bytes: &[u8], addr: u64) -> Option<RvInst> {
    let lo = u16::from_le_bytes([*bytes.first()?, *bytes.get(1)?]);
    if lo & 3 != 3 {
        return decode16(lo, addr);
    }
    if lo & 0x1c == 0x1c {
        return None; // 48-bit and longer encodings
    }
    let hi = u16::from_le_bytes([*bytes.get(2)?, *bytes.get(3)?]);
    decode32(u32::from(lo) | u32::from(hi) << 16, addr)
}

/// Bits `hi..=lo` of `w`.
fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & ((1 << (hi - lo + 1)) - 1)
}

fn decode32(w: u32, addr: u64) -> Option<RvInst> {
    use Operand::{Addr, Csr, F, Fence, Imm, Mem, Rm, Target, X};
    let rd = bits(w, 11, 7) as u8;
    let rs1 = bits(w, 19, 15) as u8;
    let rs2 = bits(w, 24, 20) as u8;
    let rs3 = bits(w, 31, 27) as u8;
    let f3 = bits(w, 14, 12);
    let f7 = bits(w, 31, 25);
    let imm_i = sext(u64::from(w >> 20), 12);
    let imm_s = sext(u64::from(bits(w, 31, 25) << 5 | bits(w, 11, 7)), 12);
    let i = |m: &str, ops: Vec<Operand>| Some(RvInst::new(4, m, ops));
    match w & 0x7f {
        0x03 => {
            let m = ["lb", "lh", "lw", "ld", "lbu", "lhu", "lwu"].get(f3 as usize)?;
            i(m, vec![X(rd), Mem(rs1, imm_i)])
        }
        0x07 => {
            let m = match f3 {
                2 => "flw",
                3 => "fld",
                _ => return None,
            };
            i(m, vec![F(rd), Mem(rs1, imm_i)])
        }
        0x0f => match f3 {
            0 if rd == 0 && rs1 == 0 => {
                let fm = bits(w, 31, 28);
                let (pred, succ) = (bits(w, 27, 24) as u8, bits(w, 23, 20) as u8);
                match (fm, pred, succ) {
                    (8, 3, 3) => i("fence.tso", vec![]),
                    (0, 15, 15) => i("fence", vec![]),
                    (0, ..) => i("fence", vec![Fence(pred), Fence(succ)]),
                    _ => None,
                }
            }
            1 if rd == 0 && rs1 == 0 && w >> 20 == 0 => i("fence.i", vec![]),
            _ => None,
        },
        0x13 => {
            let shamt = bits(w, 25, 20);
            match f3 {
                0 => i("addi", vec![X(rd), X(rs1), Imm(imm_i)]),
                2 => i("slti", vec![X(rd), X(rs1), Imm(imm_i)]),
                3 => i("sltiu", vec![X(rd), X(rs1), Imm(imm_i)]),
                4 => i("xori", vec![X(rd), X(rs1), Imm(imm_i)]),
                6 => i("ori", vec![X(rd), X(rs1), Imm(imm_i)]),
                7 => i("andi", vec![X(rd), X(rs1), Imm(imm_i)]),
                1 => match bits(w, 31, 26) {
                    0 => i("slli", vec![X(rd), X(rs1), Imm(i64::from(shamt))]),
                    0x18 => {
                        let m = ["clz", "ctz", "cpop", "", "sext.b", "sext.h"].get(rs2 as usize).filter(|m| !m.is_empty())?;
                        if f7 != 0x30 {
                            return None;
                        }
                        i(m, vec![X(rd), X(rs1)])
                    }
                    _ => None,
                },
                5 => match (bits(w, 31, 26), w >> 20) {
                    (0, _) => i("srli", vec![X(rd), X(rs1), Imm(i64::from(shamt))]),
                    (0x10, _) => i("srai", vec![X(rd), X(rs1), Imm(i64::from(shamt))]),
                    (0x18, _) => i("rori", vec![X(rd), X(rs1), Imm(i64::from(shamt))]),
                    (_, 0x287) => i("orc.b", vec![X(rd), X(rs1)]),
                    (_, 0x6b8) => i("rev8", vec![X(rd), X(rs1)]),
                    _ => None,
                },
                _ => None,
            }
        }
        0x17 => i("auipc", vec![X(rd), Imm(i64::from(w >> 12))]),
        0x1b => {
            let shamt = i64::from(bits(w, 24, 20));
            match (f3, f7) {
                (0, _) => i("addiw", vec![X(rd), X(rs1), Imm(imm_i)]),
                (1, 0) => i("slliw", vec![X(rd), X(rs1), Imm(shamt)]),
                (1, 0x30) => {
                    let m = ["clzw", "ctzw", "cpopw"].get(rs2 as usize)?;
                    i(m, vec![X(rd), X(rs1)])
                }
                (1, 0x04 | 0x05) => i("slli.uw", vec![X(rd), X(rs1), Imm(i64::from(bits(w, 25, 20)))]),
                (5, 0) => i("srliw", vec![X(rd), X(rs1), Imm(shamt)]),
                (5, 0x20) => i("sraiw", vec![X(rd), X(rs1), Imm(shamt)]),
                (5, 0x30) => i("roriw", vec![X(rd), X(rs1), Imm(shamt)]),
                _ => None,
            }
        }
        0x23 => {
            let m = ["sb", "sh", "sw", "sd"].get(f3 as usize)?;
            i(m, vec![X(rs2), Mem(rs1, imm_s)])
        }
        0x27 => {
            let m = match f3 {
                2 => "fsw",
                3 => "fsd",
                _ => return None,
            };
            i(m, vec![F(rs2), Mem(rs1, imm_s)])
        }
        0x2f => {
            let width = match f3 {
                2 => "w",
                3 => "d",
                _ => return None,
            };
            let order = match bits(w, 26, 25) {
                0 => "",
                1 => ".rl",
                2 => ".aq",
                _ => ".aqrl",
            };
            let base = match bits(w, 31, 27) {
                0b00010 => {
                    if rs2 != 0 {
                        return None;
                    }
                    return i(&format!("lr.{width}{order}"), vec![X(rd), Addr(rs1)]);
                }
                0b00011 => "sc",
                0b00001 => "amoswap",
                0b00000 => "amoadd",
                0b00100 => "amoxor",
                0b01100 => "amoand",
                0b01000 => "amoor",
                0b10000 => "amomin",
                0b10100 => "amomax",
                0b11000 => "amominu",
                0b11100 => "amomaxu",
                _ => return None,
            };
            i(&format!("{base}.{width}{order}"), vec![X(rd), X(rs2), Addr(rs1)])
        }
        0x33 => {
            let m = match (f7, f3) {
                (0x00, _) => ["add", "sll", "slt", "sltu", "xor", "srl", "or", "and"][f3 as usize],
                (0x20, 0) => "sub",
                (0x20, 5) => "sra",
                (0x20, 4) => "xnor",
                (0x20, 6) => "orn",
                (0x20, 7) => "andn",
                (0x01, _) => ["mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu"][f3 as usize],
                (0x10, 2) => "sh1add",
                (0x10, 4) => "sh2add",
                (0x10, 6) => "sh3add",
                (0x05, 4) => "min",
                (0x05, 5) => "minu",
                (0x05, 6) => "max",
                (0x05, 7) => "maxu",
                (0x30, 1) => "rol",
                (0x30, 5) => "ror",
                _ => return None,
            };
            i(m, vec![X(rd), X(rs1), X(rs2)])
        }
        0x37 => i("lui", vec![X(rd), Imm(i64::from(w >> 12))]),
        0x3b => {
            let m = match (f7, f3) {
                (0x00, 0) => "addw",
                (0x20, 0) => "subw",
                (0x00, 1) => "sllw",
                (0x00, 5) => "srlw",
                (0x20, 5) => "sraw",
                (0x01, 0) => "mulw",
                (0x01, 4) => "divw",
                (0x01, 5) => "divuw",
                (0x01, 6) => "remw",
                (0x01, 7) => "remuw",
                (0x04, 0) => "add.uw",
                (0x04, 4) if rs2 == 0 => return i("zext.h", vec![X(rd), X(rs1)]),
                (0x10, 2) => "sh1add.uw",
                (0x10, 4) => "sh2add.uw",
                (0x10, 6) => "sh3add.uw",
                (0x30, 1) => "rolw",
                (0x30, 5) => "rorw",
                _ => return None,
            };
            i(m, vec![X(rd), X(rs1), X(rs2)])
        }
        op @ (0x43 | 0x47 | 0x4b | 0x4f) => {
            let fmt = match bits(w, 26, 25) {
                0 => "s",
                1 => "d",
                _ => return None,
            };
            let base = match op {
                0x43 => "fmadd",
                0x47 => "fmsub",
                0x4b => "fnmsub",
                _ => "fnmadd",
            };
            let rm = valid_rm(f3)?;
            i(&format!("{base}.{fmt}"), vec![F(rd), F(rs1), F(rs2), F(rs3), Rm(rm)])
        }
        0x53 => {
            let fmt = match f7 & 3 {
                0 => "s",
                1 => "d",
                _ => return None,
            };
            let other = if fmt == "s" { "d" } else { "s" };
            match f7 >> 2 {
                0x00..=0x03 => {
                    let base = ["fadd", "fsub", "fmul", "fdiv"][(f7 >> 2) as usize];
                    i(&format!("{base}.{fmt}"), vec![F(rd), F(rs1), F(rs2), Rm(valid_rm(f3)?)])
                }
                0x0b if rs2 == 0 => i(&format!("fsqrt.{fmt}"), vec![F(rd), F(rs1), Rm(valid_rm(f3)?)]),
                0x04 => {
                    let base = ["fsgnj", "fsgnjn", "fsgnjx"].get(f3 as usize)?;
                    i(&format!("{base}.{fmt}"), vec![F(rd), F(rs1), F(rs2)])
                }
                0x05 => {
                    let base = ["fmin", "fmax"].get(f3 as usize)?;
                    i(&format!("{base}.{fmt}"), vec![F(rd), F(rs1), F(rs2)])
                }
                // fcvt.s.d (rs2 = 1, the source format) / fcvt.d.s (rs2 = 0).
                0x08 if u32::from(rs2) == (f7 & 3) ^ 1 => {
                    i(&format!("fcvt.{fmt}.{other}"), vec![F(rd), F(rs1), Rm(valid_rm(f3)?)])
                }
                0x14 => {
                    let base = ["fle", "flt", "feq"].get(f3 as usize)?;
                    i(&format!("{base}.{fmt}"), vec![X(rd), F(rs1), F(rs2)])
                }
                0x18 => {
                    let int = ["w", "wu", "l", "lu"].get(rs2 as usize)?;
                    i(&format!("fcvt.{int}.{fmt}"), vec![X(rd), F(rs1), Rm(valid_rm(f3)?)])
                }
                0x1a => {
                    let int = ["w", "wu", "l", "lu"].get(rs2 as usize)?;
                    i(&format!("fcvt.{fmt}.{int}"), vec![F(rd), X(rs1), Rm(valid_rm(f3)?)])
                }
                0x1c if rs2 == 0 => match f3 {
                    0 => i(&format!("fmv.x.{}", if fmt == "s" { "w" } else { "d" }), vec![X(rd), F(rs1)]),
                    1 => i(&format!("fclass.{fmt}"), vec![X(rd), F(rs1)]),
                    _ => None,
                },
                0x1e if rs2 == 0 && f3 == 0 => {
                    i(&format!("fmv.{}.x", if fmt == "s" { "w" } else { "d" }), vec![F(rd), X(rs1)])
                }
                _ => None,
            }
        }
        0x63 => {
            let m = match f3 {
                0 => "beq",
                1 => "bne",
                4 => "blt",
                5 => "bge",
                6 => "bltu",
                7 => "bgeu",
                _ => return None,
            };
            let off = sext(
                u64::from(bits(w, 31, 31) << 12 | bits(w, 7, 7) << 11 | bits(w, 30, 25) << 5 | bits(w, 11, 8) << 1),
                13,
            );
            i(m, vec![X(rs1), X(rs2), Target(addr.wrapping_add(off as u64))])
        }
        0x67 if f3 == 0 => i("jalr", vec![X(rd), Mem(rs1, imm_i)]),
        0x6f => {
            let off = sext(
                u64::from(bits(w, 31, 31) << 20 | bits(w, 19, 12) << 12 | bits(w, 20, 20) << 11 | bits(w, 30, 21) << 1),
                21,
            );
            i("jal", vec![X(rd), Target(addr.wrapping_add(off as u64))])
        }
        0x73 => match f3 {
            0 => {
                if f7 == 0x09 && rd == 0 {
                    return match (rs1, rs2) {
                        (0, 0) => i("sfence.vma", vec![]),
                        (a, 0) => i("sfence.vma", vec![X(a)]),
                        (a, b) => i("sfence.vma", vec![X(a), X(b)]),
                    };
                }
                if rd != 0 || rs1 != 0 {
                    return None;
                }
                match w >> 20 {
                    0x000 => i("ecall", vec![]),
                    0x001 => i("ebreak", vec![]),
                    0x102 => i("sret", vec![]),
                    0x302 => i("mret", vec![]),
                    0x105 => i("wfi", vec![]),
                    _ => None,
                }
            }
            4 => None,
            _ => {
                let csr = (w >> 20) as u16;
                let m = ["", "csrrw", "csrrs", "csrrc", "", "csrrwi", "csrrsi", "csrrci"][f3 as usize];
                let src = if f3 >= 5 { Imm(i64::from(rs1)) } else { X(rs1) };
                i(m, vec![X(rd), Csr(csr), src])
            }
        },
        _ => None,
    }
}

/// A rounding-mode field, if it is not one of the reserved values (5, 6).
fn valid_rm(rm: u32) -> Option<u8> {
    (rm != 5 && rm != 6).then_some(rm as u8)
}

/// Bits `hi..=lo` of a 16-bit instruction.
fn cb(h: u16, hi: u32, lo: u32) -> u32 {
    bits(u32::from(h), hi, lo)
}

/// Decode a compressed instruction into the 32-bit instruction it expands
/// to (RVC chapter of the unprivileged manual; RV64C). Reserved encodings
/// are `None`; hints decode as their expansion.
fn decode16(h: u16, addr: u64) -> Option<RvInst> {
    use Operand::{F, Imm, Mem, Target, X};
    let i = |m: &str, ops: Vec<Operand>| Some(RvInst::new(2, m, ops));
    // The popular registers x8–x15 of the 3-bit fields.
    let rd_p = (cb(h, 4, 2) + 8) as u8;
    let rs1_p = (cb(h, 9, 7) + 8) as u8;
    let rd = cb(h, 11, 7) as u8;
    let rs2 = cb(h, 6, 2) as u8;
    let imm6 = sext(u64::from(cb(h, 12, 12) << 5 | cb(h, 6, 2)), 6);
    let f3 = cb(h, 15, 13);
    // Offsets of the register-based loads and stores.
    let off_w = i64::from(cb(h, 12, 10) << 3 | cb(h, 6, 6) << 2 | cb(h, 5, 5) << 6);
    let off_d = i64::from(cb(h, 12, 10) << 3 | cb(h, 6, 5) << 6);
    match (h & 3, f3) {
        (0, 0) => {
            let nzuimm = cb(h, 12, 11) << 4 | cb(h, 10, 7) << 6 | cb(h, 6, 6) << 2 | cb(h, 5, 5) << 3;
            if nzuimm == 0 {
                // All zeros is the defined illegal instruction (`unimp`);
                // any other zero immediate is reserved.
                return (h == 0).then(|| RvInst::new(2, "unimp", vec![]));
            }
            i("addi", vec![X(rd_p), X(SP), Imm(i64::from(nzuimm))])
        }
        (0, 1) => i("fld", vec![F(rd_p), Mem(rs1_p, off_d)]),
        (0, 2) => i("lw", vec![X(rd_p), Mem(rs1_p, off_w)]),
        (0, 3) => i("ld", vec![X(rd_p), Mem(rs1_p, off_d)]),
        (0, 5) => i("fsd", vec![F(rd_p), Mem(rs1_p, off_d)]),
        (0, 6) => i("sw", vec![X(rd_p), Mem(rs1_p, off_w)]),
        (0, 7) => i("sd", vec![X(rd_p), Mem(rs1_p, off_d)]),
        // HINTs (no architectural effect) print in their compressed form.
        (1, 0) if rd == 0 && imm6 != 0 => i("c.nop", vec![Imm(imm6)]),
        (1, 0) if rd != 0 && imm6 == 0 => i("c.addi", vec![X(rd), Imm(0)]),
        (1, 0) => i("addi", vec![X(rd), X(rd), Imm(imm6)]),
        (1, 1) if rd != 0 => i("addiw", vec![X(rd), X(rd), Imm(imm6)]),
        (1, 2) if rd == 0 => i("c.li", vec![X(ZERO), Imm(imm6)]),
        (1, 2) => i("addi", vec![X(rd), X(ZERO), Imm(imm6)]),
        (1, 3) if rd == SP => {
            let nzimm = sext(
                u64::from(
                    cb(h, 12, 12) << 9 | cb(h, 6, 6) << 4 | cb(h, 5, 5) << 6 | cb(h, 4, 3) << 7 | cb(h, 2, 2) << 5,
                ),
                10,
            );
            if nzimm == 0 {
                return None;
            }
            i("addi", vec![X(SP), X(SP), Imm(nzimm)])
        }
        (1, 3) => {
            if imm6 == 0 {
                return None;
            }
            let m = if rd == 0 { "c.lui" } else { "lui" };
            i(m, vec![X(rd), Imm(imm6 & 0xfffff)])
        }
        (1, 4) => {
            let shamt = i64::from(cb(h, 12, 12) << 5 | cb(h, 6, 2));
            match cb(h, 11, 10) {
                0 if shamt == 0 => i("c.srli", vec![X(rs1_p), Imm(0)]),
                1 if shamt == 0 => i("c.srai", vec![X(rs1_p), Imm(0)]),
                0 => i("srli", vec![X(rs1_p), X(rs1_p), Imm(shamt)]),
                1 => i("srai", vec![X(rs1_p), X(rs1_p), Imm(shamt)]),
                2 => i("andi", vec![X(rs1_p), X(rs1_p), Imm(imm6)]),
                _ => {
                    let m = match (cb(h, 12, 12), cb(h, 6, 5)) {
                        (0, 0) => "sub",
                        (0, 1) => "xor",
                        (0, 2) => "or",
                        (0, 3) => "and",
                        (1, 0) => "subw",
                        (1, 1) => "addw",
                        _ => return None,
                    };
                    i(m, vec![X(rs1_p), X(rs1_p), X(rd_p)])
                }
            }
        }
        (1, 5) => {
            let off = sext(
                u64::from(
                    cb(h, 12, 12) << 11
                        | cb(h, 11, 11) << 4
                        | cb(h, 10, 9) << 8
                        | cb(h, 8, 8) << 10
                        | cb(h, 7, 7) << 6
                        | cb(h, 6, 6) << 7
                        | cb(h, 5, 3) << 1
                        | cb(h, 2, 2) << 5,
                ),
                12,
            );
            i("jal", vec![X(ZERO), Target(addr.wrapping_add(off as u64))])
        }
        (1, 6 | 7) => {
            let off = sext(
                u64::from(cb(h, 12, 12) << 8 | cb(h, 11, 10) << 3 | cb(h, 6, 5) << 6 | cb(h, 4, 3) << 1 | cb(h, 2, 2) << 5),
                9,
            );
            let m = if f3 == 6 { "beq" } else { "bne" };
            i(m, vec![X(rs1_p), X(ZERO), Target(addr.wrapping_add(off as u64))])
        }
        (2, 0) => {
            let shamt = i64::from(cb(h, 12, 12) << 5 | cb(h, 6, 2));
            if rd == 0 || shamt == 0 {
                return i("c.slli", vec![X(rd), Imm(shamt)]);
            }
            i("slli", vec![X(rd), X(rd), Imm(shamt)])
        }
        (2, 1) => {
            let off = i64::from(cb(h, 12, 12) << 5 | cb(h, 6, 5) << 3 | cb(h, 4, 2) << 6);
            i("fld", vec![F(rd), Mem(SP, off)])
        }
        (2, 2) if rd != 0 => {
            let off = i64::from(cb(h, 12, 12) << 5 | cb(h, 6, 4) << 2 | cb(h, 3, 2) << 6);
            i("lw", vec![X(rd), Mem(SP, off)])
        }
        (2, 3) if rd != 0 => {
            let off = i64::from(cb(h, 12, 12) << 5 | cb(h, 6, 5) << 3 | cb(h, 4, 2) << 6);
            i("ld", vec![X(rd), Mem(SP, off)])
        }
        (2, 4) => match (cb(h, 12, 12), rd, rs2) {
            (0, 0, 0) => None,
            (0, r, 0) => i("jalr", vec![X(ZERO), Mem(r, 0)]),
            (0, 0, s) => i("c.mv", vec![X(ZERO), X(s)]),
            (0, d, s) => i("addi", vec![X(d), X(s), Imm(0)]),
            (1, 0, 0) => i("ebreak", vec![]),
            (1, r, 0) => i("jalr", vec![X(RA), Mem(r, 0)]),
            (_, 0, s) => i("c.add", vec![X(ZERO), X(s)]),
            (_, d, s) => i("add", vec![X(d), X(d), X(s)]),
        },
        (2, 5) => {
            let off = i64::from(cb(h, 12, 10) << 3 | cb(h, 9, 7) << 6);
            i("fsd", vec![F(rs2), Mem(SP, off)])
        }
        (2, 6) => {
            let off = i64::from(cb(h, 12, 9) << 2 | cb(h, 8, 7) << 6);
            i("sw", vec![X(rs2), Mem(SP, off)])
        }
        (2, 7) => {
            let off = i64::from(cb(h, 12, 10) << 3 | cb(h, 9, 7) << 6);
            i("sd", vec![X(rs2), Mem(SP, off)])
        }
        _ => None,
    }
}
