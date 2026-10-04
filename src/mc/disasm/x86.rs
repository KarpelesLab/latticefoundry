//! The x86-64 decoder, written from the Intel 64 and IA-32 Architectures
//! Software Developer's Manual (Volume 2: the instruction formats, the
//! opcode maps of Appendix A, and the instruction reference).
//!
//! Decoding is two-step: [`decode_inst`] turns bytes into a typed
//! [`X86Inst`] (prefixes, mnemonic, operands in Intel order: registers,
//! immediates, [`Mem`] operands, relative branch displacements), and
//! [`X86Inst::render`] prints it in AT&T or Intel syntax with the spellings
//! `llvm-objdump` uses (`movq %rsp, %rbp` / `mov rbp, rsp`, `movzbl`,
//! `cltq`, `$0x10`, `-0x8(%rbp)` / `qword ptr [rbp - 0x8]`).
//!
//! Covered: the general-purpose integer instruction set of 64-bit mode
//! (legacy, `REX`, segment, `LOCK` and `REP` prefixes; every ModRM/SIB/
//! displacement/RIP-relative form; 32-bit addressing via `67`), the system
//! instructions compilers emit (`syscall`, `cpuid`, `rdtsc`, fences,
//! `endbr64`, ...), and SSE through SSE4.1 on `xmm` registers (scalar and
//! packed floating point, conversions, packed integer arithmetic, shuffles,
//! shifts, inserts/extracts), the x87 floating-point instructions, and the
//! VEX-encoded AVX/AVX2/FMA3/BMI forms (the `vex` submodule). MMX, EVEX (AVX-512)
//! and far transfers decode as `.byte`.
//!
//! Extending it: a one-byte or `0F` integer opcode is one match arm in
//! `Decoder::one_byte` / `Decoder::two_byte`; an SSE opcode is one row of
//! the `SSE`, `SSE38` or `SSE3A` tables.

use super::{Inst, Options, Syntax};

mod vex;

// ===========================================================================
// The typed instruction
// ===========================================================================

/// A memory operand.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Mem {
    /// Segment override (`fs`, `gs`, ...).
    pub seg: Option<&'static str>,
    /// Base register (`rip` for RIP-relative).
    pub base: Option<&'static str>,
    /// Index register.
    pub index: Option<&'static str>,
    /// Index scale (1, 2, 4, 8).
    pub scale: u8,
    /// Displacement (sign-extended; an absolute 64-bit `moffs` as its bits).
    pub disp: i64,
    /// The Intel size keyword (`byte`, `dword`, `xmmword`, ...); empty for
    /// `lea` and other address-only operands.
    pub size: &'static str,
    /// The displacement is a full 64-bit absolute address (`movabs`).
    pub abs64: bool,
}

/// An operand.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Operand {
    /// A register.
    Reg(&'static str),
    /// An immediate: its value (sign-extended to the operand size), whether
    /// it prints signed, and its width in bits (for unsigned printing).
    Imm {
        /// The value.
        value: i64,
        /// Print as a signed number.
        signed: bool,
        /// Width in bits.
        bits: u32,
    },
    /// A memory operand.
    Mem(Mem),
    /// A branch displacement relative to the end of the instruction.
    Rel(i64),
}

/// A decoded x86-64 instruction.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct X86Inst {
    /// Length in bytes.
    pub len: usize,
    /// Prefixes printed before the mnemonic (`rep`, `repne`, `lock`,
    /// `addr32`), space-separated; usually empty.
    pub prefix: String,
    /// The AT&T mnemonic (with its size suffix).
    pub att: String,
    /// The Intel mnemonic.
    pub intel: String,
    /// The operands in Intel order (destination first).
    pub ops: Vec<Operand>,
    /// AT&T marks the (single) operand of an indirect branch with `*`.
    pub star: bool,
    /// AT&T keeps the Intel operand order (`enter`).
    pub keep_order: bool,
}

/// Decode one instruction of `bytes` (non-empty) at `addr`, rendered in
/// `opts.syntax`. Unknown encodings become `.byte`.
pub fn decode(bytes: &[u8], addr: u64, opts: &Options) -> Inst {
    match decode_inst(bytes) {
        Some(x) => x.render(addr, opts.syntax),
        None => Inst::data(bytes, 1, true),
    }
}

/// Decode one instruction into its typed form, or `None` for an invalid,
/// unsupported or truncated encoding.
pub fn decode_inst(bytes: &[u8]) -> Option<X86Inst> {
    let mut d = Decoder { b: &bytes[..bytes.len().min(15)], pos: 0, p: Prefixes::default(), two: false, moffs: false, rep_used: false, rep_separated: false, vex: None };
    d.run()
}

fn hex(v: i64, signed: bool, bits: u32) -> String {
    if signed {
        if v < 0 { format!("-{:#x}", v.unsigned_abs()) } else { format!("{v:#x}") }
    } else {
        let u = if bits >= 64 { v as u64 } else { (v as u64) & ((1u64 << bits) - 1) };
        format!("{u:#x}")
    }
}

impl Mem {
    fn att(&self) -> String {
        let mut s = String::new();
        if let Some(seg) = self.seg {
            s.push_str(&format!("%{seg}:"));
        }
        let addressing = self.base.is_some() || self.index.is_some();
        if self.disp != 0 || !addressing {
            s.push_str(&hex(self.disp, true, 64));
        }
        if addressing {
            s.push('(');
            if let Some(b) = self.base {
                s.push_str(&format!("%{b}"));
            }
            if let Some(i) = self.index {
                s.push_str(&format!(",%{i}"));
                if self.scale != 1 {
                    s.push_str(&format!(",{}", self.scale));
                }
            }
            s.push(')');
        }
        s
    }

    fn intel(&self) -> String {
        let mut s = String::new();
        if !self.size.is_empty() {
            s.push_str(self.size);
            s.push_str(" ptr ");
        }
        if let Some(seg) = self.seg {
            s.push_str(seg);
            s.push(':');
        }
        s.push('[');
        let mut parts = String::new();
        if let Some(b) = self.base {
            parts.push_str(b);
        }
        if let Some(i) = self.index {
            if !parts.is_empty() {
                parts.push_str(" + ");
            }
            if self.scale != 1 || self.base.is_none() {
                parts.push_str(&format!("{}*", self.scale));
            }
            parts.push_str(i);
        }
        if parts.is_empty() {
            parts.push_str(&hex(self.disp, true, 64));
        } else if self.disp > 0 {
            parts.push_str(&format!(" + {:#x}", self.disp));
        } else if self.disp < 0 {
            parts.push_str(&format!(" - {:#x}", self.disp.unsigned_abs()));
        }
        s.push_str(&parts);
        s.push(']');
        s
    }
}

impl X86Inst {
    /// Render at address `addr` in `syntax`.
    pub fn render(&self, addr: u64, syntax: Syntax) -> Inst {
        let att = syntax == Syntax::Att;
        let base = if att { &self.att } else { &self.intel };
        let mnemonic = if self.prefix.is_empty() { base.clone() } else { format!("{} {base}", self.prefix) };
        let mut inst = Inst::new(self.len, mnemonic);
        let mut order: Vec<&Operand> = self.ops.iter().collect();
        if att && !self.keep_order {
            order.reverse();
        }
        for op in order {
            match op {
                Operand::Rel(d) => {
                    let t = addr.wrapping_add(self.len as u64).wrapping_add(*d as u64);
                    inst = inst.target_op(t);
                }
                Operand::Reg(r) => {
                    let star = if att && self.star { "*" } else { "" };
                    inst = inst.op(if att { format!("{star}%{r}") } else { (*r).to_owned() });
                }
                Operand::Imm { value, signed, bits } => {
                    let h = hex(*value, *signed, *bits);
                    inst = inst.op(if att { format!("${h}") } else { h });
                }
                Operand::Mem(m) => {
                    let star = if att && self.star { "*" } else { "" };
                    inst = inst.op(if att { format!("{star}{}", m.att()) } else { m.intel() });
                }
            }
        }
        inst
    }
}

// ===========================================================================
// Registers
// ===========================================================================

const R64: [&str; 16] =
    ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
const R32: [&str; 16] = [
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
];
const R16: [&str; 16] =
    ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w"];
const R8REX: [&str; 16] = [
    "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
];
const R8LEGACY: [&str; 8] = ["al", "cl", "dl", "bl", "ah", "ch", "dh", "bh"];
const XMM: [&str; 16] = [
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13",
    "xmm14", "xmm15",
];
const SEGS: [&str; 8] = ["es", "cs", "ss", "ds", "fs", "gs", "", ""];
const CC: [&str; 16] = ["o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g"];

/// The AT&T size suffix of an operand size in bits.
fn suffix(bits: u32) -> &'static str {
    match bits {
        8 => "b",
        16 => "w",
        32 => "l",
        _ => "q",
    }
}

/// The Intel size keyword of an operand size in bits.
fn ptr(bits: u32) -> &'static str {
    match bits {
        8 => "byte",
        16 => "word",
        32 => "dword",
        64 => "qword",
        80 => "tbyte",
        128 => "xmmword",
        256 => "ymmword",
        _ => "",
    }
}

// ===========================================================================
// The decoder
// ===========================================================================

#[derive(Clone, Copy, Default)]
struct Prefixes {
    opsize: bool,
    addr32: bool,
    /// The last of `F2`/`F3`, or 0.
    rep: u8,
    lock: bool,
    seg: Option<&'static str>,
    rex: u8,
}

impl Prefixes {
    fn w(&self) -> bool {
        self.rex & 8 != 0
    }
    fn r(&self) -> u8 {
        (self.rex & 4) << 1
    }
    fn x(&self) -> u8 {
        (self.rex & 2) << 2
    }
    fn b(&self) -> u8 {
        (self.rex & 1) << 3
    }
    /// The size of a `v` (16/32/64) operand.
    fn v(&self) -> u32 {
        if self.w() {
            64
        } else if self.opsize {
            16
        } else {
            32
        }
    }
    /// The size of a default-64-bit operand (`push`, `pop`, near branches).
    fn d64(&self) -> u32 {
        if self.opsize && !self.w() { 16 } else { 64 }
    }
    /// The size of a `y` (32/64 by `REX.W`) operand.
    fn y(&self) -> u32 {
        if self.w() { 64 } else { 32 }
    }
}

/// A ModRM byte's register/memory operand before its size is known.
#[derive(Clone)]
enum Rm {
    Reg(u8),
    Mem(Mem),
}

#[derive(Clone)]
struct ModRm {
    md: u8,
    /// The `reg` field with `REX.R`.
    reg: u8,
    /// The raw `reg` field (an opcode extension).
    ext: u8,
    rm: Rm,
}

struct Decoder<'a> {
    b: &'a [u8],
    pos: usize,
    p: Prefixes,
    /// Decoding a `0F`-map opcode (F2/F3 are then never `rep`).
    two: bool,
    /// The `67` prefix is consumed by the instruction (a `moffs` operand,
    /// `jecxz`) and does not print as `addr32`.
    moffs: bool,
    /// The F2/F3 prefix is part of the opcode (`pause`).
    rep_used: bool,
    /// Another legacy prefix follows the last F2/F3.
    rep_separated: bool,
    /// The VEX prefix, for a VEX-encoded instruction.
    vex: Option<vex::Vex>,
}

/// What kind of register an operand names.
#[derive(Clone, Copy, PartialEq, Eq)]
enum K {
    Gpr,
    Xmm,
}

impl<'a> Decoder<'a> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.pos)?;
        self.pos += 1;
        Some(v)
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from(self.u8()?) | u16::from(self.u8()?) << 8)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from(self.u16()?) | u32::from(self.u16()?) << 16)
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from(self.u32()?) | u64::from(self.u32()?) << 32)
    }

    fn gpr(&self, n: u8, bits: u32) -> &'static str {
        let n = usize::from(n & 15);
        match bits {
            8 if self.p.rex != 0 => R8REX[n],
            8 => R8LEGACY[n & 7],
            16 => R16[n],
            32 => R32[n],
            _ => R64[n],
        }
    }

    fn reg(&self, n: u8, bits: u32, k: K) -> Operand {
        Operand::Reg(match k {
            K::Gpr => self.gpr(n, bits),
            K::Xmm => XMM[usize::from(n & 15)],
        })
    }

    fn modrm(&mut self) -> Option<ModRm> {
        let m = self.u8()?;
        let md = m >> 6;
        let ext = (m >> 3) & 7;
        let reg = ext | self.p.r();
        let rm = m & 7;
        if md == 3 {
            return Some(ModRm { md, reg, ext, rm: Rm::Reg(rm | self.p.b()) });
        }
        let names: &[&'static str; 16] = if self.p.addr32 { &R32 } else { &R64 };
        let mut mem = Mem { seg: self.p.seg, base: None, index: None, scale: 1, disp: 0, size: "", abs64: false };
        let mut nobase = false;
        if rm == 4 {
            let sib = self.u8()?;
            let scale = 1u8 << (sib >> 6);
            let index = ((sib >> 3) & 7) | self.p.x();
            let base = sib & 7;
            nobase = base == 5 && md == 0;
            if !nobase {
                mem.base = Some(names[usize::from(base | self.p.b())]);
            }
            if index != 4 {
                mem.index = Some(names[usize::from(index)]);
                mem.scale = scale;
            } else if !(scale == 1 && (base == 4 || nobase)) {
                // An index field of 100 without REX.X is "no index"; it is
                // shown as `riz` when the SIB byte was not needed for it.
                mem.index = Some(if self.p.addr32 { "eiz" } else { "riz" });
                mem.scale = scale;
            }
        } else if rm == 5 && md == 0 {
            mem.base = Some(if self.p.addr32 { "eip" } else { "rip" });
            mem.disp = i64::from(self.u32()? as i32);
            return Some(ModRm { md, reg, ext, rm: Rm::Mem(mem) });
        } else {
            mem.base = Some(names[usize::from(rm | self.p.b())]);
        }
        if md == 1 {
            mem.disp = i64::from(self.u8()? as i8);
        } else if md == 2 || nobase {
            mem.disp = i64::from(self.u32()? as i32);
        }
        Some(ModRm { md, reg, ext, rm: Rm::Mem(mem) })
    }

    /// The r/m operand as a `bits`-wide register of kind `k`, or memory of
    /// Intel size `mem_bits`.
    fn rm(&self, m: &ModRm, bits: u32, k: K, mem_bits: u32) -> Operand {
        match &m.rm {
            Rm::Reg(n) => self.reg(*n, bits, k),
            Rm::Mem(mem) => {
                let mut mem = mem.clone();
                mem.size = ptr(mem_bits);
                Operand::Mem(mem)
            }
        }
    }

    fn imm(&mut self, bits: u32, signed: bool) -> Option<Operand> {
        let value = match bits {
            8 => i64::from(self.u8()? as i8),
            16 => i64::from(self.u16()? as i16),
            _ => i64::from(self.u32()? as i32),
        };
        Some(Operand::Imm { value, signed, bits })
    }

    /// An `Iz` immediate for a `v`-sized operation: 16 or 32 bits, printed
    /// unsigned, or a sign-extended 32-bit immediate of a 64-bit operation.
    fn iz(&mut self, v: u32) -> Option<Operand> {
        match v {
            16 => self.imm(16, false),
            32 => self.imm(32, false),
            _ => Some(Operand::Imm { value: i64::from(self.u32()? as i32), signed: true, bits: 64 }),
        }
    }

    /// An imm8 sign-extended to a `v`-sized operation.
    fn ibs(&mut self, v: u32) -> Option<Operand> {
        Some(Operand::Imm { value: i64::from(self.u8()? as i8), signed: true, bits: v })
    }

    /// A `Jcc rel32` displacement, or `rel16` under a `66` prefix.
    fn rel_z(&mut self) -> Option<Operand> {
        Some(Operand::Rel(if self.p.opsize { i64::from(self.u16()? as i16) } else { i64::from(self.u32()? as i32) }))
    }

    /// An unsigned imm8.
    fn ib(&mut self) -> Option<Operand> {
        Some(Operand::Imm { value: i64::from(self.u8()?), signed: false, bits: 8 })
    }

    fn run(&mut self) -> Option<X86Inst> {
        // A leading LOCK prints as an instruction of its own (as llvm-objdump
        // does).
        if self.b.first() == Some(&0xf0) {
            return Some(self.make(1, "lock", "lock", Vec::new()));
        }
        loop {
            let c = self.peek()?;
            match c {
                0x66 => self.p.opsize = true,
                0x67 => self.p.addr32 = true,
                0xf2 | 0xf3 => {
                    self.p.rep = c;
                    self.rep_separated = false;
                }
                0xf0 => self.p.lock = true,
                0x26 | 0x2e | 0x36 | 0x3e | 0x64 | 0x65 => {
                    self.p.seg = Some(match c {
                        0x26 => "es",
                        0x2e => "cs",
                        0x36 => "ss",
                        0x3e => "ds",
                        0x64 => "fs",
                        _ => "gs",
                    });
                }
                0x40..=0x4f => {
                    self.pos += 1;
                    self.p.rex = c;
                    // A REX prefix counts only right before the opcode.
                    match self.peek()? {
                        0x66 | 0x67 | 0xf2 | 0xf3 | 0xf0 | 0x26 | 0x2e | 0x36 | 0x3e | 0x64 | 0x65 => {
                            self.p.rex = 0;
                            continue;
                        }
                        _ => break,
                    }
                }
                _ => break,
            }
            if !matches!(c, 0xf2 | 0xf3) && self.p.rep != 0 {
                self.rep_separated = true;
            }
            self.pos += 1;
        }
        let op = self.u8()?;
        let mut x = if op == 0x0f {
            self.two = true;
            self.two_byte()?
        } else if op == 0xc4 || op == 0xc5 {
            self.two = true;
            self.vex(op)?
        } else {
            self.one_byte(op)?
        };
        let mut pre: Vec<&str> = Vec::new();
        if self.p.lock {
            pre.push("lock");
        }
        // F2/F3 print as `repne`/`rep` on a one-byte opcode (unless they are
        // part of it, as in `pause`), and on a `0F` opcode only when another
        // legacy prefix separates them from it (otherwise they select the
        // opcode, as in `popcnt` or `cvtss2si`).
        if self.p.rep != 0 && !self.rep_used && (!self.two || self.rep_separated) {
            pre.push(if self.p.rep == 0xf3 { "rep" } else { "repne" });
        }
        // An address-size prefix that changes nothing visible prints on its
        // own (32-bit memory operands already show it).
        let has_mem = x.ops.iter().any(|o| matches!(o, Operand::Mem(m) if m.base.is_some() || m.index.is_some()));
        if self.p.addr32 && !has_mem && !self.moffs {
            pre.push("addr32");
        }
        x.prefix = pre.join(" ");
        Some(x)
    }

    fn make(&self, len: usize, att: &str, intel: &str, ops: Vec<Operand>) -> X86Inst {
        X86Inst {
            len,
            prefix: String::new(),
            att: att.to_owned(),
            intel: intel.to_owned(),
            ops,
            star: false,
            keep_order: false,
        }
    }

    /// An instruction ending at the current position, `name` taking the
    /// AT&T `suffix` (Intel: bare).
    fn done(&self, name: &str, suf: &str, ops: Vec<Operand>) -> Option<X86Inst> {
        Some(self.make(self.pos, &format!("{name}{suf}"), name, ops))
    }

    fn one_byte(&mut self, op: u8) -> Option<X86Inst> {
        const ALU: [&str; 8] = ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"];
        const SHIFT: [&str; 8] = ["rol", "ror", "rcl", "rcr", "shl", "shr", "shl", "sar"];
        let v = self.p.v();
        match op {
            // ALU r/m,reg / reg,r/m / acc,imm.
            0x00..=0x3f if op & 7 < 6 => {
                let name = ALU[usize::from(op >> 3)];
                match op & 7 {
                    0 | 2 => {
                        let m = self.modrm()?;
                        let (e, g) = (self.rm(&m, 8, K::Gpr, 8), self.reg(m.reg, 8, K::Gpr));
                        let ops = if op & 7 == 0 { vec![e, g] } else { vec![g, e] };
                        self.done(name, "b", ops)
                    }
                    1 | 3 => {
                        let m = self.modrm()?;
                        let (e, g) = (self.rm(&m, v, K::Gpr, v), self.reg(m.reg, v, K::Gpr));
                        let ops = if op & 7 == 1 { vec![e, g] } else { vec![g, e] };
                        self.done(name, suffix(v), ops)
                    }
                    4 => {
                        let i = self.imm(8, true)?;
                        self.done(name, "b", vec![Operand::Reg("al"), i])
                    }
                    _ => {
                        let i = self.iz(v)?;
                        self.done(name, suffix(v), vec![Operand::Reg(self.gpr(0, v)), i])
                    }
                }
            }
            0x50..=0x5f => {
                let n = (op & 7) | self.p.b();
                let s = self.p.d64();
                self.done(if op < 0x58 { "push" } else { "pop" }, suffix(s), vec![self.reg(n, s, K::Gpr)])
            }
            0x63 => {
                let m = self.modrm()?;
                let ops = vec![self.reg(m.reg, v, K::Gpr), self.rm(&m, 32, K::Gpr, 32)];
                let mut x = self.done("movsxd", "", ops)?;
                x.att = "movslq".to_owned();
                Some(x)
            }
            0x68 => {
                let s = self.p.d64();
                let i = if s == 16 { self.imm(16, false)? } else { self.iz(64)? };
                self.done("push", suffix(s), vec![i])
            }
            0x6a => {
                let s = self.p.d64();
                let i = self.ibs(s)?;
                self.done("push", suffix(s), vec![i])
            }
            0x69 | 0x6b => {
                let m = self.modrm()?;
                let (g, e) = (self.reg(m.reg, v, K::Gpr), self.rm(&m, v, K::Gpr, v));
                let i = if op == 0x69 { self.iz(v)? } else { self.ibs(v)? };
                self.done("imul", suffix(v), vec![g, e, i])
            }
            0x70..=0x7f => {
                let d = i64::from(self.u8()? as i8);
                self.done(&format!("j{}", CC[usize::from(op & 15)]), "", vec![Operand::Rel(d)])
            }
            0x80..=0x83 => {
                let m = self.modrm()?;
                let name = ALU[usize::from(m.ext)];
                if op == 0x80 || op == 0x82 {
                    if op == 0x82 {
                        return None; // invalid in 64-bit mode
                    }
                    let e = self.rm(&m, 8, K::Gpr, 8);
                    let i = self.imm(8, true)?;
                    self.done(name, "b", vec![e, i])
                } else {
                    let e = self.rm(&m, v, K::Gpr, v);
                    let i = if op == 0x81 { self.iz(v)? } else { self.ibs(v)? };
                    self.done(name, suffix(v), vec![e, i])
                }
            }
            0x84..=0x8b => {
                let m = self.modrm()?;
                let s = if op & 1 == 0 { 8 } else { v };
                let (e, g) = (self.rm(&m, s, K::Gpr, s), self.reg(m.reg, s, K::Gpr));
                let name = match op {
                    0x84 | 0x85 => "test",
                    0x86 | 0x87 => "xchg",
                    _ => "mov",
                };
                // A register-register xchg lists the reg field first.
                let swap = op >= 0x8a || (name == "xchg" && matches!(m.rm, Rm::Reg(_)));
                let ops = if swap { vec![g, e] } else { vec![e, g] };
                self.done(name, suffix(s), ops)
            }
            0x8c | 0x8e => {
                let m = self.modrm()?;
                let seg = Operand::Reg(SEGS.get(usize::from(m.ext)).filter(|s| !s.is_empty())?);
                let s = if matches!(m.rm, Rm::Reg(_)) { v } else { 16 };
                let e = self.rm(&m, s, K::Gpr, 16);
                let ops = if op == 0x8c { vec![e, seg] } else { vec![seg, e] };
                self.done("mov", suffix(s), ops)
            }
            0x8d => {
                let m = self.modrm()?;
                let Rm::Mem(mem) = m.rm.clone() else { return None };
                self.done("lea", suffix(v), vec![self.reg(m.reg, v, K::Gpr), Operand::Mem(mem)])
            }
            0x8f => {
                let m = self.modrm()?;
                if m.ext != 0 {
                    return None;
                }
                let s = self.p.d64();
                let e = self.rm(&m, s, K::Gpr, s);
                self.done("pop", suffix(s), vec![e])
            }
            0x90 if self.p.b() == 0 => {
                if self.p.rep == 0xf3 && self.p.rex == 0 {
                    self.rep_used = true;
                    return Some(self.make(self.pos, "pause", "pause", Vec::new()));
                }
                self.done("nop", "", Vec::new())
            }
            0x90..=0x97 => {
                let n = (op & 7) | self.p.b();
                self.done("xchg", suffix(v), vec![self.reg(0, v, K::Gpr), self.reg(n, v, K::Gpr)])
            }
            0x98 => {
                let (a, i) = match v {
                    16 => ("cbtw", "cbw"),
                    32 => ("cwtl", "cwde"),
                    _ => ("cltq", "cdqe"),
                };
                Some(self.make(self.pos, a, i, Vec::new()))
            }
            0x99 => {
                let (a, i) = match v {
                    16 => ("cwtd", "cwd"),
                    32 => ("cltd", "cdq"),
                    _ => ("cqto", "cqo"),
                };
                Some(self.make(self.pos, a, i, Vec::new()))
            }
            0x9b => self.done("wait", "", Vec::new()),
            0x9c | 0x9d => {
                let s = self.p.d64();
                let base = if op == 0x9c { "pushf" } else { "popf" };
                let att = format!("{base}{}", suffix(s));
                Some(self.make(self.pos, &att, if s == 16 { base } else { &att }, Vec::new()))
            }
            0x9e => self.done("sahf", "", Vec::new()),
            0x9f => self.done("lahf", "", Vec::new()),
            0xa0..=0xa3 => {
                let s = if op & 1 == 0 { 8 } else { v };
                let disp = if self.p.addr32 { i64::from(self.u32()?) } else { self.u64()? as i64 };
                let mem = Mem { seg: self.p.seg, base: None, index: None, scale: 1, disp, size: ptr(s), abs64: !self.p.addr32 };
                let acc = self.reg(0, s, K::Gpr);
                let ops = if op < 0xa2 { vec![acc, Operand::Mem(mem)] } else { vec![Operand::Mem(mem), acc] };
                self.moffs = true;
                self.done(if self.p.addr32 { "mov" } else { "movabs" }, suffix(s), ops)
            }
            0xa4..=0xa7 | 0xaa..=0xaf => self.string_op(op),
            0xa8 => {
                let i = self.imm(8, true)?;
                self.done("test", "b", vec![Operand::Reg("al"), i])
            }
            0xa9 => {
                let i = self.iz(v)?;
                self.done("test", suffix(v), vec![self.reg(0, v, K::Gpr), i])
            }
            0xb0..=0xb7 => {
                let r = self.reg((op & 7) | self.p.b(), 8, K::Gpr);
                let i = self.imm(8, true)?;
                self.done("mov", "b", vec![r, i])
            }
            0xb8..=0xbf => {
                let r = self.reg((op & 7) | self.p.b(), v, K::Gpr);
                if v == 64 {
                    let value = self.u64()? as i64;
                    let mut x = self.done("movabs", "q", vec![r, Operand::Imm { value, signed: true, bits: 64 }])?;
                    x.intel = "movabs".to_owned();
                    Some(x)
                } else {
                    let i = self.imm(v, false)?;
                    self.done("mov", suffix(v), vec![r, i])
                }
            }
            0xc0 | 0xc1 | 0xd0..=0xd3 => {
                let m = self.modrm()?;
                let s = if op & 1 == 0 { 8 } else { v };
                let e = self.rm(&m, s, K::Gpr, s);
                if m.ext == 6 {
                    return None; // the undocumented SAL alias
                }
                let name = SHIFT[usize::from(m.ext)];
                let ops = match op {
                    0xc0 | 0xc1 => vec![e, self.ib()?],
                    0xd0 | 0xd1 => vec![e],
                    _ => vec![e, Operand::Reg("cl")],
                };
                self.done(name, suffix(s), ops)
            }
            0xc2 | 0xc3 => {
                let wide = self.p.d64() == 64;
                let ops = if op == 0xc2 { vec![self.imm(16, wide)?] } else { Vec::new() };
                let mut x = self.done("ret", suffix(self.p.d64()), ops)?;
                x.intel = "ret".to_owned();
                Some(x)
            }
            0xc6 | 0xc7 => {
                let m = self.modrm()?;
                if m.ext != 0 {
                    return None;
                }
                let s = if op == 0xc6 { 8 } else { v };
                let e = self.rm(&m, s, K::Gpr, s);
                let i = if s == 8 { self.imm(8, true)? } else { self.iz(s)? };
                self.done("mov", suffix(s), vec![e, i])
            }
            0xc8 => {
                let a = self.imm(16, true)?;
                let b = self.imm(8, true)?;
                let mut x = self.done("enter", "", vec![a, b])?;
                x.keep_order = true;
                Some(x)
            }
            0xc9 => self.done("leave", "", Vec::new()),
            0xcc => self.done("int3", "", Vec::new()),
            0xcd => {
                let i = self.ib()?;
                self.done("int", "", vec![i])
            }
            0xcf => {
                let (a, i) = match v {
                    16 => ("iretw", "iret"),
                    32 => ("iretl", "iretd"),
                    _ => ("iretq", "iretq"),
                };
                Some(self.make(self.pos, a, i, Vec::new()))
            }
            0xe0..=0xe3 => {
                let d = i64::from(self.u8()? as i8);
                let name = match op {
                    0xe0 => "loopne",
                    0xe1 => "loope",
                    0xe2 => "loop",
                    // (llvm keeps `jrcxz` + `addr32` under REX.W or 66.)
                    _ if self.p.addr32 && !self.p.w() && !self.p.opsize => {
                        self.moffs = true; // the 67 is consumed
                        "jecxz"
                    }
                    _ => "jrcxz",
                };
                self.done(name, "", vec![Operand::Rel(d)])
            }
            0xe4 | 0xe5 | 0xec | 0xed => {
                let s = if op & 1 == 0 { 8 } else if self.p.opsize && !self.p.w() { 16 } else { 32 };
                let port = if op < 0xe8 { self.ib()? } else { Operand::Reg("dx") };
                self.done("in", suffix(s), vec![self.reg(0, s, K::Gpr), port])
            }
            0xe6 | 0xe7 | 0xee | 0xef => {
                let s = if op & 1 == 0 { 8 } else if self.p.opsize && !self.p.w() { 16 } else { 32 };
                let port = if op < 0xe8 { self.ib()? } else { Operand::Reg("dx") };
                self.done("out", suffix(s), vec![port, self.reg(0, s, K::Gpr)])
            }
            0xe8 => {
                let d = Operand::Rel(i64::from(self.u32()? as i32));
                let mut x = self.done("call", "q", vec![d])?;
                x.intel = "call".to_owned();
                Some(x)
            }
            0xe9 => {
                let d = if self.p.opsize && !self.p.w() {
                    Operand::Rel(i64::from(self.u16()? as i16))
                } else {
                    Operand::Rel(i64::from(self.u32()? as i32))
                };
                self.done("jmp", "", vec![d])
            }
            0xeb => {
                let d = i64::from(self.u8()? as i8);
                self.done("jmp", "", vec![Operand::Rel(d)])
            }
            0xd8..=0xdf => self.x87(op),
            0xf4 => self.done("hlt", "", Vec::new()),
            0xf5 => self.done("cmc", "", Vec::new()),
            0xf6 | 0xf7 => {
                let m = self.modrm()?;
                let s = if op == 0xf6 { 8 } else { v };
                let e = self.rm(&m, s, K::Gpr, s);
                const G3: [&str; 8] = ["test", "", "not", "neg", "mul", "imul", "div", "idiv"];
                if m.ext == 1 {
                    return None; // the undocumented TEST alias
                }
                let ops = if m.ext == 0 {
                    let i = if s == 8 { self.imm(8, true)? } else { self.iz(s)? };
                    vec![e, i]
                } else {
                    vec![e]
                };
                self.done(G3[usize::from(m.ext)], suffix(s), ops)
            }
            0xf8 => self.done("clc", "", Vec::new()),
            0xf9 => self.done("stc", "", Vec::new()),
            0xfa => self.done("cli", "", Vec::new()),
            0xfb => self.done("sti", "", Vec::new()),
            0xfc => self.done("cld", "", Vec::new()),
            0xfd => self.done("std", "", Vec::new()),
            0xfe => {
                let m = self.modrm()?;
                let name = match m.ext {
                    0 => "inc",
                    1 => "dec",
                    _ => return None,
                };
                let e = self.rm(&m, 8, K::Gpr, 8);
                self.done(name, "b", vec![e])
            }
            0xff => {
                let m = self.modrm()?;
                match m.ext {
                    0 | 1 => {
                        let e = self.rm(&m, v, K::Gpr, v);
                        self.done(if m.ext == 0 { "inc" } else { "dec" }, suffix(v), vec![e])
                    }
                    2 | 4 => {
                        let s = 64;
                        let e = self.rm(&m, s, K::Gpr, s);
                        let name = if m.ext == 2 { "call" } else { "jmp" };
                        let mut x = self.done(name, suffix(s), vec![e])?;
                        x.star = true;
                        Some(x)
                    }
                    6 => {
                        let s = self.p.d64();
                        let e = self.rm(&m, s, K::Gpr, s);
                        self.done("push", suffix(s), vec![e])
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The x87 escape opcodes `D8`-`DF` (SDM Vol. 2, Appendix A.5).
    fn x87(&mut self, op: u8) -> Option<X86Inst> {
        let m = self.modrm()?;
        let row = usize::from(op - 0xd8);
        if let Rm::Mem(_) = m.rm {
            // (AT&T, Intel, memory size in bits; 0 = no size keyword).
            let (att, intel, bits) = X87_MEM[row][usize::from(m.ext)];
            if att.is_empty() {
                return None;
            }
            let e = self.rm(&m, 0, K::Gpr, bits);
            return Some(self.make(self.pos, att, intel, vec![e]));
        }
        let Rm::Reg(r) = m.rm else { return None };
        let i = r & 7;
        let st = Operand::Reg("st");
        let sti = Operand::Reg(STI[usize::from(i)]);
        let ext = m.ext;
        let x = |att: &str, intel: &str, ops: Vec<Operand>| Some(self.make(self.pos, att, intel, ops));
        const ARITH: [&str; 8] = ["fadd", "fmul", "fcom", "fcomp", "fsub", "fsubr", "fdiv", "fdivr"];
        match (op, ext) {
            (0xd8, 2 | 3) => x(ARITH[usize::from(ext)], ARITH[usize::from(ext)], vec![sti]),
            (0xd8, _) => x(ARITH[usize::from(ext)], ARITH[usize::from(ext)], vec![st, sti]),
            (0xd9, 0) => x("fld", "fld", vec![sti]),
            (0xd9, 1) => x("fxch", "fxch", vec![sti]),
            (0xd9, _) => {
                let name = match 0xc0 | (ext << 3) | i {
                    0xd0 => "fnop",
                    0xe0 => "fchs",
                    0xe1 => "fabs",
                    0xe4 => "ftst",
                    0xe5 => "fxam",
                    0xe8 => "fld1",
                    0xe9 => "fldl2t",
                    0xea => "fldl2e",
                    0xeb => "fldpi",
                    0xec => "fldlg2",
                    0xed => "fldln2",
                    0xee => "fldz",
                    0xf0 => "f2xm1",
                    0xf1 => "fyl2x",
                    0xf2 => "fptan",
                    0xf3 => "fpatan",
                    0xf4 => "fxtract",
                    0xf5 => "fprem1",
                    0xf6 => "fdecstp",
                    0xf7 => "fincstp",
                    0xf8 => "fprem",
                    0xf9 => "fyl2xp1",
                    0xfa => "fsqrt",
                    0xfb => "fsincos",
                    0xfc => "frndint",
                    0xfd => "fscale",
                    0xfe => "fsin",
                    0xff => "fcos",
                    _ => return None,
                };
                x(name, name, Vec::new())
            }
            (0xda, 0..=3) => {
                let name = ["fcmovb", "fcmove", "fcmovbe", "fcmovu"][usize::from(ext)];
                x(name, name, vec![st, sti])
            }
            (0xda, 5) if i == 1 => x("fucompp", "fucompp", Vec::new()),
            (0xdb, 0..=3) => {
                let name = ["fcmovnb", "fcmovne", "fcmovnbe", "fcmovnu"][usize::from(ext)];
                x(name, name, vec![st, sti])
            }
            (0xdb, 4) if i == 2 => x("fnclex", "fnclex", Vec::new()),
            (0xdb, 4) if i == 3 => x("fninit", "fninit", Vec::new()),
            (0xdb, 5) => x("fucomi", "fucomi", vec![st, sti]),
            (0xdb, 6) => x("fcomi", "fcomi", vec![st, sti]),
            // DC and DE: st(i) is the destination; AT&T spells the
            // reversed subtract/divide forms the other way round.
            (0xdc | 0xde, 0 | 1 | 4..=7) => {
                const INTEL: [&str; 8] = ["fadd", "fmul", "", "", "fsubr", "fsub", "fdivr", "fdiv"];
                const ATT: [&str; 8] = ["fadd", "fmul", "", "", "fsub", "fsubr", "fdiv", "fdivr"];
                let p = if op == 0xde { "p" } else { "" };
                x(&format!("{}{p}", ATT[usize::from(ext)]), &format!("{}{p}", INTEL[usize::from(ext)]), vec![sti, st])
            }
            (0xde, 3) if i == 1 => x("fcompp", "fcompp", Vec::new()),
            (0xdd, 0) => x("ffree", "ffree", vec![sti]),
            (0xdd, 2) => x("fst", "fst", vec![sti]),
            (0xdd, 3) => x("fstp", "fstp", vec![sti]),
            (0xdd, 4) => x("fucom", "fucom", vec![sti]),
            (0xdd, 5) => x("fucomp", "fucomp", vec![sti]),
            (0xdf, 0) => x("ffreep", "ffreep", vec![sti]),
            (0xdf, 4) if i == 0 => x("fnstsw", "fnstsw", vec![Operand::Reg("ax")]),
            (0xdf, 5) => x("fucompi", "fucompi", vec![st, sti]),
            (0xdf, 6) => x("fcompi", "fcompi", vec![st, sti]),
            _ => None,
        }
    }

    /// `movs`, `cmps`, `stos`, `lods`, `scas`.
    fn string_op(&mut self, op: u8) -> Option<X86Inst> {
        let s = if op & 1 == 0 { 8 } else { self.p.v() };
        let regs: &[&'static str; 16] = if self.p.addr32 { &R32 } else { &R64 };
        let mem = |seg: Option<&'static str>, r: usize| {
            Operand::Mem(Mem { seg, base: Some(regs[r]), index: None, scale: 1, disp: 0, size: ptr(s), abs64: false })
        };
        let src = mem(self.p.seg, 6);
        let dst = mem(Some("es"), 7);
        let acc = self.reg(0, s, K::Gpr);
        let (name, ops) = match op & !1 {
            0xa4 => ("movs", vec![dst, src]),
            0xa6 => ("cmps", vec![src, dst]),
            0xaa => ("stos", vec![dst, acc]),
            0xac => ("lods", vec![acc, src]),
            _ => ("scas", vec![acc, dst]),
        };
        let isuf = match s {
            8 => "b",
            16 => "w",
            32 => "d",
            _ => "q",
        };
        let mut x = self.done(&format!("{name}{isuf}"), "", ops)?;
        x.att = format!("{name}{}", suffix(s));
        Some(x)
    }

    fn two_byte(&mut self) -> Option<X86Inst> {
        let op = self.u8()?;
        let v = self.p.v();
        match op {
            0x05 => self.done("syscall", "", Vec::new()),
            0x0b => self.done("ud2", "", Vec::new()),
            0x01 => {
                let m = self.u8()?;
                let name = match m {
                    0xd0 => "xgetbv",
                    0xd1 => "xsetbv",
                    0xd6 => "xtest",
                    0xee => "rdpkru",
                    0xef => "wrpkru",
                    0xf9 => "rdtscp",
                    0xf8 => "swapgs",
                    _ => return None,
                };
                self.done(name, "", Vec::new())
            }
            0x18 => {
                let m = self.modrm()?;
                let Rm::Mem(_) = m.rm else { return None };
                let name = match m.ext {
                    0 => "prefetchnta",
                    1 => "prefetcht0",
                    2 => "prefetcht1",
                    3 => "prefetcht2",
                    _ => return None,
                };
                let e = self.rm(&m, 8, K::Gpr, 8);
                self.done(name, "", vec![e])
            }
            0x1e if self.p.rep == 0xf3 && matches!(self.peek(), Some(0xfa | 0xfb)) => {
                let name = if self.u8()? == 0xfa { "endbr64" } else { "endbr32" };
                Some(self.make(self.pos, name, name, Vec::new()))
            }
            0x1f => {
                let m = self.modrm()?;
                if m.ext != 0 {
                    return None;
                }
                let e = self.rm(&m, v, K::Gpr, v);
                self.done("nop", suffix(v), vec![e])
            }
            0x20 | 0x22 => {
                let m = self.modrm()?;
                let Rm::Reg(r) = m.rm else { return None };
                const CR: [&str; 16] =
                    ["cr0", "cr1", "cr2", "cr3", "cr4", "cr5", "cr6", "cr7", "cr8", "", "", "", "", "", "", ""];
                let cr = Operand::Reg(CR.get(usize::from(m.reg)).filter(|s| !s.is_empty())?);
                let g = Operand::Reg(R64[usize::from(r)]);
                let mut x = self.done("mov", "q", if op == 0x20 { vec![g, cr] } else { vec![cr, g] })?;
                x.intel = "mov".to_owned();
                Some(x)
            }
            0x31 => self.done("rdtsc", "", Vec::new()),
            0x40..=0x4f => {
                let m = self.modrm()?;
                let ops = vec![self.reg(m.reg, v, K::Gpr), self.rm(&m, v, K::Gpr, v)];
                self.done(&format!("cmov{}", CC[usize::from(op & 15)]), suffix(v), ops)
            }
            0x80..=0x8f => {
                let d = self.rel_z()?;
                self.done(&format!("j{}", CC[usize::from(op & 15)]), "", vec![d])
            }
            0x90..=0x9f => {
                let m = self.modrm()?;
                let e = self.rm(&m, 8, K::Gpr, 8);
                self.done(&format!("set{}", CC[usize::from(op & 15)]), "", vec![e])
            }
            0xa2 => self.done("cpuid", "", Vec::new()),
            0xa3 | 0xab | 0xb3 | 0xbb => {
                let m = self.modrm()?;
                let name = ["bt", "bts", "btr", "btc"][usize::from((op >> 3) & 3)];
                let ops = vec![self.rm(&m, v, K::Gpr, v), self.reg(m.reg, v, K::Gpr)];
                self.done(name, suffix(v), ops)
            }
            0xba => {
                let m = self.modrm()?;
                let name = match m.ext {
                    4 => "bt",
                    5 => "bts",
                    6 => "btr",
                    7 => "btc",
                    _ => return None,
                };
                let e = self.rm(&m, v, K::Gpr, v);
                let i = self.ib()?;
                self.done(name, suffix(v), vec![e, i])
            }
            0xa4 | 0xa5 | 0xac | 0xad => {
                let m = self.modrm()?;
                let name = if op < 0xa8 { "shld" } else { "shrd" };
                let (e, g) = (self.rm(&m, v, K::Gpr, v), self.reg(m.reg, v, K::Gpr));
                let c = if op & 1 == 0 { self.ib()? } else { Operand::Reg("cl") };
                self.done(name, suffix(v), vec![e, g, c])
            }
            0xae => {
                let m = self.modrm()?;
                if m.md == 3 {
                    if self.p.opsize || self.p.rep != 0 {
                        return None; // tpause, umwait, ...
                    }
                    let name = match m.ext {
                        5 => "lfence",
                        6 => "mfence",
                        7 => "sfence",
                        _ => return None,
                    };
                    return self.done(name, "", Vec::new());
                }
                let (name, bits) = match m.ext {
                    2 => ("ldmxcsr", 32),
                    3 => ("stmxcsr", 32),
                    7 if self.p.opsize => ("clflushopt", 8),
                    7 => ("clflush", 8),
                    _ => return None,
                };
                let e = self.rm(&m, bits, K::Gpr, bits);
                self.done(name, "", vec![e])
            }
            0xaf => {
                let m = self.modrm()?;
                let ops = vec![self.reg(m.reg, v, K::Gpr), self.rm(&m, v, K::Gpr, v)];
                self.done("imul", suffix(v), ops)
            }
            0xb0 | 0xb1 | 0xc0 | 0xc1 => {
                let m = self.modrm()?;
                let s = if op & 1 == 0 { 8 } else { v };
                let ops = vec![self.rm(&m, s, K::Gpr, s), self.reg(m.reg, s, K::Gpr)];
                self.done(if op < 0xc0 { "cmpxchg" } else { "xadd" }, suffix(s), ops)
            }
            0xb6 | 0xb7 | 0xbe | 0xbf => {
                let m = self.modrm()?;
                let src = if op & 1 == 0 { 8 } else { 16 };
                let ops = vec![self.reg(m.reg, v, K::Gpr), self.rm(&m, src, K::Gpr, src)];
                let z = if op < 0xb8 { "z" } else { "s" };
                let mut x = self.done(&format!("mov{z}x"), "", ops)?;
                x.att = format!("mov{z}{}{}", suffix(src), suffix(v));
                Some(x)
            }
            0xb8 | 0xbc | 0xbd if self.p.rep == 0xf3 => {
                let m = self.modrm()?;
                let name = match op {
                    0xb8 => "popcnt",
                    0xbc => "tzcnt",
                    _ => "lzcnt",
                };
                let ops = vec![self.reg(m.reg, v, K::Gpr), self.rm(&m, v, K::Gpr, v)];
                self.p.rep = 0;
                self.done(name, suffix(v), ops)
            }
            0xbc | 0xbd => {
                let m = self.modrm()?;
                let ops = vec![self.reg(m.reg, v, K::Gpr), self.rm(&m, v, K::Gpr, v)];
                self.done(if op == 0xbc { "bsf" } else { "bsr" }, suffix(v), ops)
            }
            0xc7 => {
                let m = self.modrm()?;
                match (m.ext, &m.rm) {
                    (1, Rm::Mem(_)) => {
                        let (name, bits) = if self.p.w() { ("cmpxchg16b", 128) } else { ("cmpxchg8b", 64) };
                        let e = self.rm(&m, bits, K::Gpr, bits);
                        self.done(name, "", vec![e])
                    }
                    (6 | 7, Rm::Reg(_)) if self.p.rep == 0 => {
                        let e = self.rm(&m, v, K::Gpr, v);
                        self.done(if m.ext == 6 { "rdrand" } else { "rdseed" }, suffix(v), vec![e])
                    }
                    _ => None,
                }
            }
            0xc8..=0xcf => {
                let r = self.reg((op & 7) | self.p.b(), v, K::Gpr);
                self.done("bswap", suffix(v), vec![r])
            }
            0x38 => {
                let op3 = self.u8()?;
                match (op3, self.p.rep) {
                    (0xf0 | 0xf1, 0) => {
                        // movbe: a byte-swapping load or store.
                        let m = self.modrm()?;
                        let Rm::Mem(_) = m.rm else { return None };
                        let (g, e) = (self.reg(m.reg, v, K::Gpr), self.rm(&m, v, K::Gpr, v));
                        self.done("movbe", suffix(v), if op3 == 0xf0 { vec![g, e] } else { vec![e, g] })
                    }
                    (0xf0 | 0xf1, 0xf2) => {
                        let m = self.modrm()?;
                        let src = if op3 == 0xf0 { 8 } else { v };
                        let ops = vec![self.reg(m.reg, self.p.y(), K::Gpr), self.rm(&m, src, K::Gpr, src)];
                        self.done("crc32", suffix(src), ops)
                    }
                    _ => self.sse_table(SSE38, op3),
                }
            }
            0x3a => {
                let op3 = self.u8()?;
                self.sse_table(SSE3A, op3)
            }
            _ => self.sse(op),
        }
    }

    /// The mandatory prefix of an SSE instruction (`F3`/`F2` over `66`).
    fn mandatory(&self) -> u8 {
        if self.p.rep != 0 {
            self.p.rep
        } else if self.p.opsize {
            0x66
        } else {
            0
        }
    }

    fn sse(&mut self, op: u8) -> Option<X86Inst> {
        let pfx = self.mandatory();
        match (op, pfx) {
            // Forms whose shape depends on ModRM.mod.
            (0x12 | 0x16, 0) => {
                let m = self.modrm()?;
                let low = op == 0x12;
                let name = match (m.md == 3, low) {
                    (true, true) => "movhlps",
                    (true, false) => "movlhps",
                    (false, true) => "movlps",
                    (false, false) => "movhps",
                };
                let ops = vec![self.reg(m.reg, 128, K::Xmm), self.rm(&m, 128, K::Xmm, 64)];
                self.sse_done(name, ops)
            }
            // cvtsi2ss/sd: AT&T suffixes the memory form.
            (0x2a, 0xf2 | 0xf3) => {
                let m = self.modrm()?;
                let y = self.p.y();
                let ops = vec![self.reg(m.reg, 128, K::Xmm), self.rm(&m, y, K::Gpr, y)];
                let name = if pfx == 0xf2 { "cvtsi2sd" } else { "cvtsi2ss" };
                let mut x = self.sse_done(name, ops)?;
                if matches!(m.rm, Rm::Mem(_)) {
                    x.att = format!("{name}{}", suffix(y));
                }
                Some(x)
            }
            (0x6e, 0x66) => {
                let m = self.modrm()?;
                let y = self.p.y();
                let ops = vec![self.reg(m.reg, 128, K::Xmm), self.rm(&m, y, K::Gpr, y)];
                self.sse_done(if y == 64 { "movq" } else { "movd" }, ops)
            }
            (0x7e, 0x66) => {
                let m = self.modrm()?;
                let y = self.p.y();
                let ops = vec![self.rm(&m, y, K::Gpr, y), self.reg(m.reg, 128, K::Xmm)];
                self.sse_done(if y == 64 { "movq" } else { "movd" }, ops)
            }
            (0x71..=0x73, 0x66) => {
                let m = self.modrm()?;
                let Rm::Reg(r) = m.rm else { return None };
                let name = match (op, m.ext) {
                    (0x71, 2) => "psrlw",
                    (0x71, 4) => "psraw",
                    (0x71, 6) => "psllw",
                    (0x72, 2) => "psrld",
                    (0x72, 4) => "psrad",
                    (0x72, 6) => "pslld",
                    (0x73, 2) => "psrlq",
                    (0x73, 3) => "psrldq",
                    (0x73, 6) => "psllq",
                    (0x73, 7) => "pslldq",
                    _ => return None,
                };
                let i = self.ib()?;
                self.sse_done(name, vec![self.reg(r, 128, K::Xmm), i])
            }
            (0xc2, _) => {
                let (sfx, bits) = match pfx {
                    0 => ("ps", 128),
                    0x66 => ("pd", 128),
                    0xf3 => ("ss", 32),
                    _ => ("sd", 64),
                };
                let m = self.modrm()?;
                let ops = vec![self.reg(m.reg, 128, K::Xmm), self.rm(&m, 128, K::Xmm, bits)];
                let imm = self.u8()?;
                const PRED: [&str; 8] = ["eq", "lt", "le", "unord", "neq", "nlt", "nle", "ord"];
                match PRED.get(usize::from(imm)) {
                    Some(p) => self.sse_done(&format!("cmp{p}{sfx}"), ops),
                    None => {
                        let mut ops = ops;
                        ops.push(Operand::Imm { value: i64::from(imm), signed: false, bits: 8 });
                        self.sse_done(&format!("cmp{sfx}"), ops)
                    }
                }
            }
            (0xc4, 0x66) => {
                let m = self.modrm()?;
                let ops = vec![self.reg(m.reg, 128, K::Xmm), self.rm(&m, 32, K::Gpr, 16), self.ib()?];
                self.sse_done("pinsrw", ops)
            }
            (0xc5, 0x66) => {
                let m = self.modrm()?;
                let Rm::Reg(r) = m.rm else { return None };
                let ops = vec![self.reg(m.reg, 32, K::Gpr), self.reg(r, 128, K::Xmm), self.ib()?];
                self.sse_done("pextrw", ops)
            }
            _ => self.sse_table(SSE, op),
        }
    }

    fn sse_done(&mut self, name: &str, ops: Vec<Operand>) -> Option<X86Inst> {
        Some(self.make(self.pos, name, name, ops))
    }

    /// Look `op` up in an SSE table under the current mandatory prefix.
    fn sse_table(&mut self, table: &[Sse], op: u8) -> Option<X86Inst> {
        let pfx = self.mandatory();
        let row = table.iter().find(|r| r.op == op && r.pfx == pfx)?;
        let m = self.modrm()?;
        let xmm = |d: &Self, n: u8| d.reg(n, 128, K::Xmm);
        let y = self.p.y();
        let mut name = row.name.to_owned();
        let ops = match row.form {
            F::VW => vec![xmm(self, m.reg), self.rm(&m, 128, K::Xmm, row.mem)],
            F::WV => vec![self.rm(&m, 128, K::Xmm, row.mem), xmm(self, m.reg)],
            F::VWI => vec![xmm(self, m.reg), self.rm(&m, 128, K::Xmm, row.mem), self.ib()?],
            F::VM => {
                let Rm::Mem(_) = m.rm else { return None };
                vec![xmm(self, m.reg), self.rm(&m, 128, K::Xmm, row.mem)]
            }
            F::MV => {
                let Rm::Mem(_) = m.rm else { return None };
                vec![self.rm(&m, 128, K::Xmm, row.mem), xmm(self, m.reg)]
            }
            F::GyW => vec![self.reg(m.reg, y, K::Gpr), self.rm(&m, 128, K::Xmm, row.mem)],
            F::GdU => {
                let Rm::Reg(r) = m.rm else { return None };
                vec![self.reg(m.reg, 32, K::Gpr), xmm(self, r)]
            }
            F::EdVI => {
                // pextrb/w/d/q, extractps: a register destination is 32 bits
                // (64 for pextrq).
                let regbits = if row.name == "pextrd" && self.p.w() {
                    name = "pextrq".to_owned();
                    64
                } else {
                    32
                };
                let mem = if name == "pextrq" { 64 } else { row.mem };
                vec![self.rm(&m, regbits, K::Gpr, mem), xmm(self, m.reg), self.ib()?]
            }
            F::VEdI => {
                let regbits = if row.name == "pinsrd" && self.p.w() {
                    name = "pinsrq".to_owned();
                    64
                } else {
                    32
                };
                let mem = if name == "pinsrq" { 64 } else { row.mem };
                vec![xmm(self, m.reg), self.rm(&m, regbits, K::Gpr, mem), self.ib()?]
            }
            F::VW0 => vec![xmm(self, m.reg), self.rm(&m, 128, K::Xmm, row.mem), Operand::Reg("xmm0")],
        };
        Some(self.make(self.pos, &name, &name, ops))
    }
}

// ===========================================================================
// x87 tables
// ===========================================================================

const STI: [&str; 8] = ["st(0)", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)"];

/// The x87 memory forms: per escape byte `D8`-`DF` and ModRM.reg, the AT&T
/// and Intel mnemonics and the memory size (empty: reserved).
static X87_MEM: [[(&str, &str, u32); 8]; 8] = [
    [("fadds", "fadd", 32), ("fmuls", "fmul", 32), ("fcoms", "fcom", 32), ("fcomps", "fcomp", 32),
     ("fsubs", "fsub", 32), ("fsubrs", "fsubr", 32), ("fdivs", "fdiv", 32), ("fdivrs", "fdivr", 32)],
    [("flds", "fld", 32), ("", "", 0), ("fsts", "fst", 32), ("fstps", "fstp", 32),
     ("fldenv", "fldenv", 0), ("fldcw", "fldcw", 16), ("fnstenv", "fnstenv", 0), ("fnstcw", "fnstcw", 16)],
    [("fiaddl", "fiadd", 32), ("fimull", "fimul", 32), ("ficoml", "ficom", 32), ("ficompl", "ficomp", 32),
     ("fisubl", "fisub", 32), ("fisubrl", "fisubr", 32), ("fidivl", "fidiv", 32), ("fidivrl", "fidivr", 32)],
    [("fildl", "fild", 32), ("fisttpl", "fisttp", 32), ("fistl", "fist", 32), ("fistpl", "fistp", 32),
     ("", "", 0), ("fldt", "fld", 80), ("", "", 0), ("fstpt", "fstp", 80)],
    [("faddl", "fadd", 64), ("fmull", "fmul", 64), ("fcoml", "fcom", 64), ("fcompl", "fcomp", 64),
     ("fsubl", "fsub", 64), ("fsubrl", "fsubr", 64), ("fdivl", "fdiv", 64), ("fdivrl", "fdivr", 64)],
    [("fldl", "fld", 64), ("fisttpll", "fisttp", 64), ("fstl", "fst", 64), ("fstpl", "fstp", 64),
     ("frstor", "frstor", 0), ("", "", 0), ("fnsave", "fnsave", 0), ("fnstsw", "fnstsw", 16)],
    [("fiadds", "fiadd", 16), ("fimuls", "fimul", 16), ("ficoms", "ficom", 16), ("ficomps", "ficomp", 16),
     ("fisubs", "fisub", 16), ("fisubrs", "fisubr", 16), ("fidivs", "fidiv", 16), ("fidivrs", "fidivr", 16)],
    [("filds", "fild", 16), ("fisttps", "fisttp", 16), ("fists", "fist", 16), ("fistps", "fistp", 16),
     ("fbld", "fbld", 80), ("fildll", "fild", 64), ("fbstp", "fbstp", 80), ("fistpll", "fistp", 64)],
];

// ===========================================================================
// SSE tables
// ===========================================================================

/// The operand shape of an SSE row, in the SDM's operand-code letters.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
enum F {
    /// `xmm, xmm/m`.
    VW,
    /// `xmm/m, xmm` (stores).
    WV,
    /// `xmm, xmm/m, imm8`.
    VWI,
    /// `xmm, m`.
    VM,
    /// `m, xmm`.
    MV,
    /// `r32/64 (REX.W), xmm/m`.
    GyW,
    /// `r32, xmm` (register only).
    GdU,
    /// `r/m, xmm, imm8` (extracts).
    EdVI,
    /// `xmm, r/m, imm8` (inserts).
    VEdI,
    /// `xmm, xmm/m, <xmm0>` (variable blends).
    VW0,
}

/// One SSE opcode under one mandatory prefix.
struct Sse {
    op: u8,
    pfx: u8,
    name: &'static str,
    form: F,
    /// The Intel memory size in bits.
    mem: u32,
}

const fn s(op: u8, pfx: u8, name: &'static str, form: F, mem: u32) -> Sse {
    Sse { op, pfx, name, form, mem }
}

const X: u32 = 128;

/// The `0F xx` SSE opcodes.
static SSE: &[Sse] = &[
    s(0x10, 0, "movups", F::VW, X),
    s(0x10, 0x66, "movupd", F::VW, X),
    s(0x10, 0xf3, "movss", F::VW, 32),
    s(0x10, 0xf2, "movsd", F::VW, 64),
    s(0x11, 0, "movups", F::WV, X),
    s(0x11, 0x66, "movupd", F::WV, X),
    s(0x11, 0xf3, "movss", F::WV, 32),
    s(0x11, 0xf2, "movsd", F::WV, 64),
    s(0x12, 0x66, "movlpd", F::VM, 64),
    s(0x12, 0xf2, "movddup", F::VW, 64),
    s(0x12, 0xf3, "movsldup", F::VW, X),
    s(0x13, 0, "movlps", F::MV, 64),
    s(0x13, 0x66, "movlpd", F::MV, 64),
    s(0x14, 0, "unpcklps", F::VW, X),
    s(0x14, 0x66, "unpcklpd", F::VW, X),
    s(0x15, 0, "unpckhps", F::VW, X),
    s(0x15, 0x66, "unpckhpd", F::VW, X),
    s(0x16, 0x66, "movhpd", F::VM, 64),
    s(0x16, 0xf3, "movshdup", F::VW, X),
    s(0x17, 0, "movhps", F::MV, 64),
    s(0x17, 0x66, "movhpd", F::MV, 64),
    s(0x28, 0, "movaps", F::VW, X),
    s(0x28, 0x66, "movapd", F::VW, X),
    s(0x29, 0, "movaps", F::WV, X),
    s(0x29, 0x66, "movapd", F::WV, X),
    s(0x2b, 0, "movntps", F::MV, X),
    s(0x2b, 0x66, "movntpd", F::MV, X),
    s(0x2c, 0xf3, "cvttss2si", F::GyW, 32),
    s(0x2c, 0xf2, "cvttsd2si", F::GyW, 64),
    s(0x2d, 0xf3, "cvtss2si", F::GyW, 32),
    s(0x2d, 0xf2, "cvtsd2si", F::GyW, 64),
    s(0x2e, 0, "ucomiss", F::VW, 32),
    s(0x2e, 0x66, "ucomisd", F::VW, 64),
    s(0x2f, 0, "comiss", F::VW, 32),
    s(0x2f, 0x66, "comisd", F::VW, 64),
    s(0x50, 0, "movmskps", F::GdU, X),
    s(0x50, 0x66, "movmskpd", F::GdU, X),
    s(0x51, 0, "sqrtps", F::VW, X),
    s(0x51, 0x66, "sqrtpd", F::VW, X),
    s(0x51, 0xf3, "sqrtss", F::VW, 32),
    s(0x51, 0xf2, "sqrtsd", F::VW, 64),
    s(0x52, 0, "rsqrtps", F::VW, X),
    s(0x52, 0xf3, "rsqrtss", F::VW, 32),
    s(0x53, 0, "rcpps", F::VW, X),
    s(0x53, 0xf3, "rcpss", F::VW, 32),
    s(0x54, 0, "andps", F::VW, X),
    s(0x54, 0x66, "andpd", F::VW, X),
    s(0x55, 0, "andnps", F::VW, X),
    s(0x55, 0x66, "andnpd", F::VW, X),
    s(0x56, 0, "orps", F::VW, X),
    s(0x56, 0x66, "orpd", F::VW, X),
    s(0x57, 0, "xorps", F::VW, X),
    s(0x57, 0x66, "xorpd", F::VW, X),
    s(0x58, 0, "addps", F::VW, X),
    s(0x58, 0x66, "addpd", F::VW, X),
    s(0x58, 0xf3, "addss", F::VW, 32),
    s(0x58, 0xf2, "addsd", F::VW, 64),
    s(0x59, 0, "mulps", F::VW, X),
    s(0x59, 0x66, "mulpd", F::VW, X),
    s(0x59, 0xf3, "mulss", F::VW, 32),
    s(0x59, 0xf2, "mulsd", F::VW, 64),
    s(0x5a, 0, "cvtps2pd", F::VW, 64),
    s(0x5a, 0x66, "cvtpd2ps", F::VW, X),
    s(0x5a, 0xf3, "cvtss2sd", F::VW, 32),
    s(0x5a, 0xf2, "cvtsd2ss", F::VW, 64),
    s(0x5b, 0, "cvtdq2ps", F::VW, X),
    s(0x5b, 0x66, "cvtps2dq", F::VW, X),
    s(0x5b, 0xf3, "cvttps2dq", F::VW, X),
    s(0x5c, 0, "subps", F::VW, X),
    s(0x5c, 0x66, "subpd", F::VW, X),
    s(0x5c, 0xf3, "subss", F::VW, 32),
    s(0x5c, 0xf2, "subsd", F::VW, 64),
    s(0x5d, 0, "minps", F::VW, X),
    s(0x5d, 0x66, "minpd", F::VW, X),
    s(0x5d, 0xf3, "minss", F::VW, 32),
    s(0x5d, 0xf2, "minsd", F::VW, 64),
    s(0x5e, 0, "divps", F::VW, X),
    s(0x5e, 0x66, "divpd", F::VW, X),
    s(0x5e, 0xf3, "divss", F::VW, 32),
    s(0x5e, 0xf2, "divsd", F::VW, 64),
    s(0x5f, 0, "maxps", F::VW, X),
    s(0x5f, 0x66, "maxpd", F::VW, X),
    s(0x5f, 0xf3, "maxss", F::VW, 32),
    s(0x5f, 0xf2, "maxsd", F::VW, 64),
    s(0x60, 0x66, "punpcklbw", F::VW, X),
    s(0x61, 0x66, "punpcklwd", F::VW, X),
    s(0x62, 0x66, "punpckldq", F::VW, X),
    s(0x63, 0x66, "packsswb", F::VW, X),
    s(0x64, 0x66, "pcmpgtb", F::VW, X),
    s(0x65, 0x66, "pcmpgtw", F::VW, X),
    s(0x66, 0x66, "pcmpgtd", F::VW, X),
    s(0x67, 0x66, "packuswb", F::VW, X),
    s(0x68, 0x66, "punpckhbw", F::VW, X),
    s(0x69, 0x66, "punpckhwd", F::VW, X),
    s(0x6a, 0x66, "punpckhdq", F::VW, X),
    s(0x6b, 0x66, "packssdw", F::VW, X),
    s(0x6c, 0x66, "punpcklqdq", F::VW, X),
    s(0x6d, 0x66, "punpckhqdq", F::VW, X),
    s(0x6f, 0x66, "movdqa", F::VW, X),
    s(0x6f, 0xf3, "movdqu", F::VW, X),
    s(0x70, 0x66, "pshufd", F::VWI, X),
    s(0x70, 0xf2, "pshuflw", F::VWI, X),
    s(0x70, 0xf3, "pshufhw", F::VWI, X),
    s(0x74, 0x66, "pcmpeqb", F::VW, X),
    s(0x75, 0x66, "pcmpeqw", F::VW, X),
    s(0x76, 0x66, "pcmpeqd", F::VW, X),
    s(0x7c, 0x66, "haddpd", F::VW, X),
    s(0x7c, 0xf2, "haddps", F::VW, X),
    s(0x7d, 0x66, "hsubpd", F::VW, X),
    s(0x7d, 0xf2, "hsubps", F::VW, X),
    s(0x7e, 0xf3, "movq", F::VW, 64),
    s(0x7f, 0x66, "movdqa", F::WV, X),
    s(0x7f, 0xf3, "movdqu", F::WV, X),
    s(0xc6, 0, "shufps", F::VWI, X),
    s(0xc6, 0x66, "shufpd", F::VWI, X),
    s(0xd0, 0x66, "addsubpd", F::VW, X),
    s(0xd0, 0xf2, "addsubps", F::VW, X),
    s(0xd1, 0x66, "psrlw", F::VW, X),
    s(0xd2, 0x66, "psrld", F::VW, X),
    s(0xd3, 0x66, "psrlq", F::VW, X),
    s(0xd4, 0x66, "paddq", F::VW, X),
    s(0xd5, 0x66, "pmullw", F::VW, X),
    s(0xd6, 0x66, "movq", F::WV, 64),
    s(0xd7, 0x66, "pmovmskb", F::GdU, X),
    s(0xd8, 0x66, "psubusb", F::VW, X),
    s(0xd9, 0x66, "psubusw", F::VW, X),
    s(0xda, 0x66, "pminub", F::VW, X),
    s(0xdb, 0x66, "pand", F::VW, X),
    s(0xdc, 0x66, "paddusb", F::VW, X),
    s(0xdd, 0x66, "paddusw", F::VW, X),
    s(0xde, 0x66, "pmaxub", F::VW, X),
    s(0xdf, 0x66, "pandn", F::VW, X),
    s(0xe0, 0x66, "pavgb", F::VW, X),
    s(0xe1, 0x66, "psraw", F::VW, X),
    s(0xe2, 0x66, "psrad", F::VW, X),
    s(0xe3, 0x66, "pavgw", F::VW, X),
    s(0xe4, 0x66, "pmulhuw", F::VW, X),
    s(0xe5, 0x66, "pmulhw", F::VW, X),
    s(0xe6, 0x66, "cvttpd2dq", F::VW, X),
    s(0xe6, 0xf3, "cvtdq2pd", F::VW, 64),
    s(0xe6, 0xf2, "cvtpd2dq", F::VW, X),
    s(0xe7, 0x66, "movntdq", F::MV, X),
    s(0xe8, 0x66, "psubsb", F::VW, X),
    s(0xe9, 0x66, "psubsw", F::VW, X),
    s(0xea, 0x66, "pminsw", F::VW, X),
    s(0xeb, 0x66, "por", F::VW, X),
    s(0xec, 0x66, "paddsb", F::VW, X),
    s(0xed, 0x66, "paddsw", F::VW, X),
    s(0xee, 0x66, "pmaxsw", F::VW, X),
    s(0xef, 0x66, "pxor", F::VW, X),
    s(0xf0, 0xf2, "lddqu", F::VM, X),
    s(0xf1, 0x66, "psllw", F::VW, X),
    s(0xf2, 0x66, "pslld", F::VW, X),
    s(0xf3, 0x66, "psllq", F::VW, X),
    s(0xf4, 0x66, "pmuludq", F::VW, X),
    s(0xf5, 0x66, "pmaddwd", F::VW, X),
    s(0xf6, 0x66, "psadbw", F::VW, X),
    s(0xf8, 0x66, "psubb", F::VW, X),
    s(0xf9, 0x66, "psubw", F::VW, X),
    s(0xfa, 0x66, "psubd", F::VW, X),
    s(0xfb, 0x66, "psubq", F::VW, X),
    s(0xfc, 0x66, "paddb", F::VW, X),
    s(0xfd, 0x66, "paddw", F::VW, X),
    s(0xfe, 0x66, "paddd", F::VW, X),
];

/// The `0F 38 xx` (SSSE3/SSE4.1/SSE4.2) opcodes.
static SSE38: &[Sse] = &[
    s(0x00, 0x66, "pshufb", F::VW, X),
    s(0x01, 0x66, "phaddw", F::VW, X),
    s(0x02, 0x66, "phaddd", F::VW, X),
    s(0x04, 0x66, "pmaddubsw", F::VW, X),
    s(0x05, 0x66, "phsubw", F::VW, X),
    s(0x06, 0x66, "phsubd", F::VW, X),
    s(0x08, 0x66, "psignb", F::VW, X),
    s(0x09, 0x66, "psignw", F::VW, X),
    s(0x0a, 0x66, "psignd", F::VW, X),
    s(0x0b, 0x66, "pmulhrsw", F::VW, X),
    s(0x10, 0x66, "pblendvb", F::VW0, X),
    s(0x14, 0x66, "blendvps", F::VW0, X),
    s(0x15, 0x66, "blendvpd", F::VW0, X),
    s(0x17, 0x66, "ptest", F::VW, X),
    s(0x1c, 0x66, "pabsb", F::VW, X),
    s(0x1d, 0x66, "pabsw", F::VW, X),
    s(0x1e, 0x66, "pabsd", F::VW, X),
    s(0x20, 0x66, "pmovsxbw", F::VW, 64),
    s(0x21, 0x66, "pmovsxbd", F::VW, 32),
    s(0x22, 0x66, "pmovsxbq", F::VW, 16),
    s(0x23, 0x66, "pmovsxwd", F::VW, 64),
    s(0x24, 0x66, "pmovsxwq", F::VW, 32),
    s(0x25, 0x66, "pmovsxdq", F::VW, 64),
    s(0x28, 0x66, "pmuldq", F::VW, X),
    s(0x29, 0x66, "pcmpeqq", F::VW, X),
    s(0x2a, 0x66, "movntdqa", F::VM, X),
    s(0x2b, 0x66, "packusdw", F::VW, X),
    s(0x30, 0x66, "pmovzxbw", F::VW, 64),
    s(0x31, 0x66, "pmovzxbd", F::VW, 32),
    s(0x32, 0x66, "pmovzxbq", F::VW, 16),
    s(0x33, 0x66, "pmovzxwd", F::VW, 64),
    s(0x34, 0x66, "pmovzxwq", F::VW, 32),
    s(0x35, 0x66, "pmovzxdq", F::VW, 64),
    s(0x37, 0x66, "pcmpgtq", F::VW, X),
    s(0x38, 0x66, "pminsb", F::VW, X),
    s(0x39, 0x66, "pminsd", F::VW, X),
    s(0x3a, 0x66, "pminuw", F::VW, X),
    s(0x3b, 0x66, "pminud", F::VW, X),
    s(0x3c, 0x66, "pmaxsb", F::VW, X),
    s(0x3d, 0x66, "pmaxsd", F::VW, X),
    s(0x3e, 0x66, "pmaxuw", F::VW, X),
    s(0x3f, 0x66, "pmaxud", F::VW, X),
    s(0x40, 0x66, "pmulld", F::VW, X),
    s(0x41, 0x66, "phminposuw", F::VW, X),
];

/// The `0F 3A xx` (SSSE3/SSE4.1) opcodes, all with an imm8.
static SSE3A: &[Sse] = &[
    s(0x08, 0x66, "roundps", F::VWI, X),
    s(0x09, 0x66, "roundpd", F::VWI, X),
    s(0x0a, 0x66, "roundss", F::VWI, 32),
    s(0x0b, 0x66, "roundsd", F::VWI, 64),
    s(0x0c, 0x66, "blendps", F::VWI, X),
    s(0x0d, 0x66, "blendpd", F::VWI, X),
    s(0x0e, 0x66, "pblendw", F::VWI, X),
    s(0x0f, 0x66, "palignr", F::VWI, X),
    s(0x14, 0x66, "pextrb", F::EdVI, 8),
    s(0x15, 0x66, "pextrw", F::EdVI, 16),
    s(0x16, 0x66, "pextrd", F::EdVI, 32),
    s(0x17, 0x66, "extractps", F::EdVI, 32),
    s(0x20, 0x66, "pinsrb", F::VEdI, 8),
    s(0x21, 0x66, "insertps", F::VWI, 32),
    s(0x22, 0x66, "pinsrd", F::VEdI, 32),
    s(0x40, 0x66, "dpps", F::VWI, X),
    s(0x41, 0x66, "dppd", F::VWI, X),
    s(0x42, 0x66, "mpsadbw", F::VWI, X),
    s(0x44, 0x66, "pclmulqdq", F::VWI, X),
    s(0x60, 0x66, "pcmpestrm", F::VWI, X),
    s(0x61, 0x66, "pcmpestri", F::VWI, X),
    s(0x62, 0x66, "pcmpistrm", F::VWI, X),
    s(0x63, 0x66, "pcmpistri", F::VWI, X),
];
