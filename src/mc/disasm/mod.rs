//! Disassemblers: machine code back to assembly text (ROADMAP Phase 6, the
//! decoder half of the encoder/decoder framework, behind `lf-dis`).
//!
//! There is one decoder per architecture LatticeFoundry emits, each written
//! from that architecture's manual:
//!
//! | Module | Architecture | Syntax |
//! | --- | --- | --- |
//! | [`x86`] | x86-64: the general-purpose ISA, x87, SSE–SSE4.2, VEX (AVX/AVX2, FMA3, BMI1/2) | AT&T (default) or Intel |
//! | [`aarch64`] | AArch64: the A64 base ISA, LSE atomics, scalar FP, Advanced SIMD | the Arm ARM's, with its preferred aliases |
//! | [`riscv`] | RISC-V RV64GC (I, M, A, F, D, C, Zicsr, Zifencei) plus Zba/Zbb | the ISA manual's, with its pseudoinstructions |
//! | [`thumb`] | Thumb-2 (ARMv7-M), with IT blocks | Arm unified assembler language |
//! | [`avr`] | AVR (AVRe+, plus the XMEGA additions) | the AVR Instruction Set Manual's |
//! | [`wasm`] | WebAssembly (MVP, sign extension, saturating truncation, bulk memory, reference types, threads, fixed-width SIMD) | the text format's instruction names |
//!
//! Every decoder turns bytes into a uniform [`Inst`]: how many bytes the
//! instruction takes, its mnemonic, its operands already rendered in the
//! architecture's syntax (and operand order), and — for a direct branch or
//! call — the absolute target address, which a [`listing`] turns into
//! a symbolic label. An encoding a decoder does not know becomes a data
//! directive (`.byte`, `.short`, `.word`, ...) via [`Inst::data`]: decoding
//! never panics and always makes progress, whatever the input.
//!
//! The entry points are [`decode`] (one instruction) and [`disassemble`] (a
//! byte range). [`objfile`] reads the code sections, symbols and relocations
//! of ELF (32/64), `.lfo`, COFF/PE, Mach-O and wasm files, and [`listing`]
//! prints an objdump-style listing with labels and inline relocation notes.
//!
//! Adding an encoding is local to one architecture module: each decoder is a
//! table of match arms over the instruction's fixed fields, so a new
//! instruction is one more arm (plus a round-trip case in its tests).

pub mod aarch64;
pub mod avr;
pub mod listing;
pub mod objfile;
pub mod riscv;
pub mod thumb;
pub mod wasm;
pub mod x86;

#[cfg(test)]
mod tests;

use std::fmt;

use crate::target::TargetArch;

/// The x86 assembly syntax to print (other architectures have one syntax).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Syntax {
    /// AT&T: `movq %rsp, %rbp`, sources first, size suffixes, `%` registers,
    /// `$` immediates.
    #[default]
    Att,
    /// Intel: `mov rbp, rsp`, destination first, `qword ptr` memory sizes.
    Intel,
}

impl Syntax {
    /// Parse `att` / `intel` (case-insensitive).
    pub fn parse(s: &str) -> Option<Syntax> {
        match s.to_ascii_lowercase().as_str() {
            "att" | "at&t" | "gas" => Some(Syntax::Att),
            "intel" => Some(Syntax::Intel),
            _ => None,
        }
    }
}

/// Decoding options.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Options {
    /// The x86 syntax.
    pub syntax: Syntax,
}

impl Options {
    /// Options with the given x86 syntax.
    pub fn with_syntax(syntax: Syntax) -> Options {
        Options { syntax }
    }
}

/// One decoded instruction (or, for bytes no decoder recognizes, a data
/// directive), in the uniform form every architecture produces.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Inst {
    /// Length in bytes (at least one).
    pub len: usize,
    /// The mnemonic, e.g. `movq`, `add`, `ldr`, `i32.const`, `.word`.
    pub mnemonic: String,
    /// The operands, rendered in the architecture's syntax and printing order.
    pub operands: Vec<String>,
    /// For a direct branch or call: the absolute target address.
    pub target: Option<u64>,
    /// Which operand renders [`target`](Inst::target) (the listing replaces it
    /// with a symbol when a relocation patches the instruction).
    pub target_operand: Option<usize>,
    /// `false` for a data directive standing in for an unknown encoding.
    pub known: bool,
}

impl Inst {
    /// A known instruction of `len` bytes with no operands yet.
    pub fn new(len: usize, mnemonic: impl Into<String>) -> Inst {
        Inst {
            len,
            mnemonic: mnemonic.into(),
            operands: Vec::new(),
            target: None,
            target_operand: None,
            known: true,
        }
    }

    /// Append an operand.
    #[must_use]
    pub fn op(mut self, operand: impl Into<String>) -> Inst {
        self.operands.push(operand.into());
        self
    }

    /// Append several operands.
    #[must_use]
    pub fn ops<S: Into<String>>(mut self, operands: impl IntoIterator<Item = S>) -> Inst {
        self.operands.extend(operands.into_iter().map(Into::into));
        self
    }

    /// Append a branch-target operand: the absolute address `addr`, printed
    /// as `0x…`, recorded as the instruction's [`target`](Inst::target).
    #[must_use]
    pub fn target_op(mut self, addr: u64) -> Inst {
        self.target = Some(addr);
        self.target_operand = Some(self.operands.len());
        self.operands.push(format!("{addr:#x}"));
        self
    }

    /// A data directive covering the first `unit` bytes of `bytes` (fewer if
    /// `bytes` is shorter): `.byte` for one byte, `.short` for two, `.word`
    /// for four, `.quad` for eight. The value is read in
    /// `endian` order and printed in hex. Used for every encoding a decoder
    /// does not recognize, and for truncated tails.
    pub fn data(bytes: &[u8], unit: usize, little_endian: bool) -> Inst {
        let unit = unit.clamp(1, 8).min(bytes.len().max(1));
        let unit = if bytes.is_empty() { 1 } else { [1, 2, 4, 8].into_iter().rev().find(|&u| u <= unit).unwrap_or(1) };
        let take = &bytes[..unit.min(bytes.len())];
        let mut v: u64 = 0;
        for (k, &b) in take.iter().enumerate() {
            if little_endian {
                v |= u64::from(b) << (8 * k);
            } else {
                v = v << 8 | u64::from(b);
            }
        }
        let mnemonic = match unit {
            1 => ".byte",
            2 => ".short",
            4 => ".word",
            _ => ".quad",
        };
        Inst {
            len: unit,
            mnemonic: mnemonic.to_owned(),
            operands: vec![format!("{v:#0w$x}", w = 2 + 2 * unit)],
            target: None,
            target_operand: None,
            known: false,
        }
    }

    /// The instruction as `mnemonic<TAB>op1, op2, ...` (no trailing tab for
    /// an instruction without operands).
    pub fn text(&self) -> String {
        if self.operands.is_empty() {
            self.mnemonic.clone()
        } else {
            format!("{}\t{}", self.mnemonic, self.operands.join(", "))
        }
    }
}

impl fmt::Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text())
    }
}

/// The smallest instruction unit of `arch`, in bytes: the step an unknown
/// encoding is skipped by (1 for x86-64 and wasm, 2 for RISC-V with C, Thumb
/// and AVR, 4 for AArch64).
pub fn min_unit(arch: TargetArch) -> usize {
    match arch {
        TargetArch::X86_64 | TargetArch::Wasm32 => 1,
        TargetArch::Riscv64 | TargetArch::Thumb | TargetArch::Avr => 2,
        TargetArch::AArch64 => 4,
    }
}

/// The line-comment marker of `arch`'s assembly syntax (used for relocation
/// notes): `#` for x86-64 and RISC-V, `//` for AArch64, `@` for Thumb, `;`
/// for AVR, `;;` for wasm.
pub fn comment_marker(arch: TargetArch) -> &'static str {
    match arch {
        TargetArch::X86_64 | TargetArch::Riscv64 => "#",
        TargetArch::AArch64 => "//",
        TargetArch::Thumb => "@",
        TargetArch::Avr => ";",
        TargetArch::Wasm32 => ";;",
    }
}

/// Decoding state carried from one instruction to the next: the Thumb
/// `ITSTATE`, which makes the instructions of an `IT` block conditional
/// (`addeq`) and changes how a 16-bit data-processing instruction prints
/// (`add` inside the block, `adds` outside). Other architectures carry none.
/// A fresh [`State::default`] is "outside any IT block".
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct State {
    /// The Thumb `ITSTATE` bits (`firstcond[3:1]` in bits 7:5, then the
    /// condition's low bit and mask in bits 4:0); 0 outside an IT block.
    pub it: u8,
}

/// Decode one instruction of `arch` from the start of `bytes`, which sit at
/// address `addr` (branch targets are absolute). Never panics; for non-empty
/// `bytes` the result's `len` is between 1 and `bytes.len()`. Empty `bytes`
/// give a zero-length `.byte` with no operands. Decodes outside any Thumb
/// IT block; [`decode_in`] carries that state through a sequence.
pub fn decode(arch: TargetArch, bytes: &[u8], addr: u64, opts: &Options) -> Inst {
    decode_in(arch, bytes, addr, opts, &mut State::default())
}

/// [`decode`], reading and advancing the inter-instruction `state`.
pub fn decode_in(arch: TargetArch, bytes: &[u8], addr: u64, opts: &Options, state: &mut State) -> Inst {
    if bytes.is_empty() {
        return Inst { len: 0, mnemonic: ".byte".to_owned(), operands: Vec::new(), target: None, target_operand: None, known: false };
    }
    let inst = match arch {
        TargetArch::X86_64 => x86::decode(bytes, addr, opts),
        TargetArch::AArch64 => aarch64::decode(bytes, addr),
        TargetArch::Riscv64 => riscv::decode(bytes, addr),
        TargetArch::Thumb => thumb::decode_in(bytes, addr, state),
        TargetArch::Avr => avr::decode(bytes, addr),
        TargetArch::Wasm32 => wasm::decode(bytes, addr),
    };
    if inst.len == 0 || inst.len > bytes.len() {
        // A decoder bug must not stall or overrun a listing.
        *state = State::default();
        return Inst::data(bytes, min_unit(arch), true);
    }
    inst
}

/// Decode `bytes` (at address `addr`) as a straight run of `arch`
/// instructions: `(address, instruction)` pairs covering every byte.
pub fn disassemble(arch: TargetArch, bytes: &[u8], addr: u64, opts: &Options) -> Vec<(u64, Inst)> {
    let mut out = Vec::new();
    let mut at = 0;
    let mut state = State::default();
    while at < bytes.len() {
        let inst = decode_in(arch, &bytes[at..], addr.wrapping_add(at as u64), opts, &mut state);
        let len = inst.len.max(1);
        out.push((addr.wrapping_add(at as u64), inst));
        at += len;
    }
    out
}

/// Parse an architecture name as `--arch` accepts it: the
/// [`TargetArch::name`]s plus common spellings (`x86-64`, `amd64`, `arm64`,
/// `riscv`, `rv64`, `thumb`, `armv7m`, `wasm`).
pub fn parse_arch(s: &str) -> Option<TargetArch> {
    Some(match s.to_ascii_lowercase().as_str() {
        "x86_64" | "x86-64" | "amd64" | "x64" => TargetArch::X86_64,
        "aarch64" | "arm64" => TargetArch::AArch64,
        "riscv64" | "riscv" | "rv64" | "rv64gc" | "riscv64gc" => TargetArch::Riscv64,
        "thumb" | "thumbv7m" | "thumbv7em" | "armv7m" | "armv7-m" | "cortex-m" => TargetArch::Thumb,
        "avr" => TargetArch::Avr,
        "wasm32" | "wasm" => TargetArch::Wasm32,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Small shared helpers for the decoders.
// ---------------------------------------------------------------------------

/// Sign-extend the low `bits` bits of `v`.
pub(crate) fn sext(v: u64, bits: u32) -> i64 {
    if bits == 0 || bits >= 64 {
        return v as i64;
    }
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}
