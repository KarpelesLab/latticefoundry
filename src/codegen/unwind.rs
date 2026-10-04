//! Unwind tables, generated from the frame layouts the prologues are built from.
//!
//! An unwinder (a debugger's backtrace, a profiler, the Windows exception
//! dispatcher, a C++ exception) needs to know, for every instruction of a
//! function, where the caller's stack pointer, return address and
//! callee-saved registers are. Each object format has its own table for it;
//! this module produces three of them from one neutral description:
//!
//! | [`UnwindTables`] | format | sections |
//! |---|---|---|
//! | [`Win64`](UnwindTables::Win64) | PE/COFF x64 | `.pdata` (`RUNTIME_FUNCTION`s) + `.xdata` (`UNWIND_INFO`s) |
//! | [`EhFrame`](UnwindTables::EhFrame) | ELF | `.eh_frame` (DWARF call-frame information) |
//! | [`CompactUnwind`](UnwindTables::CompactUnwind) | Mach-O | `__LD,__compact_unwind` (Apple's compact encodings) |
//!
//! # The description
//!
//! A backend describes each function as a [`FunctionFrame`]: the
//! [`FrameStep`]s of its prologue — each prologue instruction's
//! [`FrameOp`] and the offset just past it — plus where its epilogues
//! restore the caller's frame register and return. The steps come from the
//! *same* plan the backend splices into the function as instructions (see
//! `x86_64::encode::FrameLayout::prologue_plan`), and their offsets from
//! encoding those very instructions, so a table cannot drift from the code
//! (the same rule the [stack-usage report](crate::codegen::stack) follows).
//!
//! Registers are numbered by the hardware encoding of the target (on
//! x86-64: `rax`=0, `rcx`=1, …, `rsp`=4, `rbp`=5, …, `r15`=15; `xmm0..15`
//! for [`FrameOp::SaveXmm`]). Each writer maps them to its own numbering
//! (DWARF's differs).
//!
//! # Windows x64
//!
//! [`win64_unwind_info`] encodes an `UNWIND_INFO` (version 1, no handler)
//! following the Microsoft x64 exception-handling specification. The unwind
//! codes run in reverse prologue order: `UWOP_PUSH_NONVOL` per push,
//! `UWOP_ALLOC_SMALL`/`UWOP_ALLOC_LARGE` per fixed allocation,
//! `UWOP_SET_FPREG` where the frame register is established, and
//! `UWOP_SAVE_XMM128` (`_FAR` beyond its reach) per saved `xmm` register.
//! The specification's rules shape the prologue the Windows layout uses:
//!
//! - pushes come first, then the fixed allocation, then the frame register
//!   (`FP = RSP + 16 * FrameOffset`, `FrameOffset` ≤ 15), then the `xmm`
//!   saves, whose offsets are taken from the *frame base* `FP - 16 *
//!   FrameOffset`;
//! - once the frame register is set, the unwinder recomputes `RSP` from it,
//!   so any further stack adjustment needs no code: the rest of the frame
//!   (a large, probed allocation) and every `dyn_alloca` are unwound through
//!   `rbp`. That is how probed frames and dynamic allocation stay
//!   unwindable at every instruction — including in the middle of a probe
//!   loop, where a stack overflow faults.
//!
//! [`emit_win64`] adds the `.xdata` records and a `.pdata` entry per
//! function, whose three fields are image-relative
//! ([`RelocKind::ImageRel32`], COFF `ADDR32NB`) relocations against the
//! `.text` and `.xdata` section symbols. The linker sorts `.pdata` and points
//! the image's exception directory at it.
//!
//! # DWARF `.eh_frame`
//!
//! [`emit_eh_frame`] writes one CIE (`zR` augmentation, `pcrel|sdata4`
//! addresses, code alignment 1, data alignment −8, return address column
//! 16) and an FDE per function whose instructions track the canonical frame
//! address through the prologue — on `rsp` until the frame register is set,
//! on `rbp` after — and record where each register is saved. At every
//! epilogue, after the instruction that pops the frame register, the CFA moves
//! back to `rsp + 8` until the `ret` (`DW_CFA_remember_state` /
//! `DW_CFA_restore_state` around it), so the table is exact at every
//! instruction. The `pc_begin` fields are PC-relative
//! ([`RelocKind::Pc32`]) relocations against the `.text` section symbol.
//!
//! # Mach-O compact unwind
//!
//! [`compact_unwind_x86_64`] gives the `UNWIND_X86_64_MODE_RBP_FRAME`
//! encoding of a function whose frame is `push rbp; mov rbp, rsp` with up to
//! five of `rbx`, `r12`–`r15` saved contiguously just below `rbp`, which is
//! exactly the System V frame. [`emit_compact_unwind`] writes the 32-byte
//! `__compact_unwind` records (function address — an [`Abs64`](RelocKind::Abs64)
//! relocation — length, encoding, no personality, no LSDA); the linker
//! folds them into `__TEXT,__unwind_info`.

use crate::mc::object::{
    ObjectModule, RelocKind, Relocation, Section, SectionId, SectionKind, Symbol, SymbolBinding, SymbolId,
    SymbolType,
};
use crate::target::TargetOs;

/// Which unwind tables a compilation emits (see the [module docs](self)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnwindTables {
    /// None.
    None,
    /// Windows x64 `.pdata`/`.xdata`.
    Win64,
    /// DWARF call-frame information in `.eh_frame` (ELF).
    EhFrame,
    /// Mach-O `__LD,__compact_unwind` records.
    CompactUnwind,
}

impl UnwindTables {
    /// The tables an OS's native object format carries by default: `.pdata`
    /// on Windows and compact unwind on Darwin (their ABIs expect them), none
    /// on Linux and bare metal (`.eh_frame` is opt-in there).
    pub fn default_for(os: TargetOs) -> UnwindTables {
        match os {
            TargetOs::Windows => UnwindTables::Win64,
            TargetOs::Darwin => UnwindTables::CompactUnwind,
            _ => UnwindTables::None,
        }
    }
}

/// What one prologue instruction does to the frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FrameOp {
    /// `push reg`: the stack pointer drops by 8 and `reg` is saved at the new
    /// top.
    Push(u8),
    /// The frame register `reg` is set to `sp + offset`.
    SetFrame {
        /// The frame register.
        reg: u8,
        /// Its distance above the stack pointer.
        offset: u32,
    },
    /// The stack pointer drops by this many bytes (a multiple of 8).
    Alloc(u32),
    /// `xmm<reg>` (all 128 bits) is saved at `fp + fp_offset`, where `fp` is
    /// the frame register (set by an earlier [`FrameOp::SetFrame`]).
    SaveXmm {
        /// The `xmm` register number.
        reg: u8,
        /// The save slot's offset from the frame register.
        fp_offset: i32,
    },
}

/// One prologue instruction: its effect and the function-relative offset just
/// past it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrameStep {
    /// The offset of the first byte after the instruction.
    pub end: u32,
    /// What the instruction does to the frame.
    pub op: FrameOp,
}

/// The unwind description of one function (see the [module docs](self)).
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct FunctionFrame {
    /// The function's offset in its text section.
    pub offset: u64,
    /// The function's size in bytes.
    pub size: u64,
    /// The prologue, in instruction order.
    pub steps: Vec<FrameStep>,
    /// Each epilogue's `(end of the instruction restoring the caller's frame
    /// register, end of the return)` offsets, ascending.
    pub epilogues: Vec<(u32, u32)>,
}

// ===========================================================================
// Windows x64: UNWIND_INFO
// ===========================================================================

const UWOP_PUSH_NONVOL: u16 = 0;
const UWOP_ALLOC_LARGE: u16 = 1;
const UWOP_ALLOC_SMALL: u16 = 2;
const UWOP_SET_FPREG: u16 = 3;
const UWOP_SAVE_XMM128: u16 = 8;
const UWOP_SAVE_XMM128_FAR: u16 = 9;

/// One unwind code node (`CodeOffset | UnwindOp << 8 | OpInfo << 12`).
fn code(offset: u32, op: u16, info: u8) -> u16 {
    offset as u16 | (op << 8) | (u16::from(info & 0xf) << 12)
}

/// Encode `f`'s Windows x64 `UNWIND_INFO` (version 1, no handler).
///
/// # Errors
///
/// A prologue the format cannot describe: a push after the frame register is
/// set, a frame-register offset that is not a multiple of 16 up to 240, an
/// `xmm` save without a frame register or below the frame base, a prologue
/// longer than 255 bytes or with more than 255 code slots, or an allocation
/// that is not a multiple of 8.
pub fn win64_unwind_info(f: &FunctionFrame) -> Result<Vec<u8>, String> {
    // Each prologue operation's code nodes, in prologue order.
    let mut groups: Vec<Vec<u16>> = Vec::new();
    let mut frame: Option<(u8, u32)> = None;
    let mut prolog_end = 0u32;
    for step in &f.steps {
        let end = step.end;
        let nodes = match step.op {
            FrameOp::Push(r) => {
                if frame.is_some() {
                    return Err(format!("a push (register {r}) after the frame register is set"));
                }
                vec![code(end, UWOP_PUSH_NONVOL, r)]
            }
            // Below the frame register the unwinder recovers `RSP` from it,
            // so an allocation there (the rest of a large frame, probed) needs
            // no code.
            FrameOp::Alloc(_) if frame.is_some() => continue,
            FrameOp::Alloc(n) => {
                if n == 0 || n % 8 != 0 {
                    return Err(format!("a stack allocation of {n} bytes (not a positive multiple of 8)"));
                }
                if n <= 128 {
                    vec![code(end, UWOP_ALLOC_SMALL, (n / 8 - 1) as u8)]
                } else if n <= 512 * 1024 - 8 {
                    vec![code(end, UWOP_ALLOC_LARGE, 0), (n / 8) as u16]
                } else {
                    vec![code(end, UWOP_ALLOC_LARGE, 1), n as u16, (n >> 16) as u16]
                }
            }
            FrameOp::SetFrame { reg, offset } => {
                if offset % 16 != 0 || offset > 240 {
                    return Err(format!("a frame register offset of {offset} (not a multiple of 16 up to 240)"));
                }
                frame = Some((reg, offset));
                vec![code(end, UWOP_SET_FPREG, 0)]
            }
            FrameOp::SaveXmm { reg, fp_offset } => {
                let Some((_, fo)) = frame else {
                    return Err(format!("xmm{reg} saved before the frame register is set"));
                };
                // Offsets are from the frame base, `FP - FrameOffset`.
                let base = i64::from(fp_offset) + i64::from(fo);
                let Ok(base) = u32::try_from(base) else {
                    return Err(format!("xmm{reg} saved below the frame base"));
                };
                if base % 16 == 0 && base / 16 <= 0xffff {
                    vec![code(end, UWOP_SAVE_XMM128, reg), (base / 16) as u16]
                } else {
                    vec![code(end, UWOP_SAVE_XMM128_FAR, reg), base as u16, (base >> 16) as u16]
                }
            }
        };
        prolog_end = end;
        groups.push(nodes);
    }
    if prolog_end > 255 {
        return Err(format!("a {prolog_end}-byte prologue (at most 255)"));
    }
    // The array runs in descending prologue offset: the groups reversed, each
    // keeping its node order.
    let nodes: Vec<u16> = groups.into_iter().rev().flatten().collect();
    if nodes.len() > 255 {
        return Err(format!("{} unwind code slots (at most 255)", nodes.len()));
    }
    let (frame_reg, frame_off) = frame.map_or((0, 0), |(r, o)| (r, (o / 16) as u8));
    let mut out = vec![1, prolog_end as u8, nodes.len() as u8, (frame_reg & 0xf) | (frame_off << 4)];
    for n in &nodes {
        out.extend_from_slice(&n.to_le_bytes());
    }
    // An even number of slots keeps the structure `DWORD`-aligned.
    if nodes.len() % 2 == 1 {
        out.extend_from_slice(&[0, 0]);
    }
    Ok(out)
}

/// A local section symbol for `section` named `name`, for relocations that
/// must not follow a preemptible or weak function symbol.
fn section_symbol(obj: &mut ObjectModule, section: SectionId, name: &str) -> SymbolId {
    obj.add_symbol(Symbol::defined(name, SymbolBinding::Local, SymbolType::Section, section, 0, 0))
}

/// Add the Windows x64 `.xdata` and `.pdata` sections describing `funcs`
/// (functions of the section `text`) to `obj`.
///
/// # Errors
///
/// A function [`win64_unwind_info`] cannot describe.
pub fn emit_win64(obj: &mut ObjectModule, text: SectionId, funcs: &[FunctionFrame]) -> Result<(), String> {
    if funcs.is_empty() {
        return Ok(());
    }
    let mut xdata = Vec::new();
    let mut at = Vec::with_capacity(funcs.len());
    for f in funcs {
        at.push(xdata.len() as i64);
        xdata.extend(win64_unwind_info(f)?);
    }
    let text_name = obj.section(text).name.clone();
    let text_sym = section_symbol(obj, text, &text_name);
    let mut xs = Section::new(".xdata", SectionKind::Rodata, 4);
    xs.bytes = xdata;
    let xdata = obj.add_section(xs);
    let xdata_sym = section_symbol(obj, xdata, ".xdata");
    let mut ps = Section::new(".pdata", SectionKind::Rodata, 4);
    ps.bytes = vec![0; 12 * funcs.len()];
    let pdata = obj.add_section(ps);
    for (k, f) in funcs.iter().enumerate() {
        let base = 12 * k as u64;
        for (field, symbol, addend) in [
            (0, text_sym, f.offset as i64),
            (4, text_sym, (f.offset + f.size) as i64),
            (8, xdata_sym, at[k]),
        ] {
            obj.add_relocation(Relocation {
                section: pdata,
                offset: base + field,
                symbol,
                kind: RelocKind::ImageRel32,
                addend,
            });
        }
    }
    Ok(())
}

// ===========================================================================
// DWARF call-frame information (.eh_frame), x86-64
// ===========================================================================

const DW_CFA_ADVANCE_LOC: u8 = 0x40;
const DW_CFA_OFFSET: u8 = 0x80;
const DW_CFA_NOP: u8 = 0x00;
const DW_CFA_ADVANCE_LOC1: u8 = 0x02;
const DW_CFA_ADVANCE_LOC2: u8 = 0x03;
const DW_CFA_ADVANCE_LOC4: u8 = 0x04;
const DW_CFA_OFFSET_EXTENDED: u8 = 0x05;
const DW_CFA_REMEMBER_STATE: u8 = 0x0a;
const DW_CFA_RESTORE_STATE: u8 = 0x0b;
const DW_CFA_DEF_CFA: u8 = 0x0c;
const DW_CFA_DEF_CFA_OFFSET: u8 = 0x0e;
/// `DW_EH_PE_pcrel | DW_EH_PE_sdata4`.
const DW_EH_PE_PCREL_SDATA4: u8 = 0x1b;

/// The x86-64 DWARF register number of hardware GPR `r` (the psABI's
/// numbering: `rax rdx rcx rbx rsi rdi rbp rsp r8..r15`).
fn dwarf_gpr(r: u8) -> u8 {
    const LOW: [u8; 8] = [0, 2, 1, 3, 7, 6, 4, 5];
    LOW.get(usize::from(r)).copied().unwrap_or(r)
}
/// The x86-64 DWARF number of the return address column.
const DWARF_RA: u8 = 16;
/// The x86-64 DWARF number of `rsp`.
const DWARF_RSP: u8 = 7;
/// The x86-64 DWARF number of `xmm0` (`xmm<n>` is `17 + n`).
const DWARF_XMM0: u64 = 17;

fn uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn sleb(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && byte & 0x40 == 0) || (v == -1 && byte & 0x40 != 0);
        out.push(if done { byte } else { byte | 0x80 });
        if done {
            return;
        }
    }
}

/// Advance the CFI location from `*loc` to `to`.
fn advance(out: &mut Vec<u8>, loc: &mut u32, to: u32) {
    let delta = to.saturating_sub(*loc);
    if delta == 0 {
        return;
    }
    if delta < 0x40 {
        out.push(DW_CFA_ADVANCE_LOC | delta as u8);
    } else if delta <= 0xff {
        out.extend_from_slice(&[DW_CFA_ADVANCE_LOC1, delta as u8]);
    } else if delta <= 0xffff {
        out.push(DW_CFA_ADVANCE_LOC2);
        out.extend_from_slice(&(delta as u16).to_le_bytes());
    } else {
        out.push(DW_CFA_ADVANCE_LOC4);
        out.extend_from_slice(&delta.to_le_bytes());
    }
    *loc = to;
}

/// Record that DWARF register `reg` is saved at `CFA - cfa_minus` (a multiple
/// of 8: the data alignment factor is −8).
fn saved_at(out: &mut Vec<u8>, reg: u64, cfa_minus: i64) {
    if reg < 0x40 {
        out.push(DW_CFA_OFFSET | reg as u8);
    } else {
        out.push(DW_CFA_OFFSET_EXTENDED);
        uleb(out, reg);
    }
    uleb(out, (cfa_minus / 8) as u64);
}

/// The call-frame instructions of an x86-64 function's FDE (see the
/// [module docs](self)).
pub fn cfi_x86_64(f: &FunctionFrame) -> Vec<u8> {
    let mut out = Vec::new();
    let mut loc = 0u32;
    // The CFA is the stack pointer before the call: on entry `rsp + 8`.
    // `depth` is CFA - rsp; `fp` the frame register and the CFA's distance
    // above it, once set.
    let mut depth: i64 = 8;
    let mut fp: Option<(u8, i64)> = None;
    for step in &f.steps {
        advance(&mut out, &mut loc, step.end);
        match step.op {
            FrameOp::Push(r) => {
                depth += 8;
                if fp.is_none() {
                    out.push(DW_CFA_DEF_CFA_OFFSET);
                    uleb(&mut out, depth as u64);
                }
                saved_at(&mut out, u64::from(dwarf_gpr(r)), depth);
            }
            FrameOp::Alloc(n) => {
                depth += i64::from(n);
                if fp.is_none() {
                    out.push(DW_CFA_DEF_CFA_OFFSET);
                    uleb(&mut out, depth as u64);
                }
            }
            FrameOp::SetFrame { reg, offset } => {
                let above = depth - i64::from(offset);
                fp = Some((reg, above));
                out.push(DW_CFA_DEF_CFA);
                uleb(&mut out, u64::from(dwarf_gpr(reg)));
                uleb(&mut out, above as u64);
            }
            FrameOp::SaveXmm { reg, fp_offset } => {
                if let Some((_, above)) = fp {
                    saved_at(&mut out, DWARF_XMM0 + u64::from(reg), above - i64::from(fp_offset));
                }
            }
        }
    }
    if fp.is_some() {
        for &(pop_end, ret_end) in &f.epilogues {
            advance(&mut out, &mut loc, pop_end);
            out.push(DW_CFA_REMEMBER_STATE);
            out.extend_from_slice(&[DW_CFA_DEF_CFA, DWARF_RSP, 8]);
            if u64::from(ret_end) < f.size {
                advance(&mut out, &mut loc, ret_end);
                out.push(DW_CFA_RESTORE_STATE);
            }
        }
    }
    out
}

/// Pad a CFI record (whose 4-byte length field starts at `start`) with
/// `DW_CFA_nop` to a multiple of 8 bytes and fill in its length.
fn close_record(buf: &mut Vec<u8>, start: usize) {
    while !(buf.len() - start).is_multiple_of(8) {
        buf.push(DW_CFA_NOP);
    }
    let len = (buf.len() - start - 4) as u32;
    buf[start..start + 4].copy_from_slice(&len.to_le_bytes());
}

/// Add an x86-64 `.eh_frame` section describing `funcs` (functions of the
/// section `text`) to `obj`: one CIE, then an FDE per function.
pub fn emit_eh_frame(obj: &mut ObjectModule, text: SectionId, funcs: &[FunctionFrame]) {
    if funcs.is_empty() {
        return;
    }
    let mut buf = Vec::new();
    // --- CIE ---
    buf.extend_from_slice(&[0; 4]); // length
    buf.extend_from_slice(&0u32.to_le_bytes()); // CIE id
    buf.push(1); // version
    buf.extend_from_slice(b"zR\0");
    uleb(&mut buf, 1); // code alignment factor
    sleb(&mut buf, -8); // data alignment factor
    buf.push(DWARF_RA); // return address register
    uleb(&mut buf, 1); // augmentation data length
    buf.push(DW_EH_PE_PCREL_SDATA4);
    // On entry: CFA = rsp + 8, the return address at CFA - 8.
    buf.extend_from_slice(&[DW_CFA_DEF_CFA, DWARF_RSP, 8]);
    saved_at(&mut buf, u64::from(DWARF_RA), 8);
    close_record(&mut buf, 0);

    // --- FDEs ---
    let mut pc_fields = Vec::with_capacity(funcs.len());
    for f in funcs {
        let start = buf.len();
        buf.extend_from_slice(&[0; 4]); // length
        let cie_pointer = (buf.len()) as u32; // distance back to the CIE (at 0)
        buf.extend_from_slice(&cie_pointer.to_le_bytes());
        pc_fields.push((buf.len() as u64, f.offset as i64));
        buf.extend_from_slice(&[0; 4]); // pc_begin (relocated)
        buf.extend_from_slice(&(f.size as u32).to_le_bytes()); // pc_range
        uleb(&mut buf, 0); // augmentation data length
        buf.extend(cfi_x86_64(f));
        close_record(&mut buf, start);
    }

    let text_name = obj.section(text).name.clone();
    let text_sym = section_symbol(obj, text, &text_name);
    let mut s = Section::new(".eh_frame", SectionKind::Rodata, 8);
    s.bytes = buf;
    let eh = obj.add_section(s);
    for (offset, addend) in pc_fields {
        obj.add_relocation(Relocation { section: eh, offset, symbol: text_sym, kind: RelocKind::Pc32, addend });
    }
}

// ===========================================================================
// Mach-O compact unwind
// ===========================================================================

/// `UNWIND_X86_64_MODE_RBP_FRAME`.
const UNWIND_X86_64_MODE_RBP_FRAME: u32 = 0x0100_0000;
/// `UNWIND_ARM64_MODE_FRAME`.
pub const UNWIND_ARM64_MODE_FRAME: u32 = 0x0400_0000;

/// The compact-unwind register code of x86-64 GPR `r` (`rbx`=1, `r12`..`r15`
/// = 2..5), if it has one.
fn compact_reg(r: u8) -> Option<u32> {
    match r {
        3 => Some(1),
        12..=15 => Some(u32::from(r) - 10),
        _ => None,
    }
}

/// The `UNWIND_X86_64_MODE_RBP_FRAME` encoding of `f`, or `None` if its frame
/// is not that shape (`push rbp`, `rbp` pointing at the saved `rbp`, at most
/// five of `rbx`/`r12`–`r15` saved contiguously below it, no `xmm` saves).
pub fn compact_unwind_x86_64(f: &FunctionFrame) -> Option<u32> {
    const RBP: u8 = 5;
    let mut depth: i64 = 8; // CFA - rsp
    let mut rbp_saved_at = None; // CFA - x
    let mut rbp_at = None; // CFA - x, once set
    let mut saved: Vec<(u32, i64)> = Vec::new(); // (register code, CFA - x)
    for step in &f.steps {
        match step.op {
            FrameOp::Push(r) => {
                depth += 8;
                if r == RBP {
                    rbp_saved_at = Some(depth);
                } else {
                    saved.push((compact_reg(r)?, depth));
                }
            }
            FrameOp::Alloc(n) => depth += i64::from(n),
            FrameOp::SetFrame { reg, offset } => {
                if reg != RBP {
                    return None;
                }
                rbp_at = Some(depth - i64::from(offset));
            }
            FrameOp::SaveXmm { .. } => return None,
        }
    }
    // rbp must hold the address of the saved rbp, which sits right below the
    // return address.
    if rbp_saved_at != Some(16) || rbp_at != Some(16) || saved.len() > 5 {
        return None;
    }
    // The registers occupy rbp-8, rbp-16, ... (CFA-24, CFA-32, ...); entry 0
    // of the encoding is the lowest address.
    saved.sort_by_key(|&(_, at)| std::cmp::Reverse(at));
    let n = saved.len() as i64;
    let mut regs = 0u32;
    for (i, &(code, at)) in saved.iter().enumerate() {
        if at != 16 + 8 * (n - i as i64) {
            return None;
        }
        regs |= code << (3 * i);
    }
    Some(UNWIND_X86_64_MODE_RBP_FRAME | ((n as u32) << 16) | regs)
}

/// Add a Mach-O `__LD,__compact_unwind` section (named `__compact_unwind`
/// in the object model; the Mach-O writer places it) with one record per
/// `(function symbol, size, encoding)`.
pub fn emit_compact_unwind(obj: &mut ObjectModule, records: &[(SymbolId, u64, u32)]) {
    if records.is_empty() {
        return;
    }
    let mut s = Section::new("__compact_unwind", SectionKind::Rodata, 8);
    for &(_, size, encoding) in records {
        s.bytes.extend_from_slice(&0u64.to_le_bytes()); // function (relocated)
        s.bytes.extend_from_slice(&(size as u32).to_le_bytes());
        s.bytes.extend_from_slice(&encoding.to_le_bytes());
        s.bytes.extend_from_slice(&[0; 16]); // personality, LSDA
    }
    let cu = obj.add_section(s);
    for (k, &(symbol, _, _)) in records.iter().enumerate() {
        obj.add_relocation(Relocation {
            section: cu,
            offset: 32 * k as u64,
            symbol,
            kind: RelocKind::Abs64,
            addend: 0,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The System V frame: `push rbp; mov rbp, rsp; push rbx; push r12;
    /// sub rsp, 32`, one epilogue.
    fn sysv() -> FunctionFrame {
        FunctionFrame {
            offset: 0,
            size: 40,
            steps: vec![
                FrameStep { end: 1, op: FrameOp::Push(5) },
                FrameStep { end: 4, op: FrameOp::SetFrame { reg: 5, offset: 0 } },
                FrameStep { end: 5, op: FrameOp::Push(3) },
                FrameStep { end: 7, op: FrameOp::Push(12) },
                FrameStep { end: 11, op: FrameOp::Alloc(32) },
            ],
            epilogues: vec![(38, 39)],
        }
    }

    /// The Windows frame: `push rbp; push rsi; sub rsp, 40; lea rbp,
    /// [rsp+48]; movups [rbp-32], xmm6; sub rsp, 4096`.
    fn win() -> FunctionFrame {
        FunctionFrame {
            offset: 16,
            size: 64,
            steps: vec![
                FrameStep { end: 1, op: FrameOp::Push(5) },
                FrameStep { end: 2, op: FrameOp::Push(6) },
                FrameStep { end: 6, op: FrameOp::Alloc(40) },
                FrameStep { end: 11, op: FrameOp::SetFrame { reg: 5, offset: 48 } },
                FrameStep { end: 15, op: FrameOp::SaveXmm { reg: 6, fp_offset: -32 } },
                FrameStep { end: 22, op: FrameOp::Alloc(4096) },
            ],
            epilogues: vec![(60, 61)],
        }
    }

    #[test]
    fn win64_codes_reverse_the_prologue() {
        let info = win64_unwind_info(&win()).unwrap();
        // Version 1, prolog 15 bytes, 6 slots, frame rbp at offset 3*16.
        assert_eq!(&info[..4], &[1, 15, 6, 5 | (3 << 4)]);
        let slots: Vec<u16> = info[4..].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(
            slots,
            [
                code(15, UWOP_SAVE_XMM128, 6),
                1, // frame base = rbp - 48; slot rbp-32 is base+16
                code(11, UWOP_SET_FPREG, 0),
                code(6, UWOP_ALLOC_SMALL, 4),
                code(2, UWOP_PUSH_NONVOL, 6),
                code(1, UWOP_PUSH_NONVOL, 5),
            ]
        );
    }

    #[test]
    fn win64_rejects_what_it_cannot_express() {
        assert!(win64_unwind_info(&sysv()).is_err(), "a push after the frame register");
        let mut f = win();
        f.steps[3].op = FrameOp::SetFrame { reg: 5, offset: 8 };
        assert!(win64_unwind_info(&f).is_err());
    }

    #[test]
    fn win64_large_allocations() {
        for (n, want) in [(136u32, vec![code(4, UWOP_ALLOC_LARGE, 0), 17]), (1 << 20, vec![code(4, UWOP_ALLOC_LARGE, 1), 0, 16])] {
            let f = FunctionFrame { size: 8, steps: vec![FrameStep { end: 4, op: FrameOp::Alloc(n) }], ..FunctionFrame::default() };
            let info = win64_unwind_info(&f).unwrap();
            let slots: Vec<u16> = info[4..4 + 2 * want.len()].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            assert_eq!(slots, want);
            assert_eq!(info[3], 0, "no frame register");
        }
    }

    #[test]
    fn cfi_tracks_the_frame() {
        let cfi = cfi_x86_64(&sysv());
        #[rustfmt::skip]
        let want = [
            0x41, 0x0e, 16, 0x86, 2,     // push rbp: CFA=rsp+16, rbp at CFA-16
            0x43, 0x0c, 6, 16,           // mov rbp,rsp: CFA=rbp+16
            0x41, 0x83, 3,               // push rbx: CFA-24
            0x42, 0x8c, 4,               // push r12: CFA-32
            0x44,                        // sub rsp (CFA on rbp: nothing)
            0x40 | 27, 0x0a, 0x0c, 7, 8, // pop rbp: remember; CFA=rsp+8
            0x41, 0x0b,                  // ret: restore
        ];
        assert_eq!(cfi, want);
    }

    #[test]
    fn leb128() {
        let mut v = Vec::new();
        sleb(&mut v, -8);
        uleb(&mut v, 624_485);
        sleb(&mut v, 64);
        assert_eq!(v, [0x78, 0xe5, 0x8e, 0x26, 0xc0, 0x00]);
    }

    #[test]
    fn compact_rbp_frame() {
        // rbx at rbp-8, r12 at rbp-16: offset 2, entry 0 (rbp-16) = r12 (2),
        // entry 1 (rbp-8) = rbx (1).
        assert_eq!(compact_unwind_x86_64(&sysv()), Some(0x0100_0000 | (2 << 16) | 2 | (1 << 3)));
        assert_eq!(compact_unwind_x86_64(&win()), None, "rsi and xmm saves have no encoding");
    }
}
