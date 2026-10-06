//! The RISC-V RV64IMAFD machine opcode set ([`RvOp`]) and the
//! instruction-selection rules.
//!
//! [`RvOp`] is this target's [`Opcode`] vocabulary: a *post-isel, pre-encoding*
//! MIR whose operands are still MIR [`MachineOperand`]s (registers, immediates,
//! frame slots, labels, symbol references). RISC-V data-processing instructions
//! are genuinely three-address (`add rd, rs1, rs2`), so the isel emits one MIR op
//! per IR op with a clean `[Def d, Use a, Use b]` shape and the encoder never has
//! to synthesize a move-to-destination. A few IR ops still expand to a short
//! RISC-V idiom at encode time (a comparison becomes `slt`/`sltu` plus `xori`/
//! `seqz`/`snez`; a `select` becomes a branchless mask sequence; a remainder is a
//! plain `rem`/`remu`; a constant is a `lui`/`addi` materialization).
//!
//! ## x0 tricks, block arguments, calls, returns
//!
//! `x0` is hardwired zero: `mv rd, rs` is `addi rd, rs, 0`, a zero constant is a
//! read of `x0`, `seqz`/`snez` compare against `x0`, and `ret` is `jalr x0, ra,
//! 0`. Block arguments are realized by the framework's edge-move mechanism
//! ([`Lower::edge_to`]). Calls, the entry prologue and returns follow the LP64D
//! convention (the `abi` module): scalars in `a0`–`a7` / `fa0`–`fa7` (floats
//! overflowing into integer registers, then 8-byte stack slots in the
//! outgoing-argument area at the bottom of the caller's frame), by-value
//! structs flattened into floating-point and integer registers or passed by
//! reference, variadic arguments by the integer convention, and results in
//! `a0`/`a1`/`fa0`/`fa1` or through a hidden `a0` pointer. A struct value is,
//! at this level, a pointer to its storage (as on AArch64). RV64M has
//! hardware divide/remainder, so `div`/`divu`/`rem`/`remu` lower directly —
//! no fixed-register dance.
//!
//! ## Floating point (the F and D extensions)
//!
//! `f32`/`f64` values live in the `f` registers (`f32` NaN-boxed). Arithmetic
//! is `fadd`/`fsub`/`fmul`/`fdiv` in the dynamic rounding mode
//! (round-to-nearest-even under the default `fcsr`, as the IR requires); a
//! multiply feeding an add or subtract becomes one fused `fmadd`/`fmsub`/
//! `fnmsub` **only** when both carry `contract` (the IR's license to skip the
//! intermediate rounding). `fneg` is sign injection (`fsgnjn`), exact for
//! NaNs. `fcmp` is `feq`/`flt`/`fle` (which yield 0 for unordered operands)
//! with an `xori` for the unordered predicates, and `and`/`or` for `ord`/`one`
//! — no branch. Float-to-integer conversions truncate (`rtz`) with the
//! saturating `fcvt.{w,wu,l,lu}` (an out-of-range input is poison in the IR,
//! so saturation refines it), integer-to-float conversions use the 32-bit
//! forms for `i32` and extend anything narrower to 64 bits first, and
//! `frem` calls C's `fmod`/`fmodf`. A float `select` blends the bit patterns
//! in GPRs. `f16` (the Zfh extension) is not supported.
//!
//! ## Narrow values
//!
//! Every data-processing op is the full-width RV64 form, so a narrow value
//! (`i1`/`i8`/`i16`/`i32`, or an odd `_BitInt` width) lives in a 64-bit register
//! whose bits above its width are **not** kept clean: an `i8` add of 200 + 100
//! leaves 300 there, and a `trunc` is a plain move. Ops that only feed the low
//! bits (`add`, `mul`, `shl`, logic, stores) don't care; every op whose result
//! depends on the upper bits (compares, right shifts, division, `zext`/`sext`,
//! branch/select conditions, `switch`, integer-to-float conversions) first
//! extends via `RiscvTarget::extend64` (`sext.w`, `andi`, or an
//! `slli`+`srai`/`srli` pair). At call boundaries the LP64 psABI's
//! signedness-independent rules are honored: an `i32` argument/return is
//! sign-extended (`sext.w`) and an `i1` zero-extended, so foreign callees see
//! ABI-conformant registers.
//!
//! ## Atomics (the A extension)
//!
//! Atomics use the A extension (`lr`/`sc`, `amo*`) and `fence`, so code that
//! contains them needs an RV64IMA core; code without them stays RV64IM. See
//! `RiscvTarget::lower_atomic` for the mapping.
//!
//! Deferred (noted for a follow-up): the RV64 word forms (`addw`/`divw`/
//! `sraw`/...) as a cheaper `i32` lowering; `f16`; and the callee side of
//! variadic functions (`va_start`). Integers wider than 64 bits are
//! legalized into 64-bit parts ([`crate::codegen::wide`]); an `i128` crosses
//! the ABI in a register pair, as the psABI's 2×XLEN scalars do.

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::wide::{self, WideIsel, WideTable};
use crate::codegen::mir::{
    MBlockId, MachineInst, MachineOperand, Opcode, PReg, Reg, RegClass, StackSlot, VReg,
};
use crate::codegen::target::{CallConv, MachineTarget};
use crate::ir::inst::{BinOp, CastOp, FloatPred, InstKind, IntPred, UnaryOp};
use crate::ir::types::Type;
use crate::ir::value::{Const, ValueDef};
use crate::ir::{InstData, Module, ValueId};
use crate::support::StrInterner;

use puremp::Int;

use super::abi::{self, Assigner, Loc, Part};
use super::regs::{self, RegFile, fpr, gpr};

/// The RV64IM MIR opcode vocabulary. Operand layouts are documented per variant;
/// `Def`/`Use` are register operands, the rest are immediates, frame slots, branch
/// labels, or symbol references. `Imm width` is the operation's IR integer bit
/// width, informational only: the encoder always emits the full 64-bit form and
/// the interpreter models exactly that (see "Narrow values" above).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum RvOp {
    /// `[Def d, Use s]` — `mv d, s` (`addi d, s, 0`).
    Mv = 0,
    /// `[Def d, Imm v]` — load immediate via a `lui`/`addi` materialization.
    Li = 1,
    /// `[Def d, Use a, Use b, Imm width]` — `add d, a, b`.
    Add = 2,
    /// `[Def d, Use a, Use b, Imm width]` — `sub d, a, b`.
    Sub = 3,
    /// `[Def d, Use a, Use b, Imm width]` — `and d, a, b`.
    And = 4,
    /// `[Def d, Use a, Use b, Imm width]` — `or d, a, b`.
    Or = 5,
    /// `[Def d, Use a, Use b, Imm width]` — `xor d, a, b`.
    Xor = 6,
    /// `[Def d, Use a, Use b, Imm width]` — `mul d, a, b`.
    Mul = 7,
    /// `[Def d, Use a, Use b, Imm width]` — `mulh d, a, b` (signed high half).
    Mulh = 8,
    /// `[Def d, Use a, Imm k, Imm width]` — `addi d, a, #k`.
    Addi = 9,
    /// `[Def d, Use a, Imm k, Imm width]` — `andi d, a, #k`.
    Andi = 10,
    /// `[Def d, Use a, Imm k, Imm width]` — `ori d, a, #k`.
    Ori = 11,
    /// `[Def d, Use a, Imm k, Imm width]` — `xori d, a, #k`.
    Xori = 12,
    /// `[Def d, Use a, Use b, Imm width]` — `div d, a, b` (signed).
    Div = 13,
    /// `[Def d, Use a, Use b, Imm width]` — `divu d, a, b` (unsigned).
    Divu = 14,
    /// `[Def d, Use a, Use b, Imm width]` — `rem d, a, b` (signed).
    Rem = 15,
    /// `[Def d, Use a, Use b, Imm width]` — `remu d, a, b` (unsigned).
    Remu = 16,
    /// `[Def d, Use a, Imm shamt, Imm width]` — `slli d, a, #shamt`.
    Slli = 17,
    /// `[Def d, Use a, Imm shamt, Imm width]` — `srli d, a, #shamt`.
    Srli = 18,
    /// `[Def d, Use a, Imm shamt, Imm width]` — `srai d, a, #shamt`.
    Srai = 19,
    /// `[Def d, Use a, Use b, Imm width]` — `sll d, a, b`.
    Sll = 20,
    /// `[Def d, Use a, Use b, Imm width]` — `srl d, a, b`.
    Srl = 21,
    /// `[Def d, Use a, Use b, Imm width]` — `sra d, a, b`.
    Sra = 22,
    /// `[Def d, Use a, Use b, Imm pred, Imm width]` — set-if-condition into a GPR,
    /// synthesized from `slt`/`sltu` and `xori`/`seqz`/`snez` (see `pred_code`).
    SetCmp = 23,
    /// `[Def d, Use cond, Use t, Use f]` — branchless `d = cond ? t : f` (a mask
    /// blend). This is how `select` lowers, so a `select` on a secret condition
    /// runs without a branch (`docs/ir-design.md` §6d).
    Select = 24,
    /// `[Def d, Use ptr, Imm size]` — load `size` bytes from `[ptr]` (zero-extended).
    Load = 25,
    /// `[Use ptr, Use val, Imm size]` — store `size` bytes to `[ptr]`.
    Store = 26,
    /// `[Def d, Frame slot]` — `addi d, sp, #slot_off`.
    FrameAddr = 27,
    /// `[Def d, Global g, Imm got]` — the address of global `g`:
    /// `auipc d, %pcrel_hi(g); addi d, d, %pcrel_lo(g)`, or with `got` set its
    /// GOT entry's contents (`auipc d, %got_pcrel_hi(g); ld d, %pcrel_lo(d)`).
    GlobalAddr = 28,
    /// `[Func f | Use callee, Def a0, Def clobbers.., Use args..]` — call.
    Call = 29,
    /// `[]` — return (value already in a0; `jalr x0, ra, 0`).
    Ret = 30,
    /// `[Label t]` — unconditional jump (`jal x0, t`).
    J = 31,
    /// `[Use cond, Label t, Label f]` — `bnez cond, t; j f`.
    BrCond = 32,
    /// `[Use cond, Label default, (Imm val, Label case)...]` — multi-way branch.
    Switch = 33,
    /// `[]` — a trap (`ebreak`).
    Unreachable = 34,
    /// `[Use src, Frame slot]` — spill: `sd src, [sp, #slot_off]`.
    StoreFrame = 35,
    /// `[Def dst, Frame slot]` — reload: `ld dst, [sp, #slot_off]`.
    LoadFrame = 36,
    /// `[Imm delta]` — `addi sp, sp, #delta` (signed; prologue/epilogue).
    AddiSp = 37,
    /// `[Use r, Imm off]` — `sd r, [sp, #off]` (callee-saved / ra save).
    SaveReg = 38,
    /// `[Def r, Imm off]` — `ld r, [sp, #off]` (callee-saved / ra restore).
    RestoreReg = 39,
    /// `[Def a0, Use a7, Use a0..]` — the Linux environment call `ecall`. The
    /// syscall number is in `a7` and the arguments in `a0..a5` (moved there by
    /// isel as one consecutive run right before); the kernel returns in `a0` and
    /// preserves every other register, so `a0` is the only def.
    Ecall = 40,
    /// `[Def d, Use s]` — `sext.w d, s` (`addiw d, s, 0`): sign-extend the low
    /// 32 bits of `s` to 64.
    SextW = 41,

    // --- the A extension (atomics; see `lower_atomic` for the mapping) ------
    /// `[Imm fm, Imm pred, Imm succ]` — `fence pred, succ` (`fm` = 8 with
    /// `rw, rw` is `fence.tso`). `pred`/`succ` are the 4-bit `iorw` sets.
    Fence = 42,
    /// `[Def d, Use ptr, Use val, Imm size, Imm op, Imm aqrl]` — an atomic
    /// read-modify-write ([`RmwOp::code`](crate::ir::RmwOp::code) `op`), `d` =
    /// the old value. A 4/8-byte op with an AMO is one `amo<op>.{w,d}` (`sub`
    /// negates into `t0` first); `nand` and every 1/2-byte op expand at encode
    /// time into an LR/SC retry loop (the narrow ones on the aligned word, with
    /// the lane masked by shifts). `aqrl` bit 0 = acquire (`.aq` on the AMO /
    /// `lr`), bit 1 = `.rl` on the AMO / `lr` (`seq_cst`), bit 2 = `.rl` on
    /// `sc`. Clobbers `t0`, `t1`, `t2`, `t6` (never allocated).
    AtomicRmw = 43,
    /// `[Def d, Use ptr, Use expected, Use new, Imm size, Imm aqrl]` — a strong
    /// compare-and-exchange LR/SC loop (narrow sizes on the aligned word); `d` =
    /// the old value. `aqrl` as for [`RvOp::AtomicRmw`]. Clobbers `t0`, `t1`,
    /// `t2`, `t6`.
    CmpXchg = 44,
    /// `[Def d, Func f, Imm got]` — the address of function `f` (a function
    /// used as a value), as [`RvOp::GlobalAddr`].
    FuncAddr = 45,

    // --- the F and D extensions (scalar floating point) --------------------
    /// `[Def d, Use a, Use b, Imm width]` — `fadd.{s,d} d, a, b` (`width` 32
    /// or 64 picks the format; the rounding mode is the dynamic one, i.e.
    /// round-to-nearest-even under the default `fcsr`).
    FAdd = 46,
    /// `[Def d, Use a, Use b, Imm width]` — `fsub.{s,d}`.
    FSub = 47,
    /// `[Def d, Use a, Use b, Imm width]` — `fmul.{s,d}`.
    FMul = 48,
    /// `[Def d, Use a, Use b, Imm width]` — `fdiv.{s,d}`.
    FDiv = 49,
    /// `[Def d, Use a, Use b, Use c, Imm width, Imm kind]` — a fused
    /// multiply-add with one rounding: `kind` 0 `fmadd` (`a*b + c`), 1
    /// `fmsub` (`a*b - c`), 2 `fnmsub` (`-(a*b) + c`), 3 `fnmadd`. Selected
    /// only where the IR licenses contraction (`contract` on both the
    /// multiply and the add).
    FMadd = 50,
    /// `[Def d, Use a, Use b, Imm width, Imm funct3]` — sign injection
    /// `fsgnj{,n,x}.{s,d}` (`funct3` 0/1/2): `fneg` is `fsgnjn d, a, a` and a
    /// float-to-float move `fsgnj d, a, a`.
    FSgnj = 51,
    /// `[Def d, Use a, Use b, Imm pred, Imm width]` — a floating-point compare
    /// into a GPR (exactly 0 or 1), from `feq`/`flt`/`fle` plus `xori`/`and`/
    /// `or` (see `fcmp_code`); branch-free. Clobbers `t0`.
    FCmp = 52,
    /// `[Def d, Imm bits, Imm width]` — a float constant: its IEEE bit
    /// pattern materialized in `t0` and moved over (`fmv.{w,d}.x`).
    FLi = 53,
    /// `[Def d, Use s, Imm dst_width, Imm src_width]` — `fcvt.d.s` /
    /// `fcvt.s.d`.
    FCvtFF = 54,
    /// `[Def d, Use s, Imm signed, Imm int_width, Imm float_width]` —
    /// `fcvt.{w,wu,l,lu}.{s,d} d, s, rtz` (`int_width` 32 or 64): float to
    /// integer, truncating. The 32-bit forms sign-extend their result.
    FCvtFI = 55,
    /// `[Def d, Use s, Imm signed, Imm int_width, Imm float_width]` —
    /// `fcvt.{s,d}.{w,wu,l,lu}`: integer to float (the 32-bit forms read the
    /// low word only).
    FCvtIF = 56,
    /// `[Def d, Use s, Imm width]` — `fmv.x.w` (sign-extending) / `fmv.x.d`:
    /// a float's bits into a GPR.
    FMvXF = 57,
    /// `[Def d, Use s, Imm width]` — `fmv.w.x` (NaN-boxing) / `fmv.d.x`: a
    /// GPR's low bits into a float register.
    FMvFX = 58,

    // --- stack arguments, dynamic allocation, the frame pointer ----------------
    /// `[Def d, Imm off]` — `addi d, sp, off`: an address in the outgoing
    /// stack-argument area at the bottom of the frame.
    LeaSp = 59,
    /// `[Def d, Imm off]` — the address of the incoming stack argument at
    /// `off` (the caller's `sp + off`, i.e. this frame's base + frame size +
    /// `off`).
    LeaInArg = 60,
    /// `[Def d, Use n, Imm align]` — `dyn_alloca`: move `sp` down by `n`
    /// (rounded up to 16, probed when stack probes are on) and return an
    /// `align`-aligned pointer to the new block; the outgoing-argument area is
    /// kept at the bottom of the frame by relocating it below the block. Only
    /// in a function with a frame pointer. Clobbers `t0`, `t1`.
    DynAlloca = 61,
    /// `[]` — `addi s0, sp, 0` (prologue of a frame-pointer function).
    FpSetup = 62,
    /// `[]` — `addi sp, s0, 0` (epilogue of a frame-pointer function).
    FpRestore = 63,
    /// `[]` — `sd zero, 0(sp)` (prologue): touch the bottom of a frame whose
    /// saved registers sit above an outgoing-argument area, keeping the
    /// stack-probe invariant for the frames it calls.
    TouchSp = 64,
    /// `[Def d, Use a, Use b, Imm width]` — `mulhu d, a, b` (unsigned high
    /// half): the high part of a 128-bit product (`docs/ir-design.md` §3b).
    Mulhu = 65,
}

impl RvOp {
    /// The MIR [`Opcode`] id for this opcode.
    #[inline]
    pub fn opcode(self) -> Opcode {
        Opcode(self as u32)
    }

    /// Whether an instruction of this opcode may execute a conditional branch
    /// whose direction depends on a register operand — the constant-time
    /// audit of the lowering (`docs/ir-design.md` §6d): the terminators
    /// `BrCond`/`Switch`, the LR/SC loops of `AtomicRmw` (narrow widths,
    /// `nand`, and the min/max compares) and `CmpXchg`, and `DynAlloca`'s
    /// probe loop over its size. Everything else — in particular `Select` (a
    /// mask blend: `t & -c | f & ~-c`), `SetCmp` (`slt`/`sltu`/`xor`), the
    /// variable shifts, `Mul`, and every floating-point op (`FCmp` is
    /// `feq`/`flt`/`fle` with `xori`/`and`/`or`; the conversions are single
    /// saturating instructions, with no fix-up branches) — is straight-line
    /// code. (The prologue's probe loop counts a constant frame size.)
    pub fn may_branch_on_data(self, _operands: &[MachineOperand]) -> bool {
        matches!(
            self,
            RvOp::BrCond | RvOp::Switch | RvOp::AtomicRmw | RvOp::CmpXchg | RvOp::DynAlloca
        )
    }

    /// Decode a MIR [`Opcode`] back to an [`RvOp`].
    pub fn decode(op: Opcode) -> RvOp {
        use RvOp::*;
        const TABLE: [RvOp; 66] = [
            Mv, Li, Add, Sub, And, Or, Xor, Mul, Mulh, Addi, Andi, Ori, Xori, Div, Divu, Rem, Remu,
            Slli, Srli, Srai, Sll, Srl, Sra, SetCmp, Select, Load, Store, FrameAddr, GlobalAddr,
            Call, Ret, J, BrCond, Switch, Unreachable, StoreFrame, LoadFrame, AddiSp, SaveReg,
            RestoreReg, Ecall, SextW, Fence, AtomicRmw, CmpXchg, FuncAddr, FAdd, FSub, FMul, FDiv,
            FMadd, FSgnj, FCmp, FLi, FCvtFF, FCvtFI, FCvtIF, FMvXF, FMvFX, LeaSp, LeaInArg,
            DynAlloca, FpSetup, FpRestore, TouchSp, Mulhu,
        ];
        TABLE[op.0 as usize]
    }
}

/// A switch case value sign-extended from the scrutinee's `width` to 64 bits,
/// matching the sign-extended scrutinee it is compared against.
fn sext_case(value: &Int, width: u32) -> Int {
    if width >= 64 {
        return value.clone();
    }
    let raw = value.to_i64().map(|v| v as u64).or_else(|| value.to_u64()).unwrap_or(0);
    let shift = 64 - width;
    Int::from_i64(((raw << shift) as i64) >> shift)
}

/// A dense code for an [`IntPred`], packed into the [`RvOp::SetCmp`] immediate and
/// decoded by the encoder and interpreter.
pub(crate) fn pred_code(p: IntPred) -> u8 {
    match p {
        IntPred::Eq => 0,
        IntPred::Ne => 1,
        IntPred::Ult => 2,
        IntPred::Ule => 3,
        IntPred::Ugt => 4,
        IntPred::Uge => 5,
        IntPred::Slt => 6,
        IntPred::Sle => 7,
        IntPred::Sgt => 8,
        IntPred::Sge => 9,
    }
}

/// A dense code for a non-constant [`FloatPred`], packed into the
/// [`RvOp::FCmp`] immediate and decoded by the encoder and interpreter:
///
/// | code | predicate | sequence |
/// |---|---|---|
/// | 0 | `oeq` | `feq d, a, b` |
/// | 1 | `olt` | `flt d, a, b` |
/// | 2 | `ole` | `fle d, a, b` |
/// | 3 | `ogt` | `flt d, b, a` |
/// | 4 | `oge` | `fle d, b, a` |
/// | 5 | `ord` | `feq t0, a, a; feq d, b, b; and d, d, t0` |
/// | 6 | `one` | `flt t0, a, b; flt d, b, a; or d, d, t0` |
///
/// and `code + 8` is the negation (an `xori d, d, 1` after): `une` = !`oeq`,
/// `uge` = !`olt`, `ugt` = !`ole`, `ule` = !`ogt`, `ult` = !`oge`, `uno` =
/// !`ord`, `ueq` = !`one`. `feq`/`flt`/`fle` yield 0 when either operand is a
/// NaN, which is exactly the ordered reading.
pub(crate) fn fcmp_code(p: FloatPred) -> Option<u8> {
    Some(match p {
        FloatPred::Oeq => 0,
        FloatPred::Olt => 1,
        FloatPred::Ole => 2,
        FloatPred::Ogt => 3,
        FloatPred::Oge => 4,
        FloatPred::Ord => 5,
        FloatPred::One => 6,
        FloatPred::Une => 8,
        FloatPred::Uge => 9,
        FloatPred::Ugt => 10,
        FloatPred::Ule => 11,
        FloatPred::Ult => 12,
        FloatPred::Uno => 13,
        FloatPred::Ueq => 14,
        FloatPred::False | FloatPred::True => return None,
    })
}

/// Round `v` up to a multiple of `align` (a power of two ≥ 1).
fn align_up(v: u64, align: u64) -> u64 {
    let a = align.max(1);
    v.div_ceil(a) * a
}

/// The store width for an aggregate part of `size` bytes: the size itself
/// when it is a machine width, else a whole doubleword (the destination is a
/// home slot rounded up to 8 bytes, and a part starts at offset 0 or 8).
fn part_store_size(size: u64) -> u64 {
    if matches!(size, 1 | 2 | 4 | 8) { size } else { 8 }
}

fn def(r: PReg) -> MachineOperand {
    MachineOperand::Def(Reg::Physical(r))
}
fn use_p(r: PReg) -> MachineOperand {
    MachineOperand::Use(Reg::Physical(r))
}
fn def_v(v: VReg) -> MachineOperand {
    MachineOperand::Def(Reg::Virtual(v))
}
fn use_v(v: VReg) -> MachineOperand {
    MachineOperand::Use(Reg::Virtual(v))
}
fn imm(v: u64) -> MachineOperand {
    MachineOperand::Imm(Int::from_u64(v))
}

/// The RISC-V RV64 target: its register file/ABI plus the isel + encoding rules.
#[derive(Debug)]
pub struct RiscvTarget {
    rf: RegFile,
    /// Per module global: whether its address is loaded from the GOT (a
    /// preemptible symbol under PIC/PIE; see [`crate::codegen::linkage`]).
    /// Empty (everything PC-relative) for [`RiscvTarget::new`].
    global_got: Vec<bool>,
    /// Per module function: likewise, for a function used as a value.
    func_got: Vec<bool>,
    /// The module's `fmodf` and `fmod` (function indices), which `frem`
    /// calls.
    fmod: [Option<u32>; 2],
    /// Whether `s0` is the frame pointer (the function moves `sp` at run
    /// time: `dyn_alloca`).
    frame_pointer: bool,
    /// The module's wide-integer helpers by name (function indices): the
    /// inline multiply placeholder and the `i128`/float conversions.
    wide_helpers: Vec<(&'static str, u32)>,
    /// The higher parts of the function's `i128` values.
    wide: WideTable,
}

impl Default for RiscvTarget {
    fn default() -> Self {
        Self::new()
    }
}

impl RiscvTarget {
    /// Construct the RV64 target with its fixed register file and LP64 ABI.
    pub fn new() -> RiscvTarget {
        RiscvTarget {
            rf: RegFile::new(false),
            global_got: Vec::new(),
            func_got: Vec::new(),
            fmod: [None; 2],
            frame_pointer: false,
            wide_helpers: Vec::new(),
            wide: WideTable::default(),
        }
    }

    /// This target with `s0` reserved as the frame pointer (for a function
    /// that uses `dyn_alloca`).
    pub fn with_frame_pointer(mut self, on: bool) -> RiscvTarget {
        self.frame_pointer = on;
        self.rf = RegFile::new(on);
        self
    }

    /// Whether `s0` is reserved as the frame pointer.
    pub fn frame_pointer(&self) -> bool {
        self.frame_pointer
    }

    /// The target for compiling `module` under `opts`: under a
    /// position-independent [`RelocModel`](crate::codegen::RelocModel), the
    /// addresses of symbols that may bind outside the component come from the
    /// GOT.
    ///
    /// With `syms`, the module's `fmod`/`fmodf` declarations (which `frem`
    /// calls; see `encode::prepare`) are found by name.
    pub fn for_module(
        module: &Module,
        syms: Option<&StrInterner>,
        opts: &crate::codegen::CodegenOptions,
    ) -> RiscvTarget {
        use crate::codegen::linkage::{func_binds_locally, global_binds_locally};
        let model = opts.reloc_model;
        let global_got = (0..module.global_count())
            .map(|g| !global_binds_locally(module, crate::ir::GlobalId::from_index(g), model))
            .collect();
        let func_got = (0..module.function_count())
            .map(|f| !func_binds_locally(module, crate::ir::FuncId::from_index(f), model))
            .collect();
        let find = |name: &str| -> Option<u32> {
            let syms = syms?;
            module.functions().position(|f| syms.resolve(f.name) == name).map(|i| i as u32)
        };
        RiscvTarget {
            rf: RegFile::new(false),
            global_got,
            func_got,
            fmod: [find("fmodf"), find("fmod")],
            frame_pointer: false,
            wide_helpers: wide::helper_names().filter_map(|n| Some((n, find(n)?))).collect(),
            wide: WideTable::default(),
        }
    }

    /// Lower function `func` of `module` to MIR over this target.
    pub fn select(
        &self,
        module: &Module,
        func: crate::ir::FuncId,
    ) -> crate::codegen::mir::MachineFunction {
        self.wide.borrow_mut().clear();
        crate::codegen::isel::select(self, module, func)
    }

    /// If `v` is an integer constant operand, its value.
    fn const_of(lo: &Lower<'_, Self>, v: ValueId) -> Option<Int> {
        if let ValueDef::Const(c) = lo.func().value(v).def
            && let Const::Int { value, .. } = lo.module().consts().get(c)
        {
            return Some(value.clone());
        }
        None
    }

    /// Whether an integer constant fits the RISC-V 12-bit signed immediate field
    /// (`addi`/`andi`/`ori`/`xori`), i.e. `-2048 ..= 2047`.
    fn fits_imm12(c: &Int) -> Option<u64> {
        let v = c.to_i64()?;
        if (-2048..=2047).contains(&v) { Some((v as u64) & 0xFFF) } else { None }
    }

    /// Whether `v` is a compare result, whose register holds exactly 0 or 1.
    fn is_compare(lo: &Lower<'_, Self>, v: ValueId) -> bool {
        matches!(lo.func().value(v).def, ValueDef::Inst(id)
            if matches!(lo.func().inst(id).kind, InstKind::ICmp(_) | InstKind::FCmp(_)))
    }

    /// `v` sign- or zero-extended from its width to all 64 bits of a register.
    /// Narrow values live in 64-bit registers whose upper bits are not kept
    /// clean, so anything that reads those bits extends first: `sext.w` for a
    /// signed `i32`, `andi` for an unsigned width of at most 11 bits (the mask
    /// fits the sign-extended 12-bit immediate), else an `slli` + `srai`/`srli`
    /// pair. A 64-bit value, and a compare result being zero-extended, are
    /// already clean.
    fn extend64(&self, lo: &mut Lower<'_, Self>, v: ValueId, signed: bool) -> VReg {
        let r = lo.reg(v);
        let width = lo.int_width(v);
        if width >= 64 || (!signed && Self::is_compare(lo, v)) {
            return r;
        }
        let d = lo.fresh_vreg(RegClass::Gpr);
        if signed && width == 32 {
            lo.emit(MachineInst::new(RvOp::SextW.opcode(), vec![def_v(d), use_v(r)]));
        } else if !signed && width <= 11 {
            let m = (1u64 << width) - 1;
            lo.emit(MachineInst::new(
                RvOp::Andi.opcode(),
                vec![def_v(d), use_v(r), imm(m), imm(64)],
            ));
        } else {
            let k = u64::from(64 - width);
            let t = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(MachineInst::new(RvOp::Slli.opcode(), vec![def_v(t), use_v(r), imm(k), imm(64)]));
            let right = if signed { RvOp::Srai } else { RvOp::Srli };
            lo.emit(MachineInst::new(right.opcode(), vec![def_v(d), use_v(t), imm(k), imm(64)]));
        }
        d
    }

    /// An `i1` branch/select condition as a register holding exactly 0 or 1
    /// (`bnez` tests all 64 bits, and the branchless select blends with
    /// `0 - cond`). A compare's result already is; anything else (e.g. a
    /// `trunc` to `i1`) may carry garbage above bit 0.
    fn clean_cond(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        self.extend64(lo, v, false)
    }

    /// A call argument / return value as the LP64 psABI wants it in a register:
    /// an `i32` sign-extended to 64 bits (whatever its C signedness) and an `i1`
    /// (`_Bool`) zero-extended. Other narrow widths depend on the C type's
    /// signedness, which the IR does not carry, and are passed as is.
    fn abi_value(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        let is_int = matches!(lo.types().get(lo.func().value_type(v)), Type::Int(_));
        match lo.int_width(v) {
            32 if is_int => self.extend64(lo, v, true),
            1 if is_int => self.extend64(lo, v, false),
            _ => lo.reg(v),
        }
    }

    /// Lower an atomic memory operation or fence with the A extension, following
    /// the RVWMO mapping of the RISC-V ISA manual (Vol. I, "Memory model"
    /// appendix, the mapping of C/C++ atomics):
    ///
    /// | IR | RISC-V |
    /// |---|---|
    /// | `atomic_load relaxed` | `l{b,h,w,d}` |
    /// | `atomic_load acquire` | `l*; fence r,rw` |
    /// | `atomic_load seq_cst` | `fence rw,rw; l*; fence r,rw` |
    /// | `atomic_store relaxed` | `s{b,h,w,d}` |
    /// | `atomic_store release`/`seq_cst` | `fence rw,w; s*` |
    /// | `atomic_rmw` (32/64-bit, not `nand`) | `amo<op>.{w,d}{.aq}{.rl}` |
    /// | `atomic_rmw nand`, any 8/16-bit rmw | LR/SC loop ([`RvOp::AtomicRmw`]) |
    /// | `cmpxchg` | LR/SC loop ([`RvOp::CmpXchg`]) |
    /// | `fence acquire` / `release` / `acq_rel` / `seq_cst` | `fence r,rw` / `fence rw,w` / `fence.tso` / `fence rw,rw` |
    ///
    /// Acquire orderings set `.aq` (on the AMO or the `lr`), release orderings
    /// `.rl` (on the AMO or the `sc`), and `seq_cst` both, with `lr.aqrl` in a
    /// loop.
    fn lower_atomic(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        use crate::ir::inst::AtomicOrdering;
        const R: u64 = 0b0010;
        const W: u64 = 0b0001;
        const RW: u64 = R | W;
        let fence = |lo: &mut Lower<'_, Self>, fm: u64, pred: u64, succ: u64| {
            lo.emit(MachineInst::new(RvOp::Fence.opcode(), vec![imm(fm), imm(pred), imm(succ)]));
        };
        // The `aqrl` immediate of the rmw/cmpxchg pseudos.
        let aqrl = |acq: bool, rel: bool, seq: bool| -> u64 {
            u64::from(acq) | (u64::from(seq) << 1) | (u64::from(rel) << 2)
        };
        let ops = inst.operands();
        match &inst.kind {
            InstKind::AtomicLoad { ty, ordering, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(ops[0]);
                let size = lo.byte_size(*ty);
                if *ordering == AtomicOrdering::SeqCst {
                    fence(lo, 0, RW, RW);
                }
                lo.emit(MachineInst::new(RvOp::Load.opcode(), vec![def_v(d), use_v(ptr), imm(size)]));
                if ordering.is_acquire() {
                    fence(lo, 0, R, RW);
                }
            }
            InstKind::AtomicStore { ty, ordering, .. } => {
                let ptr = lo.reg(ops[0]);
                let val = lo.reg(ops[1]);
                let size = lo.byte_size(*ty);
                if ordering.is_release() {
                    fence(lo, 0, RW, W);
                }
                lo.emit(MachineInst::new(RvOp::Store.opcode(), vec![use_v(ptr), use_v(val), imm(size)]));
            }
            InstKind::AtomicRmw { op, ty, ordering, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(ops[0]);
                let val = lo.reg(ops[1]);
                let size = lo.byte_size(*ty);
                let seq = *ordering == AtomicOrdering::SeqCst;
                lo.emit(MachineInst::new(
                    RvOp::AtomicRmw.opcode(),
                    vec![
                        def_v(d),
                        use_v(ptr),
                        use_v(val),
                        imm(size),
                        imm(u64::from(op.code())),
                        imm(aqrl(ordering.is_acquire(), ordering.is_release(), seq)),
                    ],
                ));
            }
            InstKind::CmpXchg { ty, success, failure, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(ops[0]);
                let expected = lo.reg(ops[1]);
                let new = lo.reg(ops[2]);
                let size = lo.byte_size(*ty);
                let acq = success.is_acquire() || failure.is_acquire();
                let seq = *success == AtomicOrdering::SeqCst;
                lo.emit(MachineInst::new(
                    RvOp::CmpXchg.opcode(),
                    vec![
                        def_v(d),
                        use_v(ptr),
                        use_v(expected),
                        use_v(new),
                        imm(size),
                        imm(aqrl(acq, success.is_release(), seq)),
                    ],
                ));
            }
            InstKind::Fence(ordering) => match ordering {
                AtomicOrdering::Acquire => fence(lo, 0, R, RW),
                AtomicOrdering::Release => fence(lo, 0, RW, W),
                AtomicOrdering::AcqRel => fence(lo, 0b1000, RW, RW),
                _ => fence(lo, 0, RW, RW),
            },
            other => unreachable!("lower_atomic on {other:?}"),
        }
    }

    fn lower_bin(&self, lo: &mut Lower<'_, Self>, op: BinOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let width = lo.int_width(inst.operands()[0]);
        // The commutative/associative immediate-friendly ops that have an I-type
        // form: try to fold a small constant RHS into `addi`/`andi`/`ori`/`xori`.
        let imm_form = match op {
            BinOp::Add => Some(RvOp::Addi),
            BinOp::And => Some(RvOp::Andi),
            BinOp::Or => Some(RvOp::Ori),
            BinOp::Xor => Some(RvOp::Xori),
            _ => None,
        };
        if let Some(iop) = imm_form
            && let Some(c) = Self::const_of(lo, inst.operands()[1])
            && let Some(u) = Self::fits_imm12(&c)
        {
            let a = lo.reg(inst.operands()[0]);
            lo.emit(MachineInst::new(
                iop.opcode(),
                vec![def_v(d), use_v(a), imm(u), imm(u64::from(width))],
            ));
            return;
        }
        // Plain register three-address forms.
        let simple = match op {
            BinOp::Add => Some(RvOp::Add),
            BinOp::Sub => Some(RvOp::Sub),
            BinOp::And => Some(RvOp::And),
            BinOp::Or => Some(RvOp::Or),
            BinOp::Xor => Some(RvOp::Xor),
            BinOp::Mul => Some(RvOp::Mul),
            _ => None,
        };
        if let Some(x) = simple {
            let a = lo.reg(inst.operands()[0]);
            let b = lo.reg(inst.operands()[1]);
            lo.emit(MachineInst::new(
                x.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        match op {
            BinOp::Shl => self.lower_shift(lo, RvOp::Slli, RvOp::Sll, d, inst, width),
            BinOp::LShr => self.lower_shift(lo, RvOp::Srli, RvOp::Srl, d, inst, width),
            BinOp::AShr => self.lower_shift(lo, RvOp::Srai, RvOp::Sra, d, inst, width),
            BinOp::UDiv => self.lower_div(lo, RvOp::Divu, false, d, inst, width),
            BinOp::SDiv => self.lower_div(lo, RvOp::Div, true, d, inst, width),
            BinOp::URem => self.lower_div(lo, RvOp::Remu, false, d, inst, width),
            BinOp::SRem => self.lower_div(lo, RvOp::Rem, true, d, inst, width),
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv => self.lower_fbin(lo, op, d, inst),
            BinOp::FRem => self.lower_frem(lo, d, inst),
            other => unreachable!("{other:?} reached the RISC-V isel (legalized away)"),
        }
    }

    /// The float width (32 or 64) of a value. `f16` needs the Zfh extension,
    /// which this backend does not target.
    fn float_width(lo: &Lower<'_, Self>, v: ValueId) -> u32 {
        match lo.types().get(lo.func().value_type(v)) {
            Type::Float(k) if matches!(k.bit_width(), 32 | 64) => k.bit_width(),
            Type::Float(_) => panic!("riscv64 backend: f16 arithmetic needs the Zfh extension"),
            other => panic!("riscv64 backend: a float operation on {other:?}"),
        }
    }

    /// If `v` is a `contract` `fmul` with no other use, its operands: the
    /// multiply a `contract` add or subtract may fuse into one `fmadd`-family
    /// instruction (a single rounding, which `contract` licenses).
    fn contractible_fmul(lo: &Lower<'_, Self>, v: ValueId) -> Option<(ValueId, ValueId)> {
        let ValueDef::Inst(id) = lo.func().value(v).def else { return None };
        let m = lo.func().inst(id);
        (m.kind == InstKind::Bin(BinOp::FMul)
            && m.flags.fast.contract
            && lo.func().uses_of(v).len() == 1)
            .then(|| (m.operands()[0], m.operands()[1]))
    }

    /// Which operand (0 or 1) of the `contract` `fadd`/`fsub` `user` is a
    /// multiply it fuses with, if any (the first one that qualifies).
    fn fusion_operand(lo: &Lower<'_, Self>, user: &InstData) -> Option<usize> {
        if !matches!(user.kind, InstKind::Bin(BinOp::FAdd | BinOp::FSub)) || !user.flags.fast.contract {
            return None;
        }
        (0..2).find(|&k| Self::contractible_fmul(lo, user.operands()[k]).is_some())
    }

    /// Whether `inst` is a multiply that its single user fuses: it then emits
    /// nothing of its own.
    fn fused_away(lo: &Lower<'_, Self>, inst: &InstData) -> bool {
        let Some(r) = inst.result() else { return false };
        if Self::contractible_fmul(lo, r).is_none() {
            return false;
        }
        let u = lo.func().uses_of(r)[0];
        Self::fusion_operand(lo, lo.func().inst(u.inst)) == Some(u.operand as usize)
    }

    /// `fadd`/`fsub`/`fmul`/`fdiv`, fusing a `contract` multiply-add.
    fn lower_fbin(&self, lo: &mut Lower<'_, Self>, op: BinOp, d: VReg, inst: &InstData) {
        let w = u64::from(Self::float_width(lo, inst.operands()[0]));
        if let Some(k) = Self::fusion_operand(lo, inst) {
            let (ma, mb) = Self::contractible_fmul(lo, inst.operands()[k]).expect("fusable");
            let other = inst.operands()[1 - k];
            // a*b + c: fmadd; a*b - c: fmsub; c - a*b: fnmsub.
            let kind = match (op, k) {
                (BinOp::FAdd, _) => 0,
                (_, 0) => 1,
                _ => 2,
            };
            let (a, b, c) = (lo.reg(ma), lo.reg(mb), lo.reg(other));
            lo.emit(MachineInst::new(
                RvOp::FMadd.opcode(),
                vec![def_v(d), use_v(a), use_v(b), use_v(c), imm(w), imm(kind)],
            ));
            return;
        }
        let rop = match op {
            BinOp::FAdd => RvOp::FAdd,
            BinOp::FSub => RvOp::FSub,
            BinOp::FMul => RvOp::FMul,
            _ => RvOp::FDiv,
        };
        let a = lo.reg(inst.operands()[0]);
        let b = lo.reg(inst.operands()[1]);
        lo.emit(MachineInst::new(rop.opcode(), vec![def_v(d), use_v(a), use_v(b), imm(w)]));
    }

    /// `frem` is C's `fmod`/`fmodf` (the remainder of the truncated
    /// quotient): a call to the C library function, which the module driver
    /// declared (`encode::prepare`).
    fn lower_frem(&self, lo: &mut Lower<'_, Self>, d: VReg, inst: &InstData) {
        let w = Self::float_width(lo, inst.operands()[0]);
        let f = self.fmod[usize::from(w == 64)].unwrap_or_else(|| {
            panic!("riscv64 backend: `frem` needs `fmod`/`fmodf` declared (compile with compile_module)")
        });
        let a = lo.reg(inst.operands()[0]);
        let b = lo.reg(inst.operands()[1]);
        let (fa0, fa1) = (fpr(regs::FA0), fpr(regs::FA0 + 1));
        lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(fa0), use_v(a)]));
        lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(fa1), use_v(b)]));
        let mut operands = vec![MachineOperand::Func(f), def(self.rf.cc.ret_reg)];
        for &cs in &self.rf.caller_saved {
            if cs != self.rf.cc.ret_reg {
                operands.push(def(cs));
            }
        }
        operands.push(use_p(fa0));
        operands.push(use_p(fa1));
        lo.emit(MachineInst::new(RvOp::Call.opcode(), operands));
        lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_p(fa0)]));
    }

    /// Division / remainder. The 64-bit `div`/`rem` see every bit of both
    /// operands, so a narrow dividend and divisor are extended by the
    /// operation's signedness first.
    #[allow(clippy::too_many_arguments)]
    fn lower_div(
        &self,
        lo: &mut Lower<'_, Self>,
        rop: RvOp,
        signed: bool,
        d: VReg,
        inst: &InstData,
        width: u32,
    ) {
        let a = self.extend64(lo, inst.operands()[0], signed);
        let b = self.extend64(lo, inst.operands()[1], signed);
        lo.emit(MachineInst::new(
            rop.opcode(),
            vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
        ));
    }

    fn lower_shift(
        &self,
        lo: &mut Lower<'_, Self>,
        imm_op: RvOp,
        var_op: RvOp,
        d: VReg,
        inst: &InstData,
        width: u32,
    ) {
        // A right shift brings the bits above the width down into the result.
        let a = match imm_op {
            RvOp::Srli => self.extend64(lo, inst.operands()[0], false),
            RvOp::Srai => self.extend64(lo, inst.operands()[0], true),
            _ => lo.reg(inst.operands()[0]),
        };
        if let Some(c) = Self::const_of(lo, inst.operands()[1]) {
            let shmask = if width >= 64 { 63 } else { u64::from(width) - 1 };
            let shamt = c.to_u64().unwrap_or(0) & shmask;
            lo.emit(MachineInst::new(
                imm_op.opcode(),
                vec![def_v(d), use_v(a), imm(shamt), imm(u64::from(width))],
            ));
        } else {
            // The hardware takes the count from the low 6 bits of `rs2`; a count
            // narrower than that may carry garbage inside those bits.
            let b = if lo.int_width(inst.operands()[1]) < 6 {
                self.extend64(lo, inst.operands()[1], false)
            } else {
                lo.reg(inst.operands()[1])
            };
            lo.emit(MachineInst::new(
                var_op.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
        }
    }

    /// Conversions. Float↔float and int↔float go through `fcvt`; `zext`/`sext`
    /// (and `inttoptr` from a narrower integer) extend from the source's width,
    /// since its register's upper bits are not clean; a bitcast between a float
    /// and an integer moves the bits across files (`fmv`); truncation, ptr→int
    /// and same-file bitcasts are low-bits-preserving copies.
    fn lower_cast(&self, lo: &mut Lower<'_, Self>, op: CastOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let src = inst.operands()[0];
        let emit = |lo: &mut Lower<'_, Self>, o: RvOp, s: VReg, imms: &[u64]| {
            let mut ops = vec![def_v(d), use_v(s)];
            ops.extend(imms.iter().map(|&v| imm(v)));
            lo.emit(MachineInst::new(o.opcode(), ops));
        };
        let float_bits = |lo: &Lower<'_, Self>, ty| match lo.types().get(ty) {
            Type::Float(k) => Some(k.bit_width()),
            _ => None,
        };
        match op {
            CastOp::FpTrunc | CastOp::FpExt => {
                let (sw, dw) = (Self::float_width(lo, src), float_bits(lo, inst.ty).unwrap_or(64));
                assert!(dw != 16, "riscv64 backend: f16 needs the Zfh extension");
                let s = lo.reg(src);
                emit(lo, RvOp::FCvtFF, s, &[u64::from(dw), u64::from(sw)]);
            }
            CastOp::FpToSi | CastOp::FpToUi => {
                let fw = Self::float_width(lo, src);
                let iw = lo.types().bit_width(inst.ty).unwrap_or(64);
                assert!(iw <= 64, "riscv64 backend: a float conversion to i{iw}");
                let s = lo.reg(src);
                let signed = u64::from(op == CastOp::FpToSi);
                emit(lo, RvOp::FCvtFI, s, &[signed, if iw > 32 { 64 } else { 32 }, u64::from(fw)]);
            }
            CastOp::SiToFp | CastOp::UiToFp => {
                let fw = float_bits(lo, inst.ty).unwrap_or(64);
                assert!(fw != 16, "riscv64 backend: f16 needs the Zfh extension");
                let signed = op == CastOp::SiToFp;
                let sw = lo.int_width(src);
                assert!(sw <= 64, "riscv64 backend: a float conversion from i{sw}");
                // The 32-bit forms read only the low word; anything else is
                // converted from its 64-bit extension.
                let (s, iw) = if sw == 32 { (lo.reg(src), 32) } else { (self.extend64(lo, src, signed), 64) };
                emit(lo, RvOp::FCvtIF, s, &[u64::from(signed), iw, u64::from(fw)]);
            }
            // The source register's bits above its width are not clean.
            CastOp::ZExt | CastOp::SExt | CastOp::IntToPtr => {
                let s = self.extend64(lo, src, op == CastOp::SExt);
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_v(s)]));
            }
            CastOp::Bitcast => {
                let s = lo.reg(src);
                let w = u64::from(lo.int_width(src));
                match (lo.mf().vreg_class(s), lo.mf().vreg_class(d)) {
                    (RegClass::Gpr, RegClass::Fp) => emit(lo, RvOp::FMvFX, s, &[w]),
                    (RegClass::Fp, RegClass::Gpr) => emit(lo, RvOp::FMvXF, s, &[w]),
                    _ => lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_v(s)])),
                }
            }
            // Truncation / ptr→int: preserve the low bits.
            _ => {
                let s = lo.reg(src);
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_v(s)]));
            }
        }
    }

    /// The bits of the float register `v` (of `width` bits) in a fresh GPR.
    fn to_gpr(&self, lo: &mut Lower<'_, Self>, v: VReg, width: u64) -> VReg {
        let g = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(RvOp::FMvXF.opcode(), vec![def_v(g), use_v(v), imm(width)]));
        g
    }

    /// The branchless blend `d = f ^ ((t ^ f) & -c)` of two GPRs with `c` in
    /// {0, 1}. Every step names at most three registers,
    /// so it stays allocatable when all of them spill (a four-register
    /// `Select` needs one more spill scratch than the three the target
    /// reserves).
    fn blend(&self, lo: &mut Lower<'_, Self>, d: VReg, c: VReg, t: VReg, f: VReg) {
        let zero = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(self.li(zero, Int::ZERO));
        let mask = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(RvOp::Sub.opcode(), vec![def_v(mask), use_v(zero), use_v(c), imm(64)]));
        let diff = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(RvOp::Xor.opcode(), vec![def_v(diff), use_v(t), use_v(f), imm(64)]));
        let pick = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(RvOp::And.opcode(), vec![def_v(pick), use_v(diff), use_v(mask), imm(64)]));
        lo.emit(MachineInst::new(RvOp::Xor.opcode(), vec![def_v(d), use_v(f), use_v(pick), imm(64)]));
    }

    // --- the LP64D calling convention ---------------------------------------

    /// How many of a call's `n` arguments are named: all of them, except for
    /// a direct call to a variadic function (its parameter count; the rest
    /// follow the integer convention). An indirect call carries no signature
    /// here and is treated as non-variadic.
    fn named_count(lo: &Lower<'_, Self>, callee: ValueId, n: usize) -> usize {
        let Some(fidx) = lo.callee_func(callee) else { return n };
        let f = lo.module().function(crate::ir::FuncId::from_index(fidx as usize));
        match lo.types().get(f.sig) {
            Type::Func(ft) if ft.variadic => ft.params.len().min(n),
            _ => n,
        }
    }

    /// `base + off` in a fresh GPR (`base` itself when `off == 0`).
    fn add_off(&self, lo: &mut Lower<'_, Self>, base: VReg, off: u64) -> VReg {
        if off == 0 {
            return base;
        }
        let d = lo.fresh_vreg(RegClass::Gpr);
        if off <= 2047 {
            lo.emit(MachineInst::new(RvOp::Addi.opcode(), vec![def_v(d), use_v(base), imm(off), imm(64)]));
        } else {
            let k = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(self.li(k, Int::from_u64(off)));
            lo.emit(MachineInst::new(RvOp::Add.opcode(), vec![def_v(d), use_v(base), use_v(k), imm(64)]));
        }
        d
    }

    /// Load `size` bytes at `[ptr + off]` into a fresh register of `class`.
    fn load_at(&self, lo: &mut Lower<'_, Self>, class: RegClass, ptr: VReg, off: u64, size: u64) -> VReg {
        let p = self.add_off(lo, ptr, off);
        let d = lo.fresh_vreg(class);
        lo.emit(MachineInst::new(RvOp::Load.opcode(), vec![def_v(d), use_v(p), imm(size)]));
        d
    }

    /// Store the low `size` bytes of `v` at `[ptr + off]`.
    fn store_at(&self, lo: &mut Lower<'_, Self>, ptr: VReg, off: u64, v: VReg, size: u64) {
        let p = self.add_off(lo, ptr, off);
        lo.emit(MachineInst::new(RvOp::Store.opcode(), vec![use_v(p), use_v(v), imm(size)]));
    }

    /// Exactly `size` (1..=8) bytes at `[ptr + off]`, zero-extended into a
    /// fresh GPR: one load for a machine width, else little-endian pieces
    /// combined with shifts (never reading past the object).
    fn load_bytes(&self, lo: &mut Lower<'_, Self>, ptr: VReg, off: u64, size: u64) -> VReg {
        if matches!(size, 1 | 2 | 4 | 8) {
            return self.load_at(lo, RegClass::Gpr, ptr, off, size);
        }
        let mut acc: Option<VReg> = None;
        let mut at = 0u64;
        for piece in [4u64, 2, 1] {
            if size - at >= piece {
                let v = self.load_at(lo, RegClass::Gpr, ptr, off + at, piece);
                let v = if at == 0 {
                    v
                } else {
                    let s = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(RvOp::Slli.opcode(), vec![def_v(s), use_v(v), imm(8 * at), imm(64)]));
                    s
                };
                acc = Some(match acc {
                    None => v,
                    Some(a) => {
                        let o = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(RvOp::Or.opcode(), vec![def_v(o), use_v(a), use_v(v), imm(64)]));
                        o
                    }
                });
                at += piece;
            }
        }
        acc.expect("a nonzero size")
    }

    /// Copy `size` bytes from `[src]` to `[dst]` in 8/4/2/1-byte pieces.
    fn emit_memcpy(&self, lo: &mut Lower<'_, Self>, dst: VReg, src: VReg, size: u64) {
        let mut o = 0u64;
        while o < size {
            let chunk = [8u64, 4, 2, 1].into_iter().find(|&c| size - o >= c).expect("bytes left");
            let t = self.load_at(lo, RegClass::Gpr, src, o, chunk);
            self.store_at(lo, dst, o, t, chunk);
            o += chunk;
        }
    }

    /// A fresh frame slot for an aggregate of type `ty` (rounded up to whole
    /// doublewords, so a register part can be stored in full).
    fn agg_slot(lo: &mut Lower<'_, Self>, ty: crate::ir::types::TypeId) -> StackSlot {
        let size = align_up(lo.byte_size(ty).max(1), 8);
        let align = lo.types().align_of(ty).max(8);
        lo.new_slot(size, align)
    }

    /// `[Def d, Imm off]` of `op` (`LeaSp` / `LeaInArg`) into a fresh GPR.
    fn lea(&self, lo: &mut Lower<'_, Self>, op: RvOp, off: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(op.opcode(), vec![def_v(d), imm(off)]));
        d
    }

    /// The value one part of argument `arg` carries, in a fresh register: the
    /// scalar itself (an `i32`/`i1` extended per the psABI), a float field or
    /// an integer chunk read from the aggregate's storage, or the address of a
    /// fresh copy of it (`ty` is the argument's ABI type).
    fn arg_part(&self, lo: &mut Lower<'_, Self>, arg: ValueId, ty: crate::ir::types::TypeId, part: Part) -> VReg {
        match part {
            Part::Half(k) => wide::parts(self, lo, arg)[usize::from(k)],
            Part::Whole => {
                let r = lo.reg(arg);
                if lo.mf().vreg_class(r) == RegClass::Fp {
                    r
                } else {
                    self.abi_value(lo, arg)
                }
            }
            Part::Chunk { off, size, float: Some(_) } => {
                let p = lo.reg(arg);
                self.load_at(lo, RegClass::Fp, p, off, size)
            }
            Part::Chunk { off, size, float: None } => {
                let p = lo.reg(arg);
                let v = self.load_bytes(lo, p, off, size);
                // A 32-bit integer field travels sign-extended, like an `i32`.
                if size == 4 {
                    let x = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(RvOp::SextW.opcode(), vec![def_v(x), use_v(v)]));
                    x
                } else {
                    v
                }
            }
            Part::Ref => {
                let size = lo.byte_size(ty);
                let slot = Self::agg_slot(lo, ty);
                let dst = lo.fresh_vreg(RegClass::Gpr);
                lo.emit(self.frame_addr(dst, slot));
                let src = lo.reg(arg);
                self.emit_memcpy(lo, dst, src, size);
                dst
            }
        }
    }

    /// The register a part travels in, from a value: a float in an integer
    /// register goes over as its bits.
    fn in_class(&self, lo: &mut Lower<'_, Self>, v: VReg, class: RegClass, width: u64) -> VReg {
        match (lo.mf().vreg_class(v), class) {
            (RegClass::Fp, RegClass::Gpr) => self.to_gpr(lo, v, width),
            (RegClass::Gpr, RegClass::Fp) => {
                let f = lo.fresh_vreg(RegClass::Fp);
                lo.emit(MachineInst::new(RvOp::FMvFX.opcode(), vec![def_v(f), use_v(v), imm(width)]));
                f
            }
            _ => v,
        }
    }

    /// The float width of a part of an argument of type `ty` (64 for an
    /// integer part).
    fn part_width(lo: &Lower<'_, Self>, ty: crate::ir::types::TypeId, part: Part) -> u64 {
        match (part, lo.types().get(ty)) {
            (Part::Chunk { float: Some(w), .. }, _) => u64::from(w),
            (Part::Whole, Type::Float(k)) => u64::from(k.bit_width()),
            _ => 64,
        }
    }

    /// Lower a `call` under the LP64D convention (see the `abi` module): every
    /// part of every argument is materialized, stack parts are stored into
    /// the outgoing area, and the register parts are moved into place as one
    /// consecutive run right before the call (so no competing vreg definition
    /// sits between an argument register's write and the call). The result
    /// comes back in `a0`/`a1`/`fa0`/`fa1` (an aggregate is stored into a
    /// fresh slot whose address is the result), or in caller memory whose
    /// address is passed in `a0`.
    fn lower_call(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];
        // The legalizer's 128-bit multiply is not a real call.
        if wide::is_mul128(self, lo, callee) {
            return wide::lower_mul128(self, lo, inst);
        }
        let named = Self::named_count(lo, callee, args.len());
        let ret_ty = inst.result().map(|r| lo.func().value_type(r));
        let ret_plan = ret_ty.map(|t| abi::ret_locs(lo.types(), t));
        let sret = matches!(ret_plan, Some(None));
        // Classify by the callee's parameter types when the call is direct
        // (an aggregate argument may be written as a plain pointer to its
        // storage), else by the argument values' types.
        let sig_params: Vec<crate::ir::types::TypeId> = match lo.callee_func(callee) {
            Some(fidx) => {
                let f = lo.module().function(crate::ir::FuncId::from_index(fidx as usize));
                match lo.types().get(f.sig) {
                    Type::Func(ft) => ft.params.clone(),
                    _ => Vec::new(),
                }
            }
            None => Vec::new(),
        };
        let arg_ty = |lo: &Lower<'_, Self>, i: usize, a: ValueId| {
            sig_params.get(i).copied().unwrap_or_else(|| lo.func().value_type(a))
        };
        let mut asg = Assigner::args(sret);
        let plans: Vec<Vec<(Part, Loc)>> = args
            .iter()
            .enumerate()
            .map(|(i, &a)| asg.assign(lo.types(), arg_ty(lo, i, a), i < named))
            .collect();
        let arg_tys: Vec<crate::ir::types::TypeId> =
            args.iter().enumerate().map(|(i, &a)| arg_ty(lo, i, a)).collect();

        let mut reg_moves: Vec<(PReg, VReg)> = Vec::new();
        let mut ret_slot = None;
        if sret {
            let slot = Self::agg_slot(lo, ret_ty.expect("a result"));
            ret_slot = Some(slot);
            let p = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(self.frame_addr(p, slot));
            reg_moves.push((gpr(regs::A0), p));
        }
        for ((&arg, plan), &ty) in args.iter().zip(plans).zip(&arg_tys) {
            for (part, loc) in plan {
                let v = self.arg_part(lo, arg, ty, part);
                let width = Self::part_width(lo, ty, part);
                match loc {
                    Loc::Gpr(n) => {
                        let v = self.in_class(lo, v, RegClass::Gpr, width);
                        reg_moves.push((gpr(n), v));
                    }
                    Loc::Fpr(n) => reg_moves.push((fpr(n), v)),
                    Loc::Stack(off) => {
                        let p = self.lea(lo, RvOp::LeaSp, off);
                        let size = if lo.mf().vreg_class(v) == RegClass::Fp { width / 8 } else { 8 };
                        lo.emit(MachineInst::new(RvOp::Store.opcode(), vec![use_v(p), use_v(v), imm(size)]));
                    }
                }
            }
        }
        if asg.stack > 0 {
            lo.reserve_outgoing(align_up(asg.stack, 16));
        }
        let used: Vec<PReg> = reg_moves.iter().map(|&(r, _)| r).collect();
        for (r, v) in reg_moves {
            lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(r), use_v(v)]));
        }

        let ret_reg = self.rf.cc.ret_reg;
        let mut operands = Vec::new();
        match lo.callee_func(callee) {
            Some(fidx) => operands.push(MachineOperand::Func(fidx)),
            None => {
                let cr = lo.reg(callee);
                operands.push(use_v(cr));
            }
        }
        operands.push(def(ret_reg));
        for &cs in &self.rf.caller_saved {
            if cs != ret_reg {
                operands.push(def(cs));
            }
        }
        for &r in &used {
            operands.push(use_p(r));
        }
        lo.emit(MachineInst::new(RvOp::Call.opcode(), operands));

        let Some(d) = inst.result().map(|_| lo.result_reg(inst)) else { return };
        let ret_ty = ret_ty.expect("a result");
        match ret_plan.expect("a result") {
            None => lo.emit(self.frame_addr(d, ret_slot.expect("sret slot"))),
            Some(_) if wide::wide_ty(lo, ret_ty).is_some() => {
                // An `i128` result in a0:a1.
                wide::check_abi_width(self, wide::wide_ty(lo, ret_ty).unwrap_or(0));
                let p = wide::parts(self, lo, inst.result().expect("a result"));
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(p[0]), use_p(gpr(regs::A0))]));
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(p[1]), use_p(gpr(regs::A0 + 1))]));
            }
            Some(parts) if !abi::is_aggregate(lo.types(), ret_ty) => {
                let preg = match parts.first().map(|p| p.1) {
                    Some(Loc::Fpr(n)) => fpr(n),
                    Some(Loc::Gpr(n)) => gpr(n),
                    _ => ret_reg,
                };
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_p(preg)]));
            }
            Some(parts) => {
                // Rescue the result registers first (one consecutive run right
                // after the call), then store them into a fresh slot — unless
                // only field loads read the result: they read these vregs
                // directly (`codegen::aggret`), and there is no slot.
                let fused = lo.fused_call(inst.result().expect("an aggregate result"));
                let saved: Vec<(Part, VReg)> = parts
                    .iter()
                    .enumerate()
                    .map(|(k, &(part, loc))| {
                        let (preg, class) = match loc {
                            Loc::Fpr(n) => (fpr(n), RegClass::Fp),
                            Loc::Gpr(n) => (gpr(n), RegClass::Gpr),
                            Loc::Stack(_) => unreachable!("a register return"),
                        };
                        let v = fused.as_ref().map_or_else(|| lo.fresh_vreg(class), |p| p[k]);
                        lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(v), use_p(preg)]));
                        (part, v)
                    })
                    .collect();
                if fused.is_some() {
                    return;
                }
                let slot = Self::agg_slot(lo, ret_ty);
                lo.emit(self.frame_addr(d, slot));
                for (part, v) in saved {
                    if let Part::Chunk { off, size, .. } = part {
                        self.store_at(lo, d, off, v, part_store_size(size));
                    }
                }
            }
        }
    }

    /// The entry prologue under LP64D: every register part of every parameter
    /// is first copied out of its argument register (one consecutive run, so
    /// no allocation can clobber an incoming register first), then each
    /// parameter is built: a scalar from its register or its stack slot (an
    /// incoming stack argument is addressed above this frame, [`RvOp::LeaInArg`]);
    /// an aggregate in registers is stored into a private home slot whose
    /// address becomes the parameter; one passed by reference is its pointer.
    /// A hidden return-memory pointer is stashed in the aux slot for `ret`.
    fn lower_prologue_lp64d(&self, lo: &mut Lower<'_, Self>) {
        let entry = lo.mf().entry().expect("a function being lowered has an entry block");
        let pvs: Vec<VReg> = lo.mf().block(entry).params.clone();
        let (sig, ret_ty) = match lo.types().get(lo.func().sig) {
            Type::Func(ft) => (ft.params.clone(), ft.ret),
            _ => (Vec::new(), lo.func().sig),
        };
        let sret = abi::is_aggregate(lo.types(), ret_ty) && abi::ret_locs(lo.types(), ret_ty).is_none();
        let mut asg = Assigner::args(sret);
        let plans: Vec<Vec<(Part, Loc)>> = sig.iter().map(|&t| asg.assign(lo.types(), t, true)).collect();

        // 1. Copy every incoming register out.
        let sret_ptr = sret.then(|| {
            let v = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(v), use_p(gpr(regs::A0))]));
            v
        });
        let mut incoming: Vec<Vec<Option<VReg>>> = Vec::with_capacity(plans.len());
        for plan in &plans {
            let mut row = Vec::with_capacity(plan.len());
            for &(_, loc) in plan {
                let (preg, class) = match loc {
                    Loc::Gpr(n) => (gpr(n), RegClass::Gpr),
                    Loc::Fpr(n) => (fpr(n), RegClass::Fp),
                    Loc::Stack(_) => {
                        row.push(None);
                        continue;
                    }
                };
                let v = lo.fresh_vreg(class);
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(v), use_p(preg)]));
                row.push(Some(v));
            }
            incoming.push(row);
        }
        if let Some(v) = sret_ptr {
            let slot = lo.new_slot(8, 8);
            lo.set_aux_slot(slot);
            lo.emit(MachineInst::new(RvOp::StoreFrame.opcode(), vec![use_v(v), MachineOperand::Frame(slot)]));
        }

        // 2. Build each parameter.
        for (i, (&pv, plan)) in pvs.iter().zip(&plans).enumerate() {
            let ty = sig[i];
            // A part's value: its register, or a load from its stack slot.
            let part_value = |this: &Self, lo: &mut Lower<'_, Self>, k: usize, class: RegClass, width: u64| -> VReg {
                match (incoming[i][k], plan[k].1) {
                    (Some(v), _) => this.in_class(lo, v, class, width),
                    (None, Loc::Stack(off)) => {
                        let p = this.lea(lo, RvOp::LeaInArg, off);
                        let size = if class == RegClass::Fp { width / 8 } else { 8 };
                        this.load_at(lo, class, p, 0, size)
                    }
                    _ => unreachable!(),
                }
            };
            if let Some(n) = wide::wide_ty(lo, ty) {
                // An `i128`: its two words from registers or the stack.
                wide::check_abi_width(self, n);
                let entry_block = lo.func().entry().expect("a body");
                let pval = lo.func().block(entry_block).params()[i];
                let p = wide::parts(self, lo, pval);
                for (k, &d) in p.iter().enumerate() {
                    let v = part_value(self, lo, k, RegClass::Gpr, 64);
                    lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_v(v)]));
                }
                continue;
            }
            if !abi::is_aggregate(lo.types(), ty) {
                let class = lo.mf().vreg_class(pv);
                let width = Self::part_width(lo, ty, Part::Whole);
                let v = part_value(self, lo, 0, class, width);
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(pv), use_v(v)]));
                continue;
            }
            match plan.first().map(|p| p.0) {
                Some(Part::Ref) => {
                    let v = part_value(self, lo, 0, RegClass::Gpr, 64);
                    lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(pv), use_v(v)]));
                }
                _ => {
                    let home = Self::agg_slot(lo, ty);
                    lo.emit(self.frame_addr(pv, home));
                    for (k, &(part, _)) in plan.iter().enumerate() {
                        let Part::Chunk { off, size, float } = part else { unreachable!() };
                        let (class, width) = match float {
                            Some(w) => (RegClass::Fp, u64::from(w)),
                            None => (RegClass::Gpr, 64),
                        };
                        let v = part_value(self, lo, k, class, width);
                        let n = if float.is_some() { size } else { part_store_size(size) };
                        self.store_at(lo, pv, off, v, n);
                    }
                }
            }
        }
    }

    /// `ret`: a scalar in `a0` (an `i32`/`i1` extended) or `fa0`; an
    /// aggregate's parts read from its storage into `a0`/`a1`/`fa0`/`fa1`, or
    /// copied through the hidden return-memory pointer. The `ret` itself uses
    /// the result registers, so allocation keeps them intact up to it.
    fn lower_ret(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let ret_ty = match lo.types().get(lo.func().sig) {
            Type::Func(ft) => ft.ret,
            _ => lo.func().sig,
        };
        let mut uses = Vec::new();
        if let Some(&v) = inst.operands().first() {
            if let Some(vals) = lo.fused_ret(v) {
                // A return slot written only right before the `ret`: the stored
                // values go straight into the result registers, an integer
                // field of a 4-byte part sign-extended (as `arg_part` reads one)
                // and a narrower one zero-extended.
                let parts = abi::ret_locs(lo.types(), ret_ty).expect("a fused return is in registers");
                let mut moves = Vec::new();
                for ((part, loc), val) in parts.into_iter().zip(vals) {
                    let Some(val) = val else { continue };
                    let r = match (loc, part) {
                        (Loc::Fpr(n), _) => (fpr(n), lo.reg(val)),
                        (Loc::Gpr(n), Part::Chunk { size, .. }) if size < 8 => (gpr(n), self.extend64(lo, val, size == 4)),
                        (Loc::Gpr(n), _) => (gpr(n), lo.reg(val)),
                        (Loc::Stack(_), _) => unreachable!("a register return"),
                    };
                    moves.push(r);
                }
                for (r, val) in moves {
                    lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(r), use_v(val)]));
                    uses.push(use_p(r));
                }
            } else if abi::is_aggregate(lo.types(), ret_ty) {
                let src = lo.reg(v);
                match abi::ret_locs(lo.types(), ret_ty) {
                    None => {
                        let size = lo.byte_size(ret_ty);
                        let slot = lo.aux_slot().expect("the return-memory pointer saved by the prologue");
                        let dst = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(
                            RvOp::LoadFrame.opcode(),
                            vec![def_v(dst), MachineOperand::Frame(slot)],
                        ));
                        self.emit_memcpy(lo, dst, src, size);
                    }
                    Some(parts) => {
                        let vals: Vec<(Loc, VReg)> = parts
                            .iter()
                            .map(|&(part, loc)| (loc, self.arg_part(lo, v, ret_ty, part)))
                            .collect();
                        for (loc, val) in vals {
                            let r = match loc {
                                Loc::Gpr(n) => gpr(n),
                                Loc::Fpr(n) => fpr(n),
                                Loc::Stack(_) => unreachable!("a register return"),
                            };
                            lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(r), use_v(val)]));
                            uses.push(use_p(r));
                        }
                    }
                }
            } else if let Some(n) = wide::wide_val(lo, v) {
                // An `i128` in a0:a1.
                wide::check_abi_width(self, n);
                let p = wide::parts(self, lo, v);
                for (k, &val) in p.iter().enumerate() {
                    let r = gpr(regs::A0 + k as u16);
                    lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(r), use_v(val)]));
                    uses.push(use_p(r));
                }
            } else {
                let r = self.arg_part(lo, v, ret_ty, Part::Whole);
                let ret = match lo.mf().vreg_class(r) {
                    RegClass::Fp => self.rf.cc.fp_ret_reg,
                    RegClass::Gpr => self.rf.cc.ret_reg,
                };
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(ret), use_v(r)]));
                uses.push(use_p(ret));
            }
        }
        lo.emit(MachineInst::new(RvOp::Ret.opcode(), uses));
    }
}

impl WideIsel for RiscvTarget {
    fn wide_table(&self) -> &WideTable {
        &self.wide
    }

    fn wide_alu(&self, op: BinOp, d: VReg, a: VReg, b: VReg) -> MachineInst {
        let op = match op {
            BinOp::Or => RvOp::Or,
            BinOp::And => RvOp::And,
            BinOp::Xor => RvOp::Xor,
            BinOp::Add => RvOp::Add,
            BinOp::Mul => RvOp::Mul,
            other => unreachable!("not a part operation: {other:?}"),
        };
        MachineInst::new(op.opcode(), vec![def_v(d), use_v(a), use_v(b), imm(64)])
    }

    fn wide_umulh(&self, d: VReg, a: VReg, b: VReg) -> MachineInst {
        MachineInst::new(RvOp::Mulhu.opcode(), vec![def_v(d), use_v(a), use_v(b), imm(64)])
    }

    fn wide_sign(&self, d: VReg, a: VReg) -> MachineInst {
        MachineInst::new(RvOp::Srai.opcode(), vec![def_v(d), use_v(a), imm(63), imm(64)])
    }

    fn wide_cond(&self, lo: &mut Lower<'_, Self>, c: ValueId) -> VReg {
        self.clean_cond(lo, c)
    }

    fn wide_select(&self, lo: &mut Lower<'_, Self>, d: VReg, c: VReg, t: VReg, f: VReg) {
        self.blend(lo, d, c, t, f);
    }

    fn wide_extend(&self, lo: &mut Lower<'_, Self>, v: ValueId, signed: bool) -> VReg {
        self.extend64(lo, v, signed)
    }

    fn wide_load(&self, lo: &mut Lower<'_, Self>, ptr: VReg, off: u64) -> VReg {
        self.load_at(lo, RegClass::Gpr, ptr, off, 8)
    }

    fn wide_store(&self, lo: &mut Lower<'_, Self>, ptr: VReg, off: u64, v: VReg) {
        self.store_at(lo, ptr, off, v, 8);
    }

    fn wide_ret_hi(&self) -> PReg {
        gpr(regs::A0 + 1)
    }

    fn wide_helper(&self, _lo: &Lower<'_, Self>, name: &str) -> Option<u32> {
        self.wide_helpers.iter().find(|h| h.0 == name).map(|h| h.1)
    }

    fn wide_helper_call(&self, lo: &mut Lower<'_, Self>, f: u32, args: &[(PReg, VReg)], rets: &[(PReg, VReg)]) {
        for &(r, v) in args {
            lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(r), use_v(v)]));
        }
        let ret = self.rf.cc.ret_reg;
        let mut operands = vec![MachineOperand::Func(f), def(ret)];
        for &cs in &self.rf.caller_saved {
            if cs != ret {
                operands.push(def(cs));
            }
        }
        for &(r, _) in args {
            operands.push(use_p(r));
        }
        lo.emit(MachineInst::new(RvOp::Call.opcode(), operands));
        for &(r, v) in rets {
            lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(v), use_p(r)]));
        }
    }
}

impl MachineTarget for RiscvTarget {
    fn name(&self) -> &str {
        "riscv64"
    }

    fn reg_classes(&self) -> &[RegClass] {
        &self.rf.classes
    }

    fn allocatable(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Gpr => &self.rf.allocatable,
            RegClass::Fp => &self.rf.allocatable_fp,
        }
    }

    fn scratch(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Gpr => &self.rf.scratch,
            RegClass::Fp => &self.rf.scratch_fp,
        }
    }

    fn caller_saved(&self) -> &[PReg] {
        &self.rf.caller_saved
    }

    fn callee_saved(&self) -> &[PReg] {
        &self.rf.callee_saved
    }

    fn call_conv(&self) -> &CallConv {
        &self.rf.cc
    }

    fn is_terminator(&self, op: Opcode) -> bool {
        matches!(
            RvOp::decode(op),
            RvOp::J | RvOp::BrCond | RvOp::Switch | RvOp::Ret | RvOp::Unreachable
        )
    }

    fn is_move(&self, op: Opcode) -> bool {
        RvOp::decode(op) == RvOp::Mv
    }

    fn emit_move(&self, dst: Reg, src: Reg) -> MachineInst {
        MachineInst::new(RvOp::Mv.opcode(), vec![MachineOperand::Def(dst), MachineOperand::Use(src)])
    }

    fn emit_spill(&self, slot: StackSlot, src: PReg) -> MachineInst {
        MachineInst::new(RvOp::StoreFrame.opcode(), vec![use_p(src), MachineOperand::Frame(slot)])
    }

    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(RvOp::LoadFrame.opcode(), vec![def(dst), MachineOperand::Frame(slot)])
    }
}

impl TargetIsel for RiscvTarget {
    /// The parts [`abi::ret_locs`] returns in `a0`/`a1`/`fa0`/`fa1`.
    fn ret_parts(
        &self,
        types: &crate::ir::TypeContext,
        ty: crate::ir::types::TypeId,
    ) -> Option<Vec<crate::codegen::aggret::RetPart>> {
        abi::ret_locs(types, ty)?
            .into_iter()
            .map(|(part, _)| match part {
                Part::Chunk { off, size, float } => {
                    Some(crate::codegen::aggret::RetPart { off, size, fp: float.is_some() })
                }
                _ => None,
            })
            .collect()
    }

    fn lshr_imm(&self, dst: VReg, src: VReg, bits: u32) -> MachineInst {
        MachineInst::new(RvOp::Srli.opcode(), vec![def_v(dst), use_v(src), imm(u64::from(bits)), imm(64)])
    }

    /// A constant wider than 64 bits keeps its low 64 bits: part 0 of a wide
    /// value ([`crate::codegen::wide`] materializes all of its parts).
    fn li(&self, dst: VReg, value: Int) -> MachineInst {
        let value = if value.to_u64().is_some() || value.to_i64().is_some() { value } else { value.mod_2k(64) };
        MachineInst::new(RvOp::Li.opcode(), vec![def_v(dst), MachineOperand::Imm(value)])
    }

    fn jump(&self, dst: MBlockId) -> MachineInst {
        MachineInst::new(RvOp::J.opcode(), vec![MachineOperand::Label(dst)])
    }

    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(RvOp::FrameAddr.opcode(), vec![def_v(dst), MachineOperand::Frame(slot)])
    }

    fn global_addr(&self, dst: VReg, g: u32) -> MachineInst {
        let got = self.global_got.get(g as usize).copied().unwrap_or(false);
        MachineInst::new(
            RvOp::GlobalAddr.opcode(),
            vec![def_v(dst), MachineOperand::Global(g), imm(u64::from(got))],
        )
    }

    fn float_const(&self, dst: VReg, bits: u64, width: u32) -> MachineInst {
        MachineInst::new(RvOp::FLi.opcode(), vec![def_v(dst), imm(bits), imm(u64::from(width))])
    }

    fn lower_prologue(&self, lo: &mut Lower<'_, Self>) {
        self.lower_prologue_lp64d(lo);
    }

    fn func_addr(&self, dst: VReg, f: u32) -> MachineInst {
        let got = self.func_got.get(f as usize).copied().unwrap_or(false);
        MachineInst::new(
            RvOp::FuncAddr.opcode(),
            vec![def_v(dst), MachineOperand::Func(f), imm(u64::from(got))],
        )
    }

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        if Self::fused_away(lo, inst) {
            return;
        }
        // What the wide-integer legalization left of `i128`s.
        if wide::lower_wide(self, lo, inst) {
            return;
        }
        match &inst.kind {
            InstKind::Bin(op) => self.lower_bin(lo, *op, inst),
            InstKind::ICmp(pred) => {
                let d = lo.result_reg(inst);
                // `slt`/`sltu`/`sub` read all 64 bits: extend a narrow pair
                // first. A signed predicate needs sign-extension; for the others
                // either extension works (sign-extending both sides preserves
                // equality and unsigned order), so an `i32` uses the single
                // `sext.w` and anything else zero-extends.
                let width = lo.int_width(inst.operands()[0]);
                let signed =
                    matches!(pred, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge)
                        || width == 32;
                let a = self.extend64(lo, inst.operands()[0], signed);
                let b = self.extend64(lo, inst.operands()[1], signed);
                lo.emit(MachineInst::new(
                    RvOp::SetCmp.opcode(),
                    vec![
                        def_v(d),
                        use_v(a),
                        use_v(b),
                        imm(u64::from(pred_code(*pred))),
                        imm(u64::from(width)),
                    ],
                ));
            }
            InstKind::Cast(op) => self.lower_cast(lo, *op, inst),
            InstKind::Alloca { elem_ty } => {
                let d = lo.result_reg(inst);
                let size = lo.byte_size(*elem_ty);
                let align = lo.types().align_of(*elem_ty);
                let slot = lo.new_slot(size, align);
                lo.emit(self.frame_addr(d, slot));
            }
            InstKind::DynAlloca { align } => {
                // `sp` moves at run time: the function addresses its frame
                // through `s0` (see `RvOp::DynAlloca` and the encoder).
                assert!(self.frame_pointer, "dyn_alloca in a function compiled without a frame pointer");
                let d = lo.result_reg(inst);
                let n = self.extend64(lo, inst.operands()[0], false);
                lo.emit(MachineInst::new(
                    RvOp::DynAlloca.opcode(),
                    vec![def_v(d), use_v(n), imm(u64::from(*align))],
                ));
            }
            InstKind::Load { ty, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(inst.operands()[0]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    RvOp::Load.opcode(),
                    vec![def_v(d), use_v(ptr), imm(size)],
                ));
            }
            InstKind::Store { ty, .. } => {
                let ptr = lo.reg(inst.operands()[0]);
                let val = lo.reg(inst.operands()[1]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    RvOp::Store.opcode(),
                    vec![use_v(ptr), use_v(val), imm(size)],
                ));
            }
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(inst);
                let base = lo.reg(inst.operands()[0]);
                // The byte offset is signed; a narrow one is sign-extended.
                let off = self.extend64(lo, inst.operands()[1], true);
                lo.emit(MachineInst::new(
                    RvOp::Add.opcode(),
                    vec![def_v(d), use_v(base), use_v(off), imm(64)],
                ));
            }
            InstKind::Select if lo.mf().vreg_class(lo.result_reg(inst)) == RegClass::Fp => {
                // A float select blends the bit patterns in GPRs, branch-free
                // like the integer one.
                let d = lo.result_reg(inst);
                let c = self.clean_cond(lo, inst.operands()[0]);
                let t = lo.reg(inst.operands()[1]);
                let f = lo.reg(inst.operands()[2]);
                let (tx, fx) = (self.to_gpr(lo, t, 64), self.to_gpr(lo, f, 64));
                let g = lo.fresh_vreg(RegClass::Gpr);
                self.blend(lo, g, c, tx, fx);
                lo.emit(MachineInst::new(RvOp::FMvFX.opcode(), vec![def_v(d), use_v(g), imm(64)]));
            }
            InstKind::Select => {
                // Branchless `d = f ^ ((t ^ f) & -c)` with `c` in {0, 1}, so a
                // secret condition is constant-time (§6d); and every step names
                // at most three registers, so it stays allocatable when all of
                // them spill (a four-register `Select` needs one more spill
                // scratch than the three the target reserves).
                let d = lo.result_reg(inst);
                let c = self.clean_cond(lo, inst.operands()[0]);
                let t = lo.reg(inst.operands()[1]);
                let f = lo.reg(inst.operands()[2]);
                self.blend(lo, d, c, t, f);
            }
            InstKind::Freeze | InstKind::Declassify => {
                let d = lo.result_reg(inst);
                let s = lo.reg(inst.operands()[0]);
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_v(s)]));
            }
            InstKind::Call => self.lower_call(lo, inst),
            InstKind::InlineAsm(_) | InstKind::AsmOutput(_) => {
                panic!("riscv64 backend: {}", crate::codegen::INLINE_ASM_UNSUPPORTED)
            }
            InstKind::Syscall => {
                // Linux RISC-V syscall ABI: number in `a7`, arguments in
                // `a0..a5`, result in `a0`. Materialize every operand first, then
                // move them into the fixed registers as one consecutive run right
                // before the `ecall` (as `lower_call` does), so no vreg definition
                // sits between an ABI register's write and its read.
                let cc = &self.rf.cc;
                let vals: Vec<VReg> = inst.operands().iter().map(|&o| lo.reg(o)).collect();
                let mut moves: Vec<(PReg, VReg)> = vec![(super::regs::gpr(17), vals[0])]; // a7
                for (k, &v) in vals[1..].iter().enumerate() {
                    moves.push((cc.arg_regs[k], v));
                }
                let a0 = cc.ret_reg;
                let mut operands = vec![def(a0)];
                for &(r, v) in &moves {
                    lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def(r), use_v(v)]));
                    operands.push(use_p(r));
                }
                lo.emit(MachineInst::new(RvOp::Ecall.opcode(), operands));
                let d = lo.result_reg(inst);
                lo.emit(MachineInst::new(RvOp::Mv.opcode(), vec![def_v(d), use_p(a0)]));
            }
            InstKind::Unary(UnaryOp::FNeg) => {
                // A sign flip by sign injection (`fsgnjn d, s, s`), exact for
                // every value including NaNs (`fneg` flips the sign bit).
                let d = lo.result_reg(inst);
                let w = u64::from(Self::float_width(lo, inst.operands()[0]));
                let s = lo.reg(inst.operands()[0]);
                lo.emit(MachineInst::new(
                    RvOp::FSgnj.opcode(),
                    vec![def_v(d), use_v(s), use_v(s), imm(w), imm(1)],
                ));
            }
            InstKind::FCmp(pred) => {
                let d = lo.result_reg(inst);
                match fcmp_code(*pred) {
                    None => {
                        let v = u64::from(*pred == FloatPred::True);
                        lo.emit(MachineInst::new(RvOp::Li.opcode(), vec![def_v(d), imm(v)]));
                    }
                    Some(code) => {
                        let w = u64::from(Self::float_width(lo, inst.operands()[0]));
                        let a = lo.reg(inst.operands()[0]);
                        let b = lo.reg(inst.operands()[1]);
                        lo.emit(MachineInst::new(
                            RvOp::FCmp.opcode(),
                            vec![def_v(d), use_v(a), use_v(b), imm(u64::from(code)), imm(w)],
                        ));
                    }
                }
            }
            k if k.is_atomic() => self.lower_atomic(lo, inst),
            _ => unreachable!("terminator reached lower_inst: {:?}", inst.kind),
        }
    }

    fn lower_term(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Ret => self.lower_ret(lo, inst),
            InstKind::Br(target) => {
                let args: Vec<_> = inst.operands().to_vec();
                let e = lo.edge_to(*target, &args);
                lo.emit(self.jump(e));
            }
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let cond = self.clean_cond(lo, inst.operands()[0]);
                let ops = inst.operands();
                let tb = 1 + *true_args as usize;
                let fb = tb + *false_args as usize;
                let true_vals: Vec<_> = ops[1..tb].to_vec();
                let false_vals: Vec<_> = ops[tb..fb].to_vec();
                let te = lo.edge_to(*if_true, &true_vals);
                let fe = lo.edge_to(*if_false, &false_vals);
                lo.emit(MachineInst::new(
                    RvOp::BrCond.opcode(),
                    vec![use_v(cond), MachineOperand::Label(te), MachineOperand::Label(fe)],
                ));
            }
            InstKind::Switch(_) if wide::wide_val(lo, inst.operands()[0]).is_some() => {
                panic!("riscv64 backend: a switch on an integer wider than 64 bits is not supported")
            }
            InstKind::Switch(data) => {
                // Cases are compared as 64-bit values: sign-extend the scrutinee
                // and each case value from the scrutinee's width.
                let width = lo.int_width(inst.operands()[0]);
                let cond = self.extend64(lo, inst.operands()[0], true);
                let ops = inst.operands();
                let mut idx = 1usize;
                let dcount = data.default_args as usize;
                let default_vals: Vec<_> = ops[idx..idx + dcount].to_vec();
                idx += dcount;
                let de = lo.edge_to(data.default, &default_vals);
                let mut operands = vec![use_v(cond), MachineOperand::Label(de)];
                let cases = data.cases.clone();
                for case in &cases {
                    let n = case.args as usize;
                    let cvals: Vec<_> = ops[idx..idx + n].to_vec();
                    idx += n;
                    let ce = lo.edge_to(case.target, &cvals);
                    operands.push(MachineOperand::Imm(sext_case(&case.value, width)));
                    operands.push(MachineOperand::Label(ce));
                }
                lo.emit(MachineInst::new(RvOp::Switch.opcode(), operands));
            }
            InstKind::Unreachable => {
                lo.emit(MachineInst::new(RvOp::Unreachable.opcode(), Vec::new()));
            }
            _ => unreachable!("non-terminator reached lower_term: {:?}", inst.kind),
        }
    }
}
