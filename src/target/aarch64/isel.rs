//! The AArch64 (A64) machine opcode set and instruction-selection rules
//! (ROADMAP Phase 7).
//!
//! [`A64Op`] is this target's [`Opcode`] vocabulary: a *post-isel, pre-encoding*
//! MIR whose operands are still MIR [`MachineOperand`]s (registers, immediates,
//! frame slots, labels, symbol references). Unlike x86-64, A64 data-processing
//! instructions are genuinely **three-address** (`add Xd, Xn, Xm` writes a
//! distinct destination), so the isel emits one MIR op per IR op with a clean
//! `[Def d, Use a, Use b]` shape and the encoder never has to synthesize a
//! move-to-destination. A few IR ops still expand to a short A64 idiom at encode
//! time (e.g. an [`A64Op::CmpCset`] becomes `subs xzr,a,b; cset d,cc`, a remainder
//! becomes `sdiv;msub`, a constant becomes a `movz`/`movk` chain, a global becomes
//! `adrp`+`add`).
//!
//! ## Block arguments, calls, returns
//!
//! Block arguments are realized by the framework's edge-move mechanism
//! ([`Lower::edge_to`]). `call` moves arguments into the AAPCS64 argument
//! registers `x0`–`x7`, records `x0` and the caller-saved clobbers as fixed defs,
//! and moves the result out of `x0`; `ret` moves its value into `x0`. The
//! prologue moves incoming parameters out of the argument registers (framework
//! prologue). A64 has a hardware divide, so `sdiv`/`udiv` lower directly and
//! remainder is `sdiv` followed by `msub` — no fixed-register `rax`/`rdx` dance.
//!
//! A function used as a value is [`A64Op::FuncAddr`] (`adrp`+`add`, or a GOT
//! load under position-independent code; the encoder decides per symbol), so
//! function pointers can be stored, passed and called through (`blr`).
//!
//! ## Variadic functions (AAPCS64)
//!
//! As on x86-64, the backend implements the calling convention and a front
//! end lowers `va_start`/`va_arg`/`va_copy` itself as ordinary IR over the
//! `va_list` struct, with two frame-address hooks.
//!
//! **Callers** need nothing special on Linux: AAPCS64 passes anonymous
//! arguments exactly like named ones (`x0`–`x7`, `v0`–`v7`, then 8-byte stack
//! slots), so a variadic call is an ordinary call. **Darwin** arm64
//! ([`AArch64Target::for_os`] with [`TargetOs::Darwin`]) passes every
//! anonymous argument of a call to a (direct) variadic callee on the stack,
//! each in a slot of its size rounded up to 8 bytes (16-aligned for a 16-byte
//! vector; an aggregate over 16 bytes as a pointer to a copy). (Windows on
//! Arm, which passes anonymous floating-point values in general registers,
//! is not implemented: it gets the base rules.)
//!
//! **`va_list`** is a 32-byte, 8-aligned struct:
//!
//! | offset | field       | type    |
//! |--------|-------------|---------|
//! | 0      | `__stack`   | `void*` |
//! | 8      | `__gr_top`  | `void*` |
//! | 16     | `__vr_top`  | `void*` |
//! | 24     | `__gr_offs` | `i32`   |
//! | 28     | `__vr_offs` | `i32`   |
//!
//! **Register save area** — the prologue of a function whose signature is
//! variadic reserves [`VA_SAVE_AREA_SIZE`] (192) bytes, 16-aligned, and saves
//! `x0`–`x7` at offsets `0, 8, .., 56` and the whole 128-bit `q0`–`q7` at
//! [`VA_SAVE_VR_OFFSET`]` + 16 k` (`64, 80, .., 176`). Every register is
//! saved; `va_start`'s offsets skip the named ones.
//!
//! **Frontend hooks** — calls to these externally declared functions are not
//! calls; they materialize addresses, and are valid only in a variadic
//! function:
//!
//! - `ptr @__lf_va_reg_save_area()` → the base of the register save area
//!   (null on Darwin, which has none);
//! - `ptr @__lf_va_overflow_area()` → the first anonymous stack argument,
//!   `x29 + 16 +` the bytes of named stack arguments.
//!
//! The front end's `va_start` then sets, for `gr` named general-register and
//! `vr` named SIMD/FP-register arguments:
//! `__stack = __lf_va_overflow_area()`;
//! `__gr_top = __lf_va_reg_save_area() + 64`;
//! `__vr_top = __lf_va_reg_save_area() + 192`;
//! `__gr_offs = -8 * (8 - gr)`; `__vr_offs = -16 * (8 - vr)`. Its `va_arg`
//! of a general-register type reads `offs = __gr_offs`: if `offs >= 0` the
//! value is on the stack; else `__gr_offs = offs + 8` (or the register count
//! of the type times 8), and if that is still `<= 0` the value is at
//! `__gr_top + offs`, otherwise on the stack. A floating-point (or HFA,
//! vector) type does the same with `__vr_offs`, `__vr_top` and 16 per
//! register. A stack argument is read at `__stack`, which then advances by
//! the argument's size rounded up to 8 (aligned first for a 16-byte type).
//! `va_copy` copies the 32 bytes. On Darwin `va_list` is a plain pointer
//! initialized to `__lf_va_overflow_area()` and advanced 8 bytes per
//! argument.

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::mir::{
    MBlockId, MachineInst, MachineOperand, Opcode, PReg, Reg, RegClass, StackSlot, VReg,
};
use crate::codegen::target::{CallConv, MachineTarget};
use crate::ir::inst::{BinOp, CastOp, FloatPred, InstKind, IntPred, UnaryOp};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ValueDef};
use crate::ir::{FuncId, InstData, Module, ValueId};
use crate::support::StrInterner;
use crate::target::TargetOs;

use puremp::Int;

use super::regs::{self, RegFile};

pub(crate) mod neon;
pub use neon::NeonLegality;

/// The A64 MIR opcode vocabulary. Operand layouts are documented per variant;
/// `Def`/`Use` are register operands, the rest are immediates, frame slots,
/// branch labels, or symbol references. `Imm width` is the operation's integer
/// bit width (selects the 32-bit `W`- vs 64-bit `X`-register form).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum A64Op {
    /// `[Def d, Use s]` — `mov d, s` (`orr d, xzr, s`).
    MovRR = 0,
    /// `[Def d, Imm v]` — load immediate via a `movz`/`movk` chain.
    MovRI = 1,
    /// `[Def d, Use a, Use b, Imm width]` — `add d, a, b`.
    Add = 2,
    /// `[Def d, Use a, Use b, Imm width]` — `sub d, a, b`.
    Sub = 3,
    /// `[Def d, Use a, Use b, Imm width]` — `and d, a, b`.
    And = 4,
    /// `[Def d, Use a, Use b, Imm width]` — `orr d, a, b`.
    Or = 5,
    /// `[Def d, Use a, Use b, Imm width]` — `eor d, a, b`.
    Eor = 6,
    /// `[Def d, Use a, Use b, Imm width]` — `mul d, a, b` (`madd d, a, b, xzr`).
    Mul = 7,
    /// `[Def d, Use a, Imm imm12, Imm width]` — `add d, a, #imm12`.
    AddI = 8,
    /// `[Def d, Use a, Imm imm12, Imm width]` — `sub d, a, #imm12`.
    SubI = 9,
    /// `[Def d, Use a, Use b, Imm width]` — `sdiv d, a, b`.
    Sdiv = 10,
    /// `[Def d, Use a, Use b, Imm width]` — `udiv d, a, b`.
    Udiv = 11,
    /// `[Def d, Use m, Use n, Use a, Imm width]` — `msub d, m, n, a` (`d = a - m*n`).
    Msub = 12,
    /// `[Def d, Use a, Imm count, Imm width]` — `lsl d, a, #count`.
    LslI = 13,
    /// `[Def d, Use a, Imm count, Imm width]` — `lsr d, a, #count`.
    LsrI = 14,
    /// `[Def d, Use a, Imm count, Imm width]` — `asr d, a, #count`.
    AsrI = 15,
    /// `[Def d, Use a, Use b, Imm width]` — `lslv d, a, b`.
    LslV = 16,
    /// `[Def d, Use a, Use b, Imm width]` — `lsrv d, a, b`.
    LsrV = 17,
    /// `[Def d, Use a, Use b, Imm width]` — `asrv d, a, b`.
    AsrV = 18,
    /// `[Def d, Use a, Use b, Imm cc, Imm width]` — `subs xzr,a,b; cset d, cc`.
    CmpCset = 19,
    /// `[Def d, Use cond, Use t, Use f]` — `cmp cond,#0; csel d, t, f, ne`.
    /// This is how `select` lowers, so a `select` on a secret condition runs
    /// without a branch (`docs/ir-design.md` §6d).
    Csel = 20,
    /// `[Def d, Use ptr, Imm size]` — load `size` bytes from `[ptr]`.
    Load = 21,
    /// `[Use ptr, Use val, Imm size]` — store `size` bytes to `[ptr]`.
    Store = 22,
    /// `[Def d, Frame slot]` — `add d, sp, #slot_off`.
    FrameAddr = 23,
    /// `[Def d, Global g]` — `adrp d, g; add d, d, :lo12:g`.
    GlobalAddr = 24,
    /// `[Func f | Use callee, Def x0, Def clobbers.., Use args..]` — call.
    Call = 25,
    /// `[]` — return (value already in x0).
    Ret = 26,
    /// `[Label t]` — unconditional branch.
    B = 27,
    /// `[Use cond, Label t, Label f]` — `cbnz cond, t; b f`.
    BrCond = 28,
    /// `[Use cond, Label default, (Imm val, Label case)...]` — multi-way branch.
    Switch = 29,
    /// `[]` — a trap (`brk #1`).
    Unreachable = 30,
    /// `[Use src, Frame slot]` — spill: `str src, [sp, #slot_off]`.
    StoreFrame = 31,
    /// `[Def dst, Frame slot]` — reload: `ldr dst, [sp, #slot_off]`.
    LoadFrame = 32,
    /// `[]` — `stp x29, x30, [sp, #-16]!` (prologue).
    StpFpLr = 33,
    /// `[]` — `ldp x29, x30, [sp], #16` (epilogue).
    LdpFpLr = 34,
    /// `[]` — `mov x29, sp` (prologue).
    MovFpSp = 35,
    /// `[Imm k]` — `sub sp, sp, #k` (prologue).
    SubSp = 36,
    /// `[Imm k]` — `add sp, sp, #k` (epilogue).
    AddSp = 37,
    /// `[Use r, Imm off]` — `str r, [sp, #off]` (callee-saved save).
    SaveReg = 38,
    /// `[Def r, Imm off]` — `ldr r, [sp, #off]` (callee-saved restore).
    RestoreReg = 39,

    // --- scalar floating-point (FP/SIMD) ----------------------------------
    /// `[Def d, Use a, Use b, Imm width]` — `fadd d, a, b` (`d`/`s` by width).
    FAdd = 40,
    /// `[Def d, Use a, Use b, Imm width]` — `fsub d, a, b`.
    FSub = 41,
    /// `[Def d, Use a, Use b, Imm width]` — `fmul d, a, b`.
    FMul = 42,
    /// `[Def d, Use a, Use b, Imm width]` — `fdiv d, a, b`.
    FDiv = 43,
    /// `[Def d, Use s, Imm width]` — `fneg d, s`.
    FNeg = 44,
    /// `[Def d, Use a, Use b, Imm packed, Imm width]` — `fcmp a,b` then
    /// `cset d,cond` (with an optional second `cset`+`and`/`orr` combine). `packed`
    /// carries `cond | combine<<4 | cond2<<8` (see `fcmp_plan`).
    Fcmp = 45,
    /// `[Def d, Imm bits, Imm width]` — materialize a float constant: the exact
    /// IEEE bit pattern via a scratch gpr (`movz/movk x9; fmov d, x9`).
    LoadFConst = 46,
    /// `[Def d, Use s, Imm dst_w, Imm src_w]` — `fcvt d, s` (f32↔f64).
    Fcvt = 47,
    /// `[Def d, Use s, Imm dst_int_w, Imm src_flt_w]` — `fcvtzs d, s` (float→signed
    /// int, truncating toward zero).
    Fcvtzs = 48,
    /// `[Def d, Use s, Imm dst_int_w, Imm src_flt_w]` — `fcvtzu d, s` (float→unsigned
    /// int, truncating toward zero).
    Fcvtzu = 49,
    /// `[Def d, Use s, Imm dst_flt_w, Imm src_int_w]` — `scvtf d, s` (signed
    /// int→float).
    Scvtf = 50,
    /// `[Def d, Use s, Imm dst_flt_w, Imm src_int_w]` — `ucvtf d, s` (unsigned
    /// int→float).
    Ucvtf = 51,

    // --- aggregate (by-value struct) ABI support --------------------------
    /// `[Def d, Imm off]` — `add d, sp, #off` (unsigned `off`). Materializes an
    /// address in the reserved outgoing-argument area at the bottom of the frame
    /// (`sp` is constant after the prologue), for stack-passed call arguments.
    LeaSpOff = 52,
    /// `[Def d, Imm off]` — `add d, x29, #off` (unsigned `off`). Materializes an
    /// address relative to the frame pointer: used to address an incoming
    /// stack-passed parameter's home (`[x29 + 16 + k]`, above the saved
    /// frame-pointer/link-register pair).
    LeaFpOff = 53,
    /// `[Def x0, Use x8, Use x0.., ]` — the Linux supervisor call `svc #0`. The
    /// syscall number is in `x8` and the arguments in `x0..x5` (moved there by
    /// isel as one consecutive run right before); the kernel returns in `x0` and
    /// preserves every other register, so `x0` is the only def.
    Svc = 54,
    /// `[Def d, Use s, Imm width]` — `sbfx Xd, Xs, #0, #width` (`sbfm`):
    /// sign-extend the low `width` bits (1..=63) of `s` to all 64 bits.
    Sbfx = 55,
    /// `[Def d, Use s, Imm width]` — `ubfx Xd, Xs, #0, #width` (`ubfm`):
    /// zero-extend the low `width` bits (1..=63) of `s` to all 64 bits.
    Ubfx = 56,

    // --- atomics (ARMv8.0: see `lower_atomic` for the mapping) --------------
    /// `[Def d, Use ptr, Imm size]` — `ldar{b,h}` (load-acquire, `size`
    /// bytes, zero-extended): an `acquire`/`seq_cst` atomic load.
    LoadAcq = 57,
    /// `[Use ptr, Use val, Imm size]` — `stlr{b,h}` (store-release): a
    /// `release`/`seq_cst` atomic store.
    StoreRel = 58,
    /// `[Imm crm]` — `dmb <option>` (`0b1011` = `ish`, `0b1001` = `ishld`).
    Dmb = 59,
    /// `[Def d, Use ptr, Use val, Imm size, Imm op, Imm acqrel]` — an atomic
    /// read-modify-write ([`RmwOp::code`](crate::ir::RmwOp::code) `op`),
    /// expanded at encode time into an exclusive-monitor retry loop with an
    /// internal label:
    /// `L: ld{a}xr old, [ptr]; x16 = op(old, val); st{l}xr w17, x16, [ptr];
    /// cbnz w17, L`. `acqrel` bit 0 selects the acquiring load, bit 1 the
    /// releasing store. `x16`/`x17` (IP0/IP1, never allocated) are clobbered.
    AtomicRmw = 60,
    /// `[Def d, Use ptr, Use expected, Use new, Imm size, Imm acqrel]` — a
    /// strong compare-and-exchange, expanded at encode time into
    /// `L: ld{a}xr old, [ptr]; cmp old, expected; b.ne done;
    /// st{l}xr w17, new, [ptr]; cbnz w17, L; done:` (the compare at the access
    /// width). `acqrel` as for [`A64Op::AtomicRmw`]; `x17` is clobbered.
    CmpXchg = 61,

    // --- Advanced SIMD (NEON) 128-bit vectors (see `neon`) -----------------
    /// `[Def d, Use n, Use m, Imm op, Imm esize]` — a three-register NEON op
    /// (`neon::NeonOp` code) on `esize`-bit lanes of the full `q` register.
    NeonOp3 = 62,
    /// `[Def d, Use n, Imm op, Imm esize]` — a two-register NEON op (incl. the
    /// across-lane reductions, whose result is lane 0).
    NeonOp2 = 63,
    /// `[Def d, Use n, Imm op, Imm esize, Imm amount]` — `shl`/`ushr`/`sshr`
    /// by an immediate.
    NeonShift = 64,
    /// `[Def d, Use g, Imm esize]` — `dup Vd.T, Rn`: broadcast a GPR.
    NeonDup = 65,
    /// `[Def d, Use v, Imm esize, Imm lane]` — `dup Vd.T, Vn.T[lane]`.
    NeonDupLane = 66,
    /// `[Def g, Use v, Imm esize, Imm lane]` — `umov Rd, Vn.T[lane]`.
    NeonUmov = 67,
    /// `[Def d, Use v, Use g, Imm esize, Imm lane]` — `mov d, v; ins
    /// Vd.T[lane], Rn`.
    NeonInsGpr = 68,
    /// `[Def d, Use v, Use s, Imm esize, Imm lane]` — `mov d, v; ins
    /// Vd.T[lane], Vs.T[0]`.
    NeonInsElem = 69,
    /// `[Def d, Use ptr]` — `ldr Qd, [ptr]`.
    NeonLoad = 70,
    /// `[Use ptr, Use v]` — `str Qv, [ptr]`.
    NeonStore = 71,
    /// `[Def d, Imm lo, Imm hi]` — a 128-bit constant through `x16`:
    /// `movz/movk x16, lo; fmov Dd, x16; movz/movk x16, hi; ins Vd.d[1], x16`.
    NeonConst = 72,
    /// `[Use cond]` — `cmp cond, #0` (`subs xzr, cond, xzr`), setting the flags
    /// for the [`A64Op::CselNe`] that immediately follows. Spill and reload code
    /// the allocator may place between the two (`ldr`/`str`/`add` without
    /// `s`) never touches the flags.
    CmpZero = 73,
    /// `[Def d, Use t, Use f]` — `csel d, t, f, ne` on the flags of the
    /// preceding [`A64Op::CmpZero`]. Splitting `select` in two keeps every
    /// instruction at three register operands, which the three spill scratches
    /// always cover.
    CselNe = 74,

    // --- function addresses, dynamic stack allocation ----------------------
    /// `[Def d, Func f]` — the address of function `f`: `adrp d, f; add d, d,
    /// :lo12:f`, or under position-independent code, for a function that may
    /// be preempted, a load from its GOT entry (`adrp d, :got:f; ldr d, [d,
    /// :got_lo12:f]`). (A direct call needs no address: it is a `bl`.)
    FuncAddr = 75,
    /// `[Def d, Use n, Imm align]` — dynamic (runtime-sized) stack allocation
    /// (`dyn_alloca`): moves `sp` down by `n` bytes (rounded up to 16, plus
    /// the outgoing-argument area it relocates below the new block and any
    /// alignment slack) and returns an `align`-aligned pointer into the fresh
    /// region in `d`. With stack probes the move is one probed page at a time,
    /// a loop over the size (see `super::encode`). A function containing one
    /// addresses its frame slots from `x29` and restores `sp` from `x29` in its
    /// epilogue.
    DynAlloca = 76,
    /// `[Imm extra]` — `sub sp, x29, #extra` (epilogue of a function with a
    /// [`A64Op::DynAlloca`]): put `sp` back where the prologue left it, below
    /// the fixed frame, before the callee-saved restores.
    SpFromFp = 77,
}

impl A64Op {
    /// The MIR [`Opcode`] id for this opcode.
    #[inline]
    pub fn opcode(self) -> Opcode {
        Opcode(self as u32)
    }

    /// Whether an instruction of this opcode may execute a conditional branch
    /// whose direction depends on a register operand — the constant-time
    /// audit of the lowering (`docs/ir-design.md` §6d): the terminators
    /// `BrCond`/`Switch`, the exclusive-monitor retry loops of `AtomicRmw`
    /// and `CmpXchg` (which also compares the loaded value), and `DynAlloca`,
    /// whose stack-probe loop runs over its size. Everything else
    /// — in particular `CmpZero` + `CselNe` (a `select`), `Csel`, `CmpCset`,
    /// the variable shifts,
    /// `Mul`, `Sbfx`/`Ubfx`, the float conversions, and every NEON op
    /// (`NeonOp3` … `NeonConst`: a vector `select` is an `and`/`bic`/`orr`
    /// blend), and the GOT loads of `GlobalAddr`/`FuncAddr` — is
    /// straight-line code. (The prologue's probe loop counts a constant frame
    /// size.)
    pub fn may_branch_on_data(self, _operands: &[MachineOperand]) -> bool {
        matches!(
            self,
            A64Op::BrCond | A64Op::Switch | A64Op::AtomicRmw | A64Op::CmpXchg | A64Op::DynAlloca
        )
    }

    /// Decode a MIR [`Opcode`] back to an [`A64Op`].
    pub fn decode(op: Opcode) -> A64Op {
        use A64Op::*;
        const TABLE: [A64Op; 78] = [
            MovRR, MovRI, Add, Sub, And, Or, Eor, Mul, AddI, SubI, Sdiv, Udiv, Msub, LslI, LsrI,
            AsrI, LslV, LsrV, AsrV, CmpCset, Csel, Load, Store, FrameAddr, GlobalAddr, Call, Ret, B,
            BrCond, Switch, Unreachable, StoreFrame, LoadFrame, StpFpLr, LdpFpLr, MovFpSp, SubSp,
            AddSp, SaveReg, RestoreReg, FAdd, FSub, FMul, FDiv, FNeg, Fcmp, LoadFConst, Fcvt,
            Fcvtzs, Fcvtzu, Scvtf, Ucvtf, LeaSpOff, LeaFpOff, Svc, Sbfx, Ubfx, LoadAcq, StoreRel,
            Dmb, AtomicRmw, CmpXchg, NeonOp3, NeonOp2, NeonShift, NeonDup, NeonDupLane, NeonUmov,
            NeonInsGpr, NeonInsElem, NeonLoad, NeonStore, NeonConst, CmpZero, CselNe, FuncAddr,
            DynAlloca, SpFromFp,
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

/// Encode an [`IntPred`] as the A64 condition-code nibble used by `b.cond`/`cset`
/// (the *true* condition; `cset` inverts it internally when encoding CSINC).
pub(crate) fn cond_code(p: IntPred) -> u8 {
    match p {
        IntPred::Eq => 0x0,  // EQ
        IntPred::Ne => 0x1,  // NE
        IntPred::Uge => 0x2, // HS (C set)
        IntPred::Ult => 0x3, // LO (C clear)
        IntPred::Ugt => 0x8, // HI
        IntPred::Ule => 0x9, // LS
        IntPred::Sge => 0xA, // GE
        IntPred::Slt => 0xB, // LT
        IntPred::Sgt => 0xC, // GT
        IntPred::Sle => 0xD, // LE
    }
}

/// A64 condition-code nibbles used by `cset`/`csel`.
mod cc {
    pub(super) const EQ: u8 = 0x0;
    pub(super) const NE: u8 = 0x1;
    pub(super) const HS: u8 = 0x2; // C set (unsigned ≥ / "carry set")
    pub(super) const MI: u8 = 0x4; // N set (negative)
    pub(super) const VS: u8 = 0x6; // V set (overflow ⇒ FP unordered)
    pub(super) const VC: u8 = 0x7; // V clear (⇒ FP ordered)
    pub(super) const HI: u8 = 0x8;
    pub(super) const LS: u8 = 0x9;
    pub(super) const GE: u8 = 0xA;
    pub(super) const LT: u8 = 0xB;
    pub(super) const GT: u8 = 0xC;
    pub(super) const LE: u8 = 0xD;
}

/// The `fcmp`+`cset` plan for a floating-point predicate: the primary condition
/// code and, when a single code cannot express the ordered/unordered reading, a
/// second condition combined with `and`/`orr`. Returns `None` for the constant
/// predicates `False`/`True`, which the caller materializes directly.
///
/// After `fcmp`, NZCV encodes: unordered (a NaN operand) ⇒ `N=0,Z=0,C=1,V=1`;
/// `a<b` ⇒ `N=1,Z=0,C=0,V=0`; `a==b` ⇒ `N=0,Z=1,C=1,V=0`; `a>b` ⇒
/// `N=0,Z=0,C=1,V=0`. The condition codes below are chosen so each `FloatPred`
/// matches `ir::semantics`. `one` (ordered ≠) and `ueq` (unordered ∨ =) need two
/// codes: `one = NE ∧ VC`, `ueq = EQ ∨ VS`.
pub(crate) fn fcmp_plan(pred: FloatPred) -> Option<(u8, Combine, u8)> {
    use Combine::{And, None, Or};
    Option::Some(match pred {
        FloatPred::False | FloatPred::True => return Option::None,
        FloatPred::Oeq => (cc::EQ, None, 0),
        FloatPred::Ogt => (cc::GT, None, 0),
        FloatPred::Oge => (cc::GE, None, 0),
        FloatPred::Olt => (cc::MI, None, 0),
        FloatPred::Ole => (cc::LS, None, 0),
        FloatPred::One => (cc::NE, And, cc::VC),
        FloatPred::Ord => (cc::VC, None, 0),
        FloatPred::Ueq => (cc::EQ, Or, cc::VS),
        FloatPred::Ugt => (cc::HI, None, 0),
        FloatPred::Uge => (cc::HS, None, 0),
        FloatPred::Ult => (cc::LT, None, 0),
        FloatPred::Ule => (cc::LE, None, 0),
        FloatPred::Une => (cc::NE, None, 0),
        FloatPred::Uno => (cc::VS, None, 0),
    })
}

/// How the second `cset` of an `fcmp` plan is folded into the result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Combine {
    /// A single `cset` — no second condition.
    None,
    /// `and d, d, d2` with the second `cset`.
    And,
    /// `orr d, d, d2` with the second `cset`.
    Or,
}

impl Combine {
    /// The 4-bit code packed into the [`A64Op::Fcmp`] immediate.
    fn code(self) -> u64 {
        match self {
            Combine::None => 0,
            Combine::And => 1,
            Combine::Or => 2,
        }
    }

    /// Decode a packed 4-bit code back to a [`Combine`].
    pub(crate) fn decode(code: u64) -> Combine {
        match code {
            1 => Combine::And,
            2 => Combine::Or,
            _ => Combine::None,
        }
    }
}

/// A variadic frame-address intrinsic: a call to one of these specially-named
/// external functions materializes an address `va_start` needs instead of
/// calling anything (see the module docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VaHook {
    /// `ptr @__lf_va_reg_save_area()`: the base of the register save area.
    RegSaveArea,
    /// `ptr @__lf_va_overflow_area()`: the first anonymous stack argument.
    OverflowArea,
}

impl VaHook {
    fn from_name(name: &str) -> Option<VaHook> {
        match name {
            "__lf_va_reg_save_area" => Some(VaHook::RegSaveArea),
            "__lf_va_overflow_area" => Some(VaHook::OverflowArea),
            _ => None,
        }
    }
}

/// The size of the AAPCS64 register save area of a variadic function: the
/// general registers `x0`–`x7` (64 bytes) followed by the SIMD/FP registers
/// `q0`–`q7` (128 bytes).
pub const VA_SAVE_AREA_SIZE: u64 = 192;
/// The offset of the SIMD/FP half (`q0`–`q7`) in the register save area; it
/// is also where the general-register half ends (`__gr_top`).
pub const VA_SAVE_VR_OFFSET: u64 = 64;

/// The floating-point "ptype" field (0 = single/`s`, 1 = double/`d`) for a width.
#[inline]
pub(crate) fn ptype_of(width: u32) -> u32 {
    u32::from(width >= 64)
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

// ===========================================================================
// AAPCS64 aggregate classification
// ===========================================================================

/// How an aggregate (a struct/array passed or returned by value) crosses the
/// AAPCS64 ABI. The three cases are mutually exclusive and computed by
/// [`classify_aggregate`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum AbiClass {
    /// A Homogeneous Floating-point Aggregate: `1..=4` elements that are all the
    /// same floating-point type. Passed one element per consecutive SIMD/FP
    /// register (`v0..v7` for arguments; `v0..v3` for a return). `width` is the
    /// element bit width (32 or 64); `count` the flattened element count.
    Hfa { width: u32, count: u32 },
    /// A small aggregate (`≤ 16` bytes, not an HFA): passed as `len` consecutive
    /// general-register eightbytes (`x0..x7` for arguments; `x0`/`x1` for a
    /// return). `len` is 0, 1, or 2.
    Regs(usize),
    /// A large aggregate (`> 16` bytes, not an HFA): passed **by reference** to a
    /// caller-made copy (a pointer in the next general register); returned
    /// through an **indirect result** pointer supplied by the caller in `x8`.
    Reference,
}

/// Whether a type is an aggregate this backend represents, at the codegen level,
/// by a pointer to its in-memory storage.
fn is_aggregate(types: &TypeContext, ty: TypeId) -> bool {
    matches!(types.get(ty), Type::Struct(_) | Type::Array(_, _))
}

/// If every leaf of `ty` is a floating-point value of one identical width, that
/// width and the flattened leaf count; `None` if any leaf is not a float or the
/// float widths differ. (An HFA additionally requires `1..=4` leaves — the
/// caller enforces that bound.)
fn homogeneous_float(types: &TypeContext, ty: TypeId) -> Option<(u32, u32)> {
    fn walk(types: &TypeContext, ty: TypeId, width: &mut Option<u32>, count: &mut u32) -> bool {
        match types.get(ty) {
            Type::Float(k) => {
                let bw = k.bit_width();
                match *width {
                    None => *width = Some(bw),
                    Some(x) if x == bw => {}
                    Some(_) => return false,
                }
                *count += 1;
                true
            }
            Type::Struct(fields) => {
                let n = fields.len();
                (0..n).all(|i| {
                    let (_, fty) = types.field_offset(ty, i as u32);
                    walk(types, fty, width, count)
                })
            }
            Type::Array(elem, len) => {
                let (elem, len) = (*elem, *len);
                (0..len).all(|_| walk(types, elem, width, count))
            }
            _ => false,
        }
    }
    let (mut width, mut count) = (None, 0u32);
    if walk(types, ty, &mut width, &mut count) {
        width.map(|w| (w, count))
    } else {
        None
    }
}

/// Classify an aggregate `ty` under AAPCS64: an HFA (in FP registers), a small
/// aggregate (in general registers), or a large one (by reference / `x8`).
pub(crate) fn classify_aggregate(types: &TypeContext, ty: TypeId) -> AbiClass {
    if let Some((width, count)) = homogeneous_float(types, ty)
        && (1..=4).contains(&count)
    {
        return AbiClass::Hfa { width, count };
    }
    let size = types.size_of(ty);
    if size == 0 {
        return AbiClass::Regs(0);
    }
    if size > 16 {
        return AbiClass::Reference;
    }
    AbiClass::Regs(size.div_ceil(8) as usize)
}

/// Round `v` up to a multiple of `align` (a power of two ≥ 1).
fn align_up_u64(v: u64, align: u64) -> u64 {
    let a = align.max(1);
    v.div_ceil(a) * a
}

/// The AArch64 target: its register file/ABI plus the isel + encoding rules.
#[derive(Debug)]
pub struct AArch64Target {
    rf: RegFile,
    /// Darwin's variant of AAPCS64: anonymous (variadic) arguments go on the
    /// stack and `va_list` is a plain pointer (see the module docs).
    darwin: bool,
}

impl Default for AArch64Target {
    fn default() -> Self {
        Self::new()
    }
}

impl AArch64Target {
    /// Construct the AArch64 target with its fixed register file and AAPCS64 ABI.
    pub fn new() -> AArch64Target {
        AArch64Target { rf: RegFile::new(), darwin: false }
    }

    /// Construct the AArch64 target for `os`: the AAPCS64 base standard, with
    /// Darwin's handling of variadic calls on [`TargetOs::Darwin`] (anonymous
    /// arguments on the stack). Every other OS gets the base (Linux) rules.
    pub fn for_os(os: TargetOs) -> AArch64Target {
        AArch64Target { rf: RegFile::new(), darwin: os == TargetOs::Darwin }
    }

    /// Lower function `func` of `module` to MIR over this target.
    pub fn select(
        &self,
        module: &Module,
        func: crate::ir::FuncId,
    ) -> crate::codegen::mir::MachineFunction {
        crate::codegen::isel::select(self, module, func)
    }

    /// Like [`AArch64Target::select`], but threads the module's symbol interner
    /// so the variadic frame-address intrinsics (`__lf_va_reg_save_area` /
    /// `__lf_va_overflow_area`) are recognized by name at their call sites.
    pub fn select_with_syms(
        &self,
        module: &Module,
        func: crate::ir::FuncId,
        syms: &StrInterner,
    ) -> crate::codegen::mir::MachineFunction {
        crate::codegen::isel::select_with_syms(self, module, func, syms)
    }

    /// The parameter count of a direct callee whose signature is variadic, or
    /// `None` (a fixed-arity callee, or an indirect call, whose signature is
    /// unknown here: the front end calls variadic functions directly).
    fn variadic_callee_params(lo: &Lower<'_, Self>, callee: ValueId) -> Option<usize> {
        let fid = FuncId::from_index(lo.callee_func(callee)? as usize);
        match lo.types().get(lo.module().function(fid).sig) {
            Type::Func(ft) if ft.variadic => Some(ft.params.len()),
            _ => None,
        }
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

    /// Whether `v` is a compare result, whose register holds exactly 0 or 1
    /// (`cset` writes the whole register).
    fn is_compare(lo: &Lower<'_, Self>, v: ValueId) -> bool {
        matches!(lo.func().value(v).def, ValueDef::Inst(id)
            if matches!(lo.func().inst(id).kind, InstKind::ICmp(_) | InstKind::FCmp(_)))
    }

    /// `v` sign- or zero-extended from its width to all 64 bits of a register
    /// (`sbfx`/`ubfx Xd, Xn, #0, #width`). Narrow values live in wider registers
    /// whose upper bits are not kept clean (an `i8` add of 200 + 100 leaves 300
    /// in the register), so anything that reads those bits extends first. A
    /// 64-bit value, and a compare result being zero-extended, are already
    /// clean.
    fn extend64(&self, lo: &mut Lower<'_, Self>, v: ValueId, signed: bool) -> VReg {
        let r = lo.reg(v);
        let width = lo.int_width(v);
        if width >= 64 || (!signed && Self::is_compare(lo, v)) {
            return r;
        }
        let d = lo.fresh_vreg(RegClass::Gpr);
        let op = if signed { A64Op::Sbfx } else { A64Op::Ubfx };
        lo.emit(MachineInst::new(op.opcode(), vec![def_v(d), use_v(r), imm(u64::from(width))]));
        d
    }

    /// An integer operand ready for an operation whose result depends on the
    /// bits above the value's width (right shifts, division, int→float,
    /// compares). A 32-/64-bit value is used as is by the matching `W`/`X` form,
    /// which reads exactly its width; any other width is extended (see
    /// [`Self::extend64`]). Returns the register and the width to operate at
    /// (32 for a value that fits a `W` register, else 64).
    fn extended(&self, lo: &mut Lower<'_, Self>, v: ValueId, signed: bool) -> (VReg, u32) {
        let width = lo.int_width(v);
        if width == 32 || width >= 64 {
            return (lo.reg(v), width.min(64));
        }
        (self.extend64(lo, v, signed), if width < 32 { 32 } else { 64 })
    }

    /// An `i1` branch/select condition as a register holding exactly 0 or 1
    /// (`cbnz`/`cmp` test all 64 bits). A compare's result already is; anything
    /// else (e.g. a `trunc` to `i1`) may carry garbage above bit 0.
    fn clean_cond(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        self.extend64(lo, v, false)
    }

    /// Lower an atomic memory operation or fence (ARMv8.0 memory model, Arm ARM
    /// B2.3: `ldar`/`stlr` are RCsc acquire/release accesses, so together they
    /// also give `seq_cst`; there is no single-instruction rmw before LSE):
    ///
    /// | IR | AArch64 |
    /// |---|---|
    /// | `atomic_load relaxed` | `ldr` |
    /// | `atomic_load acquire`/`seq_cst` | `ldar` |
    /// | `atomic_store relaxed` | `str` |
    /// | `atomic_store release`/`seq_cst` | `stlr` |
    /// | `atomic_rmw` | `ld{a}xr`/`st{l}xr` loop ([`A64Op::AtomicRmw`]) |
    /// | `cmpxchg` | `ld{a}xr`/`cmp`/`st{l}xr` loop ([`A64Op::CmpXchg`]) |
    /// | `fence acquire` | `dmb ishld` |
    /// | `fence release`/`acq_rel`/`seq_cst` | `dmb ish` |
    ///
    /// The exclusive load acquires when the ordering (or, for `cmpxchg`, either
    /// ordering) is acquiring, and the exclusive store releases when it is
    /// releasing.
    fn lower_atomic(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        use crate::ir::inst::AtomicOrdering;
        let ops = inst.operands();
        let acqrel = |acq: bool, rel: bool| imm(u64::from(acq) | (u64::from(rel) << 1));
        match &inst.kind {
            InstKind::AtomicLoad { ty, ordering, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(ops[0]);
                let size = lo.byte_size(*ty);
                let op = if ordering.is_acquire() { A64Op::LoadAcq } else { A64Op::Load };
                lo.emit(MachineInst::new(op.opcode(), vec![def_v(d), use_v(ptr), imm(size)]));
            }
            InstKind::AtomicStore { ty, ordering, .. } => {
                let ptr = lo.reg(ops[0]);
                let val = lo.reg(ops[1]);
                let size = lo.byte_size(*ty);
                let op = if ordering.is_release() { A64Op::StoreRel } else { A64Op::Store };
                lo.emit(MachineInst::new(op.opcode(), vec![use_v(ptr), use_v(val), imm(size)]));
            }
            InstKind::AtomicRmw { op, ty, ordering, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(ops[0]);
                let val = lo.reg(ops[1]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    A64Op::AtomicRmw.opcode(),
                    vec![
                        def_v(d),
                        use_v(ptr),
                        use_v(val),
                        imm(size),
                        imm(u64::from(op.code())),
                        acqrel(ordering.is_acquire(), ordering.is_release()),
                    ],
                ));
            }
            InstKind::CmpXchg { ty, success, failure, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(ops[0]);
                let expected = lo.reg(ops[1]);
                let new = lo.reg(ops[2]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    A64Op::CmpXchg.opcode(),
                    vec![
                        def_v(d),
                        use_v(ptr),
                        use_v(expected),
                        use_v(new),
                        imm(size),
                        acqrel(success.is_acquire() || failure.is_acquire(), success.is_release()),
                    ],
                ));
            }
            InstKind::Fence(ordering) => {
                let crm = if *ordering == AtomicOrdering::Acquire { 0b1001 } else { 0b1011 };
                lo.emit(MachineInst::new(A64Op::Dmb.opcode(), vec![imm(crm)]));
            }
            other => unreachable!("lower_atomic on {other:?}"),
        }
    }

    fn lower_bin(&self, lo: &mut Lower<'_, Self>, op: BinOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let width = lo.int_width(inst.operands()[0]);
        // Simple commutative/register three-address ops.
        let simple = match op {
            BinOp::Add => Some(A64Op::Add),
            BinOp::Sub => Some(A64Op::Sub),
            BinOp::And => Some(A64Op::And),
            BinOp::Or => Some(A64Op::Or),
            BinOp::Xor => Some(A64Op::Eor),
            BinOp::Mul => Some(A64Op::Mul),
            _ => None,
        };
        // Scalar FP arithmetic (F32/F64). The operand width comes from the float
        // type (32 or 64); the encoder picks the `s`/`d` form.
        let fop = match op {
            BinOp::FAdd => Some(A64Op::FAdd),
            BinOp::FSub => Some(A64Op::FSub),
            BinOp::FMul => Some(A64Op::FMul),
            BinOp::FDiv => Some(A64Op::FDiv),
            _ => None,
        };
        if let Some(x) = fop {
            let a = lo.reg(inst.operands()[0]);
            let b = lo.reg(inst.operands()[1]);
            lo.emit(MachineInst::new(
                x.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        if let Some(x) = simple {
            // add/sub take a 12-bit unsigned immediate directly; use it when the
            // RHS is a constant that fits, avoiding a `movz` to materialize it.
            if matches!(op, BinOp::Add | BinOp::Sub)
                && let Some(c) = Self::const_of(lo, inst.operands()[1])
                && let Some(u) = c.to_u64()
                && u <= 0xFFF
            {
                let a = lo.reg(inst.operands()[0]);
                let imm_op = if op == BinOp::Add { A64Op::AddI } else { A64Op::SubI };
                lo.emit(MachineInst::new(
                    imm_op.opcode(),
                    vec![def_v(d), use_v(a), imm(u), imm(u64::from(width))],
                ));
                return;
            }
            let a = lo.reg(inst.operands()[0]);
            let b = lo.reg(inst.operands()[1]);
            lo.emit(MachineInst::new(
                x.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        match op {
            BinOp::Shl => self.lower_shift(lo, A64Op::LslI, A64Op::LslV, d, inst, width),
            BinOp::LShr => self.lower_shift(lo, A64Op::LsrI, A64Op::LsrV, d, inst, width),
            BinOp::AShr => self.lower_shift(lo, A64Op::AsrI, A64Op::AsrV, d, inst, width),
            BinOp::UDiv => self.lower_div(lo, A64Op::Udiv, false, d, inst),
            BinOp::URem => self.lower_div(lo, A64Op::Udiv, true, d, inst),
            BinOp::SDiv => self.lower_div(lo, A64Op::Sdiv, false, d, inst),
            BinOp::SRem => self.lower_div(lo, A64Op::Sdiv, true, d, inst),
            // `frem` has no direct A64 form (it is an `fmod` libcall); a documented
            // follow-up. `d` is an fp register, so keep the MIR well-formed with a
            // zero float constant (never executed in tests).
            _ => lo.emit(MachineInst::new(
                A64Op::LoadFConst.opcode(),
                vec![def_v(d), imm(0), imm(u64::from(width))],
            )),
        }
    }

    /// `fneg`: flip the IEEE sign bit (a sign flip, matching `ir::semantics`), via
    /// the A64 `fneg` instruction.
    fn lower_fneg(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let d = lo.result_reg(inst);
        let s = lo.reg(inst.operands()[0]);
        let width = lo.int_width(inst.operands()[0]);
        lo.emit(MachineInst::new(
            A64Op::FNeg.opcode(),
            vec![def_v(d), use_v(s), imm(u64::from(width))],
        ));
    }

    /// `fcmp`: `fcmp a,b` then `cset` (with the ordered/unordered condition from
    /// [`fcmp_plan`]). The result is an `i1` in a gpr.
    fn lower_fcmp(&self, lo: &mut Lower<'_, Self>, pred: FloatPred, inst: &InstData) {
        let d = lo.result_reg(inst);
        match fcmp_plan(pred) {
            None => {
                // `False`/`True` are constants.
                let v = u64::from(pred == FloatPred::True);
                lo.emit(MachineInst::new(A64Op::MovRI.opcode(), vec![def_v(d), imm(v)]));
            }
            Some((cond, combine, cond2)) => {
                let a = lo.reg(inst.operands()[0]);
                let b = lo.reg(inst.operands()[1]);
                let width = lo.int_width(inst.operands()[0]);
                let packed =
                    u64::from(cond) | (combine.code() << 4) | (u64::from(cond2) << 8);
                lo.emit(MachineInst::new(
                    A64Op::Fcmp.opcode(),
                    vec![def_v(d), use_v(a), use_v(b), imm(packed), imm(u64::from(width))],
                ));
            }
        }
    }

    /// Conversions. Float↔float and int↔float go through the A64 `fcvt`/`fcvtz*`/
    /// `scvtf`/`ucvtf` forms; `zext`/`sext` (and `inttoptr` from a narrower
    /// integer) extend from the source's width; every other cast (truncation,
    /// ptr→int, bitcast within a class) is a low-bits-preserving copy.
    fn lower_cast(&self, lo: &mut Lower<'_, Self>, op: CastOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let src = inst.operands()[0];
        let src_w = lo.int_width(src);
        let dst_w = lo.types().bit_width(inst.ty).unwrap_or(64);
        let emit3 = |lo: &mut Lower<'_, Self>, o: A64Op, s: VReg, a: u32, b: u32| {
            lo.emit(MachineInst::new(
                o.opcode(),
                vec![def_v(d), use_v(s), imm(u64::from(a)), imm(u64::from(b))],
            ));
        };
        match op {
            CastOp::FpTrunc | CastOp::FpExt => {
                let s = lo.reg(src);
                emit3(lo, A64Op::Fcvt, s, dst_w, src_w);
            }
            CastOp::FpToSi => {
                let s = lo.reg(src);
                emit3(lo, A64Op::Fcvtzs, s, dst_w, src_w);
            }
            CastOp::FpToUi => {
                let s = lo.reg(src);
                emit3(lo, A64Op::Fcvtzu, s, dst_w, src_w);
            }
            // The conversion reads the whole `W`/`X` source register: a narrow
            // source is extended first, and converted at the extended width.
            CastOp::SiToFp => {
                let (s, w) = self.extended(lo, src, true);
                emit3(lo, A64Op::Scvtf, s, dst_w, w);
            }
            CastOp::UiToFp => {
                let (s, w) = self.extended(lo, src, false);
                emit3(lo, A64Op::Ucvtf, s, dst_w, w);
            }
            // The source register's bits above its width are not clean.
            CastOp::ZExt | CastOp::SExt | CastOp::IntToPtr => {
                let s = self.extend64(lo, src, op == CastOp::SExt);
                lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
            }
            // Truncation / ptr→int / same-class bitcast: preserve low bits.
            _ => {
                let s = lo.reg(src);
                lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
            }
        }
    }

    fn lower_shift(
        &self,
        lo: &mut Lower<'_, Self>,
        imm_op: A64Op,
        var_op: A64Op,
        d: VReg,
        inst: &InstData,
        width: u32,
    ) {
        // A right shift brings the bits above the width down into the result:
        // extend first and shift at the extended width.
        let (a, op_w) = match imm_op {
            A64Op::LsrI => self.extended(lo, inst.operands()[0], false),
            A64Op::AsrI => self.extended(lo, inst.operands()[0], true),
            _ => (lo.reg(inst.operands()[0]), width),
        };
        if let Some(c) = Self::const_of(lo, inst.operands()[1]) {
            let shmask = if width >= 64 { 63 } else { u64::from(width) - 1 };
            let count = c.to_u64().unwrap_or(0) & shmask;
            lo.emit(MachineInst::new(
                imm_op.opcode(),
                vec![def_v(d), use_v(a), imm(count), imm(u64::from(op_w))],
            ));
        } else {
            // The hardware takes the count modulo the register width (its low 5
            // or 6 bits); a count narrower than that may carry garbage there.
            let b = if lo.int_width(inst.operands()[1]) < 6 {
                self.extend64(lo, inst.operands()[1], false)
            } else {
                lo.reg(inst.operands()[1])
            };
            lo.emit(MachineInst::new(
                var_op.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(op_w))],
            ));
        }
    }

    fn lower_div(
        &self,
        lo: &mut Lower<'_, Self>,
        div_op: A64Op,
        want_rem: bool,
        d: VReg,
        inst: &InstData,
    ) {
        // Division sees every bit of both operands (at the `W`/`X` width): extend
        // them by the division's signedness and divide at the extended width.
        let signed = div_op == A64Op::Sdiv;
        let (a, _) = self.extended(lo, inst.operands()[0], signed);
        let (b, width) = self.extended(lo, inst.operands()[1], signed);
        if !want_rem {
            lo.emit(MachineInst::new(
                div_op.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        // Remainder: q = a / b; d = a - q * b   (msub d, q, b, a).
        let q = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(
            div_op.opcode(),
            vec![def_v(q), use_v(a), use_v(b), imm(u64::from(width))],
        ));
        lo.emit(MachineInst::new(
            A64Op::Msub.opcode(),
            vec![def_v(d), use_v(q), use_v(b), use_v(a), imm(u64::from(width))],
        ));
    }

    /// Materialize `base + off` (a byte displacement) into a fresh GPR, or return
    /// `base` unchanged when `off == 0`. Uses the `add #imm12` form for a small
    /// offset, otherwise a `movz`-materialized register add.
    fn add_off(&self, lo: &mut Lower<'_, Self>, base: VReg, off: u64) -> VReg {
        if off == 0 {
            return base;
        }
        let d = lo.fresh_vreg(RegClass::Gpr);
        if off <= 0xFFF {
            lo.emit(MachineInst::new(
                A64Op::AddI.opcode(),
                vec![def_v(d), use_v(base), imm(off), imm(64)],
            ));
        } else {
            let k = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(MachineInst::new(A64Op::MovRI.opcode(), vec![def_v(k), imm(off)]));
            lo.emit(MachineInst::new(
                A64Op::Add.opcode(),
                vec![def_v(d), use_v(base), use_v(k), imm(64)],
            ));
        }
        d
    }

    /// Emit `add d, sp, #off` into a fresh GPR (addresses the outgoing
    /// stack-argument area at the bottom of the frame).
    fn lea_sp(&self, lo: &mut Lower<'_, Self>, off: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(A64Op::LeaSpOff.opcode(), vec![def_v(d), imm(off)]));
        d
    }

    /// Emit `add d, x29, #off` into a fresh GPR (addresses an incoming
    /// stack-passed parameter, above the saved fp/lr pair).
    fn lea_fp(&self, lo: &mut Lower<'_, Self>, off: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(A64Op::LeaFpOff.opcode(), vec![def_v(d), imm(off)]));
        d
    }

    /// Copy `size` bytes from `[src]` to `[dst]` (both GPR pointer vregs) in
    /// 8/4/2/1-byte chunks via a scratch GPR.
    fn emit_memcpy(&self, lo: &mut Lower<'_, Self>, dst: VReg, src: VReg, size: u64) {
        let mut o = 0u64;
        while o < size {
            let chunk = if size - o >= 8 {
                8
            } else if size - o >= 4 {
                4
            } else if size - o >= 2 {
                2
            } else {
                1
            };
            let sp = self.add_off(lo, src, o);
            let t = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(MachineInst::new(A64Op::Load.opcode(), vec![def_v(t), use_v(sp), imm(chunk)]));
            let dp = self.add_off(lo, dst, o);
            lo.emit(MachineInst::new(A64Op::Store.opcode(), vec![use_v(dp), use_v(t), imm(chunk)]));
            o += chunk;
        }
    }

    /// Copy an aggregate `arg` into the outgoing stack area at `stack_off` and
    /// return the new running stack offset (used when the argument registers of
    /// its bank are exhausted).
    fn arg_on_stack(&self, lo: &mut Lower<'_, Self>, arg: ValueId, ty: TypeId, stack_off: u64) -> u64 {
        let size = lo.byte_size(ty);
        let align = lo.types().align_of(ty).max(8);
        let at = align_up_u64(stack_off, align);
        let src = lo.reg(arg);
        let dst = self.lea_sp(lo, at);
        self.emit_memcpy(lo, dst, src, size);
        at + align_up_u64(size, 8)
    }

    /// Darwin: place the anonymous argument `arg` of a variadic call on the
    /// stack at `stack_off` (each in a slot of its size rounded up to 8 bytes,
    /// 8-aligned, or 16-aligned for a 16-byte vector; an aggregate over 16
    /// bytes as a pointer to a caller copy, as when it is named) and return the
    /// new running stack offset.
    fn anon_arg_on_stack(&self, lo: &mut Lower<'_, Self>, arg: ValueId, ty: TypeId, stack_off: u64) -> u64 {
        if is_aggregate(lo.types(), ty) {
            if classify_aggregate(lo.types(), ty) != AbiClass::Reference {
                return self.arg_on_stack(lo, arg, ty, stack_off);
            }
            let size = lo.byte_size(ty);
            let align = lo.types().align_of(ty).max(8);
            let slot = lo.new_slot(align_up_u64(size.max(8), 8), align);
            let copy = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(self.frame_addr(copy, slot));
            let src = lo.reg(arg);
            self.emit_memcpy(lo, copy, src, size);
            let dp = self.lea_sp(lo, stack_off);
            lo.emit(MachineInst::new(A64Op::Store.opcode(), vec![use_v(dp), use_v(copy), imm(8)]));
            return stack_off + 8;
        }
        let v = lo.reg(arg);
        if lo.types().is_vector(ty) {
            let at = align_up_u64(stack_off, 16);
            let dp = self.lea_sp(lo, at);
            lo.emit(MachineInst::new(A64Op::NeonStore.opcode(), vec![use_v(dp), use_v(v)]));
            return at + 16;
        }
        let size = lo.byte_size(ty);
        let dp = self.lea_sp(lo, stack_off);
        lo.emit(MachineInst::new(A64Op::Store.opcode(), vec![use_v(dp), use_v(v), imm(size)]));
        stack_off + align_up_u64(size, 8)
    }

    /// Lower a `call`, implementing the AAPCS64 ABI for by-value struct arguments
    /// and returns on top of the existing scalar/float handling.
    ///
    /// A struct value is represented, at this codegen level, by a GPR vreg holding
    /// a pointer to the struct's in-memory storage. An HFA argument is loaded
    /// element-by-element into consecutive `v` registers; a small (`≤16`-byte)
    /// aggregate is loaded eightbyte-by-eightbyte into consecutive `x` registers;
    /// a large aggregate is copied into a fresh caller stack slot and passed by a
    /// pointer. An HFA result comes back in `v0..v3`, a small result in `x0`/`x1`,
    /// and a large result through the caller-allocated indirect-result slot whose
    /// address is passed in `x8`.
    fn lower_call(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let cc = &self.rf.cc;
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];

        // The variadic frame-address intrinsics are not calls: each
        // materializes an address `va_start` needs (see the module docs).
        if let Some(hook) = lo.callee_name(callee).and_then(VaHook::from_name) {
            let d = lo.result_reg(inst);
            match hook {
                VaHook::RegSaveArea => match lo.va_reg_save() {
                    Some(slot) => lo.emit(self.frame_addr(d, slot)),
                    // Darwin has no register save area: `va_list` walks the stack.
                    None if self.darwin => lo.emit(self.li(d, Int::ZERO)),
                    None => panic!("__lf_va_reg_save_area called outside a variadic function"),
                },
                VaHook::OverflowArea => {
                    let off = lo
                        .va_overflow_off()
                        .expect("__lf_va_overflow_area called outside a variadic function");
                    lo.emit(MachineInst::new(A64Op::LeaFpOff.opcode(), vec![def_v(d), imm(off)]));
                }
            }
            return;
        }
        // Darwin passes every anonymous argument of a variadic call on the
        // stack; the base standard (Linux) treats them like named ones.
        let anon_from = if self.darwin { Self::variadic_callee_params(lo, callee) } else { None };

        // Return classification.
        let ret_ty = inst.result().map(|r| lo.func().value_type(r));
        let ret_agg = ret_ty.filter(|&t| is_aggregate(lo.types(), t));
        let ret_class = ret_agg.map(|t| classify_aggregate(lo.types(), t));
        let indirect_ret = matches!(ret_class, Some(AbiClass::Reference));

        // The final `arg-reg <- value-vreg` moves, emitted as one consecutive run
        // right before the `call` so no competing vreg definition sits in the gap
        // between an argument register's write and the call (the allocator's
        // fixed-register liveness reasons point-to-point).
        let mut reg_moves: Vec<(PReg, VReg)> = Vec::new();
        let mut int_i = 0usize;
        let mut fp_i = 0usize;
        let mut stack_off = 0u64;

        // A by-reference return: allocate the indirect-result slot and pass its
        // address in `x8` (a register outside the ordinary argument banks).
        let mut ret_slot = None;
        if indirect_ret {
            let t = ret_agg.unwrap();
            let size = align_up_u64(lo.byte_size(t).max(8), 8);
            let align = lo.types().align_of(t).max(8);
            let slot = lo.new_slot(size, align);
            ret_slot = Some(slot);
            let ptr = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(self.frame_addr(ptr, slot));
            reg_moves.push((regs::gpr(regs::X8), ptr));
        }

        for (k, &arg) in args.iter().enumerate() {
            let ty = lo.func().value_type(arg);
            if anon_from.is_some_and(|n| k >= n) {
                stack_off = self.anon_arg_on_stack(lo, arg, ty, stack_off);
                continue;
            }
            if is_aggregate(lo.types(), ty) {
                match classify_aggregate(lo.types(), ty) {
                    AbiClass::Hfa { width, count } => {
                        if fp_i + count as usize <= cc.fp_arg_regs.len() {
                            let ptr = lo.reg(arg);
                            let bytes = u64::from(width / 8);
                            for k in 0..count as usize {
                                let sp = self.add_off(lo, ptr, bytes * k as u64);
                                let d = lo.fresh_vreg(RegClass::Fp);
                                lo.emit(MachineInst::new(
                                    A64Op::Load.opcode(),
                                    vec![def_v(d), use_v(sp), imm(bytes)],
                                ));
                                let areg = cc.fp_arg_regs[fp_i];
                                fp_i += 1;
                                reg_moves.push((areg, d));
                            }
                            continue;
                        }
                        stack_off = self.arg_on_stack(lo, arg, ty, stack_off);
                    }
                    AbiClass::Regs(n) => {
                        if int_i + n <= cc.arg_regs.len() {
                            if n > 0 {
                                let ptr = lo.reg(arg);
                                for k in 0..n {
                                    let sp = self.add_off(lo, ptr, 8 * k as u64);
                                    let d = lo.fresh_vreg(RegClass::Gpr);
                                    lo.emit(MachineInst::new(
                                        A64Op::Load.opcode(),
                                        vec![def_v(d), use_v(sp), imm(8)],
                                    ));
                                    let areg = cc.arg_regs[int_i];
                                    int_i += 1;
                                    reg_moves.push((areg, d));
                                }
                            }
                            continue;
                        }
                        stack_off = self.arg_on_stack(lo, arg, ty, stack_off);
                    }
                    AbiClass::Reference => {
                        // Copy the struct into a fresh caller stack slot and pass a
                        // pointer to that copy (the callee only sees the pointer).
                        let size = lo.byte_size(ty);
                        let align = lo.types().align_of(ty).max(8);
                        let slot = lo.new_slot(align_up_u64(size.max(8), 8), align);
                        let dst = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(self.frame_addr(dst, slot));
                        let src = lo.reg(arg);
                        self.emit_memcpy(lo, dst, src, size);
                        let p = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(self.frame_addr(p, slot));
                        if int_i < cc.arg_regs.len() {
                            let areg = cc.arg_regs[int_i];
                            int_i += 1;
                            reg_moves.push((areg, p));
                        } else {
                            let dp = self.lea_sp(lo, stack_off);
                            lo.emit(MachineInst::new(
                                A64Op::Store.opcode(),
                                vec![use_v(dp), use_v(p), imm(8)],
                            ));
                            stack_off += 8;
                        }
                    }
                }
            } else {
                // Scalar / pointer / float argument.
                let v = lo.reg(arg);
                let is_fp = lo.mf().vreg_class(v) == RegClass::Fp;
                let has_reg =
                    if is_fp { fp_i < cc.fp_arg_regs.len() } else { int_i < cc.arg_regs.len() };
                if has_reg {
                    let areg = if is_fp {
                        let a = cc.fp_arg_regs[fp_i];
                        fp_i += 1;
                        a
                    } else {
                        let a = cc.arg_regs[int_i];
                        int_i += 1;
                        a
                    };
                    reg_moves.push((areg, v));
                } else {
                    // A 16-byte vector takes a 16-aligned 16-byte slot.
                    // A vector (mask included) crosses as its 16-byte register
                    // image, whatever its memory size.
                    let sz = if lo.types().is_vector(ty) { 16 } else { lo.byte_size(ty) };
                    if sz == 16 {
                        stack_off = align_up_u64(stack_off, 16);
                        let dp = self.lea_sp(lo, stack_off);
                        lo.emit(MachineInst::new(A64Op::NeonStore.opcode(), vec![use_v(dp), use_v(v)]));
                    } else {
                        let dp = self.lea_sp(lo, stack_off);
                        lo.emit(MachineInst::new(
                            A64Op::Store.opcode(),
                            vec![use_v(dp), use_v(v), imm(sz)],
                        ));
                    }
                    stack_off += sz.max(8);
                }
            }
        }
        if stack_off > 0 {
            lo.reserve_outgoing(align_up_u64(stack_off, 16));
        }

        let used_arg_regs: Vec<PReg> = reg_moves.iter().map(|&(areg, _)| areg).collect();
        for (areg, r) in reg_moves {
            lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def(areg), use_v(r)]));
        }

        // The primary return register (`x0`/`v0`); struct results reclaim their
        // registers (`x0`/`x1` or `v0..v3`), all covered by the clobber set.
        let ret_is_fp = ret_ty.is_some_and(|t| lo.types().get(t).is_float() || lo.types().is_vector(t));
        let ret_reg = if ret_is_fp { cc.fp_ret_reg } else { cc.ret_reg };

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
        // AAPCS64 preserves only the low 64 bits of v8..v15, so a vector must
        // not live across a call in one: in a function holding vectors, calls
        // clobber them too.
        if Self::holds_vectors(lo) {
            for n in 8u16..=15 {
                operands.push(def(regs::fp(n)));
            }
        }
        for &areg in &used_arg_regs {
            operands.push(use_p(areg));
        }
        lo.emit(MachineInst::new(A64Op::Call.opcode(), operands));

        match &ret_class {
            Some(AbiClass::Reference) => {
                // The result already sits in the caller-allocated indirect slot.
                let d = lo.result_reg(inst);
                lo.emit(self.frame_addr(d, ret_slot.unwrap()));
            }
            Some(AbiClass::Hfa { width, count }) => {
                let (width, count) = (*width, *count);
                let bytes = u64::from(width / 8);
                // Rescue each returned element from `v0..v3` (one consecutive run
                // right after the call), then store them into a fresh result slot.
                let mut saved: Vec<VReg> = Vec::with_capacity(count as usize);
                for k in 0..count as usize {
                    let r = regs::fp(k as u16);
                    let v = lo.fresh_vreg(RegClass::Fp);
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(v), use_p(r)]));
                    saved.push(v);
                }
                let t = ret_agg.unwrap();
                let size = align_up_u64(lo.byte_size(t).max(8), 8);
                let align = lo.types().align_of(t).max(8);
                let slot = lo.new_slot(size, align);
                let d = lo.result_reg(inst);
                lo.emit(self.frame_addr(d, slot));
                for (k, v) in saved.into_iter().enumerate() {
                    let dp = self.add_off(lo, d, bytes * k as u64);
                    lo.emit(MachineInst::new(
                        A64Op::Store.opcode(),
                        vec![use_v(dp), use_v(v), imm(bytes)],
                    ));
                }
            }
            Some(AbiClass::Regs(n)) => {
                let n = *n;
                let mut saved: Vec<VReg> = Vec::with_capacity(n);
                for k in 0..n {
                    let r = if k == 0 { cc.ret_reg } else { regs::gpr(regs::X1) };
                    let v = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(v), use_p(r)]));
                    saved.push(v);
                }
                let t = ret_agg.unwrap();
                let size = align_up_u64(lo.byte_size(t).max(8), 8);
                let align = lo.types().align_of(t).max(8);
                let slot = lo.new_slot(size, align);
                let d = lo.result_reg(inst);
                lo.emit(self.frame_addr(d, slot));
                for (k, v) in saved.into_iter().enumerate() {
                    let dp = self.add_off(lo, d, 8 * k as u64);
                    lo.emit(MachineInst::new(
                        A64Op::Store.opcode(),
                        vec![use_v(dp), use_v(v), imm(8)],
                    ));
                }
            }
            None => {
                if inst.result().is_some() {
                    let d = lo.result_reg(inst);
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_p(ret_reg)]));
                }
            }
        }
    }

    /// Lower the entry prologue with AAPCS64 aggregate / indirect-result /
    /// stack-parameter support. A register-passed struct parameter is stored into
    /// a private home slot (so the body sees it in memory) and its vreg is that
    /// slot's address; a by-reference parameter is already a pointer; a stack-
    /// passed parameter is addressed at `[x29 + 16 + off]`; a by-reference return
    /// stashes the incoming `x8` pointer into an aux slot for the return lowering.
    fn lower_prologue_aarch64(&self, lo: &mut Lower<'_, Self>) {
        let cc = &self.rf.cc;
        let entry = lo.mf().entry().expect("a function being lowered has an entry block");
        let param_vregs: Vec<VReg> = lo.mf().block(entry).params.clone();
        let (sig_params, ret_ty, variadic) = match lo.types().get(lo.func().sig) {
            Type::Func(ft) => (ft.params.clone(), ft.ret, ft.variadic),
            _ => (Vec::new(), lo.func().sig, false),
        };
        let indirect_ret = is_aggregate(lo.types(), ret_ty)
            && matches!(classify_aggregate(lo.types(), ret_ty), AbiClass::Reference);

        // A variadic function saves its argument registers so `va_arg` can
        // walk them (not on Darwin, where every anonymous argument is on the
        // stack).
        if variadic && !self.darwin {
            self.spill_va_regs(lo);
        }

        let mut int_i = 0usize;
        let mut fp_i = 0usize;
        if indirect_ret {
            // The indirect-result pointer arrives in x8; stash it for `ret`.
            let slot = lo.new_slot(8, 8);
            lo.set_aux_slot(slot);
            lo.emit(MachineInst::new(
                A64Op::StoreFrame.opcode(),
                vec![use_p(regs::gpr(regs::X8)), MachineOperand::Frame(slot)],
            ));
        }

        let mut stack_in = 16u64; // first incoming stack arg, above the saved fp/lr
        for (i, &pv) in param_vregs.iter().enumerate() {
            let ty = sig_params[i];
            if is_aggregate(lo.types(), ty) {
                match classify_aggregate(lo.types(), ty) {
                    AbiClass::Hfa { width, count } => {
                        if fp_i + count as usize <= cc.fp_arg_regs.len() {
                            let size = align_up_u64(lo.byte_size(ty).max(8), 8);
                            let align = lo.types().align_of(ty).max(8);
                            let home = lo.new_slot(size, align);
                            lo.emit(self.frame_addr(pv, home));
                            let bytes = u64::from(width / 8);
                            for k in 0..count as usize {
                                let areg = cc.fp_arg_regs[fp_i];
                                fp_i += 1;
                                let v = lo.fresh_vreg(RegClass::Fp);
                                lo.emit(MachineInst::new(
                                    A64Op::MovRR.opcode(),
                                    vec![def_v(v), use_p(areg)],
                                ));
                                let dp = self.add_off(lo, pv, bytes * k as u64);
                                lo.emit(MachineInst::new(
                                    A64Op::Store.opcode(),
                                    vec![use_v(dp), use_v(v), imm(bytes)],
                                ));
                            }
                            continue;
                        }
                        stack_in = self.param_from_stack(lo, pv, ty, stack_in);
                    }
                    AbiClass::Regs(n) => {
                        if int_i + n <= cc.arg_regs.len() {
                            let size = align_up_u64(lo.byte_size(ty).max(8), 8);
                            let align = lo.types().align_of(ty).max(8);
                            let home = lo.new_slot(size, align);
                            lo.emit(self.frame_addr(pv, home));
                            for k in 0..n {
                                let areg = cc.arg_regs[int_i];
                                int_i += 1;
                                let v = lo.fresh_vreg(RegClass::Gpr);
                                lo.emit(MachineInst::new(
                                    A64Op::MovRR.opcode(),
                                    vec![def_v(v), use_p(areg)],
                                ));
                                let dp = self.add_off(lo, pv, 8 * k as u64);
                                lo.emit(MachineInst::new(
                                    A64Op::Store.opcode(),
                                    vec![use_v(dp), use_v(v), imm(8)],
                                ));
                            }
                            continue;
                        }
                        stack_in = self.param_from_stack(lo, pv, ty, stack_in);
                    }
                    AbiClass::Reference => {
                        // A by-reference parameter is just an incoming pointer.
                        if int_i < cc.arg_regs.len() {
                            let areg = cc.arg_regs[int_i];
                            int_i += 1;
                            lo.emit(MachineInst::new(
                                A64Op::MovRR.opcode(),
                                vec![def_v(pv), use_p(areg)],
                            ));
                        } else {
                            let p = self.lea_fp(lo, stack_in);
                            lo.emit(MachineInst::new(
                                A64Op::Load.opcode(),
                                vec![def_v(pv), use_v(p), imm(8)],
                            ));
                            stack_in += 8;
                        }
                    }
                }
            } else {
                let is_fp = lo.mf().vreg_class(pv) == RegClass::Fp;
                let has_reg =
                    if is_fp { fp_i < cc.fp_arg_regs.len() } else { int_i < cc.arg_regs.len() };
                if has_reg {
                    let areg = if is_fp {
                        let a = cc.fp_arg_regs[fp_i];
                        fp_i += 1;
                        a
                    } else {
                        let a = cc.arg_regs[int_i];
                        int_i += 1;
                        a
                    };
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(pv), use_p(areg)]));
                } else {
                    // A vector (mask included) crosses as its 16-byte register
                    // image, whatever its memory size.
                    let sz = if lo.types().is_vector(ty) { 16 } else { lo.byte_size(ty) };
                    if sz == 16 {
                        stack_in = align_up_u64(stack_in, 16);
                        let p = self.lea_fp(lo, stack_in);
                        lo.emit(MachineInst::new(A64Op::NeonLoad.opcode(), vec![def_v(pv), use_v(p)]));
                    } else {
                        let p = self.lea_fp(lo, stack_in);
                        lo.emit(MachineInst::new(
                            A64Op::Load.opcode(),
                            vec![def_v(pv), use_v(p), imm(sz)],
                        ));
                    }
                    stack_in += sz.max(8);
                }
            }
        }

        // `__stack` (Darwin: the whole `va_list`) starts just past the named
        // stack arguments, at `[x29 + stack_in]`.
        if variadic {
            lo.set_va_overflow_off(stack_in);
        }
    }

    /// Save a variadic function's incoming argument registers into the
    /// AAPCS64 register save area ([`VA_SAVE_AREA_SIZE`] bytes, 16-aligned) and
    /// record its slot for `__lf_va_reg_save_area`: `x0`–`x7` at offsets
    /// `0, 8, .., 56`, then the whole 128-bit `q0`–`q7` at `64, 80, .., 176`.
    /// Every register is saved, named or not: `va_start` points `__gr_offs` /
    /// `__vr_offs` past the named ones, so `va_arg` never reads those.
    ///
    /// Each register is first copied into a fresh vreg at the very top of the
    /// prologue, while it still holds the incoming argument; the stores follow.
    fn spill_va_regs(&self, lo: &mut Lower<'_, Self>) {
        let cc = &self.rf.cc;
        let gprs: Vec<VReg> = cc
            .arg_regs
            .iter()
            .map(|&r| {
                let v = lo.fresh_vreg(RegClass::Gpr);
                lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(v), use_p(r)]));
                v
            })
            .collect();
        let vrs: Vec<VReg> = cc
            .fp_arg_regs
            .iter()
            .map(|&r| {
                let v = lo.fresh_vreg(RegClass::Fp);
                lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(v), use_p(r)]));
                v
            })
            .collect();
        let save = lo.new_slot(VA_SAVE_AREA_SIZE, 16);
        lo.set_va_reg_save(save);
        let base = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(self.frame_addr(base, save));
        for (i, v) in gprs.into_iter().enumerate() {
            let dp = self.add_off(lo, base, 8 * i as u64);
            lo.emit(MachineInst::new(A64Op::Store.opcode(), vec![use_v(dp), use_v(v), imm(8)]));
        }
        for (i, v) in vrs.into_iter().enumerate() {
            let dp = self.add_off(lo, base, VA_SAVE_VR_OFFSET + 16 * i as u64);
            lo.emit(MachineInst::new(A64Op::NeonStore.opcode(), vec![use_v(dp), use_v(v)]));
        }
    }

    /// Whether the function being lowered holds any vector value.
    fn holds_vectors(lo: &Lower<'_, Self>) -> bool {
        let f = lo.func();
        (0..f.value_count()).any(|i| lo.types().is_vector(f.value_type(ValueId::from_index(i))))
    }

    /// A stack-passed aggregate parameter: address the caller-placed copy in place
    /// at `[x29 + 16 + off]` and bind the parameter vreg to that address. Returns
    /// the new running incoming-stack offset.
    fn param_from_stack(&self, lo: &mut Lower<'_, Self>, pv: VReg, ty: TypeId, stack_in: u64) -> u64 {
        let size = lo.byte_size(ty);
        let align = lo.types().align_of(ty).max(8);
        let at = align_up_u64(stack_in, align);
        let d = self.lea_fp(lo, at);
        lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(pv), use_v(d)]));
        at + align_up_u64(size, 8)
    }
}

impl MachineTarget for AArch64Target {
    fn name(&self) -> &str {
        "aarch64"
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
            A64Op::decode(op),
            A64Op::B | A64Op::BrCond | A64Op::Switch | A64Op::Ret | A64Op::Unreachable
        )
    }

    fn is_move(&self, op: Opcode) -> bool {
        A64Op::decode(op) == A64Op::MovRR
    }

    fn emit_move(&self, dst: Reg, src: Reg) -> MachineInst {
        MachineInst::new(A64Op::MovRR.opcode(), vec![MachineOperand::Def(dst), MachineOperand::Use(src)])
    }

    fn emit_spill(&self, slot: StackSlot, src: PReg) -> MachineInst {
        MachineInst::new(A64Op::StoreFrame.opcode(), vec![use_p(src), MachineOperand::Frame(slot)])
    }

    /// A `v` register may hold a whole 128-bit vector, so its spill slot is 16
    /// bytes, 16-aligned (spilled with `str q`).
    fn spill_slot(&self, class: RegClass) -> (u64, u64) {
        match class {
            RegClass::Gpr => (8, 8),
            RegClass::Fp => (16, 16),
        }
    }

    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(A64Op::LoadFrame.opcode(), vec![def(dst), MachineOperand::Frame(slot)])
    }
}

impl TargetIsel for AArch64Target {
    fn li(&self, dst: VReg, value: Int) -> MachineInst {
        MachineInst::new(A64Op::MovRI.opcode(), vec![def_v(dst), MachineOperand::Imm(value)])
    }

    fn jump(&self, dst: MBlockId) -> MachineInst {
        MachineInst::new(A64Op::B.opcode(), vec![MachineOperand::Label(dst)])
    }

    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(A64Op::FrameAddr.opcode(), vec![def_v(dst), MachineOperand::Frame(slot)])
    }

    fn global_addr(&self, dst: VReg, g: u32) -> MachineInst {
        MachineInst::new(A64Op::GlobalAddr.opcode(), vec![def_v(dst), MachineOperand::Global(g)])
    }

    fn func_addr(&self, dst: VReg, f: u32) -> MachineInst {
        MachineInst::new(A64Op::FuncAddr.opcode(), vec![def_v(dst), MachineOperand::Func(f)])
    }

    fn float_const(&self, dst: VReg, bits: u64, width: u32) -> MachineInst {
        MachineInst::new(
            A64Op::LoadFConst.opcode(),
            vec![def_v(dst), imm(bits), imm(u64::from(width))],
        )
    }

    fn vector_const(&self, dst: VReg, types: &TypeContext, consts: &crate::ir::ConstPool, c: &Const) -> MachineInst {
        let (lo64, hi64) = crate::codegen::simd128::const_bits(types, consts, c);
        MachineInst::new(A64Op::NeonConst.opcode(), vec![def_v(dst), imm(lo64), imm(hi64)])
    }

    fn lower_prologue(&self, lo: &mut Lower<'_, Self>) {
        self.lower_prologue_aarch64(lo);
    }

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        // NEON vector code (legalized for `NeonLegality` beforehand).
        if self.lower_neon(lo, inst) {
            return;
        }
        match &inst.kind {
            InstKind::Bin(op) => self.lower_bin(lo, *op, inst),
            InstKind::ICmp(pred) => {
                let d = lo.result_reg(inst);
                // `cmp` reads the whole `W`/`X` register: a 32-/64-bit compare
                // uses the matching form, any other width is extended (by the
                // predicate's signedness) first.
                let signed =
                    matches!(pred, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge);
                let (a, _) = self.extended(lo, inst.operands()[0], signed);
                let (b, width) = self.extended(lo, inst.operands()[1], signed);
                lo.emit(MachineInst::new(
                    A64Op::CmpCset.opcode(),
                    vec![
                        def_v(d),
                        use_v(a),
                        use_v(b),
                        imm(u64::from(cond_code(*pred))),
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
            // Runtime-sized stack allocation: the encoder moves `sp` (probing
            // each page) and relocates the outgoing-argument area below the
            // new block (see [`A64Op::DynAlloca`]). The size is read as a whole
            // register, so a narrow one is zero-extended first.
            InstKind::DynAlloca { align } => {
                let d = lo.result_reg(inst);
                let n = self.extend64(lo, inst.operands()[0], false);
                lo.emit(MachineInst::new(
                    A64Op::DynAlloca.opcode(),
                    vec![def_v(d), use_v(n), imm(u64::from(*align))],
                ));
            }
            InstKind::Load { ty, .. } => {
                let d = lo.result_reg(inst);
                let ptr = lo.reg(inst.operands()[0]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    A64Op::Load.opcode(),
                    vec![def_v(d), use_v(ptr), imm(size)],
                ));
            }
            InstKind::Store { ty, .. } => {
                let ptr = lo.reg(inst.operands()[0]);
                let val = lo.reg(inst.operands()[1]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    A64Op::Store.opcode(),
                    vec![use_v(ptr), use_v(val), imm(size)],
                ));
            }
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(inst);
                let base = lo.reg(inst.operands()[0]);
                // The byte offset is signed; a narrow one is sign-extended.
                let off = self.extend64(lo, inst.operands()[1], true);
                lo.emit(MachineInst::new(
                    A64Op::Add.opcode(),
                    vec![def_v(d), use_v(base), use_v(off), imm(64)],
                ));
            }
            // A float select blends `v` registers (a GPR csel cannot).
            InstKind::Select if lo.mf().vreg_class(lo.result_reg(inst)) == RegClass::Fp => {
                let ops = inst.operands().to_vec();
                self.neon_select(lo, inst, &ops);
            }
            InstKind::Select => {
                // Branchless (`csel`), so a secret condition is constant-time (§6d).
                debug_assert!(!A64Op::Csel.may_branch_on_data(&[]));
                let d = lo.result_reg(inst);
                let c = self.clean_cond(lo, inst.operands()[0]);
                let t = lo.reg(inst.operands()[1]);
                let f = lo.reg(inst.operands()[2]);
                lo.emit(MachineInst::new(A64Op::CmpZero.opcode(), vec![use_v(c)]));
                lo.emit(MachineInst::new(A64Op::CselNe.opcode(), vec![def_v(d), use_v(t), use_v(f)]));
            }
            InstKind::Freeze | InstKind::Declassify => {
                let d = lo.result_reg(inst);
                let s = lo.reg(inst.operands()[0]);
                lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
            }
            InstKind::Call => self.lower_call(lo, inst),
            InstKind::InlineAsm(_) | InstKind::AsmOutput(_) => {
                panic!("aarch64 backend: {}", crate::codegen::INLINE_ASM_UNSUPPORTED)
            }
            InstKind::Syscall => {
                // Linux AArch64 syscall ABI: number in `x8`, arguments in
                // `x0..x5`, result in `x0`. Materialize every operand first, then
                // move them into the fixed registers as one consecutive run right
                // before the `svc` (as `lower_call` does), so no vreg definition
                // sits between an ABI register's write and its read.
                let vals: Vec<VReg> = inst.operands().iter().map(|&o| lo.reg(o)).collect();
                let mut moves: Vec<(PReg, VReg)> = vec![(regs::gpr(8), vals[0])];
                for (k, &v) in vals[1..].iter().enumerate() {
                    moves.push((regs::gpr(k as u16), v));
                }
                let x0 = regs::gpr(0);
                let mut operands = vec![def(x0)];
                for &(r, v) in &moves {
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def(r), use_v(v)]));
                    operands.push(use_p(r));
                }
                lo.emit(MachineInst::new(A64Op::Svc.opcode(), operands));
                let d = lo.result_reg(inst);
                lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def_v(d), use_p(x0)]));
            }
            InstKind::Unary(UnaryOp::FNeg) => self.lower_fneg(lo, inst),
            InstKind::FCmp(pred) => self.lower_fcmp(lo, *pred, inst),
            k if k.is_atomic() => self.lower_atomic(lo, inst),
            _ => unreachable!("terminator reached lower_inst: {:?}", inst.kind),
        }
    }

    fn lower_term(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Ret => {
                let cc = &self.rf.cc;
                let ret_ty = match lo.types().get(lo.func().sig) {
                    Type::Func(ft) => ft.ret,
                    _ => lo.func().sig,
                };
                if is_aggregate(lo.types(), ret_ty) {
                    // The return operand is a pointer to the struct's storage.
                    let src = lo.reg(inst.operands()[0]);
                    match classify_aggregate(lo.types(), ret_ty) {
                        AbiClass::Reference => {
                            // Copy the struct through the indirect-result pointer
                            // (stashed to the aux slot by the prologue).
                            let size = lo.byte_size(ret_ty);
                            let slot = lo
                                .aux_slot()
                                .expect("indirect result pointer saved by the prologue");
                            let dst = lo.fresh_vreg(RegClass::Gpr);
                            lo.emit(MachineInst::new(
                                A64Op::LoadFrame.opcode(),
                                vec![def_v(dst), MachineOperand::Frame(slot)],
                            ));
                            self.emit_memcpy(lo, dst, src, size);
                        }
                        AbiClass::Hfa { width, count } => {
                            // Place each element in `v0..v3`. Compute all source
                            // pointers first so the loads into the return
                            // registers are consecutive.
                            let bytes = u64::from(width / 8);
                            let ptrs: Vec<VReg> = (0..count as usize)
                                .map(|k| self.add_off(lo, src, bytes * k as u64))
                                .collect();
                            for (k, &p) in ptrs.iter().enumerate() {
                                let r = regs::fp(k as u16);
                                lo.emit(MachineInst::new(
                                    A64Op::Load.opcode(),
                                    vec![def(r), use_v(p), imm(bytes)],
                                ));
                            }
                        }
                        AbiClass::Regs(n) => {
                            // Place each eightbyte in `x0`/`x1`.
                            let ptrs: Vec<VReg> = (0..n)
                                .map(|k| self.add_off(lo, src, 8 * k as u64))
                                .collect();
                            for (k, &p) in ptrs.iter().enumerate() {
                                let r = if k == 0 { cc.ret_reg } else { regs::gpr(regs::X1) };
                                lo.emit(MachineInst::new(
                                    A64Op::Load.opcode(),
                                    vec![def(r), use_v(p), imm(8)],
                                ));
                            }
                        }
                    }
                } else if let Some(&v) = inst.operands().first() {
                    let r = lo.reg(v);
                    // A float return goes in v0, an integer/pointer return in x0.
                    let ret = match lo.mf().vreg_class(r) {
                        RegClass::Fp => cc.fp_ret_reg,
                        RegClass::Gpr => cc.ret_reg,
                    };
                    lo.emit(MachineInst::new(A64Op::MovRR.opcode(), vec![def(ret), use_v(r)]));
                }
                lo.emit(MachineInst::new(A64Op::Ret.opcode(), Vec::new()));
            }
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
                    A64Op::BrCond.opcode(),
                    vec![use_v(cond), MachineOperand::Label(te), MachineOperand::Label(fe)],
                ));
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
                lo.emit(MachineInst::new(A64Op::Switch.opcode(), operands));
            }
            InstKind::Unreachable => {
                lo.emit(MachineInst::new(A64Op::Unreachable.opcode(), Vec::new()));
            }
            _ => unreachable!("non-terminator reached lower_term: {:?}", inst.kind),
        }
    }
}
