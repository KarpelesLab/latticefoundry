//! The x86-64 machine opcode set and instruction-selection rules (ROADMAP
//! Phase 7).
//!
//! [`X86Op`] is this target's [`Opcode`] vocabulary. It is a *post-isel,
//! pre-encoding* MIR: operands are still MIR [`MachineOperand`]s (registers,
//! immediates, frame slots, labels, symbol references), and one MIR op may expand
//! to several machine instructions at encode time (e.g. an [`X86Op::Add`] becomes
//! `mov dst, a; add dst, b`). Keeping the two-address fixup and the
//! flags/setcc/movzx idioms as single MIR ops is what lets the register allocator
//! see clean three-address def/use information while the encoder still emits legal
//! two-address x86.
//!
//! ## Two-address handling
//!
//! x86 arithmetic is destructive (`add dst, src` computes `dst += src`). We model
//! the IR's three-address `d = a op b` as a single MIR op with operands
//! `[Def d, Use a, Use b]` and let the encoder materialize the copy:
//! since `d`, `a`, and `b` all interfere at the op (d is defined there, a and b
//! are read there), the allocator always gives them distinct physical registers,
//! so the encoder can emit `mov d, a; op d, b` unconditionally (with a `neg`
//! fixup for the one non-commutative case, `sub`, should `d` ever coincide with
//! `b`). Spilled operands are reloaded into scratch by the allocator *before* the
//! op, so the expansion still sees final physical registers.
//!
//! ## Block arguments, calls, returns
//!
//! Block arguments are realized by the framework's edge-move mechanism
//! ([`Lower::edge_to`]). `call` moves arguments into the SysV argument registers,
//! records the return register and the caller-saved clobbers as fixed defs, and
//! moves the result out of `rax`; `ret` moves its value into `rax`. The prologue
//! moves incoming parameters out of the argument registers (framework prologue).
//!
//! ## Variadic functions (System V AMD64)
//!
//! This backend implements the callee-side and caller-side ABI a C frontend
//! needs to build `<stdarg.h>`; the frontend lowers `va_arg` itself as ordinary
//! IR over the `va_list` struct, using the two frame-address intrinsics below.
//!
//! **`va_list`** is a 24-byte struct (one element of the array typedef):
//!
//! | offset | field              | type    |
//! |--------|--------------------|---------|
//! | 0      | `gp_offset`        | `u32`   |
//! | 4      | `fp_offset`        | `u32`   |
//! | 8      | `overflow_arg_area`| `void*` |
//! | 16     | `reg_save_area`    | `void*` |
//!
//! **Register save area** — the prologue of any function whose signature is
//! variadic reserves a 176-byte area and spills the incoming argument registers
//! into it (`X86_64Target::spill_va_regs`): the 6 integer regs
//! `rdi, rsi, rdx, rcx, r8, r9` at offsets `0, 8, .., 40`, then `xmm0..7` at
//! offsets `48, 64, .., 160` (16-byte stride). The SSE saves are unconditional
//! (no `test al,al` guard): reading `xmm0..7` is always safe.
//!
//! **`al` at variadic call sites** — when calling a function whose (direct)
//! signature is variadic, the caller sets `al` to the number of SSE argument
//! registers used (`mov eax, N`), per the psABI hidden-argument rule.
//!
//! **Frontend hooks** — two specially-named external functions are recognized by
//! name and lowered to frame addresses (never emitted as real calls); they are
//! only valid inside a variadic function:
//!
//! - `ptr @__lf_va_reg_save_area()` → the address of the register save area
//!   (`va_list.reg_save_area`);
//! - `ptr @__lf_va_overflow_area()` → the address of the first incoming stack
//!   argument, past every named stack argument (`va_list.overflow_arg_area`).
//!
//! The frontend's `va_start` then fills the `va_list` as:
//! `gp_offset = 8 * (named integer/pointer args in GPRs)` (≤ 48);
//! `fp_offset = 48 + 16 * (named float/double args in XMMs)` (≤ 176);
//! `reg_save_area = __lf_va_reg_save_area()`;
//! `overflow_arg_area = __lf_va_overflow_area()`. Its `va_arg` reads an integer
//! eightbyte from `reg_save_area + gp_offset` (then `gp_offset += 8`) while
//! `gp_offset < 48`, an SSE one from `reg_save_area + fp_offset`
//! (then `fp_offset += 16`) while `fp_offset < 176`, and otherwise from
//! `overflow_arg_area` (then `overflow_arg_area += 8`).
//!
//! ## Microsoft x64 (Win64)
//!
//! [`X86_64Target::with_call_conv`]`(`[`CallConvKind::Win64`]`)` — what
//! [`CodegenOptions::os`](crate::codegen::CodegenOptions::os) =
//! [`TargetOs::Windows`] selects — lowers calls, the prologue and `ret` with
//! the Microsoft x64 convention instead: arguments in `rcx, rdx, r8, r9` /
//! `xmm0..3` by position, 32 bytes of shadow space at every call, aggregates of
//! 1/2/4/8 bytes by value and others by reference, `rsi`/`rdi`/`xmm6..15`
//! callee-saved, and a pointer-walk `va_list`. The rules are in the `win64`
//! submodule's documentation.

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::linkage::TlsModel;
use crate::codegen::options::RelocModel;
use crate::codegen::mir::{
    MBlockId, MachineInst, MachineOperand, Opcode, PReg, Reg, RegClass, StackSlot, VReg,
};
use crate::codegen::target::{CallConv, MachineTarget};
use crate::ir::inst::{BinOp, CastOp, FloatPred, InstKind, IntPred, UnaryOp};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ValueDef};
use crate::ir::{FuncId, InstData, Module, ValueId};
use crate::support::{DetHashMap, DetHashSet, StrInterner};

use puremp::Int;
use std::cell::RefCell;

use super::regs::{self, RegFile};
use crate::target::{CallConvKind, TargetOs, Triple};

mod inline_asm;
mod wide;
mod win64;

pub use inline_asm::check_inline_asm;
pub(crate) use inline_asm::encode_inline_asm;

pub use wide::MUL128_PSEUDO;
pub(crate) use wide::float_helper;

pub(crate) mod vector;
pub use vector::Sse2Legality;

/// The x86-64 MIR opcode vocabulary. Operand layouts are documented per variant;
/// `Def`/`Use` are register operands, the rest are immediates, frame slots,
/// branch labels, or symbol references.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum X86Op {
    /// `[Def d, Use s]` — `mov d, s` (64-bit copy).
    MovRR = 0,
    /// `[Def d, Imm v]` — load immediate `d = v`.
    MovRI = 1,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a + b`.
    Add = 2,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a - b`.
    Sub = 3,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a & b`.
    And = 4,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a | b`.
    Or = 5,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a ^ b`.
    Xor = 6,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a * b` (imul).
    Imul = 7,
    /// `[Def d, Use a, Imm count, Imm width]` — `d = a << count`.
    ShlI = 8,
    /// `[Def d, Use a, Imm count, Imm width]` — `d = a >>u count`.
    ShrI = 9,
    /// `[Def d, Use a, Imm count, Imm width]` — `d = a >>s count`.
    SarI = 10,
    /// `[Def d, Use a, Use rcx, Imm width]` — `d = a << (cl)`.
    ShlCl = 11,
    /// `[Def d, Use a, Use rcx, Imm width]` — `d = a >>u (cl)`.
    ShrCl = 12,
    /// `[Def d, Use a, Use rcx, Imm width]` — `d = a >>s (cl)`.
    SarCl = 13,
    /// `[Def rdx, Use rax, Imm width]` — sign-extend rax into rdx (cqo/cdq).
    Cqo = 14,
    /// `[Def rdx]` — zero rdx (`xor edx, edx`).
    ZeroRdx = 15,
    /// `[Def rax, Def rdx, Use rax, Use rdx, Use b, Imm width]` — signed divide.
    Idiv = 16,
    /// `[Def rax, Def rdx, Use rax, Use rdx, Use b, Imm width]` — unsigned divide.
    Div = 17,
    /// `[Def d, Use a, Use b, Imm cc, Imm width]` — `cmp a,b; setcc d; movzx d`.
    SetccCmp = 18,
    /// `[Use r]` — `test r, r` (sets flags for a following cmov).
    Test = 19,
    /// `[Def d, Use d, Use t]` — `cmovne d, t` (move if ZF=0). This is how
    /// `select` lowers, so a `select` on a secret condition runs without a
    /// branch (constant-time discipline, `docs/ir-design.md` §6d).
    Cmovne = 20,
    /// `[Def d, Use ptr, Imm size]` — load `size` bytes from `[ptr]`.
    Load = 21,
    /// `[Use ptr, Use val, Imm size]` — store `size` bytes to `[ptr]`.
    Store = 22,
    /// `[Def d, Frame slot]` — `lea d, [rbp + slot]`.
    LeaFrame = 23,
    /// `[Def d, Global g]` — `lea d, [rip + global]` (RIP-relative).
    GlobalAddr = 24,
    /// `[Func f | Use callee, Def rax, Def clobbers.., Use args..]` — call.
    Call = 25,
    /// `[]` — return (value already in rax).
    Ret = 26,
    /// `[Label t]` — unconditional jump.
    Jmp = 27,
    /// `[Use cond, Label t, Label f]` — `test cond,cond; jne t; jmp f`.
    BrCond = 28,
    /// `[Use cond, Label default, (Imm val, Label case)...]` — multi-way branch.
    Switch = 29,
    /// `[]` — an unreachable trap (`ud2`).
    Unreachable = 30,
    /// `[Use r]` — `push r`.
    Push = 31,
    /// `[Def r]` — `pop r`.
    Pop = 32,
    /// `[]` — `mov rbp, rsp`.
    MovRbpRsp = 33,
    /// `[Imm k]` — `sub rsp, k`.
    SubRsp = 34,
    /// `[Imm k]` — `lea rsp, [rbp - k]`.
    LeaRspRbp = 35,
    /// `[Use src, Frame slot]` — spill: `mov`/`movsd` `[rbp+slot], src` (the
    /// mnemonic follows `src`'s register class).
    StoreFrame = 36,
    /// `[Def dst, Frame slot]` — reload: `mov`/`movsd` `dst, [rbp+slot]`.
    LoadFrame = 37,

    // --- SSE scalar floating-point ----------------------------------------
    /// `[Def d, Use a, Use b, Imm width]` — `d = a + b` (`addsd`/`addss`).
    FAdd = 38,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a - b` (`subsd`/`subss`).
    FSub = 39,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a * b` (`mulsd`/`mulss`).
    FMul = 40,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a / b` (`divsd`/`divss`).
    FDiv = 41,
    /// `[Def d, Use a, Use b, Imm width]` — `d = a ^ b` (`xorpd`/`xorps`); used
    /// with a sign-bit mask to implement `fneg`.
    FXor = 42,
    /// `[Def d, Imm bits, Imm width]` — materialize a float constant: load the
    /// exact bit pattern via a scratch gpr (`mov r11, bits; movq/movd d, r11`).
    LoadFConst = 43,
    /// `[Def d, Use a, Use b, Imm packed, Imm width]` — `ucomis` + `setcc`
    /// (+ parity fixup) computing the `i1` result of an `fcmp` into gpr `d`.
    FCmpSet = 44,
    /// `[Def d, Use s]` — `cvtsd2ss d, s` (F64→F32, `fptrunc`).
    Cvtsd2ss = 45,
    /// `[Def d, Use s]` — `cvtss2sd d, s` (F32→F64, `fpext`).
    Cvtss2sd = 46,
    /// `[Def d, Use s, Imm srcfloatwidth, Imm flags]` — `cvttsd2si`/`cvttss2si`
    /// (float→int, truncating). `flags` bit0 = 64-bit gpr destination.
    CvtF2si = 47,
    /// `[Def d, Use s, Imm dstfloatwidth, Imm flags]` — `cvtsi2sd`/`cvtsi2ss`
    /// (int→float). `flags` bit0 = 64-bit gpr source, bit1 = zero-extend a
    /// 32-bit source first (unsigned ≤32), bit2 = full unsigned-64 fix-up (the
    /// `shr`/`and`/`or` halve-and-round sequence plus a doubling `addsd`).
    CvtSi2f = 48,
    /// `[Def d, Func f]` — `lea d, [rip + func]` (RIP-relative): materialize a
    /// function's runtime address into a GPR for use as a function pointer, with
    /// a `Pc32` relocation to the function symbol. A *direct* call still lowers
    /// through [`X86Op::Call`] with a `Func` operand and a `Plt32` relocation;
    /// only a function used as a plain *value* reaches here.
    FuncAddr = 49,
    /// `[Def d, Use s, Imm src_w, Imm dst_w]` — sign-extend `s` (`src_w` bits)
    /// into `d` (`movsx`/`movsxd`). Implements the IR `sext`.
    Movsx = 50,
    /// `[Def d, Use s, Imm src_w, Imm dst_w]` — zero-extend `s` (`src_w` bits)
    /// into `d` (`movzx`, or a 32-bit `mov`). Implements the IR `zext`.
    Movzx = 51,

    // --- aggregate (by-value struct) ABI support --------------------------
    /// `[Def d, Imm off]` — `lea d, [rbp + off]` (signed `off`). Materializes an
    /// address relative to the frame pointer: used to address an incoming
    /// stack-passed parameter's home (`[rbp + 16 + k]`, above the return address).
    LeaRbpOff = 52,
    /// `[Def d, Imm off]` — `lea d, [rsp + off]` (unsigned `off`). Materializes an
    /// address in the reserved outgoing-argument area at the bottom of the frame
    /// (`rsp` is constant after the prologue), for stack-passed call arguments.
    LeaRspOff = 53,
    /// `[Def d, Use n, Imm align]` — dynamic (runtime-sized) stack allocation
    /// (`dyn_alloca`). Carves `n` bytes off the stack by moving `rsp` and returns
    /// an `align`-aligned pointer into the fresh region in `d`. The moving `rsp`
    /// coexists with the fixed rsp-relative outgoing-argument area: the encoder
    /// keeps the outgoing area travelling at the bottom of the frame
    /// (`[rsp, rsp + outgoing)`) by relocating it below the carved block, so
    /// stack-argument `[rsp + k]` addressing stays valid; the rbp-relative
    /// epilogue reclaims the whole dynamic region on return. See `encode_inst`
    /// (`super::encode`) for the exact expansion.
    DynAlloca = 54,
    /// `[Def rax, Def rcx, Def r11, Use rax, Use arg-regs...]` — the Linux
    /// `syscall` instruction (`0F 05`). The number is in `rax` and the arguments
    /// in `rdi, rsi, rdx, r10, r8, r9` (moved there by isel right before, as a
    /// consecutive run); the kernel returns in `rax` and the instruction itself
    /// clobbers `rcx` (return `rip`) and `r11` (saved `rflags`). Every other
    /// register is preserved by the kernel, so only those three are defs.
    Syscall = 55,

    // --- atomics (x86-64 is TSO: see `lower_atomic` for the mapping) --------
    /// `[]` — `mfence` (`0F AE F0`): a full barrier, for `fence seq_cst`.
    Mfence = 56,
    /// `[Def d, Use ptr, Use val, Imm size]` — `mov d, val; xchg [ptr], d`
    /// (`86`/`87 /r`, implicitly locked): atomically swap, leaving the old value
    /// in `d`. Also the `seq_cst` store (its result is simply unused).
    Xchg = 57,
    /// `[Def d, Use ptr, Use val, Imm size, Imm negate]` — `mov d, val;
    /// [neg d;] lock xadd [ptr], d` (`F0 0F C0`/`C1 /r`): atomic fetch-add
    /// (fetch-sub with `negate` = 1), the old value left in `d`.
    LockXadd = 58,
    /// `[Def rax, Use rax, Use ptr, Use new, Imm size]` — `lock cmpxchg [ptr],
    /// new` (`F0 0F B0`/`B1 /r`): compares `rax` (the expected value, moved
    /// there by isel right before) with `[ptr]`, stores `new` if equal, and
    /// leaves the old memory value in `rax` either way.
    LockCmpxchg = 59,
    /// `[Def rax, Def tmp, Use ptr, Use val, Imm size, Imm op]` — an atomic
    /// read-modify-write with no single x86 instruction returning the old value
    /// (`and`/`or`/`xor`/`nand`/`max`/`min`/`umax`/`umin`), expanded at encode
    /// time into a `lock cmpxchg` loop with an internal label:
    /// `mov rax, [ptr]; L: mov tmp, rax; tmp = op(tmp, val); lock cmpxchg [ptr],
    /// tmp; jne L`. The old value ends in `rax`; `tmp` is a clobbered scratch.
    RmwLoop = 60,

    // --- Microsoft x64 frame support ------------------------------------------
    /// `[Use xmm, Imm off]` — `movups [rbp + off], xmm` (signed `off`): save
    /// all 128 bits of a callee-saved `xmm6..xmm15` in the prologue (Win64).
    SaveXmm = 61,
    /// `[Def xmm, Imm off]` — `movups xmm, [rbp + off]`: restore it in the
    /// epilogue.
    RestoreXmm = 62,

    // --- SSE2 128-bit vectors (see `vector` for the lowering) ---------------
    /// `[Def d, Use a, Use b, Imm enc]` — a two-address packed SSE op
    /// `d = a OP b` (`movaps d, a; OP d, b`). `enc` packs the mandatory prefix
    /// (bits 0..8, `0` for none), the opcode after `0F` (bits 8..16), whether
    /// the op is commutative (bit 16), whether an `imm8` follows (bit 17), and
    /// that immediate (bits 24..32). (See `vector::VEnc`.)
    VOp = 63,
    /// `[Def d, Use s, Imm enc]` — a non-destructive packed op `OP d, s`
    /// (`pshufd`/`pshuflw`/`cvtdq2ps`/`cvttps2dq`), `enc` as for [`X86Op::VOp`].
    VUnary = 64,
    /// `[Def d, Use a, Imm enc]` — an immediate packed shift
    /// (`movaps d, a; psll/psrl/psra d, imm8`): `enc` bits 0..8 the opcode
    /// (`71`/`72`/`73`), 8..16 the `ModRM.reg` extension, 16..24 the count.
    VShiftI = 65,
    /// `[Def d, Use ptr, Imm aligned]` — 16-byte load (`movdqa` if `aligned`,
    /// else `movdqu`).
    VLoad = 66,
    /// `[Use ptr, Use v, Imm aligned]` — 16-byte store (`movdqa`/`movdqu`).
    VStore = 67,
    /// `[Def d, Imm lo, Imm hi]` — materialize a 128-bit constant (`pxor` for
    /// zero, `pcmpeqd` for all-ones, else two `movq` through `r11` joined by
    /// `punpcklqdq` with a free scratch xmm).
    LoadVConst = 68,
    /// `[Def x, Use g, Imm is64]` — `movd`/`movq xmm, r` (zero-extends).
    MovGprToX = 69,
    /// `[Def g, Use x, Imm is64]` — `movd`/`movq r, xmm` (the low lane).
    MovXToGpr = 70,
    /// `[Def d, Use v, Use g, Imm idx]` — `movaps d, v; pinsrw d, g, idx`.
    Pinsrw = 71,
    /// `[Def g, Use v, Imm idx]` — `pextrw g, v, idx` (zero-extended word).
    Pextrw = 72,

    // --- thread-local storage (see `lower_global_addr`) ----------------------
    /// `[Def d, Global g, Imm initial_exec]` — the address of thread-local
    /// global `g` in the current thread: `mov d, fs:[0]` (the TCB's self
    /// pointer, i.e. the thread pointer), then `lea d, [d + g@tpoff]`
    /// (local-exec, `R_X86_64_TPOFF32`) or, with `initial_exec`,
    /// `add d, [rip + g@gottpoff]` (`R_X86_64_GOTTPOFF`).
    TlsAddr = 73,
    /// `[Global g, Def rax, Def clobbers..]` — general-dynamic TLS: the
    /// canonical `data16 lea rdi, [rip + g@tlsgd]; data16 data16 rex.w call
    /// __tls_get_addr@plt` sequence (`R_X86_64_TLSGD` + `R_X86_64_PLT32`, in
    /// the exact form a linker may relax), leaving the address in `rax`. A
    /// call: every caller-saved register is a def.
    TlsGd = 74,

    // --- 128-bit integer support (see the `wide` submodule) ------------------
    /// `[Def rax, Def rdx, Use rax, Use b]` — `mul b` (`REX.W F7 /4`): the
    /// full unsigned 128-bit product of `rax` and `b` in `rdx:rax`.
    MulWide = 75,
    /// `[Use lo, Use hi, Label default, (Imm case_lo, Imm case_hi, Label case)...]`
    /// — a multi-way branch on a 128-bit scrutinee: per case, `cmp lo, case_lo;
    /// jne next; cmp hi, case_hi; je case`, then `jmp default`.
    Switch128 = 76,

    // --- inline assembly (see the `inline_asm` submodule) --------------------
    /// `[Imm id, Imm flags, operands.., Def clobbers..]` — a GCC-style inline
    /// asm statement: the function's [`crate::codegen::mir::MachineAsm`] `id`
    /// says which operand holds each asm operand (registers as defs/uses,
    /// memory operands as a use of their pointer, immediates, symbols); every
    /// clobbered register is a def. `flags` bit 0: the template is not empty
    /// (it may branch on its operands). The encoder instantiates the template
    /// and splices in the bytes rsasm assembles from it.
    InlineAsm = 77,

    // --- fused compares and immediate forms --------------------------------
    /// `[Use a, Use b, Imm cc, Imm width, Label t, Label f]` — an `icmp` whose
    /// only use is the `cond_br` that follows it: `cmp a, b; jcc t; jmp f`
    /// (the `cmp` at `width` bits, as [`X86Op::SetccCmp`]). Block layout drops
    /// the jump to whichever target falls through, inverting `cc` when it is
    /// `t`.
    CmpBr = 78,
    /// `[Use a, Imm v, Imm cc, Imm width, Label t, Label f]` — [`X86Op::CmpBr`]
    /// against a constant: `cmp a, v` (`test a, a` when `v` is 0). `v` is the
    /// constant sign-extended from `width` bits; at 64 bits it fits an imm32.
    CmpBrI = 79,
    /// `[Def d, Use a, Imm v, Imm cc, Imm width]` — [`X86Op::SetccCmp`]
    /// against a constant, `v` as for [`X86Op::CmpBrI`].
    SetccCmpI = 80,
    /// `[Def d, Use a, Imm v, Imm ext, Imm width]` — `d = a OP v` for the ALU
    /// group-1 operation `ext` (`0` add, `1` or, `4` and, `5` sub, `6` xor)
    /// with a sign-extended `imm8`/`imm32`: `mov d, a; OP d, v`, or
    /// `lea d, [a + v]` for an add/sub into another register.
    AluRI = 81,
    /// `[Def d, Use a, Imm v, Imm width]` — `imul d, a, v` (`6B`/`69`).
    ImulRI = 82,

    // --- frame ----------------------------------------------------------------
    /// `[]` — `leave` (`mov rsp, rbp; pop rbp`): the epilogue of a frame
    /// with no callee-saved registers.
    Leave = 83,
    /// `[Imm k]` — `add rsp, k`: the epilogue of a frame without a frame
    /// pointer.
    AddRsp = 84,
}

impl X86Op {
    /// The MIR [`Opcode`] id for this opcode.
    #[inline]
    pub fn opcode(self) -> Opcode {
        Opcode(self as u32)
    }

    /// Whether an instruction of this opcode with `operands` may execute a
    /// conditional jump whose direction depends on a register operand — the
    /// constant-time audit of the lowering (`docs/ir-design.md` §6d). The
    /// terminators `BrCond`/`Switch` do; so do the encode-time expansions of
    /// the `u64`↔float conversions (a sign / range test), the `lock cmpxchg`
    /// retry loop, and `dyn_alloca`'s stack-probe loop over its size. Every
    /// other opcode — in particular `Cmovne` (a `select`), `SetccCmp`,
    /// shifts by `cl`, `Imul`, `Movzx`/`Movsx` — is straight-line code. (The
    /// prologue's probe loop counts a constant frame size.) The IR-level
    /// constant-time verifier rejects secret operands for every opcode that
    /// lowers to one of these, which the constant-time isel tests check.
    pub fn may_branch_on_data(self, operands: &[MachineOperand]) -> bool {
        let flags = |i: usize| match operands.get(i) {
            Some(MachineOperand::Imm(v)) => v.to_u64().unwrap_or(0),
            _ => 0,
        };
        match self {
            X86Op::BrCond
            | X86Op::CmpBr
            | X86Op::CmpBrI
            | X86Op::Switch
            | X86Op::Switch128
            | X86Op::RmwLoop
            | X86Op::DynAlloca => true,
            // An opaque template may branch on anything, unless it is empty.
            X86Op::InlineAsm => flags(1) & 1 != 0,
            X86Op::CvtSi2f => flags(3) & 0b100 != 0,
            X86Op::CvtF2si => flags(3) & 0b10 != 0,
            // The SSE2 vector ops are straight-line data movement and
            // arithmetic (blends, not branches, implement `select`).
            X86Op::VOp
            | X86Op::VUnary
            | X86Op::VShiftI
            | X86Op::VLoad
            | X86Op::VStore
            | X86Op::LoadVConst
            | X86Op::MovGprToX
            | X86Op::MovXToGpr
            | X86Op::Pinsrw
            | X86Op::Pextrw => false,
            _ => false,
        }
    }

    /// Decode a MIR [`Opcode`] back to an [`X86Op`].
    pub fn decode(op: Opcode) -> X86Op {
        use X86Op::*;
        const TABLE: [X86Op; 85] = [
            MovRR, MovRI, Add, Sub, And, Or, Xor, Imul, ShlI, ShrI, SarI, ShlCl, ShrCl, SarCl, Cqo,
            ZeroRdx, Idiv, Div, SetccCmp, Test, Cmovne, Load, Store, LeaFrame, GlobalAddr, Call,
            Ret, Jmp, BrCond, Switch, Unreachable, Push, Pop, MovRbpRsp, SubRsp, LeaRspRbp,
            StoreFrame, LoadFrame, FAdd, FSub, FMul, FDiv, FXor, LoadFConst, FCmpSet, Cvtsd2ss,
            Cvtss2sd, CvtF2si, CvtSi2f, FuncAddr, Movsx, Movzx, LeaRbpOff, LeaRspOff, DynAlloca,
            Syscall, Mfence, Xchg, LockXadd, LockCmpxchg, RmwLoop, SaveXmm, RestoreXmm, VOp,
            VUnary, VShiftI, VLoad, VStore, LoadVConst, MovGprToX, MovXToGpr, Pinsrw, Pextrw,
            TlsAddr, TlsGd, MulWide, Switch128, InlineAsm, CmpBr, CmpBrI, SetccCmpI, AluRI, ImulRI,
            Leave, AddRsp,
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

/// The predicate that compares the same way with its operands swapped
/// (`a < b` is `b > a`).
fn swap_pred(p: IntPred) -> IntPred {
    match p {
        IntPred::Eq | IntPred::Ne => p,
        IntPred::Ugt => IntPred::Ult,
        IntPred::Uge => IntPred::Ule,
        IntPred::Ult => IntPred::Ugt,
        IntPred::Ule => IntPred::Uge,
        IntPred::Sgt => IntPred::Slt,
        IntPred::Sge => IntPred::Sle,
        IntPred::Slt => IntPred::Sgt,
        IntPred::Sle => IntPred::Sge,
    }
}

/// `value` as the immediate of an instruction operating at `width` bits: its
/// low `width` bits sign-extended, when that fits the sign-extended imm32 of
/// the x86 ALU forms (always, for a width up to 32).
pub(crate) fn imm_at(value: &Int, width: u32) -> Option<i64> {
    let bits = value.mod_2k(64).to_u64().unwrap_or(0);
    let shift = 64 - width.clamp(1, 64);
    let v = ((bits << shift) as i64) >> shift;
    i32::try_from(v).is_ok().then_some(v)
}

/// Encode an [`IntPred`] as the x86 condition-code nibble used by `setcc`/`jcc`.
pub(crate) fn cc_code(p: IntPred) -> u8 {
    match p {
        IntPred::Eq => 0x4,  // E
        IntPred::Ne => 0x5,  // NE
        IntPred::Ugt => 0x7, // A  (above)
        IntPred::Uge => 0x3, // AE (not below)
        IntPred::Ult => 0x2, // B  (below)
        IntPred::Ule => 0x6, // BE
        IntPred::Sgt => 0xF, // G
        IntPred::Sge => 0xD, // GE
        IntPred::Slt => 0xC, // L
        IntPred::Sle => 0xE, // LE
    }
}

/// The `ucomis`+`setcc` plan for an `fcmp` predicate, packed into one immediate
/// for [`X86Op::FCmpSet`]. Returns `None` for the constant predicates
/// `False`/`True`, which the caller materializes directly.
///
/// After `ucomisd a, b` the flags are: `ZF=PF=CF=1` when unordered (a NaN
/// operand), else `CF` = "below" (a<b), `ZF` = "equal", `PF` = 0. Packing:
/// bits 0..8 = the primary `setcc` code; bit 8 = swap operands (`ucomis b, a`,
/// realizing the `<`/`<=` orderings from `>`/`>=`); bits 9..11 = the combine
/// step (`0` none, `1` AND `setnp`, `2` OR `setp`) that separates the ordered
/// and unordered readings of equality.
pub(crate) fn fcmp_pack(pred: FloatPred) -> Option<u64> {
    // (primary cc, swap, combine): combine 0 = none, 1 = AND setnp, 2 = OR setp.
    let (cc, swap, combine): (u8, bool, u8) = match pred {
        FloatPred::False | FloatPred::True => return None,
        FloatPred::Oeq => (0x4, false, 1), // sete AND setnp
        FloatPred::One => (0x5, false, 0), // setne
        FloatPred::Ogt => (0x7, false, 0), // seta
        FloatPred::Oge => (0x3, false, 0), // setae
        FloatPred::Olt => (0x7, true, 0),  // ucomis b,a; seta
        FloatPred::Ole => (0x3, true, 0),  // ucomis b,a; setae
        FloatPred::Ueq => (0x4, false, 0), // sete
        FloatPred::Une => (0x5, false, 2), // setne OR setp
        FloatPred::Ugt => (0x2, true, 0),  // ucomis b,a; setb
        FloatPred::Uge => (0x6, true, 0),  // ucomis b,a; setbe
        FloatPred::Ult => (0x2, false, 0), // setb
        FloatPred::Ule => (0x6, false, 0), // setbe
        FloatPred::Ord => (0xB, false, 0), // setnp
        FloatPred::Uno => (0xA, false, 0), // setp
    };
    Some(u64::from(cc) | (u64::from(swap) << 8) | (u64::from(combine) << 9))
}

// ===========================================================================
// System V AMD64 aggregate classification
// ===========================================================================

/// The class of one "eightbyte" of an aggregate under the System V AMD64 ABI:
/// [`Eightbyte::Integer`] (holds integer/pointer data — passed in a GPR) or
/// [`Eightbyte::Sse`] (holds only `float`/`double` data — passed in an XMM).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Eightbyte {
    /// An INTEGER-class eightbyte (GPR: `rdi..r9` / `rax`, `rdx`).
    Integer,
    /// An SSE-class eightbyte (XMM: `xmm0..7` / `xmm0`, `xmm1`).
    Sse,
}

/// How an aggregate crosses the ABI: in registers (one class per eightbyte,
/// `len` 1 or 2) or entirely in memory (on the stack for arguments; via a hidden
/// `sret` pointer for a return).
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum AbiClass {
    /// Passed/returned in registers, one entry per eightbyte.
    Regs(Vec<Eightbyte>),
    /// Passed on the stack / returned through a hidden pointer.
    Memory,
}

/// Merge two eightbyte contributions per the SysV rule: any INTEGER wins,
/// otherwise SSE; an absent (`None`) contribution defers to the other.
fn merge_class(acc: Option<Eightbyte>, cls: Eightbyte) -> Option<Eightbyte> {
    match (acc, cls) {
        (None, c) => Some(c),
        (Some(Eightbyte::Integer), _) | (_, Eightbyte::Integer) => Some(Eightbyte::Integer),
        _ => Some(Eightbyte::Sse),
    }
}

/// Fold the leaf fields of `ty` (placed at absolute byte `offset`) into the
/// per-eightbyte class accumulators `ebs`.
fn classify_into(types: &TypeContext, ty: TypeId, offset: u64, ebs: &mut [Option<Eightbyte>]) {
    let cls = match types.get(ty) {
        Type::Int(_) | Type::Ptr | Type::PtrIn(_) | Type::Func(_) => Some(Eightbyte::Integer),
        // A vector field is SSE data (the psABI's SSE+SSEUP pair for a 16-byte
        // `__m128` is approximated as two SSE eightbytes).
        Type::Float(_) | Type::Vector(..) => Some(Eightbyte::Sse),
        Type::Struct(fields) => {
            let n = fields.len();
            for i in 0..n {
                let (foff, fty) = types.field_offset(ty, i as u32);
                classify_into(types, fty, offset + foff, ebs);
            }
            None
        }
        Type::Array(elem, len) => {
            let (elem, len) = (*elem, *len);
            let stride = types.stride(elem);
            for k in 0..len {
                classify_into(types, elem, offset + k * stride, ebs);
            }
            None
        }
        Type::Void => None,
    };
    if let Some(c) = cls {
        let size = types.size_of(ty).max(1);
        let first = (offset / 8) as usize;
        let last = ((offset + size - 1) / 8) as usize;
        for e in first..=last {
            if e < ebs.len() {
                ebs[e] = merge_class(ebs[e], c);
            }
        }
    }
}

/// Classify an aggregate `ty` (a struct or array passed/returned by value) into
/// its System V eightbyte classes, or [`AbiClass::Memory`] if it is larger than
/// two eightbytes (16 bytes).
pub(crate) fn classify_aggregate(types: &TypeContext, ty: TypeId) -> AbiClass {
    let size = types.size_of(ty);
    if size == 0 {
        return AbiClass::Regs(Vec::new());
    }
    if size > 16 {
        return AbiClass::Memory;
    }
    let n = size.div_ceil(8) as usize;
    let mut ebs = vec![None; n];
    classify_into(types, ty, 0, &mut ebs);
    // A never-classified eightbyte (pure padding) is SSE per the ABI.
    let classes = ebs.into_iter().map(|c| c.unwrap_or(Eightbyte::Sse)).collect();
    AbiClass::Regs(classes)
}

/// Whether a type is an aggregate (struct/array) that this backend represents,
/// at the codegen level, by a pointer to its in-memory storage.
fn is_aggregate(types: &TypeContext, ty: TypeId) -> bool {
    matches!(types.get(ty), Type::Struct(_) | Type::Array(_, _))
}

/// Round `v` up to a multiple of `align` (a power of two ≥ 1).
fn align_up_u64(v: u64, align: u64) -> u64 {
    v.div_ceil(align.max(1)) * align.max(1)
}

/// The number of INTEGER / SSE eightbytes in a register-classified aggregate.
fn count_classes(ebs: &[Eightbyte]) -> (usize, usize) {
    let int = ebs.iter().filter(|c| matches!(c, Eightbyte::Integer)).count();
    (int, ebs.len() - int)
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

/// The `icmp` results of `f` (by value index) used exactly once, by the
/// `cond_br` ending the block that computes them: the compares
/// [`X86Op::CmpBr`] fuses into their branch.
fn fusable_compares(f: &crate::ir::Function) -> DetHashSet<usize> {
    let mut uses = vec![0u32; f.value_count()];
    for (_, b) in f.blocks() {
        for &i in b.insts().iter().chain(b.terminator().as_ref()) {
            for &o in f.inst(i).operands() {
                uses[o.index()] += 1;
            }
        }
    }
    let mut out = DetHashSet::default();
    for (_, b) in f.blocks() {
        let Some(t) = b.terminator() else { continue };
        let term = f.inst(t);
        if !matches!(term.kind, InstKind::CondBr { .. }) {
            continue;
        }
        let c = term.operands()[0];
        if uses[c.index()] == 1
            && let ValueDef::Inst(id) = f.value(c).def
            && matches!(f.inst(id).kind, InstKind::ICmp(_))
            && b.insts().contains(&id)
        {
            out.insert(c.index());
        }
    }
    out
}

/// The two System V variadic frame-address intrinsics the x86-64 backend
/// recognizes by name. The C frontend declares each as an external
/// `ptr @name()` and calls it inside `va_start`; the backend replaces the call
/// with the corresponding frame address (see the [`isel`](self) module docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VaIntrinsic {
    /// `ptr @__lf_va_reg_save_area()` — the address of this variadic function's
    /// 176-byte register save area (`va_list.reg_save_area`).
    RegSaveArea,
    /// `ptr @__lf_va_overflow_area()` — the address of the first incoming stack
    /// argument, past every named stack argument (`va_list.overflow_arg_area`).
    OverflowArea,
}

impl VaIntrinsic {
    /// The intrinsic named by a direct callee, if it is one.
    fn from_name(name: &str) -> Option<VaIntrinsic> {
        match name {
            "__lf_va_reg_save_area" => Some(VaIntrinsic::RegSaveArea),
            "__lf_va_overflow_area" => Some(VaIntrinsic::OverflowArea),
            _ => None,
        }
    }
}

/// The x86-64 target: its register file/ABI plus the isel + encoding rules.
#[derive(Debug)]
pub struct X86_64Target {
    rf: RegFile,
    win64: bool,
    /// The relocation model, which picks the TLS access model
    /// ([`crate::codegen::linkage::tls_model`]).
    reloc_model: RelocModel,
    /// The higher 64-bit parts of each integer wider than 64 bits of the
    /// function being lowered, by value index (see the `wide` submodule).
    wide: RefCell<DetHashMap<usize, Vec<VReg>>>,
    /// The vreg of each register output of an inline asm but its first, by
    /// (asm result value index, output index), shared by the asm and its
    /// `asm_output`s (see the `inline_asm` submodule).
    asm_outs: RefCell<DetHashMap<(usize, usize), VReg>>,
    /// The `icmp` results (by value index) of the function being lowered
    /// whose only use is the `cond_br` ending their block: each is lowered
    /// there, fused into an [`X86Op::CmpBr`].
    fused: RefCell<DetHashSet<usize>>,
}

impl Default for X86_64Target {
    fn default() -> Self {
        Self::new()
    }
}

impl X86_64Target {
    /// Construct the x86-64 target with its fixed register file and SysV ABI.
    pub fn new() -> X86_64Target {
        X86_64Target { rf: RegFile::new(), win64: false, reloc_model: RelocModel::Static, wide: RefCell::default(), asm_outs: RefCell::default(), fused: RefCell::default() }
    }

    /// Construct the x86-64 target for the calling convention `cc`:
    /// [`CallConvKind::Win64`] selects the Microsoft x64 convention (see the
    /// [`isel`](self) module docs); anything else is System V.
    pub fn with_call_conv(cc: CallConvKind) -> X86_64Target {
        if cc == CallConvKind::Win64 {
            X86_64Target { rf: RegFile::win64(), win64: true, reloc_model: RelocModel::Static, wide: RefCell::default(), asm_outs: RefCell::default(), fused: RefCell::default() }
        } else {
            X86_64Target::new()
        }
    }

    /// Construct the x86-64 target for the calling convention of `os`
    /// (Win64 on Windows, System V elsewhere).
    pub fn for_os(os: TargetOs) -> X86_64Target {
        X86_64Target::with_call_conv(Triple::new(crate::target::TargetArch::X86_64, os).call_conv())
    }

    /// Lower for relocation model `model` (default [`RelocModel::Static`]),
    /// which decides how thread-local variables are reached.
    pub fn with_reloc_model(mut self, model: RelocModel) -> X86_64Target {
        self.reloc_model = model;
        self
    }

    /// The calling convention this target lowers calls with.
    pub fn call_conv_kind(&self) -> CallConvKind {
        if self.win64 { CallConvKind::Win64 } else { CallConvKind::SysV }
    }

    /// Lower function `func` of `module` to MIR over this target.
    pub fn select(&self, module: &Module, func: crate::ir::FuncId) -> crate::codegen::mir::MachineFunction {
        self.wide.borrow_mut().clear();
        self.asm_outs.borrow_mut().clear();
        *self.fused.borrow_mut() = fusable_compares(module.function(func));
        crate::codegen::isel::select(self, module, func)
    }

    /// Like [`X86_64Target::select`], but threads the module's symbol interner so
    /// the variadic frame-address intrinsics (`__lf_va_reg_save_area` /
    /// `__lf_va_overflow_area`) can be recognized by name at their call sites.
    pub fn select_with_syms(
        &self,
        module: &Module,
        func: crate::ir::FuncId,
        syms: &StrInterner,
    ) -> crate::codegen::mir::MachineFunction {
        self.wide.borrow_mut().clear();
        self.asm_outs.borrow_mut().clear();
        *self.fused.borrow_mut() = fusable_compares(module.function(func));
        crate::codegen::isel::select_with_syms(self, module, func, syms)
    }

    /// Resolve a value operand to a register, but materialize a **function
    /// reference used as a value** (its address taken / stored / passed) into a
    /// GPR via a RIP-relative [`X86Op::FuncAddr`] `lea`, rather than the
    /// framework default (a zero placeholder). Every other value defers to
    /// [`Lower::reg`]. A direct call is unaffected: its callee is recognized by
    /// [`Lower::callee_func`] and never routed through here.
    fn oper(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        if let ValueDef::Func(f) = lo.func().value(v).def {
            let fidx = f.index() as u32;
            let d = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(MachineInst::new(
                X86Op::FuncAddr.opcode(),
                vec![def_v(d), MachineOperand::Func(fidx)],
            ));
            return d;
        }
        lo.reg(v)
    }

    /// Whether a call's callee is a variadic function. Detected from a *direct*
    /// callee's function signature (`FuncType.variadic`); an indirect call
    /// (through a pointer) carries no signature here, so it is treated as
    /// non-variadic (the frontend passes such calls directly to known callees).
    fn callee_is_variadic(lo: &Lower<'_, Self>, callee: ValueId) -> bool {
        let Some(fidx) = lo.callee_func(callee) else { return false };
        let fid = FuncId::from_index(fidx as usize);
        matches!(lo.types().get(lo.module().function(fid).sig), Type::Func(ft) if ft.variadic)
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

    /// If `v` is an integer constant or a null pointer, its value.
    fn int_const(lo: &Lower<'_, Self>, v: ValueId) -> Option<Int> {
        if let ValueDef::Const(c) = lo.func().value(v).def
            && let Const::Null(_) = lo.module().consts().get(c)
            && !lo.types().is_vector(lo.func().value_type(v))
        {
            return Some(Int::ZERO);
        }
        Self::const_of(lo, v)
    }

    /// The compare `x pred y` (scalar integers or pointers): the operands
    /// of [`X86Op::SetccCmp`]/[`X86Op::CmpBr`] (`[Use a, Use b, Imm cc, Imm
    /// width]`) or, against a constant that fits an immediate, of
    /// [`X86Op::SetccCmpI`]/[`X86Op::CmpBrI`] (`[Use a, Imm v, Imm cc, Imm
    /// width]`, a constant first operand swapping the predicate); the flag
    /// says which.
    fn compare(&self, lo: &mut Lower<'_, Self>, pred: IntPred, x: ValueId, y: ValueId) -> (Vec<MachineOperand>, bool) {
        let width = lo.int_width(x);
        // 8/16/32/64-bit compares use the matching `cmp` form; any other
        // width is extended (by the predicate's signedness) first.
        if matches!(width, 8 | 16 | 32 | 64) {
            let imm_of = |v: ValueId| Self::int_const(lo, v).and_then(|c| imm_at(&c, width));
            let (r, k, pred) = match (imm_of(y), imm_of(x)) {
                (Some(k), _) => (x, k, pred),
                (None, Some(k)) => (y, k, swap_pred(pred)),
                (None, None) => {
                    let (a, b) = (self.oper(lo, x), self.oper(lo, y));
                    let ops = vec![use_v(a), use_v(b), imm(u64::from(cc_code(pred))), imm(u64::from(width))];
                    return (ops, false);
                }
            };
            let a = self.oper(lo, r);
            let ops = vec![
                use_v(a),
                MachineOperand::Imm(Int::from_i64(k)),
                imm(u64::from(cc_code(pred))),
                imm(u64::from(width)),
            ];
            return (ops, true);
        }
        let signed = matches!(pred, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge);
        let (a, _) = self.extended(lo, x, signed);
        let (b, w) = self.extended(lo, y, signed);
        (vec![use_v(a), use_v(b), imm(u64::from(cc_code(pred))), imm(u64::from(w))], false)
    }

    /// The `icmp` defining branch condition `c`, when it is fused into the
    /// branch (see [`fusable_compares`]): its predicate and operands.
    fn fused_compare(&self, lo: &Lower<'_, Self>, c: ValueId) -> Option<(IntPred, ValueId, ValueId)> {
        if !self.fused.borrow().contains(&c.index()) {
            return None;
        }
        let ValueDef::Inst(id) = lo.func().value(c).def else { return None };
        let inst = lo.func().inst(id);
        let InstKind::ICmp(pred) = inst.kind else { return None };
        let (x, y) = (inst.operands()[0], inst.operands()[1]);
        let ty = lo.func().value_type(x);
        if Self::wide_val(lo, x).is_some() || lo.types().is_vector(ty) {
            return None;
        }
        Some((pred, x, y))
    }

    fn lower_bin(&self, lo: &mut Lower<'_, Self>, op: BinOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let width = lo.int_width(inst.operands()[0]);
        // An ALU operation with a constant that fits an immediate.
        let group1 = match op {
            BinOp::Add => Some((0u64, true)),
            BinOp::Or => Some((1, true)),
            BinOp::And => Some((4, true)),
            BinOp::Sub => Some((5, false)),
            BinOp::Xor => Some((6, true)),
            BinOp::Mul => Some((u64::MAX, true)),
            _ => None,
        };
        if let Some((ext, commutative)) = group1
            && lo.mf().vreg_class(d) == RegClass::Gpr
        {
            let (x, y) = (inst.operands()[0], inst.operands()[1]);
            let imm_of = |v: ValueId| Self::int_const(lo, v).and_then(|c| imm_at(&c, width));
            let found = match (imm_of(y), imm_of(x)) {
                (Some(k), _) => Some((x, k)),
                (None, Some(k)) if commutative => Some((y, k)),
                _ => None,
            };
            if let Some((r, k)) = found {
                let a = self.oper(lo, r);
                let k = MachineOperand::Imm(Int::from_i64(k));
                let w = imm(u64::from(width));
                let inst = if ext == u64::MAX {
                    MachineInst::new(X86Op::ImulRI.opcode(), vec![def_v(d), use_v(a), k, w])
                } else {
                    MachineInst::new(X86Op::AluRI.opcode(), vec![def_v(d), use_v(a), k, imm(ext), w])
                };
                lo.emit(inst);
                return;
            }
        }
        let simple = match op {
            BinOp::Add => Some(X86Op::Add),
            BinOp::Sub => Some(X86Op::Sub),
            BinOp::And => Some(X86Op::And),
            BinOp::Or => Some(X86Op::Or),
            BinOp::Xor => Some(X86Op::Xor),
            BinOp::Mul => Some(X86Op::Imul),
            _ => None,
        };
        if let Some(x) = simple {
            let a = self.oper(lo, inst.operands()[0]);
            let b = self.oper(lo, inst.operands()[1]);
            lo.emit(MachineInst::new(
                x.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        // Scalar SSE floating-point arithmetic (F32/F64). The operand width comes
        // from the float type (32 or 64); the encoder picks the ss/sd form.
        let fop = match op {
            BinOp::FAdd => Some(X86Op::FAdd),
            BinOp::FSub => Some(X86Op::FSub),
            BinOp::FMul => Some(X86Op::FMul),
            BinOp::FDiv => Some(X86Op::FDiv),
            _ => None,
        };
        if let Some(x) = fop {
            let a = self.oper(lo, inst.operands()[0]);
            let b = self.oper(lo, inst.operands()[1]);
            lo.emit(MachineInst::new(
                x.opcode(),
                vec![def_v(d), use_v(a), use_v(b), imm(u64::from(width))],
            ));
            return;
        }
        match op {
            BinOp::Shl => self.lower_shift(lo, X86Op::ShlI, X86Op::ShlCl, d, inst, width),
            BinOp::LShr => self.lower_shift(lo, X86Op::ShrI, X86Op::ShrCl, d, inst, width),
            BinOp::AShr => self.lower_shift(lo, X86Op::SarI, X86Op::SarCl, d, inst, width),
            BinOp::UDiv => self.lower_div(lo, X86Op::Div, false, false, d, inst),
            BinOp::URem => self.lower_div(lo, X86Op::Div, false, true, d, inst),
            BinOp::SDiv => self.lower_div(lo, X86Op::Idiv, true, false, d, inst),
            BinOp::SRem => self.lower_div(lo, X86Op::Idiv, true, true, d, inst),
            // `frem` has no scalar SSE form (it is the `fmod` libcall). Rather
            // than silently emit a wrong result, fail loudly: the frontend must
            // lower `frem` to a call, or this backend must grow the libcall. All
            // other binops are handled above.
            BinOp::FRem => {
                panic!("x86-64 backend: `frem` is unsupported (needs an fmod libcall)")
            }
            _ => unreachable!("binop already handled: {op:?}"),
        }
    }

    /// An integer operand ready for an operation whose result depends on the
    /// bits above the value's width (right shifts, division, int→float, odd-
    /// width compares). Narrow values live in wider registers whose upper bits
    /// are not kept clean (an `i8` add of 200 + 100 leaves 300 in the register),
    /// so anything but an `i32`/`i64` is sign- or zero-extended to 64 bits first.
    /// Returns the register and the width to operate at (32 for a value that
    /// fits, else 64).
    fn extended(&self, lo: &mut Lower<'_, Self>, v: ValueId, signed: bool) -> (VReg, u32) {
        let r = self.oper(lo, v);
        let width = lo.int_width(v);
        if width == 32 || width >= 64 {
            return (r, width.min(64));
        }
        let d = lo.fresh_vreg(RegClass::Gpr);
        let op = if signed { X86Op::Movsx } else { X86Op::Movzx };
        lo.emit(MachineInst::new(op.opcode(), vec![def_v(d), use_v(r), imm(u64::from(width)), imm(64)]));
        (d, if width < 32 { 32 } else { 64 })
    }

    /// An `i1` branch/select condition as a register holding exactly 0 or 1. A
    /// compare's result already is; anything else (e.g. a `trunc` to `i1`) may
    /// carry garbage above bit 0 and is zero-extended.
    fn clean_cond(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        let is_compare = matches!(lo.func().value(v).def, ValueDef::Inst(id)
            if matches!(lo.func().inst(id).kind, InstKind::ICmp(_) | InstKind::FCmp(_)));
        if is_compare || lo.int_width(v) >= 8 {
            return self.oper(lo, v);
        }
        self.extended(lo, v, false).0
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_shift(
        &self,
        lo: &mut Lower<'_, Self>,
        imm_op: X86Op,
        cl_op: X86Op,
        d: VReg,
        inst: &InstData,
        width: u32,
    ) {
        // A right shift brings the bits above the width down into the result.
        let (a, width) = match imm_op {
            X86Op::ShrI => self.extended(lo, inst.operands()[0], false),
            X86Op::SarI => self.extended(lo, inst.operands()[0], true),
            _ => (self.oper(lo, inst.operands()[0]), width),
        };
        if let Some(c) = Self::const_of(lo, inst.operands()[1]) {
            let count = c.to_i64().unwrap_or(0) as u64;
            lo.emit(MachineInst::new(
                imm_op.opcode(),
                vec![def_v(d), use_v(a), imm(count), imm(u64::from(width))],
            ));
        } else {
            let b = self.oper(lo, inst.operands()[1]);
            let rcx = regs::gpr(regs::RCX);
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(rcx), use_v(b)]));
            lo.emit(MachineInst::new(
                cl_op.opcode(),
                vec![def_v(d), use_v(a), use_p(rcx), imm(u64::from(width))],
            ));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_div(
        &self,
        lo: &mut Lower<'_, Self>,
        div_op: X86Op,
        signed: bool,
        want_rem: bool,
        d: VReg,
        inst: &InstData,
    ) {
        // The dividend and divisor are extended to at least 32 bits (division
        // sees every bit of both), and the division runs at that width.
        let (a, _) = self.extended(lo, inst.operands()[0], signed);
        let (b, width) = self.extended(lo, inst.operands()[1], signed);
        let rax = regs::gpr(regs::RAX);
        let rdx = regs::gpr(regs::RDX);
        // dividend low half -> rax
        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(rax), use_v(a)]));
        // extend into rdx
        if signed {
            lo.emit(MachineInst::new(
                X86Op::Cqo.opcode(),
                vec![def(rdx), use_p(rax), imm(u64::from(width))],
            ));
        } else {
            lo.emit(MachineInst::new(X86Op::ZeroRdx.opcode(), vec![def(rdx)]));
        }
        lo.emit(MachineInst::new(
            div_op.opcode(),
            vec![def(rax), def(rdx), use_p(rax), use_p(rdx), use_v(b), imm(u64::from(width))],
        ));
        let src = if want_rem { rdx } else { rax };
        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_p(src)]));
    }

    /// `fneg`: flip the IEEE sign bit (matching the reference semantics, which is
    /// a sign flip, not `0 - x`). Materialize the sign mask
    /// (`0x8000_0000_0000_0000` / `0x8000_0000`) in an xmm and `xorpd`/`xorps`.
    fn lower_fneg(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let d = lo.result_reg(inst);
        let s = self.oper(lo, inst.operands()[0]);
        let width = lo.int_width(inst.operands()[0]);
        let mask_bits: u64 =
            if width == 64 { 0x8000_0000_0000_0000 } else { 0x8000_0000 };
        let mask = lo.fresh_vreg(RegClass::Fp);
        lo.emit(MachineInst::new(
            X86Op::LoadFConst.opcode(),
            vec![def_v(mask), imm(mask_bits), imm(u64::from(width))],
        ));
        lo.emit(MachineInst::new(
            X86Op::FXor.opcode(),
            vec![def_v(d), use_v(s), use_v(mask), imm(u64::from(width))],
        ));
    }

    /// `fcmp`: `ucomis` then `setcc` with the ordered/unordered parity fixup
    /// packed by [`fcmp_pack`]. The result is an `i1` in a gpr.
    fn lower_fcmp(&self, lo: &mut Lower<'_, Self>, pred: FloatPred, inst: &InstData) {
        let d = lo.result_reg(inst);
        match fcmp_pack(pred) {
            None => {
                // `False`/`True` are constants.
                let v = u64::from(pred == FloatPred::True);
                lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(d), imm(v)]));
            }
            Some(packed) => {
                let a = self.oper(lo, inst.operands()[0]);
                let b = self.oper(lo, inst.operands()[1]);
                let width = lo.int_width(inst.operands()[0]);
                lo.emit(MachineInst::new(
                    X86Op::FCmpSet.opcode(),
                    vec![def_v(d), use_v(a), use_v(b), imm(packed), imm(u64::from(width))],
                ));
            }
        }
    }

    /// Conversions. Float↔float and int↔float go through the SSE `cvt*` forms;
    /// every other cast (integer width change, ptr↔int, bitcast within a class)
    /// is a low-bits-preserving copy, matching the existing integer behavior.
    fn lower_cast(&self, lo: &mut Lower<'_, Self>, op: CastOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let s = self.oper(lo, inst.operands()[0]);
        let src_w = lo.int_width(inst.operands()[0]);
        let dst_w = lo.types().bit_width(inst.ty).unwrap_or(64);
        match op {
            CastOp::FpTrunc => lo.emit(MachineInst::new(
                X86Op::Cvtsd2ss.opcode(),
                vec![def_v(d), use_v(s)],
            )),
            CastOp::FpExt => lo.emit(MachineInst::new(
                X86Op::Cvtss2sd.opcode(),
                vec![def_v(d), use_v(s)],
            )),
            CastOp::FpToSi => {
                // bit0 of flags = 64-bit gpr destination.
                let flags = u64::from(dst_w > 32);
                lo.emit(MachineInst::new(
                    X86Op::CvtF2si.opcode(),
                    vec![def_v(d), use_v(s), imm(u64::from(src_w)), imm(flags)],
                ));
            }
            CastOp::FpToUi => {
                // A ≤32-bit unsigned result is exact through a 64-bit signed
                // `cvttsd2si` (it lands in `[0, 2^63)`). A full unsigned-64 result
                // needs the 2^63 fix-up (flags bit1): values ≥ 2^63 are converted
                // as `x - 2^63` and biased back. Truncation is toward zero.
                let flags: u64 = if dst_w > 32 { 0b11 } else { 1 };
                lo.emit(MachineInst::new(
                    X86Op::CvtF2si.opcode(),
                    vec![def_v(d), use_v(s), imm(u64::from(src_w)), imm(flags)],
                ));
            }
            CastOp::SiToFp => {
                // bit0 = 64-bit gpr source. A narrow source is sign-extended
                // first (its register's upper bits are not kept clean).
                let (s, src_w) = self.extended(lo, inst.operands()[0], true);
                let flags = u64::from(src_w > 32);
                lo.emit(MachineInst::new(
                    X86Op::CvtSi2f.opcode(),
                    vec![def_v(d), use_v(s), imm(u64::from(dst_w)), imm(flags)],
                ));
            }
            CastOp::UiToFp => {
                // Unsigned int→float: a ≤32-bit source zero-extends to 64 bits
                // (flags bit1) then a 64-bit signed conversion; a 64-bit source
                // uses the full unsigned-64 fix-up (flags bit2) — direct
                // `cvtsi2sd` when the sign bit is clear, else the halve-and-round
                // `(x>>1)|(x&1)` sequence followed by a doubling `addsd`, which
                // reproduces round-to-nearest for values ≥ 2^63.
                // A narrow source is zero-extended first; an odd width above 32
                // then fits a plain signed 64-bit conversion (it is < 2^63).
                let (s, flags) = if src_w >= 64 {
                    (s, 0b100)
                } else {
                    let (x, w) = self.extended(lo, inst.operands()[0], false);
                    (x, if w > 32 { 0b1 } else { 0b10 })
                };
                lo.emit(MachineInst::new(
                    X86Op::CvtSi2f.opcode(),
                    vec![def_v(d), use_v(s), imm(u64::from(dst_w)), imm(flags)],
                ));
            }
            CastOp::SExt => lo.emit(MachineInst::new(
                X86Op::Movsx.opcode(),
                vec![def_v(d), use_v(s), imm(u64::from(src_w)), imm(u64::from(dst_w))],
            )),
            CastOp::ZExt => lo.emit(MachineInst::new(
                X86Op::Movzx.opcode(),
                vec![def_v(d), use_v(s), imm(u64::from(src_w)), imm(u64::from(dst_w))],
            )),
            // A bitcast between an integer and a float crosses register files:
            // `movd`/`movq` carries the bits.
            CastOp::Bitcast if lo.mf().vreg_class(d) != lo.mf().vreg_class(s) => {
                let is64 = u64::from(dst_w.max(src_w) > 32);
                let op = if lo.mf().vreg_class(d) == RegClass::Fp { X86Op::MovGprToX } else { X86Op::MovXToGpr };
                lo.emit(MachineInst::new(op.opcode(), vec![def_v(d), use_v(s), imm(is64)]));
            }
            // Truncation drops high bits, and ptr↔int / same-class bitcast preserve
            // the bit pattern: a plain register copy is correct (consumers operate
            // at the result's width).
            _ => lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(s)])),
        }
    }

    /// Materialize `base + off` (a byte displacement) into a fresh GPR, or return
    /// `base` unchanged when `off == 0`.
    fn add_off(&self, lo: &mut Lower<'_, Self>, base: VReg, off: u64) -> VReg {
        if off == 0 {
            return base;
        }
        let d = lo.fresh_vreg(RegClass::Gpr);
        if let Ok(k) = i32::try_from(off) {
            lo.emit(MachineInst::new(
                X86Op::AluRI.opcode(),
                vec![def_v(d), use_v(base), imm(k as u64), imm(0), imm(64)],
            ));
            return d;
        }
        let k = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(k), imm(off)]));
        lo.emit(MachineInst::new(
            X86Op::Add.opcode(),
            vec![def_v(d), use_v(base), use_v(k), imm(64)],
        ));
        d
    }

    /// Emit `lea d, [rbp + off]` into a fresh GPR (addresses an incoming
    /// stack-passed parameter, above the return address).
    fn lea_rbp(&self, lo: &mut Lower<'_, Self>, off: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::LeaRbpOff.opcode(), vec![def_v(d), imm(off)]));
        d
    }

    /// Emit `lea d, [rsp + off]` into a fresh GPR (addresses the outgoing
    /// stack-argument area at the bottom of the frame).
    fn lea_rsp(&self, lo: &mut Lower<'_, Self>, off: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::LeaRspOff.opcode(), vec![def_v(d), imm(off)]));
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
            lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(t), use_v(sp), imm(chunk)]));
            let dp = self.add_off(lo, dst, o);
            lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(dp), use_v(t), imm(chunk)]));
            o += chunk;
        }
    }

    /// Lower a `call`, implementing the System V AMD64 ABI for by-value struct
    /// arguments and returns on top of the existing scalar/float handling.
    ///
    /// A struct value is represented, at this codegen level, by a GPR vreg
    /// holding a pointer to the struct's in-memory storage. A ≤16-byte struct
    /// argument is loaded eightbyte-by-eightbyte from that storage into the
    /// assigned integer/SSE argument registers; a MEMORY-class struct (or one
    /// that no longer fits the remaining registers) is copied into the outgoing
    /// stack area. A ≤16-byte struct result comes back in `rax`/`rdx` and/or
    /// `xmm0`/`xmm1` and is stored into a fresh result slot; a MEMORY-class result
    /// uses a hidden `sret` pointer (a caller-allocated slot passed in `rdi`).
    fn lower_call(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];
        let wide_ret = inst.result().and_then(|r| Self::wide_val(lo, r));
        // The legalizer's 128-bit multiply is not a real call.
        if lo.callee_name(callee) == Some(MUL128_PSEUDO) {
            return self.lower_mul128(lo, inst);
        }
        if wide_ret.is_some_and(|n| n != 2) || args.iter().any(|&a| Self::wide_val(lo, a).is_some_and(|n| n != 2)) {
            panic!("x86-64 backend: integers wider than 128 bits cannot be passed or returned");
        }
        if self.win64 {
            if wide_ret.is_some() || args.iter().any(|&a| Self::wide_val(lo, a).is_some()) {
                panic!("x86-64 backend: 128-bit integer arguments are not supported under the Microsoft x64 convention");
            }
            return self.lower_call_win64(lo, inst);
        }
        let cc = &self.rf.cc;

        // System V variadic frame-address intrinsics. A `call` to one of these
        // specially-named external functions is not a real call: it materializes
        // a frame address the C frontend's `va_start` needs (see the module
        // documentation for the `va_list` layout and offset conventions). They are
        // only valid inside a variadic function (the prologue set up the slots).
        match lo.callee_name(callee).and_then(VaIntrinsic::from_name) {
            Some(VaIntrinsic::RegSaveArea) => {
                let d = lo.result_reg(inst);
                let slot = lo
                    .va_reg_save()
                    .expect("__lf_va_reg_save_area called outside a variadic function");
                lo.emit(self.frame_addr(d, slot));
                return;
            }
            Some(VaIntrinsic::OverflowArea) => {
                let d = lo.result_reg(inst);
                let off = lo
                    .va_overflow_off()
                    .expect("__lf_va_overflow_area called outside a variadic function");
                lo.emit(MachineInst::new(X86Op::LeaRbpOff.opcode(), vec![def_v(d), imm(off)]));
                return;
            }
            None => {}
        }

        // Is this a call to a variadic function? Under System V the caller must
        // then set `al` to the number of vector (SSE) argument registers used.
        let variadic_call = Self::callee_is_variadic(lo, callee);

        // Return classification.
        let ret_ty = inst.result().map(|r| lo.func().value_type(r));
        let ret_agg = ret_ty.filter(|&t| is_aggregate(lo.types(), t));
        let ret_class = ret_agg.map(|t| classify_aggregate(lo.types(), t));
        let sret = matches!(ret_class, Some(AbiClass::Memory));

        // `reg_moves`: the final `arg-reg <- value-vreg` moves, emitted as one
        // consecutive run right before the `call` so no competing vreg definition
        // sits in the gap between an argument register's write and the call (the
        // register allocator's fixed-register liveness reasons point-to-point).
        let mut reg_moves: Vec<(PReg, VReg)> = Vec::new();
        let mut int_i = 0usize;
        let mut fp_i = 0usize;

        // A MEMORY-class return: allocate the return slot and pass its address as
        // the hidden first integer argument (`rdi`); the callee writes through it.
        let mut ret_slot = None;
        if sret {
            let t = ret_agg.unwrap();
            let size = align_up_u64(lo.byte_size(t).max(8), 8);
            let align = lo.types().align_of(t).max(8);
            let slot = lo.new_slot(size, align);
            ret_slot = Some(slot);
            let ptr = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(self.frame_addr(ptr, slot));
            reg_moves.push((cc.arg_regs[0], ptr));
            int_i = 1;
        }

        let mut stack_off = 0u64;
        for &arg in args {
            let ty = lo.func().value_type(arg);
            if is_aggregate(lo.types(), ty) {
                if let AbiClass::Regs(ebs) = classify_aggregate(lo.types(), ty) {
                    let (need_int, need_sse) = count_classes(&ebs);
                    if int_i + need_int <= cc.arg_regs.len()
                        && fp_i + need_sse <= cc.fp_arg_regs.len()
                    {
                        let ptr = self.oper(lo, arg);
                        for (k, c) in ebs.iter().enumerate() {
                            let (cls, areg) = match c {
                                Eightbyte::Integer => {
                                    let a = cc.arg_regs[int_i];
                                    int_i += 1;
                                    (RegClass::Gpr, a)
                                }
                                Eightbyte::Sse => {
                                    let a = cc.fp_arg_regs[fp_i];
                                    fp_i += 1;
                                    (RegClass::Fp, a)
                                }
                            };
                            let sp = self.add_off(lo, ptr, 8 * k as u64);
                            let d = lo.fresh_vreg(cls);
                            lo.emit(MachineInst::new(
                                X86Op::Load.opcode(),
                                vec![def_v(d), use_v(sp), imm(8)],
                            ));
                            reg_moves.push((areg, d));
                        }
                        continue;
                    }
                }
                // MEMORY class, or not enough registers left: pass the whole
                // aggregate in the outgoing stack area.
                let size = lo.byte_size(ty);
                let align = lo.types().align_of(ty).max(8);
                stack_off = align_up_u64(stack_off, align);
                let ptr = self.oper(lo, arg);
                let dst = self.lea_rsp(lo, stack_off);
                self.emit_memcpy(lo, dst, ptr, size);
                stack_off += align_up_u64(size, 8);
            } else if Self::wide_val(lo, arg).is_some() {
                // An `i128`: two integer registers, or a 16-aligned stack slot
                // when fewer than two are left (System V, as gcc's __int128).
                let p = self.parts(lo, arg);
                if int_i + 2 <= cc.arg_regs.len() {
                    reg_moves.push((cc.arg_regs[int_i], p[0]));
                    reg_moves.push((cc.arg_regs[int_i + 1], p[1]));
                    int_i += 2;
                } else {
                    stack_off = align_up_u64(stack_off, 16);
                    for (k, &v) in p.iter().enumerate() {
                        let dp = self.lea_rsp(lo, stack_off + 8 * k as u64);
                        lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(dp), use_v(v), imm(8)]));
                    }
                    stack_off += 16;
                }
            } else {
                // Scalar / pointer / float argument.
                let v = self.oper(lo, arg);
                let is_fp = lo.mf().vreg_class(v) == RegClass::Fp;
                let has_reg = if is_fp { fp_i < cc.fp_arg_regs.len() } else { int_i < cc.arg_regs.len() };
                if has_reg {
                    let a = if is_fp {
                        let a = cc.fp_arg_regs[fp_i];
                        fp_i += 1;
                        a
                    } else {
                        let a = cc.arg_regs[int_i];
                        int_i += 1;
                        a
                    };
                    reg_moves.push((a, v));
                } else {
                    // A 16-byte vector takes a 16-aligned 16-byte slot.
                    // A vector (mask included) crosses as its 16-byte register
                    // image, whatever its memory size.
                    let sz = if lo.types().is_vector(ty) { 16 } else { lo.byte_size(ty) };
                    if sz == 16 {
                        stack_off = align_up_u64(stack_off, 16);
                    }
                    let dp = self.lea_rsp(lo, stack_off);
                    lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(dp), use_v(v), imm(sz)]));
                    stack_off += sz.max(8);
                }
            }
        }
        if stack_off > 0 {
            lo.reserve_outgoing(align_up_u64(stack_off, 16));
        }

        let mut used_arg_regs: Vec<PReg> = reg_moves.iter().map(|&(areg, _)| areg).collect();
        for (areg, r) in reg_moves {
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(areg), use_v(r)]));
        }

        // Variadic call: `al` = number of SSE argument registers used (0..=8).
        // `mov eax, imm` sets it (and zeroes the rest of eax), matching gcc/clang.
        // rax is added to the call's used registers so its value reaches the call.
        if variadic_call {
            let rax = regs::gpr(regs::RAX);
            lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def(rax), imm(fp_i as u64)]));
            used_arg_regs.push(rax);
        }

        // The primary return register (`rax`/`xmm0`); struct results reclaim their
        // eightbytes from `rax`/`rdx`/`xmm0`/`xmm1`, all covered by the clobber set.
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
        for &areg in &used_arg_regs {
            operands.push(use_p(areg));
        }
        lo.emit(MachineInst::new(X86Op::Call.opcode(), operands));

        match &ret_class {
            Some(AbiClass::Memory) => {
                // The result already sits in the caller-allocated sret slot.
                let d = lo.result_reg(inst);
                lo.emit(self.frame_addr(d, ret_slot.unwrap()));
            }
            Some(AbiClass::Regs(ebs)) => {
                // Rescue every returned eightbyte register into a vreg (one
                // consecutive run right after the call), then store them into a
                // fresh result slot whose address becomes the result value.
                let ebs = ebs.clone();
                let mut ic = 0usize;
                let mut sc = 0usize;
                let mut saved: Vec<(usize, VReg)> = Vec::with_capacity(ebs.len());
                for (k, c) in ebs.iter().enumerate() {
                    let (cls, r) = match c {
                        Eightbyte::Integer => {
                            let r = if ic == 0 { cc.ret_reg } else { regs::gpr(regs::RDX) };
                            ic += 1;
                            (RegClass::Gpr, r)
                        }
                        Eightbyte::Sse => {
                            let r = if sc == 0 { cc.fp_ret_reg } else { regs::xmm(1) };
                            sc += 1;
                            (RegClass::Fp, r)
                        }
                    };
                    let v = lo.fresh_vreg(cls);
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(v), use_p(r)]));
                    saved.push((k, v));
                }
                let t = ret_agg.unwrap();
                let size = align_up_u64(lo.byte_size(t).max(8), 8);
                let align = lo.types().align_of(t).max(8);
                let slot = lo.new_slot(size, align);
                let d = lo.result_reg(inst);
                lo.emit(self.frame_addr(d, slot));
                for (k, v) in saved {
                    let dp = self.add_off(lo, d, 8 * k as u64);
                    lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(dp), use_v(v), imm(8)]));
                }
            }
            None if wide_ret.is_some() => {
                // An `i128` result in rax:rdx.
                let p = self.parts(lo, inst.result().unwrap());
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(p[0]), use_p(ret_reg)]));
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(p[1]), use_p(regs::gpr(regs::RDX))]));
            }
            None => {
                if inst.result().is_some() {
                    let d = lo.result_reg(inst);
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_p(ret_reg)]));
                }
            }
        }
    }

    /// Lower a `syscall` under the Linux x86-64 kernel ABI: number in `rax`,
    /// arguments in `rdi, rsi, rdx, r10, r8, r9`, result in `rax`; the `syscall`
    /// instruction clobbers `rcx` and `r11`.
    ///
    /// As in [`Self::lower_call`], every operand is materialized into a vreg
    /// *first*, and only then are the fixed-register moves emitted as one
    /// consecutive run right before the instruction, so no vreg definition sits
    /// in the gap between an ABI register's write and its read. `r10` is special:
    /// it is not allocatable but one of the spill/reload **scratch** registers, so
    /// a later reload (of a spilled operand moved into another register) could
    /// overwrite it. Its move is therefore emitted **last**: after it nothing but
    /// the `syscall` itself runs, and a reload feeding that very move targets the
    /// move's own source, which is harmless.
    fn lower_syscall(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        use regs::{R8, R9, R10, R11, RAX, RCX, RDI, RDX, RSI};
        const ARG_REGS: [u16; 6] = [RDI, RSI, RDX, R10, R8, R9];
        let ops = inst.operands();
        let vals: Vec<VReg> = ops.iter().map(|&o| self.oper(lo, o)).collect();

        let mut reg_moves: Vec<(PReg, VReg)> = vec![(regs::gpr(RAX), vals[0])];
        for (k, &v) in vals[1..].iter().enumerate() {
            reg_moves.push((regs::gpr(ARG_REGS[k]), v));
        }
        // Stable: `r10` last, everything else in ABI order.
        reg_moves.sort_by_key(|&(r, _)| r.num == R10);
        let used: Vec<PReg> = reg_moves.iter().map(|&(r, _)| r).collect();
        for (r, v) in reg_moves {
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(r), use_v(v)]));
        }

        let rax = regs::gpr(RAX);
        let mut operands = vec![def(rax), def(regs::gpr(RCX)), def(regs::gpr(R11))];
        operands.extend(used.into_iter().map(use_p));
        lo.emit(MachineInst::new(X86Op::Syscall.opcode(), operands));
        let d = lo.result_reg(inst);
        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_p(rax)]));
    }

    /// Lower an atomic memory operation or fence. x86-64 is TSO (the Intel SDM
    /// Vol. 3A §9.2 memory-ordering model): ordinary loads are not reordered
    /// with other loads, stores not with other stores, and a load may pass
    /// only an *earlier store to a different location*; `lock`-prefixed
    /// instructions and `xchg` with memory are full barriers. So:
    ///
    /// | IR | x86-64 |
    /// |---|---|
    /// | `atomic_load` (any ordering) | `mov` |
    /// | `atomic_store` relaxed/release | `mov` |
    /// | `atomic_store seq_cst` | `xchg` (a store that is also a full barrier) |
    /// | `atomic_rmw xchg` | `xchg` |
    /// | `atomic_rmw add`/`sub` | `lock xadd` (`sub` negates first) |
    /// | other `atomic_rmw` | `lock cmpxchg` loop ([`X86Op::RmwLoop`]) |
    /// | `cmpxchg` | `lock cmpxchg` (expected/old in `rax`) |
    /// | `fence seq_cst` | `mfence` |
    /// | `fence` acquire/release/acq_rel | nothing (TSO already orders them; the IR-level barrier is what kept the optimizer from moving memory ops across it) |
    ///
    /// Every `lock`ed form is a full barrier, so the rmw/cmpxchg orderings need
    /// nothing extra.
    fn lower_atomic(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        use crate::ir::inst::{AtomicOrdering, RmwOp};
        let ops = inst.operands();
        match &inst.kind {
            InstKind::AtomicLoad { ty, .. } => {
                let d = lo.result_reg(inst);
                let ptr = self.oper(lo, ops[0]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(d), use_v(ptr), imm(size)]));
            }
            InstKind::AtomicStore { ty, ordering, .. } => {
                let ptr = self.oper(lo, ops[0]);
                let val = self.oper(lo, ops[1]);
                let size = lo.byte_size(*ty);
                if *ordering == AtomicOrdering::SeqCst {
                    // `xchg` = store + full barrier; the swapped-out value is
                    // discarded.
                    let junk = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(
                        X86Op::Xchg.opcode(),
                        vec![def_v(junk), use_v(ptr), use_v(val), imm(size)],
                    ));
                } else {
                    lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(ptr), use_v(val), imm(size)]));
                }
            }
            InstKind::AtomicRmw { op, ty, .. } => {
                let d = lo.result_reg(inst);
                let ptr = self.oper(lo, ops[0]);
                let val = self.oper(lo, ops[1]);
                let size = lo.byte_size(*ty);
                match op {
                    RmwOp::Xchg => lo.emit(MachineInst::new(
                        X86Op::Xchg.opcode(),
                        vec![def_v(d), use_v(ptr), use_v(val), imm(size)],
                    )),
                    RmwOp::Add | RmwOp::Sub => lo.emit(MachineInst::new(
                        X86Op::LockXadd.opcode(),
                        vec![def_v(d), use_v(ptr), use_v(val), imm(size), imm(u64::from(*op == RmwOp::Sub))],
                    )),
                    _ => {
                        let rax = regs::gpr(regs::RAX);
                        let tmp = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(
                            X86Op::RmwLoop.opcode(),
                            vec![
                                def(rax),
                                def_v(tmp),
                                use_v(ptr),
                                use_v(val),
                                imm(size),
                                imm(u64::from(op.code())),
                            ],
                        ));
                        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_p(rax)]));
                    }
                }
            }
            InstKind::CmpXchg { ty, .. } => {
                let d = lo.result_reg(inst);
                let ptr = self.oper(lo, ops[0]);
                let expected = self.oper(lo, ops[1]);
                let new = self.oper(lo, ops[2]);
                let size = lo.byte_size(*ty);
                let rax = regs::gpr(regs::RAX);
                // The fixed-register window is exactly `mov rax, expected;
                // lock cmpxchg; mov d, rax`: every operand is materialized
                // before it, so no other definition can land in rax in between.
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(rax), use_v(expected)]));
                lo.emit(MachineInst::new(
                    X86Op::LockCmpxchg.opcode(),
                    vec![def(rax), use_p(rax), use_v(ptr), use_v(new), imm(size)],
                ));
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_p(rax)]));
            }
            InstKind::Fence(ordering) => {
                if *ordering == AtomicOrdering::SeqCst {
                    lo.emit(MachineInst::new(X86Op::Mfence.opcode(), Vec::new()));
                }
            }
            other => unreachable!("lower_atomic on {other:?}"),
        }
    }

    /// Lower the entry prologue with System V aggregate/`sret`/stack-parameter
    /// support. A register-passed struct parameter is stored into a private home
    /// slot (so the body sees it in memory) and its vreg is that slot's address; a
    /// stack-passed struct/scalar is addressed in place at `[rbp + 16 + off]`; a
    /// MEMORY-class return reserves the hidden `sret` pointer (in `rdi`) into an
    /// aux slot for the return lowering.
    fn lower_prologue_x86(&self, lo: &mut Lower<'_, Self>) {
        let cc = &self.rf.cc;
        let entry = lo.mf().entry().expect("a function being lowered has an entry block");
        let param_vregs: Vec<VReg> = lo.mf().block(entry).params.clone();
        let (sig_params, ret_ty, variadic) = match lo.types().get(lo.func().sig) {
            Type::Func(ft) => (ft.params.clone(), ft.ret, ft.variadic),
            _ => (Vec::new(), lo.func().sig, false),
        };
        let sret = is_aggregate(lo.types(), ret_ty)
            && matches!(classify_aggregate(lo.types(), ret_ty), AbiClass::Memory);

        // A variadic function spills its incoming argument registers into a
        // register save area so `va_arg` can walk them (see [`Self::spill_va_regs`]).
        if variadic {
            self.spill_va_regs(lo);
        }

        let mut int_i = 0usize;
        let mut fp_i = 0usize;
        if sret {
            // The hidden return pointer arrives in rdi; stash it for `ret`.
            let slot = lo.new_slot(8, 8);
            lo.set_aux_slot(slot);
            lo.emit(MachineInst::new(
                X86Op::StoreFrame.opcode(),
                vec![use_p(cc.arg_regs[0]), MachineOperand::Frame(slot)],
            ));
            int_i = 1;
        }

        let param_vals: Vec<ValueId> = lo.func().block(lo.func().entry().expect("an entry")).params().to_vec();
        let mut stack_in = 16u64; // first incoming stack arg, above the return address
        for (i, &pv) in param_vregs.iter().enumerate() {
            let ty = sig_params[i];
            if let Some(n) = Self::wide_ty(lo, ty) {
                // An `i128` parameter: two integer registers, or a 16-aligned
                // stack slot (see `lower_call`).
                assert_eq!(n, 2, "x86-64 backend: integers wider than 128 bits cannot be passed");
                let p = self.parts(lo, param_vals[i]);
                if int_i + 2 <= cc.arg_regs.len() {
                    for (k, &v) in p.iter().enumerate() {
                        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(v), use_p(cc.arg_regs[int_i + k])]));
                    }
                    int_i += 2;
                } else {
                    stack_in = align_up_u64(stack_in, 16);
                    for (k, &v) in p.iter().enumerate() {
                        let a = self.lea_rbp(lo, stack_in + 8 * k as u64);
                        lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(v), use_v(a), imm(8)]));
                    }
                    stack_in += 16;
                }
                continue;
            }
            if is_aggregate(lo.types(), ty) {
                let size = lo.byte_size(ty);
                let align = lo.types().align_of(ty).max(8);
                if let AbiClass::Regs(ebs) = classify_aggregate(lo.types(), ty) {
                    let (need_int, need_sse) = count_classes(&ebs);
                    if int_i + need_int <= cc.arg_regs.len()
                        && fp_i + need_sse <= cc.fp_arg_regs.len()
                    {
                        let home = lo.new_slot(align_up_u64(size.max(8), 8), align);
                        lo.emit(self.frame_addr(pv, home));
                        for (k, c) in ebs.iter().enumerate() {
                            let (cls, areg) = match c {
                                Eightbyte::Integer => {
                                    let a = cc.arg_regs[int_i];
                                    int_i += 1;
                                    (RegClass::Gpr, a)
                                }
                                Eightbyte::Sse => {
                                    let a = cc.fp_arg_regs[fp_i];
                                    fp_i += 1;
                                    (RegClass::Fp, a)
                                }
                            };
                            let v = lo.fresh_vreg(cls);
                            lo.emit(MachineInst::new(
                                X86Op::MovRR.opcode(),
                                vec![def_v(v), use_p(areg)],
                            ));
                            let dp = self.add_off(lo, pv, 8 * k as u64);
                            lo.emit(MachineInst::new(
                                X86Op::Store.opcode(),
                                vec![use_v(dp), use_v(v), imm(8)],
                            ));
                        }
                        continue;
                    }
                }
                // MEMORY class or register exhaustion: the caller placed a copy on
                // the stack; address it in place.
                stack_in = align_up_u64(stack_in, align);
                let d = self.lea_rbp(lo, stack_in);
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(pv), use_v(d)]));
                stack_in += align_up_u64(size, 8);
            } else {
                let is_fp = lo.mf().vreg_class(pv) == RegClass::Fp;
                let has_reg = if is_fp { fp_i < cc.fp_arg_regs.len() } else { int_i < cc.arg_regs.len() };
                if has_reg {
                    let a = if is_fp {
                        let a = cc.fp_arg_regs[fp_i];
                        fp_i += 1;
                        a
                    } else {
                        let a = cc.arg_regs[int_i];
                        int_i += 1;
                        a
                    };
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(pv), use_p(a)]));
                } else {
                    // A vector (mask included) crosses as its 16-byte register
                    // image, whatever its memory size.
                    let sz = if lo.types().is_vector(ty) { 16 } else { lo.byte_size(ty) };
                    if sz == 16 {
                        stack_in = align_up_u64(stack_in, 16);
                    }
                    let p = self.lea_rbp(lo, stack_in);
                    lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(pv), use_v(p), imm(sz)]));
                    stack_in += sz.max(8);
                }
            }
        }

        // `overflow_arg_area` starts just past the named stack arguments (which end
        // at `[rbp + stack_in]`). For the common case — every named argument in a
        // register — that is `rbp + 16`, right above the saved return address.
        if variadic {
            lo.set_va_overflow_off(stack_in);
        }
    }

    /// Spill a variadic function's incoming System V argument registers into a
    /// 176-byte register save area at the top of the prologue, and record the
    /// slot for `__lf_va_reg_save_area`.
    ///
    /// Layout (matching the psABI so `va_arg`'s `gp_offset`/`fp_offset` walk is
    /// correct): the 6 integer arg regs `rdi, rsi, rdx, rcx, r8, r9` at byte
    /// offsets `0, 8, .., 40`, then the 8 SSE regs `xmm0..7` at offsets
    /// `48, 64, .., 160` (16-byte stride; only the low 8 bytes of each — enough
    /// for `double`/`float` varargs — are stored). The SSE registers are saved
    /// unconditionally: reading `xmm0..7` is always safe, so no `test al,al`
    /// guard (and no prologue control flow) is needed — a correct caller only
    /// ever passes, and `va_arg` only ever reads, the registers it set up.
    ///
    /// Each incoming register is first copied into a fresh vreg (so the physical
    /// argument registers become dead immediately and the address-computation
    /// temporaries may reuse them), then stored into the save area.
    fn spill_va_regs(&self, lo: &mut Lower<'_, Self>) {
        let cc = &self.rf.cc;
        // Capture the incoming registers while they are still live.
        let gpr_vs: Vec<VReg> = (0..6)
            .map(|i| {
                let v = lo.fresh_vreg(RegClass::Gpr);
                lo.emit(MachineInst::new(
                    X86Op::MovRR.opcode(),
                    vec![def_v(v), use_p(cc.arg_regs[i])],
                ));
                v
            })
            .collect();
        let xmm_vs: Vec<VReg> = (0..8)
            .map(|i| {
                let v = lo.fresh_vreg(RegClass::Fp);
                lo.emit(MachineInst::new(
                    X86Op::MovRR.opcode(),
                    vec![def_v(v), use_p(regs::xmm(i as u16))],
                ));
                v
            })
            .collect();

        // Reserve the save area and store the captured registers into it.
        let save = lo.new_slot(176, 16);
        lo.set_va_reg_save(save);
        let base = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(self.frame_addr(base, save));
        for (i, v) in gpr_vs.into_iter().enumerate() {
            let dp = self.add_off(lo, base, 8 * i as u64);
            lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(dp), use_v(v), imm(8)]));
        }
        for (i, v) in xmm_vs.into_iter().enumerate() {
            let dp = self.add_off(lo, base, 48 + 16 * i as u64);
            lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(dp), use_v(v), imm(8)]));
        }
    }
}

impl MachineTarget for X86_64Target {
    fn name(&self) -> &str {
        "x86_64"
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
            X86Op::decode(op),
            X86Op::Jmp
                | X86Op::BrCond
                | X86Op::CmpBr
                | X86Op::CmpBrI
                | X86Op::Switch
                | X86Op::Switch128
                | X86Op::Ret
                | X86Op::Unreachable
        )
    }

    /// The copies and the two-address operations whose encoding is correct
    /// with the destination equal to the first source (`mov d, a` is skipped
    /// then): the integer and SSE ALU operations, shifts, extensions, loads,
    /// and compares (which read their sources before writing `d`).
    fn tied_use(&self, inst: &MachineInst) -> Option<usize> {
        use X86Op::*;
        match X86Op::decode(inst.opcode) {
            // Copies between vregs (of one class: their registers are equal
            // only then).
            MovRR => {
                let virt = |o: &MachineOperand| matches!(o.reg(), Some(Reg::Virtual(_)));
                (virt(&inst.operands[0]) && virt(&inst.operands[1])).then_some(1)
            }
            Add | Sub | And | Or | Xor | Imul | ShlI | ShrI | SarI | ShlCl | ShrCl | SarCl | AluRI | ImulRI
            | Movsx | Movzx | Load | SetccCmp | SetccCmpI | FAdd | FSub | FMul | FDiv => Some(1),
            _ => None,
        }
    }

    fn is_move(&self, op: Opcode) -> bool {
        X86Op::decode(op) == X86Op::MovRR
    }

    fn emit_move(&self, dst: Reg, src: Reg) -> MachineInst {
        MachineInst::new(X86Op::MovRR.opcode(), vec![MachineOperand::Def(dst), MachineOperand::Use(src)])
    }

    fn emit_spill(&self, slot: StackSlot, src: PReg) -> MachineInst {
        MachineInst::new(X86Op::StoreFrame.opcode(), vec![use_p(src), MachineOperand::Frame(slot)])
    }

    /// An xmm register may hold a whole 128-bit vector, so its spill slot is
    /// 16 bytes (spilled with `movdqu`).
    fn spill_slot(&self, class: RegClass) -> (u64, u64) {
        match class {
            RegClass::Gpr => (8, 8),
            RegClass::Fp => (16, 8),
        }
    }

    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(X86Op::LoadFrame.opcode(), vec![def(dst), MachineOperand::Frame(slot)])
    }
}

impl TargetIsel for X86_64Target {
    /// A constant wider than 64 bits keeps its low 64 bits: part 0 of a wide
    /// value (the `wide` submodule materializes all of its parts).
    fn li(&self, dst: VReg, value: Int) -> MachineInst {
        let value = if value.to_u64().is_some() || value.to_i64().is_some() { value } else { value.mod_2k(64) };
        MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(dst), MachineOperand::Imm(value)])
    }

    fn jump(&self, dst: MBlockId) -> MachineInst {
        MachineInst::new(X86Op::Jmp.opcode(), vec![MachineOperand::Label(dst)])
    }

    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(X86Op::LeaFrame.opcode(), vec![def_v(dst), MachineOperand::Frame(slot)])
    }

    /// A thread-local global's address goes through the thread pointer, by the
    /// access model [`crate::codegen::linkage::tls_model`] picks for this
    /// target's relocation model: local-exec and initial-exec are one
    /// [`X86Op::TlsAddr`], general-dynamic a [`X86Op::TlsGd`] call whose
    /// result is copied out of `rax`. Nothing here depends on data, so TLS
    /// addressing is constant-time.
    fn lower_global_addr(&self, lo: &mut Lower<'_, Self>, dst: VReg, g: u32) {
        let gid = crate::ir::GlobalId::from_index(g as usize);
        if !lo.module().global_attrs(gid).thread_local {
            lo.emit(self.global_addr(dst, g));
            return;
        }
        if self.win64 {
            panic!("x86-64 backend: thread-local storage (global #{g}) is not supported on Windows");
        }
        match crate::codegen::linkage::tls_model(lo.module(), gid, self.reloc_model) {
            model @ (TlsModel::LocalExec | TlsModel::InitialExec) => {
                let ie = u64::from(model == TlsModel::InitialExec);
                lo.emit(MachineInst::new(
                    X86Op::TlsAddr.opcode(),
                    vec![def_v(dst), MachineOperand::Global(g), imm(ie)],
                ));
            }
            TlsModel::GeneralDynamic => {
                let rax = self.rf.cc.ret_reg;
                let mut operands = vec![MachineOperand::Global(g), def(rax)];
                for &cs in &self.rf.caller_saved {
                    if cs != rax {
                        operands.push(def(cs));
                    }
                }
                lo.emit(MachineInst::new(X86Op::TlsGd.opcode(), operands));
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(dst), use_p(rax)]));
            }
        }
    }

    fn global_addr(&self, dst: VReg, g: u32) -> MachineInst {
        MachineInst::new(X86Op::GlobalAddr.opcode(), vec![def_v(dst), MachineOperand::Global(g)])
    }

    fn float_const(&self, dst: VReg, bits: u64, width: u32) -> MachineInst {
        MachineInst::new(
            X86Op::LoadFConst.opcode(),
            vec![def_v(dst), imm(bits), imm(u64::from(width))],
        )
    }

    fn vector_const(&self, dst: VReg, types: &TypeContext, consts: &crate::ir::ConstPool, c: &Const) -> MachineInst {
        let (lo64, hi64) = crate::codegen::simd128::const_bits(types, consts, c);
        MachineInst::new(X86Op::LoadVConst.opcode(), vec![def_v(dst), imm(lo64), imm(hi64)])
    }

    fn lower_prologue(&self, lo: &mut Lower<'_, Self>) {
        if self.win64 {
            let sig = lo.func().sig;
            let wide = match lo.types().get(sig) {
                Type::Func(ft) => ft.params.iter().chain([&ft.ret]).any(|&t| Self::wide_ty(lo, t).is_some()),
                _ => false,
            };
            if wide {
                panic!("x86-64 backend: 128-bit integer parameters and results are not supported under the Microsoft x64 convention");
            }
            self.lower_prologue_win64(lo);
        } else {
            self.lower_prologue_x86(lo);
        }
    }

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        // Inline asm places its own operands, whatever their types.
        match &inst.kind {
            InstKind::InlineAsm(asm) => return self.lower_inline_asm(lo, inst, asm),
            InstKind::AsmOutput(n) => return self.lower_asm_output(lo, inst, *n),
            _ => {}
        }
        // What the wide-integer legalization left of integers wider than 64
        // bits (see the `wide` submodule).
        if self.lower_wide(lo, inst) {
            return;
        }
        // SSE2 vector code (legalized for `Sse2Legality` beforehand).
        if self.lower_vector(lo, inst) {
            return;
        }
        match &inst.kind {
            InstKind::Bin(op) => self.lower_bin(lo, *op, inst),
            InstKind::ICmp(pred) => {
                // A compare fused into its branch is lowered there.
                if let Some(r) = inst.result()
                    && self.fused_compare(lo, r).is_some()
                {
                    return;
                }
                let d = lo.result_reg(inst);
                let (mut ops, is_imm) = self.compare(lo, *pred, inst.operands()[0], inst.operands()[1]);
                ops.insert(0, def_v(d));
                let op = if is_imm { X86Op::SetccCmpI } else { X86Op::SetccCmp };
                lo.emit(MachineInst::new(op.opcode(), ops));
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
                // Runtime-sized stack allocation: move `rsp` down by the (rounded)
                // byte count and hand back a pointer into the fresh region. The
                // encoder does the rsp arithmetic and the outgoing-area relocation
                // (see [`X86Op::DynAlloca`]); here we just wire the count operand
                // and the required alignment.
                let d = lo.result_reg(inst);
                let n = self.oper(lo, inst.operands()[0]);
                lo.emit(MachineInst::new(
                    X86Op::DynAlloca.opcode(),
                    vec![def_v(d), use_v(n), imm(u64::from(*align))],
                ));
            }
            // A volatile access is the same single `mov` at exactly the accessed
            // width: nothing below isel merges, splits, or removes memory ops.
            InstKind::Load { ty, .. } => {
                let d = lo.result_reg(inst);
                let ptr = self.oper(lo, inst.operands()[0]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    X86Op::Load.opcode(),
                    vec![def_v(d), use_v(ptr), imm(size)],
                ));
            }
            InstKind::Store { ty, .. } => {
                let ptr = self.oper(lo, inst.operands()[0]);
                let val = self.oper(lo, inst.operands()[1]);
                let size = lo.byte_size(*ty);
                lo.emit(MachineInst::new(
                    X86Op::Store.opcode(),
                    vec![use_v(ptr), use_v(val), imm(size)],
                ));
            }
            InstKind::PtrAdd { .. }
                if Self::int_const(lo, inst.operands()[1])
                    .and_then(|c| imm_at(&c, lo.int_width(inst.operands()[1])))
                    .is_some() =>
            {
                // A constant offset: `add d, imm` (or `lea`), the offset
                // sign-extended from its width.
                let d = lo.result_reg(inst);
                let base = self.oper(lo, inst.operands()[0]);
                let off_w = lo.int_width(inst.operands()[1]);
                let k = Self::int_const(lo, inst.operands()[1]).and_then(|c| imm_at(&c, off_w)).unwrap_or(0);
                lo.emit(MachineInst::new(
                    X86Op::AluRI.opcode(),
                    vec![def_v(d), use_v(base), MachineOperand::Imm(Int::from_i64(k)), imm(0), imm(64)],
                ));
            }
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(inst);
                let base = self.oper(lo, inst.operands()[0]);
                // The offset is a signed byte count of any integer width; a
                // narrow one must be sign-extended before the 64-bit add (an
                // i32 -4 may sit in its register as 0x0000_0000_FFFF_FFFC).
                let off_v = inst.operands()[1];
                let off_w = lo.int_width(off_v);
                let off = self.oper(lo, off_v);
                let off = if off_w >= 64 {
                    off
                } else {
                    let x = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(
                        X86Op::Movsx.opcode(),
                        vec![def_v(x), use_v(off), imm(u64::from(off_w)), imm(64)],
                    ));
                    x
                };
                lo.emit(MachineInst::new(
                    X86Op::Add.opcode(),
                    vec![def_v(d), use_v(base), use_v(off), imm(64)],
                ));
            }
            // A float select blends xmm registers (a GPR cmov cannot).
            InstKind::Select if lo.mf().vreg_class(lo.result_reg(inst)) == RegClass::Fp => {
                let ops = inst.operands().to_vec();
                self.vec_select(lo, inst, &ops);
            }
            InstKind::Select => {
                let d = lo.result_reg(inst);
                let c = self.clean_cond(lo, inst.operands()[0]);
                let t = self.oper(lo, inst.operands()[1]);
                let f = self.oper(lo, inst.operands()[2]);
                // d = f; test c,c; cmovne d, t   (cond != 0 -> t)
                // Branchless, so a secret condition is constant-time (§6d).
                debug_assert!(
                    [X86Op::MovRR, X86Op::Test, X86Op::Cmovne]
                        .iter()
                        .all(|op| !op.may_branch_on_data(&[]))
                );
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(f)]));
                lo.emit(MachineInst::new(X86Op::Test.opcode(), vec![use_v(c)]));
                lo.emit(MachineInst::new(
                    X86Op::Cmovne.opcode(),
                    vec![def_v(d), use_v(d), use_v(t)],
                ));
            }
            InstKind::Freeze | InstKind::Declassify => {
                let d = lo.result_reg(inst);
                let s = self.oper(lo, inst.operands()[0]);
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
            }
            InstKind::AtomicLoad { .. }
            | InstKind::AtomicStore { .. }
            | InstKind::AtomicRmw { .. }
            | InstKind::CmpXchg { .. }
            | InstKind::Fence(_) => self.lower_atomic(lo, inst),
            InstKind::Call => self.lower_call(lo, inst),
            InstKind::Syscall => self.lower_syscall(lo, inst),
            InstKind::Unary(UnaryOp::FNeg) => self.lower_fneg(lo, inst),
            InstKind::FCmp(pred) => self.lower_fcmp(lo, *pred, inst),
            _ => unreachable!("terminator reached lower_inst: {:?}", inst.kind),
        }
    }

    fn lower_term(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Ret if self.win64 => self.lower_ret_win64(lo, inst),
            InstKind::Ret => {
                let cc = &self.rf.cc;
                let ret_ty = match lo.types().get(lo.func().sig) {
                    Type::Func(ft) => ft.ret,
                    _ => lo.func().sig,
                };
                if is_aggregate(lo.types(), ret_ty) {
                    // The return operand is a pointer to the struct's storage.
                    let src = self.oper(lo, inst.operands()[0]);
                    match classify_aggregate(lo.types(), ret_ty) {
                        AbiClass::Memory => {
                            // Copy the struct through the hidden `sret` pointer
                            // (saved to the aux slot in the prologue) and return it.
                            let size = lo.byte_size(ret_ty);
                            let slot = lo.aux_slot().expect("sret pointer saved by the prologue");
                            let dst = lo.fresh_vreg(RegClass::Gpr);
                            lo.emit(MachineInst::new(
                                X86Op::LoadFrame.opcode(),
                                vec![def_v(dst), MachineOperand::Frame(slot)],
                            ));
                            self.emit_memcpy(lo, dst, src, size);
                            lo.emit(MachineInst::new(
                                X86Op::MovRR.opcode(),
                                vec![def(cc.ret_reg), use_v(dst)],
                            ));
                        }
                        AbiClass::Regs(ebs) => {
                            // Place each eightbyte in rax/rdx (INTEGER) or
                            // xmm0/xmm1 (SSE). Compute all source pointers first so
                            // the loads into the return registers are consecutive.
                            let ptrs: Vec<VReg> = (0..ebs.len())
                                .map(|k| self.add_off(lo, src, 8 * k as u64))
                                .collect();
                            let mut ic = 0usize;
                            let mut sc = 0usize;
                            for (k, c) in ebs.iter().enumerate() {
                                let r = match c {
                                    Eightbyte::Integer => {
                                        let r = if ic == 0 { cc.ret_reg } else { regs::gpr(regs::RDX) };
                                        ic += 1;
                                        r
                                    }
                                    Eightbyte::Sse => {
                                        let r = if sc == 0 { cc.fp_ret_reg } else { regs::xmm(1) };
                                        sc += 1;
                                        r
                                    }
                                };
                                lo.emit(MachineInst::new(
                                    X86Op::Load.opcode(),
                                    vec![def(r), use_v(ptrs[k]), imm(8)],
                                ));
                            }
                        }
                    }
                } else if let Some(&v) = inst.operands().first().filter(|&&v| Self::wide_val(lo, v).is_some()) {
                    // An `i128` in rax:rdx.
                    assert_eq!(Self::wide_val(lo, v), Some(2), "x86-64 backend: integers wider than 128 bits cannot be returned");
                    let p = self.parts(lo, v);
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(cc.ret_reg), use_v(p[0])]));
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(regs::gpr(regs::RDX)), use_v(p[1])]));
                } else if let Some(&v) = inst.operands().first() {
                    let r = self.oper(lo, v);
                    // A float return goes in xmm0, an integer/pointer return in rax.
                    let ret = match lo.mf().vreg_class(r) {
                        RegClass::Fp => cc.fp_ret_reg,
                        RegClass::Gpr => cc.ret_reg,
                    };
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(ret), use_v(r)]));
                }
                lo.emit(MachineInst::new(X86Op::Ret.opcode(), Vec::new()));
            }
            InstKind::Br(target) => {
                let args: Vec<_> = inst.operands().to_vec();
                let e = lo.edge_to(*target, &args);
                lo.emit(self.jump(e));
            }
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                // A compare used only here becomes `cmp; jcc`; any other
                // condition is tested: `test cond, cond; jne`.
                let (mut operands, op) = match self.fused_compare(lo, inst.operands()[0]) {
                    Some((pred, x, y)) => {
                        let (ops, is_imm) = self.compare(lo, pred, x, y);
                        (ops, if is_imm { X86Op::CmpBrI } else { X86Op::CmpBr })
                    }
                    None => (vec![use_v(self.clean_cond(lo, inst.operands()[0]))], X86Op::BrCond),
                };
                let ops = inst.operands();
                let tb = 1 + *true_args as usize;
                let fb = tb + *false_args as usize;
                let true_vals: Vec<_> = ops[1..tb].to_vec();
                let false_vals: Vec<_> = ops[tb..fb].to_vec();
                let te = lo.edge_to(*if_true, &true_vals);
                let fe = lo.edge_to(*if_false, &false_vals);
                operands.push(MachineOperand::Label(te));
                operands.push(MachineOperand::Label(fe));
                lo.emit(MachineInst::new(op.opcode(), operands));
            }
            InstKind::Switch(data) if Self::wide_val(lo, inst.operands()[0]).is_some() => {
                // A 128-bit scrutinee: both halves compared per case.
                let p = self.parts(lo, inst.operands()[0]);
                assert_eq!(p.len(), 2, "x86-64 backend: a switch wider than 128 bits");
                let ops = inst.operands();
                let mut idx = 1usize;
                let dcount = data.default_args as usize;
                let default_vals: Vec<_> = ops[idx..idx + dcount].to_vec();
                idx += dcount;
                let de = lo.edge_to(data.default, &default_vals);
                let mut operands = vec![use_v(p[0]), use_v(p[1]), MachineOperand::Label(de)];
                for case in &data.cases.clone() {
                    let n = case.args as usize;
                    let cvals: Vec<_> = ops[idx..idx + n].to_vec();
                    idx += n;
                    let ce = lo.edge_to(case.target, &cvals);
                    let v = case.value.mod_2k(128);
                    operands.push(imm(v.mod_2k(64).to_u64().unwrap_or(0)));
                    operands.push(imm(v.div_2k_trunc(64).to_u64().unwrap_or(0)));
                    operands.push(MachineOperand::Label(ce));
                }
                lo.emit(MachineInst::new(X86Op::Switch128.opcode(), operands));
            }
            InstKind::Switch(data) => {
                // Cases are compared as 64-bit values: sign-extend the scrutinee
                // (a 32-bit result's upper half is zero, not its sign) and each
                // case value from the scrutinee's width.
                let width = lo.int_width(inst.operands()[0]);
                let cond = if width >= 64 {
                    self.oper(lo, inst.operands()[0])
                } else {
                    let r = self.oper(lo, inst.operands()[0]);
                    let d = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(
                        X86Op::Movsx.opcode(),
                        vec![def_v(d), use_v(r), imm(u64::from(width)), imm(64)],
                    ));
                    d
                };
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
                lo.emit(MachineInst::new(X86Op::Switch.opcode(), operands));
            }
            InstKind::Unreachable => {
                lo.emit(MachineInst::new(X86Op::Unreachable.opcode(), Vec::new()));
            }
            _ => unreachable!("non-terminator reached lower_term: {:?}", inst.kind),
        }
    }
}
