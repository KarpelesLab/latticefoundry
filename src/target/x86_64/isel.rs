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
//! into it ([`X86_64Target::spill_va_regs`]): the 6 integer regs
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

use puremp::Int;

use super::regs::{self, RegFile};

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
    /// `[Def d, Use d, Use t]` — `cmovne d, t` (move if ZF=0).
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
}

impl X86Op {
    /// The MIR [`Opcode`] id for this opcode.
    #[inline]
    pub fn opcode(self) -> Opcode {
        Opcode(self as u32)
    }

    /// Decode a MIR [`Opcode`] back to an [`X86Op`].
    pub fn decode(op: Opcode) -> X86Op {
        use X86Op::*;
        const TABLE: [X86Op; 54] = [
            MovRR, MovRI, Add, Sub, And, Or, Xor, Imul, ShlI, ShrI, SarI, ShlCl, ShrCl, SarCl, Cqo,
            ZeroRdx, Idiv, Div, SetccCmp, Test, Cmovne, Load, Store, LeaFrame, GlobalAddr, Call,
            Ret, Jmp, BrCond, Switch, Unreachable, Push, Pop, MovRbpRsp, SubRsp, LeaRspRbp,
            StoreFrame, LoadFrame, FAdd, FSub, FMul, FDiv, FXor, LoadFConst, FCmpSet, Cvtsd2ss,
            Cvtss2sd, CvtF2si, CvtSi2f, FuncAddr, Movsx, Movzx, LeaRbpOff, LeaRspOff,
        ];
        TABLE[op.0 as usize]
    }
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
        Type::Int(_) | Type::Ptr | Type::Func(_) => Some(Eightbyte::Integer),
        Type::Float(_) => Some(Eightbyte::Sse),
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
}

impl Default for X86_64Target {
    fn default() -> Self {
        Self::new()
    }
}

impl X86_64Target {
    /// Construct the x86-64 target with its fixed register file and SysV ABI.
    pub fn new() -> X86_64Target {
        X86_64Target { rf: RegFile::new() }
    }

    /// Lower function `func` of `module` to MIR over this target.
    pub fn select(&self, module: &Module, func: crate::ir::FuncId) -> crate::codegen::mir::MachineFunction {
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

    fn lower_bin(&self, lo: &mut Lower<'_, Self>, op: BinOp, inst: &InstData) {
        let d = lo.result_reg(inst);
        let width = lo.int_width(inst.operands()[0]);
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
            BinOp::UDiv => self.lower_div(lo, X86Op::Div, false, false, d, inst, width),
            BinOp::URem => self.lower_div(lo, X86Op::Div, false, true, d, inst, width),
            BinOp::SDiv => self.lower_div(lo, X86Op::Idiv, true, false, d, inst, width),
            BinOp::SRem => self.lower_div(lo, X86Op::Idiv, true, true, d, inst, width),
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
        let a = self.oper(lo, inst.operands()[0]);
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
        width: u32,
    ) {
        let a = self.oper(lo, inst.operands()[0]);
        let b = self.oper(lo, inst.operands()[1]);
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
                // bit0 = 64-bit gpr source.
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
                let flags: u64 = if src_w > 32 { 0b100 } else { 0b10 };
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
        let k = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(k), imm(off)]));
        let d = lo.fresh_vreg(RegClass::Gpr);
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
        let cc = &self.rf.cc;
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];

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
                    let sz = lo.byte_size(ty);
                    let dp = self.lea_rsp(lo, stack_off);
                    lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(dp), use_v(v), imm(sz)]));
                    stack_off += 8;
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
        let ret_is_fp = ret_ty.is_some_and(|t| lo.types().get(t).is_float());
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
            None => {
                if inst.result().is_some() {
                    let d = lo.result_reg(inst);
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_p(ret_reg)]));
                }
            }
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

        let mut stack_in = 16u64; // first incoming stack arg, above the return address
        for (i, &pv) in param_vregs.iter().enumerate() {
            let ty = sig_params[i];
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
                    let sz = lo.byte_size(ty);
                    let p = self.lea_rbp(lo, stack_in);
                    lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(pv), use_v(p), imm(sz)]));
                    stack_in += 8;
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
            X86Op::Jmp | X86Op::BrCond | X86Op::Switch | X86Op::Ret | X86Op::Unreachable
        )
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

    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(X86Op::LoadFrame.opcode(), vec![def(dst), MachineOperand::Frame(slot)])
    }
}

impl TargetIsel for X86_64Target {
    fn li(&self, dst: VReg, value: Int) -> MachineInst {
        MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(dst), MachineOperand::Imm(value)])
    }

    fn jump(&self, dst: MBlockId) -> MachineInst {
        MachineInst::new(X86Op::Jmp.opcode(), vec![MachineOperand::Label(dst)])
    }

    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(X86Op::LeaFrame.opcode(), vec![def_v(dst), MachineOperand::Frame(slot)])
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

    fn lower_prologue(&self, lo: &mut Lower<'_, Self>) {
        self.lower_prologue_x86(lo);
    }

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Bin(op) => self.lower_bin(lo, *op, inst),
            InstKind::ICmp(pred) => {
                let d = lo.result_reg(inst);
                let a = self.oper(lo, inst.operands()[0]);
                let b = self.oper(lo, inst.operands()[1]);
                let width = lo.int_width(inst.operands()[0]);
                lo.emit(MachineInst::new(
                    X86Op::SetccCmp.opcode(),
                    vec![
                        def_v(d),
                        use_v(a),
                        use_v(b),
                        imm(u64::from(cc_code(*pred))),
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
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(inst);
                let base = self.oper(lo, inst.operands()[0]);
                let off = self.oper(lo, inst.operands()[1]);
                lo.emit(MachineInst::new(
                    X86Op::Add.opcode(),
                    vec![def_v(d), use_v(base), use_v(off), imm(64)],
                ));
            }
            InstKind::Select => {
                let d = lo.result_reg(inst);
                let c = self.oper(lo, inst.operands()[0]);
                let t = self.oper(lo, inst.operands()[1]);
                let f = self.oper(lo, inst.operands()[2]);
                // d = f; test c,c; cmovne d, t   (cond != 0 -> t)
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(f)]));
                lo.emit(MachineInst::new(X86Op::Test.opcode(), vec![use_v(c)]));
                lo.emit(MachineInst::new(
                    X86Op::Cmovne.opcode(),
                    vec![def_v(d), use_v(d), use_v(t)],
                ));
            }
            InstKind::Freeze => {
                let d = lo.result_reg(inst);
                let s = self.oper(lo, inst.operands()[0]);
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
            }
            InstKind::Call => self.lower_call(lo, inst),
            InstKind::Unary(UnaryOp::FNeg) => self.lower_fneg(lo, inst),
            InstKind::FCmp(pred) => self.lower_fcmp(lo, *pred, inst),
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
                let cond = self.oper(lo, inst.operands()[0]);
                let ops = inst.operands();
                let tb = 1 + *true_args as usize;
                let fb = tb + *false_args as usize;
                let true_vals: Vec<_> = ops[1..tb].to_vec();
                let false_vals: Vec<_> = ops[tb..fb].to_vec();
                let te = lo.edge_to(*if_true, &true_vals);
                let fe = lo.edge_to(*if_false, &false_vals);
                lo.emit(MachineInst::new(
                    X86Op::BrCond.opcode(),
                    vec![use_v(cond), MachineOperand::Label(te), MachineOperand::Label(fe)],
                ));
            }
            InstKind::Switch(data) => {
                let cond = self.oper(lo, inst.operands()[0]);
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
                    operands.push(MachineOperand::Imm(case.value.clone()));
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
