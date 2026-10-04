//! The Thumb-2 machine opcode set ([`ThOp`]) and the instruction-selection
//! rules, under the AAPCS base (soft-float) procedure call standard.
//!
//! [`ThOp`] is this target's [`Opcode`] vocabulary: a *post-isel,
//! pre-encoding* MIR whose operands are still MIR [`MachineOperand`]s. Every
//! data-processing op is three-address (`[Def d, Use a, Use b]`); the encoder
//! ([`super::encode`]) picks the 16-bit form when the registers and immediate
//! allow it and the 32-bit Thumb-2 form otherwise. A few ops expand to short
//! idioms at encode time: a comparison becomes `cmp` plus an `ite` block of two
//! `mov`s, a `select` a `tst` plus an `ite` block, a remainder `sdiv`/`udiv`
//! plus `mls`, a 32-bit constant `movw`/`movt`.
//!
//! ## What reaches isel
//!
//! Instruction selection runs on a module that [`super::prepare_module`] has
//! already rewritten: floating-point values are integers of the same width
//! and every floating-point operation a call to an AEABI helper
//! ([`super::softfloat`]); every integer wider than 32 bits has been split
//! into 32-bit parts ([`crate::codegen::legalize_int`], `W = 32`), except at
//! the ABI boundary and in the fixed split/join shapes the legalizer leaves.
//! Those remaining wide values (an `i64` parameter, argument, result or
//! return, a `zext`/`shl`/`or` join, an `lshr`/`trunc` split, a volatile
//! 64-bit access, a 64-bit `switch`) live in a **register group** here: part 0
//! is the value's own vreg and the higher parts are vregs this target keeps in
//! a side table (`ThumbTarget::parts`), created on first reference so a
//! definition and its uses agree whatever order the blocks are lowered in.
//!
//! ## Narrow values
//!
//! Registers are 32 bits and a narrow value (`i1`, `i8`, `i16`, an odd width)
//! keeps whatever its computation left above its width: an `i8` add of 200 +
//! 100 leaves 300 in the register, and a `trunc` is a plain move. Ops that
//! only feed the low bits (`add`, `sub`, `mul`, logic, left shifts, stores)
//! don't care; every op whose result depends on the upper bits extends first
//! (`ThumbTarget::ext`: `uxtb`/`uxth`/`sxtb`/`sxth`, `ubfx`/`sbfx`): compares,
//! right shifts, division and remainder, `zext`/`sext`, a `switch` scrutinee,
//! a narrow `ptr_add` offset, and a shift amount narrower than the 8 bits the
//! shifter reads. A branch or `select` condition is tested with `tst c, #1`,
//! so a dirty `i1` needs no extension. Results of compares and of narrow loads
//! (`ldrb`/`ldrh` zero-extend) are known clean.
//!
//! ## The AAPCS (base standard)
//!
//! Arguments are assigned to `r0`–`r3` and then the stack by the AAPCS rules
//! (stage C): a 64-bit value (an `i64`, a `double` after the soft-float
//! lowering) is doubleword aligned — it starts at an even register, or at an
//! 8-aligned stack offset — and a composite (a by-value struct, whose SSA
//! value is the address of its storage) is copied word by word into the next
//! registers, split between `r3` and the stack when it straddles them and no
//! argument went to the stack yet. Results: a word in `r0`, a 64-bit value in
//! `r0:r1`, a composite of at most 4 bytes in `r0`, and a larger composite in
//! memory whose address the caller passes as a hidden first argument in `r0`.
//! `r4`–`r11` are preserved; the stack is 8-byte aligned at every call.
//!
//! ## Division
//!
//! ARMv7-M has `sdiv`/`udiv`, so 32-bit division is inline and a remainder is
//! `sdiv`+`mls`. With [`ThumbTarget::with_hw_div`] off (ARMv6-M-style cores),
//! they call the AEABI helpers `__aeabi_idiv` / `__aeabi_uidiv` and
//! `__aeabi_idivmod` / `__aeabi_uidivmod` (remainder in `r1`). 64-bit division
//! is always a call: `__aeabi_ldivmod` / `__aeabi_uldivmod` return the quotient
//! in `r0:r1` and the remainder in `r2:r3`, so the legalizer's remainder
//! libcall names ([`LMOD_PSEUDO`], [`ULMOD_PSEUDO`]) are placeholders that this
//! isel turns into the same helper, taking `r2:r3`.
//!
//! Deferred (and rejected with a clear panic at their sites): `dyn_alloca`,
//! atomic read-modify-write and compare-exchange (ARMv7-M's `ldrex`/`strex`
//! loops), and atomics wider than 32 bits (there is no `ldrexd` on M-profile).

use std::cell::RefCell;

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::mir::{
    MBlockId, MachineInst, MachineOperand, Opcode, PReg, Reg, RegClass, StackSlot, VReg,
};
use crate::codegen::target::{CallConv, MachineTarget};
use crate::ir::inst::{BinOp, CastOp, InstKind, IntPred};
use crate::ir::types::{Type, TypeId};
use crate::ir::value::{Const, ValueDef};
use crate::ir::{InstData, Module, ValueId};
use crate::support::{DetHashMap, StrInterner};

use puremp::Int;

use super::regs::{self, RegFile, gpr};

/// The legalizer's name for a 64-bit signed remainder helper on this target: a
/// placeholder the isel lowers to a call of `__aeabi_ldivmod`, whose remainder
/// comes back in `r2:r3`.
pub const LMOD_PSEUDO: &str = "__lf_thumb_lmod";
/// The unsigned counterpart of [`LMOD_PSEUDO`] (`__aeabi_uldivmod`).
pub const ULMOD_PSEUDO: &str = "__lf_thumb_ulmod";

/// The Thumb MIR opcode vocabulary. Operand layouts are documented per
/// variant; `Def`/`Use` are register operands, the rest immediates, frame
/// slots, labels or symbol references. Every register holds 32 bits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum ThOp {
    /// `[Def d, Use s]` — `mov d, s`.
    Mov = 0,
    /// `[Def d, Imm v]` — load the low 32 bits of `v` (`movs`/`mov.w`/`mvn`/
    /// `movw`/`movw`+`movt`).
    MovImm = 1,
    /// `[Def d, Use a, Use b]` — `add d, a, b`.
    Add = 2,
    /// `[Def d, Use a, Use b]` — `sub d, a, b`.
    Sub = 3,
    /// `[Def d, Use a, Use b]` — `and d, a, b`.
    And = 4,
    /// `[Def d, Use a, Use b]` — `orr d, a, b`.
    Orr = 5,
    /// `[Def d, Use a, Use b]` — `eor d, a, b`.
    Eor = 6,
    /// `[Def d, Use a, Use b]` — `mul d, a, b`.
    Mul = 7,
    /// `[Def d, Use a, Use b]` — `sdiv d, a, b`.
    Sdiv = 8,
    /// `[Def d, Use a, Use b]` — `udiv d, a, b`.
    Udiv = 9,
    /// `[Def d, Use a, Use b]` — `sdiv ip, a, b; mls d, ip, b, a`.
    Srem = 10,
    /// `[Def d, Use a, Use b]` — `udiv ip, a, b; mls d, ip, b, a`.
    Urem = 11,
    /// `[Def d, Use a, Use b]` — `lsl d, a, b` (the shifter reads `b`'s low byte).
    Lsl = 12,
    /// `[Def d, Use a, Use b]` — `lsr d, a, b`.
    Lsr = 13,
    /// `[Def d, Use a, Use b]` — `asr d, a, b`.
    Asr = 14,
    /// `[Def d, Use a, Imm k]` — `d = a + k` (`k` a signed 32-bit value; the
    /// encoder picks `adds`/`subs`/`add.w`/`addw`/`subw`, or goes through `ip`).
    AddImm = 15,
    /// `[Def d, Use a, Imm k]` — `and d, a, #k` (or `bic`, or through `ip`).
    AndImm = 16,
    /// `[Def d, Use a, Imm k]` — `orr d, a, #k` (or `orn`, or through `ip`).
    OrrImm = 17,
    /// `[Def d, Use a, Imm k]` — `eor d, a, #k` (`mvn` for all ones).
    EorImm = 18,
    /// `[Def d, Use a, Imm k]` — `rsb d, a, #k` (`d = k - a`).
    RsbImm = 19,
    /// `[Def d, Use a, Imm n]` — `lsl d, a, #n` (`n` in 0..=31).
    LslImm = 20,
    /// `[Def d, Use a, Imm n]` — `lsr d, a, #n` (`n` in 1..=31).
    LsrImm = 21,
    /// `[Def d, Use a, Imm n]` — `asr d, a, #n` (`n` in 1..=31).
    AsrImm = 22,
    /// `[Def d, Use s, Imm width, Imm signed]` — extend the low `width` bits
    /// (1..=31) of `s`: `uxtb`/`uxth`/`sxtb`/`sxth` for 8 and 16, `ubfx`/`sbfx`
    /// otherwise.
    Ext = 23,
    /// `[Def d, Use a, Use b, Imm cond]` — `cmp a, b; ite cond; mov<cond> d, #1;
    /// mov<!cond> d, #0` (`cond` an Arm condition code, `EQ` = 0 … `LE` = 13).
    SetCmp = 24,
    /// `[Def d, Use a, Imm k, Imm cond]` — as [`ThOp::SetCmp`] against the
    /// constant `k` (`cmp`/`cmn` immediate, or through `ip`).
    SetCmpImm = 25,
    /// `[Def d, Use c, Use t, Use f]` — `tst c, #1; ite ne; movne d, t;
    /// moveq d, f` (one arm dropped when `d` already holds it).
    Select = 26,
    /// `[Def d, Use base, Imm off, Imm size]` — `ldr`/`ldrh`/`ldrb d, [base,
    /// #off]` (zero-extending).
    Load = 27,
    /// `[Use base, Use val, Imm off, Imm size]` — `str`/`strh`/`strb`.
    Store = 28,
    /// `[Def lo, Def hi, Use base]` — `ldrd lo, hi, [base]` (a volatile 64-bit
    /// load: one access).
    LoadDual = 29,
    /// `[Use base, Use lo, Use hi]` — `strd lo, hi, [base]`.
    StoreDual = 30,
    /// `[Def d, Frame slot]` — `add d, sp, #slot_off`.
    FrameAddr = 31,
    /// `[Def d, Frame slot]` — reload: `ldr d, [sp, #slot_off]`.
    LoadFrame = 32,
    /// `[Use s, Frame slot]` — spill: `str s, [sp, #slot_off]`.
    StoreFrame = 33,
    /// `[Def d, Imm off]` — `add d, sp, #off`: an address in the outgoing
    /// argument area at the bottom of the frame.
    SpAddr = 34,
    /// `[Use v, Imm off, Imm size]` — `str v, [sp, #off]`: an outgoing stack
    /// argument word.
    StoreSp = 35,
    /// `[Def d, Imm off]` — the address of the incoming stack argument at
    /// byte `off` (`sp + frame size + off`, resolved from the frame layout).
    IncAddr = 36,
    /// `[Def d, Imm off, Imm size]` — load an incoming stack argument.
    LoadInc = 37,
    /// `[Def d, Global g]` — `movw d, #:lower16:g; movt d, #:upper16:g`.
    GlobalAddr = 38,
    /// `[Def d, Func f]` — the address of function `f` (with the Thumb bit),
    /// `movw`/`movt` against its symbol.
    FuncAddr = 39,
    /// `[Func f | Use callee, Def clobbers.., Use args..]` — `bl f` / `blx callee`.
    Call = 40,
    /// `[Use r0.., ]` — return (the epilogue's `pop {.., pc}` precedes it).
    Ret = 41,
    /// `[Label t]` — `b t`.
    B = 42,
    /// `[Use c, Label t, Label f]` — `tst c, #1; bne t; b f`.
    BrCond = 43,
    /// `[Use c, Label default, (Imm v, Label case)...]` — compare-and-branch
    /// chain on a 32-bit scrutinee.
    Switch = 44,
    /// `[Use lo, Use hi, Label default, (Imm v, Label case)...]` — the same on
    /// a 64-bit scrutinee held in two registers.
    Switch64 = 45,
    /// `[]` — `udf #0` (a permanently undefined instruction: a trap).
    Udf = 46,
    /// `[Def r0, Use r7, Use r0..]` — `svc #0`, the Arm Linux EABI system call
    /// (number in `r7`, arguments in `r0`–`r5`, result in `r0`).
    Svc = 47,
    /// `[]` — `dmb sy` (a full data memory barrier).
    Dmb = 48,
    /// `[Imm k]` — the prologue's `sub sp, sp, #k` (probed at encode time when
    /// the layout asks for it).
    SubSp = 49,
    /// `[Imm k]` — the epilogue's `add sp, sp, #k`.
    AddSp = 50,
    /// `[Imm mask]` — `push {regs}` (a bitmask of r0-r12 and lr).
    Push = 51,
    /// `[Imm mask]` — `pop {regs}` (a bitmask; bit 15 pops the return address
    /// into `pc`).
    Pop = 52,
    /// `[Def d, Use s]` — `mvn d, s`.
    Mvn = 53,
}

impl ThOp {
    /// The MIR [`Opcode`] id for this opcode.
    #[inline]
    pub fn opcode(self) -> Opcode {
        Opcode(self as u32)
    }

    /// Whether an instruction of this opcode may execute a conditional branch
    /// whose direction depends on a register operand — the constant-time
    /// audit of the lowering (`docs/ir-design.md` §6d): the terminators
    /// `BrCond`, `Switch` and `Switch64`. The `IT`-block sequences are not
    /// branches: `SetCmp`/`SetCmpImm` (`cmp; ite; mov; mov`) and `Select`
    /// (`tst; ite; mov; mov`) issue every instruction of the block whatever
    /// the condition (a failing predicate turns an instruction into a no-op
    /// of the same timing), so they are straight-line code. The remainders'
    /// `sdiv`/`udiv` are variable-time, but division of a secret is rejected
    /// by the verifier, and the division helpers without hardware divide are
    /// calls, which the verifier rejects on secrets too. (The prologue's probe
    /// loop counts a constant frame size.)
    pub fn may_branch_on_data(self, _operands: &[MachineOperand]) -> bool {
        matches!(self, ThOp::BrCond | ThOp::Switch | ThOp::Switch64)
    }

    /// Decode a MIR [`Opcode`] back to a [`ThOp`].
    pub fn decode(op: Opcode) -> ThOp {
        use ThOp::*;
        const TABLE: [ThOp; 54] = [
            Mov, MovImm, Add, Sub, And, Orr, Eor, Mul, Sdiv, Udiv, Srem, Urem, Lsl, Lsr, Asr,
            AddImm, AndImm, OrrImm, EorImm, RsbImm, LslImm, LsrImm, AsrImm, Ext, SetCmp,
            SetCmpImm, Select, Load, Store, LoadDual, StoreDual, FrameAddr, LoadFrame, StoreFrame,
            SpAddr, StoreSp, IncAddr, LoadInc, GlobalAddr, FuncAddr, Call, Ret, B, BrCond, Switch,
            Switch64, Udf, Svc, Dmb, SubSp, AddSp, Push, Pop, Mvn,
        ];
        TABLE[op.0 as usize]
    }
}

/// The Arm condition code (`EQ` = 0 … `LE` = 13) that holds after `cmp a, b`
/// exactly when `a pred b`.
pub(crate) fn cond_code(p: IntPred) -> u8 {
    match p {
        IntPred::Eq => 0x0,
        IntPred::Ne => 0x1,
        IntPred::Uge => 0x2, // HS
        IntPred::Ult => 0x3, // LO
        IntPred::Ugt => 0x8, // HI
        IntPred::Ule => 0x9, // LS
        IntPred::Sge => 0xa, // GE
        IntPred::Slt => 0xb, // LT
        IntPred::Sgt => 0xc, // GT
        IntPred::Sle => 0xd, // LE
    }
}

/// The predicate with its operands swapped (`a < b` ⇔ `b > a`).
fn swap_pred(p: IntPred) -> IntPred {
    use IntPred::*;
    match p {
        Eq => Eq,
        Ne => Ne,
        Ugt => Ult,
        Uge => Ule,
        Ult => Ugt,
        Ule => Uge,
        Sgt => Slt,
        Sge => Sle,
        Slt => Sgt,
        Sle => Sge,
    }
}

fn is_signed(p: IntPred) -> bool {
    matches!(p, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge)
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
fn imm_i(v: i64) -> MachineOperand {
    MachineOperand::Imm(Int::from_i64(v))
}

/// The low 32 bits of an arbitrary-precision integer (two's complement).
pub(crate) fn low32(v: &Int) -> u32 {
    v.mod_2k(32).to_u64().unwrap_or(0) as u32
}

/// The 32-bit parts of `v` taken as a `bits`-bit pattern, least significant
/// first.
fn words_of(v: &Int, n: usize) -> Vec<u32> {
    let bits = v.mod_2k(32 * n as u32);
    (0..n).map(|k| low32(&bits.div_2k_trunc(32 * k as u32))).collect()
}

/// `v` sign-extended from `width` bits to 32 (the pattern a sign-extended
/// register holds), as a `u32`.
fn sext32(v: &Int, width: u32) -> u32 {
    let x = low32(v);
    if width >= 32 {
        return x;
    }
    let sh = 32 - width;
    (((x << sh) as i32) >> sh) as u32
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a.max(1)) * a.max(1)
}

/// The runtime helpers the isel calls directly, as function indices into the
/// prepared module (declared by [`super::prepare_module`] when needed).
#[derive(Clone, Debug, Default)]
pub struct Helpers {
    idiv: Option<u32>,
    uidiv: Option<u32>,
    idivmod: Option<u32>,
    uidivmod: Option<u32>,
    ldivmod: Option<u32>,
    uldivmod: Option<u32>,
}

impl Helpers {
    /// Find the helpers `module` declares (by name).
    pub fn resolve(module: &Module, syms: &StrInterner) -> Helpers {
        let find = |name: &str| {
            module.functions().position(|f| syms.resolve(f.name) == name).map(|i| i as u32)
        };
        Helpers {
            idiv: find("__aeabi_idiv"),
            uidiv: find("__aeabi_uidiv"),
            idivmod: find("__aeabi_idivmod"),
            uidivmod: find("__aeabi_uidivmod"),
            ldivmod: find("__aeabi_ldivmod"),
            uldivmod: find("__aeabi_uldivmod"),
        }
    }
}

/// How a call's result or a function's return value travels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RetClass {
    /// No value.
    Void,
    /// `n` words in `r0`.. (a scalar: 1; a 64-bit value: 2).
    Regs(usize),
    /// A composite of at most 4 bytes, its word in `r0`.
    SmallAgg,
    /// A composite in memory at the address passed in `r0`.
    Memory,
}

/// Where one argument word (or a whole argument) goes.
#[derive(Clone, Debug)]
enum ArgPlace {
    /// Words in core registers `r[first]..`, and `stack` further words at the
    /// stack offset `stack_off` (a split composite has both).
    Split { first: usize, regs: usize, stack_off: u64, stack: usize },
}

/// The AAPCS stage-C argument allocator: the next core register (`ncrn`) and
/// the next stacked argument offset (`nsaa`).
#[derive(Clone, Copy, Debug, Default)]
struct ArgAlloc {
    ncrn: usize,
    nsaa: u64,
}

impl ArgAlloc {
    /// Place an argument of `words` words, doubleword aligned when `dword`,
    /// possibly `splittable` across `r3` and the stack (composites).
    fn place(&mut self, words: usize, dword: bool, splittable: bool) -> ArgPlace {
        // C.3: doubleword alignment rounds the NCRN up to an even register.
        if dword && self.ncrn < 4 && !self.ncrn.is_multiple_of(2) {
            self.ncrn += 1;
        }
        // C.4: it fits in the remaining core registers.
        if self.ncrn + words <= 4 {
            let first = self.ncrn;
            self.ncrn += words;
            return ArgPlace::Split { first, regs: words, stack_off: 0, stack: 0 };
        }
        // C.5: a composite straddling r3 is split while nothing went to the
        // stack yet.
        if splittable && self.ncrn < 4 && self.nsaa == 0 {
            let first = self.ncrn;
            let regs = 4 - first;
            self.ncrn = 4;
            let stack_off = self.nsaa;
            self.nsaa += 4 * (words - regs) as u64;
            return ArgPlace::Split { first, regs, stack_off, stack: words - regs };
        }
        // C.6-C.8: the stack, 8-aligned for a doubleword-aligned argument.
        self.ncrn = 4;
        if dword {
            self.nsaa = align_up(self.nsaa, 8);
        }
        let stack_off = self.nsaa;
        self.nsaa += 4 * words as u64;
        ArgPlace::Split { first: 4, regs: 0, stack_off, stack: words }
    }
}

/// Prologue work deferred until every argument register has been read.
type Deferred = Box<dyn FnOnce(&ThumbTarget, &mut Lower<'_, ThumbTarget>)>;

/// The Thumb-2 (ARMv7-M) target: its register file and AAPCS plus the isel
/// rules.
#[derive(Debug)]
pub struct ThumbTarget {
    rf: RegFile,
    hw_div: bool,
    helpers: Helpers,
    /// The higher 32-bit parts of each wide value of the function being
    /// lowered, by `ValueId` index (part 0 is the value's own vreg).
    wide: RefCell<DetHashMap<usize, Vec<VReg>>>,
}

impl Default for ThumbTarget {
    fn default() -> Self {
        Self::new()
    }
}

impl ThumbTarget {
    /// The ARMv7-M target (hardware divide) with no runtime helpers resolved.
    pub fn new() -> ThumbTarget {
        ThumbTarget { rf: RegFile::new(), hw_div: true, helpers: Helpers::default(), wide: RefCell::default() }
    }

    /// Use (`true`, the ARMv7-M default) or avoid the `sdiv`/`udiv`
    /// instructions; without them 32-bit division calls the AEABI helpers,
    /// which [`super::prepare_module`] declares.
    pub fn with_hw_div(mut self, on: bool) -> ThumbTarget {
        self.hw_div = on;
        self
    }

    /// Call the runtime helpers `helpers` names (see [`Helpers::resolve`]).
    pub fn with_helpers(mut self, helpers: Helpers) -> ThumbTarget {
        self.helpers = helpers;
        self
    }

    /// Lower function `func` of the prepared `module` to MIR over this target.
    pub fn select(
        &self,
        module: &Module,
        func: crate::ir::FuncId,
        syms: &StrInterner,
    ) -> crate::codegen::mir::MachineFunction {
        self.wide.borrow_mut().clear();
        crate::codegen::isel::select_with_syms(self, module, func, syms)
    }

    fn emit(lo: &mut Lower<'_, Self>, op: ThOp, ops: Vec<MachineOperand>) {
        lo.emit(MachineInst::new(op.opcode(), ops));
    }

    fn fresh(lo: &mut Lower<'_, Self>) -> VReg {
        lo.fresh_vreg(RegClass::Gpr)
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

    fn is_aggregate(lo: &Lower<'_, Self>, ty: TypeId) -> bool {
        matches!(lo.types().get(ty), Type::Struct(_) | Type::Array(..))
    }

    /// The integer width of `v` if it is an integer wider than 32 bits.
    fn wide_width(lo: &Lower<'_, Self>, v: ValueId) -> Option<u32> {
        match lo.types().get(lo.func().value_type(v)) {
            Type::Int(b) if *b > 32 => Some(*b),
            _ => None,
        }
    }

    fn nparts(bits: u32) -> usize {
        bits.div_ceil(32) as usize
    }

    /// The 32-bit parts of a wide value, least significant first: a constant's
    /// words materialized here, or the value's register group (its own vreg
    /// plus the side table's higher parts).
    fn parts(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> Vec<VReg> {
        let width = Self::wide_width(lo, v).expect("a wide integer");
        let n = Self::nparts(width);
        match lo.func().value(v).def.clone() {
            ValueDef::Const(c) => {
                let words = match lo.module().consts().get(c) {
                    Const::Int { value, .. } => words_of(value, n),
                    _ => vec![0; n],
                };
                words
                    .into_iter()
                    .map(|w| {
                        let d = Self::fresh(lo);
                        Self::emit(lo, ThOp::MovImm, vec![def_v(d), imm(u64::from(w))]);
                        d
                    })
                    .collect()
            }
            ValueDef::Inst(_) | ValueDef::Param(..) => {
                let base = lo.reg(v);
                if let Some(p) = self.wide.borrow().get(&v.index()) {
                    return p.clone();
                }
                let mut p = vec![base];
                for _ in 1..n {
                    p.push(Self::fresh(lo));
                }
                self.wide.borrow_mut().insert(v.index(), p.clone());
                p
            }
            ValueDef::Global(_) | ValueDef::Func(_) => unreachable!("an address is never wide"),
        }
    }

    /// Resolve an operand to a register; a function used as a value is its
    /// address ([`ThOp::FuncAddr`]) rather than the framework's placeholder.
    fn val(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        if let ValueDef::Func(f) = lo.func().value(v).def {
            let d = Self::fresh(lo);
            Self::emit(lo, ThOp::FuncAddr, vec![def_v(d), MachineOperand::Func(f.index() as u32)]);
            return d;
        }
        if Self::wide_width(lo, v).is_some() {
            return self.parts(lo, v)[0];
        }
        lo.reg(v)
    }

    /// Whether `v`'s register is known to hold its value zero-extended to 32
    /// bits: a compare result, or a byte or halfword load (`ldrb`/`ldrh`
    /// zero-extend; a load of an odd width such as `i1` reads a whole byte,
    /// which may hold more than the value).
    fn is_zero_clean(lo: &Lower<'_, Self>, v: ValueId) -> bool {
        match lo.func().value(v).def {
            ValueDef::Inst(id) => match lo.func().inst(id).kind {
                InstKind::ICmp(_) | InstKind::FCmp(_) => true,
                InstKind::Load { ty, .. } => matches!(lo.types().get(ty), Type::Int(8 | 16)),
                _ => false,
            },
            _ => false,
        }
    }

    /// `v` extended from its width to all 32 bits of a register (see "Narrow
    /// values" in the module docs); a 32-bit (or wider: part 0) value as is.
    fn ext(&self, lo: &mut Lower<'_, Self>, v: ValueId, signed: bool) -> VReg {
        let width = lo.int_width(v);
        if let Some(c) = Self::const_of(lo, v)
            && width < 32
        {
            let d = Self::fresh(lo);
            let bits = if signed { sext32(&c, width) } else { low32(&c.mod_2k(width)) };
            Self::emit(lo, ThOp::MovImm, vec![def_v(d), imm(u64::from(bits))]);
            return d;
        }
        let r = self.val(lo, v);
        if width >= 32 || (!signed && Self::is_zero_clean(lo, v)) {
            return r;
        }
        self.ext_reg(lo, r, width, signed)
    }

    /// Extend the low `width` bits of register `r`.
    fn ext_reg(&self, lo: &mut Lower<'_, Self>, r: VReg, width: u32, signed: bool) -> VReg {
        let d = Self::fresh(lo);
        Self::emit(lo, ThOp::Ext, vec![def_v(d), use_v(r), imm(u64::from(width)), imm(u64::from(signed))]);
        d
    }

    fn mov(lo: &mut Lower<'_, Self>, d: VReg, s: VReg) {
        Self::emit(lo, ThOp::Mov, vec![def_v(d), use_v(s)]);
    }

    fn movi(lo: &mut Lower<'_, Self>, d: VReg, v: u32) {
        Self::emit(lo, ThOp::MovImm, vec![def_v(d), imm(u64::from(v))]);
    }

    /// Copy the words `src` into the value's register group (its parts).
    fn set_parts(&self, lo: &mut Lower<'_, Self>, res: ValueId, src: &[VReg]) {
        let dst = self.parts(lo, res);
        for (d, s) in dst.into_iter().zip(src.iter().copied()) {
            Self::mov(lo, d, s);
        }
    }

    fn zero(lo: &mut Lower<'_, Self>) -> VReg {
        let z = Self::fresh(lo);
        Self::movi(lo, z, 0);
        z
    }

    // --- arithmetic ---------------------------------------------------------

    fn lower_bin(&self, lo: &mut Lower<'_, Self>, op: BinOp, inst: &InstData) {
        let res = inst.result().expect("a binop defines a value");
        let (l, r) = (inst.operands()[0], inst.operands()[1]);
        if Self::wide_width(lo, res).is_some() {
            return self.lower_wide_bin(lo, op, inst);
        }
        let d = lo.result_reg(inst);
        let three = |lo: &mut Lower<'_, Self>, top: ThOp, a: VReg, b: VReg| {
            Self::emit(lo, top, vec![def_v(d), use_v(a), use_v(b)]);
        };
        match op {
            BinOp::Add | BinOp::And | BinOp::Or | BinOp::Xor | BinOp::Mul => {
                let (rop, iop) = match op {
                    BinOp::Add => (ThOp::Add, Some(ThOp::AddImm)),
                    BinOp::And => (ThOp::And, Some(ThOp::AndImm)),
                    BinOp::Or => (ThOp::Orr, Some(ThOp::OrrImm)),
                    BinOp::Xor => (ThOp::Eor, Some(ThOp::EorImm)),
                    _ => (ThOp::Mul, None),
                };
                // Commutative: a constant on either side folds into the
                // immediate form.
                let (x, k) = match (Self::const_of(lo, l), Self::const_of(lo, r)) {
                    (_, Some(c)) => (l, Some(c)),
                    (Some(c), None) => (r, Some(c)),
                    _ => (l, None),
                };
                if let (Some(iop), Some(c)) = (iop, k) {
                    let a = self.val(lo, x);
                    let kv = if iop == ThOp::AddImm { imm_i(i64::from(low32(&c) as i32)) } else { imm(u64::from(low32(&c))) };
                    Self::emit(lo, iop, vec![def_v(d), use_v(a), kv]);
                } else {
                    let a = self.val(lo, l);
                    let b = self.val(lo, r);
                    three(lo, rop, a, b);
                }
            }
            BinOp::Sub => {
                if let Some(c) = Self::const_of(lo, r) {
                    let a = self.val(lo, l);
                    let k = -i64::from(low32(&c) as i32);
                    Self::emit(lo, ThOp::AddImm, vec![def_v(d), use_v(a), imm_i(k)]);
                } else if let Some(c) = Self::const_of(lo, l) {
                    let b = self.val(lo, r);
                    Self::emit(lo, ThOp::RsbImm, vec![def_v(d), use_v(b), imm(u64::from(low32(&c)))]);
                } else {
                    let a = self.val(lo, l);
                    let b = self.val(lo, r);
                    three(lo, ThOp::Sub, a, b);
                }
            }
            BinOp::Shl | BinOp::LShr | BinOp::AShr => {
                // A right shift brings the bits above the width down.
                let a = match op {
                    BinOp::LShr => self.ext(lo, l, false),
                    BinOp::AShr => self.ext(lo, l, true),
                    _ => self.val(lo, l),
                };
                if let Some(c) = Self::const_of(lo, r) {
                    // An amount of at least the width is poison: any result.
                    let n = c.to_u64().unwrap_or(0).min(31);
                    let iop = match op {
                        BinOp::Shl => ThOp::LslImm,
                        BinOp::LShr => ThOp::LsrImm,
                        _ => ThOp::AsrImm,
                    };
                    if n == 0 {
                        Self::mov(lo, d, a);
                    } else {
                        Self::emit(lo, iop, vec![def_v(d), use_v(a), imm(n)]);
                    }
                } else {
                    // The shifter reads the amount's low byte: an amount
                    // narrower than 8 bits may carry garbage inside it.
                    let b = if lo.int_width(r) < 8 { self.ext(lo, r, false) } else { self.val(lo, r) };
                    let rop = match op {
                        BinOp::Shl => ThOp::Lsl,
                        BinOp::LShr => ThOp::Lsr,
                        _ => ThOp::Asr,
                    };
                    three(lo, rop, a, b);
                }
            }
            BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => {
                let signed = matches!(op, BinOp::SDiv | BinOp::SRem);
                let a = self.ext(lo, l, signed);
                let b = self.ext(lo, r, signed);
                if self.hw_div {
                    let top = match op {
                        BinOp::UDiv => ThOp::Udiv,
                        BinOp::SDiv => ThOp::Sdiv,
                        BinOp::URem => ThOp::Urem,
                        _ => ThOp::Srem,
                    };
                    three(lo, top, a, b);
                } else {
                    let (helper, reg) = match op {
                        BinOp::UDiv => (self.helpers.uidiv, 0),
                        BinOp::SDiv => (self.helpers.idiv, 0),
                        BinOp::URem => (self.helpers.uidivmod, 1),
                        _ => (self.helpers.idivmod, 1),
                    };
                    let f = helper.expect("the division helpers are declared by prepare_module");
                    let out = self.helper_call(lo, f, &[a, b], &[reg]);
                    Self::mov(lo, d, out[0]);
                }
            }
            _ => panic!("thumb backend: floating-point {op:?} reached isel (run prepare_module first)"),
        }
    }

    /// A call to runtime helper `f` with word arguments `args` in `r0`.., its
    /// results read from registers `outs`.
    fn helper_call(&self, lo: &mut Lower<'_, Self>, f: u32, args: &[VReg], outs: &[u16]) -> Vec<VReg> {
        let mut ops = vec![MachineOperand::Func(f)];
        for &c in &self.rf.caller_saved {
            ops.push(def(c));
        }
        for (k, &a) in args.iter().enumerate() {
            Self::emit(lo, ThOp::Mov, vec![def(gpr(k as u16)), use_v(a)]);
        }
        for k in 0..args.len() {
            ops.push(use_p(gpr(k as u16)));
        }
        Self::emit(lo, ThOp::Call, ops);
        outs.iter()
            .map(|&r| {
                let d = Self::fresh(lo);
                Self::emit(lo, ThOp::Mov, vec![def_v(d), use_p(gpr(r))]);
                d
            })
            .collect()
    }

    /// The wide operations the legalizer leaves: the join (`or` of parts,
    /// `shl` by whole parts) and split (`lshr` by whole parts) shapes, plus
    /// part-wise logic for good measure.
    fn lower_wide_bin(&self, lo: &mut Lower<'_, Self>, op: BinOp, inst: &InstData) {
        let res = inst.result().expect("a result");
        let (l, r) = (inst.operands()[0], inst.operands()[1]);
        match op {
            BinOp::And | BinOp::Or | BinOp::Xor => {
                let a = self.parts(lo, l);
                let b = self.parts(lo, r);
                let top = match op {
                    BinOp::And => ThOp::And,
                    BinOp::Or => ThOp::Orr,
                    _ => ThOp::Eor,
                };
                let out: Vec<VReg> = a
                    .iter()
                    .zip(&b)
                    .map(|(&x, &y)| {
                        let d = Self::fresh(lo);
                        Self::emit(lo, top, vec![def_v(d), use_v(x), use_v(y)]);
                        d
                    })
                    .collect();
                self.set_parts(lo, res, &out);
            }
            BinOp::Shl | BinOp::LShr => {
                let a = self.parts(lo, l);
                let n = a.len();
                let s = Self::const_of(lo, r).and_then(|c| c.to_u64()).filter(|s| s % 32 == 0).unwrap_or_else(|| {
                    panic!("thumb backend: a wide {op:?} by a non-multiple of 32 was not legalized")
                });
                let q = (s / 32) as usize;
                let zero = Self::zero(lo);
                let out: Vec<VReg> = (0..n)
                    .map(|k| {
                        let src = if op == BinOp::Shl { k.checked_sub(q) } else { Some(k + q).filter(|&i| i < n) };
                        src.map_or(zero, |i| a[i])
                    })
                    .collect();
                self.set_parts(lo, res, &out);
            }
            _ => panic!("thumb backend: wide {op:?} reached isel (not legalized)"),
        }
    }

    fn lower_icmp(&self, lo: &mut Lower<'_, Self>, pred: IntPred, inst: &InstData) {
        let d = lo.result_reg(inst);
        let (mut l, mut r) = (inst.operands()[0], inst.operands()[1]);
        let mut pred = pred;
        if Self::wide_width(lo, l).is_some() {
            panic!("thumb backend: a wide icmp reached isel (not legalized)");
        }
        if Self::const_of(lo, l).is_some() && Self::const_of(lo, r).is_none() {
            std::mem::swap(&mut l, &mut r);
            pred = swap_pred(pred);
        }
        let signed = is_signed(pred);
        let width = lo.int_width(l);
        let a = self.ext(lo, l, signed);
        let cc = u64::from(cond_code(pred));
        if let Some(c) = Self::const_of(lo, r) {
            let k = if signed { sext32(&c, width) } else { low32(&c.mod_2k(width.min(32))) };
            Self::emit(lo, ThOp::SetCmpImm, vec![def_v(d), use_v(a), imm(u64::from(k)), imm(cc)]);
        } else {
            let b = self.ext(lo, r, signed);
            Self::emit(lo, ThOp::SetCmp, vec![def_v(d), use_v(a), use_v(b), imm(cc)]);
        }
    }

    fn lower_cast(&self, lo: &mut Lower<'_, Self>, op: CastOp, inst: &InstData) {
        let res = inst.result().expect("a cast defines a value");
        let src = inst.operands()[0];
        let res_wide = Self::wide_width(lo, res).map(Self::nparts);
        let src_wide = Self::wide_width(lo, src).is_some();
        match op {
            CastOp::ZExt | CastOp::SExt => {
                let signed = op == CastOp::SExt;
                match res_wide {
                    None => {
                        let s = self.ext(lo, src, signed);
                        let d = lo.result_reg(inst);
                        Self::mov(lo, d, s);
                    }
                    Some(n) => {
                        let mut p = if src_wide { self.parts(lo, src) } else { vec![self.ext(lo, src, signed)] };
                        let fill = if signed {
                            let top = *p.last().expect("a part");
                            let f = Self::fresh(lo);
                            Self::emit(lo, ThOp::AsrImm, vec![def_v(f), use_v(top), imm(31)]);
                            f
                        } else {
                            Self::zero(lo)
                        };
                        p.resize(n, fill);
                        self.set_parts(lo, res, &p);
                    }
                }
            }
            CastOp::Trunc | CastOp::PtrToInt | CastOp::IntToPtr | CastOp::Bitcast => {
                let mut p = if src_wide {
                    self.parts(lo, src)
                } else if op == CastOp::IntToPtr {
                    vec![self.ext(lo, src, false)]
                } else {
                    vec![self.val(lo, src)]
                };
                match res_wide {
                    None => {
                        let d = lo.result_reg(inst);
                        Self::mov(lo, d, p[0]);
                    }
                    Some(n) => {
                        if p.len() < n {
                            let z = Self::zero(lo);
                            p.resize(n, z);
                        }
                        p.truncate(n);
                        self.set_parts(lo, res, &p);
                    }
                }
            }
            _ => panic!("thumb backend: floating-point {op:?} reached isel (run prepare_module first)"),
        }
    }

    // --- memory ---------------------------------------------------------------

    fn lower_load(&self, lo: &mut Lower<'_, Self>, inst: &InstData, ty: TypeId, volatile: bool) {
        let res = inst.result().expect("a load defines a value");
        let ptr = self.val(lo, inst.operands()[0]);
        let size = lo.byte_size(ty);
        if let Some(bits) = Self::wide_width(lo, res) {
            let n = Self::nparts(bits);
            if n == 2 && volatile {
                let (a, b) = (Self::fresh(lo), Self::fresh(lo));
                Self::emit(lo, ThOp::LoadDual, vec![def_v(a), def_v(b), use_v(ptr)]);
                self.set_parts(lo, res, &[a, b]);
            } else {
                let words: Vec<VReg> = (0..n)
                    .map(|k| {
                        let d = Self::fresh(lo);
                        Self::emit(lo, ThOp::Load, vec![def_v(d), use_v(ptr), imm(4 * k as u64), imm(4)]);
                        d
                    })
                    .collect();
                self.set_parts(lo, res, &words);
            }
            return;
        }
        assert!(size <= 4, "thumb backend: a {size}-byte load of a scalar");
        let d = lo.result_reg(inst);
        Self::emit(lo, ThOp::Load, vec![def_v(d), use_v(ptr), imm(0), imm(size.max(1))]);
    }

    fn lower_store(&self, lo: &mut Lower<'_, Self>, inst: &InstData, ty: TypeId, volatile: bool) {
        let (p, v) = (inst.operands()[0], inst.operands()[1]);
        let ptr = self.val(lo, p);
        let size = lo.byte_size(ty);
        if Self::wide_width(lo, v).is_some() {
            let parts = self.parts(lo, v);
            if parts.len() == 2 && volatile {
                Self::emit(lo, ThOp::StoreDual, vec![use_v(ptr), use_v(parts[0]), use_v(parts[1])]);
            } else {
                for (k, w) in parts.into_iter().enumerate() {
                    Self::emit(lo, ThOp::Store, vec![use_v(ptr), use_v(w), imm(4 * k as u64), imm(4)]);
                }
            }
            return;
        }
        assert!(size <= 4, "thumb backend: a {size}-byte store of a scalar");
        let val = self.val(lo, v);
        Self::emit(lo, ThOp::Store, vec![use_v(ptr), use_v(val), imm(0), imm(size.max(1))]);
    }

    /// Load the `bytes` (1..=4) bytes at `[ptr + off]` into the low end of a
    /// fresh register (no access past them: a composite's tail word).
    fn load_partial(&self, lo: &mut Lower<'_, Self>, ptr: VReg, off: u64, bytes: u64) -> VReg {
        let d = Self::fresh(lo);
        match bytes {
            4 | 2 | 1 => Self::emit(lo, ThOp::Load, vec![def_v(d), use_v(ptr), imm(off), imm(bytes)]),
            _ => {
                // Three bytes: a halfword, then the third byte shifted in.
                let h = Self::fresh(lo);
                Self::emit(lo, ThOp::Load, vec![def_v(h), use_v(ptr), imm(off), imm(2)]);
                let b = Self::fresh(lo);
                Self::emit(lo, ThOp::Load, vec![def_v(b), use_v(ptr), imm(off + 2), imm(1)]);
                let s = Self::fresh(lo);
                Self::emit(lo, ThOp::LslImm, vec![def_v(s), use_v(b), imm(16)]);
                Self::emit(lo, ThOp::Orr, vec![def_v(d), use_v(h), use_v(s)]);
            }
        }
        d
    }

    /// Copy `size` bytes from `[src]` to `[dst]` in word/halfword/byte chunks.
    fn memcpy(&self, lo: &mut Lower<'_, Self>, dst: VReg, src: VReg, size: u64) {
        let mut o = 0;
        while o < size {
            let chunk = match size - o {
                n if n >= 4 => 4,
                n if n >= 2 => 2,
                _ => 1,
            };
            let t = Self::fresh(lo);
            Self::emit(lo, ThOp::Load, vec![def_v(t), use_v(src), imm(o), imm(chunk)]);
            Self::emit(lo, ThOp::Store, vec![use_v(dst), use_v(t), imm(o), imm(chunk)]);
            o += chunk;
        }
    }

    // --- calls ------------------------------------------------------------------

    fn ret_class(lo: &Lower<'_, Self>, ty: TypeId) -> RetClass {
        match lo.types().get(ty) {
            Type::Void => RetClass::Void,
            Type::Struct(_) | Type::Array(..) => {
                if lo.byte_size(ty) <= 4 {
                    RetClass::SmallAgg
                } else {
                    RetClass::Memory
                }
            }
            Type::Int(b) if *b > 32 => RetClass::Regs(Self::nparts(*b)),
            _ => RetClass::Regs(1),
        }
    }

    /// A value as the AAPCS wants it in a register: an `i1` zero-extended.
    /// (The IR does not carry the C signedness of other narrow types, which are
    /// passed with whatever their register holds above their width.)
    fn abi_word(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        let is_bool = matches!(lo.types().get(lo.func().value_type(v)), Type::Int(1));
        if is_bool { self.ext(lo, v, false) } else { self.val(lo, v) }
    }

    fn lower_call(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];
        let name = lo.callee_name(callee).map(str::to_owned);

        // The legalizer's 64-bit remainder placeholders: `__aeabi_{u,}ldivmod`,
        // remainder in r2:r3.
        let (target_fn, rem_regs) = match name.as_deref() {
            Some(LMOD_PSEUDO) => (self.helpers.ldivmod, true),
            Some(ULMOD_PSEUDO) => (self.helpers.uldivmod, true),
            _ => (lo.callee_func(callee), false),
        };
        if rem_regs {
            assert!(target_fn.is_some(), "the 64-bit division helpers are declared by prepare_module");
        }

        let ret_ty = inst.result().map(|r| lo.func().value_type(r));
        let rclass = ret_ty.map_or(RetClass::Void, |t| Self::ret_class(lo, t));

        let mut alloc = ArgAlloc::default();
        let mut reg_moves: Vec<(u16, VReg)> = Vec::new();
        let mut ret_slot = None;
        if rclass == RetClass::Memory {
            let t = ret_ty.expect("a result");
            let slot = lo.new_slot(align_up(lo.byte_size(t), 4), lo.types().align_of(t).max(4));
            ret_slot = Some(slot);
            let p = Self::fresh(lo);
            lo.emit(self.frame_addr(p, slot));
            reg_moves.push((0, p));
            alloc.ncrn = 1;
        }

        // A composite parameter may receive a `ptr` argument (§6 of the IR
        // design): classify by the callee's signature when it is known.
        let sig_params: Option<Vec<TypeId>> = lo.callee_func(callee).and_then(|f| {
            match lo.types().get(lo.module().function(crate::ir::FuncId::from_index(f as usize)).sig) {
                Type::Func(ft) if ft.params.len() == args.len() => Some(ft.params.clone()),
                _ => None,
            }
        });
        for (k, &arg) in args.iter().enumerate() {
            let ty = sig_params.as_ref().map_or_else(|| lo.func().value_type(arg), |p| p[k]);
            if Self::is_aggregate(lo, ty) {
                let size = lo.byte_size(ty);
                let words = size.div_ceil(4) as usize;
                let dword = lo.types().align_of(ty) >= 8;
                let ArgPlace::Split { first, regs, stack_off, stack } = alloc.place(words, dword, true);
                let src = self.val(lo, arg);
                for k in 0..regs {
                    let off = 4 * k as u64;
                    let w = self.load_partial(lo, src, off, (size - off).min(4));
                    reg_moves.push(((first + k) as u16, w));
                }
                if stack > 0 {
                    let done = 4 * regs as u64;
                    let dst = Self::fresh(lo);
                    Self::emit(lo, ThOp::SpAddr, vec![def_v(dst), imm(stack_off)]);
                    let from = Self::fresh(lo);
                    Self::emit(lo, ThOp::AddImm, vec![def_v(from), use_v(src), imm_i(done as i64)]);
                    self.memcpy(lo, dst, from, size - done);
                }
            } else if Self::wide_width(lo, arg).is_some() {
                let p = self.parts(lo, arg);
                let ArgPlace::Split { first, regs, stack_off, .. } = alloc.place(p.len(), true, false);
                for (k, &w) in p.iter().enumerate() {
                    if k < regs {
                        reg_moves.push(((first + k) as u16, w));
                    } else {
                        let off = stack_off + 4 * (k - regs) as u64;
                        Self::emit(lo, ThOp::StoreSp, vec![use_v(w), imm(off), imm(4)]);
                    }
                }
            } else {
                let w = self.abi_word(lo, arg);
                let ArgPlace::Split { first, regs, stack_off, .. } = alloc.place(1, false, false);
                if regs == 1 {
                    reg_moves.push((first as u16, w));
                } else {
                    Self::emit(lo, ThOp::StoreSp, vec![use_v(w), imm(stack_off), imm(4)]);
                }
            }
        }
        if alloc.nsaa > 0 {
            lo.reserve_outgoing(align_up(alloc.nsaa, 8));
        }

        let callee_reg = if target_fn.is_none() { Some(self.val(lo, callee)) } else { None };
        // The argument-register moves, one consecutive run right before the call.
        for &(r, v) in &reg_moves {
            Self::emit(lo, ThOp::Mov, vec![def(gpr(r)), use_v(v)]);
        }
        let mut operands = match target_fn {
            Some(f) => vec![MachineOperand::Func(f)],
            None => vec![use_v(callee_reg.expect("an indirect callee"))],
        };
        for &c in &self.rf.caller_saved {
            operands.push(def(c));
        }
        for &(r, _) in &reg_moves {
            operands.push(use_p(gpr(r)));
        }
        Self::emit(lo, ThOp::Call, operands);

        let Some(res) = inst.result() else { return };
        match rclass {
            RetClass::Void => {}
            RetClass::Regs(n) => {
                let first = if rem_regs { 2 } else { 0 };
                let words: Vec<VReg> = (0..n)
                    .map(|k| {
                        let d = Self::fresh(lo);
                        Self::emit(lo, ThOp::Mov, vec![def_v(d), use_p(gpr((first + k) as u16))]);
                        d
                    })
                    .collect();
                if n == 1 {
                    let d = lo.result_reg(inst);
                    Self::mov(lo, d, words[0]);
                } else {
                    self.set_parts(lo, res, &words);
                }
            }
            RetClass::SmallAgg => {
                let w = Self::fresh(lo);
                Self::emit(lo, ThOp::Mov, vec![def_v(w), use_p(gpr(regs::R0))]);
                let slot = lo.new_slot(4, 4);
                let d = lo.result_reg(inst);
                lo.emit(self.frame_addr(d, slot));
                Self::emit(lo, ThOp::Store, vec![use_v(d), use_v(w), imm(0), imm(4)]);
            }
            RetClass::Memory => {
                let d = lo.result_reg(inst);
                lo.emit(self.frame_addr(d, ret_slot.expect("the result slot")));
            }
        }
    }

    /// The entry prologue: bind each parameter from its AAPCS location. Every
    /// read of an argument register comes first, then the stack loads and the
    /// composite home copies.
    fn lower_params(&self, lo: &mut Lower<'_, Self>) {
        let params: Vec<ValueId> = lo.func().block(lo.func().entry().expect("an entry")).params().to_vec();
        let ret_ty = match lo.types().get(lo.func().sig) {
            Type::Func(ft) => ft.ret,
            _ => lo.func().sig,
        };
        let mut alloc = ArgAlloc::default();
        // Work deferred until every argument register has been read.
        let mut later: Vec<Deferred> = Vec::new();
        if Self::ret_class(lo, ret_ty) == RetClass::Memory {
            let slot = lo.new_slot(4, 4);
            lo.set_aux_slot(slot);
            Self::emit(lo, ThOp::StoreFrame, vec![use_p(gpr(regs::R0)), MachineOperand::Frame(slot)]);
            alloc.ncrn = 1;
        }
        for &p in &params {
            let ty = lo.func().value_type(p);
            if Self::is_aggregate(lo, ty) {
                let size = lo.byte_size(ty);
                let words = size.div_ceil(4) as usize;
                let dword = lo.types().align_of(ty) >= 8;
                let ArgPlace::Split { first, regs, stack_off, stack } = alloc.place(words, dword, true);
                let pv = lo.reg(p);
                if regs == 0 {
                    // Entirely on the stack: use the caller's copy in place.
                    Self::emit(lo, ThOp::IncAddr, vec![def_v(pv), imm(stack_off)]);
                    continue;
                }
                let got: Vec<VReg> = (0..regs)
                    .map(|k| {
                        let w = Self::fresh(lo);
                        Self::emit(lo, ThOp::Mov, vec![def_v(w), use_p(gpr((first + k) as u16))]);
                        w
                    })
                    .collect();
                let align = lo.types().align_of(ty).max(4);
                later.push(Box::new(move |s: &Self, lo: &mut Lower<'_, Self>| {
                    let slot = lo.new_slot(4 * words as u64, align);
                    lo.emit(s.frame_addr(pv, slot));
                    for (k, w) in got.into_iter().enumerate() {
                        Self::emit(lo, ThOp::Store, vec![use_v(pv), use_v(w), imm(4 * k as u64), imm(4)]);
                    }
                    if stack > 0 {
                        let src = Self::fresh(lo);
                        Self::emit(lo, ThOp::IncAddr, vec![def_v(src), imm(stack_off)]);
                        let dst = Self::fresh(lo);
                        Self::emit(lo, ThOp::AddImm, vec![def_v(dst), use_v(pv), imm_i(4 * regs as i64)]);
                        s.memcpy(lo, dst, src, 4 * stack as u64);
                    }
                }));
            } else if let Some(bits) = Self::wide_width(lo, p) {
                let n = Self::nparts(bits);
                let ArgPlace::Split { first, regs, stack_off, .. } = alloc.place(n, true, false);
                let dst = self.parts(lo, p);
                for (k, &d) in dst.iter().enumerate() {
                    if k < regs {
                        Self::emit(lo, ThOp::Mov, vec![def_v(d), use_p(gpr((first + k) as u16))]);
                    } else {
                        let off = stack_off + 4 * (k - regs) as u64;
                        later.push(Box::new(move |_s: &Self, lo: &mut Lower<'_, Self>| {
                            Self::emit(lo, ThOp::LoadInc, vec![def_v(d), imm(off), imm(4)]);
                        }));
                    }
                }
            } else {
                let pv = lo.reg(p);
                let ArgPlace::Split { first, regs, stack_off, .. } = alloc.place(1, false, false);
                if regs == 1 {
                    Self::emit(lo, ThOp::Mov, vec![def_v(pv), use_p(gpr(first as u16))]);
                } else {
                    later.push(Box::new(move |_s: &Self, lo: &mut Lower<'_, Self>| {
                        Self::emit(lo, ThOp::LoadInc, vec![def_v(pv), imm(stack_off), imm(4)]);
                    }));
                }
            }
        }
        for f in later {
            f(self, lo);
        }
    }

    fn lower_ret(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let ret_ty = match lo.types().get(lo.func().sig) {
            Type::Func(ft) => ft.ret,
            _ => lo.func().sig,
        };
        let mut uses = Vec::new();
        if let Some(&v) = inst.operands().first() {
            match Self::ret_class(lo, ret_ty) {
                RetClass::Void => {}
                RetClass::Regs(1) => {
                    let w = self.abi_word(lo, v);
                    Self::emit(lo, ThOp::Mov, vec![def(gpr(regs::R0)), use_v(w)]);
                    uses.push(use_p(gpr(regs::R0)));
                }
                RetClass::Regs(_) => {
                    let p = self.parts(lo, v);
                    for (k, &w) in p.iter().enumerate() {
                        Self::emit(lo, ThOp::Mov, vec![def(gpr(k as u16)), use_v(w)]);
                        uses.push(use_p(gpr(k as u16)));
                    }
                }
                RetClass::SmallAgg => {
                    let src = self.val(lo, v);
                    let w = self.load_partial(lo, src, 0, lo.byte_size(ret_ty));
                    Self::emit(lo, ThOp::Mov, vec![def(gpr(regs::R0)), use_v(w)]);
                    uses.push(use_p(gpr(regs::R0)));
                }
                RetClass::Memory => {
                    let src = self.val(lo, v);
                    let slot = lo.aux_slot().expect("the result pointer saved by the prologue");
                    let dst = Self::fresh(lo);
                    Self::emit(lo, ThOp::LoadFrame, vec![def_v(dst), MachineOperand::Frame(slot)]);
                    self.memcpy(lo, dst, src, lo.byte_size(ret_ty));
                }
            }
        }
        Self::emit(lo, ThOp::Ret, uses);
    }

    /// Atomics on ARMv7-M: aligned word, halfword and byte accesses are
    /// single-copy atomic, so an atomic load or store is a plain one ordered by
    /// `dmb` barriers (the Arm mapping of C11 atomics: `ldr; dmb` for an
    /// acquire load, `dmb; str; dmb` for a sequentially consistent store, and
    /// `dmb` for any fence).
    fn lower_atomic(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        use crate::ir::inst::AtomicOrdering;
        let dmb = |lo: &mut Lower<'_, Self>| Self::emit(lo, ThOp::Dmb, Vec::new());
        match &inst.kind {
            InstKind::Fence(_) => dmb(lo),
            InstKind::AtomicLoad { ty, ordering, .. } => {
                assert!(lo.byte_size(*ty) <= 4, "thumb backend: no 64-bit atomics on ARMv7-M");
                if *ordering == AtomicOrdering::SeqCst {
                    dmb(lo);
                }
                self.lower_load(lo, inst, *ty, true);
                if ordering.is_acquire() {
                    dmb(lo);
                }
            }
            InstKind::AtomicStore { ty, ordering, .. } => {
                assert!(lo.byte_size(*ty) <= 4, "thumb backend: no 64-bit atomics on ARMv7-M");
                if ordering.is_release() {
                    dmb(lo);
                }
                self.lower_store(lo, inst, *ty, true);
                if *ordering == AtomicOrdering::SeqCst {
                    dmb(lo);
                }
            }
            other => panic!("thumb backend: {other:?} (ldrex/strex loops) is not supported yet"),
        }
    }
}

impl MachineTarget for ThumbTarget {
    fn name(&self) -> &str {
        "thumbv7m"
    }

    fn data_layout(&self) -> crate::ir::DataLayout {
        super::data_layout()
    }

    fn reg_classes(&self) -> &[RegClass] {
        &self.rf.classes
    }

    fn allocatable(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Gpr => &self.rf.allocatable,
            RegClass::Fp => &self.rf.none,
        }
    }

    fn scratch(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Gpr => &self.rf.scratch,
            RegClass::Fp => &self.rf.none,
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
            ThOp::decode(op),
            ThOp::B | ThOp::BrCond | ThOp::Switch | ThOp::Switch64 | ThOp::Ret | ThOp::Udf
        )
    }

    fn is_move(&self, op: Opcode) -> bool {
        ThOp::decode(op) == ThOp::Mov
    }

    fn emit_move(&self, dst: Reg, src: Reg) -> MachineInst {
        MachineInst::new(ThOp::Mov.opcode(), vec![MachineOperand::Def(dst), MachineOperand::Use(src)])
    }

    fn emit_spill(&self, slot: StackSlot, src: PReg) -> MachineInst {
        MachineInst::new(ThOp::StoreFrame.opcode(), vec![use_p(src), MachineOperand::Frame(slot)])
    }

    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(ThOp::LoadFrame.opcode(), vec![def(dst), MachineOperand::Frame(slot)])
    }
}

impl TargetIsel for ThumbTarget {
    fn li(&self, dst: VReg, value: Int) -> MachineInst {
        MachineInst::new(ThOp::MovImm.opcode(), vec![def_v(dst), imm(u64::from(low32(&value)))])
    }

    fn jump(&self, dst: MBlockId) -> MachineInst {
        MachineInst::new(ThOp::B.opcode(), vec![MachineOperand::Label(dst)])
    }

    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst {
        MachineInst::new(ThOp::FrameAddr.opcode(), vec![def_v(dst), MachineOperand::Frame(slot)])
    }

    fn global_addr(&self, dst: VReg, g: u32) -> MachineInst {
        MachineInst::new(ThOp::GlobalAddr.opcode(), vec![def_v(dst), MachineOperand::Global(g)])
    }

    fn lower_prologue(&self, lo: &mut Lower<'_, Self>) {
        self.lower_params(lo);
    }

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        match &inst.kind {
            InstKind::Bin(op) => self.lower_bin(lo, *op, inst),
            InstKind::ICmp(pred) => self.lower_icmp(lo, *pred, inst),
            InstKind::Cast(op) => self.lower_cast(lo, *op, inst),
            InstKind::Alloca { elem_ty } => {
                let d = lo.result_reg(inst);
                let size = lo.byte_size(*elem_ty);
                let align = lo.types().align_of(*elem_ty);
                let slot = lo.new_slot(size.max(1), align);
                lo.emit(self.frame_addr(d, slot));
            }
            // Runtime-sized stack allocation is implemented only on x86-64 so
            // far (as on AArch64 and RISC-V).
            InstKind::DynAlloca { .. } => panic!("thumb backend: dynamic `dyn_alloca` is not yet supported"),
            InstKind::Load { ty, volatile, .. } => self.lower_load(lo, inst, *ty, *volatile),
            InstKind::Store { ty, volatile, .. } => self.lower_store(lo, inst, *ty, *volatile),
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(inst);
                let base = self.val(lo, inst.operands()[0]);
                let off = inst.operands()[1];
                if let Some(c) = Self::const_of(lo, off) {
                    let k = sext32(&c, lo.int_width(off).min(32)) as i32;
                    Self::emit(lo, ThOp::AddImm, vec![def_v(d), use_v(base), imm_i(i64::from(k))]);
                } else {
                    // A byte offset is signed: a narrow one is sign-extended; a
                    // wide one wraps to its low word (addresses are 32 bits).
                    let o = self.ext(lo, off, true);
                    Self::emit(lo, ThOp::Add, vec![def_v(d), use_v(base), use_v(o)]);
                }
            }
            InstKind::Select => {
                let res = inst.result().expect("a select defines a value");
                let c = self.val(lo, inst.operands()[0]);
                if Self::wide_width(lo, res).is_some() {
                    let t = self.parts(lo, inst.operands()[1]);
                    let f = self.parts(lo, inst.operands()[2]);
                    let out: Vec<VReg> = t
                        .iter()
                        .zip(&f)
                        .map(|(&a, &b)| {
                            let d = Self::fresh(lo);
                            Self::emit(lo, ThOp::Select, vec![def_v(d), use_v(c), use_v(a), use_v(b)]);
                            d
                        })
                        .collect();
                    self.set_parts(lo, res, &out);
                } else {
                    let d = lo.result_reg(inst);
                    let t = self.val(lo, inst.operands()[1]);
                    let f = self.val(lo, inst.operands()[2]);
                    Self::emit(lo, ThOp::Select, vec![def_v(d), use_v(c), use_v(t), use_v(f)]);
                }
            }
            // `declassify` only changes what the constant-time verifier knows.
            InstKind::Freeze | InstKind::Declassify => {
                let res = inst.result().expect("a freeze defines a value");
                if Self::wide_width(lo, res).is_some() {
                    let p = self.parts(lo, inst.operands()[0]);
                    self.set_parts(lo, res, &p);
                } else {
                    let d = lo.result_reg(inst);
                    let s = self.val(lo, inst.operands()[0]);
                    Self::mov(lo, d, s);
                }
            }
            InstKind::Call => self.lower_call(lo, inst),
            InstKind::InlineAsm(_) | InstKind::AsmOutput(_) => {
                panic!("thumb backend: {}", crate::codegen::INLINE_ASM_UNSUPPORTED)
            }
            InstKind::Syscall => {
                // The Arm Linux EABI: number in r7, arguments in r0..r5 (the low
                // word of each 64-bit operand), the result in r0, sign-extended
                // to the op's i64.
                let vals: Vec<VReg> = inst.operands().iter().map(|&o| self.val(lo, o)).collect();
                let mut moves: Vec<(u16, VReg)> = vec![(7, vals[0])];
                for (k, &v) in vals[1..].iter().enumerate() {
                    moves.push((k as u16, v));
                }
                let mut operands = vec![def(gpr(regs::R0))];
                for &(r, v) in &moves {
                    Self::emit(lo, ThOp::Mov, vec![def(gpr(r)), use_v(v)]);
                    operands.push(use_p(gpr(r)));
                }
                Self::emit(lo, ThOp::Svc, operands);
                let w = Self::fresh(lo);
                Self::emit(lo, ThOp::Mov, vec![def_v(w), use_p(gpr(regs::R0))]);
                let hi = Self::fresh(lo);
                Self::emit(lo, ThOp::AsrImm, vec![def_v(hi), use_v(w), imm(31)]);
                let res = inst.result().expect("a syscall defines a value");
                self.set_parts(lo, res, &[w, hi]);
            }
            InstKind::Unary(_) | InstKind::FCmp(_) => {
                panic!("thumb backend: floating-point {:?} reached isel (run prepare_module first)", inst.kind)
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
                let cond = self.val(lo, inst.operands()[0]);
                let ops = inst.operands();
                let tb = 1 + *true_args as usize;
                let fb = tb + *false_args as usize;
                let true_vals: Vec<_> = ops[1..tb].to_vec();
                let false_vals: Vec<_> = ops[tb..fb].to_vec();
                let te = lo.edge_to(*if_true, &true_vals);
                let fe = lo.edge_to(*if_false, &false_vals);
                Self::emit(
                    lo,
                    ThOp::BrCond,
                    vec![use_v(cond), MachineOperand::Label(te), MachineOperand::Label(fe)],
                );
            }
            InstKind::Switch(data) => {
                let scrut = inst.operands()[0];
                let width = lo.int_width(scrut);
                let wide = Self::wide_width(lo, scrut).is_some();
                let mut operands = if wide {
                    let p = self.parts(lo, scrut);
                    assert_eq!(p.len(), 2, "thumb backend: a switch wider than 64 bits");
                    vec![use_v(p[0]), use_v(p[1])]
                } else {
                    // Cases are compared as sign-extended 32-bit values.
                    vec![use_v(self.ext(lo, scrut, true))]
                };
                let ops = inst.operands();
                let mut idx = 1usize;
                let dcount = data.default_args as usize;
                let default_vals: Vec<_> = ops[idx..idx + dcount].to_vec();
                idx += dcount;
                let de = lo.edge_to(data.default, &default_vals);
                operands.push(MachineOperand::Label(de));
                let cases = data.cases.clone();
                for case in &cases {
                    let n = case.args as usize;
                    let cvals: Vec<_> = ops[idx..idx + n].to_vec();
                    idx += n;
                    let ce = lo.edge_to(case.target, &cvals);
                    let v = if wide {
                        let w = words_of(&case.value, 2);
                        u64::from(w[0]) | (u64::from(w[1]) << 32)
                    } else {
                        u64::from(sext32(&case.value, width))
                    };
                    operands.push(imm(v));
                    operands.push(MachineOperand::Label(ce));
                }
                Self::emit(lo, if wide { ThOp::Switch64 } else { ThOp::Switch }, operands);
            }
            InstKind::Unreachable => Self::emit(lo, ThOp::Udf, Vec::new()),
            _ => unreachable!("non-terminator reached lower_term: {:?}", inst.kind),
        }
    }
}
