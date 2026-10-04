//! The AVR machine opcode set ([`AvrOp`]) and the instruction-selection rules.
//!
//! [`AvrOp`] is a post-isel, pre-encoding MIR over **register pairs** (see
//! [`super::regs`]). Each data-processing op carries its *container width* —
//! 8 or 16 — and works on the low register (8) or on the whole pair (16); the
//! encoder expands it into the AVR idiom (`add`/`adc`, `cp`/`cpc`, `movw`,
//! shift chains, `ld`/`st` through `Z`, ...). Everything wider than 16 bits
//! has been split into 16-bit parts by the integer legalization run before
//! isel ([`super::prepare`]); what is left wide is the ABI boundary, handled
//! here with a side table mapping each wide IR value to the vregs of its parts.
//!
//! ## Narrow values
//!
//! A value of width `w` lives in a container of 8 bits (`w ≤ 8`, the low
//! register) or 16 bits (`w ≤ 16`, the pair), and the bits of the container
//! above `w` are **not** kept clean: an `i1` produced by `trunc` may carry
//! garbage in bits 1–7. Operations that only feed low bits (`add`, `sub`,
//! `mul`, logic, `shl`, stores, `trunc`) ignore them; every operation whose
//! result depends on them — compares, right shifts, `zext`/`sext`, branch and
//! `select` conditions, `switch`, a variable shift amount, the `ptr_add`
//! offset — first extends the value to its container with [`AvrOp::Ext`]. A
//! compare result is always a clean 0/1 over the whole pair. Division and
//! remainder never reach isel (they are calls to runtime helpers, whose
//! arguments [`super::prepare`] extends), and neither does a multiply on a core
//! without `mul`.
//!
//! ## Calls
//!
//! Arguments follow the avr-gcc convention (`regs::assign_args`, see [`super::regs`]):
//! stack arguments are pushed (last byte first) before the register moves,
//! which are emitted as one run right before the `call` so no competing
//! definition sits between an argument register's write and the call; the
//! caller pops the stack arguments afterwards. The call defines every
//! call-clobbered pair, so the allocator keeps values that live across it in
//! call-saved pairs.

use std::cell::RefCell;

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::mir::{MBlockId, MachineInst, MachineOperand, Opcode, PReg, Reg, RegClass, StackSlot, VReg};
use crate::codegen::target::{CallConv, MachineTarget};
use crate::ir::inst::{BinOp, CastOp, InstKind, IntPred, RmwOp};
use crate::ir::value::{Const, ValueDef};
use crate::ir::{InstData, Module, ValueId};
use crate::support::{DetHashMap, DetHashSet};

use puremp::Int;

use super::regs::{self, ArgLoc, RegFile, pair};

/// The AVR MIR opcode vocabulary. `cw` is the container width (8 or 16);
/// `Def`/`Use` operands are register pairs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum AvrOp {
    /// `[Def d, Use s]` — `movw d, s`.
    Mov = 0,
    /// `[Def d, Imm v]` — load the low 16 bits of `v` (`ldi`, through `Z` for
    /// a low pair).
    Li = 1,
    /// `[Def d, Use a, Use b, Imm cw]` — `add`/`adc`.
    Add = 2,
    /// `[Def d, Use a, Use b, Imm cw]` — `sub`/`sbc`.
    Sub = 3,
    /// `[Def d, Use a, Use b, Imm cw]` — `and`.
    And = 4,
    /// `[Def d, Use a, Use b, Imm cw]` — `or`.
    Or = 5,
    /// `[Def d, Use a, Use b, Imm cw]` — `eor`.
    Xor = 6,
    /// `[Def d, Use a, Use b, Imm cw]` — the low `cw` bits of `a * b` with the
    /// hardware `mul` (8×8 → 16 partial products).
    Mul = 7,
    /// `[Def d, Use a, Imm k, Imm cw]` — shift left by the constant `k`.
    ShlC = 8,
    /// `[Def d, Use a, Imm k, Imm cw]` — logical shift right by `k`.
    LshrC = 9,
    /// `[Def d, Use a, Imm k, Imm cw]` — arithmetic shift right by `k`.
    AshrC = 10,
    /// `[Def d, Use a, Use n, Imm cw, Imm ct, (Def t)]` — shift left by the
    /// low byte of `n`: a counted loop, or with `ct` (a secret operand) a
    /// branch-free barrel shifter over the low 4 (3) bits of `n` that clobbers
    /// the fixed pair `t`.
    ShlV = 11,
    /// `[Def d, Use a, Use n, Imm cw, Imm ct, (Def t)]` — logical shift right
    /// by `n` (as [`AvrOp::ShlV`]).
    LshrV = 12,
    /// `[Def d, Use a, Use n, Imm cw, Imm ct, (Def t)]` — arithmetic shift
    /// right by `n` (as [`AvrOp::ShlV`]).
    AshrV = 13,
    /// `[Def d, Use s, Imm from, Imm signed, Imm cw, Imm ct]` — `d` = the low
    /// `from` bits of `s`, zero- or sign-extended to `cw` bits (a sub-byte
    /// sign extension skips with `sbrc`, or with `ct` shifts instead).
    Ext = 14,
    /// `[Def d, Use c]` — `d = c ? 0xffff : 0` for a clean 0/1 `c`.
    Mask = 15,
    /// `[Def d, Use a, Use b, Imm pred, Imm cw, Imm ct]` — `d` = 0 or 1
    /// (whole pair): a skip over an `ldi`, or with `ct` the flag read out of
    /// `SREG`.
    SetCmp = 16,
    /// `[Def d, Use ptr, Imm size, Imm space, Imm atomic]` — load 1 or 2
    /// bytes: `ld` (space 0) or `lpm` (space 1) through `Z`. `atomic` masks
    /// interrupts around a 2-byte access.
    Load = 17,
    /// `[Use ptr, Use val, Imm size, Imm atomic]` — store 1 or 2 bytes.
    Store = 18,
    /// `[Def d, Frame slot, Imm off, Imm size]` — `ldd d, Y+q`.
    LoadSlot = 19,
    /// `[Use v, Frame slot, Imm off, Imm size]` — `std Y+q, v`.
    StoreSlot = 20,
    /// `[Def d, Frame slot]` — the address `Y + q` of a slot.
    FrameAddr = 21,
    /// `[Def d, Global g]` — a global's address (`ldi` + `R_AVR_LO8_LDI` /
    /// `R_AVR_HI8_LDI`).
    GlobalAddr = 22,
    /// `[Def d, Func f]` — a function's word address (`ldi` +
    /// `R_AVR_LO8_LDI_PM` / `R_AVR_HI8_LDI_PM`).
    FuncAddr = 23,
    /// `[Def d, Imm off, Imm size]` — load an incoming stack argument.
    LoadArg = 24,
    /// `[Use v, Imm size]` — push an outgoing stack argument (high byte first).
    PushArg = 25,
    /// `[Imm n]` — pop `n` bytes of stack arguments after a call.
    PopArgs = 26,
    /// `[Func f | Use callee, Def clobbers.., Use args..]` — `call` / `icall`.
    Call = 27,
    /// `[Use results..]` — return (the epilogue is inserted before it).
    Ret = 28,
    /// `[Label t]` — `rjmp` (or `jmp`; nothing when `t` is the next block).
    Jmp = 29,
    /// `[Use c, Label t, Label f]` — branch on the low byte of `c` being nonzero.
    BrCond = 30,
    /// `[Use a, Use b, Imm pred, Imm cw, Label t, Label f]` — compare and branch.
    CmpBr = 31,
    /// `[Use c, Imm cw, Label default, (Imm v, Label case)..]` — multi-way branch.
    Switch = 32,
    /// `[]` — `break` then a self-loop.
    Unreachable = 33,
    /// `[Use v, Frame slot]` — spill a pair.
    StoreFrame = 34,
    /// `[Def d, Frame slot]` — reload a pair.
    LoadFrame = 35,
    /// `[Imm r]` — `push r` (prologue).
    Push = 36,
    /// `[Imm r]` — `pop r` (epilogue).
    Pop = 37,
    /// `[Imm n, Imm set_sp]` — prologue: `Y = SP`, then `Y -= n` and `SP = Y`
    /// (interrupts masked around the `SP` write) when `set_sp`.
    FrameEnter = 38,
    /// `[Imm n]` — epilogue: `Y += n`, `SP = Y` (interrupts masked).
    FrameLeave = 39,
    /// `[Def d, Use n]` — `SP -= n`; `d` = the new `SP + 1`.
    DynAlloca = 40,
    /// `[Def d, Use ptr, Use val, Imm size, Imm op]` — an atomic
    /// read-modify-write (interrupts masked), `d` = the old value.
    AtomicRmw = 41,
    /// `[Def d, Use ptr, Use expected, Use new (a fixed pair), Imm size]` — an
    /// atomic compare-and-exchange, `d` = the old value.
    CmpXchg = 42,
}

impl AvrOp {
    /// The MIR [`Opcode`] of this op.
    #[inline]
    pub fn opcode(self) -> Opcode {
        Opcode(self as u32)
    }

    /// Whether an instruction of this opcode may execute a conditional branch
    /// (or skip) whose direction depends on a register value — the
    /// constant-time audit of the lowering (`docs/ir-design.md` §6d): the
    /// terminators `BrCond`, `CmpBr` and `Switch`; the atomic
    /// compare-and-exchange and `min`/`max` read-modify-writes (which store
    /// conditionally; the verifier rejects secret atomics); and the
    /// **compact** forms of `SetCmp` (a skip), the variable shifts (a counted
    /// loop) and a sub-byte sign extension (`sbrc`). Isel picks those only for
    /// public operands: on a secret-derived operand it emits their
    /// constant-time forms (`ct` set) — the flag read out of `SREG`, a
    /// branch-free barrel shifter, a shift pair — which do not branch.
    /// `Select` is always a mask blend. Division and (without `mul`)
    /// multiplication are calls to the runtime: division of a secret is
    /// rejected by the verifier, and the runtime's 8/16-bit multiplies take a
    /// fixed number of iterations.
    pub fn may_branch_on_data(self, operands: &[MachineOperand]) -> bool {
        let imm_at = |k: usize| match operands.get(k) {
            Some(MachineOperand::Imm(c)) => c.to_u64(),
            _ => None,
        };
        match self {
            AvrOp::BrCond | AvrOp::CmpBr | AvrOp::Switch | AvrOp::CmpXchg => true,
            AvrOp::AtomicRmw => match imm_at(4) {
                Some(c) => matches!(RmwOp::from_code(c), Some(RmwOp::Max | RmwOp::Min | RmwOp::UMax | RmwOp::UMin)),
                None => true,
            },
            // Compact unless `ct` is set (an unknown operand list counts as
            // compact).
            AvrOp::SetCmp => imm_at(5) != Some(1),
            AvrOp::ShlV | AvrOp::LshrV | AvrOp::AshrV => imm_at(4) != Some(1),
            AvrOp::Ext => {
                let sub_byte_signed = imm_at(3) == Some(1) && imm_at(2).is_some_and(|f| f % 8 != 0);
                sub_byte_signed && imm_at(5) != Some(1)
            }
            _ => false,
        }
    }

    /// Decode a MIR [`Opcode`] back to an [`AvrOp`].
    pub fn decode(op: Opcode) -> AvrOp {
        use AvrOp::*;
        const TABLE: [AvrOp; 43] = [
            Mov, Li, Add, Sub, And, Or, Xor, Mul, ShlC, LshrC, AshrC, ShlV, LshrV, AshrV, Ext, Mask, SetCmp,
            Load, Store, LoadSlot, StoreSlot, FrameAddr, GlobalAddr, FuncAddr, LoadArg, PushArg, PopArgs,
            Call, Ret, Jmp, BrCond, CmpBr, Switch, Unreachable, StoreFrame, LoadFrame, Push, Pop, FrameEnter,
            FrameLeave, DynAlloca, AtomicRmw, CmpXchg,
        ];
        TABLE[op.0 as usize]
    }
}

/// A dense code for an [`IntPred`], packed into the `SetCmp`/`CmpBr` immediate.
pub(crate) fn pred_code(p: IntPred) -> u64 {
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

fn def_p(r: PReg) -> MachineOperand {
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
fn inst(op: AvrOp, ops: Vec<MachineOperand>) -> MachineInst {
    MachineInst::new(op.opcode(), ops)
}

/// The container width of a `bits`-wide value: 8 or 16.
#[inline]
pub(crate) fn container(bits: u32) -> u32 {
    if bits <= 8 { 8 } else { 16 }
}

/// Where a call's result goes.
enum RetTo {
    /// Into one vreg (a value of at most 16 bits).
    One(VReg),
    /// Into fresh parts of a wide value (`n` of them).
    Parts(ValueId, usize),
}

/// Per-function isel state that the framework's one-vreg-per-value model does
/// not cover: the parts of wide values, the vregs known to hold zero, and the
/// compares fused into their conditional branch.
#[derive(Debug, Default)]
struct Side {
    parts: DetHashMap<ValueId, Vec<VReg>>,
    zero: DetHashSet<VReg>,
    fused: DetHashSet<ValueId>,
    slots: DetHashMap<ValueId, StackSlot>,
    /// Per value of the function: whether it is secret-derived (empty when
    /// nothing is).
    secret: Vec<bool>,
}

/// The AVR target: register file, ABI, and the isel rules.
#[derive(Debug)]
pub struct AvrTarget {
    rf: RegFile,
    has_mul: bool,
    helpers: super::prepare::Helpers,
    side: RefCell<Side>,
}

impl Default for AvrTarget {
    fn default() -> Self {
        Self::new(true)
    }
}

impl AvrTarget {
    /// The AVR target; `has_mul` says whether the core has the hardware
    /// multiplier (AVR4 and up; AVR5 = ATmega328P has it).
    pub fn new(has_mul: bool) -> AvrTarget {
        AvrTarget { rf: RegFile::new(), has_mul, helpers: Default::default(), side: RefCell::new(Side::default()) }
    }

    /// This target calling `helpers` for the narrow operations it has no
    /// instruction for (see [`super::prepare`]).
    pub(crate) fn with_helpers(mut self, helpers: super::prepare::Helpers) -> AvrTarget {
        self.helpers = helpers;
        self
    }

    /// Whether this target uses the hardware multiplier.
    pub fn has_mul(&self) -> bool {
        self.has_mul
    }

    /// Lower function `func` of `module` (already prepared: see
    /// [`super::prepare`]) to MIR over this target.
    ///
    /// The secret-taint analysis ([`SecretTaint`](crate::analysis::secret::SecretTaint))
    /// runs first: an operation on a secret-derived value gets the
    /// constant-time (branch-free) form of the lowerings that have one, every
    /// other operation the compact form.
    pub fn select(&self, module: &Module, func: crate::ir::FuncId) -> crate::codegen::mir::MachineFunction {
        let taint = crate::analysis::secret::SecretTaint::compute(module, func);
        let secret = if taint.any_secret() { taint.secret_values() } else { Vec::new() };
        *self.side.borrow_mut() = Side { secret, ..Side::default() };
        crate::codegen::isel::select(self, module, func)
    }

    // --- value helpers ------------------------------------------------------

    /// Whether `v` is secret-derived (so its lowerings must be constant-time).
    fn secret(&self, v: ValueId) -> bool {
        self.side.borrow().secret.get(v.index()).copied().unwrap_or(false)
    }

    /// The bit width of an integer or pointer value.
    fn bits(lo: &Lower<'_, Self>, v: ValueId) -> u32 {
        lo.int_width(v)
    }

    fn is_wide(lo: &Lower<'_, Self>, v: ValueId) -> bool {
        Self::bits(lo, v) > 16
    }

    fn const_of(lo: &Lower<'_, Self>, v: ValueId) -> Option<Int> {
        if let ValueDef::Const(c) = lo.func().value(v).def {
            return match lo.module().consts().get(c) {
                Const::Int { value, .. } => Some(value.clone()),
                Const::Null(_) | Const::Poison(_) => Some(Int::ZERO),
                Const::Float { bits, .. } => Some(match bits {
                    crate::ir::FloatBits::F16(b) => Int::from_u64(u64::from(*b)),
                    crate::ir::FloatBits::F32(b) => Int::from_u64(u64::from(*b)),
                    crate::ir::FloatBits::F64(b) => Int::from_u64(*b),
                }),
                _ => None,
            };
        }
        None
    }

    fn is_compare(lo: &Lower<'_, Self>, v: ValueId) -> bool {
        matches!(lo.func().value(v).def, ValueDef::Inst(id) if matches!(lo.func().inst(id).kind, InstKind::ICmp(_)))
    }

    /// The instruction defining `v`, if any.
    fn def_inst<'l>(lo: &'l Lower<'_, Self>, v: ValueId) -> Option<&'l InstData> {
        match lo.func().value(v).def {
            ValueDef::Inst(id) => Some(lo.func().inst(id)),
            _ => None,
        }
    }

    /// The register of a value of at most 16 bits. A function reference used
    /// as a value becomes its word address.
    fn reg(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        debug_assert!(!Self::is_wide(lo, v), "a wide value reached a narrow use");
        if let ValueDef::Func(f) = lo.func().value(v).def {
            let d = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(inst(AvrOp::FuncAddr, vec![def_v(d), MachineOperand::Func(f.index() as u32)]));
            return d;
        }
        lo.reg(v)
    }

    fn konst(&self, lo: &mut Lower<'_, Self>, value: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(inst(AvrOp::Li, vec![def_v(d), imm(value & 0xffff)]));
        if value & 0xffff == 0 {
            self.side.borrow_mut().zero.insert(d);
        }
        d
    }

    fn is_zero(&self, v: VReg) -> bool {
        self.side.borrow().zero.contains(&v)
    }

    /// The 16-bit part vregs of a wide value (a constant is materialized).
    fn parts(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> Vec<VReg> {
        let n = Self::bits(lo, v).div_ceil(16) as usize;
        if let Some(c) = Self::const_of(lo, v) {
            // Constants may be stored signed: take the two's-complement pattern.
            let c = c.mod_2k(16 * n as u32);
            return (0..n).map(|k| {
                let p = c.div_2k_trunc(16 * k as u32).mod_2k(16).to_u64().unwrap_or(0);
                self.konst(lo, p)
            }).collect();
        }
        if let Some(p) = self.side.borrow().parts.get(&v) {
            return p.clone();
        }
        let p: Vec<VReg> = (0..n).map(|_| lo.fresh_vreg(RegClass::Gpr)).collect();
        self.side.borrow_mut().parts.insert(v, p.clone());
        p
    }

    /// Define the parts of the wide value `v` as `computed` (aliasing them,
    /// unless an earlier use already created the part vregs).
    fn def_parts(&self, lo: &mut Lower<'_, Self>, v: ValueId, computed: Vec<VReg>) {
        let existing = self.side.borrow().parts.get(&v).cloned();
        match existing {
            Some(p) => {
                for (d, s) in p.into_iter().zip(computed) {
                    lo.emit(inst(AvrOp::Mov, vec![def_v(d), use_v(s)]));
                }
            }
            None => {
                self.side.borrow_mut().parts.insert(v, computed);
            }
        }
    }

    /// The parts of any integer value as 16-bit vregs: a wide value's parts,
    /// or a narrow value zero- or sign-extended to one or more parts.
    fn parts_ext(&self, lo: &mut Lower<'_, Self>, v: ValueId, n: usize, signed: bool) -> Vec<VReg> {
        let mut p = if Self::is_wide(lo, v) {
            self.parts(lo, v)
        } else {
            vec![self.extend(lo, v, signed, 16)]
        };
        if p.len() < n {
            let fill = if signed {
                let top = *p.last().expect("a part");
                let f = lo.fresh_vreg(RegClass::Gpr);
                lo.emit(inst(AvrOp::AshrC, vec![def_v(f), use_v(top), imm(15), imm(16)]));
                f
            } else {
                self.konst(lo, 0)
            };
            p.resize(n, fill);
        }
        p.truncate(n);
        p
    }

    /// `v` (at most 16 bits) extended from its width to `cw` bits. A value
    /// already that wide, and a compare result (a clean 0/1), are returned as
    /// they are.
    fn extend(&self, lo: &mut Lower<'_, Self>, v: ValueId, signed: bool, cw: u32) -> VReg {
        let bits = Self::bits(lo, v);
        let r = self.reg(lo, v);
        // A full container needs nothing; a compare result is a clean 0/1
        // over the whole pair, so it is already zero-extended.
        if bits >= cw || (!signed && Self::is_compare(lo, v)) {
            return r;
        }
        let ct = self.secret(v);
        self.ext_reg(lo, r, bits, signed, cw, ct)
    }

    fn ext_reg(&self, lo: &mut Lower<'_, Self>, r: VReg, from: u32, signed: bool, cw: u32, ct: bool) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(inst(
            AvrOp::Ext,
            vec![def_v(d), use_v(r), imm(u64::from(from)), imm(u64::from(signed)), imm(u64::from(cw)), imm(u64::from(ct))],
        ));
        d
    }

    /// An `i1` condition as a clean 0/1 in the low byte.
    fn cond(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> VReg {
        self.extend(lo, v, false, 8)
    }

    /// Whether the compare `v` feeds only the conditional branch ending its own
    /// block, so the branch can do the comparison itself.
    fn fusable(lo: &Lower<'_, Self>, v: ValueId) -> bool {
        let ValueDef::Inst(id) = lo.func().value(v).def else { return false };
        if !matches!(lo.func().inst(id).kind, InstKind::ICmp(_)) {
            return false;
        }
        let uses = lo.func().uses_of(v);
        if uses.len() != 1 || uses[0].operand != 0 {
            return false;
        }
        let user = uses[0].inst;
        if !matches!(lo.func().inst(user).kind, InstKind::CondBr { .. }) {
            return false;
        }
        lo.func().blocks().any(|(_, b)| b.terminator() == Some(user) && b.insts().contains(&id))
    }

    /// The operands of a compare, extended as its predicate needs.
    fn cmp_operands(&self, lo: &mut Lower<'_, Self>, pred: IntPred, a: ValueId, b: ValueId) -> (VReg, VReg, u32) {
        let bits = Self::bits(lo, a);
        let cw = container(bits);
        let signed = matches!(pred, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge);
        let ra = self.extend(lo, a, signed, cw);
        let rb = self.extend(lo, b, signed, cw);
        (ra, rb, cw)
    }

    /// The ABI byte size of a value (`i1`/`i8` = 1, pointers = 2, ...).
    fn abi_size(lo: &Lower<'_, Self>, v: ValueId) -> u64 {
        u64::from(Self::bits(lo, v).div_ceil(8))
    }

    /// A value's registers for passing it across a call boundary: its parts,
    /// with an `i1` cleaned to 0/1.
    fn abi_parts(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> Vec<VReg> {
        if Self::is_wide(lo, v) {
            return self.parts(lo, v);
        }
        if Self::bits(lo, v) == 1 {
            return vec![self.cond(lo, v)];
        }
        vec![self.reg(lo, v)]
    }

    // --- per-opcode rules ---------------------------------------------------

    fn lower_wide_bin(&self, lo: &mut Lower<'_, Self>, op: BinOp, i: &InstData) {
        let res = i.result().expect("a binop has a result");
        let (a, b) = (i.operands()[0], i.operands()[1]);
        let n = Self::bits(lo, res).div_ceil(16) as usize;
        let out: Vec<VReg> = match op {
            BinOp::And | BinOp::Or | BinOp::Xor => {
                let pa = self.parts(lo, a);
                let pb = self.parts(lo, b);
                pa.into_iter()
                    .zip(pb)
                    .map(|(x, y)| match (op, self.is_zero(x), self.is_zero(y)) {
                        (BinOp::And, true, _) | (BinOp::And, _, true) => self.konst(lo, 0),
                        (_, true, _) => y,
                        (_, _, true) => x,
                        _ => {
                            let d = lo.fresh_vreg(RegClass::Gpr);
                            let o = match op {
                                BinOp::And => AvrOp::And,
                                BinOp::Or => AvrOp::Or,
                                _ => AvrOp::Xor,
                            };
                            lo.emit(inst(o, vec![def_v(d), use_v(x), use_v(y), imm(16)]));
                            d
                        }
                    })
                    .collect()
            }
            BinOp::Shl | BinOp::LShr | BinOp::AShr => {
                let k = Self::const_of(lo, b).and_then(|c| c.to_u64()).unwrap_or_else(|| {
                    panic!("avr backend: a wide variable shift survived legalization")
                });
                assert!(k % 16 == 0, "avr backend: a wide shift by {k} survived legalization");
                let s = (k / 16) as usize;
                let pa = self.parts(lo, a);
                let mut out = Vec::with_capacity(n);
                match op {
                    BinOp::Shl => {
                        for j in 0..n {
                            out.push(if j < s { self.konst(lo, 0) } else { pa[j - s] });
                        }
                    }
                    _ => {
                        let fill = if op == BinOp::AShr {
                            let f = lo.fresh_vreg(RegClass::Gpr);
                            lo.emit(inst(AvrOp::AshrC, vec![def_v(f), use_v(pa[n - 1]), imm(15), imm(16)]));
                            f
                        } else {
                            self.konst(lo, 0)
                        };
                        for j in 0..n {
                            out.push(if j + s < n { pa[j + s] } else { fill });
                        }
                    }
                }
                out
            }
            other => panic!("avr backend: a wide `{other:?}` survived legalization"),
        };
        self.def_parts(lo, res, out);
    }

    fn lower_bin(&self, lo: &mut Lower<'_, Self>, op: BinOp, i: &InstData) {
        let res = i.result().expect("a binop has a result");
        if Self::is_wide(lo, res) {
            return self.lower_wide_bin(lo, op, i);
        }
        let d = lo.result_reg(i);
        let (a, b) = (i.operands()[0], i.operands()[1]);
        let bits = Self::bits(lo, a);
        let cw = container(bits);
        let simple = match op {
            BinOp::Add => Some(AvrOp::Add),
            BinOp::Sub => Some(AvrOp::Sub),
            BinOp::And => Some(AvrOp::And),
            BinOp::Or => Some(AvrOp::Or),
            BinOp::Xor => Some(AvrOp::Xor),
            BinOp::Mul if self.has_mul => Some(AvrOp::Mul),
            _ => None,
        };
        if matches!(op, BinOp::Mul | BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem) && simple.is_none() {
            // A runtime helper `T f(T, T)` on the container width.
            let f = *self.helpers.get(&(op, cw)).unwrap_or_else(|| {
                panic!("avr backend: no helper for a {cw}-bit `{op:?}` (the module was not prepared)")
            });
            let signed = matches!(op, BinOp::SDiv | BinOp::SRem);
            let (ra, rb) = if op == BinOp::Mul {
                (self.reg(lo, a), self.reg(lo, b))
            } else {
                (self.extend(lo, a, signed, cw), self.extend(lo, b, signed, cw))
            };
            let bytes = u64::from(cw / 8);
            self.emit_call(lo, MachineOperand::Func(f), vec![(vec![ra], bytes), (vec![rb], bytes)], Some((bytes, RetTo::One(d))));
            return;
        }
        if let Some(o) = simple {
            let ra = self.reg(lo, a);
            let rb = self.reg(lo, b);
            lo.emit(inst(o, vec![def_v(d), use_v(ra), use_v(rb), imm(u64::from(cw))]));
            return;
        }
        let (c_op, v_op, signed) = match op {
            BinOp::Shl => (AvrOp::ShlC, AvrOp::ShlV, None),
            BinOp::LShr => (AvrOp::LshrC, AvrOp::LshrV, Some(false)),
            BinOp::AShr => (AvrOp::AshrC, AvrOp::AshrV, Some(true)),
            other => panic!("avr backend: `{other:?}` must be lowered before isel (division is a runtime call)"),
        };
        // A right shift brings the container bits above the width down.
        let ra = match signed {
            Some(s) => self.extend(lo, a, s, cw),
            None => self.reg(lo, a),
        };
        if let Some(k) = Self::const_of(lo, b) {
            // A shift by the width or more is poison: any result will do.
            let k = k.to_u64().unwrap_or(0).min(u64::from(cw) - 1);
            lo.emit(inst(c_op, vec![def_v(d), use_v(ra), imm(k), imm(u64::from(cw))]));
        } else {
            let rb = if Self::bits(lo, b) < 8 { self.extend(lo, b, false, 8) } else { self.reg(lo, b) };
            if self.secret(a) || self.secret(b) {
                // The branch-free shifter needs a scratch pair: a fixed one,
                // so the op keeps three vreg operands (the spill-scratch
                // budget).
                lo.emit(inst(v_op, vec![def_v(d), use_v(ra), use_v(rb), imm(u64::from(cw)), imm(1), def_p(pair(18))]));
            } else {
                lo.emit(inst(v_op, vec![def_v(d), use_v(ra), use_v(rb), imm(u64::from(cw)), imm(0)]));
            }
        }
    }

    fn lower_cast(&self, lo: &mut Lower<'_, Self>, op: CastOp, i: &InstData) {
        let res = i.result().expect("a cast has a result");
        let src = i.operands()[0];
        let to_bits = Self::bits(lo, res);
        let wide_src = Self::is_wide(lo, src);
        if to_bits > 16 {
            let n = to_bits.div_ceil(16) as usize;
            let p = match op {
                CastOp::SExt => self.parts_ext(lo, src, n, true),
                CastOp::ZExt | CastOp::PtrToInt | CastOp::Trunc | CastOp::Bitcast => {
                    self.parts_ext(lo, src, n, false)
                }
                other => panic!("avr backend: unsupported wide cast `{other:?}` (floats are lowered to integers first)"),
            };
            return self.def_parts(lo, res, p);
        }
        let d = lo.result_reg(i);
        let s = match op {
            CastOp::ZExt | CastOp::SExt | CastOp::IntToPtr if !wide_src => {
                let signed = op == CastOp::SExt;
                self.extend(lo, src, signed, container(to_bits))
            }
            _ if wide_src => self.parts(lo, src)[0],
            CastOp::Trunc | CastOp::PtrToInt | CastOp::Bitcast | CastOp::ZExt | CastOp::SExt | CastOp::IntToPtr => {
                self.reg(lo, src)
            }
            other => panic!("avr backend: unsupported cast `{other:?}` (floats are lowered to integers first)"),
        };
        lo.emit(inst(AvrOp::Mov, vec![def_v(d), use_v(s)]));
    }

    /// If `ptr` is the result of an `alloca` already lowered, its slot.
    fn alloca_slot(&self, ptr: ValueId) -> Option<StackSlot> {
        self.side.borrow().slots.get(&ptr).copied()
    }

    fn lower_load(&self, lo: &mut Lower<'_, Self>, i: &InstData, ty: crate::ir::TypeId, atomic: bool) {
        let res = i.result().expect("a load has a result");
        let ptr_v = i.operands()[0];
        let space = u64::from(lo.mem_addr_space(i).unwrap_or(0));
        let ptr = self.reg(lo, ptr_v);
        if Self::is_wide(lo, res) {
            let n = Self::bits(lo, res).div_ceil(16) as usize;
            let mut parts = Vec::with_capacity(n);
            for k in 0..n {
                let p = if k == 0 { ptr } else { self.ptr_plus(lo, ptr, 2 * k as u64) };
                let d = lo.fresh_vreg(RegClass::Gpr);
                lo.emit(inst(AvrOp::Load, vec![def_v(d), use_v(p), imm(2), imm(space), imm(u64::from(atomic))]));
                parts.push(d);
            }
            return self.def_parts(lo, res, parts);
        }
        let size = lo.byte_size(ty);
        assert!(size <= 2, "avr backend: a {size}-byte load of a non-integer type is not supported");
        let d = lo.result_reg(i);
        lo.emit(inst(AvrOp::Load, vec![def_v(d), use_v(ptr), imm(size), imm(space), imm(u64::from(atomic))]));
    }

    fn lower_store(&self, lo: &mut Lower<'_, Self>, i: &InstData, ty: crate::ir::TypeId, atomic: bool) {
        let (ptr_v, val_v) = (i.operands()[0], i.operands()[1]);
        if lo.mem_addr_space(i).unwrap_or(0) != 0 {
            panic!("avr backend: program memory (address space 1) is read-only");
        }
        let ptr = self.reg(lo, ptr_v);
        if Self::is_wide(lo, val_v) {
            let parts = self.parts(lo, val_v);
            for (k, v) in parts.into_iter().enumerate() {
                let p = if k == 0 { ptr } else { self.ptr_plus(lo, ptr, 2 * k as u64) };
                lo.emit(inst(AvrOp::Store, vec![use_v(p), use_v(v), imm(2), imm(u64::from(atomic))]));
            }
            return;
        }
        let size = lo.byte_size(ty);
        assert!(size <= 2, "avr backend: a {size}-byte store of a non-integer type is not supported");
        let v = self.reg(lo, val_v);
        lo.emit(inst(AvrOp::Store, vec![use_v(ptr), use_v(v), imm(size), imm(u64::from(atomic))]));
    }

    fn ptr_plus(&self, lo: &mut Lower<'_, Self>, ptr: VReg, k: u64) -> VReg {
        let c = self.konst(lo, k);
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(inst(AvrOp::Add, vec![def_v(d), use_v(ptr), use_v(c), imm(16)]));
        d
    }

    /// Emit a call: push the stack arguments, move the register arguments,
    /// `call`, pop, and move the result out. `args` are each argument's
    /// registers (its 16-bit parts) with its ABI byte size.
    fn emit_call(&self, lo: &mut Lower<'_, Self>, callee: MachineOperand, args: Vec<(Vec<VReg>, u64)>, ret: Option<(u64, RetTo)>) {
        let sizes: Vec<u64> = args.iter().map(|a| a.1).collect();
        let (locs, stack_bytes) = regs::assign_args(&sizes);
        // Stack arguments: pushed last byte first, so the first one ends up at
        // the lowest address.
        for (k, loc) in locs.iter().enumerate().rev() {
            if let ArgLoc::Stack(_) = loc {
                let size = sizes[k];
                for (j, &p) in args[k].0.iter().enumerate().rev() {
                    let bytes = (size - 2 * j as u64).min(2);
                    lo.emit(inst(AvrOp::PushArg, vec![use_v(p), imm(bytes)]));
                }
            }
        }
        // Register arguments: one run of moves right before the call.
        let mut used: Vec<PReg> = Vec::new();
        for (k, loc) in locs.iter().enumerate() {
            if let ArgLoc::Regs(base) = *loc {
                for (j, &p) in args[k].0.iter().enumerate() {
                    let r = pair(base + 2 * j as u8);
                    lo.emit(inst(AvrOp::Mov, vec![def_p(r), use_v(p)]));
                    used.push(r);
                }
            }
        }
        let mut operands = vec![callee];
        for &c in &self.rf.caller_saved {
            operands.push(def_p(c));
        }
        for r in used {
            operands.push(use_p(r));
        }
        lo.emit(inst(AvrOp::Call, operands));
        if stack_bytes > 0 {
            lo.emit(inst(AvrOp::PopArgs, vec![imm(stack_bytes)]));
        }
        if let Some((bytes, to)) = ret {
            let base = regs::ret_base(bytes);
            match to {
                RetTo::One(d) => lo.emit(inst(AvrOp::Mov, vec![def_v(d), use_p(pair(base))])),
                RetTo::Parts(v, n) => {
                    let p: Vec<VReg> = (0..n)
                        .map(|j| {
                            let d = lo.fresh_vreg(RegClass::Gpr);
                            lo.emit(inst(AvrOp::Mov, vec![def_v(d), use_p(pair(base + 2 * j as u8))]));
                            d
                        })
                        .collect();
                    self.def_parts(lo, v, p);
                }
            }
        }
    }

    fn lower_call(&self, lo: &mut Lower<'_, Self>, i: &InstData) {
        let ops = i.operands();
        let callee = ops[0];
        let args: Vec<(Vec<VReg>, u64)> =
            ops[1..].iter().map(|&a| (self.abi_parts(lo, a), Self::abi_size(lo, a))).collect();
        let target = match lo.callee_func(callee) {
            Some(f) => MachineOperand::Func(f),
            None => use_v(self.reg(lo, callee)),
        };
        let ret = i.result().map(|res| {
            let bytes = Self::abi_size(lo, res);
            let to = if Self::is_wide(lo, res) {
                RetTo::Parts(res, Self::bits(lo, res).div_ceil(16) as usize)
            } else {
                RetTo::One(lo.result_reg(i))
            };
            (bytes, to)
        });
        self.emit_call(lo, target, args, ret);
    }

    fn lower_atomic(&self, lo: &mut Lower<'_, Self>, i: &InstData) {
        let ops = i.operands();
        let narrow = |lo: &Lower<'_, Self>, ty| {
            let size = lo.byte_size(ty);
            assert!(size <= 2, "avr backend: {size}-byte atomics are not supported (at most 16 bits)");
            size
        };
        match &i.kind {
            InstKind::AtomicLoad { ty, .. } => {
                narrow(lo, *ty);
                self.lower_load(lo, i, *ty, true);
            }
            InstKind::AtomicStore { ty, .. } => {
                narrow(lo, *ty);
                self.lower_store(lo, i, *ty, true);
            }
            InstKind::AtomicRmw { op, ty, .. } => {
                let size = narrow(lo, *ty);
                let d = lo.result_reg(i);
                let ptr = self.reg(lo, ops[0]);
                // `max`/`min` compare whole containers: extend the operand
                // (the loaded value is a full byte or pair already).
                let val = match op {
                    RmwOp::Max | RmwOp::Min => self.extend(lo, ops[1], true, 8 * size as u32),
                    RmwOp::UMax | RmwOp::UMin => self.extend(lo, ops[1], false, 8 * size as u32),
                    _ => self.reg(lo, ops[1]),
                };
                lo.emit(inst(
                    AvrOp::AtomicRmw,
                    vec![def_v(d), use_v(ptr), use_v(val), imm(size), imm(u64::from(op.code()))],
                ));
            }
            InstKind::CmpXchg { ty, .. } => {
                let size = narrow(lo, *ty);
                let d = lo.result_reg(i);
                let ptr = self.reg(lo, ops[0]);
                let exp = self.extend(lo, ops[1], false, 8 * size as u32);
                let new = self.reg(lo, ops[2]);
                // `new` travels in a fixed pair, keeping the op at three vreg
                // operands (the spill-scratch budget).
                let fixed = pair(18);
                lo.emit(inst(AvrOp::Mov, vec![def_p(fixed), use_v(new)]));
                lo.emit(inst(AvrOp::CmpXchg, vec![def_v(d), use_v(ptr), use_v(exp), use_p(fixed), imm(size)]));
            }
            // A single core with in-order memory: a fence orders nothing the
            // instruction order does not already (and MIR keeps that order).
            InstKind::Fence(_) => {}
            other => unreachable!("lower_atomic on {other:?}"),
        }
    }

    fn lower_select(&self, lo: &mut Lower<'_, Self>, i: &InstData) {
        let res = i.result().expect("select has a result");
        let ops = i.operands();
        let c = self.cond(lo, ops[0]);
        let m = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(inst(AvrOp::Mask, vec![def_v(m), use_v(c)]));
        // d = f ^ ((t ^ f) & mask), branch-free, three operands per op.
        let blend = |lo: &mut Lower<'_, Self>, t: VReg, f: VReg, d: VReg| {
            let x = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(inst(AvrOp::Xor, vec![def_v(x), use_v(t), use_v(f), imm(16)]));
            let y = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(inst(AvrOp::And, vec![def_v(y), use_v(x), use_v(m), imm(16)]));
            lo.emit(inst(AvrOp::Xor, vec![def_v(d), use_v(f), use_v(y), imm(16)]));
        };
        if Self::is_wide(lo, res) {
            let pt = self.parts(lo, ops[1]);
            let pf = self.parts(lo, ops[2]);
            let out: Vec<VReg> = pt
                .into_iter()
                .zip(pf)
                .map(|(t, f)| {
                    let d = lo.fresh_vreg(RegClass::Gpr);
                    blend(lo, t, f, d);
                    d
                })
                .collect();
            return self.def_parts(lo, res, out);
        }
        let d = lo.result_reg(i);
        let t = self.reg(lo, ops[1]);
        let f = self.reg(lo, ops[2]);
        blend(lo, t, f, d);
    }

    /// A `switch` on a wide value: reduce it to a narrow case index first
    /// (`Σ (x == v_k) · (k + 1)`, the cases being distinct), then switch on
    /// that.
    fn wide_switch_index(&self, lo: &mut Lower<'_, Self>, cond: ValueId, values: &[Int]) -> VReg {
        let parts = self.parts(lo, cond);
        let total = 16 * parts.len() as u32;
        let mut idx = self.konst(lo, 0);
        for (k, v) in values.iter().enumerate() {
            let v = &v.mod_2k(total);
            let mut all: Option<VReg> = None;
            for (j, &p) in parts.iter().enumerate() {
                let pv = v.div_2k_trunc(16 * j as u32).mod_2k(16).to_u64().unwrap_or(0);
                let c = self.konst(lo, pv);
                let e = lo.fresh_vreg(RegClass::Gpr);
                // A switch condition is never secret (the verifier forbids
                // it): the compact compare.
                lo.emit(inst(AvrOp::SetCmp, vec![def_v(e), use_v(p), use_v(c), imm(pred_code(IntPred::Eq)), imm(16), imm(0)]));
                all = Some(match all {
                    None => e,
                    Some(a) => {
                        let n = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(inst(AvrOp::And, vec![def_v(n), use_v(a), use_v(e), imm(16)]));
                        n
                    }
                });
            }
            let e = all.expect("a wide value has parts");
            let m = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(inst(AvrOp::Mask, vec![def_v(m), use_v(e)]));
            let kv = self.konst(lo, k as u64 + 1);
            let t = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(inst(AvrOp::And, vec![def_v(t), use_v(m), use_v(kv), imm(16)]));
            let n = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(inst(AvrOp::Or, vec![def_v(n), use_v(idx), use_v(t), imm(16)]));
            idx = n;
        }
        idx
    }
}

impl MachineTarget for AvrTarget {
    fn name(&self) -> &str {
        "avr"
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
            RegClass::Fp => &self.rf.empty,
        }
    }

    fn scratch(&self, class: RegClass) -> &[PReg] {
        match class {
            RegClass::Gpr => &self.rf.scratch,
            RegClass::Fp => &self.rf.empty,
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
            AvrOp::decode(op),
            AvrOp::Jmp | AvrOp::BrCond | AvrOp::CmpBr | AvrOp::Switch | AvrOp::Ret | AvrOp::Unreachable
        )
    }

    fn is_move(&self, op: Opcode) -> bool {
        AvrOp::decode(op) == AvrOp::Mov
    }

    fn emit_move(&self, dst: Reg, src: Reg) -> MachineInst {
        inst(AvrOp::Mov, vec![MachineOperand::Def(dst), MachineOperand::Use(src)])
    }

    fn emit_spill(&self, slot: StackSlot, src: PReg) -> MachineInst {
        inst(AvrOp::StoreFrame, vec![use_p(src), MachineOperand::Frame(slot)])
    }

    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst {
        inst(AvrOp::LoadFrame, vec![def_p(dst), MachineOperand::Frame(slot)])
    }
}

impl TargetIsel for AvrTarget {
    fn li(&self, dst: VReg, value: Int) -> MachineInst {
        // Not recorded as a known zero: the framework also uses `li` for the
        // edge copies into block parameters, which have several definitions.
        let v = value.mod_2k(16).to_u64().unwrap_or(0);
        inst(AvrOp::Li, vec![def_v(dst), imm(v)])
    }

    fn jump(&self, dst: MBlockId) -> MachineInst {
        inst(AvrOp::Jmp, vec![MachineOperand::Label(dst)])
    }

    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst {
        inst(AvrOp::FrameAddr, vec![def_v(dst), MachineOperand::Frame(slot)])
    }

    fn global_addr(&self, dst: VReg, g: u32) -> MachineInst {
        inst(AvrOp::GlobalAddr, vec![def_v(dst), MachineOperand::Global(g)])
    }

    fn lower_prologue(&self, lo: &mut Lower<'_, Self>) {
        let entry = lo.func().entry().expect("a definition has an entry block");
        let params: Vec<ValueId> = lo.func().block(entry).params().to_vec();
        let sizes: Vec<u64> = params.iter().map(|&p| Self::abi_size(lo, p)).collect();
        let (locs, _) = regs::assign_args(&sizes);
        let mut stack_loads = Vec::new();
        for ((&p, loc), &size) in params.iter().zip(&locs).zip(&sizes) {
            let n = Self::bits(lo, p).div_ceil(16) as usize;
            let wide = Self::is_wide(lo, p);
            let targets: Vec<VReg> = if wide {
                self.parts(lo, p)
            } else {
                vec![lo.reg(p)]
            };
            debug_assert_eq!(targets.len(), n.max(1));
            match *loc {
                ArgLoc::Regs(base) => {
                    for (j, &t) in targets.iter().enumerate() {
                        lo.emit(inst(AvrOp::Mov, vec![def_v(t), use_p(pair(base + 2 * j as u8))]));
                    }
                }
                ArgLoc::Stack(off) => {
                    for (j, &t) in targets.iter().enumerate() {
                        let bytes = (size - 2 * j as u64).min(2);
                        stack_loads.push(inst(AvrOp::LoadArg, vec![def_v(t), imm(off + 2 * j as u64), imm(bytes)]));
                    }
                }
            }
        }
        for s in stack_loads {
            lo.emit(s);
        }
    }

    fn lower_inst(&self, lo: &mut Lower<'_, Self>, i: &InstData) {
        match &i.kind {
            InstKind::Bin(op) => self.lower_bin(lo, *op, i),
            InstKind::ICmp(pred) => {
                let res = i.result().expect("icmp has a result");
                if Self::fusable(lo, res) {
                    self.side.borrow_mut().fused.insert(res);
                    return;
                }
                let d = lo.result_reg(i);
                let ct = self.secret(i.operands()[0]) || self.secret(i.operands()[1]);
                let (a, b, cw) = self.cmp_operands(lo, *pred, i.operands()[0], i.operands()[1]);
                lo.emit(inst(
                    AvrOp::SetCmp,
                    vec![def_v(d), use_v(a), use_v(b), imm(pred_code(*pred)), imm(u64::from(cw)), imm(u64::from(ct))],
                ));
            }
            InstKind::Cast(op) => self.lower_cast(lo, *op, i),
            InstKind::Alloca { elem_ty } => {
                let d = lo.result_reg(i);
                let size = lo.byte_size(*elem_ty);
                let slot = lo.new_slot(size, 1);
                self.side.borrow_mut().slots.insert(i.result().expect("alloca result"), slot);
                lo.emit(self.frame_addr(d, slot));
            }
            InstKind::DynAlloca { .. } => {
                let d = lo.result_reg(i);
                let n = i.operands()[0];
                let n = if Self::is_wide(lo, n) { self.parts(lo, n)[0] } else { self.extend(lo, n, false, 16) };
                lo.emit(inst(AvrOp::DynAlloca, vec![def_v(d), use_v(n)]));
            }
            InstKind::Load { ty, .. } => {
                if let Some(slot) = self.alloca_slot(i.operands()[0])
                    && !Self::is_wide(lo, i.result().expect("load result"))
                {
                    let size = lo.byte_size(*ty);
                    let d = lo.result_reg(i);
                    lo.emit(inst(AvrOp::LoadSlot, vec![def_v(d), MachineOperand::Frame(slot), imm(0), imm(size)]));
                    return;
                }
                self.lower_load(lo, i, *ty, false);
            }
            InstKind::Store { ty, .. } => {
                if let Some(slot) = self.alloca_slot(i.operands()[0])
                    && !Self::is_wide(lo, i.operands()[1])
                {
                    let size = lo.byte_size(*ty);
                    let v = self.reg(lo, i.operands()[1]);
                    lo.emit(inst(AvrOp::StoreSlot, vec![use_v(v), MachineOperand::Frame(slot), imm(0), imm(size)]));
                    return;
                }
                self.lower_store(lo, i, *ty, false);
            }
            InstKind::PtrAdd { .. } => {
                let d = lo.result_reg(i);
                let base = self.reg(lo, i.operands()[0]);
                let off_v = i.operands()[1];
                let off = if Self::is_wide(lo, off_v) {
                    self.parts(lo, off_v)[0]
                } else {
                    self.extend(lo, off_v, true, 16)
                };
                lo.emit(inst(AvrOp::Add, vec![def_v(d), use_v(base), use_v(off), imm(16)]));
            }
            InstKind::Select => self.lower_select(lo, i),
            InstKind::Freeze | InstKind::Declassify => {
                let res = i.result().expect("freeze and declassify have a result");
                if Self::is_wide(lo, res) {
                    let p = self.parts(lo, i.operands()[0]);
                    return self.def_parts(lo, res, p);
                }
                let d = lo.result_reg(i);
                let s = self.reg(lo, i.operands()[0]);
                lo.emit(inst(AvrOp::Mov, vec![def_v(d), use_v(s)]));
            }
            InstKind::Call => self.lower_call(lo, i),
            InstKind::Syscall => panic!("avr backend: `syscall` has no meaning on a bare-metal AVR"),
            InstKind::InlineAsm(_) | InstKind::AsmOutput(_) => {
                panic!("avr backend: {}", crate::codegen::INLINE_ASM_UNSUPPORTED)
            }
            InstKind::Unary(_) | InstKind::FCmp(_) => {
                panic!("avr backend: floating point must be lowered to runtime calls first (see avr::prepare)")
            }
            k if k.is_atomic() => self.lower_atomic(lo, i),
            _ => unreachable!("terminator reached lower_inst: {:?}", i.kind),
        }
    }

    fn lower_term(&self, lo: &mut Lower<'_, Self>, i: &InstData) {
        match &i.kind {
            InstKind::Ret => {
                let mut uses = Vec::new();
                if let Some(&v) = i.operands().first() {
                    let bytes = Self::abi_size(lo, v);
                    let base = regs::ret_base(bytes);
                    let parts = self.abi_parts(lo, v);
                    for (j, p) in parts.into_iter().enumerate() {
                        let r = pair(base + 2 * j as u8);
                        lo.emit(inst(AvrOp::Mov, vec![def_p(r), use_v(p)]));
                        uses.push(use_p(r));
                    }
                }
                lo.emit(inst(AvrOp::Ret, uses));
            }
            InstKind::Br(target) => {
                let args: Vec<_> = i.operands().to_vec();
                let e = lo.edge_to(*target, &args);
                lo.emit(self.jump(e));
            }
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let ops = i.operands();
                let c = ops[0];
                let fused = self.side.borrow().fused.contains(&c);
                let head = if fused {
                    let ci = Self::def_inst(lo, c).expect("a fused compare").clone();
                    let InstKind::ICmp(pred) = ci.kind else { unreachable!() };
                    let (a, b, cw) = self.cmp_operands(lo, pred, ci.operands()[0], ci.operands()[1]);
                    vec![use_v(a), use_v(b), imm(pred_code(pred)), imm(u64::from(cw))]
                } else {
                    vec![use_v(self.cond(lo, c))]
                };
                let tb = 1 + *true_args as usize;
                let fb = tb + *false_args as usize;
                let tv: Vec<_> = ops[1..tb].to_vec();
                let fv: Vec<_> = ops[tb..fb].to_vec();
                let te = lo.edge_to(*if_true, &tv);
                let fe = lo.edge_to(*if_false, &fv);
                let mut operands = head;
                operands.push(MachineOperand::Label(te));
                operands.push(MachineOperand::Label(fe));
                lo.emit(inst(if fused { AvrOp::CmpBr } else { AvrOp::BrCond }, operands));
            }
            InstKind::Switch(data) => {
                let ops = i.operands();
                let cond = ops[0];
                let bits = Self::bits(lo, cond);
                let values: Vec<Int> = data.cases.iter().map(|c| c.value.clone()).collect();
                let (c, cw, keys): (VReg, u32, Vec<u64>) = if bits > 16 {
                    let idx = self.wide_switch_index(lo, cond, &values);
                    (idx, 16, (1..=values.len() as u64).collect())
                } else {
                    let cw = container(bits);
                    let c = self.extend(lo, cond, false, cw);
                    (c, cw, values.iter().map(|v| v.mod_2k(bits).to_u64().unwrap_or(0)).collect())
                };
                let mut idx = 1usize;
                let dcount = data.default_args as usize;
                let dv: Vec<_> = ops[idx..idx + dcount].to_vec();
                idx += dcount;
                let de = lo.edge_to(data.default, &dv);
                let mut operands = vec![use_v(c), imm(u64::from(cw)), MachineOperand::Label(de)];
                let cases = data.cases.clone();
                for (case, key) in cases.iter().zip(keys) {
                    let n = case.args as usize;
                    let cv: Vec<_> = ops[idx..idx + n].to_vec();
                    idx += n;
                    let ce = lo.edge_to(case.target, &cv);
                    operands.push(imm(key));
                    operands.push(MachineOperand::Label(ce));
                }
                lo.emit(inst(AvrOp::Switch, operands));
            }
            InstKind::Unreachable => lo.emit(inst(AvrOp::Unreachable, Vec::new())),
            _ => unreachable!("non-terminator reached lower_term: {:?}", i.kind),
        }
    }
}
