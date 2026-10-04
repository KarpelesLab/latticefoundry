//! The opcode table and instruction representation.
//!
//! Each opcode below carries, in its doc comment, its **reference semantics**:
//! the value it computes and the exact conditions under which its result is
//! **poison**. This is bet B1 (the opcode table *is* a formal semantics): the
//! prose here is the specification the executable evaluator (next task) and the
//! `z3rs`-backed verifier are checked against, so spec and implementation cannot
//! drift. See `docs/design-tenets.md` T2/B1 and `docs/ir-design.md`.
//!
//! An [`InstData`] holds an [`InstKind`] (opcode plus immediate/structural data
//! such as predicates, cast kinds, and branch targets) and a flat `operands`
//! list of [`ValueId`]s. Keeping *all* value references in one flat list is what
//! makes use/def bookkeeping and `replace_all_uses_with` uniform: a use is just
//! `(inst, operand_index)` regardless of opcode. The meaning of each operand
//! slot is fixed per opcode and documented below.
//!
//! Flag model (`docs/ir-design.md` §7): a single [`Flags`] value carries the
//! integer `nsw`/`nuw`/`exact` assumptions and the float fast-math set. **A flag
//! is an assumption that licenses optimization; violating it yields _poison_,
//! not undefined behavior.** This keeps the refinement relation total.

use crate::ir::BlockId;
use crate::ir::types::TypeId;
use crate::ir::value::ValueId;

/// A `Copy` handle to an [`InstData`] within a function's instruction arena.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct InstId(u32);

impl InstId {
    /// The dense index this id addresses.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub(crate) fn from_index(i: usize) -> Self {
        InstId(i as u32)
    }
}

/// One use of a value: the instruction that references it and which operand
/// slot does so. The def→use lists are keyed by [`ValueId`]; each entry is one
/// of these. `func.insts[use.inst].operands()[use.operand]` is guaranteed to be
/// the value whose use list contains this entry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Use {
    /// The instruction doing the referencing.
    pub inst: InstId,
    /// The operand slot within that instruction.
    pub operand: u32,
}

/// IEEE-754 fast-math relaxations. Each is an assumption; if it does not in fact
/// hold at runtime the result is **poison**. They license reassociation and
/// algebraic simplification the strict semantics forbid.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FastMath {
    /// Assume arguments and results are never NaN.
    pub nnan: bool,
    /// Assume arguments and results are never ±∞.
    pub ninf: bool,
    /// Treat the sign of a zero result as insignificant.
    pub nsz: bool,
    /// Permit reassociation of floating-point operations.
    pub reassoc: bool,
    /// Permit contraction of operations (e.g. fusing a multiply-add).
    pub contract: bool,
    /// Permit approximate implementations of library functions.
    pub afn: bool,
}

impl FastMath {
    /// Whether any fast-math relaxation is set.
    pub fn any(self) -> bool {
        self != FastMath::default()
    }
}

/// The unified instruction flag model (`docs/ir-design.md` §7).
///
/// Only the flags meaningful to a given opcode are honored; the builder sets the
/// rest to `false`. Violating any set flag makes the instruction's result
/// poison.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Flags {
    /// `nsw` — assume no signed wrap (add/sub/mul/shl). Signed overflow ⇒ poison.
    pub nsw: bool,
    /// `nuw` — assume no unsigned wrap (add/sub/mul/shl). Unsigned overflow ⇒ poison.
    pub nuw: bool,
    /// `exact` — assume an exact division/shift (udiv/sdiv/lshr/ashr). A nonzero
    /// remainder, or shifting out a set bit, ⇒ poison.
    pub exact: bool,
    /// Floating-point fast-math relaxations.
    pub fast: FastMath,
}

impl Flags {
    /// No flags set (the strict, always-defined interpretation).
    pub const NONE: Flags = Flags { nsw: false, nuw: false, exact: false, fast: FastMath { nnan: false, ninf: false, nsz: false, reassoc: false, contract: false, afn: false } };

    /// Flags with `nsw` set.
    pub fn nsw() -> Flags {
        Flags { nsw: true, ..Flags::NONE }
    }

    /// Flags with `nuw` set.
    pub fn nuw() -> Flags {
        Flags { nuw: true, ..Flags::NONE }
    }

    /// Flags with `exact` set.
    pub fn exact() -> Flags {
        Flags { exact: true, ..Flags::NONE }
    }

    /// Flags carrying the given fast-math set.
    pub fn fast(fast: FastMath) -> Flags {
        Flags { fast, ..Flags::NONE }
    }
}

/// A two-operand arithmetic/bitwise/shift operation. Operands are
/// `[lhs, rhs]`; both operands and the result share the instruction's type
/// (`i1`-and-wider integers, or a float type for the `F*` variants).
///
/// If either operand is poison, the result is poison. Additional poison
/// conditions are noted per variant and per flag (see [`Flags`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    /// Integer addition (two's-complement, wrapping unless `nsw`/`nuw`).
    Add,
    /// Integer subtraction (wrapping unless `nsw`/`nuw`).
    Sub,
    /// Integer multiplication (wrapping unless `nsw`/`nuw`).
    Mul,
    /// Unsigned division. Division by zero is **undefined behavior** (the
    /// quotient has no defined value); `exact` and a nonzero remainder ⇒ poison.
    UDiv,
    /// Signed division. Division by zero, or `INT_MIN / -1` (whose quotient is
    /// not representable), is **undefined behavior**; `exact` and a nonzero
    /// remainder ⇒ poison.
    SDiv,
    /// Unsigned remainder. Division by zero is **undefined behavior**.
    URem,
    /// Signed remainder. Division by zero, or `INT_MIN % -1` (paired with the
    /// overflowing division), is **undefined behavior**.
    SRem,
    /// Bitwise and.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise exclusive-or.
    Xor,
    /// Left shift. A shift amount ≥ the bit width is **poison**. `nsw`/`nuw`
    /// constrain the shifted-out bits as for `mul` by a power of two.
    Shl,
    /// Logical (unsigned) right shift, zero-filling. Shift amount ≥ width ⇒
    /// poison; `exact` and a shifted-out set bit ⇒ poison.
    LShr,
    /// Arithmetic (signed) right shift, sign-filling. Shift amount ≥ width ⇒
    /// poison; `exact` and a shifted-out set bit ⇒ poison.
    AShr,
    /// Floating-point addition (IEEE-754, honoring fast-math flags).
    FAdd,
    /// Floating-point subtraction.
    FSub,
    /// Floating-point multiplication.
    FMul,
    /// Floating-point division.
    FDiv,
    /// Floating-point remainder (IEEE remainder / `frem`).
    FRem,
    /// Signed minimum. No flags; poison only from a poison operand. (With
    /// the saturating ops below, a SIMD staple; valid on scalars and vectors.)
    SMin,
    /// Signed maximum.
    SMax,
    /// Unsigned minimum.
    UMin,
    /// Unsigned maximum.
    UMax,
    /// Signed saturating addition: the exact sum clamped to the signed range.
    SAddSat,
    /// Unsigned saturating addition: the exact sum clamped to `2ⁿ - 1`.
    UAddSat,
    /// Signed saturating subtraction: the exact difference clamped to the
    /// signed range.
    SSubSat,
    /// Unsigned saturating subtraction: the exact difference clamped at 0.
    USubSat,
}

impl BinOp {
    /// Whether this is a floating-point operation (result and operands are a
    /// float type and fast-math flags apply).
    pub fn is_float(self) -> bool {
        matches!(self, BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem)
    }

    /// Whether this is one of the min/max or saturating integer ops, which
    /// backends without a direct form get expanded into compares, selects and
    /// ordinary arithmetic (`crate::codegen::legalize`).
    pub fn is_minmax_sat(self) -> bool {
        matches!(
            self,
            BinOp::SMin
                | BinOp::SMax
                | BinOp::UMin
                | BinOp::UMax
                | BinOp::SAddSat
                | BinOp::UAddSat
                | BinOp::SSubSat
                | BinOp::USubSat
        )
    }
}

/// A single-operand operation. Operand is `[val]`; result shares its type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    /// Floating-point negation (flips the sign bit; honors fast-math `nsz`).
    FNeg,
}

/// Integer comparison predicate for `icmp`. The result is `i1`: `1` if the
/// relation holds, `0` otherwise. If either operand is poison, the result is
/// poison. Signedness is a property of the predicate, not the operand type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntPred {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Unsigned greater than.
    Ugt,
    /// Unsigned greater than or equal.
    Uge,
    /// Unsigned less than.
    Ult,
    /// Unsigned less than or equal.
    Ule,
    /// Signed greater than.
    Sgt,
    /// Signed greater than or equal.
    Sge,
    /// Signed less than.
    Slt,
    /// Signed less than or equal.
    Sle,
}

/// Floating-point comparison predicate for `fcmp`. Result is `i1`.
///
/// *Ordered* predicates (`O*`) are true only if neither operand is NaN and the
/// relation holds; *unordered* predicates (`U*`) are true if either operand is
/// NaN, or the relation holds. `Ord` is "neither is NaN"; `Uno` is "either is
/// NaN". If either operand is poison, the result is poison.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FloatPred {
    /// Always false.
    False,
    /// Ordered and equal.
    Oeq,
    /// Ordered and greater than.
    Ogt,
    /// Ordered and greater than or equal.
    Oge,
    /// Ordered and less than.
    Olt,
    /// Ordered and less than or equal.
    Ole,
    /// Ordered and not equal.
    One,
    /// Ordered (neither operand is NaN).
    Ord,
    /// Unordered or equal.
    Ueq,
    /// Unordered or greater than.
    Ugt,
    /// Unordered or greater than or equal.
    Uge,
    /// Unordered or less than.
    Ult,
    /// Unordered or less than or equal.
    Ule,
    /// Unordered or not equal.
    Une,
    /// Unordered (at least one operand is NaN).
    Uno,
    /// Always true.
    True,
}

/// A conversion (`cast`) opcode. Operand is `[val]`; the result type is the
/// instruction's type. If the operand is poison, the result is poison.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CastOp {
    /// Truncate an integer to a narrower integer (drops high bits).
    Trunc,
    /// Zero-extend an integer to a wider integer.
    ZExt,
    /// Sign-extend an integer to a wider integer.
    SExt,
    /// Truncate a float to a narrower float format (rounds).
    FpTrunc,
    /// Extend a float to a wider float format (exact).
    FpExt,
    /// Convert a float to an unsigned integer, rounding toward zero. A value out
    /// of the integer's range is **poison**.
    FpToUi,
    /// Convert a float to a signed integer, rounding toward zero. Out of range ⇒
    /// poison.
    FpToSi,
    /// Convert an unsigned integer to a float (rounds to nearest).
    UiToFp,
    /// Convert a signed integer to a float (rounds to nearest).
    SiToFp,
    /// Reinterpret a pointer as an integer of the pointer's address width.
    PtrToInt,
    /// Reinterpret an integer as a pointer.
    IntToPtr,
    /// Reinterpret the bits of a value as another type of the same bit width.
    Bitcast,
}

/// One arm of a [`switch`](InstKind::Switch): if the condition equals `value`,
/// control transfers to `target`, passing that edge's block arguments.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SwitchCase {
    /// The integer value this arm matches (interpreted in the condition's type).
    pub value: puremp::Int,
    /// The successor block for this arm.
    pub target: BlockId,
    /// How many of the instruction's operands are this edge's block arguments.
    pub args: u32,
}

/// The structural payload of a [`switch`](InstKind::Switch) terminator.
///
/// Operand layout: `[cond, <default args>, <case0 args>, <case1 args>, ...]`.
/// The condition is operand `0`; the remaining operands are the block arguments
/// for each outgoing edge, in the order default-then-cases, sliced by the
/// recorded arg counts.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SwitchData {
    /// The block taken when no case matches.
    pub default: BlockId,
    /// How many operands (after the condition) are the default edge's arguments.
    pub default_args: u32,
    /// The match arms, in order.
    pub cases: Vec<SwitchCase>,
}

/// The memory ordering of an atomic operation or fence (`docs/ir-design.md`
/// §6b). The names and meanings follow the C11/C++11 memory model; the IR has
/// no weaker "unordered" level, so `relaxed` is the weakest.
///
/// Every ordering makes the access itself **atomic** (indivisible: no torn
/// reads or writes, a single total modification order per location). The
/// ordering adds *inter-thread* constraints, which in turn bound code motion
/// (the reference evaluator is single-threaded, so it only sees the sequential
/// meaning; the constraints below are what optimizations must respect):
///
/// - [`Relaxed`](AtomicOrdering::Relaxed) — atomicity only; no ordering with
///   other memory operations.
/// - [`Acquire`](AtomicOrdering::Acquire) — (loads, rmw, cmpxchg, fences) no
///   memory operation that follows in program order may be performed before it.
/// - [`Release`](AtomicOrdering::Release) — (stores, rmw, cmpxchg, fences) no
///   memory operation that precedes it in program order may be performed after
///   it.
/// - [`AcqRel`](AtomicOrdering::AcqRel) — (rmw, cmpxchg, fences) both.
/// - [`SeqCst`](AtomicOrdering::SeqCst) — acquire + release, plus a single total
///   order over all `seq_cst` operations.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum AtomicOrdering {
    /// `relaxed` — atomic, but unordered with respect to other locations.
    Relaxed,
    /// `acquire` — later memory operations stay after it.
    Acquire,
    /// `release` — earlier memory operations stay before it.
    Release,
    /// `acq_rel` — both acquire and release.
    AcqRel,
    /// `seq_cst` — acquire + release + one global order of `seq_cst` ops.
    SeqCst,
}

impl AtomicOrdering {
    /// Every ordering, weakest first (also the binary encoding order).
    pub const ALL: [AtomicOrdering; 5] = [
        AtomicOrdering::Relaxed,
        AtomicOrdering::Acquire,
        AtomicOrdering::Release,
        AtomicOrdering::AcqRel,
        AtomicOrdering::SeqCst,
    ];

    /// Whether this ordering has acquire semantics (`acquire`, `acq_rel`,
    /// `seq_cst`): no later memory operation may be moved above it.
    pub fn is_acquire(self) -> bool {
        matches!(self, AtomicOrdering::Acquire | AtomicOrdering::AcqRel | AtomicOrdering::SeqCst)
    }

    /// Whether this ordering has release semantics (`release`, `acq_rel`,
    /// `seq_cst`): no earlier memory operation may be moved below it.
    pub fn is_release(self) -> bool {
        matches!(self, AtomicOrdering::Release | AtomicOrdering::AcqRel | AtomicOrdering::SeqCst)
    }

    /// The textual spelling (`relaxed`, `acquire`, `release`, `acq_rel`,
    /// `seq_cst`).
    pub fn name(self) -> &'static str {
        match self {
            AtomicOrdering::Relaxed => "relaxed",
            AtomicOrdering::Acquire => "acquire",
            AtomicOrdering::Release => "release",
            AtomicOrdering::AcqRel => "acq_rel",
            AtomicOrdering::SeqCst => "seq_cst",
        }
    }

    /// Parse a textual spelling (see [`AtomicOrdering::name`]).
    pub fn from_name(s: &str) -> Option<AtomicOrdering> {
        AtomicOrdering::ALL.into_iter().find(|o| o.name() == s)
    }

    /// This ordering's position in [`AtomicOrdering::ALL`] (a stable small code).
    pub fn code(self) -> u8 {
        AtomicOrdering::ALL.iter().position(|&x| x == self).expect("every ordering is listed") as u8
    }

    /// The ordering with the given [`AtomicOrdering::code`].
    pub fn from_code(c: u64) -> Option<AtomicOrdering> {
        usize::try_from(c).ok().and_then(|i| AtomicOrdering::ALL.get(i).copied())
    }

    /// Whether this ordering is allowed on an `atomic_load` (a load cannot
    /// release: `relaxed`, `acquire`, `seq_cst`). Also the set allowed as a
    /// `cmpxchg` failure ordering (the failure path is a pure load).
    pub fn valid_for_load(self) -> bool {
        !matches!(self, AtomicOrdering::Release | AtomicOrdering::AcqRel)
    }

    /// Whether this ordering is allowed on an `atomic_store` (a store cannot
    /// acquire: `relaxed`, `release`, `seq_cst`).
    pub fn valid_for_store(self) -> bool {
        !matches!(self, AtomicOrdering::Acquire | AtomicOrdering::AcqRel)
    }

    /// Whether this ordering is allowed on a `fence` (anything but `relaxed`,
    /// which would order nothing).
    pub fn valid_for_fence(self) -> bool {
        self != AtomicOrdering::Relaxed
    }
}

/// The read-modify-write operation of an [`atomic_rmw`](InstKind::AtomicRmw).
/// Each computes the new memory contents from the old value `old` and the
/// operand `v` (both of the accessed type, arithmetic wrapping); the
/// instruction returns `old`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RmwOp {
    /// `v` (exchange / swap). The only op also allowed on `ptr`.
    Xchg,
    /// `old + v` (wrapping).
    Add,
    /// `old - v` (wrapping).
    Sub,
    /// `old & v`.
    And,
    /// `!(old & v)`.
    Nand,
    /// `old | v`.
    Or,
    /// `old ^ v`.
    Xor,
    /// Signed maximum of `old` and `v`.
    Max,
    /// Signed minimum of `old` and `v`.
    Min,
    /// Unsigned maximum of `old` and `v`.
    UMax,
    /// Unsigned minimum of `old` and `v`.
    UMin,
}

impl RmwOp {
    /// Every operation (also the binary encoding order).
    pub const ALL: [RmwOp; 11] = [
        RmwOp::Xchg,
        RmwOp::Add,
        RmwOp::Sub,
        RmwOp::And,
        RmwOp::Nand,
        RmwOp::Or,
        RmwOp::Xor,
        RmwOp::Max,
        RmwOp::Min,
        RmwOp::UMax,
        RmwOp::UMin,
    ];

    /// The textual spelling (`xchg`, `add`, `sub`, `and`, `nand`, `or`, `xor`,
    /// `max`, `min`, `umax`, `umin`).
    pub fn name(self) -> &'static str {
        match self {
            RmwOp::Xchg => "xchg",
            RmwOp::Add => "add",
            RmwOp::Sub => "sub",
            RmwOp::And => "and",
            RmwOp::Nand => "nand",
            RmwOp::Or => "or",
            RmwOp::Xor => "xor",
            RmwOp::Max => "max",
            RmwOp::Min => "min",
            RmwOp::UMax => "umax",
            RmwOp::UMin => "umin",
        }
    }

    /// Parse a textual spelling (see [`RmwOp::name`]).
    pub fn from_name(s: &str) -> Option<RmwOp> {
        RmwOp::ALL.into_iter().find(|o| o.name() == s)
    }

    /// This operation's position in [`RmwOp::ALL`] (a stable small code, used
    /// as a machine-instruction immediate by the backends).
    pub fn code(self) -> u8 {
        RmwOp::ALL.iter().position(|&x| x == self).expect("every rmw op is listed") as u8
    }

    /// The operation with the given [`RmwOp::code`].
    pub fn from_code(c: u64) -> Option<RmwOp> {
        usize::try_from(c).ok().and_then(|i| RmwOp::ALL.get(i).copied())
    }

    /// The sequential meaning on `width`-bit two's-complement values given as
    /// their unsigned bit patterns (`old`, `v` < 2^width, `width` ≤ 64): the
    /// new memory contents, masked to `width` bits.
    pub fn apply(self, old: u64, v: u64, width: u32) -> u64 {
        let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
        let sext = |x: u64| -> i64 {
            let sh = 64 - width.min(64);
            ((x << sh) as i64) >> sh
        };
        let r = match self {
            RmwOp::Xchg => v,
            RmwOp::Add => old.wrapping_add(v),
            RmwOp::Sub => old.wrapping_sub(v),
            RmwOp::And => old & v,
            RmwOp::Nand => !(old & v),
            RmwOp::Or => old | v,
            RmwOp::Xor => old ^ v,
            RmwOp::Max => if sext(old) >= sext(v) { old } else { v },
            RmwOp::Min => if sext(old) <= sext(v) { old } else { v },
            RmwOp::UMax => old.max(v),
            RmwOp::UMin => old.min(v),
        };
        r & mask
    }
}

/// The combining operation of a [`reduce`](InstKind::Reduce) over the lanes of
/// a vector. Integer reductions are associative and commutative, so their
/// result does not depend on the order lanes are combined in; the float ones
/// are **ordered**: `((l0 op l1) op l2) op …`, lane 0 first, each step rounded
/// (a `reassoc` fast-math flag licenses any other order).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ReduceOp {
    /// Wrapping integer sum.
    Add,
    /// Wrapping integer product.
    Mul,
    /// Bitwise and.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise exclusive-or.
    Xor,
    /// Signed minimum.
    SMin,
    /// Signed maximum.
    SMax,
    /// Unsigned minimum.
    UMin,
    /// Unsigned maximum.
    UMax,
    /// Ordered floating-point sum (`fadd` lane 0 to lane N-1).
    FAdd,
    /// Ordered floating-point product (`fmul` lane 0 to lane N-1).
    FMul,
}

impl ReduceOp {
    /// Every operation (also the binary encoding order).
    pub const ALL: [ReduceOp; 11] = [
        ReduceOp::Add,
        ReduceOp::Mul,
        ReduceOp::And,
        ReduceOp::Or,
        ReduceOp::Xor,
        ReduceOp::SMin,
        ReduceOp::SMax,
        ReduceOp::UMin,
        ReduceOp::UMax,
        ReduceOp::FAdd,
        ReduceOp::FMul,
    ];

    /// The textual spelling (`add`, `mul`, `and`, `or`, `xor`, `smin`, `smax`,
    /// `umin`, `umax`, `fadd`, `fmul`).
    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Add => "add",
            ReduceOp::Mul => "mul",
            ReduceOp::And => "and",
            ReduceOp::Or => "or",
            ReduceOp::Xor => "xor",
            ReduceOp::SMin => "smin",
            ReduceOp::SMax => "smax",
            ReduceOp::UMin => "umin",
            ReduceOp::UMax => "umax",
            ReduceOp::FAdd => "fadd",
            ReduceOp::FMul => "fmul",
        }
    }

    /// Parse a textual spelling (see [`ReduceOp::name`]).
    pub fn from_name(s: &str) -> Option<ReduceOp> {
        ReduceOp::ALL.into_iter().find(|o| o.name() == s)
    }

    /// This operation's position in [`ReduceOp::ALL`] (a stable small code).
    pub fn code(self) -> u8 {
        ReduceOp::ALL.iter().position(|&x| x == self).expect("every reduce op is listed") as u8
    }

    /// The operation with the given [`ReduceOp::code`].
    pub fn from_code(c: u64) -> Option<ReduceOp> {
        usize::try_from(c).ok().and_then(|i| ReduceOp::ALL.get(i).copied())
    }

    /// Whether this is a floating-point reduction (`fadd`/`fmul`).
    pub fn is_float(self) -> bool {
        matches!(self, ReduceOp::FAdd | ReduceOp::FMul)
    }
}

/// One output operand of an [`inline_asm`](InstKind::InlineAsm): its GCC-style
/// constraint (`"=r"`, `"=&a"`, `"+m"`, ...), an optional symbolic name for
/// `%[name]` references in the template, and — for a **register output** —
/// the type of the value it produces. An **indirect** (memory) output, whose
/// constraint allows only memory ([`InlineAsm::is_indirect`]), produces no
/// value: the asm writes through the pointer operand that stands for it, and
/// `ty` is `None`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct AsmOutput {
    /// The constraint string, starting with `=` (write-only) or `+` (read-write).
    pub constraint: String,
    /// The `[name]` the template may use for this operand.
    pub name: Option<String>,
    /// The produced value's type (register outputs), `None` for an indirect one.
    pub ty: Option<TypeId>,
}

/// One input operand of an [`inline_asm`](InstKind::InlineAsm): its constraint
/// (`"r"`, `"a"`, `"m"`, `"i"`, `"0"`, ...) and optional symbolic name.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct AsmInput {
    /// The constraint string (no `=`/`+` prefix).
    pub constraint: String,
    /// The `[name]` the template may use for this operand.
    pub name: Option<String>,
}

/// The payload of an [`inline_asm`](InstKind::InlineAsm) instruction: a GCC
/// extended-asm statement (`docs/ir-design.md` §6i).
///
/// Operands are numbered as in GCC: the outputs `0..outputs.len()`, then the
/// inputs. The **value operands** of the instruction are, in order: one per
/// output that needs one (an indirect output's pointer, or a `+` register
/// output's incoming value), then one per input (an indirect input's pointer,
/// or the input value). See [`InlineAsm::operand_slots`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct InlineAsm {
    /// The assembler template, with `%0`, `%[name]`, `%k1`, `%=`, `%%` ...
    /// still unsubstituted (each target's lowering expands them).
    pub template: String,
    /// The output operands.
    pub outputs: Vec<AsmOutput>,
    /// The input operands.
    pub inputs: Vec<AsmInput>,
    /// The clobber list: register names, `"memory"`, `"cc"`.
    pub clobbers: Vec<String>,
    /// `volatile` (GCC's `asm volatile`, LLVM's `sideeffect`): the asm has
    /// effects beyond its outputs and must run exactly where written.
    pub volatile: bool,
}

/// Which asm operand one value operand of an
/// [`inline_asm`](InstKind::InlineAsm) stands for (see
/// [`InlineAsm::operand_slots`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AsmSlot {
    /// Output `i`'s value operand: the pointer of an indirect output, or the
    /// incoming value of a `+` register output.
    Output(usize),
    /// Input `i`'s value operand.
    Input(usize),
}

impl InlineAsm {
    /// A constraint with its leading modifiers (`=`, `+`, `&`, `%`) removed.
    pub fn constraint_body(c: &str) -> &str {
        c.trim_start_matches(['=', '+', '&', '%'])
    }

    /// Whether a constraint allows only memory (`m`, `o`, `V`, possibly with
    /// modifiers and alternatives), so its operand is a pointer to the memory
    /// rather than a value. A constraint that also allows a register or an
    /// immediate (`"rm"`, `"g"`) is a value operand.
    pub fn is_indirect(c: &str) -> bool {
        let mut body = c.chars().filter(|ch| !matches!(ch, '=' | '+' | '&' | '%' | ',' | '*' | '?' | '!')).peekable();
        body.peek().is_some() && body.all(|ch| matches!(ch, 'm' | 'o' | 'V'))
    }

    /// Whether a constraint allows only an immediate (`i`, `n`, or a
    /// target's immediate-range letter): its operand must be a constant (or,
    /// for `i`, a symbol address).
    pub fn is_immediate_only(c: &str) -> bool {
        let body = Self::constraint_body(c);
        !body.is_empty()
            && body.chars().all(|ch| matches!(ch, 'i' | 'n' | 's' | 'I' | 'J' | 'K' | 'L' | 'M' | 'N' | 'e' | 'Z'))
    }

    /// The output an input constraint is tied to (a matching constraint `"0"`,
    /// `"1"`, ... or `"[name]"`), if it is one.
    pub fn tied_output(&self, c: &str) -> Option<usize> {
        let body = Self::constraint_body(c);
        if !body.is_empty() && body.chars().all(|ch| ch.is_ascii_digit()) {
            return body.parse().ok();
        }
        let name = body.strip_prefix('[')?.strip_suffix(']')?;
        self.outputs.iter().position(|o| o.name.as_deref() == Some(name))
    }

    /// Whether output `i` produces a value (a register output).
    pub fn is_register_output(&self, i: usize) -> bool {
        self.outputs.get(i).is_some_and(|o| !Self::is_indirect(&o.constraint))
    }

    /// The register outputs, in order (the instruction's result is the first;
    /// `asm_output` reads the others).
    pub fn register_outputs(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.outputs.len()).filter(|&i| self.is_register_output(i))
    }

    /// The output whose value is the instruction's own result: the first
    /// register output, if any.
    pub fn result_output(&self) -> Option<usize> {
        self.register_outputs().next()
    }

    /// Which asm operand each of the instruction's value operands stands for,
    /// in operand order.
    pub fn operand_slots(&self) -> Vec<AsmSlot> {
        let mut slots = Vec::new();
        for (i, o) in self.outputs.iter().enumerate() {
            if Self::is_indirect(&o.constraint) || o.constraint.starts_with('+') {
                slots.push(AsmSlot::Output(i));
            }
        }
        slots.extend((0..self.inputs.len()).map(AsmSlot::Input));
        slots
    }

    /// Whether the asm clobbers memory (`"memory"` in the clobber list).
    pub fn clobbers_memory(&self) -> bool {
        self.clobbers.iter().any(|c| c == "memory")
    }

    /// Whether the asm may read or write memory: it clobbers memory or has an
    /// indirect (memory) operand.
    pub fn may_access_memory(&self) -> bool {
        self.clobbers_memory()
            || self.outputs.iter().any(|o| Self::is_indirect(&o.constraint))
            || self.inputs.iter().any(|i| Self::is_indirect(&i.constraint))
    }

    /// Whether the asm is an opaque effect (as strong as a call to an unknown
    /// function) rather than a pure function of its inputs: it is `volatile`,
    /// or it may access memory.
    pub fn has_side_effect(&self) -> bool {
        self.volatile || self.may_access_memory()
    }
}

/// An opcode together with its immediate/structural data.
///
/// Value operands live in [`InstData::operands`], *not* here; this carries only
/// the non-value payload (predicates, cast kinds, accessed types, alignments,
/// branch targets, edge arities). The operand-slot meaning for each opcode is
/// documented on the opcode; branch/switch args are read via the accessors on
/// [`InstData`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum InstKind {
    /// Binary op; operands `[lhs, rhs]`. Result type = instruction type.
    Bin(BinOp),
    /// Unary op; operand `[val]`. Result type = instruction type.
    Unary(UnaryOp),
    /// Integer compare; operands `[lhs, rhs]`. Result type = `i1`.
    ICmp(IntPred),
    /// Float compare; operands `[lhs, rhs]`. Result type = `i1`.
    FCmp(FloatPred),
    /// Conversion; operand `[val]`. Result type = instruction type.
    Cast(CastOp),

    /// Allocate stack space for one value of `elem_ty`; no operands. The result
    /// is a fresh, suitably aligned, non-null pointer (of pointer type) valid
    /// for the lifetime of the enclosing function activation. Reading the newly
    /// allocated memory before it is written yields poison.
    Alloca {
        /// The type whose storage is allocated.
        elem_ty: TypeId,
    },
    /// Dynamically allocate `n` bytes of stack storage; operand `[n]` is an
    /// integer byte count. The result is a fresh, `align`-aligned, non-null
    /// pointer (of pointer type) to `n` bytes of freshly allocated stack, valid
    /// for the lifetime of the enclosing function activation — the C `alloca`
    /// lifetime: the storage is reclaimed when the function returns, **not** at
    /// scope exit (so `dyn_alloca` inside a loop accumulates until return).
    /// Reading the newly allocated memory before it is written yields poison.
    /// A negative or unrepresentably-large `n` is treated as `Alloca` treats an
    /// out-of-model size (the resulting pointer is poison). `align` is the
    /// required alignment, a power of two.
    DynAlloca {
        /// The required alignment of the returned pointer, in bytes (a power of
        /// two).
        align: u32,
    },
    /// Load a value of the accessed type from memory; operand `[ptr]`. Result
    /// type = the accessed type. The accessed type and alignment live on the op
    /// (this is where opaque pointers put the type back). Loading through a
    /// poison or dangling pointer, or with insufficient alignment, is undefined
    /// behavior; loading uninitialized memory yields poison.
    ///
    /// A **volatile** load (`load volatile`) is an observable event in its own
    /// right (memory-mapped I/O): it is performed exactly once, at exactly the
    /// accessed type's width, in program order relative to every other volatile
    /// access, atomic, fence, call and syscall. It is never removed (even
    /// unused), duplicated, merged, widened or narrowed, hoisted or sunk,
    /// promoted to a register, or forwarded from a store, and its result is
    /// unknown to every analysis.
    ///
    /// A **secret** load (`load secret`) reads memory the front end declared
    /// secret (Lode's `secret[T]`): its result is secret-derived for the
    /// constant-time discipline (`docs/ir-design.md` §6d). The flag has no
    /// effect on the value semantics; it only feeds the secret-taint analysis
    /// ([`crate::analysis::secret`]) and the constant-time verifier.
    Load {
        /// The type read from memory (the result type).
        ty: TypeId,
        /// The assumed alignment of the access, in bytes (a power of two).
        align: u32,
        /// Whether the access is volatile (see above).
        volatile: bool,
        /// Whether the loaded value is secret (see above).
        secret: bool,
    },
    /// Store a value to memory; operands `[ptr, value]`. No result. Storing
    /// through a poison/dangling pointer or under-aligned is undefined behavior.
    /// A **volatile** store (`store volatile`) obeys the same rules as a volatile
    /// load: performed exactly once, at exactly its width, in order, and never
    /// removed, not even when a later store overwrites it.
    ///
    /// A **secret** store (`store secret`) writes memory the front end declared
    /// secret. It is how a secret-derived value may be written through a
    /// pointer that is not a function-local stack slot (a parameter, a loaded
    /// pointer, a non-secret global): the constant-time verifier rejects an
    /// unflagged store of a secret-derived value there (§6d). No effect on the
    /// value semantics.
    Store {
        /// The type written to memory (the type of the stored value).
        ty: TypeId,
        /// The assumed alignment of the access, in bytes (a power of two).
        align: u32,
        /// Whether the access is volatile (see [`InstKind::Load`]).
        volatile: bool,
        /// Whether the destination is secret memory (see above).
        secret: bool,
    },
    /// Atomic load; operand `[ptr]`, result type = `ty` (`i8`/`i16`/`i32`/`i64`
    /// or `ptr`). Sequentially it reads `ty` from `ptr` exactly like `load`; the
    /// access is indivisible and ordered per `ordering` (`relaxed`, `acquire` or
    /// `seq_cst`; see [`AtomicOrdering`]). `align` must be at least the type's
    /// size (natural alignment); an address that is not so aligned, poison, or
    /// dangling is undefined behavior. Never removed, duplicated, or merged;
    /// its result is unknown to every analysis.
    AtomicLoad {
        /// The type read (the result type).
        ty: TypeId,
        /// The alignment of the access, in bytes (≥ the type's size).
        align: u32,
        /// The memory ordering.
        ordering: AtomicOrdering,
    },
    /// Atomic store; operands `[ptr, value]`, no result. Sequentially a `store`
    /// of `ty`; indivisible and ordered per `ordering` (`relaxed`, `release` or
    /// `seq_cst`). Alignment and address rules as for
    /// [`AtomicLoad`](InstKind::AtomicLoad). Never removed.
    AtomicStore {
        /// The type written (the type of the stored value).
        ty: TypeId,
        /// The alignment of the access, in bytes (≥ the type's size).
        align: u32,
        /// The memory ordering.
        ordering: AtomicOrdering,
    },
    /// Atomic read-modify-write; operands `[ptr, v]`, result type = `ty`.
    /// Indivisibly reads `old` from `ptr`, writes `op(old, v)` (see [`RmwOp`])
    /// and returns `old`. `ty` is `i8`/`i16`/`i32`/`i64`, or also `ptr` for
    /// `xchg`. Any ordering is allowed. A poison `v` stores poison (the result
    /// is still the loaded `old`). Alignment and address rules as for
    /// [`AtomicLoad`](InstKind::AtomicLoad). Never removed (even with an unused
    /// result).
    AtomicRmw {
        /// The modification applied.
        op: RmwOp,
        /// The accessed type (the type of `v` and of the result).
        ty: TypeId,
        /// The alignment of the access, in bytes (≥ the type's size).
        align: u32,
        /// The memory ordering.
        ordering: AtomicOrdering,
    },
    /// Atomic compare-and-exchange (strong); operands `[ptr, expected, new]`,
    /// result type = `ty` (`i8`/`i16`/`i32`/`i64` or `ptr`). Indivisibly reads
    /// `old` from `ptr`; if `old == expected` (bitwise) it writes `new`. Returns
    /// `old`. It never fails spuriously, so the exchange happened **iff** the
    /// result equals `expected`: front ends obtain the success flag as
    /// `icmp eq %old, %expected` (instructions have a single result; see
    /// `docs/ir-design.md` §6b). `success` orders the read-modify-write when the
    /// exchange happens (any ordering); `failure` orders the plain load when it
    /// does not (`relaxed`, `acquire` or `seq_cst`). A poison `expected` or
    /// loaded value is undefined behavior (the comparison would branch on
    /// poison); a poison `new` stores poison. Never removed.
    CmpXchg {
        /// The accessed type (of `expected`, `new` and the result).
        ty: TypeId,
        /// The alignment of the access, in bytes (≥ the type's size).
        align: u32,
        /// The ordering when the exchange happens.
        success: AtomicOrdering,
        /// The ordering when it does not (a load ordering).
        failure: AtomicOrdering,
    },
    /// Memory fence; no operands, no result. Establishes its ordering
    /// (`acquire`, `release`, `acq_rel` or `seq_cst`; `relaxed` is rejected)
    /// between the memory operations around it without accessing memory
    /// itself. It has no sequential effect, but it is a code-motion barrier:
    /// never removed, and no memory operation may cross it in a direction its
    /// ordering forbids (`seq_cst` and `acq_rel` forbid both).
    Fence(AtomicOrdering),
    /// Pointer displacement; operands `[base, byte_offset]`, result is a pointer.
    /// Computes `base + byte_offset` as a byte address — this replaces
    /// `getelementptr`; structured addressing is a builder convenience that
    /// lowers to this (see `struct_field`/`array_elem`). If `inbounds` is set
    /// and the result leaves the allocation `base` points into, the result is
    /// **poison**. If either operand is poison, the result is poison.
    PtrAdd {
        /// Whether the result must stay within `base`'s allocation.
        inbounds: bool,
    },

    /// Ternary select; operands `[cond, if_true, if_false]`, `cond` is `i1`.
    /// Result = `if_true` when `cond` is 1, else `if_false`; result type =
    /// instruction type. If `cond` is poison, the result is poison; a
    /// non-selected poison operand does not taint the result.
    Select,
    /// `freeze`; operand `[val]`. If the operand is poison, produce an arbitrary
    /// but **fixed, consistent** concrete value of the type; otherwise produce
    /// the operand unchanged. This is the only way to remove poison. Result type
    /// = instruction type. (There is no `undef`.)
    Freeze,
    /// `declassify`; operand `[val]`. The identity on values (poison stays
    /// poison): result = the operand, result type = instruction type = the
    /// operand's type. It is the **explicit end of secrecy** (Lode's escape
    /// hatch for `secret[T]`): its result is public for the constant-time
    /// discipline (`docs/ir-design.md` §6d) even when its operand is
    /// secret-derived, so it may then be branched on, used as an address or
    /// divided. A pass must never replace a `declassify` result by its operand
    /// (that would re-expose the secret); folding it to a *constant* is fine.
    Declassify,
    /// Function call; operands `[callee, args...]`. `callee` is a function
    /// reference or a pointer; the rest are the arguments in order. Result type
    /// = the callee's return type (`void` for a procedure). Effects and poison
    /// propagation follow the callee's semantics.
    Call,
    /// Operating-system call (Linux syscall ABI); operands `[nr, args...]` with
    /// 0..=6 arguments. Every operand is `i64` or `ptr` — front ends extend
    /// narrower integers themselves (no implicit extension rule), so the value
    /// placed in each ABI register is exactly the operand's 64 bits. Result type
    /// = `i64`: the raw kernel return, **uninterpreted** (Linux reports failure
    /// as `-errno` in `[-4095, -1]`; the op does not decode it).
    ///
    /// Semantics: an opaque effect on the outside world, exactly as strong as a
    /// call to an unknown external function — it may read or write any memory
    /// reachable from an escaped pointer (operands included), so it is a full
    /// memory clobber; it may not be removed (even with an unused result),
    /// duplicated, reordered with other memory ops/calls/syscalls, hoisted, or
    /// speculated. Its result is unknown to every analysis. A poison operand is
    /// undefined behavior (the kernel would observe an arbitrary register).
    Syscall,
    /// GCC-style inline assembly (`docs/ir-design.md` §6i); the payload holds
    /// the template, constraints, clobbers and `volatile` flag, and the value
    /// operands are laid out per [`InlineAsm::operand_slots`]. The result is
    /// the first register output (type = that output's type), or none (`void`)
    /// when there is no register output; the other register outputs are read
    /// by [`AsmOutput`](InstKind::AsmOutput) projections.
    ///
    /// Semantics: the template is opaque. A `volatile` asm, or one that
    /// clobbers memory or has a memory operand, is an effect exactly as strong
    /// as a call to an unknown function (a full memory clobber that escapes
    /// its pointer operands; never removed, duplicated, reordered with other
    /// memory operations, calls or syscalls, hoisted or speculated). Any other
    /// asm is a **pure** function of its inputs: it may be removed when no
    /// output is used, but it is still never hoisted or speculated (its
    /// template may trap). Every output is unknown to every analysis. A poison
    /// operand is undefined behavior (the asm would observe an arbitrary
    /// register).
    InlineAsm(Box<InlineAsm>),
    /// Read register output `n` of an `inline_asm`; operand `[asm]` is that
    /// instruction's result (which names the whole asm). The result type is
    /// output `n`'s type. Pure: it only projects a value the asm produced.
    AsmOutput(u32),

    // --- SIMD vectors (`docs/ir-design.md` §6e) -----------------------------
    //
    // The ordinary value ops above — `Bin`, `Unary`, `ICmp`, `FCmp`, `Cast`,
    // `Select`, `Freeze` — also apply to vectors, **lane-wise**: lane `i` of the
    // result is the scalar op on lane `i` of each operand, with the scalar
    // op's poison rule applied per lane (an over-wide shift amount in lane 2
    // poisons only lane 2) and its UB rule applied to the whole instruction (a
    // zero divisor in any lane is UB). `icmp`/`fcmp` on `<N x T>` produce
    // `<N x i1>`; `select` takes an `i1` (whole-vector choice) or an
    // `<N x i1>` (per-lane choice) condition. A `bitcast` reinterprets the bits
    // of a whole value, lanes packed lane 0 lowest; a poison lane poisons every
    // result lane (or the whole scalar) it overlaps. `load`/`store` move a whole
    // vector (lanes at consecutive element-sized offsets).
    /// Extract one lane; operand `[vec]`, result type = the element type. The
    /// lane index is an immediate, which the verifier checks is `< N`. The
    /// result is the lane's value (poison iff that lane is poison).
    ExtractElement {
        /// The lane read.
        lane: u32,
    },
    /// Replace one lane; operands `[vec, elem]`, result type = the vector
    /// type. Lane `lane` of the result is `elem`; every other lane is `vec`'s.
    /// Only the replaced lane can become poison from a poison `elem`, and a
    /// poison `vec` leaves the replaced lane defined.
    InsertElement {
        /// The lane written.
        lane: u32,
    },
    /// Shuffle two vectors; operands `[a, b]`, both `<N x T>`; the result is
    /// `<M x T>` with `M = mask.len()`. Result lane `i` is lane `mask[i]` of the
    /// concatenation `a ++ b` (indices `0..N` pick from `a`, `N..2N` from `b`).
    /// The mask is a constant (the verifier checks every index is `< 2N`);
    /// each result lane is poison iff the lane it picks is.
    ShuffleVector(Box<[u32]>),
    /// Broadcast a scalar; operand `[scalar]` of type `T`, result `<N x T>`
    /// (the instruction type) with every lane equal to it (all lanes poison iff
    /// the scalar is).
    Splat,
    /// Reduce the lanes of a vector to one scalar; operand `[vec]` of type
    /// `<N x T>`, result `T`. See [`ReduceOp`] for the combining order. Any
    /// poison lane makes the result poison. The float reductions honor the
    /// instruction's fast-math flags exactly as a chain of `fadd`/`fmul` would.
    Reduce(ReduceOp),

    // --- terminators --------------------------------------------------------
    /// Return; operands `[value]` for a value-returning function, or `[]` for a
    /// `void` return. Ends the function activation.
    Ret,
    /// Unconditional branch to `target`; operands are the edge's block arguments
    /// (all of them), matching `target`'s parameter list by position and type.
    Br(BlockId),
    /// Conditional branch; operand layout `[cond, <true args>, <false args>]`.
    /// `cond` is `i1`. Branches to `if_true` (passing the true args) when `cond`
    /// is 1, else to `if_false` (passing the false args). Branching on poison is
    /// undefined behavior.
    CondBr {
        /// Successor taken when the condition is 1.
        if_true: BlockId,
        /// Successor taken when the condition is 0.
        if_false: BlockId,
        /// Number of leading post-condition operands that are the true edge's args.
        true_args: u32,
        /// Number of trailing operands that are the false edge's args.
        false_args: u32,
    },
    /// Multi-way branch on an integer; see [`SwitchData`] for the operand
    /// layout. Transfers to the matching case's target, or to the default.
    /// Switching on poison is undefined behavior.
    Switch(Box<SwitchData>),
    /// Marks unreachable control flow; no operands, no successors. Reaching it
    /// at runtime is undefined behavior (it asserts the path is dead).
    Unreachable,
}

impl InstKind {
    /// Whether this opcode is an atomic memory operation or a fence
    /// (`atomic_load`, `atomic_store`, `atomic_rmw`, `cmpxchg`, `fence`).
    pub fn is_atomic(&self) -> bool {
        matches!(
            self,
            InstKind::AtomicLoad { .. }
                | InstKind::AtomicStore { .. }
                | InstKind::AtomicRmw { .. }
                | InstKind::CmpXchg { .. }
                | InstKind::Fence(_)
        )
    }

    /// Whether this opcode is one of the vector-only operations
    /// (`extractelement`, `insertelement`, `shufflevector`, `splat`, `reduce`).
    pub fn is_vector_op(&self) -> bool {
        matches!(
            self,
            InstKind::ExtractElement { .. }
                | InstKind::InsertElement { .. }
                | InstKind::ShuffleVector(_)
                | InstKind::Splat
                | InstKind::Reduce(_)
        )
    }

    /// Whether this opcode is a volatile `load` or `store`.
    pub fn is_volatile(&self) -> bool {
        matches!(self, InstKind::Load { volatile: true, .. } | InstKind::Store { volatile: true, .. })
    }

    /// Whether this instruction must be kept even when its result is unused,
    /// because it does more than produce a value: every store, call, syscall,
    /// allocation, atomic operation and fence, a volatile load, and an inline
    /// asm that is volatile or touches memory. Plain loads
    /// and pure value ops are not included; terminators are kept by their own
    /// rule. Dead-code elimination and constant propagation consult this.
    ///
    /// Every atomic is kept, including an unused `relaxed` `atomic_load`:
    /// dropping one would be sound under the memory model, but keeping it is
    /// the simple, obviously-correct choice.
    pub fn has_side_effect(&self) -> bool {
        match self {
            InstKind::Alloca { .. }
            | InstKind::DynAlloca { .. }
            | InstKind::Store { .. }
            | InstKind::Call
            | InstKind::Syscall => true,
            InstKind::InlineAsm(asm) => asm.has_side_effect(),
            InstKind::Load { volatile, .. } => *volatile,
            k => k.is_atomic(),
        }
    }

    /// Whether this opcode is a block terminator (ends a basic block).
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            InstKind::Ret
                | InstKind::Br(_)
                | InstKind::CondBr { .. }
                | InstKind::Switch(_)
                | InstKind::Unreachable
        )
    }
}

/// A single instruction: opcode payload, its value operands, flags, result
/// type, and (if it produces one) its result value.
///
/// Construct instructions through the builder rather than by hand; the builder
/// keeps the use/def lists and result-value table consistent.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct InstData {
    /// The opcode and its immediate/structural data.
    pub kind: InstKind,
    /// The flag set (only opcode-relevant flags are honored).
    pub flags: Flags,
    /// The result type (the interned `void` type when there is no result).
    pub ty: TypeId,
    /// Flat list of value operands; slot meaning is per-opcode (see [`InstKind`]).
    pub(crate) operands: Vec<ValueId>,
    /// The value this instruction defines, if any (terminators, `store`, and
    /// `void` calls define none).
    pub(crate) result: Option<ValueId>,
}

impl InstData {
    /// The instruction's value operands.
    #[inline]
    pub fn operands(&self) -> &[ValueId] {
        &self.operands
    }

    /// The value this instruction defines, if any.
    #[inline]
    pub fn result(&self) -> Option<ValueId> {
        self.result
    }

    /// Whether this instruction is a block terminator.
    #[inline]
    pub fn is_terminator(&self) -> bool {
        self.kind.is_terminator()
    }

    /// The successor blocks of this terminator, in edge order (empty for a
    /// non-terminator, `ret`, or `unreachable`).
    pub fn successors(&self) -> Vec<BlockId> {
        match &self.kind {
            InstKind::Br(target) => vec![*target],
            InstKind::CondBr { if_true, if_false, .. } => vec![*if_true, *if_false],
            InstKind::Switch(data) => {
                let mut succ = Vec::with_capacity(1 + data.cases.len());
                succ.push(data.default);
                succ.extend(data.cases.iter().map(|c| c.target));
                succ
            }
            _ => Vec::new(),
        }
    }
}
