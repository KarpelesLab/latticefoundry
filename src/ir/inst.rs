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
    /// Unsigned division. Division by zero is **poison**; `exact` and a nonzero
    /// remainder ⇒ poison.
    UDiv,
    /// Signed division. Division by zero, or `INT_MIN / -1`, is **poison**;
    /// `exact` and a nonzero remainder ⇒ poison.
    SDiv,
    /// Unsigned remainder. Division by zero is **poison**.
    URem,
    /// Signed remainder. Division by zero, or `INT_MIN % -1`, is **poison**.
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
}

impl BinOp {
    /// Whether this is a floating-point operation (result and operands are a
    /// float type and fast-math flags apply).
    pub fn is_float(self) -> bool {
        matches!(self, BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem)
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
    /// Load a value of the accessed type from memory; operand `[ptr]`. Result
    /// type = the accessed type. The accessed type and alignment live on the op
    /// (this is where opaque pointers put the type back). Loading through a
    /// poison or dangling pointer, or with insufficient alignment, is undefined
    /// behavior; loading uninitialized memory yields poison.
    Load {
        /// The type read from memory (the result type).
        ty: TypeId,
        /// The assumed alignment of the access, in bytes (a power of two).
        align: u32,
    },
    /// Store a value to memory; operands `[ptr, value]`. No result. Storing
    /// through a poison/dangling pointer or under-aligned is undefined behavior.
    Store {
        /// The type written to memory (the type of the stored value).
        ty: TypeId,
        /// The assumed alignment of the access, in bytes (a power of two).
        align: u32,
    },
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
    /// Function call; operands `[callee, args...]`. `callee` is a function
    /// reference or a pointer; the rest are the arguments in order. Result type
    /// = the callee's return type (`void` for a procedure). Effects and poison
    /// propagation follow the callee's semantics.
    Call,

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
