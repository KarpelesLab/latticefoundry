//! The [`AbstractDomain`] trait: the lattice interface the one fixpoint engine
//! is parameterized by (tenet **T4**, bet **B8**).
//!
//! An abstract domain is a lattice of *abstract values*, each standing for a set
//! of concrete [`SemValue`]s via a **concretization** function γ. The engine
//! ([`crate::analysis::solver`]) never looks inside a domain; it only uses this
//! interface:
//!
//! - the lattice skeleton — [`bottom`](AbstractDomain::bottom),
//!   [`top`](AbstractDomain::top), [`join`](AbstractDomain::join) (least upper
//!   bound), [`le`](AbstractDomain::le) (the partial order ⊑), and
//!   [`widen`](AbstractDomain::widen) for termination on tall/infinite lattices;
//! - the **transfer** interface — [`abstract_const`](AbstractDomain::abstract_const)
//!   (the abstraction α of a single constant) and
//!   [`transfer`](AbstractDomain::transfer) (the abstract semantics of one
//!   instruction, given the abstract values of its operands);
//! - the **control-flow** interface — [`edge_feasible`](AbstractDomain::edge_feasible),
//!   which lets the solver do sparse conditional constant propagation by pruning
//!   branch edges a domain can prove are never taken;
//! - the **concretization** hook — [`contains`](AbstractDomain::contains), i.e.
//!   `v ∈ γ(self)`. This is the anchor **soundness** is defined against: a
//!   transfer function is sound iff, whenever every concrete operand is in the
//!   γ of its abstract operand, the concrete result is in the γ of the abstract
//!   result (see [`crate::analysis::soundness`]).
//!
//! A domain is a monotone lattice: `bottom ⊑ x ⊑ top` for every `x`, `join` is
//! the least upper bound, and `transfer` is monotone in its operand values. The
//! engine's fixpoint and soundness both rest on those laws.

use crate::ir::inst::InstData;
use crate::ir::types::TypeContext;
use crate::ir::value::Const;
use crate::ir::{EvalOutcome, SemValue, eval};

use puremp::Int;

/// The read-only context a transfer function is evaluated against: the module's
/// interned [`TypeContext`], which resolves the result type a transfer needs
/// (e.g. the target width of a cast or the format of a `freeze`).
#[derive(Debug, Clone, Copy)]
pub struct DomainCtx<'a> {
    /// The interned type table for the module under analysis.
    pub types: &'a TypeContext,
}

impl<'a> DomainCtx<'a> {
    /// Wrap a type context as a domain-transfer context.
    pub fn new(types: &'a TypeContext) -> Self {
        Self { types }
    }
}

/// A guard on a single outgoing control-flow edge, phrased over the *condition*
/// value of the terminator. The solver derives these from a terminator's shape;
/// a domain answers, via [`AbstractDomain::edge_feasible`], whether its
/// abstract knowledge of the condition admits any concrete value satisfying the
/// guard (if not, the edge is infeasible and is pruned).
///
/// The width `w` is the bit width of the condition value, so a domain can
/// normalize a case constant to the condition's two's-complement pattern.
#[derive(Debug, Clone, Copy)]
pub enum EdgeGuard<'a> {
    /// The condition (an `i1`) is `true` (non-zero) or `false` (zero).
    CondIs(bool),
    /// The condition equals `value` in the condition's `w`-bit type (a `switch`
    /// case edge).
    CondEquals {
        /// The case's match value.
        value: &'a Int,
        /// The condition's bit width.
        width: u32,
    },
    /// The condition equals **none** of `values` in the `w`-bit type (a `switch`
    /// default edge).
    CondNotAnyOf {
        /// The set of all case match values.
        values: &'a [Int],
        /// The condition's bit width.
        width: u32,
    },
}

/// A lattice with a concretization γ, transfer functions, and a control-flow
/// model — everything the one fixpoint engine needs to run an analysis.
///
/// The associated element type is `Self`: an implementor *is* an abstract value.
/// See the module docs for the laws each method must satisfy.
pub trait AbstractDomain: Clone + PartialEq + std::fmt::Debug + Sized {
    // --- lattice skeleton ---------------------------------------------------

    /// The bottom element ⊥: the empty concretization (γ(⊥) = ∅). In the solver
    /// it doubles as "not yet determined" / unreachable.
    fn bottom() -> Self;

    /// The top element ⊤: the full concretization (γ(⊤) = every concrete value
    /// of the type). "No information."
    fn top() -> Self;

    /// The least upper bound `self ⊔ other`: the most precise abstract value
    /// whose γ contains γ(self) ∪ γ(other).
    fn join(&self, other: &Self) -> Self;

    /// The partial order `self ⊑ other` (γ(self) ⊆ γ(other)). Must be consistent
    /// with [`join`](AbstractDomain::join): `a ⊑ b` iff `a.join(b) == b`.
    fn le(&self, other: &Self) -> bool;

    /// The widening operator `self ∇ next`, applied at loop headers so that a
    /// tall or infinite-height lattice still reaches a fixpoint in finite time.
    ///
    /// `next` is the post-join iterate (`self ⊔ new`); the result must be an
    /// upper bound of both `self` and `next`, and any ascending chain fed
    /// through `widen` must stabilize. The default is [`join`](AbstractDomain::join),
    /// which is correct for finite-height lattices (like flat constants).
    fn widen(&self, next: &Self) -> Self {
        self.join(next)
    }

    // --- concretization γ ---------------------------------------------------

    /// The concretization test `v ∈ γ(self)`: does the concrete value `v` belong
    /// to the set this abstract value denotes? `v` is assumed to have the type of
    /// the SSA value this abstract value describes.
    ///
    /// This is the definition of γ, and hence of **soundness**: see the module
    /// docs and [`crate::analysis::soundness`].
    fn contains(&self, v: &SemValue) -> bool;

    // --- transfer -----------------------------------------------------------

    /// The abstraction α of one constant: the abstract value denoting exactly the
    /// singleton `{c}` (or a sound over-approximation when the domain does not
    /// model `c`'s shape).
    fn abstract_const(ctx: DomainCtx<'_>, c: &Const) -> Self;

    /// The abstract semantics of one value-producing instruction: given the
    /// abstract values of its operands (in operand order), produce the abstract
    /// value of its result.
    ///
    /// Must be **monotone** in `operands` and **sound**: for any concrete
    /// operands `x_i ∈ γ(operands_i)` on which the concrete semantics
    /// ([`eval`](crate::ir::eval)) is defined and yields `r`, the result must
    /// satisfy `r ∈ γ(transfer(..))`.
    fn transfer(ctx: DomainCtx<'_>, inst: &InstData, operands: &[Self]) -> Self;

    // --- control flow -------------------------------------------------------

    /// Whether the outgoing edge guarded by `guard` is feasible given `self` as
    /// the abstract value of the branch condition — i.e. whether γ(self) contains
    /// any concrete condition value satisfying `guard`.
    ///
    /// The default marks every edge feasible (no refinement), which is always
    /// sound. A domain overrides this to prune edges and enable sparse
    /// conditional propagation.
    fn edge_feasible(&self, _guard: &EdgeGuard<'_>) -> bool {
        true
    }
}

/// Run the concrete reference semantics on `inst` with concrete `operands`,
/// returning the produced value, or `None` if the instruction is undefined
/// behavior (or not a pure value-producing opcode) on these operands.
///
/// This is a thin, panic-safe wrapper the soundness harness uses to obtain the
/// oracle result a transfer function must over-approximate. It only accepts the
/// pure opcodes [`eval`](crate::ir::eval) defines.
pub(crate) fn concrete_eval(
    types: &TypeContext,
    inst: &InstData,
    operands: &[SemValue],
) -> Option<SemValue> {
    if inst.kind.is_terminator() {
        return None;
    }
    match &inst.kind {
        // Stateful opcodes have no pure denotation; the soundness harness skips
        // them (their transfer must return `top`).
        crate::ir::InstKind::Alloca { .. }
        | crate::ir::InstKind::DynAlloca { .. }
        | crate::ir::InstKind::Load { .. }
        | crate::ir::InstKind::Store { .. }
        | crate::ir::InstKind::Call => None,
        kind => match eval(types, inst.ty, kind, &inst.flags, operands) {
            EvalOutcome::Value(v) => Some(v),
            EvalOutcome::UndefinedBehavior => None,
        },
    }
}
