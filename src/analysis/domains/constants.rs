//! The **constant-propagation** domain: the first analysis on the one lattice
//! engine, and the proof that the engine works end to end.
//!
//! The lattice is the classic flat (three-level) constant lattice:
//!
//! ```text
//!            ⊤            not a single constant / no information
//!         /  |  \
//!  Const(0) Const(1) ...  known to equal exactly this constant
//!         \  |  /
//!            ⊥            undetermined / unreachable (γ = ∅)
//! ```
//!
//! Its transfer function delegates to [`ir::fold`](crate::ir::fold) — the very
//! constant folder built on the reference semantics — so the abstract semantics
//! *are* the concrete semantics restricted to known-constant operands. That is
//! what makes the transfer sound essentially by construction (checked by
//! [`crate::analysis::soundness`]).
//!
//! Control flow is refined for sparse conditional constant propagation: a
//! `cond_br` or `switch` on a known-constant condition proves every non-matching
//! successor edge infeasible, so blocks reached only through such edges are left
//! unreachable and their values stay ⊥.

use crate::analysis::domain::{AbstractDomain, DomainCtx, EdgeGuard};
use crate::ir::inst::{InstData, InstKind};
use crate::ir::value::Const;
use crate::ir::{FoldResult, SemValue, fold};

/// An element of the flat constant lattice.
///
/// `Const` carries an interned [`Const`] value (which may itself be
/// [`Const::Poison`] — poison is an ordinary concrete value in our semantics).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ConstLattice {
    /// ⊥: undetermined or unreachable. γ = ∅.
    Bottom,
    /// Known to be exactly this constant. γ = { that value }.
    Const(Const),
    /// ⊤: not a single constant. γ = all values of the type.
    Top,
}

impl ConstLattice {
    /// Whether this is the ⊥ element.
    pub fn is_bottom(&self) -> bool {
        matches!(self, ConstLattice::Bottom)
    }

    /// Whether this is the ⊤ element.
    pub fn is_top(&self) -> bool {
        matches!(self, ConstLattice::Top)
    }

    /// The known constant, if this element is a single constant.
    pub fn as_const(&self) -> Option<&Const> {
        match self {
            ConstLattice::Const(c) => Some(c),
            _ => None,
        }
    }
}

impl AbstractDomain for ConstLattice {
    fn bottom() -> Self {
        ConstLattice::Bottom
    }

    fn top() -> Self {
        ConstLattice::Top
    }

    fn join(&self, other: &Self) -> Self {
        use ConstLattice::{Bottom, Const, Top};
        match (self, other) {
            (Bottom, x) | (x, Bottom) => x.clone(),
            (Top, _) | (_, Top) => Top,
            (Const(a), Const(b)) => {
                if a == b {
                    Const(a.clone())
                } else {
                    Top
                }
            }
        }
    }

    fn le(&self, other: &Self) -> bool {
        use ConstLattice::{Bottom, Const, Top};
        match (self, other) {
            (Bottom, _) => true,
            (_, Top) => true,
            (Const(a), Const(b)) => a == b,
            // Top ⊑ only Top; Const ⊑ only itself or Top.
            (Top, _) | (Const(_), Bottom) => false,
        }
    }

    // `widen` defaults to `join`; the flat lattice has height 2, so any
    // ascending chain stabilizes in at most two steps and no widening is needed.

    fn contains(&self, v: &SemValue) -> bool {
        match self {
            ConstLattice::Bottom => false,
            ConstLattice::Top => true,
            ConstLattice::Const(c) => const_matches_sem(c, v),
        }
    }

    fn abstract_const(_ctx: DomainCtx<'_>, c: &Const) -> Self {
        match c {
            // Aggregates are not modeled by the scalar constant lattice; a sound
            // over-approximation is ⊤.
            Const::Aggregate { .. } => ConstLattice::Top,
            _ => ConstLattice::Const(c.clone()),
        }
    }

    fn transfer(ctx: DomainCtx<'_>, inst: &InstData, operands: &[Self]) -> Self {
        // Stateful / opaque opcodes produce a value the domain cannot know.
        if matches!(
            inst.kind,
            InstKind::Alloca { .. }
                | InstKind::DynAlloca { .. }
                | InstKind::Load { .. }
                | InstKind::Call
        ) {
            return ConstLattice::Top;
        }

        // An undetermined operand keeps the result undetermined (SCCP optimism).
        if operands.iter().any(ConstLattice::is_bottom) {
            return ConstLattice::Bottom;
        }

        // Fold only when every operand is a known constant; otherwise ⊤.
        let mut consts = Vec::with_capacity(operands.len());
        for op in operands {
            match op.as_const() {
                Some(c) => consts.push(c.clone()),
                None => return ConstLattice::Top,
            }
        }
        match fold(ctx.types, inst.ty, &inst.kind, &inst.flags, &consts) {
            Some(FoldResult::Folded(c)) => ConstLattice::Const(c),
            // Undefined behavior on these constants, or no scalar-constant
            // representation of the result: conservatively ⊤ (the value, if it
            // exists at runtime at all, is not pinned to a known constant).
            Some(FoldResult::WouldBeUb) | None => ConstLattice::Top,
        }
    }

    fn edge_feasible(&self, guard: &EdgeGuard<'_>) -> bool {
        match self {
            // Unreachable / undetermined condition: no concrete value yet, so no
            // edge is proven feasible (SCCP optimism — resolved as the condition
            // rises above ⊥).
            ConstLattice::Bottom => false,
            // Unknown condition: any edge could be taken.
            ConstLattice::Top => true,
            ConstLattice::Const(c) => const_satisfies_guard(c, guard),
        }
    }
}

/// Whether the concrete value `v` is the one denoted by constant `c`.
///
/// The width of an integer comparison is taken from the concrete value `v`
/// (which carries the SSA value's type), so no type context is needed: a
/// constant `c` and a concrete `Int` match iff they agree modulo `2^width`.
fn const_matches_sem(c: &Const, v: &SemValue) -> bool {
    match (c, v) {
        (Const::Int { value, .. }, SemValue::Int { width, bits }) => &value.mod_2k(*width) == bits,
        (Const::Float { bits: cb, .. }, SemValue::Float(vb)) => cb == vb,
        (Const::Null(_), SemValue::Ptr(addr)) => addr.is_zero(),
        (Const::Poison(_), SemValue::Poison) => true,
        _ => false,
    }
}

/// Whether the concrete condition constant `c` satisfies an edge `guard`.
fn const_satisfies_guard(c: &Const, guard: &EdgeGuard<'_>) -> bool {
    let Const::Int { value, .. } = c else {
        // A non-integer (or poison) condition constant proves no edge; branching
        // on poison is undefined behavior and never a constant integer here.
        return false;
    };
    match guard {
        EdgeGuard::CondIs(want) => {
            // The condition is an `i1`; its truth is "non-zero".
            let is_true = !value.mod_2k(1).is_zero();
            is_true == *want
        }
        EdgeGuard::CondEquals { value: target, width } => {
            value.mod_2k(*width) == target.mod_2k(*width)
        }
        EdgeGuard::CondNotAnyOf { values, width } => {
            let cv = value.mod_2k(*width);
            !values.iter().any(|t| t.mod_2k(*width) == cv)
        }
    }
}
