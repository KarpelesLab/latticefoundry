//! The **pointer-nullness** domain: a tiny lattice that tracks whether a pointer
//! SSA value is the null pointer, definitely non-null, or unknown (tenet **T4**,
//! bet **B8**).
//!
//! # The lattice
//!
//! A four-point lattice over the two independent facts "could be null" and
//! "could be non-null":
//!
//! ```text
//!            MaybeNull          ⊤ — a pointer that may or may not be null
//!            /       \
//!         Null      NonNull     the two singleton facts
//!            \       /
//!             Bottom            ⊥ — undetermined / unreachable (γ = ∅)
//! ```
//!
//! so `Null ⊔ NonNull = MaybeNull`, `x ⊔ Bottom = x`, and `MaybeNull` absorbs.
//! Its height is 3 (`Bottom ⊏ {Null,NonNull} ⊏ MaybeNull`), so every ascending
//! chain stabilizes in at most two steps: [`widen`](AbstractDomain::widen) can
//! safely default to [`join`](AbstractDomain::join) and the fixpoint always
//! terminates without a bespoke widening.
//!
//! # Concretization γ
//!
//! A pointer [`SemValue`] is an opaque byte address ([`SemValue::Ptr`]), where
//! the address `0` is the null pointer (see [`crate::ir::semantics`]). Hence:
//!
//! | element     | γ (the concrete values it denotes)            |
//! | ----------- | --------------------------------------------- |
//! | `Bottom`    | ∅                                             |
//! | `Null`      | `{ Ptr(0) }` — exactly the null pointer         |
//! | `NonNull`   | `{ Ptr(a) : a ≠ 0 }` — every non-null pointer  |
//! | `MaybeNull` | every value (all pointers, poison, non-ptrs)  |
//!
//! `MaybeNull` is ⊤: it is the sound catch-all and its γ contains *everything*,
//! including [`SemValue::Poison`] and any non-pointer value — so a transfer that
//! falls back to `MaybeNull` is always sound.
//!
//! # Soundness scope (poison)
//!
//! `Null` and `NonNull` denote **defined pointer** values only; poison is not in
//! their γ. This is sound because a poison operand can only arise from an operand
//! whose abstract value is `MaybeNull` (poison ∈ γ only there), and every rule
//! that yields `Null`/`NonNull` does so from operands whose γ already excludes
//! poison, or from an opcode (`alloca`) that never produces poison. Two rules
//! (`PtrAdd` inbounds and `Select`) are stated for executions that produce a
//! *defined* pointer — exactly the refinement contract's precondition "the source
//! triggers no UB" (tenet **T2**); each is justified inline below. The bespoke
//! soundness test checks the pointer transfers directly against the reference
//! semantics ([`crate::ir::eval`]); the generic integer soundness harness in
//! [`crate::analysis::soundness`] is integer-shaped and only trivially exercises
//! a pointer domain (every integer transfer here is ⊤).

use crate::analysis::domain::{AbstractDomain, DomainCtx};
use crate::ir::SemValue;
use crate::ir::inst::{InstData, InstKind};
use crate::ir::value::Const;

/// An element of the pointer-nullness lattice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Nullness {
    /// ⊥: undetermined or unreachable. γ = ∅.
    Bottom,
    /// Known to be the null pointer. γ = `{ Ptr(0) }`.
    Null,
    /// Known to be a non-null pointer. γ = `{ Ptr(a) : a ≠ 0 }`.
    NonNull,
    /// ⊤: may or may not be null — no information. γ = every value.
    MaybeNull,
}

impl Nullness {
    /// Whether this is the ⊥ element.
    pub fn is_bottom(self) -> bool {
        matches!(self, Nullness::Bottom)
    }

    /// Whether this is the ⊤ element (`MaybeNull`).
    pub fn is_top(self) -> bool {
        matches!(self, Nullness::MaybeNull)
    }

    /// Whether this element proves the pointer is definitely non-null.
    pub fn is_non_null(self) -> bool {
        matches!(self, Nullness::NonNull)
    }

    /// Whether this element proves the pointer is definitely null.
    pub fn is_null(self) -> bool {
        matches!(self, Nullness::Null)
    }
}

impl AbstractDomain for Nullness {
    fn bottom() -> Self {
        Nullness::Bottom
    }

    fn top() -> Self {
        Nullness::MaybeNull
    }

    fn join(&self, other: &Self) -> Self {
        use Nullness::{Bottom, MaybeNull, NonNull, Null};
        match (self, other) {
            // ⊥ is the identity.
            (Bottom, x) | (x, Bottom) => *x,
            // ⊤ absorbs.
            (MaybeNull, _) | (_, MaybeNull) => MaybeNull,
            // Idempotent on the two singletons; the two distinct facts join to ⊤.
            (Null, Null) => Null,
            (NonNull, NonNull) => NonNull,
            (Null, NonNull) | (NonNull, Null) => MaybeNull,
        }
    }

    fn le(&self, other: &Self) -> bool {
        use Nullness::{Bottom, MaybeNull};
        match (self, other) {
            // ⊥ is below everything; everything is below ⊤.
            (Bottom, _) | (_, MaybeNull) => true,
            // Otherwise an element is ⊑ only itself: ⊤ ⊑ only ⊤ (handled above),
            // and a singleton is never ⊑ ⊥ or ⊑ the other singleton.
            _ => self == other,
        }
    }

    // `widen` defaults to `join`: the lattice has height 3, so any ascending
    // chain stabilizes in at most two steps and no widening is required.

    fn contains(&self, v: &SemValue) -> bool {
        match self {
            // γ(⊥) = ∅.
            Nullness::Bottom => false,
            // ⊤ contains every value, including poison and non-pointers.
            Nullness::MaybeNull => true,
            // The null pointer is exactly the zero address; a non-pointer or
            // poison value is in neither singleton's γ.
            Nullness::Null => matches!(v, SemValue::Ptr(addr) if addr.is_zero()),
            Nullness::NonNull => matches!(v, SemValue::Ptr(addr) if !addr.is_zero()),
        }
    }

    fn abstract_const(_ctx: DomainCtx<'_>, c: &Const) -> Self {
        match c {
            // The null pointer constant is exactly `Null`.
            Const::Null(_) => Nullness::Null,
            // A poison pointer constant is poison, which lives only in γ(⊤); a
            // non-pointer constant carries no nullness fact. Both are ⊤ (sound).
            //
            // Note: global- and function-address references are *not* constants —
            // they are `ValueDef::Global`/`ValueDef::Func` values the solver seeds
            // directly as `top()`. They are therefore conservatively `MaybeNull`
            // here (sound, if imprecise); refining them to `NonNull` would require
            // touching the solver's seeding, which this domain does not own.
            _ => Nullness::MaybeNull,
        }
    }

    fn transfer(_ctx: DomainCtx<'_>, inst: &InstData, operands: &[Self]) -> Self {
        // SCCP optimism: an operand from a not-yet-reached path (⊥) keeps the
        // result undetermined. γ(⊥) = ∅, so this is vacuously sound and monotone.
        if operands.iter().any(|o| o.is_bottom()) {
            return Nullness::Bottom;
        }

        match &inst.kind {
            // A fresh stack slot is always a non-null, non-poison pointer valid
            // for the activation (see `InstKind::Alloca`). Sound. A dynamic
            // (runtime-sized) allocation is likewise a fresh non-null pointer.
            InstKind::Alloca { .. } | InstKind::DynAlloca { .. } => Nullness::NonNull,

            // `ptr_add base, off`.
            InstKind::PtrAdd { inbounds } => {
                let base = operands.first().copied().unwrap_or(Nullness::MaybeNull);
                if *inbounds && base.is_non_null() {
                    // An `inbounds` displacement that stays within `base`'s
                    // allocation yields a pointer into that same (non-null)
                    // object; leaving the allocation is poison, not a null
                    // pointer, and the null address 0 is never inside a valid
                    // allocation. So on every *defined* execution the result is a
                    // non-null pointer.
                    Nullness::NonNull
                } else {
                    // Non-`inbounds` arithmetic (or an unknown/possibly-null
                    // base) can wrap to any address, including 0. ⊤ (sound).
                    Nullness::MaybeNull
                }
            }

            // `select cond, t, f`. On a defined execution `cond` is a definite
            // `i1` and the result equals one of the arms, so the result's
            // nullness is bounded by the join of the two arms.
            InstKind::Select => {
                let t = operands.get(1).copied().unwrap_or(Nullness::MaybeNull);
                let f = operands.get(2).copied().unwrap_or(Nullness::MaybeNull);
                t.join(&f)
            }

            // `freeze v` is the identity on a defined operand and, on a poison
            // operand, produces a fixed concrete value (the null pointer for a
            // pointer type). Passing the operand's nullness through is sound in
            // every case: if the operand is `NonNull`/`Null` its γ excludes
            // poison so `freeze` is the identity; if it is `MaybeNull` the frozen
            // value (some pointer, or null) is still in γ(⊤).
            InstKind::Freeze => operands.first().copied().unwrap_or(Nullness::MaybeNull),

            // Everything else that yields a pointer we cannot pin down —
            // `inttoptr` (any address), a `load`ed pointer, a `call` result — as
            // well as any non-pointer result, is ⊤. Sound catch-all.
            _ => Nullness::MaybeNull,
        }
    }

    // `edge_feasible` uses the default (every edge feasible). Precise refinement
    // of a pointer on an `icmp eq/ne ptr, null` branch is *deferred*: the
    // `EdgeGuard` interface is phrased over the branch *condition* value (the
    // `i1` result of the compare), and the sparse solver has no channel to
    // back-propagate a refinement onto the *compared pointer* on a specific edge.
    // Overriding `edge_feasible` here could only inspect the condition's own
    // nullness (always `MaybeNull`), which proves nothing — so the sound default
    // is kept rather than claiming precision the interface cannot deliver.
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::analysis::soundness::check_integer_transfer_sound;
    use crate::analysis::solver::{FixpointResult, solve};
    use crate::ir::inst::{CastOp, Flags};
    use crate::ir::types::TypeContext;
    use crate::ir::{EvalOutcome, FuncId, Module, TypeId, eval};
    use crate::support::StrInterner;

    use puremp::Int;

    /// Every lattice element, for the law tests.
    const ELEMS: [Nullness; 4] =
        [Nullness::Bottom, Nullness::Null, Nullness::NonNull, Nullness::MaybeNull];

    fn null_ptr() -> SemValue {
        SemValue::ptr(Int::ZERO)
    }

    fn non_null_ptr(addr: u64) -> SemValue {
        SemValue::ptr(Int::from_u64(addr))
    }

    fn solve_nullness(m: &Module, f: FuncId) -> FixpointResult<Nullness> {
        solve::<Nullness>(m.function(f), m.types(), m.consts())
    }

    // ---------------------------------------------------------------------
    // Lattice laws
    // ---------------------------------------------------------------------

    #[test]
    fn nullness_lattice_laws() {
        for a in ELEMS {
            // Idempotent.
            assert_eq!(a.join(&a), a, "join idempotent");
            // ⊥ identity, ⊤ absorbing.
            assert_eq!(a.join(&Nullness::Bottom), a, "bottom is join identity");
            assert_eq!(a.join(&Nullness::MaybeNull), Nullness::MaybeNull, "top absorbs");
            for b in ELEMS {
                // Commutative.
                assert_eq!(a.join(&b), b.join(&a), "join commutative");
                // `le` consistent with join: a ⊑ b iff a ⊔ b == b.
                assert_eq!(a.le(&b), a.join(&b) == b, "le consistent with join");
                for c in ELEMS {
                    // Associative.
                    assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)), "join associative");
                }
            }
        }
    }

    #[test]
    fn nullness_order_is_the_four_point_diamond() {
        use Nullness::{Bottom, MaybeNull, NonNull, Null};
        // The single defining non-trivial join.
        assert_eq!(Null.join(&NonNull), MaybeNull);
        // ⊥ below the two facts, both below ⊤; the two facts incomparable.
        assert!(Bottom.le(&Null) && Bottom.le(&NonNull) && Bottom.le(&MaybeNull));
        assert!(Null.le(&MaybeNull) && NonNull.le(&MaybeNull));
        assert!(!Null.le(&NonNull) && !NonNull.le(&Null));
        assert!(!MaybeNull.le(&Null) && !MaybeNull.le(&NonNull) && !MaybeNull.le(&Bottom));
        assert!(!Null.le(&Bottom) && !NonNull.le(&Bottom));
        // Reflexive.
        for a in ELEMS {
            assert!(a.le(&a), "le reflexive");
        }
    }

    // ---------------------------------------------------------------------
    // γ / contains
    // ---------------------------------------------------------------------

    #[test]
    fn gamma_matches_concrete_pointers() {
        // Null: exactly the zero address.
        assert!(Nullness::Null.contains(&null_ptr()));
        assert!(!Nullness::Null.contains(&non_null_ptr(0x1000)));

        // NonNull: every non-zero address, but not null.
        assert!(Nullness::NonNull.contains(&non_null_ptr(0x1000)));
        assert!(!Nullness::NonNull.contains(&null_ptr()));

        // MaybeNull (⊤): every value, including poison and non-pointers.
        assert!(Nullness::MaybeNull.contains(&null_ptr()));
        assert!(Nullness::MaybeNull.contains(&non_null_ptr(0x1000)));
        assert!(Nullness::MaybeNull.contains(&SemValue::Poison));
        assert!(Nullness::MaybeNull.contains(&SemValue::int(32, Int::from_u64(7))));

        // Bottom (⊥): nothing.
        assert!(!Nullness::Bottom.contains(&null_ptr()));
        assert!(!Nullness::Bottom.contains(&non_null_ptr(0x1000)));
        assert!(!Nullness::Bottom.contains(&SemValue::Poison));

        // A poison / non-pointer value is in neither singleton's γ.
        assert!(!Nullness::Null.contains(&SemValue::Poison));
        assert!(!Nullness::NonNull.contains(&SemValue::Poison));
        assert!(!Nullness::Null.contains(&SemValue::int(64, Int::ZERO)));
        assert!(!Nullness::NonNull.contains(&SemValue::int(64, Int::from_u64(3))));
    }

    // ---------------------------------------------------------------------
    // abstract_const (α)
    // ---------------------------------------------------------------------

    #[test]
    fn abstract_const_classifies_constants() {
        let mut m = Module::new("consts");
        let ptr = m.types_mut().ptr();
        let i32t = m.types_mut().int(32);
        let ctx = DomainCtx::new(m.types());

        // The null pointer constant.
        assert_eq!(Nullness::abstract_const(ctx, &Const::Null(ptr)), Nullness::Null);
        // A poison pointer constant is ⊤ (poison lives only in γ(⊤)).
        assert_eq!(Nullness::abstract_const(ctx, &Const::Poison(ptr)), Nullness::MaybeNull);
        // A non-pointer constant carries no nullness fact: ⊤.
        let int_c = Const::Int { ty: i32t, value: Int::ZERO };
        assert_eq!(Nullness::abstract_const(ctx, &int_c), Nullness::MaybeNull);
    }

    // ---------------------------------------------------------------------
    // Engine transfers (via the builder + the sparse solver)
    // ---------------------------------------------------------------------

    #[test]
    fn alloca_is_non_null_and_null_const_is_null() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("alloca");
        let i32t = m.types_mut().int(32);
        let void = m.types_mut().void();
        let sig = m.types_mut().func(vec![], void, false);
        let f = m.declare_function(syms.intern("f"), sig);

        let (p, n);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            p = b.alloca(i32t);
            let ptr_ty = b.value_type(p);
            n = b.null(ptr_ty);
            b.ret(None);
        }

        let r = solve_nullness(&m, f);
        assert_eq!(*r.value(p), Nullness::NonNull, "alloca is non-null");
        assert_eq!(*r.value(n), Nullness::Null, "null constant is Null");
    }

    #[test]
    fn select_between_alloca_and_null_is_maybe_null() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("select");
        let i32t = m.types_mut().int(32);
        let i1 = m.types_mut().bool();
        let void = m.types_mut().void();
        let sig = m.types_mut().func(vec![i1], void, false);
        let f = m.declare_function(syms.intern("f"), sig);

        let s;
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let cond = b.param(entry, 0);
            let p = b.alloca(i32t);
            let ptr_ty = b.value_type(p);
            let n = b.null(ptr_ty);
            s = b.select(cond, p, n);
            b.ret(None);
        }

        let r = solve_nullness(&m, f);
        assert_eq!(*r.value(s), Nullness::MaybeNull, "select of NonNull and Null is MaybeNull");
    }

    #[test]
    fn inbounds_ptr_add_of_alloca_stays_non_null() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("ptradd");
        let i32t = m.types_mut().int(32);
        let i64t = m.types_mut().int(64);
        let void = m.types_mut().void();
        let sig = m.types_mut().func(vec![], void, false);
        let f = m.declare_function(syms.intern("f"), sig);

        let (inb, oob);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let p = b.alloca(i32t);
            let off = b.const_i64(i64t, 4);
            inb = b.ptr_add(p, off, true); // inbounds
            oob = b.ptr_add(p, off, false); // not inbounds
            b.ret(None);
        }

        let r = solve_nullness(&m, f);
        assert_eq!(
            *r.value(inb),
            Nullness::NonNull,
            "inbounds ptr_add of a non-null base stays NonNull"
        );
        assert_eq!(
            *r.value(oob),
            Nullness::MaybeNull,
            "non-inbounds ptr_add is conservatively MaybeNull"
        );
    }

    #[test]
    fn loaded_pointer_and_call_result_are_maybe_null() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("load_call");
        let ptr = m.types_mut().ptr();
        let sig = m.types_mut().func(vec![ptr], ptr, false);
        let f = m.declare_function(syms.intern("f"), sig);

        let (loaded, called);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let arg = b.param(entry, 0);
            loaded = b.load(ptr, arg, 8);
            let callee = b.func_ref(f);
            called = b.call(callee, &[arg], ptr).expect("call returns a pointer");
            b.ret(Some(called));
        }

        let r = solve_nullness(&m, f);
        assert_eq!(*r.value(loaded), Nullness::MaybeNull, "a loaded pointer is MaybeNull");
        assert_eq!(*r.value(called), Nullness::MaybeNull, "a call result is MaybeNull");
    }

    #[test]
    fn inttoptr_is_maybe_null() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("inttoptr");
        let ptr = m.types_mut().ptr();
        let i64t = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![i64t], ptr, false);
        let f = m.declare_function(syms.intern("f"), sig);

        let cast;
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let x = b.param(entry, 0);
            cast = b.cast(CastOp::IntToPtr, x, ptr);
            b.ret(Some(cast));
        }

        let r = solve_nullness(&m, f);
        assert_eq!(*r.value(cast), Nullness::MaybeNull, "inttoptr could be any address");
    }

    #[test]
    fn freeze_passes_nullness_through() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("freeze");
        let i32t = m.types_mut().int(32);
        let void = m.types_mut().void();
        let sig = m.types_mut().func(vec![], void, false);
        let f = m.declare_function(syms.intern("f"), sig);

        let (fp, fnull);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let p = b.alloca(i32t);
            fp = b.freeze(p);
            let ptr_ty = b.value_type(p);
            let n = b.null(ptr_ty);
            fnull = b.freeze(n);
            b.ret(None);
        }

        let r = solve_nullness(&m, f);
        assert_eq!(*r.value(fp), Nullness::NonNull, "freeze of a NonNull stays NonNull");
        assert_eq!(*r.value(fnull), Nullness::Null, "freeze of a Null stays Null");
    }

    // ---------------------------------------------------------------------
    // Soundness — bespoke, pointer-oriented
    // ---------------------------------------------------------------------
    //
    // The generic `check_integer_transfer_sound` harness only feeds integer
    // opcodes; against a pointer domain every transfer there returns ⊤
    // (`MaybeNull`), which γ-contains any integer result — so it passes but proves
    // little. We run it anyway (below) as a regression that the domain never
    // *under*-approximates an integer op, then verify the pointer transfers
    // directly against the reference semantics (`ir::eval`) here: for every
    // sampled defined result, the transfer's abstract result must γ-contain it.

    /// Assert the transfer's abstract result γ-contains the concrete result the
    /// reference semantics produces for `kind` on `concrete` operands.
    fn assert_transfer_sound(
        types: &TypeContext,
        ptr_ty: TypeId,
        kind: InstKind,
        abstract_ops: &[Nullness],
        concrete: &[SemValue],
    ) {
        let ctx = DomainCtx::new(types);
        let inst = InstData {
            kind: kind.clone(),
            flags: Flags::NONE,
            ty: ptr_ty,
            operands: Vec::new(),
            result: None,
        };
        let abstract_result = Nullness::transfer(ctx, &inst, abstract_ops);
        if let EvalOutcome::Value(r) = eval(types, ptr_ty, &kind, &Flags::NONE, concrete) {
            assert!(
                abstract_result.contains(&r),
                "unsound: {kind:?} ops={abstract_ops:?} concrete={concrete:?} \
                 -> {r:?} not in gamma({abstract_result:?})",
            );
        }
    }

    #[test]
    fn pointer_transfers_are_sound_against_eval() {
        let mut m = Module::new("sound");
        let ptr_ty = m.types_mut().ptr();
        let types = m.types();

        // PtrAdd, inbounds, non-null base, in-bounds (non-wrapping) offsets: the
        // result is always a non-null pointer, γ-contained by NonNull.
        for &(base_addr, off) in &[(0x1000u64, 0i64), (0x1000, 4), (0x1000, -8), (0x20, 0x10)] {
            let base = SemValue::ptr(Int::from_u64(base_addr));
            let offv = SemValue::int(64, Int::from_i64(off));
            assert_transfer_sound(
                types,
                ptr_ty,
                InstKind::PtrAdd { inbounds: true },
                &[Nullness::NonNull, Nullness::MaybeNull],
                &[base, offv],
            );
        }

        // PtrAdd, not inbounds: even from a non-null base the address may wrap to
        // 0, but the transfer returns MaybeNull (⊤), which contains it.
        {
            let base = SemValue::ptr(Int::from_u64(0x8));
            let offv = SemValue::int(64, Int::from_i64(-8)); // wraps to address 0
            assert_transfer_sound(
                types,
                ptr_ty,
                InstKind::PtrAdd { inbounds: false },
                &[Nullness::NonNull, Nullness::MaybeNull],
                &[base, offv],
            );
        }

        // Select: join of the arms γ-contains the chosen arm on each condition.
        // Tight case (both arms NonNull) checks NonNull is preserved.
        for cond in [false, true] {
            let c = SemValue::boolean(cond);
            let t = SemValue::ptr(Int::from_u64(0x1000));
            let f_nn = SemValue::ptr(Int::from_u64(0x2000));
            assert_transfer_sound(
                types,
                ptr_ty,
                InstKind::Select,
                &[Nullness::MaybeNull, Nullness::NonNull, Nullness::NonNull],
                &[c.clone(), t.clone(), f_nn],
            );
            // Mixed arms (NonNull / Null) join to MaybeNull, which contains both.
            let f_null = SemValue::ptr(Int::ZERO);
            assert_transfer_sound(
                types,
                ptr_ty,
                InstKind::Select,
                &[Nullness::MaybeNull, Nullness::NonNull, Nullness::Null],
                &[c, t, f_null],
            );
        }

        // Freeze: identity on a defined pointer.
        for (addr, abs) in [(0u64, Nullness::Null), (0x1000, Nullness::NonNull)] {
            let v = SemValue::ptr(Int::from_u64(addr));
            assert_transfer_sound(types, ptr_ty, InstKind::Freeze, &[abs], &[v]);
        }
    }

    #[test]
    fn integer_harness_is_trivially_satisfied() {
        // Not the meaningful check for a pointer domain, but it must still hold:
        // every integer transfer returns ⊤, which γ-contains the concrete result.
        let report = check_integer_transfer_sound::<Nullness>(2000, 0x0BAD_F00D);
        assert!(report.is_sound(), "soundness violations: {:?}", report.violations);
        assert!(report.checked > 0);
    }

    // ---------------------------------------------------------------------
    // Edge feasibility — refinement deferred
    // ---------------------------------------------------------------------

    #[test]
    fn edge_feasibility_is_the_sound_default() {
        use crate::analysis::domain::EdgeGuard;
        // Pointer refinement on null-check branches is deferred (see the impl
        // note): the default marks every edge feasible, which is always sound.
        for a in ELEMS {
            assert!(a.edge_feasible(&EdgeGuard::CondIs(true)));
            assert!(a.edge_feasible(&EdgeGuard::CondIs(false)));
        }
    }
}
