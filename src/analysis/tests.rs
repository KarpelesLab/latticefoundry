//! Tests for the one lattice engine: the constant-propagation domain on the
//! sparse SSA solver, the lattice laws, the soundness harness, the analysis
//! manager's caching/invalidation, and widening-driven termination on a loop
//! (with a purpose-built tall-lattice mock domain).

use super::domain::{AbstractDomain, DomainCtx, EdgeGuard};
use super::domains::ConstLattice;
use super::manager::{AnalysisCache, ConstantPropagation};
use super::soundness::check_integer_transfer_sound;
use super::solver::{FixpointResult, solve};

use crate::ir::inst::{Flags, InstData, InstKind, IntPred};
use crate::ir::value::{Const, ValueId};
use crate::ir::{Module, SemValue};
use crate::pass::{Changed, ModulePass, PassManager};
use crate::support::StrInterner;

use puremp::Int;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Assert an SSA value is a known integer constant equal to `expected` (as an
/// unsigned representative in the value's width).
fn assert_const_int(result: &FixpointResult<ConstLattice>, v: ValueId, width: u32, expected: i64) {
    match result.value(v).as_const() {
        Some(Const::Int { value, .. }) => {
            let want = Int::from_i64(expected).mod_2k(width);
            assert_eq!(value.mod_2k(width), want, "value {v:?} constant mismatch");
        }
        other => panic!("expected a constant int for {v:?}, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Constant propagation over the engine
// ---------------------------------------------------------------------------

#[test]
fn folds_a_chain_of_constant_ops() {
    // f() -> i32 { a = 2 + 3; b = a * 4; c = b - 1; ret c }  ==> c == 19
    let mut syms = StrInterner::new();
    let mut m = Module::new("consts");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);

    let (a, b, c);
    {
        let mut bld = m.build(f);
        bld.create_entry_block();
        let two = bld.const_i64(i32t, 2);
        let three = bld.const_i64(i32t, 3);
        let four = bld.const_i64(i32t, 4);
        let one = bld.const_i64(i32t, 1);
        a = bld.add(two, three, Flags::NONE);
        b = bld.mul(a, four, Flags::NONE);
        c = bld.sub(b, one, Flags::NONE);
        bld.ret(Some(c));
    }

    let r = solve::<ConstLattice>(m.function(f), m.types(), m.consts());
    assert_const_int(&r, a, 32, 5);
    assert_const_int(&r, b, 32, 20);
    assert_const_int(&r, c, 32, 19);
}

#[test]
fn value_depending_on_a_parameter_is_top() {
    // f(x: i32) -> i32 { r = x + 1; ret r }  ==> r is Top (x unknown)
    let mut syms = StrInterner::new();
    let mut m = Module::new("param");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);

    let (x, r);
    {
        let mut bld = m.build(f);
        let entry = bld.create_entry_block();
        x = bld.param(entry, 0);
        let one = bld.const_i64(i32t, 1);
        r = bld.add(x, one, Flags::NONE);
        bld.ret(Some(r));
    }

    let res = solve::<ConstLattice>(m.function(f), m.types(), m.consts());
    assert!(res.value(x).is_top(), "parameter must be Top");
    assert!(res.value(r).is_top(), "value derived from a parameter must be Top");
}

#[test]
fn constant_condbr_prunes_the_untaken_edge() {
    // f() -> i32:
    //   entry: cond = true; cond_br cond, then(), else()
    //   then:  w = 5 + 5;   br merge(w)
    //   else:  z = 20 * 20; br merge(z)     [infeasible]
    //   merge(x): ret x                     ==> x == 10, else unreachable, z == Bottom
    let mut syms = StrInterner::new();
    let mut m = Module::new("cond");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);

    let (then_b, else_b, merge_b, z, w, x);
    {
        let mut bld = m.build(f);
        let entry = bld.create_entry_block();
        then_b = bld.create_block(&[]);
        else_b = bld.create_block(&[]);
        merge_b = bld.create_block(&[i32t]);

        bld.switch_to(entry);
        let t = bld.const_bool(true);
        bld.cond_br(t, then_b, &[], else_b, &[]);

        bld.switch_to(then_b);
        let five = bld.const_i64(i32t, 5);
        w = bld.add(five, five, Flags::NONE);
        bld.br(merge_b, &[w]);

        bld.switch_to(else_b);
        let twenty = bld.const_i64(i32t, 20);
        z = bld.mul(twenty, twenty, Flags::NONE);
        bld.br(merge_b, &[z]);

        bld.switch_to(merge_b);
        x = bld.param(merge_b, 0);
        bld.ret(Some(x));
    }

    let r = solve::<ConstLattice>(m.function(f), m.types(), m.consts());
    assert!(r.is_reachable(then_b), "then block must be reachable");
    assert!(!r.is_reachable(else_b), "else block must be pruned");
    assert!(r.is_reachable(merge_b));
    assert_const_int(&r, w, 32, 10);
    assert!(r.value(z).is_bottom(), "value in an unreachable block must be Bottom");
    assert_const_int(&r, x, 32, 10);
}

#[test]
fn declaration_has_no_reachable_blocks() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("decl");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t], i32t, false);
    let f = m.declare_function(syms.intern("ext"), sig);
    // No body built: it is a declaration.
    let r = solve::<ConstLattice>(m.function(f), m.types(), m.consts());
    assert_eq!(r.value_count(), m.function(f).value_count());
}

// ---------------------------------------------------------------------------
// Lattice laws for the constant domain
// ---------------------------------------------------------------------------

#[test]
fn constant_lattice_laws() {
    let mut m = Module::new("laws");
    let i32t = m.types_mut().int(32);
    let i8t = m.types_mut().int(8);
    let c1 = Const::Int { ty: i32t, value: Int::from_i64(1) };
    let c2 = Const::Int { ty: i32t, value: Int::from_i64(2) };
    let c1b = Const::Int { ty: i8t, value: Int::from_i64(1) };

    let elems = [
        ConstLattice::Bottom,
        ConstLattice::Top,
        ConstLattice::Const(c1),
        ConstLattice::Const(c2),
        ConstLattice::Const(c1b),
    ];

    for a in &elems {
        // Idempotent.
        assert_eq!(a.join(a), *a, "join idempotent");
        // Identity/absorbing.
        assert_eq!(a.join(&ConstLattice::Bottom), *a, "bottom is join identity");
        assert_eq!(a.join(&ConstLattice::Top), ConstLattice::Top, "top absorbs");
        for b in &elems {
            // Commutative.
            assert_eq!(a.join(b), b.join(a), "join commutative");
            // le consistent with join: a ⊑ b iff a ⊔ b == b.
            assert_eq!(a.le(b), &a.join(b) == b, "le consistent with join");
            for cc in &elems {
                // Associative.
                assert_eq!(a.join(b).join(cc), a.join(&b.join(cc)), "join associative");
            }
        }
    }
}

#[test]
fn gamma_contains_matches_folding() {
    // Const(i32 -1) contains the concrete i32 whose pattern is 0xFFFF_FFFF.
    let mut m = Module::new("gamma");
    let i32t = m.types_mut().int(32);
    let neg1 = ConstLattice::Const(Const::Int { ty: i32t, value: Int::MINUS_ONE });
    assert!(neg1.contains(&SemValue::int(32, Int::MINUS_ONE)));
    assert!(!neg1.contains(&SemValue::int(32, Int::from_i64(1))));
    assert!(ConstLattice::Top.contains(&SemValue::int(32, Int::from_i64(7))));
    assert!(!ConstLattice::Bottom.contains(&SemValue::int(32, Int::ZERO)));
}

// ---------------------------------------------------------------------------
// Soundness harness
// ---------------------------------------------------------------------------

#[test]
fn constants_transfer_is_sound_on_random_inputs() {
    let report = check_integer_transfer_sound::<ConstLattice>(5000, 0x1234_5678);
    assert!(report.is_sound(), "soundness violations: {:?}", report.violations);
    assert!(report.checked > 0, "the harness must actually check some cases");
    // A different seed still holds.
    let report2 = check_integer_transfer_sound::<ConstLattice>(5000, 0xDEAD_BEEF);
    assert!(report2.is_sound(), "soundness violations: {:?}", report2.violations);
}

// ---------------------------------------------------------------------------
// Analysis manager caching / invalidation
// ---------------------------------------------------------------------------

/// A no-op pass that always reports it changed the IR (to trigger invalidation).
#[derive(Debug)]
struct TouchPass;

impl ModulePass for TouchPass {
    fn name(&self) -> &str {
        "touch"
    }
    fn run(&mut self, _module: &mut Module) -> Changed {
        Changed::Yes
    }
}

fn const_module() -> (Module, crate::ir::FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("mgr");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![], i32t, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut bld = m.build(f);
        bld.create_entry_block();
        let two = bld.const_i64(i32t, 2);
        let three = bld.const_i64(i32t, 3);
        let s = bld.add(two, three, Flags::NONE);
        bld.ret(Some(s));
    }
    (m, f)
}

#[test]
fn analysis_cache_memoizes_and_invalidates() {
    let (m, f) = const_module();
    let mut cache = AnalysisCache::new();
    let cp = ConstantPropagation;

    assert!(!cache.is_cached(&cp, f));
    let _ = cache.get_or_compute(&cp, f, m.function(f), m.types(), m.consts());
    assert_eq!(cache.computations(), 1);
    assert!(cache.is_cached(&cp, f));

    // Second request is served from cache (no recomputation).
    let _ = cache.get_or_compute(&cp, f, m.function(f), m.types(), m.consts());
    assert_eq!(cache.computations(), 1, "second query must hit the cache");

    // Function-scoped invalidation forces a recompute.
    cache.invalidate_function(f);
    assert!(!cache.is_cached(&cp, f));
    let _ = cache.get_or_compute(&cp, f, m.function(f), m.types(), m.consts());
    assert_eq!(cache.computations(), 2);

    cache.invalidate_all();
    assert!(cache.is_empty());
}

#[test]
fn pass_manager_invalidates_on_change() {
    let (mut m, f) = const_module();
    let mut pm = PassManager::new();
    {
        let types = m.types();
        let consts = m.consts();
        let func = m.function(f);
        let _ = pm.analyses_mut().get_or_compute(&ConstantPropagation, f, func, types, consts);
    }
    assert!(!pm.analyses().is_empty(), "cache should be seeded");

    pm.add(Box::new(TouchPass));
    pm.run(&mut m);
    assert!(pm.analyses().is_empty(), "a Changed::Yes pass must invalidate the cache");
}

// ---------------------------------------------------------------------------
// Widening / termination on a tall-lattice mock domain
// ---------------------------------------------------------------------------

/// A deliberately tall lattice: `Bottom ⊏ Finite(0) ⊏ Finite(1) ⊏ … ⊏ Top`.
///
/// Its join is a max that never saturates, so an induction variable would climb
/// forever under plain iteration. Its `widen` collapses any strict finite
/// increase straight to `Top`, which is exactly what makes the fixpoint
/// terminate at loop headers.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Countup {
    Bottom,
    Finite(u64),
    Top,
}

impl AbstractDomain for Countup {
    fn bottom() -> Self {
        Countup::Bottom
    }
    fn top() -> Self {
        Countup::Top
    }
    fn join(&self, other: &Self) -> Self {
        use Countup::{Bottom, Finite, Top};
        match (self, other) {
            (Bottom, x) | (x, Bottom) => x.clone(),
            (Top, _) | (_, Top) => Top,
            (Finite(a), Finite(b)) => Finite(*a.max(b)),
        }
    }
    fn le(&self, other: &Self) -> bool {
        use Countup::{Bottom, Finite, Top};
        match (self, other) {
            (Bottom, _) => true,
            (_, Top) => true,
            (Finite(a), Finite(b)) => a <= b,
            (Top, _) | (Finite(_), Bottom) => false,
        }
    }
    fn widen(&self, next: &Self) -> Self {
        use Countup::{Bottom, Finite, Top};
        match (self, next) {
            // First rise out of Bottom: adopt the value (no widening yet).
            (Bottom, x) => x.clone(),
            // A strictly growing finite chain jumps to Top.
            (Finite(a), Finite(b)) if b > a => Top,
            (Finite(_), Finite(b)) => Finite(*b),
            (_, Top) | (Top, _) => Top,
            // `next ⊒ self` always holds in the solver, so this is unreachable
            // in practice; keep the match total and monotone.
            (Finite(a), Bottom) => Finite(*a),
        }
    }
    fn contains(&self, v: &SemValue) -> bool {
        match self {
            Countup::Bottom => false,
            Countup::Top => true,
            Countup::Finite(n) => matches!(v, SemValue::Int { bits, .. } if bits.to_u64() == Some(*n)),
        }
    }
    fn abstract_const(_ctx: DomainCtx<'_>, c: &Const) -> Self {
        match c {
            Const::Int { value, .. } => Countup::Finite(value.to_u64().unwrap_or(0)),
            _ => Countup::Top,
        }
    }
    fn transfer(_ctx: DomainCtx<'_>, inst: &InstData, operands: &[Self]) -> Self {
        if operands.iter().any(|o| matches!(o, Countup::Bottom)) {
            return Countup::Bottom;
        }
        // Model only integer add precisely; everything else is Top.
        match (&inst.kind, operands) {
            (InstKind::Bin(crate::ir::BinOp::Add), [Countup::Finite(a), Countup::Finite(b)]) => {
                Countup::Finite(a.wrapping_add(*b))
            }
            _ => Countup::Top,
        }
    }
    // Default `edge_feasible` (all edges feasible) keeps the loop live.
}

#[test]
fn widening_terminates_a_counting_loop() {
    // f() -> i32:
    //   entry: br header(0)
    //   header(i): cond = i < 1_000_000; cond_br cond, body(i), exit(i)
    //   body(i): i' = i + 1; br header(i')          [back edge]
    //   exit(i): ret i
    let mut syms = StrInterner::new();
    let mut m = Module::new("loop");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![], i32t, false);
    let f = m.declare_function(syms.intern("count"), sig);

    let header;
    let header_i;
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        header = b.create_block(&[i32t]);
        let body = b.create_block(&[i32t]);
        let exit = b.create_block(&[i32t]);

        b.switch_to(entry);
        let zero = b.const_i64(i32t, 0);
        b.br(header, &[zero]);

        b.switch_to(header);
        header_i = b.param(header, 0);
        let bound = b.const_i64(i32t, 1_000_000);
        let cond = b.icmp(IntPred::Slt, header_i, bound);
        b.cond_br(cond, body, &[header_i], exit, &[header_i]);

        b.switch_to(body);
        let bi = b.param(body, 0);
        let one = b.const_i64(i32t, 1);
        let next = b.add(bi, one, Flags::NONE);
        b.br(header, &[next]);

        b.switch_to(exit);
        let ev = b.param(exit, 0);
        b.ret(Some(ev));
    }

    // The key property: this returns at all. Under plain join the header value
    // would climb 0,1,2,… without bound; widening collapses it to Top.
    let r = solve::<Countup>(m.function(f), m.types(), m.consts());
    assert!(r.is_reachable(header));
    assert_eq!(*r.value(header_i), Countup::Top, "widening must lift the IV to Top");
}

#[test]
fn countup_widen_collapses_growth_but_join_does_not() {
    // Demonstrates why widening is needed: join keeps producing strictly larger
    // elements (an infinite ascending chain), while widen jumps to Top.
    let a = Countup::Finite(0);
    let b = Countup::Finite(1);
    assert_eq!(a.join(&b), Countup::Finite(1));
    assert_ne!(a.join(&b), a, "join strictly increases — the chain never converges");
    assert_eq!(a.widen(&a.join(&b)), Countup::Top, "widen collapses the growth");
    // Edge guard default: feasibility is unrefined for this domain.
    assert!(Countup::Finite(3).edge_feasible(&EdgeGuard::CondIs(true)));
}
