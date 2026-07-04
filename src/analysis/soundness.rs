//! The **soundness harness** — bet **B8**'s teeth in test form.
//!
//! A domain's transfer function is *sound* iff its abstract result γ-contains the
//! concrete result of the reference semantics ([`ir::eval`](crate::ir::eval)),
//! whenever each concrete operand lies in the γ of its abstract operand:
//!
//! ```text
//!   ∀ i. xᵢ ∈ γ(Aᵢ)   ⟹   eval(op, x⃗) ∈ γ( transfer(op, A⃗) )
//! ```
//!
//! This module *tests* that property by sampling. For each generated pure
//! integer instruction it picks concrete operands, abstracts each either exactly
//! (as its singleton constant) or loosely (as ⊤ — an operand the analysis does
//! not know), runs the concrete evaluator to get the oracle result, runs the
//! domain's transfer, and checks the containment. Undefined-behavior cases are
//! skipped (the transfer is only obliged to be sound where the concrete
//! semantics is defined). The sampler is a fixed-seed xorshift, so a run is
//! reproducible (tenet T5).
//!
//! # The `z3rs` path (domain fan-out)
//!
//! The random check is the required, always-on deliverable. The eventual
//! *complete* check (Phase 3 exit criterion "transfer-function soundness checks
//! pass") replaces sampling with a proof: encode γ and the transfer symbolically
//! and ask `z3rs` (which the crate already depends on) to discharge
//! `∀ x⃗, A⃗. (⋀ᵢ xᵢ ∈ γ(Aᵢ)) ⟹ eval(op,x⃗) ∈ γ(transfer(op,A⃗))`, reusing the same
//! QF_BV encoding of the opcode semantics the verifier's refinement tier builds
//! (`crate::verify::refinement`). A `sat` model of the negation is a concrete
//! counterexample — exactly the shape [`SoundnessReport`] already reports, so
//! the two checkers share an interface. Wiring that per-domain is the follow-up
//! for the ranges / known-bits / nullness domains.

use crate::analysis::domain::{AbstractDomain, DomainCtx, concrete_eval};
use crate::ir::inst::{BinOp, CastOp, Flags, InstData, InstKind, IntPred};
use crate::ir::types::{FloatKind, Type, TypeContext};
use crate::ir::value::Const;
use crate::ir::{SemValue, TypeId};

use puremp::Int;

/// The outcome of a soundness sampling run.
#[derive(Debug, Default)]
pub struct SoundnessReport {
    /// How many concrete cases were actually checked (defined, non-UB).
    pub checked: usize,
    /// How many cases were skipped (undefined behavior on the sampled operands).
    pub skipped: usize,
    /// Human-readable descriptions of any soundness violations found.
    pub violations: Vec<String>,
}

impl SoundnessReport {
    /// Whether every checked case was sound (no violations).
    pub fn is_sound(&self) -> bool {
        self.violations.is_empty()
    }
}

/// A tiny deterministic xorshift64 sampler — no external RNG dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// A sampled operand value of `width` bits, biased toward interesting
    /// patterns (0, 1, all-ones, sign bit) as well as uniform noise.
    fn operand(&mut self, width: u32) -> Int {
        match self.below(6) {
            0 => Int::ZERO,
            1 => Int::ONE,
            2 => Int::MINUS_ONE, // all-ones once masked
            3 => Int::ONE.mul_2k(width - 1), // sign bit / INT_MIN pattern
            _ => Int::from_u64(self.next_u64()),
        }
    }
}

/// The shape of one sampled instruction: its opcode, the operand types, the
/// result type, and the operand widths used to sample concrete values.
struct OpTemplate {
    kind: InstKind,
    flags: Flags,
    operand_tys: Vec<TypeId>,
    result_ty: TypeId,
}

/// Build the fixed pool of pure-integer op templates over a set of interned
/// widths. Flags are set on the arithmetic ops so the poison paths are covered.
fn templates(types: &mut TypeContext) -> Vec<OpTemplate> {
    let i1 = types.int(1);
    let i16 = types.int(16);
    let i32 = types.int(32);
    let i64 = types.int(64);

    let bin = |op: BinOp, flags: Flags| OpTemplate {
        kind: InstKind::Bin(op),
        flags,
        operand_tys: vec![i32, i32],
        result_ty: i32,
    };
    let cmp = |pred: IntPred| OpTemplate {
        kind: InstKind::ICmp(pred),
        flags: Flags::NONE,
        operand_tys: vec![i32, i32],
        result_ty: i1,
    };
    let cast = |op: CastOp, from: TypeId, to: TypeId| OpTemplate {
        kind: InstKind::Cast(op),
        flags: Flags::NONE,
        operand_tys: vec![from],
        result_ty: to,
    };

    vec![
        bin(BinOp::Add, Flags::NONE),
        bin(BinOp::Add, Flags::nsw()),
        bin(BinOp::Add, Flags::nuw()),
        bin(BinOp::Sub, Flags::NONE),
        bin(BinOp::Sub, Flags::nsw()),
        bin(BinOp::Mul, Flags::NONE),
        bin(BinOp::Mul, Flags::nsw()),
        bin(BinOp::And, Flags::NONE),
        bin(BinOp::Or, Flags::NONE),
        bin(BinOp::Xor, Flags::NONE),
        bin(BinOp::Shl, Flags::NONE),
        bin(BinOp::LShr, Flags::NONE),
        bin(BinOp::LShr, Flags::exact()),
        bin(BinOp::AShr, Flags::NONE),
        bin(BinOp::UDiv, Flags::NONE),
        bin(BinOp::SDiv, Flags::NONE),
        bin(BinOp::URem, Flags::NONE),
        bin(BinOp::SRem, Flags::NONE),
        cmp(IntPred::Eq),
        cmp(IntPred::Ne),
        cmp(IntPred::Ult),
        cmp(IntPred::Slt),
        cmp(IntPred::Uge),
        cmp(IntPred::Sgt),
        cast(CastOp::Trunc, i32, i16),
        cast(CastOp::ZExt, i32, i64),
        cast(CastOp::SExt, i32, i64),
    ]
}

/// The bit width to sample for an operand of `ty` (integers only here).
fn width_of(types: &TypeContext, ty: TypeId) -> u32 {
    match types.get(ty) {
        Type::Int(w) => *w,
        Type::Float(k) => k.bit_width(),
        _ => 0,
    }
}

/// Run `cases` random soundness checks for domain `D` over the pure-integer
/// opcodes, from `seed`.
///
/// For each case, every operand is abstracted either exactly (its singleton
/// constant) or as ⊤ (unknown), while the concrete value stays in the operand's
/// γ by construction — so any violation is a genuine transfer-soundness bug.
pub fn check_integer_transfer_sound<D: AbstractDomain>(cases: usize, seed: u64) -> SoundnessReport {
    let mut types = TypeContext::new();
    // Ensure a float type exists so `width_of` and unrelated interning are
    // stable; the harness itself only samples integers.
    let _ = types.float(FloatKind::F32);
    let pool = templates(&mut types);
    let ctx = DomainCtx::new(&types);

    let mut rng = Rng(seed | 1);
    let mut report = SoundnessReport::default();

    for _ in 0..cases {
        let tmpl = &pool[rng.below(pool.len())];

        // Sample concrete operands and their abstractions.
        let mut concrete = Vec::with_capacity(tmpl.operand_tys.len());
        let mut abstract_ops = Vec::with_capacity(tmpl.operand_tys.len());
        for &oty in &tmpl.operand_tys {
            let w = width_of(&types, oty);
            let value = rng.operand(w);
            concrete.push(SemValue::int(w, value.clone()));
            if rng.bool() {
                // Known: abstract exactly as the singleton constant.
                let c = Const::Int { ty: oty, value };
                abstract_ops.push(D::abstract_const(ctx, &c));
            } else {
                // Unknown: ⊤ still γ-contains the concrete value.
                abstract_ops.push(D::top());
            }
        }

        let inst = InstData {
            kind: tmpl.kind.clone(),
            flags: tmpl.flags,
            ty: tmpl.result_ty,
            operands: Vec::new(),
            result: None,
        };

        // Oracle: the concrete reference result (skip if UB / undefined here).
        let Some(concrete_result) = concrete_eval(&types, &inst, &concrete) else {
            report.skipped += 1;
            continue;
        };

        let abstract_result = D::transfer(ctx, &inst, &abstract_ops);
        report.checked += 1;
        if !abstract_result.contains(&concrete_result) {
            report.violations.push(format!(
                "unsound: {:?} flags={:?} operands={abstract_ops:?} \
                 concrete={concrete:?} -> concrete_result={concrete_result:?} \
                 not in gamma({abstract_result:?})",
                tmpl.kind, tmpl.flags,
            ));
        }
    }

    report
}
