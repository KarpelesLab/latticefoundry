//! The **`Certified`** verification tier (bet **B3**, tenet **T3**):
//! *proof-carrying IR*.
//!
//! Where the [`Refinement`](super::refinement) tier discharges a rewrite
//! obligation *every time a build runs*, the `Certified` tier discharges it
//! **once** and records the outcome as a [`RefinementCertificate`]. A downstream
//! consumer that trusts neither the optimizer nor the person who ran it can then
//! re-establish soundness by running only [`check_certificate`] — a small trusted
//! base (this checker + `z3rs`) — rather than re-deriving the whole optimization.
//!
//! ## What a certificate attests
//!
//! A [`RefinementCertificate`] is a claim about *one* transformation step
//! `src ⇒ tgt` on *one* function: the transform's name, an identifier of the
//! function, a [`Verdict`] (proven / refuted / unproven), and a **structural
//! fingerprint** ([`StructHash`]) of each of `src` and `tgt`. The fingerprint is
//! what pins a certificate to a specific pair of function bodies: a certificate
//! whose recorded `tgt` fingerprint does not match the function it is presented
//! against is *about a different rewrite* and is rejected out of hand, before the
//! solver is even consulted.
//!
//! A [`PipelineCertificate`] is an ordered chain of these steps
//! (`IR₀ ⇒ IR₁ ⇒ … ⇒ IRₙ`). Because refinement composes, a chain in which every
//! step is a proven refinement witnesses that the whole run `IR₀ ⇒ IRₙ` is a
//! refinement.
//!
//! ## Honesty about the supported subset
//!
//! The underlying [`check_refinement`](super::check_refinement) proves only the
//! single-block pure-integer subset; everything else (multi-block control flow,
//! memory, calls, floats) comes back as [`RefinementResult::Unknown`]. Production
//! records that faithfully as [`Verdict::Unproven`] — it **never** fabricates a
//! [`Verdict::Verified`]. The checker treats `Unproven` (and `Refuted`) as *not
//! certified*: only a `Verified` verdict whose z3rs re-check still returns
//! `Refines` is accepted.

use std::hash::{Hash, Hasher};

use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ConstId, ConstPool, FloatBits, ValueDef, ValueId};
use crate::ir::{FuncId, Function, InstId, Module};
use crate::pass::Changed;
use crate::transform::FunctionTransform;

use super::refinement::{RefinementResult, check_refinement};

// ---------------------------------------------------------------------------
// Verification tiers (the dial of `docs/design-tenets.md` §2).
// ---------------------------------------------------------------------------

/// The verification dial of `docs/design-tenets.md` §2. The semantics are
/// identical at every setting; only *how much is checked* changes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum VerificationTier {
    /// No checks at all (perf-critical release builds).
    Off,
    /// The cheap, solver-free structural + type invariants (the default).
    #[default]
    Structural,
    /// Each pass emits a refinement obligation discharged by `z3rs`.
    Refinement,
    /// Obligations are discharged once and cached as certificates; builds
    /// *check* certificates ([`check_certificate`]) instead of re-proving.
    Certified,
}

impl VerificationTier {
    /// Whether this tier discharges (or re-checks) refinement obligations via
    /// `z3rs` — true for [`Refinement`](VerificationTier::Refinement) and
    /// [`Certified`](VerificationTier::Certified).
    pub fn uses_solver(self) -> bool {
        matches!(self, VerificationTier::Refinement | VerificationTier::Certified)
    }
}

// ---------------------------------------------------------------------------
// Structural fingerprint.
// ---------------------------------------------------------------------------

/// A deterministic structural fingerprint of a function body.
///
/// Two functions with the same signature, blocks, instructions (opcode payload,
/// flags, result type, and operand *shape*), and constants hash equal; any
/// structural difference — a different opcode, operand, constant, or type —
/// changes the hash. It is computed by [`fingerprint`] with a fixed FNV-1a fold,
/// so the same body always yields the same value (within a host).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StructHash(u64);

impl StructHash {
    /// The raw 64-bit hash value.
    pub fn value(self) -> u64 {
        self.0
    }
}

/// A deterministic FNV-1a 64-bit hasher. Unlike the standard library's default
/// hasher this is *fixed* (no random seed), so a fingerprint is reproducible.
#[derive(Clone, Copy, Debug)]
struct Fnv(u64);

impl Fnv {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Fnv(Self::OFFSET)
    }
}

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
}

/// Compute the [`StructHash`] of `func`, resolving its constants and types
/// through the shared `types`/`consts` tables so the fingerprint reflects
/// *structure and semantics*, not incidental interning indices.
pub fn fingerprint(types: &TypeContext, consts: &ConstPool, func: &Function) -> StructHash {
    let mut h = Fnv::new();
    h.write(b"lf-fn-fingerprint-v1");
    hash_type(&mut h, types, func.sig);
    h.write_u32(func.block_count() as u32);
    for (bb, block) in func.blocks() {
        h.write_u32(bb.index() as u32);
        h.write_u32(block.params().len() as u32);
        for &p in block.params() {
            hash_type(&mut h, types, func.value_type(p));
        }
        for &iid in block.insts() {
            hash_inst(&mut h, types, consts, func, iid);
        }
        match block.terminator() {
            Some(t) => {
                h.write_u8(0xFF);
                hash_inst(&mut h, types, consts, func, t);
            }
            None => h.write_u8(0x00),
        }
    }
    StructHash(h.finish())
}

fn hash_inst(h: &mut Fnv, types: &TypeContext, consts: &ConstPool, func: &Function, iid: InstId) {
    let inst = func.inst(iid);
    inst.kind.hash(h);
    inst.flags.hash(h);
    hash_type(h, types, inst.ty);
    h.write_u32(inst.operands().len() as u32);
    for &o in inst.operands() {
        hash_value(h, types, consts, func, o);
    }
    h.write_u8(u8::from(inst.result().is_some()));
}

/// Hash a value *token*: how it is produced (with defining ids for instruction
/// results and block parameters, so the data-flow shape is captured) plus its
/// resolved type. Constants are folded in structurally.
fn hash_value(h: &mut Fnv, types: &TypeContext, consts: &ConstPool, func: &Function, v: ValueId) {
    let value = func.value(v);
    match &value.def {
        ValueDef::Inst(iid) => {
            h.write_u8(1);
            h.write_u32(iid.index() as u32);
        }
        ValueDef::Param(bb, idx) => {
            h.write_u8(2);
            h.write_u32(bb.index() as u32);
            h.write_u32(*idx);
        }
        ValueDef::Const(cid) => {
            h.write_u8(3);
            hash_const(h, types, consts, *cid);
        }
        ValueDef::Global(g) => {
            h.write_u8(4);
            h.write_u32(g.index() as u32);
        }
        ValueDef::Func(f) => {
            h.write_u8(5);
            h.write_u32(f.index() as u32);
        }
    }
    hash_type(h, types, value.ty);
}

fn hash_const(h: &mut Fnv, types: &TypeContext, consts: &ConstPool, cid: ConstId) {
    match consts.get(cid) {
        Const::Int { ty, value } => {
            h.write_u8(1);
            hash_type(h, types, *ty);
            h.write(format!("{value}").as_bytes());
        }
        Const::Float { ty, bits } => {
            h.write_u8(2);
            hash_type(h, types, *ty);
            match bits {
                FloatBits::F16(x) => {
                    h.write_u8(16);
                    h.write_u16(*x);
                }
                FloatBits::F32(x) => {
                    h.write_u8(32);
                    h.write_u32(*x);
                }
                FloatBits::F64(x) => {
                    h.write_u8(64);
                    h.write_u64(*x);
                }
            }
        }
        Const::Null(ty) => {
            h.write_u8(3);
            hash_type(h, types, *ty);
        }
        Const::Poison(ty) => {
            h.write_u8(4);
            hash_type(h, types, *ty);
        }
        Const::Aggregate { ty, elems } => {
            h.write_u8(5);
            hash_type(h, types, *ty);
            h.write_u32(elems.len() as u32);
            for &e in elems {
                hash_const(h, types, consts, e);
            }
        }
    }
}

fn hash_type(h: &mut Fnv, types: &TypeContext, ty: TypeId) {
    match types.get(ty) {
        Type::Void => h.write_u8(0),
        Type::Int(w) => {
            h.write_u8(1);
            h.write_u32(*w);
        }
        Type::Float(k) => {
            h.write_u8(2);
            h.write_u32(k.bit_width());
        }
        Type::Ptr => h.write_u8(3),
        Type::Array(elem, n) => {
            h.write_u8(4);
            hash_type(h, types, *elem);
            h.write_u64(*n);
        }
        Type::Struct(fields) => {
            h.write_u8(5);
            h.write_u32(fields.len() as u32);
            for &f in fields {
                hash_type(h, types, f);
            }
        }
        Type::Func(ft) => {
            h.write_u8(6);
            h.write_u32(ft.params.len() as u32);
            for &p in &ft.params {
                hash_type(h, types, p);
            }
            hash_type(h, types, ft.ret);
            h.write_u8(u8::from(ft.variadic));
        }
    }
}

// ---------------------------------------------------------------------------
// The certificate types.
// ---------------------------------------------------------------------------

/// The verdict a certificate records for a step. This is the certificate-level
/// mirror of [`RefinementResult`]: `Verified` is the *only* value the checker
/// accepts as a certification of soundness.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// `tgt` provably refines `src` (`z3rs` returned `unsat` of the negated
    /// obligation).
    Verified,
    /// The rewrite is provably **unsound**; the string is a distinguishing input.
    Refuted(String),
    /// No proof was produced — an out-of-subset construct, a solver `unknown`, or
    /// a solver error. Honestly *not* a proof (per the tiers in
    /// `docs/design-tenets.md` §2).
    Unproven(String),
}

impl Verdict {
    /// The verdict corresponding to a [`RefinementResult`].
    fn from_result(r: RefinementResult) -> Verdict {
        match r {
            RefinementResult::Refines => Verdict::Verified,
            RefinementResult::Counterexample(c) => Verdict::Refuted(c),
            RefinementResult::Unknown(m) => Verdict::Unproven(m),
        }
    }

    /// Whether this verdict is a proof of refinement.
    pub fn is_verified(&self) -> bool {
        matches!(self, Verdict::Verified)
    }
}

/// A certificate attesting a single transformation step `src ⇒ tgt`.
///
/// It records enough to be *independently re-validated* by [`check_certificate`]:
/// which transform produced it, which function it is about, the two structural
/// fingerprints that pin it to a specific `src`/`tgt` pair, and the [`Verdict`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RefinementCertificate {
    /// The name of the transform that produced this step.
    pub transform: String,
    /// An identifier of the function this step transformed.
    pub function: FuncId,
    /// Structural fingerprint of the source (`src`) function body.
    pub src_fingerprint: StructHash,
    /// Structural fingerprint of the target (`tgt`) function body.
    pub tgt_fingerprint: StructHash,
    /// The recorded verdict for `tgt ⊑ src`.
    pub verdict: Verdict,
}

impl RefinementCertificate {
    /// Whether this certificate *claims* a proof (its verdict is
    /// [`Verdict::Verified`]). This is only the claim — [`check_certificate`]
    /// decides whether the claim actually holds against the real functions.
    pub fn claims_verified(&self) -> bool {
        self.verdict.is_verified()
    }
}

/// An ordered chain of [`RefinementCertificate`]s attesting a whole optimization
/// run `IR₀ ⇒ IR₁ ⇒ … ⇒ IRₙ`. Because refinement composes, a chain of proven
/// steps witnesses that the run as a whole is a refinement.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct PipelineCertificate {
    /// The per-step certificates, in application order.
    pub steps: Vec<RefinementCertificate>,
}

impl PipelineCertificate {
    /// An empty pipeline certificate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether every step *claims* a proof. This is a cheap, solver-free
    /// necessary condition; [`check_pipeline`] does the real re-validation.
    pub fn all_claim_verified(&self) -> bool {
        self.steps.iter().all(RefinementCertificate::claims_verified)
    }

    /// Whether the chain *links up*: each step's source fingerprint equals the
    /// previous step's target fingerprint, so the certificates describe one
    /// contiguous `IR₀ ⇒ … ⇒ IRₙ` run rather than unrelated rewrites. This checks
    /// only the recorded fingerprints (no functions, no solver).
    pub fn is_well_linked(&self) -> bool {
        self.steps
            .windows(2)
            .all(|w| w[0].tgt_fingerprint == w[1].src_fingerprint)
    }
}

// ---------------------------------------------------------------------------
// Producing certificates.
// ---------------------------------------------------------------------------

/// Produce a [`RefinementCertificate`] for a before/after function pair by
/// fingerprinting both and discharging `tgt ⊑ src` through
/// [`check_refinement`](super::check_refinement). An out-of-subset pair yields an
/// honest [`Verdict::Unproven`] — never a fabricated `Verified`.
pub fn certify_pair(
    transform: &str,
    function: FuncId,
    types: &TypeContext,
    consts: &ConstPool,
    src: &Function,
    tgt: &Function,
) -> RefinementCertificate {
    let src_fingerprint = fingerprint(types, consts, src);
    let tgt_fingerprint = fingerprint(types, consts, tgt);
    let verdict = Verdict::from_result(check_refinement(types, consts, src, tgt));
    RefinementCertificate {
        transform: transform.to_string(),
        function,
        src_fingerprint,
        tgt_fingerprint,
        verdict,
    }
}

/// Run `transform` over function `id` of `module`, install the result if it
/// changed the body, and return a [`RefinementCertificate`] attesting the step.
///
/// This is the certifying wrapper around a [`FunctionTransform`]: it performs the
/// functional rebuild via [`Module::map_function`] and then certifies the old and
/// new bodies with [`certify_pair`]. A transform that reports [`Changed::No`]
/// leaves the body untouched and yields a trivial identity certificate
/// (a function refines itself).
pub fn run_certified(
    module: &mut Module,
    id: FuncId,
    transform: &mut dyn FunctionTransform,
) -> RefinementCertificate {
    let name = transform.name().to_string();
    let (fresh, changed) = module.map_function(id, |old, b| transform.run(old, b));
    let cert = if changed == Changed::Yes {
        certify_pair(&name, id, module.types(), module.consts(), module.function(id), &fresh)
    } else {
        // The rebuilt body is discarded (per the transform contract); the step is
        // the identity `old ⇒ old`, which refines itself.
        let old = module.function(id);
        certify_pair(&name, id, module.types(), module.consts(), old, old)
    };
    if changed == Changed::Yes {
        module.replace_function(id, fresh);
    }
    cert
}

/// Run a sequence of transforms over function `id`, installing each step's result
/// and collecting a [`PipelineCertificate`] over the whole run.
pub fn run_pipeline_certified(
    module: &mut Module,
    id: FuncId,
    transforms: &mut [&mut dyn FunctionTransform],
) -> PipelineCertificate {
    let mut steps = Vec::with_capacity(transforms.len());
    for t in transforms.iter_mut() {
        steps.push(run_certified(module, id, &mut **t));
    }
    PipelineCertificate { steps }
}

// ---------------------------------------------------------------------------
// The certificate CHECKER (the small trusted base).
// ---------------------------------------------------------------------------

/// Why [`check_certificate`] rejected a certificate.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CertRejection {
    /// The recorded source fingerprint does not match the presented `src`
    /// function — the certificate is about a *different* source body.
    SrcFingerprintMismatch {
        /// The fingerprint recorded in the certificate.
        recorded: StructHash,
        /// The fingerprint of the function actually presented.
        actual: StructHash,
    },
    /// The recorded target fingerprint does not match the presented `tgt`
    /// function — the certificate is about a *different* target body.
    TgtFingerprintMismatch {
        /// The fingerprint recorded in the certificate.
        recorded: StructHash,
        /// The fingerprint of the function actually presented.
        actual: StructHash,
    },
    /// The certificate's verdict is not [`Verdict::Verified`], so it certifies
    /// nothing (a refuted or unproven step is not a proof of soundness).
    NotVerified(Verdict),
    /// A [`Verdict::Verified`] certificate whose `z3rs` re-check no longer proves
    /// refinement — a forged or stale certificate. Carries the re-check result.
    StaleVerdict(RefinementResult),
}

/// **Re-validate** a certificate against the actual `src`/`tgt` functions — the
/// `Certified` tier's trusted base.
///
/// Independently of who produced `cert`, this:
/// 1. recomputes both structural fingerprints and rejects on any mismatch (the
///    certificate is not about *these* two functions);
/// 2. requires the verdict to be [`Verdict::Verified`];
/// 3. re-runs [`check_refinement`](super::check_refinement) and requires it to
///    still return [`RefinementResult::Refines`].
///
/// A forged `Verified` verdict (step 3 disagrees) or a mismatched/mislabelled
/// pair (step 1) is rejected — so a consumer can trust an accepted certificate by
/// running only this checker plus `z3rs`, never the optimizer that produced it.
pub fn check_certificate(
    cert: &RefinementCertificate,
    types: &TypeContext,
    consts: &ConstPool,
    src: &Function,
    tgt: &Function,
) -> Result<(), CertRejection> {
    // (1) The certificate must be *about these two functions*.
    let src_fp = fingerprint(types, consts, src);
    if src_fp != cert.src_fingerprint {
        return Err(CertRejection::SrcFingerprintMismatch {
            recorded: cert.src_fingerprint,
            actual: src_fp,
        });
    }
    let tgt_fp = fingerprint(types, consts, tgt);
    if tgt_fp != cert.tgt_fingerprint {
        return Err(CertRejection::TgtFingerprintMismatch {
            recorded: cert.tgt_fingerprint,
            actual: tgt_fp,
        });
    }

    // (2) Only a `Verified` verdict certifies anything.
    if !cert.verdict.is_verified() {
        return Err(CertRejection::NotVerified(cert.verdict.clone()));
    }

    // (3) Re-discharge the obligation: a `Verified` claim that no longer proves
    // `Refines` is forged or stale.
    match check_refinement(types, consts, src, tgt) {
        RefinementResult::Refines => Ok(()),
        other => Err(CertRejection::StaleVerdict(other)),
    }
}

/// Convenience: whether [`check_certificate`] accepts `cert`.
pub fn is_certified(
    cert: &RefinementCertificate,
    types: &TypeContext,
    consts: &ConstPool,
    src: &Function,
    tgt: &Function,
) -> bool {
    check_certificate(cert, types, consts, src, tgt).is_ok()
}

/// Why [`check_pipeline`] rejected a pipeline certificate.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PipelineRejection {
    /// The number of supplied functions is not `steps + 1` (a chain of `n` steps
    /// needs the `n + 1` IR snapshots `IR₀ … IRₙ`).
    ChainLengthMismatch {
        /// Number of certificate steps.
        steps: usize,
        /// Number of functions supplied.
        functions: usize,
    },
    /// Step `index` failed re-validation; carries the underlying reason.
    Step {
        /// The zero-based index of the failing step.
        index: usize,
        /// Why that step was rejected.
        reason: CertRejection,
    },
}

/// Re-validate a whole [`PipelineCertificate`] against the chain of IR snapshots
/// it describes.
///
/// `funcs` is the ordered chain `[IR₀, IR₁, …, IRₙ]` (one more than the number of
/// steps). Each step `i` is re-checked against `(funcs[i], funcs[i+1])` with
/// [`check_certificate`], so every link's fingerprints and `z3rs` proof are
/// re-established. All steps proving `Refines` means the composition `IR₀ ⇒ IRₙ`
/// is a refinement.
pub fn check_pipeline(
    pipeline: &PipelineCertificate,
    types: &TypeContext,
    consts: &ConstPool,
    funcs: &[&Function],
) -> Result<(), PipelineRejection> {
    if funcs.len() != pipeline.steps.len() + 1 {
        return Err(PipelineRejection::ChainLengthMismatch {
            steps: pipeline.steps.len(),
            functions: funcs.len(),
        });
    }
    for (index, step) in pipeline.steps.iter().enumerate() {
        check_certificate(step, types, consts, funcs[index], funcs[index + 1])
            .map_err(|reason| PipelineRejection::Step { index, reason })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
