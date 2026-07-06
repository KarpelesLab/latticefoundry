//! Tests for the `Certified` tier: producing certificates for real rewrites,
//! re-validating them, and — the whole point — *rejecting* forged or mismatched
//! certificates that would vouch for the wrong `src`/`tgt` pair.

use super::{
    CertRejection, PipelineCertificate, PipelineRejection, RefinementCertificate, Verdict,
    certify_pair, check_certificate, check_pipeline, fingerprint, is_certified, run_certified,
    run_pipeline_certified,
};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, Flags, IntPred};
use crate::ir::value::ValueId;
use crate::ir::{FuncId, Module, TypeId};
use crate::support::StrInterner;
use crate::transform::{Dce, Sccp};

/// A builder harness sharing one module (so all [`TypeId`]s / [`FuncId`]s stay
/// valid), able to construct several functions of a fixed `iN -> iN` signature.
struct Harness {
    module: Module,
    syms: StrInterner,
    ity: TypeId,
    sig: TypeId,
}

impl Harness {
    fn unary_int(width: u32) -> Harness {
        let mut module = Module::new("cert_test");
        let ity = module.types_mut().int(width);
        let sig = module.types_mut().func(vec![ity], ity, false);
        Harness { module, syms: StrInterner::new(), ity, sig }
    }

    /// Build a single-block `iN -> iN` function whose body returns `f(builder, param0)`.
    fn build(&mut self, name: &str, f: impl FnOnce(&mut FunctionBuilder, ValueId) -> ValueId) -> FuncId {
        let sym = self.syms.intern(name);
        let id = self.module.declare_function(sym, self.sig);
        {
            let mut b = self.module.build(id);
            let entry = b.create_entry_block();
            let x = b.block_params(entry)[0];
            let r = f(&mut b, x);
            b.ret(Some(r));
        }
        id
    }

    /// Build a function with a **loop** (a back-edge) that ultimately forwards its
    /// argument. Loops are outside the acyclic refinement subset, so the checker
    /// reports `Unknown` — an honest `Unproven` certificate.
    fn build_loop(&mut self, name: &str) -> FuncId {
        let ity = self.ity;
        let sym = self.syms.intern(name);
        let id = self.module.declare_function(sym, self.sig);
        {
            let mut b = self.module.build(id);
            let entry = b.create_entry_block();
            let x = b.block_params(entry)[0];
            let header = b.create_block(&[]);
            let exit = b.create_block(&[]);
            b.br(header, &[]);
            b.switch_to(header);
            let zero = b.const_i64(ity, 0);
            let c = b.icmp(IntPred::Ne, x, zero);
            // Back-edge header -> header keeps the CFG cyclic.
            b.cond_br(c, header, &[], exit, &[]);
            b.switch_to(exit);
            b.ret(Some(x));
        }
        id
    }

    fn func(&self, id: FuncId) -> &crate::ir::Function {
        self.module.function(id)
    }

    fn certify(&self, transform: &str, src: FuncId, tgt: FuncId) -> RefinementCertificate {
        certify_pair(
            transform,
            src,
            self.module.types(),
            self.module.consts(),
            self.func(src),
            self.func(tgt),
        )
    }
}

// ---------------------------------------------------------------------------
// Producing and accepting a certificate for a sound rewrite.
// ---------------------------------------------------------------------------

#[test]
fn verified_certificate_for_sound_rewrite_is_accepted() {
    // src = x * 2, tgt = x << 1 — a sound rewrite.
    let mut h = Harness::unary_int(8);
    let ity = h.ity;
    let src = h.build("mul2", move |b, x| {
        let two = b.const_i64(ity, 2);
        b.mul(x, two, Flags::NONE)
    });
    let tgt = h.build("shl1", move |b, x| {
        let one = b.const_i64(ity, 1);
        b.bin(BinOp::Shl, x, one, Flags::NONE)
    });

    let cert = h.certify("strength-reduce", src, tgt);
    assert_eq!(cert.verdict, Verdict::Verified, "sound rewrite must certify Verified");
    assert!(cert.claims_verified());

    // The checker re-validates it against the very same functions.
    assert!(is_certified(&cert, h.module.types(), h.module.consts(), h.func(src), h.func(tgt)));
    assert_eq!(
        check_certificate(&cert, h.module.types(), h.module.consts(), h.func(src), h.func(tgt)),
        Ok(()),
    );
}

// ---------------------------------------------------------------------------
// Forgery / tamper detection — the whole point.
// ---------------------------------------------------------------------------

#[test]
fn certificate_paired_with_wrong_target_is_rejected() {
    // A genuine Verified cert for (x*2 ⇒ x<<1) must NOT vouch for a *different*,
    // unsound target x+1 presented in its place.
    let mut h = Harness::unary_int(8);
    let ity = h.ity;
    let src = h.build("mul2", move |b, x| {
        let two = b.const_i64(ity, 2);
        b.mul(x, two, Flags::NONE)
    });
    let good_tgt = h.build("shl1", move |b, x| {
        let one = b.const_i64(ity, 1);
        b.bin(BinOp::Shl, x, one, Flags::NONE)
    });
    let bad_tgt = h.build("addone", move |b, x| {
        let one = b.const_i64(ity, 1);
        b.add(x, one, Flags::NONE)
    });

    let cert = h.certify("strength-reduce", src, good_tgt);
    assert_eq!(cert.verdict, Verdict::Verified);

    // Presented against the wrong target: rejected by fingerprint mismatch,
    // before the solver is even consulted.
    let res =
        check_certificate(&cert, h.module.types(), h.module.consts(), h.func(src), h.func(bad_tgt));
    assert!(
        matches!(res, Err(CertRejection::TgtFingerprintMismatch { .. })),
        "expected TgtFingerprintMismatch, got {res:?}",
    );
    assert!(!is_certified(&cert, h.module.types(), h.module.consts(), h.func(src), h.func(bad_tgt)));
}

#[test]
fn forged_verified_verdict_is_caught_by_recheck() {
    // A hand-forged certificate: it claims `Verified` for a genuinely unsound
    // pair (x ⇒ x+1) but carries the *correct* fingerprints, so it slips past the
    // fingerprint gate. The z3rs re-check must catch it.
    let mut h = Harness::unary_int(8);
    let ity = h.ity;
    let src = h.build("id", |_, x| x);
    let tgt = h.build("addone", move |b, x| {
        let one = b.const_i64(ity, 1);
        b.add(x, one, Flags::NONE)
    });

    // Honest certification says Refuted.
    let honest = h.certify("bogus", src, tgt);
    assert!(matches!(honest.verdict, Verdict::Refuted(_)), "got {:?}", honest.verdict);

    // Forge a `Verified` verdict with correct fingerprints.
    let forged = RefinementCertificate {
        transform: "bogus".to_string(),
        function: src,
        src_fingerprint: fingerprint(h.module.types(), h.module.consts(), h.func(src)),
        tgt_fingerprint: fingerprint(h.module.types(), h.module.consts(), h.func(tgt)),
        verdict: Verdict::Verified,
    };

    let res =
        check_certificate(&forged, h.module.types(), h.module.consts(), h.func(src), h.func(tgt));
    match res {
        Err(CertRejection::StaleVerdict(inner)) => {
            assert!(
                matches!(inner, crate::verify::RefinementResult::Counterexample(_)),
                "re-check should refute the forged pair, got {inner:?}",
            );
        }
        other => panic!("expected StaleVerdict from re-check, got {other:?}"),
    }
}

#[test]
fn swapped_source_function_is_rejected() {
    // The certificate must also be pinned to the right *source*: presenting a
    // different src is a mismatch too.
    let mut h = Harness::unary_int(8);
    let ity = h.ity;
    let src = h.build("mul2", move |b, x| {
        let two = b.const_i64(ity, 2);
        b.mul(x, two, Flags::NONE)
    });
    let tgt = h.build("shl1", move |b, x| {
        let one = b.const_i64(ity, 1);
        b.bin(BinOp::Shl, x, one, Flags::NONE)
    });
    let other_src = h.build("mul3", move |b, x| {
        let three = b.const_i64(ity, 3);
        b.mul(x, three, Flags::NONE)
    });

    let cert = h.certify("strength-reduce", src, tgt);
    let res =
        check_certificate(&cert, h.module.types(), h.module.consts(), h.func(other_src), h.func(tgt));
    assert!(
        matches!(res, Err(CertRejection::SrcFingerprintMismatch { .. })),
        "expected SrcFingerprintMismatch, got {res:?}",
    );
}

// ---------------------------------------------------------------------------
// Out-of-subset: an honest `Unproven` certificate, not accepted as verified.
// ---------------------------------------------------------------------------

#[test]
fn out_of_subset_yields_unproven_not_certified() {
    let mut h = Harness::unary_int(8);
    let src = h.build_loop("loop_src");
    let tgt = h.build_loop("loop_tgt");

    let cert = h.certify("noop", src, tgt);
    assert!(matches!(cert.verdict, Verdict::Unproven(_)), "got {:?}", cert.verdict);
    assert!(!cert.claims_verified());

    // The checker treats `Unproven` as *not certified*.
    let res = check_certificate(&cert, h.module.types(), h.module.consts(), h.func(src), h.func(tgt));
    assert!(
        matches!(res, Err(CertRejection::NotVerified(Verdict::Unproven(_)))),
        "expected NotVerified(Unproven), got {res:?}",
    );
    assert!(!is_certified(&cert, h.module.types(), h.module.consts(), h.func(src), h.func(tgt)));
}

// ---------------------------------------------------------------------------
// Pipeline certificate: composition and re-check.
// ---------------------------------------------------------------------------

#[test]
fn pipeline_certificate_composes_and_rechecks() {
    // A 2-step chain: IR0 = (x+0)*1  ⇒  IR1 = x*1  ⇒  IR2 = x.
    let mut h = Harness::unary_int(16);
    let ity = h.ity;
    let ir0 = h.build("ir0", move |b, x| {
        let zero = b.const_i64(ity, 0);
        let one = b.const_i64(ity, 1);
        let t = b.add(x, zero, Flags::NONE);
        b.mul(t, one, Flags::NONE)
    });
    let ir1 = h.build("ir1", move |b, x| {
        let one = b.const_i64(ity, 1);
        b.mul(x, one, Flags::NONE)
    });
    let ir2 = h.build("ir2", |_, x| x);

    let step0 = h.certify("simplify-add0", ir0, ir1);
    let step1 = h.certify("simplify-mul1", ir1, ir2);
    assert_eq!(step0.verdict, Verdict::Verified);
    assert_eq!(step1.verdict, Verdict::Verified);

    let pipeline = PipelineCertificate { steps: vec![step0, step1] };
    assert!(pipeline.all_claim_verified());
    assert!(pipeline.is_well_linked(), "step boundaries must chain by fingerprint");

    // Re-validate the whole chain against the IR snapshots.
    let chain = [h.func(ir0), h.func(ir1), h.func(ir2)];
    assert_eq!(check_pipeline(&pipeline, h.module.types(), h.module.consts(), &chain), Ok(()));

    // Tampering with an intermediate snapshot (feeding the wrong IR1) is caught.
    let bad_ir1 = h.build("bad_ir1", move |b, x| {
        let one = b.const_i64(ity, 1);
        b.add(x, one, Flags::NONE)
    });
    let bad_chain = [h.func(ir0), h.func(bad_ir1), h.func(ir2)];
    let res = check_pipeline(&pipeline, h.module.types(), h.module.consts(), &bad_chain);
    assert!(matches!(res, Err(PipelineRejection::Step { index: 0, .. })), "got {res:?}");

    // A wrong number of snapshots is rejected structurally.
    let short = [h.func(ir0), h.func(ir1)];
    assert!(matches!(
        check_pipeline(&pipeline, h.module.types(), h.module.consts(), &short),
        Err(PipelineRejection::ChainLengthMismatch { steps: 2, functions: 2 }),
    ));
}

// ---------------------------------------------------------------------------
// The certifying transform runner over real passes.
// ---------------------------------------------------------------------------

#[test]
fn run_certified_wraps_a_real_transform() {
    // A function SCCP will fold: b = mul x, (2 + 3)  ⇒  b = mul x, 5.
    let mut h = Harness::unary_int(8);
    let ity = h.ity;
    let make = |b: &mut FunctionBuilder, x: ValueId| {
        let two = b.const_i64(ity, 2);
        let three = b.const_i64(ity, 3);
        let a = b.add(two, three, Flags::NONE);
        b.mul(x, a, Flags::NONE)
    };
    // Keep an untouched twin to re-check the produced certificate against.
    let keep = h.build("keep", make);
    let work = h.build("work", make);

    let mut sccp = Sccp::new();
    sccp.analyze(h.func(work), h.module.types(), h.module.consts());
    let cert = run_certified(&mut h.module, work, &mut sccp);

    assert_eq!(cert.verdict, Verdict::Verified);
    assert_ne!(cert.src_fingerprint, cert.tgt_fingerprint, "SCCP changed the body");

    // `keep` is structurally the pre-image of `work`; the certificate re-validates
    // against (keep, work-after-SCCP).
    assert_eq!(
        check_certificate(&cert, h.module.types(), h.module.consts(), h.func(keep), h.func(work)),
        Ok(()),
    );
}

#[test]
fn run_pipeline_certified_over_two_passes() {
    // SCCP then DCE over one function; both steps certify.
    let mut h = Harness::unary_int(8);
    let ity = h.ity;
    let f = h.build("f", move |b, x| {
        let two = b.const_i64(ity, 2);
        let three = b.const_i64(ity, 3);
        let a = b.add(two, three, Flags::NONE);
        b.mul(x, a, Flags::NONE)
    });

    let mut sccp = Sccp::new();
    sccp.analyze(h.func(f), h.module.types(), h.module.consts());
    let mut dce = Dce;
    let mut passes: [&mut dyn crate::transform::FunctionTransform; 2] = [&mut sccp, &mut dce];
    let pipeline = run_pipeline_certified(&mut h.module, f, &mut passes);

    assert_eq!(pipeline.steps.len(), 2);
    assert!(pipeline.all_claim_verified(), "both steps should certify: {pipeline:?}");
}

// ---------------------------------------------------------------------------
// Fingerprint determinism.
// ---------------------------------------------------------------------------

#[test]
fn fingerprint_is_deterministic() {
    let mut h = Harness::unary_int(8);
    let ity = h.ity;
    let make = move |b: &mut FunctionBuilder, x: ValueId| {
        let two = b.const_i64(ity, 2);
        b.mul(x, two, Flags::NONE)
    };
    let a = h.build("a", make);
    let b = h.build("b", make);
    let different = h.build("c", move |bld, x| {
        let three = b_const(bld, ity, 3);
        bld.mul(x, three, Flags::NONE)
    });

    let fa1 = fingerprint(h.module.types(), h.module.consts(), h.func(a));
    let fa2 = fingerprint(h.module.types(), h.module.consts(), h.func(a));
    let fb = fingerprint(h.module.types(), h.module.consts(), h.func(b));
    let fc = fingerprint(h.module.types(), h.module.consts(), h.func(different));

    // Same function, recomputed — identical.
    assert_eq!(fa1, fa2);
    // Structurally identical bodies (built the same way) — identical.
    assert_eq!(fa1, fb);
    // A structurally different body — a different fingerprint.
    assert_ne!(fa1, fc);
}

/// Small helper so the "different" builder above reads cleanly.
fn b_const(b: &mut FunctionBuilder, ty: TypeId, v: i64) -> ValueId {
    b.const_i64(ty, v)
}
