//! The **`Refinement`** verification tier (bet **B2**, tenet **T3**): every
//! optimization is a *checked refinement*.
//!
//! This module is the first cut of the refinement checker. It takes a rewrite
//! `src ⇒ tgt` — two functions with the same signature, each a single
//! straight-line block of **pure** value-producing ops ending in `ret <value>`
//! — encodes both into SMT-LIB2 **QF_BV**, and asks [`z3rs`](crate::verify::smt)
//! whether `tgt` refines `src`. `unsat` of the negated obligation means the
//! rewrite is a sound refinement; `sat` yields a counterexample input; anything
//! else (`unknown`, a solver error, an unsupported construct) is reported as a
//! *sound non-answer* ([`RefinementResult::Unknown`]) and **never** mistaken for
//! a proof (per the tiers in `docs/design-tenets.md` §2).
//!
//! ## The value model: a bit-vector paired with a poison bit
//!
//! Each SSA integer value is encoded as a pair of SMT terms that exactly mirror
//! the concrete [`SemValue`](crate::ir::SemValue) model of
//! [`crate::ir::semantics`]:
//!
//! - `val : (_ BitVec N)` — the `N`-bit two's-complement bit pattern, and
//! - `poison : Bool` — whether the value is [poison](crate::ir::Const::Poison).
//!
//! Every opcode contributes a **value relation** (the bit-vector it computes)
//! and a **poison relation**. Poison propagates from any operand
//! (`res_poison = op0_poison ∨ op1_poison ∨ …`), plus each set flag adds a
//! *flag-violation* disjunct encoded in QF_BV to match `ir::semantics` exactly:
//!
//! - `nsw` / `nuw` overflow of `add`/`sub`/`mul`/`shl` — detected by redoing the
//!   op in a widened bit-vector (sign- or zero-extended) and checking the exact
//!   result differs from the (sign/zero)-extension of the `N`-bit wrapped
//!   result: i.e. the reduction modulo `2^N` lost information.
//! - `exact` on `udiv`/`sdiv`/`lshr`/`ashr` — a nonzero remainder / a
//!   shifted-out set bit, tested with a shift/round-trip identity.
//! - over-wide shift (`amount ≥ N`) — a `bvuge` against the width.
//!
//! `select` and `freeze` have the bespoke poison rules of the reference
//! semantics: a poison *condition* poisons `select` but a non-selected poison
//! arm does not; `freeze` clears poison, mapping a poison operand to a *fresh
//! unconstrained* bit-vector (so a frozen poison may be **any** value — which is
//! exactly what makes replacing a defined value by `freeze(poison)` fail
//! refinement).
//!
//! ## UB as a precondition
//!
//! Division/remainder by zero and `sdiv`/`srem` of `INT_MIN` by `-1` are the
//! only **undefined behavior** in this subset (`docs/ir-design.md` §10). Per the
//! refinement contract, the obligation is *conditional on the source being
//! UB-free*, so each `div`/`rem` in `src` contributes a UB condition that the
//! query **assumes does not occur**; symmetrically, a `div`/`rem` in `tgt`
//! contributes a UB condition that, if reachable, *breaks* refinement (the
//! target must not introduce UB the source ruled out). UB fires only on
//! non-poison operands, matching the poison-before-UB order of `ir::semantics`.
//!
//! ## The refinement relation
//!
//! With `s = src_ret` and `t = tgt_ret` (same width), a single value refines
//! per `docs/ir-design.md` §5:
//!
//! ```text
//! refines(s, t)  ≡  s.poison ∨ (¬t.poison ∧ s.val = t.val)
//! ```
//!
//! i.e. a poison source licenses any target, a defined source pins the target.
//! The whole obligation is
//!
//! ```text
//! ∀ inputs.  ¬src_ub  ⇒  ( ¬tgt_ub ∧ refines(src_ret, tgt_ret) )
//! ```
//!
//! and we hand the solver its **negation**; `unsat` ⇒ the rewrite is sound.
//!
//! ## Scope of this first cut
//!
//! In scope: integer (`Int(N)`) pure ops — `add`/`sub`/`mul`, `udiv`/`sdiv`/
//! `urem`/`srem`, `and`/`or`/`xor`, `shl`/`lshr`/`ashr`, `icmp`, `select`,
//! `freeze`, and the integer casts `trunc`/`zext`/`sext`; all their flags. Out
//! of scope (cleanly reported as [`RefinementResult::Unknown`], never a false
//! `Refines`): floating point, pointers/memory, `call`, multi-block control
//! flow, and non-integer casts.

use std::collections::HashMap;

use crate::ir::inst::{BinOp, CastOp, Flags, InstData, InstKind, IntPred};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ConstPool, ValueDef, ValueId};
use crate::ir::{FuncId, Function, Module};

use super::smt::z3rs;

/// The verdict of a refinement check.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RefinementResult {
    /// `tgt` provably refines `src`: the rewrite is sound (`unsat` of the
    /// negated obligation).
    Refines,
    /// The rewrite is **unsound**; the string is the solver's model of an input
    /// that distinguishes the two functions (`sat`).
    Counterexample(String),
    /// No proof was produced — an `unknown` from the solver, a solver error, or
    /// an out-of-scope construct. Per the verification tiers this is a *sound
    /// non-answer*: it is never treated as a proof of refinement.
    Unknown(String),
}

impl RefinementResult {
    /// Whether this result is a proof of refinement.
    pub fn is_refines(&self) -> bool {
        matches!(self, RefinementResult::Refines)
    }
}

/// The `Refinement`-tier entry point the driver (`lf-opt`) calls to discharge a
/// rewrite obligation over two functions of a [`Module`].
///
/// This is the thin, module-facing wrapper around [`check_refinement`]: it
/// resolves the two [`FuncId`]s against the module's shared type/constant tables
/// and runs the check.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefinementTier;

impl RefinementTier {
    /// Check that `tgt` refines `src` (both functions of `module`).
    pub fn check(self, module: &Module, src: FuncId, tgt: FuncId) -> RefinementResult {
        check_refinement(module.types(), module.consts(), module.function(src), module.function(tgt))
    }
}

/// Check that `tgt` **refines** `src` for the single-block pure-integer subset,
/// discharging the obligation through `z3rs`.
///
/// `types` and `consts` are the shared interning tables the two functions were
/// built against (from their owning [`Module`]). Returns [`RefinementResult`];
/// anything outside the supported subset yields [`RefinementResult::Unknown`],
/// never a false [`RefinementResult::Refines`].
pub fn check_refinement(
    types: &TypeContext,
    consts: &ConstPool,
    src: &Function,
    tgt: &Function,
) -> RefinementResult {
    match build_query(types, consts, src, tgt) {
        Ok(query) => decide(&query.script, &query.input_names),
        Err(EncErr(reason)) => RefinementResult::Unknown(format!("unsupported: {reason}")),
    }
}

// ---------------------------------------------------------------------------
// Errors and small value types.
// ---------------------------------------------------------------------------

/// An out-of-scope construct was encountered while encoding.
struct EncErr(String);

fn unsupported(what: impl Into<String>) -> EncErr {
    EncErr(what.into())
}

/// The encoded `(value, poison)` SMT terms of one SSA value, plus its width.
#[derive(Clone)]
struct Sym {
    /// An SMT term of sort `(_ BitVec width)` — the bit pattern.
    val: String,
    /// An SMT term of sort `Bool` — whether the value is poison.
    poison: String,
    /// The bit width `N`.
    width: u32,
}

/// A fully built SMT-LIB2 query plus the shared input names (for `get-value`).
struct Query {
    script: String,
    input_names: Vec<String>,
}

/// The encoding of one function: its return value and its UB conditions.
struct EncodedFn {
    ret: Sym,
    ub: Vec<String>,
}

// ---------------------------------------------------------------------------
// Query construction.
// ---------------------------------------------------------------------------

fn build_query(
    types: &TypeContext,
    consts: &ConstPool,
    src: &Function,
    tgt: &Function,
) -> Result<Query, EncErr> {
    let src_params = single_block_params_ret(src)?.0;
    let tgt_params = single_block_params_ret(tgt)?.0;

    // Signatures must match: same parameter widths and same return width. We
    // compare the concrete entry-parameter types (the function's parameters).
    if src_params.len() != tgt_params.len() {
        return Err(unsupported("parameter count mismatch between src and tgt"));
    }

    let mut enc = Enc { types, consts, out: String::new(), fresh: 0 };
    enc.line("(set-logic QF_BV)");

    // Shared symbolic inputs: one bit-vector + one poison bool per parameter,
    // universally quantified (they are free constants in the check-sat).
    let mut inputs = Vec::with_capacity(src_params.len());
    let mut input_names = Vec::new();
    for (i, &p) in src_params.iter().enumerate() {
        let w = int_width(types, src.value_type(p))
            .ok_or_else(|| unsupported("non-integer parameter"))?;
        let tw = int_width(types, tgt.value_type(tgt_params[i]))
            .ok_or_else(|| unsupported("non-integer parameter"))?;
        if w != tw {
            return Err(unsupported("parameter width mismatch between src and tgt"));
        }
        let vn = format!("in{i}");
        let pn = format!("in{i}_p");
        enc.declare_bv(&vn, w);
        enc.declare_bool(&pn);
        inputs.push(Sym { val: vn.clone(), poison: pn.clone(), width: w });
        input_names.push(vn);
        input_names.push(pn);
    }

    let src_enc = enc.encode_function(src, "s", &inputs)?;
    let tgt_enc = enc.encode_function(tgt, "t", &inputs)?;

    if src_enc.ret.width != tgt_enc.ret.width {
        return Err(unsupported("return width mismatch between src and tgt"));
    }

    // refines(s, t) ≡ s.poison ∨ (¬t.poison ∧ s.val = t.val)
    let refines = format!(
        "(or {sp} (and (not {tp}) (= {sv} {tv})))",
        sp = src_enc.ret.poison,
        tp = tgt_enc.ret.poison,
        sv = src_enc.ret.val,
        tv = tgt_enc.ret.val,
    );
    let src_ub = or_terms(&src_enc.ub);
    let tgt_ub = or_terms(&tgt_enc.ub);

    // Negated obligation: src is UB-free, yet tgt is UB or fails to refine.
    enc.assert(&format!("(and (not {src_ub}) (or {tgt_ub} (not {refines})))"));
    enc.line("(check-sat)");

    Ok(Query { script: enc.out, input_names })
}

/// Disjoin a list of boolean terms, collapsing to `false`/the single term.
fn or_terms(terms: &[String]) -> String {
    match terms {
        [] => "false".to_string(),
        [only] => only.clone(),
        many => format!("(or {})", many.join(" ")),
    }
}

/// Require a function to be a single block ending in `ret <value>`; return its
/// entry parameters and the returned value.
fn single_block_params_ret(func: &Function) -> Result<(Vec<ValueId>, ValueId), EncErr> {
    if func.block_count() != 1 {
        return Err(unsupported("multiple blocks (control flow) not supported"));
    }
    let entry = func.entry().ok_or_else(|| unsupported("function has no body"))?;
    let block = func.block(entry);
    let term_id = block.terminator().ok_or_else(|| unsupported("block is not terminated"))?;
    let term = func.inst(term_id);
    match term.kind {
        InstKind::Ret => {
            let ops = term.operands();
            if ops.len() != 1 {
                return Err(unsupported("only value-returning `ret` is supported"));
            }
            Ok((block.params().to_vec(), ops[0]))
        }
        _ => Err(unsupported("terminator is not `ret`")),
    }
}

// ---------------------------------------------------------------------------
// The encoder.
// ---------------------------------------------------------------------------

/// Accumulates one SMT-LIB2 script across both functions, minting fresh names.
struct Enc<'a> {
    types: &'a TypeContext,
    consts: &'a ConstPool,
    out: String,
    fresh: u32,
}

impl Enc<'_> {
    fn line(&mut self, s: &str) {
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn declare_bv(&mut self, name: &str, width: u32) {
        self.line(&format!("(declare-fun {name} () {})", bv_sort(width)));
    }

    fn declare_bool(&mut self, name: &str) {
        self.line(&format!("(declare-fun {name} () Bool)"));
    }

    fn assert(&mut self, term: &str) {
        self.line(&format!("(assert {term})"));
    }

    /// A fresh, unconstrained bit-vector constant of the given width.
    fn fresh_bv(&mut self, width: u32) -> String {
        let name = format!("fresh{}", self.fresh);
        self.fresh += 1;
        self.declare_bv(&name, width);
        name
    }

    /// Bind a result value to named `val`/`poison` symbols and assert their
    /// defining relations, returning the [`Sym`].
    fn bind(&mut self, prefix: &str, res: ValueId, width: u32, val: &str, poison: &str) -> Sym {
        let vn = format!("{prefix}v{}", res.index());
        let pn = format!("{prefix}p{}", res.index());
        self.declare_bv(&vn, width);
        self.declare_bool(&pn);
        self.assert(&format!("(= {vn} {val})"));
        self.assert(&format!("(= {pn} {poison})"));
        Sym { val: vn, poison: pn, width }
    }

    /// Encode a whole function's straight-line body over the shared `inputs`.
    fn encode_function(
        &mut self,
        func: &Function,
        prefix: &str,
        inputs: &[Sym],
    ) -> Result<EncodedFn, EncErr> {
        let (params, ret_val) = single_block_params_ret(func)?;
        let entry = func.entry().expect("checked above");

        let mut map: HashMap<ValueId, Sym> = HashMap::new();
        for (i, &p) in params.iter().enumerate() {
            map.insert(p, inputs[i].clone());
        }

        let mut ub = Vec::new();
        for &inst_id in func.block(entry).insts() {
            let inst = func.inst(inst_id);
            if let Some(res) = inst.result() {
                let sym = self.encode_inst(func, prefix, inst, res, &mut map, &mut ub)?;
                map.insert(res, sym);
            }
        }

        let ret = self.resolve(func, &mut map, ret_val)?;
        Ok(EncodedFn { ret, ub })
    }

    /// Resolve an operand to its [`Sym`], materializing (and caching) constants.
    fn resolve(
        &mut self,
        func: &Function,
        map: &mut HashMap<ValueId, Sym>,
        v: ValueId,
    ) -> Result<Sym, EncErr> {
        if let Some(s) = map.get(&v) {
            return Ok(s.clone());
        }
        let sym = match func.value(v).def {
            ValueDef::Const(cid) => {
                let c = self.consts.get(cid).clone();
                self.encode_const(&c)?
            }
            // Params are pre-registered and instruction results are inserted as
            // they are encoded, so anything else here is out of scope.
            _ => return Err(unsupported("value is not a constant, parameter, or pure result")),
        };
        map.insert(v, sym.clone());
        Ok(sym)
    }

    /// Encode a constant into a [`Sym`].
    fn encode_const(&mut self, c: &Const) -> Result<Sym, EncErr> {
        match c {
            Const::Int { ty, value } => {
                let w = int_width(self.types, *ty)
                    .ok_or_else(|| unsupported("non-integer integer-constant type"))?;
                Ok(Sym { val: bv_lit(value, w), poison: "false".to_string(), width: w })
            }
            Const::Poison(ty) => {
                let w = int_width(self.types, *ty)
                    .ok_or_else(|| unsupported("poison constant of a non-integer type"))?;
                // Poison of an integer type: a fresh unconstrained value, marked
                // poison. (Its bit pattern is never observed unless frozen.)
                let fv = self.fresh_bv(w);
                Ok(Sym { val: fv, poison: "true".to_string(), width: w })
            }
            Const::Float { .. } => Err(unsupported("floating-point constant")),
            Const::Null(_) => Err(unsupported("pointer constant")),
            Const::Aggregate { .. } => Err(unsupported("aggregate constant")),
        }
    }

    /// Encode one value-producing instruction, returning its result [`Sym`].
    fn encode_inst(
        &mut self,
        func: &Function,
        prefix: &str,
        inst: &InstData,
        res: ValueId,
        map: &mut HashMap<ValueId, Sym>,
        ub: &mut Vec<String>,
    ) -> Result<Sym, EncErr> {
        let mut ops = Vec::with_capacity(inst.operands().len());
        for &o in inst.operands() {
            ops.push(self.resolve(func, map, o)?);
        }
        let rw = int_width(self.types, inst.ty)
            .ok_or_else(|| unsupported("non-integer result type"))?;

        let (val, poison) = match &inst.kind {
            InstKind::Bin(op) => enc_bin(*op, &inst.flags, &ops, rw, ub)?,
            InstKind::ICmp(pred) => enc_icmp(*pred, &ops)?,
            InstKind::Select => enc_select(&ops)?,
            InstKind::Cast(op) => enc_cast(*op, &ops, rw)?,
            InstKind::Freeze => {
                let a = &ops[0];
                let fresh = self.fresh_bv(rw);
                (format!("(ite {} {fresh} {})", a.poison, a.val), "false".to_string())
            }
            InstKind::Unary(_) => return Err(unsupported("float unary op (fneg)")),
            InstKind::FCmp(_) => return Err(unsupported("float comparison (fcmp)")),
            InstKind::PtrAdd { .. } => return Err(unsupported("pointer arithmetic (ptr_add)")),
            InstKind::Alloca { .. }
            | InstKind::Load { .. }
            | InstKind::Store { .. }
            | InstKind::Call => return Err(unsupported("memory / call op")),
            InstKind::Ret
            | InstKind::Br(_)
            | InstKind::CondBr { .. }
            | InstKind::Switch(_)
            | InstKind::Unreachable => return Err(unsupported("terminator in the body")),
        };
        Ok(self.bind(prefix, res, rw, &val, &poison))
    }
}

// ---------------------------------------------------------------------------
// Per-opcode encodings (value term, poison term).
// ---------------------------------------------------------------------------

/// Binary integer op. Pushes any UB condition onto `ub`; returns `(val, poison)`.
fn enc_bin(
    op: BinOp,
    flags: &Flags,
    ops: &[Sym],
    w: u32,
    ub: &mut Vec<String>,
) -> Result<(String, String), EncErr> {
    if op.is_float() {
        return Err(unsupported("floating-point arithmetic"));
    }
    let a = &ops[0];
    let b = &ops[1];
    let mut poison = vec![a.poison.clone(), b.poison.clone()];

    let val = match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul => {
            let (opc, ext) = match op {
                BinOp::Add => ("bvadd", 1),
                BinOp::Sub => ("bvsub", 1),
                BinOp::Mul => ("bvmul", w),
                _ => unreachable!(),
            };
            let res = format!("({opc} {} {})", a.val, b.val);
            if flags.nsw {
                poison.push(ext_overflow(opc, &a.val, &b.val, &res, ext, true));
            }
            if flags.nuw {
                poison.push(ext_overflow(opc, &a.val, &b.val, &res, ext, false));
            }
            res
        }
        BinOp::And => format!("(bvand {} {})", a.val, b.val),
        BinOp::Or => format!("(bvor {} {})", a.val, b.val),
        BinOp::Xor => format!("(bvxor {} {})", a.val, b.val),
        BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => {
            let signed = matches!(op, BinOp::SDiv | BinOp::SRem);
            let zero = bv_zero(w);
            let div_zero = format!("(= {} {zero})", b.val);
            let cond = if signed {
                let int_min = bv_int_min(w);
                let minus_one = bv_all_ones(w);
                format!(
                    "(and (not {ap}) (not {bp}) (or {div_zero} (and (= {av} {int_min}) (= {bv} {minus_one}))))",
                    ap = a.poison,
                    bp = b.poison,
                    av = a.val,
                    bv = b.val,
                )
            } else {
                format!("(and (not {ap}) (not {bp}) {div_zero})", ap = a.poison, bp = b.poison)
            };
            ub.push(cond);
            if flags.exact {
                // `exact` is meaningful only on udiv/sdiv: a nonzero remainder
                // is poison.
                let rem = match op {
                    BinOp::UDiv => Some(format!("(bvurem {} {})", a.val, b.val)),
                    BinOp::SDiv => Some(format!("(bvsrem {} {})", a.val, b.val)),
                    _ => None,
                };
                if let Some(rem) = rem {
                    poison.push(format!("(distinct {rem} {zero})"));
                }
            }
            match op {
                BinOp::UDiv => format!("(bvudiv {} {})", a.val, b.val),
                BinOp::SDiv => format!("(bvsdiv {} {})", a.val, b.val),
                BinOp::URem => format!("(bvurem {} {})", a.val, b.val),
                BinOp::SRem => format!("(bvsrem {} {})", a.val, b.val),
                _ => unreachable!(),
            }
        }
        BinOp::Shl | BinOp::LShr | BinOp::AShr => {
            let width_lit = bv_lit(&puremp::Int::from_u64(u64::from(w)), w);
            poison.push(format!("(bvuge {} {width_lit})", b.val));
            match op {
                BinOp::Shl => {
                    let res = format!("(bvshl {} {})", a.val, b.val);
                    if flags.nuw {
                        // No unsigned wrap: shifting back recovers the input.
                        poison.push(format!("(distinct (bvlshr {res} {}) {})", b.val, a.val));
                    }
                    if flags.nsw {
                        // No signed wrap: arithmetic round-trip recovers the input.
                        poison.push(format!("(distinct (bvashr {res} {}) {})", b.val, a.val));
                    }
                    res
                }
                BinOp::LShr => {
                    let res = format!("(bvlshr {} {})", a.val, b.val);
                    if flags.exact {
                        poison.push(format!("(distinct (bvshl {res} {}) {})", b.val, a.val));
                    }
                    res
                }
                BinOp::AShr => {
                    let res = format!("(bvashr {} {})", a.val, b.val);
                    if flags.exact {
                        poison.push(format!("(distinct (bvshl {res} {}) {})", b.val, a.val));
                    }
                    res
                }
                _ => unreachable!(),
            }
        }
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
            return Err(unsupported("floating-point arithmetic"));
        }
    };
    Ok((val, or_terms(&poison)))
}

/// Redo `opc` in a widened bit-vector (sign/zero-extended by `ext` bits) and
/// report whether reducing modulo `2^N` lost information — i.e. the flagged
/// wrap actually occurred. Matches the exact-then-reduce check in `ir::semantics`.
fn ext_overflow(opc: &str, a: &str, b: &str, wrapped: &str, ext: u32, signed: bool) -> String {
    let extend = if signed { "sign_extend" } else { "zero_extend" };
    let ea = format!("((_ {extend} {ext}) {a})");
    let eb = format!("((_ {extend} {ext}) {b})");
    let full = format!("({opc} {ea} {eb})");
    let wrapped_ext = format!("((_ {extend} {ext}) {wrapped})");
    format!("(distinct {full} {wrapped_ext})")
}

/// Integer comparison. Result is a 1-bit value; poison propagates from operands.
fn enc_icmp(pred: IntPred, ops: &[Sym]) -> Result<(String, String), EncErr> {
    let a = &ops[0];
    let b = &ops[1];
    let cmp = match pred {
        IntPred::Eq => format!("(= {} {})", a.val, b.val),
        IntPred::Ne => format!("(distinct {} {})", a.val, b.val),
        IntPred::Ugt => format!("(bvugt {} {})", a.val, b.val),
        IntPred::Uge => format!("(bvuge {} {})", a.val, b.val),
        IntPred::Ult => format!("(bvult {} {})", a.val, b.val),
        IntPred::Ule => format!("(bvule {} {})", a.val, b.val),
        IntPred::Sgt => format!("(bvsgt {} {})", a.val, b.val),
        IntPred::Sge => format!("(bvsge {} {})", a.val, b.val),
        IntPred::Slt => format!("(bvslt {} {})", a.val, b.val),
        IntPred::Sle => format!("(bvsle {} {})", a.val, b.val),
    };
    let val = format!("(ite {cmp} (_ bv1 1) (_ bv0 1))");
    Ok((val, or_terms(&[a.poison.clone(), b.poison.clone()])))
}

/// `select cond, t, f`. A poison condition poisons the result; a non-selected
/// poison arm does not (matches `ir::semantics::eval_select`).
fn enc_select(ops: &[Sym]) -> Result<(String, String), EncErr> {
    let cond = &ops[0];
    let t = &ops[1];
    let f = &ops[2];
    let cond_true = format!("(= {} (_ bv1 1))", cond.val);
    let val = format!("(ite {cond_true} {} {})", t.val, f.val);
    let poison =
        format!("(or {} (ite {cond_true} {} {}))", cond.poison, t.poison, f.poison);
    Ok((val, poison))
}

/// Integer casts `trunc`/`zext`/`sext`. Other casts are out of scope.
fn enc_cast(op: CastOp, ops: &[Sym], rw: u32) -> Result<(String, String), EncErr> {
    let a = &ops[0];
    let val = match op {
        CastOp::Trunc => {
            if rw > a.width {
                return Err(unsupported("trunc to a wider type"));
            }
            format!("((_ extract {} 0) {})", rw - 1, a.val)
        }
        CastOp::ZExt => {
            if rw < a.width {
                return Err(unsupported("zext to a narrower type"));
            }
            format!("((_ zero_extend {}) {})", rw - a.width, a.val)
        }
        CastOp::SExt => {
            if rw < a.width {
                return Err(unsupported("sext to a narrower type"));
            }
            format!("((_ sign_extend {}) {})", rw - a.width, a.val)
        }
        CastOp::FpTrunc
        | CastOp::FpExt
        | CastOp::FpToUi
        | CastOp::FpToSi
        | CastOp::UiToFp
        | CastOp::SiToFp => return Err(unsupported("floating-point cast")),
        CastOp::PtrToInt | CastOp::IntToPtr => return Err(unsupported("pointer cast")),
        CastOp::Bitcast => return Err(unsupported("bitcast")),
    };
    Ok((val, a.poison.clone()))
}

// ---------------------------------------------------------------------------
// SMT-LIB2 literal / sort helpers.
// ---------------------------------------------------------------------------

fn bv_sort(width: u32) -> String {
    format!("(_ BitVec {width})")
}

/// A bit-vector literal of the low `width` bits of `value` (its unsigned
/// representative in `[0, 2^width)`), as `(_ bvDEC width)`.
fn bv_lit(value: &puremp::Int, width: u32) -> String {
    let masked = value.mod_2k(width); // non-negative, `[0, 2^width)`
    format!("(_ bv{masked} {width})")
}

fn bv_zero(width: u32) -> String {
    format!("(_ bv0 {width})")
}

/// `INT_MIN` bit pattern for width `N`: `2^(N-1)` (top bit set).
fn bv_int_min(width: u32) -> String {
    let v = puremp::Int::ONE.mul_2k(width - 1);
    format!("(_ bv{v} {width})")
}

/// The all-ones pattern (`-1` / `2^N − 1`) for width `N`.
fn bv_all_ones(width: u32) -> String {
    let v = puremp::Int::ONE.mul_2k(width).sub(&puremp::Int::ONE);
    format!("(_ bv{v} {width})")
}

/// The integer width of `ty`, or `None` if it is not an integer type.
fn int_width(types: &TypeContext, ty: TypeId) -> Option<u32> {
    match types.get(ty) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Solver.
// ---------------------------------------------------------------------------

/// Run the SMT-LIB2 `script` (ending in `check-sat`) and map its verdict to a
/// [`RefinementResult`]. On `sat`, a second run appends a `get-value` for the
/// shared inputs to render the counterexample (`get-value` is illegal after an
/// `unsat`, so it cannot be issued unconditionally).
fn decide(script: &str, input_names: &[String]) -> RefinementResult {
    match z3rs::cmd_context::run_smt2(script) {
        Ok(lines) => match lines.first().map(String::as_str) {
            Some("unsat") => RefinementResult::Refines,
            Some("sat") => RefinementResult::Counterexample(extract_model(script, input_names)),
            Some("unknown") => {
                RefinementResult::Unknown("solver returned unknown (budget/incompleteness)".to_string())
            }
            other => RefinementResult::Unknown(format!("unexpected solver response: {other:?}")),
        },
        Err(e) => RefinementResult::Unknown(format!("solver error: {e}")),
    }
}

/// Re-run a satisfiable `script` with a `get-value` over the inputs to render a
/// human-readable counterexample; falls back to a placeholder on any hiccup.
fn extract_model(script: &str, input_names: &[String]) -> String {
    if input_names.is_empty() {
        return "(no inputs; the two functions differ on the empty input)".to_string();
    }
    let with_get = format!("{script}(get-value ({}))\n", input_names.join(" "));
    match z3rs::cmd_context::run_smt2(&with_get) {
        Ok(lines) => lines.get(1).cloned().unwrap_or_else(|| "(model unavailable)".to_string()),
        Err(_) => "(model unavailable)".to_string(),
    }
}

#[cfg(test)]
mod tests;
