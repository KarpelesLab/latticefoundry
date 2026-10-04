//! Target-independent **vector legalization** (`docs/ir-design.md` §6c): an
//! IR→IR rewrite, run by each backend right before instruction selection, that
//! leaves only the vector types and operations the target lowers natively.
//! Correctness therefore never depends on the ISA having an instruction.
//!
//! A target describes itself with a [`VectorLegality`]: which vector *types*
//! can live whole in one register, and which vector *instructions* on those
//! types it selects directly. Everything else is **scalarized**:
//!
//! - **Illegal types are split into lanes.** A value of an illegal vector type
//!   `<N x T>` becomes `N` scalar SSA values of type `T`: block parameters
//!   gain `N` parameters (and every edge passes `N` arguments), lane-wise ops
//!   become `N` scalar ops, loads and stores become `N` element accesses at
//!   `i × size(T)` (alignment reduced to what each lane offset guarantees),
//!   the lane moves just pick lanes, reductions become an in-order chain.
//! - **Illegal types in signatures** follow one fixed, target-independent
//!   convention: an illegal vector *parameter* is passed as its `N` lanes, in
//!   place; an illegal vector *result* is returned through a hidden leading
//!   `ptr` parameter to caller-allocated storage (the function then returns
//!   `void`). Both sides of every call are rewritten by the same rule, so code
//!   compiled by LatticeFoundry agrees with itself; it is *not* the C ABI for
//!   such types (a target that wants C interop for a vector type makes it
//!   legal).
//! - **Unsupported ops on legal types are scalarized in place**: each lane is
//!   `extractelement`ed, computed by the scalar op, and `insertelement`ed into
//!   the result (starting from `poison`), so the target only needs
//!   extract/insert for its legal types.
//! - A `bitcast` touching a split vector is rebuilt from lane bits with
//!   integer shifts, truncations, extensions and ors (never through memory, so
//!   it is exact for `i1` lanes as well), preserving per-lane poison.
//!
//! - **Min/max and saturating ops** (`smin`, `uadd_sat`, …) with no direct
//!   form on the target — every scalar one, and vector ones the target does not
//!   select — are first expanded into compares, selects and wrapping
//!   arithmetic of the same (scalar or vector) type, which are then legalized
//!   like any other op (so a vector `smin` becomes a vector `icmp` + `select`
//!   where those are legal, rather than scalar code).
//!
//! Every rewrite is a refinement of the reference semantics lane by lane
//! (per-lane poison stays in its lane; a UB lane keeps the whole op UB); the
//! tests check this by executing programs before and after legalization with
//! the reference executor.

use std::borrow::Cow;

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, CastOp, Flags, InstData, InstKind, IntPred, ReduceOp};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ConstPool, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Module};

use puremp::Int;

/// What a target lowers natively. See the module docs.
pub trait VectorLegality {
    /// Whether a value of vector type `ty` can live whole in a register (and
    /// cross calls, returns and block edges as one value).
    fn legal_type(&self, types: &TypeContext, ty: TypeId) -> bool;

    /// Whether `inst` of `func` — every vector type it touches already legal —
    /// is selected directly. `consts` resolves constant operands (e.g. to
    /// recognize a uniform shift amount). `call`, `ret`, branches, `freeze` and
    /// the like on legal types are always accepted without asking.
    fn legal_inst(&self, types: &TypeContext, consts: &ConstPool, func: &Function, inst: &InstData) -> bool;
}

/// The legality of a target without vector registers: every vector type is
/// illegal, so all vector code is split into scalars.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarOnly;

impl VectorLegality for ScalarOnly {
    fn legal_type(&self, _types: &TypeContext, _ty: TypeId) -> bool {
        false
    }

    fn legal_inst(&self, _: &TypeContext, _: &ConstPool, _: &Function, _: &InstData) -> bool {
        false
    }
}

/// `module` legalized for `legality`: borrowed unchanged when it contains no
/// vector code at all (the common case costs one scan), else a legalized copy.
pub fn legalized<'m>(module: &'m Module, legality: &dyn VectorLegality) -> Cow<'m, Module> {
    if !uses_vectors(module) && !uses_minmax_sat(module) {
        return Cow::Borrowed(module);
    }
    let mut m = module.clone();
    legalize_vectors(&mut m, legality);
    Cow::Owned(m)
}

/// Whether any function of `module` mentions a vector type or op.
pub fn uses_vectors(module: &Module) -> bool {
    let types = module.types();
    module.functions().any(|f| {
        sig_has_vector(types, f.sig)
            || (0..f.value_count()).any(|i| types.is_vector(f.value_type(ValueId::from_index(i))))
            || (0..f.inst_count()).any(|i| {
                let inst = f.inst(crate::ir::InstId::from_index(i));
                inst.kind.is_vector_op()
                    || matches!(&inst.kind, InstKind::Load { ty, .. } | InstKind::Store { ty, .. } if types.is_vector(*ty))
            })
    })
}

/// Whether any function of `module` uses a min/max or saturating op.
fn uses_minmax_sat(module: &Module) -> bool {
    module.functions().any(|f| {
        (0..f.inst_count()).any(|i| {
            matches!(f.inst(crate::ir::InstId::from_index(i)).kind, InstKind::Bin(op) if op.is_minmax_sat())
        })
    })
}

fn sig_has_vector(types: &TypeContext, sig: TypeId) -> bool {
    match types.get(sig) {
        Type::Func(ft) => ft.params.iter().chain(std::iter::once(&ft.ret)).any(|&t| types.is_vector(t)),
        _ => false,
    }
}

/// Legalize every function of `module` in place for `legality`.
pub fn legalize_vectors(module: &mut Module, legality: &dyn VectorLegality) {
    expand_minmax_sat(module, legality);
    // New signatures first (they intern types), then each body.
    let n = module.function_count();
    let mut new_sigs = Vec::with_capacity(n);
    for i in 0..n {
        let sig = module.function(FuncId::from_index(i)).sig;
        new_sigs.push(legal_sig(module.types_mut(), legality, sig));
    }
    for (i, &(sig, ret_out)) in new_sigs.iter().enumerate() {
        let id = FuncId::from_index(i);
        let old = module.function(id);
        if old.is_declaration() {
            if sig != old.sig {
                let mut decl = Function::new(old.name, sig);
                decl.decl_line = old.decl_line;
                module.replace_function(id, decl);
            }
            continue;
        }
        if sig == old.sig && !body_needs(module.types(), module.consts(), legality, old) {
            continue;
        }
        let decl_line = old.decl_line;
        let (mut fresh, ()) = module.map_function(id, |old, b| {
            b.set_signature(sig);
            Rewriter::new(old, b, legality, ret_out).run();
        });
        fresh.decl_line = decl_line;
        module.replace_function(id, fresh);
    }
}

/// The legalized signature of `sig`: illegal vector parameters become their
/// lanes; an illegal vector result becomes a hidden leading `ptr` parameter and
/// a `void` result (`true` in the second slot).
fn legal_sig(types: &mut TypeContext, legality: &dyn VectorLegality, sig: TypeId) -> (TypeId, bool) {
    let Type::Func(ft) = types.get(sig).clone() else {
        return (sig, false);
    };
    let illegal = |types: &TypeContext, t: TypeId| types.is_vector(t) && !legality.legal_type(types, t);
    let ret_out = illegal(types, ft.ret);
    let mut params = Vec::with_capacity(ft.params.len() + 1);
    if ret_out {
        params.push(types.ptr());
    }
    for &p in &ft.params {
        match types.vector_parts(p) {
            Some((elem, n)) if illegal(types, p) => params.extend(std::iter::repeat_n(elem, n as usize)),
            _ => params.push(p),
        }
    }
    if !ret_out && params == ft.params {
        return (sig, false);
    }
    let ret = if ret_out { types.void() } else { ft.ret };
    (types.func(params, ret, ft.variadic), ret_out)
}

/// Whether a body has anything to legalize: an illegal vector type anywhere,
/// or an instruction on legal vectors the target does not select.
fn body_needs(types: &TypeContext, consts: &ConstPool, legality: &dyn VectorLegality, f: &Function) -> bool {
    let illegal = |t: TypeId| types.is_vector(t) && !legality.legal_type(types, t);
    if (0..f.value_count()).any(|i| illegal(f.value_type(ValueId::from_index(i)))) {
        return true;
    }
    (0..f.inst_count()).any(|i| {
        let inst = f.inst(crate::ir::InstId::from_index(i));
        let access = match &inst.kind {
            InstKind::Load { ty, .. } | InstKind::Store { ty, .. } => Some(*ty),
            _ => None,
        };
        if access.is_some_and(illegal) {
            return true;
        }
        touches_vector(types, f, inst) && !always_legal(&inst.kind) && !legality.legal_inst(types, consts, f, inst)
    })
}

/// Whether an instruction has a vector result, operand or accessed type.
fn touches_vector(types: &TypeContext, f: &Function, inst: &InstData) -> bool {
    let access = matches!(&inst.kind, InstKind::Load { ty, .. } | InstKind::Store { ty, .. } if types.is_vector(*ty));
    access
        || inst.result().is_some_and(|r| types.is_vector(f.value_type(r)))
        || inst.operands().iter().any(|&o| types.is_vector(f.value_type(o)))
}

/// Ops that only move a whole (legal) vector value: never scalarized.
fn always_legal(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Call
            | InstKind::InlineAsm(_)
            | InstKind::AsmOutput(_)
            | InstKind::Freeze
            | InstKind::Ret
            | InstKind::Br(_)
            | InstKind::CondBr { .. }
            | InstKind::Switch(_)
            | InstKind::Unreachable
    )
}

// ---------------------------------------------------------------------------
// Min/max and saturating ops
// ---------------------------------------------------------------------------

/// Whether `inst` is a min/max/saturating op the target does not select: any
/// scalar one, or a vector one on a type or with an op it lacks.
fn needs_expansion(types: &TypeContext, consts: &ConstPool, legality: &dyn VectorLegality, f: &Function, inst: &InstData) -> bool {
    let InstKind::Bin(op) = inst.kind else {
        return false;
    };
    if !op.is_minmax_sat() {
        return false;
    }
    !(types.is_vector(inst.ty) && legality.legal_type(types, inst.ty) && legality.legal_inst(types, consts, f, inst))
}

/// Expand every min/max/saturating op the target lacks (see the module docs).
fn expand_minmax_sat(module: &mut Module, legality: &dyn VectorLegality) {
    for i in 0..module.function_count() {
        let id = FuncId::from_index(i);
        let f = module.function(id);
        let any = (0..f.inst_count()).any(|k| {
            needs_expansion(module.types(), module.consts(), legality, f, f.inst(crate::ir::InstId::from_index(k)))
        });
        if !any {
            continue;
        }
        let decl_line = f.decl_line;
        let (mut fresh, ()) = module.map_function(id, |old, b| expand_function(old, b, legality));
        fresh.decl_line = decl_line;
        module.replace_function(id, fresh);
    }
}

/// Rebuild `old` into `b`, expanding the min/max/saturating ops that need it.
fn expand_function(old: &Function, b: &mut FunctionBuilder<'_>, legality: &dyn VectorLegality) {
    use crate::transform::{rebuild_terminator, remap_value};
    let entry = old.entry().expect("a body has an entry block");
    let mut blocks = Vec::with_capacity(old.block_count());
    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for bi in 0..old.block_count() {
        let bid = BlockId::from_index(bi);
        let nb = if bid == entry {
            b.create_entry_block()
        } else {
            let tys: Vec<TypeId> = old.block(bid).params().iter().map(|&p| old.value_type(p)).collect();
            b.create_block(&tys)
        };
        for (&p, &np) in old.block(bid).params().iter().zip(b.block_params(nb)) {
            vmap[p.index()] = Some(np);
        }
        blocks.push(nb);
    }
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);
    for bi in crate::transform::dom_preorder(old, &doms) {
        let bid = BlockId::from_index(bi);
        b.switch_to(blocks[bi]);
        for &iid in old.block(bid).insts() {
            b.set_line(old.inst_line(iid).unwrap_or(0));
            let inst = old.inst(iid);
            let ops: Vec<ValueId> = inst.operands().iter().map(|&o| remap_value(&mut vmap, old, b, o)).collect();
            let expand = needs_expansion(b.types(), b.consts(), legality, old, inst);
            let r = match inst.kind {
                InstKind::Bin(op) if expand => Some(expand_op(b, op, inst.ty, ops[0], ops[1])),
                _ => b.append_inst(inst.kind.clone(), ops, inst.flags, inst.result().map(|_| inst.ty)),
            };
            if let (Some(old_r), Some(nr)) = (inst.result(), r) {
                vmap[old_r.index()] = Some(nr);
            }
        }
        if let Some(t) = old.block(bid).terminator() {
            b.set_line(old.inst_line(t).unwrap_or(0));
        }
        rebuild_terminator(&mut vmap, old, b, &blocks, bid, |_, _, _| {});
    }
}

/// A constant of type `ty` (a scalar integer, or every lane of a vector).
fn int_const(b: &mut FunctionBuilder<'_>, ty: TypeId, value: Int) -> ValueId {
    match b.types().vector_parts(ty) {
        Some((elem, n)) => {
            let lane = b.intern_const(Const::Int { ty: elem, value });
            b.const_vector(ty, vec![lane; n as usize])
        }
        None => b.const_int(ty, value),
    }
}

/// Expand one min/max/saturating op on `a`, `b` of type `ty` (scalar or
/// vector) into compares, selects and wrapping arithmetic of that type.
fn expand_op(b: &mut FunctionBuilder<'_>, op: BinOp, ty: TypeId, x: ValueId, y: ValueId) -> ValueId {
    let w = b.types().bit_width(b.types().scalar_of(ty)).expect("an integer lane type");
    let pick = |b: &mut FunctionBuilder<'_>, pred: IntPred| {
        let keep_x = b.icmp(pred, x, y);
        b.select(keep_x, x, y)
    };
    match op {
        BinOp::SMin => pick(b, IntPred::Sle),
        BinOp::SMax => pick(b, IntPred::Sge),
        BinOp::UMin => pick(b, IntPred::Ule),
        BinOp::UMax => pick(b, IntPred::Uge),
        BinOp::UAddSat => {
            // The wrapped sum is below `x` exactly when the add carried out.
            let s = b.bin(BinOp::Add, x, y, Flags::NONE);
            let carry = b.icmp(IntPred::Ult, s, x);
            let max = int_const(b, ty, Int::ONE.mul_2k(w).sub(&Int::ONE));
            b.select(carry, max, s)
        }
        BinOp::USubSat => {
            let d = b.bin(BinOp::Sub, x, y, Flags::NONE);
            let ok = b.icmp(IntPred::Ugt, x, y);
            let zero = int_const(b, ty, Int::ZERO);
            b.select(ok, d, zero)
        }
        BinOp::SAddSat | BinOp::SSubSat => {
            let add = op == BinOp::SAddSat;
            let s = b.bin(if add { BinOp::Add } else { BinOp::Sub }, x, y, Flags::NONE);
            // Signed overflow: for `x + y`, both operands' signs differ from
            // the result's; for `x - y`, the operands' signs differ and the
            // result's differs from `x`'s.
            let (p, q) = if add {
                (b.bin(BinOp::Xor, s, x, Flags::NONE), b.bin(BinOp::Xor, s, y, Flags::NONE))
            } else {
                (b.bin(BinOp::Xor, x, y, Flags::NONE), b.bin(BinOp::Xor, x, s, Flags::NONE))
            };
            let both = b.bin(BinOp::And, p, q, Flags::NONE);
            let zero = int_const(b, ty, Int::ZERO);
            let ov = b.icmp(IntPred::Slt, both, zero);
            // Saturate toward `x`'s sign: `(x >>s (w-1)) ^ INT_MAX` is INT_MAX
            // for a non-negative `x` and INT_MIN for a negative one.
            let amt = int_const(b, ty, Int::from_u64(u64::from(w - 1)));
            let sign = b.bin(BinOp::AShr, x, amt, Flags::NONE);
            let max = int_const(b, ty, Int::ONE.mul_2k(w - 1).sub(&Int::ONE));
            let sat = b.bin(BinOp::Xor, sign, max, Flags::NONE);
            b.select(ov, sat, s)
        }
        _ => unreachable!("not a min/max/saturating op: {op:?}"),
    }
}

/// A rewritten value: one new value, or the lanes of a split vector.
#[derive(Clone, Debug)]
enum NewVal {
    One(ValueId),
    Lanes(Vec<ValueId>),
}

/// The per-function rewriter.
struct Rewriter<'a, 'b> {
    old: &'a Function,
    b: &'a mut FunctionBuilder<'b>,
    legality: &'a dyn VectorLegality,
    ret_out: bool,
    map: Vec<Option<NewVal>>,
    blocks: Vec<BlockId>,
    out_ptr: Option<ValueId>,
}

impl<'a, 'b> Rewriter<'a, 'b> {
    fn new(old: &'a Function, b: &'a mut FunctionBuilder<'b>, legality: &'a dyn VectorLegality, ret_out: bool) -> Self {
        Rewriter { old, b, legality, ret_out, map: vec![None; old.value_count()], blocks: Vec::new(), out_ptr: None }
    }

    fn types(&self) -> &TypeContext {
        self.b.types()
    }

    /// Whether `ty` is a vector the target cannot hold whole.
    fn illegal(&self, ty: TypeId) -> bool {
        self.types().is_vector(ty) && !self.legality.legal_type(self.types(), ty)
    }

    /// The parameter types a block parameter of type `ty` expands to.
    fn expand_ty(&self, ty: TypeId) -> Vec<TypeId> {
        match self.types().vector_parts(ty) {
            Some((elem, n)) if self.illegal(ty) => vec![elem; n as usize],
            _ => vec![ty],
        }
    }

    fn run(mut self) {
        let old = self.old;
        let entry = old.entry().expect("a body has an entry block");
        // Create every block with its expanded parameter list (the entry's
        // come from the new signature), and map the old parameters.
        let n = old.block_count();
        for bi in 0..n {
            let bid = BlockId::from_index(bi);
            let nb = if bid == entry {
                self.b.create_entry_block()
            } else {
                let tys: Vec<TypeId> =
                    old.block(bid).params().iter().flat_map(|&p| self.expand_ty(old.value_type(p))).collect();
                self.b.create_block(&tys)
            };
            self.blocks.push(nb);
        }
        for bi in 0..n {
            let bid = BlockId::from_index(bi);
            let mut params = self.b.block_params(self.blocks[bi]).to_vec().into_iter();
            if bid == entry && self.ret_out {
                self.out_ptr = params.next();
            }
            for &p in old.block(bid).params() {
                let k = self.expand_ty(old.value_type(p)).len();
                let got: Vec<ValueId> = params.by_ref().take(k).collect();
                self.map[p.index()] = Some(if self.illegal(old.value_type(p)) {
                    NewVal::Lanes(got)
                } else {
                    NewVal::One(got[0])
                });
            }
        }
        // Emit in dominator preorder so definitions precede uses.
        let cfg = ControlFlowGraph::new(old);
        let doms = Dominators::new(old, &cfg);
        for bi in crate::transform::dom_preorder(old, &doms) {
            let bid = BlockId::from_index(bi);
            self.b.switch_to(self.blocks[bi]);
            for &iid in old.block(bid).insts() {
                self.b.set_line(old.inst_line(iid).unwrap_or(0));
                self.inst(old.inst(iid));
            }
            if let Some(t) = old.block(bid).terminator() {
                self.b.set_line(old.inst_line(t).unwrap_or(0));
                self.terminator(old.inst(t));
            }
        }
    }

    // --- value mapping --------------------------------------------------------

    /// The new form of an old value, materializing constants and references
    /// (a split vector constant becomes its lane constants). A value never
    /// defined on the way here is in unreachable code: poison of its shape.
    fn get(&mut self, v: ValueId) -> NewVal {
        if let Some(nv) = &self.map[v.index()] {
            return nv.clone();
        }
        let ty = self.old.value_type(v);
        let nv = match self.old.value(v).def.clone() {
            ValueDef::Const(c) if self.illegal(ty) => NewVal::Lanes(self.const_lanes(c, ty)),
            ValueDef::Const(c) => NewVal::One(self.b.use_const(c)),
            ValueDef::Global(g) => NewVal::One(self.b.global_ref(g)),
            ValueDef::Func(f) => NewVal::One(self.b.func_ref(f)),
            ValueDef::Param(..) | ValueDef::Inst(..) => match self.types().vector_parts(ty) {
                Some((elem, n)) if self.illegal(ty) => {
                    NewVal::Lanes((0..n).map(|_| self.b.poison(elem)).collect())
                }
                _ => NewVal::One(self.b.poison(ty)),
            },
        };
        self.map[v.index()] = Some(nv.clone());
        nv
    }

    /// The lane constants of a vector constant of type `ty`.
    fn const_lanes(&mut self, c: crate::ir::ConstId, ty: TypeId) -> Vec<ValueId> {
        let (elem, n) = self.types().vector_parts(ty).expect("a vector constant");
        match self.b.consts().get(c).clone() {
            // Integer lanes are reduced to their bit pattern (constants are not
            // stored normalized, and scalar backends read e.g. a shift count
            // off the stored value).
            Const::Aggregate { elems, .. } => elems
                .into_iter()
                .map(|e| match self.b.consts().get(e).clone() {
                    Const::Int { ty, value } => {
                        let w = self.types().bit_width(ty).expect("an integer lane");
                        self.b.const_int(ty, value.mod_2k(w))
                    }
                    _ => self.b.use_const(e),
                })
                .collect(),
            _ => (0..n).map(|_| self.b.poison(elem)).collect(),
        }
    }

    /// An old non-split value as one new value.
    fn one(&mut self, v: ValueId) -> ValueId {
        match self.get(v) {
            NewVal::One(x) => x,
            NewVal::Lanes(_) => unreachable!("a split vector used where a single value is needed"),
        }
    }

    /// The lanes of an old vector value: a split vector's lanes, a vector
    /// constant's lane constants, or `extractelement`s of a legal vector.
    fn lanes(&mut self, v: ValueId) -> Vec<ValueId> {
        let ty = self.old.value_type(v);
        if let ValueDef::Const(c) = self.old.value(v).def {
            return self.const_lanes(c, ty);
        }
        match self.get(v) {
            NewVal::Lanes(l) => l,
            NewVal::One(x) => {
                let (_, n) = self.types().vector_parts(ty).expect("lanes of a vector");
                (0..n).map(|i| self.b.extract_element(x, i)).collect()
            }
        }
    }

    /// Flatten an old value into edge/call arguments.
    fn flat(&mut self, v: ValueId) -> Vec<ValueId> {
        match self.get(v) {
            NewVal::One(x) => vec![x],
            NewVal::Lanes(l) => l,
        }
    }

    /// Assemble lanes into a value of vector type `ty`: kept split if `ty` is
    /// illegal, else built by `insertelement`s into `poison` (a constant vector
    /// if every lane is a constant).
    fn build(&mut self, ty: TypeId, lanes: Vec<ValueId>) -> NewVal {
        if self.illegal(ty) {
            return NewVal::Lanes(lanes);
        }
        let consts: Option<Vec<_>> = lanes.iter().map(|&l| self.b.const_of(l)).collect();
        if let Some(cs) = consts {
            return NewVal::One(self.b.const_vector(ty, cs));
        }
        let mut acc = self.b.poison(ty);
        for (i, l) in lanes.into_iter().enumerate() {
            if self.b.const_of(l).is_some_and(|c| matches!(self.b.consts().get(c), Const::Poison(_))) {
                continue;
            }
            acc = self.b.insert_element(acc, l, i as u32);
        }
        NewVal::One(acc)
    }

    fn set(&mut self, inst: &InstData, nv: NewVal) {
        if let Some(r) = inst.result() {
            self.map[r.index()] = Some(nv);
        }
    }

    /// Copy an instruction unchanged (its vector types, if any, are legal).
    fn copy(&mut self, inst: &InstData) {
        let ops: Vec<ValueId> = inst.operands().iter().map(|&o| self.one(o)).collect();
        let rty = inst.result().map(|_| inst.ty);
        if let Some(r) = self.b.append_inst(inst.kind.clone(), ops, inst.flags, rty) {
            self.set(inst, NewVal::One(r));
        }
    }

    // --- instructions ---------------------------------------------------------

    fn inst(&mut self, inst: &InstData) {
        let old = self.old;
        let types = self.types();
        if !touches_vector(types, old, inst) {
            return self.copy(inst);
        }
        let any_illegal = inst.result().is_some_and(|r| self.illegal(old.value_type(r)))
            || inst.operands().iter().any(|&o| self.illegal(old.value_type(o)))
            || matches!(&inst.kind, InstKind::Load { ty, .. } | InstKind::Store { ty, .. } if self.illegal(*ty));
        if matches!(inst.kind, InstKind::Call) {
            return if any_illegal { self.call(inst) } else { self.copy(inst) };
        }
        if !any_illegal
            && (always_legal(&inst.kind) || self.legality.legal_inst(types, self.b.consts(), old, inst))
        {
            return self.copy(inst);
        }
        self.scalarize(inst);
    }

    /// Rewrite one vector instruction lane by lane.
    fn scalarize(&mut self, inst: &InstData) {
        let old = self.old;
        let ops = inst.operands();
        let rty = inst.ty;
        match &inst.kind {
            InstKind::Cast(CastOp::Bitcast) => {
                let nv = self.bitcast(ops[0], old.value_type(ops[0]), rty);
                self.set(inst, nv);
            }
            InstKind::Bin(_) | InstKind::Unary(_) | InstKind::ICmp(_) | InstKind::FCmp(_) | InstKind::Cast(_) | InstKind::Freeze => {
                let (elem, n) = self.types().vector_parts(rty).expect("a lane-wise vector op");
                let op_lanes: Vec<Vec<ValueId>> = ops.iter().map(|&o| self.lanes(o)).collect();
                let lanes: Vec<ValueId> = (0..n as usize)
                    .map(|i| {
                        let lane_ops = op_lanes.iter().map(|l| l[i]).collect();
                        self.b.append_inst(inst.kind.clone(), lane_ops, inst.flags, Some(elem)).expect("a result")
                    })
                    .collect();
                let nv = self.build(rty, lanes);
                self.set(inst, nv);
            }
            InstKind::Select => {
                let (_, n) = self.types().vector_parts(rty).expect("a vector select");
                let cond_vec = self.types().is_vector(old.value_type(ops[0]));
                let conds: Vec<ValueId> =
                    if cond_vec { self.lanes(ops[0]) } else { vec![self.one(ops[0]); n as usize] };
                let (t, f) = (self.lanes(ops[1]), self.lanes(ops[2]));
                let lanes = (0..n as usize).map(|i| self.b.select(conds[i], t[i], f[i])).collect();
                let nv = self.build(rty, lanes);
                self.set(inst, nv);
            }
            InstKind::ExtractElement { lane } => {
                let l = self.lanes(ops[0])[*lane as usize];
                self.set(inst, NewVal::One(l));
            }
            InstKind::InsertElement { lane } => {
                let mut l = self.lanes(ops[0]);
                l[*lane as usize] = self.one(ops[1]);
                let nv = self.build(rty, l);
                self.set(inst, nv);
            }
            InstKind::ShuffleVector(mask) => {
                let mut cat = self.lanes(ops[0]);
                cat.extend(self.lanes(ops[1]));
                let lanes = mask.iter().map(|&m| cat[m as usize]).collect();
                let nv = self.build(rty, lanes);
                self.set(inst, nv);
            }
            InstKind::Splat => {
                let (_, n) = self.types().vector_parts(rty).expect("a splat result");
                let x = self.one(ops[0]);
                let nv = self.build(rty, vec![x; n as usize]);
                self.set(inst, nv);
            }
            InstKind::Reduce(op) => {
                let lanes = self.lanes(ops[0]);
                let r = self.reduce(*op, inst.flags, lanes);
                self.set(inst, NewVal::One(r));
            }
            InstKind::Load { ty, align, volatile, secret } => {
                debug_assert!(!volatile, "the verifier rejects volatile vector accesses");
                let (elem, n) = self.types().vector_parts(*ty).expect("a vector load");
                let size = self.types().size_of(elem);
                let base = self.one(ops[0]);
                // A secret access stays secret lane by lane (§6d).
                let lanes = (0..u64::from(n))
                    .map(|i| {
                        let p = self.lane_addr(base, i * size);
                        let kind =
                            InstKind::Load { ty: elem, align: lane_align(*align, i * size), volatile: false, secret: *secret };
                        self.b.append_inst(kind, vec![p], Flags::NONE, Some(elem)).expect("a load result")
                    })
                    .collect();
                let nv = self.build(*ty, lanes);
                self.set(inst, nv);
            }
            InstKind::Store { ty, align, secret, .. } => {
                let (elem, _) = self.types().vector_parts(*ty).expect("a vector store");
                let size = self.types().size_of(elem);
                let base = self.one(ops[0]);
                let lanes = self.lanes(ops[1]);
                for (i, l) in lanes.into_iter().enumerate() {
                    let off = i as u64 * size;
                    let p = self.lane_addr(base, off);
                    let (ty, v) = self.memory_lane(elem, l);
                    let kind = InstKind::Store { ty, align: lane_align(*align, off), volatile: false, secret: *secret };
                    self.b.append_inst(kind, vec![p, v], Flags::NONE, None);
                }
            }
            other => unreachable!("no vector form of {other:?} reaches the legalizer"),
        }
    }

    /// A lane as it is stored: an `i1` lane is one byte holding exactly 0 or 1,
    /// so it is widened with `zext` to `i8` (a register may hold an `i1` with
    /// garbage above bit 0, e.g. an all-ones mask lane); other lanes as is.
    fn memory_lane(&mut self, elem: TypeId, l: ValueId) -> (TypeId, ValueId) {
        if self.types().bit_width(elem) == Some(1) && !self.types().get(elem).is_float() {
            let i8t = self.b.types_mut().int(8);
            (i8t, self.b.cast(CastOp::ZExt, l, i8t))
        } else {
            (elem, l)
        }
    }

    /// `base + off` (no `ptr_add` for lane 0).
    fn lane_addr(&mut self, base: ValueId, off: u64) -> ValueId {
        if off == 0 {
            return base;
        }
        let i64t = self.b.types_mut().int(64);
        let k = self.b.const_i64(i64t, off as i64);
        self.b.ptr_add(base, k, true)
    }

    /// Combine lanes in order, lane 0 first (the reference order).
    fn reduce(&mut self, op: ReduceOp, flags: Flags, lanes: Vec<ValueId>) -> ValueId {
        let mut it = lanes.into_iter();
        let mut acc = it.next().expect("a vector has at least one lane");
        for l in it {
            acc = match op {
                ReduceOp::Add => self.b.bin(BinOp::Add, acc, l, Flags::NONE),
                ReduceOp::Mul => self.b.bin(BinOp::Mul, acc, l, Flags::NONE),
                ReduceOp::And => self.b.bin(BinOp::And, acc, l, Flags::NONE),
                ReduceOp::Or => self.b.bin(BinOp::Or, acc, l, Flags::NONE),
                ReduceOp::Xor => self.b.bin(BinOp::Xor, acc, l, Flags::NONE),
                ReduceOp::FAdd => self.b.bin(BinOp::FAdd, acc, l, flags),
                ReduceOp::FMul => self.b.bin(BinOp::FMul, acc, l, flags),
                ReduceOp::SMin | ReduceOp::SMax | ReduceOp::UMin | ReduceOp::UMax => {
                    // `acc` wins ties, as in the reference fold.
                    let pred = match op {
                        ReduceOp::SMin => IntPred::Sle,
                        ReduceOp::SMax => IntPred::Sge,
                        ReduceOp::UMin => IntPred::Ule,
                        _ => IntPred::Uge,
                    };
                    let keep = self.b.icmp(pred, acc, l);
                    self.b.select(keep, acc, l)
                }
            };
        }
        acc
    }

    /// Rebuild a `bitcast` of old value `v` (type `from`) to `to` from lane
    /// bits: each destination lane is assembled from, or cut out of, the
    /// source lanes it overlaps (lane 0 in the low bits), with integer ops of
    /// at most the wider lane's width.
    fn bitcast(&mut self, v: ValueId, from: TypeId, to: TypeId) -> NewVal {
        let types = self.types();
        let (s_elem, s_n) = types.vector_parts(from).unwrap_or((from, 1));
        let (d_elem, d_n) = types.vector_parts(to).unwrap_or((to, 1));
        let ws = types.bit_width(s_elem).expect("scalar source lanes");
        let wd = types.bit_width(d_elem).expect("scalar destination lanes");
        let s_float = types.get(s_elem).is_float();
        let d_float = types.get(d_elem).is_float();
        let src: Vec<ValueId> = if types.is_vector(from) { self.lanes(v) } else { vec![self.one(v)] };
        let si = self.b.types_mut().int(ws);
        let di = self.b.types_mut().int(wd);
        // Source lanes as integers.
        let src: Vec<ValueId> =
            src.into_iter().map(|l| if s_float { self.b.cast(CastOp::Bitcast, l, si) } else { l }).collect();
        debug_assert_eq!(u64::from(ws) * u64::from(s_n), u64::from(wd) * u64::from(d_n));
        let mut out = Vec::with_capacity(d_n as usize);
        for j in 0..d_n {
            let lo = u64::from(j) * u64::from(wd);
            let lane = if ws == wd {
                src[j as usize]
            } else if ws > wd {
                // Cut destination lane `j` out of one wider source lane.
                let k = (lo / u64::from(ws)) as usize;
                let shift = lo % u64::from(ws);
                let mut x = src[k];
                if shift > 0 {
                    let amt = self.b.const_i64(si, shift as i64);
                    x = self.b.bin(BinOp::LShr, x, amt, Flags::NONE);
                }
                self.b.cast(CastOp::Trunc, x, di)
            } else {
                // Assemble destination lane `j` from `wd / ws` narrower lanes.
                let per = wd / ws;
                let first = (lo / u64::from(ws)) as usize;
                let mut acc: Option<ValueId> = None;
                for m in 0..per as usize {
                    let mut x = self.b.cast(CastOp::ZExt, src[first + m], di);
                    if m > 0 {
                        let amt = self.b.const_i64(di, i64::from(ws) * m as i64);
                        x = self.b.bin(BinOp::Shl, x, amt, Flags::NONE);
                    }
                    acc = Some(match acc {
                        None => x,
                        Some(a) => self.b.bin(BinOp::Or, a, x, Flags::NONE),
                    });
                }
                acc.expect("at least one source lane")
            };
            out.push(if d_float { self.b.cast(CastOp::Bitcast, lane, d_elem) } else { lane });
        }
        if self.types().is_vector(to) {
            self.build(to, out)
        } else {
            NewVal::One(out[0])
        }
    }

    /// A call touching an illegal vector: split arguments, and an illegal
    /// result comes back through a caller-allocated slot passed first.
    fn call(&mut self, inst: &InstData) {
        let ops = inst.operands();
        let callee = self.one(ops[0]);
        let rty = inst.result().map(|_| inst.ty);
        let out = rty.filter(|&t| self.illegal(t));
        let mut args = Vec::with_capacity(ops.len());
        let slot = out.map(|t| self.b.alloca(t));
        args.extend(slot);
        for &a in &ops[1..] {
            args.extend(self.flat(a));
        }
        let void = self.b.types_mut().void();
        let ret = if out.is_some() { void } else { rty.unwrap_or(void) };
        let r = self.b.call(callee, &args, ret);
        match (out, slot) {
            (Some(t), Some(p)) => {
                let (elem, n) = self.types().vector_parts(t).expect("a vector result");
                let size = self.types().size_of(elem);
                let align = self.types().align_of(t) as u32;
                let lanes = (0..u64::from(n))
                    .map(|i| {
                        let q = self.lane_addr(p, i * size);
                        self.b.load(elem, q, lane_align(align, i * size))
                    })
                    .collect();
                self.set(inst, NewVal::Lanes(lanes));
            }
            _ => {
                if let Some(r) = r {
                    self.set(inst, NewVal::One(r));
                }
            }
        }
    }

    // --- terminators ----------------------------------------------------------

    fn terminator(&mut self, term: &InstData) {
        let ops = term.operands();
        match &term.kind {
            InstKind::Ret => {
                if let (Some(out), Some(&v)) = (self.out_ptr, ops.first()) {
                    let ty = self.old.value_type(v);
                    let (elem, _) = self.types().vector_parts(ty).expect("a vector result");
                    let size = self.types().size_of(elem);
                    let align = self.types().align_of(ty) as u32;
                    for (i, l) in self.lanes(v).into_iter().enumerate() {
                        let off = i as u64 * size;
                        let p = self.lane_addr(out, off);
                        let (ty, v) = self.memory_lane(elem, l);
                        self.b.store(ty, p, v, lane_align(align, off));
                    }
                    self.b.ret(None);
                } else {
                    let v = ops.first().map(|&v| self.one(v));
                    self.b.ret(v);
                }
            }
            InstKind::Unreachable => self.b.unreachable(),
            InstKind::Br(t) => {
                let args: Vec<ValueId> = ops.iter().flat_map(|&o| self.flat(o)).collect();
                self.b.br(self.blocks[t.index()], &args);
            }
            InstKind::CondBr { if_true, if_false, true_args, .. } => {
                let ta = *true_args as usize;
                let cond = self.one(ops[0]);
                let targs: Vec<ValueId> = ops[1..1 + ta].iter().flat_map(|&o| self.flat(o)).collect();
                let fargs: Vec<ValueId> = ops[1 + ta..].iter().flat_map(|&o| self.flat(o)).collect();
                let (t, f) = (self.blocks[if_true.index()], self.blocks[if_false.index()]);
                self.b.cond_br(cond, t, &targs, f, &fargs);
            }
            InstKind::Switch(data) => {
                let cond = self.one(ops[0]);
                let mut at = 1 + data.default_args as usize;
                let dargs: Vec<ValueId> = ops[1..at].iter().flat_map(|&o| self.flat(o)).collect();
                let mut cases = Vec::with_capacity(data.cases.len());
                for c in &data.cases {
                    let n = c.args as usize;
                    let args: Vec<ValueId> = ops[at..at + n].iter().flat_map(|&o| self.flat(o)).collect();
                    at += n;
                    cases.push((c.value.clone(), self.blocks[c.target.index()], args));
                }
                self.b.switch(cond, self.blocks[data.default.index()], &dargs, cases);
            }
            other => unreachable!("not a terminator: {other:?}"),
        }
    }
}

/// The alignment an access at byte offset `off` from an `align`-aligned base
/// is guaranteed: the largest power of two dividing both.
fn lane_align(align: u32, off: u64) -> u32 {
    if off == 0 {
        return align;
    }
    let tz = off.trailing_zeros().min(31);
    align.min(1u32 << tz)
}

#[cfg(test)]
mod tests;
