//! Soft float: an IR-to-IR pass that turns every `f32`/`f64` value into its
//! IEEE bit pattern (`i32`/`i64`) and every floating-point operation into a
//! call to a libgcc-named runtime helper, for a target with no FPU.
//!
//! | operation | helper (`sf` = `f32`, `df` = `f64`) |
//! |---|---|
//! | `fadd` `fsub` `fmul` `fdiv` | `__add{sf,df}3` `__sub…3` `__mul…3` `__div…3` |
//! | `frem` | `fmodf` / `fmod` |
//! | `fneg` | an `xor` of the sign bit (no call) |
//! | `fcmp` | `__eq…2` `__ne…2` `__lt…2` `__le…2` `__gt…2` `__ge…2` `__unord…2`, compared against 0 |
//! | `fpext` / `fptrunc` | `__extendsfdf2` / `__truncdfsf2` |
//! | `fptosi` / `fptoui` | `__fix{sf,df}{si,di}` / `__fixuns{sf,df}{si,di}` (narrower results truncate) |
//! | `sitofp` / `uitofp` | `__float{si,di}{sf,df}` / `__floatun{si,di}{sf,df}` (narrower sources extend) |
//!
//! Each helper takes and returns the bit patterns (`i32` for `f32`, `i64` for
//! `f64`); the comparison helpers return an `i16` (a C `int`). On AVR a float
//! travels in the same registers as an integer of its size, so these
//! signatures are ABI-identical to the float ones.
//!
//! **The ABI boundary keeps float types**, like the integer legalization's
//! (`docs/ir-design.md` §3b): function signatures are unchanged, and a float
//! entry parameter, call argument or result, or return value is converted with
//! a `bitcast` at the boundary. The backend treats a float value there as its
//! bit pattern (it lives in the same register group).
//!
//! `f16` is not supported (the pass reports an error).

use std::collections::HashMap;

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, CastOp, Flags, FloatPred, InstId, InstKind, IntPred, UnaryOp};
use crate::ir::types::{FloatKind, Type, TypeId};
use crate::ir::value::{Const, FloatBits, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Module};
use crate::support::StrInterner;
use crate::transform::dom_preorder;

use puremp::Int;

/// A runtime helper: its name and integer signature (bit widths).
#[derive(Clone, PartialEq, Eq, Debug)]
struct Helper {
    name: String,
    params: Vec<u32>,
    ret: u32,
}

fn fsuffix(bits: u32) -> &'static str {
    if bits == 32 { "sf" } else { "df" }
}

/// The helper computing a float binary op.
fn arith(op: BinOp, bits: u32) -> Helper {
    let name = match op {
        BinOp::FAdd => format!("__add{}3", fsuffix(bits)),
        BinOp::FSub => format!("__sub{}3", fsuffix(bits)),
        BinOp::FMul => format!("__mul{}3", fsuffix(bits)),
        BinOp::FDiv => format!("__div{}3", fsuffix(bits)),
        _ => (if bits == 32 { "fmodf" } else { "fmod" }).to_owned(),
    };
    Helper { name, params: vec![bits, bits], ret: bits }
}

/// How an `fcmp` predicate is computed: the helper calls (each result
/// compared against 0 with its predicate) and whether their results are
/// or-ed (else and-ed). `None` for the constant predicates.
fn compare(pred: FloatPred, bits: u32) -> Option<(Vec<(Helper, IntPred)>, bool)> {
    let h = |n: &str| Helper { name: format!("__{n}{}2", fsuffix(bits)), params: vec![bits, bits], ret: 16 };
    use FloatPred::*;
    Some(match pred {
        False | True => return None,
        Oeq => (vec![(h("eq"), IntPred::Eq)], false),
        One => (vec![(h("unord"), IntPred::Eq), (h("ne"), IntPred::Ne)], false),
        Ogt => (vec![(h("gt"), IntPred::Sgt)], false),
        Oge => (vec![(h("ge"), IntPred::Sge)], false),
        Olt => (vec![(h("lt"), IntPred::Slt)], false),
        Ole => (vec![(h("le"), IntPred::Sle)], false),
        Ord => (vec![(h("unord"), IntPred::Eq)], false),
        Uno => (vec![(h("unord"), IntPred::Ne)], false),
        Ueq => (vec![(h("unord"), IntPred::Ne), (h("eq"), IntPred::Eq)], true),
        Une => (vec![(h("ne"), IntPred::Ne)], false),
        Ugt => (vec![(h("le"), IntPred::Sgt)], false),
        Uge => (vec![(h("lt"), IntPred::Sge)], false),
        Ult => (vec![(h("ge"), IntPred::Slt)], false),
        Ule => (vec![(h("gt"), IntPred::Sle)], false),
    })
}

/// The helper of a conversion involving a float, with the integer width it
/// works at (`None` for a `bitcast`). `from`/`to` are `(is_float, bits)`.
fn convert(op: CastOp, from: (bool, u32), to: (bool, u32)) -> Result<Option<(Helper, u32)>, String> {
    let int_w = |b: u32| -> Result<u32, String> {
        match b {
            0..=32 => Ok(32),
            33..=64 => Ok(64),
            _ => Err(format!("a conversion between a float and i{b} is not supported")),
        }
    };
    let isz = |w: u32| if w == 32 { "si" } else { "di" };
    Ok(Some(match op {
        CastOp::FpExt => (Helper { name: "__extendsfdf2".into(), params: vec![32], ret: 64 }, 0),
        CastOp::FpTrunc => (Helper { name: "__truncdfsf2".into(), params: vec![64], ret: 32 }, 0),
        CastOp::FpToSi | CastOp::FpToUi => {
            let w = int_w(to.1)?;
            let u = if op == CastOp::FpToUi { "uns" } else { "" };
            (Helper { name: format!("__fix{u}{}{}", fsuffix(from.1), isz(w)), params: vec![from.1], ret: w }, w)
        }
        CastOp::SiToFp | CastOp::UiToFp => {
            let w = int_w(from.1)?;
            let u = if op == CastOp::UiToFp { "un" } else { "" };
            (Helper { name: format!("__float{u}{}{}", isz(w), fsuffix(to.1)), params: vec![w], ret: to.1 }, w)
        }
        _ => return Ok(None),
    }))
}

/// The float width of a type (`Some(32)`/`Some(64)`), `Err` for `f16`.
fn float_bits(m: &Module, ty: TypeId) -> Result<Option<u32>, String> {
    match m.types().get(ty) {
        Type::Float(FloatKind::F16) => Err("f16 is not supported on AVR".to_owned()),
        Type::Float(k) => Ok(Some(k.bit_width())),
        _ => Ok(None),
    }
}

/// The helpers one instruction of `f` needs.
fn needed(m: &Module, f: &Function, id: InstId) -> Result<Vec<Helper>, String> {
    let i = f.inst(id);
    let ops = i.operands();
    let fb = |v: ValueId| float_bits(m, f.value_type(v));
    Ok(match &i.kind {
        InstKind::Bin(op) if op.is_float() => vec![arith(*op, fb(ops[0])?.unwrap_or(32))],
        InstKind::FCmp(pred) => match compare(*pred, fb(ops[0])?.unwrap_or(32)) {
            Some((calls, _)) => calls.into_iter().map(|(h, _)| h).collect(),
            None => Vec::new(),
        },
        InstKind::Cast(op) => {
            let from = f.value_type(ops[0]);
            let from = (fb(ops[0])?.is_some(), m.types().bit_width(from).unwrap_or(16));
            let to = (float_bits(m, i.ty)?.is_some(), m.types().bit_width(i.ty).unwrap_or(16));
            match convert(*op, from, to)? {
                Some((h, _)) => vec![h],
                None => Vec::new(),
            }
        }
        _ => Vec::new(),
    })
}

/// Whether `f` touches a float anywhere (a value of float type).
fn has_float(m: &Module, f: &Function) -> Result<bool, String> {
    let mut any = false;
    for v in 0..f.value_count() {
        if float_bits(m, f.value_type(ValueId::from_index(v)))?.is_some() {
            any = true;
        }
    }
    Ok(any)
}

/// Rewrite every function of `m` that uses floating point (see the [module
/// docs](self)), declaring the helpers it calls.
pub(crate) fn lower_floats(m: &mut Module, s: &mut StrInterner) -> Result<(), String> {
    let mut work = Vec::new();
    let mut helpers: Vec<Helper> = Vec::new();
    for fi in 0..m.function_count() {
        let fid = FuncId::from_index(fi);
        let f = m.function(fid);
        if f.is_declaration() || !has_float(m, f)? {
            continue;
        }
        for (_, b) in f.blocks() {
            for &i in b.insts() {
                for h in needed(m, f, i)? {
                    if !helpers.contains(&h) {
                        helpers.push(h);
                    }
                }
            }
        }
        work.push(fid);
    }
    let mut ids: HashMap<String, FuncId> = HashMap::new();
    for h in &helpers {
        let existing = (0..m.function_count())
            .map(FuncId::from_index)
            .find(|&f| s.resolve(m.function(f).name) == h.name);
        let params: Vec<TypeId> = h.params.iter().map(|&w| m.types_mut().int(w)).collect();
        let ret = m.types_mut().int(h.ret);
        let sig = m.types_mut().func(params, ret, false);
        let fid = match existing {
            Some(f) if m.function(f).sig == sig => f,
            Some(_) => return Err(format!("`{}` is defined with a signature other than the soft-float helper's", h.name)),
            None => m.declare_function(s.intern(&h.name), sig),
        };
        ids.insert(h.name.clone(), fid);
    }
    for fid in work {
        let (fresh, r) = m.map_function(fid, |old, b| {
            let mut sf = Sf { b, old, vmap: vec![None; old.value_count()], ids: &ids };
            sf.run()
        });
        r?;
        m.replace_function(fid, fresh);
    }
    Ok(())
}

/// The per-function rebuild state.
struct Sf<'x, 'b> {
    b: &'x mut FunctionBuilder<'b>,
    old: &'x Function,
    vmap: Vec<Option<ValueId>>,
    ids: &'x HashMap<String, FuncId>,
}

impl Sf<'_, '_> {
    /// A type with floats replaced by integers of their width.
    fn ity(&mut self, ty: TypeId) -> TypeId {
        match self.b.types().get(ty) {
            Type::Float(k) => {
                let w = k.bit_width();
                self.b.types_mut().int(w)
            }
            _ => ty,
        }
    }

    fn fbits(&self, ty: TypeId) -> Option<u32> {
        match self.b.types().get(ty) {
            Type::Float(k) => Some(k.bit_width()),
            _ => None,
        }
    }

    /// The image of an old value.
    fn val(&mut self, v: ValueId) -> ValueId {
        if let Some(x) = self.vmap[v.index()] {
            return x;
        }
        let x = match &self.old.value(v).def {
            ValueDef::Const(c) => match self.b.consts().get(*c).clone() {
                Const::Float { bits, .. } => {
                    let (raw, w) = match bits {
                        FloatBits::F16(b) => (u64::from(b), 16),
                        FloatBits::F32(b) => (u64::from(b), 32),
                        FloatBits::F64(b) => (b, 64),
                    };
                    let t = self.b.types_mut().int(w);
                    self.b.const_int(t, Int::from_u64(raw))
                }
                Const::Poison(ty) | Const::Null(ty) if self.fbits(ty).is_some() => {
                    let t = self.ity(ty);
                    self.b.poison(t)
                }
                _ => self.b.use_const(*c),
            },
            ValueDef::Global(g) => self.b.global_ref(*g),
            ValueDef::Func(f) => self.b.func_ref(*f),
            ValueDef::Param(..) | ValueDef::Inst(..) => {
                let t = self.old.value_type(v);
                let t = self.ity(t);
                self.b.poison(t)
            }
        };
        self.vmap[v.index()] = Some(x);
        x
    }

    fn set(&mut self, old: Option<ValueId>, new: ValueId) {
        if let Some(o) = old {
            self.vmap[o.index()] = Some(new);
        }
    }

    fn call(&mut self, name: &str, args: &[ValueId], ret_bits: u32) -> ValueId {
        let f = self.ids[name];
        let callee = self.b.func_ref(f);
        let rt = self.b.types_mut().int(ret_bits);
        self.b.call(callee, args, rt).expect("a helper returns a value")
    }

    /// `v` (an integer of any width) extended or truncated to `w` bits.
    fn resize(&mut self, v: ValueId, signed: bool, w: u32) -> ValueId {
        let from = self.b.types().bit_width(self.b.value_type(v)).unwrap_or(w);
        let t = self.b.types_mut().int(w);
        match from.cmp(&w) {
            std::cmp::Ordering::Equal => v,
            std::cmp::Ordering::Less => self.b.cast(if signed { CastOp::SExt } else { CastOp::ZExt }, v, t),
            std::cmp::Ordering::Greater => self.b.cast(CastOp::Trunc, v, t),
        }
    }

    /// The old value `v` as its float type again (a boundary `bitcast`).
    fn as_float(&mut self, v: ValueId) -> ValueId {
        let ty = self.old.value_type(v);
        let x = self.val(v);
        if self.fbits(ty).is_some() { self.b.cast(CastOp::Bitcast, x, ty) } else { x }
    }

    fn run(&mut self) -> Result<(), String> {
        let old = self.old;
        let n = old.block_count();
        let entry = old.entry().expect("a definition has an entry block").index();
        let cfg = ControlFlowGraph::new(old);
        let doms = Dominators::new(old, &cfg);
        let mut new_block = Vec::with_capacity(n);
        for b in 0..n {
            if b == entry {
                new_block.push(self.b.create_entry_block());
                continue;
            }
            let tys: Vec<TypeId> = old.block(BlockId::from_index(b)).params().iter().map(|&p| old.value_type(p)).collect();
            let tys: Vec<TypeId> = tys.into_iter().map(|t| self.ity(t)).collect();
            new_block.push(self.b.create_block(&tys));
        }
        for (b, &nb) in new_block.iter().enumerate() {
            if b == entry {
                continue;
            }
            let np = self.b.block_params(nb).to_vec();
            for (&p, &q) in old.block(BlockId::from_index(b)).params().iter().zip(&np) {
                self.vmap[p.index()] = Some(q);
            }
        }
        for b in dom_preorder(old, &doms) {
            self.b.switch_to(new_block[b]);
            let bb = BlockId::from_index(b);
            if b == entry {
                let np = self.b.block_params(new_block[b]).to_vec();
                for (&p, &q) in old.block(bb).params().iter().zip(&np) {
                    let ty = old.value_type(p);
                    let x = if self.fbits(ty).is_some() {
                        let t = self.ity(ty);
                        self.b.cast(CastOp::Bitcast, q, t)
                    } else {
                        q
                    };
                    self.vmap[p.index()] = Some(x);
                }
            }
            for &i in old.block(bb).insts() {
                self.inst(i)?;
            }
            if let Some(t) = old.block(bb).terminator() {
                self.term(t, &new_block);
            }
        }
        Ok(())
    }

    fn term(&mut self, t: InstId, new_block: &[BlockId]) {
        let term = self.old.inst(t);
        match &term.kind {
            InstKind::Ret => {
                let v = term.operands().first().map(|&o| self.as_float(o));
                self.b.ret(v);
            }
            kind => {
                let ops: Vec<ValueId> = term.operands().iter().map(|&o| self.val(o)).collect();
                // Blocks are created in the old order, so their ids carry over.
                debug_assert!(new_block.iter().enumerate().all(|(k, b)| b.index() == k));
                self.b.append_inst(kind.clone(), ops, term.flags, None);
            }
        }
    }

    fn inst(&mut self, id: InstId) -> Result<(), String> {
        let old = self.old;
        let i = old.inst(id);
        let ops = i.operands();
        let res = i.result();
        let fb = |s: &Self, v: ValueId| s.fbits(old.value_type(v));
        match &i.kind {
            InstKind::Bin(op) if op.is_float() => {
                let bits = fb(self, ops[0]).unwrap_or(32);
                let h = arith(*op, bits);
                let (a, c) = (self.val(ops[0]), self.val(ops[1]));
                let r = self.call(&h.name, &[a, c], bits);
                self.set(res, r);
            }
            InstKind::Unary(UnaryOp::FNeg) => {
                let bits = fb(self, ops[0]).unwrap_or(32);
                let t = self.b.types_mut().int(bits);
                let sign = self.b.const_int(t, Int::from_u64(1u64 << (bits - 1)));
                let a = self.val(ops[0]);
                let r = self.b.bin(BinOp::Xor, a, sign, Flags::NONE);
                self.set(res, r);
            }
            InstKind::FCmp(pred) => {
                let bits = fb(self, ops[0]).unwrap_or(32);
                let r = match compare(*pred, bits) {
                    None => self.b.const_bool(*pred == FloatPred::True),
                    Some((calls, any)) => {
                        let (a, c) = (self.val(ops[0]), self.val(ops[1]));
                        let t16 = self.b.types_mut().int(16);
                        let zero = self.b.const_int(t16, Int::ZERO);
                        let mut acc: Option<ValueId> = None;
                        for (h, p) in calls {
                            let r = self.call(&h.name, &[a, c], 16);
                            let bit = self.b.icmp(p, r, zero);
                            acc = Some(match acc {
                                None => bit,
                                Some(x) => self.b.bin(if any { BinOp::Or } else { BinOp::And }, x, bit, Flags::NONE),
                            });
                        }
                        acc.expect("at least one helper")
                    }
                };
                self.set(res, r);
            }
            InstKind::Cast(op) if fb(self, ops[0]).is_some() || self.fbits(i.ty).is_some() => {
                let from_ty = old.value_type(ops[0]);
                let from = (self.fbits(from_ty).is_some(), self.b.types().bit_width(from_ty).unwrap_or(16));
                let to = (self.fbits(i.ty).is_some(), self.b.types().bit_width(i.ty).unwrap_or(16));
                let a = self.val(ops[0]);
                let r = match convert(*op, from, to)? {
                    None => {
                        // A bitcast between a float and an integer (or pointer)
                        // of its width: the bit pattern itself.
                        let t = self.ity(i.ty);
                        if self.b.value_type(a) == t { a } else { self.b.cast(*op, a, t) }
                    }
                    Some((h, w)) => match op {
                        CastOp::FpToSi | CastOp::FpToUi => {
                            let r = self.call(&h.name, &[a], w);
                            self.resize(r, false, to.1)
                        }
                        CastOp::SiToFp | CastOp::UiToFp => {
                            let x = self.resize(a, *op == CastOp::SiToFp, w);
                            self.call(&h.name, &[x], to.1)
                        }
                        _ => self.call(&h.name, &[a], h.ret),
                    },
                };
                self.set(res, r);
            }
            InstKind::Call => {
                let callee = self.val(ops[0]);
                let args: Vec<ValueId> = ops[1..].iter().map(|&a| self.as_float(a)).collect();
                let r = self.b.call(callee, &args, i.ty);
                if let (Some(r), Some(o)) = (r, res) {
                    let x = if self.fbits(i.ty).is_some() {
                        let t = self.ity(i.ty);
                        self.b.cast(CastOp::Bitcast, r, t)
                    } else {
                        r
                    };
                    self.set(Some(o), x);
                }
            }
            kind => {
                let kind = match kind {
                    InstKind::Load { ty, align, volatile, secret } => InstKind::Load { ty: self.ity(*ty), align: *align, volatile: *volatile, secret: *secret },
                    InstKind::Store { ty, align, volatile, secret } => InstKind::Store { ty: self.ity(*ty), align: *align, volatile: *volatile, secret: *secret },
                    k => k.clone(),
                };
                let new_ops: Vec<ValueId> = ops.iter().map(|&o| self.val(o)).collect();
                let rt = res.map(|_| self.ity(i.ty));
                let r = self.b.append_inst(kind, new_ops, i.flags, rt);
                if let Some(r) = r {
                    self.set(res, r);
                }
            }
        }
        Ok(())
    }
}
