//! Soft-float lowering: floating-point values as integers, floating-point
//! operations as calls to the Arm run-time ABI helpers.
//!
//! A Cortex-M3 has no floating-point unit, and the AAPCS base standard (the
//! "soft-float" ABI) passes a `float` like a 32-bit integer and a `double`
//! like a 64-bit one (an even-odd register pair or an 8-aligned stack slot).
//! So this IR-to-IR pass, run before wide-integer legalization, rewrites a
//! module so that no floating-point type remains:
//!
//! - every `f16`/`f32`/`f64` value, parameter, result, block parameter,
//!   load and store becomes an `i16`/`i32`/`i64` holding the IEEE-754 bits;
//!   function signatures change accordingly (declarations too, so calls stay
//!   consistent), which is exactly the soft-float calling convention;
//! - a float constant becomes the integer with its bit pattern, a `bitcast`
//!   between a float and an integer disappears, `fneg` flips the sign bit;
//! - arithmetic, comparisons and conversions become calls to the helpers of
//!   the *Run-time ABI for the Arm Architecture* (RTABI, §4.1.2):
//!
//! | operation | `f32` | `f64` |
//! |---|---|---|
//! | `fadd` `fsub` `fmul` `fdiv` | `__aeabi_fadd` `fsub` `fmul` `fdiv` | `__aeabi_dadd` `dsub` `dmul` `ddiv` |
//! | `frem` | `fmodf` (C library) | `fmod` |
//! | `fcmp` | `__aeabi_fcmpeq` `lt` `le` `ge` `gt` `un` | `__aeabi_dcmp…` |
//! | `fptosi` / `fptoui` to ≤ 32 / ≤ 64 bits | `__aeabi_f2iz` `f2uiz` / `f2lz` `f2ulz` | `__aeabi_d2iz` `d2uiz` / `d2lz` `d2ulz` |
//! | `sitofp` / `uitofp` from ≤ 32 / ≤ 64 bits | `__aeabi_i2f` `ui2f` / `l2f` `ul2f` | `__aeabi_i2d` `ui2d` / `l2d` `ul2d` |
//! | `fpext` / `fptrunc` | `__aeabi_f2d` | `__aeabi_d2f` |
//!
//! The comparison helpers return a nonzero `int` when their relation holds
//! and are false on unordered operands, so the ordered predicates call one
//! helper, their unordered complements negate one (`ugt` is `!ole`), and
//! `one`/`ueq` or two (`olt | ogt`, `uno | oeq`). `f16` values compute in
//! `f32` (`__aeabi_h2f`, `__aeabi_f2h`, `__aeabi_d2h`): the sum, difference,
//! product and quotient of two halves are exact in `f32` before the one
//! rounding back, and an integer converts through `f64` (exact below 2^53)
//! to round once. Narrow integer sources and destinations of a conversion go
//! through the 32-bit helpers with an extension or truncation (an
//! out-of-range conversion is poison in the IR, so the helper's saturation
//! refines it); integers wider than 64 bits are a [`SoftFloatError`].
//!
//! Every helper the module could need is declared up front (a declaration
//! nothing calls costs nothing in the object). A module without any
//! floating-point type is left untouched.

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

/// Why a module cannot be lowered to soft-float.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SoftFloatError {
    /// A conversion between a floating-point type and an integer wider than
    /// 64 bits (there is no RTABI helper for it).
    WideConversion(u32),
}

impl std::fmt::Display for SoftFloatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SoftFloatError::WideConversion(w) => {
                write!(f, "a conversion between floating point and i{w} has no soft-float helper")
            }
        }
    }
}

impl std::error::Error for SoftFloatError {}

/// What [`lower_soft_float`] did.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct SoftFloatReport {
    /// The functions whose bodies or signatures were rewritten.
    pub functions: Vec<FuncId>,
}

/// The integer type holding a value of type `ty`'s bits: `i16`/`i32`/`i64`
/// for the float types, `ty` itself otherwise.
fn int_bits_of(kind: FloatKind) -> u32 {
    match kind {
        FloatKind::F16 => 16,
        FloatKind::F32 => 32,
        FloatKind::F64 => 64,
    }
}

/// The helper table: name, parameter widths, result width (all integers).
const HELPERS: &[(&str, &[u32], u32)] = &[
    ("__aeabi_fadd", &[32, 32], 32),
    ("__aeabi_fsub", &[32, 32], 32),
    ("__aeabi_fmul", &[32, 32], 32),
    ("__aeabi_fdiv", &[32, 32], 32),
    ("fmodf", &[32, 32], 32),
    ("__aeabi_dadd", &[64, 64], 64),
    ("__aeabi_dsub", &[64, 64], 64),
    ("__aeabi_dmul", &[64, 64], 64),
    ("__aeabi_ddiv", &[64, 64], 64),
    ("fmod", &[64, 64], 64),
    ("__aeabi_fcmpeq", &[32, 32], 32),
    ("__aeabi_fcmplt", &[32, 32], 32),
    ("__aeabi_fcmple", &[32, 32], 32),
    ("__aeabi_fcmpge", &[32, 32], 32),
    ("__aeabi_fcmpgt", &[32, 32], 32),
    ("__aeabi_fcmpun", &[32, 32], 32),
    ("__aeabi_dcmpeq", &[64, 64], 32),
    ("__aeabi_dcmplt", &[64, 64], 32),
    ("__aeabi_dcmple", &[64, 64], 32),
    ("__aeabi_dcmpge", &[64, 64], 32),
    ("__aeabi_dcmpgt", &[64, 64], 32),
    ("__aeabi_dcmpun", &[64, 64], 32),
    ("__aeabi_f2d", &[32], 64),
    ("__aeabi_d2f", &[64], 32),
    ("__aeabi_f2iz", &[32], 32),
    ("__aeabi_f2uiz", &[32], 32),
    ("__aeabi_f2lz", &[32], 64),
    ("__aeabi_f2ulz", &[32], 64),
    ("__aeabi_d2iz", &[64], 32),
    ("__aeabi_d2uiz", &[64], 32),
    ("__aeabi_d2lz", &[64], 64),
    ("__aeabi_d2ulz", &[64], 64),
    ("__aeabi_i2f", &[32], 32),
    ("__aeabi_ui2f", &[32], 32),
    ("__aeabi_l2f", &[64], 32),
    ("__aeabi_ul2f", &[64], 32),
    ("__aeabi_i2d", &[32], 64),
    ("__aeabi_ui2d", &[32], 64),
    ("__aeabi_l2d", &[64], 64),
    ("__aeabi_ul2d", &[64], 64),
    // Half precision: the value travels zero-extended in a word.
    ("__aeabi_h2f", &[32], 32),
    ("__aeabi_f2h", &[32], 32),
    ("__aeabi_d2h", &[64], 32),
];

/// The name of the scratch declaration that carries a rewritten function's
/// new signature while it is rebuilt.
const SCRATCH: &str = "__lf_softfloat_scratch";

/// Rewrite `module` so that no floating-point type remains (see the [module
/// docs](self)). `syms` resolves and interns the helper names.
///
/// # Errors
///
/// [`SoftFloatError::WideConversion`] for a conversion between a float and an
/// integer wider than 64 bits.
pub fn lower_soft_float(module: &mut Module, syms: &mut StrInterner) -> Result<SoftFloatReport, SoftFloatError> {
    // 1. Anything to do? Which functions, and are all conversions supported?
    let mut work = Vec::new();
    for fi in 0..module.function_count() {
        let fid = FuncId::from_index(fi);
        let f = module.function(fid);
        let types = module.types();
        let is_float = |t: TypeId| types.get(t).is_float();
        let sig_float = match types.get(f.sig) {
            Type::Func(ft) => ft.params.iter().any(|&p| is_float(p)) || is_float(ft.ret),
            _ => false,
        };
        let body_float = (0..f.value_count()).any(|v| is_float(f.value_type(ValueId::from_index(v))))
            || f.blocks().any(|(_, b)| {
                b.insts().iter().any(|&i| {
                    matches!(f.inst(i).kind, InstKind::Load { ty, .. } | InstKind::Store { ty, .. } if is_float(ty))
                })
            });
        for (_, b) in f.blocks() {
            for &i in b.insts() {
                let inst = f.inst(i);
                if let InstKind::Cast(op) = inst.kind {
                    let src = types.bit_width(f.value_type(inst.operands()[0])).unwrap_or(0);
                    let dst = types.bit_width(inst.ty).unwrap_or(0);
                    let w = match op {
                        CastOp::FpToSi | CastOp::FpToUi => dst,
                        CastOp::SiToFp | CastOp::UiToFp => src,
                        _ => 0,
                    };
                    if w > 64 {
                        return Err(SoftFloatError::WideConversion(w));
                    }
                }
            }
        }
        if sig_float || body_float {
            work.push(fid);
        }
    }
    let mut report = SoftFloatReport::default();
    if work.is_empty() {
        return Ok(report);
    }

    // 2. Declare the helpers (or find a function of that name).
    let mut helpers: HashMap<&'static str, FuncId> = HashMap::new();
    for &(name, params, ret) in HELPERS {
        let existing = (0..module.function_count())
            .map(FuncId::from_index)
            .find(|&f| syms.resolve(module.function(f).name) == name);
        let fid = match existing {
            Some(f) => f,
            None => {
                let ps: Vec<TypeId> = params.iter().map(|&w| module.types_mut().int(w)).collect();
                let r = module.types_mut().int(ret);
                let sig = module.types_mut().func(ps, r, false);
                module.declare_function(syms.intern(name), sig)
            }
        };
        helpers.insert(name, fid);
    }

    // 3. Rebuild each function under its integer signature. The rebuilt body
    //    needs the new signature before its entry block exists, so it is
    //    built over a scratch declaration carrying that signature, then moved
    //    into place.
    let void = module.types_mut().void();
    let scratch_sig = module.types_mut().func(Vec::new(), void, false);
    let scratch = module.declare_function(syms.intern(SCRATCH), scratch_sig);
    for fid in work {
        let old_sig = module.function(fid).sig;
        let new_sig = int_sig(module, old_sig);
        let (name, attrs, decl_line) = {
            let f = module.function(fid);
            (f.name, f.attrs.clone(), f.decl_line)
        };
        let mut fresh = if module.function(fid).is_declaration() {
            Function::new(name, new_sig)
        } else {
            module.replace_function(scratch, Function::new(name, new_sig));
            let (fresh, ()) = module.map_function_reading(scratch, |_, funcs, b| {
                let old = &funcs[fid.index()];
                let mut sf = Sf { b, old, vmap: vec![None; old.value_count()], helpers: &helpers };
                sf.run();
            });
            fresh
        };
        fresh.attrs = attrs;
        fresh.decl_line = decl_line;
        module.replace_function(fid, fresh);
        report.functions.push(fid);
    }
    module.replace_function(scratch, Function::new(syms.intern(SCRATCH), scratch_sig));
    Ok(report)
}

/// `ty` with its float types replaced by the integers of their width.
fn int_ty(module: &mut Module, ty: TypeId) -> TypeId {
    match module.types().get(ty) {
        Type::Float(k) => {
            let w = int_bits_of(*k);
            module.types_mut().int(w)
        }
        _ => ty,
    }
}

/// A function signature with its float parameters and result as integers.
fn int_sig(module: &mut Module, sig: TypeId) -> TypeId {
    let Type::Func(ft) = module.types().get(sig).clone() else { return sig };
    let params: Vec<TypeId> = ft.params.iter().map(|&p| int_ty(module, p)).collect();
    let ret = int_ty(module, ft.ret);
    module.types_mut().func(params, ret, ft.variadic)
}

/// The per-function rebuild state.
struct Sf<'x, 'b> {
    b: &'x mut FunctionBuilder<'b>,
    old: &'x Function,
    /// Old value → its image in the rebuilt function.
    vmap: Vec<Option<ValueId>>,
    helpers: &'x HashMap<&'static str, FuncId>,
}

impl Sf<'_, '_> {
    fn ity(&mut self, ty: TypeId) -> TypeId {
        match self.b.types().get(ty) {
            Type::Float(k) => {
                let w = int_bits_of(*k);
                self.b.types_mut().int(w)
            }
            _ => ty,
        }
    }

    fn int(&mut self, w: u32) -> TypeId {
        self.b.types_mut().int(w)
    }

    fn float_kind(&self, v: ValueId) -> Option<FloatKind> {
        match self.b.types().get(self.old.value_type(v)) {
            Type::Float(k) => Some(*k),
            _ => None,
        }
    }

    /// The image of an old value: a mapped value, a converted constant, a
    /// global / function reference, or poison for a definition in unreachable
    /// code not yet rebuilt.
    fn map(&mut self, v: ValueId) -> ValueId {
        if let Some(x) = self.vmap[v.index()] {
            return x;
        }
        let ty = self.old.value_type(v);
        let x = match self.old.value(v).def.clone() {
            ValueDef::Const(c) => match self.b.consts().get(c).clone() {
                Const::Float { bits, .. } => {
                    let (raw, w) = match bits {
                        FloatBits::F16(b) => (u64::from(b), 16),
                        FloatBits::F32(b) => (u64::from(b), 32),
                        FloatBits::F64(b) => (b, 64),
                    };
                    let t = self.int(w);
                    self.b.const_int(t, Int::from_u64(raw))
                }
                Const::Poison(_) => {
                    let t = self.ity(ty);
                    self.b.poison(t)
                }
                _ => self.b.use_const(c),
            },
            ValueDef::Global(g) => self.b.global_ref(g),
            ValueDef::Func(f) => self.b.func_ref(f),
            ValueDef::Param(..) | ValueDef::Inst(..) => {
                let t = self.ity(ty);
                self.b.poison(t)
            }
        };
        self.vmap[v.index()] = Some(x);
        x
    }

    fn set(&mut self, old: Option<ValueId>, new: ValueId) {
        if let Some(r) = old {
            self.vmap[r.index()] = Some(new);
        }
    }

    fn call(&mut self, name: &str, args: &[ValueId], ret_bits: u32) -> ValueId {
        let f = self.helpers[name];
        let callee = self.b.func_ref(f);
        let rt = self.int(ret_bits);
        self.b.call(callee, args, rt).expect("a helper returns a value")
    }

    /// An `f16`'s bits (`i16`) as an `f32`'s (`i32`).
    fn h2f(&mut self, h: ValueId) -> ValueId {
        let t = self.int(32);
        let z = self.b.cast(CastOp::ZExt, h, t);
        self.call("__aeabi_h2f", &[z], 32)
    }

    /// Round an `f32` (`i32`) or `f64` (`i64`) to an `f16` (`i16`).
    fn round_half(&mut self, v: ValueId, from64: bool) -> ValueId {
        let r = if from64 { self.call("__aeabi_d2h", &[v], 32) } else { self.call("__aeabi_f2h", &[v], 32) };
        let t = self.int(16);
        self.b.cast(CastOp::Trunc, r, t)
    }

    // --- the rebuild --------------------------------------------------------

    fn run(&mut self) {
        let old = self.old;
        self.b.set_attrs(old.attrs.clone());
        let n = old.block_count();
        let entry = old.entry().expect("a definition has an entry block").index();
        let cfg = ControlFlowGraph::new(old);
        let doms = Dominators::new(old, &cfg);
        let mut new_block: Vec<BlockId> = Vec::with_capacity(n);
        for b in 0..n {
            let bb = BlockId::from_index(b);
            let nb = if b == entry {
                self.b.create_entry_block()
            } else {
                let tys: Vec<TypeId> =
                    old.block(bb).params().iter().map(|&p| old.value_type(p)).collect::<Vec<_>>();
                let tys: Vec<TypeId> = tys.into_iter().map(|t| self.ity(t)).collect();
                self.b.create_block(&tys)
            };
            new_block.push(nb);
            let params = self.b.block_params(nb).to_vec();
            for (&op, &np) in old.block(bb).params().iter().zip(&params) {
                self.vmap[op.index()] = Some(np);
            }
        }
        for b in dom_preorder(old, &doms) {
            let bb = BlockId::from_index(b);
            self.b.switch_to(new_block[b]);
            for &i in old.block(bb).insts() {
                self.b.set_line(old.inst_line(i).unwrap_or(0));
                self.inst(i);
            }
            if let Some(t) = old.block(bb).terminator() {
                self.b.set_line(old.inst_line(t).unwrap_or(0));
                self.terminator(t, &new_block);
            }
        }
    }

    fn terminator(&mut self, t: InstId, new_block: &[BlockId]) {
        let term = self.old.inst(t);
        let ops: Vec<ValueId> = term.operands().to_vec();
        match &term.kind {
            InstKind::Ret => {
                let v = ops.first().map(|&o| self.map(o));
                self.b.ret(v);
            }
            InstKind::Unreachable => self.b.unreachable(),
            InstKind::Br(target) => {
                let args: Vec<ValueId> = ops.iter().map(|&o| self.map(o)).collect();
                self.b.br(new_block[target.index()], &args);
            }
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let (ta, fa) = (*true_args as usize, *false_args as usize);
                let cond = self.map(ops[0]);
                let targs: Vec<ValueId> = ops[1..1 + ta].iter().map(|&o| self.map(o)).collect();
                let fargs: Vec<ValueId> = ops[1 + ta..1 + ta + fa].iter().map(|&o| self.map(o)).collect();
                self.b.cond_br(cond, new_block[if_true.index()], &targs, new_block[if_false.index()], &fargs);
            }
            InstKind::Switch(data) => {
                let cond = self.map(ops[0]);
                let da = data.default_args as usize;
                let dargs: Vec<ValueId> = ops[1..1 + da].iter().map(|&o| self.map(o)).collect();
                let mut cases = Vec::with_capacity(data.cases.len());
                let mut off = 1 + da;
                for c in &data.cases {
                    let ca = c.args as usize;
                    let cargs: Vec<ValueId> = ops[off..off + ca].iter().map(|&o| self.map(o)).collect();
                    cases.push((c.value.clone(), new_block[c.target.index()], cargs));
                    off += ca;
                }
                self.b.switch(cond, new_block[data.default.index()], &dargs, cases);
            }
            _ => {}
        }
    }

    fn inst(&mut self, id: InstId) {
        let inst = self.old.inst(id);
        let ops: Vec<ValueId> = inst.operands().to_vec();
        let res = inst.result();
        match &inst.kind {
            InstKind::Bin(op) if op.is_float() => {
                let k = self.float_kind(ops[0]).expect("a float operand");
                let (a, b) = (self.map(ops[0]), self.map(ops[1]));
                let r = self.fbin(*op, k, a, b);
                self.set(res, r);
            }
            InstKind::Unary(UnaryOp::FNeg) => {
                let k = self.float_kind(ops[0]).expect("a float operand");
                let w = int_bits_of(k);
                let a = self.map(ops[0]);
                let t = self.int(w);
                let sign = self.b.const_int(t, Int::ONE.mul_2k(w - 1));
                let r = self.b.bin(BinOp::Xor, a, sign, Flags::NONE);
                self.set(res, r);
            }
            InstKind::FCmp(pred) => {
                let k = self.float_kind(ops[0]).expect("a float operand");
                let (a, b) = (self.map(ops[0]), self.map(ops[1]));
                let r = self.fcmp(*pred, k, a, b);
                self.set(res, r);
            }
            InstKind::Cast(op) if self.is_float_cast(*op, inst.ty, ops[0]) => {
                let a = self.map(ops[0]);
                let r = self.conv(*op, ops[0], inst.ty, a);
                self.set(res, r);
            }
            InstKind::Load { ty, align, volatile, secret } => {
                let t = self.ity(*ty);
                let p = self.map(ops[0]);
                let kind = InstKind::Load { ty: t, align: *align, volatile: *volatile, secret: *secret };
                let r = self.b.append_inst(kind, vec![p], inst.flags, Some(t));
                if let Some(r) = r {
                    self.set(res, r);
                }
            }
            InstKind::Store { ty, align, volatile, secret } => {
                let t = self.ity(*ty);
                let (p, v) = (self.map(ops[0]), self.map(ops[1]));
                let kind = InstKind::Store { ty: t, align: *align, volatile: *volatile, secret: *secret };
                self.b.append_inst(kind, vec![p, v], inst.flags, None);
            }
            _ => {
                let new_ops: Vec<ValueId> = ops.iter().map(|&o| self.map(o)).collect();
                let rt = res.map(|_| inst.ty).map(|t| self.ity(t));
                let r = self.b.append_inst(inst.kind.clone(), new_ops, inst.flags, rt);
                if let Some(r) = r {
                    self.set(res, r);
                }
            }
        }
    }

    /// Whether a cast involves a floating-point type.
    fn is_float_cast(&self, op: CastOp, to: TypeId, src: ValueId) -> bool {
        match op {
            CastOp::FpExt | CastOp::FpTrunc | CastOp::FpToSi | CastOp::FpToUi | CastOp::SiToFp | CastOp::UiToFp => {
                true
            }
            CastOp::Bitcast => self.b.types().get(to).is_float() || self.float_kind(src).is_some(),
            _ => false,
        }
    }

    fn fbin(&mut self, op: BinOp, k: FloatKind, a: ValueId, b: ValueId) -> ValueId {
        let (s, d) = match op {
            BinOp::FAdd => ("__aeabi_fadd", "__aeabi_dadd"),
            BinOp::FSub => ("__aeabi_fsub", "__aeabi_dsub"),
            BinOp::FMul => ("__aeabi_fmul", "__aeabi_dmul"),
            BinOp::FDiv => ("__aeabi_fdiv", "__aeabi_ddiv"),
            _ => ("fmodf", "fmod"),
        };
        match k {
            FloatKind::F32 => self.call(s, &[a, b], 32),
            FloatKind::F64 => self.call(d, &[a, b], 64),
            FloatKind::F16 => {
                let (x, y) = (self.h2f(a), self.h2f(b));
                let r = self.call(s, &[x, y], 32);
                self.round_half(r, false)
            }
        }
    }

    fn fcmp(&mut self, pred: FloatPred, k: FloatKind, a: ValueId, b: ValueId) -> ValueId {
        let (a, b, dbl) = match k {
            FloatKind::F16 => (self.h2f(a), self.h2f(b), false),
            FloatKind::F32 => (a, b, false),
            FloatKind::F64 => (a, b, true),
        };
        let name = |rel: &str| format!("__aeabi_{}cmp{rel}", if dbl { 'd' } else { 'f' });
        // `rel(a, b)` as an i1, or its negation.
        let rel = |s: &mut Self, r: &str, holds: bool| -> ValueId {
            let n = name(r);
            let c = s.call(&n, &[a, b], 32);
            let t = s.int(32);
            let zero = s.b.const_int(t, Int::ZERO);
            s.b.icmp(if holds { IntPred::Ne } else { IntPred::Eq }, c, zero)
        };
        use FloatPred::*;
        match pred {
            False | True => self.b.const_bool(pred == True),
            Oeq => rel(self, "eq", true),
            Olt => rel(self, "lt", true),
            Ole => rel(self, "le", true),
            Ogt => rel(self, "gt", true),
            Oge => rel(self, "ge", true),
            Uno => rel(self, "un", true),
            Ord => rel(self, "un", false),
            Ugt => rel(self, "le", false),
            Uge => rel(self, "lt", false),
            Ult => rel(self, "ge", false),
            Ule => rel(self, "gt", false),
            Une => rel(self, "eq", false),
            One => {
                let x = rel(self, "lt", true);
                let y = rel(self, "gt", true);
                self.b.bin(BinOp::Or, x, y, Flags::NONE)
            }
            Ueq => {
                let x = rel(self, "un", true);
                let y = rel(self, "eq", true);
                self.b.bin(BinOp::Or, x, y, Flags::NONE)
            }
        }
    }

    /// Lower a conversion whose source (old value `src`, image `a`) or
    /// destination type `to` is a float.
    fn conv(&mut self, op: CastOp, src: ValueId, to: TypeId, a: ValueId) -> ValueId {
        let to_kind = match self.b.types().get(to) {
            Type::Float(k) => Some(*k),
            _ => None,
        };
        let from_kind = self.float_kind(src);
        match op {
            // The bits are already an integer of the right width.
            CastOp::Bitcast => a,
            CastOp::FpExt | CastOp::FpTrunc => {
                let (f, t) = (from_kind.expect("a float source"), to_kind.expect("a float result"));
                match (f, t) {
                    (FloatKind::F16, FloatKind::F32) => self.h2f(a),
                    (FloatKind::F16, FloatKind::F64) => {
                        let x = self.h2f(a);
                        self.call("__aeabi_f2d", &[x], 64)
                    }
                    (FloatKind::F32, FloatKind::F64) => self.call("__aeabi_f2d", &[a], 64),
                    (FloatKind::F64, FloatKind::F32) => self.call("__aeabi_d2f", &[a], 32),
                    (FloatKind::F32, FloatKind::F16) => self.round_half(a, false),
                    (FloatKind::F64, FloatKind::F16) => self.round_half(a, true),
                    _ => a,
                }
            }
            CastOp::FpToSi | CastOp::FpToUi => {
                let signed = op == CastOp::FpToSi;
                let w = self.b.types().bit_width(to).expect("an integer result");
                let (x, dbl) = match from_kind.expect("a float source") {
                    FloatKind::F16 => (self.h2f(a), false),
                    FloatKind::F32 => (a, false),
                    FloatKind::F64 => (a, true),
                };
                let long = w > 32;
                let name = format!(
                    "__aeabi_{}2{}{}z",
                    if dbl { 'd' } else { 'f' },
                    if signed { "" } else { "u" },
                    if long { 'l' } else { 'i' }
                );
                let r = self.call(&name, &[x], if long { 64 } else { 32 });
                let full = if long { 64 } else { 32 };
                if w < full { self.b.cast(CastOp::Trunc, r, to) } else { r }
            }
            CastOp::SiToFp | CastOp::UiToFp => {
                let signed = op == CastOp::SiToFp;
                let w = self.b.types().bit_width(self.old.value_type(src)).expect("an integer source");
                let long = w > 32;
                let full = if long { 64 } else { 32 };
                let x = if w < full {
                    let t = self.int(full);
                    self.b.cast(if signed { CastOp::SExt } else { CastOp::ZExt }, a, t)
                } else {
                    a
                };
                let kind = to_kind.expect("a float result");
                // An f16 result rounds once, from the exact f64.
                let dbl = kind != FloatKind::F32;
                let name = format!(
                    "__aeabi_{}{}2{}",
                    if signed { "" } else { "u" },
                    if long { 'l' } else { 'i' },
                    if dbl { 'd' } else { 'f' }
                );
                let r = self.call(&name, &[x], if dbl { 64 } else { 32 });
                if kind == FloatKind::F16 { self.round_half(r, true) } else { r }
            }
            _ => a,
        }
    }
}
