//! Soft float: floating-point values as integers, floating-point operations
//! as calls to run-time helpers, for targets without an FPU (Cortex-M3 Thumb,
//! AVR).
//!
//! On such a target the ABI passes a float like an integer of its size, so
//! this IR-to-IR pass, run after vector legalization and before wide-integer
//! legalization, rewrites a module so that no floating-point type remains:
//!
//! - every `f16`/`f32`/`f64` value, parameter, result, block parameter,
//!   load and store becomes an `i16`/`i32`/`i64` holding the IEEE-754 bits;
//!   function signatures change accordingly (declarations too, so calls stay
//!   consistent), which is exactly the soft-float calling convention;
//! - a float constant becomes the integer with its bit pattern, a `bitcast`
//!   between a float and an integer disappears, `fneg` flips the sign bit;
//! - arithmetic, comparisons and conversions become calls to helpers whose
//!   names a [`SoftFloatAbi`] chooses:
//!
//! | operation | [`SoftFloatAbi::Aeabi`] (Arm RTABI §4.1.2) | [`SoftFloatAbi::Libgcc`] |
//! |---|---|---|
//! | `fadd` `fsub` `fmul` `fdiv` | `__aeabi_fadd` … / `__aeabi_dadd` … | `__addsf3` … / `__adddf3` … |
//! | `frem` | `fmodf` / `fmod` (C library) | `fmodf` / `fmod` |
//! | `fcmp` | `__aeabi_fcmpeq` `lt` `le` `ge` `gt` `un` (nonzero when the relation holds) | `__eqsf2` `__nesf2` `__ltsf2` `__lesf2` `__gtsf2` `__gesf2` `__unordsf2` (an `int` compared against 0) |
//! | `fptosi` / `fptoui` to ≤ 32 / ≤ 64 bits | `__aeabi_f2iz` `f2uiz` / `f2lz` `f2ulz` | `__fixsfsi` `__fixunssfsi` / `__fixsfdi` `__fixunssfdi` |
//! | `sitofp` / `uitofp` from ≤ 32 / ≤ 64 bits | `__aeabi_i2f` `ui2f` / `l2f` `ul2f` | `__floatsisf` `__floatunsisf` / `__floatdisf` `__floatundisf` |
//! | `fpext` / `fptrunc` | `__aeabi_f2d` / `__aeabi_d2f` | `__extendsfdf2` / `__truncdfsf2` |
//! | `f16` ↔ `f32`, `f64` → `f16` | `__aeabi_h2f` `f2h` `d2h` | `__extendhfsf2` `__truncsfhf2` `__truncdfhf2` |
//!
//! (the `f64` column of each is the `d`/`df` form). With the RTABI names an
//! ordered predicate calls one helper, its unordered complement negates one
//! (`ugt` is `!ole`), and `one`/`ueq` or two (`olt | ogt`, `uno | oeq`). With
//! the libgcc names each predicate compares one helper's `int` against zero
//! with the sign libgcc documents (`ult` is `__gesf2 < 0`), `one` is
//! `__unordsf2 == 0 && __nesf2 != 0` and `ueq` `__unordsf2 != 0 || __eqsf2 ==
//! 0`; `int` is the target's (16 bits on AVR). `f16` values compute in `f32`:
//! the sum, difference, product and quotient of two halves are exact in `f32`
//! before the one rounding back, and an integer converts through `f64` (exact
//! below 2^53) to round once. Narrow integer sources and destinations of a
//! conversion go through the 32-bit helpers with an extension or truncation
//! (an out-of-range conversion is poison in the IR, so the helper's result
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

/// Which run-time helper names (and comparison conventions) the soft-float
/// calls use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SoftFloatAbi {
    /// The *Run-time ABI for the Arm Architecture* (`__aeabi_fadd`, ...),
    /// comparison helpers returning a nonzero 32-bit `int` when the relation
    /// holds.
    Aeabi,
    /// The libgcc names (`__addsf3`, `__ltsf2`, `__fixsfsi`, ...), comparison
    /// helpers returning a C `int` of `int_bits` bits compared against zero.
    Libgcc {
        /// The width of the target's C `int` (32 on most targets, 16 on AVR).
        int_bits: u32,
    },
}

impl SoftFloatAbi {
    /// Every helper this ABI may call: name, parameter widths, result width
    /// (all integers).
    pub fn helpers(self) -> Vec<(String, Vec<u32>, u32)> {
        let mut out: Vec<(String, Vec<u32>, u32)> = Vec::new();
        for dbl in [false, true] {
            let w = if dbl { 64 } else { 32 };
            for op in [BinOp::FAdd, BinOp::FSub, BinOp::FMul, BinOp::FDiv, BinOp::FRem] {
                out.push((self.arith(op, dbl), vec![w, w], w));
            }
            for pred in [FloatPred::Oeq, FloatPred::One, FloatPred::Ogt, FloatPred::Oge, FloatPred::Olt, FloatPred::Ole, FloatPred::Uno] {
                for (name, _) in self.compare(pred, dbl).expect("a non-constant predicate").0 {
                    if !out.iter().any(|h| h.0 == name) {
                        out.push((name, vec![w, w], self.cmp_bits()));
                    }
                }
            }
            for long in [false, true] {
                let iw = if long { 64 } else { 32 };
                for signed in [false, true] {
                    out.push((self.fp_to_int(dbl, signed, long), vec![w], iw));
                    out.push((self.int_to_fp(signed, long, dbl), vec![iw], w));
                }
            }
        }
        out.push((self.f2d().to_owned(), vec![32], 64));
        out.push((self.d2f().to_owned(), vec![64], 32));
        let (h2f, hw) = self.h2f();
        out.push((h2f.to_owned(), vec![hw], 32));
        let (f2h, rw) = self.f2h(false);
        out.push((f2h.to_owned(), vec![32], rw));
        let (d2h, rw) = self.f2h(true);
        out.push((d2h.to_owned(), vec![64], rw));
        out
    }

    /// The helper of a binary float operation (`dbl`: on `f64`).
    pub fn arith(self, op: BinOp, dbl: bool) -> String {
        if op == BinOp::FRem {
            return (if dbl { "fmod" } else { "fmodf" }).to_owned();
        }
        let base = match op {
            BinOp::FAdd => "add",
            BinOp::FSub => "sub",
            BinOp::FMul => "mul",
            _ => "div",
        };
        match self {
            SoftFloatAbi::Aeabi => format!("__aeabi_{}{base}", if dbl { 'd' } else { 'f' }),
            SoftFloatAbi::Libgcc { .. } => format!("__{base}{}3", if dbl { "df" } else { "sf" }),
        }
    }

    /// The result width of the comparison helpers.
    pub fn cmp_bits(self) -> u32 {
        match self {
            SoftFloatAbi::Aeabi => 32,
            SoftFloatAbi::Libgcc { int_bits } => int_bits,
        }
    }

    /// How an `fcmp` predicate is computed: helper calls, each result compared
    /// against zero with its predicate, combined with `or` (`true`) or `and`.
    /// `None` for the constant predicates.
    pub fn compare(self, pred: FloatPred, dbl: bool) -> Option<(Vec<(String, IntPred)>, bool)> {
        use FloatPred::*;
        if matches!(pred, False | True) {
            return None;
        }
        Some(match self {
            SoftFloatAbi::Aeabi => {
                let h = |rel: &str| format!("__aeabi_{}cmp{rel}", if dbl { 'd' } else { 'f' });
                let yes = |rel: &str| (h(rel), IntPred::Ne);
                let no = |rel: &str| (h(rel), IntPred::Eq);
                match pred {
                    Oeq => (vec![yes("eq")], false),
                    Olt => (vec![yes("lt")], false),
                    Ole => (vec![yes("le")], false),
                    Ogt => (vec![yes("gt")], false),
                    Oge => (vec![yes("ge")], false),
                    Uno => (vec![yes("un")], false),
                    Ord => (vec![no("un")], false),
                    Ugt => (vec![no("le")], false),
                    Uge => (vec![no("lt")], false),
                    Ult => (vec![no("ge")], false),
                    Ule => (vec![no("gt")], false),
                    Une => (vec![no("eq")], false),
                    One => (vec![yes("lt"), yes("gt")], true),
                    _ => (vec![yes("un"), yes("eq")], true),
                }
            }
            SoftFloatAbi::Libgcc { .. } => {
                let h = |n: &str, p: IntPred| (format!("__{n}{}2", if dbl { "df" } else { "sf" }), p);
                match pred {
                    Oeq => (vec![h("eq", IntPred::Eq)], false),
                    One => (vec![h("unord", IntPred::Eq), h("ne", IntPred::Ne)], false),
                    Ogt => (vec![h("gt", IntPred::Sgt)], false),
                    Oge => (vec![h("ge", IntPred::Sge)], false),
                    Olt => (vec![h("lt", IntPred::Slt)], false),
                    Ole => (vec![h("le", IntPred::Sle)], false),
                    Ord => (vec![h("unord", IntPred::Eq)], false),
                    Uno => (vec![h("unord", IntPred::Ne)], false),
                    Ueq => (vec![h("unord", IntPred::Ne), h("eq", IntPred::Eq)], true),
                    Une => (vec![h("ne", IntPred::Ne)], false),
                    Ugt => (vec![h("le", IntPred::Sgt)], false),
                    Uge => (vec![h("lt", IntPred::Sge)], false),
                    Ult => (vec![h("ge", IntPred::Slt)], false),
                    _ => (vec![h("gt", IntPred::Sle)], false),
                }
            }
        })
    }

    /// The float-to-integer helper (`long`: a 64-bit result).
    pub fn fp_to_int(self, dbl: bool, signed: bool, long: bool) -> String {
        match self {
            SoftFloatAbi::Aeabi => format!(
                "__aeabi_{}2{}{}z",
                if dbl { 'd' } else { 'f' },
                if signed { "" } else { "u" },
                if long { 'l' } else { 'i' }
            ),
            SoftFloatAbi::Libgcc { .. } => format!(
                "__fix{}{}{}",
                if signed { "" } else { "uns" },
                if dbl { "df" } else { "sf" },
                if long { "di" } else { "si" }
            ),
        }
    }

    /// The integer-to-float helper (`long`: a 64-bit source).
    pub fn int_to_fp(self, signed: bool, long: bool, dbl: bool) -> String {
        match self {
            SoftFloatAbi::Aeabi => format!(
                "__aeabi_{}{}2{}",
                if signed { "" } else { "u" },
                if long { 'l' } else { 'i' },
                if dbl { 'd' } else { 'f' }
            ),
            SoftFloatAbi::Libgcc { .. } => format!(
                "__float{}{}{}",
                if signed { "" } else { "un" },
                if long { "di" } else { "si" },
                if dbl { "df" } else { "sf" }
            ),
        }
    }

    /// `f32` → `f64`.
    pub fn f2d(self) -> &'static str {
        match self {
            SoftFloatAbi::Aeabi => "__aeabi_f2d",
            SoftFloatAbi::Libgcc { .. } => "__extendsfdf2",
        }
    }

    /// `f64` → `f32`.
    pub fn d2f(self) -> &'static str {
        match self {
            SoftFloatAbi::Aeabi => "__aeabi_d2f",
            SoftFloatAbi::Libgcc { .. } => "__truncdfsf2",
        }
    }

    /// `f16` → `f32`, with the width the half travels in (the RTABI passes
    /// it zero-extended in a word).
    pub fn h2f(self) -> (&'static str, u32) {
        match self {
            SoftFloatAbi::Aeabi => ("__aeabi_h2f", 32),
            SoftFloatAbi::Libgcc { .. } => ("__extendhfsf2", 16),
        }
    }

    /// `f32` (or `f64` with `from64`) → `f16`, with the result's width.
    pub fn f2h(self, from64: bool) -> (&'static str, u32) {
        match (self, from64) {
            (SoftFloatAbi::Aeabi, false) => ("__aeabi_f2h", 32),
            (SoftFloatAbi::Aeabi, true) => ("__aeabi_d2h", 32),
            (SoftFloatAbi::Libgcc { .. }, false) => ("__truncsfhf2", 16),
            (SoftFloatAbi::Libgcc { .. }, true) => ("__truncdfhf2", 16),
        }
    }
}

/// The name of the scratch declaration that carries a rewritten function's
/// new signature while it is rebuilt.
const SCRATCH: &str = "__lf_softfloat_scratch";

/// Rewrite `module` so that no floating-point type remains (see the [module
/// docs](self)), calling `abi`'s helpers. `syms` resolves and interns the
/// helper names.
///
/// # Errors
///
/// [`SoftFloatError::WideConversion`] for a conversion between a float and an
/// integer wider than 64 bits.
pub fn lower_soft_float(
    module: &mut Module,
    syms: &mut StrInterner,
    abi: SoftFloatAbi,
) -> Result<SoftFloatReport, SoftFloatError> {
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
    let mut helpers: HashMap<String, FuncId> = HashMap::new();
    for (name, params, ret) in abi.helpers() {
        let existing = (0..module.function_count())
            .map(FuncId::from_index)
            .find(|&f| syms.resolve(module.function(f).name) == name);
        let fid = match existing {
            Some(f) => f,
            None => {
                let ps: Vec<TypeId> = params.iter().map(|&w| module.types_mut().int(w)).collect();
                let r = module.types_mut().int(ret);
                let sig = module.types_mut().func(ps, r, false);
                module.declare_function(syms.intern(&name), sig)
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
                let mut sf = Sf { b, old, vmap: vec![None; old.value_count()], helpers: &helpers, abi };
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
    helpers: &'x HashMap<String, FuncId>,
    abi: SoftFloatAbi,
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
        let (name, w) = self.abi.h2f();
        let x = if w == 16 {
            h
        } else {
            let t = self.int(w);
            self.b.cast(CastOp::ZExt, h, t)
        };
        self.call(name, &[x], 32)
    }

    /// Round an `f32` (`i32`) or `f64` (`i64`) to an `f16` (`i16`).
    fn round_half(&mut self, v: ValueId, from64: bool) -> ValueId {
        let (name, w) = self.abi.f2h(from64);
        let r = self.call(name, &[v], w);
        if w == 16 {
            return r;
        }
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
        match k {
            FloatKind::F32 => {
                let n = self.abi.arith(op, false);
                self.call(&n, &[a, b], 32)
            }
            FloatKind::F64 => {
                let n = self.abi.arith(op, true);
                self.call(&n, &[a, b], 64)
            }
            FloatKind::F16 => {
                let (x, y) = (self.h2f(a), self.h2f(b));
                let n = self.abi.arith(op, false);
                let r = self.call(&n, &[x, y], 32);
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
        let Some((calls, any)) = self.abi.compare(pred, dbl) else {
            return self.b.const_bool(pred == FloatPred::True);
        };
        let bits = self.abi.cmp_bits();
        let mut acc: Option<ValueId> = None;
        for (name, p) in calls {
            let c = self.call(&name, &[a, b], bits);
            let t = self.int(bits);
            let zero = self.b.const_int(t, Int::ZERO);
            let bit = self.b.icmp(p, c, zero);
            acc = Some(match acc {
                None => bit,
                Some(x) => self.b.bin(if any { BinOp::Or } else { BinOp::And }, x, bit, Flags::NONE),
            });
        }
        acc.expect("at least one helper")
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
                        let n = self.abi.f2d();
                        self.call(n, &[x], 64)
                    }
                    (FloatKind::F32, FloatKind::F64) => {
                        let n = self.abi.f2d();
                        self.call(n, &[a], 64)
                    }
                    (FloatKind::F64, FloatKind::F32) => {
                        let n = self.abi.d2f();
                        self.call(n, &[a], 32)
                    }
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
                let name = self.abi.fp_to_int(dbl, signed, long);
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
                let name = self.abi.int_to_fp(signed, long, dbl);
                let r = self.call(&name, &[x], if dbl { 64 } else { 32 });
                if kind == FloatKind::F16 { self.round_half(r, true) } else { r }
            }
            _ => a,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::diagnostics::FileId;

    const SRC: &str = r#"
module "sf"
func @f(f32, f32, f64, i16) -> i1 {
entry ^0(%a: f32, %b: f32, %d: f64, %n: i16):
  %s = fadd %a, %b : f32
  %e = fpext %s : f64
  %m = fmul %e, %d : f64
  %i = sitofp %n : f32
  %t = fptrunc %m : f32
  %c = fcmp ult %t, %i : i1
  ret %c
}
"#;

    /// The helpers each ABI calls for the same program (checking that no
    /// float type survives the rewrite).
    fn called(abi: SoftFloatAbi) -> Vec<String> {
        let mut syms = StrInterner::new();
        let mut m = crate::ir::text::parse_module(SRC, FileId::new(0), &mut syms).unwrap();
        lower_soft_float(&mut m, &mut syms, abi).unwrap();
        crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("{e:?}"));
        let f = m.function(FuncId::from_index(0));
        for v in 0..f.value_count() {
            assert!(!m.types().get(f.value_type(ValueId::from_index(v))).is_float());
        }
        let mut out = Vec::new();
        for (_, b) in f.blocks() {
            for &i in b.insts() {
                if let InstKind::Call = f.inst(i).kind
                    && let ValueDef::Func(g) = f.value(f.inst(i).operands()[0]).def
                {
                    out.push(syms.resolve(m.function(g).name).to_owned());
                }
            }
        }
        out
    }

    #[test]
    fn helper_names_per_abi() {
        assert_eq!(
            called(SoftFloatAbi::Aeabi),
            ["__aeabi_fadd", "__aeabi_f2d", "__aeabi_dmul", "__aeabi_i2f", "__aeabi_d2f", "__aeabi_fcmpge"]
        );
        assert_eq!(
            called(SoftFloatAbi::Libgcc { int_bits: 16 }),
            ["__addsf3", "__extendsfdf2", "__muldf3", "__floatsisf", "__truncdfsf2", "__gesf2"]
        );
        for abi in [SoftFloatAbi::Aeabi, SoftFloatAbi::Libgcc { int_bits: 32 }] {
            let names: Vec<String> = abi.helpers().into_iter().map(|h| h.0).collect();
            let mut dedup = names.clone();
            dedup.sort();
            dedup.dedup();
            assert_eq!(dedup.len(), names.len(), "every {abi:?} helper is distinct");
        }
    }
}
