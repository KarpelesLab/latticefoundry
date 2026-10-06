//! Wide-integer **legalization**: split integer operations wider than a
//! target's native width into operations on narrower *parts*
//! (`docs/ir-design.md` §3b).
//!
//! A target computes natively with the integer widths its
//! [`DataLayout`](crate::ir::DataLayout) lists as native; an 8-bit AVR has no
//! `i64` adder and a 32-bit Cortex-M no 64-bit shifter. This pass, run on the IR
//! before instruction selection, rewrites every integer value wider than a
//! *part width* `W` (by default the layout's widest native integer) as `N`
//! values of type `iW` — part `0` the least significant — and every operation
//! on such values as operations on the parts. It is target-independent: wasm32
//! (native `i64`: nothing to do), Thumb (`W = 32`) and AVR (`W = 8` or `16`)
//! share it.
//!
//! # What is expanded
//!
//! | wide operation | expansion |
//! |---|---|
//! | `and`, `or`, `xor`, `select`, `freeze` | per part |
//! | `add`, `sub` | per part with a carry / borrow chain (`icmp ult`) |
//! | `shl`, `lshr`, `ashr` by a constant | part moves and funnel shifts |
//! | `shl`, `lshr`, `ashr` by a variable | funnel shifts by `s mod W`, then a `select` ladder on `s / W` |
//! | `icmp eq`/`ne` | `or` of the parts' `xor`s against zero |
//! | `icmp` ordered | lexicographic from the top part (signed there, unsigned below) |
//! | `mul`, `udiv`, `sdiv`, `urem`, `srem` | a **libcall** (`__muldi3`, `__udivdi3`, ...; see [`libgcc_libcall`]) |
//! | `trunc`, `zext`, `sext` | part selection, zero / sign-fill parts |
//! | non-volatile `load` / `store` | one access per part, at the layout's byte order |
//! | block parameters and branch arguments | one parameter / argument per part |
//!
//! Flags (`nsw`, `nuw`, `exact`) are dropped: they only make the original
//! result poison more often, so the expansion refines it (tenet T3). A shift by
//! an amount `≥` the width, poison in the original, yields some value.
//!
//! # The ABI boundary
//!
//! A value crosses a function boundary whole: the pass keeps the function
//! signature, and a wide **entry parameter**, **call argument or result**,
//! **return value**, and the operands/results of the other operations that
//! only a backend can split (`ptrtoint`/`inttoptr`, `bitcast` and float
//! conversions, `syscall`, a `switch` condition, a `ptr_add` offset,
//! `dyn_alloca`, volatile and atomic accesses) stay wide. The pass converts at
//! those points with a fixed, recognizable shape:
//!
//! - **split** `x` into parts: `trunc x` for part 0, `trunc (lshr x, k·W)` for
//!   part `k`;
//! - **join** parts into `x`: `zext p0`, then `or` with `shl (zext pk), k·W`.
//!
//! After the pass, those split/join helpers and the boundary operations are
//! the *only* instructions touching an integer wider than `W`
//! ([`illegal_int_ops`] reports anything else). A backend lowers them in its
//! ABI seam, where a wide value lives in a register group (e.g. AVR passes an
//! `i64` in eight registers). The mul/div/rem libcalls are ordinary calls with
//! wide arguments, so they follow the same convention.
//!
//! # Limits
//!
//! A wide width must be a multiple of `W` (`i64`, `i128` on `W ∈ {8, 16, 32,
//! 64}`; not `i40` on `W = 32`), and a variable shift needs the width to fit
//! `2^W` (always true for `W ≥ 16`; `i256` at most for `W = 8`). Other cases are
//! a [`LegalizeError`]. Functions named like one of the libcalls are left
//! untouched: they implement the operation (a runtime written in IR).

use std::collections::HashMap;

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, CastOp, Flags, InstId, InstKind, IntPred};
use crate::ir::types::{Type, TypeId};
use crate::ir::value::{Const, ValueDef, ValueId};
use crate::ir::{BlockId, Endian, FuncId, Function, Module};
use crate::support::StrInterner;
use crate::transform::dom_preorder;

use puremp::Int;

/// How [`legalize_ints`] splits: the part width and the libcall naming.
#[derive(Clone, Copy, Debug)]
pub struct LegalizeOptions {
    /// The part width `W` in bits (8, 16, 32 or 64): integers wider than this
    /// are split into `W`-bit parts.
    pub part_bits: u32,
    /// The name of the function implementing a wide `mul`/`udiv`/`sdiv`/`urem`/
    /// `srem` of the given total width (`fn(T, T) -> T`).
    pub libcall_name: fn(BinOp, u32) -> String,
}

impl LegalizeOptions {
    /// Split above `part_bits`, calling libgcc-named helpers.
    pub fn new(part_bits: u32) -> LegalizeOptions {
        LegalizeOptions { part_bits, libcall_name: libgcc_libcall }
    }

    /// Split above the layout's widest native integer width.
    pub fn for_layout(layout: &crate::ir::DataLayout) -> LegalizeOptions {
        LegalizeOptions::new(layout.max_native_int())
    }
}

/// The libgcc name of the helper for a wide `op` on `bits`-bit integers:
/// `__mul{s,d,t}i3`, `__udiv…3`, `__div…3`, `__umod…3`, `__mod…3` for 32, 64
/// and 128 bits, and `__lf_{mul,udiv,div,umod,mod}_i{bits}` for other widths.
pub fn libgcc_libcall(op: BinOp, bits: u32) -> String {
    let base = match op {
        BinOp::Mul => "mul",
        BinOp::UDiv => "udiv",
        BinOp::SDiv => "div",
        BinOp::URem => "umod",
        BinOp::SRem => "mod",
        _ => "op",
    };
    match bits {
        32 => format!("__{base}si3"),
        64 => format!("__{base}di3"),
        128 => format!("__{base}ti3"),
        _ => format!("__lf_{base}_i{bits}"),
    }
}

/// Why a module cannot be legalized.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LegalizeError {
    /// The part width is not 8, 16, 32 or 64.
    BadPartWidth(u32),
    /// An integer wider than the part width is not a multiple of it.
    UnsupportedWidth(u32),
    /// A variable shift of this width cannot be expanded: its amount does not
    /// fit one part (the width exceeds `2^W`).
    ShiftTooWide(u32),
}

impl std::fmt::Display for LegalizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LegalizeError::BadPartWidth(w) => write!(f, "part width {w} is not 8, 16, 32 or 64"),
            LegalizeError::UnsupportedWidth(w) => write!(f, "i{w} is not a multiple of the part width"),
            LegalizeError::ShiftTooWide(w) => write!(f, "a variable shift of i{w} cannot be split"),
        }
    }
}

impl std::error::Error for LegalizeError {}

/// What [`legalize_ints`] did.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct LegalizeReport {
    /// The functions whose bodies were rewritten.
    pub functions: Vec<FuncId>,
    /// The libcall helpers the rewritten code calls, with the function each
    /// resolved to (an existing one of that name, or a new declaration).
    pub libcalls: Vec<(String, FuncId)>,
}

/// Legalize every function of `module` for `opts.part_bits` (see the [module
/// docs](self)). `syms` resolves and interns the libcall names. Functions with
/// no integer wider than the part width are left untouched.
pub fn legalize_ints(
    module: &mut Module,
    syms: &mut StrInterner,
    opts: &LegalizeOptions,
) -> Result<LegalizeReport, LegalizeError> {
    let w = opts.part_bits;
    if ![8, 16, 32, 64].contains(&w) {
        return Err(LegalizeError::BadPartWidth(w));
    }

    // 1. Validate the widths and collect the libcalls each function needs.
    let mut needed: Vec<(BinOp, u32)> = Vec::new();
    let mut work: Vec<FuncId> = Vec::new();
    let mut libcall_names: Vec<String> = Vec::new();
    for fi in 0..module.function_count() {
        let fid = FuncId::from_index(fi);
        let f = module.function(fid);
        if f.is_declaration() {
            continue;
        }
        let mut wide = false;
        for vi in 0..f.value_count() {
            if let Type::Int(bits) = module.types().get(f.value_type(ValueId::from_index(vi)))
                && *bits > w
            {
                if bits % w != 0 {
                    return Err(LegalizeError::UnsupportedWidth(*bits));
                }
                wide = true;
            }
        }
        if !wide {
            continue;
        }
        for (_, block) in f.blocks() {
            for &i in block.insts() {
                let inst = f.inst(i);
                let &Type::Int(bits) = module.types().get(inst.ty) else { continue };
                if bits <= w {
                    continue;
                }
                match inst.kind {
                    InstKind::Bin(op @ (BinOp::Mul | BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem))
                        if !needed.contains(&(op, bits)) =>
                    {
                        needed.push((op, bits));
                    }
                    InstKind::Bin(BinOp::Shl | BinOp::LShr | BinOp::AShr)
                        if w < 64 && u64::from(bits) > (1u64 << w) && !is_const(f, inst.operands()[1]) =>
                    {
                        return Err(LegalizeError::ShiftTooWide(bits));
                    }
                    _ => {}
                }
            }
        }
        work.push(fid);
    }
    needed.sort_by_key(|&(op, bits)| (bits, op as u8));
    for &(op, bits) in &needed {
        libcall_names.push((opts.libcall_name)(op, bits));
    }

    // 2. Resolve (or declare) each libcall. A function already named like a
    //    libcall implements it and is not itself rewritten.
    let mut report = LegalizeReport::default();
    let mut libcalls: HashMap<(BinOp, u32), FuncId> = HashMap::new();
    for (&(op, bits), name) in needed.iter().zip(&libcall_names) {
        let existing = (0..module.function_count())
            .map(FuncId::from_index)
            .find(|&f| syms.resolve(module.function(f).name) == name.as_str());
        let fid = match existing {
            Some(f) => f,
            None => {
                let t = module.types_mut().int(bits);
                let sig = module.types_mut().func(vec![t, t], t, false);
                module.declare_function(syms.intern(name), sig)
            }
        };
        libcalls.insert((op, bits), fid);
        report.libcalls.push((name.clone(), fid));
    }
    work.retain(|&f| !libcall_names.iter().any(|n| syms.resolve(module.function(f).name) == n.as_str()));

    // 3. Rebuild each function.
    let big_endian = module.data_layout().endian() == Endian::Big;
    for fid in work {
        // `map_function` carries the attributes (linkage, visibility,
        // secrecy) and the declaration line over.
        let (fresh, ()) = module.map_function(fid, |old, b| {
            let part_ty = b.types_mut().int(w);
            let mut lz = Lz { b, old, w, part_ty, vmap: vec![None; old.value_count()], libcalls: &libcalls, big_endian };
            lz.run();
        });
        module.replace_function(fid, fresh);
        report.functions.push(fid);
    }
    Ok(report)
}

/// The instructions of `func` that still compute on an integer wider than
/// `part_bits` other than through the ABI-boundary operations and the
/// split/join helpers (see the [module docs](self)). Empty after
/// [`legalize_ints`] with that part width.
pub fn illegal_int_ops(module: &Module, func: FuncId, part_bits: u32) -> Vec<InstId> {
    let f = module.function(func);
    let types = module.types();
    let wide = |v: ValueId| matches!(types.get(f.value_type(v)), Type::Int(b) if *b > part_bits);
    let wide_ty = |t: TypeId| matches!(types.get(t), Type::Int(b) if *b > part_bits);
    let mut out = Vec::new();
    for (_, block) in f.blocks() {
        for &i in block.insts().iter().chain(block.terminator().as_ref()) {
            let inst = f.inst(i);
            let ops = inst.operands();
            let res_wide = inst.result().is_some() && wide_ty(inst.ty);
            if !res_wide && !ops.iter().any(|&o| wide(o)) {
                continue;
            }
            let ok = match &inst.kind {
                // ABI-boundary operations.
                InstKind::Call
                | InstKind::Ret
                | InstKind::Syscall
                | InstKind::Switch(_)
                | InstKind::PtrAdd { .. }
                | InstKind::DynAlloca { .. }
                | InstKind::AtomicLoad { .. }
                | InstKind::AtomicStore { .. }
                | InstKind::AtomicRmw { .. }
                | InstKind::CmpXchg { .. }
                | InstKind::Load { volatile: true, .. }
                | InstKind::Store { volatile: true, .. } => true,
                InstKind::Cast(op) => match op {
                    // Split: part extraction.
                    CastOp::Trunc => !res_wide,
                    // Join: a part widened.
                    CastOp::ZExt => !wide(ops[0]),
                    CastOp::SExt => false,
                    _ => true,
                },
                // Join: parts or-ed together.
                InstKind::Bin(BinOp::Or) => true,
                // Split / join: shifts by a whole number of parts.
                InstKind::Bin(BinOp::Shl | BinOp::LShr) => const_int(module, f, ops[1])
                    .and_then(|c| c.to_u64())
                    .is_some_and(|s| s % u64::from(part_bits) == 0),
                _ => false,
            };
            if !ok {
                out.push(i);
            }
        }
    }
    out
}

/// Whether `v` is a constant.
fn is_const(f: &Function, v: ValueId) -> bool {
    matches!(f.value(v).def, ValueDef::Const(_))
}

/// The integer value of a constant operand, if it is one.
fn const_int(module: &Module, f: &Function, v: ValueId) -> Option<Int> {
    match &f.value(v).def {
        ValueDef::Const(c) => match module.consts().get(*c) {
            Const::Int { value, .. } => Some(value.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// An old value's image in the rebuilt function.
#[derive(Clone, Debug)]
enum Mapped {
    /// A value of a legal type.
    One(ValueId),
    /// A wide integer's parts, least significant first.
    Parts(Vec<ValueId>),
}

/// The per-function rebuild state.
struct Lz<'x, 'b> {
    b: &'x mut FunctionBuilder<'b>,
    old: &'x Function,
    /// The part width.
    w: u32,
    /// The part type, `iW`.
    part_ty: TypeId,
    /// Old value → its image.
    vmap: Vec<Option<Mapped>>,
    libcalls: &'x HashMap<(BinOp, u32), FuncId>,
    big_endian: bool,
}

impl Lz<'_, '_> {
    // --- types and constants ------------------------------------------------

    /// The number of parts of a wide integer type, or `None` for a legal type.
    fn parts_of(&self, ty: TypeId) -> Option<usize> {
        match self.b.types().get(ty) {
            Type::Int(bits) if *bits > self.w => Some((*bits / self.w) as usize),
            _ => None,
        }
    }

    fn is_wide(&self, v: ValueId) -> bool {
        self.parts_of(self.old.value_type(v)).is_some()
    }

    fn part_const(&mut self, v: u64) -> ValueId {
        self.b.const_int(self.part_ty, Int::from_u64(v))
    }

    fn int_const(&mut self, ty: TypeId, v: u64) -> ValueId {
        self.b.const_int(ty, Int::from_u64(v))
    }

    // --- value mapping ------------------------------------------------------

    /// The legal image of an old value: a mapped value, a constant / global /
    /// function reference, a wide value joined back whole (at a boundary), or
    /// poison for a definition in unreachable code not yet rebuilt.
    fn one(&mut self, v: ValueId) -> ValueId {
        match self.vmap[v.index()].clone() {
            Some(Mapped::One(x)) => return x,
            Some(Mapped::Parts(p)) => return self.join(&p, self.old.value_type(v)),
            None => {}
        }
        let x = match &self.old.value(v).def {
            ValueDef::Const(c) => self.b.use_const(*c),
            ValueDef::Global(g) => self.b.global_ref(*g),
            ValueDef::Func(f) => self.b.func_ref(*f),
            ValueDef::Param(..) | ValueDef::Inst(..) => {
                if self.is_wide(v) {
                    let p = self.parts(v);
                    return self.join(&p, self.old.value_type(v));
                }
                self.b.poison(self.old.value_type(v))
            }
        };
        self.vmap[v.index()] = Some(Mapped::One(x));
        x
    }

    /// The parts of a wide old value.
    fn parts(&mut self, v: ValueId) -> Vec<ValueId> {
        match self.vmap[v.index()].clone() {
            Some(Mapped::Parts(p)) => return p,
            Some(Mapped::One(x)) => return self.split(x, self.old.value_type(v)),
            None => {}
        }
        let ty = self.old.value_type(v);
        let n = self.parts_of(ty).expect("a wide value");
        let konst = match &self.old.value(v).def {
            ValueDef::Const(c) => Some(self.b.consts().get(*c).clone()),
            _ => None,
        };
        let p: Vec<ValueId> = match konst {
            Some(Const::Int { value, .. }) => {
                let total = self.w * n as u32;
                let bits = value.mod_2k(total);
                (0..n)
                    .map(|k| {
                        let part = bits.div_2k_trunc(self.w * k as u32).mod_2k(self.w);
                        self.b.const_int(self.part_ty, part)
                    })
                    .collect()
            }
            // Poison (or an unrebuilt definition in unreachable code).
            _ => (0..n).map(|_| self.b.poison(self.part_ty)).collect(),
        };
        self.vmap[v.index()] = Some(Mapped::Parts(p.clone()));
        p
    }

    /// Split a whole wide value `x` of type `ty` into parts (boundary helper).
    fn split(&mut self, x: ValueId, ty: TypeId) -> Vec<ValueId> {
        let n = self.parts_of(ty).expect("a wide value");
        (0..n)
            .map(|k| {
                let src = if k == 0 {
                    x
                } else {
                    let sh = self.int_const(ty, u64::from(self.w) * k as u64);
                    self.b.bin(BinOp::LShr, x, sh, Flags::NONE)
                };
                self.b.cast(CastOp::Trunc, src, self.part_ty)
            })
            .collect()
    }

    /// Join parts into one wide value of type `ty` (boundary helper).
    fn join(&mut self, parts: &[ValueId], ty: TypeId) -> ValueId {
        let mut acc = self.b.cast(CastOp::ZExt, parts[0], ty);
        for (k, &p) in parts.iter().enumerate().skip(1) {
            let z = self.b.cast(CastOp::ZExt, p, ty);
            let sh = self.int_const(ty, u64::from(self.w) * k as u64);
            let s = self.b.bin(BinOp::Shl, z, sh, Flags::NONE);
            acc = self.b.bin(BinOp::Or, acc, s, Flags::NONE);
        }
        acc
    }

    fn set(&mut self, old: Option<ValueId>, m: Mapped) {
        if let Some(r) = old {
            self.vmap[r.index()] = Some(m);
        }
    }

    // --- the rebuild --------------------------------------------------------

    fn run(&mut self) {
        let old = self.old;
        let n = old.block_count();
        let entry = old.entry().expect("a definition has an entry block").index();
        let cfg = ControlFlowGraph::new(old);
        let doms = Dominators::new(old, &cfg);

        // Blocks: the entry keeps the signature; any other block's wide
        // parameters become one parameter per part.
        let mut new_block: Vec<BlockId> = Vec::with_capacity(n);
        for b in 0..n {
            let bb = BlockId::from_index(b);
            if b == entry {
                new_block.push(self.b.create_entry_block());
                continue;
            }
            let mut tys = Vec::new();
            for &p in old.block(bb).params() {
                let ty = old.value_type(p);
                match self.parts_of(ty) {
                    Some(k) => tys.extend(std::iter::repeat_n(self.part_ty, k)),
                    None => tys.push(ty),
                }
            }
            new_block.push(self.b.create_block(&tys));
        }
        for (b, &nb) in new_block.iter().enumerate() {
            let bb = BlockId::from_index(b);
            if b == entry {
                continue;
            }
            let new_params = self.b.block_params(nb).to_vec();
            let mut at = 0;
            for &p in old.block(bb).params() {
                match self.parts_of(old.value_type(p)) {
                    Some(k) => {
                        self.vmap[p.index()] = Some(Mapped::Parts(new_params[at..at + k].to_vec()));
                        at += k;
                    }
                    None => {
                        self.vmap[p.index()] = Some(Mapped::One(new_params[at]));
                        at += 1;
                    }
                }
            }
        }

        for b in dom_preorder(old, &doms) {
            let bb = BlockId::from_index(b);
            self.b.switch_to(new_block[b]);
            if b == entry {
                // Wide function parameters arrive whole: split them first (at
                // the entry block's first line).
                self.b.set_line(crate::transform::block_line(old, bb));
                let params = self.b.block_params(new_block[b]).to_vec();
                for (&op, &np) in old.block(bb).params().iter().zip(&params) {
                    let ty = old.value_type(op);
                    let m = if self.parts_of(ty).is_some() { Mapped::Parts(self.split(np, ty)) } else { Mapped::One(np) };
                    self.vmap[op.index()] = Some(m);
                }
            }
            for &i in old.block(bb).insts() {
                // Every part of a split instruction takes its line.
                self.b.set_line_from(old, i);
                self.inst(i);
            }
            self.terminator(bb, &new_block);
        }
    }

    /// The edge arguments of an old branch, each wide one as its parts.
    fn edge_args(&mut self, args: &[ValueId]) -> Vec<ValueId> {
        let mut out = Vec::with_capacity(args.len());
        for &a in args {
            if self.is_wide(a) {
                out.extend(self.parts(a));
            } else {
                out.push(self.one(a));
            }
        }
        out
    }

    fn terminator(&mut self, bb: BlockId, new_block: &[BlockId]) {
        let old = self.old;
        let Some(t) = old.block(bb).terminator() else { return };
        self.b.set_line_from(old, t);
        let term = old.inst(t);
        let ops = term.operands();
        match &term.kind {
            InstKind::Ret => {
                let v = ops.first().map(|&o| self.one(o));
                self.b.ret(v);
            }
            InstKind::Unreachable => self.b.unreachable(),
            InstKind::Br(target) => {
                let args = self.edge_args(ops);
                self.b.br(new_block[target.index()], &args);
            }
            InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let (ta, fa) = (*true_args as usize, *false_args as usize);
                let cond = self.one(ops[0]);
                let targs = self.edge_args(&ops[1..1 + ta]);
                let fargs = self.edge_args(&ops[1 + ta..1 + ta + fa]);
                self.b.cond_br(cond, new_block[if_true.index()], &targs, new_block[if_false.index()], &fargs);
            }
            InstKind::Switch(data) => {
                // A wide condition is joined (a boundary): the target lowers
                // the comparisons against its constants.
                let cond = self.one(ops[0]);
                let da = data.default_args as usize;
                let dargs = self.edge_args(&ops[1..1 + da]);
                let mut cases = Vec::with_capacity(data.cases.len());
                let mut off = 1 + da;
                for c in &data.cases {
                    let ca = c.args as usize;
                    let cargs = self.edge_args(&ops[off..off + ca]);
                    cases.push((c.value.clone(), new_block[c.target.index()], cargs));
                    off += ca;
                }
                self.b.switch(cond, new_block[data.default.index()], &dargs, cases);
            }
            _ => {}
        }
    }

    fn inst(&mut self, id: InstId) {
        let old = self.old;
        let inst = old.inst(id);
        let ops = inst.operands();
        let res = inst.result();
        let res_parts = res.and_then(|_| self.parts_of(inst.ty));
        let any_wide = res_parts.is_some() || ops.iter().any(|&o| self.is_wide(o));
        if !any_wide {
            let new_ops: Vec<ValueId> = ops.iter().map(|&o| self.one(o)).collect();
            let r = self.b.append_inst(inst.kind.clone(), new_ops, inst.flags, res.map(|_| inst.ty));
            if let Some(r) = r {
                self.set(res, Mapped::One(r));
            }
            return;
        }
        match (&inst.kind, res_parts) {
            (InstKind::Bin(op), Some(n)) => {
                let m = self.bin(*op, ops[0], ops[1], n, inst.ty);
                self.set(res, Mapped::Parts(m));
            }
            (InstKind::ICmp(pred), None) => {
                let r = self.icmp(*pred, ops[0], ops[1]);
                self.set(res, Mapped::One(r));
            }
            (InstKind::Cast(CastOp::Trunc), _) => {
                let src = self.parts(ops[0]);
                match res_parts {
                    Some(n) => self.set(res, Mapped::Parts(src[..n].to_vec())),
                    None => {
                        let to = inst.ty;
                        let r = if to == self.part_ty { src[0] } else { self.b.cast(CastOp::Trunc, src[0], to) };
                        self.set(res, Mapped::One(r));
                    }
                }
            }
            (InstKind::Cast(op @ (CastOp::ZExt | CastOp::SExt)), Some(n)) => {
                let signed = *op == CastOp::SExt;
                let mut p = if self.is_wide(ops[0]) {
                    self.parts(ops[0])
                } else {
                    let v = self.one(ops[0]);
                    let v = if old.value_type(ops[0]) == self.part_ty { v } else { self.b.cast(*op, v, self.part_ty) };
                    vec![v]
                };
                let fill = if signed {
                    let sh = self.part_const(u64::from(self.w - 1));
                    self.b.bin(BinOp::AShr, *p.last().expect("a part"), sh, Flags::NONE)
                } else {
                    self.part_const(0)
                };
                p.resize(n, fill);
                self.set(res, Mapped::Parts(p));
            }
            (InstKind::Select, Some(_)) => {
                let cond = self.one(ops[0]);
                let (t, f) = (self.parts(ops[1]), self.parts(ops[2]));
                let p = t.iter().zip(&f).map(|(&a, &b)| self.b.select(cond, a, b)).collect();
                self.set(res, Mapped::Parts(p));
            }
            (InstKind::Freeze, Some(_)) => {
                let p = self.parts(ops[0]).into_iter().map(|x| self.b.freeze(x)).collect();
                self.set(res, Mapped::Parts(p));
            }
            // Each part keeps the access's `secret` flag, so secret memory
            // stays secret after the split (constant time, §6d).
            (InstKind::Load { align, volatile: false, secret, .. }, Some(n)) => {
                let ptr = self.one(ops[0]);
                let secret = *secret;
                let p = (0..n)
                    .map(|k| {
                        let (addr, a) = self.part_addr(ptr, k, n, *align);
                        let kind = InstKind::Load { ty: self.part_ty, align: a, volatile: false, secret };
                        self.b
                            .append_inst(kind, vec![addr], Flags::NONE, Some(self.part_ty))
                            .expect("a load has a result")
                    })
                    .collect();
                self.set(res, Mapped::Parts(p));
            }
            (InstKind::Store { align, volatile: false, secret, .. }, None) if self.is_wide(ops[1]) => {
                let ptr = self.one(ops[0]);
                let vals = self.parts(ops[1]);
                let n = vals.len();
                let secret = *secret;
                for (k, &v) in vals.iter().enumerate() {
                    let (addr, a) = self.part_addr(ptr, k, n, *align);
                    let kind = InstKind::Store { ty: self.part_ty, align: a, volatile: false, secret };
                    self.b.append_inst(kind, vec![addr, v], Flags::NONE, None);
                }
            }
            // Everything else is a boundary: operate on whole values, joining
            // wide operands and splitting a wide result.
            _ => {
                let new_ops: Vec<ValueId> = ops.iter().map(|&o| self.one(o)).collect();
                let r = self.b.append_inst(inst.kind.clone(), new_ops, inst.flags, res.map(|_| inst.ty));
                if let Some(r) = r {
                    let m = if res_parts.is_some() { Mapped::Parts(self.split(r, inst.ty)) } else { Mapped::One(r) };
                    self.set(res, m);
                }
            }
        }
    }

    /// The address and alignment of part `k` of an `n`-part access at `ptr`
    /// aligned to `align`: parts are laid out in the data layout's byte order.
    fn part_addr(&mut self, ptr: ValueId, k: usize, n: usize, align: u32) -> (ValueId, u32) {
        let bytes = u64::from(self.w / 8);
        let slot = if self.big_endian { n - 1 - k } else { k } as u64;
        let off = slot * bytes;
        if off == 0 {
            return (ptr, align);
        }
        // The offset as a part-width integer when it fits, else pointer-wide.
        let off_ty = if self.w >= 64 || off < (1u64 << (self.w - 1)) {
            self.part_ty
        } else {
            let space = self.b.types().addr_space(self.b.value_type(ptr)).unwrap_or(0);
            let bits = self.b.types().data_layout().pointer_bits(space);
            self.b.types_mut().int(bits)
        };
        let o = self.int_const(off_ty, off);
        let addr = self.b.ptr_add(ptr, o, false);
        let a = u64::from(align).min(1 << off.trailing_zeros());
        (addr, a as u32)
    }

    // --- arithmetic ---------------------------------------------------------

    fn bin(&mut self, op: BinOp, l: ValueId, r: ValueId, n: usize, ty: TypeId) -> Vec<ValueId> {
        let b = self.b.types().bit_width(ty).expect("an integer");
        match op {
            BinOp::And | BinOp::Or | BinOp::Xor => {
                let (a, c) = (self.parts(l), self.parts(r));
                a.iter().zip(&c).map(|(&x, &y)| self.b.bin(op, x, y, Flags::NONE)).collect()
            }
            BinOp::Add => {
                let (a, c) = (self.parts(l), self.parts(r));
                let mut out = Vec::with_capacity(n);
                let mut carry: Option<ValueId> = None;
                for k in 0..n {
                    let t = self.b.add(a[k], c[k], Flags::NONE);
                    let (s, co) = match carry {
                        None => (t, self.b.icmp(IntPred::Ult, t, a[k])),
                        Some(ci) => {
                            let z = self.b.cast(CastOp::ZExt, ci, self.part_ty);
                            let s = self.b.add(t, z, Flags::NONE);
                            let c1 = self.b.icmp(IntPred::Ult, t, a[k]);
                            let c2 = self.b.icmp(IntPred::Ult, s, t);
                            (s, self.b.bin(BinOp::Or, c1, c2, Flags::NONE))
                        }
                    };
                    out.push(s);
                    carry = (k + 1 < n).then_some(co);
                }
                out
            }
            BinOp::Sub => {
                let (a, c) = (self.parts(l), self.parts(r));
                let mut out = Vec::with_capacity(n);
                let mut borrow: Option<ValueId> = None;
                for k in 0..n {
                    let t = self.b.sub(a[k], c[k], Flags::NONE);
                    let b1 = self.b.icmp(IntPred::Ult, a[k], c[k]);
                    let (d, bo) = match borrow {
                        None => (t, b1),
                        Some(bi) => {
                            let z = self.b.cast(CastOp::ZExt, bi, self.part_ty);
                            let d = self.b.sub(t, z, Flags::NONE);
                            let b2 = self.b.icmp(IntPred::Ult, t, z);
                            (d, self.b.bin(BinOp::Or, b1, b2, Flags::NONE))
                        }
                    };
                    out.push(d);
                    borrow = (k + 1 < n).then_some(bo);
                }
                out
            }
            BinOp::Shl | BinOp::LShr | BinOp::AShr => {
                let a = self.parts(l);
                let amount = match &self.old.value(r).def {
                    ValueDef::Const(c) => match self.b.consts().get(*c) {
                        Const::Int { value, .. } => Some(value.mod_2k(b)),
                        _ => None,
                    },
                    _ => None,
                };
                match amount {
                    Some(s) => match s.to_u64().filter(|&s| s < u64::from(b)) {
                        Some(s) => self.shift_const(op, &a, s as u32),
                        // Poison in the original: any value refines it.
                        None => (0..n).map(|_| self.b.poison(self.part_ty)).collect(),
                    },
                    None => {
                        let s = self.parts(r)[0];
                        self.shift_var(op, &a, s)
                    }
                }
            }
            BinOp::Mul | BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => {
                let f = self.libcalls[&(op, b)];
                let callee = self.b.func_ref(f);
                let (x, y) = (self.one(l), self.one(r));
                let res = self.b.call(callee, &[x, y], ty).expect("the libcall returns a value");
                self.split(res, ty)
            }
            // Float ops never have an integer type.
            _ => unreachable!("{op:?} on a wide integer"),
        }
    }

    /// Shift parts `a` by the constant `s` (`< width`).
    fn shift_const(&mut self, op: BinOp, a: &[ValueId], s: u32) -> Vec<ValueId> {
        let n = a.len();
        let (q, r) = ((s / self.w) as usize, s % self.w);
        let zero = self.part_const(0);
        let fill = if op == BinOp::AShr {
            let sh = self.part_const(u64::from(self.w - 1));
            self.b.bin(BinOp::AShr, a[n - 1], sh, Flags::NONE)
        } else {
            zero
        };
        let rc = self.part_const(u64::from(r));
        let rinv = self.part_const(u64::from(self.w - r));
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let v = match op {
                BinOp::Shl => {
                    if k < q {
                        zero
                    } else if r == 0 {
                        a[k - q]
                    } else {
                        let hi = self.b.bin(BinOp::Shl, a[k - q], rc, Flags::NONE);
                        if k > q {
                            let lo = self.b.bin(BinOp::LShr, a[k - q - 1], rinv, Flags::NONE);
                            self.b.bin(BinOp::Or, hi, lo, Flags::NONE)
                        } else {
                            hi
                        }
                    }
                }
                _ => {
                    let src = k + q;
                    if src >= n {
                        fill
                    } else if r == 0 {
                        a[src]
                    } else if src == n - 1 {
                        self.b.bin(op, a[src], rc, Flags::NONE)
                    } else {
                        let lo = self.b.bin(BinOp::LShr, a[src], rc, Flags::NONE);
                        let hi = self.b.bin(BinOp::Shl, a[src + 1], rinv, Flags::NONE);
                        self.b.bin(BinOp::Or, lo, hi, Flags::NONE)
                    }
                }
            };
            out.push(v);
        }
        out
    }

    /// Shift parts `a` by the variable amount whose low part is `s`: funnel
    /// every part by `r = s mod W` (two shifts, so no shift reaches `W`), then
    /// move whole parts by `q = s / W` with a `select` ladder.
    fn shift_var(&mut self, op: BinOp, a: &[ValueId], s: ValueId) -> Vec<ValueId> {
        let n = a.len();
        let wm1 = self.part_const(u64::from(self.w - 1));
        let one = self.part_const(1);
        let log_w = self.part_const(u64::from(self.w.trailing_zeros()));
        let r = self.b.bin(BinOp::And, s, wm1, Flags::NONE);
        let rinv = self.b.bin(BinOp::Xor, r, wm1, Flags::NONE); // W - 1 - r
        let q = self.b.bin(BinOp::LShr, s, log_w, Flags::NONE);
        let fill = if op == BinOp::AShr {
            self.b.bin(BinOp::AShr, a[n - 1], wm1, Flags::NONE)
        } else {
            self.part_const(0)
        };
        // u[k]: part k of the value shifted by r.
        let mut u = Vec::with_capacity(n);
        for k in 0..n {
            let v = if op == BinOp::Shl {
                let hi = self.b.bin(BinOp::Shl, a[k], r, Flags::NONE);
                if k == 0 {
                    hi
                } else {
                    let lo1 = self.b.bin(BinOp::LShr, a[k - 1], one, Flags::NONE);
                    let lo = self.b.bin(BinOp::LShr, lo1, rinv, Flags::NONE);
                    self.b.bin(BinOp::Or, hi, lo, Flags::NONE)
                }
            } else if k == n - 1 {
                self.b.bin(op, a[k], r, Flags::NONE)
            } else {
                let lo = self.b.bin(BinOp::LShr, a[k], r, Flags::NONE);
                let hi1 = self.b.bin(BinOp::Shl, a[k + 1], one, Flags::NONE);
                let hi = self.b.bin(BinOp::Shl, hi1, rinv, Flags::NONE);
                self.b.bin(BinOp::Or, lo, hi, Flags::NONE)
            };
            u.push(v);
        }
        let eq: Vec<ValueId> = (0..n)
            .map(|j| {
                let jc = self.part_const(j as u64);
                self.b.icmp(IntPred::Eq, q, jc)
            })
            .collect();
        (0..n)
            .map(|k| {
                let mut acc = fill;
                for (j, &e) in eq.iter().enumerate() {
                    let src = if op == BinOp::Shl { k.checked_sub(j) } else { Some(k + j).filter(|&x| x < n) };
                    if let Some(src) = src {
                        acc = self.b.select(e, u[src], acc);
                    }
                }
                acc
            })
            .collect()
    }

    fn icmp(&mut self, pred: IntPred, l: ValueId, r: ValueId) -> ValueId {
        let (a, c) = (self.parts(l), self.parts(r));
        let n = a.len();
        if matches!(pred, IntPred::Eq | IntPred::Ne) {
            let mut acc = self.b.bin(BinOp::Xor, a[0], c[0], Flags::NONE);
            for k in 1..n {
                let x = self.b.bin(BinOp::Xor, a[k], c[k], Flags::NONE);
                acc = self.b.bin(BinOp::Or, acc, x, Flags::NONE);
            }
            let zero = self.part_const(0);
            return self.b.icmp(pred, acc, zero);
        }
        // Lexicographic from the top part: the top part decides unless equal,
        // then the next, ...; the lowest part applies the full predicate
        // unsigned, the others its strict form (signed for the top part).
        use IntPred::*;
        let (low, mid, top) = match pred {
            Ult => (Ult, Ult, Ult),
            Ule => (Ule, Ult, Ult),
            Ugt => (Ugt, Ugt, Ugt),
            Uge => (Uge, Ugt, Ugt),
            Slt => (Ult, Ult, Slt),
            Sle => (Ule, Ult, Slt),
            Sgt => (Ugt, Ugt, Sgt),
            Sge => (Uge, Ugt, Sgt),
            Eq | Ne => unreachable!(),
        };
        let mut acc = self.b.icmp(low, a[0], c[0]);
        for k in 1..n {
            let p = if k == n - 1 { top } else { mid };
            let strict = self.b.icmp(p, a[k], c[k]);
            let eq = self.b.icmp(Eq, a[k], c[k]);
            acc = self.b.select(eq, acc, strict);
        }
        acc
    }
}

#[cfg(test)]
mod tests;
