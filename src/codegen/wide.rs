//! Integers wider than 64 bits on a 64-bit target, shared by the AArch64 and
//! RISC-V backends (`docs/ir-design.md` §3b).
//!
//! [`prepare`] runs [`legalize_ints`] at `W = 64` first, which splits every
//! operation on an `i128` into operations on 64-bit parts and leaves only the
//! ABI boundary wide. Here a remaining wide value lives in a **register
//! group**: part 0 is the value's own vreg, the higher parts are vregs kept in
//! a side table ([`WideIsel::wide_table`]), created on first reference so a
//! definition and its uses agree whatever order the blocks are lowered in.
//! [`lower_wide`] lowers what the legalizer leaves:
//!
//! | wide operation | lowering |
//! |---|---|
//! | `zext`/`sext` into a wide type | part 0 extended to 64 bits, then zero / sign fill |
//! | `trunc` of a wide value | its low part(s) |
//! | `or`/`and`/`xor`, `select`, `freeze`, `declassify` | per part (`select` branch-free) |
//! | `shl`/`lshr`/`ashr` by a constant multiple of 64 | part moves (the legalizer's join/split shapes) |
//! | `ptrtoint` into / `inttoptr` from a wide type | the pointer is part 0, the rest zero |
//! | `sitofp`/`uitofp`/`fptosi`/`fptoui` | libgcc's `__floattidf`, `__fixunssfti`, … |
//! | volatile `load` / `store` | one 64-bit access per part, lowest address first |
//! | `mul` (the legalizer's [`MUL128`] call) | inline: `mul` for the low product, `umulh`/`mulhu` and two `mul`s |
//! | `udiv`/`sdiv`/`urem`/`srem` | libgcc's `__udivti3`, `__divti3`, `__umodti3`, `__modti3` (ordinary calls) |
//!
//! The ABI (an `i128` argument in a register pair or on the stack, a result
//! in the first two result registers) is each backend's, in its call,
//! prologue and return lowering. A wide `switch`, a wide bitcast to or from a
//! vector, wide atomics and integers wider than 128 bits at the boundary are
//! rejected with a diagnostic.

use std::cell::RefCell;

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::legalize_int::{LegalizeError, LegalizeOptions, legalize_ints, libgcc_libcall};
use crate::codegen::mir::{MachineInst, PReg, RegClass, VReg};
use crate::ir::inst::{BinOp, CastOp, InstKind};
use crate::ir::types::{Type, TypeId};
use crate::ir::value::{Const, ValueDef};
use crate::ir::{InstData, Module, ValueId};
use crate::support::{DetHashMap, StrInterner};

use puremp::Int;

/// The name under which the legalizer calls for a 128-bit `mul` on AArch64
/// and RISC-V: a placeholder that instruction selection expands inline, so no
/// runtime helper is needed for it.
pub const MUL128: &str = "__lf_multi3";

/// The higher parts of the wide values of the function being lowered, by value
/// index (part 0 is the value's own vreg).
pub type WideTable = RefCell<DetHashMap<usize, Vec<VReg>>>;

/// The few machine operations [`lower_wide`] builds a register group's
/// lowering from, all on full 64-bit registers.
pub trait WideIsel: TargetIsel {
    /// The side table of higher parts (reset for every function).
    fn wide_table(&self) -> &WideTable;
    /// `d = a op b` for `op` in `or`, `and`, `xor`, `add`, `mul`.
    fn wide_alu(&self, op: BinOp, d: VReg, a: VReg, b: VReg) -> MachineInst;
    /// `d` = the high 64 bits of the unsigned product `a · b`.
    fn wide_umulh(&self, d: VReg, a: VReg, b: VReg) -> MachineInst;
    /// `d = a >> 63`, arithmetic (every bit a copy of the sign).
    fn wide_sign(&self, d: VReg, a: VReg) -> MachineInst;
    /// An `i1` condition as a register holding exactly 0 or 1.
    fn wide_cond(&self, lo: &mut Lower<'_, Self>, c: ValueId) -> VReg;
    /// `d = c ? t : f` without a branch (`c` from [`WideIsel::wide_cond`]).
    fn wide_select(&self, lo: &mut Lower<'_, Self>, d: VReg, c: VReg, t: VReg, f: VReg);
    /// A 64-bit-or-narrower integer, or a pointer, zero- or sign-extended to
    /// a full register.
    fn wide_extend(&self, lo: &mut Lower<'_, Self>, v: ValueId, signed: bool) -> VReg;
    /// A fresh vreg loaded with the 8 bytes at `ptr + off`.
    fn wide_load(&self, lo: &mut Lower<'_, Self>, ptr: VReg, off: u64) -> VReg;
    /// Store the 8 bytes of `v` at `ptr + off`.
    fn wide_store(&self, lo: &mut Lower<'_, Self>, ptr: VReg, off: u64, v: VReg);
    /// The second result register (`x1`, `a1`): the high half of an `i128`
    /// result.
    fn wide_ret_hi(&self) -> PReg;
    /// The module index of the declared function `name` (a helper the
    /// preparation declared, or [`MUL128`]); by default found through the
    /// symbol names [`Lower`] was given.
    fn wide_helper(&self, lo: &Lower<'_, Self>, name: &str) -> Option<u32> {
        let syms = lo.syms()?;
        lo.module().functions().position(|f| syms.resolve(f.name) == name).map(|i| i as u32)
    }
    /// Call module function `f` with the arguments already in vregs: the
    /// register moves as one run right before the call, then each result
    /// register copied out.
    fn wide_helper_call(&self, lo: &mut Lower<'_, Self>, f: u32, args: &[(PReg, VReg)], rets: &[(PReg, VReg)]);
}

/// The number of 64-bit parts of an integer type wider than 64 bits.
pub fn wide_ty<T: TargetIsel>(lo: &Lower<'_, T>, ty: TypeId) -> Option<usize> {
    match lo.types().get(ty) {
        Type::Int(b) if *b > 64 => Some(b.div_ceil(64) as usize),
        _ => None,
    }
}

/// The number of 64-bit parts of `v` if it is an integer wider than 64 bits.
pub fn wide_val<T: TargetIsel>(lo: &Lower<'_, T>, v: ValueId) -> Option<usize> {
    wide_ty(lo, lo.func().value_type(v))
}

/// Check that a wide value crossing the ABI is an `i128`.
///
/// # Panics
///
/// For a wider integer: it has no register convention.
pub fn check_abi_width<T: TargetIsel>(t: &T, n: usize) {
    if n != 2 {
        panic!("{} backend: integers wider than 128 bits cannot be passed or returned", t.name());
    }
}

/// The 64-bit parts of a wide value, least significant first.
pub fn parts<T: WideIsel>(t: &T, lo: &mut Lower<'_, T>, v: ValueId) -> Vec<VReg> {
    let n = wide_val(lo, v).expect("a wide integer");
    match lo.func().value(v).def.clone() {
        ValueDef::Const(c) => {
            let words: Vec<u64> = match lo.module().consts().get(c) {
                Const::Int { value, .. } => {
                    let bits = value.mod_2k(64 * n as u32);
                    (0..n).map(|k| bits.div_2k_trunc(64 * k as u32).mod_2k(64).to_u64().unwrap_or(0)).collect()
                }
                // Poison may be any value.
                _ => vec![0; n],
            };
            words.into_iter().map(|w| movi(t, lo, w)).collect()
        }
        ValueDef::Inst(_) | ValueDef::Param(..) => {
            let p0 = lo.reg(v);
            let known = t.wide_table().borrow().get(&v.index()).cloned();
            let hi = match known {
                Some(h) => h,
                None => {
                    let h: Vec<VReg> = (1..n).map(|_| lo.fresh_vreg(RegClass::Gpr)).collect();
                    t.wide_table().borrow_mut().insert(v.index(), h.clone());
                    h
                }
            };
            std::iter::once(p0).chain(hi).collect()
        }
        ValueDef::Global(_) | ValueDef::Func(_) => unreachable!("an address is never wide"),
    }
}

/// Copy `src` into the parts of the wide value `res`.
pub fn set_parts<T: WideIsel>(t: &T, lo: &mut Lower<'_, T>, res: ValueId, src: &[VReg]) {
    let dst = parts(t, lo, res);
    for (&d, &s) in dst.iter().zip(src) {
        lo.emit(t.emit_move(crate::codegen::mir::Reg::Virtual(d), crate::codegen::mir::Reg::Virtual(s)));
    }
}

/// A fresh vreg holding `v`.
fn movi<T: WideIsel>(t: &T, lo: &mut Lower<'_, T>, v: u64) -> VReg {
    let d = lo.fresh_vreg(RegClass::Gpr);
    lo.emit(t.li(d, Int::from_u64(v)));
    d
}

/// Lower `inst` if it involves an integer wider than 64 bits (see the module
/// docs). `false` when it does not, or when the ordinary lowering already
/// handles it (a call, or an operation that only reads part 0).
pub fn lower_wide<T: WideIsel>(t: &T, lo: &mut Lower<'_, T>, inst: &InstData) -> bool {
    let ops = inst.operands().to_vec();
    let res = inst.result();
    let res_n = res.and_then(|r| wide_val(lo, r));
    if res_n.is_none() && !ops.iter().any(|&o| wide_val(lo, o).is_some()) {
        return false;
    }
    let name = t.name().to_owned();
    match (&inst.kind, res_n) {
        (InstKind::Call, _) => return false,
        (InstKind::PtrAdd { .. } | InstKind::DynAlloca { .. }, None) => return false,
        (InstKind::Syscall, None) => return false,
        (InstKind::Cast(op @ (CastOp::ZExt | CastOp::SExt)), Some(n)) => {
            let signed = *op == CastOp::SExt;
            let mut p =
                if wide_val(lo, ops[0]).is_some() { parts(t, lo, ops[0]) } else { vec![t.wide_extend(lo, ops[0], signed)] };
            let last = *p.last().expect("a part");
            let fill = if signed {
                let f = lo.fresh_vreg(RegClass::Gpr);
                lo.emit(t.wide_sign(f, last));
                f
            } else {
                movi(t, lo, 0)
            };
            p.resize(n, fill);
            set_parts(t, lo, res.expect("a result"), &p);
        }
        (InstKind::Cast(CastOp::Trunc | CastOp::IntToPtr), _) => {
            let p = parts(t, lo, ops[0]);
            match res_n {
                Some(n) => set_parts(t, lo, res.expect("a result"), &p[..n]),
                None => {
                    let d = lo.result_reg(inst);
                    lo.emit(t.emit_move(crate::codegen::mir::Reg::Virtual(d), crate::codegen::mir::Reg::Virtual(p[0])));
                }
            }
        }
        (InstKind::Cast(CastOp::PtrToInt), Some(n)) => {
            let mut p = vec![t.wide_extend(lo, ops[0], false)];
            let z = movi(t, lo, 0);
            p.resize(n, z);
            set_parts(t, lo, res.expect("a result"), &p);
        }
        (InstKind::Cast(CastOp::Bitcast), Some(_)) if wide_val(lo, ops[0]).is_some() => {
            let p = parts(t, lo, ops[0]);
            set_parts(t, lo, res.expect("a result"), &p);
        }
        (InstKind::Cast(op @ (CastOp::SiToFp | CastOp::UiToFp | CastOp::FpToSi | CastOp::FpToUi)), _) => {
            lower_float_cast(t, lo, *op, inst);
        }
        (InstKind::Bin(op @ (BinOp::Or | BinOp::And | BinOp::Xor)), Some(_)) => {
            let (a, b) = (parts(t, lo, ops[0]), parts(t, lo, ops[1]));
            let p: Vec<VReg> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| {
                    let d = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(t.wide_alu(*op, d, x, y));
                    d
                })
                .collect();
            set_parts(t, lo, res.expect("a result"), &p);
        }
        (InstKind::Bin(op @ (BinOp::Shl | BinOp::LShr | BinOp::AShr)), Some(n)) => {
            let k = const_u64(lo, ops[1])
                .filter(|s| s % 64 == 0)
                .unwrap_or_else(|| panic!("{name} backend: a wide {op:?} by a non-multiple of 64 was not legalized"));
            let k = (k / 64) as usize;
            let src = parts(t, lo, ops[0]);
            let fill = match op {
                BinOp::AShr => {
                    let f = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(t.wide_sign(f, src[n - 1]));
                    f
                }
                _ => movi(t, lo, 0),
            };
            let p: Vec<VReg> = (0..n)
                .map(|i| match op {
                    BinOp::Shl if i >= k => src[i - k],
                    BinOp::Shl => fill,
                    _ if i + k < n => src[i + k],
                    _ => fill,
                })
                .collect();
            set_parts(t, lo, res.expect("a result"), &p);
        }
        (InstKind::Select, Some(_)) => {
            let c = t.wide_cond(lo, ops[0]);
            let (tv, fv) = (parts(t, lo, ops[1]), parts(t, lo, ops[2]));
            let p: Vec<VReg> = tv
                .iter()
                .zip(&fv)
                .map(|(&a, &b)| {
                    let d = lo.fresh_vreg(RegClass::Gpr);
                    t.wide_select(lo, d, c, a, b);
                    d
                })
                .collect();
            set_parts(t, lo, res.expect("a result"), &p);
        }
        (InstKind::Freeze | InstKind::Declassify, Some(_)) => {
            let p = parts(t, lo, ops[0]);
            set_parts(t, lo, res.expect("a result"), &p);
        }
        (InstKind::Load { .. }, Some(n)) => {
            let ptr = lo.reg(ops[0]);
            let p: Vec<VReg> = (0..n).map(|k| t.wide_load(lo, ptr, 8 * k as u64)).collect();
            set_parts(t, lo, res.expect("a result"), &p);
        }
        (InstKind::Store { .. }, None) => {
            let ptr = lo.reg(ops[0]);
            let p = parts(t, lo, ops[1]);
            for (k, &v) in p.iter().enumerate() {
                t.wide_store(lo, ptr, 8 * k as u64, v);
            }
        }
        (
            InstKind::AtomicLoad { .. } | InstKind::AtomicStore { .. } | InstKind::AtomicRmw { .. } | InstKind::CmpXchg { .. },
            _,
        ) => panic!("{name} backend: atomic operations on integers wider than 64 bits are not supported"),
        (InstKind::Cast(CastOp::Bitcast), _) => {
            panic!("{name} backend: a bitcast between a vector and an integer wider than 64 bits is not supported")
        }
        (other, _) => panic!("{name} backend: a wide {other:?} reached isel (not legalized)"),
    }
    true
}

/// The constant value of `v`, if it is a small integer constant.
fn const_u64<T: TargetIsel>(lo: &Lower<'_, T>, v: ValueId) -> Option<u64> {
    let ValueDef::Const(c) = lo.func().value(v).def else { return None };
    match lo.module().consts().get(c) {
        Const::Int { value, .. } => value.to_u64(),
        _ => None,
    }
}

/// The libgcc helper converting between a 128-bit integer and a float, by
/// cast and float width.
pub fn float_helper(op: CastOp, float_bits: u32) -> Option<&'static str> {
    Some(match (op, float_bits) {
        (CastOp::SiToFp, 64) => "__floattidf",
        (CastOp::SiToFp, 32) => "__floattisf",
        (CastOp::UiToFp, 64) => "__floatuntidf",
        (CastOp::UiToFp, 32) => "__floatuntisf",
        (CastOp::FpToSi, 64) => "__fixdfti",
        (CastOp::FpToSi, 32) => "__fixsfti",
        (CastOp::FpToUi, 64) => "__fixunsdfti",
        (CastOp::FpToUi, 32) => "__fixunssfti",
        _ => return None,
    })
}

/// A conversion between a 128-bit integer and a float: a call to the libgcc
/// helper [`prepare`] declared.
fn lower_float_cast<T: WideIsel>(t: &T, lo: &mut Lower<'_, T>, op: CastOp, inst: &InstData) {
    let name = t.name().to_owned();
    let src = inst.operands()[0];
    let res = inst.result().expect("a conversion has a result");
    let to_float = matches!(op, CastOp::SiToFp | CastOp::UiToFp);
    let (wide, fty) = if to_float { (src, inst.ty) } else { (res, lo.func().value_type(src)) };
    assert_eq!(wide_val(lo, wide), Some(2), "{name} backend: only 128-bit integers convert to or from floats");
    let fbits = lo.types().bit_width(fty).unwrap_or(64);
    let helper = float_helper(op, fbits).unwrap_or_else(|| panic!("{name} backend: no {op:?} helper for f{fbits}"));
    let f = helper_index(t, lo, helper);
    let cc = t.call_conv().clone();
    if to_float {
        let p = parts(t, lo, src);
        let d = lo.result_reg(inst);
        t.wide_helper_call(lo, f, &[(cc.arg_regs[0], p[0]), (cc.arg_regs[1], p[1])], &[(cc.fp_ret_reg, d)]);
    } else {
        let s = lo.reg(src);
        let p = parts(t, lo, res);
        t.wide_helper_call(lo, f, &[(cc.fp_arg_regs[0], s)], &[(cc.ret_reg, p[0]), (t.wide_ret_hi(), p[1])]);
    }
}

/// The module index of the function named `name` (a helper the module
/// declares).
fn helper_index<T: WideIsel>(t: &T, lo: &Lower<'_, T>, name: &str) -> u32 {
    t.wide_helper(lo, name).unwrap_or_else(|| {
        panic!("{} backend: {name} is not declared (compile through the wide preparation)", t.name())
    })
}

/// Whether `callee` is the legalizer's [`MUL128`] placeholder.
pub fn is_mul128<T: WideIsel>(t: &T, lo: &Lower<'_, T>, callee: ValueId) -> bool {
    lo.callee_func(callee).is_some_and(|f| t.wide_helper(lo, MUL128) == Some(f))
}

/// The names of every helper [`prepare`] may declare.
pub fn helper_names() -> impl Iterator<Item = &'static str> {
    [MUL128].into_iter().chain(
        [CastOp::SiToFp, CastOp::UiToFp, CastOp::FpToSi, CastOp::FpToUi]
            .into_iter()
            .flat_map(|op| [32, 64].map(|w| float_helper(op, w).expect("a helper"))),
    )
}

/// The inline 128-bit multiply the legalizer's [`MUL128`] call stands for:
/// `a·b mod 2^128 = lo(a0·b0) + 2^64·(hi(a0·b0) + a0·b1 + a1·b0)`.
pub fn lower_mul128<T: WideIsel>(t: &T, lo: &mut Lower<'_, T>, inst: &InstData) {
    let ops = inst.operands();
    let res = inst.result().expect("a multiply has a result");
    let (a, b) = (parts(t, lo, ops[1]), parts(t, lo, ops[2]));
    assert!(a.len() == 2 && b.len() == 2, "{} backend: the inline multiply is 128-bit", t.name());
    let fresh = |lo: &mut Lower<'_, T>| lo.fresh_vreg(RegClass::Gpr);
    let l = fresh(lo);
    lo.emit(t.wide_alu(BinOp::Mul, l, a[0], b[0]));
    let h = fresh(lo);
    lo.emit(t.wide_umulh(h, a[0], b[0]));
    let x = fresh(lo);
    lo.emit(t.wide_alu(BinOp::Mul, x, a[0], b[1]));
    let y = fresh(lo);
    lo.emit(t.wide_alu(BinOp::Mul, y, a[1], b[0]));
    let h1 = fresh(lo);
    lo.emit(t.wide_alu(BinOp::Add, h1, h, x));
    let h2 = fresh(lo);
    lo.emit(t.wide_alu(BinOp::Add, h2, h1, y));
    set_parts(t, lo, res, &[l, h2]);
}

/// Whether some function of `module` computes with an integer wider than 64
/// bits.
pub fn has_wide_ints(module: &Module) -> bool {
    module.functions().any(|f| {
        (0..f.value_count())
            .any(|v| matches!(module.types().get(f.value_type(ValueId::from_index(v))), Type::Int(b) if *b > 64))
    })
}

/// The libcall names of the 64-bit targets: [`MUL128`] for a 128-bit `mul`
/// (expanded inline), libgcc's otherwise.
pub fn libcall(op: BinOp, bits: u32) -> String {
    match (op, bits) {
        (BinOp::Mul, 128) => MUL128.into(),
        _ => libgcc_libcall(op, bits),
    }
}

/// Prepare a copy of `module` for instruction selection of integers wider
/// than 64 bits on a 64-bit target: declare the libgcc helpers converting
/// between `i128` and floats that the module needs, then split every wide
/// integer into 64-bit parts ([`legalize_ints`] at `W = 64`, with
/// [`libcall`] names), leaving only the ABI boundary wide. The [`MUL128`]
/// placeholder takes secrets and gives a secret (§6d): it is expanded inline
/// into straight-line code.
///
/// # Errors
///
/// A [`LegalizeError`] for a width the split cannot handle.
pub fn prepare(module: &Module, syms: &StrInterner) -> Result<(Module, StrInterner), LegalizeError> {
    let bytes = crate::ir::binary::encode(module, syms);
    let mut names = StrInterner::new();
    let mut m = crate::ir::binary::decode(&bytes, &mut names).expect("a module round-trips through .lfb");
    declare_float_helpers(&mut m, &mut names);
    legalize_ints(&mut m, &mut names, &LegalizeOptions { part_bits: 64, libcall_name: libcall })?;
    let pseudo = m.functions().position(|f| f.is_declaration() && names.resolve(f.name) == MUL128);
    if let Some(i) = pseudo {
        let f = crate::ir::FuncId::from_index(i);
        let mut attrs = m.function(f).attrs.clone();
        attrs.set_param_secret(0, true);
        attrs.set_param_secret(1, true);
        attrs.secret_ret = true;
        m.set_func_attrs(f, attrs);
    }
    Ok((m, names))
}

/// The prepared copy of `module` when it has integers wider than 64 bits.
///
/// # Panics
///
/// When [`prepare`] fails: the backend cannot compile such a width.
pub fn prepared_if_wide(module: &Module, syms: &StrInterner, backend: &str) -> Option<(Module, StrInterner)> {
    has_wide_ints(module)
        .then(|| prepare(module, syms).unwrap_or_else(|e| panic!("{backend} backend: wide-integer legalization: {e}")))
}

/// Declare `(i128) -> float` / `(float) -> i128` helpers for every wide
/// integer/float conversion in `module`.
fn declare_float_helpers(module: &mut Module, names: &mut StrInterner) {
    let mut need: Vec<(&'static str, TypeId, TypeId)> = Vec::new();
    for f in module.functions() {
        for (_, b) in f.blocks() {
            for &i in b.insts() {
                let inst = f.inst(i);
                let InstKind::Cast(op) = inst.kind else { continue };
                let src = f.value_type(inst.operands()[0]);
                let (int_ty, float_ty) = match op {
                    CastOp::SiToFp | CastOp::UiToFp => (src, inst.ty),
                    CastOp::FpToSi | CastOp::FpToUi => (inst.ty, src),
                    _ => continue,
                };
                if !matches!(module.types().get(int_ty), Type::Int(b) if *b > 64) {
                    continue;
                }
                let fbits = module.types().bit_width(float_ty).unwrap_or(64);
                let Some(name) = float_helper(op, fbits) else { continue };
                let sig = if matches!(op, CastOp::SiToFp | CastOp::UiToFp) { (int_ty, float_ty) } else { (float_ty, int_ty) };
                if !need.iter().any(|n| n.0 == name) {
                    need.push((name, sig.0, sig.1));
                }
            }
        }
    }
    for (name, param, ret) in need {
        if module.functions().any(|f| names.resolve(f.name) == name) {
            continue;
        }
        let sig = module.types_mut().func(vec![param], ret, false);
        module.declare_function(names.intern(name), sig);
    }
}
