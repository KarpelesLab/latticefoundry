//! Integers wider than 64 bits on x86-64 (`docs/ir-design.md` §3b).
//!
//! The backend computes with `i128` (and wider integers) the way the 32-bit
//! targets compute with `i64`: [`super::super::prepare_module`] runs
//! [`crate::codegen::legalize_int`] at `W = 64` first, which splits every
//! operation into 64-bit parts — `add`/`sub` with carry chains, shifts with
//! funnel shifts and a `select` ladder, compares lexicographically, all
//! branch-free — and leaves only the ABI boundary wide. Here a remaining wide
//! value lives in a **register group**: part 0 is the value's own vreg, the
//! higher parts are vregs kept in a side table (`X86_64Target::wide`), created
//! on first reference so a definition and its uses agree whatever order the
//! blocks are lowered in. A wide constant is materialized part by part, and
//! [`TargetIsel::li`](crate::codegen::isel::TargetIsel::li) keeps the low 64
//! bits of one, so an operation that reads only part 0 through `Lower::reg` (a
//! `ptr_add` offset, a `dyn_alloca` size, a `syscall` argument) sees exactly
//! that.
//!
//! What reaches isel, and how it lowers:
//!
//! | wide operation | lowering |
//! |---|---|
//! | `zext`/`sext` into a wide type | part 0 extended to 64 bits, then zero / sign fill (`sar 63`) |
//! | `trunc` of a wide value | its low part(s) |
//! | `or`/`and`/`xor`, `select`, `freeze`, `declassify` | per part (`select` as `cmov` per part) |
//! | `shl`/`lshr`/`ashr` by a constant multiple of 64 | part moves (the legalizer's join/split shapes) |
//! | `ptrtoint` into / `inttoptr` from a wide type | the pointer is part 0, the rest zero |
//! | `bitcast` to / from a 128-bit vector | through a 16-byte stack slot |
//! | `sitofp`/`uitofp`/`fptosi`/`fptoui` | libgcc's `__floattidf`, `__floatuntisf`, `__fixdfti`, `__fixunssfti`, … |
//! | volatile `load`/`store` | one 64-bit access per part, lowest address first |
//! | `mul` (the legalizer's [`MUL128_PSEUDO`] libcall) | inline: `mul` for the low product, two `imul`s for the cross terms |
//! | `udiv`/`sdiv`/`urem`/`srem` | libgcc's `__udivti3`, `__divti3`, `__umodti3`, `__modti3` |
//! | `switch` | [`X86Op::Switch128`]: both halves compared per case |
//!
//! **System V ABI** (matching gcc's `__int128`): an `i128` argument takes the
//! next two integer registers, low half first (`rdi:rsi`, …), and when fewer
//! than two remain the whole value goes on the stack in a 16-byte-aligned
//! 16-byte slot — later integer arguments still take the registers left. An
//! `i128` result comes back in `rax:rdx`. Wider integers have no register
//! convention (the psABI passes `_BitInt` above 128 bits in memory) and are
//! rejected at the boundary, as are wide atomics and the Microsoft x64
//! convention.

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::mir::{MachineInst, MachineOperand, PReg, RegClass, VReg};
use crate::ir::inst::{BinOp, CastOp, InstKind};
use crate::ir::types::{Type, TypeId};
use crate::ir::value::{Const, ValueDef};
use crate::ir::{InstData, ValueId};

use super::super::regs;
use super::{X86Op, X86_64Target, def, def_v, imm, use_p, use_v};

/// The name under which the legalizer calls for a 128-bit `mul` on x86-64: a
/// placeholder that instruction selection expands inline (see the module docs),
/// so no runtime helper is needed for it.
pub const MUL128_PSEUDO: &str = "__lf_x86_64_multi3";

/// The libgcc helper converting between a 128-bit integer and a float, by
/// cast and float width: `(name, float bits)`.
pub(crate) fn float_helper(op: CastOp, float_bits: u32) -> Option<&'static str> {
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

impl X86_64Target {
    /// The number of 64-bit parts of an integer type wider than 64 bits.
    pub(super) fn wide_ty(lo: &Lower<'_, Self>, ty: TypeId) -> Option<usize> {
        match lo.types().get(ty) {
            Type::Int(b) if *b > 64 => Some(b.div_ceil(64) as usize),
            _ => None,
        }
    }

    /// The number of 64-bit parts of `v` if it is an integer wider than 64 bits.
    pub(super) fn wide_val(lo: &Lower<'_, Self>, v: ValueId) -> Option<usize> {
        Self::wide_ty(lo, lo.func().value_type(v))
    }

    /// The 64-bit parts of a wide value, least significant first (see the
    /// module docs).
    pub(super) fn parts(&self, lo: &mut Lower<'_, Self>, v: ValueId) -> Vec<VReg> {
        let n = Self::wide_val(lo, v).expect("a wide integer");
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
                words
                    .into_iter()
                    .map(|w| {
                        let d = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(d), imm(w)]));
                        d
                    })
                    .collect()
            }
            ValueDef::Inst(_) | ValueDef::Param(..) => {
                let p0 = lo.reg(v);
                let known = self.wide.borrow().get(&v.index()).cloned();
                let hi = match known {
                    Some(h) => h,
                    None => {
                        let h: Vec<VReg> = (1..n).map(|_| lo.fresh_vreg(RegClass::Gpr)).collect();
                        self.wide.borrow_mut().insert(v.index(), h.clone());
                        h
                    }
                };
                std::iter::once(p0).chain(hi).collect()
            }
            ValueDef::Global(_) | ValueDef::Func(_) => unreachable!("an address is never wide"),
        }
    }

    /// Copy `src` into the parts of the wide result `res`.
    fn set_parts(&self, lo: &mut Lower<'_, Self>, res: ValueId, src: &[VReg]) {
        let dst = self.parts(lo, res);
        for (&d, &s) in dst.iter().zip(src) {
            mov(lo, d, s);
        }
    }

    /// A fresh vreg holding `v`.
    fn movi(lo: &mut Lower<'_, Self>, v: u64) -> VReg {
        let d = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::MovRI.opcode(), vec![def_v(d), imm(v)]));
        d
    }

    /// `v` (a 64-bit-or-narrower integer or a pointer) as a full 64-bit
    /// register, zero- or sign-extended from its width.
    fn widen64(&self, lo: &mut Lower<'_, Self>, v: ValueId, signed: bool) -> VReg {
        let r = self.oper(lo, v);
        let w = lo.int_width(v);
        if w >= 64 {
            return r;
        }
        let d = lo.fresh_vreg(RegClass::Gpr);
        let op = if signed { X86Op::Movsx } else { X86Op::Movzx };
        lo.emit(MachineInst::new(op.opcode(), vec![def_v(d), use_v(r), imm(u64::from(w)), imm(64)]));
        d
    }

    /// Lower `inst` if it involves an integer wider than 64 bits (see the
    /// module docs). `false` when it does not, or when the ordinary lowering
    /// already handles it (a call, or an operation that only reads part 0).
    pub(super) fn lower_wide(&self, lo: &mut Lower<'_, Self>, inst: &InstData) -> bool {
        let ops = inst.operands().to_vec();
        let res = inst.result();
        let res_n = res.and_then(|r| Self::wide_val(lo, r));
        if res_n.is_none() && !ops.iter().any(|&o| Self::wide_val(lo, o).is_some()) {
            return false;
        }
        let kind = inst.kind.clone();
        match (&kind, res_n) {
            (InstKind::Call, _) => return false,
            (InstKind::PtrAdd { .. } | InstKind::DynAlloca { .. }, None) => return false,
            (InstKind::Syscall, None) => return false,
            (InstKind::Cast(op @ (CastOp::ZExt | CastOp::SExt)), Some(n)) => {
                let signed = *op == CastOp::SExt;
                let mut p = if Self::wide_val(lo, ops[0]).is_some() {
                    self.parts(lo, ops[0])
                } else {
                    vec![self.widen64(lo, ops[0], signed)]
                };
                let last = *p.last().expect("a part");
                let fill = if signed {
                    let f = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(X86Op::SarI.opcode(), vec![def_v(f), use_v(last), imm(63), imm(64)]));
                    f
                } else {
                    Self::movi(lo, 0)
                };
                p.resize(n, fill);
                self.set_parts(lo, res.unwrap(), &p);
            }
            (InstKind::Cast(CastOp::Trunc | CastOp::IntToPtr), _) => {
                let p = self.parts(lo, ops[0]);
                match res_n {
                    Some(n) => self.set_parts(lo, res.unwrap(), &p[..n]),
                    None => {
                        let d = lo.result_reg(inst);
                        mov(lo, d, p[0]);
                    }
                }
            }
            (InstKind::Cast(CastOp::PtrToInt), Some(n)) => {
                let mut p = vec![self.widen64(lo, ops[0], false)];
                let z = Self::movi(lo, 0);
                p.resize(n, z);
                self.set_parts(lo, res.unwrap(), &p);
            }
            (InstKind::Cast(CastOp::Bitcast), _) => self.lower_wide_bitcast(lo, inst),
            (InstKind::Cast(op @ (CastOp::SiToFp | CastOp::UiToFp | CastOp::FpToSi | CastOp::FpToUi)), _) => {
                self.lower_wide_float_cast(lo, *op, inst);
            }
            (InstKind::Bin(op @ (BinOp::Or | BinOp::And | BinOp::Xor)), Some(_)) => {
                let (a, b) = (self.parts(lo, ops[0]), self.parts(lo, ops[1]));
                let xop = match op {
                    BinOp::Or => X86Op::Or,
                    BinOp::And => X86Op::And,
                    _ => X86Op::Xor,
                };
                let p: Vec<VReg> = a
                    .iter()
                    .zip(&b)
                    .map(|(&x, &y)| {
                        let d = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(xop.opcode(), vec![def_v(d), use_v(x), use_v(y), imm(64)]));
                        d
                    })
                    .collect();
                self.set_parts(lo, res.unwrap(), &p);
            }
            (InstKind::Bin(op @ (BinOp::Shl | BinOp::LShr | BinOp::AShr)), Some(n)) => {
                let k = Self::const_of(lo, ops[1])
                    .and_then(|c| c.to_u64())
                    .filter(|s| s % 64 == 0)
                    .unwrap_or_else(|| panic!("x86-64 backend: a wide {op:?} by a non-multiple of 64 was not legalized"));
                let k = (k / 64) as usize;
                let src = self.parts(lo, ops[0]);
                let fill = match op {
                    BinOp::AShr => {
                        let f = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(
                            X86Op::SarI.opcode(),
                            vec![def_v(f), use_v(src[n - 1]), imm(63), imm(64)],
                        ));
                        f
                    }
                    _ => Self::movi(lo, 0),
                };
                let p: Vec<VReg> = (0..n)
                    .map(|i| match op {
                        BinOp::Shl if i >= k => src[i - k],
                        BinOp::Shl => fill,
                        _ if i + k < n => src[i + k],
                        _ => fill,
                    })
                    .collect();
                self.set_parts(lo, res.unwrap(), &p);
            }
            (InstKind::Select, Some(_)) => {
                let c = self.clean_cond(lo, ops[0]);
                let (t, f) = (self.parts(lo, ops[1]), self.parts(lo, ops[2]));
                let p: Vec<VReg> = t
                    .iter()
                    .zip(&f)
                    .map(|(&tv, &fv)| {
                        let d = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(fv)]));
                        lo.emit(MachineInst::new(X86Op::Test.opcode(), vec![use_v(c)]));
                        lo.emit(MachineInst::new(X86Op::Cmovne.opcode(), vec![def_v(d), use_v(d), use_v(tv)]));
                        d
                    })
                    .collect();
                self.set_parts(lo, res.unwrap(), &p);
            }
            (InstKind::Freeze | InstKind::Declassify, Some(_)) => {
                let p = self.parts(lo, ops[0]);
                self.set_parts(lo, res.unwrap(), &p);
            }
            (InstKind::Load { .. }, Some(n)) => {
                let ptr = self.oper(lo, ops[0]);
                let p: Vec<VReg> = (0..n)
                    .map(|k| {
                        let a = self.add_off(lo, ptr, 8 * k as u64);
                        let d = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(d), use_v(a), imm(8)]));
                        d
                    })
                    .collect();
                self.set_parts(lo, res.unwrap(), &p);
            }
            (InstKind::Store { .. }, None) => {
                let ptr = self.oper(lo, ops[0]);
                let p = self.parts(lo, ops[1]);
                for (k, &v) in p.iter().enumerate() {
                    let a = self.add_off(lo, ptr, 8 * k as u64);
                    lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(a), use_v(v), imm(8)]));
                }
            }
            (
                InstKind::AtomicLoad { .. }
                | InstKind::AtomicStore { .. }
                | InstKind::AtomicRmw { .. }
                | InstKind::CmpXchg { .. },
                _,
            ) => panic!("x86-64 backend: atomic operations on integers wider than 64 bits are not supported"),
            (other, _) => panic!("x86-64 backend: a wide {other:?} reached isel (not legalized)"),
        }
        true
    }

    /// A `bitcast` with a wide side: to or from a 128-bit vector through a
    /// 16-byte stack slot, or between two wide integers part by part.
    fn lower_wide_bitcast(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let src = inst.operands()[0];
        let res = inst.result().expect("a bitcast has a result");
        match (Self::wide_val(lo, src), Self::wide_val(lo, res)) {
            (Some(_), Some(_)) => {
                let p = self.parts(lo, src);
                self.set_parts(lo, res, &p);
            }
            (Some(_), None) => {
                assert!(lo.types().is_vector(inst.ty), "x86-64 backend: a wide bitcast to a non-vector type");
                let p = self.parts(lo, src);
                let slot = lo.new_slot(16, 16);
                let base = lo.fresh_vreg(RegClass::Gpr);
                lo.emit(self.frame_addr(base, slot));
                for (k, &v) in p.iter().enumerate().take(2) {
                    let a = self.add_off(lo, base, 8 * k as u64);
                    lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(a), use_v(v), imm(8)]));
                }
                let d = lo.result_reg(inst);
                lo.emit(MachineInst::new(X86Op::VLoad.opcode(), vec![def_v(d), use_v(base), imm(1)]));
            }
            (None, Some(n)) => {
                let s = self.oper(lo, src);
                assert!(
                    lo.mf().vreg_class(s) == RegClass::Fp && n == 2,
                    "x86-64 backend: a wide bitcast from a non-vector type"
                );
                let slot = lo.new_slot(16, 16);
                let base = lo.fresh_vreg(RegClass::Gpr);
                lo.emit(self.frame_addr(base, slot));
                lo.emit(MachineInst::new(X86Op::VStore.opcode(), vec![use_v(base), use_v(s), imm(1)]));
                let p: Vec<VReg> = (0..2)
                    .map(|k| {
                        let a = self.add_off(lo, base, 8 * k as u64);
                        let d = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(d), use_v(a), imm(8)]));
                        d
                    })
                    .collect();
                self.set_parts(lo, res, &p);
            }
            (None, None) => unreachable!("not a wide bitcast"),
        }
    }

    /// A conversion between a 128-bit integer and a float: a call to the libgcc
    /// helper [`super::super::prepare_module`] declared.
    fn lower_wide_float_cast(&self, lo: &mut Lower<'_, Self>, op: CastOp, inst: &InstData) {
        let src = inst.operands()[0];
        let res = inst.result().expect("a conversion has a result");
        let to_float = matches!(op, CastOp::SiToFp | CastOp::UiToFp);
        let (wide, fty) = if to_float { (src, inst.ty) } else { (res, lo.func().value_type(src)) };
        assert_eq!(Self::wide_val(lo, wide), Some(2), "x86-64 backend: only 128-bit integers convert to or from floats");
        let fbits = lo.types().bit_width(fty).unwrap_or(64);
        let name = float_helper(op, fbits).unwrap_or_else(|| panic!("x86-64 backend: no {op:?} helper for f{fbits}"));
        let f = self.helper(lo, name);
        let cc = &self.rf.cc;
        if to_float {
            let p = self.parts(lo, src);
            let d = lo.result_reg(inst);
            self.helper_call(lo, f, &[(cc.arg_regs[0], p[0]), (cc.arg_regs[1], p[1])], &[(cc.fp_ret_reg, d)]);
        } else {
            let s = self.oper(lo, src);
            let p = self.parts(lo, res);
            let rdx = regs::gpr(regs::RDX);
            self.helper_call(lo, f, &[(cc.fp_arg_regs[0], s)], &[(cc.ret_reg, p[0]), (rdx, p[1])]);
        }
    }

    /// The module index of the function named `name` (a helper the module
    /// declares).
    pub(super) fn helper(&self, lo: &Lower<'_, Self>, name: &str) -> u32 {
        let syms = lo.syms().unwrap_or_else(|| panic!("x86-64 backend: calling {name} needs the symbol names"));
        lo.module()
            .functions()
            .position(|f| syms.resolve(f.name) == name)
            .unwrap_or_else(|| panic!("x86-64 backend: {name} is not declared (compile through prepare_module)"))
            as u32
    }

    /// A call to module function `f` with arguments already in vregs: the
    /// register moves as one run right before the `call`, then each result
    /// register copied out.
    pub(super) fn helper_call(&self, lo: &mut Lower<'_, Self>, f: u32, args: &[(PReg, VReg)], rets: &[(PReg, VReg)]) {
        for &(r, v) in args {
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(r), use_v(v)]));
        }
        let first = rets.first().map(|&(r, _)| r).unwrap_or(self.rf.cc.ret_reg);
        let mut operands = vec![MachineOperand::Func(f), def(first)];
        for &cs in &self.rf.caller_saved {
            if cs != first {
                operands.push(def(cs));
            }
        }
        for &(r, _) in args {
            operands.push(use_p(r));
        }
        lo.emit(MachineInst::new(X86Op::Call.opcode(), operands));
        for &(r, v) in rets {
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(v), use_p(r)]));
        }
    }

    /// The inline 128-bit multiply the legalizer's [`MUL128_PSEUDO`] call
    /// stands for: `a·b mod 2^128 = lo(a0·b0) + 2^64·(hi(a0·b0) + a0·b1 +
    /// a1·b0)`, the full `a0·b0` product from one `mul` (`rdx:rax`).
    pub(super) fn lower_mul128(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let ops = inst.operands();
        let res = inst.result().expect("a multiply has a result");
        let (a, b) = (self.parts(lo, ops[1]), self.parts(lo, ops[2]));
        assert!(a.len() == 2 && b.len() == 2, "x86-64 backend: the inline multiply is 128-bit");
        let rax = regs::gpr(regs::RAX);
        let rdx = regs::gpr(regs::RDX);
        let x = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::Imul.opcode(), vec![def_v(x), use_v(a[0]), use_v(b[1]), imm(64)]));
        let y = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::Imul.opcode(), vec![def_v(y), use_v(a[1]), use_v(b[0]), imm(64)]));
        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(rax), use_v(a[0])]));
        lo.emit(MachineInst::new(X86Op::MulWide.opcode(), vec![def(rax), def(rdx), use_p(rax), use_v(b[0])]));
        let l = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(l), use_p(rax)]));
        let h = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(h), use_p(rdx)]));
        let h1 = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::Add.opcode(), vec![def_v(h1), use_v(h), use_v(x), imm(64)]));
        let h2 = lo.fresh_vreg(RegClass::Gpr);
        lo.emit(MachineInst::new(X86Op::Add.opcode(), vec![def_v(h2), use_v(h1), use_v(y), imm(64)]));
        self.set_parts(lo, res, &[l, h2]);
    }
}

/// `d = s` (64-bit register copy).
fn mov(lo: &mut Lower<'_, X86_64Target>, d: VReg, s: VReg) {
    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(s)]));
}
