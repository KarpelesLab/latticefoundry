//! The Microsoft x64 calling convention ("Win64"): call, prologue and return
//! lowering for Windows targets, implemented from Microsoft's published x64
//! ABI documentation (tenet T1).
//!
//! What differs from System V:
//!
//! - **One positional counter.** Argument *k* (counting a hidden return
//!   pointer as argument 0) goes in `rcx, rdx, r8, r9` for *k* < 4 if it is an
//!   integer, pointer or small aggregate, or in `xmm0..xmm3` if it is a float;
//!   from *k* = 4 on it goes in the stack slot at `[rsp + 8k]` at the call.
//! - **Shadow space.** The caller always reserves 32 bytes above the return
//!   address (the "home" slots of the four register arguments), so the first
//!   stack argument is at `[rsp + 32]` and a callee finds argument *k* at
//!   `[rbp + 16 + 8k]`. The outgoing area is at least 32 bytes for any call.
//! - **Aggregates.** A struct/array of 1, 2, 4 or 8 bytes is passed and
//!   returned by value, as an integer, in a GPR or stack slot. Any other size
//!   is passed by reference: the caller copies it into a 16-byte-aligned
//!   temporary and passes its address. A return of any other size uses a
//!   hidden pointer in `rcx`, which the callee also returns in `rax`.
//! - **Vectors** (`__m128`-class, `docs/ir-design.md` §6e) are passed by
//!   reference like a large aggregate — the caller stores the vector into a
//!   16-byte-aligned temporary and passes its address in the argument's GPR or
//!   stack slot — and returned by value in `xmm0`.
//! - **Variadic calls.** A float argument among the first four is passed in
//!   *both* its `xmm` register and the matching GPR (the callee cannot know
//!   which one to read). A variadic callee spills `rcx, rdx, r8, r9` into its
//!   home slots, so all its arguments sit contiguously on the stack and a
//!   `va_list` is a plain pointer walking 8-byte slots:
//!   `__lf_va_overflow_area()` returns the address of the first variadic
//!   argument (`rbp + 16 + 8 * named`) and `__lf_va_reg_save_area()` that of
//!   argument 0 (`rbp + 16`). `va_arg` reads 8 bytes per argument; an
//!   aggregate whose size is not 1, 2, 4 or 8 arrives as a pointer.
//! - **Callee-saved registers** are `rbx, rbp, rdi, rsi, r12..r15` and
//!   `xmm6..xmm15` ([`super::super::regs::RegFile::win64`]); the frame layout
//!   saves the ones a function writes.
//!
//! There is no `al` vector-count hint, and no red zone. Structured exception
//! handling unwind tables (`.pdata`/`.xdata`) are not emitted yet, so a
//! debugger or an SEH unwinder cannot walk through these frames.

use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::mir::{MachineInst, MachineOperand, PReg, RegClass, VReg};
use crate::ir::InstData;
use crate::ir::types::Type;

use super::{VaIntrinsic, X86Op, X86_64Target, align_up_u64, def, def_v, imm, is_aggregate, use_p, use_v};

/// Whether an aggregate of `size` bytes travels by value (in one GPR/slot).
fn by_value(size: u64) -> bool {
    matches!(size, 1 | 2 | 4 | 8)
}

/// The home of the four register arguments above the return address.
const SHADOW_SPACE: u64 = 32;

/// Where an incoming Win64 parameter slot *k* lives relative to `rbp`.
fn incoming_slot(k: usize) -> u64 {
    16 + 8 * k as u64
}

impl X86_64Target {
    /// Lower a `call` under the Microsoft x64 convention (see the module docs).
    pub(super) fn lower_call_win64(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let cc = &self.rf.cc;
        let ops = inst.operands();
        let callee = ops[0];
        let args = &ops[1..];

        // The variadic frame-address intrinsics (see the module docs).
        match lo.callee_name(callee).and_then(VaIntrinsic::from_name) {
            Some(VaIntrinsic::RegSaveArea) => {
                let d = lo.result_reg(inst);
                lo.emit(MachineInst::new(X86Op::LeaRbpOff.opcode(), vec![def_v(d), imm(incoming_slot(0))]));
                return;
            }
            Some(VaIntrinsic::OverflowArea) => {
                let d = lo.result_reg(inst);
                let off = lo
                    .va_overflow_off()
                    .expect("__lf_va_overflow_area called outside a variadic function");
                lo.emit(MachineInst::new(X86Op::LeaRbpOff.opcode(), vec![def_v(d), imm(off)]));
                return;
            }
            None => {}
        }
        let variadic_call = Self::callee_is_variadic(lo, callee);

        let ret_ty = inst.result().map(|r| lo.func().value_type(r));
        let ret_agg = ret_ty.filter(|&t| is_aggregate(lo.types(), t));
        let ret_size = ret_agg.map(|t| lo.byte_size(t));
        let sret = ret_size.is_some_and(|s| !by_value(s));

        // As on System V, the fixed-register moves are emitted as one run right
        // before the call.
        let mut reg_moves: Vec<(PReg, VReg)> = Vec::new();
        let mut slot = 0usize;

        let mut ret_slot = None;
        if sret {
            let t = ret_agg.expect("sret implies an aggregate return");
            let size = align_up_u64(lo.byte_size(t).max(8), 8);
            let align = lo.types().align_of(t).max(8);
            let s = lo.new_slot(size, align);
            ret_slot = Some(s);
            let ptr = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(self.frame_addr(ptr, s));
            reg_moves.push((cc.arg_regs[0], ptr));
            slot = 1;
        }

        for &arg in args {
            let ty = lo.func().value_type(arg);
            let (v, is_fp, store_size) = if is_aggregate(lo.types(), ty) {
                let size = lo.byte_size(ty);
                let ptr = self.oper(lo, arg);
                if by_value(size) {
                    let d = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(d), use_v(ptr), imm(size)]));
                    (d, false, 8)
                } else {
                    // By reference: a caller-owned, 16-byte-aligned copy.
                    let copy = lo.new_slot(align_up_u64(size.max(8), 8), 16);
                    let dst = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(self.frame_addr(dst, copy));
                    self.emit_memcpy(lo, dst, ptr, size);
                    (dst, false, 8)
                }
            } else if lo.types().is_vector(ty) {
                // By reference: a caller-owned, 16-byte-aligned copy.
                let v = self.oper(lo, arg);
                let copy = lo.new_slot(16, 16);
                let dst = lo.fresh_vreg(RegClass::Gpr);
                lo.emit(self.frame_addr(dst, copy));
                lo.emit(MachineInst::new(X86Op::VStore.opcode(), vec![use_v(dst), use_v(v), imm(1)]));
                (dst, false, 8)
            } else {
                let v = self.oper(lo, arg);
                let is_fp = lo.mf().vreg_class(v) == RegClass::Fp;
                (v, is_fp, lo.byte_size(ty).clamp(1, 8))
            };
            if slot < cc.arg_regs.len() {
                if is_fp {
                    reg_moves.push((cc.fp_arg_regs[slot], v));
                    if variadic_call {
                        // A variadic callee reads the GPR copy (movq gpr, xmm).
                        let g = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(g), use_v(v)]));
                        reg_moves.push((cc.arg_regs[slot], g));
                    }
                } else {
                    reg_moves.push((cc.arg_regs[slot], v));
                }
            } else {
                let dp = self.lea_rsp(lo, 8 * slot as u64);
                lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(dp), use_v(v), imm(store_size)]));
            }
            slot += 1;
        }
        // Shadow space for every call, plus the stack-passed arguments.
        lo.reserve_outgoing(align_up_u64((8 * slot as u64).max(SHADOW_SPACE), 16));

        let used_arg_regs: Vec<PReg> = reg_moves.iter().map(|&(areg, _)| areg).collect();
        for (areg, r) in reg_moves {
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(areg), use_v(r)]));
        }

        // A float or vector comes back in xmm0.
        let ret_is_fp = ret_ty.is_some_and(|t| lo.types().get(t).is_float() || lo.types().is_vector(t));
        let ret_reg = if ret_is_fp { cc.fp_ret_reg } else { cc.ret_reg };
        let mut operands = Vec::new();
        match lo.callee_func(callee) {
            Some(fidx) => operands.push(MachineOperand::Func(fidx)),
            None => {
                let cr = lo.reg(callee);
                operands.push(use_v(cr));
            }
        }
        operands.push(def(ret_reg));
        for &cs in &self.rf.caller_saved {
            if cs != ret_reg {
                operands.push(def(cs));
            }
        }
        for &areg in &used_arg_regs {
            operands.push(use_p(areg));
        }
        lo.emit(MachineInst::new(X86Op::Call.opcode(), operands));

        if let Some(s) = ret_slot {
            let d = lo.result_reg(inst);
            lo.emit(self.frame_addr(d, s));
        } else if let (Some(t), Some(size)) = (ret_agg, ret_size) {
            // A small aggregate came back in rax: store it into a result slot.
            let v = lo.fresh_vreg(RegClass::Gpr);
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(v), use_p(cc.ret_reg)]));
            let align = lo.types().align_of(t).max(8);
            let s = lo.new_slot(8, align);
            let d = lo.result_reg(inst);
            lo.emit(self.frame_addr(d, s));
            lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(d), use_v(v), imm(size)]));
        } else if inst.result().is_some() {
            let d = lo.result_reg(inst);
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_p(ret_reg)]));
        }
    }

    /// Lower the entry prologue under the Microsoft x64 convention: bind every
    /// parameter vreg, stash a hidden return pointer, and for a variadic
    /// function spill the four argument GPRs into their home slots.
    pub(super) fn lower_prologue_win64(&self, lo: &mut Lower<'_, Self>) {
        let cc = &self.rf.cc;
        let entry = lo.mf().entry().expect("a function being lowered has an entry block");
        let param_vregs: Vec<VReg> = lo.mf().block(entry).params.clone();
        let (sig_params, ret_ty, variadic) = match lo.types().get(lo.func().sig) {
            Type::Func(ft) => (ft.params.clone(), ft.ret, ft.variadic),
            _ => (Vec::new(), lo.func().sig, false),
        };
        let sret = is_aggregate(lo.types(), ret_ty) && !by_value(lo.byte_size(ret_ty));
        let first = usize::from(sret);

        // Copy every incoming argument register into a vreg first, as one run,
        // so nothing else is defined while they are live.
        let mut captured: Vec<Option<VReg>> = vec![None; cc.arg_regs.len()];
        let capture = |lo: &mut Lower<'_, Self>, preg: PReg, class: RegClass| {
            let v = lo.fresh_vreg(class);
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(v), use_p(preg)]));
            v
        };
        if variadic || sret {
            let n = if variadic { cc.arg_regs.len() } else { 1 };
            for (k, c) in captured.iter_mut().enumerate().take(n) {
                *c = Some(capture(lo, cc.arg_regs[k], RegClass::Gpr));
            }
        }
        let mut reg_param: Vec<Option<VReg>> = vec![None; param_vregs.len()];
        for (i, &pv) in param_vregs.iter().enumerate() {
            let k = first + i;
            if k >= cc.arg_regs.len() {
                break;
            }
            // A vector arrives by reference, as a pointer in the GPR.
            let by_ref = is_aggregate(lo.types(), sig_params[i]) || lo.types().is_vector(sig_params[i]);
            let is_fp = !by_ref && lo.mf().vreg_class(pv) == RegClass::Fp;
            reg_param[i] = Some(match (is_fp, captured[k]) {
                (false, Some(v)) => v,
                (false, None) => capture(lo, cc.arg_regs[k], RegClass::Gpr),
                (true, _) => capture(lo, cc.fp_arg_regs[k], RegClass::Fp),
            });
        }

        if variadic {
            // Spill rcx/rdx/r8/r9 to their home slots in the caller's frame.
            for (k, c) in captured.iter().enumerate() {
                let v = c.expect("captured above");
                let p = self.lea_rbp(lo, incoming_slot(k));
                lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(p), use_v(v), imm(8)]));
            }
        }
        if sret {
            let s = lo.new_slot(8, 8);
            lo.set_aux_slot(s);
            let v = captured[0].expect("captured above");
            lo.emit(MachineInst::new(X86Op::StoreFrame.opcode(), vec![use_v(v), MachineOperand::Frame(s)]));
        }

        for (i, &pv) in param_vregs.iter().enumerate() {
            let ty = sig_params[i];
            let k = first + i;
            let in_reg = reg_param[i];
            if is_aggregate(lo.types(), ty) {
                let size = lo.byte_size(ty);
                if by_value(size) {
                    match in_reg {
                        Some(v) => {
                            // Give the value a home so the body can address it.
                            let home = lo.new_slot(8, lo.types().align_of(ty).max(8));
                            lo.emit(self.frame_addr(pv, home));
                            lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(pv), use_v(v), imm(8)]));
                        }
                        None => {
                            // In place, in the caller's argument slot.
                            let d = self.lea_rbp(lo, incoming_slot(k));
                            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(pv), use_v(d)]));
                        }
                    }
                } else {
                    // Passed by reference: the slot holds a pointer to the copy.
                    match in_reg {
                        Some(v) => lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(pv), use_v(v)])),
                        None => {
                            let p = self.lea_rbp(lo, incoming_slot(k));
                            lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(pv), use_v(p), imm(8)]));
                        }
                    }
                }
            } else if lo.types().is_vector(ty) {
                // By reference: load the vector through the incoming pointer
                // (unaligned, whoever the caller is).
                let ptr = match in_reg {
                    Some(v) => v,
                    None => {
                        let p = self.lea_rbp(lo, incoming_slot(k));
                        let d = lo.fresh_vreg(RegClass::Gpr);
                        lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(d), use_v(p), imm(8)]));
                        d
                    }
                };
                lo.emit(MachineInst::new(X86Op::VLoad.opcode(), vec![def_v(pv), use_v(ptr), imm(0)]));
            } else {
                match in_reg {
                    Some(v) => lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(pv), use_v(v)])),
                    None => {
                        let p = self.lea_rbp(lo, incoming_slot(k));
                        let sz = lo.byte_size(ty).clamp(1, 8);
                        lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def_v(pv), use_v(p), imm(sz)]));
                    }
                }
            }
        }

        if variadic {
            lo.set_va_overflow_off(incoming_slot(first + param_vregs.len()));
        }
    }

    /// Lower `ret` under the Microsoft x64 convention: a small aggregate comes
    /// back in `rax`, any other aggregate through the hidden pointer (also
    /// returned in `rax`), a float in `xmm0`, anything else in `rax`.
    pub(super) fn lower_ret_win64(&self, lo: &mut Lower<'_, Self>, inst: &InstData) {
        let cc = &self.rf.cc;
        let ret_ty = match lo.types().get(lo.func().sig) {
            Type::Func(ft) => ft.ret,
            _ => lo.func().sig,
        };
        if is_aggregate(lo.types(), ret_ty) {
            let src = self.oper(lo, inst.operands()[0]);
            let size = lo.byte_size(ret_ty);
            if by_value(size) {
                lo.emit(MachineInst::new(X86Op::Load.opcode(), vec![def(cc.ret_reg), use_v(src), imm(size)]));
            } else {
                let s = lo.aux_slot().expect("sret pointer saved by the prologue");
                let dst = lo.fresh_vreg(RegClass::Gpr);
                lo.emit(MachineInst::new(X86Op::LoadFrame.opcode(), vec![def_v(dst), MachineOperand::Frame(s)]));
                self.emit_memcpy(lo, dst, src, size);
                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(cc.ret_reg), use_v(dst)]));
            }
        } else if let Some(&v) = inst.operands().first() {
            let r = self.oper(lo, v);
            let ret = match lo.mf().vreg_class(r) {
                RegClass::Fp => cc.fp_ret_reg,
                RegClass::Gpr => cc.ret_reg,
            };
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(ret), use_v(r)]));
        }
        lo.emit(MachineInst::new(X86Op::Ret.opcode(), Vec::new()));
    }
}

