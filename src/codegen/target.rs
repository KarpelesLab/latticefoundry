//! The target interface: what target-independent code generation needs to know
//! about a machine, *without* committing to instruction encodings (that is the
//! Phase 6 machine-code layer's job).
//!
//! A [`MachineTarget`] describes a register file (classes, the allocatable and
//! scratch physical registers of each, and the caller/callee-saved split), a
//! [`CallConv`], and a handful of builders the register allocator uses to splice
//! in moves and spill code. It is deliberately object-safe: the allocator drives
//! it through `&dyn MachineTarget`.
//!
//! Instruction selection needs *more* than this — a rule per IR opcode — so that
//! lives behind the separate [`crate::codegen::isel::TargetIsel`] trait, which a
//! concrete target implements alongside `MachineTarget`. Splitting the two keeps
//! `MachineTarget` small, real, and encoding-free, and keeps the allocator from
//! depending on isel. Phase 7 implements both traits for x86-64, AArch64, etc.

use crate::codegen::mir::{MachineInst, Opcode, PReg, Reg, RegClass, StackSlot};

/// A minimal but real calling convention: which physical registers carry the
/// leading integer arguments and the return value, and which way the stack
/// grows. Stack-passed arguments beyond [`CallConv::arg_regs`] are out of scope
/// for this phase (the abstract target keeps arities small).
#[derive(Clone, Debug)]
pub struct CallConv {
    /// Registers holding the leading integer/pointer arguments, in order.
    pub arg_regs: Vec<PReg>,
    /// Registers holding the leading floating-point arguments, in order. On
    /// System V these are `xmm0..xmm7` and are counted by a *separate* index
    /// from [`CallConv::arg_regs`] (integer and float args do not share slots).
    /// Empty for targets with no floating-point support in this phase.
    pub fp_arg_regs: Vec<PReg>,
    /// The register holding an integer/pointer return value.
    pub ret_reg: PReg,
    /// The register holding a floating-point return value (`xmm0` on System V).
    pub fp_ret_reg: PReg,
    /// Whether the stack grows toward lower addresses (true on most machines).
    pub stack_grows_down: bool,
}

/// The description of a target that target-independent codegen consumes.
///
/// Every method is encoding-free: the allocator learns the register file and the
/// ABI, and asks the target to *build* moves and spill/reload instructions, but
/// never to turn anything into bytes.
pub trait MachineTarget: std::fmt::Debug {
    /// A short human-readable target name.
    fn name(&self) -> &str;

    /// The register classes this target exposes.
    fn reg_classes(&self) -> &[RegClass];

    /// The allocatable physical registers of a class, in allocation-preference
    /// order. These are the registers linear scan may assign to vregs.
    fn allocatable(&self, class: RegClass) -> &[PReg];

    /// Physical registers of a class reserved for spill/reload code — never
    /// assigned to a vreg, always free for the rewriter to borrow. There must be
    /// enough to cover the register operands of any single instruction.
    fn scratch(&self, class: RegClass) -> &[PReg];

    /// The caller-saved physical registers: those a `call` may clobber. A value
    /// live across a call must not live in one of these.
    fn caller_saved(&self) -> &[PReg];

    /// The callee-saved physical registers: those a callee must preserve.
    fn callee_saved(&self) -> &[PReg];

    /// The calling convention.
    fn call_conv(&self) -> &CallConv;

    /// Whether an opcode is a block terminator (ends a machine block).
    fn is_terminator(&self, op: Opcode) -> bool;

    /// Whether an opcode is a register-to-register move (a copy). Used for
    /// well-formedness checks and potential future coalescing.
    fn is_move(&self, op: Opcode) -> bool;

    /// Build a register-to-register move `dst <- src`.
    fn emit_move(&self, dst: Reg, src: Reg) -> MachineInst;

    /// Build a spill: store the register `src` into stack `slot`.
    fn emit_spill(&self, slot: StackSlot, src: PReg) -> MachineInst;

    /// Build a reload: load stack `slot` into register `dst`.
    fn emit_reload(&self, dst: PReg, slot: StackSlot) -> MachineInst;
}
