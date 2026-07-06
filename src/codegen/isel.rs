//! The instruction-selection framework: the reusable driver that lowers an IR
//! [`Function`] to a [`MachineFunction`], leaving the per-opcode rules to the
//! target (via [`TargetIsel`]).
//!
//! The framework owns everything that is target-independent:
//!
//! - **Value → vreg mapping.** Each IR instruction result and block parameter
//!   gets one stable vreg, created up front; constants, globals, and function
//!   references are materialized on demand (a constant becomes a load-immediate
//!   in the block that uses it).
//! - **Block-argument lowering — the interesting decision.** IR block arguments
//!   (the block-parameter/edge-argument encoding that replaces φ-nodes) are
//!   lowered to **register moves on the edge**: a block parameter is a fixed
//!   vreg, and each predecessor writes it before branching. To place those moves
//!   without clobber hazards, every arg-carrying edge is **split**: the
//!   predecessor branches to a fresh *edge block* that performs the parallel
//!   copy (with cycle breaking via a temp) and then jumps to the real successor.
//!   This is always correct regardless of how many successors the terminator has
//!   and never needs critical-edge reasoning at the call sites.
//! - **The ABI seam.** Function parameters are moved out of the physical
//!   argument registers in an entry prologue; `call`/`ret` move through the
//!   physical arg/return registers and clobber caller-saved registers, so the
//!   allocator sees the ABI as ordinary fixed register operands.
//!
//! The target supplies only the leaf rules: how each IR opcode becomes machine
//! instructions, and small builders (`li`, `jump`, `frame_addr`, ...). See the
//! abstract virtual target for a worked implementation.

use crate::codegen::mir::{MBlockId, MachineFunction, MachineInst, Reg, RegClass, StackSlot, VReg};
use crate::codegen::target::MachineTarget;
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, FloatBits, ValueDef};
use crate::ir::{Function, InstData, Module, ValueId};
use crate::support::DetHashMap;

use puremp::Int;

/// The leaf rules a target plugs into the [`Lower`] driver: how to lower one IR
/// instruction, and the small instruction builders the framework needs.
///
/// `TargetIsel` is deliberately *not* object-safe (it is used through generics),
/// which lets its methods hand back concrete [`MachineInst`]s and take the
/// concrete [`Lower`] context. A target implements it alongside
/// [`MachineTarget`].
pub trait TargetIsel: MachineTarget + Sized {
    /// Lower one non-terminator IR instruction, emitting machine instructions
    /// into the current block via `lo`.
    fn lower_inst(&self, lo: &mut Lower<'_, Self>, inst: &InstData);

    /// Lower a terminator IR instruction. Use [`Lower::edge_to`] to realize each
    /// outgoing edge's block arguments, then emit the branch/return.
    fn lower_term(&self, lo: &mut Lower<'_, Self>, inst: &InstData);

    /// Lower the entry prologue: move each incoming parameter out of its ABI
    /// location into the parameter's vreg. The default implementation
    /// ([`Lower::default_prologue`]) draws each parameter from the next argument
    /// register of its class (the scalar System V rule) — correct for every
    /// target whose parameters each fit in one register. A target with aggregate
    /// (by-value struct) parameters, a hidden `sret` pointer, or stack-passed
    /// parameters overrides this to lay them out per its ABI.
    fn lower_prologue(&self, lo: &mut Lower<'_, Self>) {
        lo.default_prologue();
    }

    /// Build a load-immediate `dst <- value`.
    fn li(&self, dst: VReg, value: Int) -> MachineInst;

    /// Build an unconditional jump to `dst`.
    fn jump(&self, dst: MBlockId) -> MachineInst;

    /// Build "materialize the address of stack `slot` into `dst`".
    fn frame_addr(&self, dst: VReg, slot: StackSlot) -> MachineInst;

    /// Build "materialize the address of global `g` into `dst`".
    fn global_addr(&self, dst: VReg, g: u32) -> MachineInst;

    /// Build "materialize the floating-point constant with raw IEEE bit pattern
    /// `bits` (a `width`-bit value) into the floating-point register `dst`".
    ///
    /// The default falls back to [`TargetIsel::li`], which is only correct for
    /// targets that do not model a separate floating-point file (they never
    /// receive a float-typed value in practice); the x86-64 target overrides it
    /// to load the exact bit pattern into an xmm register.
    fn float_const(&self, dst: VReg, bits: u64, _width: u32) -> MachineInst {
        self.li(dst, Int::from_u64(bits))
    }
}

/// A resolved operand source during edge lowering: either a register value or an
/// immediate that will be materialized in place.
#[derive(Clone, Debug)]
enum Src {
    Reg(VReg),
    Imm(Int),
}

/// The lowering context threaded through instruction selection.
#[derive(Debug)]
pub struct Lower<'a, T: TargetIsel> {
    target: &'a T,
    module: &'a Module,
    func: &'a Function,
    mf: MachineFunction,
    /// Machine block per IR block (index by `BlockId::index`).
    block_map: Vec<MBlockId>,
    /// Stable vreg per IR value (`None` for constants/globals/functions).
    value_reg: Vec<Option<VReg>>,
    /// Per-block cache of materialized constant/global values.
    materialized: DetHashMap<(usize, usize), VReg>,
    /// The current insertion block.
    cur: MBlockId,
    /// Source line of the IR instruction currently being lowered (`0` = none),
    /// stamped onto every [`MachineInst`] emitted for it (debug info).
    cur_line: u32,
    /// A stack slot a target's prologue can stash an incoming hidden ABI pointer
    /// into (the System V `sret` return pointer), to be recovered by the return
    /// lowering. `None` unless a target uses it.
    aux_slot: Option<StackSlot>,
}

/// Map an IR type to the register class that holds it.
fn class_of(types: &TypeContext, ty: TypeId) -> RegClass {
    match types.get(ty) {
        Type::Float(_) => RegClass::Fp,
        _ => RegClass::Gpr,
    }
}

impl<'a, T: TargetIsel> Lower<'a, T> {
    fn new(target: &'a T, module: &'a Module, func: &'a Function, source: u32) -> Lower<'a, T> {
        let types = module.types();
        let mut mf = MachineFunction::new(format!("f{source}"), source);
        let n = func.block_count();
        let mut block_map = Vec::with_capacity(n);
        for _ in 0..n {
            block_map.push(mf.add_block());
        }
        let mut value_reg = vec![None; func.value_count()];
        for (bid, block) in func.blocks() {
            let mb = block_map[bid.index()];
            for &p in block.params() {
                let v = mf.new_vreg(class_of(types, func.value_type(p)));
                value_reg[p.index()] = Some(v);
                mf.block_mut(mb).params.push(v);
            }
            for &iid in block.insts() {
                if let Some(r) = func.inst(iid).result() {
                    let v = mf.new_vreg(class_of(types, func.value_type(r)));
                    value_reg[r.index()] = Some(v);
                }
            }
        }
        let entry = func.entry().map(|e| block_map[e.index()]).unwrap_or(block_map[0]);
        if let Some(e) = func.entry() {
            mf.set_entry(block_map[e.index()]);
            mf.set_num_params(func.block(e).params().len());
        }
        Lower {
            target,
            module,
            func,
            mf,
            block_map,
            value_reg,
            materialized: DetHashMap::default(),
            cur: entry,
            cur_line: 0,
            aux_slot: None,
        }
    }

    // --- accessors the target rules use ------------------------------------

    /// The module being lowered.
    #[inline]
    pub fn module(&self) -> &Module {
        self.module
    }

    /// The IR function being lowered.
    #[inline]
    pub fn func(&self) -> &Function {
        self.func
    }

    /// The type context.
    #[inline]
    pub fn types(&self) -> &TypeContext {
        self.module.types()
    }

    /// The machine function under construction (for reading; mutate via helpers).
    #[inline]
    pub fn mf(&self) -> &MachineFunction {
        &self.mf
    }

    /// The bit width of a value's integer type, treating pointers as 64-bit.
    pub fn int_width(&self, v: ValueId) -> u32 {
        self.types().bit_width(self.func.value_type(v)).unwrap_or(64)
    }

    /// The byte size of a value's type under the default data layout.
    pub fn byte_size(&self, ty: TypeId) -> u64 {
        self.types().size_of(ty)
    }

    /// If `v` is a direct reference to a function, its index; else `None`.
    pub fn callee_func(&self, v: ValueId) -> Option<u32> {
        match self.func.value(v).def {
            ValueDef::Func(f) => Some(f.index() as u32),
            _ => None,
        }
    }

    // --- vreg / slot / block helpers ---------------------------------------

    /// Allocate a fresh virtual register of a class.
    pub fn fresh_vreg(&mut self, class: RegClass) -> VReg {
        self.mf.new_vreg(class)
    }

    /// Allocate a fresh stack slot.
    pub fn new_slot(&mut self, size: u64, align: u64) -> StackSlot {
        self.mf.frame_mut().add_slot(size, align)
    }

    /// Record the slot a target's prologue stashed the incoming hidden ABI
    /// pointer (System V `sret`) into, for the return lowering to recover.
    pub fn set_aux_slot(&mut self, slot: StackSlot) {
        self.aux_slot = Some(slot);
    }

    /// The slot set by [`Lower::set_aux_slot`], if any.
    #[inline]
    pub fn aux_slot(&self) -> Option<StackSlot> {
        self.aux_slot
    }

    /// Reserve at least `bytes` of outgoing stack-argument space in the frame
    /// (the running maximum over call sites); see [`crate::codegen::mir::Frame`].
    pub fn reserve_outgoing(&mut self, bytes: u64) {
        self.mf.frame_mut().reserve_outgoing(bytes);
    }

    /// The machine block for an IR block.
    pub fn mblock(&self, b: crate::ir::BlockId) -> MBlockId {
        self.block_map[b.index()]
    }

    /// The stable vreg holding the result of `inst` (which must define a value).
    pub fn result_reg(&self, inst: &InstData) -> VReg {
        let r = inst.result().expect("instruction defines no result");
        self.value_reg[r.index()].expect("result vreg was pre-created")
    }

    /// Emit an instruction into the current block. Instructions that do not
    /// already carry a source line inherit the line of the IR instruction being
    /// lowered, so the encoder can build a `.debug_line` table.
    pub fn emit(&mut self, mut inst: MachineInst) {
        if inst.line == 0 {
            inst.line = self.cur_line;
        }
        self.mf.block_mut(self.cur).insts.push(inst);
    }

    /// Resolve an IR value operand to a register, materializing constants and
    /// global/function references into the current block on demand.
    pub fn reg(&mut self, v: ValueId) -> VReg {
        match self.func.value(v).def.clone() {
            ValueDef::Inst(_) | ValueDef::Param(_, _) => {
                self.value_reg[v.index()].expect("SSA value vreg was pre-created")
            }
            ValueDef::Const(_) | ValueDef::Global(_) | ValueDef::Func(_) => self.materialize(v),
        }
    }

    /// Materialize a constant / global / function reference into a fresh vreg,
    /// cached per block so repeated uses share one definition.
    fn materialize(&mut self, v: ValueId) -> VReg {
        let key = (self.cur.index(), v.index());
        if let Some(&r) = self.materialized.get(&key) {
            return r;
        }
        let ty = self.func.value_type(v);
        let cls = class_of(self.types(), ty);
        let d = self.mf.new_vreg(cls);
        let target = self.target;
        let inst = match self.func.value(v).def.clone() {
            ValueDef::Const(c) => match self.module.consts().get(c).clone() {
                Const::Int { value, .. } => target.li(d, value),
                Const::Null(_) | Const::Poison(_) => target.li(d, Int::ZERO),
                // A float constant loads its exact IEEE bit pattern into the fp
                // register `d` (whose class is `Fp`, since the value is float-typed).
                Const::Float { bits, .. } => {
                    let (raw, width) = match bits {
                        FloatBits::F16(b) => (u64::from(b), 16),
                        FloatBits::F32(b) => (u64::from(b), 32),
                        FloatBits::F64(b) => (b, 64),
                    };
                    target.float_const(d, raw, width)
                }
                // Aggregates are out of the scalar subset; a zero placeholder
                // keeps the MIR well-formed (never executed in tests).
                Const::Aggregate { .. } => target.li(d, Int::ZERO),
            },
            ValueDef::Global(g) => target.global_addr(d, g.index() as u32),
            // A function used as a plain value (not a direct call target): a zero
            // placeholder address (real symbol handling is Phase 6/7).
            ValueDef::Func(_) => target.li(d, Int::ZERO),
            ValueDef::Inst(_) | ValueDef::Param(_, _) => unreachable!(),
        };
        self.emit(inst);
        self.materialized.insert(key, d);
        d
    }

    /// Resolve an IR value to an edge-copy source: an immediate for a constant,
    /// otherwise its register.
    fn src(&mut self, v: ValueId) -> Src {
        if let ValueDef::Const(c) = self.func.value(v).def {
            match self.module.consts().get(c) {
                Const::Int { value, .. } => return Src::Imm(value.clone()),
                Const::Null(_) | Const::Poison(_) => return Src::Imm(Int::ZERO),
                _ => {}
            }
        }
        Src::Reg(self.reg(v))
    }

    /// Realize an outgoing edge to `succ` passing `args` as its block arguments,
    /// returning the block the predecessor should branch to. Arg-carrying edges
    /// are split into a dedicated edge block holding the parallel copy; an edge
    /// with no arguments returns the successor's own machine block.
    pub fn edge_to(&mut self, succ: crate::ir::BlockId, args: &[ValueId]) -> MBlockId {
        let succ_mb = self.mblock(succ);
        if args.is_empty() {
            return succ_mb;
        }
        let dsts: Vec<VReg> = self.mf.block(succ_mb).params.clone();
        debug_assert_eq!(dsts.len(), args.len(), "edge arity matches successor params");
        let srcs: Vec<Src> = args.iter().map(|&a| self.src(a)).collect();
        let edge = self.mf.add_block();
        let saved = self.cur;
        self.cur = edge;
        self.parallel_copy(dsts, srcs);
        let jump = self.target.jump(succ_mb);
        self.emit(jump);
        self.cur = saved;
        edge
    }

    /// Emit a parallel copy `dsts[i] <- srcs[i]` into the current block, breaking
    /// cycles with a temporary. Destinations are distinct (block parameters).
    fn parallel_copy(&mut self, dsts: Vec<VReg>, srcs: Vec<Src>) {
        // Drop self-copies (`d <- d`), which are no-ops.
        let mut copies: Vec<(VReg, Src)> = dsts
            .into_iter()
            .zip(srcs)
            .filter(|(d, s)| !matches!(s, Src::Reg(r) if r == d))
            .collect();

        while !copies.is_empty() {
            // A copy is ready if its destination is not read by any other copy.
            let ready = copies.iter().position(|(d, _)| {
                !copies.iter().any(|(_, s)| matches!(s, Src::Reg(r) if r == d))
            });
            match ready {
                Some(i) => {
                    let (d, s) = copies.remove(i);
                    let inst = match s {
                        Src::Reg(r) => self.target.emit_move(Reg::Virtual(d), Reg::Virtual(r)),
                        Src::Imm(v) => self.target.li(d, v),
                    };
                    self.emit(inst);
                }
                None => {
                    // Every remaining destination is also a source: a cycle.
                    // Break it by saving one register source into a temp.
                    let i = copies
                        .iter()
                        .position(|(_, s)| matches!(s, Src::Reg(_)))
                        .expect("a cycle must contain a register source");
                    let Src::Reg(s) = copies[i].1.clone() else { unreachable!() };
                    let t = self.mf.new_vreg(self.mf.vreg_class(s));
                    let mv = self.target.emit_move(Reg::Virtual(t), Reg::Virtual(s));
                    self.emit(mv);
                    for c in &mut copies {
                        if matches!(c.1, Src::Reg(r) if r == s) {
                            c.1 = Src::Reg(t);
                        }
                    }
                }
            }
        }
    }

    // --- the driver --------------------------------------------------------

    fn run(&mut self) {
        let t = self.target;
        let f = self.func;
        let entry = f.entry();
        for (bid, block) in f.blocks() {
            let mb = self.block_map[bid.index()];
            self.cur = mb;
            if Some(bid) == entry {
                t.lower_prologue(self);
            }
            for &iid in block.insts() {
                self.cur_line = f.inst_line(iid).unwrap_or(0);
                t.lower_inst(self, f.inst(iid));
            }
            if let Some(tid) = block.terminator() {
                self.cur_line = f.inst_line(tid).unwrap_or(0);
                t.lower_term(self, f.inst(tid));
            }
            self.cur_line = 0;
        }
    }

    /// The default entry prologue: move each incoming parameter out of its
    /// physical argument register into the parameter's vreg. This is the scalar
    /// System V rule (one register per parameter, integer and floating-point
    /// counted independently); the [`TargetIsel::lower_prologue`] hook delegates
    /// here unless a target overrides it for aggregate/`sret`/stack parameters.
    pub fn default_prologue(&mut self) {
        let t = self.target;
        let cc = t.call_conv();
        let params: Vec<VReg> = self.mf.block(self.cur).params.clone();
        // Integer/pointer and floating-point parameters draw from *separate*
        // argument-register sequences (SysV counts them independently): an
        // integer param takes the next `arg_regs` slot, a float param the next
        // `fp_arg_regs` slot.
        let mut int_i = 0usize;
        let mut fp_i = 0usize;
        let mut moves: Vec<MachineInst> = Vec::with_capacity(params.len());
        for &p in &params {
            let areg = match self.mf.vreg_class(p) {
                RegClass::Gpr => {
                    let r = cc.arg_regs[int_i];
                    int_i += 1;
                    r
                }
                RegClass::Fp => {
                    let r = cc.fp_arg_regs[fp_i];
                    fp_i += 1;
                    r
                }
            };
            moves.push(t.emit_move(Reg::Virtual(p), Reg::Physical(areg)));
        }
        for mv in moves {
            self.emit(mv);
        }
    }

    fn finish(self) -> MachineFunction {
        self.mf
    }
}

/// Lower function `func` of `module` to a [`MachineFunction`] over `target`. A
/// function with no body yields an empty machine function. The resulting MIR
/// records `func`'s index as its source, so direct calls resolve by `FuncId`.
pub fn select<T: TargetIsel>(target: &T, module: &Module, func: crate::ir::FuncId) -> MachineFunction {
    let f = module.function(func);
    let mut lo = Lower::new(target, module, f, func.index() as u32);
    if f.entry().is_some() {
        lo.run();
    }
    lo.finish()
}
