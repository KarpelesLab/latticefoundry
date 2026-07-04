//! The machine-level IR (MIR): the target-abstract data model that
//! instruction selection produces and that register allocation rewrites in
//! place (ROADMAP Phase 5).
//!
//! MIR mirrors the CFG shape of the source [`crate::ir`] — a
//! [`MachineFunction`] owns an arena of [`MachineBlock`]s, each a straight-line
//! list of [`MachineInst`]s ending in a terminator — but it is *below* SSA:
//! values live in **registers**, either infinite [`VReg`]s (virtual, produced
//! by isel) or finite [`PReg`]s (physical, produced by the allocator). An
//! instruction carries a target-defined [`Opcode`] plus a flat list of
//! [`MachineOperand`]s; the operand kind (`Def`/`Use`/immediate/frame/label)
//! records the def/use information the allocator and liveness need without
//! knowing anything about the target's encodings (that is Phase 6).
//!
//! Everything is id/arena-based and deterministic (tenets T5/T6): block order
//! is arena order, register numbers are dense, and successor edges are recovered
//! from the terminator's [`MachineOperand::Label`] operands rather than stored
//! redundantly.

use crate::support::{Arena, Id};

use puremp::Int;

/// A register class: a set of physical registers that can hold the same kind of
/// value and are interchangeable for allocation. The framework leaves room for
/// floating-point / vector files; the abstract virtual target uses only
/// [`RegClass::Gpr`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum RegClass {
    /// General-purpose integer/pointer registers.
    Gpr,
    /// Floating-point registers (reserved; unused by the virtual target).
    Fp,
}

impl RegClass {
    /// A dense index for this class, for indexing per-class tables.
    #[inline]
    pub fn index(self) -> usize {
        match self {
            RegClass::Gpr => 0,
            RegClass::Fp => 1,
        }
    }
}

/// A physical (machine) register: a class plus a small number within that class.
///
/// Physical registers are finite and target-defined. Equality is structural, so
/// the same `(class, num)` always denotes the same machine register.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PReg {
    /// The register class this physical register belongs to.
    pub class: RegClass,
    /// The register's index within its class.
    pub num: u16,
}

impl PReg {
    /// Construct a physical register of `class` with the given class-local index.
    #[inline]
    pub fn new(class: RegClass, num: u16) -> PReg {
        PReg { class, num }
    }
}

/// A virtual register: an infinite, SSA-friendly value name produced by
/// instruction selection. Its [`RegClass`] is recorded in the owning
/// [`MachineFunction`]'s vreg table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct VReg(u32);

impl VReg {
    /// The dense index this virtual register addresses in the vreg table.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Reconstruct a vreg handle from its dense index (crate-internal).
    #[inline]
    pub(crate) fn from_index(i: usize) -> VReg {
        VReg(i as u32)
    }
}

/// A register operand: either a virtual register (pre-allocation) or a physical
/// one (fixed by the ABI, or the result of allocation).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Reg {
    /// A virtual register.
    Virtual(VReg),
    /// A physical register.
    Physical(PReg),
}

impl Reg {
    /// The virtual register this operand names, if it is virtual.
    #[inline]
    pub fn as_virtual(self) -> Option<VReg> {
        match self {
            Reg::Virtual(v) => Some(v),
            Reg::Physical(_) => None,
        }
    }

    /// The physical register this operand names, if it is physical.
    #[inline]
    pub fn as_physical(self) -> Option<PReg> {
        match self {
            Reg::Physical(p) => Some(p),
            Reg::Virtual(_) => None,
        }
    }
}

/// A stack-frame slot: storage for a spill or an `alloca`. The concrete byte
/// offset is assigned late (prologue/epilogue construction, Phase 6); here a
/// slot is an abstract handle with a size and alignment.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct StackSlot(u32);

impl StackSlot {
    /// The dense index this slot addresses in the frame's slot table.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Reconstruct a slot handle from its dense index (crate-internal).
    #[inline]
    pub(crate) fn from_index(i: usize) -> StackSlot {
        StackSlot(i as u32)
    }
}

/// The size and alignment of one [`StackSlot`], in bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SlotInfo {
    /// Slot size in bytes.
    pub size: u64,
    /// Slot alignment in bytes (a power of two).
    pub align: u64,
}

/// A target-defined machine opcode, kept as an opaque interned id so the MIR
/// data model stays target-independent. Each target assigns meaning to the
/// numbers (see the abstract virtual target's `VOp`); the framework only ever
/// asks the target about an opcode (is it a terminator, a move, ...).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Opcode(pub u32);

/// A machine basic block within a [`MachineFunction`], addressed by [`MBlockId`].
pub type MBlockId = Id<MachineBlock>;

/// One operand of a [`MachineInst`].
///
/// Register operands are explicitly tagged as a **def** (written) or a **use**
/// (read); this is the def/use information the allocator and liveness reason
/// about. Non-register operands carry immediates, frame references, branch
/// labels, and symbol references.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MachineOperand {
    /// A register written by the instruction.
    Def(Reg),
    /// A register read by the instruction.
    Use(Reg),
    /// An arbitrary-precision immediate.
    Imm(Int),
    /// A reference to a stack-frame slot.
    Frame(StackSlot),
    /// A branch target (successor block).
    Label(MBlockId),
    /// A direct-call target: the index of the callee function.
    Func(u32),
    /// A reference to a module global, by index.
    Global(u32),
}

impl MachineOperand {
    /// The register named by this operand, if it is a register operand.
    #[inline]
    pub fn reg(&self) -> Option<Reg> {
        match self {
            MachineOperand::Def(r) | MachineOperand::Use(r) => Some(*r),
            _ => None,
        }
    }
}

/// A single machine instruction: an [`Opcode`] and its flat operand list.
///
/// The operand order is fixed per opcode by the target; the framework never
/// interprets it beyond walking register defs/uses and branch labels.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MachineInst {
    /// The target opcode.
    pub opcode: Opcode,
    /// The operands, in target-defined order.
    pub operands: Vec<MachineOperand>,
}

impl MachineInst {
    /// Build an instruction from an opcode and operands.
    #[inline]
    pub fn new(opcode: Opcode, operands: Vec<MachineOperand>) -> MachineInst {
        MachineInst { opcode, operands }
    }

    /// The registers this instruction defines, in operand order.
    pub fn defs(&self) -> impl Iterator<Item = Reg> + '_ {
        self.operands.iter().filter_map(|o| match o {
            MachineOperand::Def(r) => Some(*r),
            _ => None,
        })
    }

    /// The registers this instruction uses, in operand order.
    pub fn uses(&self) -> impl Iterator<Item = Reg> + '_ {
        self.operands.iter().filter_map(|o| match o {
            MachineOperand::Use(r) => Some(*r),
            _ => None,
        })
    }

    /// The block labels this instruction branches to, in operand order.
    pub fn labels(&self) -> impl Iterator<Item = MBlockId> + '_ {
        self.operands.iter().filter_map(|o| match o {
            MachineOperand::Label(b) => Some(*b),
            _ => None,
        })
    }
}

/// A machine basic block: a typed parameter list (the block-argument vregs its
/// predecessors write on the edge), a straight-line instruction sequence, and —
/// once lowered — a terminator as its last instruction.
#[derive(Clone, Default, Debug)]
pub struct MachineBlock {
    /// The block's parameter vregs (the machine form of IR block arguments).
    pub params: Vec<VReg>,
    /// The instructions, in execution order; the last is the terminator.
    pub insts: Vec<MachineInst>,
}

impl MachineBlock {
    /// The block's terminator (its last instruction), if any.
    #[inline]
    pub fn terminator(&self) -> Option<&MachineInst> {
        self.insts.last()
    }

    /// The successor blocks, recovered from the terminator's label operands.
    pub fn successors(&self) -> Vec<MBlockId> {
        match self.insts.last() {
            Some(t) => t.labels().collect(),
            None => Vec::new(),
        }
    }
}

/// Per-virtual-register metadata (currently just its class).
#[derive(Clone, Copy, Debug)]
pub struct VRegInfo {
    /// The register class this vreg must be allocated in.
    pub class: RegClass,
}

/// The stack frame of a [`MachineFunction`]: the spill/`alloca` slots.
#[derive(Clone, Default, Debug)]
pub struct Frame {
    slots: Vec<SlotInfo>,
}

impl Frame {
    /// Add a slot of the given size/alignment, returning its handle.
    pub fn add_slot(&mut self, size: u64, align: u64) -> StackSlot {
        let id = StackSlot(self.slots.len() as u32);
        self.slots.push(SlotInfo { size, align });
        id
    }

    /// The metadata of a slot.
    #[inline]
    pub fn slot(&self, slot: StackSlot) -> SlotInfo {
        self.slots[slot.index()]
    }

    /// The number of slots in the frame.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the frame has no slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// The calling-convention summary a [`MachineFunction`] records so the allocator
/// and (later) prologue construction know the ABI it was lowered against.
#[derive(Clone, Debug)]
pub struct FrameInfo {
    /// The index of the function this MIR was lowered from (its `FuncId`).
    pub source: u32,
    /// The number of incoming parameters (in the entry block).
    pub num_params: usize,
}

/// A function in machine form: an arena of blocks, a virtual-register table, a
/// stack frame, and its entry block.
#[derive(Clone, Debug)]
pub struct MachineFunction {
    /// A human-readable name (for diagnostics and MIR dumps).
    pub name: String,
    blocks: Arena<MachineBlock>,
    vregs: Vec<VRegInfo>,
    frame: Frame,
    entry: Option<MBlockId>,
    info: FrameInfo,
}

impl MachineFunction {
    /// Create an empty machine function.
    pub fn new(name: impl Into<String>, source: u32) -> MachineFunction {
        MachineFunction {
            name: name.into(),
            blocks: Arena::new(),
            vregs: Vec::new(),
            frame: Frame::default(),
            entry: None,
            info: FrameInfo { source, num_params: 0 },
        }
    }

    /// Allocate a fresh virtual register of the given class.
    pub fn new_vreg(&mut self, class: RegClass) -> VReg {
        let id = VReg(self.vregs.len() as u32);
        self.vregs.push(VRegInfo { class });
        id
    }

    /// The class of a virtual register.
    #[inline]
    pub fn vreg_class(&self, v: VReg) -> RegClass {
        self.vregs[v.index()].class
    }

    /// The number of virtual registers.
    #[inline]
    pub fn num_vregs(&self) -> usize {
        self.vregs.len()
    }

    /// Append a fresh empty block, returning its id.
    pub fn add_block(&mut self) -> MBlockId {
        self.blocks.push(MachineBlock::default())
    }

    /// Borrow a block.
    #[inline]
    pub fn block(&self, id: MBlockId) -> &MachineBlock {
        &self.blocks[id]
    }

    /// Mutably borrow a block.
    #[inline]
    pub fn block_mut(&mut self, id: MBlockId) -> &mut MachineBlock {
        &mut self.blocks[id]
    }

    /// Iterate over every block id in arena order.
    pub fn block_ids(&self) -> impl Iterator<Item = MBlockId> + '_ {
        self.blocks.ids()
    }

    /// The number of blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// The entry block.
    #[inline]
    pub fn entry(&self) -> Option<MBlockId> {
        self.entry
    }

    /// Set the entry block.
    pub fn set_entry(&mut self, block: MBlockId) {
        self.entry = Some(block);
    }

    /// The stack frame.
    #[inline]
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// The stack frame, mutably (to add spill slots).
    #[inline]
    pub fn frame_mut(&mut self) -> &mut Frame {
        &mut self.frame
    }

    /// The ABI/frame summary.
    #[inline]
    pub fn info(&self) -> &FrameInfo {
        &self.info
    }

    /// Record the number of incoming parameters.
    pub fn set_num_params(&mut self, n: usize) {
        self.info.num_params = n;
    }
}
