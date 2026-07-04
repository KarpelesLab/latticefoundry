//! The LatticeFoundry intermediate representation (IR).
//!
//! The IR is an SSA-form, index-addressed representation. Entities live in
//! flat arenas owned by their parent and reference one another through small
//! `Copy` id newtypes rather than pointers, which keeps the graph friendly to
//! Rust's ownership model. See ROADMAP Phase 1.
//!
//! This module currently defines the container hierarchy
//! (module → function → block → instruction) with a deliberately small seed
//! set of instructions. The full opcode table, SSA value numbering, and the
//! builder API are filled in over Phase 1.

pub mod types;

pub use types::{FloatKind, FuncType, Type};

use crate::support::Sym;

/// Declares a `u32`-backed id newtype with an `index()` accessor.
macro_rules! id_newtype {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
        pub struct $name(u32);

        impl $name {
            /// The dense index this id addresses.
            #[inline]
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_newtype!(
    /// Handle to a [`Function`] within a [`Module`].
    FuncId
);
id_newtype!(
    /// Handle to a [`Block`] within a [`Function`].
    BlockId
);
id_newtype!(
    /// Handle to an SSA value (an instruction result or a block argument).
    ValueId
);

/// A translation unit: the top-level container of IR.
#[derive(Debug, Default)]
pub struct Module {
    /// Human-readable module identifier (typically the source file name).
    pub name: String,
    functions: Vec<Function>,
}

impl Module {
    /// Create an empty module with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), functions: Vec::new() }
    }

    /// Append a function, returning its stable handle.
    pub fn add_function(&mut self, func: Function) -> FuncId {
        let id = FuncId(self.functions.len() as u32);
        self.functions.push(func);
        id
    }

    /// Borrow a function by handle.
    pub fn function(&self, id: FuncId) -> &Function {
        &self.functions[id.index()]
    }

    /// Iterate over every function in definition order.
    pub fn functions(&self) -> impl Iterator<Item = &Function> {
        self.functions.iter()
    }
}

/// A function definition or declaration.
///
/// A function with no blocks is an external declaration; a function with at
/// least one block is a definition whose first block is its entry.
#[derive(Debug)]
pub struct Function {
    /// The interned symbol name of the function.
    pub name: Sym,
    /// The function's type signature.
    pub sig: FuncType,
    blocks: Vec<Block>,
}

impl Function {
    /// Create a function with the given name and signature and no body.
    pub fn new(name: Sym, sig: FuncType) -> Self {
        Self { name, sig, blocks: Vec::new() }
    }

    /// Append an empty basic block, returning its handle.
    pub fn add_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(Block::default());
        id
    }

    /// Borrow a block by handle.
    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.index()]
    }

    /// Mutably borrow a block by handle.
    pub fn block_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.blocks[id.index()]
    }

    /// Whether this function is an external declaration (has no body).
    pub fn is_declaration(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// A basic block: a straight-line instruction sequence with a single entry
/// and (once complete) a single terminating instruction.
#[derive(Debug, Default)]
pub struct Block {
    /// Instructions in execution order.
    pub insts: Vec<Inst>,
}

/// A single IR instruction.
///
/// This is an intentionally small seed set used to validate the container
/// design. The complete opcode table (arithmetic, memory, control flow, φ
/// nodes, calls, ...) is built out in ROADMAP Phase 1.
#[derive(Clone, Debug)]
pub enum Inst {
    /// Integer addition of two SSA values.
    Add(ValueId, ValueId),
    /// Function return, optionally yielding a value.
    Ret(Option<ValueId>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::StrInterner;

    #[test]
    fn build_a_trivial_function() {
        let mut syms = StrInterner::new();
        let mut module = Module::new("smoke");

        let sig = FuncType::new(vec![], Type::Void);
        let mut func = Function::new(syms.intern("main"), sig);
        let entry = func.add_block();
        func.block_mut(entry).insts.push(Inst::Ret(None));

        let id = module.add_function(func);
        assert!(!module.function(id).is_declaration());
        assert_eq!(module.functions().count(), 1);
    }
}
