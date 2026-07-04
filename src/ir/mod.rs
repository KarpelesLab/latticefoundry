//! The LatticeFoundry intermediate representation (IR).
//!
//! A typed, SSA-based, target-independent IR in the LLVM-like role but of an
//! independent design. The container hierarchy is
//! `Module → Function → Block → Instruction`, all arena-allocated and referenced
//! by small `Copy` id newtypes (`FuncId`, `BlockId`, `ValueId`, `InstId`, ...),
//! never by interior pointers (tenet T5). This keeps the graph friendly to
//! Rust's ownership model and to later incremental/parallel processing.
//!
//! The design departs from LLVM in the ways recorded in `docs/ir-design.md`:
//!
//! - **Block arguments, not φ-nodes.** A [`Block`] carries a typed *parameter*
//!   list; each terminator supplies an *argument* list per successor edge. A
//!   function's parameters are its entry block's parameters. There are no phi
//!   instructions.
//! - **Poison + freeze, no `undef`** ([`value`]).
//! - **Opaque pointers + explicit offset addressing** ([`inst`], [`builder`]).
//! - **One unified flag model** ([`inst::Flags`]).
//! - Types and constants are **interned/hash-consed** from day one ([`types`],
//!   [`value`]).
//!
//! Build IR with the [`builder::FunctionBuilder`] obtained from
//! [`Module::build`]; it keeps use/def edges consistent and offers the
//! `struct_field`/`array_elem` offset helpers and `replace_all_uses_with`.

pub mod binary;
pub mod builder;
pub mod inst;
pub mod semantics;
pub mod text;
pub mod types;
pub mod value;

pub use inst::{
    BinOp, CastOp, FastMath, Flags, FloatPred, InstData, InstId, InstKind, IntPred, SwitchCase,
    SwitchData, UnaryOp, Use,
};
pub use semantics::{EvalOutcome, FoldResult, SemValue, eval, fold};
pub use types::{FloatKind, FuncType, Layout, Type, TypeContext, TypeId};
pub use value::{Const, ConstId, ConstPool, FloatBits, Value, ValueDef, ValueId};

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

            #[inline]
            #[allow(dead_code)]
            pub(crate) fn from_index(i: usize) -> Self {
                $name(i as u32)
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
    /// Handle to a [`Global`] within a [`Module`].
    GlobalId
);

/// A module-level global variable: a named, typed storage cell whose address is
/// a pointer value in the IR.
#[derive(Debug)]
pub struct Global {
    /// The interned symbol name of the global.
    pub name: Sym,
    /// The type of the value stored in the global.
    pub ty: TypeId,
    /// The initializer constant, if the global is defined here.
    pub init: Option<ConstId>,
}

/// A translation unit: the top-level container of IR.
///
/// The module owns the shared interning tables — the [`TypeContext`] and the
/// [`ConstPool`] — so that types and constants are hash-consed across every
/// function (tenet T5).
#[derive(Debug, Default)]
pub struct Module {
    /// Human-readable module identifier (typically the source file name).
    pub name: String,
    types: TypeContext,
    consts: ConstPool,
    globals: Vec<Global>,
    functions: Vec<Function>,
}

impl Module {
    /// Create an empty module with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), ..Self::default() }
    }

    /// The module's type-interning context.
    pub fn types(&self) -> &TypeContext {
        &self.types
    }

    /// The module's type-interning context, mutably (to intern new types).
    pub fn types_mut(&mut self) -> &mut TypeContext {
        &mut self.types
    }

    /// The module's constant-interning pool.
    pub fn consts(&self) -> &ConstPool {
        &self.consts
    }

    /// Intern a constant into the module pool.
    pub fn intern_const(&mut self, c: Const) -> ConstId {
        self.consts.intern(c)
    }

    /// Declare a function with the given name and signature (a `Func` type id)
    /// and no body, returning its handle. Add blocks via [`Module::build`].
    pub fn declare_function(&mut self, name: Sym, sig: TypeId) -> FuncId {
        let id = FuncId::from_index(self.functions.len());
        self.functions.push(Function::new(name, sig));
        id
    }

    /// Append a global, returning its handle.
    pub fn add_global(&mut self, global: Global) -> GlobalId {
        let id = GlobalId::from_index(self.globals.len());
        self.globals.push(global);
        id
    }

    /// Borrow a global by handle.
    pub fn global(&self, id: GlobalId) -> &Global {
        &self.globals[id.index()]
    }

    /// Borrow a function by handle.
    pub fn function(&self, id: FuncId) -> &Function {
        &self.functions[id.index()]
    }

    /// Iterate over every function in definition order.
    pub fn functions(&self) -> impl Iterator<Item = &Function> {
        self.functions.iter()
    }

    /// Iterate over every global in definition order.
    pub fn globals(&self) -> impl Iterator<Item = &Global> {
        self.globals.iter()
    }

    /// Open a [`builder::FunctionBuilder`] on the given function, borrowing the
    /// shared type context and constant pool alongside it.
    pub fn build(&mut self, func: FuncId) -> builder::FunctionBuilder<'_> {
        let Module { types, consts, functions, .. } = self;
        builder::FunctionBuilder::new(&mut functions[func.index()], types, consts)
    }
}

/// A function definition or declaration.
///
/// A function with no blocks is an external declaration; a function with at
/// least one block is a definition whose entry block holds the function's
/// parameters. The function owns the flat arenas its ids address: the value
/// table, the instruction arena, the block list, and the per-value use lists.
#[derive(Debug)]
pub struct Function {
    /// The interned symbol name of the function.
    pub name: Sym,
    /// The function's signature: a [`Type::Func`] type id.
    pub sig: TypeId,
    values: Vec<Value>,
    /// `uses[v]` is the def→use list of value `v` (parallel to `values`).
    uses: Vec<Vec<Use>>,
    insts: Vec<InstData>,
    blocks: Vec<Block>,
    entry: Option<BlockId>,
    /// Dedup table for value-less-identity values (constants, global and
    /// function references), so equal references share one [`ValueId`].
    value_cache: std::collections::HashMap<ValueDef, ValueId>,
}

impl Function {
    /// Create a function with the given name and signature and no body.
    pub fn new(name: Sym, sig: TypeId) -> Self {
        Self {
            name,
            sig,
            values: Vec::new(),
            uses: Vec::new(),
            insts: Vec::new(),
            blocks: Vec::new(),
            entry: None,
            value_cache: std::collections::HashMap::new(),
        }
    }

    /// Whether this function is an external declaration (has no body).
    pub fn is_declaration(&self) -> bool {
        self.blocks.is_empty()
    }

    /// The entry block, if the function has a body.
    pub fn entry(&self) -> Option<BlockId> {
        self.entry
    }

    /// Borrow a block by handle.
    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.index()]
    }

    /// Iterate over every block in definition order, with its id.
    pub fn blocks(&self) -> impl Iterator<Item = (BlockId, &Block)> {
        self.blocks.iter().enumerate().map(|(i, b)| (BlockId::from_index(i), b))
    }

    /// Number of blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Borrow an instruction by handle.
    pub fn inst(&self, id: InstId) -> &InstData {
        &self.insts[id.index()]
    }

    /// Number of instructions in the arena.
    pub fn inst_count(&self) -> usize {
        self.insts.len()
    }

    /// Borrow a value by handle.
    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.index()]
    }

    /// The type of a value.
    pub fn value_type(&self, id: ValueId) -> TypeId {
        self.values[id.index()].ty
    }

    /// Number of values in the table.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// The def→use list of a value.
    pub fn uses_of(&self, id: ValueId) -> &[Use] {
        &self.uses[id.index()]
    }

    // --- internal mutation used by the builder ------------------------------

    /// Allocate a fresh value, returning its id. Grows the parallel use list.
    fn push_value(&mut self, def: ValueDef, ty: TypeId) -> ValueId {
        let id = ValueId::from_index(self.values.len());
        self.values.push(Value { def, ty });
        self.uses.push(Vec::new());
        id
    }

    /// Get the existing value for a dedupable reference (constant / global /
    /// function ref), or create one. Instruction results and block parameters
    /// are unique and never routed through here.
    fn get_or_make_value(&mut self, def: ValueDef, ty: TypeId) -> ValueId {
        if let Some(&v) = self.value_cache.get(&def) {
            return v;
        }
        let v = self.push_value(def.clone(), ty);
        self.value_cache.insert(def, v);
        v
    }
}

/// A basic block: a straight-line instruction sequence with a typed parameter
/// list and (once complete) a single terminating instruction.
///
/// The parameters are the block's SSA arguments — the block-argument encoding
/// that replaces φ-nodes. Predecessors supply matching argument lists on their
/// terminators.
#[derive(Debug, Default)]
pub struct Block {
    params: Vec<ValueId>,
    insts: Vec<InstId>,
    terminator: Option<InstId>,
}

impl Block {
    /// The block's typed parameter values, in order.
    #[inline]
    pub fn params(&self) -> &[ValueId] {
        &self.params
    }

    /// The block's non-terminator instructions, in execution order.
    #[inline]
    pub fn insts(&self) -> &[InstId] {
        &self.insts
    }

    /// The block's terminator, once set.
    #[inline]
    pub fn terminator(&self) -> Option<InstId> {
        self.terminator
    }

    /// Whether the block has been terminated.
    #[inline]
    pub fn is_terminated(&self) -> bool {
        self.terminator.is_some()
    }
}

#[cfg(test)]
mod tests;
