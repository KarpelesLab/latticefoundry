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
pub mod datalayout;
pub mod inst;
pub mod merge;
pub mod semantics;
pub mod text;
pub mod types;
pub mod value;

pub use inst::{
    AsmInput, AsmOutput, AsmSlot, AtomicOrdering, BinOp, CastOp, FastMath, Flags, FloatPred, InlineAsm,
    InstData, InstId, InstKind, IntPred,
    ReduceOp, RmwOp, SwitchCase, SwitchData, UnaryOp, Use,
};
pub use datalayout::{DataLayout, DataLayoutError, Endian, PointerSpec};
pub use merge::{MergeError, merge_modules};
pub use semantics::{EvalOutcome, FoldResult, SemValue, eval, fold};
pub use types::{FloatKind, FuncType, Layout, Type, TypeContext, TypeId};
pub use value::{AddrTarget, Const, ConstId, ConstPool, FloatBits, Value, ValueDef, ValueId};

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
    /// Handle to a [`Function`] within a [`Module`]: a dense index into its
    /// function list. Ids are stable until a function is removed
    /// ([`Module::remove_functions`], or the `dfe` pass of the `-O` pipelines),
    /// which compacts the list and renumbers the later functions.
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
///
/// The global's linkage and constness live in its [`GlobalAttrs`], kept by the
/// [`Module`] beside the global (see [`Module::define_global`] /
/// [`Module::global_attrs`]) so that this struct keeps its three public fields
/// and existing builder-API callers (`Global { name, ty, init }`) stay valid.
#[derive(Clone, Debug)]
pub struct Global {
    /// The interned symbol name of the global.
    pub name: Sym,
    /// The type of the value stored in the global.
    pub ty: TypeId,
    /// The initializer constant, if the global is defined here.
    pub init: Option<ConstId>,
}

/// How a *defined* global's symbol is bound in the emitted object.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Linkage {
    /// Visible to other objects (an ELF `STB_GLOBAL` symbol). The default.
    #[default]
    External,
    /// Private to this module (an ELF `STB_LOCAL` symbol); other objects may
    /// define the same name independently.
    Internal,
    /// Visible, but yields to a strong definition elsewhere (`STB_WEAK`).
    Weak,
}

/// A symbol's **visibility**: which *linked components* (the executable and each
/// shared object) can see and preempt it, independent of its [`Linkage`]
/// (`docs/ir-design.md` §4b). Maps onto the ELF `st_other` visibility field.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, PartialOrd, Ord)]
pub enum Visibility {
    /// `STV_DEFAULT`: exported from the component and **preemptible** — another
    /// component (e.g. the executable, or an `LD_PRELOAD`ed library) may
    /// interpose its own definition, so position-independent code reaches it
    /// through the GOT/PLT. The default.
    #[default]
    Default,
    /// `STV_PROTECTED`: exported, but references from inside the defining
    /// component always bind to its own definition (not preemptible).
    Protected,
    /// `STV_HIDDEN`: not exported from the linked component at all; binds
    /// locally, so references need no GOT/PLT indirection.
    Hidden,
}

impl Visibility {
    /// The more constraining of two visibilities (the ELF gABI rule when several
    /// references/definitions of one symbol meet: hidden beats protected beats
    /// default).
    pub fn most_constraining(self, other: Visibility) -> Visibility {
        self.max(other)
    }
}

/// Per-global attributes beyond name/type/initializer (`docs/ir-design.md` §4a).
///
/// - `linkage` picks the object symbol binding of a definition. A global with no
///   initializer is always an external *reference* whatever its linkage.
/// - `visibility` picks the symbol's ELF visibility (definitions *and*
///   references; a hidden reference promises the definition is in the same
///   linked component).
/// - `constant` promises the program never stores to the global, so the backend
///   places it in read-only data (`.rodata`); a store to it faults at run time.
/// - `detached` says the global's **storage is supplied outside the IR** (for
///   example a frontend that serializes its own data section): the backend emits
///   neither storage nor a symbol definition for it, and treats it exactly like
///   a declaration, whatever its initializer. The initializer then only serves
///   analyses and keeps the global well-typed. This is what
///   [`Module::add_global`] records, preserving the pre-data-emission meaning of
///   that API for existing builder clients.
/// - `secret` says the global's contents are secret (Lode's `secret[T]`): every
///   load whose address is based on it yields a secret-derived value for the
///   constant-time discipline (`docs/ir-design.md` §6d). Its *address* is
///   public. No effect on layout or emission.
/// - `thread_local` gives every thread its own instance of the global
///   (`docs/ir-design.md` §4c): `@x` is the address of the *current thread's*
///   copy, so it is not a link-time constant (no address constant may name a
///   thread-local global) and it lives in address space 0. The backend places
///   it in `.tdata`/`.tbss` and addresses it through the thread pointer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct GlobalAttrs {
    /// The symbol binding of a definition.
    pub linkage: Linkage,
    /// The symbol visibility.
    pub visibility: Visibility,
    /// Read-only storage (`.rodata`).
    pub constant: bool,
    /// Storage is provided outside the IR; the backend emits nothing for it.
    pub detached: bool,
    /// The contents are secret (constant-time discipline).
    pub secret: bool,
    /// One instance per thread (thread-local storage).
    pub thread_local: bool,
}

impl GlobalAttrs {
    /// External, mutable, backend-emitted: the attributes of a plain `.lf`
    /// `global @x : T = c` definition.
    pub const DEFAULT: GlobalAttrs = GlobalAttrs {
        linkage: Linkage::External,
        visibility: Visibility::Default,
        constant: false,
        detached: false,
        secret: false,
        thread_local: false,
    };

    /// The attributes [`Module::add_global`] records: external, mutable, and
    /// [`detached`](GlobalAttrs::detached).
    pub const DETACHED: GlobalAttrs = GlobalAttrs {
        linkage: Linkage::External,
        visibility: Visibility::Default,
        constant: false,
        detached: true,
        secret: false,
        thread_local: false,
    };
}

/// Per-function attributes (`docs/ir-design.md` §4b, §6d), kept in
/// [`Function::attrs`].
///
/// - `linkage` picks the symbol binding of a *definition* (external →
///   `STB_GLOBAL`, internal → `STB_LOCAL`, weak → `STB_WEAK`); a body-less
///   declaration is always an external reference.
/// - `visibility` picks the ELF symbol visibility, for definitions and
///   references alike.
/// - the **secrecy** of the parameters and of the return value (§6d): a
///   `secret` parameter carries a secret-derived value into the function (the
///   constant-time verifier then forbids it from reaching a branch, an address
///   or a divisor); a `secret` return lets the function return a
///   secret-derived value, and makes every direct call's result secret-derived
///   in the caller. Secrecy is part of the function's *interface*: a caller
///   may pass a secret-derived argument only to a secret parameter.
///
/// Every functional rebuild ([`Module::map_function`]) carries the attributes
/// over to the fresh function.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct FuncAttrs {
    /// The symbol binding of a definition.
    pub linkage: Linkage,
    /// The symbol visibility.
    pub visibility: Visibility,
    /// `secret_params[i]` marks parameter `i` secret; indices past the end are
    /// public (so the default, empty vector means "no secret parameter"). The
    /// vector never ends in `false`, so equal secrecy compares equal.
    secret_params: Vec<bool>,
    /// Whether the return value is secret.
    pub secret_ret: bool,
}

impl FuncAttrs {
    /// External linkage, default visibility, nothing secret: a plain `func`.
    pub const DEFAULT: FuncAttrs = FuncAttrs {
        linkage: Linkage::External,
        visibility: Visibility::Default,
        secret_params: Vec::new(),
        secret_ret: false,
    };

    /// The given linkage and visibility, nothing secret.
    pub const fn new(linkage: Linkage, visibility: Visibility) -> FuncAttrs {
        FuncAttrs { linkage, visibility, secret_params: Vec::new(), secret_ret: false }
    }

    /// Whether parameter `i` is secret.
    pub fn is_param_secret(&self, i: usize) -> bool {
        self.secret_params.get(i).copied().unwrap_or(false)
    }

    /// Mark parameter `i` secret (or public).
    pub fn set_param_secret(&mut self, i: usize, secret: bool) {
        if self.secret_params.len() <= i {
            if !secret {
                return;
            }
            self.secret_params.resize(i + 1, false);
        }
        self.secret_params[i] = secret;
        while self.secret_params.last() == Some(&false) {
            self.secret_params.pop();
        }
    }

    /// The indices of the secret parameters, ascending.
    pub fn secret_params(&self) -> impl Iterator<Item = usize> + '_ {
        self.secret_params.iter().enumerate().filter(|(_, s)| **s).map(|(i, _)| i)
    }

    /// Whether any parameter or the return value is secret.
    pub fn has_secrets(&self) -> bool {
        self.secret_ret || self.secret_params.iter().any(|&s| s)
    }
}

/// A translation unit: the top-level container of IR.
///
/// The module owns the shared interning tables — the [`TypeContext`] and the
/// [`ConstPool`] — so that types and constants are hash-consed across every
/// function (tenet T5).
///
/// A module may name its **target** (an informational string such as
/// `"x86_64"`, carried through the text and binary forms) and carries a
/// [`DataLayout`] (inside its [`TypeContext`]; LP64 unless set), which every
/// size, alignment and pointer-width question about its types follows
/// (`docs/ir-design.md` §3a).
#[derive(Clone, Debug, Default)]
pub struct Module {
    /// Human-readable module identifier (typically the source file name).
    pub name: String,
    target: Option<String>,
    types: TypeContext,
    consts: ConstPool,
    globals: Vec<Global>,
    /// `global_attrs[g]` holds the attributes of `globals[g]` (parallel vector).
    global_attrs: Vec<GlobalAttrs>,
    /// `global_addr_space[g]` is the address space `globals[g]` lives in
    /// (parallel vector; `0` unless set).
    global_addr_space: Vec<u32>,
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

    /// The target this module was written for, if it names one (the `.lf`
    /// `target "…"` declaration). Informational: it does not change the meaning
    /// of the IR, which the [data layout](Module::data_layout) pins down.
    pub fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }

    /// Set (or clear) the module's target name.
    pub fn set_target(&mut self, target: Option<String>) {
        self.target = target;
    }

    /// The module's data layout (see [`TypeContext::data_layout`]).
    pub fn data_layout(&self) -> &DataLayout {
        self.types.data_layout()
    }

    /// Set the module's data layout. Every size, alignment and offset query —
    /// including the builder's `struct_field`/`array_elem` helpers — follows it
    /// from then on; set it before building code that depends on layout.
    pub fn set_data_layout(&mut self, layout: DataLayout) {
        self.types.set_data_layout(layout);
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

    /// The attributes (linkage, visibility, parameter / return secrecy) of a
    /// function: its [`Function::attrs`].
    pub fn func_attrs(&self, id: FuncId) -> &FuncAttrs {
        &self.functions[id.index()].attrs
    }

    /// Mark parameter `param` of function `id` secret (or public).
    pub fn set_param_secret(&mut self, id: FuncId, param: usize, secret: bool) {
        self.functions[id.index()].attrs.set_param_secret(param, secret);
    }

    /// Mark the return value of function `id` secret (or public).
    pub fn set_ret_secret(&mut self, id: FuncId, secret: bool) {
        self.functions[id.index()].attrs.secret_ret = secret;
    }

    /// Whether the module declares any secret at all: a secret parameter or
    /// return, a secret global, or a `secret` load/store. A module without one
    /// is trivially constant-time and the constant-time verifier skips it.
    pub fn has_secrets(&self) -> bool {
        self.functions.iter().any(|f| f.attrs.has_secrets())
            || self.global_attrs.iter().any(|a| a.secret)
            || self.functions.iter().any(|f| {
                f.insts.iter().any(|i| {
                    matches!(
                        i.kind,
                        InstKind::Load { secret: true, .. } | InstKind::Store { secret: true, .. }
                    )
                })
            })
    }

    /// Append a global whose storage is supplied **outside the IR**, returning
    /// its handle. The global gets [`GlobalAttrs::DETACHED`]: the backend emits
    /// no storage or symbol definition for it (its initializer, if any, only
    /// types it). This is the original builder API, kept with its original
    /// meaning — a frontend that appends its own data section keeps working.
    /// Use [`Module::define_global`] to have the backend emit the global.
    pub fn add_global(&mut self, global: Global) -> GlobalId {
        self.define_global(global, GlobalAttrs::DETACHED)
    }

    /// Append a global with explicit attributes, returning its handle. With
    /// [`GlobalAttrs::DEFAULT`] and an initializer this is an ordinary external,
    /// mutable definition that the backend lays out in `.data`/`.bss`.
    pub fn define_global(&mut self, global: Global, attrs: GlobalAttrs) -> GlobalId {
        let id = GlobalId::from_index(self.globals.len());
        self.globals.push(global);
        self.global_attrs.push(attrs);
        self.global_addr_space.push(0);
        id
    }

    /// The address space a global lives in (`0` unless set). A reference to
    /// the global (`@x` as an operand, or `ptr @x` in an initializer) is a
    /// pointer into this space.
    pub fn global_addr_space(&self, id: GlobalId) -> u32 {
        self.global_addr_space[id.index()]
    }

    /// Place a global in address space `addr_space` (e.g. AVR program memory).
    /// Set it before building references to the global: the builder types
    /// `global_ref` by it.
    pub fn set_global_addr_space(&mut self, id: GlobalId, addr_space: u32) {
        self.global_addr_space[id.index()] = addr_space;
    }

    /// The type of a reference to global `id`: a pointer into its address space.
    pub fn global_ref_type(&mut self, id: GlobalId) -> TypeId {
        let space = self.global_addr_space(id);
        self.types.ptr_in(space)
    }

    /// The type of a reference to a function: a pointer into the data layout's
    /// program address space.
    pub fn func_ref_type(&mut self) -> TypeId {
        let space = self.types.data_layout().program_addr_space();
        self.types.ptr_in(space)
    }

    /// Borrow a global by handle.
    pub fn global(&self, id: GlobalId) -> &Global {
        &self.globals[id.index()]
    }

    /// The attributes of a global.
    pub fn global_attrs(&self, id: GlobalId) -> GlobalAttrs {
        self.global_attrs[id.index()]
    }

    /// Replace the attributes of a global.
    pub fn set_global_attrs(&mut self, id: GlobalId, attrs: GlobalAttrs) {
        self.global_attrs[id.index()] = attrs;
    }

    /// Set (or clear) a global's initializer — e.g. to attach an initializer that
    /// takes the address of a global or function created after it.
    pub fn set_global_init(&mut self, id: GlobalId, init: Option<ConstId>) {
        self.globals[id.index()].init = init;
    }

    /// Number of globals in the module.
    pub fn global_count(&self) -> usize {
        self.globals.len()
    }

    /// Borrow a function by handle.
    pub fn function(&self, id: FuncId) -> &Function {
        &self.functions[id.index()]
    }

    /// Replace a function's attributes (linkage, visibility, secrecy).
    pub fn set_func_attrs(&mut self, id: FuncId, attrs: FuncAttrs) {
        self.functions[id.index()].attrs = attrs;
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
        let Module { types, consts, functions, global_addr_space, .. } = self;
        builder::FunctionBuilder::new(&mut functions[func.index()], types, consts)
            .with_global_spaces(global_addr_space)
    }

    /// The number of functions declared in this module.
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// Functional-rebuild primitive (tenet T5): construct a **fresh** function
    /// that shares this module's interning tables, hand `build` an immutable view
    /// of the *old* function alongside a [`builder::FunctionBuilder`] over the
    /// fresh one, and return the freshly built function together with `build`'s
    /// own return value. The module is left untouched — the caller decides
    /// whether to install the result with [`Module::replace_function`].
    ///
    /// The fresh function inherits the old one's attributes and `decl_line`.
    /// Instruction lines are the rebuild's job: call
    /// [`builder::FunctionBuilder::set_line_from`] before copying or replacing
    /// each old instruction so the rebuilt body keeps its source lines.
    ///
    /// This is the backbone of the [`crate::transform`] layer: transforms read
    /// the old body and reconstruct it, rather than performing fragile in-place
    /// surgery on the arena.
    pub fn map_function<R>(
        &mut self,
        id: FuncId,
        build: impl FnOnce(&Function, &mut builder::FunctionBuilder<'_>) -> R,
    ) -> (Function, R) {
        let Module { types, consts, functions, global_addr_space, .. } = self;
        let old = &functions[id.index()];
        let mut fresh = Function::new(old.name, old.sig);
        fresh.attrs = old.attrs.clone();
        fresh.decl_line = old.decl_line;
        let r = {
            let mut b = builder::FunctionBuilder::new(&mut fresh, types, consts)
                .with_global_spaces(global_addr_space);
            build(old, &mut b)
        };
        (fresh, r)
    }

    /// Interprocedural functional-rebuild primitive (tenet T5): like
    /// [`Module::map_function`], but the `build` closure additionally receives an
    /// immutable view of **every** function in the module, so a rebuild can *read*
    /// callee bodies while it reconstructs the caller.
    ///
    /// The three borrows are disjoint: `caller` and `funcs` are shared borrows of
    /// the function arena (the caller is `funcs[id]`), while the
    /// [`builder::FunctionBuilder`] holds `&mut` of the *fresh* function together
    /// with the shared type/constant tables. This is what makes the
    /// [`crate::transform::inline`] pass — which must splice one function's blocks
    /// into another — borrow-clean without cloning callee bodies into an owned
    /// form. The module is left untouched; install the result with
    /// [`Module::replace_function`].
    pub fn map_function_reading<R>(
        &mut self,
        id: FuncId,
        build: impl FnOnce(&Function, &[Function], &mut builder::FunctionBuilder<'_>) -> R,
    ) -> (Function, R) {
        let Module { types, consts, functions, global_addr_space, .. } = self;
        let funcs: &[Function] = functions.as_slice();
        let caller = &funcs[id.index()];
        let mut fresh = Function::new(caller.name, caller.sig);
        fresh.attrs = caller.attrs.clone();
        fresh.decl_line = caller.decl_line;
        let r = {
            let mut b = builder::FunctionBuilder::new(&mut fresh, types, consts)
                .with_global_spaces(global_addr_space);
            build(caller, funcs, &mut b)
        };
        (fresh, r)
    }

    /// Install `func` as the body of function `id`, replacing whatever was there.
    /// Paired with [`Module::map_function`] to commit a functional rebuild.
    pub fn replace_function(&mut self, id: FuncId, func: Function) {
        self.functions[id.index()] = func;
    }

    /// Install `func` as the body of function `id` and return the body it
    /// replaces — so a candidate rebuild can be checked in place (e.g. by the
    /// constant-time verifier) and then swapped back out.
    pub fn swap_function(&mut self, id: FuncId, func: Function) -> Function {
        std::mem::replace(&mut self.functions[id.index()], func)
    }

    // --- function enumeration, references and removal -----------------------

    /// The id of every function, in definition order (`0..function_count()`).
    pub fn func_ids(&self) -> impl Iterator<Item = FuncId> + use<> {
        (0..self.functions.len()).map(FuncId::from_index)
    }

    /// The first function named `name`, if any.
    pub fn function_by_name(&self, name: Sym) -> Option<FuncId> {
        self.functions.iter().position(|f| f.name == name).map(FuncId::from_index)
    }

    /// The functions the body of `id` references, ascending and without
    /// duplicates: every `func_ref` value with at least one use (a direct call
    /// target or an address taken), and every function named by an address
    /// constant the body uses. A declaration references nothing.
    ///
    /// Together with [`Module::global_referenced_functions`] this is the whole
    /// IR-level reference graph, which a frontend can walk from its entry point
    /// to find the functions it may [remove](Module::remove_functions).
    /// References by *symbol name* outside the IR (a global or data relocation
    /// naming the function, an inline-asm template) are not seen.
    pub fn referenced_functions(&self, id: FuncId) -> Vec<FuncId> {
        let f = &self.functions[id.index()];
        let mut out = Vec::new();
        for (v, val) in f.values.iter().enumerate() {
            if f.uses[v].is_empty() {
                continue;
            }
            match val.def {
                ValueDef::Func(g) => out.push(g),
                ValueDef::Const(c) => self.const_func_addrs(c, &mut out),
                _ => {}
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The functions whose address the initializer of global `id` contains
    /// (through [`Const::Addr`], at any aggregate depth), ascending and without
    /// duplicates.
    pub fn global_referenced_functions(&self, id: GlobalId) -> Vec<FuncId> {
        let mut out = Vec::new();
        if let Some(init) = self.globals[id.index()].init {
            self.const_func_addrs(init, &mut out);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Push every function an (aggregate) constant addresses onto `out`.
    fn const_func_addrs(&self, c: ConstId, out: &mut Vec<FuncId>) {
        let mut stack = vec![c];
        while let Some(c) = stack.pop() {
            match self.consts.get(c) {
                Const::Addr { target: AddrTarget::Func(f), .. } => out.push(*f),
                Const::Aggregate { elems, .. } => stack.extend(elems.iter().copied()),
                _ => {}
            }
        }
    }

    /// Remove one function; see [`Module::remove_functions`].
    pub fn remove_function(&mut self, id: FuncId) -> Result<Vec<Option<FuncId>>, RemoveFunctionError> {
        self.remove_functions(&[id])
    }

    /// Remove the functions `ids` from the module and **compact** the function
    /// list, returning the renumbering: `remap[old.index()]` is the new id of a
    /// surviving function, `None` for a removed one.
    ///
    /// [`FuncId`]s are dense indices into the function list, so removal shifts
    /// every later function down: **any `FuncId` held outside the module must be
    /// translated through the returned map** (or re-looked-up, e.g. with
    /// [`Module::function_by_name`]). Inside the module the renumbering is
    /// complete — every `func_ref` value and every function-address constant is
    /// rewritten — so the module stays well-formed and the relative order of the
    /// surviving functions is unchanged. Duplicates in `ids` are fine; an empty
    /// `ids` is a no-op returning the identity map.
    ///
    /// Removal is refused, leaving the module untouched, if an id is out of
    /// range or a removed function is still referenced by a surviving function
    /// (a `func_ref` value with uses, or a used address constant; see
    /// [`Module::referenced_functions`]) or by any global initializer.
    /// References *among* the removed functions do not count, so a dead cycle
    /// can be removed in one call. A stale, use-less reference to a removed
    /// function in a surviving body, or an address constant naming it that
    /// nothing uses, is tombstoned to a `poison` of the same pointer type.
    pub fn remove_functions(&mut self, ids: &[FuncId]) -> Result<Vec<Option<FuncId>>, RemoveFunctionError> {
        let n = self.functions.len();
        let mut dead = vec![false; n];
        for &id in ids {
            if id.index() >= n {
                return Err(RemoveFunctionError::OutOfRange(id));
            }
            dead[id.index()] = true;
        }
        // Every reference to a removed function must come from a removed one.
        for f in self.func_ids() {
            if dead[f.index()] {
                continue;
            }
            if let Some(&g) = self.referenced_functions(f).iter().find(|g| dead[g.index()]) {
                return Err(RemoveFunctionError::StillReferenced { func: g, by: Referrer::Function(f) });
            }
        }
        for gi in 0..self.globals.len() {
            let gid = GlobalId::from_index(gi);
            if let Some(&g) = self.global_referenced_functions(gid).iter().find(|g| dead[g.index()]) {
                return Err(RemoveFunctionError::StillReferenced { func: g, by: Referrer::Global(gid) });
            }
        }

        let mut remap: Vec<Option<FuncId>> = Vec::with_capacity(n);
        let mut next = 0;
        for &d in &dead {
            if d {
                remap.push(None);
            } else {
                remap.push(Some(FuncId::from_index(next)));
                next += 1;
            }
        }
        if next == n {
            return Ok(remap);
        }

        // Drop the removed bodies, then renumber what references functions.
        let mut i = 0;
        self.functions.retain(|_| {
            let keep = !dead[i];
            i += 1;
            keep
        });
        self.consts.remap_func_addrs(&remap);
        let Module { functions, consts, .. } = self;
        for f in functions.iter_mut() {
            let mut touched = false;
            for v in 0..f.values.len() {
                let ValueDef::Func(g) = f.values[v].def else {
                    continue;
                };
                touched = true;
                f.values[v].def = match remap[g.index()] {
                    Some(ng) => ValueDef::Func(ng),
                    // Unused (checked above): tombstone it.
                    None => ValueDef::Const(consts.intern(Const::Poison(f.values[v].ty))),
                };
            }
            if touched {
                // Rebuild the reference dedup table; the first value with a
                // given definition wins, as it did when the cache was filled.
                f.value_cache.clear();
                for (v, val) in f.values.iter().enumerate() {
                    if matches!(val.def, ValueDef::Const(_) | ValueDef::Global(_) | ValueDef::Func(_)) {
                        f.value_cache.entry(val.def.clone()).or_insert(ValueId::from_index(v));
                    }
                }
            }
        }
        Ok(remap)
    }
}

/// What holds a reference that keeps a function from being removed
/// ([`RemoveFunctionError::StillReferenced`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Referrer {
    /// The body of a surviving function.
    Function(FuncId),
    /// The initializer of a global.
    Global(GlobalId),
}

/// Why [`Module::remove_functions`] refused to remove a set of functions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RemoveFunctionError {
    /// The id does not name a function of the module.
    OutOfRange(FuncId),
    /// Function `func` is still referenced by `by`, which is not removed.
    StillReferenced {
        /// The function that was to be removed.
        func: FuncId,
        /// What still references it.
        by: Referrer,
    },
}

impl std::fmt::Display for RemoveFunctionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemoveFunctionError::OutOfRange(id) => write!(f, "function id {} is out of range", id.index()),
            RemoveFunctionError::StillReferenced { func, by } => {
                write!(f, "function {} is still referenced by ", func.index())?;
                match by {
                    Referrer::Function(g) => write!(f, "function {}", g.index()),
                    Referrer::Global(g) => write!(f, "the initializer of global {}", g.index()),
                }
            }
        }
    }
}

impl std::error::Error for RemoveFunctionError {}

/// A function definition or declaration.
///
/// A function with no blocks is an external declaration; a function with at
/// least one block is a definition whose entry block holds the function's
/// parameters. The function owns the flat arenas its ids address: the value
/// table, the instruction arena, the block list, and the per-value use lists.
#[derive(Clone, Debug)]
pub struct Function {
    /// The interned symbol name of the function.
    pub name: Sym,
    /// The function's signature: a [`Type::Func`] type id.
    pub sig: TypeId,
    /// The function's linkage, visibility and secrecy (external, default, public
    /// unless set).
    pub attrs: FuncAttrs,
    /// The 1-based source line the function is declared on, if known (debug
    /// info, tenet: optional so non-debug builds are unaffected). `None` when the
    /// function was not built from a source with line provenance.
    pub decl_line: Option<u32>,
    values: Vec<Value>,
    /// `uses[v]` is the def→use list of value `v` (parallel to `values`).
    uses: Vec<Vec<Use>>,
    insts: Vec<InstData>,
    /// Optional per-instruction source-line side table, parallel to `insts`.
    /// Entry `0` means "no line recorded". Populated only when building with
    /// debug info (e.g. the `.lf` parser under `lf build -g`); empty otherwise,
    /// so ordinary builds carry no overhead.
    inst_lines: Vec<u32>,
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
            attrs: FuncAttrs::DEFAULT,
            decl_line: None,
            values: Vec::new(),
            uses: Vec::new(),
            insts: Vec::new(),
            inst_lines: Vec::new(),
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

    /// The recorded 1-based source line of an instruction, if debug provenance
    /// was attached when the function was built. Returns `None` when no line was
    /// recorded (a non-debug build, or an instruction the builder synthesized).
    pub fn inst_line(&self, id: InstId) -> Option<u32> {
        match self.inst_lines.get(id.index()).copied() {
            Some(0) | None => None,
            Some(line) => Some(line),
        }
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
#[derive(Clone, Debug, Default)]
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
pub(crate) mod tests;
#[cfg(test)]
mod vector_tests;
#[cfg(test)]
mod asm_tests;
#[cfg(test)]
pub(crate) mod refexec;
