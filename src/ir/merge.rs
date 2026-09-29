//! IR-level module linking for link-time optimization (ROADMAP Phase 10).
//!
//! [`merge_modules`] (and its in-place primitive [`Module::link_module`]) combine
//! several IR [`Module`]s into one so the `-O` pipeline can then run over the whole
//! program — enabling **cross-module inlining** and interprocedural constant
//! folding that per-module compilation cannot see.
//!
//! ## What linking does
//!
//! Merging one module `other` into `self` is four remappings applied in order:
//!
//! 1. **Types.** Every [`TypeId`] of `other` is re-interned into `self`'s
//!    `TypeContext`. Because a composite type is always interned *after* the
//!    components it names, iterating `other`'s types in id order and remapping each
//!    component through the partial old→new map builds the full `type_map`
//!    left-to-right.
//! 2. **Constants.** Likewise every [`ConstId`] is re-interned into `self`'s
//!    `ConstPool`, remapping each constant's type and (for aggregates) its child
//!    constant ids through the maps already built.
//! 3. **Globals** and **functions** are matched by name for *cross-module symbol
//!    resolution*: a body-less declaration in one module unifies with the
//!    definition of the same name in another, and all references to the
//!    declaration are redirected to the definition. Two strong definitions of one
//!    name are an error ([`MergeError`]); a declaration with no definition anywhere
//!    stays an external (undefined) declaration. New symbols are appended.
//! 4. **Bodies.** Once the `func_map`/`global_map` are complete, each incoming
//!    function *definition* is deep-copied into its resolved slot with all
//!    interned references (types, constants, globals, functions) remapped. The
//!    function-local ids ([`ValueId`](crate::ir::ValueId)/`BlockId`/
//!    [`InstId`](crate::ir::InstId)) are preserved verbatim, since the whole value/
//!    instruction/block arena is copied in order.
//!
//! ## Symbol identity and determinism
//!
//! Symbols are compared by their interned [`Sym`] name, so **all** modules being
//! merged must have been parsed/decoded against the *same* `StrInterner`
//! (`crate::support::StrInterner`) — the LTO driver threads one interner through
//! every input. The merge is deterministic: types/constants/functions are
//! processed in id order, resolved symbols keep `self`'s existing id, and new
//! symbols are appended in the incoming module's order.

use std::collections::HashMap;

use crate::ir::inst::{InstData, InstKind};
use crate::ir::types::{FuncType, Type, TypeId};
use crate::ir::value::{AddrTarget, Const, ConstId, Value, ValueDef};
use crate::ir::{Block, Function, Global, GlobalAttrs, GlobalId, Linkage, Module};
use crate::support::Sym;

use super::FuncId;

/// A failure while linking IR modules: two modules each provide a *strong*
/// (defined) version of the same symbol, which cannot be unified, or the
/// modules were laid out for different machines.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MergeError {
    /// Two modules define a function with the same name. Carries the symbol.
    DuplicateFunction(Sym),
    /// Two modules define a global with the same name. Carries the symbol.
    DuplicateGlobal(Sym),
    /// The modules have different [data layouts](crate::ir::DataLayout) (or
    /// name different targets), so their types do not mean the same bytes.
    DataLayoutMismatch,
}

impl MergeError {
    /// The clashing symbol name, if the error is about a symbol.
    pub fn symbol(self) -> Option<Sym> {
        match self {
            MergeError::DuplicateFunction(s) | MergeError::DuplicateGlobal(s) => Some(s),
            MergeError::DataLayoutMismatch => None,
        }
    }
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::DuplicateFunction(s) => {
                write!(f, "duplicate definition of function (symbol #{})", s.index())
            }
            MergeError::DuplicateGlobal(s) => {
                write!(f, "duplicate definition of global (symbol #{})", s.index())
            }
            MergeError::DataLayoutMismatch => {
                write!(f, "modules have different targets or data layouts")
            }
        }
    }
}

impl std::error::Error for MergeError {}

/// Merge several modules into one, in order, resolving cross-module symbols.
///
/// The result takes `name`. The first module seeds the combination and each
/// subsequent module is [linked in](Module::link_module). All modules must share
/// one [`StrInterner`](crate::support::StrInterner) so names compare correctly.
pub fn merge_modules(
    modules: impl IntoIterator<Item = Module>,
    name: impl Into<String>,
) -> Result<Module, MergeError> {
    let mut combined = Module::new(name);
    for m in modules {
        combined.link_module(m)?;
    }
    Ok(combined)
}

impl Module {
    /// Link `other` into `self` in place (see the [module docs](self)).
    ///
    /// On success `self` contains the union of both modules with all of `other`'s
    /// references remapped and cross-module declarations resolved to their
    /// definitions. On a duplicate strong definition `self` is left partially
    /// modified and a [`MergeError`] is returned.
    pub fn link_module(&mut self, other: Module) -> Result<(), MergeError> {
        // 0. Both sides must describe the same machine. An empty module (the
        //    seed of `merge_modules`) adopts the incoming target and layout.
        if self.functions.is_empty() && self.globals.is_empty() {
            self.target = other.target.clone();
            self.types.set_data_layout(other.types.data_layout().clone());
        } else if self.types.data_layout() != other.types.data_layout()
            || (self.target.is_some() && other.target.is_some() && self.target != other.target)
        {
            return Err(MergeError::DataLayoutMismatch);
        } else if self.target.is_none() {
            self.target = other.target.clone();
        }

        // 1. Re-intern types in id order (components precede composites).
        let mut type_map: Vec<TypeId> = Vec::with_capacity(other.types.len());
        for ty in other.types.iter() {
            let remapped = remap_type(ty, &type_map);
            type_map.push(self.types.intern(remapped));
        }

        // 2. Resolve globals by name. Initializers are remapped in step 3, once
        //    the constant map exists (an address constant needs the global and
        //    function maps, so constants are re-interned *after* symbols).
        let mut self_globals: HashMap<Sym, GlobalId> = HashMap::new();
        for (i, g) in self.globals.iter().enumerate() {
            self_globals.insert(g.name, GlobalId::from_index(i));
        }
        let mut global_map: Vec<GlobalId> = Vec::with_capacity(other.globals.len());
        // (target, incoming global index) pairs whose type/init/attrs we install.
        let mut global_installs: Vec<(GlobalId, usize)> = Vec::new();
        for (gi, g) in other.globals.iter().enumerate() {
            let incoming = def_rank(g, other.global_attrs[gi]);
            let target = match self_globals.get(&g.name).copied() {
                Some(existing) => {
                    let i = existing.index();
                    let cur = def_rank(&self.globals[i], self.global_attrs[i]);
                    if cur == DefRank::Strong && incoming == DefRank::Strong {
                        return Err(MergeError::DuplicateGlobal(g.name));
                    }
                    // Upgrade to the stronger incoming definition; otherwise keep
                    // the existing definition / declaration.
                    if incoming > cur {
                        global_installs.push((existing, gi));
                    }
                    // Visibility merges to the most constraining of the two
                    // (ELF gABI); the install below keeps it.
                    let vis = other.global_attrs[gi].visibility;
                    let a = &mut self.global_attrs[i];
                    a.visibility = a.visibility.most_constraining(vis);
                    existing
                }
                None => {
                    // Placeholder type/init; the install in step 3 fills them in.
                    let id = self.define_global(
                        Global { name: g.name, ty: g.ty, init: None },
                        other.global_attrs[gi],
                    );
                    self.global_addr_space[id.index()] = other.global_addr_space[gi];
                    self_globals.insert(g.name, id);
                    global_installs.push((id, gi));
                    id
                }
            };
            global_map.push(target);
        }

        // 2b. Resolve functions by name, recording which incoming definitions must
        //     have their bodies installed. Two definitions of one name is an error.
        let mut self_funcs: HashMap<Sym, FuncId> = HashMap::new();
        for i in 0..self.functions.len() {
            self_funcs.insert(self.functions[i].name, FuncId::from_index(i));
        }
        let mut func_map: Vec<FuncId> = Vec::with_capacity(other.functions.len());
        // (incoming index, target id) pairs whose body we copy in step 4.
        let mut to_install: Vec<(usize, FuncId)> = Vec::new();
        for (i, f) in other.functions.iter().enumerate() {
            let sig = type_map[f.sig.index()];
            let incoming_def = !f.is_declaration();
            let target = match self_funcs.get(&f.name).copied() {
                Some(existing) => {
                    let cur_def = !self.functions[existing.index()].is_declaration();
                    match (cur_def, incoming_def) {
                        (true, true) => return Err(MergeError::DuplicateFunction(f.name)),
                        // Upgrade the existing declaration with the incoming body.
                        (false, true) => {
                            self.functions[existing.index()].sig = sig;
                            to_install.push((i, existing));
                        }
                        // Keep the existing definition / declaration.
                        (_, false) => {}
                    }
                    // The definition's linkage wins; visibility merges to the
                    // most constraining (ELF gABI).
                    let cur = &mut self.functions[existing.index()].attrs;
                    if incoming_def {
                        cur.linkage = f.attrs.linkage;
                    }
                    cur.visibility = cur.visibility.most_constraining(f.attrs.visibility);
                    existing
                }
                None => {
                    let id = self.declare_function(f.name, sig);
                    self.functions[id.index()].attrs = f.attrs;
                    self_funcs.insert(f.name, id);
                    if incoming_def {
                        to_install.push((i, id));
                    }
                    id
                }
            };
            func_map.push(target);
        }

        // 3. Re-intern constants in id order (children precede aggregates), then
        //    install the resolved globals' types, initializers, and attributes.
        let mut const_map: Vec<ConstId> = Vec::with_capacity(other.consts.len());
        for c in other.consts.iter() {
            let remapped = remap_const(c, &type_map, &const_map, &global_map, &func_map);
            const_map.push(self.consts.intern(remapped));
        }
        for (target, gi) in global_installs {
            let g = &other.globals[gi];
            let cur = &mut self.globals[target.index()];
            cur.ty = type_map[g.ty.index()];
            cur.init = g.init.map(|c| const_map[c.index()]);
            let vis = self.global_attrs[target.index()].visibility;
            let mut attrs = other.global_attrs[gi];
            attrs.visibility = attrs.visibility.most_constraining(vis);
            self.global_attrs[target.index()] = attrs;
            self.global_addr_space[target.index()] = other.global_addr_space[gi];
        }

        // 4. Copy function bodies with every reference remapped.
        for (src_idx, target) in to_install {
            let src = &other.functions[src_idx];
            let mut body = copy_body(src, &type_map, &const_map, &global_map, &func_map);
            // Keep the attributes resolved in step 2b.
            body.attrs = self.functions[target.index()].attrs;
            self.functions[target.index()] = body;
        }

        Ok(())
    }
}

/// Remap the component [`TypeId`]s of a [`Type`] through a partial old→new map
/// (only lower-id components, which are already present, are dereferenced).
fn remap_type(ty: &Type, type_map: &[TypeId]) -> Type {
    match ty {
        Type::Void | Type::Int(_) | Type::Float(_) | Type::Ptr | Type::PtrIn(_) => ty.clone(),
        Type::Array(elem, len) => Type::Array(type_map[elem.index()], *len),
        Type::Struct(fields) => {
            Type::Struct(fields.iter().map(|f| type_map[f.index()]).collect())
        }
        Type::Func(ft) => Type::Func(FuncType {
            params: ft.params.iter().map(|p| type_map[p.index()]).collect(),
            ret: type_map[ft.ret.index()],
            variadic: ft.variadic,
        }),
    }
}

/// How strongly a global defines its symbol, for cross-module resolution.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum DefRank {
    /// No initializer: a pure external declaration.
    Decl,
    /// A [`detached`](GlobalAttrs::detached) global: typed (maybe initialized)
    /// but its storage lives outside the IR, so it defines nothing here.
    Detached,
    /// A weak definition: yields to a strong one.
    Weak,
    /// A strong (external or internal) definition.
    Strong,
}

fn def_rank(g: &Global, attrs: GlobalAttrs) -> DefRank {
    match (g.init, attrs.detached, attrs.linkage) {
        (None, _, _) => DefRank::Decl,
        (Some(_), true, _) => DefRank::Detached,
        (Some(_), false, Linkage::Weak) => DefRank::Weak,
        (Some(_), false, Linkage::External | Linkage::Internal) => DefRank::Strong,
    }
}

/// Remap a [`Const`]'s type, (for aggregates) child constant ids, and (for
/// address constants) the addressed global/function.
fn remap_const(
    c: &Const,
    type_map: &[TypeId],
    const_map: &[ConstId],
    global_map: &[GlobalId],
    func_map: &[FuncId],
) -> Const {
    match c {
        Const::Int { ty, value } => Const::Int { ty: type_map[ty.index()], value: value.clone() },
        Const::Float { ty, bits } => Const::Float { ty: type_map[ty.index()], bits: *bits },
        Const::Null(ty) => Const::Null(type_map[ty.index()]),
        Const::Poison(ty) => Const::Poison(type_map[ty.index()]),
        Const::Aggregate { ty, elems } => Const::Aggregate {
            ty: type_map[ty.index()],
            elems: elems.iter().map(|e| const_map[e.index()]).collect(),
        },
        Const::Addr { ty, target, offset } => Const::Addr {
            ty: type_map[ty.index()],
            target: match target {
                AddrTarget::Global(g) => AddrTarget::Global(global_map[g.index()]),
                AddrTarget::Func(f) => AddrTarget::Func(func_map[f.index()]),
            },
            offset: *offset,
        },
    }
}

/// Deep-copy `src`'s body into a fresh [`Function`], remapping interned references
/// (types, constants, globals, functions) while preserving function-local ids.
fn copy_body(
    src: &Function,
    type_map: &[TypeId],
    const_map: &[ConstId],
    global_map: &[GlobalId],
    func_map: &[FuncId],
) -> Function {
    let mut f = Function::new(src.name, type_map[src.sig.index()]);
    f.decl_line = src.decl_line;
    f.entry = src.entry;
    f.inst_lines = src.inst_lines.clone();
    f.uses = src.uses.clone();

    // Values: remap the type and the interned kinds of the def.
    f.values = src
        .values
        .iter()
        .map(|v| Value {
            ty: type_map[v.ty.index()],
            def: match &v.def {
                ValueDef::Const(c) => ValueDef::Const(const_map[c.index()]),
                ValueDef::Global(g) => ValueDef::Global(global_map[g.index()]),
                ValueDef::Func(fid) => ValueDef::Func(func_map[fid.index()]),
                ValueDef::Inst(_) | ValueDef::Param(..) => v.def.clone(),
            },
        })
        .collect();

    // Instructions: remap the result type and any type embedded in the opcode.
    f.insts = src.insts.iter().map(|inst| remap_inst(inst, type_map)).collect();

    // Blocks are id-only structure; copy verbatim (their value/inst ids are local).
    f.blocks = src
        .blocks
        .iter()
        .map(|b| Block {
            params: b.params().to_vec(),
            insts: b.insts().to_vec(),
            terminator: b.terminator(),
        })
        .collect();

    // Rebuild the dedup cache for the value-less-identity values (whose defs the
    // remap above may have changed).
    for (i, v) in f.values.iter().enumerate() {
        if matches!(v.def, ValueDef::Const(_) | ValueDef::Global(_) | ValueDef::Func(_)) {
            f.value_cache.insert(v.def.clone(), crate::ir::ValueId::from_index(i));
        }
    }

    f
}

/// Copy an instruction, remapping the result type and any [`TypeId`] carried in
/// its opcode payload (`alloca`, `load`/`store` and the atomics). Value operands
/// are local ids and are preserved as-is.
fn remap_inst(inst: &InstData, type_map: &[TypeId]) -> InstData {
    let mut new = inst.clone();
    new.ty = type_map[inst.ty.index()];
    let m = |ty: &TypeId| type_map[ty.index()];
    new.kind = match &inst.kind {
        InstKind::Alloca { elem_ty } => InstKind::Alloca { elem_ty: m(elem_ty) },
        InstKind::Load { ty, align, volatile } => {
            InstKind::Load { ty: m(ty), align: *align, volatile: *volatile }
        }
        InstKind::Store { ty, align, volatile } => {
            InstKind::Store { ty: m(ty), align: *align, volatile: *volatile }
        }
        InstKind::AtomicLoad { ty, align, ordering } => {
            InstKind::AtomicLoad { ty: m(ty), align: *align, ordering: *ordering }
        }
        InstKind::AtomicStore { ty, align, ordering } => {
            InstKind::AtomicStore { ty: m(ty), align: *align, ordering: *ordering }
        }
        InstKind::AtomicRmw { op, ty, align, ordering } => {
            InstKind::AtomicRmw { op: *op, ty: m(ty), align: *align, ordering: *ordering }
        }
        InstKind::CmpXchg { ty, align, success, failure } => {
            InstKind::CmpXchg { ty: m(ty), align: *align, success: *success, failure: *failure }
        }
        other => other.clone(),
    };
    new
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::domains::ConstLattice;
    use crate::analysis::solver::solve;
    use crate::ir::inst::Flags;
    use crate::ir::value::{Const, ValueDef};
    use crate::ir::{FuncId, InstKind, text};
    use crate::support::StrInterner;
    use crate::transform::{OptLevel, optimize};
    use crate::verify::verify_module;

    /// Module A: `helper() -> i64 = 40` (a definition).
    fn module_a(syms: &mut StrInterner) -> Module {
        let mut m = Module::new("a");
        let i64t = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![], i64t, false);
        let helper = m.declare_function(syms.intern("helper"), sig);
        let mut b = m.build(helper);
        b.create_entry_block();
        let c = b.const_i64(i64t, 40);
        b.ret(Some(c));
        m
    }

    /// Module B: declares `helper` (external), defines `main() = helper() + 2`.
    fn module_b(syms: &mut StrInterner) -> Module {
        let mut m = Module::new("b");
        let i64t = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![], i64t, false);
        let helper = m.declare_function(syms.intern("helper"), sig); // declaration only
        let main = m.declare_function(syms.intern("main"), sig);
        let mut b = m.build(main);
        b.create_entry_block();
        let cref = b.func_ref(helper);
        let r = b.call(cref, &[], i64t).unwrap();
        let two = b.const_i64(i64t, 2);
        let s = b.add(r, two, Flags::NONE);
        b.ret(Some(s));
        m
    }

    /// The value the value-returning `ret` produces, as a constant lattice element.
    fn ret_const(m: &Module, f: FuncId) -> ConstLattice {
        let func = m.function(f);
        let res = solve::<ConstLattice>(func, m.types(), m.consts());
        for (_bid, blk) in func.blocks() {
            if let Some(t) = blk.terminator()
                && matches!(func.inst(t).kind, InstKind::Ret)
                && let Some(&v) = func.inst(t).operands().first()
            {
                return res.value(v).clone();
            }
        }
        panic!("no value-returning ret");
    }

    fn find_func(m: &Module, syms: &StrInterner, name: &str) -> FuncId {
        for i in 0..m.function_count() {
            if syms.resolve(m.function(FuncId::from_index(i)).name) == name {
                return FuncId::from_index(i);
            }
        }
        panic!("no function {name}");
    }

    #[test]
    fn merge_resolves_cross_module_call() {
        let mut syms = StrInterner::new();
        let a = module_a(&mut syms);
        let b = module_b(&mut syms);
        let merged = merge_modules([a, b], "prog").expect("merge");

        // Both symbols exist and are definitions now.
        let helper = find_func(&merged, &syms, "helper");
        let main = find_func(&merged, &syms, "main");
        assert!(!merged.function(helper).is_declaration(), "helper resolved to a definition");
        assert!(!merged.function(main).is_declaration());

        // main's call now targets the (defined) helper.
        let mf = merged.function(main);
        let mut found_call = false;
        for (_bid, blk) in mf.blocks() {
            for &i in blk.insts() {
                if matches!(mf.inst(i).kind, InstKind::Call) {
                    let callee = mf.inst(i).operands()[0];
                    assert_eq!(mf.value(callee).def, ValueDef::Func(helper));
                    found_call = true;
                }
            }
        }
        assert!(found_call, "expected the call to helper");
        assert!(verify_module(&merged).is_ok(), "merged module must verify");
    }

    #[test]
    fn cross_module_inline_after_lto() {
        let mut syms = StrInterner::new();
        let a = module_a(&mut syms);
        let b = module_b(&mut syms);
        let mut merged = merge_modules([a, b], "prog").expect("merge");

        optimize(&mut merged, OptLevel::O2);
        assert!(verify_module(&merged).is_ok());

        let main = find_func(&merged, &syms, "main");
        // The cross-module call must be gone (inlined).
        let calls = merged
            .function(main)
            .blocks()
            .flat_map(|(_, b)| b.insts().iter())
            .filter(|&&i| matches!(merged.function(main).inst(i).kind, InstKind::Call))
            .count();
        assert_eq!(calls, 0, "cross-module call should be inlined away");

        // And const-folded to 42 (helper()==40, +2).
        match ret_const(&merged, main) {
            ConstLattice::Const(Const::Int { value, .. }) => {
                assert_eq!(value.to_i64(), Some(42), "helper()+2 must fold to 42");
            }
            other => panic!("expected constant 42, got {other:?}"),
        }
    }

    #[test]
    fn merge_is_deterministic() {
        let render = || {
            let mut syms = StrInterner::new();
            let a = module_a(&mut syms);
            let b = module_b(&mut syms);
            let merged = merge_modules([a, b], "prog").expect("merge");
            text::print_module(&merged, &syms)
        };
        assert_eq!(render(), render(), "merge must be deterministic");
    }

    #[test]
    fn duplicate_definition_is_an_error() {
        let mut syms = StrInterner::new();
        let a1 = module_a(&mut syms);
        let a2 = module_a(&mut syms); // both define `helper`
        match merge_modules([a1, a2], "prog") {
            Err(MergeError::DuplicateFunction(_)) => {}
            other => panic!("expected DuplicateFunction, got {other:?}"),
        }
    }

    #[test]
    fn undefined_declaration_stays_external() {
        // Merging only B leaves `helper` an external declaration.
        let mut syms = StrInterner::new();
        let b = module_b(&mut syms);
        let merged = merge_modules([b], "prog").expect("merge");
        let helper = find_func(&merged, &syms, "helper");
        assert!(merged.function(helper).is_declaration(), "unresolved decl stays external");
    }

    // --- End-to-end: merge + optimize + link + run ----------------------------
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn lto_program_runs_with_correct_exit_code() {
        use crate::link::{ImageOptions, link_executable, write_executable};

        let mut syms = StrInterner::new();
        let a = module_a(&mut syms);
        let b = module_b(&mut syms);
        let mut merged = merge_modules([a, b], "prog").expect("merge");
        optimize(&mut merged, OptLevel::O2);
        assert!(verify_module(&merged).is_ok());

        let obj = crate::target::x86_64::compile_module(&merged, &syms);
        let image = link_executable(vec![obj], &ImageOptions::default()).expect("link");
        let path = std::env::temp_dir().join(format!("lf_lto_{}", std::process::id()));
        let path_str = path.to_str().unwrap().to_owned();
        write_executable(&path_str, &image).expect("write");
        let status = loop {
            match std::process::Command::new(&path).status() {
                Ok(s) => break s,
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec: {e}"),
            }
        };
        let _ = std::fs::remove_file(&path);
        assert_eq!(status.code(), Some(42), "LTO program must return helper()+2 = 42");
    }

    /// Global definitions merge by strength (strong beats weak beats detached
    /// beats declaration), keep their attributes, and address constants in
    /// initializers are remapped to the merged globals/functions.
    #[test]
    fn merge_globals_attrs_and_address_constants() {
        let a = "module \"a\"\n\
                 global weak @w : i64 = i64 1\n\
                 global internal constant @k : i64 = i64 3\n\
                 global @x : i64 = i64 7\n\
                 func @g() -> void {\nentry ^0:\n  ret\n}\n";
        let b = "module \"b\"\n\
                 global @tab : [2 x ptr] = [2 x ptr] (ptr @x + 8, ptr @g)\n\
                 global @w : i64 = i64 2\n\
                 global @x : i64\n\
                 func @g() -> void\n";
        let file = crate::support::diagnostics::FileId::new(0);
        let mut syms = StrInterner::new();
        let ma = text::parse_module(a, file, &mut syms).unwrap();
        let mb = text::parse_module(b, file, &mut syms).unwrap();
        let merged = merge_modules([ma, mb], "ab").expect("merge");
        verify_module(&merged).expect("merged module verifies");
        let out = text::print_module(&merged, &syms);
        assert!(out.contains("global @w : i64 = i64 2\n"), "strong beats weak:\n{out}");
        assert!(out.contains("global internal constant @k : i64 = i64 3\n"), "{out}");
        assert!(out.contains("global @x : i64 = i64 7\n"), "{out}");
        assert!(
            out.contains("global @tab : [2 x ptr] = [2 x ptr] (ptr @x + 8, ptr @g)\n"),
            "address constants remapped:\n{out}"
        );
        assert_eq!(merged.global_count(), 4);
        assert_eq!(merged.function_count(), 1);

        // Two strong definitions still clash.
        let c = "module \"c\"\nglobal @x : i64 = i64 8\n";
        let md = text::parse_module(a, file, &mut syms).unwrap();
        let mc = text::parse_module(c, file, &mut syms).unwrap();
        assert!(matches!(merge_modules([md, mc], "dc"), Err(MergeError::DuplicateGlobal(_))));
    }

    /// The definition's linkage wins and visibility merges to the most
    /// constraining one, for functions and globals alike.
    #[test]
    fn merge_resolves_linkage_and_visibility() {
        let a = "module \"a\"\n\
                 global hidden @x : i64\n\
                 func hidden @f() -> void\n\
                 func protected @g() -> void\n";
        let b = "module \"b\"\n\
                 global @x : i64 = i64 1\n\
                 func weak @f() -> void {\nentry ^0:\n  ret\n}\n\
                 func internal @g() -> void {\nentry ^0:\n  ret\n}\n";
        let file = crate::support::diagnostics::FileId::new(0);
        let mut syms = StrInterner::new();
        let ma = text::parse_module(a, file, &mut syms).unwrap();
        let mb = text::parse_module(b, file, &mut syms).unwrap();
        let merged = merge_modules([ma, mb], "ab").expect("merge");
        let out = text::print_module(&merged, &syms);
        assert!(out.contains("global hidden @x : i64 = i64 1\n"), "{out}");
        assert!(out.contains("func weak hidden @f() -> void {"), "{out}");
        assert!(out.contains("func internal protected @g() -> void {"), "{out}");
    }
}
