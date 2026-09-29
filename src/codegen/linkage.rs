//! Symbol binding decisions shared by every backend (`docs/ir-design.md` §4b).
//!
//! Two questions, both answered from the IR's [`Linkage`] and [`Visibility`]:
//!
//! 1. **Does a reference bind locally?** Under a position-independent
//!    [`RelocModel`], a symbol that might resolve to a definition in another
//!    linked component (preemption, or a definition in a shared library) must
//!    be addressed through the global offset table; a symbol that is known to
//!    resolve inside this component can be addressed directly and
//!    PC-relatively. [`func_binds_locally`] / [`global_binds_locally`] decide:
//!
//!    | model    | locally bound                                                |
//!    |----------|--------------------------------------------------------------|
//!    | `Static` | everything (the static linker resolves every address)        |
//!    | `Pie`    | `internal`, `hidden`, and anything this module defines       |
//!    | `Pic`    | `internal` and `hidden` only                                 |
//!
//!    Protected symbols are *not* treated as locally bound for address-taking:
//!    they cannot be preempted, but their canonical address may live in the
//!    executable (a PLT stub or a copy-relocated datum), so the GOT is the
//!    portable way to get it. Calls need no such care: a `PLT32` call to a
//!    locally bound symbol is resolved directly by the linker.
//!
//! 2. **What do the object's symbols look like?** [`apply_symbol_attrs`] stamps
//!    each IR function's linkage onto its definition's binding (functions are
//!    emitted `STB_GLOBAL` by the backends' drivers) and every IR symbol's
//!    visibility onto the object symbol of the same name, definition or
//!    reference.

use crate::codegen::options::RelocModel;
use crate::ir::{FuncId, GlobalId, Linkage, Module, Visibility};
use crate::mc::object::{ObjectModule, SymbolBinding};
use crate::support::StrInterner;

/// Whether a reference to function `f` binds inside the component being built,
/// so its address can be formed PC-relatively (see the [module docs](self)).
pub fn func_binds_locally(module: &Module, f: FuncId, model: RelocModel) -> bool {
    let func = module.function(f);
    binds_locally(func.attrs.linkage, func.attrs.visibility, !func.is_declaration(), model)
}

/// Whether a reference to global `g` binds inside the component being built
/// (see the [module docs](self)). A [`detached`](crate::ir::GlobalAttrs::detached)
/// global counts as defined elsewhere.
pub fn global_binds_locally(module: &Module, g: GlobalId, model: RelocModel) -> bool {
    let attrs = module.global_attrs(g);
    let defined = module.global(g).init.is_some() && !attrs.detached;
    binds_locally(attrs.linkage, attrs.visibility, defined, model)
}

fn binds_locally(linkage: Linkage, visibility: Visibility, defined: bool, model: RelocModel) -> bool {
    match model {
        RelocModel::Static => true,
        _ if linkage == Linkage::Internal || visibility == Visibility::Hidden => true,
        RelocModel::Pie => defined,
        RelocModel::Pic => false,
    }
}

/// Apply the IR linkage and visibility to `obj`'s symbols: every defined
/// function's binding follows its [`Linkage`] (external → global, internal →
/// local, weak → weak), and every function and global symbol present in `obj`
/// (definition or undefined reference) gets its IR [`Visibility`]. Call it
/// once, after the functions and globals have been emitted.
pub fn apply_symbol_attrs(module: &Module, syms: &StrInterner, obj: &mut ObjectModule) {
    for f in module.functions() {
        let Some(id) = obj.symbol_id(syms.resolve(f.name)) else { continue };
        let mut sym = obj.symbol(id).clone();
        if !sym.is_undefined() {
            sym.binding = binding(f.attrs.linkage);
        }
        sym.visibility = f.attrs.visibility.into();
        obj.add_symbol(sym);
    }
    for (i, g) in module.globals().enumerate() {
        let Some(id) = obj.symbol_id(syms.resolve(g.name)) else { continue };
        let mut sym = obj.symbol(id).clone();
        sym.visibility = module.global_attrs(GlobalId::from_index(i)).visibility.into();
        obj.add_symbol(sym);
    }
}

fn binding(linkage: Linkage) -> SymbolBinding {
    match linkage {
        Linkage::External => SymbolBinding::Global,
        Linkage::Internal => SymbolBinding::Local,
        Linkage::Weak => SymbolBinding::Weak,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::object::SymbolVisibility;
    use crate::support::diagnostics::FileId;

    const SRC: &str = "module \"l\"\n\
        global @data : i32 = i32 1\n\
        global hidden @hdata : i32 = i32 2\n\
        global internal @idata : i32 = i32 3\n\
        global protected @pdata : i32 = i32 4\n\
        global @ext : i32\n\
        global detached @det : i32 = i32 5\n\
        func @def() -> void {\nentry ^0:\n  ret\n}\n\
        func hidden @hdef() -> void {\nentry ^0:\n  ret\n}\n\
        func internal @idef() -> void {\nentry ^0:\n  ret\n}\n\
        func weak @wdef() -> void {\nentry ^0:\n  ret\n}\n\
        func @decl() -> void\n\
        func hidden @hdecl() -> void\n";

    #[test]
    fn local_binding_per_model() {
        let mut syms = StrInterner::new();
        let m = crate::ir::text::parse_module(SRC, FileId::new(0), &mut syms).unwrap();
        let g = |model| -> Vec<bool> {
            (0..m.global_count()).map(|i| global_binds_locally(&m, GlobalId::from_index(i), model)).collect()
        };
        let f = |model| -> Vec<bool> {
            (0..m.function_count()).map(|i| func_binds_locally(&m, FuncId::from_index(i), model)).collect()
        };
        assert!(g(RelocModel::Static).iter().all(|&b| b));
        assert!(f(RelocModel::Static).iter().all(|&b| b));
        //                                    data   hidden internal protected ext   detached
        assert_eq!(g(RelocModel::Pie), [true, true, true, true, false, false]);
        assert_eq!(g(RelocModel::Pic), [false, true, true, false, false, false]);
        //                                    def   hidden internal weak  decl  hidden-decl
        assert_eq!(f(RelocModel::Pie), [true, true, true, true, false, true]);
        assert_eq!(f(RelocModel::Pic), [false, true, true, false, false, true]);
    }

    #[test]
    fn object_symbols_get_linkage_and_visibility() {
        let mut syms = StrInterner::new();
        let m = crate::ir::text::parse_module(SRC, FileId::new(0), &mut syms).unwrap();
        let obj = crate::target::x86_64::compile_module(&m, &syms);
        let sym = |n: &str| obj.symbol(obj.symbol_id(n).unwrap()).clone();
        assert_eq!(sym("def").binding, SymbolBinding::Global);
        assert_eq!(sym("idef").binding, SymbolBinding::Local);
        assert_eq!(sym("wdef").binding, SymbolBinding::Weak);
        assert_eq!(sym("hdef").visibility, SymbolVisibility::Hidden);
        assert_eq!(sym("hdata").visibility, SymbolVisibility::Hidden);
        assert_eq!(sym("pdata").visibility, SymbolVisibility::Protected);
        assert_eq!(sym("data").visibility, SymbolVisibility::Default);
        assert_eq!(sym("idata").binding, SymbolBinding::Local);
    }
}
