//! lf-cc — a C frontend for LatticeFoundry (a separate crate; see Cargo.toml).
//!
//! Lexes, parses, and type-checks a freestanding subset of C (scalars and
//! pointers; no preprocessor, no libc, no aggregates in v1) and lowers it to
//! `latticefoundry::ir`, reusing the framework's verify → optimize → codegen →
//! link pipeline to produce a native x86-64 executable.
//!
//! The frontend is a clean-room implementation written from the C grammar
//! (design tenet T1): a hand-written [`lex`]er, a recursive-descent [`parse`]r,
//! a [`sema`]ntic checker that makes every C conversion explicit in a typed
//! tree, and a [`lower`]ing pass to the IR builder. No `unsafe` is used.

pub mod ast;
pub mod cstd;
pub mod headers;
pub mod layout;
pub mod lex;
pub mod lower;
pub mod parse;
pub mod preprocess;
pub mod sema;

pub use cstd::CStd;
pub use preprocess::{MacroOp, PpOptions};

use latticefoundry::ir::Module;
use latticefoundry::link::{self, ImageOptions};
use latticefoundry::mc::object::{
    ObjectModule, RelocKind, Relocation, Section, SectionId, SectionKind, Symbol, SymbolBinding,
    SymbolType,
};
use latticefoundry::support::StrInterner;
use latticefoundry::support::diagnostics::Diagnostic;
use latticefoundry::target::x86_64;
use latticefoundry::transform::pipeline::{self, OptLevel};
use latticefoundry::verify;

use sema::{FuncSig, TGlobal};

/// Compile C source text all the way to a lowered IR [`Module`] plus the symbol
/// interner its names live in. Returns the collected diagnostics on any lex,
/// parse, or type error.
///
/// `debug` requests source-line provenance (`set_line`/`set_decl_line`) so the
/// caller can emit DWARF with `compile_module_debug`.
pub fn compile_to_ir(
    source: &str,
    module_name: &str,
    debug: bool,
) -> Result<(Module, StrInterner), Vec<Diagnostic>> {
    let opts = PpOptions { main_file_name: module_name.to_owned(), ..PpOptions::default() };
    compile_to_ir_with(source, module_name, &opts, debug)
}

/// Like [`compile_to_ir`], but with explicit preprocessor/standard [`PpOptions`].
pub fn compile_to_ir_with(
    source: &str,
    module_name: &str,
    opts: &PpOptions,
    debug: bool,
) -> Result<(Module, StrInterner), Vec<Diagnostic>> {
    let program = check_source_with(source, opts)?;
    Ok(lower::lower(&program, source, module_name, debug))
}

/// Preprocess, parse, and type-check `source` under the default standard
/// (`gnu17`, no includes/defines). Exposed so tests can inspect the program.
pub fn check_source(source: &str) -> Result<sema::Program, Vec<Diagnostic>> {
    check_source_with(source, &PpOptions::default())
}

/// Preprocess `source` with `opts`, then parse and type-check it.
pub fn check_source_with(
    source: &str,
    opts: &PpOptions,
) -> Result<sema::Program, Vec<Diagnostic>> {
    let tokens = preprocess::preprocess(source, opts)?;
    let unit = parse::parse(tokens, opts.std)?;
    sema::check(&unit)
}

/// Why a full source-to-executable build failed.
#[derive(Debug)]
pub enum BuildError {
    /// Lex/parse/type errors from the front end (carry source spans).
    Frontend(Vec<Diagnostic>),
    /// A back-end failure (verification, codegen, or linking).
    Backend(String),
}

/// Compile C `source` all the way to a linked, static x86-64 executable image
/// (the raw ELF bytes; the caller writes and `chmod +x`es them).
///
/// `input_name` names the source for diagnostics/DWARF, `opt` is the
/// optimization level, and `debug` requests DWARF debug info.
pub fn build_image(
    source: &str,
    input_name: &str,
    opt: OptLevel,
    debug: bool,
) -> Result<Vec<u8>, BuildError> {
    let opts = PpOptions { main_file_name: input_name.to_owned(), ..PpOptions::default() };
    build_image_with(source, input_name, &opts, opt, debug)
}

/// Like [`build_image`], but with explicit preprocessor/standard [`PpOptions`]
/// (the driver's `--std`, `-I`, and `-D`/`-U` flags feed into these).
pub fn build_image_with(
    source: &str,
    input_name: &str,
    opts: &PpOptions,
    opt: OptLevel,
    debug: bool,
) -> Result<Vec<u8>, BuildError> {
    let program = check_source_with(source, opts).map_err(BuildError::Frontend)?;
    let (mut module, syms) = lower::lower(&program, source, input_name, debug);

    verify_or(&module, "lowered")?;
    pipeline::optimize(&mut module, opt);
    if opt != OptLevel::O0 {
        verify_or(&module, "optimized")?;
    }

    let mut obj = if debug {
        let comp_dir = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(str::to_owned))
            .unwrap_or_default();
        let source_desc = x86_64::DebugSource { file_name: input_name.to_owned(), comp_dir };
        x86_64::compile_module_debug(&module, &syms, &source_desc)
    } else {
        x86_64::compile_module(&module, &syms)
    };
    emit_globals(&mut obj, &program.globals);
    apply_static_linkage(&mut obj, &program.sigs);

    let image_opts = ImageOptions { debug, ..ImageOptions::default() };
    link::link_executable(vec![obj], &image_opts)
        .map_err(|e| BuildError::Backend(format!("link error: {e}")))
}

/// Compile a translation unit to a **relocatable ELF object** (the `-c` mode).
///
/// Unlike [`build_image_with`], this stops before linking and returns the ELF
/// `.o` bytes, so the object can be linked by an external linker against a real
/// libc (calls to undefined symbols like `printf`/`malloc` become relocations
/// the system linker resolves). This is how lf-cc-compiled hosted programs are
/// tested.
pub fn build_object_with(
    source: &str,
    input_name: &str,
    opts: &PpOptions,
    opt: OptLevel,
    debug: bool,
) -> Result<Vec<u8>, BuildError> {
    let program = check_source_with(source, opts).map_err(BuildError::Frontend)?;
    let (mut module, syms) = lower::lower(&program, source, input_name, debug);

    verify_or(&module, "lowered")?;
    pipeline::optimize(&mut module, opt);
    if opt != OptLevel::O0 {
        verify_or(&module, "optimized")?;
    }

    let mut obj = if debug {
        let comp_dir = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(str::to_owned))
            .unwrap_or_default();
        let source_desc = x86_64::DebugSource { file_name: input_name.to_owned(), comp_dir };
        x86_64::compile_module_debug(&module, &syms, &source_desc)
    } else {
        x86_64::compile_module(&module, &syms)
    };
    emit_globals(&mut obj, &program.globals);
    apply_static_linkage(&mut obj, &program.sigs);

    Ok(latticefoundry::mc::elf::write(&obj))
}

/// Give each `static` function definition internal linkage by rebinding its
/// object symbol from `Global` to `Local` (the code backend always emits
/// functions as global). This lets several translation units each define their
/// own file-scope `static` helper of the same name without colliding at link.
fn apply_static_linkage(obj: &mut ObjectModule, sigs: &[FuncSig]) {
    for sig in sigs {
        if sig.is_static
            && sig.defined
            && let Some(id) = obj.symbol_id(&sig.name)
        {
            let mut sym = obj.symbol(id).clone();
            if sym.binding != SymbolBinding::Local {
                sym.binding = SymbolBinding::Local;
                obj.add_symbol(sym);
            }
        }
    }
}

/// Verify a module (structural tier), mapping any errors to a [`BuildError`].
fn verify_or(module: &Module, stage: &str) -> Result<(), BuildError> {
    verify::verify_module(module).map_err(|diags| {
        let n = diags.iter().filter(|d| d.is_error()).count();
        BuildError::Backend(format!("{stage} IR verification failed ({n} error(s))"))
    })
}

/// Emit the module's global variables into the object, defining a symbol for
/// each (the backend's `compile_module` emits only code, so global storage is
/// contributed here). Writable globals go in `.data`; read-only objects (string
/// literals) in `.rodata`. Each global's fully-materialized initializer image is
/// copied verbatim.
fn emit_globals(obj: &mut ObjectModule, globals: &[TGlobal]) {
    if globals.is_empty() {
        return;
    }
    let mut data: Option<SectionId> = None;
    let mut rodata: Option<SectionId> = None;
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut rodata_bytes: Vec<u8> = Vec::new();
    // Address-valued fields inside globals become relocations, recorded here and
    // added after the sections exist. Each entry is `(section, field-offset,
    // target-symbol-name, addend)`.
    let mut pending: Vec<(SectionId, u64, String, i64)> = Vec::new();

    for g in globals {
        // A pure external reference (`extern T x;` with no definition here) emits
        // no storage; a use of it elsewhere in this object creates the undefined
        // symbol the linker resolves.
        if !g.defined {
            continue;
        }
        let size = g.bytes.len().max(1);
        let align = size.next_power_of_two().clamp(1, 8);
        let (sec, bytes) = if g.readonly {
            let sec = *rodata
                .get_or_insert_with(|| obj.add_section(Section::new(".rodata", SectionKind::Rodata, 8)));
            (sec, &mut rodata_bytes)
        } else {
            let sec = *data
                .get_or_insert_with(|| obj.add_section(Section::new(".data", SectionKind::Data, 8)));
            (sec, &mut data_bytes)
        };
        while !bytes.len().is_multiple_of(align) {
            bytes.push(0);
        }
        let off = bytes.len() as u64;
        bytes.extend_from_slice(&g.bytes);
        // A `static` object has internal linkage: its symbol is local. A
        // *tentative* definition (`T x;` with no initializer) may be emitted by
        // several translation units — classically through a shared header — so it
        // is bound *weakly*: the linker then merges the duplicates, and a strong
        // (initialized) definition elsewhere wins. This matches the traditional
        // `-fcommon` behavior that pre-C99-era sources such as make-3.82 rely on.
        let binding = if g.is_static {
            SymbolBinding::Local
        } else if g.tentative {
            SymbolBinding::Weak
        } else {
            SymbolBinding::Global
        };
        obj.add_symbol(Symbol::defined(
            g.name.clone(),
            binding,
            SymbolType::Object,
            sec,
            off,
            g.bytes.len() as u64,
        ));
        for r in &g.relocs {
            pending.push((sec, off + r.offset, r.symbol.clone(), r.addend));
        }
    }
    if let Some(sec) = data {
        obj.section_mut(sec).bytes = data_bytes;
    }
    if let Some(sec) = rodata {
        obj.section_mut(sec).bytes = rodata_bytes;
    }
    // Now that every defined global symbol exists, turn each recorded pointer
    // field into an absolute 64-bit relocation against its target symbol
    // (`reference_symbol` creates an undefined symbol for any not defined here).
    for (section, offset, symbol, addend) in pending {
        let sym = obj.reference_symbol(&symbol);
        obj.add_relocation(Relocation { section, offset, symbol: sym, kind: RelocKind::Abs64, addend });
    }
}

#[cfg(test)]
mod tests;
