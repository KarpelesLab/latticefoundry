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
pub mod consteval;
pub mod cstd;
pub mod headers;
pub mod layout;
pub mod lex;
pub mod lower;
pub mod parse;
pub mod preprocess;
pub mod sema;

pub use cstd::CStd;
pub use preprocess::{MacroOp, PpOptions, SourceLocation, SourceMap, default_system_include_dirs};

use latticefoundry::codegen::{CodegenOptions, RelocModel};
use latticefoundry::ir::{Module, Visibility};
use latticefoundry::link::{self, ImageOptions};
use latticefoundry::mc::asm::{self as mcasm, AsmOptions, AsmSource};
use latticefoundry::mc::object::{
    ObjectModule, RelocKind, Relocation, Section, SectionId, SectionKind, Symbol, SymbolBinding,
    SymbolType,
};
use latticefoundry::support::StrInterner;
use latticefoundry::support::diagnostics::Diagnostic;
use latticefoundry::target::{TargetArch, x86_64};
use latticefoundry::transform::pipeline::{self, OptLevel};
use latticefoundry::verify;

use sema::{FuncSig, TGlobal};

/// Code-generation choices that do not change a program's meaning but shape
/// its object: the relocation model (`-fPIC`/`-fPIE`) and the default symbol
/// visibility of definitions (`-fvisibility=`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CodegenConfig {
    /// How position-independent the code must be (default: static).
    pub reloc_model: RelocModel,
    /// The visibility of every definition without a `visibility` attribute
    /// (default: `default`).
    pub default_visibility: Visibility,
}

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
    check_source_mapped(source, opts).map_err(|(diags, _)| diags)
}

/// Like [`check_source_with`], but a failure also returns the [`SourceMap`]
/// that resolves the diagnostics' spans to file, line and column (errors in
/// included headers point into those headers; see [`SourceMap::render`]).
pub fn check_source_mapped(
    source: &str,
    opts: &PpOptions,
) -> Result<sema::Program, (Vec<Diagnostic>, SourceMap)> {
    let (tokens, map) = preprocess::preprocess_mapped(source, opts);
    let checked = tokens
        .and_then(|tokens| parse::parse(tokens, opts.std))
        .and_then(|unit| sema::check(&unit, opts.std));
    checked.map_err(|diags| (diags, map))
}

/// Why a full source-to-executable build failed.
#[derive(Debug)]
pub enum BuildError {
    /// Lex/parse/type errors from the front end, with the [`SourceMap`] their
    /// spans resolve through.
    Frontend(Vec<Diagnostic>, SourceMap),
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
    let program = check_source_mapped(source, opts).map_err(|(d, m)| BuildError::Frontend(d, m))?;
    let cfg = CodegenConfig::default();
    if !program.toplevel_asm.is_empty() {
        // The self-contained linker consumes only our own object modules; the
        // assembled file-scope asm is a separate ELF object that needs the
        // hosted (ELF) link path.
        return Err(BuildError::Backend(
            "file-scope asm needs the object path (compile_object_with + an ELF link)".to_owned(),
        ));
    }
    let obj = compile_program(&program, source, input_name, opt, debug, &cfg)?;
    link_image(vec![obj], debug)
}

/// The result of [`compile_module_with`]: the translation unit's in-memory
/// object module plus its file-scope `asm(...)` templates.
#[derive(Debug)]
pub struct CompiledModule {
    /// The compiled C code, ready for [`link_image`] or
    /// `latticefoundry::mc::elf::write`.
    pub module: ObjectModule,
    /// The file-scope asm templates, in source order (see [`CompiledObject`]).
    pub toplevel_asm: Vec<String>,
}

/// Compile one translation unit to an in-memory [`ObjectModule`] (plus its
/// file-scope asm). A unit without file-scope asm can go straight to the
/// framework's own static linker ([`link_image`]); one with it needs the ELF
/// link path, since its assembled code is a separate ELF object.
pub fn compile_module_with(
    source: &str,
    input_name: &str,
    opts: &PpOptions,
    opt: OptLevel,
    debug: bool,
) -> Result<CompiledModule, BuildError> {
    compile_module_cfg(source, input_name, opts, opt, debug, &CodegenConfig::default())
}

/// Like [`compile_module_with`], under the code-generation configuration `cfg`
/// (relocation model, default visibility).
pub fn compile_module_cfg(
    source: &str,
    input_name: &str,
    opts: &PpOptions,
    opt: OptLevel,
    debug: bool,
    cfg: &CodegenConfig,
) -> Result<CompiledModule, BuildError> {
    let program = check_source_mapped(source, opts).map_err(|(d, m)| BuildError::Frontend(d, m))?;
    let module = compile_program(&program, source, input_name, opt, debug, cfg)?;
    Ok(CompiledModule { module, toplevel_asm: program.toplevel_asm })
}

/// Link in-memory objects (from [`compile_module_with`]) into a static,
/// libc-free x86-64 executable image with the framework's own linker core,
/// which synthesizes a `_start` that calls `main` and exits with its result.
pub fn link_image(objects: Vec<ObjectModule>, debug: bool) -> Result<Vec<u8>, BuildError> {
    let image_opts = ImageOptions { debug, ..ImageOptions::default() };
    link::link_executable(objects, &image_opts)
        .map_err(|e| BuildError::Backend(format!("link error: {e}")))
}

/// Compile a translation unit to a **relocatable ELF object** (the `-c` mode).
///
/// Unlike [`build_image_with`], this stops before linking and returns the ELF
/// `.o` bytes, so the object can be linked by a GNU-style linker against a real
/// libc (calls to undefined symbols like `printf`/`malloc` become relocations
/// the linker resolves). The driver's hosted link feeds these objects to `qld`
/// through `latticefoundry::link::gnu`.
///
/// A translation unit with file-scope `asm(...)` declarations is rejected here
/// (their code would otherwise be silently dropped); use
/// [`compile_object_with`], which returns the asm alongside the object.
pub fn build_object_with(
    source: &str,
    input_name: &str,
    opts: &PpOptions,
    opt: OptLevel,
    debug: bool,
) -> Result<Vec<u8>, BuildError> {
    let out = compile_object_with(source, input_name, opts, opt, debug)?;
    if !out.toplevel_asm.is_empty() {
        return Err(BuildError::Backend(
            "file-scope asm needs compile_object_with (its assembled object must be linked too)"
                .to_owned(),
        ));
    }
    Ok(out.object)
}

/// The result of [`compile_object_with`]: the translation unit's ELF object
/// plus the templates of its file-scope `asm(...)` declarations.
#[derive(Clone, Debug)]
pub struct CompiledObject {
    /// The relocatable ELF object for the C code.
    pub object: Vec<u8>,
    /// The file-scope asm templates, in source order (empty when there are
    /// none). Assemble them with [`assemble_toplevel_asm`] and link the result
    /// next to `object`.
    pub toplevel_asm: Vec<String>,
}

/// Compile a translation unit to a relocatable ELF object, also returning its
/// file-scope `asm(...)` templates (see [`CompiledObject`]).
pub fn compile_object_with(
    source: &str,
    input_name: &str,
    opts: &PpOptions,
    opt: OptLevel,
    debug: bool,
) -> Result<CompiledObject, BuildError> {
    compile_object_cfg(source, input_name, opts, opt, debug, &CodegenConfig::default())
}

/// Like [`compile_object_with`], under the code-generation configuration `cfg`
/// (e.g. position-independent code for a shared library).
pub fn compile_object_cfg(
    source: &str,
    input_name: &str,
    opts: &PpOptions,
    opt: OptLevel,
    debug: bool,
    cfg: &CodegenConfig,
) -> Result<CompiledObject, BuildError> {
    let compiled = compile_module_cfg(source, input_name, opts, opt, debug, cfg)?;
    Ok(CompiledObject {
        object: latticefoundry::mc::elf::write(&compiled.module),
        toplevel_asm: compiled.toplevel_asm,
    })
}

/// Assemble a translation unit's file-scope asm templates (in order, as one
/// assembly source, as GCC would emit them into its `.s`) into an x86-64 ELF
/// relocatable object with our own assembler. Returns `Ok(None)` when there is
/// no file-scope asm. `input_name` names the C source for diagnostics.
pub fn assemble_toplevel_asm(
    templates: &[String],
    input_name: &str,
) -> Result<Option<Vec<u8>>, BuildError> {
    if templates.is_empty() {
        return Ok(None);
    }
    let mut text = String::new();
    for t in templates {
        text.push_str(t);
        text.push('\n');
    }
    let name = format!("{input_name} (file-scope asm)");
    mcasm::assemble(&[AsmSource { name: &name, text: &text }], &AsmOptions::new(TargetArch::X86_64))
        .map(Some)
        .map_err(|e| BuildError::Backend(format!("assembling file-scope asm: {e}")))
}

/// Lower, verify, optimize, and compile a checked program to an x86-64 object
/// module with its global data and linkage applied.
fn compile_program(
    program: &sema::Program,
    source: &str,
    input_name: &str,
    opt: OptLevel,
    debug: bool,
    cfg: &CodegenConfig,
) -> Result<ObjectModule, BuildError> {
    let (mut module, syms) = lower::lower_with(program, source, input_name, debug, cfg);

    verify_or(&module, "lowered")?;
    pipeline::optimize(&mut module, opt);
    if opt != OptLevel::O0 {
        verify_or(&module, "optimized")?;
    }

    let cg = CodegenOptions::default().with_reloc_model(cfg.reloc_model);
    let mut obj = if debug {
        let comp_dir = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(str::to_owned))
            .unwrap_or_default();
        let source_desc = x86_64::DebugSource { file_name: input_name.to_owned(), comp_dir };
        x86_64::compile_module_debug_with(&module, &syms, &source_desc, &cg).object
    } else {
        x86_64::compile_module_with(&module, &syms, &cg).object
    };
    emit_globals(&mut obj, &program.globals, &program.records, cfg);
    apply_weak_references(&mut obj, &program.sigs, &program.globals);
    Ok(obj)
}

/// Bind the undefined references to `weak`-declared functions and objects
/// weakly (`STB_WEAK`), so they resolve to address 0 when nothing defines
/// them. (Weak *definitions* get their binding from the IR linkage.)
fn apply_weak_references(obj: &mut ObjectModule, sigs: &[FuncSig], globals: &[TGlobal]) {
    let weak_refs = sigs
        .iter()
        .filter(|s| s.weak && !s.defined)
        .map(|s| s.name.as_str())
        .chain(globals.iter().filter(|g| g.weak && !g.defined).map(|g| g.name.as_str()));
    let names: Vec<String> = weak_refs.map(str::to_owned).collect();
    for name in names {
        if let Some(id) = obj.symbol_id(&name) {
            let mut sym = obj.symbol(id).clone();
            if sym.is_undefined() && sym.binding != SymbolBinding::Weak {
                sym.binding = SymbolBinding::Weak;
                obj.add_symbol(sym);
            }
        }
    }
}

/// Verify a module (structural tier), mapping any errors to a [`BuildError`].
fn verify_or(module: &Module, stage: &str) -> Result<(), BuildError> {
    verify::verify_module(module).map_err(|diags| {
        let n = diags.iter().filter(|d| d.is_error()).count();
        // Name the first problem: a failure here is a compiler bug, and the
        // message is what a bug report starts from.
        let first = diags.iter().find(|d| d.is_error()).map(|d| format!(": {}", d.message));
        BuildError::Backend(format!(
            "{stage} IR verification failed ({n} error(s)){}",
            first.unwrap_or_default()
        ))
    })
}

/// The sections [`emit_globals`] fills, in the order of [`GlobalSection`].
const GLOBAL_SECTIONS: [(&str, SectionKind); 4] = [
    (".data", SectionKind::Data),
    (".rodata", SectionKind::Rodata),
    (".tdata", SectionKind::TData),
    (".tbss", SectionKind::TBss),
];

/// Which of [`GLOBAL_SECTIONS`] a global goes to.
#[derive(Clone, Copy)]
enum GlobalSection {
    Data,
    Rodata,
    TData,
    TBss,
}

/// Emit the module's global variables into the object, defining a symbol for
/// each (the backend's `compile_module` emits only code, so global storage is
/// contributed here). Writable globals go in `.data`; read-only objects (string
/// literals) in `.rodata`; thread-local ones in `.tdata`, or `.tbss` when
/// all zero, with `STT_TLS` symbols. Each global's fully-materialized
/// initializer image is copied verbatim.
fn emit_globals(obj: &mut ObjectModule, globals: &[TGlobal], records: &ast::Records, cfg: &CodegenConfig) {
    if globals.is_empty() {
        return;
    }
    let mut sections: [Option<SectionId>; 4] = [None; 4];
    let mut contents: [Vec<u8>; 4] = Default::default();
    // `.tbss` holds no bytes: its running size.
    let mut tbss_size = 0u64;
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
        // A type wanting more than 8 (a 16-byte vector, an over-aligned
        // record) gets it, up to the sections' 16.
        let natural = (layout::align_of(records, &g.ty) as usize).min(16);
        let align = size.next_power_of_two().clamp(1, 8).max(natural);
        let which = if g.thread_local {
            if g.relocs.is_empty() && g.bytes.iter().all(|&b| b == 0) {
                GlobalSection::TBss
            } else {
                GlobalSection::TData
            }
        } else if g.readonly {
            GlobalSection::Rodata
        } else {
            GlobalSection::Data
        };
        let sec = *sections[which as usize].get_or_insert_with(|| {
            let (name, kind) = GLOBAL_SECTIONS[which as usize];
            obj.add_section(Section::new(name, kind, 16))
        });
        let off = if let GlobalSection::TBss = which {
            tbss_size = tbss_size.next_multiple_of(align as u64);
            let off = tbss_size;
            tbss_size += size as u64;
            off
        } else {
            let bytes = &mut contents[which as usize];
            while !bytes.len().is_multiple_of(align) {
                bytes.push(0);
            }
            let off = bytes.len() as u64;
            bytes.extend_from_slice(&g.bytes);
            off
        };
        // A `static` object has internal linkage: its symbol is local. A
        // *tentative* definition (`T x;` with no initializer) may be emitted by
        // several translation units — classically through a shared header — so it
        // is bound *weakly*: the linker then merges the duplicates, and a strong
        // (initialized) definition elsewhere wins. This matches the traditional
        // `-fcommon` behavior that pre-C99-era sources such as make-3.82 rely on.
        // `__attribute__((weak))` makes any definition weak.
        let binding = if g.is_static {
            SymbolBinding::Local
        } else if g.tentative || g.weak {
            SymbolBinding::Weak
        } else {
            SymbolBinding::Global
        };
        let kind = if g.thread_local { SymbolType::Tls } else { SymbolType::Object };
        let mut sym = Symbol::defined(g.name.clone(), binding, kind, sec, off, g.bytes.len() as u64);
        sym.visibility = lower::global_visibility(g, cfg).into();
        obj.add_symbol(sym);
        for r in &g.relocs {
            pending.push((sec, off + r.offset, r.symbol.clone(), r.addend));
        }
    }
    for (sec, bytes) in sections.into_iter().zip(contents) {
        if let Some(sec) = sec {
            obj.section_mut(sec).bytes = bytes;
        }
    }
    if let Some(sec) = sections[GlobalSection::TBss as usize] {
        obj.section_mut(sec).bss_size = tbss_size;
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
