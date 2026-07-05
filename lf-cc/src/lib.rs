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
pub mod lex;
pub mod lower;
pub mod parse;
pub mod sema;

use latticefoundry::ir::Module;
use latticefoundry::link::{self, ImageOptions};
use latticefoundry::mc::object::{
    ObjectModule, Section, SectionKind, Symbol, SymbolBinding, SymbolType,
};
use latticefoundry::support::StrInterner;
use latticefoundry::support::diagnostics::{Diagnostic, FileId};
use latticefoundry::target::x86_64;
use latticefoundry::transform::pipeline::{self, OptLevel};
use latticefoundry::verify;

use sema::TGlobal;

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
    let program = check_source(source)?;
    Ok(lower::lower(&program, source, module_name, debug))
}

/// Lex, parse, and type-check `source`, returning the typed [`sema::Program`] or
/// the diagnostics. Exposed so tests can inspect the checked program directly.
pub fn check_source(source: &str) -> Result<sema::Program, Vec<Diagnostic>> {
    let tokens = lex::lex(source, FileId::new(0))?;
    let unit = parse::parse(tokens)?;
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
    let program = check_source(source).map_err(BuildError::Frontend)?;
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

    let image_opts = ImageOptions { debug, ..ImageOptions::default() };
    link::link_executable(vec![obj], &image_opts)
        .map_err(|e| BuildError::Backend(format!("link error: {e}")))
}

/// Verify a module (structural tier), mapping any errors to a [`BuildError`].
fn verify_or(module: &Module, stage: &str) -> Result<(), BuildError> {
    verify::verify_module(module).map_err(|diags| {
        let n = diags.iter().filter(|d| d.is_error()).count();
        BuildError::Backend(format!("{stage} IR verification failed ({n} error(s))"))
    })
}

/// Emit the module's global variables into a `.data` section of the object,
/// defining a symbol for each (the backend's `compile_module` emits only code,
/// so global storage is contributed here). Little-endian initializer bytes are
/// laid down for the declared width.
fn emit_globals(obj: &mut ObjectModule, globals: &[TGlobal]) {
    if globals.is_empty() {
        return;
    }
    let data = obj.add_section(Section::new(".data", SectionKind::Data, 8));
    let mut bytes: Vec<u8> = Vec::new();
    for g in globals {
        let size = sema::size_of(&g.ty) as usize;
        let align = size.max(1);
        while !bytes.len().is_multiple_of(align) {
            bytes.push(0);
        }
        let off = bytes.len() as u64;
        let le = g.init.to_le_bytes();
        bytes.extend_from_slice(&le[..size.min(le.len())]);
        obj.add_symbol(Symbol::defined(
            g.name.clone(),
            SymbolBinding::Global,
            SymbolType::Object,
            data,
            off,
            size as u64,
        ));
    }
    obj.section_mut(data).bytes = bytes;
}

#[cfg(test)]
mod tests;
