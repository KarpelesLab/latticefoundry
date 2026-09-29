//! The **wasm32** backend: LatticeFoundry IR to WebAssembly modules,
//! clean-room from the WebAssembly Core Specification (binary format,
//! validation, execution) and the WebAssembly tool-conventions linking format.
//!
//! WebAssembly is a structured stack machine, so this backend does not use the
//! register-machine pipeline (MIR, instruction selection, register
//! allocation) of the other targets. It lowers the SSA IR directly:
//!
//! 1. **Legalization.** Vector code is scalarized by the generic
//!    [vector legalizer](crate::codegen::legalize) (wasm32 declares no legal
//!    vector type; SIMD128 is not used). Integers wider than 64 bits are split
//!    into 64-bit parts by
//!    [`legalize_ints`](crate::codegen::legalize_int::legalize_ints); what
//!    stays wide at the ABI seam lives in groups of `i64` locals (a wide
//!    parameter is several `i64` parameters, a wide result several results).
//! 2. **Structuring** ([`structure`]). Each function's CFG becomes nested
//!    `block` / `loop` / `if` / `br_table` constructs placed from the
//!    dominator tree; an irreducible CFG falls back to a dispatch loop.
//! 3. **Code** ([`lower`]). Every SSA value is a wasm local of its type
//!    (values whose live ranges do not overlap share one, assigned greedily
//!    in dominance order), block parameters are locals assigned on the
//!    incoming edges (a parallel copy through the operand stack), single-use
//!    pure values are recomputed in place as stack expressions, and narrow
//!    integers keep a zero-extension invariant in `i32`/`i64` containers.
//! 4. **Encoding** ([`binary`]). A [`WasmObject`] (functions with their code,
//!    data segments, symbols) is written as a self-contained module
//!    ([`WasmObject::to_linked`], what `lf build --target wasm32` produces) or
//!    as a relocatable object for `wasm-ld` ([`WasmObject::to_relocatable`],
//!    what `lf build -c --target wasm32` produces).
//!
//! # ABI and memory
//!
//! - **Data layout**: ILP32 with native `i32`/`i64` ([`data_layout`]):
//!   `e-p:32:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-S128-n32:64`.
//!   Linear memory is address space 0, the only one.
//! - **Calls**: direct `call`s; a function pointer is an index into the
//!   function table and an indirect call is a `call_indirect` (slot 0 stays
//!   empty, so calling null traps).
//! - **Stack**: `alloca`/`dyn_alloca` live on a shadow stack in linear memory
//!   under the mutable global `__stack_pointer` (16-byte aligned, growing
//!   down). The linked module puts the stack at the bottom of memory, so an
//!   overflow wraps below address 0 and traps instead of overwriting data (the
//!   same guarantee stack probes give the native targets; link with
//!   `wasm-ld --stack-first` for the same layout).
//! - **Imports**: every function the module declares but does not define and
//!   actually references is imported from module `"env"` under its name
//!   (`frem` calls `fmod`/`fmodf`, imported the same way unless the module
//!   defines them).
//! - **Exports**: `memory`, `__heap_base` (the first free byte after the data),
//!   `main`, and every defined function with external or weak linkage and
//!   default or protected visibility.
//!
//! # Supported operations
//!
//! Everything in the opcode table except `syscall` (there is no kernel; import
//! a host function instead), with these notes: `f16` is unsupported;
//! `volatile` accesses are plain accesses (wasm does not reorder or merge
//! memory accesses); atomics use the threads proposal (`i32.atomic.*`,
//! `atomic.fence`), with `nand`/`max`/`min`/`umax`/`umin` as
//! compare-exchange loops; `fptosi`/`fptoui` use the non-trapping saturating
//! conversions (an out-of-range input is poison in the IR anyway); variadic
//! calls are rejected.

pub mod binary;
pub mod leb;
mod locals;
pub mod lower;
pub mod structure;

#[cfg(test)]
mod tests;

pub use binary::{LinkError, LinkOptions, WasmObject};
pub use lower::WasmError;

use crate::codegen::legalize_int::{self, LegalizeOptions};
use crate::codegen::{CodegenOptions, CompiledModule, StackReport};
use crate::ir::{DataLayout, Module, Type};
use crate::mc::object::{ObjectModule, Section, SectionKind};
use crate::support::StrInterner;

/// The name of the one section of the [`ObjectModule`] that
/// [`compile_module_with`] returns: it holds the relocatable wasm object.
pub const OBJECT_SECTION: &str = ".wasm";

/// The wasm32 data layout: ILP32 (32-bit pointers, 4-aligned), `i64`/`f64`
/// 8-aligned, a 16-byte stack, native `i32` and `i64`.
pub fn data_layout() -> DataLayout {
    DataLayout::ilp32()
        .with_stack_align(16)
        .and_then(|l| l.with_native_ints(&[32, 64]))
        .expect("a valid layout")
}

/// A compiled module: the [`WasmObject`] (encode it with
/// [`WasmObject::to_linked`] or [`WasmObject::to_relocatable`]) and the
/// shadow-stack usage of each function.
#[derive(Clone, Debug)]
pub struct Compiled {
    /// The module, ready to encode.
    pub object: WasmObject,
    /// Per-function shadow-stack frames (the static `alloca` area; calls and
    /// wasm's own value stack are not counted).
    pub stack: StackReport,
}

/// Compile `module` for wasm32. The module must use a 32-bit-pointer layout
/// (normally [`data_layout`]).
///
/// # Errors
///
/// A [`WasmError`] naming the unsupported construct (and its function).
pub fn compile(module: &Module, syms: &StrInterner, opts: &CodegenOptions) -> Result<Compiled, WasmError> {
    if opts.reloc_model.is_pic() {
        return Err(WasmError::new(format!(
            "relocation model {:?}: wasm code is position-independent by construction; use the static model",
            opts.reloc_model
        )));
    }
    // Vectors first: wasm32 declares no legal vector type, so every vector
    // op is scalarized (and min/max/saturating ops expanded).
    let vectors = crate::codegen::legalize::legalized(module, &crate::codegen::legalize::ScalarOnly);
    let module: &Module = &vectors;
    let lowered = if needs_legalization(module) {
        // Legalize a copy: the caller's module stays as it is.
        let mut names = StrInterner::new();
        let bytes = crate::ir::binary::encode(module, syms);
        let mut copy = crate::ir::binary::decode(&bytes, &mut names)
            .map_err(|e| WasmError::new(format!("cannot copy the module for legalization: {e}")))?;
        legalize_int::legalize_ints(&mut copy, &mut names, &LegalizeOptions::new(64))
            .map_err(|e| WasmError::new(format!("integer legalization: {e}")))?;
        lower::lower_module(&copy, &names)?
    } else {
        lower::lower_module(module, syms)?
    };
    Ok(Compiled { object: lowered.object, stack: lowered.stack })
}

/// Whether some function computes with an integer wider than 64 bits.
fn needs_legalization(module: &Module) -> bool {
    module.functions().any(|f| {
        (0..f.value_count()).any(|v| {
            matches!(module.types().get(f.value_type(crate::ir::ValueId::from_index(v))), Type::Int(b) if *b > 64)
        })
    })
}

/// Compile `module` into the generic [`CompiledModule`] the other backends
/// return. A wasm module does not fit the neutral object model (its functions
/// are typed and live in index spaces, not at addresses), so the result is an
/// envelope: an [`ObjectModule`] with the relocatable wasm object as its one
/// section, [`OBJECT_SECTION`]; [`crate::mc::format::write_object_as`] with
/// [`ObjectFormat::Wasm`](crate::target::ObjectFormat::Wasm) unwraps it.
///
/// # Errors
///
/// As [`compile`].
pub fn compile_module_with(
    module: &Module,
    syms: &StrInterner,
    opts: &CodegenOptions,
) -> Result<CompiledModule, WasmError> {
    let compiled = compile(module, syms, opts)?;
    let mut obj = ObjectModule::new(module.name.clone());
    let s = obj.add_section(Section::new(OBJECT_SECTION, SectionKind::Debug, 1));
    obj.section_mut(s).bytes = compiled.object.to_relocatable();
    Ok(CompiledModule { object: obj, stack: compiled.stack })
}

/// The relocatable wasm object inside an envelope from
/// [`compile_module_with`], if `obj` is one.
pub fn envelope_bytes(obj: &ObjectModule) -> Option<&[u8]> {
    obj.sections().iter().find(|s| s.name == OBJECT_SECTION).map(|s| s.bytes.as_slice())
}
