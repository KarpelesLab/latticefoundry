//! The wasm backend's output model, [`WasmObject`], and its two binary
//! encodings (WebAssembly Core Specification §5, "Binary Format"):
//!
//! - [`WasmObject::to_linked`] — a **self-contained module**: every function
//!   the program defines, imports from `"env"` for the ones it only declares,
//!   its own memory, stack pointer, function table and data, all resolved;
//! - [`WasmObject::to_relocatable`] — a **relocatable object** for `wasm-ld`,
//!   following the WebAssembly tool-conventions linking format (`Linking.md`):
//!   memory, table and `__stack_pointer` imported from `"env"`, every patchable
//!   index a padded 5-byte LEB, a `linking` custom section (symbol table and
//!   segment info) and `reloc.CODE` / `reloc.DATA` sections.
//!
//! Function bodies are built once, independently of the encoding: a
//! [`Code`] holds the instruction bytes with *holes* ([`Fixup`]s) where an
//! index or address goes, and each encoding fills the holes its own way (the
//! shortest LEB of the resolved value, or a padded LEB plus a relocation).

use std::collections::HashMap;

use super::leb;
use crate::ir::{Linkage, Visibility};

/// A WebAssembly value type (the MVP number types).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ValType {
    /// `i32` (also every pointer, and `i1`/`i8`/`i16` values).
    I32,
    /// `i64`.
    I64,
    /// `f32`.
    F32,
    /// `f64`.
    F64,
}

impl ValType {
    /// The type's binary encoding.
    pub fn code(self) -> u8 {
        match self {
            ValType::I32 => 0x7f,
            ValType::I64 => 0x7e,
            ValType::F32 => 0x7d,
            ValType::F64 => 0x7c,
        }
    }
}

/// A function type: parameters and results.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FuncType {
    /// Parameter types.
    pub params: Vec<ValType>,
    /// Result types (several with multi-value, e.g. an `i128` as two `i64`).
    pub results: Vec<ValType>,
}

/// What a hole in the code refers to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    /// A function index, as the operand of `call` (unsigned LEB;
    /// `R_WASM_FUNCTION_INDEX_LEB`).
    Func(u32),
    /// The table slot of the function with this index, as an `i32.const`
    /// operand: a function pointer (signed LEB; `R_WASM_TABLE_INDEX_SLEB`).
    Table(u32),
    /// A type index, as the operand of `call_indirect` (unsigned LEB;
    /// `R_WASM_TYPE_INDEX_LEB`).
    Type(u32),
    /// The linear-memory address of data symbol `sym` plus `addend`, as an
    /// `i32.const` operand (signed LEB; `R_WASM_MEMORY_ADDR_SLEB`).
    Mem {
        /// Index into [`WasmObject::data_syms`].
        sym: u32,
        /// Byte offset added to the address.
        addend: i32,
    },
    /// The `__stack_pointer` global's index, as the operand of `global.get` /
    /// `global.set` (unsigned LEB; `R_WASM_GLOBAL_INDEX_LEB`).
    StackPointer,
}

impl Target {
    /// The tool-conventions relocation type of this hole.
    fn reloc_type(self) -> u8 {
        match self {
            Target::Func(_) => 0,     // R_WASM_FUNCTION_INDEX_LEB
            Target::Table(_) => 1,    // R_WASM_TABLE_INDEX_SLEB
            Target::Mem { .. } => 4,  // R_WASM_MEMORY_ADDR_SLEB
            Target::Type(_) => 6,     // R_WASM_TYPE_INDEX_LEB
            Target::StackPointer => 7, // R_WASM_GLOBAL_INDEX_LEB
        }
    }

    /// Whether the hole is a signed LEB (an `i32.const` operand).
    fn signed(self) -> bool {
        matches!(self, Target::Table(_) | Target::Mem { .. })
    }
}

/// A hole in a [`Code`] buffer: the encoded value goes *before* byte `at`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fixup {
    /// Position in [`Code::bytes`] where the value is inserted.
    pub at: usize,
    /// What the value is.
    pub target: Target,
}

/// Instruction bytes with holes for indices and addresses.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Code {
    /// The bytes around the holes.
    pub bytes: Vec<u8>,
    /// The holes, in increasing position.
    pub fixups: Vec<Fixup>,
}

impl Code {
    /// Append one byte.
    pub fn byte(&mut self, b: u8) {
        self.bytes.push(b);
    }

    /// Append an unsigned LEB128.
    pub fn u32(&mut self, v: u32) {
        leb::write_u32(&mut self.bytes, v);
    }

    /// Append a signed LEB128 (`s32`).
    pub fn i32(&mut self, v: i32) {
        leb::write_i32(&mut self.bytes, v);
    }

    /// Append a signed LEB128 (`s64`).
    pub fn i64(&mut self, v: i64) {
        leb::write_i64(&mut self.bytes, v);
    }

    /// Record a hole for `target` at the current position.
    pub fn hole(&mut self, target: Target) {
        self.fixups.push(Fixup { at: self.bytes.len(), target });
    }
}

/// A function of the module: imported (declared only) or defined.
#[derive(Clone, Debug)]
pub struct Function {
    /// The symbol name (the import field name, the export name).
    pub name: String,
    /// Index into [`WasmObject::types`].
    pub type_idx: u32,
    /// The body; `None` for an import.
    pub body: Option<Body>,
    /// Symbol binding of a definition.
    pub linkage: Linkage,
    /// Symbol visibility.
    pub visibility: Visibility,
    /// Whether the linked module exports it (and the object marks it
    /// `WASM_SYM_EXPORTED`).
    pub export: bool,
}

/// A function body: its locals beyond the parameters, and its code (ending in
/// the final `end`).
#[derive(Clone, Debug, Default)]
pub struct Body {
    /// The type of each non-parameter local, in index order.
    pub locals: Vec<ValType>,
    /// The expression.
    pub code: Code,
}

/// A reference stored in data.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DataTarget {
    /// The address of a data symbol plus an addend (`R_WASM_MEMORY_ADDR_I32`).
    Mem {
        /// Index into [`WasmObject::data_syms`].
        sym: u32,
        /// Byte offset.
        addend: i32,
    },
    /// The table slot of a function (`R_WASM_TABLE_INDEX_I32`).
    Table(u32),
}

/// A 32-bit field of a data segment holding an address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DataReloc {
    /// Byte offset of the field in its segment.
    pub offset: u32,
    /// What the field points at.
    pub target: DataTarget,
}

/// A data segment: `.rodata`, `.data` or `.bss` of the module.
#[derive(Clone, Debug)]
pub struct DataSegment {
    /// The segment name (`wasm-ld` merges segments by name prefix).
    pub name: String,
    /// Alignment in bytes.
    pub align: u64,
    /// The contents (`.bss`: zeros).
    pub bytes: Vec<u8>,
    /// The address fields.
    pub relocs: Vec<DataReloc>,
}

/// A data symbol: a global's storage, defined in a segment or external.
#[derive(Clone, Debug)]
pub struct DataSymbol {
    /// The symbol name.
    pub name: String,
    /// `(segment, offset, size)` when defined here.
    pub def: Option<(u32, u32, u32)>,
    /// Symbol binding of a definition.
    pub linkage: Linkage,
    /// Symbol visibility.
    pub visibility: Visibility,
}

/// A compiled wasm32 module before encoding (see the [module docs](self)).
#[derive(Clone, Debug, Default)]
pub struct WasmObject {
    /// The function types, deduplicated.
    pub types: Vec<FuncType>,
    type_map: HashMap<FuncType, u32>,
    /// Every function: the imports first (function index = position).
    pub funcs: Vec<Function>,
    /// The data segments.
    pub segments: Vec<DataSegment>,
    /// The data symbols.
    pub data_syms: Vec<DataSymbol>,
    /// Functions whose address is taken, in table order: slot `i + 1` holds
    /// `table[i]` (slot 0 stays empty, so calling a null pointer traps).
    pub table: Vec<u32>,
    /// Whether any code uses `call_indirect`.
    pub uses_indirect: bool,
    /// Whether any code uses the `__stack_pointer` global.
    pub uses_sp: bool,
}

/// How [`WasmObject::to_linked`] lays out linear memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LinkOptions {
    /// Bytes of shadow stack, placed first in memory so that an overflow
    /// wraps below address 0 and traps (out of bounds) instead of corrupting
    /// data. Rounded up to 16.
    pub stack_size: u32,
}

impl Default for LinkOptions {
    fn default() -> LinkOptions {
        LinkOptions { stack_size: 1 << 20 }
    }
}

/// Why a module cannot be linked into a self-contained wasm module.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LinkError(pub String);

impl std::fmt::Display for LinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LinkError {}

/// The WebAssembly page size.
pub const PAGE: u64 = 65536;

/// Section ids (Core Specification §5.5.2).
mod sec {
    pub(super) const CUSTOM: u8 = 0;
    pub(super) const TYPE: u8 = 1;
    pub(super) const IMPORT: u8 = 2;
    pub(super) const FUNCTION: u8 = 3;
    pub(super) const TABLE: u8 = 4;
    pub(super) const MEMORY: u8 = 5;
    pub(super) const GLOBAL: u8 = 6;
    pub(super) const EXPORT: u8 = 7;
    pub(super) const ELEMENT: u8 = 9;
    pub(super) const CODE: u8 = 10;
    pub(super) const DATA: u8 = 11;
}

/// Symbol-table flags (tool-conventions `Linking.md`).
mod symflag {
    pub(super) const BINDING_WEAK: u32 = 0x1;
    pub(super) const BINDING_LOCAL: u32 = 0x2;
    pub(super) const VISIBILITY_HIDDEN: u32 = 0x4;
    pub(super) const UNDEFINED: u32 = 0x10;
    pub(super) const EXPORTED: u32 = 0x20;
}

/// A module under construction: its sections in order.
struct Writer {
    out: Vec<u8>,
    /// How many sections have been written (the next one's index, which the
    /// relocation sections name).
    count: u32,
}

impl Writer {
    fn new() -> Writer {
        // Magic `\0asm`, version 1.
        Writer { out: b"\0asm\x01\x00\x00\x00".to_vec(), count: 0 }
    }

    /// Append section `id` with `payload`, returning its index.
    fn section(&mut self, id: u8, payload: &[u8]) -> u32 {
        self.out.push(id);
        leb::write_u32(&mut self.out, payload.len() as u32);
        self.out.extend_from_slice(payload);
        self.count += 1;
        self.count - 1
    }

    /// Append a custom section `name` with `payload`.
    fn custom(&mut self, name: &str, payload: &[u8]) -> u32 {
        let mut p = Vec::new();
        leb::write_name(&mut p, name);
        p.extend_from_slice(payload);
        self.section(sec::CUSTOM, &p)
    }
}

/// One relocation entry of a `reloc.*` section.
struct Reloc {
    ty: u8,
    offset: u32,
    index: u32,
    addend: Option<i32>,
}

/// Append a vector's element count.
fn count(out: &mut Vec<u8>, n: usize) {
    leb::write_u32(out, n as u32);
}

/// Append the limits `{min}` (no maximum).
fn limits(out: &mut Vec<u8>, min: u64) {
    out.push(0x00);
    leb::write_u64(out, min);
}

impl WasmObject {
    /// An empty module.
    pub fn new() -> WasmObject {
        WasmObject::default()
    }

    /// The index of `ty`, adding it if new.
    pub fn intern_type(&mut self, ty: FuncType) -> u32 {
        if let Some(&i) = self.type_map.get(&ty) {
            return i;
        }
        let i = self.types.len() as u32;
        self.types.push(ty.clone());
        self.type_map.insert(ty, i);
        i
    }

    /// The table slot of function `f`, adding it to the table if new.
    pub fn table_slot(&mut self, f: u32) -> u32 {
        match self.table.iter().position(|&t| t == f) {
            Some(i) => i as u32 + 1,
            None => {
                self.table.push(f);
                self.table.len() as u32
            }
        }
    }

    /// How many functions are imports.
    pub fn import_count(&self) -> usize {
        self.funcs.iter().take_while(|f| f.body.is_none()).count()
    }

    fn type_section(&self) -> Vec<u8> {
        let mut p = Vec::new();
        count(&mut p, self.types.len());
        for t in &self.types {
            p.push(0x60);
            count(&mut p, t.params.len());
            p.extend(t.params.iter().map(|v| v.code()));
            count(&mut p, t.results.len());
            p.extend(t.results.iter().map(|v| v.code()));
        }
        p
    }

    fn function_section(&self) -> Vec<u8> {
        let defined: Vec<&Function> = self.funcs.iter().filter(|f| f.body.is_some()).collect();
        let mut p = Vec::new();
        count(&mut p, defined.len());
        for f in defined {
            leb::write_u32(&mut p, f.type_idx);
        }
        p
    }

    /// The code section's payload, filling each hole with `fill(target, out)`,
    /// which appends the encoded value. Returns the payload and, for each hole,
    /// its offset in the payload.
    fn code_section(&self, fill: &mut dyn FnMut(Target, &mut Vec<u8>)) -> (Vec<u8>, Vec<(u32, Target)>) {
        let defined: Vec<&Body> = self.funcs.iter().filter_map(|f| f.body.as_ref()).collect();
        let mut p = Vec::new();
        let mut holes = Vec::new();
        count(&mut p, defined.len());
        for body in defined {
            let mut b = Vec::new();
            // Locals, run-length encoded by type.
            let mut runs: Vec<(u32, ValType)> = Vec::new();
            for &t in &body.locals {
                match runs.last_mut() {
                    Some((n, rt)) if *rt == t => *n += 1,
                    _ => runs.push((1, t)),
                }
            }
            count(&mut b, runs.len());
            for (n, t) in runs {
                leb::write_u32(&mut b, n);
                b.push(t.code());
            }
            let mut local_holes = Vec::new();
            let mut prev = 0;
            for fx in &body.code.fixups {
                b.extend_from_slice(&body.code.bytes[prev..fx.at]);
                local_holes.push((b.len(), fx.target));
                fill(fx.target, &mut b);
                prev = fx.at;
            }
            b.extend_from_slice(&body.code.bytes[prev..]);
            leb::write_u32(&mut p, b.len() as u32);
            let base = p.len();
            p.extend_from_slice(&b);
            holes.extend(local_holes.into_iter().map(|(at, t)| ((base + at) as u32, t)));
        }
        (p, holes)
    }

    /// Serialize as a self-contained, directly instantiable module (see the
    /// [module docs](self)). Memory layout: the shadow stack at
    /// `[0, stack_size)` (the stack pointer starts at its top), then the data
    /// segments, then the heap (`__heap_base`). Exports `memory`,
    /// `__heap_base` and every function marked for export.
    ///
    /// # Errors
    ///
    /// When a data symbol the code or data refers to is not defined here (an
    /// external global has no storage to point at without a linker).
    pub fn to_linked(&self, opts: &LinkOptions) -> Result<Vec<u8>, LinkError> {
        // Lay out memory: stack, then each segment at its alignment.
        let stack_top = u64::from(opts.stack_size).div_ceil(16) * 16;
        let mut at = stack_top.max(16);
        let mut seg_addr = Vec::with_capacity(self.segments.len());
        for s in &self.segments {
            at = at.div_ceil(s.align.max(1)) * s.align.max(1);
            seg_addr.push(at);
            at += s.bytes.len() as u64;
        }
        let heap_base = at.div_ceil(16) * 16;
        if heap_base > u64::from(u32::MAX) {
            return Err(LinkError("the stack and data do not fit a 32-bit address space".into()));
        }
        let pages = heap_base.div_ceil(PAGE).max(1);
        let sym_addr = |sym: u32, addend: i32| -> Result<i64, LinkError> {
            let s = &self.data_syms[sym as usize];
            let Some((seg, off, _)) = s.def else {
                return Err(LinkError(format!(
                    "undefined global '{}': a self-contained module needs every global defined \
                     (link the relocatable object with wasm-ld instead)",
                    s.name
                )));
            };
            Ok((seg_addr[seg as usize] as i64 + i64::from(off) + i64::from(addend)) as i32 as i64)
        };

        let mut w = Writer::new();
        w.section(sec::TYPE, &self.type_section());

        let nimports = self.import_count();
        if nimports > 0 {
            let mut p = Vec::new();
            count(&mut p, nimports);
            for f in &self.funcs[..nimports] {
                leb::write_name(&mut p, "env");
                leb::write_name(&mut p, &f.name);
                p.push(0x00); // func
                leb::write_u32(&mut p, f.type_idx);
            }
            w.section(sec::IMPORT, &p);
        }
        w.section(sec::FUNCTION, &self.function_section());

        let has_table = self.uses_indirect || !self.table.is_empty();
        if has_table {
            let mut p = Vec::new();
            count(&mut p, 1);
            p.push(0x70); // funcref
            p.push(0x01); // min and max
            leb::write_u32(&mut p, self.table.len() as u32 + 1);
            leb::write_u32(&mut p, self.table.len() as u32 + 1);
            w.section(sec::TABLE, &p);
        }
        {
            let mut p = Vec::new();
            count(&mut p, 1);
            limits(&mut p, pages);
            w.section(sec::MEMORY, &p);
        }
        {
            // Global 0: __stack_pointer (mutable); global 1: __heap_base.
            let mut p = Vec::new();
            count(&mut p, 2);
            p.extend_from_slice(&[ValType::I32.code(), 0x01, 0x41]);
            leb::write_i32(&mut p, stack_top as u32 as i32);
            p.push(0x0b);
            p.extend_from_slice(&[ValType::I32.code(), 0x00, 0x41]);
            leb::write_i32(&mut p, heap_base as u32 as i32);
            p.push(0x0b);
            w.section(sec::GLOBAL, &p);
        }
        {
            // Export names are unique: a function called `memory` or
            // `__heap_base` stays unexported.
            let exported: Vec<(usize, &Function)> = self
                .funcs
                .iter()
                .enumerate()
                .filter(|(_, f)| f.export && f.body.is_some() && !matches!(f.name.as_str(), "memory" | "__heap_base"))
                .collect();
            let mut p = Vec::new();
            count(&mut p, exported.len() + 2);
            leb::write_name(&mut p, "memory");
            p.push(0x02);
            leb::write_u32(&mut p, 0);
            leb::write_name(&mut p, "__heap_base");
            p.push(0x03);
            leb::write_u32(&mut p, 1);
            for (i, f) in exported {
                leb::write_name(&mut p, &f.name);
                p.push(0x00);
                leb::write_u32(&mut p, i as u32);
            }
            w.section(sec::EXPORT, &p);
        }
        if !self.table.is_empty() {
            let mut p = Vec::new();
            count(&mut p, 1);
            p.push(0x00); // active, table 0, funcref, offset expression
            p.push(0x41);
            leb::write_i32(&mut p, 1);
            p.push(0x0b);
            count(&mut p, self.table.len());
            for &f in &self.table {
                leb::write_u32(&mut p, f);
            }
            w.section(sec::ELEMENT, &p);
        }

        let mut err = None;
        let slot = |f: u32| self.table.iter().position(|&t| t == f).map_or(0, |i| i as i32 + 1);
        let (code, _) = self.code_section(&mut |t, out| match t {
            Target::Func(i) | Target::Type(i) => leb::write_u32(out, i),
            Target::StackPointer => leb::write_u32(out, 0),
            Target::Table(f) => leb::write_i32(out, slot(f)),
            Target::Mem { sym, addend } => match sym_addr(sym, addend) {
                Ok(a) => leb::write_i64(out, a),
                Err(e) => {
                    err.get_or_insert(e);
                    out.push(0);
                }
            },
        });
        if let Some(e) = err {
            return Err(e);
        }
        w.section(sec::CODE, &code);

        let live: Vec<usize> = (0..self.segments.len()).filter(|&i| !is_bss(&self.segments[i])).collect();
        if !live.is_empty() {
            let mut p = Vec::new();
            count(&mut p, live.len());
            for i in live {
                let s = &self.segments[i];
                let mut bytes = s.bytes.clone();
                for r in &s.relocs {
                    let v = match r.target {
                        DataTarget::Mem { sym, addend } => sym_addr(sym, addend)? as u32,
                        DataTarget::Table(f) => slot(f) as u32,
                    };
                    let o = r.offset as usize;
                    bytes[o..o + 4].copy_from_slice(&v.to_le_bytes());
                }
                p.push(0x00); // active, memory 0
                p.push(0x41);
                leb::write_i64(&mut p, seg_addr[i] as u32 as i32 as i64);
                p.push(0x0b);
                count(&mut p, bytes.len());
                p.extend_from_slice(&bytes);
            }
            w.section(sec::DATA, &p);
        }

        // The `name` section: function names, for stack traces.
        let mut names = Vec::new();
        count(&mut names, self.funcs.len());
        for (i, f) in self.funcs.iter().enumerate() {
            leb::write_u32(&mut names, i as u32);
            leb::write_name(&mut names, &f.name);
        }
        let mut p = vec![1u8]; // subsection 1: function names
        leb::write_u32(&mut p, names.len() as u32);
        p.extend_from_slice(&names);
        w.custom("name", &p);
        Ok(w.out)
    }

    /// Serialize as a relocatable object for `wasm-ld` (see the [module
    /// docs](self)). Symbols: every function (symbol index = function index),
    /// then every data symbol, then `__stack_pointer` when used.
    pub fn to_relocatable(&self) -> Vec<u8> {
        let nf = self.funcs.len() as u32;
        let nd = self.data_syms.len() as u32;
        let sp_sym = nf + nd;

        let mut w = Writer::new();
        w.section(sec::TYPE, &self.type_section());

        let nimports = self.import_count();
        {
            let has_table = self.uses_indirect;
            let mut p = Vec::new();
            count(&mut p, nimports + 1 + usize::from(self.uses_sp) + usize::from(has_table));
            leb::write_name(&mut p, "env");
            leb::write_name(&mut p, "__linear_memory");
            p.push(0x02);
            let size: u64 = self.segments.iter().map(|s| s.bytes.len() as u64 + s.align).sum();
            limits(&mut p, size.div_ceil(PAGE));
            if self.uses_sp {
                leb::write_name(&mut p, "env");
                leb::write_name(&mut p, "__stack_pointer");
                p.extend_from_slice(&[0x03, ValType::I32.code(), 0x01]);
            }
            if has_table {
                leb::write_name(&mut p, "env");
                leb::write_name(&mut p, "__indirect_function_table");
                p.extend_from_slice(&[0x01, 0x70]);
                limits(&mut p, 0);
            }
            for f in &self.funcs[..nimports] {
                leb::write_name(&mut p, "env");
                leb::write_name(&mut p, &f.name);
                p.push(0x00);
                leb::write_u32(&mut p, f.type_idx);
            }
            w.section(sec::IMPORT, &p);
        }
        w.section(sec::FUNCTION, &self.function_section());

        let (code, holes) = self.code_section(&mut |t, out| {
            let v = match t {
                Target::Func(i) | Target::Type(i) => i as i32,
                _ => 0,
            };
            if t.signed() { leb::write_i32_padded(out, v) } else { leb::write_u32_padded(out, v as u32) }
        });
        let code_idx = w.section(sec::CODE, &code);
        let code_relocs: Vec<Reloc> = holes
            .into_iter()
            .map(|(offset, t)| {
                let (index, addend) = match t {
                    Target::Func(f) | Target::Table(f) => (f, None),
                    Target::Type(ty) => (ty, None),
                    Target::Mem { sym, addend } => (nf + sym, Some(addend)),
                    Target::StackPointer => (sp_sym, None),
                };
                Reloc { ty: t.reloc_type(), offset, index, addend }
            })
            .collect();

        // Segments are laid out back to back (the linker moves them anyway).
        let mut data_relocs = Vec::new();
        let mut data_idx = None;
        if !self.segments.is_empty() {
            let mut p = Vec::new();
            count(&mut p, self.segments.len());
            let mut at = 0u64;
            for s in &self.segments {
                at = at.div_ceil(s.align.max(1)) * s.align.max(1);
                p.push(0x00);
                p.push(0x41);
                leb::write_i64(&mut p, at as i64);
                p.push(0x0b);
                count(&mut p, s.bytes.len());
                let base = p.len() as u32;
                p.extend_from_slice(&s.bytes);
                for r in &s.relocs {
                    let (ty, index, addend) = match r.target {
                        DataTarget::Mem { sym, addend } => (5, nf + sym, Some(addend)), // R_WASM_MEMORY_ADDR_I32
                        DataTarget::Table(f) => (2, f, None),                          // R_WASM_TABLE_INDEX_I32
                    };
                    data_relocs.push(Reloc { ty, offset: base + r.offset, index, addend });
                }
                at += s.bytes.len() as u64;
            }
            data_idx = Some(w.section(sec::DATA, &p));
        }

        // The `linking` section: version 2, a symbol table, segment info.
        let mut link = Vec::new();
        leb::write_u32(&mut link, 2);
        let mut table = Vec::new();
        count(&mut table, (sp_sym + u32::from(self.uses_sp)) as usize);
        for (i, f) in self.funcs.iter().enumerate() {
            table.push(0x00); // SYMTAB_FUNCTION
            let mut flags = binding_flags(f.linkage, f.visibility);
            if f.body.is_none() {
                flags = symflag::UNDEFINED;
            } else if f.export {
                flags |= symflag::EXPORTED;
            }
            leb::write_u32(&mut table, flags);
            leb::write_u32(&mut table, i as u32);
            if f.body.is_some() {
                leb::write_name(&mut table, &f.name);
            }
        }
        for s in &self.data_syms {
            table.push(0x01); // SYMTAB_DATA
            let flags = if s.def.is_some() { binding_flags(s.linkage, s.visibility) } else { symflag::UNDEFINED };
            leb::write_u32(&mut table, flags);
            leb::write_name(&mut table, &s.name);
            if let Some((seg, off, size)) = s.def {
                leb::write_u32(&mut table, seg);
                leb::write_u32(&mut table, off);
                leb::write_u32(&mut table, size);
            }
        }
        if self.uses_sp {
            table.push(0x02); // SYMTAB_GLOBAL
            leb::write_u32(&mut table, symflag::UNDEFINED);
            leb::write_u32(&mut table, 0);
        }
        link.push(8); // WASM_SYMBOL_TABLE
        leb::write_u32(&mut link, table.len() as u32);
        link.extend_from_slice(&table);
        if !self.segments.is_empty() {
            let mut info = Vec::new();
            count(&mut info, self.segments.len());
            for s in &self.segments {
                leb::write_name(&mut info, &s.name);
                leb::write_u32(&mut info, s.align.max(1).trailing_zeros());
                leb::write_u32(&mut info, 0);
            }
            link.push(5); // WASM_SEGMENT_INFO
            leb::write_u32(&mut link, info.len() as u32);
            link.extend_from_slice(&info);
        }
        w.custom("linking", &link);

        for (name, idx, relocs) in [("reloc.CODE", Some(code_idx), code_relocs), ("reloc.DATA", data_idx, data_relocs)] {
            let Some(idx) = idx else { continue };
            if relocs.is_empty() {
                continue;
            }
            let mut p = Vec::new();
            leb::write_u32(&mut p, idx);
            count(&mut p, relocs.len());
            for r in relocs {
                p.push(r.ty);
                leb::write_u32(&mut p, r.offset);
                leb::write_u32(&mut p, r.index);
                if let Some(a) = r.addend {
                    leb::write_i32(&mut p, a);
                }
            }
            w.custom(name, &p);
        }
        w.out
    }
}

/// Whether a segment is all zeros and named like `.bss` (a linked module
/// leaves it out: memory starts zeroed).
fn is_bss(s: &DataSegment) -> bool {
    s.name.starts_with(".bss") && s.bytes.iter().all(|&b| b == 0) && s.relocs.is_empty()
}

/// The binding and visibility flags of a defined symbol.
fn binding_flags(linkage: Linkage, visibility: Visibility) -> u32 {
    let mut flags = match linkage {
        Linkage::External => 0,
        Linkage::Internal => symflag::BINDING_LOCAL,
        Linkage::Weak => symflag::BINDING_WEAK,
    };
    if visibility == Visibility::Hidden {
        flags |= symflag::VISIBILITY_HIDDEN;
    }
    flags
}
