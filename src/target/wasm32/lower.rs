//! Lowering the SSA IR straight to WebAssembly: module layout (function
//! indices, imports, the function table, data) and per-function code.
//!
//! WebAssembly is a typed stack machine with an unbounded set of typed
//! *locals*, so there is no instruction selection into a register machine and
//! no register allocation: each SSA value lives in a local of its wasm type,
//! each block parameter is a local assigned on the incoming edges, and each
//! instruction is emitted as stack code that reads its operands from locals
//! (or recomputes a single-use pure expression in place) and stores its result
//! into its local. Control flow comes from the [structurizer](super::structure).
//!
//! # Values
//!
//! | IR type | wasm | invariant |
//! |---|---|---|
//! | `i1`..`i32`, `ptr`, aggregates (addresses) | one `i32` | zero-extended: the bits above the width are 0 |
//! | `i33`..`i64` | one `i64` | zero-extended |
//! | `f32` / `f64` | `f32` / `f64` | |
//! | `i128`, `i192`, … | `n` `i64` locals, least significant first | |
//!
//! The zero-extension invariant is what keeps narrow values right (the
//! "narrow values" bug class of the register backends): an operation whose
//! result can carry bits above the width (`add`, `sub`, `mul`, `shl`, signed
//! division, `ashr`, conversions) masks it, and an operation that reads the
//! value as signed (`sdiv`, `srem`, `ashr`, signed compares, `sext`,
//! `sitofp`) sign-extends its operands first (`i32.extend8_s` and friends, or a
//! shift pair). Unsigned operations and equality then need nothing. Values that
//! may come from outside — the parameters of a non-`internal` function, and
//! the results of imported or indirect calls — are masked on arrival, so a
//! host passing a sloppy `i8` cannot break the invariant.
//!
//! Integers wider than 64 bits are split by [`legalize_ints`](crate::codegen::legalize_int::legalize_ints)
//! (part width 64) before lowering. What the pass leaves wide — parameters,
//! call arguments and results, returns, and the split/join shapes around them —
//! lives in groups of `i64` locals; a wide parameter is several `i64`
//! parameters and a wide result several results (multi-value).
//!
//! # Memory
//!
//! Linear memory is address space 0 (the layout is ILP32 with native `i64`).
//! Globals live in data segments; code takes a global's address with an
//! `i32.const` the linker (or [`WasmObject::to_linked`]) fills in, and a
//! function's address is its slot in the function table (`call_indirect`
//! goes through it; slot 0 is left empty so a null call traps).
//!
//! `alloca` and `dyn_alloca` use a **shadow stack** in linear memory: the
//! mutable global `__stack_pointer` points at its top and grows down. A
//! function with a frame saves the incoming stack pointer in a local, lowers
//! it by its 16-aligned static frame (addressed from a frame-pointer local),
//! bumps it further for each `dyn_alloca`, and restores the saved value before
//! every `return`.

use std::collections::HashMap;

use super::binary::{
    Body, Code, DataReloc, DataSegment, DataSymbol, DataTarget, FuncType, Function as WFunc, Target, ValType,
    WasmObject,
};
use super::structure::{self, Arm, CfgNode, Shape, Structured};
use crate::codegen::stack::{StackReport, StackUsage};
use crate::ir::inst::{
    BinOp, CastOp, FloatPred, InstData, InstId, InstKind, IntPred, RmwOp, UnaryOp,
};
use crate::ir::types::{FloatKind, Type, TypeContext, TypeId};
use crate::ir::value::{AddrTarget, Const, FloatBits, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, GlobalId, Linkage, Module, Visibility};
use crate::mc::object::{ObjectModule, RelocKind, SymbolBinding, SymbolValue};
use crate::support::StrInterner;

use puremp::Int;

/// Why the wasm32 backend cannot compile a module.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WasmError {
    /// The function being compiled, if the problem is inside one.
    pub function: Option<String>,
    /// What is unsupported or wrong.
    pub message: String,
}

impl WasmError {
    /// A module-level error.
    pub fn new(message: impl Into<String>) -> WasmError {
        WasmError { function: None, message: message.into() }
    }
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.function {
            Some(func) => write!(f, "wasm32: in function '{func}': {}", self.message),
            None => write!(f, "wasm32: {}", self.message),
        }
    }
}

impl std::error::Error for WasmError {}

type R<T> = Result<T, WasmError>;

fn fail<T>(message: impl Into<String>) -> R<T> {
    Err(WasmError::new(message))
}

/// How an IR value is held in wasm (see the [module docs](self)).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Repr {
    /// No value.
    Void,
    /// An `i32` holding an integer of this many bits (1..=32), zero-extended.
    I32(u32),
    /// An `i64` holding an integer of this many bits (33..=64), zero-extended.
    I64(u32),
    /// An `f32`.
    F32,
    /// An `f64`.
    F64,
    /// This many `i64` parts, least significant first.
    Wide(u32),
}

impl Repr {
    /// The representation of IR type `ty`.
    pub(crate) fn of(types: &TypeContext, ty: TypeId) -> R<Repr> {
        Ok(match types.get(ty) {
            Type::Void => Repr::Void,
            &Type::Int(bits) if bits <= 32 => Repr::I32(bits),
            &Type::Int(bits) if bits <= 64 => Repr::I64(bits),
            &Type::Int(bits) if bits.is_multiple_of(64) => Repr::Wide(bits / 64),
            &Type::Int(bits) => return fail(format!("i{bits} is wider than 64 bits and not a multiple of 64")),
            Type::Float(FloatKind::F32) => Repr::F32,
            Type::Float(FloatKind::F64) => Repr::F64,
            Type::Float(FloatKind::F16) => return fail("f16 has no WebAssembly lowering"),
            Type::Ptr | Type::Func(_) | Type::Struct(_) | Type::Array(..) => Repr::I32(32),
            Type::PtrIn(n) => return fail(format!("address space {n}: wasm32 has only linear memory (space 0)")),
            Type::Vector(..) => return fail("a vector type survived vector legalization"),
        })
    }

    /// The wasm value types holding it.
    pub(crate) fn valtypes(self) -> Vec<ValType> {
        match self {
            Repr::Void => Vec::new(),
            Repr::I32(_) => vec![ValType::I32],
            Repr::I64(_) => vec![ValType::I64],
            Repr::F32 => vec![ValType::F32],
            Repr::F64 => vec![ValType::F64],
            Repr::Wide(n) => vec![ValType::I64; n as usize],
        }
    }

    /// Whether the value lives in an `i64` (for choosing opcodes).
    fn is64(self) -> bool {
        matches!(self, Repr::I64(_) | Repr::F64 | Repr::Wide(_))
    }
}

/// The wasm function type of IR signature `sig`.
fn func_type(types: &TypeContext, sig: TypeId) -> R<(FuncType, bool)> {
    let Type::Func(ft) = types.get(sig) else { return fail("a function's signature is not a function type") };
    let mut params = Vec::new();
    for &p in &ft.params {
        params.extend(Repr::of(types, p)?.valtypes());
    }
    let results = Repr::of(types, ft.ret)?.valtypes();
    Ok((FuncType { params, results }, ft.variadic))
}

/// Wasm opcodes (Core Specification §5.4), named after their text format.
#[allow(dead_code)]
mod op {
    pub(super) const UNREACHABLE: u8 = 0x00;
    pub(super) const BLOCK: u8 = 0x02;
    pub(super) const LOOP: u8 = 0x03;
    pub(super) const IF: u8 = 0x04;
    pub(super) const ELSE: u8 = 0x05;
    pub(super) const END: u8 = 0x0b;
    pub(super) const BR: u8 = 0x0c;
    pub(super) const BR_IF: u8 = 0x0d;
    pub(super) const BR_TABLE: u8 = 0x0e;
    pub(super) const RETURN: u8 = 0x0f;
    pub(super) const CALL: u8 = 0x10;
    pub(super) const CALL_INDIRECT: u8 = 0x11;
    pub(super) const DROP: u8 = 0x1a;
    pub(super) const SELECT: u8 = 0x1b;
    pub(super) const LOCAL_GET: u8 = 0x20;
    pub(super) const LOCAL_SET: u8 = 0x21;
    pub(super) const LOCAL_TEE: u8 = 0x22;
    pub(super) const GLOBAL_GET: u8 = 0x23;
    pub(super) const GLOBAL_SET: u8 = 0x24;
    pub(super) const I32_LOAD: u8 = 0x28;
    pub(super) const I64_LOAD: u8 = 0x29;
    pub(super) const F32_LOAD: u8 = 0x2a;
    pub(super) const F64_LOAD: u8 = 0x2b;
    pub(super) const I32_LOAD8_U: u8 = 0x2d;
    pub(super) const I32_LOAD16_U: u8 = 0x2f;
    pub(super) const I64_LOAD8_U: u8 = 0x31;
    pub(super) const I64_LOAD16_U: u8 = 0x33;
    pub(super) const I64_LOAD32_U: u8 = 0x35;
    pub(super) const I32_STORE: u8 = 0x36;
    pub(super) const I64_STORE: u8 = 0x37;
    pub(super) const F32_STORE: u8 = 0x38;
    pub(super) const F64_STORE: u8 = 0x39;
    pub(super) const I32_STORE8: u8 = 0x3a;
    pub(super) const I32_STORE16: u8 = 0x3b;
    pub(super) const I64_STORE8: u8 = 0x3c;
    pub(super) const I64_STORE16: u8 = 0x3d;
    pub(super) const I64_STORE32: u8 = 0x3e;
    pub(super) const I32_CONST: u8 = 0x41;
    pub(super) const I64_CONST: u8 = 0x42;
    pub(super) const F32_CONST: u8 = 0x43;
    pub(super) const F64_CONST: u8 = 0x44;
    pub(super) const I32_EQZ: u8 = 0x45;
    pub(super) const I32_EQ: u8 = 0x46;
    pub(super) const I32_NE: u8 = 0x47;
    pub(super) const I32_LT_U: u8 = 0x49;
    pub(super) const I64_EQZ: u8 = 0x50;
    pub(super) const I64_LT_U: u8 = 0x54;
    pub(super) const I32_ADD: u8 = 0x6a;
    pub(super) const I32_SUB: u8 = 0x6b;
    pub(super) const I32_AND: u8 = 0x71;
    pub(super) const I32_OR: u8 = 0x72;
    pub(super) const I32_XOR: u8 = 0x73;
    pub(super) const I32_SHL: u8 = 0x74;
    pub(super) const I32_SHR_S: u8 = 0x75;
    pub(super) const I32_SHR_U: u8 = 0x76;
    pub(super) const I64_ADD: u8 = 0x7c;
    pub(super) const I64_SUB: u8 = 0x7d;
    pub(super) const I64_AND: u8 = 0x83;
    pub(super) const I64_OR: u8 = 0x84;
    pub(super) const I64_XOR: u8 = 0x85;
    pub(super) const I64_SHL: u8 = 0x86;
    pub(super) const I64_SHR_S: u8 = 0x87;
    pub(super) const I64_SHR_U: u8 = 0x88;
    pub(super) const F32_NEG: u8 = 0x8c;
    pub(super) const F64_NEG: u8 = 0x9a;
    pub(super) const I32_WRAP_I64: u8 = 0xa7;
    pub(super) const I64_EXTEND_I32_S: u8 = 0xac;
    pub(super) const I64_EXTEND_I32_U: u8 = 0xad;
    pub(super) const F32_DEMOTE_F64: u8 = 0xb6;
    pub(super) const F64_PROMOTE_F32: u8 = 0xbb;
    pub(super) const I32_REINTERPRET_F32: u8 = 0xbc;
    pub(super) const I64_REINTERPRET_F64: u8 = 0xbd;
    pub(super) const F32_REINTERPRET_I32: u8 = 0xbe;
    pub(super) const F64_REINTERPRET_I64: u8 = 0xbf;
    pub(super) const I32_EXTEND8_S: u8 = 0xc0;
    pub(super) const I32_EXTEND16_S: u8 = 0xc1;
    pub(super) const I64_EXTEND8_S: u8 = 0xc2;
    pub(super) const I64_EXTEND16_S: u8 = 0xc3;
    pub(super) const I64_EXTEND32_S: u8 = 0xc4;
    /// The `0xFC` prefix (saturating truncation, bulk memory).
    pub(super) const PREFIX_FC: u8 = 0xfc;
    /// The `0xFE` prefix (threads: atomics).
    pub(super) const PREFIX_FE: u8 = 0xfe;
    /// The empty block type.
    pub(super) const EMPTY: u8 = 0x40;
}

/// The `(i32, i64)` opcodes of an integer comparison.
fn icmp_op(pred: IntPred) -> (u8, u8) {
    match pred {
        IntPred::Eq => (0x46, 0x51),
        IntPred::Ne => (0x47, 0x52),
        IntPred::Slt => (0x48, 0x53),
        IntPred::Ult => (0x49, 0x54),
        IntPred::Sgt => (0x4a, 0x55),
        IntPred::Ugt => (0x4b, 0x56),
        IntPred::Sle => (0x4c, 0x57),
        IntPred::Ule => (0x4d, 0x58),
        IntPred::Sge => (0x4e, 0x59),
        IntPred::Uge => (0x4f, 0x5a),
    }
}

fn is_signed(pred: IntPred) -> bool {
    matches!(pred, IntPred::Slt | IntPred::Sgt | IntPred::Sle | IntPred::Sge)
}

/// The `(i32, i64)` opcodes of an integer binary operation.
fn int_bin_op(op: BinOp) -> (u8, u8) {
    match op {
        BinOp::Add => (0x6a, 0x7c),
        BinOp::Sub => (0x6b, 0x7d),
        BinOp::Mul => (0x6c, 0x7e),
        BinOp::SDiv => (0x6d, 0x7f),
        BinOp::UDiv => (0x6e, 0x80),
        BinOp::SRem => (0x6f, 0x81),
        BinOp::URem => (0x70, 0x82),
        BinOp::And => (0x71, 0x83),
        BinOp::Or => (0x72, 0x84),
        BinOp::Xor => (0x73, 0x85),
        BinOp::Shl => (0x74, 0x86),
        BinOp::AShr => (0x75, 0x87),
        BinOp::LShr => (0x76, 0x88),
        _ => unreachable!("float op"),
    }
}

/// Module-wide lowering state: the object under construction and the maps
/// from IR entities to wasm indices.
pub(crate) struct Cx<'a> {
    module: &'a Module,
    syms: &'a StrInterner,
    obj: WasmObject,
    /// IR function → wasm function index (`None`: an unreferenced declaration).
    func_index: Vec<Option<u32>>,
    /// IR global → data symbol index.
    global_sym: Vec<u32>,
    /// The `fmod` / `fmodf` functions `frem` calls, when used.
    fmod: [Option<u32>; 2],
}

/// The result of lowering a module.
pub(crate) struct Lowered {
    pub(crate) object: WasmObject,
    pub(crate) stack: StackReport,
}

/// Lower `module` (already legalized to 64-bit parts) into a [`WasmObject`].
pub(crate) fn lower_module(module: &Module, syms: &StrInterner) -> R<Lowered> {
    let dl = module.data_layout();
    if dl.pointer_bits(0) != 32 {
        return fail(format!(
            "the module's data layout has {}-bit pointers; wasm32 needs 32 (use target::wasm32::data_layout())",
            dl.pointer_bits(0)
        ));
    }
    if dl.program_addr_space() != 0 {
        return fail("functions must live in address space 0 on wasm32");
    }
    let types = module.types();
    let nfuncs = module.function_count();

    // 1. Which declarations are referenced (they become imports), and whether
    //    `frem` needs `fmod`/`fmodf`.
    let mut referenced = vec![false; nfuncs];
    let mut need_fmod = [false; 2];
    for f in module.functions() {
        if f.is_declaration() {
            continue;
        }
        for vi in 0..f.value_count() {
            if let ValueDef::Func(fid) = f.value(ValueId::from_index(vi)).def {
                referenced[fid.index()] = true;
            }
        }
        for (_, block) in f.blocks() {
            for &i in block.insts() {
                let inst = f.inst(i);
                if inst.kind == InstKind::Bin(BinOp::FRem) {
                    need_fmod[usize::from(matches!(types.get(inst.ty), Type::Float(FloatKind::F32)))] = true;
                }
            }
        }
    }
    for g in module.globals() {
        if let Some(init) = g.init {
            mark_addr_funcs(module, init, &mut referenced);
        }
    }
    let by_name = |name: &str| -> Option<FuncId> {
        (0..nfuncs).map(FuncId::from_index).find(|&f| syms.resolve(module.function(f).name) == name)
    };
    let fmod_ids: [Option<FuncId>; 2] = [
        if need_fmod[0] { by_name("fmod") } else { None },
        if need_fmod[1] { by_name("fmodf") } else { None },
    ];
    for f in fmod_ids.into_iter().flatten() {
        referenced[f.index()] = true;
    }

    let mut cx = Cx {
        module,
        syms,
        obj: WasmObject::new(),
        func_index: vec![None; nfuncs],
        global_sym: Vec::new(),
        fmod: [None; 2],
    };

    // 2. Function indices: imports first, then the definitions.
    for (i, f) in module.functions().enumerate() {
        if !f.is_declaration() || !referenced[i] {
            continue;
        }
        let name = syms.resolve(f.name).to_owned();
        let (ty, _) = func_type(types, f.sig).map_err(|e| WasmError { function: Some(name.clone()), ..e })?;
        cx.func_index[i] = Some(cx.obj.funcs.len() as u32);
        let type_idx = cx.obj.intern_type(ty);
        cx.obj.funcs.push(WFunc {
            name,
            type_idx,
            body: None,
            linkage: Linkage::External,
            visibility: f.attrs.visibility,
            export: false,
        });
    }
    let mut synthesized: [Option<u32>; 2] = [None; 2];
    for (k, (name, vt)) in [("fmod", ValType::F64), ("fmodf", ValType::F32)].into_iter().enumerate() {
        if !need_fmod[k] || fmod_ids[k].is_some() {
            continue;
        }
        let type_idx = cx.obj.intern_type(FuncType { params: vec![vt, vt], results: vec![vt] });
        synthesized[k] = Some(cx.obj.funcs.len() as u32);
        cx.obj.funcs.push(WFunc {
            name: name.to_owned(),
            type_idx,
            body: None,
            linkage: Linkage::External,
            visibility: Visibility::Default,
            export: false,
        });
    }
    let mut defined = Vec::new();
    for (i, f) in module.functions().enumerate() {
        if f.is_declaration() {
            continue;
        }
        let name = syms.resolve(f.name).to_owned();
        let (ty, _) = func_type(types, f.sig).map_err(|e| WasmError { function: Some(name.clone()), ..e })?;
        cx.func_index[i] = Some(cx.obj.funcs.len() as u32);
        let type_idx = cx.obj.intern_type(ty);
        let export = name == "main"
            || (f.attrs.linkage != Linkage::Internal && f.attrs.visibility != Visibility::Hidden);
        cx.obj.funcs.push(WFunc {
            name,
            type_idx,
            body: None,
            linkage: f.attrs.linkage,
            visibility: f.attrs.visibility,
            export,
        });
        defined.push(FuncId::from_index(i));
    }
    // `frem` calls the module's own `fmod`/`fmodf` when it has one.
    for k in 0..2 {
        cx.fmod[k] = fmod_ids[k].and_then(|f| cx.func_index[f.index()]).or(synthesized[k]);
    }

    // 3. Data.
    lower_data(&mut cx)?;

    // 4. Code.
    let mut stack = StackReport::new();
    for fid in defined {
        let f = module.function(fid);
        let name = syms.resolve(f.name).to_owned();
        let (body, usage) =
            FnLower::run(&mut cx, f).map_err(|e| WasmError { function: Some(name.clone()), message: e.message })?;
        let idx = cx.func_index[fid.index()].expect("defined") as usize;
        cx.obj.funcs[idx].body = Some(body);
        stack.push(usage);
    }
    Ok(Lowered { object: cx.obj, stack })
}

/// Mark every function an initializer takes the address of.
fn mark_addr_funcs(module: &Module, c: crate::ir::ConstId, referenced: &mut [bool]) {
    match module.consts().get(c) {
        Const::Addr { target: AddrTarget::Func(f), .. } => referenced[f.index()] = true,
        Const::Aggregate { elems, .. } => {
            for &e in elems {
                mark_addr_funcs(module, e, referenced);
            }
        }
        _ => {}
    }
}

/// Serialize the module's globals (through the shared
/// [`codegen::data`](crate::codegen::data) emitter, with 32-bit absolute
/// relocations) and turn the result into data segments and symbols.
fn lower_data(cx: &mut Cx<'_>) -> R<()> {
    let module = cx.module;
    for (gi, _) in module.globals().enumerate() {
        let space = module.global_addr_space(GlobalId::from_index(gi));
        if space != 0 {
            return fail(format!("a global in address space {space}: wasm32 has only linear memory"));
        }
    }
    let mut scratch = ObjectModule::new("data");
    crate::codegen::data::emit_globals(module, cx.syms, &mut scratch, RelocKind::Abs32);

    let mut sym_index: HashMap<String, u32> = HashMap::new();
    for s in scratch.sections() {
        cx.obj.segments.push(DataSegment {
            name: s.name.clone(),
            align: s.align,
            bytes: if s.is_nobits() { vec![0; s.size() as usize] } else { s.bytes.clone() },
            relocs: Vec::new(),
        });
    }
    let attrs_of = |name: &str| {
        module
            .globals()
            .enumerate()
            .find(|(_, g)| cx.syms.resolve(g.name) == name)
            .map(|(i, _)| module.global_attrs(GlobalId::from_index(i)))
    };
    for s in scratch.symbols() {
        let SymbolValue::Defined { section, offset } = s.value else { continue };
        let linkage = match s.binding {
            SymbolBinding::Local => Linkage::Internal,
            SymbolBinding::Global => Linkage::External,
            SymbolBinding::Weak => Linkage::Weak,
        };
        let visibility = attrs_of(&s.name).map_or(Visibility::Default, |a| a.visibility);
        sym_index.insert(s.name.clone(), cx.obj.data_syms.len() as u32);
        cx.obj.data_syms.push(DataSymbol {
            name: s.name.clone(),
            def: Some((section.index() as u32, offset as u32, s.size as u32)),
            linkage,
            visibility,
        });
    }
    // Every IR global gets a symbol (an undefined one without storage here).
    for g in module.globals() {
        let name = cx.syms.resolve(g.name).to_owned();
        let idx = match sym_index.get(&name) {
            Some(&i) => i,
            None => {
                let i = cx.obj.data_syms.len() as u32;
                cx.obj.data_syms.push(DataSymbol {
                    name: name.clone(),
                    def: None,
                    linkage: Linkage::External,
                    visibility: Visibility::Default,
                });
                sym_index.insert(name, i);
                i
            }
        };
        cx.global_sym.push(idx);
    }
    // Address fields: a function's table slot, or a data symbol's address.
    let func_by_name: HashMap<&str, usize> =
        module.functions().enumerate().map(|(i, f)| (cx.syms.resolve(f.name), i)).collect();
    for r in scratch.relocations() {
        let name = scratch.symbol(r.symbol).name.clone();
        let target = if let Some(&fi) = func_by_name.get(name.as_str()) {
            if r.addend != 0 {
                return fail(format!("an address constant '{name} + {}': a function's address has no offset on wasm32", r.addend));
            }
            let idx = cx.func_index[fi].expect("address-taken functions are referenced");
            cx.obj.table_slot(idx);
            DataTarget::Table(idx)
        } else {
            let Some(&sym) = sym_index.get(&name) else { return fail(format!("unknown symbol '{name}' in data")) };
            DataTarget::Mem { sym, addend: r.addend as i32 }
        };
        cx.obj.segments[r.section.index()].relocs.push(DataReloc { offset: r.offset as u32, target });
    }
    Ok(())
}

/// One outgoing edge of a block: its target and arguments.
#[derive(Clone, PartialEq, Eq)]
struct Edge {
    target: BlockId,
    args: Vec<ValueId>,
}

/// How a block's terminator leaves it.
enum Exit {
    Ret(Option<ValueId>),
    Unreachable,
    /// Arms in [`Shape::Direct`] (one) or [`Shape::IfElse`] (the condition).
    Branch(Option<ValueId>),
    /// A switch in [`Shape::Table`]: the condition, the default's arm, and each
    /// case's value and arm.
    Switch { cond: ValueId, default: usize, cases: Vec<(Int, usize)> },
}

/// A block's terminator as the structurizer and emitter see it.
struct Term {
    exit: Exit,
    edges: Vec<Edge>,
    shape: Shape,
}

/// Per-function lowering state.
struct FnLower<'a, 'c> {
    cx: &'c mut Cx<'a>,
    f: &'a Function,
    types: &'a TypeContext,
    code: Code,
    nparams: u32,
    locals: Vec<ValType>,
    /// Each value's local(s) (empty: not held in a local).
    loc: Vec<Vec<u32>>,
    /// Values recomputed at their single use instead of stored.
    inline: Vec<bool>,
    terms: Vec<Term>,
    /// Frame: the saved stack pointer and the frame pointer locals.
    saved_sp: Option<u32>,
    fp: Option<u32>,
    frame_size: u64,
    alloca_off: HashMap<InstId, u64>,
    /// The dispatch-loop label local.
    label: Option<u32>,
    /// Structurizer node → block (the entry is node 0).
    node_block: Vec<usize>,
    /// Free scratch locals by type.
    scratch: HashMap<ValType, Vec<u32>>,
    // Stack-report facts.
    callees: Vec<String>,
    indirect: bool,
    dynamic: bool,
}

impl<'a, 'c> FnLower<'a, 'c> {
    fn run(cx: &'c mut Cx<'a>, f: &'a Function) -> R<(Body, StackUsage)> {
        let types = cx.module.types();
        let mut me = FnLower {
            cx,
            f,
            types,
            code: Code::default(),
            nparams: 0,
            locals: Vec::new(),
            loc: vec![Vec::new(); f.value_count()],
            inline: vec![false; f.value_count()],
            terms: Vec::new(),
            saved_sp: None,
            fp: None,
            frame_size: 0,
            alloca_off: HashMap::new(),
            label: None,
            node_block: Vec::new(),
            scratch: HashMap::new(),
            callees: Vec::new(),
            indirect: false,
            dynamic: false,
        };
        me.prepare()?;
        // Structurizer nodes are the blocks with the entry moved to node 0.
        let entry = f.entry().expect("defined").index();
        let mut node_block = vec![entry];
        node_block.extend((0..f.block_count()).filter(|&b| b != entry));
        let mut block_node = vec![0usize; f.block_count()];
        for (n, &b) in node_block.iter().enumerate() {
            block_node[b] = n;
        }
        let graph: Vec<CfgNode> = node_block
            .iter()
            .map(|&b| {
                let t = &me.terms[b];
                CfgNode { arms: t.edges.iter().map(|e| block_node[e.target.index()]).collect(), shape: t.shape }
            })
            .collect();
        if !structure::is_reducible(&graph) {
            me.label = Some(me.new_local(ValType::I32));
        }
        let program = structure::structurize(&graph);
        me.node_block = node_block;
        me.prologue()?;
        me.emit_seq(&program)?;
        me.code.byte(op::UNREACHABLE);
        me.code.byte(op::END);

        let name = me.cx.syms.resolve(f.name).to_owned();
        let usage = StackUsage {
            name,
            frame_size: me.frame_size,
            return_address: 0,
            saved_registers: 0,
            sp_adjust: me.frame_size,
            outgoing_args: 0,
            dynamic_alloca: me.dynamic,
            direct_callees: me.callees,
            indirect_calls: me.indirect,
            syscalls: false,
            probed: false,
        };
        Ok((Body { locals: me.locals, code: me.code }, usage))
    }

    // --- setup --------------------------------------------------------------

    fn repr(&self, v: ValueId) -> R<Repr> {
        Repr::of(self.types, self.f.value_type(v))
    }

    fn new_local(&mut self, t: ValType) -> u32 {
        self.locals.push(t);
        self.nparams + self.locals.len() as u32 - 1
    }

    fn take_scratch(&mut self, t: ValType) -> u32 {
        match self.scratch.get_mut(&t).and_then(Vec::pop) {
            Some(l) => l,
            None => self.new_local(t),
        }
    }

    fn free_scratch(&mut self, t: ValType, l: u32) {
        self.scratch.entry(t).or_default().push(l);
    }

    /// Locals, inlining decisions, the frame, and each block's terminator.
    fn prepare(&mut self) -> R<()> {
        let f = self.f;
        let entry = f.entry().expect("defined");
        // Parameters: the entry block's parameters are the wasm parameters.
        let mut n = 0u32;
        for &p in f.block(entry).params() {
            let k = self.repr(p)?.valtypes().len() as u32;
            self.loc[p.index()] = (n..n + k).collect();
            n += k;
        }
        self.nparams = n;

        // Where each instruction lives.
        let mut inst_block = vec![usize::MAX; f.inst_count()];
        for (b, block) in f.blocks() {
            for &i in block.insts() {
                inst_block[i.index()] = b.index();
            }
            if let Some(t) = block.terminator() {
                inst_block[t.index()] = b.index();
            }
        }

        for (b, block) in f.blocks() {
            if b != entry {
                for &p in block.params() {
                    let vts = self.repr(p)?.valtypes();
                    self.loc[p.index()] = vts.into_iter().map(|t| self.new_local(t)).collect();
                }
            }
            for &i in block.insts() {
                let inst = f.inst(i);
                if let InstKind::Alloca { elem_ty } = inst.kind {
                    let l = self.types.layout(elem_ty);
                    let align = l.align.max(1);
                    self.frame_size = self.frame_size.div_ceil(align) * align;
                    self.alloca_off.insert(i, self.frame_size);
                    self.frame_size += l.size.max(1);
                }
                if matches!(inst.kind, InstKind::DynAlloca { .. }) {
                    self.dynamic = true;
                }
                let Some(v) = inst.result() else { continue };
                let uses = f.uses_of(v);
                if uses.is_empty() {
                    continue;
                }
                let repr = self.repr(v)?;
                let pure = matches!(
                    inst.kind,
                    InstKind::Bin(op) if !matches!(op, BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem | BinOp::FRem)
                ) || matches!(
                    inst.kind,
                    InstKind::Unary(_)
                        | InstKind::ICmp(_)
                        | InstKind::FCmp(_)
                        | InstKind::Cast(_)
                        | InstKind::PtrAdd { .. }
                        | InstKind::Select
                        | InstKind::Freeze
                        | InstKind::Declassify
                        | InstKind::Alloca { .. }
                );
                if pure && !matches!(repr, Repr::Wide(_)) && uses.len() == 1 && inst_block[uses[0].inst.index()] == b.index() {
                    self.inline[v.index()] = true;
                } else {
                    self.loc[v.index()] = repr.valtypes().into_iter().map(|t| self.new_local(t)).collect();
                }
            }
        }
        self.frame_size = self.frame_size.div_ceil(16) * 16;

        for (_, block) in f.blocks() {
            let t = block.terminator().expect("verified: every block is terminated");
            let term = self.term(f.inst(t))?;
            self.terms.push(term);
        }
        Ok(())
    }

    fn term(&self, inst: &InstData) -> R<Term> {
        let ops = inst.operands();
        let direct = |exit, edges| Term { exit, edges, shape: Shape::Direct };
        Ok(match &inst.kind {
            InstKind::Ret => direct(Exit::Ret(ops.first().copied()), Vec::new()),
            InstKind::Unreachable => direct(Exit::Unreachable, Vec::new()),
            InstKind::Br(t) => direct(Exit::Branch(None), vec![Edge { target: *t, args: ops.to_vec() }]),
            &InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                let ta = true_args as usize;
                let fa = false_args as usize;
                let t = Edge { target: if_true, args: ops[1..1 + ta].to_vec() };
                let e = Edge { target: if_false, args: ops[1 + ta..1 + ta + fa].to_vec() };
                if t == e {
                    direct(Exit::Branch(None), vec![t])
                } else {
                    Term { exit: Exit::Branch(Some(ops[0])), edges: vec![t, e], shape: Shape::IfElse }
                }
            }
            InstKind::Switch(data) => {
                let cond = ops[0];
                let mut at = 1usize;
                let mut edges: Vec<Edge> = Vec::new();
                let arm_of = |e: Edge, edges: &mut Vec<Edge>| match edges.iter().position(|x| *x == e) {
                    Some(i) => i,
                    None => {
                        edges.push(e);
                        edges.len() - 1
                    }
                };
                let da = data.default_args as usize;
                let default = arm_of(Edge { target: data.default, args: ops[at..at + da].to_vec() }, &mut edges);
                at += da;
                let bits = match self.repr(cond)? {
                    Repr::I32(b) | Repr::I64(b) => b,
                    _ => return fail("a switch on an integer wider than 64 bits"),
                };
                let mut cases = Vec::new();
                for c in &data.cases {
                    let n = c.args as usize;
                    let arm = arm_of(Edge { target: c.target, args: ops[at..at + n].to_vec() }, &mut edges);
                    at += n;
                    // The first matching case wins; later duplicates are dead.
                    let v = c.value.mod_2k(bits);
                    if !cases.iter().any(|(x, _): &(Int, usize)| *x == v) {
                        cases.push((v, arm));
                    }
                }
                if edges.len() == 1 {
                    direct(Exit::Branch(None), edges)
                } else {
                    Term { exit: Exit::Switch { cond, default, cases }, edges, shape: Shape::Table }
                }
            }
            other => return fail(format!("a block ends in a non-terminator {other:?}")),
        })
    }

    /// Mask the narrow parameters of a function others may call (see the
    /// [module docs](self)) and set up the frame.
    fn prologue(&mut self) -> R<()> {
        let entry = self.f.entry().expect("defined");
        let outside_callers = self.f.attrs.linkage != Linkage::Internal;
        for &p in self.f.block(entry).params() {
            let r = self.repr(p)?;
            if needs_mask(r) && outside_callers {
                let l = self.loc[p.index()][0];
                self.local_get(l);
                self.norm(r);
                self.local_set(l);
            }
        }
        if self.frame_size > 0 || self.dynamic {
            let saved = self.new_local(ValType::I32);
            self.saved_sp = Some(saved);
            self.global_get_sp();
            self.local_set(saved);
            if self.frame_size > 0 {
                let fp = self.new_local(ValType::I32);
                self.fp = Some(fp);
                self.local_get(saved);
                self.i32_const(self.frame_size as i32);
                self.code.byte(op::I32_SUB);
                self.local_tee(fp);
                self.global_set_sp();
            }
        }
        if let Some(l) = self.label {
            self.i32_const(0);
            self.local_set(l);
        }
        Ok(())
    }

    // --- raw emission -------------------------------------------------------

    fn local_get(&mut self, l: u32) {
        self.code.byte(op::LOCAL_GET);
        self.code.u32(l);
    }

    fn local_set(&mut self, l: u32) {
        self.code.byte(op::LOCAL_SET);
        self.code.u32(l);
    }

    fn local_tee(&mut self, l: u32) {
        self.code.byte(op::LOCAL_TEE);
        self.code.u32(l);
    }

    fn i32_const(&mut self, v: i32) {
        self.code.byte(op::I32_CONST);
        self.code.i32(v);
    }

    fn i64_const(&mut self, v: i64) {
        self.code.byte(op::I64_CONST);
        self.code.i64(v);
    }

    /// An integer constant in the container of `r`.
    fn int_const(&mut self, r: Repr, v: u64) {
        if r.is64() { self.i64_const(v as i64) } else { self.i32_const(v as u32 as i32) }
    }

    fn global_get_sp(&mut self) {
        self.cx.obj.uses_sp = true;
        self.code.byte(op::GLOBAL_GET);
        self.code.hole(Target::StackPointer);
    }

    fn global_set_sp(&mut self) {
        self.cx.obj.uses_sp = true;
        self.code.byte(op::GLOBAL_SET);
        self.code.hole(Target::StackPointer);
    }

    /// A memory instruction with its `memarg`: alignment `align` bytes (capped
    /// at the access size `size`, as validation requires) and a constant offset.
    fn mem(&mut self, opcode: u8, align: u64, size: u64, offset: u32) {
        self.code.byte(opcode);
        self.code.u32(align.clamp(1, size).trailing_zeros());
        self.code.u32(offset);
    }

    /// An atomic (`0xFE`-prefixed) instruction with its naturally aligned
    /// `memarg`.
    fn atomic(&mut self, sub: u32, size: u64) {
        self.code.byte(op::PREFIX_FE);
        self.code.u32(sub);
        self.code.u32(size.trailing_zeros());
        self.code.u32(0);
    }

    /// Clear the bits of an integer above its width (the zero-extension
    /// invariant).
    fn norm(&mut self, r: Repr) {
        match r {
            Repr::I32(b) if b < 32 => {
                self.i32_const(((1u64 << b) - 1) as u32 as i32);
                self.code.byte(op::I32_AND);
            }
            Repr::I64(b) if b < 64 => {
                self.i64_const(((1u64 << b) - 1) as i64);
                self.code.byte(op::I64_AND);
            }
            _ => {}
        }
    }

    /// Sign-extend the integer on the stack from its width to its container.
    fn sext(&mut self, r: Repr) {
        match r {
            Repr::I32(8) => self.code.byte(op::I32_EXTEND8_S),
            Repr::I32(16) => self.code.byte(op::I32_EXTEND16_S),
            Repr::I32(b) if b < 32 => {
                self.i32_const(32 - b as i32);
                self.code.byte(op::I32_SHL);
                self.i32_const(32 - b as i32);
                self.code.byte(op::I32_SHR_S);
            }
            Repr::I64(8) => self.code.byte(op::I64_EXTEND8_S),
            Repr::I64(16) => self.code.byte(op::I64_EXTEND16_S),
            Repr::I64(32) => self.code.byte(op::I64_EXTEND32_S),
            Repr::I64(b) if b < 64 => {
                self.i64_const(64 - i64::from(b));
                self.code.byte(op::I64_SHL);
                self.i64_const(64 - i64::from(b));
                self.code.byte(op::I64_SHR_S);
            }
            _ => {}
        }
    }

    // --- values -------------------------------------------------------------

    /// Push value `v` (a single-local representation).
    fn value(&mut self, v: ValueId) -> R<()> {
        if self.inline[v.index()] {
            let ValueDef::Inst(i) = self.f.value(v).def else { unreachable!("only results are inlined") };
            return self.expr(i);
        }
        match &self.f.value(v).def {
            ValueDef::Inst(_) | ValueDef::Param(..) => {
                let ls = &self.loc[v.index()];
                if ls.len() != 1 {
                    return fail("a wide value used where one local is needed");
                }
                self.local_get(ls[0]);
            }
            ValueDef::Const(c) => {
                let r = self.repr(v)?;
                match self.cx.module.consts().get(*c).clone() {
                    Const::Int { value, .. } => match r {
                        Repr::I32(b) | Repr::I64(b) => self.int_const(r, value.mod_2k(b).to_u64().unwrap_or(0)),
                        _ => return fail("a wide constant used where one local is needed"),
                    },
                    Const::Float { bits: FloatBits::F32(b), .. } => {
                        self.code.byte(op::F32_CONST);
                        self.code.bytes.extend_from_slice(&b.to_le_bytes());
                    }
                    Const::Float { bits: FloatBits::F64(b), .. } => {
                        self.code.byte(op::F64_CONST);
                        self.code.bytes.extend_from_slice(&b.to_le_bytes());
                    }
                    Const::Float { bits: FloatBits::F16(_), .. } => return fail("f16 has no WebAssembly lowering"),
                    // Poison may be any value; zero is as good as any.
                    Const::Null(_) | Const::Poison(_) => self.zero(r),
                    Const::Aggregate { .. } | Const::Addr { .. } => {
                        return fail("an aggregate or address constant used as an operand");
                    }
                }
            }
            ValueDef::Global(g) => {
                let sym = self.cx.global_sym[g.index()];
                self.code.byte(op::I32_CONST);
                self.code.hole(Target::Mem { sym, addend: 0 });
            }
            ValueDef::Func(fid) => {
                let idx = self.cx.func_index[fid.index()].expect("referenced");
                self.cx.obj.table_slot(idx);
                self.code.byte(op::I32_CONST);
                self.code.hole(Target::Table(idx));
            }
        }
        Ok(())
    }

    /// Push a zero of representation `r` (every part of a wide one).
    fn zero(&mut self, r: Repr) {
        match r {
            Repr::Void => {}
            Repr::I32(_) => self.i32_const(0),
            Repr::I64(_) => self.i64_const(0),
            Repr::F32 => {
                self.code.byte(op::F32_CONST);
                self.code.bytes.extend_from_slice(&0u32.to_le_bytes());
            }
            Repr::F64 => {
                self.code.byte(op::F64_CONST);
                self.code.bytes.extend_from_slice(&0u64.to_le_bytes());
            }
            Repr::Wide(n) => {
                for _ in 0..n {
                    self.i64_const(0);
                }
            }
        }
    }

    /// Push part `k` of a value (`i64`), whatever its representation: part 0
    /// of a narrow integer is the value zero-extended to 64 bits, higher parts
    /// are zero.
    fn part(&mut self, v: ValueId, k: u32) -> R<()> {
        let r = self.repr(v)?;
        match r {
            Repr::Wide(n) => {
                if k >= n {
                    self.i64_const(0);
                    return Ok(());
                }
                match &self.f.value(v).def {
                    ValueDef::Const(c) => {
                        let bits = match self.cx.module.consts().get(*c) {
                            Const::Int { value, .. } => value.mod_2k(64 * n).div_2k_trunc(64 * k).mod_2k(64).to_u64().unwrap_or(0),
                            _ => 0,
                        };
                        self.i64_const(bits as i64);
                    }
                    _ => {
                        let l = self.loc[v.index()][k as usize];
                        self.local_get(l);
                    }
                }
            }
            Repr::I32(_) => {
                if k == 0 {
                    self.value(v)?;
                    self.code.byte(op::I64_EXTEND_I32_U);
                } else {
                    self.i64_const(0);
                }
            }
            Repr::I64(_) => {
                if k == 0 { self.value(v)? } else { self.i64_const(0) }
            }
            _ => return fail("a float used as an integer part"),
        }
        Ok(())
    }

    /// Push every wasm value of `v` (all parts of a wide value).
    fn values(&mut self, v: ValueId) -> R<()> {
        match self.repr(v)? {
            Repr::Wide(n) => {
                for k in 0..n {
                    self.part(v, k)?;
                }
                Ok(())
            }
            _ => self.value(v),
        }
    }

    /// Push `v` as an `i32` address or offset (an `i64` is wrapped, a wide
    /// value contributes its low part; a narrow offset is sign-extended).
    fn value_i32(&mut self, v: ValueId, signed: bool) -> R<()> {
        let r = self.repr(v)?;
        if let ValueDef::Const(c) = self.f.value(v).def
            && let Const::Int { value, .. } = self.cx.module.consts().get(c)
            && let Repr::I32(b) | Repr::I64(b) = r
        {
            let raw = value.mod_2k(b).to_u64().unwrap_or(0);
            let v = if signed && b < 64 && raw >> (b - 1) & 1 == 1 { raw | !((1u64 << b) - 1) } else { raw };
            self.i32_const(v as u32 as i32);
            return Ok(());
        }
        match r {
            Repr::I32(b) => {
                self.value(v)?;
                if signed && b < 32 {
                    self.sext(r);
                }
            }
            Repr::I64(_) => {
                self.value(v)?;
                self.code.byte(op::I32_WRAP_I64);
            }
            Repr::Wide(_) => {
                self.part(v, 0)?;
                self.code.byte(op::I32_WRAP_I64);
            }
            _ => return fail("a float used as an address"),
        }
        Ok(())
    }

    /// Store the wasm values on the stack into `v`'s locals.
    fn set(&mut self, v: ValueId) {
        let ls = self.loc[v.index()].clone();
        for &l in ls.iter().rev() {
            self.local_set(l);
        }
    }

    // --- structured control flow --------------------------------------------

    fn emit_seq(&mut self, items: &[Structured]) -> R<()> {
        for item in items {
            match item {
                Structured::Block(body) => {
                    self.code.byte(op::BLOCK);
                    self.code.byte(op::EMPTY);
                    self.emit_seq(body)?;
                    self.code.byte(op::END);
                }
                Structured::Loop(body) => {
                    self.code.byte(op::LOOP);
                    self.code.byte(op::EMPTY);
                    self.emit_seq(body)?;
                    self.code.byte(op::END);
                }
                Structured::Node { node, arms } => self.emit_node(*node, arms)?,
                Structured::Dispatch { order, arms } => {
                    let label = self.label.expect("dispatch label");
                    let n = order.len();
                    self.code.byte(op::LOOP);
                    self.code.byte(op::EMPTY);
                    for _ in 0..n {
                        self.code.byte(op::BLOCK);
                        self.code.byte(op::EMPTY);
                    }
                    self.local_get(label);
                    self.code.byte(op::BR_TABLE);
                    self.code.u32(n as u32 - 1);
                    for i in 0..n as u32 {
                        self.code.u32(i);
                    }
                    for (&node, node_arms) in order.iter().zip(arms) {
                        self.code.byte(op::END);
                        self.emit_node(node, node_arms)?;
                    }
                    self.code.byte(op::END);
                }
            }
        }
        Ok(())
    }

    fn emit_node(&mut self, node: usize, arms: &[Arm]) -> R<()> {
        let b = self.node_block[node];
        let f = self.f;
        let block = f.block(BlockId::from_index(b));
        for &i in block.insts() {
            self.stmt(i)?;
        }
        let term = &self.terms[b];
        let edges = term.edges.clone();
        match &term.exit {
            Exit::Ret(v) => {
                let v = *v;
                if let Some(v) = v {
                    self.values(v)?;
                }
                if let Some(s) = self.saved_sp {
                    self.local_get(s);
                    self.global_set_sp();
                }
                self.code.byte(op::RETURN);
            }
            Exit::Unreachable => self.code.byte(op::UNREACHABLE),
            Exit::Branch(None) => self.emit_arm(&edges[0], &arms[0])?,
            Exit::Branch(Some(cond)) => {
                let cond = *cond;
                self.value(cond)?;
                self.code.byte(op::IF);
                self.code.byte(op::EMPTY);
                self.emit_arm(&edges[0], &arms[0])?;
                self.code.byte(op::ELSE);
                self.emit_arm(&edges[1], &arms[1])?;
                self.code.byte(op::END);
            }
            Exit::Switch { cond, default, cases } => {
                let (cond, default, cases) = (*cond, *default, cases.clone());
                let n = edges.len();
                for _ in 0..n {
                    self.code.byte(op::BLOCK);
                    self.code.byte(op::EMPTY);
                }
                self.switch_dispatch(cond, default, &cases)?;
                for (e, a) in edges.iter().zip(arms) {
                    self.code.byte(op::END);
                    self.emit_arm(e, a)?;
                }
            }
        }
        Ok(())
    }

    /// The dispatch of a switch inside the innermost of its arm blocks: `br i`
    /// reaches arm `i`. Dense case values use a `br_table`, others a chain of
    /// compares.
    fn switch_dispatch(&mut self, cond: ValueId, default: usize, cases: &[(Int, usize)]) -> R<()> {
        let r = self.repr(cond)?;
        let vals: Vec<u64> = cases.iter().map(|(v, _)| v.to_u64().unwrap_or(0)).collect();
        let (lo, hi) = (vals.iter().copied().min().unwrap_or(0), vals.iter().copied().max().unwrap_or(0));
        let span = hi - lo;
        if !cases.is_empty() && span < 1024 && span <= 3 * cases.len() as u64 + 8 {
            let mut table = vec![default as u32; span as usize + 1];
            for (&v, (_, arm)) in vals.iter().zip(cases) {
                table[(v - lo) as usize] = *arm as u32;
            }
            // index = cond - lo; anything outside [0, span] takes the default.
            if r.is64() {
                let t = self.take_scratch(ValType::I64);
                self.value(cond)?;
                self.i64_const(lo as i64);
                self.code.byte(op::I64_SUB);
                self.local_tee(t);
                self.i64_const(span as i64 + 1);
                self.code.byte(op::I64_LT_U);
                self.code.byte(op::IF);
                self.code.byte(op::EMPTY);
                self.local_get(t);
                self.code.byte(op::I32_WRAP_I64);
                self.code.byte(op::BR_TABLE);
                self.code.u32(table.len() as u32);
                for &a in &table {
                    self.code.u32(a + 1);
                }
                self.code.u32(default as u32 + 1);
                self.code.byte(op::END);
                self.free_scratch(ValType::I64, t);
                self.code.byte(op::BR);
                self.code.u32(default as u32);
            } else {
                self.value(cond)?;
                if lo != 0 {
                    self.i32_const(lo as u32 as i32);
                    self.code.byte(op::I32_SUB);
                }
                self.code.byte(op::BR_TABLE);
                self.code.u32(table.len() as u32);
                for &a in &table {
                    self.code.u32(a);
                }
                self.code.u32(default as u32);
            }
        } else {
            let (eq, _) = if r.is64() { (0x51, 0) } else { (op::I32_EQ, 0) };
            for (&v, (_, arm)) in vals.iter().zip(cases) {
                self.value(cond)?;
                self.int_const(r, v);
                self.code.byte(eq);
                self.code.byte(op::BR_IF);
                self.code.u32(*arm as u32);
            }
            self.code.byte(op::BR);
            self.code.u32(default as u32);
        }
        Ok(())
    }

    /// Assign an edge's arguments to its target's parameters (all read before
    /// any is written: a parallel copy), then take the arm.
    fn emit_arm(&mut self, edge: &Edge, arm: &Arm) -> R<()> {
        let params = self.f.block(edge.target).params().to_vec();
        let moves: Vec<(ValueId, ValueId)> =
            edge.args.iter().zip(&params).filter(|(a, p)| a != p).map(|(&a, &p)| (a, p)).collect();
        for &(a, _) in &moves {
            self.values(a)?;
        }
        for &(_, p) in moves.iter().rev() {
            self.set(p);
        }
        match arm {
            Arm::Br(d) => {
                self.code.byte(op::BR);
                self.code.u32(*d);
            }
            Arm::Inline(body) => self.emit_seq(body)?,
            // The target's code is the next item of the sequence.
            Arm::Next => {}
            Arm::Goto { target, depth } => {
                self.i32_const(*target as i32);
                self.local_set(self.label.expect("dispatch label"));
                self.code.byte(op::BR);
                self.code.u32(*depth);
            }
        }
        Ok(())
    }

    // --- instructions -------------------------------------------------------

    /// Emit a non-terminator instruction as a statement: compute it into its
    /// local(s), or for effect, or not at all (inlined or unused and pure).
    fn stmt(&mut self, i: InstId) -> R<()> {
        let inst = self.f.inst(i);
        if let Some(v) = inst.result() {
            if self.inline[v.index()] {
                return Ok(());
            }
            if !self.loc[v.index()].is_empty() {
                self.expr(i)?;
                self.set(v);
                return Ok(());
            }
            let effect = inst.kind.has_side_effect() && !matches!(inst.kind, InstKind::Alloca { .. });
            if !effect {
                return Ok(());
            }
            self.expr(i)?;
            for _ in 0..self.repr(v)?.valtypes().len() {
                self.code.byte(op::DROP);
            }
            return Ok(());
        }
        self.expr(i)
    }

    /// Push the result of instruction `i` (nothing for a void one).
    fn expr(&mut self, i: InstId) -> R<()> {
        let inst = self.f.inst(i);
        let ops = inst.operands().to_vec();
        let rty = Repr::of(self.types, inst.ty)?;
        match &inst.kind {
            &InstKind::Bin(op) => self.bin(op, rty, ops[0], ops[1])?,
            InstKind::Unary(UnaryOp::FNeg) => {
                self.value(ops[0])?;
                self.code.byte(if rty == Repr::F64 { op::F64_NEG } else { op::F32_NEG });
            }
            &InstKind::ICmp(pred) => {
                let r = self.repr(ops[0])?;
                if let Repr::Wide(_) = r {
                    return fail("a comparison of integers wider than 64 bits (legalize first)");
                }
                let signed = is_signed(pred);
                for &o in &ops[..2] {
                    self.value(o)?;
                    if signed {
                        self.sext(r);
                    }
                }
                let (o32, o64) = icmp_op(pred);
                self.code.byte(if r.is64() { o64 } else { o32 });
            }
            &InstKind::FCmp(pred) => self.fcmp(pred, ops[0], ops[1])?,
            &InstKind::Cast(op) => self.cast(op, rty, ops[0])?,
            InstKind::Alloca { .. } => {
                let fp = self.fp.expect("frame");
                self.local_get(fp);
                let off = self.alloca_off[&i];
                if off != 0 {
                    self.i32_const(off as i32);
                    self.code.byte(op::I32_ADD);
                }
            }
            &InstKind::DynAlloca { align } => {
                let t = self.take_scratch(ValType::I32);
                self.global_get_sp();
                self.value_i32(ops[0], false)?;
                self.code.byte(op::I32_SUB);
                self.i32_const(-(i64::from(align.max(16)) as i32));
                self.code.byte(op::I32_AND);
                self.local_tee(t);
                self.global_set_sp();
                self.local_get(t);
                self.free_scratch(ValType::I32, t);
            }
            &InstKind::Load { ty, align, .. } => {
                let r = Repr::of(self.types, ty)?;
                if matches!(self.types.get(ty), Type::Struct(_) | Type::Array(..)) {
                    return fail("a load of an aggregate type");
                }
                self.load(r, ops[0], u64::from(align))?;
            }
            &InstKind::Store { ty, align, .. } => {
                let r = Repr::of(self.types, ty)?;
                if matches!(self.types.get(ty), Type::Struct(_) | Type::Array(..)) {
                    return fail("a store of an aggregate type");
                }
                self.store(r, ops[0], ops[1], u64::from(align))?;
            }
            &InstKind::AtomicLoad { ty, .. } => {
                let (r, size) = self.atomic_repr(ty)?;
                self.value(ops[0])?;
                self.atomic(0x10 + atomic_slot(r, size), size);
            }
            &InstKind::AtomicStore { ty, .. } => {
                let (r, size) = self.atomic_repr(ty)?;
                self.value(ops[0])?;
                self.value(ops[1])?;
                self.atomic(0x17 + atomic_slot(r, size), size);
            }
            &InstKind::AtomicRmw { op, ty, .. } => self.atomic_rmw(op, ty, ops[0], ops[1])?,
            &InstKind::CmpXchg { ty, .. } => {
                let (r, size) = self.atomic_repr(ty)?;
                self.value(ops[0])?;
                self.value(ops[1])?;
                self.value(ops[2])?;
                self.atomic(0x48 + atomic_slot(r, size), size);
            }
            InstKind::Fence(_) => {
                // Every ordering maps to the one sequentially consistent fence.
                self.code.byte(op::PREFIX_FE);
                self.code.u32(0x03);
                self.code.byte(0x00);
            }
            InstKind::PtrAdd { .. } => {
                self.value(ops[0])?;
                if !self.is_zero_const(ops[1]) {
                    self.value_i32(ops[1], true)?;
                    self.code.byte(op::I32_ADD);
                }
            }
            InstKind::Select => {
                if let Repr::Wide(n) = rty {
                    for k in 0..n {
                        self.part(ops[1], k)?;
                        self.part(ops[2], k)?;
                        self.value(ops[0])?;
                        self.code.byte(op::SELECT);
                    }
                } else {
                    self.value(ops[1])?;
                    self.value(ops[2])?;
                    self.value(ops[0])?;
                    self.code.byte(op::SELECT);
                }
            }
            // `declassify` only changes what the analyses may assume.
            InstKind::Freeze | InstKind::Declassify => self.values(ops[0])?,
            InstKind::Call => self.call(i, rty)?,
            InstKind::Syscall => return fail("the syscall op has no WebAssembly lowering (import a host function instead)"),
            other => return fail(format!("unexpected instruction {other:?}")),
        }
        Ok(())
    }

    fn is_zero_const(&self, v: ValueId) -> bool {
        matches!(self.f.value(v).def, ValueDef::Const(c)
            if matches!(self.cx.module.consts().get(c), Const::Int { value, .. } if value.is_zero()))
    }

    fn bin(&mut self, op: BinOp, r: Repr, a: ValueId, b: ValueId) -> R<()> {
        match r {
            Repr::F32 | Repr::F64 => {
                let is64 = r == Repr::F64;
                self.value(a)?;
                self.value(b)?;
                let code = match op {
                    BinOp::FAdd => [0x92, 0xa0],
                    BinOp::FSub => [0x93, 0xa1],
                    BinOp::FMul => [0x94, 0xa2],
                    BinOp::FDiv => [0x95, 0xa3],
                    BinOp::FRem => {
                        let f = self.cx.fmod[usize::from(!is64)].expect("fmod import");
                        self.note_callee(f);
                        self.code.byte(op::CALL);
                        self.code.hole(Target::Func(f));
                        return Ok(());
                    }
                    _ => return fail(format!("{op:?} on a float")),
                };
                self.code.byte(code[usize::from(is64)]);
            }
            Repr::I32(_) | Repr::I64(_) => {
                let (o32, o64) = int_bin_op(op);
                let signed = matches!(op, BinOp::SDiv | BinOp::SRem);
                self.value(a)?;
                if signed || op == BinOp::AShr {
                    self.sext(r);
                }
                self.value(b)?;
                if signed {
                    self.sext(r);
                }
                self.code.byte(if r.is64() { o64 } else { o32 });
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl | BinOp::SDiv | BinOp::SRem | BinOp::AShr) {
                    self.norm(r);
                }
            }
            Repr::Wide(n) => self.wide_bin(op, n, a, b)?,
            Repr::Void => return fail("a binary operation without a value"),
        }
        Ok(())
    }

    /// The wide operations the legalizer leaves at the ABI seam: bitwise ops
    /// part by part, and shifts by whole parts (the split/join shapes).
    fn wide_bin(&mut self, op: BinOp, n: u32, a: ValueId, b: ValueId) -> R<()> {
        match op {
            BinOp::And | BinOp::Or | BinOp::Xor => {
                let code = match op {
                    BinOp::And => op::I64_AND,
                    BinOp::Or => op::I64_OR,
                    _ => op::I64_XOR,
                };
                for k in 0..n {
                    self.part(a, k)?;
                    self.part(b, k)?;
                    self.code.byte(code);
                }
            }
            BinOp::Shl | BinOp::LShr | BinOp::AShr => {
                let amount = match self.f.value(b).def {
                    ValueDef::Const(c) => match self.cx.module.consts().get(c) {
                        Const::Int { value, .. } => value.to_u64(),
                        _ => None,
                    },
                    _ => None,
                };
                let Some(s) = amount.filter(|s| s % 64 == 0) else {
                    return fail("a shift of an integer wider than 64 bits by a non-constant or partial amount (legalize first)");
                };
                let k = (s / 64).min(u64::from(n)) as u32;
                match op {
                    BinOp::Shl => {
                        for j in 0..n {
                            if j < k { self.i64_const(0) } else { self.part(a, j - k)? }
                        }
                    }
                    BinOp::LShr => {
                        for j in 0..n {
                            if j + k < n { self.part(a, j + k)? } else { self.i64_const(0) }
                        }
                    }
                    _ => {
                        for j in 0..n {
                            if j + k < n {
                                self.part(a, j + k)?;
                            } else {
                                self.part(a, n - 1)?;
                                self.i64_const(63);
                                self.code.byte(op::I64_SHR_S);
                            }
                        }
                    }
                }
            }
            _ => return fail(format!("{op:?} on an integer wider than 64 bits (legalize first)")),
        }
        Ok(())
    }

    fn fcmp(&mut self, pred: FloatPred, a: ValueId, b: ValueId) -> R<()> {
        let is64 = self.repr(a)? == Repr::F64;
        // f32/f64 eq, ne, lt, gt, le, ge.
        let base: u8 = if is64 { 0x61 } else { 0x5b };
        let (eq, ne, lt, gt, le, ge) = (base, base + 1, base + 2, base + 3, base + 4, base + 5);
        let cmp = |me: &mut Self, x: ValueId, y: ValueId, code: u8| -> R<()> {
            me.value(x)?;
            me.value(y)?;
            me.code.byte(code);
            Ok(())
        };
        match pred {
            FloatPred::False => self.i32_const(0),
            FloatPred::True => self.i32_const(1),
            FloatPred::Oeq => cmp(self, a, b, eq)?,
            FloatPred::Ogt => cmp(self, a, b, gt)?,
            FloatPred::Oge => cmp(self, a, b, ge)?,
            FloatPred::Olt => cmp(self, a, b, lt)?,
            FloatPred::Ole => cmp(self, a, b, le)?,
            FloatPred::Une => cmp(self, a, b, ne)?,
            // Unordered-or-X is the negation of the opposite ordered compare.
            FloatPred::Ugt => {
                cmp(self, a, b, le)?;
                self.code.byte(op::I32_EQZ);
            }
            FloatPred::Uge => {
                cmp(self, a, b, lt)?;
                self.code.byte(op::I32_EQZ);
            }
            FloatPred::Ult => {
                cmp(self, a, b, ge)?;
                self.code.byte(op::I32_EQZ);
            }
            FloatPred::Ule => {
                cmp(self, a, b, gt)?;
                self.code.byte(op::I32_EQZ);
            }
            FloatPred::One | FloatPred::Ueq => {
                cmp(self, a, b, lt)?;
                cmp(self, a, b, gt)?;
                self.code.byte(op::I32_OR);
                if pred == FloatPred::Ueq {
                    self.code.byte(op::I32_EQZ);
                }
            }
            FloatPred::Ord => {
                cmp(self, a, a, eq)?;
                cmp(self, b, b, eq)?;
                self.code.byte(op::I32_AND);
            }
            FloatPred::Uno => {
                cmp(self, a, a, ne)?;
                cmp(self, b, b, ne)?;
                self.code.byte(op::I32_OR);
            }
        }
        Ok(())
    }

    fn cast(&mut self, cop: CastOp, to: Repr, v: ValueId) -> R<()> {
        let from = self.repr(v)?;
        match (cop, from, to) {
            // Integer resizing.
            (CastOp::Trunc | CastOp::ZExt | CastOp::SExt, Repr::I32(_) | Repr::I64(_), Repr::I32(_) | Repr::I64(_)) => {
                self.value(v)?;
                if cop == CastOp::SExt {
                    self.sext(from);
                }
                match (from.is64(), to.is64()) {
                    (true, false) => self.code.byte(op::I32_WRAP_I64),
                    (false, true) => {
                        self.code.byte(if cop == CastOp::SExt { op::I64_EXTEND_I32_S } else { op::I64_EXTEND_I32_U })
                    }
                    _ => {}
                }
                if cop != CastOp::ZExt {
                    self.norm(to);
                }
            }
            (CastOp::Trunc, Repr::Wide(_), Repr::I32(_) | Repr::I64(_)) => {
                self.part(v, 0)?;
                if !to.is64() {
                    self.code.byte(op::I32_WRAP_I64);
                }
                self.norm(to);
            }
            (CastOp::Trunc, Repr::Wide(_), Repr::Wide(n)) | (CastOp::ZExt, _, Repr::Wide(n)) => {
                for k in 0..n {
                    self.part(v, k)?;
                }
            }
            (CastOp::SExt, Repr::I32(_) | Repr::I64(_), Repr::Wide(n)) => {
                let t = self.take_scratch(ValType::I64);
                self.value(v)?;
                self.sext(from);
                if !from.is64() {
                    self.code.byte(op::I64_EXTEND_I32_S);
                }
                self.local_tee(t);
                for _ in 1..n {
                    self.local_get(t);
                    self.i64_const(63);
                    self.code.byte(op::I64_SHR_S);
                }
                self.free_scratch(ValType::I64, t);
            }
            (CastOp::SExt, Repr::Wide(m), Repr::Wide(n)) => {
                for k in 0..n {
                    if k < m {
                        self.part(v, k)?;
                    } else {
                        self.part(v, m - 1)?;
                        self.i64_const(63);
                        self.code.byte(op::I64_SHR_S);
                    }
                }
            }
            // Floats.
            (CastOp::FpTrunc | CastOp::FpExt, _, _) => {
                self.value(v)?;
                match (from, to) {
                    (Repr::F64, Repr::F32) => self.code.byte(op::F32_DEMOTE_F64),
                    (Repr::F32, Repr::F64) => self.code.byte(op::F64_PROMOTE_F32),
                    _ => {}
                }
            }
            (CastOp::FpToUi | CastOp::FpToSi, Repr::F32 | Repr::F64, Repr::I32(_) | Repr::I64(_)) => {
                // The saturating forms never trap; out of range is poison in
                // the IR, so any result will do.
                self.value(v)?;
                let sub = u32::from(to.is64()) * 4 + u32::from(from == Repr::F64) * 2 + u32::from(cop == CastOp::FpToUi);
                self.code.byte(op::PREFIX_FC);
                self.code.u32(sub);
                self.norm(to);
            }
            (CastOp::UiToFp | CastOp::SiToFp, Repr::I32(_) | Repr::I64(_), Repr::F32 | Repr::F64) => {
                self.value(v)?;
                let signed = cop == CastOp::SiToFp;
                if signed {
                    self.sext(from);
                }
                // f32.convert_i32_s 0xB2 .. f64.convert_i64_u 0xBA (skipping 0xB6).
                let base = if to == Repr::F64 { 0xb7 } else { 0xb2 };
                self.code.byte(base + u8::from(from.is64()) * 2 + u8::from(!signed));
            }
            // Pointers.
            (CastOp::PtrToInt, _, Repr::I32(_) | Repr::I64(_)) => {
                self.value(v)?;
                if to.is64() {
                    self.code.byte(op::I64_EXTEND_I32_U);
                }
                self.norm(to);
            }
            (CastOp::PtrToInt, _, Repr::Wide(n)) => {
                for k in 0..n {
                    self.part(v, k)?;
                }
            }
            (CastOp::IntToPtr, _, _) => self.value_i32(v, false)?,
            // Reinterpretation.
            (CastOp::Bitcast, _, _) => {
                self.values(v)?;
                match (from, to) {
                    (Repr::F32, Repr::I32(32)) => self.code.byte(op::I32_REINTERPRET_F32),
                    (Repr::I32(32), Repr::F32) => self.code.byte(op::F32_REINTERPRET_I32),
                    (Repr::F64, Repr::I64(64)) => self.code.byte(op::I64_REINTERPRET_F64),
                    (Repr::I64(64), Repr::F64) => self.code.byte(op::F64_REINTERPRET_I64),
                    (a, b) if a == b || (matches!(a, Repr::I32(32)) && matches!(b, Repr::I32(32))) => {}
                    _ => return fail(format!("a bitcast from {from:?} to {to:?}")),
                }
            }
            _ => return fail(format!("{cop:?} from {from:?} to {to:?}")),
        }
        Ok(())
    }

    fn load(&mut self, r: Repr, addr: ValueId, align: u64) -> R<()> {
        match r {
            Repr::F32 => {
                self.value(addr)?;
                self.mem(op::F32_LOAD, align, 4, 0);
            }
            Repr::F64 => {
                self.value(addr)?;
                self.mem(op::F64_LOAD, align, 8, 0);
            }
            Repr::I32(b) | Repr::I64(b) => {
                let size = u64::from(b.div_ceil(8));
                let is64 = r.is64();
                match (size, is64) {
                    (1, false) => {
                        self.value(addr)?;
                        self.mem(op::I32_LOAD8_U, align, 1, 0);
                    }
                    (2, false) => {
                        self.value(addr)?;
                        self.mem(op::I32_LOAD16_U, align, 2, 0);
                    }
                    (4, false) => {
                        self.value(addr)?;
                        self.mem(op::I32_LOAD, align, 4, 0);
                    }
                    (8, true) => {
                        self.value(addr)?;
                        self.mem(op::I64_LOAD, align, 8, 0);
                    }
                    _ => {
                        // 3, 5, 6 or 7 bytes: little-endian pieces.
                        let a = self.take_scratch(ValType::I32);
                        self.value(addr)?;
                        self.local_set(a);
                        let mut first = true;
                        for (off, len) in pieces(size) {
                            self.local_get(a);
                            let opc = match (len, is64) {
                                (4, true) => op::I64_LOAD32_U,
                                (2, true) => op::I64_LOAD16_U,
                                (1, true) => op::I64_LOAD8_U,
                                (2, false) => op::I32_LOAD16_U,
                                _ => op::I32_LOAD8_U,
                            };
                            self.mem(opc, 1, len, off as u32);
                            if off != 0 {
                                self.int_const(r, off * 8);
                                self.code.byte(if is64 { op::I64_SHL } else { op::I32_SHL });
                            }
                            if !first {
                                self.code.byte(if is64 { op::I64_OR } else { op::I32_OR });
                            }
                            first = false;
                        }
                        self.free_scratch(ValType::I32, a);
                    }
                }
                // Whole-byte loads are zero-extended already; `i1`, `i13`...
                // keep only their own bits.
                if !b.is_multiple_of(8) {
                    self.norm(r);
                }
            }
            Repr::Wide(n) => {
                // Volatile wide loads (the legalizer splits the others): one
                // `i64` per part, least significant at the lowest address.
                let a = self.take_scratch(ValType::I32);
                self.value(addr)?;
                self.local_set(a);
                for k in 0..n {
                    self.local_get(a);
                    self.mem(op::I64_LOAD, align, 8, k * 8);
                }
                self.free_scratch(ValType::I32, a);
            }
            Repr::Void => return fail("a load of void"),
        }
        Ok(())
    }

    fn store(&mut self, r: Repr, addr: ValueId, val: ValueId, align: u64) -> R<()> {
        match r {
            Repr::F32 | Repr::F64 => {
                self.value(addr)?;
                self.value(val)?;
                if r == Repr::F32 { self.mem(op::F32_STORE, align, 4, 0) } else { self.mem(op::F64_STORE, align, 8, 0) }
            }
            Repr::I32(b) | Repr::I64(b) => {
                let size = u64::from(b.div_ceil(8));
                let is64 = r.is64();
                let direct = match (size, is64) {
                    (1, false) => Some(op::I32_STORE8),
                    (2, false) => Some(op::I32_STORE16),
                    (4, false) => Some(op::I32_STORE),
                    (8, true) => Some(op::I64_STORE),
                    _ => None,
                };
                if let Some(opc) = direct {
                    self.value(addr)?;
                    self.value(val)?;
                    self.mem(opc, align, size, 0);
                } else {
                    let vt = if is64 { ValType::I64 } else { ValType::I32 };
                    let a = self.take_scratch(ValType::I32);
                    let x = self.take_scratch(vt);
                    self.value(addr)?;
                    self.local_set(a);
                    self.value(val)?;
                    self.local_set(x);
                    for (off, len) in pieces(size) {
                        self.local_get(a);
                        self.local_get(x);
                        if off != 0 {
                            self.int_const(r, off * 8);
                            self.code.byte(if is64 { op::I64_SHR_U } else { op::I32_SHR_U });
                        }
                        let opc = match (len, is64) {
                            (4, true) => op::I64_STORE32,
                            (2, true) => op::I64_STORE16,
                            (1, true) => op::I64_STORE8,
                            (2, false) => op::I32_STORE16,
                            _ => op::I32_STORE8,
                        };
                        self.mem(opc, 1, len, off as u32);
                    }
                    self.free_scratch(vt, x);
                    self.free_scratch(ValType::I32, a);
                }
            }
            Repr::Wide(n) => {
                let a = self.take_scratch(ValType::I32);
                self.value(addr)?;
                self.local_set(a);
                for k in 0..n {
                    self.local_get(a);
                    self.part(val, k)?;
                    self.mem(op::I64_STORE, align, 8, k * 8);
                }
                self.free_scratch(ValType::I32, a);
            }
            Repr::Void => return fail("a store of void"),
        }
        Ok(())
    }

    /// The container and byte size of an atomic access of type `ty`.
    fn atomic_repr(&self, ty: TypeId) -> R<(Repr, u64)> {
        let r = Repr::of(self.types, ty)?;
        match r {
            Repr::I32(8) | Repr::I32(16) | Repr::I32(32) | Repr::I64(64) => {
                let (Repr::I32(b) | Repr::I64(b)) = r else { unreachable!() };
                Ok((r, u64::from(b / 8)))
            }
            _ => fail(format!("an atomic access of {r:?} (atomics are i8, i16, i32, i64 or ptr)")),
        }
    }

    fn atomic_rmw(&mut self, op: RmwOp, ty: TypeId, addr: ValueId, v: ValueId) -> R<()> {
        let (r, size) = self.atomic_repr(ty)?;
        let slot = atomic_slot(r, size);
        let native = match op {
            RmwOp::Add => Some(0x1e),
            RmwOp::Sub => Some(0x25),
            RmwOp::And => Some(0x2c),
            RmwOp::Or => Some(0x33),
            RmwOp::Xor => Some(0x3a),
            RmwOp::Xchg => Some(0x41),
            _ => None,
        };
        if let Some(base) = native {
            self.value(addr)?;
            self.value(v)?;
            self.atomic(base + slot, size);
            return Ok(());
        }
        // nand / max / min / umax / umin: a compare-exchange loop.
        let is64 = r.is64();
        let vt = if is64 { ValType::I64 } else { ValType::I32 };
        let a = self.take_scratch(ValType::I32);
        let x = self.take_scratch(vt);
        let old = self.take_scratch(vt);
        self.value(addr)?;
        self.local_set(a);
        self.value(v)?;
        self.local_set(x);
        self.code.byte(op::LOOP);
        self.code.byte(op::EMPTY);
        self.local_get(a);
        self.atomic(0x10 + slot, size);
        self.local_set(old);
        self.local_get(a);
        self.local_get(old);
        match op {
            RmwOp::Nand => {
                self.local_get(old);
                self.local_get(x);
                self.code.byte(if is64 { op::I64_AND } else { op::I32_AND });
                self.int_const(r, u64::MAX);
                self.code.byte(if is64 { op::I64_XOR } else { op::I32_XOR });
                self.norm(r);
            }
            _ => {
                // new = cmp(old, x) ? old : x
                self.local_get(old);
                self.local_get(x);
                let signed = matches!(op, RmwOp::Max | RmwOp::Min);
                self.local_get(old);
                if signed {
                    self.sext(r);
                }
                self.local_get(x);
                if signed {
                    self.sext(r);
                }
                let pred = match op {
                    RmwOp::Max => IntPred::Sge,
                    RmwOp::Min => IntPred::Sle,
                    RmwOp::UMax => IntPred::Uge,
                    _ => IntPred::Ule,
                };
                let (o32, o64) = icmp_op(pred);
                self.code.byte(if is64 { o64 } else { o32 });
                self.code.byte(op::SELECT);
            }
        }
        self.atomic(0x48 + slot, size);
        self.local_get(old);
        let (ne32, ne64) = icmp_op(IntPred::Ne);
        self.code.byte(if is64 { ne64 } else { ne32 });
        self.code.byte(op::BR_IF);
        self.code.u32(0);
        self.code.byte(op::END);
        self.local_get(old);
        self.free_scratch(vt, old);
        self.free_scratch(vt, x);
        self.free_scratch(ValType::I32, a);
        Ok(())
    }

    fn note_callee(&mut self, f: u32) {
        let name = self.cx.obj.funcs[f as usize].name.clone();
        if !self.callees.contains(&name) {
            self.callees.push(name);
        }
    }

    fn call(&mut self, i: InstId, rty: Repr) -> R<()> {
        let inst = self.f.inst(i);
        let ops = inst.operands().to_vec();
        let (callee, args) = ops.split_first().expect("a call has a callee");
        // A result from one of our own functions keeps the invariant; one
        // from the host, or from a pointer that may lead there, is masked.
        let mut trusted = false;
        match self.f.value(*callee).def {
            ValueDef::Func(fid) => {
                let target = self.cx.module.function(fid);
                trusted = !target.is_declaration();
                if let Type::Func(ft) = self.types.get(target.sig)
                    && ft.params.len() != args.len()
                {
                    return fail(format!(
                        "a call to '{}' with {} arguments for {} parameters (variadic calls are not supported on wasm32)",
                        self.cx.syms.resolve(target.name),
                        args.len(),
                        ft.params.len()
                    ));
                }
                for &a in args {
                    self.values(a)?;
                }
                let idx = self.cx.func_index[fid.index()].expect("referenced");
                self.note_callee(idx);
                self.code.byte(op::CALL);
                self.code.hole(Target::Func(idx));
            }
            _ => {
                let mut params = Vec::new();
                for &a in args {
                    params.extend(self.repr(a)?.valtypes());
                    self.values(a)?;
                }
                self.value(*callee)?;
                let ty = self.cx.obj.intern_type(FuncType { params, results: rty.valtypes() });
                self.cx.obj.uses_indirect = true;
                self.indirect = true;
                self.code.byte(op::CALL_INDIRECT);
                self.code.hole(Target::Type(ty));
                self.code.byte(0x00); // table 0
            }
        }
        if needs_mask(rty) && !trusted {
            self.norm(rty);
        }
        Ok(())
    }
}

/// Whether values of `r` carry bits a host could leave set above the width.
fn needs_mask(r: Repr) -> bool {
    matches!(r, Repr::I32(b) if b < 32) || matches!(r, Repr::I64(b) if b < 64)
}

/// The position of an atomic access within each 7-opcode atomic group:
/// `i32`, `i64`, `i32` 8-bit, `i32` 16-bit, `i64` 8/16/32-bit.
fn atomic_slot(r: Repr, size: u64) -> u32 {
    match (r.is64(), size) {
        (false, 4) => 0,
        (true, 8) => 1,
        (false, 1) => 2,
        (false, 2) => 3,
        (true, 1) => 4,
        (true, 2) => 5,
        _ => 6,
    }
}

/// Split an access of `size` bytes (3, 5, 6 or 7) into naturally sized
/// little-endian pieces `(offset, length)`.
fn pieces(size: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut off = 0;
    while off < size {
        let len = [4u64, 2, 1].into_iter().find(|&l| off + l <= size).expect("1 always fits");
        out.push((off, len));
        off += len;
    }
    out
}
