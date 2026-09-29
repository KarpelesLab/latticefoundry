//! Target-independent emission of a module's **global data** into an
//! [`ObjectModule`] (ROADMAP Phase 7/8).
//!
//! Every global the module *defines* — it has an initializer and is not
//! [`detached`](crate::ir::GlobalAttrs::detached) — becomes storage in one of
//! three sections plus an `STT_OBJECT` symbol:
//!
//! | global                                   | section   | ELF type / flags            |
//! |------------------------------------------|-----------|-----------------------------|
//! | `constant`                               | `.rodata` | `PROGBITS`, `A`             |
//! | mutable, initializer has a nonzero byte or an address | `.data` | `PROGBITS`, `WA` |
//! | mutable, initializer is all zero / poison | `.bss`   | `NOBITS`, `WA`              |
//!
//! The initializer is serialized per the module's
//! [`DataLayout`](crate::ir::DataLayout) (through
//! [`TypeContext::layout`](crate::ir::TypeContext::layout)) in the layout's
//! byte order: integers as their two's-complement value truncated to the type's
//! store size, floats as their IEEE bit pattern, `null`/`poison` as zero bytes
//! (zero is a valid refinement of poison), arrays element-by-element at the
//! element stride, and structs field-by-field at their natural offsets with zero
//! padding. An address constant ([`Const::Addr`]) becomes a zero field of its
//! pointer type's size (8, 4 or 2 bytes, per its address space) plus an
//! absolute data relocation `S + offset` against the addressed global's or
//! function's symbol — the one target-specific input: [`emit_globals_per_space`]
//! asks the target for the relocation of each pointer width and address space,
//! and [`emit_globals`] takes the one kind a single-pointer-width target uses
//! (`Abs64` on the 64-bit targets, which each ELF writer maps to its `R_*_64`),
//! falling back to the generic [`RelocKind::abs_for_width`] for other widths.
//!
//! For position-independent output ([`emit_globals_with`] with `relro`), a
//! `constant` global whose initializer holds an address goes to
//! **`.data.rel.ro`** (`PROGBITS`, `WA`) instead of `.rodata`: its pointer fields
//! need load-time (dynamic) relocations, which a read-only segment cannot take
//! without text relocations; the linker places `.data.rel.ro` in the `RELRO`
//! region, which the dynamic loader makes read-only once relocated.
//!
//! Each global is placed at its type's alignment and every section takes the
//! largest alignment it holds. The symbol binding follows the global's
//! [`Linkage`]: external → `Global`, internal → `Local`, weak → `Weak`. A global
//! of zero size still reserves one byte so distinct globals have distinct
//! addresses. Globals with no initializer (or detached ones) emit nothing: a
//! reference to them — from code or from another initializer — creates the
//! undefined symbol the linker resolves.
//!
//! Emission is deterministic (tenet T5): globals are visited in module order and
//! the sections are appended in the fixed order `.rodata`, `.data`, `.bss`,
//! `.data.rel.ro`, each only when non-empty.

use crate::ir::{AddrTarget, Const, ConstId, Endian, FloatBits, GlobalId, Linkage, Module, Type};
use crate::mc::object::{
    ObjectModule, RelocKind, Relocation, Section, SectionId, SectionKind, Symbol, SymbolBinding,
    SymbolType,
};
use crate::support::StrInterner;

/// The serialized image of one global initializer: its bytes and the address
/// fields that need a relocation.
#[derive(Default)]
struct Image {
    bytes: Vec<u8>,
    relocs: Vec<AddrField>,
}

/// One address field of a serialized initializer.
struct AddrField {
    /// Byte offset of the field within the global.
    at: u64,
    /// Width of the field in bytes (the pointer size of its address space).
    width: u64,
    /// The address space of the pointer.
    space: u32,
    /// The addressed symbol.
    target: String,
    /// The byte offset added to the symbol's address.
    addend: i64,
}

/// One global placed in an output section, awaiting its symbol and relocations.
struct Placed {
    /// Index into the section accumulators (`.rodata`, `.data`, `.bss`).
    which: usize,
    offset: u64,
    name: String,
    binding: SymbolBinding,
    size: u64,
    relocs: Vec<AddrField>,
}

/// One output section being accumulated.
struct Acc {
    kind: SectionKind,
    name: &'static str,
    align: u64,
    /// Content bytes (for `.bss`, only the running size matters).
    bytes: Vec<u8>,
    size: u64,
}

impl Acc {
    fn new(kind: SectionKind, name: &'static str) -> Acc {
        Acc { kind, name, align: 1, bytes: Vec::new(), size: 0 }
    }

    /// Reserve `size` bytes at `align`, returning the offset.
    fn place(&mut self, size: u64, align: u64) -> u64 {
        self.align = self.align.max(align);
        let off = self.size.div_ceil(align) * align;
        self.size = off + size;
        if self.kind != SectionKind::Bss {
            self.bytes.resize(self.size as usize, 0);
        }
        off
    }
}

/// Emit every defined global of `module` into `obj` (see the [module
/// docs](self)). `abs_ptr` is the target's absolute pointer relocation (e.g.
/// [`RelocKind::Abs64`]), used for every address field of its width; a field
/// of another width (a pointer into an address space of a different size) uses
/// [`RelocKind::abs_for_width`]. `syms` resolves global and function names.
///
/// Call this *after* the functions have been emitted so a data reference to a
/// function binds to its existing definition (a later definition would also
/// update the symbol in place, so the order only affects symbol-table order).
///
/// # Panics
///
/// If an address field has a width no relocation covers (see
/// [`emit_globals_per_space`]).
pub fn emit_globals(
    module: &Module,
    syms: &StrInterner,
    obj: &mut ObjectModule,
    abs_ptr: RelocKind,
) {
    emit_globals_with(module, syms, obj, abs_ptr, false);
}

/// Like [`emit_globals`]; with `relro` (position-independent output), constant
/// globals holding addresses go to `.data.rel.ro` rather than `.rodata` (see the
/// [module docs](self)). `abs_ptr` relocates every address field of its width;
/// a field of another width uses [`RelocKind::abs_for_width`].
pub fn emit_globals_with(
    module: &Module,
    syms: &StrInterner,
    obj: &mut ObjectModule,
    abs_ptr: RelocKind,
    relro: bool,
) {
    let chooser = |_space: u32, bytes: u64| {
        if bytes == abs_ptr.field_width() as u64 { Some(abs_ptr) } else { RelocKind::abs_for_width(bytes) }
    };
    emit_globals_per_space(module, syms, obj, &chooser, relro);
}

/// Emit every defined global of `module` into `obj`, like [`emit_globals_with`],
/// but asking `abs_reloc(address_space, field_bytes)` for the absolute
/// relocation of each address field: a target with several pointer widths or
/// address spaces (e.g. AVR data vs. program memory) picks a kind per space.
/// The kind's [`field_width`](RelocKind::field_width) must equal `field_bytes`.
///
/// # Panics
///
/// If `abs_reloc` returns `None` (or a kind of the wrong width) for an address
/// field the module contains: the target cannot relocate a pointer of that
/// size, which its data layout should not have declared.
pub fn emit_globals_per_space(
    module: &Module,
    syms: &StrInterner,
    obj: &mut ObjectModule,
    abs_reloc: &dyn Fn(u32, u64) -> Option<RelocKind>,
    relro: bool,
) {
    let types = module.types();
    let mut accs = [
        Acc::new(SectionKind::Rodata, ".rodata"),
        Acc::new(SectionKind::Data, ".data"),
        Acc::new(SectionKind::Bss, ".bss"),
        Acc::new(SectionKind::Data, ".data.rel.ro"),
    ];
    let mut placed: Vec<Placed> = Vec::new();

    for (gi, g) in module.globals().enumerate() {
        let attrs = module.global_attrs(GlobalId::from_index(gi));
        let Some(init) = g.init else { continue };
        if attrs.detached {
            continue;
        }
        let layout = types.layout(g.ty);
        let mut img = Image { bytes: vec![0u8; layout.size as usize], relocs: Vec::new() };
        serialize(module, syms, init, 0, &mut img);

        let which = if attrs.constant && relro && !img.relocs.is_empty() {
            3
        } else if attrs.constant {
            0
        } else if img.relocs.is_empty() && img.bytes.iter().all(|&b| b == 0) {
            2
        } else {
            1
        };
        let acc = &mut accs[which];
        let off = acc.place(layout.size.max(1), layout.align.max(1));
        if acc.kind != SectionKind::Bss {
            acc.bytes[off as usize..(off + layout.size) as usize].copy_from_slice(&img.bytes);
        }
        let binding = match attrs.linkage {
            Linkage::External => SymbolBinding::Global,
            Linkage::Internal => SymbolBinding::Local,
            Linkage::Weak => SymbolBinding::Weak,
        };
        placed.push(Placed {
            which,
            offset: off,
            name: syms.resolve(g.name).to_owned(),
            binding,
            size: layout.size,
            relocs: img.relocs,
        });
    }

    // Materialize the non-empty sections in fixed order.
    let mut ids: [Option<SectionId>; 4] = [None; 4];
    for (i, acc) in accs.into_iter().enumerate() {
        if acc.size == 0 {
            continue;
        }
        let section = if acc.kind == SectionKind::Bss {
            Section::bss(acc.name, acc.align, acc.size)
        } else {
            let mut s = Section::new(acc.name, acc.kind, acc.align);
            s.bytes = acc.bytes;
            s
        };
        ids[i] = Some(obj.add_section(section));
    }

    // Define every symbol first, so relocations below bind to these definitions
    // (including internal ones) rather than to fresh undefined references.
    for p in &placed {
        let sec = ids[p.which].expect("a placed global's section exists");
        let kind = SymbolType::Object;
        obj.add_symbol(Symbol::defined(p.name.clone(), p.binding, kind, sec, p.offset, p.size));
    }
    for p in placed {
        let section = ids[p.which].expect("a placed global's section exists");
        for r in p.relocs {
            let kind = abs_reloc(r.space, r.width)
                .filter(|k| k.field_width() as u64 == r.width)
                .unwrap_or_else(|| {
                    panic!(
                        "no {}-byte absolute relocation for a pointer into address space {}",
                        r.width, r.space
                    )
                });
            let symbol = obj.reference_symbol(&r.target);
            let offset = p.offset + r.at;
            obj.add_relocation(Relocation { section, offset, symbol, kind, addend: r.addend });
        }
    }
}

/// Write constant `cid` into `img` at byte offset `at`.
fn serialize(module: &Module, syms: &StrInterner, cid: ConstId, at: u64, img: &mut Image) {
    let types = module.types();
    let big = types.data_layout().endian() == Endian::Big;
    // Scalars are produced little-endian; a big-endian layout stores them
    // most-significant byte first.
    let put = |img: &mut Image, bytes: &[u8]| {
        let at = at as usize;
        let field = &mut img.bytes[at..at + bytes.len()];
        field.copy_from_slice(bytes);
        if big {
            field.reverse();
        }
    };
    match module.consts().get(cid) {
        Const::Int { ty, value } => {
            let width = types.size_of(*ty) as usize;
            put(img, &twos_complement_le(value, width));
        }
        Const::Float { bits, .. } => match bits {
            FloatBits::F16(b) => put(img, &b.to_le_bytes()),
            FloatBits::F32(b) => put(img, &b.to_le_bytes()),
            FloatBits::F64(b) => put(img, &b.to_le_bytes()),
        },
        // Already zero-filled.
        Const::Null(_) | Const::Poison(_) => {}
        Const::Aggregate { ty, elems } => match types.get(*ty) {
            Type::Array(elem, _) => {
                let stride = types.stride(*elem);
                for (i, &e) in elems.iter().enumerate() {
                    serialize(module, syms, e, at + stride * i as u64, img);
                }
            }
            Type::Struct(_) => {
                for (i, &e) in elems.iter().enumerate() {
                    let (off, _) = types.field_offset(*ty, i as u32);
                    serialize(module, syms, e, at + off, img);
                }
            }
            // Vector lanes are packed at the element's size (an `i1` lane is one
            // byte holding 0 or 1), lane 0 first.
            Type::Vector(elem, _) => {
                let size = types.size_of(*elem);
                for (i, &e) in elems.iter().enumerate() {
                    serialize(module, syms, e, at + size * i as u64, img);
                }
            }
            // The verifier rejects an aggregate of scalar type; leave zeros.
            _ => {}
        },
        Const::Addr { ty, target, offset } => {
            let name = match *target {
                AddrTarget::Global(g) => module.global(g).name,
                AddrTarget::Func(f) => module.function(f).name,
            };
            img.relocs.push(AddrField {
                at,
                width: types.size_of(*ty),
                space: types.addr_space(*ty).unwrap_or(0),
                target: syms.resolve(name).to_owned(),
                addend: *offset,
            });
        }
    }
}

/// The `width`-byte little-endian two's-complement image of `value` (reduced
/// modulo `2^(8*width)`).
fn twos_complement_le(value: &puremp::Int, width: usize) -> Vec<u8> {
    let mut out = value.magnitude().to_bytes_le();
    out.resize(width, 0);
    if value.sign() == puremp::Sign::Negative {
        // -m mod 2^n  ==  !m + 1  (over n bits).
        let mut carry = true;
        for b in &mut out {
            let (v, c) = (!*b).overflowing_add(u8::from(carry));
            *b = v;
            carry = c;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Global, GlobalAttrs};

    #[test]
    fn twos_complement_truncates_and_negates() {
        assert_eq!(twos_complement_le(&puremp::Int::from_i64(0x1234), 2), vec![0x34, 0x12]);
        assert_eq!(twos_complement_le(&puremp::Int::from_i64(-1), 4), vec![0xff; 4]);
        assert_eq!(twos_complement_le(&puremp::Int::from_i64(-2), 2), vec![0xfe, 0xff]);
        // Wider than i64: 2^64 + 1 in 16 bytes.
        let big = puremp::Int::from_str_radix("18446744073709551617", 10).unwrap();
        let mut want = vec![0u8; 16];
        want[0] = 1;
        want[8] = 1;
        assert_eq!(twos_complement_le(&big, 16), want);
        // Truncation: 0x1ff in one byte is 0xff.
        assert_eq!(twos_complement_le(&puremp::Int::from_i64(0x1ff), 1), vec![0xff]);
    }

    /// A struct with padding, an array, a float, and an address field land in
    /// the right sections at the right offsets, with the right bindings.
    #[test]
    fn layout_sections_bindings_and_relocs() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("d");
        let i8t = m.types_mut().int(8);
        let i32t = m.types_mut().int(32);
        let f64t = m.types_mut().float(crate::ir::FloatKind::F64);
        let ptr = m.types_mut().ptr();
        let st = m.types_mut().struct_(vec![i8t, i32t, ptr]);
        let arr = m.types_mut().array(i32t, 4);
        let int = |m: &mut Module, ty, v| m.intern_const(Const::Int { ty, value: puremp::Int::from_i64(v) });

        // rodata: constant f64 1.0
        let one = m.intern_const(Const::Float { ty: f64t, bits: FloatBits::F64(1.0f64.to_bits()) });
        let k = GlobalAttrs { constant: true, ..GlobalAttrs::DEFAULT };
        m.define_global(Global { name: syms.intern("k"), ty: f64t, init: Some(one) }, k);
        // bss: zero array, internal
        let zeros: Vec<ConstId> = (0..4).map(|_| int(&mut m, i32t, 0)).collect();
        let zagg = m.intern_const(Const::Aggregate { ty: arr, elems: zeros });
        let internal = GlobalAttrs { linkage: Linkage::Internal, ..GlobalAttrs::DEFAULT };
        let zid = m.define_global(Global { name: syms.intern("z"), ty: arr, init: Some(zagg) }, internal);
        // data: {i8 7, i32 -2, ptr @z + 8}, weak
        let a = int(&mut m, i8t, 7);
        let b = int(&mut m, i32t, -2);
        let c = m.intern_const(Const::Addr { ty: ptr, target: AddrTarget::Global(zid), offset: 8 });
        let sagg = m.intern_const(Const::Aggregate { ty: st, elems: vec![a, b, c] });
        let weak = GlobalAttrs { linkage: Linkage::Weak, ..GlobalAttrs::DEFAULT };
        m.define_global(Global { name: syms.intern("s"), ty: st, init: Some(sagg) }, weak);
        // Detached and declared globals emit nothing.
        m.add_global(Global { name: syms.intern("det"), ty: i32t, init: Some(b) });
        m.define_global(Global { name: syms.intern("ext"), ty: i32t, init: None }, GlobalAttrs::DEFAULT);

        let mut obj = ObjectModule::new("d");
        emit_globals(&m, &syms, &mut obj, RelocKind::Abs64);

        let names: Vec<&str> = obj.sections().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, [".rodata", ".data", ".bss"]);
        assert_eq!(obj.sections()[0].bytes, 1.0f64.to_le_bytes());
        assert_eq!(obj.sections()[0].align, 8);
        let data = &obj.sections()[1];
        assert_eq!(data.align, 8);
        let mut want = vec![7u8, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff];
        want.extend_from_slice(&[0; 8]);
        assert_eq!(data.bytes, want);
        assert!(obj.sections()[2].is_nobits());
        assert_eq!(obj.sections()[2].size(), 16);
        assert_eq!(obj.sections()[2].align, 4);

        let sym = |n: &str| obj.symbol(obj.symbol_id(n).unwrap()).clone();
        assert_eq!(sym("k").binding, SymbolBinding::Global);
        assert_eq!(sym("z").binding, SymbolBinding::Local);
        assert_eq!(sym("s").binding, SymbolBinding::Weak);
        assert_eq!(sym("s").size, 16);
        assert_eq!(sym("s").kind, SymbolType::Object);
        assert!(obj.symbol_id("det").is_none() && obj.symbol_id("ext").is_none());

        let r = obj.relocations();
        assert_eq!(r.len(), 1);
        assert_eq!((r[0].offset, r[0].kind, r[0].addend), (8, RelocKind::Abs64, 8));
        assert_eq!(obj.symbol(r[0].symbol).name, "z");
        assert!(!obj.symbol(r[0].symbol).is_undefined(), "binds to the local definition");
    }

    /// `global @z : [2 x i16] = 0` and `global @s : {i8, i32, ptr} = {7, -2,
    /// ptr @z + 2}` under `layout`, emitted with `emit_globals(abs)`: the
    /// `.data` bytes and the one relocation `(offset, kind)`.
    fn emit_struct_under(layout: &str, abs: RelocKind) -> (Vec<u8>, u64, u64, RelocKind) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("d");
        m.set_data_layout(crate::ir::DataLayout::parse(layout).unwrap());
        let i8t = m.types_mut().int(8);
        let i16t = m.types_mut().int(16);
        let i32t = m.types_mut().int(32);
        let ptr = m.types_mut().ptr();
        let st = m.types_mut().struct_(vec![i8t, i32t, ptr]);
        let arr = m.types_mut().array(i16t, 2);
        let zero = m.intern_const(Const::Int { ty: i16t, value: puremp::Int::ZERO });
        let zagg = m.intern_const(Const::Aggregate { ty: arr, elems: vec![zero, zero] });
        let zid = m.define_global(Global { name: syms.intern("z"), ty: arr, init: Some(zagg) }, GlobalAttrs::DEFAULT);
        let a = m.intern_const(Const::Int { ty: i8t, value: puremp::Int::from_i64(7) });
        let b = m.intern_const(Const::Int { ty: i32t, value: puremp::Int::from_i64(-2) });
        let c = m.intern_const(Const::Addr { ty: ptr, target: AddrTarget::Global(zid), offset: 2 });
        let sagg = m.intern_const(Const::Aggregate { ty: st, elems: vec![a, b, c] });
        m.define_global(Global { name: syms.intern("s"), ty: st, init: Some(sagg) }, GlobalAttrs::DEFAULT);
        let mut obj = ObjectModule::new("d");
        emit_globals(&m, &syms, &mut obj, abs);
        let data = obj.sections().iter().find(|s| s.name == ".data").expect(".data");
        let s = obj.symbol(obj.symbol_id("s").unwrap());
        let r = obj.relocations();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].addend, 2);
        (data.bytes.clone(), s.size, r[0].offset, r[0].kind)
    }

    #[test]
    fn four_and_two_byte_pointers() {
        // ILP32: i8@0, i32@4, ptr@8 (4 bytes); size 12.
        let (bytes, size, at, kind) = emit_struct_under("p:32:32-n32", RelocKind::Abs32);
        assert_eq!(bytes, [7, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff, 0, 0, 0, 0]);
        assert_eq!((size, at, kind), (12, 8, RelocKind::Abs32));
        // AVR-like: everything byte-aligned, 2-byte pointers: i8@0 i32@1 ptr@5.
        let (bytes, size, at, kind) = emit_struct_under("p:16:8-i16:8-i32:8-n8", RelocKind::Abs16);
        assert_eq!(bytes, [7, 0xfe, 0xff, 0xff, 0xff, 0, 0]);
        assert_eq!((size, at, kind), (7, 5, RelocKind::Abs16));
        // A target passing its 64-bit kind still gets the generic 2-byte one.
        let (_, _, _, kind) = emit_struct_under("p:16:8-i16:8-i32:8-n8", RelocKind::Abs64);
        assert_eq!(kind, RelocKind::Abs16);
        // Big-endian stores the i32 most-significant byte first.
        let (bytes, ..) = emit_struct_under("E-p:32:32", RelocKind::Abs32);
        assert_eq!(bytes[4..8], [0xff, 0xff, 0xff, 0xfe]);
    }

    /// A per-address-space relocation choice: a data pointer to a function in
    /// program space 1 uses the kind the target returns for space 1.
    #[test]
    fn per_address_space_relocations() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("d");
        m.set_data_layout(crate::ir::DataLayout::parse("p:16:8-p1:32:8-P1").unwrap());
        let fptr = m.types_mut().ptr_in(1);
        let void = m.types_mut().void();
        let sig = m.types_mut().func(vec![], void, false);
        let f = m.declare_function(syms.intern("f"), sig);
        let c = m.intern_const(Const::Addr { ty: fptr, target: AddrTarget::Func(f), offset: 0 });
        m.define_global(Global { name: syms.intern("vec"), ty: fptr, init: Some(c) }, GlobalAttrs::DEFAULT);
        let mut obj = ObjectModule::new("d");
        let chooser = |space: u32, bytes: u64| {
            assert_eq!((space, bytes), (1, 4));
            Some(RelocKind::Abs32)
        };
        emit_globals_per_space(&m, &syms, &mut obj, &chooser, false);
        assert_eq!(obj.relocations()[0].kind, RelocKind::Abs32);
        assert_eq!(obj.sections()[0].bytes.len(), 4);
    }
}
