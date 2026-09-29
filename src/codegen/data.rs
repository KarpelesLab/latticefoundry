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
//! The initializer is serialized per the IR's data layout
//! ([`TypeContext::layout`](crate::ir::TypeContext::layout)) in little-endian
//! order: integers as their two's-complement value truncated to the type's
//! store size, floats as their IEEE bit pattern, `null`/`poison` as zero bytes
//! (zero is a valid refinement of poison), arrays element-by-element at the
//! element stride, and structs field-by-field at their natural offsets with zero
//! padding. An address constant ([`Const::Addr`]) becomes a pointer-sized zero
//! field plus an absolute data relocation `S + offset` against the addressed
//! global's or function's symbol — the one target-specific input, supplied as
//! the [`RelocKind`] a target uses for an absolute pointer (`Abs64` on the
//! 64-bit targets, which each ELF writer maps to its `R_*_64`).
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

use crate::ir::{AddrTarget, Const, ConstId, FloatBits, GlobalId, Linkage, Module, Type};
use crate::mc::object::{
    ObjectModule, RelocKind, Relocation, Section, SectionId, SectionKind, Symbol, SymbolBinding,
    SymbolType,
};
use crate::support::StrInterner;

/// The serialized image of one global initializer: its bytes and the
/// `(offset, symbol, addend)` address fields that need a relocation.
#[derive(Default)]
struct Image {
    bytes: Vec<u8>,
    relocs: Vec<(u64, String, i64)>,
}

/// One global placed in an output section, awaiting its symbol and relocations.
struct Placed {
    /// Index into the section accumulators (`.rodata`, `.data`, `.bss`).
    which: usize,
    offset: u64,
    name: String,
    binding: SymbolBinding,
    size: u64,
    relocs: Vec<(u64, String, i64)>,
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
/// [`RelocKind::Abs64`]); `syms` resolves global and function names.
///
/// Call this *after* the functions have been emitted so a data reference to a
/// function binds to its existing definition (a later definition would also
/// update the symbol in place, so the order only affects symbol-table order).
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
/// [module docs](self)).
pub fn emit_globals_with(
    module: &Module,
    syms: &StrInterner,
    obj: &mut ObjectModule,
    abs_ptr: RelocKind,
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
        serialize(module, syms, init, 0, &mut img, abs_ptr);

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
        for (field, target, addend) in p.relocs {
            let symbol = obj.reference_symbol(&target);
            let offset = p.offset + field;
            obj.add_relocation(Relocation { section, offset, symbol, kind: abs_ptr, addend });
        }
    }
}

/// Write constant `cid` into `img` at byte offset `at`.
fn serialize(
    module: &Module,
    syms: &StrInterner,
    cid: ConstId,
    at: u64,
    img: &mut Image,
    abs_ptr: RelocKind,
) {
    let types = module.types();
    let put = |img: &mut Image, bytes: &[u8]| {
        let at = at as usize;
        img.bytes[at..at + bytes.len()].copy_from_slice(bytes);
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
                    serialize(module, syms, e, at + stride * i as u64, img, abs_ptr);
                }
            }
            Type::Struct(_) => {
                for (i, &e) in elems.iter().enumerate() {
                    let (off, _) = types.field_offset(*ty, i as u32);
                    serialize(module, syms, e, at + off, img, abs_ptr);
                }
            }
            // The verifier rejects an aggregate of scalar type; leave zeros.
            _ => {}
        },
        Const::Addr { ty, target, offset } => {
            debug_assert_eq!(
                types.size_of(*ty),
                abs_ptr.field_width() as u64,
                "pointer size must match the absolute relocation width"
            );
            let name = match *target {
                AddrTarget::Global(g) => module.global(g).name,
                AddrTarget::Func(f) => module.function(f).name,
            };
            img.relocs.push((at, syms.resolve(name).to_owned(), *offset));
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
}
