//! Global data for AVR's two memories.
//!
//! [`crate::codegen::data`] serializes the initializers; this module decides
//! where they go:
//!
//! - globals in **address space 0** (data memory) get the usual `.rodata`,
//!   `.data` and `.bss` sections. On AVR all three live in SRAM — `ld` cannot
//!   read flash — so `.rodata` and `.data` are initialized by the startup code
//!   from a copy in flash (see [`super::link`]);
//! - globals in **address space 1** (program memory) all go to
//!   `.progmem.data`, which stays in flash and is read with `lpm`.
//!
//! A pointer field in an initializer is an `R_AVR_16` (a data or program-memory
//! byte address), except that the address of a **function** is its word
//! address, `R_AVR_16_PM` — what `icall` and avr-gcc's function pointers use.

use crate::mc::object::{ObjectModule, RelocKind, Relocation, Section, SectionKind, Symbol, SymbolValue};
use crate::ir::{GlobalId, Module};
use crate::support::StrInterner;

/// The name of the flash data section.
pub(crate) const PROGMEM: &str = ".progmem.data";

/// Emit every defined global of the prepared module `m` into `obj`.
pub(crate) fn emit_globals(m: &Module, s: &StrInterner, obj: &mut ObjectModule) {
    let funcs: Vec<String> = m.functions().map(|f| s.resolve(f.name).to_owned()).collect();
    for flash in [false, true] {
        // A copy of the module holding only this memory's initializers.
        let (mut copy, cs) = super::prepare::copy_module(m, s).unwrap_or_else(|e| panic!("avr backend: {e}"));
        let mut any = false;
        for g in 0..copy.global_count() {
            let id = GlobalId::from_index(g);
            if (copy.global_addr_space(id) != 0) != flash {
                copy.set_global_init(id, None);
            } else if copy.global(id).init.is_some() {
                any = true;
            }
        }
        if !any {
            continue;
        }
        let mut tmp = ObjectModule::new("data");
        crate::codegen::data::emit_globals_per_space(&copy, &cs, &mut tmp, &|_, bytes| RelocKind::abs_for_width(bytes), false);
        merge(obj, &tmp, flash, &funcs);
    }
}

/// Append the sections of `tmp` to `obj` (all into one `.progmem.data` when
/// `flash`), with their symbols and relocations.
fn merge(obj: &mut ObjectModule, tmp: &ObjectModule, flash: bool, funcs: &[String]) {
    // Where each tmp section went: (section, base offset).
    let mut place = Vec::with_capacity(tmp.sections().len());
    let mut flash_sec = None;
    for sec in tmp.sections() {
        if flash {
            let id = *flash_sec.get_or_insert_with(|| obj.add_section(Section::new(PROGMEM, SectionKind::Rodata, 1)));
            let dst = obj.section_mut(id);
            let base = dst.bytes.len() as u64;
            if sec.is_nobits() {
                dst.bytes.resize((base + sec.size()) as usize, 0);
            } else {
                dst.bytes.extend_from_slice(&sec.bytes);
            }
            place.push((id, base));
        } else {
            let mut copy = sec.clone();
            copy.align = 1;
            place.push((obj.add_section(copy), 0));
        }
    }
    for sym in tmp.symbols() {
        if let SymbolValue::Defined { section, offset } = sym.value {
            let (sec, base) = place[section.index()];
            let mut s = Symbol::defined(sym.name.clone(), sym.binding, sym.kind, sec, base + offset, sym.size);
            s.visibility = sym.visibility;
            obj.add_symbol(s);
        }
    }
    for r in tmp.relocations() {
        let (sec, base) = place[r.section.index()];
        let name = &tmp.symbol(r.symbol).name;
        let kind = if r.kind == RelocKind::Abs16 && funcs.contains(name) { RelocKind::Avr16Pm } else { r.kind };
        let symbol = obj.reference_symbol(name);
        obj.add_relocation(Relocation { section: sec, offset: base + r.offset, symbol, kind, addend: r.addend });
    }
}
