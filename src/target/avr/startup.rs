//! The startup object: the interrupt vector table and the reset code.
//!
//! ```text
//! __vectors:        jmp __init             ; reset
//!                   jmp __vector_1         ; ... one per interrupt
//! __init:           eor r1, r1             ; the zero register
//!                   out SREG, r1
//!                   ldi r28, lo8(__stack)  ; SP = RAMEND
//!                   ldi r29, hi8(__stack)
//!                   out SPH, r29
//!                   out SPL, r28
//!                   ; copy .data (and .rodata) from flash to SRAM
//!                   ldi r17, hi8(__data_end)
//!                   ldi r26, lo8(__data_start) ; ldi r27, hi8(__data_start)
//!                   ldi r30, lo8(__data_load_start) ; ldi r31, hi8(__data_load_start)
//!                   rjmp 2f
//!               1:  lpm r0, Z+ ; st X+, r0
//!               2:  cpi r26, lo8(__data_end) ; cpc r27, r17 ; brne 1b
//!                   ; zero .bss
//!                   ldi r18, hi8(__bss_end)
//!                   ldi r26, lo8(__bss_start) ; ldi r27, hi8(__bss_start)
//!                   rjmp 2f
//!               1:  st X+, r1
//!               2:  cpi r26, lo8(__bss_end) ; cpc r27, r18 ; brne 1b
//!                   call main
//!                   cli
//! __stop_program:   rjmp __stop_program    ; main's result stays in r25:r24
//! __bad_interrupt:  jmp __vectors          ; an unhandled interrupt restarts
//! ```
//!
//! Each `__vector_N` is a **weak** symbol at `__bad_interrupt`; a program
//! defining a strong `__vector_N` takes that vector (the function must then be
//! a real interrupt handler, which this backend does not generate yet).
//! `__stack`, `__data_*` and `__bss_*` are defined by [`super::link`].

use crate::mc::object::{ObjectModule, RelocKind, Relocation, Section, SectionKind, Symbol, SymbolBinding, SymbolType};

use super::encode::*;
use super::regs::{TMP, ZERO};

/// The name of the symbol main's return lands on (the final self-loop).
pub const STOP: &str = "__stop_program";

/// Build the startup object for `device`, calling `entry` (normally `main`).
pub fn object(device: &super::Device, entry: &str) -> ObjectModule {
    let mut obj = ObjectModule::new("avr_crt0");
    let text = obj.add_section(Section::new(".text", SectionKind::Text, 2));
    let mut words: Vec<u16> = Vec::new();
    let mut relocs: Vec<(usize, RelocKind, String)> = Vec::new();
    // (word index, kind, symbol): relocations recorded as we go.
    let w = |words: &mut Vec<u16>, relocs: &mut Vec<(usize, RelocKind, String)>, ws: &[u16], r: Option<(RelocKind, &str)>| {
        if let Some((k, s)) = r {
            relocs.push((words.len(), k, s.to_owned()));
        }
        words.extend_from_slice(ws);
    };
    let jmp0 = jmp(0);
    w(&mut words, &mut relocs, &jmp0, Some((RelocKind::AvrCall, "__init")));
    for v in 1..device.vectors {
        w(&mut words, &mut relocs, &jmp0, Some((RelocKind::AvrCall, &format!("__vector_{v}"))));
    }
    let init = words.len();
    w(&mut words, &mut relocs, &[eor(ZERO, ZERO), out(SREG, ZERO)], None);
    w(&mut words, &mut relocs, &[ldi(28, 0)], Some((RelocKind::AvrLo8Ldi, "__stack")));
    w(&mut words, &mut relocs, &[ldi(29, 0)], Some((RelocKind::AvrHi8Ldi, "__stack")));
    w(&mut words, &mut relocs, &[out(SPH, 29), out(SPL, 28)], None);
    // Copy the initialized data.
    w(&mut words, &mut relocs, &[ldi(17, 0)], Some((RelocKind::AvrHi8Ldi, "__data_end")));
    w(&mut words, &mut relocs, &[ldi(26, 0)], Some((RelocKind::AvrLo8Ldi, "__data_start")));
    w(&mut words, &mut relocs, &[ldi(27, 0)], Some((RelocKind::AvrHi8Ldi, "__data_start")));
    w(&mut words, &mut relocs, &[ldi(30, 0)], Some((RelocKind::AvrLo8Ldi, "__data_load_start")));
    w(&mut words, &mut relocs, &[ldi(31, 0)], Some((RelocKind::AvrHi8Ldi, "__data_load_start")));
    w(&mut words, &mut relocs, &[rjmp(2), lpm_inc(TMP), st_x_inc(TMP)], None);
    w(&mut words, &mut relocs, &[cpi(26, 0)], Some((RelocKind::AvrLo8Ldi, "__data_end")));
    w(&mut words, &mut relocs, &[cpc(27, 17), brbc(FLAG_Z, -5)], None);
    // Zero .bss.
    w(&mut words, &mut relocs, &[ldi(18, 0)], Some((RelocKind::AvrHi8Ldi, "__bss_end")));
    w(&mut words, &mut relocs, &[ldi(26, 0)], Some((RelocKind::AvrLo8Ldi, "__bss_start")));
    w(&mut words, &mut relocs, &[ldi(27, 0)], Some((RelocKind::AvrHi8Ldi, "__bss_start")));
    w(&mut words, &mut relocs, &[rjmp(1), st_x_inc(ZERO)], None);
    w(&mut words, &mut relocs, &[cpi(26, 0)], Some((RelocKind::AvrLo8Ldi, "__bss_end")));
    w(&mut words, &mut relocs, &[cpc(27, 18), brbc(FLAG_Z, -4)], None);
    w(&mut words, &mut relocs, &call(0), Some((RelocKind::AvrCall, entry)));
    w(&mut words, &mut relocs, &[CLI], None);
    let stop = words.len();
    w(&mut words, &mut relocs, &[rjmp(0)], Some((RelocKind::Avr13Pcrel, STOP)));
    let bad = words.len();
    w(&mut words, &mut relocs, &jmp0, Some((RelocKind::AvrCall, "__vectors")));

    let sec = obj.section_mut(text);
    for x in &words {
        sec.bytes.extend_from_slice(&x.to_le_bytes());
    }
    let len = 2 * words.len() as u64;
    obj.add_symbol(Symbol::defined("__vectors", SymbolBinding::Global, SymbolType::Func, text, 0, len));
    obj.add_symbol(Symbol::defined("__init", SymbolBinding::Local, SymbolType::Func, text, 2 * init as u64, 0));
    obj.add_symbol(Symbol::defined(STOP, SymbolBinding::Global, SymbolType::Func, text, 2 * stop as u64, 2));
    obj.add_symbol(Symbol::defined("__bad_interrupt", SymbolBinding::Global, SymbolType::Func, text, 2 * bad as u64, 4));
    for v in 1..device.vectors {
        obj.add_symbol(Symbol::defined(format!("__vector_{v}"), SymbolBinding::Weak, SymbolType::Func, text, 2 * bad as u64, 0));
    }
    for (at, kind, name) in relocs {
        let symbol = obj.reference_symbol(&name);
        // `rjmp` addends are relative to the field; the rest are absolute.
        obj.add_relocation(Relocation { section: text, offset: 2 * at as u64, symbol, kind, addend: 0 });
    }
    obj
}
