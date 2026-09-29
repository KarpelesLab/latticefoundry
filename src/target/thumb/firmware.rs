//! Cortex-M firmware: a vector table and reset handler for an entry symbol,
//! a linker script, and the link through `qld` into an ELF executable whose
//! loadable contents become a raw binary or Intel HEX image
//! ([`crate::link::raw`]).
//!
//! # The startup object
//!
//! [`startup_object`] builds, as an ordinary relocatable object:
//!
//! - `.isr_vector`: the ARMv7-M vector table the core reads at reset (B1.5.3
//!   of the ARMv7-M ARM) — word 0 the initial main stack pointer (`_estack`),
//!   word 1 the reset handler's address with the Thumb bit, then the system
//!   exceptions (NMI, HardFault, MemManage, BusFault, UsageFault, four
//!   reserved words, SVCall, DebugMonitor, one reserved, PendSV, SysTick) and
//!   the requested number of external interrupts, each pointing to a **weak**
//!   handler symbol (`NMI_Handler`, …, `IRQ0_Handler`, …) that defaults to
//!   `Default_Handler`, an infinite loop. A program overrides a handler by
//!   defining a function of that name;
//! - `Reset_Handler`: copies `.data` from its load address in flash
//!   (`_sidata`) to RAM (`_sdata`..`_edata`), zeroes `.bss` (`_sbss`..`_ebss`),
//!   calls the entry function, and parks in a loop if it returns.
//!
//! # The memory map
//!
//! [`linker_script`] places the vector table at the start of flash, then
//! `.text` and `.rodata`; `.data` runs from RAM with its initial image stored
//! in flash after the code; `.bss` follows it in RAM; the stack starts at the
//! top of RAM (`_estack`, full-descending). The default [`MemoryLayout`] is
//! the Cortex-M architectural one — code at `0x0000_0000`, SRAM at
//! `0x2000_0000` — with 1 MiB of flash and 64 KiB of RAM; a device picks its
//! own (an STM32's flash is at `0x0800_0000`, mirrored at 0 on boot).
//!
//! A hand-written vector table works the same way: define a `.isr_vector`
//! section (or link a C startup file) and leave [`startup_object`] out.

use std::path::Path;

use crate::mc::object::{ObjectModule, RelocKind, Relocation, Section, SectionKind, Symbol, SymbolBinding, SymbolType};

use super::encode::{Asm, T, addsub_imm8, b16, dp16, ldst_imm16, movs_imm8, movt, movw};

/// Where a Cortex-M part's flash and RAM are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MemoryLayout {
    /// The first byte of flash (the vector table's address).
    pub flash_origin: u64,
    /// The flash size in bytes.
    pub flash_size: u64,
    /// The first byte of RAM.
    pub ram_origin: u64,
    /// The RAM size in bytes (the initial stack pointer is its end).
    pub ram_size: u64,
}

impl Default for MemoryLayout {
    fn default() -> MemoryLayout {
        MemoryLayout { flash_origin: 0, flash_size: 1 << 20, ram_origin: 0x2000_0000, ram_size: 64 << 10 }
    }
}

/// The system exception handlers of vector-table entries 2..=15 (`None` for a
/// reserved entry).
const SYSTEM_HANDLERS: [Option<&str>; 14] = [
    Some("NMI_Handler"),
    Some("HardFault_Handler"),
    Some("MemManage_Handler"),
    Some("BusFault_Handler"),
    Some("UsageFault_Handler"),
    None,
    None,
    None,
    None,
    Some("SVC_Handler"),
    Some("DebugMon_Handler"),
    None,
    Some("PendSV_Handler"),
    Some("SysTick_Handler"),
];

/// The startup object for a program whose entry function is `entry`, with a
/// vector table of the 16 system entries plus `irqs` external interrupts (see
/// the [module docs](self)).
pub fn startup_object(entry: &str, irqs: usize) -> ObjectModule {
    let mut obj = ObjectModule::new("lf-cortex-m-startup");
    let vec = obj.add_section(Section::new(".isr_vector", SectionKind::Rodata, 4));
    let text = obj.add_section(Section::new(".text", SectionKind::Text, 4));

    // --- the code: Reset_Handler, then Default_Handler -----------------------
    let mut a = Asm::default();
    let addr = |a: &mut Asm, r: u32, sym: &str| {
        a.reloc(movw(r, 0), sym.to_owned(), RelocKind::ThumbMovwAbsNc, 0);
        a.reloc(movt(r, 0), sym.to_owned(), RelocKind::ThumbMovtAbs, 0);
    };
    const HS: u32 = 0x2;
    addr(&mut a, 0, "_sidata");
    addr(&mut a, 1, "_sdata");
    addr(&mut a, 2, "_edata");
    let (copy, zero, zloop, done) = (a.new_label(), a.new_label(), a.new_label(), a.new_label());
    a.bind(copy);
    a.i(dp16(10, 1, 2)); // cmp r1, r2
    a.branch(HS, zero);
    a.i(ldst_imm16(true, 4, 3, 0, 0)); // ldr r3, [r0]
    a.i(addsub_imm8(false, 0, 4)); // adds r0, #4
    a.i(ldst_imm16(false, 4, 3, 1, 0)); // str r3, [r1]
    a.i(addsub_imm8(false, 1, 4)); // adds r1, #4
    a.branch(super::encode::AL, copy);
    a.bind(zero);
    addr(&mut a, 1, "_sbss");
    addr(&mut a, 2, "_ebss");
    a.i(movs_imm8(3, 0));
    a.bind(zloop);
    a.i(dp16(10, 1, 2)); // cmp r1, r2
    a.branch(HS, done);
    a.i(ldst_imm16(false, 4, 3, 1, 0)); // str r3, [r1]
    a.i(addsub_imm8(false, 1, 4)); // adds r1, #4
    a.branch(super::encode::AL, zloop);
    a.bind(done);
    a.reloc(super::encode::b32(true, -4), entry.to_owned(), RelocKind::ThumbCall, -4);
    a.i(b16(-4)); // 1: b 1b — park if the entry returns
    let reset = a.finish();
    let reset_len = reset.bytes.len() as u64;
    let default_off = reset_len.next_multiple_of(4);
    {
        let sec = obj.section_mut(text);
        sec.bytes = reset.bytes;
        while !sec.bytes.len().is_multiple_of(4) {
            sec.bytes.extend_from_slice(&T::N(0xbf00).bytes()); // nop
        }
        sec.bytes.extend_from_slice(&b16(-4).bytes()); // Default_Handler: b .
    }
    obj.add_symbol(Symbol::defined("$t", SymbolBinding::Local, SymbolType::NoType, text, 0, 0));
    obj.add_symbol(Symbol::defined("Reset_Handler", SymbolBinding::Global, SymbolType::Func, text, 1, reset_len));
    obj.add_symbol(Symbol::defined(
        "Default_Handler",
        SymbolBinding::Weak,
        SymbolType::Func,
        text,
        default_off | 1,
        2,
    ));
    for r in reset.relocations {
        let s = obj.reference_symbol(&r.symbol);
        obj.add_relocation(Relocation { section: text, offset: r.offset, symbol: s, kind: r.kind, addend: r.addend });
    }

    // --- the vector table ----------------------------------------------------
    let irq_names: Vec<String> = (0..irqs).map(|k| format!("IRQ{k}_Handler")).collect();
    let mut entries: Vec<Option<String>> = vec![Some("_estack".into()), Some("Reset_Handler".into())];
    entries.extend(SYSTEM_HANDLERS.iter().map(|h| h.map(str::to_owned)));
    entries.extend(irq_names.iter().cloned().map(Some));
    obj.section_mut(vec).bytes = vec![0; 4 * entries.len()];
    for (k, e) in entries.iter().enumerate() {
        let Some(name) = e else { continue };
        if k >= 2 {
            // A weak handler, defaulting to Default_Handler.
            obj.add_symbol(Symbol::defined(name.clone(), SymbolBinding::Weak, SymbolType::Func, text, default_off | 1, 2));
        }
        let s = obj.reference_symbol(name);
        obj.add_relocation(Relocation { section: vec, offset: 4 * k as u64, symbol: s, kind: RelocKind::Abs32, addend: 0 });
    }
    obj
}

/// The GNU-`ld`-syntax linker script for `layout` (see the [module
/// docs](self)): vector table and code in flash, `.data` in RAM loaded from
/// flash, `.bss` in RAM, and the `_sidata`/`_sdata`/`_edata`/`_sbss`/`_ebss`/
/// `_estack` symbols the reset handler uses.
pub fn linker_script(layout: &MemoryLayout) -> String {
    format!(
        "/* LatticeFoundry Cortex-M memory map */
ENTRY(Reset_Handler)
MEMORY
{{
  FLASH (rx)  : ORIGIN = {:#x}, LENGTH = {:#x}
  RAM   (rwx) : ORIGIN = {:#x}, LENGTH = {:#x}
}}
_estack = ORIGIN(RAM) + LENGTH(RAM);
SECTIONS
{{
  .isr_vector : {{ KEEP(*(.isr_vector)) }} > FLASH
  .text : {{ *(.text .text.*) *(.rodata .rodata.*) . = ALIGN(4); }} > FLASH
  .ARM.exidx : {{ *(.ARM.exidx*) }} > FLASH
  _sidata = LOADADDR(.data);
  .data : {{ . = ALIGN(4); _sdata = .; *(.data .data.*) . = ALIGN(4); _edata = .; }} > RAM AT> FLASH
  .bss (NOLOAD) : {{ . = ALIGN(4); _sbss = .; *(.bss .bss.* COMMON) . = ALIGN(4); _ebss = .; }} > RAM
}}
",
        layout.flash_origin, layout.flash_size, layout.ram_origin, layout.ram_size
    )
}

/// Link `objects` (Thumb objects, the startup object among them) into the ELF
/// executable `output` with `qld` under [`linker_script`]`(layout)`. `extra`
/// is passed on to the linker after the objects (e.g. `-L<dir>`, `-lgcc` for
/// the run-time ABI helpers). The objects and the script are staged next to
/// `output` and removed afterwards.
///
/// # Errors
///
/// An object that cannot be written as Arm ELF, an I/O error, or the link's
/// failure (whose diagnostics `qld` printed to standard error).
pub fn link_elf(objects: &[ObjectModule], layout: &MemoryLayout, extra: &[String], output: &Path) -> Result<(), String> {
    let stem = output.file_name().map_or_else(|| "a".to_owned(), |s| s.to_string_lossy().into_owned());
    let dir = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let tag = format!("{}.lf-{}", stem, std::process::id());
    let mut staged = Vec::new();
    let script = dir.join(format!("{tag}.ld"));
    let result = (|| {
        std::fs::write(&script, linker_script(layout)).map_err(|e| format!("cannot write {}: {e}", script.display()))?;
        staged.push(script.clone());
        let mut args: Vec<String> = vec!["-m".into(), "armelf".into(), "-T".into(), script.display().to_string()];
        args.extend(["-o".into(), output.display().to_string()]);
        for (k, obj) in objects.iter().enumerate() {
            let bytes = crate::mc::elf::write_with(obj, &crate::mc::elf::ElfTarget::ARM).map_err(|e| e.to_string())?;
            let path = dir.join(format!("{tag}.{k}.o"));
            std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            args.push(path.display().to_string());
            staged.push(path);
        }
        args.extend(extra.iter().cloned());
        crate::link::gnu::link_gnu("lf", &args)
    })();
    for p in staged {
        let _ = std::fs::remove_file(p);
    }
    result
}

/// The entry point (`e_entry`, with the Thumb bit) of a little-endian ELF32
/// executable.
pub fn elf32_entry(elf: &[u8]) -> Option<u64> {
    (elf.get(4) == Some(&1)).then(|| elf.get(24..28).map(|b| u64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))))?
}
