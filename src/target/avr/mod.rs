//! The AVR backend: 8-bit AVR microcontrollers, the **AVR5** core first
//! (ATmega328P, the Arduino Uno's chip), producing ELF32 `EM_AVR` objects and
//! Intel HEX firmware.
//!
//! # Cores and instructions
//!
//! The baseline is AVR5 with at most 64 KiB of flash: the full register file
//! `r0`–`r31`, `movw`, `adiw`/`sbiw`, `mul` (8×8 → 16), `lpm Rd, Z(+)`,
//! and the 2-word `call`/`jmp`. Code never uses `elpm`, `eijmp`/`eicall` or
//! `RAMPZ` (devices with more than 64 KiB of flash), `des`, the XMEGA
//! read-modify-write instructions, or `spm`. A core without the multiplier
//! (AVR2/AVR25, most ATtinys) is supported with [`Device::has_mul`] off:
//! every multiply becomes a runtime call. The return address is 2 bytes (PC
//! ≤ 128 KiB).
//!
//! # Address spaces and pointers
//!
//! AVR is a Harvard machine. The data layout ([`data_layout`], spec
//! `e-p:16:8-p1:16:8-i16:8-i32:8-i64:8-f32:8-f64:8-S8-n8:16-P1`) declares:
//!
//! - **address space 0 = data memory** (registers, I/O, SRAM): 16-bit byte
//!   addresses, read with `ld`/`ldd` and written with `st`/`std` through the
//!   `X`/`Y`/`Z` pointer pairs. The stack, `alloca`, and ordinary globals
//!   (`.data`, `.rodata`, `.bss`) live here.
//! - **address space 1 = program memory** (flash), the layout's program space
//!   (`P1`): 16-bit pointers. A pointer to *data* in flash (a `global
//!   addrspace(1)`, placed in `.progmem.data`) is a **byte address**, read with
//!   `lpm` — [`Lower::mem_addr_space`](crate::codegen::isel::Lower::mem_addr_space)
//!   picks `lpm` over `ld` for a load through it. A pointer to a **function**
//!   is its **word address** (byte address / 2), what `icall`/`ijmp` take and
//!   what avr-gcc's function pointers hold, relocated with the `_PM`
//!   relocations. Arithmetic on a function pointer is therefore meaningless;
//!   nothing in the IR converts between the two kinds. A store through a
//!   space-1 pointer is rejected (flash is not writable by `st`).
//!
//! 16-bit pointers reach 64 KiB of either memory, which covers the AVR5
//! devices up to the ATmega644 for data and every 64 KiB-flash device for
//! code; larger flash would need `elpm` and 3-byte pointers. There is no
//! alignment requirement anywhere (every alignment in the layout is 1 byte).
//!
//! # Integers
//!
//! `i8` is native; the backend works on **register pairs** (see
//! [`regs`]): an `i8` in the low register of a pair, `i16` and
//! pointers in the whole pair (`movw`, `adiw`/`sbiw`, `add`/`adc`,
//! `cp`/`cpc`). Wider integers go through
//! [`legalize_ints`](crate::codegen::legalize_int::legalize_ints) with a part
//! width of 16 (the layout's native widths are `n8:16`); what stays wide at
//! the ABI boundary (parameters, call arguments and results, returns) lives in
//! several pairs. Narrow values follow the extension discipline described in
//! [`isel`].
//!
//! AVR has no divider and AVR5's `mul` is 8×8: an 8-bit multiply is one `mul`,
//! a 16-bit one three; wider multiplies and **every** division and remainder
//! are runtime calls — `__mulsi3`, `__udivdi3`, ... (the names
//! [`legalize_ints`](crate::codegen::legalize_int::legalize_ints) uses) for 32
//! and 64 bits, `__lf_udiv_i16`, `__lf_mod_i8`, ... for 8 and 16 bits (and
//! `__lf_mul_i8`/`_i16` without `mul`), each `T f(T, T)` under the normal
//! convention. [`runtime`] implements them all in LF IR.
//!
//! # Floating point
//!
//! Soft float only ([`softfloat`]): `f32` and `f64` are
//! their bit patterns and every operation is a libgcc-named call (`__addsf3`,
//! `__ltsf2`, `__fixsfsi`, ...). [`runtime`] implements the `f32` helpers
//! (round-to-nearest-even, IEEE 754 binary32 including subnormals, infinities
//! and NaN). Note that **avr-gcc's `double` is 32 bits** by default: a C front
//! end should map `double` to `f32` for avr-gcc compatibility. The IR's `f64`
//! is always IEEE binary64; it is lowered to the `…df…` helpers, which the LF
//! runtime does not provide (link a soft-float library that does). `f16` is not
//! supported.
//!
//! # Calls and frames
//!
//! The avr-gcc calling convention ([`regs`]); `Y` (`r29:r28`) is
//! the frame pointer, and the stack pointer `SPH:SPL` is updated with
//! interrupts masked ([`encode`]). Direct calls are `call` (`R_AVR_CALL`);
//! indirect ones `icall` through `Z`. Intra-function branches are relaxed
//! (`brXX` → `brXX` over `rjmp` → over `jmp`). There are **no stack probes**:
//! an AVR has no guard page, so a stack bound must come from the stack-usage
//! report ([`StackReport::worst_case_depth`](crate::codegen::StackReport::worst_case_depth)),
//! whose AVR frames count the 2-byte return address, the pushed registers, the
//! locals and the largest outgoing stack-argument area.
//!
//! Atomics of at most 16 bits mask interrupts around the access (`in r0,
//! SREG; cli; …; out SREG, r0`); a fence emits nothing (a single in-order
//! core). `syscall`, wider atomics, aggregate-typed loads and stores, and
//! interrupt-handler prologues (`reti`, saving `SREG`) are not supported.
//!
//! # Objects and firmware
//!
//! Objects are ELF32 `EM_AVR` (`e_flags` = the architecture, 5 for AVR5) with
//! `RELA` relocations ([`ELF`]):
//!
//! | kind | ELF | use |
//! |---|---|---|
//! | [`RelocKind::AvrCall`] | `R_AVR_CALL` (18) | `call`/`jmp` |
//! | [`RelocKind::Avr13Pcrel`] | `R_AVR_13_PCREL` (3) | `rcall`/`rjmp` (startup code) |
//! | [`RelocKind::Abs16`] | `R_AVR_16` (4) | a byte address in data |
//! | [`RelocKind::Avr16Pm`] | `R_AVR_16_PM` (5) | a function's word address in data |
//! | [`RelocKind::AvrLo8Ldi`] / [`RelocKind::AvrHi8Ldi`] | `R_AVR_LO8_LDI` (6) / `R_AVR_HI8_LDI` (7) | `ldi` of an address's bytes |
//! | [`RelocKind::AvrLo8LdiPm`] / [`RelocKind::AvrHi8LdiPm`] | `R_AVR_LO8_LDI_PM` (12) / `R_AVR_HI8_LDI_PM` (13) | `ldi` of a word address's bytes |
//!
//! [`link`] links objects into a flashable image with a Harvard layout — code
//! and `.progmem.data` in flash from address 0, the initial values of `.data`
//! and `.rodata` in flash after them, copied to SRAM at startup, `.bss` zeroed —
//! and [`startup`] supplies the vector table and reset code. `lf build
//! --target avr-atmega328p --oformat ihex` produces an Intel HEX image.
//!
//! [`RelocKind::AvrCall`]: crate::mc::object::RelocKind::AvrCall
//! [`RelocKind::Avr13Pcrel`]: crate::mc::object::RelocKind::Avr13Pcrel
//! [`RelocKind::Abs16`]: crate::mc::object::RelocKind::Abs16
//! [`RelocKind::Avr16Pm`]: crate::mc::object::RelocKind::Avr16Pm
//! [`RelocKind::AvrLo8Ldi`]: crate::mc::object::RelocKind::AvrLo8Ldi
//! [`RelocKind::AvrHi8Ldi`]: crate::mc::object::RelocKind::AvrHi8Ldi
//! [`RelocKind::AvrLo8LdiPm`]: crate::mc::object::RelocKind::AvrLo8LdiPm
//! [`RelocKind::AvrHi8LdiPm`]: crate::mc::object::RelocKind::AvrHi8LdiPm

pub mod encode;
pub mod isel;
pub mod link;
pub mod runtime;
pub mod startup;

pub(crate) mod data;
pub mod prepare;
pub mod regs;
pub mod softfloat;

#[cfg(test)]
mod interp;
#[cfg(test)]
mod tests;

pub use encode::{compile_module, compile_module_for_device, compile_module_with};
pub use isel::{AvrOp, AvrTarget};

use crate::ir::DataLayout;
use crate::mc::elf::{ElfClass, ElfTarget, RelocFormat};
use crate::mc::object::RelocKind;

/// An AVR device: what the backend, the startup code and the linker need to
/// know about one microcontroller.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Device {
    /// The device name (`atmega328p`).
    pub name: &'static str,
    /// Flash size in bytes.
    pub flash: u32,
    /// The first SRAM address (after the registers and I/O space).
    pub ram_start: u16,
    /// The last SRAM address (`RAMEND`), where the stack starts.
    pub ram_end: u16,
    /// Interrupt vectors, including reset (each a 2-word `jmp`).
    pub vectors: u32,
    /// Whether the core has the hardware multiplier.
    pub has_mul: bool,
    /// The ELF `e_flags` architecture number (5 for AVR5).
    pub arch: u32,
}

impl Device {
    /// The ATmega328P: AVR5, 32 KiB flash, 2 KiB SRAM at `0x100..=0x8ff`, 26
    /// vectors.
    pub const ATMEGA328P: Device =
        Device { name: "atmega328p", flash: 32 * 1024, ram_start: 0x100, ram_end: 0x8ff, vectors: 26, has_mul: true, arch: 5 };

    /// The ATtiny85: AVR25 (no multiplier), 8 KiB flash, 512 B SRAM at
    /// `0x60..=0x25f`, 15 vectors. (Its vectors are 1-word `rjmp`s on real
    /// hardware; the startup code here uses 2-word `jmp`, which AVR25 lacks, so
    /// this device serves compiling and testing the no-`mul` path, not flashing.)
    pub const ATTINY85_NOMUL: Device =
        Device { name: "attiny85", flash: 8 * 1024, ram_start: 0x60, ram_end: 0x25f, vectors: 15, has_mul: false, arch: 25 };

    /// The device named by a target triple's components (`avr-atmega328p`);
    /// a bare `avr` (or `avr-none`/`avr-elf`) is the ATmega328P.
    pub fn from_triple(triple: &str) -> Option<Device> {
        let lower = triple.to_ascii_lowercase();
        let mut parts = lower.split('-');
        if parts.next()? != "avr" {
            return None;
        }
        let mut dev = Device::ATMEGA328P;
        for p in parts {
            match p {
                "atmega328p" | "atmega328" | "m328p" => dev = Device::ATMEGA328P,
                "attiny85" => dev = Device::ATTINY85_NOMUL,
                "none" | "elf" | "unknown" | "" => {}
                _ => return None,
            }
        }
        Some(dev)
    }
}

/// The AVR data layout (see the [module docs](self)).
pub fn data_layout() -> DataLayout {
    DataLayout::parse("e-p:16:8-p1:16:8-i8:8-i16:8-i32:8-i64:8-f16:8-f32:8-f64:8-S8-n8:16-P1")
        .expect("the AVR layout spec is valid")
}

/// The ELF relocation number of a relocation kind on AVR.
pub fn elf_reloc_type(kind: RelocKind) -> Option<u32> {
    Some(match kind {
        RelocKind::Abs32 => 1,
        RelocKind::Avr13Pcrel => 3,
        RelocKind::Abs16 => 4,
        RelocKind::Avr16Pm => 5,
        RelocKind::AvrLo8Ldi => 6,
        RelocKind::AvrHi8Ldi => 7,
        RelocKind::AvrLo8LdiPm => 12,
        RelocKind::AvrHi8LdiPm => 13,
        RelocKind::AvrCall => 18,
        _ => return None,
    })
}

/// ELF32, little-endian, `EM_AVR` (83), `RELA`, `e_flags` 5 (AVR5).
pub const ELF: ElfTarget = ElfTarget {
    class: ElfClass::Elf32,
    endian: crate::ir::Endian::Little,
    machine: 83,
    flags: 5,
    reloc_format: RelocFormat::Rela,
    reloc_type: elf_reloc_type,
};

/// Serialize an AVR object as an ELF32 relocatable file.
///
/// # Errors
///
/// A relocation kind AVR has no number for, or a field too large for ELF32.
pub fn write_elf(obj: &crate::mc::object::ObjectModule) -> Result<Vec<u8>, crate::mc::elf::ElfError> {
    crate::mc::elf::write_with(obj, &ELF)
}
