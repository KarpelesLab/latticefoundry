//! Choosing an object-file writer: one entry point, [`write_object`], that
//! serializes an [`ObjectModule`] in the format a [`Triple`] calls for — ELF
//! ([`crate::mc::elf`]), PE/COFF ([`crate::mc::coff`]) or Mach-O
//! ([`crate::mc::macho`]) — plus the error the non-ELF writers return when a
//! module holds something their format cannot express. The wasm format holds
//! the relocatable wasm object the [wasm32 backend](crate::target::wasm32)
//! already wrote into its envelope.

use std::fmt;

use crate::mc::object::{ObjectModule, RelocKind};
use crate::target::{ObjectFormat, TargetArch, Triple};

/// Why an [`ObjectModule`] could not be written in the requested format:
/// typically a [`RelocKind`] the format (or the architecture within it) has
/// no relocation for, or an addend that does not fit the in-place field a
/// REL-style format keeps it in.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ObjectWriteError {
    message: String,
}

impl ObjectWriteError {
    /// An error with the given message.
    pub fn new(message: impl Into<String>) -> ObjectWriteError {
        ObjectWriteError { message: message.into() }
    }

    /// The error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ObjectWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ObjectWriteError {}

/// Serialize `obj` as a relocatable object for `triple` (its
/// [`object_format`](Triple::object_format) and architecture).
///
/// # Errors
///
/// See [`write_object_as`].
pub fn write_object(obj: &ObjectModule, triple: Triple) -> Result<Vec<u8>, ObjectWriteError> {
    write_object_as(obj, triple.arch, triple.object_format())
}

/// Serialize `obj` as a relocatable `format` object for `arch`.
///
/// # Errors
///
/// Returns an error when the format has no writer for `arch` (ELF is written
/// for x86-64, AArch64, 32-bit Arm Thumb, AVR and RISC-V 64; COFF and Mach-O for x86-64 and
/// AArch64; wasm for wasm32 only), or when the module holds a relocation the
/// format cannot express.
pub fn write_object_as(
    obj: &ObjectModule,
    arch: TargetArch,
    format: ObjectFormat,
) -> Result<Vec<u8>, ObjectWriteError> {
    match format {
        ObjectFormat::Elf if arch == TargetArch::Thumb => {
            crate::mc::elf::write_with(obj, &crate::mc::elf::ElfTarget::ARM)
                .map_err(|e| ObjectWriteError::new(format!("Arm ELF object: {e}")))
        }
        ObjectFormat::Elf if arch == TargetArch::AArch64 => {
            crate::mc::elf::write_with(obj, &crate::mc::elf::ElfTarget::AARCH64)
                .map_err(|e| ObjectWriteError::new(format!("AArch64 ELF object: {e}")))
        }
        ObjectFormat::Elf if arch == TargetArch::Avr => crate::target::avr::write_elf(obj)
            .map_err(|e| ObjectWriteError::new(format!("cannot write an AVR ELF object: {e}"))),
        ObjectFormat::Elf if arch == TargetArch::Riscv64 => crate::target::riscv::write_elf(obj)
            .map_err(|e| ObjectWriteError::new(format!("cannot write a RISC-V ELF object: {e}"))),
        ObjectFormat::Elf => {
            if arch != TargetArch::X86_64 {
                return Err(ObjectWriteError::new(format!(
                    "no ELF object writer for {} yet",
                    arch.name()
                )));
            }
            if let Some(r) = obj.relocations().iter().find(|r| {
                r.kind.is_instruction_field()
                    || r.kind.is_avr()
                    || r.kind.is_riscv()
                    || r.kind == RelocKind::ImageRel32
            }) {
                return Err(ObjectWriteError::new(format!(
                    "relocation {:?} cannot appear in an x86-64 ELF object",
                    r.kind
                )));
            }
            Ok(crate::mc::elf::write(obj))
        }
        ObjectFormat::Coff => {
            let machine = match arch {
                TargetArch::X86_64 => crate::mc::coff::CoffMachine::Amd64,
                TargetArch::AArch64 => crate::mc::coff::CoffMachine::Arm64,
                other => {
                    return Err(ObjectWriteError::new(format!(
                        "no COFF object writer for {}",
                        other.name()
                    )));
                }
            };
            crate::mc::coff::write(obj, machine)
        }
        ObjectFormat::MachO => {
            let cpu = match arch {
                TargetArch::X86_64 => crate::mc::macho::MachOCpu::X86_64,
                TargetArch::AArch64 => crate::mc::macho::MachOCpu::Arm64,
                other => {
                    return Err(ObjectWriteError::new(format!(
                        "no Mach-O object writer for {}",
                        other.name()
                    )));
                }
            };
            crate::mc::macho::write(obj, cpu)
        }
        ObjectFormat::Wasm => match (arch, crate::target::wasm32::envelope_bytes(obj)) {
            (TargetArch::Wasm32, Some(bytes)) => Ok(bytes.to_vec()),
            (TargetArch::Wasm32, None) => {
                Err(ObjectWriteError::new("not a wasm32 compilation (no wasm object in the module)"))
            }
            (other, _) => Err(ObjectWriteError::new(format!("no wasm object writer for {}", other.name()))),
        },
    }
}

/// Read a little-endian `u32` instruction word at `at` (the AArch64 REL-style
/// writers patch addends into instruction fields).
pub(crate) fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Write a little-endian `u32` at `at`.
pub(crate) fn write_u32(bytes: &mut [u8], at: usize, v: u32) {
    bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// Round `v` up to a multiple of `align` (a power of two; 0 and 1 mean none).
pub(crate) fn align_up(v: u64, align: u64) -> u64 {
    if align <= 1 { v } else { v.div_ceil(align) * align }
}

/// `log2` of an alignment, rounding a non-power-of-two up to the next power.
pub(crate) fn log2_align(align: u64) -> u32 {
    align.max(1).next_power_of_two().trailing_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mc::object::{RelocKind, Relocation, Section, SectionKind, Symbol, SymbolBinding, SymbolType};
    use crate::target::TargetOs;

    fn tiny() -> ObjectModule {
        let mut m = ObjectModule::new("t");
        let t = m.add_section(Section::new(".text", SectionKind::Text, 16));
        m.section_mut(t).bytes = vec![0xc3];
        m.add_symbol(Symbol::defined("f", SymbolBinding::Global, SymbolType::Func, t, 0, 1));
        m
    }

    #[test]
    fn dispatch_picks_the_format_magic() {
        let m = tiny();
        let elf = write_object(&m, Triple::new(TargetArch::X86_64, TargetOs::Linux)).unwrap();
        assert_eq!(&elf[..4], b"\x7fELF");
        let coff = write_object(&m, Triple::new(TargetArch::X86_64, TargetOs::Windows)).unwrap();
        assert_eq!(&coff[..2], &0x8664u16.to_le_bytes());
        let coff = write_object(&m, Triple::new(TargetArch::AArch64, TargetOs::Windows)).unwrap();
        assert_eq!(&coff[..2], &0xaa64u16.to_le_bytes());
        let macho = write_object(&m, Triple::new(TargetArch::X86_64, TargetOs::Darwin)).unwrap();
        assert_eq!(&macho[..4], &0xfeed_facfu32.to_le_bytes());
        // AArch64 Linux: ELF64 with e_machine = EM_AARCH64 (183).
        let elf = write_object(&m, Triple::new(TargetArch::AArch64, TargetOs::Linux)).unwrap();
        assert_eq!((&elf[..4], elf[4], u16::from_le_bytes([elf[18], elf[19]])), (&b"\x7fELF"[..], 2, 183));
    }

    #[test]
    fn unsupported_combinations_are_errors() {
        let m = tiny();
        let e = write_object(&m, Triple::new(TargetArch::Riscv64, TargetOs::Windows)).unwrap_err();
        assert!(e.message().contains("riscv64"), "{e}");
        let e = write_object_as(&m, TargetArch::Wasm32, ObjectFormat::Elf).unwrap_err();
        assert!(e.message().contains("ELF"), "{e}");

        let mut m = tiny();
        let s = m.reference_symbol("g");
        m.add_relocation(Relocation {
            section: crate::mc::object::SectionId::from_index(0),
            offset: 0,
            symbol: s,
            kind: RelocKind::Aarch64Call26,
            addend: 0,
        });
        assert!(write_object(&m, Triple::default()).is_err());
    }

    #[test]
    fn alignment_helpers() {
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(8, 4), 8);
        assert_eq!(align_up(5, 0), 5);
        assert_eq!(log2_align(1), 0);
        assert_eq!(log2_align(16), 4);
        assert_eq!(log2_align(12), 4);
    }
}
