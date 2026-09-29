//! Target triples: which architecture, which operating system, and so which
//! object format and calling convention a compilation targets.
//!
//! A [`Triple`] pairs a [`TargetArch`] with a [`TargetOs`]. The OS decides two
//! things the architecture alone does not:
//!
//! - the **object format** ([`ObjectFormat`]): ELF on Linux and bare metal,
//!   PE/COFF on Windows, Mach-O on Darwin (macOS/iOS);
//! - the **calling convention** on x86-64: the Microsoft x64 convention
//!   ("Win64") on Windows, System V everywhere else. AArch64 uses AAPCS64 on
//!   all three (Apple's and Microsoft's variants differ from it only in
//!   variadic calls, which the AArch64 backend does not lower, and in
//!   reserving `x18`, which it never allocates).
//!
//! Mach-O also prefixes every C-level symbol name with an underscore; that is
//! the [Mach-O writer's](crate::mc::macho) job, so IR and
//! [`ObjectModule`](crate::mc::object::ObjectModule) names stay unprefixed.
//!
//! [`Triple::parse`] accepts the usual spellings, e.g. `x86_64-linux`,
//! `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`,
//! `x86_64-w64-mingw32`, `x86_64-apple-darwin`, `aarch64-apple-macos`,
//! `arm64-apple-darwin`, `aarch64-pc-windows-msvc`, `aarch64-none-elf`,
//! `riscv64-unknown-linux-gnu`, `thumbv7m-none-eabi` and `thumbv7em-none-eabi`:
//! the first component is the architecture, and the OS is recognized from any
//! later component. A bare `thumbv7m` means bare metal (a Cortex-M runs no
//! Linux); every other bare architecture means Linux.

use std::fmt;

use super::TargetArch;

/// The operating system (or its absence) a compilation targets.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TargetOs {
    /// Linux (ELF, System V ABI).
    Linux,
    /// Windows (PE/COFF, the Microsoft x64 calling convention on x86-64).
    Windows,
    /// Darwin: macOS, iOS and friends (Mach-O, System V on x86-64).
    Darwin,
    /// No operating system: bare-metal / firmware (ELF, System V ABI).
    None,
}

impl TargetOs {
    /// The canonical short name (`linux`, `windows`, `darwin`, `none`).
    pub fn name(self) -> &'static str {
        match self {
            TargetOs::Linux => "linux",
            TargetOs::Windows => "windows",
            TargetOs::Darwin => "darwin",
            TargetOs::None => "none",
        }
    }
}

/// A relocatable object file format.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ObjectFormat {
    /// ELF ([`crate::mc::elf`]).
    Elf,
    /// PE/COFF ([`crate::mc::coff`]).
    Coff,
    /// Mach-O ([`crate::mc::macho`]).
    MachO,
}

impl ObjectFormat {
    /// The canonical short name (`elf`, `coff`, `macho`).
    pub fn name(self) -> &'static str {
        match self {
            ObjectFormat::Elf => "elf",
            ObjectFormat::Coff => "coff",
            ObjectFormat::MachO => "macho",
        }
    }

    /// Parse a format name: `elf`, `coff` (or `pe`), `macho` (or `mach-o`).
    pub fn parse(s: &str) -> Option<ObjectFormat> {
        match s.to_ascii_lowercase().as_str() {
            "elf" => Some(ObjectFormat::Elf),
            "coff" | "pe" | "pe-coff" => Some(ObjectFormat::Coff),
            "macho" | "mach-o" => Some(ObjectFormat::MachO),
            _ => None,
        }
    }
}

/// The calling convention a backend follows for every function it compiles.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CallConvKind {
    /// The System V AMD64 psABI (x86-64 Linux, macOS, bare metal).
    SysV,
    /// The Microsoft x64 calling convention (x86-64 Windows).
    Win64,
    /// The Arm AAPCS64 (AArch64 on every OS).
    Aapcs64,
    /// The RISC-V LP64 integer calling convention.
    RiscvLp64,
    /// The 32-bit Arm AAPCS, base standard (soft-float: floating-point values
    /// in core registers), as Cortex-M code uses it.
    Aapcs,
}

/// An architecture plus an operating system.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Triple {
    /// The instruction set.
    pub arch: TargetArch,
    /// The operating system (or [`TargetOs::None`] for bare metal).
    pub os: TargetOs,
}

impl Triple {
    /// A triple from its parts.
    pub const fn new(arch: TargetArch, os: TargetOs) -> Triple {
        Triple { arch, os }
    }

    /// The host-independent default: `x86_64-linux`, the target `lf build`
    /// has always produced.
    pub const fn default_target() -> Triple {
        Triple::new(TargetArch::X86_64, TargetOs::Linux)
    }

    /// Parse a triple such as `x86_64-pc-windows-msvc` (see the
    /// [module docs](self) for the accepted spellings). Returns `None` for an
    /// unknown architecture or OS.
    pub fn parse(s: &str) -> Option<Triple> {
        let lower = s.to_ascii_lowercase();
        let mut parts = lower.split('-');
        let arch = match parts.next()? {
            "x86_64" | "x86-64" | "amd64" | "x64" => TargetArch::X86_64,
            "aarch64" | "arm64" => TargetArch::AArch64,
            "riscv64" | "riscv64gc" | "riscv64imac" => TargetArch::Riscv64,
            // ARMv7-M (Cortex-M3) and ARMv7E-M (Cortex-M4/M7, whose DSP and
            // FPU extensions the backend does not use) share the backend.
            "thumbv7m" | "thumbv7em" | "thumb" | "thumbv7" => TargetArch::Thumb,
            _ => return None,
        };
        let rest: Vec<&str> = parts.collect();
        let mut os = None;
        for part in &rest {
            let found = match *part {
                "linux" => Some(TargetOs::Linux),
                "windows" | "win32" | "mingw32" | "w64" | "msvc" => Some(TargetOs::Windows),
                p if p.starts_with("darwin")
                    || p.starts_with("macos")
                    || p.starts_with("macosx")
                    || p.starts_with("ios")
                    || p == "apple" =>
                {
                    Some(TargetOs::Darwin)
                }
                "none" | "elf" | "eabi" => Some(TargetOs::None),
                _ => None,
            };
            // The most specific word wins over a vendor hint (`apple`, `w64`)
            // and the ABI suffix (`elf`, `msvc`): keep scanning, but never let
            // a later generic word replace an OS already named.
            if let Some(f) = found
                && matches!(os, None | Some(TargetOs::None))
            {
                os = Some(f);
            }
        }
        // A bare architecture means the default OS (Linux), like `lf build`
        // with no `--target`; any other unrecognized OS is an error.
        let os = match os {
            Some(os) => os,
            None if rest.is_empty() && arch == TargetArch::Thumb => TargetOs::None,
            None if rest.is_empty() => TargetOs::Linux,
            None => return None,
        };
        Some(Triple::new(arch, os))
    }

    /// The object format this OS uses.
    pub fn object_format(self) -> ObjectFormat {
        match self.os {
            TargetOs::Linux | TargetOs::None => ObjectFormat::Elf,
            TargetOs::Windows => ObjectFormat::Coff,
            TargetOs::Darwin => ObjectFormat::MachO,
        }
    }

    /// The calling convention a backend follows for this triple.
    pub fn call_conv(self) -> CallConvKind {
        match (self.arch, self.os) {
            (TargetArch::X86_64, TargetOs::Windows) => CallConvKind::Win64,
            (TargetArch::X86_64, _) => CallConvKind::SysV,
            (TargetArch::AArch64, _) => CallConvKind::Aapcs64,
            (TargetArch::Riscv64, _) => CallConvKind::RiscvLp64,
            (TargetArch::Thumb, _) => CallConvKind::Aapcs,
        }
    }

    /// The prefix the platform's C ABI puts in front of every symbol name
    /// (`_` on Darwin, nothing elsewhere; x86-64 Windows has none).
    pub fn symbol_prefix(self) -> &'static str {
        match self.os {
            TargetOs::Darwin => "_",
            _ => "",
        }
    }
}

impl Default for Triple {
    fn default() -> Triple {
        Triple::default_target()
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.arch.name(), self.os.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_spellings() {
        let cases = [
            ("x86_64", TargetArch::X86_64, TargetOs::Linux),
            ("x86_64-linux", TargetArch::X86_64, TargetOs::Linux),
            ("x86_64-unknown-linux-gnu", TargetArch::X86_64, TargetOs::Linux),
            ("x86_64-pc-windows-msvc", TargetArch::X86_64, TargetOs::Windows),
            ("x86_64-pc-windows-gnu", TargetArch::X86_64, TargetOs::Windows),
            ("x86_64-w64-mingw32", TargetArch::X86_64, TargetOs::Windows),
            ("x86_64-windows", TargetArch::X86_64, TargetOs::Windows),
            ("x86_64-apple-darwin", TargetArch::X86_64, TargetOs::Darwin),
            ("x86_64-apple-macosx13.0", TargetArch::X86_64, TargetOs::Darwin),
            ("arm64-apple-darwin", TargetArch::AArch64, TargetOs::Darwin),
            ("aarch64-apple-ios", TargetArch::AArch64, TargetOs::Darwin),
            ("aarch64-pc-windows-msvc", TargetArch::AArch64, TargetOs::Windows),
            ("aarch64-none-elf", TargetArch::AArch64, TargetOs::None),
            ("aarch64-unknown-none", TargetArch::AArch64, TargetOs::None),
            ("riscv64-unknown-linux-gnu", TargetArch::Riscv64, TargetOs::Linux),
            ("thumbv7m-none-eabi", TargetArch::Thumb, TargetOs::None),
            ("thumbv7em-none-eabi", TargetArch::Thumb, TargetOs::None),
            ("thumbv7m", TargetArch::Thumb, TargetOs::None),
        ];
        for (s, arch, os) in cases {
            assert_eq!(Triple::parse(s), Some(Triple::new(arch, os)), "{s}");
        }
        assert_eq!(Triple::parse("mips-linux"), None);
        assert_eq!(Triple::parse("x86_64-plan9"), None);
        assert_eq!(Triple::parse("thumbv6m-none-eabi"), None, "ARMv6-M is not supported");
        let t = Triple::parse("thumbv7m-none-eabi").unwrap();
        assert_eq!((t.call_conv(), t.object_format()), (CallConvKind::Aapcs, ObjectFormat::Elf));
    }

    #[test]
    fn os_selects_format_abi_and_prefix() {
        let win = Triple::parse("x86_64-windows").unwrap();
        assert_eq!(win.object_format(), ObjectFormat::Coff);
        assert_eq!(win.call_conv(), CallConvKind::Win64);
        assert_eq!(win.symbol_prefix(), "");
        let mac = Triple::parse("x86_64-apple-darwin").unwrap();
        assert_eq!(mac.object_format(), ObjectFormat::MachO);
        assert_eq!(mac.call_conv(), CallConvKind::SysV);
        assert_eq!(mac.symbol_prefix(), "_");
        let lin = Triple::default();
        assert_eq!(lin.object_format(), ObjectFormat::Elf);
        assert_eq!(lin.call_conv(), CallConvKind::SysV);
        assert_eq!(lin.to_string(), "x86_64-linux");
        assert_eq!(
            Triple::parse("aarch64-pc-windows-msvc").unwrap().call_conv(),
            CallConvKind::Aapcs64
        );
    }

    #[test]
    fn format_names() {
        for f in [ObjectFormat::Elf, ObjectFormat::Coff, ObjectFormat::MachO] {
            assert_eq!(ObjectFormat::parse(f.name()), Some(f));
        }
        assert_eq!(ObjectFormat::parse("pe"), Some(ObjectFormat::Coff));
        assert_eq!(ObjectFormat::parse("wasm"), None);
    }
}
