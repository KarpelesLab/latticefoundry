//! The per-target **data layout**: how IR types map onto bytes in memory.
//!
//! A [`DataLayout`] records everything the target-independent layers need to
//! know about a machine's memory model without knowing its instructions:
//!
//! - the byte order ([`Endian`]);
//! - per **address space** the pointer size and alignment ([`PointerSpec`]);
//!   address space `0` is the default data space and is always present, others
//!   are added by a target that has them (AVR program memory, for example);
//! - the ABI alignment of integers by store width (`i8`/`i16`/`i32`/`i64`, and
//!   any wider entry a target declares) and of `f16`/`f32`/`f64`;
//! - the stack alignment;
//! - the **native integer widths** the target computes with directly (the input
//!   to wide-integer legalization);
//! - the **program address space** functions live in (their references are
//!   pointers into it).
//!
//! The layout lives in the module's [`TypeContext`](crate::ir::TypeContext), so
//! every size/alignment query ([`TypeContext::layout`](crate::ir::TypeContext::layout),
//! `size_of`, `align_of`, `stride`, `field_offset`) and every consumer of them
//! (the builder's addressing helpers, the verifier, the reference evaluator,
//! global-data emission) follows it. The default is [`DataLayout::lp64`], the
//! little-endian 64-bit layout every existing backend (x86-64, AArch64, RISC-V
//! 64) uses, so a module that never sets a layout behaves exactly as before.
//!
//! # Spec strings
//!
//! A layout has a compact textual form, used by the `.lf` `datalayout "…"`
//! declaration and the `.lfb` header. It is a `-`-separated list of items, every
//! number in **bits**:
//!
//! ```text
//! spec  ::= item { "-" item }
//! item  ::= "e" | "E"                    little- / big-endian
//!         | "p" [ AS ] ":" SIZE ":" ALIGN pointer of address space AS (default 0)
//!         | "i" WIDTH ":" ALIGN           alignment of an integer of store width WIDTH
//!         | "f" WIDTH ":" ALIGN           alignment of f16 / f32 / f64
//!         | "S" ALIGN                     stack alignment
//!         | "n" WIDTH { ":" WIDTH }       native integer widths
//!         | "P" AS                        program (function) address space
//! ```
//!
//! Items not mentioned keep their [`lp64`](DataLayout::lp64) value, so
//! `"p:32:32-i64:32"` is "LP64, but with 32-bit pointers and 4-byte-aligned
//! `i64`". [`DataLayout::to_spec`] prints the canonical form (every item, in the
//! order above) and [`DataLayout::parse`] reads any valid spec, so
//! `parse(to_spec(dl)) == dl`. For example the canonical LP64 spec is
//! `e-p:64:64-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-S128-n8:16:32:64`.
//!
//! # Integer alignment rule
//!
//! An `iN` occupies its store size `ceil(N / 8)` bytes. Its alignment is looked
//! up by that size rounded up to a power of two, in bits: the entry of exactly
//! that width if there is one, else the smallest wider entry, else (wider than
//! every entry) the widest entry's alignment. Under LP64 this gives `i1`/`i8` →
//! 1, `i24` → 4, `i40` → 8 and `i128` → 8, which is the rule the IR has always
//! used.

use std::fmt;

/// Byte order of multi-byte values in memory.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Endian {
    /// Least-significant byte first (every current target).
    #[default]
    Little,
    /// Most-significant byte first.
    Big,
}

/// The size and alignment of a pointer in one address space.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PointerSpec {
    /// Width of the pointer in bits (a multiple of 8, from 8 to 64).
    pub bits: u32,
    /// ABI alignment in bytes (a power of two).
    pub align: u64,
}

impl PointerSpec {
    /// The size of the pointer in bytes.
    #[inline]
    pub fn bytes(self) -> u64 {
        u64::from(self.bits / 8)
    }
}

/// A target's data layout (see the [module docs](self)).
///
/// Build one from a preset ([`DataLayout::lp64`], [`DataLayout::ilp32`]), a
/// spec string ([`DataLayout::parse`]), or a preset refined with the `with_*`
/// builders. Every constructor keeps the invariants: address space 0 has a
/// pointer, widths and alignments are valid, tables are sorted.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct DataLayout {
    endian: Endian,
    /// `(address space, spec)`, sorted by address space; space 0 is present.
    pointers: Vec<(u32, PointerSpec)>,
    /// `(store width in bits, alignment in bytes)`, sorted by width; non-empty.
    ints: Vec<(u32, u64)>,
    /// Alignment in bytes of `f16`, `f32`, `f64`.
    floats: [u64; 3],
    /// Stack alignment in bytes.
    stack_align: u64,
    /// Native integer widths in bits, sorted and deduplicated; non-empty.
    native_ints: Vec<u32>,
    /// The address space functions live in.
    program_addr_space: u32,
}

impl Default for DataLayout {
    fn default() -> Self {
        DataLayout::lp64()
    }
}

/// Why a data-layout spec string or builder argument was rejected.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DataLayoutError(pub String);

impl fmt::Display for DataLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid data layout: {}", self.0)
    }
}

impl std::error::Error for DataLayoutError {}

fn bad<T>(msg: impl Into<String>) -> Result<T, DataLayoutError> {
    Err(DataLayoutError(msg.into()))
}

/// Validate an alignment given in bits and convert it to bytes.
fn align_bits_to_bytes(bits: u64, what: &str) -> Result<u64, DataLayoutError> {
    if bits < 8 || !bits.is_multiple_of(8) || !(bits / 8).is_power_of_two() {
        return bad(format!("{what} alignment {bits} is not a power-of-two number of bytes (in bits)"));
    }
    Ok(bits / 8)
}

impl DataLayout {
    /// The LP64 little-endian layout of the 64-bit targets (x86-64 System V,
    /// AArch64, RISC-V 64): 64-bit pointers, natural alignment of every scalar
    /// up to 8 bytes, a 16-byte stack, native `i8`..`i64`. This is the default.
    pub fn lp64() -> DataLayout {
        DataLayout {
            endian: Endian::Little,
            pointers: vec![(0, PointerSpec { bits: 64, align: 8 })],
            ints: vec![(8, 1), (16, 2), (32, 4), (64, 8)],
            floats: [2, 4, 8],
            stack_align: 16,
            native_ints: vec![8, 16, 32, 64],
            program_addr_space: 0,
        }
    }

    /// A generic ILP32 little-endian layout: 32-bit pointers, natural alignment
    /// up to 8 bytes (so `i64`/`f64` are 8-aligned, as on AAPCS and wasm32), an
    /// 8-byte stack, native `i8`..`i32`. A starting point that a 32-bit target
    /// refines with the `with_*` builders.
    pub fn ilp32() -> DataLayout {
        DataLayout {
            pointers: vec![(0, PointerSpec { bits: 32, align: 4 })],
            stack_align: 8,
            native_ints: vec![8, 16, 32],
            ..DataLayout::lp64()
        }
    }

    // --- queries ------------------------------------------------------------

    /// The byte order.
    #[inline]
    pub fn endian(&self) -> Endian {
        self.endian
    }

    /// The pointer spec of an address space, or `None` if the layout does not
    /// declare that space.
    pub fn pointer(&self, addr_space: u32) -> Option<PointerSpec> {
        self.pointers.iter().find(|(a, _)| *a == addr_space).map(|(_, p)| *p)
    }

    /// The pointer spec of an address space, falling back to address space 0's
    /// for an undeclared one (the verifier reports undeclared spaces; layout
    /// queries stay total).
    pub fn pointer_or_default(&self, addr_space: u32) -> PointerSpec {
        self.pointer(addr_space).unwrap_or(self.pointers[0].1)
    }

    /// The width in bits of a pointer into `addr_space` (see
    /// [`DataLayout::pointer_or_default`]).
    #[inline]
    pub fn pointer_bits(&self, addr_space: u32) -> u32 {
        self.pointer_or_default(addr_space).bits
    }

    /// Every declared address space with its pointer spec, ascending.
    pub fn pointers(&self) -> &[(u32, PointerSpec)] {
        &self.pointers
    }

    /// The ABI alignment in bytes of an integer of `bits` width (see the
    /// module docs for the lookup rule).
    pub fn int_align(&self, bits: u32) -> u64 {
        let size = u64::from(bits.div_ceil(8));
        if size == 0 {
            return 1;
        }
        let key = size.next_power_of_two().saturating_mul(8);
        for &(w, a) in &self.ints {
            if u64::from(w) >= key {
                return a;
            }
        }
        self.ints.last().map_or(1, |&(_, a)| a)
    }

    /// The integer alignment table: `(store width in bits, alignment in bytes)`.
    pub fn int_aligns(&self) -> &[(u32, u64)] {
        &self.ints
    }

    /// The ABI alignment in bytes of a float of `bits` width (16, 32 or 64).
    pub fn float_align(&self, bits: u32) -> u64 {
        match bits {
            16 => self.floats[0],
            32 => self.floats[1],
            _ => self.floats[2],
        }
    }

    /// The stack alignment in bytes.
    #[inline]
    pub fn stack_align(&self) -> u64 {
        self.stack_align
    }

    /// The integer widths the target computes with natively, ascending.
    pub fn native_ints(&self) -> &[u32] {
        &self.native_ints
    }

    /// Whether `bits` is a native integer width.
    pub fn is_native_int(&self, bits: u32) -> bool {
        self.native_ints.contains(&bits)
    }

    /// The widest native integer width: the machine's natural word, and the
    /// part width wide-integer legalization splits into.
    pub fn max_native_int(&self) -> u32 {
        *self.native_ints.last().expect("native_ints is never empty")
    }

    /// The address space functions live in (function references are pointers
    /// into it).
    #[inline]
    pub fn program_addr_space(&self) -> u32 {
        self.program_addr_space
    }

    // --- builders -----------------------------------------------------------

    /// Set the byte order.
    pub fn with_endian(mut self, endian: Endian) -> DataLayout {
        self.endian = endian;
        self
    }

    /// Declare (or redefine) the pointer of `addr_space`: `bits` wide (a
    /// multiple of 8 in `8..=64`) and `align` bytes aligned (a power of two).
    pub fn with_pointer(mut self, addr_space: u32, bits: u32, align: u64) -> Result<DataLayout, DataLayoutError> {
        if bits == 0 || !bits.is_multiple_of(8) || bits > 64 {
            return bad(format!("pointer width {bits} must be a multiple of 8 in 8..=64"));
        }
        if !align.is_power_of_two() {
            return bad(format!("pointer alignment {align} must be a power of two"));
        }
        let spec = PointerSpec { bits, align };
        match self.pointers.binary_search_by_key(&addr_space, |&(a, _)| a) {
            Ok(i) => self.pointers[i].1 = spec,
            Err(i) => self.pointers.insert(i, (addr_space, spec)),
        }
        Ok(self)
    }

    /// Set the alignment (bytes, a power of two) of integers of store width
    /// `bits` (a power of two, at least 8), adding a table entry if needed.
    pub fn with_int_align(mut self, bits: u32, align: u64) -> Result<DataLayout, DataLayoutError> {
        if bits < 8 || !bits.is_power_of_two() {
            return bad(format!("integer alignment entry width {bits} must be a power of two ≥ 8"));
        }
        if !align.is_power_of_two() {
            return bad(format!("integer alignment {align} must be a power of two"));
        }
        match self.ints.binary_search_by_key(&bits, |&(w, _)| w) {
            Ok(i) => self.ints[i].1 = align,
            Err(i) => self.ints.insert(i, (bits, align)),
        }
        Ok(self)
    }

    /// Set the alignment (bytes, a power of two) of the float format of `bits`
    /// width (16, 32 or 64).
    pub fn with_float_align(mut self, bits: u32, align: u64) -> Result<DataLayout, DataLayoutError> {
        let slot = match bits {
            16 => 0,
            32 => 1,
            64 => 2,
            _ => return bad(format!("no float format of width {bits}")),
        };
        if !align.is_power_of_two() {
            return bad(format!("float alignment {align} must be a power of two"));
        }
        self.floats[slot] = align;
        Ok(self)
    }

    /// Set the stack alignment (bytes, a power of two).
    pub fn with_stack_align(mut self, align: u64) -> Result<DataLayout, DataLayoutError> {
        if !align.is_power_of_two() {
            return bad(format!("stack alignment {align} must be a power of two"));
        }
        self.stack_align = align;
        Ok(self)
    }

    /// Set the native integer widths (non-empty, each nonzero).
    pub fn with_native_ints(mut self, widths: &[u32]) -> Result<DataLayout, DataLayoutError> {
        if widths.is_empty() || widths.contains(&0) {
            return bad("native integer widths must be a non-empty list of nonzero widths");
        }
        let mut w = widths.to_vec();
        w.sort_unstable();
        w.dedup();
        self.native_ints = w;
        Ok(self)
    }

    /// Set the program (function) address space. It must have a pointer.
    pub fn with_program_addr_space(mut self, addr_space: u32) -> Result<DataLayout, DataLayoutError> {
        if self.pointer(addr_space).is_none() {
            return bad(format!("program address space {addr_space} has no pointer spec"));
        }
        self.program_addr_space = addr_space;
        Ok(self)
    }

    // --- spec strings ---------------------------------------------------------

    /// Parse a spec string (grammar in the [module docs](self)); unmentioned
    /// items keep their LP64 values. The empty string is LP64.
    pub fn parse(spec: &str) -> Result<DataLayout, DataLayoutError> {
        let mut dl = DataLayout::lp64();
        // A `P` item may name a space declared later in the string.
        let mut program = None;
        if spec.is_empty() {
            return Ok(dl);
        }
        for item in spec.split('-') {
            let num = |s: &str| -> Result<u64, DataLayoutError> {
                s.parse::<u64>().map_err(|_| DataLayoutError(format!("bad number `{s}` in `{item}`")))
            };
            let num32 = |s: &str| -> Result<u32, DataLayoutError> {
                u32::try_from(num(s)?).map_err(|_| DataLayoutError(format!("number `{s}` too large in `{item}`")))
            };
            let mut chars = item.chars();
            let Some(head) = chars.next() else { return bad("empty item") };
            let rest = chars.as_str();
            match head {
                'e' | 'E' if rest.is_empty() => {
                    dl.endian = if head == 'e' { Endian::Little } else { Endian::Big };
                }
                'p' => {
                    let parts: Vec<&str> = rest.split(':').collect();
                    let [space, size, align] = parts[..] else {
                        return bad(format!("pointer item `{item}` must be p[AS]:SIZE:ALIGN"));
                    };
                    let space = if space.is_empty() { 0 } else { num32(space)? };
                    let align = align_bits_to_bytes(num(align)?, "pointer")?;
                    dl = dl.with_pointer(space, num32(size)?, align)?;
                }
                'i' | 'f' => {
                    let parts: Vec<&str> = rest.split(':').collect();
                    let [width, align] = parts[..] else {
                        return bad(format!("alignment item `{item}` must be {head}WIDTH:ALIGN"));
                    };
                    let align = align_bits_to_bytes(num(align)?, "scalar")?;
                    dl = if head == 'i' {
                        dl.with_int_align(num32(width)?, align)?
                    } else {
                        dl.with_float_align(num32(width)?, align)?
                    };
                }
                'S' => {
                    let align = align_bits_to_bytes(num(rest)?, "stack")?;
                    dl = dl.with_stack_align(align)?;
                }
                'n' => {
                    let widths = rest.split(':').map(num32).collect::<Result<Vec<_>, _>>()?;
                    dl = dl.with_native_ints(&widths)?;
                }
                'P' => program = Some(num32(rest)?),
                _ => return bad(format!("unknown item `{item}`")),
            }
        }
        if let Some(p) = program {
            dl = dl.with_program_addr_space(p)?;
        }
        Ok(dl)
    }

    /// The canonical spec string of this layout (every item, fixed order), which
    /// [`DataLayout::parse`] reads back to an equal layout.
    pub fn to_spec(&self) -> String {
        let mut items: Vec<String> = Vec::new();
        items.push(if self.endian == Endian::Little { "e".into() } else { "E".into() });
        for &(space, p) in &self.pointers {
            let space = if space == 0 { String::new() } else { space.to_string() };
            items.push(format!("p{space}:{}:{}", p.bits, p.align * 8));
        }
        for &(w, a) in &self.ints {
            items.push(format!("i{w}:{}", a * 8));
        }
        for (w, a) in [16, 32, 64].into_iter().zip(self.floats) {
            items.push(format!("f{w}:{}", a * 8));
        }
        items.push(format!("S{}", self.stack_align * 8));
        let native: Vec<String> = self.native_ints.iter().map(u32::to_string).collect();
        items.push(format!("n{}", native.join(":")));
        if self.program_addr_space != 0 {
            items.push(format!("P{}", self.program_addr_space));
        }
        items.join("-")
    }
}

impl fmt::Display for DataLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_spec())
    }
}

impl std::str::FromStr for DataLayout {
    type Err = DataLayoutError;

    fn from_str(s: &str) -> Result<DataLayout, DataLayoutError> {
        DataLayout::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lp64_spec_is_canonical_and_round_trips() {
        let dl = DataLayout::lp64();
        assert_eq!(dl.to_spec(), "e-p:64:64-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-S128-n8:16:32:64");
        assert_eq!(DataLayout::parse(&dl.to_spec()).unwrap(), dl);
        assert_eq!(DataLayout::parse("").unwrap(), dl);
        assert_eq!(DataLayout::default(), dl);
    }

    #[test]
    fn partial_specs_override_lp64() {
        let dl = DataLayout::parse("p:32:32-i64:32-n32").unwrap();
        assert_eq!(dl.pointer_bits(0), 32);
        assert_eq!(dl.int_align(64), 4);
        assert_eq!(dl.int_align(32), 4);
        assert_eq!(dl.max_native_int(), 32);
        assert_eq!(DataLayout::parse(&dl.to_spec()).unwrap(), dl);
    }

    #[test]
    fn avr_like_layout_with_two_address_spaces() {
        let spec = "e-p:16:8-p1:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-S8-n8-P1";
        let dl = DataLayout::parse(spec).unwrap();
        assert_eq!(dl.pointer(1), Some(PointerSpec { bits: 16, align: 1 }));
        assert_eq!(dl.pointer(2), None);
        assert_eq!(dl.pointer_bits(2), 16, "undeclared spaces fall back to space 0");
        assert_eq!(dl.program_addr_space(), 1);
        assert_eq!(dl.int_align(64), 1);
        assert_eq!(dl.float_align(64), 1);
        assert_eq!(dl.stack_align(), 1);
        assert_eq!(dl.native_ints(), &[8]);
        let back = DataLayout::parse(&dl.to_spec()).unwrap();
        assert_eq!(back, dl);
        assert!(dl.to_spec().ends_with("-P1"));
    }

    #[test]
    fn integer_alignment_rule() {
        let dl = DataLayout::lp64();
        for (bits, align) in [(0, 1), (1, 1), (8, 1), (9, 2), (24, 4), (33, 8), (64, 8), (128, 8), (1000, 8)] {
            assert_eq!(dl.int_align(bits), align, "i{bits}");
        }
        let dl = dl.with_int_align(128, 16).unwrap();
        assert_eq!(dl.int_align(128), 16);
        assert_eq!(dl.int_align(100), 16, "i100 stores in 13 bytes, rounded to 16");
    }

    #[test]
    fn big_endian_and_errors() {
        let dl = DataLayout::parse("E").unwrap();
        assert_eq!(dl.endian(), Endian::Big);
        for bad in ["x", "p:12:8", "p:16:12", "p:16", "i12:8", "f80:8", "S3", "n", "P3", "p:0:8", "e-"] {
            assert!(DataLayout::parse(bad).is_err(), "`{bad}` should be rejected");
        }
    }
}
