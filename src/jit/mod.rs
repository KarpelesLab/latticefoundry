//! In-process JIT execution of generated x86-64 code (ROADMAP Phase 10).
//!
//! This is the in-process analogue of the M5 native-executable path: instead of
//! writing a linked ELF file and running it as a separate process, we compile a
//! LatticeFoundry IR [`Module`](crate::ir::Module) to an [`ObjectModule`] with
//! the same x86-64 backend, lay its sections into an executable memory mapping,
//! resolve relocations against the mapping's real addresses, and call the
//! resulting machine code directly.
//!
//! # Pipeline
//!
//! 1. [`crate::target::x86_64::compile_module`] encodes the module to an
//!    [`ObjectModule`] (`.text`, plus any `.rodata`/`.data`).
//! 2. [`Jit::compile`] computes a contiguous, alignment-respecting layout for the
//!    sections, maps a writable [`ExecBuffer`] of that size, copies the section
//!    bytes in, and — now that every section's runtime address is known — applies
//!    each [`Relocation`](crate::mc::object::Relocation) in place with the same
//!    arithmetic the static linker uses (`Abs64 = S+A`, `Pc32`/`Plt32 = S+A-P`).
//! 3. The mapping is flipped to read-execute and wrapped in a [`CompiledModule`],
//!    which hands out callable typed handles by symbol name.
//!
//! # Scope and limitations
//!
//! - Targets the integer subset the x86-64 backend supports. A function that
//!   calls **another function defined in the same module** is fully resolved
//!   within the mapping (that call is a `Plt32` relocation against a local
//!   definition). A relocation to an **undefined external symbol** cannot be
//!   resolved and yields [`JitError::UnresolvedSymbol`].
//! - All sections share one `R-X` mapping. For the integer subset the backend
//!   emits only `.text`, so this is exactly the code. Were writable `.data` to
//!   appear it would be mapped read-only; that is a documented limitation, not a
//!   correctness hazard for the supported subset.
//!
//! # Safety
//!
//! All `unsafe` lives in [`exec_mem`] (memory mapping) and at the single
//! function-pointer call site in [`CompiledModule`]'s typed accessors. The one
//! obligation the caller must uphold is documented on each accessor: the chosen
//! Rust signature must match the IR function's SysV ABI signature.

pub mod exec_mem;

use crate::ir::Module;
use crate::mc::object::{ObjectModule, RelocKind, SectionKind, SymbolValue};
use crate::support::StrInterner;
use crate::support::hash::DetHashMap;
use exec_mem::ExecBuffer;
use std::io;

/// Why a JIT compilation failed.
#[derive(Debug)]
pub enum JitError {
    /// Mapping or protecting executable memory failed.
    Mmap(io::Error),
    /// A relocation targets a symbol with no definition in this module (an
    /// external reference the in-process JIT cannot bind).
    UnresolvedSymbol(String),
    /// A relocation kind this JIT does not implement (e.g. one needing a GOT).
    UnsupportedReloc(RelocKind),
    /// A PC-relative relocation value did not fit its 32-bit field.
    RelocOverflow {
        /// The name of the symbol whose address was being applied.
        symbol: String,
        /// The mapping address of the field being patched.
        at: usize,
    },
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Mmap(e) => write!(f, "executable memory error: {e}"),
            JitError::UnresolvedSymbol(n) => write!(f, "undefined reference to '{n}'"),
            JitError::UnsupportedReloc(k) => write!(f, "unsupported relocation for JIT: {k:?}"),
            JitError::RelocOverflow { symbol, at } => {
                write!(f, "relocation of '{symbol}' at {at:#x} does not fit its field")
            }
        }
    }
}

impl std::error::Error for JitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            JitError::Mmap(e) => Some(e),
            _ => None,
        }
    }
}

/// The in-process JIT entry point.
///
/// A zero-sized façade; all state lives in the [`CompiledModule`] it produces.
#[derive(Clone, Copy, Debug, Default)]
pub struct Jit;

impl Jit {
    /// Create a JIT.
    pub fn new() -> Jit {
        Jit
    }

    /// Compile `module` (whose function names are interned in `syms`) to machine
    /// code in executable memory.
    ///
    /// Only defined functions are compiled; declarations are skipped by the
    /// backend. Returns a [`CompiledModule`] owning the mapping, from which
    /// callable handles are obtained by symbol name.
    pub fn compile(self, module: &Module, syms: &StrInterner) -> Result<CompiledModule, JitError> {
        let obj = crate::target::x86_64::compile_module(module, syms);
        CompiledModule::from_object(&obj)
    }
}

/// A section placed in the mapping: its byte offset from the mapping base.
#[derive(Clone, Copy, Debug)]
struct Placement {
    offset: usize,
}

/// Round `x` up to a multiple of `align` (a power of two, or `0`/`1` for none).
fn align_up(x: usize, align: usize) -> usize {
    if align <= 1 {
        return x;
    }
    let mask = align - 1;
    (x + mask) & !mask
}

/// Machine code compiled into executable memory, plus a name→address table for
/// its functions. Dropping it unmaps the code.
#[derive(Debug)]
pub struct CompiledModule {
    buffer: ExecBuffer,
    /// Function symbol name → runtime code address.
    funcs: DetHashMap<String, usize>,
}

impl CompiledModule {
    /// Lay `obj`'s sections into a fresh executable mapping and resolve every
    /// relocation against the resulting in-memory addresses.
    fn from_object(obj: &ObjectModule) -> Result<CompiledModule, JitError> {
        // 1. Compute a contiguous, alignment-respecting layout of all sections.
        let sections = obj.sections();
        let mut placements: Vec<Placement> = Vec::with_capacity(sections.len());
        let mut cursor = 0usize;
        for s in sections {
            let off = align_up(cursor, s.align as usize);
            placements.push(Placement { offset: off });
            cursor = off + s.size() as usize;
        }
        let total = cursor.max(1);

        // 2. Map a writable buffer and copy each section's bytes in. `.bss` is
        //    left as the mapping's fresh zero fill.
        let mut buffer = ExecBuffer::new(total).map_err(JitError::Mmap)?;
        let base = buffer.base_addr();
        {
            let mem = buffer.as_mut_slice().expect("freshly mapped buffer is writable");
            for (s, p) in sections.iter().zip(&placements) {
                if !s.is_nobits() {
                    mem[p.offset..p.offset + s.bytes.len()].copy_from_slice(&s.bytes);
                }
            }
        }

        // 3. Build the runtime address of every *defined* symbol.
        let mut sym_addr: Vec<Option<usize>> = Vec::with_capacity(obj.symbols().len());
        for sym in obj.symbols() {
            let addr = match sym.value {
                SymbolValue::Defined { section, offset } => {
                    Some(base + placements[section.index()].offset + offset as usize)
                }
                SymbolValue::Undefined => None,
            };
            sym_addr.push(addr);
        }

        // 4. Apply relocations now that all addresses are known. This mirrors the
        //    static linker's arithmetic, but into the live mapping.
        {
            let mem = buffer.as_mut_slice().expect("buffer is still writable");
            for r in obj.relocations() {
                let sec_base = base + placements[r.section.index()].offset;
                let field_off = placements[r.section.index()].offset + r.offset as usize;
                let p = sec_base + r.offset as usize; // field runtime address
                let s = match sym_addr[r.symbol.index()] {
                    Some(a) => a,
                    None => {
                        return Err(JitError::UnresolvedSymbol(
                            obj.symbol(r.symbol).name.clone(),
                        ));
                    }
                };
                let a = r.addend;
                match r.kind {
                    RelocKind::Abs64 => {
                        let v = (s as u64).wrapping_add(a as u64);
                        mem[field_off..field_off + 8].copy_from_slice(&v.to_le_bytes());
                    }
                    RelocKind::Pc64 => {
                        let v = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                        mem[field_off..field_off + 8].copy_from_slice(&v.to_le_bytes());
                    }
                    RelocKind::Abs32 | RelocKind::Abs32S => {
                        let v = (s as i64).wrapping_add(a);
                        if v < i32::MIN as i64 || v > u32::MAX as i64 {
                            return Err(JitError::RelocOverflow {
                                symbol: obj.symbol(r.symbol).name.clone(),
                                at: p,
                            });
                        }
                        mem[field_off..field_off + 4].copy_from_slice(&(v as u32).to_le_bytes());
                    }
                    RelocKind::Pc32 | RelocKind::Plt32 => {
                        // A local PLT call resolves to a plain PC-relative ref.
                        let v = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                        if v < i32::MIN as i64 || v > i32::MAX as i64 {
                            return Err(JitError::RelocOverflow {
                                symbol: obj.symbol(r.symbol).name.clone(),
                                at: p,
                            });
                        }
                        mem[field_off..field_off + 4].copy_from_slice(&(v as i32).to_le_bytes());
                    }
                    RelocKind::GotPcRel => return Err(JitError::UnsupportedReloc(r.kind)),
                }
            }
        }

        // 5. Record function entry points, then flip the mapping to R-X.
        let mut funcs: DetHashMap<String, usize> = DetHashMap::default();
        for (i, sym) in obj.symbols().iter().enumerate() {
            let is_code = matches!(
                sym.value,
                SymbolValue::Defined { section, .. }
                    if sections[section.index()].kind == SectionKind::Text
            );
            if is_code && let Some(addr) = sym_addr[i] {
                funcs.insert(sym.name.clone(), addr);
            }
        }

        buffer.make_executable().map_err(JitError::Mmap)?;
        Ok(CompiledModule { buffer, funcs })
    }

    /// The runtime code address of the function named `name`, if present.
    pub fn func_addr(&self, name: &str) -> Option<usize> {
        self.funcs.get(name).copied()
    }

    /// The base address of the executable mapping (for diagnostics/tests).
    #[inline]
    pub fn base_addr(&self) -> usize {
        self.buffer.base_addr()
    }

    /// A callable handle for a unary `extern "C" fn(i64) -> i64`.
    ///
    /// # Obligation
    /// The returned closure transmutes the function's code address to an
    /// `extern "C" fn(i64) -> i64`. The **one** invariant the caller must uphold
    /// is that the named IR function's signature is exactly that under the SysV
    /// ABI (one 64-bit integer parameter, one 64-bit integer result). A mismatch
    /// is undefined behavior, exactly as calling any function through a wrong C
    /// prototype would be. If the signature matches, calling is sound: the code
    /// is valid, the mapping is executable and outlives the returned closure.
    pub fn get_fn_i64_i64(&self, name: &str) -> Option<impl Fn(i64) -> i64 + '_> {
        let addr = self.func_addr(name)?;
        Some(move |x: i64| -> i64 {
            // SAFETY: `addr` is a live entry point in `self`'s R-X mapping, which
            // the borrowed `self` keeps alive for the closure's lifetime. The
            // documented signature-match obligation above makes the transmute and
            // the call type-correct under the SysV ABI.
            #[allow(unsafe_code)]
            unsafe {
                let f: extern "C" fn(i64) -> i64 = std::mem::transmute(addr as *const ());
                f(x)
            }
        })
    }

    /// A callable handle for a binary `extern "C" fn(i64, i64) -> i64`.
    ///
    /// # Obligation
    /// As [`get_fn_i64_i64`](Self::get_fn_i64_i64): the named IR function must
    /// have exactly this SysV signature (two 64-bit integer parameters, one
    /// 64-bit integer result). A mismatch is undefined behavior.
    pub fn get_fn_i64_i64_i64(&self, name: &str) -> Option<impl Fn(i64, i64) -> i64 + '_> {
        let addr = self.func_addr(name)?;
        Some(move |x: i64, y: i64| -> i64 {
            // SAFETY: as `get_fn_i64_i64`, for the two-argument signature.
            #[allow(unsafe_code)]
            unsafe {
                let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(addr as *const ());
                f(x, y)
            }
        })
    }

    /// A callable handle for a binary `extern "C" fn(i32, i32) -> i32`.
    ///
    /// # Obligation
    /// As above, for the 32-bit signature: the named IR function must take two
    /// `i32` parameters and return an `i32` under the SysV ABI. A mismatch is
    /// undefined behavior.
    pub fn get_fn_i32_i32_i32(&self, name: &str) -> Option<impl Fn(i32, i32) -> i32 + '_> {
        let addr = self.func_addr(name)?;
        Some(move |x: i32, y: i32| -> i32 {
            // SAFETY: as `get_fn_i64_i64`, for the two-argument 32-bit signature.
            #[allow(unsafe_code)]
            unsafe {
                let f: extern "C" fn(i32, i32) -> i32 = std::mem::transmute(addr as *const ());
                f(x, y)
            }
        })
    }
}

#[cfg(test)]
mod tests;
