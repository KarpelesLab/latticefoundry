//! **Undefined-behavior sanitizer**: runtime checks for the conditions under
//! which the IR's reference semantics make an operation undefined or poison
//! (ROADMAP Phase 10, "sanitizers").
//!
//! LatticeFoundry's opcode table already states, for every operation, exactly
//! when it is undefined behavior or yields poison ([`crate::ir::inst`],
//! [`crate::ir::semantics`]; tenet T2 / bet B1). This pass turns those
//! conditions into code: before each such operation it computes the condition
//! from the operation's own operands and, when it holds, calls a reporting
//! handler (or traps).
//!
//! # The checks
//!
//! | kind ([`UbKind`]) | before | condition (from the reference semantics) |
//! |---|---|---|
//! | `signed-integer-overflow` | `add`/`sub`/`mul nsw` | the exact signed result leaves `[-2ⁿ⁻¹, 2ⁿ⁻¹)` |
//! | `unsigned-integer-overflow` | `add`/`sub`/`mul nuw` | the exact unsigned result leaves `[0, 2ⁿ)` |
//! | `shift-exponent` | `shl`/`lshr`/`ashr` | the amount, read unsigned, is `≥ n` |
//! | `shift-base` | `shl nsw`/`nuw` | the shifted-out bits change the value (as `mul` by `2ᵏ`) |
//! | `integer-divide-by-zero` | `udiv`/`sdiv`/`urem`/`srem` | the divisor is zero |
//! | `division-overflow` | `sdiv`/`srem` | `INT_MIN / -1` |
//! | `exact` | `udiv`/`sdiv`/`lshr`/`ashr exact` | a nonzero remainder / a set bit shifted out |
//! | `float-cast-overflow` | `fptosi`/`fptoui` | NaN, or the truncated value is out of range |
//! | `pointer-bounds` | `ptr_add inbounds` | the result leaves its object (one past the end is inside) |
//! | `object-size` | loads, stores, atomics | the accessed bytes leave the object |
//! | `null` | loads, stores, atomics | the address is null |
//! | `alignment` | loads, stores, atomics | the address is not a multiple of the declared alignment |
//! | `unreachable` | `unreachable` | always |
//!
//! The two bounds checks need the object: the address must trace, through
//! `ptr_add`s, to an `alloca`, a `dyn_alloca` (whose size operand is used at
//! run time) or a global defined here with a known size; anything else is not
//! checked. Vector operations are not checked, and a `mul` wider than the
//! target's widest native integer is not either (its check would need a
//! division the target lacks). The tests check each condition against
//! [`crate::ir::semantics::eval`] exhaustively on `i8` operands.
//!
//! A check whose outcome is known at compile time is left out: an operation
//! whose operands are all constants and which folds to a non-poison value, a
//! constant shift amount below the width, a nonzero constant divisor, an
//! address that is a stack slot or global (never null), and an access at a
//! constant offset inside its object with its alignment implied by the
//! object's.
//!
//! # Shape of a check
//!
//! ```text
//!   %bad = <condition over the operation's operands>
//!   %f   = freeze %bad
//!   cond_br %f, ^report, ^cont
//! ^report:
//!   call @__lf_ub_report(i32 code, ptr @loc, i64 a, i64 b)
//!   br ^cont
//! ^cont:
//!   <the operation, without the flag its check now covers>
//! ```
//!
//! The condition is frozen, so a check never branches on poison and never adds
//! undefined behavior to a program: a correct program computes the same
//! results. The flag a check covers (`nsw`, `nuw`, `exact`, `inbounds`) is
//! dropped from the operation, which is a refinement, so a program that
//! continues after a report (recover mode) computes the wrapped value instead
//! of poison.
//!
//! # The handler ABI
//!
//! `void __lf_ub_report(i32 code, ptr loc, i64 a, i64 b)`:
//!
//! - `code` = `kind | detail << 8 | width << 16`: the [`UbKind::code`], a detail
//!   byte (the operator character `+ - * / >`, or `l`/`s`/`a` for a load, a
//!   store, an atomic access), and the bit width of the operation's type;
//! - `loc` points at a mutable record `{ ptr file, ptr function, i32 line, i32
//!   reported }` (file and function names NUL-terminated; line 0 when the
//!   module carries no line information); the runtime sets `reported` so each
//!   location reports once;
//! - `a`, `b`: the operands, sign- or zero-extended to 64 bits as the kind
//!   reads them (for the bounds kinds the offset and the object size, for
//!   `alignment` the address and the alignment).
//!
//! [`runtime`] provides a freestanding implementation in LF IR, linked into the
//! module (weakly, so a program may supply its own). It writes `file:line:
//! runtime error: <message>` to file descriptor 2 with the `write` syscall,
//! then continues or exits with status 1 ([`SanitizeOptions::recover`]); a
//! division by zero or overflow, a null access and an unreachable point always
//! exit, since continuing would crash or run off the end of code.
//!
//! **Trap mode** ([`SanitizeOptions::trap`], `--sanitize-trap`) needs no
//! runtime: the report block is a volatile store of the code to the weak global
//! `__lf_ub_trap_kind` (so a debugger or core dump shows the kind) followed by
//! `unreachable`, which every backend lowers to its trap instruction (`ud2` on
//! x86-64: `SIGILL`). The volatile store is an observable event, so no
//! optimization can remove the check in front of it.
//!
//! # Composition
//!
//! The pass is not part of any `-O` pipeline: it adds observable behavior.
//! Drivers run it on the verified module **before** optimizing, so flags such
//! as `nsw` are checked before the optimizer exploits them. Its output
//! verifies; the inserted code is ordinary IR the optimizer then cleans up.
//!
//! **Secrets.** A check branches on data, which the constant-time discipline
//! (`docs/ir-design.md` §6d) forbids for secret-derived values. The pass skips
//! every check whose condition would read a secret-derived value (per
//! [`SecretTaint`]), so a constant-time function stays constant-time after the
//! pass; its public operations are still checked.

pub mod runtime;

use puremp::Int;

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::analysis::secret::SecretTaint;
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, CastOp, Flags, FloatPred, InstId, InstKind, IntPred};
use crate::ir::semantics::{FoldResult, fold};
use crate::ir::types::{FloatKind, Type, TypeId};
use crate::ir::value::{AddrTarget, Const, FloatBits, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Global, GlobalAttrs, GlobalId, Linkage, Module};
use crate::support::{StrInterner, Sym};
use crate::target::{TargetArch, TargetOs};
use crate::transform::{dom_preorder, rebuild_terminator, remap_value};

/// One kind of runtime check. [`UbKind::code`] is the number the handler
/// receives in the low byte of its first argument.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum UbKind {
    /// `nsw` violated on `add`/`sub`/`mul`.
    SignedOverflow,
    /// `nuw` violated on `add`/`sub`/`mul`.
    UnsignedOverflow,
    /// A shift amount `≥` the bit width.
    ShiftExponent,
    /// `nsw`/`nuw` violated on `shl`.
    ShiftBase,
    /// Division or remainder by zero.
    DivByZero,
    /// `INT_MIN / -1` (or `%`).
    DivOverflow,
    /// `exact` violated on a division or right shift.
    Inexact,
    /// An out-of-range (or NaN) float-to-integer conversion.
    FloatCast,
    /// `ptr_add inbounds` leaving its object.
    PointerBounds,
    /// A memory access leaving its object.
    ObjectBounds,
    /// A memory access through a null pointer.
    NullPointer,
    /// A memory access below its declared alignment.
    Misaligned,
    /// An `unreachable` reached.
    Unreachable,
}

impl UbKind {
    /// Every kind, in [`code`](UbKind::code) order.
    pub const ALL: [UbKind; 13] = [
        UbKind::SignedOverflow,
        UbKind::UnsignedOverflow,
        UbKind::ShiftExponent,
        UbKind::ShiftBase,
        UbKind::DivByZero,
        UbKind::DivOverflow,
        UbKind::Inexact,
        UbKind::FloatCast,
        UbKind::PointerBounds,
        UbKind::ObjectBounds,
        UbKind::NullPointer,
        UbKind::Misaligned,
        UbKind::Unreachable,
    ];

    /// The kind's code (1-based; 0 is never a kind).
    pub fn code(self) -> u8 {
        UbKind::ALL.iter().position(|&k| k == self).expect("every kind is listed") as u8 + 1
    }

    /// The kind with the given [`code`](UbKind::code).
    pub fn from_code(code: u8) -> Option<UbKind> {
        UbKind::ALL.get(usize::from(code).checked_sub(1)?).copied()
    }

    /// The kind's option name (`-fsanitize=` spelling).
    pub fn name(self) -> &'static str {
        match self {
            UbKind::SignedOverflow => "signed-integer-overflow",
            UbKind::UnsignedOverflow => "unsigned-integer-overflow",
            UbKind::ShiftExponent => "shift-exponent",
            UbKind::ShiftBase => "shift-base",
            UbKind::DivByZero => "integer-divide-by-zero",
            UbKind::DivOverflow => "division-overflow",
            UbKind::Inexact => "exact",
            UbKind::FloatCast => "float-cast-overflow",
            UbKind::PointerBounds => "pointer-bounds",
            UbKind::ObjectBounds => "object-size",
            UbKind::NullPointer => "null",
            UbKind::Misaligned => "alignment",
            UbKind::Unreachable => "unreachable",
        }
    }

    /// Whether the runtime exits after reporting this kind even in recover
    /// mode: continuing would crash (a division fault, a null access) or run
    /// off the end of the code (`unreachable`).
    pub fn always_fatal(self) -> bool {
        matches!(self, UbKind::DivByZero | UbKind::DivOverflow | UbKind::NullPointer | UbKind::Unreachable)
    }

    /// The bit of this kind in a [`SanitizeKinds`] set.
    fn bit(self) -> u16 {
        1 << (self.code() - 1)
    }
}

/// A set of [`UbKind`]s.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct SanitizeKinds(u16);

impl SanitizeKinds {
    /// No kind.
    pub const NONE: SanitizeKinds = SanitizeKinds(0);
    /// Every kind (`undefined`).
    pub const ALL: SanitizeKinds = SanitizeKinds((1 << UbKind::ALL.len()) - 1);

    /// The set holding exactly `kinds`.
    pub fn of(kinds: &[UbKind]) -> SanitizeKinds {
        SanitizeKinds(kinds.iter().fold(0, |s, k| s | k.bit()))
    }

    /// Whether `kind` is in the set.
    pub fn contains(self, kind: UbKind) -> bool {
        self.0 & kind.bit() != 0
    }

    /// Whether the set is empty.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The union of two sets.
    pub fn union(self, other: SanitizeKinds) -> SanitizeKinds {
        SanitizeKinds(self.0 | other.0)
    }

    /// The kinds of `self` not in `other`.
    pub fn minus(self, other: SanitizeKinds) -> SanitizeKinds {
        SanitizeKinds(self.0 & !other.0)
    }

    /// The kinds of both sets.
    pub fn intersect(self, other: SanitizeKinds) -> SanitizeKinds {
        SanitizeKinds(self.0 & other.0)
    }

    /// The kinds in the set, in code order.
    pub fn iter(self) -> impl Iterator<Item = UbKind> {
        UbKind::ALL.into_iter().filter(move |k| self.contains(*k))
    }

    /// The kinds one option name stands for: a [kind name](UbKind::name) or
    /// a group — `undefined` (everything), `shift` (`shift-exponent` +
    /// `shift-base`), `signed-integer-overflow` (also `division-overflow`, as
    /// in GCC and Clang), `bounds` (`pointer-bounds` + `object-size`),
    /// `pointer-overflow` (`pointer-bounds`), `integer` (the integer kinds).
    pub fn from_name(name: &str) -> Option<SanitizeKinds> {
        use UbKind as K;
        let set = match name {
            "undefined" => SanitizeKinds::ALL,
            "shift" => SanitizeKinds::of(&[K::ShiftExponent, K::ShiftBase]),
            "signed-integer-overflow" => SanitizeKinds::of(&[K::SignedOverflow, K::DivOverflow]),
            "bounds" => SanitizeKinds::of(&[K::PointerBounds, K::ObjectBounds]),
            "pointer-overflow" => SanitizeKinds::of(&[K::PointerBounds]),
            "integer" => SanitizeKinds::of(&[
                K::SignedOverflow,
                K::UnsignedOverflow,
                K::ShiftExponent,
                K::ShiftBase,
                K::DivByZero,
                K::DivOverflow,
            ]),
            other => SanitizeKinds::of(&[*UbKind::ALL.iter().find(|k| k.name() == other)?]),
        };
        Some(set)
    }

    /// Parse a comma-separated list of [option names](SanitizeKinds::from_name).
    pub fn parse(list: &str) -> Result<SanitizeKinds, String> {
        let mut set = SanitizeKinds::NONE;
        for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            let k = SanitizeKinds::from_name(name).ok_or_else(|| format!("unknown sanitizer check '{name}'"))?;
            set = set.union(k);
        }
        Ok(set)
    }
}

/// The Linux system-call numbering the [`runtime`] uses.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LinuxSyscalls {
    /// x86-64 (`write` = 1, `exit_group` = 231).
    X86_64,
    /// The generic table of AArch64 and RISC-V (`write` = 64, `exit_group` =
    /// 94).
    Generic,
}

/// Where the reporting handler comes from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SanitizeRuntime {
    /// Link the freestanding LF IR runtime ([`runtime`]), which writes to
    /// stderr with Linux system calls of this numbering.
    Linux(LinuxSyscalls),
    /// Only declare `__lf_ub_report`: the program (or a library) supplies it.
    External,
}

impl SanitizeRuntime {
    /// The runtime for a target: the IR runtime on Linux x86-64, AArch64 and
    /// RISC-V, an external handler elsewhere (bare metal, Windows, Darwin,
    /// wasm32).
    pub fn for_target(arch: TargetArch, os: TargetOs) -> SanitizeRuntime {
        match (arch, os) {
            (TargetArch::X86_64, TargetOs::Linux) => SanitizeRuntime::Linux(LinuxSyscalls::X86_64),
            (TargetArch::AArch64 | TargetArch::Riscv64, TargetOs::Linux) => {
                SanitizeRuntime::Linux(LinuxSyscalls::Generic)
            }
            _ => SanitizeRuntime::External,
        }
    }
}

/// What [`sanitize_module`] checks and how a failed check is handled.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SanitizeOptions {
    /// The kinds checked.
    pub kinds: SanitizeKinds,
    /// The kinds that trap instead of calling the handler.
    pub trap: SanitizeKinds,
    /// Whether the runtime continues after a report (the default) rather than
    /// exiting with status 1. Some kinds always exit ([`UbKind::always_fatal`]).
    pub recover: bool,
    /// Where the handler comes from.
    pub runtime: SanitizeRuntime,
}

impl Default for SanitizeOptions {
    /// Every kind, reported by the x86-64 Linux runtime, recovering.
    fn default() -> Self {
        SanitizeOptions {
            kinds: SanitizeKinds::ALL,
            trap: SanitizeKinds::NONE,
            recover: true,
            runtime: SanitizeRuntime::Linux(LinuxSyscalls::X86_64),
        }
    }
}

impl SanitizeOptions {
    /// Check `kinds`, reporting through the runtime for `arch`/`os`.
    pub fn new(kinds: SanitizeKinds, arch: TargetArch, os: TargetOs) -> SanitizeOptions {
        SanitizeOptions { kinds, runtime: SanitizeRuntime::for_target(arch, os), ..SanitizeOptions::default() }
    }

    /// The same options with every checked kind trapping.
    pub fn trapping(self) -> SanitizeOptions {
        SanitizeOptions { trap: self.kinds, ..self }
    }

    /// Whether some checked kind reports through the handler (so the module
    /// needs one).
    pub fn reports(self) -> bool {
        !self.kinds.minus(self.trap).is_empty()
    }
}

/// What [`sanitize_module`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SanitizeStats {
    /// The number of checks inserted, per kind (indexed by `code - 1`).
    pub checks: [u32; UbKind::ALL.len()],
    /// Checks left out because their condition reads a secret-derived value.
    pub skipped_secret: u32,
    /// The number of functions instrumented.
    pub functions: u32,
}

impl SanitizeStats {
    /// The number of checks of `kind`.
    pub fn count(&self, kind: UbKind) -> u32 {
        self.checks[usize::from(kind.code() - 1)]
    }

    /// The total number of checks.
    pub fn total(&self) -> u32 {
        self.checks.iter().sum()
    }
}

/// The name of the reporting handler.
pub const REPORT_FN: &str = "__lf_ub_report";
/// The name of the global trap mode writes the failing code to.
pub const TRAP_KIND_GLOBAL: &str = "__lf_ub_trap_kind";
/// The prefix of every symbol the pass and its runtime create.
const PREFIX: &str = "__lf_ub_";

/// Instrument every function of `module` with the checks `opts` selects and,
/// when some kind reports through the handler and the runtime is
/// [`SanitizeRuntime::Linux`], link the [`runtime`] in. `file` names the
/// source in reports. `syms` must be the interner the module's names live in.
///
/// Functions whose name starts with `__lf_ub_` (the runtime) are not
/// instrumented.
pub fn sanitize_module(
    module: &mut Module,
    syms: &mut StrInterner,
    file: &str,
    opts: &SanitizeOptions,
) -> Result<SanitizeStats, String> {
    let mut stats = SanitizeStats::default();
    if opts.kinds.is_empty() {
        return Ok(stats);
    }

    // Plan (immutable): the checks of every instruction of every function.
    let mut plans: Vec<(FuncId, Vec<Vec<Check>>)> = Vec::new();
    let secrets = module.has_secrets();
    for fi in 0..module.function_count() {
        let fid = FuncId::from_index(fi);
        let f = module.function(fid);
        if f.is_declaration() || syms.resolve(f.name).starts_with(PREFIX) {
            continue;
        }
        let taint = secrets.then(|| SecretTaint::compute(module, fid));
        let checks = plan_function(module, fid, opts, taint.as_ref(), &mut stats);
        if checks.iter().any(|c| !c.is_empty()) {
            plans.push((fid, checks));
        }
    }
    if plans.is_empty() {
        return Ok(stats);
    }

    // Declare what the checks call and reference.
    let i32t = module.types_mut().int(32);
    let i64t = module.types_mut().int(64);
    let ptr = module.types_mut().ptr();
    let void = module.types_mut().void();
    let any_report = plans.iter().any(|(_, p)| p.iter().flatten().any(|c| !c.trap));
    let any_trap = plans.iter().any(|(_, p)| p.iter().flatten().any(|c| c.trap));
    let report_fn = any_report.then(|| {
        let name = syms.intern(REPORT_FN);
        find_function(module, name).unwrap_or_else(|| {
            let sig = module.types_mut().func(vec![i32t, ptr, i64t, i64t], void, false);
            module.declare_function(name, sig)
        })
    });
    let trap_global = any_trap.then(|| {
        let name = syms.intern(TRAP_KIND_GLOBAL);
        find_global(module, name).unwrap_or_else(|| {
            let zero = module.intern_const(Const::Int { ty: i32t, value: Int::ZERO });
            let attrs = GlobalAttrs { linkage: Linkage::Weak, ..GlobalAttrs::DEFAULT };
            module.define_global(Global { name, ty: i32t, init: Some(zero) }, attrs)
        })
    });
    if any_report {
        let mut locs = LocTable::new(module, syms, file);
        for (fid, checks) in &mut plans {
            let fname = syms.resolve(module.function(*fid).name).to_owned();
            for (ii, list) in checks.iter_mut().enumerate() {
                let line = module.function(*fid).inst_line(InstId::from_index(ii)).unwrap_or(0);
                for c in list.iter_mut().filter(|c| !c.trap) {
                    c.loc = Some(locs.get(module, syms, *fid, &fname, line, c.kind));
                }
            }
        }
    }

    let ctx = EmitCtx { report_fn, trap_global, i32t, i64t };
    for (fid, checks) in plans {
        for list in &checks {
            for c in list {
                stats.checks[usize::from(c.kind.code() - 1)] += 1;
            }
        }
        stats.functions += 1;
        let (fresh, ()) = module.map_function(fid, |old, b| rebuild(old, b, &checks, &ctx));
        module.replace_function(fid, fresh);
    }

    if any_report && let SanitizeRuntime::Linux(abi) = opts.runtime {
        runtime::link_runtime(module, syms, abi, opts.recover)?;
    }
    Ok(stats)
}

fn find_function(m: &Module, name: Sym) -> Option<FuncId> {
    m.functions().position(|f| f.name == name).map(FuncId::from_index)
}

fn find_global(m: &Module, name: Sym) -> Option<GlobalId> {
    m.globals().position(|g| g.name == name).map(GlobalId::from_index)
}

// ---------------------------------------------------------------------------
// Planning.
// ---------------------------------------------------------------------------

/// One planned check.
#[derive(Clone, Debug)]
struct Check {
    kind: UbKind,
    cond: Cond,
    /// The handler's `code` argument.
    code: u32,
    /// Whether the check traps instead of reporting.
    trap: bool,
    /// The location record (reporting checks; filled in after planning).
    loc: Option<GlobalId>,
}

/// How a check's condition is computed. Operation operands are the checked
/// instruction's; `ValueId`s are old-function values.
#[derive(Clone, Debug)]
enum Cond {
    /// `add`/`sub` overflow (signed: `nsw`, else `nuw`).
    AddSub { sub: bool, signed: bool },
    /// `mul` overflow, by widening to the given width or by dividing back.
    Mul { signed: bool, widen: Option<u32> },
    /// `shl` overflow.
    Shl { signed: bool },
    /// Shift amount `≥` width.
    ShiftAmount,
    /// Divisor zero.
    DivZero,
    /// `INT_MIN / -1`.
    DivOverflow,
    /// `exact` division leaving a remainder.
    InexactDiv { signed: bool },
    /// `exact` right shift dropping a set bit.
    InexactShr,
    /// Float-to-integer conversion out of range.
    FloatCast { signed: bool, width: u32, float: FloatKind },
    /// A null address.
    Null(ValueId),
    /// An address below `align`.
    Misaligned(ValueId, u32),
    /// An offset from an object leaving it (`access` bytes accessed, or `None`
    /// for an address computation, which may point one past the end).
    Bounds { offs: Vec<ValueId>, const_off: i128, size: ObjSize, access: Option<u64> },
    /// Always (an `unreachable`).
    Always,
}

/// The size of an object a bounds check measures against.
#[derive(Clone, Copy, Debug)]
enum ObjSize {
    /// Known at compile time.
    Static(u64),
    /// A `dyn_alloca`'s byte-count operand.
    Dynamic(ValueId),
}

/// An address traced back to its object.
struct Trace {
    offs: Vec<ValueId>,
    const_off: i128,
    size: ObjSize,
    align: u64,
}

fn int_width(m: &Module, ty: TypeId) -> Option<u32> {
    match m.types().get(ty) {
        Type::Int(w) => Some(*w),
        _ => None,
    }
}

fn is_pointer(m: &Module, ty: TypeId) -> bool {
    matches!(m.types().get(ty), Type::Ptr | Type::PtrIn(_))
}

/// The signed value of an integer constant operand.
fn const_signed(m: &Module, f: &Function, v: ValueId) -> Option<i128> {
    let ValueDef::Const(c) = f.value(v).def else { return None };
    let Const::Int { ty, value } = m.consts().get(c) else { return None };
    let w = int_width(m, *ty)?;
    let bits = value.mod_2k(w);
    let s = if w > 0 && bits.bit(w - 1) { bits.sub(&Int::ONE.mul_2k(w)) } else { bits };
    s.to_i128()
}

/// Whether a type ends (recursively) in a zero-length array: a flexible array
/// member, whose object may extend past the type's size.
fn has_flexible_tail(m: &Module, ty: TypeId) -> bool {
    match m.types().get(ty) {
        Type::Array(elem, len) => *len == 0 || has_flexible_tail(m, *elem),
        Type::Struct(fields) => fields.last().is_some_and(|&l| has_flexible_tail(m, l)),
        _ => false,
    }
}

/// Trace `v` through `ptr_add`s to the object it points into.
fn trace(m: &Module, f: &Function, mut v: ValueId) -> Option<Trace> {
    let mut offs = Vec::new();
    let mut const_off: i128 = 0;
    for _ in 0..256 {
        match f.value(v).def {
            ValueDef::Inst(i) => {
                let inst = f.inst(i);
                match &inst.kind {
                    InstKind::PtrAdd { .. } => {
                        let off = inst.operands()[1];
                        match const_signed(m, f, off) {
                            Some(c) => const_off = const_off.checked_add(c)?,
                            None => offs.push(off),
                        }
                        v = inst.operands()[0];
                    }
                    InstKind::Alloca { elem_ty } => {
                        let l = m.types().layout(*elem_ty);
                        if l.size == 0 || has_flexible_tail(m, *elem_ty) {
                            return None;
                        }
                        return Some(Trace { offs, const_off, size: ObjSize::Static(l.size), align: l.align });
                    }
                    InstKind::DynAlloca { align } => {
                        let n = inst.operands()[0];
                        let size = match const_signed(m, f, n) {
                            Some(c) => ObjSize::Static(u64::try_from(c).ok()?),
                            None => ObjSize::Dynamic(n),
                        };
                        return Some(Trace { offs, const_off, size, align: u64::from(*align) });
                    }
                    _ => return None,
                }
            }
            ValueDef::Global(g) => {
                let glob = m.global(g);
                let attrs = m.global_attrs(g);
                if glob.init.is_none() || attrs.linkage == Linkage::Weak || has_flexible_tail(m, glob.ty) {
                    return None;
                }
                let l = m.types().layout(glob.ty);
                if l.size == 0 {
                    return None;
                }
                return Some(Trace { offs, const_off, size: ObjSize::Static(l.size), align: l.align });
            }
            _ => return None,
        }
    }
    None
}

/// The `code` argument: kind, detail byte, width.
fn code_of(kind: UbKind, detail: u8, width: u32) -> u32 {
    u32::from(kind.code()) | u32::from(detail) << 8 | width.min(0xffff) << 16
}

/// Plan the checks of every instruction of `fid` (indexed by instruction id).
fn plan_function(
    m: &Module,
    fid: FuncId,
    opts: &SanitizeOptions,
    taint: Option<&SecretTaint>,
    stats: &mut SanitizeStats,
) -> Vec<Vec<Check>> {
    let f = m.function(fid);
    let mut out: Vec<Vec<Check>> = vec![Vec::new(); f.inst_count()];
    for (_, blk) in f.blocks() {
        for &i in blk.insts().iter().chain(blk.terminator().iter()) {
            let mut list = plan_inst(m, f, i, opts);
            if let Some(t) = taint {
                let before = list.len();
                let inst = f.inst(i);
                list.retain(|c| !reads(c, inst.operands()).iter().any(|&v| t.is_secret(v)));
                stats.skipped_secret += (before - list.len()) as u32;
            }
            for c in &mut list {
                c.trap = opts.trap.contains(c.kind);
            }
            out[i.index()] = list;
        }
    }
    out
}

/// The values a check's condition reads.
fn reads(c: &Check, ops: &[ValueId]) -> Vec<ValueId> {
    match &c.cond {
        Cond::Null(p) | Cond::Misaligned(p, _) => vec![*p],
        Cond::Bounds { offs, size, .. } => {
            let mut v = offs.clone();
            if let ObjSize::Dynamic(n) = size {
                v.push(*n);
            }
            v
        }
        Cond::Always => Vec::new(),
        _ => ops.to_vec(),
    }
}

/// Whether every operand of `inst` is a constant and the operation folds to a
/// non-poison value (so no check can fire).
fn folds_cleanly(m: &Module, f: &Function, i: InstId) -> bool {
    let inst = f.inst(i);
    let mut consts = Vec::with_capacity(inst.operands().len());
    for &o in inst.operands() {
        let ValueDef::Const(c) = f.value(o).def else { return false };
        consts.push(m.consts().get(c).clone());
    }
    matches!(
        fold(m.types(), inst.ty, &inst.kind, &inst.flags, &consts),
        Some(FoldResult::Folded(c)) if !matches!(c, Const::Poison(_))
    )
}

fn plan_inst(m: &Module, f: &Function, i: InstId, opts: &SanitizeOptions) -> Vec<Check> {
    let inst = f.inst(i);
    let k = opts.kinds;
    let mut out = Vec::new();
    let mut push = |kind: UbKind, cond: Cond, code: u32| {
        if k.contains(kind) {
            out.push(Check { kind, cond, code, trap: false, loc: None });
        }
    };
    let ops = inst.operands();
    match &inst.kind {
        InstKind::Bin(op) => {
            let Some(w) = int_width(m, inst.ty) else { return out };
            if folds_cleanly(m, f, i) {
                return out;
            }
            let fl = inst.flags;
            match op {
                BinOp::Add | BinOp::Sub => {
                    let sub = *op == BinOp::Sub;
                    let ch = if sub { b'-' } else { b'+' };
                    if fl.nsw {
                        push(UbKind::SignedOverflow, Cond::AddSub { sub, signed: true }, code_of(UbKind::SignedOverflow, ch, w));
                    }
                    if fl.nuw {
                        push(UbKind::UnsignedOverflow, Cond::AddSub { sub, signed: false }, code_of(UbKind::UnsignedOverflow, ch, w));
                    }
                }
                BinOp::Mul => {
                    let native = m.data_layout().max_native_int();
                    let widen = (2 * w <= native).then_some(2 * w);
                    if widen.is_some() || w <= native {
                        if fl.nsw {
                            push(UbKind::SignedOverflow, Cond::Mul { signed: true, widen }, code_of(UbKind::SignedOverflow, b'*', w));
                        }
                        if fl.nuw {
                            push(UbKind::UnsignedOverflow, Cond::Mul { signed: false, widen }, code_of(UbKind::UnsignedOverflow, b'*', w));
                        }
                    }
                }
                BinOp::Shl | BinOp::LShr | BinOp::AShr => {
                    // The amount is read unsigned: a negative constant is huge.
                    let amt_ok = const_signed(m, f, ops[1]).is_some_and(|c| (0..i128::from(w)).contains(&c));
                    if !amt_ok {
                        push(UbKind::ShiftExponent, Cond::ShiftAmount, code_of(UbKind::ShiftExponent, b'<', w));
                    }
                    if *op == BinOp::Shl {
                        if fl.nsw {
                            push(UbKind::ShiftBase, Cond::Shl { signed: true }, code_of(UbKind::ShiftBase, b's', w));
                        }
                        if fl.nuw {
                            push(UbKind::ShiftBase, Cond::Shl { signed: false }, code_of(UbKind::ShiftBase, b'u', w));
                        }
                    } else if fl.exact {
                        push(UbKind::Inexact, Cond::InexactShr, code_of(UbKind::Inexact, b'>', w));
                    }
                }
                BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => {
                    let signed = matches!(op, BinOp::SDiv | BinOp::SRem);
                    let d = const_signed(m, f, ops[1]);
                    let ch = if matches!(op, BinOp::URem | BinOp::SRem) { b'%' } else { b'/' };
                    if d.is_none_or(|d| d == 0) {
                        push(UbKind::DivByZero, Cond::DivZero, code_of(UbKind::DivByZero, ch, w));
                    }
                    if signed {
                        let min = match w {
                            0 => None,
                            1..=127 => Some(-(1i128 << (w - 1))),
                            128 => Some(i128::MIN),
                            _ => None,
                        };
                        let n = const_signed(m, f, ops[0]);
                        if d.is_none_or(|d| d == -1) && n.is_none_or(|n| Some(n) == min) {
                            push(UbKind::DivOverflow, Cond::DivOverflow, code_of(UbKind::DivOverflow, ch, w));
                        }
                    }
                    if fl.exact && matches!(op, BinOp::UDiv | BinOp::SDiv) {
                        push(UbKind::Inexact, Cond::InexactDiv { signed }, code_of(UbKind::Inexact, b'/', w));
                    }
                }
                _ => {}
            }
        }
        InstKind::Cast(op @ (CastOp::FpToSi | CastOp::FpToUi)) => {
            let (Some(w), Type::Float(fk)) = (int_width(m, inst.ty), m.types().get(f.value_type(ops[0]))) else {
                return out;
            };
            if folds_cleanly(m, f, i) {
                return out;
            }
            let signed = *op == CastOp::FpToSi;
            let cond = Cond::FloatCast { signed, width: w, float: *fk };
            push(UbKind::FloatCast, cond, code_of(UbKind::FloatCast, if signed { b's' } else { b'u' }, w));
        }
        InstKind::PtrAdd { inbounds: true } => {
            if let Some(t) = trace(m, f, ops[0]) {
                let mut offs = t.offs;
                let mut const_off = t.const_off;
                match const_signed(m, f, ops[1]) {
                    Some(c) => const_off += c,
                    None => offs.push(ops[1]),
                }
                let inside = offs.is_empty()
                    && matches!(t.size, ObjSize::Static(s) if const_off >= 0 && const_off <= i128::from(s));
                if !inside {
                    let cond = Cond::Bounds { offs, const_off, size: t.size, access: None };
                    push(UbKind::PointerBounds, cond, code_of(UbKind::PointerBounds, b'p', 64));
                }
            }
        }
        InstKind::Load { ty, align, .. }
        | InstKind::Store { ty, align, .. }
        | InstKind::AtomicLoad { ty, align, .. }
        | InstKind::AtomicStore { ty, align, .. }
        | InstKind::AtomicRmw { ty, align, .. }
        | InstKind::CmpXchg { ty, align, .. } => {
            let p = ops[0];
            if !is_pointer(m, f.value_type(p)) || matches!(m.types().get(*ty), Type::Struct(_) | Type::Array(..)) {
                return out;
            }
            let detail = match &inst.kind {
                InstKind::Load { .. } => b'l',
                InstKind::Store { .. } => b's',
                _ => b'a',
            };
            let access = m.types().layout(*ty).size;
            let tr = trace(m, f, p);
            if tr.is_none() && !matches!(f.value(p).def, ValueDef::Global(_)) {
                push(UbKind::NullPointer, Cond::Null(p), code_of(UbKind::NullPointer, detail, 0));
            }
            let al = u64::from(*align);
            if al > 1 {
                let implied = tr.as_ref().is_some_and(|t| t.offs.is_empty() && t.align >= al && t.const_off.rem_euclid(i128::from(al)) == 0);
                if !implied {
                    push(UbKind::Misaligned, Cond::Misaligned(p, *align), code_of(UbKind::Misaligned, detail, 0));
                }
            }
            if let Some(t) = tr {
                let inside = t.offs.is_empty()
                    && matches!(t.size, ObjSize::Static(s) if t.const_off >= 0 && t.const_off + i128::from(access) <= i128::from(s));
                if !inside {
                    let cond = Cond::Bounds { offs: t.offs, const_off: t.const_off, size: t.size, access: Some(access) };
                    push(UbKind::ObjectBounds, cond, code_of(UbKind::ObjectBounds, detail, 0));
                }
            }
        }
        InstKind::Unreachable => push(UbKind::Unreachable, Cond::Always, code_of(UbKind::Unreachable, 0, 0)),
        _ => {}
    }
    out
}

// ---------------------------------------------------------------------------
// Location records.
// ---------------------------------------------------------------------------

/// Creates (and dedups) the location records: one per (function, line, kind).
struct LocTable {
    file: GlobalId,
    rec_ty: TypeId,
    ptr: TypeId,
    i32t: TypeId,
    funcs: std::collections::HashMap<FuncId, GlobalId>,
    recs: std::collections::HashMap<(FuncId, u32, UbKind), GlobalId>,
    next: usize,
}

/// Define an internal constant `[N x i8]` global holding `s` NUL-terminated.
fn define_cstring(m: &mut Module, syms: &mut StrInterner, name: &str, s: &str) -> GlobalId {
    let i8t = m.types_mut().int(8);
    let bytes: Vec<u8> = s.bytes().filter(|&b| b != 0).chain(std::iter::once(0)).collect();
    let ty = m.types_mut().array(i8t, bytes.len() as u64);
    let elems = bytes.iter().map(|&b| m.intern_const(Const::Int { ty: i8t, value: Int::from_u64(u64::from(b)) })).collect();
    let init = m.intern_const(Const::Aggregate { ty, elems });
    let attrs = GlobalAttrs { linkage: Linkage::Internal, constant: true, ..GlobalAttrs::DEFAULT };
    let name = syms.intern(name);
    m.define_global(Global { name, ty, init: Some(init) }, attrs)
}

impl LocTable {
    fn new(m: &mut Module, syms: &mut StrInterner, file: &str) -> LocTable {
        // Number after any symbols a previous run created, so names stay unique.
        let next = m.globals().filter(|g| syms.resolve(g.name).starts_with(PREFIX)).count();
        let file = define_cstring(m, syms, &format!("{PREFIX}file{next}"), file);
        let ptr = m.types_mut().ptr();
        let i32t = m.types_mut().int(32);
        let rec_ty = m.types_mut().struct_(vec![ptr, ptr, i32t, i32t]);
        LocTable {
            file,
            rec_ty,
            ptr,
            i32t,
            funcs: Default::default(),
            recs: Default::default(),
            next: next + 1,
        }
    }

    fn get(&mut self, m: &mut Module, syms: &mut StrInterner, fid: FuncId, fname: &str, line: u32, kind: UbKind) -> GlobalId {
        if let Some(&g) = self.recs.get(&(fid, line, kind)) {
            return g;
        }
        let n = self.next;
        self.next += 1;
        let fname_g = match self.funcs.get(&fid) {
            Some(&g) => g,
            None => {
                let g = define_cstring(m, syms, &format!("{PREFIX}fn{n}"), fname);
                self.funcs.insert(fid, g);
                g
            }
        };
        let addr = |m: &mut Module, g| m.intern_const(Const::Addr { ty: self.ptr, target: AddrTarget::Global(g), offset: 0 });
        let elems = vec![
            addr(m, self.file),
            addr(m, fname_g),
            m.intern_const(Const::Int { ty: self.i32t, value: Int::from_u64(u64::from(line)) }),
            m.intern_const(Const::Int { ty: self.i32t, value: Int::ZERO }),
        ];
        let init = m.intern_const(Const::Aggregate { ty: self.rec_ty, elems });
        let attrs = GlobalAttrs { linkage: Linkage::Internal, ..GlobalAttrs::DEFAULT };
        let name = syms.intern(&format!("{PREFIX}loc{n}"));
        let g = m.define_global(Global { name, ty: self.rec_ty, init: Some(init) }, attrs);
        self.recs.insert((fid, line, kind), g);
        g
    }
}

// ---------------------------------------------------------------------------
// Emission.
// ---------------------------------------------------------------------------

struct EmitCtx {
    report_fn: Option<FuncId>,
    trap_global: Option<GlobalId>,
    i32t: TypeId,
    i64t: TypeId,
}

/// Rebuild `old` with the planned checks in front of their instructions.
fn rebuild(old: &Function, b: &mut FunctionBuilder<'_>, checks: &[Vec<Check>], ctx: &EmitCtx) {
    if let Some(l) = old.decl_line {
        b.set_decl_line(l);
    }
    let n = old.block_count();
    let entry = old.entry().expect("a definition has an entry").index();
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);

    let mut new_block: Vec<BlockId> = Vec::with_capacity(n);
    for bi in 0..n {
        if bi == entry {
            new_block.push(b.create_entry_block());
        } else {
            let tys: Vec<TypeId> = old.block(BlockId::from_index(bi)).params().iter().map(|&p| old.value_type(p)).collect();
            new_block.push(b.create_block(&tys));
        }
    }
    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for (bi, &nb) in new_block.iter().enumerate() {
        let np = b.block_params(nb).to_vec();
        for (k, &p) in old.block(BlockId::from_index(bi)).params().iter().enumerate() {
            vmap[p.index()] = Some(np[k]);
        }
    }

    for bi in dom_preorder(old, &doms) {
        let bb = BlockId::from_index(bi);
        b.switch_to(new_block[bi]);
        let insts = old.block(bb).insts();
        // Keep the entry block's static allocas ahead of every check: the
        // frame layout expects them before the first split.
        let order: Vec<InstId> = if bi == entry {
            let (allocas, rest): (Vec<InstId>, Vec<InstId>) =
                insts.iter().partition(|&&i| matches!(old.inst(i).kind, InstKind::Alloca { .. }));
            allocas.into_iter().chain(rest).collect()
        } else {
            insts.to_vec()
        };
        for i in order {
            b.set_line(old.inst_line(i).unwrap_or(0));
            let inst = old.inst(i);
            let ops: Vec<ValueId> = inst.operands().iter().map(|&o| remap_value(&mut vmap, old, b, o)).collect();
            let mut flags = inst.flags;
            let mut kind = inst.kind.clone();
            for c in &checks[i.index()] {
                emit_check(old, b, &mut vmap, &ops, inst.ty, c, ctx);
                match c.cond {
                    Cond::AddSub { signed, .. } | Cond::Mul { signed, .. } | Cond::Shl { signed } => {
                        if signed {
                            flags.nsw = false;
                        } else {
                            flags.nuw = false;
                        }
                    }
                    Cond::InexactDiv { .. } | Cond::InexactShr => flags.exact = false,
                    Cond::Bounds { access: None, .. } => kind = InstKind::PtrAdd { inbounds: false },
                    _ => {}
                }
            }
            let result_ty = inst.result().map(|_| inst.ty);
            let nr = b.append_inst(kind, ops, flags, result_ty);
            if let Some(r) = inst.result() {
                vmap[r.index()] = nr;
            }
        }
        if let Some(t) = old.block(bb).terminator() {
            b.set_line(old.inst_line(t).unwrap_or(0));
            for c in &checks[t.index()] {
                emit_check(old, b, &mut vmap, &[], old.inst(t).ty, c, ctx);
            }
        }
        rebuild_terminator(&mut vmap, old, b, &new_block, bb, |_, _, _| {});
    }
}

/// Emit one check into the current block, leaving the builder in the block
/// that continues after it.
fn emit_check(
    old: &Function,
    b: &mut FunctionBuilder<'_>,
    vmap: &mut [Option<ValueId>],
    ops: &[ValueId],
    ty: TypeId,
    c: &Check,
    ctx: &EmitCtx,
) {
    let (bad, a_arg, b_arg) = condition(old, b, vmap, ops, ty, c, ctx);
    let Some(bad) = bad else {
        // An `unreachable`: report (or record the trap code) right before it;
        // the block keeps its own terminator.
        emit_failure(b, a_arg, b_arg, c, ctx);
        return;
    };
    let report = b.create_block(&[]);
    let cont = b.create_block(&[]);
    let frozen = b.freeze(bad);
    b.cond_br(frozen, report, &[], cont, &[]);
    b.switch_to(report);
    emit_failure(b, a_arg, b_arg, c, ctx);
    if c.trap {
        b.unreachable();
    } else {
        b.br(cont, &[]);
    }
    b.switch_to(cont);
}

/// The failure path of a check: the handler call, or the trap's volatile store
/// of the code (the caller adds the `unreachable`).
fn emit_failure(b: &mut FunctionBuilder<'_>, a_arg: ValueId, b_arg: ValueId, c: &Check, ctx: &EmitCtx) {
    let code = b.const_i64(ctx.i32t, i64::from(c.code));
    if c.trap {
        let gp = b.global_ref(ctx.trap_global.expect("trap global declared"));
        b.store_volatile(ctx.i32t, gp, code, 4);
    } else {
        let f = b.func_ref(ctx.report_fn.expect("report handler declared"));
        let loc = b.global_ref(c.loc.expect("location record"));
        let void = b.types_mut().void();
        b.call(f, &[code, loc, a_arg, b_arg], void);
    }
}

/// An integer constant of type `ty`.
fn iconst(b: &mut FunctionBuilder<'_>, ty: TypeId, v: Int) -> ValueId {
    b.const_int(ty, v)
}

/// The width of an integer type.
fn width_of(b: &FunctionBuilder<'_>, ty: TypeId) -> u32 {
    match b.types().get(ty) {
        Type::Int(w) => *w,
        _ => 0,
    }
}

/// `v` (any scalar) as a frozen `i64`, sign- or zero-extended.
fn to_i64(b: &mut FunctionBuilder<'_>, v: ValueId, signed: bool, i64t: TypeId) -> ValueId {
    let ty = b.value_type(v);
    let v = match b.types().get(ty).clone() {
        Type::Int(w) => match w.cmp(&64) {
            std::cmp::Ordering::Less => b.cast(if signed { CastOp::SExt } else { CastOp::ZExt }, v, i64t),
            std::cmp::Ordering::Equal => v,
            std::cmp::Ordering::Greater => b.cast(CastOp::Trunc, v, i64t),
        },
        Type::Ptr | Type::PtrIn(_) => {
            let bits = b.types().pointer_bits(ty).unwrap_or(64);
            let it = b.types_mut().int(bits);
            let i = b.cast(CastOp::PtrToInt, v, it);
            if bits == 64 { i } else { b.cast(CastOp::ZExt, i, i64t) }
        }
        Type::Float(k) => {
            let it = b.types_mut().int(k.bit_width());
            let i = b.cast(CastOp::Bitcast, v, it);
            if k.bit_width() == 64 { i } else { b.cast(CastOp::ZExt, i, i64t) }
        }
        _ => return b.const_i64(i64t, 0),
    };
    b.freeze(v)
}

/// An integer offset operand as `i64` (sign-extended or truncated, like
/// `ptr_add` reads it).
fn offset_i64(b: &mut FunctionBuilder<'_>, v: ValueId, i64t: TypeId) -> ValueId {
    let w = width_of(b, b.value_type(v));
    match w.cmp(&64) {
        std::cmp::Ordering::Less => b.cast(CastOp::SExt, v, i64t),
        std::cmp::Ordering::Equal => v,
        std::cmp::Ordering::Greater => b.cast(CastOp::Trunc, v, i64t),
    }
}

/// A shift amount clamped below the width (`select(amt < w, amt, 0)`), so the
/// shifts a condition computes are never poison.
fn safe_amount(b: &mut FunctionBuilder<'_>, amt: ValueId, ty: TypeId, w: u32) -> ValueId {
    let wc = iconst(b, ty, Int::from_u64(u64::from(w)));
    let ok = b.icmp(IntPred::Ult, amt, wc);
    let zero = iconst(b, ty, Int::ZERO);
    b.select(ok, amt, zero)
}

/// The bits of `±2^k` in format `kind` (infinity beyond the range).
fn pow2_bits(kind: FloatKind, k: u32, negative: bool) -> FloatBits {
    match kind {
        FloatKind::F16 => {
            let mag: u16 = if k <= 15 { ((15 + k) as u16) << 10 } else { 0x7c00 };
            FloatBits::F16(mag | if negative { 0x8000 } else { 0 })
        }
        FloatKind::F32 => {
            let mag: u32 = if k <= 127 { (127 + k) << 23 } else { 0x7f80_0000 };
            FloatBits::F32(mag | if negative { 0x8000_0000 } else { 0 })
        }
        FloatKind::F64 => {
            let mag: u64 = if k <= 1023 { u64::from(1023 + k) << 52 } else { 0x7ff0_0000_0000_0000 };
            FloatBits::F64(mag | if negative { 1 << 63 } else { 0 })
        }
    }
}

/// Whether `2^k` exceeds the finite range of `kind` (so [`pow2_bits`] gives
/// an infinity).
fn pow2_overflows(kind: FloatKind, k: u32) -> bool {
    k > match kind {
        FloatKind::F16 => 15,
        FloatKind::F32 => 127,
        FloatKind::F64 => 1023,
    }
}

/// The bits of the negative integer `-n` (`n ≥ 1`, exactly representable in
/// `kind`).
fn neg_int_bits(kind: FloatKind, n: u64) -> FloatBits {
    match kind {
        FloatKind::F16 => {
            let e = 63 - n.leading_zeros();
            let mant = if e >= 10 { (n >> (e - 10)) & 0x3ff } else { (n << (10 - e)) & 0x3ff };
            FloatBits::F16(0x8000 | ((15 + e) as u16) << 10 | mant as u16)
        }
        FloatKind::F32 => FloatBits::F32((-(n as f32)).to_bits()),
        FloatKind::F64 => FloatBits::F64((-(n as f64)).to_bits()),
    }
}

/// The significand precision (bits, implicit one included) of a format.
fn precision(kind: FloatKind) -> u32 {
    match kind {
        FloatKind::F16 => 11,
        FloatKind::F32 => 24,
        FloatKind::F64 => 53,
    }
}

/// Compute a check's condition (`None` for [`Cond::Always`]) and the
/// handler's two operand arguments, in the current block.
fn condition(
    old: &Function,
    b: &mut FunctionBuilder<'_>,
    vmap: &mut [Option<ValueId>],
    ops: &[ValueId],
    ty: TypeId,
    c: &Check,
    ctx: &EmitCtx,
) -> (Option<ValueId>, ValueId, ValueId) {
    let i64t = ctx.i64t;
    let zero64 = b.const_i64(i64t, 0);
    let w = width_of(b, ty);
    let min = || Int::ONE.mul_2k(w.saturating_sub(1));
    let args = |b: &mut FunctionBuilder<'_>, sa: bool, sb: bool| {
        let x = to_i64(b, ops[0], sa, i64t);
        let y = to_i64(b, ops[1], sb, i64t);
        (x, y)
    };
    match &c.cond {
        Cond::AddSub { sub, signed } => {
            let (x, y) = (ops[0], ops[1]);
            let bad = if *signed {
                let op = if *sub { BinOp::Sub } else { BinOp::Add };
                let r = b.bin(op, x, y, Flags::NONE);
                // add: (x^r) & (y^r) < 0; sub: (x^y) & (x^r) < 0.
                let xr = b.bin(BinOp::Xor, x, r, Flags::NONE);
                let other = if *sub { b.bin(BinOp::Xor, x, y, Flags::NONE) } else { b.bin(BinOp::Xor, y, r, Flags::NONE) };
                let t = b.bin(BinOp::And, xr, other, Flags::NONE);
                let z = iconst(b, ty, Int::ZERO);
                b.icmp(IntPred::Slt, t, z)
            } else if *sub {
                b.icmp(IntPred::Ult, x, y)
            } else {
                let r = b.bin(BinOp::Add, x, y, Flags::NONE);
                b.icmp(IntPred::Ult, r, x)
            };
            let (p, q) = args(b, *signed, *signed);
            (Some(bad), p, q)
        }
        Cond::Mul { signed, widen } => {
            let (x, y) = (ops[0], ops[1]);
            let bad = match widen {
                Some(ww) => {
                    let wt = b.types_mut().int(*ww);
                    let ext = if *signed { CastOp::SExt } else { CastOp::ZExt };
                    let xw = b.cast(ext, x, wt);
                    let yw = b.cast(ext, y, wt);
                    let p = b.bin(BinOp::Mul, xw, yw, Flags::NONE);
                    let t = b.cast(CastOp::Trunc, p, ty);
                    let back = b.cast(ext, t, wt);
                    b.icmp(IntPred::Ne, back, p)
                }
                None => {
                    // Divide the wrapped product back: it overflowed iff
                    // r / x != y (x ∉ {0, -1}); x = -1 overflows iff y = MIN.
                    let r = b.bin(BinOp::Mul, x, y, Flags::NONE);
                    let z = iconst(b, ty, Int::ZERO);
                    let one = iconst(b, ty, Int::ONE);
                    let xz = b.icmp(IntPred::Eq, x, z);
                    if *signed {
                        let m1 = iconst(b, ty, Int::MINUS_ONE);
                        let xm1 = b.icmp(IntPred::Eq, x, m1);
                        let skip = b.bin(BinOp::Or, xz, xm1, Flags::NONE);
                        let d = b.select(skip, one, x);
                        let q = b.bin(BinOp::SDiv, r, d, Flags::NONE);
                        let qne = b.icmp(IntPred::Ne, q, y);
                        let t = b.const_bool(true);
                        let keep = b.bin(BinOp::Xor, skip, t, Flags::NONE);
                        let general = b.bin(BinOp::And, keep, qne, Flags::NONE);
                        let mn = iconst(b, ty, min());
                        let ymin = b.icmp(IntPred::Eq, y, mn);
                        let neg = b.bin(BinOp::And, xm1, ymin, Flags::NONE);
                        b.bin(BinOp::Or, general, neg, Flags::NONE)
                    } else {
                        let d = b.select(xz, one, x);
                        let q = b.bin(BinOp::UDiv, r, d, Flags::NONE);
                        let qne = b.icmp(IntPred::Ne, q, y);
                        let nz = b.icmp(IntPred::Ne, x, z);
                        b.bin(BinOp::And, nz, qne, Flags::NONE)
                    }
                }
            };
            let (p, q) = args(b, *signed, *signed);
            (Some(bad), p, q)
        }
        Cond::Shl { signed } => {
            let (x, amt) = (ops[0], ops[1]);
            let s = safe_amount(b, amt, ty, w);
            let r = b.bin(BinOp::Shl, x, s, Flags::NONE);
            let back = b.bin(if *signed { BinOp::AShr } else { BinOp::LShr }, r, s, Flags::NONE);
            let bad = b.icmp(IntPred::Ne, back, x);
            let (p, q) = args(b, *signed, false);
            (Some(bad), p, q)
        }
        Cond::ShiftAmount => {
            let wc = iconst(b, ty, Int::from_u64(u64::from(w)));
            let bad = b.icmp(IntPred::Uge, ops[1], wc);
            let (p, q) = args(b, true, false);
            (Some(bad), p, q)
        }
        Cond::DivZero => {
            let z = iconst(b, ty, Int::ZERO);
            let bad = b.icmp(IntPred::Eq, ops[1], z);
            let (p, q) = args(b, true, true);
            (Some(bad), p, q)
        }
        Cond::DivOverflow => {
            let mn = iconst(b, ty, min());
            let m1 = iconst(b, ty, Int::MINUS_ONE);
            let xm = b.icmp(IntPred::Eq, ops[0], mn);
            let ym = b.icmp(IntPred::Eq, ops[1], m1);
            let bad = b.bin(BinOp::And, xm, ym, Flags::NONE);
            let (p, q) = args(b, true, true);
            (Some(bad), p, q)
        }
        Cond::InexactDiv { signed } => {
            let (x, y) = (ops[0], ops[1]);
            let z = iconst(b, ty, Int::ZERO);
            let one = iconst(b, ty, Int::ONE);
            let mut unsafe_d = b.icmp(IntPred::Eq, y, z);
            if *signed {
                let mn = iconst(b, ty, min());
                let m1 = iconst(b, ty, Int::MINUS_ONE);
                let xm = b.icmp(IntPred::Eq, x, mn);
                let ym = b.icmp(IntPred::Eq, y, m1);
                let ov = b.bin(BinOp::And, xm, ym, Flags::NONE);
                unsafe_d = b.bin(BinOp::Or, unsafe_d, ov, Flags::NONE);
            }
            let d = b.select(unsafe_d, one, y);
            let rem = b.bin(if *signed { BinOp::SRem } else { BinOp::URem }, x, d, Flags::NONE);
            let bad = b.icmp(IntPred::Ne, rem, z);
            let (p, q) = args(b, *signed, *signed);
            (Some(bad), p, q)
        }
        Cond::InexactShr => {
            let (x, amt) = (ops[0], ops[1]);
            let s = safe_amount(b, amt, ty, w);
            let one = iconst(b, ty, Int::ONE);
            let bit = b.bin(BinOp::Shl, one, s, Flags::NONE);
            let mask = b.bin(BinOp::Sub, bit, one, Flags::NONE);
            let lost = b.bin(BinOp::And, x, mask, Flags::NONE);
            let z = iconst(b, ty, Int::ZERO);
            let bad = b.icmp(IntPred::Ne, lost, z);
            let (p, q) = args(b, false, false);
            (Some(bad), p, q)
        }
        Cond::FloatCast { signed, width, float } => {
            let x = ops[0];
            let fty = b.value_type(x);
            let (lo_pred, lo_bits, hi_k) = if *signed {
                if *width <= precision(*float) {
                    // Strictly above -2^(w-1) - 1 (exactly representable).
                    let n = (1u64 << (width - 1)) + 1;
                    (FloatPred::Ule, neg_int_bits(*float, n), width - 1)
                } else {
                    // -2^(w-1) - 1 is not representable: no float lies
                    // strictly between it and -2^(w-1). When -2^(w-1) itself
                    // overflows the format, the bound is -inf, which is out
                    // of range too.
                    let bound = pow2_bits(*float, width - 1, true);
                    let pred = if pow2_overflows(*float, width - 1) { FloatPred::Ule } else { FloatPred::Ult };
                    (pred, bound, width - 1)
                }
            } else {
                (FloatPred::Ule, neg_int_bits(*float, 1), *width)
            };
            let lo = b.const_float(fty, lo_bits);
            let hi = b.const_float(fty, pow2_bits(*float, hi_k, false));
            let below = b.fcmp(lo_pred, x, lo, Flags::NONE);
            let above = b.fcmp(FloatPred::Uge, x, hi, Flags::NONE);
            let bad = b.bin(BinOp::Or, below, above, Flags::NONE);
            let p = to_i64(b, x, false, i64t);
            (Some(bad), p, zero64)
        }
        Cond::Null(p) => {
            let p = remap_value(vmap, old, b, *p);
            let pty = b.value_type(p);
            let null = b.null(pty);
            let bad = b.icmp(IntPred::Eq, p, null);
            (Some(bad), zero64, zero64)
        }
        Cond::Misaligned(p, align) => {
            let p = remap_value(vmap, old, b, *p);
            let pty = b.value_type(p);
            let bits = b.types().pointer_bits(pty).unwrap_or(64);
            let it = b.types_mut().int(bits);
            let pi = b.cast(CastOp::PtrToInt, p, it);
            let mask = iconst(b, it, Int::from_u64(u64::from(*align) - 1));
            let low = b.bin(BinOp::And, pi, mask, Flags::NONE);
            let z = iconst(b, it, Int::ZERO);
            let bad = b.icmp(IntPred::Ne, low, z);
            let a = to_i64(b, p, false, i64t);
            let al = b.const_i64(i64t, i64::from(*align));
            (Some(bad), a, al)
        }
        Cond::Bounds { offs, const_off, size, access } => {
            let mut total = b.const_i64(i64t, *const_off as i64);
            for &o in offs {
                let o = remap_value(vmap, old, b, o);
                let o = offset_i64(b, o, i64t);
                total = b.bin(BinOp::Add, total, o, Flags::NONE);
            }
            let total = b.freeze(total);
            let (bad, size_v) = match (size, access) {
                (ObjSize::Static(s), None) => {
                    let sv = b.const_i64(i64t, *s as i64);
                    (b.icmp(IntPred::Ugt, total, sv), sv)
                }
                (ObjSize::Static(s), Some(acc)) => {
                    let sv = b.const_i64(i64t, *s as i64);
                    let bad = if acc > s {
                        b.const_bool(true)
                    } else {
                        let lim = b.const_i64(i64t, (s - acc) as i64);
                        b.icmp(IntPred::Ugt, total, lim)
                    };
                    (bad, sv)
                }
                (ObjSize::Dynamic(n), acc) => {
                    let n = remap_value(vmap, old, b, *n);
                    let nw = width_of(b, b.value_type(n));
                    let n64 = match nw.cmp(&64) {
                        std::cmp::Ordering::Less => b.cast(CastOp::ZExt, n, i64t),
                        std::cmp::Ordering::Equal => n,
                        std::cmp::Ordering::Greater => b.cast(CastOp::Trunc, n, i64t),
                    };
                    let n64 = b.freeze(n64);
                    let past = b.icmp(IntPred::Ugt, total, n64);
                    let bad = match acc {
                        None => past,
                        Some(acc) => {
                            let room = b.bin(BinOp::Sub, n64, total, Flags::NONE);
                            let accv = b.const_i64(i64t, *acc as i64);
                            let short = b.icmp(IntPred::Ult, room, accv);
                            b.bin(BinOp::Or, past, short, Flags::NONE)
                        }
                    };
                    (bad, n64)
                }
            };
            (Some(bad), total, size_v)
        }
        Cond::Always => (None, zero64, zero64),
    }
}

#[cfg(test)]
mod tests;
