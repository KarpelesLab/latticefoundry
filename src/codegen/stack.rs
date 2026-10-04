//! Stack usage reporting and worst-case stack-depth analysis, and the stack
//! probing contract shared by the backends.
//!
//! # Stack usage
//!
//! After register allocation, each backend lays out a function's frame (which
//! callee-saved registers to save, where each spill/`alloca` slot lives, how far
//! to move the stack pointer) and builds the prologue from that layout. The same
//! layout produces the function's [`StackUsage`], so the reported numbers are the
//! numbers the prologue actually uses and cannot drift from the machine code.
//!
//! [`StackUsage::frame_size`] is the **static stack cost** of one activation:
//! how far below the caller's stack pointer (its value just before the call
//! instruction) the function's own stack pointer gets, at most, before any
//! dynamic allocation. It covers the return address (when the call instruction
//! pushes one), the saved frame pointer / link register and callee-saved
//! registers, spill slots, fixed `alloca` slots, the outgoing stack-argument
//! area, and alignment padding. The stack depth of a call chain is therefore the
//! sum of the frame sizes along it, and [`StackReport::worst_case_depth`]
//! computes the maximum over all paths from a root.
//!
//! A function that uses `dyn_alloca` has no static bound by itself
//! ([`StackUsage::dynamic_alloca`]); one that calls through a pointer
//! ([`StackUsage::indirect_calls`]) or calls a function outside the report needs
//! a caller-supplied bound ([`StackAssumptions`]). Syscalls use no user stack on
//! Linux (the kernel switches to its own stack), so they add nothing; they are
//! still reported ([`StackUsage::syscalls`]).
//!
//! Not counted: the x86-64 System V red zone (LF never uses it), and anything
//! the kernel pushes to deliver a signal on the same stack (a handler's frame is
//! a separate root to analyze, plus the kernel's signal frame).
//!
//! # Stack probes
//!
//! Linux (and most systems) protect a stack with a guard region below it: an
//! access there faults, turning an overflow into `SIGSEGV`. A single large
//! stack-pointer adjustment can jump *over* the guard and let the function
//! write into whatever lies below. With probing on
//! ([`CodegenOptions::stack_probes`](crate::codegen::CodegenOptions::stack_probes),
//! the default) every backend maintains this invariant:
//!
//! > Between any two consecutive stack accesses of a thread, the stack pointer
//! > never moves more than [`STACK_PROBE_INTERVAL`] bytes below the lowest
//! > address touched so far.
//!
//! Concretely: an adjustment smaller than the interval is emitted as one
//! instruction (the call's return-address push on x86-64, the `stp x29, x30`
//! pre-index store on AArch64, or the `sd ra` save on RISC-V is the next touch,
//! and the remainder is small enough that the gap stays within one interval);
//! an adjustment of at least the interval moves the stack pointer one interval
//! at a time and writes to the new top after each step (unrolled for a few
//! pages, a counted loop beyond), then applies the sub-interval remainder. On
//! x86-64 and AArch64 a `dyn_alloca` does the same at run time (touching the
//! current top first, then every interval of the requested size). An interval of 4096 bytes
//! is at most the guard size on every Linux configuration (4 KiB pages or
//! larger; the main thread's guard gap is 1 MiB by default), so no probe
//! sequence can skip the guard.

use std::collections::BTreeMap;
use std::fmt;

use crate::codegen::mir::{MachineFunction, MachineOperand, Opcode};

/// The stack-probe step in bytes: the largest distance the stack pointer moves
/// without an intervening access when probing is on. 4096 is the smallest page
/// size (and thus guard size) on the supported Linux targets.
pub const STACK_PROBE_INTERVAL: u64 = 4096;

/// The stack usage of one compiled function, extracted from its frame layout.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StackUsage {
    /// The function's symbol name.
    pub name: String,
    /// The exact static stack cost of one activation, in bytes:
    /// `return_address + saved_registers + locals + outgoing_args + padding`,
    /// i.e. the distance from the caller's stack pointer (just before the call)
    /// to this function's stack pointer after its prologue.
    pub frame_size: u64,
    /// Bytes the call instruction itself pushes (8 on x86-64; 0 on AArch64 and
    /// RISC-V, whose calls leave the return address in a register).
    pub return_address: u64,
    /// Bytes holding saved registers: the frame pointer and link register
    /// (`rbp`; `x29`/`x30`; `ra` when the function calls) plus the callee-saved
    /// registers the allocation used.
    pub saved_registers: u64,
    /// The explicit stack-pointer adjustment of the prologue in bytes — the `N`
    /// of `sub rsp, N` (x86-64), `sub sp, sp, #N` (AArch64, below the `x29`/`x30`
    /// pair), or `addi sp, sp, -N` (RISC-V, which also covers its saved
    /// registers). Probing splits it into steps but moves the same total.
    pub sp_adjust: u64,
    /// The outgoing stack-argument area at the bottom of the frame (bytes, part
    /// of [`StackUsage::sp_adjust`]).
    pub outgoing_args: u64,
    /// Whether the function contains a `dyn_alloca` (runtime-sized stack
    /// allocation): its total depth is unbounded unless the caller supplies a
    /// bound ([`StackAssumptions::dynamic`]).
    pub dynamic_alloca: bool,
    /// The symbol names of the functions this one calls directly, deduplicated,
    /// in first-call order.
    pub direct_callees: Vec<String>,
    /// Whether the function calls through a pointer (callee unknown statically).
    pub indirect_calls: bool,
    /// Whether the function executes a `syscall` (uses no user stack on Linux).
    pub syscalls: bool,
    /// Whether the prologue (and any `dyn_alloca`) was emitted with stack probes.
    pub probed: bool,
}

/// What a function's machine code calls, scanned from its MIR.
#[derive(Clone, Debug, Default)]
pub(crate) struct CallScan {
    /// Direct callees as function indices, deduplicated, in first-call order.
    pub direct: Vec<u32>,
    pub indirect: bool,
    pub syscalls: bool,
    pub dynamic_alloca: bool,
}

/// Scan `mf` for calls (`call` opcode: a `Func` first operand is a direct call,
/// anything else an indirect one), syscalls, and dynamic allocations.
pub(crate) fn scan_calls(
    mf: &MachineFunction,
    call: Opcode,
    syscall: Opcode,
    dyn_alloca: Option<Opcode>,
) -> CallScan {
    let mut scan = CallScan::default();
    for bid in mf.block_ids() {
        for inst in &mf.block(bid).insts {
            if inst.opcode == call {
                match inst.operands.first() {
                    Some(MachineOperand::Func(f)) => {
                        if !scan.direct.contains(f) {
                            scan.direct.push(*f);
                        }
                    }
                    _ => scan.indirect = true,
                }
            } else if inst.opcode == syscall {
                scan.syscalls = true;
            } else if Some(inst.opcode) == dyn_alloca {
                scan.dynamic_alloca = true;
            }
        }
    }
    scan
}

/// The stack usage of every function of one or more compiled modules.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StackReport {
    functions: Vec<StackUsage>,
}

/// Caller-supplied stack bounds for what the report cannot see.
///
/// All bounds are in bytes. An **external** bound is the whole worst-case depth
/// of that callee (its own frame plus everything it calls); an **indirect**
/// bound, keyed by the *calling* function, is the worst-case depth of any
/// function it may call through a pointer; a **dynamic** bound, keyed by the
/// function, is the most its `dyn_alloca`s allocate in one activation (their
/// sizes rounded as the target does, plus per-allocation alignment slack).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StackAssumptions {
    external: BTreeMap<String, u64>,
    indirect: BTreeMap<String, u64>,
    dynamic: BTreeMap<String, u64>,
}

impl StackAssumptions {
    /// No assumptions: every external/indirect/dynamic use is unbounded.
    pub fn new() -> StackAssumptions {
        StackAssumptions::default()
    }

    /// Assume a call to the (undefined) function `name` needs at most `bytes`.
    pub fn external(mut self, name: impl Into<String>, bytes: u64) -> StackAssumptions {
        self.external.insert(name.into(), bytes);
        self
    }

    /// Assume every indirect call made by `caller` needs at most `bytes`.
    pub fn indirect(mut self, caller: impl Into<String>, bytes: u64) -> StackAssumptions {
        self.indirect.insert(caller.into(), bytes);
        self
    }

    /// Assume the `dyn_alloca`s of one activation of `function` allocate at most
    /// `bytes` in total.
    pub fn dynamic(mut self, function: impl Into<String>, bytes: u64) -> StackAssumptions {
        self.dynamic.insert(function.into(), bytes);
        self
    }
}

/// A proven worst-case stack depth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackBound {
    /// The worst-case depth in bytes, counted from the root's caller's stack
    /// pointer just before it calls the root.
    pub bytes: u64,
    /// One deepest call path, root first. An external callee appears by name;
    /// an assumed indirect call as [`StackBound::INDIRECT`].
    pub path: Vec<String>,
}

impl StackBound {
    /// The [`StackBound::path`] entry standing for an assumed indirect callee.
    pub const INDIRECT: &'static str = "<indirect>";
}

/// Why no stack bound exists.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StackBoundError {
    /// The root function is not in the report.
    UnknownRoot(String),
    /// The call graph has a cycle reachable from the root; `cycle` lists it,
    /// starting and ending with the same function.
    Recursion {
        /// The functions on the cycle, first == last.
        cycle: Vec<String>,
    },
    /// `function` makes an indirect call and no bound was assumed for it.
    IndirectCall {
        /// The calling function.
        function: String,
    },
    /// `function` uses `dyn_alloca` and no bound was assumed for it.
    DynamicAlloca {
        /// The allocating function.
        function: String,
    },
    /// `caller` calls `callee`, which is neither in the report nor assumed.
    UnknownCallee {
        /// The calling function.
        caller: String,
        /// The undefined callee.
        callee: String,
    },
}

impl fmt::Display for StackBoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StackBoundError::UnknownRoot(n) => write!(f, "no function '{n}' in the stack report"),
            StackBoundError::Recursion { cycle } => {
                write!(f, "recursion: {}", cycle.join(" -> "))
            }
            StackBoundError::IndirectCall { function } => {
                write!(f, "'{function}' makes an indirect call with no assumed bound")
            }
            StackBoundError::DynamicAlloca { function } => {
                write!(f, "'{function}' uses dyn_alloca with no assumed bound")
            }
            StackBoundError::UnknownCallee { caller, callee } => {
                write!(f, "'{caller}' calls '{callee}', whose stack usage is unknown")
            }
        }
    }
}

impl std::error::Error for StackBoundError {}

/// The memoized result for one function: its depth and the chosen next hop.
#[derive(Clone)]
struct Solved {
    depth: u64,
    next: Option<String>,
}

impl StackReport {
    /// An empty report.
    pub fn new() -> StackReport {
        StackReport::default()
    }

    /// Append one function's usage.
    pub fn push(&mut self, usage: StackUsage) {
        self.functions.push(usage);
    }

    /// Append every function of `other` (e.g. to analyze a program compiled as
    /// several modules).
    pub fn extend(&mut self, other: StackReport) {
        self.functions.extend(other.functions);
    }

    /// Every function's usage, in definition order.
    pub fn functions(&self) -> &[StackUsage] {
        &self.functions
    }

    /// The usage of the function named `name`, if it is in the report.
    pub fn get(&self, name: &str) -> Option<&StackUsage> {
        self.functions.iter().find(|u| u.name == name)
    }

    /// The worst-case stack depth of a call to `root`: the maximum, over every
    /// call path from `root`, of the sum of the frame sizes along it (plus the
    /// assumed bounds for external callees, indirect calls, and `dyn_alloca`).
    ///
    /// Returns why no bound exists instead when a reachable function recurses,
    /// calls indirectly, uses `dyn_alloca`, or calls an undefined function
    /// without a corresponding [`StackAssumptions`] entry. When several such
    /// problems exist, the first one met in a depth-first walk (callees in call
    /// order) is reported.
    pub fn worst_case_depth(
        &self,
        root: &str,
        assume: &StackAssumptions,
    ) -> Result<StackBound, StackBoundError> {
        if self.get(root).is_none() {
            return Err(StackBoundError::UnknownRoot(root.to_owned()));
        }
        let index: BTreeMap<&str, &StackUsage> =
            self.functions.iter().map(|u| (u.name.as_str(), u)).collect();
        let mut memo: BTreeMap<String, Solved> = BTreeMap::new();
        let mut active: Vec<String> = Vec::new();
        let depth = solve(root, &index, assume, &mut memo, &mut active)?;

        let mut path = vec![root.to_owned()];
        let mut cur = root.to_owned();
        while let Some(next) = memo.get(&cur).and_then(|s| s.next.clone()) {
            path.push(next.clone());
            if !index.contains_key(next.as_str()) {
                break; // an external or indirect leaf
            }
            cur = next;
        }
        Ok(StackBound { bytes: depth, path })
    }
}

fn solve(
    name: &str,
    index: &BTreeMap<&str, &StackUsage>,
    assume: &StackAssumptions,
    memo: &mut BTreeMap<String, Solved>,
    active: &mut Vec<String>,
) -> Result<u64, StackBoundError> {
    if let Some(s) = memo.get(name) {
        return Ok(s.depth);
    }
    let u = index[name];
    active.push(name.to_owned());

    let mut extra = 0u64;
    if u.dynamic_alloca {
        extra = *assume
            .dynamic
            .get(name)
            .ok_or_else(|| StackBoundError::DynamicAlloca { function: name.to_owned() })?;
    }
    let mut best = 0u64;
    let mut next: Option<String> = None;
    if u.indirect_calls {
        best = *assume
            .indirect
            .get(name)
            .ok_or_else(|| StackBoundError::IndirectCall { function: name.to_owned() })?;
        next = Some(StackBound::INDIRECT.to_owned());
    }
    for callee in &u.direct_callees {
        let d = if index.contains_key(callee.as_str()) {
            if let Some(pos) = active.iter().position(|a| a == callee) {
                let mut cycle = active[pos..].to_vec();
                cycle.push(callee.clone());
                return Err(StackBoundError::Recursion { cycle });
            }
            solve(callee, index, assume, memo, active)?
        } else {
            *assume.external.get(callee).ok_or_else(|| StackBoundError::UnknownCallee {
                caller: name.to_owned(),
                callee: callee.clone(),
            })?
        };
        if next.is_none() || d > best {
            best = d;
            next = Some(callee.clone());
        }
    }

    active.pop();
    let depth = u.frame_size + extra + best;
    memo.insert(name.to_owned(), Solved { depth, next });
    Ok(depth)
}

impl fmt::Display for StackReport {
    /// A `-fstack-usage`-style table: one row per function with its static frame
    /// size and a qualifier (`static`, or `dynamic` for `dyn_alloca`), followed
    /// by what it calls.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let w = self.functions.iter().map(|u| u.name.len()).max().unwrap_or(0).max(8);
        writeln!(f, "{:<w$}  {:>8}  {:<8}  calls", "function", "frame", "kind")?;
        for u in &self.functions {
            let kind = if u.dynamic_alloca { "dynamic" } else { "static" };
            let mut calls: Vec<String> = u.direct_callees.clone();
            if u.indirect_calls {
                calls.push(StackBound::INDIRECT.to_owned());
            }
            if u.syscalls {
                calls.push("<syscall>".to_owned());
            }
            writeln!(f, "{:<w$}  {:>8}  {:<8}  {}", u.name, u.frame_size, kind, calls.join(", "))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(name: &str, frame: u64, callees: &[&str]) -> StackUsage {
        StackUsage {
            name: name.to_owned(),
            frame_size: frame,
            return_address: 0,
            saved_registers: 0,
            sp_adjust: frame,
            outgoing_args: 0,
            dynamic_alloca: false,
            direct_callees: callees.iter().map(|s| (*s).to_owned()).collect(),
            indirect_calls: false,
            syscalls: false,
            probed: true,
        }
    }

    fn report(us: Vec<StackUsage>) -> StackReport {
        let mut r = StackReport::new();
        for u in us {
            r.push(u);
        }
        r
    }

    #[test]
    fn diamond_takes_the_deepest_path() {
        let r = report(vec![
            usage("main", 32, &["a", "b"]),
            usage("a", 100, &["leaf"]),
            usage("b", 16, &["leaf"]),
            usage("leaf", 48, &[]),
        ]);
        let b = r.worst_case_depth("main", &StackAssumptions::new()).unwrap();
        assert_eq!(b.bytes, 32 + 100 + 48);
        assert_eq!(b.path, ["main", "a", "leaf"]);
        assert_eq!(r.worst_case_depth("leaf", &StackAssumptions::new()).unwrap().bytes, 48);
    }

    #[test]
    fn recursion_is_reported_with_its_cycle() {
        let r = report(vec![
            usage("main", 16, &["f"]),
            usage("f", 16, &["g"]),
            usage("g", 16, &["f"]),
        ]);
        assert_eq!(
            r.worst_case_depth("main", &StackAssumptions::new()),
            Err(StackBoundError::Recursion { cycle: vec!["f".into(), "g".into(), "f".into()] })
        );
        let self_rec = report(vec![usage("r", 16, &["r"])]);
        assert_eq!(
            self_rec.worst_case_depth("r", &StackAssumptions::new()),
            Err(StackBoundError::Recursion { cycle: vec!["r".into(), "r".into()] })
        );
    }

    #[test]
    fn unknowns_need_assumptions() {
        let mut ind = usage("ind", 64, &[]);
        ind.indirect_calls = true;
        let mut dy = usage("dy", 32, &[]);
        dy.dynamic_alloca = true;
        let r = report(vec![usage("main", 16, &["ext", "ind", "dy"]), ind, dy]);
        let none = StackAssumptions::new();
        assert_eq!(
            r.worst_case_depth("main", &none),
            Err(StackBoundError::UnknownCallee { caller: "main".into(), callee: "ext".into() })
        );
        let a = StackAssumptions::new().external("ext", 500);
        assert_eq!(
            r.worst_case_depth("main", &a),
            Err(StackBoundError::IndirectCall { function: "ind".into() })
        );
        let a = a.indirect("ind", 1000);
        assert_eq!(
            r.worst_case_depth("main", &a),
            Err(StackBoundError::DynamicAlloca { function: "dy".into() })
        );
        let a = a.dynamic("dy", 4096);
        let b = r.worst_case_depth("main", &a).unwrap();
        // max(ext 500, ind 64+1000, dy 32+4096) = 4128.
        assert_eq!(b.bytes, 16 + 32 + 4096);
        assert_eq!(b.path, ["main", "dy"]);
        let b = r.worst_case_depth("ind", &a).unwrap();
        assert_eq!(b.path, ["ind", StackBound::INDIRECT]);
        assert_eq!(
            r.worst_case_depth("nope", &a),
            Err(StackBoundError::UnknownRoot("nope".into()))
        );
    }

    #[test]
    fn report_table_lists_every_function() {
        let mut s = usage("sys", 16, &[]);
        s.syscalls = true;
        let r = report(vec![usage("main", 32, &["sys"]), s]);
        let t = r.to_string();
        assert!(t.contains("main") && t.contains("sys") && t.contains("<syscall>"), "{t}");
        assert_eq!(r.get("sys").unwrap().frame_size, 16);
    }
}
