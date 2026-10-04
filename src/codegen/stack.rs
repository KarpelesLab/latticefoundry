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
//! When no bound exists, [`StackReport::analyze_from`] lists **every** reason
//! ([`StackBlocker`]: each recursive group once, with its members; each
//! indirect caller, `dyn_alloca` user and unknown callee), each with a
//! shortest call path from the root, in a deterministic order (that of a
//! depth-first walk, callees in call order; the first is what
//! [`StackReport::worst_case_depth`] returns). Only the functions reachable
//! from the root count, so obstacles in dead code are ignored;
//! [`StackReport::reachable`], [`StackReport::reachable_from`] and
//! [`StackReport::call_path`] expose that reachability (e.g. to print the
//! table without dead functions, as `lf build --stack-usage` does).
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

use std::collections::{BTreeMap, BTreeSet};
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
    /// order) is reported: the first of [`StackAnalysis::blockers`] of
    /// [`StackReport::analyze_from`], which lists them all with their paths.
    pub fn worst_case_depth(
        &self,
        root: &str,
        assume: &StackAssumptions,
    ) -> Result<StackBound, StackBoundError> {
        self.analyze_from(root, assume).into_result()
    }

    /// Analyze the stack depth of a call to `root`: either the worst-case
    /// bound with its deepest path (as [`StackReport::worst_case_depth`]), or
    /// **every** reason no bound exists, each with a shortest call path from
    /// `root` ([`StackBlocker`]).
    ///
    /// Only the functions reachable from `root` through direct calls count, so
    /// a recursion, indirect call, `dyn_alloca`, or unknown callee in dead code
    /// does not block the bound. The reasons come in a deterministic order —
    /// the order a depth-first walk from `root` (callees in call order) meets
    /// them — and each strongly connected group of mutually recursive functions
    /// is reported once.
    pub fn analyze_from(&self, root: &str, assume: &StackAssumptions) -> StackAnalysis {
        let index = self.index();
        let Some((&root_key, _)) = index.get_key_value(root) else {
            return StackAnalysis {
                root: root.to_owned(),
                reachable: Vec::new(),
                result: Err(vec![StackBlocker {
                    error: StackBoundError::UnknownRoot(root.to_owned()),
                    path: Vec::new(),
                    members: Vec::new(),
                }]),
            };
        };
        let (order, parent) = breadth_first(&index, root_key);
        let reachable: Vec<String> = order.iter().map(|n| (*n).to_owned()).collect();
        let path_to = |name: &str| -> Vec<String> {
            let mut path = vec![name.to_owned()];
            let mut cur = name;
            while let Some(&p) = parent.get(cur) {
                path.push(p.to_owned());
                cur = p;
            }
            path.reverse();
            path
        };

        // Strongly connected components, to report a recursive group once.
        let mut scc = Scc::default();
        strong_connect(root_key, &index, &mut scc);
        let mut walk = Walk::default();
        find_blockers(root_key, &index, assume, &scc, &mut walk);

        if walk.found.is_empty() {
            let mut memo: BTreeMap<String, Solved> = BTreeMap::new();
            let result = match solve(root, &index, assume, &mut memo, &mut Vec::new()) {
                Ok(depth) => Ok(StackBound { bytes: depth, path: deepest_path(root, &index, &memo) }),
                // Unreachable: the walk above finds every problem `solve` can.
                Err(error) => Err(vec![StackBlocker { error, path: Vec::new(), members: Vec::new() }]),
            };
            return StackAnalysis { root: root.to_owned(), reachable, result };
        }

        let position: BTreeMap<&str, usize> = order.iter().enumerate().map(|(i, n)| (*n, i)).collect();
        let blockers = walk
            .found
            .into_iter()
            .map(|error| {
                let (path, members) = match &error {
                    StackBoundError::Recursion { cycle } => {
                        let comp = scc.comp[cycle[0].as_str()];
                        let mut members: Vec<&str> =
                            scc.comp.iter().filter(|&(_, &c)| c == comp).map(|(n, _)| *n).collect();
                        members.sort_by_key(|n| position[n]);
                        (path_to(&cycle[0]), members.into_iter().map(str::to_owned).collect())
                    }
                    StackBoundError::IndirectCall { function } | StackBoundError::DynamicAlloca { function } => {
                        (path_to(function), Vec::new())
                    }
                    StackBoundError::UnknownCallee { caller, callee } => {
                        let mut path = path_to(caller);
                        path.push(callee.clone());
                        (path, Vec::new())
                    }
                    StackBoundError::UnknownRoot(_) => (Vec::new(), Vec::new()),
                };
                StackBlocker { error, path, members }
            })
            .collect();
        StackAnalysis { root: root.to_owned(), reachable, result: Err(blockers) }
    }

    /// The functions of the report reachable from `root` through direct calls
    /// (`root` included), nearest first: breadth-first, callees in call order.
    /// Empty when `root` is not in the report. Indirect calls are not followed
    /// (their targets are unknown).
    pub fn reachable(&self, root: &str) -> Vec<String> {
        let index = self.index();
        match index.get_key_value(root) {
            Some((&root, _)) => breadth_first(&index, root).0.into_iter().map(str::to_owned).collect(),
            None => Vec::new(),
        }
    }

    /// This report restricted to the functions reachable from `root` (see
    /// [`StackReport::reachable`]), in definition order: e.g. to print the
    /// table without dead functions.
    pub fn reachable_from(&self, root: &str) -> StackReport {
        let live: BTreeSet<String> = self.reachable(root).into_iter().collect();
        StackReport { functions: self.functions.iter().filter(|u| live.contains(&u.name)).cloned().collect() }
    }

    /// A shortest call path from `root` to `target` (both included) through
    /// direct calls, or `None` when `target` is not reachable from `root`. Of
    /// several shortest paths, the one met first breadth-first (callees in
    /// call order) is returned.
    pub fn call_path(&self, root: &str, target: &str) -> Option<Vec<String>> {
        let index = self.index();
        let (&root, _) = index.get_key_value(root)?;
        let (_, parent) = breadth_first(&index, root);
        if target != root && !parent.contains_key(target) {
            return None;
        }
        let mut path = vec![target.to_owned()];
        let mut cur = target;
        while let Some(&p) = parent.get(cur) {
            path.push(p.to_owned());
            cur = p;
        }
        path.reverse();
        Some(path)
    }

    /// The functions by name (a later duplicate replaces an earlier one).
    fn index(&self) -> BTreeMap<&str, &StackUsage> {
        self.functions.iter().map(|u| (u.name.as_str(), u)).collect()
    }
}

/// The result of [`StackReport::analyze_from`]: the bound, or every reason
/// there is none, plus what is reachable from the root.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StackAnalysis {
    /// The analyzed root function.
    pub root: String,
    /// The functions of the report reachable from the root through direct
    /// calls, root first, nearest first (see [`StackReport::reachable`]).
    pub reachable: Vec<String>,
    /// The worst-case bound with one deepest path, or every reason no bound
    /// exists (never empty), in a deterministic order.
    pub result: Result<StackBound, Vec<StackBlocker>>,
}

impl StackAnalysis {
    /// The proven bound, if there is one.
    pub fn bound(&self) -> Option<&StackBound> {
        self.result.as_ref().ok()
    }

    /// Every reason no bound exists (empty when there is a bound).
    pub fn blockers(&self) -> &[StackBlocker] {
        match &self.result {
            Ok(_) => &[],
            Err(blockers) => blockers,
        }
    }

    /// Whether `name` is reachable from the root.
    pub fn is_reachable(&self, name: &str) -> bool {
        self.reachable.iter().any(|n| n == name)
    }

    /// The bound, or the first reason there is none (what
    /// [`StackReport::worst_case_depth`] returns).
    pub fn into_result(self) -> Result<StackBound, StackBoundError> {
        self.result.map_err(|blockers| {
            blockers.into_iter().next().map(|b| b.error).unwrap_or(StackBoundError::UnknownRoot(self.root))
        })
    }
}

impl fmt::Display for StackAnalysis {
    /// One line with the bound and its deepest path, or a header line followed
    /// by one indented line per reason:
    ///
    /// ```text
    /// worst-case stack from 'main': no bound (1 reason):
    ///   main -> main.main -> main.fib: 'main.fib' calls itself
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let root = &self.root;
        match &self.result {
            Ok(bound) => {
                write!(f, "worst-case stack from '{root}': {} bytes ({})", bound.bytes, bound.path.join(" -> "))
            }
            Err(blockers) => {
                let n = blockers.len();
                write!(f, "worst-case stack from '{root}': no bound ({n} reason{}):", if n == 1 { "" } else { "s" })?;
                for b in blockers {
                    write!(f, "\n  {b}")?;
                }
                Ok(())
            }
        }
    }
}

/// One reason a stack bound does not exist, with how the root reaches it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct StackBlocker {
    /// What blocks the bound. A [`StackBoundError::Recursion`] cycle is the one
    /// the depth-first walk closed first; [`StackBlocker::members`] has the
    /// whole recursive group.
    pub error: StackBoundError,
    /// A shortest call path from the root, root first, to the function at
    /// fault: the recursion's first cycle function, the indirect caller, the
    /// `dyn_alloca` user, or (for an unknown callee) the caller followed by the
    /// unknown callee. Empty for [`StackBoundError::UnknownRoot`].
    pub path: Vec<String>,
    /// For a recursion: every function of its strongly connected component
    /// (the functions that can reach each other), nearest to the root first.
    /// Empty otherwise.
    pub members: Vec<String>,
}

impl fmt::Display for StackBlocker {
    /// `path: reason`, e.g. `main -> main.main -> main.fib: 'main.fib' calls
    /// itself` or `main -> f: recursion f -> g -> f`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.path.is_empty() {
            write!(f, "{}: ", self.path.join(" -> "))?;
        }
        match &self.error {
            StackBoundError::Recursion { cycle } if cycle.len() == 2 && self.members.len() <= 1 => {
                write!(f, "'{}' calls itself", cycle[0])
            }
            StackBoundError::Recursion { cycle } => {
                write!(f, "recursion {}", cycle.join(" -> "))?;
                if self.members.len() + 1 > cycle.len() {
                    write!(f, " (recursive group: {})", self.members.join(", "))?;
                }
                Ok(())
            }
            other => write!(f, "{other}"),
        }
    }
}

/// The functions reachable from `root`, breadth-first (callees in call order),
/// and each one's predecessor on the first shortest path found.
fn breadth_first<'a>(
    index: &BTreeMap<&'a str, &'a StackUsage>,
    root: &'a str,
) -> (Vec<&'a str>, BTreeMap<&'a str, &'a str>) {
    let mut order = vec![root];
    let mut parent: BTreeMap<&'a str, &'a str> = BTreeMap::new();
    let mut i = 0;
    while let Some(&cur) = order.get(i) {
        i += 1;
        for callee in &index[cur].direct_callees {
            let Some((&callee, _)) = index.get_key_value(callee.as_str()) else { continue };
            if callee != root && !parent.contains_key(callee) {
                parent.insert(callee, cur);
                order.push(callee);
            }
        }
    }
    (order, parent)
}

/// Tarjan's strongly-connected-components state.
#[derive(Default)]
struct Scc<'a> {
    counter: usize,
    num: BTreeMap<&'a str, usize>,
    low: BTreeMap<&'a str, usize>,
    stack: Vec<&'a str>,
    on_stack: BTreeSet<&'a str>,
    /// Each visited function's component number.
    comp: BTreeMap<&'a str, usize>,
    comps: usize,
}

fn strong_connect<'a>(v: &'a str, index: &BTreeMap<&'a str, &'a StackUsage>, s: &mut Scc<'a>) {
    let n = s.counter;
    s.counter += 1;
    s.num.insert(v, n);
    s.low.insert(v, n);
    s.stack.push(v);
    s.on_stack.insert(v);
    for callee in &index[v].direct_callees {
        let Some((&w, _)) = index.get_key_value(callee.as_str()) else { continue };
        if !s.num.contains_key(w) {
            strong_connect(w, index, s);
            let low = s.low[v].min(s.low[w]);
            s.low.insert(v, low);
        } else if s.on_stack.contains(w) {
            let low = s.low[v].min(s.num[w]);
            s.low.insert(v, low);
        }
    }
    if s.low[v] == s.num[v] {
        while let Some(w) = s.stack.pop() {
            s.on_stack.remove(w);
            s.comp.insert(w, s.comps);
            if w == v {
                break;
            }
        }
        s.comps += 1;
    }
}

/// The depth-first walk collecting every blocker.
#[derive(Default)]
struct Walk<'a> {
    visited: BTreeSet<&'a str>,
    active: Vec<&'a str>,
    reported_comps: BTreeSet<usize>,
    found: Vec<StackBoundError>,
}

/// Walk from `name` exactly as [`solve`] does, but record every problem
/// instead of stopping at the first (so the first recorded is `solve`'s).
fn find_blockers<'a>(
    name: &'a str,
    index: &BTreeMap<&'a str, &'a StackUsage>,
    assume: &StackAssumptions,
    scc: &Scc<'a>,
    w: &mut Walk<'a>,
) {
    w.visited.insert(name);
    w.active.push(name);
    let u = index[name];
    if u.dynamic_alloca && !assume.dynamic.contains_key(name) {
        w.found.push(StackBoundError::DynamicAlloca { function: name.to_owned() });
    }
    if u.indirect_calls && !assume.indirect.contains_key(name) {
        w.found.push(StackBoundError::IndirectCall { function: name.to_owned() });
    }
    for callee in &u.direct_callees {
        match index.get_key_value(callee.as_str()) {
            Some((&callee, _)) => {
                if let Some(pos) = w.active.iter().position(|a| *a == callee) {
                    if w.reported_comps.insert(scc.comp[callee]) {
                        let mut cycle: Vec<String> = w.active[pos..].iter().map(|a| (*a).to_owned()).collect();
                        cycle.push(callee.to_owned());
                        w.found.push(StackBoundError::Recursion { cycle });
                    }
                } else if !w.visited.contains(callee) {
                    find_blockers(callee, index, assume, scc, w);
                }
            }
            None if !assume.external.contains_key(callee) => {
                w.found.push(StackBoundError::UnknownCallee { caller: name.to_owned(), callee: callee.clone() });
            }
            None => {}
        }
    }
    w.active.pop();
}

/// The deepest path from `root` that [`solve`] chose, root first.
fn deepest_path(root: &str, index: &BTreeMap<&str, &StackUsage>, memo: &BTreeMap<String, Solved>) -> Vec<String> {
    let mut path = vec![root.to_owned()];
    let mut cur = root.to_owned();
    while let Some(next) = memo.get(&cur).and_then(|s| s.next.clone()) {
        path.push(next.clone());
        if !index.contains_key(next.as_str()) {
            break; // an external or indirect leaf
        }
        cur = next;
    }
    path
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

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Lode's recursive program from issue #6, plus a dead library function.
    fn fib_report() -> StackReport {
        report(vec![
            usage("main", 16, &["main.main"]),
            usage("main.fib", 32, &["main.fib"]),
            usage("main.main", 80, &["main.fib"]),
            usage("std/io.eprint", 48, &["write"]),
        ])
    }

    #[test]
    fn the_issue_example_names_the_path_to_the_recursion() {
        let r = fib_report();
        let a = r.analyze_from("main", &StackAssumptions::new());
        assert_eq!(a.reachable, ["main", "main.main", "main.fib"]);
        assert!(!a.is_reachable("std/io.eprint"));
        assert_eq!(
            a.blockers(),
            [StackBlocker {
                error: StackBoundError::Recursion { cycle: names(&["main.fib", "main.fib"]) },
                path: names(&["main", "main.main", "main.fib"]),
                members: names(&["main.fib"]),
            }]
        );
        assert_eq!(
            a.to_string(),
            "worst-case stack from 'main': no bound (1 reason):\n  \
             main -> main.main -> main.fib: 'main.fib' calls itself"
        );
        // The old API still reports the cycle.
        assert_eq!(
            r.worst_case_depth("main", &StackAssumptions::new()),
            Err(StackBoundError::Recursion { cycle: names(&["main.fib", "main.fib"]) })
        );
        // The table restricted to the root drops the dead function.
        let t = r.reachable_from("main").to_string();
        assert!(t.contains("main.fib") && !t.contains("std/io.eprint"), "{t}");
        assert!(r.to_string().contains("std/io.eprint"));
    }

    #[test]
    fn every_blocker_is_reported_with_its_path() {
        let mut ind = usage("ind", 16, &[]);
        ind.indirect_calls = true;
        let mut dy = usage("dy", 16, &[]);
        dy.dynamic_alloca = true;
        let r = report(vec![
            usage("main", 16, &["a", "r1", "ind", "dy", "ext"]),
            usage("a", 16, &["r2"]),
            usage("r1", 16, &["r1"]),
            usage("r2", 16, &["r3"]),
            usage("r3", 16, &["r2"]),
            ind,
            dy,
        ]);
        let a = r.analyze_from("main", &StackAssumptions::new());
        assert!(a.bound().is_none());
        let b = a.blockers();
        assert_eq!(b.len(), 5, "{a}");
        assert_eq!(b[0].error, StackBoundError::Recursion { cycle: names(&["r2", "r3", "r2"]) });
        assert_eq!(b[0].path, ["main", "a", "r2"]);
        assert_eq!(b[0].members, ["r2", "r3"]);
        assert_eq!(b[1].error, StackBoundError::Recursion { cycle: names(&["r1", "r1"]) });
        assert_eq!(b[1].path, ["main", "r1"]);
        assert_eq!(b[2].error, StackBoundError::IndirectCall { function: "ind".into() });
        assert_eq!(b[2].path, ["main", "ind"]);
        assert_eq!(b[3].error, StackBoundError::DynamicAlloca { function: "dy".into() });
        assert_eq!(b[3].path, ["main", "dy"]);
        assert_eq!(b[4].error, StackBoundError::UnknownCallee { caller: "main".into(), callee: "ext".into() });
        assert_eq!(b[4].path, ["main", "ext"]);
        let lines: Vec<String> = b.iter().map(|b| b.to_string()).collect();
        assert_eq!(
            lines,
            [
                "main -> a -> r2: recursion r2 -> r3 -> r2",
                "main -> r1: 'r1' calls itself",
                "main -> ind: 'ind' makes an indirect call with no assumed bound",
                "main -> dy: 'dy' uses dyn_alloca with no assumed bound",
                "main -> ext: 'main' calls 'ext', whose stack usage is unknown",
            ]
        );
        assert!(a.to_string().starts_with("worst-case stack from 'main': no bound (5 reasons):\n  main -> a"));
        // The old API returns the first of them.
        assert_eq!(r.worst_case_depth("main", &StackAssumptions::new()), Err(b[0].error.clone()));
        // Assumptions remove the blockers they bound; recursion stays.
        let assume = StackAssumptions::new().indirect("ind", 64).dynamic("dy", 64).external("ext", 64);
        let a = r.analyze_from("main", &assume);
        assert_eq!(a.blockers().len(), 2, "{a}");
    }

    #[test]
    fn blockers_in_dead_code_are_ignored() {
        let mut ind = usage("dead_ind", 16, &[]);
        ind.indirect_calls = true;
        let mut dy = usage("dead_dy", 16, &[]);
        dy.dynamic_alloca = true;
        let r = report(vec![
            usage("dead_rec", 16, &["dead_rec", "main"]),
            usage("main", 32, &["leaf"]),
            ind,
            usage("leaf", 48, &[]),
            dy,
            usage("dead_ext", 16, &["ext", "dead_ind", "dead_dy"]),
        ]);
        let a = r.analyze_from("main", &StackAssumptions::new());
        assert_eq!(a.bound(), Some(&StackBound { bytes: 80, path: names(&["main", "leaf"]) }));
        assert!(a.blockers().is_empty());
        assert_eq!(a.to_string(), "worst-case stack from 'main': 80 bytes (main -> leaf)");
        assert_eq!(r.reachable("main"), ["main", "leaf"]);
        let live = r.reachable_from("main");
        assert_eq!(live.functions().iter().map(|u| u.name.as_str()).collect::<Vec<_>>(), ["main", "leaf"]);
        // From another root, the same functions block.
        assert_eq!(r.analyze_from("dead_ext", &StackAssumptions::new()).blockers().len(), 3);
        assert_eq!(r.reachable("dead_rec"), ["dead_rec", "main", "leaf"]);
        assert!(r.reachable("nope").is_empty());
        assert_eq!(r.call_path("dead_rec", "leaf"), Some(names(&["dead_rec", "main", "leaf"])));
        assert_eq!(r.call_path("main", "main"), Some(names(&["main"])));
        assert_eq!(r.call_path("main", "dead_rec"), None);
    }

    #[test]
    fn mutual_recursion_is_one_cycle_with_its_members() {
        let r = report(vec![
            usage("main", 16, &["f", "g"]),
            usage("f", 16, &["g"]),
            usage("g", 16, &["f", "h"]),
            usage("h", 16, &["g", "f"]),
        ]);
        let a = r.analyze_from("main", &StackAssumptions::new());
        let b = a.blockers();
        assert_eq!(b.len(), 1, "{a}");
        assert_eq!(b[0].error, StackBoundError::Recursion { cycle: names(&["f", "g", "f"]) });
        assert_eq!(b[0].path, ["main", "f"]);
        assert_eq!(b[0].members, ["f", "g", "h"]);
        assert_eq!(b[0].to_string(), "main -> f: recursion f -> g -> f (recursive group: f, g, h)");
        // A two-function cycle needs no group.
        let r = report(vec![usage("main", 16, &["f"]), usage("f", 16, &["g"]), usage("g", 16, &["f"])]);
        let a = r.analyze_from("main", &StackAssumptions::new());
        assert_eq!(a.blockers().len(), 1);
        assert_eq!(a.blockers()[0].members, ["f", "g"]);
        assert_eq!(a.blockers()[0].to_string(), "main -> f: recursion f -> g -> f");
    }

    #[test]
    fn blockers_come_in_a_deterministic_order() {
        let build = |order: &[&str]| {
            let mut us = vec![usage("main", 16, order)];
            for n in ["x", "y", "z"] {
                let mut u = usage(n, 16, &[]);
                u.dynamic_alloca = true;
                us.push(u);
            }
            report(us)
        };
        let functions = |r: &StackReport| -> Vec<String> {
            r.analyze_from("main", &StackAssumptions::new())
                .blockers()
                .iter()
                .map(|b| b.path.last().unwrap().clone())
                .collect()
        };
        let r = build(&["y", "z", "x"]);
        assert_eq!(functions(&r), ["y", "z", "x"]);
        assert_eq!(r.analyze_from("main", &StackAssumptions::new()), r.analyze_from("main", &StackAssumptions::new()));
        assert_eq!(functions(&build(&["x", "y", "z"])), ["x", "y", "z"]);
    }

    #[test]
    fn the_old_api_matches_the_analysis() {
        let r = report(vec![
            usage("main", 32, &["a", "b"]),
            usage("a", 100, &["leaf"]),
            usage("b", 16, &["leaf"]),
            usage("leaf", 48, &[]),
        ]);
        let none = StackAssumptions::new();
        let a = r.analyze_from("main", &none);
        assert_eq!(a.bound(), r.worst_case_depth("main", &none).ok().as_ref());
        assert_eq!(a.clone().into_result(), r.worst_case_depth("main", &none));
        let unknown = r.analyze_from("nope", &none);
        assert!(unknown.reachable.is_empty());
        assert_eq!(
            unknown.to_string(),
            "worst-case stack from 'nope': no bound (1 reason):\n  no function 'nope' in the stack report"
        );
        assert_eq!(unknown.into_result(), Err(StackBoundError::UnknownRoot("nope".into())));
    }
}
