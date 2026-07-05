# LatticeFoundry Roadmap

This document is the plan of record for building LatticeFoundry from an empty
repository into a working compiler construction framework. It describes the
architecture, the guiding constraints, and a phased build-out with concrete
deliverables and exit criteria for each phase.

The roadmap is a living document: phases are refined as earlier ones land, and
the exit criteria are what "done" means for each phase.

> **Status.** Phases **0–9 are complete** (milestone **M5** reached) and Phase
> **10 is in progress**. `lf build foo.lf -o foo` compiles a LatticeFoundry IR
> module to a static ELF64 executable that runs on the bare Linux x86-64 kernel —
> no libc, no system linker, 100% LatticeFoundry code — and a **JIT** runs the
> same code in-process. The verification bets are all live and interlocking:
> **B1** (semantics-first opcodes), **B2** (z3rs-checked refinement, used by SCCP
> and the e-graph), **B3** (proof-carrying certificates), **B4** (equality-
> saturation optimizer), **B5** (z3rs superoptimizer), **B8** (one lattice engine
> + four sound domains), **B9** (cost model). ~324 tests, every commit green
> (build/test/clippy clean; `unsafe` only in the JIT's `exec_mem`; only our own
> crates). Remaining Phase 10 breadth: a second target (AArch64/RISC-V), DWARF
> debug info, LTO, sanitizers — additive, beyond the working compiler.

---

## 1. Vision

LatticeFoundry is the reusable machinery a compiler needs *below the front
end*: once a language implementation has produced our intermediate
representation, LatticeFoundry takes it the rest of the way to optimized native
code and a linked executable. It is delivered as a single library plus a family
of small driver binaries.

The scope is deliberately close to what a framework like LLVM covers, but the
design and implementation are entirely our own.

## 2. Principles & constraints

These are hard constraints, not aspirations. They shape every phase.

1. **Clean room.** Every artifact is designed and written from first
   principles. We do **not** copy, transliterate, or line-by-line "translate"
   source, textual IR grammars, encoding tables, or file-format layouts from
   any third-party compiler, assembler, linker, or solver. Published
   *standards* we must interoperate with (the ELF specification, an
   instruction-set manual, IEEE-754) may be implemented from the spec — that is
   interoperability, not derivation. General computer-science knowledge (SSA,
   dominance, graph coloring, DPLL) is used freely.
2. **Only our own crates.** The dependency graph contains only our own focused,
   clean-room library crates — nothing from third parties, no `-sys` crates, no
   C. See §3.1 for the current set. Utilities we would normally pull from the
   ecosystem (bit-vectors, hashing, arena allocators, arg parsing) are either
   provided by one of our crates or written here.
3. **Pure, safe Rust.** `unsafe` is a `warn` lint. It is permitted only where a
   safety invariant genuinely cannot be expressed in the type system, and every
   use is documented with the invariant it upholds.
4. **Design our own formats.** Our textual IR (`.lf`), binary IR (`.lfb`), and
   native object format (`.lfo`) are our designs. Where we emit or read a
   *standard* external format (ELF, DWARF), we implement it against its public
   spec.
5. **Test as we build.** Every phase ships with unit tests; from Phase 2 onward
   we maintain golden-file and round-trip tests, and from Phase 5 onward,
   execution tests that actually run generated code.

### 2.1 Not a workspace

LatticeFoundry is a **single Cargo package**, not a workspace: one library
(`src/lib.rs`) and several binaries auto-discovered from `src/bin/`. The `lf-`
tools are binaries of this one package, not separate member crates.

### 2.2 Design tenets & bets

What makes LatticeFoundry more than a re-implementation lives in two companion
documents, and the *committed* bets below are threaded into the phases:

- [`docs/design-tenets.md`](docs/design-tenets.md) — the opinionated
  commitments (semantics-first, correctness-by-verified-refinement, one lattice
  engine, content-addressed core), the verification tiers, and the full bets
  register (*committed / staged / moonshot*).
- [`docs/ir-design.md`](docs/ir-design.md) — the concrete IR decisions (block
  arguments over φ-nodes, poison + freeze with no `undef`, opaque pointers with
  explicit offset addressing, a unified flag model, machine-checkable opcode
  semantics).

The two *committed* bets on the critical path are **B1** (the opcode table is a
formal semantics) and **B8** (a single sound abstract-interpretation engine);
**B2** (every optimization is a checked refinement) begins at Phase 2. Staged
and moonshot bets (region form, e-graphs, superoptimization, verified lowering,
provenance types) are scheduled but deliberately off the M5 critical path.

## 3. Architecture overview

The compilation pipeline, and the module of the library that owns each stage:

```
            ┌──────────────────────────────────────────────────────────┐
 front end  │  (out of scope — languages target our IR)                │
 ───────────┼──────────────────────────────────────────────────────────┤
            │                                                          │
   .lf text │  ir            typed SSA IR: module → function → block   │
            │  ir::parse     textual & binary (bitcode) readers/writers │
            │  verify        structural + type invariants  ──► z3rs     │
            │                                                          │
            │  pass          pass/analysis manager, fixpoint driver     │
            │  analysis      dominators, CFG, loops, liveness, aliasing │
            │  transform     mem2reg, DCE, const-fold, GVN, inline, ... │
            │                                                          │
            │  codegen       IR → MIR, isel, regalloc, scheduling       │
            │  mc            instruction encoding, relocations, objects │
            │  target        per-architecture description + lowering    │
            │                                                          │
   .lfo/ELF │  link          symbol resolution, relocation, layout      │
            └──────────────────────────────────────────────────────────┘

binaries: lf (umbrella)   lf-opt   lf-as   lf-dis   lf-ld
```

### Module map (all within the one `latticefoundry` library)

| Module                    | Responsibility                                          |
| ------------------------- | ------------------------------------------------------- |
| `support`                 | interning, arenas, small ADTs; numeric core (`puremp`)  |
| `ir`                      | IR data model, type system, builder, text/binary format |
| `verify`                  | well-formedness checking; SMT-backed condition checks   |
| `pass`                    | pass manager, analysis caching, pipeline description     |
| `codegen`                 | target-independent lowering to machine IR               |
| `mc`                      | machine-code encoding and object emission               |
| `target`                  | target registry and per-target tables                   |
| `link`                    | linker core                                             |

### Binaries (`src/bin/`)

| Binary   | Role                                             |
| -------- | ------------------------------------------------ |
| `lf`     | umbrella driver; ties the tools together         |
| `lf-opt` | load IR, run a pass pipeline, write IR back       |
| `lf-as`  | assembler: target assembly → relocatable object   |
| `lf-dis` | disassembler: machine code → assembly             |
| `lf-ld`  | linker: objects/archives → executable / shared obj |

### 3.1 Dependencies — our own crates only

LatticeFoundry reuses focused, clean-room library crates from our own
ecosystem. It does **not** reinvent them:

| Crate                                                    | Used for                                                   |
| -------------------------------------------------------- | ---------------------------------------------------------- |
| [`puremp`](https://github.com/KarpelesLab/puremp)        | arbitrary-precision integers/rationals/floats for IR constants and codegen constant math |
| [`z3rs`](https://github.com/KarpelesLab/z3rs)            | SMT solving for the verifier and correctness-guarded rewrites (Phase 9) |

Additional own-crates may be adopted as later phases need them, e.g.
[`compcol`](https://github.com/KarpelesLab/compcol) for compressing binary IR /
objects. **Broad tools are not taken as dependencies:** code from wide own-tools
such as [`univdreams`](https://github.com/KarpelesLab/univdreams) (a
decompiler/compiler/emulator that round-trips ELF/PE/Mach-O) may be *adapted
into this tree* for object-format handling, but that crate — being an
emulator/decompiler — is not pulled in as a dependency.

### Naming conventions

- Binaries are prefixed `lf-` (`lf-ld`, `lf-as`, ...); `lf` is the umbrella.
- Our textual IR uses the extension `.lf`; binary IR uses `.lfb`; our native
  object format uses `.lfo`.
- Public id types are small `Copy` newtypes (`FuncId`, `BlockId`, `ValueId`).

## 4. Phased plan

Phases are ordered by dependency, not by calendar. Each lists deliverables and
the exit criteria that define completion. **Bold** phases are the critical path
to "compile a function to a running native executable" (the first end-to-end
milestone, Phase 8).

### Phase 0 — Foundations & scaffolding  ✅ *in progress*

Bring up the package and the low-level support layer.

- Single-package layout (lib + `src/bin/` tools), edition 2024, only-our-crates
  policy, shared lints.
- `support`: string interner, typed-index/arena primitives; the `puremp`
  numeric core wired in (we do **not** write a bespoke bignum).
- Driver skeletons for all five binaries (`--version`/`--help`).
- `build` / `test` / `clippy` all green.

*Exit:* `cargo build/test/clippy` clean; every binary runs.
*Next in this phase:* a general arena allocator, a deterministic hash map, and a
diagnostics type with source spans.

### **Phase 1 — Core IR**  *(carries bets B1, T5)*

The typed SSA data model and the programmatic builder — designed *semantics-first*
(see [ir-design](docs/ir-design.md)).

- Complete type system: integers, floats, opaque pointers, arrays, structs,
  vectors, function types (interned/hash-consed from day one, T5).
- Full value model: instruction results, **block parameters** (block arguments,
  not φ-nodes), constants (wide integers via `puremp`), global values.
- **Poison + freeze value semantics, no `undef`** — decided before the opcode
  table, so every op has a poison rule (B1).
- Complete opcode table, each op authored **as a machine-checkable reference
  semantics** (B1): arithmetic/bitwise, comparisons, memory (`load`/`store`/
  `alloca`/`ptr_add`), control flow (`br`/`switch`/`ret`/`unreachable`), casts,
  `call`, `select`, `freeze`. Unified flag model (`nsw`/`nuw`/`exact`/fast-math),
  flag violation ⇒ poison.
- `IrBuilder` with SSA construction helpers (incl. `struct_field`/`array_elem`
  offset helpers); use/def tracking and value replacement.
- Content-addressed substrate: id/arena-based, no interior pointers, pure nodes
  hash-consable (T5) so B6/B7 can be turned on later.

*Exit:* build non-trivial functions in memory (loops, calls, branches);
use/def lists are consistent under mutation; each opcode has a semantics that
`z3rs` can consume; covered by unit tests.

### **Phase 2 — Textual & binary format, and the verifier**  *(carries bet B2, first cut)*

Make IR persistable and checkable.

- `.lf` textual **printer** and **parser** (our own grammar, block parameters
  explicit) with round-trip fidelity.
- `.lfb` binary format (compact, versioned, content-addressed friendly) with
  round-trip fidelity.
- Verifier — **structural + semantic**: single terminator per block, dominance
  of uses by defs, type agreement, block-argument arity/typing, well-typed
  constants (`Structural` tier), **plus** the first `Refinement`-tier check: a
  single rewrite emits a B2 refinement obligation discharged by `z3rs`.
- Wire the format + verifier + tier selection into `lf-opt` (load → verify →
  optionally check-refinement → print).

*Exit:* golden-file tests for the printer; `parse(print(m)) == m` and
`read(write(m)) == m` for a corpus; verifier rejects a suite of malformed
modules with precise diagnostics; one rewrite is `Refinement`-checked end to end.

### Phase 3 — Analysis: one lattice engine  *(is bet B8; carries B7 substrate)*

The analysis layer **is** a single abstract-interpretation engine, not a drawer
of bespoke analyses (tenet T4).

- A generic monotone fixpoint solver (sparse over SSA def-use) parameterized by
  an `AbstractDomain` trait (⊥, ⊑, join, widening, concretization γ).
- Domains implemented against that engine: constants, integer ranges,
  known-bits, nullness, simple alias/points-to. Each domain's transfer functions
  are **`z3rs`-checked for soundness** against the B1 opcode semantics.
- Structural analyses the engine and passes need: CFG, dominator/post-dominator
  trees, dominance frontier, natural loops, use-def/def-use.
- Pass manager (module/function granularity, textual pipeline spec) and analysis
  manager with dependency tracking, caching, and invalidation — designed for
  parallel/incremental operation (T6) on the content-addressed core (B7).

*Exit:* the fixpoint engine reproduces each domain's results and matches a
brute-force oracle on random CFGs; transfer-function soundness checks pass;
invalidation verified (a mutating pass forces recomputation).

### Phase 4 — Optimizations  *(grows bets B4, B9 on B2)*

A useful baseline of optimizations, verified by construction.

- Structural transforms: `mem2reg` (promote memory to SSA via dominance
  frontiers), aggressive/dead-store DCE, control-flow simplification, inlining
  with a cost model, loop-invariant code motion.
- **Local/algebraic optimization via equality saturation (B4):** constant
  folding, strength reduction, GVN/CSE, and peepholes expressed as **B2-verified
  rewrite rules** over an e-graph, with best-program extraction under a **cost
  lattice (B9)** — sidestepping phase ordering for this class.
- Every rewrite (structural or e-graph) carries a `Refinement`-tier obligation
  and stays valid under the verifier.

*Exit:* each transform has before/after golden tests, is `Refinement`-checked,
and preserves verifier validity; an `-O1`/`-O2` pipeline measurably shrinks a
benchmark corpus; the e-graph rule set is `z3rs`-verified.

### **Phase 5 — Target-independent code generation**

Lower optimized IR toward machine instructions.

- Machine IR (MIR): virtual registers, machine basic blocks, target opcodes.
- Instruction selection framework (pattern-based lowering from IR to MIR).
- Register allocation (start with a correct linear-scan; graph-coloring later).
- Instruction scheduling; prologue/epilogue and stack-frame construction;
  calling-convention lowering.

*Exit:* MIR for a target verifies and, once Phase 6/7 land, assembles and runs.

### **Phase 6 — Machine-code layer**

Turn instructions into bytes and objects.

- Instruction encoder/decoder framework (drives `lf-as` and `lf-dis`).
- Relocations, sections, symbols; our `.lfo` relocatable object format.
- ELF64 relocatable **writer** implemented from the ELF spec (for interop).
  Object-format plumbing may adapt code from our own `univdreams` (kept
  in-tree, not a dependency); compression of `.lfb`/`.lfo` may use `compcol`.

*Exit:* `lf-as` assembles to `.lfo`/ELF; `lf-dis` round-trips encode∘decode on
a fuzzed instruction corpus; objects are consumable by Phase 8.

### **Phase 7 — Targets**

Concrete backends. x86-64 is the bring-up target.

- **x86-64**: register file, System V ABI, integer + SSE, encodings, isel rules.
- AArch64: AAPCS64, base integer + FP/SIMD.
- RISC-V (RV64GC): base + common extensions.

*Exit (per target):* the execution test suite passes on real hardware/emulator
for the target's ABI; encodings match the architecture manual.

### **Phase 8 — Linker & first end-to-end**

Produce a runnable program.

- `link` core: multi-object symbol resolution, archive (`.a`-style) handling,
  relocation processing, section/segment layout, entry-point setup.
- ELF64 executable **writer**; static linking first.
- **Milestone: `lf` compiles a non-trivial `.lf` module to a native executable
  that runs and returns the expected result.**

*Exit:* end-to-end tests compile → link → execute across the Phase 7 targets.

### Phase 9 — Certified tier: proof-carrying IR  *(is bet B3)*

Deepen the verification story from "checked in CI" (B2, already in use since
Phase 2) to "certificate-checked" (`z3rs` is developed separately; we integrate,
not build it).

- Modules carry **certificates** of the transformations applied; a small trusted
  checker + `z3rs` re-validates a whole pipeline without trusting the optimizer
  (the `Certified` tier).
- Whole-run translation validation over an optimization pipeline; certificate
  caching so release builds check rather than re-prove.

*Exit:* a pipeline run emits certificates that the checker validates; the
trusted computing base is reduced to the checker plus `z3rs`.

### Phase 10 — Beyond the core

Depth once the pipeline is solid.

- JIT execution engine; dynamic (shared-object) linking.
- Debug info (DWARF emission from the spec) and source-level line tables.
- Link-time optimization over binary IR; profile-guided optimization hooks.
- Superoptimization / peephole synthesis driven by `z3rs`.
- Sanitizer instrumentation passes; richer alias analysis.

*Exit:* JIT runs the execution suite; debuggers show source lines for compiled
programs.

## 5. Testing strategy

- **Unit tests** in every module, from Phase 0.
- **Round-trip tests** for every serializer/deserializer (text, binary, object).
- **Golden-file tests** for printers and pass output; diffs are reviewable.
- **Property/oracle tests** for analyses (compare against brute force on random
  graphs) and for the encoder (`decode(encode(i)) == i`).
- **Execution tests** from Phase 5/8: compile and run, check observable output.

## 6. Non-goals (for now)

- Language front ends (parsers, type checkers for a source language). Languages
  target our IR; producing that IR is their concern.
- A stable public API or ABI before the pipeline is end-to-end.
- Windows/macOS object and executable formats before ELF is solid.
- Matching the performance of a mature production compiler; correctness and a
  clean, well-tested design come first.

## 7. Milestone summary

| Milestone | Meaning                                                 | Phase | Status |
| --------- | ------------------------------------------------------- | ----- | ------ |
| M0        | Package builds; drivers run                              | 0     | ✅ done |
| M1        | Build & verify SSA IR in memory                          | 1–2   | ✅ done |
| M2        | Parse/print/round-trip `.lf`; verifier rejects bad IR    | 2     | ✅ done |
| M3        | Optimization pipeline, refinement-checked                | 3–4   | ✅ done |
| M4        | Emit assembled objects for x86-64                        | 5–7   | ✅ done |
| **M5**    | **Compile `.lf` → native executable that runs**          | 8     | ✅ **done** |
| M6        | Certified tier: proof-carrying pipeline                  | 9     | ✅ done |
| M7        | JIT, debug info, LTO                                     | 10    | 🔶 JIT + superopt done; DWARF/LTO/more targets pending |
