# LatticeFoundry Roadmap

This document is the plan of record for building LatticeFoundry from an empty
repository into a working compiler construction framework. It describes the
architecture, the guiding constraints, and a phased build-out with concrete
deliverables and exit criteria for each phase.

The roadmap is a living document: phases are refined as earlier ones land, and
the exit criteria are what "done" means for each phase.

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
   source code, textual IR grammars, encoding tables, or file-format layouts
   from any existing compiler, assembler, linker, or solver. Published
   *standards* that we must interoperate with (e.g. the ELF specification, an
   instruction-set manual, the IEEE-754 standard) may be implemented from the
   spec — that is interoperability, not derivation. General computer-science
   knowledge (SSA, dominance, graph coloring, DPLL) is used freely.
2. **No third-party code.** The dependency graph contains only crates from this
   workspace. No crates.io dependencies, vendored or otherwise. Utilities we
   would normally pull from the ecosystem (bit-vectors, hashing, arena
   allocators, arg parsing) are written here.
3. **Pure, safe Rust.** `unsafe` is a `warn` lint. It is permitted only where a
   safety invariant genuinely cannot be expressed in the type system, and every
   use is documented with the invariant it upholds.
4. **Design our own formats.** Our textual IR (`.lf`), binary IR / bitcode, and
   object format (`.lfo`) are our designs. Where we emit or read a *standard*
   external format (ELF, DWARF), we implement it against its public spec.
5. **Test as we build.** Every phase ships with unit tests; from Phase 2 onward
   we maintain golden-file and round-trip tests, and from Phase 5 onward,
   execution tests that actually run generated code.

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

drivers:  lf (umbrella)   lf-opt   lf-as   lf-dis   lf-ld
```

### Crate map

| Crate / module            | Responsibility                                          |
| ------------------------- | ------------------------------------------------------- |
| `latticefoundry::support` | interning, arenas, bit-vectors, small ADTs              |
| `latticefoundry::ir`      | IR data model, type system, builder, text/binary format |
| `latticefoundry::verify`  | well-formedness checking; SMT-backed condition checks   |
| `latticefoundry::pass`    | pass manager, analysis caching, pipeline description     |
| `latticefoundry::codegen` | target-independent lowering to machine IR               |
| `latticefoundry::mc`      | machine-code encoding and object emission               |
| `latticefoundry::target`  | target registry and per-target tables                   |
| `latticefoundry::link`    | linker core                                             |
| `z3rs`                    | clean-room SMT solver (bit-vectors, LIA, arrays)        |

### Binaries

| Binary   | Role                                             |
| -------- | ------------------------------------------------ |
| `lf`     | umbrella driver; ties the tools together         |
| `lf-opt` | load IR, run a pass pipeline, write IR back       |
| `lf-as`  | assembler: target assembly → relocatable object   |
| `lf-dis` | disassembler: machine code → assembly             |
| `lf-ld`  | linker: objects/archives → executable / shared obj |

### Naming conventions

- Binaries are prefixed `lf-` (`lf-ld`, `lf-as`, ...); `lf` is the umbrella.
- Our textual IR uses the extension `.lf`; binary IR uses `.lfb`; our native
  object format uses `.lfo`.
- Public id types are small `Copy` newtypes (`FuncId`, `BlockId`, `ValueId`).

## 4. Phased plan

Phases are ordered by dependency, not by calendar. Each lists deliverables and
the exit criteria that define completion. **Bold** phases below are the
critical path to "compile a function to a running native executable" (the first
end-to-end milestone, Phase 8).

### Phase 0 — Foundations & scaffolding  ✅ *in progress*

Bring up the workspace and the low-level support layer.

- Cargo workspace, edition 2024, zero third-party deps, shared lints.
- `support`: string interner, typed-index/arena primitives.
- Driver skeletons for all five binaries (`--version`/`--help`).
- CI-friendly `build` / `test` / `clippy` all green.

*Exit:* `cargo build/test/clippy --workspace` clean; every binary runs.
*Next in this phase:* arbitrary-width integer (`ApInt`) and bit-vector types,
a general arena allocator, a deterministic hash map, and a diagnostics type
with source spans.

### **Phase 1 — Core IR**

The typed SSA data model and the programmatic builder.

- Complete type system: integers, floats, pointers (opaque), arrays, structs,
  vectors, function types.
- Full value model: instruction results, block/function arguments, constants,
  global values, φ-nodes.
- Complete opcode table: arithmetic/bitwise, comparisons, memory
  (`load`/`store`/`alloca`/`getelementptr`-equivalent), control flow
  (`br`/`switch`/`ret`/`unreachable`), casts, `call`, `select`.
- `IrBuilder` with SSA construction helpers; use/def tracking and value
  replacement.

*Exit:* build non-trivial functions in memory (loops, calls, branches);
use/def lists are consistent under mutation; covered by unit tests.

### **Phase 2 — Textual & binary format, and the verifier**

Make IR persistable and checkable.

- `.lf` textual **printer** and **parser** (our own grammar) with round-trip
  fidelity.
- `.lfb` binary format (compact, versioned) with round-trip fidelity.
- Verifier: single terminator per block, dominance of uses by defs, type
  agreement on operands and calls, entry-block/φ rules, well-typed constants.
- Wire the format + verifier into `lf-opt` (load → verify → print).

*Exit:* golden-file tests for the printer; `parse(print(m)) == m` and
`read(write(m)) == m` for a corpus; verifier rejects a suite of malformed
modules with precise diagnostics.

### Phase 3 — Pass & analysis infrastructure

The framework that optimizations plug into.

- Pass manager with module/function granularity and a textual pipeline spec
  (e.g. `-p mem2reg,dce,gvn`).
- Analysis manager with dependency tracking, caching, and invalidation.
- Core analyses: CFG, dominator/post-dominator trees, dominance frontier,
  natural loops, use-def/def-use, simple alias analysis.

*Exit:* analyses validated against brute-force oracles on random CFGs;
invalidation verified (a mutating pass forces recomputation).

### Phase 4 — Optimizations

A useful baseline optimization pipeline over the IR.

- `mem2reg` (promote memory to SSA registers using dominance frontiers).
- Dead-code / dead-store elimination, aggressive DCE.
- Constant folding & propagation, instruction combining / peephole.
- CSE / global value numbering, control-flow simplification.
- Function inlining with a cost model; loop-invariant code motion.

*Exit:* each pass has before/after golden tests and preserves verifier
validity; an `-O1`/`-O2` pipeline measurably shrinks a benchmark corpus.

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

### Phase 9 — z3rs (SMT solver) & verification

Bring the solver up and put it to work.

- DPLL(T) core; theory solvers for fixed-width bit-vectors, linear integer
  arithmetic, and arrays; a small textual query format.
- Integrate into `verify` for discharging side conditions (no-overflow, bounds,
  refinement) and into passes for correctness-guarded rewrites.
- Translation validation: prove selected optimization runs semantics-preserving.

*Exit:* solver decides a benchmark suite of bit-vector/LIA queries correctly;
at least one optimization is guarded by SMT-checked rewrites.

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
- **Differential tests** for `z3rs` against its own naive/bit-blasting path.

## 6. Non-goals (for now)

- Language front ends (parsers, type checkers for a source language). Languages
  target our IR; producing that IR is their concern.
- A stable public API or ABI before the pipeline is end-to-end.
- Windows/macOS object and executable formats before ELF is solid.
- Matching the performance of a mature production compiler; correctness and a
  clean, well-tested design come first.

## 7. Milestone summary

| Milestone | Meaning                                                 | Phase |
| --------- | ------------------------------------------------------- | ----- |
| M0        | Workspace builds; drivers run                            | 0     |
| M1        | Build & verify SSA IR in memory                          | 1–2   |
| M2        | Parse/print/round-trip `.lf`; verifier rejects bad IR    | 2     |
| M3        | `-O2` pipeline optimizes a corpus, stays valid           | 3–4   |
| M4        | Emit assembled objects for x86-64                        | 5–7   |
| **M5**    | **Compile `.lf` → native executable that runs**          | 8     |
| M6        | SMT-guarded verification/optimization                    | 9     |
| M7        | JIT, debug info, LTO                                     | 10    |
