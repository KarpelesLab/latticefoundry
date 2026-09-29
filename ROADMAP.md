# LatticeFoundry Roadmap

This document is the plan of record for building LatticeFoundry from an empty
repository into a working compiler construction framework. It describes the
architecture, the guiding constraints, and a phased build-out with concrete
deliverables and exit criteria for each phase.

The roadmap is a living document: phases are refined as earlier ones land, and
the exit criteria are what "done" means for each phase.

> **Status (2026-09).** Phases **0–9 are complete** and most of **Phase 10**
> is done. `lf build foo.lf -o foo` compiles IR to a static ELF64 executable that
> runs on the bare Linux x86-64 kernel, with no libc and no system linker. A
> **JIT** runs the same code in-process. The verification bets are all working
> and build on each other:
>
> - **B1**: semantics-first opcodes
> - **B2**: refinement checked by z3rs, now over multi-block acyclic functions
> - **B3**: proof-carrying certificates
> - **B4**: an equality-saturation optimizer
> - **B5**: a z3rs superoptimizer
> - **B8**: one lattice engine with four sound domains
> - **B9**: a cost model
>
> Other Phase 10 work that is done:
>
> - **DWARF** (`lf build -g`)
> - the **`-O0..-O3`** pipeline
> - **LTO**
> - a native **dynamic stack allocation** op (`DynAlloca`)
> - **shared-library output** on x86-64: position-independent code
>   (`CodegenOptions::reloc_model`, GOT/PLT), symbol visibility
>   (`hidden`/`protected`), `lf build --shared`/`--pie` linked by qld
> - **PE/COFF and Mach-O** object writers (x86-64, AArch64), the **Microsoft
>   x64 calling convention** for Windows targets, target triples
>   (`lf build --target`, `-c`), PE executables via qld, and **raw binary /
>   Intel HEX** firmware output
> - **stack usage reports** (per-function frame sizes from the frame layout,
>   worst-case depth over the call graph, `lf build --stack-usage`) and
>   **stack probes** (on by default) on all three targets
- **green-thread runtime support**: LF-emitted context switching on all three
  targets, x86-64 timer-signal preemption, and an opt-in yield-point pass
> - three targets:
>   - **x86-64** executes, with the full System V ABI including
>     struct-by-value and variadics.
>   - **AArch64** covers integer, FP and the AAPCS64 aggregate ABI, validated
>     with `llvm-mc` and an interpreter.
>   - **RISC-V RV64IM** covers integer, validated the same way.
>
> About 460 framework tests pass, and every commit is green: build, test and
> clippy all clean. `unsafe` appears only in the JIT's `exec_mem`, and the only
> dependencies are our own crates.
>
> The **`lf-cc`** C frontend (a separate nested crate, §8) covers C89–C23. It
> builds **gzip, bzip2, GNU make and bash** from source, and each works like the
> system build (gzip and bzip2 output is byte-identical). Milestone **M8** is
> reached, and **M9** (compile against the real `/usr/include`) is reached too.
>
> Still open in Phase 10:
>
> - position-independent code on AArch64 and RISC-V (shared-library output
>   is x86-64 only)
> - sanitizers
> - RISC-V FP, C extension and relocations
> - `DynAlloca` on AArch64 and RISC-V
> - the deferred bets: B6 (region form), B7 (full content-addressing), B10
>   (provenance types) and B11 (verified lowering)

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
| `verify`                  | well-formedness checking; SMT-backed refinement; certificates |
| `analysis`                | one lattice fixpoint engine + abstract domains          |
| `pass`                    | pass manager, analysis caching, pipeline description     |
| `transform`               | optimizations, e-graph, superoptimizer, `-O` pipeline   |
| `codegen`                 | target-independent lowering to machine IR               |
| `mc`                      | machine-code encoding, object emission, DWARF           |
| `target`                  | target registry; `x86_64`, `aarch64`, `riscv`           |
| `link`                    | static linker core; `link::gnu` bridge onto `qld`       |
| `jit`                     | in-process execution of compiled code                   |

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
| [`rsasm`](https://github.com/KarpelesLab/rsasm)          | assembling textual assembly into ELF objects (`mc::asm`, `lf-as`); only the x86, AArch64 and RISC-V backends are enabled |
| [`qld`](https://github.com/KarpelesLab/qld)              | GNU-ld-compatible linking of ELF objects, archives and shared libraries (`link::gnu`, `lf-ld`), including against the host libc |

Every **direct** dependency is one of our own crates. Third-party crates may
appear **transitively**: qld uses rayon, hashbrown and memmap2 (which pulls in
`libc`). Its `plugin` feature, qld's only FFI (`dlopen` of an LTO plugin), is
disabled.

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

### Phase 0 — Foundations & scaffolding  ✅

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

### **Phase 1 — Core IR**  *(carries bets B1, T5)*  ✅

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

### **Phase 2 — Textual & binary format, and the verifier**  *(carries bet B2, first cut)*  ✅

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

### Phase 3 — Analysis: one lattice engine  *(is bet B8; carries B7 substrate)*  ✅

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

### Phase 4 — Optimizations  *(grows bets B4, B9 on B2)*  ✅

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

### **Phase 5 — Target-independent code generation**  ✅

Lower optimized IR toward machine instructions.

- Machine IR (MIR): virtual registers, machine basic blocks, target opcodes.
- Instruction selection framework (pattern-based lowering from IR to MIR).
- Register allocation (start with a correct linear-scan; graph-coloring later).
- Instruction scheduling; prologue/epilogue and stack-frame construction;
  calling-convention lowering.

*Exit:* MIR for a target verifies and, once Phase 6/7 land, assembles and runs.

### **Phase 6 — Machine-code layer**  🔶

Turn instructions into bytes and objects.

- Instruction encoder/decoder framework (drives `lf-as` and `lf-dis`).
- Relocations, sections, symbols; our `.lfo` relocatable object format.
- ELF64 relocatable **writer** implemented from the ELF spec (for interop).
  Object-format plumbing may adapt code from our own `univdreams` (kept
  in-tree, not a dependency); compression of `.lfb`/`.lfo` may use `compcol`.

*Exit:* `lf-as` assembles to `.lfo`/ELF; `lf-dis` round-trips encode∘decode on
a fuzzed instruction corpus; objects are consumable by Phase 8.

*Progress:* the encoder, `.lfo` and the ELF writer are done, and so are PE/COFF
(`mc::coff`: AMD64 and ARM64) and Mach-O (`mc::macho`: x86-64 and arm64)
object writers from their specifications, selected by `mc::write_object` from
a `target::Triple`. They are checked with `llvm-readobj`/`llvm-objdump`/GNU
`objdump` and linked by qld into PE and Mach-O executables. `lf-as` assembles
real GNU-syntax assembly into ELF through our own `rsasm`. Still open: `lf-dis`
(no disassembler yet).

### **Phase 7 — Targets**

Concrete backends. x86-64 is the bring-up target.

- **x86-64**: register file, System V ABI, integer + SSE, encodings, isel rules.
- AArch64: AAPCS64, base integer + FP/SIMD.
- RISC-V (RV64GC): base + common extensions.

*Exit (per target):* the execution test suite passes on real hardware/emulator
for the target's ABI; encodings match the architecture manual.

*Progress:* x86-64 ✅ (integer, SSE, full System V ABI incl. struct-by-value
and variadics; executes natively). AArch64 ✅ integer + scalar FP + AAPCS64
aggregates (validated vs `llvm-mc` + an A64-MIR interpreter; no native
execution on the x86-64 host). RISC-V 🔶 RV64IM integer, plus the A
extension for atomics (validated vs `llvm-mc` + interpreter); F/D, C and
relocations remain. Volatile accesses, atomics (`atomic_load`/`atomic_store`/
`atomic_rmw`/`cmpxchg`) and fences lower on all three targets from each ISA's
memory model (x86-64 TSO `mov`/`xchg`/`lock xadd`/`lock cmpxchg`/`mfence`;
AArch64 `ldar`/`stlr`/exclusive loops/`dmb`; RISC-V AMOs, LR/SC loops and
`fence`), with native two-thread execution tests on x86-64 (see
[ir-design §6b](docs/ir-design.md)). Global data is
first-class on x86-64: `compile_module` emits every defined global into
`.rodata` (`constant`) / `.data` / `.bss` (all-zero) with its linkage as the
symbol binding and `R_X86_64_64` relocations for address-valued initializers
(`ptr @sym ± off`), through the shared `codegen::data` emitter that AArch64 and
RISC-V can adopt by passing their absolute-pointer relocation.
**Position-independent code** on x86-64 (`CodegenOptions::reloc_model`:
`Static`/`Pie`/`Pic`): addresses of preemptible or external symbols load from
the GOT (`mov reg, [rip + sym@GOTPCREL]`), locally bound ones (`internal`,
`hidden`, PIE definitions) use a direct `lea` (`R_X86_64_PC32`), calls stay
`R_X86_64_PLT32`, and pointer-holding constants move to `.data.rel.ro`; no
absolute 32-bit relocation is emitted. Symbol **visibility** (`hidden`/
`protected`, [ir-design §4b](docs/ir-design.md)) reaches ELF `st_other`, and
functions take `internal`/`weak` linkage. AArch64 and RISC-V reject the PIC
models with a clear error (`target::compile_module_for`); they need GOT
sequences (`adrp`+`ldr` `R_AARCH64_ADR_GOT_PAGE`/`LD64_GOT_LO12_NC`, `auipc`+`ld`
`R_RISCV_GOT_HI20`).

### **Phase 8 — Linker & first end-to-end**  ✅

Produce a runnable program.

- `link` core: multi-object symbol resolution, archive (`.a`-style) handling,
  relocation processing, section/segment layout, entry-point setup.
- ELF64 executable **writer**; static linking first.
- **Milestone: `lf` compiles a non-trivial `.lf` module to a native executable
  that runs and returns the expected result.**

*Exit:* end-to-end tests compile → link → execute across the Phase 7 targets.

*Progress:* the static linker core is done for `.lfo` and in-memory objects.
ELF objects, archives, shared libraries and hosted (libc) executables are linked
by our own `qld` through `link::gnu`; `lf-ld` sends each input to the right
linker. `lf build --shared [-soname N]` links PIC output into a **shared
library** (`link::gnu::shared_library_args`: `-shared -z text -z noexecstack`,
`DT_NEEDED libc`), and `--pie` a PIE against the host libc; tests load the
library from gcc-built C (`-L -l` and `dlopen`), check exports, SONAME and the
absence of text relocations, and check interposition (a default-visibility
symbol can be preempted, a hidden one cannot). Programs with static data link and run: `.rodata` maps into an `R`
segment, `.data`+`.bss` into one `RW` segment (`.bss` zero-filled through
`memsz > filesz`), and data relocations are applied in place. Segments are
packed back to back in the file and each starts on a fresh page in memory at
the same in-page offset, so a hello world is 258 bytes rather than 4 KB of
mostly padding; execution tests
cover a `.rodata` string written by `syscall`, a `.data` counter, a 512 KiB
`.bss` array, and a pointer table driving an indirect call, plus `lf build`
end to end and the same objects linked by `qld`. `link::raw` turns a linked
image into a raw binary or Intel HEX firmware file (`lf build --oformat
binary|ihex --base <addr>`), and Windows targets link PE executables through
qld's MinGW-flavor driver.

### Phase 9 — Certified tier: proof-carrying IR  *(is bet B3)*  ✅

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

*Progress:* JIT ✅, DWARF line tables (`lf build -g`, gdb-loadable) ✅,
`-O0..-O3` + LTO ✅, z3rs superoptimizer ✅, native dynamic stack allocation
(`DynAlloca`, x86-64) ✅, native `syscall` op (Linux ABI on all three targets;
x86-64 execution-tested, freestanding) ✅, per-function stack usage
(`codegen::stack`: exact static frame sizes read off each target's frame
layout, callees / indirect calls / syscalls / `dyn_alloca`, and
`StackReport::worst_case_depth` over the call graph with caller-supplied bounds;
`compile_module_with` on all three targets, `lf build --stack-usage`) ✅, stack
probes (`CodegenOptions::stack_probes`, default on: frames and x86-64
`dyn_alloca` move `sp` one 4 KiB page at a time and touch each step, so an
overflow faults on the guard; x86-64 execution-tested, AArch64/RISC-V
emulated; AArch64 frames beyond 4 KiB now encode correctly) ✅, green-thread
runtime support ([docs/runtime-support.md](docs/runtime-support.md)):
LF-emitted context save/restore/switch routines with a versioned context
layout on all three targets, x86-64 signal preemption through the `ucontext`
(`rt_sigaction` + restorer, execution-tested with a timer), and an opt-in
yield-point pass whose loop selection uses a first B9 cost lattice ✅, other output
formats (PE/COFF and Mach-O objects, the Microsoft x64 convention on x86-64
Windows — execution-tested on Linux against gcc's `ms_abi` and a
callee-saved-register harness —, `target::Triple`, raw binary / Intel HEX) ✅.
Open: dynamic linking, PGO hooks, sanitizers, richer alias analysis, Windows
unwind tables (`.pdata`/`.xdata`), Mach-O executables.

## 5. Testing strategy

- **Unit tests** in every module, from Phase 0.
- **Round-trip tests** for every serializer/deserializer (text, binary, object).
- **Golden-file tests** for printers and pass output; diffs are reviewable.
- **Property/oracle tests** for analyses (compare against brute force on random
  graphs) and for the encoder (`decode(encode(i)) == i`).
- **Execution tests** from Phase 5/8: compile and run, check observable output.

## 6. Non-goals (for now)

- Language front ends *inside the framework library*. Languages target our IR;
  the one front end we build, `lf-cc`, is a separate crate (§8) that consumes
  the library like any other client.
- A stable public API or ABI before the pipeline is end-to-end.
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
| M7        | JIT, debug info, LTO                                     | 10    | ✅ done (dynamic linking, sanitizers still open) |
| **M8**    | **`lf-cc` builds gzip from source → byte-identical to GNU gzip** | lf-cc | ✅ **done** |
| **M9**    | **`lf-cc` compiles against the real `/usr/include`**     | lf-cc | ✅ **done** (Lua, SQLite, gzip, bzip2 against real glibc) |

## 8. lf-cc: toward a bootstrap-capable C compiler

`lf-cc` (the in-tree C frontend, a *consumer* of the framework) is the proof that
LatticeFoundry can compile real-world C. The north star is **bootstrap
capability**: compiling the dependency-free packages an LFS-style base system
starts with — beginning with **gzip** (pure C, no dependencies, one of the first
things a Linux From Scratch build compiles).

Today `lf-cc` covers essentially the full C language surface through C23, has a
`-c` object-emit mode, and links against the real libc via the system linker. The
gap to a genuine bootstrap compiler is **the headers**.

### Packages built so far

| Package     | Result |
| ----------- | ------ |
| gzip 1.2.4  | ✅ **M8** — all 14 files; output byte-identical to GNU gzip |
| bzip2 1.0.8 | ✅ all 8 files; output byte-identical, interop both directions |
| make 3.82   | ✅ all 27 files; builds real projects identically to system make |
| bash 3.2    | ✅ all 130 core files; feature battery identical to system bash |
| Lua 5.4.6   | ✅ **real glibc headers**; byte-identical to gcc at -O0/-O2 (M9) |
| SQLite 3.45 | ✅ **real glibc headers**, amalgamation + shell; byte-identical to gcc (M9) |

`lf-cc` is now its own driver. It links through our `qld` against the host
libc (or statically with `-nostdlib`), so gzip builds with **no gcc or system
`ld` at any step**.

Each package exposed a handful of real gaps: K&R functions, implicit int,
GNU keyword aliases, wide literals and `alloca`, plus a few miscompiles. All of
them were fixed at the source, and every fix also counts toward M9.

Remaining C niche items: `_BitInt` wider than 64 bits, a true 80-bit
`long double`, `_Complex`, VLAs, flexible array members, and gcc object-ABI
compatibility for struct-by-value. Whole-program struct-by-value is already
correct; the gap is only in mixing `lf-cc` objects with gcc-compiled ones.

### M9 — Consume the real `/usr/include` ✅

*Status (2026-09-29):* **reached.** `/usr/include` is searched by default, and
the ~120 glibc headers tested (the 28 core ones plus ~95 more) all compile.
gzip, bzip2, Lua and SQLite build against them with output byte-identical
to gcc's. Remaining, each with a clear error today:
- GCC vector types, `_Atomic`, `__thread`/TLS;
- `__int128`/`_Float128`/`_Complex` *values* (declarations work);
- a true 80-bit `long double`;
- `__label__`, `__auto_type` and range designators;
- `tgmath.h` and `stdatomic.h`;
- C99 plain-`inline` external-definition semantics;
- SSE classification for float-only unions and packed structs.

The original plan follows.

Real system builds `#include <stdio.h>` etc., and glibc's headers are dense with
GNU/glibc constructs `lf-cc` does not yet accept. Making `lf-cc` a drop-in that
consumes the actual `/usr/include` (rather than minimal hosted-header stubs)
requires, roughly:

- *Done so far:* asm labels (including glibc's `__REDIRECT`), GNU
  extended-asm syntax (compiler barriers compile to nothing), file-scope `asm`
  (assembled with rsasm), `__extension__`, `__USER_LABEL_PREFIX__`,
  `__inline__`/`__restrict__`, and trailing `__attribute__`. The real
  `<string.h>` compiles and runs; `<stdio.h>` stops at `__builtin_va_list`.
  Still open: inline asm with instructions or operands, which needs an
  inline-asm IR op.
- **GNU C extensions** the headers use pervasively: `__attribute__((...))` (parse
  in every position, mostly ignore), `__extension__`, `__inline`/`__inline__`,
  `__restrict`, `__asm__`/`asm` (incl. asm *labels* on declarations and
  register-asm), statement expressions `({ ... })`, `__typeof__`, the `__builtin_*`
  family used in macros, `__int128`, `_Float*`/`__float128`, computed `goto`,
  `case a ... b` ranges, zero-length/flexible arrays.
- **K&R old-style** function definitions/declarations (also needed by old gzip).
- **Preprocessor completeness**: `#include_next` (glibc/`bits/*` layering),
  `__has_include`/`__has_attribute`/`__has_builtin`, and the full set of
  **predefined macros** the headers branch on (`__GNUC__`/`__GNUC_MINOR__`,
  `__STDC_VERSION__`, `__SIZEOF_*__`, `__CHAR_BIT__`, `__WORDSIZE`, the
  arch/ABI/endianness macros, feature-test handling of `_GNU_SOURCE` etc.).
- **glibc idioms**: `__THROW`/`__nonnull`/`__wur`/`__BEGIN_DECLS` (all macros —
  fall out once the above work), and enough of `<bits/*.h>` to type the public API.
- The gcc-provided freestanding headers (`<stddef.h>`, `<stdarg.h>`, …) resolved
  from gcc's own include dir or our builtins.

This is a substantial multi-phase effort (a robust GNU-C-dialect front end), but
it is what turns `lf-cc` from "compiles our programs" into "can build the base
system." M8 (gzip via minimal stubs) is the on-ramp; M9 (real headers) is the
goal; further LFS packages (`coreutils`, eventually a C library and the
compiler itself) are the horizon beyond it — `bzip2`, `make` and a shell
(`bash`) already build via stubs.
