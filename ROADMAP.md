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
> - a native **dynamic stack allocation** op (`DynAlloca`, x86-64 and
>   AArch64, with stack probes)
> - **shared-library output** on x86-64 and AArch64: position-independent
>   code (`CodegenOptions::reloc_model`, GOT/PLT), symbol visibility
>   (`hidden`/`protected`), `lf build --shared` (and `--pie` on x86-64)
>   linked by qld
> - **PE/COFF and Mach-O** object writers (x86-64, AArch64), the **Microsoft
>   x64 calling convention** for Windows targets, target triples
>   (`lf build --target`, `-c`), PE executables via qld, **Mach-O
>   executables and dylibs** via qld's ld64 flavor (`dyld` + `LC_MAIN`,
>   linking `libSystem` through a generated `.tbd` stub), and **raw binary /
>   Intel HEX** firmware output
> - **unwind tables** derived from the frame layouts: Windows x64
>   `.pdata`/`.xdata` (every frame shape, including probed frames and
>   `dyn_alloca`, through an `rbp` frame register), Mach-O compact unwind,
>   and opt-in DWARF `.eh_frame` for x86-64 ELF
> - **stack usage reports** (per-function frame sizes from the frame layout,
>   worst-case depth over the call graph, `lf build --stack-usage`) and
>   **stack probes** (on by default) on all three targets
> - **green-thread runtime support**: LF-emitted context switching on all three
>   targets, x86-64 timer-signal preemption, and an opt-in yield-point pass
> - **constant-time preservation for secret values**, a first step toward
>   B10: `secret` parameters, returns, globals and loads/stores, plus
>   `declassify`. A secret-taint analysis on the lattice engine feeds a
>   constant-time verifier (no secret branch, address, division or other
>   variable-time operation). Every pass and `-O` pipeline preserves it, and
>   `select` is branchless on all three targets
>   ([ir-design §6d](docs/ir-design.md)).
> - **SIMD vectors** `<N x T>` (per-lane poison, a generic scalarizing
>   legalizer, SSE2 lowering on x86-64, NEON on AArch64; RISC-V scalarizes)
> - an **undefined-behavior sanitizer**: checks derived from the reference
>   semantics (overflow, shifts, division, float casts, object bounds, null,
>   alignment, `unreachable`), reported by a freestanding runtime written in
>   LF IR or trapping; `lf build --sanitize=…` and `lf-cc -fsanitize=…`
>   ([ir-design §6i](docs/ir-design.md))
> - three targets:
>   - **x86-64** executes, with the full System V ABI including
>     struct-by-value and variadics.
>   - **AArch64** covers integer, FP, the AAPCS64 aggregate ABI, variadic
>     functions (Linux and Darwin), `DynAlloca`, PIC and DWARF; it writes
>     ELF objects and links Linux executables and shared libraries with qld.
>     It is validated with `llvm-mc`, an IR interpreter, and an A64 emulator
>     that runs the linked programs, including against clang-compiled C.
>   - **RISC-V RV64GC** covers integer, the A, F, D and C extensions, the
>     LP64D convention (struct-by-value, variadic calls), `DynAlloca` and
>     PIC; it writes ELF objects and links Linux executables and shared
>     libraries with qld. It is validated with `llvm-mc`, a MIR interpreter,
>     and an instruction-set simulator running the linked programs
>     differentially against the reference semantics, including against
>     clang-compiled C.
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
> - the address sanitizer (shadow memory; designed in
>   [ir-design §6i](docs/ir-design.md))
> - the deferred bets: B6 (region form), B7 (full content-addressing), B10
>   (provenance types; only the constant-time step is done) and B11 (verified
>   lowering)

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
| [`rsasm`](https://github.com/KarpelesLab/rsasm)          | assembling textual assembly into ELF objects (`mc::asm`, `lf-as`); only the x86, AArch64 and RISC-V backends are enabled (the tests also use its Arm backend to re-assemble the Thumb disassembler's output) |
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

### **Phase 6 — Machine-code layer**  ✅

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
real GNU-syntax assembly into ELF through our own `rsasm`. The ELF writer is
generic over an `ElfTarget` (ELF32 or ELF64, either byte order, `REL` or
`RELA`, the machine's relocation numbering), so the 32-bit targets get ELF32
relocatable objects (validated with `readelf`); relocation kinds include
`Abs16`, and DWARF can use 4- or 2-byte addresses.

`lf-dis` is done (`mc::disasm`): one decoder per target, each written from
its manual, giving bytes → a typed instruction → text. x86-64 prints AT&T
(default) or Intel, with the general-purpose ISA, x87, SSE–SSE4.2 and VEX
AVX/AVX2/FMA3/BMI. AArch64 covers the A64 base ISA, LSE, FP and Advanced
SIMD, with the Arm ARM's preferred aliases. RISC-V covers RV64GC with
Zicsr/Zifencei/Zba/Zbb and its pseudoinstructions. Thumb-2 covers ARMv7-M,
including IT-block state. AVR covers the whole AVRe+ set, and the AVR test
interpreter now executes the decoder's typed instructions. wasm covers MVP
through threads and fixed-width SIMD, and the wasm backend's test decoder
wraps it.

An unknown encoding prints as `.byte`/`.short`/`.word` and never panics,
which a random-byte test checks for every target. Readers for ELF32/64
(objects and linked images), `.lfo`, COFF/PE, Mach-O and wasm feed an
objdump-style listing with symbol labels, branch targets as `<sym+off>`,
`$d` data, and inline relocation notes (`callq ext  # R_X86_64_PLT32
ext-0x4`). The CLI flags are `--arch`, `--syntax att|intel`, `-d`/`-D`,
`--raw --base` and `--start`/`--stop`.

The exit criterion is met. Per target, a fuzzed corpus of LF's own encoder
helpers is decoded and re-encoded byte for byte: through rsasm for x86-64,
AArch64, RISC-V and Thumb (llvm-mc arbitrating where rsasm picks another
encoding), and through each decoder's typed form for AVR and wasm.
Separately, llvm-objdump agrees exactly with our text on the objects LF
compiles for every target and format, and on llvm-mc-assembled corpora of
the common ISA.

### **Phase 7 — Targets**

Concrete backends. x86-64 is the bring-up target.

- **x86-64**: register file, System V ABI, integer + SSE, encodings, isel rules.
- AArch64: AAPCS64, base integer + FP/SIMD.
- RISC-V (RV64GC): base + common extensions.

*Exit (per target):* the execution test suite passes on real hardware/emulator
for the target's ABI; encodings match the architecture manual.

*Progress:* x86-64 ✅ (integer, SSE, full System V ABI incl. struct-by-value
and variadics; executes natively; compare-and-branch fusion, fall-through
block layout, relaxed rel8 branches, immediate forms, a precise allocator
with register hints and copy coalescing, frame-pointer-less leaves and
shared epilogues). AArch64 ✅ integer + scalar FP + AAPCS64
aggregates + variadics (the `va_list` register save area; Darwin's
stack-passed anonymous arguments) + `DynAlloca` + PIC, ELF objects linked by
qld into Linux executables and shared libraries (validated vs `llvm-mc`, an
A64-MIR interpreter, and an A64 emulator running the qld-linked programs,
cross-checked against clang-compiled C; no native execution on the x86-64
host). RISC-V ✅ RV64GC: integer, the A extension (atomics), F and D
(`f32`/`f64` in `f0`–`f31`; fused multiply-adds only under `contract`;
branch-free compares and saturating conversions; `frem` through `fmod`),
the C extension (`lf build --target riscv64gc-linux`: every compressible
instruction in its 16-bit form, byte-identical to `llvm-mc +c`), the LP64D
convention (floats in `fa0`–`fa7` then integer registers then the stack,
structs flattened into float/integer registers or passed by reference,
variadic calls), `DynAlloca` (an `s0` frame pointer, probed), PIC, and
`EM_RISCV` ELF objects with `R_RISCV_CALL_PLT`, `PCREL_HI20`/`LO12_I`,
`GOT_HI20` and `R_RISCV_64` that qld links into static executables (`lf
build --target riscv64-linux`) and shared libraries. Validated vs `llvm-mc`,
a MIR interpreter, and an RV64IMAFDC instruction-set simulator that runs the
linked programs (and loads qld's shared libraries and PIEs, applying their
dynamic relocations) differentially against the reference evaluator —
~80 000 integer, float, struct and vector results — and runs clang-compiled
C calling ours and back. Deferred: `f16` (Zfh), integers wider than 64 bits,
the callee side of variadic functions, TLS, DWARF. Volatile accesses, atomics (`atomic_load`/`atomic_store`/
`atomic_rmw`/`cmpxchg`) and fences lower on all three targets from each ISA's
memory model (x86-64 TSO `mov`/`xchg`/`lock xadd`/`lock cmpxchg`/`mfence`;
AArch64 `ldar`/`stlr`/exclusive loops/`dmb`; RISC-V AMOs, LR/SC loops and
`fence`), with native two-thread execution tests on x86-64 (see
[ir-design §6b](docs/ir-design.md)). Global data is
first-class on x86-64: `compile_module` emits every defined global into
`.rodata` (`constant`) / `.data` / `.bss` (all-zero) with its linkage as the
symbol binding and `R_X86_64_64` relocations for address-valued initializers
(`ptr @sym ± off`), through the shared `codegen::data` emitter that AArch64 and
RISC-V use too, with their absolute-pointer relocations (`R_AARCH64_ABS64`,
`R_RISCV_64`).
**Position-independent code** on x86-64 (`CodegenOptions::reloc_model`:
`Static`/`Pie`/`Pic`): addresses of preemptible or external symbols load from
the GOT (`mov reg, [rip + sym@GOTPCREL]`), locally bound ones (`internal`,
`hidden`, PIE definitions) use a direct `lea` (`R_X86_64_PC32`), calls stay
`R_X86_64_PLT32`, and pointer-holding constants move to `.data.rel.ro`; no
absolute 32-bit relocation is emitted. Symbol **visibility** (`hidden`/
`protected`, [ir-design §4b](docs/ir-design.md)) reaches ELF `st_other`, and
functions take `internal`/`weak` linkage. AArch64 generates the same models
with `adrp`+`ldr` GOT loads (`R_AARCH64_ADR_GOT_PAGE`/`LD64_GOT_LO12_NC`) and
`adrp`+`add` for locally bound symbols; its shared libraries (qld) carry no
text relocation, and a test loads one into the emulator, applies its dynamic
relocations and interposes a symbol. RISC-V does the same with `auipc`+`ld`
GOT loads (`R_RISCV_GOT_HI20` + `R_RISCV_PCREL_LO12_I` against a label on
the `auipc`) and `auipc`+`addi` (`R_RISCV_PCREL_HI20`) for locally bound
symbols; its qld-linked shared libraries and PIEs run in the simulator.
**Thread-local storage** on x86-64 ([ir-design §4c](docs/ir-design.md)):
`global thread_local @x` lives in `.tdata`/`.tbss` (`STT_TLS`) and is reached
through `%fs` with the model the relocation model and locality pick —
local-exec (`R_X86_64_TPOFF32`), initial-exec (`R_X86_64_GOTTPOFF`) or
general-dynamic (`R_X86_64_TLSGD` + `__tls_get_addr`). The static linker
emits `PT_TLS`, relaxes initial-exec/general-dynamic to local-exec, and its
`_start` builds the TLS block and sets `%fs` without libc; execution-tested
statically, with two pthreads against glibc (qld, non-PIE and PIE), and in a
`dlopen`ed shared library. Other targets reject `thread_local` clearly.
**128-bit integers** on x86-64 ([ir-design §3b](docs/ir-design.md)): `i128`
is legalized into 64-bit parts (branch-free), multiplied inline, divided and
converted to/from floats through libgcc, and passed exactly like gcc's
`__int128` (register pairs, 16-aligned stack slots, `rax:rdx`); checked
against the reference evaluator on 1,104 random cases at `-O0` and `-O2` and
in ABI round trips with gcc in both directions.

*Non-64-bit foundation* ✅ (for wasm32, Arm Cortex-M and AVR; see
[ir-design §3a/§3b](docs/ir-design.md)): modules carry a per-target
`DataLayout` (pointer size/alignment per address space, scalar alignments,
endianness, stack alignment, native integer widths, program address space;
LP64 by default) and an optional target name, both in the `.lf` header and the
`.lfb` v4 header; `ptr addrspace(N)` pointers and globals placed in an address
space, with verifier rules and no `addrspacecast`; sizes, offsets, the reference
evaluator, global-data emission (4- and 2-byte pointer fields, per-space
relocations) and isel (`Lower::mem_addr_space`) follow the layout; and a
target-independent wide-integer legalization pass (`codegen::legalize_int`) splits
integers above the native width into parts, with libcalls for mul/div/rem,
checked against the reference evaluator at 32-, 16- and 8-bit part widths. The
three backends are built on top (below).

**SIMD vectors** (`<N x T>`, [ir-design §6e](docs/ir-design.md)) lower on all
three targets through the target-independent legalizer (`codegen::legalize`):
x86-64 keeps the 128-bit types in xmm registers and selects SSE2 (the
baseline; no SSE3+), passing `__m128`-class vectors in xmm registers under
System V and by reference under Win64; AArch64 selects NEON for the same types
(encodings diffed with `llvm-mc`); RISC-V (no V extension) scalarizes.
Execution tests on x86-64 and interpreter tests on AArch64/RISC-V check random
vector programs against the reference evaluator (Thumb, like RISC-V, scalarizes).

*Arm Cortex-M (Thumb-2)* ✅ (`target::thumb`, triples `thumbv7m-none-eabi` /
`thumbv7em-none-eabi`): ARMv7-M code under the AAPCS base (soft-float)
standard — `r0`–`r3`/stack argument passing with doubleword alignment, split
composites, `sret`; 16/32-bit encodings with `IT` blocks, `movw`/`movt` and
relaxed branches; `sdiv`/`udiv` or the `__aeabi_idiv` helpers; `i64` through
`legalize_int` at `W = 32` with register pairs at the ABI boundary; floating
point lowered to the RTABI helpers (`__aeabi_fadd`, `__aeabi_dcmplt`,
`__aeabi_f2iz`, …); ELF32 `EM_ARM` EABI5 objects with `R_ARM_THM_CALL`,
`R_ARM_THM_MOVW_ABS_NC`/`MOVT_ABS` and `R_ARM_ABS32`; stack report and probes;
and firmware: a generated vector table and reset handler, a linker script, a
qld link, and `lf build --oformat binary|ihex`. Validated against `llvm-mc` and
by running every program three ways (reference evaluator, MIR interpreter, a
Thumb-2 simulator over the encoded bytes). Deferred: FPv4-SP hard float,
ARMv6-M, `ldrex`/`strex` atomics, `dyn_alloca`, DWARF, PIC.

*wasm32* ✅ ([ir-design §6f](docs/ir-design.md)): a stack-machine backend
outside the MIR/regalloc pipeline (`target::wasm32`). The SSA IR is lowered
directly: a dominator-tree structurizer places `block`/`loop`/`if`/`br_table`
(an irreducible CFG falls back to a dispatch loop), SSA values and block
parameters become wasm locals, narrow integers keep a zero-extension invariant
in `i32`/`i64`, `i128` is legalized into `i64` parts (multi-value at the ABI),
`alloca` uses a shadow stack under `__stack_pointer`, and globals are data
segments. Output: a self-contained module (`lf build --target wasm32`) or a
relocatable object with `linking`/`reloc.*` sections that `wasm-ld` links
(`-c`). Validated by differential execution under node against a reference
interpreter (≈80 000 calls: every integer op at 12 widths, floats, casts,
control flow, memory, calls, atomics, `i128`), `llvm-objdump` decoding, and
`wasm-ld` links. Vectors are scalarized by the generic legalizer (the shared
vector fixtures run under node), and constant-time code stays branch-free
(`select` is wasm `select`; decoded bodies are scanned).

*AVR* ✅ ([ir-design §6g](docs/ir-design.md), `target::avr`, triples `avr` /
`avr-atmega328p`): an 8-bit backend for the AVR5 core (ATmega328P). The
allocator works on register pairs under the avr-gcc convention (`r25`…`r8`,
even-aligned, then the stack; `Y` the frame pointer; `SP` written with
interrupts masked); `i8` is native, `i16` and pointers fill a pair, wider
integers go through `legalize_int` at `W = 16`; flash data in address space 1
is read with `lpm`; division, multiplies beyond `mul`'s reach and soft float
(the shared `codegen::softfloat` pass with libgcc names) call a runtime written
in LF IR (integer helpers and the whole `f32` set); vectors are scalarized.
ELF32 `EM_AVR` objects with the `R_AVR_*` relocations, a Harvard firmware
linker with startup code, and `lf build --oformat ihex|binary`. Validated
against `llvm-mc`/`llvm-objdump` and by running programs on an
instruction-level AVR interpreter against the reference evaluator (integers at
every width, casts, IEEE `f32`, calls with stack arguments, flash and function
pointers, atomics, vectors, whole images from reset); constant-time code is
branch- and skip-free where it touches a secret: isel consults the
secret-taint analysis per operation and uses branch-free compares (reading
`SREG`), a barrel shifter and shift-pair sign extension only there, keeping
the compact branching forms for public code. Deferred: `f64` runtime helpers, interrupt handlers, flash beyond
64 KiB (`elpm`), wider atomics; 6502 and Z80.

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
mostly padding. On request (`--merge-rodata`, or `ImageOptions::merge_rodata`
for a library user) `.rodata` joins the `R+X` segment instead, saving its
program header at the cost of executable read-only data, so merging is
opt-in; `--function-alignment=1` packs functions back to back. With both
(`-Os`) the hello world is 202 bytes with no padding at all. Execution tests
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
(`DynAlloca`, all three targets) ✅, native `syscall` op (Linux ABI on all three targets;
x86-64 execution-tested, freestanding) ✅, bulk-memory ops `memcpy`/`memmove`/`memset`
(inline chunks, `rep movsb`/`rep stosb`, loops, `memory.copy`/`memory.fill`;
the `memopt` pass splits struct copies into scalars, forwards and drops dead
fills; see [ir-design §6k](docs/ir-design.md)) ✅, per-function stack usage
(`codegen::stack`: exact static frame sizes read off each target's frame
layout, callees / indirect calls / syscalls / `dyn_alloca`, and
`StackReport::worst_case_depth` over the call graph with caller-supplied bounds,
`StackReport::analyze_from` listing every obstacle (recursive groups, indirect
calls, `dyn_alloca`, unknown callees) with its call path from the root and
ignoring dead code, reachability queries;
`compile_module_with` on all three targets, `lf build --stack-usage`) ✅, stack
probes (`CodegenOptions::stack_probes`, default on: frames and every
`dyn_alloca` move `sp` one 4 KiB page at a time and touch each step, so an
overflow faults on the guard; x86-64 execution-tested, AArch64/RISC-V
emulated; AArch64 frames beyond 4 KiB now encode correctly)
✅, green-thread
runtime support ([docs/runtime-support.md](docs/runtime-support.md)):
LF-emitted context save/restore/switch routines with a versioned context
layout on all three targets, x86-64 signal preemption through the `ucontext`
(`rt_sigaction` + restorer, execution-tested with a timer), and an opt-in
yield-point pass whose loop selection uses a first B9 cost lattice ✅, other output
formats (PE/COFF and Mach-O objects, the Microsoft x64 convention on x86-64
Windows — execution-tested on Linux against gcc's `ms_abi` and a
callee-saved-register harness —, `target::Triple`, raw binary / Intel HEX) ✅,
Mach-O executables and dylibs (`link::darwin`: qld's ld64 flavor, `LC_MAIN`
started by `dyld`, `libSystem` from a generated text stub; structural checks
with `llvm-objdump --macho`, since they cannot run here) ✅, unwind tables
from the frame layout (`codegen::unwind`; [ir-design §6c](docs/ir-design.md)):
Windows x64 `.pdata`/`.xdata` whose codes `llvm-readobj --unwind` decodes to
exactly the prologue, Mach-O compact unwind (`RBP_FRAME` on x86-64, `FRAME`
on arm64 for frames without callee-saved registers), and DWARF `.eh_frame`
for x86-64 ELF (`--unwind-tables`, default with `-g`/`--shared`/`--pie`;
`llvm-dwarfdump` and a `gdb` backtrace) ✅,
the undefined-behavior sanitizer (`transform::sanitize`, a weak LF IR runtime,
trap mode; `lf build --sanitize`, `lf-cc -fsanitize`) ✅.
Open: dynamic linking, PGO hooks, the address sanitizer, richer alias analysis, unwind
tables for AArch64 beyond Mach-O compact unwind (Windows ARM64 `.xdata` and
ELF `.eh_frame`; see ir-design §6c).

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
| M7        | JIT, debug info, LTO                                     | 10    | ✅ done (dynamic linking, ASan still open) |
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
| coreutils 9.5 | ✅ `./configure CC=lf-cc && make`, every program; `make check` 974/1147 pass vs gcc 993, all 11 failures from the 80-bit `long double` gap ([lf-cc/docs/coreutils.md](lf-cc/docs/coreutils.md)) |

`lf-cc` is now its own driver. It links through our `qld` against the host
libc (or statically with `-nostdlib`), so gzip builds with **no gcc or system
`ld` at any step**.

It also uses the framework features built since: `-shared` (with
`-Wl,-soname,`) links shared objects that gcc-built programs `dlopen`, and
`-pie` links position-independent executables. `-fPIC`/`-fPIE` select the
relocation model. `static` gives IR internal linkage, and
`visibility(...)`/`-fvisibility=` and `weak` map onto the IR symbol attributes.
`volatile` lowers to volatile loads and stores. `__builtin_memcpy`/`memmove`/`memset`
and whole-struct copies and zero fills lower to the IR's bulk-memory ops
(a declared `memcpy` stays a call). `_Atomic`, the builtin
`<stdatomic.h>`, and the `__atomic_*`/`__sync_*` builtins lower to the IR
atomics (a two-thread program loses no update). GCC vector types lower to IR
`<N x T>` vectors and cross calls in XMM registers like gcc's. C99 plain
`inline` definitions no longer emit an external symbol.

Thread-local storage works in all three spellings (`__thread`,
`_Thread_local`, C23 `thread_local`). Such objects become IR `thread_local`
globals (ir-design §4c) in `.tdata`/`.tbss`. Executables and PIEs reach them
with local-exec or initial-exec code, and shared libraries with
general-dynamic code. The C constraints are enforced: block scope requires
`static`/`extern`, and no static initializer may take a thread-local's
address. `__int128` and `unsigned __int128` compute as IR `i128`
(ir-design §3b), with gcc's register-pair ABI in both directions. The
module datalayout declares `i128:128`, so structs and stack slots match
gcc's 16-byte alignment. Division and float conversions call libgcc, which
`-shared` links too. Integer constant expressions now fold with their C
types (promotions, usual conversions, unsigned semantics, exact to 128
bits).

Each package exposed a handful of real gaps: K&R functions, implicit int,
GNU keyword aliases, wide literals and `alloca`, plus a few miscompiles. All of
them were fixed at the source, and every fix also counts toward M9.

Remaining C niche items: `_BitInt` wider than 64 bits (and `__int128`
bit-fields), a true 80-bit
`long double`, `_Complex`, variably modified types beyond a block-scope VLA's outer bound, flexible array members, and gcc object-ABI
compatibility for struct-by-value. Whole-program struct-by-value is already
correct; the gap is only in mixing `lf-cc` objects with gcc-compiled ones.

### M9 — Consume the real `/usr/include` ✅

*Status (2026-09-29):* **reached.** `/usr/include` is searched by default, and
the ~120 glibc headers tested (the 28 core ones plus ~95 more) all compile.
gzip, bzip2, Lua and SQLite build against them with output byte-identical
to gcc's. `__thread`/`_Thread_local` and `__int128` values are done (see
above). Remaining, each with a clear error today unless noted:
- `_Float128`/`_Complex` *values* (declarations work);
- the address of a block-scope `static` object in another `static`
  object's initializer (`static int y; static int *p = &y;`);
- a true 80-bit `long double`;
- `__label__`, `__auto_type` and range designators (so gcc's own
  `<stdatomic.h>`, which uses `__auto_type`, compiles only through lf-cc's
  builtin one);
- `tgmath.h`;
- `_Atomic` structs and unions, and 16-byte atomics (including `_Atomic
  __int128`);
- `__builtin_shuffle` with a non-constant mask;
- SSE classification for float-only unions and packed structs, and for a
  struct wrapping a 16-byte vector. The backend splits such a struct into two
  SSE eightbytes where gcc passes it whole in one XMM register (SSEUP), so it
  silently disagrees with gcc objects. Vectors passed directly are correct.
- `#pragma GCC visibility push/pop` is silently ignored. The attribute and
  `-fvisibility=` work.

The original plan follows.

Real system builds `#include <stdio.h>` etc., and glibc's headers are dense with
GNU/glibc constructs `lf-cc` does not yet accept. Making `lf-cc` a drop-in that
consumes the actual `/usr/include` (rather than minimal hosted-header stubs)
requires, roughly:

- *Done so far:* asm labels (including glibc's `__REDIRECT`), file-scope
  `asm` (assembled with rsasm), `__extension__`, `__USER_LABEL_PREFIX__`,
  `__inline__`/`__restrict__`, and trailing `__attribute__`. The real
  `<string.h>` compiles and runs; `<stdio.h>` stops at `__builtin_va_list`.
  Inline asm with instructions and operands is done on x86-64 through the
  `inline_asm` IR op ([ir-design §6j](docs/ir-design.md)): GCC's constraints
  (registers, fixed registers, memory, immediates, matching, `+`, `&`),
  clobbers, named operands and operand modifiers, labels in templates, and
  register-asm variables as operands, with the template assembled by rsasm
  and spliced into the function. glibc's `<sys/io.h>`, musl-style syscall
  wrappers, `rdtsc`/`cpuid` helpers and asm atomics match gcc. Still open:
  `asm goto`, and inline asm on the other targets (a clear error today).
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
