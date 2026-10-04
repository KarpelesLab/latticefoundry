# LatticeFoundry

[![CI](https://github.com/KarpelesLab/latticefoundry/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/latticefoundry/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/latticefoundry.svg)](https://crates.io/crates/latticefoundry)
[![docs.rs](https://img.shields.io/docsrs/latticefoundry)](https://docs.rs/latticefoundry)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A clean-room compiler construction framework in pure Rust.**

LatticeFoundry is a from-scratch toolkit for building compiler back ends —
roughly the role a framework like LLVM plays, but designed and implemented
independently. It provides a typed SSA intermediate representation, a verifier,
a pass and analysis pipeline, target-independent code generation, a
machine-code / object-file layer, pluggable targets, and a linker core.

## Principles

- **Clean room.** Everything is designed and written from first principles. No
  source code, text format, encoding table, or algorithm transliteration is
  taken from any third-party compiler or toolchain. General computer-science
  concepts (SSA, dominator trees, register allocation) are fair game; another
  project's implementation is not. Where we must interoperate with a published
  standard (ELF, DWARF, an ISA manual, IEEE-754), we implement it from the spec.
- **Only our own crates.** Every direct dependency is one of our own focused,
  pure-Rust library crates. There are no `-sys` crates and no C. Third-party
  crates appear only transitively: qld uses rayon, hashbrown and memmap2.
  The direct dependencies are:
  - [`z3rs`](https://github.com/KarpelesLab/z3rs), a pure-Rust SMT solver used
    by the verifier.
  - [`puremp`](https://github.com/KarpelesLab/puremp), the arbitrary-precision
    numeric core used for wide IR constants (and a dependency of `z3rs`).
  - [`rsasm`](https://github.com/KarpelesLab/rsasm), a multi-architecture
    assembler. It turns assembly text into ELF objects (`mc::asm`, `lf-as`).
  - [`qld`](https://github.com/KarpelesLab/qld), a GNU-ld-compatible linker. It
    links ELF objects, archives and shared libraries, including against the
    host libc (`link::gnu`, `lf-ld`).
- **Pure, safe Rust.** `unsafe` is a `warn`-level lint, used only where an
  invariant genuinely cannot be expressed in the type system.

Code from our broader own-tools (e.g. object-format handling in
[`univdreams`](https://github.com/KarpelesLab/univdreams)) may be *adapted into
this tree* where useful, but such wide tools are **not** taken as dependencies.

## Layout

The framework is a single package (**not** a Cargo workspace): one library plus
the binaries under `src/bin/`. The C frontend `lf-cc/` is a separate nested
crate (see below).

```
latticefoundry/
├── Cargo.toml
├── src/
│   ├── lib.rs           the framework library
│   ├── support/         arenas, deterministic hashing, diagnostics
│   ├── ir/              typed SSA IR, type system, builder, reference
│   │                    semantics, `.lf` text + `.lfb` binary formats, module merge
│   ├── verify/          structural verifier; z3rs-backed refinement checker;
│   │                    proof-carrying certificates
│   ├── analysis/        one lattice fixpoint engine + abstract domains
│   │                    (constants, ranges, known-bits, nullness)
│   ├── pass/            pass & analysis manager
│   ├── transform/       mem2reg, DCE, simplify-cfg, SCCP, LICM, inlining,
│   │                    e-graph equality saturation, superoptimizer, -O pipeline
│   ├── codegen/         machine IR, instruction selection, register allocation,
│   │                    MIR interpreter
│   ├── mc/              encoding + fixups, ELF64 / PE-COFF / Mach-O objects,
│   │                    `.lfo`, DWARF, assembly text → objects (rsasm)
│   ├── target/          x86_64/, aarch64/, riscv/; target triples
│   ├── link/            static linker core (ELF64 executables, raw binary and
│   │                    Intel HEX firmware); bridge onto qld for ELF objects,
│   │                    archives, libc and PE executables
│   ├── jit/             in-process JIT (the only `unsafe` in the tree)
│   └── bin/
│       ├── lf.rs        compiler driver (`lf build`)
│       ├── lf-ld.rs     linker (own core for `.lfo`, qld for everything else)
│       ├── lf-as.rs     assembler (rsasm)
│       ├── lf-opt.rs    IR optimizer driver
│       └── lf-dis.rs    disassembler
├── lf-cc/               C frontend (separate crate, not a workspace member)
├── docs/                design tenets and IR design
└── ROADMAP.md
```

## Design

What makes LatticeFoundry more than a re-implementation is written down:

- [`docs/design-tenets.md`](docs/design-tenets.md) — the opinionated
  commitments (semantics-first, correctness-by-verified-refinement, one lattice
  engine for analysis, content-addressed core), the verification tiers, and the
  bets register (*committed / staged / moonshot*).
- [`docs/ir-design.md`](docs/ir-design.md) — the concrete IR decisions (block
  arguments over φ-nodes, poison + freeze with no `undef`, opaque pointers with
  explicit offset addressing, a unified flag model, machine-checkable opcode
  semantics).
- [`docs/runtime-support.md`](docs/runtime-support.md) — green-thread runtime
  support: the LF-emitted context-switching routines and their layouts, signal
  preemption, and the yield-point pass.

## Status

Roadmap phases 0–9 are complete, and most of Phase 10 is too. See
[`ROADMAP.md`](ROADMAP.md) for the full plan.

**Framework (`latticefoundry`)**

- `lf build foo.lf -o foo` compiles IR to a **static ELF64 executable that runs
  directly on Linux x86-64**. It needs no libc and no system linker; every step
  is LatticeFoundry code. `-O0`..`-O3`, `-g` (DWARF, loadable by gdb) and
  `--lto` (whole-program merge + cross-module inlining) are supported.
- The verification bets from the design tenets are live: executable opcode
  semantics (B1), refinement checking with z3rs over multi-block acyclic
  functions (B2), proof-carrying certificates (B3), an equality-saturation
  optimizer (B4), a z3rs superoptimizer (B5), one lattice engine for all
  analyses (B8) and a cost model (B9).
- `lf build --target <triple>` picks the OS as well as the architecture:
  `-c` writes a relocatable **ELF, PE/COFF or Mach-O** object (x86-64 and
  AArch64 for COFF/Mach-O), Windows targets use the **Microsoft x64 calling
  convention** and link a PE executable through qld, and `--oformat
  binary|ihex` writes a **raw binary or Intel HEX** firmware image.
- An in-process JIT runs the same code without writing an executable.
- `lf-as` assembles GNU-syntax assembly for x86-64, AArch64 and RISC-V using
  rsasm.
- `lf-dis` disassembles objects, executables and wasm modules (ELF, `.lfo`,
  COFF/PE, Mach-O, wasm) and flat binaries (`--raw --arch`) for every target:
  x86-64 (AT&T or `--syntax intel`), AArch64, RISC-V RV64GC, Thumb-2, AVR and
  wasm32. The output has symbol labels and inline relocation notes, and
  matches `llvm-objdump` on LF's own objects.
- `lf-ld` has two modes. With only `.lfo` inputs it uses our own static linker
  core. For anything else it accepts a full GNU `ld` command line and links
  with qld: ELF objects, archives, shared libraries, dynamic executables.
- `link::gnu::host_c_link_args` links backend output against the host C
  library directly, without calling a system compiler or linker.
- `lf build --shared -o libfoo.so [-soname libfoo.so.1]` builds a **shared
  library** of C-ABI functions from IR: x86-64 position-independent code
  (GOT/PLT for preemptible symbols, direct RIP-relative for `internal`/
  `hidden`), linked by qld with no text relocations. `--pie` builds a PIE
  executable against the host libc, and `-c --pic` stops at the object.
  Globals and functions take `hidden`/`protected` visibility.

| Target  | Coverage | Validation |
| ------- | -------- | ---------- |
| x86-64  | Integer, SSE float, System V and Microsoft x64 ABIs (struct-by-value, variadics), dynamic `alloca` | Runs natively; golden bytes; linked with gcc (Win64 against gcc's `ms_abi`) |
| AArch64 | Integer, scalar FP, AAPCS64 struct-by-value | Encodings checked against `llvm-mc`; A64-MIR interpreter |
| RISC-V  | RV64IM integer | Encodings checked against `llvm-mc`; interpreter |

Not done yet: position-independent code on AArch64/RISC-V, Windows unwind
tables (`.pdata`/`.xdata`), Mach-O executables, sanitizers, RISC-V FP and
relocations, dynamic `alloca` on AArch64/RISC-V, and the deferred bets (B6
region form, B7 full content-addressing, B10 provenance types, B11 verified
lowering).

**C frontend (`lf-cc`)**

`lf-cc` is a clean-room C compiler that lowers to LatticeFoundry IR and reuses
the whole pipeline. It covers C89 through C23 (`--std=`):

- a full preprocessor, including `#embed`
- aggregates, including bit-fields, floating point, and struct-by-value
- variadic functions
- C11/C23 features: `_Generic`, `constexpr`, `_BitInt(N≤64)`, `typeof`,
  `[[attributes]]`, and more
- K&R functions and common GNU extensions
- freestanding standard headers

Its test suite is differential: each program's exit status is compared with
`gcc`'s at `-O0` and `-O2`.

`lf-cc` is a complete compiler driver on its own. It accepts any mix of `.c`,
`.s`, `.o`, `.a` and `-l` inputs, and `-c` writes one object per source. By
default it links a dynamic executable against the host libc using qld;
`-nostdlib` gives a static, libc-free executable instead. It never runs gcc or
the system `ld`. GNU `asm` labels, compiler barriers and file-scope `asm`
blocks are supported; file-scope asm is assembled with rsasm. The real glibc
`<string.h>` already compiles.

Real packages built from source with `lf-cc`:

| Package     | Result |
| ----------- | ------ |
| gzip 1.2.4  | Compressed output byte-identical to GNU gzip (milestone **M8**) |
| bzip2 1.0.8 | Output byte-identical; interoperates with the system bzip2 in both directions |
| make 3.82   | All 27 files; builds real projects identically |
| bash 3.2    | All 130 core files; the feature battery matches the system bash |
| Lua 5.4.6   | Against the real glibc headers; output byte-identical to a gcc build at -O0 and -O2 |
| SQLite 3.45 | Amalgamation + shell, against the real glibc headers; byte-identical to gcc |

Milestone **M9** is reached: `lf-cc` compiles against the **real** glibc
`/usr/include`, searched by default, with no stub headers. The ~120 glibc
headers tested all compile. gzip, bzip2, Lua and SQLite build that way,
and `lf-cc` links them itself through qld. make and bash were built earlier
against stub headers.
Coverage includes GNU attributes in every position, statement expressions,
computed goto, case ranges, `__builtin_*`, and the System V struct ABI, so
objects mix with gcc-compiled code. Also covered: `volatile`, C11 atomics
(`_Atomic`, `<stdatomic.h>`, `__atomic_*`/`__sync_*`), GCC vector types,
symbol visibility and weak symbols, and `-fPIC`/`-shared`/`-pie` output.
Known gaps: `long double` is `double`, and there is no TLS or
`__int128`/`_Float128` arithmetic. The next goal is a bootstrap-capable
compiler.

Build and test `lf-cc` from inside its own directory (`cd lf-cc && cargo test`).
The root `cargo build` does not touch it.

## Building

```sh
cargo build
cargo test
cargo clippy --all-targets
```

Requires a Rust toolchain supporting the 2024 edition (1.89+, per `rsasm` and
`qld`).

## License

Licensed under the [MIT License](LICENSE).
