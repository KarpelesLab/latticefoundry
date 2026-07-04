# LatticeFoundry

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
- **Only our own crates.** The dependency graph contains only our own focused,
  clean-room library crates. Nothing third-party, no `-sys` crates, no C.
  Currently:
  - [`z3rs`](https://github.com/KarpelesLab/z3rs) — pure-Rust SMT solver, used
    by the verifier.
  - [`puremp`](https://github.com/KarpelesLab/puremp) — arbitrary-precision
    numeric core, used for wide IR constants (and `z3rs`'s own dependency).
- **Pure, safe Rust.** `unsafe` is a `warn`-level lint, used only where an
  invariant genuinely cannot be expressed in the type system.

Code from our broader own-tools (e.g. object-format handling in
[`univdreams`](https://github.com/KarpelesLab/univdreams)) may be *adapted into
this tree* where useful, but such wide tools are **not** taken as dependencies.

## Layout

This is a single package (**not** a Cargo workspace): one library plus the
binaries under `src/bin/`.

```
latticefoundry/
├── Cargo.toml
├── src/
│   ├── lib.rs            the framework library
│   ├── support/         interning, small ADTs, numeric core (puremp)
│   ├── ir/              typed SSA IR + type system
│   ├── verify/          well-formedness checks; SMT bridge to z3rs
│   ├── pass/            pass & analysis manager
│   ├── codegen/         IR → machine IR lowering
│   ├── mc/              machine-code encoding + object formats
│   ├── target/          per-architecture description tables
│   ├── link/            linker core
│   └── bin/
│       ├── lf.rs        compiler driver (umbrella front end)
│       ├── lf-ld.rs     linker
│       ├── lf-as.rs     assembler
│       ├── lf-opt.rs    IR optimizer driver
│       └── lf-dis.rs    disassembler
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

## Status

Early scaffold. The container hierarchy, type system, and tool skeletons exist
and build; the engines behind them are filled in phase by phase. See
[`ROADMAP.md`](ROADMAP.md).

## Building

```sh
cargo build
cargo test
cargo clippy --all-targets
```

Requires a Rust toolchain supporting the 2024 edition (1.88+, per `z3rs`).

## License

Licensed under the Apache License, Version 2.0.
