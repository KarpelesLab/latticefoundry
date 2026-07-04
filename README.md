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
  taken from any existing compiler or toolchain. General computer-science
  concepts (SSA, dominator trees, register allocation) are fair game;
  another project's implementation is not.
- **No third-party code.** The only dependencies are other crates in this
  workspace — for example [`z3rs`](crates/z3rs), our own SMT solver. No crates
  from crates.io.
- **Pure, safe Rust.** `unsafe` is a `warn`-level lint, used only where an
  invariant genuinely cannot be expressed in the type system.

## Layout

```
latticefoundry/
├── crates/
│   ├── latticefoundry/   the framework library (ir, verify, pass, codegen, mc, target, link)
│   └── z3rs/             a clean-room SMT solver, used by the verifier
└── bin/
    ├── lf/               compiler driver (umbrella front end)
    ├── lf-ld/            linker
    ├── lf-as/            assembler
    ├── lf-opt/           IR optimizer driver
    └── lf-dis/           disassembler
```

## Status

Early scaffold. The container hierarchy, type system, and tool skeletons exist
and build; the engines behind them are filled in phase by phase. See
[`ROADMAP.md`](ROADMAP.md).

## Building

```sh
cargo build --workspace
cargo test  --workspace
cargo clippy --workspace --all-targets
```

Requires a Rust toolchain supporting the 2024 edition (1.85+).

## License

Licensed under the Apache License, Version 2.0.
