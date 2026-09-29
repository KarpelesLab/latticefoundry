# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2](https://github.com/KarpelesLab/latticefoundry/compare/v0.0.1...v0.0.2) - 2026-09-29

### Other

- AVR backend (ir-design 6g, ROADMAP)
- constant-time audit tests, declassify, relocation-format tests
- constant-time lowering where secrets are (per-operation, taint-driven)
- one soft-float pass for Thumb and AVR; AVR scalarizes vectors
- AVR program-space pointers (byte addresses for flash data, word addresses for functions)
- differential test suite; fix block-parameter zero tracking and signed wide constants
- AVR5 backend, ELF32 EM_AVR objects and Intel HEX firmware
- scalarized vectors and constant-time checks
- share locals between values with disjoint live ranges
- random-program and optimized differential tests; unique exports
- the wasm32 backend (ir-design §6f); ROADMAP progress
- WebAssembly backend, wasm objects, and `lf build --target wasm32`
- LEB128 and control-flow structuring
- lf build: Cortex-M executables and flashable images
- Thumb-2 backend for Cortex-M with soft-float AAPCS
- legalize_int keeps declaration lines; attribute regression test
- Thumb-2 relocation kinds and the Arm EABI5 ELF32 target
- fix stack-passed masks, mask stores, float selects, shift wraps
- three-register select; more vector tests
- NEON lowering of 128-bit vectors
- SIMD vectors (ir-design 6e), ROADMAP progress
- min/max and saturating integer ops
- generic vector legalization; x86-64 SSE2 lowering
- SIMD vector types <N x T> as first-class values
- secrets and constant-time preservation (ir-design 6d, B10, ROADMAP)
- constant-time audit of instruction selection on every target
- constant-time preservation across passes
- constant-time verifier for secret values
- secret-taint analysis on the lattice engine
- secrecy annotations (secret params/returns/globals/loads/stores) and declassify
- data layout, address spaces and integer legalization (ir-design §3a/§3b); ROADMAP progress
- wide-integer legalization (split into native-width parts)
- ELF32 relocatable writer (any class, byte order, REL or RELA)
- pointer-width-generic data emission, Abs16, DWARF address size
- per-target DataLayout and pointer address spaces
- :gnu tests: retry a transient ETXTBSY when running the linked executable
- output formats, Win64 and target triples
- lf build: --target, -c/--format, --oformat binary|ihex, --base
- raw binary / Intel HEX firmware output; COFF/Mach-O link tests
- the Microsoft x64 (Win64) calling convention
- PE/COFF and Mach-O object writers; target triples
- gate an atomics test helper like its only caller (Windows dead-code warning)
- runtime support / green threads; ROADMAP progress
- green-thread context runtimes (save/restore/switch/init)
- opt-in yield-point insertion for preemptible green threads
- green-thread context runtime emitted as machine code
- shared-library output, PIC and visibility progress
- shared libraries and PIE executables via qld
- x86-64 position-independent code and symbol visibility
- symbol visibility and function linkage

## [0.0.1](https://github.com/KarpelesLab/latticefoundry/compare/v0.0.0...v0.0.1) - 2026-09-29

### Other

- constrained LR/SC loops; ROADMAP atomics progress
- A extension; lower atomics (AMOs, LR/SC loops, fences)
- lower atomics (ldar/stlr, exclusive loops, dmb)
- volatile load/store, atomics and fences
- lf build: --stack-usage and --no-stack-probes; ROADMAP progress
- stack usage report and stack probes
- stack usage report, large frames, and stack probes
- per-function stack usage report and stack probes
- M9 reached — lf-cc builds Lua, SQLite, gzip, bzip2 against real glibc headers
- bitcast between ptr and aggregates; inliner coerces address-compatible edges
- a cast's result type is part of its hash-consed node
- pack PT_LOAD segments in the file instead of page-aligning them
- global data (ir-design §4a, ROADMAP Phase 7/8 progress)
- first-class global data (.rodata/.data/.bss + data relocations)
- sign-extend narrow ptr_add offsets; fix a private doc link
- extend narrow values before every op that reads above their width
- extend narrow values before every op that reads above their width
