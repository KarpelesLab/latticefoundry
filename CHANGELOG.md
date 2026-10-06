# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3](https://github.com/KarpelesLab/latticefoundry/compare/v0.0.2...v0.0.3) - 2026-10-06

### Other

- transform, codegen: inlined struct returns, zero parts, docs
- aarch64, riscv: i128 in register pairs; two-word result tests on every 64-bit target
- transform, codegen: SROA and slot-free register-returned structs
- take function addresses with func_ref, not per-function ptr globals
- ROADMAP progress for bulk memory; clippy
- bulk-memory builtins and aggregate copies; ir-design 6k
- memopt, alias-aware memory optimization of bulk ops
- the memmove pseudo branches on its operands (constant-time isel flag)
- lower bulk memory on every target
- bulk-memory ops memcpy / memmove / memset
- a size-aware cost model and inline(always/never) hints
- mem2reg after each inlining round
- relocate section offsets so multi-object links keep every unit
- shift and cast rules to unpack packed values
- reuse values computed in a dominating block
- keep source lines through the optimizer
- fold ptr_add x, 0 and combine constant-offset ptr_add chains
- fold a branch whose edges are all the same into br
- a branch with no executable edge becomes unreachable
- give empty sections an address instead of panicking
- clippy clean-ups, stack-usage and ROADMAP notes on the new frames
- code-size regression tests for issues #8 and #9
- compare fused into select, commutative ties, short sub rsp
- fused compare-branch, block layout, immediate forms, leaf frames
- precise live ranges, register hints, copy coalescing
- :emit: relaxable branches
- link, codegen: layout for size (opt-in merged .rodata, function alignment)
- dead-function elimination, post-inline CFG clean-up, dead block params
- building GNU coreutils with lf-cc (reproduction, test results)
- saturate extraction cost instead of leaving classes unextractable
- inline asm (ir-design 6j, ROADMAP M9); x86-64 operand width checks
- lower GNU asm statements with operands to inline_asm
- ir, x86-64: GCC-style inline asm (inline_asm/asm_output)
- the undefined-behavior sanitizer (ir-design 6i, ROADMAP), an ASan design
- an undefined-behavior sanitizer pass, its IR runtime, lf build --sanitize
- drop a redundant intra-doc link target in mc::format
- unwind tables and Mach-O executables (ir-design 6c, ROADMAP, README)
- lf build: Mach-O executables and dylibs, unwind-table options
- Windows x64 .pdata/.xdata and Mach-O compact unwind from the frame layout
- stack report: every obstacle with its path from the root, reachability
- guess RISC-V from its relocation kinds in .lfo objects
- RISC-V RV64GC backend (ir-design 4b, 6c, 6d, 6h; ROADMAP)
- the C extension (compressed instructions, RV64GC)
- lf build --shared for riscv64
- static executables from lf build, a float constant-time check
- position-independent code (GOT addressing, shared libraries)
- dyn_alloca with a frame pointer and stack probes
- F/D extensions, LP64D convention, relocations and global data
- lf-cc thread-local storage and __int128 (ROADMAP 8, M9 gaps, README)
- section-less executables start at the code, not the ELF headers
- x86-64 TLS/i128 differential, CLI text checks; ROADMAP Phase 6 done, README lf-dis
- x86-64 decoder (AT&T and Intel)
- AArch64 GOT relocation kinds; PIC objects annotated and checked against llvm-objdump
- Thumb-2 (ARMv7-M) decoder
- AArch64 decoder (A64 base ISA, FP, Advanced SIMD subset)
- RISC-V RV64GC decoder (I, M, A, F, D, Zicsr, Zifencei, C, Zba, Zbb)
- WebAssembly decoder
- AVR decoder; the AVR interpreter executes its instructions
- CLI tests over objects of every target and format, raw binaries, ranges and errors
- carry decoder state between instructions (Thumb IT blocks)
- framework, object readers, listings and the lf-dis CLI
- AArch64 variadics, dyn_alloca, PIC and ELF output (ir-design, ROADMAP)
- variadics, dyn_alloca with probes, PIC, DWARF; link with qld
- ELF64 EM_AARCH64 objects (RELA) and the AArch64 GOT relocations
- lf-cc linkage/PIC/shared, volatile, atomics, vectors, C99 inline (ROADMAP 8, README)
- native i128 with gcc's __int128 ABI
- ir, x86-64, link: thread-local storage (`global thread_local`)

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
