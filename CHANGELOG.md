# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
