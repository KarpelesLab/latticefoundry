# LatticeFoundry — Design Tenets & Bets

This document records the *opinionated* architectural commitments that make
LatticeFoundry more than an LLVM re-implementation. The [ROADMAP](../ROADMAP.md)
says *what* we build and in what order; this says *why* it is shaped the way it
is, and which research directions we are betting on.

It has three parts:

1. **Tenets** — commitments that hold across the whole system.
2. **Verification tiers** — the concrete meaning of "verification-native."
3. **The bets register** — the specific innovations, each marked
   *committed* / *staged* / *moonshot*, with prior art and our angle.

---

## 1. Tenets

These are load-bearing. A change that violates a tenet is a change to the
project's identity, not a local decision.

### T1 — Clean room, our own crates, pure safe Rust

Restated from the ROADMAP §2 because everything else rests on it: designed from
first principles, dependencies limited to our own focused crates
(`puremp`, `z3rs`), `unsafe` a `warn`-level exception. We interoperate with
*published standards* (ELF, DWARF, ISAs, IEEE-754) by implementing them from
spec, never by transliterating another project's code.

### T2 — Semantics-first

Every IR operation is defined by a **machine-checkable reference semantics**
before it has an optimizer or a code path. Meaning is not something we recover
later from how passes happen to treat an opcode; it is the primary artifact, and
the implementation is checked against it.

*Rationale:* LLVM's semantics were retrofitted, and the `undef`/`poison`
tangle — and the stream of miscompiles Alive2 keeps finding — is the direct
cost of defining meaning after the fact. We own an SMT solver; we can afford to
be a semantics-first compiler, and it is our single biggest differentiator.

### T3 — Correctness by verified refinement

A transformation is not trusted because it looks right; it is trusted because it
is a proven **refinement** of the program it replaces (see
[ir-design](./ir-design.md) for the formal definition). Passes emit proof
obligations; `z3rs` discharges them. See §2 for the tiers that make this
affordable.

*Rationale:* translation validation buys most of the assurance of full verified
compilation (CompCert) at a fraction of the cost, and — unlike full
verification — it also covers passes we did not write and rules we synthesize.

### T4 — One sound lattice engine, not a drawer of analyses

Constant propagation, integer ranges, known-bits, nullness, alias sets: these
are all monotone fixpoints over lattices. LatticeFoundry provides **one**
abstract-interpretation engine parameterized by an abstract domain with an
explicit concretization (γ), and the soundness of each domain's transfer
functions is itself checked by `z3rs`. New analyses are "define a lattice + a
transfer function + prove it sound," never "write and debug another bespoke
dataflow pass."

*Rationale:* it is the settled theory (Cousot & Cousot, 1977), it collapses an
entire category of duplicated, subtly-buggy code, and it is what the project is
named after.

### T5 — Content-addressed, immutable-friendly core

The IR is built out of small `Copy` ids into arenas, never interior pointers;
types and constants are interned; pure sub-expressions are hash-consed. Nothing
in the core data model relies on mutable global state or pointer identity.

*Rationale:* this is the enabling substrate for incremental compilation,
distributed build caches, cheap structural/semantic diffing of what a pass
changed, and safe parallelism. It is nearly free to *design in now* and very
expensive to retrofit, so we commit to the substrate even where we exploit it
later.

### T6 — Parallel- and incremental-ready from day one

The pass and analysis managers are designed around an immutable-input /
new-output discipline so that independent functions (and, later, regions) can be
processed concurrently and re-processed incrementally. We do not assume a single
global mutable module the way legacy designs do.

---

## 2. Verification tiers

"Verification-native" must be affordable, or it becomes aspirational. So
verification is a **dial**, not a switch, and the semantics are identical across
settings — only *how much we check* changes.

| Tier          | What runs                                                                 | Default in       |
| ------------- | ------------------------------------------------------------------------- | ---------------- |
| `Off`         | nothing                                                                    | perf-critical release builds |
| `Structural`  | the verifier's structural + type invariants (cheap, no solver)            | every debug build |
| `Refinement`  | each pass emits a refinement obligation, discharged by `z3rs` under a work budget; a budget-exhausted `unknown` is reported, never assumed correct | **CI** |
| `Certified`   | obligations are discharged once and cached as certificates; builds *check* certificates instead of re-proving (proof-carrying IR) | release opt-in |

Design consequences:

- **Sound `unknown`.** `z3rs` is built to return a sound `unknown` under a work
  budget rather than a wrong answer or a hang. A `unknown` at `Refinement` tier
  is a review signal, not a silent pass.
- **The trusted base is small.** Under `Certified`, correctness rests on a small
  certificate checker plus `z3rs`, not on the (large, evolving) optimizer.
- **Semantics don't fork.** `Off` is not "different, looser semantics"; it is
  the same semantics with the checks elided. A program's meaning never depends
  on the tier.

---

## 3. The bets register

Each bet is honest about prior art. Our contribution is almost always
*synthesis + native, pervasive integration enabled by owning `z3rs`/`puremp`*,
not a primitive invented from nothing.

Status legend: **Committed** (core identity, design for it now) ·
**Staged** (intended, built on committed foundations, later phase) ·
**Moonshot** (high-value, high-risk; a bet we take only once the core proves
out).

| #  | Bet                                             | Status    | Lands in | Depends on |
| -- | ----------------------------------------------- | --------- | -------- | ---------- |
| B1 | Opcode table *is* a formal semantics            | Committed | 1–2      | T2         |
| B2 | Every optimization is a checked refinement      | Committed | 2–4      | B1, §2     |
| B3 | Proof-carrying IR / certificates                | Staged    | 9        | B2         |
| B4 | Equality-saturation optimizer (verified rules)  | Staged    | 4        | B2, B8, B9 |
| B5 | Native continuous superoptimization             | Moonshot  | 10       | B2, B4     |
| B6 | Region-based (RVSDG-style) mid-level form        | Staged    | 5        | T5, T6     |
| B7 | Content-addressed IR (full exploitation)         | Staged    | 3        | T5         |
| B8 | One lattice / abstract-interpretation engine     | Committed | 3        | T4         |
| B9 | Cost/resource semantics as a lattice             | Staged    | 4        | B8         |
| B10| Provenance & effects in the type system          | Moonshot  | 6+       | B1         |
| B11| Verified lowering to machine code (incl. regalloc)| Moonshot | 7–8      | B1, B2     |

### B1 — The opcode table is a formal semantics *(Committed)*

Each operation ships a reference denotation (`puremp`-exact) plus a poison /
refinement rule, checkable by `z3rs`. *Prior art:* Alive2, Vellvm, K-LLVM — all
external and partial. *Our angle:* in-tree and continuous, so spec and
implementation cannot drift. **Born with the opcode table in Phase 1**, not
added in Phase 9.

### B2 — Every optimization is a checked refinement *(Committed)*

A pass emits `after ⊑ before` (target refines source); `z3rs` discharges it per
the tiers in §2. *Prior art:* Alive2 (external, opt-in). *Our angle:* the
default correctness contract for *all* passes, including ones we synthesize.

### B3 — Proof-carrying IR *(Staged)*

A module carries a certificate of the transformations applied; a small trusted
checker + `z3rs` re-validates without trusting the optimizer. Shrinks the
trusted computing base. Enables the `Certified` tier.

### B4 — Equality-saturation optimizer *(Staged)*

Represent all equivalent programs in an e-graph and extract the best under a
cost model, dissolving phase-ordering for local/algebraic rewrites. *Prior art:*
egg, Cranelift ægraphs, Tensat. *Our angle:* the rewrite rules are **B2-verified**
and the cost model is a **real lattice (B9)** — a verified-rewrite e-graph as the
*primary* mid-level optimizer, not a side experiment.

### B5 — Native continuous superoptimization *(Moonshot)*

A Souper-style synthesizer discovers optimal sequences, proves them with `z3rs`,
and grows the verified rewrite database B4 consumes — the compiler compounds in
capability with a proof trail. *Prior art:* Souper (external, LLVM/x86-specific).

### B6 — Region-based mid-level form *(Staged)*

An RVSDG-style regionalized value/state graph between CFG-IR and codegen, where
DCE, code motion, and loop transforms are near-trivial and processing is
naturally parallel/deterministic. *Prior art:* RVSDG (Reissmann et al.),
sea-of-nodes (Click). *Our angle:* combined with the verification layer and a
parallel pass manager; CFG form retained for codegen.

### B7 — Content-addressed IR, fully exploited *(Staged)*

Given T5's substrate, turn on end-to-end content addressing: automatic
incremental recompilation, distributed caching, and semantic diffing. *Prior
art:* rustc red-green/salsa, Unison, Adapton. Substrate committed now (T5);
exploitation staged.

### B8 — One lattice / abstract-interpretation engine *(Committed)*

A single monotone fixpoint solver (sparse over SSA def-use) parameterized by an
`AbstractDomain` (⊥, ⊑, join, widening, γ), with `z3rs`-checked transfer-function
soundness. *Prior art:* Cousot & Cousot; IKOS/Astrée (niche/proprietary). *Our
angle:* the backbone of a general-purpose framework's analysis layer. **Reshapes
Phase 3.**

### B9 — Cost/resource semantics as a lattice *(Staged)*

Latency, code size, and energy modeled as an abstract domain so the optimizer
reasons about trade-offs formally (feeding B4's extraction) instead of via
magic-number heuristics.

### B10 — Provenance & effects in the type system *(Moonshot)*

Bake a principled pointer-provenance model (PNVI-style) and an effect discipline
into IR types so alias analysis is partly by-construction and verifiable, rather
than a pile of trusted attributes. High payoff, high design cost.

### B11 — Verified lowering to machine code *(Moonshot)*

Instruction-selection patterns carry semantics; `z3rs` checks that each target
encoding refines the IR it implements — translation validation extended through
codegen and a register-allocation *checker* (à la Rideau–Leroy). Tractable only
because we own the solver and carry no legacy backend.

---

## 4. How the committed bets reshape the roadmap

- **Phase 1 (Core IR)** now includes B1: the opcode table is authored *as* a
  semantics, and the value model is designed to be `z3rs`-expressible (poison +
  freeze, no `undef` — see [ir-design](./ir-design.md)). T5's content-addressed,
  id-based substrate is a Phase-1 constraint.
- **Phase 2 (Format + verifier)** now includes the first cut of B2: the verifier
  is semantic, and `lf-opt` can run at the `Refinement` tier on a rewrite.
- **Phase 3 (Analysis)** *is* B8: a single lattice engine, not a set of analyses.
- **Phase 4 (Optimization)** grows B4 (e-graph over B2-verified rules) with B9
  cost lattices, rather than a hand-ordered pass pipeline.

Everything else (B3, B5, B6, B7, B10, B11) is scheduled but explicitly *not* on
the critical path to the first end-to-end native executable (ROADMAP M5).
