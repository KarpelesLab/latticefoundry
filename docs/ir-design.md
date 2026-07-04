# LatticeFoundry — IR Design Decisions

This document records the concrete, committed decisions for the LatticeFoundry
IR, with the rationale for each fork. It is the reference the Phase 1 opcode
table and builder are implemented against. Where a decision follows from a
project-wide tenet or bet, it cites it (see [design-tenets](./design-tenets.md)).

These are decisions, not a tutorial: each fork states the option we took, the
option we rejected, and why. Open questions are collected at the end.

---

## 0. Character of the IR

A **typed, SSA-based** intermediate representation, low-level and
target-independent but target-aware (an explicit data layout). Three isomorphic
forms — in-memory, binary (`.lfb`), textual (`.lf`) — that round-trip losslessly.

The design differs from LLVM in five deliberate ways, each below: **block
arguments** instead of φ-nodes, a **poison + freeze** value model with **no
`undef`**, **explicit offset addressing** instead of `getelementptr`, a
**single unified flag model**, and a **machine-checkable semantics** attached to
every opcode (tenet T2 / bet B1).

---

## 1. Container model

`Module → Function → Block → Instruction`, all arena-allocated and referenced by
`Copy` id newtypes (`FuncId`, `BlockId`, `ValueId`), never by pointers (tenet
T5). A block is a straight-line instruction sequence ending in exactly one
**terminator**. The first block of a function is its entry.

## 2. Block arguments, not φ-nodes  *(decided: block arguments)*

Blocks take a typed **parameter list**; each terminator supplies an **argument
list** for every successor edge. A function's parameters are the entry block's
parameters. SSA merges that LLVM writes as `phi` become ordinary block
parameters passed on the branch.

- **Rejected:** φ-nodes. They carry a pile of special cases — "φs must be the
  first instructions," "a φ's operand is evaluated on the *edge*, not in the
  block," parallel-copy semantics on critical edges — that every pass must
  re-learn.
- **Why block arguments:** the edge-copy semantics become explicit and local; no
  instruction-position invariants; the representation maps cleanly onto our
  id/arena model and onto the register-transfer view codegen wants. This is the
  Cranelift/MLIR/Swift-SIL lineage and is widely considered the cleaner SSA
  encoding.

## 3. Type system  *(decided)*

- `Void`.
- `Int(width)` — arbitrary bit width; wide constants use `puremp::Int`, so there
  is no `APInt` of our own to maintain.
- `Float(F16 | F32 | F64)` at first; wider/exotic formats added only when a
  target needs them (values via `puremp`'s float support so constant-folding is
  exact and host-independent).
- `Ptr` — **opaque** (no pointee type), address spaces added lazily as
  `Ptr(addrspace)` only when a target requires them.
- Aggregates: `Array(T, n)`, `Struct(fields)`; vectors (`Vector(T, n)`) added
  with SIMD targets; scalable vectors deferred until a scalable-vector target
  (SVE/RVV) is real.
- `Func(FuncType)`.

Types are interned (structural identity, `Copy` handles) — hash-consing at the
type level is on from day one (T5).

**Rejected:** typed pointers (`i32*`). Opaque pointers are where LLVM arrived
after years of pain; we start there. The *accessed* type lives on the memory
operation (§6), which is where it is actually needed.

## 4. Value & constant model  *(decided)*

Every value has an id and a type. Value-producing things: instruction results,
block parameters, function references, global references, and constants.
Constants are interned; integer/rational constants are `puremp`-backed and thus
arbitrary-precision and host-independent. Use-def and def-use edges are
first-class (they make replace-all-uses and most rewrites cheap).

## 5. Value semantics: poison + freeze, **no `undef`**  *(decided)*

This is the core B1 decision and it must be right before the opcode table exists.

- A value is either a defined value or **poison**. Poison is a deferred
  error that taints any operation depending on it.
- **`freeze`** converts poison into an arbitrary but *fixed, consistent*
  concrete value of the type; a frozen value is no longer poison.
- There is **no `undef`.** LLVM's `undef` — a per-*use* nondeterministic value —
  is the primary source of its unsoundness and of reasoning that is painful to
  encode in an SMT solver. Poison + `freeze` is strictly simpler and sufficient.

**Refinement (the correctness contract, tenet T3 / bet B2).** A transformation
of a function `src` into `tgt` is *correct* iff `tgt` **refines** `src`:

> For every input, if `src` triggers no undefined behavior, then `tgt` triggers
> no undefined behavior, and every result (return value and final memory) of
> `tgt` *refines* the corresponding result of `src`, where a single value
> refines another iff the source value is poison, or the two values are equal.

Intuitively: poison means "any value is acceptable here," so a source that
yields poison lets the target yield anything; a source that yields a concrete
value pins the target to that value. Because we have no `undef`, value
refinement is this clean two-case relation, directly expressible to `z3rs`.

Per-value poison is tracked in the semantics (each operation has a rule for when
its result is poison, e.g. an overflowing `add nsw`), so B2 obligations are
mechanical to generate.

## 6. Memory & pointers: explicit offset addressing  *(decided)*

- Pointers are opaque (§3). Address computation is an explicit
  **`ptr_add(base, byte_offset, provenance_flag)`** rather than a typed,
  multi-index `getelementptr`.
- The builder provides typed *helpers* — `struct_field(ptr, field_index)` and
  `array_elem(ptr, index)` — that compute the byte offset from the type's layout
  (via the module data layout) and lower to `ptr_add`. Structure is thus a
  *front-end convenience*, and the IR itself carries simple, verifiable offset
  arithmetic.
- `load`/`store` carry the **accessed type** and alignment (this is where the
  type that opaque pointers dropped actually belongs), plus hooks for provenance
  metadata.

**Rejected:** LLVM's `getelementptr`. It is powerful but notoriously subtle
(`inbounds` UB, index-type rules, the "does not access memory" caveat). Explicit
offset arithmetic is simpler to specify, simpler to verify, and loses nothing we
cannot recover in the builder.

*Provenance* (a PNVI-style model) is bet B10 (moonshot): the `provenance_flag`
and the accessed-type on `load`/`store` are the hooks that keep that door open
without committing to it now.

## 7. Instruction flags: one unified model  *(decided)*

A single `Flags` mechanism attached to instructions that admit them, rather than
LLVM's per-opcode sprawl:

- Integer: `no_signed_wrap`, `no_unsigned_wrap`, `exact` (on the relevant
  arithmetic/shift/division ops).
- Float: a `FastMath` set (`no_nans`, `no_infs`, `no_signed_zeros`, `reassoc`,
  `contract`, `afn`).

**Semantics of a flag: it is an assumption that licenses optimization, and
violating it produces _poison_, not undefined behavior.** (E.g. `add nsw` whose
true result overflows is poison.) Poison-on-violation rather than UB-on-violation
keeps the refinement relation of §5 total and keeps `z3rs` obligations
first-order.

## 8. Textual & binary forms  *(sketch; specified in Phase 2)*

- `.lf` textual form: our own grammar (not LLVM's). SSA values are named,
  blocks show their parameter lists explicitly, and each op prints its flags and
  — under a verbose mode — a reference to its semantic rule. Round-trips with the
  in-memory form.
- `.lfb` binary form: compact, versioned, content-addressed friendly (T5);
  optional compression via `compcol` when that dependency is adopted.

The precise grammar and encoding are Phase 2 deliverables; the only Phase 1
commitment is that both forms are lossless and that the in-memory model does not
encode anything (pointer identity, iteration order) that a serializer cannot
reproduce.

## 9. Content-addressing & identity  *(decided substrate, staged exploitation)*

Types and constants are hash-consed now. Pure, effect-free value nodes are
designed to be hash-consable so that the region form (bet B6) and full
content-addressing (bet B7) can be turned on without reworking the core. Effectful
or positioned instructions retain identity (their id *is* their identity).
Nothing in the model uses interior mutability or pointer identity that would
break structural sharing or parallel processing (tenets T5/T6).

---

## 10. Open questions (tracked, not yet decided)

- **Exceptions / unwinding.** Model with explicit landing/handler blocks and
  edges, or keep unwinding entirely out of the mid-level IR and lower it late?
  Leaning toward explicit edges so control flow stays first-class and analyzable.
- **Integer signedness at the type level.** Keep integers sign-agnostic (signed
  vs unsigned is a property of the *operation*, as in LLVM), which we tentatively
  favor — revisit if the lattice engine (B8) wants signedness in the type.
- **Undefined behavior surface.** *(decided for the current opcode table.)* The
  UB set is kept as small as possible: among pure value-producing ops, **only**
  `udiv`/`sdiv`/`urem`/`srem` by zero and `sdiv`/`srem` of `INT_MIN` by `-1`
  trigger UB (their result is not representable). Everything else that can "go
  wrong" — `nsw`/`nuw` overflow, `exact` violation, over-wide shift, out-of-range
  or NaN float→int casts, fast-math `nnan`/`ninf` violations — yields **poison**,
  not UB. This is enforced by the reference evaluator (`ir::semantics`) and
  matched by the opcode prose. Revisit only when memory/stateful ops are added.
- **Vector poison granularity.** Per-lane poison vs. whole-value poison. Per-lane
  is more precise but complicates the refinement relation; decide with the first
  SIMD target.
- **Address-space semantics.** Deferred until a target needs more than one; the
  `Ptr` representation reserves room.

Each open question is resolved *before* the opcode or feature it governs is
frozen, and its resolution is added above with the same option/rejected/why
structure.
