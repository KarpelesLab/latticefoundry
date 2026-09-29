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

## 4a. Global data: attributes, address constants, emission  *(decided)*

A `global @x : T [= init]` is a named storage cell; `@x` as an operand is its
address. With an initializer the module **defines** it; without one it is an
external reference. Each global carries attributes:

```text
global [internal | weak] [constant] [detached] @x : T [= init]
```

- **Linkage** picks the object symbol binding of a definition: external
  (default, `STB_GLOBAL`), `internal` (`STB_LOCAL`, private to the module), or
  `weak` (`STB_WEAK`, yields to a strong definition). IR-level linking (LTO
  merge) resolves by the same strengths: strong beats weak beats detached beats
  a declaration; two strong definitions of one name are an error.
- **`constant`** promises the program never stores to it. The backend places it
  in read-only `.rodata` (a store faults at run time) and optimizations may rely
  on its initial contents.
- **`detached`** says the global's storage is supplied *outside the IR* — e.g. a
  front end that serializes its own data section. The backend emits no storage
  and no symbol definition for it, exactly as for a declaration; its
  initializer only keeps it typed. This is what the original builder call
  `Module::add_global` records, so existing builder clients keep their meaning;
  `Module::define_global` takes explicit attributes. (Attributes live beside
  the `Global` in the module so `Global { name, ty, init }` stays source
  compatible.)

**Initializers** are constants of the global's exact type (the verifier checks
this recursively): integers, floats, `null`, `poison`, aggregates, and the
**address constant** `ptr @sym ± offset` — the address of a global or function
plus a byte offset, a link-time constant. Names in an initializer may refer
forward. The text parser also accepts `[N x i8] "…"` as input sugar for a byte
array (exactly `N` UTF-8 bytes; escapes `\n \t \r \0 \\ \" \xHH`), printed back
in element form. Aggregate and address constants are initializer-only; they are
never instruction operands (use `@sym` / `ptr_add` there). The `.lfb` format
carries both from version 2 (version-1 streams still decode, with default
attributes).

**Emission** (`codegen::data`, shared by every backend; each target supplies
only its absolute-pointer relocation): every defined, non-detached global is
serialized per the data layout, little-endian — integers two's-complement at
their store size, floats as IEEE bits, `null`/`poison` as zeros (zero refines
poison), arrays at the element stride, structs at natural field offsets with
zero padding — and placed at its type's alignment in

| global | section |
|---|---|
| `constant` | `.rodata` (`PROGBITS`, `A`) |
| mutable, some nonzero byte or an address field | `.data` (`PROGBITS`, `WA`) |
| mutable, all zero / poison | `.bss` (`NOBITS`, `WA`) |

with an `STT_OBJECT` symbol of the global's size. An address field becomes a
pointer-sized zero plus an absolute relocation `S + offset` (`R_X86_64_64` on
x86-64). The static linker maps `.rodata` into an `R` segment and `.data`+`.bss`
into one `RW` segment whose `memsz` exceeds its `filesz` by the zero-filled
`.bss`.

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

**Aggregate values are addresses (the struct-by-value convention).** A value of
**aggregate type** (`Struct`/`Array`) *denotes the address of its storage* — its
runtime representation is a pointer to that storage. This is the convention the
backends already emit and gcc links against (see `build_struct_int` in the
x86-64 backend tests): a struct passed/returned by value is an SSA value of
struct type whose machine value is a pointer to the struct's bytes. Consequently
aggregate types and `ptr` are **interchangeable**:

- as the *base* of address arithmetic (`ptr_add`, and its `struct_field` /
  `array_elem` helpers) and as the *address* operand of `load` / `store` — an
  aggregate value is used directly as the base/address, and
- across the *call / return* boundary — a `ptr` may be passed where an aggregate
  parameter is declared, an aggregate value where a `ptr` parameter is declared,
  and likewise for a returned value versus the function's return type.

**Scalars stay strictly typed**; only the pointer ↔ aggregate pairing is
relaxed. The verifier (`src/verify/structural.rs`) enforces exactly this: its
`addr_compatible` predicate (equal, both `ptr`, or one `ptr` and the other an
aggregate) gates the `call`/`ret` boundary, and the `ptr_add`/`load`/`store`
base-and-address checks additionally accept an aggregate-typed operand.

## 6a. Reaching the kernel: a native `syscall` op  *(decided)*

Freestanding programs (`lf build` makes static executables that run on the
bare Linux kernel, no libc) need a way to talk to the kernel from IR. We add one
dedicated opcode:

```text
%r = syscall %nr, %a0, ..., %a5 : i64
```

- **Operands:** the syscall number plus 0..=6 arguments. Each is `i64` or
  `ptr` and fills one 64-bit register exactly. There is **no implicit
  extension**: a front end with a narrower value `zext`s/`sext`s it first (C
  passes `int` arguments sign-extended). This keeps the op's meaning a pure
  function of its operands' bits, and the verifier checks it.
- **Result:** `i64`, the kernel's **raw** return. Linux reports failure as
  `-errno` in `[-4095, -1]`; the op does not interpret that (a front end's
  `errno` wrapper does).
- **Semantics:** an opaque effect on the outside world, exactly as strong as a
  call to an unknown external function. It may read or write any memory
  reachable from an escaped pointer (its own pointer operands escape), so it is
  a full memory clobber. It may not be removed (even with an unused result),
  duplicated, reordered with other memory operations, calls or syscalls,
  hoisted, or speculated. Its result is ⊤ in every abstract domain. A poison
  operand is undefined behavior: the kernel would observe an arbitrary register.
- **Verification:** the reference evaluator (`ir::semantics`) has no
  denotation for it, like `call`. The machine interpreters (virtual, AArch64,
  RISC-V) either take an explicit syscall hook or stop with a clean "unsupported
  side effect" error; they never invent a result. The refinement checker (B2)
  treats it as uninterpreted and reports any function containing one as
  `Unknown`, so it never wrongly proves a rewrite across a syscall.
- **Lowering** (the Linux syscall ABI): x86-64 puts the number in `rax` and the
  arguments in `rdi, rsi, rdx, r10, r8, r9`, then runs `syscall`; `rcx` and
  `r11` are clobbered. AArch64 uses `x8` and `x0..x5`, then `svc #0`. RISC-V
  uses `a7` and `a0..a5`, then `ecall`. The result comes back in
  `rax`/`x0`/`a0`.
- **Rejected: inline assembly.** An asm blob is a string the optimizer and the
  verifier cannot see into. Every pass would have to treat it as the worst case,
  every target would need its own text, and B1/B2 would stop at its boundary. A
  dedicated op says exactly what it does: which registers carry which values,
  what it may touch, and what it returns. So it stays analyzable (precise
  effects rather than "anything"), portable (one IR spelling, three backend
  lowerings), and verifiable (typed operands, a checked arity, and an honest
  "unknown" in the refinement checker).

## 6b. Volatile accesses, atomics and fences  *(decided)*

Memory-mapped I/O needs accesses the optimizer must leave exactly as written,
and concurrent code (a front end's `Atomic[T]`, mutexes, lock-free queues)
needs indivisible read-modify-writes and inter-thread ordering. Both are
first-class in the IR rather than intrinsics or inline assembly, for the same
reasons as `syscall` (§6a): their effects stay precise and checkable.

### Volatile

`volatile` is a flag on the ordinary `load`/`store`:

```text
%v = load volatile %p align 4 : i32
store volatile %v, %p align 4 : i32
```

A volatile access is an observable event in its own right. It is performed
**exactly once**, at **exactly** the accessed type's width, in program order
relative to every other volatile access, atomic, fence, call and syscall. So it
is never removed (not even an unused load, nor a store a later store overwrites),
duplicated, merged with an identical neighbor, widened or narrowed, hoisted or
sunk, promoted to a register (mem2reg leaves a slot with a volatile access in
memory), or forwarded from a store; its result is ⊤ in every abstract domain.
Volatile is *not* atomic (a wide volatile access may tear) and orders nothing
but other volatile accesses: use an atomic for synchronization.

- **Rejected: volatile in the [`Flags`](#7-instruction-flags-one-unified-model)
  set.** A flag licenses optimization and its violation yields poison; volatile
  is the opposite, a restriction with no poison reading. It lives on the
  memory op, like its accessed type and alignment.

### Atomics

Five opcodes, each carrying a C11-style memory ordering:

```text
%v   = atomic_load <ord> %p align A : T                  ; relaxed | acquire | seq_cst
       atomic_store <ord> %v, %p align A : T              ; relaxed | release | seq_cst
%old = atomic_rmw <op> <ord> %p, %v align A : T           ; any ordering
%old = cmpxchg <success> <failure> %p, %expected, %new align A : T
       fence <ord>                                        ; acquire | release | acq_rel | seq_cst
```

- **Orderings** are `relaxed`, `acquire`, `release`, `acq_rel`, `seq_cst`, with
  the C11/C++11 meanings (`relaxed` is atomic but unordered; the IR has no
  weaker LLVM-style `unordered`). The verifier rejects orderings that mean
  nothing for an op: a releasing load, an acquiring store, a `relaxed` fence, and
  a `cmpxchg` *failure* ordering other than a load ordering. The failure ordering
  may be stronger than the success one (as C++17 allows).
- **`<op>`** of `atomic_rmw`: `xchg`, `add`, `sub`, `and`, `nand`, `or`, `xor`,
  `max`, `min` (signed), `umax`, `umin` (unsigned); arithmetic wraps.
- **Types** are `i8`, `i16`, `i32`, `i64` and `ptr` (`ptr` only for loads,
  stores, `xchg` and `cmpxchg`). The alignment is explicit in the text (as on
  `load`/`store`) but must be at least the type's size: atomics are **naturally
  aligned**. The builder helpers (`atomic_load`, `atomic_store`, `atomic_rmw`,
  `cmpxchg`, `fence`) always use natural alignment.
- **`cmpxchg` has one result, the old value.** Instructions produce a single
  value, and an aggregate result would be an address (§6), so rather than a
  `{T, i1}` pair the op returns `old`, and the success flag is
  `icmp eq %old, %expected` (`FunctionBuilder::cmpxchg_success`). This is exact
  because the exchange is *strong* (it never fails spuriously): it happened iff
  the old value equals `expected`, bitwise. A weak (spuriously failing) form
  would need a real second result and is left out until a front end wants it.

**Sequential semantics** (all a single-threaded evaluator observes): an
`atomic_load`/`atomic_store` is a `load`/`store` of `T`; `atomic_rmw` reads
`old`, writes `op(old, v)`, returns `old`; `cmpxchg` reads `old`, writes `new` if
`old == expected`, returns `old`; a fence does nothing. A misaligned, poison or
dangling address is UB; a poison stored/operand value stores poison; a poison
`expected` (or loaded `old`) in a `cmpxchg` is UB, since the comparison would
branch on poison.

**Constraints on optimization.** Every atomic and fence is kept by DCE and SCCP
(including an unused `relaxed` load: dropping it would be sound, but keeping it
is simpler and obviously correct), never hoisted, sunk, duplicated or merged,
and its result is ⊤ in every domain. No transform moves a memory operation
*up* across an acquire (`acquire`, `acq_rel`, `seq_cst` load, rmw, cmpxchg or
fence) or *down* across a release; `seq_cst` operations and `acq_rel`/`seq_cst`
fences are full barriers. The current passes satisfy this structurally: none of
them moves any memory operation (LICM hoists only pure ops, the e-graph emits
effectful ops verbatim in their original order, mem2reg only rewrites slots
whose address never escapes). The reference evaluator (`ir::semantics`) treats
these ops like the other stateful ops, and the refinement checker (B2) reports
any function containing one, or any result-less effect such as a store or a
fence, as `Unknown`, never as a proof.

**Binary form.** Volatile `load`/`store` use their own opcode tags (19/20) with
the same payload as tags 6/7, and the atomics use tags 21–25, so existing
`.lfb` version-2 streams are unchanged and no version bump was needed.

### Lowering

Clean-room from each ISA's memory model. Volatile accesses are one ordinary
load/store of exactly the declared width on every target. Atomics:

| IR | x86-64 (TSO) | AArch64 (ARMv8.0) |
|---|---|---|
| `atomic_load relaxed` | `mov` | `ldr` |
| `atomic_load acquire`/`seq_cst` | `mov` | `ldar` |
| `atomic_store relaxed` | `mov` | `str` |
| `atomic_store release` | `mov` | `stlr` |
| `atomic_store seq_cst` | `xchg` | `stlr` |
| `atomic_rmw xchg` | `xchg` | `ld{a}xr`/`st{l}xr` loop |
| `atomic_rmw add`/`sub` | `lock xadd` (`sub` negates first) | `ld{a}xr`/`st{l}xr` loop |
| other `atomic_rmw` | `lock cmpxchg` loop | `ld{a}xr`/`st{l}xr` loop |
| `cmpxchg` | `lock cmpxchg` (`rax`) | `ld{a}xr`/`cmp`/`b.ne`/`st{l}xr` loop |
| `fence seq_cst` | `mfence` | `dmb ish` |
| `fence acq_rel`/`release` | nothing | `dmb ish` |
| `fence acquire` | nothing | `dmb ishld` |

The RISC-V backend does not lower atomics yet (it stops with a clear "not yet
supported" error).

On x86-64 every `lock`ed instruction and `xchg` with memory is a full barrier,
so the orderings of rmw/cmpxchg need nothing more; loads are never reordered with
loads nor stores with stores, so acquire loads and release stores are plain
`mov`s, and only a `seq_cst` store (which must not pass a later load) needs the
`xchg`. The fixed `rax` of `cmpxchg` is modeled with the allocator's range-based
fixed-register intervals: `mov rax, expected; lock cmpxchg; mov old, rax` is one
contiguous window with every operand materialized before it. The `lock cmpxchg`
retry loop is a single pseudo-instruction expanded at encode time with an
internal label, so the allocator sees one instruction with `rax` and a scratch
register as clobbered defs.

On AArch64, `ldar`/`stlr` are RCsc (Arm ARM B2.3), so they implement both
acquire/release and `seq_cst` loads and stores. Without LSE there is no
single-instruction rmw, so each `atomic_rmw` and `cmpxchg` is one
pseudo-instruction expanded at encode time into a load-exclusive /
store-exclusive retry loop: the exclusive load acquires when the ordering (for
`cmpxchg`, either ordering) acquires, and the exclusive store releases when it
releases. Its result, address and operands are ordinary allocated registers
(distinct, since all are live at the one instruction), and the new value and
the store status use `x16`/`x17` (IP0/IP1), which are never allocated. A
narrow compare (`cmpxchg`, `max`/`min`) extends at the access width, so
garbage above a narrow value never causes a spurious mismatch.

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
