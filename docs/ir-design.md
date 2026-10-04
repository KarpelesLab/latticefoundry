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
- `Ptr` — **opaque** (no pointee type), in an **address space**: `ptr` is the
  default space 0, `ptr addrspace(N)` space `N` (§3a).
- Aggregates: `Array(T, n)`, `Struct(fields)`.
- Vectors `Vector(T, n)` (`<n x T>`), first-class SIMD values (§6e); scalable
  vectors deferred until a scalable-vector target (SVE/RVV) is real.
- `Func(FuncType)`.

Types are interned (structural identity, `Copy` handles) — hash-consing at the
type level is on from day one (T5).

**Rejected:** typed pointers (`i32*`). Opaque pointers are where LLVM arrived
after years of pain; we start there. The *accessed* type lives on the memory
operation (§6), which is where it is actually needed.

## 3a. Data layout and address spaces  *(decided)*

The IR is target-independent but target-*aware*: every module carries a
**data layout** (`ir::DataLayout`) that pins down how its types map onto bytes,
and may name its **target**. Both are module-level declarations in the text
form, and part of the `.lfb` header from version 4:

```text
module "blink"
target "avr"
datalayout "e-p:16:8-p1:16:8-i8:8-i16:8-i32:8-i64:8-f16:8-f32:8-f64:8-S8-n8-P1"
```

The layout records:

| item | spec | meaning |
|---|---|---|
| byte order | `e` / `E` | little / big endian |
| pointers | `p[AS]:SIZE:ALIGN` | size and ABI alignment of a pointer into address space `AS` (default 0); every space a module uses must be declared |
| integers | `iW:ALIGN` | alignment of an integer whose store size rounds up to `W` bits (a wider integer takes the widest entry's) |
| floats | `f16`/`f32`/`f64:ALIGN` | float alignments |
| stack | `S:ALIGN` | stack alignment |
| native ints | `nW:W:…` | integer widths the target computes with (the input to legalization, §3b) |
| program space | `PAS` | the address space functions live in |

All numbers are in bits. Items left out keep their **LP64** value, and LP64 —
`e-p:64:64-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-S128-n8:16:32:64` —
is the default, so a module that never declares a layout means exactly what it
meant before layouts existed (x86-64, AArch64 and RISC-V 64 all use it). The
printer writes `datalayout` only for a non-default layout and `target` only when
set; the parser accepts both optional lines right after `module`.

**Where it lives.** The layout sits in the module's `TypeContext`, so every
size, alignment and offset query (`size_of`, `align_of`, `stride`,
`field_offset`) follows it — and with them the builder's `struct_field` /
`array_elem` helpers (whose offsets are integers of the base pointer's width),
the verifier (a pointer's bit size for `bitcast`, atomics' natural alignment),
the reference evaluator (addresses wrap at their space's pointer width), global
data emission (field sizes, byte order, per-width relocations), and isel
(`Lower::int_width` of a pointer). A backend declares its layout through
`MachineTarget::data_layout`. Modules with different layouts or targets do not
link (`MergeError::DataLayoutMismatch`).

**Address spaces.** A pointer type names its space: `Type::Ptr` is space 0 and
`Type::PtrIn(n)` (`ptr addrspace(n)`, `n ≥ 1`) any other; `PtrIn(0)` interns to
`Ptr`, so each space has one type, and code written before address spaces keeps
compiling. Each space has its own pointer width in the layout — AVR, for
instance, has 16-bit data pointers in space 0 and program memory in space 1.
The rules:

- A **global** lives in an address space (`global constant addrspace(1) @tbl :
  …`, default 0); a reference to it — `@tbl` as an operand, `ptr addrspace(1)
  @tbl` in an initializer — is a pointer into that space. A **function**
  reference is a pointer into the layout's program space, and an indirect call
  goes through such a pointer.
- `alloca` / `dyn_alloca` produce space-0 pointers (the stack is data memory);
  `ptr_add` stays in its base's space; only a space-0 `ptr` is interchangeable
  with an aggregate value (§6).
- Pointers of different spaces are different types: they never unify at a
  call, return, block argument, `select` or `icmp`.
- A `load`/`store`/atomic accesses the space of its address operand. Nothing
  more is recorded on the instruction: isel reads it from the operand's type
  (`Lower::mem_addr_space`), which is how a backend picks, say, AVR's `lpm` for
  program memory and `ld` for data memory.

**AVR's program space** (the `avr` backend, `P1`): a pointer into space 1 is
16 bits and means one of two things by what it points to. The address of
*data* in flash (a `global addrspace(1)`) is a **byte** address, which `lpm`
reads; the address of a *function* is its **word** address (byte address / 2),
what `icall` takes and what avr-gcc's function pointers hold. Loads through the
first work and pointer arithmetic on them is byte arithmetic; the second are
only called. The IR does not distinguish the two kinds of pointer, and nothing
converts between them — a program never needs to, since it cannot read code as
data portably anyway. 16 bits reach 64 KiB of flash; larger devices would need
a 24-bit space-1 pointer and `elpm`.

**No `addrspacecast`.** Converting a pointer from one space to another is
rejected (`bitcast` between pointer types is invalid).

- **Rejected:** an LLVM-style `addrspacecast`. Its meaning is target-defined —
  on the targets we are building for (AVR flash vs. SRAM) the spaces are
  disjoint memories, not views of one memory, so a "cast" has no meaning the IR
  could state or the verifier check. Code that really wants to reuse an address
  numerically across spaces says so with `ptrtoint` + `inttoptr`, which are
  explicit about going through an integer (and whose integer widths follow each
  space's pointer width). A target where spaces alias (a GPU's generic space) can
  add a checked cast later.

**Binary form.** `.lfb` version 4 adds, after the module name, the target (a
presence byte and a string) and the layout spec (empty for LP64), and a
per-global address space (bit 6 of the global's attribute byte, whose bits 4–5
hold the version-3 visibility, announces a varint after the byte). `ptr
addrspace(N)` is type tag 7 with a varint space. Version 1, 2 and 3 streams
still decode, as LP64 modules with no target and every global in space 0.

## 3b. Integers a target cannot compute with: legalization  *(decided)*

The IR has arbitrary-width integers (§3); a machine does not. The layout's
native widths (`n…`) say what a target computes with, and
`codegen::legalize_int::legalize_ints` — a target-independent IR-to-IR pass run
before isel — rewrites every integer wider than a *part width* `W` (by default
the widest native width) as `N` parts of type `iW`, least significant first:

- `and`/`or`/`xor`/`select`/`freeze` per part; `add`/`sub` with an `icmp ult`
  carry/borrow chain; shifts by a constant as part moves plus funnel shifts;
  shifts by a variable as funnel shifts by `s mod W` (each done as two shifts so
  no shift amount reaches `W`) and a `select` ladder on `s / W`;
- `icmp eq`/`ne` as an `or` of per-part `xor`s; ordered compares
  lexicographically from the top part (signed there, unsigned below);
- `trunc`/`zext`/`sext` as part selection and zero/sign fill;
- non-volatile `load`/`store` as one access per part in the layout's byte order;
- block parameters and branch arguments as one per part;
- `mul`, `udiv`, `sdiv`, `urem`, `srem` as **libcalls** `T f(T, T)`, named after
  libgcc (`__muldi3`, `__udivdi3`, `__divdi3`, `__umoddi3`, `__moddi3`, and the
  `si`/`ti` forms) unless the caller supplies other names; missing helpers are
  declared. AVR has no hardware multiplier at all, so this is the right default.

Flags are dropped (the expansion refines: flags only add poison). Thumb uses
`W = 32`; AVR `W = 8` or `16`; wasm32 and x86-64 use `W = 64` for `i128`.

**x86-64 `i128`** (`target::x86_64::prepare_module`): min/max are expanded
and the float↔`i128` helpers declared, then the module is legalized at
`W = 64`, so every operation is a straight-line part computation (carry
chains as compares, shifts as funnel shifts plus a `select` ladder — `cmov`,
compares lexicographically) and nothing branches on data. Two operations
avoid libgcc: the 128-bit `mul` libcall is a placeholder name the isel
expands inline (`mul` for the low product, two `imul`s for the cross terms),
and `switch` on an `i128` compares both halves per case. Division and the
float conversions call libgcc (`__divti3`, `__udivti3`, `__modti3`,
`__umodti3`, `__floattidf`, `__floatuntisf`, `__fixdfti`, `__fixunssfti`, …).
At the boundary an `i128` lives in a register pair and follows the System V
convention exactly as gcc's `__int128`: the next two integer argument
registers (low half first), else a 16-byte-aligned stack slot (later integer
arguments still take the remaining registers), and `rax:rdx` for a result.
Integers wider than 128 bits have no register convention and are rejected at
the boundary, as are wide atomics and wide values under the Microsoft x64
convention. (The LP64 layout keeps aligning `i128` to 8 bytes; a front end
matching gcc's 16-byte `__int128` alignment declares `i128:128` in its
`datalayout`, as lf-cc does.)

**The ABI boundary stays wide.** The pass keeps function signatures: a wide
entry parameter, call argument or result, return value, and the operands and
results of the operations only a backend can split (`ptrtoint`/`inttoptr`,
`bitcast` and float conversions, `syscall`, a `switch` condition, a `ptr_add`
offset, `dyn_alloca`, volatile and atomic accesses) stay whole, joined from and
split into parts with a fixed shape (`zext`/`shl k·W`/`or` and
`lshr k·W`/`trunc`). `illegal_int_ops` checks that nothing else touches a wide
integer. A backend lowers those points in its ABI seam, where a wide value lives
in a register group.

- **Rejected:** rewriting signatures (an `i64` parameter becoming two `i32`s).
  The IR has single-result instructions and returns, so a wide return would
  need an invented multi-value convention, and how a wide value is passed (which
  registers, what order, split between registers and stack) is exactly what
  each target's ABI decides. The boundary shape leaves that decision to it.
- **Rejected:** legalizing inside isel (a value mapped to several virtual
  registers). It would couple every backend's isel to the expansion; as an IR
  pass it is target-independent, verifiable with the ordinary verifier, and
  testable against the reference evaluator.

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
global [internal | weak] [hidden | protected] [constant] [detached] [secret]
       [thread_local] [addrspace(N)] @x : T [= init]
```

- **Linkage** picks the object symbol binding of a definition: external
  (default, `STB_GLOBAL`), `internal` (`STB_LOCAL`, private to the module), or
  `weak` (`STB_WEAK`, yields to a strong definition). IR-level linking (LTO
  merge) resolves by the same strengths: strong beats weak beats detached beats
  a declaration; two strong definitions of one name are an error.
- **`constant`** promises the program never stores to it. The backend places it
  in read-only `.rodata` (a store faults at run time) and optimizations may rely
  on its initial contents.
- **`thread_local`** gives each thread its own instance (§4c).
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
only its absolute-pointer relocations): every defined, non-detached global is
serialized per the data layout (§3a), in its byte order — integers
two's-complement at their store size, floats as IEEE bits, `null`/`poison` as
zeros (zero refines poison), arrays at the element stride, structs at the
layout's field offsets with zero padding — and placed at its type's alignment in

| global | section |
|---|---|
| `constant` | `.rodata` (`PROGBITS`, `A`) |
| mutable, some nonzero byte or an address field | `.data` (`PROGBITS`, `WA`) |
| mutable, all zero / poison | `.bss` (`NOBITS`, `WA`) |

with an `STT_OBJECT` symbol of the global's size. An address field becomes a
zero field of its pointer type's size plus an absolute relocation
`S + offset` of that width — `Abs64`, `Abs32` or `Abs16` (`R_X86_64_64` on
x86-64); `emit_globals_with` lets a target pick the relocation per address
space. The static linker maps `.rodata` into an `R` segment and `.data`+`.bss`
into one `RW` segment whose `memsz` exceeds its `filesz` by the zero-filled
`.bss`.

## 4b. Symbol linkage and visibility  *(decided)*

Functions carry the same **linkage** as globals, plus every global and function
carries a **visibility** — the ELF `st_other` field, which is orthogonal to
the binding:

```text
func [internal | weak] [hidden | protected] @f(…) -> T [{ … }]
```

| visibility | ELF | exported from the `.so`/executable | preemptible |
|---|---|---|---|
| (default) | `STV_DEFAULT` | yes | yes — another component may interpose |
| `protected` | `STV_PROTECTED` | yes | no — own references bind locally |
| `hidden` | `STV_HIDDEN` | no | no |

Linkage stays about the *object* (`internal` → `STB_LOCAL`, `weak` →
`STB_WEAK`); visibility is about the *linked component*. A hidden symbol is
global across the objects of one shared library but absent from its dynamic
symbol table. Visibility applies to references too: a hidden declaration
promises the definition lives in the same component, so code reaches it
without the GOT. When IR linking meets several declarations/definitions of one
symbol, the definition's linkage wins and the visibility becomes the most
constraining of them (hidden > protected > default), as the gABI specifies for
the static linker. Attributes live in `Function::attrs` (`FuncAttrs`) and in
`GlobalAttrs::visibility`; the `.lfb` format carries them from version 3.

**Position-independent code** (`CodegenOptions::reloc_model`, x86-64 and
AArch64): a symbol is *locally bound* when it is `internal`, `hidden`, or —
for PIE — any definition in the module. On x86-64, direct calls always use
`R_X86_64_PLT32` (the linker
resolves a locally bound one directly). Taking the address of a locally bound
global or function is a RIP-relative `lea` (`R_X86_64_PC32`); any other address
(default or protected visibility, external declarations) is loaded from the
GOT with `mov reg, [rip + sym@GOTPCREL]` — protected symbols included, because
their canonical address may be the executable's. Address constants in data stay
`R_X86_64_64` for the linker to turn into dynamic relocations, and `constant`
globals holding an address move from `.rodata` to `.data.rel.ro` (made read-only
after relocation), so a shared object never needs text relocations.

AArch64 follows the same binding rules with its own sequences: a locally bound
address is `adrp`+`add` (`R_AARCH64_ADR_PREL_PG_HI21` +
`R_AARCH64_ADD_ABS_LO12_NC`), any other is loaded from the GOT with
`adrp`+`ldr` (`R_AARCH64_ADR_GOT_PAGE` + `R_AARCH64_LD64_GOT_LO12_NC`), and
calls stay `bl` (`R_AARCH64_CALL26`, which the linker routes through a PLT
entry for a preemptible callee).

RISC-V likewise: a locally bound address is `auipc`+`addi`
(`R_RISCV_PCREL_HI20` against the symbol, then `R_RISCV_PCREL_LO12_I`
against a local label on the `auipc`, since the low part is the low bits of
*that instruction's* displacement), any other is loaded from the GOT with
`auipc`+`ld` (`R_RISCV_GOT_HI20`, then the same label-relative
`R_RISCV_PCREL_LO12_I`), and calls stay `auipc`+`jalr` under one
`R_RISCV_CALL_PLT`.

## 4c. Thread-local storage  *(decided)*

A global marked `thread_local` has **one instance per thread**:

```text
global thread_local @x : i32 = i32 0
global internal hidden thread_local @y : i64
```

**Semantics.** Every thread starts with its own copy of the global, holding the
initializer (zero-filled where it is `poison` or all zero); the copy lives as
long as the thread. `@x` as an operand evaluates to the address of the
**current thread's** copy: the same pointer every time it is evaluated in one
thread, a different one in every other thread that is running at the same time.
A load or store through it touches only that thread's copy, and the address
may be handed to another thread, which then reaches the first thread's copy
(as C allows) for as long as that thread lives. The reference evaluator has a
single thread, so there a thread-local global behaves like an ordinary one.

Optimizations may treat `@x` as a pure, thread-invariant value: CSE, hoisting
and rematerialization *within one function invocation* are sound, because a
function never changes threads mid-execution. (The green-thread runtime
switches contexts on one OS thread, so this holds for it too; a runtime that
migrated a suspended context to another OS thread would have to reload
thread-local addresses after resuming.)

**Rules** (checked by the verifier):

- the address is not a link-time constant, so **no address constant** (`ptr
  @x` in an initializer) may name a thread-local global;
- a thread-local global lives in address space 0;
- `thread_local` composes with the other attributes: linkage and visibility
  mean what they always do, `constant` still promises no stores (the storage
  is still per-thread, so it is not placed in `.rodata`), `secret` still marks
  the contents (its *address* is public, and addressing it is constant-time),
  and `detached` still suppresses emission.

IR linking ORs the attribute across a declaration and its definition. The
`.lfb` form carries it as bit 1 of the version-5 global extension varint, so a
module without thread-locals encodes exactly as before (no version bump).

**Emission.** A defined thread-local goes to `.tdata` (`SHF_TLS`, `PROGBITS`)
or, when all zero, `.tbss` (`SHF_TLS`, `NOBITS`); its symbol, definition or
reference, is `STT_TLS`. Those sections form the **TLS template** a thread's
block is initialized from.

**Access models** (x86-64; `codegen::linkage::tls_model`). The model follows
from the relocation model and whether the symbol is known to be in the module
being built (defined here, `internal` or `hidden`):

| model | used for | sequence | relocation |
|---|---|---|---|
| local-exec | an executable's own variables (`Static`/`Pie`, local) | `mov r, fs:[0]` ; `lea r, [r + x@tpoff]` | `R_X86_64_TPOFF32` |
| initial-exec | other variables from an executable (`Static`/`Pie`) | `mov r, fs:[0]` ; `add r, [rip + x@gottpoff]` | `R_X86_64_GOTTPOFF` |
| general-dynamic | everything in a shared library (`Pic`) | `data16 lea rdi, [rip + x@tlsgd]` ; `data16 data16 rex.w call __tls_get_addr@plt` | `R_X86_64_TLSGD` + `R_X86_64_PLT32` |

`fs:[0]` is the thread pointer itself: under the x86-64 TLS ABI (variant II)
the thread control block starts with a pointer to itself and the TLS blocks
of the initially loaded modules sit just below it, so a local-exec offset is
negative. General-dynamic is a real call (every caller-saved register is
clobbered), written in the exact padded form the ELF TLS ABI specifies so a
linker can relax it; local-dynamic is not used. None of the sequences branches,
so thread-local addressing adds nothing to the constant-time audit (§6d).

**Linking.** qld handles TLS for hosted links (executables, PIE, shared
libraries). The static linker (`link::image`) adds a `PT_TLS` program header
over the template, relaxes initial-exec and general-dynamic accesses to
local-exec in place (an executable is the only module, so every offset is
known), and its synthesized `_start` sets up the thread pointer without libc:
it copies `.tdata` into a block in `.bss`, writes the TCB's self pointer just
past it, and calls `arch_prctl(ARCH_SET_FS)`. A program with its own `_start`
sets up `%fs` itself.

**Other targets.** Every backend built on the shared instruction-selection
framework rejects a thread-local global with a clear error (the default
`TargetIsel::lower_global_addr`), as do wasm32 and x86-64 on Windows; the
PE/COFF and Mach-O writers reject TLS sections. AArch64 (`TPIDR_EL0`) and
RISC-V (`tp`) local-exec are natural follow-ups.

- **Rejected:** TLS as an address space (`ptr addrspace(tls)`). The address of
  a thread-local *is* an ordinary pointer — C passes it around freely and
  dereferences it from any thread — so giving it a different pointer type
  would force casts at every use and make it unrepresentable in ordinary
  memory.
- **Rejected:** an explicit `thread_pointer()` op plus offsets in the IR. The
  offset is only known per access model, at link or load time, and
  general-dynamic is not an offset at all; keeping `@x` as the operand leaves
  the choice to the backend and the IR target-independent.

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

| IR | x86-64 (TSO) | AArch64 (ARMv8.0) | RISC-V (RV64IMA, RVWMO) |
|---|---|---|---|
| `atomic_load relaxed` | `mov` | `ldr` | `l*` |
| `atomic_load acquire` | `mov` | `ldar` | `l*; fence r,rw` |
| `atomic_load seq_cst` | `mov` | `ldar` | `fence rw,rw; l*; fence r,rw` |
| `atomic_store relaxed` | `mov` | `str` | `s*` |
| `atomic_store release` | `mov` | `stlr` | `fence rw,w; s*` |
| `atomic_store seq_cst` | `xchg` | `stlr` | `fence rw,w; s*` |
| `atomic_rmw xchg` | `xchg` | `ld{a}xr`/`st{l}xr` loop | `amoswap` (8/16-bit: LR/SC loop) |
| `atomic_rmw add`/`sub` | `lock xadd` (`sub` negates first) | `ld{a}xr`/`st{l}xr` loop | `amoadd` (`sub` negates first; 8/16-bit: LR/SC loop) |
| `atomic_rmw and`/`or`/`xor`/`max`/`min`/`umax`/`umin` | `lock cmpxchg` loop | `ld{a}xr`/`st{l}xr` loop | `amo<op>` (8/16-bit: LR/SC loop) |
| `atomic_rmw nand` | `lock cmpxchg` loop | `ld{a}xr`/`st{l}xr` loop | LR/SC loop |
| `cmpxchg` | `lock cmpxchg` (`rax`) | `ld{a}xr`/`cmp`/`b.ne`/`st{l}xr` loop | LR/SC loop |
| `fence seq_cst` | `mfence` | `dmb ish` | `fence rw,rw` |
| `fence acq_rel` | nothing | `dmb ish` | `fence.tso` |
| `fence release` | nothing | `dmb ish` | `fence rw,w` |
| `fence acquire` | nothing | `dmb ishld` | `fence r,rw` |

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

RISC-V follows the RVWMO mapping of the ISA manual's memory-model appendix.
Code with atomics needs the A extension (RV64IMA); code without stays RV64IM.
An AMO carries `.aq` for an acquiring ordering and `.rl` for a releasing one;
an LR/SC loop puts `.aq` (and for `seq_cst` also `.rl`) on the `lr` and `.rl`
on the `sc`. The A extension has no byte or halfword AMOs or LR/SC, so 8/16-bit
rmw and `cmpxchg` run an LR/SC loop on the naturally aligned 32-bit word that
contains the lane (natural alignment guarantees the lane never straddles two
words), inserting the new lane with shifts rather than a mask register. The
loops use the encoder's never-allocated scratch registers `t0`/`t1`/`t2`/`t6`,
plus the destination as the `sc` status, since the old value is recomputed
from the loaded word after the loop. Every loop is a *constrained* LR/SC loop
(at most 16 base-ISA instructions from `lr` to the retry branch, no memory
accesses or backward branches inside), which is what the ISA's
forward-progress guarantee requires.

## 6c. Targets: triples, calling conventions, object formats  *(decided)*

The IR does not name a calling convention or an object format: a module is
the same whichever OS it runs on. The choice is made once per compilation by a
**target triple** (`target::Triple` = architecture + OS), passed to codegen as
`CodegenOptions::os` and to the object writers through `mc::write_object`.

- **Calling convention** (`Triple::call_conv`): x86-64 uses the Microsoft x64
  convention on Windows and System V elsewhere; AArch64 uses AAPCS64 (the Apple
  and Microsoft variants differ in variadic calls and in reserving `x18`,
  which the backend never allocates; it implements the base standard's
  variadic calls, with the `va_list` register save area, and Darwin's, where
  anonymous arguments go on the stack; Windows on Arm's are not lowered);
  RISC-V uses LP64D (§6h); Cortex-M Thumb (`thumbv7m`) uses the 32-bit
  AAPCS base standard, so a
  floating-point value travels in core registers like the integer of its width
  (the soft-float lowering makes it one before isel).
  Every function in a module follows the same convention; there is no
  per-function `ms_abi`/`sysv_abi` attribute yet.
- **Object format** (`Triple::object_format`): ELF on Linux and bare metal,
  PE/COFF on Windows, Mach-O on Darwin; a wasm module for wasm32 (§6f). Each
  writer maps the generic `RelocKind`s onto its format and rejects the ones it
  cannot express with an error, never silently.
- **Symbol names** stay the IR names everywhere; the Mach-O writer adds the
  platform's leading underscore itself.
- **Rejected: a convention per call site.** It would let two ABIs meet inside
  one module, which no front end needs today and every backend would have to
  support; a module-wide choice keeps the lowering one well-tested path per
  target.

## 6d. Secrets and constant-time preservation  *(decided; a first step of B10)*

A front end with secret types (Lode's `secret[T]`) needs a guarantee that
survives the optimizer and the backend: a secret value never decides a branch,
never forms an address, and never feeds an instruction whose timing depends
on its operands. The IR carries the secrecy the front end declared, one
analysis derives where it flows, one verifier enforces the rules, and every
pass and lowering is audited against them.

### Representation

Secrecy is declared at the sources and derived everywhere else:

```text
global internal secret @key : [32 x i8] = ...     ; the contents are secret
func hidden @ladder(ptr, ptr, secret i64, i64) -> void ; parameter 2 is secret
func @ct_eq(ptr, ptr, i64) -> secret i64          ; the result is secret
  %x = load secret %p align 8 : i64               ; reads secret memory
  store secret %v, %p align 8 : i64               ; writes secret memory
  %b = declassify %v : i64                        ; the end of secrecy
```

- **Parameters and returns**: the function's `FuncAttrs` (§4b), which holds
  linkage and visibility, also holds `secret_params` and `secret_ret`. They are
  part of the function's interface. Every functional rebuild (`map_function`)
  carries the attributes over to the fresh function; LTO merging keeps the
  union of both sides' secrecy.
- **Globals**: the `secret` bit of `GlobalAttrs`. The global's *address* is
  public; loads based on it yield secrets.
- **Memory**: a `secret` flag on `load` and `store` (orthogonal to
  `volatile`). The front end sets it wherever the accessed type is secret.
- **`declassify`**: the identity on values (poison included), whose result is
  public. It is Lode's explicit escape hatch, and the only way out.

In text, `secret` follows the other global attributes and precedes
`addrspace(N)` (`global internal hidden constant secret addrspace(1) @k`), and
precedes each secret parameter type and the return type in a function header.

Builder: `Module::set_param_secret`/`set_ret_secret`/`set_func_attrs`,
`FuncAttrs::set_param_secret`, `GlobalAttrs::secret`,
`FunctionBuilder::load_secret`/`store_secret`/`declassify`.

Binary (`.lfb` version 5): bit 7 of the global attribute byte and of the
function attribute byte says an **extension varint** of further flags
follows, after the address space for a global. This way the attribute bytes,
whose low seven bits are all in use, never run out again. Global extension bit
0 is `secret`. Function extension bit 0 is a secret return; bit 1 says a
secret-parameter list follows (a count, then ascending indices, bounded by
the signature's arity). Unknown extension bits are rejected. `load secret` and
`store secret` use opcode tags 26/27 (with a trailing volatile byte), and
`declassify` uses tag 28. A module without secrets encodes exactly as in
version 4 apart from the version number, and version 1–4 streams still
decode.

- **Rejected: a `secret` qualifier on value types.** Every type-directed piece
  of the compiler (interning, the verifier's type agreement, casts, isel's
  register classes, the refinement encoder) would have to learn about it,
  though no operation *computes* differently on a secret. Secrecy is a
  property of data flow, not of representation, so it is declared at the few
  places data enters and derived by an analysis (tenet T4).
- **Rejected: a flag on every instruction.** It would have to be kept
  consistent by every pass that creates an instruction, and one forgotten
  flag is a silent leak. Derived taint cannot be forgotten.
- **Rejected: `classify` as an instruction.** A front end marks its sources
  (parameters, memory, globals, returns); an arbitrary mid-function
  "becomes secret" point has no use a `secret` parameter or load does not
  cover.

### The secret-taint analysis

`analysis::secret::SecretTaint` is a forward analysis on the one lattice
engine (B8): the domain is `Bottom ⊑ Public ⊑ Secret`, a pure operation is the
join of its operands, a block parameter joins its incoming arguments, and
`declassify` is public. The module context the domain's pure transfer cannot
see is supplied through the engine's new `SolveHooks`: which entry parameters
are secret, which direct callees return a secret, and a memory summary.

Memory is conservative and flow-insensitive. Each address is traced to a root:
a stack slot whose address never escapes (it is only used, through
`ptr_add`/`bitcast`/`freeze`, as the address of loads, stores and atomics), a
directly named global, or unknown. A root may hold a secret once a
secret-derived value, or a `store secret`, is stored to it, and a secret global
holds one from the start. A call or syscall that receives a secret argument
taints unknown memory. A load is secret when it is flagged or when its root may
hold a secret. Unknown memory may also be reached through any global stored
to. A global read by name, however, is secret only if it is `secret` or this
function stores a secret to it by name. A write through an unknown pointer
(a `store secret`, or a callee's) declares secret memory, and the modular
contract below makes its readers use `load secret`. This keeps a public
global, such as a green-thread preemption flag, public in a function that
writes secrets through pointers. Value taint and the memory summary are
iterated together to a joint fixpoint.

Taint tracks data flow only. That is sound because the verifier forbids secret
branch conditions: without secret-dependent control flow there is no implicit
flow to track.

Across functions, memory is modular. A secret that crosses a function boundary
through memory is declared at both ends (`store secret`/`load secret`, or a
secret global). Function-pointer types carry no secrecy, so an indirect call's
arguments must be public and its result is public.

### The constant-time verifier

`verify::constant_time` rejects a function in which a secret-derived value
reaches any of the following:

| use | why |
|---|---|
| a `cond_br` condition or a `switch` scrutinee | control flow |
| the address of a `load`, `store` or atomic; the base or offset of a `ptr_add` | cache timing |
| a call target | control flow |
| either operand of `udiv`/`sdiv`/`urem`/`srem` | early-exit dividers on every target |
| any float arithmetic, `fcmp`, or float conversion | subnormal slow paths; x86-64's `u64`↔float conversions branch |
| the size of a `dyn_alloca` | its stack-probe loop runs over the size |
| an operand of a `syscall` | it leaves the program |
| the value of an atomic store, rmw or cmpxchg; an rmw or cmpxchg on memory that may hold a secret | retry loops compare memory contents |
| a public parameter of a direct call; an indirect or variadic argument | secrecy is part of the callee's interface |
| the `ret` of a function whose return is not `secret` | the same |
| an unflagged `store` to memory that is not a non-escaping stack slot or a secret global | secrets cross functions through memory only when declared |
| a shift amount or a multiply operand, under `CtPolicy::STRICT` only | variable-latency shifters and multipliers on some small cores |

Secrets may be used with integer `add`/`sub`/`and`/`or`/`xor`, shifts and `mul`
(under `CtPolicy::DEFAULT`, the policy for x86-64, AArch64 and RV64, where
they are constant-time), `icmp`, integer casts, `bitcast`, `fneg`, `freeze`,
`select`, block arguments, stores to non-escaping slots and secret globals, and
`declassify`. A secret `select` condition is fine because every backend lowers
`select` without a branch (below).

A diagnostic names the value by its `.lf` print name, the use, and the
shortest chain back to the source:

```text
function #0: constant-time violation: secret-derived %7 is the condition of a
`cond_br` (`cond_br` in block ^3); it derives from %3 <- %5 <- %0 <- secret
parameter 0 (use `select`, or `declassify` a value that may be public)
```

`verify_module` runs the check on every module that declares a secret
(`Module::has_secrets`); a module without one is trivially constant-time and
costs nothing. So `lf`, `lf-opt`, and every test that re-verifies after a pass
enforce it.

### Passes preserve it

The audit covered every transform:

- **mem2reg, sccp, simplify_cfg, dce, licm, inline**: no change needed. None
  of them creates a branch, an address or a division out of existing data
  flow. They fold constant branches, straighten blocks, hoist pure operations
  and promote slots. `declassify` is pure, so it can be hoisted or removed
  when dead, but no pass replaces it by its operand. A pass may only lose
  taint, and only where the value really is public: a folded constant, or a
  promoted slot whose stored value was public.
- **egraph** (B4): the rules introduce no branch, no division, and no variable
  shift amount (`x*2^k → x<<k` shifts by a constant). Extraction also compares
  `(tainted, cost)` lexicographically, so in every e-class a representative
  that does not depend on a secret beats a cheaper one that does. A merge can
  therefore never make a public value (a branch condition, an address)
  secret-derived. This guards future synthesized rules; the built-in rules
  never put such a pair in one class.
- **superopt** (B5): `superoptimize_ct` makes the constant-time verifier a
  second gate after `z3rs`. A candidate may not add violations under the
  policy, which matters for synthesized variable shifts under
  `CtPolicy::STRICT`.

The tests run the verifier after every single pass: on a Montgomery-ladder
conditional swap, a constant-time memcmp, and a mixed fixture; on 40 random
constant-time modules through every pass; and on 25 through `-O1..-O3` and
random pass orders.

### Instruction selection

Each target's MIR opcode set has `may_branch_on_data()`, the instructions
that may take a conditional branch depending on a register value:

- x86-64: the terminators, the `u64`↔float fix-ups, the `lock cmpxchg` loop,
  and `dyn_alloca`'s probe loop;
- AArch64: the terminators, the atomic retry loops and `dyn_alloca`'s probe
  loop;
- RISC-V: the terminators, the atomic retry loops and `dyn_alloca`'s probe
  loop (its float compares are `feq`/`flt`/`fle` with `xori`/`and`/`or`, and
  its float-to-integer conversions single saturating instructions, so
  floating point adds no branch);
- Thumb: the terminators only. Its compare-and-set and `select` are `IT`
  blocks (`cmp`/`tst`, `ite`, two `mov`s), which issue every instruction
  whatever the condition, so they are not branches.

Instruction selection creates no blocks of its own. The tests check, on all
three targets, that every function's data-dependent MIR branches are exactly
its IR `cond_br`/`switch` terminators. A straight-line function over every
operation allowed on secrets, at widths 8 to 64 and at `-O0` and `-O2`,
compiles to code with no conditional branch at all. The A64 and RISC-V words
are decoded directly; x86-64 is disassembled with `llvm-mc` when it is
installed.

`select` is `cmov` on x86-64, `csel` on AArch64, a mask blend on RISC-V and
an `IT`-predicated pair of `mov`s on Thumb, asserted at each lowering site.
Thumb is checked on the module it selects from, after soft-float lowering and
64-bit legalization. Those add calls to run-time helpers with public
parameters (`__aeabi_fadd`, `__aeabi_ldivmod`, `__aeabi_lmul`, …), so a secret
reaching one is a violation of the prepared module: float operations and
division are already rejected in the source, and a secret 64-bit multiply,
allowed by the default policy, is caught there (or up front by the strict
policy, the right one for Cortex-M). Every inherently branchy or variable-time
lowering (the float conversions, atomics, `dyn_alloca`, division) has its
secret operands rejected by the verifier.

The same holds under position-independent code, where globals are reached
through the GOT (on x86-64 and AArch64), and under the Win64 convention,
where stack-passed arguments and `xmm` saves are added. The tests
disassemble (or, on AArch64, decode) the whole `.text` of those builds.

### Code generation passes

- **Integer legalization** (§3b) splits wide integers without branching.
  Carry and borrow chains and ordered compares are `icmp`/`select`, a
  variable shift is a funnel shift plus a `select` ladder, and it mirrors the
  CFG block for block. A split `load secret`/`store secret` keeps the flag on
  every part. A wide `mul`/`div`/`rem` becomes a libgcc-style libcall of
  unknown timing whose parameters are public. The verifier therefore rejects
  it on secrets after legalization, which is correct until a runtime provides
  constant-time helpers declared with `secret` parameters.
- **Yield points** (green threads) branch on a volatile load of the public
  preemption flag. The flag stays public even in a function that writes
  secrets through pointers (see the memory model above), so a constant-time
  loop keeps verifying after the pass.

### Limitations

- The guarantee covers code the compiler generates, not microarchitectural
  side channels beyond control flow, addresses and operand-dependent latency
  (speculation, frequency scaling).
- Only the three current targets are covered. `CtPolicy::STRICT` exists for
  cores with variable-time shifts or multiplies; drivers use the default
  policy until such a target lands.
- Memory tracking is flow-insensitive and per function. It can reject a load
  as secret that, in program order, precedes the store that taints its root.
- Floating point stays outside the constant-time subset, and so does a wide
  `mul`/`div`/`rem` on a target that legalizes it into a libcall.

## 6e. SIMD vectors  *(decided)*

A vector `<N x T>` is a first-class **value**: `N ≥ 1` lanes of `i1`, `i8`,
`i16`, `i32`, `i64` or a float type. Unlike an array (an address, §6) it lives
in registers where the target allows, crosses calls and block edges by value,
and is loaded and stored whole. In memory its lanes are packed at the element
size, lane 0 first (an `i1` lane takes one byte holding 0 or 1), each lane in
the data layout's byte order (§3a); it is aligned to its size rounded up to a
power of two, capped at 16 bytes and never below the element's alignment.

### Operations

The ordinary value ops apply **lane-wise** to vectors: the binops (`add` …
`ashr`, `fadd` … `frem`, and the min/max and saturating ops below), `fneg`,
the casts (with equal lane counts), and `freeze`. `icmp`/`fcmp` on `<N x T>`
give `<N x i1>`; `select` takes an `i1` condition (choose a whole arm) or an
`<N x i1>` one (choose per lane). Five ops are vector-only:

```text
%e = extractelement %v, 3 : i32                            ; lane 3
%w = insertelement %v, %x, 0 : <4 x i32>                   ; lane 0 replaced
%s = shufflevector %a, %b, [7, 0, 5, 2] : <4 x i32>        ; lanes of a ++ b
%b = splat %x : <4 x i32>                                  ; broadcast
%r = reduce add %v : i32      ; add mul and or xor smin smax umin umax fadd fmul
```

Lane indices and shuffle masks are **constants** (the verifier checks them in
range), and a shuffle may change the lane count (`M = mask.len()`). The
integer reductions are associative, so their order is unobservable; `reduce
fadd`/`fmul` are **ordered** (`((l0 op l1) op l2) …`, each step rounded), and
a `reassoc` flag licenses any other order. Vector constants are ordinary
operands: `<4 x i32> (i32 1, i32 poison, i32 3, i32 4)`.

Eight integer binops, valid on scalars and vectors, carry the common SIMD
idioms: `smin`, `smax`, `umin`, `umax`, and the saturating `sadd_sat`,
`uadd_sat`, `ssub_sat`, `usub_sat` (the exact result clamped to the type's
range). They are never poison except from a poison operand.

- **Rejected: dynamic lane indices.** SSE and NEON take lane numbers as
  immediates; a dynamic index goes through memory anyway, which a front end
  writes explicitly (`store` + `ptr_add` + `load`). Constant indices keep the
  op's meaning total (no out-of-range case) and trivially checkable.
- **Rejected: an "undef" shuffle-mask entry.** There is no `undef` (§5); a
  front end that does not care about a lane picks any in-range index.
- **Rejected: intrinsics for min/max/saturation.** An opaque call hides the
  meaning from every pass and from the verifier; eight real opcodes with a
  reference semantics cost little and lower directly on SSE2/NEON.

### Semantics: poison per lane

This resolves the §10 open question in favour of **per-lane poison**. The
reference evaluator carries a vector as one value per lane, so:

- lane-wise ops apply the scalar poison rules **per lane**: an over-wide shift
  amount, a violated `nsw`/`nuw`/`exact`, an out-of-range `fptosi` poison only
  that lane;
- the scalar UB rules apply to the **whole instruction**: a zero divisor (or
  `INT_MIN / -1`) in any lane makes the `udiv`/`sdiv`/`urem`/`srem` UB;
- `select` with a vector condition poisons only the lanes whose condition lane
  is poison; `freeze` fixes each poison lane (to zero in the evaluator);
- `extractelement` is poison iff the lane is; `insertelement` defines the
  written lane even in a poison vector; a shuffle lane is poison iff the lane
  it picks is; `splat` of poison is all-poison; `reduce` is poison if any lane
  is;
- `bitcast` reinterprets the packed bits (lane 0 least significant, as in
  memory on a little-endian machine), and a poison source lane poisons exactly
  the result lanes (or the scalar) it overlaps.

A whole-value `poison` of vector type means every lane is poison.
Refinement (§5) is lane-wise: a target vector refines a source vector iff each
lane refines. The refinement checker (B2) encodes scalars only, so it reports a
function using vector ops as `Unknown`; the min/max/saturating ops have exact
SMT encodings.

### Verification, text and binary form

The verifier checks vector well-formedness (lane count `1..=65536`, lane types
above), lane-count agreement of lane-wise ops and compares, `<N x i1>` select
conditions against `N`-lane arms, lane/mask ranges, `bitcast` total widths,
and forbids `volatile` vector accesses (a vector access may be split into
lanes, which "exactly once, at exactly the width" cannot allow) and vector
atomics. In `.lfb`, the vector type is type tag 16 and the vector ops are
opcode tags 40–44 (the min/max/saturating binops are binop codes 18–25).
These are new tag values only, so the format stays at version 5: a stream
without vectors is unchanged, and an older reader rejects a vector stream's
tags as invalid rather than misreading them.

### Legalization

Each backend runs a target-independent IR→IR **legalizer**
(`codegen::legalize`) before instruction selection, parameterized by the
target's `VectorLegality`: which vector types it holds whole in a register and
which ops on them it selects. Everything else is rewritten, so correctness
never depends on the ISA having an instruction:

1. min/max/saturating ops without a direct form are expanded into compares,
   selects and wrapping arithmetic of the same type (a vector `smin` stays a
   vector `icmp` + `select` where those are legal);
2. illegal vector types are **split into lanes**: block parameters, edge
   arguments, loads/stores (element accesses at `i × size`, alignment reduced
   to what each offset guarantees), lane-wise ops, lane moves, reductions (an
   in-order chain), and `bitcast` (rebuilt from lane bits with shifts,
   truncations, extensions and ors — exact for `i1` lanes too);
3. ops on legal types the target does not select are scalarized in place
   through `extractelement`/`insertelement`.

Illegal vector types in signatures follow one convention on every target: a
parameter is passed as its `N` lanes, and a result is returned through a hidden
leading `ptr` to caller-allocated storage. Both sides of every call obey it, so
LatticeFoundry code agrees with itself; C interop for a vector type requires
the target to make that type legal.

### Lowering

| target | legal vector types | lowering |
|---|---|---|
| x86-64 (SSE2, the baseline) | `<16 x i8>`, `<8 x i16>`, `<4 x i32>`, `<2 x i64>`, `<4 x f32>`, `<2 x f64>`, masks `<16/8/4/2 x i1>` | see below; System V passes them in `xmm0..7` and returns in `xmm0` (`__m128`); Win64 passes them by reference (a 16-byte-aligned caller copy) and returns in `xmm0` (§6c) |
| AArch64 (NEON) | the same ten | see below; passed in `v0..v7`, returned in `v0` (AAPCS64 short vectors) |
| RISC-V | none (the V extension is out of scope) | fully scalarized |
| Thumb (Cortex-M) | none | fully scalarized, before soft-float lowering and 64-bit legalization |

On both SIMD targets a mask `<N x i1>` lives in a vector register as `N`
lanes of `128 / N` bits, each all-ones or all-zeros — what their compares
produce and their bitwise blends consume (`codegen::simd128`). On x86-64 no
instruction beyond SSE2 is used:

| IR | SSE2 |
|---|---|
| `add`/`sub` | `padd{b,w,d,q}` / `psub{b,w,d,q}` |
| `mul` i16 / i32 | `pmullw` / `pmuludq` ×2 + `pshufd` + `punpckldq` (no SSE4.1 `pmulld`) |
| `and`/`or`/`xor` | `pand`/`por`/`pxor` |
| `shl`/`lshr` i16–i64, `ashr` i16/i32, uniform constant amount | `psll`/`psrl`/`psra` `imm8` |
| `umin`/`umax` i8, `smin`/`smax` i16 | `pminub`/`pmaxub`, `pminsw`/`pmaxsw` |
| saturating add/sub i8/i16 | `padds`/`paddus`/`psubs`/`psubus` |
| float `fadd`/`fsub`/`fmul`/`fdiv`, `fneg` | `addps`/`addpd`…, `xorps` with the sign bit |
| `icmp` i8–i32 | `pcmpeq`/`pcmpgt` (operands swapped, results negated, sign-flipped for unsigned) |
| `icmp eq`/`ne` i64 | `pcmpeqd` + `pshufd` + `pand` |
| `fcmp` | `cmpps`/`cmppd` (`one`/`ueq` as two compares) |
| `select` | `pand`/`pandn`/`por` (an `i1` condition broadcast with `movd` + `pshufd`) |
| `sitofp`/`fptosi` i32↔f32 | `cvtdq2ps`/`cvttps2dq` |
| mask ↔ int of the lane width | a copy (`sext`), `pand` (`zext`), `pand` + `pcmpeq` (`trunc`) |
| `extractelement` / `insertelement` | `pshufd` + `movd`/`movq`, `pextrw` / `pinsrw`, `movsd`, `punpcklqdq`, `unpcklpd` |
| `shufflevector` 32/64-bit lanes | `pshufd`, `shufps`, `shufpd` |
| `splat` | `movd`/`movq` + `pshufd` |
| `load`/`store` | `movdqa` (align ≥ 16) / `movdqu` |

Everything else (division, `frem`, byte shifts, variable shifts, `mul` on
bytes and quadwords, ordered i64 compares, other casts and shuffles,
reductions, mask loads/stores) is scalarized. xmm spills are 16 bytes.

NEON is far more regular, so AArch64 scalarizes much less:

| IR | NEON |
|---|---|
| `add`/`sub`/`and`/`or`/`xor`, `mul` i8–i32 | `add`/`sub`/`and`/`orr`/`eor`, `mul` |
| shifts, uniform constant / per lane | `shl`/`ushr`/`sshr #imm` / `ushl`, `ushl`/`sshl` by `neg` |
| min/max i8–i32, saturating add/sub (all widths) | `smin`/`smax`/`umin`/`umax`, `sqadd`/`uqadd`/`sqsub`/`uqsub` |
| float arithmetic, `fneg` | `fadd`/`fsub`/`fmul`/`fdiv`, `fneg` |
| `icmp` (all widths), `fcmp` | `cmeq`/`cmgt`/`cmge`/`cmhi`/`cmhs`, `fcmeq`/`fcmgt`/`fcmge` (+ `mvn`/`orr` for the unordered forms) |
| `select` | `and` + `bic` + `orr` |
| int ↔ float of the lane width | `scvtf`/`ucvtf`/`fcvtzs`/`fcvtzu` |
| lane moves | `umov`, `dup`, `ins`; any same-length shuffle via `tbl` (+ `orr` for two sources) |
| `reduce add`/min/max | `addv`/`sminv`/…, `addp` for i64 |
| `load`/`store` | `ldr q`/`str q` |

A function holding vectors treats `v8..v15` as clobbered by calls (AAPCS64
preserves only their low halves), and `v` spills are 16-byte `str q`.
Scalarized on AArch64: division, `frem`, i64 `mul` (i64 min/max expand to a
vector compare and blend), other casts, lane-count-changing shuffles, and the
other reductions.

### Constant time

Vectors follow §6d lane by lane: the secret taint flows through lane-wise ops,
lane moves and reductions like through their scalar forms, and the verifier
rejects the same uses (a vector division, a float or multiply reduction, a
branch on an extracted secret lane). A vector `select` on a secret mask is a
bitwise blend (`pand`/`pandn`/`por`, NEON `and`/`bic`/`orr`), scalarized
selects are the targets' branchless selects, and none of the vector machine
ops branches (each backend's `may_branch_on_data` audit lists them). Split
loads and stores keep their `secret` flag.
## 6f. wasm32: a stack-machine target  *(decided)*

WebAssembly has typed locals instead of registers and structured control flow
instead of jumps, so `target::wasm32` does **not** go through MIR, instruction
selection and register allocation. It lowers the SSA IR directly, clean-room
from the WebAssembly Core Specification and the tool-conventions linking
format:

- **Layout.** ILP32 with native `i32`/`i64` and a 16-byte stack
  (`e-p:32:32-…-S128-n32:64`, `wasm32::data_layout`); linear memory is address
  space 0, the only one. `lf build --target wasm32` gives a module without a
  `datalayout` line this layout.
- **Control flow.** A structurizer places each function's blocks from the
  dominator tree and a reverse postorder: a backward edge targets a loop
  header wrapped in `loop`; a node with several forward in-edges (a merge
  node) follows a `block` its immediate dominator's code branches out of; any
  other node is emitted inline at its one incoming edge. Two-way branches use
  `if`/`else`, switches nested blocks with a `br_table` (dense cases) or a
  `br_if` chain. An **irreducible** CFG becomes a dispatch loop (`loop` +
  `br_table` over a label local).
- **Values.** Every SSA value is a local of its wasm type and every block
  parameter a local assigned on the incoming edges (all arguments are pushed
  before any parameter is written: a parallel copy). A pure value with a single
  use in its own block is recomputed there as a stack expression instead.
  Values whose live ranges do not overlap share a local: SSA interference is
  "live at the other's definition", so a greedy pass in dominator-tree
  preorder over liveness computed on the emitted code (block parameters live
  from the incoming edges, inlined leaves read at their root's user) assigns
  them without conflicts.
  Integers up to 32 bits live in an `i32`, up to 64 in an `i64`, always
  **zero-extended**: operations that can set bits above the width mask them,
  signed operations sign-extend their operands first, and parameters of
  non-`internal` functions and results of host or indirect calls are masked on
  arrival. Wider integers are legalized into `i64` parts (§3b); what stays wide
  at the ABI seam is a group of `i64` locals, several parameters, and a
  multi-value result.
- **Memory.** Loads and stores use the narrow `load8_u`… forms with alignment
  hints; odd sizes (3, 5, 6, 7 bytes) are split. `alloca`/`dyn_alloca` live on a
  **shadow stack** under the mutable global `__stack_pointer`, restored before
  every return. Globals are data segments (`.rodata`, `.data`, `.bss` from the
  shared emitter), their address a relocated `i32.const`.
- **Calls.** Direct `call`; a function pointer is a slot of the function table
  (slot 0 is empty so a null call traps) and an indirect call a
  `call_indirect`. Undefined functions are imported from `"env"`; `frem` calls
  `fmod`/`fmodf`.
- **Other ops.** `fptosi`/`fptoui` use the saturating conversions (out of
  range is poison, so they must not trap); volatile accesses are plain
  accesses; atomics use the threads proposal, with `nand`/`max`/`min`/`umax`/
  `umin` as compare-exchange loops and every fence an `atomic.fence`. `syscall`,
  `f16`, other address spaces and variadic calls are errors.
- **Vectors** (§6e). wasm32 declares no legal vector type, so the generic
  legalizer scalarizes all vector code before lowering (SIMD128 is not used
  yet); min/max and saturating ops are expanded by the same pass.
- **Constant time** (§6d). There is no MIR to audit, so the guarantee is
  structural: the only conditional control flow the backend emits (`if`,
  `br_if`, `br_table`) comes from IR `cond_br`/`switch` terminators (and the
  atomic retry loops, which the verifier rejects on secrets); `select` is always
  the branchless wasm `select`, and narrow-value masking, sign extension and
  the `i128` expansion (a `select` ladder for variable shifts) are straight-line.
  `declassify` is the identity. Tests decode the emitted bodies to check it.
- **Output.** A self-contained module (memory, stack pointer, table, data;
  exports `memory`, `__heap_base`, `main` and the default/protected-visibility
  functions) whose shadow stack sits at the bottom of memory, so an overflow
  wraps below 0 and traps; or a relocatable object (`linking` symbol table and
  segment info, `reloc.CODE`/`reloc.DATA`, padded 5-byte LEBs at every
  relocated field, memory/table/`__stack_pointer` imported) for `wasm-ld`.

- **Rejected: a MIR-based wasm backend.** Register allocation has nothing to
  allocate on wasm, and the MIR's flat blocks with jumps would have to be
  re-structured anyway; lowering from the SSA IR keeps the dominator tree,
  block arguments and value types that the structurizer and local assignment
  need.
- **Rejected: node splitting for irreducible control flow.** It can blow up
  code size exponentially; a dispatch loop is linear, and irreducible CFGs are
  rare in front-end output.

## 6g. AVR: an 8-bit Harvard target  *(decided)*

`target::avr` targets the AVR5 core (ATmega328P) through the ordinary
MIR/regalloc pipeline, clean-room from the AVR Instruction Set Manual, the
published avr-gcc ABI and the ELF/AVR relocation list:

- **Layout.** `e-p:16:8-p1:16:8-i8:8-…-S8-n8:16-P1`: 16-bit pointers into
  data memory (space 0) and program memory (space 1, the program space);
  every alignment is one byte. A data pointer into flash is a byte address
  (`lpm`), a function pointer a word address (§3a). A module without a layout
  gets the same one with functions in space 0.
- **Registers are pairs.** One vreg is one register pair: `i8` in its low
  register (the high one don't-care), `i16`/pointers in the pair; the
  avr-gcc argument and result registers are all even-aligned, so the ABI maps
  onto pairs exactly. Integers above 16 bits go through §3b at `W = 16`.
- **Preparation**, in order: vector legalization (no legal vector type),
  soft float (`codegen::softfloat` with libgcc names and a 16-bit `int`),
  integer legalization. Division, and multiplication `mul` cannot do, are
  runtime calls; the runtime is LF IR compiled by the backend, one object per
  function so the firmware linker takes only what a program uses.
- **Constant time, per operation.** Flash is scarce, so the branch-free
  lowerings are used only where a secret is: isel runs the secret-taint
  analysis on the prepared function, and an operation with a secret-derived
  operand gets the constant-time form — the compare's flag read out of
  `SREG`, a barrel shifter, a shift-pair sign extension — while public ones
  keep the compact forms (a skip over an `ldi`, a counted loop, `sbrc`).
  `select` is always a mask blend.
  Helper calls take public parameters, so a secret wide multiply shows up as a
  violation in the prepared module; the 8/16-bit multiply helpers (called on
  a core without `mul`, possibly on secrets) take a fixed number of
  iterations.

- **Rejected: 8-bit allocation units.** A register per byte would make every
  pointer and `i16` two vregs, which the one-vreg-per-value isel and the
  linear-scan allocator do not model; pairs cost an unused register for each
  live `i8`.
- **Rejected: always branch-free.** It costs every program flash (the
  soft-float runtime alone no longer fit an ATmega328P) for the few that
  handle secrets; the taint analysis already knows which operations those
  are.
- **Rejected: word-addressed data in program memory.** `lpm` reads bytes, so
  flash data pointers are byte addresses; only functions use word addresses,
  matching avr-gcc's function pointers.

## 6h. RISC-V: RV64GC and the LP64D convention  *(decided)*

`target::riscv` targets RV64GC (the I base with the M, A, F, D and C
extensions) under the LP64D psABI, through the ordinary MIR/regalloc
pipeline, clean-room from the RISC-V ISA manual and the RISC-V ELF psABI:

- **Floating point.** `f32`/`f64` live in `f0`–`f31` (a single NaN-boxed).
  The IR fixes the value of every float result but not the payload of a NaN,
  and RISC-V arithmetic returns the canonical NaN, so no fix-up is needed;
  `fneg` is a sign injection (`fsgnjn`), exact for NaNs as the IR requires.
  A multiply feeding an add or subtract becomes one fused instruction only
  when **both** carry `contract` (the IR's only license to skip a rounding),
  and only when the multiply has no other use. `fptosi`/`fptoui` are the
  saturating `fcvt` with `rtz` (out of range is poison, so saturation refines
  it); `frem` is a call to C's `fmod`/`fmodf`, declared by the backend when a
  module needs it. `f16` would need Zfh and is rejected.
- **LP64D.** An argument is placed part by part: a scalar in the next `a`
  register, a named float in the next `fa` register and then like an integer
  of its size (integer register, then an 8-byte stack slot), a variadic float
  like an integer. A struct (flattened through nested structs and arrays)
  whose fields are one float, two floats, or a float and an integer of at
  most 8 bytes goes in one `fa`/`a` register per field when enough remain;
  otherwise up to 16 bytes travel as one or two 8-byte integer chunks (split
  between `a7` and the stack if need be) and anything larger by reference.
  A result is placed as a first argument would be with only `a0`/`a1` and
  `fa0`/`fa1`; one that does not fit comes back through memory whose address
  is a hidden `a0` argument. A struct value is, in the backend, a pointer to
  its storage; the call site classifies by the callee's parameter types, so a
  plain pointer may be passed where a struct is expected.
- **Addressing.** Everything is PC-relative (the `medany` code model): a call
  is `auipc ra`+`jalr` under `R_RISCV_CALL_PLT`, an address `auipc`+`addi`
  with `R_RISCV_PCREL_HI20`/`R_RISCV_PCREL_LO12_I` — the low part's symbol is
  a local label on the `auipc`, not the target — or a GOT load under PIC
  (§4b). No `R_RISCV_RELAX` is emitted, so linking never depends on the
  global pointer.
- **Frames.** From `sp` up: the outgoing stack arguments, `ra` and the
  callee-saved `s`/`fs` registers, the slots. A function using `dyn_alloca`
  reserves `s0` as a frame pointer (slots and incoming arguments are
  addressed from it) and keeps the outgoing area at the bottom of the moving
  `sp`; with probes, it touches the current top with a load (`0(sp)` may hold
  a saved register) and then every new page.
- **The C extension** is a final encoding choice: with it on, every
  instruction that has a 16-bit form takes it, except those a relocation
  patches, branches and jumps (their displacements are fixed up after
  layout), and the word-counted loops of the probe, `dyn_alloca` and LR/SC
  expansions. The object then carries `EF_RISCV_RVC`.
- **Validation.** The host cannot run RISC-V code, so an instruction-set
  simulator with its own decoder (and its own compressed-instruction
  expander) runs the linked machine code — linked by a test linker, or by qld
  and loaded from the file with its dynamic relocations applied —
  differentially against the reference evaluator and a MIR interpreter, and
  runs clang-compiled `rv64gc` C calling the backend's functions and back.

- **Rejected: relaxation.** `R_RISCV_RELAX` lets the linker shorten
  `auipc` pairs into `jal` or `gp`-relative forms; it is an optimization whose
  `gp` form needs startup code to set `gp`, and unrelaxed code is correct
  everywhere.
- **Rejected: compressed branches.** `c.j`/`c.beqz` would need a relaxation
  loop over block layout (their ranges are short); branches are a small part
  of the code.

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
- **Vector poison granularity.** *(decided: per lane, §6e.)*
- **Address-space semantics.** *(decided, §3a: per-space pointer widths, no
  `addrspacecast`.)*

Each open question is resolved *before* the opcode or feature it governs is
frozen, and its resolution is added above with the same option/rejected/why
structure.
