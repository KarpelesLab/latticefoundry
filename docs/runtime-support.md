# Runtime support: green threads and preemption

LatticeFoundry gives a front end with green threads (Lode first, issue #3) two
things:

1. **Context switching routines**, emitted as machine code by LF itself. They
   are not a C runtime and need no libc.
2. An opt-in **yield-point pass** that makes long-running loops check a
   "preempt requested" flag.

## 1. Why runtime functions and not IR intrinsics

Each backend provides the routines as a small object: `.text.lf_rt` holds the
code, with one global symbol per routine. The front end links that object next
to its own compiled module:

```rust
use latticefoundry::target::x86_64::{compile_module, runtime};
let objs = vec![compile_module(&m, &syms), runtime::context_runtime_object()];
// or: runtime::emit_context_runtime(&mut my_object);
```

The IR only *declares* them (`func @lf_ctx_switch(ptr, ptr) -> void`) and calls
them like any other external function. We chose this over IR intrinsics for
three reasons:

- **Every optimizer pass already handles them correctly.** A call to an
  external function is opaque. It clobbers all memory and makes its pointer
  arguments escape. Nothing is moved across it, and nothing is inlined into or
  out of it. That is exactly the full barrier a context switch needs, and no
  pass had to change.
- **The code is target-specific.** It saves registers and uses kernel
  structure layouts. Isel lowering would scatter that code across three
  backends and the MIR, while one function per target stays auditable.
- **It is still LF output.** The routines are encoded by hand with the same
  emitter and object model as the rest of the backend, and they link through
  LF's own static linker.

## 2. The context (`LfCtx`), layout version 1

Each target documents its layout in `target::<arch>::runtime::layout`, with a
constant for every offset. The version word lets a later layout grow the
struct. A context must be **16-byte aligned**. In IR, declare it as
`[N x i128]`, which is 16-aligned (for x86-64, `[42 x i128]`, 672 bytes).

| target  | size | contents                                                                 |
|---------|------|--------------------------------------------------------------------------|
| x86-64  | 672  | 16 GPRs (encoding order), `rip`, `rflags`, version, kind, 512-byte `fxsave64` image (x87, MXCSR, `xmm0..15`) |
| AArch64 | 800  | `x0..x30`, `sp`, `pc`, `NZCV`, version, kind, `FPSR`, `FPCR`, `q0..q31`   |
| RV64    | 528  | `pc` (in the `x0` slot), `x1..x31`, `fcsr`, version, kind, `f0..f31`     |

A context has one of two **kinds**:

- A *cooperative* context holds only what the ABI makes callee-saved across a
  call: the callee-saved GPRs, `sp`, the resume address, the FP control state,
  and on AArch64/RISC-V the callee-saved FP registers.
- A *full* context holds everything.

Every restore dispatches on the kind, so threads saved either way can resume
one another.

**x86-64 vector state.** Version 1 saves the x87 and SSE state with `FXSAVE`,
the baseline every x86-64 CPU has. It does not save AVX state: the upper halves
of `ymm`/`zmm` and the AVX-512 mask registers. LF emits no AVX code, so no
LF-compiled thread has live state there. An `XSAVE` area sized from CPUID leaf
`0xD` can be appended in a later version, flagged in the reserved word at
`0x98`. Writing a context into a signal frame clears the frame's AVX
`XSTATE_BV` bits, so the resumed thread gets the initial (zero) AVX state
rather than the interrupted thread's.

## 3. The routines

| symbol                  | signature                                    | what                                               |
|-------------------------|----------------------------------------------|----------------------------------------------------|
| `lf_ctx_save`           | `(ctx) -> i64`                               | cooperative save; returns 0, then 1 when resumed (like `setjmp`) |
| `lf_ctx_save_full`      | `(ctx) -> i64`                               | full save; returns 0, then 1                        |
| `lf_ctx_restore`        | `(ctx) -> !`                                 | resume a context of either kind                    |
| `lf_ctx_switch`         | `(from, to)`                                 | the green-thread switch (cooperative fast path)    |
| `lf_ctx_switch_full`    | `(from, to)`                                 | switch that saves every register, as at the call   |
| `lf_ctx_init`           | `(ctx, stack_top, entry, arg)`               | a fresh thread: calls `entry(arg)` on `stack_top` (rounded down to 16); `exit_group(result)` if it returns |
| `lf_ctx_from_ucontext`  | `(ctx, uc)`                                  | x86-64: capture a signal's interrupted state        |
| `lf_ctx_to_ucontext`    | `(uc, ctx)`                                  | x86-64: make `rt_sigreturn` resume `ctx`            |
| `lf_ctx_preempt`        | `(uc, from, to)`                             | x86-64: both of the above                          |
| `lf_ctx_uc_in_runtime`  | `(uc) -> i64`                                | x86-64: whether the signal hit inside these routines |
| `lf_sig_install`        | `(signo, handler, flags) -> i64`             | x86-64: `rt_sigaction` with `SA_SIGINFO`, `SA_RESTORER` and `flags` |
| `lf_sig_restorer`       | none (the signal return trampoline)           | x86-64: `rt_sigreturn`                             |

Like `setjmp`, a function that calls `lf_ctx_save*` can rely after the second
return only on values it did not change after the save. The frame's spill slots
may have been reused in between. `lf_ctx_switch` has no such caveat.

**Resuming a full context without losing a register.** On x86-64, the restore
writes `rflags` and `rip` *below the target's 128-byte red zone* and points
`rsp` at them. It then loads all 16 GPRs and finishes with `popfq; ret 128`.
Nothing is lost, even for a thread that a signal interrupted in the middle of a
leaf function.

A64 and RISC-V cannot branch without a register. There, a full restore gives
back every register except the one that carries the target address: `x17`
(IP1) on AArch64 and `t6` on RISC-V. Contexts taken at a call are unaffected,
because that register is dead at a call. On RISC-V, `gp` and `tp` belong to the
process or OS thread and are never restored.

## 4. Preemption from a timer signal (x86-64 Linux)

1. `lf_sig_install(SIGALRM, handler, SA_RESTART)` issues `rt_sigaction` with
   `SA_SIGINFO | SA_RESTORER` and `lf_sig_restorer`. Without libc, nothing else
   supplies the restorer. Arm the timer with `setitimer`, through the IR
   `syscall` instruction.
2. The kernel pushes a signal frame on the interrupted thread's stack. The
   frame holds a `ucontext_t`, whose `uc_mcontext` is a `struct sigcontext`
   with the GPRs, `rip` and `eflags`, plus a pointer to the FP/XSAVE state. The
   kernel then calls `handler(signo, info, uc)`.
3. The handler calls `lf_ctx_preempt(uc, current, next)`. This copies the
   interrupted registers and the FXSAVE part of `fpstate` into `current`,
   producing a full context. It normalizes x87/SSE state that `XSTATE_BV` marks
   as initial. It then writes `next` into the `ucontext` and `fpstate`, keeping
   the kernel's `sw_reserved` bytes and editing `XSTATE_BV` as described above.
4. The handler returns into the restorer. `rt_sigreturn` loads the edited
   frame, so **the kernel resumes `next`**, with the signal mask recorded in
   the frame.
5. Later, a switch back to `current` resumes it exactly where the signal hit.
   That switch can be `lf_ctx_switch` from another thread or another signal.

The handler must not preempt a thread that is in the middle of a switch.
`lf_ctx_uc_in_runtime(uc)` reports a signal that landed inside the routines.
The scheduler's own critical sections need a "preemption disabled" flag that
the handler checks.

The kernel layouts come from the documented Linux UAPI ABI:

- `uc_mcontext` is at offset 40.
- The `sigcontext` order is `r8..r15, rdi, rsi, rbp, rbx, rdx, rax, rcx, rsp,
  rip, eflags`, and `fpstate` is at offset 184.
- `FP_XSTATE_MAGIC1` is at `fpstate + 464`, and `XSTATE_BV` is at
  `fpstate + 512`.

**AArch64 and RISC-V:** the ucontext mapping is not provided yet. On arm64 the
vector state is a `fpsimd_context` record, magic `0x46508001`, in
`uc_mcontext.__reserved`, possibly followed by SVE/ZA records that would also
need handling; the kernel supplies a vDSO restorer. On riscv64 the FP state
follows `user_regs_struct` in the `sigcontext`.

## 5. Yield points (`transform::yield_points`)

`YieldPoints` puts this check on every back edge of each loop whose cost is not
proven small:

```text
  %f = load volatile @flag : i32 ; cond_br (%f != 0), ^call, ^header
  ^call: call @yield_fn() ; br ^header
```

Use it through a `YieldConfig`:

- `YieldConfig { flag, yield_fn, max_cost }` names the flag global and the
  yield function by symbol. If they are absent, the pass declares them: the
  flag as an external `i32`, and the function as `() -> void`.
- `analyze_loops` returns each loop's trip bound and cost.
- `optimize_with_yield_points(module, level, config)` runs the `-O` pipeline
  and then the pass.

The pass is in no default pipeline because it adds an observable call: it is
not a refinement of its input. It is also idempotent.

**Which loops get a check** is decided by a first slice of bet B9, a cost
lattice `Cost = Bounded(n) ⊑ Unbounded` with saturating arithmetic:

- Every instruction has a static cost. A call costs 10, because the callee's
  loops carry their own checks.
- A loop costs `trip_bound × (its own blocks + its inner loops)`.
- A loop gets a check unless its cost is proven to be at most `max_cost`
  (default 10 000).

The trip bound comes from the analysis engine (bet B8). The loop needs an exit
test that every iteration passes: `icmp` of a header induction variable,
stepped by a positive constant, or of its stepped value, against a
loop-invariant bound. The bound's range is taken from the ranges and
known-bits domains, so a limit of `and %n, 15` bounds the loop as well as a
constant does.

The analysis is conservative: every cycle whose cost is not bounded passes a
check. Unbounded recursion is not covered.

## 6. Tests

- **x86-64**: static executables with no libc, built and linked by LF.
  - Cooperative ping-pong between two green threads on `mmap`ed stacks, at
    O0 to O3.
  - A thread entry that returns.
  - `save` returning twice, for both kinds.
  - `lf_ctx_switch_full` preserving every GPR, the arithmetic flags, and
    `xmm0..15` against a thread that clobbers them all.
  - Preemption: `setitimer` sends `SIGALRM` into a hand-assembled busy loop
    that holds distinctive values in every register. The handler switches to
    thread B through the ucontext. B clobbers everything and switches back
    with `lf_ctx_switch`. A resumes with every register and `xmm` intact,
    checked register by register and with a checksum.
  - The yield pass end to end: a timer handler sets the flag, and the inserted
    check calls a yield function that switches green threads.
- **AArch64 / RISC-V**:
  - The whole runtime listing, assembled by `llvm-mc`, matches our bytes.
  - A small simulator of exactly the instructions the runtime uses runs full
    save/restore round trips, the double return, and a switch into a fresh
    thread and back to its `exit_group`.
