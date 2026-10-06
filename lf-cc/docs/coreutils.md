# Building GNU coreutils with lf-cc

GNU coreutils 9.5, with the gnulib it bundles, configures and builds with
lf-cc as its only C compiler: every program the gcc build produces, and `libstdbuf.so`.
The only things not built are the two optional SIMD helper libraries
(`libcksum_pclmul.a`, `libwc_avx2.a`): configure finds that the
`<immintrin.h>` intrinsics are missing and leaves them out. No source
patches or extra flags are needed.

## Reproduction

```sh
# lf-cc itself
cd lf-cc && CARGO_TARGET_DIR=$PWD/target cargo build --release
LFCC=$PWD/target/release/lf-cc

# coreutils, out of tree
curl -LO https://ftp.gnu.org/gnu/coreutils/coreutils-9.5.tar.xz
tar xf coreutils-9.5.tar.xz
mkdir build-lf && cd build-lf
../coreutils-9.5/configure CC=$LFCC
make -j4
make -j4 check            # tests/ then gnulib-tests/
```

Configure detects `$LFCC -E` as the preprocessor and `gcc3` as the dependency
style (`-MT -MD -MP -MF`). A complete rebuild takes seconds.

## Test-suite results (2026-10-04, x86-64 Gentoo, glibc, non-root)

| Suite                 | Compiler   | PASS | SKIP | FAIL | ERROR |
| --------------------- | ---------- | ---: | ---: | ---: | ----: |
| `tests/` (653)        | gcc 15.3   |  543 |  110 |    0 |     0 |
| `tests/` (653)        | lf-cc      |  524 |  118 |   10 |     1 |
| `gnulib-tests/` (494) | gcc 15.3   |  450 |   44 |    0 |     0 |
| `gnulib-tests/` (494) | lf-cc      |  450 |   44 |    0 |     0 |

Almost every remaining failure has one cause: lf-cc's `long double` is a
`double`, but glibc's `long double` is the x87 80-bit format.

- **The 80-bit `long double` (most of the failures).** gnulib copes with
  it inside the program (`HAVE_SAME_LONG_DOUBLE_AS_DOUBLE`, plus its own
  `strtold`, `frexpl` and `printf`). But glibc's fortified `printf` family
  (`__printf_chk`, chosen because gnulib's `config.h` defines
  `_FORTIFY_SOURCE` at `-O2`) still receives a `double` where it reads
  `%Lf`/`%Lg` as an 80-bit value. That breaks:
  - `seq` (`seq.pl`, `seq-extra-number`, `seq-locale`, `seq-precision`);
  - `numfmt.pl` and `sort-float`;
  - `sort -g` on NaNs: `nan_compare` gets inconsistent results, and
    `sort-NaN-infloop` hangs intermittently;
  - `tests/init.sh`'s `getlimits` reports `LDBL_MAX=0`, which breaks
    `sleep.sh` and `timeout-large-parameters`;
  - tests that build their input with `seq`: `cksum.sh`, `du/inodes.sh`,
    and the set-up of `ls/abmon-align` (the ERROR).

  The extra skips include `csplit-heap`, `cut-huge-range`, `dd/no-allocate`
  and `printf-surprise`: their `get_min_ulimit_v_` loops over `seq`.
  `seq-long-double` also skips, because it needs `long double != double`.

  The fix is a true x87 `long double` in the backend: an IR `f80` type,
  x87 code, and the System V memory-class ABI with an `st(0)` return.
- **Debug info across objects (fixed since this run).** `rm/r-root` sets a
  gdb breakpoint on a line of `remove.c`. The optimizer keeps the line table
  at `-O2` (issue #19; `tail/inotify-race{,2}` pass). In this run the test
  still skipped: lf-cc wrote DWARF section offsets (`DW_FORM_strp`,
  `DW_AT_stmt_list`, the unit's abbrev offset) without relocations. After
  the link, every compile unit therefore read the first object's strings
  and line table, and gdb found no `remove.c`. These offsets are now
  relocated against their section's symbol, so each object keeps its own
  compile unit. A small repro of the same shape passes: `-O2 -g` objects
  plus an archive member, with gdb stopping at a `remove.c` line in the
  second object (`separate_debug_objects_keep_their_compile_units` in
  `tests/driver.rs`). The full suite has not been rerun since.

## What coreutils needed

These lf-cc gaps were fixed, each with a test in `tests/coreutils.rs`:

- **Driver:**
  - `-E` with line markers and `#pragma` pass-through;
  - `-M`, `-MM`, `-MD`, `-MMD`, `-MF`, `-MT`, `-MQ` and `-MP`;
  - `-x <lang>`, and `-` to read standard input.
- **Preprocessor:**
  - a block comment spanning lines inside a directive;
  - `#if`/`#ifdef` inside a macro's arguments;
  - characters that start no token (an error only if they reach the parser);
  - line splices inside string literals;
  - `#pragma weak`;
  - `[[...]]` attributes in the GNU dialects.
- **Language:**
  - block-scope VLAs, whose stack is reused per declaration so a loop
    stays bounded;
  - `conflicting types` on incompatible redeclarations, which configure's
    signature probes rely on;
  - member access on struct rvalues;
  - a struct member initialized from a struct expression;
  - designator chains, and designators into anonymous members;
  - address constants with arithmetic, `&function`, and block-scope
    statics in static initializers;
  - `sizeof` of members and block-scope objects in static initializers;
  - floating constants cast in integer constant expressions;
  - `_Static_assert` messages made of adjacent literals;
  - `__attribute__((constructor/destructor))`;
  - an error for arrays larger than `PTRDIFF_MAX`;
  - discarded struct lvalues;
  - conditionals of struct-returning calls.
- **Root crate:** the e-graph extractor now saturates the tree cost
  instead of leaving deep shared DAGs unextractable. gnulib's `sm3.c`
  panicked at `-O2`.

Not implemented: variably modified types beyond a block-scope array's
outer bound (VLA parameters `int a[n][n]`, pointers to VLAs, VLA
typedefs). So configure's separate VLA probe answers "no", and gnulib
defines `__STDC_NO_VLA__`, which coreutils does not need.
