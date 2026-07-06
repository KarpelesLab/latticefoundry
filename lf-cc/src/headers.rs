//! Builtin freestanding C standard headers.
//!
//! LatticeFoundry's `lf-cc` targets a *freestanding* x86-64 Linux (LP64)
//! environment with no libc on disk, so the small set of headers a freestanding
//! translation unit is entitled to (`<stddef.h>`, `<stdint.h>`, `<stdbool.h>`,
//! `<limits.h>`, `<stdalign.h>`, `<iso646.h>`, `<stdnoreturn.h>`, `<float.h>`,
//! `<stdarg.h>`) is embedded here as source text and handed to the preprocessor
//! when an `#include` of one of these names is not satisfied on the `-I` search
//! path. See [`crate::preprocess`] for how the fallback is wired in.
//!
//! All values are written for the frozen target ABI (design tenet T1, from the
//! standard + the psABI): two's-complement, `char` = 1 byte and **signed**,
//! `short` = 2, `int` = 4, `long` = `long long` = pointer = 8. `size_t` /
//! `uintptr_t` are `unsigned long`; `ptrdiff_t` / `intptr_t` are `long`;
//! `wchar_t` is `int`. Each header is std-agnostic where it can be, and consults
//! the predefined `__STDC_VERSION__` macro where its contents must differ between
//! C23 (which promoted several library macros to keywords) and earlier revisions,
//! so the same embedded text is correct under every `--std`.

/// Look up a builtin freestanding header by its include name (e.g. `"stdint.h"`
/// or `"sys/types.h"` — only the freestanding set is recognized). Returns the
/// header's source text, or `None` if no builtin header has that name.
pub fn builtin_header(name: &str) -> Option<&'static str> {
    Some(match name {
        "stddef.h" => STDDEF_H,
        "stdint.h" => STDINT_H,
        "stdbool.h" => STDBOOL_H,
        "limits.h" => LIMITS_H,
        "stdalign.h" => STDALIGN_H,
        "iso646.h" => ISO646_H,
        "stdnoreturn.h" => STDNORETURN_H,
        "float.h" => FLOAT_H,
        "stdarg.h" => STDARG_H,
        _ => return None,
    })
}

/// `<stddef.h>`: `NULL`, `size_t`, `ptrdiff_t`, `wchar_t`, `max_align_t`,
/// `offsetof`.
const STDDEF_H: &str = r##"#ifndef _LF_STDDEF_H
#define _LF_STDDEF_H

#define NULL ((void*)0)

typedef unsigned long size_t;
typedef long ptrdiff_t;
typedef int wchar_t;

/* An object type whose alignment is the greatest fundamental alignment. The
   exact members are unspecified; only its alignment is normative. */
typedef struct { long long __lf_ll; double __lf_d; } max_align_t;

#define offsetof(t, m) ((size_t)&((t*)0)->m)

#endif /* _LF_STDDEF_H */
"##;

/// `<stdint.h>`: exact-/least-/fast-width integer typedefs, pointer/max types,
/// their limit macros, and the `INTn_C`/`UINTn_C` constant-suffix macros. LP64.
const STDINT_H: &str = r##"#ifndef _LF_STDINT_H
#define _LF_STDINT_H

/* Exact-width integer types. */
typedef signed char        int8_t;
typedef short              int16_t;
typedef int                int32_t;
typedef long               int64_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long      uint64_t;

/* Minimum-width integer types. */
typedef signed char        int_least8_t;
typedef short              int_least16_t;
typedef int                int_least32_t;
typedef long               int_least64_t;
typedef unsigned char      uint_least8_t;
typedef unsigned short     uint_least16_t;
typedef unsigned int       uint_least32_t;
typedef unsigned long      uint_least64_t;

/* Fastest minimum-width integer types (LP64: the wider ones are `long`). */
typedef signed char        int_fast8_t;
typedef long               int_fast16_t;
typedef long               int_fast32_t;
typedef long               int_fast64_t;
typedef unsigned char      uint_fast8_t;
typedef unsigned long      uint_fast16_t;
typedef unsigned long      uint_fast32_t;
typedef unsigned long      uint_fast64_t;

/* Integer types capable of holding object pointers. */
typedef long               intptr_t;
typedef unsigned long      uintptr_t;

/* Greatest-width integer types. */
typedef long               intmax_t;
typedef unsigned long      uintmax_t;

/* Limits of exact-width integer types. */
#define INT8_MIN   (-128)
#define INT16_MIN  (-32768)
#define INT32_MIN  (-2147483647 - 1)
#define INT64_MIN  (-9223372036854775807L - 1)
#define INT8_MAX   127
#define INT16_MAX  32767
#define INT32_MAX  2147483647
#define INT64_MAX  9223372036854775807L
#define UINT8_MAX  255
#define UINT16_MAX 65535
#define UINT32_MAX 4294967295U
#define UINT64_MAX 18446744073709551615UL

/* Limits of minimum-width integer types. */
#define INT_LEAST8_MIN   INT8_MIN
#define INT_LEAST16_MIN  INT16_MIN
#define INT_LEAST32_MIN  INT32_MIN
#define INT_LEAST64_MIN  INT64_MIN
#define INT_LEAST8_MAX   INT8_MAX
#define INT_LEAST16_MAX  INT16_MAX
#define INT_LEAST32_MAX  INT32_MAX
#define INT_LEAST64_MAX  INT64_MAX
#define UINT_LEAST8_MAX  UINT8_MAX
#define UINT_LEAST16_MAX UINT16_MAX
#define UINT_LEAST32_MAX UINT32_MAX
#define UINT_LEAST64_MAX UINT64_MAX

/* Limits of fastest minimum-width integer types. */
#define INT_FAST8_MIN   INT8_MIN
#define INT_FAST16_MIN  INT64_MIN
#define INT_FAST32_MIN  INT64_MIN
#define INT_FAST64_MIN  INT64_MIN
#define INT_FAST8_MAX   INT8_MAX
#define INT_FAST16_MAX  INT64_MAX
#define INT_FAST32_MAX  INT64_MAX
#define INT_FAST64_MAX  INT64_MAX
#define UINT_FAST8_MAX  UINT8_MAX
#define UINT_FAST16_MAX UINT64_MAX
#define UINT_FAST32_MAX UINT64_MAX
#define UINT_FAST64_MAX UINT64_MAX

/* Limits of pointer-holding and greatest-width integer types. */
#define INTPTR_MIN   (-9223372036854775807L - 1)
#define INTPTR_MAX   9223372036854775807L
#define UINTPTR_MAX  18446744073709551615UL
#define INTMAX_MIN   (-9223372036854775807L - 1)
#define INTMAX_MAX   9223372036854775807L
#define UINTMAX_MAX  18446744073709551615UL

/* Limits of other integer types defined in <stddef.h>/<wchar.h>. */
#define PTRDIFF_MIN  (-9223372036854775807L - 1)
#define PTRDIFF_MAX  9223372036854775807L
#define SIZE_MAX     18446744073709551615UL
#define SIG_ATOMIC_MIN (-2147483647 - 1)
#define SIG_ATOMIC_MAX 2147483647
#define WCHAR_MIN    (-2147483647 - 1)
#define WCHAR_MAX    2147483647
#define WINT_MIN     0U
#define WINT_MAX     4294967295U

/* Macros for integer constants of a given minimum-width type. */
#define INT8_C(c)    c
#define INT16_C(c)   c
#define INT32_C(c)   c
#define INT64_C(c)   c ## L
#define UINT8_C(c)   c
#define UINT16_C(c)  c
#define UINT32_C(c)  c ## U
#define UINT64_C(c)  c ## UL
#define INTMAX_C(c)  c ## L
#define UINTMAX_C(c) c ## UL

#endif /* _LF_STDINT_H */
"##;

/// `<stdbool.h>`: `bool`/`true`/`false` macros pre-C23; a near no-op under C23
/// where they are keywords (matching gcc). `__bool_true_false_are_defined`.
const STDBOOL_H: &str = r##"#ifndef _LF_STDBOOL_H
#define _LF_STDBOOL_H

#if !defined(__STDC_VERSION__) || __STDC_VERSION__ <= 201710L
/* Before C23, bool/true/false are library macros. */
#define bool  _Bool
#define true  1
#define false 0
#endif

#define __bool_true_false_are_defined 1

#endif /* _LF_STDBOOL_H */
"##;

/// `<limits.h>`: `CHAR_BIT`, and the width limits of the standard integer types.
/// `char` is signed on this target, so `CHAR_MIN`/`CHAR_MAX` == `SCHAR_*`.
const LIMITS_H: &str = r##"#ifndef _LF_LIMITS_H
#define _LF_LIMITS_H

#define CHAR_BIT   8
#define MB_LEN_MAX 16

#define SCHAR_MIN  (-128)
#define SCHAR_MAX  127
#define UCHAR_MAX  255

/* Plain char is signed on this target. */
#define CHAR_MIN   (-128)
#define CHAR_MAX   127

#define SHRT_MIN   (-32768)
#define SHRT_MAX   32767
#define USHRT_MAX  65535

#define INT_MIN    (-2147483647 - 1)
#define INT_MAX    2147483647
#define UINT_MAX   4294967295U

#define LONG_MIN   (-9223372036854775807L - 1)
#define LONG_MAX   9223372036854775807L
#define ULONG_MAX  18446744073709551615UL

#define LLONG_MIN  (-9223372036854775807LL - 1)
#define LLONG_MAX  9223372036854775807LL
#define ULLONG_MAX 18446744073709551615ULL

#endif /* _LF_LIMITS_H */
"##;

/// `<stdalign.h>`: `alignas`/`alignof` macros (→ `_Alignas`/`_Alignof`) pre-C23;
/// under C23 they are keywords so the header only defines the `*_is_defined`
/// probes, matching gcc.
const STDALIGN_H: &str = r##"#ifndef _LF_STDALIGN_H
#define _LF_STDALIGN_H

#if !defined(__STDC_VERSION__) || __STDC_VERSION__ <= 201710L
/* Before C23, alignas/alignof are library macros. */
#define alignas _Alignas
#define alignof _Alignof
#endif

#define __alignas_is_defined 1
#define __alignof_is_defined 1

#endif /* _LF_STDALIGN_H */
"##;

/// `<iso646.h>`: alternative spellings of the logical/bitwise operators.
const ISO646_H: &str = r##"#ifndef _LF_ISO646_H
#define _LF_ISO646_H

#define and    &&
#define and_eq &=
#define bitand &
#define bitor  |
#define compl  ~
#define not    !
#define not_eq !=
#define or     ||
#define or_eq  |=
#define xor    ^
#define xor_eq ^=

#endif /* _LF_ISO646_H */
"##;

/// `<stdnoreturn.h>`: `noreturn` macro (→ `_Noreturn`) pre-C23.
const STDNORETURN_H: &str = r##"#ifndef _LF_STDNORETURN_H
#define _LF_STDNORETURN_H

#if !defined(__STDC_VERSION__) || __STDC_VERSION__ <= 201710L
#define noreturn _Noreturn
#endif

#endif /* _LF_STDNORETURN_H */
"##;

/// `<float.h>`: characteristics of the IEEE-754 binary32 (`float`) and binary64
/// (`double`) types (and the x87 80-bit `long double`).
const FLOAT_H: &str = r##"#ifndef _LF_FLOAT_H
#define _LF_FLOAT_H

#define FLT_RADIX        2
#define FLT_ROUNDS       1
#define FLT_EVAL_METHOD  0
#define DECIMAL_DIG      21

#define FLT_MANT_DIG     24
#define DBL_MANT_DIG     53
#define LDBL_MANT_DIG    64

#define FLT_DIG          6
#define DBL_DIG          15
#define LDBL_DIG         18

#define FLT_MIN_EXP      (-125)
#define DBL_MIN_EXP      (-1021)
#define LDBL_MIN_EXP     (-16381)

#define FLT_MAX_EXP      128
#define DBL_MAX_EXP      1024
#define LDBL_MAX_EXP     16384

#define FLT_MIN_10_EXP   (-37)
#define DBL_MIN_10_EXP   (-307)
#define LDBL_MIN_10_EXP  (-4931)

#define FLT_MAX_10_EXP   38
#define DBL_MAX_10_EXP   308
#define LDBL_MAX_10_EXP  4932

#define FLT_MAX          3.40282346638528859811704183484516925e+38F
#define DBL_MAX          1.79769313486231570814527423731704357e+308
#define LDBL_MAX         1.18973149535723176502e+4932L

#define FLT_EPSILON      1.19209289550781250000000000000000000e-7F
#define DBL_EPSILON      2.22044604925031308084726333618164062e-16
#define LDBL_EPSILON     1.08420217248550443401e-19L

#define FLT_MIN          1.17549435082228750796873653722224568e-38F
#define DBL_MIN          2.22507385850720138309023271733240406e-308
#define LDBL_MIN         3.36210314311209350626e-4932L

#define FLT_TRUE_MIN     1.40129846432481707092372958328991613e-45F
#define DBL_TRUE_MIN     4.94065645841246544176568792868221372e-324
#define LDBL_TRUE_MIN    3.64519953188247460253e-4951L

#define FLT_HAS_SUBNORM  1
#define DBL_HAS_SUBNORM  1
#define LDBL_HAS_SUBNORM 1

#endif /* _LF_FLOAT_H */
"##;

/// `<stdarg.h>`: the System V AMD64 `va_list` and the `va_*` macros.
///
/// `va_list` is the psABI `__va_list_tag[1]` (a one-element array, so it decays
/// to a `__va_list_tag*` when passed to the builtins). The macros expand to the
/// compiler builtins the frontend recognizes and lowers against the register
/// save area / overflow area set up by a variadic function's prologue. Defined
/// under every `--std` (variadic functions predate C89).
const STDARG_H: &str = r##"#ifndef _LF_STDARG_H
#define _LF_STDARG_H

typedef struct __va_list_tag {
    unsigned gp_offset;
    unsigned fp_offset;
    void *overflow_arg_area;
    void *reg_save_area;
} __va_list_tag;

typedef __va_list_tag va_list[1];

#define va_start(ap, last) __builtin_va_start(ap, last)
#define va_arg(ap, type)   __builtin_va_arg(ap, type)
#define va_end(ap)         __builtin_va_end(ap)
#define va_copy(dst, src)  __builtin_va_copy(dst, src)

#endif /* _LF_STDARG_H */
"##;
