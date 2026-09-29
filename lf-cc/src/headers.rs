//! Builtin compiler-provided C standard headers.
//!
//! The headers a *freestanding* translation unit is entitled to (`<stddef.h>`,
//! `<stdint.h>`, `<stdbool.h>`, `<limits.h>`, `<stdalign.h>`, `<iso646.h>`,
//! `<stdnoreturn.h>`, `<float.h>`, `<stdarg.h>`) belong to the compiler, not to
//! the C library, so `lf-cc` embeds them here as source text. The preprocessor
//! places them on the header search chain after `-I`/`-isystem` and before the
//! host's system directories (see [`crate::preprocess`]), which is where a C
//! library expects "the compiler's headers" to be:
//!
//! * glibc obtains single items from `<stddef.h>`/`<stdarg.h>` through the
//!   `__need_*` protocol, which these headers honor;
//! * in a hosted translation unit `<limits.h>` and `<stdint.h>` layer over the
//!   C library's own headers of the same name with `#include_next`.
//!
//! With `-nostdinc` they are not consulted at all.
//!
//! All values are written for the frozen target ABI (design tenet T1, from the
//! standard + the psABI): two's-complement, `char` = 1 byte and **signed**,
//! `short` = 2, `int` = 4, `long` = `long long` = pointer = 8. `size_t` /
//! `uintptr_t` are `unsigned long`; `ptrdiff_t` / `intptr_t` are `long`;
//! `wchar_t` is `int`. Each header is std-agnostic where it can be, and consults
//! the predefined `__STDC_VERSION__` macro where its contents must differ between
//! C23 (which promoted several library macros to keywords) and earlier revisions,
//! so the same embedded text is correct under every `--std`.

/// Look up a builtin header by its include name (e.g. `"stdint.h"`; only the
/// freestanding set is recognized). Returns the header's source text, or
/// `None` if no builtin header has that name.
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

/// `<stddef.h>`: `NULL`, `size_t`, `ptrdiff_t`, `wchar_t`, `max_align_t` (C11),
/// `offsetof`.
///
/// It also honors the *selective-definition protocol* C library headers use to
/// obtain single items from it without the rest of the header's namespace: if
/// any `__need_X` macro is defined on entry (`__need_size_t`,
/// `__need_ptrdiff_t`, `__need_wchar_t`, `__need_wint_t`, `__need_NULL`,
/// `__need_max_align_t`, `__need_offsetof`), only the requested items are
/// provided and each `__need_X` is undefined again. `wint_t` is only ever
/// provided on request; the `_WINT_T` macro records that it has been (the
/// convention glibc's `<bits/types/wint_t.h>` checks). Every item has its own
/// guard, so any sequence of full and partial inclusions defines each once.
const STDDEF_H: &str = r##"/* lf-cc builtin <stddef.h> */
#if !defined __need_size_t && !defined __need_ptrdiff_t && !defined __need_wchar_t \
    && !defined __need_wint_t && !defined __need_NULL && !defined __need_max_align_t \
    && !defined __need_offsetof
/* The whole header. */
# define _LF_STDDEF_H 1
# define __need_size_t
# define __need_ptrdiff_t
# define __need_wchar_t
# define __need_NULL
# define __need_offsetof
# if defined __STDC_VERSION__ && __STDC_VERSION__ >= 201112L
#  define __need_max_align_t
# endif
#endif

#ifdef __need_size_t
# ifndef _LF_SIZE_T
#  define _LF_SIZE_T
typedef __SIZE_TYPE__ size_t;
# endif
# undef __need_size_t
#endif

#ifdef __need_ptrdiff_t
# ifndef _LF_PTRDIFF_T
#  define _LF_PTRDIFF_T
typedef __PTRDIFF_TYPE__ ptrdiff_t;
# endif
# undef __need_ptrdiff_t
#endif

#ifdef __need_wchar_t
# ifndef _LF_WCHAR_T
#  define _LF_WCHAR_T
typedef __WCHAR_TYPE__ wchar_t;
# endif
# undef __need_wchar_t
#endif

#ifdef __need_wint_t
# ifndef _WINT_T
#  define _WINT_T 1
typedef __WINT_TYPE__ wint_t;
# endif
# undef __need_wint_t
#endif

#ifdef __need_NULL
# undef NULL
# define NULL ((void *)0)
# undef __need_NULL
#endif

#ifdef __need_max_align_t
# ifndef _LF_MAX_ALIGN_T
#  define _LF_MAX_ALIGN_T
/* An object type whose alignment is the greatest fundamental alignment. The
   exact members are unspecified; only its alignment is normative. */
typedef struct { long long __lf_ll; double __lf_d; } max_align_t;
# endif
# undef __need_max_align_t
#endif

#ifdef __need_offsetof
# undef offsetof
# define offsetof(t, m) __builtin_offsetof(t, m)
# undef __need_offsetof
#endif
"##;

/// `<stdint.h>`: exact-/least-/fast-width integer typedefs, pointer/max types,
/// their limit macros, and the `INTn_C`/`UINTn_C` constant-suffix macros. LP64.
///
/// In a hosted translation unit whose C library has its own `<stdint.h>`,
/// that header is used instead (`#include_next`): the library's other headers
/// define the same typedefs under its own guard macros, so its `<stdint.h>` is
/// the one that composes with them.
const STDINT_H: &str = r##"/* lf-cc builtin <stdint.h> */
#if __STDC_HOSTED__ && defined __has_include_next && __has_include_next(<stdint.h>)
# include_next <stdint.h>
#elif !defined _LF_STDINT_H
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
///
/// In a hosted translation unit the C library's `<limits.h>` (POSIX limits such
/// as `PATH_MAX`) is layered on top with `#include_next`. glibc's header chains
/// back to "the compiler's `<limits.h>`" unless `_GCC_LIMITS_H_` is defined —
/// the macro glibc documents as the compiler header's marker — so this header
/// defines it to say its limits are already in place.
const LIMITS_H: &str = r##"/* lf-cc builtin <limits.h> */
#ifndef _LF_LIMITS_H
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

#define _GCC_LIMITS_H_ 1

#if __STDC_HOSTED__ && defined __has_include_next && __has_include_next(<limits.h>)
# include_next <limits.h>
#endif

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

/// `<float.h>`: characteristics of the floating types, written in terms of the
/// predefined `__FLT_*__`/`__DBL_*__`/`__LDBL_*__` macros so there is one
/// source of truth: IEEE-754 binary32 `float`, binary64 `double`, and
/// `long double` as lf-cc implements it (currently binary64 as well).
const FLOAT_H: &str = r##"/* lf-cc builtin <float.h> */
#ifndef _LF_FLOAT_H
#define _LF_FLOAT_H

#define FLT_RADIX        __FLT_RADIX__
#define FLT_ROUNDS       1
#define FLT_EVAL_METHOD  __FLT_EVAL_METHOD__
#define DECIMAL_DIG      __DECIMAL_DIG__

#define FLT_MANT_DIG     __FLT_MANT_DIG__
#define DBL_MANT_DIG     __DBL_MANT_DIG__
#define LDBL_MANT_DIG    __LDBL_MANT_DIG__

#define FLT_DIG          __FLT_DIG__
#define DBL_DIG          __DBL_DIG__
#define LDBL_DIG         __LDBL_DIG__

#define FLT_DECIMAL_DIG  __FLT_DECIMAL_DIG__
#define DBL_DECIMAL_DIG  __DBL_DECIMAL_DIG__
#define LDBL_DECIMAL_DIG __LDBL_DECIMAL_DIG__

#define FLT_MIN_EXP      __FLT_MIN_EXP__
#define DBL_MIN_EXP      __DBL_MIN_EXP__
#define LDBL_MIN_EXP     __LDBL_MIN_EXP__

#define FLT_MAX_EXP      __FLT_MAX_EXP__
#define DBL_MAX_EXP      __DBL_MAX_EXP__
#define LDBL_MAX_EXP     __LDBL_MAX_EXP__

#define FLT_MIN_10_EXP   __FLT_MIN_10_EXP__
#define DBL_MIN_10_EXP   __DBL_MIN_10_EXP__
#define LDBL_MIN_10_EXP  __LDBL_MIN_10_EXP__

#define FLT_MAX_10_EXP   __FLT_MAX_10_EXP__
#define DBL_MAX_10_EXP   __DBL_MAX_10_EXP__
#define LDBL_MAX_10_EXP  __LDBL_MAX_10_EXP__

#define FLT_MAX          __FLT_MAX__
#define DBL_MAX          __DBL_MAX__
#define LDBL_MAX         __LDBL_MAX__

#define FLT_EPSILON      __FLT_EPSILON__
#define DBL_EPSILON      __DBL_EPSILON__
#define LDBL_EPSILON     __LDBL_EPSILON__

#define FLT_MIN          __FLT_MIN__
#define DBL_MIN          __DBL_MIN__
#define LDBL_MIN         __LDBL_MIN__

#define FLT_TRUE_MIN     __FLT_DENORM_MIN__
#define DBL_TRUE_MIN     __DBL_DENORM_MIN__
#define LDBL_TRUE_MIN    __LDBL_DENORM_MIN__

#define FLT_HAS_SUBNORM  __FLT_HAS_DENORM__
#define DBL_HAS_SUBNORM  __DBL_HAS_DENORM__
#define LDBL_HAS_SUBNORM __LDBL_HAS_DENORM__

#endif /* _LF_FLOAT_H */
"##;

/// `<stdarg.h>`: the System V AMD64 `va_list` and the `va_*` macros.
///
/// `va_list` is the psABI `__va_list_tag[1]` (a one-element array, so it decays
/// to a `__va_list_tag*` when passed to the builtins). The macros expand to the
/// compiler builtins the frontend recognizes and lowers against the register
/// save area / overflow area set up by a variadic function's prologue. Defined
/// under every `--std` (variadic functions predate C89).
///
/// The underlying type is also exported as `__gnuc_va_list`, and a C library
/// header that only needs that name (glibc's `<stdio.h>`, `<wchar.h>`, …)
/// requests it alone with `#define __need___va_list` before the include: then
/// nothing else is defined and `__need___va_list` is undefined again. `va_list`
/// itself is guarded by `_VA_LIST_DEFINED`, the macro glibc's headers use when
/// they declare `va_list` from `__gnuc_va_list` themselves.
const STDARG_H: &str = r##"/* lf-cc builtin <stdarg.h> */
#ifndef _LF_GNUC_VA_LIST
#define _LF_GNUC_VA_LIST
#if __has_builtin(__builtin_va_list)
typedef __builtin_va_list __gnuc_va_list;
#else
typedef struct __va_list_tag {
    unsigned gp_offset;
    unsigned fp_offset;
    void *overflow_arg_area;
    void *reg_save_area;
} __va_list_tag;
typedef __va_list_tag __gnuc_va_list[1];
#endif
#endif /* _LF_GNUC_VA_LIST */

#ifdef __need___va_list
# undef __need___va_list
#elif !defined _LF_STDARG_H
#define _LF_STDARG_H

#ifndef _VA_LIST_DEFINED
#define _VA_LIST_DEFINED
typedef __gnuc_va_list va_list;
#endif

#define va_start(ap, last) __builtin_va_start(ap, last)
#define va_arg(ap, type)   __builtin_va_arg(ap, type)
#define va_end(ap)         __builtin_va_end(ap)
#define va_copy(dst, src)  __builtin_va_copy(dst, src)
#define __va_copy(dst, src) __builtin_va_copy(dst, src)

#endif /* _LF_STDARG_H */
"##;
