//! GNU inline asm with operands (`docs/ir-design.md` §6i), end to end.
//!
//! Each program uses the inline-asm idioms real glibc/Linux code does
//! (`rdtsc` into a pair of outputs, `cpuid` with four, `xchg`/`lock xadd`
//! atomics, `bswap`, `rep movsb`, musl-style syscall wrappers with
//! register-asm variables, named operands and operand modifiers, labels in
//! templates, `<sys/io.h>`). It is compiled by lf-cc at `-O0` and `-O2`,
//! linked against the host C library with qld, run, and its stdout and exit
//! status compared with the same program built by the host gcc. Tests skip
//! when no host C runtime (or no gcc) is found.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::link::gnu::{HostCrt, host_c_link_args, link_gnu};
use latticefoundry::transform::pipeline::OptLevel;
use lf_cc::{BuildError, CStd, PpOptions};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-cc-inline-asm-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn which(prog: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(prog)).find(|c| c.is_file())
}

/// Options for a hosted gnu17 build against the host's real headers.
fn hosted() -> PpOptions {
    PpOptions {
        std: CStd::parse("gnu17").expect("known std"),
        stdinc_dirs: lf_cc::default_system_include_dirs(),
        hosted: true,
        ..PpOptions::default()
    }
}

/// Run `exe`, returning `(exit code, stdout)`.
fn run(exe: &Path) -> (i32, String) {
    let out = Command::new(exe).output().expect("run executable");
    let code = out.status.code().unwrap_or_else(|| panic!("{} died: {}", exe.display(), out.status));
    (code, String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Build `src` with lf-cc at `opt` and link it with qld against the host C
/// library; return the executable.
fn lf_build(crt: &HostCrt, dir: &Path, tag: &str, src: &str, opt: OptLevel) -> PathBuf {
    let input = format!("{tag}.c");
    let out = lf_cc::compile_object_with(src, &input, &hosted(), opt, false)
        .unwrap_or_else(|e| panic!("lf-cc failed on '{tag}': {e:?}"));
    let obj = dir.join(format!("{tag}.o"));
    std::fs::write(&obj, &out.object).unwrap();
    let exe = dir.join(tag);
    let args = host_c_link_args(crt, &[obj.as_path()], &[], &exe);
    link_gnu("inline-asm-test", &args).unwrap_or_else(|e| panic!("link of '{tag}' failed: {e}"));
    exe
}

/// Compile `src` with lf-cc (`-O0`, `-O2`) and, when present, gcc; run all
/// and require the same exit status and stdout everywhere. Returns lf-cc's.
fn differential(name: &str, src: &str) -> Option<(i32, String)> {
    let Some(crt) = HostCrt::discover() else {
        eprintln!("skipping {name}: no host C runtime");
        return None;
    };
    let dir = scratch(name);
    let o0 = run(&lf_build(&crt, &dir, &format!("{name}_O0"), src, OptLevel::O0));
    let o2 = run(&lf_build(&crt, &dir, &format!("{name}_O2"), src, OptLevel::O2));
    assert_eq!(o0, o2, "{name}: lf-cc -O0 vs -O2");
    if let Some(gcc) = which("gcc") {
        let c = dir.join(format!("{name}.c"));
        let exe = dir.join(format!("{name}_gcc"));
        std::fs::write(&c, src).unwrap();
        let std = common::gcc_std_flag(&gcc, "gnu17");
        let status = Command::new(&gcc).args([std.as_str(), "-O0", "-o"]).arg(&exe).arg(&c).status().unwrap();
        assert!(status.success(), "gcc failed on {name}");
        assert_eq!(o0, run(&exe), "{name}: lf-cc vs gcc");
    } else {
        eprintln!("{name}: gcc not installed, comparison skipped");
    }
    Some(o0)
}

#[test]
fn rdtsc_and_cpuid() {
    let src = r#"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static inline uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
    __asm__ __volatile__("cpuid"
                         : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3])
                         : "0"(leaf), "2"(sub));
}

int main(void) {
    uint64_t t1 = rdtsc(), t2 = rdtsc();
    unsigned r[4];
    char vendor[13];
    cpuid(0, 0, r);
    memcpy(vendor, &r[1], 4);
    memcpy(vendor + 4, &r[3], 4);
    memcpy(vendor + 8, &r[2], 4);
    vendor[12] = 0;
    unsigned max = r[0];
    cpuid(1, 0, r);
    /* Family/model/stepping and the SSE2 bit are stable on one machine. */
    printf("%s max>=1:%d sig:%08x sse2:%u ordered:%d\n", vendor, max >= 1, r[0] & 0x0fff3fff,
           (r[3] >> 26) & 1, t2 >= t1 && t1 != 0);
    return 0;
}
"#;
    if let Some((code, out)) = differential("rdtsc_cpuid", src) {
        assert_eq!(code, 0);
        assert!(out.contains("max>=1:1") && out.contains("sse2:1") && out.contains("ordered:1"), "{out}");
    }
}

#[test]
fn atomics_bswap_and_bit_scans() {
    let src = r#"
#include <stdio.h>

static inline int xchg(int *p, int v) {
    __asm__ __volatile__("xchgl %0, %1" : "=r"(v), "+m"(*p) : "0"(v) : "memory");
    return v;
}

static inline int fetch_add(int *p, int v) {
    __asm__ __volatile__("lock; xaddl %0, %1" : "+r"(v), "+m"(*p) : : "memory", "cc");
    return v;
}

static inline int cas(long *p, long old, long new_) {
    long prev;
    __asm__ __volatile__("lock; cmpxchgq %2, %1"
                         : "=a"(prev), "+m"(*p)
                         : "r"(new_), "0"(old)
                         : "memory", "cc");
    return prev == old;
}

static inline unsigned bswap32(unsigned x) {
    __asm__("bswap %0" : "=r"(x) : "0"(x));
    return x;
}

static inline unsigned long long bswap64(unsigned long long x) {
    __asm__("bswapq %0" : "+r"(x));
    return x;
}

static inline int bsr(unsigned x) {
    int r;
    __asm__("bsrl %1, %0" : "=r"(r) : "rm"(x) : "cc");
    return r;
}

int main(void) {
    int a = 5, b = 0;
    long c = 7;
    int old = xchg(&a, 9);
    int prev = fetch_add(&b, 3);
    prev += fetch_add(&b, 4);
    int ok1 = cas(&c, 7, 11), ok2 = cas(&c, 7, 13);
    printf("%d %d %d %d %ld %d %d\n", old, a, prev, b, c, ok1, ok2);
    printf("%08x %016llx %d %d\n", bswap32(0x11223344u), bswap64(0x0102030405060708ull), bsr(1), bsr(0x80000u));
    __asm__ __volatile__("" ::: "memory");
    return a + b - 16;
}
"#;
    if let Some((code, out)) = differential("atomics", src) {
        assert_eq!(code, 0);
        assert_eq!(out, "5 9 3 7 11 1 0\n44332211 0807060504030201 0 19\n");
    }
}

#[test]
fn string_ops_named_operands_and_modifiers() {
    let src = r#"
#include <stdio.h>
#include <stddef.h>

static void *copy(void *dst, const void *src, size_t n) {
    void *ret = dst;
    __asm__ __volatile__("rep movsb" : "+D"(dst), "+S"(src), "+c"(n) : : "memory");
    return ret;
}

static void fill(void *dst, int c, size_t n) {
    __asm__ __volatile__("rep stosb" : "+D"(dst), "+c"(n) : "a"(c) : "memory");
}

struct pair { long lo, hi; };

int main(void) {
    char buf[16] = {0};
    fill(buf, 'x', 5);
    copy(buf + 5, "-asm-", 6);
    int lo, word;
    unsigned v = 0x1234abcd;
    /* `%b`, `%w`, `%k`, `%h` and named operands. */
    __asm__("movzbl %b[in], %k[out]" : [out] "=r"(lo) : [in] "r"(v));
    __asm__("movzwl %w1, %0" : "=r"(word) : "r"(v));
    unsigned char high;
    __asm__("movb %h1, %0" : "=Q"(high) : "Q"(v));
    /* Memory operands, `%H` (the next eightbyte) and immediates. */
    struct pair p = {40, 2};
    long sum;
    __asm__("movq %1, %0; addq %H1, %0" : "=&r"(sum) : "m"(p));
    long k;
    __asm__("movq %1, %0; addq %2, %0" : "=r"(k) : "i"(sizeof(struct pair)), "n"(26));
    /* An early-clobber output written before the inputs are read. */
    long e;
    __asm__("movq $100, %0; addq %1, %0; addq %2, %0" : "=&r"(e) : "r"(sum), "r"(k));
    printf("%s %x %x %x %ld %ld %ld\n", buf, lo, word, high, sum, k, e);
    return 0;
}
"#;
    if let Some((code, out)) = differential("strings", src) {
        assert_eq!(code, 0);
        assert_eq!(out, "xxxxx-asm- cd abcd ab 42 42 184\n");
    }
}

#[test]
fn labels_xmm_and_clobbers() {
    let src = r#"
#include <stdio.h>

static long count_bits(unsigned long x) {
    long n;
    __asm__("xorl %k0, %k0\n"
            "1:\n\t"
            "testq %1, %1\n\t"
            "jz 2f\n\t"
            "leaq -1(%1), %%rdx\n\t"
            "andq %%rdx, %1\n\t"
            "incq %0\n\t"
            "jmp 1b\n"
            "2:"
            : "=&r"(n), "+r"(x)
            :
            : "rdx", "cc");
    return n;
}

static int sign(long v) {
    int r;
    __asm__("testq %1, %1\n\t"
            "js .Lneg%=\n\t"
            "movl $1, %0\n\t"
            "jmp .Lend%=\n"
            ".Lneg%=:\n\t"
            "movl $-1, %0\n"
            ".Lend%=:"
            : "=r"(r) : "r"(v) : "cc");
    return r;
}

static double hyp(double a, double b) {
    double r;
    __asm__("mulsd %0, %0\n\tmulsd %1, %1\n\taddsd %1, %0\n\tsqrtsd %0, %0" : "=x"(r), "+x"(b) : "0"(a));
    return r;
}

int main(void) {
    long a = 3, b = 4, c = 5, d = 6, e = 7, f = 8, g = 9, h = 10;
    /* Every value stays live across an asm that clobbers nearly every register. */
    __asm__ __volatile__("xorl %%eax, %%eax; xorl %%ebx, %%ebx; xorl %%ecx, %%ecx; xorl %%edx, %%edx;"
                         "xorl %%esi, %%esi; xorl %%edi, %%edi; xorl %%r8d, %%r8d; xorl %%r9d, %%r9d;"
                         "xorl %%r10d, %%r10d; xorl %%r11d, %%r11d; xorl %%r12d, %%r12d; xorl %%r13d, %%r13d;"
                         "xorl %%r14d, %%r14d; xorl %%r15d, %%r15d"
                         ::: "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11",
                             "r12", "r13", "r14", "r15", "cc", "memory");
    __asm__ volatile("pause");
    asm("nop");
    printf("%ld %ld %d %d %.1f\n", count_bits(0xF0F0ul), a + b + c + d + e + f + g + h, sign(-5), sign(5),
           hyp(3.0, 4.0));
    return 0;
}
"#;
    if let Some((code, out)) = differential("labels", src) {
        assert_eq!(code, 0);
        assert_eq!(out, "8 52 -1 1 5.0\n");
    }
}

#[test]
fn musl_style_syscall_wrappers() {
    // musl's x86_64 `syscall_arch.h`, with register-asm variables for the
    // fourth to sixth arguments.
    let src = r#"
#include <stdio.h>
#include <unistd.h>

static inline long __syscall0(long n) {
    unsigned long ret;
    __asm__ __volatile__ ("syscall" : "=a"(ret) : "a"(n) : "rcx", "r11", "memory");
    return ret;
}

static inline long __syscall3(long n, long a1, long a2, long a3) {
    unsigned long ret;
    __asm__ __volatile__ ("syscall" : "=a"(ret) : "a"(n), "D"(a1), "S"(a2),
                          "d"(a3) : "rcx", "r11", "memory");
    return ret;
}

static inline long __syscall6(long n, long a1, long a2, long a3, long a4, long a5, long a6) {
    unsigned long ret;
    register long r10 __asm__("r10") = a4;
    register long r8 __asm__("r8") = a5;
    register long r9 __asm__("r9") = a6;
    __asm__ __volatile__ ("syscall" : "=a"(ret) : "a"(n), "D"(a1), "S"(a2),
                          "d"(a3), "r"(r10), "r"(r8), "r"(r9) : "rcx", "r11", "memory");
    return ret;
}

int main(void) {
    long pid = __syscall0(39);
    fflush(stdout);
    long w = __syscall3(1, 1, (long)"hello from syscall\n", 19);
    /* mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) */
    long m = __syscall6(9, 0, 4096, 3, 0x22, -1, 0);
    int mapped = m > 0 && m % 4096 == 0;
    if (mapped) {
        ((char *)m)[100] = 42;
        mapped = ((char *)m)[100] == 42 && __syscall3(11, m, 4096, 0) == 0;
    }
    printf("pid:%d wrote:%ld mapped:%d\n", pid == getpid(), w, mapped);
    return 0;
}
"#;
    if let Some((code, out)) = differential("syscalls", src) {
        assert_eq!(code, 0);
        assert_eq!(out, "hello from syscall\npid:1 wrote:19 mapped:1\n");
    }
}

#[test]
fn sys_io_h_compiles() {
    // glibc's real `<sys/io.h>` port I/O helpers (`inb`/`outb`/`insb`, whose
    // `"Nd"` constraint takes a constant port or `dx`). Port I/O needs
    // privileges, so the program only runs the paths that do not touch a port.
    if !Path::new("/usr/include/x86_64-linux-gnu/sys/io.h").is_file() && !Path::new("/usr/include/sys/io.h").is_file() {
        eprintln!("skipping: no <sys/io.h>");
        return;
    }
    let src = r#"
#include <sys/io.h>
#include <stdio.h>

unsigned char read_port(unsigned short port) { return inb(port); }
unsigned char read_fixed(void) { return inb_p(0x80); }
void write_port(unsigned short port, unsigned char v) { outb(v, port); outw(v, port); outl(v, port); }
void read_many(unsigned short port, void *buf, unsigned long n) { insb(port, buf, n); insl(port, buf, n); }
void write_many(unsigned short port, const void *buf, unsigned long n) { outsw(port, buf, n); }

int main(int argc, char **argv) {
    (void)argv;
    if (argc > 5) {
        write_port(0x80, 1);
        static unsigned char b[4];
        read_many(0x80, b, 1);
        write_many(0x80, b, 1);
        return read_port(0x80) + read_fixed();
    }
    printf("io ok\n");
    return 0;
}
"#;
    if let Some((code, out)) = differential("sys_io", src) {
        assert_eq!((code, out.as_str()), (0, "io ok\n"));
    }
}

#[test]
fn errors_are_clear() {
    let opts = PpOptions { std: CStd::parse("gnu17").unwrap(), ..PpOptions::default() };
    // `asm goto` is rejected by the front end.
    let src = "int main(void){ int a = 1; asm goto (\"jmp %l[done]\" : : \"r\"(a) : \"memory\" : done); done: return 0; }";
    match lf_cc::check_source_with(src, &opts) {
        Err(d) => assert!(d[0].message.contains("`asm goto` is not supported"), "{d:?}"),
        Ok(_) => panic!("asm goto accepted"),
    }
    // Operand shape errors are front-end errors.
    for (body, want) in [
        ("int r; asm(\"\" : \"r\"(r));", "must start with '=' or '+'"),
        ("asm(\"\" : \"=r\"(1));", "not an lvalue"),
        ("int x = 1; asm(\"\" : : \"i\"(x));", "must be a constant"),
        ("struct s { long a, b; } v; asm(\"\" : : \"r\"(v));", "needs a memory constraint"),
    ] {
        let src = format!("int main(void){{ {body} return 0; }}");
        match lf_cc::check_source_with(&src, &opts) {
            Err(d) => assert!(d.iter().any(|d| d.message.contains(want)), "{body}: {d:?}"),
            Ok(_) => panic!("{body}: accepted"),
        }
    }
    // Constraint, clobber and template errors come from the backend check,
    // naming the function.
    for (body, want) in [
        ("int r; asm(\"frobnicate %0\" : \"=r\"(r)); return r;", "function `main`: inline asm:"),
        ("int r; asm(\"\" : \"=r\"(r) : : \"bogus\"); return r;", "unknown register `bogus`"),
        ("long r; asm(\"\" : \"=t\"(r)); return r;", "x87"),
    ] {
        let src = format!("int main(void){{ {body} }}");
        match lf_cc::compile_object_with(&src, "e.c", &opts, OptLevel::O0, false) {
            Err(BuildError::Backend(m)) => assert!(m.contains(want), "{body}: {m}"),
            other => panic!("{body}: expected a backend error, got {:?}", other.map(|_| ())),
        }
    }
}
