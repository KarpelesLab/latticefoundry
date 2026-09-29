//! M9: programs that `#include` the real system (glibc) headers, compiled by
//! lf-cc, linked against the host libc, run, and checked against the expected
//! stdout/stderr/exit status — and against the same program built by gcc when
//! gcc is installed.
//!
//! Two front-end paths are exercised:
//!
//! - lf-cc as a user runs it: its own preprocessor, its builtin freestanding
//!   headers, and the default `/usr/include` search — at -O0, and at -O2
//!   (which predefines `__OPTIMIZE__`, turning on glibc's `extern __inline`
//!   definitions and <ctype.h>'s statement-expression macros);
//! - when gcc is available, `gcc -E` as a preprocessing *oracle*, so the
//!   parser/sema/lowering are tested on exactly the text gcc itself compiles
//!   (also in the `-O2 -D_GNU_SOURCE` configuration, which turns on glibc's
//!   `extern __inline` fast paths).
//!
//! Every test skips (with a message) when there is no `/usr/include/stdio.h` or
//! no host C runtime.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use latticefoundry::link::gnu::HostCrt;

const LF_CC: &str = env!("CARGO_BIN_EXE_lf-cc");

/// One program: its source, and the expected stdout, stderr, and exit status.
struct Prog {
    name: &'static str,
    src: &'static str,
    stdout: &'static str,
    stderr: &'static str,
    exit: i32,
}

fn programs() -> Vec<Prog> {
    vec![
        Prog {
            name: "stdio_formatting",
            src: r#"
#include <stdio.h>
#include <stdarg.h>

static int fmt(char *buf, size_t n, const char *f, ...) {
    va_list ap;
    va_start(ap, f);
    int r = vsnprintf(buf, n, f, ap);
    va_end(ap);
    return r;
}

static void logmsg(FILE *to, const char *f, ...) {
    va_list ap, aq;
    va_start(ap, f);
    va_copy(aq, ap);
    vfprintf(to, f, ap);
    va_end(ap);
    char tmp[64];
    vsnprintf(tmp, sizeof tmp, f, aq);
    va_end(aq);
    printf("[%s]\n", tmp);
}

int main(void) {
    char buf[96];
    int r = fmt(buf, sizeof buf, "%s-%d-%5.2f-%x-%lld", "abc", 42, 3.14159, 255u, -5LL);
    printf("%d %s\n", r, buf);
    logmsg(stdout, "%d+%d=%ld %s|", 2, 3, 5L, "ok");
    snprintf(buf, sizeof buf, "%-6s|%6.2f|%+d|%c|%%|%e", "hi", 2.0 / 3, 5, 'z', 12345.678);
    puts(buf);
    int a; double d; char w[16];
    int n = sscanf("12 abc 3.5", "%d %15s %lf", &a, w, &d);
    printf("sscanf=%d %d %s %.2f\n", n, a, w, d);
    fprintf(stderr, "to stderr %d\n", 7);
    fputs("fputs\n", stdout);
    putchar('!');
    putchar('\n');
    return 3;
}
"#,
            stdout: "18 abc-42- 3.14-ff--5\n\
                     2+3=5 ok|[2+3=5 ok|]\n\
                     hi    |  0.67|+5|z|%|1.234568e+04\n\
                     sscanf=3 12 abc 3.50\n\
                     fputs\n\
                     !\n",
            stderr: "to stderr 7\n",
            exit: 3,
        },
        Prog {
            name: "stdlib_services",
            src: r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct rec { char name[16]; int score; };

static int by_score(const void *a, const void *b) {
    const struct rec *x = a, *y = b;
    return x->score - y->score;
}

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

int main(void) {
    int v[] = {5, 3, 9, 1, 7, -2};
    qsort(v, sizeof v / sizeof v[0], sizeof v[0], cmp_int);
    for (size_t i = 0; i < sizeof v / sizeof v[0]; i++) printf("%d ", v[i]);
    printf("\n");
    struct rec rs[] = {{"carol", 70}, {"alice", 90}, {"bob", 80}};
    qsort(rs, 3, sizeof rs[0], by_score);
    for (int i = 0; i < 3; i++) printf("%s:%d ", rs[i].name, rs[i].score);
    printf("\n");
    int key = 7;
    int *hit = bsearch(&key, v, 6, sizeof(int), cmp_int);
    printf("bsearch=%ld\n", hit ? (long)(hit - v) : -1L);
    char *end;
    long l = strtol("  -1234xyz", &end, 10);
    printf("%ld [%s] %lu %d %.1f %lld\n", l, end, strtoul("ff", NULL, 16), atoi("  314"),
           strtod("2.5e3", NULL), strtoll("-9000000000", NULL, 10));
    printf("abs=%d labs=%ld\n", abs(-17), labs(-123456789L));
    const char *unset = getenv("LF_CC_M9_SURELY_UNSET_VARIABLE");
    printf("getenv=%s\n", unset ? unset : "(null)");
    char *m = malloc(8);
    strcpy(m, "Hello");
    m = realloc(m, 64);
    strcat(m, ", world");
    printf("%s %zu\n", m, strlen(m));
    free(m);
    int *z = calloc(4, sizeof(int));
    printf("calloc=%d%d%d%d\n", z[0], z[1], z[2], z[3]);
    free(z);
    exit(EXIT_SUCCESS);
}
"#,
            stdout: "-2 1 3 5 7 9 \n\
                     carol:70 bob:80 alice:90 \n\
                     bsearch=4\n\
                     -1234 [xyz] 255 314 2500.0 -9000000000\n\
                     abs=17 labs=123456789\n\
                     getenv=(null)\n\
                     Hello, world 12\n\
                     calloc=0000\n",
            stderr: "",
            exit: 0,
        },
        Prog {
            name: "string_and_ctype",
            src: r#"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

int main(void) {
    char buf[64];
    memset(buf, 0, sizeof buf);
    memcpy(buf, "abc", 3);
    strncat(buf, "defgh", 3);
    printf("%s %zu %d %d\n", buf, strlen(buf), strcmp("abc", "abd") < 0, strncmp("abcd", "abcf", 3));
    printf("%s %s %s\n", strchr("hello", 'l'), strrchr("hello", 'l'), strstr("haystack", "st"));
    printf("memcmp=%d memchr=%s\n", memcmp("ab", "ac", 2) < 0, (char *)memchr("xyz", 'y', 3));
    char tok[] = "a,b,,c";
    for (char *t = strtok(tok, ","); t; t = strtok(NULL, ",")) printf("<%s>", t);
    printf("\n");
    char *dup = strdup("dup");
    printf("%s %zu %zu\n", dup, strspn("aaab", "a"), strcspn("abc", "c"));
    free(dup);
    int alpha = 0, digit = 0, up = 0, space = 0;
    for (const char *p = "Hello World 123\t"; *p; p++) {
        unsigned char c = (unsigned char)*p;
        alpha += isalpha(c) != 0;
        digit += isdigit(c) != 0;
        up += isupper(c) != 0;
        space += isspace(c) != 0;
    }
    char s[] = "MiXeD";
    for (char *p = s; *p; p++) *p = (char)(isupper((unsigned char)*p) ? tolower((unsigned char)*p) : toupper((unsigned char)*p));
    printf("alpha=%d digit=%d upper=%d space=%d %s %d\n", alpha, digit, up, space, s, isxdigit('f') != 0);
    return 0;
}
"#,
            stdout: "abcdef 6 1 0\n\
                     llo lo stack\n\
                     memcmp=1 memchr=yz\n\
                     <a><b><c>\n\
                     dup 3 2\n\
                     alpha=10 digit=3 upper=2 space=3 mIxEd 1\n",
            stderr: "",
            exit: 0,
        },
        Prog {
            name: "errno_and_math",
            src: r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

int main(void) {
    errno = 0;
    strtol("99999999999999999999999", NULL, 10);
    printf("erange=%d\n", errno == ERANGE);
    errno = 0;
    FILE *f = fopen("/nonexistent-lf-cc-m9/file", "r");
    printf("fopen=%d enoent=%d %s\n", f == NULL, errno == ENOENT, strerror(ENOENT));
    printf("sqrt=%.6f pow=%.1f fabs=%.2f floor=%.1f ceil=%.1f fmod=%.1f\n",
           sqrt(2.0), pow(2.0, 10.0), fabs(-3.5), floor(2.7), ceil(2.1), fmod(7.5, 2.0));
    printf("sin=%.4f exp=%.4f log=%.4f atan2=%.4f sqrtf=%.3f\n",
           sin(1.0), exp(1.0), log(10.0), atan2(1.0, 1.0), (double)sqrtf(9.0f));
    double z = 0.0;
    printf("isnan=%d isinf=%d isfinite=%d isnormal=%d signbit=%d\n", isnan(z / z) != 0,
           isinf(1.0 / z) != 0, isfinite(1.0) != 0, isnormal(1e-310) != 0, signbit(-2.0) != 0);
    printf("huge=%d nan=%d inf=%d fpclassify=%d\n", HUGE_VAL > 1e308, isnan(NAN) != 0,
           INFINITY > 1e308, fpclassify(0.0) == FP_ZERO);
    printf("isgreater=%d isunordered=%d\n", isgreater(2.0, 1.0) != 0, isunordered(NAN, 1.0) != 0);
    return 0;
}
"#,
            stdout: "erange=1\n\
                     fopen=1 enoent=1 No such file or directory\n\
                     sqrt=1.414214 pow=1024.0 fabs=3.50 floor=2.0 ceil=3.0 fmod=1.5\n\
                     sin=0.8415 exp=2.7183 log=2.3026 atan2=0.7854 sqrtf=3.000\n\
                     isnan=1 isinf=1 isfinite=1 isnormal=0 signbit=1\n\
                     huge=1 nan=1 inf=1 fpclassify=1\n\
                     isgreater=1 isunordered=1\n",
            stderr: "",
            exit: 0,
        },
        Prog {
            name: "posix_services",
            src: r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <limits.h>
#include <setjmp.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <arpa/inet.h>
#include <assert.h>

static jmp_buf jb;
static volatile sig_atomic_t got;
static void on_sig(int s) { got = s; }
static void jump(int v) { longjmp(jb, v); }
struct rec { char name[16]; int score; };

int main(void) {
    printf("offsetof=%zu sizeof=%zu\n", offsetof(struct rec, score), sizeof(struct rec));
    int64_t big = INT64_C(1) << 40;
    uint32_t u = UINT32_MAX;
    printf("%" PRId64 " %" PRIu32 " %" PRIx32 " %d\n", big, u, u, INT_MAX);
    printf("htonl=%08x htons=%04x ntohl=%08x\n", htonl(0x11223344u), htons(0xabcd),
           ntohl(htonl(0xdeadbeefu)));
    bool flag = true;
    assert(flag);
    int r = setjmp(jb);
    if (r == 0) jump(42);
    printf("longjmp=%d\n", r);
    signal(SIGUSR1, on_sig);
    raise(SIGUSR1);
    printf("signal=%d\n", got == SIGUSR1);
    struct tm tm;
    memset(&tm, 0, sizeof tm);
    tm.tm_year = 100; tm.tm_mon = 1; tm.tm_mday = 29; tm.tm_hour = 12; tm.tm_min = 34;
    char buf[64];
    strftime(buf, sizeof buf, "%Y-%m-%d %H:%M", &tm);
    printf("strftime=%s\n", buf);
    int fds[2];
    if (pipe(fds) == 0) {
        if (write(fds[1], "pipe!", 5) != 5) return 9;
        close(fds[1]);
        ssize_t n = read(fds[0], buf, sizeof buf);
        close(fds[0]);
        printf("pipe=%.*s\n", (int)n, buf);
    }
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) _exit(7);
    int st = 0;
    waitpid(pid, &st, 0);
    printf("child exit=%d exited=%d\n", WEXITSTATUS(st), WIFEXITED(st) != 0);
    int fd = open("/nonexistent-lf-cc-m9", O_RDONLY);
    printf("open=%d\n", fd);
    struct stat sb;
    int sr = stat("/", &sb);
    printf("stat=%d isdir=%d\n", sr, S_ISDIR(sb.st_mode) != 0);
    return 0;
}
"#,
            stdout: "offsetof=16 sizeof=20\n\
                     1099511627776 4294967295 ffffffff 2147483647\n\
                     htonl=44332211 htons=cdab ntohl=deadbeef\n\
                     longjmp=42\n\
                     signal=1\n\
                     strftime=2000-02-29 12:34\n\
                     pipe=pipe!\n\
                     child exit=7 exited=1\n\
                     open=-1\n\
                     stat=0 isdir=1\n",
            stderr: "",
            exit: 0,
        },
    ]
}

fn which(prog: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(prog)).find(|c| c.is_file())
}

fn prerequisites() -> bool {
    if !Path::new("/usr/include/stdio.h").is_file() {
        eprintln!("skipping: no /usr/include/stdio.h");
        return false;
    }
    if HostCrt::discover().is_none() {
        eprintln!("skipping: no host C runtime (crt1.o) found");
        return false;
    }
    true
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lf-cc-m9-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(exe: &Path) -> Output {
    for _ in 0..50 {
        match Command::new(exe).output() {
            Err(e) if e.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            other => return other.expect("run executable"),
        }
    }
    panic!("{} stayed busy (ETXTBSY)", exe.display());
}

/// Compare a run against the program's expectations; describe any mismatch.
fn check(p: &Prog, how: &str, out: &Output, failures: &mut Vec<String>) {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stdout != p.stdout || stderr != p.stderr || out.status.code() != Some(p.exit) {
        failures.push(format!(
            "{} [{how}]: exit {:?} (expected {})\n--- stdout:\n{stdout}--- expected:\n{}\
             --- stderr:\n{stderr}--- expected:\n{}",
            p.name, out.status.code(), p.exit, p.stdout, p.stderr
        ));
    }
}

/// Compile `args` with lf-cc; on failure record the diagnostics and return false.
fn lf_cc(dir: &Path, args: &[&str], what: &str, failures: &mut Vec<String>) -> bool {
    let out = Command::new(LF_CC).args(args).current_dir(dir).output().expect("run lf-cc");
    if !out.status.success() {
        failures.push(format!("{what}: lf-cc failed:\n{}", String::from_utf8_lossy(&out.stderr)));
        return false;
    }
    true
}

#[test]
fn real_header_programs_with_own_preprocessor() {
    if !prerequisites() {
        return;
    }
    let s = Scratch::new("own");
    let mut failures = Vec::new();
    for p in programs() {
        let src = format!("{}.c", p.name);
        std::fs::write(s.0.join(&src), p.src).expect("write source");
        for (tag, flags) in
            [("O0", &["-O0"][..]), ("O2", &["-O2"][..]), ("O2-gnu", &["-O2", "-D_GNU_SOURCE"][..])]
        {
            let exe = format!("{}.{tag}", p.name);
            let mut args: Vec<&str> = flags.to_vec();
            args.extend([src.as_str(), "-o", exe.as_str(), "-lm"]);
            if lf_cc(&s.0, &args, &format!("{} ({tag})", p.name), &mut failures) {
                check(&p, &format!("lf-cc {tag}"), &run(&s.0.join(&exe)), &mut failures);
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The parser/sema/lowering on exactly the text gcc compiles: preprocess with
/// `gcc -E` (plain, and with `-O2 -D_GNU_SOURCE`, which enables glibc's
/// `extern __inline` definitions), compile the result with `lf-cc -nostdinc`,
/// and also confirm gcc's own build agrees with the expected output.
#[test]
fn real_header_programs_with_gcc_preprocessing() {
    if !prerequisites() {
        return;
    }
    let Some(gcc) = which("gcc") else {
        eprintln!("skipping: gcc (the preprocessing oracle) is not installed");
        return;
    };
    let s = Scratch::new("gcc-e");
    let mut failures = Vec::new();
    for p in programs() {
        let src = format!("{}.c", p.name);
        std::fs::write(s.0.join(&src), p.src).expect("write source");
        // gcc's own build validates the expectations themselves.
        let gexe = format!("{}.gcc", p.name);
        let ok = Command::new(&gcc)
            .args(["-std=gnu17", "-w", &src, "-o", &gexe, "-lm"])
            .current_dir(&s.0)
            .status()
            .expect("run gcc")
            .success();
        assert!(ok, "gcc failed to build {}", p.name);
        check(&p, "gcc", &run(&s.0.join(&gexe)), &mut failures);
        for (tag, flags) in [("plain", &[][..]), ("O2-gnu", &["-O2", "-D_GNU_SOURCE"][..])] {
            let pre = format!("{}.{tag}.i.c", p.name);
            let out = Command::new(&gcc)
                .args(["-std=gnu17", "-E", "-P"])
                .args(flags)
                .arg(&src)
                .current_dir(&s.0)
                .output()
                .expect("run gcc -E");
            assert!(out.status.success(), "gcc -E failed on {}", p.name);
            std::fs::write(s.0.join(&pre), &out.stdout).expect("write preprocessed");
            let exe = format!("{}.{tag}", p.name);
            let what = format!("{} (gcc -E {tag})", p.name);
            if lf_cc(&s.0, &["-nostdinc", &pre, "-o", &exe, "-lm"], &what, &mut failures) {
                check(&p, &format!("gcc -E {tag} + lf-cc"), &run(&s.0.join(&exe)), &mut failures);
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The M9 target headers, and more of the C library's public headers.
const HEADERS: &[&str] = &[
    "stdio.h", "stdlib.h", "string.h", "unistd.h", "errno.h", "ctype.h", "fcntl.h", "signal.h",
    "sys/stat.h", "sys/types.h", "time.h", "stdarg.h", "stddef.h", "limits.h", "setjmp.h",
    "dirent.h", "math.h", "utime.h", "locale.h", "wchar.h", "stdint.h", "inttypes.h", "assert.h",
    "sys/wait.h", "sys/time.h", "pwd.h", "grp.h", "termios.h", "strings.h", "sys/socket.h",
    "netinet/in.h", "arpa/inet.h", "netdb.h", "sys/mman.h", "sys/ioctl.h", "poll.h",
    "sys/select.h", "pthread.h", "sched.h", "regex.h", "glob.h", "fnmatch.h", "getopt.h",
    "libgen.h", "search.h", "iconv.h", "langinfo.h", "wctype.h", "fenv.h", "complex.h",
    "sys/resource.h", "sys/utsname.h", "sys/uio.h", "sys/un.h", "sys/param.h", "endian.h",
    "byteswap.h", "err.h", "ftw.h", "dlfcn.h", "stdbool.h", "float.h", "alloca.h", "malloc.h",
    "syslog.h", "semaphore.h", "spawn.h",
];

/// Write `#include <h>` + an empty `main` for header `h`; returns the file stem.
fn header_probe(dir: &Path, h: &str) -> String {
    let stem = h.replace(['/', '.'], "_");
    std::fs::write(
        dir.join(format!("{stem}.c")),
        format!("#include <{h}>\nint main(void) {{ return 0; }}\n"),
    )
    .expect("write source");
    stem
}

/// Every header, `#include`d alone, compiles and links through lf-cc's own
/// preprocessor (plain, and `-O2 -D_GNU_SOURCE`).
#[test]
fn every_target_header_compiles_with_own_preprocessor() {
    if !prerequisites() {
        return;
    }
    let s = Scratch::new("own-headers");
    let mut failures = Vec::new();
    for h in HEADERS {
        if !Path::new("/usr/include").join(h).is_file() {
            continue;
        }
        let stem = header_probe(&s.0, h);
        let src = format!("{stem}.c");
        for (tag, flags) in [("plain", &[][..]), ("O2-gnu", &["-O2", "-D_GNU_SOURCE"][..])] {
            let exe = format!("{stem}.{tag}");
            let mut args: Vec<&str> = flags.to_vec();
            args.extend([src.as_str(), "-o", exe.as_str()]);
            lf_cc(&s.0, &args, &format!("<{h}> ({tag})"), &mut failures);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every header, `#include`d alone in its gcc-preprocessed form (plain and
/// `-O2 -D_GNU_SOURCE`), compiles and links.
#[test]
fn every_target_header_compiles_in_gcc_preprocessed_form() {
    if !prerequisites() {
        return;
    }
    let Some(gcc) = which("gcc") else {
        eprintln!("skipping: gcc (the preprocessing oracle) is not installed");
        return;
    };
    let s = Scratch::new("headers");
    let mut failures = Vec::new();
    for h in HEADERS {
        if !Path::new("/usr/include").join(h).is_file() {
            continue;
        }
        let stem = header_probe(&s.0, h);
        let src = format!("{stem}.c");
        for (tag, flags) in [("plain", &[][..]), ("O2-gnu", &["-O2", "-D_GNU_SOURCE"][..])] {
            let pre = format!("{stem}.{tag}.i.c");
            let out = Command::new(&gcc)
                .args(["-std=gnu17", "-E", "-P"])
                .args(flags)
                .arg(&src)
                .current_dir(&s.0)
                .output()
                .expect("run gcc -E");
            if !out.status.success() {
                continue; // not usable on this system
            }
            std::fs::write(s.0.join(&pre), &out.stdout).expect("write preprocessed");
            let exe = format!("{stem}.{tag}");
            lf_cc(&s.0, &["-nostdinc", &pre, "-o", &exe], &format!("<{h}> ({tag})"), &mut failures);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
