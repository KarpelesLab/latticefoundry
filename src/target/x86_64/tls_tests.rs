//! Thread-local storage on x86-64 (`docs/ir-design.md` §4c): the access
//! sequence and relocation of each TLS model, the `.tdata`/`.tbss` sections
//! and `STT_TLS` symbols, and — on an x86-64 Linux host — execution: a static
//! no-libc image built by our linker (which sets up the thread pointer and
//! relaxes initial-exec and general-dynamic accesses), a hosted program linked
//! by qld against glibc whose two pthreads each see their own copy, and a
//! `dlopen`ed shared library using general-dynamic TLS.

use crate::codegen::{CodegenOptions, RelocModel};
use crate::ir::text::parse_module;
use crate::mc::object::{ObjectModule, RelocKind, SectionKind, SymbolType};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::target::x86_64::compile_module_with;

/// The thread-locals of the static test's main module: initialized (`.tdata`)
/// and zero (`.tbss`) ones, one declared here and defined elsewhere (`@ext`),
/// and a function using a general-dynamic access from another module.
const MAIN: &str = r#"module "tls_main"
global thread_local @big : i64 = i64 1234605616436508552
global thread_local @small : i32 = i32 7
global thread_local @zero : i64 = i64 0
global internal thread_local @tiny : i16 = i16 0
global thread_local @ext : i32
global @plain : i64 = i64 5

func @gd_get() -> i64
func @gd_set(i64) -> void

func @bit(i1, i32) -> i32 {
entry ^0(%c: i1, %k: i32):
  %z = zext %c : i32
  %s = shl %z, %k : i32
  ret %s
}

func @main() -> i32 {
entry ^0:
  %a = load @big align 8 : i64
  %c0 = icmp ne %a, i64 1234605616436508552 : i1
  %b = load @small align 4 : i32
  %c1 = icmp ne %b, i32 7 : i1
  %z = load @zero align 8 : i64
  %t = load @tiny align 2 : i16
  %zt = zext %t : i64
  %zz = or %z, %zt : i64
  %c2 = icmp ne %zz, i64 0 : i1
  store i64 99, @zero align 8 : i64
  store i16 -3, @tiny align 2 : i16
  %z2 = load @zero align 8 : i64
  %t2 = load @tiny align 2 : i16
  %c3a = icmp ne %z2, i64 99 : i1
  %c3b = icmp ne %t2, i16 -3 : i1
  %c3 = or %c3a, %c3b : i1
  %e = load @ext align 4 : i32
  %c4 = icmp ne %e, i32 -5 : i1
  call @gd_set(i64 40) : void
  %g = call @gd_get() : i64
  %c5 = icmp ne %g, i64 42 : i1
  %p = load @plain align 8 : i64
  %a2 = load @big align 8 : i64
  %c6 = icmp ne %a2, i64 1234605616436508552 : i1
  %r0 = call @bit(%c0, i32 0) : i32
  %r1 = call @bit(%c1, i32 1) : i32
  %r2 = call @bit(%c2, i32 2) : i32
  %r3 = call @bit(%c3, i32 3) : i32
  %r4 = call @bit(%c4, i32 4) : i32
  %r5 = call @bit(%c5, i32 5) : i32
  %r6 = call @bit(%c6, i32 6) : i32
  %o1 = or %r0, %r1 : i32
  %o2 = or %o1, %r2 : i32
  %o3 = or %o2, %r3 : i32
  %o4 = or %o3, %r4 : i32
  %o5 = or %o4, %r5 : i32
  %o6 = or %o5, %r6 : i32
  %p32 = trunc %p : i32
  %pc = icmp ne %p32, i32 5 : i1
  %r7 = call @bit(%pc, i32 7) : i32
  %o7 = or %o6, %r7 : i32
  ret %o7
}
"#;

/// The other module of the static test, compiled as position-independent code
/// for a shared library: every access is general-dynamic.
const LIB: &str = r#"module "tls_lib"
global thread_local @ext : i32 = i32 -5
global thread_local @gdv : i64 = i64 2

func @gd_get() -> i64 {
entry ^0:
  %v = load @gdv align 8 : i64
  ret %v
}

func @gd_set(i64) -> void {
entry ^0(%x: i64):
  %v = load @gdv align 8 : i64
  %n = add %v, %x : i64
  store %n, @gdv align 8 : i64
  ret
}
"#;

fn compile(src: &str, model: RelocModel) -> ObjectModule {
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).expect("parse");
    crate::verify::verify_module(&m).expect("verify");
    compile_module_with(&m, &syms, &CodegenOptions::default().with_reloc_model(model)).object
}

/// The relocations of `obj` against `sym`, as `(kind, addend)`.
fn relocs_to(obj: &ObjectModule, sym: &str) -> Vec<(RelocKind, i64)> {
    obj.relocations()
        .iter()
        .filter(|r| obj.symbol(r.symbol).name == sym)
        .map(|r| (r.kind, r.addend))
        .collect()
}

/// Each model's relocation: local-exec for a definition in a (static or PIE)
/// executable, initial-exec for a declaration there, general-dynamic (with its
/// `__tls_get_addr` call) for everything in a shared library.
#[test]
fn tls_models_pick_their_relocations() {
    for model in [RelocModel::Static, RelocModel::Pie] {
        let obj = compile(MAIN, model);
        assert!(relocs_to(&obj, "big").iter().all(|r| *r == (RelocKind::TpOff32, 0)), "{model:?}");
        assert!(!relocs_to(&obj, "big").is_empty());
        assert!(relocs_to(&obj, "tiny").iter().all(|r| r.0 == RelocKind::TpOff32));
        assert_eq!(relocs_to(&obj, "ext"), vec![(RelocKind::GotTpOff, -4)], "{model:?}");
        assert!(relocs_to(&obj, "__tls_get_addr").is_empty());
    }
    let obj = compile(LIB, RelocModel::Pic);
    assert!(relocs_to(&obj, "gdv").iter().all(|r| *r == (RelocKind::TlsGd, -4)));
    assert_eq!(relocs_to(&obj, "gdv").len(), 2);
    assert_eq!(relocs_to(&obj, "__tls_get_addr"), vec![(RelocKind::Plt32, -4); 2]);
}

/// The exact instruction bytes: `mov reg, fs:[0]` then `lea reg, [reg +
/// x@tpoff]` (local-exec) or `add reg, [rip + x@gottpoff]` (initial-exec), and
/// the 16-byte general-dynamic sequence a linker can relax.
#[test]
fn tls_access_sequences_are_canonical() {
    let text = |obj: &ObjectModule| {
        obj.sections().iter().find(|s| s.kind == SectionKind::Text).unwrap().bytes.clone()
    };
    let obj = compile(MAIN, RelocModel::Static);
    let t = text(&obj);
    for r in obj.relocations() {
        let at = r.offset as usize;
        match r.kind {
            RelocKind::TpOff32 => {
                // REX.W(+R+B) 8D, ModRM mod=10 reg=rm, [SIB 24] — preceded by
                // `64 REX.W 8B /r 25 00000000` with the same register.
                let sib = usize::from(t[at - 1] == 0x24);
                let m = t[at - 1 - sib];
                assert_eq!(m >> 6, 2);
                assert_eq!((m >> 3) & 7, m & 7);
                assert_eq!(t[at - 2 - sib], 0x8D);
                let mov = at - 3 - sib - 9;
                assert_eq!(t[mov], 0x64);
                assert_eq!(&t[mov + 2..mov + 3], &[0x8B]);
                assert_eq!(t[mov + 3] & 0xC7, 0x04);
                assert_eq!(&t[mov + 4..mov + 9], &[0x25, 0, 0, 0, 0]);
            }
            RelocKind::GotTpOff => {
                assert_eq!(t[at - 3] & 0xFB, 0x48);
                assert_eq!(t[at - 2], 0x03);
                assert_eq!(t[at - 1] & 0xC7, 0x05);
                assert_eq!(t[at - 3 - 9], 0x64);
            }
            _ => {}
        }
    }
    let obj = compile(LIB, RelocModel::Pic);
    let t = text(&obj);
    let gd: Vec<_> = obj.relocations().iter().filter(|r| r.kind == RelocKind::TlsGd).collect();
    assert_eq!(gd.len(), 2);
    for r in gd {
        let at = r.offset as usize;
        assert_eq!(&t[at - 4..at], &[0x66, 0x48, 0x8D, 0x3D]);
        assert_eq!(&t[at + 4..at + 8], &[0x66, 0x66, 0x48, 0xE8]);
        let call = obj.relocations().iter().find(|c| c.offset == r.offset + 8).expect("call");
        assert_eq!(call.kind, RelocKind::Plt32);
        assert_eq!(obj.symbol(call.symbol).name, "__tls_get_addr");
    }
}

/// `.tdata` holds the initialized thread-locals, `.tbss` the zero ones; their
/// symbols (and the reference to a declared one) are `STT_TLS`; ordinary data
/// is untouched.
#[test]
fn tls_sections_and_symbols() {
    let obj = compile(MAIN, RelocModel::Static);
    let tdata = obj.sections().iter().find(|s| s.kind == SectionKind::TData).expect(".tdata");
    assert_eq!(tdata.name, ".tdata");
    assert_eq!(tdata.bytes.len(), 12);
    assert_eq!(&tdata.bytes[..8], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&tdata.bytes[8..], &7u32.to_le_bytes());
    let tbss = obj.sections().iter().find(|s| s.kind == SectionKind::TBss).expect(".tbss");
    assert_eq!((tbss.name.as_str(), tbss.size(), tbss.align), (".tbss", 10, 8));
    for name in ["big", "small", "zero", "tiny", "ext"] {
        let id = obj.symbol_id(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(obj.symbol(id).kind, SymbolType::Tls, "{name}");
    }
    let plain = obj.symbol_id("plain").unwrap();
    assert_eq!(obj.symbol(plain).kind, SymbolType::Object);

    // The ELF writer flags them SHF_TLS and the symbols STT_TLS.
    let elf = crate::mc::elf::write(&obj);
    let shoff = u64::from_le_bytes(elf[0x28..0x30].try_into().unwrap()) as usize;
    let shnum = u16::from_le_bytes([elf[0x3C], elf[0x3D]]) as usize;
    let tls_flags: Vec<(u32, u64)> = (0..shnum)
        .map(|i| {
            let h = shoff + i * 64;
            let ty = u32::from_le_bytes(elf[h + 4..h + 8].try_into().unwrap());
            let flags = u64::from_le_bytes(elf[h + 8..h + 16].try_into().unwrap());
            (ty, flags)
        })
        .filter(|&(_, f)| f & 0x400 != 0)
        .collect();
    assert_eq!(tls_flags, vec![(1, 0x403), (8, 0x403)], "PROGBITS + NOBITS, WAT");
}

/// The other targets refuse thread-local storage clearly.
#[test]
fn tls_is_rejected_where_unsupported() {
    let src = "module \"t\"\nglobal thread_local @x : i32 = i32 1\n\
        func @f() -> i32 {\nentry ^0:\n  %v = load @x align 4 : i32\n  ret %v\n}\n";
    let mut syms = StrInterner::new();
    let m = parse_module(src, FileId::new(0), &mut syms).unwrap();
    let r = std::panic::catch_unwind(|| {
        crate::target::aarch64::compile_module(&m, &syms);
    });
    let msg = r.expect_err("aarch64 rejects TLS");
    let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("thread-local storage"), "{msg}");
    let opts = CodegenOptions::default().with_os(crate::target::TargetOs::Windows);
    let r = std::panic::catch_unwind(|| {
        compile_module_with(&m, &syms, &opts);
    });
    assert!(r.is_err(), "Win64 rejects TLS");
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod native {
    use super::*;
    use crate::link::{ImageOptions, link_executable};
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    fn scratch(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("lf-tls-{name}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Run a program, retrying a transient ETXTBSY (errno 26).
    fn run(cmd: &mut Command) -> Output {
        loop {
            match cmd.output() {
                Ok(o) => return o,
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec: {e}"),
            }
        }
    }

    fn have_gcc() -> bool {
        Command::new("gcc").arg("--version").output().is_ok_and(|o| o.status.success())
    }

    fn write_obj(dir: &Path, name: &str, obj: &ObjectModule) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, crate::mc::elf::write(obj)).unwrap();
        p
    }

    fn gcc_c(dir: &Path, name: &str, src: &str, extra: &[&str]) -> PathBuf {
        let c = dir.join(format!("{name}.c"));
        std::fs::write(&c, src).unwrap();
        let o = dir.join(format!("{name}.o"));
        let out = Command::new("gcc").arg("-c").arg("-O1").args(extra).arg(&c).arg("-o").arg(&o).output().unwrap();
        assert!(out.status.success(), "gcc: {}", String::from_utf8_lossy(&out.stderr));
        o
    }

    /// A static image with no libc: our `_start` builds the TLS block and sets
    /// `%fs`; local-exec accesses read the initial values, stores stick, and
    /// the initial-exec (`@ext`) and general-dynamic (`@gdv`, PIC module)
    /// accesses, relaxed by our linker, reach the other module's variables.
    #[test]
    fn static_image_initializes_and_reads_tls() {
        let objs = vec![compile(MAIN, RelocModel::Static), compile(LIB, RelocModel::Pic)];
        let image = link_executable(objs, &ImageOptions::default()).expect("link");
        // A PT_TLS header: filesz = the .tdata image, memsz with .tbss.
        let phnum = u16::from_le_bytes([image[56], image[57]]) as usize;
        let tls = (0..phnum)
            .map(|i| 64 + i * 56)
            .find(|&h| u32::from_le_bytes(image[h..h + 4].try_into().unwrap()) == 7)
            .expect("PT_TLS");
        let rd = |o: usize| u64::from_le_bytes(image[o..o + 8].try_into().unwrap());
        assert_eq!(rd(tls + 32), 32, "p_filesz: 12 + 4 + 8 (two .tdata sections)");
        assert!(rd(tls + 40) > rd(tls + 32), "p_memsz covers .tbss");
        assert_eq!(rd(tls + 48), 8, "p_align");

        let dir = scratch("static");
        let exe = dir.join("tls_static");
        crate::link::write_executable(exe.to_str().unwrap(), &image).unwrap();
        let out = run(&mut Command::new(&exe));
        assert_eq!(out.status.code(), Some(0), "failed checks (bit mask)");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The C side of the hosted test: two pthreads, each running the LF
    /// functions against its own copies of the thread-locals (one defined by
    /// the LF module, one by this file and reached initial-exec from LF).
    const PTHREAD_C: &str = r#"
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>

__thread int32_t ext = -5;
extern __thread int64_t big;
extern __thread int64_t zero;
int64_t lf_work(int64_t);
void *lf_addr(void);

struct res { int64_t k, r1, r2, big, zero; int32_t ext; int same, first; void *addr; };

static void *worker(void *p) {
    struct res *r = p;
    r->first = big == 1234605616436508552LL && zero == 0 && ext == -5;
    r->r1 = lf_work(r->k);
    r->r2 = lf_work(r->k);
    r->big = big; r->zero = zero; r->ext = ext;
    r->addr = lf_addr();
    r->same = r->addr == (void *)&big;
    return 0;
}

int main(void) {
    struct res a = { .k = 3 }, b = { .k = 1000 }, m = { .k = 77 };
    pthread_t ta, tb;
    pthread_create(&ta, 0, worker, &a);
    pthread_create(&tb, 0, worker, &b);
    pthread_join(ta, 0);
    pthread_join(tb, 0);
    worker(&m);  /* the main thread too: its copy is independent of both */
    struct res *rs[3] = { &a, &b, &m };
    int bad = 0;
    for (int i = 0; i < 3; i++) {
        struct res *r = rs[i];
        int64_t k = r->k;
        if (!r->first || !r->same) bad |= 1 << (4 * i);
        if (r->r1 != 1234605616436508552LL + k + 2 * k + (-5 + k)) bad |= 2 << (4 * i);
        if (r->zero != 4 * k || r->ext != -5 + 2 * k) bad |= 4 << (4 * i);
        if (r->big != 1234605616436508552LL + 2 * k) bad |= 8 << (4 * i);
    }
    if (a.addr == b.addr || a.addr == m.addr) bad |= 4096;
    printf("bad=%d\n", bad);
    return bad;
}
"#;

    /// The LF side: `lf_work(k)` adds `k` to `big`, `2k` to `zero` and `k` to
    /// the C-defined `ext`, returning the sum of the updated values.
    const PTHREAD_LF: &str = r#"module "tls_hosted"
global thread_local @big : i64 = i64 1234605616436508552
global thread_local @zero : i64 = i64 0
global thread_local @ext : i32

func @lf_work(i64) -> i64 {
entry ^0(%k: i64):
  %b = load @big align 8 : i64
  %b2 = add %b, %k : i64
  store %b2, @big align 8 : i64
  %z = load @zero align 8 : i64
  %k2 = add %k, %k : i64
  %z2 = add %z, %k2 : i64
  store %z2, @zero align 8 : i64
  %e = load @ext align 4 : i32
  %k32 = trunc %k : i32
  %e2 = add %e, %k32 : i32
  store %e2, @ext align 4 : i32
  %e64 = sext %e2 : i64
  %s1 = add %b2, %z2 : i64
  %s2 = add %s1, %e64 : i64
  %s3 = sub %s2, %z2 : i64
  %s4 = add %s3, %k2 : i64
  ret %s4
}

func @lf_addr() -> ptr {
entry ^0:
  ret @big
}
"#;

    /// A hosted program: the LF object (local-exec for its own variables,
    /// initial-exec for the C one) and a gcc-compiled pthread harness, linked
    /// by qld against glibc — non-PIE and PIE. Each thread sees its own
    /// initialized copies.
    #[test]
    fn hosted_pthreads_each_see_their_own_tls() {
        if !have_gcc() {
            eprintln!("skipping: gcc not found");
            return;
        }
        let Some(crt) = crate::link::gnu::HostCrt::discover() else {
            eprintln!("skipping: no host C runtime");
            return;
        };
        for (model, pie) in [(RelocModel::Static, false), (RelocModel::Pie, true)] {
            let dir = scratch(if pie { "pie" } else { "exec" });
            let lf = write_obj(&dir, "lf.o", &compile(PTHREAD_LF, model));
            let c = gcc_c(&dir, "harness", PTHREAD_C, if pie { &["-fPIE"] } else { &["-fno-pie"] });
            let exe = dir.join("prog");
            let objs: [&Path; 2] = [&lf, &c];
            let extra = vec!["-lpthread".to_owned()];
            let args = if pie {
                crate::link::gnu::host_c_pie_link_args(&crt, &objs, &extra, &exe)
            } else {
                crate::link::gnu::host_c_link_args(&crt, &objs, &extra, &exe)
            };
            crate::link::gnu::link_gnu("ld", &args).expect("qld link");
            let out = run(&mut Command::new(&exe));
            assert!(
                out.status.success(),
                "{model:?}: {}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A shared library of PIC (general-dynamic for every thread-local, via
    /// `__tls_get_addr`), linked by qld and `dlopen`ed by a C program whose two
    /// threads each get fresh, initialized copies.
    #[test]
    fn dlopened_library_uses_general_dynamic_tls() {
        if !have_gcc() {
            eprintln!("skipping: gcc not found");
            return;
        }
        let crt = crate::link::gnu::HostCrt::discover();
        let dir = scratch("gd");
        let lib = write_obj(&dir, "lib.o", &compile(GD_LIB, RelocModel::Pic));
        let so = dir.join("libtlsgd.so");
        let args = crate::link::gnu::shared_library_args(crt.as_ref(), &[&lib], None, &[], &so);
        crate::link::gnu::link_gnu("ld", &args).expect("qld -shared");
        let c = dir.join("main.c");
        std::fs::write(&c, DLOPEN_C).unwrap();
        let exe = dir.join("main");
        let out = Command::new("gcc").arg(&c).arg("-o").arg(&exe).arg("-ldl").arg("-lpthread").output().unwrap();
        assert!(out.status.success(), "gcc: {}", String::from_utf8_lossy(&out.stderr));
        let out = run(Command::new(&exe).arg(&so));
        assert!(
            out.status.success(),
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    const GD_LIB: &str = r#"module "tlsgd"
global thread_local @counter : i64 = i64 100
global hidden thread_local @scratch : i64 = i64 0

func @tls_bump(i64) -> i64 {
entry ^0(%x: i64):
  %c = load @counter align 8 : i64
  %n = add %c, i64 1 : i64
  store %n, @counter align 8 : i64
  %s = load @scratch align 8 : i64
  %s2 = add %s, i64 10 : i64
  store %s2, @scratch align 8 : i64
  %r = add %n, %s2 : i64
  ret %r
}

func @tls_counter_addr() -> ptr {
entry ^0:
  ret @counter
}
"#;

    const DLOPEN_C: &str = r#"
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>

static int64_t (*bump)(int64_t);
static void *(*addr)(void);

struct res { int64_t a, b, c; void *p; };

static void *worker(void *p) {
    struct res *r = p;
    r->a = bump(0); r->b = bump(0); r->c = bump(0);
    r->p = addr();
    return 0;
}

int main(int argc, char **argv) {
    (void)argc;
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
    bump = (int64_t (*)(int64_t))dlsym(h, "tls_bump");
    addr = (void *(*)(void))dlsym(h, "tls_counter_addr");
    if (!bump || !addr) { printf("dlsym\n"); return 2; }
    struct res r[2];
    pthread_t t[2];
    for (int i = 0; i < 2; i++) pthread_create(&t[i], 0, worker, &r[i]);
    for (int i = 0; i < 2; i++) pthread_join(t[i], 0);
    int bad = 0;
    for (int i = 0; i < 2; i++) {
        if (r[i].a != 111 || r[i].b != 122 || r[i].c != 133) bad |= 1 << i;
    }
    if (r[0].p == r[1].p) bad |= 8;
    struct res m; worker(&m);
    if (m.a != 111 || *(int64_t *)m.p != 103) bad |= 16;
    printf("bad=%d %ld %ld %ld\n", bad, (long)r[0].a, (long)r[0].b, (long)r[0].c);
    return bad;
}
"#;
}
