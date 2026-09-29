//! Shared libraries and PIE executables from `.lf` IR through the real `lf`
//! driver (`lf build --shared` / `--pie` / `-c --pic`), linked by qld.
//!
//! The library exports C-ABI functions that use their own global data, call
//! libc (`strlen`), and take a callback; it also has hidden symbols. The ELF
//! properties (SONAME, no text relocations, the dynamic relocations, which
//! symbols are exported) are checked with a small ELF reader here, so no
//! external tool is needed; `readelf` cross-checks them when present. Loading
//! the library from C (`-L -l` and `dlopen`), interposition, and a
//! `-c --pic` object in a gcc-linked PIE need `gcc` and are skipped without it.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const LIB: &str = r#"module "demo"

global @lf_counter : i64 = i64 0
global hidden @hidden_state : i64 = i64 100
global constant @greeting : [6 x i8] = [6 x i8] "hello\0"
global constant @greeting_ptr : ptr = ptr @greeting

func @strlen(ptr) -> i64

func @lf_bump(i64) -> i64 {
entry ^0(%x: i64):
  %c = load @lf_counter align 8 : i64
  %n = add %c, %x : i64
  store %n, @lf_counter align 8 : i64
  %h = load @hidden_state align 8 : i64
  %r = add %n, %h : i64
  ret %r
}

func @lf_len(ptr) -> i64 {
entry ^0(%s: ptr):
  %n = call @strlen(%s) : i64
  ret %n
}

func @lf_greeting_len() -> i64 {
entry ^0:
  %p = load @greeting_ptr align 8 : ptr
  %n = call @strlen(%p) : i64
  ret %n
}

func @lf_apply(ptr, i64) -> i64 {
entry ^0(%f: ptr, %x: i64):
  %r = call %f(%x) : i64
  %s = add %r, i64 1 : i64
  ret %s
}

func @lf_answer() -> i64 {
entry ^0:
  ret i64 42
}

func @lf_ask() -> i64 {
entry ^0:
  %r = call @lf_answer() : i64
  ret %r
}

func @lf_answer_addr() -> ptr {
entry ^0:
  ret @lf_answer
}

func hidden @lf_secret() -> i64 {
entry ^0:
  ret i64 7
}

func @lf_ask_secret() -> i64 {
entry ^0:
  %r = call @lf_secret() : i64
  ret %r
}
"#;

/// A library that `LD_PRELOAD` puts in front of `libdemo.so`.
const PRELOAD: &str = r#"module "pre"

func @lf_answer() -> i64 {
entry ^0:
  ret i64 77
}

func @lf_secret() -> i64 {
entry ^0:
  ret i64 77
}
"#;

/// A C program linked against the library with `-L -l`. It interposes the
/// library's default-visibility `lf_answer` (the library's own call must reach
/// it) and tries to interpose the hidden `lf_secret` (which must not work).
const MAIN_C: &str = r#"#include <stdio.h>
#include <stdint.h>

int64_t lf_bump(int64_t);
int64_t lf_len(const char *);
int64_t lf_greeting_len(void);
int64_t lf_apply(int64_t (*)(int64_t), int64_t);
int64_t lf_ask(void);
int64_t lf_ask_secret(void);
void *lf_answer_addr(void);
extern int64_t lf_counter;

static int64_t twice(int64_t x) { return 2 * x; }

int64_t lf_answer(void) { return 99; }
int64_t lf_secret(void) { return 99; }

int main(void) {
    printf("bump %ld %ld\n", (long)lf_bump(5), (long)lf_bump(1));
    printf("counter %ld\n", (long)lf_counter);
    printf("len %ld greeting %ld\n", (long)lf_len("abcdefg"), (long)lf_greeting_len());
    printf("apply %ld\n", (long)lf_apply(twice, 20));
    printf("ask %ld secret %ld\n", (long)lf_ask(), (long)lf_ask_secret());
    printf("same address %d\n", lf_answer_addr() == (void *)lf_answer);
    return 0;
}
"#;

/// A C program that loads the library with `dlopen` and calls it via `dlsym`.
const DLOPEN_C: &str = r#"#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>

static int64_t inc(int64_t x) { return x + 1; }

int main(int argc, char **argv) {
    (void)argc;
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
    int64_t (*bump)(int64_t) = (int64_t (*)(int64_t))dlsym(h, "lf_bump");
    int64_t (*len)(const char *) = (int64_t (*)(const char *))dlsym(h, "lf_len");
    int64_t (*apply)(int64_t (*)(int64_t), int64_t) =
        (int64_t (*)(int64_t (*)(int64_t), int64_t))dlsym(h, "lf_apply");
    int64_t (*ask)(void) = (int64_t (*)(void))dlsym(h, "lf_ask");
    int64_t (*ask_secret)(void) = (int64_t (*)(void))dlsym(h, "lf_ask_secret");
    if (!bump || !len || !apply || !ask || !ask_secret) { printf("dlsym failed\n"); return 2; }
    printf("bump %ld len %ld apply %ld\n", (long)bump(3), (long)len("xyz"), (long)apply(inc, 9));
    printf("ask %ld secret %ld\n", (long)ask(), (long)ask_secret());
    printf("hidden %d %d\n", dlsym(h, "lf_secret") == NULL, dlsym(h, "hidden_state") == NULL);
    return dlclose(h);
}
"#;

/// A PIE program: prints the run-time addresses of `main` and a global (ASLR
/// moves them between runs), and a value read through a pointer constant.
const PIE: &str = r#"module "pie"

global constant @fmt : [11 x i8] = [11 x i8] "%p %p %ld\n\0"
global @value : i64 = i64 40
global constant @vptr : ptr = ptr @value

func @printf(ptr, ...) -> i32

func @main() -> i32 {
entry ^0:
  %p = load @vptr align 8 : ptr
  %v = load %p align 8 : i64
  %w = add %v, i64 2 : i64
  %r = call @printf(@fmt, @main, @value, %w) : i32
  ret i32 0
}
"#;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-shared-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn lf(args: &[&dyn AsRef<std::ffi::OsStr>]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_lf"));
    cmd.arg("build");
    for a in args {
        cmd.arg(a);
    }
    cmd.output().expect("run lf")
}

fn lf_ok(args: &[&dyn AsRef<std::ffi::OsStr>]) {
    let out = lf(args);
    assert!(out.status.success(), "lf build failed: {}", String::from_utf8_lossy(&out.stderr));
}

/// Run a program, retrying a transient ETXTBSY (errno 26) from a concurrent fork.
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

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// Build `libdemo.so` (SONAME `libdemo.so.1`) in `dir`, plus the
/// `libdemo.so.1` name the dynamic loader looks for.
fn build_lib(dir: &Path) -> PathBuf {
    let src = dir.join("demo.lf");
    std::fs::write(&src, LIB).unwrap();
    let so = dir.join("libdemo.so");
    lf_ok(&[&"--shared", &src, &"-o", &so, &"-soname", &"libdemo.so.1"]);
    std::fs::copy(&so, dir.join("libdemo.so.1")).unwrap();
    so
}

fn gcc(dir: &Path, c_src: &str, name: &str, extra: &[&dyn AsRef<std::ffi::OsStr>]) -> PathBuf {
    let c = dir.join(format!("{name}.c"));
    std::fs::write(&c, c_src).unwrap();
    let exe = dir.join(name);
    let mut cmd = Command::new("gcc");
    cmd.arg(&c).arg("-o").arg(&exe);
    for a in extra {
        cmd.arg(a);
    }
    let out = cmd.output().expect("run gcc");
    assert!(out.status.success(), "gcc {name}: {}", String::from_utf8_lossy(&out.stderr));
    exe
}

// ---------------------------------------------------------------------------
// A minimal ELF64 reader for the dynamic-linking properties.
// ---------------------------------------------------------------------------

struct Elf {
    bytes: Vec<u8>,
}

struct Shdr {
    kind: u32,
    offset: usize,
    size: usize,
    link: usize,
    entsize: usize,
}

const SHT_RELA: u32 = 4;
const SHT_DYNAMIC: u32 = 6;
const SHT_DYNSYM: u32 = 11;
const DT_NEEDED: u64 = 1;
const DT_SONAME: u64 = 14;
const DT_TEXTREL: u64 = 22;
const DT_FLAGS: u64 = 30;
const DF_TEXTREL: u64 = 4;
const DT_STRTAB: u64 = 5;

impl Elf {
    fn read(path: &Path) -> Elf {
        Elf { bytes: std::fs::read(path).unwrap() }
    }
    fn u16(&self, o: usize) -> usize {
        u16::from_le_bytes([self.bytes[o], self.bytes[o + 1]]) as usize
    }
    fn u32(&self, o: usize) -> u32 {
        u32::from_le_bytes(self.bytes[o..o + 4].try_into().unwrap())
    }
    fn u64(&self, o: usize) -> u64 {
        u64::from_le_bytes(self.bytes[o..o + 8].try_into().unwrap())
    }
    fn e_type(&self) -> usize {
        self.u16(16)
    }
    fn sections(&self) -> Vec<Shdr> {
        let (shoff, shnum) = (self.u64(0x28) as usize, self.u16(0x3C));
        (0..shnum)
            .map(|i| {
                let h = shoff + i * 64;
                Shdr {
                    kind: self.u32(h + 4),
                    offset: self.u64(h + 24) as usize,
                    size: self.u64(h + 32) as usize,
                    link: self.u32(h + 40) as usize,
                    entsize: self.u64(h + 56) as usize,
                }
            })
            .collect()
    }
    fn cstr(&self, at: usize) -> String {
        let end = self.bytes[at..].iter().position(|&b| b == 0).unwrap();
        String::from_utf8(self.bytes[at..at + end].to_vec()).unwrap()
    }
    /// The `.dynamic` entries as `(tag, value)`.
    fn dynamic(&self) -> Vec<(u64, u64)> {
        let secs = self.sections();
        let d = secs.iter().find(|s| s.kind == SHT_DYNAMIC).expect(".dynamic");
        (0..d.size / 16)
            .map(|i| (self.u64(d.offset + i * 16), self.u64(d.offset + i * 16 + 8)))
            .take_while(|&(tag, _)| tag != 0)
            .collect()
    }
    /// Resolve a string in `.dynstr` (the `.dynsym`'s linked string table).
    fn dynstr(&self, off: u64) -> String {
        let secs = self.sections();
        let sym = secs.iter().find(|s| s.kind == SHT_DYNSYM).expect(".dynsym");
        self.cstr(secs[sym.link].offset + off as usize)
    }
    /// Defined `.dynsym` symbols (the exports) as `(name, st_other visibility)`.
    fn exports(&self) -> Vec<(String, u8)> {
        let secs = self.sections();
        let sym = secs.iter().find(|s| s.kind == SHT_DYNSYM).expect(".dynsym");
        let strs = secs[sym.link].offset;
        (1..sym.size / 24)
            .map(|i| sym.offset + i * 24)
            .filter(|&e| self.u16(e + 6) != 0)
            .map(|e| (self.cstr(strs + self.u32(e) as usize), self.bytes[e + 5] & 3))
            .collect()
    }
    /// Every dynamic relocation type.
    fn dyn_reloc_types(&self) -> Vec<u32> {
        let secs = self.sections();
        let mut out = Vec::new();
        for s in secs.iter().filter(|s| s.kind == SHT_RELA && secs[s.link].kind == SHT_DYNSYM) {
            for i in 0..s.size / s.entsize.max(24) {
                out.push(self.u64(s.offset + i * 24 + 8) as u32);
            }
        }
        out
    }
}

const R_X86_64_64: u32 = 1;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;

#[test]
fn shared_library_has_soname_exports_and_no_text_relocations() {
    let dir = scratch("props");
    let so = build_lib(&dir);
    let elf = Elf::read(&so);
    assert_eq!(elf.e_type(), 3, "ET_DYN");

    let dynamic = elf.dynamic();
    let soname = dynamic.iter().find(|d| d.0 == DT_SONAME).expect("DT_SONAME");
    assert_eq!(elf.dynstr(soname.1), "libdemo.so.1");
    assert!(dynamic.iter().any(|d| d.0 == DT_NEEDED && elf.dynstr(d.1).starts_with("libc.so")));
    assert!(dynamic.iter().all(|d| d.0 != DT_TEXTREL), "no DT_TEXTREL");
    assert!(dynamic.iter().all(|d| d.0 != DT_FLAGS || d.1 & DF_TEXTREL == 0), "no DF_TEXTREL");
    assert!(dynamic.iter().any(|d| d.0 == DT_STRTAB));

    // Exports: every default-visibility definition; never a hidden symbol.
    let exports = elf.exports();
    let names: Vec<&str> = exports.iter().map(|(n, _)| n.as_str()).collect();
    for want in ["lf_bump", "lf_len", "lf_apply", "lf_ask", "lf_answer", "lf_counter", "greeting_ptr"] {
        assert!(names.contains(&want), "{want} exported: {names:?}");
    }
    for hidden in ["lf_secret", "hidden_state"] {
        assert!(!names.contains(&hidden), "{hidden} must not be exported: {names:?}");
    }
    assert!(exports.iter().all(|(_, vis)| *vis == 0), "exports are STV_DEFAULT");

    // Only the expected dynamic relocations: GOT slots, the PLT slot for
    // strlen, and the pointer in `.data.rel.ro` (absolute or relative).
    let types = elf.dyn_reloc_types();
    assert!(!types.is_empty());
    let allowed = [R_X86_64_64, R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT, R_X86_64_RELATIVE];
    assert!(types.iter().all(|t| allowed.contains(t)), "unexpected dynamic relocs: {types:?}");
    assert!(types.contains(&R_X86_64_JUMP_SLOT), "strlen goes through the PLT: {types:?}");

    // Cross-check with binutils when available.
    if have("readelf") {
        let d = run(Command::new("readelf").arg("-dW").arg(&so));
        let d = String::from_utf8_lossy(&d.stdout);
        assert!(d.contains("Library soname: [libdemo.so.1]"), "{d}");
        assert!(!d.contains("TEXTREL"), "{d}");
        let syms = run(Command::new("readelf").arg("--dyn-syms").arg("-W").arg(&so));
        let syms = String::from_utf8_lossy(&syms.stdout);
        assert!(syms.contains(" lf_bump") && !syms.contains("lf_secret"), "{syms}");
        let r = run(Command::new("readelf").arg("-rW").arg(&so));
        let r = String::from_utf8_lossy(&r.stdout);
        for line in r.lines().filter(|l| l.contains("R_X86_64_")) {
            let ok = ["R_X86_64_64", "R_X86_64_GLOB_DAT", "R_X86_64_JUMP_SLOT", "R_X86_64_RELATIVE"];
            assert!(ok.iter().any(|k| line.contains(k)), "unexpected dynamic reloc: {line}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn c_program_links_against_the_library_and_interposes() {
    if !have("gcc") {
        eprintln!("skipping: gcc not found");
        return;
    }
    let dir = scratch("link");
    build_lib(&dir);
    let rpath = format!("-Wl,-rpath,{}", dir.display());
    let libdir = format!("-L{}", dir.display());
    let exe = gcc(&dir, MAIN_C, "main", &[&libdir, &"-ldemo", &rpath]);
    let out = run(&mut Command::new(&exe));
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        stdout,
        "bump 106 101\n\
         counter 6\n\
         len 7 greeting 5\n\
         apply 41\n\
         ask 99 secret 7\n\
         same address 1\n",
        "own data, libc call, callback, interposition of lf_answer (not of hidden lf_secret), \
         one canonical function address"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dlopen_dlsym_and_ld_preload_interposition() {
    if !have("gcc") {
        eprintln!("skipping: gcc not found");
        return;
    }
    let dir = scratch("dlopen");
    let so = build_lib(&dir);
    let exe = gcc(&dir, DLOPEN_C, "dl", &[&"-ldl"]);
    let out = run(Command::new(&exe).arg(&so));
    assert!(out.status.success(), "{out:?}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "bump 103 len 3 apply 11\nask 42 secret 7\nhidden 1 1\n"
    );

    // LD_PRELOAD a library (also built by lf) defining lf_answer and lf_secret:
    // the preemptible one is interposed, the hidden one is not.
    let pre_src = dir.join("pre.lf");
    std::fs::write(&pre_src, PRELOAD).unwrap();
    let pre = dir.join("libpre.so");
    lf_ok(&[&"--shared", &pre_src, &"-o", &pre]);
    let out = run(Command::new(&exe).arg(&so).env("LD_PRELOAD", &pre));
    assert!(out.status.success(), "{out:?}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "bump 103 len 3 apply 11\nask 77 secret 7\nhidden 1 1\n"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The first two fields `printf("%p %p …")` printed: the addresses of `main`
/// and of a global.
fn addresses(out: &Output) -> (String, String) {
    let s = String::from_utf8_lossy(&out.stdout);
    let mut f = s.split_whitespace();
    let a = f.next().unwrap().to_owned();
    let b = f.next().unwrap().to_owned();
    assert_eq!(f.next(), Some("42"), "value read through a relocated pointer: {s}");
    (a, b)
}

#[test]
fn pie_executable_runs_at_a_randomized_base() {
    let dir = scratch("pie");
    let src = dir.join("pie.lf");
    std::fs::write(&src, PIE).unwrap();
    let exe = dir.join("pie");
    let out = lf(&[&"--pie", &src, &"-o", &exe]);
    if !out.status.success() && String::from_utf8_lossy(&out.stderr).contains("no crt1.o") {
        eprintln!("skipping: no host C runtime");
        return;
    }
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let elf = Elf::read(&exe);
    assert_eq!(elf.e_type(), 3, "a PIE is ET_DYN");
    assert!(elf.dynamic().iter().all(|d| d.0 != DT_TEXTREL));

    // ASLR: across a few runs the load address changes (unless disabled).
    let runs: Vec<(String, String)> = (0..4).map(|_| addresses(&run(&mut Command::new(&exe)))).collect();
    let aslr_on = std::fs::read_to_string("/proc/sys/kernel/randomize_va_space")
        .is_ok_and(|v| v.trim() != "0");
    if aslr_on {
        assert!(runs.iter().any(|r| r.0 != runs[0].0), "main never moved: {runs:?}");
        assert!(runs.iter().any(|r| r.1 != runs[0].1), "data never moved: {runs:?}");
    }

    // A `-c --pic` object (the shared-library model) links into a gcc PIE.
    if have("gcc") {
        let obj = dir.join("pie.o");
        lf_ok(&[&"-c", &"--pic", &src, &"-o", &obj]);
        let exe2 = dir.join("pie2");
        let out = Command::new("gcc").arg("-pie").arg(&obj).arg("-o").arg(&exe2).output().unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        addresses(&run(&mut Command::new(&exe2)));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn driver_rejects_contradictory_output_flags() {
    let dir = scratch("flags");
    let src = dir.join("x.lf");
    std::fs::write(&src, "module \"x\"\n").unwrap();
    for (args, msg) in [
        (vec!["--shared", "--pie"], "exclusive"),
        (vec!["-soname", "x"], "-soname only applies"),
        (vec!["--pic"], "--pic only applies"),
        (vec!["--shared", "--entry", "f"], "--entry only applies"),
        (vec!["-lm"], "-L/-l only apply"),
    ] {
        let mut a: Vec<&dyn AsRef<std::ffi::OsStr>> = args.iter().map(|s| s as _).collect();
        a.push(&src);
        let out = lf(&a);
        assert!(!out.status.success(), "{args:?} should fail");
        assert!(String::from_utf8_lossy(&out.stderr).contains(msg), "{args:?}: {out:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
