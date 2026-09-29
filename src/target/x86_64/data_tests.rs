//! Execution tests for **global data** on x86-64: `.rodata` / `.data` / `.bss`
//! storage emitted by [`crate::codegen::data`] and the `R_X86_64_64` data
//! relocations of address-valued initializers.
//!
//! Each program is `.lf` text, optionally run through the `-O2` pipeline,
//! compiled by our x86-64 backend, linked by our own static linker, and run on
//! the bare kernel (the freestanding `lf build` path). Two more checks cover the
//! object file itself: `readelf` (when installed) must accept the ELF sections
//! and relocations, and qld (`link::gnu`, our GNU-ld-compatible linker) must
//! link the object into a working static executable.

use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;

use crate::ir::Module;
use crate::link::{ImageOptions, link_executable, write_executable};
use crate::mc::object::{RelocKind, SectionKind};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::transform::pipeline::{OptLevel, optimize};

/// Parse `src`, verify, optimize at `level`, verify again, and return it.
fn prepare(src: &str, level: OptLevel) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse .lf: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    optimize(&mut m, level);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify after {level:?}: {e:?}"));
    (m, syms)
}

/// A unique temp path for one test artifact.
fn temp_path(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("lf_data_{tag}_{}_{uniq}", std::process::id()))
}

/// Run the executable at `path`, returning `(stdout, status)`.
fn run(path: &PathBuf) -> (Vec<u8>, std::process::ExitStatus) {
    // Retry a transient ETXTBSY (errno 26): another test thread's fork may
    // briefly hold a writable fd to the file just written.
    let child = loop {
        match std::process::Command::new(path).stdout(std::process::Stdio::piped()).spawn() {
            Ok(c) => break c,
            Err(e) if e.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(e) => panic!("exec our native binary: {e}"),
        }
    };
    let out = child.wait_with_output().expect("wait for child");
    (out.stdout, out.status)
}

/// Compile + link `src` at `level` with our own linker, run it, and return
/// `(stdout, status, image size)`.
fn build_and_run(src: &str, level: OptLevel, tag: &str) -> (Vec<u8>, std::process::ExitStatus, usize) {
    let (m, syms) = prepare(src, level);
    let obj = super::compile_module(&m, &syms);
    let image = link_executable(vec![obj], &ImageOptions::default()).expect("link should succeed");
    let path = temp_path(tag);
    write_executable(path.to_str().unwrap(), &image).expect("write executable");
    let (out, status) = run(&path);
    let _ = std::fs::remove_file(&path);
    (out, status, image.len())
}

/// A string constant in `.rodata`, written with the `write` syscall.
const RODATA_HELLO: &str = r#"
module "rodata_hello"
global constant @msg : [14 x i8] = [14 x i8] "Hello, rodata\n"

func @main() -> i64 {
entry ^0:
  %n = syscall i64 1, i64 1, @msg, i64 14 : i64
  %rc = sub %n, i64 14 : i64
  ret %rc
}
"#;

#[test]
fn rodata_string_written_by_syscall() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let (out, status, _) = build_and_run(RODATA_HELLO, level, "hello");
        assert_eq!(out, b"Hello, rodata\n", "stdout at {level:?}");
        assert_eq!(status.code(), Some(0), "exit status at {level:?}");
    }
    // The string lives in a read-only .rodata section, not in .data.
    let (m, syms) = prepare(RODATA_HELLO, OptLevel::O0);
    let obj = super::compile_module(&m, &syms);
    let ro = obj.sections().iter().find(|s| s.name == ".rodata").expect(".rodata emitted");
    assert_eq!(ro.kind, SectionKind::Rodata);
    assert_eq!(ro.bytes, b"Hello, rodata\n");
    assert!(obj.sections().iter().all(|s| s.name != ".data" && s.name != ".bss"));
}

/// A store to a `constant` global faults: `.rodata` is mapped without `W`.
#[test]
fn store_to_constant_global_faults() {
    let src = r#"
module "ro_store"
global constant @k : i64 = i64 1
func @main() -> i64 {
entry ^0:
  store i64 2, @k align 8 : i64
  ret i64 0
}
"#;
    let (_, status, _) = build_and_run(src, OptLevel::O0, "rostore");
    assert_eq!(status.signal(), Some(11), "expected SIGSEGV, got {status:?}");
}

/// A mutable `.data` counter (initialized to 5) incremented in a loop by an
/// internal `.data` step (3), ten times: 5 + 10*3 = 35.
const COUNTER: &str = r#"
module "counter"
global @counter : i64 = i64 5
global internal @step : i64 = i64 3

func @main() -> i64 {
entry ^0:
  br ^1(i64 0)
^1(%i: i64):
  %c = load @counter align 8 : i64
  %s = load @step align 8 : i64
  %c1 = add %c, %s : i64
  store %c1, @counter align 8 : i64
  %i1 = add %i, i64 1 : i64
  %done = icmp eq %i1, i64 10 : i1
  cond_br %done, ^2, ^1(%i1)
^2:
  %r = load @counter align 8 : i64
  ret %r
}
"#;

#[test]
fn data_counter_incremented_in_loop() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let (_, status, _) = build_and_run(COUNTER, level, "counter");
        assert_eq!(status.code(), Some(35), "exit status at {level:?}");
    }
    let (m, syms) = prepare(COUNTER, OptLevel::O0);
    let obj = super::compile_module(&m, &syms);
    let data = obj.sections().iter().find(|s| s.name == ".data").expect(".data emitted");
    assert_eq!(data.bytes, [5, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]);
    let step = obj.symbol(obj.symbol_id("step").unwrap());
    assert_eq!(step.binding, crate::mc::object::SymbolBinding::Local, "internal → local");
}

/// A 512 KiB zero-initialized `.bss` array (plus a small all-zero aggregate):
/// first summed (must be 0 — the loader zero-fills memory past the file image),
/// then filled with `arr[i] = i` and summed again (= 65535*65536/2).
const BSS: &str = r#"
module "bss"
global @arr : [65536 x i64] = [65536 x i64] poison
global @small : [4 x i32] = [4 x i32] (i32 0, i32 0, i32 0, i32 0)

func @main() -> i64 {
entry ^0:
  br ^1(i64 0, i64 0)
^1(%i: i64, %s: i64):
  %off = mul %i, i64 8 : i64
  %p = ptr_add @arr, %off : ptr
  %v = load %p align 8 : i64
  %s1 = add %s, %v : i64
  store %i, %p align 8 : i64
  %i1 = add %i, i64 1 : i64
  %d = icmp eq %i1, i64 65536 : i1
  cond_br %d, ^2(%s1), ^1(%i1, %s1)
^2(%z: i64):
  br ^3(i64 0, i64 0)
^3(%j: i64, %t: i64):
  %off2 = mul %j, i64 8 : i64
  %q = ptr_add @arr, %off2 : ptr
  %w = load %q align 8 : i64
  %t1 = add %t, %w : i64
  %j1 = add %j, i64 1 : i64
  %d2 = icmp eq %j1, i64 65536 : i1
  cond_br %d2, ^4(%t1), ^3(%j1, %t1)
^4(%sum: i64):
  %sp = ptr_add @small, i64 12 : ptr
  %e = load %sp align 4 : i32
  %e64 = sext %e : i64
  %z2 = add %z, %e64 : i64
  %ok1 = icmp eq %z2, i64 0 : i1
  %ok2 = icmp eq %sum, i64 2147450880 : i1
  %ok = and %ok1, %ok2 : i1
  %r = select %ok, i64 42, i64 1 : i64
  ret %r
}
"#;

#[test]
fn bss_array_zero_filled_and_summed() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let (_, status, image_len) = build_and_run(BSS, level, "bss");
        assert_eq!(status.code(), Some(42), "exit status at {level:?}");
        // The 512 KiB of zeros occupy memory, not file space (memsz > filesz).
        assert!(image_len < 64 * 1024, "image is {image_len} bytes at {level:?}");
    }
    let (m, syms) = prepare(BSS, OptLevel::O0);
    let obj = super::compile_module(&m, &syms);
    let bss = obj.sections().iter().find(|s| s.name == ".bss").expect(".bss emitted");
    assert!(bss.is_nobits());
    assert_eq!(bss.size(), 65536 * 8 + 16);
    assert!(obj.sections().iter().all(|s| s.name != ".data" && s.name != ".rodata"));
}

/// A constant pointer table initialized with the addresses of globals (one with
/// an offset, one internal), of a function (a *forward* reference), and of a
/// `.data` global that points at itself; plus a pointer with a negative offset.
/// `main` loads through each and makes an indirect call through the table:
/// add(7, 5) + 30 = 42.
const TABLE: &str = r#"
module "table"
global constant @table : {ptr, ptr, ptr, ptr} = {ptr, ptr, ptr, ptr} (ptr @a, ptr @b + 8, ptr @add, ptr @self)
global @a : i64 = i64 7
global internal @b : [2 x i64] = [2 x i64] (i64 30, i64 5)
global @self : ptr = ptr @self
global @bm : ptr = ptr @b - 8

func @add(i64, i64) -> i64 {
entry ^0(%x: i64, %y: i64):
  %s = add %x, %y : i64
  ret %s
}

func @main() -> i64 {
entry ^0:
  %pa = load @table align 8 : ptr
  %x = load %pa align 8 : i64
  %t1 = ptr_add @table, i64 8 : ptr
  %pb = load %t1 align 8 : ptr
  %y = load %pb align 8 : i64
  %t2 = ptr_add @table, i64 16 : ptr
  %f = load %t2 align 8 : ptr
  %r = call %f(%x, %y) : i64
  %t3 = ptr_add @table, i64 24 : ptr
  %ps = load %t3 align 8 : ptr
  %pss = load %ps align 8 : ptr
  %i_ps = ptrtoint %ps : i64
  %i_pss = ptrtoint %pss : i64
  %same = icmp eq %i_ps, %i_pss : i1
  %bm = load @bm align 8 : ptr
  %b1 = ptr_add %bm, i64 16 : ptr
  %y2 = load %b1 align 8 : i64
  %same2 = icmp eq %y, %y2 : i1
  %ok = and %same, %same2 : i1
  %r2 = add %r, i64 30 : i64
  %res = select %ok, %r2, i64 1 : i64
  ret %res
}
"#;

#[test]
fn pointer_table_with_data_relocations_and_indirect_call() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let (_, status, _) = build_and_run(TABLE, level, "table");
        assert_eq!(status.code(), Some(42), "exit status at {level:?}");
    }
    // The table's four fields are Abs64 relocations in .rodata; @self and @bm
    // add two more in .data.
    let (m, syms) = prepare(TABLE, OptLevel::O0);
    let obj = super::compile_module(&m, &syms);
    let sec_of = |name: &str| obj.sections().iter().position(|s| s.name == name).unwrap();
    let (ro, data) = (sec_of(".rodata"), sec_of(".data"));
    let data_relocs: Vec<_> =
        obj.relocations().iter().filter(|r| r.section.index() == ro || r.section.index() == data).collect();
    assert_eq!(data_relocs.len(), 6);
    assert!(data_relocs.iter().all(|r| r.kind == RelocKind::Abs64));
    let in_ro: Vec<(u64, &str, i64)> = data_relocs
        .iter()
        .filter(|r| r.section.index() == ro)
        .map(|r| (r.offset, obj.symbol(r.symbol).name.as_str(), r.addend))
        .collect();
    assert_eq!(in_ro, [(0, "a", 0), (8, "b", 8), (16, "add", 0), (24, "self", 0)]);
    assert!(data_relocs.iter().any(|r| obj.symbol(r.symbol).name == "b" && r.addend == -8));
}

/// Whether `cmd` can be run on this host.
fn tool_available(cmd: &str) -> bool {
    std::process::Command::new(cmd).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// `readelf` parses our ELF object's data sections, symbols, and relocations.
#[test]
fn readelf_accepts_data_sections_and_relocations() {
    if !tool_available("readelf") {
        eprintln!("skipping: readelf not installed");
        return;
    }
    let (m, syms) = prepare(TABLE, OptLevel::O0);
    let elf = crate::mc::elf::write(&super::compile_module(&m, &syms));
    let path = temp_path("readelf").with_extension("o");
    std::fs::write(&path, elf).unwrap();
    let out = std::process::Command::new("readelf").args(["-W", "-S", "-r", "-s"]).arg(&path).output().unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(out.status.success(), "readelf failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    for needle in [".rodata", ".data", ".rela.rodata", ".rela.data", "R_X86_64_64", "OBJECT"] {
        assert!(text.contains(needle), "readelf output lacks `{needle}`:\n{text}");
    }
}

/// qld (our GNU-ld-compatible linker) links an object with `.rodata`/`.data`/
/// `.bss` and data relocations into a working static executable. The program
/// is its own freestanding `_start`: it writes a message found through a
/// pointer in `.rodata`, then exits with a `.data` value plus a `.bss` value.
#[test]
fn qld_links_our_data_sections() {
    let src = r#"
module "qld_data"
global constant @msg : [5 x i8] = [5 x i8] "qld!\n"
global constant @pmsg : ptr = ptr @msg
global @val : i64 = i64 40
global @zero : i64 = i64 0

func @_start() -> void {
entry ^0:
  %p = load @pmsg align 8 : ptr
  %n = syscall i64 1, i64 1, %p, i64 5 : i64
  %v = load @val align 8 : i64
  %z = load @zero align 8 : i64
  %c = add %v, %z : i64
  %c2 = add %c, i64 2 : i64
  %x = syscall i64 60, %c2 : i64
  unreachable
}
"#;
    let (m, syms) = prepare(src, OptLevel::O0);
    let obj = super::compile_module(&m, &syms);
    assert!(obj.sections().iter().any(|s| s.name == ".bss"));
    let dir = temp_path("qld");
    std::fs::create_dir_all(&dir).unwrap();
    let (o, exe) = (dir.join("d.o"), dir.join("d"));
    std::fs::write(&o, crate::mc::elf::write(&obj)).unwrap();
    let args: Vec<std::ffi::OsString> =
        vec!["-static".into(), "-o".into(), exe.clone().into(), o.clone().into()];
    crate::link::gnu::link_gnu("test", &args).expect("qld links our object");
    let (out, status) = run(&exe);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(out, b"qld!\n");
    assert_eq!(status.code(), Some(42));
}

/// An object with `.rodata`/`.data`/`.bss` and data relocations round-trips
/// through our `.lfo` format unchanged, and still links from the decoded copy.
#[test]
fn lfo_round_trips_data_object() {
    for src in [TABLE, BSS, COUNTER] {
        let (m, syms) = prepare(src, OptLevel::O0);
        let obj = super::compile_module(&m, &syms);
        let back = crate::mc::lfo::decode(&crate::mc::lfo::encode(&obj)).expect("decode .lfo");
        assert_eq!(back, obj);
        link_executable(vec![back], &ImageOptions::default()).expect("decoded object links");
    }
}
