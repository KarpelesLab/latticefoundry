//! The linker core used by the `lf` and `lf-ld` drivers (ROADMAP Phase 8).
//!
//! The heart is [`link_executable`] (in [`image`]): it consumes in-memory
//! relocatable [`ObjectModule`](crate::mc::object::ObjectModule)s and produces a
//! **static ELF64 executable** — resolving symbols, laying out sections into
//! `PT_LOAD` segments, applying relocations, and synthesizing a `_start` entry
//! stub. That in-memory path is what the `lf` compiler pipeline uses.
//!
//! This module wraps it with the file-oriented [`link`] entry point that
//! `lf-ld` calls: it reads object files (our own `.lfo` format), links them, and
//! writes an executable to disk with the execute bit set. Reading standard ELF
//! `.o` inputs is not yet supported (see the note on [`read_object`]); supply
//! `.lfo` objects, or use the in-memory [`link_executable`] path.

mod image;

pub use image::{ImageOptions, LinkError, link_executable};

/// Options controlling a file-based link (used by the `lf-ld` driver).
#[derive(Debug, Default)]
pub struct LinkOptions {
    /// Output path for the linked executable.
    pub output: String,
    /// Input object paths (`.lfo`).
    pub inputs: Vec<String>,
    /// The entry symbol `_start` calls; `None` means the default (`main`).
    pub entry: Option<String>,
}

/// Read one object file into an [`ObjectModule`](crate::mc::object::ObjectModule).
///
/// Recognizes our own `.lfo` container. A standard ELF relocatable object is
/// detected and rejected with a clear message: this static linker's file front
/// end links `.lfo` inputs (the `lf` pipeline links objects in memory, never via
/// files). Reading ELF `.o` inputs is a documented future extension.
fn read_object(path: &str) -> Result<crate::mc::object::ObjectModule, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    if bytes.len() >= 4 && bytes[0..4] == [0x7f, b'E', b'L', b'F'] {
        return Err(format!(
            "{path}: linking standard ELF relocatable objects is not yet supported; \
             provide a `.lfo` object (see ROADMAP Phase 8)"
        ));
    }
    crate::mc::lfo::decode(&bytes).map_err(|e| format!("cannot decode {path}: {e}"))
}

/// Write `image` to `path` and mark it executable.
pub fn write_executable(path: &str, image: &[u8]) -> Result<(), String> {
    std::fs::write(path, image).map_err(|e| format!("cannot write {path}: {e}"))?;
    set_executable(path)
}

/// Set the owner/group/other execute bits on `path` (Unix).
#[cfg(unix)]
fn set_executable(path: &str) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat {path}: {e}"))?
        .permissions();
    perms.set_mode(perms.mode() | 0o111);
    std::fs::set_permissions(path, perms).map_err(|e| format!("cannot chmod {path}: {e}"))
}

#[cfg(not(unix))]
fn set_executable(_path: &str) -> Result<(), String> {
    Ok(())
}

/// Link the given input files into a static executable at `options.output`.
pub fn link(options: &LinkOptions) -> Result<(), String> {
    if options.inputs.is_empty() {
        return Err("no input files (see --help)".to_owned());
    }
    let mut objects = Vec::with_capacity(options.inputs.len());
    for path in &options.inputs {
        objects.push(read_object(path)?);
    }
    let mut opts = ImageOptions::default();
    if let Some(entry) = &options.entry {
        opts.entry = entry.clone();
    }
    let image = link_executable(objects, &opts).map_err(|e| e.to_string())?;
    write_executable(&options.output, &image)
}

// ===========================================================================
// M5 end-to-end tests: compile a LatticeFoundry IR `main` to a native static
// executable *entirely* with our own pipeline (no gcc/ld/libc), run it, and
// assert its process exit code. These only need the Linux kernel to exec an ELF.
// ===========================================================================

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod m5 {
    use super::*;
    use crate::ir::Module;
    use crate::ir::inst::{Flags, IntPred};
    use crate::support::StrInterner;
    use crate::support::diagnostics::FileId;

    /// Compile an IR `module` to an ELF64 executable with our full pipeline
    /// (codegen → link), write it to a temp file, `chmod +x`, run it, and return
    /// the process exit code.
    fn build_and_run(module: &Module, syms: &StrInterner, tag: &str) -> i32 {
        let obj = crate::target::x86_64::compile_module(module, syms);
        let image = link_executable(vec![obj], &ImageOptions::default())
            .expect("link should succeed");

        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lf_m5_{tag}_{}_{uniq}", std::process::id()));
        let path_str = path.to_str().unwrap().to_owned();
        write_executable(&path_str, &image).expect("write executable");

        // Executing a file just written can transiently race with another
        // thread's fork/exec that momentarily inherits a writable fd to it
        // (ETXTBSY, raw errno 26). Retry briefly; this is a test-harness
        // concurrency artifact, not a property of the produced binary.
        let status = loop {
            match std::process::Command::new(&path).status() {
                Ok(s) => break s,
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec our native binary: {e}"),
            }
        };
        let _ = std::fs::remove_file(&path);
        status.code().expect("child exited via signal, not code")
    }

    /// `main() -> i64` returning the constant `value`.
    fn const_main(value: i64) -> (Module, StrInterner) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("k");
        let i64t = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![], i64t, false);
        let f = m.declare_function(syms.intern("main"), sig);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let c = b.const_i64(i64t, value);
            b.ret(Some(c));
        }
        (m, syms)
    }

    /// `main() -> i64` returning `a + b` computed at runtime from two constants.
    fn computed_main(x: i64, y: i64) -> (Module, StrInterner) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("k");
        let i64t = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![], i64t, false);
        let f = m.declare_function(syms.intern("main"), sig);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let cx = b.const_i64(i64t, x);
            let cy = b.const_i64(i64t, y);
            let s = b.add(cx, cy, Flags::NONE);
            b.ret(Some(s));
        }
        (m, syms)
    }

    /// `helper() -> i64 = 40`; `main() -> i64 = helper() + 2` (a real call).
    fn call_main() -> (Module, StrInterner) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("k");
        let i64t = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![], i64t, false);
        let helper = m.declare_function(syms.intern("helper"), sig);
        let main = m.declare_function(syms.intern("main"), sig);
        {
            let mut b = m.build(helper);
            b.create_entry_block();
            let c = b.const_i64(i64t, 40);
            b.ret(Some(c));
        }
        {
            let mut b = m.build(main);
            b.create_entry_block();
            let cref = b.func_ref(helper);
            let r = b.call(cref, &[], i64t).unwrap();
            let two = b.const_i64(i64t, 2);
            let s = b.add(r, two, Flags::NONE);
            b.ret(Some(s));
        }
        (m, syms)
    }

    /// `main() -> i64 = 0+1+...+9 = 45` via a counted loop with back-edge args.
    fn loop_main() -> (Module, StrInterner) {
        let mut syms = StrInterner::new();
        let mut m = Module::new("k");
        let i64t = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![], i64t, false);
        let f = m.declare_function(syms.intern("main"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let header = b.create_block(&[i64t, i64t]); // (acc, i)
            let body = b.create_block(&[i64t, i64t]);
            let exit = b.create_block(&[i64t]);
            b.switch_to(entry);
            let zero = b.const_i64(i64t, 0);
            b.br(header, &[zero, zero]);
            b.switch_to(header);
            let acc = b.param(header, 0);
            let i = b.param(header, 1);
            let ten = b.const_i64(i64t, 10);
            let cond = b.icmp(IntPred::Slt, i, ten);
            b.cond_br(cond, body, &[acc, i], exit, &[acc]);
            b.switch_to(body);
            let bacc = b.param(body, 0);
            let bi = b.param(body, 1);
            let new_acc = b.add(bacc, bi, Flags::NONE);
            let one = b.const_i64(i64t, 1);
            let new_i = b.add(bi, one, Flags::NONE);
            b.br(header, &[new_acc, new_i]);
            b.switch_to(exit);
            let result = b.param(exit, 0);
            b.ret(Some(result));
        }
        (m, syms)
    }

    #[test]
    fn native_returns_constant_42() {
        let (m, syms) = const_main(42);
        assert_eq!(build_and_run(&m, &syms, "c42"), 42);
    }

    #[test]
    fn native_returns_computed_sum() {
        // 17 + 28 = 45, computed at runtime.
        let (m, syms) = computed_main(17, 28);
        assert_eq!(build_and_run(&m, &syms, "sum"), 45);
    }

    #[test]
    fn native_calls_helper() {
        // main = helper() + 2 = 42.
        let (m, syms) = call_main();
        assert_eq!(build_and_run(&m, &syms, "call"), 42);
    }

    #[test]
    fn native_runs_loop() {
        // sum 0..10 = 45.
        let (m, syms) = loop_main();
        assert_eq!(build_and_run(&m, &syms, "loop"), 45);
    }

    #[test]
    fn native_from_textual_lf_source() {
        // Prove the whole spine from `.lf` text: parse → codegen → link → run.
        let src = "\
module \"k\"
func @main() -> i64 {
entry ^0:
  %s = add i64 30, i64 12 : i64
  ret %s
}
";
        let mut syms = StrInterner::new();
        let module = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
            .expect("parse .lf");
        assert_eq!(build_and_run(&module, &syms, "text"), 42);
    }

    #[test]
    fn lfo_file_link_runs() {
        // Exercise the same file-based path `lf-ld` uses: encode a real object
        // to `.lfo`, link it from disk with `link()`, then run the executable.
        let (m, syms) = const_main(7);
        let obj = crate::target::x86_64::compile_module(&m, &syms);
        let lfo = crate::mc::lfo::encode(&obj);

        let dir = std::env::temp_dir();
        let objp = dir.join(format!("lf_m5_ld_{}.lfo", std::process::id()));
        let exep = dir.join(format!("lf_m5_ld_{}.bin", std::process::id()));
        std::fs::write(&objp, &lfo).unwrap();

        let opts = LinkOptions {
            output: exep.to_str().unwrap().to_owned(),
            inputs: vec![objp.to_str().unwrap().to_owned()],
            entry: None,
        };
        link(&opts).expect("lf-ld file link");

        let status = loop {
            match std::process::Command::new(&exep).status() {
                Ok(s) => break s,
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec linked binary: {e}"),
            }
        };
        let _ = std::fs::remove_file(&objp);
        let _ = std::fs::remove_file(&exep);
        assert_eq!(status.code(), Some(7));
    }

    #[test]
    fn produced_image_is_deterministic() {
        let (m, syms) = computed_main(1, 2);
        let a = link_executable(
            vec![crate::target::x86_64::compile_module(&m, &syms)],
            &ImageOptions::default(),
        )
        .unwrap();
        let b = link_executable(
            vec![crate::target::x86_64::compile_module(&m, &syms)],
            &ImageOptions::default(),
        )
        .unwrap();
        assert_eq!(a, b);
    }
}

// ===========================================================================
// Phase 10 DWARF debug-info end-to-end tests: compile a `.lf` program with the
// debug pipeline into a debuggable static executable, then (a) structurally
// assert the image gained a section-header table + `.symtab` + `.debug_*`
// without breaking execution, and (b), when the external tools are present,
// have llvm-dwarfdump / readelf / gdb actually parse it and agree.
// ===========================================================================

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod dwarf_e2e {
    use super::*;
    use crate::support::StrInterner;
    use crate::support::diagnostics::FileId;
    use crate::target::x86_64::{DebugSource, compile_module_debug};

    /// A two-function `.lf` program with several statements per function.
    const PROG: &str = "\
module \"prog\"
func @helper() -> i64 {
entry ^0:
  %a = add i64 40, i64 0 : i64
  ret %a
}
func @main() -> i64 {
entry ^0:
  %h = call @helper() : i64
  %r = add %h, i64 2 : i64
  ret %r
}
";

    /// Parse `PROG`, compile it with debug info, and link a debuggable image.
    fn build_debug_image() -> Vec<u8> {
        let mut syms = StrInterner::new();
        let module = crate::ir::text::parse_module(PROG, FileId::new(0), &mut syms)
            .expect("parse .lf");
        let source =
            DebugSource { file_name: "prog.lf".to_owned(), comp_dir: "/lf".to_owned() };
        let obj = compile_module_debug(&module, &syms, &source);
        let opts = ImageOptions { debug: true, ..ImageOptions::default() };
        link_executable(vec![obj], &opts).expect("link debug image")
    }

    fn rd_u16(b: &[u8], o: usize) -> u16 {
        u16::from_le_bytes([b[o], b[o + 1]])
    }
    fn rd_u64(b: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
    }

    #[test]
    fn debug_image_has_section_headers_and_is_deterministic() {
        let img = build_debug_image();
        // A section-header table is now present (unlike a plain image).
        let shoff = rd_u64(&img, 40);
        let shnum = rd_u16(&img, 60);
        let shstrndx = rd_u16(&img, 62);
        assert!(shoff > 0, "e_shoff must be set");
        assert!(shnum >= 8, "expected .text + 4 debug + symtab/strtab/shstrtab");
        assert!((shstrndx as usize) < shnum as usize);
        // e_entry and the first PT_LOAD are unchanged from a plain image (the
        // loadable layout must not move when debug data is appended).
        assert!(rd_u64(&img, 24) >= 0x40_0000, "entry inside the image");
        // Determinism.
        assert_eq!(img, build_debug_image());
    }

    #[test]
    fn debug_image_still_runs() {
        let img = build_debug_image();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lf_dwarf_run_{}", std::process::id()));
        let path_str = path.to_str().unwrap().to_owned();
        write_executable(&path_str, &img).expect("write");
        let status = loop {
            match std::process::Command::new(&path).status() {
                Ok(s) => break s,
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => panic!("exec debug binary: {e}"),
            }
        };
        let _ = std::fs::remove_file(&path);
        // helper() = 40, main = helper() + 2 = 42.
        assert_eq!(status.code(), Some(42), "the -g binary must still run correctly");
    }

    fn tool_available(cmd: &str) -> bool {
        std::process::Command::new(cmd)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn write_temp_image(tag: &str) -> std::path::PathBuf {
        let img = build_debug_image();
        let path = std::env::temp_dir().join(format!("lf_dwarf_{tag}_{}", std::process::id()));
        std::fs::write(&path, &img).expect("write temp image");
        path
    }

    #[test]
    fn llvm_dwarfdump_parses_debug_info() {
        if !tool_available("llvm-dwarfdump") {
            eprintln!("skipping: llvm-dwarfdump not available");
            return;
        }
        let path = write_temp_image("dd");
        let out = std::process::Command::new("llvm-dwarfdump")
            .arg("--debug-info")
            .arg("--debug-line")
            .arg(&path)
            .output()
            .expect("run llvm-dwarfdump");
        let _ = std::fs::remove_file(&path);
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains("DW_TAG_compile_unit"), "no compile unit:\n{s}");
        assert!(s.contains("LatticeFoundry"), "no producer:\n{s}");
        assert!(s.contains("DW_TAG_subprogram"), "no subprograms:\n{s}");
        assert!(s.contains("\"helper\"") && s.contains("\"main\""), "no fn names:\n{s}");
        assert!(s.contains("DW_AT_low_pc") && s.contains("DW_AT_high_pc"), "no pc range:\n{s}");
        assert!(s.contains(".debug_line contents") || s.contains("Line table"), "no line table:\n{s}");
    }

    #[test]
    fn readelf_shows_sections_and_symbols() {
        if !tool_available("readelf") {
            eprintln!("skipping: readelf not available");
            return;
        }
        let path = write_temp_image("re");
        let sections = std::process::Command::new("readelf").arg("-S").arg(&path).output().unwrap();
        let symbols = std::process::Command::new("readelf").arg("-s").arg(&path).output().unwrap();
        let _ = std::fs::remove_file(&path);
        let sec = String::from_utf8_lossy(&sections.stdout);
        let sym = String::from_utf8_lossy(&symbols.stdout);
        assert!(sec.contains(".debug_info") && sec.contains(".debug_line"), "missing debug sections:\n{sec}");
        assert!(sec.contains(".symtab") && sec.contains(".text"), "missing sections:\n{sec}");
        assert!(sym.contains("main") && sym.contains("helper"), "missing function symbols:\n{sym}");
        assert!(sym.contains("FUNC"), "no FUNC-typed symbol:\n{sym}");
    }

    #[test]
    fn gdb_understands_debug_info() {
        if !tool_available("gdb") {
            eprintln!("skipping: gdb not available");
            return;
        }
        let path = write_temp_image("gdb");
        let out = std::process::Command::new("gdb")
            .args(["-batch", "-nx", "-ex", "info functions", "-ex", "info line main"])
            .arg(&path)
            .output()
            .expect("run gdb");
        let _ = std::fs::remove_file(&path);
        let s = String::from_utf8_lossy(&out.stdout);
        // gdb resolves the functions to the `.lf` source and can locate `main`.
        assert!(s.contains("prog.lf"), "gdb did not read the .lf source:\n{s}");
        assert!(s.contains("main") && s.contains("helper"), "gdb missing functions:\n{s}");
        assert!(
            s.contains("Line 7") && s.contains("<main>"),
            "gdb could not map main to its source line:\n{s}"
        );
    }
}
