//! The linker core used by the `lf` and `lf-ld` drivers (ROADMAP Phase 8).
//!
//! The heart is [`link_executable`] (in `image`): it consumes in-memory
//! relocatable [`ObjectModule`](crate::mc::object::ObjectModule)s and produces a
//! **static ELF64 executable** — resolving symbols, laying out sections into
//! `PT_LOAD` segments, applying relocations, and synthesizing a `_start` entry
//! stub. That in-memory path is what the `lf` compiler pipeline uses.
//!
//! This module wraps it with the file-oriented [`link`] entry point that
//! `lf-ld` calls: it reads object files (our own `.lfo` format), links them, and
//! writes an executable to disk with the execute bit set.
//!
//! Standard ELF objects, archives and shared libraries are linked by [`gnu`],
//! a bridge onto our GNU-ld-compatible linker `qld`.

pub mod gnu;
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
/// end links `.lfo` inputs; ELF inputs go through [`gnu::link_gnu`] (`lf-ld`
/// routes them there automatically).
fn read_object(path: &str) -> Result<crate::mc::object::ObjectModule, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    if bytes.len() >= 4 && bytes[0..4] == [0x7f, b'E', b'L', b'F'] {
        return Err(format!(
            "{path}: this is an ELF object; link ELF inputs with `link::gnu` \
             (qld) — `lf-ld` does so automatically"
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
    fn native_narrow_icmp_ignores_upper_register_bits() {
        // An i8/i16 add can leave carries above the value's width in the host
        // register (200 + 100 is 300 in a 32-bit register, but 44 as an i8).
        // Comparisons must look at the value's own width only. Each function
        // returns 1 when the comparison sees the wrapped value.
        let src = "\
module \"k\"
func @ult8(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %c = icmp ult %s, %a : i1
  %r = zext %c : i64
  ret %r
}
func @slt8(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %c = icmp slt %s, i8 0 : i1
  %r = zext %c : i64
  ret %r
}
func @eq16(i16, i16) -> i64 {
entry ^0(%a: i16, %b: i16):
  %s = add %a, %b : i16
  %c = icmp eq %s, i16 4 : i1
  %r = zext %c : i64
  ret %r
}
func @main() -> i64 {
entry ^0:
  %x = call @ult8(i8 -56, i8 100) : i64
  %y = call @slt8(i8 100, i8 100) : i64
  %z = call @eq16(i16 -2, i16 6) : i64
  %xy = shl %y, i64 1 : i64
  %xz = shl %z, i64 2 : i64
  %t = or %x, %xy : i64
  %u = or %t, %xz : i64
  ret %u
}
";
        let mut syms = StrInterner::new();
        let module = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
            .expect("parse .lf");
        assert_eq!(build_and_run(&module, &syms, "narrow_icmp"), 0b111);
    }

    /// Build `src`, run it, and return its exit code (bits = failing checks).
    fn run_lf(src: &str, tag: &str) -> i32 {
        let mut syms = StrInterner::new();
        let module = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
            .expect("parse .lf");
        build_and_run(&module, &syms, tag)
    }

    #[test]
    fn native_narrow_ops_ignore_upper_register_bits() {
        // Every op whose result depends on bits above an i8's width must see the
        // wrapped value: 200 + 100 is 44 as an i8 (300 in a 32-bit register), and
        // 100 + 100 is -56 as an i8 (200 in the register). Each function returns
        // 0 when correct; main ORs a distinct bit per failing check.
        let src = "\
module \"k\"
func @lshr(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = lshr %s, i8 1 : i8
  %c = icmp ne %r, i8 22 : i1
  %z = zext %c : i64
  ret %z
}
func @ashr(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = ashr %s, i8 1 : i8
  %c = icmp ne %r, i8 -28 : i1
  %z = zext %c : i64
  ret %z
}
func @udiv(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = udiv %s, i8 2 : i8
  %c = icmp ne %r, i8 22 : i1
  %z = zext %c : i64
  ret %z
}
func @urem(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = urem %s, i8 3 : i8
  %c = icmp ne %r, i8 2 : i1
  %z = zext %c : i64
  ret %z
}
func @sdiv(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = sdiv %s, i8 2 : i8
  %c = icmp ne %r, i8 -28 : i1
  %z = zext %c : i64
  ret %z
}
func @srem(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = srem %s, i8 3 : i8
  %c = icmp ne %r, i8 -2 : i1
  %z = zext %c : i64
  ret %z
}
func @switch(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  switch %s, ^1 [44: ^2]
^1:
  ret i64 1
^2:
  ret i64 0
}
func @condbr(i32) -> i64 {
entry ^0(%a: i32):
  %t = trunc %a : i1
  cond_br %t, ^1, ^2
^1:
  ret i64 1
^2:
  ret i64 0
}
func @sitofp(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %f = sitofp %s : f64
  %i = fptosi %f : i64
  %c = icmp ne %i, i64 -56 : i1
  %z = zext %c : i64
  ret %z
}
func @uitofp(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %f = uitofp %s : f64
  %i = fptosi %f : i64
  %c = icmp ne %i, i64 44 : i1
  %z = zext %c : i64
  ret %z
}
func @main() -> i64 {
entry ^0:
  %v0 = call @lshr(i8 -56, i8 100) : i64
  %v1 = call @ashr(i8 100, i8 100) : i64
  %v2 = call @udiv(i8 -56, i8 100) : i64
  %v3 = call @urem(i8 -56, i8 100) : i64
  %v4 = call @sdiv(i8 100, i8 100) : i64
  %v5 = call @srem(i8 100, i8 100) : i64
  %v6 = call @switch(i8 -56, i8 100) : i64
  %v7 = call @condbr(i32 2) : i64
  %v8 = call @sitofp(i8 100, i8 100) : i64
  %v9 = call @uitofp(i8 -56, i8 100) : i64
  %s1 = shl %v1, i64 1 : i64
  %s2 = shl %v2, i64 2 : i64
  %s3 = shl %v3, i64 3 : i64
  %s4 = shl %v4, i64 4 : i64
  %s5 = shl %v5, i64 5 : i64
  %s6 = shl %v6, i64 6 : i64
  %s7 = shl %v7, i64 7 : i64
  %s8 = shl %v8, i64 8 : i64
  %s9 = shl %v9, i64 9 : i64
  %o1 = or %v0, %s1 : i64
  %o2 = or %o1, %s2 : i64
  %o3 = or %o2, %s3 : i64
  %o4 = or %o3, %s4 : i64
  %o5 = or %o4, %s5 : i64
  %o6 = or %o5, %s6 : i64
  %o7 = or %o6, %s7 : i64
  %o8 = or %o7, %s8 : i64
  %o9 = or %o8, %s9 : i64
  %lo = and %o9, i64 255 : i64
  %hi = lshr %o9, i64 8 : i64
  ret %lo
}
";
        // An exit status holds 8 bits: run once for the low bits, once for the rest.
        let code = run_lf(src, "narrow_ops_lo")
            | (run_lf(&src.replace("ret %lo", "ret %hi"), "narrow_ops_hi") << 8);
        let names = ["lshr", "ashr", "udiv", "urem", "sdiv", "srem", "switch", "cond_br", "sitofp", "uitofp"];
        let failing: Vec<_> =
            names.iter().enumerate().filter(|(i, _)| code & (1 << i) != 0).map(|(_, n)| *n).collect();
        assert!(failing.is_empty(), "narrow ops saw dirty upper bits: {failing:?} (exit {code})");
    }

    #[test]
    fn native_narrow_values_dirty_above_width() {
        // More shapes of the same invariant: `0 - 1` as an i8 is 255 unsigned
        // but 0xFFFFFFFF in a 32-bit register; a `trunc` to i1 keeps the source's
        // other bits; odd widths (i24) have no `cmp`/`movzx` form of their own; a
        // switch compares 64-bit values, and case values may not fit an imm32.
        let src = "\
module \"k\"
func @ushr(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = sub %a, %b : i8
  %r = lshr %s, i8 1 : i8
  %c = icmp ne %r, i8 127 : i1
  %z = zext %c : i64
  ret %z
}
func @udiv(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = sub %a, %b : i8
  %r = udiv %s, i8 2 : i8
  %c = icmp ne %r, i8 127 : i1
  %z = zext %c : i64
  ret %z
}
func @uitofp(i8, i8) -> i64 {
entry ^0(%a: i8, %b: i8):
  %s = sub %a, %b : i8
  %f = uitofp %s : f64
  %i = fptosi %f : i64
  %c = icmp ne %i, i64 255 : i1
  %z = zext %c : i64
  ret %z
}
func @zext1(i32) -> i64 {
entry ^0(%a: i32):
  %t = trunc %a : i1
  %z = zext %t : i64
  ret %z
}
func @select1(i32) -> i64 {
entry ^0(%a: i32):
  %t = trunc %a : i1
  %r = select %t, i64 1, i64 0 : i64
  ret %r
}
func @cmp24(i32) -> i64 {
entry ^0(%a: i32):
  %t = trunc %a : i24
  %c = icmp ne %t, i24 5 : i1
  %z = zext %c : i64
  ret %z
}
func @switch32(i32, i32) -> i64 {
entry ^0(%a: i32, %b: i32):
  %s = sub %a, %b : i32
  switch %s, ^1 [-1: ^2]
^1:
  ret i64 1
^2:
  ret i64 0
}
func @switch64(i64) -> i64 {
entry ^0(%a: i64):
  switch %a, ^1 [4294967296: ^2]
^1:
  ret i64 1
^2:
  ret i64 0
}
func @main() -> i64 {
entry ^0:
  %v0 = call @ushr(i8 0, i8 1) : i64
  %v1 = call @udiv(i8 0, i8 1) : i64
  %v2 = call @uitofp(i8 0, i8 1) : i64
  %v3 = call @zext1(i32 2) : i64
  %v4 = call @select1(i32 2) : i64
  %v5 = call @cmp24(i32 16777221) : i64
  %v6 = call @switch32(i32 0, i32 1) : i64
  %v7 = call @switch64(i64 4294967296) : i64
  %s1 = shl %v1, i64 1 : i64
  %s2 = shl %v2, i64 2 : i64
  %s3 = shl %v3, i64 3 : i64
  %s4 = shl %v4, i64 4 : i64
  %s5 = shl %v5, i64 5 : i64
  %s6 = shl %v6, i64 6 : i64
  %s7 = shl %v7, i64 7 : i64
  %o1 = or %v0, %s1 : i64
  %o2 = or %o1, %s2 : i64
  %o3 = or %o2, %s3 : i64
  %o4 = or %o3, %s4 : i64
  %o5 = or %o4, %s5 : i64
  %o6 = or %o5, %s6 : i64
  %o7 = or %o6, %s7 : i64
  ret %o7
}
";
        let code = run_lf(src, "narrow_dirty");
        let names = ["lshr", "udiv", "uitofp", "zext i1", "select i1", "icmp i24", "switch i32", "switch imm64"];
        let failing: Vec<_> =
            names.iter().enumerate().filter(|(i, _)| code & (1 << i) != 0).map(|(_, n)| *n).collect();
        assert!(failing.is_empty(), "narrow values mishandled: {failing:?} (exit {code})");
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
