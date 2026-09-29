//! The binary encodings: section layout of linked modules, validation by
//! node, relocatable objects decoded by `llvm-objdump` (instruction bytes,
//! symbols, relocations) and linked by `wasm-ld`, plus the stack report,
//! stack-overflow behavior, and the errors for what wasm32 cannot express.

use std::process::Command;

use super::behavior::{ATOMICS, CALLS, CONTROL, MEMORY, WIDE};
use super::node::{self, Call};
use super::{differential_module, linked, no_node, parse};
use crate::codegen::CodegenOptions;
use crate::target::wasm32::{compile, leb};

/// Run an LLVM tool from `PATH` on `bytes` (written to a scratch file),
/// returning its stdout; `None` when the tool is not installed.
fn llvm_tool(tool: &str, args: &[&str], bytes: &[u8]) -> Option<String> {
    Command::new(tool).arg("--version").output().ok()?;
    let dir = node::scratch(tool);
    let path = dir.join("in.o");
    std::fs::write(&path, bytes).unwrap();
    let out = Command::new(tool).args(args).arg(&path).output().expect("run tool");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(out.status.success(), "{tool} {args:?} failed: {}", String::from_utf8_lossy(&out.stderr));
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The `(id, payload)` of each section of a module.
fn sections(wasm: &[u8]) -> Vec<(u8, Vec<u8>)> {
    assert_eq!(&wasm[..8], b"\0asm\x01\0\0\0", "magic and version");
    let mut at = 8;
    let mut out = Vec::new();
    while at < wasm.len() {
        let id = wasm[at];
        at += 1;
        let len = leb::read_u64(wasm, &mut at).expect("section size") as usize;
        out.push((id, wasm[at..at + len].to_vec()));
        at += len;
    }
    assert_eq!(at, wasm.len(), "sections end exactly at the end of the module");
    out
}

/// The names in an export section payload.
fn export_names(payload: &[u8]) -> Vec<String> {
    let mut at = 0;
    let n = leb::read_u64(payload, &mut at).unwrap();
    (0..n)
        .map(|_| {
            let len = leb::read_u64(payload, &mut at).unwrap() as usize;
            let name = String::from_utf8(payload[at..at + len].to_vec()).unwrap();
            at += len + 1; // name, kind
            leb::read_u64(payload, &mut at).unwrap();
            name
        })
        .collect()
}

const SMALL: &str = r#"
module "small"
global @k : i64 = i64 2
global @tab : [2 x ptr] = [2 x ptr] (ptr @helper, ptr @k)
global @zero : [64 x i8] = [64 x i8] "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
func @ext(i32) -> i32
func @helper(i64) -> i64 {
entry ^0(%a: i64):
  %b = load @k align 8 : i64
  %s = add %a, %b : i64
  ret %s
}
func internal @private() -> i32 {
entry ^0:
  ret i32 1
}
func hidden @hid() -> i32 {
entry ^0:
  %p = call @private() : i32
  ret %p
}
func @main() -> i64 {
entry ^0:
  %slot = alloca i64 : ptr
  store i64 40, %slot align 8 : i64
  %v = load %slot align 8 : i64
  %f = load @tab align 4 : ptr
  %r = call %f(%v) : i64
  %e = call @ext(i32 1) : i32
  %h = call @hid() : i32
  %eh = add %e, %h : i32
  %ee = zext %eh : i64
  %t = add %r, %ee : i64
  ret %t
}
"#;

#[test]
fn linked_module_layout_and_exports() {
    let (m, syms) = parse(SMALL);
    let wasm = linked(&m, &syms);
    let secs = sections(&wasm);
    let ids: Vec<u8> = secs.iter().map(|(id, _)| *id).collect();
    // type, import, function, table, memory, global, export, element, code,
    // data, then the custom `name` section.
    assert_eq!(ids, [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 0]);
    let exports = export_names(&secs[6].1);
    assert_eq!(exports, ["memory", "__heap_base", "helper", "main"], "hidden and internal functions stay private");
    // The one import is `env.ext`.
    let import = &secs[1].1;
    assert!(import.windows(3).any(|w| w == b"env") && import.windows(3).any(|w| w == b"ext"));
    // `.bss` (all zeros) has no data segment: one segment for `.data`.
    let mut at = 0;
    assert_eq!(leb::read_u64(&secs[9].1, &mut at), Some(1));
}

#[test]
fn every_program_validates() {
    if node::node().is_none() {
        return no_node("every_program_validates");
    }
    for (i, src) in [SMALL, CONTROL, MEMORY, CALLS, ATOMICS, WIDE].into_iter().enumerate() {
        let (m, syms) = parse(src);
        let c = compile(&m, &syms, &CodegenOptions::default()).unwrap();
        let linked = c.object.to_linked(&Default::default()).unwrap();
        let object = c.object.to_relocatable();
        assert_eq!(node::validate(&format!("v{i}"), &linked), Some(true), "linked module {i}");
        assert_eq!(node::validate(&format!("o{i}"), &object), Some(true), "relocatable object {i}");
        // Both decode completely.
        sections(&linked);
        sections(&object);
    }
}

#[test]
fn object_symbols_and_relocations() {
    let (m, syms) = parse(SMALL);
    let obj = compile(&m, &syms, &CodegenOptions::default()).unwrap().object.to_relocatable();
    let secs = sections(&obj);
    let customs: Vec<String> = secs
        .iter()
        .filter(|(id, _)| *id == 0)
        .map(|(_, p)| {
            let mut at = 0;
            let n = leb::read_u64(p, &mut at).unwrap() as usize;
            String::from_utf8(p[at..at + n].to_vec()).unwrap()
        })
        .collect();
    assert_eq!(customs, ["linking", "reloc.CODE", "reloc.DATA"]);

    let Some(syms_out) = llvm_tool("llvm-objdump", &["-t"], &obj) else {
        eprintln!("skipping object_symbols_and_relocations: no llvm-objdump");
        return;
    };
    let lines: Vec<Vec<&str>> = syms_out.lines().map(|l| l.split_whitespace().collect()).collect();
    let has = |want: &[&str]| lines.iter().any(|l| want.iter().all(|w| l.contains(w)));
    assert!(has(&["*UND*", "ext"]), "{syms_out}");
    assert!(has(&["g", "F", "CODE", "helper"]), "{syms_out}");
    assert!(has(&["l", "F", "CODE", "private"]), "{syms_out}");
    assert!(has(&["F", "CODE", ".hidden", "hid"]), "{syms_out}");
    assert!(has(&["g", "O", "DATA", "00000008", "k"]), "{syms_out}");
    assert!(has(&["g", "O", "DATA", "tab"]), "{syms_out}");
    assert!(has(&["*UND*", "__stack_pointer"]), "{syms_out}");

    let relocs = llvm_tool("llvm-objdump", &["-r"], &obj).unwrap();
    for want in [
        "R_WASM_MEMORY_ADDR_SLEB  k+0",
        "R_WASM_GLOBAL_INDEX_LEB  __stack_pointer+0",
        "R_WASM_MEMORY_ADDR_SLEB  tab+0",
        "R_WASM_TYPE_INDEX_LEB",
        "R_WASM_FUNCTION_INDEX_LEB ext+0",
        "R_WASM_FUNCTION_INDEX_LEB hid+0",
        "R_WASM_TABLE_INDEX_I32   helper+0",
        "R_WASM_MEMORY_ADDR_I32   k+0",
    ] {
        assert!(relocs.contains(want), "missing `{want}` in:\n{relocs}");
    }
}

const OPS: &str = r#"
module "ops"
global @g8 : i8 = i8 1
global @g64 : i64 = i64 1
global @fns : [1 x ptr] = [1 x ptr] (ptr @sext8)

func @sext8(i8) -> i32 {
entry ^0(%a: i8):
  %r = sext %a : i32
  ret %r
}

func @sext16_64(i16) -> i64 {
entry ^0(%a: i16):
  %r = sext %a : i64
  ret %r
}

func @sat(f32, f64) -> i64 {
entry ^0(%a: f32, %b: f64):
  %x = fptosi %a : i32
  %y = fptoui %a : i32
  %z = fptosi %b : i32
  %w = fptoui %b : i32
  %p = fptosi %a : i64
  %q = fptoui %a : i64
  %s = fptosi %b : i64
  %t = fptoui %b : i64
  %x1 = add %x, %y : i32
  %x2 = add %z, %w : i32
  %x3 = xor %x1, %x2 : i32
  %xe = zext %x3 : i64
  %y1 = add %p, %q : i64
  %y2 = add %s, %t : i64
  %y3 = xor %y1, %y2 : i64
  %r = add %y3, %xe : i64
  ret %r
}

func @conv(i32, i64) -> f64 {
entry ^0(%a: i32, %b: i64):
  %x = sitofp %a : f32
  %y = uitofp %b : f64
  %z = fpext %x : f64
  %r = fadd %y, %z : f64
  %i = bitcast %r : i64
  %f = bitcast %i : f64
  ret %f
}

func @atom(i8) -> i64 {
entry ^0(%v: i8):
  %a = atomic_rmw add seq_cst @g8, %v align 1 : i8
  %b = cmpxchg seq_cst seq_cst @g8, %a, %v align 1 : i8
  fence seq_cst
  %c = atomic_load seq_cst @g64 align 8 : i64
  %d = atomic_rmw xchg seq_cst @g64, %c align 8 : i64
  %be = zext %b : i64
  %r = add %d, %be : i64
  ret %r
}

func @mem(ptr) -> i64 {
entry ^0(%p: ptr):
  %a = load %p align 1 : i8
  %b = load %p align 2 : i16
  %c = load %p align 1 : i40
  store %a, %p align 1 : i8
  store %b, %p align 2 : i16
  store %c, %p align 1 : i40
  %ae = zext %a : i64
  %be = zext %b : i64
  %ce = zext %c : i64
  %s = add %ae, %be : i64
  %r = add %s, %ce : i64
  ret %r
}

func @sw(i32, i32) -> i32 {
entry ^0(%x: i32, %k: i32):
  %f = load @fns align 4 : ptr
  %t = trunc %k : i8
  %c = call %f(%t) : i32
  %s = icmp slt %x, %c : i1
  %m = select %s, %x, %c : i32
  switch %m, ^4 [0: ^1, 1: ^2, 2: ^3]
^1:
  ret i32 10
^2:
  ret i32 20
^3:
  ret i32 30
^4:
  ret %m
}
"#;

/// Our instruction bytes, decoded by an independent disassembler: each
/// function's instructions (ignoring local traffic) must contain these
/// mnemonics in this order.
#[test]
fn instruction_encodings_decode_as_intended() {
    let (m, syms) = parse(OPS);
    let obj = compile(&m, &syms, &CodegenOptions::default()).unwrap().object.to_relocatable();
    let Some(dis) = llvm_tool("llvm-objdump", &["-d"], &obj) else {
        eprintln!("skipping instruction_encodings_decode_as_intended: no llvm-objdump");
        return;
    };
    let expect: &[(&str, &[&str])] = &[
        ("sext8", &["i32.const\t255", "i32.and", "i32.extend8_s", "return", "unreachable", "end"]),
        ("sext16_64", &["i32.extend16_s", "i64.extend_i32_s", "return"]),
        (
            "sat",
            &[
                "i64.trunc_sat_f32_s",
                "i64.trunc_sat_f32_u",
                "i64.add",
                "i64.trunc_sat_f64_s",
                "i64.trunc_sat_f64_u",
                "i64.xor",
                "i32.trunc_sat_f32_s",
                "i32.trunc_sat_f32_u",
                "i32.trunc_sat_f64_s",
                "i32.trunc_sat_f64_u",
                "i64.extend_i32_u",
            ],
        ),
        (
            "conv",
            &["f64.convert_i64_u", "f32.convert_i32_s", "f64.promote_f32", "f64.add", "i64.reinterpret_f64", "f64.reinterpret_i64"],
        ),
        (
            "atom",
            &["i32.atomic.rmw8.add_u\t0", "i32.atomic.rmw8.cmpxchg_u\t0", "atomic.fence", "i64.atomic.load\t0", "i64.atomic.rmw.xchg\t0"],
        ),
        (
            "mem",
            &[
                "i32.load8_u\t0",
                "i32.load16_u\t0",
                "i64.load32_u\t0",
                "i64.load8_u\t4",
                "i64.shl",
                "i64.or",
                "i32.store8\t0",
                "i32.store16\t0",
                "i64.store32\t0",
                "i64.shr_u",
                "i64.store8\t4",
            ],
        ),
        ("sw", &["i32.load\t0", "call_indirect", "i32.lt_s", "select", "block", "block", "block", "block", "br_table \t{1, 2, 3, 0}"]),
    ];
    for (func, want) in expect {
        let header = format!("<{func}>:");
        let start = dis.find(&header).unwrap_or_else(|| panic!("no {func} in:\n{dis}")) + header.len();
        // The body runs to the next function's `<name>:` header line.
        let body = &dis[start..];
        let body = &body[..body.find(">:\n").and_then(|e| body[..e].rfind('\n')).unwrap_or(body.len())];
        let mut rest = body;
        for w in *want {
            let at = rest.find(w).unwrap_or_else(|| panic!("{func}: `{w}` missing or out of order in:\n{body}"));
            rest = &rest[at + w.len()..];
        }
        assert!(!body.contains("<unknown>"), "{func}: undecodable bytes:\n{body}");
    }
}

/// Two separately compiled modules, linked by `wasm-ld`, run under node and
/// compared with the reference interpreter on the IR-linked program.
#[test]
fn wasm_ld_links_our_objects() {
    const LIB: &str = r#"
module "lib"
global @base : i32 = i32 1000
global @names : [2 x ptr] = [2 x ptr] (ptr @base, ptr @base)
func @scale(i32) -> i32 {
entry ^0(%x: i32):
  %b = load @base align 4 : i32
  %r = mul %x, %b : i32
  ret %r
}
func internal @helper(i32) -> i32 {
entry ^0(%x: i32):
  %r = add %x, i32 1 : i32
  ret %r
}
func @get_helper() -> ptr {
entry ^0:
  ret @helper
}
"#;
    const APP: &str = r#"
module "app"
global @base : i32
global @counter : i32 = i32 0
func @scale(i32) -> i32
func @get_helper() -> ptr
func @run(i32) -> i32 {
entry ^0(%x: i32):
  %s = call @scale(%x) : i32
  %f = call @get_helper() : ptr
  %h = call %f(%s) : i32
  %b = load @base align 4 : i32
  %c = load @counter align 4 : i32
  %c2 = add %c, i32 1 : i32
  store %c2, @counter align 4 : i32
  %t = add %h, %b : i32
  %r = add %t, %c2 : i32
  %big = alloca [100 x i32] : ptr
  %p = ptr_add %big, i32 396 : ptr
  store %r, %p align 4 : i32
  %l = load %p align 4 : i32
  ret %l
}
"#;
    let ld = Command::new("wasm-ld").arg("--version").output();
    if ld.is_err() {
        eprintln!("skipping wasm_ld_links_our_objects: no wasm-ld");
        return;
    }
    if node::node().is_none() {
        return no_node("wasm_ld_links_our_objects");
    }
    let dir = node::scratch("wasm-ld");
    let mut paths = Vec::new();
    for (name, src) in [("lib", LIB), ("app", APP)] {
        let (m, syms) = parse(src);
        let obj = compile(&m, &syms, &CodegenOptions::default()).unwrap().object.to_relocatable();
        let p = dir.join(format!("{name}.o"));
        std::fs::write(&p, obj).unwrap();
        paths.push(p);
    }
    let out = dir.join("linked.wasm");
    let status = Command::new("wasm-ld")
        .args(["--no-entry", "--stack-first", "-o"])
        .arg(&out)
        .args(&paths)
        .output()
        .expect("wasm-ld");
    assert!(status.status.success(), "wasm-ld: {}", String::from_utf8_lossy(&status.stderr));
    let wasm = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    // The reference: the same two modules linked at the IR level.
    let mut syms = crate::support::StrInterner::new();
    let mods: Vec<crate::ir::Module> = [LIB, APP]
        .iter()
        .map(|s| {
            let mut m = crate::ir::text::parse_module(s, crate::support::diagnostics::FileId::new(0), &mut syms).unwrap();
            m.set_data_layout(crate::target::wasm32::data_layout());
            m
        })
        .collect();
    let merged = crate::ir::merge_modules(mods, "linked").unwrap();
    let cases: Vec<(&str, Vec<u128>)> = [0u128, 1, 7, 1000].iter().map(|&x| ("run", vec![x])).collect();
    let t = differential_module("wasm-ld-run", &merged, &syms, &wasm, &cases).unwrap();
    assert_eq!(t.compared, 4);
}

/// The linked module puts the shadow stack first: running out of it wraps
/// the stack pointer below 0, and the next access traps instead of
/// overwriting data.
#[test]
fn stack_overflow_traps() {
    const SRC: &str = r#"
module "deep"
global @sentinel : i32 = i32 12345
func @down(i32) -> i32 {
entry ^0(%n: i32):
  %buf = alloca [256 x i32] : ptr
  store %n, %buf align 4 : i32
  %z = icmp eq %n, i32 0 : i1
  cond_br %z, ^1, ^2
^1:
  %s = load @sentinel align 4 : i32
  ret %s
^2:
  %m = sub %n, i32 1 : i32
  %r = call @down(%m) : i32
  ret %r
}
"#;
    let (m, syms) = parse(SRC);
    let wasm = linked(&m, &syms);
    let calls = [
        Call { func: "down".into(), args: vec![("i32", 100)], rets: vec!["i32"] },
        // 1 MiB of stack holds about 1000 frames of 1 KiB.
        Call { func: "down".into(), args: vec![("i32", 5000)], rets: vec!["i32"] },
        Call { func: "down".into(), args: vec![("i32", 3)], rets: vec!["i32"] },
    ];
    let Some(out) = node::run("deep", &wasm, &calls) else { return no_node("stack_overflow_traps") };
    assert_eq!(out[0], Ok(vec![12345]));
    let trap = out[1].as_ref().expect_err("overflow must trap");
    assert!(trap.contains("out of bounds"), "{trap}");
    // The trap left the stack pointer below its base; the data is intact.
    assert!(matches!(&out[2], Ok(v) if v == &[12345]) || out[2].is_err(), "{:?}", out[2]);
}

/// A host passing garbage above a narrow parameter's width: exported
/// functions mask on entry.
#[test]
fn narrow_parameters_from_the_host_are_masked() {
    const SRC: &str = r#"
module "narrow"
func @f(i8, i16, i1) -> i32 {
entry ^0(%a: i8, %b: i16, %c: i1):
  %ae = sext %a : i32
  %be = zext %b : i32
  %s = add %ae, %be : i32
  %ce = zext %c : i32
  %r = add %s, %ce : i32
  ret %r
}
"#;
    let (m, syms) = parse(SRC);
    let wasm = linked(&m, &syms);
    let calls = [Call { func: "f".into(), args: vec![("i32", 0x1ff), ("i32", 0xffff_0003), ("i32", 0x6)], rets: vec!["i32"] }];
    let Some(out) = node::run("narrow", &wasm, &calls) else { return no_node("narrow_parameters_from_the_host_are_masked") };
    // a = -1, b = 3, c = 0.
    assert_eq!(out[0], Ok(vec![2]));
}

#[test]
fn stack_report() {
    let (m, syms) = parse(MEMORY);
    let c = compile(&m, &syms, &CodegenOptions::default()).unwrap();
    let get = |n: &str| c.stack.get(n).unwrap_or_else(|| panic!("{n}"));
    assert_eq!(get("widths").frame_size, 32);
    assert_eq!(get("fmem").frame_size, 16);
    assert_eq!(get("table_sum").frame_size, 0);
    assert!(get("dyn").dynamic_alloca);
    assert_eq!(get("depth").direct_callees, ["depth"]);
    let (m, syms) = parse(CALLS);
    let c = compile(&m, &syms, &CodegenOptions::default()).unwrap();
    assert!(c.stack.get("apply").unwrap().indirect_calls);
    assert_eq!(c.stack.get("host").unwrap().direct_callees, ["host_void", "host_mul3", "host_i64", "host_half"]);
}

#[test]
fn unsupported_constructs_are_errors() {
    let err = |src: &str| -> String {
        let (m, syms) = parse(src);
        compile(&m, &syms, &CodegenOptions::default()).expect_err("must fail").to_string()
    };
    let e = err("module \"s\"\nfunc @f(i64) -> i64 {\nentry ^0(%a: i64):\n  %r = syscall i64 60, %a : i64\n  ret %r\n}\n");
    assert!(e.contains("in function 'f'") && e.contains("syscall"), "{e}");
    let e = err("module \"h\"\nfunc @f(f16) -> f16 {\nentry ^0(%a: f16):\n  ret %a\n}\n");
    assert!(e.contains("f16"), "{e}");
    let e = err(
        "module \"v\"\nfunc @p(i32, ...) -> i32\nfunc @f() -> i32 {\nentry ^0:\n  %r = call @p(i32 1, i32 2) : i32\n  ret %r\n}\n",
    );
    assert!(e.contains("variadic"), "{e}");

    // A 64-bit-pointer module.
    let mut syms = crate::support::StrInterner::new();
    let m = crate::ir::text::parse_module(
        "module \"lp64\"\nfunc @f() -> i32 {\nentry ^0:\n  ret i32 0\n}\n",
        crate::support::diagnostics::FileId::new(0),
        &mut syms,
    )
    .unwrap();
    let e = compile(&m, &syms, &CodegenOptions::default()).unwrap_err().to_string();
    assert!(e.contains("64-bit pointers"), "{e}");

    // Position-independent code is not a thing on wasm.
    let (m, syms) = parse(SMALL);
    let e = compile(&m, &syms, &CodegenOptions::default().with_pic(true)).unwrap_err().to_string();
    assert!(e.contains("position-independent"), "{e}");

    // A global defined elsewhere links with wasm-ld, not stand-alone.
    let (m, syms) = parse("module \"x\"\nglobal @g : i32\nfunc @f() -> i32 {\nentry ^0:\n  %v = load @g align 4 : i32\n  ret %v\n}\n");
    let c = compile(&m, &syms, &CodegenOptions::default()).unwrap();
    let e = c.object.to_linked(&Default::default()).unwrap_err().to_string();
    assert!(e.contains("undefined global 'g'"), "{e}");
    assert!(!c.object.to_relocatable().is_empty());
}

/// Wide multiplication and division become libgcc-named imports (the host or
/// a runtime object supplies them); export names stay unique.
#[test]
fn libcall_imports_and_unique_exports() {
    let src = r#"
module "libcalls"
func @mul128(i128, i128) -> i128 {
entry ^0(%a: i128, %b: i128):
  %r = mul %a, %b : i128
  ret %r
}
func @udiv128(i128, i128) -> i128 {
entry ^0(%a: i128, %b: i128):
  %r = udiv %a, %b : i128
  ret %r
}
func @memory() -> i32 {
entry ^0:
  ret i32 7
}
"#;
    let (m, syms) = parse(src);
    let wasm = linked(&m, &syms);
    let secs = sections(&wasm);
    let imports = &secs.iter().find(|(id, _)| *id == 2).expect("an import section").1;
    for name in [&b"__multi3"[..], b"__udivti3"] {
        assert!(imports.windows(name.len()).any(|w| w == name), "import {}", String::from_utf8_lossy(name));
    }
    let exports = export_names(&secs.iter().find(|(id, _)| *id == 7).unwrap().1);
    assert_eq!(exports, ["memory", "__heap_base", "mul128", "udiv128"]);
    if node::node().is_some() {
        assert_eq!(node::validate("libcalls", &wasm), Some(true));
    }
}

/// Values whose live ranges do not overlap share locals: a chain of 200
/// values, each used twice (so none is recomputed in place), needs only a
/// couple of locals, and a loop's parameters keep theirs apart.
#[test]
fn locals_are_reused() {
    let mut body = String::from("  %v0 = add %a, i32 1 : i32\n");
    for k in 1..200 {
        body += &format!("  %v{k} = mul %v{}, %v{} : i32\n", k - 1, k - 1);
    }
    body += "  ret %v199\n";
    let src = format!("module \"chain\"\nfunc @chain(i32) -> i32 {{\nentry ^0(%a: i32):\n{body}}}\n");
    let (m, syms) = parse(&src);
    let c = compile(&m, &syms, &CodegenOptions::default()).unwrap();
    let locals = &c.object.funcs[0].body.as_ref().unwrap().locals;
    assert!(locals.len() <= 2, "{} locals for a chain", locals.len());
    if node::node().is_some() {
        let wasm = c.object.to_linked(&Default::default()).unwrap();
        let t = differential_module("chain", &m, &syms, &wasm, &[("chain", vec![3]), ("chain", vec![0xdead_beef])]);
        assert_eq!(t.unwrap().compared, 2);
    }
}

#[test]
fn data_layout_is_ilp32_with_native_i64() {
    let dl = crate::target::wasm32::data_layout();
    assert_eq!(dl.to_spec(), "e-p:32:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-S128-n32:64");
    assert_eq!(crate::codegen::legalize_int::LegalizeOptions::for_layout(&dl).part_bits, 64);
}

/// The target registry and the object writer dispatch.
#[test]
fn registry_and_envelope() {
    use crate::target::{ObjectFormat, TargetArch, Triple, compile_module_for};
    let (m, syms) = parse(SMALL);
    let compiled = compile_module_for(TargetArch::Wasm32, &m, &syms, &CodegenOptions::default()).unwrap();
    let triple = Triple::parse("wasm32-unknown-unknown").unwrap();
    let bytes = crate::mc::write_object(&compiled.object, triple).unwrap();
    assert_eq!(&bytes[..4], b"\0asm");
    assert!(sections(&bytes).iter().any(|(id, p)| *id == 0 && p[1..].starts_with(b"linking")));
    assert!(compiled.stack.get("main").is_some());
    // Other formats for wasm32, and wasm for other architectures, are errors.
    assert!(crate::mc::format::write_object_as(&compiled.object, TargetArch::Wasm32, ObjectFormat::Elf).is_err());
    assert!(crate::mc::format::write_object_as(&compiled.object, TargetArch::X86_64, ObjectFormat::Wasm).is_err());
    // An unsupported construct is a CodegenError, not a panic.
    let (m, syms) = parse("module \"s\"\nfunc @f(i64) -> i64 {\nentry ^0(%a: i64):\n  %r = syscall i64 60, %a : i64\n  ret %r\n}\n");
    let e = compile_module_for(TargetArch::Wasm32, &m, &syms, &CodegenOptions::default()).unwrap_err();
    assert!(e.to_string().starts_with("wasm32: "), "{e}");
}
