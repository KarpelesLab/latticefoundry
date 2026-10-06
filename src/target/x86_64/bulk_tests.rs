//! x86-64 bulk memory (`docs/ir-design.md` §6k): `memcpy`/`memmove`/`memset`
//! run in-process (JIT) across every length 0..=300, constant and variable,
//! at unaligned and aligned offsets, against Rust's own slice operations and
//! with guard bytes around the destination; the `rep movsb`/`rep stosb` and
//! inline SSE shapes in the disassembly; the C-library path linked with gcc;
//! and the Lode code-size repros.

use std::fmt::Write as _;

use crate::codegen::CodegenOptions;
use crate::ir::Module;
use crate::jit::Jit;
use crate::mc::disasm::{Options, Syntax, decode};
use crate::mc::object::{ObjectModule, SymbolValue};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::target::TargetArch;
use crate::transform::pipeline::{OptLevel, optimize};

/// The largest length exercised.
const MAX: usize = 300;

fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    (m, syms)
}

/// A test function `@name(ptr args) -> i64` reading `dst`, `src`/`byte` and
/// `n` from `args[0..3]` (8 bytes each) and running one bulk op on them (the
/// length a constant when `n` is given).
fn gen_fn(out: &mut String, name: &str, op: &str, align: u32, n: Option<usize>) {
    let _ = writeln!(out, "func @{name}(ptr) -> i64 {{\nentry ^0(%a: ptr):");
    out.push_str("  %d = load %a align 8 : ptr\n  %a1 = ptr_add %a, i64 8 : ptr\n  %s = load %a1 align 8 : ptr\n");
    out.push_str("  %bw = load %a1 align 8 : i64\n  %b = trunc %bw : i8\n");
    out.push_str("  %a2 = ptr_add %a, i64 16 : ptr\n  %nv = load %a2 align 8 : i64\n");
    let len = n.map_or("%nv".to_string(), |n| format!("i64 {n}"));
    match op {
        "memset" => {
            let _ = writeln!(out, "  memset %d, %b, {len} align {align}");
        }
        "memset0" => {
            let _ = writeln!(out, "  memset %d, i8 0, {len} align {align}");
        }
        _ => {
            let _ = writeln!(out, "  {op} %d, %s, {len} align {align}");
        }
    }
    out.push_str("  ret i64 0\n}\n\n");
}

/// Every constant length 0..=MAX and a variable one, for `op` at `align`.
fn module_for(op: &str, align: u32) -> String {
    let mut src = String::from("module \"bulk\"\n\n");
    gen_fn(&mut src, "var", op, align, None);
    for n in 0..=MAX {
        gen_fn(&mut src, &format!("c{n}"), op, align, Some(n));
    }
    src
}

/// Call `f(args)` where `args = [dst, src_or_byte, n]`.
fn call(f: &dyn Fn(i64) -> i64, dst: *mut u8, mid: u64, n: usize) {
    let args: [u64; 3] = [dst as u64, mid, n as u64];
    assert_eq!(f(args.as_ptr() as i64), 0);
}

/// Run `op` at `align` for every length and the offsets that respect it,
/// comparing with `model` over two guarded buffers.
fn exhaustive(op: &str, align: u32) {
    let (m, syms) = parse(&module_for(op, align));
    let cm = Jit::new().compile(&m, &syms).expect("jit");
    let offs: Vec<usize> = if align == 1 { vec![0, 1, 3, 7, 8, 13] } else { vec![0, align as usize, 2 * align as usize] };
    // Buffers aligned to 64 bytes by over-allocating u128s.
    const LEN: usize = MAX + 128;
    let mut dst_store = vec![0u128; LEN / 16 + 8];
    let mut src_store = vec![0u128; LEN / 16 + 8];
    let pattern = |i: usize| (i as u8).wrapping_mul(37).wrapping_add(11);
    for n in 0..=MAX {
        for f_name in ["var".to_string(), format!("c{n}")] {
            let f = cm.get_fn_i64_i64(&f_name).expect("compiled");
            for &doff in &offs {
                for &soff in &offs {
                    let dst = &mut bytemuck_u8(&mut dst_store)[..LEN];
                    for (i, b) in dst.iter_mut().enumerate() {
                        *b = !pattern(i);
                    }
                    let mut want = dst.to_vec();
                    let src = &mut bytemuck_u8(&mut src_store)[..LEN];
                    for (i, b) in src.iter_mut().enumerate() {
                        *b = pattern(i);
                    }
                    let src = src.to_vec();
                    let dptr = bytemuck_u8(&mut dst_store).as_mut_ptr();
                    let sptr = bytemuck_u8(&mut src_store).as_mut_ptr();
                    match op {
                        "memcpy" => {
                            want[doff..doff + n].copy_from_slice(&src[soff..soff + n]);
                            call(&f, unsafe_add(dptr, doff), unsafe_add(sptr, soff) as u64, n);
                        }
                        "memmove" => {
                            // Within the destination buffer: overlap in both
                            // directions.
                            want.copy_within(soff + 8..soff + 8 + n, doff);
                            call(&f, unsafe_add(dptr, doff), unsafe_add(dptr, soff + 8) as u64, n);
                        }
                        "memset" => {
                            want[doff..doff + n].fill(0xA5);
                            call(&f, unsafe_add(dptr, doff), 0xA5, n);
                        }
                        _ => {
                            want[doff..doff + n].fill(0);
                            call(&f, unsafe_add(dptr, doff), 0x77, n);
                        }
                    }
                    let got = &bytemuck_u8(&mut dst_store)[..LEN];
                    assert!(got == want.as_slice(), "{op} align {align} @{f_name} n {n} doff {doff} soff {soff}");
                }
            }
        }
    }
}

/// The bytes of a `u128` buffer (16-byte aligned storage).
fn bytemuck_u8(v: &mut [u128]) -> &mut [u8] {
    let len = v.len() * 16;
    // SAFETY: a `[u128]` is `len` initialized bytes with no padding, `u8` has
    // alignment 1, and the returned slice borrows `v` mutably.
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), len)
    }
}

/// `p + off` as an address (only passed to the JIT-compiled code).
fn unsafe_add(p: *mut u8, off: usize) -> *mut u8 {
    p.wrapping_add(off)
}

#[test]
fn memcpy_every_length_unaligned() {
    exhaustive("memcpy", 1);
}

#[test]
fn memcpy_every_length_aligned() {
    exhaustive("memcpy", 8);
    exhaustive("memcpy", 16);
}

#[test]
fn memmove_every_length_both_directions() {
    exhaustive("memmove", 1);
    exhaustive("memmove", 4);
}

#[test]
fn memset_every_length() {
    exhaustive("memset", 1);
    exhaustive("memset0", 8);
    exhaustive("memset", 16);
}

/// The bytes of the defined function `name` in `obj`'s `.text`.
fn func_bytes<'a>(obj: &'a ObjectModule, name: &str) -> &'a [u8] {
    let sym = obj.symbols().iter().find(|s| s.name == name).expect("function symbol");
    let SymbolValue::Defined { section, offset } = sym.value else { panic!("{name} undefined") };
    &obj.section(section).bytes[offset as usize..(offset + sym.size) as usize]
}

/// Intel-syntax disassembly of `bytes`, one instruction per line.
fn disasm(bytes: &[u8]) -> String {
    let opts = Options { syntax: Syntax::Intel };
    let mut out = String::new();
    let mut at = 0;
    while at < bytes.len() {
        let i = decode(TargetArch::X86_64, &bytes[at..], at as u64, &opts);
        assert!(i.known, "undecodable bytes at {at}: {:02x?}", &bytes[at..]);
        out.push_str(&i.text().replace('\t', " "));
        out.push('\n');
        at += i.len;
    }
    out
}

#[test]
fn instruction_shapes() {
    let src = r#"module "s"
func @big(ptr, ptr) -> void {
entry ^0(%d: ptr, %s: ptr):
  memcpy %d, %s, i64 4096 align 8
  ret
}
func @var(ptr, ptr, i64) -> void {
entry ^0(%d: ptr, %s: ptr, %n: i64):
  memcpy %d, %s, %n align 1
  ret
}
func @fill(ptr, i8, i32) -> void {
entry ^0(%d: ptr, %b: i8, %n: i32):
  memset %d, %b, %n align 1
  ret
}
func @move(ptr, ptr, i64) -> void {
entry ^0(%d: ptr, %s: ptr, %n: i64):
  memmove %d, %s, %n align 1
  ret
}
func @small(ptr, ptr) -> void {
entry ^0(%d: ptr, %s: ptr):
  memcpy %d, %s, i64 40 align 16
  ret
}
func @zero16(ptr) -> void {
entry ^0(%d: ptr):
  memset %d, i8 0, i64 16 align 1
  ret
}
"#;
    let (m, syms) = parse(src);
    let obj = super::compile_module(&m, &syms);
    let big = disasm(func_bytes(&obj, "big"));
    assert!(big.contains("rep movsb") && big.contains("mov ecx, 0x1000") || big.contains("mov rcx, 0x1000"), "{big}");
    let var = disasm(func_bytes(&obj, "var"));
    assert!(var.contains("rep movsb"), "{var}");
    let fill = disasm(func_bytes(&obj, "fill"));
    assert!(fill.contains("rep stosb"), "{fill}");
    let mv = disasm(func_bytes(&obj, "move"));
    assert!(mv.contains("std") && mv.contains("cld") && mv.matches("rep movsb").count() == 2, "{mv}");
    // 40 bytes at 16-byte alignment: two aligned SSE moves and one 8-byte one.
    let small = disasm(func_bytes(&obj, "small"));
    assert!(!small.contains("rep"), "{small}");
    assert_eq!(small.matches("movdqa").count(), 4, "{small}");
    let zero16 = disasm(func_bytes(&obj, "zero16"));
    assert!(zero16.contains("movdqu") && !zero16.contains("rep"), "{zero16}");
}

/// The libc path: with the option on, long and variable ops call `memcpy`,
/// `memmove` and `memset` (resolved by gcc's libc), short constant ones stay
/// inline; the program checks the results itself.
#[test]
fn libc_calls_when_hosted() {
    use std::process::Command;
    if !Command::new("gcc").arg("--version").output().is_ok_and(|o| o.status.success()) {
        return;
    }
    let src = r#"module "h"
func @lf_copy(ptr, ptr, i64) -> void {
entry ^0(%d: ptr, %s: ptr, %n: i64):
  memcpy %d, %s, %n align 1
  ret
}
func @lf_move(ptr, ptr, i64) -> void {
entry ^0(%d: ptr, %s: ptr, %n: i64):
  memmove %d, %s, %n align 1
  ret
}
func @lf_fill(ptr, i8, i64) -> void {
entry ^0(%d: ptr, %b: i8, %n: i64):
  memset %d, %b, %n align 1
  ret
}
func @lf_small(ptr, ptr) -> void {
entry ^0(%d: ptr, %s: ptr):
  memcpy %d, %s, i64 16 align 1
  ret
}
"#;
    let (m, syms) = parse(src);
    let opts = CodegenOptions::default().with_bulk_memory_libcalls(true);
    let obj = super::compile_module_with(&m, &syms, &opts).object;
    let undefined: Vec<&str> = obj
        .symbols()
        .iter()
        .filter(|s| matches!(s.value, SymbolValue::Undefined))
        .map(|s| s.name.as_str())
        .collect();
    for want in ["memcpy", "memmove", "memset"] {
        assert!(undefined.contains(&want), "{want} is called: {undefined:?}");
    }
    for f in ["lf_copy", "lf_move", "lf_fill", "lf_small"] {
        assert!(!disasm(func_bytes(&obj, f)).contains("rep"), "@{f} uses no rep");
    }
    let dir = std::env::temp_dir().join(format!("lf-bulk-libc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let o = dir.join("lf.o");
    std::fs::write(&o, crate::mc::elf::write(&obj)).unwrap();
    let c = dir.join("main.c");
    std::fs::write(
        &c,
        r#"#include <string.h>
#include <stdio.h>
void lf_copy(void *, const void *, long);
void lf_move(void *, const void *, long);
void lf_fill(void *, char, long);
void lf_small(void *, const void *);
int main(void) {
  unsigned char a[600], b[600], w[600];
  for (int n = 0; n <= 300; n++) {
    for (int i = 0; i < 600; i++) { a[i] = (unsigned char)(i * 7); b[i] = (unsigned char)(i * 13 + 1); }
    memcpy(w, a, 600); memcpy(w + 3, b + 5, n);
    lf_copy(a + 3, b + 5, n);
    if (memcmp(a, w, 600)) { printf("copy %d\n", n); return 1; }
    memcpy(w, a, 600); memmove(w + 10, w + 2, n);
    lf_move(a + 10, a + 2, n);
    if (memcmp(a, w, 600)) { printf("move up %d\n", n); return 1; }
    memcpy(w, a, 600); memmove(w + 2, w + 10, n);
    lf_move(a + 2, a + 10, n);
    if (memcmp(a, w, 600)) { printf("move down %d\n", n); return 1; }
    memcpy(w, a, 600); memset(w + 1, 0x5a, n);
    lf_fill(a + 1, 0x5a, n);
    if (memcmp(a, w, 600)) { printf("fill %d\n", n); return 1; }
  }
  memcpy(w, a, 600); memcpy(w + 100, b, 16); lf_small(a + 100, b);
  if (memcmp(a, w, 600)) { printf("small\n"); return 1; }
  printf("ok\n");
  return 0;
}
"#,
    )
    .unwrap();
    let exe = dir.join("main");
    let out = Command::new("gcc").arg("-O1").arg(&c).arg(&o).arg("-o").arg(&exe).output().unwrap();
    assert!(out.status.success(), "gcc: {}", String::from_utf8_lossy(&out.stderr));
    let out = loop {
        match Command::new(&exe).output() {
            Ok(o) => break o,
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
            Err(e) => panic!("exec: {e}"),
        }
    };
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ok\n");
}

/// Lode's two size problems (issue #21), before and after: a buffered
/// `print` zero-filling a 256-byte stack buffer, and the buffer's `put`
/// copying `n` bytes in. "Before" is what Lode emits without the op (the
/// fill unrolled into word stores up to 16, a loop above; the copy a byte
/// loop); "after" is one `memset` / `memcpy`.
#[test]
fn lode_repros_shrink() {
    const BEFORE: &str = r#"module "before"
func @print_buf(ptr) -> i64 {
entry ^0(%out: ptr):
  %buf = alloca [256 x i8] : ptr
  br ^1(i64 0)
^1(%i: i64):
  %more = icmp ult %i, i64 32 : i1
  cond_br %more, ^2, ^3
^2:
  %o = mul %i, i64 8 : i64
  %p = ptr_add %buf, %o : ptr
  store i64 0, %p align 8 : i64
  %i2 = add %i, i64 1 : i64
  br ^1(%i2)
^3:
  %r = call @sink(%buf) : i64
  ret %r
}
func @put(ptr, ptr, i64, i64) -> i64 {
entry ^0(%buf: ptr, %src: ptr, %len: i64, %n: i64):
  br ^1(i64 0)
^1(%i: i64):
  %more = icmp ult %i, %n : i1
  cond_br %more, ^2, ^3
^2:
  %s = ptr_add %src, %i : ptr
  %v = load %s align 1 : i8
  %at = add %len, %i : i64
  %d = ptr_add %buf, %at : ptr
  store %v, %d align 1 : i8
  %i2 = add %i, i64 1 : i64
  br ^1(%i2)
^3:
  %new = add %len, %n : i64
  ret %new
}
func @sink(ptr) -> i64
"#;
    const AFTER: &str = r#"module "after"
func @print_buf(ptr) -> i64 {
entry ^0(%out: ptr):
  %buf = alloca [256 x i8] : ptr
  memset %buf, i8 0, i64 256 align 8
  %r = call @sink(%buf) : i64
  ret %r
}
func @put(ptr, ptr, i64, i64) -> i64 {
entry ^0(%buf: ptr, %src: ptr, %len: i64, %n: i64):
  %d = ptr_add %buf, %len : ptr
  memcpy %d, %src, %n align 1
  %new = add %len, %n : i64
  ret %new
}
func @sink(ptr) -> i64
"#;
    let size = |src: &str, f: &str| {
        let mut syms = StrInterner::new();
        let mut m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms).unwrap();
        optimize(&mut m, OptLevel::O2);
        let obj = super::compile_module(&m, &syms);
        let b = func_bytes(&obj, f);
        (b.len(), disasm(b))
    };
    for f in ["print_buf", "put"] {
        let (before, bl) = size(BEFORE, f);
        let (after, al) = size(AFTER, f);
        eprintln!("lode repro @{f}: {before} -> {after} bytes\n{al}");
        assert!(after < before, "@{f}: {before} -> {after}\nbefore:\n{bl}\nafter:\n{al}");
    }
    let (fill, text) = size(AFTER, "print_buf");
    assert!(text.contains("rep stosb") && fill <= 48, "{fill} bytes:\n{text}");
    let (put, text) = size(AFTER, "put");
    assert!(text.contains("rep movsb") && put <= 32, "{put} bytes:\n{text}");
}
