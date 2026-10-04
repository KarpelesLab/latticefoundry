//! WebAssembly decoder tests: round trips against the encoder, and llvm-objdump
//! differential tests.
//!
//! - **Golden texts** for every immediate form.
//! - **Round trip over compiled code**: the IR corpora compiled to wasm
//!   objects (padded 5-byte LEBs at every relocation) and to linked modules
//!   (shortest LEBs); every function body decodes completely, with no
//!   unknown encoding, ending exactly at the body's end, and re-encoding the
//!   typed instructions reproduces the bytes exactly.
//! - **Fuzzed round trip**: random well-formed instructions of every opcode
//!   the decoder knows, with random immediates and LEB widths, encode →
//!   decode → the same typed instruction.
//! - **Differential**: every known opcode against `llvm-mc --disassemble`,
//!   and the compiled objects against `llvm-objdump -d` (skipped when LLVM is
//!   not installed).

use std::process::Command;

use super::corpus::{FLOATS, INTS};
use super::{Rng, assert_clean, differential, llvm_tool, object_file, parse_for};
use crate::codegen::CodegenOptions;
use crate::mc::disasm::objfile;
use crate::mc::disasm::wasm::{BlockType, Imm, Kind, WasmInst, decode_inst, info};
use crate::mc::disasm::{Options, decode};
use crate::target::wasm32::binary::LinkOptions;
use crate::target::{ObjectFormat, TargetArch};

/// Atomics, 64-bit arithmetic, narrow memory accesses, a sparse switch on
/// `i64` and indirect calls.
const EXTRA: &str = r#"
module "extra"

global @w32 : i32 = i32 100
global @w8 : i8 = i8 -3
global @w64 : i64 = i64 -5
global @bytes : [8 x i8] = [8 x i8] "abcdefgh"

func @atomics(i32, i64) -> i64 {
entry ^0(%v: i32, %w: i64):
  %a = atomic_rmw add seq_cst @w32, %v align 4 : i32
  %b = atomic_rmw xchg seq_cst @w64, %w align 8 : i64
  %c = atomic_rmw or seq_cst @w8, i8 1 align 1 : i8
  %d = cmpxchg seq_cst seq_cst @w32, %a, %v align 4 : i32
  fence seq_cst
  %e = atomic_load seq_cst @w64 align 8 : i64
  atomic_store seq_cst %v, @w32 align 4 : i32
  %ax = zext %a : i64
  %cx = sext %c : i64
  %dx = zext %d : i64
  %s1 = add %ax, %b : i64
  %s2 = add %s1, %cx : i64
  %s3 = add %s2, %dx : i64
  %s4 = add %s3, %e : i64
  ret %s4
}

func @narrow_mem(i32) -> i64 {
entry ^0(%i: i32):
  %p = ptr_add @bytes, %i : ptr
  %b = load %p align 1 : i8
  %q = ptr_add @bytes, i32 2 : ptr
  %h = load %q align 2 : i16
  store i16 7, %q align 2 : i16
  store %b, %p align 1 : i8
  %bx = sext %b : i64
  %hx = zext %h : i64
  %r = mul %bx, %hx : i64
  %rr = lshr %r, i64 3 : i64
  %rt = ashr %rr, i64 1 : i64
  ret %rt
}

func @sparse(i64) -> i32 {
entry ^0(%x: i64):
  switch %x, ^4 [1000000: ^1, -5: ^2, 81985529216486895: ^3, 7: ^1]
^1:
  ret i32 1
^2:
  ret i32 2
^3:
  ret i32 3
^4:
  ret i32 0
}

func internal @inc(i64) -> i64 {
entry ^0(%x: i64):
  %r = add %x, i64 1 : i64
  ret %r
}

global constant @fns : [1 x ptr] = [1 x ptr] (ptr @inc)

func @indirect(i64) -> i64 {
entry ^0(%x: i64):
  %f = load @fns align 4 : ptr
  %r = call %f(%x) : i64
  ret %r
}

func @fsel(f64, f64, f32) -> f64 {
entry ^0(%a: f64, %b: f64, %c: f32):
  %lt = fcmp olt %a, %b : i1
  %m = select %lt, %a, %b : f64
  %n = fneg %m : f64
  %ce = fpext %c : f64
  %k = fmul %n, f64 0x4004000000000000 : f64
  %r = fadd %k, %ce : f64
  ret %r
}
"#;

const PROGRAMS: [(&str, &str); 3] = [("ints", INTS), ("floats", FLOATS), ("extra", EXTRA)];

/// Compile `src` for wasm32 (after `-O1`).
fn wasm_object(src: &str) -> crate::target::wasm32::binary::WasmObject {
    let (mut m, syms) = parse_for(TargetArch::Wasm32, src);
    crate::transform::pipeline::optimize(&mut m, crate::transform::pipeline::OptLevel::O1);
    crate::target::wasm32::compile(&m, &syms, &CodegenOptions::default()).expect("compile for wasm32").object
}

// ===========================================================================
// A test-only encoder from the typed form
// ===========================================================================

/// An unsigned LEB128 of exactly `width` bytes.
fn uleb(out: &mut Vec<u8>, v: u64, width: u8) {
    for i in 0..u32::from(width) {
        let group = if 7 * i < 64 { (v >> (7 * i)) & 0x7f } else { 0 };
        let more = if i + 1 < u32::from(width) { 0x80 } else { 0 };
        out.push(group as u8 | more);
    }
}

/// A signed LEB128 of exactly `width` bytes.
fn sleb(out: &mut Vec<u8>, v: i64, width: u8) {
    for i in 0..u32::from(width) {
        let group = (v >> (7 * i).min(63)) & 0x7f;
        let more = if i + 1 < u32::from(width) { 0x80 } else { 0 };
        out.push(group as u8 | more);
    }
}

/// The shortest LEB widths.
fn uleb_len(v: u64) -> u8 {
    let mut n = 1;
    let mut v = v >> 7;
    while v != 0 {
        n += 1;
        v >>= 7;
    }
    n
}

fn sleb_len(v: i64) -> u8 {
    let mut n = 1;
    let mut v = v;
    loop {
        let byte = v & 0x7f;
        v >>= 7;
        if (v == 0 && byte & 0x40 == 0) || (v == -1 && byte & 0x40 != 0) {
            return n;
        }
        n += 1;
    }
}

/// Encode `w`, with each LEB in the width recorded in `w.widths`.
fn encode(w: &WasmInst) -> Vec<u8> {
    let mut out = Vec::new();
    let mut widths = w.widths.iter().copied();
    let mut next = || widths.next().expect("a width per LEB");
    match w.prefix {
        Some(p) => {
            out.push(p);
            uleb(&mut out, u64::from(w.opcode), next());
        }
        None => out.push(w.opcode as u8),
    }
    match &w.imm {
        Imm::None => {}
        Imm::Block(BlockType::Empty) => out.push(0x40),
        Imm::Block(BlockType::Value(t)) => out.push(*t),
        Imm::Block(BlockType::Index(i)) => sleb(&mut out, *i as i64, next()),
        Imm::Index(i) => uleb(&mut out, u64::from(*i), next()),
        Imm::Index2(a, b) => {
            uleb(&mut out, u64::from(*a), next());
            uleb(&mut out, u64::from(*b), next());
        }
        Imm::BrTable { targets, default } => {
            uleb(&mut out, targets.len() as u64, next());
            for t in targets {
                uleb(&mut out, u64::from(*t), next());
            }
            uleb(&mut out, u64::from(*default), next());
        }
        Imm::I32(v) => sleb(&mut out, i64::from(*v), next()),
        Imm::I64(v) => sleb(&mut out, *v, next()),
        Imm::F32(b) => out.extend_from_slice(&b.to_le_bytes()),
        Imm::F64(b) => out.extend_from_slice(&b.to_le_bytes()),
        Imm::Mem { align, offset } => {
            uleb(&mut out, u64::from(*align), next());
            uleb(&mut out, *offset, next());
        }
        Imm::MemLane { align, offset, lane } => {
            uleb(&mut out, u64::from(*align), next());
            uleb(&mut out, *offset, next());
            out.push(*lane);
        }
        Imm::Lane(l) => out.push(*l),
        Imm::V128(b) | Imm::Shuffle(b) => out.extend_from_slice(b),
        Imm::RefType(t) => out.push(*t),
        Imm::Types(ts) => {
            uleb(&mut out, ts.len() as u64, next());
            out.extend_from_slice(ts);
        }
        Imm::Fence(b) => out.push(*b),
    }
    assert!(widths.next().is_none(), "unused widths in {w:?}");
    out
}

/// Every `(prefix, opcode)` the decoder knows.
fn known_opcodes() -> Vec<(Option<u8>, u32)> {
    let mut all = Vec::new();
    for op in 0..=255u32 {
        if info(None, op).is_some() {
            all.push((None, op));
        }
    }
    for p in [0xfcu8, 0xfd, 0xfe] {
        for op in 0..0x200u32 {
            if info(Some(p), op).is_some() {
                all.push((Some(p), op));
            }
        }
    }
    all
}

/// A random instruction of `(prefix, op)`: random immediates in random (at
/// least shortest, at most 5- or 10-byte) LEB widths.
fn random_inst(rng: &mut Rng, prefix: Option<u8>, op: u32) -> WasmInst {
    let (_, kind) = info(prefix, op).expect("known");
    let mut widths = Vec::new();
    let mut width = |rng: &mut Rng, min: u8, max: u8| -> u8 {
        let w = if rng.below(4) == 0 { min + rng.below(u64::from(max - min) + 1) as u8 } else { min };
        widths.push(w);
        w
    };
    if prefix.is_some() {
        width(rng, uleb_len(u64::from(op)), 5);
    }
    let idx = |rng: &mut Rng| -> u32 { if rng.below(3) == 0 { rng.next() as u32 } else { rng.below(40) as u32 } };
    let imm = match kind {
        Kind::None => Imm::None,
        Kind::Block => match rng.below(3) {
            0 => Imm::Block(BlockType::Empty),
            1 => Imm::Block(BlockType::Value([0x7f, 0x7e, 0x7d, 0x7c, 0x7b, 0x70, 0x6f][rng.below(7) as usize])),
            _ => {
                let i = rng.below(1 << 20);
                width(rng, sleb_len(i as i64), 5);
                Imm::Block(BlockType::Index(i))
            }
        },
        Kind::Index => {
            let i = idx(rng);
            width(rng, uleb_len(u64::from(i)), 5);
            Imm::Index(i)
        }
        Kind::Index2 => {
            let (a, b) = (idx(rng), idx(rng));
            width(rng, uleb_len(u64::from(a)), 5);
            width(rng, uleb_len(u64::from(b)), 5);
            Imm::Index2(a, b)
        }
        Kind::BrTable => {
            let n = rng.below(6);
            width(rng, uleb_len(n), 5);
            let targets: Vec<u32> = (0..n).map(|_| idx(rng)).collect();
            for t in &targets {
                width(rng, uleb_len(u64::from(*t)), 5);
            }
            let default = idx(rng);
            width(rng, uleb_len(u64::from(default)), 5);
            Imm::BrTable { targets, default }
        }
        Kind::I32 => {
            let v = rng.next() as i32 >> rng.below(32);
            width(rng, sleb_len(i64::from(v)), 5);
            Imm::I32(v)
        }
        Kind::I64 => {
            let v = rng.next() as i64 >> rng.below(64);
            width(rng, sleb_len(v), 10);
            Imm::I64(v)
        }
        Kind::F32 => Imm::F32(rng.next() as u32),
        Kind::F64 => Imm::F64(rng.next()),
        Kind::Mem(_) | Kind::MemLane(_) => {
            let align = rng.below(5) as u32;
            let offset = rng.next() >> rng.below(64);
            width(rng, uleb_len(u64::from(align)), 5);
            width(rng, uleb_len(offset), 10);
            if matches!(kind, Kind::Mem(_)) {
                Imm::Mem { align, offset }
            } else {
                Imm::MemLane { align, offset, lane: rng.below(16) as u8 }
            }
        }
        Kind::Lane => Imm::Lane(rng.below(16) as u8),
        Kind::V128 => Imm::V128(rng.next().to_le_bytes().repeat(2).try_into().unwrap()),
        Kind::Shuffle => Imm::Shuffle(core::array::from_fn(|_| rng.below(32) as u8)),
        Kind::RefType => Imm::RefType(if rng.below(2) == 0 { 0x70 } else { 0x6f }),
        Kind::Types => {
            let n = rng.below(3);
            width(rng, uleb_len(n), 5);
            Imm::Types((0..n).map(|_| [0x7f, 0x7e, 0x7d, 0x7c][rng.below(4) as usize]).collect())
        }
        Kind::Fence => Imm::Fence(0),
    };
    WasmInst { prefix, opcode: op, imm, widths }
}

// ===========================================================================
// Golden texts
// ===========================================================================

fn text(bytes: &[u8]) -> String {
    let i = decode(TargetArch::Wasm32, bytes, 0, &Options::default());
    assert_eq!(i.len, bytes.len(), "{bytes:02x?} decodes as `{}`", i.text());
    i.text()
}

#[test]
fn golden() {
    assert_eq!(text(&[0x41, 0xc7, 0x9f, 0x7f]), "i32.const\t-12345");
    assert_eq!(text(&[0x42, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f]), "i64.const\t-9223372036854775808");
    assert_eq!(text(&[0x20, 0x03]), "local.get\t3");
    assert_eq!(text(&[0x10, 0x80, 0x80, 0x80, 0x80, 0x00]), "call\t0");
    assert_eq!(text(&[0x11, 0x02, 0x00]), "call_indirect\t2");
    assert_eq!(text(&[0x11, 0x02, 0x01]), "call_indirect\t2, 1");
    assert_eq!(text(&[0x02, 0x40]), "block");
    assert_eq!(text(&[0x03, 0x7e]), "loop\ti64");
    assert_eq!(text(&[0x04, 0x05]), "if\ttype[5]");
    assert_eq!(text(&[0x0e, 0x02, 0x01, 0x00, 0x03]), "br_table\t{1, 0, 3}");
    assert_eq!(text(&[0x29, 0x03, 0x10]), "i64.load\t16");
    assert_eq!(text(&[0x28, 0x00, 0x04]), "i32.load\t4:p2align=0");
    assert_eq!(text(&[0x3a, 0x00, 0x07]), "i32.store8\t7");
    assert_eq!(text(&[0x43, 0x00, 0x00, 0xc0, 0x3f]), "f32.const\t0x1.8p0");
    assert_eq!(text(&[0x43, 0x00, 0x00, 0x80, 0xbf]), "f32.const\t-0x1p0");
    assert_eq!(text(&[0x43, 0x01, 0x00, 0x00, 0x00]), "f32.const\t0x1p-149");
    assert_eq!(text(&[0x43, 0x00, 0x00, 0xc0, 0x7f]), "f32.const\tnan");
    assert_eq!(text(&[0x43, 0x00, 0x00, 0x80, 0xff]), "f32.const\t-infinity");
    assert_eq!(text(&[0x44, 0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xb9, 0x3f]), "f64.const\t0x1.999999999999ap-4");
    assert_eq!(text(&[0x44, 0x01, 0, 0, 0, 0, 0, 0, 0]), "f64.const\t0x0.0000000000001p-1022");
    assert_eq!(text(&[0x44, 0, 0, 0, 0, 0, 0, 0, 0x80]), "f64.const\t-0x0p0");
    assert_eq!(text(&[0xfc, 0x06]), "i64.trunc_sat_f64_s");
    assert_eq!(text(&[0xfc, 0x0a, 0x00, 0x00]), "memory.copy\t0, 0");
    assert_eq!(text(&[0xfc, 0x08, 0x01, 0x00]), "memory.init\t1, 0");
    assert_eq!(text(&[0xfe, 0x03, 0x00]), "atomic.fence");
    assert_eq!(text(&[0xfe, 0x1e, 0x02, 0x08]), "i32.atomic.rmw.add\t8");
    assert_eq!(text(&[0xfe, 0x4e, 0x03, 0x00]), "i64.atomic.rmw32.cmpxchg_u\t0:p2align=3");
    assert_eq!(text(&[0xfe, 0x00, 0x02, 0x00]), "memory.atomic.notify\t0");
    assert_eq!(text(&[0xd0, 0x70]), "ref.null_func");
    assert_eq!(text(&[0x1c, 0x01, 0x7f]), "select\ti32");
    assert_eq!(text(&[0xfd, 0x0f]), "i8x16.splat");
    assert_eq!(text(&[0xfd, 0x15, 0x03]), "i8x16.extract_lane_s\t3");
    assert_eq!(text(&[0xfd, 0x54, 0x00, 0x00, 0x01]), "v128.load8_lane\t0, 1");
    assert_eq!(text(&[0xfd, 0xae, 0x01]), "i32x4.add");
    let mut c = vec![0xfd, 0x0c];
    c.extend(0..16u8);
    assert_eq!(text(&c), "v128.const\t50462976, 117835012, 185207048, 252579084");
    // Unknown and truncated encodings are single bytes of data.
    for bad in [&[0xffu8][..], &[0x41, 0x80], &[0xfc, 0x7f], &[0x0e, 0x05, 0x00], &[0xfd, 0x9a, 0x01], &[0x43, 0, 0]] {
        let i = decode(TargetArch::Wasm32, bad, 0, &Options::default());
        assert!(!i.known && i.len == 1, "{bad:02x?}: {}", i.text());
    }
}

// ===========================================================================
// Round trips
// ===========================================================================

/// Decode `bytes` completely, re-encode every instruction and compare.
/// Returns the instruction count.
fn round_trip_region(what: &str, bytes: &[u8]) -> usize {
    let mut at = 0;
    let mut n = 0;
    while at < bytes.len() {
        let (inst, len) = decode_inst(&bytes[at..])
            .unwrap_or_else(|| panic!("{what}: unknown encoding at {at}: {:02x?}", &bytes[at..bytes.len().min(at + 8)]));
        assert!(at + len <= bytes.len());
        assert_eq!(encode(&inst), &bytes[at..at + len], "{what}: {inst:?} re-encodes differently");
        at += len;
        n += 1;
    }
    assert_eq!(at, bytes.len(), "{what}: decoding ends exactly at the body's end");
    assert_eq!(bytes.last(), Some(&0x0b), "{what}: a body ends with `end`");
    n
}

#[test]
fn round_trip_compiled_code() {
    let mut total = 0;
    let mut bodies = 0;
    for (name, src) in PROGRAMS {
        let obj = wasm_object(src);
        for (kind, file) in [("object", obj.to_relocatable()), ("linked", obj.to_linked(&LinkOptions::default()).expect("link"))] {
            let bin = objfile::read(&file).expect("read wasm");
            let code = bin.sections.iter().find(|s| s.name == "CODE").expect("a code section");
            assert!(!code.regions.is_empty());
            for r in &code.regions {
                let what = format!("{name} {kind} @{:#x}", r.start);
                total += round_trip_region(&what, &code.bytes[r.start as usize..r.end as usize]);
                bodies += 1;
            }
        }
    }
    eprintln!("wasm round trip: {total} instructions in {bodies} function bodies re-encode exactly");
    assert!(total > 500);
}

#[test]
fn round_trip_fuzzed() {
    let ops = known_opcodes();
    let mut rng = Rng(0x7a5d_1234_9876_0001);
    let mut n = 0;
    for _ in 0..30 {
        for &(p, op) in &ops {
            let inst = random_inst(&mut rng, p, op);
            let bytes = encode(&inst);
            let (back, len) = decode_inst(&bytes).unwrap_or_else(|| panic!("{inst:?} ({bytes:02x?}) does not decode"));
            assert_eq!(len, bytes.len(), "{inst:?}");
            assert_eq!(back, inst, "{bytes:02x?}");
            let text = decode(TargetArch::Wasm32, &bytes, 0, &Options::default());
            assert!(text.known && text.len == bytes.len());
            n += 1;
        }
    }
    // Random sequences decode back into the same instruction stream.
    for _ in 0..200 {
        let seq: Vec<WasmInst> = (0..20)
            .map(|_| {
                let (p, op) = ops[rng.below(ops.len() as u64) as usize];
                random_inst(&mut rng, p, op)
            })
            .collect();
        let bytes: Vec<u8> = seq.iter().flat_map(encode).collect();
        let mut at = 0;
        for want in &seq {
            let (got, len) = decode_inst(&bytes[at..]).expect("decodes");
            assert_eq!(&got, want);
            at += len;
        }
        assert_eq!(at, bytes.len());
    }
    eprintln!("wasm fuzzed round trip: {n} instructions over {} opcodes", ops.len());
}

// ===========================================================================
// Differential tests
// ===========================================================================

/// One canonical instance of every known opcode, disassembled by `llvm-mc`
/// and compared with ours. Encodings llvm-mc rejects (proposals it does not
/// enable or know, like typed `select`) are skipped and counted.
#[test]
fn every_opcode_matches_llvm_mc() {
    let Some(mc) = llvm_tool("llvm-mc") else {
        eprintln!("skipping every_opcode_matches_llvm_mc: no llvm-mc");
        return;
    };
    let mut rng = Rng(5);
    let insts: Vec<(Vec<u8>, String)> = known_opcodes()
        .into_iter()
        .map(|(p, op)| {
            let mut w = random_inst(&mut rng, p, op);
            // Shortest LEBs, small indices (llvm-mc's own limits).
            w = canonical(w);
            let bytes = encode(&w);
            let ours = decode(TargetArch::Wasm32, &bytes, 0, &Options::default()).text();
            (bytes, ours)
        })
        .collect();
    // One llvm-mc run per instruction (a rejected encoding makes llvm-mc
    // resume decoding inside it, possibly running into the next line), in
    // parallel.
    let run = |bytes: &[u8]| -> Option<String> {
        use std::io::Write;
        let mut child = Command::new(&mc)
            .args(["--disassemble", "-triple=wasm32"])
            .arg("-mattr=+simd128,+atomics,+bulk-memory,+reference-types,+tail-call,+exception-handling,+sign-ext,+nontrapping-fptoint,+multivalue")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("run llvm-mc");
        let line = bytes.iter().map(|x| format!("{x:#04x}")).collect::<Vec<_>>().join(" ") + "\n";
        child.stdin.take().expect("stdin").write_all(line.as_bytes()).expect("write");
        let out = child.wait_with_output().expect("llvm-mc output");
        if String::from_utf8_lossy(&out.stderr).contains("invalid instruction encoding") {
            return None;
        }
        let lines: Vec<String> =
            String::from_utf8_lossy(&out.stdout).lines().filter(|l| l.starts_with('\t')).map(str::to_owned).collect();
        Some(lines.join("\n"))
    };
    let theirs: Vec<Option<String>> = std::thread::scope(|sc| {
        let handles: Vec<_> =
            insts.chunks(insts.len().div_ceil(16)).map(|c| sc.spawn(move || c.iter().map(|(b, _)| run(b)).collect::<Vec<_>>())).collect();
        handles.into_iter().flat_map(|h| h.join().expect("llvm-mc thread")).collect()
    });
    let (mut compared, mut skipped, mut bad) = (0, 0, Vec::new());
    for ((bytes, ours), t) in insts.iter().zip(&theirs) {
        let Some(t) = t else {
            skipped += 1;
            continue;
        };
        compared += 1;
        let a = fixup(super::normalize(TargetArch::Wasm32, ours));
        let b = fixup(super::normalize(TargetArch::Wasm32, t));
        if a != b {
            bad.push(format!("{bytes:02x?}: ours `{ours}` | llvm-mc `{}`", t.trim()));
        }
    }
    eprintln!("wasm vs llvm-mc: {compared} opcodes compared, {skipped} rejected by llvm-mc, {} mismatches", bad.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    assert!(compared > 400, "{compared}");
}

/// `w` with the shortest LEBs and small indices.
fn canonical(mut w: WasmInst) -> WasmInst {
    match &mut w.imm {
        Imm::Index(i) => *i %= 4,
        Imm::Index2(a, b) => {
            *a %= 4;
            *b = 0;
        }
        Imm::BrTable { targets, default } => {
            targets.iter_mut().for_each(|t| *t %= 4);
            *default %= 4;
        }
        Imm::Block(b) => *b = BlockType::Empty,
        Imm::Mem { align, offset } | Imm::MemLane { align, offset, .. } => {
            *align %= 4;
            *offset %= 1000;
        }
        Imm::Shuffle(l) => l.iter_mut().for_each(|x| *x %= 16),
        Imm::Lane(l) => *l %= 2,
        _ => {}
    }
    // Recompute the shortest widths.
    let mut widths = Vec::new();
    if w.prefix.is_some() {
        widths.push(uleb_len(u64::from(w.opcode)));
    }
    match &w.imm {
        Imm::Index(i) => widths.push(uleb_len(u64::from(*i))),
        Imm::Index2(a, b) => widths.extend([uleb_len(u64::from(*a)), uleb_len(u64::from(*b))]),
        Imm::BrTable { targets, default } => {
            widths.push(uleb_len(targets.len() as u64));
            widths.extend(targets.iter().map(|t| uleb_len(u64::from(*t))));
            widths.push(uleb_len(u64::from(*default)));
        }
        Imm::I32(v) => widths.push(sleb_len(i64::from(*v))),
        Imm::I64(v) => widths.push(sleb_len(*v)),
        Imm::Mem { align, offset } | Imm::MemLane { align, offset, .. } => {
            widths.extend([uleb_len(u64::from(*align)), uleb_len(*offset)]);
        }
        Imm::Types(ts) => widths.push(uleb_len(ts.len() as u64)),
        _ => {}
    }
    w.widths = widths;
    w
}

/// Spelling differences from LLVM that are not decoding errors:
///
/// - LLVM annotates structured control flow with `#` comments
///   (`# label0:`, `# 1: up to label0`), dropped here;
/// - LLVM prints the untyped `select` (`0x1b`) as `f32.select` (its
///   instruction selection keeps one typed record per operand type and the
///   disassembler cannot tell them apart); the instruction is `select`;
/// - LLVM keeps the pre-standard names of the SIMD extending loads
///   (`i16x8.load8x8_s`); the specification names them `v128.load8x8_s`.
fn fixup(s: String) -> String {
    let s = match s.find('#') {
        Some(i) => s[..i].trim_end().to_owned(),
        None => s,
    };
    if s == "f32.select" {
        return "select".to_owned();
    }
    for (old, new) in [("i16x8.load8x8", "v128.load8x8"), ("i32x4.load16x4", "v128.load16x4"), ("i64x2.load32x2", "v128.load32x2")] {
        if let Some(rest) = s.strip_prefix(old) {
            return format!("{new}{rest}");
        }
    }
    s
}

#[test]
fn objects_match_llvm_objdump() {
    let mut total = 0;
    for (name, src) in PROGRAMS {
        let obj = super::compile(TargetArch::Wasm32, src);
        let file = object_file(TargetArch::Wasm32, &obj, ObjectFormat::Wasm);
        let Some(report) = differential(TargetArch::Wasm32, &file, &[], &Options::default(), &fixup) else {
            eprintln!("skipping objects_match_llvm_objdump: no llvm-objdump");
            return;
        };
        assert_clean(&format!("wasm32 {name}"), &report);
        total += report.compared;
    }
    eprintln!("wasm32 objects vs llvm-objdump: {total} instructions agree");
}
