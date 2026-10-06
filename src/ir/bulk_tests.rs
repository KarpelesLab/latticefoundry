//! Tests for the bulk-memory ops `memcpy` / `memmove` / `memset`
//! (`docs/ir-design.md` §6k): the reference semantics (overlap, zero length,
//! poison, alignment, bounds), text and binary round trips, the builder and
//! the verifier.

use std::collections::HashMap;

use puremp::Int;

use crate::ir::inst::{Flags, InstKind};
use crate::ir::refexec::{ExecError, run_named};
use crate::ir::semantics::{ByteMemory, SemValue, exec_bulk_memory};
use crate::ir::{Function, Module};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

/// A sparse byte memory: absent = not allocated, `None` = poison.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct Mem(HashMap<u64, Option<u8>>);

impl Mem {
    /// Allocate `[base, base + len)` holding `0, 1, 2, ...` (mod 256).
    fn with(regions: &[(u64, u64)]) -> Mem {
        let mut m = Mem::default();
        for &(base, len) in regions {
            for i in 0..len {
                m.0.insert(base + i, Some(i as u8));
            }
        }
        m
    }

    fn bytes(&self, base: u64, len: u64) -> Vec<Option<u8>> {
        (0..len).map(|i| self.0.get(&(base + i)).copied().flatten()).collect()
    }
}

impl ByteMemory for Mem {
    fn read_byte(&self, addr: u64) -> Option<Option<u8>> {
        self.0.get(&addr).copied()
    }

    fn write_byte(&mut self, addr: u64, byte: Option<u8>) {
        self.0.insert(addr, byte);
    }
}

const MEMCPY: InstKind = InstKind::MemCopy { align: 1, volatile: false, overlapping: false };
const MEMMOVE: InstKind = InstKind::MemCopy { align: 1, volatile: false, overlapping: true };
const MEMSET: InstKind = InstKind::MemSet { align: 1, volatile: false };

fn p(a: u64) -> SemValue {
    SemValue::ptr(Int::from_u64(a))
}

fn n64(n: u64) -> SemValue {
    SemValue::int(64, Int::from_u64(n))
}

fn byte(b: u8) -> SemValue {
    SemValue::int(8, Int::from_u64(u64::from(b)))
}

#[test]
fn memcpy_copies_and_rejects_partial_overlap() {
    let mut m = Mem::with(&[(0x100, 16), (0x200, 16)]);
    exec_bulk_memory(&MEMCPY, &[p(0x200), p(0x104), n64(8)], &mut m).expect("disjoint copy");
    assert_eq!(m.bytes(0x200, 8), (4..12).map(Some).collect::<Vec<_>>());
    assert_eq!(m.bytes(0x208, 8), (8..16).map(Some).collect::<Vec<_>>(), "the rest is untouched");

    // Partial overlap, either direction, is UB and leaves memory unchanged.
    for (d, s) in [(0x102, 0x100), (0x100, 0x102), (0x107, 0x100)] {
        let before = m.clone();
        let e = exec_bulk_memory(&MEMCPY, &[p(d), p(s), n64(8)], &mut m).expect_err("overlap");
        assert!(e.contains("overlap"), "{e}");
        assert_eq!(m, before);
    }
    // Adjacent ranges do not overlap; identical ones are allowed.
    exec_bulk_memory(&MEMCPY, &[p(0x108), p(0x100), n64(8)], &mut m).expect("adjacent");
    assert_eq!(m.bytes(0x108, 8), m.bytes(0x100, 8));
    let before = m.clone();
    exec_bulk_memory(&MEMCPY, &[p(0x100), p(0x100), n64(16)], &mut m).expect("dst == src");
    assert_eq!(m, before);
}

#[test]
fn memmove_handles_overlap_in_both_directions() {
    // Forward overlap (dst > src) and backward (dst < src), all small offsets.
    for len in 1..12u64 {
        for d in 0..8u64 {
            for s in 0..8u64 {
                let mut m = Mem::with(&[(0x100, 24)]);
                let mut want: Vec<Option<u8>> = m.bytes(0x100, 24);
                let tmp: Vec<Option<u8>> = want[s as usize..(s + len) as usize].to_vec();
                want[d as usize..(d + len) as usize].copy_from_slice(&tmp);
                exec_bulk_memory(&MEMMOVE, &[p(0x100 + d), p(0x100 + s), n64(len)], &mut m).expect("memmove");
                assert_eq!(m.bytes(0x100, 24), want, "len {len} dst +{d} src +{s}");
            }
        }
    }
}

#[test]
fn zero_length_is_a_no_op_whatever_the_pointers() {
    let mut m = Mem::with(&[(0x100, 4)]);
    let before = m.clone();
    for kind in [MEMCPY, MEMMOVE] {
        for (d, s) in [(p(0), p(0)), (SemValue::Poison, SemValue::Poison), (p(0xdead), p(0x101))] {
            exec_bulk_memory(&kind, &[d, s, n64(0)], &mut m).expect("n = 0");
        }
    }
    exec_bulk_memory(&MEMSET, &[SemValue::Poison, SemValue::Poison, SemValue::int(32, Int::ZERO)], &mut m)
        .expect("n = 0");
    // Even a misaligned pointer: nothing is accessed.
    let aligned = InstKind::MemSet { align: 8, volatile: false };
    exec_bulk_memory(&aligned, &[p(0x101), byte(1), n64(0)], &mut m).expect("n = 0");
    assert_eq!(m, before);
}

#[test]
fn poison_rules() {
    let mut m = Mem::with(&[(0x100, 8), (0x200, 8)]);
    // A poison length is UB, even with valid pointers.
    assert!(exec_bulk_memory(&MEMCPY, &[p(0x200), p(0x100), SemValue::Poison], &mut m).is_err());
    assert!(exec_bulk_memory(&MEMSET, &[p(0x200), byte(0), SemValue::Poison], &mut m).is_err());
    // A poison pointer with n > 0 is UB.
    assert!(exec_bulk_memory(&MEMCPY, &[SemValue::Poison, p(0x100), n64(1)], &mut m).is_err());
    assert!(exec_bulk_memory(&MEMCPY, &[p(0x200), SemValue::Poison, n64(1)], &mut m).is_err());
    assert!(exec_bulk_memory(&MEMSET, &[SemValue::Poison, byte(0), n64(1)], &mut m).is_err());
    // A poison byte fills with poison bytes.
    exec_bulk_memory(&MEMSET, &[p(0x202), SemValue::Poison, n64(3)], &mut m).expect("poison byte");
    assert_eq!(m.bytes(0x200, 6), vec![Some(0), Some(1), None, None, None, Some(5)]);
    // Poison source bytes are copied as poison, the defined ones as they are.
    exec_bulk_memory(&MEMCPY, &[p(0x100), p(0x200), n64(8)], &mut m).expect("copy");
    assert_eq!(m.bytes(0x100, 8), m.bytes(0x200, 8));
    assert_eq!(m.bytes(0x102, 1), vec![None]);
}

#[test]
fn bounds_and_alignment() {
    let mut m = Mem::with(&[(0x100, 8), (0x200, 8)]);
    // One byte past either allocation is UB.
    assert!(exec_bulk_memory(&MEMCPY, &[p(0x200), p(0x101), n64(8)], &mut m).is_err());
    assert!(exec_bulk_memory(&MEMCPY, &[p(0x201), p(0x100), n64(8)], &mut m).is_err());
    assert!(exec_bulk_memory(&MEMSET, &[p(0x100), byte(7), n64(9)], &mut m).is_err());
    assert!(exec_bulk_memory(&MEMSET, &[p(0x100), byte(7), SemValue::int(64, Int::from_u64(u64::MAX))], &mut m).is_err());
    // The declared alignment holds for both pointers.
    let a4 = InstKind::MemCopy { align: 4, volatile: false, overlapping: false };
    assert!(exec_bulk_memory(&a4, &[p(0x200), p(0x100), n64(4)], &mut m).is_ok());
    assert!(exec_bulk_memory(&a4, &[p(0x202), p(0x100), n64(4)], &mut m).is_err());
    assert!(exec_bulk_memory(&a4, &[p(0x200), p(0x102), n64(4)], &mut m).is_err());
    // Any length width, read unsigned: an i8 255 is 255 bytes.
    let mut big = Mem::with(&[(0x1000, 255)]);
    exec_bulk_memory(&MEMSET, &[p(0x1000), byte(9), SemValue::int(8, Int::from_u64(255))], &mut big).expect("i8 n");
    assert!(big.bytes(0x1000, 255).iter().all(|&b| b == Some(9)));
}

const BULK_LF: &str = r#"module "bulk"

func @f(ptr, ptr, i64, i8) -> void {
entry ^0(%0: ptr, %1: ptr, %2: i64, %3: i8):
  memcpy %0, %1, %2 align 1
  memcpy volatile %0, %1, i64 16 align 8
  memmove %1, %0, %2 align 4
  memmove volatile %1, %0, i32 3 align 1
  memset %0, %3, %2 align 2
  memset volatile %0, i8 0, i16 256 align 16
  ret
}
"#;

fn parse(src: &str, syms: &mut StrInterner) -> Module {
    crate::ir::text::parse_module(src, FileId::new(0), syms).unwrap_or_else(|e| panic!("parse: {e:?}"))
}

fn bulk_kinds(f: &Function) -> Vec<InstKind> {
    f.blocks().flat_map(|(_, b)| b.insts().iter().map(|&i| f.inst(i).kind.clone())).filter(|k| k.is_bulk_memory()).collect()
}

#[test]
fn text_and_binary_round_trip() {
    let mut syms = StrInterner::new();
    let m = parse(BULK_LF, &mut syms);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let text = crate::ir::text::print_module(&m, &syms);
    for line in BULK_LF.lines().filter(|l| l.contains("mem")) {
        assert!(text.contains(line.trim()), "missing `{}` in\n{text}", line.trim());
    }
    let again = parse(&text, &mut syms);
    assert_eq!(crate::ir::text::print_module(&again, &syms), text);

    let bytes = crate::ir::binary::encode(&m, &syms);
    let mut back = StrInterner::new();
    let m2 = crate::ir::binary::decode(&bytes, &mut back).expect("decode");
    assert_eq!(crate::ir::binary::encode(&m2, &back), bytes, "binary form is stable");
    assert_eq!(crate::ir::text::print_module(&m2, &back), text);
    let kinds = bulk_kinds(m2.function(crate::ir::FuncId::from_index(0)));
    assert_eq!(kinds.len(), 6);
    assert_eq!(kinds[3], InstKind::MemCopy { align: 1, volatile: true, overlapping: true });
    assert_eq!(kinds[5], InstKind::MemSet { align: 16, volatile: true });
    for cut in (bytes.len() / 2..bytes.len()).step_by(5) {
        assert!(crate::ir::binary::decode(&bytes[..cut], &mut StrInterner::new()).is_err());
    }
    // An unknown flag bit is rejected, not misread: find the memset tag (48)
    // followed by its flag byte 0 and set bit 1 (only `memcpy` has it).
    let at = bytes.windows(3).position(|w| w == [48, 0, 2]).expect("the first memset");
    let mut bad = bytes.clone();
    bad[at + 1] = 2;
    assert!(crate::ir::binary::decode(&bad, &mut StrInterner::new()).is_err());
    // A module without bulk ops encodes as before (no new tags).
    assert!(!crate::ir::binary::encode(&Module::new("x"), &syms).is_empty());
}

#[test]
fn builder_emits_each_flavor() {
    let mut syms = StrInterner::new();
    let mut m = Module::new("b");
    let i64t = m.types_mut().int(64);
    let i8t = m.types_mut().int(8);
    let ptr = m.types_mut().ptr();
    let void = m.types_mut().void();
    let sig = m.types_mut().func(vec![ptr, ptr], void, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let e = b.create_entry_block();
        let (d, s) = (b.param(e, 0), b.param(e, 1));
        let n = b.const_int(i64t, Int::from_u64(32));
        let z = b.const_int(i8t, Int::ZERO);
        b.memcpy(d, s, n, 8);
        b.memmove(d, s, n, 1);
        b.memset(d, z, n, 4);
        b.bulk_memory(InstKind::MemSet { align: 1, volatile: true }, d, z, n);
        b.ret(None);
    }
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let text = crate::ir::text::print_module(&m, &syms);
    for want in ["memcpy %0, %1, i64 32 align 8", "memmove %0, %1, i64 32 align 1", "memset %0, i8 0, i64 32 align 4", "memset volatile %0, i8 0, i64 32 align 1"] {
        assert!(text.contains(want), "missing `{want}` in\n{text}");
    }
    let kinds = bulk_kinds(m.function(f));
    assert!(kinds.iter().all(|k| k.has_side_effect() && k.is_bulk_memory()));
    assert_eq!(kinds.iter().filter(|k| k.is_volatile()).count(), 1);
}

#[test]
fn verifier_rejects_malformed_bulk_ops() {
    let cases: &[(&str, &str)] = &[
        ("memcpy %2, %1, %2 align 1", "destination must be a pointer"),
        ("memcpy %0, %2, %2 align 1", "source must be a pointer"),
        ("memset %0, %2, %2 align 1", "memset byte must be an i8"),
        ("memset %0, %3, %0 align 1", "length must be an integer"),
        ("memcpy %0, %1, i128 4 align 1", "at most 64 bits"),
        ("memmove %0, %1, %2 align 3", "memmove alignment 3"),
        ("memset %0, %3, %2 align 0", "memset alignment 0"),
    ];
    for (body, want) in cases {
        let src = format!(
            "module \"v\"\nfunc @f(ptr, ptr, i64, i8) -> void {{\nentry ^0(%0: ptr, %1: ptr, %2: i64, %3: i8):\n  {body}\n  ret\n}}\n"
        );
        let mut syms = StrInterner::new();
        let m = parse(&src, &mut syms);
        let diags = crate::verify::verify_module(&m).expect_err(body);
        assert!(diags.iter().any(|d| d.message.contains(want)), "{body}: wanted `{want}` in {diags:?}");
    }
    // Arity.
    let mut syms = StrInterner::new();
    let mut m = Module::new("a");
    let ptr = m.types_mut().ptr();
    let void = m.types_mut().void();
    let sig = m.types_mut().func(vec![ptr], void, false);
    let f = m.declare_function(syms.intern("f"), sig);
    {
        let mut b = m.build(f);
        let e = b.create_entry_block();
        let d = b.param(e, 0);
        b.append_inst(MEMCPY, vec![d, d], Flags::NONE, None);
        b.ret(None);
    }
    assert!(crate::verify::verify_module(&m).is_err());
    // A result name on a bulk op is a parse error (it produces no value).
    let src = "module \"r\"\nfunc @f(ptr) -> void {\nentry ^0(%0: ptr):\n  %1 = memset %0, i8 0, i64 1 align 1\n  ret\n}\n";
    assert!(crate::ir::text::parse_module(src, FileId::new(0), &mut StrInterner::new()).is_err());
}

/// Whole programs through the reference executor: a struct copy, a fill, an
/// overlapping move, and the UB cases.
const PROG_LF: &str = r#"module "prog"

func @copy_sum() -> i64 {
entry ^0:
  %a = alloca [4 x i64] : ptr
  %b = alloca [4 x i64] : ptr
  store i64 11, %a align 8 : i64
  %a1 = ptr_add %a, i64 8 : ptr
  store i64 22, %a1 align 8 : i64
  %a2 = ptr_add %a, i64 16 : ptr
  store i64 33, %a2 align 8 : i64
  memcpy %b, %a, i64 24 align 8
  %b2 = ptr_add %b, i64 16 : ptr
  %x = load %b2 align 8 : i64
  %y = load %b align 8 : i64
  %r = add %x, %y : i64
  ret %r
}

func @fill(i64) -> i64 {
entry ^0(%n: i64):
  %a = alloca [8 x i8] : ptr
  store i64 0, %a align 8 : i64
  memset %a, i8 171, %n align 1
  %v = load %a align 8 : i64
  ret %v
}

func @shift() -> i64 {
entry ^0:
  %a = alloca [8 x i8] : ptr
  store i64 506097522914230528, %a align 8 : i64
  %a1 = ptr_add %a, i64 1 : ptr
  memmove %a1, %a, i64 7 align 1
  %v = load %a align 8 : i64
  ret %v
}

func @overlap() -> i64 {
entry ^0:
  %a = alloca [8 x i8] : ptr
  store i64 0, %a align 8 : i64
  %a1 = ptr_add %a, i64 1 : ptr
  memcpy %a1, %a, i64 7 align 1
  ret i64 0
}

func @uninit() -> i64 {
entry ^0:
  %a = alloca i64 : ptr
  %b = alloca i64 : ptr
  memcpy %b, %a, i64 8 align 8
  %v = load %b align 8 : i64
  ret %v
}
"#;

#[test]
fn whole_programs_in_the_reference_executor() {
    let mut syms = StrInterner::new();
    let m = parse(PROG_LF, &mut syms);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let run = |name: &str, args: &[SemValue]| run_named(&m, &syms, name, args);
    assert_eq!(run("copy_sum", &[]), Ok(Some(n64(44))));
    assert_eq!(run("fill", &[n64(0)]), Ok(Some(n64(0))));
    assert_eq!(run("fill", &[n64(3)]), Ok(Some(n64(0xab_abab))));
    assert_eq!(run("fill", &[n64(8)]), Ok(Some(n64(0xabab_abab_abab_abab))));
    assert!(matches!(run("fill", &[n64(9)]), Err(ExecError::Ub(_))), "past the slot");
    // 0x0706050403020100 shifted up one byte, the low byte kept.
    assert_eq!(run("shift", &[]), Ok(Some(n64(0x0605_0403_0201_0000))));
    assert!(matches!(run("overlap", &[]), Err(ExecError::Ub(e)) if e.contains("overlap")));
    // Uninitialized bytes copy as poison.
    assert_eq!(run("uninit", &[]), Ok(Some(SemValue::Poison)));
}
