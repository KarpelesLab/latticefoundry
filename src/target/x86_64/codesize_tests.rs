//! Code-size regression tests for the x86-64 backend (issues #8 and #9):
//! compare-and-branch fusion, block layout and branch relaxation, immediate
//! forms, register hints, leaf frames and shared epilogues. Each program is
//! checked on the bytes it compiles to (decoded with LF's own disassembler)
//! and, where it runs, on what it does.

use crate::codegen::mir::{MachineFunction, MachineInst, MachineOperand, Reg};
use crate::ir::Module;
use crate::link::{ImageOptions, link_executable, write_executable};
use crate::mc::disasm::{Options, Syntax, decode};
use crate::mc::object::{ObjectModule, SymbolValue};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;
use crate::target::TargetArch;
use crate::transform::pipeline::{OptLevel, optimize};

use super::isel::{X86Op, X86_64Target};
use super::regs;

/// Parse `src`, verify, optimize at `level`, verify again.
fn prepare(src: &str, level: OptLevel) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let mut m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse .lf: {e:?}"));
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    optimize(&mut m, level);
    crate::verify::verify_module(&m).unwrap_or_else(|e| panic!("verify after {level:?}: {e:?}"));
    (m, syms)
}

/// The bytes of the defined function `name` in `obj`'s `.text`.
fn func_bytes<'a>(obj: &'a ObjectModule, name: &str) -> &'a [u8] {
    let sym = obj.symbols().iter().find(|s| s.name == name).expect("function symbol");
    let SymbolValue::Defined { section, offset } = sym.value else { panic!("{name} undefined") };
    &obj.section(section).bytes[offset as usize..(offset + sym.size) as usize]
}

/// One decoded instruction: its length and Intel-syntax text.
#[derive(Debug)]
struct Decoded {
    len: usize,
    mnemonic: String,
    text: String,
}

/// Decode `bytes` instruction by instruction.
fn disasm(bytes: &[u8]) -> Vec<Decoded> {
    let opts = Options { syntax: Syntax::Intel };
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let i = decode(TargetArch::X86_64, &bytes[at..], at as u64, &opts);
        assert!(i.known, "undecodable bytes at {at}: {:02x?}", &bytes[at..]);
        out.push(Decoded { len: i.len, mnemonic: i.mnemonic.clone(), text: i.text().replace('\t', " ") });
        at += i.len;
    }
    out
}

/// A listing of `code` for assertion messages.
fn listing(code: &[Decoded]) -> String {
    code.iter().map(|d| format!("  {}\n", d.text)).collect()
}

/// Link `obj` into a static executable, run it, and return its stdout and
/// exit status.
fn run(obj: ObjectModule, tag: &str) -> (Vec<u8>, i32) {
    let image = link_executable(vec![obj], &ImageOptions::default()).expect("link");
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let uniq = SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("lf_size_{tag}_{}_{uniq}", std::process::id()));
    write_executable(path.to_str().unwrap(), &image).expect("write executable");
    let child = loop {
        match std::process::Command::new(&path).stdout(std::process::Stdio::piped()).spawn() {
            Ok(c) => break c,
            // A transient ETXTBSY from another test thread's fork.
            Err(e) if e.raw_os_error() == Some(26) => std::thread::sleep(std::time::Duration::from_millis(5)),
            Err(e) => panic!("exec: {e}"),
        }
    };
    let out = child.wait_with_output().expect("wait");
    let _ = std::fs::remove_file(&path);
    (out.stdout, out.status.code().expect("exit status"))
}

/// The issue's syscall retry loop (Lode's `std/os.write_all`), with a `main`
/// that writes a buffer through it (0 once everything is written) and then
/// fails one write (`-EFAULT` from a null buffer), exiting with `0 - -14`.
const WRITE_ALL: &str = r#"
module "loop"
func @f(ptr, i64) -> i64 {
entry ^0(%p0: ptr, %n0: i64):
  br ^1(%p0, %n0)
^1(%p: ptr, %n: i64):
  %z = icmp eq %n, i64 0 : i1
  cond_br %z, ^2, ^3
^2:
  ret i64 0
^3:
  %r = syscall i64 1, i64 1, %p, %n : i64
  %e = icmp eq %r, i64 -4 : i1
  cond_br %e, ^1(%p, %n), ^4
^4:
  %bad = icmp slt %r, i64 1 : i1
  cond_br %bad, ^5, ^6
^5:
  ret %r
^6:
  %n2 = sub %n, %r : i64
  %p2 = ptr_add %p, %r : ptr
  br ^1(%p2, %n2)
}
func @main() -> i64 {
entry ^0:
  %buf = alloca [16 x i8] : ptr
  store i64 8022916924116329800, %buf align 8 : i64
  %hi = ptr_add %buf, i64 8 : ptr
  store i64 174353522, %hi align 8 : i64
  %r = call @f(%buf, i64 12) : i64
  %e = call @f(ptr null, i64 5) : i64
  %s = sub %r, %e : i64
  ret %s
}
"#;

#[test]
fn write_all_loop_is_small_and_runs() {
    for level in [OptLevel::O0, OptLevel::O2] {
        let (m, syms) = prepare(WRITE_ALL, level);
        let obj = super::compile_module(&m, &syms);
        let code = disasm(func_bytes(&obj, "f"));
        let size: usize = code.iter().map(|d| d.len).sum();
        let text = listing(&code);
        // 202 bytes before the fusion, layout, immediates and hints.
        assert!(size <= 64, "@f is {size} bytes at {level:?}:\n{text}");
        for d in &code {
            // No materialized flag (setcc/movzx/test of it), no 10-byte
            // constant, no frame or callee-saved register in this leaf.
            assert!(!d.mnemonic.starts_with("set") && !d.mnemonic.starts_with("movzx"), "{level:?}:\n{text}");
            assert!(!d.mnemonic.starts_with("movabs") && d.len < 10, "{level:?}:\n{text}");
            assert!(!d.mnemonic.starts_with("push") && !d.mnemonic.starts_with("pop"), "{level:?}:\n{text}");
            assert!(!d.text.contains("rbp") && !d.text.contains("rbx") && !d.text.contains("r12"), "{level:?}:\n{text}");
            // Every branch reaches with a rel8.
            if d.mnemonic.starts_with('j') {
                assert_eq!(d.len, 2, "{level:?}: long branch `{}`:\n{text}", d.text);
            }
        }
        // The compares against constants are immediate forms.
        assert!(code.iter().any(|d| d.text.starts_with("cmp") && d.text.ends_with("-0x4")), "{level:?}:\n{text}");
        assert!(code.iter().any(|d| d.mnemonic == "test"), "{level:?}:\n{text}");
        let (out, status) = run(obj, "write_all");
        assert_eq!(out, b"Hello World\n");
        assert_eq!(status, 14, "{level:?}");
    }
}

/// Values flow into the registers they are consumed from: arguments passed
/// on stay in their registers, a result is computed straight into `rax`
/// or the next call's argument register.
const HINTS: &str = r#"
module "hints"
func @ext(i64, i64) -> i64
func @fwd(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %r = call @ext(%a, %b) : i64
  ret %r
}
func @dec(i64) -> i64 {
entry ^0(%a: i64):
  %r = sub %a, i64 3 : i64
  ret %r
}
func @chain(i64) -> i64 {
entry ^0(%a: i64):
  %x = add %a, i64 1 : i64
  %r = call @ext(%x, %a) : i64
  %y = call @ext(%r, i64 0) : i64
  ret %y
}
"#;

#[test]
fn register_hints_avoid_copies() {
    let (m, syms) = prepare(HINTS, OptLevel::O0);
    let obj = super::compile_module(&m, &syms);
    // fwd: the arguments are already where ext wants them, its result where
    // fwd returns it: a frame and the call, no copy.
    let code = disasm(func_bytes(&obj, "fwd"));
    let movs: Vec<&str> = code.iter().filter(|d| d.mnemonic == "mov").map(|d| d.text.as_str()).collect();
    assert_eq!(movs, ["mov rbp, rsp"], "fwd:\n{}", listing(&code));
    // dec: a leaf, one `lea rax, [rdi - 3]`, then `ret`.
    let code = disasm(func_bytes(&obj, "dec"));
    assert_eq!(code.len(), 2, "dec:\n{}", listing(&code));
    assert_eq!(code[0].text, "lea rax, [rdi - 0x3]", "dec:\n{}", listing(&code));
    // chain: `a` moves out of rdi once (it lives across the add into rdi),
    // and the first result feeds the second call from rax with one copy.
    let code = disasm(func_bytes(&obj, "chain"));
    let movs = code.iter().filter(|d| d.mnemonic == "mov").count();
    assert!(movs <= 3, "chain:\n{}", listing(&code));
}

/// A leaf whose values all fit the caller-saved registers saves none, has
/// no frame pointer, and its several returns need no epilogue.
const LEAF: &str = r#"
module "leaf"
func @mix(i64, i64, i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64):
  %x = mul %a, %b : i64
  %y = mul %c, %d : i64
  %z = add %x, %y : i64
  %w = add %z, %a : i64
  %v = add %w, %b : i64
  %u = sub %v, %c : i64
  %t = sub %u, %d : i64
  %neg = icmp slt %t, i64 0 : i1
  cond_br %neg, ^1, ^2
^1:
  ret %x
^2:
  ret %t
}
func @main() -> i64 {
entry ^0:
  %r = call @mix(i64 3, i64 5, i64 7, i64 2) : i64
  ret %r
}
"#;

#[test]
fn leaf_functions_use_caller_saved_registers_and_no_frame() {
    let (m, syms) = prepare(LEAF, OptLevel::O0);
    let opts = crate::codegen::CodegenOptions::default();
    let out = super::compile_module_with(&m, &syms, &opts);
    let code = disasm(func_bytes(&out.object, "mix"));
    let text = listing(&code);
    for d in &code {
        assert!(!d.mnemonic.starts_with("push") && !d.mnemonic.starts_with("pop"), "mix:\n{text}");
        assert!(!d.text.contains("rbp") && !d.text.contains("rsp"), "mix:\n{text}");
    }
    let rets = code.iter().filter(|d| d.mnemonic == "ret").count();
    assert_eq!(rets, 2, "mix: both returns are a bare ret:\n{text}");
    let usage = out.stack.get("mix").unwrap();
    assert_eq!((usage.frame_size, usage.saved_registers, usage.sp_adjust), (8, 0, 0));
    // 15 + 14 + 3 + 5 - 7 - 2 = 28 is not negative.
    let (_, status) = run(out.object, "leaf");
    assert_eq!(status, 28);
}

/// Several returns from a function with callee-saved registers share one
/// epilogue.
const SHARED: &str = r#"
module "shared"
func @ext(i64) -> i64 {
entry ^0(%x: i64):
  %r = add %x, i64 100 : i64
  ret %r
}
func @two(i64) -> i64 {
entry ^0(%x: i64):
  %c = icmp slt %x, i64 0 : i1
  cond_br %c, ^1, ^2
^1:
  %a = call @ext(%x) : i64
  %s = add %a, %x : i64
  ret %s
^2:
  %b = call @ext(i64 7) : i64
  %t = mul %b, %x : i64
  ret %t
^3:
  ret i64 0
}
func @main() -> i64 {
entry ^0:
  %p = call @two(i64 -40) : i64
  %q = call @two(i64 2) : i64
  %r = add %p, %q : i64
  ret %r
}
"#;

#[test]
fn returns_share_one_epilogue() {
    let (m, syms) = prepare(SHARED, OptLevel::O0);
    let obj = super::compile_module(&m, &syms);
    let code = disasm(func_bytes(&obj, "two"));
    let text = listing(&code);
    let count = |m: &str| code.iter().filter(|d| d.mnemonic == m).count();
    assert!(code.iter().any(|d| d.mnemonic == "push" && d.text != "push rbp"), "two saves a register:\n{text}");
    assert_eq!(count("ret"), 1, "two:\n{text}");
    assert_eq!(code.iter().filter(|d| d.text == "pop rbp").count(), 1, "two:\n{text}");
    // (-40 + 100 - 40) + (107 * 2) = 234.
    let (_, status) = run(obj, "shared");
    assert_eq!(status, 234);
}

/// A conditional branch over a block of exactly 127 bytes takes the rel8
/// form; one byte more needs the rel32 form (and the jump back over both
/// stays short or grows accordingly).
#[test]
fn branch_relaxation_at_the_rel8_boundary() {
    let target = X86_64Target::new();
    let encode = |filler: usize| -> Vec<u8> {
        // entry: test rax, rax; je done  (else falls through into `body`)
        // body:  `filler` bytes of ud2 / mov, ret
        // done:  ret
        let mut mf = MachineFunction::new("relax", 0);
        let entry = mf.add_block();
        let body = mf.add_block();
        let done = mf.add_block();
        mf.set_entry(entry);
        let rax = MachineOperand::Use(Reg::Physical(regs::gpr(regs::RAX)));
        let imm = |v: i64| MachineOperand::Imm(puremp::Int::from_i64(v));
        mf.block_mut(entry).insts.push(MachineInst::new(
            X86Op::CmpBrI.opcode(),
            vec![rax, imm(0), imm(4), imm(64), MachineOperand::Label(done), MachineOperand::Label(body)],
        ));
        let mut left = filler;
        while left > 0 {
            if left % 2 == 1 {
                // mov rcx, rdx (3 bytes)
                mf.block_mut(body).insts.push(MachineInst::new(
                    X86Op::MovRR.opcode(),
                    vec![
                        MachineOperand::Def(Reg::Physical(regs::gpr(regs::RCX))),
                        MachineOperand::Use(Reg::Physical(regs::gpr(regs::RDX))),
                    ],
                ));
                left -= 3;
            } else {
                mf.block_mut(body).insts.push(MachineInst::new(X86Op::Unreachable.opcode(), Vec::new()));
                left -= 2;
            }
        }
        mf.block_mut(body).insts.push(MachineInst::new(X86Op::Ret.opcode(), Vec::new()));
        mf.block_mut(done).insts.push(MachineInst::new(X86Op::Ret.opcode(), Vec::new()));
        let layout = super::encode::layout_frame(&mf, &target);
        let name = |_: u32| String::from("relax");
        super::encode::encode_function(&mf, &layout, &name, &name).bytes
    };
    // 126 bytes of filler + the 1-byte ret: the jump skips exactly 127.
    let short = encode(126);
    assert_eq!(&short[..5], &[0x48, 0x85, 0xc0, 0x74, 127], "{short:02x?}");
    assert_eq!(short.len(), 3 + 2 + 127 + 1);
    let long = encode(127);
    assert_eq!(&long[..9], &[0x48, 0x85, 0xc0, 0x0f, 0x84, 128, 0, 0, 0], "{long:02x?}");
    assert_eq!(long.len(), 3 + 6 + 128 + 1);
}
