//! The sanitizer's reporting runtime, written in LF IR.
//!
//! [`link_runtime`] parses [`runtime_source`] and links it into the module
//! being instrumented, so a sanitized program needs no C library and no
//! separately built runtime: it is freestanding, like the AVR runtime. Every
//! symbol is **weak**, so several sanitized objects linked together keep one
//! copy and a program may supply its own `__lf_ub_report`.
//!
//! `void __lf_ub_report(i32 code, ptr loc, i64 a, i64 b)` (the ABI is in the
//! [parent module](super)):
//!
//! 1. returns at once if the location already reported (its `reported` word
//!    is set), after the exit decision below;
//! 2. formats `file:line: runtime error: <message>\n` into a 256-byte stack
//!    buffer (`file: in function f: runtime error: …` when the line is
//!    unknown), with the operands in decimal (an address in hex);
//! 3. writes it to file descriptor 2 with the `write` system call;
//! 4. exits with status 1 (`exit_group`) when the module was sanitized without
//!    recovery or the kind is [always fatal](super::UbKind::always_fatal);
//!    otherwise returns.
//!
//! The system-call numbers are the target's ([`LinuxSyscalls`]): the runtime
//! runs on Linux x86-64, AArch64 and RISC-V.

use crate::ir::Module;
use crate::ir::datalayout::DataLayout;
use crate::ir::text::parse_module;
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

use super::{LinuxSyscalls, REPORT_FN, UbKind};

/// The fixed strings of the runtime: `(symbol suffix, text)`.
const STRINGS: &[(&str, &str)] = &[
    ("colon", ":"),
    ("head", ": runtime error: "),
    ("in", ": in function "),
    ("nl", "\\n"),
    ("minus", "-"),
    ("x", "0x"),
    ("i", "i"),
    ("empty", ""),
    ("unsigned", " (unsigned)"),
    ("type", " cannot be represented in type i"),
    ("load", "load"),
    ("store", "store"),
    ("atomic", "atomic access"),
    ("k1", "signed integer overflow: "),
    ("k2", "unsigned integer overflow: "),
    ("k3a", "shift exponent "),
    ("k3b", " is too large for "),
    ("k3c", "-bit type"),
    ("k4a", "left shift of "),
    ("k4b", " by "),
    ("k4c", " places cannot be represented in type i"),
    ("k5", "division by zero"),
    ("k6a", "division of "),
    ("k6b", " by -1 cannot be represented in type i"),
    ("k7", "exact operation lost information: "),
    ("k8", "floating-point value is outside the range of representable values of type "),
    ("k9", "pointer offset "),
    ("k10", " at offset "),
    ("kb", " is out of bounds for an object of "),
    ("kbytes", " bytes"),
    ("k11", " of null pointer"),
    ("k12a", " of misaligned address "),
    ("k12b", " for type with alignment "),
    ("k13", "execution reached an unreachable program point"),
];

/// The helper functions: string, character, operator, decimal and hex output
/// into a buffer at a position, each returning the new position (capped so
/// the buffer always has room for the final newline).
const HELPERS: &str = r#"
func weak @__lf_ub_put(ptr, i64, ptr) -> i64 {
entry ^0(%buf: ptr, %pos: i64, %s: ptr):
  br ^1(%pos, %s)
^1(%p: i64, %q: ptr):
  %c = load %q align 1 : i8
  %z = icmp eq %c, i8 0 : i1
  %full = icmp uge %p, i64 250 : i1
  %stop = or %z, %full : i1
  cond_br %stop, ^2, ^3
^3:
  %d = ptr_add %buf, %p : ptr
  store %c, %d align 1 : i8
  %p2 = add %p, i64 1 : i64
  %q2 = ptr_add %q, i64 1 : ptr
  br ^1(%p2, %q2)
^2:
  ret %p
}

func weak @__lf_ub_putc(ptr, i64, i8) -> i64 {
entry ^0(%buf: ptr, %pos: i64, %c: i8):
  %full = icmp uge %pos, i64 250 : i1
  cond_br %full, ^1, ^2
^2:
  %d = ptr_add %buf, %pos : ptr
  store %c, %d align 1 : i8
  %p = add %pos, i64 1 : i64
  ret %p
^1:
  ret %pos
}

func weak @__lf_ub_putop(ptr, i64, i8) -> i64 {
entry ^0(%buf: ptr, %pos: i64, %c: i8):
  %p1 = call @__lf_ub_putc(%buf, %pos, i8 32) : i64
  %p2 = call @__lf_ub_putc(%buf, %p1, %c) : i64
  %lt = icmp eq %c, i8 60 : i1
  %gt = icmp eq %c, i8 62 : i1
  %dbl = or %lt, %gt : i1
  cond_br %dbl, ^1, ^2(%p2)
^1:
  %p3 = call @__lf_ub_putc(%buf, %p2, %c) : i64
  br ^2(%p3)
^2(%p4: i64):
  %p5 = call @__lf_ub_putc(%buf, %p4, i8 32) : i64
  ret %p5
}

func weak @__lf_ub_putu(ptr, i64, i64) -> i64 {
entry ^0(%buf: ptr, %pos: i64, %v: i64):
  %tmp = alloca [24 x i8] : ptr
  %end = ptr_add %tmp, i64 23 : ptr
  store i8 0, %end align 1 : i8
  br ^1(%v, i64 23)
^1(%x: i64, %i: i64):
  %q = udiv %x, i64 10 : i64
  %m = mul %q, i64 10 : i64
  %d = sub %x, %m : i64
  %d8 = trunc %d : i8
  %ch = add %d8, i8 48 : i8
  %i2 = sub %i, i64 1 : i64
  %pp = ptr_add %tmp, %i2 : ptr
  store %ch, %pp align 1 : i8
  %more = icmp ne %q, i64 0 : i1
  cond_br %more, ^1(%q, %i2), ^2(%i2)
^2(%s: i64):
  %sp = ptr_add %tmp, %s : ptr
  %r = call @__lf_ub_put(%buf, %pos, %sp) : i64
  ret %r
}

func weak @__lf_ub_puti(ptr, i64, i64) -> i64 {
entry ^0(%buf: ptr, %pos: i64, %v: i64):
  %neg = icmp slt %v, i64 0 : i1
  cond_br %neg, ^1, ^2(%pos, %v)
^1:
  %p1 = call @__lf_ub_put(%buf, %pos, @__lf_ub_s_minus) : i64
  %n = sub i64 0, %v : i64
  br ^2(%p1, %n)
^2(%p: i64, %x: i64):
  %r = call @__lf_ub_putu(%buf, %p, %x) : i64
  ret %r
}

func weak @__lf_ub_putx(ptr, i64, i64) -> i64 {
entry ^0(%buf: ptr, %pos: i64, %v: i64):
  %tmp = alloca [24 x i8] : ptr
  %end = ptr_add %tmp, i64 23 : ptr
  store i8 0, %end align 1 : i8
  br ^1(%v, i64 23)
^1(%x: i64, %i: i64):
  %d = and %x, i64 15 : i64
  %d8 = trunc %d : i8
  %small = icmp ult %d8, i8 10 : i1
  %dec = add %d8, i8 48 : i8
  %hex = add %d8, i8 87 : i8
  %ch = select %small, %dec, %hex : i8
  %i2 = sub %i, i64 1 : i64
  %pp = ptr_add %tmp, %i2 : ptr
  store %ch, %pp align 1 : i8
  %q = lshr %x, i64 4 : i64
  %more = icmp ne %q, i64 0 : i1
  cond_br %more, ^1(%q, %i2), ^2(%i2)
^2(%s: i64):
  %p0 = call @__lf_ub_put(%buf, %pos, @__lf_ub_s_x) : i64
  %sp = ptr_add %tmp, %s : ptr
  %r = call @__lf_ub_put(%buf, %p0, %sp) : i64
  ret %r
}
"#;

/// One step of a message: a fixed string, an operand, or the type width.
enum Piece {
    /// A runtime string (`__lf_ub_s_<name>`).
    S(&'static str),
    /// The access word (`load`/`store`/`atomic access`).
    Access,
    /// Operand `a` signed / unsigned / hex, operand `b` signed / unsigned.
    ASigned,
    AUnsigned,
    AHex,
    BSigned,
    BUnsigned,
    /// The operator (the detail byte, spaced).
    Op,
    /// The bit width.
    Width,
    /// ` (unsigned)` when the detail byte is `u`.
    UnsignedNote,
}

/// The message of each kind, as pieces.
fn message(kind: UbKind) -> Vec<Piece> {
    use Piece::*;
    match kind {
        UbKind::SignedOverflow => vec![S("k1"), ASigned, Op, BSigned, S("type"), Width],
        UbKind::UnsignedOverflow => vec![S("k2"), AUnsigned, Op, BUnsigned, S("type"), Width],
        UbKind::ShiftExponent => vec![S("k3a"), BUnsigned, S("k3b"), Width, S("k3c")],
        UbKind::ShiftBase => vec![S("k4a"), ASigned, S("k4b"), BUnsigned, S("k4c"), Width],
        UbKind::DivByZero => vec![S("k5")],
        UbKind::DivOverflow => vec![S("k6a"), ASigned, S("k6b"), Width],
        UbKind::Inexact => vec![S("k7"), ASigned, Op, BSigned],
        UbKind::FloatCast => vec![S("k8"), S("i"), Width, UnsignedNote],
        UbKind::PointerBounds => vec![S("k9"), ASigned, S("kb"), BUnsigned, S("kbytes")],
        UbKind::ObjectBounds => vec![Access, S("k10"), ASigned, S("kb"), BUnsigned, S("kbytes")],
        UbKind::NullPointer => vec![Access, S("k11")],
        UbKind::Misaligned => vec![Access, S("k12a"), AHex, S("k12b"), BUnsigned],
        UbKind::Unreachable => vec![S("k13")],
    }
}

/// The `.lf` source of the runtime for `abi`. Without `recover`, every report
/// exits. `layout` gives the pointer size of the location record.
pub fn runtime_source(abi: LinuxSyscalls, recover: bool, layout: &DataLayout) -> String {
    let (write, exit) = match abi {
        LinuxSyscalls::X86_64 => (1, 231),
        LinuxSyscalls::Generic => (64, 94),
    };
    let ptr = layout.pointer_or_default(0);
    let (psz, pal) = (ptr.bytes(), ptr.align);
    let (line_off, flags_off) = (2 * psz, 2 * psz + 4);

    let mut s = String::from("module \"lf_ub_runtime\"\n");
    for (name, text) in STRINGS {
        // `\n` is written escaped in the table: count it as one byte.
        let n = text.replace("\\n", "\n").len() + 1;
        s.push_str(&format!("global weak constant @__lf_ub_s_{name} : [{n} x i8] = [{n} x i8] \"{text}\\0\"\n"));
    }
    s.push_str(HELPERS);

    let mut cases = Vec::new();
    let mut bodies = String::new();
    for kind in UbKind::ALL {
        let k = kind.code();
        cases.push(format!("{k}: ^{}", 100 + u32::from(k)));
        bodies.push_str(&format!("^{}:\n", 100 + u32::from(k)));
        let mut pos = "%p4".to_owned();
        for (j, piece) in message(kind).iter().enumerate() {
            let next = format!("%m{k}_{j}");
            let call = match piece {
                Piece::S(name) => format!("call @__lf_ub_put(%buf, {pos}, @__lf_ub_s_{name})"),
                Piece::Access => format!("call @__lf_ub_put(%buf, {pos}, %acc)"),
                Piece::ASigned => format!("call @__lf_ub_puti(%buf, {pos}, %a)"),
                Piece::AUnsigned => format!("call @__lf_ub_putu(%buf, {pos}, %a)"),
                Piece::AHex => format!("call @__lf_ub_putx(%buf, {pos}, %a)"),
                Piece::BSigned => format!("call @__lf_ub_puti(%buf, {pos}, %b)"),
                Piece::BUnsigned => format!("call @__lf_ub_putu(%buf, {pos}, %b)"),
                Piece::Op => format!("call @__lf_ub_putop(%buf, {pos}, %det)"),
                Piece::Width => format!("call @__lf_ub_putu(%buf, {pos}, %w)"),
                Piece::UnsignedNote => format!("call @__lf_ub_put(%buf, {pos}, %unote)"),
            };
            bodies.push_str(&format!("  {next} = {call} : i64\n"));
            pos = next;
        }
        bodies.push_str(&format!("  br ^80({pos})\n"));
    }
    let fatal: Vec<u8> = UbKind::ALL.iter().filter(|k| k.always_fatal()).map(|k| k.code()).collect();
    let mut fatal_ir = String::new();
    let mut acc = if recover { "i1 0".to_owned() } else { "i1 1".to_owned() };
    for (j, code) in fatal.iter().enumerate() {
        fatal_ir.push_str(&format!("  %fk{j} = icmp eq %kind9, i32 {code} : i1\n"));
        fatal_ir.push_str(&format!("  %fa{j} = or {acc}, %fk{j} : i1\n"));
        acc = format!("%fa{j}");
    }

    s.push_str(&format!(
        r#"
func weak @{REPORT_FN}(i32, ptr, i64, i64) -> void {{
entry ^0(%code: i32, %loc: ptr, %a: i64, %b: i64):
  %fp = ptr_add %loc, i64 {flags_off} : ptr
  %flags = load %fp align 4 : i32
  %seen = icmp ne %flags, i32 0 : i1
  cond_br %seen, ^90, ^1
^1:
  store i32 1, %fp align 4 : i32
  %buf = alloca [256 x i8] : ptr
  %file = load %loc align {pal} : ptr
  %p0 = call @__lf_ub_put(%buf, i64 0, %file) : i64
  %lp = ptr_add %loc, i64 {line_off} : ptr
  %line = load %lp align 4 : i32
  %hasline = icmp ne %line, i32 0 : i1
  cond_br %hasline, ^2, ^3
^2:
  %p1 = call @__lf_ub_put(%buf, %p0, @__lf_ub_s_colon) : i64
  %line64 = zext %line : i64
  %p2 = call @__lf_ub_putu(%buf, %p1, %line64) : i64
  %p2b = call @__lf_ub_put(%buf, %p2, @__lf_ub_s_head) : i64
  br ^4(%p2b)
^3:
  %fnp = ptr_add %loc, i64 {psz} : ptr
  %fname = load %fnp align {pal} : ptr
  %p3 = call @__lf_ub_put(%buf, %p0, @__lf_ub_s_in) : i64
  %p3b = call @__lf_ub_put(%buf, %p3, %fname) : i64
  %p3c = call @__lf_ub_put(%buf, %p3b, @__lf_ub_s_head) : i64
  br ^4(%p3c)
^4(%p4: i64):
  %kind = and %code, i32 255 : i32
  %d1 = lshr %code, i32 8 : i32
  %d2 = and %d1, i32 255 : i32
  %det = trunc %d2 : i8
  %w1 = lshr %code, i32 16 : i32
  %w = zext %w1 : i64
  %isst = icmp eq %det, i8 115 : i1
  %isat = icmp eq %det, i8 97 : i1
  %acc1 = select %isst, @__lf_ub_s_store, @__lf_ub_s_load : ptr
  %acc = select %isat, @__lf_ub_s_atomic, %acc1 : ptr
  %isu = icmp eq %det, i8 117 : i1
  %unote = select %isu, @__lf_ub_s_unsigned, @__lf_ub_s_empty : ptr
  switch %kind, ^80(%p4) [{cases}]
{bodies}^80(%pe: i64):
  %pn = call @__lf_ub_put(%buf, %pe, @__lf_ub_s_nl) : i64
  %wr = syscall i64 {write}, i64 2, %buf, %pn : i64
  br ^90
^90:
  %kind9 = and %code, i32 255 : i32
{fatal_ir}  cond_br {acc}, ^91, ^92
^91:
  %ex = syscall i64 {exit}, i64 1 : i64
  unreachable
^92:
  ret
}}
"#,
        cases = cases.join(", "),
    ));
    s
}

/// Link the runtime into `module` (unless it already defines the handler),
/// with `module`'s data layout and target.
pub fn link_runtime(
    module: &mut Module,
    syms: &mut StrInterner,
    abi: LinuxSyscalls,
    recover: bool,
) -> Result<(), String> {
    let name = syms.intern(REPORT_FN);
    if module.functions().any(|f| f.name == name && !f.is_declaration()) {
        return Ok(());
    }
    let src = runtime_source(abi, recover, module.data_layout());
    let mut rt = parse_module(&src, FileId::new(0), syms).map_err(|diags| {
        let msgs: Vec<String> = diags.iter().map(|d| d.message.clone()).collect();
        format!("sanitizer runtime does not parse: {}", msgs.join("; "))
    })?;
    rt.set_data_layout(module.data_layout().clone());
    rt.set_target(module.target().map(str::to_owned));
    module.link_module(rt).map_err(|e| format!("linking the sanitizer runtime: {e}"))
}
