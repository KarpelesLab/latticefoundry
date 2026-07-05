//! The `.lf` textual form of the LatticeFoundry IR: a printer and a parser.
//!
//! This module renders an in-memory [`Module`] to a readable, LatticeFoundry-
//! specific textual syntax (`.lf`) and parses that syntax back into an equal
//! module. The two directions round-trip **losslessly**: the printer is
//! *canonical* (its output depends only on the module's structure, never on
//! internal id-allocation order), so for any module `m`
//!
//! ```text
//! print(parse(print(m))) == print(m)
//! ```
//!
//! and the parsed module is structurally identical to the original.
//!
//! The grammar is our own — it borrows familiar spellings for opcodes but is not
//! LLVM's `.ll`. In particular, SSA merges use **block arguments** (a block's
//! typed parameter list, with per-edge argument lists on terminators), not
//! φ-nodes, matching the IR model (`docs/ir-design.md` §2).
//!
//! # Names
//!
//! Function and global names are interned [`Sym`](crate::support::Sym)s, which
//! live in a [`StrInterner`] the module does not own. The printer therefore
//! takes a `&StrInterner` to resolve them, and the parser takes a
//! `&mut StrInterner` to intern them. Passing the *same* interner to both is
//! what makes the round-trip name-preserving.
//!
//! # Grammar (EBNF-ish)
//!
//! ```text
//! module      ::= "module" STRING { item }
//! item        ::= global | func
//!
//! global      ::= "global" "@" name ":" type [ "=" const ]
//! func        ::= "func" "@" name fnsig [ body ]
//! fnsig       ::= "(" [ type { "," type } [ "," "..." ] | "..." ] ")" "->" type
//! body        ::= "{" { block } "}"
//! block       ::= [ "entry" ] "^" INT [ "(" [ param { "," param } ] ")" ] ":" { inst }
//! param       ::= "%" name ":" type
//!
//! inst        ::= [ "%" name "=" ] op
//! op          ::= binop | "fneg" fm operand ":" type
//!               | "icmp" ipred operand "," operand ":" type
//!               | "fcmp" fpred fm operand "," operand ":" type
//!               | castop operand ":" type
//!               | "alloca" type ":" type
//!               | "load" operand "align" INT ":" type
//!               | "store" operand "," operand "align" INT ":" type
//!               | "ptr_add" [ "inbounds" ] operand "," operand ":" type
//!               | "select" operand "," operand "," operand ":" type
//!               | "freeze" operand ":" type
//!               | "call" operand "(" [ operand { "," operand } ] ")" ":" type
//!               | "ret" [ operand ]
//!               | "br" target
//!               | "cond_br" operand "," target "," target
//!               | "switch" operand "," target "[" [ case { "," case } ] "]"
//!               | "unreachable"
//! binop       ::= ("add"|"sub"|"mul"|"shl") iflags operand "," operand ":" type
//!               | ("udiv"|"sdiv"|"lshr"|"ashr") iflags operand "," operand ":" type
//!               | ("urem"|"srem"|"and"|"or"|"xor") operand "," operand ":" type
//!               | ("fadd"|"fsub"|"fmul"|"fdiv"|"frem") fm operand "," operand ":" type
//! iflags      ::= { "nsw" | "nuw" | "exact" }
//! fm          ::= { "nnan" | "ninf" | "nsz" | "reassoc" | "contract" | "afn" }
//! target      ::= "^" INT [ "(" [ operand { "," operand } ] ")" ]
//! case        ::= INT ":" target
//!
//! operand     ::= "%" name | "@" name | const
//! const       ::= type ( INT | "0x" HEX | "null" | "poison"
//!                       | "(" [ const { "," const } ] ")" )
//! type        ::= "void" | "i" INT | "f16" | "f32" | "f64" | "ptr"
//!               | "[" INT "x" type "]" | "{" [ type { "," type } ] "}"
//!               | "fn" fnsig
//! name        ::= IDENT | STRING
//! ```
//!
//! `;` begins a line comment. Integer constants are arbitrary precision
//! (`puremp::Int`); floating-point constants print as the raw IEEE bit pattern in
//! hex so they are exact and host-independent. Value operands that are constants,
//! global references, or function references are written inline (never as an
//! `%`-name), so only instruction results and block parameters receive `%`-names.

use std::collections::HashMap;
use std::fmt;

use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, CastOp, FastMath, Flags, FloatPred, InstId, InstKind, IntPred, UnaryOp};
use crate::ir::types::{FloatKind, Type};
use crate::ir::value::{Const, ConstId, FloatBits, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Function, Global, GlobalId, Module, TypeId};
use crate::support::StrInterner;
use crate::support::diagnostics::{Diagnostic, FileId, Span};

// ===========================================================================
// Printer
// ===========================================================================

/// A [`fmt::Display`] adapter that renders a [`Module`] in `.lf` textual form.
///
/// Obtain one via [`display`]; formatting it (`format!`, `write!`, `to_string`)
/// produces the same text as [`print_module`].
#[derive(Debug)]
pub struct ModuleDisplay<'a> {
    module: &'a Module,
    syms: &'a StrInterner,
}

impl fmt::Display for ModuleDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_module(f, self.module, self.syms)
    }
}

/// Build a [`fmt::Display`] adapter over `module`, resolving names through `syms`.
pub fn display<'a>(module: &'a Module, syms: &'a StrInterner) -> ModuleDisplay<'a> {
    ModuleDisplay { module, syms }
}

/// Render `module` to its `.lf` textual form as an owned [`String`].
pub fn print_module(module: &Module, syms: &StrInterner) -> String {
    let mut s = String::new();
    // Writing to a String is infallible.
    let _ = write_module(&mut s, module, syms);
    s
}

/// Render `module` to any [`fmt::Write`] sink, resolving names through `syms`.
pub fn write_module<W: fmt::Write>(f: &mut W, module: &Module, syms: &StrInterner) -> fmt::Result {
    write!(f, "module ")?;
    write_quoted(f, &module.name)?;
    writeln!(f)?;

    for g in module.globals() {
        writeln!(f)?;
        write_global(f, module, syms, g)?;
    }

    for func in module.functions() {
        writeln!(f)?;
        write_function(f, module, syms, func)?;
    }
    Ok(())
}

fn write_global<W: fmt::Write>(
    f: &mut W,
    module: &Module,
    syms: &StrInterner,
    g: &Global,
) -> fmt::Result {
    write!(f, "global ")?;
    write_name(f, syms.resolve(g.name))?;
    write!(f, " : ")?;
    write_type(f, module, g.ty)?;
    if let Some(init) = g.init {
        write!(f, " = ")?;
        write_const(f, module, init)?;
    }
    writeln!(f)
}

fn write_function<W: fmt::Write>(
    f: &mut W,
    module: &Module,
    syms: &StrInterner,
    func: &Function,
) -> fmt::Result {
    write!(f, "func ")?;
    write_name(f, syms.resolve(func.name))?;
    write_signature(f, module, func.sig)?;

    if func.is_declaration() {
        return writeln!(f);
    }

    let names = value_names(func);
    writeln!(f, " {{")?;
    for (bid, block) in func.blocks() {
        // Block header.
        if Some(bid) == func.entry() {
            write!(f, "entry ")?;
        }
        write!(f, "^{}", bid.index())?;
        if !block.params().is_empty() {
            write!(f, "(")?;
            for (i, &p) in block.params().iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "%{}: ", names[&p])?;
                write_type(f, module, func.value(p).ty)?;
            }
            write!(f, ")")?;
        }
        writeln!(f, ":")?;

        for &iid in block.insts() {
            write!(f, "  ")?;
            write_inst(f, module, syms, func, &names, iid)?;
            writeln!(f)?;
        }
        if let Some(t) = block.terminator() {
            write!(f, "  ")?;
            write_inst(f, module, syms, func, &names, t)?;
            writeln!(f)?;
        }
    }
    writeln!(f, "}}")
}

/// Assign each named value (block parameters and instruction results) a stable
/// print name, numbered in canonical walk order so the output is independent of
/// internal `ValueId` allocation order.
fn value_names(func: &Function) -> HashMap<ValueId, u32> {
    let mut map = HashMap::new();
    let mut n = 0u32;
    for (_bid, block) in func.blocks() {
        for &p in block.params() {
            map.insert(p, n);
            n += 1;
        }
        for &iid in block.insts() {
            if let Some(r) = func.inst(iid).result() {
                map.insert(r, n);
                n += 1;
            }
        }
        if let Some(t) = block.terminator()
            && let Some(r) = func.inst(t).result()
        {
            map.insert(r, n);
            n += 1;
        }
    }
    map
}

fn write_inst<W: fmt::Write>(
    f: &mut W,
    module: &Module,
    syms: &StrInterner,
    func: &Function,
    names: &HashMap<ValueId, u32>,
    iid: InstId,
) -> fmt::Result {
    let data = func.inst(iid);
    if let Some(r) = data.result() {
        write!(f, "%{} = ", names[&r])?;
    }
    let ops = data.operands();
    let op = |f: &mut W, v: ValueId| write_operand(f, module, syms, func, names, v);

    match &data.kind {
        InstKind::Bin(b) => {
            write!(f, "{}", binop_name(*b))?;
            if b.is_float() {
                write_fastmath(f, data.flags.fast)?;
            } else {
                write_iflags(f, data.flags)?;
            }
            write!(f, " ")?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            op(f, ops[1])?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Unary(UnaryOp::FNeg) => {
            write!(f, "fneg")?;
            write_fastmath(f, data.flags.fast)?;
            write!(f, " ")?;
            op(f, ops[0])?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::ICmp(p) => {
            write!(f, "icmp {} ", ipred_name(*p))?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            op(f, ops[1])?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::FCmp(p) => {
            write!(f, "fcmp {}", fpred_name(*p))?;
            write_fastmath(f, data.flags.fast)?;
            write!(f, " ")?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            op(f, ops[1])?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Cast(c) => {
            write!(f, "{} ", castop_name(*c))?;
            op(f, ops[0])?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Alloca { elem_ty } => {
            write!(f, "alloca ")?;
            write_type(f, module, *elem_ty)?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Load { ty, align } => {
            write!(f, "load ")?;
            op(f, ops[0])?;
            write!(f, " align {align} : ")?;
            write_type(f, module, *ty)
        }
        InstKind::Store { ty, align } => {
            // operands are [ptr, value]; print value first for readability.
            write!(f, "store ")?;
            op(f, ops[1])?;
            write!(f, ", ")?;
            op(f, ops[0])?;
            write!(f, " align {align} : ")?;
            write_type(f, module, *ty)
        }
        InstKind::PtrAdd { inbounds } => {
            write!(f, "ptr_add")?;
            if *inbounds {
                write!(f, " inbounds")?;
            }
            write!(f, " ")?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            op(f, ops[1])?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Select => {
            write!(f, "select ")?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            op(f, ops[1])?;
            write!(f, ", ")?;
            op(f, ops[2])?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Freeze => {
            write!(f, "freeze ")?;
            op(f, ops[0])?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Call => {
            write!(f, "call ")?;
            op(f, ops[0])?;
            write!(f, "(")?;
            for (i, &a) in ops[1..].iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                op(f, a)?;
            }
            write!(f, ") : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Ret => {
            write!(f, "ret")?;
            if let Some(&v) = ops.first() {
                write!(f, " ")?;
                op(f, v)?;
            }
            Ok(())
        }
        InstKind::Br(target) => {
            write!(f, "br ")?;
            write_target(f, module, syms, func, names, *target, ops)
        }
        InstKind::CondBr { if_true, if_false, true_args, false_args } => {
            let t = *true_args as usize;
            let ff = *false_args as usize;
            write!(f, "cond_br ")?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            write_target(f, module, syms, func, names, *if_true, &ops[1..1 + t])?;
            write!(f, ", ")?;
            write_target(f, module, syms, func, names, *if_false, &ops[1 + t..1 + t + ff])
        }
        InstKind::Switch(data_box) => {
            let da = data_box.default_args as usize;
            write!(f, "switch ")?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            write_target(f, module, syms, func, names, data_box.default, &ops[1..1 + da])?;
            write!(f, " [")?;
            let mut off = 1 + da;
            for (i, case) in data_box.cases.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let n = case.args as usize;
                write!(f, "{}: ", case.value)?;
                write_target(f, module, syms, func, names, case.target, &ops[off..off + n])?;
                off += n;
            }
            write!(f, "]")
        }
        InstKind::Unreachable => write!(f, "unreachable"),
    }
}

#[allow(clippy::too_many_arguments)]
fn write_target<W: fmt::Write>(
    f: &mut W,
    module: &Module,
    syms: &StrInterner,
    func: &Function,
    names: &HashMap<ValueId, u32>,
    target: BlockId,
    args: &[ValueId],
) -> fmt::Result {
    write!(f, "^{}", target.index())?;
    if !args.is_empty() {
        write!(f, "(")?;
        for (i, &a) in args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write_operand(f, module, syms, func, names, a)?;
        }
        write!(f, ")")?;
    }
    Ok(())
}

fn write_operand<W: fmt::Write>(
    f: &mut W,
    module: &Module,
    syms: &StrInterner,
    func: &Function,
    names: &HashMap<ValueId, u32>,
    v: ValueId,
) -> fmt::Result {
    match &func.value(v).def {
        ValueDef::Inst(_) | ValueDef::Param(_, _) => write!(f, "%{}", names[&v]),
        ValueDef::Const(cid) => write_const(f, module, *cid),
        ValueDef::Global(g) => write_name(f, syms.resolve(module.global(*g).name)),
        ValueDef::Func(fu) => write_name(f, syms.resolve(module.function(*fu).name)),
    }
}

fn write_const<W: fmt::Write>(f: &mut W, module: &Module, cid: ConstId) -> fmt::Result {
    let c = module.consts().get(cid);
    write_type(f, module, c.type_id())?;
    match c {
        Const::Int { value, .. } => write!(f, " {value}"),
        Const::Float { bits, .. } => match bits {
            FloatBits::F16(b) => write!(f, " 0x{b:04x}"),
            FloatBits::F32(b) => write!(f, " 0x{b:08x}"),
            FloatBits::F64(b) => write!(f, " 0x{b:016x}"),
        },
        Const::Null(_) => write!(f, " null"),
        Const::Poison(_) => write!(f, " poison"),
        Const::Aggregate { elems, .. } => {
            write!(f, " (")?;
            for (i, &e) in elems.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write_const(f, module, e)?;
            }
            write!(f, ")")
        }
    }
}

fn write_signature<W: fmt::Write>(f: &mut W, module: &Module, sig: TypeId) -> fmt::Result {
    let Type::Func(ft) = module.types().get(sig) else {
        // A non-Func signature should not occur; print defensively.
        write!(f, "(<bad-sig>) -> ")?;
        return write_type(f, module, sig);
    };
    write!(f, "(")?;
    for (i, &p) in ft.params.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write_type(f, module, p)?;
    }
    if ft.variadic {
        if ft.params.is_empty() {
            write!(f, "...")?;
        } else {
            write!(f, ", ...")?;
        }
    }
    write!(f, ") -> ")?;
    write_type(f, module, ft.ret)
}

fn write_type<W: fmt::Write>(f: &mut W, module: &Module, ty: TypeId) -> fmt::Result {
    match module.types().get(ty) {
        Type::Void => write!(f, "void"),
        Type::Int(w) => write!(f, "i{w}"),
        Type::Float(FloatKind::F16) => write!(f, "f16"),
        Type::Float(FloatKind::F32) => write!(f, "f32"),
        Type::Float(FloatKind::F64) => write!(f, "f64"),
        Type::Ptr => write!(f, "ptr"),
        Type::Array(elem, n) => {
            write!(f, "[{n} x ")?;
            write_type(f, module, *elem)?;
            write!(f, "]")
        }
        Type::Struct(fields) => {
            write!(f, "{{")?;
            for (i, &fl) in fields.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write_type(f, module, fl)?;
            }
            write!(f, "}}")
        }
        Type::Func(ft) => {
            write!(f, "fn(")?;
            for (i, &p) in ft.params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write_type(f, module, p)?;
            }
            if ft.variadic {
                if ft.params.is_empty() {
                    write!(f, "...")?;
                } else {
                    write!(f, ", ...")?;
                }
            }
            write!(f, ") -> ")?;
            write_type(f, module, ft.ret)
        }
    }
}

/// Print a name, either bare (`@foo`) or quoted (`@"foo bar"`).
fn write_name<W: fmt::Write>(f: &mut W, name: &str) -> fmt::Result {
    write!(f, "@")?;
    if is_plain_ident(name) {
        write!(f, "{name}")
    } else {
        write_quoted(f, name)
    }
}

fn write_quoted<W: fmt::Write>(f: &mut W, s: &str) -> fmt::Result {
    write!(f, "\"")?;
    for ch in s.chars() {
        match ch {
            '"' => write!(f, "\\\"")?,
            '\\' => write!(f, "\\\\")?,
            '\n' => write!(f, "\\n")?,
            '\t' => write!(f, "\\t")?,
            _ => write!(f, "{ch}")?,
        }
    }
    write!(f, "\"")
}

fn is_plain_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' || c == '$' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$')
}

fn write_iflags<W: fmt::Write>(f: &mut W, flags: Flags) -> fmt::Result {
    if flags.nsw {
        write!(f, " nsw")?;
    }
    if flags.nuw {
        write!(f, " nuw")?;
    }
    if flags.exact {
        write!(f, " exact")?;
    }
    Ok(())
}

fn write_fastmath<W: fmt::Write>(f: &mut W, fm: FastMath) -> fmt::Result {
    if fm.nnan {
        write!(f, " nnan")?;
    }
    if fm.ninf {
        write!(f, " ninf")?;
    }
    if fm.nsz {
        write!(f, " nsz")?;
    }
    if fm.reassoc {
        write!(f, " reassoc")?;
    }
    if fm.contract {
        write!(f, " contract")?;
    }
    if fm.afn {
        write!(f, " afn")?;
    }
    Ok(())
}

fn binop_name(b: BinOp) -> &'static str {
    match b {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::UDiv => "udiv",
        BinOp::SDiv => "sdiv",
        BinOp::URem => "urem",
        BinOp::SRem => "srem",
        BinOp::And => "and",
        BinOp::Or => "or",
        BinOp::Xor => "xor",
        BinOp::Shl => "shl",
        BinOp::LShr => "lshr",
        BinOp::AShr => "ashr",
        BinOp::FAdd => "fadd",
        BinOp::FSub => "fsub",
        BinOp::FMul => "fmul",
        BinOp::FDiv => "fdiv",
        BinOp::FRem => "frem",
    }
}

fn ipred_name(p: IntPred) -> &'static str {
    match p {
        IntPred::Eq => "eq",
        IntPred::Ne => "ne",
        IntPred::Ugt => "ugt",
        IntPred::Uge => "uge",
        IntPred::Ult => "ult",
        IntPred::Ule => "ule",
        IntPred::Sgt => "sgt",
        IntPred::Sge => "sge",
        IntPred::Slt => "slt",
        IntPred::Sle => "sle",
    }
}

fn fpred_name(p: FloatPred) -> &'static str {
    match p {
        FloatPred::False => "false",
        FloatPred::Oeq => "oeq",
        FloatPred::Ogt => "ogt",
        FloatPred::Oge => "oge",
        FloatPred::Olt => "olt",
        FloatPred::Ole => "ole",
        FloatPred::One => "one",
        FloatPred::Ord => "ord",
        FloatPred::Ueq => "ueq",
        FloatPred::Ugt => "ugt",
        FloatPred::Uge => "uge",
        FloatPred::Ult => "ult",
        FloatPred::Ule => "ule",
        FloatPred::Une => "une",
        FloatPred::Uno => "uno",
        FloatPred::True => "true",
    }
}

fn castop_name(c: CastOp) -> &'static str {
    match c {
        CastOp::Trunc => "trunc",
        CastOp::ZExt => "zext",
        CastOp::SExt => "sext",
        CastOp::FpTrunc => "fptrunc",
        CastOp::FpExt => "fpext",
        CastOp::FpToUi => "fptoui",
        CastOp::FpToSi => "fptosi",
        CastOp::UiToFp => "uitofp",
        CastOp::SiToFp => "sitofp",
        CastOp::PtrToInt => "ptrtoint",
        CastOp::IntToPtr => "inttoptr",
        CastOp::Bitcast => "bitcast",
    }
}

// ===========================================================================
// Lexer
// ===========================================================================

#[derive(Clone, Debug, PartialEq)]
enum TokKind {
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Eq,
    Arrow,
    Ellipsis,
    Percent,
    Caret,
    At,
    Minus,
    Ident(String),
    Num(String),
    Str(String),
    Eof,
}

#[derive(Clone, Debug)]
struct Tok {
    kind: TokKind,
    span: Span,
}

/// Maps a byte offset into the source to its 1-based line number, for attaching
/// source-line debug provenance to parsed IR. Built once per parse.
#[derive(Clone, Debug, Default)]
struct LineIndex {
    /// Byte offset at which each line starts (`line_starts[0] == 0`).
    line_starts: Vec<u32>,
}

impl LineIndex {
    fn new(src: &str) -> LineIndex {
        let mut line_starts = vec![0u32];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push((i + 1) as u32);
            }
        }
        LineIndex { line_starts }
    }

    /// The 1-based line number containing byte `offset`.
    fn line_of(&self, offset: u32) -> u32 {
        // The last line start that is `<= offset`; its index + 1 is the line.
        self.line_starts.partition_point(|&s| s <= offset) as u32
    }
}

fn lex(src: &str, file: FileId) -> Result<Vec<Tok>, Diagnostic> {
    let b = src.as_bytes();
    let n = b.len();
    let mut i = 0usize;
    let mut toks = Vec::new();
    let sp = |s: usize, e: usize| Span::new(file, s as u32, e as u32);

    while i < n {
        let c = b[i];
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
            }
            b';' => {
                while i < n && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'(' => {
                toks.push(Tok { kind: TokKind::LParen, span: sp(i, i + 1) });
                i += 1;
            }
            b')' => {
                toks.push(Tok { kind: TokKind::RParen, span: sp(i, i + 1) });
                i += 1;
            }
            b'{' => {
                toks.push(Tok { kind: TokKind::LBrace, span: sp(i, i + 1) });
                i += 1;
            }
            b'}' => {
                toks.push(Tok { kind: TokKind::RBrace, span: sp(i, i + 1) });
                i += 1;
            }
            b'[' => {
                toks.push(Tok { kind: TokKind::LBracket, span: sp(i, i + 1) });
                i += 1;
            }
            b']' => {
                toks.push(Tok { kind: TokKind::RBracket, span: sp(i, i + 1) });
                i += 1;
            }
            b',' => {
                toks.push(Tok { kind: TokKind::Comma, span: sp(i, i + 1) });
                i += 1;
            }
            b':' => {
                toks.push(Tok { kind: TokKind::Colon, span: sp(i, i + 1) });
                i += 1;
            }
            b'=' => {
                toks.push(Tok { kind: TokKind::Eq, span: sp(i, i + 1) });
                i += 1;
            }
            b'%' => {
                toks.push(Tok { kind: TokKind::Percent, span: sp(i, i + 1) });
                i += 1;
            }
            b'^' => {
                toks.push(Tok { kind: TokKind::Caret, span: sp(i, i + 1) });
                i += 1;
            }
            b'@' => {
                toks.push(Tok { kind: TokKind::At, span: sp(i, i + 1) });
                i += 1;
            }
            b'-' => {
                if i + 1 < n && b[i + 1] == b'>' {
                    toks.push(Tok { kind: TokKind::Arrow, span: sp(i, i + 2) });
                    i += 2;
                } else {
                    toks.push(Tok { kind: TokKind::Minus, span: sp(i, i + 1) });
                    i += 1;
                }
            }
            b'.' => {
                if i + 2 < n && b[i + 1] == b'.' && b[i + 2] == b'.' {
                    toks.push(Tok { kind: TokKind::Ellipsis, span: sp(i, i + 3) });
                    i += 3;
                } else {
                    return Err(Diagnostic::error("unexpected '.'").with_span(sp(i, i + 1)));
                }
            }
            b'"' => {
                let start = i;
                i += 1;
                let mut s = String::new();
                loop {
                    if i >= n {
                        return Err(Diagnostic::error("unterminated string literal")
                            .with_span(sp(start, n)));
                    }
                    match b[i] {
                        b'"' => {
                            i += 1;
                            break;
                        }
                        b'\\' => {
                            if i + 1 >= n {
                                return Err(Diagnostic::error("unterminated escape")
                                    .with_span(sp(start, n)));
                            }
                            let e = b[i + 1];
                            match e {
                                b'"' => s.push('"'),
                                b'\\' => s.push('\\'),
                                b'n' => s.push('\n'),
                                b't' => s.push('\t'),
                                _ => {
                                    return Err(Diagnostic::error(format!(
                                        "unknown escape '\\{}'",
                                        e as char
                                    ))
                                    .with_span(sp(i, i + 2)));
                                }
                            }
                            i += 2;
                        }
                        _ => {
                            // Copy one UTF-8 char; continuation bytes pass through.
                            let ch_start = i;
                            i += 1;
                            while i < n && (b[i] & 0xC0) == 0x80 {
                                i += 1;
                            }
                            s.push_str(&src[ch_start..i]);
                        }
                    }
                }
                toks.push(Tok { kind: TokKind::Str(s), span: sp(start, i) });
            }
            c if c.is_ascii_digit() => {
                let start = i;
                if c == b'0' && i + 1 < n && (b[i + 1] == b'x' || b[i + 1] == b'X') {
                    i += 2;
                    while i < n && b[i].is_ascii_hexdigit() {
                        i += 1;
                    }
                } else {
                    while i < n && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                toks.push(Tok {
                    kind: TokKind::Num(src[start..i].to_string()),
                    span: sp(start, i),
                });
            }
            c if c.is_ascii_alphabetic() || c == b'_' || c == b'$' => {
                let start = i;
                i += 1;
                while i < n {
                    let d = b[i];
                    if d.is_ascii_alphanumeric() || d == b'_' || d == b'.' || d == b'$' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                toks.push(Tok {
                    kind: TokKind::Ident(src[start..i].to_string()),
                    span: sp(start, i),
                });
            }
            _ => {
                return Err(Diagnostic::error(format!("unexpected character '{}'", c as char))
                    .with_span(sp(i, i + 1)));
            }
        }
    }
    toks.push(Tok { kind: TokKind::Eof, span: sp(n, n) });
    Ok(toks)
}

// ===========================================================================
// Parser AST
// ===========================================================================

#[derive(Debug)]
enum ConstAst {
    Int(TypeId, puremp::Int),
    Float(TypeId, FloatBits),
    Null(TypeId),
    Poison(TypeId),
}

#[derive(Debug)]
enum Operand {
    Value(String, Span),
    Const(ConstAst),
    Ref(String, Span),
}

#[derive(Debug)]
enum OpAst {
    Bin(BinOp, Flags, Operand, Operand),
    Unary(UnaryOp, Flags, Operand),
    ICmp(IntPred, Operand, Operand),
    FCmp(FloatPred, Flags, Operand, Operand),
    Cast(CastOp, Operand, TypeId),
    Alloca(TypeId),
    Load(TypeId, u32, Operand),
    Store(TypeId, u32, Operand, Operand),
    PtrAdd(bool, Operand, Operand),
    Select(Operand, Operand, Operand),
    Freeze(Operand),
    Call(Operand, Vec<Operand>, TypeId),
    Ret(Option<Operand>),
    Br(u32, Vec<Operand>),
    CondBr(Operand, u32, Vec<Operand>, u32, Vec<Operand>),
    Switch(Operand, u32, Vec<Operand>, Vec<(puremp::Int, u32, Vec<Operand>)>),
    Unreachable,
}

#[derive(Debug)]
struct InstAst {
    result: Option<String>,
    span: Span,
    op: OpAst,
}

#[derive(Debug)]
struct BlockAst {
    label: u32,
    is_entry: bool,
    entry_span: Span,
    params: Vec<(String, TypeId)>,
    insts: Vec<InstAst>,
}

#[derive(Debug)]
struct BodyAst {
    blocks: Vec<BlockAst>,
}

// ===========================================================================
// Parser
// ===========================================================================

/// Parse `src` (identified by `file`) into a [`Module`], interning names through
/// `syms`. On success the returned module is structurally equal to any module
/// whose printout equals `src`. On failure, returns the collected diagnostics
/// (each with a source [`Span`]).
pub fn parse_module(
    src: &str,
    file: FileId,
    syms: &mut StrInterner,
) -> Result<Module, Vec<Diagnostic>> {
    let toks = lex(src, file).map_err(|d| vec![d])?;
    let mut p = Parser { toks, pos: 0, lines: LineIndex::new(src) };
    p.parse_module(syms).map_err(|d| vec![d])
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    lines: LineIndex,
}

type PResult<T> = Result<T, Diagnostic>;

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos]
    }

    fn peek_kind(&self) -> &TokKind {
        &self.toks[self.pos].kind
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn prev_span(&self) -> Span {
        self.toks[self.pos.saturating_sub(1)].span
    }

    fn span(&self) -> Span {
        self.toks[self.pos].span
    }

    fn err<T>(&self, span: Span, msg: impl Into<String>) -> PResult<T> {
        Err(Diagnostic::error(msg).with_span(span))
    }

    fn expect(&mut self, kind: &TokKind, what: &str) -> PResult<Tok> {
        if &self.peek().kind == kind {
            Ok(self.bump())
        } else {
            self.err(self.span(), format!("expected {what}"))
        }
    }

    fn eat(&mut self, kind: &TokKind) -> bool {
        if &self.peek().kind == kind {
            self.bump();
            true
        } else {
            false
        }
    }

    fn at_ident(&self, s: &str) -> bool {
        matches!(self.peek_kind(), TokKind::Ident(id) if id == s)
    }

    fn eat_ident(&mut self, s: &str) -> bool {
        if self.at_ident(s) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_ident(&mut self, s: &str) -> PResult<Span> {
        if self.at_ident(s) {
            Ok(self.bump().span)
        } else {
            self.err(self.span(), format!("expected `{s}`"))
        }
    }

    fn expect_any_ident(&mut self) -> PResult<(String, Span)> {
        let sp = self.span();
        if let TokKind::Ident(id) = self.peek_kind().clone() {
            self.bump();
            Ok((id, sp))
        } else {
            self.err(sp, "expected an identifier")
        }
    }

    /// A `@`-prefixed name: a bare identifier or a quoted string.
    fn parse_name(&mut self) -> PResult<String> {
        self.expect(&TokKind::At, "`@`")?;
        match self.peek_kind().clone() {
            TokKind::Ident(id) => {
                self.bump();
                Ok(id)
            }
            TokKind::Str(s) => {
                self.bump();
                Ok(s)
            }
            TokKind::Num(nm) => {
                self.bump();
                Ok(nm)
            }
            _ => self.err(self.span(), "expected a name after `@`"),
        }
    }

    /// A `%`-prefixed value name (identifier or number), returned as a string key.
    fn parse_value_name(&mut self) -> PResult<(String, Span)> {
        let sp = self.span();
        self.expect(&TokKind::Percent, "`%`")?;
        match self.peek_kind().clone() {
            TokKind::Ident(id) => {
                self.bump();
                Ok((id, sp.merge(self.prev_span())))
            }
            TokKind::Num(nm) => {
                self.bump();
                Ok((nm, sp.merge(self.prev_span())))
            }
            _ => self.err(self.span(), "expected a value name after `%`"),
        }
    }

    fn parse_u32(&mut self) -> PResult<u32> {
        let sp = self.span();
        if let TokKind::Num(nm) = self.peek_kind().clone() {
            self.bump();
            nm.parse::<u32>().map_err(|_| Diagnostic::error("invalid integer").with_span(sp))
        } else {
            self.err(sp, "expected an integer")
        }
    }

    fn parse_u64(&mut self) -> PResult<u64> {
        let sp = self.span();
        if let TokKind::Num(nm) = self.peek_kind().clone() {
            self.bump();
            nm.parse::<u64>().map_err(|_| Diagnostic::error("invalid integer").with_span(sp))
        } else {
            self.err(sp, "expected an integer")
        }
    }

    // --- module ------------------------------------------------------------

    fn parse_module(&mut self, syms: &mut StrInterner) -> PResult<Module> {
        let mut module = Module::new("");
        self.expect_ident("module")?;
        let name_sp = self.span();
        let name = match self.peek_kind().clone() {
            TokKind::Str(s) => {
                self.bump();
                s
            }
            _ => return self.err(name_sp, "expected a module name string"),
        };
        module.name = name;

        let mut func_names: HashMap<String, FuncId> = HashMap::new();
        let mut global_names: HashMap<String, GlobalId> = HashMap::new();
        let mut pending: Vec<(FuncId, BodyAst, u32)> = Vec::new();

        loop {
            match self.peek_kind() {
                TokKind::Eof => break,
                TokKind::Ident(id) if id == "global" => {
                    self.parse_global(&mut module, syms, &mut global_names)?;
                }
                TokKind::Ident(id) if id == "func" => {
                    let (fid, body, decl_line) =
                        self.parse_func(&mut module, syms, &mut func_names)?;
                    if let Some(body) = body {
                        pending.push((fid, body, decl_line));
                    }
                }
                _ => {
                    return self
                        .err(self.span(), "expected a top-level item (`global` or `func`)");
                }
            }
        }

        for (fid, body, decl_line) in pending {
            lower_body(&mut module, fid, &body, decl_line, &self.lines, &func_names, &global_names)?;
        }
        Ok(module)
    }

    fn parse_global(
        &mut self,
        module: &mut Module,
        syms: &mut StrInterner,
        global_names: &mut HashMap<String, GlobalId>,
    ) -> PResult<()> {
        self.expect_ident("global")?;
        let name = self.parse_name()?;
        self.expect(&TokKind::Colon, "`:`")?;
        let ty = self.parse_type(module)?;
        let init =
            if self.eat(&TokKind::Eq) { Some(self.parse_const(module)?) } else { None };
        let sym = syms.intern(&name);
        let gid = module.add_global(Global { name: sym, ty, init });
        global_names.insert(name, gid);
        Ok(())
    }

    fn parse_func(
        &mut self,
        module: &mut Module,
        syms: &mut StrInterner,
        func_names: &mut HashMap<String, FuncId>,
    ) -> PResult<(FuncId, Option<BodyAst>, u32)> {
        let func_kw = self.expect_ident("func")?;
        let decl_line = self.lines.line_of(func_kw.start);
        let name = self.parse_name()?;
        let (params, ret, variadic) = self.parse_fn_sig(module)?;
        let sig = module.types_mut().func(params, ret, variadic);
        let sym = syms.intern(&name);
        let fid = module.declare_function(sym, sig);
        func_names.insert(name, fid);

        let body = if matches!(self.peek_kind(), TokKind::LBrace) {
            Some(self.parse_body(module)?)
        } else {
            None
        };
        Ok((fid, body, decl_line))
    }

    fn parse_fn_sig(&mut self, module: &mut Module) -> PResult<(Vec<TypeId>, TypeId, bool)> {
        self.expect(&TokKind::LParen, "`(`")?;
        let mut params = Vec::new();
        let mut variadic = false;
        if !matches!(self.peek_kind(), TokKind::RParen) {
            loop {
                if matches!(self.peek_kind(), TokKind::Ellipsis) {
                    self.bump();
                    variadic = true;
                    break;
                }
                params.push(self.parse_type(module)?);
                if self.eat(&TokKind::Comma) {
                    continue;
                }
                break;
            }
        }
        self.expect(&TokKind::RParen, "`)`")?;
        self.expect(&TokKind::Arrow, "`->`")?;
        let ret = self.parse_type(module)?;
        Ok((params, ret, variadic))
    }

    fn parse_body(&mut self, module: &mut Module) -> PResult<BodyAst> {
        self.expect(&TokKind::LBrace, "`{`")?;
        let mut blocks = Vec::new();
        while !matches!(self.peek_kind(), TokKind::RBrace) {
            if matches!(self.peek_kind(), TokKind::Eof) {
                return self.err(self.span(), "unexpected end of input in function body");
            }
            blocks.push(self.parse_block(module)?);
        }
        self.expect(&TokKind::RBrace, "`}`")?;
        Ok(BodyAst { blocks })
    }

    fn parse_block(&mut self, module: &mut Module) -> PResult<BlockAst> {
        let entry_span = self.span();
        let is_entry = self.eat_ident("entry");
        self.expect(&TokKind::Caret, "`^`")?;
        let label = self.parse_u32()?;
        let mut params = Vec::new();
        if self.eat(&TokKind::LParen) {
            if !matches!(self.peek_kind(), TokKind::RParen) {
                loop {
                    let (pname, _) = self.parse_value_name()?;
                    self.expect(&TokKind::Colon, "`:`")?;
                    let ty = self.parse_type(module)?;
                    params.push((pname, ty));
                    if self.eat(&TokKind::Comma) {
                        continue;
                    }
                    break;
                }
            }
            self.expect(&TokKind::RParen, "`)`")?;
        }
        self.expect(&TokKind::Colon, "`:`")?;

        let mut insts = Vec::new();
        while !matches!(self.peek_kind(), TokKind::Caret | TokKind::RBrace | TokKind::Eof) {
            insts.push(self.parse_inst(module)?);
        }
        Ok(BlockAst { label, is_entry, entry_span, params, insts })
    }

    fn parse_inst(&mut self, module: &mut Module) -> PResult<InstAst> {
        let start = self.span();
        let result = if matches!(self.peek_kind(), TokKind::Percent) {
            let (name, _) = self.parse_value_name()?;
            self.expect(&TokKind::Eq, "`=`")?;
            Some(name)
        } else {
            None
        };
        let (opname, op_sp) = self.expect_any_ident()?;
        let op = self.parse_op(module, &opname, op_sp)?;
        Ok(InstAst { result, span: start.merge(self.prev_span()), op })
    }

    fn parse_op(&mut self, module: &mut Module, opname: &str, op_sp: Span) -> PResult<OpAst> {
        if let Some(b) = binop_from_name(opname) {
            let flags =
                if b.is_float() { self.parse_fastmath() } else { self.parse_iflags() };
            let a = self.parse_operand(module)?;
            self.expect(&TokKind::Comma, "`,`")?;
            let c = self.parse_operand(module)?;
            self.expect(&TokKind::Colon, "`:`")?;
            let _ty = self.parse_type(module)?;
            return Ok(OpAst::Bin(b, flags, a, c));
        }
        if let Some(c) = castop_from_name(opname) {
            let a = self.parse_operand(module)?;
            self.expect(&TokKind::Colon, "`:`")?;
            let ty = self.parse_type(module)?;
            return Ok(OpAst::Cast(c, a, ty));
        }
        match opname {
            "fneg" => {
                let flags = self.parse_fastmath();
                let a = self.parse_operand(module)?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ty = self.parse_type(module)?;
                Ok(OpAst::Unary(UnaryOp::FNeg, flags, a))
            }
            "icmp" => {
                let (pn, psp) = self.expect_any_ident()?;
                let pred = ipred_from_name(&pn)
                    .ok_or_else(|| Diagnostic::error("unknown icmp predicate").with_span(psp))?;
                let a = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let c = self.parse_operand(module)?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ty = self.parse_type(module)?;
                Ok(OpAst::ICmp(pred, a, c))
            }
            "fcmp" => {
                let (pn, psp) = self.expect_any_ident()?;
                let pred = fpred_from_name(&pn)
                    .ok_or_else(|| Diagnostic::error("unknown fcmp predicate").with_span(psp))?;
                let flags = self.parse_fastmath();
                let a = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let c = self.parse_operand(module)?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ty = self.parse_type(module)?;
                Ok(OpAst::FCmp(pred, flags, a, c))
            }
            "alloca" => {
                let elem = self.parse_type(module)?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ptr = self.parse_type(module)?;
                Ok(OpAst::Alloca(elem))
            }
            "load" => {
                let ptr = self.parse_operand(module)?;
                self.expect_ident("align")?;
                let align = self.parse_u32()?;
                self.expect(&TokKind::Colon, "`:`")?;
                let ty = self.parse_type(module)?;
                Ok(OpAst::Load(ty, align, ptr))
            }
            "store" => {
                let val = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let ptr = self.parse_operand(module)?;
                self.expect_ident("align")?;
                let align = self.parse_u32()?;
                self.expect(&TokKind::Colon, "`:`")?;
                let ty = self.parse_type(module)?;
                Ok(OpAst::Store(ty, align, val, ptr))
            }
            "ptr_add" => {
                let inbounds = self.eat_ident("inbounds");
                let base = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let off = self.parse_operand(module)?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ptr = self.parse_type(module)?;
                Ok(OpAst::PtrAdd(inbounds, base, off))
            }
            "select" => {
                let cond = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let t = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let ff = self.parse_operand(module)?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ty = self.parse_type(module)?;
                Ok(OpAst::Select(cond, t, ff))
            }
            "freeze" => {
                let v = self.parse_operand(module)?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ty = self.parse_type(module)?;
                Ok(OpAst::Freeze(v))
            }
            "call" => {
                let callee = self.parse_operand(module)?;
                self.expect(&TokKind::LParen, "`(`")?;
                let mut args = Vec::new();
                if !matches!(self.peek_kind(), TokKind::RParen) {
                    loop {
                        args.push(self.parse_operand(module)?);
                        if self.eat(&TokKind::Comma) {
                            continue;
                        }
                        break;
                    }
                }
                self.expect(&TokKind::RParen, "`)`")?;
                self.expect(&TokKind::Colon, "`:`")?;
                let ret = self.parse_type(module)?;
                Ok(OpAst::Call(callee, args, ret))
            }
            "ret" => {
                if matches!(self.peek_kind(), TokKind::Caret | TokKind::RBrace | TokKind::Eof) {
                    Ok(OpAst::Ret(None))
                } else {
                    Ok(OpAst::Ret(Some(self.parse_operand(module)?)))
                }
            }
            "br" => {
                let (label, args) = self.parse_target(module)?;
                Ok(OpAst::Br(label, args))
            }
            "cond_br" => {
                let cond = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let (tl, ta) = self.parse_target(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let (fl, fa) = self.parse_target(module)?;
                Ok(OpAst::CondBr(cond, tl, ta, fl, fa))
            }
            "switch" => {
                let cond = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let (dl, da) = self.parse_target(module)?;
                self.expect(&TokKind::LBracket, "`[`")?;
                let mut cases = Vec::new();
                if !matches!(self.peek_kind(), TokKind::RBracket) {
                    loop {
                        let value = self.parse_signed_int()?;
                        self.expect(&TokKind::Colon, "`:`")?;
                        let (cl, ca) = self.parse_target(module)?;
                        cases.push((value, cl, ca));
                        if self.eat(&TokKind::Comma) {
                            continue;
                        }
                        break;
                    }
                }
                self.expect(&TokKind::RBracket, "`]`")?;
                Ok(OpAst::Switch(cond, dl, da, cases))
            }
            "unreachable" => Ok(OpAst::Unreachable),
            _ => self.err(op_sp, format!("unknown opcode `{opname}`")),
        }
    }

    fn parse_target(&mut self, module: &mut Module) -> PResult<(u32, Vec<Operand>)> {
        self.expect(&TokKind::Caret, "`^`")?;
        let label = self.parse_u32()?;
        let mut args = Vec::new();
        if self.eat(&TokKind::LParen) {
            if !matches!(self.peek_kind(), TokKind::RParen) {
                loop {
                    args.push(self.parse_operand(module)?);
                    if self.eat(&TokKind::Comma) {
                        continue;
                    }
                    break;
                }
            }
            self.expect(&TokKind::RParen, "`)`")?;
        }
        Ok((label, args))
    }

    fn parse_iflags(&mut self) -> Flags {
        let mut fl = Flags::NONE;
        loop {
            if self.eat_ident("nsw") {
                fl.nsw = true;
            } else if self.eat_ident("nuw") {
                fl.nuw = true;
            } else if self.eat_ident("exact") {
                fl.exact = true;
            } else {
                break;
            }
        }
        fl
    }

    fn parse_fastmath(&mut self) -> Flags {
        let mut fm = FastMath::default();
        loop {
            if self.eat_ident("nnan") {
                fm.nnan = true;
            } else if self.eat_ident("ninf") {
                fm.ninf = true;
            } else if self.eat_ident("nsz") {
                fm.nsz = true;
            } else if self.eat_ident("reassoc") {
                fm.reassoc = true;
            } else if self.eat_ident("contract") {
                fm.contract = true;
            } else if self.eat_ident("afn") {
                fm.afn = true;
            } else {
                break;
            }
        }
        Flags::fast(fm)
    }

    fn parse_operand(&mut self, module: &mut Module) -> PResult<Operand> {
        match self.peek_kind() {
            TokKind::Percent => {
                let (name, sp) = self.parse_value_name()?;
                Ok(Operand::Value(name, sp))
            }
            TokKind::At => {
                let sp = self.span();
                let name = self.parse_name()?;
                Ok(Operand::Ref(name, sp))
            }
            _ => Ok(Operand::Const(self.parse_const_operand(module)?)),
        }
    }

    fn parse_const_operand(&mut self, module: &mut Module) -> PResult<ConstAst> {
        let ty_sp = self.span();
        let ty = self.parse_type(module)?;
        if self.eat_ident("null") {
            return Ok(ConstAst::Null(ty));
        }
        if self.eat_ident("poison") {
            return Ok(ConstAst::Poison(ty));
        }
        if matches!(self.peek_kind(), TokKind::LParen) {
            return self.err(
                self.span(),
                "aggregate constants are only allowed as global initializers",
            );
        }
        match module.types().get(ty).clone() {
            Type::Int(_) => {
                let value = self.parse_signed_int()?;
                Ok(ConstAst::Int(ty, value))
            }
            Type::Float(k) => {
                let bits = self.parse_float_bits(k)?;
                Ok(ConstAst::Float(ty, bits))
            }
            _ => self.err(ty_sp, "expected an integer or float constant"),
        }
    }

    /// Parse a constant used as a global initializer, interning it into the
    /// module and returning its [`ConstId`]. Unlike operand constants, these may
    /// be aggregates.
    fn parse_const(&mut self, module: &mut Module) -> PResult<ConstId> {
        let ty_sp = self.span();
        let ty = self.parse_type(module)?;
        if self.eat_ident("null") {
            return Ok(module.intern_const(Const::Null(ty)));
        }
        if self.eat_ident("poison") {
            return Ok(module.intern_const(Const::Poison(ty)));
        }
        if self.eat(&TokKind::LParen) {
            let mut elems = Vec::new();
            if !matches!(self.peek_kind(), TokKind::RParen) {
                loop {
                    elems.push(self.parse_const(module)?);
                    if self.eat(&TokKind::Comma) {
                        continue;
                    }
                    break;
                }
            }
            self.expect(&TokKind::RParen, "`)`")?;
            return Ok(module.intern_const(Const::Aggregate { ty, elems }));
        }
        match module.types().get(ty).clone() {
            Type::Int(_) => {
                let value = self.parse_signed_int()?;
                Ok(module.intern_const(Const::Int { ty, value }))
            }
            Type::Float(k) => {
                let bits = self.parse_float_bits(k)?;
                Ok(module.intern_const(Const::Float { ty, bits }))
            }
            _ => self.err(ty_sp, "expected a constant payload"),
        }
    }

    fn parse_signed_int(&mut self) -> PResult<puremp::Int> {
        let start = self.span();
        let neg = self.eat(&TokKind::Minus);
        let sp = self.span();
        let TokKind::Num(nm) = self.peek_kind().clone() else {
            return self.err(sp, "expected an integer literal");
        };
        self.bump();
        let text = if neg { format!("-{nm}") } else { nm };
        puremp::Int::from_str_radix(&text, 10)
            .map_err(|_| Diagnostic::error("invalid integer literal").with_span(start.merge(sp)))
    }

    fn parse_float_bits(&mut self, k: FloatKind) -> PResult<FloatBits> {
        let sp = self.span();
        let TokKind::Num(nm) = self.peek_kind().clone() else {
            return self.err(sp, "expected a `0x` float bit pattern");
        };
        self.bump();
        let hex = nm.strip_prefix("0x").or_else(|| nm.strip_prefix("0X")).ok_or_else(|| {
            Diagnostic::error("float constants must be written as `0x<bits>`").with_span(sp)
        })?;
        let raw = u64::from_str_radix(hex, 16)
            .map_err(|_| Diagnostic::error("invalid float bit pattern").with_span(sp))?;
        match k {
            FloatKind::F16 => {
                if raw > u64::from(u16::MAX) {
                    return self.err(sp, "f16 bit pattern out of range");
                }
                Ok(FloatBits::F16(raw as u16))
            }
            FloatKind::F32 => {
                if raw > u64::from(u32::MAX) {
                    return self.err(sp, "f32 bit pattern out of range");
                }
                Ok(FloatBits::F32(raw as u32))
            }
            FloatKind::F64 => Ok(FloatBits::F64(raw)),
        }
    }

    fn parse_type(&mut self, module: &mut Module) -> PResult<TypeId> {
        let sp = self.span();
        match self.peek_kind().clone() {
            TokKind::Ident(id) => {
                self.bump();
                match id.as_str() {
                    "void" => Ok(module.types_mut().void()),
                    "ptr" => Ok(module.types_mut().ptr()),
                    "f16" => Ok(module.types_mut().float(FloatKind::F16)),
                    "f32" => Ok(module.types_mut().float(FloatKind::F32)),
                    "f64" => Ok(module.types_mut().float(FloatKind::F64)),
                    "fn" => {
                        let (params, ret, variadic) = self.parse_fn_sig(module)?;
                        Ok(module.types_mut().func(params, ret, variadic))
                    }
                    other => {
                        if let Some(rest) = other.strip_prefix('i')
                            && !rest.is_empty()
                            && rest.bytes().all(|b| b.is_ascii_digit())
                            && let Ok(w) = rest.parse::<u32>()
                        {
                            return Ok(module.types_mut().int(w));
                        }
                        self.err(sp, format!("unknown type `{other}`"))
                    }
                }
            }
            TokKind::LBracket => {
                self.bump();
                let len = self.parse_u64()?;
                self.expect_ident("x")?;
                let elem = self.parse_type(module)?;
                self.expect(&TokKind::RBracket, "`]`")?;
                Ok(module.types_mut().array(elem, len))
            }
            TokKind::LBrace => {
                self.bump();
                let mut fields = Vec::new();
                if !matches!(self.peek_kind(), TokKind::RBrace) {
                    loop {
                        fields.push(self.parse_type(module)?);
                        if self.eat(&TokKind::Comma) {
                            continue;
                        }
                        break;
                    }
                }
                self.expect(&TokKind::RBrace, "`}`")?;
                Ok(module.types_mut().struct_(fields))
            }
            _ => self.err(sp, "expected a type"),
        }
    }
}

// ===========================================================================
// Lowering (AST -> IR via the builder)
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn lower_body(
    module: &mut Module,
    fid: FuncId,
    body: &BodyAst,
    decl_line: u32,
    lines: &LineIndex,
    func_names: &HashMap<String, FuncId>,
    global_names: &HashMap<String, GlobalId>,
) -> PResult<()> {
    // Validate exactly one entry block.
    let entry_count = body.blocks.iter().filter(|b| b.is_entry).count();
    if entry_count != 1 {
        let sp = body
            .blocks
            .first()
            .map(|b| b.entry_span)
            .unwrap_or_else(|| Span::point(FileId::new(0), 0));
        return Err(Diagnostic::error(format!(
            "function body must have exactly one `entry` block, found {entry_count}"
        ))
        .with_span(sp));
    }

    let mut b = module.build(fid);
    b.set_decl_line(decl_line);
    let mut label_to_block: HashMap<u32, BlockId> = HashMap::new();
    let mut names: HashMap<String, ValueId> = HashMap::new();

    // Sub-pass 1: create every block (in ascending label order so block ids are
    // assigned deterministically) and bind its parameter names.
    let mut order: Vec<&BlockAst> = body.blocks.iter().collect();
    order.sort_by_key(|blk| blk.label);
    for blk in order {
        let bid = if blk.is_entry {
            b.create_entry_block()
        } else {
            let tys: Vec<TypeId> = blk.params.iter().map(|(_, t)| *t).collect();
            b.create_block(&tys)
        };
        label_to_block.insert(blk.label, bid);
        let pids = b.block_params(bid).to_vec();
        if pids.len() != blk.params.len() {
            return Err(Diagnostic::error(format!(
                "block ^{} declares {} parameters but its signature has {}",
                blk.label,
                blk.params.len(),
                pids.len()
            ))
            .with_span(blk.entry_span));
        }
        for ((pname, _), pid) in blk.params.iter().zip(pids) {
            names.insert(pname.clone(), pid);
        }
    }

    // Sub-pass 2: emit instructions per block.
    for blk in &body.blocks {
        let bid = label_to_block[&blk.label];
        b.switch_to(bid);
        for inst in &blk.insts {
            b.set_line(lines.line_of(inst.span.start));
            let result =
                emit_inst(&mut b, inst, &names, &label_to_block, func_names, global_names)?;
            if let Some(rname) = &inst.result {
                match result {
                    Some(v) => {
                        names.insert(rname.clone(), v);
                    }
                    None => {
                        return Err(Diagnostic::error(
                            "instruction has a result name but produces no value",
                        )
                        .with_span(inst.span));
                    }
                }
            }
        }
    }
    Ok(())
}

fn emit_inst(
    b: &mut FunctionBuilder<'_>,
    inst: &InstAst,
    names: &HashMap<String, ValueId>,
    labels: &HashMap<u32, BlockId>,
    func_names: &HashMap<String, FuncId>,
    global_names: &HashMap<String, GlobalId>,
) -> PResult<Option<ValueId>> {
    let block_of = |label: u32, sp: Span| -> PResult<BlockId> {
        labels
            .get(&label)
            .copied()
            .ok_or_else(|| Diagnostic::error(format!("unknown block ^{label}")).with_span(sp))
    };

    Ok(match &inst.op {
        OpAst::Bin(op, flags, a, c) => {
            let lhs = resolve_operand(b, a, names, func_names, global_names)?;
            let rhs = resolve_operand(b, c, names, func_names, global_names)?;
            Some(b.bin(*op, lhs, rhs, *flags))
        }
        OpAst::Unary(UnaryOp::FNeg, flags, a) => {
            let v = resolve_operand(b, a, names, func_names, global_names)?;
            Some(b.fneg(v, *flags))
        }
        OpAst::ICmp(pred, a, c) => {
            let lhs = resolve_operand(b, a, names, func_names, global_names)?;
            let rhs = resolve_operand(b, c, names, func_names, global_names)?;
            Some(b.icmp(*pred, lhs, rhs))
        }
        OpAst::FCmp(pred, flags, a, c) => {
            let lhs = resolve_operand(b, a, names, func_names, global_names)?;
            let rhs = resolve_operand(b, c, names, func_names, global_names)?;
            Some(b.fcmp(*pred, lhs, rhs, *flags))
        }
        OpAst::Cast(op, a, ty) => {
            let v = resolve_operand(b, a, names, func_names, global_names)?;
            Some(b.cast(*op, v, *ty))
        }
        OpAst::Alloca(elem) => Some(b.alloca(*elem)),
        OpAst::Load(ty, align, ptr) => {
            let p = resolve_operand(b, ptr, names, func_names, global_names)?;
            Some(b.load(*ty, p, *align))
        }
        OpAst::Store(ty, align, val, ptr) => {
            let v = resolve_operand(b, val, names, func_names, global_names)?;
            let p = resolve_operand(b, ptr, names, func_names, global_names)?;
            b.store(*ty, p, v, *align);
            None
        }
        OpAst::PtrAdd(inbounds, base, off) => {
            let ba = resolve_operand(b, base, names, func_names, global_names)?;
            let of = resolve_operand(b, off, names, func_names, global_names)?;
            Some(b.ptr_add(ba, of, *inbounds))
        }
        OpAst::Select(cond, t, ff) => {
            let c = resolve_operand(b, cond, names, func_names, global_names)?;
            let tv = resolve_operand(b, t, names, func_names, global_names)?;
            let fv = resolve_operand(b, ff, names, func_names, global_names)?;
            Some(b.select(c, tv, fv))
        }
        OpAst::Freeze(v) => {
            let val = resolve_operand(b, v, names, func_names, global_names)?;
            Some(b.freeze(val))
        }
        OpAst::Call(callee, args, ret) => {
            let cv = resolve_operand(b, callee, names, func_names, global_names)?;
            let mut avs = Vec::with_capacity(args.len());
            for a in args {
                avs.push(resolve_operand(b, a, names, func_names, global_names)?);
            }
            b.call(cv, &avs, *ret)
        }
        OpAst::Ret(v) => {
            let rv = match v {
                Some(op) => Some(resolve_operand(b, op, names, func_names, global_names)?),
                None => None,
            };
            b.ret(rv);
            None
        }
        OpAst::Br(label, args) => {
            let target = block_of(*label, inst.span)?;
            let mut avs = Vec::with_capacity(args.len());
            for a in args {
                avs.push(resolve_operand(b, a, names, func_names, global_names)?);
            }
            b.br(target, &avs);
            None
        }
        OpAst::CondBr(cond, tl, ta, fl, fa) => {
            let c = resolve_operand(b, cond, names, func_names, global_names)?;
            let tblock = block_of(*tl, inst.span)?;
            let fblock = block_of(*fl, inst.span)?;
            let mut tavs = Vec::with_capacity(ta.len());
            for a in ta {
                tavs.push(resolve_operand(b, a, names, func_names, global_names)?);
            }
            let mut favs = Vec::with_capacity(fa.len());
            for a in fa {
                favs.push(resolve_operand(b, a, names, func_names, global_names)?);
            }
            b.cond_br(c, tblock, &tavs, fblock, &favs);
            None
        }
        OpAst::Switch(cond, dl, da, cases) => {
            let c = resolve_operand(b, cond, names, func_names, global_names)?;
            let default = block_of(*dl, inst.span)?;
            let mut davs = Vec::with_capacity(da.len());
            for a in da {
                davs.push(resolve_operand(b, a, names, func_names, global_names)?);
            }
            let mut case_data = Vec::with_capacity(cases.len());
            for (value, label, args) in cases {
                let target = block_of(*label, inst.span)?;
                let mut avs = Vec::with_capacity(args.len());
                for a in args {
                    avs.push(resolve_operand(b, a, names, func_names, global_names)?);
                }
                case_data.push((value.clone(), target, avs));
            }
            b.switch(c, default, &davs, case_data);
            None
        }
        OpAst::Unreachable => {
            b.unreachable();
            None
        }
    })
}

fn resolve_operand(
    b: &mut FunctionBuilder<'_>,
    op: &Operand,
    names: &HashMap<String, ValueId>,
    func_names: &HashMap<String, FuncId>,
    global_names: &HashMap<String, GlobalId>,
) -> PResult<ValueId> {
    match op {
        Operand::Value(name, sp) => names
            .get(name)
            .copied()
            .ok_or_else(|| Diagnostic::error(format!("undefined value `%{name}`")).with_span(*sp)),
        Operand::Const(c) => Ok(match c {
            ConstAst::Int(ty, v) => b.const_int(*ty, v.clone()),
            ConstAst::Float(ty, bits) => b.const_float(*ty, *bits),
            ConstAst::Null(ty) => b.null(*ty),
            ConstAst::Poison(ty) => b.poison(*ty),
        }),
        Operand::Ref(name, sp) => {
            if let Some(&f) = func_names.get(name) {
                Ok(b.func_ref(f))
            } else if let Some(&g) = global_names.get(name) {
                Ok(b.global_ref(g))
            } else {
                Err(Diagnostic::error(format!("unknown reference `@{name}`")).with_span(*sp))
            }
        }
    }
}

fn binop_from_name(s: &str) -> Option<BinOp> {
    Some(match s {
        "add" => BinOp::Add,
        "sub" => BinOp::Sub,
        "mul" => BinOp::Mul,
        "udiv" => BinOp::UDiv,
        "sdiv" => BinOp::SDiv,
        "urem" => BinOp::URem,
        "srem" => BinOp::SRem,
        "and" => BinOp::And,
        "or" => BinOp::Or,
        "xor" => BinOp::Xor,
        "shl" => BinOp::Shl,
        "lshr" => BinOp::LShr,
        "ashr" => BinOp::AShr,
        "fadd" => BinOp::FAdd,
        "fsub" => BinOp::FSub,
        "fmul" => BinOp::FMul,
        "fdiv" => BinOp::FDiv,
        "frem" => BinOp::FRem,
        _ => return None,
    })
}

fn castop_from_name(s: &str) -> Option<CastOp> {
    Some(match s {
        "trunc" => CastOp::Trunc,
        "zext" => CastOp::ZExt,
        "sext" => CastOp::SExt,
        "fptrunc" => CastOp::FpTrunc,
        "fpext" => CastOp::FpExt,
        "fptoui" => CastOp::FpToUi,
        "fptosi" => CastOp::FpToSi,
        "uitofp" => CastOp::UiToFp,
        "sitofp" => CastOp::SiToFp,
        "ptrtoint" => CastOp::PtrToInt,
        "inttoptr" => CastOp::IntToPtr,
        "bitcast" => CastOp::Bitcast,
        _ => return None,
    })
}

fn ipred_from_name(s: &str) -> Option<IntPred> {
    Some(match s {
        "eq" => IntPred::Eq,
        "ne" => IntPred::Ne,
        "ugt" => IntPred::Ugt,
        "uge" => IntPred::Uge,
        "ult" => IntPred::Ult,
        "ule" => IntPred::Ule,
        "sgt" => IntPred::Sgt,
        "sge" => IntPred::Sge,
        "slt" => IntPred::Slt,
        "sle" => IntPred::Sle,
        _ => return None,
    })
}

fn fpred_from_name(s: &str) -> Option<FloatPred> {
    Some(match s {
        "false" => FloatPred::False,
        "oeq" => FloatPred::Oeq,
        "ogt" => FloatPred::Ogt,
        "oge" => FloatPred::Oge,
        "olt" => FloatPred::Olt,
        "ole" => FloatPred::Ole,
        "one" => FloatPred::One,
        "ord" => FloatPred::Ord,
        "ueq" => FloatPred::Ueq,
        "ugt" => FloatPred::Ugt,
        "uge" => FloatPred::Uge,
        "ult" => FloatPred::Ult,
        "ule" => FloatPred::Ule,
        "une" => FloatPred::Une,
        "uno" => FloatPred::Uno,
        "true" => FloatPred::True,
        _ => return None,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::inst::{BinOp, Flags, IntPred};
    use crate::ir::value::FloatBits;

    fn file() -> FileId {
        FileId::new(0)
    }

    /// Round-trip a module: printing the parse of a print must reproduce the
    /// original print byte-for-byte. Because the printer is canonical (its output
    /// is a pure function of module structure), this equality *is* a structural
    /// equality between the original module and the re-parsed one. Returns the
    /// re-parsed module for any additional structural assertions.
    fn round_trip(module: &Module, syms: &mut StrInterner) -> Module {
        let text1 = print_module(module, syms);
        let parsed = match parse_module(&text1, file(), syms) {
            Ok(m) => m,
            Err(diags) => panic!("parse failed: {diags:?}\n---\n{text1}"),
        };
        let text2 = print_module(&parsed, syms);
        assert_eq!(text1, text2, "round-trip not idempotent\n--- first ---\n{text1}\n--- second ---\n{text2}");
        parsed
    }

    #[test]
    fn empty_module() {
        let syms = StrInterner::new();
        let m = Module::new("empty");
        let text = print_module(&m, &syms);
        assert_eq!(text, "module \"empty\"\n");
        let mut syms2 = StrInterner::new();
        let m2 = parse_module(&text, file(), &mut syms2).expect("parse");
        assert_eq!(m2.name, "empty");
        assert_eq!(m2.functions().count(), 0);
    }

    #[test]
    fn arithmetic_and_flags() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("arith");
        let i32_ = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32_, i32_], i32_, false);
        let f = m.declare_function(syms.intern("f"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let x = b.param(e, 0);
            let y = b.param(e, 1);
            let s = b.add(x, y, Flags::nsw());
            let c = b.const_i64(i32_, 7);
            let s2 = b.mul(s, c, Flags { nsw: true, nuw: true, ..Flags::NONE });
            b.ret(Some(s2));
        }
        let parsed = round_trip(&m, &mut syms);
        assert_eq!(parsed.functions().count(), 1);
        assert_eq!(parsed.function(FuncId::from_index(0)).block_count(), 1);
    }

    #[test]
    fn loop_with_back_edge() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("loops");
        let i64_ = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![i64_], i64_, false);
        let f = m.declare_function(syms.intern("sum"), sig);
        {
            let mut b = m.build(f);
            let entry = b.create_entry_block();
            let n = b.param(entry, 0);
            let header = b.create_block(&[i64_, i64_]);
            let body = b.create_block(&[i64_, i64_]);
            let exit = b.create_block(&[i64_]);

            b.switch_to(entry);
            let zero = b.const_i64(i64_, 0);
            b.br(header, &[zero, zero]);

            b.switch_to(header);
            let acc = b.param(header, 0);
            let i = b.param(header, 1);
            let cond = b.icmp(IntPred::Slt, i, n);
            b.cond_br(cond, body, &[acc, i], exit, &[acc]);

            b.switch_to(body);
            let bacc = b.param(body, 0);
            let bi = b.param(body, 1);
            let new_acc = b.add(bacc, bi, Flags::nsw());
            let one = b.const_i64(i64_, 1);
            let new_i = b.add(bi, one, Flags::nsw());
            b.br(header, &[new_acc, new_i]);

            b.switch_to(exit);
            let result = b.param(exit, 0);
            b.ret(Some(result));
        }
        let parsed = round_trip(&m, &mut syms);
        assert_eq!(parsed.function(FuncId::from_index(0)).block_count(), 4);
    }

    #[test]
    fn call_select_icmp() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("cs");
        let i64_ = m.types_mut().int(64);
        let unary = m.types_mut().func(vec![i64_], i64_, false);
        let binary = m.types_mut().func(vec![i64_, i64_], i64_, false);
        let g = m.declare_function(syms.intern("g"), unary);
        let f = m.declare_function(syms.intern("f"), binary);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let a = b.param(e, 0);
            let bv = b.param(e, 1);
            let gref = b.func_ref(g);
            let c = b.call(gref, &[a], i64_).expect("call result");
            let cond = b.icmp(IntPred::Sgt, a, bv);
            let sel = b.select(cond, c, bv);
            b.ret(Some(sel));
        }
        let parsed = round_trip(&m, &mut syms);
        // g stays a declaration.
        assert!(parsed.function(FuncId::from_index(0)).is_declaration());
        assert!(!parsed.function(FuncId::from_index(1)).is_declaration());
    }

    #[test]
    fn switch_and_wide_constants() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("sw");
        let i128_ = m.types_mut().int(128);
        let i32_ = m.types_mut().int(32);
        let sig = m.types_mut().func(vec![i32_], i128_, false);
        let f = m.declare_function(syms.intern("classify"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let x = b.param(e, 0);
            let a = b.create_block(&[]);
            let bl = b.create_block(&[]);
            let d = b.create_block(&[]);

            b.switch_to(e);
            let big = puremp::Int::from_i64(2).pow(100);
            let neg = puremp::Int::from_i64(-5);
            b.switch(
                x,
                d,
                &[],
                vec![(puremp::Int::from_i64(1), a, vec![]), (puremp::Int::from_i64(2), bl, vec![])],
            );

            b.switch_to(a);
            let cbig = b.const_int(i128_, big);
            b.ret(Some(cbig));
            b.switch_to(bl);
            let cneg = b.const_int(i128_, neg);
            b.ret(Some(cneg));
            b.switch_to(d);
            let zero = b.const_i64(i128_, 0);
            b.ret(Some(zero));
        }
        let parsed = round_trip(&m, &mut syms);
        assert_eq!(parsed.function(FuncId::from_index(0)).block_count(), 4);
    }

    #[test]
    fn types_globals_memory() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("mem");
        let i8_ = m.types_mut().int(8);
        let i32_ = m.types_mut().int(32);
        let i64_ = m.types_mut().int(64);
        let f64_ = m.types_mut().float(FloatKind::F64);
        let arr = m.types_mut().array(i32_, 4);
        let s = m.types_mut().struct_(vec![i8_, i32_, f64_]);
        let ptr = m.types_mut().ptr();

        // A global with an aggregate initializer.
        let c1 = m.intern_const(Const::Int { ty: i32_, value: puremp::Int::from_i64(1) });
        let c2 = m.intern_const(Const::Int { ty: i32_, value: puremp::Int::from_i64(2) });
        let c3 = m.intern_const(Const::Int { ty: i32_, value: puremp::Int::from_i64(3) });
        let c4 = m.intern_const(Const::Int { ty: i32_, value: puremp::Int::from_i64(4) });
        let agg = m.intern_const(Const::Aggregate { ty: arr, elems: vec![c1, c2, c3, c4] });
        m.add_global(Global { name: syms.intern("table"), ty: arr, init: Some(agg) });
        // A declared (uninitialized) global.
        m.add_global(Global { name: syms.intern("slot"), ty: ptr, init: None });

        let void = m.types_mut().void();
        let sig = m.types_mut().func(vec![ptr], void, false);
        let f = m.declare_function(syms.intern("use_mem"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let p = b.param(e, 0);
            let sp = b.alloca(s);
            let field2 = b.struct_field(sp, s, 1);
            let v = b.load(i32_, field2, 4);
            let vext = b.cast(CastOp::SExt, v, i64_);
            let fbits = b.const_float(f64_, FloatBits::F64(1.5f64.to_bits()));
            let fb2 = b.bin(BinOp::FAdd, fbits, fbits, Flags::fast(FastMath { nnan: true, ..FastMath::default() }));
            let _ = fb2;
            let idx = b.const_i64(i64_, 2);
            let ep = b.array_elem(p, i32_, idx);
            b.store(i32_, ep, v, 4);
            let _ = vext;
            b.ret(None);
        }
        let parsed = round_trip(&m, &mut syms);
        assert_eq!(parsed.globals().count(), 2);
    }

    #[test]
    fn fadd_helper() {
        // Exercise the float binop path directly through the builder-less API by
        // building a small module.
        let mut syms = StrInterner::new();
        let mut m = Module::new("fp");
        let f32_ = m.types_mut().float(FloatKind::F32);
        let sig = m.types_mut().func(vec![f32_, f32_], f32_, false);
        let f = m.declare_function(syms.intern("h"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let x = b.param(e, 0);
            let y = b.param(e, 1);
            let z = b.bin(BinOp::FAdd, x, y, Flags::fast(FastMath {
                nnan: true,
                ninf: true,
                nsz: true,
                reassoc: true,
                contract: true,
                afn: true,
            }));
            let neg = b.fneg(z, Flags::NONE);
            let cmp = b.fcmp(FloatPred::Olt, neg, z, Flags::NONE);
            let sel = b.select(cmp, neg, z);
            b.ret(Some(sel));
        }
        round_trip(&m, &mut syms);
    }

    #[test]
    fn variadic_and_ptr_and_null_poison() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("misc");
        let i32_ = m.types_mut().int(32);
        let ptr = m.types_mut().ptr();
        // A variadic declaration.
        let vsig = m.types_mut().func(vec![ptr], i32_, true);
        m.declare_function(syms.intern("printf"), vsig);

        let sig = m.types_mut().func(vec![], ptr, false);
        let f = m.declare_function(syms.intern("mk"), sig);
        {
            let mut b = m.build(f);
            b.create_entry_block();
            let nn = b.null(ptr);
            let ps = b.poison(ptr);
            let frozen = b.freeze(ps);
            let cond = b.const_bool(true);
            let sel = b.select(cond, nn, frozen);
            b.ret(Some(sel));
        }
        round_trip(&m, &mut syms);
    }

    // --- diagnostic tests ---------------------------------------------------

    #[test]
    fn error_on_unknown_opcode() {
        let mut syms = StrInterner::new();
        let src = "module \"x\"\nfunc @f() -> void {\nentry ^0:\n  %0 = frobnicate : i32\n}\n";
        let err = parse_module(src, file(), &mut syms).unwrap_err();
        assert!(!err.is_empty());
        assert!(err[0].span.is_some(), "diagnostic must carry a span");
        assert!(err[0].message.contains("frobnicate"), "message: {}", err[0].message);
    }

    #[test]
    fn error_on_undefined_value() {
        let mut syms = StrInterner::new();
        let src = "module \"x\"\nfunc @f(i32) -> i32 {\nentry ^0(%0: i32):\n  %1 = add %0, %99 : i32\n  ret %1\n}\n";
        let err = parse_module(src, file(), &mut syms).unwrap_err();
        assert!(err[0].span.is_some());
        assert!(err[0].message.contains("undefined value"), "message: {}", err[0].message);
    }

    #[test]
    fn error_on_missing_type() {
        let mut syms = StrInterner::new();
        let src = "module \"x\"\nfunc @f() -> \n";
        let err = parse_module(src, file(), &mut syms).unwrap_err();
        assert!(err[0].span.is_some());
        assert!(err[0].message.contains("type"), "message: {}", err[0].message);
    }

    #[test]
    fn error_span_points_at_offending_token() {
        let mut syms = StrInterner::new();
        let src = "module \"x\"\nglobal @g : nonsense\n";
        let err = parse_module(src, file(), &mut syms).unwrap_err();
        let span = err[0].span.expect("span");
        // The bad type token `nonsense` starts at byte offset of its position.
        let at = &src[span.start as usize..span.end as usize];
        assert_eq!(at, "nonsense");
    }
}
