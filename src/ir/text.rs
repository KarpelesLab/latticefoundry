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
//! module      ::= "module" STRING [ "target" STRING ] [ "datalayout" STRING ]
//!                 { item }
//! item        ::= global | func
//!
//! global      ::= "global" [ linkage ] [ visibility ] [ "constant" ] [ "detached" ]
//!                 [ "secret" ] [ "thread_local" ] [ "addrspace" "(" INT ")" ] "@" name ":" type
//!                 [ "=" init ]
//! func        ::= "func" [ linkage ] [ visibility ] "@" name funcsig [ body ]
//! funcsig     ::= "(" [ [ "secret" ] type { "," [ "secret" ] type } [ "," "..." ]
//!                 | "..." ] ")" "->" [ "secret" ] type
//! linkage     ::= "internal" | "weak"
//! visibility  ::= "hidden" | "protected"
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
//!               | "dyn_alloca" operand "align" INT ":" type
//!               | "load" [ "volatile" ] [ "secret" ] operand "align" INT ":" type
//!               | "store" [ "volatile" ] [ "secret" ] operand "," operand "align" INT
//!                 ":" type
//!               | "atomic_load" ordering operand "align" INT ":" type
//!               | "atomic_store" ordering operand "," operand "align" INT ":" type
//!               | "atomic_rmw" rmwop ordering operand "," operand "align" INT ":" type
//!               | "cmpxchg" ordering ordering operand "," operand "," operand
//!                 "align" INT ":" type
//!               | "fence" ordering
//!               | "ptr_add" [ "inbounds" ] operand "," operand ":" type
//!               | "select" operand "," operand "," operand ":" type
//!               | "freeze" operand ":" type
//!               | "declassify" operand ":" type
//!               | "extractelement" operand "," INT ":" type
//!               | "insertelement" operand "," operand "," INT ":" type
//!               | "shufflevector" operand "," operand "," "[" INT { "," INT } "]"
//!                 ":" type
//!               | "splat" operand ":" type
//!               | "reduce" redop fm operand ":" type
//!               | "call" operand "(" [ operand { "," operand } ] ")" ":" type
//!               | "syscall" operand { "," operand } ":" type
//!               | "ret" [ operand ]
//!               | "br" target
//!               | "cond_br" operand "," target "," target
//!               | "switch" operand "," target "[" [ case { "," case } ] "]"
//!               | "unreachable"
//! binop       ::= ("add"|"sub"|"mul"|"shl") iflags operand "," operand ":" type
//!               | ("udiv"|"sdiv"|"lshr"|"ashr") iflags operand "," operand ":" type
//!               | ("urem"|"srem"|"and"|"or"|"xor") operand "," operand ":" type
//!               | ("smin"|"smax"|"umin"|"umax"|"sadd_sat"|"uadd_sat"|"ssub_sat"
//!                 |"usub_sat") operand "," operand ":" type
//!               | ("fadd"|"fsub"|"fmul"|"fdiv"|"frem") fm operand "," operand ":" type
//! iflags      ::= { "nsw" | "nuw" | "exact" }
//! ordering    ::= "relaxed" | "acquire" | "release" | "acq_rel" | "seq_cst"
//! rmwop       ::= "xchg" | "add" | "sub" | "and" | "nand" | "or" | "xor"
//!               | "max" | "min" | "umax" | "umin"
//! redop       ::= "add" | "mul" | "and" | "or" | "xor" | "smin" | "smax"
//!               | "umin" | "umax" | "fadd" | "fmul"
//! fm          ::= { "nnan" | "ninf" | "nsz" | "reassoc" | "contract" | "afn" }
//! target      ::= "^" INT [ "(" [ operand { "," operand } ] ")" ]
//! case        ::= INT ":" target
//!
//! operand     ::= "%" name | "@" name | const
//! const       ::= type ( INT | "0x" HEX | "null" | "poison" )
//!               | vtype "(" const { "," const } ")"
//! init        ::= type ( INT | "0x" HEX | "null" | "poison"
//!                       | "(" [ init { "," init } ] ")"
//!                       | "@" name [ ( "+" | "-" ) INT ]
//!                       | STRING )
//! type        ::= "void" | "i" INT | "f16" | "f32" | "f64"
//!               | "ptr" [ "addrspace" "(" INT ")" ]
//!               | "[" INT "x" type "]" | "{" [ type { "," type } ] "}"
//!               | vtype | "fn" fnsig
//! vtype       ::= "<" INT "x" type ">"
//! name        ::= IDENT | STRING
//! ```
//!
//! The module header may name the target (`target "x86_64"`, informational) and
//! give the [data layout](crate::ir::DataLayout) as its spec string
//! (`datalayout "e-p:32:32-…"`, `docs/ir-design.md` §3a). Both are optional and
//! printed only when present / not the default LP64 layout, so a module that
//! never sets them prints exactly as before. `ptr addrspace(N)` is a pointer
//! into address space `N` (`addrspace(0)` is plain `ptr`, which is how it
//! prints), and a global's `addrspace(N)` attribute places it in space `N`, so
//! `@x` is then a `ptr addrspace(N)`.
//!
//! Global attributes (`docs/ir-design.md` §4a): the linkage keyword picks the
//! symbol binding of a definition (external when omitted); the visibility
//! keyword picks the ELF symbol visibility of globals and functions (default
//! when omitted; §4b); `constant` places the
//! global in read-only data; `detached` marks storage supplied outside the IR
//! (the backend emits nothing for it). An `init` is a global initializer: it may
//! be an aggregate (`(` … `)`), an **address constant** `ptr @sym + 8` (a global
//! or function address plus a byte offset, resolved at link time; names may be
//! forward references), or — as input-only sugar for an `[N x i8]` array — a
//! string literal whose UTF-8 bytes number exactly `N` (escapes `\n \t \r \0
//! \\ \" \xHH`, the last for `HH < 0x80`). The printer always writes the
//! element form. Aggregate and address constants never appear as instruction
//! operands — except a **vector constant**, `<4 x i32> (i32 1, i32 2, i32
//! poison, i32 4)`, which is an ordinary first-class operand (vectors are
//! values, not addresses; `docs/ir-design.md` §6e).
//!
//! `;` begins a line comment. Integer constants are arbitrary precision
//! (`puremp::Int`); floating-point constants print as the raw IEEE bit pattern in
//! hex so they are exact and host-independent. Value operands that are constants,
//! global references, or function references are written inline (never as an
//! `%`-name), so only instruction results and block parameters receive `%`-names.

use std::collections::HashMap;
use std::fmt;

use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{
    AtomicOrdering, BinOp, CastOp, FastMath, Flags, FloatPred, InstId, InstKind, IntPred, ReduceOp,
    RmwOp, UnaryOp,
};
use crate::ir::types::{FloatKind, Type};
use crate::ir::value::{AddrTarget, Const, ConstId, FloatBits, ValueDef, ValueId};
use crate::ir::{
    BlockId, FuncAttrs, FuncId, Function, Global, GlobalAttrs, GlobalId, Linkage, Module, TypeId,
    Visibility,
};
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
    if let Some(target) = module.target() {
        write!(f, "target ")?;
        write_quoted(f, target)?;
        writeln!(f)?;
    }
    if *module.data_layout() != crate::ir::DataLayout::lp64() {
        write!(f, "datalayout ")?;
        write_quoted(f, &module.data_layout().to_spec())?;
        writeln!(f)?;
    }

    for (gi, g) in module.globals().enumerate() {
        writeln!(f)?;
        write_global(f, module, syms, gi, g, module.global_attrs(GlobalId::from_index(gi)))?;
    }

    for fi in 0..module.function_count() {
        writeln!(f)?;
        write_function(f, module, syms, FuncId::from_index(fi))?;
    }
    Ok(())
}

fn write_global<W: fmt::Write>(
    f: &mut W,
    module: &Module,
    syms: &StrInterner,
    gi: usize,
    g: &Global,
    attrs: GlobalAttrs,
) -> fmt::Result {
    write!(f, "global ")?;
    write_linkage_visibility(f, attrs.linkage, attrs.visibility)?;
    if attrs.constant {
        write!(f, "constant ")?;
    }
    if attrs.detached {
        write!(f, "detached ")?;
    }
    if attrs.secret {
        write!(f, "secret ")?;
    }
    if attrs.thread_local {
        write!(f, "thread_local ")?;
    }
    let space = module.global_addr_space(GlobalId::from_index(gi));
    if space != 0 {
        write!(f, "addrspace({space}) ")?;
    }
    write_name(f, syms.resolve(g.name))?;
    write!(f, " : ")?;
    write_type(f, module, g.ty)?;
    if let Some(init) = g.init {
        write!(f, " = ")?;
        write_const(f, module, syms, init)?;
    }
    writeln!(f)
}

/// Write the optional linkage and visibility keywords (each followed by a space).
fn write_linkage_visibility<W: fmt::Write>(
    f: &mut W,
    linkage: Linkage,
    visibility: Visibility,
) -> fmt::Result {
    match linkage {
        Linkage::External => {}
        Linkage::Internal => write!(f, "internal ")?,
        Linkage::Weak => write!(f, "weak ")?,
    }
    match visibility {
        Visibility::Default => Ok(()),
        Visibility::Hidden => write!(f, "hidden "),
        Visibility::Protected => write!(f, "protected "),
    }
}

fn write_function<W: fmt::Write>(
    f: &mut W,
    module: &Module,
    syms: &StrInterner,
    fid: FuncId,
) -> fmt::Result {
    let func = module.function(fid);
    write!(f, "func ")?;
    write_linkage_visibility(f, func.attrs.linkage, func.attrs.visibility)?;
    write_name(f, syms.resolve(func.name))?;
    write_signature_attrs(f, module, func.sig, Some(module.func_attrs(fid)))?;

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
    value_print_names(func)
}

/// The number each named value (block parameter or instruction result) is
/// printed as (`%N`) by [`print_module`], so diagnostics can name a value the
/// way the `.lf` rendering of its function does.
pub fn value_print_names(func: &Function) -> HashMap<ValueId, u32> {
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
        InstKind::DynAlloca { align } => {
            write!(f, "dyn_alloca ")?;
            op(f, ops[0])?;
            write!(f, " align {align} : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Load { ty, align, volatile, secret } => {
            write!(f, "load ")?;
            if *volatile {
                write!(f, "volatile ")?;
            }
            if *secret {
                write!(f, "secret ")?;
            }
            op(f, ops[0])?;
            write!(f, " align {align} : ")?;
            write_type(f, module, *ty)
        }
        InstKind::Store { ty, align, volatile, secret } => {
            // operands are [ptr, value]; print value first for readability.
            write!(f, "store ")?;
            if *volatile {
                write!(f, "volatile ")?;
            }
            if *secret {
                write!(f, "secret ")?;
            }
            op(f, ops[1])?;
            write!(f, ", ")?;
            op(f, ops[0])?;
            write!(f, " align {align} : ")?;
            write_type(f, module, *ty)
        }
        InstKind::AtomicLoad { ty, align, ordering } => {
            write!(f, "atomic_load {} ", ordering.name())?;
            op(f, ops[0])?;
            write!(f, " align {align} : ")?;
            write_type(f, module, *ty)
        }
        InstKind::AtomicStore { ty, align, ordering } => {
            // Like `store`: value first, then the address.
            write!(f, "atomic_store {} ", ordering.name())?;
            op(f, ops[1])?;
            write!(f, ", ")?;
            op(f, ops[0])?;
            write!(f, " align {align} : ")?;
            write_type(f, module, *ty)
        }
        InstKind::AtomicRmw { op: rmw, ty, align, ordering } => {
            write!(f, "atomic_rmw {} {} ", rmw.name(), ordering.name())?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            op(f, ops[1])?;
            write!(f, " align {align} : ")?;
            write_type(f, module, *ty)
        }
        InstKind::CmpXchg { ty, align, success, failure } => {
            write!(f, "cmpxchg {} {} ", success.name(), failure.name())?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            op(f, ops[1])?;
            write!(f, ", ")?;
            op(f, ops[2])?;
            write!(f, " align {align} : ")?;
            write_type(f, module, *ty)
        }
        InstKind::Fence(ordering) => write!(f, "fence {}", ordering.name()),
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
        InstKind::Declassify => {
            write!(f, "declassify ")?;
            op(f, ops[0])?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::ExtractElement { lane } => {
            write!(f, "extractelement ")?;
            op(f, ops[0])?;
            write!(f, ", {lane} : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::InsertElement { lane } => {
            write!(f, "insertelement ")?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            op(f, ops[1])?;
            write!(f, ", {lane} : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::ShuffleVector(mask) => {
            write!(f, "shufflevector ")?;
            op(f, ops[0])?;
            write!(f, ", ")?;
            op(f, ops[1])?;
            write!(f, ", [")?;
            for (i, m) in mask.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{m}")?;
            }
            write!(f, "] : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Splat => {
            write!(f, "splat ")?;
            op(f, ops[0])?;
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::Reduce(r) => {
            write!(f, "reduce {}", r.name())?;
            write_fastmath(f, data.flags.fast)?;
            write!(f, " ")?;
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
        InstKind::Syscall => {
            write!(f, "syscall ")?;
            for (i, &a) in ops.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                op(f, a)?;
            }
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::InlineAsm(asm) => {
            // `inline_asm [volatile] "tmpl" [outs(...)] [ins(...)]
            // [clobbers(...)] : T`; each operand is its constraint, an
            // optional `[name]`, the type of a register output, and the value
            // operand standing for it (if any) in parentheses.
            write!(f, "inline_asm ")?;
            if asm.volatile {
                write!(f, "volatile ")?;
            }
            write_quoted(f, &asm.template)?;
            let slots = asm.operand_slots();
            let operand_of = |slot: crate::ir::inst::AsmSlot| slots.iter().position(|&s| s == slot).map(|i| ops[i]);
            if !asm.outputs.is_empty() {
                write!(f, " outs(")?;
                for (i, o) in asm.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write_quoted(f, &o.constraint)?;
                    if let Some(n) = &o.name {
                        write!(f, " [{n}]")?;
                    }
                    if let Some(ty) = o.ty {
                        write!(f, " ")?;
                        write_type(f, module, ty)?;
                    }
                    if let Some(v) = operand_of(crate::ir::inst::AsmSlot::Output(i)) {
                        write!(f, " (")?;
                        op(f, v)?;
                        write!(f, ")")?;
                    }
                }
                write!(f, ")")?;
            }
            if !asm.inputs.is_empty() {
                write!(f, " ins(")?;
                for (i, a) in asm.inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write_quoted(f, &a.constraint)?;
                    if let Some(n) = &a.name {
                        write!(f, " [{n}]")?;
                    }
                    if let Some(v) = operand_of(crate::ir::inst::AsmSlot::Input(i)) {
                        write!(f, " (")?;
                        op(f, v)?;
                        write!(f, ")")?;
                    }
                }
                write!(f, ")")?;
            }
            if !asm.clobbers.is_empty() {
                write!(f, " clobbers(")?;
                for (i, c) in asm.clobbers.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write_quoted(f, c)?;
                }
                write!(f, ")")?;
            }
            write!(f, " : ")?;
            write_type(f, module, data.ty)
        }
        InstKind::AsmOutput(n) => {
            write!(f, "asm_output ")?;
            op(f, ops[0])?;
            write!(f, ", {n} : ")?;
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
        ValueDef::Const(cid) => write_const(f, module, syms, *cid),
        ValueDef::Global(g) => write_name(f, syms.resolve(module.global(*g).name)),
        ValueDef::Func(fu) => write_name(f, syms.resolve(module.function(*fu).name)),
    }
}

fn write_const<W: fmt::Write>(
    f: &mut W,
    module: &Module,
    syms: &StrInterner,
    cid: ConstId,
) -> fmt::Result {
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
                write_const(f, module, syms, e)?;
            }
            write!(f, ")")
        }
        Const::Addr { target, offset, .. } => {
            let name = match *target {
                AddrTarget::Global(g) => module.global(g).name,
                AddrTarget::Func(fu) => module.function(fu).name,
            };
            write!(f, " ")?;
            write_name(f, syms.resolve(name))?;
            match offset.cmp(&0) {
                std::cmp::Ordering::Greater => write!(f, " + {offset}"),
                std::cmp::Ordering::Less => write!(f, " - {}", offset.unsigned_abs()),
                std::cmp::Ordering::Equal => Ok(()),
            }
        }
    }
}

/// Write a signature, marking `secret` parameters and return (a function
/// header's [`FuncAttrs`]).
fn write_signature_attrs<W: fmt::Write>(
    f: &mut W,
    module: &Module,
    sig: TypeId,
    attrs: Option<&FuncAttrs>,
) -> fmt::Result {
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
        if attrs.is_some_and(|a| a.is_param_secret(i)) {
            write!(f, "secret ")?;
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
    if attrs.is_some_and(|a| a.secret_ret) {
        write!(f, "secret ")?;
    }
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
        Type::PtrIn(space) => write!(f, "ptr addrspace({space})"),
        Type::Array(elem, n) => {
            write!(f, "[{n} x ")?;
            write_type(f, module, *elem)?;
            write!(f, "]")
        }
        Type::Vector(elem, n) => {
            write!(f, "<{n} x ")?;
            write_type(f, module, *elem)?;
            write!(f, ">")
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
            '\r' => write!(f, "\\r")?,
            '\0' => write!(f, "\\0")?,
            c if c.is_ascii_control() => write!(f, "\\x{:02x}", c as u32)?,
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
        BinOp::SMin => "smin",
        BinOp::SMax => "smax",
        BinOp::UMin => "umin",
        BinOp::UMax => "umax",
        BinOp::SAddSat => "sadd_sat",
        BinOp::UAddSat => "uadd_sat",
        BinOp::SSubSat => "ssub_sat",
        BinOp::USubSat => "usub_sat",
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
    Plus,
    Lt,
    Gt,
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
            b'+' => {
                toks.push(Tok { kind: TokKind::Plus, span: sp(i, i + 1) });
                i += 1;
            }
            b'<' => {
                toks.push(Tok { kind: TokKind::Lt, span: sp(i, i + 1) });
                i += 1;
            }
            b'>' => {
                toks.push(Tok { kind: TokKind::Gt, span: sp(i, i + 1) });
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
                                b'r' => s.push('\r'),
                                b'0' => s.push('\0'),
                                b'x' => {
                                    // `\xHH`: an ASCII byte (HH < 0x80) in hex.
                                    let hex = src.get(i + 2..i + 4).unwrap_or("");
                                    match u8::from_str_radix(hex, 16) {
                                        Ok(v) if hex.len() == 2 && v < 0x80 => s.push(char::from(v)),
                                        _ => {
                                            return Err(Diagnostic::error(
                                                "`\\x` escape needs two hex digits below 80",
                                            )
                                            .with_span(sp(i, (i + 4).min(n))));
                                        }
                                    }
                                    i += 2;
                                }
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
    /// A vector constant: its type and one constant per lane.
    Vector(TypeId, Vec<ConstAst>),
}

/// A parsed global initializer, resolved to a [`ConstId`] only after every
/// top-level item is known (address constants may name later symbols).
#[derive(Debug)]
enum InitAst {
    /// A scalar leaf (`int`/`float`/`null`/`poison`), already interned.
    Leaf(ConstId),
    /// An aggregate of the given type.
    Aggregate(TypeId, Vec<InitAst>),
    /// `ptr @name ± offset`.
    Addr(TypeId, String, i64, Span),
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
    DynAlloca(u32, Operand),
    Syscall(Vec<Operand>),
    /// `(ty, align, volatile, secret, ptr)`.
    Load(TypeId, u32, bool, bool, Operand),
    /// `(ty, align, volatile, secret, value, ptr)`.
    Store(TypeId, u32, bool, bool, Operand, Operand),
    AtomicLoad(TypeId, u32, AtomicOrdering, Operand),
    AtomicStore(TypeId, u32, AtomicOrdering, Operand, Operand),
    AtomicRmw(RmwOp, TypeId, u32, AtomicOrdering, Operand, Operand),
    CmpXchg(TypeId, u32, AtomicOrdering, AtomicOrdering, Operand, Operand, Operand),
    Fence(AtomicOrdering),
    PtrAdd(bool, Operand, Operand),
    Select(Operand, Operand, Operand),
    Freeze(Operand),
    Declassify(Operand),
    ExtractElement(Operand, u32),
    InsertElement(Operand, Operand, u32),
    ShuffleVector(Operand, Operand, Vec<u32>, TypeId),
    Splat(Operand, TypeId),
    Reduce(ReduceOp, Flags, Operand),
    Call(Operand, Vec<Operand>, TypeId),
    /// The asm and its value operands, in `operand_slots` order.
    InlineAsm(Box<crate::ir::inst::InlineAsm>, Vec<Operand>),
    AsmOutput(Operand, u32, TypeId),
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

    /// `"align" INT ":" type` — the common tail of the memory operations.
    fn parse_align_type(&mut self, module: &mut Module) -> PResult<(u32, TypeId)> {
        self.expect_ident("align")?;
        let align = self.parse_u32()?;
        self.expect(&TokKind::Colon, "`:`")?;
        let ty = self.parse_type(module)?;
        Ok((align, ty))
    }

    /// An atomic memory ordering keyword (`relaxed`, `acquire`, `release`,
    /// `acq_rel`, `seq_cst`).
    fn parse_ordering(&mut self) -> PResult<AtomicOrdering> {
        let (name, sp) = self.expect_any_ident()?;
        AtomicOrdering::from_name(&name).ok_or_else(|| {
            Diagnostic::error(format!(
                "expected a memory ordering (relaxed, acquire, release, acq_rel, seq_cst), found `{name}`"
            ))
            .with_span(sp)
        })
    }

    /// A string literal's text.
    fn expect_str(&mut self, what: &str) -> PResult<String> {
        let sp = self.span();
        if let TokKind::Str(s) = self.peek_kind().clone() {
            self.bump();
            Ok(s)
        } else {
            self.err(sp, format!("expected {what} (a string literal)"))
        }
    }

    /// The body of an `inline_asm` (after the opcode): see `write_inst`.
    fn parse_inline_asm(&mut self, module: &mut Module) -> PResult<OpAst> {
        let volatile = self.eat_ident("volatile");
        let template = self.expect_str("the asm template")?;
        let mut asm = crate::ir::inst::InlineAsm {
            template,
            outputs: Vec::new(),
            inputs: Vec::new(),
            clobbers: Vec::new(),
            volatile,
        };
        // Value operands, tagged by the asm operand they stand for.
        let mut out_ops: Vec<(usize, Operand)> = Vec::new();
        let mut in_ops: Vec<(usize, Operand)> = Vec::new();
        let list = |p: &mut Parser, kw: &str, item: &mut dyn FnMut(&mut Parser, usize) -> PResult<()>| {
            if !p.eat_ident(kw) {
                return Ok(());
            }
            p.expect(&TokKind::LParen, "`(`")?;
            let mut i = 0;
            if !matches!(p.peek_kind(), TokKind::RParen) {
                loop {
                    item(p, i)?;
                    i += 1;
                    if p.eat(&TokKind::Comma) {
                        continue;
                    }
                    break;
                }
            }
            p.expect(&TokKind::RParen, "`)`").map(|_| ())
        };
        // `[name]`, then (outputs only) a type, then `(operand)`.
        fn name(p: &mut Parser) -> PResult<Option<String>> {
            if p.eat(&TokKind::LBracket) {
                let (n, _) = p.expect_any_ident()?;
                p.expect(&TokKind::RBracket, "`]`")?;
                Ok(Some(n))
            } else {
                Ok(None)
            }
        }
        fn operand(p: &mut Parser, module: &mut Module) -> PResult<Option<Operand>> {
            if p.eat(&TokKind::LParen) {
                let v = p.parse_operand(module)?;
                p.expect(&TokKind::RParen, "`)`")?;
                Ok(Some(v))
            } else {
                Ok(None)
            }
        }
        list(self, "outs", &mut |p, i| {
            let constraint = p.expect_str("an output constraint")?;
            let name = name(p)?;
            let ty = if matches!(p.peek_kind(), TokKind::LParen | TokKind::Comma | TokKind::RParen) {
                None
            } else {
                Some(p.parse_type(module)?)
            };
            if let Some(v) = operand(p, module)? {
                out_ops.push((i, v));
            }
            asm.outputs.push(crate::ir::inst::AsmOutput { constraint, name, ty });
            Ok(())
        })?;
        list(self, "ins", &mut |p, i| {
            let constraint = p.expect_str("an input constraint")?;
            let name = name(p)?;
            let sp = p.span();
            match operand(p, module)? {
                Some(v) => in_ops.push((i, v)),
                None => return p.err(sp, "an asm input needs an operand `(value)`"),
            }
            asm.inputs.push(crate::ir::inst::AsmInput { constraint, name });
            Ok(())
        })?;
        list(self, "clobbers", &mut |p, _| {
            let c = p.expect_str("a clobber")?;
            asm.clobbers.push(c);
            Ok(())
        })?;
        self.expect(&TokKind::Colon, "`:`")?;
        let sp = self.span();
        let ty = self.parse_type(module)?;
        // The operands must be exactly the ones the constraints call for.
        let slots = asm.operand_slots();
        let given: Vec<crate::ir::inst::AsmSlot> = out_ops
            .iter()
            .map(|&(i, _)| crate::ir::inst::AsmSlot::Output(i))
            .chain(in_ops.iter().map(|&(i, _)| crate::ir::inst::AsmSlot::Input(i)))
            .collect();
        if given != slots {
            return self.err(
                sp,
                "inline_asm operands do not match its constraints (an indirect or `+` output, and every input, takes one `(value)`)",
            );
        }
        let want = asm.result_output().and_then(|i| asm.outputs[i].ty).unwrap_or_else(|| module.types_mut().void());
        if ty != want {
            return self.err(sp, "an inline_asm's type must be its first register output's type (or `void`)");
        }
        let operands = out_ops.into_iter().chain(in_ops).map(|(_, v)| v).collect();
        Ok(OpAst::InlineAsm(Box::new(asm), operands))
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

        // Optional header declarations: the target name, then the data layout.
        if self.eat_ident("target") {
            let sp = self.span();
            match self.peek_kind().clone() {
                TokKind::Str(s) => {
                    self.bump();
                    module.set_target(Some(s));
                }
                _ => return self.err(sp, "expected a target name string"),
            }
        }
        if self.eat_ident("datalayout") {
            let sp = self.span();
            match self.peek_kind().clone() {
                TokKind::Str(s) => {
                    self.bump();
                    match crate::ir::DataLayout::parse(&s) {
                        Ok(dl) => module.set_data_layout(dl),
                        Err(e) => return self.err(sp, e.to_string()),
                    }
                }
                _ => return self.err(sp, "expected a data layout spec string"),
            }
        }

        let mut func_names: HashMap<String, FuncId> = HashMap::new();
        let mut global_names: HashMap<String, GlobalId> = HashMap::new();
        let mut pending: Vec<(FuncId, BodyAst, u32)> = Vec::new();
        let mut pending_inits: Vec<(GlobalId, InitAst)> = Vec::new();

        loop {
            match self.peek_kind() {
                TokKind::Eof => break,
                TokKind::Ident(id) if id == "global" => {
                    let (gid, init) = self.parse_global(&mut module, syms, &mut global_names)?;
                    if let Some(init) = init {
                        pending_inits.push((gid, init));
                    }
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

        for (gid, init) in pending_inits {
            let c = lower_init(&mut module, &init, &func_names, &global_names)?;
            module.set_global_init(gid, Some(c));
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
    ) -> PResult<(GlobalId, Option<InitAst>)> {
        self.expect_ident("global")?;
        let mut attrs = GlobalAttrs::DEFAULT;
        (attrs.linkage, attrs.visibility) = self.parse_linkage_visibility();
        attrs.constant = self.eat_ident("constant");
        attrs.detached = self.eat_ident("detached");
        attrs.secret = self.eat_ident("secret");
        attrs.thread_local = self.eat_ident("thread_local");
        let space = if self.eat_ident("addrspace") {
            self.expect(&TokKind::LParen, "`(`")?;
            let space = self.parse_u32()?;
            self.expect(&TokKind::RParen, "`)`")?;
            space
        } else {
            0
        };
        let name = self.parse_name()?;
        self.expect(&TokKind::Colon, "`:`")?;
        let ty = self.parse_type(module)?;
        let init = if self.eat(&TokKind::Eq) { Some(self.parse_init(module)?) } else { None };
        let sym = syms.intern(&name);
        // The initializer is attached once every name is known (`lower_init`).
        let gid = module.define_global(Global { name: sym, ty, init: None }, attrs);
        module.set_global_addr_space(gid, space);
        global_names.insert(name, gid);
        Ok((gid, init))
    }

    fn parse_func(
        &mut self,
        module: &mut Module,
        syms: &mut StrInterner,
        func_names: &mut HashMap<String, FuncId>,
    ) -> PResult<(FuncId, Option<BodyAst>, u32)> {
        let func_kw = self.expect_ident("func")?;
        let decl_line = self.lines.line_of(func_kw.start);
        let (linkage, visibility) = self.parse_linkage_visibility();
        let name = self.parse_name()?;
        let mut attrs = FuncAttrs::new(linkage, visibility);
        let (params, ret, variadic) = self.parse_fn_sig_attrs(module, Some(&mut attrs))?;
        let sig = module.types_mut().func(params, ret, variadic);
        let sym = syms.intern(&name);
        let fid = module.declare_function(sym, sig);
        module.set_func_attrs(fid, attrs);
        func_names.insert(name, fid);

        let body = if matches!(self.peek_kind(), TokKind::LBrace) {
            Some(self.parse_body(module)?)
        } else {
            None
        };
        Ok((fid, body, decl_line))
    }

    /// Parse the optional `internal`/`weak` linkage and `hidden`/`protected`
    /// visibility keywords of a `global` or `func` header.
    fn parse_linkage_visibility(&mut self) -> (Linkage, Visibility) {
        let linkage = if self.eat_ident("internal") {
            Linkage::Internal
        } else if self.eat_ident("weak") {
            Linkage::Weak
        } else {
            Linkage::External
        };
        let visibility = if self.eat_ident("hidden") {
            Visibility::Hidden
        } else if self.eat_ident("protected") {
            Visibility::Protected
        } else {
            Visibility::Default
        };
        (linkage, visibility)
    }

    fn parse_fn_sig(&mut self, module: &mut Module) -> PResult<(Vec<TypeId>, TypeId, bool)> {
        self.parse_fn_sig_attrs(module, None)
    }

    /// Parse a signature; with `attrs` (a function header), a `secret` keyword
    /// may precede each parameter type and the return type.
    fn parse_fn_sig_attrs(
        &mut self,
        module: &mut Module,
        mut attrs: Option<&mut FuncAttrs>,
    ) -> PResult<(Vec<TypeId>, TypeId, bool)> {
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
                if let Some(a) = attrs.as_deref_mut()
                    && self.eat_ident("secret")
                {
                    a.set_param_secret(params.len(), true);
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
        if let Some(a) = attrs
            && self.eat_ident("secret")
        {
            a.secret_ret = true;
        }
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
            "dyn_alloca" => {
                let n = self.parse_operand(module)?;
                self.expect_ident("align")?;
                let align = self.parse_u32()?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ptr = self.parse_type(module)?;
                Ok(OpAst::DynAlloca(align, n))
            }
            "load" => {
                let volatile = self.eat_ident("volatile");
                let secret = self.eat_ident("secret");
                let ptr = self.parse_operand(module)?;
                let (align, ty) = self.parse_align_type(module)?;
                Ok(OpAst::Load(ty, align, volatile, secret, ptr))
            }
            "store" => {
                let volatile = self.eat_ident("volatile");
                let secret = self.eat_ident("secret");
                let val = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let ptr = self.parse_operand(module)?;
                let (align, ty) = self.parse_align_type(module)?;
                Ok(OpAst::Store(ty, align, volatile, secret, val, ptr))
            }
            "atomic_load" => {
                let ordering = self.parse_ordering()?;
                let ptr = self.parse_operand(module)?;
                let (align, ty) = self.parse_align_type(module)?;
                Ok(OpAst::AtomicLoad(ty, align, ordering, ptr))
            }
            "atomic_store" => {
                let ordering = self.parse_ordering()?;
                let val = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let ptr = self.parse_operand(module)?;
                let (align, ty) = self.parse_align_type(module)?;
                Ok(OpAst::AtomicStore(ty, align, ordering, val, ptr))
            }
            "atomic_rmw" => {
                let (name, sp) = self.expect_any_ident()?;
                let rmw = RmwOp::from_name(&name)
                    .ok_or_else(|| Diagnostic::error("unknown atomic_rmw operation").with_span(sp))?;
                let ordering = self.parse_ordering()?;
                let ptr = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let val = self.parse_operand(module)?;
                let (align, ty) = self.parse_align_type(module)?;
                Ok(OpAst::AtomicRmw(rmw, ty, align, ordering, ptr, val))
            }
            "cmpxchg" => {
                let success = self.parse_ordering()?;
                let failure = self.parse_ordering()?;
                let ptr = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let expected = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let new = self.parse_operand(module)?;
                let (align, ty) = self.parse_align_type(module)?;
                Ok(OpAst::CmpXchg(ty, align, success, failure, ptr, expected, new))
            }
            "fence" => Ok(OpAst::Fence(self.parse_ordering()?)),
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
            "declassify" => {
                let v = self.parse_operand(module)?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ty = self.parse_type(module)?;
                Ok(OpAst::Declassify(v))
            }
            "extractelement" => {
                let v = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let lane = self.parse_u32()?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ty = self.parse_type(module)?;
                Ok(OpAst::ExtractElement(v, lane))
            }
            "insertelement" => {
                let v = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let x = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let lane = self.parse_u32()?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ty = self.parse_type(module)?;
                Ok(OpAst::InsertElement(v, x, lane))
            }
            "shufflevector" => {
                let a = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let c = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                self.expect(&TokKind::LBracket, "`[`")?;
                let mut mask = vec![self.parse_u32()?];
                while self.eat(&TokKind::Comma) {
                    mask.push(self.parse_u32()?);
                }
                self.expect(&TokKind::RBracket, "`]`")?;
                self.expect(&TokKind::Colon, "`:`")?;
                let ty = self.parse_type(module)?;
                Ok(OpAst::ShuffleVector(a, c, mask, ty))
            }
            "splat" => {
                let v = self.parse_operand(module)?;
                self.expect(&TokKind::Colon, "`:`")?;
                let ty = self.parse_type(module)?;
                Ok(OpAst::Splat(v, ty))
            }
            "reduce" => {
                let (name, sp) = self.expect_any_ident()?;
                let r = ReduceOp::from_name(&name)
                    .ok_or_else(|| Diagnostic::error("unknown reduce operation").with_span(sp))?;
                let flags = self.parse_fastmath();
                let v = self.parse_operand(module)?;
                self.expect(&TokKind::Colon, "`:`")?;
                let _ty = self.parse_type(module)?;
                Ok(OpAst::Reduce(r, flags, v))
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
            "syscall" => {
                // `syscall nr {, arg} : i64` — the result is always `i64`.
                let mut ops = vec![self.parse_operand(module)?];
                while self.eat(&TokKind::Comma) {
                    ops.push(self.parse_operand(module)?);
                }
                self.expect(&TokKind::Colon, "`:`")?;
                let span = self.span();
                let ty = self.parse_type(module)?;
                if ty != module.types_mut().int(64) {
                    return self.err(span, "a syscall's result type must be `i64`");
                }
                Ok(OpAst::Syscall(ops))
            }
            "inline_asm" => self.parse_inline_asm(module),
            "asm_output" => {
                let v = self.parse_operand(module)?;
                self.expect(&TokKind::Comma, "`,`")?;
                let n = self.parse_u32()?;
                self.expect(&TokKind::Colon, "`:`")?;
                let ty = self.parse_type(module)?;
                Ok(OpAst::AsmOutput(v, n, ty))
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
            // A vector constant is a first-class operand: one constant per lane.
            if let Some((elem, n)) = module.types().vector_parts(ty) {
                self.bump();
                let mut lanes = Vec::with_capacity(n as usize);
                loop {
                    let lane_sp = self.span();
                    let c = self.parse_const_operand(module)?;
                    let cty = match &c {
                        ConstAst::Int(t, _)
                        | ConstAst::Float(t, _)
                        | ConstAst::Null(t)
                        | ConstAst::Poison(t)
                        | ConstAst::Vector(t, _) => *t,
                    };
                    if cty != elem {
                        return self.err(lane_sp, "vector constant lane has the wrong type");
                    }
                    lanes.push(c);
                    if !self.eat(&TokKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokKind::RParen, "`)`")?;
                if lanes.len() != n as usize {
                    return self.err(
                        ty_sp.merge(self.prev_span()),
                        format!("vector constant has {} lane(s) but its type has {n}", lanes.len()),
                    );
                }
                return Ok(ConstAst::Vector(ty, lanes));
            }
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

    /// Parse a global initializer. Unlike operand constants, these may be
    /// aggregates, address constants (`ptr @sym + off`), or `[N x i8]` string
    /// sugar. Scalar leaves are interned immediately; names are resolved later
    /// by [`lower_init`], so an initializer may reference a later item.
    fn parse_init(&mut self, module: &mut Module) -> PResult<InitAst> {
        let ty_sp = self.span();
        let ty = self.parse_type(module)?;
        if self.eat_ident("null") {
            return Ok(InitAst::Leaf(module.intern_const(Const::Null(ty))));
        }
        if self.eat_ident("poison") {
            return Ok(InitAst::Leaf(module.intern_const(Const::Poison(ty))));
        }
        if self.eat(&TokKind::LParen) {
            let mut elems = Vec::new();
            if !matches!(self.peek_kind(), TokKind::RParen) {
                loop {
                    elems.push(self.parse_init(module)?);
                    if self.eat(&TokKind::Comma) {
                        continue;
                    }
                    break;
                }
            }
            self.expect(&TokKind::RParen, "`)`")?;
            return Ok(InitAst::Aggregate(ty, elems));
        }
        if matches!(self.peek_kind(), TokKind::At) {
            let sp = self.span();
            let name = self.parse_name()?;
            let offset = if self.eat(&TokKind::Plus) {
                self.parse_offset(false)?
            } else if self.eat(&TokKind::Minus) {
                self.parse_offset(true)?
            } else {
                0
            };
            return Ok(InitAst::Addr(ty, name, offset, sp.merge(self.prev_span())));
        }
        if let TokKind::Str(text) = self.peek_kind().clone() {
            let sp = self.span();
            self.bump();
            let i8t = module.types_mut().int(8);
            let len = match module.types().get(ty) {
                Type::Array(elem, n) if *elem == i8t => *n,
                _ => return self.err(ty_sp, "a string initializer needs an `[N x i8]` type"),
            };
            if text.len() as u64 != len {
                return self.err(
                    sp,
                    format!("string has {} byte(s) but the array type has {len}", text.len()),
                );
            }
            let elems = text
                .bytes()
                .map(|b| {
                    let value = puremp::Int::from_i64(i64::from(b));
                    InitAst::Leaf(module.intern_const(Const::Int { ty: i8t, value }))
                })
                .collect();
            return Ok(InitAst::Aggregate(ty, elems));
        }
        match module.types().get(ty).clone() {
            Type::Int(_) => {
                let value = self.parse_signed_int()?;
                Ok(InitAst::Leaf(module.intern_const(Const::Int { ty, value })))
            }
            Type::Float(k) => {
                let bits = self.parse_float_bits(k)?;
                Ok(InitAst::Leaf(module.intern_const(Const::Float { ty, bits })))
            }
            _ => self.err(ty_sp, "expected a constant payload"),
        }
    }

    /// The unsigned byte offset after `+`/`-` in an address constant, negated
    /// if `neg`.
    fn parse_offset(&mut self, neg: bool) -> PResult<i64> {
        let sp = self.span();
        let mag = self.parse_u64()?;
        let v = if neg { 0i64.checked_sub_unsigned(mag) } else { i64::try_from(mag).ok() };
        v.ok_or_else(|| Diagnostic::error("address offset out of range").with_span(sp))
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
                    "ptr" => {
                        if self.eat_ident("addrspace") {
                            self.expect(&TokKind::LParen, "`(`")?;
                            let space = self.parse_u32()?;
                            self.expect(&TokKind::RParen, "`)`")?;
                            Ok(module.types_mut().ptr_in(space))
                        } else {
                            Ok(module.types_mut().ptr())
                        }
                    }
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
            TokKind::Lt => {
                self.bump();
                let lanes = self.parse_u32()?;
                self.expect_ident("x")?;
                let elem = self.parse_type(module)?;
                self.expect(&TokKind::Gt, "`>`")?;
                Ok(module.types_mut().vector(elem, lanes))
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
        OpAst::DynAlloca(align, n) => {
            let nv = resolve_operand(b, n, names, func_names, global_names)?;
            Some(b.dyn_alloca(nv, *align))
        }
        OpAst::Load(ty, align, volatile, secret, ptr) => {
            let p = resolve_operand(b, ptr, names, func_names, global_names)?;
            let kind =
                InstKind::Load { ty: *ty, align: *align, volatile: *volatile, secret: *secret };
            b.append_inst(kind, vec![p], Flags::NONE, Some(*ty))
        }
        OpAst::Store(ty, align, volatile, secret, val, ptr) => {
            let v = resolve_operand(b, val, names, func_names, global_names)?;
            let p = resolve_operand(b, ptr, names, func_names, global_names)?;
            let kind =
                InstKind::Store { ty: *ty, align: *align, volatile: *volatile, secret: *secret };
            b.append_inst(kind, vec![p, v], Flags::NONE, None)
        }
        // The atomics carry an explicit alignment in text, so they are built
        // from raw parts (the builder helpers always use natural alignment).
        OpAst::AtomicLoad(ty, align, ordering, ptr) => {
            let p = resolve_operand(b, ptr, names, func_names, global_names)?;
            let kind = InstKind::AtomicLoad { ty: *ty, align: *align, ordering: *ordering };
            b.append_inst(kind, vec![p], Flags::NONE, Some(*ty))
        }
        OpAst::AtomicStore(ty, align, ordering, val, ptr) => {
            let v = resolve_operand(b, val, names, func_names, global_names)?;
            let p = resolve_operand(b, ptr, names, func_names, global_names)?;
            let kind = InstKind::AtomicStore { ty: *ty, align: *align, ordering: *ordering };
            b.append_inst(kind, vec![p, v], Flags::NONE, None)
        }
        OpAst::AtomicRmw(rmw, ty, align, ordering, ptr, val) => {
            let p = resolve_operand(b, ptr, names, func_names, global_names)?;
            let v = resolve_operand(b, val, names, func_names, global_names)?;
            let kind =
                InstKind::AtomicRmw { op: *rmw, ty: *ty, align: *align, ordering: *ordering };
            b.append_inst(kind, vec![p, v], Flags::NONE, Some(*ty))
        }
        OpAst::CmpXchg(ty, align, success, failure, ptr, expected, new) => {
            let p = resolve_operand(b, ptr, names, func_names, global_names)?;
            let e = resolve_operand(b, expected, names, func_names, global_names)?;
            let n = resolve_operand(b, new, names, func_names, global_names)?;
            let kind =
                InstKind::CmpXchg { ty: *ty, align: *align, success: *success, failure: *failure };
            b.append_inst(kind, vec![p, e, n], Flags::NONE, Some(*ty))
        }
        OpAst::Fence(ordering) => {
            b.fence(*ordering);
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
        OpAst::Declassify(v) => {
            let val = resolve_operand(b, v, names, func_names, global_names)?;
            Some(b.declassify(val))
        }
        OpAst::ExtractElement(v, lane) => {
            let val = resolve_operand(b, v, names, func_names, global_names)?;
            Some(b.extract_element(val, *lane))
        }
        OpAst::InsertElement(v, x, lane) => {
            let val = resolve_operand(b, v, names, func_names, global_names)?;
            let xv = resolve_operand(b, x, names, func_names, global_names)?;
            Some(b.insert_element(val, xv, *lane))
        }
        OpAst::ShuffleVector(a, c, mask, ty) => {
            let av = resolve_operand(b, a, names, func_names, global_names)?;
            let cv = resolve_operand(b, c, names, func_names, global_names)?;
            let kind = InstKind::ShuffleVector(mask.clone().into_boxed_slice());
            b.append_inst(kind, vec![av, cv], Flags::NONE, Some(*ty))
        }
        OpAst::Splat(v, ty) => {
            let val = resolve_operand(b, v, names, func_names, global_names)?;
            b.append_inst(InstKind::Splat, vec![val], Flags::NONE, Some(*ty))
        }
        OpAst::Reduce(r, flags, v) => {
            let val = resolve_operand(b, v, names, func_names, global_names)?;
            Some(b.reduce(*r, val, *flags))
        }
        OpAst::Call(callee, args, ret) => {
            let cv = resolve_operand(b, callee, names, func_names, global_names)?;
            let mut avs = Vec::with_capacity(args.len());
            for a in args {
                avs.push(resolve_operand(b, a, names, func_names, global_names)?);
            }
            b.call(cv, &avs, *ret)
        }
        OpAst::Syscall(ops) => {
            let nr = resolve_operand(b, &ops[0], names, func_names, global_names)?;
            let mut avs = Vec::with_capacity(ops.len() - 1);
            for a in &ops[1..] {
                avs.push(resolve_operand(b, a, names, func_names, global_names)?);
            }
            Some(b.syscall(nr, &avs))
        }
        OpAst::InlineAsm(asm, ops) => {
            let mut avs = Vec::with_capacity(ops.len());
            for a in ops {
                avs.push(resolve_operand(b, a, names, func_names, global_names)?);
            }
            b.inline_asm((**asm).clone(), &avs)
        }
        OpAst::AsmOutput(v, n, ty) => {
            let val = resolve_operand(b, v, names, func_names, global_names)?;
            Some(b.asm_output(val, *n, *ty))
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

/// Intern a parsed global initializer, resolving address-constant names against
/// the module's functions (preferred, as for operands) and globals.
fn lower_init(
    module: &mut Module,
    init: &InitAst,
    func_names: &HashMap<String, FuncId>,
    global_names: &HashMap<String, GlobalId>,
) -> PResult<ConstId> {
    Ok(match init {
        InitAst::Leaf(c) => *c,
        InitAst::Aggregate(ty, elems) => {
            let mut ids = Vec::with_capacity(elems.len());
            for e in elems {
                ids.push(lower_init(module, e, func_names, global_names)?);
            }
            module.intern_const(Const::Aggregate { ty: *ty, elems: ids })
        }
        InitAst::Addr(ty, name, offset, sp) => {
            let target = if let Some(&f) = func_names.get(name) {
                AddrTarget::Func(f)
            } else if let Some(&g) = global_names.get(name) {
                AddrTarget::Global(g)
            } else {
                return Err(Diagnostic::error(format!("unknown reference `@{name}`")).with_span(*sp));
            };
            module.intern_const(Const::Addr { ty: *ty, target, offset: *offset })
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
            ConstAst::Vector(ty, lanes) => {
                let ids = lanes.iter().map(|l| intern_const_ast(b, l)).collect();
                b.const_vector(*ty, ids)
            }
            other => {
                let id = intern_const_ast(b, other);
                b.use_const(id)
            }
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

/// Intern a parsed operand constant (recursively for a vector's lanes).
fn intern_const_ast(b: &mut FunctionBuilder<'_>, c: &ConstAst) -> ConstId {
    match c {
        ConstAst::Int(ty, v) => b.intern_const(Const::Int { ty: *ty, value: v.clone() }),
        ConstAst::Float(ty, bits) => b.intern_const(Const::Float { ty: *ty, bits: *bits }),
        ConstAst::Null(ty) => b.intern_const(Const::Null(*ty)),
        ConstAst::Poison(ty) => b.intern_const(Const::Poison(*ty)),
        ConstAst::Vector(ty, lanes) => {
            let elems = lanes.iter().map(|l| intern_const_ast(b, l)).collect();
            b.intern_const(Const::Aggregate { ty: *ty, elems })
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
        "smin" => BinOp::SMin,
        "smax" => BinOp::SMax,
        "umin" => BinOp::UMin,
        "umax" => BinOp::UMax,
        "sadd_sat" => BinOp::SAddSat,
        "uadd_sat" => BinOp::UAddSat,
        "ssub_sat" => BinOp::SSubSat,
        "usub_sat" => BinOp::USubSat,
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
    fn dyn_alloca_round_trips() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("dyn");
        let i64_ = m.types_mut().int(64);
        let sig = m.types_mut().func(vec![i64_], i64_, false);
        let f = m.declare_function(syms.intern("d"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let n = b.param(e, 0);
            let p = b.dyn_alloca(n, 32);
            let v = b.load(i64_, p, 8);
            b.ret(Some(v));
        }
        let parsed = round_trip(&m, &mut syms);
        let func = parsed.function(FuncId::from_index(0));
        let has = (0..func.inst_count()).any(|i| {
            matches!(
                func.inst(crate::ir::InstId::from_index(i)).kind,
                crate::ir::InstKind::DynAlloca { align: 32 }
            )
        });
        assert!(has, "round-tripped module must contain dyn_alloca align 32");
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

    /// A module exercising `syscall` with 0, 3 and 6 arguments (constant,
    /// parameter and pointer operands), including an unused result.
    fn syscall_module(syms: &mut StrInterner) -> Module {
        let mut m = Module::new("sys");
        let i64_ = m.types_mut().int(64);
        let ptr = m.types_mut().ptr();
        let sig = m.types_mut().func(vec![i64_, ptr], i64_, false);
        let f = m.declare_function(syms.intern("s"), sig);
        {
            let mut b = m.build(f);
            let e = b.create_entry_block();
            let x = b.param(e, 0);
            let p = b.param(e, 1);
            let nr_pid = b.const_i64(i64_, 39);
            let pid = b.syscall(nr_pid, &[]);
            let one = b.const_i64(i64_, 1);
            b.syscall(one, &[one, p, x]);
            let nine = b.const_i64(i64_, 9);
            let r = b.syscall(nine, &[x, pid, one, x, p, x]);
            b.ret(Some(r));
        }
        m
    }

    #[test]
    fn syscall_round_trips() {
        let mut syms = StrInterner::new();
        let m = syscall_module(&mut syms);
        let text = print_module(&m, &syms);
        assert!(text.contains("= syscall i64 39 : i64"), "{text}");
        assert!(text.contains("syscall i64 1, i64 1, %"), "{text}");
        let parsed = round_trip(&m, &mut syms);
        let func = parsed.function(FuncId::from_index(0));
        let arities: Vec<usize> = (0..func.inst_count())
            .map(|i| func.inst(crate::ir::InstId::from_index(i)))
            .filter(|d| matches!(d.kind, crate::ir::InstKind::Syscall))
            .map(|d| d.operands().len())
            .collect();
        assert_eq!(arities, vec![1, 4, 7], "syscall operand counts survive the round trip");
    }

    #[test]
    fn syscall_parses_from_source_and_rejects_non_i64_result() {
        let ok = "module \"x\"\nfunc @f(i64) -> i64 {\nentry ^0(%a: i64):\n  %r = syscall i64 60, %a : i64\n  ret %r\n}\n";
        let mut syms = StrInterner::new();
        let m = parse_module(ok, file(), &mut syms).expect("parse");
        assert!(crate::verify::verify_module(&m).is_ok());

        let bad = "module \"x\"\nfunc @f(i64) -> i32 {\nentry ^0(%a: i64):\n  %r = syscall i64 60, %a : i32\n  ret %r\n}\n";
        let mut syms = StrInterner::new();
        assert!(parse_module(bad, file(), &mut syms).is_err(), "a syscall result must be i64");
    }

    /// Global attributes (linkage / constant / detached), address constants with
    /// positive, negative, and forward references (to a later global and a later
    /// function), and nested aggregates survive a print → parse → print round trip
    /// and keep their structure.
    const GLOBAL_DATA_SRC: &str = r#"module "gd"

global internal constant @tab : [3 x ptr] = [3 x ptr] (ptr @x, ptr @x + 16, ptr @f)

global weak @x : {i8, [2 x i32]} = {i8, [2 x i32]} (i8 -1, [2 x i32] (i32 1, i32 2))

global constant detached @d : ptr = ptr @tab - 8

global @ext : i32

func @f() -> void {
entry ^0:
  ret
}
"#;

    /// Linkage and visibility keywords on globals and functions print in
    /// canonical order and round-trip.
    #[test]
    fn visibility_and_function_linkage_round_trip() {
        let src = "module \"v\"\n\n\
                   global hidden @h : i32 = i32 1\n\n\
                   global internal protected constant @p : i32 = i32 2\n\n\
                   func weak hidden @w() -> void {\nentry ^0:\n  ret\n}\n\n\
                   func internal @i() -> void {\nentry ^0:\n  ret\n}\n\n\
                   func hidden @ext() -> void\n";
        let mut syms = StrInterner::new();
        let m = parse_module(src, file(), &mut syms).expect("parse");
        assert_eq!(print_module(&m, &syms), src, "the source is in canonical form");
        let parsed = round_trip(&m, &mut syms);
        assert_eq!(parsed.global_attrs(GlobalId::from_index(0)).visibility, Visibility::Hidden);
        assert_eq!(parsed.global_attrs(GlobalId::from_index(1)).visibility, Visibility::Protected);
        let attrs: Vec<FuncAttrs> = parsed.functions().map(|f| f.attrs.clone()).collect();
        assert_eq!(attrs[0], FuncAttrs::new(Linkage::Weak, Visibility::Hidden));
        assert_eq!(attrs[1], FuncAttrs::new(Linkage::Internal, Visibility::Default));
        assert_eq!(attrs[2], FuncAttrs::new(Linkage::External, Visibility::Hidden));
    }

    #[test]
    fn global_attrs_and_address_constants_round_trip() {
        let mut syms = StrInterner::new();
        let m = parse_module(GLOBAL_DATA_SRC, file(), &mut syms).expect("parse");
        assert_eq!(print_module(&m, &syms), GLOBAL_DATA_SRC, "the source is in canonical form");
        let parsed = round_trip(&m, &mut syms);
        crate::verify::verify_module(&parsed).expect("verifies");

        let attrs: Vec<GlobalAttrs> =
            (0..parsed.global_count()).map(|i| parsed.global_attrs(GlobalId::from_index(i))).collect();
        assert_eq!(
            attrs[0],
            GlobalAttrs { linkage: Linkage::Internal, constant: true, ..GlobalAttrs::DEFAULT }
        );
        assert_eq!(attrs[1], GlobalAttrs { linkage: Linkage::Weak, ..GlobalAttrs::DEFAULT });
        assert_eq!(attrs[2], GlobalAttrs { constant: true, detached: true, ..GlobalAttrs::DEFAULT });
        assert_eq!(attrs[3], GlobalAttrs::DEFAULT);

        let tab = parsed.global(GlobalId::from_index(0)).init.unwrap();
        let Const::Aggregate { elems, .. } = parsed.consts().get(tab) else { panic!("aggregate") };
        let addrs: Vec<(AddrTarget, i64)> = elems
            .iter()
            .map(|&e| match parsed.consts().get(e) {
                Const::Addr { target, offset, .. } => (*target, *offset),
                other => panic!("expected an address constant, got {other:?}"),
            })
            .collect();
        assert_eq!(
            addrs,
            [
                (AddrTarget::Global(GlobalId::from_index(1)), 0),
                (AddrTarget::Global(GlobalId::from_index(1)), 16),
                (AddrTarget::Func(FuncId::from_index(0)), 0),
            ]
        );
        let d = parsed.global(GlobalId::from_index(2)).init.unwrap();
        assert!(matches!(parsed.consts().get(d), Const::Addr { offset: -8, .. }));
    }

    /// `[N x i8] "…"` is parse-only sugar for the element form, with the added
    /// `\0`, `\r`, and `\xHH` escapes; the length must match exactly.
    #[test]
    fn string_initializer_sugar() {
        let src = "module \"s\"\nglobal constant @m : [6 x i8] = [6 x i8] \"a\\x41\\n\\r\\t\\0\"\n";
        let mut syms = StrInterner::new();
        let m = parse_module(src, file(), &mut syms).expect("parse");
        let text = print_module(&m, &syms);
        assert!(
            text.contains("[6 x i8] (i8 97, i8 65, i8 10, i8 13, i8 9, i8 0)"),
            "printed in element form:\n{text}"
        );
        round_trip(&m, &mut syms);

        for (bad, why) in [
            ("global @m : [3 x i8] = [3 x i8] \"ab\"", "length mismatch"),
            ("global @m : [2 x i32] = [2 x i32] \"ab\"", "non-i8 element type"),
            ("global @m : [1 x i8] = [1 x i8] \"\\x80\"", "non-ASCII \\x escape"),
        ] {
            let src = format!("module \"s\"\n{bad}\n");
            let mut syms = StrInterner::new();
            assert!(parse_module(&src, file(), &mut syms).is_err(), "should reject: {why}");
        }
    }

    #[test]
    fn address_constant_errors() {
        // Unknown symbol.
        let src = "module \"e\"\nglobal @p : ptr = ptr @nope\n";
        let mut syms = StrInterner::new();
        assert!(parse_module(src, file(), &mut syms).is_err());
        // An address constant is not an instruction operand.
        let src = "module \"e\"\nglobal @g : i64 = i64 0\nfunc @f() -> ptr {\nentry ^0:\n  ret ptr @g\n}\n";
        let mut syms = StrInterner::new();
        assert!(parse_module(src, file(), &mut syms).is_err());
    }

    /// The pre-existing builder API (`Module::add_global`) records detached
    /// globals, which print with the `detached` keyword and round-trip as such.
    #[test]
    fn builder_add_global_is_detached() {
        let mut syms = StrInterner::new();
        let mut m = Module::new("b");
        let i32t = m.types_mut().int(32);
        let c = m.intern_const(Const::Int { ty: i32t, value: puremp::Int::from_i64(3) });
        m.add_global(Global { name: syms.intern("g"), ty: i32t, init: Some(c) });
        let text = print_module(&m, &syms);
        assert!(text.contains("global detached @g : i32 = i32 3"), "{text}");
        let parsed = round_trip(&m, &mut syms);
        assert_eq!(parsed.global_attrs(GlobalId::from_index(0)), GlobalAttrs::DETACHED);
    }

    #[test]
    fn volatile_and_atomics_round_trip() {
        let mut syms = StrInterner::new();
        let m = crate::ir::tests::atomics_module(&mut syms);
        let text = print_module(&m, &syms);
        for needle in [
            "= load volatile @g align 4 : i32",
            "store volatile %",
            "= atomic_load acquire %0 align 8 : i64",
            "atomic_store release %",
            "= atomic_rmw nand acq_rel %0, i64 4 align 8 : i64",
            "= atomic_rmw umin seq_cst @g, i32 10 align 4 : i32",
            "= cmpxchg seq_cst relaxed %0, %1, i64 42 align 8 : i64",
            "= cmpxchg acq_rel acquire @gp, ptr null, %0 align 16 : ptr",
            "fence acquire\n",
            "fence seq_cst\n",
        ] {
            assert!(text.contains(needle), "missing `{needle}` in\n{text}");
        }
        let parsed = round_trip(&m, &mut syms);
        assert!(crate::verify::verify_module(&parsed).is_ok());
        // The flags and orderings are structurally preserved, not just printed.
        let a = m.function(FuncId::from_index(0));
        let b = parsed.function(FuncId::from_index(0));
        let kinds = |f: &Function| -> Vec<InstKind> {
            f.blocks().flat_map(|(_, bl)| bl.insts().iter().map(|&i| f.inst(i).kind.clone())).collect()
        };
        assert_eq!(kinds(a), kinds(b));
    }

    #[test]
    fn plain_load_and_store_print_as_before() {
        // A non-volatile access keeps its original spelling.
        let src = "module \"x\"\nfunc @f(ptr) -> i32 {\nentry ^0(%p: ptr):\n  %v = load %p align 4 : i32\n  store %v, %p align 4 : i32\n  ret %v\n}\n";
        let mut syms = StrInterner::new();
        let m = parse_module(src, file(), &mut syms).expect("parse");
        let text = print_module(&m, &syms);
        assert!(text.contains("= load %0 align 4 : i32"), "{text}");
        assert!(text.contains("  store %1, %0 align 4 : i32"), "{text}");
        assert!(!text.contains("volatile"), "{text}");
    }

    #[test]
    fn bad_orderings_and_rmw_ops_are_parse_errors() {
        let wrap = |body: &str| {
            format!("module \"x\"\nfunc @f(ptr) -> void {{\nentry ^0(%p: ptr):\n  {body}\n  ret\n}}\n")
        };
        for (body, why) in [
            ("fence monotonic", "not an ordering name"),
            ("fence", "missing ordering"),
            ("%x = atomic_rmw frob seq_cst %p, i32 1 align 4 : i32", "unknown rmw op"),
            ("%x = atomic_load %p align 4 : i32", "missing ordering"),
            ("%x = cmpxchg seq_cst %p, i32 0, i32 1 align 4 : i32", "missing failure ordering"),
            ("%x = atomic_load acquire %p : i32", "missing align"),
        ] {
            let mut syms = StrInterner::new();
            assert!(parse_module(&wrap(body), file(), &mut syms).is_err(), "{why}: `{body}` should not parse");
        }
    }

    /// A module for a 16-bit target with a second (program-memory) address
    /// space: its `target`/`datalayout` header, a global placed in space 1, and
    /// `ptr addrspace(1)` values all round-trip, and it verifies.
    const ADDRSPACE_SRC: &str = "module \"avr\"
target \"avr\"
datalayout \"e-p:16:8-p1:16:8-i8:8-i16:8-i32:8-i64:8-f16:8-f32:8-f64:8-S8-n8-P1\"

global constant addrspace(1) @tbl : [2 x i8] = [2 x i8] (i8 1, i8 2)

global @ptab : ptr addrspace(1) = ptr addrspace(1) @tbl + 1

func @rd(i16) -> i8 {
entry ^0(%0: i16):
  %1 = ptr_add @tbl, %0 : ptr addrspace(1)
  %2 = load %1 align 1 : i8
  ret %2
}
";

    #[test]
    fn target_datalayout_and_addrspace_round_trip() {
        let mut syms = StrInterner::new();
        let m = parse_module(ADDRSPACE_SRC, file(), &mut syms).expect("parse");
        assert_eq!(print_module(&m, &syms), ADDRSPACE_SRC, "the source is in canonical form");
        let parsed = round_trip(&m, &mut syms);
        assert_eq!(parsed.target(), Some("avr"));
        assert_eq!(parsed.data_layout().pointer_bits(0), 16);
        assert_eq!(parsed.data_layout().program_addr_space(), 1);
        assert_eq!(parsed.global_addr_space(GlobalId::from_index(0)), 1);
        assert_eq!(parsed.global_addr_space(GlobalId::from_index(1)), 0);
        crate::verify::verify_module(&parsed).expect("verifies");
        // Function references are pointers into the program address space.
        let f = parsed.function(FuncId::from_index(0));
        let at = (0..f.value_count())
            .map(crate::ir::ValueId::from_index)
            .find(|&v| matches!(f.value(v).def, ValueDef::Global(_)))
            .expect("a global reference");
        assert_eq!(parsed.types().get(f.value_type(at)), &Type::PtrIn(1));
    }

    /// Every combination of global linkage, visibility, `constant`, `detached`
    /// and `addrspace`, and of function linkage and visibility, in a module
    /// with a target and a data layout, parses and prints back verbatim.
    #[test]
    fn every_attribute_combination_round_trips() {
        let mut src = String::from(
            "module \"combo\"\ntarget \"avr\"\ndatalayout \"e-p:16:8-p1:16:8-i8:8-i16:8-i32:8-i64:8-f16:8-f32:8-f64:8-S8-n8-P1\"\n",
        );
        // The printer writes every global before every function.
        let mut funcs = String::new();
        let mut n = 0;
        for linkage in ["", "internal ", "weak "] {
            for vis in ["", "hidden ", "protected "] {
                for constant in ["", "constant "] {
                    for detached in ["", "detached "] {
                        for space in ["", "addrspace(1) "] {
                            src.push_str(&format!(
                                "\nglobal {linkage}{vis}{constant}{detached}{space}@g{n} : i16 = i16 {n}\n"
                            ));
                            n += 1;
                        }
                    }
                }
                funcs.push_str(&format!("\nfunc {linkage}{vis}@f{n}(ptr addrspace(1)) -> i8 {{\nentry ^0(%0: ptr addrspace(1)):\n  %1 = load %0 align 1 : i8\n  ret %1\n}}\n"));
                n += 1;
            }
        }
        src.push_str(&funcs);
        let mut syms = StrInterner::new();
        let m = parse_module(&src, file(), &mut syms).expect("parse");
        assert_eq!(print_module(&m, &syms), src, "every combination prints back verbatim");
        let parsed = round_trip(&m, &mut syms);
        assert_eq!(parsed.global_count(), 72);
        let spaces: Vec<u32> = (0..72).map(|g| parsed.global_addr_space(GlobalId::from_index(g))).collect();
        assert_eq!(spaces.iter().filter(|&&s| s == 1).count(), 36);
        // And through the binary form.
        let bytes = crate::ir::binary::encode(&parsed, &syms);
        let back = crate::ir::binary::decode(&bytes, &mut syms).expect("decode");
        assert_eq!(print_module(&back, &syms), src);
    }

    #[test]
    fn default_layout_and_space_zero_print_as_before() {
        // No `target`, the LP64 layout, and `addrspace(0)` print nothing new.
        let src = "module \"x\"\ndatalayout \"e\"\n\nglobal addrspace(0) @g : ptr addrspace(0) = ptr null\n";
        let mut syms = StrInterner::new();
        let m = parse_module(src, file(), &mut syms).expect("parse");
        assert_eq!(print_module(&m, &syms), "module \"x\"\n\nglobal @g : ptr = ptr null\n");
    }

    #[test]
    fn bad_datalayout_is_a_parse_error() {
        let mut syms = StrInterner::new();
        let err = parse_module("module \"x\"\ndatalayout \"p:12:8\"\n", file(), &mut syms).unwrap_err();
        assert!(format!("{err:?}").contains("pointer width 12"), "{err:?}");
        assert!(parse_module("module \"x\"\ntarget x86\n", file(), &mut syms).is_err());
        assert!(parse_module("module \"x\"\nfunc @f(ptr addrspace) -> void\n", file(), &mut syms).is_err());
    }
}
