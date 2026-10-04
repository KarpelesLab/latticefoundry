//! GCC-style inline assembly on x86-64 (`docs/ir-design.md` §6i).
//!
//! An `inline_asm` goes through three steps, each written from GCC's
//! documented constraint semantics (the i386 machine constraints and operand
//! modifiers of the GCC manual), not from any compiler's code:
//!
//! 1. **Planning** ([`plan`]): each operand's constraint picks a place — a
//!    register of a class (`r`, `q`, `x`, ...), a fixed register (`a`, `b`,
//!    `c`, `d`, `S`, `D`, and a free one of `Q`/`R`), memory (`m`), an
//!    immediate (`i`, `n` and the range letters), a symbol (`i` on an
//!    address), or the register of the output it is tied to (`"0"`, `+`).
//!    The plan also resolves the clobber list and checks what GCC rejects
//!    (two outputs in one register, an early-clobber output in an input's
//!    register, a clobbered operand register, too many registers).
//!    [`check_inline_asm`] runs it ahead of code generation, so a bad
//!    statement is a clean error.
//! 2. **Selection** (`X86_64Target::lower_inline_asm`): every register input
//!    is copied into a fresh virtual register right before the asm, every
//!    fixed-register input is moved into its register in one consecutive run
//!    (the range-based fixed-register model `syscall` and `div` use), and the
//!    asm is one [`X86Op::InlineAsm`] instruction whose operands are those
//!    registers (outputs as defs, tied and `+` outputs as both), the pointer
//!    registers of memory operands, the immediates and symbols, and a def of
//!    every clobbered register. Each output is copied out of its register
//!    right after. Every asm operand is live at the asm, so the allocator
//!    gives inputs and outputs distinct registers: early clobber (`&`) always
//!    holds. Clobbered callee-saved registers are saved by the prologue like
//!    any other written one.
//! 3. **Encoding** ([`encode_inline_asm`]): after allocation the template is
//!    instantiated ([`substitute`]: `%0`, `%[name]`, the `b`/`h`/`w`/`k`/`q`
//!    size modifiers, `c`/`P`/`n`/`a`/`z`/`H`/`x`/`t`/`g`/`V`, `%=`, `%%`,
//!    `{att|intel}`), assembled on its own with `rsasm` (AT&T syntax), and the
//!    bytes are spliced into the function with their relocations. Labels the
//!    template defines (`1:`/`1b`/`1f`, `.L%=`) are local to the statement.

use super::{X86_64Target, X86Op, def, def_v, imm, use_p, use_v};
use crate::codegen::isel::{Lower, TargetIsel};
use crate::codegen::mir::{
    MachineAsm, MachineAsmKind, MachineAsmOperand, MachineInst, MachineOperand, PReg, Reg, RegClass, VReg,
};
use crate::ir::inst::{AsmSlot, InlineAsm, InstKind};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ValueDef};
use crate::ir::{Function, InstData, Module, ValueId};
use crate::mc::asm::{AsmFragment, FragmentTarget};
use crate::mc::emit::{Emitter, Ref};
use crate::mc::object::RelocKind;
use crate::support::StrInterner;
use crate::target::TargetArch;
use crate::target::x86_64::regs::{self, RAX, RBX, RCX, RDI, RDX, RSI};

use puremp::Int;

// ===========================================================================
// Constraints
// ===========================================================================

/// What one constraint string allows (the union over its alternatives).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Letters {
    gpr: bool,
    xmm: bool,
    mem: bool,
    imm: bool,
    sym: bool,
    /// A register its letter names (`a`, `b`, ...) or a `{reg}` constraint
    /// (how a front end passes a GNU register-asm variable's register).
    fixed: Option<PReg>,
    /// A set its letter restricts to (`Q`, `R`).
    subset: Option<&'static [u16]>,
    /// The range an immediate letter allows (inclusive).
    range: Option<(i128, i128)>,
    /// A matching constraint (`"0"`, `"[name]"`): the text after modifiers.
    tied: Option<String>,
    early: bool,
}

/// Parse an x86-64 constraint (GCC's common and i386 machine constraints).
fn letters(c: &str) -> Result<Letters, String> {
    let mut l = Letters::default();
    let body = c.trim_start_matches(['=', '+']);
    let mut chars = body.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '&' => l.early = true,
            '%' | ',' | '*' | '?' | '!' | ' ' | '\t' => {}
            '0'..='9' => {
                let mut s = ch.to_string();
                while let Some(&d) = chars.peek().filter(|d| d.is_ascii_digit()) {
                    s.push(d);
                    chars.next();
                }
                l.tied = Some(s);
            }
            '[' => {
                let mut s = String::from("[");
                for d in chars.by_ref() {
                    s.push(d);
                    if d == ']' {
                        break;
                    }
                }
                l.tied = Some(s);
            }
            'r' | 'q' | 'l' | 'p' | 'U' => l.gpr = true,
            'Q' => {
                l.gpr = true;
                l.subset = Some(&[RAX, RBX, RCX, RDX]);
            }
            'R' => {
                l.gpr = true;
                l.subset = Some(&[RAX, RBX, RCX, RDX, RSI, RDI]);
            }
            'a' => l.fixed = Some(regs::gpr(RAX)),
            'b' => l.fixed = Some(regs::gpr(RBX)),
            'c' => l.fixed = Some(regs::gpr(RCX)),
            'd' => l.fixed = Some(regs::gpr(RDX)),
            'S' => l.fixed = Some(regs::gpr(RSI)),
            'D' => l.fixed = Some(regs::gpr(RDI)),
            'x' | 'v' => l.xmm = true,
            'm' | 'o' | 'V' => l.mem = true,
            'g' | 'X' => {
                l.gpr = true;
                l.mem = true;
                l.imm = true;
                l.sym = true;
            }
            'i' => {
                l.imm = true;
                l.sym = true;
            }
            's' => l.sym = true,
            'n' | 'E' | 'F' => l.imm = true,
            'I' | 'J' | 'K' | 'L' | 'M' | 'N' | 'e' | 'Z' => {
                l.imm = true;
                l.range = Some(match ch {
                    'I' => (0, 31),
                    'J' => (0, 63),
                    'K' => (-128, 127),
                    'L' => (0, 0xFFFF_FFFF),
                    'M' => (0, 3),
                    'N' => (0, 255),
                    'e' => (i128::from(i32::MIN), i128::from(i32::MAX)),
                    _ => (0, i128::from(u32::MAX)),
                });
            }
            'A' => return Err(format!("constraint `{c}`: the `A` (rdx:rax pair) constraint is not supported")),
            'f' | 't' | 'u' => return Err(format!("constraint `{c}`: x87 register constraints are not supported")),
            'y' => return Err(format!("constraint `{c}`: MMX register constraints are not supported")),
            'Y' => return Err(format!("constraint `{c}`: the `Y` constraints are not supported")),
            '{' => {
                let name: String = chars.by_ref().take_while(|&d| d != '}').collect();
                let r = match clobber_reg(&name) {
                    Ok(Some(r)) => r,
                    _ => return Err(format!("constraint `{c}`: `{name}` is not a register an operand can use")),
                };
                if r.class == RegClass::Gpr && matches!(r.num, 4 | 5) {
                    return Err(format!("constraint `{c}`: `{name}` is the stack or frame pointer"));
                }
                l.fixed = Some(r);
            }
            other => return Err(format!("constraint `{c}`: unknown constraint letter `{other}`")),
        }
    }
    Ok(l)
}

/// A clobber-list entry as a register (`None` for `memory`, `cc` and the
/// registers this backend never uses, such as the x87 stack).
fn clobber_reg(name: &str) -> Result<Option<PReg>, String> {
    let n = name.trim().trim_start_matches('%').to_ascii_lowercase();
    let gpr = |r: u16| Ok(Some(regs::gpr(r)));
    match n.as_str() {
        "memory" | "cc" | "flags" | "fpsr" | "fpcr" | "dirflag" | "st" | "mxcsr" => return Ok(None),
        "rax" | "eax" | "ax" | "al" | "ah" => return gpr(RAX),
        "rbx" | "ebx" | "bx" | "bl" | "bh" => return gpr(RBX),
        "rcx" | "ecx" | "cx" | "cl" | "ch" => return gpr(RCX),
        "rdx" | "edx" | "dx" | "dl" | "dh" => return gpr(RDX),
        "rsi" | "esi" | "si" | "sil" => return gpr(RSI),
        "rdi" | "edi" | "di" | "dil" => return gpr(RDI),
        "rsp" | "esp" | "sp" | "spl" => return Ok(None),
        "rbp" | "ebp" | "bp" | "bpl" => {
            return Err("clobbering the frame pointer `rbp` is not supported".to_owned());
        }
        _ => {}
    }
    if n.starts_with("st(") || (n.starts_with("mm") && n[2..].parse::<u8>().is_ok_and(|k| k < 8)) {
        return Ok(None);
    }
    if let Some(rest) = n.strip_prefix('r') {
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        let suffix = &rest[digits.len()..];
        if let Ok(k) = digits.parse::<u16>()
            && (8..=15).contains(&k)
            && matches!(suffix, "" | "d" | "w" | "b" | "l")
        {
            return gpr(k);
        }
    }
    for p in ["xmm", "ymm", "zmm"] {
        if let Some(k) = n.strip_prefix(p).and_then(|d| d.parse::<u16>().ok()) {
            return if k < 16 {
                Ok(Some(regs::xmm(k)))
            } else {
                Err(format!("register `{name}` does not exist without AVX-512"))
            };
        }
    }
    Err(format!("unknown register `{name}` in the clobber list"))
}

// ===========================================================================
// Planning
// ===========================================================================

/// What an asm operand is bound to in the IR.
#[derive(Clone, Debug, PartialEq, Eq)]
enum OpValue {
    /// A pure (`=`) register output: no incoming value.
    None,
    /// A computed value or a pointer (memory operands).
    Value(ValueId),
    /// An integer constant.
    Const(Int),
    /// A global's address (`Global(index)`) or a function's (`Func(index)`).
    Global(u32),
    /// A function's address.
    Func(u32),
}

/// One asm operand as the planner sees it, in GCC order.
#[derive(Clone, Debug)]
struct OpInfo {
    constraint: String,
    is_output: bool,
    /// The memory operand's pointer stands for it (an indirect operand).
    indirect: bool,
    /// The value's width in bits.
    bits: u32,
    /// The value is a float or a vector (it lives in an xmm by default).
    fp: bool,
    value: OpValue,
    /// For a `+` output: the incoming value.
    incoming: Option<ValueId>,
}

/// Where an operand goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// Any allocatable general register.
    Gpr,
    /// Any allocatable xmm register.
    Xmm,
    /// This register.
    Fixed(PReg),
    /// Memory addressed by the operand's pointer.
    Mem,
    /// A non-constant input whose constraint allows only memory and
    /// immediates: stored to a stack slot that is passed as memory.
    MemSpill,
    /// An immediate.
    Imm,
    /// A symbol's address as an immediate.
    Sym,
    /// The register of output `j`.
    Tied(usize),
}

/// The plan of one asm statement.
#[derive(Clone, Debug)]
struct Plan {
    places: Vec<Place>,
    /// For each output, the input tied to it (a matching constraint).
    tied_input: Vec<Option<usize>>,
    /// The clobbered registers.
    clobbers: Vec<PReg>,
}

/// The value's bit width and whether it is float/vector.
fn type_info(types: &TypeContext, ty: TypeId) -> (u32, bool) {
    match types.get(ty) {
        Type::Int(b) => (*b, false),
        Type::Float(_) => (types.bit_width(ty).unwrap_or(64), true),
        Type::Vector(..) => ((types.size_of(ty) * 8) as u32, true),
        _ => (64, false),
    }
}

/// The planner's view of every operand of `inst`, an `inline_asm` of `f`.
fn operand_infos(module: &Module, f: &Function, inst: &InstData, asm: &InlineAsm) -> Vec<OpInfo> {
    let types = module.types();
    let slots = asm.operand_slots();
    let operand = |slot: AsmSlot| slots.iter().position(|&s| s == slot).map(|i| inst.operands()[i]);
    let classify = |v: ValueId| match &f.value(v).def {
        ValueDef::Const(c) => match module.consts().get(*c) {
            Const::Int { value, .. } => OpValue::Const(value.clone()),
            Const::Null(_) | Const::Poison(_) if !types.is_vector(f.value_type(v)) => OpValue::Const(Int::ZERO),
            _ => OpValue::Value(v),
        },
        ValueDef::Global(g) => OpValue::Global(g.index() as u32),
        ValueDef::Func(fid) => OpValue::Func(fid.index() as u32),
        _ => OpValue::Value(v),
    };
    let mut out = Vec::new();
    for (j, o) in asm.outputs.iter().enumerate() {
        let indirect = InlineAsm::is_indirect(&o.constraint);
        let v = operand(AsmSlot::Output(j));
        let (bits, fp) = o.ty.map_or((64, false), |t| type_info(types, t));
        out.push(OpInfo {
            constraint: o.constraint.clone(),
            is_output: true,
            indirect,
            bits,
            fp,
            value: if indirect { v.map_or(OpValue::None, OpValue::Value) } else { OpValue::None },
            incoming: if indirect { None } else { v },
        });
    }
    for (k, a) in asm.inputs.iter().enumerate() {
        let v = operand(AsmSlot::Input(k)).expect("every input has an operand");
        let indirect = InlineAsm::is_indirect(&a.constraint);
        let (bits, fp) = type_info(types, f.value_type(v));
        out.push(OpInfo {
            constraint: a.constraint.clone(),
            is_output: false,
            indirect,
            bits,
            fp,
            value: if indirect { OpValue::Value(v) } else { classify(v) },
            incoming: None,
        });
    }
    out
}

/// Plan an asm statement (see the module docs), or say why it cannot be
/// compiled.
fn plan(asm: &InlineAsm, infos: &[OpInfo]) -> Result<Plan, String> {
    let nout = asm.outputs.len();
    let mut places = Vec::with_capacity(infos.len());
    let mut tied_input = vec![None; nout];
    let mut parsed = Vec::with_capacity(infos.len());
    for info in infos {
        parsed.push(letters(&info.constraint)?);
    }
    for (k, (info, l)) in infos.iter().zip(&parsed).enumerate() {
        let c = &info.constraint;
        let place = if info.indirect {
            Place::Mem
        } else if let Some(t) = &l.tied {
            if info.is_output {
                return Err(format!("output constraint `{c}` cannot be a matching constraint"));
            }
            let j = asm
                .tied_output(t)
                .filter(|&j| j < nout)
                .ok_or_else(|| format!("matching constraint `{c}` names no output operand"))?;
            if !asm.is_register_output(j) {
                return Err(format!("matching constraint `{c}` names a memory output"));
            }
            if asm.outputs[j].constraint.starts_with('+') || tied_input[j].is_some() {
                return Err(format!("output {j} is already tied to another input"));
            }
            tied_input[j] = Some(k - nout);
            Place::Tied(j)
        } else if l.imm && matches!(&info.value, OpValue::Const(_)) {
            let OpValue::Const(v) = &info.value else { unreachable!() };
            if let (Some((lo, hi)), Some(x)) = (l.range, v.to_i64().map(i128::from).or_else(|| v.to_u64().map(i128::from)))
                && !(lo..=hi).contains(&x)
            {
                return Err(format!("constant {x} is out of range for constraint `{c}`"));
            }
            Place::Imm
        } else if l.sym && matches!(info.value, OpValue::Global(_) | OpValue::Func(_)) {
            Place::Sym
        } else if let Some(r) = l.fixed {
            Place::Fixed(r)
        } else if l.gpr && l.subset.is_none() && !(info.fp && l.xmm) {
            Place::Gpr
        } else if l.gpr && l.subset.is_some() {
            // Resolved to one free register of the set below.
            Place::Gpr
        } else if l.xmm {
            Place::Xmm
        } else if l.mem && !info.is_output {
            Place::MemSpill
        } else if l.imm || l.sym {
            return Err(format!("constraint `{c}` needs a constant operand"));
        } else {
            return Err(format!("constraint `{c}` cannot hold this operand"));
        };
        if info.is_output && matches!(place, Place::Imm | Place::Sym | Place::MemSpill) {
            return Err(format!("output constraint `{c}` must allow a register or memory"));
        }
        match place {
            Place::Gpr | Place::Fixed(_) if info.bits > 64 => {
                return Err(format!("a {}-bit operand does not fit a general register (`{c}`)", info.bits));
            }
            Place::Xmm if info.bits > 128 => {
                return Err(format!("a {}-bit operand does not fit an xmm register (`{c}`)", info.bits));
            }
            Place::MemSpill if info.fp || info.bits > 64 => {
                return Err(format!("constraint `{c}` needs this operand in memory; pass its address instead"));
            }
            _ => {}
        }
        places.push(place);
    }

    let mut clobbers: Vec<PReg> = Vec::new();
    for c in &asm.clobbers {
        if let Some(r) = clobber_reg(c)?
            && !clobbers.contains(&r)
        {
            clobbers.push(r);
        }
    }

    // Fixed registers: outputs distinct; inputs distinct unless they are the
    // same value; an early-clobber output apart from every input; no clobber
    // on an operand.
    let mut out_fixed: Vec<PReg> = Vec::new();
    let mut in_fixed: Vec<(PReg, &OpValue)> = Vec::new();
    for (k, place) in places.iter().enumerate() {
        let Place::Fixed(r) = *place else { continue };
        let info = &infos[k];
        if info.is_output {
            if out_fixed.contains(&r) {
                return Err(format!("two outputs need register {}", reg_name(r, 64)));
            }
            out_fixed.push(r);
        } else {
            if in_fixed.iter().any(|&(q, v)| q == r && *v != info.value) {
                return Err(format!("two inputs need register {}", reg_name(r, 64)));
            }
            in_fixed.push((r, &info.value));
        }
        if clobbers.contains(&r) {
            return Err(format!("register {} is both an operand and clobbered", reg_name(r, 64)));
        }
    }
    for (j, l) in parsed.iter().enumerate().take(nout) {
        if let Place::Fixed(r) = places[j]
            && l.early
            && in_fixed.iter().any(|&(q, _)| q == r)
        {
            return Err(format!("early-clobber output {j} and an input both need register {}", reg_name(r, 64)));
        }
    }
    // A tied input of a fixed output, or a `+` one, is moved into that
    // register, so it competes with the fixed inputs too.
    for j in 0..nout {
        let Place::Fixed(r) = places[j] else { continue };
        let tied = tied_input[j].is_some() || infos[j].incoming.is_some();
        if tied && in_fixed.iter().any(|&(q, _)| q == r) {
            return Err(format!("register {} holds both a tied output's input and another input", reg_name(r, 64)));
        }
    }

    // `Q`/`R`: one free register of the set, picked statically.
    let mut taken: Vec<PReg> = out_fixed.iter().copied().chain(in_fixed.iter().map(|&(r, _)| r)).chain(clobbers.iter().copied()).collect();
    for (k, l) in parsed.iter().enumerate() {
        if let (Some(set), Place::Gpr) = (l.subset, places[k]) {
            let r = set
                .iter()
                .map(|&n| regs::gpr(n))
                .find(|r| !taken.contains(r))
                .ok_or_else(|| format!("no register of constraint `{}` is free", infos[k].constraint))?;
            taken.push(r);
            places[k] = Place::Fixed(r);
        }
    }

    // Register pressure at the asm: every operand register is live there.
    let gpr_free = [RAX, RCX, RDX, RSI, RDI, 8, 9, 12, 13, 14, 15]
        .iter()
        .filter(|&&n| !taken.contains(&regs::gpr(n)))
        .count();
    let xmm_free = (0u16..=12).filter(|&n| !taken.contains(&regs::xmm(n))).count();
    let need_gpr = places.iter().filter(|p| matches!(p, Place::Gpr | Place::Mem | Place::MemSpill)).count();
    let need_xmm = places.iter().filter(|p| matches!(p, Place::Xmm)).count();
    if need_gpr > gpr_free || need_xmm > xmm_free {
        return Err("the asm needs more registers than are available".to_owned());
    }
    Ok(Plan { places, tied_input, clobbers })
}

// ===========================================================================
// Template instantiation
// ===========================================================================

/// An operand after allocation, as the template sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Placed {
    Reg(PReg),
    Mem(PReg),
    Imm(i128),
    Sym(String),
}

/// AT&T name of `r` at `bits` (8/16/32/64; an xmm ignores it).
fn reg_name(r: PReg, bits: u32) -> String {
    if r.class == RegClass::Fp {
        return format!("%xmm{}", r.num);
    }
    const BASE: [&str; 8] = ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di"];
    let n = r.num as usize;
    if n >= 8 {
        let suffix = match bits {
            8 => "b",
            16 => "w",
            32 => "d",
            _ => "",
        };
        return format!("%r{n}{suffix}");
    }
    let b = BASE[n];
    match bits {
        8 if n < 4 => format!("%{}l", &b[..1]),
        8 => format!("%{b}l"),
        16 => format!("%{b}"),
        32 => format!("%e{b}"),
        _ => format!("%r{b}"),
    }
}

/// The register size `%N` prints for a `bits`-wide value.
fn default_size(bits: u32) -> u32 {
    match bits {
        0..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        _ => 64,
    }
}

/// Instantiate `template` for x86-64 AT&T syntax: `ops` are the operands in
/// GCC order, with their widths and names; `unique` is what `%=` prints.
fn substitute(template: &str, ops: &[(Placed, u32, Option<&str>)], unique: u64) -> Result<String, String> {
    let mut out = String::with_capacity(template.len() + 16);
    let mut chars = template.chars().peekable();
    // Inside `{att|intel}`: 0 = outside, 1 = in the first alternative, 2 =
    // skipping the others.
    let mut dialect = 0;
    while let Some(ch) = chars.next() {
        match ch {
            '{' if dialect == 0 => {
                dialect = 1;
                continue;
            }
            '|' if dialect == 1 => {
                dialect = 2;
                continue;
            }
            '}' if dialect != 0 => {
                dialect = 0;
                continue;
            }
            _ if dialect == 2 => continue,
            '%' => {}
            c => {
                out.push(c);
                continue;
            }
        }
        let Some(&next) = chars.peek() else {
            return Err("the template ends with a lone `%`".to_owned());
        };
        match next {
            '%' | '{' | '}' | '|' => {
                chars.next();
                out.push(next);
                continue;
            }
            '=' => {
                chars.next();
                out.push_str(&unique.to_string());
                continue;
            }
            _ => {}
        }
        let modifier = if next.is_ascii_alphabetic() {
            chars.next();
            Some(next)
        } else {
            None
        };
        let index = match chars.peek() {
            Some('[') => {
                chars.next();
                let mut name = String::new();
                for c in chars.by_ref() {
                    if c == ']' {
                        break;
                    }
                    name.push(c);
                }
                ops.iter()
                    .position(|(_, _, n)| *n == Some(name.as_str()))
                    .ok_or_else(|| format!("`%[{name}]` names no operand"))?
            }
            Some(d) if d.is_ascii_digit() => {
                let mut s = String::new();
                while let Some(&d) = chars.peek().filter(|d| d.is_ascii_digit()) {
                    s.push(d);
                    chars.next();
                }
                s.parse::<usize>().map_err(|_| format!("bad operand number `%{s}`"))?
            }
            _ => {
                return Err(match modifier {
                    Some(m) => format!("`%{m}` is not followed by an operand"),
                    None => "a `%` is not followed by an operand (write `%%` for a literal `%`)".to_owned(),
                });
            }
        };
        let (placed, bits, _) = ops.get(index).ok_or_else(|| format!("operand number {index} is out of range"))?;
        out.push_str(&operand_text(placed, *bits, modifier)?);
    }
    Ok(out)
}

/// One operand's text under an optional modifier.
fn operand_text(placed: &Placed, bits: u32, modifier: Option<char>) -> Result<String, String> {
    let bad = |m: char| Err(format!("operand modifier `%{m}` does not apply to this operand"));
    Ok(match (placed, modifier) {
        (Placed::Reg(r), None) => reg_name(*r, default_size(bits)),
        (Placed::Reg(r), Some('b')) => reg_name(*r, 8),
        (Placed::Reg(r), Some('w')) => reg_name(*r, 16),
        (Placed::Reg(r), Some('k')) => reg_name(*r, 32),
        (Placed::Reg(r), Some('q')) => reg_name(*r, 64),
        (Placed::Reg(r), Some('h')) if r.class == RegClass::Gpr && r.num < 4 => {
            format!("%{}h", ["a", "c", "d", "b"][r.num as usize])
        }
        (Placed::Reg(r), Some('x')) if r.class == RegClass::Fp => format!("%xmm{}", r.num),
        (Placed::Reg(r), Some('t')) if r.class == RegClass::Fp => format!("%ymm{}", r.num),
        (Placed::Reg(r), Some('g')) if r.class == RegClass::Fp => format!("%zmm{}", r.num),
        (Placed::Reg(r), Some('V')) => reg_name(*r, default_size(bits)).trim_start_matches('%').to_owned(),
        (Placed::Reg(r), Some('a')) => format!("({})", reg_name(*r, 64)),
        (Placed::Reg(_) | Placed::Mem(_), Some('z')) => {
            match default_size(bits) {
                8 => "b",
                16 => "w",
                32 => "l",
                _ => "q",
            }
            .to_owned()
        }
        (Placed::Mem(r), None | Some('a' | 'b' | 'w' | 'k' | 'q' | 'c' | 'P' | 'p')) => format!("({})", reg_name(*r, 64)),
        (Placed::Mem(r), Some('H')) => format!("8({})", reg_name(*r, 64)),
        // A size modifier leaves a constant as it is (`inb %w1` with a
        // constant port prints `$128`).
        (Placed::Imm(v), None | Some('b' | 'w' | 'k' | 'q' | 'h')) => format!("${v}"),
        (Placed::Imm(v), Some('c' | 'P' | 'p' | 'a')) => v.to_string(),
        (Placed::Imm(v), Some('n')) => (-v).to_string(),
        (Placed::Sym(s), None) => format!("${s}"),
        (Placed::Sym(s), Some('c' | 'P' | 'p' | 'a')) => s.clone(),
        (_, Some(m)) if m.is_ascii_alphabetic() => return bad(m),
        _ => return Err("unsupported operand".to_owned()),
    })
}

// ===========================================================================
// Checking ahead of code generation
// ===========================================================================

/// Check every `inline_asm` of `module` for x86-64 code generation: its
/// constraints and clobbers are planned (see [`plan`]) and its template is
/// instantiated with stand-in registers and assembled, so that a bad
/// statement is reported here, naming its function, rather than as a backend
/// panic.
///
/// # Errors
///
/// The first problem found, as `function `f`: inline asm: ...`.
pub fn check_inline_asm(module: &Module, syms: &StrInterner) -> Result<(), String> {
    for f in module.functions() {
        let fname = syms.resolve(f.name);
        for (_, b) in f.blocks() {
            for &i in b.insts() {
                let inst = f.inst(i);
                let InstKind::InlineAsm(asm) = &inst.kind else { continue };
                check_one(module, syms, f, inst, asm).map_err(|e| format!("function `{fname}`: inline asm: {e}"))?;
            }
        }
    }
    Ok(())
}

fn check_one(module: &Module, syms: &StrInterner, f: &Function, inst: &InstData, asm: &InlineAsm) -> Result<(), String> {
    let infos = operand_infos(module, f, inst, asm);
    let plan = plan(asm, &infos)?;
    // Stand-in registers: distinct free ones per class.
    let mut taken: Vec<PReg> = plan.clobbers.clone();
    taken.extend(plan.places.iter().filter_map(|p| match p {
        Place::Fixed(r) => Some(*r),
        _ => None,
    }));
    let next = |class: RegClass, taken: &mut Vec<PReg>| -> PReg {
        let pool: Vec<PReg> = match class {
            RegClass::Gpr => [RAX, RCX, RDX, RSI, RDI, 8, 9, 12, 13, 14, 15].map(regs::gpr).to_vec(),
            RegClass::Fp => (0u16..=12).map(regs::xmm).collect(),
        };
        let r = pool.into_iter().find(|r| !taken.contains(r)).expect("the plan checked register pressure");
        taken.push(r);
        r
    };
    let mut placed: Vec<Placed> = Vec::with_capacity(infos.len());
    for (k, info) in infos.iter().enumerate() {
        let p = match plan.places[k] {
            Place::Gpr => Placed::Reg(next(RegClass::Gpr, &mut taken)),
            Place::Xmm => Placed::Reg(next(RegClass::Fp, &mut taken)),
            Place::Fixed(r) => Placed::Reg(r),
            Place::Mem | Place::MemSpill => Placed::Mem(next(RegClass::Gpr, &mut taken)),
            Place::Imm => match &info.value {
                OpValue::Const(v) => Placed::Imm(const_i128(v)),
                _ => unreachable!("planned as an immediate"),
            },
            Place::Sym => Placed::Sym(match info.value {
                OpValue::Global(g) => syms.resolve(module.global(crate::ir::GlobalId::from_index(g as usize)).name).to_owned(),
                OpValue::Func(x) => syms.resolve(module.function(crate::ir::FuncId::from_index(x as usize)).name).to_owned(),
                _ => unreachable!("planned as a symbol"),
            }),
            Place::Tied(j) => placed[j].clone(),
        };
        placed.push(p);
    }
    let ops: Vec<(Placed, u32, Option<&str>)> = placed
        .into_iter()
        .zip(&infos)
        .enumerate()
        .map(|(k, (p, info))| (p, info.bits, operand_name(asm, k)))
        .collect();
    let text = substitute(&asm.template, &ops, 0)?;
    if !text.trim().is_empty() {
        crate::mc::asm::assemble_fragment(TargetArch::X86_64, "inline asm", &text)?;
    }
    Ok(())
}

/// The `[name]` of GCC operand `k`.
fn operand_name(asm: &InlineAsm, k: usize) -> Option<&str> {
    let n = asm.outputs.len();
    if k < n { asm.outputs[k].name.as_deref() } else { asm.inputs[k - n].name.as_deref() }
}

/// An immediate as a signed 128-bit value (its two's-complement bits when it
/// was stored unsigned).
fn const_i128(v: &Int) -> i128 {
    v.to_i64()
        .map(i128::from)
        .or_else(|| v.to_u64().map(|u| i128::from(u as i64)))
        .unwrap_or(0)
}

// ===========================================================================
// Instruction selection
// ===========================================================================

impl X86_64Target {
    /// The vreg that holds register output `j` of the asm whose result is
    /// value `asm_val` (shared by the asm's lowering and `asm_output`s, in
    /// whichever order they are selected).
    fn asm_out_vreg(&self, lo: &mut Lower<'_, Self>, asm_val: ValueId, j: usize, class: RegClass) -> VReg {
        if let Some(&v) = self.asm_outs.borrow().get(&(asm_val.index(), j)) {
            return v;
        }
        let v = lo.fresh_vreg(class);
        self.asm_outs.borrow_mut().insert((asm_val.index(), j), v);
        v
    }

    /// Lower an `asm_output`: copy the register the asm left output `n` in.
    pub(super) fn lower_asm_output(&self, lo: &mut Lower<'_, Self>, inst: &InstData, n: u32) {
        let asm_val = inst.operands()[0];
        let d = lo.result_reg(inst);
        let InstKind::InlineAsm(asm) = &lo.func().inst(match lo.func().value(asm_val).def {
            ValueDef::Inst(i) => i,
            _ => unreachable!("the verifier checks asm_output's operand"),
        })
        .kind
        else {
            unreachable!("the verifier checks asm_output's operand")
        };
        let src = if asm.result_output() == Some(n as usize) {
            lo.reg(asm_val)
        } else {
            let class = lo.mf().vreg_class(d);
            self.asm_out_vreg(lo, asm_val, n as usize, class)
        };
        lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(d), use_v(src)]));
    }

    /// Lower an `inline_asm` (see the module docs).
    pub(super) fn lower_inline_asm(&self, lo: &mut Lower<'_, Self>, inst: &InstData, asm: &InlineAsm) {
        let infos = operand_infos(lo.module(), lo.func(), inst, asm);
        let plan = plan(asm, &infos).unwrap_or_else(|e| panic!("x86-64 backend: inline asm: {e}"));
        let nout = asm.outputs.len();
        let class = |p: Place| if p == Place::Xmm { RegClass::Fp } else { RegClass::Gpr };
        let may_branch = !asm.template.trim().is_empty();
        let mut ops: Vec<MachineOperand> = vec![imm(0), imm(u64::from(may_branch))];
        let mut desc: Vec<MachineAsmOperand> = Vec::with_capacity(infos.len());
        let mut fixed_moves: Vec<(PReg, VReg)> = Vec::new();
        let mut post: Vec<(VReg, Reg)> = Vec::new();
        let op_desc = |kind, slot, info: &OpInfo, fixed, name: Option<&String>| MachineAsmOperand {
            name: name.cloned(),
            kind,
            slot,
            bits: info.bits,
            fixed,
        };

        // Outputs.
        for j in 0..nout {
            let info = &infos[j];
            let name = asm.outputs[j].name.as_ref();
            match plan.places[j] {
                Place::Mem => {
                    let OpValue::Value(p) = info.value else { unreachable!("a memory output has a pointer") };
                    let src = self.oper(lo, p);
                    let t = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(t), use_v(src)]));
                    desc.push(op_desc(MachineAsmKind::Mem, ops.len(), info, false, name));
                    ops.push(use_v(t));
                }
                place => {
                    let incoming = info.incoming.or_else(|| {
                        plan.tied_input[j].map(|k| match &infos[nout + k].value {
                            OpValue::Value(v) => *v,
                            _ => inst.operands()[asm.operand_slots().iter().position(|&s| s == AsmSlot::Input(k)).expect("an input slot")],
                        })
                    });
                    let reg = match place {
                        Place::Fixed(r) => Reg::Physical(r),
                        p => Reg::Virtual(lo.fresh_vreg(class(p))),
                    };
                    if let Some(v) = incoming {
                        let src = self.oper(lo, v);
                        match reg {
                            Reg::Physical(r) => fixed_moves.push((r, src)),
                            Reg::Virtual(t) => {
                                lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(t), use_v(src)]));
                            }
                        }
                    }
                    let fixed = matches!(place, Place::Fixed(_));
                    desc.push(op_desc(MachineAsmKind::Reg, ops.len(), info, fixed, name));
                    ops.push(MachineOperand::Def(reg));
                    if incoming.is_some() {
                        ops.push(MachineOperand::Use(reg));
                    }
                    let asm_val = inst.result().expect("an asm with a register output has a result");
                    let ty = asm.outputs[j].ty.expect("a register output has a type");
                    let dest_class = if matches!(lo.types().get(ty), Type::Float(_) | Type::Vector(..)) {
                        RegClass::Fp
                    } else {
                        RegClass::Gpr
                    };
                    let dest = if asm.result_output() == Some(j) {
                        lo.result_reg(inst)
                    } else {
                        self.asm_out_vreg(lo, asm_val, j, dest_class)
                    };
                    post.push((dest, reg));
                }
            }
        }

        // Inputs.
        for k in 0..asm.inputs.len() {
            let info = &infos[nout + k];
            let name = asm.inputs[k].name.as_ref();
            let v = match &info.value {
                OpValue::Value(v) => Some(*v),
                _ => None,
            };
            match plan.places[nout + k] {
                Place::Tied(j) => {
                    let mut d = desc[j].clone();
                    d.name = name.cloned();
                    d.bits = info.bits;
                    desc.push(d);
                }
                Place::Imm => {
                    let OpValue::Const(c) = &info.value else { unreachable!("planned as an immediate") };
                    desc.push(op_desc(MachineAsmKind::Imm, ops.len(), info, false, name));
                    ops.push(MachineOperand::Imm(c.clone()));
                }
                Place::Sym => {
                    desc.push(op_desc(MachineAsmKind::Sym, ops.len(), info, false, name));
                    ops.push(match info.value {
                        OpValue::Global(g) => MachineOperand::Global(g),
                        OpValue::Func(f) => MachineOperand::Func(f),
                        _ => unreachable!("planned as a symbol"),
                    });
                }
                Place::Mem => {
                    let src = self.oper(lo, v.expect("a memory input has a pointer"));
                    let t = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(t), use_v(src)]));
                    desc.push(op_desc(MachineAsmKind::Mem, ops.len(), info, false, name));
                    ops.push(use_v(t));
                }
                Place::MemSpill => {
                    let value = self.asm_input_value(lo, inst, asm, k);
                    let src = self.oper(lo, value);
                    let slot = lo.new_slot(8, 8);
                    let t = lo.fresh_vreg(RegClass::Gpr);
                    lo.emit(self.frame_addr(t, slot));
                    lo.emit(MachineInst::new(X86Op::Store.opcode(), vec![use_v(t), use_v(src), imm(8)]));
                    desc.push(op_desc(MachineAsmKind::Mem, ops.len(), info, false, name));
                    ops.push(use_v(t));
                }
                Place::Fixed(r) => {
                    let value = self.asm_input_value(lo, inst, asm, k);
                    let src = self.oper(lo, value);
                    if !fixed_moves.iter().any(|&(q, s)| q == r && s == src) {
                        fixed_moves.push((r, src));
                    }
                    desc.push(op_desc(MachineAsmKind::Reg, ops.len(), info, true, name));
                    ops.push(use_p(r));
                }
                p @ (Place::Gpr | Place::Xmm) => {
                    let value = self.asm_input_value(lo, inst, asm, k);
                    let src = self.oper(lo, value);
                    let t = lo.fresh_vreg(class(p));
                    lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(t), use_v(src)]));
                    desc.push(op_desc(MachineAsmKind::Reg, ops.len(), info, false, name));
                    ops.push(use_v(t));
                }
            }
        }

        // The fixed-register moves, as one run right before the asm. A move
        // into `r10` or `xmm13` (the first spill-reload scratch of its class,
        // nameable by a `{reg}` constraint) goes last, so no reload feeding a
        // later move overwrites it (as for `syscall`).
        fixed_moves.sort_by_key(|&(r, _)| r == regs::gpr(10) || r == regs::xmm(13));
        for (r, src) in fixed_moves {
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def(r), use_v(src)]));
        }
        let clobbers_from = ops.len();
        ops.extend(plan.clobbers.iter().map(|&r| def(r)));
        let id = lo.add_inline_asm(MachineAsm { template: asm.template.clone(), operands: desc, clobbers_from });
        ops[0] = imm(u64::from(id));
        lo.emit(MachineInst::new(X86Op::InlineAsm.opcode(), ops));
        for (dest, reg) in post {
            lo.emit(MachineInst::new(X86Op::MovRR.opcode(), vec![def_v(dest), MachineOperand::Use(reg)]));
        }
    }

    /// The value operand of input `k`.
    fn asm_input_value(&self, _lo: &Lower<'_, Self>, inst: &InstData, asm: &InlineAsm, k: usize) -> ValueId {
        let i = asm.operand_slots().iter().position(|&s| s == AsmSlot::Input(k)).expect("an input slot");
        inst.operands()[i]
    }
}

// ===========================================================================
// Encoding
// ===========================================================================

/// The relocation kind and field width of an x86-64 ELF relocation type.
fn reloc_kind(ty: u32) -> Option<RelocKind> {
    Some(match ty {
        1 => RelocKind::Abs64,
        2 => RelocKind::Pc32,
        4 => RelocKind::Plt32,
        9 | 41 | 42 => RelocKind::GotPcRel,
        10 => RelocKind::Abs32,
        11 => RelocKind::Abs32S,
        19 => RelocKind::TlsGd,
        22 => RelocKind::GotTpOff,
        23 => RelocKind::TpOff32,
        24 => RelocKind::Pc64,
        _ => return None,
    })
}

/// The scratch registers the allocator reloads spilled operands into, which
/// an allocated asm operand must never end up in alongside a fixed one.
fn is_scratch(r: PReg) -> bool {
    match r.class {
        RegClass::Gpr => matches!(r.num, 3 | 10 | 11),
        RegClass::Fp => r.num >= 13,
    }
}

/// Encode an allocated [`X86Op::InlineAsm`] instruction: instantiate the
/// template of `asm` with the registers the allocator chose, assemble it with
/// rsasm, and splice the bytes and relocations into `e`. `func` names the
/// function (for diagnostics and for absolute references to the template's
/// own labels), `unique` is what `%=` prints.
///
/// # Panics
///
/// When the instantiated template does not assemble (which
/// [`check_inline_asm`] reports cleanly beforehand), or when the allocator
/// had to put an operand in a scratch register that collides with another
/// operand (more registers than the function has).
pub(crate) fn encode_inline_asm(
    e: &mut Emitter,
    inst: &MachineInst,
    asm: &MachineAsm,
    func: &str,
    unique: u64,
    global_name: &dyn Fn(u32) -> String,
    func_name: &dyn Fn(u32) -> String,
) {
    let reg_at = |slot: usize| match &inst.operands[slot] {
        MachineOperand::Def(Reg::Physical(p)) | MachineOperand::Use(Reg::Physical(p)) => *p,
        other => panic!("inline asm operand {slot} is not an allocated register: {other:?}"),
    };
    let mut placed = Vec::with_capacity(asm.operands.len());
    let mut used: Vec<(usize, PReg)> = Vec::new();
    for o in &asm.operands {
        let p = match o.kind {
            MachineAsmKind::Reg => Placed::Reg(reg_at(o.slot)),
            MachineAsmKind::Mem => Placed::Mem(reg_at(o.slot)),
            MachineAsmKind::Imm => match &inst.operands[o.slot] {
                MachineOperand::Imm(v) => Placed::Imm(const_i128(v)),
                other => panic!("inline asm immediate is {other:?}"),
            },
            MachineAsmKind::Sym => match inst.operands[o.slot] {
                MachineOperand::Global(g) => Placed::Sym(global_name(g)),
                MachineOperand::Func(f) => Placed::Sym(func_name(f)),
                ref other => panic!("inline asm symbol is {other:?}"),
            },
        };
        if let Placed::Reg(r) | Placed::Mem(r) = p
            && !used.iter().any(|&(s, _)| s == o.slot)
        {
            used.push((o.slot, r));
        }
        placed.push((p, o.bits, o.name.as_deref()));
    }
    // Every register operand (by slot) and clobber is a distinct register:
    // they are all live at the asm. Only a spilled operand reloaded into a
    // scratch register could break that.
    let clobbered: Vec<PReg> = inst.operands[asm.clobbers_from..].iter().filter_map(|o| o.reg().and_then(Reg::as_physical)).collect();
    for (i, &(s, r)) in used.iter().enumerate() {
        let shared = used[..i].iter().any(|&(_, q)| q == r) || clobbered.contains(&r);
        let allocated = asm.operands.iter().any(|o| o.slot == s && !o.fixed);
        if shared && allocated && is_scratch(r) {
            panic!("x86-64 backend: inline asm in `{func}` needs more registers than the function has free");
        }
    }
    let text = substitute(&asm.template, &placed, unique)
        .unwrap_or_else(|err| panic!("x86-64 backend: inline asm in `{func}`: {err}"));
    if text.trim().is_empty() {
        return;
    }
    let name = format!("inline asm in `{func}`");
    let frag = crate::mc::asm::assemble_fragment(TargetArch::X86_64, &name, &text)
        .unwrap_or_else(|err| panic!("x86-64 backend: {name}: {err}"));
    splice(e, &frag, func);
}

/// Append `frag` to `e`, turning its relocations into references: an external
/// symbol stays a symbol; a PC-relative reference to the fragment's own code
/// becomes a label; an absolute one becomes a reference to the function's
/// own symbol at the right offset.
fn splice(e: &mut Emitter, frag: &AsmFragment, func: &str) {
    let base = e.offset();
    let mut pos = 0usize;
    for r in &frag.relocs {
        let at = r.offset as usize;
        let kind = reloc_kind(r.kind)
            .unwrap_or_else(|| panic!("x86-64 backend: inline asm in `{func}`: unsupported relocation type {}", r.kind));
        let width = kind.field_width();
        e.bytes(&frag.bytes[pos..at]);
        // The emitter folds the field width into a PC-relative addend.
        let pc_adj = if kind.is_pcrel() { width as i64 } else { 0 };
        match &r.target {
            FragmentTarget::External(sym) => e.reference(kind, Ref::Symbol(sym.clone()), r.addend + pc_adj),
            FragmentTarget::Local(off) if kind.is_pcrel() => {
                let l = e.create_label();
                e.bind_label_at(l, base + off);
                e.reference(kind, Ref::Label(l), r.addend + pc_adj);
            }
            FragmentTarget::Local(off) => {
                e.reference(kind, Ref::Symbol(func.to_owned()), (base + off) as i64 + r.addend);
            }
        }
        pos = at + width;
    }
    e.bytes(&frag.bytes[pos..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::x86_64::regs::{RBP, RSP};

    fn gpr(n: u16) -> PReg {
        regs::gpr(n)
    }

    #[test]
    fn register_names_by_size() {
        assert_eq!(reg_name(gpr(RAX), 64), "%rax");
        assert_eq!(reg_name(gpr(RAX), 32), "%eax");
        assert_eq!(reg_name(gpr(RAX), 16), "%ax");
        assert_eq!(reg_name(gpr(RAX), 8), "%al");
        assert_eq!(reg_name(gpr(RSI), 8), "%sil");
        assert_eq!(reg_name(gpr(RDI), 32), "%edi");
        assert_eq!(reg_name(gpr(9), 32), "%r9d");
        assert_eq!(reg_name(gpr(12), 16), "%r12w");
        assert_eq!(reg_name(gpr(15), 8), "%r15b");
        assert_eq!(reg_name(regs::xmm(7), 64), "%xmm7");
        assert_eq!(reg_name(gpr(RSP), 64), "%rsp");
        assert_eq!(reg_name(gpr(RBP), 8), "%bpl");
    }

    #[test]
    fn substitution_and_modifiers() {
        let ops = vec![
            (Placed::Reg(gpr(RAX)), 32, Some("out")),
            (Placed::Reg(gpr(RCX)), 64, None),
            (Placed::Imm(-5), 32, Some("k")),
            (Placed::Mem(gpr(RDX)), 64, None),
            (Placed::Sym("foo".into()), 64, None),
        ];
        let s = |t: &str| substitute(t, &ops, 42).unwrap();
        assert_eq!(s("mov %1, %0"), "mov %rcx, %eax");
        assert_eq!(s("%k1 %w1 %b1 %q0 %h0"), "%ecx %cx %cl %rax %ah");
        assert_eq!(s("add %[k], %[out]"), "add $-5, %eax");
        assert_eq!(s("%c2 %n2 %c4 %4 %w2"), "-5 5 foo $foo $-5");
        assert_eq!(s("incl %3; mov%z0 %H3"), "incl (%rdx); movl 8(%rdx)");
        assert_eq!(s("1: jmp 1b; .L%=: %%rax"), "1: jmp 1b; .L42: %rax");
        assert_eq!(s("{movl %1, %0|mov %0, %1}"), "movl %rcx, %eax");
        assert_eq!(s("%{x%}"), "{x}");
        assert!(substitute("%5", &ops, 0).is_err());
        assert!(substitute("%[nope]", &ops, 0).is_err());
        assert!(substitute("%y0", &ops, 0).is_err());
        assert!(substitute("50%", &ops, 0).is_err());
    }

    #[test]
    fn constraint_letters() {
        let l = letters("=&a").unwrap();
        assert!(l.early && l.fixed == Some(gpr(RAX)));
        assert_eq!(letters("{r10}").unwrap().fixed, Some(gpr(10)));
        assert_eq!(letters("={xmm3}").unwrap().fixed, Some(regs::xmm(3)));
        assert!(letters("{rsp}").is_err() && letters("{bogus}").is_err());
        assert!(letters("+rm").unwrap().gpr);
        assert_eq!(letters("0").unwrap().tied.as_deref(), Some("0"));
        assert_eq!(letters("[x]").unwrap().tied.as_deref(), Some("[x]"));
        assert!(letters("A").is_err());
        assert!(letters("t").is_err());
        assert!(letters("w").is_err());
        assert_eq!(letters("N").unwrap().range, Some((0, 255)));
    }

    #[test]
    fn clobber_names() {
        assert_eq!(clobber_reg("rax").unwrap(), Some(gpr(RAX)));
        assert_eq!(clobber_reg("%ecx").unwrap(), Some(gpr(RCX)));
        assert_eq!(clobber_reg("r11").unwrap(), Some(gpr(11)));
        assert_eq!(clobber_reg("r8d").unwrap(), Some(gpr(8)));
        assert_eq!(clobber_reg("xmm3").unwrap(), Some(regs::xmm(3)));
        assert_eq!(clobber_reg("memory").unwrap(), None);
        assert_eq!(clobber_reg("cc").unwrap(), None);
        assert!(clobber_reg("rbp").is_err());
        assert!(clobber_reg("foo").is_err());
    }
}
