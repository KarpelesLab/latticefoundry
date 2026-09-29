//! The differential suite: IR programs run three ways and compared.
//!
//! 1. **The reference**: an interpreter over the *original* IR (before
//!    soft-float lowering and legalization) whose every value-producing step
//!    is [`crate::ir::eval`] — the executable semantics — with floating point
//!    computed exactly by `puremp`, plus memory, calls and control flow.
//! 2. **The MIR interpreter** ([`super::interp`]) over the Thumb isel's output
//!    for the prepared module, before register allocation.
//! 3. **The machine code**: the full pipeline (isel, allocation, frame layout,
//!    encoding), linked by [`super::sim::link`] and executed by the Thumb-2
//!    simulator ([`super::sim`]), which decodes the bytes itself.
//!
//! The soft-float and division helpers are Rust functions over IEEE bits
//! ([`super::sim::aeabi`]); the reference never calls them, so a wrong
//! lowering to a helper shows up as a mismatch. Every executor shares one
//! memory image (the compiled object's data, laid out once), so addresses and
//! initial data agree. A case whose reference result is poison or undefined
//! behavior is skipped (any result refines it).

use std::collections::HashMap;

use crate::ir::inst::InstKind;
use crate::ir::types::{FloatKind, Type, TypeId};
use crate::ir::value::{Const, FloatBits, ValueDef, ValueId};
use crate::ir::{EvalOutcome, FuncId, Module, SemValue};
use crate::support::StrInterner;

use puremp::Int;

use super::encode::ThumbOptions;
use super::sim::{Cpu, Image, Memory};
use super::tests::parse;

// ===========================================================================
// The reference interpreter
// ===========================================================================

struct Ref<'a> {
    m: &'a Module,
    syms: &'a StrInterner,
    mem: Memory,
    symbols: &'a HashMap<String, u32>,
    sp: u32,
    steps: u64,
}

fn value_bits(v: &SemValue) -> Option<u64> {
    match v {
        SemValue::Int { bits, .. } => bits.to_u64(),
        SemValue::Float(FloatBits::F16(b)) => Some(u64::from(*b)),
        SemValue::Float(FloatBits::F32(b)) => Some(u64::from(*b)),
        SemValue::Float(FloatBits::F64(b)) => Some(*b),
        SemValue::Ptr(a) => a.to_u64(),
        SemValue::Poison | SemValue::Vector(_) => None,
    }
}

impl Ref<'_> {
    fn size(&self, ty: TypeId) -> u32 {
        self.m.types().size_of(ty) as u32
    }

    fn value_of_bits(&self, ty: TypeId, raw: u64) -> SemValue {
        match self.m.types().get(ty) {
            Type::Int(w) => SemValue::int(*w, Int::from_u64(raw)),
            Type::Float(FloatKind::F16) => SemValue::Float(FloatBits::F16(raw as u16)),
            Type::Float(FloatKind::F32) => SemValue::Float(FloatBits::F32(raw as u32)),
            Type::Float(FloatKind::F64) => SemValue::Float(FloatBits::F64(raw)),
            _ => SemValue::ptr(Int::from_u64(raw & 0xffff_ffff)),
        }
    }

    fn load(&self, ty: TypeId, addr: u32) -> SemValue {
        let n = self.size(ty);
        let mut raw = 0u64;
        for k in 0..n.min(8) {
            raw |= u64::from(self.mem.read8(addr + k)) << (8 * k);
        }
        if let Type::Int(w) = self.m.types().get(ty)
            && *w > 64
        {
            let mut v = Int::ZERO;
            for k in (0..n).rev() {
                v = v.mul_2k(8).add(&Int::from_u64(u64::from(self.mem.read8(addr + k))));
            }
            return SemValue::int(*w, v);
        }
        self.value_of_bits(ty, raw)
    }

    fn store(&mut self, ty: TypeId, addr: u32, v: &SemValue) -> Result<(), String> {
        let n = self.size(ty);
        let bits = match v {
            SemValue::Int { bits, .. } => bits.clone(),
            SemValue::Poison => return Err("store of poison".into()),
            other => Int::from_u64(value_bits(other).expect("a value")),
        };
        for k in 0..n {
            let byte = bits.div_2k_trunc(8 * k).mod_2k(8).to_u64().unwrap_or(0) as u8;
            self.mem.write8(addr + k, byte);
        }
        Ok(())
    }

    fn addr(v: &SemValue) -> Result<u32, String> {
        match v {
            SemValue::Ptr(a) => Ok(a.to_u64().unwrap_or(0) as u32),
            SemValue::Int { bits, .. } => Ok(bits.to_u64().unwrap_or(0) as u32),
            _ => Err(format!("not an address: {v:?}")),
        }
    }

    fn operand(&self, f: &crate::ir::Function, env: &HashMap<ValueId, SemValue>, v: ValueId) -> Result<SemValue, String> {
        Ok(match &f.value(v).def {
            ValueDef::Const(c) => match self.m.consts().get(*c) {
                Const::Int { value, ty } => SemValue::int(self.m.types().bit_width(*ty).unwrap_or(64), value.clone()),
                Const::Float { bits, .. } => SemValue::Float(*bits),
                Const::Null(_) => SemValue::ptr(Int::ZERO),
                Const::Poison(_) => SemValue::Poison,
                other => return Err(format!("unsupported constant operand {other:?}")),
            },
            ValueDef::Global(g) => {
                let name = self.syms.resolve(self.m.global(*g).name);
                SemValue::ptr(Int::from_u64(u64::from(self.symbols[name])))
            }
            ValueDef::Func(fid) => {
                let name = self.syms.resolve(self.m.function(*fid).name);
                SemValue::ptr(Int::from_u64(u64::from(self.symbols[name])))
            }
            ValueDef::Inst(_) | ValueDef::Param(..) => {
                env.get(&v).cloned().ok_or_else(|| format!("use of an undefined value {v:?}"))?
            }
        })
    }

    fn call(&mut self, fid: FuncId, args: Vec<SemValue>) -> Result<Option<SemValue>, String> {
        let f = self.m.function(fid);
        let Some(entry) = f.entry() else {
            return Err(format!("call to external {}", self.syms.resolve(f.name)));
        };
        let saved_sp = self.sp;
        let mut env: HashMap<ValueId, SemValue> = HashMap::new();
        for (&p, a) in f.block(entry).params().iter().zip(args) {
            env.insert(p, a);
        }
        let mut block = entry;
        loop {
            let b = f.block(block);
            for &iid in b.insts() {
                self.steps += 1;
                if self.steps > 5_000_000 {
                    return Err("reference step budget exhausted".into());
                }
                let inst = f.inst(iid);
                let ops: Vec<SemValue> =
                    inst.operands().iter().map(|&o| self.operand(f, &env, o)).collect::<Result<_, _>>()?;
                let r = match &inst.kind {
                    InstKind::Alloca { elem_ty } => {
                        let size = self.size(*elem_ty).max(1);
                        let align = self.m.types().align_of(*elem_ty).max(1) as u32;
                        self.sp = (self.sp - size) & !(align - 1);
                        Some(SemValue::ptr(Int::from_u64(u64::from(self.sp))))
                    }
                    InstKind::Load { ty, .. } | InstKind::AtomicLoad { ty, .. } => {
                        Some(self.load(*ty, Self::addr(&ops[0])?))
                    }
                    InstKind::Store { ty, .. } | InstKind::AtomicStore { ty, .. } => {
                        self.store(*ty, Self::addr(&ops[0])?, &ops[1])?;
                        None
                    }
                    InstKind::Fence(_) => None,
                    InstKind::Call => {
                        let callee = inst.operands()[0];
                        let target = match f.value(callee).def {
                            ValueDef::Func(t) => t,
                            _ => {
                                let a = Self::addr(&ops[0])?;
                                let name = self
                                    .symbols
                                    .iter()
                                    .find(|&(_, &v)| v == a)
                                    .map(|(n, _)| n.clone())
                                    .ok_or("indirect call to an unknown address")?;
                                FuncId::from_index(
                                    self.m
                                        .functions()
                                        .position(|g| self.syms.resolve(g.name) == name)
                                        .ok_or("indirect call to a non-function")?,
                                )
                            }
                        };
                        self.call(target, ops[1..].to_vec())?
                    }
                    k if k.is_terminator() => unreachable!(),
                    kind => match crate::ir::eval(self.m.types(), inst.ty, kind, &inst.flags, &ops) {
                        EvalOutcome::Value(v) => Some(v),
                        EvalOutcome::UndefinedBehavior => return Err("undefined behavior".into()),
                    },
                };
                if let (Some(res), Some(v)) = (inst.result(), r) {
                    env.insert(res, v);
                }
            }
            let t = f.inst(b.terminator().ok_or("a block without terminator")?);
            let ops: Vec<SemValue> = t.operands().iter().map(|&o| self.operand(f, &env, o)).collect::<Result<_, _>>()?;
            let (next, args): (crate::ir::BlockId, Vec<SemValue>) = match &t.kind {
                InstKind::Ret => {
                    self.sp = saved_sp;
                    return Ok(ops.into_iter().next());
                }
                InstKind::Unreachable => return Err("reached unreachable".into()),
                InstKind::Br(target) => (*target, ops),
                InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                    let c = value_bits(&ops[0]).ok_or("branch on poison")?;
                    let (ta, fa) = (*true_args as usize, *false_args as usize);
                    if c & 1 != 0 {
                        (*if_true, ops[1..1 + ta].to_vec())
                    } else {
                        (*if_false, ops[1 + ta..1 + ta + fa].to_vec())
                    }
                }
                InstKind::Switch(data) => {
                    let SemValue::Int { width, bits } = &ops[0] else { return Err("switch on poison".into()) };
                    let mut off = 1 + data.default_args as usize;
                    let mut dest = (data.default, ops[1..off].to_vec());
                    for c in &data.cases {
                        let n = c.args as usize;
                        if c.value.mod_2k(*width) == *bits {
                            dest = (c.target, ops[off..off + n].to_vec());
                            break;
                        }
                        off += n;
                    }
                    dest
                }
                other => return Err(format!("unexpected terminator {other:?}")),
            };
            for (&p, a) in f.block(next).params().iter().zip(args) {
                env.insert(p, a);
            }
            block = next;
        }
    }
}

// ===========================================================================
// The harness
// ===========================================================================

/// The words `args` occupy in `r0`–`r3` for a function of signature `sig`
/// (an AAPCS subset: scalars, 64-bit values in even register pairs).
fn arg_words(m: &Module, sig: TypeId, args: &[u64]) -> Vec<u32> {
    let Type::Func(ft) = m.types().get(sig) else { unreachable!() };
    let mut words = Vec::new();
    for (&p, &a) in ft.params.iter().zip(args) {
        if m.types().size_of(p) == 8 {
            if words.len() % 2 == 1 {
                words.push(0);
            }
            words.push(a as u32);
            words.push((a >> 32) as u32);
        } else {
            words.push(a as u32);
        }
    }
    assert!(words.len() <= 4, "the harness passes arguments in registers only");
    words
}

/// The bits of a return value of type `ty` from `r0`–`r3`, masked to its width.
fn ret_bits(m: &Module, ty: TypeId, r: [u32; 4]) -> u64 {
    let full = u64::from(r[0]) | u64::from(r[1]) << 32;
    let bits = match m.types().get(ty) {
        Type::Int(w) => *w,
        Type::Float(FloatKind::F16) => 16,
        Type::Float(FloatKind::F32) => 32,
        Type::Float(FloatKind::F64) => 64,
        _ => 32,
    };
    if bits >= 64 { full } else { full & ((1u64 << bits) - 1) }
}

/// A compiled program ready to run three ways.
struct Harness {
    m: Module,
    syms: StrInterner,
    image: Image,
    pm: Module,
    ps: StrInterner,
    funcs: Vec<crate::codegen::MachineFunction>,
    func_names: Vec<String>,
    global_names: Vec<String>,
}

impl Harness {
    fn new(src: &str, topts: &ThumbOptions) -> Harness {
        let (m, syms) = parse(src);
        let compiled = super::compile_module_thumb(&m, &syms, &crate::codegen::CodegenOptions::default(), topts);
        let image = super::sim::link(&[&compiled.object]).unwrap_or_else(|e| panic!("link: {e}"));
        let (pm, ps) = super::prepare_module(&m, &syms, topts).expect("prepares");
        if let Err(d) = crate::verify::verify_module(&pm) {
            panic!("the prepared module does not verify: {d:?}\n{}", crate::ir::text::print_module(&pm, &ps));
        }
        let target = super::ThumbTarget::new()
            .with_hw_div(topts.hw_div)
            .with_helpers(super::isel::Helpers::resolve(&pm, &ps));
        let funcs = (0..pm.function_count())
            .map(|i| {
                let fid = FuncId::from_index(i);
                if pm.function(fid).is_declaration() {
                    crate::codegen::MachineFunction::new(ps.resolve(pm.function(fid).name), i as u32)
                } else {
                    target.select(&pm, fid, &ps)
                }
            })
            .collect();
        let func_names = pm.functions().map(|f| ps.resolve(f.name).to_owned()).collect();
        let global_names = pm.globals().map(|g| ps.resolve(g.name).to_owned()).collect();
        Harness { m, syms, image, pm, ps, funcs, func_names, global_names }
    }

    fn func(&self, name: &str) -> FuncId {
        FuncId::from_index(
            self.m.functions().position(|f| self.syms.resolve(f.name) == name).unwrap_or_else(|| panic!("no @{name}")),
        )
    }

    /// Run `name(args)` three ways; `None` when the reference is poison or UB.
    fn run(&self, name: &str, args: &[u64]) -> Option<(u64, u64, u64)> {
        let fid = self.func(name);
        let f = self.m.function(fid);
        let Type::Func(ft) = self.m.types().get(f.sig).clone() else { unreachable!() };
        let sem: Vec<SemValue> = ft
            .params
            .iter()
            .zip(args)
            .map(|(&t, &a)| match self.m.types().get(t) {
                Type::Int(w) => SemValue::int(*w, Int::from_u64(a)),
                Type::Float(FloatKind::F16) => SemValue::Float(FloatBits::F16(a as u16)),
                Type::Float(FloatKind::F32) => SemValue::Float(FloatBits::F32(a as u32)),
                Type::Float(FloatKind::F64) => SemValue::Float(FloatBits::F64(a)),
                _ => SemValue::ptr(Int::from_u64(a)),
            })
            .collect();
        let mut r = Ref {
            m: &self.m,
            syms: &self.syms,
            mem: self.image.mem.clone(),
            symbols: &self.image.symbols,
            sp: super::sim::STACK_TOP,
            steps: 0,
        };
        let want = match r.call(fid, sem) {
            Ok(Some(v)) => value_bits(&v)?,
            Ok(None) => 0,
            Err(e) if e.contains("undefined behavior") || e.contains("poison") => return None,
            Err(e) => panic!("reference {name}{args:x?}: {e}"),
        };
        let want = ret_bits(&self.m, ft.ret, [want as u32, (want >> 32) as u32, 0, 0]);
        let (mir, hw) = self.exec(name, args);
        Some((want, mir, hw))
    }

    /// Run `name(args)` on the MIR interpreter and on the simulator.
    fn exec(&self, name: &str, args: &[u64]) -> (u64, u64) {
        let f = self.m.function(self.func(name));
        let Type::Func(ft) = self.m.types().get(f.sig).clone() else { unreachable!() };
        let words = arg_words(&self.m, f.sig, args);

        let prog = super::interp::Program {
            funcs: &self.funcs,
            func_names: &self.func_names,
            global_names: &self.global_names,
            symbols: &self.image.symbols,
        };
        let pidx = self.pm.functions().position(|g| self.ps.resolve(g.name) == name).expect("prepared");
        let mir = super::interp::run(&prog, self.image.mem.clone(), pidx, &words)
            .unwrap_or_else(|e| panic!("MIR interpreter, {name}{args:x?}: {e}"));
        let mir = ret_bits(&self.m, ft.ret, mir);

        let mut cpu = Cpu::new(self.image.mem.clone(), self.image.helpers.clone());
        let entry = self.image.symbols[name];
        let hw = cpu.call(entry, &words).unwrap_or_else(|e| panic!("simulator, {name}{args:x?}: {e}"));
        let hw = ret_bits(&self.m, ft.ret, hw);
        (mir, hw)
    }

    /// Check `name` on every argument tuple of `cases`; returns how many
    /// were compared (not skipped).
    fn check(&self, name: &str, cases: &[Vec<u64>]) -> usize {
        let mut n = 0;
        for args in cases {
            if let Some((want, mir, hw)) = self.run(name, args) {
                assert_eq!(mir, want, "MIR interpreter: {name}({args:#x?})");
                assert_eq!(hw, want, "machine code: {name}({args:#x?})");
                n += 1;
            }
        }
        n
    }
}

/// Interesting operand values for a `bits`-bit integer.
fn samples(bits: u32) -> Vec<u64> {
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let mut v: Vec<u64> = vec![0, 1, 2, 3, 7, 0x7f, 0x80, 0xff, 0x100, 0x7fff, 0x8000, 0xffff, 0x1_0000, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0x1_0000_0000, 0x7fff_ffff_ffff_ffff, 0x8000_0000_0000_0000, u64::MAX, u64::MAX - 1];
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..4 {
        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        v.push(x);
    }
    let mut out: Vec<u64> = v.into_iter().map(|a| a & mask).collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn pairs(bits: u32) -> Vec<Vec<u64>> {
    let s = samples(bits);
    s.iter().flat_map(|&a| s.iter().map(move |&b| vec![a, b])).collect()
}

// ===========================================================================
// Integer arithmetic at every width
// ===========================================================================

const BINOPS: [&str; 13] = ["add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr", "udiv", "sdiv", "urem", "srem"];

fn binop_module(bits: u32) -> String {
    let t = format!("i{bits}");
    let mut src = String::from("module \"ops\"\n");
    for op in BINOPS {
        src += &format!(
            "func @{op}({t}, {t}) -> {t} {{\nentry ^0(%a: {t}, %b: {t}):\n  %r = {op} %a, %b : {t}\n  ret %r\n}}\n"
        );
    }
    for pred in ["eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"] {
        src += &format!(
            "func @{pred}({t}, {t}) -> i32 {{\nentry ^0(%a: {t}, %b: {t}):\n  %c = icmp {pred} %a, %b : i1\n  %r = zext %c : i32\n  ret %r\n}}\n"
        );
    }
    src
}

fn check_width(bits: u32, topts: &ThumbOptions) -> usize {
    let h = Harness::new(&binop_module(bits), topts);
    let all = pairs(bits);
    let shifts: Vec<Vec<u64>> = samples(bits)
        .into_iter()
        .flat_map(|a| [0u64, 1, 3, 7, 8, 15, 16, 31, 32, 33, 63].into_iter().map(move |s| vec![a, s]))
        .collect();
    let mut n = 0;
    for op in BINOPS {
        let cases = if op.contains("sh") { &shifts } else { &all };
        n += h.check(op, cases);
    }
    for pred in ["eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"] {
        n += h.check(pred, &all);
    }
    n
}

#[test]
fn integer_ops_at_every_width_match_the_reference() {
    let mut n = 0;
    for bits in [8, 16, 32, 64, 1, 5, 24] {
        n += check_width(bits, &ThumbOptions::default());
    }
    eprintln!("integer ops: {n} results compared three ways");
    assert!(n > 5000);
}

#[test]
fn division_through_the_aeabi_helpers_matches() {
    let topts = ThumbOptions::default().with_hw_div(false);
    let mut n = 0;
    for bits in [8, 32, 64] {
        n += check_width(bits, &topts);
    }
    assert!(n > 1000);
}

#[test]
fn casts_match_the_reference() {
    let widths = [1u32, 8, 16, 24, 32, 64];
    let mut src = String::from("module \"casts\"\n");
    for &from in &widths {
        for &to in &widths {
            let (f, t) = (format!("i{from}"), format!("i{to}"));
            let ops: &[&str] = match from.cmp(&to) {
                std::cmp::Ordering::Less => &["zext", "sext"],
                std::cmp::Ordering::Greater => &["trunc"],
                std::cmp::Ordering::Equal => &[],
            };
            for op in ops {
                // Each cast result is also compared at full width by
                // re-extending it: that reads the bits above a narrow result.
                let widen = if to < 64 { "  %w = zext %r : i64\n  ret %w\n" } else { "  ret %r\n" };
                src += &format!(
                    "func @{op}_{from}_{to}({f}, {f}) -> i64 {{\nentry ^0(%a: {f}, %b: {f}):\n  %s = add %a, %b : {f}\n  %r = {op} %s : {t}\n{widen}}}\n"
                );
            }
        }
    }
    let h = Harness::new(&src, &ThumbOptions::default());
    let mut n = 0;
    for &from in &widths {
        for &to in &widths {
            for op in ["zext", "sext", "trunc"] {
                let name = format!("{op}_{from}_{to}");
                if h.m.functions().any(|f| h.syms.resolve(f.name) == name) {
                    let s = samples(from);
                    let cases: Vec<Vec<u64>> = s.iter().map(|&a| vec![a, *s.last().unwrap()]).collect();
                    n += h.check(&name, &cases);
                }
            }
        }
    }
    assert!(n > 300, "{n}");
}

// ===========================================================================
// Programs: control flow, recursion, memory, calls
// ===========================================================================

const PROGRAMS: &str = "\
module \"programs\"
global @table : [8 x i32] = [8 x i32] (i32 3, i32 1, i32 4, i32 1, i32 5, i32 9, i32 2, i32 6)
global @wide : [3 x i64] = [3 x i64] (i64 1, i64 -2, i64 81985529216486895)
global @counter : i32 = i32 0
global @bytes : [4 x i8] = [4 x i8] (i8 200, i8 100, i8 -1, i8 7)
global @ptrs : [2 x ptr] = [2 x ptr] (ptr @table + 8, ptr @wide)
global @fns : [2 x ptr] = [2 x ptr] (ptr @double, ptr @square)

func @sum_to(i32) -> i32 {
entry ^0(%n: i32):
  br ^1(i32 0, i32 0)
^1(%i: i32, %acc: i32):
  %c = icmp slt %i, %n : i1
  cond_br %c, ^2, ^3
^2:
  %a = add %acc, %i : i32
  %j = add %i, i32 1 : i32
  br ^1(%j, %a)
^3:
  ret %acc
}

func @fact(i64) -> i64 {
entry ^0(%n: i64):
  %z = icmp ule %n, i64 1 : i1
  cond_br %z, ^1, ^2
^1:
  ret i64 1
^2:
  %m = sub %n, i64 1 : i64
  %r = call @fact(%m) : i64
  %p = mul %n, %r : i64
  ret %p
}

func @fib(i32) -> i32 {
entry ^0(%n: i32):
  %c = icmp slt %n, i32 2 : i1
  cond_br %c, ^1, ^2
^1:
  ret %n
^2:
  %a = sub %n, i32 1 : i32
  %b = sub %n, i32 2 : i32
  %x = call @fib(%a) : i32
  %y = call @fib(%b) : i32
  %s = add %x, %y : i32
  ret %s
}

func @classify(i32) -> i32 {
entry ^0(%x: i32):
  switch %x, ^9 [0: ^1, 1: ^2, -5: ^3, 1000000: ^4, 255: ^5]
^1:
  ret i32 10
^2:
  ret i32 20
^3:
  ret i32 30
^4:
  ret i32 40
^5:
  ret i32 50
^9:
  ret i32 99
}

func @classify64(i64) -> i32 {
entry ^0(%x: i64):
  switch %x, ^9 [0: ^1, 4294967296: ^2, -1: ^3]
^1:
  ret i32 1
^2:
  ret i32 2
^3:
  ret i32 3
^9:
  ret i32 0
}

func @table_sum() -> i32 {
entry ^0:
  br ^1(i32 0, i32 0)
^1(%i: i32, %acc: i32):
  %c = icmp ult %i, i32 8 : i1
  cond_br %c, ^2, ^3
^2:
  %off = mul %i, i32 4 : i32
  %p = ptr_add @table, %off : ptr
  %v = load %p align 4 : i32
  %a = add %acc, %v : i32
  %j = add %i, i32 1 : i32
  br ^1(%j, %a)
^3:
  ret %acc
}

func @wide_mem(i32) -> i64 {
entry ^0(%k: i32):
  %off = mul %k, i32 8 : i32
  %p = ptr_add @wide, %off : ptr
  %v = load %p align 8 : i64
  %w = add %v, i64 4294967297 : i64
  store %w, %p align 8 : i64
  %again = load %p align 8 : i64
  ret %again
}

func @bytes_sum() -> i32 {
entry ^0:
  %p1 = ptr_add @bytes, i32 1 : ptr
  %p2 = ptr_add @bytes, i32 2 : ptr
  %a = load @bytes align 1 : i8
  %b = load %p1 align 1 : i8
  %c = load %p2 align 1 : i8
  %s = add %a, %b : i8
  %t = add %s, %c : i8
  %x = sext %t : i32
  %y = zext %a : i32
  %r = add %x, %y : i32
  ret %r
}

func @via_ptrs() -> i64 {
entry ^0:
  %pp = load @ptrs align 4 : ptr
  %v = load %pp align 4 : i32
  %q = ptr_add @ptrs, i32 4 : ptr
  %pw = load %q align 4 : ptr
  %w = load %pw align 8 : i64
  %v64 = sext %v : i64
  %r = add %w, %v64 : i64
  ret %r
}

func @double(i32) -> i32 {
entry ^0(%x: i32):
  %r = add %x, %x : i32
  ret %r
}

func @square(i32) -> i32 {
entry ^0(%x: i32):
  %r = mul %x, %x : i32
  ret %r
}

func @indirect(i32, i32) -> i32 {
entry ^0(%which: i32, %x: i32):
  %off = mul %which, i32 4 : i32
  %slot = ptr_add @fns, %off : ptr
  %f = load %slot align 4 : ptr
  %r = call %f(%x) : i32
  %g = call @double(%r) : i32
  ret %g
}

func @fnptr_value(i32) -> i32 {
entry ^0(%x: i32):
  %c = icmp eq %x, i32 0 : i1
  %f = select %c, @double, @square : ptr
  %r = call %f(i32 7) : i32
  ret %r
}

func @bump(i32) -> i32 {
entry ^0(%d: i32):
  %v = load @counter align 4 : i32
  %w = add %v, %d : i32
  store %w, @counter align 4 : i32
  %x = load volatile @counter align 4 : i32
  ret %x
}

func @locals(i32) -> i32 {
entry ^0(%x: i32):
  %a = alloca [4 x i32] : ptr
  %p1 = ptr_add %a, i32 4 : ptr
  %p3 = ptr_add %a, i32 12 : ptr
  store %x, %a align 4 : i32
  store i32 100, %p1 align 4 : i32
  %y = mul %x, i32 3 : i32
  store %y, %p3 align 4 : i32
  %b = load %a align 4 : i32
  %c = load %p1 align 4 : i32
  %d = load %p3 align 4 : i32
  %s = add %b, %c : i32
  %t = add %s, %d : i32
  ret %t
}

func @many(i32, i32, i32, i32, i32, i32, i64, i32) -> i64 {
entry ^0(%a: i32, %b: i32, %c: i32, %d: i32, %e: i32, %f: i32, %g: i64, %h: i32):
  %ab = sub %a, %b : i32
  %cd = mul %c, %d : i32
  %ef = xor %e, %f : i32
  %s1 = add %ab, %cd : i32
  %s2 = add %s1, %ef : i32
  %s3 = add %s2, %h : i32
  %w = sext %s3 : i64
  %r = add %w, %g : i64
  ret %r
}

func @pairs(i32, i64, i32, i64) -> i64 {
entry ^0(%a: i32, %b: i64, %c: i32, %d: i64):
  %aw = zext %a : i64
  %cw = zext %c : i64
  %x = mul %b, %aw : i64
  %y = sub %d, %cw : i64
  %r = xor %x, %y : i64
  ret %r
}

func @call_many(i32) -> i64 {
entry ^0(%x: i32):
  %r = call @many(%x, i32 2, i32 3, i32 4, i32 5, i32 6, i64 -4294967296, i32 8) : i64
  %s = call @pairs(i32 3, i64 81985529216486895, %x, i64 -7) : i64
  %t = xor %r, %s : i64
  ret %t
}

func @wide_shifts(i64, i32) -> i64 {
entry ^0(%v: i64, %s: i32):
  %sw = zext %s : i64
  %a = shl %v, %sw : i64
  %b = lshr %v, %sw : i64
  %c = ashr %v, %sw : i64
  %d = xor %a, %b : i64
  %e = add %d, %c : i64
  ret %e
}

func @i128_math(i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64):
  %x = sext %a : i128
  %y = zext %b : i128
  %p = add %x, %y : i128
  %q = shl %p, i128 7 : i128
  %s = add %q, i128 340282366920938463463374607431768211455 : i128
  %h = lshr %s, i128 64 : i128
  %l = trunc %s : i64
  %ht = trunc %h : i64
  %r = xor %l, %ht : i64
  ret %r
}
";

// ===========================================================================
// By-value composites
// ===========================================================================

const COMPOSITES: &str = "\
module \"composites\"
func @small_sum({ i16, i8 }) -> i32 {
entry ^0(%s: { i16, i8 }):
  %p = ptr_add %s, i32 2 : ptr
  %a = load %s align 2 : i16
  %b = load %p align 1 : i8
  %aw = zext %a : i32
  %bw = zext %b : i32
  %r = add %aw, %bw : i32
  ret %r
}

func @big_sum({ i32, i64, i32, i32 }) -> i64 {
entry ^0(%s: { i32, i64, i32, i32 }):
  %p1 = ptr_add %s, i32 8 : ptr
  %p2 = ptr_add %s, i32 16 : ptr
  %p3 = ptr_add %s, i32 20 : ptr
  %a = load %s align 4 : i32
  %b = load %p1 align 8 : i64
  %c = load %p2 align 4 : i32
  %d = load %p3 align 4 : i32
  %aw = zext %a : i64
  %cw = sext %c : i64
  %dw = zext %d : i64
  %s1 = add %aw, %b : i64
  %s2 = sub %s1, %cw : i64
  %s3 = add %s2, %dw : i64
  ret %s3
}

func @split(i32, i32, { i32, i32, i32 }) -> i32 {
entry ^0(%x: i32, %y: i32, %s: { i32, i32, i32 }):
  %p1 = ptr_add %s, i32 4 : ptr
  %p2 = ptr_add %s, i32 8 : ptr
  %a = load %s align 4 : i32
  %b = load %p1 align 4 : i32
  %c = load %p2 align 4 : i32
  %t = mul %c, i32 1000 : i32
  %u = mul %b, i32 100 : i32
  %s1 = add %a, %u : i32
  %s2 = add %s1, %t : i32
  %s3 = add %s2, %x : i32
  %s4 = sub %s3, %y : i32
  ret %s4
}

func @make_small(i32) -> { i8, i8, i16 } {
entry ^0(%x: i32):
  %s = alloca { i8, i8, i16 } : ptr
  %p1 = ptr_add %s, i32 1 : ptr
  %p2 = ptr_add %s, i32 2 : ptr
  %a = trunc %x : i8
  %b = add %a, i8 1 : i8
  %c = trunc %x : i16
  store %a, %s align 1 : i8
  store %b, %p1 align 1 : i8
  store %c, %p2 align 2 : i16
  ret %s
}

func @make_big(i32, i64) -> { i64, i32, i32 } {
entry ^0(%x: i32, %y: i64):
  %s = alloca { i64, i32, i32 } : ptr
  %p1 = ptr_add %s, i32 8 : ptr
  %p2 = ptr_add %s, i32 12 : ptr
  store %y, %s align 8 : i64
  store %x, %p1 align 4 : i32
  %z = mul %x, i32 -3 : i32
  store %z, %p2 align 4 : i32
  ret %s
}

func @d_small(i32) -> i32 {
entry ^0(%x: i32):
  %sm = alloca { i16, i8 } : ptr
  %sm1 = ptr_add %sm, i32 2 : ptr
  %t = trunc %x : i16
  store %t, %sm align 2 : i16
  store i8 9, %sm1 align 1 : i8
  %r1 = call @small_sum(%sm) : i32
  ret %r1
}

func @d_big(i32) -> i64 {
entry ^0(%x: i32):
  %bg = alloca { i32, i64, i32, i32 } : ptr
  %b1 = ptr_add %bg, i32 8 : ptr
  %b2 = ptr_add %bg, i32 16 : ptr
  %b3 = ptr_add %bg, i32 20 : ptr
  store %x, %bg align 4 : i32
  store i64 1311768467463790320, %b1 align 8 : i64
  store i32 -17, %b2 align 4 : i32
  store i32 23, %b3 align 4 : i32
  %r2 = call @big_sum(%bg) : i64
  ret %r2
}

func @d_split(i32) -> i32 {
entry ^0(%x: i32):
  %tr = alloca { i32, i32, i32 } : ptr
  %t1 = ptr_add %tr, i32 4 : ptr
  %t2 = ptr_add %tr, i32 8 : ptr
  store i32 1, %tr align 4 : i32
  store i32 2, %t1 align 4 : i32
  store %x, %t2 align 4 : i32
  %r3 = call @split(i32 5, i32 6, %tr) : i32
  ret %r3
}

func @d_make_small(i32) -> i32 {
entry ^0(%x: i32):
  %ms = call @make_small(%x) : { i8, i8, i16 }
  %v = load %ms align 4 : i32
  ret %v
}

func @d_make_big(i32) -> i64 {
entry ^0(%x: i32):
  %mbig = call @make_big(%x, i64 -81985529216486895) : { i64, i32, i32 }
  %n1 = ptr_add %mbig, i32 8 : ptr
  %n2 = ptr_add %mbig, i32 12 : ptr
  %na = load %mbig align 8 : i64
  %nb = load %n1 align 4 : i32
  %nc = load %n2 align 4 : i32
  %wb = zext %nb : i64
  %wc = zext %nc : i64
  %sh = shl %wc, i64 32 : i64
  %o = or %wb, %sh : i64
  %r = xor %o, %na : i64
  ret %r
}

func @drive(i32) -> i64 {
entry ^0(%x: i32):
  %sm = alloca { i16, i8 } : ptr
  %sm1 = ptr_add %sm, i32 2 : ptr
  store i16 -3, %sm align 2 : i16
  store i8 9, %sm1 align 1 : i8
  %r1 = call @small_sum(%sm) : i32
  %bg = alloca { i32, i64, i32, i32 } : ptr
  %b1 = ptr_add %bg, i32 8 : ptr
  %b2 = ptr_add %bg, i32 16 : ptr
  %b3 = ptr_add %bg, i32 20 : ptr
  store %x, %bg align 4 : i32
  store i64 1311768467463790320, %b1 align 8 : i64
  store i32 -17, %b2 align 4 : i32
  store i32 23, %b3 align 4 : i32
  %r2 = call @big_sum(%bg) : i64
  %tr = alloca { i32, i32, i32 } : ptr
  %t1 = ptr_add %tr, i32 4 : ptr
  %t2 = ptr_add %tr, i32 8 : ptr
  store i32 1, %tr align 4 : i32
  store i32 2, %t1 align 4 : i32
  store %x, %t2 align 4 : i32
  %r3 = call @split(i32 5, i32 6, %tr) : i32
  %ms = call @make_small(%x) : { i8, i8, i16 }
  %m1 = ptr_add %ms, i32 1 : ptr
  %m2 = ptr_add %ms, i32 2 : ptr
  %ma = load %ms align 1 : i8
  %mb = load %m1 align 1 : i8
  %mc = load %m2 align 2 : i16
  %mbig = call @make_big(%x, i64 -81985529216486895) : { i64, i32, i32 }
  %n1 = ptr_add %mbig, i32 8 : ptr
  %n2 = ptr_add %mbig, i32 12 : ptr
  %na = load %mbig align 8 : i64
  %nb = load %n1 align 4 : i32
  %nc = load %n2 align 4 : i32
  %w1 = zext %r1 : i64
  %w3 = sext %r3 : i64
  %wa = zext %ma : i64
  %wb = zext %mb : i64
  %wc = sext %mc : i64
  %wnb = zext %nb : i64
  %wnc = sext %nc : i64
  %s1 = add %w1, %r2 : i64
  %s2 = add %s1, %w3 : i64
  %s3 = add %s2, %wa : i64
  %s4 = mul %s3, i64 3 : i64
  %s5 = add %s4, %wb : i64
  %s6 = add %s5, %wc : i64
  %s7 = xor %s6, %na : i64
  %s8 = add %s7, %wnb : i64
  %s9 = add %s8, %wnc : i64
  ret %s9
}
";

#[test]
fn programs_match_the_reference() {
    let h = Harness::new(PROGRAMS, &ThumbOptions::default());
    let mut n = 0;
    n += h.check("sum_to", &[vec![0], vec![1], vec![10], vec![1000], vec![0xffff_fff0]]);
    n += h.check("fact", &[vec![0], vec![1], vec![5], vec![20], vec![25]]);
    n += h.check("fib", &[vec![0], vec![1], vec![10], vec![20]]);
    n += h.check("classify", &[vec![0], vec![1], vec![2], vec![0xffff_fffb], vec![1_000_000], vec![255], vec![256]]);
    n += h.check("classify64", &[vec![0], vec![1 << 32], vec![u64::MAX], vec![1], vec![0xffff_ffff]]);
    n += h.check("table_sum", &[vec![]]);
    n += h.check("wide_mem", &[vec![0], vec![1], vec![2]]);
    n += h.check("bytes_sum", &[vec![]]);
    n += h.check("via_ptrs", &[vec![]]);
    n += h.check("indirect", &[vec![0, 5], vec![1, 5], vec![1, 0xffff]]);
    n += h.check("fnptr_value", &[vec![0], vec![1]]);
    n += h.check("bump", &[vec![3], vec![0xffff_ffff]]);
    n += h.check("locals", &[vec![7], vec![0x8000_0000]]);
    n += h.check("call_many", &[vec![1], vec![0x7fff_ffff], vec![0x8000_0001]]);
    let shift_cases: Vec<Vec<u64>> = samples(64)
        .into_iter()
        .flat_map(|v| [0u64, 1, 31, 32, 33, 63].into_iter().map(move |s| vec![v, s]))
        .collect();
    n += h.check("wide_shifts", &shift_cases);
    n += h.check("i128_math", &pairs(64)[..200]);
    eprintln!("programs: {n} results compared three ways");
    assert!(n > 250, "{n}");
}

#[test]
fn by_value_composites_follow_the_aapcs() {
    let h = Harness::new(COMPOSITES, &ThumbOptions::default());
    let xs = [vec![0], vec![1], vec![0x1234_5678], vec![0xffff_ffff]];
    let mut n = 0;
    for f in ["d_small", "d_big", "d_split", "d_make_small", "d_make_big", "drive"] {
        n += h.check(f, &xs);
    }
    assert_eq!(n, 24);
}

// ===========================================================================
// Soft float
// ===========================================================================

const FLOATS: &str = "\
module \"floats\"
global @scale : f64 = f64 0x4004000000000000
global @fs : [2 x f32] = [2 x f32] (f32 0x3fc00000, f32 0xc0200000)

func @fadd32(f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32):
  %r = fadd %a, %b : f32
  ret %r
}
func @fsub32(f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32):
  %r = fsub %a, %b : f32
  ret %r
}
func @fmul32(f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32):
  %r = fmul %a, %b : f32
  ret %r
}
func @fdiv32(f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32):
  %r = fdiv %a, %b : f32
  ret %r
}
func @frem32(f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32):
  %r = frem %a, %b : f32
  ret %r
}
func @fadd64(f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64):
  %r = fadd %a, %b : f64
  ret %r
}
func @fsub64(f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64):
  %r = fsub %a, %b : f64
  ret %r
}
func @fmul64(f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64):
  %r = fmul %a, %b : f64
  ret %r
}
func @fdiv64(f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64):
  %r = fdiv %a, %b : f64
  ret %r
}
func @frem64(f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64):
  %r = frem %a, %b : f64
  ret %r
}
func @fneg64(f64) -> f64 {
entry ^0(%a: f64):
  %r = fneg %a : f64
  ret %r
}
";

/// f32 / f64 test values: signed zeros, subnormals, normals, extremes,
/// infinities; NaNs are exercised through the comparisons.
fn f32s() -> Vec<u64> {
    [0.0f32, -0.0, 1.0, -1.5, 0.1, 3.0e38, -3.0e38, 1.0e-40, 123_456.79, -7.25, 16_777_217.0, f32::INFINITY, f32::NEG_INFINITY]
        .iter()
        .map(|x| u64::from(x.to_bits()))
        .collect()
}

fn f64s() -> Vec<u64> {
    [0.0f64, -0.0, 1.0, -1.5, 0.1, 1.0e300, -1.0e300, 5.0e-324, 123_456.789, -7.25, 9_007_199_254_740_993.0, f64::INFINITY, f64::NEG_INFINITY]
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn fpairs(v: &[u64]) -> Vec<Vec<u64>> {
    v.iter().flat_map(|&a| v.iter().map(move |&b| vec![a, b])).collect()
}

/// Whether a result is a NaN of the given width (payloads may differ between
/// the reference and the host's arithmetic; the suite compares NaN-ness).
fn is_nan(bits: u64, w: u32) -> bool {
    match w {
        32 => f32::from_bits(bits as u32).is_nan(),
        _ => f64::from_bits(bits).is_nan(),
    }
}

#[test]
fn soft_float_arithmetic_matches_the_reference() {
    let h = Harness::new(FLOATS, &ThumbOptions::default());
    let mut n = 0;
    for (name, w, vals) in [
        ("fadd32", 32, f32s()),
        ("fsub32", 32, f32s()),
        ("fmul32", 32, f32s()),
        ("fdiv32", 32, f32s()),
        ("frem32", 32, f32s()),
        ("fadd64", 64, f64s()),
        ("fsub64", 64, f64s()),
        ("fmul64", 64, f64s()),
        ("fdiv64", 64, f64s()),
        ("frem64", 64, f64s()),
    ] {
        for args in fpairs(&vals) {
            let Some((want, mir, hw)) = h.run(name, &args) else { continue };
            if is_nan(want, w) {
                assert!(is_nan(mir, w) && is_nan(hw, w), "{name}({args:x?}) should be NaN");
            } else {
                assert_eq!(mir, want, "MIR interpreter: {name}({args:x?})");
                assert_eq!(hw, want, "machine code: {name}({args:x?})");
            }
            n += 1;
        }
    }
    n += h.check("fneg64", &f64s().into_iter().map(|a| vec![a]).collect::<Vec<_>>());
    eprintln!("soft-float arithmetic: {n} results compared three ways");
    assert!(n > 1500);
}

#[test]
fn soft_float_comparisons_match_the_reference() {
    let preds = ["false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno", "true"];
    let mut src = String::from("module \"fcmp\"\n");
    for p in preds {
        for (t, w) in [("f32", 32), ("f64", 64)] {
            src += &format!(
                "func @{p}{w}({t}, {t}) -> i32 {{\nentry ^0(%a: {t}, %b: {t}):\n  %c = fcmp {p} %a, %b : i1\n  %r = zext %c : i32\n  ret %r\n}}\n"
            );
        }
    }
    let h = Harness::new(&src, &ThumbOptions::default());
    let mut v32 = f32s();
    v32.push(u64::from(f32::NAN.to_bits()));
    let mut v64 = f64s();
    v64.push(f64::NAN.to_bits());
    let mut n = 0;
    for p in preds {
        n += h.check(&format!("{p}32"), &fpairs(&v32));
        n += h.check(&format!("{p}64"), &fpairs(&v64));
    }
    eprintln!("soft-float comparisons: {n} results compared three ways");
    assert!(n > 5000);
}

#[test]
fn soft_float_conversions_match_the_reference() {
    let mut src = String::from("module \"conv\"\n");
    for (ft, fw) in [("f32", 32), ("f64", 64)] {
        for iw in [8u32, 16, 32, 64] {
            for op in ["fptosi", "fptoui"] {
                src += &format!(
                    "func @{op}_{fw}_{iw}({ft}) -> i{iw} {{\nentry ^0(%a: {ft}):\n  %r = {op} %a : i{iw}\n  ret %r\n}}\n"
                );
            }
            for op in ["sitofp", "uitofp"] {
                src += &format!(
                    "func @{op}_{iw}_{fw}(i{iw}) -> {ft} {{\nentry ^0(%a: i{iw}):\n  %r = {op} %a : {ft}\n  ret %r\n}}\n"
                );
            }
        }
    }
    src += "func @ext(f32) -> f64 {\nentry ^0(%a: f32):\n  %r = fpext %a : f64\n  ret %r\n}\n";
    src += "func @trunc(f64) -> f32 {\nentry ^0(%a: f64):\n  %r = fptrunc %a : f32\n  ret %r\n}\n";
    src += "func @half(f32) -> f32 {\nentry ^0(%a: f32):\n  %h = fptrunc %a : f16\n  %s = fadd %h, %h : f16\n  %r = fpext %s : f32\n  ret %r\n}\n";
    src += "func @half64(f64, i32) -> f64 {\nentry ^0(%a: f64, %i: i32):\n  %h = fptrunc %a : f16\n  %k = sitofp %i : f16\n  %m = fmul %h, %k : f16\n  %c = fcmp olt %h, %k : i1\n  %s = select %c, %m, %h : f16\n  %r = fpext %s : f64\n  ret %r\n}\n";
    src += "func @bits(f64) -> i64 {\nentry ^0(%a: f64):\n  %b = bitcast %a : i64\n  %c = xor %b, i64 1 : i64\n  %d = bitcast %c : f64\n  %e = fadd %d, %a : f64\n  %r = bitcast %e : i64\n  ret %r\n}\n";
    let h = Harness::new(&src, &ThumbOptions::default());
    let mut n = 0;
    let ffloats: Vec<u64> = [0.0f32, -0.0, 1.0, -1.0, 1.5, -2.5, 127.9, -128.0, 200.0, 255.0, 65535.0, 3.0e9, -2.0e9, 1.0e18, 1.8e19, 0.4]
        .iter()
        .map(|x| u64::from(x.to_bits()))
        .collect();
    let dfloats: Vec<u64> = [0.0f64, -0.0, 1.0, -1.0, 1.5, -2.5, 127.9, -128.0, 200.0, 255.0, 65535.0, 3.0e9, -2.0e9, 1.0e18, 1.8e19, 0.4, 4.0e20]
        .iter()
        .map(|x| x.to_bits())
        .collect();
    for (fw, vals) in [(32, &ffloats), (64, &dfloats)] {
        for iw in [8u32, 16, 32, 64] {
            for op in ["fptosi", "fptoui"] {
                let cases: Vec<Vec<u64>> = vals.iter().map(|&v| vec![v]).collect();
                n += h.check(&format!("{op}_{fw}_{iw}"), &cases);
            }
            for op in ["sitofp", "uitofp"] {
                let cases: Vec<Vec<u64>> = samples(iw).into_iter().map(|v| vec![v]).collect();
                n += h.check(&format!("{op}_{iw}_{fw}"), &cases);
            }
        }
    }
    n += h.check("ext", &f32s().into_iter().map(|v| vec![v]).collect::<Vec<_>>());
    n += h.check("trunc", &f64s().into_iter().map(|v| vec![v]).collect::<Vec<_>>());
    n += h.check("half", &f32s().into_iter().map(|v| vec![v]).collect::<Vec<_>>());
    let hcases: Vec<Vec<u64>> = f64s().into_iter().flat_map(|v| [0u64, 3, 0xffff_fffe, 70000].map(|i| vec![v, i])).collect();
    n += h.check("half64", &hcases);
    n += h.check("bits", &f64s().into_iter().map(|v| vec![v]).collect::<Vec<_>>());
    eprintln!("soft-float conversions: {n} results compared three ways");
    assert!(n > 500, "{n}");
}

#[test]
fn soft_float_program_with_float_globals_and_calls() {
    let src = "\
module \"fprog\"
global @coef : [4 x f64] = [4 x f64] (f64 0x3ff0000000000000, f64 0xbfe0000000000000, f64 0x3fc5555555555555, f64 0x4000000000000000)
global @acc : f32 = f32 0x00000000

func @poly(f64) -> f64 {
entry ^0(%x: f64):
  br ^1(i32 3, f64 0x0000000000000000)
^1(%i: i32, %r: f64):
  %off = mul %i, i32 8 : i32
  %p = ptr_add @coef, %off : ptr
  %c = load %p align 8 : f64
  %m = fmul %r, %x : f64
  %s = fadd %m, %c : f64
  %z = icmp eq %i, i32 0 : i1
  %j = sub %i, i32 1 : i32
  cond_br %z, ^2, ^1(%j, %s)
^2:
  ret %s
}

func @mix(i32, f64, f32, i64, f64) -> f64 {
entry ^0(%a: i32, %b: f64, %c: f32, %d: i64, %e: f64):
  %af = sitofp %a : f64
  %cf = fpext %c : f64
  %df = sitofp %d : f64
  %s1 = fadd %af, %b : f64
  %s2 = fmul %s1, %cf : f64
  %s3 = fsub %s2, %df : f64
  %s4 = fdiv %s3, %e : f64
  ret %s4
}

func @main(i32) -> i64 {
entry ^0(%n: i32):
  %x = sitofp %n : f64
  %p = call @poly(%x) : f64
  %q = call @mix(%n, %p, f32 0x40490fdb, i64 -123456789, f64 0x3ff8000000000000) : f64
  %old = load @acc align 4 : f32
  %qf = fptrunc %q : f32
  %new = fadd %old, %qf : f32
  store %new, @acc align 4 : f32
  %big = fcmp ogt %q, %p : i1
  %r = fptosi %q : i64
  %b = zext %big : i64
  %sh = shl %b, i64 62 : i64
  %t = xor %r, %sh : i64
  ret %t
}
";
    let h = Harness::new(src, &ThumbOptions::default());
    let n = h.check("main", &[vec![0], vec![1], vec![7], vec![0xffff_fffd], vec![1000]]);
    assert_eq!(n, 5);
}

// ===========================================================================
// Narrow values with dirty upper register bits
// ===========================================================================

/// The narrow-value probes of the other targets (`git log "narrow values"`),
/// run three ways: every function returns 0 when its narrow value is handled
/// right, whatever the register holds above the value's width.
#[test]
fn narrow_values_ignore_dirty_upper_bits() {
    let src = "\
module \"narrow\"
func @ult8(i8, i8) -> i32 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %c = icmp ult %s, %a : i1
  %r = zext %c : i32
  ret %r
}
func @lshr8(i8, i8) -> i8 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = lshr %s, i8 1 : i8
  ret %r
}
func @ashr8(i8, i8) -> i8 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = ashr %s, i8 1 : i8
  ret %r
}
func @udiv8(i8, i8) -> i8 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %r = udiv %s, i8 3 : i8
  ret %r
}
func @srem16(i16, i16) -> i16 {
entry ^0(%a: i16, %b: i16):
  %s = add %a, %b : i16
  %r = srem %s, i16 7 : i16
  ret %r
}
func @switch8(i8, i8) -> i32 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  switch %s, ^1 [44: ^2, -56: ^3]
^1:
  ret i32 1
^2:
  ret i32 2
^3:
  ret i32 3
}
func @condbr(i32) -> i32 {
entry ^0(%a: i32):
  %t = trunc %a : i1
  cond_br %t, ^1, ^2
^1:
  ret i32 1
^2:
  ret i32 0
}
func @select1(i32) -> i32 {
entry ^0(%a: i32):
  %t = trunc %a : i1
  %r = select %t, i32 10, i32 20 : i32
  ret %r
}
func @zext1(i32) -> i64 {
entry ^0(%a: i32):
  %t = trunc %a : i1
  %z = zext %t : i64
  ret %z
}
func @cmp24(i32) -> i32 {
entry ^0(%a: i32):
  %t = trunc %a : i24
  %c = icmp slt %t, i24 5 : i1
  %r = zext %c : i32
  ret %r
}
func @shamt(i32, i8) -> i32 {
entry ^0(%a: i32, %s: i8):
  %m = and %s, i8 7 : i8
  %t = trunc %m : i3
  %w = zext %t : i32
  %r = shl %a, %w : i32
  ret %r
}
func @ptroff(i32, i8) -> i32 {
entry ^0(%x: i32, %o: i8):
  %buf = alloca [16 x i32] : ptr
  %mid = ptr_add %buf, i32 32 : ptr
  %e = ptr_add %mid, %o : ptr
  store %x, %e align 1 : i32
  %v = load %e align 1 : i32
  ret %v
}
func @uitofp8(i8, i8) -> i32 {
entry ^0(%a: i8, %b: i8):
  %s = add %a, %b : i8
  %f = uitofp %s : f32
  %i = fptosi %f : i32
  ret %i
}
func @sitofp16(i16, i16) -> i32 {
entry ^0(%a: i16, %b: i16):
  %s = add %a, %b : i16
  %f = sitofp %s : f64
  %i = fptosi %f : i32
  ret %i
}
func @arg_i1(i32) -> i32 {
entry ^0(%a: i32):
  %t = trunc %a : i1
  %r = call @takes_bool(%t) : i32
  ret %r
}
func @takes_bool(i1) -> i32 {
entry ^0(%b: i1):
  %w = zext %b : i32
  ret %w
}
";
    let h = Harness::new(src, &ThumbOptions::default());
    let mut n = 0;
    let dirty8: Vec<Vec<u64>> = vec![vec![200, 100], vec![100, 100], vec![0xff, 1], vec![0x80, 0x80], vec![0, 0]];
    for f in ["ult8", "lshr8", "ashr8", "udiv8", "switch8", "uitofp8"] {
        n += h.check(f, &dirty8);
    }
    let dirty16: Vec<Vec<u64>> = vec![vec![0xffff, 0xffff], vec![0x8000, 0x8000], vec![30000, 30000], vec![1, 2]];
    for f in ["srem16", "sitofp16"] {
        n += h.check(f, &dirty16);
    }
    let words: Vec<Vec<u64>> = [0u64, 1, 2, 3, 0xffff_fffe, 0x0100_0004, 0x00ff_ffff, 0x0080_0000].iter().map(|&x| vec![x]).collect();
    for f in ["condbr", "select1", "zext1", "cmp24", "arg_i1"] {
        n += h.check(f, &words);
    }
    n += h.check("shamt", &[vec![1, 0xff], vec![3, 0xf9], vec![0x8000_0001, 0x21]]);
    n += h.check("ptroff", &[vec![0x1234_5678, 0xfc], vec![7, 4], vec![9, 0xe0]]);
    assert!(n >= 60, "{n}");
}

// ===========================================================================
// Volatile, atomics, and the machine-code shape
// ===========================================================================

#[test]
fn volatile_and_atomic_accesses() {
    let src = "\
module \"mmio\"
global @reg64 : i64 = i64 0
global @flag : i32 = i32 0
func @poke(i64) -> i64 {
entry ^0(%v: i64):
  store volatile %v, @reg64 align 8 : i64
  %r = load volatile @reg64 align 8 : i64
  %s = add %r, i64 1 : i64
  ret %s
}
func @publish(i32) -> i32 {
entry ^0(%v: i32):
  atomic_store release %v, @flag align 4 : i32
  fence seq_cst
  %r = atomic_load acquire @flag align 4 : i32
  ret %r
}
";
    let h = Harness::new(src, &ThumbOptions::default());
    assert_eq!(h.check("poke", &[vec![0x1234_5678_9abc_def0], vec![u64::MAX]]), 2);
    assert_eq!(h.check("publish", &[vec![42]]), 1);
    // One `ldrd`/`strd` each, and `dmb` barriers.
    let (m, syms) = parse(src);
    let obj = super::compile_module(&m, &syms);
    let elf = crate::mc::elf::write_with(&obj, &crate::mc::elf::ElfTarget::ARM).unwrap();
    if let Some(text) = super::tests::objdump(&elf, "mmio") {
        assert!(text.contains("strd") && text.contains("ldrd") && text.contains("dmb"), "{text}");
    }
}

/// Every function of the suites, disassembled by `llvm-objdump`, decodes
/// without an unknown instruction, and the disassembler agrees with the
/// simulator's decoder on the instruction boundaries (a 16/32-bit mix-up
/// would shift them).
#[test]
fn llvm_objdump_decodes_every_compiled_function() {
    for src in [PROGRAMS, COMPOSITES] {
        let (m, syms) = parse(src);
        let obj = super::compile_module(&m, &syms);
        let elf = crate::mc::elf::write_with(&obj, &crate::mc::elf::ElfTarget::ARM).unwrap();
        let Some(text) = super::tests::objdump(&elf, "suite") else {
            eprintln!("skipping: no llvm-objdump");
            return;
        };
        assert!(!text.contains("<unknown>"), "{text}");
        let code = &obj.sections().iter().find(|s| s.name == ".text").unwrap().bytes;
        // The disassembly's instruction addresses are exactly the boundaries
        // the Thumb length rule gives.
        let mut ours = Vec::new();
        let mut at = 0usize;
        while at + 1 < code.len() {
            ours.push(at);
            let h = u16::from_le_bytes([code[at], code[at + 1]]);
            at += if matches!(h >> 11, 0b11101..=0b11111) { 4 } else { 2 };
        }
        let theirs: Vec<usize> = text
            .lines()
            .filter_map(|l| {
                let (addr, rest) = l.trim_start().split_once(':')?;
                (!rest.trim().is_empty() && !rest.contains("R_ARM")).then(|| usize::from_str_radix(addr.trim(), 16).ok())?
            })
            .collect();
        assert_eq!(ours, theirs, "instruction boundaries");
    }
}

// ===========================================================================
// Vectors: scalarized, then soft-float and 64-bit legalization
// ===========================================================================

/// Run the shared vector fixtures (`(i64, i64, i64, i64) -> i64` functions)
/// on Thumb: each case gets a no-argument wrapper (the four `i64`s would not
/// fit the harness's register-only call), and its result must match the
/// reference executor on the original vector IR.
fn check_vectors(what: &str, src: &str, cases: &[crate::target::vector_fixtures::Case]) -> usize {
    let want = crate::target::vector_fixtures::reference(src, cases);
    let mut full = src.to_owned();
    for (k, (name, args)) in cases.iter().enumerate() {
        let a: Vec<String> = args.iter().map(|x| format!("i64 {x}")).collect();
        full += &format!(
            "func @case{k}() -> i64 {{\nentry ^0:\n  %r = call @{name}({}) : i64\n  ret %r\n}}\n",
            a.join(", ")
        );
    }
    let h = Harness::new(&full, &ThumbOptions::default());
    assert!(!crate::codegen::legalize::uses_vectors(&h.pm), "{what}: every vector is scalarized");
    let mut n = 0;
    for (k, w) in want.iter().enumerate() {
        let Some(w) = w else { continue };
        let (mir, hw) = h.exec(&format!("case{k}"), &[]);
        assert_eq!(mir, *w, "{what}: MIR interpreter, {:?}", cases[k]);
        assert_eq!(hw, *w, "{what}: machine code, {:?}", cases[k]);
        n += 1;
    }
    n
}

#[test]
fn vector_programs_are_scalarized_and_run() {
    use crate::target::vector_fixtures as vf;
    let mut n = 0;
    // Every function of each fixture, on the shared inputs.
    for (what, src) in [
        ("int arith", vf::int_arith_src()),
        ("compares", vf::compare_src()),
        ("floats", vf::FLOAT_SRC.to_owned()),
        ("lanes", vf::lanes_src()),
        ("masks", vf::masks_src()),
    ] {
        let names: Vec<&str> = src
            .split("func @")
            .skip(1)
            .filter_map(|rest| rest.split_once('(').map(|(n, _)| n))
            .collect();
        n += check_vectors(what, &src, &vf::cases(&names, &vf::INPUTS));
    }
    let mut rng = vf::Rng(0x7b7);
    for p in 0..3u64 {
        // Thumb has soft float: float vectors included.
        let (src, names) = vf::random_program(0x5000 + p, 4, 6, true);
        let mut cs = Vec::new();
        for name in &names {
            for _ in 0..2 {
                cs.push((name.clone(), vf::random_inputs(&mut rng)));
            }
        }
        n += check_vectors(&format!("random{p}"), &src, &cs);
    }
    eprintln!("vector programs: {n} results compared on Thumb");
    assert!(n >= 300, "{n}");
}
