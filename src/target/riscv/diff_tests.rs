//! The differential suite: IR programs run three ways and compared.
//!
//! 1. **The reference**: an interpreter over the *original* IR whose every
//!    value-producing step is [`crate::ir::eval`] — the executable semantics —
//!    plus memory, calls and control flow.
//! 2. **The MIR interpreter** ([`super::interp`]) over the RISC-V isel's
//!    output, before register allocation.
//! 3. **The machine code**: the full pipeline (isel, allocation, frame layout,
//!    encoding, relocations), linked by [`super::sim::link`] and executed by
//!    the RV64 simulator ([`super::sim`]), which decodes the bytes itself.
//!
//! Every executor shares one memory image (the compiled object's data, laid
//! out once), so addresses and initial data agree. Arguments are placed by
//! the LP64D rules ([`super::abi`]) — in `a0`–`a7`, `fa0`–`fa7` and stack
//! slots — with garbage above the width of every narrow integer but `i32`
//! (which the psABI sign-extends), so a callee that trusts upper bits fails.
//! A case whose reference result is poison or undefined behavior is skipped
//! (any result refines it), and two NaNs compare equal whatever their
//! payloads (the IR does not fix NaN payloads; RISC-V produces the canonical
//! NaN).

use std::collections::HashMap;

use crate::codegen::CodegenOptions;
use crate::codegen::mir::{MachineFunction, PReg};
use crate::ir::inst::InstKind;
use crate::ir::types::{FloatKind, Type, TypeId};
use crate::ir::value::{Const, FloatBits, ValueDef, ValueId};
use crate::ir::{EvalOutcome, FuncId, Module, SemValue};
use crate::support::StrInterner;
use crate::support::diagnostics::FileId;

use puremp::Int;

use super::abi::{self, Assigner, Loc, Part};
use super::regs::{fpr, gpr};
use super::sim::{Cpu, Image, Memory, STACK_TOP};

pub(super) fn parse(src: &str) -> (Module, StrInterner) {
    let mut syms = StrInterner::new();
    let m = crate::ir::text::parse_module(src, FileId::new(0), &mut syms)
        .unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    if let Err(d) = crate::verify::verify_module(&m) {
        panic!("verify: {d:#?}\n{src}");
    }
    (m, syms)
}

// ===========================================================================
// The reference interpreter
// ===========================================================================

struct Ref<'a> {
    m: &'a Module,
    syms: &'a StrInterner,
    mem: Memory,
    symbols: &'a HashMap<String, u64>,
    sp: u64,
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
    fn size(&self, ty: TypeId) -> u64 {
        self.m.types().size_of(ty)
    }

    fn value_of_bits(&self, ty: TypeId, raw: u64) -> SemValue {
        match self.m.types().get(ty) {
            Type::Int(w) => SemValue::int(*w, Int::from_u64(raw).mod_2k(*w)),
            Type::Float(FloatKind::F16) => SemValue::Float(FloatBits::F16(raw as u16)),
            Type::Float(FloatKind::F32) => SemValue::Float(FloatBits::F32(raw as u32)),
            Type::Float(FloatKind::F64) => SemValue::Float(FloatBits::F64(raw)),
            _ => SemValue::ptr(Int::from_u64(raw)),
        }
    }

    fn load(&self, ty: TypeId, addr: u64) -> SemValue {
        let n = self.size(ty);
        let raw = self.mem.read(addr, n.min(8));
        self.value_of_bits(ty, raw)
    }

    fn store(&mut self, ty: TypeId, addr: u64, v: &SemValue) -> Result<(), String> {
        let n = self.size(ty);
        let bits = match v {
            SemValue::Int { bits, .. } => bits.to_u64().unwrap_or(0),
            SemValue::Poison => return Err("store of poison".into()),
            other => value_bits(other).ok_or("a stored value")?,
        };
        self.mem.write(addr, n.min(8), bits);
        Ok(())
    }

    fn addr(v: &SemValue) -> Result<u64, String> {
        match v {
            SemValue::Ptr(a) => Ok(a.to_u64().unwrap_or(0)),
            SemValue::Int { bits, .. } => Ok(bits.to_u64().unwrap_or(0)),
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
                SemValue::ptr(Int::from_u64(self.symbols[name]))
            }
            ValueDef::Func(fid) => {
                let name = self.syms.resolve(self.m.function(*fid).name);
                SemValue::ptr(Int::from_u64(self.symbols[name]))
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
                        let align = self.m.types().align_of(*elem_ty).max(1);
                        self.sp = (self.sp - size) & !(align - 1);
                        Some(SemValue::ptr(Int::from_u64(self.sp)))
                    }
                    InstKind::DynAlloca { align } => {
                        let n = value_bits(&ops[0]).ok_or("dyn_alloca of poison")?;
                        self.sp = (self.sp - n.max(1)) & !(u64::from(*align).max(1) - 1);
                        Some(SemValue::ptr(Int::from_u64(self.sp)))
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

/// Where `args` (raw bit patterns per parameter) travel for a call of
/// signature `sig` under LP64D: register images and the stack-argument bytes.
/// Narrow integers carry garbage above their width (but an `i32` is
/// sign-extended, as the psABI requires of a C caller); a single in a float
/// register is NaN-boxed.
fn place(m: &Module, sig: TypeId, args: &[u64]) -> (Vec<(PReg, u64)>, Vec<u8>) {
    let Type::Func(ft) = m.types().get(sig) else { unreachable!() };
    let mut asg = Assigner::args(false);
    let mut regs = Vec::new();
    let mut stack = Vec::new();
    for (&p, &a) in ft.params.iter().zip(args) {
        let parts = asg.assign(m.types(), p, true);
        assert!(parts.len() == 1 && parts[0].0 == Part::Whole, "the harness passes scalars");
        let ty = m.types().get(p).clone();
        let v = match ty {
            Type::Int(32) => a as u32 as i32 as i64 as u64,
            Type::Int(w) if w < 64 => (a & ((1u64 << w) - 1)) | (0xa5a5_5a5a_c3c3_3c3cu64 << w),
            _ => a,
        };
        let reg_v = match (ty, parts[0].1) {
            (Type::Float(FloatKind::F32), Loc::Fpr(_)) => 0xffff_ffff_0000_0000 | (a & 0xffff_ffff),
            (Type::Float(FloatKind::F32), _) => a as u32 as i32 as i64 as u64,
            _ => v,
        };
        match parts[0].1 {
            Loc::Gpr(n) => regs.push((gpr(n), reg_v)),
            Loc::Fpr(n) => regs.push((fpr(n), reg_v)),
            Loc::Stack(off) => {
                let off = off as usize;
                stack.resize(off + 8, 0);
                stack[off..off + 8].copy_from_slice(&reg_v.to_le_bytes());
            }
        }
    }
    (regs, stack)
}

/// The width in bits of a scalar return type.
fn ret_width(m: &Module, ty: TypeId) -> u32 {
    match m.types().get(ty) {
        Type::Int(w) => *w,
        Type::Float(k) => k.bit_width(),
        _ => 64,
    }
}

fn masked(v: u64, w: u32) -> u64 {
    if w >= 64 { v } else { v & ((1u64 << w) - 1) }
}

/// Whether a value of type `ty` with bits `v` is a NaN.
fn is_nan(m: &Module, ty: TypeId, v: u64) -> bool {
    match m.types().get(ty) {
        Type::Float(FloatKind::F32) => f32::from_bits(v as u32).is_nan(),
        Type::Float(FloatKind::F64) => f64::from_bits(v).is_nan(),
        _ => false,
    }
}

/// A compiled program ready to run three ways.
pub(super) struct Harness {
    pub(super) m: Module,
    syms: StrInterner,
    pub(super) image: Image,
    funcs: Vec<MachineFunction>,
    target: super::RiscvTarget,
    globals: Vec<u64>,
    func_addrs: Vec<u64>,
    names: Vec<String>,
}

impl Harness {
    pub(super) fn new(src: &str) -> Harness {
        Harness::with_options(src, &CodegenOptions::default())
    }

    pub(super) fn with_options(src: &str, opts: &CodegenOptions) -> Harness {
        let (m, syms) = parse(src);
        let compiled = super::compile_module_with(&m, &syms, opts);
        let image = super::sim::link(&[&compiled.object]).unwrap_or_else(|e| panic!("link: {e}"));
        // The module the backend selects from (vectors scalarized, `fmod`
        // declared), lowered for the MIR interpreter.
        let legal = crate::codegen::legalize::legalized(&m, &crate::codegen::legalize::ScalarOnly).into_owned();
        let (pm, ps) = super::encode::prepare(&legal, &syms).unwrap_or_else(|| {
            let bytes = crate::ir::binary::encode(&legal, &syms);
            let mut s = StrInterner::new();
            (crate::ir::binary::decode(&bytes, &mut s).expect("round-trips"), s)
        });
        let funcs: Vec<MachineFunction> = (0..pm.function_count())
            .map(|i| {
                let fid = FuncId::from_index(i);
                if pm.function(fid).is_declaration() {
                    return MachineFunction::new(ps.resolve(pm.function(fid).name), i as u32);
                }
                let t = super::encode::target_for(&pm, fid, Some(&ps), opts);
                t.select(&pm, fid)
            })
            .collect();
        let addr = |name: &str| image.symbols.get(name).copied().unwrap_or(0);
        let names: Vec<String> = pm.functions().map(|f| ps.resolve(f.name).to_owned()).collect();
        let func_addrs = names.iter().map(|n| addr(n)).collect();
        let globals = pm.globals().map(|g| addr(ps.resolve(g.name))).collect();
        let target = super::RiscvTarget::for_module(&pm, Some(&ps), opts);
        Harness { m, syms, image, funcs, target, globals, func_addrs, names }
    }

    pub(super) fn func(&self, name: &str) -> FuncId {
        FuncId::from_index(
            self.m.functions().position(|f| self.syms.resolve(f.name) == name).unwrap_or_else(|| panic!("no @{name}")),
        )
    }

    /// The reference result of `name(args)`; `None` when poison or UB.
    pub(super) fn reference(&self, name: &str, args: &[u64]) -> Option<u64> {
        let fid = self.func(name);
        let f = self.m.function(fid);
        let Type::Func(ft) = self.m.types().get(f.sig).clone() else { unreachable!() };
        let sem: Vec<SemValue> = ft
            .params
            .iter()
            .zip(args)
            .map(|(&t, &a)| match self.m.types().get(t) {
                Type::Int(w) => SemValue::int(*w, Int::from_u64(masked(a, *w))),
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
            sp: STACK_TOP - (1 << 30),
            steps: 0,
        };
        let want = match r.call(fid, sem) {
            Ok(Some(v)) => value_bits(&v)?,
            Ok(None) => 0,
            Err(e) if e.contains("undefined behavior") || e.contains("poison") => return None,
            Err(e) => panic!("reference {name}{args:x?}: {e}"),
        };
        Some(masked(want, ret_width(&self.m, ft.ret)))
    }

    /// Run `name(args)` on the MIR interpreter and on the simulator, returning
    /// both results (masked to the return width).
    pub(super) fn exec(&self, name: &str, args: &[u64]) -> (u64, u64) {
        let fid = self.func(name);
        let f = self.m.function(fid);
        let Type::Func(ft) = self.m.types().get(f.sig).clone() else { unreachable!() };
        let (regs, stack) = place(&self.m, f.sig, args);
        let out_reg = match abi::ret_locs(self.m.types(), ft.ret).and_then(|p| p.first().map(|x| x.1)) {
            Some(Loc::Fpr(n)) => fpr(n),
            _ => gpr(10),
        };
        let w = ret_width(&self.m, ft.ret);

        let prog = super::interp::Program {
            target: &self.target,
            funcs: &self.funcs,
            globals: &self.globals,
            func_addrs: &self.func_addrs,
            names: &self.names,
        };
        let pidx = self.names.iter().position(|n| n == name).expect("prepared");
        let inputs: Vec<(PReg, Int)> = regs.iter().map(|&(r, v)| (r, Int::from_u64(v))).collect();
        let out = super::interp::run_program(&prog, self.image.mem.clone(), pidx, &inputs, &stack)
            .unwrap_or_else(|e| panic!("MIR interpreter, {name}{args:x?}: {e}"));
        let mir = masked(out.reg(out_reg), w);

        let hw = self.run_sim(name, &regs, &stack).unwrap_or_else(|e| panic!("simulator, {name}{args:x?}: {e}"));
        let hw = masked(if out_reg.class == crate::codegen::mir::RegClass::Fp { hw.1 } else { hw.0 }, w);
        (mir, hw)
    }

    /// Run `name` on the simulator with the given registers and stack
    /// arguments, returning `(a0, fa0)`.
    pub(super) fn run_sim(&self, name: &str, regs: &[(PReg, u64)], stack: &[u8]) -> Result<(u64, u64), String> {
        let mut cpu = Cpu::new(&self.image);
        let sp = STACK_TOP - 4096;
        cpu.mem.write_bytes(sp, stack);
        for &(r, v) in regs {
            match r.class {
                crate::codegen::mir::RegClass::Gpr => cpu.x[r.num as usize] = v,
                crate::codegen::mir::RegClass::Fp => cpu.f[r.num as usize] = v,
            }
        }
        // Callee-saved registers must come back intact.
        let canary = |i: usize| 0x5eed_0000_0000_0000u64 | (i as u64) << 8;
        let callee_x = [8usize, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];
        let callee_f = [8usize, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];
        for &i in &callee_x {
            cpu.x[i] = canary(i);
        }
        for &i in &callee_f {
            cpu.f[i] = canary(i + 64);
        }
        let entry = self.image.symbols[name];
        cpu.call(entry, &[], &[], sp).map_err(|e| e.to_string())?;
        for &i in &callee_x {
            if cpu.x[i] != canary(i) {
                return Err(format!("x{i} not preserved"));
            }
        }
        for &i in &callee_f {
            if cpu.f[i] != canary(i + 64) {
                return Err(format!("f{i} not preserved"));
            }
        }
        if cpu.x[2] != sp {
            return Err("sp not restored".into());
        }
        Ok((cpu.x[10], cpu.f[10]))
    }

    /// Check `name` on every argument tuple of `cases`; returns how many were
    /// compared (not skipped).
    pub(super) fn check(&self, name: &str, cases: &[Vec<u64>]) -> usize {
        let fid = self.func(name);
        let ret = match self.m.types().get(self.m.function(fid).sig) {
            Type::Func(ft) => ft.ret,
            _ => unreachable!(),
        };
        let mut n = 0;
        for args in cases {
            let Some(want) = self.reference(name, args) else { continue };
            let (mir, hw) = self.exec(name, args);
            let same = |got: u64| got == want || (is_nan(&self.m, ret, got) && is_nan(&self.m, ret, want));
            assert!(same(mir), "MIR interpreter: {name}({args:#x?}) = {mir:#x}, want {want:#x}");
            assert!(same(hw), "machine code: {name}({args:#x?}) = {hw:#x}, want {want:#x}");
            n += 1;
        }
        n
    }
}

// ===========================================================================
// Operand samples
// ===========================================================================

/// Interesting operand values for a `bits`-bit integer.
pub(super) fn samples(bits: u32) -> Vec<u64> {
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let mut v: Vec<u64> = vec![0, 1, 2, 3, 7, 0x7f, 0x80, 0xff, 0x100, 0x7fff, 0x8000, 0xffff, 0x1_0000, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0x1_0000_0000, 0x7fff_ffff_ffff_ffff, 0x8000_0000_0000_0000, u64::MAX, u64::MAX - 1];
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..3 {
        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        v.push(x);
    }
    let mut out: Vec<u64> = v.into_iter().map(|a| a & mask).collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn pairs(s: &[u64]) -> Vec<Vec<u64>> {
    s.iter().flat_map(|&a| s.iter().map(move |&b| vec![a, b])).collect()
}

/// `f32` edge cases: zeros, ones, NaNs, infinities, the extremes, subnormals,
/// a value whose sum rounds (ties), and a few ordinary values.
pub(super) fn f32s() -> Vec<u64> {
    let v: [f32; 22] = [
        0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, -2.5, 3.0e9, -3.0e9, f32::MAX, f32::MIN, f32::MIN_POSITIVE,
        1.0e-40, -1.0e-45, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 16_777_217.0, 0.1, 1.0e20, -7.75,
    ];
    let mut out: Vec<u64> = v.iter().map(|x| u64::from(x.to_bits())).collect();
    out.push(0x7f80_0001); // a signaling NaN
    out.push(0xffc0_0123); // a negative NaN with a payload
    out
}

/// `f64` edge cases (as [`f32s`]).
pub(super) fn f64s() -> Vec<u64> {
    let v: [f64; 24] = [
        0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, -2.5, 3.0e9, -3.0e9, 9.3e18, -9.3e18, 1.9e19, f64::MAX,
        f64::MIN, f64::MIN_POSITIVE, 5.0e-324, -2.0e-310, f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
        9_007_199_254_740_993.0, 0.1, -123.456,
    ];
    let mut out: Vec<u64> = v.iter().map(|x| x.to_bits()).collect();
    out.push(0x7ff0_0000_0000_0001);
    out.push(0xfff8_0000_0000_4321);
    out
}

// ===========================================================================
// Integer arithmetic at every width, through the machine code
// ===========================================================================

const BINOPS: [&str; 13] = ["add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr", "udiv", "sdiv", "urem", "srem"];
const IPREDS: [&str; 10] = ["eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"];

#[test]
fn integer_ops_at_every_width_match_the_reference() {
    let mut n = 0;
    for bits in [8u32, 16, 32, 64, 1, 5, 24, 40] {
        let t = format!("i{bits}");
        let mut src = String::from("module \"ops\"\n");
        for op in BINOPS {
            src += &format!("func @{op}({t}, {t}) -> {t} {{\nentry ^0(%a: {t}, %b: {t}):\n  %r = {op} %a, %b : {t}\n  ret %r\n}}\n");
        }
        for pred in IPREDS {
            src += &format!("func @{pred}({t}, {t}) -> i32 {{\nentry ^0(%a: {t}, %b: {t}):\n  %c = icmp {pred} %a, %b : i1\n  %r = zext %c : i32\n  ret %r\n}}\n");
        }
        let h = Harness::new(&src);
        let all = pairs(&samples(bits));
        let shifts: Vec<Vec<u64>> = samples(bits)
            .into_iter()
            .flat_map(|a| [0u64, 1, 3, 7, 8, 15, 16, 31, 32, 33, 63].into_iter().map(move |s| vec![a, s]))
            .collect();
        for op in BINOPS {
            n += h.check(op, if op.contains("sh") { &shifts } else { &all });
        }
        for pred in IPREDS {
            n += h.check(pred, &all);
        }
    }
    eprintln!("integer ops: {n} results compared three ways");
    assert!(n > 10_000);
}

// ===========================================================================
// Floating point
// ===========================================================================

const FBINOPS: [&str; 5] = ["fadd", "fsub", "fmul", "fdiv", "frem"];
const FPREDS: [&str; 16] = [
    "false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno", "true",
];

#[test]
fn float_arithmetic_and_compares_match_the_reference() {
    let mut n = 0;
    for (t, vals) in [("f32", f32s()), ("f64", f64s())] {
        let mut src = String::from("module \"fops\"\n");
        for op in FBINOPS {
            src += &format!("func @{op}({t}, {t}) -> {t} {{\nentry ^0(%a: {t}, %b: {t}):\n  %r = {op} %a, %b : {t}\n  ret %r\n}}\n");
        }
        for pred in FPREDS {
            src += &format!("func @{pred}({t}, {t}) -> i32 {{\nentry ^0(%a: {t}, %b: {t}):\n  %c = fcmp {pred} %a, %b : i1\n  %r = zext %c : i32\n  ret %r\n}}\n");
        }
        src += &format!("func @fneg({t}) -> {t} {{\nentry ^0(%a: {t}):\n  %r = fneg %a : {t}\n  ret %r\n}}\n");
        let h = Harness::new(&src);
        let all = pairs(&vals);
        for op in FBINOPS {
            n += h.check(op, &all);
        }
        for pred in FPREDS {
            n += h.check(pred, &all);
        }
        // `fneg` flips the sign bit of every value, NaNs included: compare
        // exactly.
        for &v in &vals {
            let w = if t == "f32" { 32 } else { 64 };
            let want = v ^ (1u64 << (w - 1));
            let (mir, hw) = h.exec("fneg", &[v]);
            assert_eq!((mir, hw), (want, want), "fneg {t} {v:#x}");
            n += 1;
        }
    }
    eprintln!("float arithmetic/compares: {n} results compared three ways");
    assert!(n > 3000);
}

#[test]
fn float_conversions_match_the_reference() {
    let mut n = 0;
    let mut src = String::from("module \"conv\"\n");
    let ints = [1u32, 5, 8, 16, 32, 40, 64];
    for ft in ["f32", "f64"] {
        for &w in &ints {
            for (op, nm) in [("fptosi", "s"), ("fptoui", "u")] {
                src += &format!("func @{nm}{ft}to{w}({ft}) -> i{w} {{\nentry ^0(%a: {ft}):\n  %r = {op} %a : i{w}\n  ret %r\n}}\n");
            }
            for (op, nm) in [("sitofp", "s"), ("uitofp", "u")] {
                src += &format!("func @{nm}{w}to{ft}(i{w}) -> {ft} {{\nentry ^0(%a: i{w}):\n  %r = {op} %a : {ft}\n  ret %r\n}}\n");
            }
        }
    }
    src += "func @ext(f32) -> f64 {\nentry ^0(%a: f32):\n  %r = fpext %a : f64\n  ret %r\n}\n";
    src += "func @trunc(f64) -> f32 {\nentry ^0(%a: f64):\n  %r = fptrunc %a : f32\n  ret %r\n}\n";
    src += "func @bc32(f32) -> i32 {\nentry ^0(%a: f32):\n  %r = bitcast %a : i32\n  ret %r\n}\n";
    src += "func @bc64(i64) -> f64 {\nentry ^0(%a: i64):\n  %r = bitcast %a : f64\n  ret %r\n}\n";
    src += "func @bcf32(i32) -> f32 {\nentry ^0(%a: i32):\n  %r = bitcast %a : f32\n  ret %r\n}\n";
    let h = Harness::new(&src);
    // Float sources: the edge cases plus values near every integer range.
    let near = |w: u32| -> Vec<f64> {
        let p = 2f64.powi(w as i32);
        vec![p - 1.0, p, p / 2.0 - 1.0, p / 2.0, -p / 2.0, -p / 2.0 - 1.0, 0.99, -0.99, 255.5, -128.75, 1e6 + 0.5]
    };
    for ft in ["f32", "f64"] {
        let base = if ft == "f32" { f32s() } else { f64s() };
        for &w in &ints {
            let mut vals = base.clone();
            for x in near(w) {
                vals.push(if ft == "f32" { u64::from((x as f32).to_bits()) } else { x.to_bits() });
            }
            let cases: Vec<Vec<u64>> = vals.iter().map(|&v| vec![v]).collect();
            n += h.check(&format!("s{ft}to{w}"), &cases);
            n += h.check(&format!("u{ft}to{w}"), &cases);
            let icases: Vec<Vec<u64>> = samples(w).into_iter().map(|v| vec![v]).collect();
            n += h.check(&format!("s{w}to{ft}"), &icases);
            n += h.check(&format!("u{w}to{ft}"), &icases);
        }
    }
    let one = |v: Vec<u64>| -> Vec<Vec<u64>> { v.into_iter().map(|x| vec![x]).collect() };
    n += h.check("ext", &one(f32s()));
    n += h.check("trunc", &one(f64s()));
    n += h.check("bc32", &one(f32s()));
    n += h.check("bc64", &one(f64s()));
    n += h.check("bcf32", &one(samples(32)));
    eprintln!("float conversions: {n} results compared three ways");
    assert!(n > 800);
}

/// Floats through memory, globals, block arguments, `select`, calls with
/// more float arguments than registers, mixed with integers, and enough
/// live values to spill the floating-point file.
#[test]
fn float_programs_match_the_reference() {
    let src = r#"
module "fprog"
global @k : f64 = f64 0x400921fb54442d18
global @acc : f32 = f32 0x00000000

func @poly(f64, i32) -> f64 {
entry ^0(%x: f64, %n: i32):
  %pi = load @k align 8 : f64
  br ^1(i32 0, f64 0x0000000000000000)
^1(%i: i32, %s: f64):
  %c = icmp slt %i, %n : i1
  cond_br %c, ^2, ^3(%s)
^2:
  %m = fmul %s, %x : f64
  %fi = sitofp %i : f64
  %t = fadd %m, %fi : f64
  %u = fadd %t, %pi : f64
  %j = add %i, i32 1 : i32
  br ^1(%j, %u)
^3(%r: f64):
  ret %r
}

func @sel(f32, f32, i8) -> f32 {
entry ^0(%a: f32, %b: f32, %k: i8):
  %c = icmp ugt %k, i8 100 : i1
  %r = select %c, %a, %b : f32
  %old = load @acc align 4 : f32
  %new = fadd %old, %r : f32
  store %new, @acc align 4 : f32
  ret %r
}

func @many(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f32, i64, f64) -> f64 {
entry ^0(%a: f64, %b: f64, %c: f64, %d: f64, %e: f64, %f: f64, %g: f64, %h: f64, %i: f64, %j: f64, %k: f64, %l: f32, %m: i64, %n: f64):
  %le = fpext %l : f64
  %mf = sitofp %m : f64
  %s1 = fmul %a, f64 0x4000000000000000 : f64
  %s2 = fsub %b, %s1 : f64
  %s3 = fmul %c, %s2 : f64
  %s4 = fadd %d, %s3 : f64
  %s5 = fdiv %e, %s4 : f64
  %s6 = fadd %f, %s5 : f64
  %s7 = fmul %g, %s6 : f64
  %s8 = fsub %h, %s7 : f64
  %s9 = fadd %i, %s8 : f64
  %s10 = fmul %j, %s9 : f64
  %s11 = fadd %k, %s10 : f64
  %s12 = fadd %le, %s11 : f64
  %s13 = fadd %mf, %s12 : f64
  %s14 = fsub %s13, %n : f64
  ret %s14
}

func @calls_many(f64, i64) -> f64 {
entry ^0(%x: f64, %y: i64):
  %x2 = fadd %x, f64 0x3ff0000000000000 : f64
  %t = fptrunc %x2 : f32
  %r = call @many(%x, %x2, %x, %x2, %x, %x2, %x, %x2, %x, %x2, %x, %t, %y, %x2) : f64
  %r2 = call @many(%r, %x, %x2, %x, %x2, %x, %x2, %x, %x2, %x, %x2, %t, %y, %x) : f64
  %s = fadd %r, %r2 : f64
  ret %s
}

func @mixed(i64, f32, i8, f64, i32, f32, i16, f64, i64, f32, i64, f64, i64, f32, i64, f64, i64, f32, i64, f64) -> f64 {
entry ^0(%a: i64, %b: f32, %c: i8, %d: f64, %e: i32, %f: f32, %g: i16, %h: f64, %i: i64, %j: f32, %k: i64, %l: f64, %m: i64, %n: f32, %o: i64, %p: f64, %q: i64, %r: f32, %s: i64, %t: f64):
  %ai = sitofp %a : f64
  %bi = fpext %b : f64
  %ci = sitofp %c : f64
  %ei = sitofp %e : f64
  %fi = fpext %f : f64
  %gi = uitofp %g : f64
  %ii = sitofp %i : f64
  %ji = fpext %j : f64
  %ki = sitofp %k : f64
  %mi = sitofp %m : f64
  %ni = fpext %n : f64
  %oi = sitofp %o : f64
  %qi = sitofp %q : f64
  %ri = fpext %r : f64
  %si = sitofp %s : f64
  %x1 = fadd %ai, %bi : f64
  %x2 = fmul %x1, %ci : f64
  %x3 = fsub %x2, %d : f64
  %x4 = fadd %x3, %ei : f64
  %x5 = fmul %x4, %fi : f64
  %x6 = fadd %x5, %gi : f64
  %x7 = fsub %x6, %h : f64
  %x8 = fadd %x7, %ii : f64
  %x9 = fadd %x8, %ji : f64
  %x10 = fmul %x9, %ki : f64
  %x11 = fadd %x10, %l : f64
  %x12 = fsub %x11, %mi : f64
  %x13 = fadd %x12, %ni : f64
  %x14 = fadd %x13, %oi : f64
  %x15 = fmul %x14, %p : f64
  %x16 = fadd %x15, %qi : f64
  %x17 = fadd %x16, %ri : f64
  %x18 = fsub %x17, %si : f64
  %x19 = fadd %x18, %t : f64
  ret %x19
}

func @calls_mixed(i64, f64) -> f64 {
entry ^0(%x: i64, %y: f64):
  %f = fptrunc %y : f32
  %c = trunc %x : i8
  %e = trunc %x : i32
  %g = trunc %x : i16
  %r = call @mixed(%x, %f, %c, %y, %e, %f, %g, %y, %x, %f, %x, %y, %x, %f, %x, %y, %x, %f, %x, %y) : f64
  ret %r
}

func @pressure(f64) -> f64 {
entry ^0(%x: f64):
  %a0 = fadd %x, f64 0x3ff0000000000000 : f64
  %a1 = fmul %a0, %x : f64
  %a2 = fadd %a1, %a0 : f64
  %a3 = fmul %a2, %a1 : f64
  %a4 = fsub %a3, %a2 : f64
  %a5 = fadd %a4, %a3 : f64
  %a6 = fmul %a5, %a0 : f64
  %a7 = fadd %a6, %a1 : f64
  %a8 = fsub %a7, %a2 : f64
  %a9 = fadd %a8, %a3 : f64
  %a10 = fmul %a9, %a4 : f64
  %a11 = fadd %a10, %a5 : f64
  %a12 = fsub %a11, %a6 : f64
  %a13 = fadd %a12, %a7 : f64
  %a14 = fmul %a13, %a8 : f64
  %a15 = fadd %a14, %a9 : f64
  %a16 = fsub %a15, %a10 : f64
  %a17 = fadd %a16, %a11 : f64
  %a18 = fmul %a17, %a12 : f64
  %a19 = fadd %a18, %a13 : f64
  %a20 = fsub %a19, %a14 : f64
  %a21 = fadd %a20, %a15 : f64
  %a22 = fmul %a21, %a16 : f64
  %a23 = fadd %a22, %a17 : f64
  %a24 = fsub %a23, %a18 : f64
  %a25 = fadd %a24, %a19 : f64
  %a26 = fmul %a25, %a20 : f64
  %a27 = fadd %a26, %a21 : f64
  %a28 = fsub %a27, %a22 : f64
  %a29 = fadd %a28, %a23 : f64
  %a30 = fmul %a29, %a24 : f64
  %a31 = fadd %a30, %a25 : f64
  %a32 = fsub %a31, %a26 : f64
  %a33 = fadd %a32, %a27 : f64
  %a34 = fmul %a33, %a28 : f64
  %a35 = fadd %a34, %a29 : f64
  %r0 = fadd %a35, %a0 : f64
  %r1 = fadd %r0, %a1 : f64
  %r2 = fadd %r1, %a2 : f64
  %r3 = fadd %r2, %a3 : f64
  %r4 = fadd %r3, %a4 : f64
  %r5 = fadd %r4, %a5 : f64
  %r6 = fadd %r5, %a6 : f64
  %r7 = fadd %r6, %a7 : f64
  %r8 = fadd %r7, %a8 : f64
  %r9 = fadd %r8, %a9 : f64
  %r10 = fadd %r9, %a10 : f64
  %r11 = fadd %r10, %a11 : f64
  %r12 = fadd %r11, %a12 : f64
  %r13 = fadd %r12, %a13 : f64
  %r14 = fadd %r13, %a14 : f64
  %r15 = fadd %r14, %a15 : f64
  %r16 = fadd %r15, %a16 : f64
  %r17 = fadd %r16, %a17 : f64
  %r18 = fadd %r17, %a18 : f64
  %r19 = fadd %r18, %a19 : f64
  %r20 = fadd %r19, %a20 : f64
  %r21 = fadd %r20, %a21 : f64
  %r22 = fadd %r21, %a22 : f64
  %r23 = fadd %r22, %a23 : f64
  %r24 = fadd %r23, %a24 : f64
  %r25 = fadd %r24, %a25 : f64
  %r26 = fadd %r25, %a26 : f64
  %r27 = fadd %r26, %a27 : f64
  %r28 = fadd %r27, %a28 : f64
  %r29 = fadd %r28, %a29 : f64
  %r30 = fadd %r29, %a30 : f64
  %r31 = fadd %r30, %a31 : f64
  %r32 = fadd %r31, %a32 : f64
  %r33 = fadd %r32, %a33 : f64
  %r34 = fadd %r33, %a34 : f64
  %r35 = call @poly(%r34, i32 2) : f64
  %r36 = fadd %r35, %a0 : f64
  %r37 = fadd %r36, %a17 : f64
  %r38 = fadd %r37, %a35 : f64
  ret %r38
}
"#;
    let h = Harness::new(src);
    let mut n = 0;
    let xs: Vec<u64> = [0.0f64, 1.5, -2.25, 1e-3, 3.0e10, f64::NAN, -0.0].iter().map(|x| x.to_bits()).collect();
    for &x in &xs {
        for nn in [0u64, 1, 5, 17] {
            n += h.check("poly", &[vec![x, nn]]);
        }
        n += h.check("calls_many", &[vec![x, 7], vec![x, u64::MAX]]);
        n += h.check("calls_mixed", &[vec![3, x], vec![u64::MAX - 1000, x], vec![0x1234_5678_9abc, x]]);
        n += h.check("pressure", &[vec![x]]);
    }
    let fs: Vec<u64> = f32s();
    for &a in fs.iter().step_by(3) {
        for k in [0u64, 100, 101, 255, 0x1ff] {
            n += h.check("sel", &[vec![a, 0x4049_0fdb, k]]);
        }
    }
    eprintln!("float programs: {n} results compared three ways");
    assert!(n > 100);
}

/// One struct shape for [`structs_cross_calls_in_every_class`]: its type,
/// the IR building a value from two scalars `%x: i64` and `%y: f64` into the
/// storage `%p`, and the IR folding a value at `%q` into an `f64` `%r`.
struct Shape {
    name: &'static str,
    ty: &'static str,
    build: &'static str,
    fold: &'static str,
}

const SHAPES: &[Shape] = &[
    // FP convention: one float, two floats, float + integer (both orders).
    Shape { name: "d", ty: "{f64}", build: "store %y, %p align 8 : f64", fold: "%r = load %q align 8 : f64" },
    Shape {
        name: "dd",
        ty: "{f64, f64}",
        build: "store %y, %p align 8 : f64\n  %p1 = ptr_add %p, i64 8 : ptr\n  %y2 = fmul %y, f64 0xc000000000000000 : f64\n  store %y2, %p1 align 8 : f64",
        fold: "%a = load %q align 8 : f64\n  %q1 = ptr_add %q, i64 8 : ptr\n  %b = load %q1 align 8 : f64\n  %r = fsub %a, %b : f64",
    },
    Shape {
        name: "fi",
        ty: "{f32, i32}",
        build: "%t = fptrunc %y : f32\n  store %t, %p align 4 : f32\n  %p1 = ptr_add %p, i64 4 : ptr\n  %x32 = trunc %x : i32\n  store %x32, %p1 align 4 : i32",
        fold: "%a = load %q align 4 : f32\n  %q1 = ptr_add %q, i64 4 : ptr\n  %b = load %q1 align 4 : i32\n  %ad = fpext %a : f64\n  %bd = sitofp %b : f64\n  %r = fadd %ad, %bd : f64",
    },
    Shape {
        name: "bd",
        ty: "{i8, f64}",
        build: "%x8 = trunc %x : i8\n  store %x8, %p align 1 : i8\n  %p1 = ptr_add %p, i64 8 : ptr\n  store %y, %p1 align 8 : f64",
        fold: "%a = load %q align 1 : i8\n  %q1 = ptr_add %q, i64 8 : ptr\n  %b = load %q1 align 8 : f64\n  %ad = sitofp %a : f64\n  %r = fsub %b, %ad : f64",
    },
    Shape {
        name: "af",
        ty: "{[1 x f32], {f32}}",
        build: "%t = fptrunc %y : f32\n  store %t, %p align 4 : f32\n  %p1 = ptr_add %p, i64 4 : ptr\n  %u = fneg %t : f32\n  store %u, %p1 align 4 : f32",
        fold: "%a = load %q align 4 : f32\n  %q1 = ptr_add %q, i64 4 : ptr\n  %b = load %q1 align 4 : f32\n  %s = fmul %a, %b : f32\n  %r = fpext %s : f64",
    },
    // Integer convention: 1, 7, 12 and 16 bytes; by reference past 16.
    Shape { name: "c", ty: "{i8}", build: "%x8 = trunc %x : i8\n  store %x8, %p align 1 : i8", fold: "%a = load %q align 1 : i8\n  %r = uitofp %a : f64" },
    Shape {
        name: "s7",
        ty: "[7 x i8]",
        build: "%x32 = trunc %x : i32\n  store %x32, %p align 1 : i32\n  %p1 = ptr_add %p, i64 3 : ptr\n  %h = lshr %x, i64 24 : i64\n  %h32 = trunc %h : i32\n  store %h32, %p1 align 1 : i32",
        fold: "%a = load %q align 1 : i32\n  %q1 = ptr_add %q, i64 3 : ptr\n  %b = load %q1 align 1 : i32\n  %ab = zext %a : i64\n  %bb = zext %b : i64\n  %sh = shl %bb, i64 24 : i64\n  %o = or %ab, %sh : i64\n  %r = uitofp %o : f64",
    },
    Shape {
        name: "fff",
        ty: "{f32, f32, f32}",
        build: "%t = fptrunc %y : f32\n  store %t, %p align 4 : f32\n  %p1 = ptr_add %p, i64 4 : ptr\n  %t1 = fadd %t, f32 0x3f800000 : f32\n  store %t1, %p1 align 4 : f32\n  %p2 = ptr_add %p, i64 8 : ptr\n  %t2 = fadd %t1, f32 0x40000000 : f32\n  store %t2, %p2 align 4 : f32",
        fold: "%a = load %q align 4 : f32\n  %q1 = ptr_add %q, i64 4 : ptr\n  %b = load %q1 align 4 : f32\n  %q2 = ptr_add %q, i64 8 : ptr\n  %c = load %q2 align 4 : f32\n  %s = fsub %a, %b : f32\n  %s2 = fmul %s, %c : f32\n  %r = fpext %s2 : f64",
    },
    Shape {
        name: "ll",
        ty: "{i64, i64}",
        build: "store %x, %p align 8 : i64\n  %p1 = ptr_add %p, i64 8 : ptr\n  %x2 = mul %x, i64 3 : i64\n  store %x2, %p1 align 8 : i64",
        fold: "%a = load %q align 8 : i64\n  %q1 = ptr_add %q, i64 8 : ptr\n  %b = load %q1 align 8 : i64\n  %d = sub %b, %a : i64\n  %r = sitofp %d : f64",
    },
    Shape {
        name: "big",
        ty: "{i64, f64, i64}",
        build: "store %x, %p align 8 : i64\n  %p1 = ptr_add %p, i64 8 : ptr\n  store %y, %p1 align 8 : f64\n  %p2 = ptr_add %p, i64 16 : ptr\n  %x2 = xor %x, i64 255 : i64\n  store %x2, %p2 align 8 : i64",
        fold: "%a = load %q align 8 : i64\n  %q1 = ptr_add %q, i64 8 : ptr\n  %b = load %q1 align 8 : f64\n  %q2 = ptr_add %q, i64 16 : ptr\n  %c = load %q2 align 8 : i64\n  %d = sub %c, %a : i64\n  %dd = sitofp %d : f64\n  %r = fadd %dd, %b : f64",
    },
];

/// By-value structs of every LP64D class cross calls as arguments and
/// results: alone (registers), after eight doubles (the FP fields fall back
/// to the integer convention), after seven longs (a 16-byte value splits
/// between `a7` and the stack) and after eight longs (all on the stack).
#[test]
fn structs_cross_calls_in_every_class() {
    let mut src = String::from("module \"structs\"\n");
    for s in SHAPES {
        let (n, t) = (s.name, s.ty);
        src += &format!(
            "func @mk_{n}(i64, f64) -> {t} {{\nentry ^0(%x: i64, %y: f64):\n  %p = alloca {t} : ptr\n  {}\n  ret %p\n}}\n",
            s.build
        );
        src += &format!("func @use_{n}({t}) -> f64 {{\nentry ^0(%q: ptr):\n  {}\n  ret %r\n}}\n", s.fold);
        // After eight doubles: no FP argument register left.
        src += &format!(
            "func @use8d_{n}(f64, f64, f64, f64, f64, f64, f64, f64, {t}, f64) -> f64 {{\n\
             entry ^0(%d0: f64, %d1: f64, %d2: f64, %d3: f64, %d4: f64, %d5: f64, %d6: f64, %d7: f64, %q: ptr, %z: f64):\n  \
             {}\n  %s0 = fadd %r, %d0 : f64\n  %s1 = fadd %s0, %d7 : f64\n  %s2 = fadd %s1, %z : f64\n  ret %s2\n}}\n",
            s.fold
        );
        // After seven longs: one integer register left.
        src += &format!(
            "func @use7l_{n}(i64, i64, i64, i64, i64, i64, i64, {t}, i64) -> f64 {{\n\
             entry ^0(%l0: i64, %l1: i64, %l2: i64, %l3: i64, %l4: i64, %l5: i64, %l6: i64, %q: ptr, %z: i64):\n  \
             {}\n  %w = add %l0, %l6 : i64\n  %w2 = add %w, %z : i64\n  %wf = sitofp %w2 : f64\n  %s = fadd %r, %wf : f64\n  ret %s\n}}\n",
            s.fold
        );
        // After eight longs and eight doubles: on the stack.
        src += &format!(
            "func @use16_{n}(i64, i64, i64, i64, i64, i64, i64, i64, f64, f64, f64, f64, f64, f64, f64, f64, {t}, i64) -> f64 {{\n\
             entry ^0(%l0: i64, %l1: i64, %l2: i64, %l3: i64, %l4: i64, %l5: i64, %l6: i64, %l7: i64, \
             %d0: f64, %d1: f64, %d2: f64, %d3: f64, %d4: f64, %d5: f64, %d6: f64, %d7: f64, %q: ptr, %z: i64):\n  \
             {}\n  %w = add %l7, %z : i64\n  %wf = sitofp %w : f64\n  %s = fadd %r, %wf : f64\n  %s2 = fadd %s, %d7 : f64\n  ret %s2\n}}\n",
            s.fold
        );
        src += &format!(
            "func @t_{n}(i64, f64) -> f64 {{\nentry ^0(%x: i64, %y: f64):\n  \
             %v = call @mk_{n}(%x, %y) : {t}\n  %a = call @use_{n}(%v) : f64\n  \
             %b = call @use8d_{n}(%y, %y, %y, %y, %y, %y, %y, %y, %v, %y) : f64\n  \
             %c = call @use7l_{n}(%x, %x, %x, %x, %x, %x, %x, %v, %x) : f64\n  \
             %d = call @use16_{n}(%x, %x, %x, %x, %x, %x, %x, %x, %y, %y, %y, %y, %y, %y, %y, %y, %v, %x) : f64\n  \
             %s1 = fadd %a, %b : f64\n  %s2 = fadd %s1, %c : f64\n  %s3 = fadd %s2, %d : f64\n  ret %s3\n}}\n"
        );
    }
    let h = Harness::new(&src);
    let mut n = 0;
    for s in SHAPES {
        for (x, y) in [(5u64, 1.5f64), (u64::MAX - 7, -0.25), (0x1234_5678_9abc_def0, 3.0e9), (0, -0.0)] {
            n += h.check(&format!("t_{}", s.name), &[vec![x, y.to_bits()]]);
        }
    }
    assert_eq!(n, 4 * SHAPES.len());
}

/// `contract` lets a multiply-add fuse into one `fmadd`-family instruction:
/// the machine code then rounds once (the reference, which never fuses,
/// rounds twice), so the result is the exactly-rounded `a*b + c`. Without
/// `contract`, nothing fuses.
#[test]
fn contract_fuses_multiply_add_and_nothing_else_does() {
    let src = r#"
module "fma"
func @fma(f64, f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64, %c: f64):
  %m = fmul contract %a, %b : f64
  %r = fadd contract %m, %c : f64
  ret %r
}
func @fms(f32, f32, f32) -> f32 {
entry ^0(%a: f32, %b: f32, %c: f32):
  %m = fmul contract %a, %b : f32
  %r = fsub contract %m, %c : f32
  ret %r
}
func @fnms(f64, f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64, %c: f64):
  %m = fmul contract %a, %b : f64
  %r = fsub contract %c, %m : f64
  ret %r
}
func @nofuse(f64, f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64, %c: f64):
  %m = fmul contract %a, %b : f64
  %r = fadd %m, %c : f64
  ret %r
}
func @twouses(f64, f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64, %c: f64):
  %m = fmul contract %a, %b : f64
  %r = fadd contract %m, %c : f64
  %s = fadd contract %r, %m : f64
  ret %s
}
"#;
    let h = Harness::new(src);
    let ops = |name: &str| -> Vec<super::RvOp> {
        let mf = &h.funcs[h.names.iter().position(|n| n == name).unwrap()];
        mf.block_ids().flat_map(|b| mf.block(b).insts.iter().map(|i| super::RvOp::decode(i.opcode))).collect()
    };
    for name in ["fma", "fms", "fnms"] {
        let o = ops(name);
        assert!(o.contains(&super::RvOp::FMadd) && !o.contains(&super::RvOp::FMul), "{name}: {o:?}");
    }
    for name in ["nofuse", "twouses"] {
        assert!(!ops(name).contains(&super::RvOp::FMadd), "{name}");
    }
    // 1 + 2^-30 squared minus its rounded square: only a fused multiply-add
    // sees the 2^-60 term.
    let a = 1.0f64 + 2f64.powi(-30);
    let c = -(a * a);
    let (mir, hw) = h.exec("fma", &[a.to_bits(), a.to_bits(), c.to_bits()]);
    let want = a.mul_add(a, c);
    assert_ne!(want, 0.0);
    assert_eq!((f64::from_bits(mir), f64::from_bits(hw)), (want, want));
    let (_, hw) = h.exec("fnms", &[a.to_bits(), a.to_bits(), (a * a).to_bits()]);
    assert_eq!(f64::from_bits(hw), (-a).mul_add(a, a * a));
    let (_, hw) = h.exec("nofuse", &[a.to_bits(), a.to_bits(), c.to_bits()]);
    assert_eq!(f64::from_bits(hw), 0.0);
    // The f32 fmsub: compare against the host's fused single.
    let x = 1.0f32 + 2f32.powi(-12);
    let (_, hw) = h.exec("fms", &[u64::from(x.to_bits()), u64::from(x.to_bits()), u64::from((x * x).to_bits())]);
    assert_eq!(f32::from_bits(hw as u32), x.mul_add(x, -(x * x)));
}

/// The shared vector fixtures — integer and float lanes, compares, lane
/// moves, masks, edge cases, register pressure, and random programs (with
/// float vectors, now that the backend has the F and D extensions) —
/// scalarized and run as machine code against the reference executor.
#[test]
fn vector_programs_are_scalarized_and_run_as_machine_code() {
    use crate::target::vector_fixtures as vf;
    let mut srcs: Vec<(String, String)> = vec![
        ("int".into(), vf::int_arith_src()),
        ("cmp".into(), vf::compare_src()),
        ("flt".into(), vf::FLOAT_SRC.to_string()),
        ("lane".into(), vf::lanes_src()),
        ("mask".into(), vf::masks_src()),
        ("edges".into(), vf::EDGES_SRC.to_string()),
        ("pressure".into(), vf::PRESSURE_SRC.to_string()),
    ];
    for p in 0..4u64 {
        srcs.push((format!("rand{p}"), vf::random_program(0x7000 + p, 5, 8, true).0));
    }
    let mut rng = vf::Rng(0x5c6);
    let mut n = 0;
    for (what, src) in srcs {
        let (m, syms) = vf::parse(&src);
        // The `(i64, i64, i64, i64) -> i64` test functions.
        let names: Vec<String> = m
            .functions()
            .filter(|f| {
                matches!(m.types().get(f.sig), Type::Func(ft) if ft.params.len() == 4 && !f.is_declaration())
            })
            .map(|f| syms.resolve(f.name).to_owned())
            .collect();
        let mut cs: Vec<vf::Case> = Vec::new();
        for name in &names {
            for a in vf::INPUTS {
                cs.push((name.clone(), a.to_vec()));
            }
            cs.push((name.clone(), vf::random_inputs(&mut rng)));
        }
        let want = vf::reference(&src, &cs);
        let h = Harness::new(&src);
        for ((name, args), w) in cs.iter().zip(&want) {
            let Some(w) = w else { continue };
            let a: Vec<u64> = args.iter().map(|&x| x as u64).collect();
            let (mir, hw) = h.exec(name, &a);
            assert_eq!(mir, *w, "{what}: MIR interpreter @{name}({args:x?})");
            assert_eq!(hw, *w, "{what}: machine code @{name}({args:x?})");
            n += 1;
        }
    }
    eprintln!("vector programs: {n} results compared");
    assert!(n > 200);
}
