//! A whole-program **reference interpreter** for IR modules, the oracle the
//! wasm differential tests compare against.
//!
//! Pure operations go through the reference evaluator ([`crate::ir::eval`]),
//! so their meaning — wrapping, poison, UB — is exactly the IR semantics; this
//! file only adds what the evaluator leaves to a machine model: control flow
//! and block arguments, calls, a flat little-endian byte memory holding the
//! globals (serialized by the shared [`crate::codegen::data`] emitter) and a
//! downward stack for `alloca`/`dyn_alloca`, sequential atomics, and the host
//! functions the tests import (mirrored in the node harness).

use std::collections::HashMap;

use crate::ir::inst::{InstKind, RmwOp};
use crate::ir::types::{FloatKind, Type, TypeId};
use crate::ir::value::{Const, FloatBits, ValueDef, ValueId};
use crate::ir::{BlockId, EvalOutcome, FuncId, Module, SemValue};
use crate::mc::object::{ObjectModule, RelocKind, SymbolValue};
use crate::support::StrInterner;

use puremp::Int;

/// Where the interpreter's function "addresses" start (never dereferenced).
const FUNC_BASE: u64 = 0xf000_0000;
/// Where the globals start.
const DATA_BASE: u64 = 0x1000;
/// Memory size; the stack grows down from the top.
const MEM_SIZE: usize = 4 << 20;

/// Why a run stopped without a result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
    /// The program has undefined behavior on these inputs.
    Ub(String),
    /// It ran too long (or recursed too deep).
    Budget,
}

pub(crate) struct Interp<'m> {
    m: &'m Module,
    syms: &'m StrInterner,
    mem: Vec<u8>,
    global_addr: Vec<u64>,
    sp: u64,
    steps: u64,
    depth: u32,
}

fn ub<T>(why: impl Into<String>) -> Result<T, Stop> {
    Err(Stop::Ub(why.into()))
}

/// The unsigned `u64` of an integer or pointer value (`None` for poison).
pub(crate) fn bits_u64(v: &SemValue) -> Option<u64> {
    match v {
        SemValue::Int { bits, .. } | SemValue::Ptr(bits) => bits.mod_2k(64).to_u64(),
        SemValue::Float(FloatBits::F32(b)) => Some(u64::from(*b)),
        SemValue::Float(FloatBits::F64(b)) => Some(*b),
        _ => None,
    }
}

impl<'m> Interp<'m> {
    pub(crate) fn new(m: &'m Module, syms: &'m StrInterner) -> Interp<'m> {
        let mut me = Interp {
            m,
            syms,
            mem: vec![0; MEM_SIZE],
            global_addr: vec![0; m.global_count()],
            sp: MEM_SIZE as u64,
            steps: 0,
            depth: 0,
        };
        me.init_globals();
        me
    }

    fn func_by_name(&self, name: &str) -> Option<FuncId> {
        (0..self.m.function_count()).map(FuncId::from_index).find(|&f| self.syms.resolve(self.m.function(f).name) == name)
    }

    fn init_globals(&mut self) {
        let mut obj = ObjectModule::new("ref");
        crate::codegen::data::emit_globals(self.m, self.syms, &mut obj, RelocKind::Abs32);
        let mut base = Vec::new();
        let mut at = DATA_BASE;
        for s in obj.sections() {
            at = at.div_ceil(s.align.max(1)) * s.align.max(1);
            base.push(at);
            if !s.is_nobits() {
                self.mem[at as usize..at as usize + s.bytes.len()].copy_from_slice(&s.bytes);
            }
            at += s.size();
        }
        let mut addr_of: HashMap<String, u64> = HashMap::new();
        for s in obj.symbols() {
            if let SymbolValue::Defined { section, offset } = s.value {
                addr_of.insert(s.name.clone(), base[section.index()] + offset);
            }
        }
        // Globals without storage here still get a (zeroed) cell.
        for (i, g) in self.m.globals().enumerate() {
            let name = self.syms.resolve(g.name).to_owned();
            let a = *addr_of.entry(name).or_insert_with(|| {
                at = at.div_ceil(16) * 16;
                let a = at;
                at += self.m.types().size_of(g.ty).max(1);
                a
            });
            self.global_addr[i] = a;
        }
        for r in obj.relocations() {
            let name = &obj.symbol(r.symbol).name;
            let target = match self.func_by_name(name) {
                Some(f) => FUNC_BASE + f.index() as u64,
                None => addr_of[name.as_str()],
            };
            let at = (base[r.section.index()] + r.offset) as usize;
            let v = (target as i64 + r.addend) as u32;
            self.mem[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    /// Run `name(args)`; `Ok(None)` for a void function.
    pub(crate) fn run(&mut self, name: &str, args: Vec<SemValue>) -> Result<Option<SemValue>, Stop> {
        let f = self.func_by_name(name).unwrap_or_else(|| panic!("no function {name}"));
        self.steps = 0;
        // Deep recursion in the program is deep recursion here: give it room.
        std::thread::scope(|s| {
            std::thread::Builder::new()
                .stack_size(512 << 20)
                .spawn_scoped(s, || self.call(f, args))
                .expect("spawn")
                .join()
                .expect("reference interpreter panicked")
        })
    }

    fn const_value(&self, c: &Const) -> SemValue {
        let types = self.m.types();
        match c {
            Const::Int { ty, value } => SemValue::int(types.bit_width(*ty).expect("int"), value.clone()),
            Const::Float { bits, .. } => SemValue::Float(*bits),
            Const::Null(_) => SemValue::Ptr(Int::ZERO),
            Const::Poison(_) => SemValue::Poison,
            other => panic!("constant {other:?} as an operand"),
        }
    }

    fn call(&mut self, fid: FuncId, args: Vec<SemValue>) -> Result<Option<SemValue>, Stop> {
        let f = self.m.function(fid);
        if f.is_declaration() {
            return self.host(self.syms.resolve(f.name), &args);
        }
        self.depth += 1;
        if self.depth > 2000 {
            return Err(Stop::Budget);
        }
        let saved_sp = self.sp;
        let types = self.m.types();
        let mut vals: Vec<Option<SemValue>> = vec![None; f.value_count()];
        let mut block = f.entry().expect("defined");
        for (&p, a) in f.block(block).params().iter().zip(args) {
            vals[p.index()] = Some(a);
        }
        let get = |me: &Self, vals: &[Option<SemValue>], v: ValueId| -> SemValue {
            match &f.value(v).def {
                ValueDef::Inst(_) | ValueDef::Param(..) => vals[v.index()].clone().expect("defined before use"),
                ValueDef::Const(c) => me.const_value(me.m.consts().get(*c)),
                ValueDef::Global(g) => SemValue::Ptr(Int::from_u64(me.global_addr[g.index()])),
                ValueDef::Func(fi) => SemValue::Ptr(Int::from_u64(FUNC_BASE + fi.index() as u64)),
            }
        };
        let result = loop {
            let b = f.block(block);
            for &i in b.insts() {
                self.steps += 1;
                if self.steps > 2_000_000 {
                    return Err(Stop::Budget);
                }
                let inst = f.inst(i);
                let ops: Vec<SemValue> = inst.operands().iter().map(|&o| get(self, &vals, o)).collect();
                let out = match &inst.kind {
                    InstKind::Alloca { elem_ty } => {
                        let l = types.layout(*elem_ty);
                        Some(self.alloc(l.size.max(1), l.align.max(1)))
                    }
                    InstKind::DynAlloca { align } => {
                        let Some(n) = bits_u64(&ops[0]) else { return ub("dyn_alloca of poison") };
                        Some(self.alloc(n, u64::from(*align).max(16)))
                    }
                    InstKind::Load { ty, .. } | InstKind::AtomicLoad { ty, .. } => Some(self.load(*ty, &ops[0])?),
                    InstKind::Store { ty, .. } | InstKind::AtomicStore { ty, .. } => {
                        self.store(*ty, &ops[0], &ops[1])?;
                        None
                    }
                    InstKind::AtomicRmw { op, ty, .. } => {
                        let old = self.load(*ty, &ops[0])?;
                        let width = types.int_or_ptr_bits(*ty).expect("int");
                        let (Some(o), Some(v)) = (bits_u64(&old), bits_u64(&ops[1])) else {
                            return ub("poison in an atomic rmw");
                        };
                        let new = if *op == RmwOp::Xchg { v } else { op.apply(o, v, width) };
                        let nv = self.value_of_bits(*ty, new);
                        self.store(*ty, &ops[0], &nv)?;
                        Some(old)
                    }
                    InstKind::CmpXchg { ty, .. } => {
                        let old = self.load(*ty, &ops[0])?;
                        if bits_u64(&old) == bits_u64(&ops[1]) {
                            self.store(*ty, &ops[0], &ops[2])?;
                        }
                        Some(old)
                    }
                    InstKind::Fence(_) => None,
                    InstKind::Call => {
                        let callee = match &ops[0] {
                            SemValue::Ptr(a) => {
                                let a = a.to_u64().unwrap_or(0);
                                if a < FUNC_BASE || a >= FUNC_BASE + self.m.function_count() as u64 {
                                    return ub("call through a non-function pointer");
                                }
                                FuncId::from_index((a - FUNC_BASE) as usize)
                            }
                            _ => return ub("call through poison"),
                        };
                        self.call(callee, ops[1..].to_vec())?
                    }
                    InstKind::Syscall => panic!("syscall in a wasm test"),
                    kind => match crate::ir::eval(types, inst.ty, kind, &inst.flags, &ops) {
                        EvalOutcome::Value(v) => Some(v),
                        EvalOutcome::UndefinedBehavior => return ub(format!("{kind:?}")),
                    },
                };
                if let (Some(r), Some(v)) = (inst.result(), out) {
                    vals[r.index()] = Some(v);
                }
            }
            let t = f.inst(b.terminator().expect("terminated"));
            let ops: Vec<SemValue> = t.operands().iter().map(|&o| get(self, &vals, o)).collect();
            let (target, args): (BlockId, Vec<SemValue>) = match &t.kind {
                InstKind::Ret => break ops.first().cloned(),
                InstKind::Unreachable => return ub("reached unreachable"),
                InstKind::Br(target) => (*target, ops),
                InstKind::CondBr { if_true, if_false, true_args, false_args } => {
                    let (ta, fa) = (*true_args as usize, *false_args as usize);
                    match bits_u64(&ops[0]) {
                        Some(1) => (*if_true, ops[1..1 + ta].to_vec()),
                        Some(0) => (*if_false, ops[1 + ta..1 + ta + fa].to_vec()),
                        _ => return ub("branch on poison"),
                    }
                }
                InstKind::Switch(data) => {
                    let SemValue::Int { width, bits } = &ops[0] else { return ub("switch on poison") };
                    let mut at = 1 + data.default_args as usize;
                    let mut chosen = (data.default, ops[1..at].to_vec());
                    for c in &data.cases {
                        let n = c.args as usize;
                        if c.value.mod_2k(*width) == *bits {
                            chosen = (c.target, ops[at..at + n].to_vec());
                            break;
                        }
                        at += n;
                    }
                    chosen
                }
                other => panic!("terminator {other:?}"),
            };
            for (&p, a) in f.block(target).params().iter().zip(args) {
                vals[p.index()] = Some(a);
            }
            block = target;
        };
        self.sp = saved_sp;
        self.depth -= 1;
        Ok(result)
    }

    fn alloc(&mut self, size: u64, align: u64) -> SemValue {
        self.sp = (self.sp - size) / align * align;
        SemValue::Ptr(Int::from_u64(self.sp))
    }

    fn addr(&self, p: &SemValue, size: u64) -> Result<usize, Stop> {
        let SemValue::Ptr(a) = p else { return ub("access through poison") };
        let a = a.to_u64().unwrap_or(u64::MAX);
        if a < 16 || a.saturating_add(size) > MEM_SIZE as u64 {
            return ub(format!("access at {a:#x} out of the interpreter's memory"));
        }
        Ok(a as usize)
    }

    fn value_of_bits(&self, ty: TypeId, v: u64) -> SemValue {
        match self.m.types().get(ty) {
            Type::Int(w) => SemValue::int(*w, Int::from_u64(v)),
            Type::Float(FloatKind::F32) => SemValue::Float(FloatBits::F32(v as u32)),
            Type::Float(FloatKind::F64) => SemValue::Float(FloatBits::F64(v)),
            _ => SemValue::Ptr(Int::from_u64(v & 0xffff_ffff)),
        }
    }

    fn load(&mut self, ty: TypeId, p: &SemValue) -> Result<SemValue, Stop> {
        let size = self.m.types().size_of(ty);
        let a = self.addr(p, size)?;
        let bytes = &self.mem[a..a + size as usize];
        let mut v = Int::ZERO;
        for (k, &b) in bytes.iter().enumerate() {
            v = v.add(&Int::from_u64(u64::from(b)).mul_2k(8 * k as u32));
        }
        Ok(match self.m.types().get(ty) {
            Type::Int(w) => SemValue::int(*w, v),
            Type::Float(FloatKind::F32) => SemValue::Float(FloatBits::F32(v.to_u64().unwrap() as u32)),
            Type::Float(FloatKind::F64) => SemValue::Float(FloatBits::F64(v.to_u64().unwrap())),
            _ => SemValue::Ptr(v),
        })
    }

    fn store(&mut self, ty: TypeId, p: &SemValue, v: &SemValue) -> Result<(), Stop> {
        let size = self.m.types().size_of(ty);
        let a = self.addr(p, size)?;
        let bits = match v {
            SemValue::Int { bits, .. } | SemValue::Ptr(bits) => bits.clone(),
            SemValue::Float(FloatBits::F32(b)) => Int::from_u64(u64::from(*b)),
            SemValue::Float(FloatBits::F64(b)) => Int::from_u64(*b),
            SemValue::Float(FloatBits::F16(b)) => Int::from_u64(u64::from(*b)),
            SemValue::Poison => Int::ZERO,
            SemValue::Vector(_) => panic!("vectors are not modeled here (use ir::refexec)"),
        };
        for k in 0..size as usize {
            let byte = bits.div_2k_trunc(8 * k as u32).mod_2k(8).to_u64().unwrap() as u8;
            self.mem[a + k] = byte;
        }
        Ok(())
    }

    /// The host functions the tests import (the node harness has the same).
    fn host(&mut self, name: &str, args: &[SemValue]) -> Result<Option<SemValue>, Stop> {
        let u = |i: usize| bits_u64(&args[i]).unwrap_or(0);
        let f64_of = |i: usize| match &args[i] {
            SemValue::Float(FloatBits::F64(b)) => f64::from_bits(*b),
            SemValue::Float(FloatBits::F32(b)) => f64::from(f32::from_bits(*b)),
            _ => 0.0,
        };
        Ok(Some(match name {
            "fmod" => SemValue::Float(FloatBits::F64((f64_of(0) % f64_of(1)).to_bits())),
            "fmodf" => SemValue::Float(FloatBits::F32(((f64_of(0) % f64_of(1)) as f32).to_bits())),
            "host_mul3" => SemValue::int(32, Int::from_u64((u(0) as u32).wrapping_mul(3).wrapping_add(1).into())),
            "host_i64" => SemValue::int(64, Int::from_u64(u(0).wrapping_mul(3).wrapping_add(1))),
            // Returns garbage above an i8: the callee side must mask it.
            "host_sloppy8" => SemValue::int(8, Int::from_u64(0x1234 + 0x700)),
            "host_half" => SemValue::Float(FloatBits::F64((f64_of(0) * 0.5).to_bits())),
            "host_void" => return Ok(None),
            other => panic!("no host function {other}"),
        }))
    }
}
