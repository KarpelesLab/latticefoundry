//! A small whole-function executor built on the reference semantics
//! ([`crate::ir::semantics::eval`]), for tests (compiled only under `cfg(test)`).
//!
//! It runs a function of a [`Module`] on concrete [`SemValue`] arguments with a
//! byte-addressed memory (each byte defined or poison), so a whole program —
//! block arguments, `alloca`/`load`/`store`, globals, direct calls — can be
//! evaluated by the *same* per-opcode semantics the optimizer and backends are
//! checked against. Every value-producing opcode goes through `eval`; only the
//! stateful ones (`alloca`, `load`, `store`, `call`) and control flow are
//! interpreted here. Undefined behavior (as `eval` defines it, plus a branch on
//! poison, `unreachable`, an access through a poison or unallocated address)
//! stops execution with an error, as does anything unsupported (atomics,
//! `syscall`, a call to a declaration).
//!
//! Memory layout matches `TypeContext::layout`: integers at their store size,
//! little-endian; floats as IEEE bits; a vector's lanes packed at the element
//! size (an `i1` lane is one byte holding 0 or 1); pointers as 8 bytes.

use std::collections::HashMap;

use puremp::Int;

use crate::ir::inst::InstKind;
use crate::ir::semantics::{EvalOutcome, SemValue, eval};
use crate::ir::types::{FloatKind, Type, TypeId};
use crate::ir::value::{Const, ConstId, FloatBits, ValueDef, ValueId};
use crate::ir::{BlockId, FuncId, Module};

/// Why execution stopped without a result.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum ExecError {
    /// The program has undefined behavior (the message says where).
    Ub(String),
    /// The executor does not model something the program uses.
    Unsupported(String),
}

/// The executor state: memory plus a bump allocator.
pub(crate) struct Exec<'m> {
    module: &'m Module,
    mem: HashMap<u64, Option<u8>>,
    next: u64,
    globals: Vec<u64>,
    steps: u64,
}

impl<'m> Exec<'m> {
    /// A fresh executor with every global allocated and initialized.
    pub(crate) fn new(module: &'m Module) -> Exec<'m> {
        let mut ex = Exec { module, mem: HashMap::new(), next: 0x1_0000, globals: Vec::new(), steps: 0 };
        let globals: Vec<_> = module.globals().map(|g| (g.ty, g.init)).collect();
        for (ty, init) in globals {
            let addr = ex.alloc(ty);
            if let Some(c) = init {
                ex.write_const(addr, c);
            }
            ex.globals.push(addr);
        }
        ex
    }

    fn alloc(&mut self, ty: TypeId) -> u64 {
        let l = self.module.types().layout(ty);
        let align = l.align.max(8);
        let addr = self.next.div_ceil(align) * align;
        self.next = addr + l.size.max(1) + 16;
        for i in 0..l.size {
            self.mem.insert(addr + i, None);
        }
        addr
    }

    fn write_const(&mut self, addr: u64, c: ConstId) {
        let types = self.module.types();
        match self.module.consts().get(c).clone() {
            Const::Aggregate { ty, elems } => match types.get(ty).clone() {
                Type::Vector(elem, _) => {
                    let sz = types.size_of(elem);
                    for (i, e) in elems.into_iter().enumerate() {
                        self.write_const(addr + sz * i as u64, e);
                    }
                }
                Type::Array(elem, _) => {
                    let st = types.stride(elem);
                    for (i, e) in elems.into_iter().enumerate() {
                        self.write_const(addr + st * i as u64, e);
                    }
                }
                Type::Struct(_) => {
                    for (i, e) in elems.into_iter().enumerate() {
                        let (off, _) = types.field_offset(ty, i as u32);
                        self.write_const(addr + off, e);
                    }
                }
                _ => {}
            },
            other => {
                let ty = other.type_id();
                let v = self.const_value(c);
                self.store(addr, ty, &v);
            }
        }
    }

    /// The semantic value of a constant operand.
    fn const_value(&self, c: ConstId) -> SemValue {
        let types = self.module.types();
        match self.module.consts().get(c) {
            Const::Int { ty, value } => SemValue::int(types.bit_width(*ty).unwrap_or(64), value.clone()),
            Const::Float { bits, .. } => SemValue::Float(*bits),
            Const::Null(_) => SemValue::ptr(Int::ZERO),
            Const::Poison(_) => SemValue::Poison,
            Const::Aggregate { elems, .. } => SemValue::Vector(elems.iter().map(|&e| self.const_value(e)).collect()),
            Const::Addr { .. } => SemValue::Poison,
        }
    }

    /// Store `v` of type `ty` at `addr` (poison bytes for poison lanes).
    fn store(&mut self, addr: u64, ty: TypeId, v: &SemValue) {
        let types = self.module.types();
        if let Some((elem, n)) = types.vector_parts(ty) {
            let sz = types.size_of(elem);
            for i in 0..n as usize {
                self.store(addr + sz * i as u64, elem, &v.lane(i));
            }
            return;
        }
        let size = types.size_of(ty);
        let raw: Option<u64> = match v {
            SemValue::Int { bits, .. } => bits.to_u64(),
            SemValue::Float(FloatBits::F16(b)) => Some(u64::from(*b)),
            SemValue::Float(FloatBits::F32(b)) => Some(u64::from(*b)),
            SemValue::Float(FloatBits::F64(b)) => Some(*b),
            SemValue::Ptr(a) => a.to_u64(),
            _ => None,
        };
        for i in 0..size {
            self.mem.insert(addr + i, raw.map(|r| if i < 8 { (r >> (8 * i)) as u8 } else { 0 }));
        }
    }

    /// Load a value of type `ty` from `addr` (a poison byte poisons its lane).
    fn load(&self, addr: u64, ty: TypeId) -> Result<SemValue, ExecError> {
        let types = self.module.types();
        if let Some((elem, n)) = types.vector_parts(ty) {
            let sz = types.size_of(elem);
            let lanes = (0..n as u64).map(|i| self.load(addr + sz * i, elem)).collect::<Result<_, _>>()?;
            return Ok(SemValue::Vector(lanes));
        }
        let size = types.size_of(ty);
        let mut raw = 0u64;
        for i in 0..size {
            match self.mem.get(&(addr + i)) {
                None => return Err(ExecError::Ub(format!("load from unallocated address {:#x}", addr + i))),
                Some(None) => return Ok(SemValue::Poison),
                Some(Some(b)) => {
                    if i < 8 {
                        raw |= u64::from(*b) << (8 * i);
                    }
                }
            }
        }
        Ok(match types.get(ty) {
            Type::Int(w) => SemValue::int(*w, Int::from_u64(raw)),
            Type::Float(FloatKind::F16) => SemValue::Float(FloatBits::F16(raw as u16)),
            Type::Float(FloatKind::F32) => SemValue::Float(FloatBits::F32(raw as u32)),
            Type::Float(FloatKind::F64) => SemValue::Float(FloatBits::F64(raw)),
            Type::Ptr => SemValue::ptr(Int::from_u64(raw)),
            _ => return Err(ExecError::Unsupported("load of an aggregate".into())),
        })
    }

    fn addr_of(v: &SemValue) -> Result<u64, ExecError> {
        match v {
            SemValue::Ptr(a) => a.to_u64().ok_or_else(|| ExecError::Ub("wild pointer".into())),
            _ => Err(ExecError::Ub("access through a poison or non-pointer address".into())),
        }
    }

    /// Run function `f` on `args`, returning its result (`None` for `void`).
    pub(crate) fn call(&mut self, f: FuncId, args: &[SemValue]) -> Result<Option<SemValue>, ExecError> {
        let module = self.module;
        let func = module.function(f);
        let Some(entry) = func.entry() else {
            return Err(ExecError::Unsupported("call to a declaration".into()));
        };
        let mut env: Vec<Option<SemValue>> = vec![None; func.value_count()];
        let mut block = entry;
        let mut incoming: Vec<SemValue> = args.to_vec();
        loop {
            for (i, &p) in func.block(block).params().iter().enumerate() {
                env[p.index()] = incoming.get(i).cloned();
            }
            let get = |env: &Vec<Option<SemValue>>, ex: &Exec<'_>, v: ValueId| -> SemValue {
                match &func.value(v).def {
                    ValueDef::Const(c) => ex.const_value(*c),
                    ValueDef::Global(g) => SemValue::ptr(Int::from_u64(ex.globals[g.index()])),
                    ValueDef::Func(fid) => SemValue::ptr(Int::from_u64(0xF000_0000 + fid.index() as u64)),
                    _ => env[v.index()].clone().expect("value defined before use"),
                }
            };
            for &iid in func.block(block).insts() {
                self.steps += 1;
                if self.steps > 5_000_000 {
                    return Err(ExecError::Unsupported("step limit".into()));
                }
                let inst = func.inst(iid);
                let ops: Vec<SemValue> = inst.operands().iter().map(|&o| get(&env, self, o)).collect();
                let result = match &inst.kind {
                    InstKind::Alloca { elem_ty } => Some(SemValue::ptr(Int::from_u64(self.alloc(*elem_ty)))),
                    InstKind::Load { ty, .. } => Some(self.load(Self::addr_of(&ops[0])?, *ty)?),
                    InstKind::Store { ty, .. } => {
                        let a = Self::addr_of(&ops[0])?;
                        if self.load(a, *ty).is_err() {
                            return Err(ExecError::Ub("store to unallocated memory".into()));
                        }
                        self.store(a, *ty, &ops[1]);
                        None
                    }
                    InstKind::Call => {
                        let callee = match &func.value(inst.operands()[0]).def {
                            ValueDef::Func(fid) => *fid,
                            _ => return Err(ExecError::Unsupported("indirect call".into())),
                        };
                        self.call(callee, &ops[1..])?
                    }
                    k if k.has_side_effect() || k.is_atomic() => {
                        return Err(ExecError::Unsupported(format!("{k:?}")));
                    }
                    k => match eval(module.types(), inst.ty, k, &inst.flags, &ops) {
                        EvalOutcome::Value(v) => Some(v),
                        EvalOutcome::UndefinedBehavior => return Err(ExecError::Ub(format!("{k:?}"))),
                    },
                };
                if let (Some(r), Some(v)) = (inst.result(), result) {
                    env[r.index()] = Some(v);
                }
            }
            let t = func.block(block).terminator().expect("terminated block");
            let term = func.inst(t);
            let ops: Vec<SemValue> = term.operands().iter().map(|&o| get(&env, self, o)).collect();
            let truthy = |v: &SemValue| -> Result<bool, ExecError> {
                match v {
                    SemValue::Int { bits, .. } => Ok(!bits.is_zero()),
                    _ => Err(ExecError::Ub("branch on poison".into())),
                }
            };
            let (next, args): (BlockId, Vec<SemValue>) = match &term.kind {
                InstKind::Ret => return Ok(ops.first().cloned()),
                InstKind::Br(b) => (*b, ops),
                InstKind::CondBr { if_true, if_false, true_args, .. } => {
                    let ta = *true_args as usize;
                    if truthy(&ops[0])? {
                        (*if_true, ops[1..1 + ta].to_vec())
                    } else {
                        (*if_false, ops[1 + ta..].to_vec())
                    }
                }
                InstKind::Switch(data) => {
                    let SemValue::Int { width, bits } = &ops[0] else {
                        return Err(ExecError::Ub("switch on poison".into()));
                    };
                    let mut off = 1 + data.default_args as usize;
                    let mut pick = (data.default, ops[1..off].to_vec());
                    for c in &data.cases {
                        let n = c.args as usize;
                        if c.value.mod_2k(*width) == *bits {
                            pick = (c.target, ops[off..off + n].to_vec());
                            break;
                        }
                        off += n;
                    }
                    pick
                }
                _ => return Err(ExecError::Ub("reached unreachable".into())),
            };
            block = next;
            incoming = args;
        }
    }
}

/// Run the function named `name` of `module` on `args` in a fresh executor.
pub(crate) fn run_named(
    module: &Module,
    syms: &crate::support::StrInterner,
    name: &str,
    args: &[SemValue],
) -> Result<Option<SemValue>, ExecError> {
    let idx = module
        .functions()
        .position(|f| syms.resolve(f.name) == name)
        .unwrap_or_else(|| panic!("no function @{name}"));
    Exec::new(module).call(FuncId::from_index(idx), args)
}
