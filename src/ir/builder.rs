//! The IR builder.
//!
//! [`FunctionBuilder`] is the front door for constructing a function body. It
//! owns an insertion point (the current block), creates blocks with typed
//! parameter lists, appends instructions (returning their result [`ValueId`]),
//! sets terminators with their per-edge block-argument lists, and maintains the
//! **use/def edges** so that [`FunctionBuilder::replace_all_uses_with`] (RAUW)
//! and later rewrites are cheap and consistent.
//!
//! It also provides the typed addressing helpers the opaque-pointer design
//! relies on — [`FunctionBuilder::struct_field`] and
//! [`FunctionBuilder::array_elem`] — which compute byte offsets from the module
//! data layout and lower to `ptr_add` (`docs/ir-design.md` §6).

use crate::ir::inst::{
    BinOp, CastOp, Flags, FloatPred, InstData, InstId, InstKind, IntPred, SwitchCase, SwitchData,
    UnaryOp, Use,
};
use crate::ir::types::{Type, TypeContext, TypeId};
use crate::ir::value::{Const, ConstPool, FloatBits, ValueDef, ValueId};
use crate::ir::{BlockId, Block, FuncId, Function, GlobalId};

/// A builder with an insertion point over one [`Function`].
///
/// Obtain one via [`crate::ir::Module::build`]; it borrows the function together
/// with the module's shared [`TypeContext`] and [`ConstPool`].
#[derive(Debug)]
pub struct FunctionBuilder<'a> {
    func: &'a mut Function,
    types: &'a mut TypeContext,
    consts: &'a mut ConstPool,
    cur: Option<BlockId>,
}

impl<'a> FunctionBuilder<'a> {
    /// Wrap a function and the shared interning tables in a builder.
    pub fn new(
        func: &'a mut Function,
        types: &'a mut TypeContext,
        consts: &'a mut ConstPool,
    ) -> Self {
        Self { func, types, consts, cur: None }
    }

    // --- blocks & insertion point ------------------------------------------

    /// Create a block with the given typed parameter list, returning its id.
    /// Each parameter becomes an SSA value (a block argument).
    pub fn create_block(&mut self, params: &[TypeId]) -> BlockId {
        let block = BlockId::from_index(self.func.blocks.len());
        self.func.blocks.push(Block::default());
        for (idx, &ty) in params.iter().enumerate() {
            let v = self.func.push_value(ValueDef::Param(block, idx as u32), ty);
            self.func.blocks[block.index()].params.push(v);
        }
        block
    }

    /// Create the function's entry block, taking its parameter list from the
    /// function signature, mark it as the entry, and set it as the insertion
    /// point. The entry block's parameters are the function's parameters.
    pub fn create_entry_block(&mut self) -> BlockId {
        let params = match self.types.get(self.func.sig) {
            Type::Func(ft) => ft.params.clone(),
            _ => panic!("function signature is not a Func type"),
        };
        let block = self.create_block(&params);
        self.func.entry = Some(block);
        self.cur = Some(block);
        block
    }

    /// Set the insertion point to `block`.
    pub fn switch_to(&mut self, block: BlockId) {
        self.cur = Some(block);
    }

    /// The current insertion block, if any.
    pub fn current_block(&self) -> Option<BlockId> {
        self.cur
    }

    /// The parameter values of a block.
    pub fn block_params(&self, block: BlockId) -> &[ValueId] {
        self.func.block(block).params()
    }

    /// The `idx`-th parameter value of a block.
    pub fn param(&self, block: BlockId, idx: u32) -> ValueId {
        self.func.block(block).params()[idx as usize]
    }

    /// The type of a value.
    pub fn value_type(&self, v: ValueId) -> TypeId {
        self.func.value_type(v)
    }

    // --- constants & references --------------------------------------------

    /// A constant of an integer type from an arbitrary-precision value.
    pub fn const_int(&mut self, ty: TypeId, value: puremp::Int) -> ValueId {
        let c = self.consts.intern(Const::Int { ty, value });
        self.func.get_or_make_value(ValueDef::Const(c), ty)
    }

    /// A constant of an integer type from an `i64`.
    pub fn const_i64(&mut self, ty: TypeId, value: i64) -> ValueId {
        self.const_int(ty, puremp::Int::from_i64(value))
    }

    /// A boolean (`i1`) constant.
    pub fn const_bool(&mut self, value: bool) -> ValueId {
        let ty = self.types.bool();
        self.const_i64(ty, i64::from(value))
    }

    /// A floating-point constant from its exact IEEE bit pattern.
    pub fn const_float(&mut self, ty: TypeId, bits: FloatBits) -> ValueId {
        let c = self.consts.intern(Const::Float { ty, bits });
        self.func.get_or_make_value(ValueDef::Const(c), ty)
    }

    /// The null pointer constant of a pointer type.
    pub fn null(&mut self, ty: TypeId) -> ValueId {
        let c = self.consts.intern(Const::Null(ty));
        self.func.get_or_make_value(ValueDef::Const(c), ty)
    }

    /// A poison constant of any type.
    pub fn poison(&mut self, ty: TypeId) -> ValueId {
        let c = self.consts.intern(Const::Poison(ty));
        self.func.get_or_make_value(ValueDef::Const(c), ty)
    }

    /// A reference to a function (its address / callable), typed as a pointer.
    pub fn func_ref(&mut self, func: FuncId) -> ValueId {
        let ty = self.types.ptr();
        self.func.get_or_make_value(ValueDef::Func(func), ty)
    }

    /// A reference to a global (its address), typed as a pointer.
    pub fn global_ref(&mut self, global: GlobalId) -> ValueId {
        let ty = self.types.ptr();
        self.func.get_or_make_value(ValueDef::Global(global), ty)
    }

    // --- instruction emission core -----------------------------------------

    /// Append an instruction to the current block, wiring up its result value
    /// (if `result_ty` is `Some`) and its use/def edges. Terminators are routed
    /// to the block's terminator slot; everything else to the instruction list.
    fn emit(
        &mut self,
        kind: InstKind,
        operands: Vec<ValueId>,
        flags: Flags,
        result_ty: Option<TypeId>,
    ) -> Option<ValueId> {
        let block = self.cur.expect("no insertion point set");
        debug_assert!(
            !self.func.block(block).is_terminated(),
            "appending to an already-terminated block",
        );
        let is_term = kind.is_terminator();
        let inst = InstId::from_index(self.func.insts.len());
        let result = result_ty.map(|ty| self.func.push_value(ValueDef::Inst(inst), ty));
        let ty = result_ty.unwrap_or_else(|| self.types.void());

        for (i, &op) in operands.iter().enumerate() {
            self.func.uses[op.index()].push(Use { inst, operand: i as u32 });
        }
        self.func.insts.push(InstData { kind, flags, ty, operands, result });

        let b = &mut self.func.blocks[block.index()];
        if is_term {
            b.terminator = Some(inst);
        } else {
            b.insts.push(inst);
        }
        result
    }

    // --- arithmetic / bitwise / shifts -------------------------------------

    /// A two-operand op whose result shares the left operand's type.
    pub fn bin(&mut self, op: BinOp, lhs: ValueId, rhs: ValueId, flags: Flags) -> ValueId {
        let ty = self.value_type(lhs);
        self.emit(InstKind::Bin(op), vec![lhs, rhs], flags, Some(ty)).expect("bin has a result")
    }

    /// Integer addition.
    pub fn add(&mut self, lhs: ValueId, rhs: ValueId, flags: Flags) -> ValueId {
        self.bin(BinOp::Add, lhs, rhs, flags)
    }

    /// Integer subtraction.
    pub fn sub(&mut self, lhs: ValueId, rhs: ValueId, flags: Flags) -> ValueId {
        self.bin(BinOp::Sub, lhs, rhs, flags)
    }

    /// Integer multiplication.
    pub fn mul(&mut self, lhs: ValueId, rhs: ValueId, flags: Flags) -> ValueId {
        self.bin(BinOp::Mul, lhs, rhs, flags)
    }

    /// Floating-point negation.
    pub fn fneg(&mut self, val: ValueId, flags: Flags) -> ValueId {
        let ty = self.value_type(val);
        self.emit(InstKind::Unary(UnaryOp::FNeg), vec![val], flags, Some(ty))
            .expect("fneg has a result")
    }

    /// Integer comparison; result is `i1`.
    pub fn icmp(&mut self, pred: IntPred, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.types.bool();
        self.emit(InstKind::ICmp(pred), vec![lhs, rhs], Flags::NONE, Some(ty))
            .expect("icmp has a result")
    }

    /// Floating-point comparison; result is `i1`.
    pub fn fcmp(&mut self, pred: FloatPred, lhs: ValueId, rhs: ValueId, flags: Flags) -> ValueId {
        let ty = self.types.bool();
        self.emit(InstKind::FCmp(pred), vec![lhs, rhs], flags, Some(ty))
            .expect("fcmp has a result")
    }

    /// A conversion to `to_ty`.
    pub fn cast(&mut self, op: CastOp, val: ValueId, to_ty: TypeId) -> ValueId {
        self.emit(InstKind::Cast(op), vec![val], Flags::NONE, Some(to_ty)).expect("cast has a result")
    }

    // --- memory ------------------------------------------------------------

    /// Allocate stack storage for one value of `elem_ty`; result is a pointer.
    pub fn alloca(&mut self, elem_ty: TypeId) -> ValueId {
        let ptr = self.types.ptr();
        self.emit(InstKind::Alloca { elem_ty }, Vec::new(), Flags::NONE, Some(ptr))
            .expect("alloca has a result")
    }

    /// Load a value of `ty` from `ptr` with the given alignment.
    pub fn load(&mut self, ty: TypeId, ptr: ValueId, align: u32) -> ValueId {
        self.emit(InstKind::Load { ty, align }, vec![ptr], Flags::NONE, Some(ty))
            .expect("load has a result")
    }

    /// Store `val` (of `ty`) to `ptr` with the given alignment.
    pub fn store(&mut self, ty: TypeId, ptr: ValueId, val: ValueId, align: u32) {
        self.emit(InstKind::Store { ty, align }, vec![ptr, val], Flags::NONE, None);
    }

    /// Displace a pointer by a byte offset; result is a pointer.
    pub fn ptr_add(&mut self, base: ValueId, byte_offset: ValueId, inbounds: bool) -> ValueId {
        let ptr = self.types.ptr();
        self.emit(InstKind::PtrAdd { inbounds }, vec![base, byte_offset], Flags::NONE, Some(ptr))
            .expect("ptr_add has a result")
    }

    // --- addressing helpers (lower to ptr_add) ------------------------------

    /// Address of struct field `field_idx` of `struct_ty` at base pointer
    /// `base`. Computes the field's byte offset from the data layout and emits a
    /// `ptr_add` (in-bounds). Returns the field pointer.
    pub fn struct_field(&mut self, base: ValueId, struct_ty: TypeId, field_idx: u32) -> ValueId {
        let (offset, _field_ty) = self.types.field_offset(struct_ty, field_idx);
        let i64_ = self.types.int(64);
        let off = self.const_i64(i64_, offset as i64);
        self.ptr_add(base, off, true)
    }

    /// Address of element `index` of an array of `elem_ty` at base pointer
    /// `base`. Computes `index * stride(elem_ty)` (with `index` an `i64`) and
    /// emits a `ptr_add` (in-bounds). Returns the element pointer.
    pub fn array_elem(&mut self, base: ValueId, elem_ty: TypeId, index: ValueId) -> ValueId {
        let stride = self.types.stride(elem_ty);
        let i64_ = self.types.int(64);
        let stride_v = self.const_i64(i64_, stride as i64);
        let scaled = self.mul(index, stride_v, Flags::NONE);
        self.ptr_add(base, scaled, true)
    }

    // --- select / freeze / call --------------------------------------------

    /// Ternary select; result shares the type of the selected values.
    pub fn select(&mut self, cond: ValueId, if_true: ValueId, if_false: ValueId) -> ValueId {
        let ty = self.value_type(if_true);
        self.emit(InstKind::Select, vec![cond, if_true, if_false], Flags::NONE, Some(ty))
            .expect("select has a result")
    }

    /// `freeze`: pin a possibly-poison value to a concrete one.
    pub fn freeze(&mut self, val: ValueId) -> ValueId {
        let ty = self.value_type(val);
        self.emit(InstKind::Freeze, vec![val], Flags::NONE, Some(ty)).expect("freeze has a result")
    }

    /// A call to `callee` with `args`, returning a value of `ret_ty` (or `None`
    /// when `ret_ty` is `void`).
    pub fn call(&mut self, callee: ValueId, args: &[ValueId], ret_ty: TypeId) -> Option<ValueId> {
        let mut operands = Vec::with_capacity(1 + args.len());
        operands.push(callee);
        operands.extend_from_slice(args);
        let void = self.types.void();
        let result_ty = if ret_ty == void { None } else { Some(ret_ty) };
        self.emit(InstKind::Call, operands, Flags::NONE, result_ty)
    }

    // --- terminators -------------------------------------------------------

    /// Return an optional value.
    pub fn ret(&mut self, value: Option<ValueId>) {
        let operands = value.into_iter().collect();
        self.emit(InstKind::Ret, operands, Flags::NONE, None);
    }

    /// Unconditional branch to `target`, passing `args` as its block arguments.
    pub fn br(&mut self, target: BlockId, args: &[ValueId]) {
        self.emit(InstKind::Br(target), args.to_vec(), Flags::NONE, None);
    }

    /// Conditional branch on `cond` (an `i1`), passing each edge its own block
    /// arguments.
    pub fn cond_br(
        &mut self,
        cond: ValueId,
        if_true: BlockId,
        true_args: &[ValueId],
        if_false: BlockId,
        false_args: &[ValueId],
    ) {
        let mut operands = Vec::with_capacity(1 + true_args.len() + false_args.len());
        operands.push(cond);
        operands.extend_from_slice(true_args);
        operands.extend_from_slice(false_args);
        let kind = InstKind::CondBr {
            if_true,
            if_false,
            true_args: true_args.len() as u32,
            false_args: false_args.len() as u32,
        };
        self.emit(kind, operands, Flags::NONE, None);
    }

    /// Multi-way branch on `cond`. Each case is `(match value, target, args)`;
    /// `default`/`default_args` is the fall-through edge.
    pub fn switch(
        &mut self,
        cond: ValueId,
        default: BlockId,
        default_args: &[ValueId],
        cases: Vec<(puremp::Int, BlockId, Vec<ValueId>)>,
    ) {
        let mut operands = Vec::new();
        operands.push(cond);
        operands.extend_from_slice(default_args);
        let mut case_data = Vec::with_capacity(cases.len());
        for (value, target, args) in cases {
            case_data.push(SwitchCase { value, target, args: args.len() as u32 });
            operands.extend_from_slice(&args);
        }
        let data =
            SwitchData { default, default_args: default_args.len() as u32, cases: case_data };
        self.emit(InstKind::Switch(Box::new(data)), operands, Flags::NONE, None);
    }

    /// Mark the current block's control flow as unreachable.
    pub fn unreachable(&mut self) {
        self.emit(InstKind::Unreachable, Vec::new(), Flags::NONE, None);
    }

    // --- rewriting ---------------------------------------------------------

    /// Replace every use of `old` with `new`, moving `old`'s use list onto
    /// `new` and rewriting each referencing operand in place (RAUW). After this,
    /// `old` has no uses.
    pub fn replace_all_uses_with(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }
        let uses = std::mem::take(&mut self.func.uses[old.index()]);
        for u in &uses {
            self.func.insts[u.inst.index()].operands[u.operand as usize] = new;
            self.func.uses[new.index()].push(*u);
        }
    }
}
