//! **Bulk-memory legalization** (`docs/ir-design.md` §6k): an IR→IR rewrite,
//! run by every backend before instruction selection (through
//! [`legalize_vectors`](crate::codegen::legalize::legalize_vectors)), that
//! turns `memcpy` / `memmove` / `memset` into what the target selects.
//!
//! A target describes itself with a [`BulkMemoryLowering`]:
//!
//! - A **constant length** whose expansion takes at most
//!   [`max_inline`](BulkMemoryLowering::max_inline) accesses each way becomes
//!   straight-line loads and stores: the widest chunk that fits the remaining
//!   bytes (16 bytes as a `<16 x i8>` vector where the target has one, else the
//!   machine word, then halving down to a byte), never wider than the known
//!   alignment unless the target allows unaligned accesses. A `memset` stores
//!   the byte replicated across each chunk (a constant, or `zext` + shifts +
//!   ors of a variable byte, or a vector `splat`). A `memmove` loads every
//!   chunk before storing any, so overlap cannot matter; a `memcpy` interleaves
//!   them.
//! - Anything else stays a bulk-memory op when the target selects them itself
//!   ([`native`](BulkMemoryLowering::native): `rep movsb`/`rep stosb` on
//!   x86-64, `memory.copy`/`memory.fill` on wasm32), with its length converted
//!   to the pointer-sized integer the selector expects.
//! - Otherwise it becomes a **loop**: a `memcpy`/`memset` walks the bytes in
//!   word-sized chunks (as wide as alignment allows) and finishes the tail
//!   with byte accesses (straight-line when the length is constant); a
//!   `memmove` picks a forward byte loop when `dst <= src` and a backward one
//!   otherwise (always forward when the pointers are in different address
//!   spaces, which cannot overlap).
//!
//! A volatile op is expanded with volatile scalar accesses (every byte is
//! still touched exactly once). Every rewrite refines the op's reference
//! meaning ([`crate::ir::exec_bulk_memory`]): `n == 0` touches nothing, the
//! bytes moved are the same (poison bytes included, as the chunk loads and
//! stores carry per-byte poison), and the tests check it by running programs
//! before and after with the reference executor.
//!
//! [`bulk_memory_libcalls`] is the hosted alternative for what would stay
//! native: calls to the C library's `memcpy`, `memmove` and `memset`.

use puremp::Int;

use crate::analysis::cfg::{ControlFlowGraph, Dominators};
use crate::ir::builder::FunctionBuilder;
use crate::ir::inst::{BinOp, CastOp, Flags, InstKind, IntPred};
use crate::ir::types::TypeId;
use crate::ir::value::{Const, ValueDef, ValueId};
use crate::ir::{BlockId, DataLayout, FuncId, Function, InstId, Module};
use crate::support::StrInterner;
use crate::transform::{dom_preorder, rebuild_terminator, remap_value};

/// How a target lowers the bulk-memory ops (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BulkMemoryLowering {
    /// The widest scalar integer access, in bytes (a power of two, at most 8).
    pub word: u32,
    /// Whether 16-byte `<16 x i8>` vector loads and stores (and an `i8`
    /// `splat`) may be used for non-volatile ops.
    pub vector16: bool,
    /// Whether an access may be wider than the alignment known for it.
    pub unaligned: bool,
    /// The most accesses (each way) a constant-length op is expanded into
    /// inline; longer ones loop or stay native.
    pub max_inline: u32,
    /// Whether the target selects the ops it is left with.
    pub native: bool,
    /// For a native target: whether an op with a non-constant length must be
    /// skipped when the length is zero (wasm's `memory.copy` traps on an
    /// out-of-bounds pointer even then, while the IR op touches nothing).
    pub guard_zero: bool,
}

impl BulkMemoryLowering {
    /// The lowering every target gets unless it says otherwise: pointer-sized
    /// (at most 8-byte) aligned chunks, up to 8 accesses inline, loops beyond.
    pub fn portable(layout: &DataLayout) -> BulkMemoryLowering {
        let word = (layout.pointer_bits(0) / 8).clamp(1, 8).next_power_of_two();
        BulkMemoryLowering { word, vector16: false, unaligned: false, max_inline: 8, native: false, guard_zero: false }
    }
}

/// Whether any function of `module` uses a bulk-memory op.
pub fn uses_bulk_memory(module: &Module) -> bool {
    module.functions().any(|f| (0..f.inst_count()).any(|i| f.inst(InstId::from_index(i)).kind.is_bulk_memory()))
}

/// Rewrite every bulk-memory op of `module` for `lowering` (see the module
/// docs). Ops a native target keeps are only normalized (pointer-sized
/// length).
pub fn legalize_bulk_memory(module: &mut Module, lowering: &BulkMemoryLowering) {
    for i in 0..module.function_count() {
        let id = FuncId::from_index(i);
        let f = module.function(id);
        if f.is_declaration() || !body_needs(module, f, lowering) {
            continue;
        }
        let decl_line = f.decl_line;
        let (mut fresh, ()) = module.map_function(id, |old, b| rebuild(old, b, lowering));
        fresh.decl_line = decl_line;
        module.replace_function(id, fresh);
    }
}

/// Whether some bulk op of `f` needs rewriting.
fn body_needs(module: &Module, f: &Function, lowering: &BulkMemoryLowering) -> bool {
    let pbits = module.data_layout().pointer_bits(0);
    (0..f.inst_count()).any(|i| {
        let inst = f.inst(InstId::from_index(i));
        inst.kind.is_bulk_memory()
            && (!lowering.native
                || const_len(module, f, inst.operands()[2]).is_some_and(|n| n == 0 || plan(lowering, &inst.kind, n).is_some())
                || module.types().bit_width(f.value_type(inst.operands()[2])) != Some(pbits)
                || lowering.guard_zero)
    })
}

/// The constant value of `v`, if it is an integer constant.
fn const_len(module: &Module, f: &Function, v: ValueId) -> Option<u64> {
    match f.value(v).def {
        ValueDef::Const(c) => match module.consts().get(c) {
            Const::Int { value, .. } => value.to_u64(),
            _ => None,
        },
        _ => None,
    }
}

/// The straight-line chunks `(offset, size)` covering `[start, start + n)`
/// with alignment `align` at offset 0, or `None` when there are more than
/// `max` of them.
fn chunks(lowering: &BulkMemoryLowering, start: u64, n: u64, align: u32, vector: bool, max: u32) -> Option<Vec<(u64, u32)>> {
    let mut out = Vec::new();
    let mut off = start;
    let end = start + n;
    while off < end {
        let known = align_at(align, off);
        let mut size = if vector { 16 } else { lowering.word };
        while size > 1 && (u64::from(size) > end - off || (!lowering.unaligned && known < size)) {
            size /= 2;
        }
        if out.len() as u32 >= max {
            return None;
        }
        out.push((off, size));
        off += u64::from(size);
    }
    Some(out)
}

/// The alignment known for byte `off` of a range aligned to `align`.
fn align_at(align: u32, off: u64) -> u32 {
    let align = align.max(1);
    if off == 0 { align } else { align.min(1u32 << off.trailing_zeros().min(31)) }
}

/// The inline expansion of a constant-length op, if short enough.
fn plan(lowering: &BulkMemoryLowering, kind: &InstKind, n: u64) -> Option<Vec<(u64, u32)>> {
    let (align, volatile) = match kind {
        InstKind::MemCopy { align, volatile, .. } | InstKind::MemSet { align, volatile } => (*align, *volatile),
        _ => return None,
    };
    let vector = lowering.vector16 && !volatile;
    chunks(lowering, 0, n, align, vector, lowering.max_inline)
}

/// Rebuild `old` with its bulk-memory ops rewritten.
fn rebuild(old: &Function, b: &mut FunctionBuilder<'_>, lowering: &BulkMemoryLowering) {
    rebuild_with(old, b, |b, kind, ops| Expander { b, lowering }.expand(kind, ops[0], ops[1], ops[2]));
}

/// Emits the expansion of one op at the builder's insertion point, leaving it
/// in the block that continues after the op.
struct Expander<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    lowering: &'a BulkMemoryLowering,
}

/// Which bulk op is being expanded.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Copy,
    Move,
    Set,
}

impl Expander<'_, '_> {
    fn expand(&mut self, kind: &InstKind, dst: ValueId, mid: ValueId, n: ValueId) {
        let (op, align, volatile) = match *kind {
            InstKind::MemCopy { align, volatile, overlapping } => {
                (if overlapping { Op::Move } else { Op::Copy }, align, volatile)
            }
            InstKind::MemSet { align, volatile } => (Op::Set, align, volatile),
            _ => unreachable!("not a bulk-memory op"),
        };
        let pbits = self.b.types().data_layout().pointer_bits(0);
        let pty = self.b.types_mut().int(pbits);
        let cn = self.const_u64(n);
        if cn == Some(0) {
            return;
        }
        if let Some(cn) = cn
            && let Some(plan) = plan(self.lowering, kind, cn)
        {
            self.straight(op, dst, mid, &plan, align, volatile);
            return;
        }
        let np = int_to(self.b, n, pty);
        if self.lowering.native {
            let kind = kind.clone();
            if self.lowering.guard_zero && cn.is_none() {
                let zero = self.b.const_int(pty, Int::ZERO);
                let nz = self.b.icmp(IntPred::Ne, np, zero);
                let (yes, done) = (self.b.create_block(&[]), self.b.create_block(&[]));
                self.b.cond_br(nz, yes, &[], done, &[]);
                self.b.switch_to(yes);
                self.b.bulk_memory(kind, dst, mid, np);
                self.b.br(done, &[]);
                self.b.switch_to(done);
            } else {
                self.b.bulk_memory(kind, dst, mid, np);
            }
            return;
        }
        let dst = self.as_ptr(dst);
        match op {
            Op::Move => {
                let src = self.as_ptr(mid);
                self.memmove_loops(dst, src, np, pty, volatile);
            }
            Op::Copy | Op::Set => {
                let mid = if op == Op::Copy { self.as_ptr(mid) } else { mid };
                // The widest chunk the alignment (or the target) allows.
                let mut w = self.lowering.word;
                if !self.lowering.unaligned {
                    w = w.min(align.max(1));
                }
                let mut values = SplatCache::default();
                match cn {
                    Some(cn) => {
                        let wl = u64::from(w);
                        let main = cn / wl * wl;
                        let lim = self.b.const_int(pty, Int::from_u64(main));
                        let start = self.b.const_int(pty, Int::ZERO);
                        self.forward_loop(op, dst, mid, start, lim, w, align.min(w), volatile, pty, &mut values);
                        let tail = chunks(self.lowering, main, cn - main, align, false, u32::MAX)
                            .expect("an unbounded plan always exists");
                        self.straight(op, dst, mid, &tail, align, volatile);
                    }
                    None => {
                        let start = self.b.const_int(pty, Int::ZERO);
                        let lim = if w > 1 {
                            let mask = self.b.const_int(pty, Int::from_u64(u64::MAX << w.trailing_zeros()).mod_2k(pbits));
                            self.b.bin(BinOp::And, np, mask, Flags::NONE)
                        } else {
                            np
                        };
                        self.forward_loop(op, dst, mid, start, lim, w, align.min(w), volatile, pty, &mut values);
                        if w > 1 {
                            self.forward_loop(op, dst, mid, lim, np, 1, 1, volatile, pty, &mut values);
                        }
                    }
                }
            }
        }
    }

    /// Straight-line accesses for `plan` (offsets from the op's pointers).
    fn straight(&mut self, op: Op, dst: ValueId, mid: ValueId, plan: &[(u64, u32)], align: u32, volatile: bool) {
        let pbits = self.b.types().data_layout().pointer_bits(0);
        let pty = self.b.types_mut().int(pbits);
        let mut values = SplatCache::default();
        let mut loaded: Vec<(ValueId, TypeId, u64, u32)> = Vec::new();
        for &(off, size) in plan {
            let ty = self.chunk_ty(size);
            let a = align_at(align, off).min(size);
            let offv = self.b.const_int(pty, Int::from_u64(off));
            match op {
                Op::Set => {
                    let v = values.get(self.b, mid, ty, size);
                    let d = self.at(dst, offv, off);
                    self.store(ty, d, v, a, volatile);
                }
                Op::Copy | Op::Move => {
                    let s = self.at(mid, offv, off);
                    let v = self.load(ty, s, a, volatile);
                    if op == Op::Copy {
                        let d = self.at(dst, offv, off);
                        self.store(ty, d, v, a, volatile);
                    } else {
                        loaded.push((v, ty, off, a));
                    }
                }
            }
        }
        for (v, ty, off, a) in loaded {
            let offv = self.b.const_int(pty, Int::from_u64(off));
            let d = self.at(dst, offv, off);
            self.store(ty, d, v, a, volatile);
        }
    }

    /// `for (o = start; o < lim; o += w) chunk(o)`, with `w`-byte accesses of
    /// alignment `a`.
    #[allow(clippy::too_many_arguments)]
    fn forward_loop(
        &mut self,
        op: Op,
        dst: ValueId,
        mid: ValueId,
        start: ValueId,
        lim: ValueId,
        w: u32,
        a: u32,
        volatile: bool,
        pty: TypeId,
        values: &mut SplatCache,
    ) {
        if let (Some(s), Some(l)) = (self.const_u64(start), self.const_u64(lim))
            && s >= l
        {
            return;
        }
        let ty = self.chunk_ty(w);
        // The fill value is computed once, before the loop.
        let fill = (op == Op::Set).then(|| values.get(self.b, mid, ty, w));
        let head = self.b.create_block(&[pty]);
        let body = self.b.create_block(&[]);
        let exit = self.b.create_block(&[]);
        self.b.br(head, &[start]);
        self.b.switch_to(head);
        let o = self.b.param(head, 0);
        let more = self.b.icmp(IntPred::Ult, o, lim);
        self.b.cond_br(more, body, &[], exit, &[]);
        self.b.switch_to(body);
        let d = self.b.ptr_add(dst, o, false);
        let v = match fill {
            Some(v) => v,
            None => {
                let s = self.b.ptr_add(mid, o, false);
                self.load(ty, s, a, volatile)
            }
        };
        self.store(ty, d, v, a, volatile);
        let step = self.b.const_int(pty, Int::from_u64(u64::from(w)));
        let next = self.b.add(o, step, Flags::NONE);
        self.b.br(head, &[next]);
        self.b.switch_to(exit);
    }

    /// A `memmove` as byte loops: forward when `dst <= src`, else backward.
    fn memmove_loops(&mut self, dst: ValueId, src: ValueId, n: ValueId, pty: TypeId, volatile: bool) {
        let i8t = self.b.types_mut().int(8);
        let zero = self.b.const_int(pty, Int::ZERO);
        let one = self.b.const_int(pty, Int::ONE);
        let same_space = self.b.value_type(dst) == self.b.value_type(src);
        let exit = self.b.create_block(&[]);
        let fwd_head = self.b.create_block(&[pty]);
        if same_space {
            let back_head = self.b.create_block(&[pty]);
            let down = self.b.icmp(IntPred::Ugt, dst, src);
            self.b.cond_br(down, back_head, &[n], fwd_head, &[zero]);
            // Backward: for (k = n; k != 0; ) { k -= 1; dst[k] = src[k]; }
            self.b.switch_to(back_head);
            let k = self.b.param(back_head, 0);
            let more = self.b.icmp(IntPred::Ne, k, zero);
            let body = self.b.create_block(&[]);
            self.b.cond_br(more, body, &[], exit, &[]);
            self.b.switch_to(body);
            let k1 = self.b.sub(k, one, Flags::NONE);
            let s = self.b.ptr_add(src, k1, false);
            let v = self.load(i8t, s, 1, volatile);
            let d = self.b.ptr_add(dst, k1, false);
            self.store(i8t, d, v, 1, volatile);
            self.b.br(back_head, &[k1]);
        } else {
            self.b.br(fwd_head, &[zero]);
        }
        self.b.switch_to(fwd_head);
        let o = self.b.param(fwd_head, 0);
        let more = self.b.icmp(IntPred::Ult, o, n);
        let body = self.b.create_block(&[]);
        self.b.cond_br(more, body, &[], exit, &[]);
        self.b.switch_to(body);
        let s = self.b.ptr_add(src, o, false);
        let v = self.load(i8t, s, 1, volatile);
        let d = self.b.ptr_add(dst, o, false);
        self.store(i8t, d, v, 1, volatile);
        let o1 = self.b.add(o, one, Flags::NONE);
        self.b.br(fwd_head, &[o1]);
        self.b.switch_to(exit);
    }

    /// The type of a `size`-byte chunk.
    fn chunk_ty(&mut self, size: u32) -> TypeId {
        if size == 16 && self.lowering.vector16 {
            let i8t = self.b.types_mut().int(8);
            self.b.types_mut().vector(i8t, 16)
        } else {
            self.b.types_mut().int(size * 8)
        }
    }

    /// `base + off` (the base itself at offset 0).
    fn at(&mut self, base: ValueId, offv: ValueId, off: u64) -> ValueId {
        if off == 0 { base } else { self.b.ptr_add(base, offv, false) }
    }

    fn load(&mut self, ty: TypeId, p: ValueId, align: u32, volatile: bool) -> ValueId {
        if volatile { self.b.load_volatile(ty, p, align) } else { self.b.load(ty, p, align) }
    }

    fn store(&mut self, ty: TypeId, p: ValueId, v: ValueId, align: u32, volatile: bool) {
        if volatile {
            self.b.store_volatile(ty, p, v, align);
        } else {
            self.b.store(ty, p, v, align);
        }
    }

    /// An address operand as a pointer value (an aggregate value denotes its
    /// address; `ptr_add 0` turns it into a plain `ptr`).
    fn as_ptr(&mut self, v: ValueId) -> ValueId {
        let ty = self.b.value_type(v);
        if self.b.types().is_ptr(ty) {
            return v;
        }
        let pbits = self.b.types().data_layout().pointer_bits(0);
        let pty = self.b.types_mut().int(pbits);
        let zero = self.b.const_int(pty, Int::ZERO);
        self.b.ptr_add(v, zero, false)
    }

    fn const_u64(&self, v: ValueId) -> Option<u64> {
        let c = self.b.const_of(v)?;
        match self.b.consts().get(c) {
            Const::Int { value, .. } => value.to_u64(),
            _ => None,
        }
    }
}

/// `v` (an integer) converted to `ty` (zero-extended or truncated: a length
/// that does not fit the address space cannot be valid anyway).
fn int_to(b: &mut FunctionBuilder<'_>, v: ValueId, ty: TypeId) -> ValueId {
    let from = b.value_type(v);
    let (fw, tw) = (b.types().bit_width(from).unwrap_or(0), b.types().bit_width(ty).unwrap_or(0));
    if fw != tw
        && let Some(c) = b.const_of(v)
        && let Const::Int { value, .. } = b.consts().get(c)
    {
        let value = value.mod_2k(tw);
        return b.const_int(ty, value);
    }
    match fw.cmp(&tw) {
        std::cmp::Ordering::Equal => v,
        std::cmp::Ordering::Less => b.cast(CastOp::ZExt, v, ty),
        std::cmp::Ordering::Greater => b.cast(CastOp::Trunc, v, ty),
    }
}

/// The `memset` byte replicated across each chunk type, built once per type.
#[derive(Default)]
struct SplatCache {
    made: Vec<(TypeId, ValueId)>,
}

impl SplatCache {
    fn get(&mut self, b: &mut FunctionBuilder<'_>, byte: ValueId, ty: TypeId, size: u32) -> ValueId {
        if let Some(&(_, v)) = self.made.iter().find(|&&(t, _)| t == ty) {
            return v;
        }
        let v = if b.types().is_vector(ty) {
            b.splat(byte, 16)
        } else if size == 1 {
            byte
        } else {
            let known = b.const_of(byte).and_then(|c| match b.consts().get(c) {
                Const::Int { value, .. } => value.to_u64(),
                _ => None,
            });
            match known {
                Some(k) => {
                    let mut x = Int::ZERO;
                    for _ in 0..size {
                        x = x.mul_2k(8).add(&Int::from_u64(k & 0xff));
                    }
                    b.const_int(ty, x)
                }
                None => {
                    let mut v = b.cast(CastOp::ZExt, byte, ty);
                    let mut shift = 8;
                    while shift < size * 8 {
                        let amt = b.const_int(ty, Int::from_u64(u64::from(shift)));
                        let hi = b.bin(BinOp::Shl, v, amt, Flags::NONE);
                        v = b.bin(BinOp::Or, v, hi, Flags::NONE);
                        shift *= 2;
                    }
                    v
                }
            }
        };
        self.made.push((ty, v));
        v
    }
}

/// The C library functions [`bulk_memory_libcalls`] calls.
const LIBC: [&str; 3] = ["memcpy", "memmove", "memset"];

/// Replace every bulk-memory op left in `module` with a call to the C
/// library's `memcpy`, `memmove` or `memset` (declared on demand, with the
/// length as the pointer-sized integer and the `memset` byte widened to
/// `i32`), skipping a constant zero length. For hosted code that links a
/// libc; run after [`legalize_bulk_memory`] so short constant ops stay
/// inline.
pub fn bulk_memory_libcalls(module: &mut Module, names: &mut StrInterner) {
    if !uses_bulk_memory(module) {
        return;
    }
    let pbits = module.data_layout().pointer_bits(0);
    let ptr = module.types_mut().ptr();
    let pty = module.types_mut().int(pbits);
    let i32t = module.types_mut().int(32);
    let mut callee = [None; 3];
    for (k, name) in LIBC.iter().enumerate() {
        let sym = names.intern(name);
        let mid = if k == 2 { i32t } else { ptr };
        callee[k] = Some(module.function_by_name(sym).unwrap_or_else(|| {
            let sig = module.types_mut().func(vec![ptr, mid, pty], ptr, false);
            module.declare_function(sym, sig)
        }));
    }
    for i in 0..module.function_count() {
        let id = FuncId::from_index(i);
        let f = module.function(id);
        if f.is_declaration() || !(0..f.inst_count()).any(|k| f.inst(InstId::from_index(k)).kind.is_bulk_memory()) {
            continue;
        }
        let decl_line = f.decl_line;
        let (mut fresh, ()) = module.map_function(id, |old, b| {
            let lowering = Libcalls { callee: callee.map(|c| c.expect("declared")), pty, i32t, ptr };
            rebuild_with(old, b, |b, kind, ops| lowering.emit(b, kind, ops));
        });
        fresh.decl_line = decl_line;
        module.replace_function(id, fresh);
    }
}

/// The callees and types of [`bulk_memory_libcalls`].
struct Libcalls {
    callee: [FuncId; 3],
    pty: TypeId,
    i32t: TypeId,
    ptr: TypeId,
}

impl Libcalls {
    fn emit(&self, b: &mut FunctionBuilder<'_>, kind: &InstKind, ops: &[ValueId]) {
        let k = match kind {
            InstKind::MemCopy { overlapping: false, .. } => 0,
            InstKind::MemCopy { overlapping: true, .. } => 1,
            _ => 2,
        };
        if let Some(c) = b.const_of(ops[2])
            && matches!(b.consts().get(c), Const::Int { value, .. } if value.is_zero())
        {
            return;
        }
        let n = int_to(b, ops[2], self.pty);
        let mid = if k == 2 { b.cast(CastOp::ZExt, ops[1], self.i32t) } else { ops[1] };
        let f = b.func_ref(self.callee[k]);
        b.call(f, &[ops[0], mid, n], self.ptr);
    }
}

/// Rebuild `old`, handing each bulk-memory op (with remapped operands) to
/// `emit`, which may split the current block: the rest of the old block
/// continues wherever it leaves the builder.
fn rebuild_with(
    old: &Function,
    b: &mut FunctionBuilder<'_>,
    mut emit: impl FnMut(&mut FunctionBuilder<'_>, &InstKind, &[ValueId]),
) {
    if let Some(l) = old.decl_line {
        b.set_decl_line(l);
    }
    let n = old.block_count();
    let entry = old.entry().expect("a definition has an entry").index();
    let cfg = ControlFlowGraph::new(old);
    let doms = Dominators::new(old, &cfg);
    let mut new_block: Vec<BlockId> = Vec::with_capacity(n);
    for bi in 0..n {
        if bi == entry {
            new_block.push(b.create_entry_block());
        } else {
            let tys: Vec<TypeId> = old.block(BlockId::from_index(bi)).params().iter().map(|&p| old.value_type(p)).collect();
            new_block.push(b.create_block(&tys));
        }
    }
    let mut vmap: Vec<Option<ValueId>> = vec![None; old.value_count()];
    for (bi, &nb) in new_block.iter().enumerate() {
        let np = b.block_params(nb).to_vec();
        for (k, &p) in old.block(BlockId::from_index(bi)).params().iter().enumerate() {
            vmap[p.index()] = Some(np[k]);
        }
    }
    for bi in dom_preorder(old, &doms) {
        let bb = BlockId::from_index(bi);
        b.switch_to(new_block[bi]);
        let insts = old.block(bb).insts();
        // The entry block's static allocas stay ahead of any split: the frame
        // layout expects them in the entry block.
        let order: Vec<InstId> = if bi == entry {
            let (allocas, rest): (Vec<InstId>, Vec<InstId>) =
                insts.iter().partition(|&&i| matches!(old.inst(i).kind, InstKind::Alloca { .. }));
            allocas.into_iter().chain(rest).collect()
        } else {
            insts.to_vec()
        };
        for i in order {
            b.set_line(old.inst_line(i).unwrap_or(0));
            let inst = old.inst(i);
            let ops: Vec<ValueId> = inst.operands().iter().map(|&o| remap_value(&mut vmap, old, b, o)).collect();
            if inst.kind.is_bulk_memory() {
                emit(b, &inst.kind, &ops);
                continue;
            }
            let result_ty = inst.result().map(|_| inst.ty);
            let nr = b.append_inst(inst.kind.clone(), ops, inst.flags, result_ty);
            if let Some(r) = inst.result() {
                vmap[r.index()] = nr;
            }
        }
        if let Some(t) = old.block(bb).terminator() {
            b.set_line(old.inst_line(t).unwrap_or(0));
        }
        rebuild_terminator(&mut vmap, old, b, &new_block, bb, |_, _, _| {});
    }
}

#[cfg(test)]
mod tests;
