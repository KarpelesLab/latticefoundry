//! Tests for the code-generation layer: lowering IR to MIR over the abstract
//! virtual target, liveness, linear-scan register allocation with spilling, an
//! end-to-end MIR interpreter, and determinism.

use crate::codegen::interp;
use crate::codegen::mir::{MachineBlock, MachineFunction, Reg, RegClass, VReg};
use crate::codegen::regalloc::{self, Assign};
use crate::codegen::target::MachineTarget;
use crate::codegen::vtarget::{VOp, VirtualTarget};
use crate::ir::inst::{Flags, IntPred};
use crate::ir::{FuncId, Module};
use crate::support::StrInterner;

use puremp::Int;

// --- IR fixtures -----------------------------------------------------------

/// `add_mul(a, b) = (a + b) * a` over `i32`.
fn build_add_mul() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
    let f = m.declare_function(syms.intern("add_mul"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let s = b.add(a, bb, Flags::NONE);
        let p = b.mul(s, a, Flags::NONE);
        b.ret(Some(p));
    }
    (m, f)
}

/// `max(a, b)`: a diamond that passes the larger value as a block argument to
/// the join block, exercising block-argument lowering on both edges.
fn build_diamond_max() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i32t = m.types_mut().int(32);
    let sig = m.types_mut().func(vec![i32t, i32t], i32t, false);
    let f = m.declare_function(syms.intern("max"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let bb = b.param(entry, 1);
        let then_b = b.create_block(&[]);
        let else_b = b.create_block(&[]);
        let join = b.create_block(&[i32t]);

        let cond = b.icmp(IntPred::Sgt, a, bb);
        b.cond_br(cond, then_b, &[], else_b, &[]);

        b.switch_to(then_b);
        b.br(join, &[a]);

        b.switch_to(else_b);
        b.br(join, &[bb]);

        b.switch_to(join);
        let r = b.param(join, 0);
        b.ret(Some(r));
    }
    (m, f)
}

/// `sum(n) = 0 + 1 + ... + (n-1)` over `i64` — a loop with back-edge block args.
fn build_loop_sum() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("sum"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let n = b.param(entry, 0);
        let header = b.create_block(&[i64t, i64t]);
        let body = b.create_block(&[i64t, i64t]);
        let exit = b.create_block(&[i64t]);

        b.switch_to(entry);
        let zero = b.const_i64(i64t, 0);
        b.br(header, &[zero, zero]);

        b.switch_to(header);
        let acc = b.param(header, 0);
        let i = b.param(header, 1);
        let cond = b.icmp(IntPred::Slt, i, n);
        b.cond_br(cond, body, &[acc, i], exit, &[acc]);

        b.switch_to(body);
        let bacc = b.param(body, 0);
        let bi = b.param(body, 1);
        let new_acc = b.add(bacc, bi, Flags::NONE);
        let one = b.const_i64(i64t, 1);
        let new_i = b.add(bi, one, Flags::NONE);
        b.br(header, &[new_acc, new_i]);

        b.switch_to(exit);
        let result = b.param(exit, 0);
        b.ret(Some(result));
    }
    (m, f)
}

/// A caller `caller(x) = callee(x) + callee(x)` and its callee `callee(y) = y*3`.
fn build_call() -> (Module, FuncId, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let callee = m.declare_function(syms.intern("callee"), sig);
    let caller = m.declare_function(syms.intern("caller"), sig);
    {
        let mut b = m.build(callee);
        let entry = b.create_entry_block();
        let y = b.param(entry, 0);
        let three = b.const_i64(i64t, 3);
        let r = b.mul(y, three, Flags::NONE);
        b.ret(Some(r));
    }
    {
        let mut b = m.build(caller);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let cref1 = b.func_ref(callee);
        let c1 = b.call(cref1, &[x], i64t).unwrap();
        let cref2 = b.func_ref(callee);
        let c2 = b.call(cref2, &[x], i64t).unwrap();
        let s = b.add(c1, c2, Flags::NONE);
        b.ret(Some(s));
    }
    (m, callee, caller)
}

/// `roundtrip(x)`: alloca an i64, store x, load it back, return it.
fn build_mem() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("roundtrip"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let slot = b.alloca(i64t);
        b.store(i64t, slot, x, 8);
        let loaded = b.load(i64t, slot, 8);
        b.ret(Some(loaded));
    }
    (m, f)
}

/// `dynroundtrip(x)`: `dyn_alloca` 8 bytes, store `x`, then a *second*
/// `dyn_alloca` with a sentinel store, and finally reload the first slot. A
/// correct implementation returns `x` (the two dynamic regions are distinct).
fn build_dyn_mem() -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("dynroundtrip"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let x = b.param(entry, 0);
        let eight = b.const_i64(i64t, 8);
        let p = b.dyn_alloca(eight, 16);
        b.store(i64t, p, x, 8);
        let q = b.dyn_alloca(eight, 16);
        let sentinel = b.const_i64(i64t, -1);
        b.store(i64t, q, sentinel, 8);
        let loaded = b.load(i64t, p, 8); // must still be x, not the sentinel
        b.ret(Some(loaded));
    }
    (m, f)
}

// --- MIR well-formedness ---------------------------------------------------

/// Assert the structural invariants a well-formed [`MachineFunction`] must hold.
fn assert_well_formed(mf: &MachineFunction, target: &VirtualTarget) {
    let nblocks = mf.num_blocks();
    for bid in mf.block_ids() {
        let block = mf.block(bid);
        assert!(!block.insts.is_empty(), "block has no instructions");
        let last = block.insts.last().unwrap();
        assert!(target.is_terminator(last.opcode), "block does not end in a terminator");
        for inst in &block.insts[..block.insts.len() - 1] {
            assert!(!target.is_terminator(inst.opcode), "terminator in the middle of a block");
        }
        for lbl in last.labels() {
            assert!(lbl.index() < nblocks, "branch label out of range");
        }
        // Within a block, a vreg first defined here is defined before it is used.
        let mut defined = std::collections::HashSet::new();
        for inst in &block.insts {
            for r in inst.uses() {
                if let Reg::Virtual(v) = r
                    && first_def_in_block(block, v)
                {
                    assert!(defined.contains(&v), "use of {v:?} precedes its def in-block");
                }
            }
            for r in inst.defs() {
                if let Reg::Virtual(v) = r {
                    defined.insert(v);
                }
            }
        }
    }
}

/// Whether `v` is defined (not merely used) somewhere in `block`.
fn first_def_in_block(block: &MachineBlock, v: VReg) -> bool {
    block.insts.iter().any(|inst| inst.defs().any(|r| r == Reg::Virtual(v)))
}

/// Assert every register operand in `mf` is physical (no vreg survived regalloc).
fn assert_all_physical(mf: &MachineFunction) {
    for bid in mf.block_ids() {
        for inst in &mf.block(bid).insts {
            for r in inst.defs().chain(inst.uses()) {
                assert!(matches!(r, Reg::Physical(_)), "a virtual register survived allocation");
            }
        }
    }
}

// --- lowering + interpreter tests ------------------------------------------

#[test]
fn lower_arithmetic_is_well_formed_and_runs() {
    let target = VirtualTarget::new();
    let (m, f) = build_add_mul();
    let mf = target.select(&m, f);
    assert_well_formed(&mf, &target);

    let funcs = vec![mf];
    for (a, b) in [(3i64, 4i64), (10, -2), (0, 7)] {
        let expected = ((a as i32).wrapping_add(b as i32)).wrapping_mul(a as i32) as u32;
        let got = interp::run(&target, &funcs, 0, &[Int::from_i64(a), Int::from_i64(b)])
            .unwrap()
            .unwrap();
        assert_eq!(got, Int::from_u64(u64::from(expected)), "add_mul({a},{b})");
    }
}

#[test]
fn lower_diamond_with_block_args() {
    let target = VirtualTarget::new();
    let (m, f) = build_diamond_max();
    let mf = target.select(&m, f);
    assert_well_formed(&mf, &target);
    assert!(mf.num_blocks() > 4, "arg-carrying edges should be split into edge blocks");

    let funcs = vec![mf];
    for (a, b) in [(3i32, 4i32), (9, 2), (-1, -5)] {
        // Present arguments as their 32-bit unsigned patterns (as an ABI would).
        let expected = a.max(b) as u32;
        let ai = Int::from_u64(u64::from(a as u32));
        let bi = Int::from_u64(u64::from(b as u32));
        let got = interp::run(&target, &funcs, 0, &[ai, bi]).unwrap().unwrap();
        assert_eq!(got, Int::from_u64(u64::from(expected)), "max({a},{b})");
    }
}

#[test]
fn lower_loop_runs() {
    let target = VirtualTarget::new();
    let (m, f) = build_loop_sum();
    let mf = target.select(&m, f);
    assert_well_formed(&mf, &target);

    let funcs = vec![mf];
    for n in [0i64, 1, 5, 10] {
        let expected: i64 = (0..n).sum();
        let got = interp::run(&target, &funcs, 0, &[Int::from_i64(n)]).unwrap().unwrap();
        assert_eq!(got, Int::from_i64(expected), "sum({n})");
    }
}

#[test]
fn lower_dyn_alloca_runs() {
    let target = VirtualTarget::new();
    let (m, f) = build_dyn_mem();
    let mf = target.select(&m, f);
    assert_well_formed(&mf, &target);
    let funcs = vec![mf];
    for x in [0i64, 42, 1234567] {
        let got = interp::run(&target, &funcs, 0, &[Int::from_i64(x)]).unwrap().unwrap();
        assert_eq!(got, Int::from_i64(x), "dynroundtrip({x})");
    }
}

#[test]
fn lower_call_runs() {
    let target = VirtualTarget::new();
    let (m, callee, caller) = build_call();
    let mf_callee = target.select(&m, callee);
    let mf_caller = target.select(&m, caller);
    assert_well_formed(&mf_callee, &target);
    assert_well_formed(&mf_caller, &target);

    let funcs = vec![mf_callee, mf_caller];
    for x in [2i64, 7, 100] {
        let expected = x * 3 + x * 3;
        let got = interp::run(&target, &funcs, caller.index(), &[Int::from_i64(x)])
            .unwrap()
            .unwrap();
        assert_eq!(got, Int::from_i64(expected), "caller({x})");
    }
}

#[test]
fn lower_memory_roundtrip_runs() {
    let target = VirtualTarget::new();
    let (m, f) = build_mem();
    let mf = target.select(&m, f);
    assert_well_formed(&mf, &target);

    let funcs = vec![mf];
    for x in [0i64, 42, 1234567] {
        let got = interp::run(&target, &funcs, 0, &[Int::from_i64(x)]).unwrap().unwrap();
        assert_eq!(got, Int::from_i64(x), "roundtrip({x})");
    }
}

// --- liveness --------------------------------------------------------------

#[test]
fn liveness_matches_hand_derivation() {
    // add_mul(a,b): `a` is used by both the add and the mul, so it outlives `b`,
    // which dies at the add.
    let target = VirtualTarget::new();
    let (m, f) = build_add_mul();
    let mf = target.select(&m, f);
    let live = regalloc::compute_liveness(&mf);
    let intervals = regalloc::build_intervals(&mf, &live);

    let entry = mf.entry().unwrap();
    let a_vreg = mf.block(entry).params[0];
    let b_vreg = mf.block(entry).params[1];
    let ia = intervals.get(a_vreg).expect("a is live");
    let ib = intervals.get(b_vreg).expect("b is live");
    assert!(ia.end > ib.end, "a (used by both add and mul) must outlive b");
}

// --- register allocation ---------------------------------------------------

/// No two vregs with overlapping intervals share a physical register.
fn assert_interference_free(mf: &MachineFunction, alloc: &regalloc::Allocation) {
    let colored: Vec<(VReg, regalloc::Interval, crate::codegen::mir::PReg)> = (0..mf.num_vregs())
        .filter_map(|i| {
            let v = VReg::from_index(i);
            match (alloc.intervals.get(v), alloc.assign[i]) {
                (Some(iv), Some(Assign::Reg(p))) => Some((v, iv, p)),
                _ => None,
            }
        })
        .collect();
    for (a_i, &(va, ia, pa)) in colored.iter().enumerate() {
        for &(vb, ib, pb) in &colored[a_i + 1..] {
            if ia.overlaps(ib) {
                assert_ne!(pa, pb, "overlapping vregs {va:?} and {vb:?} share {pa:?}");
            }
        }
    }
}

#[test]
fn allocate_simple_function() {
    let target = VirtualTarget::new();
    let (m, f) = build_add_mul();
    let mut mf = target.select(&m, f);
    let alloc = regalloc::allocate(&mut mf, &target);
    assert_all_physical(&mf);
    assert_interference_free(&mf, &alloc);
    assert_well_formed(&mf, &target);
    assert_eq!(alloc.spills, 0, "a tiny function should not spill");

    let funcs = vec![mf];
    let got = interp::run(&target, &funcs, 0, &[Int::from_i64(3), Int::from_i64(4)])
        .unwrap()
        .unwrap();
    assert_eq!(got, Int::from_i64((3 + 4) * 3));
}

#[test]
fn allocate_loop_preserves_behavior() {
    let target = VirtualTarget::new();
    let (m, f) = build_loop_sum();
    let mut mf = target.select(&m, f);
    let alloc = regalloc::allocate(&mut mf, &target);
    assert_all_physical(&mf);
    assert_interference_free(&mf, &alloc);
    assert_well_formed(&mf, &target);

    let funcs = vec![mf];
    for n in [0i64, 3, 10] {
        let expected: i64 = (0..n).sum();
        let got = interp::run(&target, &funcs, 0, &[Int::from_i64(n)]).unwrap().unwrap();
        assert_eq!(got, Int::from_i64(expected));
    }
}

/// Many simultaneously-live values, to force spilling.
fn build_high_pressure(n: usize) -> (Module, FuncId) {
    let mut syms = StrInterner::new();
    let mut m = Module::new("t");
    let i64t = m.types_mut().int(64);
    let sig = m.types_mut().func(vec![i64t], i64t, false);
    let f = m.declare_function(syms.intern("pressure"), sig);
    {
        let mut b = m.build(f);
        let entry = b.create_entry_block();
        let a = b.param(entry, 0);
        let mut vals = Vec::new();
        for k in 0..n {
            let c = b.const_i64(i64t, k as i64);
            vals.push(b.add(a, c, Flags::NONE));
        }
        let mut acc = vals[0];
        for &v in &vals[1..] {
            acc = b.add(acc, v, Flags::NONE);
        }
        b.ret(Some(acc));
    }
    (m, f)
}

#[test]
fn high_pressure_triggers_spills_and_stays_correct() {
    let target = VirtualTarget::new();
    // 20 simultaneously-live temporaries against 13 allocatable GPRs -> spills.
    let (m, f) = build_high_pressure(20);
    let mut mf = target.select(&m, f);
    let alloc = regalloc::allocate(&mut mf, &target);
    assert!(alloc.spills > 0, "high register pressure must spill");
    assert_all_physical(&mf);
    assert_interference_free(&mf, &alloc);
    assert_well_formed(&mf, &target);

    let funcs = vec![mf];
    for a in [0i64, 5, 1000] {
        let expected = 20 * a + (0..20i64).sum::<i64>();
        let got = interp::run(&target, &funcs, 0, &[Int::from_i64(a)]).unwrap().unwrap();
        assert_eq!(got, Int::from_i64(expected), "pressure({a})");
    }
}

// --- determinism -----------------------------------------------------------

#[test]
fn lowering_and_allocation_are_deterministic() {
    let target = VirtualTarget::new();

    let mir_dump = |with_alloc: bool| -> Vec<String> {
        let (m, f) = build_loop_sum();
        let mut mf = target.select(&m, f);
        if with_alloc {
            regalloc::allocate(&mut mf, &target);
        }
        dump(&mf)
    };

    assert_eq!(mir_dump(false), mir_dump(false), "isel is deterministic");
    assert_eq!(mir_dump(true), mir_dump(true), "isel + regalloc is deterministic");
}

/// A stable textual rendering of a machine function for equality comparison.
fn dump(mf: &MachineFunction) -> Vec<String> {
    let mut out = Vec::new();
    for bid in mf.block_ids() {
        out.push(format!("block {}:", bid.index()));
        for inst in &mf.block(bid).insts {
            out.push(format!("  {:?} {:?}", VOp::decode(inst.opcode), inst.operands));
        }
    }
    out
}

// --- sanity on the register file -------------------------------------------

#[test]
fn register_file_shape() {
    let target = VirtualTarget::new();
    assert_eq!(target.allocatable(RegClass::Gpr).len(), 13);
    assert_eq!(target.scratch(RegClass::Gpr).len(), 3);
    assert_eq!(target.caller_saved().len(), 8);
    assert_eq!(target.callee_saved().len(), 5);
    assert_eq!(target.call_conv().arg_regs.len(), 4);
}
