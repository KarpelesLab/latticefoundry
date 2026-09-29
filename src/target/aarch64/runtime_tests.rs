//! Tests for the AArch64 context runtime ([`super::runtime`]): the encodings
//! against `llvm-mc`, and the routines' behavior on a small A64 simulator of
//! exactly the instructions the runtime uses (this host cannot run A64 code).

use super::runtime::{assemble, layout};
use crate::target::rt_words::{WordAsm, llvm_mc_text};

#[test]
fn encodings_match_llvm_mc() {
    let mut a = assemble();
    let ours = a.finish();
    let Some(theirs) = llvm_mc_text("aarch64", "", &a.listing(), "a64") else {
        return; // llvm-mc not installed
    };
    assert_eq!(ours.len(), theirs.len(), "size differs:\n{}", a.listing());
    for (i, (x, y)) in ours.chunks(4).zip(theirs.chunks(4)).enumerate() {
        assert_eq!(x, y, "word {i} differs: ours {x:02x?} llvm-mc {y:02x?}\n{}", a.listing());
    }
}

#[test]
fn object_defines_the_routines() {
    let obj = super::runtime::context_runtime_object();
    for name in ["lf_ctx_save", "lf_ctx_save_full", "lf_ctx_restore", "lf_ctx_switch", "lf_ctx_switch_full", "lf_ctx_init"] {
        let id = obj.symbol_id(name).expect("defined");
        assert!(obj.symbol(id).size > 0, "{name}");
    }
    assert!(obj.relocations().is_empty());
    assert_eq!(layout::SIZE % layout::ALIGN, 0);
    assert_eq!(layout::v(0) % 16, 0);
}

// ---------------------------------------------------------------------------
// A tiny A64 simulator
// ---------------------------------------------------------------------------

const CODE: u64 = 0x1000;
const MEM: usize = 0x40000;
const CTX_A: u64 = 0x10000;
const CTX_B: u64 = 0x11000;
const STACK_TOP: u64 = 0x30008; // deliberately misaligned: init rounds down
/// A return address outside the code: reaching it ends a run.
const SENTINEL: u64 = 0xDEAD0;
const ENTRY: u64 = 0xBEEF0;

#[derive(Clone, PartialEq, Debug)]
struct Cpu {
    x: [u64; 31],
    sp: u64,
    pc: u64,
    nzcv: u64,
    fpsr: u64,
    fpcr: u64,
    v: [u128; 32],
}

#[derive(PartialEq, Debug)]
enum Stop {
    At(u64),
    Svc,
}

struct Sim {
    cpu: Cpu,
    mem: Vec<u8>,
    syms: Vec<(&'static str, u64)>,
}

impl Sim {
    fn new() -> Sim {
        let mut a: WordAsm = assemble();
        let bytes = a.finish();
        let mut mem = vec![0u8; MEM];
        mem[CODE as usize..CODE as usize + bytes.len()].copy_from_slice(&bytes);
        let syms = a.routines.iter().map(|r| (r.name, CODE + r.start as u64)).collect();
        let cpu = Cpu { x: [0; 31], sp: 0x2F000, pc: 0, nzcv: 0, fpsr: 0, fpcr: 0, v: [0; 32] };
        Sim { cpu, mem, syms }
    }
    fn sym(&self, name: &str) -> u64 {
        self.syms.iter().find(|s| s.0 == name).expect("symbol").1
    }
    fn rd(&self, a: u64, n: usize) -> u128 {
        let mut b = [0u8; 16];
        b[..n].copy_from_slice(&self.mem[a as usize..a as usize + n]);
        u128::from_le_bytes(b)
    }
    fn wr(&mut self, a: u64, n: usize, v: u128) {
        self.mem[a as usize..a as usize + n].copy_from_slice(&v.to_le_bytes()[..n]);
    }
    fn get(&self, r: u32) -> u64 {
        if r == 31 { 0 } else { self.cpu.x[r as usize] }
    }
    fn set(&mut self, r: u32, v: u64) {
        if r != 31 {
            self.cpu.x[r as usize] = v;
        }
    }
    fn base(&self, r: u32) -> u64 {
        if r == 31 { self.cpu.sp } else { self.cpu.x[r as usize] }
    }
    /// Call routine `name` (lr = SENTINEL) and run until a stop.
    fn call(&mut self, name: &str) -> Stop {
        self.cpu.x[30] = SENTINEL;
        self.cpu.pc = self.sym(name);
        self.run()
    }
    fn run(&mut self) -> Stop {
        for _ in 0..10_000 {
            let pc = self.cpu.pc;
            if !(CODE..CODE + 0x4000).contains(&pc) {
                return Stop::At(pc);
            }
            let w = self.rd(pc, 4) as u32;
            let (rt, rn, rt2) = (w & 31, (w >> 5) & 31, (w >> 10) & 31);
            let simm7 = (((w >> 15) & 0x7F) as i32) << 25 >> 25;
            let uimm12 = u64::from((w >> 10) & 0xFFF);
            let mut next = pc + 4;
            match w {
                _ if w & 0xFFC0_0000 == 0xA900_0000 || w & 0xFFC0_0000 == 0xA940_0000 => {
                    let ad = self.base(rn).wrapping_add((simm7 * 8) as u64);
                    if w & 0x0040_0000 != 0 {
                        let (a, b) = (self.rd(ad, 8) as u64, self.rd(ad + 8, 8) as u64);
                        self.set(rt, a);
                        self.set(rt2, b);
                    } else {
                        let (a, b) = (self.get(rt), self.get(rt2));
                        self.wr(ad, 8, a.into());
                        self.wr(ad + 8, 8, b.into());
                    }
                }
                _ if w & 0xFFC0_0000 == 0xA880_0000 => {
                    let ad = self.base(rn);
                    let (a, b) = (self.get(rt), self.get(rt2));
                    self.wr(ad, 8, a.into());
                    self.wr(ad + 8, 8, b.into());
                    self.cpu.x[rn as usize] = ad.wrapping_add((simm7 * 8) as u64);
                }
                _ if w & 0xFFC0_0000 == 0xAD00_0000 || w & 0xFFC0_0000 == 0xAD40_0000 => {
                    let ad = self.base(rn).wrapping_add((simm7 * 16) as u64);
                    if w & 0x0040_0000 != 0 {
                        self.cpu.v[rt as usize] = self.rd(ad, 16);
                        self.cpu.v[rt2 as usize] = self.rd(ad + 16, 16);
                    } else {
                        let (a, b) = (self.cpu.v[rt as usize], self.cpu.v[rt2 as usize]);
                        self.wr(ad, 16, a);
                        self.wr(ad + 16, 16, b);
                    }
                }
                _ if w & 0xFF80_0000 == 0xF900_0000 => {
                    let ad = self.base(rn) + uimm12 * 8;
                    if w & 0x0040_0000 != 0 {
                        let v = self.rd(ad, 8) as u64;
                        self.set(rt, v);
                    } else {
                        let v = self.get(rt);
                        self.wr(ad, 8, v.into());
                    }
                }
                _ if w & 0xFF80_0000 == 0xB900_0000 => {
                    let ad = self.base(rn) + uimm12 * 4;
                    if w & 0x0040_0000 != 0 {
                        let v = self.rd(ad, 4) as u64;
                        self.set(rt, v);
                    } else {
                        let v = self.get(rt);
                        self.wr(ad, 4, u128::from(v as u32));
                    }
                }
                _ if w & 0xFF80_0000 == 0x9100_0000 => {
                    let v = self.base(rn) + uimm12;
                    if rt == 31 { self.cpu.sp = v } else { self.cpu.x[rt as usize] = v }
                }
                _ if w & 0xFF80_0000 == 0xD280_0000 => self.set(rt, u64::from((w >> 5) & 0xFFFF)),
                _ if w & 0xFF80_0000 == 0x5280_0000 => self.set(rt, u64::from((w >> 5) & 0xFFFF)),
                _ if w & 0xFFE0_FFE0 == 0xAA00_03E0 => {
                    let v = self.get((w >> 16) & 31);
                    self.set(rt, v);
                }
                _ if w & 0xFFF0_0000 == 0xD530_0000 || w & 0xFFF0_0000 == 0xD510_0000 => {
                    let read = w & 0x0020_0000 != 0;
                    let reg = match w & 0x0007_FFE0 {
                        0x3_4200 => &mut self.cpu.nzcv,
                        0x3_4400 => &mut self.cpu.fpcr,
                        0x3_4420 => &mut self.cpu.fpsr,
                        o => panic!("sysreg {o:#x}"),
                    };
                    if read {
                        let v = *reg;
                        self.set(rt, v);
                    } else {
                        *reg = if rt == 31 { 0 } else { self.cpu.x[rt as usize] };
                    }
                }
                _ if w & 0xFFFF_FC1F == 0xD61F_0000 || w & 0xFFFF_FC1F == 0xD65F_0000 => next = self.get(rn),
                _ if w & 0xFFFF_FC1F == 0xD63F_0000 => {
                    next = self.get(rn);
                    self.cpu.x[30] = pc + 4;
                }
                _ if w & 0xFC00_0000 == 0x1400_0000 => {
                    next = pc.wrapping_add(((((w & 0x03FF_FFFF) << 6) as i32 >> 6) * 4) as u64);
                }
                _ if w & 0xFF00_001F == 0x5400_0001 => {
                    if self.cpu.nzcv & (1 << 30) == 0 {
                        next = pc.wrapping_add((((((w >> 5) & 0x7FFFF) << 13) as i32 >> 13) * 4) as u64);
                    }
                }
                _ if w & 0xFF00_0000 == 0x3500_0000 => {
                    if self.get(rt) as u32 != 0 {
                        next = pc.wrapping_add((((((w >> 5) & 0x7FFFF) << 13) as i32 >> 13) * 4) as u64);
                    }
                }
                _ if w & 0x9F00_0000 == 0x1000_0000 => {
                    let imm = ((w >> 29) & 3) | ((w >> 5) & 0x7FFFF) << 2;
                    let imm = ((imm << 11) as i32 >> 11) as i64;
                    self.set(rt, pc.wrapping_add(imm as u64));
                }
                _ if w & 0xFFE0_FC1F == 0xEB00_001F => {
                    let (a, b) = (self.get(rn), self.get((w >> 16) & 31));
                    let z = u64::from(a == b);
                    let c = u64::from(a >= b);
                    self.cpu.nzcv = (((a.wrapping_sub(b) >> 63) & 1) << 31) | (z << 30) | (c << 29);
                }
                0x927C_EC21 => self.cpu.x[1] &= !15,
                0xD400_0001 => return Stop::Svc,
                _ => panic!("unsimulated word {w:#010x} at {pc:#x}"),
            }
            self.cpu.pc = next;
        }
        panic!("runaway simulation");
    }
    fn ctx_u64(&self, ctx: u64, off: usize) -> u64 {
        self.rd(ctx + off as u64, 8) as u64
    }
    fn ctx_u32(&self, ctx: u64, off: usize) -> u32 {
        self.rd(ctx + off as u64, 4) as u32
    }
}

/// A distinctive register state (sp 16-aligned inside the stack area).
fn fill(cpu: &mut Cpu, seed: u64) {
    for (i, x) in cpu.x.iter_mut().enumerate() {
        *x = seed.rotate_left(i as u32 * 3) ^ (i as u64 * 0x0101_0101);
    }
    cpu.sp = 0x2E000 - (seed & 0xF0);
    cpu.nzcv = 0xA000_0000; // N and C
    cpu.fpsr = 0x9;
    cpu.fpcr = 0x0040_0000; // RMode = +inf
    for (i, v) in cpu.v.iter_mut().enumerate() {
        *v = (u128::from(seed) << 64 | u128::from(seed ^ 0xFFFF)) .rotate_left(i as u32 * 5);
    }
}

#[test]
fn full_save_then_restore_round_trips_everything_but_x17() {
    let mut s = Sim::new();
    fill(&mut s.cpu, 0x1234_5678_9ABC_DEF0);
    s.cpu.x[0] = CTX_A;
    let before = s.cpu.clone();
    assert_eq!(s.call("lf_ctx_save_full"), Stop::At(SENTINEL));
    assert_eq!(s.cpu.x[0], 0, "first return is 0");
    // The image holds the caller's state (x0 = 1: the resumed return value).
    assert_eq!(s.ctx_u64(CTX_A, layout::x(0)), 1);
    for r in 1..30 {
        if r == 9 {
            continue; // scratch, saved before use
        }
        assert_eq!(s.ctx_u64(CTX_A, layout::x(r)), before.x[r as usize], "x{r}");
    }
    assert_eq!(s.ctx_u64(CTX_A, layout::x(9)), before.x[9], "x9 saved before it is used");
    assert_eq!(s.ctx_u64(CTX_A, layout::x(30)), SENTINEL);
    assert_eq!(s.ctx_u64(CTX_A, layout::PC), SENTINEL);
    assert_eq!(s.ctx_u64(CTX_A, layout::SP), before.sp);
    assert_eq!(s.ctx_u64(CTX_A, layout::NZCV), before.nzcv);
    assert_eq!(u64::from(s.ctx_u32(CTX_A, layout::FPCR)), before.fpcr);
    assert_eq!(u64::from(s.ctx_u32(CTX_A, layout::FPSR)), before.fpsr);
    assert_eq!(s.ctx_u32(CTX_A, layout::KIND), layout::KIND_FULL);
    assert_eq!(s.ctx_u32(CTX_A, layout::VERSION_OFF), layout::VERSION);
    for q in 0..32 {
        assert_eq!(s.rd(CTX_A + layout::v(q) as u64, 16), before.v[q as usize], "q{q}");
    }

    // Scramble everything and resume: the saved state comes back, x0 = 1.
    fill(&mut s.cpu, 0x0F0F_0F0F_5555_AAAA);
    s.cpu.x[0] = CTX_A;
    s.cpu.pc = s.sym("lf_ctx_restore");
    assert_eq!(s.run(), Stop::At(SENTINEL));
    let mut expect = before.clone();
    expect.x[0] = 1;
    expect.x[30] = SENTINEL;
    expect.x[17] = s.cpu.x[17]; // the branch register
    expect.pc = SENTINEL;
    assert_eq!(s.cpu, expect);
}

#[test]
fn cooperative_switch_starts_a_fresh_thread_and_comes_back() {
    let mut s = Sim::new();
    // lf_ctx_init(CTX_B, STACK_TOP, ENTRY, 77)
    s.cpu.x[0] = CTX_B;
    s.cpu.x[1] = STACK_TOP;
    s.cpu.x[2] = ENTRY;
    s.cpu.x[3] = 77;
    assert_eq!(s.call("lf_ctx_init"), Stop::At(SENTINEL));
    assert_eq!(s.ctx_u32(CTX_B, layout::KIND), layout::KIND_COOP);

    // A: callee-saved registers distinctive, then lf_ctx_switch(A, B).
    fill(&mut s.cpu, 0xAAAA_0000_1111_2222);
    let a_state = s.cpu.clone();
    s.cpu.x[0] = CTX_A;
    s.cpu.x[1] = CTX_B;
    // B starts at the trampoline, which calls ENTRY(77) on its own stack.
    assert_eq!(s.call("lf_ctx_switch"), Stop::At(ENTRY));
    assert_eq!(s.cpu.x[0], 77, "entry gets its argument");
    assert_eq!(s.cpu.sp, STACK_TOP & !15, "on its own 16-aligned stack");
    let stub_ret = s.cpu.x[30];

    // B (inside ENTRY) switches back: lf_ctx_switch(B, A) resumes A after its
    // call with every callee-saved register intact.
    fill(&mut s.cpu, 0xBBBB_3333_4444_5555);
    s.cpu.x[0] = CTX_B;
    s.cpu.x[1] = CTX_A;
    s.cpu.x[30] = 0xB0B0;
    s.cpu.sp = 0x2F800;
    s.cpu.pc = s.sym("lf_ctx_switch");
    assert_eq!(s.run(), Stop::At(SENTINEL));
    for r in 19..=29 {
        assert_eq!(s.cpu.x[r], a_state.x[r], "x{r}");
    }
    assert_eq!(s.cpu.sp, a_state.sp);
    assert_eq!(s.cpu.x[0], 1);
    for q in 8..16 {
        assert_eq!(s.cpu.v[q], a_state.v[q], "q{q}");
    }
    assert_eq!(s.cpu.fpcr, a_state.fpcr);

    // Resume B: it continues at its switch's return address; when ENTRY
    // returns (to the trampoline) the thread exits with ENTRY's result.
    s.cpu.x[0] = CTX_A;
    s.cpu.x[1] = CTX_B;
    assert_eq!(s.call("lf_ctx_switch"), Stop::At(0xB0B0));
    assert_eq!(s.cpu.sp, 0x2F800);
    s.cpu.x[0] = 42;
    s.cpu.pc = stub_ret;
    assert_eq!(s.run(), Stop::Svc);
    assert_eq!((s.cpu.x[8], s.cpu.x[0]), (94, 42), "exit_group(42)");
}

#[test]
fn save_returns_twice() {
    let mut s = Sim::new();
    fill(&mut s.cpu, 7);
    s.cpu.x[0] = CTX_A;
    let before = s.cpu.clone();
    assert_eq!(s.call("lf_ctx_save"), Stop::At(SENTINEL));
    assert_eq!(s.cpu.x[0], 0);
    fill(&mut s.cpu, 99);
    s.cpu.x[0] = CTX_A;
    s.cpu.pc = s.sym("lf_ctx_restore");
    assert_eq!(s.run(), Stop::At(SENTINEL));
    assert_eq!(s.cpu.x[0], 1);
    for r in 19..=29 {
        assert_eq!(s.cpu.x[r], before.x[r]);
    }
    assert_eq!(s.cpu.sp, before.sp);
}
