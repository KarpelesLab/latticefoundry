//! Tests for the RISC-V context runtime ([`super::runtime`]): the encodings
//! against `llvm-mc`, and the routines' behavior on a small RV64 simulator of
//! exactly the instructions the runtime uses.

use super::runtime::{assemble, layout};
use crate::target::rt_words::{WordAsm, llvm_mc_text};

#[test]
fn encodings_match_llvm_mc() {
    let mut a = assemble();
    let ours = a.finish();
    let Some(theirs) = llvm_mc_text("riscv64", "+d,-relax,-c", &a.listing(), "rv") else {
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
}

// ---------------------------------------------------------------------------
// A tiny RV64 simulator
// ---------------------------------------------------------------------------

const CODE: u64 = 0x1000;
const MEM: usize = 0x40000;
const CTX_A: u64 = 0x10000;
const CTX_B: u64 = 0x11000;
const STACK_TOP: u64 = 0x30008;
const SENTINEL: u64 = 0xDEAD0;
const ENTRY: u64 = 0xBEEF0;

#[derive(Clone, PartialEq, Debug)]
struct Cpu {
    x: [u64; 32],
    f: [u64; 32],
    fcsr: u32,
    pc: u64,
}

#[derive(PartialEq, Debug)]
enum Stop {
    At(u64),
    Ecall,
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
        Sim { cpu: Cpu { x: [0; 32], f: [0; 32], fcsr: 0, pc: 0 }, mem, syms }
    }
    fn sym(&self, name: &str) -> u64 {
        self.syms.iter().find(|s| s.0 == name).expect("symbol").1
    }
    fn rd(&self, a: u64, n: usize) -> u64 {
        let mut b = [0u8; 8];
        b[..n].copy_from_slice(&self.mem[a as usize..a as usize + n]);
        u64::from_le_bytes(b)
    }
    fn wr(&mut self, a: u64, n: usize, v: u64) {
        self.mem[a as usize..a as usize + n].copy_from_slice(&v.to_le_bytes()[..n]);
    }
    fn set(&mut self, r: u32, v: u64) {
        if r != 0 {
            self.cpu.x[r as usize] = v;
        }
    }
    fn call(&mut self, name: &str) -> Stop {
        self.cpu.x[1] = SENTINEL;
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
            let (op, rd, f3, rs1, rs2) = (w & 0x7F, (w >> 7) & 31, (w >> 12) & 7, (w >> 15) & 31, (w >> 20) & 31);
            let imm_i = (w as i32 >> 20) as i64 as u64;
            let imm_s = (((w as i32 >> 25) << 5) | ((w >> 7) & 31) as i32) as i64 as u64;
            let a = self.cpu.x[rs1 as usize];
            let mut next = pc + 4;
            match (op, f3) {
                (0x03, 2) => {
                    let v = self.rd(a.wrapping_add(imm_i), 4) as u32 as i32 as i64 as u64;
                    self.set(rd, v);
                }
                (0x03, 3) => {
                    let v = self.rd(a.wrapping_add(imm_i), 8);
                    self.set(rd, v);
                }
                (0x07, 3) => self.cpu.f[rd as usize] = self.rd(a.wrapping_add(imm_i), 8),
                (0x23, 2) => {
                    let v = self.cpu.x[rs2 as usize];
                    self.wr(a.wrapping_add(imm_s), 4, v);
                }
                (0x23, 3) => {
                    let v = self.cpu.x[rs2 as usize];
                    self.wr(a.wrapping_add(imm_s), 8, v);
                }
                (0x27, 3) => {
                    let v = self.cpu.f[rs2 as usize];
                    self.wr(a.wrapping_add(imm_s), 8, v);
                }
                (0x13, 0) => self.set(rd, a.wrapping_add(imm_i)),
                (0x13, 7) => self.set(rd, a & imm_i),
                (0x17, _) => self.set(rd, pc.wrapping_add((w & 0xFFFF_F000) as i32 as i64 as u64)),
                (0x67, 0) => {
                    next = a.wrapping_add(imm_i) & !1;
                    self.set(rd, pc + 4);
                }
                (0x6F, _) => {
                    let i = ((w >> 31) & 1) << 20 | ((w >> 21) & 0x3FF) << 1 | ((w >> 20) & 1) << 11 | ((w >> 12) & 0xFF) << 12;
                    next = pc.wrapping_add(((i << 11) as i32 >> 11) as i64 as u64);
                    self.set(rd, pc + 4);
                }
                (0x63, 1) => {
                    if a != self.cpu.x[rs2 as usize] {
                        let i = ((w >> 31) & 1) << 12 | ((w >> 25) & 0x3F) << 5 | ((w >> 8) & 0xF) << 1 | ((w >> 7) & 1) << 11;
                        next = pc.wrapping_add(((i << 19) as i32 >> 19) as i64 as u64);
                    }
                }
                (0x73, 0) if w == 0x73 => return Stop::Ecall,
                (0x73, 2) if w >> 20 == 3 => {
                    let v = u64::from(self.cpu.fcsr);
                    self.set(rd, v);
                }
                (0x73, 1) if w >> 20 == 3 => {
                    let old = u64::from(self.cpu.fcsr);
                    self.cpu.fcsr = a as u32 & 0xFF;
                    self.set(rd, old);
                }
                _ => panic!("unsimulated word {w:#010x} at {pc:#x}"),
            }
            self.cpu.pc = next;
        }
        panic!("runaway simulation");
    }
    fn ctx(&self, ctx: u64, off: usize) -> u64 {
        self.rd(ctx + off as u64, 8)
    }
}

fn fill(cpu: &mut Cpu, seed: u64) {
    for (i, x) in cpu.x.iter_mut().enumerate().skip(1) {
        *x = seed.rotate_left(i as u32 * 3) ^ (i as u64 * 0x0101_0101);
    }
    cpu.x[2] = 0x2E000 - (seed & 0xF0);
    cpu.fcsr = (seed as u32 & 0x7) << 5 | 0x3;
    for (i, f) in cpu.f.iter_mut().enumerate() {
        *f = seed.rotate_left(i as u32 * 7) ^ 0x3FF0_0000_0000_0000;
    }
}

#[test]
fn full_save_then_restore_round_trips_everything_but_t6() {
    let mut s = Sim::new();
    fill(&mut s.cpu, 0x1234_5678_9ABC_DEF0);
    s.cpu.x[10] = CTX_A;
    let before = s.cpu.clone();
    assert_eq!(s.call("lf_ctx_save_full"), Stop::At(SENTINEL));
    assert_eq!(s.cpu.x[10], 0, "first return is 0");
    assert_eq!(s.ctx(CTX_A, layout::x(10)), 1, "resumed return value");
    for r in 2..32 {
        if r == 10 {
            continue;
        }
        assert_eq!(s.ctx(CTX_A, layout::x(r)), before.x[r as usize], "x{r}");
    }
    assert_eq!(s.ctx(CTX_A, layout::x(1)), SENTINEL);
    assert_eq!(s.ctx(CTX_A, layout::PC), SENTINEL);
    for fr in 0..32 {
        assert_eq!(s.ctx(CTX_A, layout::f(fr)), before.f[fr as usize], "f{fr}");
    }
    assert_eq!(s.ctx(CTX_A, layout::FCSR) as u32, before.fcsr);
    assert_eq!(s.ctx(CTX_A, layout::VERSION_OFF) as u32, layout::VERSION);
    assert_eq!((s.ctx(CTX_A, layout::VERSION_OFF) >> 32) as u32, layout::KIND_FULL);

    fill(&mut s.cpu, 0x0F0F_0F0F_5555_AAAA);
    let (gp, tp) = (s.cpu.x[3], s.cpu.x[4]);
    s.cpu.x[10] = CTX_A;
    s.cpu.pc = s.sym("lf_ctx_restore");
    assert_eq!(s.run(), Stop::At(SENTINEL));
    let mut expect = before.clone();
    expect.x[10] = 1;
    expect.x[1] = SENTINEL;
    expect.x[31] = SENTINEL; // t6 carried the resume address
    expect.x[3] = gp; // gp and tp are never restored
    expect.x[4] = tp;
    expect.x[5] = s.cpu.x[5]; // t0: scratch after it was saved
    expect.pc = SENTINEL;
    assert_eq!(s.cpu, expect);
    assert_eq!(s.cpu.x[5], before.x[5], "t0 restored too");
}

#[test]
fn cooperative_switch_starts_a_fresh_thread_and_comes_back() {
    let mut s = Sim::new();
    s.cpu.x[10] = CTX_B;
    s.cpu.x[11] = STACK_TOP;
    s.cpu.x[12] = ENTRY;
    s.cpu.x[13] = 77;
    assert_eq!(s.call("lf_ctx_init"), Stop::At(SENTINEL));

    fill(&mut s.cpu, 0xAAAA_0000_1111_2222);
    let a_state = s.cpu.clone();
    s.cpu.x[10] = CTX_A;
    s.cpu.x[11] = CTX_B;
    assert_eq!(s.call("lf_ctx_switch"), Stop::At(ENTRY));
    assert_eq!(s.cpu.x[10], 77, "entry gets its argument");
    assert_eq!(s.cpu.x[2], STACK_TOP & !15, "on its own 16-aligned stack");
    let stub_ret = s.cpu.x[1];

    fill(&mut s.cpu, 0xBBBB_3333_4444_5555);
    s.cpu.x[10] = CTX_B;
    s.cpu.x[11] = CTX_A;
    s.cpu.x[1] = 0xB0B0;
    s.cpu.x[2] = 0x2F800;
    s.cpu.pc = s.sym("lf_ctx_switch");
    assert_eq!(s.run(), Stop::At(SENTINEL));
    for r in [2usize, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] {
        assert_eq!(s.cpu.x[r], a_state.x[r], "x{r}");
    }
    for fr in [8usize, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] {
        assert_eq!(s.cpu.f[fr], a_state.f[fr], "f{fr}");
    }
    assert_eq!(s.cpu.fcsr, a_state.fcsr);
    assert_eq!(s.cpu.x[10], 1);

    s.cpu.x[10] = CTX_A;
    s.cpu.x[11] = CTX_B;
    assert_eq!(s.call("lf_ctx_switch"), Stop::At(0xB0B0));
    assert_eq!(s.cpu.x[2], 0x2F800);
    s.cpu.x[10] = 42;
    s.cpu.pc = stub_ret;
    assert_eq!(s.run(), Stop::Ecall);
    assert_eq!((s.cpu.x[17], s.cpu.x[10]), (94, 42), "exit_group(42)");
}

#[test]
fn save_returns_twice() {
    let mut s = Sim::new();
    fill(&mut s.cpu, 7);
    s.cpu.x[10] = CTX_A;
    let before = s.cpu.clone();
    assert_eq!(s.call("lf_ctx_save"), Stop::At(SENTINEL));
    assert_eq!(s.cpu.x[10], 0);
    fill(&mut s.cpu, 99);
    s.cpu.x[10] = CTX_A;
    s.cpu.pc = s.sym("lf_ctx_restore");
    assert_eq!(s.run(), Stop::At(SENTINEL));
    assert_eq!(s.cpu.x[10], 1);
    for r in [2usize, 8, 9, 18, 27] {
        assert_eq!(s.cpu.x[r], before.x[r]);
    }
}
