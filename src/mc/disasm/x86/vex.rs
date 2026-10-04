//! VEX-encoded instructions (SDM Vol. 2, §2.3): the AVX/AVX2 forms of the
//! SSE tables (three-operand, 128- or 256-bit), the AVX2 broadcasts,
//! permutes and lane inserts/extracts, FMA3, and the BMI1/BMI2 general-
//! purpose instructions.
//!
//! An SSE row becomes `v<name>`: a computational row gains the
//! non-destructive source `VEX.vvvv` as its second operand; moves,
//! conversions between equal widths, compares into flags, and the other
//! one-source forms (listed in [`ONE_SOURCE`]) keep two operands and
//! require `vvvv` = 0. `VEX.L` selects `ymm` registers and 256-bit memory
//! for packed rows; scalar rows ignore it. Rows whose vector widths differ
//! between operands (`cvtpd2ps`, ...) are only decoded where unambiguous.

use super::{Decoder, F, K, ModRm, Operand, Rm, SSE, SSE38, SSE3A, Sse, X86Inst, suffix};

/// The VEX fields the operand decoders need.
#[derive(Clone, Copy, Default)]
pub(super) struct Vex {
    /// `VEX.L`: 256-bit vectors.
    pub(super) l: bool,
    /// `VEX.vvvv` (already inverted): the extra register operand.
    pub(super) vvvv: u8,
    /// `VEX.pp` as the implied prefix byte (0, 66, F3, F2).
    pub(super) pfx: u8,
}

const YMM: [&str; 16] = [
    "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
    "ymm14", "ymm15",
];

/// SSE rows that keep two operands (no `vvvv`) in their VEX form.
const ONE_SOURCE: &[&str] = &[
    "movups", "movupd", "movaps", "movapd", "movdqa", "movdqu", "movddup", "movshdup", "movsldup", "movntps",
    "movntpd", "movntdq", "movntdqa", "lddqu", "sqrtps", "sqrtpd", "rsqrtps", "rcpps", "ucomiss", "ucomisd",
    "comiss", "comisd", "cvtdq2ps", "cvtps2dq", "cvttps2dq", "ptest", "pabsb", "pabsw", "pabsd", "phminposuw",
    "pshufd", "pshuflw", "pshufhw", "roundps", "roundpd", "movmskps", "movmskpd", "pmovmskb", "cvttss2si",
    "cvttsd2si", "cvtss2si", "cvtsd2si", "pextrb", "pextrw", "pextrd", "extractps", "pcmpestrm", "pcmpestri",
    "pcmpistrm", "pcmpistri",
];

/// Rows whose source is half the destination's width (`ymm` <- `xmm`).
const WIDENING: &[&str] = &[
    "pmovsxbw", "pmovsxbd", "pmovsxbq", "pmovsxwd", "pmovsxwq", "pmovsxdq", "pmovzxbw", "pmovzxbd", "pmovzxbq",
    "pmovzxwd", "pmovzxwq", "pmovzxdq", "cvtps2pd", "cvtdq2pd",
];

/// Rows that narrow (`xmm` <- `ymm`), or are only valid at 128 bits.
const NO_VEX256: &[&str] =
    &["cvtpd2ps", "cvttpd2dq", "cvtpd2dq", "movlpd", "movhpd", "movlps", "movhps", "insertps", "phminposuw", "dppd", "pcmpestrm", "pcmpestri",
    "pcmpistrm", "pcmpistri"];

/// The 32 VEX compare predicates of `vcmpps`/`vcmppd`/`vcmpss`/`vcmpsd`.
const CMP_PRED: [&str; 32] = [
    "eq", "lt", "le", "unord", "neq", "nlt", "nle", "ord", "eq_uq", "nge", "ngt", "false", "neq_oq", "ge", "gt",
    "true", "eq_os", "lt_oq", "le_oq", "unord_s", "neq_us", "nlt_uq", "nle_uq", "ord_s", "eq_us", "nge_uq", "ngt_uq",
    "false_os", "neq_os", "ge_oq", "gt_oq", "true_us",
];

impl Decoder<'_> {
    /// A VEX instruction starting with `C4` or `C5` (already consumed).
    pub(super) fn vex(&mut self, c: u8) -> Option<X86Inst> {
        if self.p.opsize || self.p.rep != 0 || self.p.rex != 0 || self.p.lock {
            return None; // #UD
        }
        let b1 = self.u8()?;
        let (map, w, r, x, b, b2) = if c == 0xc5 {
            (1, 0, !b1 >> 7 & 1, 0, 0, b1)
        } else {
            let b2 = self.u8()?;
            (b1 & 0x1f, b2 >> 7, !b1 >> 7 & 1, !b1 >> 6 & 1, !b1 >> 5 & 1, b2)
        };
        self.p.rex = 0x40 | w << 3 | r << 2 | x << 1 | b;
        let v = Vex { l: b2 & 4 != 0, vvvv: !b2 >> 3 & 15, pfx: [0, 0x66, 0xf3, 0xf2][usize::from(b2 & 3)] };
        self.vex = Some(v);
        let op = self.u8()?;
        match map {
            1 => self.vex_0f(op, v),
            2 => self.vex_38(op, v),
            3 => self.vex_3a(op, v),
            _ => None,
        }
    }

    /// A vector register of the VEX length.
    fn vr(&self, n: u8, l: bool) -> Operand {
        Operand::Reg(if l { YMM[usize::from(n & 15)] } else { super::XMM[usize::from(n & 15)] })
    }

    /// The r/m operand as a vector register or memory of `bits`.
    fn vrm(&self, m: &ModRm, l: bool, bits: u32) -> Operand {
        match &m.rm {
            Rm::Reg(n) => self.vr(*n, l),
            Rm::Mem(_) => self.rm(m, 0, K::Gpr, bits),
        }
    }

    fn vdone(&self, name: &str, ops: Vec<Operand>) -> Option<X86Inst> {
        Some(self.make(self.pos, name, name, ops))
    }

    fn vex_0f(&mut self, op: u8, v: Vex) -> Option<X86Inst> {
        let pfx = v.pfx;
        let h = self.vr(v.vvvv, v.l);
        match (op, pfx) {
            (0x77, 0) => {
                if v.vvvv != 0 {
                    return None;
                }
                self.vdone(if v.l { "vzeroall" } else { "vzeroupper" }, Vec::new())
            }
            (0x10 | 0x11, 0xf3 | 0xf2) => {
                // vmovss/vmovsd: two operands with memory, three between
                // registers.
                let name = if pfx == 0xf3 { "vmovss" } else { "vmovsd" };
                let bits = if pfx == 0xf3 { 32 } else { 64 };
                let m = self.modrm()?;
                let g = self.vr(m.reg, false);
                let h = self.vr(v.vvvv, false);
                match (&m.rm, op) {
                    (Rm::Mem(_), _) if v.vvvv != 0 => None,
                    (Rm::Mem(_), 0x10) => self.vdone(name, vec![g, self.rm(&m, 0, K::Gpr, bits)]),
                    (Rm::Mem(_), _) => self.vdone(name, vec![self.rm(&m, 0, K::Gpr, bits), g]),
                    (Rm::Reg(n), 0x10) => self.vdone(name, vec![g, h, self.vr(*n, false)]),
                    (Rm::Reg(n), _) => self.vdone(name, vec![self.vr(*n, false), h, g]),
                }
            }
            (0x12 | 0x16, 0) => {
                if v.l {
                    return None;
                }
                let m = self.modrm()?;
                let name = match (matches!(m.rm, Rm::Reg(_)), op == 0x12) {
                    (true, true) => "vmovhlps",
                    (true, false) => "vmovlhps",
                    (false, true) => "vmovlps",
                    (false, false) => "vmovhps",
                };
                let ops = vec![self.vr(m.reg, false), self.vr(v.vvvv, false), self.vrm(&m, false, 64)];
                self.vdone(name, ops)
            }
            (0x2a, 0xf2 | 0xf3) => {
                let m = self.modrm()?;
                let y = self.p.y();
                let name = if pfx == 0xf2 { "vcvtsi2sd" } else { "vcvtsi2ss" };
                let ops = vec![self.vr(m.reg, false), self.vr(v.vvvv, false), self.rm(&m, y, K::Gpr, y)];
                let mut x = self.vdone(name, ops)?;
                if matches!(m.rm, Rm::Mem(_)) {
                    x.att = format!("{name}{}", suffix(y));
                }
                Some(x)
            }
            (0x6e | 0x7e, 0x66) => {
                if v.l || v.vvvv != 0 {
                    return None;
                }
                let m = self.modrm()?;
                let y = self.p.y();
                let name = if y == 64 { "vmovq" } else { "vmovd" };
                let (g, e) = (self.vr(m.reg, false), self.rm(&m, y, K::Gpr, y));
                self.vdone(name, if op == 0x6e { vec![g, e] } else { vec![e, g] })
            }
            (0x7e, 0xf3) | (0xd6, 0x66) => {
                if v.l || v.vvvv != 0 {
                    return None;
                }
                let m = self.modrm()?;
                let (g, e) = (self.vr(m.reg, false), self.vrm(&m, false, 64));
                self.vdone("vmovq", if op == 0x7e { vec![g, e] } else { vec![e, g] })
            }
            (0x71..=0x73, 0x66) => {
                let m = self.modrm()?;
                let Rm::Reg(r) = m.rm else { return None };
                let name = match (op, m.ext) {
                    (0x71, 2) => "vpsrlw",
                    (0x71, 4) => "vpsraw",
                    (0x71, 6) => "vpsllw",
                    (0x72, 2) => "vpsrld",
                    (0x72, 4) => "vpsrad",
                    (0x72, 6) => "vpslld",
                    (0x73, 2) => "vpsrlq",
                    (0x73, 3) => "vpsrldq",
                    (0x73, 6) => "vpsllq",
                    (0x73, 7) => "vpslldq",
                    _ => return None,
                };
                let ops = vec![h, self.vr(r, v.l), self.ib()?];
                self.vdone(name, ops)
            }
            (0xc2, _) => {
                let (sfx, bits, packed) = match pfx {
                    0 => ("ps", 128, true),
                    0x66 => ("pd", 128, true),
                    0xf3 => ("ss", 32, false),
                    _ => ("sd", 64, false),
                };
                let l = v.l && packed;
                let m = self.modrm()?;
                let ops = vec![self.vr(m.reg, l), self.vr(v.vvvv, l), self.vrm(&m, l, if l { 256 } else { bits })];
                let imm = self.u8()?;
                match CMP_PRED.get(usize::from(imm)) {
                    Some(p) => self.vdone(&format!("vcmp{p}{sfx}"), ops),
                    None => {
                        let mut ops = ops;
                        ops.push(Operand::Imm { value: i64::from(imm), signed: false, bits: 8 });
                        self.vdone(&format!("vcmp{sfx}"), ops)
                    }
                }
            }
            (0xc4, 0x66) => {
                if v.l {
                    return None;
                }
                let m = self.modrm()?;
                let ops = vec![self.vr(m.reg, false), self.vr(v.vvvv, false), self.rm(&m, 32, K::Gpr, 16), self.ib()?];
                self.vdone("vpinsrw", ops)
            }
            (0xc5, 0x66) => {
                if v.l || v.vvvv != 0 {
                    return None;
                }
                let m = self.modrm()?;
                let Rm::Reg(r) = m.rm else { return None };
                let ops = vec![self.reg(m.reg, 32, K::Gpr), self.vr(r, false), self.ib()?];
                self.vdone("vpextrw", ops)
            }
            _ => {
                let row = SSE.iter().find(|r| r.op == op && r.pfx == pfx)?;
                self.vex_row(row, v)
            }
        }
    }

    fn vex_38(&mut self, op: u8, v: Vex) -> Option<X86Inst> {
        let w = self.p.w();
        match (op, v.pfx) {
            // AVX2 broadcasts: a register or memory element to every lane.
            (0x18 | 0x19 | 0x58 | 0x59 | 0x78 | 0x79 | 0x5a, 0x66) => {
                let (name, bits) = match op {
                    0x18 => ("vbroadcastss", 32),
                    0x19 => ("vbroadcastsd", 64),
                    0x58 => ("vpbroadcastd", 32),
                    0x59 => ("vpbroadcastq", 64),
                    0x78 => ("vpbroadcastb", 8),
                    0x79 => ("vpbroadcastw", 16),
                    _ => ("vbroadcasti128", 128),
                };
                if v.vvvv != 0 || w || ((op == 0x19 || op == 0x5a) && !v.l) {
                    return None;
                }
                let m = self.modrm()?;
                if op == 0x5a && matches!(m.rm, Rm::Reg(_)) {
                    return None;
                }
                let ops = vec![self.vr(m.reg, v.l), self.vrm(&m, false, bits)];
                self.vdone(name, ops)
            }
            (0x16 | 0x36, 0x66) => {
                if !v.l || w {
                    return None;
                }
                let m = self.modrm()?;
                let ops = vec![self.vr(m.reg, true), self.vr(v.vvvv, true), self.vrm(&m, true, 256)];
                self.vdone(if op == 0x16 { "vpermps" } else { "vpermd" }, ops)
            }
            (0x45..=0x47, 0x66) => {
                let name = match (op, w) {
                    (0x45, false) => "vpsrlvd",
                    (0x45, true) => "vpsrlvq",
                    (0x46, false) => "vpsravd",
                    (0x47, false) => "vpsllvd",
                    (0x47, true) => "vpsllvq",
                    _ => return None,
                };
                let m = self.modrm()?;
                let bits = if v.l { 256 } else { 128 };
                let ops = vec![self.vr(m.reg, v.l), self.vr(v.vvvv, v.l), self.vrm(&m, v.l, bits)];
                self.vdone(name, ops)
            }
            (0x96..=0x9f | 0xa6..=0xaf | 0xb6..=0xbf, 0x66) => self.fma(op, v),
            (0xf2, 0) => {
                // andn r, vvvv, r/m
                let y = self.p.y();
                let m = self.modrm()?;
                if v.l {
                    return None;
                }
                let ops = vec![self.reg(m.reg, y, K::Gpr), self.reg(v.vvvv, y, K::Gpr), self.rm(&m, y, K::Gpr, y)];
                self.bmi("andn", y, ops)
            }
            (0xf3, 0) => {
                let y = self.p.y();
                let m = self.modrm()?;
                let name = match m.ext {
                    1 => "blsr",
                    2 => "blsmsk",
                    3 => "blsi",
                    _ => return None,
                };
                if v.l {
                    return None;
                }
                let ops = vec![self.reg(v.vvvv, y, K::Gpr), self.rm(&m, y, K::Gpr, y)];
                self.bmi(name, y, ops)
            }
            (0xf5..=0xf7, _) => {
                let y = self.p.y();
                if v.l {
                    return None;
                }
                // (name, the vvvv register is the last operand)
                let (name, vvvv_last) = match (op, v.pfx) {
                    (0xf5, 0) => ("bzhi", true),
                    (0xf5, 0xf3) => ("pext", false),
                    (0xf5, 0xf2) => ("pdep", false),
                    (0xf6, 0xf2) => ("mulx", false),
                    (0xf7, 0) => ("bextr", true),
                    (0xf7, 0x66) => ("shlx", true),
                    (0xf7, 0xf3) => ("sarx", true),
                    (0xf7, 0xf2) => ("shrx", true),
                    _ => return None,
                };
                let m = self.modrm()?;
                let (g, b, e) = (self.reg(m.reg, y, K::Gpr), self.reg(v.vvvv, y, K::Gpr), self.rm(&m, y, K::Gpr, y));
                self.bmi(name, y, if vvvv_last { vec![g, e, b] } else { vec![g, b, e] })
            }
            _ => {
                let row = SSE38.iter().find(|r| r.op == op && r.pfx == v.pfx)?;
                self.vex_row(row, v)
            }
        }
    }

    fn vex_3a(&mut self, op: u8, v: Vex) -> Option<X86Inst> {
        let w = self.p.w();
        match (op, v.pfx) {
            (0x00 | 0x01, 0x66) => {
                if !v.l || !w || v.vvvv != 0 {
                    return None;
                }
                let m = self.modrm()?;
                let ops = vec![self.vr(m.reg, true), self.vrm(&m, true, 256), self.ib()?];
                self.vdone(if op == 0 { "vpermq" } else { "vpermpd" }, ops)
            }
            (0x02, 0x66) => {
                if w {
                    return None;
                }
                let m = self.modrm()?;
                let bits = if v.l { 256 } else { 128 };
                let ops = vec![self.vr(m.reg, v.l), self.vr(v.vvvv, v.l), self.vrm(&m, v.l, bits), self.ib()?];
                self.vdone("vpblendd", ops)
            }
            (0x06 | 0x46, 0x66) => {
                if !v.l || w {
                    return None;
                }
                let m = self.modrm()?;
                let ops = vec![self.vr(m.reg, true), self.vr(v.vvvv, true), self.vrm(&m, true, 256), self.ib()?];
                self.vdone(if op == 0x06 { "vperm2f128" } else { "vperm2i128" }, ops)
            }
            (0x18 | 0x38, 0x66) => {
                if !v.l || w {
                    return None;
                }
                let m = self.modrm()?;
                let ops = vec![self.vr(m.reg, true), self.vr(v.vvvv, true), self.vrm(&m, false, 128), self.ib()?];
                self.vdone(if op == 0x18 { "vinsertf128" } else { "vinserti128" }, ops)
            }
            (0x19 | 0x39, 0x66) => {
                if !v.l || w || v.vvvv != 0 {
                    return None;
                }
                let m = self.modrm()?;
                let ops = vec![self.vrm(&m, false, 128), self.vr(m.reg, true), self.ib()?];
                self.vdone(if op == 0x19 { "vextractf128" } else { "vextracti128" }, ops)
            }
            (0x4a..=0x4c, 0x66) => {
                if w {
                    return None;
                }
                let name = ["vblendvps", "vblendvpd", "vpblendvb"][usize::from(op - 0x4a)];
                let m = self.modrm()?;
                let bits = if v.l { 256 } else { 128 };
                let (g, h, e) = (self.vr(m.reg, v.l), self.vr(v.vvvv, v.l), self.vrm(&m, v.l, bits));
                let is4 = self.u8()? >> 4;
                self.vdone(name, vec![g, h, e, self.vr(is4, v.l)])
            }
            (0xf0, 0xf2) => {
                if v.l || v.vvvv != 0 {
                    return None;
                }
                let y = self.p.y();
                let m = self.modrm()?;
                let ops = vec![self.reg(m.reg, y, K::Gpr), self.rm(&m, y, K::Gpr, y), self.ib()?];
                self.bmi("rorx", y, ops)
            }
            _ => {
                let row = SSE3A.iter().find(|r| r.op == op && r.pfx == v.pfx)?;
                self.vex_row(row, v)
            }
        }
    }

    /// A BMI instruction on general registers (AT&T adds the size suffix).
    fn bmi(&self, name: &str, y: u32, ops: Vec<Operand>) -> Option<X86Inst> {
        Some(self.make(self.pos, &format!("{name}{}", suffix(y)), name, ops))
    }

    /// FMA3: `vf[n]madd/msub[sub/add]{132,213,231}{ps,pd,ss,sd}`.
    fn fma(&mut self, op: u8, v: Vex) -> Option<X86Inst> {
        let order = match op >> 4 {
            0x9 => "132",
            0xa => "213",
            _ => "231",
        };
        let low = op & 15;
        let (kind, scalar) = match low {
            0x6 => ("vfmaddsub", false),
            0x7 => ("vfmsubadd", false),
            0x8 => ("vfmadd", false),
            0x9 => ("vfmadd", true),
            0xa => ("vfmsub", false),
            0xb => ("vfmsub", true),
            0xc => ("vfnmadd", false),
            0xd => ("vfnmadd", true),
            0xe => ("vfnmsub", false),
            _ => ("vfnmsub", true),
        };
        let w = self.p.w();
        let sfx = match (scalar, w) {
            (false, false) => "ps",
            (false, true) => "pd",
            (true, false) => "ss",
            (true, true) => "sd",
        };
        let l = v.l && !scalar;
        let bits = match (scalar, w) {
            (true, false) => 32,
            (true, true) => 64,
            _ if l => 256,
            _ => 128,
        };
        let m = self.modrm()?;
        let ops = vec![self.vr(m.reg, l), self.vr(v.vvvv, l), self.vrm(&m, l, bits)];
        self.vdone(&format!("{kind}{order}{sfx}"), ops)
    }

    /// The VEX form of an SSE table row.
    fn vex_row(&mut self, row: &Sse, v: Vex) -> Option<X86Inst> {
        let name = format!("v{}", row.name);
        let widening = WIDENING.contains(&row.name);
        let one = ONE_SOURCE.contains(&row.name) || widening;
        // Shifts by a count in an xmm register take it at 128 bits.
        let count = matches!(row.name, "psrlw" | "psrld" | "psrlq" | "psraw" | "psrad" | "psllw" | "pslld" | "psllq");
        let dup = row.name == "movddup";
        let scalar = row.mem < 128 && !widening && !dup && !matches!(row.form, F::GdU);
        if (v.l && (NO_VEX256.contains(&row.name) || matches!(row.form, F::EdVI | F::VEdI | F::GyW)))
            || matches!(row.form, F::VW0)
            || (one && v.vvvv != 0)
            || (matches!(row.form, F::WV | F::MV) && v.vvvv != 0)
        {
            return None;
        }
        // `vlddqu`/`vmovntdqa` and the narrowing conversions with a memory
        // source: llvm spells the latter with an x/y suffix; leave them.
        if matches!(row.name, "cvtpd2ps" | "cvttpd2dq" | "cvtpd2dq") {
            let m = self.modrm()?;
            if !matches!(m.rm, Rm::Reg(_)) || v.vvvv != 0 {
                return None;
            }
            let ops = vec![self.vr(m.reg, false), self.vrm(&m, false, 128)];
            return self.vdone(&name, ops);
        }
        let m = self.modrm()?;
        let l = v.l && !scalar;
        // Operand widths: destination and second source at the vector
        // length; the r/m source half as wide for widening rows.
        let src_l = if widening || count { false } else { l };
        let mem = if dup {
            if l { 256 } else { 64 }
        } else if count {
            128
        } else if scalar || widening {
            if widening && l { row.mem * 2 } else { row.mem }
        } else if l {
            256
        } else {
            128
        };
        let g = self.vr(m.reg, l);
        let h = self.vr(v.vvvv, l);
        let ops = match row.form {
            F::VW | F::VM => {
                if matches!(row.form, F::VM) && matches!(m.rm, Rm::Reg(_)) {
                    return None;
                }
                let e = self.vrm(&m, src_l, mem);
                if one { vec![g, e] } else { vec![g, h, e] }
            }
            F::WV | F::MV => {
                if matches!(row.form, F::MV) && matches!(m.rm, Rm::Reg(_)) {
                    return None;
                }
                vec![self.vrm(&m, l, mem), g]
            }
            F::VWI => {
                let e = self.vrm(&m, src_l, mem);
                let i = self.ib()?;
                if one { vec![g, e, i] } else { vec![g, h, e, i] }
            }
            F::GyW => {
                let y = self.p.y();
                vec![self.reg(m.reg, y, K::Gpr), self.vrm(&m, false, row.mem)]
            }
            F::GdU => {
                let Rm::Reg(r) = m.rm else { return None };
                vec![self.reg(m.reg, 32, K::Gpr), self.vr(r, v.l)]
            }
            F::EdVI => {
                let w = self.p.w();
                let (nm, regbits, mbits) = if row.name == "pextrd" && w { ("vpextrq", 64, 64) } else { (name.as_str(), 32, row.mem) };
                let ops = vec![self.rm(&m, regbits, K::Gpr, mbits), self.vr(m.reg, false), self.ib()?];
                return self.vdone(nm, ops);
            }
            F::VEdI => {
                let w = self.p.w();
                let (nm, regbits, mbits) = if row.name == "pinsrd" && w { ("vpinsrq", 64, 64) } else { (name.as_str(), 32, row.mem) };
                let ops = vec![self.vr(m.reg, false), self.vr(v.vvvv, false), self.rm(&m, regbits, K::Gpr, mbits), self.ib()?];
                return self.vdone(nm, ops);
            }
            F::VW0 => return None,
        };
        self.vdone(&name, ops)
    }
}
