add x0, x1, #16
add x0, x1, #16, lsl #12
add sp, sp, #32
add x0, sp, #0
mov x0, sp
sub w3, w4, #4095
subs x0, x1, #1
cmp x1, #5
cmn w1, #5
add x0, x1, x2, lsl #3
add x0, x1, w2, sxtw #2
add x0, sp, x2
add sp, x1, x2
sub x0, x1, x2, asr #7
neg x0, x1
negs w0, w1, lsl #2
cmp x0, x1
cmp w0, w1, uxtb
and x0, x1, #0xff
orr w0, wzr, #0x3
mov x0, #0x1234
mov x0, #-2
mov w0, #-1
movk x0, #0x1234, lsl #16
movz x0, #0, lsl #48
movn x0, #0, lsl #32
movz x0, #1, lsl #16
movn w0, #0xffff
tst x0, #8
eor x0, x1, x2, ror #3
mvn x0, x1
bic x0, x1, x2
ands x0, x1, x2
tst w0, w1
lsl x0, x1, #3
lsr w0, w1, #3
asr x0, x1, #63
ubfx x0, x1, #4, #8
sbfx x0, x1, #4, #8
ubfiz x0, x1, #4, #8
sbfiz w0, w1, #4, #8
bfi x0, x1, #4, #8
bfxil x0, x1, #4, #8
sxtw x0, w1
sxtb w0, w1
uxtb w0, w1
uxth w0, w1
sxth x0, w1
ubfx x0, x1, #0, #32
extr x0, x1, x2, #7
ror x0, x1, #7
adc x0, x1, x2
sbcs w0, w1, w2
ngc x0, x1
csel x0, x1, x2, eq
csinc x0, x1, x2, ne
cset w0, lt
csetm x0, ge
cinc x0, x1, hi
cneg x0, x1, mi
cinv x0, x1, le
csinv x0, x1, x2, vs
csneg x0, x1, x2, al
ccmp x0, x1, #4, ne
ccmn w0, #5, #2, eq
rbit x0, x1
rev w0, w1
rev x0, x1
rev16 x0, x1
rev32 x0, x1
clz x0, x1
cls w0, w1
udiv x0, x1, x2
sdiv w0, w1, w2
lsl x0, x1, x2
asr w0, w1, w2
ror x0, x1, x2
crc32b w0, w1, w2
crc32cx w0, w1, x2
mul x0, x1, x2
madd x0, x1, x2, x3
msub x0, x1, x2, x3
mneg x0, x1, x2
smull x0, w1, w2
umull x0, w1, w2
smaddl x0, w1, w2, x3
umsubl x0, w1, w2, x3
umulh x0, x1, x2
smulh x0, x1, x2
b .+8
bl .+0x100
b.ne .-4
b.al .+8
cbz x0, .+16
cbnz w0, .+16
tbz x0, #40, .+16
tbnz w0, #3, .-16
br x16
blr x8
ret
ret x1
svc #0
brk #0x3e8
hlt #1
nop
yield
dmb ish
dmb ishld
dsb sy
dmb #0
isb
mrs x0, tpidr_el0
msr tpidr_el0, x1
mrs x0, nzcv
mrs x1, fpcr
msr fpsr, x2
mrs x0, cntvct_el0
adr x0, .+16
adrp x0, .+0x2000
ldr x0, [x1]
ldr x0, [x1, #8]
ldr w0, [sp, #4092]
ldrb w0, [x1, #1]
ldrsb x0, [x1, #1]
ldrsb w0, [x1]
ldrh w0, [x1, #2]
ldrsh x0, [x1, #2]
ldrsw x0, [x1, #4]
strb w0, [x1]
strh w0, [x1, #2]
str x0, [x1, #-8]!
str x0, [x1], #8
ldrb w0, [x1], #-1
ldur x0, [x1, #-3]
stur w0, [x1, #1]
ldurb w0, [x1, #-1]
ldursw x0, [x1, #-4]
ldr x0, [x1, x2]
ldr x0, [x1, x2, lsl #3]
ldr w0, [x1, w2, sxtw]
ldrb w0, [x1, x2]
ldrb w0, [x1, x2, lsl #0]
ldr x0, [x1, w2, uxtw #3]
ldrh w0, [x1, x2, sxtx #1]
ldr x0, .+16
ldr w0, .+16
ldrsw x0, .+16
prfm pldl1keep, [x0]
stp x29, x30, [sp, #-16]!
ldp x29, x30, [sp], #16
stp x0, x1, [sp, #16]
ldp w0, w1, [x2]
ldpsw x0, x1, [x2, #8]
stp d8, d9, [sp, #-16]!
ldp q0, q1, [x0, #32]
stp s0, s1, [x0]
stnp x0, x1, [x2]
ldxr x0, [x1]
ldaxr w0, [x1]
ldxrb w0, [x1]
stxr w2, x0, [x1]
stlxr w2, w0, [x1]
stlxrh w2, w0, [x1]
ldar x0, [x1]
ldarb w0, [x1]
stlr w0, [x1]
stlrh w0, [x1]
ldxp x0, x1, [x2]
stxp w3, x0, x1, [x2]
ldadd x0, x1, [x2]
ldaddal w0, w1, [x2]
ldclrb w0, w1, [x2]
ldsetlh w0, w1, [x2]
ldeor x0, x1, [x2]
stadd x0, [x2]
swp x0, x1, [x2]
swpal w0, w1, [x2]
cas x0, x1, [x2]
casal w0, w1, [x2]
casb w0, w1, [x2]
fmov d0, d1
fmov s0, s1
fmov d0, #1.0
fmov s0, #-2.5
fmov d0, #0.125
fmov x0, d1
fmov d0, x1
fmov w0, s1
fmov s0, w1
fmov x0, v1.d[1]
fmov v0.d[1], x1
fadd d0, d1, d2
fsub s0, s1, s2
fmul d0, d1, d2
fdiv d0, d1, d2
fmax d0, d1, d2
fmin d0, d1, d2
fmaxnm d0, d1, d2
fminnm s0, s1, s2
fnmul d0, d1, d2
fmadd d0, d1, d2, d3
fmsub s0, s1, s2, s3
fnmadd d0, d1, d2, d3
fnmsub d0, d1, d2, d3
fneg d0, d1
fabs s0, s1
fsqrt d0, d1
fcvt d0, s1
fcvt s0, d1
fcvt h0, s1
fcvt d0, h1
fcvt s0, h1
fcmp d0, d1
fcmp s0, #0.0
fcmpe d0, d1
fcmpe d0, #0.0
fccmp d0, d1, #0, eq
fccmpe s0, s1, #3, lt
fcsel d0, d1, d2, gt
fcvtzs x0, d1
fcvtzu w0, s1
fcvtzs w0, d1, #3
scvtf d0, x1
ucvtf s0, w1
scvtf d0, w1, #4
fcvtas x0, d0
fcvtms w0, s0
fcvtns x0, d0
fcvtps w0, s0
fcvtau x0, d0
frintm d0, d1
frintz s0, s1
frinta d0, d1
frintn d0, d1
frintp d0, d1
frintx d0, d1
frinti d0, d1
ldr d0, [x1, #8]
ldr s0, [x1, #4]
str q0, [sp, #16]
ldr q0, [x0]
ldr b0, [x0]
ldr h0, [x0, #2]
ldur d0, [x1, #-8]
str d0, [x1, #-8]!
ldr d0, [x1, x2, lsl #3]
ldr q0, .+16
ldr s0, .+16
mov v0.16b, v1.16b
orr v0.16b, v1.16b, v2.16b
and v0.16b, v1.16b, v2.16b
bic v0.16b, v1.16b, v2.16b
eor v0.16b, v1.16b, v2.16b
orn v0.16b, v1.16b, v2.16b
bsl v0.16b, v1.16b, v2.16b
bit v0.16b, v1.16b, v2.16b
mov v0.8b, v1.8b
add v0.4s, v1.4s, v2.4s
add v0.2d, v1.2d, v2.2d
sub v0.8h, v1.8h, v2.8h
mul v0.16b, v1.16b, v2.16b
add v0.2s, v1.2s, v2.2s
cmeq v0.4s, v1.4s, v2.4s
cmgt v0.4s, v1.4s, v2.4s
cmge v0.8h, v1.8h, v2.8h
cmhi v0.2d, v1.2d, v2.2d
cmhs v0.16b, v1.16b, v2.16b
cmtst v0.4s, v1.4s, v2.4s
sshl v0.4s, v1.4s, v2.4s
ushl v0.4s, v1.4s, v2.4s
smax v0.4s, v1.4s, v2.4s
umin v0.8h, v1.8h, v2.8h
smin v0.4s, v1.4s, v2.4s
umax v0.8h, v1.8h, v2.8h
sqadd v0.16b, v1.16b, v2.16b
uqsub v0.16b, v1.16b, v2.16b
uqadd v0.16b, v1.16b, v2.16b
sqsub v0.8h, v1.8h, v2.8h
addp v0.4s, v1.4s, v2.4s
mla v0.4s, v1.4s, v2.4s
fadd v0.4s, v1.4s, v2.4s
fsub v0.2d, v1.2d, v2.2d
fmul v0.4s, v1.4s, v2.4s
fdiv v0.2d, v1.2d, v2.2d
fmax v0.4s, v1.4s, v2.4s
fmin v0.2d, v1.2d, v2.2d
fcmeq v0.4s, v1.4s, v2.4s
fcmge v0.2d, v1.2d, v2.2d
fcmgt v0.4s, v1.4s, v2.4s
fmla v0.4s, v1.4s, v2.4s
tbl v0.16b, {v1.16b}, v2.16b
neg v0.4s, v1.4s
mvn v0.16b, v1.16b
abs v0.4s, v1.4s
cnt v0.8b, v1.8b
fneg v0.2d, v1.2d
fabs v0.4s, v1.4s
fsqrt v0.2d, v1.2d
scvtf v0.4s, v1.4s
ucvtf v0.2d, v1.2d
fcvtzs v0.4s, v1.4s
fcvtzu v0.2d, v1.2d
cmeq v0.4s, v1.4s, #0
rev64 v0.16b, v1.16b
addv s0, v1.4s
addv b0, v1.16b
smaxv h0, v1.8h
sminv b0, v1.16b
umaxv s0, v1.4s
uminv s0, v1.4s
uaddlv h0, v1.8b
addp d0, v1.2d
shl v0.4s, v1.4s, #3
ushr v0.2d, v1.2d, #63
sshr v0.16b, v1.16b, #1
ushr d0, d1, #3
shl d0, d1, #3
sshll v0.8h, v1.8b, #0
ushll v0.4s, v1.4h, #2
dup v0.4s, w1
dup v0.2d, x1
dup v0.16b, w1
dup v0.8h, v1.h[3]
dup v0.2d, v1.d[1]
umov w0, v1.b[3]
umov w0, v1.h[3]
mov x0, v1.d[1]
mov w0, v1.s[1]
smov x0, v1.h[2]
smov w0, v1.b[2]
mov v0.s[1], w1
mov v0.d[1], x1
mov v0.b[3], w1
mov v0.s[1], v1.s[0]
mov v0.b[15], v1.b[2]
mov v0.d[1], v1.d[0]
movi v0.2d, #0
movi v0.16b, #0xff
movi v0.4s, #1, lsl #8
movi v0.2d, #0xff00ff00ff00ff00
movi d0, #0
mvni v0.4s, #1
xtn v0.8b, v1.8h
xtn2 v0.16b, v1.8h
uzp1 v0.4s, v1.4s, v2.4s
uzp2 v0.8h, v1.8h, v2.8h
zip1 v0.4s, v1.4s, v2.4s
zip2 v0.16b, v1.16b, v2.16b
trn1 v0.4s, v1.4s, v2.4s
ext v0.16b, v1.16b, v2.16b, #4
ld1 {v0.16b}, [x0]
st1 {v0.4s, v1.4s}, [x1]
ld1 {v0.2d}, [x0], #16
ld1 {v0.4s}, [x0], x2
ld1r {v0.4s}, [x0]
ld1 {v0.s}[1], [x0]
st1 {v0.d}[1], [x0]
fmov v0.4s, #1.0
fmov v0.2d, #-0.5
fadd s0, s1, s2
udf #0
hint #34
add w0, w1, #1
sub sp, sp, #16
and w0, w1, #0xfffffffe
orr x0, x1, #0x5555555555555555
eor x0, x1, #0xfffffffffffff
ands w0, w1, #0x1
mov x0, #0x10000
mov x0, #0xffff0000ffff0000
add x0, x0, #0x123, lsl #12
sub x16, x16, #1, lsl #12
add x0, x1, x2, uxtx
sub sp, sp, x16, uxtx
add x0, sp, x16, uxtx #0
add w0, wsp, w1
adds x0, x1, #4
subs w0, w1, w2
negs x0, x1
cmp sp, #16
nop
svc #16
hvc #0x20
brk #0
dmb #3
dsb #0
dsb #4
dsb ish
clrex
isb #3
msr daifset, #2
msr daifclr, #0xf
dc civac, x0
dc zva, x1
ic ivau, x2
ic iallu
sys #1, c2, c3, #4, x5
prfm #0x1f, [x0]
prfm pstl2strm, [x0, #8]
mrs x0, s3_4_c15_c2_1
movz w0, #0x8000, lsl #16
mov w0, #0x10000
bfc x0, #4, #8
bfi x0, xzr, #4, #8
hint #7
hint #25
hint #29
hint #0x7f
wfi
wfe
sev
sevl
esb
csdb
mov x0, #0
ldtr x0, [x1, #8]
sttrb w0, [x1]
ldr x0, [x1, #-256]!
ldaprb w0, [x0]
ldapr x0, [x0]
umull v0.2d, v1.2s, v2.2s
saddw v0.8h, v1.8h, v2.8b
fmov h0, h1
fadd h0, h1, h2
cmeq d0, d1, d2
add d0, d1, d2
fmaxnmv s0, v1.4s
fcvtl v0.2d, v1.2s
fcvtn v0.2s, v1.2d
ld2 {v0.4s, v1.4s}, [x0]
st4 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0], #32
ld1 {v0.16b, v1.16b, v2.16b}, [x0]
ld1r {v0.8b}, [x0], #1
tbl v0.8b, {v0.16b, v1.16b}, v2.8b
tbx v0.16b, {v1.16b}, v2.16b
orr v0.4s, #0x1, lsl #8
bic v0.8h, #0x2
movi v0.4s, #0x2, msl #8
movi v0.8b, #0x7
movi v0.4h, #0x7, lsl #8
mul v0.4s, v1.4s, v2.s[1]
fmla v0.4s, v1.4s, v2.s[1]
sqrdmulh v0.4s, v1.4s, v2.4s
not v0.8b, v1.8b
eret
drps
casp x0, x1, x2, x3, [x4]
ldaxp w0, w1, [x2]
stlxp w0, x1, x2, [x3]
ldnp q0, q1, [x2, #-32]
stlur w0, [x1, #-4]
ldapur x0, [x1, #8]
uxtw x0, w1
mov x0, xzr
mov w0, wzr
orr x0, xzr, x1, lsl #2
cmp xzr, x1
neg xzr, x1
sxtl v0.8h, v1.8b
dup v0.2s, w1
fmov v0.2s, #1.0
movi v0.2s, #0x1
xpaclri
paciasp
autiasp
bti
bti j
ret x30
br x30
