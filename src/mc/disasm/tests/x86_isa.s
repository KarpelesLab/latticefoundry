pushq %rbp
pushq %r12
popq %r15
pushq $0x10
pushq $0x12345
pushw %ax
movq %rsp, %rbp
movl %eax, %r9d
movw %ax, %cx
movb %al, %dil
movb %ah, %bl
movq (%rax), %rcx
movq %rcx, 8(%rsp)
movl -0x14(%rbp), %eax
movq 0x100(%r12,%r13,8), %rax
movq (,%rbx,4), %rax
movq 0x1234, %rax
movl %fs:0x0, %eax
movq %fs:0x28, %rax
movq %gs:(%rax), %rdx
movq 0x10(%rip), %rax
movabsq $0x123456789abcdef0, %rax
movabsb 0x1122334455667788, %al
movl $0x12345678, %eax
movq $-0x1, %rax
movw $0x1234, (%rax)
movb $0x7f, 0x3(%rbx)
movl $0x5, %r10d
leaq 0x10(%rsp), %rdi
leaq (%rax,%rbx,2), %rcx
leal -0x1(%rdi), %eax
leaq 0x0(%rip), %rax
addq %rax, %rbx
addl $0x7, %eax
addq $0x1000, %rsp
addb $0x1, %al
addw $0x100, %ax
addl $0x100000, %eax
adcq %rcx, %rdx
sbbl %eax, %eax
subq $0x18, %rsp
subl (%rdi), %esi
andl $0xff, %eax
andq $-0x10, %rsp
orq %rdx, %rax
orb $0x4, (%rcx)
xorl %eax, %eax
xorq %r11, %rax
cmpq %rdi, %r8
cmpl $0x0, %eax
cmpb $0x41, (%rsi)
cmpw %ax, %cx
cmpq $0x7fffffff, %rax
testl %edi, %edi
testb $0x1, %al
testq $0x100, %rbx
testb %sil, %dil
incl %eax
incq (%rax)
decq %rcx
decb %al
negq %rax
negl %r9d
notq %rdx
mulq %rcx
imulq %rdi
imulq %rdi, %rax
imull $0x64, %ecx, %eax
imulq $0x12345, %rsi, %rdx
imull $-0x3, %eax, %eax
divq %rcx
divl %esi
idivq %rdi
idivl (%rax)
shlq $0x3f, %rcx
shrq $0x3f, %rcx
sarq $0x3, %rcx
shlq %rcx
shrl %cl, %eax
sarq %cl, %rax
roll $0x5, %eax
rorq $0x10, %rdx
rolb %cl, %al
shldq $0x4, %rax, %rdx
shrdq %cl, %rax, %rdx
xchgq %rax, %rbx
xchgl %ecx, (%rdx)
xchgq %rax, %r8
lock xaddq %rax, (%rdi)
lock xaddl %ecx, (%rsi)
lock cmpxchgq %rcx, (%rdi)
lock cmpxchgb %cl, (%rdi)
lock cmpxchg16b (%rdi)
lock addq $0x1, (%rax)
lock orl %eax, (%rbx)
lock incq (%rax)
btq %rax, %rcx
btl $0x3, %eax
btsq $0x3f, %rax
btrl %ecx, %edx
btcq $0x1, (%rax)
bsfq %rax, %rcx
bsrl %eax, %edx
popcntq %rax, %rcx
lzcntl %eax, %edx
tzcntq %rdi, %rax
bswapq %rax
bswapl %r9d
movzbl %al, %eax
movzbl %dil, %edi
movzwl %ax, %ecx
movzbq (%rax), %rcx
movzwq %r8w, %r9
movsbl %cl, %eax
movswl %ax, %edx
movsbq %al, %rax
movswq (%rsi), %rax
movslq %eax, %rcx
movslq (%rdi), %rax
cbtw
cwtl
cltq
cwtd
cltd
cqto
sete %al
setne %dil
setl %r8b
setg (%rax)
setb %cl
seta %dl
setbe %al
setae %al
setle %al
setge %al
sets %al
setns %al
setp %al
setnp %al
seto %al
setno %al
cmoveq %rax, %rbx
cmovnel %ecx, %edx
cmovlq (%rdi), %rax
cmovgeq %r8, %r9
cmovaq %rax, %rcx
cmovbq %rax, %rcx
jmp *%rax
jmpq *(%rax)
jmpq *0x8(%rax,%rcx,8)
callq *%rax
callq *0x10(%rbx)
retq
retq $0x8
leave
nop
nopw %ax
nopl (%rax)
nopl 0x0(%rax)
nopl 0x0(%rax,%rax)
nopw 0x0(%rax,%rax)
nopl 0x0(%rax,%rax)
nopw %cs:0x0(%rax,%rax)
int3
ud2
hlt
syscall
cpuid
rdtsc
endbr64
pause
mfence
lfence
sfence
cld
std
clc
stc
cmc
rep movsb
rep movsq
rep stosb
rep stosq
movsb
stosl
lodsb
scasb
repne scasb
int $0x80
movss %xmm0, %xmm1
movss (%rax), %xmm2
movss %xmm3, 0x4(%rsp)
movsd %xmm0, %xmm1
movsd -0x8(%rbp), %xmm9
movsd %xmm15, (%r12)
movaps %xmm0, %xmm2
movaps (%rax), %xmm1
movaps %xmm1, (%rax)
movups (%rdi), %xmm0
movups %xmm0, (%rdi)
movapd %xmm1, %xmm2
movupd (%rax), %xmm3
movdqa %xmm0, %xmm1
movdqa (%rax), %xmm1
movdqu %xmm0, (%rdi)
movdqu (%rsi), %xmm7
movd %eax, %xmm0
movd %xmm0, %eax
movq %rax, %xmm0
movq %xmm0, %rax
movq %xmm1, %xmm2
movq (%rax), %xmm3
movq %xmm3, (%rax)
movd (%rax), %xmm1
addss %xmm1, %xmm0
addsd %xmm1, %xmm0
subss %xmm1, %xmm0
subsd (%rax), %xmm0
mulss %xmm1, %xmm0
mulsd %xmm1, %xmm0
divss %xmm1, %xmm0
divsd %xmm1, %xmm0
minss %xmm1, %xmm0
maxsd %xmm1, %xmm0
sqrtsd %xmm1, %xmm0
sqrtss %xmm1, %xmm0
sqrtps %xmm1, %xmm0
addps %xmm1, %xmm0
addpd %xmm1, %xmm0
subps %xmm1, %xmm0
mulpd %xmm1, %xmm0
divps %xmm1, %xmm0
minps %xmm1, %xmm0
maxpd %xmm1, %xmm0
andps %xmm1, %xmm0
andpd %xmm1, %xmm0
andnps %xmm1, %xmm0
andnpd %xmm1, %xmm0
orps %xmm1, %xmm0
orpd %xmm1, %xmm0
xorps %xmm1, %xmm0
xorpd %xmm1, %xmm0
ucomiss %xmm1, %xmm0
ucomisd %xmm1, %xmm0
comiss %xmm1, %xmm0
comisd (%rax), %xmm0
cvtsi2sd %eax, %xmm0
cvtsi2sd %rax, %xmm0
cvtsi2ss %eax, %xmm0
cvtsi2ss %rax, %xmm1
cvtsi2sdl (%rax), %xmm0
cvtsi2sdq (%rax), %xmm0
cvttsd2si %xmm0, %eax
cvttsd2si %xmm0, %rax
cvttss2si %xmm0, %eax
cvttss2si %xmm0, %rax
cvtsd2si %xmm0, %rax
cvtss2si %xmm0, %eax
cvtss2sd %xmm0, %xmm1
cvtsd2ss %xmm0, %xmm1
cvtdq2ps %xmm0, %xmm1
cvttps2dq %xmm0, %xmm1
cvtps2pd %xmm0, %xmm1
cvtpd2ps %xmm0, %xmm1
cvtdq2pd %xmm0, %xmm1
cvttpd2dq %xmm0, %xmm1
cmpltsd %xmm1, %xmm0
cmpeqps %xmm1, %xmm0
cmpunordpd %xmm1, %xmm0
cmpnless %xmm1, %xmm0
paddb %xmm1, %xmm0
paddw %xmm1, %xmm0
paddd %xmm1, %xmm0
paddq %xmm1, %xmm0
psubb %xmm1, %xmm0
psubw %xmm1, %xmm0
psubd %xmm1, %xmm0
psubq %xmm1, %xmm0
pmullw %xmm1, %xmm0
pmuludq %xmm1, %xmm0
pmulld %xmm1, %xmm0
pand %xmm1, %xmm0
pandn %xmm1, %xmm0
por %xmm1, %xmm0
pxor %xmm1, %xmm0
pxor %xmm8, %xmm15
pcmpeqb %xmm1, %xmm0
pcmpeqw %xmm1, %xmm0
pcmpeqd %xmm1, %xmm0
pcmpeqq %xmm1, %xmm0
pcmpgtb %xmm1, %xmm0
pcmpgtw %xmm1, %xmm0
pcmpgtd %xmm1, %xmm0
pcmpgtq %xmm1, %xmm0
pshufd $0xb1, %xmm1, %xmm0
pshuflw $0x1b, %xmm1, %xmm0
pshufhw $0x1b, %xmm1, %xmm0
pshufb %xmm1, %xmm0
shufps $0x88, %xmm1, %xmm0
shufpd $0x1, %xmm1, %xmm0
punpcklbw %xmm1, %xmm0
punpcklwd %xmm1, %xmm0
punpckldq %xmm1, %xmm0
punpcklqdq %xmm1, %xmm0
punpckhbw %xmm1, %xmm0
punpckhwd %xmm1, %xmm0
punpckhdq %xmm1, %xmm0
punpckhqdq %xmm1, %xmm0
unpcklps %xmm1, %xmm0
unpckhpd %xmm1, %xmm0
packsswb %xmm1, %xmm0
packuswb %xmm1, %xmm0
packssdw %xmm1, %xmm0
packusdw %xmm1, %xmm0
psllw $0x3, %xmm0
pslld $0x3, %xmm0
psllq $0x3, %xmm0
psrlw $0x3, %xmm0
psrld $0x3, %xmm0
psrlq $0x3, %xmm0
psraw $0x3, %xmm0
psrad $0x3, %xmm0
pslldq $0x4, %xmm0
psrldq $0x4, %xmm0
psllw %xmm1, %xmm0
psrld %xmm1, %xmm0
psraw %xmm1, %xmm0
pminub %xmm1, %xmm0
pmaxub %xmm1, %xmm0
pminsw %xmm1, %xmm0
pmaxsw %xmm1, %xmm0
pminsb %xmm1, %xmm0
pmaxsb %xmm1, %xmm0
pminuw %xmm1, %xmm0
pmaxuw %xmm1, %xmm0
pminsd %xmm1, %xmm0
pmaxsd %xmm1, %xmm0
pminud %xmm1, %xmm0
pmaxud %xmm1, %xmm0
paddsb %xmm1, %xmm0
paddsw %xmm1, %xmm0
paddusb %xmm1, %xmm0
paddusw %xmm1, %xmm0
psubsb %xmm1, %xmm0
psubsw %xmm1, %xmm0
psubusb %xmm1, %xmm0
psubusw %xmm1, %xmm0
pavgb %xmm1, %xmm0
pmovmskb %xmm0, %eax
movmskps %xmm0, %eax
movmskpd %xmm0, %eax
pextrw $0x3, %xmm0, %eax
pinsrw $0x3, %eax, %xmm0
pextrb $0x3, %xmm0, %eax
pextrd $0x3, %xmm0, %eax
pextrq $0x1, %xmm0, %rax
pinsrb $0x3, %eax, %xmm0
pinsrd $0x3, %eax, %xmm0
pinsrq $0x1, %rax, %xmm0
insertps $0x10, %xmm1, %xmm0
extractps $0x1, %xmm0, %eax
pmovzxbw %xmm1, %xmm0
pmovsxwd %xmm1, %xmm0
pmovzxdq %xmm1, %xmm0
pblendw $0xf, %xmm1, %xmm0
blendps $0x3, %xmm1, %xmm0
roundsd $0x9, %xmm1, %xmm0
roundss $0xa, %xmm1, %xmm0
roundps $0xb, %xmm1, %xmm0
pabsd %xmm1, %xmm0
palignr $0x4, %xmm1, %xmm0
ptest %xmm1, %xmm0
rcpps %xmm1, %xmm0
rsqrtss %xmm1, %xmm0
haddps %xmm1, %xmm0
movlhps %xmm1, %xmm0
movhlps %xmm1, %xmm0
movlps (%rax), %xmm0
movhps %xmm0, (%rax)
movhpd (%rax), %xmm0
movntdq %xmm0, (%rax)
lddqu (%rax), %xmm0
movddup %xmm1, %xmm0
movshdup %xmm1, %xmm0
ldmxcsr (%rax)
stmxcsr (%rax)
prefetcht0 (%rax)
prefetchnta (%rax)
clflush (%rax)
xgetbv
rdtscp
movq %cr0, %rax
flds (%rax)
fldl 0x8(%rsp)
fldt (%rdi)
fstps -0x4(%rbp)
fstpl (%rax)
fstpt 0x10(%rsp)
fildl (%rax)
fildll (%rax)
filds (%rax)
fistpll (%rax)
fisttpll (%rax)
fistpl (%rax)
fadds (%rax)
faddl (%rax)
fmull (%rax)
fdivrl (%rax)
fiaddl (%rax)
fidivrs (%rax)
fcomps (%rax)
fldcw (%rax)
fnstcw (%rax)
fnstsw (%rax)
fnstsw %ax
fnstenv (%rax)
fldenv (%rax)
fadd %st(1), %st
fadd %st, %st(2)
faddp %st, %st(1)
fsubp %st, %st(1)
fsubrp %st, %st(1)
fdivp %st, %st(3)
fdivrp %st, %st(1)
fmulp %st, %st(1)
fsub %st(1), %st
fsubr %st, %st(1)
fxch %st(1)
fld %st(0)
fstp %st(1)
fst %st(2)
ffree %st(3)
fucomi %st(1), %st
fucompi %st(1), %st
fcomi %st(2), %st
fcmovbe %st(1), %st
fcmovnu %st(3), %st
fucompp
fcompp
fchs
fabs
fldz
fld1
fldpi
fsqrt
frndint
fscale
fprem
fxam
ftst
fninit
fnclex
pcmpistri $0x12, (%rax), %xmm0
pcmpestrm $0x40, %xmm1, %xmm0
movbel (%rsi), %eax
movbeq %rax, (%rdi)
crc32b %cl, %eax
crc32l %ecx, %eax
crc32q %rcx, %rax
xtest
rdpkru
vzeroupper
vzeroall
vmovdqu (%rdi), %ymm0
vmovdqu %ymm1, 0x20(%rsi)
vmovdqa %ymm0, %ymm1
vmovups (%rax), %xmm2
vmovaps %xmm3, (%rax)
vpcmpeqb (%rdi), %ymm0, %ymm1
vpcmpeqb %xmm2, %xmm1, %xmm3
vpmovmskb %ymm1, %eax
vpminub %ymm2, %ymm3, %ymm4
vpor %ymm1, %ymm2, %ymm3
vpxor %xmm0, %xmm0, %xmm0
vpand (%rax), %ymm1, %ymm2
vptest %ymm1, %ymm1
vpbroadcastb %xmm0, %ymm0
vpbroadcastd (%rax), %ymm1
vpbroadcastq %xmm2, %xmm3
vbroadcastss (%rax), %ymm0
vbroadcastsd %xmm1, %ymm2
vpshufb %ymm1, %ymm2, %ymm3
vpalignr $0x8, %ymm1, %ymm2, %ymm3
vpblendvb %ymm4, %ymm1, %ymm2, %ymm3
vblendvps %xmm4, %xmm1, %xmm2, %xmm3
vinserti128 $0x1, %xmm1, %ymm2, %ymm3
vextracti128 $0x1, %ymm1, %xmm2
vinsertf128 $0x1, (%rax), %ymm2, %ymm3
vextractf128 $0x1, %ymm1, (%rax)
vperm2i128 $0x21, %ymm1, %ymm2, %ymm3
vpermq $0x4e, %ymm1, %ymm2
vpermd %ymm1, %ymm2, %ymm3
vpsllvd %ymm1, %ymm2, %ymm3
vpsrlvq %xmm1, %xmm2, %xmm3
vpsravd %ymm1, %ymm2, %ymm3
vpblendd $0xf0, %ymm1, %ymm2, %ymm3
vaddps %ymm1, %ymm2, %ymm3
vaddsd %xmm1, %xmm2, %xmm3
vmulpd (%rax), %ymm1, %ymm2
vsqrtsd %xmm1, %xmm2, %xmm3
vsqrtps %ymm1, %ymm2
vxorps %ymm0, %ymm0, %ymm0
vmovss (%rax), %xmm0
vmovss %xmm1, %xmm2, %xmm3
vmovsd %xmm0, (%rax)
vmovq %rax, %xmm0
vmovq %xmm0, %rax
vmovd %eax, %xmm1
vmovq %xmm1, %xmm2
vcvtsi2sd %rax, %xmm1, %xmm0
vcvtsi2sdl (%rax), %xmm1, %xmm0
vcvttsd2si %xmm0, %rax
vcvtss2sd %xmm1, %xmm2, %xmm3
vcvtdq2ps %ymm1, %ymm2
vcvtps2pd %xmm1, %ymm2
vpmovzxbw %xmm1, %ymm2
vpmovsxdq (%rax), %ymm2
vucomisd %xmm1, %xmm0
vcmpltps %ymm1, %ymm2, %ymm3
vcmpeq_uqsd %xmm1, %xmm2, %xmm3
vpshufd $0x1b, %ymm1, %ymm2
vpsrldq $0x4, %ymm1, %ymm2
vpslld $0x3, %xmm1, %xmm2
vpsrad %xmm1, %ymm2, %ymm3
vpinsrd $0x1, %eax, %xmm1, %xmm2
vpinsrq $0x1, %rax, %xmm1, %xmm2
vpextrq $0x1, %xmm1, %rax
vpextrb $0x1, %xmm1, (%rax)
vpunpcklbw %ymm1, %ymm2, %ymm3
vroundsd $0x9, %xmm1, %xmm2, %xmm3
vfmadd231sd %xmm1, %xmm2, %xmm3
vfmadd213ps %ymm1, %ymm2, %ymm3
vfnmsub132pd (%rax), %xmm2, %xmm3
vfmaddsub231ps %ymm1, %ymm2, %ymm3
andnq %rax, %rbx, %rcx
blsrl %eax, %ecx
blsil %eax, %ecx
bzhiq %rax, %rbx, %rcx
bextrl %eax, (%rdi), %ecx
shlxq %rax, %rbx, %rcx
sarxl %eax, %ebx, %ecx
shrxq %rax, (%rdi), %rcx
pdepq %rax, %rbx, %rcx
pextl %eax, %ebx, %ecx
mulxq %rax, %rbx, %rcx
rorxq $0x7, %rax, %rcx
movq %fs:0x0, %rax
addq 0x0(%rip), %rax
mulq %rcx
mulq (%rax)
movq $0x1, %rax
addq $0x10, %rax
.byte 0x66, 0x48, 0x8d, 0x3d, 0, 0, 0, 0
.byte 0x66, 0x66, 0x48, 0xe8, 0, 0, 0, 0
.byte 0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0
.byte 0x48, 0xc7, 0xc0, 0, 0, 0, 0
.byte 0x48, 0x81, 0xc0, 0, 0, 0, 0
