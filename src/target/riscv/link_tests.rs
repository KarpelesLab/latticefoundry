//! Objects, relocations and links: the RISC-V ELF objects are checked with
//! `llvm-readobj`/`llvm-objdump`, linked by qld into static executables whose
//! headers and resolved code are checked, and executed — loaded from the
//! linked file into the simulator ([`super::sim::load_elf`]).
//!
//! The LP64D convention is checked against an independent implementation:
//! when `clang` can target riscv64, C code compiled by it (`-march=rv64gc
//! -mabi=lp64d`, so with compressed instructions) is linked with ours, and C
//! calls our functions and ours call C's, with floats overflowing into integer
//! registers, structs of every class, a split struct and variadic doubles.

use std::path::{Path, PathBuf};

use crate::codegen::CodegenOptions;
use crate::support::StrInterner;

use super::diff_tests::parse;
use super::sim::{Cpu, Image, STACK_TOP, load_elf};

/// A private scratch directory for one test.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-rv-link-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run a tool, returning its stdout when it ran and succeeded.
fn tool(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Compile `src` and write it as a RISC-V ELF object at `path`.
fn write_object(src: &str, path: &Path, opts: &CodegenOptions) -> (crate::ir::Module, StrInterner) {
    let (m, syms) = parse(src);
    let compiled = super::compile_module_with(&m, &syms, opts);
    let bytes = super::write_elf(&compiled.object).expect("an ELF object");
    std::fs::write(path, bytes).unwrap();
    (m, syms)
}

/// Link `objects` statically with qld (`-m elf64lriscv`), returning the
/// executable's bytes.
fn qld_static(objects: &[&Path], out: &Path, entry: &str) -> Vec<u8> {
    let mut args: Vec<String> = vec!["-m".into(), "elf64lriscv".into(), "-static".into(), "-e".into(), entry.into()];
    args.push("-o".into());
    args.push(out.to_str().unwrap().into());
    args.extend(objects.iter().map(|p| p.to_str().unwrap().to_owned()));
    crate::link::gnu::link_gnu("qld", &args).unwrap_or_else(|e| panic!("qld: {e}"));
    std::fs::read(out).unwrap()
}

/// Call `name` in the linked image with integer / float register arguments,
/// returning `(a0, fa0)`.
fn call(image: &Image, name: &str, xs: &[u64], fs: &[u64]) -> (u64, u64) {
    let mut cpu = Cpu::new(image);
    let entry = *image.symbols.get(name).unwrap_or_else(|| panic!("no symbol {name}"));
    cpu.call(entry, xs, fs, STACK_TOP - 4096).unwrap_or_else(|e| panic!("{name}: {e}"));
    (cpu.x[10], cpu.f[10])
}

const PROGRAM: &str = r#"
module "prog"
global @counter : i64 = i64 5
global constant @table : [3 x ptr] = [3 x ptr] (ptr @counter, ptr @step, ptr @table + 8)
global @fp : ptr = ptr @step
global @half : f64 = f64 0x3fe0000000000000

func @step(i64) -> i64 {
entry ^0(%x: i64):
  %c = load @counter align 8 : i64
  %r = add %x, %c : i64
  store %r, @counter align 8 : i64
  ret %r
}

func @entry(i64) -> i64 {
entry ^0(%x: i64):
  %a = call @step(%x) : i64
  %f = load @fp align 8 : ptr
  %b = call %f(%a) : i64
  %t1 = ptr_add @table, i64 8 : ptr
  %g = load %t1 align 8 : ptr
  %c = call %g(%b) : i64
  %t2 = ptr_add @table, i64 16 : ptr
  %p = load %t2 align 8 : ptr
  %self = load %p align 8 : ptr
  %eq = icmp eq %self, @step : i1
  %e = zext %eq : i64
  %h = load @half align 8 : f64
  %cf = sitofp %c : f64
  %m = fmul %cf, %h : f64
  %mi = fptosi %m : i64
  %s = add %mi, %e : i64
  ret %s
}
"#;

/// The object's header, sections, symbols and relocations, as `llvm-readobj`
/// reads them; and the code, as `llvm-objdump` disassembles it with the
/// relocations attached.
#[test]
fn objects_carry_the_psabi_relocations() {
    let dir = scratch("obj");
    let obj = dir.join("p.o");
    write_object(PROGRAM, &obj, &CodegenOptions::default());
    let path = obj.to_str().unwrap();
    let Some(ro) = tool("llvm-readobj", &["-h", "-r", "-s", "--symbols", path]) else {
        eprintln!("skipping objects_carry_the_psabi_relocations: no llvm-readobj");
        return;
    };
    for needle in [
        "Machine: EM_RISCV",
        "EF_RISCV_FLOAT_ABI_DOUBLE",
        "R_RISCV_CALL_PLT step",
        "R_RISCV_PCREL_HI20 counter",
        "R_RISCV_PCREL_HI20 fp",
        "R_RISCV_PCREL_HI20 table",
        "R_RISCV_PCREL_HI20 step",
        "R_RISCV_PCREL_LO12_I .Lpcrel_hi",
        "R_RISCV_64 counter 0x0",
        "R_RISCV_64 step 0x0",
        "R_RISCV_64 table 0x8",
        ".rela.text",
        ".rela.rodata",
        ".rela.data",
    ] {
        assert!(ro.contains(needle), "`{needle}` missing:\n{ro}");
    }
    assert!(!ro.contains("R_RISCV_RELAX"), "no relaxation is requested");
    // Every low-part relocation names a local label sitting on an `auipc`
    // with a high-part relocation.
    let dis = tool("llvm-objdump", &["-dr", "--mattr=+m,+a,+f,+d", path]).unwrap();
    let lines: Vec<&str> = dis.lines().collect();
    let mut lo12 = 0;
    for (k, l) in lines.iter().enumerate() {
        if let Some(label) = l.split("R_RISCV_PCREL_LO12_I").nth(1).map(str::trim) {
            lo12 += 1;
            let at = lines.iter().position(|x| x.contains(&format!("<{label}>:"))).expect("the label");
            assert!(lines[at + 1].contains("auipc") && lines[at + 2].contains("R_RISCV_PCREL_HI20"), "{label}");
            assert!(k > at);
        }
    }
    assert!(lo12 >= 5, "{dis}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// qld links the object into a static executable; loaded into the
/// simulator, it runs: the PC-relative addresses, the PLT-free direct call,
/// the function pointers in `.data`/`.rodata` (`R_RISCV_64`) and the indirect
/// calls through them all resolve.
#[test]
fn qld_links_an_executable_that_runs() {
    let dir = scratch("exe");
    let (obj, exe) = (dir.join("p.o"), dir.join("p"));
    write_object(PROGRAM, &obj, &CodegenOptions::default());
    let bytes = qld_static(&[&obj], &exe, "entry");
    assert_eq!(u16::from_le_bytes([bytes[16], bytes[17]]), 2, "ET_EXEC");
    assert_eq!(u16::from_le_bytes([bytes[18], bytes[19]]), 243, "EM_RISCV");
    let image = load_elf(&bytes).unwrap();
    assert_eq!(u64::from_le_bytes(bytes[24..32].try_into().unwrap()), image.symbols["entry"], "e_entry");
    // step(10) = 15 (counter 15); step(15) = 30 (counter 30); step(30) = 60
    // (counter 60); 60 * 0.5 = 30, plus the self-pointer check.
    let (a0, _) = call(&image, "entry", &[10], &[]);
    assert_eq!(a0, 31);
    if let Some(ro) = tool("llvm-readobj", &["-h", "-l", exe.to_str().unwrap()]) {
        assert!(ro.contains("Type: Executable") && ro.contains("PT_LOAD"), "{ro}");
    }
    if let Some(dis) = tool("llvm-objdump", &["-d", "--mattr=+m,+a,+f,+d", exe.to_str().unwrap()]) {
        // The call resolved to the function itself.
        let line = dis.lines().find(|l| l.contains("jalr") && l.contains("<step>")).expect("a resolved call");
        assert!(line.contains("ra"), "{line}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `link_static_executable` (what `lf build --target riscv64-linux` does):
/// `_start` calls the entry and exits with its result through `ecall`.
#[test]
fn static_executables_start_and_exit() {
    let dir = scratch("start");
    let src = PROGRAM.replace("func @entry(i64) -> i64 {\nentry ^0(%x: i64):", "func @main() -> i64 {\nentry ^0:\n  %x = add i64 0, i64 10 : i64");
    let (m, syms) = parse(&src);
    let exe = dir.join("main");
    super::link_static_executable(&super::compile_module(&m, &syms), "main", exe.to_str().unwrap()).unwrap();
    let image = load_elf(&std::fs::read(&exe).unwrap()).unwrap();
    let mut cpu = Cpu::new(&image);
    let exited = std::rc::Rc::new(std::cell::Cell::new(None));
    let seen = exited.clone();
    cpu.syscall = Some(Box::new(move |nr, args| {
        seen.set(Some((nr, args[0])));
        Err("exited".into())
    }));
    let r = cpu.call(image.symbols["_start"], &[], &[], STACK_TOP - 4096);
    assert_eq!(r, Err(super::sim::Fault::Other("exited".into())));
    assert_eq!(exited.get(), Some((94, 31)), "exit_group(main())");
    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// Interoperation with C compiled by clang
// ===========================================================================

const INTEROP_C: &str = r#"
struct fi { float f; int i; };
struct dd { double a, b; };
struct big { long a, b, c; };
struct ll { long a, b; };
struct bd { signed char b; double d; };

double lf_fi(struct fi);
double lf_dd(struct dd);
struct dd lf_mk_dd(double);
struct fi lf_mk_fi(double, long);
struct big lf_mk_big(long);
double lf_many(double, double, double, double, double, double, double, double, double, float, long);
long lf_split(long, long, long, long, long, long, long, struct ll, long);
double lf_bd(struct bd);

double c_many(double a, double b, double c, double d, double e, double f, double g, double h,
              double i, float j, long k) { return a + 2 * b + 3 * h + 5 * i + 7 * j + k; }
double c_fi(struct fi s) { return s.f * 2 + s.i; }
struct dd c_mk_dd(double x) { struct dd r = { x + 1, x * 3 }; return r; }
struct big c_mk_big(long x) { struct big r = { x, x * 2, x * 3 }; return r; }
double c_vsum(int n, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, n);
    double s = 0;
    for (int k = 0; k < n; k++) s = s * 10 + __builtin_va_arg(ap, double);
    __builtin_va_end(ap);
    return s;
}
long c_split(long a, long b, long c, long d, long e, long f, long g, struct ll s, long z) {
    return a + b + c + d + e + f + g * 11 + s.a * 3 + s.b * 5 + z * 7;
}
double c_bd(struct bd s) { return s.d - s.b; }

double d_fi(double x, long i) { struct fi s = { (float)x, (int)i }; return lf_fi(s); }
double d_dd(double x) { struct dd s = { x, x / 4 }; return lf_dd(s); }
double d_mk_dd(double x) { struct dd r = lf_mk_dd(x); return r.a * 10 + r.b; }
double d_mk_fi(double x, long i) { struct fi r = lf_mk_fi(x, i); return r.f + r.i * 100.0; }
long d_mk_big(long x) { struct big r = lf_mk_big(x); return r.a + r.b * 3 + r.c * 7; }
double d_many(double x, long k) { return lf_many(x, x + 1, x, x, x, x, x, x + 2, x + 3, (float)x, k); }
long d_split(long x) { struct ll s = { x, -x }; return lf_split(1, 2, 3, 4, 5, 6, 7, s, x); }
double d_bd(double x) { struct bd s = { -5, x }; return lf_bd(s); }
"#;

const INTEROP_LF: &str = r#"
module "interop"
func @c_many(f64, f64, f64, f64, f64, f64, f64, f64, f64, f32, i64) -> f64
func @c_fi({f32, i32}) -> f64
func @c_mk_dd(f64) -> {f64, f64}
func @c_mk_big(i64) -> {i64, i64, i64}
func @c_vsum(i32, ...) -> f64
func @c_split(i64, i64, i64, i64, i64, i64, i64, {i64, i64}, i64) -> i64
func @c_bd({i8, f64}) -> f64

func @lf_fi({f32, i32}) -> f64 {
entry ^0(%s: ptr):
  %f = load %s align 4 : f32
  %p = ptr_add %s, i64 4 : ptr
  %i = load %p align 4 : i32
  %fd = fpext %f : f64
  %id = sitofp %i : f64
  %r = fsub %fd, %id : f64
  ret %r
}
func @lf_dd({f64, f64}) -> f64 {
entry ^0(%s: ptr):
  %a = load %s align 8 : f64
  %p = ptr_add %s, i64 8 : ptr
  %b = load %p align 8 : f64
  %t = fmul %b, f64 0x4024000000000000 : f64
  %r = fsub %a, %t : f64
  ret %r
}
func @lf_mk_dd(f64) -> {f64, f64} {
entry ^0(%x: f64):
  %s = alloca {f64, f64} : ptr
  %n = fneg %x : f64
  store %n, %s align 8 : f64
  %p = ptr_add %s, i64 8 : ptr
  %h = fmul %x, f64 0x3fe0000000000000 : f64
  store %h, %p align 8 : f64
  ret %s
}
func @lf_mk_fi(f64, i64) -> {f32, i32} {
entry ^0(%x: f64, %i: i64):
  %s = alloca {f32, i32} : ptr
  %f = fptrunc %x : f32
  store %f, %s align 4 : f32
  %p = ptr_add %s, i64 4 : ptr
  %t = trunc %i : i32
  %u = add %t, i32 -3 : i32
  store %u, %p align 4 : i32
  ret %s
}
func @lf_mk_big(i64) -> {i64, i64, i64} {
entry ^0(%x: i64):
  %s = alloca {i64, i64, i64} : ptr
  store %x, %s align 8 : i64
  %p1 = ptr_add %s, i64 8 : ptr
  %x1 = add %x, i64 1 : i64
  store %x1, %p1 align 8 : i64
  %p2 = ptr_add %s, i64 16 : ptr
  %x2 = mul %x, i64 -2 : i64
  store %x2, %p2 align 8 : i64
  ret %s
}
func @lf_many(f64, f64, f64, f64, f64, f64, f64, f64, f64, f32, i64) -> f64 {
entry ^0(%a: f64, %b: f64, %c: f64, %d: f64, %e: f64, %f: f64, %g: f64, %h: f64, %i: f64, %j: f32, %k: i64):
  %jd = fpext %j : f64
  %kd = sitofp %k : f64
  %s1 = fmul %b, f64 0x4000000000000000 : f64
  %s2 = fadd %a, %s1 : f64
  %s3 = fmul %i, f64 0x4014000000000000 : f64
  %s4 = fadd %s2, %s3 : f64
  %s5 = fadd %s4, %jd : f64
  %s6 = fsub %s5, %kd : f64
  %s7 = fadd %s6, %h : f64
  ret %s7
}
func @lf_split(i64, i64, i64, i64, i64, i64, i64, {i64, i64}, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64, %d: i64, %e: i64, %f: i64, %g: i64, %s: ptr, %z: i64):
  %x = load %s align 8 : i64
  %p = ptr_add %s, i64 8 : ptr
  %y = load %p align 8 : i64
  %t1 = mul %x, i64 3 : i64
  %t2 = mul %y, i64 5 : i64
  %t3 = mul %z, i64 7 : i64
  %t4 = mul %g, i64 11 : i64
  %u1 = add %t1, %t2 : i64
  %u2 = add %u1, %t3 : i64
  %u3 = add %u2, %t4 : i64
  %u4 = add %u3, %a : i64
  ret %u4
}
func @lf_bd({i8, f64}) -> f64 {
entry ^0(%s: ptr):
  %b = load %s align 1 : i8
  %p = ptr_add %s, i64 8 : ptr
  %d = load %p align 8 : f64
  %bd = sitofp %b : f64
  %r = fmul %d, %bd : f64
  ret %r
}

func @call_c_many(f64, i64) -> f64 {
entry ^0(%x: f64, %k: i64):
  %x1 = fadd %x, f64 0x3ff0000000000000 : f64
  %x2 = fadd %x, f64 0x4000000000000000 : f64
  %x3 = fadd %x, f64 0x4008000000000000 : f64
  %j = fptrunc %x : f32
  %r = call @c_many(%x, %x1, %x, %x, %x, %x, %x, %x2, %x3, %j, %k) : f64
  ret %r
}
func @call_c_fi(f64, i64) -> f64 {
entry ^0(%x: f64, %i: i64):
  %s = alloca {f32, i32} : ptr
  %f = fptrunc %x : f32
  store %f, %s align 4 : f32
  %p = ptr_add %s, i64 4 : ptr
  %t = trunc %i : i32
  store %t, %p align 4 : i32
  %r = call @c_fi(%s) : f64
  ret %r
}
func @call_c_mk_dd(f64) -> f64 {
entry ^0(%x: f64):
  %s = call @c_mk_dd(%x) : {f64, f64}
  %a = load %s align 8 : f64
  %p = ptr_add %s, i64 8 : ptr
  %b = load %p align 8 : f64
  %r = fsub %a, %b : f64
  ret %r
}
func @call_c_mk_big(i64) -> i64 {
entry ^0(%x: i64):
  %s = call @c_mk_big(%x) : {i64, i64, i64}
  %a = load %s align 8 : i64
  %p = ptr_add %s, i64 16 : ptr
  %c = load %p align 8 : i64
  %t = mul %c, i64 100 : i64
  %r = add %a, %t : i64
  ret %r
}
func @call_c_vsum(f64) -> f64 {
entry ^0(%x: f64):
  %y = fadd %x, f64 0x3ff0000000000000 : f64
  %z = fadd %x, f64 0x4000000000000000 : f64
  %r = call @c_vsum(i32 3, %x, %y, %z) : f64
  ret %r
}
func @call_c_split(i64) -> i64 {
entry ^0(%x: i64):
  %s = alloca {i64, i64} : ptr
  store %x, %s align 8 : i64
  %p = ptr_add %s, i64 8 : ptr
  %n = sub i64 0, %x : i64
  store %n, %p align 8 : i64
  %r = call @c_split(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, %s, %x) : i64
  ret %r
}
func @call_c_bd(f64) -> f64 {
entry ^0(%x: f64):
  %s = alloca {i8, f64} : ptr
  store i8 -5, %s align 1 : i8
  %p = ptr_add %s, i64 8 : ptr
  store %x, %p align 8 : f64
  %r = call @c_bd(%s) : f64
  ret %r
}
"#;

/// C (compiled by clang for LP64D) and our code call each other in both
/// directions, through a qld-linked executable run in the simulator.
#[test]
fn lp64d_interoperates_with_clang() {
    let dir = scratch("interop");
    let (c_src, c_obj, lf_obj, exe) = (dir.join("c.c"), dir.join("c.o"), dir.join("lf.o"), dir.join("x"));
    std::fs::write(&c_src, INTEROP_C).unwrap();
    let clang = tool(
        "clang",
        &[
            "--target=riscv64-unknown-linux-gnu",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-O1",
            "-fno-pic",
            "-mno-relax",
            "-ffreestanding",
            "-c",
            c_src.to_str().unwrap(),
            "-o",
            c_obj.to_str().unwrap(),
        ],
    );
    if clang.is_none() {
        eprintln!("skipping lp64d_interoperates_with_clang: no clang with a riscv64 target");
        return;
    }
    write_object(INTEROP_LF, &lf_obj, &CodegenOptions::default());
    let bytes = qld_static(&[&lf_obj, &c_obj], &exe, "d_fi");
    let image = load_elf(&bytes).unwrap();
    // clang's rv64gc code is mostly compressed: the simulator expands it.
    let mut cpu = Cpu::new(&image);
    cpu.call(image.symbols["d_many"], &[3], &[1.0f64.to_bits()], STACK_TOP - 4096).unwrap();
    assert!(cpu.compressed > 0, "clang's code ran compressed instructions");
    let d = |v: f64| v.to_bits();
    let fr = |r: (u64, u64)| f64::from_bits(r.1);

    // C calls us.
    for (x, i) in [(1.5f64, 7i64), (-2.25, -9), (1.0e6, 123_456)] {
        let xf = f64::from(x as f32);
        assert_eq!(fr(call(&image, "d_fi", &[i as u64], &[d(x)])), xf - i as f64, "d_fi");
        assert_eq!(fr(call(&image, "d_dd", &[], &[d(x)])), x - x / 4.0 * 10.0, "d_dd");
        assert_eq!(fr(call(&image, "d_mk_dd", &[], &[d(x)])), -x * 10.0 + x * 0.5, "d_mk_dd");
        assert_eq!(fr(call(&image, "d_mk_fi", &[i as u64], &[d(x)])), f64::from(x as f32) + (i - 3) as f64 * 100.0, "d_mk_fi");
        assert_eq!(call(&image, "d_mk_big", &[i as u64], &[]).0 as i64, i + (i + 1) * 3 + (-2 * i) * 7, "d_mk_big");
        let many = x + 2.0 * (x + 1.0) + 5.0 * (x + 3.0) + xf - i as f64 + (x + 2.0);
        assert_eq!(fr(call(&image, "d_many", &[i as u64], &[d(x)])), many, "d_many");
        assert_eq!(call(&image, "d_split", &[i as u64], &[]).0 as i64, i * 3 - i * 5 + i * 7 + 77 + 1, "d_split");
        assert_eq!(fr(call(&image, "d_bd", &[], &[d(x)])), x * -5.0, "d_bd");

        // We call C.
        let r = fr(call(&image, "call_c_many", &[i as u64], &[d(x)]));
        assert_eq!(r, x + 2.0 * (x + 1.0) + 3.0 * (x + 2.0) + 5.0 * (x + 3.0) + 7.0 * xf + i as f64, "c_many");
        assert_eq!(fr(call(&image, "call_c_fi", &[i as u64], &[d(x)])), xf * 2.0 + i as f64, "c_fi");
        assert_eq!(fr(call(&image, "call_c_mk_dd", &[], &[d(x)])), (x + 1.0) - x * 3.0, "c_mk_dd");
        assert_eq!(call(&image, "call_c_mk_big", &[i as u64], &[]).0 as i64, i + i * 3 * 100, "c_mk_big");
        assert_eq!(fr(call(&image, "call_c_vsum", &[], &[d(x)])), (x * 10.0 + (x + 1.0)) * 10.0 + (x + 2.0), "c_vsum");
        assert_eq!(
            call(&image, "call_c_split", &[i as u64], &[]).0 as i64,
            1 + 2 + 3 + 4 + 5 + 6 + 77 + i * 3 - i * 5 + i * 7,
            "c_split"
        );
        assert_eq!(fr(call(&image, "call_c_bd", &[], &[d(x)])), x + 5.0, "c_bd");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
