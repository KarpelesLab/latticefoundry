//! Unwind tables and Mach-O executables, checked against external tools.
//!
//! - **Windows x64**: `llvm-readobj --unwind` must decode every function's
//!   `.pdata`/`.xdata` in a COFF object, and its unwind codes must be exactly
//!   what an independent decode of the function's prologue bytes says (pushes,
//!   the fixed allocation, the frame register, the `xmm` saves) — for a leaf,
//!   callee-saved registers, `xmm6..15` saves, a large probed frame and
//!   `dyn_alloca`. A PE executable linked by qld carries them in its exception
//!   directory (`llvm-readobj`, `llvm-objdump --unwind-info`).
//! - **Mach-O executables and dylibs** from `lf build --target
//!   {x86_64,aarch64}-apple-darwin`: headers, load commands (`LC_MAIN` at
//!   `_main`, `LC_LOAD_DYLINKER`, `libSystem`, the arm64 code signature) and
//!   the compact unwind folded into `__unwind_info`. They cannot run here.
//! - **ELF `.eh_frame`**: `llvm-dwarfdump --eh-frame` rows against the decoded
//!   prologue, and `gdb`'s backtrace through LF frames of a PIE.
//!
//! A missing tool skips its check.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use latticefoundry::codegen::CodegenOptions;
use latticefoundry::ir::text;
use latticefoundry::mc::object::{ObjectModule, SymbolType, SymbolValue};
use latticefoundry::mc::write_object;
use latticefoundry::support::StrInterner;
use latticefoundry::support::diagnostics::FileId;
use latticefoundry::target::{self, TargetArch, TargetOs, Triple};

/// Frames of every shape: a leaf, callee-saved registers, xmm saves (live
/// floats across calls), a large probed frame and a dynamic allocation.
const SRC: &str = r#"
module "uw"
func @ext(f64) -> f64
func @leaf(i64) -> i64 {
entry ^0(%x: i64):
  ret %x
}
func @saved(i64, i64, i64) -> i64 {
entry ^0(%a: i64, %b: i64, %c: i64):
  %x = call @leaf(%a) : i64
  %y = call @leaf(%b) : i64
  %z = call @leaf(%c) : i64
  %s = add %x, %y : i64
  %t = add %s, %z : i64
  %u = add %t, %a : i64
  %v = add %u, %b : i64
  %w = add %v, %c : i64
  ret %w
}
func @xmm(f64, f64, f64) -> f64 {
entry ^0(%a: f64, %b: f64, %c: f64):
  %x = call @ext(%a) : f64
  %y = call @ext(%b) : f64
  %z = call @ext(%c) : f64
  %s = fadd %x, %y : f64
  %t = fadd %s, %z : f64
  %u = fadd %t, %a : f64
  %v = fadd %u, %b : f64
  %w = fadd %v, %c : f64
  ret %w
}
func @big(i64) -> i64 {
entry ^0(%n: i64):
  %a = alloca [13000 x i8] : ptr
  store i8 5, %a align 1 : i8
  %v = load %a align 1 : i8
  %r = zext %v : i64
  %c = call @leaf(%r) : i64
  %d = call @saved(%c, %n, %r) : i64
  %e = add %d, %n : i64
  %f = add %e, %c : i64
  ret %f
}
func @dyn(i64) -> i64 {
entry ^0(%n: i64):
  %p = dyn_alloca %n align 16 : ptr
  store i64 1, %p align 8 : i64
  %v = load %p align 8 : i64
  %c = call @leaf(%v) : i64
  ret %c
}
func @main() -> i64 {
entry ^0:
  %r = call @saved(i64 1, i64 2, i64 3) : i64
  %b = call @big(%r) : i64
  %d = call @dyn(i64 32) : i64
  %s = add %b, %d : i64
  ret %s
}
"#;

const FUNCS: [&str; 6] = ["leaf", "saved", "xmm", "big", "dyn", "main"];

fn compile(os: TargetOs, opts: CodegenOptions) -> ObjectModule {
    let mut syms = StrInterner::new();
    let m = text::parse_module(SRC, FileId::new(0), &mut syms).expect("parse");
    target::x86_64::compile_module_with(&m, &syms, &opts.with_os(os)).object
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lf-unwind-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Find an LLVM tool on `PATH` or in `/usr/lib/llvm/*/bin`.
fn llvm_tool(name: &str) -> Option<PathBuf> {
    let on_path = Command::new(name).arg("--version").output().is_ok_and(|o| o.status.success());
    if on_path {
        return Some(PathBuf::from(name));
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir("/usr/lib/llvm")
        .ok()?
        .flatten()
        .map(|e| e.path().join("bin").join(name))
        .filter(|p| p.is_file())
        .collect();
    dirs.sort();
    dirs.pop()
}

/// Run `tool args.. file`, returning stdout, or `None` (skip) when the tool
/// is missing. A tool that rejects the file fails the test.
fn inspect(tool: &str, args: &[&str], file: &Path) -> Option<String> {
    let Some(path) = llvm_tool(tool) else {
        eprintln!("skipping {tool} check: not installed");
        return None;
    };
    let out = Command::new(&path).args(args).arg(file).output().expect("run tool");
    assert!(
        out.status.success(),
        "{tool} {args:?} rejected {}: {}",
        file.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Each defined function's `(offset, bytes)` in `.text`, by name.
fn functions(obj: &ObjectModule) -> BTreeMap<String, (u64, Vec<u8>)> {
    let mut out = BTreeMap::new();
    for s in obj.symbols() {
        if let SymbolValue::Defined { section, offset } = s.value
            && s.kind == SymbolType::Func
        {
            let bytes = &obj.section(section).bytes[offset as usize..(offset + s.size) as usize];
            out.insert(s.name.clone(), (offset, bytes.to_vec()));
        }
    }
    out
}

const GPR: [&str; 16] = [
    "RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15",
];

/// One unwind code, normalized: `(prologue offset, operation, register, value)`.
type Code = (u32, String, String, u64);

/// Decode a Windows-shape prologue from its bytes, independently of the
/// compiler: the unwind codes it needs (in prologue order) and the prologue
/// size (the end of the last instruction that needs one).
fn decode_win_prologue(code: &[u8]) -> (Vec<Code>, u32) {
    let mut out: Vec<Code> = Vec::new();
    let mut pc = 0usize;
    let mut frame_off: Option<u64> = None;
    let mut end = 0;
    loop {
        let c = &code[pc..];
        let (len, item) = if c[0] == 0x55 || (0x50..=0x57).contains(&c[0]) {
            (1, Some(("PUSH_NONVOL", GPR[usize::from(c[0] - 0x50)].to_owned(), 0)))
        } else if c[0] == 0x41 && (0x50..=0x57).contains(&c[1]) {
            (2, Some(("PUSH_NONVOL", GPR[usize::from(c[1] - 0x50 + 8)].to_owned(), 0)))
        } else if c.starts_with(&[0x48, 0x81, 0xEC]) && frame_off.is_none() {
            let n = u64::from(u32::from_le_bytes(c[3..7].try_into().unwrap()));
            let op = if n <= 128 { "ALLOC_SMALL" } else { "ALLOC_LARGE" };
            (7, Some((op, String::new(), n)))
        } else if c.starts_with(&[0x48, 0x89, 0xE5]) {
            frame_off = Some(0);
            (3, Some(("SET_FPREG", "RBP".to_owned(), 0)))
        } else if c.starts_with(&[0x48, 0x8D, 0x6C, 0x24]) {
            frame_off = Some(u64::from(c[4]));
            (5, Some(("SET_FPREG", "RBP".to_owned(), u64::from(c[4]))))
        } else if let Some(at) = [0usize, 1].into_iter().find(|&p| {
            (p == 0 || c[0] == 0x44) && c[p..].starts_with(&[0x0F, 0x11]) && c[p + 2] & 7 == 5 && c[p + 2] >> 6 != 0
        }) {
            // movups [rbp + disp], xmm: offset from the frame base rbp - frame_off.
            let modrm = c[at + 2];
            let reg = ((modrm >> 3) & 7) + if at == 1 { 8 } else { 0 };
            let (disp, dl) = if modrm >> 6 == 1 {
                (i64::from(c[at + 3] as i8), 1)
            } else {
                (i64::from(i32::from_le_bytes(c[at + 3..at + 7].try_into().unwrap())), 4)
            };
            let off = (disp + frame_off.expect("xmm save after the frame register") as i64) as u64;
            (at + 3 + dl, Some(("SAVE_XMM128", format!("XMM{reg}"), off)))
        } else {
            return (out, end);
        };
        pc += len;
        if let Some((op, reg, v)) = item {
            end = pc as u32;
            out.push((end, op.to_owned(), reg, v));
        }
    }
}

/// One `RuntimeFunction` from `llvm-readobj --unwind`. In an object, the
/// addresses are relocations, printed `symbol +0xN (field offset)`; in an
/// image, `(address)`.
#[derive(Debug, Default)]
struct RuntimeFunction {
    start_sym: String,
    start: u64,
    end_plus: u64,
    end: u64,
    prolog: u32,
    frame_reg: String,
    frame_off: u64,
    codes: Vec<Code>,
}

fn hex(s: &str) -> u64 {
    let s = s.trim().trim_start_matches("0x");
    u64::from_str_radix(s, 16).unwrap_or_else(|_| panic!("hex {s}"))
}

/// The value inside the last `(0x...)` of a line.
fn paren_hex(line: &str) -> u64 {
    let open = line.rfind("(0x").expect("(0x");
    hex(&line[open + 1..line.rfind(')').unwrap()])
}

/// Parse `llvm-readobj --unwind` output.
fn parse_readobj_unwind(out: &str) -> Vec<RuntimeFunction> {
    let mut funcs = Vec::new();
    let mut cur: Option<RuntimeFunction> = None;
    for line in out.lines().map(str::trim) {
        if line.starts_with("RuntimeFunction {") {
            if let Some(f) = cur.take() {
                funcs.push(f);
            }
            cur = Some(RuntimeFunction::default());
        }
        let Some(f) = cur.as_mut() else { continue };
        if let Some(v) = line.strip_prefix("StartAddress:") {
            f.start = paren_hex(v);
            f.start_sym = v.split_whitespace().next().unwrap().to_owned();
        } else if let Some(v) = line.strip_prefix("EndAddress:") {
            f.end = paren_hex(v);
            f.end_plus = v.split_whitespace().find_map(|w| w.strip_prefix('+')).map_or(0, hex);
        } else if let Some(v) = line.strip_prefix("PrologSize:") {
            f.prolog = v.trim().parse().unwrap();
        } else if let Some(v) = line.strip_prefix("FrameRegister:") {
            f.frame_reg = v.split_whitespace().next().unwrap().to_owned();
        } else if let Some(v) = line.strip_prefix("FrameOffset:") {
            f.frame_off = hex(v);
        } else if line.starts_with("0x") && line.contains(": ") {
            // `0x15: SET_FPREG reg=RBP, offset=0x30`, `0x10: ALLOC_SMALL size=8`
            let (at, rest) = line.split_once(": ").unwrap();
            let mut words = rest.split_whitespace();
            let op = words.next().unwrap().to_owned();
            let (mut reg, mut value) = (String::new(), 0);
            for w in words {
                let w = w.trim_end_matches(',');
                if let Some(r) = w.strip_prefix("reg=") {
                    reg = r.to_owned();
                } else if let Some(v) = w.strip_prefix("offset=") {
                    value = hex(v);
                } else if let Some(v) = w.strip_prefix("size=") {
                    value = v.parse().unwrap();
                }
            }
            f.codes.push((hex(at) as u32, op, reg, value));
        }
    }
    funcs.extend(cur);
    funcs
}

#[test]
fn win64_unwind_codes_match_the_prologues() {
    let triple = Triple::new(TargetArch::X86_64, TargetOs::Windows);
    let obj = compile(TargetOs::Windows, CodegenOptions::default());
    let funcs = functions(&obj);
    assert_eq!(funcs.len(), FUNCS.len());
    let dir = scratch("coff");
    let path = dir.join("uw.obj");
    std::fs::write(&path, write_object(&obj, triple).expect("COFF")).unwrap();
    let Some(out) = inspect("llvm-readobj", &["--unwind"], &path) else { return };
    let rfs = parse_readobj_unwind(&out);
    assert_eq!(rfs.len(), FUNCS.len(), "one RUNTIME_FUNCTION per function:\n{out}");

    let mut saw = BTreeMap::new();
    for (name, (_, bytes)) in &funcs {
        let rf = rfs.iter().find(|r| r.start_sym == *name).unwrap_or_else(|| panic!("no entry for {name}:\n{out}"));
        assert_eq!(rf.end_plus, bytes.len() as u64, "{name}: end address");
        let (mut want, size) = decode_win_prologue(bytes);
        assert_eq!(rf.prolog, size, "{name}: prologue size");
        assert_eq!((rf.frame_reg.as_str(), rf.frame_off * 16), ("RBP", want.iter().find(|c| c.1 == "SET_FPREG").unwrap().3), "{name}");
        want.reverse();
        assert_eq!(rf.codes, want, "{name}: codes vs the decoded prologue");
        for c in &want {
            *saw.entry(c.1.clone()).or_insert(0) += 1;
        }
        // Everything below the frame register — the probed rest of a large
        // frame, a dyn_alloca — is unwound through rbp: no code after the
        // prologue.
        if name == "big" {
            let after = &bytes[size as usize..];
            assert!(after.windows(5).any(|w| w == [0x48, 0x83, 0x0C, 0x24, 0x00]), "big is probed");
        }
    }
    for op in ["PUSH_NONVOL", "SET_FPREG", "ALLOC_SMALL", "SAVE_XMM128"] {
        assert!(saw.contains_key(op), "the fixtures exercise {op}: {saw:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn win64_unwind_tables_reach_the_pe_exception_directory() {
    let dir = scratch("pe");
    let src = dir.join("uw.lf");
    // `ext` defined in a second module so the program links.
    std::fs::write(&src, SRC).unwrap();
    let ext = dir.join("ext.lf");
    std::fs::write(&ext, "module \"e\"\nfunc @ext(f64) -> f64 {\nentry ^0(%a: f64):\n  ret %a\n}\n").unwrap();
    let exe = dir.join("uw.exe");
    let st = Command::new(env!("CARGO_BIN_EXE_lf"))
        .args(["build", "--target", "x86_64-windows", "-o"])
        .args([&exe, &src, &ext])
        .status()
        .unwrap();
    assert!(st.success());
    if let Some(out) = inspect("llvm-readobj", &["--file-headers", "--unwind"], &exe) {
        let n = out.matches("RuntimeFunction {").count();
        assert_eq!(n, FUNCS.len() + 1, "{out}");
        let size = out.lines().find_map(|l| l.trim().strip_prefix("ExceptionTableSize:")).expect("exception directory");
        assert_eq!(hex(size), 12 * n as u64);
        for f in parse_readobj_unwind(&out) {
            assert!(f.end > f.start && f.frame_reg == "RBP", "{f:?}");
        }
    }
    if let Some(out) = inspect("llvm-objdump", &["--unwind-info"], &exe) {
        assert!(out.contains("UOP_SetFPReg") && out.contains("UOP_SaveXMM128") && out.contains("UOP_PushNonVol"), "{out}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Mach-O
// ---------------------------------------------------------------------------

/// `lf build <dir>/uw.lf args..`, asserting success.
fn lf_build(dir: &Path, args: &[&str]) {
    let src = dir.join("uw.lf");
    std::fs::write(&src, SRC).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_lf")).arg("build").arg(&src).args(args).output().unwrap();
    assert!(out.status.success(), "lf build {args:?}: {}", String::from_utf8_lossy(&out.stderr));
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// The load commands of a Mach-O image: `(cmd, offset)`.
fn load_commands(b: &[u8]) -> Vec<(u32, usize)> {
    let mut out = Vec::new();
    let mut o = 32;
    for _ in 0..u32_at(b, 16) {
        out.push((u32_at(b, o), o));
        o += u32_at(b, o + 4) as usize;
    }
    out
}

#[test]
fn macho_executables_and_dylibs() {
    const LC_MAIN: u32 = 0x8000_0028;
    const LC_LOAD_DYLINKER: u32 = 0xe;
    const LC_LOAD_DYLIB: u32 = 0xc;
    const LC_ID_DYLIB: u32 = 0xd;
    const LC_CODE_SIGNATURE: u32 = 0x1d;
    for (triple, arch, cpu, encoding) in [
        ("x86_64-apple-darwin", "x86_64", 0x0100_0007u32, "0x01000000"),
        ("aarch64-apple-darwin", "arm64", 0x0100_000c, "0x04000000"),
    ] {
        let dir = scratch(&format!("macho-{arch}"));
        let exe = dir.join("uw");
        lf_build(&dir, &["--target", triple, "-o", exe.to_str().unwrap()]);
        let b = std::fs::read(&exe).unwrap();
        assert_eq!((u32_at(&b, 0), u32_at(&b, 4), u32_at(&b, 12)), (0xfeed_facf, cpu, 2), "{arch}: MH_EXECUTE");
        let cmds = load_commands(&b);
        let has = |c: u32| cmds.iter().any(|&(k, _)| k == c);
        assert!(has(LC_MAIN) && has(LC_LOAD_DYLINKER) && has(LC_LOAD_DYLIB), "{arch}: {cmds:x?}");
        assert_eq!(has(LC_CODE_SIGNATURE), arch == "arm64", "{arch}: arm64 is signed");
        let at = cmds.iter().find(|&&(k, _)| k == LC_MAIN).unwrap().1;
        let entryoff = u64::from_le_bytes(b[at + 8..at + 16].try_into().unwrap());

        if let Some(out) = inspect("llvm-objdump", &["--macho", "--private-headers"], &exe) {
            assert!(out.contains("name /usr/lib/dyld"), "{out}");
            assert!(out.contains("name /usr/lib/libSystem.B.dylib"), "{out}");
            assert!(out.contains(&format!("entryoff {entryoff}")), "{out}");
        }
        if let Some(out) = inspect("llvm-readobj", &["--file-headers", "--macho-dysymtab"], &exe) {
            assert!(out.contains("FileType: Executable (0x2)") && out.contains("MH_PIE"), "{out}");
        }
        // LC_MAIN's entry is `_main` (its address less the __TEXT base).
        if let Some(out) = inspect("llvm-nm", &[], &exe) {
            let main = out.lines().find(|l| l.ends_with(" T _main")).expect("_main");
            assert_eq!(hex(main.split_whitespace().next().unwrap()) - 0x1_0000_0000, entryoff, "{out}");
        }
        // The compact unwind records became `__unwind_info`.
        if let Some(out) = inspect("llvm-objdump", &["--macho", "--unwind-info"], &exe) {
            assert!(out.contains(&format!("encoding[0]: {encoding}")), "{arch}: {out}");
        }

        // --shared: an MH_DYLIB with its install name.
        let lib = dir.join("libuw.dylib");
        lf_build(&dir, &["--target", triple, "--shared", "-o", lib.to_str().unwrap()]);
        let d = std::fs::read(&lib).unwrap();
        assert_eq!(u32_at(&d, 12), 6, "MH_DYLIB");
        let cmds = load_commands(&d);
        let id = cmds.iter().find(|&&(k, _)| k == LC_ID_DYLIB).expect("LC_ID_DYLIB").1;
        let name_at = id + u32_at(&d, id + 8) as usize;
        assert!(d[name_at..].starts_with(b"@rpath/libuw.dylib\0"));
        if let Some(out) = inspect("llvm-objdump", &["--macho", "--exports-trie"], &lib) {
            for f in FUNCS {
                assert!(out.contains(&format!("_{f}")), "{out}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ---------------------------------------------------------------------------
// ELF .eh_frame
// ---------------------------------------------------------------------------

/// The System V prologue's pushed callee-saved registers, decoded from bytes
/// (`push rbp; mov rbp, rsp; push ...`).
fn sysv_pushes(code: &[u8]) -> Vec<&'static str> {
    assert_eq!(&code[..4], &[0x55, 0x48, 0x89, 0xE5]);
    let mut pc = 4;
    let mut out = Vec::new();
    loop {
        if (0x50..=0x57).contains(&code[pc]) {
            out.push(GPR[usize::from(code[pc] - 0x50)]);
            pc += 1;
        } else if code[pc] == 0x41 && (0x50..=0x57).contains(&code[pc + 1]) {
            out.push(GPR[usize::from(code[pc + 1] - 0x50 + 8)]);
            pc += 2;
        } else {
            return out;
        }
    }
}

#[test]
fn eh_frame_matches_the_prologues() {
    use latticefoundry::codegen::UnwindTables;
    let obj = compile(TargetOs::Linux, CodegenOptions::default().with_unwind_tables(UnwindTables::EhFrame));
    let dir = scratch("eh");
    let path = dir.join("uw.o");
    std::fs::write(&path, latticefoundry::mc::elf::write(&obj)).unwrap();
    let Some(out) = inspect("llvm-dwarfdump", &["--eh-frame"], &path) else { return };
    let funcs = functions(&obj);
    assert_eq!(out.matches(" FDE ").count(), funcs.len(), "{out}");
    for (name, (offset, bytes)) in &funcs {
        // This function's FDE: `pc=<start>...<end>`.
        let tag = format!("pc={offset:08x}...{:08x}", offset + bytes.len() as u64);
        let fde = out.split(" FDE ").find(|f| f.contains(&tag)).unwrap_or_else(|| panic!("{name}: {tag}\n{out}"));
        assert!(fde.contains("CFA=RSP+16: RBP=[CFA-16]"), "{name}: {fde}");
        // In the body the CFA is rbp-based and every push is recorded in
        // order below the saved rbp.
        let mut want = String::from("CFA=RBP+16: RBP=[CFA-16]");
        for (k, r) in sysv_pushes(bytes).iter().enumerate() {
            want += &format!(", {r}=[CFA-{}]", 24 + 8 * k);
        }
        want += ", RIP=[CFA-8]";
        assert!(fde.contains(&want), "{name}: want {want}\n{fde}");
        // After each epilogue's pop rbp, back on rsp.
        assert!(fde.contains("DW_CFA_remember_state") && fde.contains(": CFA=RSP+8: "), "{name}: {fde}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gdb_backtraces_through_lf_frames() {
    let gdb = Command::new("gdb").arg("--version").output().is_ok_and(|o| o.status.success());
    if !gdb || latticefoundry::link::gnu::HostCrt::discover().is_none() {
        eprintln!("skipping: gdb or the host C runtime is missing");
        return;
    }
    let dir = scratch("gdb");
    let src = dir.join("bt.lf");
    std::fs::write(
        &src,
        r#"module "bt"
func @abort() -> void
func @inner(i64) -> i64 {
entry ^0(%n: i64):
  %p = dyn_alloca %n align 16 : ptr
  store i64 7, %p align 8 : i64
  call @abort() : void
  %v = load %p align 8 : i64
  ret %v
}
func @middle(i64) -> i64 {
entry ^0(%n: i64):
  %a = alloca [20000 x i8] : ptr
  store i8 1, %a align 1 : i8
  %r = call @inner(%n) : i64
  %s = add %r, %n : i64
  ret %s
}
func @main() -> i32 {
entry ^0:
  %r = call @middle(i64 64) : i64
  %t = trunc %r : i32
  ret %t
}
"#,
    )
    .unwrap();
    let exe = dir.join("bt");
    let st = Command::new(env!("CARGO_BIN_EXE_lf")).args(["build", "--pie", "-o"]).args([&exe, &src]).status().unwrap();
    assert!(st.success());
    if let Some(out) = inspect("llvm-readelf", &["-S"], &exe) {
        assert!(out.contains(".eh_frame_hdr") && out.contains(".eh_frame "), "{out}");
    }
    let out = Command::new("gdb").args(["-batch", "-nx", "-ex", "run", "-ex", "bt"]).arg(&exe).output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    let pos = |f: &str| s.find(&format!(" in {f} ()")).unwrap_or_else(|| panic!("no {f} in the backtrace:\n{s}"));
    assert!(pos("abort") < pos("inner") && pos("inner") < pos("middle") && pos("middle") < pos("main"), "{s}");
    let _ = std::fs::remove_dir_all(&dir);
}
