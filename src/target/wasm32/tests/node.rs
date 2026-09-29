//! Running wasm modules under **node** (skipped when it is not installed).
//!
//! A test writes the module and a list of calls to a scratch directory and runs
//! one node process over them: it instantiates the module with
//! `WebAssembly.instantiate` (providing the `env` host functions, mirrored in
//! [`super::refinterp`]), calls each export, and prints every result's bits in
//! hex (or the trap).

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

/// One call: the export, its arguments as `(wasm type, hex bits)`, and the
/// wasm types of its results.
#[derive(Clone, Debug)]
pub(crate) struct Call {
    pub(crate) func: String,
    pub(crate) args: Vec<(&'static str, u64)>,
    pub(crate) rets: Vec<&'static str>,
}

/// The outcome of one call: each result's bits, or the trap message.
pub(crate) type Outcome = Result<Vec<u64>, String>;

const RUNNER: &str = r#"
const fs = require('fs');
const [,, wasmPath, callsPath, mode] = process.argv;
const bytes = fs.readFileSync(wasmPath);
const calls = mode === 'validate' ? [] : JSON.parse(fs.readFileSync(callsPath, 'utf8'));
const host = {
  fmod: (a, b) => a % b,
  fmodf: (a, b) => Math.fround(a % b),
  host_mul3: (x) => (Math.imul(x, 3) + 1) | 0,
  host_i64: (x) => BigInt.asIntN(64, x * 3n + 1n),
  host_sloppy8: () => 0x1234 + 0x700,
  host_half: (x) => x * 0.5,
  host_void: () => {},
};
const buf = new DataView(new ArrayBuffer(8));
function toJs(t, hex) {
  const b = BigInt('0x' + hex);
  switch (t) {
    case 'i32': return Number(BigInt.asIntN(32, b));
    case 'i64': return BigInt.asIntN(64, b);
    case 'f32': buf.setUint32(0, Number(b), true); return buf.getFloat32(0, true);
    case 'f64': buf.setBigUint64(0, b, true); return buf.getFloat64(0, true);
  }
}
function toHex(t, v) {
  switch (t) {
    case 'i32': return (v >>> 0).toString(16);
    case 'i64': return BigInt.asUintN(64, v).toString(16);
    case 'f32': buf.setFloat32(0, v, true); return buf.getUint32(0, true).toString(16);
    case 'f64': buf.setFloat64(0, v, true); return buf.getBigUint64(0, true).toString(16);
  }
}
(async () => {
  if (mode === 'validate') {
    console.log(WebAssembly.validate(bytes) ? 'valid' : 'invalid');
    return;
  }
  const mod = await WebAssembly.compile(bytes);
  const env = {};
  for (const imp of WebAssembly.Module.imports(mod)) {
    if (imp.kind !== 'function') continue;
    env[imp.name] = host[imp.name] || (() => { throw new Error('unexpected import ' + imp.name); });
  }
  const inst = await WebAssembly.instantiate(mod, { env });
  const out = [];
  for (const c of calls) {
    try {
      const f = inst.exports[c.func];
      if (!f) throw new Error('no export ' + c.func);
      let r = f(...c.args.map(([t, h]) => toJs(t, h)));
      if (c.rets.length === 0) { out.push('ok'); continue; }
      if (c.rets.length === 1) r = [r];
      out.push('ok ' + c.rets.map((t, i) => toHex(t, r[i])).join(' '));
    } catch (e) {
      out.push('trap ' + String(e.message || e).replace(/\n/g, ' '));
    }
  }
  console.log(out.join('\n'));
})().catch((e) => { console.log('fatal ' + String(e.message || e).replace(/\n/g, ' ')); });
"#;

/// The node binary, if installed.
pub(crate) fn node() -> Option<PathBuf> {
    let out = Command::new("node").arg("--version").output().ok()?;
    out.status.success().then(|| PathBuf::from("node"))
}

/// A fresh scratch directory for one test.
pub(crate) fn scratch(tag: &str) -> PathBuf {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("lf-wasm32-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Instantiate `wasm` under node and run `calls`. `None` without node.
///
/// # Panics
///
/// If the module fails to compile or instantiate.
pub(crate) fn run(tag: &str, wasm: &[u8], calls: &[Call]) -> Option<Vec<Outcome>> {
    let node = node()?;
    let dir = scratch(tag);
    let wasm_path = dir.join("m.wasm");
    let calls_path = dir.join("calls.json");
    let runner = dir.join("run.js");
    std::fs::write(&wasm_path, wasm).unwrap();
    std::fs::write(&runner, RUNNER).unwrap();
    let mut json = String::from("[");
    for (i, c) in calls.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let args: Vec<String> = c.args.iter().map(|(t, v)| format!("[{},\"{v:x}\"]", json_str(t))).collect();
        let rets: Vec<String> = c.rets.iter().map(|t| json_str(t)).collect();
        json.push_str(&format!("{{\"func\":{},\"args\":[{}],\"rets\":[{}]}}", json_str(&c.func), args.join(","), rets.join(",")));
    }
    json.push(']');
    std::fs::write(&calls_path, json).unwrap();
    let out = Command::new(node).arg(&runner).arg(&wasm_path).arg(&calls_path).output().expect("run node");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "node failed ({}): {text}{}", out.status, String::from_utf8_lossy(&out.stderr));
    assert!(!text.starts_with("fatal"), "module rejected by node: {text} (saved in {})", wasm_path.display());
    let results: Vec<Outcome> = text
        .lines()
        .map(|l| {
            if let Some(rest) = l.strip_prefix("trap ") {
                Err(rest.to_owned())
            } else {
                let rest = l.strip_prefix("ok").expect("ok or trap").trim();
                Ok(rest.split_whitespace().map(|h| u64::from_str_radix(h, 16).expect("hex")).collect())
            }
        })
        .collect();
    assert_eq!(results.len(), calls.len(), "node output: {text}");
    let _ = std::fs::remove_dir_all(&dir);
    Some(results)
}

/// Whether node's `WebAssembly.validate` accepts `wasm` (`None` without node).
pub(crate) fn validate(tag: &str, wasm: &[u8]) -> Option<bool> {
    let node = node()?;
    let dir = scratch(tag);
    let wasm_path = dir.join("m.wasm");
    let runner = dir.join("run.js");
    std::fs::write(&wasm_path, wasm).unwrap();
    std::fs::write(&runner, RUNNER).unwrap();
    let out = Command::new(node).arg(&runner).arg(&wasm_path).arg("-").arg("validate").output().expect("run node");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let _ = std::fs::remove_dir_all(&dir);
    Some(text.trim() == "valid")
}
