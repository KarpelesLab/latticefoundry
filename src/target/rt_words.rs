//! A small fixed-width (32-bit word) assembler shared by the AArch64 and RISC-V
//! context-switching runtimes ([`super::aarch64::runtime`],
//! [`super::riscv::runtime`]).
//!
//! Each instruction is pushed as its encoded word **together with its assembly
//! text**, so tests can assemble the text with `llvm-mc` and compare bytes
//! (the encoder gate), and PC-relative branches to local labels are patched
//! once every label is bound.

use crate::mc::object::{ObjectModule, SectionKind, Symbol, SymbolBinding, SymbolType};

/// A local label inside a [`WordAsm`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct WLabel(usize);

/// Patches a PC-relative displacement (in bytes, target − instruction) into a
/// word.
pub(crate) type Patch = fn(u32, i64) -> u32;

/// One emitted routine: name, byte range, and global/local binding.
#[derive(Clone, Debug)]
pub(crate) struct Routine {
    pub(crate) name: &'static str,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) global: bool,
}

/// Words + assembly text + label fixups.
#[derive(Debug, Default)]
pub(crate) struct WordAsm {
    pub(crate) words: Vec<u32>,
    /// One line per word, or a `name:` label line (not a word).
    pub(crate) text: Vec<String>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, WLabel, Patch)>,
    pub(crate) routines: Vec<Routine>,
}

impl WordAsm {
    pub(crate) fn new() -> WordAsm {
        WordAsm::default()
    }
    /// Append `word`, spelled `text` in assembly.
    pub(crate) fn emit(&mut self, word: u32, text: String) {
        self.words.push(word);
        self.text.push(format!("  {text}"));
    }
    pub(crate) fn label(&mut self) -> WLabel {
        self.labels.push(None);
        WLabel(self.labels.len() - 1)
    }
    /// The assembly name of `l`.
    pub(crate) fn name(l: WLabel) -> String {
        format!(".Lrt{}", l.0)
    }
    pub(crate) fn bind(&mut self, l: WLabel) {
        self.labels[l.0] = Some(self.words.len());
        self.text.push(format!("{}:", Self::name(l)));
    }
    /// Append a word whose displacement to `target` is patched later.
    pub(crate) fn emit_ref(&mut self, word: u32, text: String, target: WLabel, patch: Patch) {
        self.fixups.push((self.words.len(), target, patch));
        self.emit(word, text);
    }
    /// Start a routine (closing the previous one).
    pub(crate) fn begin(&mut self, name: &'static str, global: bool) {
        self.close();
        let start = self.words.len() * 4;
        self.routines.push(Routine { name, start, end: start, global });
        self.text.push(format!("{name}:"));
    }
    fn close(&mut self) {
        let here = self.words.len() * 4;
        if let Some(r) = self.routines.last_mut()
            && r.end == r.start
        {
            r.end = here;
        }
    }
    /// Resolve every fixup; return the little-endian bytes.
    pub(crate) fn finish(&mut self) -> Vec<u8> {
        self.close();
        for &(at, l, patch) in &self.fixups {
            let target = self.labels[l.0].expect("runtime label bound");
            let delta = (target as i64 - at as i64) * 4;
            self.words[at] = patch(self.words[at], delta);
        }
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
    /// The whole program as assembly text (after [`finish`](Self::finish)).
    #[cfg(test)]
    pub(crate) fn listing(&self) -> String {
        let mut s = String::from("  .text\n");
        for l in &self.text {
            s.push_str(l);
            s.push('\n');
        }
        s
    }
    /// Append the finished code to `obj` as `.text.lf_rt` with its symbols.
    pub(crate) fn into_object(mut self, obj: &mut ObjectModule) {
        let bytes = self.finish();
        let mut sec = crate::mc::object::Section::new(".text.lf_rt", SectionKind::Text, 16);
        sec.bytes = bytes;
        let sid = obj.add_section(sec);
        for r in &self.routines {
            let binding = if r.global { SymbolBinding::Global } else { SymbolBinding::Local };
            obj.add_symbol(Symbol::defined(
                r.name,
                binding,
                SymbolType::Func,
                sid,
                r.start as u64,
                (r.end - r.start) as u64,
            ));
        }
    }
}

/// Assemble `listing` with `llvm-mc` for `triple` (plus `-mattr=mattr` when
/// non-empty) and return the `.text` bytes; `None` when `llvm-mc` or
/// `llvm-objcopy` is unavailable.
#[cfg(test)]
pub(crate) fn llvm_mc_text(triple: &str, mattr: &str, listing: &str, tag: &str) -> Option<Vec<u8>> {
    let dir = std::env::temp_dir();
    let stem = format!("lf_rtw_{tag}_{}", std::process::id());
    let src = dir.join(format!("{stem}.s"));
    let obj = dir.join(format!("{stem}.o"));
    let bin = dir.join(format!("{stem}.bin"));
    std::fs::write(&src, listing).ok()?;
    let mut mc = std::process::Command::new("llvm-mc");
    mc.arg(format!("--triple={triple}")).arg("-filetype=obj").arg("-o").arg(&obj).arg(&src);
    if !mattr.is_empty() {
        mc.arg(format!("-mattr={mattr}"));
    }
    let status = mc.status();
    let _ = std::fs::remove_file(&src);
    let Ok(status) = status else { return None };
    if !status.success() {
        let _ = std::fs::remove_file(&obj);
        panic!("llvm-mc rejected the listing:\n{listing}");
    }
    let st = std::process::Command::new("llvm-objcopy")
        .args(["-O", "binary", "--only-section=.text"])
        .arg(&obj)
        .arg(&bin)
        .status();
    let _ = std::fs::remove_file(&obj);
    if !st.ok()?.success() {
        return None;
    }
    let bytes = std::fs::read(&bin).ok();
    let _ = std::fs::remove_file(&bin);
    bytes
}
