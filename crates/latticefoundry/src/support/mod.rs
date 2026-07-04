//! Foundational support utilities: interning and small ADTs.
//!
//! These are the low-level building blocks the rest of the framework relies
//! on. See ROADMAP Phase 0.

use std::collections::HashMap;

/// A cheap, copyable handle into an interning table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Sym(u32);

impl Sym {
    /// The dense index backing this symbol.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A deduplicating string interner.
///
/// Interning turns repeated identifiers (type names, symbol names, ...) into
/// cheap `Copy` [`Sym`] handles that compare and hash in constant time.
#[derive(Debug, Default)]
pub struct StrInterner {
    map: HashMap<String, Sym>,
    strs: Vec<String>,
}

impl StrInterner {
    /// Create an empty interner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern `s`, returning a stable handle. Equal strings intern to equal
    /// handles.
    pub fn intern(&mut self, s: &str) -> Sym {
        if let Some(&sym) = self.map.get(s) {
            return sym;
        }
        let sym = Sym(self.strs.len() as u32);
        self.strs.push(s.to_owned());
        self.map.insert(s.to_owned(), sym);
        sym
    }

    /// Resolve a previously interned handle back to its string.
    pub fn resolve(&self, sym: Sym) -> &str {
        &self.strs[sym.index()]
    }

    /// Number of distinct strings interned so far.
    pub fn len(&self) -> usize {
        self.strs.len()
    }

    /// Whether nothing has been interned yet.
    pub fn is_empty(&self) -> bool {
        self.strs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_stable_and_deduplicated() {
        let mut i = StrInterner::new();
        let a = i.intern("foo");
        let b = i.intern("bar");
        let c = i.intern("foo");
        assert_eq!(a, c, "equal strings must intern equal");
        assert_ne!(a, b, "distinct strings must intern distinct");
        assert_eq!(i.resolve(a), "foo");
        assert_eq!(i.len(), 2);
    }
}
