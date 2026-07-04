//! Deterministic hashing for compiler-internal maps and sets.
//!
//! The standard library's [`HashMap`] seeds its hasher randomly per process, so
//! its iteration order varies from run to run. That is a fine defence against
//! HashDoS for untrusted input, but it is poison for a compiler that wants
//! **reproducible** and eventually **content-addressed** output (tenet T5):
//! anything that iterates a map — printing IR, hashing a module, ordering work —
//! would become nondeterministic.
//!
//! This module provides a fast, safe, dependency-free deterministic hasher in
//! the FxHash family (a rotate / xor / multiply word mixer) plus the
//! [`DetHashMap`] and [`DetHashSet`] aliases that use it. Because the seed is a
//! fixed constant, two runs over the same insertions iterate in the same order.
//!
//! Use these for **compiler-internal** collections where determinism matters,
//! *not* for maps keyed by untrusted external input (they offer no HashDoS
//! resistance).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

// A large odd constant with well-distributed bits; the multiplier in the
// classic Fibonacci-style multiplicative mixer.
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
const ROTATE: u32 = 5;

/// A fast, deterministic [`Hasher`] using a rotate / xor / multiply word mixer.
///
/// Each machine word of input is folded in as
/// `hash = (hash.rotate_left(5) ^ word).wrapping_mul(SEED)`. It is *not*
/// cryptographic and offers no HashDoS resistance; its job is speed and a
/// fixed, reproducible seed.
#[derive(Debug, Clone, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add_word(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[..8]);
            self.add_word(u64::from_le_bytes(buf));
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[..4]);
            self.add_word(u64::from(u32::from_le_bytes(buf)));
            bytes = &bytes[4..];
        }
        if bytes.len() >= 2 {
            let mut buf = [0u8; 2];
            buf.copy_from_slice(&bytes[..2]);
            self.add_word(u64::from(u16::from_le_bytes(buf)));
            bytes = &bytes[2..];
        }
        if let Some(&b) = bytes.first() {
            self.add_word(u64::from(b));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add_word(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add_word(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add_word(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add_word(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add_word(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// A [`BuildHasher`] that yields a fresh, fixed-seed [`FxHasher`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher::default()
    }
}

/// A [`HashMap`] with deterministic iteration order (fixed-seed [`FxHasher`]).
///
/// Construct with [`Default::default`] or [`HashMap::with_hasher`]. See the
/// module docs for when to prefer this over `std`'s randomized `HashMap`.
pub type DetHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

/// A [`HashSet`] with deterministic iteration order (fixed-seed [`FxHasher`]).
pub type DetHashSet<T> = HashSet<T, FxBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        FxBuildHasher.hash_one(value)
    }

    #[test]
    fn hashing_is_reproducible() {
        // Same input, independent hasher instances -> identical output.
        assert_eq!(hash_of(&"latticefoundry"), hash_of(&"latticefoundry"));
        assert_eq!(hash_of(&12345u64), hash_of(&12345u64));
        assert_ne!(hash_of(&1u64), hash_of(&2u64));
    }

    #[test]
    fn writer_paths_agree_across_lengths() {
        // Exercise the 8/4/2/1 byte tail paths without panicking.
        for len in 0..20usize {
            let bytes: Vec<u8> = (0..len as u8).collect();
            let mut a = FxHasher::default();
            let mut b = FxHasher::default();
            a.write(&bytes);
            b.write(&bytes);
            assert_eq!(a.finish(), b.finish());
        }
    }

    #[test]
    fn map_iteration_order_is_deterministic() {
        let build = |offset: u32| {
            let mut m: DetHashMap<u32, u32> = DetHashMap::default();
            for k in 0..64u32 {
                m.insert(k, k + offset);
            }
            m.into_iter().collect::<Vec<_>>()
        };
        // Two independently built maps with the same keys iterate identically.
        let a = build(0);
        let b: Vec<(u32, u32)> = {
            let mut m: DetHashMap<u32, u32> = DetHashMap::default();
            for k in 0..64u32 {
                m.insert(k, k);
            }
            m.into_iter().collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn map_basic_ops() {
        let mut m: DetHashMap<&str, i32> = DetHashMap::default();
        m.insert("a", 1);
        m.insert("b", 2);
        assert_eq!(m.get("a"), Some(&1));
        assert_eq!(m.get("b"), Some(&2));
        assert_eq!(m.get("c"), None);
        assert!(m.contains_key("a"));
        assert_eq!(m.len(), 2);
        *m.get_mut("a").unwrap() += 10;
        assert_eq!(m["a"], 11);
    }

    #[test]
    fn set_basic_ops_and_determinism() {
        let mut s: DetHashSet<u32> = DetHashSet::default();
        for k in [5u32, 3, 9, 1, 7] {
            s.insert(k);
        }
        assert!(s.contains(&9));
        assert!(!s.contains(&4));
        assert_eq!(s.len(), 5);
        assert!(!s.insert(9)); // already present

        let mut s2: DetHashSet<u32> = DetHashSet::default();
        for k in [5u32, 3, 9, 1, 7] {
            s2.insert(k);
        }
        assert_eq!(
            s.iter().collect::<Vec<_>>(),
            s2.iter().collect::<Vec<_>>()
        );
    }
}
