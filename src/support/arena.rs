//! A generic typed arena with `Copy` index handles.
//!
//! The IR and every later phase are built on id/arena substrate rather than
//! interior pointers (tenet T5): values live in a contiguous [`Vec`] and are
//! referred to by a small `Copy` [`Id`]. Because [`Id`] is parameterized by the
//! element type, an [`Id<A>`] and an [`Id<B>`] are *distinct, non-interchangeable*
//! types — the compiler rejects indexing one arena with another arena's handle.
//!
//! This id/arena model is what makes the core content-addressable and
//! parallel-/incremental-ready: handles are plain integers with no lifetime or
//! pointer identity, so they serialize losslessly and can be shared freely.

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// A cheap, copyable handle to a value of type `T` stored in an [`Arena<T>`].
///
/// The type parameter is a compile-time tag only (it occupies no space): it
/// makes handles into different arenas incompatible, so a `BlockId` can never be
/// used to index a function arena by mistake. Construct handles through
/// [`Arena::push`] (or [`Id::from_index`]) and recover the backing index with
/// [`Id::index`].
pub struct Id<T> {
    raw: u32,
    // `fn() -> T` keeps `Id<T>: Send + Sync + Copy` regardless of `T` and makes
    // the tag purely phantom (no ownership or drop obligation for `T`).
    _marker: PhantomData<fn() -> T>,
}

impl<T> Id<T> {
    /// Build a handle from a dense index.
    ///
    /// This is mainly for deserialization; within a program handles come from
    /// [`Arena::push`]. The index must fit in a `u32`.
    #[inline]
    pub fn from_index(index: usize) -> Self {
        debug_assert!(index <= u32::MAX as usize, "arena index overflows u32");
        Self {
            raw: index as u32,
            _marker: PhantomData,
        }
    }

    /// The dense index backing this handle.
    #[inline]
    pub fn index(self) -> usize {
        self.raw as usize
    }

    /// The raw `u32` backing this handle.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.raw
    }
}

// Manual trait impls: deriving would place an unwanted `T: Trait` bound, but a
// handle's identity is just its `u32`, independent of `T`.
impl<T> Clone for Id<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> PartialEq for Id<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.raw.cmp(&other.raw)
    }
}

impl<T> std::hash::Hash for Id<T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<T> std::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Id({})", self.raw)
    }
}

/// A growable, contiguous store of `T` values addressed by [`Id<T>`].
///
/// Pushing a value yields a stable handle that stays valid for the lifetime of
/// the arena; existing handles never move or dangle. This is the fundamental
/// container the IR container hierarchy (module → function → block →
/// instruction) is built from.
#[derive(Debug, Clone)]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        // Not derived: `#[derive(Default)]` would require `T: Default`, but an
        // empty arena needs no such bound.
        Self { items: Vec::new() }
    }
}

impl<T> Arena<T> {
    /// Create an empty arena.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty arena with room for `capacity` elements.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    /// Append `value`, returning its freshly allocated handle.
    #[inline]
    pub fn push(&mut self, value: T) -> Id<T> {
        let id = Id::from_index(self.items.len());
        self.items.push(value);
        id
    }

    /// Borrow the value for `id`, or `None` if the handle is out of range.
    #[inline]
    pub fn get(&self, id: Id<T>) -> Option<&T> {
        self.items.get(id.index())
    }

    /// Mutably borrow the value for `id`, or `None` if out of range.
    #[inline]
    pub fn get_mut(&mut self, id: Id<T>) -> Option<&mut T> {
        self.items.get_mut(id.index())
    }

    /// Number of values in the arena.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the arena holds no values.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Iterate over the stored values.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// Mutably iterate over the stored values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.items.iter_mut()
    }

    /// Iterate over `(handle, value)` pairs in insertion order.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (Id<T>, &T)> {
        self.items
            .iter()
            .enumerate()
            .map(|(i, v)| (Id::from_index(i), v))
    }

    /// Iterate over all handles in the arena, in insertion order.
    pub fn ids(&self) -> impl Iterator<Item = Id<T>> {
        (0..self.items.len()).map(Id::from_index)
    }
}

impl<T> Index<Id<T>> for Arena<T> {
    type Output = T;

    #[inline]
    fn index(&self, id: Id<T>) -> &T {
        &self.items[id.index()]
    }
}

impl<T> IndexMut<Id<T>> for Arena<T> {
    #[inline]
    fn index_mut(&mut self, id: Id<T>) -> &mut T {
        &mut self.items[id.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Distinct element types produce distinct, non-interchangeable id types.
    #[derive(Debug, PartialEq)]
    struct Foo(u32);
    #[derive(Debug, PartialEq)]
    struct Bar(&'static str);

    #[test]
    fn push_returns_sequential_handles() {
        let mut a: Arena<Foo> = Arena::new();
        let i0 = a.push(Foo(10));
        let i1 = a.push(Foo(20));
        let i2 = a.push(Foo(30));

        assert_eq!(i0.index(), 0);
        assert_eq!(i1.index(), 1);
        assert_eq!(i2.index(), 2);
        assert_ne!(i0, i1);
        assert_eq!(a.len(), 3);
        assert!(!a.is_empty());
    }

    #[test]
    fn indexing_and_get() {
        let mut a: Arena<Foo> = Arena::new();
        let id = a.push(Foo(42));
        assert_eq!(a[id], Foo(42));
        assert_eq!(a.get(id), Some(&Foo(42)));

        a[id].0 = 7;
        assert_eq!(a[id], Foo(7));
        a.get_mut(id).unwrap().0 = 99;
        assert_eq!(a[id].0, 99);

        let bogus = Id::<Foo>::from_index(100);
        assert_eq!(a.get(bogus), None);
    }

    #[test]
    fn iteration() {
        let mut a: Arena<Foo> = Arena::new();
        let ids: Vec<_> = (0..3).map(|n| a.push(Foo(n))).collect();

        let vals: Vec<u32> = a.iter().map(|f| f.0).collect();
        assert_eq!(vals, vec![0, 1, 2]);

        let pairs: Vec<(usize, u32)> =
            a.iter_enumerated().map(|(id, f)| (id.index(), f.0)).collect();
        assert_eq!(pairs, vec![(0, 0), (1, 1), (2, 2)]);

        let collected_ids: Vec<_> = a.ids().collect();
        assert_eq!(collected_ids, ids);

        for f in a.iter_mut() {
            f.0 += 10;
        }
        assert_eq!(a.iter().map(|f| f.0).collect::<Vec<_>>(), vec![10, 11, 12]);
    }

    #[test]
    fn handles_are_ordered_and_hashable() {
        use std::collections::BTreeSet;
        let mut a: Arena<Bar> = Arena::new();
        let x = a.push(Bar("x"));
        let y = a.push(Bar("y"));
        assert!(x < y);

        let mut set = BTreeSet::new();
        set.insert(y);
        set.insert(x);
        assert_eq!(set.into_iter().collect::<Vec<_>>(), vec![x, y]);
    }

    #[test]
    fn id_round_trips_through_index() {
        let id = Id::<Foo>::from_index(1234);
        assert_eq!(id.index(), 1234);
        assert_eq!(id.as_u32(), 1234);
    }
}
