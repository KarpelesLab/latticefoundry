//! LEB128, the variable-length integer encoding of the WebAssembly binary
//! format (Core Specification §5.2.2 "Integers").
//!
//! An unsigned `uN` is written seven bits at a time, least significant group
//! first, with the high bit of every byte but the last set; a signed `sN` the
//! same way over its two's-complement value, stopping once the remaining value
//! is all sign bits and the last byte's bit 6 agrees with it. The spec allows
//! any encoding up to `ceil(N / 7)` bytes, so a value may also be written
//! **padded** to that maximum (continuation bits on filler groups). Relocatable
//! objects use the padded 5-byte form for every field a linker patches, so the
//! patched value never changes the code's size (tool-conventions `Linking.md`).

/// Append `v` as an unsigned LEB128 (`u32`/`u64`), in its shortest form.
pub fn write_u64(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Append `v` as an unsigned LEB128 in its shortest form.
pub fn write_u32(out: &mut Vec<u8>, v: u32) {
    write_u64(out, u64::from(v));
}

/// Append `v` as a signed LEB128 (`s32`/`s64`), in its shortest form.
pub fn write_i64(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7; // arithmetic: keeps the sign
        let done = (v == 0 && byte & 0x40 == 0) || (v == -1 && byte & 0x40 != 0);
        if done {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Append `v` as a signed LEB128 in its shortest form.
pub fn write_i32(out: &mut Vec<u8>, v: i32) {
    write_i64(out, i64::from(v));
}

/// Append `v` as an unsigned LEB128 padded to exactly 5 bytes (the maximum
/// for a `u32`): the form a relocatable object uses for patchable indices.
pub fn write_u32_padded(out: &mut Vec<u8>, v: u32) {
    let mut v = v;
    for i in 0..5 {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        out.push(if i < 4 { byte | 0x80 } else { byte });
    }
}

/// Append `v` as a signed LEB128 padded to exactly 5 bytes (the maximum for
/// an `s32`): the form of a patchable `i32.const` operand.
pub fn write_i32_padded(out: &mut Vec<u8>, v: i32) {
    let mut v = i64::from(v);
    for i in 0..5 {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        out.push(if i < 4 { byte | 0x80 } else { byte });
    }
}

/// Decode an unsigned LEB128 at `*at`, advancing it. `None` on truncation or
/// a value wider than 64 bits.
pub fn read_u64(bytes: &[u8], at: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes.get(*at)?;
        *at += 1;
        if shift >= 64 {
            return None;
        }
        result |= u64::from(byte & 0x7f) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Some(result);
        }
    }
}

/// Decode a signed LEB128 at `*at`, advancing it. `None` on truncation or a
/// value wider than 64 bits.
pub fn read_i64(bytes: &[u8], at: &mut usize) -> Option<i64> {
    let mut result = 0i64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes.get(*at)?;
        *at += 1;
        if shift >= 64 {
            return None;
        }
        result |= i64::from(byte & 0x7f) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < 64 && byte & 0x40 != 0 {
                result |= -1i64 << shift;
            }
            return Some(result);
        }
    }
}

/// Append a length-prefixed UTF-8 name (Core Specification §5.2.4).
pub fn write_name(out: &mut Vec<u8>, name: &str) {
    write_u32(out, name.len() as u32);
    out.extend_from_slice(name.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(v: u64) -> Vec<u8> {
        let mut o = Vec::new();
        write_u64(&mut o, v);
        o
    }

    fn s(v: i64) -> Vec<u8> {
        let mut o = Vec::new();
        write_i64(&mut o, v);
        o
    }

    #[test]
    fn unsigned_golden() {
        assert_eq!(u(0), [0x00]);
        assert_eq!(u(1), [0x01]);
        assert_eq!(u(127), [0x7f]);
        assert_eq!(u(128), [0x80, 0x01]);
        assert_eq!(u(624_485), [0xe5, 0x8e, 0x26]);
        assert_eq!(u(u64::from(u32::MAX)), [0xff, 0xff, 0xff, 0xff, 0x0f]);
        assert_eq!(u(u64::MAX).len(), 10);
    }

    #[test]
    fn signed_golden() {
        assert_eq!(s(0), [0x00]);
        assert_eq!(s(-1), [0x7f]);
        assert_eq!(s(63), [0x3f]);
        assert_eq!(s(64), [0xc0, 0x00]);
        assert_eq!(s(-64), [0x40]);
        assert_eq!(s(-65), [0xbf, 0x7f]);
        assert_eq!(s(-123_456), [0xc0, 0xbb, 0x78]);
        assert_eq!(s(i64::from(i32::MIN)), [0x80, 0x80, 0x80, 0x80, 0x78]);
        assert_eq!(s(i64::MIN).len(), 10);
    }

    #[test]
    fn padded_forms_are_five_bytes_and_decode() {
        for v in [0u32, 1, 127, 128, 0xdead_beef, u32::MAX] {
            let mut o = Vec::new();
            write_u32_padded(&mut o, v);
            assert_eq!(o.len(), 5);
            let mut at = 0;
            assert_eq!(read_u64(&o, &mut at), Some(u64::from(v)));
            assert_eq!(at, 5);
        }
        for v in [0i32, 1, -1, 63, 64, -64, -65, i32::MIN, i32::MAX] {
            let mut o = Vec::new();
            write_i32_padded(&mut o, v);
            assert_eq!(o.len(), 5);
            let mut at = 0;
            assert_eq!(read_i64(&o, &mut at), Some(i64::from(v)), "{v}");
        }
        let mut o = Vec::new();
        write_u32_padded(&mut o, 3);
        assert_eq!(o, [0x83, 0x80, 0x80, 0x80, 0x00]);
    }

    #[test]
    fn round_trips() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..2000 {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let v = x >> (x % 64);
            let enc = u(v);
            let mut at = 0;
            assert_eq!(read_u64(&enc, &mut at), Some(v));
            assert_eq!(at, enc.len());
            let sv = v as i64;
            let enc = s(sv);
            let mut at = 0;
            assert_eq!(read_i64(&enc, &mut at), Some(sv));
            assert_eq!(at, enc.len());
        }
        assert_eq!(read_u64(&[0x80], &mut 0), None);
    }
}
