//! Helpers shared by the integration tests that compare against the host gcc.

use std::path::Path;
use std::process::{Command, Stdio};

/// The `-std=` flag to give `gcc` for `std`. gcc before 14 knows C23 only by
/// its draft name, so `c23`/`gnu23` fall back to `c2x`/`gnu2x` when this gcc
/// rejects the final spelling (e.g. Ubuntu 24.04's gcc 13).
pub(crate) fn gcc_std_flag(gcc: &Path, std: &str) -> String {
    let flag = format!("-std={std}");
    let Some(draft) = std.strip_suffix("23").map(|p| format!("-std={p}2x")) else {
        return flag;
    };
    let accepted = Command::new(gcc)
        .args([flag.as_str(), "-E", "-x", "c", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if accepted { flag } else { draft }
}
