//! The linker core used by the `lf-ld` driver. See ROADMAP Phase 8.
//!
//! Consumes relocatable objects and static archives, resolves symbols,
//! applies relocations, lays out sections, and writes an executable or shared
//! object.

/// Options controlling a link.
#[derive(Debug, Default)]
pub struct LinkOptions {
    /// Output path for the linked image.
    pub output: String,
    /// Input object and archive paths.
    pub inputs: Vec<String>,
}

/// Link the given inputs into an output image (placeholder).
pub fn link(_options: &LinkOptions) -> Result<(), String> {
    Err("linker not yet implemented — see ROADMAP Phase 8".to_owned())
}
