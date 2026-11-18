//! Glyphs from egui's own icon fonts.
//!
//! The `monochrome_emoji_fonts` feature on `egui` bundles NotoEmoji, so these
//! render in the UI colour without shipping an icon font of our own. Every
//! glyph here comes from NotoEmoji, whose uniform scale keeps the buttons one
//! weight; the private-use code points that only `emoji-icon-font` covers
//! (such as `🗁`, `🗐` or `🗙`) stay out, because they show up at a different
//! size than their neighbours.

pub const OPEN: &str = "📂";
pub const SCREENSHOT: &str = "📷";
pub const AUTO_DETECT: &str = "🔍";
pub const DRAW_GRID: &str = "✏";
pub const UNDO: &str = "↩";
pub const REDO: &str = "↪";
pub const LEARN: &str = "📖";
pub const COPY: &str = "📋";
pub const SAVE: &str = "💾";
pub const OK: &str = "✔";
pub const FAIL: &str = "✖";
pub const WARNING: &str = "⚠";

/// `icon` in front of a label, with the space the two need between them.
pub fn label(icon: &str, text: impl std::fmt::Display) -> String {
    format!("{icon} {text}")
}
