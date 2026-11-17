//! Glyphs from egui's own icon fonts.
//!
//! The `monochrome_emoji_fonts` feature on `egui` bundles NotoEmoji and
//! emoji-icon-font, so these render in the UI colour without shipping an icon
//! font of our own. Only code points covered by those two fonts belong here.

pub const OPEN: &str = "🗁";
pub const SCREENSHOT: &str = "📷";
pub const AUTO_DETECT: &str = "🔍";
pub const DRAW_GRID: &str = "⛶";
pub const UNDO: &str = "↺";
pub const REDO: &str = "↻";
pub const GRID: &str = "📐";
pub const CONFIDENCE: &str = "🎯";
pub const PAINT: &str = "🎨";
pub const VALIDATION: &str = "☑";
pub const LEARN: &str = "📖";
pub const EXPORT: &str = "📤";
pub const COPY: &str = "🗐";
pub const SAVE: &str = "💾";
pub const OK: &str = "✔";
pub const FAIL: &str = "🗙";
pub const WARNING: &str = "⚠";

/// `icon` in front of a label, with the space the two need between them.
pub fn label(icon: &str, text: impl std::fmt::Display) -> String {
    format!("{icon} {text}")
}
