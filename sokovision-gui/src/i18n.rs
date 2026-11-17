//! Interface language selection.
//!
//! The texts themselves live in `locales/*.yml`; [`tr!`] looks a key up in the
//! language the system is set to.

use sokovision_core::Cell;

/// Locale code for the language the system is set to.
///
/// Everything that is not Chinese reads English: those are the only two
/// translations, and `i18n!` falls back to English for unknown locales.
fn system_locale() -> &'static str {
    match sys_locale::get_locale().as_deref() {
        Some(locale) if locale.starts_with("zh") => "zh-CN",
        _ => "en",
    }
}

/// Makes every following [`tr!`] lookup use the system language.
pub fn apply_system_language() {
    rust_i18n::set_locale(system_locale());
}

/// Message in the current interface language.
macro_rules! tr {
    ($key:literal $(, $name:ident = $value:expr)* $(,)?) => {
        ::rust_i18n::t!($key $(, $name = $value)*).into_owned()
    };
}
pub(crate) use tr;

/// Short name of a cell state, used by the palette, the inspector and status
/// messages.
pub fn cell_label(cell: Cell) -> String {
    match cell {
        Cell::Empty => tr!("cell_empty"),
        Cell::Wall => tr!("cell_wall"),
        Cell::Goal => tr!("cell_goal"),
        Cell::Box => tr!("cell_box"),
        Cell::BoxOnGoal => tr!("cell_box_on_goal"),
        Cell::Player => tr!("cell_player"),
        Cell::PlayerOnGoal => tr!("cell_player_on_goal"),
        Cell::Unknown => tr!("cell_unknown"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_locales_ship_the_same_key() {
        let zh = rust_i18n::t!("open_image", locale = "zh-CN");
        let en = rust_i18n::t!("open_image", locale = "en");

        assert_eq!(zh.as_ref(), "打开图片");
        assert_eq!(en.as_ref(), "Open Image");
    }

    #[test]
    fn every_cell_state_has_a_name() {
        let cells = [
            Cell::Empty,
            Cell::Wall,
            Cell::Goal,
            Cell::Box,
            Cell::BoxOnGoal,
            Cell::Player,
            Cell::PlayerOnGoal,
            Cell::Unknown,
        ];
        for cell in cells {
            assert!(!cell_label(cell).is_empty());
        }
    }

    #[test]
    fn locales_define_matching_keys_without_garbled_brackets() {
        let keys = |source: &str| -> Vec<String> {
            source
                .lines()
                .filter(|line| !line.trim().is_empty() && !line.starts_with(' '))
                .filter_map(|line| line.split_once(':').map(|(key, _)| key.trim().to_owned()))
                .collect()
        };
        let en = include_str!("../locales/en.yml");
        let zh = include_str!("../locales/zh-CN.yml");

        assert_eq!(keys(en), keys(zh));
        assert!(!zh.contains("d ("), "stray spacing typo in zh-CN.yml");
        assert!(
            !zh.contains('（') && !zh.contains('）'),
            "zh-CN.yml uses ASCII parentheses consistently"
        );
    }
}
