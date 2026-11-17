//! Desktop front end: open, paste or capture a screenshot, then fix the
//! recognized board by hand.

// Release builds are a plain GUI app: no console window next to the window.
// Debug builds keep it, so panics and `eprintln!` still land somewhere.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod canvas;
mod grid;
mod i18n;
mod icons;
mod panel;
mod skins;

use std::fs;
use std::sync::Arc;

use eframe::egui;
use egui::{Color32, FontData, FontDefinitions, FontFamily, Key, Modifiers};

use crate::app::{App, take_paste_events};
use crate::i18n::tr;

rust_i18n::i18n!("locales", fallback = "en");

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1040.0, 700.0])
            .with_min_inner_size([860.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Sokovision",
        options,
        Box::new(|creation_context| Ok(Box::new(App::new(creation_context)))),
    )
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.tick(ctx);
    }

    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root_ui.ctx().clone();

        // Releasing the pointer ends a stepper drag, so the next change records
        // its own undo step again.
        if !ctx.input(|input| input.pointer.any_down()) {
            self.release_grid_history();
        }
        self.handle_shortcuts(&ctx);
        self.handle_paste(&ctx);
        if let Some(path) = dropped_path(&ctx) {
            self.open_path(&ctx, path);
        }

        self.draw_toolbar(root_ui, &ctx);
        self.draw_status_bar(root_ui);
        self.draw_side_panel(&ctx, root_ui);
        egui::CentralPanel::no_frame().show(root_ui, |ui| self.draw_canvas(ui, &ctx));
    }
}

impl App {
    /// egui-winit turns the platform clipboard shortcut into paste events
    /// instead of a plain Ctrl+V key event, so both have to be handled here.
    fn handle_paste(&mut self, ctx: &egui::Context) {
        let (pasted_image, paste_requested) = take_paste_events(ctx);
        let paste_shortcut = ctx.input_mut(|input| input.consume_key(Modifiers::CTRL, Key::V));
        if let Some(image) = pasted_image {
            self.set_color_image(ctx, &image);
        } else if paste_requested || paste_shortcut {
            self.paste_image(ctx);
        }
    }

    fn draw_toolbar(&mut self, root_ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::Panel::top("toolbar").show(root_ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button(icons::label(icons::OPEN, tr!("open_image")))
                    .clicked()
                {
                    self.open_dialog(ctx);
                }
                if ui
                    .button(icons::label(icons::SCREENSHOT, tr!("screenshot")))
                    .on_hover_text(tr!("screenshot_tip"))
                    .clicked()
                {
                    self.request_screenshot(ctx);
                }
                if ui
                    .add_enabled(
                        self.can_undo(),
                        egui::Button::new(icons::label(icons::UNDO, tr!("undo"))),
                    )
                    .on_hover_text(tr!("undo_tip"))
                    .clicked()
                {
                    self.undo(ctx);
                }
                if ui
                    .add_enabled(
                        self.can_redo(),
                        egui::Button::new(icons::label(icons::REDO, tr!("redo"))),
                    )
                    .on_hover_text(tr!("redo_tip"))
                    .clicked()
                {
                    self.redo(ctx);
                }
            });
        });
    }

    fn draw_status_bar(&mut self, root_ui: &mut egui::Ui) {
        egui::Panel::bottom("status").show(root_ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let zoom = self.active_zoom();
                    ui.label(tr!("zoom", percent = (zoom * 100.0).round() as i32));
                });
            });
        });
    }
}

/// The look of the whole window in one place: rounded widgets, a bluer accent
/// and a bit more air between controls. Everything else stays default egui, so
/// there is no bespoke widget code to keep in sync as the app grows.
pub(crate) fn install_theme(ctx: &egui::Context) {
    const ACCENT: Color32 = Color32::from_rgb(72, 132, 235);
    const EDGE: Color32 = Color32::from_rgb(54, 57, 65);

    // The app reads as one dark surface whatever the desktop does; without this
    // a light system theme would leave the canvas grey and the chrome white.
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(9.0, 4.0);
        style.spacing.interact_size.y = 22.0;
        style.spacing.scroll = egui::style::ScrollStyle::solid();
        style.text_styles = [
            (egui::TextStyle::Heading, egui::FontId::proportional(18.0)),
            (egui::TextStyle::Body, egui::FontId::proportional(15.0)),
            (egui::TextStyle::Button, egui::FontId::proportional(15.0)),
            (egui::TextStyle::Small, egui::FontId::proportional(13.0)),
            (egui::TextStyle::Monospace, egui::FontId::monospace(13.0)),
        ]
        .into();

        let visuals = &mut style.visuals;
        let radius = egui::CornerRadius::same(6);
        for widget in [
            &mut visuals.widgets.noninteractive,
            &mut visuals.widgets.inactive,
            &mut visuals.widgets.hovered,
            &mut visuals.widgets.active,
            &mut visuals.widgets.open,
        ] {
            widget.corner_radius = radius;
        }
        visuals.panel_fill = Color32::from_rgb(30, 31, 35);
        visuals.window_fill = Color32::from_rgb(30, 31, 35);
        visuals.extreme_bg_color = Color32::from_rgb(20, 21, 24);
        visuals.faint_bg_color = Color32::from_rgb(38, 39, 44);
        visuals.window_stroke = egui::Stroke::new(1.0, EDGE);
        visuals.selection.bg_fill = ACCENT;
        visuals.selection.stroke = egui::Stroke::new(1.0, Color32::WHITE);
        visuals.hyperlink_color = ACCENT;
        visuals.window_corner_radius = egui::CornerRadius::same(8);

        // Chrome is quiet, controls lift off it, the pressed state carries the
        // accent: enough contrast to find things without a rainbow.
        visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, EDGE);
        visuals.widgets.noninteractive.fg_stroke.color = Color32::from_rgb(198, 202, 210);
        visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(45, 47, 53);
        visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, Color32::TRANSPARENT);
        visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(57, 60, 68);
        visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, Color32::from_rgb(78, 82, 92));
        visuals.widgets.active.weak_bg_fill = Color32::from_rgb(66, 70, 80);
        visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    });
}

/// Two exclusive choices drawn as one control: no gap, only the outer corners
/// rounded, equal halves, the active one carrying the accent. Returns whichever
/// half was clicked this frame — including the active one, so callers can re-run
/// it — plus the group response for a tooltip. Shared by the toolbar and the
/// side panel so both read the same.
pub(crate) fn segmented<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    current: T,
    choices: [(T, &str); 2],
    enabled: bool,
) -> (Option<T>, egui::Response) {
    let font = egui::TextStyle::Button.resolve(ui.style());
    let padding = ui.spacing().button_padding.x;
    let widest = choices
        .iter()
        .map(|(_, label)| {
            ui.painter()
                .layout_no_wrap((*label).to_owned(), font.clone(), Color32::WHITE)
                .size()
                .x
        })
        .fold(0.0_f32, f32::max)
        + 2.0 * padding;

    let mut picked = None;
    let response = ui
        .scope(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            let round = ui.visuals().widgets.inactive.corner_radius.nw;
            let corners = [
                egui::CornerRadius {
                    nw: round,
                    sw: round,
                    ne: 0,
                    se: 0,
                },
                egui::CornerRadius {
                    nw: 0,
                    sw: 0,
                    ne: round,
                    se: round,
                },
            ];
            for (index, (value, label)) in choices.into_iter().enumerate() {
                // A plain button for the inactive half: `selectable` drops the
                // frame when unselected, which would leave one bare label next
                // to one filled chip instead of two halves of one control.
                let button = if value == current {
                    egui::Button::selectable(true, label)
                } else {
                    egui::Button::new(label)
                };
                let button = button
                    .corner_radius(corners[index])
                    .min_size(egui::vec2(widest, 22.0));
                if ui.add_enabled(enabled, button).clicked() {
                    picked = Some(value);
                }
            }
        })
        .response;
    (picked, response)
}

fn dropped_path(ctx: &egui::Context) -> Option<std::path::PathBuf> {
    ctx.input(|input| {
        input
            .raw
            .dropped_files
            .first()
            .map(|file| file.path().to_path_buf())
    })
}

/// Picks the first CJK font that exists, so the Chinese UI text renders.
pub(crate) fn install_cjk_font(ctx: &egui::Context) {
    const CANDIDATES: [&str; 7] = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ];
    for path in CANDIDATES {
        let Ok(bytes) = fs::read(path) else {
            continue;
        };
        let mut fonts = FontDefinitions::default();
        fonts.font_data.insert(
            "sokovision-cjk".to_owned(),
            Arc::new(FontData::from_owned(bytes)),
        );
        // Appended, not prepended: Latin text must keep egui's own font, or a
        // CJK font's wide Latin face takes over.
        fonts
            .families
            .entry(FontFamily::Proportional)
            .or_default()
            .push("sokovision-cjk".to_owned());
        fonts
            .families
            .entry(FontFamily::Monospace)
            .or_default()
            .push("sokovision-cjk".to_owned());
        ctx.set_fonts(fonts);
        break;
    }
}
