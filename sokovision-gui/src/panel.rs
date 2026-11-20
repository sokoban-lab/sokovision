//! The control panel on the right.

use eframe::egui;
use egui::{AtomExt as _, Color32, RichText, vec2};
use sokovision_core::{Cell, MAX_BOARD_AXIS};

use crate::app::App;
use crate::i18n::{cell_label, tr};
use crate::icons;

/// Symbols offered in the paint palette, in XSB order.
pub(crate) const CELL_CHOICES: [(Cell, char); 7] = [
    (Cell::Empty, '-'),
    (Cell::Wall, '#'),
    (Cell::Goal, '.'),
    (Cell::Box, '$'),
    (Cell::BoxOnGoal, '*'),
    (Cell::Player, '@'),
    (Cell::PlayerOnGoal, '+'),
];

/// How wide the panel is when it first appears.
const PANEL_WIDTH: f32 = 300.0;
/// Height of the XSB preview box.
const PREVIEW_HEIGHT: f32 = 170.0;

impl App {
    pub fn draw_side_panel(&mut self, ctx: &egui::Context, root_ui: &mut egui::Ui) {
        let image_size = self
            .image
            .as_ref()
            .map(|image| (image.rgba.width(), image.rgba.height()));
        egui::Panel::right("controls")
            .resizable(true)
            .default_size(PANEL_WIDTH)
            // A little more air than the default side-panel frame: controls
            // otherwise sit right against the window edge and the scrollbar.
            .frame(
                egui::Frame::side_top_panel(root_ui.style())
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(root_ui, |ui| {
                // The warnings and the XSB preview together are taller than the
                // panel on a short window, so the whole column scrolls instead
                // of clipping whatever happens to land last.
                egui::ScrollArea::vertical()
                    .id_salt("panel_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.panel_body(ctx, ui, image_size));
            });
    }

    fn panel_body(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        image_size: Option<(u32, u32)>,
    ) {
        let mut sections = SectionState::default();
        if self.is_cropping_capture() {
            sections.heading(ui, &tr!("screenshot"));
            ui.small(tr!("capture_crop_hint"));
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button(tr!("capture_use_full")).clicked() {
                    self.import_full_capture(ctx);
                }
                if ui.button(tr!("capture_cancel")).clicked() {
                    self.cancel_capture_crop(ctx);
                }
            });
            return;
        }

        // The mean of the per-cell scores: one glance tells you whether the run
        // is worth checking cell by cell.
        if let Some(confidence) = self.overall_confidence() {
            sections.heading(ui, &tr!("overall_confidence"));
            let text = format!("{:.1}%", confidence * 100.0);
            let color = confidence_color(confidence);
            ui.horizontal(|ui| {
                let font = egui::TextStyle::Small.resolve(ui.style());
                let text_width = ui
                    .painter()
                    .layout_no_wrap(text.clone(), font, color)
                    .size()
                    .x;
                let spacing = ui.spacing().item_spacing.x;
                let bar_width = (ui.available_width() - text_width - spacing).max(24.0);
                ui.add(
                    egui::ProgressBar::new(confidence)
                        .fill(color)
                        .desired_width(bar_width)
                        .desired_height(10.0)
                        .corner_radius(3.0),
                );
                ui.small(RichText::new(text).color(color));
            });
            if self.is_searching() {
                ui.add_space(2.0);
                ui.small(tr!("searching_grids"));
            }
            ui.add_space(4.0);
        }

        self.grid_section(ctx, ui, image_size, &mut sections);
        self.paint_section(ui, &mut sections);
        self.validation_section(ui, &mut sections);
        self.warnings_section(ui);
        self.export_section(ui, &mut sections);
        self.learning_section(ui, &mut sections);
        ui.add_space(2.0);
    }

    fn grid_section(
        &mut self,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        image_size: Option<(u32, u32)>,
        sections: &mut SectionState,
    ) {
        sections.heading(ui, &tr!("section_grid"));
        // Detection and hand drawing alternate freely, so they stay two
        // actions instead of one control with a persistent state.
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    image_size.is_some(),
                    egui::Button::new(icons::label(icons::DRAW_GRID, tr!("manual_grid"))),
                )
                .on_hover_text(tr!("manual_grid_tip"))
                .clicked()
            {
                self.start_hand_drawn_grid();
            }
            if ui
                .add_enabled(
                    image_size.is_some(),
                    egui::Button::new(icons::label(icons::AUTO_DETECT, tr!("auto_detect"))),
                )
                .on_hover_text(tr!("auto_detect_tip"))
                .clicked()
            {
                self.run_recognition(ctx, None);
            }
        });
        ui.add_space(4.0);
        let Some(grid) = self.grid_edit else {
            ui.small(tr!("no_grid_hint"));
            return;
        };

        let (mut rows, mut columns) = (grid.rows, grid.columns);
        let mut change = StepperChange::default();
        egui::Grid::new("grid_editor")
            .num_columns(2)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                ui.label(tr!("rows"));
                change = change | count_stepper(ui, &mut rows, "grid_rows");
                ui.end_row();
                ui.label(tr!("columns"));
                change = change | count_stepper(ui, &mut columns, "grid_columns");
                ui.end_row();
            });
        if change.changed {
            let resized = grid.with_counts(rows, columns);
            match image_size.and_then(|(width, height)| resized.clamp_to_image(width, height)) {
                Some(grid) => {
                    self.grid_edit = Some(grid);
                    self.apply_grid_live(ctx, change.dragging);
                }
                None => self.status = tr!("grid_out_of_image"),
            }
        }
        ui.small(tr!("handles_hint"));
    }

    fn paint_section(&mut self, ui: &mut egui::Ui, sections: &mut SectionState) {
        sections.heading(ui, &tr!("section_paint"));
        // A button centres its text, so labels of different lengths start at
        // different x. Split into a fixed symbol slot and a growing label so
        // both columns line up: the slot is as wide as the widest symbol.
        let font = egui::TextStyle::Button.resolve(ui.style());
        let symbol_slot = CELL_CHOICES
            .iter()
            .map(|(_, symbol)| {
                ui.painter()
                    .layout_no_wrap(symbol.to_string(), font.clone(), Color32::WHITE)
                    .size()
                    .x
            })
            .fold(0.0_f32, f32::max);
        let row_height = ui.fonts_mut(|fonts| fonts.row_height(&font));
        let column_width = ((ui.available_width() - 6.0) * 0.5).max(80.0);
        egui::Grid::new("paint_palette")
            .num_columns(2)
            .spacing([6.0, 4.0])
            .show(ui, |ui| {
                for (index, (cell, symbol)) in CELL_CHOICES.into_iter().enumerate() {
                    let selected = self.paint_cell == cell;
                    let symbol = egui::Atom::from(symbol.to_string())
                        .atom_size(vec2(symbol_slot, row_height))
                        .atom_align(egui::Align2::LEFT_CENTER);
                    let label = egui::Atom::from(cell_label(cell))
                        .atom_grow(true)
                        .atom_align(egui::Align2::LEFT_CENTER);
                    // `selectable` drops the frame when unselected, leaving bare
                    // labels; an ordinary button keeps the palette a tidy block
                    // of cards, with the chosen cell carrying the accent.
                    let button = if selected {
                        egui::Button::selectable(true, (symbol, label))
                    } else {
                        egui::Button::new((symbol, label))
                    };
                    let clicked = ui.add(button.min_size(vec2(column_width, 20.0))).clicked();
                    if clicked {
                        self.paint_cell = cell;
                    }
                    if index % 2 == 1 {
                        ui.end_row();
                    }
                }
                ui.end_row();
            });

        ui.horizontal(|ui| {
            ui.label(tr!("paint_scope"));
            let (picked, response) = crate::segmented(
                ui,
                self.batch_by_appearance,
                [
                    (false, tr!("scope_single").as_str()),
                    (true, tr!("scope_group").as_str()),
                ],
                true,
            );
            response.on_hover_text(tr!("scope_tip"));
            if let Some(batch) = picked {
                self.batch_by_appearance = batch;
            }
        });

        let Some((row, column)) = self.selected else {
            return;
        };
        let (current_cell, confidence, group_info, alternatives) = {
            let Some(result) = self.result.as_ref() else {
                return;
            };
            let Some(cell) = result.cells.get(row).and_then(|cells| cells.get(column)) else {
                return;
            };
            let group_info = cell
                .appearance
                .map(|appearance| (appearance, result.appearance_cells(appearance).len()));
            (
                cell.cell,
                cell.confidence,
                group_info,
                cell.alternatives.clone(),
            )
        };
        ui.add_space(4.0);
        ui.small(tr!(
            "cell_details",
            row = row,
            column = column,
            label = cell_label(current_cell),
            symbol = current_cell,
            confidence = format!("{:.3}", confidence)
        ));
        if let Some((appearance, count)) = group_info {
            ui.small(tr!(
                "appearance_group",
                appearance = appearance,
                count = count
            ));
        }
        for candidate in alternatives {
            let label = tr!(
                "alternative",
                label = cell_label(candidate.cell),
                score = format!("{:.3}", candidate.score)
            );
            if ui
                .small_button(label)
                .on_hover_text(tr!("alternative_tip"))
                .clicked()
            {
                self.apply_cell_choice(row, column, candidate.cell);
            }
        }
    }

    fn validation_section(&mut self, ui: &mut egui::Ui, sections: &mut SectionState) {
        if self.result.is_none() {
            return;
        }
        sections.heading(ui, &tr!("section_validation"));
        match self.derived().problems.as_ref() {
            Ok(problems) if problems.is_empty() => {
                ui.colored_label(
                    Color32::from_rgb(120, 220, 140),
                    icons::label(icons::OK, tr!("validation_ok")),
                );
            }
            Ok(problems) => {
                for problem in problems {
                    ui.colored_label(
                        Color32::from_rgb(240, 120, 110),
                        icons::label(icons::FAIL, problem),
                    );
                }
            }
            Err(error) => {
                ui.colored_label(
                    Color32::from_rgb(240, 120, 110),
                    icons::label(icons::FAIL, error),
                );
            }
        }
    }

    /// Recognition diagnostics are long and only interesting while something
    /// looks off, so they fold away instead of pushing the export off-screen.
    fn warnings_section(&mut self, ui: &mut egui::Ui) {
        let Some(result) = self.result.as_ref() else {
            return;
        };
        if result.warnings.is_empty() {
            return;
        }
        let count = result.warnings.len();
        egui::CollapsingHeader::new(icons::label(
            icons::WARNING,
            tr!("warnings_header", count = count),
        ))
        .id_salt("warnings")
        .show(ui, |ui| {
            for warning in &result.warnings {
                ui.colored_label(
                    Color32::from_rgb(225, 180, 80),
                    icons::label(icons::WARNING, warning),
                );
            }
        });
    }

    fn export_section(&mut self, ui: &mut egui::Ui, sections: &mut SectionState) {
        sections.heading(ui, &tr!("section_export"));
        if ui
            .checkbox(&mut self.trim_export, tr!("trim_export"))
            .changed()
        {
            self.invalidate_derived();
        }
        let valid = self.derived().xsb.is_ok();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    valid,
                    egui::Button::new(icons::label(icons::COPY, tr!("copy"))),
                )
                .clicked()
            {
                self.copy_xsb();
            }
            if ui
                .add_enabled(
                    valid,
                    egui::Button::new(icons::label(icons::SAVE, tr!("save"))),
                )
                .clicked()
            {
                self.save_xsb();
            }
        });

        // Borrow the cached preview instead of cloning it every frame; a label
        // on a canvas also reads with more contrast than a disabled text field.
        let preview = self.derived().preview.as_str();
        egui::ScrollArea::vertical()
            .id_salt("xsb_preview")
            .max_height(PREVIEW_HEIGHT)
            .show(ui, |ui| {
                egui::Frame::canvas(ui.style()).show(ui, |ui| {
                    ui.set_min_width(ui.available_width());
                    ui.monospace(preview);
                });
            });
    }

    fn learning_section(&mut self, ui: &mut egui::Ui, sections: &mut SectionState) {
        sections.heading(ui, &tr!("section_learning"));
        ui.checkbox(&mut self.auto_learn, tr!("auto_learn"))
            .on_hover_text(tr!("auto_learn_tip"));
        if !self.auto_learn {
            // Learning by hand is only worth offering while the board could
            // actually be stored, so the button says why it is dark.
            let ready = self.result.is_some()
                && self
                    .derived()
                    .problems
                    .as_ref()
                    .is_ok_and(|problems| problems.is_empty());
            if ui
                .add_enabled(
                    ready,
                    egui::Button::new(icons::label(icons::LEARN, tr!("learn"))),
                )
                .on_hover_text(tr!("learn_now_tip"))
                .on_disabled_hover_text(tr!("learn_needs_valid"))
                .clicked()
            {
                self.learn_manually();
            }
        }
        let count = self.remembered_skins();
        if count == 0 {
            ui.small(tr!("skin_none_learned"));
            return;
        }
        ui.small(tr!("skins_learned", count = count));
        if let Some(matched) = self.result.as_ref().and_then(|result| result.skin.as_ref()) {
            ui.small(tr!(
                "skin_applied",
                name = matched.name.as_str(),
                matched = matched.matched,
                appearances = matched.appearances
            ));
        }
        if ui.small_button(tr!("forget_skins")).clicked() {
            self.forget_skins();
        }
    }
}

/// Red at "barely recognized" through yellow to green at "sure". Walking the
/// hue circle keeps the midpoint vivid, where a straight red-to-green RGB mix
/// would go muddy brown.
fn confidence_color(confidence: f32) -> Color32 {
    let hue = confidence.clamp(0.0, 1.0) * (120.0 / 360.0);
    Color32::from(egui::ecolor::Hsva::new(hue, 0.72, 0.82, 1.0))
}

/// Tracks whether the panel has drawn a section heading yet, so the hairline
/// only separates sections instead of leading the first one.
#[derive(Default)]
struct SectionState {
    started: bool,
}

impl SectionState {
    /// A compact section header. Text only: an icon here would fight the body
    /// for attention, and the muted colour plus the hairline already say
    /// "heading".
    fn heading(&mut self, ui: &mut egui::Ui, title: &str) {
        ui.add_space(6.0);
        if self.started {
            ui.separator();
        }
        self.started = true;
        ui.add_space(2.0);
        // Body size and muted: the headings are signposts, not content.
        let color = ui.visuals().weak_text_color();
        ui.label(RichText::new(title).strong().color(color));
        ui.add_space(2.0);
    }
}

/// What a stepper did this frame. `dragging` means the value is still under the
/// pointer, so a burst of changes can share one undo step.
#[derive(Default, Clone, Copy)]
struct StepperChange {
    changed: bool,
    dragging: bool,
}

impl std::ops::BitOr for StepperChange {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self {
            changed: self.changed || other.changed,
            dragging: self.dragging || other.dragging,
        }
    }
}

/// A `−` / value / `+` triple so a row or column count can be nudged one step
/// at a time or typed directly, without reaching for the keyboard first.
fn count_stepper(ui: &mut egui::Ui, value: &mut u32, id_salt: &str) -> StepperChange {
    let mut change = StepperChange::default();
    ui.push_id(id_salt, |ui| {
        ui.horizontal(|ui| {
            if ui
                .add(egui::Button::new("−").min_size(vec2(26.0, 0.0)))
                .on_hover_text(tr!("decrease"))
                .clicked()
            {
                *value = value.saturating_sub(1).max(1);
                change.changed = true;
            }
            let response = ui.add(
                egui::DragValue::new(value)
                    .speed(0.15)
                    .range(1..=MAX_BOARD_AXIS),
            );
            if response.changed() {
                change.changed = true;
                change.dragging = response.dragged();
            }
            if ui
                .add(egui::Button::new("+").min_size(vec2(26.0, 0.0)))
                .on_hover_text(tr!("increase"))
                .clicked()
            {
                *value = (*value + 1).min(MAX_BOARD_AXIS);
                change.changed = true;
            }
        });
    });
    change
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_colour_runs_red_to_green() {
        let low = confidence_color(0.0);
        let mid = confidence_color(0.5);
        let high = confidence_color(1.0);

        assert!(low.r() > low.g(), "0.0 should read as red: {low:?}");
        assert!(high.g() > high.r(), "1.0 should read as green: {high:?}");
        assert!(
            mid.r() > mid.b() && mid.g() > mid.b(),
            "0.5 should pass through yellow: {mid:?}"
        );
    }

    #[test]
    fn stepper_changes_merge() {
        let merged = StepperChange {
            changed: true,
            dragging: false,
        } | StepperChange {
            changed: false,
            dragging: true,
        };

        assert!(merged.changed && merged.dragging);
    }
}
