//! The image canvas: zooming, panning, the grid overlay and cell painting.

use eframe::egui;
use egui::{
    Align, Button, Color32, CursorIcon, FontId, Layout, PointerButton, Pos2, Rect, Sense, Stroke,
    StrokeKind, Vec2, pos2, vec2,
};
use sokovision_core::{Cell, Grid, RecognitionResult};

use crate::app::{App, History};
use crate::grid;
use crate::i18n::tr;

/// Zoom range the wheel can reach.
const MIN_ZOOM: f32 = 0.2;
const MAX_ZOOM: f32 = 8.0;
/// Cell size in screen pixels below which the XSB symbol is not drawn.
const SYMBOL_MIN_SIZE: f32 = 15.0;
/// Screen pixels left around a hand-drawn grid when the view zooms onto it,
/// enough for the corner handles and the edge bars to stay inside the canvas.
const GRID_FIT_MARGIN: f32 = 24.0;
/// Accent of the outline handles and the rubber band.
const ACCENT: Color32 = Color32::from_rgb(245, 180, 60);
/// The rubber band's translucent fill, a lighter accent than the outline.
const RUBBER_BAND_FILL: Color32 = Color32::from_rgba_unmultiplied_const(245, 180, 60, 40);
/// Guide lines through the pointer while a grid is being drawn.
const GUIDE: Color32 = Color32::from_rgba_unmultiplied_const(245, 180, 60, 200);
/// Dark edge that keeps the accent handles readable on light tile art.
const HANDLE_EDGE: Color32 = Color32::from_rgb(90, 60, 10);
/// XSB glyphs in the order of [`symbol_slot`], with `?` for an unresolved cell.
const SYMBOLS: [&str; 8] = ["-", "#", ".", "$", "*", "@", "+", "?"];

/// Index of a cell's glyph in [`SYMBOLS`].
fn symbol_slot(cell: Cell) -> usize {
    match cell {
        Cell::Empty => 0,
        Cell::Wall => 1,
        Cell::Goal => 2,
        Cell::Box => 3,
        Cell::BoxOnGoal => 4,
        Cell::Player => 5,
        Cell::PlayerOnGoal => 6,
        Cell::Unknown => 7,
    }
}

/// Every glyph laid out once per frame. Laying text out per cell meant six
/// layouts for each cell on the board, every repaint.
struct SymbolGlyphs {
    outline: [std::sync::Arc<egui::Galley>; SYMBOLS.len()],
    fill: [std::sync::Arc<egui::Galley>; SYMBOLS.len()],
}

impl SymbolGlyphs {
    fn new(painter: &egui::Painter, size: f32) -> Self {
        let font = FontId::monospace(size.clamp(9.0, 18.0));
        let layout = |color| {
            SYMBOLS.map(|symbol| painter.layout_no_wrap(symbol.to_owned(), font.clone(), color))
        };
        Self {
            outline: layout(Color32::from_black_alpha(210)),
            fill: layout(Color32::from_rgb(255, 245, 210)),
        }
    }

    /// Ring the glyph in black before drawing it, otherwise a pale character
    /// disappears on a pale tile.
    fn draw(&self, painter: &egui::Painter, rect: Rect, cell: Cell) {
        let index = symbol_slot(cell);
        let center = rect.center();
        for offset in [
            vec2(-1.0, 0.0),
            vec2(1.0, 0.0),
            vec2(0.0, -1.0),
            vec2(0.0, 1.0),
        ] {
            let galley = &self.outline[index];
            painter.galley(
                center + offset - galley.size() * 0.5,
                galley.clone(),
                Color32::WHITE,
            );
        }
        let galley = &self.fill[index];
        painter.galley(center - galley.size() * 0.5, galley.clone(), Color32::WHITE);
    }
}

/// Cursor that says which way the grabbed part of the outline can move.
fn cursor_for(handle: grid::Handle) -> CursorIcon {
    match handle {
        grid::Handle::Corner(grid::Corner::TopLeft | grid::Corner::BottomRight) => {
            CursorIcon::ResizeNwSe
        }
        grid::Handle::Corner(_) => CursorIcon::ResizeNeSw,
        grid::Handle::Edge(grid::Edge::Left | grid::Edge::Right) => CursorIcon::ResizeHorizontal,
        grid::Handle::Edge(_) => CursorIcon::ResizeVertical,
    }
}

/// Where an image lands on the canvas at the current zoom and pan, so a
/// pointer maps to image pixels.
#[derive(Clone, Copy)]
struct View {
    image_size: Vec2,
    scale: f32,
    image_rect: Rect,
}

impl View {
    fn new(image_size: Vec2, area: Rect, zoom: f32, pan: Vec2) -> Self {
        let fit = (area.width() / image_size.x)
            .min(area.height() / image_size.y)
            .max(0.001);
        let scale = fit * zoom;
        Self {
            image_size,
            scale,
            image_rect: Rect::from_center_size(area.center() + pan, image_size * scale),
        }
    }

    /// Unclamped image coordinates under a screen position.
    fn to_image_unclamped(self, pointer: Pos2) -> Pos2 {
        let position = (pointer - self.image_rect.min) / self.scale;
        pos2(position.x, position.y)
    }

    /// Image position under a screen position, clamped to the image.
    fn to_image(self, pointer: Pos2) -> Pos2 {
        let position = self.to_image_unclamped(pointer);
        pos2(
            position.x.clamp(0.0, self.image_size.x),
            position.y.clamp(0.0, self.image_size.y),
        )
    }

    /// Screen rectangle spanned by two image positions.
    fn band(&self, start: Pos2, end: Pos2) -> Rect {
        let to_screen = |point: Pos2| self.image_rect.min + point.to_vec2() * self.scale;
        Rect::from_two_pos(to_screen(start), to_screen(end))
    }
}

/// The rectangle a drag is drawing: a translucent fill with an accent outline.
fn draw_rubber_band(painter: &egui::Painter, band: Rect) {
    painter.rect_filled(band, 0.0, RUBBER_BAND_FILL);
    painter.rect_stroke(band, 0.0, Stroke::new(1.5, ACCENT), StrokeKind::Middle);
}

/// Draws the pixel dimensions of a dragged crop rectangle next to its top-left
/// corner, clamped inside `area` so an edge-to-edge drag never clips its readout.
fn draw_size_badge(painter: &egui::Painter, area: Rect, band: Rect, width: u32, height: u32) {
    if width == 0 || height == 0 {
        return;
    }
    let text = format!("{width} × {height} px");
    let galley = painter.layout_no_wrap(
        text,
        FontId::monospace(12.0),
        Color32::from_rgb(255, 245, 215),
    );
    let padding = vec2(6.0, 3.0);
    let badge_size = galley.size() + padding * 2.0;
    let preferred_y = if band.top() - badge_size.y - 4.0 >= area.top() {
        band.top() - badge_size.y - 4.0
    } else {
        (band.top() + 4.0).min(area.bottom() - badge_size.y)
    };
    let x = band.left().clamp(
        area.left() + 2.0,
        (area.right() - badge_size.x - 2.0).max(area.left()),
    );
    let y = preferred_y.max(area.top() + 2.0);
    let badge_rect = Rect::from_min_size(pos2(x, y), badge_size);
    painter.rect_filled(badge_rect, 4.0, Color32::from_black_alpha(195));
    painter.galley(badge_rect.min + padding, galley, Color32::WHITE);
}

/// Applies pointer-anchored wheel zoom and secondary/middle-drag pan to
/// `(zoom, pan)`, returning the updated [`View`].
fn update_zoom_and_pan(
    ctx: &egui::Context,
    response: &egui::Response,
    image_size: Vec2,
    zoom: &mut f32,
    pan: &mut Vec2,
) -> View {
    let mut view = View::new(image_size, response.rect, *zoom, *pan);
    if response.hovered() {
        let scroll = ctx.input(|input| input.smooth_scroll_delta.y);
        // Zoom about the pointer: the image pixel under the cursor has to
        // stay there, otherwise the board slides away while it grows.
        if scroll != 0.0
            && let Some(pointer) = response.hover_pos()
        {
            let anchor = (pointer - view.image_rect.min) / view.scale;
            *zoom = (*zoom * (scroll * 0.0015).exp()).clamp(MIN_ZOOM, MAX_ZOOM);
            let scale = View::new(image_size, response.rect, *zoom, *pan).scale;
            // Solve `image_rect.min + anchor * scale == pointer` for pan.
            *pan = pointer - response.rect.center() + image_size * scale * 0.5 - anchor * scale;
            view = View::new(image_size, response.rect, *zoom, *pan);
        }
    }
    if response.dragged_by(PointerButton::Secondary) || response.dragged_by(PointerButton::Middle) {
        *pan += response.drag_delta();
        view = View::new(image_size, response.rect, *zoom, *pan);
    }
    view
}

/// Lines through the pointer that span the whole canvas, so a board edge can
/// be lined up with them before and while the rectangle is dragged.
fn draw_guides(painter: &egui::Painter, area: Rect, pointer: Pos2) {
    let stroke = Stroke::new(1.0, GUIDE);
    painter.line_segment(
        [pos2(pointer.x, area.top()), pos2(pointer.x, area.bottom())],
        stroke,
    );
    painter.line_segment(
        [pos2(area.left(), pointer.y), pos2(area.right(), pointer.y)],
        stroke,
    );
}

/// Zoom and pan that put `grid` in the middle of `area`, as large as it fits
/// with [`GRID_FIT_MARGIN`] to spare. Clamped to the wheel's zoom range, which
/// wins for a board far smaller or larger than the window.
fn fit_grid(grid: Grid, image_size: Vec2, area: Rect) -> (f32, Vec2) {
    let fit = (area.width() / image_size.x)
        .min(area.height() / image_size.y)
        .max(0.001);
    // A canvas narrower than the margin needs a smaller one instead of a
    // negative target size.
    let margin = GRID_FIT_MARGIN
        .min(area.width() * 0.25)
        .min(area.height() * 0.25);
    let wanted = ((area.width() - 2.0 * margin) / grid.width())
        .min((area.height() - 2.0 * margin) / grid.height());
    let zoom = (wanted / fit).clamp(MIN_ZOOM, MAX_ZOOM);
    let scale = fit * zoom;
    let middle = vec2(
        grid.origin_x + grid.width() * 0.5,
        grid.origin_y + grid.height() * 0.5,
    );
    (zoom, (image_size * 0.5 - middle) * scale)
}

impl App {
    pub fn draw_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.capture_crop.is_some() {
            self.draw_capture_crop(ui, ctx);
            return;
        }
        let Some((image_size, texture)) = self.image.as_ref().map(|image| {
            (
                vec2(image.rgba.width() as f32, image.rgba.height() as f32),
                image.texture.id(),
            )
        }) else {
            self.draw_empty_hint(ui, ctx);
            return;
        };

        let available = ui.available_size().max(vec2(1.0, 1.0));
        let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
        painter.rect_filled(response.rect, 0.0, ui.visuals().extreme_bg_color);

        let view = update_zoom_and_pan(ctx, &response, image_size, &mut self.zoom, &mut self.pan);

        painter.image(
            texture,
            view.image_rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );

        // The overlay follows the grid being edited only while a handle is
        // dragged; otherwise it mirrors the recognized grid. With no grid at
        // all the canvas switches to drawing one by hand.
        let overlay = self.overlay_grid().map(|grid| {
            if self.grid_drag.is_some() {
                self.grid_edit.unwrap_or(grid)
            } else {
                grid
            }
        });
        let Some(grid) = overlay else {
            if self.drawing_grid {
                self.handle_grid_create(ctx, &response, &painter, view);
            }
            return;
        };

        let hovered_handle = response
            .hover_pos()
            .and_then(|pointer| grid::handle_at(&grid, pointer, view.image_rect, view.scale));

        if let Some(result) = self.result.as_ref() {
            // While the grid is being adjusted the cells still describe the
            // previous outline. Their tint, symbols and highlights would sit on
            // the wrong lines, so only the grid itself is drawn until the read
            // catches up.
            let settled = !self.grid_is_stale();
            let hovered = if settled {
                self.hovered_cell(&response, hovered_handle, view)
            } else {
                None
            };
            let painter = painter.with_clip_rect(view.image_rect);
            self.draw_cells(&painter, result, grid, view, hovered, settled);
        }

        self.draw_handles(&painter, grid, view);

        if let Some(handle) = hovered_handle {
            ctx.set_cursor_icon(cursor_for(handle));
        }

        self.handle_grid_drag(ctx, &response, Some(grid), view);
    }

    /// Rubber band that becomes a grid. Only reachable while there is no grid:
    /// detection found nothing, or the toolbar dropped the old grid, so the
    /// user marks the board by hand instead.
    fn handle_grid_create(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        painter: &egui::Painter,
        view: View,
    ) {
        if ctx.input(|input| input.pointer.primary_pressed()) {
            self.grid_create = response
                .interact_pointer_pos()
                .filter(|pointer| view.image_rect.contains(*pointer))
                .map(|pointer| {
                    let start = view.to_image(pointer);
                    (start, start)
                });
        }
        if response.dragged_by(PointerButton::Primary)
            && let Some((start, _)) = self.grid_create
            && let Some(pointer) = response.interact_pointer_pos()
        {
            self.grid_create = Some((start, view.to_image(pointer)));
        }

        // The guides replace the small crosshair cursor: they run the full
        // canvas so an edge can be matched against the board's outline.
        if let Some(pointer) = response
            .interact_pointer_pos()
            .or_else(|| response.hover_pos())
            .filter(|pointer| response.rect.contains(*pointer))
        {
            draw_guides(painter, response.rect, pointer);
        }
        if let Some((start, end)) = self.grid_create {
            draw_rubber_band(painter, view.band(start, end));
        }

        if response.drag_stopped_by(PointerButton::Primary) {
            if let Some((start, end)) = self.grid_create.take() {
                match grid::region_between(start, end, view.image_size) {
                    Some(region) => {
                        self.create_grid_in(ctx, region);
                        // The dragged rectangle is what the user wants to work
                        // on, so the view follows it instead of staying on the
                        // whole-image fit the rectangle was drawn in.
                        if let Some(grid) = self.overlay_grid() {
                            (self.zoom, self.pan) = fit_grid(grid, view.image_size, response.rect);
                        }
                    }
                    None => self.status = tr!("drag_too_small"),
                }
            }
            return;
        }
        // A press that never turned into a drag leaves no rectangle behind.
        if ctx.input(|input| !input.pointer.primary_down()) {
            self.grid_create = None;
        }
    }

    /// Frozen screenshot crop view: wheel zoom, secondary-drag pan, full-canvas
    /// crosshair guides, image-pixel size badge, and release-to-crop import.
    fn draw_capture_crop(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let available = ui.available_size().max(vec2(1.0, 1.0));
        let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
        painter.rect_filled(response.rect, 0.0, ui.visuals().extreme_bg_color);

        let (view, texture, selection) = {
            let Some(crop) = self.capture_crop.as_mut() else {
                return;
            };
            let image_size = vec2(crop.rgba.width() as f32, crop.rgba.height() as f32);
            let view =
                update_zoom_and_pan(ctx, &response, image_size, &mut crop.zoom, &mut crop.pan);
            if ctx.input(|input| input.pointer.primary_pressed()) {
                crop.selection = response
                    .interact_pointer_pos()
                    .filter(|pointer| view.image_rect.contains(*pointer))
                    .map(|pointer| {
                        let start = view.to_image(pointer);
                        (start, start)
                    });
            }
            if response.dragged_by(PointerButton::Primary)
                && let Some((start, _)) = crop.selection
                && let Some(pointer) = response.interact_pointer_pos()
            {
                crop.selection = Some((start, view.to_image(pointer)));
            }
            (view, crop.texture.id(), crop.selection)
        };

        painter.image(
            texture,
            view.image_rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );

        if let Some(pointer) = response
            .interact_pointer_pos()
            .or_else(|| response.hover_pos())
            .filter(|pointer| response.rect.contains(*pointer))
        {
            draw_guides(&painter, response.rect, pointer);
        }

        if let Some((start, end)) = selection {
            let band = view.band(start, end);
            draw_rubber_band(&painter, band);
            let width = (end.x - start.x).abs().round() as u32;
            let height = (end.y - start.y).abs().round() as u32;
            draw_size_badge(&painter, response.rect, band, width, height);
        }

        if response.drag_stopped_by(PointerButton::Primary) {
            let taken = self
                .capture_crop
                .as_mut()
                .and_then(|crop| crop.selection.take());
            if let Some((start, end)) = taken {
                match grid::region_between(start, end, view.image_size) {
                    Some(region) => {
                        if let Some(crop) = self.capture_crop.take() {
                            let cropped = image::imageops::crop_imm(
                                &crop.rgba,
                                region.x,
                                region.y,
                                region.width,
                                region.height,
                            )
                            .to_image();
                            self.set_image(ctx, cropped);
                        }
                    }
                    None => self.status = tr!("drag_too_small"),
                }
            }
            return;
        }
        if ctx.input(|input| !input.pointer.primary_down())
            && let Some(crop) = self.capture_crop.as_mut()
        {
            crop.selection = None;
        }
    }

    /// Cell under the pointer that a click would paint. `None` while the
    /// outline is being dragged or hovered, so the handles keep their own
    /// cursor and feedback.
    fn hovered_cell(
        &self,
        response: &egui::Response,
        hovered_handle: Option<grid::Handle>,
        view: View,
    ) -> Option<(usize, usize)> {
        if self.grid_drag.is_some() || hovered_handle.is_some() {
            return None;
        }
        let pointer = response.hover_pos()?;
        self.cell_at(pointer, view.image_rect, view.scale)
    }

    fn draw_cells(
        &self,
        painter: &egui::Painter,
        result: &RecognitionResult,
        grid: Grid,
        view: View,
        hovered: Option<(usize, usize)>,
        cells: bool,
    ) {
        let group_appearance = if cells {
            hovered_group(result, hovered, self.batch_by_appearance)
        } else {
            None
        };
        let symbol_width = grid.cell_width * view.scale;
        let glyphs = (cells && symbol_width >= SYMBOL_MIN_SIZE)
            .then(|| SymbolGlyphs::new(painter, symbol_width * 0.42));

        for row in 0..grid.rows {
            for column in 0..grid.columns {
                let rect = grid::cell_rect(&grid, row, column, view.image_rect, view.scale);
                let position = (row as usize, column as usize);
                // A row or column added by hand has no cell until the pending
                // read lands, but its line is part of the grid the user is
                // dragging and has to be there right away.
                let recognized = result
                    .cells
                    .get(position.0)
                    .and_then(|cells| cells.get(position.1));
                let selected_here = cells && self.selected == Some(position);
                let hovered_here = cells && hovered == Some(position);
                let in_group = cells
                    && group_appearance.is_some()
                    && recognized.and_then(|cell| cell.appearance) == group_appearance;

                // The tint follows each cell's own score instead of a fixed
                // threshold: a shaky cell glows, a sure one is left alone, and
                // the strength tracks the score rather than stepping.
                if cells && let Some(recognized) = recognized {
                    let tint = if recognized.cell == Cell::Unknown {
                        Some(Color32::from_rgba_unmultiplied(220, 50, 50, 90))
                    } else {
                        let uncertainty = (1.0 - recognized.confidence).clamp(0.0, 1.0);
                        let alpha = (uncertainty.powf(1.5) * 210.0) as u8;
                        (alpha > 3).then(|| Color32::from_rgba_unmultiplied(232, 92, 58, alpha))
                    };
                    if let Some(tint) = tint {
                        painter.rect_filled(rect, 0.0, tint);
                    }
                }

                // A wash over the whole group, so the reach of a click reads at
                // a glance even on busy tile art.
                if in_group {
                    painter.rect_filled(
                        rect,
                        0.0,
                        Color32::from_rgba_unmultiplied(90, 180, 255, 60),
                    );
                }
                if hovered_here && !selected_here {
                    painter.rect_filled(
                        rect,
                        0.0,
                        Color32::from_rgba_unmultiplied(120, 220, 255, 80),
                    );
                }

                // A dark backing stroke first, so the grid keeps its edge on
                // light screenshots as well as dark ones.
                painter.rect_stroke(
                    rect,
                    0.0,
                    Stroke::new(2.0, Color32::from_black_alpha(110)),
                    StrokeKind::Middle,
                );
                painter.rect_stroke(
                    rect,
                    0.0,
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 110)),
                    StrokeKind::Middle,
                );

                if selected_here {
                    painter.rect_stroke(
                        rect,
                        0.0,
                        Stroke::new(3.0, Color32::from_rgb(80, 230, 120)),
                        StrokeKind::Outside,
                    );
                } else if in_group {
                    painter.rect_stroke(
                        rect,
                        0.0,
                        Stroke::new(2.5, Color32::from_rgb(90, 180, 255)),
                        StrokeKind::Inside,
                    );
                }
                if hovered_here && !selected_here {
                    painter.rect_stroke(
                        rect,
                        0.0,
                        Stroke::new(3.0, Color32::from_rgb(150, 235, 255)),
                        StrokeKind::Outside,
                    );
                }

                if let (Some(glyphs), Some(recognized)) = (&glyphs, recognized) {
                    glyphs.draw(painter, rect, recognized.cell);
                }
            }
        }
    }

    fn draw_handles(&self, painter: &egui::Painter, grid: Grid, view: View) {
        for (_, point) in grid::corner_points(&grid, view.image_rect, view.scale) {
            let handle = Rect::from_center_size(point, vec2(12.0, 12.0));
            painter.rect_filled(handle, 2.0, ACCENT);
            painter.rect_stroke(
                handle,
                2.0,
                Stroke::new(1.5, HANDLE_EDGE),
                StrokeKind::Outside,
            );
        }

        // Small bars on the middle of each side, so it is visible that a single
        // edge can be dragged and not just the corners.
        let outline = grid::outline_rect(&grid, view.image_rect, view.scale);
        for (center, size) in [
            (pos2(outline.center().x, outline.min.y), vec2(16.0, 4.0)),
            (pos2(outline.center().x, outline.max.y), vec2(16.0, 4.0)),
            (pos2(outline.min.x, outline.center().y), vec2(4.0, 16.0)),
            (pos2(outline.max.x, outline.center().y), vec2(4.0, 16.0)),
        ] {
            let handle = Rect::from_center_size(center, size);
            painter.rect_filled(handle, 1.5, ACCENT);
            painter.rect_stroke(
                handle,
                1.5,
                Stroke::new(1.0, HANDLE_EDGE),
                StrokeKind::Outside,
            );
        }
    }

    /// Outline dragging and cell painting, in that order of priority. A handle
    /// only wins once the press turns into a drag, so clicking a border cell
    /// still paints it.
    fn handle_grid_drag(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        overlay: Option<Grid>,
        view: View,
    ) {
        /// Screen-pixel movement on a grabbed handle that starts resizing right
        /// away (finer than egui's ~6 px drag slop, while leaving 0 px clicks to
        /// paint border cells).
        const HANDLE_DRAG_THRESHOLD: f32 = 1.5;

        // Hit test where the press started: by the time egui reports a drag the
        // pointer has already left the small handle, which used to fall through
        // to painting the cell underneath. The grab offset uses unclamped image
        // coordinates so grabbing the outer half of a border handle stays exact.
        if ctx.input(|input| input.pointer.primary_pressed()) {
            self.grid_press = response.interact_pointer_pos().and_then(|pointer| {
                let grid = overlay?;
                let handle = grid::handle_at(&grid, pointer, view.image_rect, view.scale)?;
                let grab = grid::handle_position(&grid, handle) - view.to_image_unclamped(pointer);
                Some((handle, grab, pointer, grid))
            });
            // A hand on the outline is already an explicit statement of
            // intent, before egui has reported a drag: a scan or pending settle
            // landing in between would otherwise pull the grid out from under
            // the pointer and the drag would continue on the bounced outline.
            if self.grid_press.is_some() {
                self.cancel_grid_search();
                self.cancel_grid_settle();
            }
        }
        if self.grid_drag.is_none()
            && let Some((handle, _, press_pos, _)) = self.grid_press
        {
            let moved = response
                .interact_pointer_pos()
                .is_some_and(|pointer| pointer.distance(press_pos) >= HANDLE_DRAG_THRESHOLD);
            if response.drag_started_by(PointerButton::Primary)
                || (ctx.input(|input| input.pointer.primary_down()) && moved)
            {
                self.grid_drag = Some(handle);
                self.cancel_grid_settle();
            }
        }

        if let Some(handle) = self.grid_drag {
            self.drag_handle(ctx, response, handle, view);
            return;
        }

        let clicked = response.clicked();
        let dragged = response.dragged_by(PointerButton::Primary);
        if (clicked || dragged)
            && let Some(grid) = overlay
            && let Some(pointer) = response.interact_pointer_pos()
            && Self::cell_in_grid(grid, pointer, view.image_rect, view.scale).is_some()
        {
            // The cells still belong to the board before a pending row or
            // column change; read them now so a click paints what is on screen
            // and records its undo step after the settled grid.
            self.settle_grid_now();
            if let Some((row, column)) = self.cell_at(pointer, view.image_rect, view.scale) {
                self.paint_from_canvas(row, column, dragged);
            }
        }

        if ctx.input(|input| !input.pointer.primary_down()) {
            self.grid_press = None;
            self.finish_paint_stroke();
        }
    }

    fn drag_handle(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        handle: grid::Handle,
        view: View,
    ) {
        let primary_down = ctx.input(|input| input.pointer.primary_down());
        let stopped = response.drag_stopped_by(PointerButton::Primary) || !primary_down;
        let pointer = response
            .interact_pointer_pos()
            .or_else(|| ctx.input(|input| input.pointer.interact_pos()));
        // Compute from the original `start_grid` and clamp the target handle
        // position to `[0, image_size]` *before* resizing, so releasing near an
        // image border never shifts the opposite edge or drops a row/column.
        if (primary_down || stopped)
            && let Some(pointer) = pointer
            && let Some((_, grab, _, start_grid)) = self.grid_press
        {
            self.cancel_grid_search();
            let raw = view.to_image_unclamped(pointer) + grab;
            let position = pos2(
                raw.x.clamp(0.0, view.image_size.x),
                raw.y.clamp(0.0, view.image_size.y),
            );
            self.grid_edit = Some(grid::resize(start_grid, handle, position));
        }
        if !stopped {
            return;
        }

        self.grid_drag = None;
        self.grid_press = None;
        let Some(grid) = self.grid_edit.and_then(|grid| self.clamp_grid(grid)) else {
            self.grid_edit = self.result.as_ref().map(|result| result.grid);
            return;
        };
        // A hand-resized outline is an explicit statement of intent: read the
        // board with exactly the grid the user dragged. Re-detecting inside
        // the new rectangle used to adopt nearby row and column counts, and
        // the neighbour scan that followed could change them once more, so the
        // cells visibly jumped away from the released outline.
        self.cancel_grid_search();
        self.cancel_grid_settle();
        self.grid_edit = Some(grid);
        self.region = Some(grid::region(
            grid,
            view.image_size.x.round() as u32,
            view.image_size.y.round() as u32,
        ));
        if self.recognize_with(Some(grid), None, History::Record)
            && let Some(result) = self.result.as_ref()
        {
            self.status = tr!(
                "grid_resized",
                rows = result.grid.rows,
                columns = result.grid.columns,
                width = format!("{:.1}", result.grid.cell_width),
                height = format!("{:.1}", result.grid.cell_height)
            );
        }
    }

    /// The empty canvas reads as one sentence, with the three verbs as buttons.
    fn draw_empty_hint(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        const HEIGHT: f32 = 24.0;
        let open_label = tr!("empty_open");
        let paste_label = tr!("empty_paste");
        let screenshot_label = tr!("screenshot");
        let separator = tr!("empty_separator");
        let hint = tr!("empty_hint");
        let open = button_width(ui, &open_label);
        let paste = button_width(ui, &paste_label);
        let screenshot_width = button_width(ui, &screenshot_label);
        let spacing = ui.spacing().item_spacing.x;
        // A centred vertical layout stretches every child to the full width, so
        // the row has to be measured up front to keep it in the middle.
        let row_width = open
            + text_width(ui, &separator)
            + paste
            + text_width(ui, &hint)
            + screenshot_width
            + spacing * 4.0;

        ui.vertical_centered(|ui| {
            ui.add_space(((ui.available_height() - HEIGHT) * 0.5).max(0.0));
            ui.allocate_ui_with_layout(
                vec2(row_width, HEIGHT),
                Layout::left_to_right(Align::Center),
                |ui| {
                    let button = |ui: &mut egui::Ui, label: &str, width: f32| {
                        ui.add(Button::new(label).min_size(vec2(width, HEIGHT)))
                            .clicked()
                    };
                    if button(ui, &open_label, open) {
                        self.open_dialog(ctx);
                    }
                    ui.label(&separator);
                    if button(ui, &paste_label, paste) {
                        self.paste_image(ctx);
                    }
                    ui.label(&hint);
                    if button(ui, &screenshot_label, screenshot_width) {
                        self.request_screenshot(ctx);
                    }
                },
            );
        });
    }
}

/// Width of a run of body text, so a one-line hint can be measured before any
/// of it is laid out.
fn text_width(ui: &egui::Ui, text: &str) -> f32 {
    let font_id = egui::TextStyle::Body.resolve(ui.style());
    ui.painter()
        .layout_no_wrap(text.to_owned(), font_id, Color32::WHITE)
        .size()
        .x
}

/// Width a button gets when it is given this label: a hair wider than both its
/// text and the minimum interact size, so `min_size` always decides the width
/// and the measured value is the one that ends up on screen.
fn button_width(ui: &egui::Ui, text: &str) -> f32 {
    let spacing = ui.spacing();
    let text = text_width(ui, text) + spacing.button_padding.x * 2.0;
    text.max(spacing.interact_size.x) + 4.0
}

/// Appearance group previewed under the pointer. Only 「同外观」 has a group to
/// show, and only while the pointer is on a cell, so the highlight leaves with
/// the pointer instead of sticking to the last pick.
fn hovered_group(
    result: &RecognitionResult,
    hovered: Option<(usize, usize)>,
    batch: bool,
) -> Option<u32> {
    if !batch {
        return None;
    }
    let (row, column) = hovered?;
    result.cells.get(row)?.get(column)?.appearance
}

#[cfg(test)]
mod tests {
    use super::*;
    use sokovision_core::RecognizedCell;

    fn recognized(appearance: Option<u32>) -> RecognizedCell {
        RecognizedCell {
            cell: Cell::Box,
            confidence: 1.0,
            alternatives: Vec::new(),
            appearance,
        }
    }

    fn result() -> RecognitionResult {
        RecognitionResult {
            grid: Grid::new(0.0, 0.0, 10.0, 10.0, 2, 2),
            cells: vec![
                vec![recognized(Some(0)), recognized(Some(0))],
                vec![recognized(Some(1)), recognized(None)],
            ],
            warnings: Vec::new(),
            skin: None,
            relaxed: false,
        }
    }

    #[test]
    fn the_group_highlight_follows_only_the_hovered_cell() {
        let result = result();

        assert_eq!(
            hovered_group(&result, Some((1, 0)), true),
            Some(1),
            "「同外观」highlights the hovered cell's group"
        );
        assert_eq!(
            hovered_group(&result, Some((1, 0)), false),
            None,
            "「单格」highlights no group"
        );
        assert_eq!(
            hovered_group(&result, None, true),
            None,
            "leaving the canvas clears the group instead of keeping the last pick"
        );
        assert_eq!(
            hovered_group(&result, Some((1, 1)), true),
            None,
            "a cell without an appearance has no group"
        );
    }
}
