//! Application state and the actions the UI triggers.

use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::Instant;

use eframe::egui;
use egui::{TextureHandle, TextureOptions, Vec2};
use sokovision_core::{
    Cell, Grid, PixelRect, RecognitionResult, RecognizeOptions, SkinLibrary, recognize,
    refine_grid, validate_board,
};

use crate::grid::{self, Handle};
use crate::i18n::{self, cell_label, tr};
use crate::skins::{self, SkinSaver};

/// Image opened in the app, with its uploaded texture.
pub(crate) struct LoadedImage {
    /// Shared with the grid search worker, which must not copy the pixels.
    pub rgba: Arc<image::RgbaImage>,
    pub texture: TextureHandle,
}

/// Frozen desktop screenshot shown on the canvas while the user drags a crop
/// rectangle around the puzzle (or presses `Esc` to keep the previous board).
pub(crate) struct CaptureCrop {
    pub rgba: image::RgbaImage,
    pub texture: TextureHandle,
    pub zoom: f32,
    pub pan: Vec2,
    pub selection: Option<(egui::Pos2, egui::Pos2)>,
}

/// Whether a recognition run pushes the previous board onto the undo stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum History {
    Record,
    Keep,
}

/// A background scan over the grid resolutions next to the detected one.
struct GridSearch {
    receiver: Receiver<(u64, Option<RecognitionResult>)>,
}

/// Seconds the row or column count has to stay still before the board is read
/// again. A dragged stepper reports a change every frame, and reading the
/// whole board is far too slow to run at frame rate.
const GRID_SETTLE_DELAY: f64 = 0.5;

/// What the side panel shows about the current board. Building these walks the
/// whole board and re-parses the level, so they are kept until the board or the
/// trim choice changes instead of being rebuilt on every repaint.
pub(crate) struct Derived {
    pub xsb: Result<String, String>,
    pub problems: Result<Vec<String>, String>,
    pub preview: String,
    pub confidence: Option<f32>,
}

impl Derived {
    fn new(result: Option<&RecognitionResult>, trim: bool) -> Self {
        let Some(result) = result else {
            return Self {
                xsb: Err(tr!("no_level")),
                problems: Err(tr!("no_level")),
                preview: String::new(),
                confidence: None,
            };
        };
        let board = result.board();
        // Validation looks at the whole board, not the trimmed export, so the
        // verdict does not change with the export checkbox.
        let problems = validate_board(&board);
        let export = if trim { board.trimmed() } else { board };
        let has_cells = result.cells.iter().flatten().next().is_some();
        Self {
            xsb: export.to_xsb().map_err(|error| error.to_string()),
            problems: Ok(problems),
            preview: export.to_xsb_lossy(),
            confidence: has_cells.then(|| result.quality()),
        }
    }
}

pub(crate) struct App {
    pub image: Option<LoadedImage>,
    pub result: Option<RecognitionResult>,
    /// Export, validation and score of `result`, cached between repaints.
    derived: Option<Derived>,
    /// Grid being edited; follows `result.grid` until the user drags a handle.
    pub grid_edit: Option<Grid>,
    /// Set while a stepper drag keeps re-recognizing: the undo entry is taken
    /// once for the whole gesture instead of once per intermediate count.
    grid_history_open: bool,
    /// Input time of the last hand change to the row or column count. The board
    /// is read again only once the count has been still for
    /// [`GRID_SETTLE_DELAY`]; reading it per frame is what made a drag stutter.
    grid_settle_at: Option<f64>,
    pub grid_drag: Option<Handle>,
    /// Outline part the current primary press started on, plus the offset from
    /// the pointer to that edge in image pixels. A drag moves the edge by the
    /// pointer delta, so grabbing a handle slightly off its line does not
    /// shift the whole grid.
    pub grid_press: Option<(Handle, Vec2)>,
    pub region: Option<PixelRect>,
    /// Rubber band the user is drawing to create a grid, in image coordinates.
    /// Used while there is no grid: detection found nothing, or the toolbar
    /// dropped the old one.
    pub grid_create: Option<(egui::Pos2, egui::Pos2)>,
    /// Set by the manual grid action. The canvas only turns a drag into a grid
    /// while this is on, so an image whose detection failed stays as it is
    /// instead of silently switching to drawing mode.
    pub drawing_grid: bool,
    /// Whether `start_hand_drawn_grid` pushed a board onto `history`, so `Esc`
    /// can restore that board cleanly without leaving a dead undo entry.
    draw_grid_pushed: bool,
    pub paint_cell: Cell,
    pub batch_by_appearance: bool,
    /// Set while a primary drag paints cells, so the whole stroke shares one
    /// undo entry and defers skin learning until the button comes up.
    paint_history_open: bool,
    /// Last cell painted during the current drag, so pointer jitter inside one
    /// cell does not re-paint and re-validate on every frame.
    last_drag_painted: Option<(usize, usize)>,
    pub selected: Option<(usize, usize)>,
    pub zoom: f32,
    pub pan: Vec2,
    history: VecDeque<RecognitionResult>,
    redo: VecDeque<RecognitionResult>,
    pub trim_export: bool,
    pub status: String,
    capture_started: Option<Instant>,
    pub(crate) capture_crop: Option<CaptureCrop>,
    grid_search: Option<GridSearch>,
    /// Bumped whenever the grid is edited by hand, so a scan that started
    /// before the edit cannot pull the grid out from under the user.
    search_token: u64,
    /// Remember a valid board as soon as it is recognized. Off leaves storing
    /// to the Learning section's button, so the grid can be adjusted without
    /// filing every intermediate resolution as a skin.
    pub auto_learn: bool,
    /// Skins learned from completed boards, and where they live on disk.
    skins: SkinLibrary,
    skin_saver: Option<Arc<SkinSaver>>,
    /// A learned change waiting to be written; writing on every click would
    /// stall the UI on slow or scanned disks.
    skins_dirty: bool,
    skins_changed_at: Option<Instant>,
}

impl App {
    pub fn new(creation_context: &eframe::CreationContext<'_>) -> Self {
        crate::install_cjk_font(&creation_context.egui_ctx);
        crate::install_theme(&creation_context.egui_ctx);
        i18n::apply_system_language();
        let skins_path = skins::library_path();
        let mut skins = skins_path
            .as_deref()
            .and_then(skins::load)
            .unwrap_or_default();
        // Prototypes from older builds are not converted; drop them and let
        // the next save rewrite the file in the current format.
        let pruned = skins.prune();
        let skin_saver = skins_path.map(|path| Arc::new(SkinSaver::new(path)));
        let mut app = Self::base(skins, skin_saver, pruned);
        if let Some(path) = std::env::args_os().nth(1) {
            app.open_path(&creation_context.egui_ctx, PathBuf::from(path));
        }
        app
    }

    /// App with no picture. Shared with the tests, so a new field is only
    /// listed here instead of in every constructor.
    fn base(skins: SkinLibrary, skin_saver: Option<Arc<SkinSaver>>, skins_dirty: bool) -> Self {
        Self {
            image: None,
            result: None,
            derived: None,
            grid_edit: None,
            grid_history_open: false,
            grid_settle_at: None,
            grid_drag: None,
            grid_press: None,
            region: None,
            grid_create: None,
            drawing_grid: false,
            draw_grid_pushed: false,
            paint_cell: Cell::Wall,
            batch_by_appearance: true,
            paint_history_open: false,
            last_drag_painted: None,
            selected: None,
            zoom: 1.0,
            pan: Vec2::ZERO,
            history: VecDeque::new(),
            redo: VecDeque::new(),
            trim_export: true,
            status: tr!("start_hint"),
            capture_started: None,
            capture_crop: None,
            grid_search: None,
            search_token: 0,
            auto_learn: true,
            skins,
            skin_saver,
            skins_dirty,
            skins_changed_at: skins_dirty.then(Instant::now),
        }
    }

    pub fn set_image(&mut self, ctx: &egui::Context, rgba: image::RgbaImage) {
        if rgba.width() == 0 || rgba.height() == 0 {
            self.status = tr!("empty_image");
            return;
        }
        let color_image = egui::ColorImage::from_rgba_unmultiplied(
            [rgba.width() as usize, rgba.height() as usize],
            rgba.as_raw(),
        );
        let texture = ctx.load_texture("source", color_image, TextureOptions::LINEAR);
        self.capture_crop = None;
        self.image = Some(LoadedImage {
            rgba: Arc::new(rgba),
            texture,
        });
        self.result = None;
        self.invalidate_derived();
        self.grid_edit = None;
        self.grid_history_open = false;
        self.grid_settle_at = None;
        self.grid_drag = None;
        self.grid_press = None;
        self.region = None;
        self.grid_create = None;
        self.drawing_grid = false;
        self.draw_grid_pushed = false;
        self.paint_history_open = false;
        self.last_drag_painted = None;
        self.selected = None;
        self.history.clear();
        self.redo.clear();
        self.zoom = 1.0;
        self.pan = Vec2::ZERO;
        self.cancel_grid_search();
        self.run_recognition(ctx, None);
    }

    /// Recognizes the whole image, or the given grid, or the selected region.
    pub fn run_recognition(&mut self, ctx: &egui::Context, grid: Option<Grid>) -> bool {
        self.drawing_grid = false;
        self.draw_grid_pushed = false;
        self.cancel_grid_settle();
        match (grid, self.region) {
            (Some(grid), _) => {
                let Some(grid) = self.clamp_grid(grid) else {
                    self.status = tr!("grid_out_of_image");
                    return false;
                };
                self.grid_edit = Some(grid);
                self.recognize_with(Some(grid), None, History::Record)
            }
            (None, Some(region)) => {
                if self.recognize_region(region) {
                    // Detection only gets close; the neighbourhood scan looks
                    // for a resolution that explains the pixels better.
                    self.start_grid_search(ctx);
                    true
                } else {
                    self.status = tr!("region_no_grid");
                    false
                }
            }
            (None, None) => {
                let recognized = self.recognize_with(None, None, History::Record);
                if recognized {
                    self.start_grid_search(ctx);
                }
                recognized
            }
        }
    }

    /// Detects the grid resolution inside the dragged region and recognizes the
    /// board with that grid.
    pub fn recognize_region(&mut self, region: PixelRect) -> bool {
        let Some(grid) = self
            .detect_region_grid(region)
            .and_then(|detected| grid::snap_to_region(region, detected))
            .and_then(|grid| self.clamp_grid(grid))
        else {
            return false;
        };
        self.recognize_with(Some(grid), None, History::Record)
    }

    /// Grid the canvas draws and the panel edits. `None` until the user has a
    /// grid: detection is not run automatically any more, and the canvas asks
    /// for a rectangle instead.
    pub fn overlay_grid(&self) -> Option<Grid> {
        self.grid_edit
            .or_else(|| self.result.as_ref().map(|result| result.grid))
    }

    /// Searches `region` for a board and adopts it as the grid. When nothing is
    /// detected there, the rectangle itself becomes the grid, so the row and
    /// column counts can still be fixed by hand.
    pub fn create_grid_in(&mut self, ctx: &egui::Context, region: PixelRect) {
        self.drawing_grid = false;
        self.draw_grid_pushed = false;
        self.cancel_grid_search();
        self.cancel_grid_settle();
        self.region = Some(region);
        if self.recognize_region(region) {
            if let Some(result) = self.result.as_ref() {
                self.status = tr!(
                    "grid_created",
                    rows = result.grid.rows,
                    columns = result.grid.columns
                );
            }
        } else {
            self.region = None;
            let grid = grid::covering(region);
            self.grid_edit = Some(grid);
            self.recognize_with(Some(grid), None, History::Record);
            self.status = tr!("grid_fallback");
        }
        self.start_grid_search(ctx);
    }

    /// The panel's 「手动框选」. A hand-drawn rectangle *is* the grid, so the
    /// grid on screen goes first and the canvas falls back to the rubber band;
    /// the dropped board stays on the undo stack, so a stray click is cheap.
    pub fn start_hand_drawn_grid(&mut self) {
        if self.image.is_none() {
            self.status = tr!("open_first");
            return;
        }
        self.settle_grid_now();
        let before_len = self.history.len();
        self.push_history();
        self.draw_grid_pushed = self.history.len() > before_len || self.result.is_some();
        self.cancel_grid_search();
        self.cancel_grid_settle();
        self.drawing_grid = true;
        self.grid_create = None;
        self.grid_edit = None;
        self.grid_drag = None;
        self.grid_press = None;
        self.region = None;
        self.paint_history_open = false;
        self.last_drag_painted = None;
        self.selected = None;
        self.result = None;
        self.invalidate_derived();
        self.status = tr!("drag_rectangle_hint");
    }

    /// Cancels `drawing_grid` (on `Esc`), restoring any board dropped when hand
    /// drawing started without leaving a dead entry on the undo stack.
    pub fn cancel_hand_drawn_grid(&mut self, ctx: &egui::Context) -> bool {
        if !self.drawing_grid {
            return false;
        }
        self.drawing_grid = false;
        self.grid_create = None;
        if self.draw_grid_pushed
            && let Some(previous) = self.history.pop_back()
        {
            self.grid_edit = Some(previous.grid);
            self.result = Some(previous);
            self.invalidate_derived();
        }
        self.draw_grid_pushed = false;
        self.status = tr!("draw_cancelled");
        ctx.request_repaint();
        true
    }

    pub(crate) fn clamp_grid(&self, grid: Grid) -> Option<Grid> {
        let image = self.image.as_ref()?;
        grid.clamp_to_image(image.rgba.width(), image.rgba.height())
    }

    fn detect_region_grid(&self, region: PixelRect) -> Option<Grid> {
        let image = self.image.as_ref()?;
        let options = RecognizeOptions {
            region: Some(region),
            grid_only: true,
            skins: None,
            ..RecognizeOptions::default()
        };
        recognize(&image.rgba, &options)
            .ok()
            .map(|result| result.grid)
    }

    /// Recognizes with an optional fixed grid or, when no grid is given, lets
    /// the detector resolve the resolution inside `region`.
    pub fn recognize_with(
        &mut self,
        grid: Option<Grid>,
        region: Option<PixelRect>,
        history: History,
    ) -> bool {
        let Some(image) = self.image.as_ref() else {
            self.status = tr!("open_first");
            return false;
        };
        let options = RecognizeOptions {
            region,
            grid,
            grid_only: false,
            skins: (!self.skins.skins.is_empty()).then(|| self.skins.clone()),
        };
        match recognize(&image.rgba, &options) {
            Ok(result) => {
                if history == History::Record && self.result.is_some() {
                    self.push_history();
                }
                self.grid_edit = Some(result.grid);
                let state = match result.unknown_cells().len() {
                    0 => tr!("all_cells_confirmed"),
                    unknown => tr!("cells_need_review", count = unknown),
                };
                self.status = tr!(
                    "recognized",
                    rows = result.grid.rows,
                    columns = result.grid.columns,
                    state = state
                );
                self.result = Some(result);
                self.invalidate_derived();
                // Learning waits for a settled grid: a stepper drag would
                // otherwise remember every intermediate resolution.
                if !self.grid_history_open && self.auto_learn {
                    self.learn_current(true);
                }
                true
            }
            Err(error) => {
                self.status = tr!("recognize_failed", error = error);
                false
            }
        }
    }

    pub fn push_history(&mut self) {
        self.redo.clear();
        let Some(result) = self.result.as_ref() else {
            return;
        };
        if self.history.back() == Some(result) {
            return;
        }
        self.history.push_back(result.clone());
        if self.history.len() > 100 {
            self.history.pop_front();
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.history.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self, ctx: &egui::Context) {
        self.cancel_grid_search();
        self.cancel_grid_settle();
        let Some(result) = self.history.pop_back() else {
            self.status = tr!("nothing_to_undo");
            return;
        };
        if let Some(current) = self.result.take()
            && current != result
            && self.redo.back() != Some(&current)
        {
            self.redo.push_back(current);
            if self.redo.len() > 100 {
                self.redo.pop_front();
            }
        }
        self.drawing_grid = false;
        self.draw_grid_pushed = false;
        self.grid_create = None;
        self.grid_edit = Some(result.grid);
        self.grid_history_open = false;
        self.paint_history_open = false;
        self.last_drag_painted = None;
        self.grid_drag = None;
        self.grid_press = None;
        self.selected = None;
        self.result = Some(result);
        self.invalidate_derived();
        self.status = tr!("undone");
        ctx.request_repaint();
    }

    pub fn redo(&mut self, ctx: &egui::Context) {
        self.cancel_grid_search();
        self.cancel_grid_settle();
        let Some(result) = self.redo.pop_back() else {
            self.status = tr!("nothing_to_redo");
            return;
        };
        if let Some(current) = self.result.take()
            && current != result
            && self.history.back() != Some(&current)
        {
            self.history.push_back(current);
            if self.history.len() > 100 {
                self.history.pop_front();
            }
        }
        self.drawing_grid = false;
        self.draw_grid_pushed = false;
        self.grid_create = None;
        self.grid_edit = Some(result.grid);
        self.grid_history_open = false;
        self.paint_history_open = false;
        self.last_drag_painted = None;
        self.grid_drag = None;
        self.grid_press = None;
        self.selected = None;
        self.result = Some(result);
        self.invalidate_derived();
        self.status = tr!("redone");
        ctx.request_repaint();
    }

    /// Current board, export text, validation verdict and score. Rebuilt only
    /// after a change that made `derived` stale.
    pub fn derived(&mut self) -> &Derived {
        if self.derived.is_none() {
            let derived = Derived::new(self.result.as_ref(), self.trim_export);
            self.derived = Some(derived);
        }
        self.derived.as_ref().expect("derived was just filled")
    }

    pub(crate) fn invalidate_derived(&mut self) {
        self.derived = None;
    }
    pub fn xsb(&mut self) -> Result<String, String> {
        self.derived().xsb.clone()
    }

    /// Mean per-cell score across the whole board — the one number the
    /// confidence bar reports.
    pub fn overall_confidence(&mut self) -> Option<f32> {
        self.derived().confidence
    }

    pub fn is_searching(&self) -> bool {
        self.grid_search.is_some()
    }

    /// How many skins the app has learned so far.
    pub fn remembered_skins(&self) -> usize {
        self.skins.skins.len()
    }

    /// Drops every learned skin and the file behind them.
    pub fn forget_skins(&mut self) {
        if self.skins.skins.is_empty() {
            return;
        }
        self.skins = SkinLibrary::default();
        self.write_skins_now();
        self.skins_dirty = false;
        self.skins_changed_at = None;
        self.status = tr!("skins_forgotten");
    }

    /// Remembers the tiles of the current board once it is complete and valid.
    /// Corrections arrive here too, so a fixed label replaces an earlier guess.
    /// Automatic learning additionally refuses relaxed or unresolved reads, so
    /// a best-effort guess is never filed as a known tile; the manual button
    /// trusts the user and only requires a structurally valid board.
    /// Returns whether the library stored anything it did not have before.
    fn learn_current(&mut self, automatic: bool) -> bool {
        let trusted = self.result.as_ref().is_some_and(|result| {
            if automatic {
                result.learnable()
            } else {
                validate_board(&result.board()).is_empty()
            }
        });
        if !trusted {
            return false;
        }
        let skin = {
            let result = self.result.as_ref().expect("validation needs a result");
            let Some(image) = self.image.as_ref() else {
                return false;
            };
            let name = tr!("skin_default_name", index = self.skins.skins.len() + 1);
            result.skin(&image.rgba, name)
        };
        let Some(skin) = skin else {
            return false;
        };
        if self.skins.remember(skin) == Some(true) {
            self.skins_dirty = true;
            self.skins_changed_at = Some(Instant::now());
            true
        } else {
            false
        }
    }

    /// The Learning section's button. Learns the current board on demand,
    /// which is how the skins of a hand-tuned grid are finally stored when
    /// automatic learning is off.
    pub fn learn_manually(&mut self) {
        // Learn the grid on screen, even when the settle read is still due.
        self.settle_grid_now();
        self.status = if self.learn_current(false) {
            tr!("skin_learned_now")
        } else {
            tr!("skin_unchanged")
        };
    }

    /// Writes learned skins once the board has been quiet for a moment, so a
    /// burst of corrections does not repeatedly block the UI on disk writes.
    fn flush_skins(&mut self, ctx: &egui::Context) {
        if !self.skins_dirty {
            return;
        }
        const SAVE_DELAY: std::time::Duration = std::time::Duration::from_secs(1);
        if let Some(changed_at) = self.skins_changed_at {
            let elapsed = changed_at.elapsed();
            if elapsed < SAVE_DELAY {
                ctx.request_repaint_after(SAVE_DELAY - elapsed);
                return;
            }
        }
        self.save_skins();
        self.skins_dirty = false;
        self.skins_changed_at = None;
    }

    fn save_skins(&self) {
        let Some(saver) = self.skin_saver.as_ref() else {
            return;
        };
        let Ok(text) = serde_json::to_string(&self.skins) else {
            return;
        };
        saver.save(text);
    }

    /// Blocking write used only when the window closes.
    fn write_skins_now(&self) {
        let Some(saver) = self.skin_saver.as_ref() else {
            return;
        };
        let Ok(text) = serde_json::to_string(&self.skins) else {
            return;
        };
        saver.save_now(&text);
    }

    pub fn copy_xsb(&mut self) {
        let Some(xsb) = self.export_xsb(tr!("cannot_copy")) else {
            return;
        };
        match arboard::Clipboard::new().and_then(|mut clipboard| clipboard.set_text(xsb)) {
            Ok(()) => self.status = tr!("xsb_copied"),
            Err(error) => self.status = tr!("copy_failed", error = error),
        }
    }

    pub fn save_xsb(&mut self) {
        let Some(xsb) = self.export_xsb(tr!("cannot_save")) else {
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter(tr!("xsb_filter"), &["xsb", "txt"])
            .set_file_name("level.xsb")
            .save_file()
        else {
            return;
        };
        match fs::write(&path, format!("{xsb}\n")) {
            Ok(()) => self.status = tr!("saved_to", path = path.display()),
            Err(error) => self.status = tr!("save_failed", error = error),
        }
    }

    /// XSB text of the export board, reporting the failure in the status bar.
    fn export_xsb(&mut self, action: String) -> Option<String> {
        // The export has to describe the grid currently on screen, not the one
        // before the last nudge to the row or column count.
        self.settle_grid_now();
        match self.xsb() {
            Ok(xsb) => Some(xsb),
            Err(error) => {
                self.status = tr!("action_failed", action = action, error = error);
                None
            }
        }
    }

    pub fn open_dialog(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter(tr!("image_filter"), &["png", "jpg", "jpeg", "bmp"])
            .pick_file()
        else {
            return;
        };
        self.open_path(ctx, path);
    }

    pub fn open_path(&mut self, ctx: &egui::Context, path: PathBuf) {
        match image::open(&path) {
            Ok(image) => self.set_image(ctx, image.to_rgba8()),
            Err(error) => {
                self.status = tr!("cannot_open", path = path.display(), error = error);
            }
        }
    }

    pub fn request_screenshot(&mut self, ctx: &egui::Context) {
        self.capture_started = Some(Instant::now());
        self.status = tr!("capturing");
        ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
        ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaintWhileHidden);
        ctx.request_repaint();
    }

    /// Captures all displays once the window had time to hide and freezes the
    /// combined desktop on the canvas for interactive cropping.
    pub fn finish_screenshot(&mut self, ctx: &egui::Context) {
        let captured = capture_screen();
        ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
        match captured {
            Ok(image) => self.begin_capture_crop(ctx, image),
            Err(error) => self.status = error,
        }
    }

    /// Enters frozen-capture crop mode with `rgba` without discarding any board
    /// already open underneath.
    pub fn begin_capture_crop(&mut self, ctx: &egui::Context, rgba: image::RgbaImage) {
        if rgba.width() == 0 || rgba.height() == 0 {
            self.status = tr!("empty_image");
            return;
        }
        let color_image = egui::ColorImage::from_rgba_unmultiplied(
            [rgba.width() as usize, rgba.height() as usize],
            rgba.as_raw(),
        );
        let texture = ctx.load_texture("capture_preview", color_image, TextureOptions::LINEAR);
        self.capture_crop = Some(CaptureCrop {
            rgba,
            texture,
            zoom: 1.0,
            pan: Vec2::ZERO,
            selection: None,
        });
        self.status = tr!("capture_crop_hint");
        ctx.request_repaint();
    }

    /// Whether the canvas is currently showing a frozen screenshot waiting to
    /// be cropped.
    pub fn is_cropping_capture(&self) -> bool {
        self.capture_crop.is_some()
    }

    /// Zoom factor of whichever view is active on the canvas.
    pub fn active_zoom(&self) -> f32 {
        self.capture_crop
            .as_ref()
            .map_or(self.zoom, |crop| crop.zoom)
    }

    /// Imports the entire frozen screenshot without cropping.
    pub fn import_full_capture(&mut self, ctx: &egui::Context) {
        if let Some(crop) = self.capture_crop.take() {
            self.set_image(ctx, crop.rgba);
        }
    }

    /// Dismisses frozen-capture crop mode and returns to the previous image.
    pub fn cancel_capture_crop(&mut self, ctx: &egui::Context) -> bool {
        if self.capture_crop.take().is_some() {
            self.status = tr!("capture_cancelled");
            ctx.request_repaint();
            true
        } else {
            false
        }
    }

    pub fn paste_image(&mut self, ctx: &egui::Context) {
        let image = match arboard::Clipboard::new() {
            Ok(mut clipboard) => clipboard.get_image(),
            Err(error) => {
                self.status = tr!("clipboard_unavailable", error = error);
                return;
            }
        };
        match image {
            Ok(image) => {
                let size = (image.width as u32, image.height as u32);
                match image::RgbaImage::from_raw(size.0, size.1, image.bytes.into_owned()) {
                    Some(rgba) => self.set_image(ctx, rgba),
                    None => self.status = tr!("clipboard_invalid"),
                }
            }
            Err(arboard::Error::ContentNotAvailable) => {
                self.status = tr!("clipboard_no_image");
            }
            Err(error) => self.status = tr!("clipboard_read_failed", error = error),
        }
    }

    pub fn set_color_image(&mut self, ctx: &egui::Context, image: &egui::ColorImage) {
        let [width, height] = image.size;
        let mut rgba = Vec::with_capacity(width * height * 4);
        for pixel in &image.pixels {
            rgba.extend_from_slice(&pixel.to_srgba_unmultiplied());
        }
        match image::RgbaImage::from_raw(width as u32, height as u32, rgba) {
            Some(rgba) => self.set_image(ctx, rgba),
            None => self.status = tr!("clipboard_invalid"),
        }
    }

    /// Takes a hand change to the row or column count and schedules the board
    /// read for when the count settles. While the value is being dragged the
    /// undo entry is recorded once for the whole gesture, and quick clicks
    /// share the entry of the pending read, so undo returns to the board that
    /// was on screen before the burst.
    pub fn apply_grid_live(&mut self, ctx: &egui::Context, dragging: bool) {
        if self.grid_edit.is_none() {
            return;
        }
        // Counting rows or columns by hand is an explicit statement of intent,
        // so the scan is dropped here and deliberately not restarted — it would
        // only argue with the number the user just typed.
        self.cancel_grid_search();
        let already_open = self.grid_settle_at.is_some() || (dragging && self.grid_history_open);
        if self.result.is_some() && !already_open {
            self.push_history();
        }
        self.grid_history_open = dragging;
        self.grid_settle_at = Some(ctx.input(|input| input.time));
        ctx.request_repaint_after(std::time::Duration::from_secs_f64(GRID_SETTLE_DELAY));
    }

    /// Reads the hand-edited board once the count has been still for
    /// [`GRID_SETTLE_DELAY`], so a run of stepper changes costs one read
    /// instead of one per frame.
    fn settle_grid(&mut self, ctx: &egui::Context) {
        let Some(changed_at) = self.grid_settle_at else {
            return;
        };
        let elapsed = ctx.input(|input| input.time) - changed_at;
        if elapsed < GRID_SETTLE_DELAY {
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(
                GRID_SETTLE_DELAY - elapsed,
            ));
            return;
        }
        self.grid_settle_at = None;
        let Some(grid) = self.grid_edit else {
            return;
        };
        self.recognize_with(Some(grid), None, History::Keep);
    }

    /// Reads a hand-edited grid right away, for work that acts on the cells
    /// before the next settle frame.
    pub(crate) fn settle_grid_now(&mut self) {
        if self.grid_settle_at.take().is_some()
            && let Some(grid) = self.grid_edit
        {
            self.recognize_with(Some(grid), None, History::Keep);
        }
    }

    /// Drops a waiting read because the caller is about to replace the board,
    /// which the settle must not overwrite with the stale grid.
    pub(crate) fn cancel_grid_settle(&mut self) {
        self.grid_settle_at = None;
    }

    /// Whether the outline on screen no longer matches the last read: a hand
    /// change is waiting to be read, or a handle is under the pointer.
    pub(crate) fn grid_is_stale(&self) -> bool {
        self.grid_settle_at.is_some() || self.grid_drag.is_some()
    }

    /// Ends a stepper or paint drag, so the next change records its own undo
    /// step again.
    pub fn release_grid_history(&mut self) {
        let was_open = self.grid_history_open;
        self.grid_history_open = false;
        // A board read in the middle of a drag was not learned then, and with
        // the read settled there is no further one coming, so learn it here. A
        // waiting read stores the settled board itself.
        if was_open && self.grid_settle_at.is_none() && self.auto_learn {
            self.learn_current(true);
        }
        self.finish_paint_stroke();
    }

    /// Closes an active drag-paint stroke and learns the resulting board if
    /// automatic learning is enabled.
    pub(crate) fn finish_paint_stroke(&mut self) {
        let was_painting = self.paint_history_open;
        self.paint_history_open = false;
        self.last_drag_painted = None;
        if was_painting && self.auto_learn {
            self.learn_current(true);
        }
    }

    /// Scores the resolutions next to the detected one on a worker thread, so a
    /// near-miss on the cell pitch can be corrected without blocking the frame.
    pub fn start_grid_search(&mut self, ctx: &egui::Context) {
        let (Some(result), Some(image)) = (self.result.as_ref(), self.image.as_ref()) else {
            return;
        };
        let base = result.grid;
        let base_confidence = result.quality();
        let skins = self.skins.clone();
        let rgba = Arc::clone(&image.rgba);
        self.search_token += 1;
        let token = self.search_token;
        let (sender, receiver) = channel();
        self.grid_search = Some(GridSearch { receiver });

        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let best = refine_grid(
                &rgba,
                base,
                base_confidence,
                (!skins.skins.is_empty()).then_some(&skins),
            );
            // The receiver is gone when the user edited the grid meanwhile,
            // which is exactly the case this send is allowed to fail in.
            let _ = sender.send((token, best));
            ctx.request_repaint();
        });
    }

    pub fn cancel_grid_search(&mut self) {
        self.search_token += 1;
        self.grid_search = None;
    }

    fn poll_grid_search(&mut self, ctx: &egui::Context) {
        let Some(search) = self.grid_search.as_ref() else {
            return;
        };
        match search.receiver.try_recv() {
            Ok((token, best)) => {
                self.grid_search = None;
                if token != self.search_token {
                    return;
                }
                let Some(best) = best else {
                    return;
                };
                let rows = best.grid.rows;
                let columns = best.grid.columns;
                let confidence = best.quality();
                self.drawing_grid = false;
                self.cancel_grid_settle();
                if self.result.is_some() {
                    self.push_history();
                }
                self.grid_edit = Some(best.grid);
                self.result = Some(best);
                self.invalidate_derived();
                if !self.grid_history_open && self.auto_learn {
                    self.learn_current(true);
                }
                self.status = tr!(
                    "better_grid",
                    rows = rows,
                    columns = columns,
                    confidence = format!("{:.1}", confidence * 100.0)
                );
                ctx.request_repaint();
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => self.grid_search = None,
        }
    }

    /// Image position under the pointer, if it is inside `grid`.
    pub fn cell_in_grid(
        grid: Grid,
        pointer: egui::Pos2,
        image_rect: egui::Rect,
        scale: f32,
    ) -> Option<(usize, usize)> {
        if !image_rect.contains(pointer) {
            return None;
        }
        let position = (pointer - image_rect.min) / scale;
        let column = ((position.x - grid.origin_x) / grid.cell_width).floor();
        let row = ((position.y - grid.origin_y) / grid.cell_height).floor();
        if row < 0.0 || column < 0.0 || row >= grid.rows as f32 || column >= grid.columns as f32 {
            return None;
        }
        Some((row as usize, column as usize))
    }

    /// Image position under the pointer, if it is inside the recognized grid.
    pub fn cell_at(
        &self,
        pointer: egui::Pos2,
        image_rect: egui::Rect,
        scale: f32,
    ) -> Option<(usize, usize)> {
        let result = self.result.as_ref()?;
        Self::cell_in_grid(result.grid, pointer, image_rect, scale)
    }

    /// Paints a cell from a canvas click or drag, recording at most one undo
    /// step per stroke and skipping repeated frames on the same cell.
    pub(crate) fn paint_from_canvas(&mut self, row: usize, column: usize, dragging: bool) {
        if dragging && self.last_drag_painted == Some((row, column)) {
            return;
        }
        if dragging {
            self.last_drag_painted = Some((row, column));
        }
        let changes = self.would_change_on_paint(row, column);
        if changes && !self.paint_history_open {
            self.push_history();
            if dragging {
                self.paint_history_open = true;
            }
        }
        if !changes && dragging {
            self.selected = Some((row, column));
            return;
        }
        self.paint_cell(row, column);
    }

    fn would_change_on_paint(&self, row: usize, column: usize) -> bool {
        let Some(result) = self.result.as_ref() else {
            return false;
        };
        let Some(selected) = result.cells.get(row).and_then(|cells| cells.get(column)) else {
            return false;
        };
        let target = self.paint_cell;
        match selected.appearance.filter(|_| self.batch_by_appearance) {
            Some(appearance) => result
                .cells
                .iter()
                .flatten()
                .filter(|cell| cell.appearance == Some(appearance))
                .any(|cell| cell.cell != target || cell.confidence < 1.0),
            None => selected.cell != target || selected.confidence < 1.0,
        }
    }

    pub fn paint_cell(&mut self, row: usize, column: usize) {
        let cell = self.paint_cell;
        let Some(result) = self.result.as_mut() else {
            return;
        };
        let painted = result.paint(row, column, cell, self.batch_by_appearance);
        self.invalidate_derived();
        self.selected = Some((row, column));
        let label = cell_label(cell);
        self.status = if painted > 1 {
            tr!("filled_group", count = painted, label = label)
        } else {
            tr!("changed_cell", row = row, column = column, label = label)
        };
        if !self.paint_history_open && self.auto_learn {
            self.learn_current(true);
        }
    }

    /// Applies an alternative candidate picked in the inspector to `(row, column)`,
    /// recording undo history and adopting the chosen symbol in the palette.
    pub fn apply_cell_choice(&mut self, row: usize, column: usize, choice: Cell) {
        self.settle_grid_now();
        self.paint_cell = choice;
        if self.would_change_on_paint(row, column) {
            self.push_history();
        }
        self.paint_cell(row, column);
    }

    /// Shifts `grid_edit` by `(dx, dy)` image pixels within the image bounds and
    /// schedules a debounced board read.
    pub fn nudge_grid(&mut self, ctx: &egui::Context, dx: f32, dy: f32) -> bool {
        let (Some(mut grid), Some(image)) = (self.grid_edit, self.image.as_ref()) else {
            return false;
        };
        let max_x = (image.rgba.width() as f32 - grid.width()).max(0.0);
        let max_y = (image.rgba.height() as f32 - grid.height()).max(0.0);
        let origin_x = (grid.origin_x + dx).clamp(0.0, max_x);
        let origin_y = (grid.origin_y + dy).clamp(0.0, max_y);
        if (origin_x - grid.origin_x).abs() < f32::EPSILON
            && (origin_y - grid.origin_y).abs() < f32::EPSILON
        {
            return false;
        }
        grid.origin_x = origin_x;
        grid.origin_y = origin_y;
        self.grid_edit = Some(grid);
        self.apply_grid_live(ctx, false);
        true
    }

    /// Handles global and canvas keyboard shortcuts (`Ctrl+Z`, `Ctrl+Y` /
    /// `Ctrl+Shift+Z`, `Esc`, `1`–`7` / XSB characters, and arrow keys).
    pub fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if self.capture_crop.is_some() {
            if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
                self.cancel_capture_crop(ctx);
            }
            return;
        }
        let ctrl_shift = egui::Modifiers::CTRL | egui::Modifiers::SHIFT;
        if ctx.input_mut(|input| {
            input.consume_key(ctrl_shift, egui::Key::Z)
                || input.consume_key(egui::Modifiers::CTRL, egui::Key::Y)
        }) {
            self.redo(ctx);
            return;
        }
        if ctx.input_mut(|input| input.consume_key(egui::Modifiers::CTRL, egui::Key::Z)) {
            self.undo(ctx);
            return;
        }
        if self.drawing_grid
            && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
        {
            self.cancel_hand_drawn_grid(ctx);
            return;
        }
        if ctx.egui_wants_keyboard_input() {
            return;
        }

        const NUM_KEYS: [egui::Key; 7] = [
            egui::Key::Num1,
            egui::Key::Num2,
            egui::Key::Num3,
            egui::Key::Num4,
            egui::Key::Num5,
            egui::Key::Num6,
            egui::Key::Num7,
        ];
        for (index, key) in NUM_KEYS.into_iter().enumerate() {
            if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, key)) {
                self.paint_cell = crate::panel::CELL_CHOICES[index].0;
                return;
            }
        }

        let picked_symbol = ctx.input_mut(|input| {
            if input.modifiers.ctrl || input.modifiers.command || input.modifiers.alt {
                return None;
            }
            let mut found = None;
            input.events.retain(|event| {
                if found.is_none()
                    && let egui::Event::Text(text) = event
                    && text.chars().count() == 1
                    && let Some(ch) = text.chars().next()
                    && let Some(&(cell, _)) = crate::panel::CELL_CHOICES
                        .iter()
                        .find(|(_, symbol)| *symbol == ch)
                {
                    found = Some(cell);
                    return false;
                }
                true
            });
            found
        });
        if let Some(cell) = picked_symbol {
            self.paint_cell = cell;
            return;
        }

        if self.grid_edit.is_some() {
            for (mods, step) in [(egui::Modifiers::SHIFT, 5.0), (egui::Modifiers::NONE, 1.0)] {
                let mut dx = 0.0f32;
                let mut dy = 0.0f32;
                ctx.input_mut(|input| {
                    if input.consume_key(mods, egui::Key::ArrowLeft) {
                        dx -= step;
                    }
                    if input.consume_key(mods, egui::Key::ArrowRight) {
                        dx += step;
                    }
                    if input.consume_key(mods, egui::Key::ArrowUp) {
                        dy -= step;
                    }
                    if input.consume_key(mods, egui::Key::ArrowDown) {
                        dy += step;
                    }
                });
                if dx != 0.0 || dy != 0.0 {
                    self.nudge_grid(ctx, dx, dy);
                    break;
                }
            }
        }
    }

    /// Per-frame work that does not depend on the layout.
    pub fn tick(&mut self, ctx: &egui::Context) {
        self.settle_grid(ctx);
        self.poll_grid_search(ctx);
        self.flush_skins(ctx);
        let Some(started) = self.capture_started else {
            return;
        };
        // The window has to be hidden for at least one frame before the screen
        // is captured, otherwise the app itself ends up in the screenshot.
        const HIDE_DELAY: std::time::Duration = std::time::Duration::from_millis(350);
        if started.elapsed() >= HIDE_DELAY {
            self.capture_started = None;
            self.finish_screenshot(ctx);
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Finish any debounced or in-flight save with the final state.
        let pending = self
            .skin_saver
            .as_ref()
            .is_some_and(|saver| saver.has_pending_write());
        if self.skins_dirty || pending {
            self.write_skins_now();
        }
    }
}

fn capture_screen() -> Result<image::RgbaImage, String> {
    let screens =
        screenshots::Screen::all().map_err(|error| tr!("screen_info_failed", error = error))?;
    if screens.is_empty() {
        return Err(tr!("no_display"));
    }
    let mut captures = Vec::with_capacity(screens.len());
    for screen in screens {
        let info = screen.display_info;
        let raw = screen
            .capture()
            .map_err(|error| tr!("capture_failed", error = error))?;
        let rgba = image::RgbaImage::from_raw(raw.width(), raw.height(), raw.into_raw())
            .ok_or_else(|| tr!("invalid_capture"))?;
        captures.push((info.x, info.y, info.width, info.height, rgba));
    }
    stitch_monitors(captures)
}

/// Stitches one or more captured monitors `(x, y, logical_w, logical_h, rgba)`
/// into a single virtual-desktop image.
fn stitch_monitors(
    captures: Vec<(i32, i32, u32, u32, image::RgbaImage)>,
) -> Result<image::RgbaImage, String> {
    if captures.is_empty() {
        return Err(tr!("no_display"));
    }
    if captures.len() == 1 {
        let (_, _, _, _, rgba) = captures.into_iter().next().expect("len == 1");
        if rgba.width() == 0 || rgba.height() == 0 {
            return Err(tr!("empty_image"));
        }
        return Ok(rgba);
    }

    let mut placed = Vec::with_capacity(captures.len());
    let mut min_x = i64::MAX;
    let mut min_y = i64::MAX;
    let mut max_x = i64::MIN;
    let mut max_y = i64::MIN;
    for (x, y, logical_w, logical_h, rgba) in captures {
        if rgba.width() == 0 || rgba.height() == 0 {
            continue;
        }
        let scale_x = rgba.width() as f64 / logical_w.max(1) as f64;
        let scale_y = rgba.height() as f64 / logical_h.max(1) as f64;
        let px = (x as f64 * scale_x).round() as i64;
        let py = (y as f64 * scale_y).round() as i64;
        min_x = min_x.min(px);
        min_y = min_y.min(py);
        max_x = max_x.max(px + rgba.width() as i64);
        max_y = max_y.max(py + rgba.height() as i64);
        placed.push((px, py, rgba));
    }
    if placed.is_empty() {
        return Err(tr!("empty_image"));
    }
    let total_w = u32::try_from(max_x - min_x)
        .ok()
        .filter(|&w| (1..=32_768).contains(&w))
        .ok_or_else(|| tr!("invalid_capture"))?;
    let total_h = u32::try_from(max_y - min_y)
        .ok()
        .filter(|&h| (1..=32_768).contains(&h))
        .ok_or_else(|| tr!("invalid_capture"))?;

    let mut desktop = image::RgbaImage::from_pixel(total_w, total_h, image::Rgba([0, 0, 0, 255]));
    for (px, py, rgba) in placed {
        image::imageops::replace(&mut desktop, &rgba, px - min_x, py - min_y);
    }
    Ok(desktop)
}

/// Consumes paste events emitted by the platform integration and returns the
/// image attached to a clipboard paste, if any.
pub fn take_paste_events(ctx: &egui::Context) -> (Option<std::sync::Arc<egui::ColorImage>>, bool) {
    let mut image = None;
    let mut requested = false;
    ctx.input_mut(|input| {
        input.events.retain(|event| match event {
            egui::Event::PasteImage(source) => {
                image = Some(source.clone());
                false
            }
            egui::Event::Paste(_) => {
                requested = true;
                false
            }
            _ => true,
        });
    });
    (image, requested)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sokovision_core::Board;

    /// App holding one flat image: no edges, so detection finds no grid.
    fn app_with_flat_image(ctx: &egui::Context) -> App {
        // The assertions below read the Chinese texts.
        rust_i18n::set_locale("zh-CN");
        let rgba = image::RgbaImage::from_pixel(320, 240, image::Rgba([10, 10, 10, 255]));
        let color = egui::ColorImage::from_rgba_unmultiplied(
            [rgba.width() as usize, rgba.height() as usize],
            rgba.as_raw(),
        );
        let texture = ctx.load_texture("source", color, TextureOptions::LINEAR);
        let mut app = App::base(SkinLibrary::default(), None, false);
        app.image = Some(LoadedImage {
            rgba: Arc::new(rgba),
            texture,
        });
        app
    }

    #[test]
    fn a_failed_detection_shows_no_grid_without_starting_to_draw() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);

        assert!(!app.run_recognition(&ctx, None), "flat image has no grid");
        assert!(app.result.is_none());
        assert!(
            app.overlay_grid().is_none(),
            "a failed detection must not invent a grid"
        );
        assert!(!app.drawing_grid, "the canvas stays out of drawing mode");
        assert!(app.status.contains("手动框选"), "{}", app.status);
    }

    /// Dragging a rectangle has to produce a grid even where detection fails,
    /// otherwise the row and column steppers stay out of reach.
    #[test]
    fn a_dragged_rectangle_becomes_the_grid() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.run_recognition(&ctx, None);
        app.cancel_grid_search();

        app.create_grid_in(&ctx, PixelRect::new(20, 10, 240, 200));
        app.cancel_grid_search();

        let grid = app.overlay_grid().expect("the drag created a grid");
        assert_eq!((grid.origin_x, grid.origin_y), (20.0, 10.0));
        assert!((grid.width() - 240.0).abs() < 1.0, "{grid:?}");
        assert!((grid.height() - 200.0).abs() < 1.0, "{grid:?}");
        assert!(grid.rows >= 2 && grid.columns >= 2, "{grid:?}");
        assert!(app.result.is_some(), "cells are read with the new grid");
        assert!(app.status.contains("未检测到网格"), "{}", app.status);
    }

    /// A drag only builds a grid once the manual action armed it: importing an
    /// image that detection cannot read must not switch the canvas to drawing.
    #[test]
    fn a_drag_only_builds_a_grid_after_the_manual_action() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.run_recognition(&ctx, None);
        app.cancel_grid_search();

        let (start, end) = (egui::pos2(50.0, 25.0), egui::pos2(650.0, 575.0));
        play_drag(&ctx, &mut app, start, end);
        assert!(
            app.overlay_grid().is_none(),
            "a plain drag must not create a grid"
        );

        app.start_hand_drawn_grid();
        play_drag_at(&ctx, &mut app, 0.4, start, end);
        assert!(
            app.overlay_grid().is_some(),
            "the manual action arms the drag"
        );
        app.cancel_grid_search();
    }

    /// Draw Grid mode shows guides that span the whole canvas instead of only
    /// the tiny cursor cross, so an edge can be lined up against the board.
    #[test]
    fn drawing_a_grid_shows_full_canvas_guides() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.start_hand_drawn_grid();

        let pointer = egui::pos2(300.0, 200.0);
        // Hover is resolved against the previous frame's widget, so the
        // pointer has to rest on the canvas for one frame first.
        run_frame(
            &ctx,
            &mut app,
            input_at(0.0, vec![egui::Event::PointerMoved(pointer)]),
        );
        let mut output = ctx.run_ui(input_at(0.1, Vec::new()), |ui| app.draw_canvas(ui, &ctx));
        output.textures_delta.clear();

        let spans = |vertical: bool| {
            output
                .shapes
                .iter()
                .filter_map(|clipped| match &clipped.shape {
                    egui::Shape::LineSegment { points, .. } => Some(*points),
                    _ => None,
                })
                .any(|[a, b]| {
                    if vertical {
                        a.x == pointer.x && b.x == pointer.x && (a.y - b.y).abs() > 500.0
                    } else {
                        a.y == pointer.y && b.y == pointer.y && (a.x - b.x).abs() > 500.0
                    }
                })
        };

        assert!(spans(true), "a vertical guide runs through the pointer");
        assert!(spans(false), "a horizontal guide runs through the pointer");
    }

    /// The wheel zooms around the pointer, so the image pixel under the cursor
    /// stays put instead of the board sliding away as it grows.
    #[test]
    fn the_wheel_zooms_around_the_pointer() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);

        let pointer = egui::pos2(600.0, 450.0);
        // Hover is resolved against the previous frame, so rest the pointer on
        // the canvas for one frame first.
        run_frame(
            &ctx,
            &mut app,
            input_at(0.0, vec![egui::Event::PointerMoved(pointer)]),
        );

        let area = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let image_size = egui::vec2(320.0, 240.0);
        let image_point = |app: &App| {
            let fit = (area.width() / image_size.x).min(area.height() / image_size.y);
            let scale = fit * app.zoom;
            let image_rect =
                egui::Rect::from_center_size(area.center() + app.pan, image_size * scale);
            (pointer - image_rect.min) / scale
        };
        let before = image_point(&app);

        run_frame(&ctx, &mut app, input_at(0.1, vec![wheel(50.0)]));

        assert!(app.zoom > 1.0, "the wheel zooms in: {}", app.zoom);
        let after = image_point(&app);
        assert!(
            (before - after).length() < 0.01,
            "the pixel under the pointer stays put: {before:?} -> {after:?}"
        );
    }

    /// 「手动框选」 is drag-a-rectangle-to-build-a-grid, so it drops the grid
    /// first: the picture stays, the board goes on the undo stack, and the
    /// canvas falls back to the rubber band.
    #[test]
    fn hand_drawing_drops_the_grid_and_can_be_undone() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.create_grid_in(&ctx, PixelRect::new(20, 10, 240, 200));
        app.cancel_grid_search();
        assert!(app.overlay_grid().is_some());

        app.start_hand_drawn_grid();

        assert!(app.image.is_some(), "the picture is still there");
        assert!(app.overlay_grid().is_none(), "the grid is gone");
        assert!(app.result.is_none());
        assert!(app.status.contains("拖出一个矩形"), "{}", app.status);
        assert!(app.can_undo(), "dropping the grid is undoable");

        app.undo(&ctx);
        assert!(app.overlay_grid().is_some(), "undo brings the grid back");
    }

    /// Drawing the rectangle then replaces the dropped board with a new grid.
    #[test]
    fn a_hand_drawn_grid_replaces_the_existing_one() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.create_grid_in(&ctx, PixelRect::new(20, 10, 240, 200));
        app.cancel_grid_search();
        let before = app.overlay_grid().expect("grid");

        app.start_hand_drawn_grid();
        // The 320x240 image fills the 800x600 canvas at 2.5x, so this screen
        // rectangle is 40,32..200,192 in image pixels.
        play_drag(
            &ctx,
            &mut app,
            egui::pos2(100.0, 80.0),
            egui::pos2(500.0, 480.0),
        );
        app.cancel_grid_search();

        let after = app.overlay_grid().expect("the new grid");
        assert_ne!(
            (after.origin_x, after.origin_y),
            (before.origin_x, before.origin_y)
        );
        assert!((after.origin_x - 40.0).abs() < 1.0, "{after:?}");
        assert!((after.origin_y - 32.0).abs() < 1.0, "{after:?}");
        assert!((after.width() - 160.0).abs() < 2.0, "{after:?}");
    }

    /// A hand-drawn rectangle is the new focus: the view zooms and pans onto
    /// it, leaving a margin so the outline handles stay visible.
    #[test]
    fn a_hand_drawn_grid_is_fitted_into_the_view() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.start_hand_drawn_grid();
        // 100,80..500,480 on screen is 40,32..200,192 in a 320x240 image,
        // all inside the 800x600 canvas.
        play_drag(
            &ctx,
            &mut app,
            egui::pos2(100.0, 80.0),
            egui::pos2(500.0, 480.0),
        );
        app.cancel_grid_search();

        let grid = app.grid_edit.expect("the drag created a grid");
        let area = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let image_size = egui::vec2(320.0, 240.0);
        // The transform the canvas builds from the stored zoom and pan.
        let fit = (area.width() / image_size.x).min(area.height() / image_size.y);
        let scale = fit * app.zoom;
        let image_rect = egui::Rect::from_center_size(area.center() + app.pan, image_size * scale);
        let corner = |x: f32, y: f32| image_rect.min + egui::vec2(x, y) * scale;
        let grid_rect = egui::Rect::from_min_max(
            corner(grid.origin_x, grid.origin_y),
            corner(grid.origin_x + grid.width(), grid.origin_y + grid.height()),
        );

        assert!(app.zoom > 1.0, "the board is zoomed in: {}", app.zoom);
        assert!(
            (grid_rect.center() - area.center()).length() < 0.5,
            "the board is centred: {grid_rect:?}"
        );
        assert!(
            (grid_rect.height() - (area.height() - 48.0)).abs() < 0.5,
            "the board fills the shorter side up to a 24 px margin: {grid_rect:?}"
        );
        assert!(
            area.shrink(23.0).contains_rect(grid_rect),
            "the handles have room around the board: {grid_rect:?}"
        );
    }

    /// Nothing to drop means nothing on the undo stack either.
    #[test]
    fn hand_drawing_without_a_grid_does_not_stack_undo() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);

        app.start_hand_drawn_grid();

        assert!(app.overlay_grid().is_none());
        assert!(!app.can_undo(), "no board was dropped");
        assert!(app.status.contains("拖出一个矩形"), "{}", app.status);
    }

    /// Dragging a row or column stepper reports a new count every frame, and
    /// reading the whole board per frame is what made it stutter. The read
    /// waits for the count to settle, and the burst stays one undo step.
    #[test]
    fn a_grid_change_is_read_once_it_settles() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.create_grid_in(&ctx, PixelRect::new(20, 10, 240, 200));
        app.cancel_grid_search();
        let before = app.grid_edit.expect("the drag created a grid");

        let nudge = |app: &mut App, time: f64, rows: u32| {
            let mut output = ctx.run_ui(input_at(time, Vec::new()), |_| {});
            output.textures_delta.clear();
            app.grid_edit = Some(before.with_counts(rows, before.columns));
            app.apply_grid_live(&ctx, false);
        };
        nudge(&mut app, 0.0, before.rows + 1);
        nudge(&mut app, 0.05, before.rows + 2);

        assert_eq!(app.grid_edit.expect("grid").rows, before.rows + 2);
        assert_eq!(
            app.result.as_ref().expect("board").grid.rows,
            before.rows,
            "no read while the count is still changing"
        );
        assert_eq!(app.history.len(), 1, "the burst is one undo step");

        // The first frame after the quiet period reads the settled count.
        let mut output = ctx.run_ui(
            input_at(0.05 + GRID_SETTLE_DELAY + 0.01, Vec::new()),
            |_| {},
        );
        output.textures_delta.clear();
        app.tick(&ctx);

        assert_eq!(
            app.result.as_ref().expect("board").grid.rows,
            before.rows + 2,
            "the settled grid is read"
        );
        assert!(app.grid_settle_at.is_none(), "nothing is left waiting");

        app.undo(&ctx);
        assert_eq!(
            app.grid_edit.expect("grid").rows,
            before.rows,
            "one undo returns to the board before the burst"
        );
    }

    /// The read waits, but the grid does not: a row or column added by hand is
    /// drawn with its own lines before the board has cells for it. The stale
    /// cell indicators stay out of the way while the outline is adjusted.
    #[test]
    fn a_pending_grid_change_draws_only_the_grid() {
        let ctx = egui::Context::default();
        let mut app = app_with_learnable_board(&ctx);
        let read = app.result.as_ref().expect("board").grid;
        ctx.run_ui(input_at(0.0, Vec::new()), |_| {})
            .textures_delta
            .clear();
        app.grid_edit = Some(read.with_counts(read.rows + 1, read.columns));
        app.apply_grid_live(&ctx, false);
        // A shaky cell would normally carry a tint; it belongs to the old
        // outline now and must not sit on the new lines.
        app.result.as_mut().expect("board").cells[0][0].confidence = 0.2;

        let mut output = ctx.run_ui(input_at(0.0, Vec::new()), |ui| app.draw_canvas(ui, &ctx));
        output.textures_delta.clear();

        // The added row is the one the stale read has no cells for. The 120x72
        // image sits at (0, 60) in the 800x600 canvas at a 20/3 scale, so the
        // row spans y 420..540 and the first cell x 0..160.
        let cell = egui::Rect::from_min_max(egui::pos2(0.0, 420.0), egui::pos2(160.0, 540.0));
        let bordered = output.shapes.iter().any(|clipped| match &clipped.shape {
            egui::Shape::Rect(rect) => {
                rect.stroke.width > 0.0
                    && (rect.rect.min - cell.min).length() < 1.0
                    && (rect.rect.max - cell.max).length() < 1.0
            }
            _ => false,
        });
        assert!(bordered, "the added row still shows its cell border");

        let first = egui::Rect::from_min_max(egui::pos2(0.0, 60.0), egui::pos2(160.0, 180.0));
        let tinted = output.shapes.iter().any(|clipped| match &clipped.shape {
            egui::Shape::Rect(rect) => {
                rect.fill != egui::Color32::TRANSPARENT
                    && (rect.rect.min - first.min).length() < 1.0
                    && (rect.rect.max - first.max).length() < 1.0
            }
            _ => false,
        });
        assert!(!tinted, "the stale tint is hidden while the grid settles");
    }

    /// Dragging the middle of the left edge must resize that edge only, on real
    /// frames — the wiring between `handle_at` and `resize` is what breaks.
    #[test]
    fn dragging_an_edge_resizes_only_that_side() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.create_grid_in(&ctx, PixelRect::new(20, 10, 240, 200));
        app.cancel_grid_search();
        let before = app.grid_edit.expect("the drag created a grid");

        // 320x240 image in an 800x600 window: scale 2.5, image at the origin.
        // The left edge is at x = 20 * 2.5 = 50, the middle of it at y = 275.
        play_drag(
            &ctx,
            &mut app,
            egui::pos2(50.0, 275.0),
            egui::pos2(30.0, 275.0),
        );

        let after = app.grid_edit.expect("the grid survives the drag");
        assert!((after.origin_x - 12.0).abs() < 0.01, "{after:?}");
        assert_eq!(
            (after.origin_y, after.rows, after.columns),
            (before.origin_y, before.rows, before.columns)
        );
        assert!((after.cell_height - before.cell_height).abs() < 1e-3);
        assert!(
            after.cell_width > before.cell_width,
            "the left edge moved outwards, so the cells got wider: {before:?} -> {after:?}"
        );
    }

    /// A board drawn as bright grid lines on a dark background, which
    /// detection finds on its own, so the grid under test has the approximate
    /// origin and pitch a real screenshot produces.
    fn app_with_detected_board(ctx: &egui::Context) -> App {
        let (origin, cell, rows, columns) = (20u32, 24u32, 5u32, 6u32);
        let mut rgba = image::RgbaImage::from_pixel(320, 240, image::Rgba([20, 20, 20, 255]));
        let line = image::Rgba([220, 220, 220, 255]);
        for row in 0..=rows {
            let y = origin + row * cell;
            for x in origin..=origin + columns * cell {
                rgba.put_pixel(x, y, line);
            }
        }
        for column in 0..=columns {
            let x = origin + column * cell;
            for y in origin..=origin + rows * cell {
                rgba.put_pixel(x, y, line);
            }
        }
        let color = egui::ColorImage::from_rgba_unmultiplied(
            [rgba.width() as usize, rgba.height() as usize],
            rgba.as_raw(),
        );
        let texture = ctx.load_texture("source", color, TextureOptions::LINEAR);
        let mut app = App::base(SkinLibrary::default(), None, false);
        app.image = Some(LoadedImage {
            rgba: Arc::new(rgba),
            texture,
        });
        app
    }

    /// Releasing a resized handle is the user's own decision about the grid:
    /// the grid must stay exactly where it was dragged instead of being
    /// re-detected into a different resolution.
    #[test]
    fn releasing_a_resized_handle_keeps_the_dragged_grid() {
        let ctx = egui::Context::default();
        let mut app = app_with_detected_board(&ctx);
        assert!(
            app.run_recognition(&ctx, None),
            "the drawn board is detected"
        );
        app.cancel_grid_search();
        let before = app.grid_edit.expect("detected grid");
        assert_eq!((before.rows, before.columns), (5, 6), "{before:?}");

        // Cut the board short on the right: press on the edge, release at
        // image x = 150, which is screen x = 375 at the canvas' 2.5x.
        let right = (before.origin_x + before.width()) * 2.5;
        let middle = (before.origin_y + before.height() * 0.5) * 2.5;
        play_drag(
            &ctx,
            &mut app,
            egui::pos2(right, middle),
            egui::pos2(375.0, middle),
        );

        let after = app.grid_edit.expect("the grid survives the drag");
        assert_eq!(
            (after.rows, after.columns),
            (before.rows, before.columns),
            "the dragged resolution must survive re-detection"
        );
        assert!(
            (after.origin_x - before.origin_x).abs() < 0.01
                && (after.origin_y - before.origin_y).abs() < 0.01,
            "the untouched edges must not move: {before:?} -> {after:?}"
        );
        assert!(
            (after.cell_height - before.cell_height).abs() < 0.01,
            "the untouched axis keeps its cell size: {before:?} -> {after:?}"
        );
        assert!(
            (after.origin_x + after.width() - 150.0).abs() < 0.01,
            "the right edge stays where the pointer was released: {after:?}"
        );
        assert!(
            !app.is_searching(),
            "the neighbour scan must not fight a hand adjustment"
        );
    }

    /// The neighbour scan finishing between the grab of a handle and the
    /// first drag frame must not bounce the grid: holding the outline is an
    /// explicit statement of intent, so the grab drops the scan.
    #[test]
    fn grabbing_a_handle_drops_a_landed_scan_result() {
        let ctx = egui::Context::default();
        let mut app = app_with_detected_board(&ctx);
        assert!(
            app.run_recognition(&ctx, None),
            "the drawn board is detected"
        );
        let before = app.grid_edit.expect("detected grid");

        // A finished scan waiting in the channel with the current token: the
        // next tick would adopt it.
        let (sender, receiver) = channel();
        app.search_token += 1;
        sender
            .send((
                app.search_token,
                Some(RecognitionResult {
                    grid: before.with_counts(before.rows + 1, before.columns),
                    cells: Vec::new(),
                    warnings: Vec::new(),
                    skin: None,
                    relaxed: false,
                }),
            ))
            .expect("the receiver is alive");
        app.grid_search = Some(GridSearch { receiver });

        // Press the right edge and hold: the drag has not started yet, so the
        // tick runs while the outline is only grabbed.
        let edge = egui::pos2(
            (before.origin_x + before.width()) * 2.5,
            (before.origin_y + before.height() * 0.5) * 2.5,
        );
        run_frame(
            &ctx,
            &mut app,
            input_at(0.0, vec![egui::Event::PointerMoved(edge)]),
        );
        run_frame(
            &ctx,
            &mut app,
            input_at(0.1, vec![primary_button(edge, true)]),
        );
        app.tick(&ctx);

        assert_eq!(
            app.grid_edit,
            Some(before),
            "the grabbed outline must not bounce to the scanned grid"
        );
        assert!(!app.is_searching(), "the grab drops the scan");
    }

    /// Grabbing a handle a little off its line must not move the outline onto
    /// the pointer: the edge follows the drag delta only.
    #[test]
    fn dragging_a_handle_keeps_the_grab_offset() {
        let ctx = egui::Context::default();
        let mut app = app_with_detected_board(&ctx);
        assert!(
            app.run_recognition(&ctx, None),
            "the drawn board is detected"
        );
        app.cancel_grid_search();
        let before = app.grid_edit.expect("detected grid");
        let right = (before.origin_x + before.width()) * 2.5;
        let middle = (before.origin_y + before.height() * 0.5) * 2.5;

        // Press 5 screen px inside the right edge, then drag 10 px further left.
        play_drag(
            &ctx,
            &mut app,
            egui::pos2(right - 5.0, middle),
            egui::pos2(right - 15.0, middle),
        );

        let after = app.grid_edit.expect("the grid survives the drag");
        let before_edge = before.origin_x + before.width();
        let after_edge = after.origin_x + after.width();
        assert!(
            (before_edge - after_edge - 4.0).abs() < 0.6,
            "the edge moves by the 4 image px the pointer travelled, not onto it: \
             {before:?} -> {after:?}"
        );
    }

    /// A horizontal drag on a corner keeps the other axis where it was, even
    /// though the press landed a few pixels inside the corner.
    #[test]
    fn a_horizontal_corner_drag_keeps_the_other_axis() {
        let ctx = egui::Context::default();
        let mut app = app_with_detected_board(&ctx);
        assert!(
            app.run_recognition(&ctx, None),
            "the drawn board is detected"
        );
        app.cancel_grid_search();
        let before = app.grid_edit.expect("detected grid");
        let corner = egui::pos2(
            (before.origin_x + before.width()) * 2.5,
            before.origin_y * 2.5,
        );

        // Grab the top right corner 4 px inside both axes and drag 12 px right.
        play_drag(
            &ctx,
            &mut app,
            corner + egui::vec2(-4.0, 4.0),
            corner + egui::vec2(8.0, 4.0),
        );

        let after = app.grid_edit.expect("the grid survives the drag");
        assert!(
            (after.origin_y - before.origin_y).abs() < 0.01,
            "a horizontal drag must not move the top edge: {before:?} -> {after:?}"
        );
        let before_edge = before.origin_x + before.width();
        let after_edge = after.origin_x + after.width();
        assert!(
            (after_edge - before_edge - 4.8).abs() < 0.6,
            "the corner moves by the 4.8 image px the pointer travelled: \
             {before:?} -> {after:?}"
        );
    }

    /// The whole outline is draggable, so a plain click on a border cell has to
    /// keep painting it instead of resizing the grid.
    #[test]
    fn clicking_a_border_cell_paints_instead_of_resizing() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.create_grid_in(&ctx, PixelRect::new(20, 10, 240, 200));
        app.cancel_grid_search();
        let before = app.grid_edit.expect("the drag created a grid");

        // Middle of the top edge, which sits at y = 10 * 2.5 = 25.
        play_click(&ctx, &mut app, egui::pos2(400.0, 25.0));

        assert_eq!(app.grid_edit, Some(before), "a click must not resize");
        let result = app.result.as_ref().expect("the board is recognized");
        assert_eq!(result.cells[0][8].cell, Cell::Wall, "the cell was painted");
    }

    /// Hover, press, drag and release, one frame each: what a real drag sends.
    fn play_drag(ctx: &egui::Context, app: &mut App, start: egui::Pos2, end: egui::Pos2) {
        play_drag_at(ctx, app, 0.0, start, end);
    }

    /// [`play_drag`] for a context whose clock already ran past `time`.
    fn play_drag_at(
        ctx: &egui::Context,
        app: &mut App,
        time: f64,
        start: egui::Pos2,
        end: egui::Pos2,
    ) {
        let frames = [
            vec![egui::Event::PointerMoved(start)],
            vec![primary_button(start, true)],
            vec![egui::Event::PointerMoved(end)],
            vec![primary_button(end, false)],
        ];
        for (index, events) in frames.into_iter().enumerate() {
            run_frame(ctx, app, input_at(time + index as f64 * 0.1, events));
        }
    }

    /// Press and release on the same spot, which egui reports as a click.
    fn play_click(ctx: &egui::Context, app: &mut App, pos: egui::Pos2) {
        let frames = [
            vec![egui::Event::PointerMoved(pos)],
            vec![primary_button(pos, true)],
            vec![primary_button(pos, false)],
        ];
        for (index, events) in frames.into_iter().enumerate() {
            run_frame(ctx, app, input_at(index as f64 * 0.1, events));
        }
    }

    /// One mouse wheel notch, the event a backend reports for a scroll.
    fn wheel(delta: f32) -> egui::Event {
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, delta),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Unknown,
            modifiers: egui::Modifiers::default(),
        }
    }

    fn primary_button(pos: egui::Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        }
    }

    /// Input for one frame of an 800x600 window.
    fn input_at(time: f64, events: Vec<egui::Event>) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            time: Some(time),
            events,
            ..Default::default()
        }
    }

    /// One headless frame of the canvas, so pointer input can be replayed.
    fn run_frame(ctx: &egui::Context, app: &mut App, input: egui::RawInput) {
        let mut output = ctx.run_ui(input, |ui| app.draw_canvas(ui, ctx));
        // No renderer consumes the deltas here; dropping them is the point.
        output.textures_delta.clear();
    }

    #[test]
    fn consumes_paste_events() {
        use std::sync::Arc;

        let ctx = egui::Context::default();
        let image = Arc::new(egui::ColorImage::filled([2, 3], egui::Color32::WHITE));
        ctx.input_mut(|input| {
            input.events.push(egui::Event::PasteImage(image.clone()));
            input.events.push(egui::Event::Paste("text".to_owned()));
        });

        let (pasted, requested) = take_paste_events(&ctx);

        assert!(requested);
        assert_eq!(pasted.expect("image event").size, [2, 3]);
        assert!(ctx.input(|input| input.events.is_empty()));
    }

    /// App holding a small, complete board, with one solid colour per tile so
    /// the remembered prototypes stay distinguishable.
    fn app_with_learnable_board(ctx: &egui::Context) -> App {
        use sokovision_core::RecognizedCell;

        let mut app = app_with_flat_image(ctx);
        let board = Board::from_xsb("#####\n#@$.#\n#####").unwrap();
        // One solid colour per tile, so the remembered prototypes stay
        // distinguishable instead of all matching each other.
        let mut rgba = image::RgbaImage::from_pixel(120, 72, image::Rgba([0, 0, 0, 255]));
        let color = |cell: Cell| match cell {
            Cell::Wall => image::Rgba([30, 30, 30, 255]),
            Cell::Player => image::Rgba([0, 150, 0, 255]),
            Cell::Box => image::Rgba([120, 75, 35, 255]),
            Cell::Goal => image::Rgba([200, 20, 30, 255]),
            _ => image::Rgba([230, 230, 210, 255]),
        };
        for (row, cells) in board.cells.iter().enumerate() {
            for (column, cell) in cells.iter().enumerate() {
                for y in row as u32 * 24..(row as u32 + 1) * 24 {
                    for x in column as u32 * 24..(column as u32 + 1) * 24 {
                        rgba.put_pixel(x, y, color(*cell));
                    }
                }
            }
        }
        app.image.as_mut().expect("image").rgba = Arc::new(rgba);
        let cells = board
            .cells
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&cell| RecognizedCell {
                        cell,
                        confidence: 1.0,
                        alternatives: Vec::new(),
                        appearance: Some(match cell {
                            Cell::Wall => 0,
                            Cell::Player => 1,
                            Cell::Box => 2,
                            Cell::Goal => 3,
                            _ => 4,
                        }),
                    })
                    .collect()
            })
            .collect();
        app.result = Some(RecognitionResult {
            grid: Grid::new(0.0, 0.0, 24.0, 24.0, 3, 5),
            cells,
            warnings: Vec::new(),
            skin: None,
            relaxed: false,
        });
        app
    }

    /// A complete, valid board teaches the skin library; the same board teaches
    /// it only once, and clearing drops it again.
    #[test]
    fn a_valid_board_is_learned_once() {
        let ctx = egui::Context::default();
        let mut app = app_with_learnable_board(&ctx);

        assert!(app.learn_current(false), "a valid board is remembered");
        assert_eq!(app.remembered_skins(), 1);
        assert!(!app.learn_current(false), "learning again merges");
        assert_eq!(app.remembered_skins(), 1);

        app.forget_skins();
        assert_eq!(app.remembered_skins(), 0);
    }

    /// Reading the same picture again after learning must reproduce the board
    /// that was learned, not fall back to the plain visual guess.
    #[test]
    fn a_learned_board_is_read_back_unchanged() {
        let ctx = egui::Context::default();
        let mut app = app_with_learnable_board(&ctx);
        let grid = app.result.as_ref().expect("board").grid;
        let before = app.result.as_ref().expect("board").board();
        assert!(app.learn_current(false), "the board is remembered");

        assert!(app.run_recognition(&ctx, Some(grid)));
        assert_eq!(
            app.result.as_ref().expect("board").board(),
            before,
            "the remembered labels are used again"
        );
    }

    /// With automatic learning off, only the Learning section's button stores
    /// a board, so adjusting the grid cannot file half-finished skins.
    #[test]
    fn automatic_learning_can_be_switched_off() {
        let ctx = egui::Context::default();
        let mut app = app_with_learnable_board(&ctx);
        assert!(app.auto_learn, "automatic learning is the default");

        app.paint_cell(0, 0);
        assert_eq!(
            app.remembered_skins(),
            1,
            "automatic learning stores the board"
        );

        app.forget_skins();
        app.auto_learn = false;
        app.paint_cell(0, 1);
        assert_eq!(
            app.remembered_skins(),
            0,
            "nothing is stored while automatic learning is off"
        );

        app.learn_manually();
        assert_eq!(
            app.remembered_skins(),
            1,
            "the button stores the board on demand"
        );
    }

    /// An unfinished board has too many guesses to be worth remembering.
    #[test]
    fn an_unresolved_board_is_not_learned() {
        use sokovision_core::RecognizedCell;

        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.result = Some(RecognitionResult {
            grid: Grid::new(0.0, 0.0, 24.0, 24.0, 3, 5),
            cells: vec![
                vec![
                    RecognizedCell {
                        cell: Cell::Unknown,
                        confidence: 0.0,
                        alternatives: Vec::new(),
                        appearance: Some(0),
                    };
                    5
                ];
                3
            ],
            warnings: Vec::new(),
            skin: None,
            relaxed: false,
        });

        app.learn_current(false);
        assert_eq!(app.remembered_skins(), 0);
    }

    /// A relaxed read satisfied the rules only by dropping them, so it is a
    /// guess: automatic learning refuses it, but the manual button trusts the
    /// user and still stores the structurally valid board.
    #[test]
    fn a_relaxed_read_is_not_auto_learned() {
        let ctx = egui::Context::default();
        let mut app = app_with_learnable_board(&ctx);
        app.result.as_mut().expect("board").relaxed = true;

        assert!(!app.learn_current(true), "a relaxed guess is not filed");
        assert_eq!(app.remembered_skins(), 0);

        assert!(
            app.learn_current(false),
            "the manual button still learns it"
        );
        assert_eq!(app.remembered_skins(), 1);
    }

    #[test]
    fn clicking_outside_the_grid_or_re_painting_same_cell_does_not_stack_undo() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.create_grid_in(&ctx, PixelRect::new(40, 40, 160, 120));
        app.cancel_grid_search();
        app.history.clear();

        // (5, 5) on screen is (2, 2) in the 320x240 image, outside the grid.
        play_click(&ctx, &mut app, egui::pos2(5.0, 5.0));
        assert!(
            !app.can_undo(),
            "clicking outside the grid leaves history untouched"
        );

        // Paint (0, 0) once, then click it again with the same brush.
        app.batch_by_appearance = false;
        play_click_at(&ctx, &mut app, 0.5, egui::pos2(120.0, 120.0));
        assert_eq!(app.history.len(), 1);
        play_click_at(&ctx, &mut app, 1.0, egui::pos2(120.0, 120.0));
        assert_eq!(
            app.history.len(),
            1,
            "painting an already confirmed identical cell does not duplicate undo"
        );
    }

    #[test]
    fn painting_while_a_grid_change_is_pending_settles_first() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.create_grid_in(&ctx, PixelRect::new(20, 10, 240, 200));
        app.cancel_grid_search();
        app.history.clear();
        app.batch_by_appearance = false;
        let before = app.grid_edit.expect("grid");

        ctx.run_ui(input_at(0.0, Vec::new()), |_| {})
            .textures_delta
            .clear();
        app.grid_edit = Some(before.with_counts(before.rows + 1, before.columns));
        app.apply_grid_live(&ctx, false);
        assert!(app.grid_settle_at.is_some());

        // Paint inside the grid before the 0.5s settle delay has elapsed.
        play_click_at(&ctx, &mut app, 0.1, egui::pos2(100.0, 80.0));
        assert!(app.grid_settle_at.is_none(), "click settled the grid");
        assert_eq!(app.result.as_ref().unwrap().grid.rows, before.rows + 1);
        assert_eq!(
            app.history.len(),
            2,
            "stepper edit and paint each have their own undo step"
        );

        app.undo(&ctx);
        assert_eq!(
            app.result.as_ref().unwrap().grid.rows,
            before.rows + 1,
            "first undo reverts only the painted cell, keeping the new row count"
        );
        app.undo(&ctx);
        assert_eq!(
            app.result.as_ref().unwrap().grid.rows,
            before.rows,
            "second undo reverts the row count"
        );
    }

    #[test]
    fn dragging_across_cells_records_one_undo_step() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.create_grid_in(&ctx, PixelRect::new(20, 10, 240, 200));
        app.cancel_grid_search();
        app.history.clear();
        app.batch_by_appearance = false;

        let frames = [
            vec![egui::Event::PointerMoved(egui::pos2(80.0, 80.0))],
            vec![primary_button(egui::pos2(80.0, 80.0), true)],
            vec![egui::Event::PointerMoved(egui::pos2(150.0, 80.0))],
            vec![egui::Event::PointerMoved(egui::pos2(220.0, 80.0))],
            vec![primary_button(egui::pos2(220.0, 80.0), false)],
        ];
        for (index, events) in frames.into_iter().enumerate() {
            run_frame(&ctx, &mut app, input_at(index as f64 * 0.05, events));
        }

        let painted = app
            .result
            .as_ref()
            .unwrap()
            .cells
            .iter()
            .flatten()
            .filter(|cell| cell.cell == Cell::Wall)
            .count();
        assert!(painted >= 2, "drag painted multiple cells: {painted}");
        assert_eq!(app.history.len(), 1, "whole stroke is one undo step");
    }

    #[test]
    fn landed_grid_search_adopts_result_directly_and_records_undo() {
        let ctx = egui::Context::default();
        let mut app = app_with_detected_board(&ctx);
        assert!(app.run_recognition(&ctx, None));
        app.cancel_grid_search();
        app.history.clear();
        let before = app.grid_edit.expect("detected grid");
        let improved_grid = before.with_counts(before.rows + 1, before.columns);

        let (sender, receiver) = channel();
        app.search_token += 1;
        sender
            .send((
                app.search_token,
                Some(RecognitionResult {
                    grid: improved_grid,
                    cells: Vec::new(),
                    warnings: Vec::new(),
                    skin: None,
                    relaxed: false,
                }),
            ))
            .expect("alive");
        app.grid_search = Some(GridSearch { receiver });

        app.tick(&ctx);
        assert_eq!(app.grid_edit, Some(improved_grid));
        assert_eq!(app.history.len(), 1, "previous board is saved for undo");
        app.undo(&ctx);
        assert_eq!(app.grid_edit, Some(before));
    }

    #[test]
    fn undo_and_redo_round_trip_and_clear_on_new_edit() {
        let ctx = egui::Context::default();
        let mut app = app_with_learnable_board(&ctx);
        app.batch_by_appearance = false;

        app.apply_cell_choice(1, 1, Cell::Goal);
        assert_eq!(app.result.as_ref().unwrap().cells[1][1].cell, Cell::Goal);
        assert!(app.can_undo());
        assert!(!app.can_redo());

        app.undo(&ctx);
        assert_eq!(app.result.as_ref().unwrap().cells[1][1].cell, Cell::Player);
        assert!(app.can_redo());

        app.redo(&ctx);
        assert_eq!(app.result.as_ref().unwrap().cells[1][1].cell, Cell::Goal);
        assert!(!app.can_redo());

        app.undo(&ctx);
        assert!(app.can_redo());
        app.apply_cell_choice(1, 1, Cell::Box);
        assert!(
            !app.can_redo(),
            "a new forward edit invalidates the stale redo branch"
        );
    }

    #[test]
    fn escape_cancels_hand_drawn_grid_without_polluting_history() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.create_grid_in(&ctx, PixelRect::new(20, 10, 240, 200));
        app.cancel_grid_search();
        app.history.clear();
        let before = app.grid_edit.expect("grid");

        app.start_hand_drawn_grid();
        assert!(app.drawing_grid);
        assert!(app.result.is_none());

        ctx.run_ui(
            input_at(
                0.1,
                vec![egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
            ),
            |_| app.handle_shortcuts(&ctx),
        )
        .textures_delta
        .clear();

        assert!(!app.drawing_grid);
        assert_eq!(app.grid_edit, Some(before));
        assert!(app.result.is_some());
        assert!(
            !app.can_undo(),
            "Esc restores the board without leaving a dead entry on history"
        );
    }

    #[test]
    fn clicking_an_alternative_paints_selected_cell_and_records_undo() {
        use sokovision_core::CellScore;

        let ctx = egui::Context::default();
        let mut app = app_with_learnable_board(&ctx);
        app.batch_by_appearance = false;
        app.selected = Some((1, 2));
        app.result.as_mut().unwrap().cells[1][2].alternatives = vec![CellScore {
            cell: Cell::BoxOnGoal,
            score: 0.72,
        }];

        // Render the side panel once to lay out the alternative button, then
        // verify `apply_cell_choice` updates both the board and undo history.
        let mut output = ctx.run_ui(input_at(0.0, Vec::new()), |ui| {
            app.draw_side_panel(&ctx, ui)
        });
        output.textures_delta.clear();

        app.apply_cell_choice(1, 2, Cell::BoxOnGoal);
        assert_eq!(app.paint_cell, Cell::BoxOnGoal);
        assert_eq!(
            app.result.as_ref().unwrap().cells[1][2].cell,
            Cell::BoxOnGoal
        );
        assert!(app.can_undo());
    }

    #[test]
    fn shortcuts_pick_palette_cells_and_nudge_grid_origin() {
        let ctx = egui::Context::default();
        let mut app = app_with_flat_image(&ctx);
        app.create_grid_in(&ctx, PixelRect::new(20, 10, 240, 200));
        app.cancel_grid_search();
        app.history.clear();
        let before = app.grid_edit.expect("grid");

        let send_events = |app: &mut App, time: f64, events: Vec<egui::Event>| {
            ctx.run_ui(input_at(time, events), |_| app.handle_shortcuts(&ctx))
                .textures_delta
                .clear();
        };

        send_events(
            &mut app,
            0.1,
            vec![egui::Event::Key {
                key: egui::Key::Num3,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert_eq!(app.paint_cell, Cell::Goal);

        send_events(&mut app, 0.2, vec![egui::Event::Text("*".to_owned())]);
        assert_eq!(app.paint_cell, Cell::BoxOnGoal);

        send_events(
            &mut app,
            0.3,
            vec![egui::Event::Key {
                key: egui::Key::ArrowRight,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::SHIFT,
            }],
        );
        let nudged = app.grid_edit.expect("grid");
        assert!((nudged.origin_x - (before.origin_x + 5.0)).abs() < 1e-4);
        assert!(app.grid_settle_at.is_some());
        assert_eq!(app.history.len(), 1);
    }

    #[test]
    fn stitch_monitors_combines_offset_and_scaled_displays() {
        let left = image::RgbaImage::from_pixel(100, 80, image::Rgba([200, 10, 20, 255]));
        let right = image::RgbaImage::from_pixel(150, 120, image::Rgba([20, 40, 210, 255]));
        let stitched = stitch_monitors(vec![
            (-100, 0, 100, 80, left),
            // Logical width 100 at 1.5x DPI -> physical 150x120 starting at x=0.
            (0, 0, 100, 80, right),
        ])
        .expect("valid desktop");

        assert_eq!((stitched.width(), stitched.height()), (250, 120));
        assert_eq!(stitched.get_pixel(10, 10).0, [200, 10, 20, 255]);
        assert_eq!(stitched.get_pixel(140, 100).0, [20, 40, 210, 255]);
    }

    #[test]
    fn frozen_capture_imports_dragged_crop_region_and_esc_keeps_previous_board() {
        let ctx = egui::Context::default();
        let mut app = app_with_learnable_board(&ctx);
        let original_grid = app.result.as_ref().expect("board").grid;

        let mut desktop = image::RgbaImage::from_pixel(320, 240, image::Rgba([15, 15, 20, 255]));
        for y in 40..160 {
            for x in 60..220 {
                desktop.put_pixel(x, y, image::Rgba([90, 140, 210, 255]));
            }
        }

        // Entering frozen capture keeps the existing board underneath so Esc is lossless.
        app.begin_capture_crop(&ctx, desktop.clone());
        assert!(app.is_cropping_capture());
        ctx.run_ui(
            input_at(
                0.1,
                vec![egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
            ),
            |_| app.handle_shortcuts(&ctx),
        )
        .textures_delta
        .clear();
        assert!(!app.is_cropping_capture());
        assert_eq!(app.result.as_ref().expect("preserved").grid, original_grid);

        // Now capture again and drag a rectangle (60, 40)..(220, 160) in image
        // coordinates, which maps to (150, 100)..(550, 400) at 2.5x on 800x600.
        app.begin_capture_crop(&ctx, desktop);
        play_drag(
            &ctx,
            &mut app,
            egui::pos2(150.0, 100.0),
            egui::pos2(550.0, 400.0),
        );
        app.cancel_grid_search();

        assert!(!app.is_cropping_capture());
        let imported = &app.image.as_ref().expect("cropped image").rgba;
        assert_eq!((imported.width(), imported.height()), (160, 120));
        assert_eq!(imported.get_pixel(0, 0).0, [90, 140, 210, 255]);
    }

    fn play_click_at(ctx: &egui::Context, app: &mut App, time: f64, pos: egui::Pos2) {
        let frames = [
            vec![egui::Event::PointerMoved(pos)],
            vec![primary_button(pos, true)],
            vec![primary_button(pos, false)],
        ];
        for (index, events) in frames.into_iter().enumerate() {
            run_frame(ctx, app, input_at(time + index as f64 * 0.05, events));
        }
    }
}
