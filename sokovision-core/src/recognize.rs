use image::RgbaImage;
use serde::{Deserialize, Serialize};

use crate::infer;
use crate::skin::{Skin, SkinLibrary, SkinMatch, SkinTile};
use crate::{Board, Cell, Error, Grid, PixelRect, Result, detect, validate_board};

/// Knobs for [`recognize`]. Everything defaults to full automatic detection.
#[derive(Debug, Clone, Default)]
pub struct RecognizeOptions {
    /// Search only inside this region instead of the whole image.
    pub region: Option<PixelRect>,
    /// Use this grid instead of detecting one.
    pub grid: Option<Grid>,
    /// Only detect the grid and leave every cell unresolved.
    pub grid_only: bool,
    /// Remembered tile appearances. When a skin explains the board, its
    /// symbols bias the classification.
    pub skins: Option<SkinLibrary>,
}

/// A symbol a cell could have been, with a relative score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellScore {
    pub cell: Cell,
    pub score: f32,
}

/// One cell of a recognized board.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecognizedCell {
    pub cell: Cell,
    pub confidence: f32,
    pub alternatives: Vec<CellScore>,
    /// Visual appearance group assigned by automatic inference. Cells in the
    /// same group share one tile image, so a manual correction of one of them
    /// usually applies to the whole group.
    pub appearance: Option<u32>,
}

impl RecognizedCell {
    pub(crate) fn unknown() -> Self {
        Self {
            cell: Cell::Unknown,
            confidence: 0.0,
            alternatives: Vec::new(),
            appearance: None,
        }
    }
}

/// What [`recognize`] found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecognitionResult {
    pub grid: Grid,
    pub cells: Vec<Vec<RecognizedCell>>,
    pub warnings: Vec<String>,
    /// Remembered skin that explained this board, when one matched.
    #[serde(default)]
    pub skin: Option<SkinMatch>,
    /// Whether the Sokoban constraints had to be relaxed to force a visual
    /// guess.
    #[serde(default)]
    pub relaxed: bool,
}

impl RecognitionResult {
    pub fn board(&self) -> Board {
        Board {
            cells: self
                .cells
                .iter()
                .map(|row| row.iter().map(|cell| cell.cell).collect())
                .collect(),
        }
    }

    /// Positions that share one visual appearance group.
    pub fn appearance_cells(&self, appearance: u32) -> Vec<(usize, usize)> {
        self.positions(|cell| cell.appearance == Some(appearance))
    }

    /// Positions of every cell that is still unresolved, in row-major order.
    pub fn unknown_cells(&self) -> Vec<(usize, usize)> {
        self.positions(|cell| cell.cell == Cell::Unknown)
    }

    /// Comparable quality of the whole result: cell confidence penalised by
    /// validation problems, relaxed constraints and unresolved cells.
    pub fn quality(&self) -> f32 {
        let count = self.cells.iter().flatten().count().max(1);
        let mean = self
            .cells
            .iter()
            .flatten()
            .map(|cell| cell.confidence)
            .sum::<f32>()
            / count as f32;
        let problems = validate_board(&self.board()).len() as f32;
        let unknown = self.unknown_cells().len() as f32;
        (mean - 0.25 * problems - if self.relaxed { 0.35 } else { 0.0 } - 0.02 * unknown)
            .clamp(0.0, 1.0)
    }

    /// Whether the result is trustworthy enough to teach the skin library:
    /// structurally valid, nothing unresolved, and the level constraints were
    /// not relaxed to force a guess. A relaxed read is a best-effort visual
    /// guess, so remembering it would file noise as if it were a known tile.
    pub fn learnable(&self) -> bool {
        self.unknown_cells().is_empty() && validate_board(&self.board()).is_empty() && !self.relaxed
    }

    /// Positions of the cells that satisfy `keep`, in row-major order.
    fn positions(&self, keep: impl Fn(&RecognizedCell) -> bool) -> Vec<(usize, usize)> {
        let mut positions = Vec::new();
        for (row, cells) in self.cells.iter().enumerate() {
            for (column, cell) in cells.iter().enumerate() {
                if keep(cell) {
                    positions.push((row, column));
                }
            }
        }
        positions
    }

    /// Sets `cell` at the position and, when `same_appearance` is set and the
    /// position belongs to a group, on every cell of that group. Returns how
    /// many cells were painted.
    pub fn paint(&mut self, row: usize, column: usize, cell: Cell, same_appearance: bool) -> usize {
        let Some(selected) = self.cells.get(row).and_then(|cells| cells.get(column)) else {
            return 0;
        };
        let group = selected.appearance.filter(|_| same_appearance);

        let mut painted = 0;
        for (row_index, cells) in self.cells.iter_mut().enumerate() {
            for (column_index, target) in cells.iter_mut().enumerate() {
                let selected = match group {
                    Some(appearance) => target.appearance == Some(appearance),
                    None => row_index == row && column_index == column,
                };
                if !selected {
                    continue;
                }
                target.cell = cell;
                target.confidence = 1.0;
                target.alternatives.clear();
                painted += 1;
            }
        }
        if painted > 0
            && self.relaxed
            && self.unknown_cells().is_empty()
            && validate_board(&self.board()).is_empty()
        {
            self.relaxed = false;
        }
        painted
    }

    /// Remembers the tiles on this board: one prototype per appearance group,
    /// sampled from a cell of that group. A group whose cells ended up with
    /// different symbols is decided by the painted cell (confidence 1.0), so a
    /// hand correction is what gets remembered instead of competing with the
    /// guess it replaced. `None` when no cell can be read.
    pub fn skin(&self, image: &RgbaImage, name: impl Into<String>) -> Option<Skin> {
        let mut selected: Vec<(u32, Cell, f32, PixelRect)> = Vec::new();
        for (row, cells) in self.cells.iter().enumerate() {
            for (column, recognized) in cells.iter().enumerate() {
                let Some(appearance) = recognized.appearance else {
                    continue;
                };
                if recognized.cell == Cell::Unknown {
                    continue;
                }
                let Some(rect) = self.grid.cell_rect(row, column) else {
                    continue;
                };
                if rect.right() > image.width()
                    || rect.bottom() > image.height()
                    || !infer::usable_tile(rect)
                {
                    continue;
                }
                let candidate = (appearance, recognized.cell, recognized.confidence, rect);
                match selected.iter().position(|(group, ..)| *group == appearance) {
                    Some(index) => {
                        if recognized.confidence > selected[index].2 {
                            selected[index] = candidate;
                        }
                    }
                    None => selected.push(candidate),
                }
            }
        }

        let tiles: Vec<SkinTile> = selected
            .into_iter()
            .filter_map(|(_, cell, _, rect)| {
                let (color, peak) = infer::tile_prototype_rect(image, rect)?;
                let color = crate::skin::coarse(&color)?;
                Some(SkinTile {
                    cell,
                    color,
                    peak: Some(peak),
                })
            })
            .collect();
        (!tiles.is_empty()).then(|| Skin {
            name: name.into(),
            tiles,
        })
    }
}

/// Reads a Sokoban board out of a screenshot.
pub fn recognize(image: &RgbaImage, options: &RecognizeOptions) -> Result<RecognitionResult> {
    if image.width() == 0 || image.height() == 0 {
        return Err(Error::EmptyImage);
    }

    let grids = match options.grid {
        Some(grid) => vec![grid.validate(image.width(), image.height())?],
        None => {
            let region = options
                .region
                .unwrap_or_else(|| PixelRect::new(0, 0, image.width(), image.height()))
                .validate_inside(image.width(), image.height())?;
            detect::candidates_in(image, region)?
        }
    };

    if options.grid_only {
        let grid = grids
            .into_iter()
            .next()
            .unwrap()
            .clamp_to_image(image.width(), image.height())
            .ok_or_else(|| Error::GridDetection("detected grid is outside the image".to_owned()))?;
        let mut warnings = Vec::new();
        if options.grid.is_none() {
            warnings.push(detected_warning(grid));
        }
        warnings.push("grid detection only; every cell needs manual confirmation".to_owned());
        return Ok(RecognitionResult {
            grid,
            cells: vec![vec![RecognizedCell::unknown(); grid.columns as usize]; grid.rows as usize],
            warnings,
            skin: None,
            relaxed: false,
        });
    }

    let primary_cells = grids[0].rows as usize * grids[0].columns as usize;
    let mut best: Option<(RecognitionResult, f32)> = None;
    for (index, grid) in grids.into_iter().enumerate() {
        let cell_count = grid.rows as usize * grid.columns as usize;
        let grid = grid
            .clamp_to_image(image.width(), image.height())
            .ok_or_else(|| Error::GridDetection("detected grid is outside the image".to_owned()))?;
        let mut warnings = Vec::new();
        if options.grid.is_none() {
            warnings.push(detected_warning(grid));
        }
        let inference = infer::infer(image, grid, options.skins.as_ref());
        let incomplete_structure = inference.incomplete_structure;
        let mut result = RecognitionResult {
            grid,
            cells: inference.cells,
            warnings,
            skin: inference.skin,
            relaxed: inference.relaxed,
        };
        result.warnings.extend(inference.warnings);
        result.warnings.extend(validate_board(&result.board()));
        let score = result.quality();
        if index == 0 {
            let retry = needs_retry(&result, incomplete_structure);
            best = Some((result, score));
            if !retry {
                break;
            }
            continue;
        }
        // An alternative may only replace the primary when it reads as a
        // legal board itself. A half-pitch fit doubles both axes, so a grid
        // with over twice the cells is a subharmonic of the primary pitch,
        // not a correction of it.
        let acceptable = validate_board(&result.board()).is_empty()
            && result.unknown_cells().is_empty()
            && cell_count <= 2 * primary_cells;
        if acceptable
            && best
                .as_ref()
                .is_none_or(|(_, best_score)| score > *best_score)
        {
            best = Some((result, score));
        }
    }
    Ok(best.unwrap().0)
}

fn detected_warning(grid: Grid) -> String {
    format!(
        "automatically detected a {}x{} grid at ({:.1}, {:.1}), {}x{} px cells",
        grid.columns, grid.rows, grid.origin_x, grid.origin_y, grid.cell_width, grid.cell_height
    )
}

fn needs_retry(result: &RecognitionResult, incomplete_structure: bool) -> bool {
    !validate_board(&result.board()).is_empty()
        || !result.unknown_cells().is_empty()
        || result.relaxed
        || incomplete_structure
}

#[cfg(test)]
mod tests {
    use image::{Rgba, RgbaImage};

    use super::*;

    fn drawn_tile(cell: Cell) -> RgbaImage {
        let mut image = RgbaImage::from_pixel(24, 24, Rgba([232, 226, 210, 255]));
        match cell {
            Cell::Wall => {
                image = RgbaImage::from_pixel(24, 24, Rgba([40, 55, 95, 255]));
                for x in (0..24).step_by(6) {
                    for y in 0..24 {
                        image.put_pixel(x, y, Rgba([150, 160, 190, 255]));
                    }
                }
                for y in (0..24).step_by(8) {
                    for x in 0..24 {
                        image.put_pixel(x, y, Rgba([120, 130, 170, 255]));
                    }
                }
            }
            Cell::Goal => {
                for y in 8..16 {
                    for x in 8..16 {
                        image.put_pixel(x, y, Rgba([200, 30, 40, 255]));
                    }
                }
            }
            Cell::Box => {
                image = RgbaImage::from_pixel(24, 24, Rgba([120, 75, 35, 255]));
                for index in 4..20 {
                    for width in 0..2 {
                        image.put_pixel(index, 8 + width, Rgba([230, 200, 150, 255]));
                        image.put_pixel(8 + width, index, Rgba([230, 200, 150, 255]));
                    }
                }
            }
            Cell::Player | Cell::PlayerOnGoal => {
                for y in 5..19 {
                    for x in 6..18 {
                        let dx = (x as f32 - 11.5) / 6.0;
                        let dy = (y as f32 - 11.5) / 7.0;
                        if dx * dx + dy * dy <= 1.0 {
                            image.put_pixel(x, y, Rgba([30, 150, 70, 255]));
                        }
                    }
                }
                for y in 8..13 {
                    for x in 9..15 {
                        image.put_pixel(x, y, Rgba([240, 200, 60, 255]));
                    }
                }
            }
            Cell::Empty | Cell::BoxOnGoal | Cell::Unknown => {}
        }
        image
    }

    fn recognized(cell: Cell, appearance: Option<u32>) -> RecognizedCell {
        RecognizedCell {
            cell,
            confidence: 0.5,
            alternatives: Vec::new(),
            appearance,
        }
    }

    fn render(board: &Board) -> RgbaImage {
        const CELL: u32 = 24;
        let mut image = RgbaImage::from_pixel(
            board.width() as u32 * CELL,
            board.height() as u32 * CELL,
            Rgba([0, 0, 0, 255]),
        );
        for (row, cells) in board.cells.iter().enumerate() {
            for (column, cell) in cells.iter().enumerate() {
                image::imageops::replace(
                    &mut image,
                    &drawn_tile(*cell),
                    (column as u32 * CELL) as i64,
                    (row as u32 * CELL) as i64,
                );
            }
        }
        image
    }

    fn recognize_rendered(level: &str) -> RecognitionResult {
        let board = Board::from_xsb(level).unwrap();
        let grid = Grid::new(
            0.0,
            0.0,
            24.0,
            24.0,
            board.height() as u32,
            board.width() as u32,
        );
        recognize(
            &render(&board),
            &RecognizeOptions {
                grid: Some(grid),
                ..RecognizeOptions::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn paints_every_cell_of_one_appearance() {
        let mut result = RecognitionResult {
            grid: Grid::new(0.0, 0.0, 24.0, 24.0, 2, 3),
            cells: vec![
                vec![
                    recognized(Cell::Box, Some(0)),
                    recognized(Cell::Box, Some(0)),
                    recognized(Cell::Empty, Some(1)),
                ],
                vec![
                    recognized(Cell::Goal, Some(1)),
                    recognized(Cell::Player, None),
                    recognized(Cell::Box, Some(0)),
                ],
            ],
            warnings: Vec::new(),
            skin: None,
            relaxed: false,
        };

        assert_eq!(result.paint(0, 0, Cell::Goal, true), 3);
        assert_eq!(result.cells[0][1].cell, Cell::Goal);
        assert_eq!(result.cells[1][2].cell, Cell::Goal);
        assert_eq!(result.cells[0][2].cell, Cell::Empty);
        assert_eq!(result.cells[1][0].cell, Cell::Goal);
        assert_eq!(result.cells[0][0].confidence, 1.0);

        assert_eq!(result.paint(0, 0, Cell::Box, false), 1);
        assert_eq!(result.cells[0][0].cell, Cell::Box);
        assert_eq!(result.cells[1][2].cell, Cell::Goal);

        assert_eq!(result.paint(1, 1, Cell::Wall, true), 1);
        assert_eq!(result.cells[1][1].cell, Cell::Wall);
    }

    /// One appearance keeps one prototype: when a hand correction left two
    /// symbols in the same group, the painted cell decides, so re-reading the
    /// board does not fall back to the guess the user just fixed.
    #[test]
    fn a_painted_cell_decides_what_its_appearance_is() {
        let image = RgbaImage::from_pixel(48, 48, Rgba([40, 40, 40, 255]));
        let mut result = RecognitionResult {
            grid: Grid::new(0.0, 0.0, 24.0, 24.0, 2, 2),
            cells: vec![
                vec![
                    recognized(Cell::Box, Some(0)),
                    recognized(Cell::BoxOnGoal, Some(0)),
                ],
                vec![
                    recognized(Cell::Empty, Some(1)),
                    recognized(Cell::Empty, Some(1)),
                ],
            ],
            warnings: Vec::new(),
            skin: None,
            relaxed: false,
        };
        // Only painting sets a cell to certainty; inference stops short of it.
        result.cells[0][1].confidence = 1.0;

        let skin = result.skin(&image, "learned").expect("tiles");

        assert_eq!(
            skin.tiles.len(),
            2,
            "one prototype per appearance: {skin:?}"
        );
        assert_eq!(
            skin.tiles[0].cell,
            Cell::BoxOnGoal,
            "the painted symbol replaces the guess it was painted over"
        );
    }

    #[test]
    fn infers_a_board_from_visuals_only() {
        let level = "#######\n#@--$.#\n#-$---#\n#--.--#\n#######";
        let result = recognize_rendered(level);

        assert!(
            result.unknown_cells().is_empty(),
            "unresolved cells: {:?}\nwarnings: {:?}",
            result.unknown_cells(),
            result.warnings
        );
        assert_eq!(result.board().to_xsb().unwrap(), level);
        assert!(
            !result.warnings.iter().any(|warning| {
                warning.contains("cannot satisfy") || warning.contains("validation failed")
            }),
            "{:?}",
            result.warnings
        );
    }

    #[test]
    fn infers_a_player_standing_on_a_goal_from_counts() {
        let level = "#######\n#+--$.#\n#-$---#\n#--$.-#\n#######";
        let result = recognize_rendered(level);

        assert!(
            result.unknown_cells().is_empty(),
            "unresolved cells: {:?}\nwarnings: {:?}",
            result.unknown_cells(),
            result.warnings
        );
        assert_eq!(result.board().to_xsb().unwrap(), level);
    }

    #[test]
    fn relaxed_flag_gates_learning_and_quality_independent_of_warning_text() {
        let mut result = recognize_rendered("#####\n#@$.#\n#####");
        result.warnings.clear();
        let baseline = result.quality();
        assert!(result.learnable());

        result.relaxed = true;
        assert!(!result.learnable());
        assert!(result.quality() < baseline);
    }

    #[test]
    fn relaxed_reads_cap_wall_confidence_and_recover_once_repaired_by_hand() {
        let mut result = recognize_rendered("#######\n#@..--#\n#######");

        assert!(result.relaxed);
        assert!(!result.learnable());
        assert!(
            result
                .cells
                .iter()
                .flatten()
                .all(|cell| cell.confidence <= 0.55),
            "relaxed reads must not report high-confidence walls"
        );
        assert!(
            !result.cells[1][2].alternatives.is_empty(),
            "relaxed interior cells still offer visual alternatives"
        );

        result.paint(1, 2, Cell::Box, false);
        assert_eq!(
            result.board().to_xsb().unwrap(),
            "#######\n#@$.--#\n#######"
        );
        assert!(!result.relaxed);
        assert!(result.learnable());
    }
}
