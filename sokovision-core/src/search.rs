//! Local hill-climbing search around a grid's row and column counts.
//!
//! Edge-based pitch estimation lands close to the true cell size, yet tight
//! hand-drawn brackets or non-integer scaling can still miss by one to three
//! rows or columns. [`refine_grid`] walks neighbouring `(rows, columns)`
//! counts, keeping a step only when its [`RecognitionResult::quality`] beats
//! the current best by at least [`MIN_GAIN`], and tracks visited resolutions so
//! overlapping neighbourhoods are never evaluated twice.

use std::collections::HashSet;

use image::RgbaImage;

use crate::{Grid, RecognitionResult, RecognizeOptions, SkinLibrary, recognize};

/// Minimum [`RecognitionResult::quality`] gain required before a step is taken,
/// preventing tiny rounding differences from nudging the grid back and forth.
const MIN_GAIN: f32 = 0.01;
/// Maximum number of hill-climbing steps away from the starting resolution.
const MAX_STEPS: usize = 3;

/// Climbs from `base` towards neighbouring `(rows, columns)` resolutions that
/// explain `image` clearly better, evaluating each `(rows, columns)` pair at
/// most once. Returns the improved [`RecognitionResult`] only when a better
/// grid is found.
pub fn refine_grid(
    image: &RgbaImage,
    base: Grid,
    base_quality: f32,
    skins: Option<&SkinLibrary>,
) -> Option<RecognitionResult> {
    let skins = skins.filter(|library| !library.skins.is_empty());
    let (width, height) = (image.width(), image.height());
    climb_grid(base, base_quality, &mut |neighbour, visited| {
        let grid = neighbour.clamp_to_image(width, height)?;
        if (grid.rows, grid.columns) != (neighbour.rows, neighbour.columns)
            && !visited.insert((grid.rows, grid.columns))
        {
            return None;
        }
        let options = RecognizeOptions {
            grid: Some(grid),
            skins: skins.cloned(),
            ..RecognizeOptions::default()
        };
        let result = recognize(image, &options).ok()?;
        let quality = result.quality();
        Some((grid, result, quality))
    })
}

/// Generic steepest-ascent walk over [`Grid::neighbour_grids`] with memoized
/// `(rows, columns)` states.
fn climb_grid<T>(
    base: Grid,
    base_quality: f32,
    evaluate: &mut impl FnMut(Grid, &mut HashSet<(u32, u32)>) -> Option<(Grid, T, f32)>,
) -> Option<T> {
    let mut visited = HashSet::with_capacity(1 + MAX_STEPS * 8);
    visited.insert((base.rows, base.columns));

    let mut current_grid = base;
    let mut current_quality = base_quality;
    let mut best_value = None;

    for _ in 0..MAX_STEPS {
        let mut step_best: Option<(Grid, T, f32)> = None;
        for neighbour in current_grid.neighbour_grids() {
            if visited.contains(&(neighbour.rows, neighbour.columns)) {
                continue;
            }
            // Mark the requested counts visited up front; `evaluate` also
            // records any clamped counts so neither alias can be scored twice.
            visited.insert((neighbour.rows, neighbour.columns));
            let Some((resolved_grid, value, quality)) = evaluate(neighbour, &mut visited) else {
                continue;
            };
            if quality < current_quality + MIN_GAIN {
                continue;
            }
            if step_best
                .as_ref()
                .is_none_or(|(_, _, best_quality)| quality > *best_quality)
            {
                step_best = Some((resolved_grid, value, quality));
            }
        }
        let Some((next_grid, next_value, next_quality)) = step_best else {
            break;
        };
        current_grid = next_grid;
        current_quality = next_quality;
        best_value = Some(next_value);
    }

    best_value
}

#[cfg(test)]
mod tests {
    use image::Rgba;

    use super::*;
    use crate::{Board, Cell};

    #[test]
    fn climb_deduplicates_overlapping_neighbourhoods_across_steps() {
        let base = Grid::new(0.0, 0.0, 24.0, 24.0, 10, 10);
        let target = (10u32, 13u32);
        let mut evaluated = Vec::new();

        let winner = climb_grid(base, 0.40, &mut |grid, _| {
            evaluated.push((grid.rows, grid.columns));
            let distance = grid.rows.abs_diff(target.0) + grid.columns.abs_diff(target.1);
            let quality = 0.95 - distance as f32 * 0.12;
            Some((grid, (grid.rows, grid.columns), quality))
        });

        assert_eq!(winner, Some(target));
        let unique: HashSet<(u32, u32)> = evaluated.iter().copied().collect();
        assert_eq!(
            evaluated.len(),
            unique.len(),
            "every (rows, columns) pair must be evaluated at most once: {evaluated:?}"
        );
        // Step 1 checks 8 neighbours, steps 2 and 3 check only the 3 newly
        // exposed columns each (14 total instead of 24 without memoization).
        assert_eq!(evaluated.len(), 14, "{evaluated:?}");
    }

    #[test]
    fn climb_ignores_improvements_below_min_gain() {
        let base = Grid::new(0.0, 0.0, 24.0, 24.0, 6, 8);
        let winner = climb_grid(base, 0.90, &mut |grid, _| {
            Some((grid, grid, 0.90 + MIN_GAIN * 0.5))
        });
        assert!(winner.is_none());
    }

    #[test]
    fn refine_grid_recovers_true_board_counts_from_nearby_guess() {
        let board = Board::from_xsb("#######\n#     #\n#@ $ .#\n#     #\n#######").unwrap();
        let tile = 24u32;
        let mut image = RgbaImage::from_pixel(7 * tile, 5 * tile, Rgba([232, 226, 210, 255]));
        for (row, cells) in board.cells.iter().enumerate() {
            for (column, cell) in cells.iter().enumerate() {
                let x0 = column as u32 * tile;
                let y0 = row as u32 * tile;
                match cell {
                    Cell::Wall => {
                        for y in y0..y0 + tile {
                            for x in x0..x0 + tile {
                                let mortar =
                                    (x - x0).is_multiple_of(6) || (y - y0).is_multiple_of(8);
                                let color = if mortar {
                                    Rgba([145, 155, 185, 255])
                                } else {
                                    Rgba([40, 55, 95, 255])
                                };
                                image.put_pixel(x, y, color);
                            }
                        }
                    }
                    Cell::Player => {
                        for y in y0 + 5..y0 + tile - 5 {
                            for x in x0 + 6..x0 + tile - 6 {
                                image.put_pixel(x, y, Rgba([30, 150, 70, 255]));
                            }
                        }
                    }
                    Cell::Box => {
                        for y in y0 + 2..y0 + tile - 2 {
                            for x in x0 + 2..x0 + tile - 2 {
                                let cross = (x - x0 == 9) || (y - y0 == 9);
                                let color = if cross {
                                    Rgba([230, 200, 150, 255])
                                } else {
                                    Rgba([120, 75, 35, 255])
                                };
                                image.put_pixel(x, y, color);
                            }
                        }
                    }
                    Cell::Goal => {
                        for y in y0 + 8..y0 + tile - 8 {
                            for x in x0 + 8..x0 + tile - 8 {
                                image.put_pixel(x, y, Rgba([200, 30, 40, 255]));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        let off_grid = Grid::new(0.0, 0.0, tile as f32, tile as f32, 5, 7).with_counts(6, 6);
        let initial = recognize(
            &image,
            &RecognizeOptions {
                grid: Some(off_grid),
                ..RecognizeOptions::default()
            },
        )
        .unwrap();
        assert_ne!(initial.board(), board);

        let refined = refine_grid(&image, off_grid, initial.quality(), None)
            .expect("climb finds the 5x7 board");
        assert_eq!((refined.grid.rows, refined.grid.columns), (5, 7));
        assert_eq!(refined.board(), board);
    }
}
