use serde::{Deserialize, Serialize};

use crate::{Error, MAX_BOARD_AXIS, Result};

/// An axis-aligned rectangle in image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PixelRect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Smallest x coordinate past the right edge.
    pub const fn right(self) -> u32 {
        self.x.saturating_add(self.width)
    }

    /// Smallest y coordinate past the bottom edge.
    pub const fn bottom(self) -> u32 {
        self.y.saturating_add(self.height)
    }

    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Shrinks the rectangle by `margin` pixels on every side, if anything is
    /// left over.
    pub fn inset(self, margin: u32) -> Option<Self> {
        let margin = margin.min(self.width / 2).min(self.height / 2);
        let rect = Self::new(
            self.x + margin,
            self.y + margin,
            self.width - margin * 2,
            self.height - margin * 2,
        );
        (!rect.is_empty()).then_some(rect)
    }

    pub(crate) fn validate_inside(self, image_width: u32, image_height: u32) -> Result<Self> {
        if self.is_empty() || self.right() > image_width || self.bottom() > image_height {
            return Err(Error::InvalidRegion(self));
        }
        Ok(self)
    }
}

/// The board grid: where cells start, how large they are and how many there are.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    pub origin_x: f32,
    pub origin_y: f32,
    pub cell_width: f32,
    pub cell_height: f32,
    pub rows: u32,
    pub columns: u32,
}

impl Grid {
    pub const fn new(
        origin_x: f32,
        origin_y: f32,
        cell_width: f32,
        cell_height: f32,
        rows: u32,
        columns: u32,
    ) -> Self {
        Self {
            origin_x,
            origin_y,
            cell_width,
            cell_height,
            rows,
            columns,
        }
    }

    pub fn width(self) -> f32 {
        self.cell_width * self.columns as f32
    }

    pub fn height(self) -> f32 {
        self.cell_height * self.rows as f32
    }

    /// Re-derives the cell pitch from new row and column counts while keeping
    /// the covered rectangle unchanged, so adjusting the resolution does not
    /// move the board outline.
    pub fn with_counts(self, rows: u32, columns: u32) -> Self {
        let rows = rows.max(1);
        let columns = columns.max(1);
        Self::new(
            self.origin_x,
            self.origin_y,
            (self.width() / columns as f32).max(1.0),
            (self.height() / rows as f32).max(1.0),
            rows,
            columns,
        )
    }

    /// Distinct resolutions one step (`-1..=1` rows and columns) away from
    /// `self` within `1..=MAX_BOARD_AXIS`, preserving the covered outline.
    pub fn neighbour_grids(self) -> Vec<Self> {
        let mut grids = Vec::with_capacity(8);
        for row_delta in [-1i32, 0, 1] {
            for column_delta in [-1i32, 0, 1] {
                if row_delta == 0 && column_delta == 0 {
                    continue;
                }
                let rows = (self.rows as i32 + row_delta).clamp(1, MAX_BOARD_AXIS as i32) as u32;
                let columns =
                    (self.columns as i32 + column_delta).clamp(1, MAX_BOARD_AXIS as i32) as u32;
                if (rows, columns) == (self.rows, self.columns) {
                    continue;
                }
                if grids
                    .iter()
                    .any(|existing: &Self| (existing.rows, existing.columns) == (rows, columns))
                {
                    continue;
                }
                grids.push(self.with_counts(rows, columns));
            }
        }
        grids
    }

    /// Pixel rectangle covered by one cell, rounded to whole pixels.
    pub fn cell_rect(self, row: usize, column: usize) -> Option<PixelRect> {
        if row >= self.rows as usize || column >= self.columns as usize {
            return None;
        }
        let left = self.origin_x + self.cell_width * column as f32;
        let top = self.origin_y + self.cell_height * row as f32;
        let right = left + self.cell_width;
        let bottom = top + self.cell_height;
        let finite = left.is_finite() && top.is_finite() && right.is_finite() && bottom.is_finite();
        if !finite || left < 0.0 || top < 0.0 {
            return None;
        }
        let (x, y) = (left.round() as u32, top.round() as u32);
        let rect = PixelRect::new(
            x,
            y,
            (right.round() as u32).saturating_sub(x),
            (bottom.round() as u32).saturating_sub(y),
        );
        (!rect.is_empty()).then_some(rect)
    }

    /// Fits the grid into the image by clamping the origin and dropping rows or
    /// columns that would fall past the right or bottom edge.
    pub fn clamp_to_image(mut self, image_width: u32, image_height: u32) -> Option<Self> {
        if image_width == 0 || image_height == 0 {
            return None;
        }
        self.validate_shape().ok()?;

        let (width, height) = (image_width as f32, image_height as f32);
        self.origin_x = self.origin_x.clamp(0.0, (width - self.width()).max(0.0));
        self.origin_y = self.origin_y.clamp(0.0, (height - self.height()).max(0.0));
        self.columns = self
            .columns
            .min(fit_count(width - self.origin_x, self.cell_width));
        self.rows = self
            .rows
            .min(fit_count(height - self.origin_y, self.cell_height));
        self.validate(image_width, image_height).ok()?;
        Some(self)
    }

    pub fn validate(self, image_width: u32, image_height: u32) -> Result<Self> {
        self.validate_shape()?;
        if self.origin_x < 0.0 || self.origin_y < 0.0 {
            return Err(Error::InvalidGrid(
                "grid origin must not be negative".to_owned(),
            ));
        }
        if self.origin_x + self.width() > image_width as f32
            || self.origin_y + self.height() > image_height as f32
        {
            return Err(Error::InvalidGrid(
                "grid extends outside the image".to_owned(),
            ));
        }
        Ok(self)
    }

    /// Checks everything that does not depend on the image size or on the
    /// position of the grid.
    fn validate_shape(self) -> Result<Self> {
        if self.rows == 0 || self.columns == 0 {
            return Err(Error::InvalidGrid(
                "rows and columns must be greater than zero".to_owned(),
            ));
        }
        let finite = self.origin_x.is_finite()
            && self.origin_y.is_finite()
            && self.cell_width.is_finite()
            && self.cell_height.is_finite();
        if !finite {
            return Err(Error::InvalidGrid("grid values must be finite".to_owned()));
        }
        if self.cell_width < 2.0 || self.cell_height < 2.0 {
            return Err(Error::InvalidGrid(
                "cell width and height must be at least 2 pixels".to_owned(),
            ));
        }
        Ok(self)
    }
}

/// How many whole cells of `pitch` fit into `extent`, but never zero.
fn fit_count(extent: f32, pitch: f32) -> u32 {
    ((extent / pitch).floor() as u32).max(1)
}

/// The up, down, left and right neighbours of a cell that stay on the board.
pub(crate) fn neighbours(
    row: usize,
    column: usize,
    rows: usize,
    columns: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let up = row.checked_sub(1).map(|row| (row, column));
    let down = (row + 1 < rows).then_some((row + 1, column));
    let left = column.checked_sub(1).map(|column| (row, column));
    let right = (column + 1 < columns).then_some((row, column + 1));
    [up, down, left, right].into_iter().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_a_valid_grid_unchanged() {
        let grid = Grid::new(10.0, 12.0, 20.0, 20.0, 4, 5);
        assert_eq!(grid.clamp_to_image(200, 200), Some(grid));
    }

    #[test]
    fn moves_an_outside_grid_back_inside_the_image() {
        let grid = Grid::new(180.0, 170.0, 20.0, 20.0, 4, 5);
        let clamped = grid.clamp_to_image(200, 200).expect("positive cells");

        assert_eq!((clamped.rows, clamped.columns), (4, 5));
        assert_eq!(clamped.origin_x, 100.0);
        assert_eq!(clamped.origin_y, 120.0);
    }

    #[test]
    fn drops_rows_and_columns_that_cannot_fit() {
        let grid = Grid::new(0.0, 0.0, 20.0, 20.0, 20, 20);
        let clamped = grid.clamp_to_image(100, 80).expect("positive cells");
        assert_eq!((clamped.rows, clamped.columns), (4, 5));
    }

    #[test]
    fn moves_a_negative_origin_inside_the_image() {
        let grid = Grid::new(-10.0, -8.0, 20.0, 20.0, 4, 5);
        let clamped = grid.clamp_to_image(200, 200).expect("positive cells");

        assert_eq!((clamped.origin_x, clamped.origin_y), (0.0, 0.0));
        assert_eq!((clamped.rows, clamped.columns), (4, 5));
    }

    #[test]
    fn rejects_cells_below_two_pixels() {
        let grid = Grid::new(0.0, 0.0, 1.0, 1.0, 4, 5);
        assert!(grid.validate(200, 200).is_err());
        assert_eq!(grid.clamp_to_image(200, 200), None);
    }

    #[test]
    fn insets_a_rectangle_on_every_side() {
        let rect = PixelRect::new(10, 10, 40, 30).inset(4).expect("roomy");
        assert_eq!(rect, PixelRect::new(14, 14, 32, 22));
        assert_eq!(PixelRect::new(0, 0, 4, 4).inset(4), None);
    }

    #[test]
    fn recounting_keeps_the_covered_area_and_refuses_zero_counts() {
        let grid = Grid::new(10.0, 20.0, 36.0, 36.0, 11, 19);
        let recounted = grid.with_counts(12, 20);

        assert_eq!((recounted.rows, recounted.columns), (12, 20));
        assert!((recounted.width() - grid.width()).abs() < 1e-3);
        assert!((recounted.height() - grid.height()).abs() < 1e-3);
        assert_eq!((recounted.origin_x, recounted.origin_y), (10.0, 20.0));

        let non_empty = Grid::new(0.0, 0.0, 10.0, 10.0, 4, 5).with_counts(0, 0);
        assert_eq!((non_empty.rows, non_empty.columns), (1, 1));
    }

    #[test]
    fn neighbour_grids_keep_the_outline_and_deduplicate_boundaries() {
        let base = Grid::new(10.0, 20.0, 36.0, 36.0, 11, 19);
        let grids = base.neighbour_grids();

        assert_eq!(grids.len(), 8);
        for grid in &grids {
            assert_eq!((grid.origin_x, grid.origin_y), (10.0, 20.0));
            assert!((grid.width() - base.width()).abs() < 1e-3);
            assert!((grid.height() - base.height()).abs() < 1e-3);
            assert_ne!((grid.rows, grid.columns), (base.rows, base.columns));
        }

        let tiny = Grid::new(0.0, 0.0, 20.0, 20.0, 1, 1).neighbour_grids();
        assert_eq!(tiny.len(), 3, "clamped (1, 1) duplicates are omitted");
        assert!(tiny.iter().all(|grid| grid.rows >= 1 && grid.columns >= 1));

        let huge =
            Grid::new(0.0, 0.0, 20.0, 20.0, MAX_BOARD_AXIS, MAX_BOARD_AXIS).neighbour_grids();
        assert_eq!(huge.len(), 3);
        assert!(
            huge.iter()
                .all(|grid| grid.rows <= MAX_BOARD_AXIS && grid.columns <= MAX_BOARD_AXIS)
        );
    }
}
