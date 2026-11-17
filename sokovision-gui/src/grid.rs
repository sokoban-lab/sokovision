//! Grid geometry: resizing, snapping and hit testing.

use egui::{Pos2, Rect, Vec2, pos2, vec2};
use sokovision_core::{Grid, MAX_BOARD_AXIS, PixelRect};

/// Corner of the grid outline the pointer grabbed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Side of the grid outline the pointer grabbed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    Top,
    Bottom,
    Left,
    Right,
}

/// Part of the grid outline a drag can move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handle {
    Corner(Corner),
    Edge(Edge),
}

/// Radius in screen pixels within which a corner handle reacts.
const HANDLE_RADIUS: f32 = 10.0;
/// Radius in screen pixels within which an edge reacts. Smaller than the
/// corner radius because the whole outline is draggable now, and the border
/// cells have to stay clickable.
const EDGE_RADIUS: f32 = 6.0;
/// Smallest cell size a corner drag may produce, in pixels.
const MIN_CELL: f32 = 4.0;
/// Smallest rubber band, in image pixels, that is taken for a board outline.
const MIN_DRAG: f32 = 16.0;
/// Cell aspect ratio a detected grid must have before it is trusted.
const CELL_ASPECT: std::ops::RangeInclusive<f32> = 0.65..=1.55;
/// Rows a hand-drawn grid starts with along the shorter side. Sokoban boards
/// are rarely taller than this, so the outline lands in the right ballpark.
const DEFAULT_ROWS: f32 = 12.0;

/// Screen positions of the four handles, clockwise from the top left.
pub fn corner_points(grid: &Grid, image_rect: Rect, scale: f32) -> [(Corner, Pos2); 4] {
    let left = image_rect.min.x + grid.origin_x * scale;
    let top = image_rect.min.y + grid.origin_y * scale;
    let right = image_rect.min.x + (grid.origin_x + grid.width()) * scale;
    let bottom = image_rect.min.y + (grid.origin_y + grid.height()) * scale;
    [
        (Corner::TopLeft, pos2(left, top)),
        (Corner::TopRight, pos2(right, top)),
        (Corner::BottomLeft, pos2(left, bottom)),
        (Corner::BottomRight, pos2(right, bottom)),
    ]
}

pub fn corner_at(grid: &Grid, pointer: Pos2, image_rect: Rect, scale: f32) -> Option<Corner> {
    corner_points(grid, image_rect, scale)
        .into_iter()
        .find(|(_, point)| point.distance(pointer) <= HANDLE_RADIUS)
        .map(|(corner, _)| corner)
}

/// Screen rectangle the grid outline covers.
pub fn outline_rect(grid: &Grid, image_rect: Rect, scale: f32) -> Rect {
    let points = corner_points(grid, image_rect, scale);
    Rect::from_min_max(points[0].1, points[3].1)
}

/// The outline handle under the pointer. Corners win over edges, and edges win
/// in the order of how close they are, so a drag near a corner keeps resizing
/// both axes.
pub fn handle_at(grid: &Grid, pointer: Pos2, image_rect: Rect, scale: f32) -> Option<Handle> {
    if let Some(corner) = corner_at(grid, pointer, image_rect, scale) {
        return Some(Handle::Corner(corner));
    }

    let outline = outline_rect(grid, image_rect, scale);
    let reachable = Rect::from_min_max(
        outline.min - Vec2::splat(EDGE_RADIUS),
        outline.max + Vec2::splat(EDGE_RADIUS),
    );
    if !reachable.contains(pointer) {
        return None;
    }

    [
        (Edge::Left, (pointer.x - outline.min.x).abs()),
        (Edge::Right, (pointer.x - outline.max.x).abs()),
        (Edge::Top, (pointer.y - outline.min.y).abs()),
        (Edge::Bottom, (pointer.y - outline.max.y).abs()),
    ]
    .into_iter()
    .filter(|(_, distance)| *distance <= EDGE_RADIUS)
    .min_by(|left, right| left.1.total_cmp(&right.1))
    .map(|(edge, _)| Handle::Edge(edge))
}

/// Image position of the grabbed part of the outline: the corner itself, or
/// the middle of the grabbed edge.
pub fn handle_position(grid: &Grid, handle: Handle) -> Pos2 {
    let (right, bottom) = (grid.origin_x + grid.width(), grid.origin_y + grid.height());
    match handle {
        Handle::Corner(Corner::TopLeft) => pos2(grid.origin_x, grid.origin_y),
        Handle::Corner(Corner::TopRight) => pos2(right, grid.origin_y),
        Handle::Corner(Corner::BottomLeft) => pos2(grid.origin_x, bottom),
        Handle::Corner(Corner::BottomRight) => pos2(right, bottom),
        Handle::Edge(Edge::Left) => pos2(grid.origin_x, (grid.origin_y + bottom) * 0.5),
        Handle::Edge(Edge::Right) => pos2(right, (grid.origin_y + bottom) * 0.5),
        Handle::Edge(Edge::Top) => pos2((grid.origin_x + right) * 0.5, grid.origin_y),
        Handle::Edge(Edge::Bottom) => pos2((grid.origin_x + right) * 0.5, bottom),
    }
}

/// Moves the grabbed part of the axis aligned grid: the opposite side stays put
/// and the cell size is recomputed from the new extent. Dragging an edge only
/// changes that axis, so the other cell size is left alone.
pub fn resize(grid: Grid, handle: Handle, pointer: Pos2) -> Grid {
    let mut left = grid.origin_x;
    let mut top = grid.origin_y;
    let mut right = grid.origin_x + grid.width();
    let mut bottom = grid.origin_y + grid.height();
    match handle {
        Handle::Corner(Corner::TopLeft) => {
            left = pointer.x;
            top = pointer.y;
        }
        Handle::Corner(Corner::TopRight) => {
            right = pointer.x;
            top = pointer.y;
        }
        Handle::Corner(Corner::BottomLeft) => {
            left = pointer.x;
            bottom = pointer.y;
        }
        Handle::Corner(Corner::BottomRight) => {
            right = pointer.x;
            bottom = pointer.y;
        }
        Handle::Edge(Edge::Left) => left = pointer.x,
        Handle::Edge(Edge::Right) => right = pointer.x,
        Handle::Edge(Edge::Top) => top = pointer.y,
        Handle::Edge(Edge::Bottom) => bottom = pointer.y,
    }

    let mut grid = grid;
    if right - left >= MIN_CELL * grid.columns as f32 {
        grid.origin_x = left;
        grid.cell_width = (right - left) / grid.columns as f32;
    }
    if bottom - top >= MIN_CELL * grid.rows as f32 {
        grid.origin_y = top;
        grid.cell_height = (bottom - top) / grid.rows as f32;
    }
    grid
}

/// The bracket the user dragged is the board outline, so the detected cell
/// counts are spread across exactly that area. Detection inside a tight crop
/// is slightly biased by the inset sprites, which would shift the grid.
pub fn snap_to_region(region: PixelRect, detected: Grid) -> Option<Grid> {
    let columns = detected.columns.max(1);
    let rows = detected.rows.max(1);
    let cell_width = region.width as f32 / columns as f32;
    let cell_height = region.height as f32 / rows as f32;
    if !CELL_ASPECT.contains(&(cell_width / cell_height)) {
        return None;
    }
    Some(Grid::new(
        region.x as f32,
        region.y as f32,
        cell_width,
        cell_height,
        rows,
        columns,
    ))
}

/// A grid covering `region`, used when nothing has been detected yet.
///
/// Cells are square and roughly [`DEFAULT_ROWS`] of them fit along the shorter
/// side, so the counts are in the right ballpark for a Sokoban board and only
/// need nudging with the steppers.
pub fn covering(region: PixelRect) -> Grid {
    let width = region.width.max(1) as f32;
    let height = region.height.max(1) as f32;
    let mut cell = (width.min(height) / DEFAULT_ROWS).clamp(8.0, 96.0);
    // A very lopsided bracket would otherwise ask for more cells than any
    // board has, which the recognizer refuses to work on; widen the cells
    // instead so the drag still yields a usable grid.
    cell = cell
        .max(width / MAX_BOARD_AXIS as f32)
        .max(height / MAX_BOARD_AXIS as f32);
    let columns = ((width / cell).round() as u32).clamp(1, MAX_BOARD_AXIS);
    let rows = ((height / cell).round() as u32).clamp(1, MAX_BOARD_AXIS);
    Grid::new(
        region.x as f32,
        region.y as f32,
        width / columns as f32,
        height / rows as f32,
        rows,
        columns,
    )
}

/// Pixel rectangle spanned by two image positions, clamped to the image.
/// `None` when the drag is too small to hold a board.
pub fn region_between(start: Pos2, end: Pos2, image_size: Vec2) -> Option<PixelRect> {
    let left = start.x.min(end.x).clamp(0.0, image_size.x);
    let right = start.x.max(end.x).clamp(0.0, image_size.x);
    let top = start.y.min(end.y).clamp(0.0, image_size.y);
    let bottom = start.y.max(end.y).clamp(0.0, image_size.y);
    let (width, height) = ((right - left).round(), (bottom - top).round());
    if width < MIN_DRAG || height < MIN_DRAG {
        return None;
    }
    Some(PixelRect::new(
        left.round() as u32,
        top.round() as u32,
        width as u32,
        height as u32,
    ))
}

/// Image area spanned by the grid, clamped to the image bounds.
pub fn region(grid: Grid, image_width: u32, image_height: u32) -> PixelRect {
    let width = image_width.max(1) as f32;
    let height = image_height.max(1) as f32;
    let left = grid.origin_x.clamp(0.0, width - 1.0);
    let top = grid.origin_y.clamp(0.0, height - 1.0);
    let right = (grid.origin_x + grid.width()).clamp(left + 1.0, width);
    let bottom = (grid.origin_y + grid.height()).clamp(top + 1.0, height);
    PixelRect::new(
        left.round() as u32,
        top.round() as u32,
        (right - left).round().max(1.0) as u32,
        (bottom - top).round().max(1.0) as u32,
    )
}

/// Screen rectangle of one cell.
pub fn cell_rect(grid: &Grid, row: u32, column: u32, image_rect: Rect, scale: f32) -> Rect {
    let left = grid.origin_x + grid.cell_width * column as f32;
    let top = grid.origin_y + grid.cell_height * row as f32;
    Rect::from_min_size(
        image_rect.min + vec2(left * scale, top * scale),
        vec2(grid.cell_width * scale, grid.cell_height * scale),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dragging_top_left_keeps_the_opposite_corner() {
        let grid = Grid::new(0.0, 0.0, 10.0, 10.0, 4, 5);
        let resized = resize(grid, Handle::Corner(Corner::TopLeft), pos2(5.0, 4.0));

        assert_eq!((resized.origin_x, resized.origin_y), (5.0, 4.0));
        assert!((resized.cell_width - 9.0).abs() < 1e-6);
        assert!((resized.cell_height - 9.0).abs() < 1e-6);
        assert!((resized.origin_x + resized.width() - 50.0).abs() < 1e-6);
        assert!((resized.origin_y + resized.height() - 40.0).abs() < 1e-6);
    }

    #[test]
    fn dragging_bottom_right_keeps_the_origin() {
        let grid = Grid::new(0.0, 0.0, 10.0, 10.0, 4, 5);
        let resized = resize(grid, Handle::Corner(Corner::BottomRight), pos2(60.0, 50.0));

        assert_eq!((resized.origin_x, resized.origin_y), (0.0, 0.0));
        assert!((resized.cell_width - 12.0).abs() < 1e-6);
        assert!((resized.cell_height - 12.5).abs() < 1e-6);
    }

    #[test]
    fn dragging_top_right_keeps_the_left_edge() {
        let grid = Grid::new(0.0, 0.0, 10.0, 10.0, 4, 5);
        let resized = resize(grid, Handle::Corner(Corner::TopRight), pos2(30.0, 10.0));

        assert_eq!((resized.origin_x, resized.origin_y), (0.0, 10.0));
        assert!((resized.cell_width - 6.0).abs() < 1e-6);
        assert!((resized.cell_height - 7.5).abs() < 1e-6);
    }

    #[test]
    fn ignores_collapsed_extents() {
        let grid = Grid::new(0.0, 0.0, 10.0, 10.0, 4, 5);
        assert_eq!(
            resize(grid, Handle::Corner(Corner::TopLeft), pos2(49.0, 39.0)),
            grid
        );
    }

    #[test]
    fn dragging_an_edge_moves_only_that_edge() {
        let grid = Grid::new(0.0, 0.0, 10.0, 10.0, 4, 5);

        let left = resize(grid, Handle::Edge(Edge::Left), pos2(5.0, 999.0));
        assert_eq!((left.origin_x, left.origin_y), (5.0, 0.0));
        assert!((left.cell_width - 9.0).abs() < 1e-6, "{left:?}");
        assert!(
            (left.cell_height - 10.0).abs() < 1e-6,
            "height is untouched"
        );
        assert!((left.height() - grid.height()).abs() < 1e-6);

        let top = resize(grid, Handle::Edge(Edge::Top), pos2(-999.0, 4.0));
        assert_eq!((top.origin_x, top.origin_y), (0.0, 4.0));
        assert!((top.cell_height - 9.0).abs() < 1e-6, "{top:?}");
        assert!((top.cell_width - 10.0).abs() < 1e-6, "width is untouched");

        let right = resize(grid, Handle::Edge(Edge::Right), pos2(60.0, -999.0));
        assert_eq!((right.origin_x, right.origin_y), (0.0, 0.0));
        assert!((right.cell_width - 12.0).abs() < 1e-6, "{right:?}");
        assert!((right.cell_height - 10.0).abs() < 1e-6);

        let bottom = resize(grid, Handle::Edge(Edge::Bottom), pos2(-999.0, 50.0));
        assert_eq!((bottom.origin_x, bottom.origin_y), (0.0, 0.0));
        assert!((bottom.cell_height - 12.5).abs() < 1e-6, "{bottom:?}");
        assert!((bottom.cell_width - 10.0).abs() < 1e-6);
    }

    #[test]
    fn edge_hit_test_prefers_the_corner_and_keeps_a_small_radius() {
        let grid = Grid::new(10.0, 20.0, 30.0, 30.0, 5, 5);
        let image_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(200.0, 200.0));
        let outline = outline_rect(&grid, image_rect, 1.0);
        let middle_of_top = pos2(outline.center().x, outline.min.y);

        assert_eq!(
            handle_at(&grid, middle_of_top, image_rect, 1.0),
            Some(Handle::Edge(Edge::Top))
        );
        assert_eq!(
            handle_at(
                &grid,
                pos2(outline.min.x, outline.center().y),
                image_rect,
                1.0
            ),
            Some(Handle::Edge(Edge::Left))
        );
        // Just inside the outline is a cell to paint, not a handle to drag.
        assert_eq!(
            handle_at(
                &grid,
                middle_of_top + vec2(0.0, EDGE_RADIUS + 2.0),
                image_rect,
                1.0
            ),
            None
        );
        // A corner still wins over the edges meeting there.
        assert_eq!(
            handle_at(&grid, outline.min, image_rect, 1.0),
            Some(Handle::Corner(Corner::TopLeft))
        );
        // Far away from the outline nothing reacts.
        assert_eq!(handle_at(&grid, pos2(100.0, 100.0), image_rect, 1.0), None);
    }

    #[test]
    fn corner_hit_test_uses_the_handle_radius() {
        let grid = Grid::new(10.0, 20.0, 30.0, 30.0, 5, 5);
        let image_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(200.0, 200.0));
        let corner = corner_points(&grid, image_rect, 1.0)[0].1;

        assert_eq!(
            corner_at(&grid, corner + vec2(6.0, 6.0), image_rect, 1.0),
            Some(Corner::TopLeft)
        );
        assert_eq!(
            corner_at(&grid, corner + vec2(25.0, 25.0), image_rect, 1.0),
            None
        );
    }

    #[test]
    fn snapped_grid_spans_the_region() {
        let region = PixelRect::new(1, 170, 684, 396);
        let detected = Grid::new(5.0, 174.0, 35.6, 35.3, 11, 19);
        let grid = snap_to_region(region, detected).expect("square cells");

        assert_eq!((grid.origin_x, grid.origin_y), (1.0, 170.0));
        assert_eq!((grid.rows, grid.columns), (11, 19));
        assert!((grid.cell_width - 36.0).abs() < 1e-6);
        assert!((grid.cell_height - 36.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_detected_grids_with_wrong_aspect() {
        let region = PixelRect::new(0, 0, 720, 440);
        let detected = Grid::new(5.0, 174.0, 72.0, 22.0, 20, 10);
        assert!(snap_to_region(region, detected).is_none());
    }

    #[test]
    fn covering_fills_the_dragged_rectangle_with_square_cells() {
        let region = PixelRect::new(120, 80, 640, 480);
        let grid = covering(region);

        assert_eq!((grid.origin_x, grid.origin_y), (120.0, 80.0));
        assert!((grid.width() - 640.0).abs() < 1e-3, "{grid:?}");
        assert!((grid.height() - 480.0).abs() < 1e-3, "{grid:?}");
        assert!(
            (grid.cell_width / grid.cell_height - 1.0).abs() < 0.15,
            "{grid:?}"
        );
        assert!(grid.rows >= 2 && grid.columns >= 2, "{grid:?}");
    }

    #[test]
    fn covering_never_asks_for_more_cells_than_a_board_has() {
        // A bracket dragged across a whole screen is very lopsided: the cells
        // have to grow, not the grid get hundreds of columns wide.
        for region in [
            PixelRect::new(0, 0, 2560, 200),
            PixelRect::new(0, 0, 200, 1392),
            PixelRect::new(0, 0, 2560, 1392),
        ] {
            let grid = covering(region);
            assert!(
                grid.rows <= MAX_BOARD_AXIS && grid.columns <= MAX_BOARD_AXIS,
                "{region:?}: {grid:?}"
            );
            assert!(
                (grid.width() - region.width as f32).abs() < 1e-3,
                "{grid:?}"
            );
            assert!(
                (grid.height() - region.height as f32).abs() < 1e-3,
                "{grid:?}"
            );
        }
    }

    #[test]
    fn covering_survives_degenerate_regions() {
        for region in [
            PixelRect::new(0, 0, 1, 1),
            PixelRect::new(0, 0, 3, 4000),
            PixelRect::new(7, 9, 4000, 3),
        ] {
            let grid = covering(region);
            assert!(grid.rows >= 1 && grid.columns >= 1, "{region:?}: {grid:?}");
            assert!(
                grid.cell_width >= 1.0 && grid.cell_height >= 1.0,
                "{grid:?}"
            );
        }
    }

    #[test]
    fn a_drag_becomes_a_region_whichever_way_it_is_drawn() {
        let size = vec2(800.0, 600.0);
        let down_right = region_between(pos2(100.0, 50.0), pos2(500.0, 350.0), size);
        let up_left = region_between(pos2(500.0, 350.0), pos2(100.0, 50.0), size);

        assert_eq!(down_right, Some(PixelRect::new(100, 50, 400, 300)));
        assert_eq!(down_right, up_left);
    }

    #[test]
    fn a_drag_is_clamped_to_the_image_and_ignores_stray_clicks() {
        let size = vec2(800.0, 600.0);

        assert_eq!(
            region_between(pos2(-40.0, -20.0), pos2(400.0, 300.0), size),
            Some(PixelRect::new(0, 0, 400, 300))
        );
        assert_eq!(
            region_between(pos2(700.0, 500.0), pos2(900.0, 700.0), size),
            Some(PixelRect::new(700, 500, 100, 100))
        );
        assert_eq!(
            region_between(pos2(10.0, 10.0), pos2(18.0, 20.0), size),
            None
        );
    }

    #[test]
    fn region_is_clamped_to_the_image() {
        let grid = Grid::new(-5.0, -3.0, 10.0, 10.0, 4, 5);
        assert_eq!(region(grid, 100, 80), PixelRect::new(0, 0, 45, 37));

        let grid = Grid::new(90.0, 70.0, 10.0, 10.0, 4, 5);
        assert_eq!(region(grid, 100, 80), PixelRect::new(90, 70, 10, 10));
    }
}
