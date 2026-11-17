//! Grid detection: find the board inside a screenshot and fit a cell grid.
//!
//! Detection works on an edge projection of the image: grid lines show up as
//! periodic peaks, and the pitch of those peaks gives the cell size. Three
//! strategies run in order and the first one that explains the pixels wins:
//!
//! 1. line fit on the game canvas (the area between the light UI bars),
//! 2. line fit on the whole search region,
//! 3. content box + autocorrelation pitch, for skins that draw no grid lines.

mod axis;
mod content;

use image::{GrayImage, RgbaImage};

use crate::{Error, Grid, MAX_BOARD_AXIS, PixelRect, Result};

use self::axis::{fit_axis, projections, reconcile_aspect};
use self::content::{
    content_grid, fallback_grid, frame_level, game_canvas, grow_to_content, modal_level,
};

/// Smallest and largest cell pitch we accept, in pixels.
pub(super) const MIN_PITCH: u32 = 12;
pub(super) const MAX_PITCH: u32 = 128;
/// Radius used to snap a sample onto the local peak of an edge projection.
const SAMPLE_RADIUS: u32 = 2;
/// A run of grid lines needs at least this many lines to describe an axis.
pub(super) const MIN_LINES: usize = 2;

/// Maps every pixel inside `region` to its distance from the dominant colour.
///
/// Skins can draw walls with almost the same luminance as the floor, so a plain
/// grayscale conversion hides the board; colour distance keeps those edges.
fn contrast_image_in(image: &RgbaImage, region: PixelRect) -> GrayImage {
    let background = dominant_color_in(image, region);
    let stride = image.width() as usize * 4;
    let raw = image.as_raw();
    let width = region.width as usize;
    let height = region.height as usize;
    let mut out = Vec::with_capacity(width * height);
    for y in region.y as usize..(region.y + region.height) as usize {
        let start = y * stride + region.x as usize * 4;
        for &[red, green, blue, _] in raw[start..start + width * 4].as_chunks::<4>().0 {
            let dr = red as f32 - background[0];
            let dg = green as f32 - background[1];
            let db = blue as f32 - background[2];
            let distance = (dr * dr + dg * dg + db * db).sqrt();
            out.push(distance.min(255.0) as u8);
        }
    }
    GrayImage::from_raw(region.width, region.height, out).expect("exact pixel count")
}

#[cfg(test)]
fn dominant_color(image: &RgbaImage) -> [f32; 3] {
    dominant_color_in(image, PixelRect::new(0, 0, image.width(), image.height()))
}

/// Mean colour of the most common 12-bit colour bucket inside `region`.
fn dominant_color_in(image: &RgbaImage, region: PixelRect) -> [f32; 3] {
    let stride = image.width() as usize * 4;
    let raw = image.as_raw();
    let width = region.width as usize;
    let mut histogram = [0u32; 4096];
    for y in region.y as usize..(region.y + region.height) as usize {
        let start = y * stride + region.x as usize * 4;
        for &rgba in raw[start..start + width * 4].as_chunks::<4>().0 {
            histogram[color_bin(rgba)] += 1;
        }
    }
    let Some((dominant, _)) = histogram.iter().enumerate().max_by_key(|(_, n)| **n) else {
        return [0.0; 3];
    };

    let mut sum = [0.0f32; 3];
    let mut count = 0.0f32;
    for y in region.y as usize..(region.y + region.height) as usize {
        let start = y * stride + region.x as usize * 4;
        for &rgba in raw[start..start + width * 4].as_chunks::<4>().0 {
            if color_bin(rgba) == dominant {
                sum[0] += rgba[0] as f32;
                sum[1] += rgba[1] as f32;
                sum[2] += rgba[2] as f32;
                count += 1.0;
            }
        }
    }
    if count == 0.0 {
        return [0.0; 3];
    }
    sum.map(|total| total / count)
}

fn color_bin([red, green, blue, _]: [u8; 4]) -> usize {
    ((red as usize >> 4) << 8) | ((green as usize >> 4) << 4) | (blue as usize >> 4)
}

/// Luminance image of `region`, using the same integer sRGB weights as [`image`].
fn luma_image_in(image: &RgbaImage, region: PixelRect) -> GrayImage {
    let stride = image.width() as usize * 4;
    let raw = image.as_raw();
    let width = region.width as usize;
    let height = region.height as usize;
    let mut out = Vec::with_capacity(width * height);
    for y in region.y as usize..(region.y + region.height) as usize {
        let start = y * stride + region.x as usize * 4;
        for &[red, green, blue, _] in raw[start..start + width * 4].as_chunks::<4>().0 {
            let value = (2126 * red as u32 + 7152 * green as u32 + 722 * blue as u32) / 10_000;
            out.push(value as u8);
        }
    }
    GrayImage::from_raw(region.width, region.height, out).expect("exact pixel count")
}

/// Fits a grid inside `region`. The grid is fitted on the contrast map while
/// the game canvas is found on the original luminance image, where title and
/// status bars are actually light.
#[cfg(test)]
pub(crate) fn detect(contrast: &GrayImage, luma: &GrayImage, region: PixelRect) -> Result<Grid> {
    candidates(contrast, luma, region)?
        .into_iter()
        .next()
        .ok_or_else(|| Error::GridDetection("could not detect a grid".to_owned()))
}

/// Detects grid candidates inside `region`, preprocessing only the pixels of
/// `region` rather than the entire screenshot.
pub(crate) fn candidates_in(image: &RgbaImage, region: PixelRect) -> Result<Vec<Grid>> {
    let whole_image = region.x == 0
        && region.y == 0
        && region.width == image.width()
        && region.height == image.height();
    let letterboxed = whole_image.then(|| trim_letterbox(image, region)).flatten();

    let mut grids = Vec::new();
    if let Some((trimmed, framed_window)) = letterboxed {
        let trimmed_grids = region_candidates(image, trimmed, true);
        let full_grids = region_candidates(image, region, true);
        let (first, second) = if framed_window {
            (trimmed_grids, full_grids)
        } else {
            (full_grids, trimmed_grids)
        };
        for grid in first.into_iter().chain(second) {
            push_candidate(&mut grids, grid);
        }
    } else {
        grids = region_candidates(image, region, whole_image);
    }
    filter_candidates(grids)
}

fn region_candidates(image: &RgbaImage, region: PixelRect, allow_canvas: bool) -> Vec<Grid> {
    let contrast = contrast_image_in(image, region);
    let local = PixelRect::new(0, 0, region.width, region.height);
    let canvas = allow_canvas
        .then(|| {
            let luma = luma_image_in(image, region);
            game_canvas(&luma, local)
        })
        .flatten();
    let mut grids = fit_candidates(&contrast, local, canvas);
    for grid in &mut grids {
        grid.origin_x += region.x as f32;
        grid.origin_y += region.y as f32;
    }
    grids
}

/// Strips a uniform 4-sided outer letterbox/padding from `region`, returning
/// the inner rectangle and whether that inner rectangle begins with a window
/// frame or full-width title bar.
fn trim_letterbox(image: &RgbaImage, region: PixelRect) -> Option<(PixelRect, bool)> {
    let pad_color = uniform_perimeter_color(image, region, 0.88)?;
    let row_uniform = |y: u32| {
        let matching = (region.x..region.right())
            .filter(|&x| close_rgb(image.get_pixel(x, y).0, pad_color, 8))
            .count();
        matching as f32 / region.width.max(1) as f32 >= 0.97
    };
    let col_uniform = |x: u32| {
        let matching = (region.y..region.bottom())
            .filter(|&y| close_rgb(image.get_pixel(x, y).0, pad_color, 8))
            .count();
        matching as f32 / region.height.max(1) as f32 >= 0.97
    };

    let mut top = region.y;
    while top < region.bottom() && row_uniform(top) {
        top += 1;
    }
    let mut bottom = region.bottom();
    while bottom > top && row_uniform(bottom - 1) {
        bottom -= 1;
    }
    let mut left = region.x;
    while left < region.right() && col_uniform(left) {
        left += 1;
    }
    let mut right = region.right();
    while right > left && col_uniform(right - 1) {
        right -= 1;
    }

    let trimmed = PixelRect::new(left, top, right - left, bottom - top);
    let padded_all_sides = top - region.y >= 4
        && region.bottom() - bottom >= 4
        && left - region.x >= 4
        && region.right() - right >= 4;
    if !padded_all_sides || trimmed.width < MIN_PITCH * 4 || trimmed.height < MIN_PITCH * 4 {
        return None;
    }

    let inner_frame = uniform_perimeter_color(image, trimmed, 0.85).is_some();
    let title_bar = uniform_row_color(image, trimmed, trimmed.y, 0.75);
    Some((trimmed, inner_frame || title_bar))
}

fn uniform_perimeter_color(image: &RgbaImage, rect: PixelRect, min_share: f32) -> Option<[u8; 3]> {
    if rect.width < 4 || rect.height < 4 {
        return None;
    }
    let mut histogram = [0u32; 4096];
    let mut total = 0u32;
    let mut sample = |x: u32, y: u32| {
        histogram[color_bin(image.get_pixel(x, y).0)] += 1;
        total += 1;
    };
    for x in rect.x..rect.right() {
        sample(x, rect.y);
        sample(x, rect.bottom() - 1);
    }
    for y in rect.y + 1..rect.bottom() - 1 {
        sample(rect.x, y);
        sample(rect.right() - 1, y);
    }
    let (dominant, _) = histogram.iter().enumerate().max_by_key(|(_, n)| **n)?;
    let mut sum = [0u64; 3];
    let mut count = 0u64;
    for x in rect.x..rect.right() {
        for y in [rect.y, rect.bottom() - 1] {
            let p = image.get_pixel(x, y).0;
            if color_bin(p) == dominant {
                sum[0] += p[0] as u64;
                sum[1] += p[1] as u64;
                sum[2] += p[2] as u64;
                count += 1;
            }
        }
    }
    if count == 0 {
        return None;
    }
    let color = [
        (sum[0] / count) as u8,
        (sum[1] / count) as u8,
        (sum[2] / count) as u8,
    ];
    let mut matching = 0u32;
    for x in rect.x..rect.right() {
        for y in [rect.y, rect.bottom() - 1] {
            if close_rgb(image.get_pixel(x, y).0, color, 10) {
                matching += 1;
            }
        }
    }
    for y in rect.y + 1..rect.bottom() - 1 {
        for x in [rect.x, rect.right() - 1] {
            if close_rgb(image.get_pixel(x, y).0, color, 10) {
                matching += 1;
            }
        }
    }
    (matching as f32 / total.max(1) as f32 >= min_share).then_some(color)
}

fn uniform_row_color(image: &RgbaImage, rect: PixelRect, y: u32, min_share: f32) -> bool {
    let mut histogram = [0u32; 4096];
    for x in rect.x..rect.right() {
        histogram[color_bin(image.get_pixel(x, y).0)] += 1;
    }
    let max_bin = histogram.into_iter().max().unwrap_or(0);
    max_bin as f32 / rect.width.max(1) as f32 >= min_share
}

fn close_rgb(pixel: [u8; 4], reference: [u8; 3], tolerance: u8) -> bool {
    pixel[0].abs_diff(reference[0]) <= tolerance
        && pixel[1].abs_diff(reference[1]) <= tolerance
        && pixel[2].abs_diff(reference[2]) <= tolerance
}

/// Grid candidates ordered by the detector's own preference. The first one is
/// the historical single result; the rest let recognition retry a near miss
/// without re-running detection from scratch.
#[cfg(test)]
pub(crate) fn candidates(
    contrast: &GrayImage,
    luma: &GrayImage,
    region: PixelRect,
) -> Result<Vec<Grid>> {
    filter_candidates(fit_candidates(contrast, region, game_canvas(luma, region)))
}

fn filter_candidates(mut grids: Vec<Grid>) -> Result<Vec<Grid>> {
    if let Some(grid) = grids.first()
        && (grid.rows > MAX_BOARD_AXIS || grid.columns > MAX_BOARD_AXIS)
    {
        return Err(Error::GridDetection(format!(
            "detected a {}x{} grid, denser than the {MAX_BOARD_AXIS}x{MAX_BOARD_AXIS} limit",
            grid.columns, grid.rows
        )));
    }
    grids.retain(|grid| grid.rows <= MAX_BOARD_AXIS && grid.columns <= MAX_BOARD_AXIS);
    // Subharmonic fits (2x/3x density) blow the cell count past any normal
    // puzzle; try board-sized candidates first while keeping dense boards valid.
    grids.sort_by_key(|grid| grid.rows * grid.columns > 600);
    grids.truncate(9);
    if grids.is_empty() {
        return Err(Error::GridDetection("could not detect a grid".to_owned()));
    }
    Ok(grids)
}

#[cfg(test)]
fn fit(image: &GrayImage, region: PixelRect, canvas: Option<PixelRect>) -> Result<Grid> {
    fit_candidates(image, region, canvas)
        .into_iter()
        .next()
        .ok_or_else(|| Error::GridDetection("could not detect a grid".to_owned()))
}

fn fit_candidates(image: &GrayImage, region: PixelRect, canvas: Option<PixelRect>) -> Vec<Grid> {
    let mut candidates = Vec::new();
    let full = fit_lines(image, region);
    let cropped = canvas.map(|canvas| fit_lines(image, canvas));
    let pitch = cropped
        .as_ref()
        .and_then(|result| result.as_ref().ok())
        .or(full.as_ref().ok())
        .map(|grid| (grid.cell_width + grid.cell_height) * 0.5);

    let mut contents: Vec<(Grid, f32)> = Vec::new();
    if let Some(pitch) = pitch {
        let area = canvas.unwrap_or(region);
        for background in [modal_level(image, area), frame_level(image, area)] {
            if let Some(candidate) = content_grid(image, area, pitch, background)
                && !contents.iter().any(|(grid, _)| *grid == candidate.0)
            {
                contents.push(candidate);
            }
        }
        if !contents.is_empty() {
            let fits = [cropped.as_ref(), Some(&full)]
                .into_iter()
                .flatten()
                .filter_map(|result| result.as_ref().ok());
            for line_fit in fits {
                if contents
                    .iter()
                    .any(|(grid, _)| (grid.rows, grid.columns) == (line_fit.rows, line_fit.columns))
                {
                    push_candidate(&mut candidates, *line_fit);
                    break;
                }
            }
            if candidates.is_empty() {
                contents.sort_by(|left, right| left.1.total_cmp(&right.1));
                push_candidate(&mut candidates, contents[0].0);
            }
        }
    }

    if candidates.is_empty() {
        match cropped {
            Some(Ok(grid)) => push_candidate(&mut candidates, grid),
            _ => match full {
                Ok(grid) => push_candidate(&mut candidates, grid),
                Err(_) => {
                    if let Some(grid) = fallback_grid(image, canvas.unwrap_or(region)) {
                        push_candidate(&mut candidates, grid);
                    }
                }
            },
        }
    }

    for (grid, _) in contents {
        push_candidate(&mut candidates, grid);
    }
    if let Some(Ok(grid)) = cropped {
        push_candidate(&mut candidates, grid);
    }
    if let Ok(grid) = full {
        push_candidate(&mut candidates, grid);
    }
    if let Some(grid) = fallback_grid(image, canvas.unwrap_or(region)) {
        push_candidate(&mut candidates, grid);
    }
    candidates
}

fn push_candidate(candidates: &mut Vec<Grid>, grid: Grid) {
    if !candidates.contains(&grid) {
        candidates.push(grid);
    }
}

/// Fits a grid on the grid lines of `region`.
fn fit_lines(image: &GrayImage, region: PixelRect) -> Result<Grid> {
    if region.width < MIN_PITCH * 2 || region.height < MIN_PITCH * 2 {
        return Err(Error::GridDetection(
            "the search region is too small for a grid".to_owned(),
        ));
    }

    let (column_profile, row_profile) = projections(image, region);
    let mut horizontal = fit_axis(&column_profile, region.width)
        .ok_or_else(|| Error::GridDetection("no vertical grid lines found".to_owned()))?;
    let mut vertical = fit_axis(&row_profile, region.height)
        .ok_or_else(|| Error::GridDetection("no horizontal grid lines found".to_owned()))?;
    let reconciled = reconcile_aspect(
        &mut horizontal,
        &mut vertical,
        &column_profile,
        &row_profile,
    );

    let mut grid = Grid::new(
        region.x as f32 + horizontal.origin,
        region.y as f32 + vertical.origin,
        horizontal.pitch,
        vertical.pitch,
        vertical.cells,
        horizontal.cells,
    );
    if reconciled {
        grid = grow_to_content(image, region, grid);
    }
    grid.clamp_to_image(image.width(), image.height())
        .ok_or_else(|| Error::GridDetection("detected grid is outside the image".to_owned()))
}

#[cfg(test)]
mod tests {
    use image::{GrayImage, Luma};

    use super::*;

    fn synthetic_grid(origin: (u32, u32), cell: u32, rows: u32, columns: u32) -> GrayImage {
        let width = columns * cell + 30;
        let height = rows * cell + 30;
        let mut image = GrayImage::from_pixel(width, height, Luma([20]));
        // Texture inside cells makes the image closer to a game screenshot.
        for y in 0..height {
            for x in 0..width {
                if (x / 5 + y / 7) % 2 == 0 {
                    image.put_pixel(x, y, Luma([35]));
                }
            }
        }
        for line in 0..=rows {
            let y = origin.1 + line * cell;
            for x in origin.0..=origin.0 + columns * cell {
                image.put_pixel(x, y, Luma([220]));
            }
        }
        for line in 0..=columns {
            let x = origin.0 + line * cell;
            for y in origin.1..=origin.1 + rows * cell {
                image.put_pixel(x, y, Luma([220]));
            }
        }
        image
    }

    #[test]
    fn detects_offset_grid() {
        let image = synthetic_grid((17, 21), 24, 4, 6);
        let grid = fit(
            &image,
            PixelRect::new(0, 0, image.width(), image.height()),
            None,
        )
        .unwrap();

        assert!((grid.origin_x - 17.0).abs() <= 2.0, "{grid:?}");
        assert!((grid.origin_y - 21.0).abs() <= 2.0, "{grid:?}");
        assert!((grid.cell_width - 24.0).abs() <= 0.75, "{grid:?}");
        assert!((grid.cell_height - 24.0).abs() <= 0.75, "{grid:?}");
        assert_eq!((grid.rows, grid.columns), (4, 6));
    }

    /// A grid drawn on a flat floor, so the line fit locks onto the real pitch
    /// instead of the beat pattern of the textured fixture.
    fn clean_grid(cell: u32, rows: u32, columns: u32) -> GrayImage {
        let origin = 10;
        let mut image = GrayImage::from_pixel(
            columns * cell + origin * 2,
            rows * cell + origin * 2,
            Luma([30]),
        );
        for line in 0..=rows {
            let y = origin + line * cell;
            for x in origin..=origin + columns * cell {
                image.put_pixel(x, y, Luma([220]));
            }
        }
        for line in 0..=columns {
            let x = origin + line * cell;
            for y in origin..=origin + rows * cell {
                image.put_pixel(x, y, Luma([220]));
            }
        }
        image
    }

    #[test]
    fn rejects_a_grid_denser_than_the_limit() {
        // A desktop screenshot full of text and icons looks like a very fine
        // grid to the line fit; the per-cell inference would then crawl.
        let image = clean_grid(20, 20, 120);
        let region = PixelRect::new(0, 0, image.width(), image.height());
        let error = detect(&image, &image, region).expect_err("120 columns is not a board");
        assert!(error.to_string().contains("denser"), "{error}");
        assert!(error.to_string().contains("100x100"), "{error}");

        // The limit is inclusive: 100x100 is still a (very large) board.
        let image = clean_grid(20, 100, 100);
        let region = PixelRect::new(0, 0, image.width(), image.height());
        let grid = detect(&image, &image, region).expect("100x100 is accepted");
        assert_eq!((grid.rows, grid.columns), (100, 100), "{grid:?}");

        // One cell more on either axis is over the line.
        let image = clean_grid(20, 101, 101);
        let region = PixelRect::new(0, 0, image.width(), image.height());
        let error = detect(&image, &image, region).expect_err("101x101 is over the limit");
        assert!(error.to_string().contains("denser"), "{error}");

        // A board-sized grid still goes through untouched.
        let image = clean_grid(24, 10, 15);
        let region = PixelRect::new(0, 0, image.width(), image.height());
        let grid = detect(&image, &image, region).expect("a normal board is accepted");
        assert_eq!((grid.rows, grid.columns), (10, 15), "{grid:?}");
    }

    #[test]
    fn rejects_empty_image() {
        let image = GrayImage::from_pixel(200, 200, Luma([10]));
        let region = PixelRect::new(0, 0, 200, 200);
        assert!(fit(&image, region, None).is_err());
    }

    #[test]
    fn measures_the_background_as_the_dominant_colour() {
        let mut image = RgbaImage::from_pixel(4, 4, image::Rgba([10, 200, 30, 255]));
        image.put_pixel(0, 0, image::Rgba([250, 250, 250, 255]));

        let background = dominant_color(&image);
        assert!((background[1] - 200.0).abs() < 2.0, "{background:?}");
        assert!(background[0] < 20.0, "{background:?}");
    }

    #[test]
    fn detects_fractional_pitch_grid_without_phase_drift() {
        let (origin_x, origin_y, pitch, rows, columns) = (14.0f32, 18.0f32, 32.5f32, 16u32, 20u32);
        let width = (origin_x + columns as f32 * pitch + 20.0).ceil() as u32;
        let height = (origin_y + rows as f32 * pitch + 20.0).ceil() as u32;
        let mut image = GrayImage::from_pixel(width, height, Luma([25]));
        for row in 0..=rows {
            let y = (origin_y + row as f32 * pitch).round() as u32;
            for x in origin_x as u32..=(origin_x + columns as f32 * pitch).round() as u32 {
                image.put_pixel(x, y, Luma([215]));
            }
        }
        for column in 0..=columns {
            let x = (origin_x + column as f32 * pitch).round() as u32;
            for y in origin_y as u32..=(origin_y + rows as f32 * pitch).round() as u32 {
                image.put_pixel(x, y, Luma([215]));
            }
        }

        let grid = fit(&image, PixelRect::new(0, 0, width, height), None).unwrap();
        assert_eq!((grid.rows, grid.columns), (rows, columns), "{grid:?}");
        assert!((grid.cell_width - pitch).abs() <= 0.5, "{grid:?}");
        assert!((grid.cell_height - pitch).abs() <= 0.5, "{grid:?}");
    }

    #[test]
    fn trims_outer_letterbox_around_a_framed_window() {
        let mut image = RgbaImage::from_pixel(240, 200, image::Rgba([210, 210, 210, 255]));
        // Inner dark window frame inside a 15 px light letterbox.
        for y in 15..185 {
            for x in 15..225 {
                image.put_pixel(x, y, image::Rgba([30, 30, 30, 255]));
            }
        }
        for y in 25..175 {
            for x in 25..215 {
                image.put_pixel(x, y, image::Rgba([140, 70, 50, 255]));
            }
        }

        let (trimmed, framed) =
            trim_letterbox(&image, PixelRect::new(0, 0, 240, 200)).expect("letterbox");
        assert_eq!(trimmed, PixelRect::new(15, 15, 210, 170));
        assert!(framed);
    }
}
