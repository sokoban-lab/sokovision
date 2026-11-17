//! Finding the board inside the image and measuring the level it contains.

use std::collections::VecDeque;

use image::GrayImage;

use super::MAX_PITCH;
use super::MIN_PITCH;
use super::axis::projections;
use crate::{Grid, PixelRect};

/// Screenshot chrome (title, toolbars, status bar) adds strong horizontal lines
/// that bias the row phase. When the whole image is searched, detect the game
/// canvas between the light UI bars and fit the grid there instead.
///
/// Only thick light bands count: flat-tile skins draw light grid lines inside
/// the board, and a single such line must not be mistaken for a toolbar.
pub(super) fn game_canvas(image: &GrayImage, region: PixelRect) -> Option<PixelRect> {
    let whole_image = region.x == 0
        && region.y == 0
        && region.width == image.width()
        && region.height == image.height();
    let rows = region.height as usize;
    if !whole_image || rows < MIN_PITCH as usize * 4 {
        return None;
    }

    let mut light = vec![false; rows];
    for (index, is_light) in light.iter_mut().enumerate() {
        let y = region.y + index as u32;
        let count = (region.x..region.right())
            .filter(|&x| image.get_pixel(x, y)[0] > 200)
            .count();
        *is_light = count as f32 / region.width as f32 > 0.5;
    }

    let mut bands: Vec<(usize, usize)> = Vec::new();
    let mut index = 0usize;
    while index < rows {
        if !light[index] {
            index += 1;
            continue;
        }
        let start = index;
        while index < rows && light[index] {
            index += 1;
        }
        match bands.last_mut() {
            Some((_, end)) if start - *end <= 3 => *end = index,
            _ => bands.push((start, index)),
        }
    }

    let thick = |(start, end): &(usize, usize)| end - start >= 8;
    let limit = rows / 5;
    let top = bands
        .iter()
        .filter(|band| thick(band) && band.1 <= limit)
        .map(|band| band.1)
        .max()
        .unwrap_or(0);
    let bottom = bands
        .iter()
        .filter(|band| thick(band) && band.0 >= rows - limit)
        .map(|band| band.0)
        .min()
        .unwrap_or(rows);
    if top == 0 && bottom == rows {
        return None;
    }

    let height = bottom.saturating_sub(top);
    (height >= MIN_PITCH as usize * 4)
        .then(|| PixelRect::new(region.x, region.y + top as u32, region.width, height as u32))
}

/// Most common level in `region`.
pub(super) fn modal_level(image: &GrayImage, region: PixelRect) -> u8 {
    modal_level_of(image, region, |_, _| true)
}

/// Most common level along the border of `region`, i.e. the window frame.
pub(super) fn frame_level(image: &GrayImage, region: PixelRect) -> u8 {
    let band = 6.min(region.width / 4).min(region.height / 4);
    if band == 0 {
        return modal_level(image, region);
    }
    modal_level_of(image, region, |x, y| {
        x < region.x + band
            || x + band >= region.right()
            || y < region.y + band
            || y + band >= region.bottom()
    })
}

fn modal_level_of(image: &GrayImage, region: PixelRect, keep: impl Fn(u32, u32) -> bool) -> u8 {
    let mut histogram = [0u32; 256];
    for y in region.y..region.bottom() {
        for x in region.x..region.right() {
            if keep(x, y) {
                histogram[image.get_pixel(x, y)[0] as usize] += 1;
            }
        }
    }
    histogram
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| **count)
        .map_or(0, |(level, _)| level as u8)
}

/// Mean level of the most common 16-level bucket, used as "the floor colour".
fn background_level(image: &GrayImage, region: PixelRect) -> Option<u8> {
    let mut histogram = [0u32; 16];
    for y in region.y..region.bottom() {
        for x in region.x..region.right() {
            histogram[(image.get_pixel(x, y)[0] / 16) as usize] += 1;
        }
    }
    let total: u32 = histogram.iter().sum();
    let (bin, count) = histogram.iter().enumerate().max_by_key(|(_, n)| **n)?;
    if *count as f32 / (total as f32) < 0.08 {
        return None;
    }

    let (low, high) = ((bin * 16) as u32, (bin * 16 + 15) as u32);
    let mut sum = 0u64;
    let mut samples = 0u64;
    for y in region.y..region.bottom() {
        for x in region.x..region.right() {
            let value = image.get_pixel(x, y)[0] as u32;
            if (low..=high).contains(&value) {
                sum += value as u64;
                samples += 1;
            }
        }
    }
    (samples > 0).then_some((sum / samples) as u8)
}

fn cell_has_content(image: &GrayImage, rect: PixelRect, background: u8) -> bool {
    let mut different = 0u32;
    let mut total = 0u32;
    for y in rect.y..rect.bottom().min(image.height()) {
        for x in rect.x..rect.right().min(image.width()) {
            if image.get_pixel(x, y)[0].abs_diff(background) > 16 {
                different += 1;
            }
            total += 1;
        }
    }
    total > 0 && different as f32 / total as f32 > 0.15
}

/// Flat tile skins often draw no lines between equal neighbours, so the edge
/// projection can only see part of the grid. The level itself is still a large
/// connected block of non-background pixels; its bounding box gives the true
/// row and column counts, which are then combined with the detected pitch.
pub(super) fn content_grid(
    image: &GrayImage,
    region: PixelRect,
    pitch: f32,
    background: u8,
) -> Option<(Grid, f32)> {
    if pitch < MIN_PITCH as f32 {
        return None;
    }
    let (size, bounds) = largest_content_box(image, region, background, 4)?;
    if (size as f32) < (region.width * region.height) as f32 * 0.05 {
        return None;
    }

    let column_fit = bounds.width as f32 / pitch;
    let row_fit = bounds.height as f32 / pitch;
    let columns = (column_fit.round() as u32).max(1);
    let rows = (row_fit.round() as u32).max(1);
    if columns < 2 || rows < 2 {
        return None;
    }

    let cell_width = bounds.width as f32 / columns as f32;
    let cell_height = bounds.height as f32 / rows as f32;
    if !(0.7..=1.3).contains(&(cell_width / pitch)) || !(0.7..=1.3).contains(&(cell_height / pitch))
    {
        return None;
    }
    let error = (column_fit - columns as f32)
        .abs()
        .max((row_fit - rows as f32).abs());

    let grid = Grid::new(
        bounds.x as f32,
        bounds.y as f32,
        cell_width,
        cell_height,
        rows,
        columns,
    );
    grid.validate(image.width(), image.height())
        .ok()
        .map(|grid| (grid, error))
}

/// Fallback for block skins whose walls are drawn as solid slabs: the edge
/// projection has no long run of evenly spaced lines, but the level is still a
/// large block whose cell size appears as a period in the projection.
pub(super) fn fallback_grid(image: &GrayImage, region: PixelRect) -> Option<Grid> {
    let (column_profile, row_profile) = projections(image, region);
    let pitch = correlation_pitch(&column_profile)
        .or_else(|| correlation_pitch(&row_profile))
        .filter(|pitch| (MIN_PITCH..=MAX_PITCH).contains(pitch))?;

    let background = modal_level(image, region);
    let margin = (pitch / 4).clamp(4, 16);
    let (size, bounds) = largest_content_box(image, region, background, margin)?;
    if (size as f32) < (region.width * region.height) as f32 * 0.02 {
        return None;
    }

    let columns = ((bounds.width as f32 / pitch as f32).round() as u32).max(2);
    let rows = ((bounds.height as f32 / pitch as f32).round() as u32).max(2);
    let cell = pitch as f32;
    let grid = Grid::new(
        (bounds.right() as f32 - columns as f32 * cell).max(0.0),
        (bounds.bottom() as f32 - rows as f32 * cell).max(0.0),
        cell,
        cell,
        rows,
        columns,
    );
    if let Ok(grid) = grid.validate(image.width(), image.height()) {
        return Some(grid);
    }

    let grid = Grid::new(
        bounds.x as f32,
        bounds.y as f32,
        bounds.width as f32 / columns as f32,
        bounds.height as f32 / rows as f32,
        rows,
        columns,
    );
    grid.validate(image.width(), image.height()).ok()
}

/// Fundamental period of an edge projection. Harmonics of the true pitch
/// correlate equally well, so the smallest divisor that keeps most of the
/// correlation is preferred.
fn correlation_pitch(profile: &[f32]) -> Option<u32> {
    let mean = profile.iter().sum::<f32>() / profile.len() as f32;
    let centered: Vec<f32> = profile.iter().map(|value| value - mean).collect();
    let energy: f32 = centered.iter().map(|value| value * value).sum();
    if energy <= f32::EPSILON {
        return None;
    }
    let max_lag = (profile.len() / 2).min(MAX_PITCH as usize);
    if max_lag < MIN_PITCH as usize {
        return None;
    }
    let correlation = |lag: usize| {
        centered
            .iter()
            .zip(&centered[lag..])
            .map(|(left, right)| left * right)
            .sum::<f32>()
            / energy
    };

    let best = (MIN_PITCH as usize..=max_lag)
        .max_by(|left, right| correlation(*left).total_cmp(&correlation(*right)))?;
    let best_value = correlation(best);
    let mut pitch = best;
    for divisor in 2..=4 {
        if best % divisor != 0 {
            continue;
        }
        let candidate = best / divisor;
        if candidate >= MIN_PITCH as usize && correlation(candidate) >= 0.6 * best_value {
            pitch = pitch.min(candidate);
        }
    }
    Some(pitch as u32)
}

/// Bounding box of the largest 8-connected region that differs from the
/// background, ignoring a thin border so window frames and status bars do not
/// stretch the level.
fn largest_content_box(
    image: &GrayImage,
    region: PixelRect,
    background: u8,
    margin: u32,
) -> Option<(usize, PixelRect)> {
    let area = region.inset(margin)?;
    if area.width <= MIN_PITCH || area.height <= MIN_PITCH {
        return None;
    }
    let width = area.width as usize;
    let index = |x: u32, y: u32| (y - area.y) as usize * width + (x - area.x) as usize;
    let differs = |x: u32, y: u32| image.get_pixel(x, y)[0].abs_diff(background) > 30;

    let mut visited = vec![false; width * area.height as usize];
    let mut queue = VecDeque::new();
    let mut best: Option<(usize, PixelRect)> = None;

    for y in area.y..area.bottom() {
        for x in area.x..area.right() {
            if visited[index(x, y)] || !differs(x, y) {
                continue;
            }
            let mut size = 0usize;
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (x, y, x, y);
            visited[index(x, y)] = true;
            queue.push_back((x, y));
            while let Some((current_x, current_y)) = queue.pop_front() {
                size += 1;
                min_x = min_x.min(current_x);
                min_y = min_y.min(current_y);
                max_x = max_x.max(current_x);
                max_y = max_y.max(current_y);
                for (next_x, next_y) in neighbours(current_x, current_y, area) {
                    if visited[index(next_x, next_y)] || !differs(next_x, next_y) {
                        continue;
                    }
                    visited[index(next_x, next_y)] = true;
                    queue.push_back((next_x, next_y));
                }
            }
            if best.is_none_or(|(best_size, _)| size > best_size) {
                let bounds = PixelRect::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
                best = Some((size, bounds));
            }
        }
    }
    best
}

/// The eight neighbours of a pixel that still lie inside `area`.
fn neighbours(x: u32, y: u32, area: PixelRect) -> impl Iterator<Item = (u32, u32)> {
    let (center_x, center_y) = (x as i64, y as i64);
    (-1i64..=1)
        .flat_map(move |dy| (-1i64..=1).map(move |dx| (center_x + dx, center_y + dy)))
        .filter(move |&(x, y)| (x, y) != (center_x, center_y))
        .filter(move |&(x, y)| {
            x >= area.x as i64
                && y >= area.y as i64
                && x < area.right() as i64
                && y < area.bottom() as i64
        })
        .map(|(x, y)| (x as u32, y as u32))
}

/// Grows the grid over neighbouring cells that carry content, and drops edge
/// rows and columns that do not.
pub(super) fn grow_to_content(image: &GrayImage, region: PixelRect, initial: Grid) -> Grid {
    let Some(background) = background_level(image, region) else {
        return initial;
    };
    let mut grid = initial;

    let left = grid.origin_x.max(region.x as f32);
    let right = (grid.origin_x + grid.width()).min(region.right() as f32);
    let row_rect = |index: i32| {
        let top = grid.origin_y + grid.cell_height * index as f32;
        let bottom = top + grid.cell_height;
        inside_band(region.y as f32, region.bottom() as f32, top, bottom).then(|| {
            PixelRect::new(
                left.round() as u32,
                top.round() as u32,
                (right - left).round() as u32,
                (bottom - top).round() as u32,
            )
        })
    };
    let (start, end) = grow_axis(image, background, 0, grid.rows as i32, row_rect);
    grid.origin_y += grid.cell_height * start as f32;
    grid.rows = (end - start) as u32;

    let top = grid.origin_y.max(region.y as f32);
    let bottom = (grid.origin_y + grid.height()).min(region.bottom() as f32);
    let column_rect = |index: i32| {
        let left = grid.origin_x + grid.cell_width * index as f32;
        let right = left + grid.cell_width;
        inside_band(region.x as f32, region.right() as f32, left, right).then(|| {
            PixelRect::new(
                left.round() as u32,
                top.round() as u32,
                (right - left).round() as u32,
                (bottom - top).round() as u32,
            )
        })
    };
    let (start, end) = grow_axis(image, background, 0, grid.columns as i32, column_rect);
    grid.origin_x += grid.cell_width * start as f32;
    grid.columns = (end - start) as u32;
    grid
}

fn inside_band(lower: f32, upper: f32, start: f32, end: f32) -> bool {
    start >= lower && end <= upper
}

/// Shrinks `[start, end)` past empty cells at both ends, then extends it over
/// every neighbouring cell that carries content.
fn grow_axis(
    image: &GrayImage,
    background: u8,
    mut start: i32,
    mut end: i32,
    rect_of: impl Fn(i32) -> Option<PixelRect>,
) -> (i32, i32) {
    let filled =
        |index: i32| rect_of(index).is_some_and(|rect| cell_has_content(image, rect, background));
    while end - start > 1 && !filled(start) {
        start += 1;
    }
    while end - start > 1 && !filled(end - 1) {
        end -= 1;
    }
    while filled(start - 1) {
        start -= 1;
    }
    while filled(end) {
        end += 1;
    }
    (start, end)
}
