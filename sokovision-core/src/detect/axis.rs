//! Fitting one grid axis to the peaks of an edge projection.

use image::GrayImage;

use super::{MAX_PITCH, MIN_LINES, MIN_PITCH, SAMPLE_RADIUS};
use crate::PixelRect;

/// A fitted grid axis: where the first line sits, how far apart lines are and
/// how many cells they enclose.
#[derive(Debug, Clone, Copy)]
pub(super) struct AxisFit {
    pub origin: f32,
    pub pitch: f32,
    pub cells: u32,
}

/// Fits one axis: find the pitch, then the phase whose lines sit on peaks.
pub(super) fn fit_axis(profile: &[f32], axis_length: u32) -> Option<AxisFit> {
    if profile.len() < MIN_PITCH as usize * 2 {
        return None;
    }
    let mean = profile.iter().sum::<f32>() / profile.len() as f32;
    if mean <= f32::EPSILON {
        return None;
    }
    let max_pitch = MAX_PITCH.min(axis_length / 3).max(MIN_PITCH);

    let peaks = find_peaks(profile, mean);
    if peaks.len() < MIN_LINES {
        return None;
    }
    let pitch = estimate_pitch(&peaks, max_pitch)?;

    // Harmonics of the true pitch look periodic too, so the divisors are tried
    // as well and the best scoring one wins. Half-integer pitches arise from
    // 2x harmonics (e.g. 125% DPI scaling), whereas 3x/4x stay integer-stepped
    // so diagonal stripe textures do not synthesize spurious fractional fits.
    let mut candidates = vec![pitch, pitch * 0.5];
    candidates.extend((3..=4).map(|divisor| (pitch.round() as u32 / divisor) as f32));
    candidates.retain(|candidate| *candidate >= MIN_PITCH as f32);
    candidates.sort_by(f32::total_cmp);
    candidates.dedup_by(|left, right| (*left - *right).abs() < 0.12);

    candidates
        .into_iter()
        .filter_map(|pitch| {
            let (phase, score) = best_phase(profile, pitch)?;
            if score <= 0.0 {
                return None;
            }
            fit_run(profile, pitch, phase, mean)
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(fit, _)| fit)
}

/// Refits one axis at a known pitch, searching the phase around `anchor`.
fn fit_axis_at_pitch(profile: &[f32], pitch: f32, anchor: f32) -> Option<(AxisFit, f32)> {
    let mean = profile.iter().sum::<f32>() / profile.len() as f32;
    if mean <= f32::EPSILON || pitch < MIN_PITCH as f32 {
        return None;
    }
    let period = pitch.round().max(1.0);
    let base_phase = anchor.rem_euclid(period);
    [-1.0f32, 0.0, 1.0]
        .into_iter()
        .filter_map(|delta| {
            let phase = (base_phase + delta).rem_euclid(period);
            fit_run(profile, pitch, phase, mean)
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
}

/// The phase whose sample points land on the strongest lines on average.
fn best_phase(profile: &[f32], pitch: f32) -> Option<(f32, f32)> {
    let steps = pitch.ceil().max(1.0) as u32;
    (0..steps)
        .filter_map(|step| {
            let phase = step as f32;
            if phase >= pitch {
                return None;
            }
            let mut sum = 0.0;
            let mut count = 0usize;
            loop {
                let position = (phase + count as f32 * pitch).round() as usize;
                if position >= profile.len() {
                    break;
                }
                sum += local_peak(profile, position as u32);
                count += 1;
            }
            (count >= MIN_LINES).then(|| (phase, sum / count as f32))
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
}

/// Scores one (pitch, phase) pair: locates the longest run of lines, refines it
/// with a linear fit and returns the fit with a quality value.
fn fit_run(profile: &[f32], pitch: f32, phase: f32, global_mean: f32) -> Option<(AxisFit, f32)> {
    if pitch < MIN_PITCH as f32 || phase >= profile.len() as f32 {
        return None;
    }
    let slots = ((profile.len() as f32 - phase) / pitch).ceil() as usize;
    if slots < MIN_LINES {
        return None;
    }

    let mut positions = Vec::with_capacity(slots);
    let mut values = Vec::with_capacity(slots);
    for slot in 0..slots {
        let expected = (phase + slot as f32 * pitch).round() as usize;
        if expected >= profile.len() {
            break;
        }
        let (position, value) = local_peak_at(profile, expected as u32);
        positions.push(position as f32);
        values.push(value);
    }
    if positions.len() < MIN_LINES {
        return None;
    }

    let peak = values.iter().copied().fold(0.0f32, f32::max);
    if peak <= f32::EPSILON {
        return None;
    }
    let threshold = (global_mean * 1.8).max(peak * 0.3);
    let strong: Vec<bool> = values.iter().map(|value| *value >= threshold).collect();

    let (mut start, mut end, _) = runs(&strong, &values)
        .into_iter()
        .filter(|(start, end, _)| end - start + 1 >= MIN_LINES)
        .max_by(|left, right| {
            (left.1 - left.0)
                .cmp(&(right.1 - right.0))
                .then_with(|| left.2.total_cmp(&right.2))
        })?;

    // Small boards can have boundary lines that are weaker than the brick or
    // floor texture. If the strong run covers less than half of the axis, grow
    // it over neighbouring slots while at least two consecutive lines remain
    // above the profile average, so a few weak lines do not truncate the grid.
    if (end - start) as f32 * pitch < profile.len() as f32 * 0.5 {
        let extension = (global_mean * 1.15).max(peak * 0.15);
        while start >= 2 && values[start - 1] >= extension && values[start - 2] >= extension {
            start -= 1;
        }
        if start == 1 && values[0] >= extension {
            start = 0;
        }
        while end + 1 < values.len()
            && values[end + 1] >= extension
            && (end + 2 >= values.len() || values[end + 2] >= extension)
        {
            end += 1;
        }
    }

    let count = end - start + 1;
    let (slope, intercept) = line_fit(&positions[start..=end], pitch);
    if !slope.is_finite() || !(MIN_PITCH as f32 * 0.75..=MAX_PITCH as f32 * 1.25).contains(&slope) {
        return None;
    }

    let strong_values: Vec<f32> = strong[start..=end]
        .iter()
        .zip(&values[start..=end])
        .filter(|(is_strong, _)| **is_strong)
        .map(|(_, value)| *value)
        .collect();
    if strong_values.len() < MIN_LINES {
        return None;
    }
    let average = strong_values.iter().sum::<f32>() / strong_values.len() as f32;
    let coverage = strong_values.len() as f32 / count as f32;
    let quality = strong_values.len() as f32 * average * coverage / global_mean;

    Some((
        AxisFit {
            origin: intercept,
            pitch: slope,
            cells: (end - start) as u32,
        },
        quality,
    ))
}

/// Groups strong slots into runs, tolerating one weak slot inside a run because
/// a sprite can hide a single grid line.
fn runs(strong: &[bool], values: &[f32]) -> Vec<(usize, usize, f32)> {
    let mut runs = Vec::new();
    let mut index = 0;
    while index < strong.len() {
        if !strong[index] {
            index += 1;
            continue;
        }
        let start = index;
        let mut end = index;
        while end + 1 < strong.len() {
            if strong[end + 1] {
                end += 1;
            } else if end + 2 < strong.len() && strong[end + 2] {
                end += 2;
            } else {
                break;
            }
        }
        runs.push((start, end, values[start..=end].iter().sum()));
        index = end + 1;
    }
    runs
}

fn find_peaks(profile: &[f32], mean: f32) -> Vec<(f32, f32)> {
    let threshold = mean * 1.6;
    let mut peaks: Vec<(f32, f32)> = Vec::new();
    for index in 1..profile.len().saturating_sub(1) {
        let value = profile[index];
        if value < threshold || value < profile[index - 1] || value < profile[index + 1] {
            continue;
        }
        match peaks.last_mut() {
            Some(last) if index as f32 - last.0 <= 3.0 => {
                if value > last.1 {
                    *last = (index as f32, value);
                }
            }
            _ => peaks.push((index as f32, value)),
        }
    }
    peaks
}

/// Picks the pitch that most pairs of peaks agree on, weighted by peak strength.
fn estimate_pitch(peaks: &[(f32, f32)], max_pitch: u32) -> Option<f32> {
    let mut histogram = vec![0.0f32; max_pitch as usize + 2];
    for (index, left) in peaks.iter().enumerate() {
        for right in &peaks[index + 1..] {
            let difference = (right.0 - left.0).round() as i32;
            if difference < MIN_PITCH as i32 || difference > max_pitch as i32 {
                continue;
            }
            histogram[difference as usize] += left.1.min(right.1).max(1.0);
        }
    }

    let score_at = |pitch: u32| {
        let lower = pitch.saturating_sub(1) as usize;
        let upper = (pitch + 1).min(max_pitch) as usize;
        histogram[lower..=upper].iter().sum::<f32>()
    };
    let best = (MIN_PITCH..=max_pitch)
        .max_by(|left, right| score_at(*left).total_cmp(&score_at(*right)))?;
    if score_at(best) <= 0.0 {
        return None;
    }

    let lower = best.saturating_sub(1) as usize;
    let upper = (best + 1).min(max_pitch) as usize;
    let mut weighted = 0.0;
    let mut weight = 0.0;
    for (pitch, value) in histogram.iter().enumerate().take(upper + 1).skip(lower) {
        weighted += pitch as f32 * value;
        weight += value;
    }
    let refined = if weight > 0.0 {
        weighted / weight
    } else {
        best as f32
    };
    Some(refined.clamp(MIN_PITCH as f32, max_pitch as f32))
}

/// Strongest value within `SAMPLE_RADIUS` of `position`.
fn local_peak(profile: &[f32], position: u32) -> f32 {
    local_peak_at(profile, position).1
}

/// Strongest value within `SAMPLE_RADIUS` of `position`, with its index.
fn local_peak_at(profile: &[f32], position: u32) -> (u32, f32) {
    let last = profile.len().saturating_sub(1) as u32;
    let start = position.saturating_sub(SAMPLE_RADIUS) as usize;
    let end = (position + SAMPLE_RADIUS).min(last) as usize;
    let mut best = (position.min(last), f32::NEG_INFINITY);
    for (index, value) in profile.iter().enumerate().take(end + 1).skip(start) {
        if *value > best.1 {
            best = (index as u32, *value);
        }
    }
    best
}

/// Least squares fit of sample position against slot index. `fallback` is the
/// slope used when a single sample cannot define one.
fn line_fit(positions: &[f32], fallback: f32) -> (f32, f32) {
    let count = positions.len() as f32;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    for (slot, position) in positions.iter().enumerate() {
        let x = slot as f32;
        sum_x += x;
        sum_y += position;
        sum_xy += x * position;
        sum_xx += x * x;
    }
    let denominator = count * sum_xx - sum_x * sum_x;
    let slope = if denominator.abs() > f32::EPSILON {
        (count * sum_xy - sum_x * sum_y) / denominator
    } else {
        fallback
    };
    (slope, (sum_y - slope * sum_x) / count)
}

/// Absolute horizontal and vertical gradient of every pixel, averaged over the
/// perpendicular axis.
pub(super) fn projections(image: &GrayImage, region: PixelRect) -> (Vec<f32>, Vec<f32>) {
    let (left, top) = (region.x, region.y);
    let (right, bottom) = (region.right(), region.bottom());
    let mut columns = vec![0.0; region.width as usize];
    let mut rows = vec![0.0; region.height as usize];

    for x in left.saturating_add(1)..right.saturating_sub(1) {
        let mut sum = 0u32;
        for y in top..bottom {
            sum += gradient(image.get_pixel(x - 1, y)[0], image.get_pixel(x + 1, y)[0]);
        }
        columns[(x - left) as usize] = sum as f32 / region.height.max(1) as f32;
    }
    for y in top.saturating_add(1)..bottom.saturating_sub(1) {
        let mut sum = 0u32;
        for x in left..right {
            sum += gradient(image.get_pixel(x, y - 1)[0], image.get_pixel(x, y + 1)[0]);
        }
        rows[(y - top) as usize] = sum as f32 / region.width.max(1) as f32;
    }
    (columns, rows)
}

/// Absolute difference of two levels, capped so one sharp edge cannot dominate
/// the projection.
fn gradient(left: u8, right: u8) -> u32 {
    left.abs_diff(right).min(96) as u32
}

/// Cells are square in every known skin. When the two axes disagree, refit the
/// odd one at the pitch of the other; very small pitches on dense grids are
/// halves of a textured sprite (e.g. dithered brick courses) and are doubled.
pub(super) fn reconcile_aspect(
    horizontal: &mut AxisFit,
    vertical: &mut AxisFit,
    column_profile: &[f32],
    row_profile: &[f32],
) -> bool {
    let smaller = horizontal.pitch.min(vertical.pitch);
    let larger = horizontal.pitch.max(vertical.pitch);
    let subharmonic = larger < 20.0 && column_profile.len() >= 360 && row_profile.len() >= 360;

    let (target, refit_vertical) = if larger / smaller > 1.35 {
        (larger, horizontal.pitch > vertical.pitch)
    } else if subharmonic {
        (larger * 2.0, false)
    } else {
        return false;
    };
    let target = target.clamp(MIN_PITCH as f32, MAX_PITCH as f32);

    if subharmonic {
        let fit_harmonic = |profile: &[f32], pitch: f32| {
            let mean = profile.iter().sum::<f32>() / profile.len() as f32;
            let (phase, score) = best_phase(profile, pitch)?;
            (score > 0.0)
                .then(|| fit_run(profile, pitch, phase, mean))
                .flatten()
        };
        let best = [2.0f32, 3.0]
            .into_iter()
            .filter_map(|factor| {
                let pitch = (larger * factor).clamp(MIN_PITCH as f32, MAX_PITCH as f32);
                let (h_fit, h_score) = fit_harmonic(column_profile, pitch)?;
                let (v_fit, v_score) = fit_harmonic(row_profile, pitch)?;
                (h_fit.cells >= 3 && v_fit.cells >= 3).then_some((h_fit, v_fit, h_score + v_score))
            })
            .max_by(|left, right| left.2.total_cmp(&right.2));
        let Some((horizontal_fit, vertical_fit, _)) = best else {
            return false;
        };
        *horizontal = horizontal_fit;
        *vertical = vertical_fit;
        return true;
    }

    let (profile, axis) = if refit_vertical {
        (row_profile, vertical)
    } else {
        (column_profile, horizontal)
    };
    let Some((fit, _)) = fit_axis_at_pitch(profile, target, axis.origin) else {
        return false;
    };
    *axis = fit;
    true
}
