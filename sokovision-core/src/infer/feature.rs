//! Per-cell image features.

use image::imageops::{self, FilterType};
use image::{GenericImageView, RgbaImage};

use crate::PixelRect;

/// Side length of the normalised tile image.
const SIZE: usize = 24;
/// Fraction of the tile dropped on every side, so sprite borders do not leak in.
const INSET: f32 = 0.08;
/// Width of the border band that describes the surface a sprite is drawn on.
const BORDER: usize = 4;
/// Half-open centre window, where goal dots and player sprites live.
const CENTER: std::ops::Range<usize> = 8..16;
/// Slightly wider interior window for localized dot peaks, so a 2–3 px grid
/// phase offset does not push a small goal dot into the dead zone between
/// `BORDER` and `CENTER`.
const PEAK_WINDOW: std::ops::Range<usize> = 6..18;

/// Everything the classifier needs to know about one cell image.
pub(crate) struct Feature {
    /// Normalised RGB of the whole tile, row-major.
    pub color: Vec<f32>,
    pub mean: [f32; 3],
    /// Mean colour of the tile border: the floor the sprite sits on.
    pub border: [f32; 3],
    /// Centre versus border colour difference.
    pub marker: f32,
    /// Mean centre deviation from the tile mean.
    pub center: f32,
    /// Mean luminance gradient, i.e. how textured the tile is.
    pub edge: f32,
    /// Strongest centre deviation: a small dot in an otherwise flat tile.
    pub peak: f32,
    /// Share of centre pixels that deviate from the border.
    pub sparse: f32,
}

impl Feature {
    pub(crate) fn usable_size(width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
        let inset_x = (width as f32 * INSET).round() as u32;
        let inset_y = (height as f32 * INSET).round() as u32;
        let inner_width = width.checked_sub(inset_x * 2)?;
        let inner_height = height.checked_sub(inset_y * 2)?;
        (inner_width >= 2 && inner_height >= 2).then_some((
            inset_x,
            inset_y,
            inner_width,
            inner_height,
        ))
    }

    pub(crate) fn extract_rect(image: &RgbaImage, rect: PixelRect) -> Option<Self> {
        let (inset_x, inset_y, width, height) = Self::usable_size(rect.width, rect.height)?;
        let cropped = imageops::crop_imm(image, rect.x + inset_x, rect.y + inset_y, width, height);
        Self::from_cropped(&*cropped)
    }

    fn from_cropped<I>(cropped: &I) -> Option<Self>
    where
        I: GenericImageView<Pixel = image::Rgba<u8>>,
    {
        let resized = imageops::resize(cropped, SIZE as u32, SIZE as u32, FilterType::Triangle);

        let mut color = Vec::with_capacity(SIZE * SIZE * 3);
        let mut luminance = [0.0f32; SIZE * SIZE];
        let mut mean = [0.0f32; 3];
        for (index, pixel) in resized.pixels().enumerate() {
            let [red, green, blue, _] = pixel.0;
            let values = [
                red as f32 / 255.0,
                green as f32 / 255.0,
                blue as f32 / 255.0,
            ];
            color.extend_from_slice(&values);
            luminance[index] = 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2];
            for (total, value) in mean.iter_mut().zip(values) {
                *total += value;
            }
        }
        let pixels = (SIZE * SIZE) as f32;
        for total in &mut mean {
            *total /= pixels;
        }

        let mut edge = 0.0f32;
        for y in 1..SIZE - 1 {
            for x in 1..SIZE - 1 {
                let index = y * SIZE + x;
                let dx = luminance[index + 1] - luminance[index - 1];
                let dy = luminance[index + SIZE] - luminance[index - SIZE];
                edge += (dx.abs() + dy.abs()).min(1.0);
            }
        }

        let border = border_mean(&color);
        let center = center_mean(&color);

        let mut marker = 0.0f32;
        for channel in 0..3 {
            marker += (center[channel] - border[channel]).abs();
        }

        let mut deviation = 0.0f32;
        let mut patterned = 0.0f32;
        for y in CENTER {
            for x in CENTER {
                let index = (y * SIZE + x) * 3;
                let mut difference = 0.0f32;
                for channel in 0..3 {
                    let value = color[index + channel];
                    deviation += (value - mean[channel]).abs();
                    difference += (value - border[channel]).abs();
                }
                difference /= 3.0;
                if difference > 0.10 {
                    patterned += 1.0;
                }
            }
        }
        let mut peak = 0.0f32;
        for y in PEAK_WINDOW {
            for x in PEAK_WINDOW {
                let index = (y * SIZE + x) * 3;
                let mut mean_diff = 0.0f32;
                let mut max_diff = 0.0f32;
                for channel in 0..3 {
                    let diff = (color[index + channel] - border[channel]).abs();
                    mean_diff += diff;
                    max_diff = max_diff.max(diff);
                }
                mean_diff /= 3.0;
                peak = peak.max(mean_diff.max(0.6 * max_diff));
            }
        }
        let center_pixels = (CENTER.len() * CENTER.len()) as f32;

        Some(Self {
            color,
            mean,
            border,
            marker: marker / 3.0,
            // Scaled to roughly [0, 0.35] to match `player_score`'s divisor.
            center: deviation / (center_pixels * 9.0),
            edge: edge / pixels,
            peak,
            sparse: patterned / center_pixels,
        })
    }
}

/// Mean colour of the border band.
fn border_mean(color: &[f32]) -> [f32; 3] {
    let mut sum = ColourSum::default();
    for y in 0..BORDER {
        for x in 0..SIZE {
            sum.add(color, x, y);
            sum.add(color, x, SIZE - 1 - y);
        }
    }
    for y in BORDER..SIZE - BORDER {
        for x in 0..BORDER {
            sum.add(color, x, y);
            sum.add(color, SIZE - 1 - x, y);
        }
    }
    sum.mean()
}

/// Mean colour of the centre window.
fn center_mean(color: &[f32]) -> [f32; 3] {
    let mut sum = ColourSum::default();
    for y in CENTER {
        for x in CENTER {
            sum.add(color, x, y);
        }
    }
    sum.mean()
}

#[derive(Default)]
struct ColourSum {
    total: [f32; 3],
    count: f32,
}

impl ColourSum {
    fn add(&mut self, color: &[f32], x: usize, y: usize) {
        let index = (y * SIZE + x) * 3;
        for (total, value) in self.total.iter_mut().zip(&color[index..index + 3]) {
            *total += value;
        }
        self.count += 1.0;
    }

    fn mean(self) -> [f32; 3] {
        self.total.map(|channel| channel / self.count)
    }
}
