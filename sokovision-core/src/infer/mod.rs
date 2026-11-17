//! Visual inference: turn cell pixels into board symbols.
//!
//! Nothing here knows about a particular skin. Cells are grouped by appearance,
//! and the rules of Sokoban (exactly one player, as many boxes as goals) decide
//! what each appearance group means.

mod classify;
mod cluster;
mod feature;
mod set;
mod solve;
mod wall;

use image::RgbaImage;

use crate::{Grid, PixelRect, RecognizedCell, SkinLibrary, SkinMatch};

use self::cluster::Sample;
use self::feature::Feature;

/// Result of reading every cell of a grid.
pub(crate) struct Inference {
    pub cells: Vec<Vec<RecognizedCell>>,
    pub warnings: Vec<String>,
    /// Remembered skin that explained the board, if one matched.
    pub skin: Option<SkinMatch>,
    /// Whether the level rules had to be dropped to produce a visual guess.
    pub relaxed: bool,
    /// Whether walls, usable cells or an enclosed unique player were missing.
    pub incomplete_structure: bool,
}

pub(crate) fn infer(image: &RgbaImage, grid: Grid, skins: Option<&SkinLibrary>) -> Inference {
    let (rows, columns) = (grid.rows as usize, grid.columns as usize);
    let mut cells = vec![vec![RecognizedCell::unknown(); columns]; rows];
    let mut warnings = Vec::new();

    let mut samples = Vec::new();
    let mut valid = vec![vec![false; columns]; rows];
    for (row, valid_row) in valid.iter_mut().enumerate() {
        for (column, valid_cell) in valid_row.iter_mut().enumerate() {
            let Some(rect) = grid.cell_rect(row, column) else {
                continue;
            };
            if rect.right() > image.width() || rect.bottom() > image.height() {
                continue;
            }
            let Some(feature) = Feature::extract_rect(image, rect) else {
                continue;
            };
            samples.push(Sample {
                row,
                column,
                feature,
            });
            *valid_cell = true;
        }
    }

    if samples.is_empty() {
        warnings.push("no usable cells were found inside the grid".to_owned());
        return Inference {
            cells,
            warnings,
            skin: None,
            relaxed: false,
            incomplete_structure: true,
        };
    }

    let clusters = cluster::cluster(&samples, &valid, rows, columns);
    warnings.push(format!(
        "automatic inference grouped the board into {} tile appearances",
        clusters.len()
    ));

    let matched = skins.and_then(|skins| {
        let samples: Vec<(&[f32], Option<f32>)> = clusters
            .iter()
            .map(|cluster| (cluster.color.as_slice(), Some(cluster.peak)))
            .collect();
        skins.match_skin(&samples)
    });
    let (skin, hints) = match matched {
        Some((skin, hints)) => (Some(skin), hints),
        None => (None, vec![None; clusters.len()]),
    };

    let classification = classify::classify(
        &clusters,
        samples.len(),
        &valid,
        rows,
        columns,
        &hints,
        &mut warnings,
    );

    let mut appearance_at = vec![vec![None; columns]; rows];
    for (appearance, cluster) in clusters.iter().enumerate() {
        for &(row, column) in &cluster.cells {
            appearance_at[row][column] = Some(appearance);
        }
    }
    for row in 0..rows {
        for column in 0..columns {
            let Some(appearance) = appearance_at[row][column] else {
                continue;
            };
            cells[row][column] = RecognizedCell {
                cell: classification.classes[appearance],
                confidence: classification.confidence[appearance],
                alternatives: classification.alternatives[appearance].clone(),
                appearance: Some(appearance as u32),
            };
        }
    }

    Inference {
        cells,
        warnings,
        skin,
        relaxed: classification.relaxed,
        incomplete_structure: classification.incomplete_structure,
    }
}

/// Whether `rect` is large enough after border inset to yield a tile feature.
pub(crate) fn usable_tile(rect: PixelRect) -> bool {
    Feature::usable_size(rect.width, rect.height).is_some()
}

/// Normalised tile image and centre peak for `rect`, as stored in a learned skin.
pub(crate) fn tile_prototype_rect(image: &RgbaImage, rect: PixelRect) -> Option<(Vec<f32>, f32)> {
    Feature::extract_rect(image, rect).map(|feature| (feature.color, feature.peak))
}

/// Mean absolute channel difference of two colours, all channels weighted
/// equally.
pub(super) fn mean_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / 3.0
}

/// Mean absolute difference of two normalised tile images, bailing out as soon
/// as the pair cannot be close enough to matter.
pub(super) fn color_distance(left: &[f32], right: &[f32]) -> f32 {
    let length = left.len().min(right.len());
    if length == 0 {
        return 0.0;
    }
    let limit = cluster::CLUSTER_DISTANCE * length as f32;
    let mut difference = 0.0;
    for index in 0..length {
        difference += (left[index] - right[index]).abs();
        if difference > limit {
            break;
        }
    }
    difference / length as f32
}
