//! Grouping cells that share one tile image.

use std::collections::VecDeque;

use crate::geometry;

use super::color_distance;
use super::feature::Feature;
use super::mean_distance;
use super::set::DisjointSet;

/// Tiles closer than this in mean colour are compared pixel by pixel at all.
const MEAN_PREFILTER: f32 = 0.30;
/// Mean per-pixel difference below which two tiles are the same appearance.
pub(super) const CLUSTER_DISTANCE: f32 = 0.045;
/// A floor cluster splits when its centre peaks are this far apart...
const PEAK_SPREAD: f32 = 0.12;
/// ...and the flatter half starts at most this flat...
const PEAK_LOW: f32 = 0.08;
/// ...and the busier half reaches at least this high...
const PEAK_HIGH: f32 = 0.16;
/// ...with an empty valley at least this wide between the two modes, so a
/// smooth texture gradient is never cut in half.
const PEAK_VALLEY: f32 = 0.07;

/// One cell, ready for clustering.
pub(crate) struct Sample {
    pub row: usize,
    pub column: usize,
    pub feature: Feature,
}

/// Cells that share one tile image, with the averages the classifier works on.
pub(crate) struct Cluster {
    pub cells: Vec<(usize, usize)>,
    /// Mean normalised tile image, used to match learned skins.
    pub color: Vec<f32>,
    pub border: [f32; 3],
    pub edge: f32,
    pub marker: f32,
    pub center: f32,
    pub peak: f32,
    pub sparse: f32,
    /// How many of the cells touch the board edge.
    pub boundary: usize,
    /// Share of the remaining cells that the cluster cuts off from the border.
    pub enclosure: f32,
}

impl Cluster {
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn boundary_fraction(&self) -> f32 {
        self.boundary as f32 / self.len().max(1) as f32
    }
}

pub(crate) fn cluster(
    samples: &[Sample],
    valid: &[Vec<bool>],
    rows: usize,
    columns: usize,
) -> Vec<Cluster> {
    let mut set = DisjointSet::new(samples.len());
    for (index, sample) in samples.iter().enumerate() {
        for (other, candidate) in samples.iter().enumerate().skip(index + 1) {
            if mean_distance(sample.feature.mean, candidate.feature.mean) >= MEAN_PREFILTER {
                continue;
            }
            if color_distance(&sample.feature.color, &candidate.feature.color) < CLUSTER_DISTANCE {
                set.union(index, other);
            }
        }
    }

    let groups = set
        .groups(0..samples.len())
        .into_iter()
        .flat_map(|group| split_by_peak(&group, samples))
        .collect();

    build_clusters(groups, samples, valid, rows, columns)
}

/// Groups are ordered by size, largest first, so appearance 0 is the floor of
/// most skins.
fn build_clusters(
    mut groups: Vec<Vec<usize>>,
    samples: &[Sample],
    valid: &[Vec<bool>],
    rows: usize,
    columns: usize,
) -> Vec<Cluster> {
    groups.sort_by(|left, right| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| position(left[0], samples).cmp(&position(right[0], samples)))
    });

    let valid_total = samples.len();
    groups
        .into_iter()
        .map(|group| {
            let cells: Vec<(usize, usize)> = group
                .iter()
                .map(|&index| (samples[index].row, samples[index].column))
                .collect();
            let count = group.len() as f32;
            let mut color = vec![0.0f32; samples[group[0]].feature.color.len()];
            let mut border = [0.0f32; 3];
            let (mut edge, mut marker, mut center, mut peak, mut sparse) =
                (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for &index in &group {
                let feature = &samples[index].feature;
                for (total, channel) in color.iter_mut().zip(&feature.color) {
                    *total += channel;
                }
                for (total, channel) in border.iter_mut().zip(feature.border) {
                    *total += channel;
                }
                edge += feature.edge;
                marker += feature.marker;
                center += feature.center;
                peak += feature.peak;
                sparse += feature.sparse;
            }
            let boundary = cells
                .iter()
                .filter(|(row, column)| {
                    *row == 0 || *row + 1 == rows || *column == 0 || *column + 1 == columns
                })
                .count();
            let enclosure = enclosure(&cells, valid, rows, columns, valid_total);

            Cluster {
                color: color.into_iter().map(|total| total / count).collect(),
                border: border.map(|total| total / count),
                edge: edge / count,
                marker: marker / count,
                center: center / count,
                peak: peak / count,
                sparse: sparse / count,
                boundary,
                enclosure,
                cells,
            }
        })
        .collect()
}

fn position(index: usize, samples: &[Sample]) -> (usize, usize) {
    (samples[index].row, samples[index].column)
}

/// A floor cluster can contain the goal markers when they are only a few pixels
/// wide. Their per-cell peak deviation sits above a clear valley over plain
/// floor, whereas JPEG subsampling can spread the marked mode itself; cutting
/// at the widest gap keeps dim goal dots out of the floor half without
/// splitting a continuous texture gradient.
fn split_by_peak(group: &[usize], samples: &[Sample]) -> Vec<Vec<usize>> {
    if group.len() < 2 {
        return vec![group.to_vec()];
    }
    let mut peaks: Vec<f32> = group
        .iter()
        .map(|&index| samples[index].feature.peak)
        .collect();
    peaks.sort_by(f32::total_cmp);
    let (low, high) = (peaks[0], *peaks.last().unwrap());
    if high - low <= PEAK_SPREAD || low > PEAK_LOW || high < PEAK_HIGH {
        return vec![group.to_vec()];
    }

    let Some((split, gap)) = peaks
        .windows(2)
        .enumerate()
        .filter(|(_, pair)| pair[0] <= 0.25)
        .map(|(index, pair)| (index, pair[1] - pair[0]))
        .max_by(|left, right| left.1.total_cmp(&right.1))
    else {
        return vec![group.to_vec()];
    };
    if gap < PEAK_VALLEY {
        return vec![group.to_vec()];
    }

    let threshold = (peaks[split] + peaks[split + 1]) * 0.5;
    let (plain, marked): (Vec<usize>, Vec<usize>) = group
        .iter()
        .copied()
        .partition(|&index| samples[index].feature.peak <= threshold);
    if plain.is_empty() || marked.is_empty() {
        return vec![group.to_vec()];
    }
    vec![plain, marked]
}

/// Share of the cells outside the cluster that the cluster cuts off from the
/// board border. Walls enclose almost everything, floor encloses nothing.
pub(super) fn enclosure(
    cells: &[(usize, usize)],
    valid: &[Vec<bool>],
    rows: usize,
    columns: usize,
    valid_total: usize,
) -> f32 {
    let interior = valid_total.saturating_sub(cells.len());
    if interior == 0 {
        return 0.0;
    }
    let mut blocked = vec![vec![false; columns]; rows];
    for &(row, column) in cells {
        blocked[row][column] = true;
    }

    let mut reached = vec![vec![false; columns]; rows];
    let mut queue = VecDeque::new();
    let border = (0..rows)
        .flat_map(|row| [(row, 0), (row, columns - 1)])
        .chain((0..columns).flat_map(|column| [(0, column), (rows - 1, column)]));
    for (row, column) in border {
        reach(&mut queue, &mut reached, valid, &blocked, row, column);
    }

    let mut reachable = 0usize;
    while let Some((row, column)) = queue.pop_front() {
        reachable += 1;
        for (next_row, next_column) in geometry::neighbours(row, column, rows, columns) {
            reach(
                &mut queue,
                &mut reached,
                valid,
                &blocked,
                next_row,
                next_column,
            );
        }
    }

    (interior - reachable) as f32 / interior as f32
}

/// Marks a cell as reachable from the border and queues it, unless it is a wall,
/// already seen or outside the board.
fn reach(
    queue: &mut VecDeque<(usize, usize)>,
    reached: &mut [Vec<bool>],
    valid: &[Vec<bool>],
    blocked: &[Vec<bool>],
    row: usize,
    column: usize,
) {
    if valid[row][column] && !blocked[row][column] && !reached[row][column] {
        reached[row][column] = true;
        queue.push_back((row, column));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_with_peak(column: usize, peak: f32) -> Sample {
        Sample {
            row: 0,
            column,
            feature: Feature {
                color: vec![0.0; 24 * 24 * 3],
                mean: [0.0; 3],
                border: [0.0; 3],
                marker: 0.0,
                center: 0.0,
                edge: 0.0,
                peak,
                sparse: 0.0,
            },
        }
    }

    #[test]
    fn splits_at_the_valley_between_plain_floor_and_spread_goal_dots() {
        // One goal dot dimmed by JPEG subsampling (0.22) sits well below the
        // midpoint (0.285) of [0.01..0.56], yet above a 0.19 valley over floor.
        let peaks = [0.01, 0.02, 0.03, 0.22, 0.31, 0.40, 0.48, 0.56];
        let samples: Vec<Sample> = peaks
            .into_iter()
            .enumerate()
            .map(|(column, peak)| sample_with_peak(column, peak))
            .collect();
        let group: Vec<usize> = (0..samples.len()).collect();

        assert_eq!(
            split_by_peak(&group, &samples),
            vec![vec![0, 1, 2], vec![3, 4, 5, 6, 7]]
        );
    }

    #[test]
    fn keeps_a_smooth_texture_gradient_as_one_cluster() {
        let peaks = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24];
        let samples: Vec<Sample> = peaks
            .into_iter()
            .enumerate()
            .map(|(column, peak)| sample_with_peak(column, peak))
            .collect();
        let group: Vec<usize> = (0..samples.len()).collect();

        assert_eq!(split_by_peak(&group, &samples), vec![group]);
    }
}
