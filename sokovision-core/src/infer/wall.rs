//! Picking the wall and floor appearances.

use super::cluster::Cluster;
use super::mean_distance;
use super::set::DisjointSet;

/// Score above which an appearance looks like a wall on its own.
const WALL_SCORE: f32 = 0.20;
/// Colour distance, brightness ignored, that still counts as one material.
const MATERIAL_DISTANCE: f32 = 0.08;
/// Marker strength above which an appearance carries goal dots.
const MARKER: f32 = 0.15;

/// Flags the appearances that are walls, and returns whether any wall-like
/// structure was found.
pub(super) fn find(
    clusters: &[Cluster],
    valid: &[Vec<bool>],
    rows: usize,
    columns: usize,
    valid_total: usize,
    warnings: &mut Vec<String>,
) -> (Vec<bool>, bool) {
    let scores: Vec<f32> = clusters
        .iter()
        .map(|cluster| {
            // A one-cell cluster that happens to touch the border would take the
            // whole boundary term, so the term is weighted by how much of the
            // board the appearance actually covers.
            let mass = (cluster.len() as f32 / valid_total.max(1) as f32).sqrt();
            cluster.enclosure * (0.5 + cluster.edge) + 0.35 * cluster.boundary_fraction() * mass
        })
        .collect();
    let best = scores.iter().copied().fold(0.0f32, f32::max);
    let mut is_wall = vec![false; clusters.len()];
    let mut found = true;

    if best >= WALL_SCORE {
        let primary = index_of_max(&scores);
        is_wall[primary] = true;
        let floor = largest(clusters, |_| true);
        for (index, score) in scores.iter().enumerate() {
            if index == primary {
                continue;
            }
            let structural = *score >= 0.5 * best && clusters[index].enclosure >= 0.2;
            // Dithering and sprite phase split one wall material into several
            // appearances. Fragments share the wall colour but enclose nothing
            // on their own, so only the floor (the largest appearance) needs
            // the stricter border/enclosure guard before joining the wall.
            let variant_limit = if clusters[index].boundary > 0 {
                0.03
            } else {
                0.035
            };
            let close =
                mean_distance(clusters[index].border, clusters[primary].border) < variant_limit;
            let guarded = Some(index) == floor;
            let variant = close
                && (!guarded || clusters[index].boundary > 0 || clusters[index].enclosure > 0.05);
            if structural || variant {
                is_wall[index] = true;
            }
        }
    } else if let Some(group) = material_group(clusters, valid, rows, columns, valid_total) {
        for index in group {
            is_wall[index] = true;
        }
    } else {
        warnings.push("no wall-like tile appearance was found".to_owned());
        found = false;
    }
    (is_wall, found)
}

/// Beveled wall sprites shade one material into several appearances. When no
/// single appearance looks wall-like, group appearances by material colour
/// (ignoring brightness) and keep the group whose union separates the inside
/// from the outside.
fn material_group(
    clusters: &[Cluster],
    valid: &[Vec<bool>],
    rows: usize,
    columns: usize,
    valid_total: usize,
) -> Option<Vec<usize>> {
    if clusters.len() < 2 {
        return None;
    }
    let anchor = largest(clusters, |_| true)?;
    let chroma: Vec<[f32; 3]> = clusters
        .iter()
        .map(|cluster| chromaticity(cluster.border))
        .collect();

    let mut set = DisjointSet::new(clusters.len());
    for left in 0..clusters.len() {
        if left == anchor {
            continue;
        }
        for right in left + 1..clusters.len() {
            if right == anchor {
                continue;
            }
            if mean_distance(chroma[left], chroma[right]) < MATERIAL_DISTANCE {
                set.union(left, right);
            }
        }
    }

    set.groups((0..clusters.len()).filter(|&index| index != anchor))
        .into_iter()
        .map(|group| {
            let cells: Vec<(usize, usize)> = group
                .iter()
                .flat_map(|&index| clusters[index].cells.iter().copied())
                .collect();
            let score = super::cluster::enclosure(&cells, valid, rows, columns, valid_total);
            (group, score)
        })
        .filter(|(_, score)| *score >= WALL_SCORE)
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(group, _)| group)
}

/// Cells wedged into a corner of the walls: two perpendicular neighbours are
/// walls, so a box standing there can never be pushed out again.
pub(super) fn corners(wall_at: &[Vec<bool>]) -> Vec<Vec<bool>> {
    let rows = wall_at.len();
    let columns = wall_at.first().map_or(0, Vec::len);
    let wall = |row: isize, column: isize| {
        (0..rows as isize).contains(&row)
            && (0..columns as isize).contains(&column)
            && wall_at[row as usize][column as usize]
    };

    let mut corners = vec![vec![false; columns]; rows];
    for (row, corners_row) in corners.iter_mut().enumerate() {
        for (column, corner) in corners_row.iter_mut().enumerate() {
            if wall_at[row][column] {
                continue;
            }
            let (row, column) = (row as isize, column as isize);
            let (up, down) = (wall(row - 1, column), wall(row + 1, column));
            let (left, right) = (wall(row, column - 1), wall(row, column + 1));
            *corner = (up && left) || (up && right) || (down && left) || (down && right);
        }
    }
    corners
}

/// Colour with the brightness divided out, so two shades of one material match.
fn chromaticity(border: [f32; 3]) -> [f32; 3] {
    let luma = (border[0] + border[1] + border[2]) / 3.0;
    if luma < 0.05 {
        return border;
    }
    border.map(|channel| channel / luma)
}

/// The floor is the appearance that other tiles are drawn on: walls surround
/// it, boxes sit on it, and goal markers are stamped over it. Large levels in
/// flat skins let the floor reach the canvas border, so the most common
/// interior tile can be a box rather than the floor. When that happens, use
/// the goal markers' border colour to point back at the floor appearance.
pub(super) fn floor_reference(
    clusters: &[Cluster],
    is_wall: &[bool],
    is_background: &[bool],
) -> Option<[f32; 3]> {
    let interior = |index: usize| !is_wall[index] && !is_background[index];
    let default = largest(clusters, |index| {
        interior(index) && clusters[index].len() > 1
    });
    let background = largest(clusters, |index| !is_wall[index] && is_background[index]);
    let border = |index: Option<usize>| index.map(|index| clusters[index].border);

    // A small bright sprite on the floor looks exactly like a goal dot: both
    // push the centre average away from the border colour.
    let markers: Vec<[f32; 3]> = (0..clusters.len())
        .filter(|&index| {
            !is_wall[index]
                && clusters[index].len() > 1
                && (clusters[index].marker >= MARKER
                    || (clusters[index].peak >= 0.15 && clusters[index].sparse <= 0.12))
        })
        .map(|index| clusters[index].border)
        .collect();

    if markers.is_empty() {
        return match (default, background) {
            (Some(default), Some(background))
                if clusters[background].len() > clusters[default].len() =>
            {
                border(Some(background))
            }
            (Some(default), _) => border(Some(default)),
            (None, Some(background)) => border(Some(background)),
            (None, None) => None,
        };
    }

    let proximity = |border: [f32; 3]| {
        markers
            .iter()
            .map(|marker| mean_distance(border, *marker))
            .fold(f32::INFINITY, f32::min)
    };
    match (default, background) {
        (Some(default), Some(background))
            if proximity(clusters[background].border) + 0.05
                < proximity(clusters[default].border) =>
        {
            border(Some(background))
        }
        (Some(default), _) => border(Some(default)),
        (None, Some(background)) => border(Some(background)),
        (None, None) => None,
    }
}

/// Index of the largest cluster that satisfies `keep`, ties going to the
/// earlier (and therefore more common) appearance.
fn largest(clusters: &[Cluster], keep: impl Fn(usize) -> bool) -> Option<usize> {
    (0..clusters.len())
        .filter(|&index| keep(index))
        .max_by_key(|&index| clusters[index].len())
}

fn index_of_max(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(rows: &[&str]) -> Vec<Vec<bool>> {
        rows.iter()
            .map(|row| row.chars().map(|ch| ch == '#').collect())
            .collect()
    }

    #[test]
    fn marks_cells_with_two_perpendicular_wall_neighbours() {
        let corners = corners(&mask(&[
            "#######", "#-#-#-#", "#-----#", "###-###", "#######",
        ]));

        assert!(corners[1][1], "walls above and left");
        assert!(corners[1][3], "walls above, left and right");
        assert!(corners[1][5], "walls above and left");
        assert!(corners[2][1], "walls below and left");
        assert!(!corners[2][2], "walls above and below only");
        assert!(!corners[2][3], "flat floor");
        assert!(!corners[0][0], "wall cells are not corners");
        assert!(!corners[3][2], "wall cells stay walls");
    }
}
