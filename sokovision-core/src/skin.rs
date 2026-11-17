//! Remembered tile appearances: a skin library.
//!
//! Recognition works without any knowledge of a particular skin. A library
//! adds what earlier levels taught: when an appearance group looks like a
//! remembered tile, the symbol that tile turned out to be is used as a strong
//! hint, so a skin only has to be worked out once.

use serde::{Deserialize, Serialize};

use crate::Cell;

/// Side length of the normalised tile image produced by recognition.
const TILE_SIZE: usize = 24;
/// Side length the stored prototype is reduced to. Texture phase and rescaling
/// change individual pixels a lot, so matching looks at coarse colour blocks;
/// the centre dot of a goal still moves its block.
const MATCH_SIZE: usize = 4;
/// Values in one stored prototype: RGB per block.
const MATCH_VALUES: usize = MATCH_SIZE * MATCH_SIZE * 3;
/// Mean coarse colour distance under which a tile counts as a match.
const SKIN_DISTANCE: f32 = 0.05;
/// Appearances a skin must explain before it is considered the match.
const MIN_MATCHED: usize = 2;
/// Matches a new skin needs before it is folded into an existing one.
const MIN_OVERLAP: usize = 3;

/// One remembered tile: its coarse image and the symbol it turned out to be.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkinTile {
    pub cell: Cell,
    /// Coarse RGB blocks, row-major: 4×4×3 values.
    pub color: Vec<f32>,
    /// Localised centre peak deviation, so a tiny goal dot is not confused
    /// with plain floor whose 4×4 average is nearly identical.
    #[serde(default)]
    pub peak: Option<f32>,
}

/// A named group of tiles: one game's skin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Skin {
    pub name: String,
    pub tiles: Vec<SkinTile>,
}

/// How well a remembered skin explained a board.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkinMatch {
    pub name: String,
    /// Appearance groups that matched a remembered tile.
    pub matched: usize,
    /// Appearance groups on the board.
    pub appearances: usize,
}

/// Every remembered skin, in the order it was learned.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SkinLibrary {
    pub skins: Vec<Skin>,
}

impl Skin {
    /// Nearest tile within the match distance.
    fn nearest(&self, color: &[f32], peak: Option<f32>) -> Option<(usize, f32)> {
        self.tiles
            .iter()
            .enumerate()
            .filter_map(|(index, tile)| {
                tile_distance(color, peak, tile).map(|distance| (index, distance))
            })
            .filter(|&(_, distance)| distance < SKIN_DISTANCE)
            .min_by(|left, right| left.1.total_cmp(&right.1))
    }

    /// Folds newer observations in, replacing tiles that were seen before. The
    /// newest label wins, so a manual correction repairs an earlier guess.
    /// Returns whether anything was stored differently.
    fn absorb(&mut self, incoming: Skin) -> bool {
        let mut changed = false;
        for tile in incoming.tiles {
            match self.nearest(&tile.color, tile.peak) {
                Some((index, _)) => {
                    if self.tiles[index] != tile {
                        self.tiles[index] = tile;
                        changed = true;
                    }
                }
                None => {
                    self.tiles.push(tile);
                    changed = true;
                }
            }
        }
        changed
    }
}

impl SkinLibrary {
    /// Drops tiles written in an older prototype format. Returns whether
    /// anything was removed.
    pub fn prune(&mut self) -> bool {
        let before: usize = self.skins.iter().map(|skin| skin.tiles.len()).sum();
        for skin in &mut self.skins {
            skin.tiles.retain(|tile| tile.color.len() == MATCH_VALUES);
        }
        self.skins.retain(|skin| !skin.tiles.is_empty());
        let after: usize = self.skins.iter().map(|skin| skin.tiles.len()).sum();
        after != before
    }

    /// Files a learned skin under an existing one when it is clearly the same,
    /// otherwise keeps it as a new skin. Returns whether the library changed,
    /// or `None` when the skin carries no tiles.
    pub fn remember(&mut self, skin: Skin) -> Option<bool> {
        if skin.tiles.is_empty() {
            return None;
        }
        let merged = self
            .skins
            .iter()
            .enumerate()
            .map(|(index, existing)| (index, overlap(existing, &skin)))
            .filter(|&(_, overlap)| {
                overlap.matches >= MIN_OVERLAP
                    && overlap.matches * 2 >= skin.tiles.len()
                    && overlap.content
            })
            .max_by_key(|&(_, overlap)| overlap.matches)
            .map(|(index, _)| index);

        Some(match merged {
            Some(index) => self.skins[index].absorb(skin),
            None => {
                self.skins.push(skin);
                true
            }
        })
    }

    /// The skin that explains the most appearance groups, with the symbol each
    /// group is believed to have. `None` when no skin explains enough of the
    /// board to be trusted.
    pub(crate) fn match_skin(
        &self,
        samples: &[(&[f32], Option<f32>)],
    ) -> Option<(SkinMatch, Vec<Option<Cell>>)> {
        let mut best: Option<(u32, SkinMatch, Vec<Option<Cell>>)> = None;
        for skin in &self.skins {
            let mut hints = Vec::with_capacity(samples.len());
            let mut matched = 0;
            let mut content = 0;
            for &(color, peak) in samples {
                let hint = skin
                    .nearest(color, peak)
                    .map(|(index, _)| skin.tiles[index].cell);
                if let Some(cell) = hint {
                    matched += 1;
                    if !matches!(cell, Cell::Wall | Cell::Empty) {
                        content += 1;
                    }
                }
                hints.push(hint);
            }
            // Explaining only the walls and floor says nothing about the skin;
            // every game has those.
            let usable = matched >= MIN_MATCHED && matched * 2 >= samples.len() && content >= 1;
            if !usable {
                continue;
            }
            let score = content as u32 * 2 + matched as u32;
            if best.as_ref().is_none_or(|(best, ..)| score > *best) {
                best = Some((
                    score,
                    SkinMatch {
                        name: skin.name.clone(),
                        matched,
                        appearances: samples.len(),
                    },
                    hints,
                ));
            }
        }
        best.map(|(_, matched, hints)| (matched, hints))
    }
}

/// How much a new skin overlaps an existing one.
#[derive(Clone, Copy)]
struct Overlap {
    matches: usize,
    /// Whether at least one match is more than wall or floor.
    content: bool,
}

/// Mean colour distance between a tile and a stored prototype, plus a penalty
/// when one tile is flat and the other carries a sharp centre marker.
fn tile_distance(color: &[f32], peak: Option<f32>, prototype: &SkinTile) -> Option<f32> {
    let color_dist = if color.len() == MATCH_VALUES {
        mean_absolute(color, &prototype.color)
    } else {
        mean_absolute(&coarse(color)?, &prototype.color)
    };
    let mark_penalty = match (peak, prototype.peak) {
        (Some(left), Some(right)) => 0.08 * (markedness(left) - markedness(right)).abs(),
        _ => 0.0,
    };
    Some(color_dist + mark_penalty)
}

/// Maps a centre peak into [0, 1] across the flat-to-marked transition so both
/// bright and dimmed markers saturate near 1 while plain floor stays at 0.
fn markedness(peak: f32) -> f32 {
    ((peak - 0.06) / 0.14).clamp(0.0, 1.0)
}

/// Reduces a normalised tile to the 4×4 prototype stored in a skin.
pub(crate) fn coarse(color: &[f32]) -> Option<Vec<f32>> {
    if color.len() != TILE_SIZE * TILE_SIZE * 3 {
        return None;
    }
    let step = TILE_SIZE / MATCH_SIZE;
    let block = (step * step) as f32;
    let mut coarse = Vec::with_capacity(MATCH_VALUES);
    for block_y in 0..MATCH_SIZE {
        for block_x in 0..MATCH_SIZE {
            for channel in 0..3 {
                let mut total = 0.0;
                for y in block_y * step..(block_y + 1) * step {
                    for x in block_x * step..(block_x + 1) * step {
                        total += color[(y * TILE_SIZE + x) * 3 + channel];
                    }
                }
                coarse.push(total / block);
            }
        }
    }
    Some(coarse)
}

fn mean_absolute(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || left.len() != right.len() {
        return f32::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / left.len() as f32
}

fn overlap(existing: &Skin, incoming: &Skin) -> Overlap {
    let mut matches = 0;
    let mut content = false;
    for tile in &incoming.tiles {
        if let Some((index, _)) = existing.nearest(&tile.color, tile.peak) {
            matches += 1;
            if !matches!(existing.tiles[index].cell, Cell::Wall | Cell::Empty) {
                content = true;
            }
        }
    }
    Overlap { matches, content }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(cell: Cell, color: f32) -> SkinTile {
        SkinTile {
            cell,
            color: vec![color; MATCH_VALUES],
            peak: None,
        }
    }

    fn skin(name: &str, cells: [Cell; 3], colors: [f32; 3]) -> Skin {
        Skin {
            name: name.to_owned(),
            tiles: cells
                .into_iter()
                .zip(colors)
                .map(|(cell, color)| tile(cell, color))
                .collect(),
        }
    }

    #[test]
    fn repeated_observations_stay_one_skin() {
        let mut library = SkinLibrary::default();
        let observation = || {
            skin(
                "first",
                [Cell::Wall, Cell::Box, Cell::Goal],
                [0.1, 0.5, 0.9],
            )
        };

        assert_eq!(library.remember(observation()), Some(true));
        assert_eq!(library.remember(observation()), Some(false));
        assert_eq!(library.skins.len(), 1);
        assert_eq!(library.skins[0].name, "first");
    }

    #[test]
    fn a_corrected_label_replaces_the_old_one() {
        let mut library = SkinLibrary::default();
        let _ = library.remember(skin(
            "first",
            [Cell::Wall, Cell::Box, Cell::Goal],
            [0.1, 0.5, 0.9],
        ));
        assert_eq!(
            library.remember(skin(
                "again",
                [Cell::Wall, Cell::Goal, Cell::Goal],
                [0.1, 0.5, 0.9],
            )),
            Some(true)
        );

        assert_eq!(library.skins.len(), 1);
        let corrected = &library.skins[0].tiles[1];
        assert_eq!(corrected.cell, Cell::Goal);
    }

    #[test]
    fn a_different_skin_is_kept_separately() {
        let mut library = SkinLibrary::default();
        let _ = library.remember(skin(
            "first",
            [Cell::Wall, Cell::Box, Cell::Goal],
            [0.1, 0.5, 0.9],
        ));
        let _ = library.remember(skin(
            "second",
            [Cell::Wall, Cell::Box, Cell::Goal],
            [0.2, 0.6, 0.8],
        ));

        assert_eq!(library.skins.len(), 2);
    }

    #[test]
    fn older_prototypes_are_dropped() {
        let mut library = SkinLibrary::default();
        library.skins.push(Skin {
            name: "old".to_owned(),
            tiles: vec![
                SkinTile {
                    cell: Cell::Wall,
                    color: vec![0.1; TILE_SIZE * TILE_SIZE * 3],
                    peak: None,
                },
                tile(Cell::Box, 0.5),
            ],
        });

        assert!(library.prune());
        assert_eq!(library.skins[0].tiles, vec![tile(Cell::Box, 0.5)]);
        assert!(!library.prune());
    }

    #[test]
    fn a_skin_explains_matching_clusters() {
        let mut library = SkinLibrary::default();
        let _ = library.remember(skin(
            "first",
            [Cell::Wall, Cell::Box, Cell::Goal],
            [0.1, 0.5, 0.9],
        ));

        let colors: Vec<Vec<f32>> = [0.1, 0.5, 0.9]
            .into_iter()
            .map(|color| vec![color; TILE_SIZE * TILE_SIZE * 3])
            .collect();
        let samples: Vec<(&[f32], Option<f32>)> = colors
            .iter()
            .map(|color| (color.as_slice(), None))
            .collect();
        let (matched, hints) = library.match_skin(&samples).expect("the skin matches");

        assert_eq!(matched.name, "first");
        assert_eq!(matched.matched, 3);
        assert_eq!(
            hints,
            vec![Some(Cell::Wall), Some(Cell::Box), Some(Cell::Goal)]
        );
    }

    #[test]
    fn walls_and_floor_alone_do_not_identify_a_skin() {
        let mut library = SkinLibrary::default();
        let _ = library.remember(Skin {
            name: "floors".to_owned(),
            tiles: vec![tile(Cell::Wall, 0.1), tile(Cell::Empty, 0.2)],
        });

        let colors: Vec<Vec<f32>> = [0.1, 0.2]
            .into_iter()
            .map(|color| vec![color; TILE_SIZE * TILE_SIZE * 3])
            .collect();
        let samples: Vec<(&[f32], Option<f32>)> = colors
            .iter()
            .map(|color| (color.as_slice(), None))
            .collect();
        assert!(library.match_skin(&samples).is_none());
    }

    #[test]
    fn peak_keeps_tiny_goal_dots_distinct_from_identical_coarse_floor() {
        let mut library = SkinLibrary::default();
        let skin = Skin {
            name: "dots".to_owned(),
            tiles: vec![
                SkinTile {
                    cell: Cell::Wall,
                    color: vec![0.7; MATCH_VALUES],
                    peak: Some(0.4),
                },
                SkinTile {
                    cell: Cell::Empty,
                    color: vec![0.0; MATCH_VALUES],
                    peak: Some(0.01),
                },
                SkinTile {
                    cell: Cell::Goal,
                    color: vec![0.004; MATCH_VALUES],
                    peak: Some(0.45),
                },
            ],
        };
        assert_eq!(library.remember(skin.clone()), Some(true));
        // Re-learning the same skin must not collapse Goal and Empty together.
        assert_eq!(library.remember(skin), Some(false));
        assert_eq!(library.skins[0].tiles.len(), 3);

        let wall = vec![0.7; TILE_SIZE * TILE_SIZE * 3];
        let floor = vec![0.0; TILE_SIZE * TILE_SIZE * 3];
        let dimmed_goal = vec![0.001; TILE_SIZE * TILE_SIZE * 3];
        let samples = [
            (wall.as_slice(), Some(0.4)),
            (floor.as_slice(), Some(0.01)),
            (dimmed_goal.as_slice(), Some(0.22)),
        ];
        let (_, hints) = library.match_skin(&samples).expect("matches dots skin");
        assert_eq!(
            hints,
            vec![Some(Cell::Wall), Some(Cell::Empty), Some(Cell::Goal)]
        );
    }
}
