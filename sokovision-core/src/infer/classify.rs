//! Assigning a symbol to every appearance group.

use crate::{Cell, CellScore};

use super::cluster::Cluster;
use super::mean_distance;
use super::solve::{self, ClassChoice, ClusterOptions, HINT_BONUS};
use super::wall;

/// How many single-cell appearances are tried as the player.
const PLAYER_CANDIDATES: usize = 8;
/// Above this many groups the alternatives are scored from the group's own
/// options instead of re-solving the whole board per candidate.
const EXACT_ALTERNATIVES: usize = 12;
/// Tie breaker in favour of the plain player when the sprite looks the same
/// either way.
const ON_GOAL_PENALTY: f32 = 1e-4;

pub(super) struct Classification {
    pub classes: Vec<Cell>,
    pub confidence: Vec<f32>,
    pub alternatives: Vec<Vec<CellScore>>,
    pub relaxed: bool,
    pub incomplete_structure: bool,
}

/// One complete assignment of symbols to appearance groups.
#[derive(Clone)]
struct Attempt {
    player: Option<usize>,
    on_goal: bool,
    total: f32,
    /// Score of the chosen player tile, kept so the confidence pass does not
    /// have to look the clusters up again.
    player_value: f32,
    /// Whether the remembered player tile was standing on a goal.
    hint_on_goal: Option<bool>,
    /// Appearance groups that need a symbol, in board order.
    order: Vec<usize>,
    options: Vec<ClusterOptions>,
    choice: Vec<ClassChoice>,
}

struct Input<'a> {
    clusters: &'a [Cluster],
    /// Remembered symbol per appearance group, from a learned skin.
    hints: &'a [Option<Cell>],
    interior: &'a [usize],
    is_wall: &'a [bool],
    cornered: &'a [bool],
    valid_total: usize,
    anchor_border: Option<[f32; 3]>,
}

pub(super) fn classify(
    clusters: &[Cluster],
    valid_total: usize,
    valid: &[Vec<bool>],
    rows: usize,
    columns: usize,
    hints: &[Option<Cell>],
    warnings: &mut Vec<String>,
) -> Classification {
    let count = clusters.len();
    let (mut is_wall, found_wall) =
        wall::find(clusters, valid, rows, columns, valid_total, warnings);
    let is_background = materials(hints, &mut is_wall, clusters);
    let mut wall_at = vec![vec![false; columns]; rows];
    for (index, cluster) in clusters.iter().enumerate() {
        if !is_wall[index] {
            continue;
        }
        for &(row, column) in &cluster.cells {
            wall_at[row][column] = true;
        }
    }
    let corners = wall::corners(&wall_at);
    let cornered: Vec<bool> = clusters
        .iter()
        .map(|cluster| {
            cluster
                .cells
                .iter()
                .any(|&(row, column)| corners[row][column])
        })
        .collect();
    let anchor_border = wall::floor_reference(clusters, &is_wall, &is_background);
    let interior: Vec<usize> = (0..count)
        .filter(|&index| !is_wall[index] && !is_background[index])
        .collect();
    let input = Input {
        clusters,
        hints,
        interior: &interior,
        is_wall: &is_wall,
        cornered: &cornered,
        valid_total,
        anchor_border,
    };

    let mut classes = vec![Cell::Unknown; count];
    let mut confidence = vec![0.0f32; count];
    let mut alternatives = vec![Vec::new(); count];
    for index in 0..count {
        if is_wall[index] {
            classes[index] = Cell::Wall;
            confidence[index] = 0.98;
        } else if is_background[index] {
            classes[index] = Cell::Empty;
            confidence[index] = 0.85;
        }
    }

    let (players, clean_player) = player_candidates(clusters, &interior, &is_wall, warnings);
    let incomplete_structure = !found_wall || !clean_player;
    let players = prefer_hinted_player(players, hints, &is_wall);
    let (attempts, relaxed) = attempt_all(&input, &players, warnings);
    let Some((best, second_total)) = attempts else {
        for slot in &mut confidence {
            *slot = slot.min(0.55);
        }
        return Classification {
            classes,
            confidence,
            alternatives,
            relaxed,
            incomplete_structure: true,
        };
    };

    for (position, &cluster) in best.order.iter().enumerate() {
        classes[cluster] = best.choice[position].cell();
    }
    if let Some(player) = best.player {
        classes[player] = if best.on_goal {
            Cell::PlayerOnGoal
        } else {
            Cell::Player
        };
    }
    score_attempt(
        &best,
        second_total,
        relaxed,
        &mut confidence,
        &mut alternatives,
    );

    Classification {
        classes,
        confidence,
        alternatives,
        relaxed,
        incomplete_structure,
    }
}

/// What each appearance is made of. A remembered symbol decides for a group
/// the skin explains, so a corrected wall or floor survives the next read; the
/// geometric wall guess only fills in where the skin is silent. The level rules
/// still decide what a non-wall group may be.
fn materials(hints: &[Option<Cell>], is_wall: &mut [bool], clusters: &[Cluster]) -> Vec<bool> {
    for (index, slot) in is_wall.iter_mut().enumerate() {
        match hints.get(index).copied().flatten() {
            Some(Cell::Wall) => *slot = true,
            Some(_) => *slot = false,
            None => {}
        }
    }
    clusters
        .iter()
        .enumerate()
        .map(|(index, cluster)| {
            let hinted = hints.get(index).copied().flatten();
            !is_wall[index]
                && cluster.boundary > 0
                && !matches!(hinted, Some(cell) if cell != Cell::Empty)
        })
        .collect()
}

/// The player is the only tile that appears exactly once. Candidates are ranked
/// by how much they look like a sprite, so a dithering fragment of a box does
/// not win.
fn player_candidates(
    clusters: &[Cluster],
    interior: &[usize],
    is_wall: &[bool],
    warnings: &mut Vec<String>,
) -> (Vec<Option<usize>>, bool) {
    let mut candidates: Vec<usize> = interior
        .iter()
        .copied()
        .filter(|&index| clusters[index].len() == 1)
        .collect();
    rank(&mut candidates, clusters);
    let mut clean_player = true;

    if candidates.is_empty() {
        clean_player = false;
        candidates = (0..clusters.len())
            .filter(|&index| !is_wall[index] && clusters[index].len() == 1)
            .collect();
        rank(&mut candidates, clusters);
        if !candidates.is_empty() {
            warnings.push(
                "no unique player tile was found inside the walled area; using the most distinctive unique tile"
                    .to_owned(),
            );
        }
    }

    if candidates.is_empty() {
        warnings.push("no unique tile can be used as the player".to_owned());
        return (vec![None], false);
    }
    (candidates.into_iter().map(Some).collect(), clean_player)
}

fn rank(candidates: &mut Vec<usize>, clusters: &[Cluster]) {
    candidates.sort_by(|&left, &right| {
        player_score(&clusters[right]).total_cmp(&player_score(&clusters[left]))
    });
    candidates.truncate(PLAYER_CANDIDATES);
}

/// A remembered player tile goes first, so the rules only have to confirm the
/// skin instead of rediscovering the sprite from scratch.
fn prefer_hinted_player(
    mut players: Vec<Option<usize>>,
    hints: &[Option<Cell>],
    is_wall: &[bool],
) -> Vec<Option<usize>> {
    let hinted = hints
        .iter()
        .position(|hint| matches!(hint, Some(Cell::Player | Cell::PlayerOnGoal)));
    let Some(index) = hinted.filter(|&index| !is_wall[index]) else {
        return players;
    };
    players.retain(|&player| player != Some(index));
    players.insert(0, Some(index));
    players
}

/// Tries every player candidate under the level constraints. Returns the best
/// attempt with the score of the runner-up, and whether the constraints had to
/// be dropped.
fn attempt_all(
    input: &Input,
    players: &[Option<usize>],
    warnings: &mut Vec<String>,
) -> (Option<(Attempt, f32)>, bool) {
    let hinted = players.first().copied().flatten().filter(|&player| {
        input
            .hints
            .get(player)
            .is_some_and(|hint| hint.is_some_and(is_player_hint))
    });

    let mut attempts: Vec<Attempt> = Vec::new();
    for &player in players {
        let Some(attempt) = attempt(input, player, false, true) else {
            continue;
        };
        attempts.push(attempt);
        // The remembered player produced a board that satisfies every rule, so
        // the other unique tiles cannot be the player. Skipping them keeps a
        // large board from re-running the search for each of them.
        if player == hinted {
            break;
        }
    }
    by_score(&mut attempts);
    if let Some(best) = attempts.first() {
        let second = attempts.get(1).map_or(f32::NEG_INFINITY, |next| next.total);
        return (Some((best.clone(), second)), false);
    }

    warnings.push(
        "the tile appearances cannot satisfy the level constraints (one player, equal boxes and goals); showing the best visual guess"
            .to_owned(),
    );
    let mut fallback: Vec<Attempt> = players
        .iter()
        .filter_map(|&player| attempt(input, player, true, false))
        .collect();
    by_score(&mut fallback);
    (
        fallback
            .first()
            .cloned()
            .map(|best| (best, f32::NEG_INFINITY)),
        true,
    )
}

/// Sorts attempts by score, best first.
fn by_score(attempts: &mut [Attempt]) {
    attempts.sort_by(|left, right| right.total.total_cmp(&left.total));
}

fn attempt(
    input: &Input,
    player: Option<usize>,
    all_classes: bool,
    require_valid: bool,
) -> Option<Attempt> {
    let order: Vec<usize> = input
        .interior
        .iter()
        .copied()
        .filter(|&index| Some(index) != player)
        .collect();
    if order.is_empty() {
        return None;
    }

    let anchor_border = input
        .anchor_border
        .unwrap_or_else(|| fallback_anchor(input.clusters, &order));
    let options: Vec<ClusterOptions> = order
        .iter()
        .map(|&index| {
            solve::build(
                input.clusters,
                index,
                anchor_border,
                input.valid_total,
                all_classes,
                input.cornered[index],
                input.hints[index].and_then(class_hint),
            )
        })
        .collect();
    let player_hint = player.and_then(|player| input.hints[player]);
    let player_value = player
        .map(|player| {
            let value = player_value(input.clusters, input.is_wall, player);
            if player_hint.is_some_and(is_player_hint) {
                value + HINT_BONUS
            } else {
                value
            }
        })
        .unwrap_or(0.0);
    let hint_on_goal =
        player_hint
            .filter(|cell| is_player_hint(*cell))
            .and_then(|cell| match cell {
                Cell::Player => Some(false),
                Cell::PlayerOnGoal => Some(true),
                _ => None,
            });

    let mut best: Option<Attempt> = None;
    for on_goal in [false, true] {
        if on_goal && player.is_none() {
            continue;
        }
        let Some((score, choice)) = solve::solve(&options, on_goal, require_valid, None) else {
            continue;
        };
        let total = total_of(score, player_value, on_goal, hint_on_goal);
        if best
            .as_ref()
            .is_none_or(|current: &Attempt| total > current.total)
        {
            best = Some(Attempt {
                player,
                on_goal,
                total,
                player_value,
                hint_on_goal,
                order: order.clone(),
                options: options.clone(),
                choice,
            });
        }
    }
    best
}

/// Remembered symbol of an interior appearance as a solver choice.
fn class_hint(cell: Cell) -> Option<ClassChoice> {
    match cell {
        Cell::Empty => Some(ClassChoice::Empty),
        Cell::Goal => Some(ClassChoice::Goal),
        Cell::Box => Some(ClassChoice::Box),
        Cell::BoxOnGoal => Some(ClassChoice::BoxOnGoal),
        _ => None,
    }
}

fn is_player_hint(cell: Cell) -> bool {
    matches!(cell, Cell::Player | Cell::PlayerOnGoal)
}

/// Total score of an attempt: the board score plus the player bonus.
fn total_of(score: f32, player_value: f32, on_goal: bool, hint_on_goal: Option<bool>) -> f32 {
    let hint = if hint_on_goal == Some(on_goal) {
        HINT_BONUS
    } else {
        0.0
    };
    score + player_value + hint - if on_goal { ON_GOAL_PENALTY } else { 0.0 }
}

/// Without a floor reference, the largest remaining appearance is the floor.
fn fallback_anchor(clusters: &[Cluster], order: &[usize]) -> [f32; 3] {
    order
        .iter()
        .copied()
        .max_by(|&left, &right| {
            clusters[left]
                .len()
                .cmp(&clusters[right].len())
                .then_with(|| clusters[right].edge.total_cmp(&clusters[left].edge))
                .then_with(|| clusters[right].marker.total_cmp(&clusters[left].marker))
        })
        .map(|index| clusters[index].border)
        .unwrap_or([0.0; 3])
}

/// Fills in how sure the assignment is: for every group, the score gap to the
/// best assignment that gives it another symbol.
fn score_attempt(
    best: &Attempt,
    second_total: f32,
    relaxed: bool,
    confidence: &mut [f32],
    alternatives: &mut [Vec<CellScore>],
) {
    if relaxed {
        for slot in confidence.iter_mut() {
            *slot = slot.min(0.55);
        }
        for (position, &cluster) in best.order.iter().enumerate() {
            confidence[cluster] = 0.45;
            let chosen = best.choice[position];
            let chosen_score = best.options[position]
                .options
                .iter()
                .find(|option| option.class == chosen)
                .map_or(0.0, |option| option.score);
            let mut candidates: Vec<CellScore> = best.options[position]
                .options
                .iter()
                .filter(|option| option.class != chosen)
                .map(|option| CellScore {
                    cell: option.class.cell(),
                    score: relative_score((chosen_score - option.score).max(0.0)),
                })
                .collect();
            candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
            alternatives[cluster] = candidates;
        }
        if let Some(player) = best.player {
            confidence[player] = 0.45;
        }
        return;
    }

    let total =
        |score: f32, on_goal: bool| total_of(score, best.player_value, on_goal, best.hint_on_goal);

    if best.order.len() <= EXACT_ALTERNATIVES {
        for (position, &cluster) in best.order.iter().enumerate() {
            let chosen = best.choice[position];
            let mut gap = f32::INFINITY;
            let mut candidates = Vec::new();
            for option in &best.options[position].options {
                if option.class == chosen {
                    continue;
                }
                let Some((score, _)) = solve::solve(
                    &best.options,
                    best.on_goal,
                    true,
                    Some((cluster, option.class)),
                ) else {
                    continue;
                };
                let difference = (best.total - total(score, best.on_goal)).max(0.0);
                gap = gap.min(difference);
                candidates.push(CellScore {
                    cell: option.class.cell(),
                    score: relative_score(difference),
                });
            }
            candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
            alternatives[cluster] = candidates;
            confidence[cluster] = confidence_from_gap(gap);
        }
    } else {
        for (position, &cluster) in best.order.iter().enumerate() {
            let chosen = best.choice[position];
            let chosen_score = best.options[position]
                .options
                .iter()
                .find(|option| option.class == chosen)
                .map_or(0.0f32, |option| option.score);
            let runner_up = best.options[position]
                .options
                .iter()
                .filter(|option| option.class != chosen)
                .map(|option| option.score)
                .fold(f32::NEG_INFINITY, f32::max);
            confidence[cluster] = confidence_from_gap((chosen_score - runner_up).max(0.0));
        }
    }

    let Some(player) = best.player else {
        return;
    };
    let mut gap = f32::INFINITY;
    if second_total.is_finite() {
        gap = gap.min((best.total - second_total).max(0.0));
    }
    let on_goal = !best.on_goal;
    if let Some((score, _)) = solve::solve(&best.options, on_goal, true, None) {
        let difference = (best.total - total(score, on_goal)).max(0.0);
        gap = gap.min(difference);
        alternatives[player].push(CellScore {
            cell: if on_goal {
                Cell::PlayerOnGoal
            } else {
                Cell::Player
            },
            score: relative_score(difference),
        });
        alternatives[player].sort_by(|left, right| right.score.total_cmp(&left.score));
    }
    confidence[player] = confidence_from_gap(gap);
}

fn player_score(cluster: &Cluster) -> f32 {
    0.5 * (cluster.center / 0.35).clamp(0.0, 1.0)
        + 0.5 * (cluster.marker / 0.30).clamp(0.0, 1.0)
        + 0.35 * (cluster.edge / 0.20).clamp(0.0, 1.0)
}

/// A single cell that repeats an appearance used elsewhere is a dithering
/// fragment of a box, not the player.
fn player_value(clusters: &[Cluster], is_wall: &[bool], player: usize) -> f32 {
    let nearest = clusters
        .iter()
        .enumerate()
        .filter(|&(index, _)| index != player && !is_wall[index])
        .map(|(_, cluster)| mean_distance(cluster.border, clusters[player].border))
        .fold(f32::INFINITY, f32::min);
    let duplicate = if nearest.is_finite() {
        (1.0 - nearest / 0.10).clamp(0.0, 1.0)
    } else {
        0.0
    };
    player_score(&clusters[player]) - 0.6 * duplicate
}

fn confidence_from_gap(gap: f32) -> f32 {
    if !gap.is_finite() {
        return 0.99;
    }
    0.5 + 0.49 * (1.0 - (-gap * 1.5).exp())
}

fn relative_score(gap: f32) -> f32 {
    if !gap.is_finite() {
        0.0
    } else {
        (1.0 - gap / (gap + 1.0)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::cluster::Cluster;

    fn cluster(boundary: usize) -> Cluster {
        Cluster {
            cells: Vec::new(),
            color: Vec::new(),
            border: [0.0; 3],
            edge: 0.0,
            marker: 0.0,
            center: 0.0,
            peak: 0.0,
            sparse: 0.0,
            boundary,
            enclosure: 0.0,
        }
    }

    #[test]
    fn remembered_materials_beat_the_geometric_guess() {
        let clusters = [cluster(1), cluster(1), cluster(1), cluster(1)];
        let hints = [Some(Cell::Empty), Some(Cell::Wall), Some(Cell::Box), None];
        let mut is_wall = vec![true, false, false, false];

        let is_background = materials(&hints, &mut is_wall, &clusters);

        assert_eq!(
            is_wall,
            vec![false, true, false, false],
            "a remembered floor is not a wall, a remembered wall is"
        );
        assert_eq!(
            is_background,
            vec![true, false, false, true],
            "only floor and groups the skin is silent about may be background"
        );
    }
}
