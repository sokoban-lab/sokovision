//! Choosing a symbol for every appearance group under the rules of Sokoban.
//!
//! The decision is a dynamic program over two counters: `goals - boxes` (plus
//! one for a player standing on a goal) and whether a box has been placed. A
//! complete assignment is valid when the first counter is zero and at least one
//! box exists, which is exactly "as many boxes as goals".

use super::cluster::Cluster;
use super::mean_distance;
use crate::Cell;

/// Appearances closer than this to the floor are drawn on it, so they can only
/// be floor or goals; everything else can only be a box.
const FLOOR_LINK: f32 = 0.10;
/// Score a choice gains when it agrees with a remembered tile. Large enough to
/// outvote the visual guess, small enough that the level constraints still win.
pub(super) const HINT_BONUS: f32 = 4.0;
/// Hair-thin tie-breaker above `classify::ON_GOAL_PENALTY` (`1e-4`), so a
/// single split-off `*` stays `(*, @)` instead of flipping the player to `+`
/// (`($, +)`), without being large enough to erase faint goal clusters.
const COVERED_GOAL_TIE: f32 = 2e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ClassChoice {
    Empty,
    Goal,
    Box,
    BoxOnGoal,
}

impl ClassChoice {
    pub fn cell(self) -> Cell {
        match self {
            Self::Empty => Cell::Empty,
            Self::Goal => Cell::Goal,
            Self::Box => Cell::Box,
            Self::BoxOnGoal => Cell::BoxOnGoal,
        }
    }
}

/// One symbol an appearance group may take, with its effect on the counters.
#[derive(Debug, Clone, Copy)]
pub(super) struct Choice {
    pub class: ClassChoice,
    /// Contribution to `goals - boxes + player_on_goal`.
    diff: i64,
    /// Contribution to the box count.
    boxes: i64,
    pub score: f32,
}

impl Choice {
    const fn new(class: ClassChoice, diff: i64, boxes: i64, score: f32) -> Self {
        Self {
            class,
            diff,
            boxes,
            score,
        }
    }
}

/// The choices available to one appearance group.
#[derive(Debug, Clone)]
pub(super) struct ClusterOptions {
    pub cluster: usize,
    pub options: Vec<Choice>,
}

pub(super) fn build(
    clusters: &[Cluster],
    index: usize,
    anchor_border: [f32; 3],
    valid_total: usize,
    all_classes: bool,
    corner: bool,
    hint: Option<ClassChoice>,
) -> ClusterOptions {
    let cluster = &clusters[index];
    let distance = mean_distance(cluster.border, anchor_border);
    let background = (1.0 - distance / 0.12).clamp(0.0, 1.0);
    let marker = (cluster.marker / 0.30).clamp(0.0, 1.0);
    let edge = (cluster.edge / 0.20).clamp(0.0, 1.0);
    let peak = (cluster.peak * (1.0 - cluster.sparse)).clamp(0.0, 1.0);
    let size_score = 0.5 * (cluster.len() as f32 / valid_total.max(1) as f32).sqrt();

    // Tiny goal dots barely move the centre average (`marker`) but do peak
    // locally, so plain floor must also be penalised for a localized deviation.
    let plain = (1.0 - peak / 0.35).clamp(0.0, 1.0);
    let empty = 3.0 * background * background + 0.8 * (1.0 - marker) * plain + size_score;
    let goal = 3.0 * background * background + 2.5 * marker + 3.0 * peak + 0.3 * size_score;
    let box_score = 2.5 * (1.0 - background) * (1.0 - background) + 0.6 * edge + 0.2 * size_score;
    // A goal under a box is barely visible through the sprite, so the equal
    // box/goal count constraint decides; this only outvotes flipping `@` to `+`.
    let box_on_goal = box_score + COVERED_GOAL_TIE;

    let size = cluster.len() as i64;
    let empty_choice = Choice::new(ClassChoice::Empty, 0, 0, empty);
    let goal_choice = Choice::new(ClassChoice::Goal, size, 0, goal);
    let box_choice = Choice::new(ClassChoice::Box, -size, size, box_score);
    let box_on_goal_choice = Choice::new(ClassChoice::BoxOnGoal, 0, size, box_on_goal);

    // A remembered tile may contradict the floor link; the rules still decide
    // whether the assignment works out. A box wedged into a wall corner stays
    // impossible, learned or not.
    let wanted = |class: ClassChoice| {
        let on_floor = matches!(class, ClassChoice::Empty | ClassChoice::Goal);
        all_classes || Some(class) == hint || on_floor == (distance < FLOOR_LINK)
    };
    let mut options = Vec::new();
    for (choice, allowed) in [
        (empty_choice, wanted(ClassChoice::Empty)),
        (goal_choice, wanted(ClassChoice::Goal)),
        (box_choice, !corner && wanted(ClassChoice::Box)),
        (box_on_goal_choice, wanted(ClassChoice::BoxOnGoal)),
    ] {
        if allowed {
            options.push(choice);
        }
    }
    if let Some(hint) = hint {
        for option in &mut options {
            if option.class == hint {
                option.score += HINT_BONUS;
            }
        }
        // Try the remembered symbol first: a good solution found early makes
        // the score bound prune the rest of the search.
        if let Some(position) = options.iter().position(|option| option.class == hint) {
            let hinted = options.remove(position);
            options.insert(0, hinted);
        }
    }
    ClusterOptions {
        cluster: index,
        options,
    }
}

/// Best scoring valid assignment, or `None` when no assignment satisfies the
/// counters.
///
/// The search is a dynamic program over (difference, has a box) so a board with
/// many appearance groups stays polynomial. Without `require_valid` the groups
/// are independent and the best symbol per group is enough.
pub(super) fn solve(
    options: &[ClusterOptions],
    player_on_goal: bool,
    require_valid: bool,
    forced: Option<(usize, ClassChoice)>,
) -> Option<(f32, Vec<ClassChoice>)> {
    if !require_valid {
        return visual_best(options, forced);
    }

    // Every group can move the goal/box difference by at most its own size.
    let offset: i64 = options
        .iter()
        .map(|group| {
            group
                .options
                .iter()
                .map(|option| option.diff.abs())
                .max()
                .unwrap_or(0)
        })
        .sum();
    let width = (offset * 2 + 1) as usize;
    let states = width * 2;
    let unreachable = f32::NEG_INFINITY;

    // Layer 0: no groups decided, difference zero, no box yet.
    let mut scores = vec![unreachable; states];
    scores[offset as usize * 2] = 0.0;
    let mut layers: Vec<Vec<Option<Back>>> = Vec::with_capacity(options.len());

    for group in options {
        let mut next = vec![unreachable; states];
        let mut back: Vec<Option<Back>> = vec![None; states];
        for (state, &score) in scores.iter().enumerate() {
            if score == unreachable {
                continue;
            }
            let difference = (state / 2) as i64 - offset;
            let had_box = state % 2 == 1;
            for (option_index, choice) in group.options.iter().enumerate() {
                if forced.is_some_and(|(cluster, class)| {
                    cluster == group.cluster && class != choice.class
                }) {
                    continue;
                }
                let difference = difference + choice.diff;
                let has_box = had_box || choice.boxes > 0;
                let target = (difference + offset) as usize * 2 + usize::from(has_box);
                let score = score + choice.score;
                if score > next[target] {
                    next[target] = score;
                    back[target] = Some(Back {
                        option: option_index as u8,
                        previous_had_box: had_box,
                    });
                }
            }
        }
        layers.push(back);
        scores = next;
    }

    let final_difference = -i64::from(player_on_goal);
    if final_difference.abs() > offset {
        return None;
    }
    // Exactly balanced and at least one box on the board.
    let mut state = (final_difference + offset) as usize * 2 + 1;
    let score = scores[state];
    if score == unreachable {
        return None;
    }

    let mut chosen = vec![ClassChoice::Empty; options.len()];
    for (index, group) in options.iter().enumerate().rev() {
        let back = layers[index][state]?;
        let choice = group.options[back.option as usize];
        chosen[index] = choice.class;
        let difference = (state / 2) as i64 - offset - choice.diff;
        state = (difference + offset) as usize * 2 + usize::from(back.previous_had_box);
    }
    Some((score, chosen))
}

/// Best symbol per group when no rule has to hold.
fn visual_best(
    options: &[ClusterOptions],
    forced: Option<(usize, ClassChoice)>,
) -> Option<(f32, Vec<ClassChoice>)> {
    let mut total = 0.0;
    let mut chosen = Vec::with_capacity(options.len());
    for group in options {
        let mut best: Option<&Choice> = None;
        for choice in &group.options {
            if forced
                .is_some_and(|(cluster, class)| cluster == group.cluster && class != choice.class)
            {
                continue;
            }
            if best.is_none_or(|best| choice.score > best.score) {
                best = Some(choice);
            }
        }
        let best = best?;
        total += best.score;
        chosen.push(best.class);
    }
    Some((total, chosen))
}

/// Which option a state of the dynamic program came from.
#[derive(Clone, Copy)]
struct Back {
    option: u8,
    previous_had_box: bool,
}
