//! Checking that a recognized board is a playable level.

use std::collections::VecDeque;
use std::str::FromStr;

use soukoban::map::Map;

use crate::geometry::neighbours;
use crate::{Board, Cell};

/// Validates a recognized board and returns human readable problems.
///
/// Box/player counts are checked by the `soukoban` map parser, so the messages
/// are its exact diagnostics. A missing outer wall is checked separately.
pub fn validate_board(board: &Board) -> Vec<String> {
    if board.is_empty() {
        return vec!["level is empty".to_owned()];
    }

    let unknown = board.unknown_cells();
    if !unknown.is_empty() {
        let preview: Vec<String> = unknown
            .iter()
            .take(5)
            .map(|(row, column)| format!("({row}, {column})"))
            .collect();
        return vec![format!(
            "{} cells are unresolved: {}",
            unknown.len(),
            preview.join(", ")
        )];
    }

    let mut warnings = Vec::new();
    if let Err(error) = Map::from_str(&board.to_xsb_lossy()) {
        warnings.push(format!("level validation failed: {error}"));
    }
    warnings.extend(enclosure_warnings(board));
    warnings
}

/// Level content must not be reachable from outside the walls.
fn enclosure_warnings(board: &Board) -> Vec<String> {
    let (rows, columns) = (board.height(), board.width());
    if rows == 0 || columns == 0 {
        return Vec::new();
    }
    if board.cells.iter().flatten().all(|cell| *cell != Cell::Wall) {
        return vec!["level has no walls".to_owned()];
    }

    let mut reached = vec![vec![false; columns]; rows];
    let mut queue = VecDeque::new();
    let border = (0..rows)
        .flat_map(|row| [(row, 0), (row, columns - 1)])
        .chain((0..columns).flat_map(|column| [(0, column), (rows - 1, column)]));
    for (row, column) in border {
        flood(&mut queue, &mut reached, board, row, column);
    }

    while let Some((row, column)) = queue.pop_front() {
        for (next_row, next_column) in neighbours(row, column, rows, columns) {
            flood(&mut queue, &mut reached, board, next_row, next_column);
        }
    }

    for (row, cells) in board.cells.iter().enumerate() {
        for (column, cell) in cells.iter().enumerate() {
            if reached[row][column]
                && matches!(
                    cell,
                    Cell::Box | Cell::BoxOnGoal | Cell::Goal | Cell::Player | Cell::PlayerOnGoal
                )
            {
                return vec![format!(
                    "level is not enclosed by walls: {} at row {}, column {} is reachable from outside",
                    cell.name(),
                    row,
                    column
                )];
            }
        }
    }
    Vec::new()
}

/// Marks a non-wall cell as reachable from outside the walls and queues it.
fn flood(
    queue: &mut VecDeque<(usize, usize)>,
    reached: &mut [Vec<bool>],
    board: &Board,
    row: usize,
    column: usize,
) {
    if board.get(row, column) == Some(Cell::Wall) || reached[row][column] {
        return;
    }
    reached[row][column] = true;
    queue.push_back((row, column));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warnings(level: &str) -> Vec<String> {
        validate_board(&Board::from_xsb(level).unwrap())
    }

    #[test]
    fn accepts_an_enclosed_level_with_matching_counts() {
        assert!(warnings("#####\n#@$.#\n#####").is_empty());
    }

    #[test]
    fn reports_missing_player() {
        let found = warnings("#####\n#-$.#\n#####");
        assert!(
            found.iter().any(|warning| warning.contains("no player")),
            "{found:?}"
        );
    }

    #[test]
    fn reports_multiple_players() {
        let found = warnings("######\n#@$@.#\n######");
        assert!(
            found
                .iter()
                .any(|warning| warning.contains("more than one player")),
            "{found:?}"
        );
    }

    #[test]
    fn reports_box_goal_mismatch() {
        let found = warnings("#####\n#@$-#\n#####");
        assert!(
            found
                .iter()
                .any(|warning| warning.contains("mismatch between number of boxes and goals")),
            "{found:?}"
        );
    }

    #[test]
    fn reports_missing_boxes_and_goals() {
        let found = warnings("#####\n#@--#\n#####");
        assert!(
            found
                .iter()
                .any(|warning| warning.contains("no box or goal")),
            "{found:?}"
        );
    }

    #[test]
    fn reports_content_reachable_from_outside() {
        let found = warnings("#####\n#@$.#\n#---#");
        assert!(
            found
                .iter()
                .any(|warning| warning.contains("not enclosed by walls")),
            "{found:?}"
        );
    }

    #[test]
    fn reports_unresolved_cells() {
        let board = Board::new(vec![vec![Cell::Player, Cell::Unknown]]).unwrap();
        let found = validate_board(&board);
        assert_eq!(found.len(), 1);
        assert!(found[0].contains("1 cells are unresolved"), "{found:?}");
    }
}
