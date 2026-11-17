use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// One cell of a Sokoban board.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cell {
    /// Floor, or the blank area outside the level.
    Empty,
    Wall,
    Goal,
    Box,
    BoxOnGoal,
    Player,
    PlayerOnGoal,
    /// Not resolved yet; cannot be exported as XSB.
    Unknown,
}

impl Cell {
    /// The standard XSB character, or `None` for [`Cell::Unknown`].
    pub const fn symbol(self) -> Option<char> {
        match self {
            Self::Empty => Some('-'),
            Self::Wall => Some('#'),
            Self::Goal => Some('.'),
            Self::Box => Some('$'),
            Self::BoxOnGoal => Some('*'),
            Self::Player => Some('@'),
            Self::PlayerOnGoal => Some('+'),
            Self::Unknown => None,
        }
    }

    /// Reads an XSB character. Both `-` and the usual blanks map to
    /// [`Cell::Empty`].
    pub const fn from_symbol(symbol: char) -> Option<Self> {
        match symbol {
            '-' | '_' | ' ' => Some(Self::Empty),
            '#' => Some(Self::Wall),
            '.' => Some(Self::Goal),
            '$' => Some(Self::Box),
            '*' => Some(Self::BoxOnGoal),
            '@' => Some(Self::Player),
            '+' => Some(Self::PlayerOnGoal),
            _ => None,
        }
    }

    /// Human readable name, used in diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Wall => "wall",
            Self::Goal => "goal",
            Self::Box => "box",
            Self::BoxOnGoal => "box_on_goal",
            Self::Player => "player",
            Self::PlayerOnGoal => "player_on_goal",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buffer = [0u8; 4];
        formatter.write_str(self.symbol().unwrap_or('?').encode_utf8(&mut buffer))
    }
}

/// A rectangular board.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Board {
    pub cells: Vec<Vec<Cell>>,
}

impl Board {
    /// Wraps rows that are known to be rectangular.
    pub fn new(cells: Vec<Vec<Cell>>) -> Result<Self> {
        let width = cells.first().map_or(0, Vec::len);
        if width == 0 {
            return Err(Error::EmptyBoard);
        }
        if cells.iter().any(|row| row.len() != width) {
            return Err(Error::RaggedBoard);
        }
        Ok(Self { cells })
    }

    /// Parses an XSB level. Short rows are padded with floor, a leading byte
    /// order mark and blank lines around the level are ignored.
    pub fn from_xsb(source: &str) -> Result<Self> {
        let source = source.strip_prefix('\u{feff}').unwrap_or(source);
        let blank = |line: &&str| line.trim().is_empty();
        let lines: Vec<&str> = source.lines().skip_while(blank).collect();
        let end = lines.iter().rposition(|line| !line.trim().is_empty());
        let lines = &lines[..end.map_or(0, |index| index + 1)];

        let width = lines
            .iter()
            .map(|line| line.chars().count())
            .max()
            .unwrap_or(0);
        if width == 0 {
            return Err(Error::EmptyBoard);
        }

        let mut cells = Vec::with_capacity(lines.len());
        for (row, line) in lines.iter().enumerate() {
            let mut parsed = Vec::with_capacity(width);
            for (column, ch) in line.chars().enumerate() {
                let cell = Cell::from_symbol(ch).ok_or(Error::UnsupportedXsbChar {
                    ch,
                    row: row + 1,
                    column: column + 1,
                })?;
                parsed.push(cell);
            }
            parsed.resize(width, Cell::Empty);
            cells.push(parsed);
        }
        Ok(Self { cells })
    }

    /// Exports the board as XSB, refusing unresolved cells.
    pub fn to_xsb(&self) -> Result<String> {
        match self.unknown_cells().first() {
            Some(&(row, column)) => Err(Error::UnconfirmedCell {
                row: row + 1,
                column: column + 1,
            }),
            None => Ok(self.to_xsb_lossy()),
        }
    }

    /// Exports the board as XSB, writing `?` for unresolved cells.
    pub fn to_xsb_lossy(&self) -> String {
        let mut output = String::with_capacity((self.width() + 1) * self.height());
        for (index, row) in self.cells.iter().enumerate() {
            if index > 0 {
                output.push('\n');
            }
            output.extend(row.iter().map(|cell| cell.symbol().unwrap_or('?')));
        }
        output
    }

    pub fn width(&self) -> usize {
        self.cells.first().map_or(0, Vec::len)
    }

    pub fn height(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.width() == 0
    }

    pub fn get(&self, row: usize, column: usize) -> Option<Cell> {
        self.cells.get(row)?.get(column).copied()
    }

    /// Positions of every unresolved cell, in row-major order.
    pub fn unknown_cells(&self) -> Vec<(usize, usize)> {
        let mut unknown = Vec::new();
        for (row, cells) in self.cells.iter().enumerate() {
            for (column, cell) in cells.iter().enumerate() {
                if *cell == Cell::Unknown {
                    unknown.push((row, column));
                }
            }
        }
        unknown
    }

    /// Drops the floor rows and columns around the level.
    pub fn trimmed(&self) -> Self {
        let used = |cell: &Cell| *cell != Cell::Empty;
        let Some(first_row) = self.cells.iter().position(|row| row.iter().any(used)) else {
            return Self { cells: Vec::new() };
        };
        let last_row = self
            .cells
            .iter()
            .rposition(|row| row.iter().any(used))
            .unwrap_or(first_row);
        let used_column = |column: usize| {
            self.cells[first_row..=last_row]
                .iter()
                .any(|row| used(&row[column]))
        };
        let first_column = (0..self.width()).find(|&column| used_column(column));
        let last_column = (0..self.width()).rfind(|&column| used_column(column));
        let (Some(first_column), Some(last_column)) = (first_column, last_column) else {
            return Self { cells: Vec::new() };
        };

        Self {
            cells: self.cells[first_row..=last_row]
                .iter()
                .map(|row| row[first_column..=last_column].to_vec())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEVEL: &str = "########---------\n\
#---#--#---------\n\
#...$$-#######---\n\
#...#--------#---\n\
#...#$######-#---\n\
#...#------#-##--\n\
#####-$-$#-#--#--\n\
----#-$-$-$-#-#--\n\
----#--#-$--@-#--\n\
----#-$$-$#####--\n\
----#-----#------\n\
----#######------\n\
-----------------";

    #[test]
    fn parses_and_serializes_xsb() {
        let board = Board::from_xsb(LEVEL).unwrap();
        assert_eq!(board.width(), 17);
        assert_eq!(board.height(), 13);
        assert_eq!(board.to_xsb().unwrap(), LEVEL);
    }

    #[test]
    fn pads_short_rows_and_accepts_spaces() {
        let board = Board::from_xsb("###\n#@ \n").unwrap();
        assert_eq!(board.width(), 3);
        assert_eq!(board.get(1, 2), Some(Cell::Empty));
        assert_eq!(board.to_xsb().unwrap(), "###\n#@-");
    }

    #[test]
    fn ignores_byte_order_mark_and_blank_lines() {
        let board = Board::from_xsb("\u{feff}\n\n#@#\r\n\n").unwrap();
        assert_eq!(board.to_xsb().unwrap(), "#@#");
    }

    #[test]
    fn rejects_invalid_character_with_location() {
        let error = Board::from_xsb("###\n#!+\n").unwrap_err();
        assert!(matches!(
            error,
            Error::UnsupportedXsbChar {
                ch: '!',
                row: 2,
                column: 2
            }
        ));
    }

    #[test]
    fn refuses_to_export_unknown_cells() {
        let board = Board::new(vec![vec![Cell::Player, Cell::Unknown]]).unwrap();
        assert!(matches!(
            board.to_xsb(),
            Err(Error::UnconfirmedCell { row: 1, column: 2 })
        ));
        assert_eq!(board.to_xsb_lossy(), "@?");
    }

    #[test]
    fn trims_only_empty_outer_rows_and_columns() {
        let board = Board::from_xsb("-----\n--###\n--#@#\n--###\n-----\n").unwrap();
        let trimmed = board.trimmed();
        assert_eq!((trimmed.width(), trimmed.height()), (3, 3));
        assert_eq!(trimmed.get(1, 1), Some(Cell::Player));
    }

    #[test]
    fn trimming_an_empty_board_yields_an_empty_board() {
        let board = Board::from_xsb("-----\n-----\n").unwrap();
        assert!(board.trimmed().is_empty());
    }
}
