use thiserror::Error;

use crate::PixelRect;

/// Everything that can go wrong while reading a board or a screenshot.
#[derive(Debug, Error)]
pub enum Error {
    #[error("empty board")]
    EmptyBoard,

    #[error("board rows have different widths")]
    RaggedBoard,

    #[error("unsupported XSB character {ch:?} at row {row}, column {column}")]
    UnsupportedXsbChar { ch: char, row: usize, column: usize },

    #[error("cell at row {row}, column {column} is not confirmed")]
    UnconfirmedCell { row: usize, column: usize },

    #[error("image is empty")]
    EmptyImage,

    #[error("region {0:?} is outside the image")]
    InvalidRegion(PixelRect),

    #[error("could not detect a grid: {0}")]
    GridDetection(String),

    #[error("invalid grid: {0}")]
    InvalidGrid(String),
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
