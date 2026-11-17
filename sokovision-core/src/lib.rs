//! Recognize Sokoban boards from screenshots.
//!
//! The crate is pure image processing: it takes pixels in and gives a board
//! out. Screenshots, file dialogs and clipboards belong to the callers, which
//! keeps `sokovision-core` usable from the CLI, the GUI and third-party code.
//!
//! ```no_run
//! use sokovision_core::{RecognizeOptions, image::open, recognize};
//!
//! let image = open("level.png")?.to_rgba8();
//! let result = recognize(&image, &RecognizeOptions::default())?;
//! println!("{}", result.board().to_xsb()?);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod board;
mod detect;
mod error;
mod geometry;
mod infer;
mod recognize;
mod search;
mod skin;
mod validate;

pub use board::{Board, Cell};
pub use error::{Error, Result};
pub use geometry::{Grid, PixelRect};
pub use recognize::{CellScore, RecognitionResult, RecognizeOptions, RecognizedCell, recognize};
pub use search::refine_grid;
pub use skin::{Skin, SkinLibrary, SkinMatch, SkinTile};
pub use validate::validate_board;

pub use image;

/// Largest board, per axis, that automatic detection accepts: 100x100.
///
/// No Sokoban level comes close, so anything denser is the detector latching
/// onto desktop, text or UI texture rather than a level. The limit doubles as a
/// hang guard: per-cell inference is quadratic in the cell count, and a
/// full-screen screenshot used to fit a 182x79 grid and freeze the app. The GUI
/// caps its manual row and column steppers with the same number.
pub const MAX_BOARD_AXIS: u32 = 100;
