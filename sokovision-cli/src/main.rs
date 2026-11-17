//! Command line front end for `sokovision-core`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use sokovision_core::{
    Cell, CellScore, Grid, PixelRect, RecognitionResult, RecognizeOptions, SkinLibrary, SkinMatch,
    recognize, validate_board,
};

/// Exit code used when a board was read but still needs attention.
const NEEDS_ATTENTION: u8 = 2;

/// Command-line flags accepted by the `sokovision` binary.
#[derive(Debug, Parser)]
#[command(
    name = "sokovision",
    version,
    about = "Recognize a Sokoban board image and export XSB"
)]
struct Arguments {
    /// Input screenshot or cropped board image.
    image: PathBuf,

    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Output format.
    #[arg(short, long, value_enum, default_value_t = Format::Xsb)]
    format: Format,

    /// Restrict grid detection to x,y,width,height in image pixels.
    #[arg(long, value_parser = parse_region)]
    region: Option<PixelRect>,

    /// Skip detection using x,y,cell_width,cell_height,rows,columns.
    #[arg(long, value_parser = parse_grid)]
    grid: Option<Grid>,

    /// Only detect the grid and leave all cells unresolved.
    #[arg(long)]
    grid_only: bool,

    /// Remove empty rows and columns around the board when exporting XSB.
    #[arg(long)]
    trim: bool,

    /// Load a JSON skin library before recognition.
    #[arg(long)]
    skins: Option<PathBuf>,

    /// Update and save a JSON skin library when the recognized board is valid.
    #[arg(long)]
    save_skins: Option<PathBuf>,
}

impl Arguments {
    /// Builds [`RecognizeOptions`] from the parsed flags and optional loaded skins.
    fn options(&self, skins: Option<SkinLibrary>) -> RecognizeOptions {
        RecognizeOptions {
            region: self.region,
            grid: self.grid,
            grid_only: self.grid_only,
            skins: skins.filter(|library| !library.skins.is_empty()),
        }
    }
}

/// Output representation selected on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// Plain XSB level text.
    Xsb,
    /// Structured JSON report with grid geometry, cell confidences and warnings.
    Json,
}

/// Structured JSON output produced by `--format json`.
#[derive(Debug, Serialize)]
struct Report {
    /// Input image width in pixels.
    width: u32,
    /// Input image height in pixels.
    height: u32,
    /// Fitted or supplied board grid.
    grid: Grid,
    /// Mean per-cell confidence across the board.
    quality: f32,
    /// Flattened cell-by-cell recognition details.
    cells: Vec<ReportCell>,
    /// Exported XSB string (`None` when unresolved cells remain).
    xsb: Option<String>,
    /// Whether the board passed all Sokoban structural checks.
    valid: bool,
    /// Whether global solver constraints had to be relaxed.
    relaxed: bool,
    /// Remembered skin matched during classification, if any.
    skin: Option<SkinMatch>,
    /// Structural validation errors, empty when `valid` is `true`.
    validation: Vec<String>,
    /// Non-fatal recognition notes.
    warnings: Vec<String>,
}

/// Per-cell entry inside [`Report`].
#[derive(Debug, Serialize)]
struct ReportCell {
    /// Zero-based row index.
    row: usize,
    /// Zero-based column index.
    column: usize,
    /// Assigned cell state.
    cell: Cell,
    /// Calibrated cell confidence in `0.0..=1.0`.
    confidence: f32,
    /// Runner-up candidate cell states with their scores.
    alternatives: Vec<CellScore>,
}

impl Report {
    /// Assembles a serializable [`Report`] from `result` and its validation output.
    fn new(
        image: &image::RgbaImage,
        result: &RecognitionResult,
        xsb: Option<String>,
        validation: &[String],
    ) -> Self {
        let mut cells = Vec::with_capacity((result.grid.rows * result.grid.columns) as usize);
        for (row, recognized) in result.cells.iter().enumerate() {
            for (column, cell) in recognized.iter().enumerate() {
                cells.push(ReportCell {
                    row,
                    column,
                    cell: cell.cell,
                    confidence: cell.confidence,
                    alternatives: cell.alternatives.clone(),
                });
            }
        }
        Self {
            width: image.width(),
            height: image.height(),
            grid: result.grid,
            quality: result.quality(),
            cells,
            xsb,
            valid: validation.is_empty(),
            relaxed: result.relaxed,
            skin: result.skin.clone(),
            validation: validation.to_vec(),
            warnings: result.warnings.clone(),
        }
    }
}

/// Outcome of running the CLI pipeline on a decoded image.
struct RunOutcome {
    /// Formatted stdout/file payload (`None` when XSB export failed on `Unknown` cells).
    content: Option<String>,
    /// `true` when XSB export succeeded and structural validation found no issues.
    complete_and_valid: bool,
    /// Underlying recognition result.
    result: RecognitionResult,
    /// Structural validation messages on the full recognized board.
    validation: Vec<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(NEEDS_ATTENTION),
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

/// Parses CLI arguments, runs recognition and writes output/skins.
fn run() -> Result<bool> {
    let arguments = Arguments::parse();
    let image = image::open(&arguments.image)
        .with_context(|| format!("failed to open {}", arguments.image.display()))?
        .to_rgba8();

    let outcome = execute(&arguments, &image)?;
    for warning in &outcome.result.warnings {
        eprintln!("warning: {warning}");
    }
    for warning in &outcome.validation {
        if !outcome.result.warnings.contains(warning) {
            eprintln!("warning: {warning}");
        }
    }
    if outcome.content.is_none() && arguments.format == Format::Xsb {
        report_unresolved(&outcome.result);
    }
    if let Some(content) = &outcome.content {
        write(arguments.output.as_deref(), content)?;
    }
    Ok(outcome.complete_and_valid)
}

/// Core CLI pipeline separated from `std::env::args` and `stderr` printing so
/// unit tests can exercise trimming, JSON formatting and skin persistence.
fn execute(arguments: &Arguments, image: &image::RgbaImage) -> Result<RunOutcome> {
    let loaded_skins = match arguments.skins.as_deref() {
        Some(path) if arguments.save_skins.as_deref() == Some(path) && !path.exists() => {
            Some(SkinLibrary::default())
        }
        Some(path) => Some(load_skin_library(path)?),
        None => None,
    };

    let result = recognize(image, &arguments.options(loaded_skins.clone()))?;

    // Validate the full board before trimming outer empty rows/columns so the
    // validity verdict never depends on `--trim` (matching `sokovision-gui`).
    let full_board = result.board();
    let validation = validate_board(&full_board);
    let export_board = if arguments.trim {
        full_board.trimmed()
    } else {
        full_board
    };
    let xsb = export_board.to_xsb().ok();

    if let Some(save_path) = arguments.save_skins.as_deref()
        && result.learnable()
    {
        let mut library = if arguments.skins.as_deref() == Some(save_path) {
            loaded_skins.unwrap_or_default()
        } else if save_path.exists() {
            load_skin_library(save_path)?
        } else {
            loaded_skins.unwrap_or_default()
        };
        let skin_name = arguments
            .image
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
            .map_or_else(
                || format!("Skin {}", library.skins.len() + 1),
                str::to_owned,
            );
        if let Some(skin) = result.skin(image, skin_name) {
            library.remember(skin);
            save_skin_library(save_path, &library)?;
        }
    }

    let content = match arguments.format {
        Format::Xsb => xsb.as_ref().map(|xsb| format!("{xsb}\n")),
        Format::Json => {
            let report = Report::new(image, &result, xsb.clone(), &validation);
            Some(format!("{}\n", serde_json::to_string_pretty(&report)?))
        }
    };

    Ok(RunOutcome {
        content,
        complete_and_valid: xsb.is_some() && validation.is_empty(),
        result,
        validation,
    })
}

/// Reads and prunes a [`SkinLibrary`] JSON file from `path`.
fn load_skin_library(path: &Path) -> Result<SkinLibrary> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut library: SkinLibrary = serde_json::from_str(&text)
        .with_context(|| format!("invalid skin library in {}", path.display()))?;
    library.prune();
    Ok(library)
}

/// Writes `library` as pretty-printed JSON to `path`, creating parent folders.
fn save_skin_library(path: &Path, library: &SkinLibrary) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    let json = format!("{}\n", serde_json::to_string_pretty(library)?);
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

/// Prints the first unresolved cell coordinates to standard error.
fn report_unresolved(result: &RecognitionResult) {
    let unknown = result.unknown_cells();
    let preview: Vec<String> = unknown
        .iter()
        .take(20)
        .map(|(row, column)| format!("({row}, {column})"))
        .collect();
    eprintln!(
        "error: {} cells are unresolved; first positions: {}",
        unknown.len(),
        preview.join(", ")
    );
    if unknown.len() > preview.len() {
        eprintln!("error: use --format json for the complete result");
    }
}

/// Writes `content` to `path` or standard output.
fn write(path: Option<&Path>, content: &str) -> Result<()> {
    match path {
        Some(path) => fs::write(path, content)
            .with_context(|| format!("failed to write {}", path.display()))?,
        None => print!("{content}"),
    }
    Ok(())
}

/// Parses `--region x,y,width,height`.
fn parse_region(value: &str) -> std::result::Result<PixelRect, String> {
    let [x, y, width, height] = split(value, "x,y,width,height")?;
    let region = PixelRect::new(
        number(x, "x")?,
        number(y, "y")?,
        number(width, "width")?,
        number(height, "height")?,
    );
    if region.is_empty() {
        return Err("region width and height must be greater than zero".to_owned());
    }
    Ok(region)
}

/// Parses `--grid x,y,cell_width,cell_height,rows,columns`.
fn parse_grid(value: &str) -> std::result::Result<Grid, String> {
    let [x, y, cell_width, cell_height, rows, columns] =
        split(value, "x,y,cell_width,cell_height,rows,columns")?;
    let rows = number(rows, "rows")?;
    let columns = number(columns, "columns")?;
    if rows == 0 || columns == 0 {
        return Err("rows and columns must be greater than zero".to_owned());
    }
    Ok(Grid::new(
        number(x, "x")?,
        number(y, "y")?,
        number(cell_width, "cell width")?,
        number(cell_height, "cell height")?,
        rows,
        columns,
    ))
}

/// Splits a comma separated argument into exactly `N` trimmed parts.
fn split<'a, const N: usize>(
    value: &'a str,
    expected: &str,
) -> std::result::Result<[&'a str; N], String> {
    let parts: Vec<&str> = value.split(',').map(str::trim).collect();
    <[&str; N]>::try_from(parts).map_err(|_| format!("{value:?} must be {expected}"))
}

/// Parses one numeric component of a CLI geometry flag.
fn number<T: std::str::FromStr>(value: &str, label: &str) -> std::result::Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("{label} must be a number, got {value:?}"))
}

#[cfg(test)]
mod tests {
    use image::{Rgba, RgbaImage};
    use sokovision_core::Board;

    use super::*;

    fn synthetic_board_image(board: &Board, tile: u32) -> RgbaImage {
        let mut image = RgbaImage::from_pixel(
            board.width() as u32 * tile,
            board.height() as u32 * tile,
            Rgba([232, 226, 210, 255]),
        );
        for (row, cells) in board.cells.iter().enumerate() {
            for (column, cell) in cells.iter().enumerate() {
                let x0 = column as u32 * tile;
                let y0 = row as u32 * tile;
                match cell {
                    Cell::Wall => {
                        for y in y0..y0 + tile {
                            for x in x0..x0 + tile {
                                let mortar =
                                    (x - x0).is_multiple_of(6) || (y - y0).is_multiple_of(8);
                                let color = if mortar {
                                    Rgba([145, 155, 185, 255])
                                } else {
                                    Rgba([40, 55, 95, 255])
                                };
                                image.put_pixel(x, y, color);
                            }
                        }
                    }
                    Cell::Player => {
                        for y in y0 + 5..y0 + tile - 5 {
                            for x in x0 + 6..x0 + tile - 6 {
                                image.put_pixel(x, y, Rgba([30, 150, 70, 255]));
                            }
                        }
                    }
                    Cell::Box => {
                        for y in y0 + 2..y0 + tile - 2 {
                            for x in x0 + 2..x0 + tile - 2 {
                                let cross = (x - x0 == 9) || (y - y0 == 9);
                                let color = if cross {
                                    Rgba([230, 200, 150, 255])
                                } else {
                                    Rgba([120, 75, 35, 255])
                                };
                                image.put_pixel(x, y, color);
                            }
                        }
                    }
                    Cell::Goal => {
                        for y in y0 + 8..y0 + tile - 8 {
                            for x in x0 + 8..x0 + tile - 8 {
                                image.put_pixel(x, y, Rgba([200, 30, 40, 255]));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        image
    }

    #[test]
    fn parses_region_and_grid_arguments_and_rejects_invalid_shapes() {
        assert_eq!(
            parse_region("10, 20, 300, 200").unwrap(),
            PixelRect::new(10, 20, 300, 200)
        );
        assert!(parse_region("10,20,0,200").is_err());
        assert!(parse_region("10,20,300").is_err());

        let grid = parse_grid("4.5, 8, 24, 24, 6, 8").unwrap();
        assert_eq!((grid.rows, grid.columns), (6, 8));
        assert!((grid.origin_x - 4.5).abs() < 1e-6);
        assert!(parse_grid("0,0,24,24,0,8").is_err());
    }

    #[test]
    fn trim_exports_compact_xsb_while_keeping_full_board_validation() {
        let board = Board::from_xsb(
            "#########--\n#       #--\n#@ $$ ..#--\n#       #--\n#########--\n-----------",
        )
        .unwrap();
        let tile = 24u32;
        let image = synthetic_board_image(&board, tile);
        let grid = Grid::new(0.0, 0.0, tile as f32, tile as f32, 6, 11);

        let untrimmed = Arguments::try_parse_from([
            "sokovision",
            "padded.png",
            "--grid",
            &format!("0,0,{tile},{tile},6,11"),
        ])
        .unwrap();
        let trimmed = Arguments::try_parse_from([
            "sokovision",
            "padded.png",
            "--grid",
            &format!("0,0,{tile},{tile},6,11"),
            "--trim",
        ])
        .unwrap();

        let out_full = execute(&untrimmed, &image).unwrap();
        let out_trim = execute(&trimmed, &image).unwrap();
        assert_eq!(out_full.validation, out_trim.validation);
        assert!(out_trim.complete_and_valid, "{:?}", out_trim.validation);
        assert_eq!(out_trim.result.grid, grid);
        assert_eq!(
            out_trim.content.as_deref(),
            Some("#########\n#-------#\n#@-$$-..#\n#-------#\n#########\n")
        );

        // Even when trimming collapses an all-floor board to 0x0, validation
        // inspects the full board so `--trim` never alters diagnostics.
        let empty = Board::from_xsb("----\n----\n----\n----").unwrap();
        let empty_img = synthetic_board_image(&empty, tile);
        let full_empty =
            Arguments::try_parse_from(["sokovision", "empty.png", "--grid", "0,0,24,24,4,4"])
                .unwrap();
        let trim_empty = Arguments::try_parse_from([
            "sokovision",
            "empty.png",
            "--grid",
            "0,0,24,24,4,4",
            "--trim",
        ])
        .unwrap();
        assert_eq!(
            execute(&full_empty, &empty_img).unwrap().validation,
            execute(&trim_empty, &empty_img).unwrap().validation
        );
    }

    #[test]
    fn cli_learns_and_reuses_skin_library_across_runs() {
        let board = Board::from_xsb("#######\n#@ $ .#\n#######").unwrap();
        let tile = 24u32;
        let image = synthetic_board_image(&board, tile);

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let skin_path = std::env::temp_dir().join(format!("sokovision-cli-test-{unique}.json"));

        let first = Arguments::try_parse_from([
            "sokovision",
            "demo_level.png",
            "--grid",
            "0,0,24,24,3,7",
            "--skins",
            skin_path.to_str().unwrap(),
            "--save-skins",
            skin_path.to_str().unwrap(),
        ])
        .unwrap();
        let first_outcome = execute(&first, &image).unwrap();
        assert!(first_outcome.complete_and_valid);
        assert!(skin_path.exists(), "valid board saved skin library");

        let second = Arguments::try_parse_from([
            "sokovision",
            "demo_level.png",
            "--grid",
            "0,0,24,24,3,7",
            "--format",
            "json",
            "--skins",
            skin_path.to_str().unwrap(),
        ])
        .unwrap();
        let second_outcome = execute(&second, &image).unwrap();
        let _ = fs::remove_file(&skin_path);

        assert!(second_outcome.complete_and_valid);
        assert_eq!(
            second_outcome
                .result
                .skin
                .as_ref()
                .map(|matched| matched.name.as_str()),
            Some("demo_level")
        );
        let json: serde_json::Value =
            serde_json::from_str(second_outcome.content.as_deref().unwrap()).unwrap();
        assert_eq!(json["valid"], true);
        assert_eq!(json["skin"]["name"], "demo_level");
    }
}
