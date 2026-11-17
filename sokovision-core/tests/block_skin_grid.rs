use sokovision_core::{
    Cell, Grid, PixelRect, RecognitionResult, RecognizeOptions, image::load_from_memory, recognize,
};

fn detect_grid(bytes: &[u8]) -> Grid {
    let image = load_from_memory(bytes).unwrap().to_rgba8();
    let options = RecognizeOptions {
        grid_only: true,
        ..RecognizeOptions::default()
    };
    recognize(&image, &options).unwrap().grid
}

fn recognize_board(bytes: &[u8]) -> RecognitionResult {
    let image = load_from_memory(bytes).unwrap().to_rgba8();
    recognize(&image, &RecognizeOptions::default()).unwrap()
}

fn wall_mask(result: &RecognitionResult) -> Vec<String> {
    result
        .board()
        .to_xsb_lossy()
        .lines()
        .map(|line| {
            line.chars()
                .map(|cell| if cell == '#' { '#' } else { '-' })
                .collect()
        })
        .collect()
}

#[track_caller]
fn assert_grid(grid: Grid, rows: u32, columns: u32, cell: f32) {
    assert_eq!(grid.rows, rows, "{grid:?}");
    assert_eq!(grid.columns, columns, "{grid:?}");
    assert!(
        (grid.cell_width - cell).abs() <= 1.0,
        "cell width {} != {cell}: {grid:?}",
        grid.cell_width
    );
    assert!(
        (grid.cell_height - cell).abs() <= 1.0,
        "cell height {} != {cell}: {grid:?}",
        grid.cell_height
    );
}

#[test]
fn detects_grid_of_block_skin_with_chroma_only_walls() {
    let grid = detect_grid(include_bytes!("fixtures/xsokoban_brown.png"));
    assert_grid(grid, 11, 19, 40.0);
}

#[test]
fn detects_grid_of_sparse_skin_without_floor_lines() {
    let grid = detect_grid(include_bytes!("fixtures/xsokoban_blue.png"));
    assert_grid(grid, 11, 19, 40.0);
}

#[test]
fn detects_grid_of_scaled_block_skin() {
    let grid = detect_grid(include_bytes!("fixtures/xsokoban_silver.png"));
    assert_grid(grid, 11, 19, 36.0);
}

#[test]
fn recognizes_bevel_shaded_walls_as_one_material() {
    let silver = recognize_board(include_bytes!("fixtures/xsokoban_silver.png"));
    let blue = recognize_board(include_bytes!("fixtures/xsokoban_blue.png"));

    assert!(
        !silver
            .warnings
            .iter()
            .any(|warning| warning.contains("no wall-like")),
        "warnings: {:?}",
        silver.warnings
    );
    assert_eq!(wall_mask(&silver), wall_mask(&blue));
}

#[test]
fn re_detects_the_grid_inside_a_region() {
    let bytes = include_bytes!("fixtures/xsokoban_silver.png");
    let image = load_from_memory(bytes).unwrap().to_rgba8();
    let full = recognize(
        &image,
        &RecognizeOptions {
            grid_only: true,
            ..RecognizeOptions::default()
        },
    )
    .unwrap()
    .grid;

    let region = PixelRect::new(
        full.origin_x.round() as u32,
        full.origin_y.round() as u32,
        full.width().round() as u32,
        full.height().round() as u32,
    );
    let regional = recognize(
        &image,
        &RecognizeOptions {
            region: Some(region),
            grid_only: true,
            ..RecognizeOptions::default()
        },
    )
    .unwrap()
    .grid;

    assert_eq!(
        (regional.rows, regional.columns),
        (full.rows, full.columns),
        "regional {regional:?} vs full {full:?}"
    );
    assert!(
        (regional.cell_width - full.cell_width).abs() <= 1.0
            && (regional.cell_height - full.cell_height).abs() <= 1.0,
        "regional {regional:?} vs full {full:?}"
    );
    // A tight region is biased inwards by the inset wall sprites; the GUI
    // snaps the detected counts back onto the dragged region.
    assert!(
        (regional.origin_x - full.origin_x).abs() <= full.cell_width * 0.5
            && (regional.origin_y - full.origin_y).abs() <= full.cell_height * 0.5,
        "regional {regional:?} vs full {full:?}"
    );
}

#[test]
fn groups_cells_by_tile_appearance() {
    let result = recognize_board(include_bytes!("fixtures/xsokoban_blue.png"));

    let mut groups: std::collections::HashMap<Cell, Vec<u32>> = std::collections::HashMap::new();
    for row in &result.cells {
        for cell in row {
            if matches!(cell.cell, Cell::Goal | Cell::Box) {
                let appearance = cell
                    .appearance
                    .expect("inferred cells carry an appearance group");
                groups.entry(cell.cell).or_default().push(appearance);
            }
        }
    }

    for (cell, appearances) in [("goal", Cell::Goal), ("box", Cell::Box)] {
        let appearances = groups.get(&appearances).expect("both classes exist");
        assert_eq!(appearances.len(), 6, "{cell} count");
        assert!(
            appearances.iter().all(|group| *group == appearances[0]),
            "{cell} cells share one appearance: {appearances:?}"
        );
    }
    assert_ne!(
        groups[&Cell::Goal][0],
        groups[&Cell::Box][0],
        "goals and boxes must not share an appearance"
    );
}

#[test]
fn detects_grid_of_dithered_dos_skin() {
    let grid = detect_grid(include_bytes!("fixtures/dos_sokoban.png"));
    assert_grid(grid, 11, 22, 30.0);
}

#[test]
fn recognizes_tiny_dot_goals_on_flat_floor() {
    let result = recognize_board(include_bytes!("fixtures/xsokoban_dots.jpeg"));
    let expected = "\
############--
#..--#-----###
#..--#-$--$--#
#..--#$####--#
#..----@-##--#
#..--#-#--$-##
######-##$-$-#
--#-$--$-$-$-#
--#----#-----#
--############";
    assert_eq!(result.board().to_xsb_lossy().trim(), expected);
}
