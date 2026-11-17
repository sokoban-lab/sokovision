use sokovision_core::{
    Board, Cell, Grid, RecognitionResult, RecognizeOptions, SkinLibrary,
    image::{Rgba, RgbaImage, imageops, load_from_memory},
    recognize,
};

fn load(bytes: &[u8]) -> RgbaImage {
    load_from_memory(bytes).unwrap().to_rgba8()
}

fn recognize_default(image: &RgbaImage) -> RecognitionResult {
    recognize(image, &RecognizeOptions::default()).unwrap()
}

fn recognize_with(image: &RgbaImage, library: SkinLibrary) -> RecognitionResult {
    recognize(
        image,
        &RecognizeOptions {
            skins: Some(library),
            ..RecognizeOptions::default()
        },
    )
    .unwrap()
}

fn remember(image: &RgbaImage, result: &RecognitionResult, name: &str) -> SkinLibrary {
    let mut library = SkinLibrary::default();
    library
        .remember(result.skin(image, name).expect("the board has tiles"))
        .expect("the skin has tiles");
    library
}

#[test]
fn a_remembered_skin_recognizes_another_level_of_the_same_game() {
    let source = load(include_bytes!("fixtures/ysokoban_simplified.png"));
    let library = remember(&source, &recognize_default(&source), "ySokoban");

    let fixtures: [&[u8]; 2] = [
        include_bytes!("fixtures/ysokoban_sample.png"),
        include_bytes!("fixtures/ysokoban_solved.png"),
    ];
    for bytes in fixtures {
        let image = load(bytes);
        let baseline = recognize_default(&image);
        let result = recognize_with(&image, library.clone());

        let matched = result
            .skin
            .as_ref()
            .expect("the same skin should match another level");
        assert_eq!(matched.name, "ySokoban");
        assert!(matched.matched >= 2, "{matched:?}");
        assert_eq!(
            result.board(),
            baseline.board(),
            "a skin that agrees with the pixels must not change the board"
        );
    }
}

#[test]
fn a_remembered_skin_survives_a_different_resolution() {
    let source = load(include_bytes!("fixtures/brick_screenshot.png"));
    let library = remember(&source, &recognize_default(&source), "brick");

    let scaled = load(include_bytes!("fixtures/scaled_screenshot.jpg"));
    let baseline = recognize_default(&scaled);
    let result = recognize_with(&scaled, library);

    assert!(
        result.skin.is_some(),
        "the same skin at another scale should match"
    );
    assert_eq!(result.board(), baseline.board());
}

/// Stored prototypes stay coarse, so a learned library stays small and cheap
/// to write.
#[test]
fn learned_prototypes_are_coarse() {
    let image = load(include_bytes!("fixtures/plain_screenshot.jpg"));
    let skin = recognize_default(&image)
        .skin(&image, "coarse")
        .expect("tiles");

    assert!(!skin.tiles.is_empty());
    for tile in skin.tiles {
        assert_eq!(tile.color.len(), 4 * 4 * 3, "{:?}", tile.cell);
    }
}

#[test]
fn a_remembered_skin_does_not_apply_to_another_game() {
    let source = load(include_bytes!("fixtures/brick_screenshot.png"));
    let library = remember(&source, &recognize_default(&source), "brick");

    let other = load(include_bytes!("fixtures/xsokoban_blue.png"));
    let baseline = recognize_default(&other);
    let result = recognize_with(&other, library);

    assert!(result.skin.is_none(), "{:?}", result.skin);
    assert_eq!(result.board(), baseline.board());
}

/// The library is a strong prior: when it says a tile is something else, the
/// classification follows it. This is the mechanism that makes a corrected
/// skin beat the visual guess on later levels.
#[test]
fn remembered_labels_win_over_the_visual_guess() {
    // An open room with one box and one goal, both away from wall corners, so
    // swapping the two labels stays a legal Sokoban board.
    let level = "\
########
#------#
#@$-.-##
#------#
########";
    let (image, grid) = render(level);
    let options = RecognizeOptions {
        grid: Some(grid),
        ..RecognizeOptions::default()
    };
    let baseline = recognize(&image, &options).unwrap();
    assert_eq!(baseline.board().to_xsb().unwrap(), level);

    let mut skin = baseline.skin(&image, "swapped").expect("tiles");
    for tile in &mut skin.tiles {
        tile.cell = match tile.cell {
            Cell::Box => Cell::Goal,
            Cell::Goal => Cell::Box,
            other => other,
        };
    }
    let mut library = SkinLibrary::default();
    library.remember(skin).expect("the skin has tiles");

    let result = recognize(
        &image,
        &RecognizeOptions {
            skins: Some(library),
            ..options
        },
    )
    .unwrap();
    let swapped = |cell: Cell| match cell {
        Cell::Box => Cell::Goal,
        Cell::Goal => Cell::Box,
        other => other,
    };
    let expected: Vec<Vec<Cell>> = baseline
        .board()
        .cells
        .iter()
        .map(|row| row.iter().copied().map(swapped).collect())
        .collect();

    assert_eq!(result.board().cells, expected);
}

fn render(level: &str) -> (RgbaImage, Grid) {
    const CELL: u32 = 24;
    let board = Board::from_xsb(level).unwrap();
    let grid = Grid::new(
        0.0,
        0.0,
        CELL as f32,
        CELL as f32,
        board.height() as u32,
        board.width() as u32,
    );
    let mut image = RgbaImage::from_pixel(
        board.width() as u32 * CELL,
        board.height() as u32 * CELL,
        Rgba([0, 0, 0, 255]),
    );
    for (row, cells) in board.cells.iter().enumerate() {
        for (column, cell) in cells.iter().enumerate() {
            imageops::replace(
                &mut image,
                &drawn_tile(*cell),
                (column as u32 * CELL) as i64,
                (row as u32 * CELL) as i64,
            );
        }
    }
    (image, grid)
}

fn drawn_tile(cell: Cell) -> RgbaImage {
    let mut image = RgbaImage::from_pixel(24, 24, Rgba([232, 226, 210, 255]));
    match cell {
        Cell::Wall => {
            image = RgbaImage::from_pixel(24, 24, Rgba([40, 55, 95, 255]));
            for x in (0..24).step_by(6) {
                for y in 0..24 {
                    image.put_pixel(x, y, Rgba([150, 160, 190, 255]));
                }
            }
            for y in (0..24).step_by(8) {
                for x in 0..24 {
                    image.put_pixel(x, y, Rgba([120, 130, 170, 255]));
                }
            }
        }
        Cell::Goal => {
            for y in 8..16 {
                for x in 8..16 {
                    image.put_pixel(x, y, Rgba([200, 30, 40, 255]));
                }
            }
        }
        Cell::Box => {
            image = RgbaImage::from_pixel(24, 24, Rgba([120, 75, 35, 255]));
            for index in 4..20 {
                for width in 0..2 {
                    image.put_pixel(index, 8 + width, Rgba([230, 200, 150, 255]));
                    image.put_pixel(8 + width, index, Rgba([230, 200, 150, 255]));
                }
            }
        }
        Cell::Player | Cell::PlayerOnGoal => {
            for y in 5..19 {
                for x in 6..18 {
                    let dx = (x as f32 - 11.5) / 6.0;
                    let dy = (y as f32 - 11.5) / 7.0;
                    if dx * dx + dy * dy <= 1.0 {
                        image.put_pixel(x, y, Rgba([30, 150, 70, 255]));
                    }
                }
            }
            for y in 8..13 {
                for x in 9..15 {
                    image.put_pixel(x, y, Rgba([240, 200, 60, 255]));
                }
            }
        }
        Cell::Empty | Cell::BoxOnGoal | Cell::Unknown => {}
    }
    image
}
