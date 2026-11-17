use std::collections::HashMap;
use std::time::Instant;

use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use image::{ExtendedColorType, ImageEncoder, Rgba, RgbaImage, load_from_memory};
use sokovision_core::{Board, Cell, RecognizeOptions, SkinLibrary, recognize, validate_board};

const PLAIN: &str = "\
########-------
#---#--#-------
#...$$-#######-
#...#--------#-
#...#$######-#-
#...#------#-##
#####-$-$#-#--#
----#-$-$-$-#-#
----#--#-$--@-#
----#-$$-$#####
----#-----#----
----#######----";

const BRICK: &str = "\
#############
#---##------#
#-**$--*--#-#
#-----***-#-#
#-###--*@#--#
#--#--*****-#
#-***--#----#
#*-*--#.#**-#
#--**#-*--*-#
#--*-#-***--#
##-*-#-*--*-#
##-*##-***--#
##--------###
#############";

const YSOKOBAN_SIMPLIFIED: &str = "\
---------##########
---#######--------#
--##------*####*#-#
-##---*****-----*-#
-#--***-----##@-*-#
-#$**---****###-*-#
##-*--###----##-*-#
#--*-*##----$##-*-#
#-*.**-$*****---*-#
#-*-*---*-----*-*-#
#-#-*-#####****-*-#
#-#-*-#####---*-*-#
#-#-*-#####-$##-*-#
#-#-.-#####--##-*-#
#-#-*-######--#-.-#
#-##*-#-----#-#####
#-##--#.$.$-#-#----
#--*.*#-##-#--#----
#-------##---##----
##############-----";

const YSOKOBAN_SAMPLE: &str = "\
---------#####-
##########---#-
#------$-$$$-#-
#-$##-#--#-#-#-
#-$----#-#-#-#-
##$##.*#.*---#-
-#---.....-#-#-
-#$-##.@.##-$#-
-#-#-.....---#-
-#---*.#*.##$##
-#-#-#-#----$-#
-#-#-#--#-##$-#
-#-$$$-$------#
-#---##########
-#####---------";

const YSOKOBAN_SOLVED: &str = "\
####################-
#-................-#-
#-.###############-##
##.###--------------#
-#.##-$$$$$$$$$$$$$-#
-#.##-$...........$-#
-#.##-$.$$$$$$$$$.$-#
-#.##-$.$.......$.$-#
-#.##-$.$.$$$$$.$.$-#
-#.##-$.$.$-@-$.$.$-#
-#.##-$.$.$-#-$.$.$.#
-#.##-$.$.$---$.$.$-#
-#.##-$.$.$$$$$.$.$-#
-#.##-$.$.......$.$-#
-#.##-$.$$$$$$$$$.$-#
-#.##-$...........$-#
-#.---$$$$$$$$$$$$$-#
-#--#---------------#
-####################";

const DOTS: &str = "\
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

const DOS_RED_WALL: &str = "\
----#####-------------
----#---#-------------
----#*--#-------------
--###--*###-----------
--#--*--*-#-----------
###-#-###-#-----######
#---#-###-#######----#
#-*--*-------@-------#
#####-####-#-####----#
----#------###--######
----########----------";

struct Row {
    case: &'static str,
    variant: String,
    exact: bool,
    grid_shape: bool,
    correct: usize,
    total: usize,
    high_confidence_errors: usize,
    unknown: usize,
    relaxed: bool,
    validation: usize,
    confidence: f32,
    millis: u128,
}

/// Accuracy of cells bucketed by reported confidence, to check calibration.
#[derive(Default)]
struct Calibration {
    /// (correct, total) per bucket: >=0.9, >=0.7, >=0.5, <0.5.
    buckets: [(usize, usize); 4],
}

impl Calibration {
    fn add(&mut self, confidence: f32, correct: bool) {
        let bucket = if confidence >= 0.9 {
            0
        } else if confidence >= 0.7 {
            1
        } else if confidence >= 0.5 {
            2
        } else {
            3
        };
        self.buckets[bucket].1 += 1;
        if correct {
            self.buckets[bucket].0 += 1;
        }
    }

    fn report(&self, label: &str) {
        let names = [">=0.9", ">=0.7", ">=0.5", "<0.5"];
        let parts: Vec<String> = self
            .buckets
            .iter()
            .zip(names)
            .map(|(&(correct, total), name)| {
                format!(
                    "{name}={correct}/{total}({:.3})",
                    correct as f32 / total.max(1) as f32
                )
            })
            .collect();
        println!("{label} {}", parts.join(" "));
    }
}

fn main() {
    let cases: [(&str, &[u8], &str); 8] = [
        (
            "plain",
            include_bytes!("../tests/fixtures/plain_screenshot.jpg"),
            PLAIN,
        ),
        (
            "brick",
            include_bytes!("../tests/fixtures/brick_screenshot.png"),
            BRICK,
        ),
        (
            "scaled",
            include_bytes!("../tests/fixtures/scaled_screenshot.jpg"),
            BRICK,
        ),
        (
            "ysokoban-simplified",
            include_bytes!("../tests/fixtures/ysokoban_simplified.png"),
            YSOKOBAN_SIMPLIFIED,
        ),
        (
            "ysokoban-sample",
            include_bytes!("../tests/fixtures/ysokoban_sample.png"),
            YSOKOBAN_SAMPLE,
        ),
        (
            "ysokoban-solved",
            include_bytes!("../tests/fixtures/ysokoban_solved.png"),
            YSOKOBAN_SOLVED,
        ),
        (
            "xsokoban-dots",
            include_bytes!("../tests/fixtures/xsokoban_dots.jpeg"),
            DOTS,
        ),
        (
            "dos-red-wall",
            include_bytes!("../tests/fixtures/dos_red_wall.jpeg"),
            DOS_RED_WALL,
        ),
    ];

    println!(
        "case,variant,exact,grid_shape,correct,total,high_confidence_errors,unknown,relaxed,validation,confidence,ms"
    );
    let mut rows = Vec::new();
    let mut calibration = Calibration::default();
    for (case, bytes, expected) in cases {
        let image = load_from_memory(bytes).unwrap().to_rgba8();
        let expected = Board::from_xsb(expected).unwrap();
        for (variant, image) in variants(&image) {
            rows.push(evaluate(
                case,
                variant,
                &image,
                &expected,
                &RecognizeOptions::default(),
                &mut calibration,
            ));
        }
    }
    report("TOTAL", &rows);
    calibration.report("CALIBRATION");

    // Skin pass: learn a skin from each original, then re-read every perturbed
    // variant with that skin. Reveals whether remembered tiles help robustness
    // or pollute the read when the original was itself imperfect.
    let mut skin_rows = Vec::new();
    for (case, bytes, expected) in cases {
        let image = load_from_memory(bytes).unwrap().to_rgba8();
        let expected = Board::from_xsb(expected).unwrap();
        let images = variants(&image);
        let mut library = SkinLibrary::default();
        let original = recognize(&images[0].1, &RecognizeOptions::default()).unwrap();
        if let Some(skin) = original.skin(&images[0].1, case) {
            library.remember(skin);
        }
        let options = RecognizeOptions {
            skins: Some(library),
            ..RecognizeOptions::default()
        };
        for (variant, image) in images {
            skin_rows.push(evaluate(
                case,
                variant,
                &image,
                &expected,
                &options,
                &mut Calibration::default(),
            ));
        }
    }
    report("SKIN_TOTAL", &skin_rows);

    // Pollution pass: learn from a degraded (0.8 nearest) read, then read the
    // original back. A naive auto-learn stores the degraded guess; this shows
    // whether such an observation corrupts a board that normally reads exactly.
    let mut polluted_rows = Vec::new();
    for (case, bytes, expected) in cases {
        let image = load_from_memory(bytes).unwrap().to_rgba8();
        let expected = Board::from_xsb(expected).unwrap();
        let images = variants(&image);
        let source = recognize(&images[1].1, &RecognizeOptions::default()).unwrap();
        let mut library = SkinLibrary::default();
        if source.learnable()
            && let Some(skin) = source.skin(&images[1].1, case)
        {
            library.remember(skin);
        }
        let options = RecognizeOptions {
            skins: Some(library),
            ..RecognizeOptions::default()
        };
        polluted_rows.push(evaluate(
            case,
            "original-from-degraded".to_owned(),
            &images[0].1,
            &expected,
            &options,
            &mut Calibration::default(),
        ));
    }
    report("POLLUTION_TOTAL", &polluted_rows);
}

fn report(label: &str, rows: &[Row]) {
    for row in rows {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{:.4},{}",
            row.case,
            row.variant,
            row.exact,
            row.grid_shape,
            row.correct,
            row.total,
            row.high_confidence_errors,
            row.unknown,
            row.relaxed,
            row.validation,
            row.confidence,
            row.millis
        );
    }

    let total: usize = rows.iter().map(|row| row.total).sum();
    let correct: usize = rows.iter().map(|row| row.correct).sum();
    let exact = rows.iter().filter(|row| row.exact).count();
    let grid_shape = rows.iter().filter(|row| row.grid_shape).count();
    let high_confidence_errors: usize = rows.iter().map(|row| row.high_confidence_errors).sum();
    let unknown: usize = rows.iter().map(|row| row.unknown).sum();
    let relaxed = rows.iter().filter(|row| row.relaxed).count();
    let validation: usize = rows.iter().map(|row| row.validation).sum();
    let confidence = rows.iter().map(|row| row.confidence).sum::<f32>() / rows.len() as f32;
    let millis: u128 = rows.iter().map(|row| row.millis).sum();
    println!(
        "{label} cases={} exact={} grid_shape={} cell_accuracy={:.4} high_confidence_errors={} unknown={} relaxed={} validation={} mean_confidence={:.4} total_ms={}",
        rows.len(),
        exact,
        grid_shape,
        correct as f32 / total.max(1) as f32,
        high_confidence_errors,
        unknown,
        relaxed,
        validation,
        confidence,
        millis
    );
}

fn evaluate(
    case: &'static str,
    variant: String,
    image: &RgbaImage,
    expected: &Board,
    options: &RecognizeOptions,
    calibration: &mut Calibration,
) -> Row {
    let start = Instant::now();
    let result = recognize(image, options).unwrap();
    let millis = start.elapsed().as_millis();
    let actual = result.board();
    let expected_trimmed = expected.trimmed();
    let actual_trimmed = actual.trimmed();
    let exact = actual_trimmed.to_xsb_lossy() == expected_trimmed.to_xsb_lossy();
    let grid_shape = result.grid.rows as usize == expected.height()
        && result.grid.columns as usize == expected.width();
    let (correct, total, high_confidence_errors) = compare(expected, &actual, &result, calibration);
    let cells = result.cells.iter().flatten().count();
    let confidence = result
        .cells
        .iter()
        .flatten()
        .map(|cell| cell.confidence)
        .sum::<f32>()
        / cells.max(1) as f32;
    Row {
        case,
        variant,
        exact,
        grid_shape,
        correct,
        total,
        high_confidence_errors,
        unknown: result.unknown_cells().len(),
        relaxed: result.relaxed,
        validation: validate_board(&actual).len(),
        confidence,
        millis,
    }
}

fn compare(
    expected: &Board,
    actual: &Board,
    result: &sokovision_core::RecognitionResult,
    calibration: &mut Calibration,
) -> (usize, usize, usize) {
    let expected = expected.trimmed();
    let actual_trimmed = actual.trimmed();
    let (top, left) = content_origin(actual);
    let rows = expected.height().max(actual_trimmed.height());
    let columns = expected.width().max(actual_trimmed.width());
    let mut correct = 0;
    let mut high_confidence_errors = 0;
    for row in 0..rows {
        for column in 0..columns {
            let expected_cell = expected.get(row, column).unwrap_or(Cell::Empty);
            let actual_cell = actual_trimmed.get(row, column).unwrap_or(Cell::Empty);
            let recognized = result
                .cells
                .get(top + row)
                .and_then(|cells| cells.get(left + column));
            let is_correct = expected_cell == actual_cell;
            if let Some(cell) = recognized {
                calibration.add(cell.confidence, is_correct);
            }
            if is_correct {
                correct += 1;
                continue;
            }
            if recognized.is_some_and(|cell| cell.confidence >= 0.9) {
                high_confidence_errors += 1;
            }
        }
    }
    (correct, rows * columns, high_confidence_errors)
}

fn content_origin(board: &Board) -> (usize, usize) {
    let used = |cell: &Cell| *cell != Cell::Empty;
    let top = board
        .cells
        .iter()
        .position(|row| row.iter().any(used))
        .unwrap_or(0);
    let left = (0..board.width())
        .find(|&column| board.cells.iter().any(|row| used(&row[column])))
        .unwrap_or(0);
    (top, left)
}

fn variants(image: &RgbaImage) -> Vec<(String, RgbaImage)> {
    let width = image.width();
    let height = image.height();
    vec![
        ("original".to_owned(), image.clone()),
        (
            "scale-0.8-nearest".to_owned(),
            imageops::resize(
                image,
                (width * 4 / 5).max(1),
                (height * 4 / 5).max(1),
                FilterType::Nearest,
            ),
        ),
        (
            "scale-1.25-triangle".to_owned(),
            imageops::resize(
                image,
                (width * 5 / 4).max(1),
                (height * 5 / 4).max(1),
                FilterType::Triangle,
            ),
        ),
        ("jpeg-70".to_owned(), jpeg(image, 70)),
        ("dark-24".to_owned(), shift(image, -24, -24, -24)),
        ("warm".to_owned(), shift(image, 12, 0, -12)),
        ("pad-13".to_owned(), pad(image, 13)),
    ]
}

fn jpeg(image: &RgbaImage, quality: u8) -> RgbaImage {
    let rgb = image::DynamicImage::ImageRgba8(image.clone()).to_rgb8();
    let mut bytes = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut bytes, quality);
    encoder
        .write_image(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            ExtendedColorType::Rgb8,
        )
        .unwrap();
    load_from_memory(&bytes).unwrap().to_rgba8()
}

fn shift(image: &RgbaImage, red: i16, green: i16, blue: i16) -> RgbaImage {
    let mut shifted = image.clone();
    for pixel in shifted.pixels_mut() {
        pixel.0[0] = (pixel.0[0] as i16 + red).clamp(0, 255) as u8;
        pixel.0[1] = (pixel.0[1] as i16 + green).clamp(0, 255) as u8;
        pixel.0[2] = (pixel.0[2] as i16 + blue).clamp(0, 255) as u8;
    }
    shifted
}

fn pad(image: &RgbaImage, margin: u32) -> RgbaImage {
    let mut counts: HashMap<[u8; 4], usize> = HashMap::new();
    for pixel in image.pixels() {
        *counts.entry(pixel.0).or_default() += 1;
    }
    let background = counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(color, _)| color)
        .unwrap_or([0, 0, 0, 255]);
    let mut padded = RgbaImage::from_pixel(
        image.width() + margin * 2,
        image.height() + margin * 2,
        Rgba(background),
    );
    imageops::replace(&mut padded, image, margin as i64, margin as i64);
    padded
}
