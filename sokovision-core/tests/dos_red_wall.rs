use sokovision_core::{RecognizeOptions, image::load_from_memory, recognize};

/// A DOS skin where walls are red character glyphs drawn on a black floor.
///
/// The wall glyph is a repeating `$`-shaped stroke whose phase shifts from cell
/// to cell, so the wall appearance fragments into several clusters that do not
/// enclose anything on their own. The floor is pure black and therefore shares
/// its colour with the letterbox around the board. Both of those used to push
/// the walls into `Cell::Empty` and hand the floor anchor to a box cluster.
const EXPECTED: &str = "\
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

fn recognize_fixture() -> sokovision_core::RecognitionResult {
    let bytes = include_bytes!("fixtures/dos_red_wall.jpeg");
    let image = load_from_memory(bytes).unwrap().to_rgba8();
    recognize(&image, &RecognizeOptions::default()).unwrap()
}

#[test]
fn recognizes_character_walls_on_a_black_floor() {
    let result = recognize_fixture();

    assert!(
        result.unknown_cells().is_empty(),
        "unresolved cells: {:?}\nwarnings: {:?}",
        result.unknown_cells(),
        result.warnings
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|warning| warning.contains("cannot satisfy")),
        "constraint inference failed: {:?}",
        result.warnings
    );
    assert_eq!(result.board().to_xsb().unwrap(), EXPECTED);
}

/// The whole bug hinged on which appearance became the floor anchor: once the
/// black floor was mistaken for background, the anchor fell through to a box
/// cluster and every box turned into an empty cell.
#[test]
fn keeps_the_black_floor_as_the_floor_anchor() {
    let result = recognize_fixture();
    let xsb = result.board().to_xsb().unwrap();

    assert_eq!(xsb.matches('*').count(), 6, "{xsb}");
    assert_eq!(xsb.matches('@').count(), 1, "{xsb}");
    assert!(
        !xsb.contains('$') && !xsb.contains('.'),
        "boxes and goals should collapse onto one covered appearance: {xsb}"
    );
}

/// A single border-touching cell that happens to share the wall colour used to
/// win the wall vote outright, because the boundary term was not weighted by
/// how much of the board the appearance covers.
#[test]
fn keeps_the_bulk_of_the_red_wall_as_the_wall() {
    let result = recognize_fixture();
    let xsb = result.board().to_xsb().unwrap();

    assert_eq!(xsb.matches('#').count(), 79, "{xsb}");
}
