use sokovision_core::{RecognizeOptions, image::load_from_memory, recognize};

const EXPECTED: &str = "\
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

#[test]
fn recognizes_full_board_with_brick_wall_skin() {
    let bytes = include_bytes!("fixtures/brick_screenshot.png");
    let image = load_from_memory(bytes).unwrap().to_rgba8();
    let result = recognize(&image, &RecognizeOptions::default()).unwrap();

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
