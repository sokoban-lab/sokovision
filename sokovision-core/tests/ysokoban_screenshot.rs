use sokovision_core::{RecognizeOptions, image::load_from_memory, recognize};

fn recognize_fixture(bytes: &[u8]) -> String {
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
    result.board().to_xsb().unwrap()
}

#[test]
fn recognizes_flat_skin_with_many_boxes_on_goals() {
    let xsb = recognize_fixture(include_bytes!("fixtures/ysokoban_simplified.png"));
    assert_eq!(
        xsb,
        "\
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
##############-----"
    );
}

#[test]
fn recognizes_flat_skin_with_sparse_boxes() {
    let xsb = recognize_fixture(include_bytes!("fixtures/ysokoban_sample.png"));
    assert_eq!(
        xsb,
        "\
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
-#####---------"
    );
}

#[test]
fn recognizes_flat_skin_with_boxes_and_goals_untouched() {
    let xsb = recognize_fixture(include_bytes!("fixtures/ysokoban_solved.png"));
    assert_eq!(
        xsb,
        "\
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
-####################"
    );
}
