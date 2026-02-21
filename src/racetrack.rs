use std::fs;

use crate::{drs::DRS, race_config::RaceConfiguration};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};

use serde::{Deserialize, Serialize};
use toml;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RaceTrack {
    pub pit_entry: f64,
    pub pit_lane_displacement: f64,
    pub drs: DRS,
}

#[derive(Debug, EnumString, Display, EnumIter)]
#[strum(serialize_all = "snake_case")]
enum Tracks {
    Melbourne,   // Australia
    Shanghai,    // China
    Suzuka,      // Japan
    Sakhir,      // Bahrain
    Jeddah,      // Saudi Arabia
    Miami,       // USA
    Imola,       // Italy
    Monaco,      // Monaco
    Barcelona,   // Spain
    Montreal,    // Canada
    Spielberg,   // Austria
    Silverstone, // United Kingdom
    Spa,         // Belgium
    Budapest,    // Hungary
    Zandvoort,   // Netherlands
    Monza,       // Italy
    Baku,        // Azerbaijan
    Singapore,   // Singapore
    Austin,      // USA
    MexicoCity,  // Mexico
    SaoPaulo,    // Brazil
    LasVegas,    // USA
    Lusail,      // Qatar
    AbuDhabi,    // United Arab Emirates
}

impl RaceTrack {
    pub fn create_race_track(track_name: &str, race_config: RaceConfiguration) -> RaceTrack {
        let contents = get_track(track_name);

        let mut race_track: RaceTrack =
            toml::from_str(&contents).expect("Failed to create track from toml config");

        race_track.drs.delta_for_drs_activation = race_config.delta_for_drs_activation;
        race_track.drs.drs_activation_lap = race_config.drs_activation_lap;

        race_track
    }

    fn show_all_tracks() -> String {
        let track_names = Tracks::iter()
            .map(|t| t.to_string())
            .collect::<Vec<String>>();
        let track_names = track_names.join("\n");

        track_names
    }

    fn get_track_name(user_input: &str) -> Tracks {
        let track_name = user_input
            .to_lowercase()
            .parse::<Tracks>()
            .unwrap_or_else(|_| {
                panic!(
                    "Sorry, could not find track. Here are the availiable tracks:\n\n{}\n",
                    RaceTrack::show_all_tracks()
                )
            });

        track_name
    }
    #[allow(dead_code, unused_variables)]
    fn read_track_from_file(track: &str) {
        let track_name = RaceTrack::get_track_name(track);

        let project_root = env!("CARGO_MANIFEST_DIR");
        println!("{project_root}");
        let file_path = format!("{}/src/tracks/{track_name}.toml", project_root);
        println!("{file_path}");

        let contents = fs::read_to_string(&file_path).expect(&format!(
            "Could not get track: {} at {}",
            track_name, file_path
        ));
    }
}
// TODO
// THIS IS HORRIFIC, I REALLY NEED TO REMOVE THIS SOON.
fn get_track(user_input: &str) -> &str {
    match RaceTrack::get_track_name(user_input) {
        Tracks::Melbourne => include_str!("tracks/melbourne.toml"),
        Tracks::Shanghai => include_str!("tracks/shanghai.toml"),
        Tracks::Suzuka => include_str!("tracks/suzuka.toml"),
        Tracks::Sakhir => include_str!("tracks/sakhir.toml"),
        Tracks::Jeddah => include_str!("tracks/jeddah.toml"),
        Tracks::Miami => include_str!("tracks/miami.toml"),
        Tracks::Imola => include_str!("tracks/imola.toml"),
        Tracks::Monaco => include_str!("tracks/monaco.toml"),
        Tracks::Barcelona => include_str!("tracks/barcelona.toml"),
        Tracks::Montreal => include_str!("tracks/montreal.toml"),
        Tracks::Spielberg => include_str!("tracks/spielberg.toml"),
        Tracks::Silverstone => include_str!("tracks/silverstone.toml"),
        Tracks::Spa => include_str!("tracks/spa.toml"),
        Tracks::Budapest => include_str!("tracks/budapest.toml"),
        Tracks::Zandvoort => include_str!("tracks/zandvoort.toml"),
        Tracks::Monza => include_str!("tracks/monza.toml"),
        Tracks::Baku => include_str!("tracks/baku.toml"),
        Tracks::Singapore => include_str!("tracks/singapore.toml"),
        Tracks::Austin => include_str!("tracks/austin.toml"),
        Tracks::MexicoCity => include_str!("tracks/mexico_city.toml"),
        Tracks::SaoPaulo => include_str!("tracks/sao_paulo.toml"),
        Tracks::LasVegas => include_str!("tracks/las_vegas.toml"),
        Tracks::Lusail => include_str!("tracks/lusail.toml"),
        Tracks::AbuDhabi => include_str!("tracks/abu_dhabi.toml"),
    }
}
