use pyo3::FromPyObject;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, FromPyObject, Default, Deserialize, Serialize)]
pub struct RaceConfiguration {
    pub pit_lane_time_loss: f64,
    pub num_laps: u8,

    pub drs_boost: f64,
    pub drs_activation_lap: u8,
    pub delta_for_drs_activation: f64,

    pub time_lost_due_to_being_overtaken: f64,
    pub time_lost_performing_overtake: f64,
    pub min_time_lost_due_to_failed_overtake_attempt: f64,
    pub max_time_lost_due_to_failed_overtake_attempt: f64,

    pub overtaking_threshold: f64,
    pub race_start_stationary_time_penalty: f64,
    pub race_start_grid_position_time_penalty: f64,

    pub total_fuel: f64,
    pub fuel_consumption_per_lap: f64,
    pub fuel_effect_seconds_per_kg: f64,
}
