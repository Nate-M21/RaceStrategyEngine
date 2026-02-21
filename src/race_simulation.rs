use pyo3::FromPyObject;
use serde::{Deserialize, Serialize};

use crate::{
    driver::Driver,
    race_config::RaceConfiguration,
    traits::{DisplayResult, MonteCarloSimulation, RaceSimulationCore},
};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub enum Coretype {
    LapDiscrete,
    TimeDiscrete,
}
#[derive(Debug, FromPyObject, Clone, Serialize, Deserialize)]
pub struct DriverParameters {
    pub name: String,
    pub position: u8,
    pub starting_position: u8,
    pub strategy: Vec<(String, u8)>,

    #[pyo3(default = 1)]
    pub driver_current_lap: u8,
    #[pyo3(default = 0.0)]
    pub driver_race_time: f64,

    #[pyo3(default = false)]
    pub driver_in_pit_lane: bool,
    #[pyo3(default = 0.0)]
    pub driver_time_spent_in_pit_lane: f64,

    #[pyo3(default = 1.0)]
    pub driver_race_progress: f64,
    #[pyo3(default)]
    pub driver_accumulated_lap_times: Vec<f64>,
}

impl DriverParameters {
    pub fn into_sim_data(self) -> DriverSimData {
        DriverSimData {
            position: self.position,
            starting_position: self.starting_position,
            strategy: self.strategy,
            driver_current_lap: self.driver_current_lap,
            driver_race_time: self.driver_race_time,
            driver_in_pit_lane: self.driver_in_pit_lane,
            driver_time_spent_in_pit_lane: self.driver_time_spent_in_pit_lane,
            driver_race_progress: self.driver_race_progress,
            driver_accumulated_lap_times: self.driver_accumulated_lap_times,
        }
    }
}

pub struct DriverSimData {
    pub position: u8,
    pub starting_position: u8,
    pub strategy: Vec<(String, u8)>,

    pub driver_current_lap: u8,
    pub driver_race_time: f64,

    pub driver_in_pit_lane: bool,
    pub driver_time_spent_in_pit_lane: f64,

    pub driver_race_progress: f64,
    pub driver_accumulated_lap_times: Vec<f64>,
}
impl DriverSimData {
    pub fn new_from_params(
        driver_params: &DriverParameters,
        strategy: Vec<(String, u8)>,
        driver_accumulated_lap_times: Vec<f64>,
    ) -> DriverSimData {
        Self {
            position: driver_params.position,
            starting_position: driver_params.starting_position,
            strategy,
            driver_current_lap: driver_params.driver_current_lap,
            driver_race_time: driver_params.driver_race_time,
            driver_in_pit_lane: driver_params.driver_in_pit_lane,
            driver_time_spent_in_pit_lane: driver_params.driver_time_spent_in_pit_lane,
            driver_race_progress: driver_params.driver_race_progress,
            driver_accumulated_lap_times,
        }
    }
}

pub type SimulationData = Vec<DriverParameters>;

pub struct DriverResult {
    pub name: String,
    pub starting_position: u8,
    pub driver_position: u8,
    pub driver_race_time: f64,
    pub points: u8,
    pub strategy: Vec<(String, u8)>,
    pub driver_laps_behind_traffic: u8,
    pub laps_pitted_on: Vec<u8>,
}

pub type SimulationResult = Vec<DriverResult>;

#[derive(Clone)]
pub struct RaceSimulation<R: RaceSimulationCore> {
    pub race_sim_core: R,
    drivers: HashMap<String, Driver>,
    race_config: RaceConfiguration,
}

impl<R: RaceSimulationCore> RaceSimulation<R> {
    pub fn new(
        race_sim: R,
        drivers: HashMap<String, Driver>,
        race_config: RaceConfiguration,
    ) -> Self {
        Self {
            race_sim_core: race_sim,
            drivers,
            race_config,
        }
    }

    pub fn run_simulation(&self, simulation_data: SimulationData) -> SimulationResult {
        let drivers = self.incorp_data(simulation_data);

        self.core_simulation(drivers)
    }

    fn core_simulation(&self, drivers: Vec<Driver>) -> SimulationResult {
        let mut new_sim = self.race_sim_core.create_new_simulation(drivers);

        new_sim.set_simulation_starting_point();

        new_sim.run_simulation();
        let result = new_sim.get_simulation_result();

        result
    }

    fn incorp_data(&self, simulation_data: SimulationData) -> Vec<Driver> {
        let drivers_hash_map = &self.drivers;

        let mut drivers: Vec<Driver> = Vec::with_capacity(22);

        for data in simulation_data {
            let mut driver = drivers_hash_map[&data.name].clone();

            driver.setup_for_simulation(data.into_sim_data(), &self.race_config);

            drivers.push(driver);
        }

        drivers
    }
}

impl<R: RaceSimulationCore> MonteCarloSimulation for RaceSimulation<R> {
    fn get_drivers_base_line_hashmaps(&self) -> &HashMap<String, Driver> {
        &self.drivers
    }

    fn get_race_config(&self) -> &RaceConfiguration {
        &self.race_config
    }

    fn run_single_simulation(
        &self,
        drivers: Vec<Driver>,
        _race_config: &RaceConfiguration,
    ) -> SimulationResult {
        self.core_simulation(drivers)
    }
}

impl<R: RaceSimulationCore> DisplayResult for RaceSimulation<R> {}
