use std::collections::HashMap;

use crate::{
    driver::Driver,
    race_config::RaceConfiguration,
    race_simulation::{DriverParameters, SimulationResult},
};

pub trait RaceSimulationCore: Clone + Send + 'static {
    fn run_simulation(&mut self) -> &[Driver];

    fn set_simulation_starting_point(&mut self);

    fn incorporate_driver_data(&mut self, simulation_data_map: HashMap<String, DriverParameters>);

    fn get_simulation_result(&mut self) -> SimulationResult;

    fn create_new_simulation(&self, drivers: Vec<Driver>) -> Self;

    fn race_config(&self) -> &RaceConfiguration;
}
