#![recursion_limit = "256"]

use std::collections::HashMap;

use pyo3::prelude::*;

pub mod algorithms;
mod driver;
mod drs;
pub mod environment;
pub mod lap_discrete_core;
pub mod live_data;
pub mod python_extensions;
pub mod race_config;
pub mod race_simulation;
pub mod racetrack;
pub mod real_time_strategy;
pub mod time_discrete_core;
pub mod traits;
pub mod utils;

use driver::Driver;
use race_config::RaceConfiguration;
use race_simulation::SimulationData;
use utils::{save_complete_simulation_state, smart_round as internal_smart_round};

use crate::python_extensions::{real_time_strategy_engine::RealTimeStrategy, simulation_engine::SimulationEngine, strategy_environment::RaceStrategyEnvironment};

/// Scales values to a target sum, optionally considering their relative contributions.
#[pyfunction]
fn smart_round(list: Vec<f64>, target_sum: i32, scale_by_contribution: bool) -> Vec<i32> {
    internal_smart_round(list, target_sum, scale_by_contribution)
}

#[pyfunction]
fn save_test_data(
    race_config: RaceConfiguration,
    drivers: HashMap<String, Driver>,
    sim_data: SimulationData,
    path: String,
) -> PyResult<()> {
    save_complete_simulation_state(&race_config, &drivers, &sim_data, &path)?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn strategy_engine_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smart_round, m)?)?;
    m.add_class::<SimulationEngine>()?;
    m.add_class::<RealTimeStrategy>()?;
    m.add_class::<RaceStrategyEnvironment>()?;
    m.add_function(wrap_pyfunction!(save_test_data, m)?)?;
    Ok(())
}
