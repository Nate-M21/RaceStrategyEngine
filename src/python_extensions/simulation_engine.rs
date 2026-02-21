use std::collections::HashMap;

use pyo3::{Py, PyAny, Python, prelude::*, pyclass, pymethods, types::PyDict};

use crate::{
    driver::Driver,
    lap_discrete_core::LapRaceSim,
    race_config::RaceConfiguration,
    race_simulation::{DriverResult, RaceSimulation, SimulationData},
    time_discrete_core::TimeRaceSim,
    traits::{DisplayResult, MonteCarloResults, MonteCarloSimulation},
};

enum RaceSimType {
    LapDiscrete(RaceSimulation<LapRaceSim>),
    TimeDiscrete(RaceSimulation<TimeRaceSim>),
}
#[pyclass]
pub struct SimulationEngine {
    sim: RaceSimType,
}

#[pymethods]
impl SimulationEngine {
    #[new]
    pub fn new(
        drivers: HashMap<String, Driver>,
        race_config: RaceConfiguration,
        simulation_type: String,
        track_name: &str,
        time_step: f64,
    ) -> Self {
        let coretype: &str = &simulation_type;

        let mut drivers_vec = Vec::with_capacity(22);
        for (_name, driver) in drivers.clone() {
            drivers_vec.push(driver);
        }

        let race_sim_type = match coretype {
            "lap_discrete" => {
                let sim = LapRaceSim::new(drivers_vec, race_config);
                let race_sim = RaceSimulation::new(sim, drivers, race_config);
                RaceSimType::LapDiscrete(race_sim)
            }
            "time_discrete" => {
                let sim = TimeRaceSim::new(drivers_vec, race_config, track_name, time_step);
                let race_sim = RaceSimulation::new(sim, drivers, race_config);
                RaceSimType::TimeDiscrete(race_sim)
            }
            _ => panic!("Select 'time_discrete' or 'lap_discrete"),
        };

        Self { sim: race_sim_type }
    }

    pub fn run_simulation(
        &self,
        sim_data: SimulationData,
        show_results: bool,
        py: Python,
    ) -> Py<PyAny> {
        let result = match &self.sim {
            RaceSimType::LapDiscrete(race_simulation) => race_simulation.run_simulation(sim_data),
            RaceSimType::TimeDiscrete(race_simulation) => race_simulation.run_simulation(sim_data),
        };

        if show_results {
            match &self.sim {
                RaceSimType::LapDiscrete(race_simulation) => {
                    race_simulation.display_results(&result)
                }
                RaceSimType::TimeDiscrete(race_simulation) => {
                    race_simulation.display_results(&result)
                }
            }
        }

        SimulationEngine::get_results_py(py, &result)
    }

    pub fn run_monte_carlo_simulations(
        &self,
        simulation_data: SimulationData,
        alternate_strategies: HashMap<String, HashMap<String, Vec<Vec<(String, u8)>>>>,
        max_stops: u8,
        num_simulations: u64,
        method: &str,
    ) -> MonteCarloResults {
        let result = match &self.sim {
            RaceSimType::LapDiscrete(race_simulation) => race_simulation
                .run_monte_carlo_simulations(
                    simulation_data,
                    alternate_strategies,
                    max_stops,
                    num_simulations,
                    method,
                ),
            RaceSimType::TimeDiscrete(race_simulation) => race_simulation
                .run_monte_carlo_simulations(
                    simulation_data,
                    alternate_strategies,
                    max_stops,
                    num_simulations,
                    method,
                ),
        };

        // let names = result.start_positions,;

        // let result = MonteCarloResultsPython { names, start_positions: todo!(), positions: todo!(), points: todo!(), race_times: todo!(), laps_behind_traffic: todo!(), amount_of_stops: todo!(), compounds_used: todo!(), strategies: todo!(), laps_pitted_on: todo!() };

        result
    }
}

impl SimulationEngine {
    fn get_results_py(py: Python, drivers: &Vec<DriverResult>) -> Py<PyAny> {
        let race_result = PyDict::new(py);
        for driver in drivers {
            let driver_result = PyDict::new(py);
            driver_result
                .set_item("position", driver.driver_position)
                .expect("Failed to set position");
            driver_result
                .set_item("total_time", driver.driver_race_time)
                .expect("Failed to set total_time");

            race_result
                .set_item(driver.name.clone(), driver_result)
                .expect("Failed to set driver information to race result dict");
        }

        race_result.into()
    }
}
