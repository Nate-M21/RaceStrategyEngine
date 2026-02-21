use std::{
    collections::HashMap,
    thread::sleep,
    time::{Duration, Instant},
};

use pyo3::{pyclass, pymethods};

use crate::{
    driver::Driver, lap_discrete_core::LapRaceSim, race_config::RaceConfiguration,
    race_simulation::SimulationData, real_time_strategy::RealTimeStrategy as RustRealTimeStrategy,
    time_discrete_core::TimeRaceSim,
};

enum RealTimeStrategyType {
    LapDiscrete(RustRealTimeStrategy<LapRaceSim>),
    TimeDiscrete(RustRealTimeStrategy<TimeRaceSim>),
}

#[pyclass]
pub struct RealTimeStrategy {
    race_engine: RealTimeStrategyType,
}

#[pymethods]
impl RealTimeStrategy {
    #[new]
    fn new(
        drivers: HashMap<String, Driver>,
        race_config: RaceConfiguration,
        simulation_type: &str,
        track_name: &str,
        time_step: f64,
    ) -> Self {
        let mut drivers_vec = Vec::with_capacity(22);
        for (_name, driver) in drivers.clone() {
            drivers_vec.push(driver);
        }

        let coretype: &str = &simulation_type;

        let engine = match coretype {
            "lap_discrete" => {
                let race_sim = LapRaceSim::new(drivers_vec, race_config);
                let engine = RustRealTimeStrategy::new(race_sim, drivers, race_config);
                RealTimeStrategyType::LapDiscrete(engine)
            }
            "time_discrete" => {
                let race_sim = TimeRaceSim::new(drivers_vec, race_config, track_name, time_step);
                let engine = RustRealTimeStrategy::new(race_sim, drivers, race_config);
                RealTimeStrategyType::TimeDiscrete(engine);
                panic!("Time discrete is currently not allowed");
            }
            _ => panic!("Select 'time_discrete' or 'lap_discrete"),
        };

        Self {
            race_engine: engine,
        }
    }

    fn start_strategy_engine(&mut self) {
        match &mut self.race_engine {
            RealTimeStrategyType::LapDiscrete(race_strategy_engine) => {
                race_strategy_engine.start_strategy_engine()
            }
            RealTimeStrategyType::TimeDiscrete(race_strategy_engine) => {
                race_strategy_engine.start_strategy_engine()
            }
        }
    }

    fn stop_strategy_engine(&mut self) {
        match &mut self.race_engine {
            RealTimeStrategyType::LapDiscrete(race_strategy_engine) => {
                race_strategy_engine.stop_strategy_engine()
            }
            RealTimeStrategyType::TimeDiscrete(race_strategy_engine) => {
                race_strategy_engine.stop_strategy_engine()
            }
        }
    }

    fn ingest_new_data(&mut self, new_simulation_data: SimulationData, current_lap: u8) {
        match &mut self.race_engine {
            RealTimeStrategyType::LapDiscrete(race_strategy_engine) => {
                race_strategy_engine.ingest_new_data(new_simulation_data, current_lap)
            }
            RealTimeStrategyType::TimeDiscrete(race_strategy_engine) => {
                race_strategy_engine.ingest_new_data(new_simulation_data, current_lap)
            }
        }
    }

    fn get_predictions(
        &self,
        wait_time_in_secs: f64,
        sleep_time_in_secs: f64,
    ) -> HashMap<String, Vec<(String, u8)>> {
        let start_time = Instant::now();

        while start_time.elapsed().as_secs_f64() < wait_time_in_secs {
            let results = match &self.race_engine {
                RealTimeStrategyType::LapDiscrete(race_strategy_engine) => {
                    race_strategy_engine.get_predictions()
                }
                RealTimeStrategyType::TimeDiscrete(race_strategy_engine) => {
                    race_strategy_engine.get_predictions()
                }
            };

            if let Some(predictions) = results {
                return predictions;
            }
            sleep(Duration::from_secs_f64(sleep_time_in_secs));
        }
        panic!(
            "Getting strategies took longer than {}s you set as wait time",
            wait_time_in_secs
        );
    }
}
