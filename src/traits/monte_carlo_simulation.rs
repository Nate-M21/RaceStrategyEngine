use crate::driver::Driver;
use crate::race_config::RaceConfiguration;
use crate::race_simulation::{DriverResult, SimulationData};
use crate::utils::create_drivers_with_random_strategies;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use pyo3::pyclass;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;

#[pyclass]
pub struct MonteCarloResults {
    #[pyo3(get, set)]
    pub names: Vec<String>, //
    #[pyo3(get, set)]
    pub start_positions: Vec<u8>,
    #[pyo3(get, set)]
    pub positions: Vec<u8>,
    #[pyo3(get, set)]
    pub points: Vec<u8>,
    #[pyo3(get, set)]
    pub race_times: Vec<f64>,
    #[pyo3(get, set)]
    pub laps_behind_traffic: Vec<u8>,
    #[pyo3(get, set)]
    pub amount_of_stops: Vec<u8>,
    #[pyo3(get, set)]
    pub compounds_used: Vec<Vec<String>>,
    #[pyo3(get, set)]
    pub strategies: Vec<Vec<(String, u8)>>,
    #[pyo3(get, set)]
    laps_pitted_on: Vec<Vec<u8>>,
}

pub trait MonteCarloSimulation {
    fn run_monte_carlo_simulations(
        &self,
        simulation_data: SimulationData,
        alternate_strategies: HashMap<String, HashMap<String, Vec<Vec<(String, u8)>>>>,
        max_stops: u8,
        num_simulations: u64,
        method: &str,
    ) -> MonteCarloResults
    where
        Self: Sync,
    {
        let race_config = self.get_race_config();

        let pb = ProgressBar::new(num_simulations);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:100.yellow/blue}] {human_pos}/{human_len} ({per_sec}, ETA: {eta})")
            .unwrap()
            .progress_chars("█▓░"));

        let all_results: Vec<MonteCarloResults> = match method {
            "multi_core" => (0..num_simulations)
                .into_par_iter()
                .progress_with(pb)
                .map(|_x| {
                    self.run_single_monte_carlo_simulation(
                        &simulation_data,
                        &alternate_strategies,
                        max_stops,
                        race_config,
                    )
                })
                .collect(),
            "single_core" => (0..num_simulations)
                .into_iter()
                .progress_with(pb)
                .map(|_x| {
                    self.run_single_monte_carlo_simulation(
                        &simulation_data,
                        &alternate_strategies,
                        max_stops,
                        race_config,
                    )
                })
                .collect(),
            _ => panic!("'method must be either multi_core' or 'single_core' "),
        };

        let capacity = num_simulations as usize;
        let mut names = Vec::with_capacity(capacity);
        let mut start_positions = Vec::with_capacity(capacity);
        let mut positions = Vec::with_capacity(capacity);
        let mut race_times = Vec::with_capacity(capacity);
        let mut laps_behind_traffic = Vec::with_capacity(capacity);
        let mut strategies = Vec::with_capacity(capacity);
        let mut amount_of_stops = Vec::with_capacity(capacity);
        let mut compounds_used = Vec::with_capacity(capacity);
        let mut points = Vec::with_capacity(capacity);
        let mut laps_pitted_on = Vec::with_capacity(22);

        for mut single_result in all_results {
            names.append(&mut single_result.names);
            start_positions.append(&mut single_result.start_positions);
            positions.append(&mut single_result.positions);
            race_times.append(&mut single_result.race_times);
            laps_behind_traffic.append(&mut single_result.laps_behind_traffic);
            strategies.append(&mut single_result.strategies);
            amount_of_stops.append(&mut single_result.amount_of_stops);
            compounds_used.append(&mut single_result.compounds_used);
            points.append(&mut single_result.points);
            laps_pitted_on.append(&mut single_result.laps_pitted_on);
        }

        let mut pit_stop_laps = vec![Vec::with_capacity(capacity); max_stops as usize];

        for pitted_laps in laps_pitted_on {
            for (index, pitting_lap) in pitted_laps.iter().enumerate() {
                pit_stop_laps[index].push(*pitting_lap);
            }
        }

        let result = MonteCarloResults {
            names,
            start_positions,
            positions,
            race_times,
            laps_behind_traffic,
            amount_of_stops,
            compounds_used,
            strategies,
            points,
            laps_pitted_on: pit_stop_laps,
        };

        result
    }

    fn run_single_monte_carlo_simulation(
        &self,
        simulation_data: &SimulationData,
        alternate_strategies: &HashMap<String, HashMap<String, Vec<Vec<(String, u8)>>>>,
        max_stops: u8,
        race_config: &RaceConfiguration,
    ) -> MonteCarloResults {
        // Apply random strategies
        let drivers_hash_map = self.get_drivers_base_line_hashmaps();

        // TODO be able to change how strategies are sampled
        let drivers = create_drivers_with_random_strategies(
            simulation_data,
            alternate_strategies,
            drivers_hash_map,
            race_config,
        );

        let race_sim_drivers = self.run_single_simulation(drivers, race_config);

        let mut names = Vec::with_capacity(22);
        let mut start_positions = Vec::with_capacity(22);
        let mut positions = Vec::with_capacity(22);
        let mut points = Vec::with_capacity(22);
        let mut race_times = Vec::with_capacity(22);
        let mut laps_behind_traffic = Vec::with_capacity(22);
        let mut strategies = Vec::with_capacity(22);
        let mut amount_of_stops = Vec::with_capacity(22);
        let mut compounds_used = Vec::with_capacity(22);
        let mut laps_pitted_on = Vec::with_capacity(22);
        let max_stops_len = max_stops as usize;
        for mut driver in race_sim_drivers {
            names.push(driver.name);
            positions.push(driver.driver_position);
            race_times.push(driver.driver_race_time);
            laps_behind_traffic.push(driver.driver_laps_behind_traffic);
            start_positions.push(driver.starting_position);
            points.push(driver.points);
            let strategy = driver.strategy;
            let compounds: Vec<String> = strategy
                .iter()
                .map(|(compounds, _laps)| compounds.clone())
                .collect();
            let amount_stops = (strategy.len() - 1) as u8;

            compounds_used.push(compounds);
            amount_of_stops.push(amount_stops);
            strategies.push(strategy);
            driver.laps_pitted_on.resize(max_stops_len, 0);
            laps_pitted_on.push(driver.laps_pitted_on);
        }

        let result = MonteCarloResults {
            names,
            start_positions,
            positions,
            race_times,
            laps_behind_traffic,
            amount_of_stops,
            compounds_used,
            strategies,
            points,
            laps_pitted_on,
        };

        result
    }

    fn run_single_simulation(
        &self,
        drivers: Vec<Driver>,
        race_config: &RaceConfiguration,
    ) -> Vec<DriverResult>;

    fn get_drivers_base_line_hashmaps(&self) -> &HashMap<String, Driver>;

    fn get_race_config(&self) -> &RaceConfiguration;
}
