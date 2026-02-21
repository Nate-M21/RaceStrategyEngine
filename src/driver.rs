use std::{
    collections::{HashMap, HashSet},
    mem,
};

use pyo3::FromPyObject;
use rand::rng;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};

use crate::{race_config::RaceConfiguration, race_simulation::DriverSimData};

#[derive(Debug, Clone, Default, FromPyObject, Serialize, Deserialize)]
pub struct DriverVariabilityParams {
    pub min_lap_time_variation: f64,
    pub lap_time_mean_deviation: f64,
    pub lap_time_std_dev: f64,
    pub mean_tyre_change_time: f64,
    pub std_dev_tyre_change_time: f64,
}

#[derive(Debug, Default, Clone, FromPyObject, Serialize, Deserialize)]
pub struct PitInfo {
    pub lap_entered_pits: u8,
    pub total_pit_loss_for_lap: f64,
    pub simulation_activated: bool,
}

pub struct CurrentStint {
    pub current_compound: String,
    pub current_tyre_age: u8,
    pub current_lap: u8,
}

// TODO make another struct that will interface with Python so in rust i can use native things such
// as Arc for tyre_models_baseline and driver variability as this dont change and cloning over threads is wasteful
#[derive(Debug, Default, Clone, FromPyObject, Serialize, Deserialize)]
pub struct Driver {
    pub name: String,

    agent_controlled: bool,

    tyre_models_baseline: HashMap<String, Vec<f64>>,
    driver_variabilty_params: DriverVariabilityParams,

    #[pyo3(default)]
    pub driver_position: u8,
    #[pyo3(default)]
    pub starting_position: u8,

    #[pyo3(default)]
    pub strategy: Vec<(String, u8)>,

    // Race progression
    #[pyo3(default = 1.0)]
    pub driver_race_progress: f64,
    #[pyo3(default = 1.0)]
    pub predicted_driver_race_progress: f64,
    #[pyo3(default = 1)]
    pub current_lap: u8,

    #[pyo3(default)]
    pub driver_laps_behind_traffic: u8,
    #[pyo3(default)]
    laps_in_traffic_set: HashSet<u8>,

    #[pyo3(default)]
    pub pit_stop: PitInfo,

    #[pyo3(default)]
    pub pitting_laps: Vec<u8>,
    #[pyo3(default)]
    pub out_laps: Vec<u8>,
    #[pyo3(default)]
    pub laps_pitted_on: Vec<u8>,
    #[pyo3(default)]
    pub driver_in_pit_lane: bool,
    #[pyo3(default)]
    pub time_spent_in_pit_lane: f64,
    #[pyo3(default)]
    pub precomputed_pit_losses: Vec<f64>,

    // DRS status
    #[pyo3(default)]
    pub drs_is_active: bool,
    #[pyo3(default)]
    pub drs_last_lap_checked: u8,
    #[pyo3(default)]
    pub drs_last_detection_point: f64,

    // times
    #[pyo3(default)]
    pub driver_race_time: f64,
    #[pyo3(default)]
    pub driver_accumulated_lap_times: Vec<f64>,
    #[pyo3(default = [0.0;78].to_vec())]
    pub driver_precomputed_lap_times: Vec<f64>,

    #[pyo3(default)]
    pub precomputed_failed_overtake_penalties: Vec<f64>,
    #[pyo3(default)]
    tyre_performance_benchmark: HashMap<String, f64>,
}

impl Driver {
    // #[allow(dead_code)]
    pub fn new(
        name: String,
        agent_controlled: bool,
        tyre_models: HashMap<String, Vec<f64>>,
        driver_variabilty_params: DriverVariabilityParams,
    ) -> Self {
        let tyre_performance_benchmark = tyre_models.keys().map(|key| (key.clone(), 0.0)).collect();

        Self {
            name,
            agent_controlled,
            strategy: Vec::with_capacity(10),
            current_lap: 1,
            driver_race_progress: 1.0,
            predicted_driver_race_progress: 1.0,

            tyre_performance_benchmark,
            tyre_models_baseline: tyre_models,
            driver_position: 0,
            starting_position: 0,
            driver_laps_behind_traffic: 0,

            // Bases on strategy
            pitting_laps: Vec::with_capacity(10),
            out_laps: Vec::with_capacity(10),
            precomputed_pit_losses: Vec::with_capacity(10),
            precomputed_failed_overtake_penalties: Vec::with_capacity(78),
            driver_precomputed_lap_times: [0.0; 78].to_vec(),

            laps_pitted_on: Vec::with_capacity(10),
            driver_in_pit_lane: false,
            time_spent_in_pit_lane: 0.0,

            drs_is_active: false,
            drs_last_lap_checked: 0,
            drs_last_detection_point: 0.0,

            driver_race_time: 0.0,
            driver_accumulated_lap_times: Vec::with_capacity(78),

            driver_variabilty_params,
            laps_in_traffic_set: HashSet::with_capacity(78),
            pit_stop: Default::default(),
        }
    }
    pub fn add_lap_behind_traffic(&mut self, current_lap: u8) {
        self.laps_in_traffic_set.insert(current_lap);
        self.driver_laps_behind_traffic = self.laps_in_traffic_set.len() as u8;
    }
    pub fn get_driver_lap_time(&self, lap: usize) -> f64 {
        let index = lap - 1;
        // when im testing rl stuff
        // let lap_time = self.driver_precomputed_lap_times[index].unwrap_or_else(|| panic!("No lap time value
        // for {} for lap: {}. This shouldn't happen unless I'm calculating lap times on the fly for an A.I agent
        // in the reinforcement learning environment. The lap driver entered pits {} the current lap progress is {} and race progress is {}",  self.name, lap, self.pit_stop.lap_entered_pits, self.get_current_lap_progress(), self.driver_race_progress));

        // let lap_time = self.driver_precomputed_lap_times[index].unwrap();

        let lap_time = self.driver_precomputed_lap_times[index];

        return lap_time;
    }

    pub fn get_current_lap_progress(&self) -> f64 {
        self.driver_race_progress.fract()
    }

    fn create_baseline_for_strategy(&self, strategy: &Vec<(String, u8)>) -> Vec<f64> {
        let mut baseline_lap_times = Vec::with_capacity(78);
        for (compound, stint_length) in strategy {
            let lap_times: &Vec<f64> = self.tyre_models_baseline.get(&compound.clone()).unwrap();
            let stint_length: usize = *stint_length as usize;

            let stint_slice: &[f64] = &lap_times[0..stint_length];
            baseline_lap_times.extend_from_slice(stint_slice);
        }

        baseline_lap_times
    }

    fn precompute_tyre_performance_benchmark(&mut self) {
        let driver_effect = self.get_driver_effect();
        if self.tyre_performance_benchmark.is_empty() {
            self.tyre_performance_benchmark = self
                .tyre_models_baseline
                .keys()
                .map(|key| (key.clone(), 0.0))
                .collect();
        }
        self.tyre_performance_benchmark
            .iter_mut()
            .for_each(|(compound, time_benchmark)| {
                *time_benchmark = self.tyre_models_baseline[compound][0] + driver_effect
            });
    }

    fn precompute_true_lap_times(
        &mut self,
        strategy: &Vec<(String, u8)>,
        race_config: &RaceConfiguration,
    ) {
        let num_laps = race_config.num_laps;
        let tyre_model_contribution = self.create_baseline_for_strategy(strategy);

        for current_lap in 1..=num_laps {
            let index = current_lap as usize - 1;
            let base_lap_time = tyre_model_contribution[index];

            let true_lap_time = self.compute_lap_time(current_lap, base_lap_time, race_config);

            self.driver_precomputed_lap_times[index] = true_lap_time;
        }
    }

    fn get_baseline_laptime(&self, compound: &str) -> &Vec<f64> {
        let compound_base_lap_times = self.tyre_models_baseline.get(compound).unwrap();
        compound_base_lap_times
    }

    pub fn get_current_stint(&self) -> CurrentStint {
        let cumalative_lap = match self.laps_pitted_on.last() {
            Some(last_pit_lap) => *last_pit_lap,
            None => 0,
        };

        let current_tyre_age = self.current_lap - cumalative_lap;

        let current_tyre_age = if current_tyre_age > 0 {
            current_tyre_age - 1
        } else {
            current_tyre_age
        };

        let index = self.laps_pitted_on.len();
        let current_compound = &self.strategy[index].0;
        let current_lap = self.current_lap;

        CurrentStint {
            current_compound: current_compound.to_string(),
            current_tyre_age,
            current_lap,
        }
    }

    pub fn calculate_lap_time(
        &self,
        compound: &str,
        tyre_age: u8,
        lap_num: u8,
        race_config: &RaceConfiguration,
    ) -> f64 {
        let base_lines = self.get_baseline_laptime(compound);
        let base_lap_time = base_lines[tyre_age as usize];

        let true_lap_time = self.compute_lap_time(lap_num, base_lap_time, race_config);

        true_lap_time
    }
    fn compute_lap_time(
        &self,
        current_lap: u8,
        base_lap_time: f64,
        race_config: &RaceConfiguration,
    ) -> f64 {
        let driver_effect = self.get_driver_effect();
        let fuel_effect = Driver::get_fuel_effect(current_lap, race_config);

        if current_lap == 1 {
            let race_start_effect =
                Driver::get_race_start_effect(self.starting_position, race_config);

            let true_lap_time = base_lap_time + race_start_effect + driver_effect + fuel_effect;

            return true_lap_time;
        } else {
            let true_lap_time = base_lap_time + driver_effect + fuel_effect;

            return true_lap_time;
        }
    }

    fn get_driver_effect(&self) -> f64 {
        let gauss = Normal::new(
            self.driver_variabilty_params.lap_time_mean_deviation,
            self.driver_variabilty_params.lap_time_std_dev,
        )
        .unwrap();
        let random_var = gauss.sample(&mut rng());

        let driver_effect = f64::max(
            self.driver_variabilty_params.min_lap_time_variation,
            random_var,
        );

        driver_effect
    }

    fn get_race_start_effect(driver_start_position: u8, race_config: &RaceConfiguration) -> f64 {
        let race_start_grid_position_time_effect =
            driver_start_position as f64 * race_config.race_start_grid_position_time_penalty;

        let race_start_effect =
            race_config.race_start_stationary_time_penalty + race_start_grid_position_time_effect;

        race_start_effect
    }

    pub fn get_lap_times(&self) -> Vec<f64> {
        let mut lap_times = Vec::with_capacity(78);
        for index in 0..self.driver_accumulated_lap_times.len() {
            if index == 0 {
                let lap_time = self.driver_accumulated_lap_times[index];
                lap_times.push(lap_time);
            } else {
                let previous_index = index - 1;
                let lap_time = self.driver_accumulated_lap_times[index]
                    - self.driver_accumulated_lap_times[previous_index];
                lap_times.push(lap_time);
            }
        }
        lap_times
    }
    fn precompute_pit_stop_info(
        &mut self,
        strategy: &Vec<(String, u8)>,
        race_config: &RaceConfiguration,
    ) {
        let pit_strategy = strategy[..strategy.len() - 1].to_vec();
        self.pitting_laps = Vec::with_capacity(10);
        let mut total = 0;
        self.out_laps = Vec::with_capacity(10);
        self.precomputed_pit_losses = Vec::with_capacity(10);
        for (_, stint_length) in pit_strategy {
            total += stint_length;
            self.pitting_laps.push(total);
            self.out_laps.push(total + 1);

            let pit_loss = self.calculate_pit_loss(race_config);
            self.precomputed_pit_losses.push(pit_loss);
        }
    }

    pub fn calculate_pit_loss(&self, race_config: &RaceConfiguration) -> f64 {
        let gauss = Normal::new(
            self.driver_variabilty_params.mean_tyre_change_time,
            self.driver_variabilty_params.std_dev_tyre_change_time,
        )
        .unwrap();
        let tyre_change_time = gauss.sample(&mut rng());

        let pit_loss = tyre_change_time + race_config.pit_lane_time_loss;

        pit_loss
    }

    fn precompute_overtake_penalties(&mut self, race_config: &RaceConfiguration) {
        // lap discrete simulation
        self.precomputed_failed_overtake_penalties = Vec::with_capacity(78);

        let min_pen = race_config.min_time_lost_due_to_failed_overtake_attempt;
        let max_pen = race_config.max_time_lost_due_to_failed_overtake_attempt;
        let num_laps = race_config.num_laps;
        let gauss = Uniform::new(min_pen, max_pen).unwrap();

        for _ in 1..=num_laps {
            let overtake_penalty = gauss.sample(&mut rng());

            self.precomputed_failed_overtake_penalties
                .push(overtake_penalty);
        }
    }

    fn select_strategy(&mut self, strategy: Vec<(String, u8)>, race_config: &RaceConfiguration) {
        self.precompute_overtake_penalties(race_config);
        self.precompute_true_lap_times(&strategy, race_config);
        self.precompute_pit_stop_info(&strategy, race_config);
        self.strategy = strategy;
    }

    /// Sets up a driver for simulation in the correct sequence to ensure accurate
    /// lap time calculations based on starting position.
    pub fn setup_for_simulation(
        &mut self,
        mut data: DriverSimData,
        race_config: &RaceConfiguration,
    ) {
        // Extract strategy before passing data to set_driver_starting_point
        let strategy = std::mem::take(&mut data.strategy);

        // Step 1: Set starting point first
        self.set_driver_starting_point(data);

        // Step 2: Then set strategy and calculate related values
        self.select_strategy(strategy, race_config);

        // Step 3: Then add driver effect to tyre model base lines for rl training
        self.precompute_tyre_performance_benchmark()
    }

    pub fn set_driver_starting_point(&mut self, start_point: DriverSimData) {
        self.driver_position = start_point.position;
        self.starting_position = start_point.starting_position;
        self.driver_race_progress = start_point.driver_race_progress;
        self.current_lap = start_point.driver_current_lap;
        self.driver_race_time = start_point.driver_race_time;
        self.driver_accumulated_lap_times = start_point.driver_accumulated_lap_times;
        self.driver_in_pit_lane = start_point.driver_in_pit_lane;
        self.time_spent_in_pit_lane = start_point.driver_time_spent_in_pit_lane;
    }

    pub fn get_all_compounds(&self) -> Vec<&String> {
        self.tyre_models_baseline.keys().collect::<Vec<&String>>()
    }

    pub fn clean_residuals(&mut self) {
        // ill call it reset later but for now

        *self = Self {
            name: mem::take(&mut self.name),
            tyre_models_baseline: mem::take(&mut self.tyre_models_baseline),
            driver_variabilty_params: mem::take(&mut self.driver_variabilty_params),
            driver_precomputed_lap_times: vec![0.0; 78],
            tyre_performance_benchmark: mem::take(&mut self.tyre_performance_benchmark),
            agent_controlled: self.agent_controlled,
            driver_race_progress: 1.0,
            ..Default::default()
        };
    }
}

impl Driver {
    pub fn get_starting_compound(strategy: Vec<(String, u8)>) -> String {
        let first_stint = &strategy[0];
        let starting_compound = first_stint.0.clone();

        starting_compound
    }

    pub fn precompute_agent_info(
        &mut self,
        starting_compound: &str,
        race_config: &RaceConfiguration,
    ) {
        self.precompute_overtake_penalties(race_config);
        self.precompute_tyre_performance_benchmark();

        let tyre_age = 0;
        let lap_num = 1;
        let lap_time = self.calculate_lap_time(starting_compound, tyre_age, lap_num, race_config);

        self.driver_precomputed_lap_times[0] = lap_time;

        let strategy = vec![(starting_compound.to_string(), lap_num)];

        self.strategy = strategy
    }

    pub fn get_tyre_performance(
        &self,
        raw_lap_time: f64,
        current_lap: u8,
        race_config: &RaceConfiguration,
    ) -> f64 {
        let fuel_effect = Driver::get_fuel_effect(current_lap, race_config);
        let race_start_effect = if current_lap == 1 {
            Driver::get_race_start_effect(self.starting_position, race_config)
        } else {
            0.0
        };

        let tyre_performance = raw_lap_time - fuel_effect - race_start_effect;

        tyre_performance
    }

    fn get_fuel_effect(current_lap: u8, race_config: &RaceConfiguration) -> f64 {
        let current_amount_of_fuel =
            race_config.total_fuel - race_config.fuel_consumption_per_lap * current_lap as f64;
        let fuel_effect = current_amount_of_fuel * race_config.fuel_effect_seconds_per_kg;

        fuel_effect
    }

    pub fn add_pit_stop(
        &mut self,
        compound: String,
        pitting_lap: u8,
        race_config: &RaceConfiguration,
    ) {
        // TODO stop agent from pittong on the final lap i think this is DSQ in F1 need to check
        // if self.driver_race_progress > race_config.num_laps.into() {
        //         return;
        //     }
        let index = self.laps_pitted_on.len();
        // if i already adding pit lap but now would to modify the exact lap instead of push just change it
        if self.pitting_laps.get(index).is_some() {
            self.pitting_laps[index] = pitting_lap;
            self.out_laps[index] = pitting_lap + 1;
            self.strategy.last_mut().unwrap().0 = compound; // change the compound if nessacary too

        // Dont add pit lap thats already there and also to prevent the agent from selecting a lap past the current
        } else if !self.laps_pitted_on.contains(&pitting_lap) {
            self.pitting_laps.push(pitting_lap);
            self.out_laps.push(pitting_lap + 1);
            let pit_loss = self.calculate_pit_loss(race_config);
            self.precomputed_pit_losses.push(pit_loss);
            self.add_stint_to_strategy(compound);
        }
    }

    pub fn remove_pit_stop(&mut self) {
        // if pitting laps has laps that havent been accomplished yet, remove the eddition
        let executed_stops = self.laps_pitted_on.len();
        let planned_stops = self.pitting_laps.len();
        if planned_stops > executed_stops {
            // self.pitting_laps.pop();
            // self.out_laps.pop();
            // self.precomputed_pit_losses.pop();
            // self.remove_stint_from_strategy();

            self.pitting_laps.truncate(executed_stops);
            self.out_laps.truncate(executed_stops);
            self.precomputed_pit_losses.truncate(executed_stops);
            self.strategy.truncate(executed_stops + 1);
        }
    }
    fn add_stint_to_strategy(&mut self, compound: String) {
        let strategy = (compound, 1);
        self.strategy.push(strategy);
    }
    #[allow(dead_code)]
    fn remove_stint_from_strategy(&mut self) {
        self.strategy.pop();
    }

    pub fn update_strategy(&mut self) {
        let index = self.laps_pitted_on.len();

        let CurrentStint {
            current_compound,
            current_tyre_age,
            current_lap: _,
        } = self.get_current_stint();

        let stint = (current_compound, current_tyre_age + 1);

        self.strategy[index] = stint
    }

    pub fn choose_next_lap_compound(
        &mut self,
        compound: &str,
        tyre_age: u8,
        race_config: &RaceConfiguration,
    ) {
        // Early return if we have finished the race, nothing left to do not even update strategy or for time discrete
        // if we are the starting or on the final lap you CANT choose next lap compound you are about to finish
        // exit early
        if self.current_lap > race_config.num_laps
            || self.driver_race_progress > race_config.num_laps.into()
        {
            return;
        }
        // self.update_strategy(); // update correctness of strategy until driver race is complete ie above condition

        let next_lap = self.current_lap + 1;

        let next_lap_lap_time = self.calculate_lap_time(compound, tyre_age, next_lap, race_config);

        let index_for_next_lap = self.current_lap as usize;
        // Due to start counting from 0 the index for the next lap is the number for this lap
        self.driver_precomputed_lap_times[index_for_next_lap] = next_lap_lap_time;
    }

    fn get_used_compounds_set(&self) -> HashSet<&String> {
        let len = self.laps_pitted_on.len();
        let current_strategy: &[(String, u8)] = &self.strategy[..=len];

        let used_compounds = current_strategy.iter().map(|(compound, _stint)| compound);
        let mut set_used_compounds = HashSet::with_capacity(4);
        for compound in used_compounds {
            set_used_compounds.insert(compound);
        }
        set_used_compounds
    }

    pub fn different_compounds_used_count(&self) -> u8 {
        let used_set = self.get_used_compounds_set();
        used_set.len() as u8
    }

    pub fn is_agent(&self) -> bool {
        self.agent_controlled
    }

    pub fn set_agent_status(&mut self, agent_controlled: bool) {
        self.agent_controlled = agent_controlled
    }

    pub fn pitted_previous_lap(&self) -> bool {
        let last_lap_pit = match self.laps_pitted_on.last() {
            Some(lap) => lap,
            None => return false, // if there is no last value means no pit stops
        };

        self.current_lap == last_lap_pit + 1
    }

    pub fn get_reference_lap_time(&self) -> f64 {
        let out_lap = match self.out_laps.last() {
            Some(lap) => *lap,
            None => 0,
        };

        self.driver_precomputed_lap_times[out_lap as usize]
    }

    pub fn get_tyre_models_fastest_time(&self, compound: &str) -> f64 {
        self.tyre_performance_benchmark[compound]
    }
}
