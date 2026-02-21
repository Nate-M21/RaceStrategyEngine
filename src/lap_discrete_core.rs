use std::collections::HashMap;
use std::mem;

use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::driver::{CurrentStint, Driver};
use crate::race_config::RaceConfiguration;
use crate::race_simulation::{DriverParameters, DriverResult};
use crate::traits::{
    DisplayResult, DriverObservation, RaceCompliance, RaceSimulationCore,
    RaceStrategyEnvironmentCore,
};
use crate::utils::get_f1_points;

struct PittingDriver {
    name: String,
    race_time: f64,
    old_position: i8,
    new_position: Option<i8>,
    positions_to_move: Option<i8>,
    driver_index: usize,
}
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct LapRaceSim {
    pub drivers: Vec<Driver>,
    pub race_config: RaceConfiguration,
    starting_lap: u8,
    current_lap: u8, // For RL use, just easier to grasp then seeing starting_lap change throughtout simulation
}

impl LapRaceSim {
    pub fn new(drivers: Vec<Driver>, race_config: RaceConfiguration) -> Self {
        let starting_lap = 1;
        let current_lap_rl = 1;
        Self {
            drivers,
            race_config,
            starting_lap,
            current_lap: current_lap_rl,
        }
    }

    pub fn simulation(&mut self) {
        for current_lap in self.starting_lap..=self.race_config.num_laps {
            self.lap_sim_core_logic(current_lap);
        }
    }

    fn lap_sim_core_logic(&mut self, current_lap: u8) {
        for driver in self.drivers.iter_mut() {
            let driver_lap_time = driver.get_driver_lap_time(current_lap as usize);

            driver.current_lap = current_lap;
            driver.driver_race_progress = current_lap as f64;

            driver.driver_race_time += driver_lap_time;
        }
        self.sort_post_pit_stop(current_lap);

        self.drivers
            .sort_by(|a, b| a.driver_position.cmp(&b.driver_position));

        for driver_index in 0..self.drivers.len() {
            // let mut driver = mem::take(&mut self.drivers[driver_index]);
            let mut driver = unsafe { &mut *self.drivers.as_mut_ptr().add(driver_index) };

            self.check_overtake(&mut driver, current_lap);

            // self.drivers[driver_index] = driver
        }
        for driver in self.drivers.iter_mut() {
            driver
                .driver_accumulated_lap_times
                .push(driver.driver_race_time);
        }
    }

    fn check_overtake(&mut self, driver: &mut Driver, current_lap: u8) {
        let current_position = driver.driver_position;

        if current_position > 1 {
            for position in (1..current_position).rev() {
                let driver_ahead_index = match self.get_driver_ahead_index(position) {
                    Some(index) => index,
                    None => {
                        driver.driver_position -= 1;
                        continue;
                    }
                };
                let driver_ahead = &mut self.drivers[driver_ahead_index];

                let driver_is_trailing = driver.driver_race_time > driver_ahead.driver_race_time;
                let delta = driver.driver_race_time - driver_ahead.driver_race_time;
                let within_drs_range = delta < self.race_config.delta_for_drs_activation;
                let drs_is_active = current_lap >= self.race_config.drs_activation_lap;

                if driver_is_trailing && within_drs_range && drs_is_active {
                    driver.driver_race_time -= self.race_config.drs_boost;
                }

                if driver.driver_race_time < driver_ahead.driver_race_time {
                    let delta = driver_ahead.driver_race_time - driver.driver_race_time;
                    if delta > self.race_config.overtaking_threshold {
                        std::mem::swap(
                            &mut driver.driver_position,
                            &mut driver_ahead.driver_position,
                        );
                        driver.driver_race_time += self.race_config.time_lost_performing_overtake;
                        driver_ahead.driver_race_time +=
                            self.race_config.time_lost_due_to_being_overtaken;
                    } else {
                        driver.driver_laps_behind_traffic += 1; //note self - use the function here, check performance
                        let failed_overtake_penalty =
                            driver.precomputed_failed_overtake_penalties[current_lap as usize - 1];
                        driver.driver_race_time =
                            driver_ahead.driver_race_time + failed_overtake_penalty;
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    fn sort_post_pit_stop(&mut self, current_lap: u8) {
        let mut pitting_drivers = vec![];
        for (index, driver) in self.drivers.iter_mut().enumerate() {
            // NB im using out laps! I send out laps and put them in pit laps variable

            if driver.out_laps.contains(&current_lap) {
                let total_pit_loss = match driver.precomputed_pit_losses.pop() {
                    Some(whole_pit_loss) => whole_pit_loss,
                    None => {
                        let mut rng = rand::rng();

                        // Default values I made up
                        let mean = 3.0; // driver_object.mean_tyre_change_time
                        let std_dev = 0.5; // driver_object.std_dev_tyre_change_time

                        // Create a normal distribution
                        let normal = Normal::new(mean, std_dev).unwrap();

                        // Generate a random value from the normal distribution

                        let tyre_change_time = normal.sample(&mut rng);

                        self.race_config.pit_lane_time_loss + tyre_change_time
                    }
                };

                driver.driver_race_time += total_pit_loss;

                // For RL, the agent would have selected pit on lap n which means they come out at n + 1,
                // For standard simulation this doesnt matter because out laps are recomputed already
                // So it works for both easily
                driver.laps_pitted_on.push(current_lap - 1);

                let pitting_driver = PittingDriver {
                    name: driver.name.clone(),
                    race_time: driver.driver_race_time,
                    old_position: driver.driver_position as i8,
                    new_position: None,
                    positions_to_move: None,
                    driver_index: index,
                };
                pitting_drivers.push(pitting_driver);
            }
        }
        // Finding out where to place the pitting drivers
        let mut driver_ref: Vec<&Driver> = self.drivers.iter().collect();
        driver_ref.sort_by(|a, b| a.driver_race_time.total_cmp(&b.driver_race_time));
        for (index, driver) in driver_ref.iter().enumerate() {
            let new_position = index as i8 + 1;
            for pit_driver in pitting_drivers.iter_mut() {
                if pit_driver.name == driver.name {
                    pit_driver.new_position = Some(new_position);
                    pit_driver.positions_to_move = Some(new_position - pit_driver.old_position)
                }
            }
        }

        pitting_drivers.sort_by(|a, b| a.race_time.total_cmp(&b.race_time));
        pitting_drivers.reverse();
        for pit_driver in pitting_drivers.iter() {
            let main_driver_object =
                unsafe { &mut *self.drivers.as_mut_ptr().add(pit_driver.driver_index) };

            let mut positions_to_move = pit_driver.positions_to_move.unwrap();

            while positions_to_move > 0 {
                let position = main_driver_object.driver_position + 1;

                match self.get_driver_ahead_index(position) {
                    Some(index) => {
                        self.drivers[index].driver_position -= 1;

                        main_driver_object.driver_position += 1;
                    }
                    None => {
                        main_driver_object.driver_position += 1;
                    }
                };

                positions_to_move -= 1;
            }
        }
    }

    fn get_driver_ahead_index(&self, position: u8) -> Option<usize> {
        let mut driver_ahead = None;

        for (index, driver) in self.drivers.iter().enumerate() {
            if driver.driver_position == position {
                driver_ahead = Some(index);
                break;
            }
        }

        return driver_ahead;
    }
}

impl DisplayResult for LapRaceSim {}

impl RaceSimulationCore for LapRaceSim {
    fn run_simulation(&mut self) -> &[Driver] {
        self.simulation();
        &self.drivers
    }

    fn set_simulation_starting_point(&mut self) {
        let mut driver_in_first = &self.drivers[0];
        for driver in self.drivers.iter() {
            if driver.driver_position == 1 {
                driver_in_first = driver;
                break;
            }
        }
        self.starting_lap = driver_in_first.current_lap;
        self.current_lap = driver_in_first.current_lap;
    }

    fn get_simulation_result(&mut self) -> Vec<DriverResult> {
        let mut drivers = Vec::with_capacity(22);
        for driver in self.drivers.iter_mut() {
            let name = driver.name.clone();
            let strategy = mem::take(&mut driver.strategy);
            let driver_position = driver.driver_position;
            let starting_position = driver.starting_position;
            let driver_race_time = driver.driver_race_time;
            let driver_laps_behind_traffic = driver.driver_laps_behind_traffic;
            let laps_pitted_on = mem::take(&mut driver.laps_pitted_on);
            let points = get_f1_points(driver_position);
            drivers.push(DriverResult {
                name,
                driver_position,
                starting_position,
                driver_race_time,
                strategy,
                driver_laps_behind_traffic,
                points,
                laps_pitted_on,
            });
        }
        drivers
    }

    fn incorporate_driver_data(
        &mut self,
        mut simulation_data_map: HashMap<String, DriverParameters>,
    ) {
        for driver in self.drivers.iter_mut() {
            driver.clean_residuals();
            let name = &driver.name;

            let sim_data = simulation_data_map.remove(name).unwrap().into_sim_data();

            driver.setup_for_simulation(sim_data, &self.race_config)
        }

        *self = Self {
            race_config: mem::take(&mut self.race_config),
            drivers: mem::take(&mut self.drivers),
            current_lap: 1,
            starting_lap: 1,
        };
    }

    fn race_config(&self) -> &RaceConfiguration {
        &self.race_config
    }

    fn create_new_simulation(&self, drivers: Vec<Driver>) -> Self {
        let starting_lap = 1;
        let current_lap = 1;
        let race_config = self.race_config;

        Self {
            drivers,
            race_config,
            starting_lap,
            current_lap,
        }
    }
}

impl RaceStrategyEnvironmentCore for LapRaceSim {
    fn step(&mut self) -> bool {
        if self.current_lap > self.race_config.num_laps {
            return true;
        }
        self.lap_sim_core_logic(self.current_lap);
        self.get_mut_agent_driver().update_strategy();

        self.current_lap += 1;

        if self.current_lap > self.race_config.num_laps {
            return true;
        }

        false
    }

    fn get_current_step(&self) -> usize {
        self.current_lap as usize
    }

    fn reset(&mut self, simulation_data_map: &HashMap<String, DriverParameters>) {
        for driver in self.drivers.iter_mut() {
            driver.clean_residuals();
            let name = &driver.name;

            let mut sim_data = simulation_data_map[name].clone().into_sim_data();

            match driver.is_agent() {
                true => {
                    let strategy = mem::take(&mut sim_data.strategy);
                    driver.set_driver_starting_point(sim_data);

                    let starting_compound = Driver::get_starting_compound(strategy);

                    driver.precompute_agent_info(&starting_compound, &self.race_config);
                }
                false => driver.setup_for_simulation(sim_data, &self.race_config),
            }

            driver.current_lap = 1;
            driver.driver_race_progress = 0.0; // Since lap discrete i start at 0.0
        }

        *self = Self {
            race_config: mem::take(&mut self.race_config),
            drivers: mem::take(&mut self.drivers),
            current_lap: 1,
            starting_lap: 1,
        };
    }

    fn get_current_lap(&self) -> u8 {
        // Since im stepping at 1 lap (lap), this will guve me the end of previous lap / start of next lap
        // So this can never be 1. Start if lap 1 has no data, only end of lap 1, ie start of lap 2
        self.current_lap
    }

    fn get_drivers(&self) -> &[Driver] {
        &self.drivers
    }

    fn get_driver_observations(&self) -> Vec<crate::traits::DriverObservation> {
        let num_laps = self.get_race_config().num_laps as f64;
        let num_laps_capacity = num_laps as usize;
        let mut drivers: Vec<&Driver> = self.get_drivers().iter().collect();
        drivers.sort_by(|a, b| a.driver_position.cmp(&b.driver_position));

        let lead_driver = drivers[0];

        let mut driver_observations = Vec::with_capacity(22);

        for (index, driver) in drivers.iter().enumerate() {
            // let mut lap_times = [0.0;78];
            // lap_times[..driver.driver_accumulated_lap_times.len()].copy_from_slice(&driver.driver_accumulated_lap_times);
            let driver_name = driver.name.clone();
            let driver_position = driver.driver_position;

            let mut lap_times = driver.get_lap_times();
            lap_times.resize(num_laps_capacity, 0.0);

            let race_time = lap_times.iter().sum();

            let number_of_pit_stops = driver.laps_pitted_on.len() as u8;

            let mut laps_pitted_on = driver.laps_pitted_on.clone();
            laps_pitted_on.resize(num_laps_capacity, 0);

            let pitted_previous_lap = driver.pitted_previous_lap();

            let different_compounds_used_count = driver.different_compounds_used_count();

            let race_compliance = self.driver_is_regulatory_compliant(driver);

            let compound_compliant = race_compliance.compound_compliant;
            let pit_lane_compliant = race_compliance.pit_lane_compliant;
            let regulatory_compliant = race_compliance.regulatory_compliant;

            let CurrentStint {
                current_compound,
                current_tyre_age,
                current_lap: _,
            } = driver.get_current_stint();

            let race_progress = driver.driver_race_progress / num_laps;

            let delta_to_benchmark_tyre_performance =
                self.get_delta_to_benchmark_tyre_performance(driver, &self.race_config);

            // lap discrete i increment by lap so every step is the end of lap so lap progress is 1 unless starting
            // Adding one to the tyre age for the same reason lap progress is always 1.0 when stepping
            // its lap discrete, tyre age is caluclated at the start of the lap with stint since I am viewing the
            // end of lap it should be +1
            let (lap_progress, current_tyre_age) = if race_progress < 1.0 {
                (0.0, current_tyre_age)
            } else {
                (1.0, current_tyre_age + 1)
            };
            let current_stint = (current_compound, current_tyre_age);

            let driver_in_pit_lane = false; // it is never true so best be explict about it and say none
            let is_agent = if driver.is_agent() { true } else { false };
            let relative_intervals = vec![]; // Will be calculated later
            let interval_behind = 0.0; // Will be calculated later
            if index == 0 {
                let delta_to_leader = 0.0;
                let interval = 0.0;

                driver_observations.push(DriverObservation {
                    driver_position,
                    interval_ahead: interval,
                    delta_to_leader,
                    current_stint,
                    lap_progress,
                    race_progress,
                    is_agent,
                    race_time,
                    lap_times,
                    laps_pitted_on,
                    different_compounds_used_count,
                    regulatory_compliant,
                    driver_in_pit_lane,
                    driver_name,
                    number_of_pit_stops,
                    pitted_previous_lap,
                    compound_compliant,
                    pit_lane_compliant,
                    delta_to_benchmark_tyre_performance,
                    relative_intervals,
                    interval_behind,
                });
            } else {
                let delta_to_leader = driver.driver_race_time - lead_driver.driver_race_time;

                let driver_ahead_index = index - 1;
                let interval =
                    delta_to_leader - driver_observations[driver_ahead_index].delta_to_leader;

                driver_observations.push(DriverObservation {
                    driver_position,
                    interval_ahead: interval,
                    delta_to_leader,
                    current_stint,
                    lap_progress,
                    race_progress,
                    is_agent,
                    race_time,
                    lap_times,
                    laps_pitted_on,
                    different_compounds_used_count,
                    regulatory_compliant,
                    driver_in_pit_lane,
                    driver_name,
                    number_of_pit_stops,
                    pitted_previous_lap,
                    compound_compliant,
                    pit_lane_compliant,
                    delta_to_benchmark_tyre_performance,
                    relative_intervals,
                    interval_behind,
                });
            }
        }

        self.calculate_relative_intervals_in_place(&mut driver_observations);

        driver_observations.sort_by(|a, b| {
            b.is_agent
                .cmp(&a.is_agent)
                .then(a.driver_position.cmp(&b.driver_position))
        });

        driver_observations
    }

    fn get_mut_agent_driver(&mut self) -> &mut Driver {
        let mut driver = None;

        for drv in self.drivers.iter_mut() {
            if drv.is_agent() {
                driver = Some(drv)
            }
        }

        driver.unwrap()
    }

    fn get_race_config(&self) -> RaceConfiguration {
        self.race_config
    }

    fn get_drivers_in_the_pit_lane(&self) -> HashMap<&str, Option<bool>> {
        let mut pitting = HashMap::with_capacity(22);
        for driver in self.drivers.iter() {
            pitting.insert(driver.name.as_str(), None);
        }
        pitting
    }

    fn get_mut_drivers(&mut self) -> &mut [Driver] {
        &mut self.drivers
    }

    fn get_pit_lane_entry(&self) -> Option<f64> {
        None
    }

    fn driver_is_regulatory_compliant(&self, driver: &Driver) -> RaceCompliance {
        let different_compounds_used_count = driver.different_compounds_used_count();
        let compound_compliant = different_compounds_used_count > 1;

        let pit_lane_compliant = if driver.current_lap >= self.race_config.num_laps
            && driver.driver_in_pit_lane == false
        {
            true
        } else {
            false
        };

        let regulatory_compliant = compound_compliant && pit_lane_compliant;

        RaceCompliance {
            compound_compliant,
            pit_lane_compliant,
            regulatory_compliant,
        }
    }

    fn driver_finished_race(&self, driver: &Driver) -> bool {
        driver.current_lap >= self.race_config.num_laps
    }

    fn simple_action(&mut self, compound: Option<&str>, race_config: &RaceConfiguration) {
        let pitting_lap = self.get_agent_driver().current_lap;
        let decision = self.build_pit_decision(compound, pitting_lap);
        let can_modify_future_stops = false;
        self.execute_pit_decision(decision, race_config, can_modify_future_stops);
    }

    fn get_max_steps(&self) -> usize {
        // The max amount of laps in F1 is Monaco currently
        78
    }

    fn get_active_drivers(&self) -> usize {
        // For now im not doing DNF, full course yellows etc for lap discrte so this will stay simple
        self.drivers.len()
    }
}
