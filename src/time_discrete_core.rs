use crate::driver::{CurrentStint, Driver};
use crate::drs::DrsEligibilty;
use crate::race_config::RaceConfiguration;
use crate::race_simulation::{DriverParameters, DriverResult};
use crate::racetrack::RaceTrack;
use crate::traits::{
    DisplayResult, DriverObservation, RaceCompliance, RaceSimulationCore,
    RaceStrategyEnvironmentCore,
};
use crate::utils::get_f1_points;
use rand::random_range;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::mem;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeRaceSim {
    time_step: f64,
    pub race_time: f64,
    pub drivers: Vec<Driver>,
    race_config: RaceConfiguration,
    race_track: RaceTrack,
    current_step: usize,
}

impl TimeRaceSim {
    pub fn new(
        drivers: Vec<Driver>,
        race_config: RaceConfiguration,
        track_name: &str,
        time_step: f64,
    ) -> Self {
        if time_step < 0.1 || time_step > 1.0 {
            panic!("Time step can only be in the range of 0.1 and 1.0")
        }

        let race_track = RaceTrack::create_race_track(track_name, race_config);
        Self {
            time_step,
            race_time: 0.0,
            drivers,
            race_config,
            race_track,
            current_step: 1,
        }
    }

    fn race_not_finished(&self) -> bool {
        for driver in self.drivers.iter() {
            if driver.current_lap <= self.race_config.num_laps {
                return true;
            }
        }
        return false;
    }

    fn driver_completed_race(&self, driver: &Driver) -> bool {
        driver.current_lap > self.race_config.num_laps
    }

    pub fn simulation(&mut self) {
        while self.race_not_finished() {
            self.time_sim_core_logic();
        }
    }
    fn time_sim_core_logic(&mut self) {
        self.race_time += self.time_step;

        for driver_index in 0..self.drivers.len() {
            self.calculate_driver_predictions(driver_index);
        }

        self.drivers
            .sort_by(|a, b| a.driver_position.cmp(&b.driver_position));

        for driver_index in 0..self.drivers.len() {
            self.apply_predictions(driver_index);
        }

        self.update_all_drivers_race_progress();
    }

    fn update_all_drivers_race_progress(&mut self) {
        for driver in self.drivers.iter_mut() {
            if driver.current_lap > self.race_config.num_laps {
                continue;
            }
            let new_lap = driver.driver_race_progress.trunc() as u8;

            if driver.driver_race_progress > driver.current_lap.into() {
                driver.driver_accumulated_lap_times.push(self.race_time);

                driver.current_lap = new_lap;
            }
            driver.driver_race_time = self.race_time;
        }
    }

    fn calculate_driver_predictions(&mut self, driver_index: usize) {
        if self.driver_completed_race(&self.drivers[driver_index]) {
            return; // driver is done nothing left to do
        }

        // let mut driver = mem::take(&mut self.drivers[driver_index]);
        let mut driver = unsafe { &mut *self.drivers.as_mut_ptr().add(driver_index) };

        // adding the previous iterations actual race progress as the baseline for new prediction
        driver.predicted_driver_race_progress = driver.driver_race_progress;

        let driver_pit_status = self.driver_is_pitting(&mut driver);

        match driver_pit_status {
            DriverPitStatus::DriverInPitlane(progression_in_pitlane) => {
                driver.predicted_driver_race_progress += progression_in_pitlane
            }
            DriverPitStatus::DriverNotInPitlane => {
                match self
                    .race_track
                    .drs
                    .driver_drs_eligibilty_at_detection(&driver, &self.drivers)
                {
                    DrsEligibilty::NotAtDetection => (), // do nothing we not at the detetion point

                    DrsEligibilty::AtDetectionPoint(detection_point, drs_state) => {
                        if !self
                            .race_track
                            .drs
                            .checked_driver_drs_status_at_point(&mut driver, detection_point)
                        {
                            driver.drs_is_active = drs_state;
                            driver.drs_last_detection_point = detection_point
                        }
                    }
                }

                if driver.drs_is_active
                    && self.race_track.drs.driver_in_drs_activation_zone(&driver)
                {
                    driver.predicted_driver_race_progress += self
                        .race_track
                        .drs
                        .progress_on_track_with_drs(&driver, self.time_step)
                } else {
                    driver.predicted_driver_race_progress += self.progression_on_track(&driver)
                }
            }
        }
    }

    fn apply_predictions(&mut self, driver_index: usize) {
        if self.driver_completed_race(&self.drivers[driver_index]) {
            return;
        }

        self.drivers[driver_index].driver_race_progress = self.get_race_progress(driver_index);
    }

    fn get_race_progress(&mut self, driver_index: usize) -> f64 {
        if self.drivers[driver_index].driver_position == 1 {
            return self.drivers[driver_index].predicted_driver_race_progress;
        } else {
            let race_progress = self.process_overtakes(driver_index);

            return race_progress;
        }
    }

    fn process_overtakes(&mut self, driver_index: usize) -> f64 {
        let positions_ahead = self.drivers[driver_index].driver_position - 1;
        let predicted_driver_race_progress =
            self.drivers[driver_index].predicted_driver_race_progress;

        for position_ahead in (1..=positions_ahead).rev() {
            let driver_ahead_index = self.get_driver_ahead(position_ahead);

            let driver_ahead_index = match driver_ahead_index {
                DriverAhead::DriverAhead(index) => index,
                DriverAhead::Retired => {
                    let _driver_ahead_index =
                        todo!("Yet to add the functionality of driver retiring");
                    // When the driver can retire the function is pretty simple, it just like suceesful overtake
                    // then continue to next iteration, NB: for future me code is below

                    // self.drivers[driver_index].driver_race_progress =
                    //     self.get_race_progress_with_overtake(driver_index, _driver_ahead_index);
                    // self.swap_positions(driver_index, _driver_ahead_index);
                    // continue;
                }
                DriverAhead::CompletedRace => break,
            };

            let overtake_sucess = self.simulate_overtaking(
                &self.drivers[driver_index],
                &self.drivers[driver_ahead_index],
            );
            match overtake_sucess {
                OvertakeAttemptSuccess::Success => {
                    self.drivers[driver_index].driver_race_progress =
                        self.get_race_progress_with_overtake(driver_index, driver_ahead_index);
                    self.swap_positions(driver_index, driver_ahead_index);
                } // swap then go next iteration
                OvertakeAttemptSuccess::Failed => {
                    {
                        let driver = &mut self.drivers[driver_index];
                        driver.add_lap_behind_traffic(driver.current_lap);
                    }

                    return self.failed_overtake_progress(driver_index, driver_ahead_index);
                } // end check retrun value between current progress and ahead progress
                OvertakeAttemptSuccess::NoAttempt => break, // end check return predicted
            }
        }

        return predicted_driver_race_progress;
    }

    fn swap_positions(&mut self, driver_index: usize, driver_ahead_index: usize) {
        self.drivers[driver_index].driver_position -= 1;
        self.drivers[driver_ahead_index].driver_position += 1;
    }

    fn get_race_progress_with_overtake(
        &mut self,
        driver_index: usize,
        driver_ahead_index: usize,
    ) -> f64 {
        // Driver if succefull overtakes the driver ahead this means he can end up anywhere between driver ahead and
        // predicted amount
        let driver_ahead_race_progress = self.drivers[driver_ahead_index].driver_race_progress;

        let race_progress_max = self.get_max_progress(driver_index, driver_ahead_index);
        let range = driver_ahead_race_progress..=race_progress_max;

        random_range(range)
    }

    fn get_max_progress(&mut self, driver_index: usize, driver_ahead_index: usize) -> f64 {
        let driver_ahead_position = self.drivers[driver_ahead_index].driver_position;
        let predicted_race_progress = self.drivers[driver_index].predicted_driver_race_progress;
        let race_max_progress = if driver_ahead_position == 1 {
            // if the current driver ahead is first there is no one ahead this you can progress freely
            predicted_race_progress
        } else {
            // we get the driver ahead of his race progress and that is max we cant pass
            let position = driver_ahead_position - 1;
            let driver_ahead_of_ahead = self.get_driver_ahead(position);
            let index = match driver_ahead_of_ahead {
                DriverAhead::DriverAhead(usize) => usize,
                DriverAhead::Retired => return predicted_race_progress,
                DriverAhead::CompletedRace => return predicted_race_progress,
            };
            // the progress we cant be ahead because we havent simulated an overtake on them
            let max_progress = self.drivers[index].driver_race_progress;

            max_progress
        };

        race_max_progress
    }

    fn failed_overtake_progress(&self, driver_index: usize, driver_ahead_index: usize) -> f64 {
        let driver = &self.drivers[driver_index];

        let driver_ahead = &self.drivers[driver_ahead_index];

        random_range(driver.driver_race_progress..=driver_ahead.driver_race_progress)
    }

    fn simulate_overtaking(
        &self,
        driver: &Driver,
        driver_ahead: &Driver,
    ) -> OvertakeAttemptSuccess {
        if self.driver_completed_race(driver_ahead) {
            // the driver ahead is done no need to check anything by logic everyone ahead is also done
            return OvertakeAttemptSuccess::NoAttempt;
        }

        let driver_behind_predicted_race_progress = driver.predicted_driver_race_progress;
        let driver_ahead_race_progress = driver_ahead.driver_race_progress;

        let either_driver_in_pits = driver_ahead.driver_in_pit_lane || driver.driver_in_pit_lane;

        let driver_ends_up_ahead =
            driver_behind_predicted_race_progress > driver_ahead_race_progress;

        if driver_ends_up_ahead && either_driver_in_pits {
            return OvertakeAttemptSuccess::Success;
        } else if driver_ends_up_ahead {
            let (driver_behind_lap_time, driver_ahead_lap_time) =
                self.get_battle_times(driver, &driver_ahead);

            let driver_has_better_pace = driver_behind_lap_time < driver_ahead_lap_time;
            let faster_than_threshold = (driver_ahead_lap_time - driver_behind_lap_time)
                > self.race_config.overtaking_threshold;

            if driver_has_better_pace && faster_than_threshold {
                return OvertakeAttemptSuccess::Success;
            } else {
                return OvertakeAttemptSuccess::Failed;
            }
        } else {
            return OvertakeAttemptSuccess::NoAttempt;
        }
    }

    fn progression_on_track(&self, driver: &Driver) -> f64 {
        let current_lap = driver.current_lap as usize;

        let lap_time = driver.get_driver_lap_time(current_lap);

        let progress_per_time_step = self.time_step / lap_time;

        return progress_per_time_step;
    }

    fn get_battle_times(&self, driver_behind: &Driver, driver_ahead: &Driver) -> (f64, f64) {
        let current_lap = driver_ahead.current_lap as usize;

        let mut driver_behind_lap_time = driver_behind.get_driver_lap_time(current_lap);
        let mut driver_ahead_lap_time = driver_ahead.get_driver_lap_time(current_lap);

        if driver_behind.drs_is_active
            && self
                .race_track
                .drs
                .driver_in_drs_activation_zone(driver_behind)
        {
            driver_behind_lap_time -= self.race_track.drs.get_drs_boost(driver_behind);
        }

        if driver_ahead.drs_is_active
            && self
                .race_track
                .drs
                .driver_in_drs_activation_zone(driver_ahead)
        {
            driver_ahead_lap_time -= self.race_track.drs.get_drs_boost(driver_ahead)
        }

        return (driver_behind_lap_time, driver_ahead_lap_time);
    }

    fn get_driver_ahead(&self, position: u8) -> DriverAhead {
        let mut driver_ahead = DriverAhead::Retired;

        for (index, driver) in self.drivers.iter().enumerate() {
            if driver.driver_position == position && self.driver_completed_race(driver) {
                driver_ahead = DriverAhead::CompletedRace;
                break;
            } else if driver.driver_position == position {
                driver_ahead = DriverAhead::DriverAhead(index);
                break;
            }
        }

        return driver_ahead;
    }

    fn progress_per_step_in_pit_lane(&self, driver: &Driver) -> (f64, f64) {
        const TWO_LAPS: f64 = 2.0;

        let effective_laps = TWO_LAPS - self.race_track.pit_lane_displacement;

        let current_lap = driver.pit_stop.lap_entered_pits as usize;

        let lap_time_in = driver.get_driver_lap_time(current_lap);

        // if the driver pits on the final lap there is no next lap value to sample from for how fast he will be
        // it will be out of bounds because he finishing on this lap so i am using it if he ever does
        let next_lap = min(current_lap + 1, self.race_config.num_laps as usize);

        let lap_time_out = driver.get_driver_lap_time(next_lap);
        let total_time_with_pit =
            lap_time_in + lap_time_out + driver.pit_stop.total_pit_loss_for_lap;

        let avg_lap_time = (lap_time_in + lap_time_out) / TWO_LAPS;

        let normal_progression_rate = self.time_step / avg_lap_time;

        let steps_normal = effective_laps / normal_progression_rate;

        let time_normal = steps_normal * self.time_step;

        let pit_section_time = total_time_with_pit - time_normal;

        let steps_in_pit_section = pit_section_time / self.time_step;

        let pit_progression_rate = self.race_track.pit_lane_displacement / steps_in_pit_section;

        return (pit_section_time, pit_progression_rate);
    }

    fn driver_is_pitting(&self, driver: &mut Driver) -> DriverPitStatus {
        let driver_current_lap_progress = driver.get_current_lap_progress();
        let current_lap = driver.current_lap;

        let driver_by_pit_lane_entry = driver_current_lap_progress >= self.race_track.pit_entry;

        let is_pitting_lap = driver.pitting_laps.contains(&current_lap);

        let has_not_pitted_this_lap = !driver.laps_pitted_on.contains(&current_lap);

        if driver_by_pit_lane_entry && is_pitting_lap && has_not_pitted_this_lap {
            driver.driver_in_pit_lane = true;
            driver.pit_stop.simulation_activated = true;
            driver.laps_pitted_on.push(current_lap);
            driver.pit_stop.lap_entered_pits = current_lap;
            driver.pit_stop.total_pit_loss_for_lap = driver
                .precomputed_pit_losses
                .pop()
                .expect("Tried to get a pit loss but it was empty. Should not have been")
        }

        if driver.driver_in_pit_lane {
            // for use in live state updates from live data, when choice of entering the pit lane isnt made by
            // the simulation
            if driver.pit_stop.simulation_activated == false {
                eprintln!(
                    "The pit stop was filled with default values because the driver
                entered the pit lane via live data rather than simulation decision-making"
                );
                driver.pit_stop.simulation_activated = true;
                driver.pit_stop.lap_entered_pits = driver.current_lap;
                driver.pit_stop.total_pit_loss_for_lap =
                    driver.calculate_pit_loss(&self.race_config)
            }

            let (time_to_spend_in_pit_lane, progress_per_step) =
                self.progress_per_step_in_pit_lane(driver);

            if driver.time_spent_in_pit_lane < time_to_spend_in_pit_lane {
                driver.time_spent_in_pit_lane += self.time_step;

                return DriverPitStatus::DriverInPitlane(progress_per_step);
            } else {
                // exiting the pit lane reset parameters
                driver.pit_stop.simulation_activated = false;
                driver.driver_in_pit_lane = false;
                driver.time_spent_in_pit_lane = 0.0;

                return DriverPitStatus::DriverNotInPitlane;
            }
        } else {
            return DriverPitStatus::DriverNotInPitlane;
        }
    }

    fn get_reference_lap_time(&self, lead_driver: &Driver) -> f64 {
        let lap = {
            if self.driver_completed_race(lead_driver) {
                self.race_config.num_laps
            } else {
                lead_driver.current_lap
            }
        } as usize;

        lead_driver.get_driver_lap_time(lap)
    }
}

impl DisplayResult for TimeRaceSim {}

impl RaceSimulationCore for TimeRaceSim {
    fn run_simulation(&mut self) -> &[Driver] {
        self.simulation();
        &self.drivers
    }

    fn set_simulation_starting_point(&mut self) {
        let mut driver_in_last = &self.drivers[0];
        let last_place = self.drivers.len() as u8;

        for driver in self.drivers.iter() {
            if driver.driver_position == last_place {
                driver_in_last = driver;
                break;
            }
        }

        self.race_time = driver_in_last.driver_race_time;
    }

    fn get_simulation_result(&mut self) -> Vec<DriverResult> {
        let mut drivers = Vec::with_capacity(22);
        for driver in self.drivers.iter_mut() {
            let name = driver.name.clone();
            let strategy = mem::take(&mut driver.strategy);
            let starting_position = driver.starting_position;
            let driver_position = driver.driver_position;
            let driver_race_time = driver.driver_race_time;
            let driver_laps_behind_traffic = driver.driver_laps_behind_traffic;
            let points = get_f1_points(driver_position);
            let laps_pitted_on = mem::take(&mut driver.laps_pitted_on);

            drivers.push(DriverResult {
                name,
                starting_position,
                driver_position,
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
            time_step: self.time_step,
            race_time: 0.0,
            drivers: mem::take(&mut self.drivers),
            race_config: mem::take(&mut self.race_config),
            current_step: 1,
            race_track: mem::take(&mut self.race_track),
        };
    }

    fn race_config(&self) -> &RaceConfiguration {
        &self.race_config
    }

    fn create_new_simulation(&self, drivers: Vec<Driver>) -> Self {
        let time_step = self.time_step;
        let race_time = 0.0;
        let race_config = self.race_config;
        let race_track = self.race_track.clone();
        let current_step = 1;

        Self {
            time_step,
            race_time,
            drivers,
            race_config,
            race_track,
            current_step,
        }
    }
}

impl RaceStrategyEnvironmentCore for TimeRaceSim {
    fn step(&mut self) -> bool {
        self.time_sim_core_logic();
        self.current_step += 1;

        if self.driver_completed_race(self.get_agent_driver()) {
            return true;
        }
        self.get_mut_agent_driver().update_strategy();

        false
    }

    fn get_current_step(&self) -> usize {
        self.current_step
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
        }

        *self = Self {
            time_step: self.time_step,
            race_time: 0.0,
            current_step: 1,
            drivers: mem::take(&mut self.drivers),
            race_config: mem::take(&mut self.race_config),
            race_track: mem::take(&mut self.race_track),
        };
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

        let ref_time = self.get_reference_lap_time(lead_driver);

        let mut driver_observations = Vec::with_capacity(22);

        for (index, driver) in drivers.iter().enumerate() {
            let driver_name = driver.name.clone();
            let driver_position = driver.driver_position;
            let lap_progress = driver.get_current_lap_progress();
            let race_progress = driver.driver_race_progress / num_laps;
            let mut lap_times = driver.get_lap_times();
            lap_times.resize(num_laps_capacity, 0.0);
            let race_time = lap_times.iter().sum();
            let number_of_pit_stops = driver.laps_pitted_on.len() as u8;
            let mut laps_pitted_on = driver.laps_pitted_on.clone();
            laps_pitted_on.resize(num_laps_capacity, 0);

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
            let current_stint = (current_compound.to_owned(), current_tyre_age);

            let driver_in_pit_lane = driver.driver_in_pit_lane;
            let pitted_previous_lap = driver.pitted_previous_lap();

            let is_agent = if driver.is_agent() { true } else { false };

            let delta_to_benchmark_tyre_performance =
                self.get_delta_to_benchmark_tyre_performance(driver, &self.race_config);

            let relative_intervals = vec![]; // Will be calculated later
            let interval_behind = 0.0; // Will be calculated later

            if index == 0 || self.driver_completed_race(driver) {
                // short circuit so it will always do lead driver first

                let interval = 0.0;
                let delta_to_leader = 0.0;

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
                let gap = lead_driver.driver_race_progress - driver.driver_race_progress;
                let delta_to_leader = gap * ref_time;

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

        // Ensuring i recieve a consistent order of observations
        driver_observations.sort_by(|a, b| {
            b.is_agent
                .cmp(&a.is_agent)
                .then(a.driver_name.cmp(&b.driver_name))
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
            pitting.insert(driver.name.as_str(), Some(driver.driver_in_pit_lane));
        }
        pitting
    }

    fn get_mut_drivers(&mut self) -> &mut [Driver] {
        &mut self.drivers
    }

    fn get_pit_lane_entry(&self) -> Option<f64> {
        Some(self.race_track.pit_entry)
    }

    fn driver_is_regulatory_compliant(&self, driver: &Driver) -> RaceCompliance {
        let different_compounds_used_count = driver.different_compounds_used_count();
        let compound_compliant = different_compounds_used_count > 1;

        let pit_lane_compliant =
            if self.driver_completed_race(driver) && driver.driver_in_pit_lane == false {
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
        self.driver_completed_race(driver)
    }

    fn simple_action(&mut self, compound: Option<&str>, race_config: &RaceConfiguration) {
        let pitting_lap = self.get_agent_driver().current_lap;

        let decision = self.build_pit_decision(compound, pitting_lap);
        let can_reverse = true;
        self.execute_pit_decision(decision, race_config, can_reverse);
    }

    fn get_max_steps(&self) -> usize {
        // There a max time limit of 2 hours in F1 regulations for race
        7200
    }

    fn get_active_drivers(&self) -> usize {
        // TODO update this logic when i add DNF logic
        self.drivers.len()
    }
}

enum OvertakeAttemptSuccess {
    Success,
    Failed,
    NoAttempt,
}

enum DriverAhead {
    DriverAhead(usize),
    Retired,
    CompletedRace,
}

enum DriverPitStatus {
    DriverInPitlane(f64),
    DriverNotInPitlane,
}

#[allow(dead_code)]
enum RaceState {
    Green,
    SafetyCar,
    VirtualSafetyCar,
}
