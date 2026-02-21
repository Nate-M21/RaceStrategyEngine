use std::collections::HashMap;
use std::mem;

use crate::driver::{CurrentStint, Driver};
use crate::race_config::RaceConfiguration;
use crate::race_simulation::DriverParameters;
use crate::utils::{BoundedStack, modify_stints_randomly, select_random_strategy};
use pyo3::pyclass;
use rand::rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};

#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct DriverObservation {
    #[pyo3(get, set)]
    pub is_agent: bool,
    #[pyo3(get, set)]
    pub driver_name: String,
    #[pyo3(get, set)]
    pub driver_position: u8,
    #[pyo3(get, set)]
    pub lap_progress: f64,
    #[pyo3(get, set)]
    pub race_progress: f64,
    #[pyo3(get, set)]
    pub interval_ahead: f64,
    #[pyo3(get, set)]
    pub interval_behind: f64,
    #[pyo3(get, set)]
    pub delta_to_leader: f64,
    #[pyo3(get, set)]
    pub relative_intervals: Vec<f64>,
    #[pyo3(get, set)]
    pub delta_to_benchmark_tyre_performance: f64,
    #[pyo3(get, set)]
    pub current_stint: (String, u8),
    #[pyo3(get, set)]
    pub race_time: f64,
    #[pyo3(get, set)]
    pub lap_times: Vec<f64>,
    #[pyo3(get, set)]
    pub driver_in_pit_lane: bool,
    #[pyo3(get, set)]
    pub laps_pitted_on: Vec<u8>,
    #[pyo3(get, set)]
    pub pitted_previous_lap: bool,
    #[pyo3(get, set)]
    pub number_of_pit_stops: u8,
    #[pyo3(get, set)]
    pub different_compounds_used_count: u8,
    #[pyo3(get, set)]
    pub compound_compliant: bool, // F1 says use 2 different compounds
    #[pyo3(get, set)]
    pub pit_lane_compliant: bool, // F1 says you cant end the race in pit lane
    #[pyo3(get, set)]
    pub regulatory_compliant: bool, // both the above combined
}
pub struct RaceCompliance {
    pub compound_compliant: bool,   // F1 says use 2 different compounds
    pub pit_lane_compliant: bool,   // F1 says you cant end the race in pit lane
    pub regulatory_compliant: bool, // both the above combined
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StackedObservations<T: Clone> {
    pub observations: BoundedStack<T>,
    head: usize,
    buffer_filled_with_real: bool,
    first_observation: Option<T>,
}

impl<T: Clone> StackedObservations<T> {
    pub fn new(max_size: usize) -> Self {
        let head = reset_head(max_size);
        let filled_with_real = false;
        Self {
            observations: BoundedStack::new(max_size),
            head: head,
            buffer_filled_with_real: filled_with_real,
            first_observation: None,
        }
    }

    pub fn update_max_size(&mut self, max_size: usize) {
        if self.first_observation.is_none() {
            self.observations = BoundedStack::new(max_size)
        } else {
            self.observations.max_size = max_size;
            self.add_observation_and_fill_buffer_with_dummy();
        }
    }

    pub fn add_observation(&mut self, observation: T) {
        // at the start when there is no data I need to fill
        if !self.observations.is_full() {
            // adding until filled with reset observation or first obs
            if self.first_observation.is_none() {
                self.first_observation = Some(observation);
            }
            self.add_observation_and_fill_buffer_with_dummy();
        // at the point where now the buffer is filled with actual i can just add to end and pop of the front
        // and this can work for both Transformer sequencing and normal stacking for MLP
        } else if self.buffer_filled_with_real {
            self.observations.push(observation);
        } else {
            self.overwrite_dummy_data_with_real(observation);
        }
    }

    fn overwrite_dummy_data_with_real(&mut self, observation: T) {
        self.observations.data[self.head] = observation;
        self.move_head_next_position();
    }

    pub fn clear(&mut self) {
        self.observations.clear();
        self.head = reset_head(self.observations.max_size);
        self.buffer_filled_with_real = false;
        self.first_observation = None;
    }

    fn move_head_next_position(&mut self) {
        if self.head == self.observations.len() - 1 {
            self.buffer_filled_with_real = true;
            self.head = reset_head(self.observations.max_size);
        } else {
            self.head += 1;
        }
    }

    fn add_observation_and_fill_buffer_with_dummy(&mut self) {
        while !self.observations.is_full() {
            self.observations.push(
                self.first_observation
                    .clone()
                    .expect("First observation tried to be added to stack but it was still None"),
            );
        }
    }
}
/// function used to reset the head pointer of each element to write over. The reason for 0 is when the max size is 1 ie
/// no stacking reseting the head to 1 leads to index out of bound error. And the reason for 1 is I am looking at
/// sequence of observations  i dont want to overwrite the first one
fn reset_head(max_size: usize) -> usize {
    let head = if max_size == 1 { 0 } else { 1 };
    head
}

impl StackedObservations<Vec<f32>> {
    pub fn flatten_observations(&self) -> Vec<f32> {
        let stacked_observations: Vec<f32> = self.observations.iter().flatten().cloned().collect();

        stacked_observations
    }

    pub fn get_observations(&self) -> Vec<Vec<f32>> {
        self.observations.iter().cloned().collect()
    }
}

pub enum PitDecision {
    PitThisLap {
        pit_compound: String,
        pitting_lap: u8,
        next_lap_tyre_age: u8,
    },
    PitFutureLap {
        pit_compound: String,
        pitting_lap: u8,
        current_compound: String,
        next_lap_tyre_age: u8,
    },
    NoPit {
        current_compound: String,
        next_lap_tyre_age: u8,
    },
}

pub trait RaceStrategyEnvironmentCore: Send + Sync + Clone {
    fn step(&mut self) -> bool;

    fn reset(&mut self, simulation_data_map: &HashMap<String, DriverParameters>);

    fn get_current_step(&self) -> usize;

    fn get_max_steps(&self) -> usize;

    fn get_active_drivers(&self) -> usize;

    fn branch_from_state(&self) -> Self
    where
        Self: Sized,
    {
        self.clone()
    }

    fn compound_index_map(&self) -> HashMap<&String, f64> {
        let actions = self.available_simple_actions();
        let mut map = HashMap::with_capacity(5);

        for (index, action) in actions.iter().enumerate() {
            if let Some(compound) = action {
                map.insert(*compound, index as f64);
            }
        }

        map
    }

    fn calculate_relative_intervals_in_place(&self, driver_observations: &mut [DriverObservation]) {
        // Driver observations must be sorted before inserted into function

        // TODO optimize this later, check when j has position +1 the position of i then use that instead for
        // interval behind
        let last_index = driver_observations.len() - 1;
        let n_drivers = driver_observations.len();
        for i in 0..driver_observations.len() {
            let mut driver_relative_intervals = Vec::with_capacity(n_drivers);

            let driver_delta_to_leader = driver_observations[i].delta_to_leader;

            if i != last_index {
                let driver_behind = &driver_observations[i + 1];
                driver_observations[i].interval_behind =
                    driver_delta_to_leader - driver_behind.delta_to_leader;
            } else {
                driver_observations[i].interval_behind = 0.0
            }

            for j in 0..driver_observations.len() {
                let relative_interval =
                    driver_delta_to_leader - driver_observations[j].delta_to_leader;
                driver_relative_intervals.push(relative_interval);
            }
            driver_observations[i].relative_intervals = driver_relative_intervals;
        }
    }
    fn randomize_driver_strategies(
        &mut self,
        mut simulation_data_map: HashMap<String, DriverParameters>,
        race_config: &RaceConfiguration,
        alternate_strategies: &HashMap<String, HashMap<String, Vec<Vec<(String, u8)>>>>,
        lap_variation_range: (i8, i8),
        stochastic_competitor_strategies: bool,
        stochastic_starting_compound: bool,
        stochastic_positions: bool,
        stochastic_agent_control: bool,
    ) -> HashMap<String, DriverParameters> {
        if !stochastic_starting_compound
            && !stochastic_competitor_strategies
            && !stochastic_positions
            && !stochastic_agent_control
        {
            // both agent and competitors determinstic, i return immediately
            return simulation_data_map;
        }
        let mut rng = rng();
        let drivers_len = self.get_drivers().len();
        let stochastic_positions_vec = match stochastic_positions {
            true => {
                let mut positions = (1..=drivers_len).collect::<Vec<usize>>();
                positions.shuffle(&mut rng);
                Some(positions)
            }
            false => None,
        };

        let bool_vector = match stochastic_agent_control {
            true => {
                let mut bools = vec![false; drivers_len - 1];
                bools.push(true);
                bools.shuffle(&mut rng);
                Some(bools)
            }
            false => None,
        };
        for (index, driver) in self.get_mut_drivers().iter_mut().enumerate() {
            // To ensure that if stochastic agent control is on but stocasticas competitor straegies is off the
            // strategies revert to orginal after the driver has been the agent
            if stochastic_agent_control {
                let bools = bool_vector.as_ref().unwrap();
                let agent_controlled = bools[index];

                driver.set_agent_status(agent_controlled);
            }

            let name = &driver.name;
            if stochastic_positions {
                let positions = stochastic_positions_vec.as_ref().unwrap();
                // give the driver a random position, includes the agent
                simulation_data_map
                    .get_mut(&driver.name)
                    .unwrap()
                    .starting_position = positions[index] as u8;
                simulation_data_map.get_mut(name).unwrap().position = positions[index] as u8;
            }

            if driver.is_agent() && !stochastic_starting_compound {
                // if I dont want to change the agent tyres continue but might change competitors
                continue;
            }

            if !driver.is_agent() && !stochastic_competitor_strategies {
                // if i dont want to change the competitors strategies
                continue;
            }

            let strategies = &alternate_strategies[name];
            let mut random_strategy = select_random_strategy(strategies).clone();
            random_strategy.shuffle(&mut rng);
            modify_stints_randomly(&mut random_strategy, race_config, lap_variation_range);
            simulation_data_map.get_mut(name).unwrap().strategy = random_strategy;
        }

        simulation_data_map
    }

    fn dueling(
        &mut self,
        simulation_data_map: &mut HashMap<String, DriverParameters>,
        race_config: &RaceConfiguration,
        alternate_strategies: &HashMap<String, HashMap<String, Vec<Vec<(String, u8)>>>>,
        lap_variation_range: (i8, i8),
        stochastic_starting_compound: bool,
        stochastic_positions: bool,
    ) {
        let agent = self.get_agent_driver();

        let name = &agent.name;
        let mut rng = rng();
        if self.driver_finished_race(agent)
            && self
                .driver_is_regulatory_compliant(agent)
                .regulatory_compliant
        {
            // if driver finished race and he was compliant apply his strategy to be used
            let agent = self.get_mut_agent_driver();
            simulation_data_map.get_mut(&agent.name).unwrap().strategy =
                mem::take(&mut agent.strategy);
        } else {
            // if the driver did an invalid strategy pick a random one for the next race for NPC to use
            let strategies = &alternate_strategies[name];
            let mut random_strategy = select_random_strategy(strategies).clone();
            random_strategy.shuffle(&mut rng);
            modify_stints_randomly(&mut random_strategy, race_config, lap_variation_range);
            simulation_data_map.get_mut(name).unwrap().strategy = random_strategy;
        }

        let drivers_len = self.get_drivers().len();

        let bool_vector = {
            let mut bools = vec![false; drivers_len - 1];
            bools.push(true);
            bools.shuffle(&mut rng);
            bools
        };

        let positions_vector = match stochastic_positions {
            true => {
                let mut positions = (1..=drivers_len as u8).collect::<Vec<u8>>();
                positions.shuffle(&mut rng);
                Some(positions)
            }
            false => None,
        };

        for (index, driver) in self.get_mut_drivers().iter_mut().enumerate() {
            if let Some(ref positions) = positions_vector {
                simulation_data_map
                    .get_mut(&driver.name)
                    .unwrap()
                    .starting_position = positions[index];
                simulation_data_map.get_mut(&driver.name).unwrap().position = positions[index];
            }

            let agent_controlled = bool_vector[index];

            driver.set_agent_status(agent_controlled);

            // Only checking once ive assigned agent status to prevent collisons
            if driver.is_agent() {
                // shuffling so i get a random different starting compound
                if stochastic_starting_compound {
                    simulation_data_map
                        .get_mut(&driver.name)
                        .unwrap()
                        .strategy
                        .shuffle(&mut rng);
                }
            }
        }
    }

    fn get_delta_to_benchmark_tyre_performance(
        &self,
        driver: &Driver,
        race_config: &RaceConfiguration,
    ) -> f64 {
        let CurrentStint {
            current_compound,
            current_tyre_age: _,
            mut current_lap,
        } = driver.get_current_stint();
        let baseline_compound_performance = driver.get_tyre_models_fastest_time(&current_compound);

        if current_lap > race_config.num_laps {
            current_lap -= 1
        }

        let raw_lap_time = driver.get_driver_lap_time(current_lap as usize);

        let current_tyre_performance =
            driver.get_tyre_performance(raw_lap_time, current_lap, race_config);

        current_tyre_performance - baseline_compound_performance
    }

    fn get_agent_driver(&self) -> &Driver {
        let mut driver = None;

        for drv in self.get_drivers() {
            if drv.is_agent() {
                driver = Some(drv);
            }
        }

        driver.unwrap()
    }

    fn driver_finished_race(&self, driver: &Driver) -> bool;

    fn driver_is_regulatory_compliant(&self, driver: &Driver) -> RaceCompliance;

    fn available_simple_actions(&self) -> Vec<Option<&String>> {
        let agent = self.get_agent_driver();
        let mut compounds = agent.get_all_compounds();

        compounds.sort();

        let mut actions = vec![];

        for compound in compounds {
            actions.push(Some(compound));
        }
        actions.push(None);

        actions
    }

    fn available_complex_actions(&self) -> Vec<Option<&String>> {
        let agent = self.get_agent_driver();
        let mut compounds = agent.get_all_compounds();

        compounds.sort();
        let num_laps = self.get_race_config().num_laps as usize;

        let compounds: Vec<Option<&String>> = compounds
            .into_iter()
            .map(|compound| Some(compound))
            .collect();

        let mut actions: Vec<Option<&String>> = vec![];

        let compound_actions = compounds.repeat(num_laps - 1);

        actions.extend(compound_actions);
        actions.push(None);

        actions
    }

    fn get_current_lap(&self) -> u8 {
        // NOTE Lap discrete has its own impl

        let drivers = self.get_drivers();
        for driver in drivers {
            if driver.driver_position == 1 {
                return driver.current_lap;
            }
        }

        0 // This should never happen anyways, there always a driver in first
    }
    fn simple_action(&mut self, compound: Option<&str>, race_config: &RaceConfiguration);

    /// When the agent can plan and say when to pit not just make a decision on the current lap. Mainly meant for ROB
    fn complex_action(
        &mut self,
        compound: Option<&str>,
        pitting_lap: u8,
        race_config: &RaceConfiguration,
    ) {
        let decision = self.build_pit_decision(compound, pitting_lap);
        let can_modify_future_stops = true; // always true for complex actions because it can select future
        // plans and then later change them
        self.execute_pit_decision(decision, race_config, can_modify_future_stops);
    }

    fn build_pit_decision(&self, compound: Option<&str>, pitting_lap: u8) -> PitDecision {
        let CurrentStint {
            current_compound,
            current_tyre_age,
            current_lap,
        } = self.get_agent_driver().get_current_stint();

        let decision = match compound {
            Some(pit_compound) if pitting_lap == current_lap => PitDecision::PitThisLap {
                pit_compound: pit_compound.to_owned(),
                pitting_lap,
                next_lap_tyre_age: 0,
            },

            Some(pit_compound) if pitting_lap > current_lap => PitDecision::PitFutureLap {
                pit_compound: pit_compound.to_owned(),
                pitting_lap,
                current_compound: current_compound,
                next_lap_tyre_age: current_tyre_age + 1,
            },

            None => PitDecision::NoPit {
                current_compound: current_compound,
                next_lap_tyre_age: current_tyre_age + 1,
            },
            _ => panic!(
                "Invalid Pit decision. Selected to pit in the past. The pitting lap is {pitting_lap} and the compound is {compound:?} the current lap is {current_lap}"
            ),
        };
        decision
    }

    fn execute_pit_decision(
        &mut self,
        decision: PitDecision,
        race_config: &RaceConfiguration,
        can_modify_future_stops: bool,
    ) {
        let agent = self.get_mut_agent_driver();

        match decision {
            PitDecision::PitThisLap {
                pit_compound,
                pitting_lap,
                next_lap_tyre_age,
            } => {
                agent.add_pit_stop(pit_compound.clone(), pitting_lap, race_config);
                agent.choose_next_lap_compound(&pit_compound, next_lap_tyre_age, race_config);
            }
            PitDecision::PitFutureLap {
                pit_compound,
                pitting_lap,
                current_compound,
                next_lap_tyre_age,
            } => {
                agent.add_pit_stop(pit_compound, pitting_lap, race_config);
                agent.choose_next_lap_compound(&current_compound, next_lap_tyre_age, race_config);
            }
            PitDecision::NoPit {
                current_compound,
                next_lap_tyre_age,
            } => {
                agent.choose_next_lap_compound(&current_compound, next_lap_tyre_age, race_config);
                if can_modify_future_stops {
                    agent.remove_pit_stop(); // if the previous choice was to pit and it changes this removes it, if not
                    // does nothing
                }
            }
        }
    }

    fn get_pit_lane_entry(&self) -> Option<f64>;

    fn get_mut_agent_driver(&mut self) -> &mut Driver;

    fn get_drivers(&self) -> &[Driver];

    fn get_mut_drivers(&mut self) -> &mut [Driver];

    fn get_race_config(&self) -> RaceConfiguration;

    fn get_driver_observations(&self) -> Vec<DriverObservation>;

    fn get_drivers_in_the_pit_lane(&self) -> HashMap<&str, Option<bool>>;
}

pub fn get_field_name(observation_field: &str) -> FieldName {
    let track_name = observation_field
        .to_lowercase()
        .parse::<FieldName>()
        .unwrap_or_else(|_| {
            panic!(
                "Sorry, could not find the field. Here are the availiable fields to select from:\n\n{}\n",
                show_all_fields()
            )
        });

    track_name
}

fn show_all_fields() -> String {
    let field_names = FieldName::iter()
        .map(|field_name| field_name.to_string())
        .collect::<Vec<String>>();
    let track_names = field_names.join("\n");

    track_names
}

#[derive(Debug, EnumString, Display, EnumIter)]
#[strum(serialize_all = "snake_case")]
pub enum FieldName {
    IsAgent,

    DriverName,

    DriverPosition,

    LapProgress,

    RaceProgress,

    IntervalAhead,

    IntervalBehind,

    DeltaToLeader,

    RelativeIntervals,

    DeltaToBenchmarkTyrePerformance,

    CurrentStint,

    RaceTime,

    LapTimes,

    DriverInPitLane,

    LapsPittedOn,

    PittedPreviousLap,

    NumberOfPitStops,

    DifferentCompoundsUsedCount,

    CompoundCompliant, // F1 says use 2 different compounds

    PitLaneCompliant, // F1 says you cant end the race in pit lane

    RegulatoryCompliant, // both the above combined
}
