use std::{collections::HashMap, mem};

use serde::{Deserialize, Serialize};

use crate::{
    driver::Driver,
    race_config::RaceConfiguration,
    race_simulation::{DriverParameters, SimulationData},
    traits::{
        DriverObservation, RaceCompliance, RaceStrategyEnvironmentCore,
        race_strategy_environment_core::{FieldName, get_field_name},
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaceEnv<E: RaceStrategyEnvironmentCore> {
    pub race_env_sim: E,
    race_config: RaceConfiguration,
    simulation_data_map: HashMap<String, DriverParameters>,
    pub driver_id_map: HashMap<String, u8>,
    alternate_strategies: HashMap<String, HashMap<String, Vec<Vec<(String, u8)>>>>,
    lap_variation_range: (i8, i8),
    stochastic_competitor_strategies: bool,
    stochastic_starting_compound: bool,
    stochastic_positions: bool,
    stochastic_agent_control: bool,
    duelling_self_play: bool,
    agent_selected_fields: Vec<String>,
    competitor_selected_fields: Vec<String>,

    compound_index_map: HashMap<String, f64>,
    index_to_compound: HashMap<usize, String>,
    compound_to_one_hot: HashMap<String, Vec<f32>>,
}

impl<E: RaceStrategyEnvironmentCore> RaceEnv<E> {
    pub fn new(
        race_env_sim: E,
        simulation_data: SimulationData,
        alternate_strategies: HashMap<String, HashMap<String, Vec<Vec<(String, u8)>>>>,
        lap_variation_range: (i8, i8),
        stochastic_competitor_strategies: bool,
        stochastic_starting_compound: bool,
        stochastic_positions: bool,
        stochastic_agent_control: bool,
        duelling_self_play: bool,
        agent_selected_fields: Vec<String>,
        competitor_selected_fields: Vec<String>,
    ) -> Self {
        let race_config = race_env_sim.get_race_config();
        let mut simulation_data_map = HashMap::with_capacity(22);
        let mut driver_id_map = HashMap::with_capacity(22);

        for (index, driver_data) in simulation_data.into_iter().enumerate() {
            driver_id_map.insert(driver_data.name.clone(), index as u8);
            simulation_data_map.insert(driver_data.name.clone(), driver_data);
        }

        if duelling_self_play && (stochastic_competitor_strategies || stochastic_agent_control) {
            println!(
                "⚠️  WARNING: Dueling mode ignores stochastic_competitor_strategies and stochastic_agent_control parameters"
            );
        }
        let num_compounds = race_env_sim.get_agent_driver().get_all_compounds().len();
        let compound_index_map: HashMap<&String, f64> = race_env_sim.compound_index_map();

        let mut index_to_compound: HashMap<usize, String> = HashMap::with_capacity(num_compounds);
        let mut compound_to_one_hot: HashMap<String, Vec<f32>> =
            HashMap::with_capacity(num_compounds);
        let mut compound_index = HashMap::with_capacity(num_compounds);

        for (compound, index) in compound_index_map.iter() {
            let mut lap_choice_one_hot = vec![0.0f32; num_compounds];
            let index = *index as usize;
            lap_choice_one_hot[index] = 1.0;
            let compound = compound.to_string();

            index_to_compound.insert(index, compound.clone());
            compound_to_one_hot.insert(compound.clone(), lap_choice_one_hot);
            compound_index.insert(compound, index as f64);
        }

        Self {
            race_env_sim,
            race_config,
            alternate_strategies,
            lap_variation_range,
            stochastic_competitor_strategies,
            stochastic_starting_compound,
            simulation_data_map,
            driver_id_map,
            stochastic_positions,
            stochastic_agent_control,
            duelling_self_play,
            agent_selected_fields,
            competitor_selected_fields,
            index_to_compound,
            compound_to_one_hot,
            compound_index_map: compound_index,
        }
    }

    pub fn step(&mut self) -> (bool, Vec<DriverObservation>) {
        let done = self.race_env_sim.step();
        let obs = self.race_env_sim.get_driver_observations();

        (done, obs)
    }

    pub fn branch(&self) -> RaceEnv<E> {
        self.clone()
    }

    pub fn reset(&mut self) -> Vec<DriverObservation> {
        if self.duelling_self_play {
            self.race_env_sim.dueling(
                &mut self.simulation_data_map,
                &self.race_config,
                &self.alternate_strategies,
                self.lap_variation_range,
                self.stochastic_starting_compound,
                self.stochastic_positions,
            );

            self.race_env_sim.reset(&self.simulation_data_map);
        } else {
            let simulation_data_map = self.simulation_data_map.clone();
            let reset_simulation_data_map = self.race_env_sim.randomize_driver_strategies(
                simulation_data_map,
                &self.race_config,
                &self.alternate_strategies,
                self.lap_variation_range,
                self.stochastic_competitor_strategies,
                self.stochastic_starting_compound,
                self.stochastic_positions,
                self.stochastic_agent_control,
            );

            self.race_env_sim.reset(&reset_simulation_data_map);
        };

        self.race_env_sim.get_driver_observations()
    }
    pub fn get_current_state_observation(&self) -> Vec<DriverObservation> {
        self.race_env_sim.get_driver_observations()
    }
    pub fn print_grid(&self) {
        let mut drivers = self
            .race_env_sim
            .get_drivers()
            .iter()
            .collect::<Vec<&Driver>>();
        drivers.sort_by(|a, b| a.driver_position.cmp(&b.driver_position));
        drivers.iter().for_each(|d| {
            if d.is_agent() {
                println!(
                    "{:.4} | P{} - {} (Agent): {:?}",
                    d.driver_race_time, d.driver_position, d.name, d.strategy
                )
            } else {
                println!(
                    "{:.4} | P{} - {}: {:?}",
                    d.driver_race_time, d.driver_position, d.name, d.strategy
                )
            }
        });
    }

    pub fn available_simple_actions(&self) -> Vec<Option<&String>> {
        self.race_env_sim.available_simple_actions()
    }

    pub fn available_complex_actions(&self) -> Vec<Option<&String>> {
        self.race_env_sim.available_complex_actions()
    }

    pub fn simple_action(&mut self, compound: Option<&str>) {
        if self
            .race_env_sim
            .driver_finished_race(self.race_env_sim.get_agent_driver())
        {
            return;
        }
        self.race_env_sim.simple_action(compound, &self.race_config);
    }
    pub fn get_agent_results(&self) -> AgentInfo {
        let agent = self.race_env_sim.get_agent_driver();

        let name = agent.name.clone();
        let starting_position = agent.starting_position;
        let lap = agent.current_lap;
        let position = agent.driver_position;
        let race_time = agent.driver_race_time as f32;
        let strategy = agent.strategy.clone();
        let laps_pitted_on = agent.laps_pitted_on.clone();
        let number_of_stops = laps_pitted_on.len() as u8;
        let different_compounds_used_count = agent.different_compounds_used_count();

        let compliance = self.race_env_sim.driver_is_regulatory_compliant(agent);

        let _pit_lane_compliant = compliance.pit_lane_compliant;
        let _compound_compliant = compliance.compound_compliant;
        let is_regulatory_compliant = compliance.regulatory_compliant;

        AgentInfo {
            name,
            starting_position,
            position,
            race_time,
            strategy,
            laps_pitted_on,
            different_compounds_used_count,
            is_regulatory_compliant,
            lap,
            number_of_stops,
        }
    }

    pub fn complex_action(&mut self, compound: Option<&str>, pitting_lap: u8) {
        if self
            .race_env_sim
            .driver_finished_race(self.race_env_sim.get_agent_driver())
        {
            return;
        }
        let race_config = &self.race_config;

        self.race_env_sim
            .complex_action(compound, pitting_lap, race_config);
    }

    pub fn get_agent_current_lap(&self) -> u8 {
        self.race_env_sim.get_agent_driver().current_lap
    }

    pub fn get_agent_current_lap_progress(&self) -> f64 {
        self.race_env_sim
            .get_agent_driver()
            .get_current_lap_progress()
    }

    pub fn get_pit_lane_entry(&self) -> Option<f64> {
        self.race_env_sim.get_pit_lane_entry()
    }

    pub fn get_agent_compliance(&self) -> RaceCompliance {
        self.race_env_sim
            .driver_is_regulatory_compliant(self.race_env_sim.get_agent_driver())
    }

    pub fn compound_index_map(&self, compound: &String) -> f64 {
        self.compound_index_map[compound]
    }

    pub fn compound_index_to_compound_name(&self, index: usize) -> &str {
        &self.index_to_compound[&index]
    }

    pub fn compound_to_one_hot(&self, compound: &String) -> &Vec<f32> {
        &self.compound_to_one_hot[compound]
    }
    pub fn get_selected_driver_fields_vector(&self, mut obs: DriverObservation) -> Vec<f32> {
        // TODO make this more efficent
        let mut array = Vec::with_capacity(200);
        let selected_fields = if obs.is_agent {
            &self.agent_selected_fields
        } else {
            &self.competitor_selected_fields
        };
        for observation_field in selected_fields {
            match get_field_name(observation_field) {
                FieldName::IsAgent => array.push(obs.is_agent as u8 as f32),
                FieldName::DriverName => array.push(self.driver_id_map[&obs.driver_name] as f32),
                FieldName::DriverPosition => array.push(obs.driver_position as f32),
                FieldName::LapProgress => array.push(obs.lap_progress as f32),
                FieldName::RaceProgress => array.push(obs.race_progress as f32),
                FieldName::IntervalAhead => array.push(obs.interval_ahead as f32),
                FieldName::IntervalBehind => array.push(obs.interval_behind as f32),
                FieldName::DeltaToLeader => array.push(obs.delta_to_leader as f32),
                FieldName::RelativeIntervals => {
                    array.extend(obs.relative_intervals.iter().map(|x| *x as f32))
                }
                FieldName::DeltaToBenchmarkTyrePerformance => {
                    array.push(obs.delta_to_benchmark_tyre_performance as f32)
                }
                FieldName::CurrentStint => {
                    let (compound, tyre_age) = mem::take(&mut obs.current_stint); // avoiding copy
                    let compound_id = self.compound_index_map(&compound);

                    array.push(compound_id as f32);
                    array.push(tyre_age as f32);
                }
                FieldName::RaceTime => array.push(obs.race_time as f32),
                FieldName::LapTimes => array.extend(obs.lap_times.iter().map(|x| *x as f32)),
                FieldName::DriverInPitLane => array.push(obs.driver_in_pit_lane as u8 as f32),
                FieldName::LapsPittedOn => {
                    array.extend(obs.laps_pitted_on.iter().map(|x| *x as f32))
                }
                FieldName::PittedPreviousLap => array.push(obs.pitted_previous_lap as u8 as f32),
                FieldName::NumberOfPitStops => array.push(obs.number_of_pit_stops as f32),
                FieldName::DifferentCompoundsUsedCount => {
                    array.push(obs.different_compounds_used_count as f32)
                }
                FieldName::CompoundCompliant => array.push(obs.compound_compliant as u8 as f32),
                FieldName::PitLaneCompliant => array.push(obs.pit_lane_compliant as u8 as f32),
                FieldName::RegulatoryCompliant => array.push(obs.regulatory_compliant as u8 as f32),
            };
        }
        array
    }
}

#[derive(Debug)]
pub struct AgentInfo {
    pub name: String,
    pub starting_position: u8,
    pub position: u8,
    pub lap: u8,
    pub race_time: f32,
    pub strategy: Vec<(String, u8)>,
    pub laps_pitted_on: Vec<u8>,
    pub different_compounds_used_count: u8,
    pub is_regulatory_compliant: bool,
    pub number_of_stops: u8,
}
