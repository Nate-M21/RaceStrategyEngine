use std::{cmp::min, collections::HashMap, mem, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    driver::{CurrentStint, Driver},
    environment::{AgentInfo, RaceEnv},
    lap_discrete_core::LapRaceSim,
    race_config::RaceConfiguration,
    race_simulation::SimulationData,
    time_discrete_core::TimeRaceSim,
    traits::{
        DriverObservation, RaceCompliance, RaceStrategyEnvironmentCore,
        gym::{GymEnvironment, MCTSGymEnvironment},
        race_strategy_environment_core::StackedObservations,
    },
    utils::{SharedNormalization, argmax, create_fully_connected_edge_index, get_f1_points},
};
enum Action {
    Simple {
        compound: Option<String>,
    },
    Complex {
        compound: Option<String>,
        pitting_lap: u8,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
enum ActionComplexityType {
    Simple,
    Complex,
}
impl ActionComplexityType {
    fn new(action_complexity: &str) -> Self {
        let action_type = match action_complexity {
            "simple" => Self::Simple,
            "complex" => Self::Complex,
            _ => panic!("Invalid action complexity type, please select between simple and complex"),
        };

        action_type
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ObservationType {
    Agent,
    Drivers,
    Graph(RaceGraphState),
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub struct RaceGraphState {
    pub num_drivers: usize,
    pub features_per_node: usize,
}
impl ObservationType {
    fn new(race_perspective: &str, num_drivers: usize, single_step_obs_dim: usize) -> Self {
        let step_type = match race_perspective {
            "agent_only" => Self::Agent,
            "all_drivers" => Self::Drivers,
            "graph" => {
                let features_per_node = single_step_obs_dim / num_drivers;
                Self::Graph(RaceGraphState {
                    num_drivers,
                    features_per_node,
                })
            }
            _ => panic!(
                "Incorrect step type selected, please choose between, agent, drivers and graph"
            ),
        };

        step_type
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum RaceEnvType {
    LapDiscrete(RaceEnv<LapRaceSim>),
    TimeDiscrete(RaceEnv<TimeRaceSim>),
}

impl RaceEnvType {
    pub fn new(
        drivers: Vec<Driver>,
        race_config: RaceConfiguration,
        simulation_data: SimulationData,
        alternate_strategies: HashMap<String, HashMap<String, Vec<Vec<(String, u8)>>>>,
        stochastic_competitor_strategies: bool,
        stochastic_starting_compound: bool,
        stochastic_positions: bool,
        stochastic_agent_control: bool,
        duelling_self_play: bool,
        lap_variation_range: (i8, i8),
        simulation_type: &str,
        track_name: &str,
        time_step: f64,
        agent_selected_fields: Vec<String>,
        competitor_selected_fields: Vec<String>,
    ) -> Self {
        let coretype: &str = &simulation_type;

        let race_env_type = match coretype {
            "lap_discrete" => {
                let sim = LapRaceSim::new(drivers, race_config);
                let race_env = RaceEnv::new(
                    sim,
                    simulation_data,
                    alternate_strategies,
                    lap_variation_range,
                    stochastic_competitor_strategies,
                    stochastic_starting_compound,
                    stochastic_positions,
                    stochastic_agent_control,
                    duelling_self_play,
                    agent_selected_fields,
                    competitor_selected_fields,
                );
                RaceEnvType::LapDiscrete(race_env)
            }
            "time_discrete" => {
                let sim = TimeRaceSim::new(drivers, race_config, track_name, time_step);
                let race_env = RaceEnv::new(
                    sim,
                    simulation_data,
                    alternate_strategies,
                    lap_variation_range,
                    stochastic_competitor_strategies,
                    stochastic_starting_compound,
                    stochastic_positions,
                    stochastic_agent_control,
                    duelling_self_play,
                    agent_selected_fields,
                    competitor_selected_fields,
                );
                RaceEnvType::TimeDiscrete(race_env)
            }
            _ => panic!("Select 'time_discrete' or 'lap_discrete"),
        };

        race_env_type
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RaceStrategyEnvironment {
    env: RaceEnvType,
    num_drivers: usize,
    steps_needed: Option<u8>,
    complex_action_taken: (Option<String>, u8),
    pub observations: StackedObservations<Vec<f32>>,
    shared_normalization: Option<SharedNormalization>,
    obs_size: usize,
    single_step_obs_dim: usize,
    stack_size: usize,
    pub obs_type: ObservationType,
    action_complexity_type: ActionComplexityType,
}

impl RaceStrategyEnvironment {
    pub fn new(
        drivers_hash_map: HashMap<String, Driver>,
        race_config: RaceConfiguration,
        race_perspective: &str,
        action_complexity: &str,
        simulation_data: SimulationData,
        alternate_strategies: HashMap<String, HashMap<String, Vec<Vec<(String, u8)>>>>,
        stochastic_competitor_strategies: bool,
        stochastic_starting_compound: bool,
        stochastic_positions: bool,
        stochastic_agent_control: bool,
        duelling_self_play: bool,
        lap_variation_range: (i8, i8),
        simulation_type: &str,
        track_name: &str,
        time_step: f64,
        mut agent_selected_fields: Vec<String>,
        mut competitor_selected_fields: Vec<String>,
        stack_size: usize,
        normalize_observations: bool,
        norm_stats_inference_only: bool,
        norms_stats_paths: &str,
        obs_size: usize,
        single_step_obs_dim: usize,
    ) -> Self {
        let mut drivers: Vec<Driver> = Vec::with_capacity(22);

        for sim_data in simulation_data.iter() {
            let data = sim_data.clone();
            let mut driver = drivers_hash_map[&data.name].clone();
            let mut sim_data = data.into_sim_data();

            match driver.is_agent() {
                true => {
                    let strategy = mem::take(&mut sim_data.strategy);
                    driver.set_driver_starting_point(sim_data);

                    let starting_compound = Driver::get_starting_compound(strategy);

                    driver.precompute_agent_info(&starting_compound, &race_config);
                }
                false => driver.setup_for_simulation(sim_data, &race_config),
            }

            drivers.push(driver);
        }

        let num_drivers = drivers.len();
        let observation_type =
            ObservationType::new(race_perspective, num_drivers, single_step_obs_dim);

        if matches!(observation_type, ObservationType::Graph(_)) {
            agent_selected_fields.sort();
            competitor_selected_fields.sort();

            println!("{:?}", agent_selected_fields);

            assert_eq!(
                agent_selected_fields, competitor_selected_fields,
                "Graph observation mode requires identical field lists for all nodes.\nAgent fields: {:?}\nCompetitor fields: {:?}",
                agent_selected_fields, competitor_selected_fields,
            )
        }

        let env = RaceEnvType::new(
            drivers,
            race_config,
            simulation_data,
            alternate_strategies,
            stochastic_competitor_strategies,
            stochastic_starting_compound,
            stochastic_positions,
            stochastic_agent_control,
            duelling_self_play,
            lap_variation_range,
            simulation_type,
            track_name,
            time_step,
            agent_selected_fields,
            competitor_selected_fields,
        );

        let action_complexity_type = ActionComplexityType::new(action_complexity);
        let steps_needed = Some(0);
        let complex_action = (None, 0);
        let obs_rms = if normalize_observations {
            println!("Normalizing Observations");

            let batch_size = obs_size / stack_size;
            println!(
                "The obs size: {}, the stack size: {}, the batch_size; {}, the single step observation size; {}",
                obs_size, stack_size, batch_size, single_step_obs_dim
            );
            let shared_norm =
                SharedNormalization::new(single_step_obs_dim, norm_stats_inference_only);
            if norm_stats_inference_only {
                let path = Path::new(norms_stats_paths);
                shared_norm.load_stats(path);
            };
            println!("The count: {}", shared_norm.rms.read().unwrap().count);
            Some(shared_norm)
        } else {
            None
        };
        Self {
            env,
            steps_needed,
            complex_action_taken: complex_action,
            observations: StackedObservations::new(stack_size),
            shared_normalization: obs_rms,
            obs_size,
            single_step_obs_dim,
            stack_size,
            obs_type: observation_type,
            action_complexity_type,
            num_drivers,
        }
    }

    pub fn get_agent_name(&self) -> &std::string::String {
        let name = match &self.env {
            RaceEnvType::LapDiscrete(race_env) => &race_env.race_env_sim.get_agent_driver().name,
            RaceEnvType::TimeDiscrete(race_env) => &race_env.race_env_sim.get_agent_driver().name,
        };

        name
    }

    fn step_environment(&mut self) -> (bool, Vec<f32>) {
        let (done, obs) = self._step();

        let obs = self.flatten_observations_smart(obs);
        (done, obs)
    }

    pub fn step_dict<'a>(&mut self) -> (bool, HashMap<&str, Vec<f32>>) {
        let (done, obs) = self._step();

        let obs = self.make_vector_from_dict(obs);
        (done, obs)
    }

    fn _step(&mut self) -> (bool, Vec<DriverObservation>) {
        let (done, obs) = match &mut self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.step(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.step(),
        };
        (done, obs)
    }

    fn reset_environment(&mut self) -> Vec<f32> {
        let obs = self._reset();

        let obs = self.flatten_observations_smart(obs);

        obs
    }

    pub fn reset_dict<'a>(&mut self) -> HashMap<&str, Vec<f32>> {
        let obs = self._reset();

        let obs = self.make_vector_from_dict(obs);

        obs
    }

    pub fn _reset(&mut self) -> Vec<DriverObservation> {
        let obs = match &mut self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.reset(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.reset(),
        };

        self.observations.clear();
        obs
    }

    pub fn get_current_state_observation<'a>(&self) -> Vec<f32> {
        let obs = match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.get_current_state_observation(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.get_current_state_observation(),
        };
        let obs = self.get_array_observation(obs);

        obs
    }
    fn clean_branching_variables(&mut self) {
        self.complex_action_taken = (None, 0);
        self.steps_needed = Some(0);
    }

    pub fn branch_from_state(&self) -> RaceStrategyEnvironment {
        let mut branch = self.clone();

        match self.steps_needed {
            Some(n) => {
                let (compound, pitting_lap) = &self.complex_action_taken;
                for _ in 0..n {
                    if branch.get_agent_current_lap() == *pitting_lap && compound.is_some() {
                        branch.simple_action(compound.as_deref());
                    } else {
                        branch.simple_action(None);
                    }

                    branch.step_environment();
                }
            }
            None => {
                branch.simple_action(None);
                while !branch.step_environment().0 {
                    branch.simple_action(None);
                    // keep stepping until done, agent does not want to perform any stops
                }
            }
        }

        // Setting the steps needed for branch to zero, since i cloned it would have remanats of the parent
        branch.steps_needed = Some(0);
        branch.complex_action_taken = (None, 0);
        branch
    }

    pub fn print_grid(&self) {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.print_grid(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.print_grid(),
        }
    }

    pub fn get_agent_current_lap(&self) -> u8 {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.get_agent_current_lap(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.get_agent_current_lap(),
        }
    }

    fn get_env_current_step(&self) -> usize {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.race_env_sim.get_current_step(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.race_env_sim.get_current_step(),
        }
    }

    pub fn get_agent_race_progress(&self) -> f64 {
        let agent = self.get_agent();
        agent.driver_race_progress
    }

    fn get_agent(&self) -> &Driver {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.race_env_sim.get_agent_driver(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.race_env_sim.get_agent_driver(),
        }
    }

    pub fn get_agent_current_lap_progress(&self) -> f64 {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.get_agent_current_lap_progress(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.get_agent_current_lap_progress(),
        }
    }

    pub fn get_pit_lane_entry(&self) -> Option<f64> {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.get_pit_lane_entry(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.get_pit_lane_entry(),
        }
    }

    pub fn get_agent_compliance(&self) -> (bool, bool, bool) {
        let compliance = self.get_race_compliance();

        (
            compliance.compound_compliant,
            compliance.pit_lane_compliant,
            compliance.regulatory_compliant,
        )
    }

    fn _get_complex_action_mask(&self) -> Vec<f32> {
        let actions = self.available_complex_actions();
        let num_compounds = self.get_num_compounds();
        let num_laps = self.get_race_config().num_laps;
        let current_lap = self.get_agent_current_lap();
        let race_progress = self.get_agent_race_progress();

        if race_progress > num_laps.into() {
            // for time discrete. if we are on the final lap you cant pit and end the race in the pit lane
            let mut td_mask = vec![0.0; actions.len()];
            *td_mask.last_mut().unwrap() = 1.0;
            return td_mask;
        }

        let compound_compliance = self.get_race_compliance().compound_compliant;

        let mut mask = vec![1.0; actions.len()];

        // This done because in time discrete because of discretization when stepping when driver completes his final
        // lap his current lap will be slighly over race laps
        let lap = min(current_lap, num_laps) as usize;

        let past_actions = (lap - 1) * num_compounds;
        for i in 0..past_actions {
            mask[i] = 0.0;
        }

        // if agent have not used both compounds you cant finish race, therefore cant jump to branch

        if !compound_compliance {
            let current_compound = self.get_current_stint().0;
            *mask.last_mut().unwrap() = 0.0;
            for (index, action) in (actions.iter().enumerate()).rev() {
                if let Some(compound) = action {
                    if **compound == current_compound {
                        mask[index] = 0.0;
                        break;
                    }
                }
            }
        }

        mask
    }

    fn _get_simple_action_mask(&self) -> Vec<f32> {
        let actions = self.available_simple_actions();
        let current_lap = self.get_agent_current_lap();

        let compound_compliance = self.get_race_compliance().compound_compliant;

        let num_laps = self.get_race_config().num_laps;
        let race_progress = self.get_agent_race_progress();

        if race_progress > num_laps.into() {
            // for time discrete. if we are on the final lap you cant pit and end the race in the pit lane
            let mut td_mask = vec![0.0; actions.len()];
            *td_mask.last_mut().unwrap() = 1.0;
            return td_mask;
        }

        let mut mask = vec![1.0; actions.len()];
        // different to complex action because there is no option to 'complete' race with None, but a decision
        // is made at every step just need to make sre before the race end the agent makes a compliant choice
        let penultimate_lap = num_laps - 1;
        if current_lap >= penultimate_lap {
            if !compound_compliance {
                let current_compound = self.get_current_stint().0;
                *mask.last_mut().unwrap() = 0.0;
                for (index, action) in actions.iter().enumerate() {
                    if let Some(compound) = action {
                        if **compound == current_compound {
                            mask[index] = 0.0;
                            break;
                        }
                    }
                }
            }
        }

        mask
    }

    fn get_action_mask(&self) -> Vec<f32> {
        let action_mask = match &self.action_complexity_type {
            ActionComplexityType::Simple => self._get_simple_action_mask(),
            ActionComplexityType::Complex => self._get_complex_action_mask(),
        };
        action_mask
    }

    pub fn get_num_compounds(&self) -> usize {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env
                .race_env_sim
                .get_agent_driver()
                .get_all_compounds()
                .len(),
            RaceEnvType::TimeDiscrete(race_env) => race_env
                .race_env_sim
                .get_agent_driver()
                .get_all_compounds()
                .len(),
        }
    }

    pub fn get_current_stint(&self) -> (String, u8) {
        let current_stint = match &self.env {
            RaceEnvType::LapDiscrete(race_env) => {
                race_env.race_env_sim.get_agent_driver().get_current_stint()
            }
            RaceEnvType::TimeDiscrete(race_env) => {
                race_env.race_env_sim.get_agent_driver().get_current_stint()
            }
        };

        let CurrentStint {
            current_compound,
            current_tyre_age,
            current_lap: _,
        } = current_stint;

        (current_compound, current_tyre_age)
    }

    pub fn get_race_config(&self) -> RaceConfiguration {
        let race_config = match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.race_env_sim.get_race_config(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.race_env_sim.get_race_config(),
        };

        race_config
    }

    pub fn get_num_drivers(&self) -> usize {
        self.num_drivers
    }

    pub fn num_active_drivers(&self) -> usize {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.race_env_sim.get_active_drivers(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.race_env_sim.get_active_drivers(),
        }
    }

    pub fn create_fully_connected_edge_index(
        &self,
        include_self_loops: bool,
    ) -> Vec<(usize, usize)> {
        let num_active_drivers = self.num_active_drivers();

        create_fully_connected_edge_index(num_active_drivers, include_self_loops)
    }

    fn simple_action(&mut self, compound: Option<&str>) {
        match &mut self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.simple_action(compound),
            RaceEnvType::TimeDiscrete(race_env) => race_env.simple_action(compound),
        }
    }

    fn complex_action(&mut self, compound: Option<String>, pitting_lap: u8) {
        match &mut self.env {
            RaceEnvType::LapDiscrete(race_env) => {
                race_env.complex_action(compound.as_deref(), pitting_lap)
            }
            RaceEnvType::TimeDiscrete(race_env) => {
                race_env.complex_action(compound.as_deref(), pitting_lap)
            }
        }

        self.create_branching_variables(compound, pitting_lap);
    }

    fn create_branching_variables(&mut self, compound: Option<String>, pitting_lap: u8) {
        self.steps_needed = if compound.is_none() {
            None
        } else {
            Some(pitting_lap - self.get_agent_current_lap())
        };

        self.complex_action_taken = (compound, pitting_lap);
    }

    fn take_action(&mut self, action: Action) {
        match action {
            Action::Simple { compound } => self.simple_action(compound.as_deref()),
            Action::Complex {
                compound,
                pitting_lap,
            } => self.complex_action(compound, pitting_lap),
        }
    }

    fn available_simple_actions(&self) -> Vec<Option<&String>> {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.available_simple_actions(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.available_simple_actions(),
        }
    }

    fn available_complex_actions(&self) -> Vec<Option<&String>> {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.available_complex_actions(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.available_complex_actions(),
        }
    }

    pub fn available_actions(&self) -> Vec<Option<&String>> {
        match self.action_complexity_type {
            ActionComplexityType::Simple => self.available_simple_actions(),
            ActionComplexityType::Complex => self.available_complex_actions(),
        }
    }

    fn flatten_observations_smart(&mut self, observations: Vec<DriverObservation>) -> Vec<f32> {
        let array = self.get_array_observation(observations);

        let stacked = self.stack_observations(array);

        stacked
    }

    fn get_array_observation(&self, observations: Vec<DriverObservation>) -> Vec<f32> {
        let mut array = Vec::with_capacity(observations.len() * 200);

        match self.obs_type {
            ObservationType::Agent => {
                for obs in observations {
                    if obs.is_agent {
                        let driver_vector = self.get_driver_vector(obs);
                        array.extend(driver_vector);

                        break;
                    }
                }
            }
            ObservationType::Drivers => {
                for obs in observations {
                    let driver_vector = self.get_driver_vector(obs);
                    array.extend(driver_vector);
                }
            }
            ObservationType::Graph(_) => {
                for obs in observations {
                    let driver_vector = self.get_driver_vector(obs);
                    array.extend(driver_vector);
                }
            }
        }
        array
    }

    fn stack_observations(&mut self, array: Vec<f32>) -> Vec<f32> {
        let normilzed_array = if let Some(ref norm) = self.shared_normalization {
            let obs = norm.normalize(array);
            obs
        } else {
            array
        };
        self.observations.add_observation(normilzed_array);

        let stacked_array = self.observations.flatten_observations();
        stacked_array
    }

    fn get_driver_vector(&self, obs: DriverObservation) -> Vec<f32> {
        let driver_vector = match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.get_selected_driver_fields_vector(obs),
            RaceEnvType::TimeDiscrete(race_env) => race_env.get_selected_driver_fields_vector(obs),
        };
        driver_vector
    }

    pub fn make_vector_from_dict(
        &self,
        observations: Vec<DriverObservation>,
    ) -> HashMap<&str, Vec<f32>> {
        let mut dict = HashMap::with_capacity(2);
        let mut agent_array = Vec::with_capacity(100);
        let mut drivers_array = Vec::with_capacity((observations.len() - 1) * 200);

        for obs in observations {
            if obs.is_agent {
                let driver_vector = self.get_driver_vector(obs);
                agent_array.extend(driver_vector);
            } else {
                let driver_vector = self.get_driver_vector(obs);
                drivers_array.extend(driver_vector);
            }
        }
        dict.insert("agent", agent_array);
        dict.insert("drivers", drivers_array);

        dict
    }

    fn translate_action(&self, action: usize) -> Action {
        let action = match self.action_complexity_type {
            ActionComplexityType::Simple => {
                let action_index = action;
                let actions = self.available_simple_actions();
                let compound = actions[action_index].cloned();
                Action::Simple { compound }
            }
            ActionComplexityType::Complex => {
                let action_index = action;
                let actions = self.available_complex_actions();
                let compound = actions[action_index].cloned();
                let pitting_lap = ((action_index / self.get_num_compounds()) + 1) as u8;

                Action::Complex {
                    compound,
                    pitting_lap,
                }
            }
        };

        action
    }

    fn calculate_reward(&self, done: bool) -> f32 {
        let mut reward = 0.0;

        if !done {
            return reward;
        };

        if !self.get_race_compliance().regulatory_compliant {
            reward = -1000.0;
            return reward;
        };

        let result = self.get_agent_results();

        let time_reward = -result.race_time / 10.0;
        let race_position = result.position as f32;
        let position_reward = position_reward(race_position, self.num_drivers as f32);

        reward = position_reward + time_reward;

        reward
    }

    fn get_max_env_steps(&self) -> usize {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.race_env_sim.get_max_steps(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.race_env_sim.get_max_steps(),
        }
    }

    fn get_action_space(&self) -> usize {
        let action_space = match self.action_complexity_type {
            ActionComplexityType::Simple => self.available_simple_actions().len(),
            ActionComplexityType::Complex => self.available_complex_actions().len(),
        };
        action_space
    }

    pub fn encode_strategy(&self, strategy: &[(String, u8)]) -> Vec<f32> {
        let num_compounds = self.get_num_compounds();
        let num_laps = self.get_race_config().num_laps as usize;
        let encoding_capacity = num_compounds * num_laps;
        let mut strategy_encoding = Vec::with_capacity(encoding_capacity);
        for (compound, laps) in strategy {
            let mut lap_choice = vec![0.0; num_compounds];

            let compund_index = self.compound_index_map(compound) as usize;
            let num_laps = *laps as usize;

            lap_choice[compund_index] = 1.0;

            let encoded_stint = lap_choice.repeat(num_laps);

            strategy_encoding.extend(encoded_stint);
        }

        strategy_encoding.resize(encoding_capacity, 0.0);

        strategy_encoding
    }

    pub fn decode_strategy(&self, encoded_strategy: &[f32]) -> (Vec<(String, u8)>, Vec<String>) {
        let num_laps = self.get_race_config().num_laps as usize;
        let mut decoded_strategy: Vec<(String, u8)> = Vec::with_capacity(num_laps);

        let mut strategy_compounds = Vec::with_capacity(num_laps);
        let num_compounds = self.get_num_compounds();

        for lap_choice in encoded_strategy.chunks_exact(num_compounds) {
            let index = argmax(lap_choice);
            let current_compound_name = self.compound_index_to_compound_name(index);

            strategy_compounds.push(current_compound_name.to_string());
            match decoded_strategy.last_mut() {
                Some(current_stint) => {
                    let (compound, lap) = current_stint;

                    // the next lap the agent is using the same compound so increment
                    if current_compound_name == compound {
                        *lap += 1
                    }
                    // the agent has changed compounds
                    else {
                        let stint = (current_compound_name.to_string(), 1);
                        decoded_strategy.push(stint);
                    }
                }

                // the strategy is empty so add the first stint
                None => {
                    let stint = (current_compound_name.to_string(), 1);
                    decoded_strategy.push(stint);
                }
            }
        }

        (decoded_strategy, strategy_compounds)
    }

    fn compound_index_to_compound_name(&self, index: usize) -> &str {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.compound_index_to_compound_name(index),
            RaceEnvType::TimeDiscrete(race_env) => race_env.compound_index_to_compound_name(index),
        }
    }

    fn compound_index_map(&self, compound: &String) -> f64 {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.compound_index_map(compound),
            RaceEnvType::TimeDiscrete(race_env) => race_env.compound_index_map(compound),
        }
    }

    pub fn save_normalization_stats(&self, path: &Path) {
        match &self.shared_normalization {
            Some(shared_norm) => shared_norm.save_stats(path),
            None => (),
        };
    }

    pub fn load_normalization_stats(&self, path: &Path) {
        match &self.shared_normalization {
            Some(shared_norm) => shared_norm.load_stats(path),
            None => (),
        };
    }
}

impl GymEnvironment for RaceStrategyEnvironment {
    type Observation = Vec<f32>;

    type Reward = f32;

    type Terminated = bool;

    type Truncated = bool;

    type Info = HashMap<String, AgentInfo>;

    fn reset(&mut self) -> (Self::Observation, Self::Info) {
        let obs = self.reset_environment();
        let agent_info = self.get_agent_results();
        let mut info = HashMap::new();
        info.insert("Agent".to_string(), agent_info);

        (obs, info)
    }

    fn step(
        &mut self,
        action: usize,
    ) -> (
        Self::Observation,
        Self::Reward,
        Self::Terminated,
        Self::Truncated,
        Self::Info,
    ) {
        let action = self.translate_action(action);
        self.take_action(action);
        let (done, obs) = self.step_environment();
        let agent_info = self.get_agent_results();
        let mut info = HashMap::new();
        info.insert("Agent".to_string(), agent_info);

        let reward = self.calculate_reward(done);

        (obs, reward, done, done, info)
    }

    fn get_current_step(&self) -> usize {
        self.get_env_current_step()
    }

    fn get_current_significant_step(&self) -> usize {
        self.get_agent_current_lap() as usize
    }

    fn action_space(&self) -> usize {
        let action_space = self.get_action_space();

        action_space
    }

    fn observation_size(&self) -> usize {
        self.obs_size
    }

    fn stack_size(&self) -> usize {
        self.stack_size
    }

    fn max_steps(&self) -> usize {
        self.get_max_env_steps()
    }

    fn save_norm_stats(&self, path: &Path) {
        self.save_normalization_stats(path);
    }

    fn load_norm_stats(&self, path: &Path) {
        self.load_normalization_stats(path);
    }

    fn single_step_obs_dim(&self) -> usize {
        self.single_step_obs_dim
    }
}

impl MCTSGymEnvironment for RaceStrategyEnvironment {
    fn branch(&self) -> Self {
        self.branch_from_state()
    }

    fn get_legal_actions(&self) -> Vec<f32> {
        let action_mask = self.get_action_mask();

        action_mask
    }

    fn clear(&mut self) {
        self.clean_branching_variables();
    }

    fn show_info(&self) {
        self.print_grid();

        println!(
            "count: {}, first: {}",
            self.shared_normalization
                .as_ref()
                .unwrap()
                .rms
                .read()
                .unwrap()
                .count,
            self.shared_normalization
                .as_ref()
                .unwrap()
                .rms
                .read()
                .unwrap()
                .mean[0]
        )
    }

    fn get_current_encoded_strategy(&self) -> Vec<f32> {
        let agent_info = self.get_agent_results();
        let lap_pitted_on = agent_info.laps_pitted_on.len();
        // this is to exclude the next compound the agent has select
        let strategy = &agent_info.strategy[0..=lap_pitted_on];
        let current_encoded_strategy = self.encode_strategy(strategy);
        current_encoded_strategy
    }
}

fn position_reward(race_position: f32, total_drivers: f32) -> f32 {
    let base_reward = ((total_drivers - race_position + 1.0) / total_drivers) * 100.0;

    let top_10_bonus = get_f1_points(race_position as u8) as f32 * 100.0;

    base_reward + top_10_bonus
}

impl RaceStrategyEnvironment {
    fn get_race_compliance(&self) -> RaceCompliance {
        let compliance = match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.get_agent_compliance(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.get_agent_compliance(),
        };
        compliance
    }

    pub fn get_agent_results(&self) -> AgentInfo {
        match &self.env {
            RaceEnvType::LapDiscrete(race_env) => race_env.get_agent_results(),
            RaceEnvType::TimeDiscrete(race_env) => race_env.get_agent_results(),
        }
    }
}
