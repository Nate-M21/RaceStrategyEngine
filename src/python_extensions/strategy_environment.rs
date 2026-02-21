use std::{collections::HashMap, mem};

use numpy::{IntoPyArray, PyArray, PyArrayMethods, ndarray::Dim};
use pyo3::{Bound, Python, pyclass, pymethods};

use crate::{
    algorithms::strategy::{
        ObservationType, RaceStrategyEnvironment as RustRaceStrategyEnvironment,
    },
    driver::Driver,
    race_config::RaceConfiguration,
    race_simulation::SimulationData,
    traits::gym::{GymEnvironment, MCTSGymEnvironment},
};

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct RaceStrategyEnvironment {
    race_strategy_environment: RustRaceStrategyEnvironment,
}

#[pymethods]
impl RaceStrategyEnvironment {
    #[new]
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
        agent_selected_fields: Vec<String>,
        competitor_selected_fields: Vec<String>,
        stack_size: usize,
        normalize_observations: bool,
        norm_stats_inference_only: bool,
        norms_stats_paths: &str,
        obs_size: usize,
        single_step_obs_dim: usize,
    ) -> Self {
        let mut drivers: Vec<Driver> = Vec::with_capacity(22);

        for sim_params in simulation_data.iter() {
            let data = sim_params.clone();
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
        let rust_env = RustRaceStrategyEnvironment::new(
            drivers_hash_map,
            race_config,
            race_perspective,
            action_complexity,
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
            stack_size,
            normalize_observations,
            norm_stats_inference_only,
            norms_stats_paths,
            obs_size,
            single_step_obs_dim,
        );

        Self {
            race_strategy_environment: rust_env,
        }
    }

    pub fn get_agent_name(&self) -> &String {
        let name = self.race_strategy_environment.get_agent_name();

        name
    }

    fn step<'a>(
        &mut self,
        action: usize,
        py: Python<'a>,
    ) -> (bool, pyo3::Bound<'a, PyArray<f32, Dim<[usize; 1]>>>) {
        let (obs, _reward, terminated, truncated, _info) =
            self.race_strategy_environment.step(action);
        let done = terminated || truncated;
        let obs = obs.into_pyarray(py);

        (done, obs)
    }

    fn step_graph<'a>(
        &mut self,
        action: usize,
        py: Python<'a>,
    ) -> (bool, pyo3::Bound<'a, PyArray<f32, Dim<[usize; 3]>>>) {
        let (done, obs) = self.step(action, py);

        let obs = self.transform_into_graph_observation(obs);

        (done, obs)
    }

    pub fn step_dict<'a>(
        &mut self,
        py: Python<'a>,
    ) -> (
        bool,
        HashMap<String, pyo3::Bound<'a, PyArray<f32, Dim<[usize; 1]>>>>,
    ) {
        let (done, obs) = self.race_strategy_environment.step_dict();
        let numpy_obs = convert_to_numpy_dict(obs, py);

        (done, numpy_obs)
    }

    fn reset<'a>(&mut self, py: Python<'a>) -> pyo3::Bound<'a, PyArray<f32, Dim<[usize; 1]>>> {
        let (obs, _info) = self.race_strategy_environment.reset();

        let obs = obs.into_pyarray(py);

        obs
    }

    fn reset_graph<'a>(
        &mut self,
        py: Python<'a>,
    ) -> pyo3::Bound<'a, PyArray<f32, Dim<[usize; 3]>>> {
        let obs = self.reset(py);

        self.transform_into_graph_observation(obs)
    }

    fn reset_dict<'a>(
        &mut self,
        py: Python<'a>,
    ) -> HashMap<String, pyo3::Bound<'a, PyArray<f32, Dim<[usize; 1]>>>> {
        let obs = self.race_strategy_environment.reset_dict();

        let numpy_obs = convert_to_numpy_dict(obs, py);

        numpy_obs
    }

    fn get_current_state_observation<'a>(
        &self,
        py: Python<'a>,
    ) -> pyo3::Bound<'a, PyArray<f32, Dim<[usize; 1]>>> {
        let obs = self
            .race_strategy_environment
            .get_current_state_observation()
            .into_pyarray(py);

        obs
    }

    fn get_current_graph_state_observation<'a>(
        &self,
        py: Python<'a>,
    ) -> pyo3::Bound<'a, PyArray<f32, Dim<[usize; 3]>>> {
        let obs = self.get_current_state_observation(py);

        self.transform_into_graph_observation(obs)
    }

    fn clean_branching_variables(&mut self) {
        self.race_strategy_environment.clear();
    }

    fn branch_from_state(&self) -> RaceStrategyEnvironment {
        let branch = self.race_strategy_environment.branch_from_state();
        let mut new_env = self.clone();
        new_env.race_strategy_environment = branch;

        new_env
    }

    fn print_strategies(&self) {
        self.race_strategy_environment.print_grid();
    }

    fn get_agent_current_lap(&self) -> u8 {
        self.race_strategy_environment.get_agent_current_lap()
    }

    fn get_agent_current_lap_progress(&self) -> f64 {
        self.race_strategy_environment
            .get_agent_current_lap_progress()
    }

    fn get_pit_lane_entry(&self) -> Option<f64> {
        self.race_strategy_environment.get_pit_lane_entry()
    }

    fn get_agent_compliance(&self) -> (bool, bool, bool) {
        self.race_strategy_environment.get_agent_compliance()
    }

    fn get_action_mask<'a>(
        &self,
        py: Python<'a>,
    ) -> pyo3::Bound<'a, PyArray<f32, Dim<[usize; 1]>>> {
        self.race_strategy_environment
            .get_legal_actions()
            .into_pyarray(py)
    }

    fn get_num_compounds(&self) -> usize {
        self.race_strategy_environment.get_num_compounds()
    }

    fn get_current_stint(&self) -> (String, u8) {
        self.race_strategy_environment.get_current_stint()
    }

    fn available_actions(&self) -> Vec<Option<&String>> {
        self.race_strategy_environment.available_actions()
    }

    fn get_num_active_drivers(&self) -> usize {
        self.race_strategy_environment.num_active_drivers()
    }

    fn get_fully_connected_edge_index(&self, include_self_loops: bool) -> Vec<(usize, usize)> {
        self.race_strategy_environment
            .create_fully_connected_edge_index(include_self_loops)
    }

    fn get_agent_results(
        &self,
    ) -> (
        String,
        u8,
        u8,
        f32,
        Vec<(std::string::String, u8)>,
        Vec<u8>,
        u8,
        bool,
    ) {
        let agent_result = self.race_strategy_environment.get_agent_results();

        (
            agent_result.name,
            agent_result.starting_position,
            agent_result.position,
            agent_result.race_time,
            agent_result.strategy,
            agent_result.laps_pitted_on,
            agent_result.different_compounds_used_count,
            agent_result.is_regulatory_compliant,
        )
    }
}

impl RaceStrategyEnvironment {
    fn transform_into_graph_observation<'a>(
        &self,
        obs: pyo3::Bound<'a, PyArray<f32, Dim<[usize; 1]>>>,
    ) -> pyo3::Bound<'a, PyArray<f32, Dim<[usize; 3]>>> {
        if let ObservationType::Graph(state) = self.race_strategy_environment.obs_type {
            let stack_size = self.race_strategy_environment.stack_size();
            let num_active_drivers = self.race_strategy_environment.num_active_drivers();
            let obs = obs
                .reshape((stack_size, num_active_drivers, state.features_per_node))
                .expect("Failed to reshape observation");

            return obs;
        } else {
            panic!("Environment was not set to graph mode")
        }
    }
}

fn convert_to_numpy_dict<'a>(
    dict_observation: HashMap<&str, Vec<f32>>,
    py: Python<'a>,
) -> HashMap<String, pyo3::Bound<'a, PyArray<f32, Dim<[usize; 1]>>>> {
    let mut numpy_obs = HashMap::new();

    for (key, value) in dict_observation {
        let value: Bound<'a, PyArray<f32, Dim<[usize; 1]>>> = value.into_pyarray(py);
        numpy_obs.insert(key.to_owned(), value);
    }
    numpy_obs
}
