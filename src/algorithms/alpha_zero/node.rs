use std::{collections::HashMap, f32::NEG_INFINITY};

use crate::{
    algorithms::{
        alpha_zero::alpha_zero_config::AlphaZeroConfig, muzero_old::muzero_mcts::MinMaxStats,
    },
    traits::gym::MCTSGymEnvironment,
    utils::argmax,
};

pub struct AlphaZeroNode<Environment: MCTSGymEnvironment> {
    pub state: Environment,
    pub reward: Environment::Reward,
    pub done: bool,
    pub current_observation: Environment::Observation,
    config: AlphaZeroConfig,
    pub action: Option<usize>,

    pub valid_actions: Vec<f32>,

    pub children_dict: HashMap<usize, AlphaZeroNode<Environment>>,
    pub expanded: bool,

    pub visit_count: u32,
    pub value_sum: f32,

    pub children_action_probabilties: Vec<f32>,
}

impl<Environment: MCTSGymEnvironment> AlphaZeroNode<Environment> {
    pub fn new_root(
        state: Environment,
        reward: Environment::Reward,
        done: bool,
        current_observation: Environment::Observation,
        config: AlphaZeroConfig,
        action_to_take: Option<usize>,
        visit_count: u32,
    ) -> AlphaZeroNode<Environment> {
        let state = state.branch();
        let valid_actions = state.get_legal_actions();
        Self {
            state,
            reward,
            done,
            current_observation,
            config,
            action: action_to_take,
            valid_actions,
            children_dict: HashMap::new(),
            expanded: false,
            visit_count,
            value_sum: 0.0,
            children_action_probabilties: Vec::new(),
        }
    }
    pub fn select(&mut self, min_max_stats: &MinMaxStats) -> &mut AlphaZeroNode<Environment> {
        let children_puct_values = self.get_children_puct_values(min_max_stats);
        let action = argmax(&children_puct_values);

        // this should work, but doesnt
        // let child = if let Some(child) = self.children_dict.get_mut(&action) {
        //     child
        // } else {
        //     self.create_child_node(action)
        // };

        let child = if self.children_dict.contains_key(&action) {
            self.get_child_node(action)
        } else {
            self.create_child_node(action)
        };

        child
    }

    pub fn expand(&mut self, action_probabilties: Vec<f32>) {
        self.expanded = true;
        self.children_action_probabilties = action_probabilties
    }

    pub fn new_child(
        state: Environment,
        reward: Environment::Reward,
        done: bool,
        current_observation: Environment::Observation,
        config: AlphaZeroConfig,
        action_to_take: usize,
    ) -> AlphaZeroNode<Environment> {
        let state = state.branch();
        let valid_actions = state.get_legal_actions();
        Self {
            state,
            reward,
            done,
            current_observation,
            config,
            action: Some(action_to_take),
            valid_actions,
            children_dict: HashMap::new(),
            expanded: false,
            visit_count: 0,
            value_sum: 0.0,
            children_action_probabilties: Vec::new(),
        }
    }

    fn get_children_puct_values(&self, min_max_stats: &MinMaxStats) -> Vec<f32> {
        let mut puct_values = vec![NEG_INFINITY; self.state.action_space()];

        for (action, child_prior_action_probability) in
            self.children_action_probabilties.iter().enumerate()
        {
            if *child_prior_action_probability > 0.0 {
                let (child_value_sum, child_visit_count) =
                    if let Some(child) = self.children_dict.get(&action) {
                        (child.value_sum, child.visit_count as f32)
                    } else {
                        (0.0, 0.0)
                    };

                let q_value = if child_visit_count > 0.0 {
                    child_value_sum / child_visit_count
                } else {
                    0.0
                };

                let normalized_q_value = min_max_stats.normalize(q_value);

                let exploration = self.config.exploration_constant
                    * child_prior_action_probability
                    * f32::sqrt(self.visit_count as f32)
                    / (1.0 + child_visit_count);

                puct_values[action] = normalized_q_value + exploration
            }
        }

        puct_values
    }

    pub fn get_child_node(&mut self, action: usize) -> &mut AlphaZeroNode<Environment> {
        self.children_dict.get_mut(&action).unwrap()
    }

    pub fn create_child_node(&mut self, action: usize) -> &mut AlphaZeroNode<Environment> {
        let mut child_branch = self.state.clone();

        let (obs, reward, terminated, truncated, _info) = child_branch.step(action.clone());
        let done = terminated.into() || truncated.into();

        let child =
            AlphaZeroNode::new_child(child_branch, reward, done, obs, self.config, action.clone());

        self.children_dict.insert(action.clone(), child);

        self.children_dict.get_mut(&action).unwrap()
    }

    pub fn children(
        &self,
    ) -> std::collections::hash_map::Values<'_, usize, AlphaZeroNode<Environment>> {
        self.children_dict.values()
    }

    pub fn value(&self) -> f32 {
        if self.visit_count == 0 {
            return 0.0;
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}
