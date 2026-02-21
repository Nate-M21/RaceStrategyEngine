use std::collections::HashMap;

use burn::prelude::Backend;

use crate::{
    algorithms::{
        muzero::{muzero_config::MuzeroConfig, muzero_mcts::Node},
        strategy::RaceStrategyEnvironment,
    },
    environment::AgentInfo,
    traits::gym::{GymEnvironment, MCTSGymEnvironment},
};

#[derive(Clone)]
pub struct Game {
    pub environment: RaceStrategyEnvironment,
    pub history: Vec<Action>,
    rewards: Vec<f32>,
    observations: Vec<Vec<f32>>,
    child_visits: Vec<Vec<f32>>,
    root_values: Vec<f32>,
    action_space_size: usize,
    discount: f32,
    terminal: bool, // extra for me
}

impl Game {
    pub fn new_game(environment: RaceStrategyEnvironment, config: MuzeroConfig) -> Game {
        let discount = config.discount;
        let action_space_size = config.action_space;

        let observations = Vec::with_capacity(config.max_moves);

        Self {
            environment,
            history: Default::default(),
            rewards: Default::default(),
            observations,
            child_visits: Default::default(),
            root_values: Default::default(),
            action_space_size,
            discount,
            terminal: false,
        }
    }
    pub fn terminal(&self) -> bool {
        self.terminal
    }

    pub fn legal_actions(&self) -> Vec<f32> {
        self.environment.get_legal_actions()
    }

    pub fn reset(&mut self) -> (Vec<f32>, HashMap<String, AgentInfo>) {
        let (observation, info) = self.environment.reset();
        self.observations.push(observation.clone());

        (observation, info)
    }

    pub fn clear(&mut self) {
        self.environment.clear();
    }

    pub fn apply(&mut self, action: Action) -> (Vec<f32>, HashMap<String, AgentInfo>) {
        let (observation, reward, terminated, truncated, info) =
            self.environment.step(action.index);
        self.rewards.push(reward);
        self.history.push(action);
        self.observations.push(observation.clone());
        self.terminal = terminated || truncated;

        (observation, info)
    }

    pub fn show_info(&self) {
        self.environment.show_info();
    }

    pub fn store_search_statistics<B: Backend>(&mut self, root: &Node<B>) {
        let mut sum_visits = 0;
        for child in root.children_dict.values() {
            sum_visits += child.visit_count;
        }

        let mut action_space = Vec::with_capacity(self.action_space_size);
        for index in 0..self.action_space_size {
            action_space.push(Action { index });
        }

        let mut visits = Vec::with_capacity(self.action_space_size);
        for action in action_space {
            if let Some(child) = root.children_dict.get(&action.index) {
                visits.push(child.visit_count as f32 / sum_visits as f32)
            } else {
                visits.push(0.0);
            }
        }
        self.child_visits.push(visits);

        self.root_values.push(root.value());
    }

    pub fn make_image(&self, state_index: usize) -> Vec<f32> {
        // i dont know what this is supposed to be when iim not doing images but rather MLP
        self.observations[state_index].clone()
    }

    pub fn make_targets(
        &self,
        state_index: usize,
        num_unroll_steps: u32,
        td_steps: u32,
        _to_play: Player,
        config: MuzeroConfig,
    ) -> Vec<(f32, Option<f32>, Vec<f32>)> {
        let mut targets = Vec::new();

        for current_index in state_index..state_index + num_unroll_steps as usize + 1 {
            let bootstrap_index = current_index + td_steps as usize;

            let mut value = if bootstrap_index < self.root_values.len() {
                self.root_values[bootstrap_index] * self.discount.powf(td_steps as f32)
            } else {
                0.0
            };

            // Clamp both start and end indices
            let start_index = current_index.min(self.rewards.len());
            let end_index = bootstrap_index.min(self.rewards.len());

            // Only iterate if start < end
            if start_index < end_index {
                for (index, reward) in (self.rewards[start_index..end_index]).iter().enumerate() {
                    value += reward * self.discount.powf(index as f32)
                }
            }

            let last_reward = if current_index > 0 && current_index <= self.rewards.len() {
                Some(self.rewards[current_index - 1])
            } else {
                None
            };

            if current_index < self.root_values.len() {
                targets.push((value, last_reward, self.child_visits[current_index].clone()));
            } else {
                let action_space_size = config.action_space;
                let uniform_prob = 1.0 / action_space_size as f32;
                let absorbing_state_action_probabilities = vec![uniform_prob; action_space_size];
                targets.push((0.0, last_reward, absorbing_state_action_probabilities));
            }
        }

        targets
    }

    pub fn to_play(&self) -> Player {
        Player {}
    }

    pub fn action_history(&self) -> ActionHistory {
        ActionHistory {
            history: self.history.clone(),
            action_space_size: self.action_space_size,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Action {
    pub index: usize,
}

impl Action {
    pub fn new(index: usize) -> Self {
        Self { index }
    }
}

pub struct Player {}

#[derive(Debug, Clone)]
pub struct ActionHistory {
    history: Vec<Action>,
    action_space_size: usize,
}

impl ActionHistory {
    pub fn add_action(&mut self, action: Action) {
        self.history.push(action);
    }

    pub fn last_action(&self) -> Action {
        self.history
            .last()
            .expect("Failed to get last action as there isnt one")
            .clone()
    }

    pub fn action_space(&self) -> Vec<Action> {
        let mut actions = Vec::with_capacity(self.action_space_size);
        for index in 0..self.action_space_size {
            actions.push(Action { index });
        }

        actions
    }

    pub fn to_play(&self) -> Player {
        Player {}
    }
}
