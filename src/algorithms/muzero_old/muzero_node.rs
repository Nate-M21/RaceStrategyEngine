use std::collections::HashMap;

use burn::{Tensor, prelude::Backend};

use crate::{
    algorithms::muzero_old::{muzero_config::MuzeroConfig, muzero_mcts::MinMaxStats},
    utils::argmax,
};

pub struct MuzeroNode<B: Backend, const D: usize> {
    pub visit_count: u32,
    pub value_sum: f32,

    pub children_action_probabilties: Vec<f32>,

    pub children_dict: HashMap<usize, MuzeroNode<B, D>>,
    pub reward: f32,
    pub latent_representation: Tensor<B, D>,
    pub expanded: bool,
    pub action: Option<usize>,

    config: MuzeroConfig,
}

impl<B: Backend, const D: usize> MuzeroNode<B, D> {
    pub fn expand(&mut self, action_probabilities: Vec<f32>) {
        self.expanded = true;
        self.children_action_probabilties = action_probabilities
    }

    pub fn select_action(&self, min_max_stats: &MinMaxStats) -> usize {
        let children_puct_values = self.get_children_puct_values(min_max_stats);
        // let children_puct_values = self.get_children_puct_values_az(min_max_stats);

        let action = argmax(&children_puct_values);

        action
    }

    pub fn new_root(
        initial_latent_representation: Tensor<B, D>, // From representation function h(observation) this will be my first mapping from real to latent state
        config: MuzeroConfig,
    ) -> Self {
        Self {
            visit_count: 0,
            value_sum: 0.0,
            children_action_probabilties: Vec::new(),
            children_dict: HashMap::new(),
            reward: 0.0,
            latent_representation: initial_latent_representation,
            expanded: false,
            action: None,
            config,
        }
    }

    fn new_child(
        latent_representation: Tensor<B, D>, //  From dynamics function g(parent_state, action)
        reward: f32,                         //  Also from dynamics function
        action: usize,
        config: MuzeroConfig,
    ) -> Self {
        Self {
            latent_representation,
            reward,
            action: Some(action),
            children_dict: HashMap::new(),
            expanded: false,
            visit_count: 0,
            value_sum: 0.0,
            children_action_probabilties: Vec::new(),
            config,
        }
    }

    pub fn get_child_node(&mut self, action: usize) -> &mut MuzeroNode<B, D> {
        self.children_dict.get_mut(&action).unwrap()
    }

    pub fn create_child_node(
        &mut self,
        action: usize,
        latent_representation: Tensor<B, D>, // From model.dynamics() in the MCTS
        reward: f32,                         // From model.dynamics() in the MCTS
    ) -> &mut MuzeroNode<B, D> {
        let child = MuzeroNode::new_child(latent_representation, reward, action, self.config);

        self.children_dict.insert(action.clone(), child);

        self.children_dict.get_mut(&action).unwrap()
    }

    fn get_children_puct_values(&self, min_max_stats: &MinMaxStats) -> Vec<f32> {
        let mut puct_values = vec![f32::NEG_INFINITY; self.config.action_space];

        for (action, child_prior_action_probability) in
            self.children_action_probabilties.iter().enumerate()
        {
            if *child_prior_action_probability > 0.0 {
                let (child_value_sum, child_visit_count, child_reward) =
                    if let Some(child) = self.children_dict.get(&action) {
                        (child.value_sum, child.visit_count as f32, child.reward)
                    } else {
                        (0.0, 0.0, 0.0)
                    };

                let mut pb_c = f32::ln(
                    (self.visit_count as f32 + self.config.pb_c_base as f32 + 1.0)
                        / self.config.pb_c_base as f32,
                ) + self.config.pb_c_init as f32;
                pb_c *= f32::sqrt(self.visit_count as f32) / (child_visit_count + 1.0);

                let value_score = if child_visit_count > 0.0 {
                    let q_value = child_value_sum / child_visit_count;
                    min_max_stats.normalize(child_reward + self.config.discount * q_value)
                } else {
                    0.0
                };

                let prior_score = pb_c * child_prior_action_probability;

                puct_values[action] = value_score + prior_score
            }
        }

        puct_values
    }

    pub fn children(&self) -> std::collections::hash_map::Values<'_, usize, MuzeroNode<B, D>> {
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
