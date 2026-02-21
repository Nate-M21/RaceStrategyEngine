use std::collections::HashMap;

use burn::{Tensor, prelude::Backend};

use crate::{
    algorithms::stochastic_muzero::{
        stochastic_muzero::MinMaxStats, stochastic_muzero_config::StochasticMuzeroConfig,
    },
    utils::argmax,
};

pub struct MuZeroNode<B: Backend> {
    pub visit_count: u32,
    pub value_sum: f32,
    pub children_dict: HashMap<usize, MuZeroNode<B>>,
    pub reward: f32,
    pub latent_representation: Option<Tensor<B, 2>>,
    pub prior: f32,
    pub expanded: bool,
    pub children_action_probabilties: Vec<f32>,
}

impl<B: Backend> MuZeroNode<B> {
    pub fn select_action(
        &self,
        min_max_stats: &MinMaxStats,
        config: &StochasticMuzeroConfig,
    ) -> usize {
        let children_puct_values = self.get_children_puct_values(min_max_stats, config);

        let action = argmax(&children_puct_values);

        action
    }

    pub fn expand(&mut self, action_probabilties: Vec<f32>) {
        self.expanded = true;
        self.children_action_probabilties = action_probabilties
    }

    fn get_children_puct_values(
        &self,
        min_max_stats: &MinMaxStats,
        config: &StochasticMuzeroConfig,
    ) -> Vec<f32> {
        let mut puct_values = vec![f32::NEG_INFINITY; config.action_space];

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
                    (self.visit_count as f32 + config.pb_c_base as f32 + 1.0)
                        / config.pb_c_base as f32,
                ) + config.pb_c_init as f32;
                pb_c *= f32::sqrt(self.visit_count as f32) / (child_visit_count + 1.0);

                let value_score = if child_visit_count > 0.0 {
                    let q_value = child_value_sum / child_visit_count;
                    min_max_stats.normalize(child_reward + config.discount * q_value)
                } else {
                    0.0
                };

                let prior_score = pb_c * child_prior_action_probability;

                puct_values[action] = value_score + prior_score
            }
        }

        puct_values
    }
}
