use std::path::Path;

use crate::{
    algorithms::alpha_zero::{
        alpha_zero_config::AlphaZeroConfig, node::AlphaZeroNode, replay_buffer::ReplayBuffer,
    },
    traits::gym::MCTSGymEnvironment,
};

pub trait ActorCritic: Send + Sync + Clone {
    type TransitionType: Clone;
    type ObservationType;

    fn get_action_probs_and_value<Environment>(
        &self,
        node: &AlphaZeroNode<Environment>,
        apply_dirichlet_noise: bool,
        config: AlphaZeroConfig,
    ) -> (Vec<f32>, f32)
    where
        Environment: MCTSGymEnvironment<Observation = Self::ObservationType>;

    fn get_raw_action_and_value_logits<Environment>(
        &self,
        node: &AlphaZeroNode<Environment>,
    ) -> (Vec<f32>, f32)
    where
        Environment: MCTSGymEnvironment<Observation = Self::ObservationType>;

    fn get_valid_actions<'a, Environment: MCTSGymEnvironment>(
        &self,
        node: &'a AlphaZeroNode<Environment>,
    ) -> &'a [f32] {
        &node.valid_actions
    }

    fn predict(
        &self,
        observation: &[f32],
        current_time_step: Option<usize>,
        legal_actions: Option<&[f32]>,
    ) -> Vec<Vec<f32>>;

    fn get_action_space(&self) -> usize;

    fn get_observation_space(&self) -> usize;

    fn train_model(
        &mut self,
        replay_buffer: &mut ReplayBuffer<Self::TransitionType>,
        config: &AlphaZeroConfig,
    );

    fn save_model(&self, path: &Path);

    fn load_model(&mut self, path: &Path);

    fn predict_with_attention(
        &self,
        observation: &[f32],
        _strategy_encoding: &[f32],
        current_time_step: Option<usize>,
        legal_actions: Option<&[f32]>,
    ) -> (Vec<Vec<f32>>, Option<AttentionWeights>) {
        // Default implementation just calls predict and returns None for attention
        (
            self.predict(observation, current_time_step, legal_actions),
            None,
        )
    }
}

pub struct AttentionWeights {
    pub drivers_edge_index: Vec<(usize, usize)>, // [(source, target), ...]
    pub drivers_weights: Vec<Vec<f32>>,          // [num_edges, num_heads]

    pub laps_edge_index: Vec<(usize, usize)>,
    pub laps_weights: Vec<Vec<f32>>,
}
