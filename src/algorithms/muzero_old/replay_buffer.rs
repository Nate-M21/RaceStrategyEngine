use rand::{Rng, rng};

use crate::{algorithms::muzero_old::muzero::MuzeroTransition, utils::BoundedStack};

pub struct MuzeroReplayBuffer {
    batch_size: usize,
    // Buffer stores distinct EPISODES (Vec<Transition>), not a flat list of transitions
    buffer: BoundedStack<Vec<MuzeroTransition>>,
}

impl MuzeroReplayBuffer {
    pub fn new(max_episodes: usize, batch_size: usize) -> Self {
        let buffer = BoundedStack::new(max_episodes);
        Self { batch_size, buffer }
    }

    pub fn add_episode(&mut self, episode: Vec<MuzeroTransition>) {
        self.buffer.push(episode);
    }

    pub fn sample_batch(&self, num_unroll_steps: usize) -> Vec<Vec<MuzeroTransition>> {
        let rng = &mut rand::rng();
        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            let episode = self.sample_episode();
            let episode_length = episode.len();
            // Taking a random start position in that episode
            let start_index = rng.random_range(0..episode_length);

            let mut sequence = Vec::with_capacity(num_unroll_steps + 1);

            for k in 0..=num_unroll_steps {
                let current_index = start_index + k;

                if current_index < episode_length {
                    sequence.push(episode[current_index].clone());
                } else {
                    if let Some(last_transition) = episode.last() {
                        let mut absorbing_state = last_transition.clone();

                        absorbing_state.reward = 0.0;
                        absorbing_state.value_target = 0.0;
                        let action_space_size = absorbing_state.action_probabilities.len();
                        let uniform_prob = 1.0 / action_space_size as f32;
                        absorbing_state.action_probabilities =
                            vec![uniform_prob; action_space_size];
                        sequence.push(absorbing_state);
                    }
                }
            }
            batch.push(sequence);
        }

        batch
    }

    /// Sample a epsiode from buffer
    fn sample_episode(&self) -> &Vec<MuzeroTransition> {
        let num_epsiodes = self.buffer.len();
        let episode_index = rng().random_range(0..num_epsiodes);

        let random_episode = &self.buffer.data[episode_index];

        random_episode
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}
