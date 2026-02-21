use rand::random_range;

use crate::{
    algorithms::muzero::{
        game_environment::{Action, Game},
        muzero_config::MuzeroConfig,
    },
    utils::BoundedStack,
};

pub struct ReplayBuffer {
    batch_size: usize,
    // Buffer stores distinct EPISODES (Vec<Transition>), not a flat list of transitions
    buffer: BoundedStack<Game>,
}

impl ReplayBuffer {
    pub fn new(max_episodes: usize, batch_size: usize) -> Self {
        let buffer = BoundedStack::new(max_episodes);
        Self { batch_size, buffer }
    }

    pub fn save_game(&mut self, game: Game) {
        self.buffer.push(game);
    }

    fn sample_game(&self) -> Game {
        let num_epsiodes = self.buffer.len();
        let random_episode_index = random_range(0..num_epsiodes);
        let episode = self.buffer.data[random_episode_index].clone();

        episode
    }

    fn sample_position(&self, game: &Game) -> usize {
        let random_starting_point = random_range(0..game.history.len());
        random_starting_point
    }

    pub fn sample_batch(
        &self,
        num_unroll_steps: u32,
        td_steps: u32,
        config: MuzeroConfig,
    ) -> Vec<(Vec<f32>, Vec<Action>, Vec<(f32, Option<f32>, Vec<f32>)>)> {
        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            let game = self.sample_game();
            let start_index = self.sample_position(&game);

            let observation = game.make_image(start_index);
            let targets = game.make_targets(
                start_index,
                num_unroll_steps,
                td_steps,
                game.to_play(),
                config,
            );

            // Pad actions to ALWAYS have exactly num_unroll_steps actions
            let mut actions = Vec::with_capacity(num_unroll_steps as usize);
            for k in 0..num_unroll_steps as usize {
                let action_index = start_index + k;
                if action_index < game.history.len() {
                    actions.push(game.history[action_index]);
                } else {
                    // Dummy action for absorbing state (will pair with uniform policy target)
                    actions.push(Action::new(0));
                }
            }

            batch.push((observation, actions, targets));
        }

        batch
    }
}
