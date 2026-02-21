use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, RwLock},
};

use crate::{
    algorithms::{
        alpha_zero::{
            alpha_zero_config::AlphaZeroConfig, alpha_zero_mcts::AlphaZeroMCTS,
            replay_buffer::ReplayBuffer,
        },
        strategy::RaceStrategyEnvironment,
    },
    environment::AgentInfo,
    traits::{
        actor_critic::ActorCritic,
        gym::{GymEnvironment, MCTSGymEnvironment},
    },
    utils::argmax,
};

use indicatif::{
    MultiProgress, ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle,
};
use rand::{rng, rngs::ThreadRng};
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Clone, Debug)]
pub struct RobTransition {
    pub observation: Vec<f32>,
    pub action_probabilities: Vec<f32>,
    pub total_reward: f32,
    pub current_transition_strategy_encoding: Vec<f32>,
    pub episode_end_strategy_encoding: Vec<f32>,
    pub final_position: u32,
    pub transition_number: u32,
}

#[derive(Debug)]
/// The Paths of R.O.B, a struct that allows the testing of different race optimasation bots
pub struct PathsOfRob<Model: ActorCritic> {
    config: AlphaZeroConfig,
    pub model: Arc<RwLock<Model>>,
    environment: RaceStrategyEnvironment,
    replay_buffer: ReplayBuffer<RobTransition>,
}

impl<Model: ActorCritic<ObservationType = Vec<f32>, TransitionType = RobTransition>>
    PathsOfRob<Model>
{
    pub fn new(
        config: AlphaZeroConfig,
        model: Model,
        environment: RaceStrategyEnvironment,
    ) -> Self {
        let model = Arc::new(RwLock::new(model));
        let replay_buffer = ReplayBuffer::new(config.buffer_size, config.batch_size);

        // println!("AlphaZero\n{}\n{:?}\n{}\n", "-".repeat(100), alpha_zero, "-".repeat(100) );
        Self {
            config,
            model,
            environment,
            replay_buffer,
        }
    }

    pub fn learn(&mut self) {
        let multi_progress = MultiProgress::new();
        let shared_environment = Arc::new(RwLock::new(self.environment.clone()));
        let pb = ProgressBar::new(self.config.num_iterations.into());
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:100.cyan/blue}] {human_pos}/{human_len} ({per_sec}, ETA: {eta})")
            .unwrap()
            .progress_chars("█▓░"));

        let pb = multi_progress.add(pb);

        for num in (0..self.config.num_iterations).progress_with(pb) {
            let pb2 = ProgressBar::new(self.config.episode_iterations.into());
            pb2.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:100.yellow/blue}] {human_pos}/{human_len} ({per_sec}, ETA: {eta})")
            .unwrap()
            .progress_chars("█▓░"));
            let pb2 = multi_progress.add(pb2);

            // todo optimization, i could create threads with spawn and keep on and just send the latest model after
            // updating and with an atomic flag tell threads to stop while i update the model, then threads
            // have access to Arc<Buffer> and all send there
            let episodes: Vec<_> = (0..self.config.episode_iterations)
                .into_par_iter()
                .progress_with(pb2)
                .map(|_| {
                    PathsOfRob::play_episode(
                        Arc::clone(&shared_environment),
                        AlphaZeroMCTS::new(self.config, Arc::clone(&self.model)),
                        self.config,
                    )
                }) // Sequential
                .collect();

            for episode in episodes {
                self.replay_buffer.add_episode(episode);
            }
            self.model
                .write()
                .unwrap()
                .train_model(&mut self.replay_buffer, &self.config);

            let file_name = format!("rob_model_{}", num);

            self.model.read().unwrap().save_model(Path::new(&file_name));
            let path = Path::new("saved_norm_stats");
            shared_environment.read().unwrap().save_norm_stats(path);
        }
    }

    fn play_episode(
        shared_environment: Arc<RwLock<RaceStrategyEnvironment>>,
        mcts: AlphaZeroMCTS<Model>,
        config: AlphaZeroConfig,
    ) -> Vec<RobTransition> {
        let mut transitions = Vec::new();
        // Getting current state of the enviornemnt making local copy to work on then sending the changes back
        // This mainly so the dueling archiecture works, if was not doing dueling and just monte carlo
        // I could just send owned environment each time and not worry about sending the end env back

        let (mut local_environment, mut obs, mut info) = {
            let mut locked_env = shared_environment.write().unwrap();
            let (obs, info) = locked_env.reset();
            let env = locked_env.clone();

            (env, obs, info)
        };

        let mut total_reward = 0.0;
        let rng = &mut rng();

        loop {
            local_environment.clear();
            let transition_number = local_environment.get_current_step() as u32;
            let lap = local_environment.get_current_significant_step();

            let move_number = lap;

            // signifant step equates to the lap agent is on, because in time discrete you can be on different time
            // step but still be on the same lap, so this is my way to tell, in generic standard way.
            // let move_number = local_environment.get_current_significant_step();
            let current_env_state = local_environment.clone();
            let current_observation = obs.clone();
            let action_probabilities = mcts.search(current_env_state, current_observation, true);

            let action = get_action(action_probabilities.clone(), move_number, rng, config);

            let current_encoded_strategy = get_strategy_encoding(&local_environment, &mut info);

            transitions.push((
                obs,
                action_probabilities,
                transition_number,
                current_encoded_strategy,
            ));

            let (new_obs, reward, terminated, truncated, new_info) = local_environment.step(action);

            obs = new_obs;
            info = new_info;

            total_reward += reward;

            let done = terminated || truncated;

            if done {
                // Send back the results of local environment, so it could the basis of another thread

                let agent_info = info.remove("Agent").unwrap();
                let final_position = agent_info.position as u32;
                let final_strategy: Vec<(String, u8)> = agent_info.strategy;

                let encoded_strategy = local_environment.encode_strategy(&final_strategy);

                *shared_environment.write().unwrap() = local_environment;

                let mut episode_transitions = Vec::new();
                for (
                    observation,
                    action_probabilities,
                    transition_number,
                    current_strategy_encoding,
                ) in transitions
                {
                    let final_strategy_encoding = encoded_strategy.clone();
                    episode_transitions.push(RobTransition {
                        observation,
                        action_probabilities,
                        total_reward,
                        final_position,
                        transition_number,
                        episode_end_strategy_encoding: final_strategy_encoding,
                        current_transition_strategy_encoding: current_strategy_encoding,
                    });
                }

                return episode_transitions;
            }
        }
    }


}

fn get_strategy_encoding(
    environment: &RaceStrategyEnvironment,
    info: &mut HashMap<String, AgentInfo>,
) -> Vec<f32> {
    let agent_info = info.remove("Agent").unwrap();
    let lap_pitted_on = agent_info.laps_pitted_on.len();
    // this is to exclude the next compound the agent has select
    let strategy = &agent_info.strategy[0..=lap_pitted_on];
    let current_encoded_strategy = environment.encode_strategy(strategy);
    current_encoded_strategy
}

fn get_action(
    action_probabilities: Vec<f32>,
    move_number: usize,
    mut rng: &mut ThreadRng,
    config: AlphaZeroConfig,
) -> usize {
    // In AlphaZero when move number was greater than 30 they effectively took the argmax
    if move_number >= 30 {
        return argmax(&action_probabilities);
    } else {
        let action_probabilities = apply_temperature(action_probabilities, config);

        let dist = WeightedIndex::new(&action_probabilities).unwrap();

        let action = dist.sample(&mut rng);

        action
    }
}

fn apply_temperature(mut action_probabilities: Vec<f32>, config: AlphaZeroConfig) -> Vec<f32> {
    let temperature = 1.0 / config.temperature;
    action_probabilities
        .iter_mut()
        .for_each(|n| *n = n.powf(temperature));

    let action_sum = action_probabilities.iter().sum::<f32>();

    action_probabilities
        .iter_mut()
        .for_each(|n| *n /= action_sum);

    action_probabilities
}
