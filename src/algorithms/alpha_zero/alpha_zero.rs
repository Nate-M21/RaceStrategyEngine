use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, RwLock},
};

use crate::{
    algorithms::alpha_zero::{
        alpha_zero_config::AlphaZeroConfig, alpha_zero_mcts::AlphaZeroMCTS,
        replay_buffer::ReplayBuffer,
    },
    environment::AgentInfo,
    traits::{actor_critic::ActorCritic, gym::MCTSGymEnvironment},
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
pub struct Transition {
    pub observation: Vec<f32>,
    pub action_probabilities: Vec<f32>,
    pub total_reward: f32,
}

#[derive(Debug)]
pub struct AlphaZero<Environment: MCTSGymEnvironment, Model: ActorCritic> {
    config: AlphaZeroConfig,
    model: Arc<RwLock<Model>>,
    environment: Environment,
    replay_buffer: ReplayBuffer<Transition>,
}

impl<
    Environment: MCTSGymEnvironment,
    Model: ActorCritic<ObservationType = Vec<f32>, TransitionType = Transition>,
> AlphaZero<Environment, Model>
{
    pub fn new(config: AlphaZeroConfig, model: Model, environment: Environment) -> Self {
        let model = Arc::new(RwLock::new(model));
        let replay_buffer = ReplayBuffer::new(config.buffer_size, config.batch_size);

        Self {
            config,
            model,
            environment,
            replay_buffer,
        }
    }

    pub fn learn(&mut self)
    where
        Environment: MCTSGymEnvironment<
                Observation = Vec<f32>,
                Reward = f32,
                Terminated = bool,
                Truncated = bool,
                Info = HashMap<String, AgentInfo>,
            >,
    {
        let multi_progress = MultiProgress::new();
        let shared_environment = Arc::new(RwLock::new(self.environment.clone()));
        let pb = ProgressBar::new(self.config.num_iterations.into());
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:100.cyan/blue}] {human_pos}/{human_len} ({per_sec}, ETA: {eta})")
            .unwrap()
            .progress_chars("█▓░"));

        let pb = multi_progress.add(pb);

        for _ in (0..self.config.num_iterations).progress_with(pb) {
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
                    AlphaZero::play_episode(
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

            
        }
        println!("{}\n{}", "@".repeat(100), "Done training");
        let path = Path::new("saved_norm_stats");
        shared_environment.read().unwrap().save_norm_stats(path);
    }

    fn play_episode(
        shared_environment: Arc<RwLock<Environment>>,
        mcts: AlphaZeroMCTS<Model>,
        config: AlphaZeroConfig,
    ) -> Vec<Transition>
    where
        Environment: MCTSGymEnvironment<
                Observation = Vec<f32>,
                Reward = f32,
                Terminated = bool,
                Truncated = bool,
            >,
    {
        let mut transitions = Vec::new();
        // Getting current state of the enviornemnt making local copy to work on then sending the changes back
        // This mainly so the dueling archiecture works, if was not doing dueling and just monte carlo
        // I could just send owned environment each time and not worry about sending the end env back
        let (mut local_environment, mut obs, _info) = {
            let mut locked_env = shared_environment.write().unwrap();
            let (obs, info) = locked_env.reset();
            let env = locked_env.clone();

            (env, obs, info)
        };
        let mut total_reward = 0.0;
        let rng = &mut rng();

        loop {
            local_environment.clear();

            // signifant step equates to the lap agent is on, because in time discrete you can be on different time
            // step but still be on the same lap, so this is my way to tell, in generic standard way.
            let move_number = local_environment.get_current_significant_step();
            let current_env_state = local_environment.clone();
            let current_observation = obs.clone();
            let action_probabilities = mcts.search(current_env_state, current_observation, true);

            let action = get_action(action_probabilities.clone(), move_number, rng, config);

            transitions.push((obs, action_probabilities));

            let (new_obs, reward, terminated, truncated, _info) = local_environment.step(action);
            obs = new_obs;

            total_reward += reward;

            let done = terminated || truncated;

            if done {
                // Reset and Send back the results of local environment, so it could the basis of another thread
                *shared_environment.write().unwrap() = local_environment;

                let mut episode_transitions: Vec<Transition> = Vec::new();
                for (observation, action_probabilities) in transitions {
                    episode_transitions.push(Transition {
                        observation,
                        action_probabilities,
                        total_reward,
                    });
                }

                return episode_transitions;
            }
        }
    }

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
