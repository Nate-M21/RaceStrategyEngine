use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, RwLock},
};

use burn::{prelude::Backend, tensor::backend::AutodiffBackend};
use indicatif::{
    MultiProgress, ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle,
};
use rand::{rng, rngs::ThreadRng};
use rand_distr::{Distribution, weighted::WeightedIndex};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    algorithms::muzero_old::{
        muzero_config::MuzeroConfig, muzero_mcts::MuzeroMcts, muzero_model::MuzeroModel,
        replay_buffer::MuzeroReplayBuffer,
    },
    environment::AgentInfo,
    traits::gym::MCTSGymEnvironment,
    utils::argmax,
};

#[derive(Debug, Clone)]
pub struct MuzeroTransition {
    pub observation: Vec<f32>,
    pub action_probabilities: Vec<f32>,
    pub reward: f32,
    pub root_value: f32,
    pub action: usize,
    pub value_target: f32,
}

pub struct Muzero<Environment: MCTSGymEnvironment, B: Backend + AutodiffBackend> {
    config: MuzeroConfig,
    model: Arc<RwLock<MuzeroModel<B>>>,
    environment: Environment,
    replay_buffer: MuzeroReplayBuffer,
}

impl<Environment: MCTSGymEnvironment, B: Backend + AutodiffBackend> Muzero<Environment, B> {
    pub fn new(config: MuzeroConfig, model: MuzeroModel<B>, environment: Environment) -> Self {
        let model = Arc::new(RwLock::new(model));
        let replay_buffer = MuzeroReplayBuffer::new(config.buffer_size, config.batch_size);

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
        let stats_path = Path::new("/Users/nate/Desktop/rust_sim_core/python/saved_norm_stats");
        self.environment.load_norm_stats(stats_path);
        let multi_progress = MultiProgress::new();
        // let action_space = self.model.read().unwrap().get_action_space();
        let shared_environment = Arc::new(RwLock::new(self.environment.clone()));
        Muzero::test_model(
            Arc::clone(&shared_environment),
            MuzeroMcts::new(self.config, Arc::clone(&self.model)),
        );
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
                    Muzero::play_episode(
                        Arc::clone(&shared_environment),
                        MuzeroMcts::new(self.config, Arc::clone(&self.model)),
                        self.config,
                    )
                })
                .collect();

            for episode in episodes {
                self.replay_buffer.add_episode(episode);
            }
            self.model
                .write()
                .unwrap()
                .train_model(&mut self.replay_buffer, &self.config);

            Muzero::test_model(
                Arc::clone(&shared_environment),
                MuzeroMcts::new(self.config, Arc::clone(&self.model)),
            );
        }
    }

    fn play_episode(
        shared_environment: Arc<RwLock<Environment>>,
        mcts: MuzeroMcts<B>,
        config: MuzeroConfig,
    ) -> Vec<MuzeroTransition>
    where
        Environment: MCTSGymEnvironment<
                Observation = Vec<f32>,
                Reward = f32,
                Terminated = bool,
                Truncated = bool,
                Info = HashMap<String, AgentInfo>,
            >,
    {
        let mut transitions = Vec::new();

        let (mut local_environment, mut obs, _info) = {
            let mut locked_env = shared_environment.write().unwrap();
            let (obs, info) = locked_env.reset();
            let env = locked_env.clone();

            (env, obs, info)
        };

        let rng = &mut rng();

        loop {
            local_environment.clear();
            let move_number = local_environment.get_current_significant_step();

            let legal_actions = local_environment.get_legal_actions();
            let observation = obs.clone();

            let (action_probabilities, root_value) =
                mcts.search(observation, legal_actions.clone());

            let action = get_action(action_probabilities.clone(), move_number, rng, config);

            let (new_obs, reward, terminated, truncated, _info) = local_environment.step(action);
            transitions.push((obs, action_probabilities, reward, root_value, action));

            obs = new_obs;

            let done = terminated || truncated;

            if done {
                // Reset and Send back the results of local environment, so it could the basis of another thread
                *shared_environment.write().unwrap() = local_environment;

                let episode_transitions_final = make_targets(config, transitions);

                return episode_transitions_final;
            }
        }
    }

    fn test_model(shared_environment: Arc<RwLock<Environment>>, mcts: MuzeroMcts<B>)
    where
        Environment: MCTSGymEnvironment<
                Observation = Vec<f32>,
                Reward = f32,
                Terminated = bool,
                Truncated = bool,
                Info = HashMap<String, AgentInfo>,
            >,
    {
        let (mut local_environment, mut obs, _info) = {
            let mut locked_env = shared_environment.write().unwrap();
            let (obs, info) = locked_env.reset();
            let env = locked_env.clone();

            (env, obs, info)
        };

        println!("{}", "-".repeat(50));
        println!("Starting Grid");
        local_environment.show_info();
        let compounds = ["hard", "medium", "soft", "None"];
        let config = mcts.config;
        loop {
            local_environment.clear();
            let lap = local_environment.get_current_significant_step();
            let current_time_step = local_environment.get_current_step();

            let legal_actions = local_environment.get_legal_actions();
            let observation = obs.clone();

            let dirichlet_noise = false;
            let predictions = mcts
                .model
                .read()
                .unwrap()
                .get_initial_action_probs_and_value(
                    &observation,
                    &legal_actions,
                    dirichlet_noise,
                    config,
                );
            let mut priors = predictions.action_probabilities;
            let value = predictions.value;
            priors.iter_mut().for_each(|num| *num = *num * 100.0);

            let (mut action_probabilities, _root_value) =
                mcts.search(observation, legal_actions.clone());

            action_probabilities
                .iter_mut()
                .for_each(|prior| *prior = *prior * 100.0);

            let action = argmax(&action_probabilities);

            println!(
                "Lap: {}, Transition: {},\nPriors: {:?}\n\nAction Probabilities (MCTS): {:?}\n\n-----------\n Value: {:?} {}\n",
                lap,
                current_time_step,
                priors,
                action_probabilities,
                value,
                "#".repeat(20)
            );

            if action_probabilities.len() > 5 {
                let pitting_lap = ((action / 3) + 1) as u8;
                let compound = if action != 162 {
                    compounds[action % 3]
                } else {
                    "None"
                };

                println!(
                    "\nThe legal_actions: {:?}\n\nThe action probabilties (MCTS): {:?}\n\nThe action taken: {} | Translated to lap: {} The compound is: {compound}",
                    legal_actions, action_probabilities, action, pitting_lap
                );
            } else {
                let compound = compounds[action];
                println!(
                    "\nThe legal_actions: {:?}\n\nThe action probabilties (MCTS): {:?}\n\nThe action taken: {} | The compound is: {compound}",
                    legal_actions, action_probabilities, action,
                );
            }

            let (new_obs, _reward, terminated, truncated, _info) = local_environment.step(action);

            obs = new_obs;

            let done = terminated || truncated;

            if done {
                println!("{}", "=".repeat(5));
                println!("End Result");
                local_environment.show_info();
                println!("{}", "=".repeat(5));
                println!("{:?}", _info);
                println!("{}", "-".repeat(50));
                // Reset and Send back the results of local environment, so it could the basis of another thread
                break;
            }
        }
    }
}

fn make_targets(
    config: MuzeroConfig,
    transitions: Vec<(Vec<f32>, Vec<f32>, f32, f32, usize)>,
) -> Vec<MuzeroTransition> {
    let discount = config.discount;
    // The gamma (γ) value
    let td_steps = config.td_steps;
    // The n value (e.g., 5 or 10)

    let episode_length = transitions.len();
    let mut episode_transitions_final = Vec::new();

    // 1. Convert tuples to transitions and calculate targets.
    for t in 0..episode_length {
        let (observation, action_probabilities, reward, root_value, action) =
            transitions[t].clone();

        // --- N-Step Target Calculation (z_t) ---
        let mut value_target = 0.0;

        // a) Sum N rewards: ∑(γ^(k-1) * r_{t+k})
        for k in 0..td_steps {
            let lookahead_index = t + k as usize;

            if lookahead_index < episode_length {
                let r_k = transitions[lookahead_index].2; // The reward at time t+k
                value_target += discount.powf(k as f32) * r_k;
            } else {
                // Game ended within the N steps, stop summing rewards
                break;
            }
        }

        // b) Bootstrap with V_{t+n}: + γ^n * V_{t+n}
        let bootstrap_index = t + td_steps as usize;

        if bootstrap_index < episode_length {
            // If we have a full N steps, bootstrap with the MCTS value V_{t+n}
            let v_tn = transitions[bootstrap_index].3; // root_value at time t+n
            value_target += discount.powf(td_steps as f32) * v_tn;
        } else {
            // If the game ended before N steps, the final term is implicitly 0.
            // The value_target is purely Monte Carlo return from that point on.
        }
        // ----------------------------------------

        episode_transitions_final.push(MuzeroTransition {
            observation,
            action_probabilities,
            reward,
            root_value,
            action,
            value_target, // Store the computed z_t
        });
    }
    episode_transitions_final
}

fn get_action(
    action_probabilities: Vec<f32>,
    _move_number: usize,
    mut rng: &mut ThreadRng,
    config: MuzeroConfig,
) -> usize {
    let action_probabilities = apply_temperature(action_probabilities, config);

    let sum: f32 = action_probabilities.iter().sum();
    if sum == 0.0 || sum.is_nan() {
        println!("\n{}", "-".repeat(30));
        println!("WARNING!!!!!!!!!!!!!!");
        println!("ERROR: All action probabilities are zero or NaN!");
        println!("Raw probs: {:?}", action_probabilities);
        println!("\n{}\n", "-".repeat(30));
        // panic!("Invalid action distribution");
    }

    let dist = WeightedIndex::new(&action_probabilities).unwrap();

    let action = dist.sample(&mut rng);

    action
}

fn apply_temperature(mut action_probabilities: Vec<f32>, config: MuzeroConfig) -> Vec<f32> {
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
