use std::{
    iter::zip,
    path::Path,
    sync::{Arc, RwLock},
};

use crate::{
    algorithms::{
        helpers::scalar_to_support_batch,
        muzero::{
            game_environment::{Action, Game},
            muzero_config::MuzeroConfig,
            muzero_mcts::{MinMaxStats, Node, backpropagate, expand_node, run_mcts},
            muzero_model::{MuzeroModel, encode_action},
            replay_buffer::ReplayBuffer,
        },
        strategy::RaceStrategyEnvironment,
    },
    traits::gym::GymEnvironment,
    utils::argmax,
};
use burn::{
    Tensor,
    optim::{GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, loss::cross_entropy_with_logits},
};
use indicatif::{
    MultiProgress, ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle,
};
use rand::{rng, rngs::ThreadRng};
use rand_distr::{Distribution, weighted::WeightedIndex};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct Muzero<B: Backend + AutodiffBackend> {
    config: MuzeroConfig,
    model: Arc<RwLock<MuzeroModel<B>>>,
    environment: RaceStrategyEnvironment,
    replay_buffer: ReplayBuffer,
}

impl<B: Backend + AutodiffBackend> Muzero<B> {
    pub fn new(
        config: MuzeroConfig,
        model: MuzeroModel<B>,
        environment: RaceStrategyEnvironment,
    ) -> Self {
        let model = Arc::new(RwLock::new(model));
        let replay_buffer = ReplayBuffer::new(config.buffer_size, config.batch_size);

        Self {
            config,
            model,
            environment,
            replay_buffer,
        }
    }

    pub fn learn(&mut self) {
        let stats_path = Path::new("/Users/nate/Desktop/rust_sim_core/python/saved_norm_stats");
        self.environment.load_norm_stats(stats_path);
        let multi_progress = MultiProgress::new();
        let shared_environment = Arc::new(RwLock::new(self.environment.clone()));

        Muzero::test_model(
            self.config,
            Arc::clone(&shared_environment),
            Arc::clone(&self.model),
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

            let episodes: Vec<_> = (0..self.config.episode_iterations)
                .into_par_iter()
                .progress_with(pb2)
                .map(|_| {
                    Muzero::play_episode(
                        self.config,
                        Arc::clone(&shared_environment),
                        Arc::clone(&self.model),
                    )
                })
                .collect();

            for episode in episodes {
                self.replay_buffer.save_game(episode);
            }
            train_network(
                self.config,
                &mut self.model.write().unwrap(),
                &mut self.replay_buffer,
            );
            Muzero::test_model(
                self.config,
                Arc::clone(&shared_environment),
                Arc::clone(&self.model),
            );
        }
    }

    fn play_episode(
        config: MuzeroConfig,
        shared_environment: Arc<RwLock<RaceStrategyEnvironment>>,
        shared_network: Arc<RwLock<MuzeroModel<B>>>,
    ) -> Game {
        let local_environment = shared_environment.read().unwrap().clone();
        let network = shared_network.read().unwrap();
        let mut game = Game::new_game(local_environment, config);
        let (mut observation, _info) = game.reset();
        while !game.terminal() && game.history.len() < config.max_moves {
            game.clear();
            let mini_max_stats =
                &mut MinMaxStats::new(config.known_maximum_reward, config.known_minimum_reward);

            let mut root = Node::<B>::new(0.0);

            let legal_actions = game.legal_actions();
            // the network output of get intial has exploration noise

            let dirichlet_noise = true;
            let network_output = network.get_initial_action_probs_and_value(
                &observation,
                &legal_actions,
                dirichlet_noise,
                config,
            );
            let value = network_output.value;
            expand_node(
                &mut root,
                game.to_play(),
                game.action_history().action_space(),
                network_output,
            );

            backpropagate(
                vec![&mut root],
                value,
                game.to_play(),
                config.discount,
                mini_max_stats,
            );

            run_mcts(
                config,
                &mut root,
                game.action_history(),
                &network,
                mini_max_stats,
            );

            let action = select_action(config, game.history.len(), &root, &network);
            (observation, _) = game.apply(Action::new(action));

            game.store_search_statistics(&root);
        }
        std::mem::swap(
            &mut game.environment,
            &mut *shared_environment.write().unwrap(),
        );
        game
    }

    fn test_model(
        config: MuzeroConfig,
        shared_environment: Arc<RwLock<RaceStrategyEnvironment>>,
        shared_network: Arc<RwLock<MuzeroModel<B>>>,
    ) -> Game {
        let local_environment = shared_environment.read().unwrap().clone();
        let network = shared_network.read().unwrap();
        let mut game = Game::new_game(local_environment, config);
        let (mut observation, _info) = game.reset();
        println!("Starting Grid");
        let mut info = _info;
        game.show_info();
        while !game.terminal() && game.history.len() < config.max_moves {
            game.clear();
            let mini_max_stats =
                &mut MinMaxStats::new(config.known_maximum_reward, config.known_minimum_reward);

            let mut root = Node::<B>::new(0.0);

            let legal_actions = game.legal_actions();
            // the network output of get intial has exploration noise

            let dirichlet_noise = false;
            let network_output = network.get_initial_action_probs_and_value(
                &observation,
                &legal_actions,
                dirichlet_noise,
                config,
            );
            let value = network_output.value;
            expand_node(
                &mut root,
                game.to_play(),
                game.action_history().action_space(),
                network_output,
            );

            backpropagate(
                vec![&mut root],
                value,
                game.to_play(),
                config.discount,
                mini_max_stats,
            );

            run_mcts(
                config,
                &mut root,
                game.action_history(),
                &network,
                mini_max_stats,
            );

            let action = argmax_action(config, &root);
            (observation, info) = game.apply(Action::new(action));
        }
        println!("{}", "=".repeat(5));
        println!("End Result");
        game.show_info();
        println!("{}", "=".repeat(5));
        println!("{:?}", info);
        println!("{}", "-".repeat(50));
        std::mem::swap(
            &mut game.environment,
            &mut *shared_environment.write().unwrap(),
        );
        game
    }
}

fn select_action<B: Backend + AutodiffBackend>(
    config: MuzeroConfig,
    num_moves: usize,
    node: &Node<B>,
    _network: &MuzeroModel<B>,
) -> usize {
    let action_probs = get_action_probs(config, node);
    let rng = &mut rng();
    let action = get_single_action(action_probs, num_moves, rng, config);
    action
}

fn get_action_probs<B: Backend>(config: MuzeroConfig, node: &Node<B>) -> Vec<f32> {
    let mut action_probs = vec![0.0; config.action_space];

    for (action, child) in node.children_dict.iter() {
        action_probs[*action] = child.visit_count as f32;
    }

    let action_sum = action_probs.iter().sum::<f32>();
    for action_prob in action_probs.iter_mut() {
        *action_prob /= action_sum;
    }
    action_probs
}

fn get_single_action(
    action_probabilities: Vec<f32>,
    _move_number: usize,
    mut rng: &mut ThreadRng,
    config: MuzeroConfig,
) -> usize {
    let action_probabilities = apply_temperature(action_probabilities, config);

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

fn train_network<B: Backend + AutodiffBackend>(
    config: MuzeroConfig,
    network: &mut MuzeroModel<B>,
    replay_buffer: &mut ReplayBuffer,
) {
    for it in 0..config.training_steps {
        println!("{it}/{}", config.training_steps);
        let batch: Vec<(Vec<f32>, Vec<Action>, Vec<(f32, Option<f32>, Vec<f32>)>)> =
            replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config);

        // update_weights(network, batch, config);
        update_weights_batch(network, batch, config);
    }
    println!("Done training")
}

fn _update_weights<B: Backend + AutodiffBackend>(
    network: &mut MuzeroModel<B>,
    batch: Vec<(Vec<f32>, Vec<Action>, Vec<(f32, Option<f32>, Vec<f32>)>)>,
    config: MuzeroConfig,
) {
    let batch_length = batch.len() as f32;

    let device = &network.networks.get_device();
    let mut loss: Tensor<B, 1> = Tensor::from_data([0], device);

    let mut total_policy_loss = Tensor::from_data([0.0], device);
    let mut total_value_loss = Tensor::from_data([0.0], device);
    let mut total_reward_loss = Tensor::from_data([0.0], device);
    for (observation, actions, targets) in batch {
        let network_output: crate::algorithms::muzero::muzero_model::MuzeroNetworkOutput<B> =
            network.initial_inference(&observation);
        let mut hidden_state = network_output.latent_representation.clone();
        let mut predictions = vec![(1.0, network_output)];

        let actions_len = actions.len() as f32;
        for action in actions {
            let network_output = network.recurrent_inference(hidden_state, action.index);
            let hidden_state_k = network_output.latent_representation.clone();
            predictions.push((1.0 / actions_len, network_output));

            hidden_state = scale_gradient(hidden_state_k, 0.5);
        }

        let zipped = zip(predictions, targets);

        for (k, (prediction, target)) in zipped.enumerate() {
            let (gradient_scale, network_output) = prediction;
            let (target_value, target_reward, target_policy) = target;
            let mut k_loss = Tensor::from_data([0], device);
            let target_policy = Tensor::from_data(target_policy.as_slice(), device);
            let policy_loss =
                cross_entropy_with_logits(network_output.policy_logits, target_policy);

            total_policy_loss = total_policy_loss + policy_loss.clone();

            k_loss = k_loss + policy_loss;

            let target_value = scalar_to_support_batch(&[target_value], device).squeeze_dim(0);
            let value_loss = cross_entropy_with_logits(network_output.value_logits, target_value);
            total_value_loss = total_value_loss + value_loss.clone();
            k_loss = k_loss + value_loss;
            if k > 0 {
                let reward_value = scalar_to_support_batch(
                    // to take into account absorbing state, when there is No value the default will be 0.0
                    &[target_reward.unwrap_or(0.0)],
                    device,
                )
                .squeeze_dim(0);
                let reward_loss =
                    cross_entropy_with_logits(network_output.reward_logits, reward_value);

                total_reward_loss = total_reward_loss + reward_loss.clone();

                k_loss = k_loss + reward_loss;
            }
            let k_loss = k_loss.unsqueeze_dim(0);

            loss = loss + scale_gradient(k_loss, gradient_scale).squeeze_dim(0);
        }
    }

    let policy_loss_print = total_policy_loss.to_data().to_vec::<f32>().unwrap()[0];
    let value_loss_print = total_value_loss.to_data().to_vec::<f32>().unwrap()[0];
    let reward_loss_print = total_reward_loss.to_data().to_vec::<f32>().unwrap()[0];
    let total_loss_print = loss.to_data().to_vec::<f32>().unwrap()[0];

    loss = loss / batch_length;

    let total_loss_scaled_print = loss.to_data().to_vec::<f32>().unwrap()[0];

    println!(
        "policy loss = {:.6}, value loss = {:.6}, reward loss = {:.6}, total loss = {:.6} | Total loss - backward (scaled by batch size) = {:.6}",
        policy_loss_print,
        value_loss_print,
        reward_loss_print,
        total_loss_print,
        total_loss_scaled_print
    );

    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &network.networks);

    let lr = config.learning_rate_init as f64;

    network.networks = network
        .optimizer
        .step(lr, network.networks.clone(), grads_params)
}

fn update_weights_batch<B: Backend + AutodiffBackend>(
    network: &mut MuzeroModel<B>,
    batch: Vec<(Vec<f32>, Vec<Action>, Vec<(f32, Option<f32>, Vec<f32>)>)>,
    config: MuzeroConfig,
) {
    let capacity = batch.len();
    let action_space = config.action_space;

    let device = &network.networks.get_device();
    let mut total_loss: Tensor<B, 1> = Tensor::from_data([0], device);

    let mut total_reward_loss = Tensor::from_data([0.0], device);

    let mut initial_observations = Vec::with_capacity(capacity);
    let mut initial_value_targets = Vec::with_capacity(capacity);
    let mut initial_policy_targets = Vec::with_capacity(capacity);

    let mut actions = Vec::with_capacity(capacity);
    let mut sequence_value_targets = Vec::with_capacity(capacity);

    for (observation, action, targets) in batch {
        let observation: Tensor<B, 1> = Tensor::from_data(observation.as_slice(), device);
        initial_observations.push(observation);

        let (target_value, _target_reward, target_policy) = &targets[0];

        initial_value_targets.push(*target_value);

        let policy_target: Tensor<B, 1> = Tensor::from_data(target_policy.as_slice(), device);

        initial_policy_targets.push(policy_target);

        actions.push(action);
        // im leaving out the first target because thats included in initial
        sequence_value_targets.push(targets[1..].to_vec())
    }

    let observations = Tensor::stack(initial_observations, 0);
    let initial_value_targets = scalar_to_support_batch(&initial_value_targets, device);
    let initial_policy_targets = Tensor::stack(initial_policy_targets, 0);

    let initial_hidden_states = network
        .networks
        .representation_network
        .forward(observations);

    let (initial_policy_prediction, initial_value_prediction) = network
        .networks
        .prediction_network
        .forward(initial_hidden_states.clone());
    let mut total_policy_loss =
        cross_entropy_with_logits(initial_policy_prediction, initial_policy_targets);
    let mut total_value_loss =
        cross_entropy_with_logits(initial_value_prediction, initial_value_targets);

    let mut hidden_state = initial_hidden_states;
    let mut loss = scale_gradient(
        (total_policy_loss.clone() + total_value_loss.clone()).unsqueeze(),
        1.0,
    )
    .squeeze_dim(0);
    total_loss = total_loss + loss.clone();

    // Process each unroll step k across ALL sequences in batch
    for k in 0..config.num_unroll_steps as usize {
        let mut actions_k = Vec::with_capacity(capacity);
        let mut policy_targets_k: Vec<Tensor<B, 1>> = Vec::with_capacity(capacity);
        let mut value_targets_k = Vec::with_capacity(capacity);
        let mut reward_targets_k = Vec::with_capacity(capacity);

        // For each sequence in batch, get data at step k
        let actions_and_targets = zip(&actions, &sequence_value_targets);
        for (action_seq, target_seq) in actions_and_targets {
            let action_tensor: Tensor<B, 1> =
                encode_action(action_seq[k].index, action_space, device).squeeze_dim(0);
            let (target_value, target_reward, target_policy) = &target_seq[k];

            actions_k.push(action_tensor);
            policy_targets_k.push(Tensor::from_data(target_policy.as_slice(), device));
            value_targets_k.push(*target_value);
            // This line below didnt work and kept getting the None value it should not have
            // reward_targets_k.push(target_reward.expect("Value of reward should not be None, as I excluded first index"));

            reward_targets_k.push(target_reward.unwrap_or(0.0));
        }

        let action_batch = Tensor::stack(actions_k, 0); // [Batch, Action_Dim]
        let target_policy_batch = Tensor::stack(policy_targets_k, 0);
        let target_value_batch = scalar_to_support_batch(&value_targets_k, device);
        let target_reward_batch = scalar_to_support_batch(&reward_targets_k, device);

        let (reward_prediction, mut next_hidden_state) = network.networks.dynamics_network.forward(
            hidden_state, // The hidden state from the previous iteration
            action_batch,
        );

        let (policy_prediction, value_prediction) = network
            .networks
            .prediction_network
            .forward(next_hidden_state.clone());

        next_hidden_state = scale_gradient(next_hidden_state, 0.5);
        hidden_state = next_hidden_state;

        let k_policy_loss = cross_entropy_with_logits(policy_prediction, target_policy_batch);
        let k_value_loss = cross_entropy_with_logits(value_prediction, target_value_batch);
        let k_reward_loss = cross_entropy_with_logits(reward_prediction, target_reward_batch);

        total_policy_loss = total_policy_loss + k_policy_loss.clone();
        total_value_loss = total_value_loss + k_value_loss.clone();
        total_reward_loss = total_reward_loss + k_reward_loss.clone();
        let total_k_loss = k_policy_loss + k_reward_loss + k_value_loss;
        total_loss = total_loss + total_k_loss.clone();

        loss = loss + scale_gradient(total_k_loss.unsqueeze(), 0.5).squeeze_dim(0);
    }

    let policy_loss_print = total_policy_loss.to_data().to_vec::<f32>().unwrap()[0];
    let value_loss_print = total_value_loss.to_data().to_vec::<f32>().unwrap()[0];
    let reward_loss_print = total_reward_loss.to_data().to_vec::<f32>().unwrap()[0];
    let total_loss_print = loss.to_data().to_vec::<f32>().unwrap()[0];

    let total_loss_scaled_print = loss.to_data().to_vec::<f32>().unwrap()[0];

    println!(
        "policy loss = {:.6}, value loss = {:.6}, reward loss = {:.6}, total loss = {:.6} | Total loss - backward (scaled by batch size) = {:.6}",
        policy_loss_print,
        value_loss_print,
        reward_loss_print,
        total_loss_print,
        total_loss_scaled_print
    );

    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &network.networks);

    let lr = config.learning_rate_init as f64;

    network.networks = network
        .optimizer
        .step(lr, network.networks.clone(), grads_params)
}

fn scale_gradient<B: Backend + AutodiffBackend>(tensor: Tensor<B, 2>, scale: f32) -> Tensor<B, 2> {
    tensor.clone() * scale + tensor.detach() * (1.0 - scale)
}

fn argmax_action<B: Backend>(config: MuzeroConfig, node: &Node<B>) -> usize {
    let action_probs = get_action_probs(config, node);
    let action = argmax(&action_probs);
    action
}

pub fn test_model<B: AutodiffBackend + Backend>(
    config: MuzeroConfig,
    environment: RaceStrategyEnvironment,
    network: &MuzeroModel<B>,
) -> Game {
    let mut game = Game::new_game(environment, config);
    let (mut observation, _info) = game.reset();
    println!("Starting Grid");
    let mut info = _info;
    game.show_info();
    while !game.terminal() && game.history.len() < config.max_moves {
        game.clear();
        let mini_max_stats =
            &mut MinMaxStats::new(config.known_maximum_reward, config.known_minimum_reward);

        let mut root = Node::<B>::new(0.0);

        let legal_actions = game.legal_actions();
        // the network output of get intial has exploration noise

        let dirichlet_noise = false;
        let network_output = network.get_initial_action_probs_and_value(
            &observation,
            &legal_actions,
            dirichlet_noise,
            config,
        );
        let value = network_output.value;
        expand_node(
            &mut root,
            game.to_play(),
            game.action_history().action_space(),
            network_output,
        );

        backpropagate(
            vec![&mut root],
            value,
            game.to_play(),
            config.discount,
            mini_max_stats,
        );

        run_mcts(
            config,
            &mut root,
            game.action_history(),
            &network,
            mini_max_stats,
        );

        let action = argmax_action(config, &root);
        (observation, info) = game.apply(Action::new(action));
    }

    println!("{}", "=".repeat(5));
    println!("End Result");
    game.show_info();
    println!("{}", "=".repeat(5));
    println!("{:?}", info);
    println!("{}", "-".repeat(50));
    game
}
