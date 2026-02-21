use std::{
    collections::HashMap,
    iter::zip,
};

use crate::{
    algorithms::{
        muzero::{
            muzero_config::MuzeroConfig,
            muzero_model::{MuzeroModel, MuzeroModelOutput, scalar_to_support_batch},
        },
        strategy::RaceStrategyEnvironment,
    },
    environment::AgentInfo,
    traits::gym::{GymEnvironment, MCTSGymEnvironment},
    utils::BoundedStack,
};
use burn::{
    Tensor,
    optim::{GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, loss::cross_entropy_with_logits},
};
use indicatif::{
    MultiProgress, ProgressBar, ProgressIterator, ProgressStyle,
};
use rand::{random_range, rng, rngs::ThreadRng};
use rand_distr::{Distribution, weighted::WeightedIndex};

#[derive(Debug, Clone, Copy)]
pub struct MinMaxStats {
    maximum: f32,
    minimum: f32,
}

impl MinMaxStats {
    pub fn new(maximum: Option<f32>, minimum: Option<f32>) -> Self {
        let maximum = match maximum {
            Some(max) => max,
            None => f32::NEG_INFINITY,
        };

        let minimum = match minimum {
            Some(min) => min,
            None => f32::INFINITY,
        };

        Self { maximum, minimum }
    }
    pub fn update(&mut self, value: f32) {
        self.maximum = f32::max(self.maximum, value);
        self.minimum = f32::min(self.minimum, value);
    }

    pub fn normalize(&self, value: f32) -> f32 {
        if self.maximum > self.minimum {
            return (value - self.minimum) / (self.maximum - self.minimum);
        }
        value
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

pub struct Node<B: Backend> {
    pub visit_count: u32,
    pub value_sum: f32,
    pub to_play: Player,
    pub children_dict: HashMap<usize, Node<B>>,
    pub reward: f32,
    pub hidden_state: Option<Tensor<B, 2>>,
    pub prior: f32,
}

impl<B: Backend> Node<B> {
    pub fn new(prior: f32) -> Self {
        Self {
            visit_count: 0,
            value_sum: 0.0,
            to_play: Player {},
            children_dict: HashMap::new(),
            reward: 0.0,
            hidden_state: None,
            prior,
        }
    }

    pub fn expanded(&self) -> bool {
        self.children_dict.len() > 0
    }

    pub fn value(&self) -> f32 {
        if self.visit_count == 0 {
            return 0.0;
        }
        self.value_sum / self.visit_count as f32
    }
}
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

#[derive(Clone)]
pub struct Game {
    pub environment: RaceStrategyEnvironment,
    history: Vec<Action>,
    rewards: Vec<f32>,
    observations: Vec<Vec<f32>>,
    child_visits: Vec<Vec<f32>>,
    root_values: Vec<f32>,
    action_space_size: usize,
    discount: f32,
    terminal: bool,
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
    fn terminal(&self) -> bool {
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

    fn save_game(&mut self, game: Game) {
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
        let mut games = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            games.push(self.sample_game())
        }
        let mut gam_pos = Vec::with_capacity(self.batch_size);

        for game in games {
            let pos = self.sample_position(&game);
            gam_pos.push((game, pos))
        }

        for (game, index) in gam_pos {
            let end_index = (index + num_unroll_steps as usize).min(game.history.len());

            let history = &game.history[index..end_index];
            let targets =
                game.make_targets(index, num_unroll_steps, td_steps, game.to_play(), config);
            let a = (game.make_image(index), history.to_vec(), targets);
            batch.push(a);
        }

        batch
    }
}

pub fn selfplay<B: Backend + AutodiffBackend>(
    config: MuzeroConfig,
    environment: RaceStrategyEnvironment,
    mut network: MuzeroModel<B>,
) {
    let mut replay_buffer = ReplayBuffer::new(config.buffer_size, config.batch_size);
    let multi_progress = MultiProgress::new();
    let num_iterations = config.num_iterations as u64;
    let pb = ProgressBar::new(num_iterations);
    pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:100.cyan/blue}] {human_pos}/{human_len} ({per_sec}, ETA: {eta})")
            .unwrap()
            .progress_chars("█▓░"));

    let pb = multi_progress.add(pb);

    for _ in (0..num_iterations).progress_with(pb) {
        let game = play_game(config, environment.clone(), &network);
        replay_buffer.save_game(game);
        train_network(config, &mut network, &mut replay_buffer);
    }
}

pub fn play_game<B: AutodiffBackend + Backend>(
    config: MuzeroConfig,
    environment: RaceStrategyEnvironment,
    network: &MuzeroModel<B>,
) -> Game {
    let mut game = Game::new_game(environment, config);
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
    game
}

fn run_mcts<B: Backend + AutodiffBackend>(
    config: MuzeroConfig,
    root: &mut Node<B>,
    action_history: ActionHistory,
    network: &MuzeroModel<B>,
    mini_max_stats: &mut MinMaxStats,
) {
    let capacity = config.num_simulation as usize;
    for _ in 0..config.num_simulation {
        let mut search_path = Vec::with_capacity(capacity);
        let root_ptr = root as *mut Node<B>;
        search_path.push(root_ptr);

        let mut node = unsafe { &mut *root_ptr };

        let mut history = action_history.clone();

        while node.expanded() {
            let (action, child_node) = select_child(config, node, mini_max_stats);
            node = child_node;
            history.add_action(action);
            search_path.push(node);
        }
        // I've reached a leaf node. To expand it, I need the parent's hidden state (index -2)
        // and the action taken. The dynamics network combines parent_state + action to predict
        // where the child state would be and what reward ii would get.
        let index = search_path.len() - 2;
        let parent = search_path[index];
        let latent_representation = unsafe { &*parent }.hidden_state.clone().unwrap();

        let network_output =
            network.get_action_probs_and_value(latent_representation, history.last_action().index);
        let value = network_output.value;
        expand_node(
            node,
            history.to_play(),
            history.action_space(),
            network_output,
        );
        backpropagate(
            search_path,
            value,
            history.to_play(),
            config.discount,
            mini_max_stats,
        );
    }
}

fn select_child<'a, B: Backend>(
    config: MuzeroConfig,
    node: &'a mut Node<B>,
    mini_max_stats: &mut MinMaxStats,
) -> (Action, &'a mut Node<B>) {
    let mut highest_score = f32::NEG_INFINITY;
    let mut best_action = Action::new(0);

    for (action, child) in node.children_dict.iter() {
        let score = ucb_score(config, &node, child, mini_max_stats);

        if score > highest_score {
            highest_score = score;
            best_action = Action::new(*action);
        }
    }

    let best_child = node.children_dict.get_mut(&best_action.index).unwrap();

    (best_action, best_child)
}

fn ucb_score<B: Backend>(
    config: MuzeroConfig,
    parent: &Node<B>,
    child: &Node<B>,
    mini_max_stats: &mut MinMaxStats,
) -> f32 {
    let mut pb_c = f32::ln(((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) as f32)
        + config.pb_c_init;

    pb_c *= f32::sqrt(parent.visit_count as f32) / (child.visit_count + 1) as f32;

    let prior_score = pb_c * child.prior;
    let value_score = if child.visit_count > 0 {
        let value = child.reward + config.discount * child.value();
        mini_max_stats.normalize(value)
    } else {
        0.0
    };

    prior_score + value_score
}

fn expand_node<B: Backend + AutodiffBackend>(
    node: &mut Node<B>,
    to_play: Player,
    _actions: Vec<Action>,
    network_output: MuzeroModelOutput<B>,
) {
    node.to_play = to_play;
    node.hidden_state = Some(Tensor::from_inner(network_output.latent_representation));
    node.reward = network_output.reward;

    for (action, prior) in network_output.action_probabilities.iter().enumerate() {
        node.children_dict.insert(action, Node::new(*prior));
    }
}

fn backpropagate<B: Backend>(
    search_path: Vec<*mut Node<B>>,
    value: f32,
    _to_play: Player,
    discount: f32,
    mini_max_stats: &mut MinMaxStats,
) {
    let mut value = value;

    for &node_ptr in search_path.iter().rev() {
        unsafe {
            let node = &mut *node_ptr;
            node.value_sum += value;
            node.visit_count += 1;
            mini_max_stats.update(node.value());
            value = node.reward + discount * value;
        }
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
    let action = get_action(action_probs, num_moves, rng, config);
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

fn get_action(
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
    for _ in 0..config.training_steps {
        let batch: Vec<(Vec<f32>, Vec<Action>, Vec<(f32, Option<f32>, Vec<f32>)>)> =
            replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config);
        update_weights(network, batch, config);
    }
}

fn update_weights<B: Backend + AutodiffBackend>(
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
        let network_output = network.initial_inference(&observation);
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
                    &[target_reward.expect("The value found to be None")],
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

fn scale_gradient<B: Backend + AutodiffBackend>(tensor: Tensor<B, 2>, scale: f32) -> Tensor<B, 2> {
    tensor.clone() * scale + tensor.detach() * (1.0 - scale)
}
