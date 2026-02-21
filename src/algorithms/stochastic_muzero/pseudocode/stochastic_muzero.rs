use std::
    collections::HashMap
;

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

use crate::{
    algorithms::{
        stochastic_muzero::{
            stochastic_muzero::{Action, Outcome}, stochastic_muzero_config::StochasticMuzeroConfig, stochastic_muzero_model::{
                AfterState, LatentState, Network, NetworkOutput, encode_action,
                scalar_to_support_batch,
            }
        },
        strategy::RaceStrategyEnvironment,
    },
    environment::AgentInfo,
    traits::gym::{GymEnvironment, MCTSGymEnvironment},
    utils::{BoundedStack, argmax},
};


#[derive(Debug, Clone, Copy)]
pub struct Player {}

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
    /// History of actions taken. Will only ever contain ActionOrOutcome::Action variant.
    history: Vec<ActionOrOutcome>,
    rewards: Vec<f32>,
    observations: Vec<Vec<f32>>,
    child_visits: Vec<Vec<f32>>,
    root_values: Vec<f32>,
    action_space_size: usize,
    discount: f32,
    terminal: bool, // extra for me
    initial_player: Player,
}

impl Game {
    pub fn new_game(environment: RaceStrategyEnvironment, config: StochasticMuzeroConfig) -> Game {
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
            initial_player: Player {},
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

    pub fn apply(&mut self, action: ActionOrOutcome) -> (Vec<f32>, HashMap<String, AgentInfo>) {
        let (observation, reward, terminated, truncated, info) =
            self.environment.step(action.index());
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
        for child in root.children.values() {
            sum_visits += child.visit_count;
        }

        let mut action_space = Vec::with_capacity(self.action_space_size);
        for index in 0..self.action_space_size {
            action_space.push(Action { index });
        }

        let mut visits = Vec::with_capacity(self.action_space_size);
        for action in action_space {
            if let Some(child) = root.children.get(&action.index) {
                visits.push(child.visit_count as f32 / sum_visits as f32)
            } else {
                visits.push(0.0);
            }
        }
        self.child_visits.push(visits);

        self.root_values.push(root.value());
    }

    pub fn make_image(&self, state_index: usize) -> Vec<f32> {
        // i dont know what this is supposed to be when iim not doing images but rather MLP
        self.observations[state_index].clone()
    }

    pub fn make_targets(
        &self,
        state_index: usize,
        num_unroll_steps: u32,
        td_steps: u32,
        _to_play: Player,
        config: StochasticMuzeroConfig,
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
                // let uniform_prob = 1.0 / action_space_size as f32;
                // let absorbing_state_action_probabilities = vec![uniform_prob; action_space_size];
                let mut terminal_policy = vec![0.0; action_space_size];
                terminal_policy[action_space_size - 1] = 1.0; // ← None action
                // targets.push((0.0, last_reward, absorbing_state_action_probabilities ));
                targets.push((0.0, last_reward, terminal_policy));
            }
        }

        targets
    }

    pub fn to_play(&self) -> Player {
        Player {}
    }

    pub fn action_outcome_history(&self) -> ActionOutcomeHistory {
        ActionOutcomeHistory {
            initial_player: self.initial_player,
            history: self.history.clone(),
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
        config: StochasticMuzeroConfig,
    ) -> Vec<(
        Vec<Vec<f32>>,                     // observations
        Vec<ActionOrOutcome>,              // actions
        Vec<(f32, Option<f32>, Vec<f32>)>, // targets
    )> {
        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            let game = self.sample_game();
            let start_index = self.sample_position(&game);

            let targets = game.make_targets(
                start_index,
                num_unroll_steps,
                td_steps,
                game.to_play(),
                config,
            );

            let mut observations = Vec::new();
            let mut actions = Vec::new();

            for k in 0..=num_unroll_steps as usize {
                let current_index = start_index + k;

                // Observation: Repeat last if past end
                if current_index < game.observations.len() {
                    observations.push(game.observations[current_index].clone());
                } else {
                    observations.push(game.observations.last().unwrap().clone());
                }

                // Action: Use dummy action for absorbing states (won't affect learning due to targets)
                if k < num_unroll_steps as usize {
                    if current_index < game.history.len() {
                        actions.push(game.history[current_index].clone());
                    } else {
                        // Dummy action - targets already have uniform policy
                        actions.push(ActionOrOutcome::Action(Action::new(0)));
                    }
                }
            }

            batch.push((observations, actions, targets));
        }

        batch
    }
}

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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ActionOrOutcome {
    Action(Action),
    Outcome(Outcome),
}

impl ActionOrOutcome {
    pub fn new_outcome(index: usize) -> ActionOrOutcome {
        ActionOrOutcome::Outcome(Outcome::new(index))
    }

    pub fn new_action(index: usize) -> ActionOrOutcome {
        ActionOrOutcome::Action(Action::new(index))
    }

    pub fn index(&self) -> usize {
        match &self {
            ActionOrOutcome::Action(action) => action.index,
            ActionOrOutcome::Outcome(outcome) => outcome.index,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActionOutcomeHistory {
    pub initial_player: Player,
    history: Vec<ActionOrOutcome>,
}

impl ActionOutcomeHistory {
    pub fn new(player: Player, history: Option<Vec<ActionOrOutcome>>) -> ActionOutcomeHistory {
        let history = if let Some(history) = history {
            history
        } else {
            Vec::new()
        };
        Self {
            initial_player: player,
            history,
        }
    }
    pub fn add_action_or_outcome(&mut self, action_or_outcome: ActionOrOutcome) {
        self.history.push(action_or_outcome);
    }

    pub fn last_action_or_outcome(&self) -> ActionOrOutcome {
        self.history
            .last()
            .expect("There has no last element")
            .clone()
    }

    pub fn to_play(&self) -> Player {
        Player {}
    }
}

#[derive(Debug, Clone)]
enum NodeState<B: Backend> {
    LatentState(LatentState<B>),
    AfterState(AfterState<B>),
}

pub struct Node<B: Backend> {
    visit_count: u32,
    to_play: Player,
    prior: f32,
    value_sum: f32,
    children: HashMap<usize, Node<B>>,
    state: Option<NodeState<B>>,
    is_chance: bool,
    reward: f32,
}

impl<B: Backend> Node<B> {
    pub fn new(prior: f32, is_chance: Option<bool>) -> Self {
        let is_chance = match is_chance {
            Some(value) => value,
            None => false,
        };
        Self {
            visit_count: 0,
            to_play: Player {},
            prior,
            value_sum: 0.0,
            children: HashMap::with_capacity(200),
            state: None,
            is_chance,
            reward: 0.0,
        }
    }

    pub fn expanded(&self) -> bool {
        self.children.len() > 0
    }

    pub fn value(&self) -> f32 {
        if self.visit_count == 0 {
            return 0.0;
        }
        self.value_sum / self.visit_count as f32
    }
}

fn run_mcts<B: Backend + AutodiffBackend>(
    config: StochasticMuzeroConfig,
    root: &mut Node<B>,
    network: &Network<B>,
    action_outcome_history: ActionOutcomeHistory,
    mini_max_stats: &mut MinMaxStats,
) {
    let capacity = config.num_simulations as usize;

    for _ in 0..config.num_simulations {
        let mut history = action_outcome_history.clone();
        let mut search_path = Vec::with_capacity(capacity);
        let root_ptr = root as *mut Node<B>;
        search_path.push(root_ptr);

        let mut node = unsafe { &mut *root_ptr };

        while node.expanded() {
            let (action_or_outcome, child_node) = select_child(config, node, mini_max_stats);
            node = child_node;
            history.add_action_or_outcome(action_or_outcome);
            search_path.push(node);
        }

        let index = search_path.len() - 2;
        let parent = search_path[index];
        let latent_representation = unsafe { &*parent }.state.clone().unwrap();
        let is_chance = unsafe { &*parent }.is_chance;

        let (state, network_output, is_child_chance) = if is_chance {
            let outcome = match history.last_action_or_outcome() {
                ActionOrOutcome::Action(_action) => panic!("Should be an outcome"),
                ActionOrOutcome::Outcome(outcome) => outcome,
            };

            let parent_state = match latent_representation {
                NodeState::LatentState(_latent_state) => panic!("Should be an afterstate"),
                NodeState::AfterState(after_state) => after_state,
            };
            let (child_state, reward) = network.dynamics(parent_state, outcome);
            let mut network_output = network.predictions(child_state.clone());
            network_output.reward = Some(reward);
            let is_child_chance = false;

            (
                NodeState::LatentState(child_state),
                network_output,
                is_child_chance,
            )
        } else {
            let action = match history.last_action_or_outcome() {
                ActionOrOutcome::Action(action) => action,
                ActionOrOutcome::Outcome(_outcome) => panic!("Should be an action"),
            };

            let parent_state = match latent_representation {
                NodeState::LatentState(latent_state) => latent_state,
                NodeState::AfterState(_after_state) => panic!("Should be a latentstate"),
            };
            let child_state = network.afterstate_dynamics(parent_state, action);
            let network_output = network.afterstate_predictions(child_state.clone());
            let is_child_chance = true;

            (
                NodeState::AfterState(child_state),
                network_output,
                is_child_chance,
            )
        };
        let value = network_output.value;
        let discount = config.discount;
        expand_node(
            node,
            state,
            network_output,
            history.to_play(),
            is_child_chance,
        );

        backpropagate(
            search_path,
            value,
            history.to_play(),
            discount,
            mini_max_stats,
        );
    }
}

fn select_child<'a, B: Backend>(
    config: StochasticMuzeroConfig,
    node: &'a mut Node<B>,
    mini_max_stats: &mut MinMaxStats,
) -> (ActionOrOutcome, &'a mut Node<B>) {
    if node.is_chance {
        let mut rng = rng();
        let capacity = config.action_space;
        let mut outcomes = Vec::with_capacity(capacity);
        let mut probs = Vec::with_capacity(capacity);
        for (outcome, child) in node.children.iter() {
            outcomes.push(outcome);
            probs.push(child.prior);
        }
        let dist = WeightedIndex::new(&probs).unwrap();

        let outcome = dist.sample(&mut rng);

        return (
            ActionOrOutcome::new_outcome(outcome),
            node.children.get_mut(&outcome).unwrap(),
        );
    }

    let mut highest_score = f32::NEG_INFINITY;
    let mut best_action = ActionOrOutcome::new_action(0);

    for (action, child) in node.children.iter() {
        let score = ucb_score(config, &node, child, mini_max_stats);

        if score > highest_score {
            highest_score = score;
            best_action = ActionOrOutcome::new_action(*action);
        }
    }

    let best_child = node.children.get_mut(&best_action.index()).unwrap();

    (best_action, best_child)
}

fn ucb_score<B: Backend>(
    config: StochasticMuzeroConfig,
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

fn expand_node<B: Backend>(
    node: &mut Node<B>,
    state: NodeState<B>,
    network_output: NetworkOutput,
    player: Player,
    is_chance: bool,
) {
    node.to_play = player;
    node.state = Some(state);
    node.is_chance = is_chance;
    node.reward = network_output.reward.unwrap_or(0.0);

    for (action_or_outcome, prob) in network_output.probabilties {
        node.children
            .insert(action_or_outcome.index(), Node::new(prob, None));
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

pub fn selfplay<B: Backend + AutodiffBackend>(
    config: StochasticMuzeroConfig,
    environment: RaceStrategyEnvironment,
    mut network: Network<B>,
    
) {
    
    let mut replay_buffer = ReplayBuffer::new(config.buffer_size, config.batch_size);
    let multi_progress = MultiProgress::new();
    let num_iterations = 8; //config.num_iterations as u64;
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
    config: StochasticMuzeroConfig,
    environment: RaceStrategyEnvironment,
    network: &Network<B>,
) -> Game {
    let mut game = Game::new_game(environment, config);
    let (mut observation, _info) = game.reset();
    while !game.terminal() && game.history.len() < config.max_moves {
        game.clear();
        let mini_max_stats =
            &mut MinMaxStats::new(config.known_maximum_reward, config.known_minimum_reward);

        let mut root = Node::<B>::new(0.0, None);

        let legal_actions = game.legal_actions();
        // the network output of get intial has exploration noise

        let dirichlet_noise = true;
        let (network_output, latent_state) = network.get_initial_action_probs_and_value(
            &observation,
            &legal_actions,
            dirichlet_noise,
            config,
        );
        let value = network_output.value;
        let is_chance = false;
        let state = NodeState::LatentState(latent_state);

        expand_node(&mut root, state, network_output, game.to_play(), is_chance);

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
            &network,
            game.action_outcome_history(),
            mini_max_stats,
        );

        let action = select_action(config, game.history.len(), &root, &network);
        (observation, _) = game.apply(ActionOrOutcome::Action(Action::new(action)));

        game.store_search_statistics(&root);
    }
    game
}


fn select_action<B: Backend + AutodiffBackend>(
    config: StochasticMuzeroConfig,
    num_moves: usize,
    node: &Node<B>,
    _network: &Network<B>,
) -> usize {
    let action_probs = get_action_probs(config, node);
    let rng = &mut rng();
    let action = get_action(action_probs, num_moves, rng, config);
    action
}

fn argmax_action<B: Backend>(config: StochasticMuzeroConfig, node: &Node<B>) -> usize {
    let action_probs = get_action_probs(config, node);
    let action = argmax(&action_probs);
    action
}

fn get_action_probs<B: Backend>(config: StochasticMuzeroConfig, node: &Node<B>) -> Vec<f32> {
    let mut action_probs = vec![0.0; config.action_space];

    for (action, child) in node.children.iter() {
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
    config: StochasticMuzeroConfig,
) -> usize {
    let action_probabilities = apply_temperature(action_probabilities, config);

    let dist = WeightedIndex::new(&action_probabilities).unwrap();

    let action = dist.sample(&mut rng);

    action
}

fn apply_temperature(
    mut action_probabilities: Vec<f32>,
    config: StochasticMuzeroConfig,
) -> Vec<f32> {
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
    config: StochasticMuzeroConfig,
    network: &mut Network<B>,
    replay_buffer: &mut ReplayBuffer,
) {
    for it in 0..config.training_steps {
        println!("{it}/{}", config.training_steps);
        let batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps, config);
        update_weights_batch(network, batch, config);
    }
    println!("Done training")
}

fn update_weights_batch<B: Backend + AutodiffBackend>(
    network: &mut Network<B>,
    batch: Vec<(
        Vec<Vec<f32>>,
        Vec<ActionOrOutcome>,
        Vec<(f32, Option<f32>, Vec<f32>)>,
    )>,
    config: StochasticMuzeroConfig,
) {
    let batch_size = batch.len();
    let action_space = config.action_space;
    let device = &network.get_device();

    // -- Pre-allocate vectors --
    let mut initial_observations_data: Vec<Tensor<B, 1>> = Vec::with_capacity(batch_size);
    let mut initial_value_targets = Vec::with_capacity(batch_size);
    let mut initial_policy_targets_data: Vec<Tensor<B, 1>> = Vec::with_capacity(batch_size);

    // Transposed data for unrolling
    let mut all_observations = Vec::with_capacity(batch_size);
    let mut all_actions = Vec::with_capacity(batch_size);
    let mut all_targets = Vec::with_capacity(batch_size);

    for (observations, actions, targets) in batch {
        // t=0 data
        initial_observations_data.push(Tensor::from_data(observations[0].as_slice(), device));
        let (target_val, _, target_pol) = &targets[0];
        initial_value_targets.push(*target_val);
        initial_policy_targets_data.push(Tensor::from_data(target_pol.as_slice(), device));

        // t>0 data
        all_observations.push(observations[1..].to_vec());
        all_actions.push(actions);
        all_targets.push(targets[1..].to_vec());
    }

    // 1. Initial Inference (t=0)
    let obs_batch = Tensor::stack(initial_observations_data, 0);
    let target_val_batch = scalar_to_support_batch(&initial_value_targets, device);
    let target_pol_batch = Tensor::stack(initial_policy_targets_data, 0);

    let mut latent_state = network.networks.representation_network.forward(obs_batch);
    let (pol_pred, val_pred) = network
        .networks
        .prediction_network
        .forward(latent_state.clone());

    let mut total_loss = cross_entropy_with_logits(pol_pred, target_pol_batch)
        + cross_entropy_with_logits(val_pred, target_val_batch);

    // 2. Unroll Loop
    for k in 0..config.num_unroll_steps as usize {
        let gradient_scale = 1.0 / config.num_unroll_steps as f32;

        // Gather step k data across batch
        let mut actions_k: Vec<Tensor<B, 1>> = Vec::with_capacity(batch_size);
        let mut next_obs_k: Vec<Tensor<B, 1>> = Vec::with_capacity(batch_size);
        let mut val_targs_k = Vec::with_capacity(batch_size);
        let mut rew_targs_k = Vec::with_capacity(batch_size);
        let mut pol_targs_k: Vec<Tensor<B, 1>> = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // Action
            let action_idx = match all_actions[i][k] {
                ActionOrOutcome::Action(a) => a.index,
                // If you padded correctly in ReplayBuffer, this panic won't hit
                _ => panic!("Outcome found in action slot"),
            };
            actions_k.push(encode_action(action_idx, action_space, device).squeeze_dim(0));

            // Next Observation (for Encoder)
            next_obs_k.push(Tensor::from_data(all_observations[i][k].as_slice(), device));

            // Targets
            let (v, r, p) = &all_targets[i][k];
            val_targs_k.push(*v);
            rew_targs_k.push(r.unwrap_or(0.0));
            pol_targs_k.push(Tensor::from_data(p.as_slice(), device));
        }

        // Stack Tensors
        let action_batch = Tensor::stack(actions_k, 0); // [Batch, ActionDim]
        let next_obs_batch = Tensor::stack(next_obs_k, 0); // [Batch, ObsDim]
        let pol_targs_batch = Tensor::stack(pol_targs_k, 0);
        let val_targs_batch = scalar_to_support_batch(&val_targs_k, device);
        let rew_targs_batch = scalar_to_support_batch(&rew_targs_k, device);

        // --- Core Logic ---

        // A. Afterstate Dynamics
        let afterstate = network
            .networks
            .afterstate_dynamics_network
            .forward(latent_state, action_batch);

        // B. Afterstate Prediction
        let (chance_logits, q_logits) = network
            .networks
            .afterstate_prediction_network
            .forward(afterstate.clone());

        let afterstate_loss = cross_entropy_with_logits(q_logits, val_targs_batch.clone());

        // C. Encoder (Ground Truth Code)
        let chance_code_true = network.networks.encoder_network.forward(next_obs_batch);

        // Detach chance code for the loss target (we don't want to optimize encoder via this loss directly, usually)
        let chance_loss =
            cross_entropy_with_logits(chance_logits, chance_code_true.clone().detach());

        // D. Dynamics
        let (rew_logits, next_latent) = network
            .networks
            .dynamics_network
            .forward(afterstate, chance_code_true);

        // E. Prediction
        let (next_pol, next_val) = network
            .networks
            .prediction_network
            .forward(next_latent.clone());

        let dynamics_loss = cross_entropy_with_logits(next_pol, pol_targs_batch)
            + cross_entropy_with_logits(next_val, val_targs_batch)
            + cross_entropy_with_logits(rew_logits, rew_targs_batch);

        // --- Accumulate Loss with Mask & Scaling ---
        let step_loss = afterstate_loss + chance_loss + dynamics_loss;

        // But here we just scale the loss contribution directly.
        total_loss = total_loss + scale_gradient(step_loss.unsqueeze_dim(0), gradient_scale).mean();

        // Prepare next state (scaled gradient 0.5)
        latent_state = scale_gradient(next_latent, 0.5);
    }

    // Optimize
    let grads = total_loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &network.networks);
    let lr = config.learning_rate_init as f64;
    network.networks = network
        .optimizer
        .step(lr, network.networks.clone(), grads_params);
}

fn scale_gradient<B: Backend + AutodiffBackend>(tensor: Tensor<B, 2>, scale: f32) -> Tensor<B, 2> {
    tensor.clone() * scale + tensor.detach() * (1.0 - scale)
}
