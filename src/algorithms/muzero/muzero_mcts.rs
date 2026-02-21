use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use burn::{Tensor, prelude::Backend, tensor::backend::AutodiffBackend};

use crate::{
    algorithms::muzero::{
        game_environment::{Action, ActionHistory, Player},
        muzero_config::MuzeroConfig,
        muzero_model::{MuzeroModel, MuzeroModelOutput},
    },
    traits::gym::MCTSGymEnvironment,
};

pub fn run_mcts<B: Backend + AutodiffBackend>(
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

pub fn expand_node<B: Backend + AutodiffBackend>(
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

pub fn backpropagate<B: Backend>(
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

pub struct MuZeroMcts<B: Backend + AutodiffBackend> {
    pub model: Arc<RwLock<MuzeroModel<B>>>,
}

impl<B: Backend + AutodiffBackend> MuZeroMcts<B> {
    pub fn search<Enviroment: MCTSGymEnvironment>(&self) {}
}
