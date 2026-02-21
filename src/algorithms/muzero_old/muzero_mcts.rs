use std::{
    marker::PhantomData,
    rc::Rc,
    sync::{Arc, RwLock},
};

use burn::{prelude::Backend, tensor::backend::AutodiffBackend};

use crate::algorithms::muzero_old::{
    muzero_config::MuzeroConfig,
    muzero_model::{MuzeroModel, MuzeroModelOutput},
    muzero_node::MuzeroNode,
};

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
// Goin got use for both Muzero and Stochastic Muzero so model doesnt need to be given the env and can
// plan by itself too

pub struct MuzeroMcts<B: Backend + AutodiffBackend> {
    pub config: MuzeroConfig,
    action_space: usize,
    pub model: Arc<RwLock<MuzeroModel<B>>>,
    _not_thread_safe: PhantomData<Rc<()>>,
}

impl<B: Backend + AutodiffBackend> MuzeroMcts<B> {
    pub fn new(config: MuzeroConfig, model: Arc<RwLock<MuzeroModel<B>>>) -> Self {
        let action_space = config.action_space;
        Self {
            config,
            action_space,
            model,
            _not_thread_safe: PhantomData,
        }
    }
    pub fn search(&self, observation: Vec<f32>, legal_actions: Vec<f32>) -> (Vec<f32>, f32) {
        let model = self.model.read().unwrap();
        let capacity = self.config.num_simulation as usize;

        // Use representation function to get initial hidden state
        let dirichlet_noise = true;
        let MuzeroModelOutput {
            value: root_value,
            reward: _,
            latent_representation: initial_latent_representation,
            action_probabilities,
        }: MuzeroModelOutput<B> = model.get_initial_action_probs_and_value(
            &observation,
            &legal_actions,
            dirichlet_noise,
            self.config,
        );

        let mut min_max_stats = MinMaxStats::new(
            self.config.known_maximum_reward,
            self.config.known_minimum_reward,
        );

        // Create root node with initial hidden state
        let mut root = MuzeroNode::new_root(initial_latent_representation, self.config);
        root.expand(action_probabilities);
        backpropagate(
            vec![&mut root as *mut _],
            root_value,
            self.config.discount,
            &mut min_max_stats,
        );
        // Run MCTS simulations
        for _ in 0..self.config.num_simulation {
            let mut tree_path = Vec::with_capacity(capacity);
            tree_path.push(&mut root as *mut _);

            let mut node: &mut MuzeroNode<<B as AutodiffBackend>::InnerBackend, 2> = &mut root;

            while node.expanded {
                // Similar to Alphazero method, I am using select action instead of select
                // as now each node does not have access to underlying dynamics that is in model
                // so i do the selection in mcts so i can create a child if necaserry calling the model
                let action = node.select_action(&min_max_stats);

                node = if node.children_dict.contains_key(&action) {
                    node.get_child_node(action)
                } else {
                    let MuzeroModelOutput {
                        value,
                        reward,
                        latent_representation,
                        action_probabilities,
                    } = model
                        .get_action_probs_and_value(node.latent_representation.clone(), action);
                    let child = node.create_child_node(action, latent_representation, reward);

                    // To avoid calling the model again outside with the same input im backpropgating inside
                    // the while loop unlike my AlphaZero implemntation
                    tree_path.push(child);
                    child.expand(action_probabilities);
                    backpropagate(tree_path, value, self.config.discount, &mut min_max_stats);

                    break;
                };

                tree_path.push(node);
            }
        }

        let mut action_probs = vec![0.0; self.action_space];
        for child in root.children() {
            if let Some(action) = child.action {
                action_probs[action] = child.visit_count as f32;
            }
        }

        let action_sum = action_probs.iter().sum::<f32>();
        for action_prob in action_probs.iter_mut() {
            *action_prob /= action_sum;
        }

        (action_probs, root.value())
    }
}

fn backpropagate<B: Backend, const D: usize>(
    tree_path: Vec<*mut MuzeroNode<B, D>>,
    value: f32,
    discount: f32,
    min_max_stats: &mut MinMaxStats,
) {
    let mut value = value;

    for &node_ptr in tree_path.iter().rev() {
        unsafe {
            let node: &mut MuzeroNode<B, D> = &mut *node_ptr;
            node.value_sum += value;
            node.visit_count += 1;
            min_max_stats.update(node.value());
            value = node.reward + discount * value;
        }
    }
}
