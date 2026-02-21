use std::{
    f32::NEG_INFINITY,
    marker::PhantomData,
    rc::Rc,
    sync::{Arc, RwLock},
};

use rand::Rng;

use crate::{
    algorithms::{
        alpha_zero::{alpha_zero_config::AlphaZeroConfig, node::AlphaZeroNode},
        muzero_old::muzero_mcts::MinMaxStats,
    },
    traits::{actor_critic::ActorCritic, gym::MCTSGymEnvironment},
};

pub struct AlphaZeroMCTS<Model: ActorCritic> {
    config: AlphaZeroConfig,
    action_space: usize,
    pub model: Arc<RwLock<Model>>,
    _not_thread_safe: PhantomData<Rc<()>>,
}

impl<Model: ActorCritic<ObservationType = Vec<f32>>> AlphaZeroMCTS<Model> {
    pub fn new(config: AlphaZeroConfig, model: Arc<RwLock<Model>>) -> AlphaZeroMCTS<Model> {
        let action_space = model.read().unwrap().get_action_space();
        Self {
            config,
            action_space,
            model,
            _not_thread_safe: PhantomData,
        }
    }

    pub fn search<Environment: MCTSGymEnvironment>(
        &self,
        current_env_state: Environment,
        current_observation: Environment::Observation,
        root_dirichlet_noise: bool,
    ) -> Vec<f32>
    where
        Environment: MCTSGymEnvironment<Observation = Vec<f32>, Reward = f32>,
    {
        if self.config.use_gumbel {
            self.gumbel_search(current_env_state, current_observation)
        } else {
            self.puct_search(current_env_state, current_observation, root_dirichlet_noise)
        }
    }

    pub fn puct_search<Environment: MCTSGymEnvironment>(
        &self,
        current_env_state: Environment,
        current_observation: Environment::Observation,
        root_dirichlet_noise: bool,
    ) -> Vec<f32>
    where
        Environment: MCTSGymEnvironment<Observation = Vec<f32>, Reward = f32>,
    {
        let model = self.model.read().unwrap();
        let capacity = self.config.num_searches as usize;
        let mut root = AlphaZeroNode::new_root(
            current_env_state,
            0.0,
            false,
            current_observation,
            self.config,
            None,
            1,
        );
        let dirichlet_noise = true;
        let (action_probabilties, _value) =
            model.get_action_probs_and_value(&root, dirichlet_noise, self.config);

        let mut min_max_stats = MinMaxStats::new(
            self.config.known_maximum_reward,
            self.config.known_minimum_reward,
        );

        root.expand(action_probabilties);

        for _ in 0..self.config.num_searches {
            let mut tree_path = Vec::with_capacity(capacity); // Track the path
            tree_path.push(&mut root as *mut _); // always add root
            let mut node = &mut root;

            while node.expanded {
                node = node.select(&min_max_stats);
                tree_path.push(node as *mut _);
            }
            let value = if !node.done {
                let (action_probabilties, value) =
                    model.get_action_probs_and_value(node, root_dirichlet_noise, self.config);
                node.expand(action_probabilties);
                value
            } else {
                node.reward
            };
            backpropagate(tree_path, value, &mut min_max_stats);
        }

        let mut action_probs = vec![0.0; self.action_space];

        for child in root.children() {
            action_probs[child.action.unwrap()] = child.visit_count as f32
        }

        let action_sum = action_probs.iter().sum::<f32>();
        for action_prob in action_probs.iter_mut() {
            *action_prob /= action_sum
        }

        action_probs
    }

    pub fn gumbel_search<Environment: MCTSGymEnvironment>(
        &self,
        current_env_state: Environment,
        current_observation: Environment::Observation,
    ) -> Vec<f32>
    where
        Environment: MCTSGymEnvironment<Observation = Vec<f32>, Reward = f32>,
    {
        let model = self.model.read().unwrap();

        let mut root = AlphaZeroNode::new_root(
            current_env_state.clone(),
            0.0,
            false,
            current_observation,
            self.config,
            None,
            1,
        );

        // Raw Logits & Value
        // use logits directly for numerical stability in Gumbel-Max
        let (mut policy_logits, value_logit) = model.get_raw_action_and_value_logits(&root);

        // Mask invalid actions in logits
        let legal_actions = root.state.get_legal_actions();
        for (i, is_legal) in legal_actions.iter().enumerate() {
            if *is_legal == 0.0 {
                policy_logits[i] = NEG_INFINITY;
            }
        }

        // also need probabilities for V_mix calculation later
        // Simple softmax implementation for the valid actions
        let max_logit = policy_logits.iter().fold(NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0;
        let mut root_policy_probs = vec![0.0; self.action_space];
        for (i, &logit) in policy_logits.iter().enumerate() {
            if logit > NEG_INFINITY {
                let p = (logit - max_logit).exp();
                root_policy_probs[i] = p;
                sum_exp += p;
            }
        }
        for p in root_policy_probs.iter_mut() {
            *p /= sum_exp;
        }

        root.expand(root_policy_probs.clone());

        let mut min_max_stats = MinMaxStats::new(
            self.config.known_maximum_reward,
            self.config.known_minimum_reward,
        );

        //  Gumbel-Top-k Sampling
        let mut rng = rand::rng();
        let gumbel_noise: Vec<f32> = (0..self.action_space)
            .map(|_| -(-rng.random::<f32>().ln()).ln())
            .collect();

        // Score = g + logits
        let m = self
            .config
            .gumbel_sample_size
            .min(self.action_space)
            .min(self.config.num_searches as usize);
        let mut candidates: Vec<usize> = (0..self.action_space)
            .filter(|&i| legal_actions[i] == 1.0) // Only consider legal actions
            .collect();

        candidates.sort_by(|&a, &b| {
            let score_a = gumbel_noise[a] + policy_logits[a];
            let score_b = gumbel_noise[b] + policy_logits[b];
            score_b.partial_cmp(&score_a).unwrap()
        });
        candidates.truncate(m);

        // Sequential Halving
        let num_simulations = self.config.num_searches as usize;
        let num_phases = (m as f32).log2().ceil() as usize;
        let visits_per_phase = if num_phases > 0 {
            num_simulations / num_phases
        } else {
            num_simulations
        };
        let mut active_candidates = candidates.clone();

        for _phase in 0..num_phases {
            if active_candidates.len() <= 1 {
                break;
            }
            let visits_per_act = (visits_per_phase / active_candidates.len()).max(1);

            for &action_idx in &active_candidates {
                for _ in 0..visits_per_act {
                    let mut tree_path = vec![&mut root as *mut _];
                    let mut node = &mut root;

                    // Force first selection to be the candidate
                    if node.children_dict.contains_key(&action_idx) {
                        node = node.get_child_node(action_idx);
                    } else {
                        node = node.create_child_node(action_idx);
                    }
                    tree_path.push(node as *mut _);

                    // Standard PUCT for deeper levels
                    while node.expanded {
                        node = node.select(&min_max_stats);
                        tree_path.push(node as *mut _);
                    }

                    let value = if !node.done {
                        let (probs, val) =
                            model.get_action_probs_and_value(node, false, self.config);
                        node.expand(probs);
                        val
                    } else {
                        node.reward
                    };

                    backpropagate(tree_path, value, &mut min_max_stats);
                }
            }

            // Halving Step: Score = g + logits + sigma(q)
            let max_visit = root.children().map(|c| c.visit_count).max().unwrap_or(1);
            let sigma_scale = (self.config.c_visit + max_visit as f32) * self.config.c_scale;

            active_candidates.sort_by(|&a, &b| {
                let get_q = |act: usize| -> f32 {
                    root.children_dict
                        .get(&act)
                        .map(|c| min_max_stats.normalize(c.value()))
                        .unwrap_or(0.0)
                };

                let score_a = gumbel_noise[a] + policy_logits[a] + sigma_scale * get_q(a);
                let score_b = gumbel_noise[b] + policy_logits[b] + sigma_scale * get_q(b);
                score_b.partial_cmp(&score_a).unwrap()
            });

            let keep_count = active_candidates.len() / 2;
            active_candidates.truncate(keep_count);
        }

        //calculatinh V_mix (Mixed Value)
        // v_mix = (1 / (1 + sum_N)) * (v_net + sum_N * v_search)
        // where v_search = sum(pi(a)*q(a)) / sum(pi(a)) for visited actions
        let root_v_net = value_logit; // Assuming logic exists to unscale raw logit

        let mut sum_n = 0;
        let mut weighted_q_sum = 0.0;
        let mut weight_sum = 0.0;

        for child in root.children() {
            if let Some(act_idx) = child.action {
                sum_n += child.visit_count;
                let prob = root_policy_probs[act_idx];
                let q_norm = min_max_stats.normalize(child.value());

                weighted_q_sum += prob * q_norm;
                weight_sum += prob;
            }
        }

        let v_mix = if weight_sum > 0.0 {
            let v_search = weighted_q_sum / weight_sum;
            let total_n = sum_n as f32;
            (1.0 / (1.0 + total_n)) * (min_max_stats.normalize(root_v_net) + total_n * v_search)
        } else {
            min_max_stats.normalize(root_v_net)
        };

        // Construct Improved Policy (Completed Q-Values)
        let mut improved_logits = vec![NEG_INFINITY; self.action_space];
        let max_visit = root.children().map(|c| c.visit_count).max().unwrap_or(1);
        let sigma_scale = (self.config.c_visit + max_visit as f32) * self.config.c_scale;

        for action in 0..self.action_space {
            if legal_actions[action] == 0.0 {
                continue;
            }

            // Completed Q: If visited use Q(a), else use v_mix
            let q_val = if let Some(child) = root.children_dict.get(&action) {
                if child.visit_count > 0 {
                    min_max_stats.normalize(child.value())
                } else {
                    v_mix
                }
            } else {
                v_mix
            };

            improved_logits[action] = policy_logits[action] + (sigma_scale * q_val);
        }

        // Softmax to get target policy
        let max_imp_logit = improved_logits.iter().fold(NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0;
        let mut policy_target = vec![0.0; self.action_space];

        for (i, &l) in improved_logits.iter().enumerate() {
            if l > NEG_INFINITY {
                let exp_val = (l - max_imp_logit).exp();
                policy_target[i] = exp_val;
                sum_exp += exp_val;
            }
        }
        for p in policy_target.iter_mut() {
            if sum_exp > 0.0 {
                *p /= sum_exp;
            }
        }

        policy_target
    }
}

fn backpropagate<Environment: MCTSGymEnvironment>(
    tree_path: Vec<*mut AlphaZeroNode<Environment>>,
    value: f32,
    min_max_stats: &mut MinMaxStats,
) {
    for &node_ptr in tree_path.iter() {
        unsafe {
            let node: &mut AlphaZeroNode<Environment> = &mut *node_ptr;
            node.value_sum += value;
            node.visit_count += 1;
            min_max_stats.update(node.value());
        }
    }
}
