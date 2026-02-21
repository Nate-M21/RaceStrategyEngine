use std::collections::HashMap;

use burn::{
    Tensor,
    config::Config,
    grad_clipping::GradientClippingConfig,
    module::{AutodiffModule, Module},
    nn::{LeakyRelu, LeakyReluConfig, Linear, LinearConfig},
    optim::{AdamW, AdamWConfig, adaptor::OptimizerAdaptor},
    prelude::{Backend, ToElement},
    tensor::{TensorData, activation::softmax, backend::AutodiffBackend},
};
use rand::rng;
use rand_distr::Distribution;

use crate::algorithms::{
    alpha_zero::dirichlet::StableDirichlet,
    helpers::{STRATEGY_VALUE_SUPPORT_SIZE, normalize_hidden_state, support_to_scalar},
    stochastic_muzero::{
        stochastic_muzero::{Action, ActionOrOutcome, Outcome},
        stochastic_muzero_config::StochasticMuzeroConfig,
    },
};
#[derive(Debug, Clone)]
pub struct LatentState<B: Backend> {
    pub latent_representation: Tensor<B, 2>,
}

#[derive(Debug, Clone)]
pub struct AfterState<B: Backend> {
    pub latent_representation: Tensor<B, 2>,
}

pub struct NetworkOutput {
    pub value: f32,
    pub probabilties: HashMap<ActionOrOutcome, f32>,
    pub reward: Option<f32>,
}

unsafe impl<B: Backend + AutodiffBackend> Send for Network<B> {}
unsafe impl<B: Backend + AutodiffBackend> Sync for Network<B> {}

pub struct Network<B: Backend + AutodiffBackend> {
    pub networks: StochasticMuzeroNetworks<B>,
    pub optimizer_config: AdamWConfig, // Just for viewing purposes
    pub optimizer: OptimizerAdaptor<AdamW, StochasticMuzeroNetworks<B>, B>,
}

impl<B: Backend + AutodiffBackend> Network<B> {
    pub fn get_device(&self) -> <B as Backend>::Device {
        let device = self
            .networks
            .representation_network
            .representation_shared_trunk[0]
            .weight
            .device();

        device
    }
    pub fn display_model(&self) {
        println!(
            "{}\n\nOptimizer:\n{:?}",
            self.networks, self.optimizer_config
        )
    }
    pub fn get_initial_action_probs_and_value(
        &self,
        observation: &[f32],
        legal_actions: &[f32],
        apply_dirichlet_noise: bool,
        config: StochasticMuzeroConfig,
    ) -> (NetworkOutput, LatentState<B>) {
        let StochasticMuzeroNetworkOutput {
            value_logits,
            reward_logits,
            latent_representation,
            policy_logits,
        } = self.valid_initial_inference(observation);

        let epsilon = 1e-8;
        let mut action_probabilities = softmax(policy_logits, 0);
        action_probabilities = action_probabilities + epsilon;
        let priors_sum = action_probabilities.clone().sum();
        action_probabilities = action_probabilities.div(priors_sum);

        let value = support_to_scalar(value_logits);
        // this reward is zero
        let _reward = support_to_scalar(reward_logits);

        if apply_dirichlet_noise {
            action_probabilities = add_dirichlet_noise(action_probabilities, config);
        }
        let valid_actions = Tensor::from_data(legal_actions, &action_probabilities.device());
        action_probabilities = apply_legal_actions(action_probabilities, valid_actions);
        let action_probabilities = action_probabilities.to_data().to_vec::<f32>().unwrap();
        let mut probabilties = HashMap::new();

        for (index, prior) in action_probabilities.iter().enumerate() {
            let action = ActionOrOutcome::new_action(index);
            probabilties.insert(action, *prior);
        }

        let latent_representation = LatentState {
            latent_representation: Tensor::from_inner(latent_representation),
        };

        // im putting None to explict could just use 0.0
        let network_output = NetworkOutput {
            value,
            probabilties,
            reward: None,
        };

        (network_output, latent_representation)
    }

    fn valid_initial_inference(&self, observation: &[f32]) -> StochasticMuzeroNetworkOutput<B> {
        let device = self
            .networks
            .representation_network
            .representation_shared_trunk[0]
            .weight
            .device();
        let observation = TensorData::new(observation.to_vec(), [1, observation.len()]);
        let observation = Tensor::from_data(observation, &device);
        let latent_representation = self
            .networks
            .representation_network
            .valid()
            .forward(observation);

        let (policy_logits, value_logits) = self
            .networks
            .prediction_network
            .valid()
            .forward(latent_representation.clone());
        let reward_logits = Tensor::from_data([0], &device);

        // TODO this needs to be more robust but works for me now as ill be dealing with Tensor B , 2 all the time
        let value_logits = value_logits.squeeze_dim(0);
        let policy_logits = policy_logits.squeeze_dim(0);
        StochasticMuzeroNetworkOutput {
            value_logits,
            reward_logits,
            latent_representation,
            policy_logits,
        }
    }

    pub fn representation(&self, observation: &[f32]) -> LatentState<B> {
        let device = &self
            .networks
            .representation_network
            .representation_shared_trunk[0]
            .weight
            .device();
        let observation = Tensor::from_data(observation, device);
        let latent_representaion = self
            .networks
            .representation_network
            .valid()
            .forward(observation);
        let latent_representaion = Tensor::from_inner(latent_representaion);
        LatentState {
            latent_representation: latent_representaion,
        }
    }

    pub fn predictions(&self, state: LatentState<B>) -> NetworkOutput {
        let latent_representation = state.latent_representation.inner();
        let (policy_logits, value_logits) = self
            .networks
            .prediction_network
            .valid()
            .forward(latent_representation);
        let value_logits = value_logits.squeeze_dim(0);
        let policy_logits = policy_logits.squeeze_dim(0);
        let epsilon = 1e-8;
        let mut action_probabilities = softmax(policy_logits, 0);
        action_probabilities = action_probabilities + epsilon;
        let priors_sum = action_probabilities.clone().sum();
        action_probabilities = action_probabilities.div(priors_sum);
        let action_probabilities = action_probabilities.to_data().to_vec::<f32>().unwrap();

        let value = support_to_scalar(value_logits);
        let mut probabilties = HashMap::new();

        for (index, prior) in action_probabilities.iter().enumerate() {
            let action = ActionOrOutcome::new_action(index);
            probabilties.insert(action, *prior);
        }

        NetworkOutput {
            value,
            probabilties,
            reward: None,
        }
    }

    pub fn afterstate_dynamics(&self, state: LatentState<B>, action: Action) -> AfterState<B> {
        let device = &self
            .networks
            .afterstate_dynamics_network
            .dynamics_shared_trunk[0]
            .weight
            .device();

        let latent_representation = state.latent_representation.inner();
        let action_space: usize = self.networks.action_space;
        let action = encode_action(action.index, action_space, device);
        let after_state = self
            .networks
            .afterstate_dynamics_network
            .valid()
            .forward(latent_representation, action);
        let latent_representaion = Tensor::from_inner(after_state);
        AfterState {
            latent_representation: latent_representaion,
        }
    }

    pub fn afterstate_predictions(&self, state: AfterState<B>) -> NetworkOutput {
        let after_state = state.latent_representation.inner();
        let (chance_logits, q_value_logits) = self
            .networks
            .afterstate_prediction_network
            .valid()
            .forward(after_state);

        let q_value_logits = q_value_logits.squeeze_dim(0);
        let chance_logits = chance_logits.squeeze_dim(0);
        let epsilon = 1e-8;
        let mut action_probabilities = softmax(chance_logits, 0);
        action_probabilities = action_probabilities + epsilon;
        let priors_sum = action_probabilities.clone().sum();
        action_probabilities = action_probabilities.div(priors_sum);
        let action_probabilities = action_probabilities.to_data().to_vec::<f32>().unwrap();

        let value = support_to_scalar(q_value_logits);
        let mut probabilties = HashMap::new();

        for (index, prior) in action_probabilities.iter().enumerate() {
            let action = ActionOrOutcome::new_outcome(index);
            probabilties.insert(action, *prior);
        }

        NetworkOutput {
            value,
            probabilties,
            reward: None,
        }
    }

    pub fn dynamics(&self, state: AfterState<B>, outcome: Outcome) -> (LatentState<B>, f32) {
        let device = &self.networks.dynamics_network.dynamics_shared_trunk[0]
            .weight
            .device();

        let after_state = state.latent_representation.inner();
        let cookbook_space = self.networks.codebook_size;
        let chance_code = encode_action(outcome.index, cookbook_space, device);
        let (reward_logits, next_state) = self
            .networks
            .dynamics_network
            .valid()
            .forward(after_state, chance_code);

        let reward = support_to_scalar(reward_logits.squeeze_dim(0));
        let latent_representaion = Tensor::from_inner(next_state);

        (
            LatentState {
                latent_representation: latent_representaion,
            },
            reward,
        )
    }

    pub fn encoder(&self, observation: &[f32]) -> Outcome {
        let device = &self.networks.encoder_network.encoder_shared_trunk[0]
            .weight
            .device();

        let observation = TensorData::new(observation.to_vec(), [1, observation.len()]);
        let observation = Tensor::from_data(observation, device);

        let chance_code_one_hot: Tensor<B, 2> = self.networks.encoder_network.forward(observation);

        let index_tensor: Tensor<B, 2, burn::prelude::Int> = chance_code_one_hot.argmax(1);

        let index: usize = index_tensor.into_scalar().to_usize();

        Outcome::new(index)
    }
}

#[derive(Module, Debug)]
pub struct StochasticMuzeroNetworks<B: Backend> {
    // Standard MuZero components
    pub representation_network: RepresentationNetwork<B>, // h
    pub prediction_network: PredictionNetwork<B>,         // f

    // NEW from Muzero :Stochastic MuZero components
    pub afterstate_dynamics_network: AfterstateDynamicsNetwork<B>, // φ
    pub afterstate_prediction_network: AfterstatePredictionNetwork<B>, // ψ
    pub dynamics_network: DynamicsNetwork<B>,                      // g
    pub encoder_network: EncoderNetwork<B>,                        // e

    pub action_space: usize,
    pub codebook_size: usize,
}

#[derive(Module, Debug)]
// Network that takes observation and maps it into a latento representation
pub struct RepresentationNetwork<B: Backend> {
    representation_shared_trunk: Vec<Linear<B>>,

    activation: LeakyRelu,
}
impl<B: Backend> RepresentationNetwork<B> {
    /// takes the raw observation and encodes it into a latent representation, i prefer latent representaion to
    /// hiddent state
    pub fn forward(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = observation;

        for layer in self.representation_shared_trunk.iter() {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        x = normalize_hidden_state(x);

        x
    }
}
#[derive(Module, Debug)]
// This network is takes over the given transition function given by the known model in AlphaZero
pub struct DynamicsNetwork<B: Backend> {
    dynamics_shared_trunk: Vec<Linear<B>>,

    activation: LeakyRelu,
    reward_head: Linear<B>,
    next_state_head: Linear<B>,
}

impl<B: Backend> DynamicsNetwork<B> {
    pub fn forward(
        &self,
        after_state: Tensor<B, 2>,
        chance_code: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = Tensor::cat(vec![after_state, chance_code], 1);

        for layer in self.dynamics_shared_trunk.iter() {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        let reward = self.reward_head.forward(x.clone());

        let next_state = self.next_state_head.forward(x);
        let next_state = normalize_hidden_state(next_state);

        (reward, next_state)
    }
}
#[derive(Module, Debug)]
// This network is basically like the actor critic in AlphaZero but here it works on hidden state / latent representaion
pub struct PredictionNetwork<B: Backend> {
    prediction_shared_trunk: Vec<Linear<B>>,
    activation: LeakyRelu,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

impl<B: Backend> PredictionNetwork<B> {
    pub fn forward(&self, latent_representation: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = latent_representation;

        for layer in self.prediction_shared_trunk.iter() {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        let policy = self.policy_head.forward(x.clone());
        let value = self.value_head.forward(x);

        (policy, value)
    }
}

#[derive(Module, Debug)]
pub struct AfterstateDynamicsNetwork<B: Backend> {
    dynamics_shared_trunk: Vec<Linear<B>>,
    activation: LeakyRelu,
    afterstate_head: Linear<B>,
}

impl<B: Backend> AfterstateDynamicsNetwork<B> {
    /// Takes latent state + action → produces afterstate
    pub fn forward(
        &self,
        latent_representation: Tensor<B, 2>,
        action: Tensor<B, 2>, // One-hot encoded action
    ) -> Tensor<B, 2> {
        let mut x = Tensor::cat(vec![latent_representation, action], 1);

        for layer in self.dynamics_shared_trunk.iter() {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        let mut after_state = self.afterstate_head.forward(x);
        after_state = normalize_hidden_state(after_state);

        after_state
    }
}

#[derive(Module, Debug)]
pub struct AfterstatePredictionNetwork<B: Backend> {
    prediction_shared_trunk: Vec<Linear<B>>,
    activation: LeakyRelu,
    chance_distribution_head: Linear<B>, // Outputs logits over codebook
    q_value_head: Linear<B>,             // Outputs Q-value (SUPPORT_SIZE)
}

impl<B: Backend> AfterstatePredictionNetwork<B> {
    /// Takes afterstate → produces (chance_distribution_logits, q_value_logits)
    pub fn forward(&self, after_state: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = after_state;

        for layer in self.prediction_shared_trunk.iter() {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        let chance_logits = self.chance_distribution_head.forward(x.clone());
        let q_value_logits = self.q_value_head.forward(x);

        (chance_logits, q_value_logits)
    }
}

#[derive(Module, Debug)]
pub struct EncoderNetwork<B: Backend> {
    encoder_shared_trunk: Vec<Linear<B>>,
    activation: LeakyRelu,
    codebook_head: Linear<B>, // Outputs logits over codebook_size
}

impl<B: Backend> EncoderNetwork<B> {
    /// Takes observation → produces chance_code (one-hot vector)
    pub fn forward(&self, observation: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = observation;

        for layer in self.encoder_shared_trunk.iter() {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        let logits = self.codebook_head.forward(x); // [batch, codebook_size]

        // Apply Gumbel-Softmax / Straight-through estimator
        gumbel_softmax_straight_through(logits)
    }
}
#[derive(Config, Debug)]
pub struct StochasticMuzeroModelConfig {
    hidden_size: usize,
    observation_space: usize,
    action_space: usize,
    cookbook_size: usize,
    latent_space: usize,
}
impl StochasticMuzeroModelConfig {
    pub fn init<B: Backend + AutodiffBackend>(&self, device: &B::Device) -> Network<B> {
        let dynamics_input_size = self.latent_space + self.cookbook_size; // afterstate + chance_code
        let afterstate_dynamics_input_size = self.latent_space + self.action_space; // latent + action

        // Initialize all shared trunks
        let mut dynamics_shared_trunk = Vec::with_capacity(6);
        let mut representation_shared_trunk = Vec::with_capacity(6);
        let mut prediction_shared_trunk = Vec::with_capacity(6);
        let mut afterstate_dynamics_shared_trunk = Vec::with_capacity(6);
        let mut afterstate_prediction_shared_trunk = Vec::with_capacity(6);
        let mut encoder_shared_trunk = Vec::with_capacity(6);

        // First layers (input layers)
        dynamics_shared_trunk
            .push(LinearConfig::new(dynamics_input_size, self.hidden_size).init(device));
        representation_shared_trunk
            .push(LinearConfig::new(self.observation_space, self.hidden_size).init(device));
        prediction_shared_trunk
            .push(LinearConfig::new(self.latent_space, self.hidden_size).init(device));
        afterstate_dynamics_shared_trunk
            .push(LinearConfig::new(afterstate_dynamics_input_size, self.hidden_size).init(device));
        afterstate_prediction_shared_trunk
            .push(LinearConfig::new(self.latent_space, self.hidden_size).init(device));
        encoder_shared_trunk
            .push(LinearConfig::new(self.observation_space, self.hidden_size).init(device));

        for _ in 0..4 {
            dynamics_shared_trunk
                .push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
            representation_shared_trunk
                .push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
            prediction_shared_trunk
                .push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
            afterstate_dynamics_shared_trunk
                .push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
            afterstate_prediction_shared_trunk
                .push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
            encoder_shared_trunk
                .push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
        }

        // Final layer for representation network (outputs latent representation)
        representation_shared_trunk
            .push(LinearConfig::new(self.hidden_size, self.latent_space).init(device));

        let networks = StochasticMuzeroNetworks {
            representation_network: RepresentationNetwork {
                representation_shared_trunk,
                activation: LeakyReluConfig::new().init(),
            },
            prediction_network: PredictionNetwork {
                prediction_shared_trunk,

                activation: LeakyReluConfig::new().init(),
                policy_head: LinearConfig::new(self.hidden_size, self.action_space).init(device),
                value_head: LinearConfig::new(self.hidden_size, STRATEGY_VALUE_SUPPORT_SIZE)
                    .init(device),
            },
            afterstate_dynamics_network: AfterstateDynamicsNetwork {
                dynamics_shared_trunk: afterstate_dynamics_shared_trunk,
                activation: LeakyReluConfig::new().init(),
                afterstate_head: LinearConfig::new(self.hidden_size, self.latent_space)
                    .init(device),
            },
            afterstate_prediction_network: AfterstatePredictionNetwork {
                prediction_shared_trunk: afterstate_prediction_shared_trunk,
                activation: LeakyReluConfig::new().init(),
                chance_distribution_head: LinearConfig::new(self.hidden_size, self.cookbook_size)
                    .init(device),
                q_value_head: LinearConfig::new(self.hidden_size, STRATEGY_VALUE_SUPPORT_SIZE)
                    .init(device),
            },
            dynamics_network: DynamicsNetwork {
                dynamics_shared_trunk,

                activation: LeakyReluConfig::new().init(),
                reward_head: LinearConfig::new(self.hidden_size, STRATEGY_VALUE_SUPPORT_SIZE)
                    .init(device),
                next_state_head: LinearConfig::new(self.hidden_size, self.latent_space)
                    .init(device),
            },
            encoder_network: EncoderNetwork {
                encoder_shared_trunk,
                activation: LeakyReluConfig::new().init(),
                codebook_head: LinearConfig::new(self.hidden_size, self.cookbook_size).init(device),
            },
            action_space: self.action_space,
            codebook_size: self.cookbook_size,
        };

        let optimizer_config = AdamWConfig::new()
            .with_weight_decay(0.0001)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));

        let optimizer = optimizer_config.init();

        Network {
            networks,
            optimizer_config,
            optimizer,
        }
    }
}

// Helper for straight-through estimator
fn gumbel_softmax_straight_through<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 2> {
    let y_soft = softmax(logits.clone(), 1);

    let num_classes = logits.shape().dims[1];

    let index = logits.argmax(1);

    let index: Tensor<B, 1, burn::prelude::Int> = index.squeeze_dim(1);

    let y_hard = Tensor::one_hot(index, num_classes);

    let y_hard = y_hard.float();

    (y_hard - y_soft.clone()).detach() + y_soft
}

pub fn add_dirichlet_noise<B: Backend>(
    action_probabilities: Tensor<B, 1>,
    config: StochasticMuzeroConfig,
) -> Tensor<B, 1> {
    let device = action_probabilities.device();

    let mut rng = rng();

    let dirichlet = StableDirichlet::new(config.root_dirichlet_alpha, config.action_space).unwrap();

    let noise = dirichlet.sample(&mut rng);

    let noise_tensor = Tensor::from_data(noise.as_slice(), &device);

    let action_probabilities = {
        ((1.0 - config.root_dirichlet_alpha as f64) * action_probabilities)
            + (config.root_dirichlet_fraction as f64 * noise_tensor)
    };

    action_probabilities
}

pub fn encode_action<B: Backend>(
    action: usize,
    action_space: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut one_hot = vec![0.0; action_space];
    let one_hot_length = one_hot.len();
    one_hot[action] = 1.0;
    let one_hot = TensorData::new(one_hot, [1, one_hot_length]);
    Tensor::from_data(one_hot, device)
}

fn apply_legal_actions<B: Backend>(
    mut action_probabilities: Tensor<B, 1>,
    valid_actions: Tensor<B, 1>,
) -> Tensor<B, 1> {
    action_probabilities = action_probabilities * valid_actions;

    let action_sum = action_probabilities.clone().sum();

    let action_probabilities = action_probabilities / action_sum;
    action_probabilities
}

pub struct StochasticMuzeroNetworkOutput<B: Backend + AutodiffBackend> {
    pub value_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    pub reward_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    pub latent_representation: Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
    pub policy_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
}
