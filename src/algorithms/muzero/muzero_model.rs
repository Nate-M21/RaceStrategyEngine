use burn::{
    Tensor,
    config::Config,
    grad_clipping::GradientClippingConfig,
    module::{AutodiffModule, Module},
    nn::{LeakyRelu, LeakyReluConfig, Linear, LinearConfig},
    optim::{
        Sgd, SgdConfig, adaptor::OptimizerAdaptor, decay::WeightDecayConfig,
        momentum::MomentumConfig,
    },
    prelude::Backend,
    tensor::{TensorData, activation::softmax, backend::AutodiffBackend},
};
use rand::rng;
use rand_distr::Distribution;

use crate::algorithms::{
    alpha_zero::dirichlet::StableDirichlet,
    helpers::{STRATEGY_VALUE_SUPPORT_SIZE, normalize_hidden_state, support_to_scalar},
    muzero::muzero_config::MuzeroConfig,
};

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
        latent_representation: Tensor<B, 2>,
        action: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = Tensor::cat(vec![latent_representation, action], 1);

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
pub struct MuzeroNetworks<B: Backend> {
    pub representation_network: RepresentationNetwork<B>, // obs → hidden_state
    pub dynamics_network: DynamicsNetwork<B>, // (hidden_state, action) → (reward, next_hidden_state)
    pub prediction_network: PredictionNetwork<B>,

    action_space: usize,
}

impl<B: Backend> MuzeroNetworks<B> {
    pub fn get_device(&self) -> B::Device {
        self.representation_network.representation_shared_trunk[0]
            .weight
            .device()
    }
}

unsafe impl<B: Backend + AutodiffBackend> Send for MuzeroModel<B> {}
unsafe impl<B: Backend + AutodiffBackend> Sync for MuzeroModel<B> {}

#[derive(Clone)]
pub struct MuzeroModel<B: Backend + AutodiffBackend> {
    pub networks: MuzeroNetworks<B>,
    // optimizer_config: AdamWConfig, // Just for viewing purposes
    // pub optimizer: OptimizerAdaptor<burn::optim::AdamW, MuzeroNetworks<B>, B>,
    optimizer_config: SgdConfig, // Just for viewing purposes
    pub optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, MuzeroNetworks<B>, B>,

    // learning_rate: lr_scheduler::exponential::ExponentialLrScheduler,
    action_space: usize,
}

impl<B: Backend + AutodiffBackend> MuzeroModel<B> {
    pub fn initial_inference(&self, observation: &[f32]) -> MuzeroNetworkOutput<B> {
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
            // .valid()
            .forward(observation);

        let (policy_logits, value_logits) = self
            .networks
            .prediction_network
            // .valid()
            .forward(latent_representation.clone());
        let reward_logits = Tensor::from_data([0], &device);

        // TODO this needs to be more robust but works for me now as ill be dealing with Tensor B , 2 all the time
        let value_logits = value_logits.squeeze_dim(0);
        let policy_logits = policy_logits.squeeze_dim(0);
        MuzeroNetworkOutput {
            value_logits,
            reward_logits,
            latent_representation,
            policy_logits,
        }
    }

    pub fn recurrent_inference(
        &self,
        latent_representation: Tensor<B, 2>,
        action: usize,
    ) -> MuzeroNetworkOutput<B> {
        let device = self.networks.dynamics_network.dynamics_shared_trunk[0]
            .weight
            .device();

        let action_tensor = encode_action(action, self.action_space, &device);
        let (reward_logits, next_latent_representation) = self
            .networks
            .dynamics_network
            // .valid()
            .forward(latent_representation, action_tensor);

        let (policy_logits, value_logits) = self
            .networks
            .prediction_network
            // .valid()
            .forward(next_latent_representation.clone());

        let value_logits = value_logits.squeeze_dim(0);
        let reward_logits = reward_logits.squeeze_dim(0);
        let policy_logits = policy_logits.squeeze_dim(0);

        MuzeroNetworkOutput {
            value_logits,
            reward_logits,
            latent_representation: next_latent_representation,
            policy_logits,
        }
    }

    fn valid_initial_inference(&self, observation: &[f32]) -> ValidMuzeroNetworkOutput<B> {
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
        ValidMuzeroNetworkOutput {
            value_logits,
            reward_logits,
            latent_representation,
            policy_logits,
        }
    }

    fn valid_recurrent_inference(
        &self,
        latent_representation: Tensor<B, 2>,
        action: usize,
    ) -> ValidMuzeroNetworkOutput<B> {
        let device = self.networks.dynamics_network.dynamics_shared_trunk[0]
            .weight
            .device();

        let action_tensor = encode_action(action, self.action_space, &device);
        let (reward_logits, next_latent_representation) = self
            .networks
            .dynamics_network
            .valid()
            .forward(latent_representation.inner(), action_tensor);

        let (policy_logits, value_logits) = self
            .networks
            .prediction_network
            .valid()
            .forward(next_latent_representation.clone());

        let value_logits = value_logits.squeeze_dim(0);
        let reward_logits = reward_logits.squeeze_dim(0);
        let policy_logits = policy_logits.squeeze_dim(0);

        ValidMuzeroNetworkOutput {
            value_logits,
            reward_logits,
            latent_representation: next_latent_representation,
            policy_logits,
        }
    }

    pub fn get_initial_action_probs_and_value(
        &self,
        observation: &[f32],
        legal_actions: &[f32],
        apply_dirichlet_noise: bool,
        config: MuzeroConfig,
    ) -> MuzeroModelOutput<B> {
        let ValidMuzeroNetworkOutput {
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
        let reward = support_to_scalar(reward_logits);

        if apply_dirichlet_noise {
            action_probabilities = add_dirichlet_noise(action_probabilities, config);
        }
        let valid_actions = Tensor::from_data(legal_actions, &action_probabilities.device());
        action_probabilities = apply_legal_actions(action_probabilities, valid_actions);
        let action_probabilities = action_probabilities.to_data().to_vec::<f32>().unwrap();
        let latent_representation = latent_representation;
        MuzeroModelOutput {
            value,
            reward,
            latent_representation,
            action_probabilities,
        }
    }

    pub fn get_action_probs_and_value(
        &self,
        latent_representation: Tensor<B, 2>,
        action: usize,
    ) -> MuzeroModelOutput<B> {
        let ValidMuzeroNetworkOutput {
            value_logits,
            reward_logits,
            latent_representation,
            policy_logits,
        } = self.valid_recurrent_inference(latent_representation, action);

        let epsilon = 1e-8;
        let mut action_probabilities = softmax(policy_logits, 0);
        action_probabilities = action_probabilities + epsilon;
        let priors_sum = action_probabilities.clone().sum();
        let action_probabilities = action_probabilities.div(priors_sum);

        // let value = unscale_value_1k(value_logits.to_data().to_vec::<f32>().unwrap()[0]);
        // let reward = unscale_value_1k(reward_logits.to_data().to_vec::<f32>().unwrap()[0]);

        let value = support_to_scalar(value_logits);
        let reward = support_to_scalar(reward_logits);
        let action_probabilities = action_probabilities.to_data().to_vec::<f32>().unwrap();

        let latent_representation = latent_representation;

        MuzeroModelOutput {
            value,
            reward,
            latent_representation,
            action_probabilities,
        }
    }

    pub fn display_model(&self) {
        println!(
            "{}\n\nOptimizer:\n{:?}",
            self.networks, self.optimizer_config
        )
    }
}

pub fn add_dirichlet_noise<B: Backend>(
    action_probabilities: Tensor<B, 1>,
    config: MuzeroConfig,
) -> Tensor<B, 1> {
    let device = action_probabilities.device();

    let mut rng = rng();

    let dirichlet = StableDirichlet::new(config.dirichlet_alpha, config.action_space).unwrap();

    let noise = dirichlet.sample(&mut rng);

    let noise_tensor = Tensor::from_data(noise.as_slice(), &device);

    let action_probabilities = {
        ((1.0 - config.dirichlet_epsilon as f64) * action_probabilities)
            + (config.dirichlet_epsilon as f64 * noise_tensor)
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

#[derive(Config, Debug)]
pub struct MuzeroModelConfig {
    hidden_size: usize,
    observation_space: usize,
    action_space: usize,
    latent_representation: usize, // The key MuZero parameter
}

impl MuzeroModelConfig {
    pub fn init<B: Backend + AutodiffBackend>(&self, device: &B::Device) -> MuzeroModel<B> {
        let dynamics_input_size = self.latent_representation + self.action_space;

        let mut dynamics_shared_trunk = Vec::with_capacity(6);
        let mut representation_shared_trunk = Vec::with_capacity(6);
        let mut prediction_shared_trunk = Vec::with_capacity(6);

        dynamics_shared_trunk
            .push(LinearConfig::new(dynamics_input_size, self.hidden_size).init(device));
        representation_shared_trunk
            .push(LinearConfig::new(self.observation_space, self.hidden_size).init(device));
        prediction_shared_trunk
            .push(LinearConfig::new(self.latent_representation, self.hidden_size).init(device));

        for _ in 0..4 {
            dynamics_shared_trunk
                .push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
            representation_shared_trunk
                .push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
            prediction_shared_trunk
                .push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
        }

        representation_shared_trunk
            .push(LinearConfig::new(self.hidden_size, self.latent_representation).init(device));

        let networks = MuzeroNetworks {
            // Representation Network (h: Observation -> Latent)
            action_space: self.action_space,
            representation_network: RepresentationNetwork {
                representation_shared_trunk,
                activation: LeakyReluConfig::new().init(),
            },

            //  Dynamics Network (g: Latent + Action -> Next Latent & Reward)
            dynamics_network: DynamicsNetwork {
                dynamics_shared_trunk,

                activation: LeakyReluConfig::new().init(),
                reward_head: LinearConfig::new(self.hidden_size, STRATEGY_VALUE_SUPPORT_SIZE)
                    .init(device),
                next_state_head: LinearConfig::new(self.hidden_size, self.latent_representation)
                    .init(device),
            },

            //  Prediction Network (f: Latent -> Policy & Value)
            prediction_network: PredictionNetwork {
                prediction_shared_trunk,

                activation: LeakyReluConfig::new().init(),
                policy_head: LinearConfig::new(self.hidden_size, self.action_space).init(device),
                value_head: LinearConfig::new(self.hidden_size, STRATEGY_VALUE_SUPPORT_SIZE)
                    .init(device),
            },
        };

        let optimizer_config = SgdConfig::new()
            .with_momentum(Some(
                MomentumConfig::new().with_momentum(0.9).with_nesterov(true),
            ))
            .with_weight_decay(Some(WeightDecayConfig::new(0.0001)))
            .with_gradient_clipping(Some(GradientClippingConfig::Norm(1.0)));

        // let optimizer_config = AdamWConfig::new()
        //     .with_weight_decay(0.0001)
        //     .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));
        let optimizer = optimizer_config.init();
        // learning_rate = lr_scheduler::exponential::ExponentialLrSchedulerConfig::new(initial_lr, gamma)

        MuzeroModel {
            networks,
            optimizer,
            optimizer_config,
            action_space: self.action_space,
        }
    }
}

// TODO these two Outputs are based on MLP architecture, change to Transformers or Images these have to change
// Need to think of a better method
pub struct MuzeroModelOutput<B: Backend + AutodiffBackend> {
    pub value: f32,
    pub reward: f32,
    pub latent_representation: Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
    pub action_probabilities: Vec<f32>,
}

pub struct MuzeroNetworkOutput<B: Backend + AutodiffBackend> {
    pub value_logits: Tensor<B, 1>,
    pub reward_logits: Tensor<B, 1>,
    pub latent_representation: Tensor<B, 2>,
    pub policy_logits: Tensor<B, 1>,
}

pub struct ValidMuzeroNetworkOutput<B: Backend + AutodiffBackend> {
    pub value_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    pub reward_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    pub latent_representation: Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
    pub policy_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
}
