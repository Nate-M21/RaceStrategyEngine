use burn::{
    Tensor,
    config::Config,
    grad_clipping::GradientClippingConfig,
    module::{AutodiffModule, Module},
    nn::{LeakyRelu, LeakyReluConfig, Linear, LinearConfig},
    optim::{
        GradientsParams, Optimizer, Sgd, SgdConfig, adaptor::OptimizerAdaptor,
        decay::WeightDecayConfig, momentum::MomentumConfig,
    },
    prelude::Backend,
    tensor::{
        TensorData, activation::softmax, backend::AutodiffBackend, loss::cross_entropy_with_logits,
    },
};
use rand::rng;
use rand_distr::Distribution;

use crate::algorithms::{
    alpha_zero::dirichlet::StableDirichlet,
    helpers::{
        STRATEGY_VALUE_SUPPORT_SIZE, normalize_hidden_state, scalar_to_support_batch,
        support_to_scalar,
    },
    muzero_old::{muzero_config::MuzeroConfig, replay_buffer::MuzeroReplayBuffer},
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

unsafe impl<B: Backend + AutodiffBackend> Send for MuzeroModel<B> {}
unsafe impl<B: Backend + AutodiffBackend> Sync for MuzeroModel<B> {}

pub struct MuzeroModel<B: Backend + AutodiffBackend> {
    networks: MuzeroNetworks<B>,
    // optimizer_config: AdamWConfig, // Just for viewing purposes
    // optimizer: OptimizerAdaptor<AdamW, MuzeroNetworks<B>, B>,
    optimizer_config: SgdConfig, // Just for viewing purposes
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, MuzeroNetworks<B>, B>,
    // learning_rate: lr_scheduler::exponential::ExponentialLrScheduler,
    action_space: usize,
}

impl<B: Backend> MuzeroNetworks<B> {
    fn get_device(&self) -> B::Device {
        self.representation_network.representation_shared_trunk[0]
            .weight
            .device()
    }
}

impl<B: Backend + AutodiffBackend> MuzeroModel<B> {
    fn initial_inference(&self, observation: &[f32]) -> MuzeroNetworkOutput<B> {
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
        MuzeroNetworkOutput {
            value_logits,
            reward_logits,
            latent_representation,
            policy_logits,
        }
    }

    fn recurrent_inference(
        &self,
        latent_representation: Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
        action: usize,
    ) -> MuzeroNetworkOutput<B> {
        let device = self.networks.dynamics_network.dynamics_shared_trunk[0]
            .weight
            .device();

        let action_tensor = encode_action(action, self.action_space, &device);
        let (reward_logits, next_latent_representation) = self
            .networks
            .dynamics_network
            .valid()
            .forward(latent_representation, action_tensor);

        let (policy_logits, value_logits) = self
            .networks
            .prediction_network
            .valid()
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

    pub fn get_initial_action_probs_and_value(
        &self,
        observation: &[f32],
        legal_actions: &[f32],
        apply_dirichlet_noise: bool,
        config: MuzeroConfig,
    ) -> MuzeroModelOutput<B> {
        let MuzeroNetworkOutput {
            value_logits,
            reward_logits,
            latent_representation,
            policy_logits,
        } = self.initial_inference(observation);

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
        MuzeroModelOutput {
            value,
            reward,
            latent_representation,
            action_probabilities,
        }
    }

    pub fn get_action_probs_and_value(
        &self,
        latent_representation: Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
        action: usize,
    ) -> MuzeroModelOutput<B> {
        let MuzeroNetworkOutput {
            value_logits,
            reward_logits,
            latent_representation,
            policy_logits,
        } = self.recurrent_inference(latent_representation, action);

        let epsilon = 1e-8;
        let mut action_probabilities = softmax(policy_logits, 0);
        action_probabilities = action_probabilities + epsilon;
        let priors_sum = action_probabilities.clone().sum();
        let action_probabilities = action_probabilities.div(priors_sum);

        let value = support_to_scalar(value_logits);
        let reward = support_to_scalar(reward_logits);
        let action_probabilities = action_probabilities.to_data().to_vec::<f32>().unwrap();

        MuzeroModelOutput {
            value,
            reward,
            latent_representation,
            action_probabilities,
        }
    }
    pub fn train_model(&mut self, replay_buffer: &mut MuzeroReplayBuffer, config: &MuzeroConfig) {
        let device = &self.networks.get_device();

        let mut training_networks = self.networks.clone();

        let timing = std::time::Instant::now();

        for iteration in 1..=config.training_steps {
            let timing_in = std::time::Instant::now();
            let batch_of_sequences = replay_buffer.sample_batch(config.num_unroll_steps as usize);

            let capacity = batch_of_sequences.len();
            let mut initial_observations = Vec::with_capacity(capacity);
            // let mut initial_latent_representations = Vec::with_capacity(capacity);
            let mut initial_policy_targets = Vec::with_capacity(capacity);
            let mut initial_value_targets = Vec::with_capacity(capacity);

            for sequence in batch_of_sequences.iter() {
                // At each sequence i am getting the starting observation, hidden state, value and policy target
                let initial_sequence_observation = &sequence[0].observation;
                let initial_policy_target = &sequence[0].action_probabilities;
                let initial_value_target = sequence[0].value_target;

                let tensor_initial_sequence_observation: Tensor<B, 1> =
                    Tensor::from_data(initial_sequence_observation.as_slice(), device);
                // let tensor_initial_latent_representation: Tensor<B, 1> = self.networks.representation_network.forward(tensor_initial_sequence_observation);

                let tensor_initial_policy_target: Tensor<B, 1> =
                    Tensor::from_data(initial_policy_target.as_slice(), device);

                initial_observations.push(tensor_initial_sequence_observation);
                // initial_latent_representations.push(tensor_initial_latent_representation);
                initial_policy_targets.push(tensor_initial_policy_target);
                initial_value_targets.push(initial_value_target);
            }

            let initial_observations_batch: Tensor<B, 2> = Tensor::stack(initial_observations, 0);
            let initial_latent_representations_batch: Tensor<B, 2> = training_networks
                .representation_network
                .forward(initial_observations_batch);

            let initial_value_targets_batch: Tensor<B, 2> =
                scalar_to_support_batch(&initial_value_targets, device);

            let initial_policy_targets_batch: Tensor<B, 2> =
                Tensor::stack(initial_policy_targets, 0);

            let (initial_policy_prediction, initial_value_prediction) = training_networks
                .prediction_network
                .forward(initial_latent_representations_batch.clone());

            let initial_policy_loss =
                cross_entropy_with_logits(initial_policy_prediction, initial_policy_targets_batch);
            let initial_value_loss =
                cross_entropy_with_logits(initial_value_prediction, initial_value_targets_batch);

            let mut total_policy_loss = initial_policy_loss.clone();
            let mut total_value_loss = initial_value_loss.clone();
            let mut total_reward_loss = Tensor::from_data([0.0], device);

            let mut total_loss = initial_policy_loss + initial_value_loss;

            let gradient_scale = 1.0 / config.num_unroll_steps as f32;

            let batch_size = capacity;

            let mut hidden_state = initial_latent_representations_batch;
            for k in 0..config.num_unroll_steps as usize {
                let next_step_index = k + 1;

                let mut actions_k = Vec::with_capacity(batch_size);
                let mut policy_targets_k: Vec<Tensor<B, 1>> = Vec::with_capacity(batch_size);
                let mut value_targets_k = Vec::with_capacity(batch_size);
                let mut reward_targets_k = Vec::with_capacity(batch_size);

                for sequence in batch_of_sequences.iter() {
                    // Safety check: Ensure sequence is long enough
                    if next_step_index < sequence.len() {
                        let prev_action = sequence[k].action; // Action taken at previous step
                        let target_policy = &sequence[next_step_index].action_probabilities;
                        let target_value = sequence[next_step_index].value_target;
                        let target_reward = sequence[next_step_index].reward;

                        // Encode Action (One Hot)

                        let action_tensor = encode_action(prev_action, self.action_space, device);

                        let action_tensor: Tensor<B, 1> = action_tensor.squeeze_dim(0);

                        actions_k.push(action_tensor);
                        policy_targets_k.push(Tensor::from_data(target_policy.as_slice(), device));
                        value_targets_k.push(target_value);
                        reward_targets_k.push(target_reward);
                    } else {
                        // EDGE CASE: If a game ended, we usually pad with zeros or mask this loss.
                        // For simplicity here, I'm assuming sample_batch returns full length sequences.
                        // If not, you need to handle masking here.
                        // This should not hppen because in replay buffer i ensure it cant happen,
                        // but just in case to spot a bug
                        panic!("Sequence too short for unroll steps!");
                    }
                }

                // Stack batch for step K
                let action_batch = Tensor::stack(actions_k, 0); // [Batch, Action_Dim]
                let target_policy_batch = Tensor::stack(policy_targets_k, 0);
                let target_value_batch = scalar_to_support_batch(&value_targets_k, device);
                let target_reward_batch = scalar_to_support_batch(&reward_targets_k, device);

                // Dynamics Forward: g(s_{k-1}, a_k) -> r_k, s_k
                let (pred_reward, next_hidden_state) = training_networks.dynamics_network.forward(
                    hidden_state, // The hidden state from the previous iteration
                    action_batch,
                );

                // Prediction Forward: f(s_k) -> p_k, v_k
                let (pred_policy, pred_value) = training_networks
                    .prediction_network
                    .forward(next_hidden_state.clone());

                let next_hidden_state = scale_gradient(next_hidden_state, 0.5);

                // Accumulate Loss
                let l_policy = cross_entropy_with_logits(pred_policy, target_policy_batch);
                let l_value = cross_entropy_with_logits(pred_value, target_value_batch);
                let l_reward = cross_entropy_with_logits(pred_reward, target_reward_batch);

                total_policy_loss = total_policy_loss + l_policy.clone();
                total_value_loss = total_value_loss + l_value.clone();
                total_reward_loss = total_reward_loss + l_reward.clone();

                // Add scaled loss to total
                total_loss =
                    total_loss + (l_policy + l_value + l_reward).mul_scalar(gradient_scale);

                // Update hidden state for next loop iteration
                hidden_state = next_hidden_state;
            }

            let policy_loss_print = total_policy_loss.to_data().to_vec::<f32>().unwrap()[0];
            let value_loss_print = total_value_loss.to_data().to_vec::<f32>().unwrap()[0];
            let reward_loss_print = total_reward_loss.to_data().to_vec::<f32>().unwrap()[0];
            let total_loss_print = total_loss.to_data().to_vec::<f32>().unwrap()[0];

            let grads = total_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &training_networks);

            println!(
                "Training iteration {}/{}: policy loss = {:.6}, value loss = {:.6}, reward loss = {:.6}, total loss = {:.6} | Took {:.4}s",
                iteration,
                config.training_steps,
                policy_loss_print,
                value_loss_print,
                reward_loss_print,
                total_loss_print,
                timing_in.elapsed().as_secs_f32()
            );

            training_networks = self.optimizer.step(
                config.learning_rate_init as f64,
                training_networks,
                grads_params,
            )
        }

        println!(
            "Training completed in {:.2}s",
            timing.elapsed().as_secs_f32()
        );

        self.networks = training_networks;
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

fn scale_gradient<B: AutodiffBackend, const D: usize>(
    tensor: Tensor<B, D>,
    scale: f32,
) -> Tensor<B, D> {
    // This works because:
    // Forward: scale*x + (1-scale)*x = x
    // Backward: only first term gets gradients, scaled by 'scale'
    tensor.clone().mul_scalar(scale) + tensor.detach().mul_scalar(1.0 - scale)
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

        for _ in 0..5 {
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
            action_space: self.action_space,
            representation_network: RepresentationNetwork {
                representation_shared_trunk,
                activation: LeakyReluConfig::new().init(),
            },

            dynamics_network: DynamicsNetwork {
                dynamics_shared_trunk,

                activation: LeakyReluConfig::new().init(),
                reward_head: LinearConfig::new(self.hidden_size, STRATEGY_VALUE_SUPPORT_SIZE)
                    .init(device),
                next_state_head: LinearConfig::new(self.hidden_size, self.latent_representation)
                    .init(device),
            },

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
            .with_weight_decay(Some(WeightDecayConfig::new(0.001)))
            .with_gradient_clipping(Some(GradientClippingConfig::Norm(5.0)));

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

// TODO agin these two Outputs are based on MLP architecture, change to Transformers or Images these have to change
// Need to think of a better method
pub struct MuzeroModelOutput<B: Backend + AutodiffBackend> {
    pub value: f32,
    pub reward: f32,
    pub latent_representation: Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
    pub action_probabilities: Vec<f32>,
}

struct MuzeroNetworkOutput<B: Backend + AutodiffBackend> {
    value_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    reward_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    latent_representation: Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
    policy_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
}
