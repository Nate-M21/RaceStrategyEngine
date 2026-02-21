use std::{path::Path, time::Instant};

use burn::{
    config::Config,
    grad_clipping::GradientClippingConfig,
    module::{AutodiffModule, Module},
    nn::{
        Linear, LinearConfig, Relu,
        loss::{MseLoss, Reduction},
    },
    optim::{
        GradientsParams, Optimizer, Sgd, SgdConfig, adaptor::OptimizerAdaptor,
        decay::WeightDecayConfig, momentum::MomentumConfig,
    },
    prelude::Backend,
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::{
        Tensor, activation::softmax, backend::AutodiffBackend, loss::cross_entropy_with_logits,
    },
};

use crate::{
    algorithms::{
        alpha_zero::{
            alpha_zero::Transition, alpha_zero_config::AlphaZeroConfig, node::AlphaZeroNode,
            replay_buffer::ReplayBuffer,
        },
        helpers::{add_dirichlet_noise, apply_legal_actions},
    },
    traits::actor_critic::ActorCritic,
    utils::{scale_value_1k, unscale_value_1k},
};

// Putting this on top to stress to wrap this in Locks
unsafe impl<B: Backend + AutodiffBackend> Send for AlphaZeroModel<B> {}
unsafe impl<B: Backend + AutodiffBackend> Sync for AlphaZeroModel<B> {}

#[derive(Module, Debug)]
pub struct ModelBackend<B: Backend> {
    shared_trunk: Vec<Linear<B>>,

    activation: Relu,
    policy_head: Linear<B>,
    value_head: Linear<B>,

    observation_space: usize,
    action_space: usize,
}
impl<B: Backend> ModelBackend<B> {
    fn _shared_trunk_forward<const D: usize>(&self, observation: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = observation;
        for layer in self.shared_trunk.iter() {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        x
    }

    fn forward<const D: usize>(&self, observation: Tensor<B, D>) -> (Tensor<B, D>, Tensor<B, D>) {
        let x = self._shared_trunk_forward(observation);

        let policy_logits = self.policy_head.forward(x.clone());

        let value_logit = self.value_head.forward(x);

        (policy_logits, value_logit)
    }

    fn get_device(&self) -> <B as burn::prelude::Backend>::Device {
        self.shared_trunk[0].weight.device()
    }
}

#[derive(Clone)]
pub struct AlphaZeroModel<B: Backend + AutodiffBackend> {
    model: ModelBackend<B>,
    optimizer_config: SgdConfig,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, ModelBackend<B>, B>,
}

impl<B: Backend + AutodiffBackend> AlphaZeroModel<B> {
    pub fn new(model: ModelBackend<B>) -> Self {
        let optimizer_config = SgdConfig::new()
            .with_momentum(Some(
                MomentumConfig::new().with_momentum(0.9).with_nesterov(true),
            ))
            .with_weight_decay(Some(WeightDecayConfig::new(0.0001)))
            .with_gradient_clipping(Some(GradientClippingConfig::Norm(5.0)));

        let optimizer = optimizer_config.init();
        Self {
            model,
            optimizer,
            optimizer_config,
        }
    }

    fn _predict(
        &self,
        observation: &[f32],
    ) -> (
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    ) {
        let device = self.model.get_device();

        let observation = Tensor::from_data(observation, &device);
        let (policy_logits, value_logit) = self.model.valid().forward(observation);

        let epsilon = 1e-8;
        let mut action_probabilities = softmax(policy_logits, 0);
        action_probabilities = action_probabilities + epsilon;
        let priors_sum = action_probabilities.clone().sum();
        let action_probabilities = action_probabilities.div(priors_sum);

        (action_probabilities, value_logit)
    }

    fn calculate_action_and_value(
        &self,
        observation: &[f32],
    ) -> (
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    ) {
        let (action_probabilities, value_logit) = self._predict(observation);
        (action_probabilities, value_logit)
    }

    pub fn display_model(&self) {
        println!("{}\n{:?}", self.model, self.optimizer_config)
    }
}

impl<B: Backend + AutodiffBackend> ActorCritic for AlphaZeroModel<B> {
    type TransitionType = Transition;
    type ObservationType = Vec<f32>;

    fn predict(
        &self,
        observation: &[f32],
        _current_time_step: Option<usize>,
        legal_actions: Option<&[f32]>,
    ) -> Vec<Vec<f32>> {
        let (mut action_probabilities, value_logit) = self._predict(observation);
        if let Some(legal_actions) = legal_actions {
            let device = action_probabilities.device();
            let legal_actions = Tensor::from_data(legal_actions, &device);
            action_probabilities = apply_legal_actions::<B>(action_probabilities, legal_actions);
        }

        let value = unscale_value_1k(value_logit.to_data().into_vec::<f32>().unwrap()[0]);

        vec![
            action_probabilities.to_data().into_vec::<f32>().unwrap(),
            vec![value],
        ]
    }

    fn get_raw_action_and_value_logits<Environment>(
        &self,
        node: &AlphaZeroNode<Environment>,
    ) -> (Vec<f32>, f32)
    where
        Environment: crate::traits::gym::MCTSGymEnvironment<Observation = Self::ObservationType>,
    {
        let device = self.model.get_device();
        let observation = node.current_observation.as_slice();

        let observation: Tensor<<B as AutodiffBackend>::InnerBackend, 1> =
            Tensor::from_data(observation, &device);
        let (policy_logits, value_logit) = self.model.valid().forward(observation);

        let policy_logits = policy_logits
            .to_data()
            .to_vec::<f32>()
            .expect("Failed to convert Tensor to Vec");

        let value = unscale_value_1k(value_logit.to_data().into_vec::<f32>().unwrap()[0]);

        (policy_logits, value)
    }

    fn get_action_probs_and_value<Environment>(
        &self,
        node: &AlphaZeroNode<Environment>,
        apply_dirichlet_noise: bool,
        config: AlphaZeroConfig,
    ) -> (Vec<f32>, f32)
    where
        Environment: crate::traits::gym::MCTSGymEnvironment<Observation = Self::ObservationType>,
    {
        let observation = &node.current_observation;
        let (action_probabilities, value) = self.calculate_action_and_value(&observation);
        let device = action_probabilities.device();
        let action_probabilities = if apply_dirichlet_noise {
            add_dirichlet_noise::<B>(action_probabilities, self.model.action_space, config)
        } else {
            action_probabilities
        };

        let valid_actions = self.get_valid_actions(node);
        let valid_actions = Tensor::from_data(valid_actions, &device);

        let action_probabilities = apply_legal_actions::<B>(action_probabilities, valid_actions);

        let action_probabilities = action_probabilities
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to convert tensor of actions probs to vector");
        let value = value
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to convert tensor of value logitto vector")[0];

        (action_probabilities, unscale_value_1k(value))
    }

    fn get_action_space(&self) -> usize {
        self.model.action_space
    }

    fn get_observation_space(&self) -> usize {
        self.model.observation_space
    }

    fn train_model(
        &mut self,
        replay_buffer: &mut ReplayBuffer<Self::TransitionType>,
        config: &AlphaZeroConfig,
    ) {
        let device = &self.model.get_device();
        let mse_loss = MseLoss::new();
        let mut training_model = self.model.clone();
        let timing = std::time::Instant::now();
        for iteration in 0..config.training_steps {
            let timing_in = Instant::now();
            let sample_of_transitions = replay_buffer.sample_batch();

            let batch_size = sample_of_transitions.len();

            let mut observations = Vec::with_capacity(batch_size);
            let mut policy_targets = Vec::with_capacity(batch_size);
            let mut value_targets = Vec::with_capacity(batch_size);

            for transition in sample_of_transitions {
                let observation_vec: Vec<f32> = transition.observation;
                let policy_target_vec: Vec<f32> = transition.action_probabilities;

                let value_target_vec: Vec<f32> = vec![scale_value_1k(transition.total_reward)];

                let observation_tensor: Tensor<B, 1> =
                    Tensor::from_data(observation_vec.as_slice(), device);
                let policy_target_tensor: Tensor<B, 1> =
                    Tensor::from_data(policy_target_vec.as_slice(), device);
                let value_target_tensor: Tensor<B, 1> =
                    Tensor::from_data(value_target_vec.as_slice(), device);

                observations.push(observation_tensor);
                policy_targets.push(policy_target_tensor);
                value_targets.push(value_target_tensor);
            }

            let observations_batch: Tensor<B, 2> = Tensor::stack(observations, 0);
            let policy_targets_batch: Tensor<B, 2> = Tensor::stack(policy_targets, 0);
            let value_targets_batch: Tensor<B, 2> = Tensor::stack(value_targets, 0);

            let (policy_predictions, value_predictions) =
                training_model.forward(observations_batch);

            let policy_loss = cross_entropy_with_logits(policy_predictions, policy_targets_batch);

            let value_loss =
                mse_loss.forward(value_predictions, value_targets_batch, Reduction::Mean);

            let total_loss = policy_loss.clone() + value_loss.clone();
            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &training_model);
            println!(
                "Training iteration {}/{}: value loss = {}, policy loss = {}, total loss = {:.6} | Took {}s",
                iteration,
                config.training_steps,
                value_loss.to_data().to_vec::<f32>().unwrap()[0],
                policy_loss.to_data().to_vec::<f32>().unwrap()[0],
                total_loss.to_data().to_vec::<f32>().unwrap()[0],
                timing_in.elapsed().as_secs_f32()
            );
            training_model =
                self.optimizer
                    .step(config.learning_rate as f64, training_model, grads);
        }
        println!(
            "Took {} to complete training:",
            timing.elapsed().as_secs_f32()
        );

        self.model = training_model;
    }

    fn save_model(&self, path: &Path) {
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        self.model
            .clone()
            .save_file(path, &recorder)
            .expect("Failed to save file");
    }

    fn load_model(&mut self, path: &Path) {
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();

        self.model = self
            .model
            .clone()
            .load_file(path, &recorder, &self.model.get_device())
            .expect("Failed to load file");
    }
}

#[derive(Config, Debug)]
pub struct AlphaZeroModelConfig {
    hidden_size: usize,
    observation_space: usize,
    action_space: usize,
}

impl AlphaZeroModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ModelBackend<B> {
        let mut shared_trunk = Vec::with_capacity(5);
        shared_trunk.push(LinearConfig::new(self.observation_space, self.hidden_size).init(device));
        for _ in 0..4 {
            shared_trunk.push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
        }
        ModelBackend {
            shared_trunk,
            activation: Relu,
            policy_head: LinearConfig::new(self.hidden_size, self.action_space).init(device),
            value_head: LinearConfig::new(self.hidden_size, 1).init(device),
            observation_space: self.observation_space,
            action_space: self.action_space,
        }
    }
}
