use std::{path::Path, time::Instant};

use burn::{
    config::Config,
    grad_clipping::GradientClippingConfig,
    lr_scheduler::{
        LrScheduler,
        linear::{LinearLrScheduler, LinearLrSchedulerConfig},
    },
    module::{AutodiffModule, Module},
    nn::{
        Linear, LinearConfig,
        attention::{generate_autoregressive_mask, generate_padding_mask},
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    optim::{
        GradientsParams, Optimizer, Sgd, SgdConfig, adaptor::OptimizerAdaptor,
        decay::WeightDecayConfig, momentum::MomentumConfig,
    },
    prelude::Backend,
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::{
        Bool, Device, Int, Tensor, TensorData, activation::softmax, backend::AutodiffBackend,
        loss::cross_entropy_with_logits,
    },
};

use crate::{
    algorithms::{
        alpha_zero::{alpha_zero_config::AlphaZeroConfig, node::AlphaZeroNode},
        helpers::{
            STRATEGY_VALUE_SUPPORT_SIZE, add_dirichlet_noise, apply_legal_actions,
            scalar_to_support_batch, support_to_scalar,
        },
        rob::rob::RobTransition,
    },
    traits::actor_critic::ActorCritic,
};

// Putting this on top to stress to wrap this in Locks
unsafe impl<B: Backend + AutodiffBackend> Send for TronModel<B> {}
unsafe impl<B: Backend + AutodiffBackend> Sync for TronModel<B> {}

#[derive(Module, Debug)]
pub struct ModelBackend<B: Backend> {
    linear1: Linear<B>,

    transformer: TransformerEncoder<B>,

    policy_head: Linear<B>,
    value_head: Linear<B>,
    position_head: Linear<B>,
    observation_space: usize,
    action_space: usize,
    num_drivers: usize,
    max_sequence_length: usize,
    features_per_sequence: usize,

    positional_encoding: Tensor<B, 3>,
}

impl<B: Backend> ModelBackend<B> {
    fn forward(
        &self,
        observatiom: Tensor<B, 3>,
        mask_pad: Option<Tensor<B, 2, Bool>>,
        mask_attn: Option<Tensor<B, 3, Bool>>,
        timesteps_indices: Tensor<B, 1, Int>,
    ) -> TronNetworkOuput<B> {
        let x = self.linear1.forward(observatiom);
        let [batch_size, _seq_len, _d_model] = x.dims();

        // repeatnig the positional encoding to match batch size, so each lap in the sequence gets
        let positional_encoding = self.positional_encoding.clone().repeat_dim(0, batch_size);

        let tensor = x + positional_encoding;

        let x = create_transformer_input(tensor, mask_pad, mask_attn);

        let x = self.transformer.forward(x);

        let [batch_size, _seq_len, hidden_dim] = x.dims();
        // let last_token_slice = x.slice([0..batch_size, (seq_len - 1)..seq_len, 0..hidden_dim]);

        // let last_token = last_token_slice.squeeze_dim(1);

        let indices = timesteps_indices.reshape([batch_size, 1]);

        // TODO try search burn docs to see if there is better an dfaster way to do this
        // To try get the whole vector at the specific index for every batch item
        // Basically aim is get the valid index of the current time step ie the last token
        let gather_indices = indices.unsqueeze_dim::<3>(2).repeat_dim(2, hidden_dim);

        let gathered_token = x.gather(1, gather_indices);

        let last_token = gathered_token.squeeze_dim(1);

        let policy_logits = self.policy_head.forward(last_token.clone());
        let value_logits = self.value_head.forward(last_token.clone());
        let position_logits = self.position_head.forward(last_token.clone());

        TronNetworkOuput {
            policy_logits,
            value_logits,
            position_logits,
            last_token,
        }
    }

    fn get_device(&self) -> Device<B> {
        self.linear1.weight.device()
    }
}
// T.R.O.N - Transformer-based Race Optimisation Network
#[derive(Clone)]

pub struct TronModel<B: Backend + AutodiffBackend> {
    model: ModelBackend<B>,

    training_device: Device<B>,
    learning_rate_schedule: ThreePhaseScheduler,

    // optimizer: OptimizerAdaptor<AdamW, ModelBackend<B>, B>,
    // optimizer_config: AdamWConfig,
    optimizer_config: SgdConfig,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, ModelBackend<B>, B>,
}

impl<B: Backend + AutodiffBackend> TronModel<B> {
    pub fn new(model: ModelBackend<B>, training_device: Device<B>) -> Self {
        // let optimizer_config =
        //     AdamWConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));

        let optimizer_config = SgdConfig::new()
            .with_momentum(Some(
                MomentumConfig::new().with_momentum(0.9).with_nesterov(true),
            ))
            .with_weight_decay(Some(WeightDecayConfig::new(0.0001)))
            .with_gradient_clipping(Some(GradientClippingConfig::Norm(5.0)));

        let optimizer = optimizer_config.init::<B, ModelBackend<B>>();

        let learning_rate_schedule = ThreePhaseScheduler::new(
            200,     // warmup: 1k steps to reach 0.02
            5000,    // stable: 5k steps at 0.02
            14000,   // decay: 14k steps down to 0.0002
            1e-7,    // initial_lr: 0.000001
            0.0001,  // peak_lr: 0.02
            0.00002, // final_lr: 0.0002
        );

        // let learning_rate_schedule: NoamLrScheduler = NoamLrSchedulerConfig::new(0.016).with_warmup_steps(200).with_model_size(256).init().unwrap();

        Self {
            model,
            optimizer,
            training_device,
            learning_rate_schedule,
            optimizer_config,
        }
    }
    fn _predict(
        &self,
        observation: &[f32],
        current_step: usize,
    ) -> (
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    ) {
        let (policy_logits, value_logit, position_logits, _last_token) =
            self.get_forward_outputs(observation, current_step);

        let position_probabilities = softmax(position_logits, 0);

        let epsilon = 1e-8;
        let mut action_probabilities = softmax(policy_logits, 0);
        action_probabilities = action_probabilities + epsilon;
        let priors_sum = action_probabilities.clone().sum();
        let action_probabilities = action_probabilities.div(priors_sum);

        (action_probabilities, value_logit, position_probabilities)
    }

    fn get_forward_outputs(
        &self,
        observation: &[f32],
        current_step: usize,
    ) -> (
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    ) {
        let device = self.model.get_device();

        let batch_size = 1;
        let seq_length = self.model.max_sequence_length;
        let max_seq_length = Some(seq_length);
        let pad_token = 0;

        let mut tokens = (1..=current_step).collect::<Vec<usize>>();
        tokens.resize(seq_length, 0);
        let tokens_list = vec![tokens];

        let features = self.model.features_per_sequence;
        let mut observation = observation.to_vec();

        let target_index = (current_step as i64) - 1;

        // Create the tensor: [target_index]
        let timesteps_indices = Tensor::from_data([target_index], &device);

        observation.resize(seq_length * features, 0.0);

        let observation = TensorData::new(observation, [batch_size, seq_length, features]);
        let observation: Tensor<<B as AutodiffBackend>::InnerBackend, 3> =
            Tensor::from_data(observation, &device);

        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, &device);
        let mask_pad = generate_padding_mask(pad_token, tokens_list, max_seq_length, &device).mask;
        let TronNetworkOuput {
            policy_logits,
            value_logits,
            last_token,
            position_logits,
        } = self.model.valid().forward(
            observation,
            Some(mask_pad),
            Some(mask_attn),
            timesteps_indices,
        );

        let policy_logits = policy_logits.squeeze_dim(0);
        let value_logit = value_logits.squeeze_dim(0);
        let position_logits = position_logits.squeeze_dim(0);
        let last_token = last_token.squeeze_dim(0);
        (policy_logits, value_logit, position_logits, last_token)
    }

    fn calculate_action_and_value(
        &self,
        observation: &[f32],
        current_step: usize,
    ) -> (
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    ) {
        let (action_probabilities, value_logit, _position_probabilities) =
            self._predict(observation, current_step);
        (action_probabilities, value_logit)
    }

    pub fn display_model(&self) {
        println!(
            "{}\n\n{:?}\nInference Device: {:?}\nTraining Device: {:?}\n\nLearning Rate Schedule{:?}",
            self.model,
            self.optimizer_config,
            self.model.get_device(),
            self.training_device,
            self.learning_rate_schedule
        )
    }
}

impl<B: Backend + AutodiffBackend> ActorCritic for TronModel<B> {
    type TransitionType = RobTransition;
    type ObservationType = Vec<f32>;

    fn predict(
        &self,
        observation: &[f32],
        current_time_step: Option<usize>,
        legal_actions: Option<&[f32]>,
    ) -> Vec<Vec<f32>> {
        let current_time_step =
            current_time_step.expect("Tranformer variant requires a time step value");
        let (mut action_probabilities, value_logits, position_probabilities) =
            self._predict(observation, current_time_step);

        if let Some(legal_actions) = legal_actions {
            let device = action_probabilities.device();
            let legal_actions = Tensor::from_data(legal_actions, &device);
            action_probabilities = apply_legal_actions::<B>(action_probabilities, legal_actions);
        }

        let value = support_to_scalar(value_logits);

        vec![
            action_probabilities.to_data().into_vec::<f32>().unwrap(),
            vec![value],
            position_probabilities.to_data().into_vec::<f32>().unwrap(),
        ]
    }

    fn get_raw_action_and_value_logits<Environment>(
        &self,
        node: &AlphaZeroNode<Environment>,
    ) -> (Vec<f32>, f32)
    where
        Environment: crate::traits::gym::MCTSGymEnvironment<Observation = Self::ObservationType>,
    {
        let observation = &node.current_observation;
        let current_step = node.state.get_current_step();

        let (policy_logits, value_logits, _position_logits, _last_token) =
            self.get_forward_outputs(observation, current_step);

        let policy_logits = policy_logits
            .to_data()
            .to_vec::<f32>()
            .expect("Failed to convert Tensor to Vec");
        let value = support_to_scalar(value_logits);

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
        let step = node.state.get_current_step();

        let (action_probabilities, value_logits) =
            self.calculate_action_and_value(&observation, step);
        let device = action_probabilities.device();
        let mut action_probabilities = if apply_dirichlet_noise {
            add_dirichlet_noise::<B>(action_probabilities, self.model.action_space, config)
        } else {
            action_probabilities
        };

        let valid_actions = self.get_valid_actions(node);
        let valid_actions = Tensor::from_data(valid_actions, &device);

        action_probabilities = action_probabilities * valid_actions;

        let action_sum = action_probabilities.clone().sum();

        let action_probabilities = action_probabilities / action_sum;

        let action_probabilities = action_probabilities
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to convert tensor of actions probs to vector");
        let value = support_to_scalar(value_logits);

        (action_probabilities, value)
    }

    fn get_action_space(&self) -> usize {
        self.model.action_space
    }

    fn get_observation_space(&self) -> usize {
        self.model.observation_space
    }

    fn train_model(
        &mut self,
        replay_buffer: &mut crate::algorithms::alpha_zero::replay_buffer::ReplayBuffer<
            Self::TransitionType,
        >,
        config: &AlphaZeroConfig,
    ) {
        let orginial_device = &self.model.get_device();
        let training_device = &self.training_device;

        let cross_entropy_loss_config = CrossEntropyLossConfig::new();
        let cross_entropy_loss = cross_entropy_loss_config.init(training_device);

        let mut training_model = self.model.clone().fork(training_device);

        let mut current_lr = 0.0;

        let seq_length = self.model.max_sequence_length; // the current max number of lap im using is 55
        let features = self.model.features_per_sequence;
        let max_seq_length = Some(seq_length);
        let pad_token = 0;
        let timing = Instant::now();
        for iteration in 0..config.training_steps {
            let timing_in = Instant::now();
            let sample = replay_buffer.sample_batch();
            let batch_size = sample.len();

            // Create individual tensors first
            let mut observations = Vec::with_capacity(batch_size);
            let mut policy_targets = Vec::with_capacity(batch_size);
            let mut value_targets = Vec::with_capacity(batch_size);
            let mut position_targets = Vec::with_capacity(batch_size);
            let mut valid_indices = Vec::with_capacity(batch_size);
            let mut tokens_list = Vec::with_capacity(batch_size);

            for transition in sample {
                let time_step = transition.transition_number as usize;

                let index = time_step - 1;
                valid_indices.push(index as i64);

                let mut tokens = (1..=time_step).collect::<Vec<usize>>();
                tokens.resize(seq_length, 0); // Pad to 55 with zeros

                tokens_list.push(tokens);

                let mut observation_vec: Vec<f32> = transition.observation;
                observation_vec.resize(seq_length * features, 0.0);
                let policy_target_vec: Vec<f32> = transition.action_probabilities;
                let value_target = transition.total_reward;
                let final_position_index = transition.final_position - 1; // for cross entropy i minus 1

                let data = TensorData::new(observation_vec, [seq_length, features]);
                let observation_tensor: Tensor<B, 2> = Tensor::from_data(data, training_device);

                let policy_target_tensor: Tensor<B, 1> =
                    Tensor::from_data(policy_target_vec.as_slice(), training_device);

                observations.push(observation_tensor);
                policy_targets.push(policy_target_tensor);
                value_targets.push(value_target);
                position_targets.push(final_position_index);
            }
            // println!("The value targets: {:?}", &value_targets[0..100]);
            let observations: Tensor<B, 3> = Tensor::stack(observations, 0);
            let policy_targets: Tensor<B, 2> = Tensor::stack(policy_targets, 0);
            let value_targets: Tensor<B, 2> =
                scalar_to_support_batch(&value_targets, training_device);
            let position_targets: Tensor<B, 1, Int> =
                Tensor::from_data(position_targets.as_slice(), training_device);

            let timesteps_indices = Tensor::from_data(valid_indices.as_slice(), training_device);

            let mask_pad =
                generate_padding_mask(pad_token, tokens_list, max_seq_length, training_device).mask;
            let mask_attn = generate_autoregressive_mask(batch_size, seq_length, training_device);

            let TronNetworkOuput {
                policy_logits: policy_predictions,
                value_logits: value_predictions,
                last_token: _,
                position_logits: position_predictions,
            } = training_model.forward(
                observations,
                Some(mask_pad),
                Some(mask_attn),
                timesteps_indices,
            );

            let policy_loss = cross_entropy_with_logits(policy_predictions, policy_targets);

            let value_loss = cross_entropy_with_logits(value_predictions, value_targets);

            let position_loss = cross_entropy_loss.forward(position_predictions, position_targets);

            let value_loss_print = value_loss.to_data().to_vec::<f32>().unwrap()[0];
            let policy_loss_print = policy_loss.to_data().to_vec::<f32>().unwrap()[0];
            let position_loss_print = position_loss.to_data().to_vec::<f32>().unwrap()[0];
            let total_loss_print = value_loss_print + policy_loss_print + position_loss_print;

            let total_loss = policy_loss + value_loss + position_loss;
            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &training_model);

            println!(
                "Training iteration {}/{}: value loss = {:.6} , policy loss = {:.6}, position loss = {:.6} , total loss (policy + value + position) = {:.6} | Took {:.4}s",
                iteration,
                config.training_steps,
                value_loss_print,
                policy_loss_print,
                position_loss_print,
                total_loss_print,
                timing_in.elapsed().as_secs_f32()
            );
            let current_learning_rate = config.learning_rate as f64;
            // let current_learning_rate = self.learning_rate_schedule.step();
            current_lr = current_learning_rate;
            training_model = self
                .optimizer
                .step(current_learning_rate, training_model, grads);
        }
        self.model = training_model.fork(orginial_device);

        println!(
            "Took {} to complete training. Current learning rate is: {}, FULL schedule is {:?}",
            timing.elapsed().as_secs_f32(),
            current_lr,
            self.learning_rate_schedule
        );
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
pub struct TronModelConfig {
    observation_space: usize,
    max_timesteps: usize,
    action_space: usize,
    num_drivers: usize,
    d_model: usize,
    d_ff: usize,
    n_heads: usize,
    n_layers: usize,
}

impl TronModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ModelBackend<B> {
        let features_per_sequence = self.observation_space;
        // the plus 1 is for buffer purposes when im indexing, especially when using complex action space,
        // when None is selected
        let max_timesteps = self.max_timesteps + 1;
        let positional_encoding =
            create_positional_encoding::<B>(max_timesteps, self.d_model, device);

        ModelBackend {
            linear1: LinearConfig::new(features_per_sequence, self.d_model).init(device),
            transformer: TransformerEncoderConfig::new(
                self.d_model,
                self.d_ff,
                self.n_heads,
                self.n_layers,
            )
            .with_norm_first(true)
            .with_dropout(0.0)
            .init(device),
            policy_head: LinearConfig::new(self.d_model, self.action_space).init(device),
            value_head: LinearConfig::new(self.d_model, STRATEGY_VALUE_SUPPORT_SIZE).init(device),
            position_head: LinearConfig::new(self.d_model, self.num_drivers).init(device),
            observation_space: self.observation_space,
            action_space: self.action_space,
            num_drivers: self.num_drivers,
            positional_encoding,
            max_sequence_length: max_timesteps,
            features_per_sequence,
        }
    }
}

fn create_transformer_input<B: Backend>(
    tensor: Tensor<B, 3>,
    mask_pad: Option<Tensor<B, 2, Bool>>,
    mask_attn: Option<Tensor<B, 3, Bool>>,
) -> TransformerEncoderInput<B> {
    match (mask_pad, mask_attn) {
        (Some(pad), Some(attn)) => TransformerEncoderInput::new(tensor)
            .mask_pad(pad)
            .mask_attn(attn),
        (None, Some(attn)) => TransformerEncoderInput::new(tensor).mask_attn(attn),
        (Some(pad), None) => TransformerEncoderInput::new(tensor).mask_pad(pad),
        (None, None) => TransformerEncoderInput::new(tensor),
    }
}

pub fn create_positional_encoding<B: Backend>(
    seq_length: usize,
    d_model: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut encoding = vec![vec![0.0; d_model]; seq_length];

    for pos in 0..seq_length {
        for i in 0..d_model {
            // Note to self
            // Compute: pos / 10000^(2*(i//2)/d_model)
            // Even dimensions sin
            // Odd dimensions cos
            let exponent = (2 * (i / 2)) as f32 / d_model as f32;
            let angle = pos as f32 / f32::powf(10000.0, exponent);

            encoding[pos][i] = if i % 2 == 0 {
                f32::sin(angle)
            } else {
                f32::cos(angle)
            };
        }
    }

    let flat: Vec<f32> = encoding.into_iter().flatten().collect();
    let data = TensorData::new(flat, [1, seq_length, d_model]);
    Tensor::from_data(data, device)
}

pub struct TronNetworkOuput<B: Backend> {
    pub policy_logits: Tensor<B, 2>,
    pub position_logits: Tensor<B, 2>,
    pub value_logits: Tensor<B, 2>,
    pub last_token: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct ThreePhaseScheduler {
    warmup: LinearLrScheduler,
    stable: LinearLrScheduler, // Flat line (same start/end)
    decay: LinearLrScheduler,

    warmup_steps: usize,
    stable_end: usize,
    current_step: usize,
}

impl ThreePhaseScheduler {
    pub fn new(
        warmup_steps: usize,
        stable_steps: usize,
        decay_steps: usize,
        initial_lr: f64, // 0.000001
        peak_lr: f64,    // 0.02
        final_lr: f64,   // 0.0002
    ) -> Self {
        // Phase 1: Warmup 0.000001 → 0.02
        let warmup = LinearLrSchedulerConfig::new(initial_lr, initial_lr, warmup_steps)
            .init()
            .unwrap();

        // Phase 2: Stable 0.02 → 0.02 (flat line)
        let stable = LinearLrSchedulerConfig::new(peak_lr, peak_lr, stable_steps)
            .init()
            .unwrap();

        // Phase 3: Decay 0.02 → 0.0002
        let decay = LinearLrSchedulerConfig::new(peak_lr, final_lr, decay_steps)
            .init()
            .unwrap();

        Self {
            warmup,
            stable,
            decay,
            warmup_steps,
            stable_end: warmup_steps + stable_steps,
            current_step: 0,
        }
    }

    pub fn step(&mut self) -> f64 {
        self.current_step += 1;

        if self.current_step <= self.warmup_steps {
            self.warmup.step()
        } else if self.current_step <= self.stable_end {
            self.stable.step()
        } else {
            self.decay.step()
        }
    }
}
