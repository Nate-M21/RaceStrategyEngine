use std::path::Path;

use burn::{
    Tensor,
    config::Config,
    grad_clipping::GradientClippingConfig,
    module::{AutodiffModule, Module},
    nn::{LeakyRelu, LeakyReluConfig, Linear, LinearConfig, loss::CrossEntropyLossConfig},
    optim::{GradientsParams, Optimizer, RmsPropConfig, adaptor::OptimizerAdaptor},
    prelude::Backend,
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::{
        Device, Int, TensorData, activation::softmax, backend::AutodiffBackend,
        loss::cross_entropy_with_logits,
    },
};

use crate::{
    algorithms::{
        alpha_zero::{
            alpha_zero_config::AlphaZeroConfig, node::AlphaZeroNode, replay_buffer::ReplayBuffer,
        },
        helpers::{
            STRATEGY_VALUE_SUPPORT_SIZE, add_dirichlet_noise, apply_legal_actions,
            scalar_to_support_batch, support_to_scalar,
        },
        iris::{
            gatv2::{GATv2Conv, GATv2ConvConfig},
            graph_transformer::{TransformerConv, TransformerConvConfig},
        },
        rob::rob::RobTransition,
    },
    traits::actor_critic::{ActorCritic, AttentionWeights},
    utils::create_fully_connected_edge_index,
};

unsafe impl<B: Backend + AutodiffBackend> Send for IrisModel<B> {}
unsafe impl<B: Backend + AutodiffBackend> Sync for IrisModel<B> {}

#[derive(Module, Debug, Clone, Copy)]
pub enum GNNType {
    GATv2,
    Transformer,
}

#[derive(Module, Debug)]
pub struct ModelBackend<B: Backend> {
    pub conv1_gat: Option<GATv2Conv<B>>,
    pub conv2_gat: Option<GATv2Conv<B>>,
    pub conv3_gat: Option<GATv2Conv<B>>,
    pub conv4_gat: Option<GATv2Conv<B>>,

    pub conv1_trans: Option<TransformerConv<B>>,
    pub conv2_trans: Option<TransformerConv<B>>,
    pub conv3_trans: Option<TransformerConv<B>>,
    pub conv4_trans: Option<TransformerConv<B>>,
    pub conv5_trans: Option<TransformerConv<B>>,

    pub activation: LeakyRelu,

    
    pub post_pool: Vec<Linear<B>>,

    // TODO make this more flexible when i fully switch to Gatv2 so i can switch back Transformerconv
    // for testing
    pub conv1_strategy: GATv2Conv<B>,
    pub conv2_strategy: GATv2Conv<B>,
    pub conv3_strategy: GATv2Conv<B>,
    pub conv4_strategy: GATv2Conv<B>,

    // Task heads
    pub policy_head: Linear<B>,
    pub value_head: Linear<B>,
    pub position_head: Linear<B>,
    pub strategy_head: Linear<B>,

    // Metadata
    features_per_node: usize,
    features_per_node_strategy: usize,
    action_space: usize,
    num_drivers: usize,
    num_laps: usize,
    num_compounds: usize,
    gnn_type: GNNType,
    node_dim: usize,

    drivers_edge_index: Tensor<B, 2, Int>,
    laps_edge_index: Tensor<B, 2, Int>,
}

impl<B: Backend> ModelBackend<B> {
    /// Full forward pass
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        strategy_observation: Tensor<B, 3>,
        return_attention: bool,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        NetworkModelAttention<B>,
    ) {
        let x_out = self.forward_with_attention(x, strategy_observation, return_attention, false);
        x_out
    }

    /// Forward pass with optional attention weight return
    pub fn forward_with_attention(
        &self,
        x: Tensor<B, 3>,
        strategy_observation: Tensor<B, 3>,
        return_attention: bool,
        first_layer_attention: bool,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        NetworkModelAttention<B>,
    ) {
        let batch_size = x.dims()[0];

        let (x_graph, driver_attention) =
            self._gnn_forward(x, return_attention, first_layer_attention);

        // Global pooling
        let x_pooled = self.global_mean_pool(x_graph, self.num_drivers, batch_size);

        // Post-pool MLP residual im doing this as i found it trains better
        let mut x_final = x_pooled;
        for layer in self.post_pool.iter() {
            x_final = layer.forward(x_final.clone()) + x_final;
            // x_final = layer.forward(x_final);
            x_final = self.activation.forward(x_final)
        }

        let strategy_observation = inject_normalized_lap_batched(strategy_observation);

        let (strategy_node_embeddings, lap_attention) =
            self._strategy_gnn_forward(strategy_observation, x_final.clone(), return_attention);

        let x_strategy_pooled =
            self.global_mean_pool(strategy_node_embeddings.clone(), self.num_laps, batch_size);

        let strategy_logits = self.strategy_head.forward(strategy_node_embeddings);

        let strategy_logits =
            strategy_logits.reshape([batch_size, self.num_laps * self.num_compounds]);

        let position_logits = self.position_head.forward(x_strategy_pooled.clone());

        // Task heads
        let policy_logits = self.policy_head.forward(x_strategy_pooled.clone());
        let value_logits = self.value_head.forward(x_strategy_pooled);

        let attention = NetworkModelAttention {
            driver_attention,
            lap_attention,
        };

        (
            policy_logits,
            value_logits,
            position_logits,
            strategy_logits,
            attention,
        )
    }

    /// Performs GNN forward pass on node features
    ///
    /// # Arguments
    /// * `x` - Node features [batch_size, num_drivers, features_per_node]
    /// * `return_attention` - Whether to return attention weights from final layer
    ///
    /// # Returns
    /// * Node embeddings [batch_size * num_drivers, hidden]
    /// * Optional attention weights (edge_index, alpha)
    fn _gnn_forward(
        &self,
        x: Tensor<B, 3>,
        return_attention: bool,
        first_layer_attention: bool,
    ) -> (Tensor<B, 2>, Option<(Tensor<B, 2, Int>, Tensor<B, 2>)>) {
        // let device = x.device();
        let batch_size = x.dims()[0];

        // Flatten to [batch_size * num_drivers, features]
        // let x_flat = x.reshape([batch_size * self.num_drivers, self.features_per_node]);

        // Create batched edge index
        // let batched_edge_index = Self::create_batched_edge_index(
        //     batch_size,
        //     self.drivers_edge_index.clone(),
        //     self.num_drivers,
        //     &device,
        // );

        // Choose architecture path
        let (x_out, attention) = match self.gnn_type {
            GNNType::GATv2 => {
                let x_flat = x;

                // Layer 1
                let (x1, first_attn) = self
                    .conv1_gat
                    .as_ref()
                    .unwrap()
                    .forward_dense(x_flat, return_attention);

                let x1 = self.activation.forward(x1);

                // Layer 2
                let (x2, _) = self.conv2_gat.as_ref().unwrap().forward_dense(x1, false);
                let x2 = self.activation.forward(x2);

                // Layer 3
                let (x3, _) = self.conv3_gat.as_ref().unwrap().forward_dense(x2, false);
                let x3 = self.activation.forward(x3);

                // Layer 4 (with optional attention return)
                let (x4, final_attn) = self
                    .conv4_gat
                    .as_ref()
                    .unwrap()
                    .forward_dense(x3, return_attention);
                let x4 = self.activation.forward(x4);

                let attn = if first_layer_attention {
                    first_attn
                } else {
                    final_attn
                };

                let hidden_dim = x4.dims()[2];
                let x_flat = x4.reshape([batch_size * self.num_drivers, hidden_dim]);

                (x_flat, attn)
            }
            GNNType::Transformer => {
                let x_flat = x;

                // Layer 1
                let (x1, first_attn) = self
                    .conv1_trans
                    .as_ref()
                    .unwrap()
                    .forward_dense(x_flat, return_attention);
                let x1 = self.activation.forward(x1);

                // Layer 2
                let (x2, _) = self.conv2_trans.as_ref().unwrap().forward_dense(x1, false);
                let x2 = self.activation.forward(x2);

                // Layer 3
                let (x3, _) = self.conv3_trans.as_ref().unwrap().forward_dense(x2, false);
                let x3 = self.activation.forward(x3);

                // Layer 4
                let (x4, _) = self.conv4_trans.as_ref().unwrap().forward_dense(x3, false);
                let x4 = self.activation.forward(x4);

                // Layer 5
                let (x5, final_attn) = self
                    .conv5_trans
                    .as_ref()
                    .unwrap()
                    .forward_dense(x4, return_attention);
                let x5 = self.activation.forward(x5);

                let attn = if first_layer_attention {
                    first_attn
                } else {
                    final_attn
                };

                let hidden_dim = x5.dims()[2];
                let x_flat = x5.reshape([batch_size * self.num_drivers, hidden_dim]);

                (x_flat, attn)
            }
        };

        (x_out, attention)
    }

    fn _strategy_gnn_forward(
        &self,
        x: Tensor<B, 3>,
        post_pool: Tensor<B, 2>,
        return_attention: bool,
    ) -> (
        Tensor<B, 2>,
        Option<(burn::Tensor<B, 2, Int>, burn::Tensor<B, 2>)>,
    ) {
        let batch_size = x.dims()[0];
        let hidden_size = post_pool.dims()[1]; // Get hidden dim

        let (x1, _) = self.conv1_strategy.forward_dense(x, false);

        let x1 = self.activation.forward(x1);

        let (x2, _) = self.conv2_strategy.forward_dense(x1, false);

        let x2 = self.activation.forward(x2);

        let post_pool_expanded =
            post_pool
                .unsqueeze_dim::<3>(1)
                .expand([batch_size, self.num_laps, hidden_size]);

        let x2 = x2 + post_pool_expanded;

        let (x3, _) = self.conv3_strategy.forward_dense(x2, false);

        let x3 = self.activation.forward(x3);

        let (x4, final_attn) = self.conv4_strategy.forward_dense(x3, return_attention);

        let x4 = self.activation.forward(x4);

        let hidden_dim = x4.dims()[2];
        let x_flat = x4.reshape([batch_size * self.num_laps, hidden_dim]);

        (x_flat, final_attn)
    }

    /// Global mean pooling - makes model permutation invariant
    fn global_mean_pool(
        &self,
        x: Tensor<B, 2>,
        num_nodes: usize,
        batch_size: usize,
    ) -> Tensor<B, 2> {
        let hidden = x.dims()[1];
        let x_reshaped = x.reshape([batch_size, num_nodes, hidden]);
        x_reshaped.mean_dim(1).squeeze_dim::<2>(1) // [batch_size, hidden]
    }

    /// Creates batched edge index for parallel processing
    /// Shifts node indices for each graph in the batch
    pub fn create_batched_edge_index(
        batch_size: usize,
        base_edge_index: Tensor<B, 2, Int>,
        num_nodes: usize,
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        let num_edges = base_edge_index.dims()[1];

        // Create offsets [0, N, 2N, ...]
        let offsets = (0..batch_size)
            .map(|i| (i * num_nodes) as i64)
            .collect::<Vec<i64>>();

        let offsets = TensorData::new(offsets, [batch_size]);
        let offsets = Tensor::<B, 1, Int>::from_data(offsets, device);

        // Reshape to [1, batch_size, 1] so we can broadcast against edge_index which we will reshape
        let offsets = offsets.reshape([1, batch_size, 1]);

        // Reshape edge_index to [2, 1, num_edges]
        let edges = base_edge_index.clone().reshape([2, 1, num_edges]);

        let batched_edges = edges + offsets;

        batched_edges.reshape([2, batch_size * num_edges])
    }

    fn get_device(&self) -> B::Device {
        match self.gnn_type {
            GNNType::GATv2 => self.conv1_gat.as_ref().unwrap().lin_source.weight.device(),
            GNNType::Transformer => self.conv1_trans.as_ref().unwrap().lin_query.weight.device(),
        }
    }
}
const NO_RETURN_ATTENTION: bool = false;
const RETURN_ATTENTION: bool = true;
/// I.R.I.S - Ineffable Race Intelligence System
#[derive(Clone)]
pub struct IrisModel<B: Backend + AutodiffBackend> {
    model: ModelBackend<B>,
    training_device: Device<B>,
    optimizer_config: burn::optim::SgdConfig,
    optimizer: OptimizerAdaptor<burn::optim::Sgd<B::InnerBackend>, ModelBackend<B>, B>,
    // optimizer_config: burn::optim::AdamWConfig,
    // optimizer: OptimizerAdaptor<burn::optim::AdamW, ModelBackend<B>, B>,
}

impl<B: Backend + AutodiffBackend> IrisModel<B> {
    pub fn new(model: ModelBackend<B>, training_device: Device<B>) -> Self {
        let optimizer_config = burn::optim::SgdConfig::new()
            .with_momentum(Some(
                burn::optim::momentum::MomentumConfig::new()
                    .with_momentum(0.9)
                    .with_nesterov(true),
            ))
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(0.00001)))
            .with_gradient_clipping(Some(GradientClippingConfig::Norm(1.0)));

        // let optimizer_config = burn::optim::AdamWConfig::new()
        //     .with_weight_decay(0.0001)
        //     .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));

        // TODO, test rmsprop after demo on Petar's advice
        RmsPropConfig::new();

        let optimizer = optimizer_config.init();

        Self {
            model,
            optimizer,
            optimizer_config,
            training_device,
        }
    }

    fn _predict(
        &self,
        observation: &[f32],
        strategy_encoding: &[f32],
        return_attention: bool,
    ) -> (
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Option<ModelAttention<<B as AutodiffBackend>::InnerBackend>>,
    ) {
        let (policy_logits, value_logits, position_logits, strategy_logits, attention) =
            self.get_forward_outputs(observation, strategy_encoding, return_attention);

        // Process policy
        let epsilon = 1e-8;
        let mut action_probabilities = softmax(policy_logits, 1);
        action_probabilities = action_probabilities + epsilon;
        let priors_sum = action_probabilities.clone().sum_dim(1);
        let action_probabilities = action_probabilities / priors_sum.unsqueeze();

        // Squeeze to 1D
        let action_probabilities = action_probabilities.squeeze_dim(0);

        let value_logits = value_logits.squeeze_dim(0);
        let position_probabilities = softmax(position_logits, 1).squeeze_dim(0);

        let strategy_logits = strategy_logits.squeeze_dim(0);

        let attention: Option<ModelAttention<<B as AutodiffBackend>::InnerBackend>> =
            match return_attention {
                true => Some(ModelAttention {
                    driver_attention: attention.driver_attention.unwrap(),
                    lap_attention: attention.lap_attention.unwrap(),
                }),
                false => None,
            };

        (
            action_probabilities,
            value_logits,
            position_probabilities,
            strategy_logits,
            attention,
        )
    }

    fn get_forward_outputs(
        &self,
        observation: &[f32],
        strategy_encoding: &[f32],
        return_attention: bool,
    ) -> (
        Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 2>,
        NetworkModelAttention<<B as AutodiffBackend>::InnerBackend>,
    ) {
        let device = self.model.get_device();

        // Reshape flat observation to [1, num_drivers, features_per_node]
        let features_per_node = self.model.features_per_node;
        let num_drivers = self.model.num_drivers;

        let num_laps = self.model.num_laps;
        let num_compounds = self.model.num_compounds;

        let batch_size = 1;
        let observation = TensorData::new(
            observation.to_vec(),
            [batch_size, num_drivers, features_per_node],
        );

        let observation: Tensor<<B as AutodiffBackend>::InnerBackend, 3> =
            Tensor::from_data(observation, &device);

        let strategy_observation = TensorData::new(
            strategy_encoding.to_vec(),
            [batch_size, num_laps, num_compounds],
        );

        let strategy_observation: Tensor<<B as AutodiffBackend>::InnerBackend, 3> =
            Tensor::from_data(strategy_observation, &device);

        let (policy_logits, value_logits, position_logits, strategy_logits, attention) = self
            .model
            .valid()
            .forward(observation, strategy_observation, return_attention);
        (
            policy_logits,
            value_logits,
            position_logits,
            strategy_logits,
            attention,
        )
    }

    fn calculate_action_and_value(
        &self,
        observation: &[f32],
        strategy_encoding: &[f32],
    ) -> (
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
        Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    ) {
        let (action_probabilities, value_logit, _, _, _) =
            self._predict(observation, &strategy_encoding, NO_RETURN_ATTENTION);
        (action_probabilities, value_logit)
    }

    pub fn display_model(&self) {
        println!("{}\n{:?}", self.model, self.optimizer_config);
    }
}

impl<B: Backend + AutodiffBackend> ActorCritic for IrisModel<B> {
    type TransitionType = RobTransition;
    type ObservationType = Vec<f32>;

    fn predict(
        &self,
        _observation: &[f32],
        _current_time_step: Option<usize>,
        _legal_actions: Option<&[f32]>,
    ) -> Vec<Vec<f32>> {
        if true {
            panic!("I.R.I.S model needs to be used with predict_with_attention function");
        }

        let capacity = self.model.num_compounds * self.model.num_laps;

        // i dont need strategy encoding here its not used for anythin so making it nothing
        let strategy_encoding = vec![0.0; capacity];
        let (
            mut action_probabilities,
            value_logits,
            position_probabilities,
            _strategy_logits,
            _attention,
        ) = self._predict(_observation, &strategy_encoding, NO_RETURN_ATTENTION);

        if let Some(legal_actions) = _legal_actions {
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

    fn predict_with_attention(
        &self,
        observation: &[f32],
        strategy_encoding: &[f32],
        _current_time_step: Option<usize>,
        legal_actions: Option<&[f32]>,
    ) -> (Vec<Vec<f32>>, Option<AttentionWeights>) {
        let (
            mut action_probabilities,
            value_logits,
            position_probabilities,
            strategy_logits,
            attention,
        ) = self._predict(observation, &strategy_encoding, RETURN_ATTENTION);

        if let Some(legal_actions) = legal_actions {
            let device = action_probabilities.device();
            let legal_actions = Tensor::from_data(legal_actions, &device);
            action_probabilities = apply_legal_actions::<B>(action_probabilities, legal_actions);
        }
        let strategy_values = strategy_logits.to_data().into_vec::<f32>().unwrap();

        let value = support_to_scalar(value_logits);

        let ModelAttention {
            driver_attention,
            lap_attention,
        } = attention.unwrap();
        let (drivers_edge_index_tensor, drivers_attention_weights_tensor) = driver_attention;
        let (laps_edge_index_tensor, laps_attention_weights_tensor) = lap_attention;

        let (drivers_edge_index_vec, drivers_weights_vec) =
            extract_edges_and_weights(drivers_edge_index_tensor, drivers_attention_weights_tensor);
        let (laps_edge_index_vec, laps_weights_vec) =
            extract_edges_and_weights(laps_edge_index_tensor, laps_attention_weights_tensor);
        let attention_weights = AttentionWeights {
            drivers_edge_index: drivers_edge_index_vec,
            drivers_weights: drivers_weights_vec,
            laps_edge_index: laps_edge_index_vec,
            laps_weights: laps_weights_vec,
        };

        let predictions = vec![
            action_probabilities.to_data().into_vec::<f32>().unwrap(),
            vec![value],
            position_probabilities.to_data().into_vec::<f32>().unwrap(),
            strategy_values,
        ];

        (predictions, Some(attention_weights))
    }

    fn get_raw_action_and_value_logits<Environment>(
        &self,
        node: &AlphaZeroNode<Environment>,
    ) -> (Vec<f32>, f32)
    where
        Environment: crate::traits::gym::MCTSGymEnvironment<Observation = Self::ObservationType>,
    {
        let observation = &node.current_observation;
        let strategy_encoding = &node.state.get_current_encoded_strategy();
        let (policy_logits, value_logits, _position_logits, _strategy_logits, _attention) =
            self.get_forward_outputs(observation, strategy_encoding, NO_RETURN_ATTENTION);

        let policy_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1> =
            policy_logits.squeeze_dim(0);
        let value_logits: Tensor<<B as AutodiffBackend>::InnerBackend, 1> =
            value_logits.squeeze_dim(0);

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
        let strategy = &node.state.get_current_encoded_strategy();
        let (action_probabilities, value_logits) =
            self.calculate_action_and_value(observation, strategy);
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
            .expect("Failed to convert action probs");

        let value = support_to_scalar(value_logits);

        (action_probabilities, value)
    }

    fn get_action_space(&self) -> usize {
        self.model.action_space
    }

    fn get_observation_space(&self) -> usize {
        self.model.num_drivers * self.model.features_per_node
    }

    fn train_model(
        &mut self,
        replay_buffer: &mut ReplayBuffer<Self::TransitionType>,
        config: &AlphaZeroConfig,
    ) {
        let orginial_device = &self.model.get_device();
        let training_device = &self.training_device;
        let mut training_model = self.model.clone().fork(training_device);

        let num_drivers = training_model.num_drivers;
        let num_laps = training_model.num_laps;
        let num_compounds = training_model.num_compounds;

        let cross_entropy_loss_config = CrossEntropyLossConfig::new();
        let cross_entropy_loss = cross_entropy_loss_config.init(training_device);

        println!("Starting IRIS training...");
        let timing = std::time::Instant::now();

        for iteration in 0..config.training_steps {
            let timing_in = std::time::Instant::now();
            let timing_sample = std::time::Instant::now();
            let sample_of_transitions = replay_buffer.sample_batch();
            let timing_sample = timing_sample.elapsed().as_secs_f32();

            let batch_size = sample_of_transitions.len();

            let mut observations = Vec::with_capacity(batch_size);
            let mut policy_targets = Vec::with_capacity(batch_size);
            let mut value_targets = Vec::with_capacity(batch_size);
            let mut position_targets = Vec::with_capacity(batch_size);
            let mut strategy_observations = Vec::with_capacity(batch_size);
            let mut strategy_targets = Vec::with_capacity(batch_size);

            for transition in sample_of_transitions {
                let obs_flat: Vec<f32> = transition.observation;
                let features_per_node = training_model.features_per_node;

                // Reshape to [num_drivers, features_per_node]
                let data = TensorData::new(obs_flat, [num_drivers, features_per_node]);

                let obs_tensor: Tensor<B, 2> = Tensor::from_data(data, training_device);

                let policy_target_vec: Vec<f32> = transition.action_probabilities;
                let policy_target: Tensor<B, 1> =
                    Tensor::from_data(policy_target_vec.as_slice(), training_device);

                let strategy_observation_vec = transition.current_transition_strategy_encoding;

                let data = TensorData::new(strategy_observation_vec, [num_laps, num_compounds]);
                let stratgey_observation: Tensor<B, 2> = Tensor::from_data(data, training_device);

                let strategy_target_vec = transition.episode_end_strategy_encoding;
                let strategy_target: Tensor<B, 1> =
                    Tensor::from_data(strategy_target_vec.as_slice(), training_device);

                let final_position_index = transition.final_position - 1;
                let value_target = transition.total_reward;

                observations.push(obs_tensor);
                policy_targets.push(policy_target);
                value_targets.push(value_target);
                position_targets.push(final_position_index);
                strategy_targets.push(strategy_target);
                strategy_observations.push(stratgey_observation);
            }

            // Stack to batches: [batch_size, num_drivers, features]
            let observations_batch: Tensor<B, 3> = Tensor::stack(observations, 0);
            let policy_targets_batch: Tensor<B, 2> = Tensor::stack(policy_targets, 0);
            let value_targets_batch: Tensor<B, 2> =
                scalar_to_support_batch(&value_targets, training_device);
            let position_targets: Tensor<B, 1, Int> =
                Tensor::from_data(position_targets.as_slice(), training_device);
            let strategy_target_batch: Tensor<B, 2> = Tensor::stack(strategy_targets, 0);

            let strategy_observation = Tensor::stack(strategy_observations, 0);

            // Forward pass
            // training the other heads
            let (
                policy_predictions,
                value_predictions,
                position_predictions,
                strategy_predictions,
                _sttention,
            ) = training_model.forward(
                observations_batch,
                strategy_observation,
                NO_RETURN_ATTENTION,
            );

            let policy_loss = cross_entropy_with_logits(policy_predictions, policy_targets_batch);
            let value_loss = cross_entropy_with_logits(value_predictions, value_targets_batch);
            let position_loss = cross_entropy_loss.forward(position_predictions, position_targets);
            // let strategy_loss =
            //     cross_entropy_with_logits(strategy_predictions, strategy_target_batch);
            let strategy_loss = strategy_loss_per_lap(
                strategy_predictions,  // [batch, 165]
                strategy_target_batch, // [batch, 165]
                num_laps,              // 55
                num_compounds,         // 3
            );

            let value_loss_print = value_loss.to_data().to_vec::<f32>().unwrap()[0];
            let policy_loss_print = policy_loss.to_data().to_vec::<f32>().unwrap()[0];
            let position_loss_print = position_loss.to_data().to_vec::<f32>().unwrap()[0];
            let strategy_loss_print = strategy_loss.to_data().to_vec::<f32>().unwrap()[0];
            let total_loss_print =
                value_loss_print + policy_loss_print + position_loss_print + strategy_loss_print;

            let total_loss = policy_loss + value_loss + position_loss + strategy_loss;
            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &training_model);

            println!(
                "Iteration {}/{}: val={:.6} pol={:.6} pos={:.6} strat={:.6} total={:.6} |  {:.4}s  (Duration to sample - {:.4}s)",
                iteration + 1,
                config.training_steps,
                value_loss_print,
                policy_loss_print,
                position_loss_print,
                strategy_loss_print,
                total_loss_print,
                timing_in.elapsed().as_secs_f32(),
                timing_sample
            );

            let current_learning_rate = config.learning_rate as f64;
            training_model = self
                .optimizer
                .step(current_learning_rate, training_model, grads);
        }

        println!(
            "Training complete: {:.2}s total\n",
            timing.elapsed().as_secs_f32()
        );

        self.model = training_model.fork(orginial_device);
    }

    fn save_model(&self, path: &Path) {
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        self.model
            .clone()
            .save_file(path, &recorder)
            .expect("Failed to save IRIS model");
    }

    fn load_model(&mut self, path: &Path) {
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
        self.model = self
            .model
            .clone()
            .load_file(path, &recorder, &self.model.get_device())
            .expect("Failed to load IRIS model");
    }
}

fn extract_edges_and_weights<B: Backend>(
    drivers_edge_index_tensor: Tensor<B, 2, Int>,
    drivers_attention_weights_tensor: Tensor<B, 2>,
) -> (Vec<(usize, usize)>, Vec<Vec<f32>>) {
    // Convert edge_index [2, num_edges] to Vec<(usize, usize)>
    let edge_data = drivers_edge_index_tensor.to_data();
    let edge_vec: Vec<i64> = edge_data.into_vec().unwrap();
    let num_edges = edge_vec.len() / 2;

    let edge_index_vec: Vec<(usize, usize)> = (0..num_edges)
        .map(|i| (edge_vec[i] as usize, edge_vec[num_edges + i] as usize))
        .collect();

    // Convert attention weights [num_edges, num_heads] to Vec<Vec<f32>>
    let weights_data = drivers_attention_weights_tensor.to_data();
    let num_heads = drivers_attention_weights_tensor.dims()[1];
    let weights_flat: Vec<f32> = weights_data.into_vec().unwrap();

    let weights_vec: Vec<Vec<f32>> = (0..num_edges)
        .map(|i| {
            (0..num_heads)
                .map(|h| weights_flat[i * num_heads + h])
                .collect()
        })
        .collect();
    (edge_index_vec, weights_vec)
}

//  Model Configuration

#[derive(Config, Debug)]
pub struct IrisModelConfig {
    hidden_size: usize,
    node_dim: usize,
    num_heads: usize,
    features_per_node: usize,
    action_space: usize,
    num_drivers: usize,
    num_laps: usize,
    num_compounds: usize,
    gnn_type: String, // "gatv2" or "transformer"
}

impl IrisModelConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        drivers_edges: &[(usize, usize)],
    ) -> ModelBackend<B> {
        let gnn_type = match self.gnn_type.as_str() {
            "transformer" => GNNType::Transformer,
            _ => GNNType::GATv2, // Default to GATv2 , need to test it after demo over TransformerConv
        };

        let head_dim = self.node_dim / self.num_heads;

        // Convert edges to tensor
        let num_edges = drivers_edges.len();
        let sources: Vec<i64> = drivers_edges.iter().map(|(src, _)| *src as i64).collect();
        let targets: Vec<i64> = drivers_edges.iter().map(|(_, tgt)| *tgt as i64).collect();

        let mut drivers_edge_data = Vec::with_capacity(2 * num_edges);
        drivers_edge_data.extend(sources);
        drivers_edge_data.extend(targets);

        let drivers_edge_tensor =
            Tensor::from_data(TensorData::new(drivers_edge_data, [2, num_edges]), device);

        let laps_edges = create_fully_connected_edge_index(self.num_laps, true);

        let num_edges = laps_edges.len();
        let sources: Vec<i64> = laps_edges.iter().map(|(src, _)| *src as i64).collect();
        let targets: Vec<i64> = laps_edges.iter().map(|(_, tgt)| *tgt as i64).collect();

        let mut laps_edge_data = Vec::with_capacity(2 * num_edges);
        laps_edge_data.extend(sources);
        laps_edge_data.extend(targets);

        let laps_edge_tensor =
            Tensor::from_data(TensorData::new(laps_edge_data, [2, num_edges]), device);

        let (conv1_gat, conv2_gat, conv3_gat, conv4_gat) = match gnn_type {
            GNNType::GATv2 => (
                Some(
                    GATv2ConvConfig::new(self.features_per_node, head_dim)
                        .with_heads(self.num_heads)
                        .with_concat(true)
                        .with_residual(true)
                        .init(device),
                ),
                Some(
                    GATv2ConvConfig::new(self.node_dim, head_dim)
                        .with_heads(self.num_heads)
                        .with_concat(true)
                        .with_residual(true)
                        .init(device),
                ),
                Some(
                    GATv2ConvConfig::new(self.node_dim, head_dim)
                        .with_heads(self.num_heads)
                        .with_concat(true)
                        .with_residual(true)
                        .init(device),
                ),
                Some(
                    GATv2ConvConfig::new(self.node_dim, head_dim)
                        .with_heads(self.num_heads)
                        .with_concat(true)
                        .with_residual(true)
                        .init(device),
                ),
            ),
            _ => (None, None, None, None),
        };

        let (conv1_trans, conv2_trans, conv3_trans, conv4_trans, conv5_trans) = match gnn_type {
            GNNType::Transformer => (
                Some(
                    TransformerConvConfig::new(self.features_per_node, head_dim)
                        .with_heads(self.num_heads)
                        .with_concat(true)
                        .with_root_weight(true)
                        .with_beta(true)
                        .init(device),
                ),
                Some(
                    TransformerConvConfig::new(self.node_dim, head_dim)
                        .with_heads(self.num_heads)
                        .with_concat(true)
                        .with_root_weight(true)
                        .with_beta(true)
                        .init(device),
                ),
                Some(
                    TransformerConvConfig::new(self.node_dim, head_dim)
                        .with_heads(self.num_heads)
                        .with_concat(true)
                        .with_root_weight(true)
                        .with_beta(true)
                        .init(device),
                ),
                Some(
                    TransformerConvConfig::new(self.node_dim, head_dim)
                        .with_heads(self.num_heads)
                        .with_concat(true)
                        .with_root_weight(true)
                        .with_beta(true)
                        .init(device),
                ),
                Some(
                    TransformerConvConfig::new(self.node_dim, head_dim)
                        .with_heads(self.num_heads)
                        .with_concat(true)
                        .with_root_weight(true)
                        .with_beta(true)
                        .init(device),
                ),
            ),
            _ => (None, None, None, None, None),
        };

        // Post-pooling MLP after GNN layer
        let mut post_pool = Vec::with_capacity(4);
        post_pool.push(LinearConfig::new(self.node_dim, self.hidden_size).init(device));
        post_pool.push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
        post_pool.push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
        post_pool.push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));
        post_pool.push(LinearConfig::new(self.hidden_size, self.hidden_size).init(device));

        // the one is for lap feature i inject
        let features_per_node_strategy: usize = self.num_compounds + 1;

        // TODO also maket Gatv2 version of this on Petar's advice could gain better performation due to dynamic attention
        
        // let conv1_strategy = TransformerConvConfig::new(features_per_node_strategy, head_dim)
        //     .with_heads(self.num_heads)
        //     .with_concat(true)
        //     .with_root_weight(true)
        //     .with_beta(true)
        //     .init(device);

        // let conv2_strategy = TransformerConvConfig::new(self.node_dim, head_dim)
        //     .with_heads(self.num_heads)
        //     .with_concat(true)
        //     .with_root_weight(true)
        //     .with_beta(true)
        //     .init(device);

        // let conv3_strategy = TransformerConvConfig::new(self.node_dim, head_dim)
        //     .with_heads(self.num_heads)
        //     .with_concat(true)
        //     .with_root_weight(true)
        //     .with_beta(true)
        //     .init(device);

        // let conv4_strategy = TransformerConvConfig::new(self.node_dim, head_dim)
        //     .with_heads(self.num_heads)
        //     .with_concat(true)
        //     .with_root_weight(true)
        //     .with_beta(true)
        //     .init(device);

        let conv1_strategy = GATv2ConvConfig::new(features_per_node_strategy, head_dim)
            .with_heads(self.num_heads)
            .with_concat(true)
            .with_residual(true)
            .init(device);
        let conv2_strategy = GATv2ConvConfig::new(self.node_dim, head_dim)
            .with_heads(self.num_heads)
            .with_concat(true)
            .with_residual(true)
            .init(device);
        let conv3_strategy = GATv2ConvConfig::new(self.node_dim, head_dim)
            .with_heads(self.num_heads)
            .with_concat(true)
            .with_residual(true)
            .init(device);

        let conv4_strategy = GATv2ConvConfig::new(self.node_dim, head_dim)
            .with_heads(self.num_heads)
            .with_concat(true)
            .with_residual(true)
            .init(device);

        ModelBackend {
            conv1_gat,
            conv2_gat,
            conv3_gat,
            conv4_gat,
            conv1_trans,
            conv2_trans,
            conv3_trans,
            conv4_trans,
            conv5_trans,
            activation: LeakyReluConfig::new().init(),
            post_pool,
            policy_head: LinearConfig::new(self.hidden_size, self.action_space).init(device),
            value_head: LinearConfig::new(self.hidden_size, STRATEGY_VALUE_SUPPORT_SIZE)
                .init(device),
            position_head: LinearConfig::new(self.hidden_size, self.num_drivers).init(device),
            strategy_head: LinearConfig::new(self.node_dim, self.num_compounds).init(device),
            conv1_strategy,
            conv2_strategy,
            conv3_strategy,
            conv4_strategy,
            features_per_node: self.features_per_node,
            features_per_node_strategy,
            action_space: self.action_space,
            num_drivers: self.num_drivers,
            num_compounds: self.num_compounds,
            num_laps: self.num_laps,
            node_dim: self.node_dim,
            gnn_type,
            drivers_edge_index: drivers_edge_tensor,
            laps_edge_index: laps_edge_tensor,
        }
    }
}

fn strategy_loss_per_lap<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    num_laps: usize,
    num_compounds: usize,
) -> Tensor<B, 1> {
    let batch_size = predictions.dims()[0];

    // Reshape to [batch * num_laps, num_compounds]
    let preds = predictions.reshape([batch_size * num_laps, num_compounds]);
    let tgts = targets.reshape([batch_size * num_laps, num_compounds]);

    cross_entropy_with_logits(preds, tgts)
}

pub fn inject_normalized_lap_batched<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let device = x.device();
    let [batch, n_points, _dims] = x.dims();

    let idx = Tensor::<B, 1, Int>::arange(0..n_points as i64, &device)
        .float()
        .div_scalar((n_points - 1) as f32)
        .reshape([1, n_points, 1])
        .expand([batch, n_points, 1]);

    Tensor::cat(vec![idx, x], 2)
}

pub struct NetworkModelAttention<B: Backend> {
    driver_attention: Option<(Tensor<B, 2, Int>, Tensor<B, 2>)>,
    lap_attention: Option<(Tensor<B, 2, Int>, Tensor<B, 2>)>,
}

pub struct ModelAttention<B: Backend> {
    driver_attention: (Tensor<B, 2, Int>, Tensor<B, 2>),
    lap_attention: (Tensor<B, 2, Int>, Tensor<B, 2>),
}
