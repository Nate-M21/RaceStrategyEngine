use burn::{
    Tensor,
    prelude::Backend,
    tensor::{IndexingUpdateOp, Int, TensorData, activation::softmax, backend::AutodiffBackend},
};
use rand::rng;
use rand_distr::Distribution;

use crate::algorithms::alpha_zero::{
    alpha_zero_config::AlphaZeroConfig, dirichlet::StableDirichlet,
};

pub fn apply_legal_actions<B: AutodiffBackend>(
    mut action_probabilities: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    valid_actions: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
) -> Tensor<<B as AutodiffBackend>::InnerBackend, 1> {
    action_probabilities = action_probabilities * valid_actions;

    let action_sum = action_probabilities.clone().sum();

    let action_probabilities = action_probabilities / action_sum;
    action_probabilities
}

pub fn add_dirichlet_noise<B: AutodiffBackend>(
    action_probabilities: Tensor<<B as AutodiffBackend>::InnerBackend, 1>,
    action_space: usize,
    config: AlphaZeroConfig,
) -> Tensor<<B as AutodiffBackend>::InnerBackend, 1> {
    let device = action_probabilities.device();

    let mut rng = rng();

    let dirichlet = StableDirichlet::new(config.dirichlet_alpha, action_space).unwrap();

    let noise = dirichlet.sample(&mut rng);

    let noise_tensor = Tensor::from_data(noise.as_slice(), &device);

    let action_probabilities = {
        ((1.0 - config.dirichlet_epsilon as f64) * action_probabilities)
            + (config.dirichlet_epsilon as f64 * noise_tensor)
    };

    action_probabilities
}

pub const STRATEGY_VALUE_SUPPORT_SIZE: usize = 151; // Bins from -300 to 300
const HALF_WIDTH: f32 = 75.0;

fn h_transform(x: f32) -> f32 {
    let eps = 0.001;
    x.signum() * ((x.abs() + 1.0).sqrt() - 1.0 + eps * x)
}

fn h_inverse(x: f32) -> f32 {
    let eps = 0.001;
    if x == 0.0 {
        return 0.0;
    }
    let x_abs = x.abs();
    // Simplified inverse for clarity
    let val = ((1.0 + 4.0 * eps * (x_abs + 1.0 + eps)).sqrt() - 1.0) / (2.0 * eps);
    x.signum() * (val * val - 1.0)
}

pub fn support_to_scalar<B: Backend>(logits: Tensor<B, 1>) -> f32 {
    let device = logits.device();
    let probs = softmax(logits, 0);

    let support_values: Vec<f32> = (0..STRATEGY_VALUE_SUPPORT_SIZE)
        .map(|i| (i as f32) - HALF_WIDTH)
        .collect();
    let support_tensor = Tensor::<B, 1>::from_data(support_values.as_slice(), &device);

    let transformed_val = (probs * support_tensor)
        .sum()
        .to_data()
        .as_slice::<f32>()
        .unwrap()[0];

    h_inverse(transformed_val)
}

pub fn scalar_to_support_batch<B: Backend>(scalars: &[f32], device: &B::Device) -> Tensor<B, 2> {
    let batch_size = scalars.len();
    let mut batch_data = Vec::with_capacity(batch_size * STRATEGY_VALUE_SUPPORT_SIZE);

    for &scalar in scalars {
        let transformed = h_transform(scalar).clamp(-HALF_WIDTH, HALF_WIDTH);
        let shifted = transformed + HALF_WIDTH;
        let lower = shifted.floor() as usize;
        let upper = (lower + 1).min(STRATEGY_VALUE_SUPPORT_SIZE - 1);
        let p_upper = shifted - lower as f32;

        let mut data = vec![0.0; STRATEGY_VALUE_SUPPORT_SIZE];
        data[lower] = 1.0 - p_upper;
        data[upper] = p_upper;
        batch_data.extend(data);
    }

    let data = TensorData::new(batch_data, [batch_size, STRATEGY_VALUE_SUPPORT_SIZE]);
    Tensor::from_data(data, device)
}

pub fn normalize_hidden_state<B: Backend>(state: Tensor<B, 2>) -> Tensor<B, 2> {
    let epsilon = 1e-5;

    // Min-max normalization per batch sample (dim 1)
    let min_vals = state.clone().min_dim(1);
    let max_vals = state.clone().max_dim(1);
    let range = max_vals - min_vals.clone() + epsilon;

    (state - min_vals) / range
}
/// GNN helpers

pub fn gather_nodes<B: Backend>(x: Tensor<B, 3>, indices: Tensor<B, 1, Int>) -> Tensor<B, 3> {
    x.select(0, indices)
}

pub fn scatter_add<B: Backend>(
    messages: Tensor<B, 3>,
    target_indices: Tensor<B, 1, Int>,
    num_nodes: usize,
) -> Tensor<B, 3> {
    let device = messages.device();
    let dims = messages.dims();
    let heads = dims[1];
    let out_channels = dims[2];

    let out = Tensor::zeros([num_nodes, heads, out_channels], &device);

    let indices_expanded = target_indices
        .reshape([dims[0], 1, 1])
        .repeat_dim(1, heads)
        .repeat_dim(2, out_channels);

    out.scatter(0, indices_expanded, messages, IndexingUpdateOp::Add)
}
pub fn edge_softmax<B: Backend>(
    logits: Tensor<B, 2>,
    target_indices: Tensor<B, 1, Int>,
    num_nodes: usize,
) -> Tensor<B, 2> {
    let device = logits.device();
    let heads = logits.dims()[1];
    let num_edges = logits.dims()[0];

    // Stabilization subtracting max per head to prevent overflow.
    let max_val = logits.clone().max_dim(0); // Shape: [1, heads]
    let logits_stable = logits - max_val; // Broadcasting: [num_edges, heads] - [1, heads]

    let exp_logits = logits_stable.exp();

    let mut denominators = Tensor::zeros([num_nodes, heads], &device);
    let indices_expanded = target_indices
        .clone()
        .reshape([num_edges, 1])
        .repeat_dim(1, heads);

    denominators = denominators.scatter(
        0,
        indices_expanded,
        exp_logits.clone(),
        IndexingUpdateOp::Add,
    );

    let gathered_denominators = denominators.select(0, target_indices);

    // small epsilon to avoid division by zero
    exp_logits / (gathered_denominators + 1e-6)
}
