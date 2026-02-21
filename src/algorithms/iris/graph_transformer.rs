use burn::tensor::activation::softmax;
use burn::{
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Int, Tensor, activation::sigmoid},
};

use crate::algorithms::helpers::{edge_softmax, gather_nodes, scatter_add};
/// Graph Transformer Convolution Layer
///
/// Implements the graph transformer from "Masked Label Prediction: Unified Message Passing Model"
/// Uses multi-head dot-product attention mechanism similar to the standard Transformer.
///
/// Formula: x'_i = W_1·x_i + Σ_j α_ij·(W_2·x_j + W_6·e_ij)
/// where: α_ij = softmax((W_3·x_i)^T · (W_4·x_j + W_6·e_ij) / sqrt(d))
#[derive(Module, Debug)]
pub struct TransformerConv<B: Backend> {
    pub lin_key: Linear<B>,
    pub lin_query: Linear<B>,
    pub lin_value: Linear<B>,
    pub lin_edge: Option<Linear<B>>,
    pub lin_skip: Linear<B>,
    pub lin_beta: Option<Linear<B>>,

    heads: usize,
    out_channels: usize,
    concat: bool,
    beta: bool,
    dropout: f64,
    root_weight: bool,
}

pub struct TransformerConvConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub heads: usize,
    pub concat: bool,
    pub beta: bool,
    pub dropout: f64,
    pub edge_dim: Option<usize>,
    pub bias: bool,
    pub root_weight: bool,
}

impl Default for TransformerConvConfig {
    fn default() -> Self {
        Self {
            in_channels: 0,
            out_channels: 0,
            heads: 1,
            concat: true,
            beta: false,
            dropout: 0.0,
            edge_dim: None,
            bias: true,
            root_weight: true,
        }
    }
}

impl TransformerConvConfig {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            ..Default::default()
        }
    }

    pub fn with_heads(mut self, heads: usize) -> Self {
        self.heads = heads;
        self
    }

    pub fn with_concat(mut self, concat: bool) -> Self {
        self.concat = concat;
        self
    }

    pub fn with_beta(mut self, beta: bool) -> Self {
        self.beta = beta;
        self
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn with_edge_dim(mut self, edge_dim: Option<usize>) -> Self {
        self.edge_dim = edge_dim;
        self
    }

    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_root_weight(mut self, root_weight: bool) -> Self {
        self.root_weight = root_weight;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerConv<B> {
        let hidden_size = self.heads * self.out_channels;

        // Q, K, V projections
        let lin_query = LinearConfig::new(self.in_channels, hidden_size)
            .with_bias(self.bias)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 })
            .init(device);

        let lin_key = LinearConfig::new(self.in_channels, hidden_size)
            .with_bias(self.bias)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 })
            .init(device);

        let lin_value = LinearConfig::new(self.in_channels, hidden_size)
            .with_bias(self.bias)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 })
            .init(device);

        // Edge feature transformation
        let lin_edge = self.edge_dim.map(|edge_dim| {
            LinearConfig::new(edge_dim, hidden_size)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device)
        });

        // Skip connection
        let skip_out = if self.concat {
            hidden_size
        } else {
            self.out_channels
        };

        let lin_skip = LinearConfig::new(self.in_channels, skip_out)
            .with_bias(self.bias)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 })
            .init(device);

        // Beta gating (optional)
        let lin_beta = if self.beta && self.root_weight {
            let beta_in = if self.concat {
                3 * hidden_size
            } else {
                3 * self.out_channels
            };
            Some(
                LinearConfig::new(beta_in, 1)
                    .with_bias(false)
                    .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                    .init(device),
            )
        } else {
            None
        };

        TransformerConv {
            lin_key,
            lin_query,
            lin_value,
            lin_edge,
            lin_skip,
            lin_beta,
            heads: self.heads,
            out_channels: self.out_channels,
            concat: self.concat,
            beta: self.beta && self.root_weight,
            dropout: self.dropout,
            root_weight: self.root_weight,
        }
    }
}

impl<B: Backend> TransformerConv<B> {
    /// Batched dense forward pass for fully-connected graphs.
    ///
    /// Takes a 3D tensor — no edge index needed, every node attends to every other node
    /// within its own graph. Batches never cross-attend.
    ///
    /// # Arguments
    /// * `x` - Node features [batch, num_nodes, in_channels]
    /// * `return_attention` - Whether to return attention weights
    ///
    /// # Returns
    /// * Node embeddings [batch, num_nodes, out_channels * heads]
    /// * Optional attention in SPARSE format (edge_index [2, E], alpha [E, heads])
    ///   converted from dense [batch, heads, N, N] for compatibility with predict_with_attention
    ///   NOTE: attention is only returned for batch item 0 (used for inference which is batch=1)
    pub fn forward_dense(
        &self,
        x: Tensor<B, 3>,
        return_attention: bool,
    ) -> (Tensor<B, 3>, Option<(Tensor<B, 2, Int>, Tensor<B, 2>)>) {
        let [batch, num_nodes, _in_channels] = x.dims();
        let heads = self.heads;
        let k_dim = self.out_channels;
        let hidden = heads * k_dim;

        let q = self
            .lin_query
            .forward(x.clone())
            .reshape([batch, num_nodes, heads, k_dim])
            .permute([0, 2, 1, 3]); // [batch, heads, N, k_dim]

        let k = self
            .lin_key
            .forward(x.clone())
            .reshape([batch, num_nodes, heads, k_dim])
            .permute([0, 2, 1, 3]); // [batch, heads, N, k_dim]

        let v = self
            .lin_value
            .forward(x.clone())
            .reshape([batch, num_nodes, heads, k_dim])
            .permute([0, 2, 1, 3]); // [batch, heads, N, k_dim]

        // Scaled dot-product attention
        // [batch, heads, N, k_dim] @ [batch, heads, k_dim, N] -> [batch, heads, N, N]
        let scale = (k_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;

        // Softmax over source dimension (last dim = which node am I attending to)
        let attn_weights = softmax(scores, 3); // [batch, heads, N, N]

        // Aggregate values
        // [batch, heads, N, N] @ [batch, heads, N, k_dim] -> [batch, heads, N, k_dim]
        let out_heads = attn_weights.clone().matmul(v);

        // [batch, heads, N, k_dim] -> [batch, N, heads, k_dim] -> [batch, N, heads*k_dim]
        let mut out = if self.concat {
            out_heads
                .permute([0, 2, 1, 3])
                .reshape([batch, num_nodes, hidden])
        } else {
            out_heads
                .permute([0, 2, 1, 3])
                .mean_dim(2)
                .reshape([batch, num_nodes, k_dim])
        };

        // Skip connection
        if self.root_weight {
            let x_skip = self.lin_skip.forward(x.clone()); // [batch, N, hidden]

            if let Some(ref lin_beta) = self.lin_beta {
                let diff = out.clone() - x_skip.clone();
                let concat_input = Tensor::cat(vec![out.clone(), x_skip.clone(), diff], 2);
                let beta = sigmoid(lin_beta.forward(concat_input)); // [batch, N, 1]
                out = beta.clone() * x_skip + (beta.neg() + 1.0) * out;
            } else {
                out = out + x_skip;
            }
        }

        // Convert dense attention to sparse format for compatibility
        // Only for batch item 0 — inference is always batch=1 anyway
        let attention = if return_attention {
            // attn_weights: [batch, heads, N, N]
            //  wan edge_index [2, N*N], alpha [N*N, heads]
            let device = x.device();

            let attn_0 = attn_weights
                .slice([0..1, 0..heads, 0..num_nodes, 0..num_nodes])
                .squeeze_dim(0); // [heads, N, N]

            // Build edge_index for fully connected graph [2, N*N]
            let mut sources_vec: Vec<i64> = Vec::with_capacity(num_nodes * num_nodes);
            let mut targets_vec: Vec<i64> = Vec::with_capacity(num_nodes * num_nodes);
            for target in 0..num_nodes {
                for source in 0..num_nodes {
                    sources_vec.push(source as i64);
                    targets_vec.push(target as i64);
                }
            }
            let mut edge_data = sources_vec.clone();
            edge_data.extend(targets_vec);
            let edge_index = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(edge_data, [2, num_nodes * num_nodes]),
                &device,
            );

            // attn_0: [heads, N, N] -> permute to [N, N, heads] -> reshape [N*N, heads]
            let alpha = attn_0
                .permute([1, 2, 0]) // [N, N, heads]
                .reshape([num_nodes * num_nodes, heads]); // [N*N, heads]

            Some((edge_index, alpha))
        } else {
            None
        };

        (out, attention)
    }

    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        edge_index: Tensor<B, 2, Int>,
        edge_attr: Option<Tensor<B, 2>>,
        return_attention_weights: bool,
    ) -> (Tensor<B, 2>, Option<(Tensor<B, 2, Int>, Tensor<B, 2>)>) {
        let num_nodes = x.dims()[0];
        let num_edges = edge_index.dims()[1];

        let sources = edge_index
            .clone()
            .slice([0..1, 0..num_edges])
            .reshape([num_edges]);
        let targets = edge_index
            .clone()
            .slice([1..2, 0..num_edges])
            .reshape([num_edges]);

        let query = self.lin_query.forward(x.clone());
        let key = self.lin_key.forward(x.clone());
        let value = self.lin_value.forward(x.clone());

        let query = query.reshape([num_nodes, self.heads, self.out_channels]);
        let key = key.reshape([num_nodes, self.heads, self.out_channels]);
        let value = value.reshape([num_nodes, self.heads, self.out_channels]);

        let query_i = gather_nodes(query, targets.clone());
        let mut key_j = gather_nodes(key, sources.clone());
        let mut value_j = gather_nodes(value, sources.clone());

        if let Some(edge_attr) = edge_attr {
            if let Some(ref lin_edge) = self.lin_edge {
                let edge_feat = lin_edge.forward(edge_attr);
                let edge_feat = edge_feat.reshape([num_edges, self.heads, self.out_channels]);
                key_j = key_j + edge_feat.clone();
                value_j = value_j + edge_feat;
            }
        }

        let scale = (self.out_channels as f64).sqrt();
        let alpha_logits = (query_i * key_j).sum_dim(2).squeeze_dim(2) / scale;

        let alpha = edge_softmax(alpha_logits, targets.clone(), num_nodes);

        let alpha_expanded = alpha.clone().unsqueeze_dim(2);
        let messages = value_j * alpha_expanded;

        let out = scatter_add(messages, targets, num_nodes);

        let mut out = if self.concat {
            out.reshape([num_nodes, self.heads * self.out_channels])
        } else {
            out.mean_dim(1).squeeze_dim(1)
        };

        if self.root_weight {
            let x_skip = self.lin_skip.forward(x);

            if let Some(ref lin_beta) = self.lin_beta {
                let diff = out.clone() - x_skip.clone();
                let concat_input = Tensor::cat(vec![out.clone(), x_skip.clone(), diff], 1);
                let beta = lin_beta.forward(concat_input);
                let beta = sigmoid(beta);
                let beta_broadcast = beta.repeat_dim(1, out.dims()[1]);
                out = beta_broadcast.clone() * x_skip + (1.0 - beta_broadcast) * out;
            } else {
                out = out + x_skip;
            }
        }

        let attention_weights: Option<(Tensor<B, 2, Int>, Tensor<B, 2>)> =
            if return_attention_weights {
                Some((edge_index, alpha))
            } else {
                None
            };

        (out, attention_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Candle;
    use burn::tensor::TensorData;

    #[test]
    fn test_transformer_conv_forward() {
        type B = Candle;
        let device = Default::default();

        let config = TransformerConvConfig::new(16, 8)
            .with_heads(4)
            .with_concat(true)
            .with_root_weight(true);

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 2>::random(
            [10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let sources = vec![0i64, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let targets = vec![1i64, 2, 0, 2, 0, 1, 4, 5, 3, 5];

        let mut edge_data = Vec::with_capacity(20);
        for i in 0..10 {
            edge_data.push(sources[i]);
        }
        for i in 0..10 {
            edge_data.push(targets[i]);
        }

        let edge_index = Tensor::<B, 2, burn::tensor::Int>::from_data(
            TensorData::new(edge_data, [2, 10]),
            &device,
        );

        let (out, attention) = layer.forward(x, edge_index.clone(), None, true);

        assert_eq!(out.dims(), [10, 32]); // 10 nodes, 4 heads * 8 channels
        assert!(attention.is_some());

        if let Some((edge_idx, alpha)) = attention {
            assert_eq!(edge_idx.dims(), edge_index.dims());
            assert_eq!(alpha.dims(), [10, 4]); // 10 edges, 4 heads
        }
    }

    #[test]
    fn test_transformer_conv_with_beta() {
        type B = Candle;
        let device = Default::default();

        let config = TransformerConvConfig::new(16, 8)
            .with_heads(4)
            .with_concat(true)
            .with_beta(true)
            .with_root_weight(true);

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 2>::random(
            [10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let sources = vec![0i64, 1, 2, 3, 4];
        let targets = vec![1i64, 2, 3, 4, 0];

        let mut edge_data = Vec::with_capacity(10);
        for i in 0..5 {
            edge_data.push(sources[i]);
        }
        for i in 0..5 {
            edge_data.push(targets[i]);
        }

        let edge_index = Tensor::<B, 2, burn::tensor::Int>::from_data(
            TensorData::new(edge_data, [2, 5]),
            &device,
        );

        let (out, _) = layer.forward(x, edge_index, None, false);
        assert_eq!(out.dims(), [10, 32]);
    }

    #[test]
    fn test_forward_dense_batched() {
        type B = Candle;
        let device = Default::default();

        let config = TransformerConvConfig::new(16, 8)
            .with_heads(4)
            .with_concat(true)
            .with_root_weight(true);

        let layer = config.init::<B>(&device);

        // batch=2, 10 nodes, 16 features
        let x = Tensor::<B, 3>::random(
            [2, 10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let (out, attention) = layer.forward_dense(x, true);

        // output: [batch, N, heads * k_dim] = [2, 10, 32]
        assert_eq!(out.dims(), [2, 10, 32]);

        // attention sparse format for batch 0: edge_index [2, 100], alpha [100, 4]
        assert!(attention.is_some());
        if let Some((edge_idx, alpha)) = attention {
            assert_eq!(edge_idx.dims(), [2, 100]); // 10*10 = 100 edges
            assert_eq!(alpha.dims(), [100, 4]); // 100 edges, 4 heads
        }
    }

    #[test]
    fn test_forward_dense_no_attention() {
        type B = Candle;
        let device = Default::default();

        let config = TransformerConvConfig::new(16, 8)
            .with_heads(4)
            .with_concat(true)
            .with_root_weight(true);

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 3>::random(
            [4, 20, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let (out, attention) = layer.forward_dense(x, false);

        assert_eq!(out.dims(), [4, 20, 32]);
        assert!(attention.is_none());
    }

    #[test]
    fn test_forward_dense_batches_independent() {
        // Verify batch 0 and batch 1 don't cross-attend
        type B = Candle;
        let device = Default::default();

        let config = TransformerConvConfig::new(4, 4)
            .with_heads(1)
            .with_concat(true)
            .with_root_weight(false); // disable skip for cleaner test

        let layer = config.init::<B>(&device);

        // Two identical batches
        let x_single = Tensor::<B, 3>::random(
            [1, 5, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let x_double = Tensor::cat(vec![x_single.clone(), x_single.clone()], 0);

        let (out_single, _) = layer.forward_dense(x_single, false);
        let (out_double, _) = layer.forward_dense(x_double, false);

        // batch 0 of double should equal single
        let out_double_0 = out_double.slice([0..1, 0..5, 0..4]);
        let out_single_0 = out_single.slice([0..1, 0..5, 0..4]);

        let diff = (out_double_0 - out_single_0).abs().max().to_data();
        let max_diff: f32 = diff.as_slice().unwrap()[0];
        assert!(
            max_diff < 1e-5,
            "Batches are not independent! max diff: {}",
            max_diff
        );
    }
}
