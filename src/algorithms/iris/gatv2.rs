use burn::{
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
    prelude::Backend,
    tensor::{
        Int, Tensor, TensorData,
        activation::{leaky_relu, softmax},
    },
};

use crate::algorithms::helpers::{edge_softmax, gather_nodes, scatter_add};

/// GATv2 (Graph Attention Network v2) Convolution Layer
///
/// Implements the improved attention mechanism from "How Attentive are Graph Attention Networks?"
/// Key difference from GAT: applies LeakyReLU AFTER combining source and target features,
/// making attention dynamic (conditioned on the query node).
///
/// Formula: α_ij = softmax(a^T · LeakyReLU(Θ_s·x_i + Θ_t·x_j))
#[derive(Module, Debug)]
pub struct GATv2Conv<B: Backend> {
    pub lin_source: Linear<B>,
    pub lin_target: Linear<B>,
    pub att: Param<Tensor<B, 2>>,
    pub lin_edge: Option<Linear<B>>,
    pub residual: Option<Linear<B>>,
    pub bias: Option<Param<Tensor<B, 1>>>,

    heads: usize,
    out_channels: usize,
    concat: bool,
    negative_slope: f64,
    dropout: f64,
    share_weights: bool,
}
pub struct GATv2ConvConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub heads: usize,
    pub concat: bool,
    pub negative_slope: f64,
    pub dropout: f64,
    pub edge_dim: Option<usize>,
    pub bias: bool,
    pub share_weights: bool,
    pub residual: bool,
}

impl Default for GATv2ConvConfig {
    fn default() -> Self {
        Self {
            in_channels: 0,
            out_channels: 0,
            heads: 1,
            concat: true,
            negative_slope: 0.2,
            dropout: 0.0,
            edge_dim: None,
            bias: true,
            share_weights: false,
            residual: false,
        }
    }
}

impl GATv2ConvConfig {
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

    pub fn with_negative_slope(mut self, slope: f64) -> Self {
        self.negative_slope = slope;
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

    pub fn with_share_weights(mut self, share: bool) -> Self {
        self.share_weights = share;
        self
    }

    pub fn with_residual(mut self, residual: bool) -> Self {
        self.residual = residual;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> GATv2Conv<B> {
        let hidden_size = self.heads * self.out_channels;

        let lin_source = LinearConfig::new(self.in_channels, hidden_size)
            .with_bias(self.bias)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 })
            .init(device);

        let lin_target = if self.share_weights {
            lin_source.clone()
        } else {
            LinearConfig::new(self.in_channels, hidden_size)
                .with_bias(self.bias)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device)
        };

        // Initialize Attention
        let att = Tensor::random(
            [self.heads, self.out_channels],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            device,
        );
        let gain = 1.0;
        let fan_in = self.out_channels as f64;
        let std = gain * (2.0 / fan_in).sqrt();
        let att = att * std;

        let lin_edge = self.edge_dim.map(|edge_dim| {
            LinearConfig::new(edge_dim, hidden_size)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device)
        });

        let total_out = if self.concat {
            self.heads * self.out_channels
        } else {
            self.out_channels
        };

        let residual = if self.residual {
            Some(
                LinearConfig::new(self.in_channels, total_out)
                    .with_bias(false)
                    .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                    .init(device),
            )
        } else {
            None
        };

        let bias = if self.bias {
            Some(Param::from_tensor(Tensor::zeros([total_out], device)))
        } else {
            None
        };

        GATv2Conv {
            lin_source,
            lin_target,
            att: Param::from_tensor(att), // Correctly wrapped
            lin_edge,
            residual,
            bias,
            heads: self.heads,
            out_channels: self.out_channels,
            concat: self.concat,
            negative_slope: self.negative_slope,
            dropout: self.dropout,
            share_weights: self.share_weights,
        }
    }
}

impl<B: Backend> GATv2Conv<B> {
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

        let residual_out = self.residual.as_ref().map(|res| res.forward(x.clone()));

        let x_source = self.lin_source.forward(x.clone());
        let x_target = self.lin_target.forward(x);

        let x_source = x_source.reshape([num_nodes, self.heads, self.out_channels]);
        let x_target = x_target.reshape([num_nodes, self.heads, self.out_channels]);

        let x_i = gather_nodes(x_target.clone(), targets.clone());
        let x_j = gather_nodes(x_source.clone(), sources.clone());

        let mut combined = x_i.clone() + x_j.clone();

        if let Some(edge_attr) = edge_attr {
            if let Some(ref lin_edge) = self.lin_edge {
                let edge_feat = lin_edge.forward(edge_attr);
                let edge_feat = edge_feat.reshape([num_edges, self.heads, self.out_channels]);
                combined = combined + edge_feat;
            }
        }

        let combined = leaky_relu(combined, self.negative_slope);

        let att_broadcast = self.att.val().unsqueeze_dim(0);
        let att_broadcast = att_broadcast.repeat_dim(0, num_edges);
        let alpha_logits = (combined * att_broadcast).sum_dim(2).squeeze_dim(2);

        let alpha = edge_softmax(alpha_logits, targets.clone(), num_nodes);

        let alpha_expanded = alpha.clone().unsqueeze_dim(2);
        let messages = x_j * alpha_expanded;

        let out = scatter_add(messages, targets, num_nodes);

        let mut out = if self.concat {
            out.reshape([num_nodes, self.heads * self.out_channels])
        } else {
            out.mean_dim(1).squeeze_dim(1)
        };

        if let Some(res) = residual_out {
            out = out + res;
        }

        if let Some(ref bias) = self.bias {
            let bias_broadcast = bias.val().unsqueeze_dim(0).repeat_dim(0, num_nodes);
            out = out + bias_broadcast;
        }

        let attention_weights = if return_attention_weights {
            Some((edge_index, alpha))
        } else {
            None
        };

        (out, attention_weights)
    }
}
impl<B: Backend> GATv2Conv<B> {
    /// Batched dense forward pass for fully-connected graphs.
    /// Avoids edge_index scatter/gather operations.
    pub fn forward_dense(
        &self,
        x: Tensor<B, 3>,
        return_attention: bool,
    ) -> (Tensor<B, 3>, Option<(Tensor<B, 2, Int>, Tensor<B, 2>)>) {
        let [batch, num_nodes, _in_channels] = x.dims();
        let heads = self.heads;
        let out_c = self.out_channels;

        // Optional residual projection
        let residual_out = self.residual.as_ref().map(|res| res.forward(x.clone()));

        let x_source = self
            .lin_source
            .forward(x.clone())
            .reshape([batch, num_nodes, heads, out_c])
            .permute([0, 2, 1, 3]); // [batch, heads, N, out_c]

        let x_target = self
            .lin_target
            .forward(x)
            .reshape([batch, num_nodes, heads, out_c])
            .permute([0, 2, 1, 3]); // [batch, heads, N, out_c]

        // Broadcast and add to get pairwise combinations: target(i) + source(j)
        // x_target (i) becomes [batch, heads, N, 1, out_c]
        // x_source (j) becomes [batch, heads, 1, N, out_c]
        let x_i = x_target.clone().unsqueeze_dim(3);
        let x_j = x_source.clone().unsqueeze_dim(2);

        // combined shape: [batch, heads, N, N, out_c]
        let combined = leaky_relu(x_i + x_j, self.negative_slope);

        // Apply attention vector 
        // self.att is [heads, out_c], broadcast to [1, heads, 1, 1, out_c]
        let att_broadcast = self.att.val().reshape([1, heads, 1, 1, out_c]);

        // Multiply and sum over the feature dimension to get logits [batch, heads, N, N]
        let alpha_logits = (combined * att_broadcast).sum_dim(4).squeeze_dim(4);

        // Softmax over the source dimension (j = last dimension)
        let alpha = softmax(alpha_logits, 3); // [batch, heads, N, N]

        // Aggregate messages via batch matrix multiplication
        // [batch, heads, N, N] @ [batch, heads, N, out_c] -> [batch, heads, N, out_c]
        let out_heads = alpha.clone().matmul(x_source);

        //  Reshape / Pool outputs
        let mut out = if self.concat {
            out_heads
                .permute([0, 2, 1, 3])
                .reshape([batch, num_nodes, heads * out_c])
        } else {
            out_heads
                .permute([0, 2, 1, 3])
                .mean_dim(2)
                .reshape([batch, num_nodes, out_c])
        };

        if let Some(res) = residual_out {
            out = out + res;
        }

        if let Some(ref bias) = self.bias {
            // bias is [total_out], broadcast to [batch, N, total_out]
            let bias_broadcast = bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
            out = out + bias_broadcast;
        }

        let attention = if return_attention {
            let device = out.device();
            let attn_0 = alpha
                .slice([0..1, 0..heads, 0..num_nodes, 0..num_nodes])
                .squeeze_dim(0); // [heads, N, N]

            let mut sources_vec = Vec::with_capacity(num_nodes * num_nodes);
            let mut targets_vec = Vec::with_capacity(num_nodes * num_nodes);
            for target in 0..num_nodes {
                for source in 0..num_nodes {
                    sources_vec.push(source as i64);
                    targets_vec.push(target as i64);
                }
            }

            let mut edge_data = sources_vec.clone();
            edge_data.extend(targets_vec);
            let edge_index = Tensor::<B, 2, Int>::from_data(
                TensorData::new(edge_data, [2, num_nodes * num_nodes]),
                &device,
            );

            let alpha_sparse = attn_0
                .permute([1, 2, 0])
                .reshape([num_nodes * num_nodes, heads]);

            Some((edge_index, alpha_sparse))
        } else {
            None
        };

        (out, attention)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Candle;
    use burn::tensor::{ElementConversion, TensorData};


    /// Build a simple ring edge_index for `n` nodes (i → i+1, wrapping).
    fn ring_edges<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 2, Int> {
        let sources: Vec<i64> = (0..n as i64).collect();
        let targets: Vec<i64> = (1..n as i64).chain(std::iter::once(0)).collect();
        let mut data = sources.clone();
        data.extend(targets);
        Tensor::<B, 2, Int>::from_data(TensorData::new(data, [2, n]), device)
    }

    // sparse forward

    #[test]
    fn test_gatv2_forward() {
        type B = Candle;
        let device = Default::default();

        let config = GATv2ConvConfig::new(16, 8).with_heads(4).with_concat(true);

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 2>::random(
            [10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let sources = vec![
            0i64, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
        ];
        let targets = vec![
            1i64, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4, 7, 8, 6, 8, 6, 7, 0, 1,
        ];
        let mut edge_data = sources.clone();
        edge_data.extend(targets);
        let edge_index =
            Tensor::<B, 2, Int>::from_data(TensorData::new(edge_data, [2, 20]), &device);

        let (out, attention) = layer.forward(x, edge_index.clone(), None, true);

        assert_eq!(out.dims(), [10, 32]); // 10 nodes, 4 heads * 8 channels
        assert!(attention.is_some());
        if let Some((edge_idx, alpha)) = attention {
            assert_eq!(edge_idx.dims(), edge_index.dims());
            assert_eq!(alpha.dims(), [20, 4]); // 20 edges, 4 heads
        }
    }

    #[test]
    fn test_gatv2_with_residual() {
        type B = Candle;
        let device = Default::default();

        let config = GATv2ConvConfig::new(16, 8)
            .with_heads(4)
            .with_concat(true)
            .with_residual(true);

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 2>::random(
            [10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let edge_index = ring_edges::<B>(10, &device);

        let (out, _) = layer.forward(x, edge_index, None, false);
        assert_eq!(out.dims(), [10, 32]);
    }

    #[test]
    fn test_gatv2_no_concat() {
        // mean-pooling over heads → output is [N, out_channels]
        type B = Candle;
        let device = Default::default();

        let config = GATv2ConvConfig::new(16, 8).with_heads(4).with_concat(false);

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 2>::random(
            [10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let edge_index = ring_edges::<B>(10, &device);

        let (out, _) = layer.forward(x, edge_index, None, false);
        assert_eq!(out.dims(), [10, 8]); // mean over heads → out_channels only
    }

    #[test]
    fn test_gatv2_with_edge_attr() {
        type B = Candle;
        let device = Default::default();

        let edge_dim = 6usize;
        let config = GATv2ConvConfig::new(16, 8)
            .with_heads(4)
            .with_concat(true)
            .with_edge_dim(Some(edge_dim));

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 2>::random(
            [10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let edge_index = ring_edges::<B>(10, &device);
        let edge_attr = Tensor::<B, 2>::random(
            [10, edge_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let (out, _) = layer.forward(x, edge_index, Some(edge_attr), false);
        assert_eq!(out.dims(), [10, 32]);
    }

    #[test]
    fn test_gatv2_share_weights() {
        // lin_source and lin_target share the same weights → still valid forward
        type B = Candle;
        let device = Default::default();

        let config = GATv2ConvConfig::new(16, 8)
            .with_heads(4)
            .with_concat(true)
            .with_share_weights(true);

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 2>::random(
            [10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let edge_index = ring_edges::<B>(10, &device);

        let (out, _) = layer.forward(x, edge_index, None, false);
        assert_eq!(out.dims(), [10, 32]);
    }

    // dense forward

    #[test]
    fn test_forward_dense_batched() {
        type B = Candle;
        let device = Default::default();

        let config = GATv2ConvConfig::new(16, 8).with_heads(4).with_concat(true);

        let layer = config.init::<B>(&device);

        // batch=2, 10 nodes, 16 features
        let x = Tensor::<B, 3>::random(
            [2, 10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let (out, attention) = layer.forward_dense(x, true);

        assert_eq!(out.dims(), [2, 10, 32]); // [batch, N, heads * out_c]
        assert!(attention.is_some());
        if let Some((edge_idx, alpha)) = attention {
            assert_eq!(edge_idx.dims(), [2, 100]); // 10*10 = 100 pairs from batch 0
            assert_eq!(alpha.dims(), [100, 4]); // 100 pairs, 4 heads
        }
    }

    #[test]
    fn test_forward_dense_no_attention() {
        type B = Candle;
        let device = Default::default();

        let config = GATv2ConvConfig::new(16, 8).with_heads(4).with_concat(true);

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
    fn test_forward_dense_no_concat() {
        type B = Candle;
        let device = Default::default();

        let config = GATv2ConvConfig::new(16, 8).with_heads(4).with_concat(false);

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 3>::random(
            [2, 10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let (out, _) = layer.forward_dense(x, false);
        assert_eq!(out.dims(), [2, 10, 8]); // mean over heads → out_channels only
    }

    #[test]
    fn test_forward_dense_with_residual() {
        type B = Candle;
        let device = Default::default();

        let config = GATv2ConvConfig::new(16, 8)
            .with_heads(4)
            .with_concat(true)
            .with_residual(true);

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 3>::random(
            [2, 10, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let (out, _) = layer.forward_dense(x, false);
        assert_eq!(out.dims(), [2, 10, 32]);
    }

    #[test]
    fn test_forward_dense_batches_independent() {
        // Verify batch 0 and batch 1 don't cross-attend.
        // Two identical batches must produce identical outputs.
        type B = Candle;
        let device = Default::default();

        let config = GATv2ConvConfig::new(4, 4)
            .with_heads(1)
            .with_concat(true)
            .with_residual(false);

        let layer = config.init::<B>(&device);

        let x_single = Tensor::<B, 3>::random(
            [1, 5, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let x_double = Tensor::cat(vec![x_single.clone(), x_single.clone()], 0);

        let (out_single, _) = layer.forward_dense(x_single, false);
        let (out_double, _) = layer.forward_dense(x_double, false);

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

    #[test]
    fn test_forward_dense_attention_sums_to_one() {
        // Each row of the N×N attention matrix (over source nodes j)
        // should sum to 1.0 after softmax.
        type B = Candle;
        let device = Default::default();

        let config = GATv2ConvConfig::new(8, 4).with_heads(2).with_concat(true);

        let layer = config.init::<B>(&device);

        let x = Tensor::<B, 3>::random(
            [1, 5, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // alpha_sparse: [N*N, heads] = [25, 2]
        let (_, attention) = layer.forward_dense(x, true);
        let (_, alpha) = attention.unwrap();

        // Sum each head's weights for every target node i (rows 0..5, 5..10, ...)
        // We just check the total sum = N (5 target nodes × 1.0 each per head)
        let total: f32 = alpha.sum().into_scalar().elem();
        let expected = 5.0 * 2.0; // N target nodes * heads
        assert!(
            (total - expected).abs() < 1e-3,
            "Attention weights don't sum to N*heads. Got {total}, expected {expected}"
        );
    }
}
