/*
    Policy network. Consumes a state and samples actions from the policy network weights.

    Code adapted from https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/deep_cfr_tf2.py
*/

use std::borrow::Borrow;

use tch::{
    nn::{self, ModuleT, Path},
    Scalar, Kind,
};

pub struct PolicyNetwork {
    layers: Vec<nn::Linear>,
    norm: nn::LayerNorm,
    out_layer: nn::Linear,
}

impl PolicyNetwork {
    pub fn new<'a, P>(path: P, in_dim: i64, layer_sizes: Vec<usize>, num_actions: usize) -> Self
    where
        P: Borrow<Path<'a>>,
    {
        let path = path.borrow();
        let mut layers = Vec::with_capacity(layer_sizes.len());
        layers.push(nn::linear(
            &(path / "policy_linear_0"),
            in_dim,
            layer_sizes[0] as i64,
            Default::default(),
        ));
        for (i, &size) in layer_sizes.iter().skip(1).enumerate() {
            let layer_name = format!("policy_linear_{}", i + 1);
            let layer_path = &(path / layer_name);
            let in_dim = layer_sizes[i] as i64;
            // TODO: Handle skip connections
            let layer = nn::linear(layer_path, in_dim, size as i64, Default::default());
            layers.push(layer);
        }

        let norm_shape = vec![layer_sizes[layer_sizes.len() - 1] as i64];
        let norm = nn::layer_norm(&(path / "policy_norm"), norm_shape, Default::default());
        let out_layer = nn::linear(
            &(path / "policy_linear_out"),
            *layer_sizes.last().unwrap() as i64,
            num_actions as i64,
            Default::default(),
        );
        Self {
            layers,
            norm,
            out_layer,
        }
    }

    pub fn forward_t(
        &self,
        xs: &tch::Tensor,
        action_mask: &tch::Tensor,
        train: bool,
    ) -> tch::Tensor {
        let mut x = self.layers[0].forward_t(xs, train).leaky_relu();
        for layer in self.layers.iter().skip(1) {
            x = layer.forward_t(&x, train).leaky_relu();
        }

        let hidden = self.norm.forward_t(&x, train);
        let out = self.out_layer.forward_t(&hidden, train);
        let bool_mask = action_mask.not_equal(Scalar::float(1.0));
        let out = out.masked_fill(&bool_mask, Scalar::float(-1e9));
        out.softmax(-1, tch::Kind::Float)
    }
}
