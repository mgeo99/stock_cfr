use std::borrow::Borrow;

use tch::nn::{self, ModuleT, Path};

/// Skip-Dense layer
pub struct SkipDense {
    linear: nn::Linear,
}

impl SkipDense {
    pub fn new<'a, P>(path: P, in_dim: i64, out_dim: i64) -> Self
    where
        P: Borrow<Path<'a>>,
    {
        let path = path.borrow();
        let linear = nn::linear(&(path / "skip_dense"), in_dim, out_dim, Default::default());
        Self { linear }
    }
}

impl ModuleT for SkipDense {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        let hidden = self.linear.forward_t(xs, train);
        hidden + xs
    }
}
