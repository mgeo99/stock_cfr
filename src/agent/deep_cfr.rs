use tch::{nn, Kind};

use crate::env::state::StockState;

use super::{advantage::AdvantageNetwork, policy::PolicyNetwork};

pub struct DeepCFRConfig {
    /// Device to place all networks/tensors
    pub device: tch::Device,
    /// Number of layers and their sizes for the policy network
    pub policy_layers: Vec<usize>,
    /// Number of layers and their sizes for the advantage network
    pub advantage_layers: Vec<usize>,
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size for the policy network updates
    pub batch_size_policy: usize,
    /// Batch size for the advantage network updates
    pub batch_size_adv: usize,
    /// Capacity for the reservoir buffer
    pub memory_capacity: usize,
    /// # of training steps for the policy net
    pub policy_train_steps: usize,
    /// # of training steps for the advantage net
    pub adv_train_steps: usize,
    /// Whether to re-initialize the advantage network after each training iteration
    /// The original DeepCFR paper notes that this typically performs better when true
    pub reinitialize_adv: bool,
    /// Number of actions in the game
    pub num_actions: usize,
}

impl Default for DeepCFRConfig {
    fn default() -> Self {
        Self {
            device: tch::Device::cuda_if_available(),
            policy_layers: vec![256, 256],
            advantage_layers: vec![128, 128],
            learning_rate: 1e-3,
            batch_size_policy: 2048,
            batch_size_adv: 2048,
            memory_capacity: 1e6 as usize,
            policy_train_steps: 5000,
            adv_train_steps: 750,
            reinitialize_adv: true,
            num_actions: 21,
        }
    }
}

pub struct DeepCFRAgent {
    /// Configuration for the agent
    config: DeepCFRConfig,
    /// Variable store for the policy network
    policy_vars: nn::VarStore,
    /// Variable store for the advantage network
    adv_vars: nn::VarStore,
    /// Policy network
    policy: PolicyNetwork,
    /// Advantage network
    advantage: AdvantageNetwork,
}

impl DeepCFRAgent {
    #[inline]
    fn get_obs_tensor(&self, state: &StockState) -> tch::Tensor {
        let obs = state.get_observation();
        let obs = tch::Tensor::try_from(obs).unwrap();
        obs.to(self.config.device)
    }
    #[inline]
    fn get_mask_tensor(&self, state: &StockState) -> tch::Tensor {
        let valid_actions = state.get_actions();
        let mask = tch::Tensor::zeros(
            &[1, state.all_actions.len() as i64],
            (Kind::Float, self.config.device),
        );
        for act in valid_actions {
            mask[0][act] = 1.0f32;
        }
        mask
    }

    fn get_matched_regrets(
        &self,
        obs: &tch::Tensor,
        mask: &tch::Tensor,
    ) -> (tch::Tensor, tch::Tensor) {
        let adv = self.advantage.forward_t(obs, mask, false);
        let advantages = adv.clamp_min(0.0f32);
        let regret_sum = advantages.sum(Kind::Float);
        let matched_regrets;
        if regret_sum > 0.0f32 {
            matched_regrets = advantages / regret_sum;
        } else {
            let adv_filled = adv.masked_fill(action_mask, -1e9f32);
            let best_idx = adv_filled.argmax(-1, false);
            matched_regrets = best_idx.one_hot(self.config.num_actions);
        }
        (advantages, matched_regrets)
    }

    fn sample_action_from_advantage(&self, state: &StockState) {
        let obs = self.get_obs_tensor(state);
        let mask = self.get_mask_tensor(state);
    }
}
