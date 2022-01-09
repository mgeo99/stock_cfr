use ndarray::Array1;
use ndarray_rand::rand_distr::{Distribution, WeightedIndex};
use tch::{
    nn::{self, OptimizerConfig},
    Kind, Scalar,
};

use crate::{
    agent::buffer::BufferEntry,
    env::{game::StockEnv, state::StockState},
};

use super::{
    advantage::AdvantageNetwork, buffer::ReservoirBuffer, dataset::ReservoirDataset,
    policy::PolicyNetwork,
};

#[derive(Debug)]
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
    /// Observation tensor length
    pub obs_tensor_size: usize,
    /// Number of traversals to collect data for the advantage network
    pub num_traversals: usize,
    /// Number of iterations where we do a full update of the advantage network
    pub num_iterations: usize,
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
            num_traversals: 100,
            num_iterations: 100,
            obs_tensor_size: 7,
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
    /// Advantage memory buffer
    advantage_memory: ReservoirBuffer,
    /// Policy memory buffer
    policy_memory: ReservoirBuffer,
    /// Current iteration across all solves
    curr_iter: usize,
}

impl DeepCFRAgent {
    pub fn new(config: DeepCFRConfig) -> Self {
        let policy_vars = nn::VarStore::new(config.device);
        let adv_vars = nn::VarStore::new(config.device);
        let policy = PolicyNetwork::new(
            &policy_vars.root(),
            config.obs_tensor_size as i64,
            config.policy_layers.clone(),
            config.num_actions,
        );
        let advantage = AdvantageNetwork::new(
            &adv_vars.root(),
            config.obs_tensor_size as i64,
            config.advantage_layers.clone(),
            config.num_actions,
        );
        let advantage_memory = ReservoirBuffer::new(config.memory_capacity);
        let policy_memory = ReservoirBuffer::new(config.memory_capacity);

        Self {
            adv_vars,
            advantage,
            advantage_memory,
            config,
            curr_iter: 0,
            policy,
            policy_memory,
            policy_vars,
        }
    }
    pub fn train_agent(&mut self, env: &StockEnv) {
        for i in 1..=self.config.num_iterations {
            self.curr_iter = i;
            println!("Iteration: {}", i);
            // Run a set number of traversals through the game initializing a new state each time
            for _ in 0..self.config.num_traversals {
                let init_state = env.start();
                self.traverse_game_tree(&init_state);
            }
            // if self.config.reinitialize_adv {
            //     // TODO: Re-Initialize the advantage network to account for
            // }
            let adv_loss = self.update_advantage_net();
            println!("\tAdvantage Loss: {}", adv_loss);
        }

        // Once we finish all the network iterations, train the final policy network
        self.update_policy_net();
    }

    #[inline]
    fn get_obs_tensor(&self, state: &StockState) -> tch::Tensor {
        let obs = state.get_observation();
        let obs = tch::Tensor::try_from(obs).unwrap();
        obs.to(self.config.device)
    }
    #[inline]
    fn get_mask_tensor(&self, state: &StockState) -> tch::Tensor {
        let mask = state.get_action_mask();
        let mask = tch::Tensor::try_from(mask).unwrap();
        mask.unsqueeze(0)
            .to_kind(Kind::Float)
            .to(self.config.device)
    }

    fn get_matched_regrets(
        &self,
        obs: &tch::Tensor,
        mask: &tch::Tensor,
    ) -> (tch::Tensor, tch::Tensor) {
        let adv = self.advantage.forward_t(obs, mask, false);
        let advantages = adv.clamp_min(Scalar::float(0.0));
        let regret_sum = advantages.sum(Kind::Float);
        let matched_regrets;
        if f32::try_from(&regret_sum).unwrap() > 0.0f32 {
            matched_regrets = &advantages / &regret_sum;
        } else {
            let adv_filled = adv.masked_fill(mask, Scalar::float(-1e9));
            let best_idx = adv_filled.argmax(-1, false);
            matched_regrets = best_idx.one_hot(self.config.num_actions as i64);
        }
        (advantages, matched_regrets)
    }

    fn sample_action_from_advantage(&self, state: &StockState) -> (Array1<f32>, Array1<f32>) {
        let obs = self.get_obs_tensor(state);
        let mask = self.get_mask_tensor(state);
        let (adv, matched_regrets) = self.get_matched_regrets(&obs, &mask);
        let adv = adv.view(-1);
        let matched_regrets = matched_regrets.view(-1);
        let adv = Array1::from_vec(adv.into());
        let regrets = Array1::from_vec(matched_regrets.into());
        (adv, regrets)
    }

    fn traverse_game_tree(&mut self, state: &StockState) -> f32 {
        if state.is_terminal() {
            return state.get_rewards();
        }

        // Normal CFR will alternate between players, but in this case we treat the market itself as a player
        // so just iterate over all possible actions and compute child values for each regret.
        // We can then just perform regret matching

        let (_, strategy) = self.sample_action_from_advantage(state);
        let mut expected_payoff = Array1::<f32>::zeros([strategy.shape()[0]]);

        // Go through legal actions and compute expected payoffs for each.
        // Using this information we can then determine the expected value of our current strategy in
        // this state
        let valid_actions = state.get_actions();
        for &act in valid_actions.iter() {
            let next_state = state.transition(act);
            expected_payoff[act] = self.traverse_game_tree(&next_state);
        }
        let expected_value = (&expected_payoff * &strategy).sum();
        let action_mask = state.get_action_mask();
        let sampled_regret = (&expected_payoff - expected_value) * &action_mask;
        self.advantage_memory.add(BufferEntry {
            info_state: state.get_observation(),
            data: sampled_regret,
            action_mask: action_mask.clone(),
            step: self.curr_iter,
        });

        // TODO: See if its valid to sample a strategy update on each turn as well and use that for the reservoir buffer
        let probs = &strategy / strategy.sum();
        let dist = WeightedIndex::new(probs).unwrap();
        let mut rng = rand::thread_rng();
        let sampled_action = dist.sample(&mut rng);

        // Do we really need to traverse the game tree again or just apply a multiplier to the value that was
        // already computed above?
        //let sample_state = state.transition(sampled_action);
        let sampled_value = expected_payoff[sampled_action]; //self.traverse_game_tree(&sample_state);
        self.policy_memory.add(BufferEntry {
            info_state: state.get_observation(),
            data: strategy,
            action_mask,
            step: self.curr_iter,
        });

        // Expected value of all actions given the strategy vs the sampled action
        expected_value + sampled_value
    }

    fn update_advantage_net(&self) -> f32 {
        let mut dataset = ReservoirDataset::new(
            &self.advantage_memory,
            self.config.device,
            self.config.adv_train_steps,
            self.config.batch_size_adv,
        );
        let mut opt = nn::AdamW::default()
            .build(&self.adv_vars, self.config.learning_rate as f64)
            .unwrap();
        let curr_iter = tch::Tensor::of_slice(&[self.curr_iter as f32]).to(self.config.device);
        let mut overall_loss = 1e9f32;
        while let Some(batch) = dataset.next_batch() {
            let preds = self
                .advantage
                .forward_t(&batch.info_states, &batch.action_masks, true);
            // Compute all the losses, but weight them based on the step that observation was taken at. This is so that we can balance updates
            // based on how recent the data is
            let losses = preds.mse_loss(&batch.observations, tch::Reduction::None);
            let weights = batch.steps * Scalar::float(2.0) / &curr_iter;
            let loss = (losses * weights).mean(Kind::Float);
            opt.backward_step(&loss);
            overall_loss = f32::try_from(loss).unwrap();
        }
        overall_loss
    }

    fn update_policy_net(&self) -> f32 {
        let mut dataset = ReservoirDataset::new(
            &self.policy_memory,
            self.config.device,
            self.config.policy_train_steps,
            self.config.batch_size_policy,
        );
        let mut opt = nn::AdamW::default()
            .build(&self.policy_vars, self.config.learning_rate as f64)
            .unwrap();
        let curr_iter = tch::Tensor::of_slice(&[self.curr_iter as f32]).to(self.config.device);
        let mut overall_loss = 1e9f32;
        while let Some(batch) = dataset.next_batch() {
            let preds = self
                .policy
                .forward_t(&batch.info_states, &batch.action_masks, true);
            // Compute all the losses, but weight them based on the step that observation was taken at. This is so that we can balance updates
            // based on how recent the data is
            let losses = preds.mse_loss(&batch.observations, tch::Reduction::None);
            let weights = batch.steps * Scalar::float(2.0) / &curr_iter;
            let loss = (losses * weights).mean(Kind::Float);
            opt.backward_step(&loss);
            overall_loss = f32::try_from(loss).unwrap();
        }
        overall_loss
    }
}
