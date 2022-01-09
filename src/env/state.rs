use bimap::BiMap;
use ndarray::Array1;

use super::types::{StockAction, StockQuote};

#[derive(Debug)]
pub struct StockState<'a> {
    /// Original tick data for the stock
    pub ticks: &'a [StockQuote],
    /// Current position in the simulation
    pub pos: usize,
    /// Amount of cash the player holds
    pub cash_amount: f32,
    /// Amount in assets the player holds
    pub asset_amount: f32,
    /// Current number of stocks held
    pub shares_held: f32,
    /// All available actions in the game
    pub all_actions: &'a BiMap<StockAction, usize>,
}

impl<'a> StockState<'a> {
    /// Gets the observation as an array
    pub fn get_observation(&self) -> Array1<f32> {
        let mut obs = Array1::zeros(7);
        // TODO: Actual feature engineering and relative change
        obs[0] = self.cash_amount;
        obs[1] = self.asset_amount;
        obs[2] = self.ticks[self.pos].open;
        obs[3] = self.ticks[self.pos].close;
        obs[4] = self.ticks[self.pos].high;
        obs[5] = self.ticks[self.pos].low;
        obs[6] = self.ticks[self.pos].volume;

        obs
    }

    /// Gets the valid actions in this state
    pub fn get_actions(&self) -> Vec<usize> {
        let mut valid_actions = vec![];
        for (act, &id) in self.all_actions.iter() {
            match act {
                &StockAction::Hold => valid_actions.push(id),
                &StockAction::Buy(shares) => {
                    let cost = self.ticks[self.pos - 1].close * (shares as f32);
                    if cost < self.cash_amount {
                        valid_actions.push(id)
                    }
                }
                &StockAction::Sell(shares) => {
                    if (shares as f32) < self.shares_held {
                        valid_actions.push(id)
                    }
                }
            }
        }

        valid_actions
    }

    /// Computes a mask for all valid actions
    pub fn get_action_mask(&self) -> Array1<f32> {
        let mut mask = Array1::zeros([self.all_actions.len()]);
        self.get_actions()
            .into_iter()
            .for_each(|i| mask[i] = 1.0f32);
        mask
    }

    pub fn transition(&self, action_id: usize) -> StockState<'a> {
        // Mutate portfolio value depending on the action at the current price
        let next_price = self.ticks[self.pos].close;
        let prev_price = self.ticks[self.pos - 1].close;
        let next_shares_held = match self.all_actions.get_by_right(&action_id).unwrap() {
            &StockAction::Buy(shares) => shares as f32 + self.shares_held,
            &StockAction::Hold => self.shares_held,
            &StockAction::Sell(shares) => self.shares_held - shares as f32,
        };
        let next_cash_amount = match self.all_actions.get_by_right(&action_id).unwrap() {
            &StockAction::Buy(shares) => self.cash_amount - (shares as f32 * next_price),
            &StockAction::Hold => self.cash_amount,
            &StockAction::Sell(shares) => self.cash_amount + (shares as f32 * next_price),
        };
        let next_asset_amount = next_shares_held as f32 * next_price;
        // println!("======= Transition ======");
        // println!("Shares: {}", next_shares_held);
        // println!("Cash: {}", next_cash_amount);
        // println!("Assets: {}", next_asset_amount);
        Self {
            all_actions: self.all_actions,
            asset_amount: next_asset_amount,
            cash_amount: next_cash_amount,
            pos: self.pos + 1,
            shares_held: next_shares_held,
            ticks: self.ticks,
        }
    }

    pub fn is_terminal(&self) -> bool {
        // Out of ticker data
        if self.pos == self.ticks.len() - 1 {
            return true;
        }

        // Ran out of money and have no shares to sell
        if self.shares_held == 0.0 && self.cash_amount <= 0.0 {
            return true;
        }

        false
    }

    pub fn get_rewards(&self) -> f32 {
        // TODO: Make this more robust (i.e., relative to the initial value)
        let portfolio_value = self.asset_amount + self.cash_amount;
        portfolio_value
    }
}
