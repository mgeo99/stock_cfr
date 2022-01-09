use std::{fs::File, ops::Range, path::Path};

use super::{
    state::StockState,
    types::{StockAction, StockQuote},
};
use bimap::BiMap;
use chrono::NaiveDateTime;
use parquet::{
    file::{reader::FileReader, serialized_reader::SerializedFileReader},
    record::RowAccessor,
};
use rand::prelude::*;

#[derive(Debug)]
pub struct StockEnvConfig {
    /// Range of cash amounts to start each game with
    pub cash_amount_range: Range<f32>,
    /// The action space is divided up to buy/sell X number of shares in this range
    pub share_action_range: Range<usize>,
    /// Max number of steps before the environment terminates
    pub max_steps: usize,
}

impl Default for StockEnvConfig {
    fn default() -> Self {
        Self {
            cash_amount_range: 10000.0..50000.0,
            share_action_range: 1..10,
            max_steps: 100,
        }
    }
}

pub struct StockEnv {
    ticks: Vec<StockQuote>,
    config: StockEnvConfig,
    all_actions: BiMap<StockAction, usize>,
}

impl StockEnv {
    pub fn new(config: StockEnvConfig) -> Self {
        let mut all_actions = BiMap::new();
        all_actions.insert(StockAction::Hold, 0);
        for i in config.share_action_range.start..=config.share_action_range.end {
            all_actions.insert(StockAction::Buy(i), all_actions.len());
            all_actions.insert(StockAction::Sell(i), all_actions.len());
        }

        Self {
            ticks: vec![],
            all_actions,
            config,
        }
    }

    pub fn num_actions(&self) -> usize {
        self.all_actions.len()
    }

    pub fn start(&self) -> StockState {
        let mut rng = rand::thread_rng();
        let start_pos = rng.gen_range(0..self.ticks.len() - self.config.max_steps);
        StockState {
            all_actions: &self.all_actions,
            asset_amount: 0.0,
            cash_amount: rng.gen_range(self.config.cash_amount_range.clone()),
            pos: 1,
            shares_held: 0.0,
            ticks: &self.ticks[start_pos..start_pos + self.config.max_steps],
        }
    }

    pub fn load_price_data<P: AsRef<Path>>(&mut self, path: P) {
        let mut stock_data = Vec::new();
        let file = File::open(path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let schema = reader.metadata().file_metadata().schema();
        let fields = schema.get_fields();

        let timestamp_field = fields.iter().position(|x| x.name() == "date").unwrap();
        let open_field = fields.iter().position(|x| x.name() == "open").unwrap();
        let low_field = fields.iter().position(|x| x.name() == "low").unwrap();
        let high_field = fields.iter().position(|x| x.name() == "high").unwrap();
        let close_field = fields.iter().position(|x| x.name() == "close").unwrap();
        let volume_field = fields.iter().position(|x| x.name() == "volume").unwrap();
        let symbol_field = fields.iter().position(|x| x.name() == "symbol").unwrap();

        let mut iter = reader.get_row_iter(None).unwrap();
        while let Some(record) = iter.next() {
            let timestamp = record.get_string(timestamp_field).unwrap();
            let timestamp = NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%d %H:%M:%S").unwrap();
            let tick = StockQuote {
                timestamp: timestamp.into(),
                open: record.get_double(open_field).unwrap() as f32,
                low: record.get_double(low_field).unwrap() as f32,
                high: record.get_double(high_field).unwrap() as f32,
                close: record.get_double(close_field).unwrap() as f32,
                volume: record.get_long(volume_field).unwrap() as f32,
                symbol: record.get_string(symbol_field).unwrap().to_owned(),
            };

            stock_data.push(tick);
        }
        println!("Read {} records", stock_data.len());
        self.ticks = stock_data;
    }
}
