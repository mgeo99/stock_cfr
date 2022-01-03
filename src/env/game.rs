use std::{fs::File, path::Path};

use bimap::BiMap;
use chrono::NaiveDateTime;
use parquet::{
    file::{reader::FileReader, serialized_reader::SerializedFileReader},
    record::RowAccessor,
};

use super::types::{StockQuote, StockAction};

pub struct StockEnv {
    ticks: Vec<StockQuote>,
    all_actions: BiMap<StockAction, usize>
}

impl StockEnv {
    pub fn new(max_action_shares: usize) -> Self {
        let mut all_actions = BiMap::new();
        all_actions.insert(StockAction::Hold, 0);
        for i in 1..=max_action_shares {
            all_actions.insert(StockAction::Buy(i), all_actions.len());
            all_actions.insert(StockAction::Sell(i), all_actions.len());
        }

        Self { ticks: vec![], all_actions }
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
