use crate::env::game::StockEnv;

mod env;
mod agent;

fn main() {
    println!("Hello, world!");
    let mut env = StockEnv::new(10);
    env.load_price_data("aapl.parquet");
}
