use crate::{env::game::{StockEnv, StockEnvConfig}, agent::deep_cfr::{DeepCFRConfig, DeepCFRAgent}};

mod env;
mod agent;

fn main() {
    println!("Hello, world!");
    let env_config = StockEnvConfig::default();
    println!("{:?}", env_config);
    let mut env = StockEnv::new(env_config);
    env.load_price_data("aapl.parquet");


    let mut agent_config = DeepCFRConfig::default();
    agent_config.num_actions = env.num_actions();
    println!("{:?}", agent_config);

    let mut agent = DeepCFRAgent::new(agent_config);
    
    agent.train_agent(&env);
}
