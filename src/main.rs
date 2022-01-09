use crate::{
    agent::deep_cfr::{DeepCFRAgent, DeepCFRConfig},
    env::game::{StockEnv, StockEnvConfig},
};

mod agent;
mod env;

extern crate torch_sys;

fn main() {
    // Hack to make sure CUDA can be found on Windows
    unsafe { torch_sys::dummy_cuda_dependency() };

    let env_config = StockEnvConfig {
        cash_amount_range: 10000.0..25000.0,
        share_action_range: 1..3,
        max_steps: 10,
    };
    println!("{:?}", env_config);
    let mut env = StockEnv::new(env_config);
    env.load_price_data("aapl.parquet");

    let mut agent_config = DeepCFRConfig::default();
    agent_config.num_actions = env.num_actions();
    println!("{:?}", agent_config);

    let mut agent = DeepCFRAgent::new(agent_config);

    agent.train_agent(&env);

    // Now simulate the environment
    let mut state = env.start();
    let initial_cash = state.cash_amount;
    println!("Starting Simulation");
    while !state.is_terminal() {
        let action = agent.sample_action(&state);
        println!("{:?}", action);
        state = state.transition(action.selected);
    }
    let rew = state.get_rewards();
    println!("Simulation Ended");
    println!("\tInitial Cash: {}", initial_cash);
    println!("\tCash: {}", state.cash_amount);
    println!("\tAssets: {}", state.asset_amount);
    println!("\tShares: {}", state.shares_held);
    println!("\tPortfolio Value: {}", rew);
    println!("\tReturn Pct: {}", rew / initial_cash);
    println!("{:?}", state);
}
