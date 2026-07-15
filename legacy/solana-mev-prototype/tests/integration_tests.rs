use mev_bot_solana::bot::solana_mev_bot::SolanaMevBot;
use mev_bot_solana::config::Config;
use mev_bot_solana::dex::dex_manager::DexManager;
use mev_bot_solana::monitoring::metrics::Metrics;
use mev_bot_solana::strategies::copy_trade_strategy::CopyTradeStrategy;
use mev_bot_solana::strategies::sniping_strategy::SnipingStrategy;
use mev_bot_solana::utils::config_parser::parse_config;
use solana_client::rpc_client::RpcClient;
use solana_sdk::signature::read_keypair_file;
use std::sync::Arc;

#[tokio::test]
async fn test_mev_bot_integration() {
    let config = parse_config("config.toml").expect("Failed to parse config");

    let rpc_client = Arc::new(RpcClient::new_with_commitment(
        config.solana.rpc_url.clone(),
        config.solana.commitment.clone(),
    ));

    let metrics = Arc::new(Metrics::new());

    let dex_manager = Arc::new(tokio::sync::Mutex::new(DexManager::new(
        rpc_client.clone(),
        config.dexes.clone(),
    )));

    let sniping_strategy = Arc::new(tokio::sync::Mutex::new(SnipingStrategy::new(
        rpc_client.clone(),
        dex_manager.clone(),
        config.bot.max_position_size,
    )));

    let copy_trade_strategy = Arc::new(tokio::sync::Mutex::new(CopyTradeStrategy::new(
        rpc_client.clone(),
        dex_manager.clone(),
        config.bot.max_position_size,
    )));

    let authority_keypair = read_keypair_file(config.bot.keypair_path.clone())
        .expect("Failed to read keypair file");

    let mut mev_bot = SolanaMevBot::new(
        rpc_client,
        authority_keypair,
        vec![
            sniping_strategy.clone(),
            copy_trade_strategy.clone(),
        ],
        config.bot.profit_threshold,
        metrics,
    );

    tokio::spawn(async move {
        mev_bot.run().await;
    });

    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

    let orders = mev_bot.metrics.get_orders().await;
    assert!(!orders.is_empty());

    let profits = mev_bot.metrics.get_profits().await;
    assert!(!profits.is_empty());

    let volumes = mev_bot.metrics.get_volumes().await;
    assert!(!volumes.is_empty());
}