use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use solana_mev_bot::bot::solana_mev_bot::SolanaMevBot;
use solana_mev_bot::dex::{raydium, serum, orca};
use solana_mev_bot::strategies::{sniping_strategy, copy_trade_strategy};
use solana_mev_bot::utils::solana;

#[tokio::test]
async fn test_solana_mev_bot() {
    let rpc_url = "https://api.devnet.solana.com";
    let ws_url = "wss://api.devnet.solana.com";
    let payer_keypair = solana::load_keypair("path/to/keypair.json");
    let target_accounts = HashMap::new();
    let profit_threshold = 0.01;

    let rpc_client = Arc::new(solana_client::rpc_client::RpcClient::new_with_commitment(
        rpc_url.to_string(),
        solana_sdk::commitment_config::CommitmentConfig::confirmed(),
    ));

    let mut solana_mev_bot = SolanaMevBot::new(
        rpc_client.clone(),
        ws_url.to_string(),
        payer_keypair,
        target_accounts,
        profit_threshold,
        vec![
            Arc::new(Mutex::new(raydium::Raydium::new(rpc_client.clone()))),
            Arc::new(Mutex::new(serum::Serum::new(rpc_client.clone()))),
            Arc::new(Mutex::new(orca::Orca::new(rpc_client.clone()))),
        ],
        Arc::new(Mutex::new(sniping_strategy::SnipingStrategy::new(rpc_client.clone()))),
        Arc::new(Mutex::new(copy_trade_strategy::CopyTradeStrategy::new(rpc_client.clone()))),
    );

    let result = solana_mev_bot.run().await;
    assert!(result.is_ok());
}