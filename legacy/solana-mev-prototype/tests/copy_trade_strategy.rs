use crate::bot::strategies::copy_trade_strategy::CopyTradeStrategy;
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;

#[tokio::test]
async fn test_copy_trade_strategy() {
    let rpc_client = solana_client::rpc_client::RpcClient::new("https://api.mainnet-beta.solana.com".to_string());
    let mut strategy = CopyTradeStrategy::new(rpc_client);
    
    strategy.set_trade_threshold(1000.0);
    strategy.set_max_trade_amount(10000.0);
    
    let trader_account = Pubkey::new_unique();
    let mut target_accounts = HashMap::new();
    target_accounts.insert(trader_account, crate::AccountInfo {
        token_balance: 0.0,
        token_price: 0.0,
    });
    
    let targets = strategy.find_opportunities(&target_accounts).await;
    assert_eq!(targets.len(), 1);
    
    let target = &targets[0];
    assert_eq!(target.trader_account, trader_account);
    assert_eq!(target.trade_amount, 5000.0);
}