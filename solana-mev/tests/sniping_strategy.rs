use crate::bot::strategies::sniping_strategy::SnipingStrategy;
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;

#[tokio::test]
async fn test_sniping_strategy() {
    let rpc_client = solana_client::rpc_client::RpcClient::new("https://api.mainnet-beta.solana.com".to_string());
    let mut strategy = SnipingStrategy::new(rpc_client);
    
    let token_mint = Pubkey::new_unique();
    strategy.add_target_token(token_mint);
    strategy.set_max_price(10.0);
    strategy.set_min_liquidity(1000.0);
    
    let mut target_accounts = HashMap::new();
    target_accounts.insert(token_mint, crate::AccountInfo {
        token_balance: 5000.0,
        token_price: 8.0,
    });
    
    let opportunities = strategy.find_opportunities(&target_accounts).await;
    assert_eq!(opportunities.len(), 1);
    
    let opportunity = &opportunities[0];
    assert_eq!(opportunity.target_account, token_mint);
    assert_eq!(opportunity.token_mint, token_mint);
    assert_eq!(opportunity.expected_price, 8.0);
    assert_eq!(opportunity.token_balance, 5000.0);
}