use crate::dex::dex_integration::DexIntegration;
use crate::models::market_conditions::MarketConditions;
use crate::models::mev_opportunity::MevOpportunity;
use crate::strategies::strategy::Strategy;
use solana_client::rpc_client::RpcClient;
use std::collections::HashMap;
use solana_sdk::pubkey::Pubkey;

pub struct StrategyManager {
    rpc_client: RpcClient,
    strategies: Vec<Box<dyn Strategy>>,
    dex_integrations: Vec<Box<dyn DexIntegration>>,
}

impl StrategyManager {
    pub fn new(rpc_client: RpcClient, dex_integrations: Vec<Box<dyn DexIntegration>>) -> Self {
        Self {
            rpc_client,
            strategies: Vec::new(),
            dex_integrations,
        }
    }

    pub fn update(&mut self, market_conditions: &MarketConditions) {
        for strategy in &mut self.strategies {
            strategy.update(market_conditions);
        }
    }

    pub fn add_strategy(&mut self, strategy: impl Strategy + 'static) {
        self.strategies.push(Box::new(strategy));
    }

    pub async fn find_opportunities(&self, target_accounts: &HashMap<Pubkey, crate::AccountInfo>) -> Vec<MevOpportunity> {
        let mut opportunities = Vec::new();
        for strategy in &self.strategies {
            let strategy_opportunities = strategy.find_opportunities(target_accounts).await;
            opportunities.extend(strategy_opportunities);
        }
        opportunities
    }
}