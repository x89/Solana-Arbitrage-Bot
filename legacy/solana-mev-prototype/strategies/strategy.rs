use crate::models::market_conditions::MarketConditions;
use crate::models::mev_opportunity::MevOpportunity;
use async_trait::async_trait;
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;

#[async_trait]
pub trait Strategy {
    fn update(&mut self, market_conditions: &MarketConditions);
    async fn find_opportunities(&self, target_accounts: &HashMap<Pubkey, crate::AccountInfo>) -> Vec<MevOpportunity>;
}