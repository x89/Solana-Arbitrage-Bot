use crate::models::market_conditions::MarketConditions;
use crate::models::mev_opportunity::MevOpportunity;
use crate::strategies::strategy::Strategy;
use async_trait::async_trait;
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;

pub struct SandwichStrategy {}

#[async_trait]
impl Strategy for SandwichStrategy {
    fn update(&mut self, _market_conditions: &MarketConditions) {}

    async fn find_opportunities(&self, _target_accounts: &HashMap<Pubkey, crate::AccountInfo>) -> Vec<MevOpportunity> {
        Vec::new()
    }
}