use async_trait::async_trait;
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;

#[async_trait]
pub trait DexIntegration {
    async fn get_prices(&self) -> HashMap<String, f64>;
    async fn get_account_balances(&self, account: &Pubkey) -> HashMap<String, f64>;
    async fn place_order(&self, market: &str, side: &str, size: f64, price: f64) -> Option<String>;
    async fn cancel_order(&self, order_id: &str) -> bool;
}