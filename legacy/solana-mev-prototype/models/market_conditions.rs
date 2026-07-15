use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;

pub struct MarketConditions {
    pub liquidity: f64,
    pub volume: f64,
    pub volatility: f64,
    pub account_balances: HashMap<Pubkey, f64>,
}