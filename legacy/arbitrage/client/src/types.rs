use std::fmt;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DexType {
    Meteora,
    Raydium,
    Orca,
    Whirlpool,
}

impl fmt::Display for DexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DexType::Meteora => write!(f, "Meteora"),
            DexType::Raydium => write!(f, "Raydium"),
            DexType::Orca => write!(f, "Orca"),
            DexType::Whirlpool => write!(f, "Whirlpool"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteType {
    TwoHop,
    Triangle,
    OrcaWhirlpool,
}

#[derive(Debug, Clone)]
pub struct PriceData {
    pub dex_type: DexType,
    pub token_in: String,
    pub token_out: String,
    pub amount_in: f64,
    pub amount_out: f64,
    pub price: f64,
    pub liquidity: f64,
    pub timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub route_type: RouteType,
    pub amount_in: f64,
    pub token_in: String,
    pub intermediate_amounts: Vec<f64>,
    pub expected_output: f64,
    pub profit_ratio: f64,
    pub steps: Vec<PriceData>,
}

#[derive(Debug, Clone)]
pub struct SwapStep {
    pub dex_type: DexType,
    pub token_in: String,
    pub token_out: String,
    pub amount_in: f64,
    pub minimum_amount_out: f64,
    pub pool_address: String,
}

#[derive(Debug)]
pub struct Route {
    pub steps: Vec<SwapStep>,
    pub expected_profit: f64,
    pub total_fee: f64,
}

// Constants for token addresses and DEX program IDs
pub const RAYDIUM_PROGRAM_ID: &str = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8";
pub const ORCA_WHIRLPOOL_PROGRAM_ID: &str = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc";
pub const METEORA_PROGRAM_ID: &str = "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K";

// Common token addresses
pub const USDC_MINT: &str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";
pub const WSOL_MINT: &str = "So11111111111111111111111111111111111111112"; 