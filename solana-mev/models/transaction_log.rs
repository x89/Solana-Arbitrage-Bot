use crate::models::market_conditions::MarketConditions;

pub struct TransactionLog {
    pub signature: String,
    pub market_conditions: MarketConditions,
}