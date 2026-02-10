use crate::models::market::Market;

pub struct SnipingOpportunity {
    pub market: Market,
    pub price: f64,
    pub liquidity: f64,
}   