use crate::models::market::Market;
use crate::models::order::Order;
use solana_sdk::pubkey::Pubkey;

pub struct CopyTradeOpportunity {
    pub trader: Pubkey,
    pub market: Market,
    pub trade: Order,
}