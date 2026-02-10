use solana_sdk::pubkey::Pubkey;

pub struct CopyTradeTarget {
    pub trader_account: Pubkey,
    pub target_token: Pubkey,
    pub trade_amount: f64,
}