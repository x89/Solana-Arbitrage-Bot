use anchor_lang::prelude::*;

#[error_code]
pub enum ErrorCode {
    #[msg("No Profit at the end. Reverting...")]
    NoProfit,
    #[msg("Trying to swap when Information is invalid.")]
    InvalidState,
    #[msg("not enough funds: amount_in > src_balance.")]
    NotEnoughFunds,
}