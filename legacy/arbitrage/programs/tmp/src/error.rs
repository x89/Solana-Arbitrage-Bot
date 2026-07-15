use anchor_lang::prelude::*;

#[error_code]
pub enum ErrorCode {
    #[msg("No profit at the end. Reverting...")]
    NoProfit,
    #[msg("Trying to swap when information is invalid.")]
    InvalidState,
    #[msg("Not enough funds: amount_in > src_balance.")]
    NotEnoughFunds,
}

#[error_code]
pub enum RaydiumSwapError {
    #[msg("Invalid pool state")]
    InvalidPoolState,
    #[msg("Slippage tolerance exceeded")]
    SlippageExceeded,
}