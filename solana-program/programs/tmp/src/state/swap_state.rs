use anchor_lang::prelude::*;

#[account]
pub struct RaydiumSwapState {
    pub bump: u8,
    pub authority: Pubkey,
    pub initialized: bool,
    pub last_swap_timestamp: i64,
}

impl RaydiumSwapState {
    pub const LEN: usize = 8 + // discriminator
        1 + // bump
        32 + // authority
        1 + // initialized
        8; // last_swap_timestamp
}