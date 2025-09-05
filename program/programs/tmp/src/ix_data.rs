use anchor_lang::prelude::*;

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct SwapData {
    pub instruction: u8, 
    pub amount_in: u64, 
    pub minimum_amount_out: u64, 
}
