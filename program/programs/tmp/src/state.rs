use anchor_lang::prelude::*;

#[account]
#[derive(Default)]
pub struct SwapState { 
    pub start_balance: u64, // start of swap balance
    pub swap_input: u64,  // output of swap 
    pub is_valid: bool, // saftey 
}
