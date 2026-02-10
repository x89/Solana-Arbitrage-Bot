// File: program/src/state.rs

use anchor_lang::prelude::*;

#[account]
#[derive(Default)]
pub struct ArbitrageState {
    pub authority: Pubkey,
    pub total_profit: u64,
    pub total_trades: u64,
}

#[account]
#[derive(Default)]
pub struct SwapState {
    pub start_balance: u64,
    pub swap_input: u64,
    pub is_valid: bool,
    pub input_token: Pubkey,
    pub current_token: Pubkey,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct RoutePlan {
    pub steps: Vec<ArbitrageStep>,
    pub input_token: Pubkey,
    pub output_token: Pubkey,
    pub minimum_output_amount: u64,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub enum ArbitrageStep {
    Orca(u64),
    Raydium(u64),
    Meteora(u64),
    Jupiter(u64),
}

impl ArbitrageStep {
    pub fn get_output_token(&self) -> Pubkey {
        // Implement logic to return the output token for each DEX
        // This is a placeholder and should be replaced with actual implementation
        Pubkey::default()
    }
}