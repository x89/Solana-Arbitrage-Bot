// File: src/lib.rs

use anchor_lang::prelude::*;
use crate::state::RaydiumSwapState;

pub mod ix_data;
pub mod state;
pub mod swaps;
pub mod error;

use swaps::orca::*;
use swaps::raydium::*;
use swaps::meteora::*;
use swaps::jupiter::*;

declare_id!("Fg6PaFpoGXkYsidMpWTK6W2BeZ7FEfcYkg476zPFsLnS");

#[program]
pub mod tmp {
    use super::*;

    pub fn initialize_raydium_swap_state(ctx: Context<InitializeRaydiumSwapState>) -> Result<()> {
        let swap_state = &mut ctx.accounts.swap_state;
        swap_state.bump = *ctx.bumps.get("swap_state").unwrap();
        swap_state.authority = ctx.accounts.authority.key();
        swap_state.initialized = true;
        swap_state.last_swap_timestamp = Clock::get()?.unix_timestamp;
        Ok(())
    }

    pub fn orca_swap(
        ctx: Context<OrcaSwap>,
        amount_in: u64,
        minimum_amount_out: u64,
    ) -> Result<()> {
        ctx.accounts.process_swap(amount_in, minimum_amount_out)
    }

    pub fn raydium_swap(
        ctx: Context<RaydiumSwap>,
        amount_in: u64,
        minimum_amount_out: u64,
    ) -> Result<()> {
        ctx.accounts.process_swap(amount_in, minimum_amount_out)
    }

    pub fn meteora_swap(
        ctx: Context<MeteoraSwap>,
        amount_in: u64,
        minimum_amount_out: u64,
    ) -> Result<()> {
        ctx.accounts.process_swap(amount_in, minimum_amount_out)
    }

    pub fn jupiter_swap(
        ctx: Context<JupiterSwap>,
        amount_in: u64,
        minimum_amount_out: u64,
        route_data: Vec<u8>,
    ) -> Result<()> {
        ctx.accounts.process_swap(amount_in, minimum_amount_out, route_data)
    }
}

#[derive(Accounts)]
pub struct InitializeRaydiumSwapState<'info> {
    #[account(
        init,
        payer = authority,
        space = RaydiumSwapState::LEN,
        seeds = [b"raydium_swap_state", authority.key().as_ref()],
        bump
    )]
    pub swap_state: Account<'info, RaydiumSwapState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}
