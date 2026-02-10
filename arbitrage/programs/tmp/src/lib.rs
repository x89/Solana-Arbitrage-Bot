// File: src/lib.rs

use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount};

pub mod swaps;
pub mod state;
pub mod ix_data;

use crate::swaps::{
    meteora::*,
    raydium::*,
    orca::*,
    jupiter::*,
};

declare_id!("ArbitrageProgram11111111111111111111111111111111");

#[program]
pub mod arbitrage {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let swap_state = &mut ctx.accounts.swap_state;
        swap_state.owner = ctx.accounts.owner.key();
        swap_state.profit_threshold = 5000; // 0.5% in bps
        swap_state.active_routes = 0;
        Ok(())
    }

    // Two-Hop Arbitrage (Meteora -> Raydium)
    pub fn execute_two_hop_arbitrage(
        ctx: Context<TwoHopArbitrage>,
        amount_in: u64,
        minimum_intermediate_amount: u64,
        minimum_final_amount: u64,
    ) -> Result<()> {
        // First swap on Meteora
        _meteora_swap(
            &ctx.accounts.meteora_accounts,
            amount_in,
            minimum_intermediate_amount,
        )?;

        // Second swap on Raydium
        _raydium_swap(
            &ctx.accounts.raydium_accounts,
            minimum_intermediate_amount,
            minimum_final_amount,
        )?;

        // Verify profit
        let profit = minimum_final_amount.checked_sub(amount_in)
            .ok_or(ErrorCode::ArithmeticError)?;
        
        require!(
            profit >= ctx.accounts.swap_state.profit_threshold,
            ErrorCode::InsufficientProfit
        );

        Ok(())
    }

    // Triangle Arbitrage (Meteora -> Meteora -> Raydium)
    pub fn execute_triangle_arbitrage(
        ctx: Context<TriangleArbitrage>,
        amount_in: u64,
        minimum_amount_1: u64,
        minimum_amount_2: u64,
        minimum_final_amount: u64,
    ) -> Result<()> {
        // First Meteora swap
        _meteora_swap(
            &ctx.accounts.meteora_accounts_1,
            amount_in,
            minimum_amount_1,
        )?;

        // Second Meteora swap
        _meteora_swap(
            &ctx.accounts.meteora_accounts_2,
            minimum_amount_1,
            minimum_amount_2,
        )?;

        // Final Raydium swap
        _raydium_swap(
            &ctx.accounts.raydium_accounts,
            minimum_amount_2,
            minimum_final_amount,
        )?;

        // Verify profit
        let profit = minimum_final_amount.checked_sub(amount_in)
            .ok_or(ErrorCode::ArithmeticError)?;
        
        require!(
            profit >= ctx.accounts.swap_state.profit_threshold,
            ErrorCode::InsufficientProfit
        );

        Ok(())
    }

    // Multi-DEX Arbitrage (Orca -> Whirlpool -> Orca)
    pub fn execute_orca_whirlpool_arbitrage(
        ctx: Context<OrcaWhirlpoolArbitrage>,
        amount_in: u64,
        minimum_amount_1: u64,
        minimum_final_amount: u64,
    ) -> Result<()> {
        // First Orca swap
        _orca_swap(
            &ctx.accounts.orca_accounts_1,
            amount_in,
            minimum_amount_1,
        )?;

        // Whirlpool swap
        _orca_swap(
            &ctx.accounts.whirlpool_accounts,
            minimum_amount_1,
            minimum_final_amount,
        )?;

        // Verify profit
        let profit = minimum_final_amount.checked_sub(amount_in)
            .ok_or(ErrorCode::ArithmeticError)?;
        
        require!(
            profit >= ctx.accounts.swap_state.profit_threshold,
            ErrorCode::InsufficientProfit
        );

        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,
    #[account(
        init,
        payer = owner,
        space = 8 + 32 + 8 + 8,
        seeds = [b"swap_state"],
        bump
    )]
    pub swap_state: Account<'info, SwapState>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct TwoHopArbitrage<'info> {
    pub user: Signer<'info>,
    #[account(mut)]
    pub swap_state: Account<'info, SwapState>,
    pub meteora_accounts: MeteoraSwap<'info>,
    pub raydium_accounts: RaydiumSwap<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct TriangleArbitrage<'info> {
    pub user: Signer<'info>,
    #[account(mut)]
    pub swap_state: Account<'info, SwapState>,
    pub meteora_accounts_1: MeteoraSwap<'info>,
    pub meteora_accounts_2: MeteoraSwap<'info>,
    pub raydium_accounts: RaydiumSwap<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct OrcaWhirlpoolArbitrage<'info> {
    pub user: Signer<'info>,
    #[account(mut)]
    pub swap_state: Account<'info, SwapState>,
    pub orca_accounts_1: OrcaSwap<'info>,
    pub whirlpool_accounts: OrcaSwap<'info>,
    pub token_program: Program<'info, Token>,
}

#[account]
pub struct SwapState {
    pub owner: Pubkey,
    pub profit_threshold: u64,
    pub active_routes: u64,
}

#[error_code]
pub enum ErrorCode {
    #[msg("Insufficient profit margin")]
    InsufficientProfit,
    #[msg("Arithmetic error")]
    ArithmeticError,
    #[msg("Slippage tolerance exceeded")]
    SlippageExceeded,
}