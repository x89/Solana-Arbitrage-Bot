use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount};
use raydium_amm_v3::{cpi, program::AmmV3};

#[derive(Accounts)]
pub struct RaydiumSwap<'info> {
    pub amm_program: Program<'info, AmmV3>,
    pub pool_state: Account<'info, PoolState>,
    pub amm_config: Account<'info, AmmConfig>,
    #[account(mut)]
    pub user_source_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_destination_token: Account<'info, TokenAccount>,
    pub observation_state: Account<'info, ObservationState>,
    // Add other required accounts
}

pub fn _raydium_swap<'info>(
    ctx: &Context<'_, '_, '_, 'info, RaydiumSwap<'info>>,
    amount_in: u64,
    minimum_amount_out: u64,
) -> Result<()> {
    let cpi_program = ctx.accounts.amm_program.to_account_info();
    let cpi_accounts = cpi::accounts::Swap {
        pool_state: ctx.accounts.pool_state.to_account_info(),
        amm_config: ctx.accounts.amm_config.to_account_info(),
        token_program: ctx.accounts.token_program.to_account_info(),
        // Map other required accounts
    };

    let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);

    raydium_amm_v3::cpi::swap(
        cpi_ctx,
        amount_in,
        minimum_amount_out,
        sqrt_price_limit_x64,
        is_base_input,
    )
}
