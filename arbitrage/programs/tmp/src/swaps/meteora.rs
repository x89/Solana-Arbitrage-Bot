use anchor_lang::prelude::*;
use anchor_spl::token::TokenAccount;
use solana_program::instruction::{AccountMeta, Instruction};
use crate::state::SwapState;

pub fn _meteora_swap<'info>(
    ctx: &Context<'_, '_, '_, 'info, MeteoraSwap<'info>>,
    amount_in: u64,
    minimum_amount_out: u64,
) -> Result<()> {
    let ix = meteora_sdk::instruction::swap(
        ctx.accounts.pool.key,
        ctx.accounts.pool_signer.key,
        ctx.accounts.input_token_account.key,
        ctx.accounts.output_token_account.key,
        ctx.accounts.input_vault.key,
        ctx.accounts.output_vault.key,
        ctx.accounts.user_input_token_account.key,
        ctx.accounts.user_output_token_account.key,
        ctx.accounts.user_authority.key,
        amount_in,
        minimum_amount_out,
    )?;

    solana_program::program::invoke(
        &ix,
        &[
            ctx.accounts.pool.to_account_info(),
            ctx.accounts.pool_signer.to_account_info(),
            ctx.accounts.input_token_account.to_account_info(),
            ctx.accounts.output_token_account.to_account_info(),
            ctx.accounts.input_vault.to_account_info(),
            ctx.accounts.output_vault.to_account_info(),
            ctx.accounts.user_input_token_account.to_account_info(),
            ctx.accounts.user_output_token_account.to_account_info(),
            ctx.accounts.user_authority.to_account_info(),
            ctx.accounts.token_program.to_account_info(),
        ],
    )?;

    Ok(())
}

#[derive(Accounts)]
pub struct MeteoraSwap<'info> {
    /// CHECK: This is not dangerous because we don't read or write from this account
    pub pool: AccountInfo<'info>,
    /// CHECK: This is not dangerous because we don't read or write from this account
    pub pool_signer: AccountInfo<'info>,
    /// CHECK: This is not dangerous because we don't read or write from this account
    pub input_token_account: AccountInfo<'info>,
    /// CHECK: This is not dangerous because we don't read or write from this account
    pub output_token_account: AccountInfo<'info>,
    /// CHECK: This is not dangerous because we don't read or write from this account
    pub input_vault: AccountInfo<'info>,
    /// CHECK: This is not dangerous because we don't read or write from this account
    pub output_vault: AccountInfo<'info>,
    #[account(mut)]
    pub user_input_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_output_token_account: Account<'info, TokenAccount>,
    pub user_authority: Signer<'info>,
    /// CHECK: This is not dangerous because we don't read or write from this account
    pub token_program: AccountInfo<'info>,
    #[account(mut)]
    pub swap_state: Account<'info, SwapState>,
}