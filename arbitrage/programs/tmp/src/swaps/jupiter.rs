use anchor_lang::prelude::*;
use anchor_lang::solana_program::instruction::Instruction;
use anchor_spl::token::TokenAccount;

// Import necessary types from jupiter-cpi crate
use jupiter_cpi::{
    JupiterIx, JupiterIxParams, JupiterIxResult, Route, SwapMode,
};

pub fn jupiter_swap<'info>(
    ctx: Context<'_, '_, '_, 'info, JupiterSwap<'info>>,
    amount_in: u64,
    minimum_amount_out: u64,
) -> Result<()> {
    // Create JupiterIxParams
    let params = JupiterIxParams {
        amount_in,
        minimum_amount_out,
        route: ctx.accounts.route.clone(),
        swap_mode: SwapMode::ExactIn,
        platform_fee: None, // Set platform fee if needed
        token_ledger: None, // Set token ledger if needed
    };

    // Create Jupiter instruction
    let jupiter_ix = JupiterIx::new(params)?;

    // Convert Jupiter instruction to Solana instruction
    let ix: Instruction = jupiter_ix.into();

    // Invoke the Jupiter swap instruction
    anchor_lang::solana_program::program::invoke(
        &ix,
        &[
            ctx.accounts.jupiter_program.to_account_info(),
            ctx.accounts.token_program.to_account_info(),
            ctx.accounts.user_authority.to_account_info(),
            ctx.accounts.user_source_token.to_account_info(),
            ctx.accounts.user_destination_token.to_account_info(),
            // Add other required accounts based on the route
        ],
    )?;

    Ok(())
}

#[derive(Accounts)]
pub struct JupiterSwap<'info> {
    pub jupiter_program: AccountInfo<'info>,
    pub token_program: AccountInfo<'info>,
    pub user_authority: Signer<'info>,
    #[account(mut)]
    pub user_source_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_destination_token: Account<'info, TokenAccount>,
    // Add other required accounts based on the route
    pub route: Route,
}