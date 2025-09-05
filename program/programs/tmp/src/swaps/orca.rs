use anchor_lang::prelude::*;
use solana_program::instruction::{AccountMeta, Instruction};
use solana_program;
use anchor_spl::{
    token::{TokenAccount}
};
use anchor_lang::{Accounts};
use crate::ix_data::SwapData;
use crate::state::SwapState;

pub fn _orca_swap<'info>(
    ctx: &Context<'_, '_, '_, 'info, OrcaSwap<'info>>, 
    amount_in: u64
) -> Result<()> {
    
    let data = SwapData {
        instruction: 1, // swap instruction 
        amount_in: amount_in,
        minimum_amount_out: 0, // no saftey lmfao 
    };

    let ix_accounts = vec![
        AccountMeta::new_readonly(*ctx.accounts.token_swap.key, false),
        AccountMeta::new_readonly(*ctx.accounts.authority.key, false),
        AccountMeta::new_readonly(*ctx.accounts.user_transfer_authority.key, true),
        
        AccountMeta::new(ctx.accounts.user_src.key(), false),
        AccountMeta::new(*ctx.accounts.pool_src.key, false),
        AccountMeta::new(*ctx.accounts.pool_dst.key, false),
        AccountMeta::new(ctx.accounts.user_dst.key(), false),
        AccountMeta::new(*ctx.accounts.pool_mint.key, false),
        AccountMeta::new(*ctx.accounts.fee_account.key, false),
      
        AccountMeta::new_readonly(*ctx.accounts.token_program.key, false),
    ];

    let instruction = Instruction {
        program_id: *ctx.accounts.token_swap_program.key,
        accounts: ix_accounts,
        data: data.try_to_vec()?,
    };

    let accounts = vec![
        ctx.accounts.token_swap.to_account_info(),
        ctx.accounts.authority.to_account_info(),
        ctx.accounts.user_transfer_authority.to_account_info(),
        ctx.accounts.user_src.to_account_info(),
        ctx.accounts.pool_src.to_account_info(),
        ctx.accounts.pool_dst.to_account_info(),
        ctx.accounts.user_dst.to_account_info(),
        ctx.accounts.pool_mint.to_account_info(),
        ctx.accounts.fee_account.to_account_info(),
        ctx.accounts.token_program.to_account_info(),
        ctx.accounts.token_swap_program.to_account_info(),
    ];

    solana_program::program::invoke(
        &instruction, 
        &accounts, 
    )?;
    
    Ok(())
}

#[derive(Accounts)]
pub struct OrcaSwap<'info> {
    pub token_swap: AccountInfo<'info>,
    pub authority: AccountInfo<'info>,
    pub user_transfer_authority : Signer<'info>,
    #[account(mut)]
    pub user_src: Account<'info, TokenAccount>,
    #[account(mut)]
    pub pool_src: AccountInfo<'info>,
    #[account(mut)]
    pub pool_dst: AccountInfo<'info>,
    #[account(mut)]
    pub user_dst: Account<'info, TokenAccount>,
    #[account(mut)]
    pub pool_mint: AccountInfo<'info>,
    #[account(mut)]
    pub fee_account: AccountInfo<'info>,
    pub token_program: AccountInfo<'info>,
    pub token_swap_program: AccountInfo<'info>,
    #[account(mut, seeds=[b"swap_state"], bump)] 
    pub swap_state: Account<'info, SwapState>,
}


