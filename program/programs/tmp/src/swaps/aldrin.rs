use anchor_lang::prelude::*;
use solana_program::instruction::{AccountMeta, Instruction};
use solana_program;
use anchor_spl::{
    token::{TokenAccount}
};
use anchor_lang::{Accounts};
use sha2::{Digest, Sha256};

use crate::state::SwapState;

pub fn _aldrin_swap_v1<'info>(
    ctx: &Context<'_, '_, '_, 'info, AldrinSwapV1<'info>>, 
    amount_in: u64,
    is_inverted: bool,
) -> Result<()> {

    // anchor method discriminator 
    let key = "global:swap".to_string();
    let mut hasher = Sha256::new(); 
    hasher.update(key);
    let result = hasher.finalize();
    let fcn_name = &result.as_slice()[..8];

    let amount_in_bytes = &amount_in.try_to_vec().unwrap()[..];
    let amount_out_bytes = &(0 as u64).try_to_vec().unwrap()[..];
    let bid_ask_flag = if is_inverted { 1 } else { 0 }; // 0 = bid, 1 = ask 
    let bid_ask = &[bid_ask_flag];
    let data = [
        fcn_name, 
        amount_in_bytes, 
        amount_out_bytes, 
        bid_ask, 
    ].concat();

    let ix_accounts = vec![
        AccountMeta::new_readonly(*ctx.accounts.pool_public_key.key, false),
        AccountMeta::new_readonly(*ctx.accounts.pool_signer.key, false),
        
        AccountMeta::new(*ctx.accounts.pool_mint.key, false),
        AccountMeta::new(*ctx.accounts.base_token_vault.key, false), 
        AccountMeta::new(*ctx.accounts.quote_token_vault.key, false), 
        AccountMeta::new(*ctx.accounts.fee_pool_token_account.key, false),

        AccountMeta::new_readonly(*ctx.accounts.user_transfer_authority.key, true),
        AccountMeta::new(ctx.accounts.user_base_ata.key(), false), 
        AccountMeta::new(ctx.accounts.user_quote_ata.key(), false), 
        AccountMeta::new_readonly(*ctx.accounts.token_program.key, false),
    ];

    // gotta work with the bytes to get this right 
    let instruction = Instruction {
        program_id: *ctx.accounts.aldrin_v1_program.key,
        accounts: ix_accounts,
        data: data,
    };

    let accounts = vec![
        ctx.accounts.pool_public_key.to_account_info(),
        ctx.accounts.pool_signer.to_account_info(),
        ctx.accounts.pool_mint.to_account_info(),
        ctx.accounts.base_token_vault.to_account_info(),
        ctx.accounts.quote_token_vault.to_account_info(),
        ctx.accounts.fee_pool_token_account.to_account_info(),
        ctx.accounts.user_transfer_authority.to_account_info(),
        ctx.accounts.user_base_ata.to_account_info(),
        ctx.accounts.user_quote_ata.to_account_info(),
        ctx.accounts.token_program.to_account_info(),
        ctx.accounts.aldrin_v1_program.to_account_info()
    ];

    solana_program::program::invoke(
        &instruction, 
        &accounts, 
    )?;

    Ok(())
}

pub fn _aldrin_swap_v2<'info>(
    ctx: &Context<'_, '_, '_, 'info, AldrinSwapV2<'info>>, 
    amount_in: u64,
    is_inverted: bool,
) -> Result<()> {

    // anchor method discriminator 
    let key = "global:swap".to_string();
    let mut hasher = Sha256::new(); 
    hasher.update(key);
    let result = hasher.finalize();
    let fcn_name = &result.as_slice()[..8];

    let amount_in_bytes = &amount_in.try_to_vec().unwrap()[..];
    let amount_out_bytes = &(0 as u64).try_to_vec().unwrap()[..];
    let bid_ask_flag = if is_inverted { 1 } else { 0 }; // 0 = bid, 1 = ask 
    let bid_ask = &[bid_ask_flag];
    let data = [
        fcn_name, 
        amount_in_bytes, 
        amount_out_bytes, 
        bid_ask, 
    ].concat();

    let ix_accounts = vec![
        AccountMeta::new_readonly(*ctx.accounts.pool_public_key.key, false),
        AccountMeta::new_readonly(*ctx.accounts.pool_signer.key, false),
        
        AccountMeta::new(*ctx.accounts.pool_mint.key, false),
        AccountMeta::new(*ctx.accounts.base_token_vault.key, false), 
        AccountMeta::new(*ctx.accounts.quote_token_vault.key, false), 
        AccountMeta::new(*ctx.accounts.fee_pool_token_account.key, false),

        AccountMeta::new_readonly(*ctx.accounts.user_transfer_authority.key, true),
        AccountMeta::new(ctx.accounts.user_base_ata.key(), false), 
        AccountMeta::new(ctx.accounts.user_quote_ata.key(), false), 
        AccountMeta::new_readonly(*ctx.accounts.curve.key, false),
        AccountMeta::new_readonly(*ctx.accounts.token_program.key, false),
    ];

    // gotta work with the bytes to get this right 
    let instruction = Instruction {
        program_id: *ctx.accounts.aldrin_v2_program.key,
        accounts: ix_accounts,
        data: data,
    };

    let accounts = vec![
        ctx.accounts.pool_public_key.to_account_info(),
        ctx.accounts.pool_signer.to_account_info(),
        ctx.accounts.pool_mint.to_account_info(),
        ctx.accounts.base_token_vault.to_account_info(),
        ctx.accounts.quote_token_vault.to_account_info(),
        ctx.accounts.fee_pool_token_account.to_account_info(),
        ctx.accounts.user_transfer_authority.to_account_info(),
        ctx.accounts.user_base_ata.to_account_info(),
        ctx.accounts.user_quote_ata.to_account_info(),
        ctx.accounts.curve.to_account_info(),
        ctx.accounts.token_program.to_account_info(),
        ctx.accounts.aldrin_v2_program.to_account_info()
    ];

    solana_program::program::invoke(
        &instruction, 
        &accounts, 
    )?;

    Ok(())
}

#[derive(Accounts)]
pub struct AldrinSwapV1<'info> {
    pub pool_public_key: AccountInfo<'info>,
    pub pool_signer: AccountInfo<'info>,
    #[account(mut)]
    pub pool_mint: AccountInfo<'info>,
    #[account(mut)]
    pub base_token_vault: AccountInfo<'info>,
    #[account(mut)]
    pub quote_token_vault: AccountInfo<'info>,
    #[account(mut)]
    pub fee_pool_token_account: AccountInfo<'info>,
    pub user_transfer_authority : Signer<'info>,
    #[account(mut)]
    pub user_base_ata: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_quote_ata: Account<'info, TokenAccount>,
    pub aldrin_v1_program: AccountInfo<'info>,
    pub token_program: AccountInfo<'info>,
    #[account(mut, seeds=[b"swap_state"], bump)] 
    pub swap_state: Account<'info, SwapState>,
}

#[derive(Accounts)]
pub struct AldrinSwapV2<'info> {
    pub pool_public_key: AccountInfo<'info>,
    pub pool_signer: AccountInfo<'info>,
    #[account(mut)]
    pub pool_mint: AccountInfo<'info>,
    #[account(mut)]
    pub base_token_vault: AccountInfo<'info>,
    #[account(mut)]
    pub quote_token_vault: AccountInfo<'info>,
    #[account(mut)]
    pub fee_pool_token_account: AccountInfo<'info>,
    pub user_transfer_authority : Signer<'info>,
    #[account(mut)]
    pub user_base_ata: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_quote_ata: Account<'info, TokenAccount>,
    pub aldrin_v2_program: AccountInfo<'info>,
    pub curve: AccountInfo<'info>, // v2 difference! 
    pub token_program: AccountInfo<'info>,
    #[account(mut, seeds=[b"swap_state"], bump)] 
    pub swap_state: Account<'info, SwapState>,
}
