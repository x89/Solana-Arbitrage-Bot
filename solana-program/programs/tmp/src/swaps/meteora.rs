use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount};
use dynamic_amm::{self, cpi::accounts::Swap as MeteoraSwapAccounts, program::DynamicAmm};

// Meteora program ID
pub const METEORA_PROGRAM_ID: &str = "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo";

#[derive(Accounts)]
pub struct MeteoraSwap<'info> {
    /// CHECK: Validated by Meteora program
    #[account(mut)]
    pub pool: AccountInfo<'info>,
    
    /// CHECK: Validated by Meteora program
    #[account(mut)]
    pub a_vault: AccountInfo<'info>,
    /// CHECK: Validated by Meteora program
    #[account(mut)]
    pub b_vault: AccountInfo<'info>,
    
    #[account(mut)]
    pub a_token_vault: Account<'info, TokenAccount>,
    #[account(mut)]
    pub b_token_vault: Account<'info, TokenAccount>,
    
    /// CHECK: Validated by Meteora program
    #[account(mut)]
    pub a_vault_lp_mint: AccountInfo<'info>,
    /// CHECK: Validated by Meteora program
    #[account(mut)]
    pub b_vault_lp_mint: AccountInfo<'info>,
    
    #[account(mut)]
    pub a_vault_lp: Account<'info, TokenAccount>,
    #[account(mut)]
    pub b_vault_lp: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub admin_token_fee: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub user_source_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_destination_token: Account<'info, TokenAccount>,
    
    pub user: Signer<'info>,
    pub token_program: Program<'info, Token>,
    /// CHECK: Validated by Meteora program
    pub vault_program: AccountInfo<'info>,
    
    pub meteora_program: Program<'info, DynamicAmm>,
}

impl<'info> MeteoraSwap<'info> {
    pub fn process_swap(
        &self,
        amount_in: u64,
        minimum_amount_out: u64,
    ) -> Result<()> {
        let cpi_accounts = MeteoraSwapAccounts {
            pool: self.pool.to_account_info(),
            a_vault: self.a_vault.to_account_info(),
            b_vault: self.b_vault.to_account_info(),
            a_token_vault: self.a_token_vault.to_account_info(),
            b_token_vault: self.b_token_vault.to_account_info(),
            a_vault_lp_mint: self.a_vault_lp_mint.to_account_info(),
            b_vault_lp_mint: self.b_vault_lp_mint.to_account_info(),
            a_vault_lp: self.a_vault_lp.to_account_info(),
            b_vault_lp: self.b_vault_lp.to_account_info(),
            admin_token_fee: self.admin_token_fee.to_account_info(),
            user_source_token: self.user_source_token.to_account_info(),
            user_destination_token: self.user_destination_token.to_account_info(),
            user: self.user.to_account_info(),
            token_program: self.token_program.to_account_info(),
            vault_program: self.vault_program.to_account_info(),
        };

        let cpi_ctx = CpiContext::new(
            self.meteora_program.to_account_info(),
            cpi_accounts,
        );

        dynamic_amm::cpi::swap(
            cpi_ctx,
            amount_in,
            minimum_amount_out,
        )?;

        Ok(())
    }
}