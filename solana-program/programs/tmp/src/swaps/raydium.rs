use anchor_lang::prelude::*;
use anchor_lang::solana_program::{instruction::Instruction, program::invoke};
use anchor_spl::token::{Token, TokenAccount};

use crate::error::ErrorCode;
use crate::state::RaydiumSwapState;

// Raydium AMM program ID
pub const RAYDIUM_AMM_V4_PROGRAM_ID: &str = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8";

#[derive(Accounts)]
pub struct RaydiumSwap<'info> {
    /// CHECK: Validated by Raydium program
    pub amm_id: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    #[account(mut)]
    pub amm_authority: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    #[account(mut)]
    pub amm_open_orders: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    #[account(mut)]
    pub pool_coin_token_account: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    #[account(mut)]
    pub pool_pc_token_account: AccountInfo<'info>,
    /// CHECK: Serum program
    pub serum_program_id: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    #[account(mut)]
    pub serum_market: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    #[account(mut)]
    pub serum_bids: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    #[account(mut)]
    pub serum_asks: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    #[account(mut)]
    pub serum_event_queue: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    #[account(mut)]
    pub serum_coin_vault_account: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    #[account(mut)]
    pub serum_pc_vault_account: AccountInfo<'info>,
    /// CHECK: Validated by Raydium program
    pub serum_vault_signer: AccountInfo<'info>,
    #[account(mut)]
    pub user_source_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_destination_token: Account<'info, TokenAccount>,
    pub user_authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
    #[account(mut)]
    pub swap_state: Account<'info, RaydiumSwapState>,
}

impl<'info> RaydiumSwap<'info> {
    pub fn process_swap(
        &self,
        amount_in: u64,
        minimum_amount_out: u64,
    ) -> Result<()> {
        let ix = Instruction {
            program_id: self.amm_id.key(),
            accounts: vec![
                AccountMeta::new(self.amm_id.key(), false),
                AccountMeta::new(self.amm_authority.key(), false),
                AccountMeta::new(self.amm_open_orders.key(), false),
                AccountMeta::new(self.pool_coin_token_account.key(), false),
                AccountMeta::new(self.pool_pc_token_account.key(), false),
                AccountMeta::new_readonly(self.serum_program_id.key(), false),
                AccountMeta::new(self.serum_market.key(), false),
                AccountMeta::new(self.serum_bids.key(), false),
                AccountMeta::new(self.serum_asks.key(), false),
                AccountMeta::new(self.serum_event_queue.key(), false),
                AccountMeta::new(self.serum_coin_vault_account.key(), false),
                AccountMeta::new(self.serum_pc_vault_account.key(), false),
                AccountMeta::new_readonly(self.serum_vault_signer.key(), false),
                AccountMeta::new(self.user_source_token.key(), false),
                AccountMeta::new(self.user_destination_token.key(), false),
                AccountMeta::new_readonly(self.user_authority.key(), true),
                AccountMeta::new_readonly(self.token_program.key(), false),
            ],
            data: self.build_swap_instruction_data(amount_in, minimum_amount_out),
        };

        invoke(
            &ix,
            &[
                self.amm_id.to_account_info(),
                self.amm_authority.to_account_info(),
                self.amm_open_orders.to_account_info(),
                self.pool_coin_token_account.to_account_info(),
                self.pool_pc_token_account.to_account_info(),
                self.serum_program_id.to_account_info(),
                self.serum_market.to_account_info(),
                self.serum_bids.to_account_info(),
                self.serum_asks.to_account_info(),
                self.serum_event_queue.to_account_info(),
                self.serum_coin_vault_account.to_account_info(),
                self.serum_pc_vault_account.to_account_info(),
                self.serum_vault_signer.to_account_info(),
                self.user_source_token.to_account_info(),
                self.user_destination_token.to_account_info(),
                self.user_authority.to_account_info(),
                self.token_program.to_account_info(),
            ],
        ).map_err(|_| ErrorCode::RaydiumSwapFailed)?;

        Ok(())
    }

    fn build_swap_instruction_data(&self, amount_in: u64, minimum_amount_out: u64) -> Vec<u8> {
        let mut data = Vec::with_capacity(9);
        data.push(9u8); // Swap instruction discriminator
        data.extend_from_slice(&amount_in.to_le_bytes());
        data.extend_from_slice(&minimum_amount_out.to_le_bytes());
        data
    }
}
