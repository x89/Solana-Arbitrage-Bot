use anchor_lang::prelude::*;
use anchor_lang::solana_program::instruction::{AccountMeta, Instruction};
use anchor_spl::token::{Token, TokenAccount};

// Whirlpool program ID
pub const WHIRLPOOL_PROGRAM_ID: &str = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc";

#[derive(Accounts)]
pub struct OrcaSwap<'info> {
    pub token_program: Program<'info, Token>,
    pub token_authority: Signer<'info>,
    /// CHECK: This is the Orca Whirlpool state account that's validated by the Whirlpool program
    #[account(mut)]
    pub whirlpool: AccountInfo<'info>,
    #[account(mut)]
    pub token_owner_account_a: Account<'info, TokenAccount>,
    #[account(mut)]
    pub token_vault_a: Account<'info, TokenAccount>,
    #[account(mut)]
    pub token_owner_account_b: Account<'info, TokenAccount>,
    #[account(mut)]
    pub token_vault_b: Account<'info, TokenAccount>,
    /// CHECK: This is the tick array account that's validated by the Whirlpool program
    #[account(mut)]
    pub tick_array_0: AccountInfo<'info>,
    /// CHECK: This is the tick array account that's validated by the Whirlpool program
    #[account(mut)]
    pub tick_array_1: AccountInfo<'info>,
    /// CHECK: This is the tick array account that's validated by the Whirlpool program
    #[account(mut)]
    pub tick_array_2: AccountInfo<'info>,
    /// CHECK: This is the oracle account that's validated by the Whirlpool program
    #[account(mut)]
    pub oracle: AccountInfo<'info>,
    /// CHECK: This is the Whirlpool program ID
    #[account(address = WHIRLPOOL_PROGRAM_ID.parse::<Pubkey>().unwrap())]
    pub whirlpool_program: AccountInfo<'info>,
}

impl<'info> OrcaSwap<'info> {
    pub fn process_swap(
        &self,
        amount_in: u64,
        minimum_amount_out: u64,
    ) -> Result<()> {
        let ix_data = {
            let mut data = Vec::with_capacity(32);
            data.extend_from_slice(&[2]); // Swap instruction discriminator
            data.extend_from_slice(&amount_in.to_le_bytes());
            data.extend_from_slice(&minimum_amount_out.to_le_bytes());
            data.extend_from_slice(&[0]); // sqrt_price_limit (0 means no limit)
            data.extend_from_slice(&[1]); // amount_specified_is_input
            data.extend_from_slice(&[1]); // a_to_b
            data
        };

        let accounts = vec![
            AccountMeta::new(self.whirlpool.key(), false),
            AccountMeta::new_readonly(self.token_program.key(), false),
            AccountMeta::new_readonly(self.token_authority.key(), true),
            AccountMeta::new(self.token_owner_account_a.key(), false),
            AccountMeta::new(self.token_vault_a.key(), false),
            AccountMeta::new(self.token_owner_account_b.key(), false),
            AccountMeta::new(self.token_vault_b.key(), false),
            AccountMeta::new(self.tick_array_0.key(), false),
            AccountMeta::new(self.tick_array_1.key(), false),
            AccountMeta::new(self.tick_array_2.key(), false),
            AccountMeta::new(self.oracle.key(), false),
        ];

        let instruction = Instruction {
            program_id: self.whirlpool_program.key(),
            accounts,
            data: ix_data,
        };

        anchor_lang::solana_program::program::invoke(
            &instruction,
            &[
                self.whirlpool.to_account_info(),
                self.token_program.to_account_info(),
                self.token_authority.to_account_info(),
                self.token_owner_account_a.to_account_info(),
                self.token_vault_a.to_account_info(),
                self.token_owner_account_b.to_account_info(),
                self.token_vault_b.to_account_info(),
                self.tick_array_0.to_account_info(),
                self.tick_array_1.to_account_info(),
                self.tick_array_2.to_account_info(),
                self.oracle.to_account_info(),
            ],
        )?;

        Ok(())
    }
}
