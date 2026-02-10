use anchor_lang::prelude::*;
use anchor_lang::solana_program::{instruction::Instruction, program::invoke};
use anchor_spl::token::{Token, TokenAccount};

// Jupiter program ID
pub const JUPITER_PROGRAM_ID: &str = "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB";

#[derive(Accounts)]
pub struct JupiterSwap<'info> {
    /// CHECK: Validated by Jupiter program
    #[account(address = JUPITER_PROGRAM_ID.parse::<Pubkey>().unwrap())]
    pub jupiter_program: AccountInfo<'info>,
    pub token_program: Program<'info, Token>,
    pub user_authority: Signer<'info>,
    #[account(mut)]
    pub user_source_token: Account<'info, TokenAccount>,
    #[account(mut)]
    pub user_destination_token: Account<'info, TokenAccount>,
    /// CHECK: Remaining accounts will be populated based on the route
    pub remaining_accounts: UncheckedAccount<'info>,
}

impl<'info> JupiterSwap<'info> {
    pub fn process_swap(
        &self,
        amount_in: u64,
        minimum_amount_out: u64,
        route_data: Vec<u8>,
    ) -> Result<()> {
        // Create route data with amount_in and minimum_amount_out
        let mut final_route_data = Vec::with_capacity(route_data.len() + 16);
        final_route_data.extend_from_slice(&amount_in.to_le_bytes());
        final_route_data.extend_from_slice(&minimum_amount_out.to_le_bytes());
        final_route_data.extend_from_slice(&route_data);

        // Get remaining accounts for the route
        let accounts: Vec<AccountMeta> = vec![
            AccountMeta::new_readonly(self.token_program.key(), false),
            AccountMeta::new_readonly(self.user_authority.key(), true),
            AccountMeta::new(self.user_source_token.key(), false),
            AccountMeta::new(self.user_destination_token.key(), false),
        ];

        // Create the instruction
        let ix = Instruction {
            program_id: self.jupiter_program.key(),
            accounts,
            data: final_route_data,
        };

        // Collect account infos
        let account_infos = vec![
            self.jupiter_program.to_account_info(),
            self.token_program.to_account_info(),
            self.user_authority.to_account_info(),
            self.user_source_token.to_account_info(),
            self.user_destination_token.to_account_info(),
            self.remaining_accounts.to_account_info(),
        ];

        // Invoke the Jupiter swap instruction
        invoke(
            &ix,
            &account_infos,
        )?;

        Ok(())
    }
}
