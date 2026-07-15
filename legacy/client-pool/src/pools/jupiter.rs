use core::panic;

use std::collections::HashMap;
use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use crate::pool::PoolOperations;
use crate::serialize::token::{WrappedPubkey};

use crate::utils::{derive_token_address};

use solana_sdk::pubkey::Pubkey;

use anchor_client::{Program, Cluster};
use solana_sdk::instruction::Instruction;

use solana_sdk::clock::Epoch;
use solana_sdk::account::Account;
use solana_sdk::account_info::AccountInfo;
use crate::constants::*;

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct JupiterPool {
    pub own_address: WrappedPubkey,
    pub base_mint: WrappedPubkey,
    pub quote_mint: WrappedPubkey,
    pub base_vault: WrappedPubkey,
    pub quote_vault: WrappedPubkey,
    pub swap_program_id: WrappedPubkey,
    pub taker_fee_pct: f64,
    #[serde(skip)]
    pub accounts: Option<Vec<Option<Account>>>,
    #[serde(skip)]
    pub open_orders: Option<HashMap<String, String>>,
}

fn account_info<'a>(pk: &'a Pubkey, account: &'a mut Account) -> AccountInfo<'a> {
    AccountInfo::new(
        pk,
        false,
        true,
        &mut account.lamports,
        &mut account.data,
        &account.owner,
        false,
        Epoch::default(),
    )
}

impl PoolOperations for JupiterPool {
    fn get_name(&self) -> String {
        "Jupiter".to_string()
    }

    fn get_update_accounts(&self) -> Vec<Pubkey> {
        vec![
            self.own_address.0,
            self.base_vault.0,
            self.quote_vault.0,
        ]
    }

    fn set_update_accounts(
        &mut self,
        accounts: Vec<Option<Account>>,
        cluster: Cluster,
    ) {
        self.accounts = Some(accounts);
        // Load open orders from a file or other source as needed
        self.open_orders = Some(HashMap::new()); // Placeholder
    }

    fn mint_2_addr(&self, _mint: &Pubkey) -> Pubkey {
        panic!("Function not implemented");
    }

    fn get_mints(&self) -> Vec<Pubkey> {
        let mut mints = vec![
            self.base_mint.0,
            self.quote_mint.0,
        ];
        mints.sort();
        mints
    }

    fn mint_2_scale(&self, mint: &Pubkey) -> u64 {
        // Implement logic to return the scale based on the mint
        panic!("Invalid mint provided");
    }

    fn get_quote_with_amounts_scaled(
        &self,
        amount_in: u128,
        mint_in: &Pubkey,
        _mint_out: &Pubkey,
    ) -> u128 {
        // Logic to calculate the quote based on the amount in
        // Placeholder logic
        amount_in / 2 // Replace with actual calculation
    }

    fn swap_ix(
        &self,
        program: &Program,
        owner: &Pubkey,
        mint_in: &Pubkey,
        _mint_out: &Pubkey,
    ) -> Vec<Instruction> {
        let base_ata = derive_token_address(owner, &self.base_mint);
        let quote_ata = derive_token_address(owner, &self.quote_mint);

        // Construct the swap instruction
        let swap_ix = program
            .request()
            .accounts(tmp_accounts::RaydiumSwap {
                token_swap: self.address.0, 
                authority: authority_pda,
                user_transfer_authority: *owner,
                user_src,
                pool_src,
                user_dst,
                pool_dst,
                lp_mint: self.lp_token_mint.0,
                fee_account: self.fee_account.0,
                token_program: *TOKEN_PROGRAM_ID,
                swap_program: *RAYDIUM_PROGRAM_ID,
                swap_state,
            })
            .args(tmp_ix::RaydiumSwap { })
            .instructions()
            .unwrap();
    
    }

    fn can_trade(
        &self,
        mint_in: &Pubkey,
        _mint_out: &Pubkey,
    ) -> bool {
        // Check if trading is possible based on the current state
        true // Placeholder
    }
}
