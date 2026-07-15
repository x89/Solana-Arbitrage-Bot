use std::collections::HashMap;
use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use solana_sdk::account::Account;
use crate::serialize::token::{Token, WrappedPubkey, unpack_token_account};
use crate::serialize::pool::JSONFeeStructure;
use crate::pool::PoolOperations;

use anchor_client::solana_sdk::pubkey::Pubkey;
use anchor_client::Cluster;
use anchor_client::Program;

use solana_sdk::instruction::Instruction;

use tmp::accounts as tmp_accounts;
use tmp::instruction as tmp_ix;

use crate::pool_utils::base::CurveType;
use crate::utils::{str2pubkey, derive_token_address};
use crate::pool_utils::{
    raydium::{get_pool_quote_with_amounts},
    fees::Fees,
};
use crate::constants::*;

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RaydiumPool {
    pub address: WrappedPubkey,
    pub nonce: u64,
    pub authority: WrappedPubkey,
    pub lp_token_mint: WrappedPubkey,
    pub fee_account: WrappedPubkey,
    pub token_ids: Vec<String>,
    pub tokens: HashMap<String, Token>,
    pub fee_structure: JSONFeeStructure,
    #[serde(default)]
    pub pool_amounts: HashMap<String, u128>,
}

impl PoolOperations for RaydiumPool {
    fn swap_ix(&self, 
        program: &Program,
        owner: &Pubkey,
        mint_in: &Pubkey, 
        mint_out: &Pubkey
    ) -> Vec<Instruction> {
        let (swap_state, _) = Pubkey::find_program_address(
            &[b"swap_state"], 
            &program.id()
        );
        let user_src = derive_token_address(owner, mint_in);
        let user_dst = derive_token_address(owner, mint_out); 

        let (authority_pda, _) = Pubkey::find_program_address(
            &[&self.address.to_bytes()],
            &RAYDIUM_PROGRAM_ID 
        );

        let pool_src = self.mint_2_addr(mint_in);
        let pool_dst = self.mint_2_addr(mint_out);

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

        swap_ix
    }

    fn get_quote_with_amounts_scaled(
        &self, 
        scaled_amount_in: u128, 
        mint_in: &Pubkey,
        mint_out: &Pubkey,
    ) -> u128 {
        let pool_src_amount = self.pool_amounts.get(&mint_in.to_string()).unwrap();
        let pool_dst_amount = self.pool_amounts.get(&mint_out.to_string()).unwrap();

        // Compute fees
        let trader_fee = &self.fee_structure.trader_fee;
        let fees = Fees {
            trade_fee_numerator: trader_fee.numerator,
            trade_fee_denominator: trader_fee.denominator,
            owner_trade_fee_numerator: 0,
            owner_trade_fee_denominator: 0,
            owner_withdraw_fee_numerator: 0,
            owner_withdraw_fee_denominator: 0,
            host_fee_numerator: 0,
            host_fee_denominator: 0,
        };

        // Get quote using Raydium's logic
        get_pool_quote_with_amounts(
            scaled_amount_in,
            ctype,
            self.amp, 
            &fees, 
            *pool_src_amount, 
            *pool_dst_amount, 
            None,
        ).unwrap()
    }

    fn get_update_accounts(&self) -> Vec<Pubkey> {
        self.get_mints()
            .iter()
            .map(|mint| self.mint_2_addr(mint))
            .collect()        
    }

    fn can_trade(&self, 
        _mint_in: &Pubkey,
        _mint_out: &Pubkey
    ) -> bool {
        for amount in self.pool_amounts.values() {
            if *amount == 0 { return false; }
        }
        true
    }

    fn set_update_accounts(&mut self, accounts: Vec<Option<Account>>, _cluster: Cluster) { 
        let ids: Vec<String> = self
            .get_mints()
            .iter()
            .map(|mint| mint.to_string())
            .collect();
        
        let acc_data0 = &accounts[0].as_ref().unwrap().data;
        let acc_data1 = &accounts[1].as_ref().unwrap().data;

        let amount0 = unpack_token_account(acc_data0).amount as u128;
        let amount1 = unpack_token_account(acc_data1).amount as u128;

        self.pool_amounts.insert(ids[0].clone(), amount0);
        self.pool_amounts.insert(ids[1].clone(), amount1);
    }

    fn get_name(&self) -> String {
        "Raydium".to_string()
    }

    fn mint_2_addr(&self, mint: &Pubkey) -> Pubkey {
        let token = self.tokens.get(&mint.to_string()).unwrap();
        token.addr.0
    }

    fn mint_2_scale(&self, mint: &Pubkey) -> u64 {
        let token = self.tokens.get(&mint.to_string()).unwrap();
        token.scale
    }

    fn get_mints(&self) -> Vec<Pubkey> {
        let mut mints: Vec<Pubkey> = self.token_ids
            .iter()
            .map(|k| str2pubkey(k))
            .collect();
        // Sort to ensure consistency across different pools 
        mints.sort();
        mints
    }
}
