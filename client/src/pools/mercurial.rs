use std::collections::HashMap;
use std::fmt::Debug;
use serde;
use serde::{Deserialize, Serialize};
use crate::serialize::token::{Token, WrappedPubkey, unpack_token_account};
use crate::pool::PoolOperations;

use anchor_client::solana_sdk::pubkey::Pubkey;
use anchor_client::Cluster;
use anchor_client::Program;

use solana_sdk::account::Account;
use solana_sdk::instruction::Instruction;

use tmp::accounts as tmp_accounts;
use tmp::instruction as tmp_ix;

use crate::utils::{str2pubkey, derive_token_address};
use crate::constants::*;
use crate::pool_utils::stable::Stable;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MercurialPool {
    pub pool_account: WrappedPubkey,
    pub pool_token_mint: WrappedPubkey,
    pub authority: WrappedPubkey,
    pub token_ids: Vec<String>,
    pub tokens: HashMap<String, Token>,
    pub amp: u64,
    // unique
    pub fee_numerator: u64,
    pub admin_numerator: u64,
    pub precision_factor: u64,
    pub precision_multiplier: Vec<u64>,
    // to set later 
    #[serde(skip)]
    pub pool_amounts: HashMap<String, u128>
}

impl PoolOperations for MercurialPool {
    fn swap_ix(&self, 
        program: &Program,
        owner: &Pubkey,
        mint_in: &Pubkey, 
        mint_out: &Pubkey
    ) -> Vec<Instruction> {
        let (swap_state_pda, _) = Pubkey::find_program_address(
            &[b"swap_state"], 
            &program.id()
        );
        let user_src = derive_token_address(owner, mint_in);
        let user_dst = derive_token_address(owner, mint_out); 

        let pool0 = &self.tokens[&self.token_ids[0]].addr;
        let pool1 = &self.tokens[&self.token_ids[1]].addr;

        let swap_ix = program
            .request()
            .accounts(tmp_accounts::MercurialSwap {
                pool_account: self.pool_account.0,
                authority: self.authority.0,
                user_transfer_authority: *owner,
                user_src,
                user_dst, 
                pool_src: pool0.0, // src/dst order doesnt matter ??
                pool_dst: pool1.0, // src/dst order doesnt matter ??
                token_program: *TOKEN_PROGRAM_ID,
                mercurial_swap_program: *MERCURIAL_PROGRAM_ID,
                swap_state: swap_state_pda,
            })
            .args(tmp_ix::MercurialSwap { })
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
        let fee_denom = 10_u128.pow(10); 

        let calculator = Stable {
            amp: self.amp, 
            fee_numerator: self.fee_numerator as u128, 
            fee_denominator: fee_denom,
        };

        // only stable swap pools here 
        let pool_src_amount = self.pool_amounts.get(&mint_in.to_string()).unwrap();
        let pool_dst_amount = self.pool_amounts.get(&mint_out.to_string()).unwrap();
        let pool_amounts = [*pool_src_amount, *pool_dst_amount];

        let input_idx = self.token_ids
            .iter()
            .position(|m| *m == mint_in.to_string())
            .unwrap();
        let output_idx = (input_idx + 1) % 2; 

        let percision_multipliers = [
            self.precision_multiplier[input_idx], 
            self.precision_multiplier[output_idx]
        ];

        

        calculator.get_quote(
            pool_amounts,    
            percision_multipliers, 
            scaled_amount_in 
        )
    }

    fn get_name(&self) -> String {
         
        "Mercurial".to_string()
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

    fn get_update_accounts(&self) -> Vec<Pubkey> {
        // pool vault amount 
        let accounts = self
            .get_mints()
            .iter()
            .map(|mint| self.mint_2_addr(mint))
            .collect();        
        accounts 
    }

    fn set_update_accounts(&mut self, accounts: Vec<Option<Account>>, _cluster: Cluster) { 
        let ids: Vec<String> = self
            .get_mints()
            .iter()
            .map(|mint| mint.to_string())
            .collect();
        let id0 = &ids[0];
        let id1 = &ids[1];
        
        let acc_data0 = &accounts[0].as_ref().unwrap().data;
        let acc_data1 = &accounts[1].as_ref().unwrap().data;

        let amount0 = unpack_token_account(acc_data0).amount as u128;
        let amount1 = unpack_token_account(acc_data1).amount as u128;

        self.pool_amounts.insert(id0.clone(), amount0);
        self.pool_amounts.insert(id1.clone(), amount1);
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
        // sort so that its consistent across different pools 
        mints.sort();
        mints
    }
}