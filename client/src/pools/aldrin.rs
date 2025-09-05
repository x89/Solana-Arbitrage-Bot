use std::collections::HashMap;
use std::fmt::Debug;
use std::str::FromStr;

use serde;
use serde::{Deserialize, Serialize};

use anchor_client::solana_sdk::pubkey::Pubkey;
use anchor_client::Program;
use anchor_client::Cluster;

use solana_sdk::account::Account;
use solana_sdk::instruction::Instruction;

use tmp::accounts as tmp_accounts;
use tmp::instruction as tmp_ix;

use crate::serialize::token::{Token, WrappedPubkey, unpack_token_account};
use crate::serialize::pool::JSONFeeStructure; 
use crate::pool::PoolOperations;
use crate::pool_utils::base::CurveType;
use crate::utils::{str2pubkey, derive_token_address};
use crate::pool_utils::{
    orca::{get_pool_quote_with_amounts},
    fees::Fees,
};
use crate::constants::*;

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AldrinPool {
    pub lp_token_freeze_vault: WrappedPubkey,
    pub pool_mint: WrappedPubkey,
    pub pool_signer: WrappedPubkey,
    pub pool_signer_nonce: u64,
    pub authority: WrappedPubkey,
    pub initializer_account: WrappedPubkey,
    pub fee_base_account: WrappedPubkey,
    pub fee_quote_account: WrappedPubkey,
    pub fee_pool_token_account: WrappedPubkey,
    // !
    pub token_ids: Vec<String>,
    pub tokens: HashMap<String, Token>,
    pub fees: JSONFeeStructure,
    pub curve_type: u8,
    //
    pub curve: WrappedPubkey,
    pub pool_public_key: WrappedPubkey,
    pub pool_version: u8,
    // to set later 
    #[serde(skip)]
    pub pool_amounts: HashMap<String, u128>
}

impl PoolOperations for AldrinPool {
    fn swap_ix(&self, 
        program: &Program,
        owner: &Pubkey,
        _mint_in: &Pubkey, 
        mint_out: &Pubkey
    ) -> Vec<Instruction> {
        let (state_pda, _) = Pubkey::find_program_address(
            &[b"swap_state"], 
            &program.id()
        );

        let base_token_mint = &self.token_ids[0];
        let quote_token_mint = &self.token_ids[1];

        let base_token_vault = self.tokens
            .get(base_token_mint)
            .unwrap()
            .addr.0;
        let quote_token_vault = self.tokens
            .get(quote_token_mint)
            .unwrap()
            .addr.0;

        let is_inverted = &mint_out.to_string() == quote_token_mint;
        let user_base_ata = derive_token_address(
            owner, 
            &Pubkey::from_str(base_token_mint).unwrap()
        );
        let user_quote_ata = derive_token_address(
            owner, 
            &Pubkey::from_str(quote_token_mint).unwrap()
        );

        let swap_ix;
        if self.pool_version == 1 {
            swap_ix = program
                .request()
                .accounts(tmp_accounts::AldrinSwapV1 {
                    pool_public_key: self.pool_public_key.0,
                    pool_signer: self.pool_signer.0,
                    pool_mint: self.pool_mint.0,
                    base_token_vault, 
                    quote_token_vault, 
                    fee_pool_token_account: self.fee_pool_token_account.0,
                    user_transfer_authority: *owner,
                    user_base_ata, 
                    user_quote_ata,
                    // ...
                    aldrin_v1_program: *ALDRIN_V1_PROGRAM_ID,
                    token_program: *TOKEN_PROGRAM_ID,
                    swap_state: state_pda, 
                })
                .args(tmp_ix::AldrinSwapV1 { is_inverted })
                .instructions()
                .unwrap();
        } else { 
            swap_ix = program
                .request()
                .accounts(tmp_accounts::AldrinSwapV2 {
                    pool_public_key: self.pool_public_key.0,
                    pool_signer: self.pool_signer.0,
                    pool_mint: self.pool_mint.0,
                    base_token_vault, 
                    quote_token_vault, 
                    fee_pool_token_account: self.fee_pool_token_account.0,
                    user_transfer_authority: *owner,
                    user_base_ata, 
                    user_quote_ata,
                    // ...
                    aldrin_v2_program: *ALDRIN_V2_PROGRAM_ID,
                    curve: self.curve.0,
                    token_program: *TOKEN_PROGRAM_ID,
                    swap_state: state_pda, 
                })
                .args(tmp_ix::AldrinSwapV2 { is_inverted })
                .instructions()
                .unwrap();
        }
        swap_ix
    }

    fn get_quote_with_amounts_scaled(
        &self, 
        scaled_amount_in: u128, 
        mint_in: &Pubkey,
        mint_out: &Pubkey,
    ) -> u128 {
        
        let pool_src_amount = *self.pool_amounts.get(&mint_in.to_string()).unwrap();
        let pool_dst_amount = *self.pool_amounts.get(&mint_out.to_string()).unwrap();

        // compute fees 
        let trader_fee = &self.fees.trader_fee;
        let owner_fee = &self.fees.owner_fee;
        let fees = Fees {
            trade_fee_numerator: trader_fee.numerator,
            trade_fee_denominator: trader_fee.denominator,
            owner_trade_fee_numerator: owner_fee.numerator,
            owner_trade_fee_denominator: owner_fee.denominator,
            owner_withdraw_fee_numerator: 0,
            owner_withdraw_fee_denominator: 0,
            host_fee_numerator: 0,
            host_fee_denominator: 0,
        };

        let ctype = if self.curve_type == 1 { 
            CurveType::Stable
        } else {
            CurveType::ConstantProduct 
        };

        // get quote -- works for either constant product or stable swap 
        

        get_pool_quote_with_amounts(
            scaled_amount_in,
            ctype,
            170, // from sdk 
            &fees, 
            pool_src_amount, 
            pool_dst_amount, 
            None,
        ).unwrap()
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

    fn get_name(&self) -> String {
        
        if self.pool_version == 1 { 
            "AldrinV1".to_string()
        } else { 
            "AldrinV2".to_string()
        }
    }

    fn get_update_accounts(&self) -> Vec<Pubkey> {
        // pool vault amount 
        // TODO: replace with token_ids + ['addr'] key
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