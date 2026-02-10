use anchor_client::solana_sdk::pubkey::Pubkey;
use anchor_client::Program;
use solana_sdk::account::Account;
use solana_sdk::instruction::Instruction;

use crate::pools::*;
use std::fmt::Debug;

use anchor_client::Cluster;

#[derive(Debug)]
pub struct PoolDir {
    pub tipe: PoolType,
    pub dir_path: String,
}

#[derive(Debug)]
pub enum PoolType {
    OrcaPoolType,
    RaydiumType,
    MeteoraType,
    PhoenixType,
    LifinityType,
    JupiterType,
}

pub fn pool_factory(tipe: &PoolType, json_str: &String) -> Box<dyn PoolOperations> {
    match tipe {
        PoolType::OrcaPoolType => {
            let pool: OrcaPool = serde_json::from_str(json_str).unwrap();
            Box::new(pool)
        }
        PoolType::RaydiumType => {
            let pool: RaydiumPool = serde_json::from_str(json_str).unwrap();
            Box::new(pool)
        }
        PoolType::SerumPoolType => {
            let pool: SerumPool = serde_json::from_str(json_str).unwrap();
            Box::new(pool)
        }
        PoolType::JupiterPoolType => {
            let pool: JupiterPool = serde_json::from_str(json_str).unwrap();
            Box::new(pool)
        }
    }
}

pub trait PoolOperations: Debug {
    fn get_name(&self) -> String;
    fn get_update_accounts(&self) -> Vec<Pubkey>;
    fn set_update_accounts(&mut self, accounts: Vec<Option<Account>>, cluster: Cluster);

    fn mint_2_addr(&self, mint: &Pubkey) -> Pubkey;
    fn get_mints(&self) -> Vec<Pubkey>;
    fn mint_2_scale(&self, mint: &Pubkey) -> u64;

    fn get_quote_with_amounts_scaled(
        &self,
        amount_in: u128,
        mint_in: &Pubkey,
        mint_out: &Pubkey,
    ) -> u128;
    fn swap_ix(
        &self,
        program: &Program,
        owner: &Pubkey,
        mint_in: &Pubkey,
        mint_out: &Pubkey,
    ) -> Vec<Instruction>;

    fn can_trade(&self, mint_in: &Pubkey, mint_out: &Pubkey) -> bool; // used for tests
}

// clone_trait_object!(PoolOperations);
