use super::DexProtocol;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use solana_client::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use std::sync::Arc;
use std::str::FromStr;
use raydium_amm::state::{Loadable, AmmInfo};
use spl_token_2022::amount_to_ui_amount;
use common::common_utils;
use solana_client::rpc_filter::{Memcmp, RpcFilterType};
use std::time::Duration;

const AMM_PROGRAM: &str = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8";

#[derive(Clone)]
pub struct RaydiumDex {
    program_id: Pubkey,
}

impl RaydiumDex {
    pub fn new(program_id: &str) -> Result<Self> {
        Ok(Self {
            program_id: Pubkey::from_str(program_id)?,
        })
    }

    async fn get_pool_price(
        rpc_client: Arc<RpcClient>,
        pool_id: Option<&str>,
        mint: Option<&str>,
    ) -> Result<(f64, f64, f64)> {
        let (amm_pool_id, pool_state) = Self::get_pool_state(rpc_client.clone(), pool_id, mint).await?;
        
        let load_pubkeys = vec![pool_state.pc_vault, pool_state.coin_vault];
        let rsps = common::rpc::get_multiple_accounts(&rpc_client, &load_pubkeys)?;
        
        // Add proper error handling for vault accounts
        let amm_pc_vault_account = rsps[0].clone()
            .ok_or_else(|| anyhow!("Failed to fetch PC vault account"))?;
        let amm_coin_vault_account = rsps[1].clone()
            .ok_or_else(|| anyhow!("Failed to fetch coin vault account"))?;
        
        let amm_pc_vault = common_utils::unpack_token(&amm_pc_vault_account.data)
            .map_err(|e| anyhow!("Failed to unpack PC vault token: {}", e))?;
        let amm_coin_vault = common_utils::unpack_token(&amm_coin_vault_account.data)
            .map_err(|e| anyhow!("Failed to unpack coin vault token: {}", e))?;
        
        let (base_account, quote_account) = if amm_coin_vault.base.is_native() {
            (
                (
                    pool_state.pc_vault_mint,
                    amount_to_ui_amount(amm_pc_vault.base.amount, pool_state.pc_decimals as u8),
                ),
                (
                    pool_state.coin_vault_mint,
                    amount_to_ui_amount(amm_coin_vault.base.amount, pool_state.coin_decimals as u8),
                ),
            )
        } else {
            (
                (
                    pool_state.coin_vault_mint,
                    amount_to_ui_amount(amm_coin_vault.base.amount, pool_state.coin_decimals as u8),
                ),
                (
                    pool_state.pc_vault_mint,
                    amount_to_ui_amount(amm_pc_vault.base.amount, pool_state.pc_decimals as u8),
                ),
            )
        };
    
        let price = quote_account.1 / base_account.1;
        
        println!(
            "calculate pool[{}]: {}: {}, {}: {}, price: {} sol",
            amm_pool_id, base_account.0, base_account.1, quote_account.0, quote_account.1, price
        );
    
        Ok((base_account.1, quote_account.1, price))
    }

    async fn get_pool_state(
        rpc_client: Arc<RpcClient>,
        pool_id: Option<&str>,
        mint: Option<&str>,
    ) -> Result<(Pubkey, AmmInfo)> {
        if let Some(pool_id) = pool_id {
            let amm_pool_id = Pubkey::from_str(pool_id)?;
            let account_data = common::rpc::get_account(&rpc_client, &amm_pool_id)?
                .ok_or(anyhow!("NotFoundPool: pool state not found"))?;
            
            // Check if we're dealing with a v4 or v3 pool
            let pool_state = if account_data.len() == 752 {
                // V4 pool
                AmmInfo::load_from_bytes(&account_data)?.to_owned()
            } else if account_data.len() == 637 {
                // V3 pool
                let mut padded_data = vec![0u8; 752];
                padded_data[..account_data.len()].copy_from_slice(&account_data);
                AmmInfo::load_from_bytes(&padded_data)?.to_owned()
            } else {
                return Err(anyhow!(
                    "Unexpected account data size: {}. Expected either 752 (v4) or 637 (v3)",
                    account_data.len()
                ));
            };

            Ok((amm_pool_id, pool_state))
        } else if let Some(mint) = mint {
            Self::get_pool_state_by_mint(rpc_client, mint).await
        } else {
            Err(anyhow!("NotFoundPool: pool state not found"))
        }
    }

    async fn get_pool_state_by_mint(
        rpc_client: Arc<RpcClient>,
        mint: &str,
    ) -> Result<(Pubkey, AmmInfo)> {
        const AMM_INFO_SIZE: usize = 752;
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: Duration = Duration::from_secs(1);
        
        let pairs = vec![
            (Some(spl_token::native_mint::ID), Pubkey::from_str(mint).ok()),
            (Pubkey::from_str(mint).ok(), Some(spl_token::native_mint::ID)),
        ];

        let amm_program = Pubkey::from_str(AMM_PROGRAM).expect("Failed to parse AMM program ID");
        
        for (coin_mint, pc_mint) in pairs {
            let filters = match (coin_mint, pc_mint) {
                (None, None) => Some(vec![RpcFilterType::DataSize(AMM_INFO_SIZE as u64)]),
                (Some(coin_mint), None) => Some(vec![
                    RpcFilterType::Memcmp(Memcmp::new_base58_encoded(400, &coin_mint.to_bytes())),
                    RpcFilterType::DataSize(AMM_INFO_SIZE as u64),
                ]),
                (None, Some(pc_mint)) => Some(vec![
                    RpcFilterType::Memcmp(Memcmp::new_base58_encoded(432, &pc_mint.to_bytes())),
                    RpcFilterType::DataSize(AMM_INFO_SIZE as u64),
                ]),
                (Some(coin_mint), Some(pc_mint)) => Some(vec![
                    RpcFilterType::Memcmp(Memcmp::new_base58_encoded(400, &coin_mint.to_bytes())),
                    RpcFilterType::Memcmp(Memcmp::new_base58_encoded(432, &pc_mint.to_bytes())),
                    RpcFilterType::DataSize(AMM_INFO_SIZE as u64),
                ]),
            };
            
            // Add retry logic for RPC calls
            for retry in 0..MAX_RETRIES {
                match common::rpc::get_program_accounts_with_filters(&rpc_client, amm_program, filters.clone()) {
                    Ok(pools) => {
                        if !pools.is_empty() {
                            let pool = &pools[0];
                            if pool.1.data.len() == AMM_INFO_SIZE {
                                if let Ok(pool_state) = AmmInfo::load_from_bytes(&pool.1.data) {
                                    return Ok((pool.0, pool_state.clone()));
                                }
                            }
                        }
                        break; // Break if successful but no valid pools found
                    }
                    Err(e) => {
                        if retry < MAX_RETRIES - 1 {
                            println!("RPC error (attempt {}/{}): {}. Retrying...", retry + 1, MAX_RETRIES, e);
                            tokio::time::sleep(RETRY_DELAY).await;
                            continue;
                        } else {
                            println!("Failed to get program accounts after {} attempts: {}", MAX_RETRIES, e);
                        }
                    }
                }
            }
        }
        
        Err(anyhow!("NotFoundPool: pool state not found"))
    }
}

#[async_trait]
impl DexProtocol for RaydiumDex {
    fn name(&self) -> &str {
        "Raydium"
    }

    fn clone_box(&self) -> Box<dyn DexProtocol + Send + Sync> {
        Box::new(self.clone())
    }

    async fn get_token_price(&self, rpc_client: Arc<RpcClient>, token_mint: &str) -> Result<Option<f64>> {
        match Self::get_pool_price(rpc_client, None, Some(token_mint)).await {
            Ok((_base, _quote, price)) => Ok(Some(price)),
            Err(_) => Ok(None)
        }
    }
} 