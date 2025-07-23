use anyhow::Result;
use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::pubkey::Pubkey;

#[derive(Debug, BorshDeserialize, BorshSerialize)]
pub struct VertigoPool {
    pub mint_a: Pubkey,
    pub mint_b: Pubkey,
    pub owner: Pubkey,
    // Add other fields as needed based on actual Vertigo pool structure
}

impl VertigoPool {
    pub fn try_deserialize(data: &mut &[u8]) -> Result<Self> {
        Self::try_from_slice(data)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize VertigoPool: {}", e))
    }
}

#[derive(Debug)]
pub struct VertigoInfo {
    pub mint_a: Pubkey,
    pub mint_b: Pubkey,
    pub pool: Pubkey,
}

impl VertigoInfo {
    pub fn load_checked(data: &[u8], pool: &Pubkey) -> Result<Self> {
        let mut data_slice = &data[..];
        let vertigo_pool = VertigoPool::try_deserialize(&mut data_slice)?;

        Ok(Self {
            mint_a: vertigo_pool.mint_a,
            mint_b: vertigo_pool.mint_b,
            pool: pool.to_owned(),
        })
    }

    pub fn get_token_and_sol_vaults(&self, base_mint: &str, sol_mint: &Pubkey) -> (Pubkey, Pubkey) {
        let token_x_vault = if base_mint == self.mint_a.to_string() {
            derive_vault_address(&self.pool, &self.mint_b).0
        } else {
            derive_vault_address(&self.pool, &self.mint_a).0
        };

        let token_base_vault = if base_mint == self.mint_a.to_string() {
            derive_vault_address(&self.pool, &self.mint_a).0
        } else {
            derive_vault_address(&self.pool, &self.mint_b).0
        };

        (token_x_vault, token_base_vault)
    }
}

/// Helper function to derive vault PDA
pub fn derive_vault_address(pool: &Pubkey, mint: &Pubkey) -> (Pubkey, u8) {
    use crate::dex::vertigo::constants::vertigo_program_id;

    Pubkey::find_program_address(&[pool.as_ref(), mint.as_ref()], &vertigo_program_id())
}
