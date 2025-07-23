use solana_program::pubkey::Pubkey;
use anyhow::Result;

const COIN_VAULT_OFFSET: usize = 336; // coinVault/tokenVaultA
const PC_VAULT_OFFSET: usize = 368; // pcVault/tokenVaultB
const COIN_MINT_OFFSET: usize = 400; // coinMint/tokenMintA
const PC_MINT_OFFSET: usize = 432; // pcMint/tokenMintB

#[derive(Debug)]
pub struct RaydiumAmmInfo {
    pub coin_mint: Pubkey,
    pub pc_mint: Pubkey,
    pub coin_vault: Pubkey,
    pub pc_vault: Pubkey,
}

impl RaydiumAmmInfo {
    pub fn load_checked(data: &[u8]) -> Result<Self> {
        if data.len() < PC_MINT_OFFSET + 32 {
            return Err(anyhow::anyhow!("Invalid data length for RaydiumAmmInfo"));
        }
        
        let coin_vault = Pubkey::new(&data[COIN_VAULT_OFFSET..COIN_VAULT_OFFSET + 32]);
        let pc_vault = Pubkey::new(&data[PC_VAULT_OFFSET..PC_VAULT_OFFSET + 32]);
        let coin_mint = Pubkey::new(&data[COIN_MINT_OFFSET..COIN_MINT_OFFSET + 32]);
        let pc_mint = Pubkey::new(&data[PC_MINT_OFFSET..PC_MINT_OFFSET + 32]);
        
        Ok(Self {
            coin_mint,
            pc_mint,
            coin_vault,
            pc_vault,
        })
    }
}
