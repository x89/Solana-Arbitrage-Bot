use anyhow::Result;
use solana_sdk::pubkey::Pubkey;

pub struct SolfiInfo {
    pub base_mint: Pubkey,
    pub quote_mint: Pubkey,
    pub base_vault: Pubkey,
    pub quote_vault: Pubkey,
}

impl SolfiInfo {
    pub fn load_checked(data: &[u8]) -> Result<Self> {
        let base_mint = Pubkey::new(&data[2664..2696]);
        let quote_mint = Pubkey::new(&data[2696..2728]);
        let base_vault = Pubkey::new(&data[2736..2768]);
        let quote_vault = Pubkey::new(&data[2768..2800]);

        Ok(Self {
            base_mint,
            quote_mint,
            base_vault,
            quote_vault,
        })
    }
}
