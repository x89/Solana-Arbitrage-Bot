use anyhow::Result;
use solana_sdk::pubkey::Pubkey;

pub struct MeteoraDAmmV2Info {
    pub base_mint: Pubkey,
    pub quote_mint: Pubkey,
    pub base_vault: Pubkey,
    pub quote_vault: Pubkey,
}

impl MeteoraDAmmV2Info {
    pub fn load_checked(data: &[u8]) -> Result<Self> {
        let base_mint = Pubkey::new(&data[168..200]);
        let quote_mint = Pubkey::new(&data[200..232]);
        let base_vault = Pubkey::new(&data[232..264]);
        let quote_vault = Pubkey::new(&data[264..296]);
        Ok(Self {
            base_mint,
            quote_mint,
            base_vault,
            quote_vault,
        })
    }
}
