use std::str::FromStr;

use anyhow::Result;
use solana_program::pubkey::Pubkey;

#[derive(Debug)]
pub struct PumpAmmInfo {
    pub base_mint: Pubkey,
    pub quote_mint: Pubkey,
    pub pool_base_token_account: Pubkey,
    pub pool_quote_token_account: Pubkey,
    pub coin_creator_vault_authority: Pubkey,
}

impl PumpAmmInfo {
    pub fn load_checked(data: &[u8]) -> Result<Self> {
        let data = &data[8 + 1 + 2 + 32..];

        if data.len() < 4 * 32 + 8 {
            // 4 Pubkeys (32 bytes each) + lp_supply (8 bytes)
            return Err(anyhow::anyhow!("Invalid data length for PumpAmmInfo"));
        }

        let base_mint = Pubkey::from(<[u8; 32]>::try_from(&data[0..32]).unwrap());
        let quote_mint = Pubkey::from(<[u8; 32]>::try_from(&data[32..64]).unwrap());
        let pool_base_token_account = Pubkey::from(<[u8; 32]>::try_from(&data[96..128]).unwrap());
        let pool_quote_token_account = Pubkey::from(<[u8; 32]>::try_from(&data[128..160]).unwrap());

        let pump_program_id =
            Pubkey::from_str("pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA").unwrap();
        println!("data: {:?}", data.len());

        let coin_creator = if data.len() < 257 {
            Pubkey::default()
        } else {
            Pubkey::new(&data[168..200])
        };
        let key = Pubkey::find_program_address(
            &[b"creator_vault", coin_creator.as_ref()],
            &pump_program_id,
        );

        println!("coin_creator: {:?}", key.0);

        Ok(Self {
            base_mint,
            quote_mint,
            pool_base_token_account,
            pool_quote_token_account,
            coin_creator_vault_authority: key.0,
        })
    }
}
