use solana_program::pubkey::Pubkey;
use std::str::FromStr;

pub const PUMP_PROGRAM_ID: &str = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA";
pub const PUMP_FEE_WALLET: &str = "JCRGumoE9Qi5BBgULTgdgTLjSgkCMSbF62ZZfGs84JeU";

pub fn pump_program_id() -> Pubkey {
    Pubkey::from_str(PUMP_PROGRAM_ID).unwrap()
}

pub fn pump_fee_wallet() -> Pubkey {
    Pubkey::from_str(PUMP_FEE_WALLET).unwrap()
}
