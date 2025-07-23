use solana_program::pubkey::Pubkey;
use std::str::FromStr;

pub fn solfi_program_id() -> Pubkey {
    Pubkey::from_str("SoLFiHG9TfgtdUXUjWAxi3LtvYuFyDLVhBWxdMZxyCe").unwrap()
}
