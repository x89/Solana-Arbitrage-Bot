use solana_program::pubkey::Pubkey;
use std::str::FromStr;

pub const WHIRLPOOL_PROGRAM_ID: &str = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc";
pub const MAX_TICK_INDEX: i32 = 443636;
pub const MIN_TICK_INDEX: i32 = -443636;

pub fn whirlpool_program_id() -> Pubkey {
    Pubkey::from_str(WHIRLPOOL_PROGRAM_ID).unwrap()
}
